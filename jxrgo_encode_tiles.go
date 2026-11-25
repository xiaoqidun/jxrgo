// Copyright 2026 肖其顿 (XIAO QI DUN)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package jxrgo

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"math"
)

// jxrHardTileReserve 为局部构造描述及可复用宏块容量保留固定工作空间
const jxrHardTileReserve uint64 = 4096

// jxrHardTileInput 保存局部显示窗口、编码几何及原始样本容量，重叠软分块另指定真实边缘和核心
type jxrHardTileInput struct {
	region  image.Rectangle
	rawWork uint64
	width   uint64
	height  uint64
	left    uint32
	top     uint32
	right   uint32
	bottom  uint32
	columns uint32
	rows    uint32
	window  jxrForwardWindow
}

// RasterRowReader 从上到下单次提供整宽原生行带，y为显示网格起始行，target为零起点临时行带（未写位置为零）
// 完整采样时行带高1、纵向二倍采样时高2；回调内不得保留target引用、并发修改、替换元数据或重入输出
// 回调自身存储和I/O由调用方约束不计入工作预算，返回错误将原样向上传递
type RasterRowReader func(ctx context.Context, y int, target *Raster) error

// jxrRowInput 保存单遍行源的环形样本缓存，描述和行视图不保留完整原图
type jxrRowInput struct {
	info    Info
	read    RasterRowReader
	cache   *Raster
	row     Raster
	view    Raster
	next    int
	height  int
	step    int
	maxWork uint64
}

// RasterTileReader 按全图显示窗口region提供局部原生样本，target为零起点临时网格且每次调用前清零
// 软重叠窗口可能彼此重叠，各次调用须返回同一稳定原图；回调内不得保留引用、修改元数据或重入输出
// 回调自身存储和I/O由调用方约束不计入工作预算，返回错误将原样向上传递
type RasterTileReader func(ctx context.Context, region image.Rectangle, target *Raster) error

// planJXRHardTile 计算独立分块或软重叠邻域的局部几何及有界存储，不读取或复制原始像素
// 入参: info 完整原生信息, source 全局编码头, tile 原分块序号
// 返回: jxrHardTileInput 局部计划, error 布局、采样或容量错误
func planJXRHardTile(info Info, source *jxrStream, tile int) (jxrHardTileInput, error) {
	if source == nil {
		return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile layout", ErrInvalid)
	}
	h := &source.header
	columns, rows := len(h.columns)-1, len(h.rows)-1
	if columns < 1 || rows < 1 || tile < 0 || tile >= columns*rows {
		return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile index", ErrInvalid)
	}
	left, top := int(h.columns[tile%columns]), int(h.rows[tile/columns])
	right, bottom := int(h.columns[tile%columns+1]), int(h.rows[tile/columns+1])
	var window jxrForwardWindow
	if !h.hardTiles && h.overlap != 0 {
		core := image.Rect(left, top, right, bottom)
		left, top = max(0, left-2), max(0, top-2)
		right, bottom = min(int(h.columns[columns]), right+2), min(int(h.rows[rows]), bottom+2)
		window.core = core.Sub(image.Pt(left, top))
		window.edges = [4]bool{left == 0, top == 0, right == int(h.columns[columns]), bottom == int(h.rows[rows])}
	}
	if (source.primary.color == 1 || source.primary.color == 2) && h.overlap == 2 && right-left < 2 {
		return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile dimensions", ErrInvalid)
	}
	alignX, alignY := 1, 1
	if h.color == 1 {
		alignX, alignY = 2, 2
	} else if h.color == 2 {
		alignX, alignY = 2, 1
	}
	cx0 := max(0, left*16-int(h.left))
	cy0 := max(0, top*16-int(h.top))
	cx1 := min(int(h.width), right*16-int(h.left))
	cy1 := min(int(h.height), bottom*16-int(h.top))
	empty := cx0 >= cx1 || cy0 >= cy1
	if cx0 >= cx1 {
		if right*16 <= int(h.left) {
			cx0, cx1 = 0, alignX
		} else {
			cx0, cx1 = int(h.width)-alignX, int(h.width)
		}
	}
	if cy0 >= cy1 {
		if bottom*16 <= int(h.top) {
			cy0, cy1 = 0, alignY
		} else {
			cy0, cy1 = int(h.height)-alignY, int(h.height)
		}
	}
	coded := image.Rect(cx0, cy0, cx1, cy1)
	if coded.Empty() {
		return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile dimensions", ErrInvalid)
	}
	localLeft := cx0 + int(h.left) - left*16
	localTop := cy0 + int(h.top) - top*16
	localRight := (right-left)*16 - localLeft - coded.Dx()
	localBottom := (bottom-top)*16 - localTop - coded.Dy()
	region := jxrRegionPosition(coded, int(h.width), int(h.height), h.transform)
	work := uint64(1024) + uint64(len(info.Components))*384
	if info.ColorEncoding != nil {
		work += 4
	}
	for _, component := range info.Components {
		sx, sy := int(component.XStep), int(component.YStep)
		if sx < 1 || sy < 1 || region.Min.X%sx != 0 || region.Max.X%sx != 0 || region.Min.Y%sy != 0 || region.Max.Y%sy != 0 {
			return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile sampling", ErrInvalid)
		}
		width := uint64(2)
		if component.Precision == 32 {
			width = 4
		}
		count := uint64(region.Dx()/sx) * uint64(region.Dy()/sy)
		if count > (^uint64(0)-work)/width {
			return jxrHardTileInput{}, fmt.Errorf("%w: local hard tile input capacity", ErrLimit)
		}
		work += count * width
	}
	if empty || localLeft < 0 || localLeft > 63 || localTop < 0 || localTop > 63 || localRight < 0 || localRight > 63 || localBottom < 0 || localBottom > 63 {
		window.sample = jxrSampleWindow{enabled: true, width: coded.Dx(), height: coded.Dy(), offsetX: localLeft, offsetY: localTop}
		return jxrHardTileInput{region: region, rawWork: work, width: uint64((right - left) * 16), height: uint64((bottom - top) * 16), columns: uint32(right - left), rows: uint32(bottom - top), window: window}, nil
	}
	return jxrHardTileInput{region: region, rawWork: work, width: uint64(coded.Dx()), height: uint64(coded.Dy()), left: uint32(localLeft), top: uint32(localTop), right: uint32(localRight), bottom: uint32(localBottom), columns: uint32(right - left), rows: uint32(bottom - top), window: window}, nil
}

// copyJXRHardTile 在预留容量中复制局部原始分量，可复用内部样本存储，不复制ICC或保留其他分块样本引用
// 入参: ctx Context, raster 完整原始图像, plan 局部计划, maximum 局部原图及描述容量, local 可复用独占图像，nil时创建
// 返回: *Raster 有界局部原生图像, error 采样、资源或取消错误
func copyJXRHardTile(ctx context.Context, raster *Raster, plan jxrHardTileInput, maximum uint64, local *Raster) (*Raster, error) {
	local, err := resizeJXRHardTile(ctx, raster.info, plan, maximum, local)
	if err != nil {
		return nil, err
	}
	for index, component := range local.info.Components {
		original, err := raster.component(index)
		if err != nil {
			return nil, err
		}
		for y := range component.Bounds.Dy() {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for x := range component.Bounds.Dx() {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				bits, err := original.sampleBitsAt(plan.region.Min.X/int(component.XStep)+x, plan.region.Min.Y/int(component.YStep)+y)
				if err != nil {
					return nil, err
				}
				local.planes[index].storeBits(y*component.Bounds.Dx()+x, bits)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return local, nil
}

// resizeJXRHardTile 共用局部原始分量的有界构造与容量复用，不读取任何原图样本
// 入参: ctx Context, input 完整图像元数据, plan 局部计划, maximum 局部样本和描述容量, local 可复用独占图像
// 返回: *Raster 局部图像，已复用的样本尚未清空或填充, error 布局、容量或取消错误
func resizeJXRHardTile(ctx context.Context, input Info, plan jxrHardTileInput, maximum uint64, local *Raster) (*Raster, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if plan.rawWork > maximum {
		return nil, &LimitError{Resource: "local hard tile raw memory", Limit: maximum, Required: plan.rawWork}
	}
	input.ICCProfile, input.EXIF, input.GPS, input.XMP, input.IPTC, input.Description = nil, nil, nil, nil, nil, nil
	if local == nil {
		info, _, err := jxrRegionInfo(input, plan.region)
		if err != nil {
			return nil, err
		}
		metadata := uint64(512) + uint64(len(info.Components))*128
		local, err = newRasterContext(ctx, info, Limits{MaxPixels: uint64(info.Bounds.Dx()) * uint64(info.Bounds.Dy()), MaxMemoryBytes: maximum - metadata})
		if err != nil {
			return nil, err
		}
	} else {
		work, err := jxrRasterWork(local, maximum)
		if err != nil {
			return nil, err
		}
		metadata := uint64(512) + uint64(len(input.Components))*128
		if metadata > maximum-work || len(local.planes) != len(input.Components) || len(local.info.Components) != len(input.Components) {
			return nil, &LimitError{Resource: "retained hard tile raw capacity", Limit: maximum, Required: jxrSaturatedAdd(work, metadata)}
		}
		input.Bounds = image.Rect(0, 0, plan.region.Dx(), plan.region.Dy())
		originals := input.Components
		input.Components = local.info.Components
		for index, original := range originals {
			component := original
			component.Bounds = image.Rect(0, 0, input.Bounds.Dx()/int(component.XStep), input.Bounds.Dy()/int(component.YStep))
			count := component.Bounds.Dx() * component.Bounds.Dy()
			plane := &local.planes[index]
			if component.Precision == 32 {
				if count > cap(plane.words) {
					return nil, &LimitError{Resource: "reused hard tile word capacity", Limit: uint64(cap(plane.words)), Required: uint64(count)}
				}
				plane.words = plane.words[:count]
			} else {
				if count > cap(plane.values) {
					return nil, &LimitError{Resource: "reused hard tile sample capacity", Limit: uint64(cap(plane.values)), Required: uint64(count)}
				}
				plane.values = plane.values[:count]
			}
			plane.bounds, input.Components[index] = component.Bounds, component
		}
		local.info = input
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return local, nil
}

// encodeJXRHardTileSourceTo 共用完整原图与逐块提供器的局部变换，预检最大容量后每次只装载当前分块
// 入参: ctx Context, w 可定位输出, info 完整原生元数据, source 完整编码头, options 除源元数据外的工作上限, load 当前分块装载器
// 返回: error 预检、样本提供、变换、编码、取消或I/O错误
func encodeJXRHardTileSourceTo(ctx context.Context, w io.WriteSeeker, info Info, source *jxrStream, options jxrStreamOutput, load func(jxrHardTileInput, uint64, *Raster) (*Raster, error)) error {
	settingsWork, err := options.tileWork(ctx, options.maxMemory)
	if err != nil {
		return err
	}
	tiles := (len(source.header.columns) - 1) * (len(source.header.rows) - 1)
	var largest jxrHardTileInput
	columns, rows := uint32(0), uint32(0)
	for tile := range tiles {
		if err := ctx.Err(); err != nil {
			return err
		}
		plan, err := planJXRHardTile(info, source, tile)
		if err != nil {
			return err
		}
		if plan.rawWork > largest.rawWork {
			largest = plan
		}
		columns = max(columns, plan.columns)
		rows = max(rows, plan.rows)
	}
	template := *source
	template.header.width, template.header.height = uint64(columns)*16, uint64(rows)*16
	template.header.left, template.header.top, template.header.right, template.header.bottom = 0, 0, 0, 0
	template.header.columns, template.header.rows = []uint32{0, columns}, []uint32{0, rows}
	bands := 1
	if source.header.frequency {
		bands = 4 - int(source.primary.bands)
	}
	var templateOffsets [4]uint64
	template.header.index, template.packets = source.header.frequency, templateOffsets[:bands]
	estimate := jxrCoefficientImage{primary: jxrCoefficients{plane: template.primary}}
	if template.alpha != nil {
		estimate.alpha = &jxrCoefficients{plane: *template.alpha}
	}
	maximum := estimate.outputWork(&template)
	for _, extra := range []uint64{largest.rawWork, settingsWork, jxrHardTileReserve} {
		if maximum > options.maxMemory || extra > options.maxMemory-maximum {
			return &LimitError{Resource: "local hard tile working memory", Limit: options.maxMemory, Required: jxrSaturatedAdd(maximum, extra)}
		}
		maximum += extra
	}
	retained := source.headerWork() + jxrHardTileReserve
	if retained > options.maxMemory || maximum > options.maxMemory-retained {
		return &LimitError{Resource: "retained hard tile header memory", Limit: options.maxMemory, Required: jxrSaturatedAdd(retained, maximum)}
	}
	var local *Raster
	var planes [2]*jxrCoefficients
	frame := *source
	var horizontal, vertical [4]uint32
	var offset [36]uint64
	frame.header.columns, frame.header.rows, frame.packets = horizontal[:2:2], vertical[:2:2], offset[:bands:bands]
	frame.header.index = source.header.frequency
	return writeJXRPacketsTo(ctx, w, source, options, retained+maximum, func(tile int, output *jxrTileOutput) error {
		plan, err := planJXRHardTile(info, source, tile)
		if err != nil {
			return err
		}
		available := maximum - largest.rawWork
		if local == nil {
			local, err = resizeJXRHardTile(ctx, info, largest, largest.rawWork, nil)
			if err != nil {
				return err
			}
			allocated := uint64(0)
			for index, plane := range [2]*jxrPlane{&template.primary, template.alpha} {
				if plane != nil {
					planes[index], err = newJXRTransformPlane(ctx, &template.header, plane, available-allocated)
					if err != nil {
						return err
					}
					allocated += uint64(cap(planes[index].low)+cap(planes[index].high))*8 + uint64(cap(planes[index].info))*6
				}
			}
		}
		local, err = load(plan, largest.rawWork, local)
		if err != nil {
			return err
		}
		window, localTile := jxrBuildTileFrame(&frame, source.header.frequency, bands, &plan, &horizontal, &vertical, &offset)
		selected := options
		if len(options.tiles) != 1 {
			selected.tiles = options.tiles[tile : tile+1 : tile+1]
		}
		state, err := forwardJXRNativeWindowStorage(ctx, local, &frame, &selected, available, &planes, window)
		if err != nil {
			return err
		}
		working, ok := jxrWorkTotal(options.maxMemory, state.outputWork(&frame), jxrHardTileReserve, output.packet[0].budget.limit)
		if !ok {
			required := jxrSaturatedAdd(state.outputWork(&frame), jxrSaturatedAdd(jxrHardTileReserve, output.packet[0].budget.limit))
			return &LimitError{Resource: "local tile output memory", Limit: options.maxMemory, Required: required}
		}
		output.maxMemory = working
		_, err = state.writeTile(ctx, &frame, localTile, output)
		return err
	})
}

// jxrBuildTileFrame 根据单块窗口计划装配局部编码流头部、核心分块序号与软重叠窗口指针
// 入参: frame 目标局部码流头, sourceFrequency 源码流是否频率布局, bands 频带数, plan 分块窗口计划, horizontal,vertical 分块边界栈存储, offset 包偏移栈存储
// 返回: *jxrForwardWindow 可选局部窗口指针, int 局部核心分块序号
func jxrBuildTileFrame(frame *jxrStream, sourceFrequency bool, bands int, plan *jxrHardTileInput, horizontal, vertical *[4]uint32, offset *[36]uint64) (*jxrForwardWindow, int) {
	frame.header.width, frame.header.height = plan.width, plan.height
	frame.header.left, frame.header.top, frame.header.right, frame.header.bottom = plan.left, plan.top, plan.right, plan.bottom
	var window *jxrForwardWindow
	localTile := 0
	if plan.window.core.Empty() {
		horizontal[0], vertical[0] = 0, 0
		horizontal[1], vertical[1] = plan.columns, plan.rows
		frame.header.columns, frame.header.rows = horizontal[:2:2], vertical[:2:2]
		if plan.window.sample.enabled {
			window = &plan.window
		}
	} else {
		var x, y int
		window = &plan.window
		frame.header.columns, x = jxrForwardWindowEdges(horizontal, plan.columns, window.core.Min.X, window.core.Max.X)
		frame.header.rows, y = jxrForwardWindowEdges(vertical, plan.rows, window.core.Min.Y, window.core.Max.Y)
		localTile = y*(len(frame.header.columns)-1) + x
	}
	count := (len(frame.header.columns) - 1) * (len(frame.header.rows) - 1) * bands
	frame.packets = offset[:count:count]
	frame.header.index = sourceFrequency || len(frame.packets) > bands
	return window, localTile
}

// jxrForwardWindowEdges 将局部核心边界组织为合法分块边界，非核心分块仅用于寻址而不写出
// 入参: storage 四项独占存储, extent 局部宏块总长, left,right 核心宏块边界
// 返回: []uint32 局部分块边界, int 核心分块序号
func jxrForwardWindowEdges(storage *[4]uint32, extent uint32, left, right int) ([]uint32, int) {
	storage[0] = 0
	count := 1
	if left != 0 {
		storage[count] = uint32(left)
		count++
	}
	index := count - 1
	storage[count] = uint32(right)
	count++
	if uint32(right) != extent {
		storage[count] = extent
		count++
	}
	return storage[:count:count], index
}

// EncodeRows 从上到下单遍读取原生行带并按Format写入JXR文件或独立码流，零Format默认输出FormatJXR
// w实现io.WriteSeeker时直接原地回填且须位于目标起点；普通顺序io.Writer优先借用EncodeOptions.Scratch或有界内存暂存回填后流式拷贝，不关闭或截断输出
// 入参: ctx Context，nil采用Background, w 目标输出流, info 原生描述, read 顺序行源, opts 编码选项
// 返回: error 参数、能力、资源、源、取消、定位或写入错误
func EncodeRows(ctx context.Context, w io.Writer, info Info, read RasterRowReader, opts *EncodeOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil || read == nil {
		return &FormatError{Op: "EncodeRows", Detail: "nil row output or reader"}
	}
	file := opts == nil || opts.Format != FormatCodestream
	wrapped := func(ctx context.Context, y int, target *Raster) error {
		if err := read(ctx, y, target); err != nil {
			return jxrCallbackError{err: err}
		}
		return nil
	}
	if ws, ok := w.(io.WriteSeeker); ok {
		return wrapJXRError("EncodeRows", encodeJXRRows(ctx, ws, info, wrapped, opts, file))
	}
	return wrapJXRError("EncodeRows", encodeJXRRowsSequential(ctx, w, info, wrapped, opts, file))
}

// jxrDefaultRowTileSize 按编码网格尺寸与方向动态计算十六对齐默认分块尺寸，确保行列分块数均不超过4096且转置方向保持最小环形行缓冲
// 入参: input 已规范化原生描述, format 像素格式, settings 已校验编码选项
// 返回: image.Point 十六对齐的默认编码分块尺寸
func jxrDefaultRowTileSize(input Info, format jxrPixelFormat, settings EncodeOptions) image.Point {
	transform := jxrInputTransform(input, format.color, uint8(settings.Orientation))
	codedW, codedH := uint64(input.Bounds.Dx()), uint64(input.Bounds.Dy())
	if transform&4 != 0 {
		codedW, codedH = codedH, codedW
	}
	var mbW, mbH uint64
	if settings.Window != nil {
		mbW = (codedW + uint64(settings.Window.Left) + uint64(settings.Window.Right) + 15) / 16
		mbH = (codedH + uint64(settings.Window.Top) + uint64(settings.Window.Bottom) + 15) / 16
	} else {
		mbW = (codedW + 15) / 16
		mbH = (codedH + 15) / 16
	}
	prefMBX, prefMBY := uint64(16), uint64(1)
	if transform&4 != 0 {
		prefMBX, prefMBY = 1, 16
	}
	internalColor := format.color
	if settings.Subsampling == Subsampling420 {
		internalColor = 1
	} else if settings.Subsampling == Subsampling422 {
		internalColor = 2
	}
	if (internalColor == 1 || internalColor == 2) && settings.Overlap == OverlapTwo && mbW >= 2 && prefMBX < 2 {
		prefMBX = 2
	}
	tileMBX := max(prefMBX, (mbW+4095)/4096)
	tileMBY := max(prefMBY, (mbH+4095)/4096)
	tileMBX = min(tileMBX, 65535)
	tileMBY = min(tileMBY, 65535)
	return image.Pt(int(tileMBX*16), int(tileMBY*16))
}

// encodeJXRRows 预检顺序行约束并从逐块编码总额度中预留环形样本及全部行描述容量
// 入参: ctx 已规范化Context, w 可定位输出, info 源描述, read 顺序行源, opts 编码选项, file 是否写JXR容器文件
// 返回: error 参数、能力、资源、源或I/O错误
func encodeJXRRows(ctx context.Context, w io.WriteSeeker, info Info, read RasterRowReader, opts *EncodeOptions, file bool) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if settings.Uncompressed {
		return encodeJXRRowsUncompressed(ctx, w, info, read, settings)
	}
	settings = jxrResolveEncodeOptions(settings)
	format := FormatCodestream
	if file {
		if info.ImageType&ImageTypePreview != 0 {
			return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
		}
		format = FormatJXR
	}
	settings.Format = format
	limits := settings.Limits.normalized()
	count := len(info.Components)
	metadata := uint64(4096) + uint64(count)*512
	if count < 1 || count > rasterMaxComponents+1 || info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || uint64(info.Bounds.Max.X) > 1<<32 || uint64(info.Bounds.Max.Y) > 1<<32 {
		return fmt.Errorf("%w: row source geometry", ErrInvalid)
	}
	if metadata >= limits.MaxMemoryBytes {
		return &LimitError{Resource: "row source descriptors", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(metadata, 1)}
	}
	if uint64(info.Bounds.Dx()) > limits.MaxPixels/uint64(info.Bounds.Dy()) {
		return &LimitError{Resource: "row source pixels", Limit: limits.MaxPixels, Required: jxrSaturatedMul(uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy()))}
	}
	input := info
	input.ICCProfile, input.EXIF, input.GPS, input.XMP, input.IPTC, input.Description = nil, nil, nil, nil, nil, nil
	input.Components = cloneComponentInfo(info.Components)
	input.ColorEncoding = cloneColorEncoding(info.ColorEncoding)
	if err := normalizeRasterInfo(&input); err != nil {
		return err
	}
	if err := checkJXRModelComponents(&jxrPlane{channels: input.Channels}); err != nil {
		return err
	}
	formatInfo, err := rasterPixelFormat(input)
	if err != nil {
		return err
	}
	if settings.TileSize == (image.Point{}) {
		settings.TileSize = jxrDefaultRowTileSize(input, formatInfo, settings)
	}
	var alpha jxrAlphaRowStorage
	if settings.SeparateAlpha {
		var held uint64
		alpha, held, err = makeJXRAlphaRowStorage(input, settings.Scratch, limits.MaxMemoryBytes-metadata)
		if err != nil {
			return err
		}
		limits.MaxMemoryBytes -= held
	}
	height := jxrRingHeight(input, formatInfo.color, settings)
	cacheInfo, work, err := jxrRegionInfo(input, image.Rect(0, 0, input.Bounds.Dx(), height))
	if err != nil {
		return err
	}
	if work >= limits.MaxMemoryBytes-metadata {
		return &LimitError{Resource: "row ring storage", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(metadata, jxrSaturatedAdd(work, 1))}
	}
	limits.MaxMemoryBytes -= work + metadata
	settings.Limits = limits
	step := 1
	for _, component := range input.Components {
		step = max(step, int(component.YStep))
	}
	source := jxrRowInput{info: cacheInfo, read: read, height: input.Bounds.Dy(), step: step, maxWork: work}
	if file {
		if settings.SeparateAlpha {
			return encodeJXRSeparateRows(ctx, w, info, &source, &alpha, settings)
		}
		return encodeJXRFileTileInput(ctx, w, info, source.load, &settings, true)
	}
	return encodeJXRCodestreamTileInput(ctx, w, info, source.load, &settings, true)
}

// load 将逐块需要的显示窗口从行环形缓存投影到目标，源回调只按递增行号读取一次
// 入参: ctx Context, region 已校验的完整图像显示窗口, target 当前局部原生图像
// 返回: error 资源、顺序、源或取消错误
func (source *jxrRowInput) load(ctx context.Context, region image.Rectangle, target *Raster) error {
	if source.cache == nil {
		cache, err := newRasterContext(ctx, source.info, Limits{MaxPixels: uint64(source.info.Bounds.Dx()) * uint64(source.info.Bounds.Dy()), MaxMemoryBytes: source.maxWork})
		if err != nil {
			return err
		}
		source.cache = cache
		source.row.info = source.info
		source.row.info.Components = cloneComponentInfo(source.info.Components)
		source.row.planes = make([]imageComponents, len(source.info.Components))
	}
	capacity := source.cache.Bounds().Dy()
	if region.Min.Y < max(0, source.next-capacity) || region.Max.Y > source.height || region.Dy() > capacity {
		return fmt.Errorf("%w: row cache window order", ErrInvalid)
	}
	for source.next < region.Max.Y {
		if err := source.readNext(ctx); err != nil {
			return err
		}
	}
	if region.Min.Y < max(0, source.next-capacity) {
		return fmt.Errorf("%w: row cache overwritten window", ErrInvalid)
	}
	return jxrProjectRingRegion(ctx, source.cache, target, region)
}

// jxrRingHeight 计算单遍流式行编码所需的有界环形行缓存高度，转置方向取TileSize.X并按软重叠追加邻域行
// 入参: info 已规范化原生描述, color 输出像素颜色编码, settings 已解析编码选项
// 返回: int 环形缓存高度
func jxrRingHeight(info Info, color uint8, settings EncodeOptions) int {
	tileHeight := settings.TileSize.Y
	if jxrInputTransform(info, color, uint8(settings.Orientation))&4 != 0 {
		tileHeight = settings.TileSize.X
	}
	dispH := info.Bounds.Dy()
	height := min(dispH, tileHeight)
	if !settings.HardTiles && settings.Overlap != OverlapNone {
		height += min(64, dispH-height)
	}
	return height
}

// jxrBindRingRowView 将环形缓存指定行槽位绑定为零起点可写临时行视图并清零，不分配堆存储
// 入参: ring 环形行缓存, rowStorage 可复用行视图骨架, view 返回用浅拷贝视图槽位, y 当前显示起始行, height 本步行高
// 返回: *Raster 指向环形缓存当前槽位的零起点临时视图
func jxrBindRingRowView(ring, rowStorage, view *Raster, y, height int) *Raster {
	rowStorage.info.Bounds = image.Rect(0, 0, ring.info.Bounds.Dx(), height)
	for index := range rowStorage.planes {
		component := ring.info.Components[index]
		component.Bounds.Max.Y = height / int(component.YStep)
		rowStorage.info.Components[index] = component
		plane := ring.planes[index]
		from := (y / int(component.YStep) % plane.bounds.Dy()) * plane.bounds.Dx()
		count := component.Bounds.Dx() * component.Bounds.Dy()
		plane.bounds = component.Bounds
		if plane.words != nil {
			plane.words = plane.words[from : from+count : from+count]
			clear(plane.words)
		} else {
			plane.values = plane.values[from : from+count : from+count]
			clear(plane.values)
		}
		rowStorage.planes[index] = plane
	}
	*view = *rowStorage
	return view
}

// jxrProjectRingRegion 将环形行缓存中指定显示窗口的全部原生分量拷贝至目标局部图像
// 入参: ctx 已规范化Context, ring 环形行缓存, target 目标局部图像, region 已校验显示窗口
// 返回: error 取消错误
func jxrProjectRingRegion(ctx context.Context, ring, target *Raster, region image.Rectangle) error {
	for index, comp := range target.info.Components {
		dst := &target.planes[index]
		src := &ring.planes[index]
		sx, sy := int(comp.XStep), int(comp.YStep)
		x0, y0 := region.Min.X/sx, region.Min.Y/sy
		for y := range comp.Bounds.Dy() {
			if err := ctx.Err(); err != nil {
				return err
			}
			from := ((y0+y)%src.bounds.Dy())*src.bounds.Dx() + x0
			to := y * comp.Bounds.Dx()
			for x := range comp.Bounds.Dx() {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				dst.storeBits(to+x, src.sampleBits(from+x))
			}
		}
	}
	return ctx.Err()
}

// readNext 将下一条原生行带直接填入环形存储，清零行视图且不为每行分配样本
// 入参: ctx Context
// 返回: error 源、顺序或取消错误
func (source *jxrRowInput) readNext(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	height := min(source.step, source.height-source.next)
	view := jxrBindRingRowView(source.cache, &source.row, &source.view, source.next, height)
	if err := source.read(ctx, source.next, view); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	source.next += height
	return nil
}

// EncodeTiles 从有界逐块样本源按Format写入JXR文件或独立码流，自动适配可定位输出与顺序输出
// opts须给出正的十六倍数TileSize；w实现io.WriteSeeker时走单遍回填路径，否则走两遍顺序路径
// 入参: ctx Context，nil采用Background, w 输出流, info 完整原生图像描述, read 逐块样本提供器, opts 编码选项
// 返回: error 元数据、布局、能力、精度、资源、提供器、取消或写入错误
func EncodeTiles(ctx context.Context, w io.Writer, info Info, read RasterTileReader, opts *EncodeOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil || read == nil {
		return &FormatError{Op: "EncodeTiles", Detail: "nil tiled output or reader"}
	}
	codestream := opts != nil && opts.Format == FormatCodestream
	wrapped := func(ctx context.Context, region image.Rectangle, target *Raster) error {
		if err := read(ctx, region, target); err != nil {
			return jxrCallbackError{err: err}
		}
		return nil
	}
	var err error
	if ws, ok := w.(io.WriteSeeker); ok {
		if codestream {
			err = encodeJXRCodestreamTileInput(ctx, ws, info, wrapped, opts, false)
		} else {
			err = encodeJXRFileTileInput(ctx, ws, info, wrapped, opts, false)
		}
	} else if codestream {
		err = encodeJXRCodestreamTilesSequential(ctx, w, info, wrapped, opts)
	} else {
		err = encodeJXRFileTilesSequential(ctx, w, info, wrapped, opts)
	}
	return wrapJXRError("EncodeTiles", err)
}

// encodeJXRFileTileInput 共用文件窗口及顺序行源入口，保持普通窗口源原有包顺序及独立透明度规则
// 入参: ctx 已规范化Context, w 可定位输出, info 原生描述, read 样本提供器, opts 编码选项, rowOrder 是否按显示行调度
// 返回: error 预检、样本、编码、资源或I/O错误
func encodeJXRFileTileInput(ctx context.Context, w io.WriteSeeker, info Info, read RasterTileReader, opts *EncodeOptions, rowOrder bool) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if settings.TileSize.X == 0 || settings.TileSize.Y == 0 {
		return fmt.Errorf("%w: file tile input requires explicit Annex A tiles", ErrUnsupported)
	}
	if info.ImageType&ImageTypePreview != 0 {
		return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	settings.Format = FormatJXR
	info, retained, err := prepareJXRTileInfo(ctx, info, settings)
	if err != nil {
		return err
	}
	if settings.SeparateAlpha {
		return encodeJXRSeparateTileFileTo(ctx, w, info, settings, retained, read)
	}
	metadata := Raster{info: info}
	source, output, err := makeJXREncodingPlanRetained(ctx, &metadata, &settings, retained)
	if err != nil {
		return jxrPlanLimitError(err, output.stream.maxMemory, retained)
	}
	output.stream.rowOrder = rowOrder
	return writeJXRFileTo(ctx, w, info, &source, output.stream, retained, func(options jxrStreamOutput) error {
		options.maxMemory -= retained
		return encodeJXRProvidedTilesTo(ctx, w, info, &source, options, read)
	})
}

// encodeJXRCodestreamTileInput 共用随机窗口及顺序行源入口，行源仅改变有索引包的物理提交顺序
// 入参: ctx 已规范化Context, w 可定位输出, info 原生描述, read 样本提供器, opts 编码选项, rowOrder 是否按显示行调度
// 返回: error 预检、样本、编码、资源或I/O错误
func encodeJXRCodestreamTileInput(ctx context.Context, w io.WriteSeeker, info Info, read RasterTileReader, opts *EncodeOptions, rowOrder bool) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if settings.Format != FormatUnknown && settings.Format != FormatCodestream || settings.SeparateAlpha || settings.TileSize.X == 0 || settings.TileSize.Y == 0 {
		return fmt.Errorf("%w: tile input requires explicit tiles in a bare codestream", ErrUnsupported)
	}
	settings.Format = FormatCodestream
	info, retained, err := prepareJXRTileInfo(ctx, info, settings)
	if err != nil {
		return err
	}
	metadata := Raster{info: info}
	source, output, err := makeJXREncodingPlanRetained(ctx, &metadata, &settings, retained)
	if err != nil {
		return jxrPlanLimitError(err, output.stream.maxMemory, retained)
	}
	output.stream.maxMemory -= retained
	output.stream.rowOrder = rowOrder
	return encodeJXRProvidedTilesTo(ctx, w, info, &source, output.stream, read)
}

// prepareJXRTileInfo 预检并复制逐块源描述，文件路径保留全部容器元数据而独立码流明确丢弃文件元数据
// 入参: ctx Context, info 原始描述, settings 已校验的逐块编码选项
// 返回: Info 独占描述快照, uint64 常驻元数据额度, error 采样、资源、能力或取消错误
func prepareJXRTileInfo(ctx context.Context, info Info, settings EncodeOptions) (Info, uint64, error) {
	limits := settings.Limits.normalized()
	count := len(info.Components)
	if count < 1 || count > rasterMaxComponents+1 || info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || uint64(info.Bounds.Max.X) > 1<<32 || uint64(info.Bounds.Max.Y) > 1<<32 {
		return Info{}, 0, fmt.Errorf("%w: tiled source metadata", ErrInvalid)
	}
	if uint64(info.Bounds.Dx()) > limits.MaxPixels/uint64(info.Bounds.Dy()) {
		return Info{}, 0, &LimitError{Resource: "tiled source pixels", Limit: limits.MaxPixels, Required: jxrSaturatedMul(uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy()))}
	}
	retained := uint64(512) + uint64(count)*128
	if settings.Format == FormatJXR {
		info = jxrWithContainerMetadata(info, &settings)
		for _, blob := range [][]byte{info.ICCProfile, info.EXIF, info.GPS, info.XMP, info.IPTC} {
			if uint64(len(blob)) > math.MaxUint32 {
				return Info{}, 0, &LimitError{Resource: "tiled file metadata field capacity", Limit: math.MaxUint32, Required: uint64(len(blob))}
			}
			amount := uint64(cap(blob)) + uint64(len(blob))
			if retained > limits.MaxMemoryBytes || amount > limits.MaxMemoryBytes-retained {
				return Info{}, 0, &LimitError{Resource: "tiled file retained metadata memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, amount)}
			}
			retained += amount
		}
		if descWork := jxrDescriptionWork(info.Description); descWork != 0 {
			amount := descWork * 2
			if retained > limits.MaxMemoryBytes || amount > limits.MaxMemoryBytes-retained {
				return Info{}, 0, &LimitError{Resource: "tiled file retained description memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, amount)}
			}
			retained += amount
		}
	}
	if info.ColorEncoding != nil {
		retained += 4
	}
	if retained > limits.MaxMemoryBytes || jxrInputHeaderReserve > limits.MaxMemoryBytes-retained {
		return Info{}, 0, &LimitError{Resource: "tiled source header memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, jxrInputHeaderReserve)}
	}
	info.Components = cloneComponentInfo(info.Components)
	info.ColorEncoding = cloneColorEncoding(info.ColorEncoding)
	if err := normalizeRasterInfo(&info); err != nil {
		return Info{}, 0, err
	}
	if err := checkJXRModelComponents(&jxrPlane{channels: info.Channels}); err != nil {
		return Info{}, 0, err
	}
	if !settings.Uncompressed {
		if info.Components[0].Format == SampleFloat && info.Components[0].Precision == 32 && settings.Float == nil {
			return Info{}, 0, fmt.Errorf("%w: tile input requires explicit float parameters", ErrUnsupported)
		}
		if info.Alpha && settings.Alpha != nil && info.Components[info.Channels].Format == SampleFloat && info.Components[info.Channels].Precision == 32 && settings.Alpha.Float == nil {
			return Info{}, 0, fmt.Errorf("%w: tile input requires explicit alpha float parameters", ErrUnsupported)
		}
	}
	if settings.Format == FormatJXR {
		for _, target := range []*[]byte{&info.ICCProfile, &info.EXIF, &info.GPS, &info.XMP, &info.IPTC} {
			if len(*target) == 0 {
				*target = nil
				continue
			}
			if err := ctx.Err(); err != nil {
				return Info{}, 0, err
			}
			original := *target
			copied := make([]byte, len(original))
			for at := 0; at < len(original); at += 16384 {
				if err := ctx.Err(); err != nil {
					return Info{}, 0, err
				}
				copy(copied[at:min(at+16384, len(original))], original[at:min(at+16384, len(original))])
			}
			*target = copied
		}
		info.Description = cloneImageDescription(info.Description)
	} else {
		info.ICCProfile, info.EXIF, info.GPS, info.XMP, info.IPTC, info.Description = nil, nil, nil, nil, nil, nil
	}
	return info, retained, nil
}

// encodeJXRProvidedTilesTo 共用文件与独立码流的逐块样本装载，局部缓冲每次清零且提供器只看到临时视图
// 入参: ctx Context, w 可定位输出, info 独占元数据, source 编码布局, options 不含常驻元数据的额度, read 样本提供器
// 返回: error 提供器、精度、资源、取消或I/O错误
func encodeJXRProvidedTilesTo(ctx context.Context, w io.WriteSeeker, info Info, source *jxrStream, options jxrStreamOutput, read RasterTileReader) error {
	return encodeJXRHardTileSourceTo(ctx, w, info, source, options, func(plan jxrHardTileInput, maximum uint64, local *Raster) (*Raster, error) {
		local, err := resizeJXRHardTile(ctx, info, plan, maximum, local)
		if err != nil {
			return nil, err
		}
		for _, plane := range local.planes {
			clear(plane.values)
			clear(plane.words)
		}
		view := *local
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := read(ctx, plan.region, &view); err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return local, nil
	})
}

// encodeJXRRowsUncompressed 将顺序行带按0xBC03无压缩容器格式写出，无转置方向使用行级流式缓冲，转置方向在预算内打包
// 入参: ctx 已规范化Context, w 起点为零的可定位输出, info 原始图像描述, read 顺序行带读取回调, settings 已校验无压缩选项
// 返回: error 格式、样本、资源、源、取消或I/O错误
func encodeJXRRowsUncompressed(ctx context.Context, w io.WriteSeeker, info Info, read RasterRowReader, settings EncodeOptions) error {
	if info.ImageType&ImageTypePreview != 0 {
		return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	_, _, err := encodeJXRRowsUncompressedAt(ctx, w, info, read, settings, 8)
	return err
}

// encodeJXRRowsUncompressedAt 在指定目录位置将顺序行带按0xBC03无压缩容器格式写出，支持单图与多图容器
// 入参: ctx 已规范化Context, w 当前输出, info 原始图像描述, read 顺序行带读取回调, settings 已校验无压缩选项, start 目录起点
// 返回: uint64 图像结束位置, uint64 当前目录链接位置, error 格式、样本、资源、源、取消或I/O错误
func encodeJXRRowsUncompressedAt(ctx context.Context, w io.WriteSeeker, info Info, read RasterRowReader, settings EncodeOptions, start uint64) (uint64, uint64, error) {
	if start < 8 || start&1 != 0 || start > math.MaxUint32 {
		return 0, 0, fmt.Errorf("%w: uncompressed row directory position", ErrInvalid)
	}
	settings.Format = FormatJXR
	info, retained, err := prepareJXRTileInfo(ctx, info, settings)
	if err != nil {
		return 0, 0, err
	}
	if settings.SeparateAlpha && !info.Alpha {
		return 0, 0, fmt.Errorf("%w: separate alpha without source alpha", ErrInvalid)
	}
	limits := settings.Limits.normalized()
	maxBytes := settings.MaxOutputBytes
	if maxBytes == 0 {
		maxBytes = 64 << 20
	}
	delta := start - 8
	maxBytes = min(maxBytes, uint64(math.MaxUint32)-delta)
	identifier, format, err := jxrRasterIdentifier(info, [16]byte{})
	if err != nil {
		return 0, 0, err
	}
	transform := jxrInputTransform(info, format.color, uint8(settings.Orientation))
	codedW, codedH := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
	if transform&4 != 0 {
		codedW, codedH = codedH, codedW
	}
	layout, err := jxrMakeUncompressedLayout(identifier[15], format, codedW, codedH, settings.SeparateAlpha)
	if err != nil {
		return 0, 0, err
	}
	if format.name == "BlackWhite" && format.depth == 15 {
		format.depth = 0
	}
	c := jxrContainer{width: codedW, height: codedH, format: format, pixelFormat: identifier, transform: uint32(transform), uncompressed: true}
	c.setInfoMetadata(info)
	exifNode, gpsNode, err := checkJXRContainerMetadataLayout(ctx, &c, limits.MaxPixels, limits.MaxMemoryBytes-retained, settings.SeparateAlpha)
	if err != nil {
		return 0, 0, err
	}
	b := newJXRBitWriter(ctx, maxBytes, limits.MaxMemoryBytes-retained)
	if b.err != nil {
		return 0, 0, b.err
	}
	var offsets jxrContainerOffsets
	writeJXRContainerBytes(&b, &c, true, settings.SeparateAlpha, &offsets, exifNode, gpsNode)
	if b.err != nil {
		return 0, 0, b.err
	}
	header := b.data
	mainLocal := uint64(len(header))
	if layout.mainBytes > maxBytes-mainLocal {
		return 0, 0, &LimitError{Resource: "max output bytes", Limit: maxBytes, Required: jxrSaturatedAdd(mainLocal, layout.mainBytes), Detail: "uncompressed row output bytes"}
	}
	totalLocal := mainLocal + layout.mainBytes
	alphaLocal, padMain := uint64(0), false
	if settings.SeparateAlpha {
		if totalLocal&1 != 0 {
			if totalLocal >= maxBytes {
				return 0, 0, &LimitError{Resource: "max output bytes", Limit: maxBytes, Required: jxrSaturatedAdd(totalLocal, 1), Detail: "uncompressed row alpha alignment"}
			}
			totalLocal++
			padMain = true
		}
		alphaLocal = totalLocal
		if layout.alphaBytes > maxBytes-totalLocal {
			return 0, 0, &LimitError{Resource: "max output bytes", Limit: maxBytes, Required: jxrSaturatedAdd(totalLocal, layout.alphaBytes), Detail: "uncompressed row alpha output bytes"}
		}
		totalLocal += layout.alphaBytes
	}
	binary.LittleEndian.PutUint32(header[offsets.imageSize:], uint32(layout.mainBytes))
	if settings.SeparateAlpha {
		binary.LittleEndian.PutUint32(header[offsets.alphaOffset:], uint32(alphaLocal))
		binary.LittleEndian.PutUint32(header[offsets.alphaSize:], uint32(layout.alphaBytes))
	}
	link, err := relocateJXRImageDirectory(header, start)
	if err != nil {
		return 0, 0, err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, 0, err
	}
	wanted := int64(start)
	if start == 8 {
		wanted = 0
	}
	if position != wanted {
		return 0, 0, fmt.Errorf("%w: uncompressed row output position", ErrInvalid)
	}
	writeHeader := header
	if start != 8 {
		writeHeader = header[8:]
	}
	step := 1
	for _, component := range info.Components {
		step = max(step, int(component.YStep))
	}
	cleanInfo := info
	cleanInfo.ICCProfile, cleanInfo.EXIF, cleanInfo.GPS, cleanInfo.XMP, cleanInfo.IPTC, cleanInfo.Description = nil, nil, nil, nil, nil, nil
	rowInfo, rowRasterWork, err := jxrRegionInfo(cleanInfo, image.Rect(0, 0, info.Bounds.Dx(), step))
	if err != nil {
		return 0, 0, err
	}
	baseHeld := retained + uint64(cap(header))
	mainStart := delta + mainLocal
	alphaStart := delta + alphaLocal
	total := delta + totalLocal
	if transform&4 != 0 {
		if err := writeJXRUncompressedRowTransposed(ctx, w, info, rowInfo, read, &layout, writeHeader, transform, step, padMain, baseHeld, rowRasterWork, limits); err != nil {
			return 0, 0, err
		}
		return total, link, nil
	}
	if err := writeJXRUncompressedRowStream(ctx, w, info, rowInfo, read, &layout, writeHeader, transform, step, mainStart, alphaStart, total, baseHeld, rowRasterWork, limits); err != nil {
		return 0, 0, err
	}
	return total, link, nil
}

// writeJXRUncompressedRowStream 在无转置方向下以单行带常量高度流式打包并写出无压缩容器图像
// 入参: ctx Context, w 位于当前目录起点的可定位输出, info 完整图像信息, rowInfo 单行带描述, read 行带回调, layout 无压缩布局, header 当前目录前缀, transform 编码方向, step 单次读取行高, mainStart 主区起点, alphaStart 透明区起点, total 当前图像结束位置, baseHeld 已占内存, rowRasterWork 行带内存, limits 资源限制
// 返回: error 样本、资源、源、取消或I/O错误
func writeJXRUncompressedRowStream(ctx context.Context, w io.WriteSeeker, info, rowInfo Info, read RasterRowReader, layout *jxrUncompressedLayout, header []byte, transform uint8, step int, mainStart, alphaStart, total, baseHeld, rowRasterWork uint64, limits Limits) error {
	bufWork := layout.rowBufferWork()
	if bufWork > uint64(^uint(0)>>1) {
		return &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: bufWork, Detail: "uncompressed row buffer capacity"}
	}
	if _, ok := jxrWorkTotal(limits.MaxMemoryBytes, baseHeld, rowRasterWork, bufWork); !ok {
		return &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(baseHeld, jxrSaturatedAdd(rowRasterWork, bufWork)), Detail: "uncompressed row streaming memory"}
	}
	rowRaster, err := newRasterContext(ctx, rowInfo, Limits{MaxPixels: limits.MaxPixels, MaxMemoryBytes: rowRasterWork})
	if err != nil {
		return err
	}
	rowBuf := make([]byte, int(layout.rowBytes))
	var alphaBuf []byte
	if !layout.planar && layout.separateAlpha {
		alphaBuf = make([]byte, int(layout.alphaRowBytes))
	}
	if err := writeEncoderOutput(ctx, w, header); err != nil {
		return err
	}
	needSeek := layout.planar || layout.separateAlpha || transform&1 != 0
	if needSeek {
		var zeros [32768]byte
		for remaining := total - mainStart; remaining != 0; {
			chunk := min(remaining, uint64(len(zeros)))
			if err := writeEncoderOutput(ctx, w, zeros[:int(chunk)]); err != nil {
				return err
			}
			remaining -= chunk
		}
	}
	dispH := info.Bounds.Dy()
	for y := 0; y < dispH; y += step {
		if err := ctx.Err(); err != nil {
			return err
		}
		for pi := range rowRaster.planes {
			clear(rowRaster.planes[pi].values)
			clear(rowRaster.planes[pi].words)
		}
		view := *rowRaster
		if err := read(ctx, y, &view); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := jxrWriteUncompressedRowStreamStep(ctx, w, info, rowRaster.planes, layout, rowBuf, alphaBuf, transform, y, step, mainStart, alphaStart, needSeek); err != nil {
			return err
		}
	}
	if needSeek {
		if err := seekJXRImageOutput(ctx, w, total); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// writeJXRUncompressedRowTransposed 在转置方向下以单遍行带散射打包无压缩载荷并顺序写出
// 入参: ctx Context, w 位于当前目录起点的可定位输出, info 完整图像信息, rowInfo 单行带描述, read 行带回调, layout 无压缩布局, header 当前目录前缀, transform 编码方向, step 单次读取行高, padMain 主区后是否需偶数对齐零字节, baseHeld 已占内存, rowRasterWork 行带内存, limits 资源限制
// 返回: error 样本、资源、源、取消或I/O错误
func writeJXRUncompressedRowTransposed(ctx context.Context, w io.WriteSeeker, info, rowInfo Info, read RasterRowReader, layout *jxrUncompressedLayout, header []byte, transform uint8, step int, padMain bool, baseHeld, rowRasterWork uint64, limits Limits) error {
	rawTotal := layout.mainBytes + layout.alphaBytes
	if rawTotal > uint64(^uint(0)>>1) {
		return &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: rawTotal, Detail: "uncompressed transposed payload capacity"}
	}
	if _, ok := jxrWorkTotal(limits.MaxMemoryBytes, baseHeld, rowRasterWork, rawTotal); !ok {
		return &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(baseHeld, jxrSaturatedAdd(rowRasterWork, rawTotal)), Detail: "uncompressed transposed row memory"}
	}
	rowRaster, err := newRasterContext(ctx, rowInfo, Limits{MaxPixels: limits.MaxPixels, MaxMemoryBytes: rowRasterWork})
	if err != nil {
		return err
	}
	mainData := make([]byte, int(layout.mainBytes))
	var alphaData []byte
	if layout.separateAlpha {
		alphaData = make([]byte, int(layout.alphaBytes))
	}
	dispH := info.Bounds.Dy()
	for y := 0; y < dispH; y += step {
		if err := ctx.Err(); err != nil {
			return err
		}
		for pi := range rowRaster.planes {
			clear(rowRaster.planes[pi].values)
			clear(rowRaster.planes[pi].words)
		}
		view := *rowRaster
		if err := read(ctx, y, &view); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := jxrScatterUncompressedRowTransposed(ctx, info, rowRaster.planes, layout, mainData, alphaData, transform, y, step); err != nil {
			return err
		}
	}
	if err := writeEncoderOutput(ctx, w, header); err != nil {
		return err
	}
	if err := writeEncoderOutput(ctx, w, mainData); err != nil {
		return err
	}
	if padMain {
		if err := writeEncoderOutput(ctx, w, []byte{0}); err != nil {
			return err
		}
	}
	if layout.separateAlpha {
		if err := writeEncoderOutput(ctx, w, alphaData); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// jxrWriteUncompressedRowStreamStep 在无转置方向下打包单步原生行并写出至可定位无压缩容器流
// 入参: ctx 已规范化Context, w 可定位输出, info 完整图像信息, rowPlanes 当前行带平面切片, layout 无压缩布局, rowBuf,alphaBuf 可复用打包行缓冲, transform 编码方向, y 当前显示行, step 单步行高, mainStart,alphaStart 主区与透明区起点, needSeek 是否需要定位写入
// 返回: error 打包、定位、写入或取消错误
func jxrWriteUncompressedRowStreamStep(ctx context.Context, w io.WriteSeeker, info Info, rowPlanes []imageComponents, layout *jxrUncompressedLayout, rowBuf, alphaBuf []byte, transform uint8, y, step int, mainStart, alphaStart uint64, needSeek bool) error {
	dispW := info.Bounds.Dx()
	codedW, codedH := int(layout.width), int(layout.height)
	totalComp := len(info.Components)
	if layout.planar {
		stepBytes := int(layout.sampleBytes)
		for ci := range totalComp {
			sx, sy, baseOffset, dstRowBytes, inAlpha := layout.planeSpec(ci)
			cw, ch := codedW/sx, codedH/sy
			plane := &rowPlanes[ci]
			baseFile := mainStart + baseOffset
			if inAlpha {
				baseFile = alphaStart + baseOffset
			}
			for ry := range step / sy {
				dy := y/sy + ry
				cy := dy
				if transform&1 != 0 {
					cy = ch - 1 - dy
				}
				rowOut := rowBuf[:int(dstRowBytes)]
				for dx := range cw {
					if dx&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					cx := dx
					if transform&2 != 0 {
						cx = cw - 1 - dx
					}
					val := plane.sampleBits(ry*cw + dx)
					if err := jxrPackUncompressedSample(rowOut[cx*stepBytes:], val, layout.sampleBytes, layout.tenBit); err != nil {
						return err
					}
				}
				if err := seekJXRImageOutput(ctx, w, baseFile+uint64(cy)*dstRowBytes); err != nil {
					return err
				}
				if err := writeEncoderOutput(ctx, w, rowOut); err != nil {
					return err
				}
			}
		}
		return ctx.Err()
	}
	clear(rowBuf)
	cy := y
	if transform&1 != 0 {
		cy = codedH - 1 - y
	}
	var samples [9]uint32
	for dx := range dispW {
		if dx&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		cx := dx
		if transform&2 != 0 {
			cx = codedW - 1 - dx
		}
		for ci := range totalComp {
			samples[ci] = rowPlanes[ci].sampleBits(dx)
		}
		if err := jxrPackUncompressedPixel(layout, rowBuf, alphaBuf, cx, samples[:totalComp]); err != nil {
			return err
		}
	}
	if !needSeek {
		return writeEncoderOutput(ctx, w, rowBuf)
	}
	if err := seekJXRImageOutput(ctx, w, mainStart+uint64(cy)*layout.rowBytes); err != nil {
		return err
	}
	if err := writeEncoderOutput(ctx, w, rowBuf); err != nil {
		return err
	}
	if layout.separateAlpha {
		if err := seekJXRImageOutput(ctx, w, alphaStart+uint64(cy)*layout.alphaRowBytes); err != nil {
			return err
		}
		if err := writeEncoderOutput(ctx, w, alphaBuf); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// jxrScatterUncompressedRowTransposed 在转置方向下将单步原生行散射打包至内存无压缩主区与可选透明区缓冲
// 入参: ctx 已规范化Context, info 完整图像信息, rowPlanes 当前行带平面切片, layout 无压缩布局, mainData,alphaData 主区与透明区缓冲, transform 编码方向, y 当前显示行, step 单步行高
// 返回: error 打包或取消错误
func jxrScatterUncompressedRowTransposed(ctx context.Context, info Info, rowPlanes []imageComponents, layout *jxrUncompressedLayout, mainData, alphaData []byte, transform uint8, y, step int) error {
	inverse := [8]uint8{0, 1, 2, 3, 7, 5, 6, 4}
	inv := inverse[transform]
	dispW, dispH := info.Bounds.Dx(), info.Bounds.Dy()
	totalComp := len(info.Components)
	if layout.planar {
		stepBytes := int(layout.sampleBytes)
		for ci := range totalComp {
			_, _, baseOffset, dstRowBytes, inAlpha := layout.planeSpec(ci)
			dst := mainData
			if inAlpha {
				dst = alphaData
			}
			comp := info.Components[ci]
			dsx, dsy := int(comp.XStep), int(comp.YStep)
			dw, dh := dispW/dsx, dispH/dsy
			plane := &rowPlanes[ci]
			for ry := range step / dsy {
				dy := y/dsy + ry
				for dx := range dw {
					if dx&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					val := plane.sampleBits(ry*dw + dx)
					cx, cy := jxrRasterPosition(dx, dy, dw, dh, inv)
					rowDst := dst[baseOffset+uint64(cy)*dstRowBytes:]
					if err := jxrPackUncompressedSample(rowDst[cx*stepBytes:], val, layout.sampleBytes, layout.tenBit); err != nil {
						return err
					}
				}
			}
		}
		return ctx.Err()
	}
	var samples [9]uint32
	for dx := range dispW {
		if dx&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for ci := range totalComp {
			samples[ci] = rowPlanes[ci].sampleBits(dx)
		}
		cx, cy := jxrRasterPosition(dx, y, dispW, dispH, inv)
		rowDst := mainData[uint64(cy)*layout.rowBytes:]
		var alphaRow []byte
		if layout.separateAlpha {
			alphaRow = alphaData[uint64(cy)*layout.alphaRowBytes:]
		}
		if err := jxrPackUncompressedPixel(layout, rowDst, alphaRow, cx, samples[:totalComp]); err != nil {
			return err
		}
	}
	return ctx.Err()
}
