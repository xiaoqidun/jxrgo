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
	"fmt"
	"image"
	"sort"
)

// jxrCoefficientRows 提供单个局部块的连续宏块行，具体布局自行保留其频带和预测状态
type jxrCoefficientRows interface {
	readRow(context.Context, *jxrCoefficientImage, int) error
}

// jxrRegionRowOutput 保存逐行转换共用的原生输出、坐标及工作分量，不持有整个局部块样本
type jxrRegionRowOutput struct {
	source    *jxrRasterSource
	target    *Raster
	origin    image.Point
	region    image.Rectangle
	width     int
	height    int
	transform uint8
	alphaOnly bool
	values    []int64
	codes     []uint32
	samples   *jxrUpsampleRows
}

// DecodeRegion 按显示方向逐块重建指定区域为原点归零的独立原生图像，不分配整幅系数图像
// region须非空、位于Info范围内且边界对齐各分量采样网格，仅解码相交分块及必要重叠邻域，NewDecoderAt按需读取
// 入参: ctx Context，nil采用Background, region 显示方向的像素区域
// 返回: *Raster 独立原生区域图像, error 区域、数据、能力、资源、取消或关闭错误
func (d *Decoder) DecodeRegion(ctx context.Context, region image.Rectangle) (*Raster, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	raster, err := d.decodeRegionLocked(ctx, region)
	if err != nil {
		return nil, wrapJXRError("Decoder.DecodeRegion", err)
	}
	return raster, nil
}

// decodeRegionLocked 在调用方持有锁时重建区域，供显式图像选择共用相同窗口及预算规则
// 入参: ctx 已规范化Context, region 显示方向区域
// 返回: *Raster 原生区域, error 区域、数据、资源、取消或关闭错误
func (d *Decoder) decodeRegionLocked(ctx context.Context, region image.Rectangle) (*Raster, error) {
	if d.closed {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	d.bindInputContext(ctx)
	defer d.bindInputContext(nil)
	if len(d.data) == 0 {
		return nil, fmt.Errorf("%w: uninitialized decoder", ErrInvalid)
	}
	info, outputWork, err := jxrRegionInfo(d.info, region)
	if err != nil {
		return nil, err
	}
	sources := [2]jxrRasterSource{d.source}
	sources[0].alpha = nil
	sources[0].container.alpha = nil
	sources[0].container.alphaInput = nil
	sources[0].container.format.alpha = d.source.primary.alpha != nil
	if !sources[0].container.format.alpha {
		sources[0].container.format.premultiply = false
	}
	passes := 1
	if d.source.alpha != nil {
		passes++
		s := d.source.alpha
		sources[1] = jxrCodedSource(*s, d.source.container.alpha)
		sources[1].container.transform = 0
		sources[1].setWorkers(d.workers)
	}
	retained := d.inputWork() + d.source.primary.headerWork() + uint64(cap(d.source.container.profiles))*4 + uint64(len(d.info.Components))*256 + uint64(len(d.info.ICCProfile)) + 4096
	if d.source.alpha != nil {
		retained += d.source.alpha.headerWork()
	}
	if retained > d.limits.MaxMemoryBytes || outputWork > d.limits.MaxMemoryBytes-retained {
		return nil, &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, outputWork), Detail: "JPEG XR region output memory"}
	}
	available := d.limits.MaxMemoryBytes - retained - outputWork
	for pass := range passes {
		source := &sources[pass]
		if err := checkJXRModelComponents(&source.primary.primary); err != nil {
			return nil, err
		}
		h := &source.primary.header
		count := (len(h.columns) - 1) * (len(h.rows) - 1)
		if count > 1 && !h.index {
			return nil, fmt.Errorf("%w: JPEG XR region packet index", ErrUnsupported)
		}
		bounds := source.regionTileRange(region, uint8(d.source.container.transform))
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				selected, _, err := source.regionTile(y*(len(h.columns)-1) + x)
				if err != nil {
					return nil, err
				}
				if work := selected.regionDecodeWork(); work > min(available, uint64(^uint(0)>>1)) {
					return nil, &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(jxrSaturatedAdd(retained, outputWork), work), Detail: "JPEG XR region tile memory"}
				}
			}
		}
	}
	output, err := newRasterContext(ctx, info, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: outputWork})
	if err != nil {
		return nil, err
	}
	for pass := range passes {
		source := &sources[pass]
		h := &source.primary.header
		bounds := source.regionTileRange(region, uint8(d.source.container.transform))
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				selected, coded, err := source.regionTile(y*(len(h.columns)-1) + x)
				if err != nil {
					return nil, err
				}
				if err := selected.copyRegionTile(ctx, output, coded.Min, region, int(h.width), int(h.height), uint8(d.source.container.transform), pass != 0, available); err != nil {
					return nil, err
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// jxrRegionInfo 保留原生数值及颜色元数据，建立网格对齐区域的零起点分量描述和存储预算
// 入参: original 原图信息, region 显示区域
// 返回: Info 区域信息, uint64 样本及元数据工作预算, error 区域或网格错误
func jxrRegionInfo(original Info, region image.Rectangle) (Info, uint64, error) {
	if region.Empty() || !region.In(original.Bounds) || len(original.Components) == 0 {
		return Info{}, 0, fmt.Errorf("%w: JPEG XR region bounds", ErrInvalid)
	}
	work := uint64(512) + uint64(len(original.Components))*256 + uint64(len(original.ICCProfile))
	if original.ColorEncoding != nil {
		work += 4
	}
	for _, component := range original.Components {
		sx, sy := int(component.XStep), int(component.YStep)
		if sx < 1 || sy < 1 || region.Min.X%sx != 0 || region.Max.X%sx != 0 || region.Min.Y%sy != 0 || region.Max.Y%sy != 0 {
			return Info{}, 0, fmt.Errorf("%w: JPEG XR region sampling alignment", ErrInvalid)
		}
		bytes := component.storageBytes()
		count := uint64(region.Dx()/sx) * uint64(region.Dy()/sy)
		if count > uint64(^uint(0)>>1)/bytes || count > (^uint64(0)-work)/bytes {
			return Info{}, 0, fmt.Errorf("%w: JPEG XR region sample storage", ErrLimit)
		}
		work += count * bytes
	}
	info := original
	info.Bounds = image.Rect(0, 0, region.Dx(), region.Dy())
	info.Components = cloneComponentInfo(original.Components)
	for index := range info.Components {
		component := &info.Components[index]
		component.Bounds = image.Rect(0, 0, region.Dx()/int(component.XStep), region.Dy()/int(component.YStep))
	}
	return info, work, nil
}

// regionTileRange 反映射显示区域并二分查找相交分块，不为未涉及的分块创建局部头部
// 入参: region 已校验的显示区域, transform 容器方向
// 返回: image.Rectangle 分块列行的半开范围
func (source *jxrRasterSource) regionTileRange(region image.Rectangle, transform uint8) image.Rectangle {
	h := &source.primary.header
	width, height := int(h.width), int(h.height)
	if transform&4 != 0 {
		width, height = height, width
	}
	inverse := [8]uint8{0, 1, 2, 3, 7, 5, 6, 4}
	coded := jxrRegionPosition(region, width, height, inverse[transform])
	x0 := sort.Search(len(h.columns)-1, func(n int) bool { return int(h.columns[n+1])*16-int(h.left) > coded.Min.X })
	x1 := sort.Search(len(h.columns)-1, func(n int) bool { return int(h.columns[n])*16-int(h.left) >= coded.Max.X })
	y0 := sort.Search(len(h.rows)-1, func(n int) bool { return int(h.rows[n+1])*16-int(h.top) > coded.Min.Y })
	y1 := sort.Search(len(h.rows)-1, func(n int) bool { return int(h.rows[n])*16-int(h.top) >= coded.Max.Y })
	return image.Rect(x0, y0, x1, y1)
}

// regionTile 建立目标原始分块及其必要相邻分块的局部视图，不复制或改写包字节
// 入参: tile 原码流中的分块编号
// 返回: jxrRasterSource 局部重建视图, image.Rectangle 目标分块的原编码可见区域, error 索引错误
func (source *jxrRasterSource) regionTile(tile int) (jxrRasterSource, image.Rectangle, error) {
	h := &source.primary.header
	columns, rows := len(h.columns)-1, len(h.rows)-1
	if columns < 1 || rows < 1 || tile < 0 || tile >= columns*rows {
		return jxrRasterSource{}, image.Rectangle{}, fmt.Errorf("%w: JPEG XR region tile index", ErrInvalid)
	}
	x, y := tile%columns, tile/columns
	x0, y0 := int(h.columns[x])*16, int(h.rows[y])*16
	x1, y1 := int(h.columns[x+1])*16, int(h.rows[y+1])*16
	coded := image.Rect(x0-int(h.left), y0-int(h.top), x1-int(h.left), y1-int(h.top)).Intersect(image.Rect(0, 0, int(h.width), int(h.height)))
	if coded.Empty() {
		return jxrRasterSource{}, coded, nil
	}
	selected := *source
	selected.primary.header = *h
	local := &selected.primary.header
	local.width, local.height, local.transform = uint64(coded.Dx()), uint64(coded.Dy()), 0
	local.left, local.top = uint32(coded.Min.X+int(h.left)-x0), uint32(coded.Min.Y+int(h.top)-y0)
	local.right, local.bottom = uint32(x1-x0-coded.Dx())-local.left, uint32(y1-y0-coded.Dy())-local.top
	local.columns, local.rows = []uint32{0, uint32((x1 - x0) / 16)}, []uint32{0, uint32((y1 - y0) / 16)}
	packets := 1
	if h.frequency {
		packets = 4 - int(source.primary.primary.bands)
	}
	if tile*packets > len(source.primary.packets)-packets {
		return jxrRasterSource{}, image.Rectangle{}, fmt.Errorf("%w: JPEG XR region packet storage", ErrInvalid)
	}
	selected.primary.packets = source.primary.packets[tile*packets : (tile+1)*packets : (tile+1)*packets]
	selected.container.width, selected.container.height, selected.container.transform = uint64(coded.Dx()), uint64(coded.Dy()), 0
	selected.container.profile, selected.container.encoding = nil, nil
	selected.container.colorHint = ColorHintNone
	upsample := (source.primary.primary.color == 1 || source.primary.primary.color == 2) && (h.color == 3 || h.color == 7)
	if columns*rows > 1 && (h.overlap != 0 && !h.hardTiles || upsample) {
		return source.regionNeighborhood(tile, selected, coded)
	}
	return selected, coded, nil
}

// jxrRegionPosition 将编码可见网格中的矩形映射到显示方向，保留半开边界
// 入参: bounds 原矩形, width, height 完整编码图像尺寸, transform 容器方向
// 返回: image.Rectangle 显示网格矩形
func jxrRegionPosition(bounds image.Rectangle, width, height int, transform uint8) image.Rectangle {
	x0, y0 := jxrRasterPosition(bounds.Min.X, bounds.Min.Y, width, height, transform)
	x1, y1 := jxrRasterPosition(bounds.Max.X-1, bounds.Max.Y-1, width, height, transform)
	return image.Rect(min(x0, x1), min(y0, y1), max(x0, x1)+1, max(y0, y1)+1)
}

// copyRegionTile 重建单块并复制所需样本，局部图像引用不跨越下一块的分配
// 入参: ctx Context, target 区域输出, origin 分块编码网格原点, region 显示区域, width, height 原编码尺寸, transform 显示方向, alphaOnly 是否只复制独立透明度, maximum 单块工作预算
// 返回: error 重建、网格、样本或取消错误
func (source *jxrRasterSource) copyRegionTile(ctx context.Context, target *Raster, origin image.Point, region image.Rectangle, width, height int, transform uint8, alphaOnly bool, maximum uint64) error {
	if source.coefficientRegionRows() {
		return source.copyCoefficientRegionRows(ctx, target, origin, region, width, height, transform, alphaOnly, maximum)
	}
	stride := source.container.format.channels
	if source.container.format.alpha {
		stride++
	}
	units := (uint64(stride)*source.sampleBytes() + 7) / 8
	tile, err := source.decodeContext(ctx, uint64(source.container.width)*uint64(source.container.height)*units, maximum)
	if err != nil {
		return err
	}
	start := 0
	if alphaOnly {
		start = len(tile.info.Components) - 1
	}
	for index := start; index < len(tile.info.Components); index++ {
		component, err := tile.Component(index)
		if err != nil {
			return err
		}
		destination := index
		if alphaOnly {
			destination = len(target.info.Components) - 1
		}
		out := &target.planes[destination]
		info := target.info.Components[destination]
		sx, sy := int(component.info.XStep), int(component.info.YStep)
		x0, y0 := region.Min.X/int(info.XStep), region.Min.Y/int(info.YStep)
		for y := range component.Bounds().Dy() {
			if err := ctx.Err(); err != nil {
				return err
			}
			for x := range component.Bounds().Dx() {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				px, py := jxrRasterPosition(x+origin.X/sx, y+origin.Y/sy, width/sx, height/sy, transform)
				px, py = px-x0, py-y0
				if !image.Pt(px, py).In(out.bounds) {
					continue
				}
				bits, err := component.SampleBits(x, y)
				if err != nil {
					return err
				}
				out.storeBits(py*out.bounds.Dx()+px, bits)
			}
		}
	}
	return nil
}

// regionNeighborhood 保留原始分块预测边界并在目标块外扩展两宏块重叠重建邻域，仅裁剪输出描述
// 入参: tile 原分块编号, selected 目标分块视图, coded 原编码可见区域
// 返回: jxrRasterSource 含邻块的局部重建源, image.Rectangle 目标可见区域, error 索引错误
func (source *jxrRasterSource) regionNeighborhood(tile int, selected jxrRasterSource, coded image.Rectangle) (jxrRasterSource, image.Rectangle, error) {
	h := &source.primary.header
	columns := len(h.columns) - 1
	x, y := tile%columns, tile/columns
	x0 := sort.Search(len(h.columns)-1, func(n int) bool { return h.columns[n+1] > uint32(max(0, int(h.columns[x])-2)) })
	x1 := sort.Search(len(h.columns)-1, func(n int) bool { return h.columns[n] >= h.columns[x+1]+2 })
	y0 := sort.Search(len(h.rows)-1, func(n int) bool { return h.rows[n+1] > uint32(max(0, int(h.rows[y])-2)) })
	y1 := sort.Search(len(h.rows)-1, func(n int) bool { return h.rows[n] >= h.rows[y+1]+2 })
	packets := 1
	if h.frequency {
		packets = 4 - int(source.primary.primary.bands)
	}
	if x0 >= x1 || y0 >= y1 || x1-x0 > 5 || y1-y0 > 5 || (y1*columns)*packets > len(source.primary.packets) {
		return jxrRasterSource{}, image.Rectangle{}, fmt.Errorf("%w: JPEG XR region neighborhood index", ErrInvalid)
	}
	local := &selected.primary.header
	local.columns, local.rows = make([]uint32, x1-x0+1), make([]uint32, y1-y0+1)
	for n := range local.columns {
		local.columns[n] = h.columns[x0+n] - h.columns[x0]
	}
	for n := range local.rows {
		local.rows[n] = h.rows[y0+n] - h.rows[y0]
	}
	local.left, local.top = uint32(coded.Min.X+int(h.left))-h.columns[x0]*16, uint32(coded.Min.Y+int(h.top))-h.rows[y0]*16
	local.right = local.columns[len(local.columns)-1]*16 - local.left - uint32(coded.Dx())
	local.bottom = local.rows[len(local.rows)-1]*16 - local.top - uint32(coded.Dy())
	selected.primary.packets = make([]uint64, (x1-x0)*(y1-y0)*packets)
	for row := y0; row < y1; row++ {
		from := (row*columns + x0) * packets
		to := (row - y0) * (x1 - x0) * packets
		copy(selected.primary.packets[to:to+(x1-x0)*packets], source.primary.packets[from:from+(x1-x0)*packets])
	}
	return selected, coded, nil
}

// coefficientRegionRows 判定局部块能否逐行重建，额外熵状态和邻行窗口只在总容量较整块更小时采用
// 返回: bool 是否采用连续行系数及必要重叠邻域
func (source *jxrRasterSource) coefficientRegionRows() bool {
	s := &source.primary
	if source.alpha != nil || s.header.overlap > 2 || len(s.header.columns) < 2 || len(s.header.rows) < 2 || len(s.header.columns) > 6 || len(s.header.rows) > 6 {
		return false
	}
	multiple := len(s.header.columns) != 2 || len(s.header.rows) != 2
	color := source.container.format.color
	if color == 1 || color == 2 {
		if s.primary.color != color || s.header.left%2 != 0 || color == 1 && s.header.top%2 != 0 {
			return false
		}
	} else if (s.primary.color == 1 || s.primary.color == 2) && s.header.color != 3 && s.header.color != 7 {
		return false
	}
	if !multiple && s.header.overlap == 0 && !s.header.frequency && jxrUpsampleRowWork(s) == 0 {
		return true
	}
	work := jxrCoefficientRowWork(s)
	auxiliary := uint64(max(3, s.primary.channels, source.container.format.channels))*12 + 256
	return work <= ^uint64(0)-auxiliary && work+auxiliary < source.decodeWork()
}

// regionDecodeWork 计算局部块实际采用的重建工作，符合行条件时不计整块系数或临时样本图
// 返回: uint64 工作字节数，溢出返回最大值
func (source *jxrRasterSource) regionDecodeWork() uint64 {
	if !source.coefficientRegionRows() {
		return source.decodeWork()
	}
	work := jxrCoefficientRowWork(&source.primary)
	auxiliary := uint64(max(3, source.primary.primary.channels, source.container.format.channels))*12 + 256
	if work > ^uint64(0)-auxiliary {
		return ^uint64(0)
	}
	return work + auxiliary
}

// copyCoefficientRegionRows 逐行解码局部块全部选中包并校验熵与可见样本，仅向目标区域写入原生码值
// 入参: ctx Context, target 区域输出, origin 局部块编码原点, region 显示区域, width, height 原编码尺寸, transform 显示方向, alphaOnly 是否只复制独立透明度, maximum 当前块预算
// 返回: error 数据、数值、资源或取消错误
func (source *jxrRasterSource) copyCoefficientRegionRows(ctx context.Context, target *Raster, origin image.Point, region image.Rectangle, width, height int, transform uint8, alphaOnly bool, maximum uint64) error {
	if source.regionDecodeWork() > min(maximum, uint64(^uint(0)>>1)) {
		return fmt.Errorf("%w: JPEG XR region row memory", ErrLimit)
	}
	s := &source.primary
	frame := jxrRowReconstructionStream(s)
	var reader jxrCoefficientRows
	var err error
	if len(s.header.columns) != 2 || len(s.header.rows) != 2 {
		reader, err = newJXRNeighborhoodRows(ctx, source.container.image, s, maximum)
	} else if s.header.frequency {
		reader, err = newJXRFrequencyRows(ctx, source.container.image, s, 0, maximum)
	} else {
		reader, err = newJXRSpatialRows(ctx, source.container.image, s, 0, maximum)
	}
	if err != nil {
		return err
	}
	channels := max(3, s.primary.channels, source.container.format.channels)
	output := jxrRegionRowOutput{source: source, target: target, origin: origin, region: region, width: width, height: height, transform: transform, alphaOnly: alphaOnly, values: make([]int64, channels), codes: make([]uint32, channels)}
	if jxrUpsampleRowWork(&frame) != 0 {
		output.samples, err = newJXRUpsampleRows(ctx, &frame, maximum)
		if err != nil {
			return err
		}
	}
	if err := walkJXRReconstructedRows(ctx, s, reader, maximum, output.accept); err != nil {
		return err
	}
	return output.finish(ctx)
}

// accept 将已完成后滤波的样本行交给必要色度邻域缓存，无垂直上采样时直接复制区域
// 入参: ctx Context, h 当前样本窗口, views 主分量样本, alpha 透明度样本, row 原块行号
// 返回: error 行缓存、样本转换、输出或取消错误
func (output *jxrRegionRowOutput) accept(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error {
	if output.samples != nil {
		return output.samples.accept(ctx, jxrSampleRowOutput{region: output}, h, views, alpha, row)
	}
	return output.copy(ctx, h, views, alpha, row)
}

// finish 在熵和重建全部成功后刷新延迟的色度输出，不允许尾包失败返回部分图像
// 入参: ctx Context
// 返回: error 最后一行输出、数值或取消错误
func (output *jxrRegionRowOutput) finish(ctx context.Context) error {
	if output.samples != nil {
		return output.samples.finish(ctx, jxrSampleRowOutput{region: output})
	}
	return ctx.Err()
}

// copy 将完整重建行的可见原生码值复制到目标区域，数值检查仍覆盖该块的全部可见样本
// 入参: ctx Context, h 当前样本窗口及目标行偏移, views 主平面样本, alpha 透明度视图, macroY 原块宏块行号
// 返回: error 输出数值、分量或取消错误
func (output *jxrRegionRowOutput) copy(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, macroY int) error {
	if color := output.source.container.format.color; color == 1 || color == 2 {
		return output.copyNative(ctx, h, views, alpha, macroY)
	}
	source := output.source
	s := &source.primary
	target, origin, region := output.target, output.origin, output.region
	width, height, transform, alphaOnly := output.width, output.height, output.transform, output.alphaOnly
	values, codes := output.values, output.codes
	for y := range 16 {
		if err := ctx.Err(); err != nil {
			return err
		}
		visibleY := macroY*16 + y - int(s.header.top)
		if visibleY < 0 || visibleY >= int(s.header.height) {
			continue
		}
		for x := range int(s.header.width) {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if err := jxrRasterSampleComponents(views, h, &s.primary, x, y, values); err != nil {
				return err
			}
			count, err := jxrRasterBitsComponents(h, &s.primary, source.container.format.number, values, false, codes)
			if err != nil {
				return err
			}
			if count != source.container.format.channels {
				return fmt.Errorf("%w: JPEG XR spatial region row channels", ErrInvalid)
			}
			alphaCode := uint32(0)
			if alpha != nil {
				sample := [1]int64{alpha.at(x+int(h.left), y+int(h.top))}
				var result [1]uint32
				if _, err := jxrRasterBitsComponents(h, s.alpha, source.container.format.number, sample[:], true, result[:]); err != nil {
					return err
				}
				alphaCode = result[0]
			}
			px, py := jxrRasterPosition(x+origin.X, visibleY+origin.Y, width, height, transform)
			if !image.Pt(px, py).In(region) {
				continue
			}
			at := (py-region.Min.Y)*region.Dx() + px - region.Min.X
			if !alphaOnly {
				for i := range count {
					target.planes[i].storeBits(at, codes[i])
				}
			}
			if alphaOnly && !source.container.format.alpha {
				alphaCode = codes[0]
			}
			if source.container.format.alpha || alphaOnly {
				target.planes[len(target.planes)-1].storeBits(at, alphaCode)
			}
		}
	}
	return ctx.Err()
}

// copyNative 按各原生分量采样网格转换已重建行，不进行色度插值、RGB转换或相位推断
// 入参: ctx Context, h 当前样本窗口, views 主平面样本视图, alpha 透明度视图, macroY 原块宏块行号
// 返回: error 输出数值、透明度布局或取消错误
func (output *jxrRegionRowOutput) copyNative(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, macroY int) error {
	source, target := output.source, output.target
	s := &source.primary
	count := source.container.format.channels
	if source.container.format.alpha {
		count++
	}
	first := 0
	if output.alphaOnly {
		first = source.container.format.channels
	}
	for index := first; index < count; index++ {
		sx, sy := 1, 1
		if index > 0 && index < source.container.format.channels {
			sx = 2
			if source.container.format.color == 1 {
				sy = 2
			}
		}
		destination := index
		if output.alphaOnly {
			destination = len(target.info.Components) - 1
		}
		info, out := target.info.Components[destination], &target.planes[destination]
		for y := range 16 / sy {
			if err := ctx.Err(); err != nil {
				return err
			}
			visibleY := macroY*16/sy + y - int(s.header.top)/sy
			if visibleY < 0 || visibleY >= int(s.header.height)/sy {
				continue
			}
			for x := range int(s.header.width) / sx {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				var values [3]int64
				valueIndex := index
				if index == source.container.format.channels {
					if alpha == nil || s.alpha == nil {
						return fmt.Errorf("%w: JPEG XR native row alpha plane", ErrInvalid)
					}
					values[0] = alpha.at(x+int(h.left), y+int(h.top))
					if _, err := jxrIntegerOutputComponents(h, s.alpha, values[:], true); err != nil {
						return err
					}
					valueIndex = 0
				} else {
					values[index] = views[index].at(x+int(h.left)/sx, y+int(h.top)/sy)
					if _, err := jxrIntegerOutputComponents(h, &s.primary, values[:], false); err != nil {
						return err
					}
				}
				px, py := jxrRasterPosition(x+output.origin.X/sx, visibleY+output.origin.Y/sy, output.width/sx, output.height/sy, output.transform)
				px, py = px-output.region.Min.X/int(info.XStep), py-output.region.Min.Y/int(info.YStep)
				if image.Pt(px, py).In(out.bounds) {
					out.storeBits(py*out.bounds.Dx()+px, uint32(values[valueIndex]))
				}
			}
		}
	}
	return ctx.Err()
}
