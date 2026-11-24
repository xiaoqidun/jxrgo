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

// jxrAlphaRowStorage 保存单遍行源的透明度码值，使用有界内存或调用方提供的暂存
type jxrAlphaRowStorage struct {
	storage     io.ReadWriteSeeker
	data        []byte
	buffer      []byte
	length      uint64
	position    uint64
	sampleBytes int
}

// jxrSeparateInput 保存独立透明度的真实样本、两个编码计划及共同常驻容量，不保存完整压缩结果
type jxrSeparateInput struct {
	primary Raster
	alpha   *Raster
	layout  [2]jxrStream
	output  [2]jxrStreamOutput
	working uint64
}

// jxrSeparateOutput 明确双流目录布局和前缀预算，将样本装载方式与压缩包提交分离
type jxrSeparateOutput struct {
	info         Info
	layout       [2]jxrStream
	maximum      uint64
	prefixMemory uint64
	maxPixels    uint64
}

// jxrSeparateTileInput 保存有界双流提供器的独占元数据、编码计划和共用工作额度，不保存整幅样本
type jxrSeparateTileInput struct {
	file   jxrSeparateOutput
	info   [2]Info
	output [2]jxrStreamOutput
	mirror bool
}

// encodeJXRSeparateAlpha 将颜色与透明度写入独立码流，共用输入、已编码数据和封装的容量预算
// 入参: ctx Context, raster 原生图像, options 文件编码选项
// 返回: []byte 完整JXR文件, error 参数、精度、资源或取消错误
func encodeJXRSeparateAlpha(ctx context.Context, raster *Raster, options EncodeOptions) ([]byte, error) {
	if err := checkJXREncodeOptions(options); err != nil {
		return nil, err
	}
	limits := options.Limits.normalized()
	inputWork, err := jxrRasterWork(raster, limits.MaxMemoryBytes)
	if err != nil {
		return nil, err
	}
	if !raster.info.Alpha || raster.info.Channels < 1 || len(raster.info.Components) != raster.info.Channels+1 {
		return nil, fmt.Errorf("%w: separate alpha source components", ErrInvalid)
	}
	identifier, format, err := jxrRasterIdentifier(raster.info, [16]byte{})
	if err != nil {
		return nil, err
	}
	primary := *raster
	primary.info.Alpha, primary.info.Premultiplied, primary.info.PixelFormat = false, false, ""
	primary.info.Components = primary.info.Components[:raster.info.Channels:raster.info.Channels]
	primaryWork, err := jxrRasterWork(&primary, limits.MaxMemoryBytes)
	if err != nil {
		return nil, err
	}
	if inputWork >= limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "separate alpha input memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(inputWork, 1)}
	}
	alpha, err := newJXRAlphaRaster(ctx, raster, options.MirrorSeparateAlpha, limits.MaxMemoryBytes-inputWork, limits.MaxPixels)
	if err != nil {
		return nil, err
	}
	alphaWork, err := jxrRasterWork(alpha, limits.MaxMemoryBytes-inputWork)
	if err != nil {
		return nil, err
	}
	additional := inputWork - primaryWork + alphaWork
	alphaOptions, alphaOptionsWork, err := jxrSeparateAlphaOptions(ctx, options, limits.MaxMemoryBytes-inputWork-alphaWork)
	if err != nil {
		return nil, err
	}
	if inputWork+alphaWork+alphaOptionsWork >= limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "separate alpha options memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(inputWork+alphaWork+alphaOptionsWork, 1)}
	}
	mainOptions := options
	mainOptions.Format, mainOptions.SeparateAlpha, mainOptions.MirrorSeparateAlpha, mainOptions.Alpha = FormatCodestream, false, false, nil
	mainOptions.Limits = limits
	mainOptions.Limits.MaxMemoryBytes -= additional + alphaOptionsWork
	main, mainOutput, err := makeJXREncodingPlan(ctx, &primary, &mainOptions)
	if err != nil {
		return nil, jxrPlanLimitError(err, mainOutput.stream.maxMemory, primaryWork)
	}
	if main.headerWork() >= mainOutput.stream.maxMemory {
		return nil, &LimitError{Resource: "separate alpha primary plan memory", Limit: mainOutput.stream.maxMemory, Required: jxrSaturatedAdd(main.headerWork(), 1)}
	}
	mainOutput.stream.maxMemory -= main.headerWork()
	codedPrimary, err := encodeJXRNative(ctx, &primary, main, mainOutput)
	if err != nil {
		return nil, err
	}
	settingsWork, err := mainOutput.stream.tileWork(ctx, limits.MaxMemoryBytes)
	if err != nil {
		return nil, err
	}
	retained := inputWork + alphaOptionsWork
	for _, amount := range []uint64{uint64(cap(codedPrimary)), main.headerWork(), settingsWork} {
		if amount >= limits.MaxMemoryBytes-retained {
			return nil, &LimitError{Resource: "separate alpha retained primary memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, jxrSaturatedAdd(amount, 1))}
		}
		retained += amount
	}
	alphaOptions.Orientation = Orientation(main.header.transform)
	alphaOptions.Limits = limits
	alphaOptions.Limits.MaxMemoryBytes -= retained
	secondary, alphaOutput, err := makeJXREncodingPlan(ctx, alpha, &alphaOptions)
	if err != nil {
		return nil, jxrPlanLimitError(err, alphaOutput.stream.maxMemory, alphaWork)
	}
	prepareJXRSeparateAlphaPrimary(&secondary, &alphaOutput.stream, options.MirrorSeparateAlpha)
	if secondary.headerWork() >= alphaOutput.stream.maxMemory {
		return nil, &LimitError{Resource: "separate alpha secondary plan memory", Limit: alphaOutput.stream.maxMemory, Required: jxrSaturatedAdd(secondary.headerWork(), 1)}
	}
	alphaOutput.stream.maxMemory -= secondary.headerWork()
	codedAlpha, err := encodeJXRNative(ctx, alpha, secondary, alphaOutput)
	if err != nil {
		return nil, err
	}
	alphaSettings, err := alphaOutput.stream.tileWork(ctx, limits.MaxMemoryBytes)
	if err != nil {
		return nil, err
	}
	for _, amount := range []uint64{alphaWork, uint64(cap(codedAlpha)), secondary.headerWork(), alphaSettings, 32768} {
		if amount >= limits.MaxMemoryBytes-retained {
			return nil, &LimitError{Resource: "separate alpha container working memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, jxrSaturatedAdd(amount, 1))}
		}
		retained += amount
	}
	c := jxrContainer{width: main.header.width, height: main.header.height, dpi: raster.info.Resolution, format: format, pixelFormat: identifier, transform: uint32(main.header.transform), image: codedPrimary, alpha: codedAlpha, profiles: main.profiles, imageBands: uint32(main.primary.bands), alphaBands: uint32(secondary.alpha.bands), colorHint: raster.info.ColorHint, encoding: raster.info.ColorEncoding, imageType: raster.info.ImageType}
	meta := jxrWithContainerMetadata(raster.info, &options)
	if len(meta.ICCProfile) != 0 {
		c.profile = meta.ICCProfile
	}
	if len(meta.EXIF) != 0 {
		c.exif = meta.EXIF
	}
	if len(meta.XMP) != 0 {
		c.xmp = meta.XMP
	}
	if len(meta.IPTC) != 0 {
		c.iptc = meta.IPTC
	}
	if main.header.transform&4 != 0 {
		c.dpi.X, c.dpi.Y = c.dpi.Y, c.dpi.X
	}
	w := newJXRBitWriter(ctx, mainOutput.stream.maxBytes, limits.MaxMemoryBytes-retained)
	writeJXRContainer(&w, &c, limits.MaxPixels)
	if w.err != nil {
		return nil, w.err
	}
	return w.data, ctx.Err()
}

// newJXRAlphaRaster 创建标准独立透明度的辅助主图及透明度，十位格式采用RGB输出及单分量内部主平面
// 入参: ctx Context, raster 原始图像, mirror 是否向辅助主平面复制样本, maxMemory 新图像字节上限, maxPixels 像素上限
// 返回: *Raster 占位主图及透明度, error 参数、资源或取消错误
func newJXRAlphaRaster(ctx context.Context, raster *Raster, mirror bool, maxMemory, maxPixels uint64) (*Raster, error) {
	component, err := raster.component(raster.info.Channels)
	if err != nil {
		return nil, err
	}
	info := jxrAlphaRasterInfo(raster.Bounds(), component.Info())
	output, err := newRasterContext(ctx, info, Limits{MaxMemoryBytes: maxMemory, MaxPixels: maxPixels})
	if err != nil {
		return nil, err
	}
	for y := range info.Bounds.Dy() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for x := range info.Bounds.Dx() {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			bits, err := component.sampleBitsAt(x, y)
			if err != nil {
				return nil, err
			}
			at := y*info.Bounds.Dx() + x
			output.planes[info.Channels].storeBits(at, bits)
			if mirror {
				for index := range info.Channels {
					output.planes[index].storeBits(at, bits)
				}
			}
		}
	}
	return output, ctx.Err()
}

// jxrAlphaRasterInfo 共用独立透明度及其辅助主平面的原生描述，十位输出采用标准RGB格式
// 入参: bounds 显示方向原生范围, component 透明度原生分量描述
// 返回: Info 辅助码流的原生描述，不包含或伪造样本存储
func jxrAlphaRasterInfo(bounds image.Rectangle, component ComponentInfo) Info {
	info := Info{Bounds: bounds, Channels: 1, ColorSpace: ColorGray, Alpha: true, Components: []ComponentInfo{component, component}}
	if component.Precision == 10 {
		info.Channels, info.ColorSpace = 3, ColorRGB
		info.Components = []ComponentInfo{component, component, component, component}
	}
	return info
}

// jxrSeparateAlphaOptions 在预算内快照独立透明度，默认逐组继承主平面首分量及完整宏块索引
// 入参: ctx Context, o 原始文件编码选项, maximum 原配置与独占副本的共同容量上限
// 返回: EncodeOptions 独立透明度选项, uint64 原配置与副本的保守常驻容量, error 资源或取消错误
func jxrSeparateAlphaOptions(ctx context.Context, o EncodeOptions, maximum uint64) (EncodeOptions, uint64, error) {
	p := PlaneOptions{Band: o.Band, Scaled: o.Scaled, Shift: o.Shift, Float: o.Float}
	q, first := o.Quantization, true
	if o.Alpha != nil {
		p = *o.Alpha
		q, first = p.Quantization, false
	}
	retained, err := jxrQuantizationWork(ctx, q, maximum)
	if err != nil {
		return EncodeOptions{}, 0, err
	}
	quantization, owned, err := snapshotJXRQuantization(ctx, q, maximum-retained, first)
	if err != nil {
		return EncodeOptions{}, 0, err
	}
	p.Quantization = quantization
	if p.Float != nil {
		value := *p.Float
		p.Float = &value
	}
	return EncodeOptions{Format: FormatCodestream, Layout: o.Layout, Overlap: o.Overlap, Orientation: o.Orientation, TileSize: o.TileSize,
		HardTiles: o.HardTiles, Lossy: o.Lossy, LegacyHighpassPrediction: o.LegacyHighpassPrediction, Workers: o.Workers,
		Band: p.Band, Trim: o.Trim, Scaled: p.Scaled, Shift: p.Shift, Float: p.Float, Alpha: &p, MaxOutputBytes: o.MaxOutputBytes}, retained + owned, ctx.Err()
}

// prepareJXRSeparateAlphaPrimary 确定辅助主平面的灰度布局，镜像时复用透明度的编码参数和量化表
// 入参: layout 已建立的双平面布局, output 含独占逐块量化表的输出计划, mirror 是否采用同一透明度编码参数
func prepareJXRSeparateAlphaPrimary(layout *jxrStream, output *jxrStreamOutput, mirror bool) {
	if mirror {
		layout.primary = *layout.alpha
		for index := range output.tiles {
			output.tiles[index].quant[0] = output.tiles[index].quant[1]
		}
		return
	}
	if layout.header.depth != 9 {
		return
	}
	layout.primary.color, layout.primary.channels = 0, 1
	layout.primary.dcQuant, layout.primary.lpQuant, layout.primary.hpQuant = nil, nil, nil
	layout.primary.dcUniform, layout.primary.lpUniform, layout.primary.hpUniform = true, true, true
	for index := range output.tiles {
		q := &output.tiles[index].quant[0]
		q.dc, q.lp[0], q.hp[0] = q.dc[:1], q.lp[0][:1], q.hp[0][:1]
		q.dc[0], q.lp[0][0], q.hp[0][0] = 0, 0, 0
	}
}

// makeJXRAlphaRowStorage 预留完整透明度缓存或固定大小I/O缓冲，不分配整幅颜色图像
// 入参: info 已规范化的原生描述, scratch 可选暂存, maximum 可用工作额度
// 返回: jxrAlphaRowStorage 透明度暂存, uint64 常驻额度, error 参数或资源错误
func makeJXRAlphaRowStorage(info Info, scratch io.ReadWriteSeeker, maximum uint64) (jxrAlphaRowStorage, uint64, error) {
	var result jxrAlphaRowStorage
	if !info.Alpha || info.Channels < 1 || len(info.Components) != info.Channels+1 {
		return result, 0, fmt.Errorf("%w: row alpha source components", ErrInvalid)
	}
	component := info.Components[info.Channels]
	if component.XStep != 1 || component.YStep != 1 {
		return result, 0, fmt.Errorf("%w: row alpha sample grid", ErrInvalid)
	}
	result.sampleBytes = 2
	if component.Precision == 32 {
		result.sampleBytes = 4
	}
	pixels := uint64(info.Bounds.Dx()) * uint64(info.Bounds.Dy())
	if pixels > uint64(^uint64(0)>>1)/uint64(result.sampleBytes) {
		return result, 0, &LimitError{Resource: "row alpha scratch extent", Limit: uint64(^uint64(0) >> 1), Required: jxrSaturatedMul(pixels, uint64(result.sampleBytes))}
	}
	result.length, result.storage = pixels*uint64(result.sampleBytes), scratch
	capacity := result.length
	if scratch != nil {
		capacity = min(capacity, 32768)
	}
	if maximum < 512 || capacity >= maximum-512 || capacity > uint64(^uint(0)>>1) {
		return result, 0, &LimitError{Resource: "row alpha scratch memory", Limit: maximum, Required: jxrSaturatedAdd(512, jxrSaturatedAdd(capacity, 1))}
	}
	if scratch == nil {
		result.data = make([]byte, int(capacity))
	} else {
		result.buffer = make([]byte, int(capacity))
	}
	return result, 512 + capacity, nil
}

// encodeJXRSeparateRows 复用主图行缓存，首遍暂存透明度，次遍不再读取原始行源
// 入参: ctx Context, w 文件输出, info 完整描述, source 共用行缓存, alpha 透明度暂存, options 文件选项
// 返回: error 参数、资源、源、暂存、取消或输出错误
func encodeJXRSeparateRows(ctx context.Context, w io.WriteSeeker, info Info, source *jxrRowInput, alpha *jxrAlphaRowStorage, options EncodeOptions) error {
	if info.ImageType&ImageTypePreview != 0 {
		return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	info, retained, err := prepareJXRTileInfo(ctx, info, options)
	if err != nil {
		return err
	}
	input, err := makeJXRSeparateTileInput(ctx, info, options, retained)
	if err != nil {
		return err
	}
	if alpha.storage != nil {
		if sameInterfaceTarget(w, alpha.storage) {
			return fmt.Errorf("%w: row alpha scratch aliases output", ErrInvalid)
		}
		if err := seekJXRImageOutput(ctx, alpha.storage, 0); err != nil {
			return err
		}
	}
	read := source.read
	source.read = func(ctx context.Context, y int, target *Raster) error {
		if err := read(ctx, y, target); err != nil {
			return err
		}
		return alpha.transfer(ctx, &source.row.planes[info.Channels], true)
	}
	_, _, err = writeJXRSeparatePacketsAt(ctx, w, input.file, 8, func(index int, maximum uint64) error {
		if index == 1 {
			if source.next != source.height || alpha.position != alpha.length {
				return fmt.Errorf("%w: incomplete row alpha scratch", ErrInvalid)
			}
			if alpha.storage != nil {
				if err := seekJXRImageOutput(ctx, alpha.storage, 0); err != nil {
					return err
				}
			}
			alpha.position, source.next = 0, 0
			source.read = func(ctx context.Context, y int, target *Raster) error {
				return alpha.transfer(ctx, &target.planes[info.Channels], false)
			}
		}
		selected := input.output[index]
		selected.maxBytes, selected.rowOrder = maximum, true
		return encodeJXRProjectedTilesTo(ctx, w, input.file.info, input.info[index], &input.file.layout[index], selected, source.load, index == 1, input.mirror)
	})
	return err
}

// transfer 按原生位模式保存或恢复一条透明度行带，不改变精度、预乘状态或浮点特殊值
// 入参: ctx Context, plane 临时透明度行带, save 是否写入暂存
// 返回: error 容量、暂存或取消错误
func (s *jxrAlphaRowStorage) transfer(ctx context.Context, plane *imageComponents, save bool) error {
	count := plane.bounds.Dx() * plane.bounds.Dy()
	if s.position > s.length || uint64(count) > (s.length-s.position)/uint64(s.sampleBytes) {
		return fmt.Errorf("%w: row alpha scratch capacity", ErrInvalid)
	}
	for at := 0; at < count; {
		if err := ctx.Err(); err != nil {
			return err
		}
		part := min(count-at, 32768/s.sampleBytes)
		size := part * s.sampleBytes
		var data []byte
		if s.storage == nil {
			data = s.data[s.position : s.position+uint64(size)]
		} else {
			data = s.buffer[:size]
			if !save {
				if err := readJXRAlphaRowStorage(ctx, s.storage, data); err != nil {
					return err
				}
			}
		}
		for index := range part {
			value := data[index*s.sampleBytes:]
			if save {
				bits := plane.sampleBits(at + index)
				if s.sampleBytes == 2 {
					binary.LittleEndian.PutUint16(value, uint16(bits))
				} else {
					binary.LittleEndian.PutUint32(value, bits)
				}
			} else {
				bits := uint32(0)
				if s.sampleBytes == 2 {
					bits = uint32(binary.LittleEndian.Uint16(value))
				} else {
					bits = binary.LittleEndian.Uint32(value)
				}
				plane.storeBits(at+index, bits)
			}
		}
		if s.storage != nil && save {
			if err := writeEncoderOutput(ctx, s.storage, data); err != nil {
				return err
			}
		}
		s.position += uint64(size)
		at += part
	}
	return ctx.Err()
}

// readJXRAlphaRowStorage 填满当前暂存片段，校验读取计数并限制无进展读取
// 入参: ctx Context, source 透明度暂存, target 本次完整片段
// 返回: error 读取协议、底层、截断或取消错误
func readJXRAlphaRowStorage(ctx context.Context, source io.Reader, target []byte) error {
	empty := 0
	for len(target) != 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := source.Read(target)
		if n < 0 || n > len(target) {
			return fmt.Errorf("%w: row alpha scratch read count", ErrInvalid)
		}
		target = target[n:]
		if len(target) == 0 {
			if err != nil && err != io.EOF {
				return err
			}
			return ctx.Err()
		}
		if err != nil {
			if err == io.EOF {
				return io.ErrUnexpectedEOF
			}
			return err
		}
		if n == 0 {
			empty++
			if empty == 100 {
				return io.ErrNoProgress
			}
		} else {
			empty = 0
		}
	}
	return ctx.Err()
}

// makeJXRSeparateInput 在共用预算内建立主图视图和完整透明度副本，保留标准辅助主平面
// 入参: ctx Context, raster 原生图像, options 已校验文件选项
// 返回: jxrSeparateInput 常驻输入和编码计划, error 参数、能力、资源或取消错误
func makeJXRSeparateInput(ctx context.Context, raster *Raster, options EncodeOptions) (jxrSeparateInput, error) {
	var result jxrSeparateInput
	limits := options.Limits.normalized()
	inputWork, err := jxrRasterWork(raster, limits.MaxMemoryBytes)
	if err != nil {
		return result, err
	}
	if !raster.info.Alpha || raster.info.Channels < 1 || len(raster.info.Components) != raster.info.Channels+1 {
		return result, fmt.Errorf("%w: separate alpha source components", ErrInvalid)
	}
	if _, _, err := jxrRasterIdentifier(raster.info, [16]byte{}); err != nil {
		return result, err
	}
	if inputWork >= limits.MaxMemoryBytes || 32768 >= limits.MaxMemoryBytes-inputWork {
		return result, &LimitError{Resource: "separate alpha input memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(inputWork, 32769)}
	}
	result.working = inputWork + 32768
	result.primary = *raster
	result.primary.info.Alpha, result.primary.info.Premultiplied, result.primary.info.PixelFormat = false, false, ""
	result.primary.info.Components = result.primary.info.Components[:raster.info.Channels:raster.info.Channels]
	result.alpha, err = newJXRAlphaRaster(ctx, raster, options.MirrorSeparateAlpha, limits.MaxMemoryBytes-result.working, limits.MaxPixels)
	if err != nil {
		return result, err
	}
	alphaWork, err := jxrRasterWork(result.alpha, limits.MaxMemoryBytes-result.working)
	if err != nil {
		return result, err
	}
	if alphaWork >= limits.MaxMemoryBytes-result.working {
		return result, &LimitError{Resource: "separate alpha copy memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(result.working, jxrSaturatedAdd(alphaWork, 1))}
	}
	result.working += alphaWork
	mainOptions := options
	mainOptions.Format, mainOptions.SeparateAlpha, mainOptions.MirrorSeparateAlpha, mainOptions.Alpha = FormatCodestream, false, false, nil
	alphaOptions, alphaOptionsWork, err := jxrSeparateAlphaOptions(ctx, options, limits.MaxMemoryBytes-result.working)
	if err != nil {
		return result, err
	}
	result.working += alphaOptionsWork
	if result.working >= limits.MaxMemoryBytes {
		return result, &LimitError{Resource: "separate alpha metadata memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(result.working, 1)}
	}
	for index, source := range [2]*Raster{&result.primary, result.alpha} {
		selected := mainOptions
		if index == 1 {
			selected = alphaOptions
			selected.Orientation = Orientation(result.layout[0].header.transform)
		}
		sourceWork, err := jxrRasterWork(source, limits.MaxMemoryBytes)
		if err != nil {
			return result, err
		}
		selected.Limits = limits
		selected.Limits.MaxMemoryBytes -= result.working - sourceWork
		layout, output, err := makeJXREncodingPlan(ctx, source, &selected)
		if err != nil {
			return result, jxrPlanLimitError(err, output.stream.maxMemory, sourceWork)
		}
		if index == 1 {
			prepareJXRSeparateAlphaPrimary(&layout, &output.stream, options.MirrorSeparateAlpha)
		}
		settingsWork, err := output.stream.tileWork(ctx, limits.MaxMemoryBytes-result.working)
		if err != nil {
			return result, err
		}
		for _, amount := range [2]uint64{layout.headerWork(), settingsWork} {
			if amount >= limits.MaxMemoryBytes-result.working {
				return result, &LimitError{Resource: "separate alpha plan memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(result.working, jxrSaturatedAdd(amount, 1))}
			}
			result.working += amount
		}
		result.layout[index], result.output[index] = layout, output.stream
	}
	for index, source := range [2]*Raster{&result.primary, result.alpha} {
		sourceWork, err := jxrRasterWork(source, limits.MaxMemoryBytes)
		if err != nil {
			return result, err
		}
		result.output[index].maxMemory = limits.MaxMemoryBytes - result.working + sourceWork
	}
	return result, ctx.Err()
}

// writeJXRSeparateFileAt 在指定目录位置写出真实双码流，将目录地址回填为全文件坐标且不重复非首图文件头
// 入参: ctx Context, w 当前可定位输出, raster 原生图像, options 当前图像已合并的选项, directory 当前目录位置，首图为八
// 返回: uint64 图像结束位置, uint64 当前目录链接位置, error 参数、资源、取消、定位或写入错误
func writeJXRSeparateFileAt(ctx context.Context, w io.WriteSeeker, raster *Raster, options EncodeOptions, directory uint64) (uint64, uint64, error) {
	if directory < 8 || directory&1 != 0 || directory > math.MaxUint32 {
		return 0, 0, fmt.Errorf("%w: incremental separate alpha directory position", ErrInvalid)
	}
	input, err := makeJXRSeparateInput(ctx, raster, options)
	if err != nil {
		return 0, 0, err
	}
	limits := options.Limits.normalized()
	file := jxrSeparateOutput{info: jxrWithContainerMetadata(raster.info, &options), layout: input.layout, maximum: input.output[0].maxBytes, prefixMemory: limits.MaxMemoryBytes - input.working, maxPixels: limits.MaxPixels}
	return writeJXRSeparatePacketsAt(ctx, w, file, directory, func(index int, maximum uint64) error {
		source := &input.primary
		if index == 1 {
			source = input.alpha
		}
		selected := input.output[index]
		selected.maxBytes = maximum
		return encodeJXRCodestreamTo(ctx, w, source, input.layout[index], selected)
	})
}

// writeJXRSeparatePacketsAt 共用双流文件前缀、偶数边界及三个字段的全文件回填，具体样本源由编码器提供
// 入参: ctx Context, w 当前可定位输出, file 双流布局和资源, directory 当前目录位置, encode 按序提交主图或透明度包的编码器
// 返回: uint64 图像结束位置, uint64 当前目录链接位置, error 元数据、资源、取消或I/O错误
func writeJXRSeparatePacketsAt(ctx context.Context, w io.WriteSeeker, file jxrSeparateOutput, directory uint64, encode func(int, uint64) error) (uint64, uint64, error) {
	if directory < 8 || directory&1 != 0 || directory > math.MaxUint32 || encode == nil {
		return 0, 0, fmt.Errorf("%w: incremental separate alpha directory position", ErrInvalid)
	}
	delta := directory - 8
	localMaximum := min(file.maximum, uint64(math.MaxUint32)-delta)
	maximum := delta + localMaximum
	header, offsets, err := makeJXRFilePrefixParts(ctx, file.info, &file.layout[0], &file.layout[1], file.maxPixels, localMaximum, file.prefixMemory)
	if err != nil {
		return 0, 0, err
	}
	link, err := relocateJXRImageDirectory(header, directory)
	if err != nil {
		return 0, 0, err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, 0, err
	}
	wanted := int64(directory)
	if directory == 8 {
		wanted = 0
	}
	if position != wanted {
		return 0, 0, fmt.Errorf("%w: incremental separate alpha output position", ErrInvalid)
	}
	start := delta + uint64(len(header))
	if start >= maximum {
		return 0, 0, &LimitError{Resource: "incremental separate alpha file capacity", Limit: maximum, Required: jxrSaturatedAdd(start, 1)}
	}
	if directory != 8 {
		header = header[8:]
	}
	if err := writeEncoderOutput(ctx, w, header); err != nil {
		return 0, 0, err
	}
	header = nil
	var starts, sizes [2]uint64
	for index := range 2 {
		if start >= maximum {
			return 0, 0, &LimitError{Resource: "incremental separate alpha stream capacity", Limit: maximum, Required: jxrSaturatedAdd(start, 1)}
		}
		starts[index] = start
		if err := encode(index, maximum-start); err != nil {
			return 0, 0, err
		}
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		end, err := w.Seek(0, io.SeekCurrent)
		if err != nil {
			return 0, 0, err
		}
		if end < 0 || uint64(end) < start || uint64(end) > maximum {
			return 0, 0, fmt.Errorf("%w: incremental separate alpha output extent", ErrInvalid)
		}
		sizes[index] = uint64(end) - start
		start = uint64(end)
		if index == 0 && start&1 != 0 {
			if start == maximum {
				return 0, 0, &LimitError{Resource: "incremental separate alpha alignment", Limit: maximum, Required: jxrSaturatedAdd(start, 1)}
			}
			if err := writeEncoderOutput(ctx, w, []byte{0}); err != nil {
				return 0, 0, err
			}
			start++
		}
	}
	for index, at := range [3]uint64{offsets.imageSize, offsets.alphaOffset, offsets.alphaSize} {
		if err := seekJXRImageOutput(ctx, w, delta+at); err != nil {
			return 0, 0, err
		}
		var field [4]byte
		binary.LittleEndian.PutUint32(field[:], uint32([3]uint64{sizes[0], starts[1], sizes[1]}[index]))
		if err := writeEncoderOutput(ctx, w, field[:]); err != nil {
			return 0, 0, err
		}
	}
	if err := seekJXRImageOutput(ctx, w, start); err != nil {
		return 0, 0, err
	}
	return start, link, ctx.Err()
}

// makeJXRSeparateTileInput 在常驻元数据预算后建立两个仅布局计划，浮点参数已由逐块入口显式校验
// 入参: ctx Context, info 已复制原生元数据, options 已校验文件选项, retained 原始元数据及ICC容量
// 返回: jxrSeparateTileInput 两个真实码流的布局, error 参数、能力、资源或取消错误
func makeJXRSeparateTileInput(ctx context.Context, info Info, options EncodeOptions, retained uint64) (jxrSeparateTileInput, error) {
	result := jxrSeparateTileInput{mirror: options.MirrorSeparateAlpha}
	if !info.Alpha || info.Channels < 1 || len(info.Components) != info.Channels+1 {
		return result, fmt.Errorf("%w: separate tile alpha source components", ErrInvalid)
	}
	limits := options.Limits.normalized()
	if retained >= limits.MaxMemoryBytes || 34816 >= limits.MaxMemoryBytes-retained {
		return result, &LimitError{Resource: "separate tile alpha metadata memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, 34817)}
	}
	working := retained + 34816
	result.info[0] = info
	result.info[0].Alpha, result.info[0].Premultiplied, result.info[0].PixelFormat = false, false, ""
	result.info[0].Components = result.info[0].Components[:info.Channels:info.Channels]
	result.info[1] = jxrAlphaRasterInfo(info.Bounds, info.Components[info.Channels])
	mainOptions := options
	mainOptions.Format, mainOptions.SeparateAlpha, mainOptions.MirrorSeparateAlpha, mainOptions.Alpha = FormatCodestream, false, false, nil
	alphaOptions, alphaOptionsWork, err := jxrSeparateAlphaOptions(ctx, options, limits.MaxMemoryBytes-working)
	if err != nil {
		return result, err
	}
	working += alphaOptionsWork
	if working >= limits.MaxMemoryBytes {
		return result, &LimitError{Resource: "separate tile alpha metadata memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(working, 1)}
	}
	for index := range 2 {
		selected := mainOptions
		if index == 1 {
			selected = alphaOptions
			selected.Orientation = Orientation(result.file.layout[0].header.transform)
		}
		selected.Limits = limits
		selected.Limits.MaxMemoryBytes -= working
		metadata := Raster{info: result.info[index]}
		layout, output, err := makeJXREncodingPlanRetained(ctx, &metadata, &selected, 0)
		if err != nil {
			return result, jxrPlanLimitError(err, limits.MaxMemoryBytes, working)
		}
		if index == 1 {
			prepareJXRSeparateAlphaPrimary(&layout, &output.stream, options.MirrorSeparateAlpha)
		}
		settingsWork, err := output.stream.tileWork(ctx, limits.MaxMemoryBytes-working)
		if err != nil {
			return result, err
		}
		for _, amount := range [2]uint64{layout.headerWork(), settingsWork} {
			if amount >= limits.MaxMemoryBytes-working {
				return result, &LimitError{Resource: "separate tile alpha plan memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(working, jxrSaturatedAdd(amount, 1))}
			}
			working += amount
		}
		result.file.layout[index], result.output[index] = layout, output.stream
	}
	for index := range result.output {
		result.output[index].maxMemory = limits.MaxMemoryBytes - working
	}
	result.file.info, result.file.maximum, result.file.maxPixels = info, result.output[0].maxBytes, limits.MaxPixels
	result.file.prefixMemory = limits.MaxMemoryBytes - working
	return result, ctx.Err()
}

// encodeJXRSeparateTileFileTo 将有界提供器分两遍读取为独立主图和透明度，复用双流目录提交而不分配完整原图
// 入参: ctx Context, w 起点为零的输出, info 独占原生元数据, options 已校验选项, retained 常驻描述容量, read 稳定逐块提供器
// 返回: error 元数据、布局、资源、提供器、取消或I/O错误
func encodeJXRSeparateTileFileTo(ctx context.Context, w io.WriteSeeker, info Info, options EncodeOptions, retained uint64, read RasterTileReader) error {
	input, err := makeJXRSeparateTileInput(ctx, info, options, retained)
	if err != nil {
		return err
	}
	_, _, err = writeJXRSeparateTileAt(ctx, w, &input, 8, read)
	return err
}

// writeJXRSeparateTileAt 复用已校验且独占的双流逐块计划，在任意合法目录位置提交两个真实码流
// 入参: ctx Context, w 当前可定位输出, input 双流元数据与计划, directory 当前目录位置, read 稳定提供器
// 返回: uint64 图像结束位置, uint64 当前链接位置, error 资源、提供器、取消或I/O错误
func writeJXRSeparateTileAt(ctx context.Context, w io.WriteSeeker, input *jxrSeparateTileInput, directory uint64, read RasterTileReader) (uint64, uint64, error) {
	return writeJXRSeparatePacketsAt(ctx, w, input.file, directory, func(index int, maximum uint64) error {
		selected := input.output[index]
		selected.maxBytes = maximum
		return encodeJXRProjectedTilesTo(ctx, w, input.file.info, input.info[index], &input.file.layout[index], selected, read, index == 1, input.mirror)
	})
}

// encodeJXRProjectedTilesTo 保留有界原始块供回调写入，再只复制当前码流需要的颜色或透明度
// 入参: ctx Context, w 当前码流输出, original 回调原生描述, coded 当前码流原生描述, source 完整头部, options 除全局元数据外的额度, read 稳定提供器, alpha 是否只提取透明度, mirror 是否向辅助主平面复制透明度
// 返回: error 原生网格、容量、样本、提供器、取消或I/O错误
func encodeJXRProjectedTilesTo(ctx context.Context, w io.WriteSeeker, original, coded Info, source *jxrStream, options jxrStreamOutput, read RasterTileReader, alpha, mirror bool) error {
	tiles := (len(source.header.columns) - 1) * (len(source.header.rows) - 1)
	var largest jxrHardTileInput
	for tile := range tiles {
		if err := ctx.Err(); err != nil {
			return err
		}
		plan, err := planJXRHardTile(original, source, tile)
		if err != nil {
			return err
		}
		if plan.rawWork > largest.rawWork {
			largest = plan
		}
	}
	if largest.rawWork >= options.maxMemory || jxrHardTileReserve >= options.maxMemory-largest.rawWork {
		return &LimitError{Resource: "separate tile alpha source working memory", Limit: options.maxMemory, Required: jxrSaturatedAdd(largest.rawWork, jxrSaturatedAdd(jxrHardTileReserve, 1))}
	}
	options.maxMemory -= largest.rawWork + jxrHardTileReserve
	var raw *Raster
	return encodeJXRHardTileSourceTo(ctx, w, coded, source, options, func(plan jxrHardTileInput, maximum uint64, local *Raster) (*Raster, error) {
		var err error
		if raw == nil {
			raw, err = resizeJXRHardTile(ctx, original, largest, largest.rawWork, nil)
			if err != nil {
				return nil, err
			}
		}
		originalPlan := plan
		originalPlan.rawWork = largest.rawWork
		raw, err = resizeJXRHardTile(ctx, original, originalPlan, largest.rawWork, raw)
		if err != nil {
			return nil, err
		}
		for _, plane := range raw.planes {
			clear(plane.values)
			clear(plane.words)
		}
		view := *raw
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := read(ctx, plan.region, &view); err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		local, err = resizeJXRHardTile(ctx, coded, plan, maximum, local)
		if err != nil {
			return nil, err
		}
		for _, plane := range local.planes {
			clear(plane.values)
			clear(plane.words)
		}
		for index, component := range local.info.Components {
			from := index
			if alpha {
				if index != coded.Channels && !mirror {
					continue
				}
				from = original.Channels
			}
			input, err := raw.component(from)
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
					bits, err := input.sampleBitsAt(x, y)
					if err != nil {
						return nil, err
					}
					local.planes[index].storeBits(y*component.Bounds.Dx()+x, bits)
				}
			}
		}
		return local, ctx.Err()
	})
}
