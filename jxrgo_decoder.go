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
	"image/color"
	"io"
	"math"
	"sync"
)

// Decoder 持有图像索引并串行执行解码，不关闭调用方输入流
// NewDecoder独立复制输入，NewDecoderAt引用调用方须保持稳定的随机输入
// 创建后不得复制，关闭后不再保留输入，可重复解码为独立图像
type Decoder struct {
	mu                       sync.Mutex
	data                     []byte
	source                   jxrRasterSource
	limits                   Limits
	info                     Info
	closed                   bool
	legacyHighpassPrediction bool
	workers                  int
	fileInput                *jxrReaderAt
	inputExtra               uint64
}

// decoderInput 在每次读取前后检查取消状态，不中断调用方已阻塞的Read
type decoderInput struct {
	ctx    context.Context
	source io.Reader
}

// jxrRasterSource 保存已校验容器与各独立码流，不重写不一致的透明度标志
type jxrRasterSource struct {
	container jxrContainer
	primary   jxrStream
	alpha     *jxrStream
	format    Format
}

// NewDecoder 读取有界输入并检查容器文件或独立码流的头部，不重建像素
// 返回前预留常驻输入、索引和一份Info副本的总容量，不计调用方继续持有的既往返回值
// 入参: ctx Context，nil采用Background, r 输入流, opts 解码选项，nil采用默认上限
// 返回: *Decoder 解码器, error 输入、结构、能力、资源或取消错误
func NewDecoder(ctx context.Context, r io.Reader, opts *DecodeOptions) (*Decoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	decoder, err := newDecoder(ctx, r, opts)
	if err != nil {
		return nil, wrapJXRError("NewDecoder", err)
	}
	return decoder, nil
}

// newDecoder 执行NewDecoder的核心读取与头部解析逻辑，由公开入口统一归一化错误
// 入参: ctx 已规范化Context, r 输入流, opts 解码选项
// 返回: *Decoder 解码器, error 输入、结构、能力、资源或取消错误
func newDecoder(ctx context.Context, r io.Reader, opts *DecodeOptions) (*Decoder, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil {
		return nil, fmt.Errorf("%w: nil input", ErrInvalid)
	}
	limits := DefaultLimits()
	workers := 0
	legacyHighpass := false
	if opts != nil {
		if opts.Workers < 0 {
			return nil, fmt.Errorf("%w: negative decode workers", ErrInvalid)
		}
		limits = opts.Limits.normalized()
		workers = opts.Workers
		legacyHighpass = opts.LegacyHighpassPrediction
	}
	data, err := readDecoderInput(&decoderInput{ctx: ctx, source: r}, limits)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(cap(data)) > limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: uint64(cap(data)), Detail: "decoder retained input memory"}
	}
	parser := jxrParser{ctx: ctx, limit: limits.MaxMemoryBytes - uint64(cap(data)), legacyHighpassPrediction: legacyHighpass}
	source, err := parser.rasterSource(data, limits.MaxPixels)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := parser.info(&source)
	if err != nil {
		return nil, err
	}
	decoder := &Decoder{data: data, source: source, limits: limits, info: info, legacyHighpassPrediction: legacyHighpass}
	decoder.setWorkers(workers)
	if err := decoder.checkInfoCopy(); err != nil {
		return nil, err
	}
	return decoder, nil
}

// setWorkers 将规范化后的并发上限同步至解码器及其内部图像源
// 入参: requested 调用方请求的并发上限
func (d *Decoder) setWorkers(requested int) {
	workers := normalizeJXRWorkers(requested)
	d.workers = workers
	d.source.setWorkers(workers)
}

// Info 返回方向变换后的图像信息副本，关闭后或接收者为nil时返回零值
// 构造时已检查常驻存储与单份副本的预算，调用方持有的既往副本不计入后续操作预算
// 返回: Info 图像信息
func (d *Decoder) Info() Info {
	if d == nil {
		return Info{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return cloneInfo(d.info)
}

// checkInfoCopy 检查常驻输入、头部及单份公开元数据副本的总活动容量，不分配副本
// 返回: error 容量不足时返回资源错误
func (d *Decoder) checkInfoCopy() error {
	retained := d.inputWork() + d.retainedHeaderWork()
	work := uint64(len(d.info.Components))*128 + uint64(len(d.info.ICCProfile))
	if d.info.ColorEncoding != nil {
		work += 4
	}
	if retained > d.limits.MaxMemoryBytes || work > d.limits.MaxMemoryBytes-retained {
		return &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, work), Detail: "JPEG XR metadata copy memory"}
	}
	return nil
}

// Decode 在工作预算内重建原生分量，不进行ICC转换、解除预乘或显示范围裁剪
// 各解码阶段响应取消且取消后可重新解码，按工作内存预算自动选择整帧或行流重建
// 入参: ctx Context，nil采用Background
// 返回: *Raster 原生分量图像, error 数据、能力、资源、关闭或取消错误
func (d *Decoder) Decode(ctx context.Context) (*Raster, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	raster, err := d.decodeLocked(ctx, 0)
	if err != nil {
		return nil, wrapJXRError("Decoder.Decode", err)
	}
	return raster, nil
}

// decodeReserved 在工作预算内预留额外输出容量后重建首图，不修改解码器自身上限
// 入参: ctx 已规范化Context, extra 额外预留字节数
// 返回: *Raster 原生分量图像, error 数据、能力、资源、关闭或取消错误
func (d *Decoder) decodeReserved(ctx context.Context, extra uint64) (*Raster, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.decodeLocked(ctx, extra)
}

// decodeLocked 在调用方持有解码器锁时重建首图，供显式图像选择与显示路径共用相同资源规则
// 入参: ctx 已规范化Context, extra 额外预留字节数
// 返回: *Raster 原生图像, error 结构、资源、取消或关闭错误
func (d *Decoder) decodeLocked(ctx context.Context, extra uint64) (*Raster, error) {
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
	if extra >= d.limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(extra, 1), Detail: "decoding memory budget"}
	}
	memoryLimit := d.limits.MaxMemoryBytes - extra
	stride := d.info.Channels
	if d.info.Alpha {
		stride++
	}
	width := d.source.sampleBytes()
	pixels := uint64(d.info.Bounds.Dx()) * uint64(d.info.Bounds.Dy())
	retained := d.inputWork() + uint64(len(d.info.ICCProfile)) + uint64(cap(d.source.container.profiles))*4
	if d.info.ColorEncoding != nil {
		retained += 4
	}
	full, ok := jxrWorkTotal(^uint64(0), retained, d.source.decodeWork())
	if !ok {
		full = ^uint64(0)
	}
	plan, output, err := d.planDecodeRows(ctx, full, extra)
	if err != nil {
		return nil, err
	}
	if output != 0 {
		return d.decodeRows(ctx, plan, output)
	}
	if retained > memoryLimit || d.source.decodeWork() > memoryLimit-retained {
		return nil, &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(extra, jxrSaturatedAdd(retained, d.source.decodeWork())), Detail: "decoding memory budget"}
	}
	units := (uint64(stride)*width + 7) / 8
	if pixels > ^uint64(0)/units {
		return nil, &LimitError{Resource: "max pixels", Limit: ^uint64(0) / units, Required: pixels, Detail: "output capacity"}
	}
	raster, err := d.source.decodeContext(ctx, pixels*units, memoryLimit-retained)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	raster.info = cloneInfo(d.info)
	return raster, nil
}

// Close 释放内部输入及索引，可重复调用，接收者为nil时直接返回nil，不影响已返回的图像
// 返回: error 错误信息
func (d *Decoder) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.data, d.source, d.info, d.closed = nil, jxrRasterSource{}, Info{}, true
	d.fileInput, d.inputExtra = nil, 0
	return nil
}

// Read 读取前后响应取消，保留非EOF底层错误及原始读取计数
// 入参: data 目标缓冲区
// 返回: int 已读字节数, error 读取或取消错误
func (r *decoderInput) Read(data []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	n, err := r.source.Read(data)
	if err == nil || err == io.EOF {
		if canceled := r.ctx.Err(); canceled != nil {
			return n, canceled
		}
	}
	return n, err
}

// readDecoderInput 在增长缓冲区前检查新旧容量及读取缓冲的总内存，不无限等待无进展输入
// 入参: source 输入流, limits 输入及内存上限
// 返回: []byte 独立输入数据, error 读取、资源或无进展错误
func readDecoderInput(source io.Reader, limits Limits) ([]byte, error) {
	limit := min(limits.MaxInputBytes, uint64(^uint(0)>>1)-1)
	bufferSize := min(uint64(32<<10), limit+1, limits.MaxMemoryBytes/2)
	if bufferSize == 0 {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: 2, Detail: "input buffer memory"}
	}
	buffer := make([]byte, int(bufferSize))
	var data []byte
	idle := 0
	for {
		remaining := limit - uint64(len(data))
		requested := buffer[:min(uint64(len(buffer)), remaining+1)]
		n, err := source.Read(requested)
		if n < 0 || n > len(requested) {
			return nil, fmt.Errorf("%w: input read count", ErrInvalid)
		}
		if err != nil && err != io.EOF {
			return nil, err
		}
		if uint64(n) > remaining {
			return nil, &LimitError{Resource: "max input bytes", Limit: limit, Required: uint64(len(data)) + uint64(n), Detail: "input byte budget"}
		}
		if n != 0 {
			idle = 0
			required := uint64(len(data) + n)
			if required > uint64(cap(data)) {
				capacity := max(required, min(limit, max(uint64(512), uint64(cap(data))*2)))
				available := limits.MaxMemoryBytes - bufferSize - uint64(cap(data))
				if capacity > available {
					capacity = required
				}
				if capacity > available {
					return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(jxrSaturatedAdd(bufferSize, uint64(cap(data))), required), Detail: "input growth memory"}
				}
				next := make([]byte, len(data), int(capacity))
				copy(next, data)
				data = next
			}
			data = append(data, buffer[:n]...)
		} else {
			idle++
		}
		if err == io.EOF {
			return data, nil
		}
		if idle >= 100 {
			return nil, io.ErrNoProgress
		}
	}
}

// setWorkers 将并发上限同步至主码流及可选独立透明度码流
// 入参: workers 已规范化的并发上限
func (source *jxrRasterSource) setWorkers(workers int) {
	source.primary.workers = workers
	if source.alpha != nil {
		source.alpha.workers = workers
	}
}

// rasterSource 在共享预算内检查容器尺寸、像素格式、频带及透明度布局
// 入参: data 完整JPEG XR文件, maxPixels 输出像素预算
// 返回: jxrRasterSource 已绑定图像源, error 数据、能力或资源错误
func (parser *jxrParser) rasterSource(data []byte, maxPixels uint64) (jxrRasterSource, error) {
	var source jxrRasterSource
	var err error
	if len(data) >= 8 && string(data[:8]) == "WMPHOTO\x00" {
		return parser.codedSource(data, maxPixels)
	}
	source.format = FormatJXR
	source.container, err = parser.container(data, maxPixels)
	if err != nil {
		return source, err
	}
	return parser.containerSource(&source.container, maxPixels)
}

// containerSource 校验容器及独立码流绑定，共用主流和独立透明度的解析预算
// 入参: container 图像容器, maxPixels 输出像素预算
// 返回: jxrRasterSource 已绑定图像源, error 数据、能力或资源错误
func (parser *jxrParser) containerSource(container *jxrContainer, maxPixels uint64) (jxrRasterSource, error) {
	source := jxrRasterSource{container: *container, format: FormatJXR}
	var err error
	c := &source.container
	source.primary, err = parser.streamView(c.image, c.imageInput, maxPixels)
	if err != nil {
		return source, err
	}
	return parser.bindContainerSource(source, maxPixels)
}

// bindContainerSource 校验已经定位的主码流与容器的对应关系，独立透明度仍按原语法读取
// 入参: source 含已解析主码流的容器, maxPixels 输出像素预算
// 返回: jxrRasterSource 已绑定图像源, error 数据、能力或资源错误
func (parser *jxrParser) bindContainerSource(source jxrRasterSource, maxPixels uint64) (jxrRasterSource, error) {
	c := &source.container
	s := &source.primary
	h := &s.header
	depth := c.format.depth
	if c.format.name == "BlackWhite" && h.depth == 15 {
		depth = 15
	}
	channels := s.primary.channels
	switch h.color {
	case 0:
		channels = 1
	case 1, 2, 3, 7, 8:
		channels = 3
	case 4, 5:
		channels = 4
	}
	if h.width != uint64(c.width) || h.height != uint64(c.height) || h.color != c.format.color || h.depth != depth || channels != c.format.channels ||
		c.missingBandTags&jxrMissingImageBandTag == 0 && s.primary.bands != uint8(c.imageBands) {
		return source, fmt.Errorf("%w: JPEG XR container and primary stream", ErrInvalid)
	}
	c.format.depth = depth
	c.imageBands = uint32(s.primary.bands)
	if !c.separateAlpha() {
		if c.format.alpha != h.alpha || c.format.alpha != (s.alpha != nil) || h.premultiply && !c.format.premultiply {
			return source, fmt.Errorf("%w: JPEG XR interleaved alpha layout", ErrInvalid)
		}
	} else {
		if h.alpha || !c.format.alpha {
			return source, fmt.Errorf("%w: JPEG XR separate primary alpha layout", ErrInvalid)
		}
		alpha, err := parser.streamView(c.alpha, c.alphaInput, maxPixels)
		if err != nil {
			return source, err
		}
		if !alpha.header.alpha || alpha.alpha == nil || alpha.primary.channels != 1 || alpha.header.width != h.width || alpha.header.height != h.height ||
			alpha.header.depth != h.depth || c.missingBandTags&jxrMissingAlphaBandTag == 0 && alpha.alpha.bands != uint8(c.alphaBands) ||
			alpha.header.premultiply && !c.format.premultiply {
			return source, fmt.Errorf("%w: JPEG XR separate alpha stream", ErrInvalid)
		}
		c.alphaBands = uint32(alpha.alpha.bands)
		source.alpha = &alpha
	}
	if c.format.channels > 8 {
		return source, fmt.Errorf("%w: JPEG XR raster color format", ErrUnsupported)
	}
	return source, nil
}

// reconstructContext 执行主平面及透明度的两级重建，不分配整幅中间样本图像
// 入参: ctx Context
// 返回: []jxrComponentView 主平面视图, *jxrComponentView 透明度视图, error 数值、数据或取消错误
func (c *jxrCoefficientImage) reconstructContext(ctx context.Context) ([]jxrComponentView, *jxrComponentView, error) {
	primary, err := c.primary.reconstructContext(ctx, &c.header)
	if err != nil {
		return nil, nil, err
	}
	if c.alpha == nil {
		return primary, nil, nil
	}
	alpha, err := c.alpha.reconstructContext(ctx, &c.header)
	if err != nil {
		return nil, nil, err
	}
	return primary, &alpha[0], nil
}

// reconstructContext 按标准顺序执行两级逆变换及重叠滤波，各阶段响应取消
// 入参: ctx Context, h 当前图像头
// 返回: []jxrComponentView 原地重建样本视图, error 数值、数据或取消错误
func (c *jxrCoefficients) reconstructContext(ctx context.Context, h *jxrHeader) ([]jxrComponentView, error) {
	if _, err := c.inverseFirstContext(ctx, h); err != nil {
		return nil, err
	}
	if h.overlap == 2 {
		if err := c.filterGrid(ctx, h, false, false); err != nil {
			return nil, err
		}
	}
	views, err := c.inverseSecondContext(ctx, h)
	if err != nil {
		return nil, err
	}
	if h.overlap != 0 {
		if err := c.filterGrid(ctx, h, true, false); err != nil {
			return nil, err
		}
	}
	return views, nil
}

// decodeWork 计算各独立码流的重建工作及原生输出容量，溢出时返回最大值
// 返回: uint64 不包含输入及调用方保留图像的工作字节数
func (source *jxrRasterSource) decodeWork() uint64 {
	count, width := source.sampleCount(), source.sampleBytes()
	if width == 0 || count > ^uint64(0)/width {
		return ^uint64(0)
	}
	channels := max(3, source.primary.primary.channels, source.container.format.channels)
	if channels > rasterMaxComponents || channels < 1 {
		return ^uint64(0)
	}
	auxiliary := uint64(channels)*12 + 256
	if source.container.format.color == 1 || source.container.format.color == 2 {
		auxiliary = 0
	}
	if count*width > ^uint64(0)-auxiliary {
		return ^uint64(0)
	}
	work := count*width + auxiliary
	for _, stream := range [2]*jxrStream{&source.primary, source.alpha} {
		if stream != nil {
			amount := stream.coefficientWork()
			if amount > ^uint64(0)-work {
				return ^uint64(0)
			}
			work += amount
		}
	}
	return work
}

// decodeContext 从已校验索引重建像素，复用头部及分块索引并响应取消
// 入参: ctx Context, maxPixels 剩余RGBA64等价输出容量, maxWork 重建及原生输出的工作字节上限
// 返回: *Raster 原生分量图像, error 重建、数据、资源或取消错误
func (source *jxrRasterSource) decodeContext(ctx context.Context, maxPixels, maxWork uint64) (*Raster, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := &source.container
	stride := c.format.channels
	if c.format.alpha {
		stride++
	}
	count := uint64(c.width) * uint64(c.height)
	bytesPerSample := source.sampleBytes()
	units := (uint64(stride)*bytesPerSample + 7) / 8
	if count > maxPixels/units {
		return nil, &LimitError{Resource: "max pixels", Limit: maxPixels / units, Required: count, Detail: "JPEG XR output storage"}
	}
	if count > uint64(^uint(0)>>1)/(bytesPerSample*uint64(stride)) {
		return nil, &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: jxrSaturatedMul(count, bytesPerSample*uint64(stride)), Detail: "JPEG XR output storage"}
	}
	budgetLimit := min(maxWork, uint64(^uint(0)>>1))
	decodeWork := source.decodeWork()
	if decodeWork > budgetLimit {
		return nil, &LimitError{Resource: "max memory bytes", Limit: budgetLimit, Required: decodeWork, Detail: "JPEG XR combined decoding storage"}
	}
	slack := budgetLimit - decodeWork
	state, err := source.primary.coefficientsContext(ctx, c.image, jxrWorkerBudgetAllowance(slack, source.primary.coefficientWork()))
	if err != nil {
		return nil, err
	}
	views, alphaView, err := state.reconstructContext(ctx)
	if err != nil {
		return nil, err
	}
	alphaHeader, alphaPlane, embedded := &state.header, state.alpha, true
	if source.alpha != nil {
		alphaState, err := source.alpha.coefficientsContext(ctx, c.alpha, jxrWorkerBudgetAllowance(slack, source.alpha.coefficientWork()))
		if err != nil {
			return nil, err
		}
		opacity, interleaved, err := alphaState.reconstructContext(ctx)
		if err != nil {
			return nil, err
		}
		alphaHeader, alphaPlane, alphaView = &alphaState.header, alphaState.alpha, interleaved
		embedded = alphaPlane != nil
		if !embedded {
			alphaPlane, alphaView = &alphaState.primary, &opacity[0]
		}
	}
	if c.format.color == 1 || c.format.color == 2 {
		return source.decodeSubsampled(ctx, state, views, alphaView, alphaHeader, alphaPlane, maxPixels)
	}
	width, height := int(c.width), int(c.height)
	if c.transform&4 != 0 {
		width, height = height, width
	}
	bounds := image.Rect(0, 0, width, height)
	pixels, err := newImageComponents(bounds, c.format.channels, c.format.alpha, bytesPerSample, maxPixels)
	if err != nil {
		return nil, err
	}
	opacityNumber := source.opacityFormat().number
	if done, err := source.decodeUnsignedFast(ctx, state, views, alphaView, alphaHeader, alphaPlane, embedded, opacityNumber, pixels); err != nil {
		return nil, err
	} else if done {
		return &Raster{native: pixels, info: source.info()}, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	channels := max(3, state.primary.plane.channels, c.format.channels)
	values, components := make([]int64, channels), make([]uint32, channels)
	h, p := &state.header, &state.primary.plane
	if err := checkJXRPlaneComponents(p); err != nil {
		return nil, err
	}
	if len(views) != p.channels || len(values) < p.channels {
		return nil, fmt.Errorf("%w: JPEG XR sample component storage", ErrInvalid)
	}
	upsample := (p.color == 1 || p.color == 2) && (h.color == 3 || h.color == 7)
	left, top := int(h.left), int(h.top)
	for y := range int(c.height) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sy := y + top
		for x := range int(c.width) {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			sx := x + left
			if !upsample {
				for i := range views {
					values[i] = views[i].at(sx, sy)
				}
			} else {
				values[0] = views[0].at(sx, sy)
				for i := 1; i < len(views); i++ {
					values[i] = sampleJXRChroma(&views[i], p, i, sx, sy)
				}
			}
			n, err := jxrRasterBitsComponents(h, p, c.format.number, values, false, components)
			if err != nil {
				return nil, err
			}
			if n != c.format.channels {
				return nil, fmt.Errorf("%w: JPEG XR output component count", ErrInvalid)
			}
			alpha := uint32(0)
			if alphaView != nil {
				sample := [1]int64{alphaView.at(x+int(alphaHeader.left), y+int(alphaHeader.top))}
				var codes [1]uint32
				_, err := jxrRasterBitsComponents(alphaHeader, &alphaPlane.plane, opacityNumber, sample[:], embedded, codes[:])
				if err != nil {
					return nil, err
				}
				alpha = codes[0]
			}
			px, py := jxrRasterPosition(x, y, int(c.width), int(c.height), uint8(c.transform))
			at := (py*width + px) * stride
			for i := range n {
				pixels.storeBits(at+i, components[i])
			}
			if pixels.alpha {
				pixels.storeBits(at+n, alpha)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &Raster{native: pixels, info: source.info()}, nil
}

// decodeUnsignedFast 对无旋转无子采样的常规八位与十六位无符号整数格式直接重建像素存储
// 入参: ctx Context, state 主码流重建状态, views 主平面样本视图, alphaView 透明度样本视图, alphaHeader 透明度图像头, alphaPlane 透明度平面, embedded 是否内嵌透明度, opacityNumber 透明度数值格式, pixels 目标像素存储
// 返回: bool 是否已由快路径完成, error 数值范围或取消错误
func (source *jxrRasterSource) decodeUnsignedFast(ctx context.Context, state *jxrCoefficientImage, views []jxrComponentView, alphaView *jxrComponentView, alphaHeader *jxrHeader, alphaPlane *jxrCoefficients, embedded bool, opacityNumber uint8, pixels *imageComponents) (bool, error) {
	c := &source.container
	h, p := &state.header, &state.primary.plane
	if c.format.number != jxrUnsigned || c.transform != 0 || pixels.words != nil {
		return false, nil
	}
	if h.depth != 1 && (h.depth != 2 || p.shift != 0) {
		return false, nil
	}
	switch h.color {
	case 0:
		if p.color != 0 || p.channels != 1 || c.format.channels != 1 {
			return false, nil
		}
	case 3, 7:
		if p.color != 3 || p.channels != 3 || c.format.channels != 3 {
			return false, nil
		}
	case 4, 5:
		if p.color != 4 || p.channels != 4 || c.format.channels != 4 {
			return false, nil
		}
	default:
		return false, nil
	}
	w, ht := int(c.width), int(c.height)
	left, top := int(h.left), int(h.top)
	if w <= 0 || ht <= 0 || left < 0 || top < 0 || len(views) != p.channels || len(views[0].data) == 0 {
		return false, nil
	}
	vStride, vCols := views[0].stride, views[0].columns
	if left+w > views[0].width || top+ht > views[0].height {
		return false, nil
	}
	for i := range views {
		v := &views[i]
		if len(v.data) != len(views[0].data) || &v.data[0] != &views[0].data[0] || v.xShift != 4 || v.yShift != 4 || v.micro != 4 || v.stride != vStride || v.columns != vCols || v.offset != i*256 {
			return false, nil
		}
	}
	data := views[0].data
	maxIdx := ((top+ht-1)>>4*vCols+(left+w-1)>>4)*vStride + (p.channels-1)*256 + 255
	if maxIdx >= len(data) {
		return false, nil
	}
	stride := c.format.channels
	if pixels.alpha {
		stride++
	}
	if len(pixels.values) < w*ht*stride || pixels.alpha != (alphaView != nil) {
		return false, nil
	}
	var aData []int64
	var aStride, aCols, aOffset, aLeft, aTop int
	var aOff, aRound, aHigh int64
	var aScale uint8
	if alphaView != nil {
		if opacityNumber != jxrUnsigned || alphaPlane == nil || alphaPlane.plane.color != 0 || alphaPlane.plane.channels != 1 || !embedded && alphaHeader.color != 0 {
			return false, nil
		}
		if alphaHeader.depth != 1 && (alphaHeader.depth != 2 || alphaPlane.plane.shift != 0) {
			return false, nil
		}
		aLeft, aTop = int(alphaHeader.left), int(alphaHeader.top)
		if len(alphaView.data) == 0 || alphaView.xShift != 4 || alphaView.yShift != 4 || alphaView.micro != 4 || aLeft < 0 || aTop < 0 || aLeft+w > alphaView.width || aTop+ht > alphaView.height {
			return false, nil
		}
		aData, aStride, aCols, aOffset = alphaView.data, alphaView.stride, alphaView.columns, alphaView.offset
		aMaxIdx := ((aTop+ht-1)>>4*aCols+(aLeft+w-1)>>4)*aStride + aOffset + 255
		if aMaxIdx >= len(aData) {
			return false, nil
		}
		_ = aData[aMaxIdx]
		aBias := int64(128)
		aHigh = 255
		if alphaHeader.depth == 2 {
			aBias, aHigh = 32768, 65535
			if alphaPlane.plane.scaled {
				aScale, aRound = 3, 4
			}
		} else if alphaPlane.plane.scaled {
			aScale, aRound = 3, 3
		}
		aOff = aBias << aScale
	}
	bias, high := int64(128), int64(255)
	scale, round := uint8(0), int64(0)
	if h.depth == 2 {
		bias, high = 32768, 65535
		if p.scaled {
			scale, round = 3, 4
		}
	} else if p.scaled {
		scale, round = 3, 3
	}
	off := bias << scale
	halfOff := (bias >> 1) << scale
	ch := c.format.channels
	dst := pixels.values
	_ = data[maxIdx]
	for y := range ht {
		if err := ctx.Err(); err != nil {
			return true, err
		}
		sy := y + top
		rowBase := (sy>>4)*vCols*vStride + (sy&12)<<4 + (sy&3)<<2
		aRowBase := 0
		if alphaView != nil {
			ay := y + aTop
			aRowBase = (ay>>4)*aCols*aStride + aOffset + (ay&12)<<4 + (ay&3)<<2
		}
		outRow := dst[y*w*stride : (y+1)*w*stride]
		for start := 0; start < w; start += 1024 {
			if err := ctx.Err(); err != nil {
				return true, err
			}
			end := min(start+1024, w)
			n := end - start
			out := outRow[start*stride : end*stride]
			_ = out[n*stride-1]
			xBase := start + left
			switch h.color {
			case 0:
				for i := range n {
					sx := xBase + i
					v0 := data[rowBase+(sx>>4)*vStride+(sx&12)<<2+(sx&3)]
					y0 := v0 + off
					if ((uint64(v0)+jxrInt32Bias)|(uint64(y0)+jxrInt32Bias))>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					out[i*stride] = uint16(min(high, max(0, (y0+round)>>scale)))
				}
			case 7:
				for i := range n {
					sx := xBase + i
					idx := rowBase + (sx>>4)*vStride + (sx&12)<<2 + (sx & 3)
					v0, v1, v2 := data[idx], data[idx+256], data[idx+512]
					t := -v1
					g := v0 - (t >> 1)
					r := t + g - ((v2 + 1) >> 1)
					b := v2 + r
					r0, g0, b0 := r+off, g+off, b+off
					ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) |
						(uint64(t) + jxrInt32Bias) | (uint64(g) + jxrInt32Bias) | (uint64(r) + jxrInt32Bias) | (uint64(b) + jxrInt32Bias) |
						(uint64(r0) + jxrInt32Bias) | (uint64(g0) + jxrInt32Bias) | (uint64(b0) + jxrInt32Bias)
					if ov>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					o := i * stride
					out[o] = uint16(min(high, max(0, (r0+round)>>scale)))
					out[o+1] = uint16(min(high, max(0, (g0+round)>>scale)))
					out[o+2] = uint16(min(high, max(0, (b0+round)>>scale)))
				}
			case 3:
				for i := range n {
					sx := xBase + i
					idx := rowBase + (sx>>4)*vStride + (sx&12)<<2 + (sx & 3)
					v0, v1, v2 := data[idx], data[idx+256], data[idx+512]
					y0, u0, w0 := v0+off, v1+off, v2+off
					ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) |
						(uint64(y0) + jxrInt32Bias) | (uint64(u0) + jxrInt32Bias) | (uint64(w0) + jxrInt32Bias)
					if ov>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					o := i * stride
					out[o] = uint16(min(high, max(0, (y0+round)>>scale)))
					out[o+1] = uint16(min(high, max(0, (u0+round)>>scale)))
					out[o+2] = uint16(min(high, max(0, (w0+round)>>scale)))
				}
			case 4:
				for i := range n {
					sx := xBase + i
					idx := rowBase + (sx>>4)*vStride + (sx&12)<<2 + (sx & 3)
					v0, v1, v2, v3 := data[idx], data[idx+256], data[idx+512], data[idx+768]
					k := v3 + (v0 >> 1)
					m := k - v0 - (v1 >> 1)
					cv := v1 + m + (v2 >> 1)
					yv := cv - v2
					c0, m0, y0, k0 := cv+halfOff, m+halfOff, yv+halfOff, k-halfOff
					ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) | (uint64(v3) + jxrInt32Bias) |
						(uint64(k) + jxrInt32Bias) | (uint64(m) + jxrInt32Bias) | (uint64(cv) + jxrInt32Bias) | (uint64(yv) + jxrInt32Bias) |
						(uint64(c0) + jxrInt32Bias) | (uint64(m0) + jxrInt32Bias) | (uint64(y0) + jxrInt32Bias) | (uint64(k0) + jxrInt32Bias)
					if ov>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					o := i * stride
					out[o] = uint16(min(high, max(0, (c0+round)>>scale)))
					out[o+1] = uint16(min(high, max(0, (m0+round)>>scale)))
					out[o+2] = uint16(min(high, max(0, (y0+round)>>scale)))
					out[o+3] = uint16(min(high, max(0, (k0+round)>>scale)))
				}
			case 5:
				for i := range n {
					sx := xBase + i
					idx := rowBase + (sx>>4)*vStride + (sx&12)<<2 + (sx & 3)
					v0, v1, v2, v3 := data[idx], data[idx+256], data[idx+512], data[idx+768]
					c0, m0, y0, k0 := v1+off, v2+off, v3+off, v0+off
					ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) | (uint64(v3) + jxrInt32Bias) |
						(uint64(c0) + jxrInt32Bias) | (uint64(m0) + jxrInt32Bias) | (uint64(y0) + jxrInt32Bias) | (uint64(k0) + jxrInt32Bias)
					if ov>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					o := i * stride
					out[o] = uint16(min(high, max(0, (c0+round)>>scale)))
					out[o+1] = uint16(min(high, max(0, (m0+round)>>scale)))
					out[o+2] = uint16(min(high, max(0, (y0+round)>>scale)))
					out[o+3] = uint16(min(high, max(0, (k0+round)>>scale)))
				}
			}
			if alphaView != nil {
				axBase := start + aLeft
				for i := range n {
					ax := axBase + i
					av := aData[aRowBase+(ax>>4)*aStride+(ax&12)<<2+(ax&3)]
					a0 := av + aOff
					if ((uint64(av)+jxrInt32Bias)|(uint64(a0)+jxrInt32Bias))>>32 != 0 {
						return true, fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					}
					out[i*stride+ch] = uint16(min(aHigh, max(0, (a0+aRound)>>aScale)))
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return true, err
	}
	return true, nil
}

// jxrRasterSampleComponents 从扩展样本视图取得可见像素，色度在输出格式要求时按示例滤波器上采样
// 入参: views 主平面分量视图, h 图像头, p 主平面参数, x, y 可见坐标, values 颜色转换前的样本目标
// 返回: error 分量或目标容量错误
func jxrRasterSampleComponents(views []jxrComponentView, h *jxrHeader, p *jxrPlane, x, y int, values []int64) error {
	switch p.color {
	case 0:
		if p.channels == 1 && len(views) == 1 && len(values) >= 1 {
			values[0] = views[0].at(x+int(h.left), y+int(h.top))
			return nil
		}
	case 3:
		if p.channels == 3 && len(views) == 3 && len(values) >= 3 {
			sx, sy := x+int(h.left), y+int(h.top)
			values[0] = views[0].at(sx, sy)
			values[1] = views[1].at(sx, sy)
			values[2] = views[2].at(sx, sy)
			return nil
		}
	case 4:
		if p.channels == 4 && len(views) == 4 && len(values) >= 4 {
			sx, sy := x+int(h.left), y+int(h.top)
			values[0] = views[0].at(sx, sy)
			values[1] = views[1].at(sx, sy)
			values[2] = views[2].at(sx, sy)
			values[3] = views[3].at(sx, sy)
			return nil
		}
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if len(views) != p.channels || len(values) < p.channels {
		return fmt.Errorf("%w: JPEG XR sample component storage", ErrInvalid)
	}
	x, y = x+int(h.left), y+int(h.top)
	upsample := (p.color == 1 || p.color == 2) && (h.color == 3 || h.color == 7)
	if !upsample {
		for i := range views {
			values[i] = views[i].at(x, y)
		}
		return nil
	}
	values[0] = views[0].at(x, y)
	for i := 1; i < len(views); i++ {
		values[i] = sampleJXRChroma(&views[i], p, i, x, y)
	}
	return nil
}

// jxrHalfValue 按IEEE半精度位模式取得精确数值，保留符号零、非规格化数和非有限数
// 入参: bits 半精度位模式
// 返回: float64 对应数值
func jxrHalfValue(bits uint16) float64 {
	sign := uint64(bits>>15) << 63
	exponent, fraction := bits>>10&31, bits&1023
	if exponent == 31 {
		return math.Float64frombits(sign | 0x7ff0000000000000 | uint64(fraction)<<42)
	}
	value := math.Ldexp(float64(fraction), -24)
	if exponent != 0 {
		value = math.Ldexp(float64(1024+fraction), int(exponent)-25)
	}
	if sign != 0 {
		value = -value
	}
	return value
}

// jxrRasterPosition 按方向枚举映射坐标，文件容器中的从属方向不参与变换
// 入参: x, y 可见源坐标, width, height 源尺寸, transform 容器方向
// 返回: int 目标横坐标, int 目标纵坐标
func jxrRasterPosition(x, y, width, height int, transform uint8) (int, int) {
	if transform&4 != 0 {
		x, y, width, height = height-1-y, x, height, width
	}
	if transform&2 != 0 {
		x = width - 1 - x
	}
	if transform&1 != 0 {
		y = height - 1 - y
	}
	return x, y
}

// init 向标准image包注册JPEG XR解码器
func init() {
	image.RegisterFormat("jxr", "II\xbc\x01", Decode, DecodeConfig)
	image.RegisterFormat("jxr", "WMPHOTO\x00", Decode, DecodeConfig)
}

// Decode 解码为标准图像，扩展范围及多通道图像应通过Decoder.Decode读取原生分量
// 入参: r 输入流
// 返回: image.Image 标准图像, error 输入、解码或颜色解释错误
func Decode(r io.Reader) (image.Image, error) {
	img, err := decodeContextImage(context.Background(), r, nil)
	if err != nil {
		return nil, wrapJXRError("Decode", err)
	}
	return img, nil
}

// DecodeContext 在资源上限内解码并转换为非预乘十六位sRGB图像，不关闭调用方输入流
// 支持CICP、文件颜色提示及未配置ICC的常规CMYK转换，BT.2100 PQ与HLG按BT.2446方法A映射到100 cd/m²
// 入参: ctx Context，nil采用Background, r 输入流, opts 解码选项，nil采用默认上限
// 返回: image.Image 非预乘十六位标准图像, error 输入、颜色、资源或取消错误
func DecodeContext(ctx context.Context, r io.Reader, opts *DecodeOptions) (image.Image, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	img, err := decodeContextImage(ctx, r, opts)
	if err != nil {
		return nil, wrapJXRError("DecodeContext", err)
	}
	return img, nil
}

// decodeContextImage 执行标准sRGB图像解码及显示映射，不修改解码器资源上限
// 入参: ctx 已规范化Context, r 输入流, opts 解码选项
// 返回: image.Image 非预乘十六位标准图像, error 输入、颜色、资源或取消错误
func decodeContextImage(ctx context.Context, r io.Reader, opts *DecodeOptions) (image.Image, error) {
	d, err := newDecoder(ctx, r, opts)
	if err != nil {
		return nil, err
	}
	defer d.Close()
	info := d.info
	mapping, err := jxrBasicRasterDisplay(info)
	if err != nil {
		return nil, err
	}
	if err := checkJXRDisplayPositions(info); err != nil {
		return nil, err
	}
	bounds := info.Bounds
	count := uint64(bounds.Dx()) * uint64(bounds.Dy())
	maxInt := uint64(^uint(0) >> 1)
	if count > maxInt/8 {
		return nil, &LimitError{Resource: "max memory bytes", Limit: maxInt, Required: jxrSaturatedMul(count, 8), Detail: "display image memory"}
	}
	if count*8 >= d.limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(count*8, 1), Detail: "display image memory"}
	}
	raster, err := d.decodeReserved(ctx, count*8)
	if err != nil {
		return nil, err
	}
	return renderJXRDisplay(ctx, raster, info, mapping, ResampleNearest, nil)
}

// DecodeConfig 读取方向变换后的标准显示尺寸，不重建像素
// 入参: r 输入流
// 返回: image.Config 标准显示配置, error 输入或头部错误
func DecodeConfig(r io.Reader) (image.Config, error) {
	cfg, err := decodeConfigContext(context.Background(), r, nil)
	if err != nil {
		return image.Config{}, wrapJXRError("DecodeConfig", err)
	}
	return cfg, nil
}

// DecodeConfigContext 在资源上限内校验头部并返回十六位显示配置，不校验ICC或多通道显示转换支持
// 入参: ctx Context，nil采用Background, r 输入流, opts 解码选项，nil采用默认上限
// 返回: image.Config 十六位显示配置, error 输入、头部、资源或取消错误
func DecodeConfigContext(ctx context.Context, r io.Reader, opts *DecodeOptions) (image.Config, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg, err := decodeConfigContext(ctx, r, opts)
	if err != nil {
		return image.Config{}, wrapJXRError("DecodeConfigContext", err)
	}
	return cfg, nil
}

// decodeConfigContext 执行头部校验与显示配置提取，由公开入口唯一归一化错误
// 入参: ctx 已规范化Context, r 输入流, opts 解码选项
// 返回: image.Config 十六位显示配置, error 输入、头部、资源或取消错误
func decodeConfigContext(ctx context.Context, r io.Reader, opts *DecodeOptions) (image.Config, error) {
	d, err := newDecoder(ctx, r, opts)
	if err != nil {
		return image.Config{}, err
	}
	defer d.Close()
	return image.Config{ColorModel: color.NRGBA64Model, Width: d.info.Bounds.Dx(), Height: d.info.Bounds.Dy()}, nil
}

// displayLinear 将线性颜色分量转换为sRGB显示值，不修改原始分量
// 入参: value 线性分量
// 返回: float64 显示分量
func displayLinear(value float64) float64 {
	if value <= 0.0031308 {
		return value * 12.92
	}
	return 1.055*math.Pow(value, 1/2.4) - 0.055
}

// displayWord 将显示分量转换为十六位值，非数值采用零，扩展范围在显示阶段裁剪
// 入参: value 显示分量
// 返回: uint16 十六位值
func displayWord(value float64) uint16 {
	if math.IsNaN(value) {
		return 0
	}
	return uint16(math.Floor(colorClamp(value)*65535 + 0.5))
}
