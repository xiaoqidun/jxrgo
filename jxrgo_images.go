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

// jxrTileImagePlanWork 为每幅逐块源的固定描述、原选项值和独占平面选项保留容量，ICC及量化数组另计
const jxrTileImagePlanWork uint64 = 4096

// ImageInput 指定多图文件中的一幅原生图像及其编码选项，Options为nil采用无损JXR
// Raster、Tiles与Rows三选一；Tiles不可与Raster或Rows混用，Rows可与Raster组合流式编码多页或主图与预览图
// Info或Raster.Info中的ImageType控制预览及页序列语义
type ImageInput struct {
	Raster  *Raster
	Info    Info
	Tiles   RasterTileReader
	Rows    RasterRowReader
	Options *EncodeOptions
}

// FileEncodeOptions 限制整个多图文件的输出及活动工作存储，零值采用默认上限
// MaxPixels限制所有图像可见像素之和；MaxMemoryBytes包括全部输入及当前图像编码工作
// 每图显式设置的资源上限可以进一步收紧，不扩大文件上限
type FileEncodeOptions struct {
	Limits         Limits
	MaxOutputBytes uint64
}

// jxrFileImagePlan 保存逐图原生存储容量及预览格式校验所需的固定标识，不缓存编码图像
type jxrFileImagePlan struct {
	work       uint64
	identifier [16]byte
}

// jxrTileImagePlan 保存调用前的独占元数据和量化配置快照，不保留其他图像的系数或样本
type jxrTileImagePlan struct {
	info       Info
	read       RasterTileReader
	rows       RasterRowReader
	raster     *Raster
	options    EncodeOptions
	retained   uint64
	identifier [16]byte
}

// ImageCount 计数JPEG XR容器目录链中的图像数量，独立码流固定返回一，不重建像素或校验各图像全部码流
// 顺序遍历并检测目录环路，确保目录链接完整有效
// 入参: ctx Context，nil采用Background
// 返回: int 图像数量, error 目录、取消或关闭错误
func (d *Decoder) ImageCount(ctx context.Context) (int, error) {
	if d == nil {
		return 0, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	_, count, err := d.imageDirectory(ctx, -1)
	if err != nil {
		return 0, wrapJXRError("Decoder.ImageCount", err)
	}
	return count, nil
}

// ImageInfo 按ImageCount的索引顺序读取指定图像的独立信息，不改变默认首图及其缓存
// ImageType保留预览及页序列标志，预览图的对应关系按目录顺序解释
// 入参: ctx Context，nil采用Background, index 从零开始的目录序号
// 返回: Info 独立元数据副本, error 序号、数据、资源、取消或关闭错误
func (d *Decoder) ImageInfo(ctx context.Context, index int) (Info, error) {
	if d == nil {
		return Info{}, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	selected, err := d.imageDecoder(ctx, index)
	if err != nil {
		return Info{}, wrapJXRError("Decoder.ImageInfo", err)
	}
	if err := selected.checkInfoCopy(); err != nil {
		return Info{}, wrapJXRError("Decoder.ImageInfo", err)
	}
	info := cloneInfo(selected.info)
	if err := ctx.Err(); err != nil {
		return Info{}, err
	}
	return info, nil
}

// ImageDecoder 按ImageCount的索引顺序创建指定图像的独立子解码器，共享底层随机读取源与只读输入
// 子解码器继承KeepOrientation、Band、Workers及受控内存预算，支持Info、Decode、DecodeStrips、DecodeRegion、DecodeReduced与独立Close，与父解码器严格并发安全
// 入参: ctx Context，nil采用Background, index 从零开始的目录序号
// 返回: *Decoder 独立子图像解码器, error 序号、数据、资源、取消或关闭错误
func (d *Decoder) ImageDecoder(ctx context.Context, index int) (*Decoder, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	selected, err := d.imageDecoderIndependent(ctx, index, nil)
	if err != nil {
		return nil, wrapJXRError("Decoder.ImageDecoder", err)
	}
	return selected, nil
}

// ImageDecoderWithOptions 按ImageCount的索引顺序创建指定图像的独立子解码器并允许显式覆写解码选项与独立预算
// opts为nil时等价于ImageDecoder继承父解码器选项与剩余预算；opts非nil时为子解码器启用独立Limits、Workers、Band、KeepOrientation与LegacyHighpassPrediction，不受父第0页预算永久缩水影响
// 入参: ctx Context，nil采用Background, index 从零开始的目录序号, opts 可选独立解码选项
// 返回: *Decoder 独立子图像解码器, error 序号、选项、数据、资源、取消或关闭错误
func (d *Decoder) ImageDecoderWithOptions(ctx context.Context, index int, opts *DecodeOptions) (*Decoder, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if opts != nil {
		if opts.Workers < 0 {
			return nil, wrapJXRError("Decoder.ImageDecoderWithOptions", fmt.Errorf("%w: negative decode workers", ErrInvalid))
		}
		if opts.Band > BandDC {
			return nil, wrapJXRError("Decoder.ImageDecoderWithOptions", fmt.Errorf("%w: decode band", ErrInvalid))
		}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	selected, err := d.imageDecoderIndependent(ctx, index, opts)
	if err != nil {
		return nil, wrapJXRError("Decoder.ImageDecoderWithOptions", err)
	}
	return selected, nil
}

// DecodeImage 校验目录链接并重建指定图像，主图及独立透明度使用该目录自己的字段及码流
// 不把整幅输入复制为单图文件，内存预算包含仍保留的首图索引和元数据
// 入参: ctx Context，nil采用Background, index 从零开始的目录序号
// 返回: *Raster 独立原生图像, error 序号、数据、能力、资源、取消或关闭错误
func (d *Decoder) DecodeImage(ctx context.Context, index int) (*Raster, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	raster, err := d.decodeImageLocked(ctx, index)
	if err != nil {
		return nil, wrapJXRError("Decoder.DecodeImage", err)
	}
	return raster, nil
}

// decodeImageLocked 在持有父解码器锁时解码明确图像，保持首图和临时视图的预算规则
// 入参: ctx Context, index 图像序号
// 返回: *Raster 独占图像, error 解码或资源错误
func (d *Decoder) decodeImageLocked(ctx context.Context, index int) (*Raster, error) {
	selected, err := d.imageDecoder(ctx, index)
	if err != nil {
		return nil, err
	}
	if selected == d {
		return d.decodeLocked(ctx, 0)
	}
	return selected.decodeLocked(ctx, 0)
}

// imageDirectory 逐个检查目录并以常量空间检测链接环路，显式图像查询不忽略后续坏链接
// 入参: ctx 已规范化Context, index 目标序号，负一表示只计数
// 返回: jxrDirectory 目标目录视图, int 总数, error 目录、序号、取消或关闭错误
func (d *Decoder) imageDirectory(ctx context.Context, index int) (jxrDirectory, int, error) {
	if d.closed {
		return jxrDirectory{}, 0, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return jxrDirectory{}, 0, err
	}
	if len(d.data) < 8 || index < -1 {
		return jxrDirectory{}, 0, fmt.Errorf("%w: JPEG XR image directory request", ErrInvalid)
	}
	if d.source.format == FormatCodestream || d.singleImage {
		if index > 0 {
			return jxrDirectory{}, 0, fmt.Errorf("%w: JPEG XR image index", ErrInvalid)
		}
		return jxrDirectory{}, 1, nil
	}
	available := d.limits.MaxMemoryBytes
	if d.fileInput != nil {
		defer func() { d.fileInput.ctx = nil }()
		retained := d.inputWork() + d.retainedHeaderWork()
		if retained > available {
			return jxrDirectory{}, 0, &LimitError{Resource: "JPEG XR retained directory source", Limit: available, Required: retained}
		}
		available -= retained
	}
	anchor := uint64(binary.LittleEndian.Uint32(d.data[4:]))
	current, err := d.imageDirectoryAt(ctx, anchor, available)
	if err != nil {
		return jxrDirectory{}, 0, err
	}
	return jxrScanImageDirectories(ctx, current, anchor, available, index, func(offset, maximum uint64) (jxrDirectory, error) {
		return d.imageDirectoryAt(ctx, offset, maximum)
	})
}

// jxrScanImageDirectories 顺序校验容器目录链，并保留指定目录的视图
// 入参: ctx Context, current 首目录视图, anchor 环路检测锚点偏移, available 可用临时容量, index 目标序号, read 目录读取回调
// 返回: jxrDirectory 选中目录视图, int 目录总数, error 目录、序号、资源或取消错误
func jxrScanImageDirectories(ctx context.Context, current jxrDirectory, anchor, available uint64, index int, read func(uint64, uint64) (jxrDirectory, error)) (jxrDirectory, int, error) {
	length := current.dataSize()
	selected := jxrDirectory{}
	power, span, count := uint64(1), uint64(0), 0
	for {
		if err := ctx.Err(); err != nil {
			return jxrDirectory{}, 0, err
		}
		if count == index {
			selected = current
		}
		count++
		next := uint64(current.next)
		if next == 0 {
			break
		}
		span++
		if next == anchor || uint64(count) > length/2 {
			return jxrDirectory{}, 0, fmt.Errorf("%w: JPEG XR image directory cycle", ErrInvalid)
		}
		if span == power {
			anchor, power, span = next, power*2, 0
		}
		current = jxrDirectory{}
		work := selected.storageWork()
		if work > available {
			return jxrDirectory{}, 0, &LimitError{Resource: "JPEG XR selected directory capacity", Limit: available, Required: work}
		}
		var err error
		current, err = read(next, available-work)
		if err != nil {
			return jxrDirectory{}, 0, err
		}
	}
	if index >= count {
		return jxrDirectory{}, 0, fmt.Errorf("%w: JPEG XR image index", ErrInvalid)
	}
	return selected, count, nil
}

// retainedHeaderWork 计算首图索引及元数据的活动容量，不重复计入共享文件字节
// 返回: uint64 首图仍保留的头部及信息容量
func (d *Decoder) retainedHeaderWork() uint64 {
	work := uint64(512) + d.source.primary.headerWork() + uint64(cap(d.source.container.profiles))*4 + uint64(len(d.info.Components))*128 + jxrDescriptionWork(d.source.container.description) + jxrDescriptionWork(d.info.Description)
	if d.source.alpha != nil {
		work += d.source.alpha.headerWork()
	}
	if d.source.container.encoding != nil {
		work += 4
	}
	if d.info.ColorEncoding != nil {
		work += 4
	}
	return work
}

// imageDecoder 在首图锁保护下建立临时独占目标视图，共享输入但不缓存其他图像或忽略其预算
// 入参: ctx 已规范化Context, index 从零开始的目录序号
// 返回: *Decoder 首图自身或独占临时解码器, error 目录、数据、资源、取消或关闭错误
func (d *Decoder) imageDecoder(ctx context.Context, index int) (*Decoder, error) {
	if d.closed {
		return nil, ErrClosed
	}
	if index < 0 {
		return nil, fmt.Errorf("%w: JPEG XR image index", ErrInvalid)
	}
	directory, _, err := d.imageDirectory(ctx, index)
	if err != nil {
		return nil, err
	}
	if index == 0 {
		return d, nil
	}
	return d.parseDirectoryDecoderWithOptions(ctx, &directory, false, nil)
}

// imageDecoderIndependent 在持有父解码器锁时建立与父解码器完全无可变状态共享的独立单图子解码器
// 入参: ctx 已规范化Context, index 从零开始的目录序号, opts 可选独立解码选项覆写
// 返回: *Decoder 独立子图像解码器, error 目录、数据、资源、取消或关闭错误
func (d *Decoder) imageDecoderIndependent(ctx context.Context, index int, opts *DecodeOptions) (*Decoder, error) {
	if d.closed {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if index < 0 {
		return nil, fmt.Errorf("%w: JPEG XR image index", ErrInvalid)
	}
	directory, _, err := d.imageDirectory(ctx, index)
	if err != nil {
		return nil, err
	}
	if d.source.format == FormatCodestream || d.singleImage {
		source := d.source
		primaryInput := source.primary.input
		if primaryInput == nil {
			primaryInput = source.container.imageInput
		}
		var clonedPrimary *jxrReaderAt
		if primaryInput != nil {
			clonedPrimary = cloneJXRReaderAt(primaryInput)
			if source.primary.input != nil {
				source.primary.input = clonedPrimary
			}
			if source.container.imageInput != nil {
				source.container.imageInput = clonedPrimary
			}
		}
		if source.primary.alpha != nil {
			primaryAlpha := *source.primary.alpha
			source.primary.alpha = &primaryAlpha
		}
		if source.alpha != nil {
			alphaCopy := *source.alpha
			if alphaCopy.alpha != nil {
				alphaPlane := *alphaCopy.alpha
				alphaCopy.alpha = &alphaPlane
			}
			source.alpha = &alphaCopy
		}
		var alphaInput *jxrReaderAt
		if source.alpha != nil {
			alphaInput = source.alpha.input
		}
		if alphaInput == nil {
			alphaInput = source.container.alphaInput
		}
		if alphaInput != nil {
			clonedAlpha := clonedPrimary
			if alphaInput != primaryInput {
				clonedAlpha = cloneJXRReaderAt(alphaInput)
			}
			if source.alpha != nil && source.alpha.input != nil {
				source.alpha.input = clonedAlpha
			}
			if source.container.alphaInput != nil {
				source.container.alphaInput = clonedAlpha
			}
		}
		limits := d.limits
		workers := d.workers
		band := d.decodeBand
		legacyHighpass := d.legacyHighpassPrediction
		keepOrientation := d.keepOrientation
		info := cloneInfo(d.info)
		if opts == nil {
			retained, input := d.retainedHeaderWork(), d.inputWork()
			shared := uint64(cap(d.data)) + d.inputExtra
			parentExtra := retained
			if input > shared {
				parentExtra += input - shared
			}
			if parentExtra >= limits.MaxMemoryBytes {
				return nil, &LimitError{Resource: "JPEG XR retained first image memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(parentExtra, 1)}
			}
			limits.MaxMemoryBytes -= parentExtra
		} else {
			limits = opts.Limits.normalized()
			workers = opts.Workers
			band = opts.Band
			legacyHighpass = opts.LegacyHighpassPrediction
			keepOrientation = opts.KeepOrientation
			if keepOrientation {
				source.container.transform = 0
			} else {
				source.container.transform = uint32(source.container.orientation)
			}
			source.primary.primary.legacyHighpassPrediction = legacyHighpass
			if source.primary.alpha != nil {
				source.primary.alpha.legacyHighpassPrediction = legacyHighpass
			}
			if source.alpha != nil {
				source.alpha.primary.legacyHighpassPrediction = legacyHighpass
				if source.alpha.alpha != nil {
					source.alpha.alpha.legacyHighpassPrediction = legacyHighpass
				}
			}
			if source.container.transform != d.source.container.transform {
				info = cloneInfo(source.info())
			}
			if err := checkJXRDimensions(uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy()), limits.MaxPixels); err != nil {
				return nil, err
			}
		}
		selected := &Decoder{data: d.data, source: source, info: info, limits: limits, singleImage: true, inputExtra: d.inputExtra, legacyHighpassPrediction: legacyHighpass, keepOrientation: keepOrientation}
		selected.setWorkers(workers)
		selected.setDecodeBand(band)
		selected.bindInputContext(nil)
		if err := selected.checkInfoCopy(); err != nil {
			return nil, err
		}
		return selected, nil
	}
	return d.parseDirectoryDecoderWithOptions(ctx, &directory, true, opts)
}

// parseDirectoryDecoderWithOptions 解析指定容器目录并按继承或显式独立选项装配子图像解码器
// 入参: ctx 已规范化Context, directory 目标容器目录, independent 是否构建独立单图子解码器, opts 可选独立解码选项
// 返回: *Decoder 目标图像解码器, error 数据、能力、资源或取消错误
func (d *Decoder) parseDirectoryDecoderWithOptions(ctx context.Context, directory *jxrDirectory, independent bool, opts *DecodeOptions) (*Decoder, error) {
	limits := d.limits
	workers := d.workers
	band := d.decodeBand
	legacyHighpass := d.legacyHighpassPrediction
	keepOrientation := d.keepOrientation
	var parserLimit uint64
	var parentExtra uint64
	if opts == nil {
		retained, input := d.retainedHeaderWork(), d.inputWork()
		if input > limits.MaxMemoryBytes || retained > limits.MaxMemoryBytes-input {
			return nil, &LimitError{Resource: "JPEG XR retained first image memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(input, retained)}
		}
		parserLimit = limits.MaxMemoryBytes - input - retained
		parentExtra = retained + (input - uint64(cap(d.data)))
	} else {
		limits = opts.Limits.normalized()
		workers = opts.Workers
		band = opts.Band
		legacyHighpass = opts.LegacyHighpassPrediction
		keepOrientation = opts.KeepOrientation
		if uint64(cap(d.data)) >= limits.MaxMemoryBytes {
			return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(uint64(cap(d.data)), 1), Detail: "decoder retained input memory"}
		}
		parserLimit = limits.MaxMemoryBytes - uint64(cap(d.data))
	}
	parser := jxrParser{ctx: ctx, limit: parserLimit, legacyHighpassPrediction: legacyHighpass}
	if d.fileInput != nil {
		d.fileInput.ctx = ctx
		defer func() { d.fileInput.ctx = nil }()
	}
	if err := parser.reserve(directory.storageWork()); err != nil {
		return nil, err
	}
	container, err := parser.directoryContainer(directory, limits.MaxPixels)
	if err != nil {
		return nil, err
	}
	source, err := parser.containerSource(&container, limits.MaxPixels)
	if err != nil {
		return nil, err
	}
	if keepOrientation {
		source.container.transform = 0
	}
	info, err := parser.info(&source)
	if err != nil {
		return nil, err
	}
	var fileInput *jxrReaderAt
	if !independent && d.fileInput != nil {
		parentExtra -= jxrReaderAtWork
		fileInput = d.fileInput
	}
	limits.MaxMemoryBytes -= parentExtra
	extra := uint64(cap(container.exif)) + uint64(cap(container.gps))
	if d.fileInput != nil {
		extra += uint64(cap(container.profile)) + uint64(cap(container.xmp)) + uint64(cap(container.iptc))
	}
	selected := &Decoder{data: d.data, source: source, info: info, limits: limits, singleImage: independent, fileInput: fileInput, inputExtra: extra, legacyHighpassPrediction: legacyHighpass, keepOrientation: keepOrientation}
	selected.setWorkers(workers)
	selected.setDecodeBand(band)
	selected.bindInputContext(nil)
	if independent {
		if err := selected.checkInfoCopy(); err != nil {
			return nil, err
		}
	}
	return selected, nil
}

// cloneJXRReaderAt 复制随机读取视图的只读边界与底层读取源并分配独立缓冲，避免多解码器并发竞争
// 入参: src 原随机读取视图，nil返回nil
// 返回: *jxrReaderAt 独立缓冲的随机读取视图
func cloneJXRReaderAt(src *jxrReaderAt) *jxrReaderAt {
	if src == nil {
		return nil
	}
	return &jxrReaderAt{source: src.source, base: src.base, size: src.size}
}

// EncodeImages 逐图流式编码并写入JXR多图容器文件，支持整图Raster、有界分块Tiles或单遍顺序行源Rows，每图写完后才链接前一目录
// w须位于文件起点且支持绝对定位，不关闭或截断输出，失败可能残留部分数据；调用期间不得并发修改输入图像及选项
// 入参: ctx Context，nil采用Background, w 可定位输出, images 按目录顺序排列的图像, opts 全文件资源上限
// 返回: error 容器关系、格式、精度、资源、取消、定位或写入错误
func EncodeImages(ctx context.Context, w io.WriteSeeker, images []ImageInput, opts *FileEncodeOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil || len(images) == 0 {
		return &FormatError{Op: "EncodeImages", Detail: "JPEG XR image file input"}
	}
	limits, maximum := DefaultLimits(), uint64(64<<20)
	if opts != nil {
		limits = opts.Limits.normalized()
		if opts.MaxOutputBytes != 0 {
			maximum = opts.MaxOutputBytes
		}
	}
	maximum = min(maximum, uint64(math.MaxUint32))
	hasTiles, hasRows := false, false
	for _, item := range images {
		if item.Tiles != nil {
			hasTiles = true
		}
		if item.Rows != nil {
			hasRows = true
		}
	}
	if hasTiles {
		if maximum < 8 {
			return &LimitError{Resource: "JPEG XR tiled image file output bytes", Limit: maximum, Required: 8}
		}
		if limits.MaxMemoryBytes < 4096 || uint64(len(images)) > (limits.MaxMemoryBytes-4096)/jxrTileImagePlanWork || uint64(len(images)) > uint64(^uint(0)>>1)/jxrTileImagePlanWork {
			return &LimitError{Resource: "JPEG XR tiled image file planning memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(4096, jxrSaturatedMul(uint64(len(images)), jxrTileImagePlanWork))}
		}
		for _, item := range images {
			if item.Raster != nil || item.Rows != nil || item.Tiles == nil {
				return &FormatError{Op: "EncodeImages", Detail: "mixed raster and tiled image inputs"}
			}
		}
		return wrapJXRError("EncodeImages", encodeJXRImagesTiles(ctx, w, images, limits, maximum))
	}
	if hasRows {
		if maximum < 8 {
			return &LimitError{Resource: "JPEG XR row image file output bytes", Limit: maximum, Required: 8}
		}
		if limits.MaxMemoryBytes < 4096 || uint64(len(images)) > (limits.MaxMemoryBytes-4096)/jxrTileImagePlanWork || uint64(len(images)) > uint64(^uint(0)>>1)/jxrTileImagePlanWork {
			return &LimitError{Resource: "JPEG XR row image file planning memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(4096, jxrSaturatedMul(uint64(len(images)), jxrTileImagePlanWork))}
		}
		for _, item := range images {
			if (item.Raster == nil) == (item.Rows == nil) {
				return &FormatError{Op: "EncodeImages", Detail: "invalid row or raster image input"}
			}
		}
		return wrapJXRError("EncodeImages", encodeJXRImagesRows(ctx, w, images, limits, maximum))
	}
	return wrapJXRError("EncodeImages", encodeJXRImages(ctx, w, images, limits, maximum))
}

// encodeJXRImages 执行多图预检、资源合并与逐包增量写出，不保留单幅完整压缩缓冲
// 入参: ctx 已规范化Context, w 可定位输出, images 按目录顺序的输入, limits 全文件资源上限, maximum 全文件输出上限
// 返回: error 元数据、关系、能力、精度、资源、取消或I/O错误
func encodeJXRImages(ctx context.Context, w io.WriteSeeker, images []ImageInput, limits Limits, maximum uint64) error {
	if maximum < 8 {
		return &LimitError{Resource: "JPEG XR image file output bytes", Limit: maximum, Required: 8}
	}
	if limits.MaxMemoryBytes < 4096 || uint64(len(images)) > (limits.MaxMemoryBytes-4096)/40 {
		return &LimitError{Resource: "JPEG XR image file planning memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(4096, jxrSaturatedMul(uint64(len(images)), 40))}
	}
	state := uint64(4096) + uint64(len(images))*40
	plans, inputs, err := planJXRImageFile(ctx, images, limits, state, maximum)
	if err != nil {
		return err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if position != 0 {
		return fmt.Errorf("%w: JPEG XR image file output position", ErrInvalid)
	}
	end, previous := uint64(8), uint64(0)
	for index, input := range images {
		if err := ctx.Err(); err != nil {
			return err
		}
		if end > maximum {
			return &LimitError{Resource: "JPEG XR image file output size", Limit: maximum, Required: end}
		}
		memory := limits.MaxMemoryBytes - state - (inputs - plans[index].work)
		options, err := jxrFileImageOptions(input.Options, limits, memory, maximum-end+8)
		if err != nil {
			return err
		}
		end, previous, err = writeJXRFileImageTo(ctx, w, input.Raster, &options, end, previous, maximum, index == len(images)-1)
		if err != nil {
			return err
		}
	}
	return ctx.Err()
}

// planJXRImageFile 预检全部原生存储、累计像素及预览和页序列约束
// 入参: ctx Context, images 图像输入, limits 全文件资源, state 计划及固定工作容量, maximum 全文件字节上限
// 返回: []jxrFileImagePlan 固定计划, uint64 全部输入容量, error 元数据、关系、资源或取消错误
func planJXRImageFile(ctx context.Context, images []ImageInput, limits Limits, state, maximum uint64) ([]jxrFileImagePlan, uint64, error) {
	plans := make([]jxrFileImagePlan, len(images))
	inputs, pixels := uint64(0), uint64(0)
	for index, input := range images {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		work, err := jxrRasterWork(input.Raster, limits.MaxMemoryBytes-state)
		if err != nil {
			return nil, 0, err
		}
		if work > limits.MaxMemoryBytes-state-inputs {
			return nil, 0, &LimitError{Resource: "JPEG XR combined image inputs", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(inputs, work))}
		}
		inputs += work
		options, err := jxrFileImageOptions(input.Options, limits, limits.MaxMemoryBytes-state, maximum)
		if err != nil {
			return nil, 0, err
		}
		info := input.Raster.info
		if options.SeparateAlpha && !info.Alpha {
			return nil, 0, fmt.Errorf("%w: JPEG XR separate alpha without source alpha", ErrInvalid)
		}
		if info.ImageType&^3 != 0 || options.Orientation > OrientationRotate90FlipXY {
			return nil, 0, fmt.Errorf("%w: JPEG XR image type or orientation", ErrInvalid)
		}
		width, height := uint64(input.Raster.Bounds().Dx()), uint64(input.Raster.Bounds().Dy())
		if height == 0 {
			return nil, 0, fmt.Errorf("%w: JPEG XR image dimensions", ErrInvalid)
		}
		if width > (limits.MaxPixels-pixels)/height {
			return nil, 0, &LimitError{Resource: "JPEG XR combined image pixels", Limit: limits.MaxPixels, Required: jxrSaturatedAdd(pixels, jxrSaturatedMul(width, height))}
		}
		pixels += width * height
		plans[index].work = work
		plans[index].identifier, _, err = jxrRasterIdentifier(info, [16]byte{})
		if err != nil {
			return nil, 0, err
		}
	}
	if err := checkJXRImageSequence(ctx, len(images), func(index int) (Info, [16]byte, Orientation) {
		orientation := OrientationNone
		if images[index].Options != nil {
			orientation = images[index].Options.Orientation
		}
		return images[index].Raster.info, plans[index].identifier, orientation
	}); err != nil {
		return nil, 0, err
	}
	return plans, inputs, nil
}

// jxrPreviewFormat 检查允许用于不同原始格式图像的预览GUID，完全相同GUID由调用方先行允许
// 入参: id 标准GUID末字节
// 返回: bool 是否为明确列出的预览格式
func jxrPreviewFormat(id byte) bool {
	if id >= 0x20 && id <= 0x25 || id >= 0x2e && id <= 0x33 {
		return true
	}
	switch id {
	case 0x08, 0x0c, 0x0f, 0x1c, 0x2c, 0x44, 0x45, 0x48, 0x4c, 0x4d, 0x50, 0x54, 0x56:
		return true
	default:
		return false
	}
}

// jxrFileImageOptions 合并当前图像和整个文件的上限，不允许图像自行扩大剩余容量
// 入参: original 当前图像选项, limits 文件上限, memory 当前图像可用存储, output 当前独立文件暂存可用字节
// 返回: EncodeOptions 独立选项副本, error 文件类型或资源错误
func jxrFileImageOptions(original *EncodeOptions, limits Limits, memory, output uint64) (EncodeOptions, error) {
	options := EncodeOptions{}
	if original != nil {
		options = *original
	}
	if options.Format != FormatUnknown && options.Format != FormatJXR {
		return options, fmt.Errorf("%w: JPEG XR multi-image file format", ErrInvalid)
	}
	if options.Limits.MaxPixels != 0 {
		limits.MaxPixels = min(limits.MaxPixels, options.Limits.MaxPixels)
	}
	if options.Limits.MaxMemoryBytes != 0 {
		memory = min(memory, options.Limits.MaxMemoryBytes)
	}
	if options.MaxOutputBytes != 0 {
		output = min(output, options.MaxOutputBytes)
	}
	if memory == 0 {
		return options, &LimitError{Resource: "JPEG XR remaining image memory", Limit: 0, Required: 1}
	}
	if output == 0 {
		return options, &LimitError{Resource: "JPEG XR remaining image output bytes", Limit: 0, Required: 1}
	}
	options.Format, options.MaxOutputBytes = FormatJXR, output
	options.Limits = limits
	options.Limits.MaxMemoryBytes = memory
	return options, nil
}

// writeJXRFileImageTo 在全文件额度内逐包写一幅原图，完成索引、长度和偶数填充后才链接前图
// 入参: ctx Context, w 可定位输出, raster 当前原生图像, options 已合并的图像资源, start 当前目录位置, previous 前一链接位置, maximum 全文件字节上限, last 是否最后图像
// 返回: uint64 文件结束位置, uint64 当前链接位置, error 格式、精度、资源、取消或I/O错误
func writeJXRFileImageTo(ctx context.Context, w io.WriteSeeker, raster *Raster, options *EncodeOptions, start, previous, maximum uint64, last bool) (uint64, uint64, error) {
	if start < 8 || start > maximum || options == nil {
		return 0, 0, fmt.Errorf("%w: incremental image file parameters", ErrInvalid)
	}
	var end, link uint64
	var err error
	if options.Uncompressed {
		selected := *options
		selected.MaxOutputBytes = min(selected.MaxOutputBytes, maximum-start+8)
		data, packErr := encodeUncompressedJXR(ctx, raster, selected)
		if packErr != nil {
			return 0, 0, packErr
		}
		link, err = relocateJXRImageDirectory(data, start)
		if err == nil {
			position, seekErr := w.Seek(0, io.SeekCurrent)
			if seekErr != nil {
				return 0, 0, seekErr
			}
			wanted, payload := int64(start), data
			if start == 8 {
				wanted = 0
			} else {
				payload = data[8:]
			}
			if position != wanted {
				return 0, 0, fmt.Errorf("%w: incremental uncompressed file output position", ErrInvalid)
			}
			if err = writeEncoderOutput(ctx, w, payload); err == nil {
				end = uint64(wanted) + uint64(len(payload))
			}
		}
	} else if options.SeparateAlpha {
		selected := *options
		selected.MaxOutputBytes = min(selected.MaxOutputBytes, maximum-start+8)
		end, link, err = writeJXRSeparateFileAt(ctx, w, raster, selected, start)
	} else {
		layout, output, planErr := makeJXREncodingPlan(ctx, raster, options)
		if planErr != nil {
			if planErr == errJXREncodingPlanMemory || planErr == errJXRInputHeaderMemory {
				retained, _ := jxrRasterWork(raster, output.stream.maxMemory)
				return 0, 0, jxrPlanLimitError(planErr, output.stream.maxMemory, retained)
			}
			return 0, 0, planErr
		}
		retained, sourceErr := jxrRasterWork(raster, output.stream.maxMemory)
		if sourceErr != nil {
			return 0, 0, sourceErr
		}
		output.stream.maxBytes = min(output.stream.maxBytes, maximum-start+8)
		end, link, err = writeJXRFileAt(ctx, w, jxrWithContainerMetadata(raster.info, options), &layout, output.stream, retained, start, func(settings jxrStreamOutput) error {
			return encodeJXRCodestreamTo(ctx, w, raster, layout, settings)
		})
	}
	if err != nil {
		return 0, 0, err
	}
	return finishJXRImageOutput(ctx, w, start, end, previous, link, maximum, last, "incremental image file")
}

// relocateJXRImageDirectory 共用完整图像及仅目录前缀的地址重定位，不读取或推测压缩包
// 入参: data 目录位于八的独占文件缓冲或前缀, start 当前目录在完整文件中的起点
// 返回: uint64 全文件目录链接位置, error 目录、类型、地址或容量错误
func relocateJXRImageDirectory(data []byte, start uint64) (uint64, error) {
	if start < 8 || start&1 != 0 || start > math.MaxUint32 {
		return 0, fmt.Errorf("%w: JPEG XR relocated directory position", ErrInvalid)
	}
	directory, err := readJXRDirectory(data)
	if err != nil || binary.LittleEndian.Uint32(data[4:]) != 8 || directory.next != 0 {
		return 0, fmt.Errorf("%w: JPEG XR generated image directory", ErrInvalid)
	}
	link := start + 2 + uint64(len(directory.entries))
	if link > math.MaxUint32 {
		return 0, fmt.Errorf("%w: JPEG XR relocated directory capacity", ErrLimit)
	}
	for index := range len(directory.entries) / 12 {
		entry := directory.entries[index*12 : (index+1)*12]
		tag, typ := binary.LittleEndian.Uint16(entry), binary.LittleEndian.Uint16(entry[2:])
		if typ == 0 || typ > 12 {
			return 0, fmt.Errorf("%w: JPEG XR generated field type", ErrInvalid)
		}
		count := uint64(binary.LittleEndian.Uint32(entry[4:]))
		width := jxrTIFFTypeWidth[typ]
		size := width * count
		if (tag == 0x8769 || tag == 0x8825) && typ == 4 && count == 1 {
			if start > 8 {
				localOffset := uint64(binary.LittleEndian.Uint32(entry[8:]))
				if err := relocateJXRSubIFD(data, localOffset, start-8, tag == 0x8769, 0); err != nil {
					return 0, err
				}
				position := localOffset + start - 8
				if position > math.MaxUint32 {
					return 0, fmt.Errorf("%w: JPEG XR relocated Sub-IFD offset", ErrLimit)
				}
				binary.LittleEndian.PutUint32(entry[8:], uint32(position))
			}
			continue
		}
		if size > 4 || tag == 0xbcc0 || tag == 0xbcc2 {
			position := uint64(binary.LittleEndian.Uint32(entry[8:])) + start - 8
			if position > math.MaxUint32 {
				return 0, fmt.Errorf("%w: JPEG XR relocated field offset", ErrLimit)
			}
			binary.LittleEndian.PutUint32(entry[8:], uint32(position))
		}
	}
	return link, nil
}

// finishJXRImageOutput 校验当前图像字节上限、执行偶数边界填充并最后提交前一目录链接，不在未完成图像上提前建立链接
// 入参: ctx Context, w 可定位输出, start 当前目录, end 当前图像结束位置, previous 前一链接位置, link 当前链接位置, maximum 全文件上限, last 是否末图, scope 错误资源前缀
// 返回: uint64 文件结束位置, uint64 当前链接位置, error 容量、取消、定位或写入错误
func finishJXRImageOutput(ctx context.Context, w io.WriteSeeker, start, end, previous, link, maximum uint64, last bool, scope string) (uint64, uint64, error) {
	if end > maximum {
		return 0, 0, &LimitError{Resource: scope + " output capacity", Limit: maximum, Required: end}
	}
	if !last && end&1 != 0 {
		if end == maximum {
			return 0, 0, &LimitError{Resource: scope + " alignment capacity", Limit: maximum, Required: jxrSaturatedAdd(end, 1)}
		}
		if err := writeEncoderOutput(ctx, w, []byte{0}); err != nil {
			return 0, 0, err
		}
		end++
	}
	if previous != 0 {
		if err := seekJXRImageOutput(ctx, w, previous); err != nil {
			return 0, 0, err
		}
		var next [4]byte
		binary.LittleEndian.PutUint32(next[:], uint32(start))
		if err := writeEncoderOutput(ctx, w, next[:]); err != nil {
			return 0, 0, err
		}
		if err := seekJXRImageOutput(ctx, w, end); err != nil {
			return 0, 0, err
		}
	}
	return end, link, ctx.Err()
}

// seekJXRImageOutput 检查输出定位结果，保留底层错误并响应取消
// 入参: ctx Context, w 可定位输出, position 文件绝对位置
// 返回: error 定位协议、底层或取消错误
func seekJXRImageOutput(ctx context.Context, w io.WriteSeeker, position uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	actual, err := w.Seek(int64(position), io.SeekStart)
	if err != nil {
		return err
	}
	if actual != int64(position) {
		return fmt.Errorf("%w: JPEG XR output seek position", ErrInvalid)
	}
	return nil
}

// checkJXRImageSequence 共用容器页序列和预览对应规则，不要求源已持有完整样本或建立全部编码索引
// 入参: ctx Context, count 图像数, describe 按序返回已校验且调用期间不变的描述、GUID和请求方向
// 返回: error 页标志、实际编码尺寸、预览对应、格式或取消错误
func checkJXRImageSequence(ctx context.Context, count int, describe func(int) (Info, [16]byte, Orientation)) error {
	var dimensions [2]image.Point
	pages := ImageTypeDefault
	for index := range count {
		if err := ctx.Err(); err != nil {
			return err
		}
		info, _, orientation := describe(index)
		if info.ImageType&^3 != 0 || orientation > OrientationRotate90FlipXY {
			return fmt.Errorf("%w: JPEG XR image type or orientation", ErrInvalid)
		}
		if index == 0 {
			pages = info.ImageType & ImageTypePage
		}
		if info.ImageType&ImageTypePage != pages {
			return fmt.Errorf("%w: JPEG XR mixed page image types", ErrInvalid)
		}
		if pages != 0 {
			format, err := rasterPixelFormat(info)
			if err != nil {
				return err
			}
			shape := info.Bounds.Size()
			if jxrInputTransform(info, format.color, uint8(orientation))&4 != 0 {
				shape.X, shape.Y = shape.Y, shape.X
			}
			group := int(info.ImageType & ImageTypePreview)
			if dimensions[group] != (image.Point{}) && dimensions[group] != shape {
				return fmt.Errorf("%w: JPEG XR page image dimensions", ErrInvalid)
			}
			dimensions[group] = shape
		}
	}
	base := 0
	for index := range count {
		if err := ctx.Err(); err != nil {
			return err
		}
		info, identifier, _ := describe(index)
		if info.ImageType&ImageTypePreview == 0 {
			continue
		}
		for base < count {
			if err := ctx.Err(); err != nil {
				return err
			}
			other, _, _ := describe(base)
			if other.ImageType&ImageTypePreview == 0 {
				break
			}
			base++
		}
		if base == count {
			return fmt.Errorf("%w: JPEG XR preview without image", ErrInvalid)
		}
		_, original, _ := describe(base)
		if identifier != original && !jxrPreviewFormat(identifier[15]) {
			return fmt.Errorf("%w: JPEG XR preview pixel format", ErrInvalid)
		}
		base++
	}
	return ctx.Err()
}

// encodeJXRImagesTiles 从有界原生提供器逐图逐包写JXR容器文件，预检并复制全部描述、ICC及完整量化配置
// 入参: ctx 已规范化Context, w 可定位输出, images 按目录顺序的原生逐块源, limits 全文件资源上限, maximum 全文件输出上限
// 返回: error 参数、描述、对应关系、能力、精度、资源、提供器、取消或I/O错误
func encodeJXRImagesTiles(ctx context.Context, w io.WriteSeeker, images []ImageInput, limits Limits, maximum uint64) error {
	state := uint64(4096) + uint64(len(images))*jxrTileImagePlanWork
	plans, retained, err := planJXRTileImageFile(ctx, images, limits, state, maximum)
	if err != nil {
		return err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if position != 0 {
		return fmt.Errorf("%w: JPEG XR tiled image file output position", ErrInvalid)
	}
	end, previous := uint64(8), uint64(0)
	for index := range plans {
		if err := ctx.Err(); err != nil {
			return err
		}
		if end > maximum {
			return &LimitError{Resource: "JPEG XR tiled image file output size", Limit: maximum, Required: end}
		}
		plan := &plans[index]
		memory := limits.MaxMemoryBytes - state - (retained - plan.retained)
		options, err := jxrFileImageOptions(&plan.options, limits, memory, maximum-end+8)
		if err != nil {
			return err
		}
		end, previous, err = writeJXRTileFileImageTo(ctx, w, plan, options, end, previous, maximum, index == len(plans)-1)
		if err != nil {
			return err
		}
	}
	return ctx.Err()
}

// planJXRTileImageFile 在任何回调及输出前快照全部描述和量化索引，先为共存的原配置及副本预检容量
// 入参: ctx Context, images 原始逐块源, limits 全文件上限, state 固定描述容量, maximum 全文件输出上限
// 返回: []jxrTileImagePlan 独占快照, uint64 全部快照的额外常驻容量, error 描述、选项、关系、资源或取消错误
func planJXRTileImageFile(ctx context.Context, images []ImageInput, limits Limits, state, maximum uint64) ([]jxrTileImagePlan, uint64, error) {
	if state >= limits.MaxMemoryBytes {
		return nil, 0, &LimitError{Resource: "tiled image snapshot planning memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, 1)}
	}
	quantMaximum, quantWork := (limits.MaxMemoryBytes-state)/2, uint64(0)
	for _, input := range images {
		var options EncodeOptions
		if input.Options != nil {
			options = *input.Options
		}
		amount, err := jxrTileOptionsQuantizationWork(ctx, options, quantMaximum-quantWork)
		if err != nil {
			return nil, 0, err
		}
		quantWork += amount
	}
	plans := make([]jxrTileImagePlan, len(images))
	retained, pixels := uint64(0), uint64(0)
	reserved := quantWork * 2
	for index, input := range images {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		if input.Tiles == nil {
			return nil, 0, fmt.Errorf("%w: nil tiled image reader", ErrInvalid)
		}
		if reserved >= limits.MaxMemoryBytes-state-retained {
			return nil, 0, &LimitError{Resource: "tiled image description and quantization memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, 1)))}
		}
		options, err := jxrFileImageOptions(input.Options, limits, limits.MaxMemoryBytes-state-retained-reserved, maximum)
		if err != nil {
			return nil, 0, err
		}
		if err := checkJXREncodeOptions(options); err != nil {
			return nil, 0, err
		}
		if options.TileSize.X == 0 || options.TileSize.Y == 0 {
			return nil, 0, fmt.Errorf("%w: tiled images require explicit tiles", ErrUnsupported)
		}
		info, work, err := prepareJXRTileInfo(ctx, input.Info, options)
		if err != nil {
			return nil, 0, err
		}
		if options.SeparateAlpha && !info.Alpha {
			return nil, 0, fmt.Errorf("%w: JPEG XR separate alpha without source alpha", ErrInvalid)
		}
		clearJXRMergedMetadata(&options)
		if work > limits.MaxMemoryBytes-state-retained-reserved {
			return nil, 0, &LimitError{Resource: "tiled image descriptions memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, work)))}
		}
		retained += work
		width, height := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
		if width > (limits.MaxPixels-pixels)/height {
			return nil, 0, &LimitError{Resource: "combined tiled image pixels", Limit: limits.MaxPixels, Required: jxrSaturatedAdd(pixels, jxrSaturatedMul(width, height))}
		}
		pixels += width * height
		identifier, _, err := jxrRasterIdentifier(info, [16]byte{})
		if err != nil {
			return nil, 0, err
		}
		reader := input.Tiles
		wrapped := func(ctx context.Context, region image.Rectangle, target *Raster) error {
			if err := reader(ctx, region, target); err != nil {
				return jxrCallbackError{err: err}
			}
			return nil
		}
		plans[index] = jxrTileImagePlan{info: info, read: wrapped, options: options, retained: work, identifier: identifier}
	}
	if err := checkJXRImageSequence(ctx, len(plans), func(index int) (Info, [16]byte, Orientation) {
		return plans[index].info, plans[index].identifier, plans[index].options.Orientation
	}); err != nil {
		return nil, 0, err
	}
	for index := range plans {
		plan := &plans[index]
		amount, err := jxrTileOptionsQuantizationWork(ctx, plan.options, quantWork)
		if err != nil {
			return nil, 0, err
		}
		quantWork -= amount
		options, work, err := snapshotJXRTileOptions(ctx, plan.info, plan.options, limits.MaxMemoryBytes-state-retained-quantWork*2)
		if err != nil {
			return nil, 0, err
		}
		plan.options, plan.retained = options, plan.retained+work
		retained += work
	}
	return plans, retained, ctx.Err()
}

// jxrTileOptionsQuantizationWork 计算主平面及显式透明度配置的完整容量，为批量快照预检共用额度
// 入参: ctx Context, options 编码选项, maximum 两份原始配置共同容量上限
// 返回: uint64 原配置的保守容量, error 资源或取消错误
func jxrTileOptionsQuantizationWork(ctx context.Context, options EncodeOptions, maximum uint64) (uint64, error) {
	work, err := jxrQuantizationWork(ctx, options.Quantization, maximum)
	if err != nil {
		return 0, err
	}
	if options.Alpha != nil {
		amount, err := jxrQuantizationWork(ctx, options.Alpha.Quantization, maximum-work)
		if err != nil {
			return 0, err
		}
		work += amount
	}
	return work, ctx.Err()
}

// clearJXRMergedMetadata 清空已合并并深拷贝至目标Info的容器元数据覆盖字段，避免二次覆盖或回调篡改
// 入参: options 待清理的编码选项指针
func clearJXRMergedMetadata(options *EncodeOptions) {
	options.Resolution = nil
	options.ICCProfile = nil
	options.ColorHint = nil
	options.ColorEncoding = nil
	options.EXIF = nil
	options.GPS = nil
	options.XMP = nil
	options.IPTC = nil
	options.Description = nil
}

// snapshotJXRTileOptions 在共同容量内复制完整量化配置，不让首图回调改变后图的集合、复用和索引
// 入参: ctx Context, info 已校验原生描述, options 已合并基本选项, maximum 原配置与副本共同容量
// 返回: EncodeOptions 独占选项, uint64 量化原配置与副本的保守常驻容量, error 平面、量化、资源或取消错误
func snapshotJXRTileOptions(ctx context.Context, info Info, options EncodeOptions, maximum uint64) (EncodeOptions, uint64, error) {
	options = jxrResolveEncodeOptions(options)
	if (options.Alpha != nil || options.SeparateAlpha) && !info.Alpha {
		return options, 0, fmt.Errorf("%w: tiled image alpha options without alpha", ErrInvalid)
	}
	if err := checkJXRQuantizationGeometry(ctx, info, options); err != nil {
		return options, 0, err
	}
	work, err := jxrTileOptionsQuantizationWork(ctx, options, maximum)
	if err != nil {
		return options, 0, err
	}
	if work > maximum-work {
		return options, 0, &LimitError{Resource: "tiled image quantization snapshot memory", Limit: maximum, Required: jxrSaturatedAdd(work, work)}
	}
	metadata := Raster{info: info}
	main := PlaneOptions{Band: options.Band, Scaled: options.Scaled, Shift: options.Shift, Float: options.Float, Quantization: options.Quantization}
	_, err = makeJXREncodePlane(ctx, &metadata, main, false, options.Alpha == nil, options.Lossy, options.Overlap)
	if err != nil {
		return options, 0, err
	}
	if options.Float != nil {
		value := *options.Float
		options.Float = &value
	}
	clearJXRMergedMetadata(&options)
	if options.Window != nil {
		value := *options.Window
		options.Window = &value
	}
	if options.ChromaCenteringX != nil {
		value := *options.ChromaCenteringX
		options.ChromaCenteringX = &value
	}
	if options.ChromaCenteringY != nil {
		value := *options.ChromaCenteringY
		options.ChromaCenteringY = &value
	}
	if options.Quantization != nil {
		quantization, amount, err := snapshotJXRQuantization(ctx, options.Quantization, maximum-work, false)
		if err != nil {
			return options, 0, err
		}
		options.Quantization = quantization
		work += amount
	}
	if options.Alpha != nil {
		value := *options.Alpha
		_, err := makeJXREncodePlane(ctx, &metadata, value, true, false, options.Lossy, options.Overlap)
		if err != nil {
			return options, 0, err
		}
		if value.Float != nil {
			floating := *value.Float
			value.Float = &floating
		}
		if value.Quantization != nil {
			quantization, amount, err := snapshotJXRQuantization(ctx, value.Quantization, maximum-work, false)
			if err != nil {
				return options, 0, err
			}
			value.Quantization = quantization
			work += amount
		}
		options.Alpha = &value
	}
	return options, work, ctx.Err()
}

// writeJXRTileFileImageTo 在全文件坐标中只构建当前源的计划，完整包及字段完成后才链接前图
// 入参: ctx Context, w 当前输出, plan 当前独占描述, options 当前合并额度, start 当前目录, previous 前链接, maximum 全文件字节上限, last 是否末图
// 返回: uint64 文件末尾, uint64 当前链接位置, error 布局、资源、提供器、取消或I/O错误
func writeJXRTileFileImageTo(ctx context.Context, w io.WriteSeeker, plan *jxrTileImagePlan, options EncodeOptions, start, previous, maximum uint64, last bool) (uint64, uint64, error) {
	var end, link uint64
	var err error
	if options.SeparateAlpha {
		input, planErr := makeJXRSeparateTileInput(ctx, plan.info, options, plan.retained)
		if planErr != nil {
			return 0, 0, planErr
		}
		end, link, err = writeJXRSeparateTileAt(ctx, w, &input, start, plan.read)
	} else {
		metadata := Raster{info: plan.info}
		source, output, planErr := makeJXREncodingPlanRetained(ctx, &metadata, &options, plan.retained)
		if planErr != nil {
			return 0, 0, jxrPlanLimitError(planErr, output.stream.maxMemory, plan.retained)
		}
		end, link, err = writeJXRFileAt(ctx, w, plan.info, &source, output.stream, plan.retained, start, func(selected jxrStreamOutput) error {
			selected.maxMemory -= plan.retained
			return encodeJXRProvidedTilesTo(ctx, w, plan.info, &source, selected, plan.read)
		})
	}
	if err != nil {
		return 0, 0, err
	}
	return finishJXRImageOutput(ctx, w, start, end, previous, link, maximum, last, "tiled image file")
}

// encodeJXRImagesRows 逐图流式编码单遍行源与可选整图组合的多图JXR容器，每图写完后才链接前一目录
// 入参: ctx 已规范化Context, w 可定位输出, images 按目录顺序排列的图像输入, limits 全文件资源上限, maximum 全文件输出上限
// 返回: error 元数据、关系、能力、精度、资源、提供器、取消或I/O错误
func encodeJXRImagesRows(ctx context.Context, w io.WriteSeeker, images []ImageInput, limits Limits, maximum uint64) error {
	state := uint64(4096) + uint64(len(images))*jxrTileImagePlanWork
	plans, retained, err := planJXRRowImageFile(ctx, images, limits, state, maximum)
	if err != nil {
		return err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if position != 0 {
		return fmt.Errorf("%w: JPEG XR row image file output position", ErrInvalid)
	}
	end, previous := uint64(8), uint64(0)
	for index := range plans {
		if err := ctx.Err(); err != nil {
			return err
		}
		if end > maximum {
			return &LimitError{Resource: "JPEG XR row image file output size", Limit: maximum, Required: end}
		}
		plan := &plans[index]
		memory := limits.MaxMemoryBytes - state - (retained - plan.retained)
		options, err := jxrFileImageOptions(&plan.options, limits, memory, maximum-end+8)
		if err != nil {
			return err
		}
		if plan.raster != nil {
			end, previous, err = writeJXRFileImageTo(ctx, w, plan.raster, &options, end, previous, maximum, index == len(plans)-1)
		} else {
			end, previous, err = writeJXRRowFileImageTo(ctx, w, plan, options, end, previous, maximum, index == len(plans)-1)
		}
		if err != nil {
			return err
		}
	}
	return ctx.Err()
}

// planJXRRowImageFile 预检多图行源与整图条目的元数据、像素总量、页序列关系与独占选项快照
// 入参: ctx Context, images 多图输入, limits 全文件上限, state 计划固定内存, maximum 全文件输出上限
// 返回: []jxrTileImagePlan 独占快照, uint64 全部常驻额外内存, error 元数据、选项、关系、资源或取消错误
func planJXRRowImageFile(ctx context.Context, images []ImageInput, limits Limits, state, maximum uint64) ([]jxrTileImagePlan, uint64, error) {
	if state >= limits.MaxMemoryBytes {
		return nil, 0, &LimitError{Resource: "row image snapshot planning memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, 1)}
	}
	quantMaximum, quantWork := (limits.MaxMemoryBytes-state)/2, uint64(0)
	for _, input := range images {
		if input.Raster != nil || input.Options != nil && input.Options.Uncompressed {
			continue
		}
		var options EncodeOptions
		if input.Options != nil {
			options = *input.Options
		}
		amount, err := jxrTileOptionsQuantizationWork(ctx, options, quantMaximum-quantWork)
		if err != nil {
			return nil, 0, err
		}
		quantWork += amount
	}
	plans := make([]jxrTileImagePlan, len(images))
	retained, pixels := uint64(0), uint64(0)
	reserved := quantWork * 2
	for index, input := range images {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		if reserved >= limits.MaxMemoryBytes-state-retained {
			return nil, 0, &LimitError{Resource: "row image description and quantization memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, 1)))}
		}
		budget := limits.MaxMemoryBytes - state - retained - reserved
		options, err := jxrFileImageOptions(input.Options, limits, budget, maximum)
		if err != nil {
			return nil, 0, err
		}
		if err := checkJXREncodeOptions(options); err != nil {
			return nil, 0, err
		}
		if input.Raster != nil {
			work, err := jxrRasterWork(input.Raster, budget)
			if err != nil {
				return nil, 0, err
			}
			if work > budget {
				return nil, 0, &LimitError{Resource: "row image file combined inputs", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, work)))}
			}
			info := input.Raster.info
			if options.SeparateAlpha && !info.Alpha {
				return nil, 0, fmt.Errorf("%w: JPEG XR separate alpha without source alpha", ErrInvalid)
			}
			nextWork, required, ok := jxrContainerMetadataRetained(work, budget, info, &options)
			if !ok {
				return nil, 0, &LimitError{Resource: "row image file combined metadata", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, required)))}
			}
			work = nextWork
			qAmount, err := jxrTileOptionsQuantizationWork(ctx, options, budget-work)
			if err != nil {
				return nil, 0, err
			}
			work += qAmount
			width, height := uint64(input.Raster.Bounds().Dx()), uint64(input.Raster.Bounds().Dy())
			if height == 0 {
				return nil, 0, fmt.Errorf("%w: JPEG XR image dimensions", ErrInvalid)
			}
			if width > (limits.MaxPixels-pixels)/height {
				return nil, 0, &LimitError{Resource: "JPEG XR combined image pixels", Limit: limits.MaxPixels, Required: jxrSaturatedAdd(pixels, jxrSaturatedMul(width, height))}
			}
			pixels += width * height
			identifier, _, err := jxrRasterIdentifier(info, [16]byte{})
			if err != nil {
				return nil, 0, err
			}
			retained += work
			plans[index] = jxrTileImagePlan{info: info, raster: input.Raster, options: options, retained: work, identifier: identifier}
			continue
		}
		info, work, err := prepareJXRTileInfo(ctx, input.Info, options)
		if err != nil {
			return nil, 0, err
		}
		if options.SeparateAlpha && !info.Alpha {
			return nil, 0, fmt.Errorf("%w: JPEG XR separate alpha without source alpha", ErrInvalid)
		}
		clearJXRMergedMetadata(&options)
		if work > budget {
			return nil, 0, &LimitError{Resource: "row image file descriptions memory", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(state, jxrSaturatedAdd(retained, jxrSaturatedAdd(reserved, work)))}
		}
		retained += work
		width, height := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
		if width > (limits.MaxPixels-pixels)/height {
			return nil, 0, &LimitError{Resource: "combined row image pixels", Limit: limits.MaxPixels, Required: jxrSaturatedAdd(pixels, jxrSaturatedMul(width, height))}
		}
		pixels += width * height
		identifier, format, err := jxrRasterIdentifier(info, [16]byte{})
		if err != nil {
			return nil, 0, err
		}
		if !options.Uncompressed {
			resolved := jxrResolveEncodeOptions(options)
			if resolved.TileSize == (image.Point{}) {
				resolved.TileSize = jxrDefaultRowTileSize(info, format, resolved)
			}
			options = resolved
		}
		reader := input.Rows
		wrapped := func(ctx context.Context, y int, target *Raster) error {
			if err := reader(ctx, y, target); err != nil {
				return jxrCallbackError{err: err}
			}
			return nil
		}
		plans[index] = jxrTileImagePlan{info: info, rows: wrapped, options: options, retained: work, identifier: identifier}
	}
	if err := checkJXRImageSequence(ctx, len(plans), func(index int) (Info, [16]byte, Orientation) {
		return plans[index].info, plans[index].identifier, plans[index].options.Orientation
	}); err != nil {
		return nil, 0, err
	}
	for index := range plans {
		plan := &plans[index]
		if plan.raster != nil || plan.options.Uncompressed {
			continue
		}
		amount, err := jxrTileOptionsQuantizationWork(ctx, plan.options, quantWork)
		if err != nil {
			return nil, 0, err
		}
		quantWork -= amount
		options, work, err := snapshotJXRTileOptions(ctx, plan.info, plan.options, limits.MaxMemoryBytes-state-retained-quantWork*2)
		if err != nil {
			return nil, 0, err
		}
		plan.options, plan.retained = options, plan.retained+work
		retained += work
	}
	return plans, retained, ctx.Err()
}

// writeJXRRowFileImageTo 在多图文件指定目录位置按单遍行流编码单幅图像并完成偶数对齐与目录链接
// 入参: ctx Context, w 当前输出, plan 当前独占计划, options 当前合并选项, start 当前目录起点, previous 前一链接位置, maximum 全文件上限, last 是否末图
// 返回: uint64 文件末尾位置, uint64 当前链接位置, error 布局、资源、提供器、取消或I/O错误
func writeJXRRowFileImageTo(ctx context.Context, w io.WriteSeeker, plan *jxrTileImagePlan, options EncodeOptions, start, previous, maximum uint64, last bool) (uint64, uint64, error) {
	var end, link uint64
	var err error
	if options.Uncompressed {
		end, link, err = encodeJXRRowsUncompressedPreparedAt(ctx, w, plan.info, plan.retained, plan.rows, options, start)
	} else {
		limits := options.Limits.normalized()
		if plan.retained >= limits.MaxMemoryBytes {
			return 0, 0, &LimitError{Resource: "row image file retained metadata", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(plan.retained, 1)}
		}
		input := jxrStripContainerMetadata(plan.info)
		formatInfo, fmtErr := rasterPixelFormat(input)
		if fmtErr != nil {
			return 0, 0, fmtErr
		}
		var alpha jxrAlphaRowStorage
		if options.SeparateAlpha {
			var held uint64
			alpha, held, err = makeJXRAlphaRowStorage(input, options.Scratch, limits.MaxMemoryBytes-plan.retained)
			if err != nil {
				return 0, 0, err
			}
			limits.MaxMemoryBytes -= held
		}
		height := jxrRingHeight(input, formatInfo.color, options)
		cacheInfo, ringWork, ringErr := jxrRegionInfo(input, image.Rect(0, 0, input.Bounds.Dx(), height))
		if ringErr != nil {
			return 0, 0, ringErr
		}
		if ringWork >= limits.MaxMemoryBytes-plan.retained {
			return 0, 0, &LimitError{Resource: "row ring storage", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(plan.retained, jxrSaturatedAdd(ringWork, 1))}
		}
		limits.MaxMemoryBytes -= ringWork
		options.Limits = limits
		step := 1
		for _, component := range input.Components {
			step = max(step, int(component.YStep))
		}
		source := jxrRowInput{info: cacheInfo, read: plan.rows, height: input.Bounds.Dy(), step: step, maxWork: ringWork}
		if options.SeparateAlpha {
			end, link, err = encodeJXRSeparateRowsPreparedAt(ctx, w, plan.info, plan.retained, &source, &alpha, options, start)
		} else {
			metadata := Raster{info: plan.info}
			streamLayout, output, planErr := makeJXREncodingPlanRetained(ctx, &metadata, &options, plan.retained)
			if planErr != nil {
				return 0, 0, jxrPlanLimitError(planErr, output.stream.maxMemory, plan.retained)
			}
			output.stream.rowOrder = true
			end, link, err = writeJXRFileAt(ctx, w, plan.info, &streamLayout, output.stream, plan.retained, start, func(selected jxrStreamOutput) error {
				selected.maxMemory -= plan.retained
				return encodeJXRProvidedTilesTo(ctx, w, plan.info, &streamLayout, selected, source.load)
			})
		}
	}
	if err != nil {
		return 0, 0, err
	}
	return finishJXRImageOutput(ctx, w, start, end, previous, link, maximum, last, "row image file")
}
