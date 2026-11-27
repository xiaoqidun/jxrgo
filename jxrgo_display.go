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
	"math"
)

// ColorTransform 将单个像素的原生颜色分量samples及原透明度alpha显式转换为非预乘sRGB三通道
// samples不含透明度且保持原生数值域与预乘状态，切片仅在本次调用内有效不得保留，无透明度时alpha为一
// 返回值仅在最终显示输出时裁剪并量化，非nil错误直接中止显示转换
type ColorTransform func(samples []float64, alpha float64) (red, green, blue float64, err error)

// DisplayOptions 指定原生分量取样、显式颜色变换、ICC意图、HDR峰值亮度及输出限制，零值采用最近邻和内建基本显示策略
// Transform非nil时由调用方负责色彩和预乘；ApplyICC启用内建ICC颜色解释并按Intent选择渲染意图，Transform优先
// HDRPeakNits指定BT.2446方法A的HDR标称峰值亮度（零值对PQ默认10000、对HLG默认1000，不高于100 nit按相对100 nit物理裁剪，100至200 nit连续平滑过渡）；Limits计入输出及库内临时存储
type DisplayOptions struct {
	Filter      ResampleFilter
	Transform   ColorTransform
	Limits      Limits
	ApplyICC    bool
	Intent      RenderingIntent
	HDRPeakNits float64
}

// LinearOptions 指定线性浮点行读取的采样滤波、色彩管理、渲染意图、HLG峰值亮度、解预乘及资源限制
// opts为nil时默认采用双线性滤波；ApplyICC启用内建ICC，ApplyCICP启用PTM CICP，ApplyICC优先于ApplyCICP
// HDRPeakNits指定HLG逆传递与BT.2100 OOTF的参考显示峰值亮度（零值默认1000 nit），Unpremultiply为真时解除预乘，输出未裁剪D65线性scRGB
type LinearOptions struct {
	Filter        ResampleFilter
	ApplyCICP     bool
	ApplyICC      bool
	Unpremultiply bool
	Intent        RenderingIntent
	HDRPeakNits   float64
	Limits        Limits
}

// LinearReader 提供面向文档与HDR渲染引擎的按行线性scRGB浮点读取视图
// 每像素按顺序写入红、绿、蓝与透明度四个未裁剪float64值，保留负色域与大于一的高动态范围亮度
// 构造后完全不可变，当源Raster保持只读时支持多Goroutine安全并发调用ReadRow
type LinearReader struct {
	raster        *Raster
	info          Info
	components    []Component
	mapping       jxrDisplayMapping
	filter        ResampleFilter
	unpremultiply bool
	defaultLinear bool
	useNearest    bool
}

// RGBOptions 指定设备域非线性RGB行读取的采样滤波、CICP矩阵解码、解预乘及资源限制
// opts为nil时默认采用双线性滤波；ApplyCICPMatrix为真时对YCC按PTM CICP矩阵与范围解码（无声明或矩阵未指定时回退默认BT.601）而不执行逆传递线性化或基色转换
// 忽略内嵌ICC与ColorHintUnspecified，保留负色域与大于一扩展范围供外部色彩管理管线使用
type RGBOptions struct {
	Filter          ResampleFilter
	ApplyCICPMatrix bool
	Unpremultiply   bool
	Limits          Limits
}

// RGBReader 提供面向外部色彩管理管线的按行设备域RGB浮点读取视图
// 每像素按顺序写入红、绿、蓝与透明度四个未裁剪float64值，对YCC完成相位上采样与矩阵解码而不执行逆传递线性化
// 构造后完全不可变，当源Raster保持只读时支持多Goroutine安全并发调用ReadRow
type RGBReader struct {
	raster        *Raster
	info          Info
	components    []Component
	ycc           jxrYCCDecoder
	hasYCC        bool
	filter        ResampleFilter
	unpremultiply bool
	useNearest    bool
}

// ColorConverter 持有预解析的只读无锁ICC、CICP与HDR显示转换管线，供跨行带低内存逐行复用
// 构造后完全不可变，当源Raster保持只读时支持多Goroutine安全并发调用各逐行与跨行带转换方法
type ColorConverter struct {
	info      Info
	mapping   jxrDisplayMapping
	filter    ResampleFilter
	transform ColorTransform
}

// StripRegion 保存条带在完整显示网格中的矩形范围及其原生样本视图，供全方向无缝跨带转换定位上下或左右邻域
type StripRegion struct {
	Region image.Rectangle
	Raster *Raster
}

// jxrStripNeighbors 保存经显示几何解算后的上下或左右相邻条带指针，单次转换至多沿单轴存在邻带
type jxrStripNeighbors struct {
	top    *Raster
	bottom *Raster
	left   *Raster
	right  *Raster
}

// Display 按分量相位将原生图像绘制为十六位非预乘sRGB图像，不修改原图也不构造整幅重采样中间图
// 未知色度相位按ITU-T T.832默认居中相位处理；色彩按Transform优先、ApplyICC内建ICC或默认CICP与BT.2446方法A策略转换
// 入参: ctx Context，nil采用Background, opts 显示策略及限制，nil采用默认值
// 返回: *image.NRGBA64 独立显示图像, error 参数、颜色、数值、资源、变换或取消错误
func (r *Raster) Display(ctx context.Context, opts *DisplayOptions) (*image.NRGBA64, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	img, err := r.display(ctx, opts)
	if err != nil {
		return nil, wrapJXRError("Raster.Display", err)
	}
	return img, nil
}

// display 执行原生分量的十六位sRGB显示转换，由公开入口统一归一化错误
// 入参: ctx 已规范化Context, opts 显示策略及限制
// 返回: *image.NRGBA64 独立显示图像, error 参数、颜色、数值、资源、变换或取消错误
func (r *Raster) display(ctx context.Context, opts *DisplayOptions) (*image.NRGBA64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Bounds().Empty() || r.info.Channels < 1 || r.info.Channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: display raster", ErrInvalid)
	}
	filter, limits, intent, peakNits := ResampleNearest, DefaultLimits(), IntentRelative, 0.0
	var transform ColorTransform
	if opts != nil {
		filter, limits, transform, intent, peakNits = opts.Filter, opts.Limits.normalized(), opts.Transform, opts.Intent, opts.HDRPeakNits
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: display filter", ErrInvalid)
	}
	if intent > IntentAbsolute {
		return nil, fmt.Errorf("%w: display rendering intent", ErrInvalid)
	}
	if !geometryFinite(peakNits) || peakNits < 0 {
		return nil, fmt.Errorf("%w: display HDR peak luminance", ErrInvalid)
	}
	info := r.info
	bounds := info.Bounds
	count := info.Channels
	if info.Alpha {
		count++
	}
	if count != len(info.Components) {
		return nil, fmt.Errorf("%w: display components", ErrInvalid)
	}
	managed := transform == nil && opts != nil && opts.ApplyICC
	var mapping jxrDisplayMapping
	if transform == nil && !managed {
		var err error
		mapping, err = jxrBasicRasterDisplay(info, peakNits)
		if err != nil {
			return nil, err
		}
	}
	pixels := uint64(bounds.Dx()) * uint64(bounds.Dy())
	scratch := uint64(512) + uint64(count)*136
	if pixels > limits.MaxPixels {
		return nil, &LimitError{Resource: "max pixels", Limit: limits.MaxPixels, Required: pixels, Detail: "display pixels"}
	}
	maxInt := uint64(^uint(0) >> 1)
	if pixels > maxInt/8 {
		return nil, &LimitError{Resource: "max memory bytes", Limit: maxInt, Required: jxrSaturatedMul(pixels, 8), Detail: "display output memory"}
	}
	if scratch > limits.MaxMemoryBytes || pixels > (limits.MaxMemoryBytes-scratch)/8 {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(scratch, jxrSaturatedMul(pixels, 8)), Detail: "display working memory"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if managed {
		var err error
		mapping, err = jxrManagedRasterDisplay(ctx, info, limits.MaxMemoryBytes-scratch-pixels*8, peakNits, intent)
		if err != nil {
			return nil, err
		}
	}
	return renderJXRDisplay(ctx, r, info, mapping, filter, transform)
}

// NewLinearReader 为原生图像创建按行读取未裁剪线性scRGB浮点像素的读取器，不修改原图也不分配整幅输出缓冲
// 支持灰度、RGB、全十六种YCC子采样及CMYK与多通道ICC图像，未知色度相位按标准居中相位处理
// 入参: ctx Context，nil采用Background, opts 线性读取配置，nil默认双线性滤波
// 返回: *LinearReader 线性行读取器, error 参数、颜色、资源或取消错误
func (r *Raster) NewLinearReader(ctx context.Context, opts *LinearOptions) (*LinearReader, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	reader, err := r.newLinearReader(ctx, opts)
	if err != nil {
		return nil, wrapJXRError("Raster.NewLinearReader", err)
	}
	return reader, nil
}

// newLinearReader 执行LinearReader的预算校验、色彩管线装配与分量绑定，由公开入口归一化错误
// 入参: ctx 已规范化Context, opts 线性读取配置
// 返回: *LinearReader 线性行读取器, error 参数、颜色、资源或取消错误
func (r *Raster) newLinearReader(ctx context.Context, opts *LinearOptions) (*LinearReader, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Bounds().Empty() || r.info.Channels < 1 || r.info.Channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: linear reader raster", ErrInvalid)
	}
	filter, limits, intent, peakNits := ResampleBilinear, DefaultLimits(), IntentRelative, 0.0
	applyCICP, applyICC, unpremultiply := false, false, false
	if opts != nil {
		filter, limits, intent, peakNits = opts.Filter, opts.Limits.normalized(), opts.Intent, opts.HDRPeakNits
		applyCICP, applyICC, unpremultiply = opts.ApplyCICP, opts.ApplyICC, opts.Unpremultiply
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: linear reader filter", ErrInvalid)
	}
	if intent > IntentAbsolute {
		return nil, fmt.Errorf("%w: linear reader rendering intent", ErrInvalid)
	}
	if !geometryFinite(peakNits) || peakNits < 0 {
		return nil, fmt.Errorf("%w: linear reader HDR peak luminance", ErrInvalid)
	}
	info := r.info
	count := info.Channels
	if info.Alpha {
		count++
	}
	if count != len(info.Components) {
		return nil, fmt.Errorf("%w: linear reader components", ErrInvalid)
	}
	scratch := uint64(512) + uint64(count)*136
	if scratch > limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: scratch, Detail: "linear reader working memory"}
	}
	var mapping jxrDisplayMapping
	defaultLinear := false
	switch {
	case applyICC && len(info.ICCProfile) != 0:
		icc, err := jxrReadICCTransform(ctx, info.ICCProfile, info, limits.MaxMemoryBytes-scratch, intent)
		if err != nil {
			return nil, err
		}
		mapping.icc = icc
	case applyCICP && info.ColorEncoding != nil:
		var supported bool
		mapping, supported = jxrCICPDisplay(info, *info.ColorEncoding, peakNits)
		if !supported {
			return nil, fmt.Errorf("%w: linear CICP color interpretation", ErrUnsupported)
		}
	default:
		if info.ColorHint == ColorHintUnspecified {
			return nil, fmt.Errorf("%w: unspecified linear color space", ErrUnsupported)
		}
		switch info.ColorSpace {
		case ColorGray:
			if info.Channels != 1 {
				return nil, fmt.Errorf("%w: linear gray channels", ErrUnsupported)
			}
			defaultLinear = true
		case ColorRGB:
			if info.Channels != 3 {
				return nil, fmt.Errorf("%w: linear RGB channels", ErrUnsupported)
			}
			defaultLinear = true
		case ColorYCC:
			var supported bool
			mapping, supported = jxrCICPDisplay(info, ColorEncoding{Primaries: 1, Transfer: 13, Matrix: 5, FullRange: true}, 0)
			if !supported {
				return nil, fmt.Errorf("%w: linear YCC channels", ErrUnsupported)
			}
		case ColorCMYK, ColorCMYKDirect:
			if info.Channels != 4 {
				return nil, fmt.Errorf("%w: linear CMYK channels", ErrUnsupported)
			}
			defaultLinear = true
		default:
			return nil, fmt.Errorf("%w: linear color interpretation", ErrUnsupported)
		}
	}
	components := make([]Component, count)
	for index := range count {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var err error
		components[index], err = r.component(index)
		if err != nil {
			return nil, err
		}
	}
	return &LinearReader{
		raster:        r,
		info:          info,
		components:    components,
		mapping:       mapping,
		filter:        filter,
		unpremultiply: unpremultiply,
		defaultLinear: defaultLinear,
		useNearest:    filter == ResampleNearest && info.Bounds.Min == (image.Point{}),
	}, ctx.Err()
}

// ReadRow 读取指定显示行并按每像素红、绿、蓝、透明度四个float64写入dst，输出未裁剪的D65线性scRGB
// dst长度须不少于图像宽度乘以四，当源Raster保持只读时支持多Goroutine安全并发调用
// 入参: ctx Context，nil采用Background, y 目标行坐标, dst 输出浮点切片
// 返回: error 参数、坐标、数值或取消错误
func (lr *LinearReader) ReadRow(ctx context.Context, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := lr.readRow(ctx, y, dst); err != nil {
		return wrapJXRError("LinearReader.ReadRow", err)
	}
	return nil
}

// readRow 执行单行原生采样、线性色彩转换及预乘处理，由公开入口归一化错误
// 入参: ctx 已规范化Context, y 目标行坐标, dst 输出浮点切片
// 返回: error 参数、坐标、数值或取消错误
func (lr *LinearReader) readRow(ctx context.Context, y int, dst []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if lr == nil || lr.raster == nil {
		return fmt.Errorf("%w: nil linear reader", ErrInvalid)
	}
	bounds := lr.info.Bounds
	width, height := bounds.Dx(), bounds.Dy()
	row := y - bounds.Min.Y
	if row < 0 || row >= height || int64(len(dst)) < int64(width)*4 {
		return fmt.Errorf("%w: linear reader row or destination", ErrInvalid)
	}
	count := len(lr.components)
	var valueStorage [9]float64
	values := valueStorage[:]
	if count > len(values) {
		values = make([]float64, count)
	} else {
		values = values[:count]
	}
	for x := range width {
		if x&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if lr.useNearest {
			if err := jxrDisplayNearestPixel(lr.raster, x, row, values); err != nil {
				return err
			}
		} else {
			for index, component := range lr.components {
				var err error
				values[index], err = displayJXRSample(component, x, row, width, height, lr.filter)
				if err != nil {
					return err
				}
			}
		}
		alpha := 1.0
		if lr.info.Alpha {
			alpha = values[lr.info.Channels]
		}
		srcLinear := lr.mapping.linear
		if lr.defaultLinear {
			srcLinear = lr.info.Linear
		}
		red, green, blue, err := lr.mapping.linearColor(ctx, lr.info, values, alpha, srcLinear, lr.unpremultiply)
		if err != nil {
			return err
		}
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = red, green, blue, alpha
	}
	return ctx.Err()
}

// linearColor 将单个已采样原生像素转换为未裁剪的D65线性scRGB三通道，供LinearReader与ColorConverter复用
// 入参: ctx 已规范化Context, info 原生描述, values 可复用原生样本工作区, alpha 当前像素透明度, srcLinear 默认分支源是否线性光, unpremultiply 是否解除预乘
// 返回: float64 线性红、绿、蓝scRGB值, error ICC数值或取消错误
func (mapping *jxrDisplayMapping) linearColor(ctx context.Context, info Info, values []float64, alpha float64, srcLinear, unpremultiply bool) (float64, float64, float64, error) {
	if mapping.icc.channels != 0 {
		repremultiply := false
		if info.Premultiplied {
			if !(alpha > 0) || !geometryFinite(alpha) {
				return 0, 0, 0, nil
			}
			for index := range info.Channels {
				values[index] /= alpha
			}
			repremultiply = !unpremultiply
		}
		red, green, blue, err := mapping.icc.linearColor(ctx, values)
		if err != nil {
			return 0, 0, 0, err
		}
		if repremultiply {
			red *= alpha
			green *= alpha
			blue *= alpha
		}
		return red, green, blue, nil
	}
	if mapping.cicp.transfer != 0 {
		red, green, blue := mapping.cicp.linearColor(info, values, alpha, unpremultiply)
		return red, green, blue, nil
	}
	if srcLinear && (info.ColorSpace == ColorGray || info.ColorSpace == ColorRGB) {
		var red, green, blue float64
		if info.ColorSpace == ColorGray {
			red, green, blue = values[0], values[0], values[0]
		} else {
			red, green, blue = values[0], values[1], values[2]
		}
		if info.Premultiplied && unpremultiply {
			if !(alpha > 0) || !geometryFinite(alpha) {
				return 0, 0, 0, nil
			}
			red /= alpha
			green /= alpha
			blue /= alpha
		}
		return red, green, blue, nil
	}
	repremultiply := false
	if info.Premultiplied {
		if !(alpha > 0) || !geometryFinite(alpha) {
			return 0, 0, 0, nil
		}
		for index := range info.Channels {
			values[index] /= alpha
		}
		repremultiply = !unpremultiply
	}
	var red, green, blue float64
	switch info.ColorSpace {
	case ColorGray:
		v := jxrCICPInverseTransfer(values[0], 13, true)
		red, green, blue = v, v, v
	case ColorRGB:
		red = jxrCICPInverseTransfer(values[0], 13, true)
		green = jxrCICPInverseTransfer(values[1], 13, true)
		blue = jxrCICPInverseTransfer(values[2], 13, true)
	case ColorCMYK, ColorCMYKDirect:
		red = jxrCICPInverseTransfer((1-values[0])*(1-values[3]), 13, true)
		green = jxrCICPInverseTransfer((1-values[1])*(1-values[3]), 13, true)
		blue = jxrCICPInverseTransfer((1-values[2])*(1-values[3]), 13, true)
	}
	if repremultiply {
		red *= alpha
		green *= alpha
		blue *= alpha
	}
	return red, green, blue, nil
}

// NewRGBReader 为原生图像创建按行读取未裁剪设备域RGB浮点像素的读取器，不执行逆传递线性化或内嵌ICC转换
// 支持灰度、RGB、全十六种YCC子采样及CMYK，对YCC完成相位上采样与矩阵解码，供外部色彩管理管线套用外部ICC
// 入参: ctx Context，nil采用Background, opts 设备RGB读取配置，nil默认双线性滤波与T.832默认BT.601矩阵
// 返回: *RGBReader 设备RGB行读取器, error 参数、颜色、资源或取消错误
func (r *Raster) NewRGBReader(ctx context.Context, opts *RGBOptions) (*RGBReader, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	reader, err := r.newRGBReader(ctx, opts)
	if err != nil {
		return nil, wrapJXRError("Raster.NewRGBReader", err)
	}
	return reader, nil
}

// newRGBReader 执行RGBReader的参数预算校验、YCC矩阵解码器准备与分量绑定，由公开入口归一化错误
// 入参: ctx 已规范化Context, opts 设备RGB读取配置
// 返回: *RGBReader 设备RGB行读取器, error 参数、颜色、资源或取消错误
func (r *Raster) newRGBReader(ctx context.Context, opts *RGBOptions) (*RGBReader, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Bounds().Empty() || r.info.Channels < 1 || r.info.Channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: RGB reader raster", ErrInvalid)
	}
	filter, limits := ResampleBilinear, DefaultLimits()
	applyCICPMatrix, unpremultiply := false, false
	if opts != nil {
		filter, limits = opts.Filter, opts.Limits.normalized()
		applyCICPMatrix, unpremultiply = opts.ApplyCICPMatrix, opts.Unpremultiply
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: RGB reader filter", ErrInvalid)
	}
	info := r.info
	count := info.Channels
	if info.Alpha {
		count++
	}
	if count != len(info.Components) {
		return nil, fmt.Errorf("%w: RGB reader components", ErrInvalid)
	}
	scratch := uint64(512) + uint64(count)*136
	if scratch > limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: scratch, Detail: "RGB reader working memory"}
	}
	var ycc jxrYCCDecoder
	hasYCC := false
	switch info.ColorSpace {
	case ColorGray:
		if info.Channels != 1 {
			return nil, fmt.Errorf("%w: RGB reader gray channels", ErrUnsupported)
		}
	case ColorRGB:
		if info.Channels != 3 {
			return nil, fmt.Errorf("%w: RGB reader RGB channels", ErrUnsupported)
		}
	case ColorYCC:
		var encoding *ColorEncoding
		if applyCICPMatrix {
			encoding = info.ColorEncoding
		}
		var ok bool
		ycc, ok = jxrNewYCCDecoder(info, encoding)
		if !ok {
			return nil, fmt.Errorf("%w: RGB reader YCC layout", ErrUnsupported)
		}
		hasYCC = true
	case ColorCMYK, ColorCMYKDirect:
		if info.Channels != 4 {
			return nil, fmt.Errorf("%w: RGB reader CMYK channels", ErrUnsupported)
		}
	default:
		return nil, fmt.Errorf("%w: RGB reader color interpretation", ErrUnsupported)
	}
	components := make([]Component, count)
	for index := range count {
		var err error
		components[index], err = r.component(index)
		if err != nil {
			return nil, err
		}
	}
	return &RGBReader{
		raster:        r,
		info:          info,
		components:    components,
		ycc:           ycc,
		hasYCC:        hasYCC,
		filter:        filter,
		unpremultiply: unpremultiply,
		useNearest:    filter == ResampleNearest && info.Bounds.Min == (image.Point{}),
	}, ctx.Err()
}

// ReadRow 读取指定显示行并按每像素红、绿、蓝、透明度四个float64写入dst，保持设备非线性编码且不裁剪
// dst长度须不少于图像宽度乘以四，当源Raster保持只读时支持多Goroutine安全并发调用
// 入参: ctx Context，nil采用Background, y 目标行坐标, dst 输出浮点切片
// 返回: error 参数、坐标、数值或取消错误
func (rr *RGBReader) ReadRow(ctx context.Context, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := rr.readRow(ctx, y, dst); err != nil {
		return wrapJXRError("RGBReader.ReadRow", err)
	}
	return nil
}

// readRow 执行单行原生采样、YCC矩阵解码及可选解预乘，由公开入口归一化错误
// 入参: ctx 已规范化Context, y 目标行坐标, dst 输出浮点切片
// 返回: error 参数、坐标、数值或取消错误
func (rr *RGBReader) readRow(ctx context.Context, y int, dst []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if rr == nil || rr.raster == nil {
		return fmt.Errorf("%w: nil RGB reader", ErrInvalid)
	}
	bounds := rr.info.Bounds
	width, height := bounds.Dx(), bounds.Dy()
	row := y - bounds.Min.Y
	if row < 0 || row >= height || int64(len(dst)) < int64(width)*4 {
		return fmt.Errorf("%w: RGB reader row or destination", ErrInvalid)
	}
	count := len(rr.components)
	var valueStorage [5]float64
	values := valueStorage[:count]
	for x := range width {
		if x&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if rr.useNearest {
			if err := jxrDisplayNearestPixel(rr.raster, x, row, values); err != nil {
				return err
			}
		} else {
			for index, component := range rr.components {
				var err error
				values[index], err = displayJXRSample(component, x, row, width, height, rr.filter)
				if err != nil {
					return err
				}
			}
		}
		alpha := 1.0
		if rr.info.Alpha {
			alpha = values[rr.info.Channels]
		}
		red, green, blue := rr.rgbPixel(values, alpha)
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = red, green, blue, alpha
	}
	return ctx.Err()
}

// rgbPixel 将单个已采样原生像素转换为未裁剪的设备域RGB三通道
// 入参: values 可复用原生样本工作区, alpha 当前像素透明度
// 返回: float64 设备域红、绿、蓝值
func (rr *RGBReader) rgbPixel(values []float64, alpha float64) (float64, float64, float64) {
	info := rr.info
	repremultiply := false
	if info.Premultiplied && (rr.unpremultiply || rr.hasYCC || info.ColorSpace == ColorCMYK || info.ColorSpace == ColorCMYKDirect) {
		if !(alpha > 0) || !geometryFinite(alpha) {
			return 0, 0, 0
		}
		for index := range info.Channels {
			values[index] /= alpha
		}
		repremultiply = !rr.unpremultiply
	}
	var red, green, blue float64
	if rr.hasYCC {
		rgb := rr.ycc.rgb(values)
		red, green, blue = rgb[0], rgb[1], rgb[2]
	} else {
		switch info.ColorSpace {
		case ColorGray:
			red, green, blue = values[0], values[0], values[0]
		case ColorRGB:
			red, green, blue = values[0], values[1], values[2]
		case ColorCMYK, ColorCMYKDirect:
			red = (1 - values[0]) * (1 - values[3])
			green = (1 - values[1]) * (1 - values[3])
			blue = (1 - values[2]) * (1 - values[3])
		}
	}
	if repremultiply {
		red *= alpha
		green *= alpha
		blue *= alpha
	}
	return red, green, blue
}

// renderJXRDisplay 共用标准解码与原生显示转换的直通整数快路径与通用采样渲染循环
// 入参: ctx 已规范化Context, r 原生图像, info 图像元数据, mapping 已选显示映射, filter 采样策略, transform 可选显式色彩变换
// 返回: *image.NRGBA64 独立十六位非预乘显示图像, error 采样、颜色、数值、变换或取消错误
func renderJXRDisplay(ctx context.Context, r *Raster, info Info, mapping jxrDisplayMapping, filter ResampleFilter, transform ColorTransform) (*image.NRGBA64, error) {
	bounds := info.Bounds
	if filter == ResampleNearest && transform == nil {
		if precision, interleaved, ok := canDisplayJXRDirect(r, info, mapping); ok {
			output := image.NewNRGBA64(bounds)
			if err := displayJXRDirect(ctx, r, info, precision, interleaved, output); err != nil {
				return nil, err
			}
			return output, nil
		}
	}
	count := info.Channels
	if info.Alpha {
		count++
	}
	var compStorage [9]Component
	var valueStorage [9]float64
	components := compStorage[:]
	values := valueStorage[:]
	if count > len(components) {
		components = make([]Component, count)
		values = make([]float64, count)
	} else {
		components = components[:count]
		values = values[:count]
	}
	for index := range count {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var err error
		components[index], err = r.component(index)
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	output := image.NewNRGBA64(bounds)
	useNearestFast := filter == ResampleNearest && bounds.Min == (image.Point{})
	for y := range bounds.Dy() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for x := range bounds.Dx() {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if useNearestFast {
				if err := jxrDisplayNearestPixel(r, x, y, values); err != nil {
					return nil, err
				}
			} else {
				for index, component := range components {
					var err error
					values[index], err = displayJXRSample(component, x, y, bounds.Dx(), bounds.Dy(), filter)
					if err != nil {
						return nil, err
					}
				}
			}
			alpha := 1.0
			if info.Alpha {
				alpha = values[info.Channels]
			}
			var red, green, blue float64
			if transform != nil {
				var err error
				red, green, blue, err = transform(values[:info.Channels:info.Channels], alpha)
				if err != nil {
					return nil, jxrCallbackError{err: err}
				}
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			} else {
				var err error
				red, green, blue, err = mapping.color(ctx, info, values, alpha)
				if err != nil {
					return nil, err
				}
			}
			output.SetNRGBA64(bounds.Min.X+x, bounds.Min.Y+y, color.NRGBA64{R: displayWord(red), G: displayWord(green), B: displayWord(blue), A: displayWord(alpha)})
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// jxrDisplayNearestPixel 按原生相位取得整幅显示网格上的分量，兼容交织和独立色度
// 入参: raster 原生图像, x,y 零起点的显示坐标, values 可复用分量工作区
// 返回: error 采样位置或索引错误
func jxrDisplayNearestPixel(raster *Raster, x, y int, values []float64) error {
	width, height := raster.info.Bounds.Dx(), raster.info.Bounds.Dy()
	at := y*width + x
	if raster.native != nil {
		base := at * len(values)
		for index := range values {
			values[index] = sampleValue(raster.native.sampleBits(base+index), raster.info.Components[index])
		}
		return nil
	}
	for index := range values {
		info := raster.info.Components[index]
		plane := &raster.planes[index]
		if info.XStep == 1 && info.YStep == 1 {
			values[index] = sampleValue(plane.sampleBits(at), info)
			continue
		}
		var err error
		values[index], err = displayJXRSample(Component{source: plane, info: info}, x, y, width, height, ResampleNearest)
		if err != nil {
			return err
		}
	}
	return nil
}

// displayJXRSample 在显示网格取分量数值，不在双线性计算后重新量化原生编码
// 入参: source 原生分量, x,y 显示坐标, width,height 显示尺寸, filter 采样策略
// 返回: float64 原生分量数值, error 样本或非有限插值错误
func displayJXRSample(source Component, x, y, width, height int, filter ResampleFilter) (float64, error) {
	info := source.info
	x0, x1, fx := resampleJXRAxis(x, width, width, info.Bounds.Dx(), jxrEffectiveOffset(info.XOffset, info.XStep, info.XPositionKnown), info.XStep)
	y0, y1, fy := resampleJXRAxis(y, height, height, info.Bounds.Dy(), jxrEffectiveOffset(info.YOffset, info.YStep, info.YPositionKnown), info.YStep)
	if filter == ResampleNearest {
		if fx >= 0.5 {
			x0 = x1
		}
		if fy >= 0.5 {
			y0 = y1
		}
		return source.sampleAt(info.Bounds.Min.X+x0, info.Bounds.Min.Y+y0)
	}
	if fx == 0 && fy == 0 {
		return source.sampleAt(info.Bounds.Min.X+x0, info.Bounds.Min.Y+y0)
	}
	var values [4]float64
	for index, point := range [4]image.Point{{X: x0, Y: y0}, {X: x1, Y: y0}, {X: x0, Y: y1}, {X: x1, Y: y1}} {
		if index&1 != 0 && fx == 0 || index&2 != 0 && fy == 0 {
			continue
		}
		value, err := source.sampleAt(info.Bounds.Min.X+point.X, info.Bounds.Min.Y+point.Y)
		if err != nil {
			return 0, err
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("%w: nonfinite display interpolation", ErrInvalid)
		}
		values[index] = value
	}
	return (values[0]*(1-fx)+values[1]*fx)*(1-fy) + (values[2]*(1-fx)+values[3]*fx)*fy, nil
}

// displayJXRColor 按基本显示策略解释无ICC的分量，不应用调用方变换或猜测其他颜色空间
// 入参: info 原生信息, values 可复用颜色工作值, alpha 原透明度, linear 是否转为sRGB
// 返回: float64 非预乘红、绿、蓝显示值
func displayJXRColor(info Info, values []float64, alpha float64, linear bool) (float64, float64, float64) {
	if info.Premultiplied {
		if !(alpha > 0) || !geometryFinite(alpha) {
			return 0, 0, 0
		}
		jxrUnpremultiplyDisplay(values[:info.Channels], alpha)
	}
	var red, green, blue float64
	switch info.ColorSpace {
	case ColorGray:
		red, green, blue = values[0], values[0], values[0]
	case ColorRGB:
		red, green, blue = values[0], values[1], values[2]
	case ColorCMYK, ColorCMYKDirect:
		red = (1 - values[0]) * (1 - values[3])
		green = (1 - values[1]) * (1 - values[3])
		blue = (1 - values[2]) * (1 - values[3])
		if info.Components[0].Precision == 8 {
			c, m, y, k := uint8(math.Round(colorClamp(values[0])*255)), uint8(math.Round(colorClamp(values[1])*255)), uint8(math.Round(colorClamp(values[2])*255)), uint8(math.Round(colorClamp(values[3])*255))
			r, g, b, _ := (color.CMYK{C: c, M: m, Y: y, K: k}).RGBA()
			red, green, blue = float64(r)/65535, float64(g)/65535, float64(b)/65535
		}
	}
	if linear {
		red, green, blue = displayLinear(red), displayLinear(green), displayLinear(blue)
	}
	return red, green, blue
}

// jxrUnpremultiplyDisplay 按原透明度恢复非预乘设备分量，非正或非有限透明度清零颜色，有限正数直接相除
// 入参: values 原生颜色工作值, alpha 原透明度
func jxrUnpremultiplyDisplay(values []float64, alpha float64) {
	if !(alpha > 0) || !geometryFinite(alpha) {
		clear(values)
		return
	}
	for index := range values {
		values[index] /= alpha
	}
}

// canDisplayJXRDirect 检查原生图像是否满足无变换、无子采样、非预乘的整数直通显示条件
// 入参: raster 原生图像, info 图像元数据, mapping 已选显示映射
// 返回: uint8 分量位深, bool 是否为交织存储, bool 是否可走整数快路径
func canDisplayJXRDirect(raster *Raster, info Info, mapping jxrDisplayMapping) (uint8, bool, bool) {
	if raster == nil || mapping.linear || mapping.cicp.transfer != 0 || mapping.icc.channels != 0 || info.Premultiplied {
		return 0, false, false
	}
	bounds := info.Bounds
	if bounds.Empty() || bounds.Min != (image.Point{}) {
		return 0, false, false
	}
	switch info.ColorSpace {
	case ColorGray:
		if info.Channels != 1 {
			return 0, false, false
		}
	case ColorRGB:
		if info.Channels != 3 {
			return 0, false, false
		}
	case ColorCMYK, ColorCMYKDirect:
		if info.Channels != 4 {
			return 0, false, false
		}
	default:
		return 0, false, false
	}
	stride := info.Channels
	if info.Alpha {
		stride++
	}
	if len(info.Components) != stride {
		return 0, false, false
	}
	precision := info.Components[0].Precision
	if info.ColorSpace == ColorCMYK || info.ColorSpace == ColorCMYKDirect {
		if precision != 8 {
			return 0, false, false
		}
	} else if precision != 8 && precision != 16 {
		return 0, false, false
	}
	for _, component := range info.Components {
		if component.Format != SampleUnsigned || component.Inverted || component.Precision != precision ||
			component.XStep != 1 || component.YStep != 1 ||
			jxrEffectiveOffset(component.XOffset, component.XStep, component.XPositionKnown) != 0 ||
			jxrEffectiveOffset(component.YOffset, component.YStep, component.YPositionKnown) != 0 ||
			component.Bounds != bounds {
			return 0, false, false
		}
	}
	pixels := bounds.Dx() * bounds.Dy()
	if raster.native != nil && len(raster.planes) == 0 {
		n := raster.native
		if n.words != nil || n.channels != info.Channels || n.alpha != info.Alpha || n.bounds != bounds || len(n.values) < pixels*stride {
			return 0, false, false
		}
		return precision, true, true
	}
	if raster.native == nil && len(raster.planes) == stride {
		for index := range stride {
			p := &raster.planes[index]
			if p.words != nil || p.channels != 1 || p.alpha || p.bounds != bounds || len(p.values) < pixels {
				return 0, false, false
			}
		}
		return precision, false, true
	}
	return 0, false, false
}

// displayJXRDirect 用纯整数算术直接填充非预乘十六位标准图像缓冲区，与通用浮点路径逐位一致
// 入参: ctx Context, raster 原生图像, info 图像元数据, precision 分量位深, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirect(ctx context.Context, raster *Raster, info Info, precision uint8, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		displayJXRDirectRow64(raster, info, precision, interleaved, y, width, output.Pix[y*output.Stride:y*output.Stride+width*8])
	}
	return ctx.Err()
}

// NewColorConverter 预解析并构造可跨多个DecodeStrips回调复用的只读无锁行级色彩转换器
// 独立持有颜色元数据及ICC配置副本，不重复解析ICC查表或浮点管线；构造后支持多Goroutine安全并发调用
// 入参: ctx Context，nil采用Background, info 原生图像元数据, opts 显示选项，nil采用默认最近邻与基本显示策略
// 返回: *ColorConverter 可复用行级色彩转换器, error 参数、颜色、资源或取消错误
func NewColorConverter(ctx context.Context, info Info, opts *DisplayOptions) (*ColorConverter, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	conv, err := newColorConverter(ctx, info, opts)
	if err != nil {
		return nil, wrapJXRError("NewColorConverter", err)
	}
	return conv, nil
}

// newColorConverter 执行ColorConverter的参数校验、预算检查与色彩管线预解析，由公开入口归一化错误
// 入参: ctx 已规范化Context, info 原生图像元数据, opts 显示选项
// 返回: *ColorConverter 行级色彩转换器, error 参数、颜色、资源或取消错误
func newColorConverter(ctx context.Context, info Info, opts *DisplayOptions) (*ColorConverter, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if info.Channels < 1 || info.Channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: color converter channels", ErrInvalid)
	}
	count := info.Channels
	if info.Alpha {
		count++
	}
	if count != len(info.Components) {
		return nil, fmt.Errorf("%w: color converter components", ErrInvalid)
	}
	for _, comp := range info.Components {
		if comp.XStep == 0 || comp.YStep == 0 || comp.Precision == 0 {
			return nil, fmt.Errorf("%w: color converter component layout", ErrInvalid)
		}
	}
	filter, limits, intent, peakNits := ResampleNearest, DefaultLimits(), IntentRelative, 0.0
	var transform ColorTransform
	if opts != nil {
		filter, limits, transform, intent, peakNits = opts.Filter, opts.Limits.normalized(), opts.Transform, opts.Intent, opts.HDRPeakNits
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: color converter filter", ErrInvalid)
	}
	if intent > IntentAbsolute {
		return nil, fmt.Errorf("%w: color converter rendering intent", ErrInvalid)
	}
	if !geometryFinite(peakNits) || peakNits < 0 {
		return nil, fmt.Errorf("%w: color converter HDR peak luminance", ErrInvalid)
	}
	managed := transform == nil && opts != nil && opts.ApplyICC
	scratch := uint64(512) + uint64(count)*136
	if managed && len(info.ICCProfile) != 0 {
		scratch = jxrSaturatedAdd(scratch, uint64(len(info.ICCProfile)))
	}
	if scratch > limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: scratch, Detail: "color converter working memory"}
	}
	convInfo := Info{
		Bounds:        info.Bounds,
		Channels:      info.Channels,
		ColorSpace:    info.ColorSpace,
		ColorHint:     info.ColorHint,
		ColorEncoding: cloneColorEncoding(info.ColorEncoding),
		Alpha:         info.Alpha,
		Premultiplied: info.Premultiplied,
		Linear:        info.Linear,
		Components:    cloneComponentInfo(info.Components),
	}
	if len(info.ICCProfile) != 0 {
		if managed {
			convInfo.ICCProfile = cloneBytes(info.ICCProfile)
		} else if transform == nil {
			convInfo.ICCProfile = []byte{0}
		}
	}
	var mapping jxrDisplayMapping
	if transform == nil {
		var err error
		if managed {
			mapping, err = jxrManagedRasterDisplay(ctx, convInfo, limits.MaxMemoryBytes-scratch, peakNits, intent)
		} else {
			mapping, err = jxrBasicRasterDisplay(convInfo, peakNits)
		}
		if err != nil {
			return nil, err
		}
	}
	return &ColorConverter{info: convInfo, mapping: mapping, filter: filter, transform: transform}, ctx.Err()
}

// ConvertRowNRGBA 将源图像指定行直接转换为紧凑八位非预乘sRGB RGBA字节扫描行，不分配整幅中间图像
// dst长度须不少于源图像宽度乘以四，包含BT.2446 HDR映射与无色彩变换原生整数快路径
// 入参: ctx Context，nil采用Background, src 源原生图像或单条行带, y 目标行坐标, dst 目标八位RGBA字节切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertRowNRGBA(ctx context.Context, src *Raster, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowNRGBA(ctx, nil, src, nil, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertRowNRGBA", err)
	}
	return nil
}

// ConvertRowNRGBA64 将源图像指定行直接转换为大端十六位非预乘sRGB RGBA字节扫描行，字节序与image.NRGBA64.Pix完全一致
// dst长度须不少于源图像宽度乘以八，不分配整幅中间图像
// 入参: ctx Context，nil采用Background, src 源原生图像或单条行带, y 目标行坐标, dst 目标十六位大端RGBA字节切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertRowNRGBA64(ctx context.Context, src *Raster, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowNRGBA64(ctx, nil, src, nil, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertRowNRGBA64", err)
	}
	return nil
}

// ConvertRowLinear 将源图像指定行直接转换为未裁剪的D65线性scRGB浮点RGBA扫描行
// 每像素按顺序写入非预乘红、绿、蓝与原透明度四个float64值，dst长度须不少于源图像宽度乘以四
// 入参: ctx Context，nil采用Background, src 源原生图像或单条行带, y 目标行坐标, dst 目标线性scRGB浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertRowLinear(ctx context.Context, src *Raster, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowLinear(ctx, nil, src, nil, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertRowLinear", err)
	}
	return nil
}

// ConvertRowXYZ 将源图像指定行直接转换为D50 PCS XYZ与原透明度四个float64值，供外部CMM直连避免sRGB裁切
// 当源为ICC时省略末端XYZ到sRGB矩阵直接返回D50 PCS XYZ；当源为CICP或默认sRGB时经Bradford D65到D50适配输出D50 XYZ
// 入参: ctx Context，nil采用Background, src 源原生图像或单条行带, y 目标行坐标, dst 目标D50 XYZ与Alpha浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertRowXYZ(ctx context.Context, src *Raster, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowXYZ(ctx, nil, src, nil, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertRowXYZ", err)
	}
	return nil
}

// ConvertStripRowNRGBA 结合相邻上下行带将当前行带局部行号y转换为八位非预乘sRGB RGBA扫描行
// 在ResampleBilinear下跨带边界读取相邻带色度样本以彻底消除YCC420/422的跨带接缝，prev或next为nil表示图像顶底边界
// 仅适用于自顶向下的全宽水平相邻行带；遇到转置竖直列带时返回ErrInvalid，全部八种方向应使用ConvertStripRegionNRGBA
// 入参: ctx Context，nil采用Background, prev 上方相邻行带（可nil）, curr 当前行带, next 下方相邻行带（可nil）, y 当前带行坐标, dst 目标八位RGBA字节切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRowNRGBA(ctx context.Context, prev, curr, next *Raster, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowNRGBA(ctx, prev, curr, next, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRowNRGBA", err)
	}
	return nil
}

// ConvertStripRowNRGBA64 结合相邻上下行带将当前行带局部行号y转换为大端十六位非预乘sRGB RGBA扫描行
// 在ResampleBilinear下跨带边界读取相邻带色度样本以彻底消除YCC420/422的跨带接缝，prev或next为nil表示图像顶底边界
// 仅适用于自顶向下的全宽水平相邻行带；遇到转置竖直列带时返回ErrInvalid，全部八种方向应使用ConvertStripRegionNRGBA64
// 入参: ctx Context，nil采用Background, prev 上方相邻行带（可nil）, curr 当前行带, next 下方相邻行带（可nil）, y 当前带行坐标, dst 目标十六位大端RGBA字节切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRowNRGBA64(ctx context.Context, prev, curr, next *Raster, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowNRGBA64(ctx, prev, curr, next, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRowNRGBA64", err)
	}
	return nil
}

// ConvertStripRowLinear 结合相邻上下行带将当前行带局部行号y转换为未裁剪的D65线性scRGB浮点RGBA扫描行
// 在ResampleBilinear下跨带边界读取相邻带色度样本以消除YCC420/422的跨带接缝，prev或next为nil表示图像顶底边界
// 仅适用于自顶向下的全宽水平相邻行带；遇到转置竖直列带时返回ErrInvalid，全部八种方向应使用ConvertStripRegionLinear
// 入参: ctx Context，nil采用Background, prev 上方相邻行带（可nil）, curr 当前行带, next 下方相邻行带（可nil）, y 当前带行坐标, dst 目标线性scRGB浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRowLinear(ctx context.Context, prev, curr, next *Raster, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowLinear(ctx, prev, curr, next, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRowLinear", err)
	}
	return nil
}

// ConvertStripRowXYZ 结合相邻上下行带将当前行带局部行号y转换为D50 PCS XYZ与原透明度四个float64值
// 在ResampleBilinear下跨带边界读取相邻带色度样本以消除YCC420/422的跨带接缝，prev或next为nil表示图像顶底边界
// 仅适用于自顶向下的全宽水平相邻行带；遇到转置竖直列带时返回ErrInvalid，全部八种方向应使用ConvertStripRegionXYZ
// 入参: ctx Context，nil采用Background, prev 上方相邻行带（可nil）, curr 当前行带, next 下方相邻行带（可nil）, y 当前带行坐标, dst 目标D50 XYZ与Alpha浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRowXYZ(ctx context.Context, prev, curr, next *Raster, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := c.convertStripRowXYZ(ctx, prev, curr, next, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRowXYZ", err)
	}
	return nil
}

// ConvertStripRegionNRGBA 按显示区域几何自动识别正序水平行带、倒序水平行带与转置竖直列带的上下或左右邻域，将当前带局部行y转换为八位非预乘sRGB RGBA扫描行
// 支持DecodeStrips输出的全部八种方向条带无缝色度重采样，零值StripRegion表示图像边界
// 入参: ctx Context，nil采用Background, prev 回调序前一邻带（可零值）, curr 当前条带, next 回调序后一邻带（可零值）, y 当前带局部行坐标, dst 目标八位RGBA字节切片
// 返回: error 参数、区域几何、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRegionNRGBA(ctx context.Context, prev, curr, next StripRegion, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	neighbors, err := c.resolveStripRegionNeighbors(prev, curr, next)
	if err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionNRGBA", err)
	}
	if err := c.convertStripNeighborsNRGBA(ctx, neighbors, curr.Raster, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionNRGBA", err)
	}
	return nil
}

// ConvertStripRegionNRGBA64 按显示区域几何自动识别正序水平行带、倒序水平行带与转置竖直列带的上下或左右邻域，将当前带局部行y转换为大端十六位非预乘sRGB RGBA扫描行
// 支持DecodeStrips输出的全部八种方向条带无缝色度重采样，零值StripRegion表示图像边界
// 入参: ctx Context，nil采用Background, prev 回调序前一邻带（可零值）, curr 当前条带, next 回调序后一邻带（可零值）, y 当前带局部行坐标, dst 目标十六位大端RGBA字节切片
// 返回: error 参数、区域几何、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRegionNRGBA64(ctx context.Context, prev, curr, next StripRegion, y int, dst []byte) error {
	if ctx == nil {
		ctx = context.Background()
	}
	neighbors, err := c.resolveStripRegionNeighbors(prev, curr, next)
	if err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionNRGBA64", err)
	}
	if err := c.convertStripNeighborsNRGBA64(ctx, neighbors, curr.Raster, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionNRGBA64", err)
	}
	return nil
}

// ConvertStripRegionLinear 按显示区域几何自动识别正序水平行带、倒序水平行带与转置竖直列带的上下或左右邻域，将当前带局部行y转换为未裁剪D65线性scRGB浮点扫描行
// 支持DecodeStrips输出的全部八种方向条带无缝色度重采样，零值StripRegion表示图像边界
// 入参: ctx Context，nil采用Background, prev 回调序前一邻带（可零值）, curr 当前条带, next 回调序后一邻带（可零值）, y 当前带局部行坐标, dst 目标线性scRGB浮点切片
// 返回: error 参数、区域几何、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRegionLinear(ctx context.Context, prev, curr, next StripRegion, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	neighbors, err := c.resolveStripRegionNeighbors(prev, curr, next)
	if err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionLinear", err)
	}
	if err := c.convertStripNeighborsLinear(ctx, neighbors, curr.Raster, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionLinear", err)
	}
	return nil
}

// ConvertStripRegionXYZ 按显示区域几何自动识别正序水平行带、倒序水平行带与转置竖直列带的上下或左右邻域，将当前带局部行y转换为D50 PCS XYZ与原透明度扫描行
// 支持DecodeStrips输出的全部八种方向条带无缝色度重采样，零值StripRegion表示图像边界
// 入参: ctx Context，nil采用Background, prev 回调序前一邻带（可零值）, curr 当前条带, next 回调序后一邻带（可零值）, y 当前带局部行坐标, dst 目标D50 XYZ与Alpha浮点切片
// 返回: error 参数、区域几何、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) ConvertStripRegionXYZ(ctx context.Context, prev, curr, next StripRegion, y int, dst []float64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	neighbors, err := c.resolveStripRegionNeighbors(prev, curr, next)
	if err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionXYZ", err)
	}
	if err := c.convertStripNeighborsXYZ(ctx, neighbors, curr.Raster, y, dst); err != nil {
		return wrapJXRError("ColorConverter.ConvertStripRegionXYZ", err)
	}
	return nil
}

// checkSource 校验源图像或单条行带与转换器元数据在颜色空间、通道及分量编码上严格一致
// 入参: src 待转换图像或行带
// 返回: error 不兼容或无效参数错误
func (c *ColorConverter) checkSource(src *Raster) error {
	if c == nil {
		return fmt.Errorf("%w: nil color converter", ErrInvalid)
	}
	if src == nil || src.Bounds().Empty() {
		return fmt.Errorf("%w: color converter raster", ErrInvalid)
	}
	info := src.info
	if info.ColorSpace != c.info.ColorSpace || info.Channels != c.info.Channels || info.Alpha != c.info.Alpha || info.Premultiplied != c.info.Premultiplied || info.Linear != c.info.Linear || len(info.Components) != len(c.info.Components) {
		return fmt.Errorf("%w: color converter source mismatch", ErrInvalid)
	}
	for index, comp := range info.Components {
		want := c.info.Components[index]
		if !jxrSameSampleEncoding(comp, want) || comp.XStep != want.XStep || comp.YStep != want.YStep {
			return fmt.Errorf("%w: color converter component mismatch", ErrInvalid)
		}
	}
	return nil
}

// checkStripNeighbors 校验当前行带与可选相邻上下行带的兼容性、分量行宽一致、YStep对齐及一致色度相位
// 入参: prev 上方行带, curr 当前行带, next 下方行带
// 返回: error 不兼容或无效参数错误
func (c *ColorConverter) checkStripNeighbors(prev, curr, next *Raster) error {
	if err := c.checkSource(curr); err != nil {
		return err
	}
	width, height := curr.Bounds().Dx(), curr.Bounds().Dy()
	hasNeighbor := prev != nil || next != nil
	if hasNeighbor && !c.info.Bounds.Empty() && width != c.info.Bounds.Dx() {
		return fmt.Errorf("%w: color converter horizontal strip width mismatch", ErrInvalid)
	}
	for _, comp := range curr.info.Components {
		if comp.Bounds.Dx() <= 0 || comp.Bounds.Dy() <= 0 {
			return fmt.Errorf("%w: color converter strip component bounds", ErrInvalid)
		}
		if hasNeighbor && (height%int(comp.YStep) != 0 || comp.Bounds.Dy() != height/int(comp.YStep)) {
			return fmt.Errorf("%w: color converter strip vertical step alignment", ErrInvalid)
		}
	}
	for _, neighbor := range [2]*Raster{prev, next} {
		if neighbor == nil {
			continue
		}
		if err := c.checkSource(neighbor); err != nil {
			return err
		}
		if neighbor.Bounds().Dx() != width {
			return fmt.Errorf("%w: color converter strip width mismatch", ErrInvalid)
		}
		nHeight := neighbor.Bounds().Dy()
		for index, comp := range neighbor.info.Components {
			currComp := curr.info.Components[index]
			if !jxrSharedChromaPhase(comp, currComp) {
				return fmt.Errorf("%w: color converter strip chroma phase mismatch", ErrInvalid)
			}
			if comp.Bounds.Dx() != currComp.Bounds.Dx() || comp.Bounds.Dy() <= 0 {
				return fmt.Errorf("%w: color converter strip component width mismatch", ErrInvalid)
			}
			if nHeight%int(comp.YStep) != 0 || comp.Bounds.Dy() != nHeight/int(comp.YStep) {
				return fmt.Errorf("%w: color converter strip neighbor vertical step alignment", ErrInvalid)
			}
		}
	}
	return nil
}

// checkStripRegionItem 校验单个条带区域与原生样本尺寸、边界及分量采样步长对齐一致
// 入参: item 待校验条带区域
// 返回: error 条带区域、样本或步长不合法错误
func (c *ColorConverter) checkStripRegionItem(item StripRegion) error {
	if err := c.checkSource(item.Raster); err != nil {
		return err
	}
	bounds := item.Raster.Bounds()
	if item.Region.Empty() || item.Region.Min.X < 0 || item.Region.Min.Y < 0 || item.Region.Dx() != bounds.Dx() || item.Region.Dy() != bounds.Dy() {
		return fmt.Errorf("%w: color converter strip region mismatch", ErrInvalid)
	}
	if !c.info.Bounds.Empty() && !item.Region.In(c.info.Bounds) {
		return fmt.Errorf("%w: color converter strip region out of bounds", ErrInvalid)
	}
	for _, comp := range item.Raster.info.Components {
		sx, sy := int(comp.XStep), int(comp.YStep)
		if item.Region.Min.X%sx != 0 || item.Region.Min.Y%sy != 0 || item.Region.Dx()%sx != 0 || item.Region.Dy()%sy != 0 || comp.Bounds.Dx() != item.Region.Dx()/sx || comp.Bounds.Dy() != item.Region.Dy()/sy {
			return fmt.Errorf("%w: color converter strip region step alignment", ErrInvalid)
		}
	}
	return nil
}

// resolveStripRegionNeighbors 根据显示区域精确贴边关系将prev与next解析为上下或左右邻带，自动适配正倒序水平带与四类竖直列带
// 入参: prev 前一邻带, curr 当前条带, next 后一邻带
// 返回: jxrStripNeighbors 已校验的上下或左右邻带, error 几何错位、重叠或混轴错误
func (c *ColorConverter) resolveStripRegionNeighbors(prev, curr, next StripRegion) (jxrStripNeighbors, error) {
	var neighbors jxrStripNeighbors
	if err := c.checkStripRegionItem(curr); err != nil {
		return neighbors, err
	}
	for _, item := range [2]StripRegion{prev, next} {
		if item.Raster == nil && item.Region.Empty() {
			continue
		}
		if item.Raster == nil || item.Region.Empty() {
			return jxrStripNeighbors{}, fmt.Errorf("%w: color converter incomplete strip neighbor", ErrInvalid)
		}
		if err := c.checkStripRegionItem(item); err != nil {
			return jxrStripNeighbors{}, err
		}
		for index, comp := range item.Raster.info.Components {
			if !jxrSharedChromaPhase(comp, curr.Raster.info.Components[index]) {
				return jxrStripNeighbors{}, fmt.Errorf("%w: color converter strip chroma phase mismatch", ErrInvalid)
			}
		}
		switch {
		case item.Region.Min.X == curr.Region.Min.X && item.Region.Max.X == curr.Region.Max.X && item.Region.Max.Y == curr.Region.Min.Y:
			if neighbors.top != nil {
				return jxrStripNeighbors{}, fmt.Errorf("%w: color converter duplicate top strip", ErrInvalid)
			}
			neighbors.top = item.Raster
		case item.Region.Min.X == curr.Region.Min.X && item.Region.Max.X == curr.Region.Max.X && item.Region.Min.Y == curr.Region.Max.Y:
			if neighbors.bottom != nil {
				return jxrStripNeighbors{}, fmt.Errorf("%w: color converter duplicate bottom strip", ErrInvalid)
			}
			neighbors.bottom = item.Raster
		case item.Region.Min.Y == curr.Region.Min.Y && item.Region.Max.Y == curr.Region.Max.Y && item.Region.Max.X == curr.Region.Min.X:
			if neighbors.left != nil {
				return jxrStripNeighbors{}, fmt.Errorf("%w: color converter duplicate left strip", ErrInvalid)
			}
			neighbors.left = item.Raster
		case item.Region.Min.Y == curr.Region.Min.Y && item.Region.Max.Y == curr.Region.Max.Y && item.Region.Min.X == curr.Region.Max.X:
			if neighbors.right != nil {
				return jxrStripNeighbors{}, fmt.Errorf("%w: color converter duplicate right strip", ErrInvalid)
			}
			neighbors.right = item.Raster
		default:
			return jxrStripNeighbors{}, fmt.Errorf("%w: color converter strip neighbor not adjacent", ErrInvalid)
		}
	}
	if (neighbors.top != nil || neighbors.bottom != nil) && (neighbors.left != nil || neighbors.right != nil) {
		return jxrStripNeighbors{}, fmt.Errorf("%w: color converter mixed strip neighbor axes", ErrInvalid)
	}
	return neighbors, nil
}

// prepareStripRowBounds 校验当前条带行坐标及目标切片长度并返回局部零起点行号与条带尺寸
// 入参: curr 当前条带, y 目标行坐标, dstLen 目标缓冲元素长度, elemPerPixel 每像素目标元素数
// 返回: int 零起点局部行号, int 条带宽度, int 条带高度, error 坐标或目标长度错误
func prepareStripRowBounds(curr *Raster, y, dstLen, elemPerPixel int) (int, int, int, error) {
	bounds := curr.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	row := y - bounds.Min.Y
	if row < 0 || row >= height || int64(dstLen) < int64(width)*int64(elemPerPixel) {
		return 0, 0, 0, fmt.Errorf("%w: color converter row or destination", ErrInvalid)
	}
	return row, width, height, nil
}

// convertStripRowNRGBA 执行八位非预乘sRGB RGBA逐行及上下跨带转换，含原生整数快路径
// 入参: ctx 已规范化Context, prev 上方行带, curr 当前行带, next 下方行带, y 目标行坐标, dst 目标八位RGBA缓冲
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripRowNRGBA(ctx context.Context, prev, curr, next *Raster, y int, dst []byte) error {
	if err := c.checkStripNeighbors(prev, curr, next); err != nil {
		return err
	}
	return c.convertStripNeighborsNRGBA(ctx, jxrStripNeighbors{top: prev, bottom: next}, curr, y, dst)
}

// convertStripNeighborsNRGBA 在已校验邻带几何上执行八位非预乘sRGB RGBA逐行及跨带转换
// 入参: ctx 已规范化Context, neighbors 上下或左右邻带, curr 当前条带, y 目标行坐标, dst 目标八位RGBA缓冲
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripNeighborsNRGBA(ctx context.Context, neighbors jxrStripNeighbors, curr *Raster, y int, dst []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	row, width, height, err := prepareStripRowBounds(curr, y, len(dst), 4)
	if err != nil {
		return err
	}
	if c.filter == ResampleNearest && c.transform == nil && c.mapping.cicp.transfer == 0 && c.mapping.icc.channels == 0 {
		isRGBE := curr.info.ColorSpace == ColorRGB && curr.info.Components[0].Format == SampleRGBE
		if (!c.mapping.linear && !isRGBE && (curr.info.ColorSpace == ColorGray || curr.info.ColorSpace == ColorRGB)) ||
			(c.mapping.linear && isRGBE) {
			if err := curr.readRowNRGBA(y, dst); err == nil {
				return ctx.Err()
			}
		} else if !c.mapping.linear {
			if precision, interleaved, ok := canDisplayJXRDirect(curr, curr.info, c.mapping); ok && precision == 8 {
				displayJXRDirectCMYK8Row8(curr, curr.info.Alpha, interleaved, row, width, dst[:width*4])
				return ctx.Err()
			}
		}
	}
	return c.walkStripRowNeighbors(ctx, neighbors, curr, row, width, height, func(x int, values []float64, alpha float64) error {
		red, green, blue, err := c.displayPixel(ctx, curr.info, values, alpha)
		if err != nil {
			return err
		}
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = displayByte(red), displayByte(green), displayByte(blue), displayByte(alpha)
		return nil
	})
}

// convertStripRowNRGBA64 执行大端十六位非预乘sRGB RGBA逐行及上下跨带转换，含原生整数快路径
// 入参: ctx 已规范化Context, prev 上方行带, curr 当前行带, next 下方行带, y 目标行坐标, dst 目标十六位大端RGBA缓冲
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripRowNRGBA64(ctx context.Context, prev, curr, next *Raster, y int, dst []byte) error {
	if err := c.checkStripNeighbors(prev, curr, next); err != nil {
		return err
	}
	return c.convertStripNeighborsNRGBA64(ctx, jxrStripNeighbors{top: prev, bottom: next}, curr, y, dst)
}

// convertStripNeighborsNRGBA64 在已校验邻带几何上执行大端十六位非预乘sRGB RGBA逐行及跨带转换
// 入参: ctx 已规范化Context, neighbors 上下或左右邻带, curr 当前条带, y 目标行坐标, dst 目标十六位大端RGBA缓冲
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripNeighborsNRGBA64(ctx context.Context, neighbors jxrStripNeighbors, curr *Raster, y int, dst []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	row, width, height, err := prepareStripRowBounds(curr, y, len(dst), 8)
	if err != nil {
		return err
	}
	if c.filter == ResampleNearest && c.transform == nil {
		if precision, interleaved, ok := canDisplayJXRDirect(curr, curr.info, c.mapping); ok {
			displayJXRDirectRow64(curr, curr.info, precision, interleaved, row, width, dst[:width*8])
			return ctx.Err()
		}
	}
	return c.walkStripRowNeighbors(ctx, neighbors, curr, row, width, height, func(x int, values []float64, alpha float64) error {
		red, green, blue, err := c.displayPixel(ctx, curr.info, values, alpha)
		if err != nil {
			return err
		}
		r16, g16, b16, a16 := displayWord(red), displayWord(green), displayWord(blue), displayWord(alpha)
		base := x * 8
		dst[base], dst[base+1], dst[base+2], dst[base+3], dst[base+4], dst[base+5], dst[base+6], dst[base+7] = byte(r16>>8), byte(r16), byte(g16>>8), byte(g16), byte(b16>>8), byte(b16), byte(a16>>8), byte(a16)
		return nil
	})
}

// convertStripRowLinear 执行未裁剪D65线性scRGB RGBA逐行及上下跨带转换
// 入参: ctx 已规范化Context, prev 上方行带, curr 当前行带, next 下方行带, y 目标行坐标, dst 目标线性scRGB浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripRowLinear(ctx context.Context, prev, curr, next *Raster, y int, dst []float64) error {
	if err := c.checkStripNeighbors(prev, curr, next); err != nil {
		return err
	}
	return c.convertStripNeighborsLinear(ctx, jxrStripNeighbors{top: prev, bottom: next}, curr, y, dst)
}

// convertStripNeighborsLinear 在已校验邻带几何上执行未裁剪D65线性scRGB RGBA逐行及跨带转换
// 入参: ctx 已规范化Context, neighbors 上下或左右邻带, curr 当前条带, y 目标行坐标, dst 目标线性scRGB浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripNeighborsLinear(ctx context.Context, neighbors jxrStripNeighbors, curr *Raster, y int, dst []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	row, width, height, err := prepareStripRowBounds(curr, y, len(dst), 4)
	if err != nil {
		return err
	}
	return c.walkStripRowNeighbors(ctx, neighbors, curr, row, width, height, func(x int, values []float64, alpha float64) error {
		red, green, blue, err := c.linearPixel(ctx, curr.info, values, alpha)
		if err != nil {
			return err
		}
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = red, green, blue, alpha
		return nil
	})
}

// convertStripRowXYZ 执行D50 PCS XYZ与原透明度逐行及上下跨带转换
// 入参: ctx 已规范化Context, prev 上方行带, curr 当前行带, next 下方行带, y 目标行坐标, dst 目标D50 XYZ与Alpha浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripRowXYZ(ctx context.Context, prev, curr, next *Raster, y int, dst []float64) error {
	if err := c.checkStripNeighbors(prev, curr, next); err != nil {
		return err
	}
	return c.convertStripNeighborsXYZ(ctx, jxrStripNeighbors{top: prev, bottom: next}, curr, y, dst)
}

// convertStripNeighborsXYZ 在已校验邻带几何上执行D50 PCS XYZ与原透明度逐行及跨带转换
// 入参: ctx 已规范化Context, neighbors 上下或左右邻带, curr 当前条带, y 目标行坐标, dst 目标D50 XYZ与Alpha浮点切片
// 返回: error 参数、坐标、色彩、数值、变换或取消错误
func (c *ColorConverter) convertStripNeighborsXYZ(ctx context.Context, neighbors jxrStripNeighbors, curr *Raster, y int, dst []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	row, width, height, err := prepareStripRowBounds(curr, y, len(dst), 4)
	if err != nil {
		return err
	}
	return c.walkStripRowNeighbors(ctx, neighbors, curr, row, width, height, func(x int, values []float64, alpha float64) error {
		vx, vy, vz, err := c.xyzPixel(ctx, curr.info, values, alpha)
		if err != nil {
			return err
		}
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = vx, vy, vz, alpha
		return nil
	})
}

// walkStripRowNeighbors 在单行内为各分量执行零分配栈上采样并驱动逐像素输出回调，支持跨带边界读取上下或左右邻带色度
// 入参: ctx 已规范化Context, neighbors 上下或左右邻带, curr 当前条带, row 零起点行号, width,height 当前带尺寸, emit 单像素输出回调
// 返回: error 采样、色彩或取消错误
func (c *ColorConverter) walkStripRowNeighbors(ctx context.Context, neighbors jxrStripNeighbors, curr *Raster, row, width, height int, emit func(int, []float64, float64) error) error {
	count := len(curr.info.Components)
	var valueStorage [9]float64
	values := valueStorage[:]
	if count > len(valueStorage) {
		values = make([]float64, count)
	} else {
		values = values[:count]
	}
	horizontal := neighbors.left != nil || neighbors.right != nil
	firstRaster, secondRaster := neighbors.top, neighbors.bottom
	if horizontal {
		firstRaster, secondRaster = neighbors.left, neighbors.right
	}
	hasFirst, hasSecond := firstRaster != nil, secondRaster != nil
	useNearestFast := c.filter == ResampleNearest && !hasFirst && !hasSecond && curr.Bounds().Min == (image.Point{})
	var currStorage, firstStorage, secondStorage [9]Component
	var currComps, firstComps, secondComps []Component
	if !useNearestFast {
		if count > len(currStorage) {
			currComps = make([]Component, count)
			if hasFirst {
				firstComps = make([]Component, count)
			}
			if hasSecond {
				secondComps = make([]Component, count)
			}
		} else {
			currComps = currStorage[:count]
			if hasFirst {
				firstComps = firstStorage[:count]
			}
			if hasSecond {
				secondComps = secondStorage[:count]
			}
		}
		for index := range count {
			var err error
			currComps[index], err = curr.component(index)
			if err != nil {
				return err
			}
			if hasFirst {
				firstComps[index], err = firstRaster.component(index)
				if err != nil {
					return err
				}
			}
			if hasSecond {
				secondComps[index], err = secondRaster.component(index)
				if err != nil {
					return err
				}
			}
		}
	}
	for x := range width {
		if x&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if useNearestFast {
			if err := jxrDisplayNearestPixel(curr, x, row, values); err != nil {
				return err
			}
		} else {
			for index := range count {
				var firstComp, secondComp Component
				if hasFirst {
					firstComp = firstComps[index]
				}
				if hasSecond {
					secondComp = secondComps[index]
				}
				var err error
				values[index], err = displayJXRStripSampleNeighbors(firstComp, currComps[index], secondComp, hasFirst, hasSecond, horizontal, x, row, width, height, c.filter)
				if err != nil {
					return err
				}
			}
		}
		alpha := 1.0
		if curr.info.Alpha {
			alpha = values[curr.info.Channels]
		}
		if err := emit(x, values, alpha); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// displayPixel 计算单个已采样像素的非预乘sRGB显示三通道值
// 入参: ctx 已规范化Context, info 源元数据, values 可复用样本切片, alpha 当前透明度
// 返回: float64 红、绿、蓝sRGB值, error 颜色、变换或取消错误
func (c *ColorConverter) displayPixel(ctx context.Context, info Info, values []float64, alpha float64) (float64, float64, float64, error) {
	if c.transform != nil {
		red, green, blue, err := c.transform(values[:info.Channels:info.Channels], alpha)
		if err != nil {
			return 0, 0, 0, jxrCallbackError{err: err}
		}
		if err := ctx.Err(); err != nil {
			return 0, 0, 0, err
		}
		return red, green, blue, nil
	}
	return c.mapping.color(ctx, info, values, alpha)
}

// linearPixel 计算单个已采样像素的未裁剪非预乘D65线性scRGB三通道值
// 入参: ctx 已规范化Context, info 源元数据, values 可复用样本切片, alpha 当前透明度
// 返回: float64 线性红、绿、蓝scRGB值, error 颜色、变换或取消错误
func (c *ColorConverter) linearPixel(ctx context.Context, info Info, values []float64, alpha float64) (float64, float64, float64, error) {
	if c.transform != nil {
		red, green, blue, err := c.transform(values[:info.Channels:info.Channels], alpha)
		if err != nil {
			return 0, 0, 0, jxrCallbackError{err: err}
		}
		if err := ctx.Err(); err != nil {
			return 0, 0, 0, err
		}
		return jxrCICPInverseTransfer(red, 13, true), jxrCICPInverseTransfer(green, 13, true), jxrCICPInverseTransfer(blue, 13, true), nil
	}
	return c.mapping.linearColor(ctx, info, values, alpha, c.mapping.linear, true)
}

// xyzPixel 计算单个已采样像素的D50 PCS XYZ三刺激值，ICC源省略末端XYZ到sRGB矩阵直出PCS XYZ
// 入参: ctx 已规范化Context, info 源元数据, values 可复用样本切片, alpha 当前透明度
// 返回: float64 D50三刺激值X、Y、Z, error 颜色、变换或取消错误
func (c *ColorConverter) xyzPixel(ctx context.Context, info Info, values []float64, alpha float64) (float64, float64, float64, error) {
	if c.transform == nil && c.mapping.icc.channels != 0 {
		if info.Premultiplied {
			if !(alpha > 0) || !geometryFinite(alpha) {
				return 0, 0, 0, nil
			}
			for index := range info.Channels {
				values[index] /= alpha
			}
		}
		return c.mapping.icc.xyzColor(ctx, values)
	}
	red, green, blue, err := c.linearPixel(ctx, info, values, alpha)
	if err != nil {
		return 0, 0, 0, err
	}
	xyz := jxrApplyColorMatrix(jxrSRGBToXYZD50, [3]float64{red, green, blue})
	return xyz[0], xyz[1], xyz[2], nil
}

// displayJXRStripSampleNeighbors 在当前带及上下或左右邻带内按分量相位采样，双线性模式下无缝跨带读取相邻色度行或列
// 入参: firstComp,currComp,secondComp 三带同一分量视图, hasFirst,hasSecond 是否存在前后邻带, horizontal 是否左右拼接, x,row 零起点显示坐标, width,height 当前带尺寸, filter 采样滤波
// 返回: float64 采样数值, error 样本或非有限插值错误
func displayJXRStripSampleNeighbors(firstComp, currComp, secondComp Component, hasFirst, hasSecond, horizontal bool, x, row, width, height int, filter ResampleFilter) (float64, error) {
	info := currComp.info
	if info.XStep == 1 && info.YStep == 1 && info.Bounds.Dx() == width && info.Bounds.Dy() == height {
		return currComp.sampleAt(info.Bounds.Min.X+x, info.Bounds.Min.Y+row)
	}
	offsetX := jxrEffectiveOffset(info.XOffset, info.XStep, info.XPositionKnown)
	offsetY := jxrEffectiveOffset(info.YOffset, info.YStep, info.YPositionKnown)
	currW, currH := info.Bounds.Dx(), info.Bounds.Dy()
	var x0, x1, y0, y1 int
	var fx, fy float64
	if horizontal {
		minX, maxX := 0, currW-1
		if hasFirst {
			minX = -1
		}
		if hasSecond {
			maxX = currW
		}
		posX := (float64(x) - offsetX) / float64(info.XStep)
		if posX < float64(minX) {
			posX = float64(minX)
		} else if posX > float64(maxX) {
			posX = float64(maxX)
		}
		x0 = int(math.Floor(posX))
		x1 = min(x0+1, maxX)
		fx = posX - float64(x0)
		y0, y1, fy = resampleJXRAxis(row, height, height, currH, offsetY, info.YStep)
	} else {
		x0, x1, fx = resampleJXRAxis(x, width, width, currW, offsetX, info.XStep)
		minY, maxY := 0, currH-1
		if hasFirst {
			minY = -1
		}
		if hasSecond {
			maxY = currH
		}
		posY := (float64(row) - offsetY) / float64(info.YStep)
		if posY < float64(minY) {
			posY = float64(minY)
		} else if posY > float64(maxY) {
			posY = float64(maxY)
		}
		y0 = int(math.Floor(posY))
		y1 = min(y0+1, maxY)
		fy = posY - float64(y0)
	}
	if filter == ResampleNearest {
		if fx >= 0.5 {
			x0 = x1
		}
		if fy >= 0.5 {
			y0 = y1
		}
		return jxrStripComponentSampleNeighbors(firstComp, currComp, secondComp, horizontal, currW, currH, x0, y0)
	}
	if fx == 0 && fy == 0 {
		return jxrStripComponentSampleNeighbors(firstComp, currComp, secondComp, horizontal, currW, currH, x0, y0)
	}
	var values [4]float64
	for index, point := range [4]image.Point{{X: x0, Y: y0}, {X: x1, Y: y0}, {X: x0, Y: y1}, {X: x1, Y: y1}} {
		if index&1 != 0 && fx == 0 || index&2 != 0 && fy == 0 {
			continue
		}
		value, err := jxrStripComponentSampleNeighbors(firstComp, currComp, secondComp, horizontal, currW, currH, point.X, point.Y)
		if err != nil {
			return 0, err
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("%w: nonfinite display interpolation", ErrInvalid)
		}
		values[index] = value
	}
	return (values[0]*(1-fx)+values[1]*fx)*(1-fy) + (values[2]*(1-fx)+values[3]*fx)*fy, nil
}

// jxrStripComponentSampleNeighbors 按跨带局部坐标从当前带或上下/左右邻带边界读取单个分量样本
// 入参: firstComp,currComp,secondComp 三带分量视图, horizontal 是否左右拼接, currW,currH 当前带分量尺寸, cx,cy 分量局部坐标
// 返回: float64 分量数值, error 坐标或视图错误
func jxrStripComponentSampleNeighbors(firstComp, currComp, secondComp Component, horizontal bool, currW, currH, cx, cy int) (float64, error) {
	if horizontal {
		if cx < 0 {
			return firstComp.sampleAt(firstComp.info.Bounds.Max.X-1, firstComp.info.Bounds.Min.Y+cy)
		}
		if cx >= currW {
			return secondComp.sampleAt(secondComp.info.Bounds.Min.X, secondComp.info.Bounds.Min.Y+cy)
		}
		return currComp.sampleAt(currComp.info.Bounds.Min.X+cx, currComp.info.Bounds.Min.Y+cy)
	}
	if cy < 0 {
		return firstComp.sampleAt(firstComp.info.Bounds.Min.X+cx, firstComp.info.Bounds.Max.Y-1)
	}
	if cy >= currH {
		return secondComp.sampleAt(secondComp.info.Bounds.Min.X+cx, secondComp.info.Bounds.Min.Y)
	}
	return currComp.sampleAt(currComp.info.Bounds.Min.X+cx, currComp.info.Bounds.Min.Y+cy)
}

// displayByte 将显示分量转换为八位无符号字节，非数值采用零并在单位区间裁剪
// 入参: value 显示分量
// 返回: byte 八位无符号值
func displayByte(value float64) byte {
	return byte(math.Floor(colorClamp(value)*255 + 0.5))
}

// displayJXRDirectCMYK8Row8 在整数域将单行八位无符号CMYK转换为紧凑八位NRGBA字节，与通用浮点路径逐位一致
// 入参: raster 源图像, alpha 是否含透明度, interleaved 是否交织存储, row 零起点行号, width 行宽, dst 目标字节切片
func displayJXRDirectCMYK8Row8(raster *Raster, alpha, interleaved bool, row, width int, dst []byte) {
	rowBase := row * width
	if interleaved {
		stride := 4
		if alpha {
			stride = 5
		}
		src := raster.native.values[rowBase*stride : (rowBase+width)*stride]
		for x := range width {
			si, di := x*stride, x*4
			c, m, yv, k := uint32(uint8(src[si])), uint32(uint8(src[si+1])), uint32(uint8(src[si+2])), uint32(uint8(src[si+3]))
			w := (255 - k) * 257
			r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
			av := byte(0xff)
			if alpha {
				av = byte(src[si+4])
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(uint16(r)), jxrScale16To8(uint16(g)), jxrScale16To8(uint16(b)), av
		}
		return
	}
	c0, m0, y0, k0 := raster.planes[0].values[rowBase:rowBase+width], raster.planes[1].values[rowBase:rowBase+width], raster.planes[2].values[rowBase:rowBase+width], raster.planes[3].values[rowBase:rowBase+width]
	var a0 []uint16
	if alpha {
		a0 = raster.planes[4].values[rowBase : rowBase+width]
	}
	for x := range width {
		c, m, yv, k := uint32(uint8(c0[x])), uint32(uint8(m0[x])), uint32(uint8(y0[x])), uint32(uint8(k0[x]))
		w := (255 - k) * 257
		r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
		av := byte(0xff)
		if alpha {
			av = byte(a0[x])
		}
		di := x * 4
		dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(uint16(r)), jxrScale16To8(uint16(g)), jxrScale16To8(uint16(b)), av
	}
}

// displayJXRDirectRow64 用纯整数算术直接填充单行大端十六位非预乘RGBA字节缓冲
// 入参: raster 源图像, info 图像元数据, precision 分量位深, interleaved 是否交织存储, row 零起点行号, width 行宽, dst 目标字节切片
func displayJXRDirectRow64(raster *Raster, info Info, precision uint8, interleaved bool, row, width int, dst []byte) {
	rowBase := row * width
	switch info.ColorSpace {
	case ColorGray:
		if !info.Alpha {
			var src []uint16
			if interleaved {
				src = raster.native.values[rowBase : rowBase+width]
			} else {
				src = raster.planes[0].values[rowBase : rowBase+width]
			}
			if precision == 8 {
				for x, v := range src {
					b, di := byte(v), x*8
					dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = b, b, b, b, b, b, 0xff, 0xff
				}
				return
			}
			for x, v := range src {
				hi, lo, di := byte(v>>8), byte(v), x*8
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = hi, lo, hi, lo, hi, lo, 0xff, 0xff
			}
			return
		}
		if interleaved {
			src := raster.native.values[rowBase*2 : (rowBase+width)*2]
			if precision == 8 {
				for x := range width {
					si, di := x*2, x*8
					b, a := byte(src[si]), byte(src[si+1])
					dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = b, b, b, b, b, b, a, a
				}
				return
			}
			for x := range width {
				si, di := x*2, x*8
				v, a := src[si], src[si+1]
				hi, lo, ah, al := byte(v>>8), byte(v), byte(a>>8), byte(a)
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = hi, lo, hi, lo, hi, lo, ah, al
			}
			return
		}
		g0, a0 := raster.planes[0].values[rowBase:rowBase+width], raster.planes[1].values[rowBase:rowBase+width]
		if precision == 8 {
			for x := range width {
				b, a, di := byte(g0[x]), byte(a0[x]), x*8
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = b, b, b, b, b, b, a, a
			}
			return
		}
		for x := range width {
			v, a, di := g0[x], a0[x], x*8
			hi, lo, ah, al := byte(v>>8), byte(v), byte(a>>8), byte(a)
			dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = hi, lo, hi, lo, hi, lo, ah, al
		}
	case ColorRGB:
		if interleaved {
			stride := 3
			if info.Alpha {
				stride = 4
			}
			src := raster.native.values[rowBase*stride : (rowBase+width)*stride]
			if precision == 8 {
				for x := range width {
					si, di := x*stride, x*8
					rv, gv, bv := byte(src[si]), byte(src[si+1]), byte(src[si+2])
					av := byte(0xff)
					if info.Alpha {
						av = byte(src[si+3])
					}
					dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = rv, rv, gv, gv, bv, bv, av, av
				}
				return
			}
			for x := range width {
				si, di := x*stride, x*8
				rv, gv, bv := src[si], src[si+1], src[si+2]
				av := uint16(0xffff)
				if info.Alpha {
					av = src[si+3]
				}
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), byte(av>>8), byte(av)
			}
			return
		}
		r0, g0, b0 := raster.planes[0].values[rowBase:rowBase+width], raster.planes[1].values[rowBase:rowBase+width], raster.planes[2].values[rowBase:rowBase+width]
		var a0 []uint16
		if info.Alpha {
			a0 = raster.planes[3].values[rowBase : rowBase+width]
		}
		if precision == 8 {
			for x := range width {
				rv, gv, bv, di := byte(r0[x]), byte(g0[x]), byte(b0[x]), x*8
				av := byte(0xff)
				if info.Alpha {
					av = byte(a0[x])
				}
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = rv, rv, gv, gv, bv, bv, av, av
			}
			return
		}
		for x := range width {
			rv, gv, bv, di := r0[x], g0[x], b0[x], x*8
			av := uint16(0xffff)
			if info.Alpha {
				av = a0[x]
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), byte(av>>8), byte(av)
		}
	default:
		if interleaved {
			stride := 4
			if info.Alpha {
				stride = 5
			}
			src := raster.native.values[rowBase*stride : (rowBase+width)*stride]
			for x := range width {
				si, di := x*stride, x*8
				c, m, yv, k := uint32(uint8(src[si])), uint32(uint8(src[si+1])), uint32(uint8(src[si+2])), uint32(uint8(src[si+3]))
				w := (255 - k) * 257
				r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
				av := byte(0xff)
				if info.Alpha {
					av = byte(src[si+4])
				}
				dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), av, av
			}
			return
		}
		c0, m0, y0, k0 := raster.planes[0].values[rowBase:rowBase+width], raster.planes[1].values[rowBase:rowBase+width], raster.planes[2].values[rowBase:rowBase+width], raster.planes[3].values[rowBase:rowBase+width]
		var a0 []uint16
		if info.Alpha {
			a0 = raster.planes[4].values[rowBase : rowBase+width]
		}
		for x := range width {
			c, m, yv, k, di := uint32(uint8(c0[x])), uint32(uint8(m0[x])), uint32(uint8(y0[x])), uint32(uint8(k0[x])), x*8
			w := (255 - k) * 257
			r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
			av := byte(0xff)
			if info.Alpha {
				av = byte(a0[x])
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), av, av
		}
	}
}

// colorClamp 将显示或透明度分量安全裁剪到单位闭区间并将NaN归一化为零
// 入参: value 待裁剪数值
// 返回: float64 单位闭区间数值
func colorClamp(value float64) float64 {
	if !(value > 0) {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
