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
// HDRPeakNits指定BT.2446方法A的HDR标称峰值亮度（零值对PQ默认10000、对HLG默认1000）；Limits计入输出及库内临时存储
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
// opts为nil时默认采用双线性滤波；ApplyCICPMatrix为真时对YCC优先按PTM CICP矩阵与范围解码而不执行逆传递线性化或基色转换
// 忽略内嵌ICC与ColorHintUnspecified，保留负色域与大于一扩展范围供外部色彩管理管线使用
type RGBOptions struct {
	Filter          ResampleFilter
	ApplyCICPMatrix bool
	Unpremultiply   bool
	Limits          Limits
}

// RGBReader 提供面向外部色彩管理管线的按行设备域RGB浮点读取视图
// 每像素按顺序写入红、绿、蓝与透明度四个未裁剪float64值，对YCC完成相位上采样与矩阵解码而不执行逆传递线性化
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
	if peakNits != 0 && (!geometryFinite(peakNits) || peakNits < 100) {
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
	if peakNits != 0 && (!geometryFinite(peakNits) || peakNits < 100) {
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
// dst长度须不少于图像宽度乘以四，并发读取须由调用方保证源Raster保持只读且不共享ICCFloat管线状态
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
	if row < 0 || row >= height || len(dst) < width*4 {
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
		red, green, blue, err := lr.linearPixel(ctx, values, alpha)
		if err != nil {
			return err
		}
		base := x * 4
		dst[base], dst[base+1], dst[base+2], dst[base+3] = red, green, blue, alpha
	}
	return ctx.Err()
}

// linearPixel 将单个已采样原生像素转换为未裁剪的D65线性scRGB三通道
// 入参: ctx 已规范化Context, values 可复用原生样本工作区, alpha 当前像素透明度
// 返回: float64 线性红、绿、蓝scRGB值, error ICC数值或取消错误
func (lr *LinearReader) linearPixel(ctx context.Context, values []float64, alpha float64) (float64, float64, float64, error) {
	info := lr.info
	if lr.mapping.icc.channels != 0 {
		repremultiply := false
		if info.Premultiplied {
			if !(alpha > 0) || !geometryFinite(alpha) {
				return 0, 0, 0, nil
			}
			for index := range info.Channels {
				values[index] /= alpha
			}
			repremultiply = !lr.unpremultiply
		}
		red, green, blue, err := lr.mapping.icc.linearColor(ctx, values)
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
	if lr.mapping.cicp.transfer != 0 {
		red, green, blue := lr.mapping.cicp.linearColor(info, values, alpha, lr.unpremultiply)
		return red, green, blue, nil
	}
	srcLinear := lr.mapping.linear
	if lr.defaultLinear {
		srcLinear = info.Linear
	}
	if srcLinear && (info.ColorSpace == ColorGray || info.ColorSpace == ColorRGB) {
		var red, green, blue float64
		if info.ColorSpace == ColorGray {
			red, green, blue = values[0], values[0], values[0]
		} else {
			red, green, blue = values[0], values[1], values[2]
		}
		if info.Premultiplied && lr.unpremultiply {
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
		repremultiply = !lr.unpremultiply
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
// dst长度须不少于图像宽度乘以四，并发读取须由调用方保证源Raster保持只读
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
	if row < 0 || row >= height || len(dst) < width*4 {
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
		if !(alpha > 0) {
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

// jxrUnpremultiplyDisplay 按显示透明度恢复非预乘设备分量
// 入参: values 原生颜色工作值, alpha 原透明度
func jxrUnpremultiplyDisplay(values []float64, alpha float64) {
	alpha = colorClamp(alpha)
	if math.IsNaN(alpha) || alpha == 0 {
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
	switch info.ColorSpace {
	case ColorGray:
		if precision == 8 {
			return displayJXRDirectGray8(ctx, raster, info.Alpha, interleaved, output)
		}
		return displayJXRDirectGray16(ctx, raster, info.Alpha, interleaved, output)
	case ColorRGB:
		if precision == 8 {
			return displayJXRDirectRGB8(ctx, raster, info.Alpha, interleaved, output)
		}
		return displayJXRDirectRGB16(ctx, raster, info.Alpha, interleaved, output)
	default:
		return displayJXRDirectCMYK8(ctx, raster, info.Alpha, interleaved, output)
	}
}

// displayJXRDirectGray8 将八位无符号灰度及可选透明度直接写入十六位显示缓冲区
// 入参: ctx Context, raster 原生图像, alpha 是否含透明度, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirectGray8(ctx context.Context, raster *Raster, alpha, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	if !alpha {
		var p0 []uint16
		if interleaved {
			p0 = raster.native.values
		} else {
			p0 = raster.planes[0].values
		}
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := p0[rowBase+start : rowBase+end]
				dst := pixRow[start*8 : end*8]
				_ = src[n-1]
				_ = dst[n*8-1]
				for i := range n {
					v := byte(src[i])
					o := i * 8
					dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = v, v, v, v, v, v, 0xff, 0xff
				}
			}
		}
		return ctx.Err()
	}
	if interleaved {
		srcAll := raster.native.values
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := srcAll[(rowBase+start)*2 : (rowBase+end)*2]
				dst := pixRow[start*8 : end*8]
				_ = src[n*2-1]
				_ = dst[n*8-1]
				for i := range n {
					si, di := i*2, i*8
					v, a := byte(src[si]), byte(src[si+1])
					dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = v, v, v, v, v, v, a, a
				}
			}
		}
		return ctx.Err()
	}
	p0, pa := raster.planes[0].values, raster.planes[1].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		rowBase := y * width
		pixRow := output.Pix[y*output.Stride:]
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			s0, sa := p0[rowBase+start:rowBase+end], pa[rowBase+start:rowBase+end]
			dst := pixRow[start*8 : end*8]
			_ = s0[n-1]
			_ = sa[n-1]
			_ = dst[n*8-1]
			for i := range n {
				v, a := byte(s0[i]), byte(sa[i])
				o := i * 8
				dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = v, v, v, v, v, v, a, a
			}
		}
	}
	return ctx.Err()
}

// displayJXRDirectGray16 将十六位无符号灰度及可选透明度直接写入十六位显示缓冲区
// 入参: ctx Context, raster 原生图像, alpha 是否含透明度, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirectGray16(ctx context.Context, raster *Raster, alpha, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	if !alpha {
		var p0 []uint16
		if interleaved {
			p0 = raster.native.values
		} else {
			p0 = raster.planes[0].values
		}
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := p0[rowBase+start : rowBase+end]
				dst := pixRow[start*8 : end*8]
				_ = src[n-1]
				_ = dst[n*8-1]
				for i := range n {
					v := src[i]
					hi, lo := byte(v>>8), byte(v)
					o := i * 8
					dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = hi, lo, hi, lo, hi, lo, 0xff, 0xff
				}
			}
		}
		return ctx.Err()
	}
	if interleaved {
		srcAll := raster.native.values
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := srcAll[(rowBase+start)*2 : (rowBase+end)*2]
				dst := pixRow[start*8 : end*8]
				_ = src[n*2-1]
				_ = dst[n*8-1]
				for i := range n {
					si, di := i*2, i*8
					v, a := src[si], src[si+1]
					hi, lo, ah, al := byte(v>>8), byte(v), byte(a>>8), byte(a)
					dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = hi, lo, hi, lo, hi, lo, ah, al
				}
			}
		}
		return ctx.Err()
	}
	p0, pa := raster.planes[0].values, raster.planes[1].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		rowBase := y * width
		pixRow := output.Pix[y*output.Stride:]
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			s0, sa := p0[rowBase+start:rowBase+end], pa[rowBase+start:rowBase+end]
			dst := pixRow[start*8 : end*8]
			_ = s0[n-1]
			_ = sa[n-1]
			_ = dst[n*8-1]
			for i := range n {
				v, a := s0[i], sa[i]
				hi, lo, ah, al := byte(v>>8), byte(v), byte(a>>8), byte(a)
				o := i * 8
				dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = hi, lo, hi, lo, hi, lo, ah, al
			}
		}
	}
	return ctx.Err()
}

// displayJXRDirectRGB8 将八位无符号RGB及可选透明度直接写入十六位显示缓冲区
// 入参: ctx Context, raster 原生图像, alpha 是否含透明度, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirectRGB8(ctx context.Context, raster *Raster, alpha, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	if interleaved {
		srcAll := raster.native.values
		if !alpha {
			for y := range height {
				if err := ctx.Err(); err != nil {
					return err
				}
				rowBase := y * width
				pixRow := output.Pix[y*output.Stride:]
				for start := 0; start < width; start += 1024 {
					if err := ctx.Err(); err != nil {
						return err
					}
					end := min(start+1024, width)
					n := end - start
					src := srcAll[(rowBase+start)*3 : (rowBase+end)*3]
					dst := pixRow[start*8 : end*8]
					_ = src[n*3-1]
					_ = dst[n*8-1]
					for i := range n {
						si, di := i*3, i*8
						rv, gv, bv := byte(src[si]), byte(src[si+1]), byte(src[si+2])
						dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = rv, rv, gv, gv, bv, bv, 0xff, 0xff
					}
				}
			}
			return ctx.Err()
		}
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := srcAll[(rowBase+start)*4 : (rowBase+end)*4]
				dst := pixRow[start*8 : end*8]
				_ = src[n*4-1]
				_ = dst[n*8-1]
				for i := range n {
					si, di := i*4, i*8
					rv, gv, bv, av := byte(src[si]), byte(src[si+1]), byte(src[si+2]), byte(src[si+3])
					dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = rv, rv, gv, gv, bv, bv, av, av
				}
			}
		}
		return ctx.Err()
	}
	r0, g0, b0 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values
	if !alpha {
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				rs, gs, bs := r0[rowBase+start:rowBase+end], g0[rowBase+start:rowBase+end], b0[rowBase+start:rowBase+end]
				dst := pixRow[start*8 : end*8]
				_ = rs[n-1]
				_ = gs[n-1]
				_ = bs[n-1]
				_ = dst[n*8-1]
				for i := range n {
					rv, gv, bv := byte(rs[i]), byte(gs[i]), byte(bs[i])
					o := i * 8
					dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = rv, rv, gv, gv, bv, bv, 0xff, 0xff
				}
			}
		}
		return ctx.Err()
	}
	a0 := raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		rowBase := y * width
		pixRow := output.Pix[y*output.Stride:]
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			rs, gs, bs, as := r0[rowBase+start:rowBase+end], g0[rowBase+start:rowBase+end], b0[rowBase+start:rowBase+end], a0[rowBase+start:rowBase+end]
			dst := pixRow[start*8 : end*8]
			_ = rs[n-1]
			_ = gs[n-1]
			_ = bs[n-1]
			_ = as[n-1]
			_ = dst[n*8-1]
			for i := range n {
				rv, gv, bv, av := byte(rs[i]), byte(gs[i]), byte(bs[i]), byte(as[i])
				o := i * 8
				dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = rv, rv, gv, gv, bv, bv, av, av
			}
		}
	}
	return ctx.Err()
}

// displayJXRDirectRGB16 将十六位无符号RGB及可选透明度直接写入十六位显示缓冲区
// 入参: ctx Context, raster 原生图像, alpha 是否含透明度, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirectRGB16(ctx context.Context, raster *Raster, alpha, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	if interleaved {
		srcAll := raster.native.values
		if !alpha {
			for y := range height {
				if err := ctx.Err(); err != nil {
					return err
				}
				rowBase := y * width
				pixRow := output.Pix[y*output.Stride:]
				for start := 0; start < width; start += 1024 {
					if err := ctx.Err(); err != nil {
						return err
					}
					end := min(start+1024, width)
					n := end - start
					src := srcAll[(rowBase+start)*3 : (rowBase+end)*3]
					dst := pixRow[start*8 : end*8]
					_ = src[n*3-1]
					_ = dst[n*8-1]
					for i := range n {
						si, di := i*3, i*8
						rv, gv, bv := src[si], src[si+1], src[si+2]
						dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), 0xff, 0xff
					}
				}
			}
			return ctx.Err()
		}
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := srcAll[(rowBase+start)*4 : (rowBase+end)*4]
				dst := pixRow[start*8 : end*8]
				_ = src[n*4-1]
				_ = dst[n*8-1]
				for i := range n {
					si, di := i*4, i*8
					rv, gv, bv, av := src[si], src[si+1], src[si+2], src[si+3]
					dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), byte(av>>8), byte(av)
				}
			}
		}
		return ctx.Err()
	}
	r0, g0, b0 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values
	if !alpha {
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				rs, gs, bs := r0[rowBase+start:rowBase+end], g0[rowBase+start:rowBase+end], b0[rowBase+start:rowBase+end]
				dst := pixRow[start*8 : end*8]
				_ = rs[n-1]
				_ = gs[n-1]
				_ = bs[n-1]
				_ = dst[n*8-1]
				for i := range n {
					rv, gv, bv := rs[i], gs[i], bs[i]
					o := i * 8
					dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), 0xff, 0xff
				}
			}
		}
		return ctx.Err()
	}
	a0 := raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		rowBase := y * width
		pixRow := output.Pix[y*output.Stride:]
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			rs, gs, bs, as := r0[rowBase+start:rowBase+end], g0[rowBase+start:rowBase+end], b0[rowBase+start:rowBase+end], a0[rowBase+start:rowBase+end]
			dst := pixRow[start*8 : end*8]
			_ = rs[n-1]
			_ = gs[n-1]
			_ = bs[n-1]
			_ = as[n-1]
			_ = dst[n*8-1]
			for i := range n {
				rv, gv, bv, av := rs[i], gs[i], bs[i], as[i]
				o := i * 8
				dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = byte(rv>>8), byte(rv), byte(gv>>8), byte(gv), byte(bv>>8), byte(bv), byte(av>>8), byte(av)
			}
		}
	}
	return ctx.Err()
}

// displayJXRDirectCMYK8 按与color.CMYK.RGBA完全等价的整数公式将八位CMYK直接写入十六位显示缓冲区
// 入参: ctx Context, raster 原生图像, alpha 是否含透明度, interleaved 是否交织存储, output 目标标准图像
// 返回: error 取消错误
func displayJXRDirectCMYK8(ctx context.Context, raster *Raster, alpha, interleaved bool, output *image.NRGBA64) error {
	width, height := output.Rect.Dx(), output.Rect.Dy()
	if interleaved {
		srcAll := raster.native.values
		if !alpha {
			for y := range height {
				if err := ctx.Err(); err != nil {
					return err
				}
				rowBase := y * width
				pixRow := output.Pix[y*output.Stride:]
				for start := 0; start < width; start += 1024 {
					if err := ctx.Err(); err != nil {
						return err
					}
					end := min(start+1024, width)
					n := end - start
					src := srcAll[(rowBase+start)*4 : (rowBase+end)*4]
					dst := pixRow[start*8 : end*8]
					_ = src[n*4-1]
					_ = dst[n*8-1]
					for i := range n {
						si, di := i*4, i*8
						c, m, yv, k := uint32(uint8(src[si])), uint32(uint8(src[si+1])), uint32(uint8(src[si+2])), uint32(uint8(src[si+3]))
						w := (255 - k) * 257
						r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
						dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), 0xff, 0xff
					}
				}
			}
			return ctx.Err()
		}
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				src := srcAll[(rowBase+start)*5 : (rowBase+end)*5]
				dst := pixRow[start*8 : end*8]
				_ = src[n*5-1]
				_ = dst[n*8-1]
				for i := range n {
					si, di := i*5, i*8
					c, m, yv, k, av := uint32(uint8(src[si])), uint32(uint8(src[si+1])), uint32(uint8(src[si+2])), uint32(uint8(src[si+3])), byte(src[si+4])
					w := (255 - k) * 257
					r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
					dst[di+0], dst[di+1], dst[di+2], dst[di+3], dst[di+4], dst[di+5], dst[di+6], dst[di+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), av, av
				}
			}
		}
		return ctx.Err()
	}
	c0, m0, y0, k0 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	if !alpha {
		for y := range height {
			if err := ctx.Err(); err != nil {
				return err
			}
			rowBase := y * width
			pixRow := output.Pix[y*output.Stride:]
			for start := 0; start < width; start += 1024 {
				if err := ctx.Err(); err != nil {
					return err
				}
				end := min(start+1024, width)
				n := end - start
				cs, ms, ys, ks := c0[rowBase+start:rowBase+end], m0[rowBase+start:rowBase+end], y0[rowBase+start:rowBase+end], k0[rowBase+start:rowBase+end]
				dst := pixRow[start*8 : end*8]
				_ = cs[n-1]
				_ = ms[n-1]
				_ = ys[n-1]
				_ = ks[n-1]
				_ = dst[n*8-1]
				for i := range n {
					c, m, yv, k := uint32(uint8(cs[i])), uint32(uint8(ms[i])), uint32(uint8(ys[i])), uint32(uint8(ks[i]))
					w := (255 - k) * 257
					r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
					o := i * 8
					dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), 0xff, 0xff
				}
			}
		}
		return ctx.Err()
	}
	a0 := raster.planes[4].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		rowBase := y * width
		pixRow := output.Pix[y*output.Stride:]
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			cs, ms, ys, ks, as := c0[rowBase+start:rowBase+end], m0[rowBase+start:rowBase+end], y0[rowBase+start:rowBase+end], k0[rowBase+start:rowBase+end], a0[rowBase+start:rowBase+end]
			dst := pixRow[start*8 : end*8]
			_ = cs[n-1]
			_ = ms[n-1]
			_ = ys[n-1]
			_ = ks[n-1]
			_ = as[n-1]
			_ = dst[n*8-1]
			for i := range n {
				c, m, yv, k, av := uint32(uint8(cs[i])), uint32(uint8(ms[i])), uint32(uint8(ys[i])), uint32(uint8(ks[i])), byte(as[i])
				w := (255 - k) * 257
				r, g, b := (255-c)*w/255, (255-m)*w/255, (255-yv)*w/255
				o := i * 8
				dst[o+0], dst[o+1], dst[o+2], dst[o+3], dst[o+4], dst[o+5], dst[o+6], dst[o+7] = byte(r>>8), byte(r), byte(g>>8), byte(g), byte(b>>8), byte(b), av, av
			}
		}
	}
	return ctx.Err()
}

// colorClamp 将显示或透明度分量裁剪到单位区间
// 入参: value 有限分量
// 返回: float64 裁剪结果
func colorClamp(value float64) float64 {
	return min(1, max(0, value))
}
