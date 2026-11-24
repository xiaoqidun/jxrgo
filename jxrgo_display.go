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

// DisplayOptions 指定原生分量取样、显式颜色变换及输出限制，零值采用最近邻和内建基本显示策略
// Transform非nil时由调用方负责色彩和预乘；ApplyICC启用内建ICC颜色解释，Transform优先
// Limits计入输出及库内临时存储，不计调用方原图、既往输出和变换函数自行持有或分配的存储
type DisplayOptions struct {
	Filter    ResampleFilter
	Transform ColorTransform
	Limits    Limits
	ApplyICC  bool
}

// Display 按已知分量相位将原生图像绘制为十六位非预乘sRGB图像，不修改原图也不构造整幅重采样中间图
// 色彩按Transform优先、ApplyICC内建ICC或默认CICP与BT.2446方法A策略转换，并发调用须保证原图只读与Transform并发安全
// 入参: ctx Context，nil采用Background, opts 显示策略及限制，nil采用默认值
// 返回: *image.NRGBA64 独立显示图像, error 参数、相位、颜色、数值、资源、变换或取消错误
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
// 返回: *image.NRGBA64 独立显示图像, error 参数、相位、颜色、数值、资源、变换或取消错误
func (r *Raster) display(ctx context.Context, opts *DisplayOptions) (*image.NRGBA64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Bounds().Empty() || r.info.Channels < 1 || r.info.Channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: display raster", ErrInvalid)
	}
	filter, limits := ResampleNearest, DefaultLimits()
	var transform ColorTransform
	if opts != nil {
		filter, limits, transform = opts.Filter, opts.Limits.normalized(), opts.Transform
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: display filter", ErrInvalid)
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
		mapping, err = jxrBasicRasterDisplay(info)
		if err != nil {
			return nil, err
		}
	}
	if err := checkJXRDisplayPositions(info); err != nil {
		return nil, err
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
		mapping, err = jxrManagedRasterDisplay(ctx, info, limits.MaxMemoryBytes-scratch-pixels*8)
		if err != nil {
			return nil, err
		}
	}
	return renderJXRDisplay(ctx, r, info, mapping, filter, transform)
}

// checkJXRDisplayPositions 校验全部原生分量在显示采样前具备已知位置相位
// 入参: info 原生图像信息
// 返回: error 存在未知采样位置时返回不支持错误
func checkJXRDisplayPositions(info Info) error {
	for _, component := range info.Components {
		if !component.XPositionKnown || !component.YPositionKnown {
			return fmt.Errorf("%w: display requires known sample positions", ErrUnsupported)
		}
	}
	return nil
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
		if !info.XPositionKnown || !info.YPositionKnown {
			return fmt.Errorf("%w: display requires known sample positions", ErrUnsupported)
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
	x0, x1, fx := resampleJXRAxis(x, width, width, info.Bounds.Dx(), info.XOffset, info.XStep)
	y0, y1, fy := resampleJXRAxis(y, height, height, info.Bounds.Dy(), info.YOffset, info.YStep)
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
			component.XStep != 1 || component.YStep != 1 || component.XOffset != 0 || component.YOffset != 0 || component.Bounds != bounds {
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
