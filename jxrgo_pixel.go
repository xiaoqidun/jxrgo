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
	"math"
)

// jxrInt32Bias 保存三十二位有符号整数样本零点偏移量
const jxrInt32Bias = uint64(1) << 31

const (
	// ResampleNearest 表示最近邻几何重采样滤波
	ResampleNearest ResampleFilter = iota
	// ResampleBilinear 表示双线性几何重采样滤波
	ResampleBilinear
)

// jxrStandardPixelFormats 保存按标准GUID末字节索引的像素格式查找表
var jxrStandardPixelFormats = makeJXRPixelFormats()

// ResampleFilter 指定原生分量的显式几何滤波，不改变传递函数、ICC或颜色空间
type ResampleFilter uint8

// ResampleOptions 指定滤波及输出资源限制，零值采用最近邻，不计入调用方保留的原图
// 双线性按原生数值域计算并按目标编码取最近偶数，不是线性光或颜色管理操作
type ResampleOptions struct {
	Filter ResampleFilter
	Limits Limits
}

// jxrIntegerOutputComponents 执行整数样本的颜色转换、偏置、缩放及范围裁剪，输入已完成采样转换
// 入参: h 图像头, p 当前平面, values 逐分量重建样本并原地替换, alpha 是否为透明度平面
// 返回: int 输出分量数, error 颜色、数值能力或数据错误
func jxrIntegerOutputComponents(h *jxrHeader, p *jxrPlane, values []int64, alpha bool) (int, error) {
	if h.depth == 4 || h.depth == 7 || h.color == 8 && !alpha {
		return 0, fmt.Errorf("%w: JPEG XR non-integer output", ErrUnsupported)
	}
	count, err := jxrColorOutputComponents(h, p, values, alpha)
	if err != nil {
		return 0, err
	}
	color := h.color
	if alpha {
		color = 0
	}
	return jxrIntegerScaleComponents(h, p, values, color, count)
}

// jxrOutputCount 校验平面及表183的颜色组合，确定转换后有效分量数
// 入参: h 图像头, p 当前平面, alpha 是否为透明度平面
// 返回: int 输出分量数, error 平面或颜色组合错误
func jxrOutputCount(h *jxrHeader, p *jxrPlane, alpha bool) (int, error) {
	if err := checkJXRPlaneComponents(p); err != nil {
		return 0, err
	}
	color := h.color
	if alpha {
		color = 0
	}
	if !jxrPlaneColorAllowed(color, h.depth, p.color) {
		return 0, fmt.Errorf("%w: JPEG XR output color combination", ErrInvalid)
	}
	switch color {
	case 0:
		return 1, nil
	case 1, 2, 3, 7, 8:
		return 3, nil
	case 4, 5:
		return 4, nil
	case 6:
		return p.channels, nil
	default:
		return 0, fmt.Errorf("%w: JPEG XR output color", ErrUnsupported)
	}
}

// jxrColorOutputComponents 执行T.832表183规定的输出颜色转换，采样转换由调用方先行完成
// 入参: h 图像头, p 当前平面, values 原地转换的分量值, alpha 是否为透明度平面
// 返回: int 输出分量数, error 颜色组合或数值范围错误
func jxrColorOutputComponents(h *jxrHeader, p *jxrPlane, values []int64, alpha bool) (int, error) {
	count, err := jxrOutputCount(h, p, alpha)
	if err != nil {
		return 0, err
	}
	if len(values) < max(p.channels, count) {
		return 0, fmt.Errorf("%w: JPEG XR color output storage", ErrInvalid)
	}
	color := h.color
	if alpha {
		color = 0
	}
	if err := checkJXRWords(values[:p.channels], true); err != nil {
		return 0, err
	}
	check := newJXRArithmetic(true)
	switch color {
	case 0:
		count = 1
	case 1, 2, 3:
		count = 3
		if p.color == 0 && color == 3 {
			values[1], values[2] = 0, 0
		}
	case 4:
		k := check.value(values[3] + (values[0] >> 1))
		m := check.value(k - values[0] - (values[1] >> 1))
		c := check.value(values[1] + m + (values[2] >> 1))
		y := check.value(c - values[2])
		values[0], values[1], values[2], values[3] = c, m, y, k
		count = 4
	case 5:
		values[0], values[1], values[2], values[3] = values[1], values[2], values[3], values[0]
		count = 4
	case 6:
	case 7, 8:
		count = 3
		if p.color == 0 {
			values[1], values[2] = values[0], values[0]
		} else {
			t := check.value(-values[1])
			g := check.value(values[0] - (t >> 1))
			r := check.value(t + g - ((values[2] + 1) >> 1))
			b := check.value(values[2] + r)
			values[0], values[1], values[2] = r, g, b
		}
		if !h.redBlue && (h.depth == 8 || h.depth == 9 || h.depth == 10) {
			values[0], values[2] = values[2], values[0]
		}
	default:
		return 0, fmt.Errorf("%w: JPEG XR output color", ErrUnsupported)
	}
	if err := check.error(); err != nil {
		return 0, err
	}
	return count, nil
}

// jxrIntegerScaleComponents 执行整数分量的偏置、缩放、范围裁剪和二值位模式转换
// T.832表199将BD1BLACK1亮度反转为原生位模式
// 入参: h 图像头, p 当前平面, values 原地替换的分量值, color 输出颜色格式, count 分量数
// 返回: int 输出分量数, error 数值范围或位深错误
func jxrIntegerScaleComponents(h *jxrHeader, p *jxrPlane, values []int64, color uint8, count int) (int, error) {
	if count < 1 || count > rasterMaxComponents || count > len(values) {
		return 0, fmt.Errorf("%w: JPEG XR output component count", ErrInvalid)
	}
	if err := checkJXRWords(values[:count], true); err != nil {
		return 0, err
	}
	bias := int64(0)
	switch h.depth {
	case 0, 15, 3, 6:
	case 1:
		bias = 128
	case 2:
		bias = 32768
	case 8:
		bias = 16
	case 9:
		bias = 512
	case 10:
		bias = 32
	default:
		return 0, fmt.Errorf("%w: JPEG XR integer output depth", ErrUnsupported)
	}
	if h.depth == 2 || h.depth == 3 || h.depth == 6 {
		bias >>= p.shift
	}
	scale, round := uint8(0), int64(0)
	if p.scaled {
		scale, round = 3, 3
		if h.depth == 0 || h.depth == 15 || h.depth == 2 {
			round = 4
		}
	}
	for i := range count {
		componentBias := bias
		if color == 4 {
			componentBias >>= 1
			if i == 3 {
				componentBias = -componentBias
			}
		}
		value := values[i] + (componentBias << scale)
		if err := checkJXRWords([]int64{value}, true); err != nil {
			return 0, err
		}
		componentScale := scale
		if color == 7 && h.depth == 10 && i != 1 {
			componentScale++
		}
		value = (value + round) >> componentScale
		if h.depth == 2 || h.depth == 3 || h.depth == 6 {
			if p.shift >= 32 {
				if value != 0 {
					return 0, fmt.Errorf("%w: JPEG XR signed32 output range", ErrInvalid)
				}
			} else if value < (-1<<31)>>p.shift || value > (1<<31-1)>>p.shift {
				return 0, fmt.Errorf("%w: JPEG XR signed32 output range", ErrInvalid)
			} else {
				value <<= p.shift
			}
			if h.depth == 6 {
				values[i] = value
				continue
			}
		}
		low, high := int64(0), int64(255)
		switch h.depth {
		case 0, 15:
			high = 1
		case 2:
			high = 65535
		case 3:
			low, high = -32768, 32767
		case 8:
			high = 31
		case 9:
			high = 1023
		case 10:
			high = 31
			if i == 1 {
				high = 63
			}
		}
		value = min(high, max(low, value))
		if h.depth == 15 {
			value ^= 1
		}
		values[i] = value
	}
	return count, nil
}

// jxrIntegerInput 将整数位模式转换为去偏置、预缩放的颜色样本，尚不执行颜色变换
// 精确模式拒绝有效位丢弃，显式有损模式按附录D.1算术右移，不改变原始位模式
// BD1BLACK1按T.832表199的逆运算恢复变换域使用的亮度方向
// 入参: h 图像头, p 目标平面参数, bits 原生位模式, values 目标样本缓冲, alpha 是否为透明度平面, exact 是否要求输入精确恢复
// 返回: int 输入颜色分量数, error 格式、范围或精度错误
func jxrIntegerInput(h *jxrHeader, p *jxrPlane, bits []uint32, values []int64, alpha, exact bool) (int, error) {
	color, count, bias, precision, shift, scale, err := jxrIntegerInputConfig(h, p, alpha)
	if err != nil {
		return 0, err
	}
	if len(bits) < count || len(values) < count {
		return 0, fmt.Errorf("%w: JPEG XR integer input component count", ErrInvalid)
	}
	for index := range count {
		value, err := jxrIntegerInputSample(h, color, bias, precision, shift, scale, index, bits[index], exact)
		if err != nil {
			return 0, err
		}
		values[index] = value
	}
	return count, nil
}

// jxrIntegerInputConfig 校验整数输入颜色与位深并返回逐通道共享的偏置、移位及缩放参数
// 入参: h 图像头, p 目标平面参数, alpha 是否为透明度平面
// 返回: uint8 有效颜色模式, int 通道数, int64 偏置值, uint8 位深, uint8 移位位数, uint8 预缩放位移, error 格式或能力错误
func jxrIntegerInputConfig(h *jxrHeader, p *jxrPlane, alpha bool) (uint8, int, int64, uint8, uint8, uint8, error) {
	color := h.color
	if alpha {
		color = 0
	}
	count := p.channels
	switch color {
	case 0:
		count = 1
	case 1, 2, 3, 7:
		count = 3
	case 4, 5:
		count = 4
	case 6:
	default:
		return 0, 0, 0, 0, 0, 0, fmt.Errorf("%w: JPEG XR integer input color", ErrUnsupported)
	}
	if count < 1 || count > rasterMaxComponents {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf("%w: JPEG XR integer input component count", ErrInvalid)
	}
	bias, precision := int64(0), uint8(0)
	switch h.depth {
	case 0, 15:
		precision = 1
	case 1:
		bias, precision = 128, 8
	case 2:
		bias, precision = 32768, 16
	case 3:
		precision = 16
	case 6:
		precision = 32
	case 8:
		bias, precision = 16, 5
	case 9:
		bias, precision = 512, 10
	case 10:
		bias, precision = 32, 5
	default:
		return 0, 0, 0, 0, 0, 0, fmt.Errorf("%w: JPEG XR non-integer input", ErrUnsupported)
	}
	shift := uint8(0)
	if h.depth == 2 || h.depth == 3 || h.depth == 6 {
		shift = p.shift
		bias >>= shift
	}
	scale := uint8(0)
	if p.scaled {
		scale = 3
	}
	return color, count, bias, precision, shift, scale, nil
}

// jxrIntegerInputSample 将单个通道的整数原生位模式转换为去偏置预缩放样本
// 入参: h 图像头, color 有效颜色模式, bias 偏置值, precision 基础位深, shift 移位位数, scale 预缩放位移, index 通道索引, bits 原生位模式, exact 是否精确恢复
// 返回: int64 变换域输入样本, error 位宽、精度或范围错误
func jxrIntegerInputSample(h *jxrHeader, color uint8, bias int64, precision, shift, scale uint8, index int, bits uint32, exact bool) (int64, error) {
	width := precision
	if h.depth == 10 && index == 1 {
		width = 6
	}
	if uint64(bits) >= uint64(1)<<width {
		return 0, fmt.Errorf("%w: JPEG XR integer input bit width", ErrInvalid)
	}
	value := int64(bits)
	if h.depth == 3 {
		value = int64(int16(bits))
	} else if h.depth == 6 {
		value = int64(int32(bits))
	} else if h.depth == 15 {
		value ^= 1
	}
	if exact && (shift >= 32 && value != 0 || shift < 32 && value&((int64(1)<<shift)-1) != 0) {
		return 0, fmt.Errorf("%w: JPEG XR input shift discards sample bits", ErrUnsupported)
	}
	value >>= shift
	if shift >= 32 && value != 0 {
		return 0, fmt.Errorf("%w: JPEG XR shifted input has no signed32 output", ErrUnsupported)
	}
	componentScale := scale
	if color == 7 && h.depth == 10 && index != 1 {
		componentScale++
	}
	componentBias := bias
	if color == 4 {
		componentBias >>= 1
		if index == 3 {
			componentBias = -componentBias
		}
	}
	value = (value << componentScale) - (componentBias << scale)
	if value < -int64(1)<<31 || value > int64(1)<<31-1 {
		return 0, fmt.Errorf("%w: JPEG XR word range", ErrInvalid)
	}
	return value, nil
}

// jxrFloatOutputComponents 将重建样本转换为半精度或单精度浮点位模式，不转换未定义的填充分量
// 入参: h 图像头, p 当前平面, values 逐分量重建及颜色转换工作缓冲, alpha 是否为透明度平面, result 浮点位模式目标
// 返回: int 分量数, error 数值、存储或颜色错误
func jxrFloatOutputComponents(h *jxrHeader, p *jxrPlane, values []int64, alpha bool, result []uint32) (int, error) {
	if h.depth != 4 && h.depth != 7 {
		return 0, fmt.Errorf("%w: JPEG XR floating-point depth", ErrInvalid)
	}
	if !alpha && h.color != 0 && (h.color != 7 || p.color != 3) {
		return 0, fmt.Errorf("%w: JPEG XR floating-point color", ErrInvalid)
	}
	count, err := jxrOutputCount(h, p, alpha)
	if err != nil {
		return 0, err
	}
	if len(result) < count {
		return 0, fmt.Errorf("%w: JPEG XR floating-point output storage", ErrInvalid)
	}
	if _, err := jxrColorOutputComponents(h, p, values, alpha); err != nil {
		return 0, err
	}
	for i := range count {
		value := values[i]
		if p.scaled {
			value = (value + 3) >> 3
		}
		result[i], err = jxrFloatBits(value, h.depth, p.mantissa, p.exponent)
		if err != nil {
			return 0, err
		}
	}
	return count, nil
}

// jxrFloatBits 按T.832表192恢复浮点位模式，保留符号、非规格化数及非有限数编码
// 入参: value 输出缩放后的整数值, depth 浮点位深, mantissa 有效尾数位数, bias 码流指数偏置
// 返回: uint32 浮点位模式, error 参数或数值范围错误
func jxrFloatBits(value int64, depth, mantissa uint8, bias int8) (uint32, error) {
	if err := checkJXRWords([]int64{value}, true); err != nil {
		return 0, err
	}
	sign := uint32(0)
	if value < 0 {
		sign, value = 1, -value
	}
	if depth == 4 {
		return sign<<15 | uint32(min(value, 32767)), nil
	}
	if depth != 7 || mantissa > 23 {
		return 0, fmt.Errorf("%w: JPEG XR floating-point mantissa", ErrUnsupported)
	}
	if err := checkJXRWords([]int64{value}, true); err != nil {
		return 0, err
	}
	unit := int64(1) << mantissa
	exponent, fraction := value>>mantissa, value&(unit-1)|unit
	if exponent == 0 {
		fraction ^= unit
		exponent = 1
	}
	exponent += 127 - int64(bias)
	for fraction < unit && exponent > 1 && fraction > 0 {
		exponent--
		fraction <<= 1
	}
	if fraction < unit {
		exponent = 0
	} else {
		fraction ^= unit
	}
	if exponent < 0 || exponent > 255 {
		return 0, fmt.Errorf("%w: JPEG XR floating-point exponent", ErrInvalid)
	}
	return sign<<31 | uint32(exponent)<<23 | uint32(fraction)<<(23-mantissa), nil
}

// float64HalfBits 直接将双精度值舍入为IEEE半精度，保留符号零及非有限数类别
// 不经过单精度中间值，避免半精度中点附近的二次舍入
// 入参: value 双精度数值
// 返回: uint16 半精度位模式
func float64HalfBits(value float64) uint16 {
	bits := math.Float64bits(value)
	sign := uint16(bits>>48) & 0x8000
	exponent := int(bits>>52&2047) - 1023
	fraction := bits & (1<<52 - 1)
	if exponent == 1024 {
		if fraction == 0 {
			return sign | 0x7c00
		}
		payload := uint16(fraction >> 42)
		if payload == 0 {
			payload = 1
		}
		return sign | 0x7c00 | payload
	}
	if exponent > 15 {
		return sign | 0x7c00
	}
	if exponent < -25 {
		return sign
	}
	shift := uint(42)
	if exponent < -14 {
		shift = uint(28 - exponent)
	}
	significand := fraction | 1<<52
	valueBits := significand >> shift
	remainder := significand & (1<<shift - 1)
	midpoint := uint64(1) << (shift - 1)
	if remainder > midpoint || remainder == midpoint && valueBits&1 != 0 {
		valueBits++
	}
	if exponent >= -14 {
		valueBits += uint64(exponent+14) << 10
	}
	return sign | uint16(valueBits)
}

// jxrFloatInput 按规范性浮点输出映射构造编码样本，不执行颜色变换
// 入参: h 图像头, p 目标平面, bits 原生浮点位模式, values 样本缓冲, alpha 是否为透明度, exact 是否要求位模式精确恢复
// 返回: int 输入分量数, error 格式、精度或数值范围错误
func jxrFloatInput(h *jxrHeader, p *jxrPlane, bits []uint32, values []int64, alpha, exact bool) (int, error) {
	color, count := h.color, 1
	if alpha {
		color = 0
	}
	if color == 7 && p.color == 3 && p.channels == 3 {
		count = 3
	} else if color != 0 || p.color != 0 || p.channels != 1 {
		return 0, fmt.Errorf("%w: JPEG XR floating input color", ErrInvalid)
	}
	if len(bits) < count || len(values) < count {
		return 0, fmt.Errorf("%w: JPEG XR floating input components", ErrInvalid)
	}
	for index := range count {
		value, err := jxrFloatInputBits(bits[index], h.depth, p.mantissa, p.exponent, exact)
		if err != nil {
			return 0, err
		}
		if p.scaled {
			value <<= 3
		}
		if err := checkJXRWords([]int64{value}, true); err != nil {
			return 0, err
		}
		values[index] = value
	}
	return count, nil
}

// jxrFloatInputBits 构造T.832表192的逆映射，允许舍入时选取数值最近的相邻编码
// 等距采用偶数编码；符号零及非有限数的位模式变化不作为精确输入接受
// 入参: bits 浮点位模式, depth 浮点位深, mantissa 尾数长度, bias 指数偏置, exact 是否要求位模式精确恢复
// 返回: int64 编码样本, error 格式、精度或数值范围错误
func jxrFloatInputBits(bits uint32, depth, mantissa uint8, bias int8, exact bool) (int64, error) {
	if depth == 4 {
		if bits > 65535 {
			return 0, fmt.Errorf("%w: JPEG XR half input bit width", ErrInvalid)
		}
		value := int64(bits & 32767)
		if bits&32768 != 0 {
			if value == 0 && exact {
				return 0, fmt.Errorf("%w: JPEG XR negative floating zero", ErrUnsupported)
			}
			value = -value
		}
		return value, nil
	}
	if depth != 7 || mantissa > 23 {
		return 0, fmt.Errorf("%w: JPEG XR floating input parameters", ErrInvalid)
	}
	negative := bits>>31 != 0
	absolute := bits & 0x7fffffff
	exponent := int64(absolute >> 23)
	fraction := int64(absolute&0x7fffff) | 1<<23
	if exponent == 0 {
		fraction ^= 1 << 23
		exponent = 1
	}
	exponent += int64(bias) - 127
	if exponent <= 1 {
		if exponent < 1 {
			fraction >>= uint64(1 - exponent)
		}
		exponent = 1
		if fraction&(1<<23) == 0 {
			exponent = 0
		}
	}
	value := exponent<<mantissa | (fraction&0x7fffff)>>(23-mantissa)
	maximum := min(int64(1<<31-1), (int64(bias)+129)<<mantissa-1)
	if value > maximum {
		return 0, fmt.Errorf("%w: JPEG XR floating input working range", ErrInvalid)
	}
	lower, err := jxrFloatBits(value, 7, mantissa, bias)
	if err != nil {
		return 0, err
	}
	if lower != absolute {
		if exact {
			return 0, fmt.Errorf("%w: JPEG XR floating input precision", ErrUnsupported)
		}
		if absolute>>23 == 255 {
			if absolute&0x7fffff != 0 && lower&0x7fffff == 0 && value < maximum {
				value++
				lower, err = jxrFloatBits(value, 7, mantissa, bias)
			}
			if err != nil || lower>>23 != 255 || (lower&0x7fffff == 0) != (absolute&0x7fffff == 0) {
				return 0, fmt.Errorf("%w: JPEG XR non-finite input representation", ErrUnsupported)
			}
		} else if value < maximum {
			upper, err := jxrFloatBits(value+1, 7, mantissa, bias)
			if err != nil {
				return 0, err
			}
			input := float64(math.Float32frombits(absolute))
			lowDistance := input - float64(math.Float32frombits(lower))
			highDistance := float64(math.Float32frombits(upper)) - input
			if lowDistance > highDistance || lowDistance == highDistance && value&1 != 0 {
				value++
			}
		}
	}
	if negative {
		if value == 0 && exact {
			return 0, fmt.Errorf("%w: JPEG XR negative floating zero", ErrUnsupported)
		}
		value = -value
	}
	return value, nil
}

// jxrRGBEOutputComponents 按T.832表193转换共享指数的RGBE分量并裁剪到字节范围
// 入参: h 图像头, p 主平面, values 逐分量重建样本
// 返回: [4]uint8 RGBE值, error 颜色或位深错误
func jxrRGBEOutputComponents(h *jxrHeader, p *jxrPlane, values []int64) ([4]uint8, error) {
	var result [4]uint8
	if h.color != 8 || h.depth != 1 || p.color != 3 {
		return result, fmt.Errorf("%w: JPEG XR RGBE format", ErrInvalid)
	}
	if _, err := jxrColorOutputComponents(h, p, values, false); err != nil {
		return result, err
	}
	return jxrRGBEPackComponents(p, values), nil
}

// jxrRGBEPackComponents 按T.832表193缩放RGB整数样本并构造共享指数，不重复执行颜色变换
// 入参: p 平面缩放参数, values 已通过数值校验的RGB样本
// 返回: [4]uint8 RGBE值
func jxrRGBEPackComponents(p *jxrPlane, values []int64) [4]uint8 {
	var result [4]uint8
	var mantissa, exponent [3]int64
	shared := int64(0)
	for i := range 3 {
		value := values[i]
		if p.scaled {
			value = (value + 3) >> 3
		}
		if value > 0 {
			mantissa[i], exponent[i] = value, 1
			if value>>7 > 1 {
				mantissa[i], exponent[i] = value&127+128, value>>7
			}
		}
		shared = max(shared, exponent[i])
	}
	for i := range 3 {
		if shared > exponent[i] {
			mantissa[i] = (2*mantissa[i] + 1) >> uint64(shared-exponent[i]+1)
		}
		result[i] = uint8(min(255, max(0, mantissa[i])))
	}
	result[3] = uint8(min(255, shared))
	return result
}

// jxrRGBEInput 将共享指数位模式转换为编码侧RGB样本，精确模式拒绝无法原样恢复的表示
// 入参: h 图像头, p 目标平面, bits 每个分量的低八位尾数与高八位共享指数, values 样本缓冲, exact 是否要求位模式精确恢复
// 返回: int 输入分量数, error 格式、共享指数或精度错误
func jxrRGBEInput(h *jxrHeader, p *jxrPlane, bits []uint32, values []int64, exact bool) (int, error) {
	if h.color != 8 || h.depth != 1 || p.color != 3 || p.channels != 3 || len(bits) < 3 || len(values) < 3 {
		return 0, fmt.Errorf("%w: JPEG XR RGBE input format", ErrInvalid)
	}
	exponent := bits[0] >> 8
	for index := range 3 {
		if bits[index] > 65535 || bits[index]>>8 != exponent {
			return 0, fmt.Errorf("%w: JPEG XR RGBE input shared exponent", ErrInvalid)
		}
		mantissa, e := int64(bits[index]&255), int64(exponent)
		value := int64(0)
		if mantissa != 0 && e != 0 {
			e--
			appendBit := int64(1)
			for mantissa < 128 && e > 0 {
				mantissa = mantissa*2 + appendBit
				appendBit = 0
				e--
			}
			value = mantissa
			if e != 0 {
				value = mantissa&127 + (e+1)<<7
			}
		}
		if p.scaled {
			value <<= 3
		}
		values[index] = value
	}
	if exact {
		var converted [16]int64
		copy(converted[:], values[:3])
		output := jxrRGBEPackComponents(p, converted[:])
		for index := range 3 {
			if uint32(output[index])|uint32(output[3])<<8 != bits[index] {
				return 0, fmt.Errorf("%w: JPEG XR RGBE input representation", ErrUnsupported)
			}
		}
	}
	return 3, nil
}

// jxrRasterBitsComponents 按标准输出原生位模式，不先归一化或转换为显示精度
// 入参: h 图像头, p 当前平面, number 数值格式, values 重建及颜色转换工作缓冲, alpha 是否为透明度平面, result 原生位模式目标
// 返回: int 分量数, error 数值、存储或格式错误
func jxrRasterBitsComponents(h *jxrHeader, p *jxrPlane, number uint8, values []int64, alpha bool, result []uint32) (int, error) {
	if number == jxrUnsigned && (h.depth == 1 || h.depth == 2 && p.shift == 0) {
		if count, ok := jxrFastUnsignedBitsComponents(h, p, values, alpha, result); ok {
			return count, nil
		}
	}
	count, err := jxrOutputCount(h, p, alpha)
	if err != nil {
		return 0, err
	}
	if len(result) < count || len(values) < max(count, p.channels) {
		return 0, fmt.Errorf("%w: JPEG XR native output storage", ErrInvalid)
	}
	if number == jxrHalf || number == jxrFloat {
		return jxrFloatOutputComponents(h, p, values, alpha, result)
	}
	if number == jxrRGBE && !alpha {
		bits, err := jxrRGBEOutputComponents(h, p, values)
		if err != nil {
			return 0, err
		}
		for index := range 3 {
			result[index] = uint32(bits[index]) | uint32(bits[3])<<8
		}
		return 3, nil
	}
	count, err = jxrIntegerOutputComponents(h, p, values, alpha)
	if err != nil {
		return 0, err
	}
	for index := range count {
		result[index] = uint32(values[index])
		if number != jxrFixed32 {
			result[index] &= 65535
		}
	}
	return count, nil
}

// jxrFastUnsignedBitsComponents 在无移位八位或十六位无符号整数路径直接完成颜色转换、范围检查与裁剪
// 入参: h 图像头, p 当前平面, values 重建样本, alpha 是否为透明度平面, result 原生位模式目标
// 返回: int 分量数, bool 是否已完成快路径输出
func jxrFastUnsignedBitsComponents(h *jxrHeader, p *jxrPlane, values []int64, alpha bool, result []uint32) (int, bool) {
	color := h.color
	if alpha {
		color = 0
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
	switch color {
	case 0:
		if p.color != 0 || p.channels != 1 || len(values) < 1 || len(result) < 1 {
			return 0, false
		}
		v0 := values[0]
		y0 := v0 + off
		if ((uint64(v0)+jxrInt32Bias)|(uint64(y0)+jxrInt32Bias))>>32 != 0 {
			return 0, false
		}
		result[0] = uint32(min(high, max(0, (y0+round)>>scale)))
		return 1, true
	case 7:
		if len(values) < 3 || len(result) < 3 {
			return 0, false
		}
		if p.color == 0 {
			if p.channels != 1 {
				return 0, false
			}
			v0 := values[0]
			r0 := v0 + off
			if ((uint64(v0)+jxrInt32Bias)|(uint64(r0)+jxrInt32Bias))>>32 != 0 {
				return 0, false
			}
			c := uint32(min(high, max(0, (r0+round)>>scale)))
			result[0], result[1], result[2] = c, c, c
			return 3, true
		}
		if p.color > 3 || p.channels != 3 {
			return 0, false
		}
		v0, v1, v2 := values[0], values[1], values[2]
		t := -v1
		g := v0 - (t >> 1)
		r := t + g - ((v2 + 1) >> 1)
		b := v2 + r
		r0, g0, b0 := r+off, g+off, b+off
		ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) |
			(uint64(t) + jxrInt32Bias) | (uint64(g) + jxrInt32Bias) | (uint64(r) + jxrInt32Bias) | (uint64(b) + jxrInt32Bias) |
			(uint64(r0) + jxrInt32Bias) | (uint64(g0) + jxrInt32Bias) | (uint64(b0) + jxrInt32Bias)
		if ov>>32 != 0 {
			return 0, false
		}
		result[0] = uint32(min(high, max(0, (r0+round)>>scale)))
		result[1] = uint32(min(high, max(0, (g0+round)>>scale)))
		result[2] = uint32(min(high, max(0, (b0+round)>>scale)))
		return 3, true
	case 4:
		if p.color != 4 || p.channels != 4 || len(values) < 4 || len(result) < 4 {
			return 0, false
		}
		halfOff := (bias >> 1) << scale
		v0, v1, v2, v3 := values[0], values[1], values[2], values[3]
		k := v3 + (v0 >> 1)
		m := k - v0 - (v1 >> 1)
		cv := v1 + m + (v2 >> 1)
		yv := cv - v2
		c0, m0, y0, k0 := cv+halfOff, m+halfOff, yv+halfOff, k-halfOff
		ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) | (uint64(v3) + jxrInt32Bias) |
			(uint64(k) + jxrInt32Bias) | (uint64(m) + jxrInt32Bias) | (uint64(cv) + jxrInt32Bias) | (uint64(yv) + jxrInt32Bias) |
			(uint64(c0) + jxrInt32Bias) | (uint64(m0) + jxrInt32Bias) | (uint64(y0) + jxrInt32Bias) | (uint64(k0) + jxrInt32Bias)
		if ov>>32 != 0 {
			return 0, false
		}
		result[0] = uint32(min(high, max(0, (c0+round)>>scale)))
		result[1] = uint32(min(high, max(0, (m0+round)>>scale)))
		result[2] = uint32(min(high, max(0, (y0+round)>>scale)))
		result[3] = uint32(min(high, max(0, (k0+round)>>scale)))
		return 4, true
	case 5:
		if p.color != 4 || p.channels != 4 || len(values) < 4 || len(result) < 4 {
			return 0, false
		}
		v0, v1, v2, v3 := values[0], values[1], values[2], values[3]
		c0, m0, y0, k0 := v1+off, v2+off, v3+off, v0+off
		ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) | (uint64(v3) + jxrInt32Bias) |
			(uint64(c0) + jxrInt32Bias) | (uint64(m0) + jxrInt32Bias) | (uint64(y0) + jxrInt32Bias) | (uint64(k0) + jxrInt32Bias)
		if ov>>32 != 0 {
			return 0, false
		}
		result[0] = uint32(min(high, max(0, (c0+round)>>scale)))
		result[1] = uint32(min(high, max(0, (m0+round)>>scale)))
		result[2] = uint32(min(high, max(0, (y0+round)>>scale)))
		result[3] = uint32(min(high, max(0, (k0+round)>>scale)))
		return 4, true
	case 1, 2, 3:
		if len(values) < 3 || len(result) < 3 {
			return 0, false
		}
		if color == 3 && p.color == 0 {
			if p.channels != 1 {
				return 0, false
			}
			v0 := values[0]
			y0 := v0 + off
			if ((uint64(v0)+jxrInt32Bias)|(uint64(y0)+jxrInt32Bias))>>32 != 0 {
				return 0, false
			}
			result[0] = uint32(min(high, max(0, (y0+round)>>scale)))
			z := uint32(min(high, max(0, (off+round)>>scale)))
			result[1], result[2] = z, z
			return 3, true
		}
		if color != 3 && p.color != color || color == 3 && p.color > 3 || p.channels != 3 {
			return 0, false
		}
		v0, v1, v2 := values[0], values[1], values[2]
		y0, u0, w0 := v0+off, v1+off, v2+off
		ov := (uint64(v0) + jxrInt32Bias) | (uint64(v1) + jxrInt32Bias) | (uint64(v2) + jxrInt32Bias) |
			(uint64(y0) + jxrInt32Bias) | (uint64(u0) + jxrInt32Bias) | (uint64(w0) + jxrInt32Bias)
		if ov>>32 != 0 {
			return 0, false
		}
		result[0] = uint32(min(high, max(0, (y0+round)>>scale)))
		result[1] = uint32(min(high, max(0, (u0+round)>>scale)))
		result[2] = uint32(min(high, max(0, (w0+round)>>scale)))
		return 3, true
	}
	return 0, false
}

// decodeSubsampled 在独立网格中保留YCC420或YCC422分量，不进行未请求的色度上采样
// 入参: ctx 取消上下文, state 重建主平面, views 主平面样本视图, alpha 透明度视图, alphaHeader 透明度头, alphaPlane 透明度参数, maxPixels RGBA64等价输出容量
// 返回: *Raster 原生网格图像, error 分量、数值、资源或取消错误
func (s *jxrRasterSource) decodeSubsampled(ctx context.Context, state *jxrCoefficientImage, views []jxrComponentView, alpha *jxrComponentView, alphaHeader *jxrHeader, alphaPlane *jxrCoefficients, maxPixels uint64) (*Raster, error) {
	count := s.sampleCount()
	units := count / 4
	if count%4 != 0 {
		units++
	}
	if count > uint64(^uint(0)>>1)/s.sampleBytes() || units > maxPixels {
		return nil, fmt.Errorf("%w: subsampled output storage", ErrLimit)
	}
	info := s.info()
	raster := &Raster{info: info, planes: make([]imageComponents, len(info.Components))}
	opacityNumber := s.opacityFormat().number
	for index, component := range info.Components {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		samples := component.Bounds.Dx() * component.Bounds.Dy()
		plane := imageComponents{bounds: component.Bounds, channels: 1}
		if component.Precision == 32 {
			plane.words = make([]uint32, samples)
		} else {
			plane.values = make([]uint16, samples)
		}
		sx, sy := 1, 1
		if index > 0 && index < info.Channels {
			sx = 2
			if s.container.format.color == 1 {
				sy = 2
			}
		}
		width, height := int(s.container.width)/sx, int(s.container.height)/sy
		for y := range height {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for x := range width {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				var values [3]int64
				var bits uint32
				if index == info.Channels {
					if alpha == nil || alphaPlane == nil {
						return nil, fmt.Errorf("%w: subsampled alpha plane", ErrInvalid)
					}
					values[0] = alpha.at(x+int(alphaHeader.left), y+int(alphaHeader.top))
					var codes [1]uint32
					if _, err := jxrRasterBitsComponents(alphaHeader, &alphaPlane.plane, opacityNumber, values[:], alphaHeader.alpha, codes[:]); err != nil {
						return nil, err
					}
					bits = codes[0]
				} else {
					values[index] = views[index].at(x+int(state.header.left)/sx, y+int(state.header.top)/sy)
					if _, err := jxrIntegerOutputComponents(&state.header, &state.primary.plane, values[:], false); err != nil {
						return nil, err
					}
					bits = uint32(values[index])
				}
				px, py := jxrRasterPosition(x, y, width, height, uint8(s.container.transform))
				plane.storeBits(py*component.Bounds.Dx()+px, bits)
			}
		}
		raster.planes[index] = plane
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return raster, nil
}

// sampleJXRChroma 按T.832表180依次进行垂直及水平上采样，不分配整幅中间图像
// 入参: v 色度样本视图, p 平面参数, component 色度分量编号, x, y 亮度样本坐标
// 返回: int64 上采样后的色度值
func sampleJXRChroma(v *jxrComponentView, p *jxrPlane, component, x, y int) int64 {
	chromaX, chromaY := p.chromaPhase(component)
	k, neighbor, weight := jxrSamplePosition(x, v.width, chromaX)
	if p.color == 1 {
		row, next, vertical := jxrSamplePosition(y, v.height, chromaY)
		a := ((8-vertical)*v.at(k, row) + vertical*v.at(k, next) + 4) >> 3
		b := ((8-vertical)*v.at(neighbor, row) + vertical*v.at(neighbor, next) + 4) >> 3
		return ((8-weight)*a + weight*b + 4) >> 3
	}
	return ((8-weight)*v.at(k, y) + weight*v.at(neighbor, y) + 4) >> 3
}

// jxrSamplePosition 计算两抽头采样位置及相邻样本权重，未指定中心时采用半样本位置
// 入参: position 输出坐标, length 输入长度, centering 四分之一样本单位的中心位置
// 返回: int 当前样本位置, int 相邻样本位置, int64 相邻样本权重
func jxrSamplePosition(position, length int, centering uint8) (int, int, int64) {
	if centering > 4 {
		centering = 2
	}
	k := position / 2
	if position&1 == 0 {
		return k, max(0, k-1), int64(centering)
	}
	return k, min(length-1, k+1), 4 - int64(centering)
}

// Resample 将原生分量显式重采样到指定大小的完整采样网格，保持颜色空间、ICC及预乘状态
// 像素中心映射为(x+0.5)*原宽/新宽-0.5，随后按分量已知偏移和间隔定位，边界采用端点延拓
// 最近邻中点选择较大坐标并保留位模式，RGBE双线性尚不支持
// 双线性落在原样点时保留位模式，其余插值拒绝非有限贡献；分辨率随尺寸调整以保持物理大小
// 未知分量相位返回错误；不猜测YCC或多通道的RGB解释，不替代压缩域降分辨率解码
// 入参: ctx 取消上下文，nil采用Background, size 输出可见尺寸, opts 滤波及输出限制
// 返回: *Raster 独立完整采样原生图像, error 参数、相位、能力、数值、资源或取消错误
func (r *Raster) Resample(ctx context.Context, size image.Point, opts *ResampleOptions) (*Raster, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	result, err := r.resample(ctx, size, opts)
	if err != nil {
		return nil, wrapJXRError("Raster.Resample", err)
	}
	return result, nil
}

// resample 执行原生分量的几何重采样，由公开入口统一归一化错误
// 入参: ctx 已规范化取消上下文, size 输出可见尺寸, opts 滤波及输出限制
// 返回: *Raster 独立完整采样原生图像, error 参数、相位、能力、数值、资源或取消错误
func (r *Raster) resample(ctx context.Context, size image.Point, opts *ResampleOptions) (*Raster, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Bounds().Empty() || size.X < 1 || size.Y < 1 || len(r.info.Components) == 0 {
		return nil, fmt.Errorf("%w: resampling geometry", ErrInvalid)
	}
	filter, limits := ResampleNearest, DefaultLimits()
	if opts != nil {
		filter, limits = opts.Filter, opts.Limits.normalized()
	}
	if filter > ResampleBilinear {
		return nil, fmt.Errorf("%w: resampling filter", ErrInvalid)
	}
	metadata := uint64(512) + uint64(len(r.info.Components))*128
	if metadata >= limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(metadata, 1), Detail: "resampling output storage"}
	}
	if uint64(size.X) > limits.MaxPixels/uint64(size.Y) {
		return nil, &LimitError{Resource: "max pixels", Limit: limits.MaxPixels, Required: jxrSaturatedMul(uint64(size.X), uint64(size.Y)), Detail: "resampling output pixels"}
	}
	for _, component := range r.info.Components {
		if !component.XPositionKnown || !component.YPositionKnown {
			return nil, fmt.Errorf("%w: resampling requires known sample positions", ErrUnsupported)
		}
		if filter == ResampleBilinear && component.Format == SampleRGBE {
			return nil, fmt.Errorf("%w: bilinear resampling sample encoding", ErrUnsupported)
		}
	}
	info := r.info
	info.Bounds, info.PixelFormat = image.Rectangle{Max: size}, ""
	info.Components = cloneComponentInfo(info.Components)
	info.Resolution.X *= float64(size.X) / float64(r.Bounds().Dx())
	info.Resolution.Y *= float64(size.Y) / float64(r.Bounds().Dy())
	for index := range info.Components {
		component := &info.Components[index]
		component.Bounds, component.XStep, component.YStep = info.Bounds, 1, 1
		component.XOffset, component.YOffset = 0, 0
		component.XPositionKnown, component.YPositionKnown = true, true
	}
	limits.MaxMemoryBytes -= metadata
	output, err := newRasterContext(ctx, info, limits)
	if err != nil {
		return nil, err
	}
	for index := range info.Components {
		source, err := r.component(index)
		if err != nil {
			return nil, err
		}
		for y := range size.Y {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			y0, y1, fy := resampleJXRAxis(y, size.Y, r.Bounds().Dy(), source.info.Bounds.Dy(), source.info.YOffset, source.info.YStep)
			if filter == ResampleNearest && fy >= 0.5 {
				y0 = y1
			}
			for x := range size.X {
				if x&1023 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				x0, x1, fx := resampleJXRAxis(x, size.X, r.Bounds().Dx(), source.info.Bounds.Dx(), source.info.XOffset, source.info.XStep)
				if filter == ResampleNearest {
					if fx >= 0.5 {
						x0 = x1
					}
					at, err := source.sampleIndex(x0, y0)
					if err != nil {
						return nil, err
					}
					output.planes[index].copySample(y*size.X+x, source.source, at)
					continue
				}
				bits, err := resampleJXRBilinear(source, x0, x1, y0, y1, fx, fy)
				if err != nil {
					return nil, err
				}
				output.planes[index].storeBits(y*size.X+x, bits)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// resampleJXRAxis 将完整输出网格中心映射到原分量网格，返回端点延拓后的相邻坐标及权重
// 入参: at 输出坐标, target 输出轴长, original 原图轴长, count 原分量轴长, offset 分量相位, step 采样间隔
// 返回: int 较小坐标, int 较大坐标, float64 较大坐标权重
func resampleJXRAxis(at, target, original, count int, offset float64, step uint8) (int, int, float64) {
	position := ((float64(at)+0.5)*float64(original)/float64(target) - 0.5 - offset) / float64(step)
	position = max(0, min(float64(count-1), position))
	low := int(math.Floor(position))
	return low, min(low+1, count-1), position - float64(low)
}

// resampleJXRBilinear 在原生码值或浮点数值域滤波，不通过单位区间往返损失整数中点
// 入参: source 原分量, x0,x1,y0,y1 相邻坐标, fx,fy 插值权重
// 返回: uint32 目标编码位模式, error 坐标、非有限贡献或数值错误
func resampleJXRBilinear(source Component, x0, x1, y0, y1 int, fx, fy float64) (uint32, error) {
	if fx == 0 && fy == 0 {
		return source.sampleBitsAt(x0, y0)
	}
	value, err := resampleJXRBilinearValue(source, x0, x1, y0, y1, fx, fy)
	if err != nil {
		return 0, err
	}
	if source.info.Format == SampleFloat {
		return sampleValueBits(value, source.info)
	}
	return uint32(int64(math.RoundToEven(value))) & uint32((uint64(1)<<source.info.Precision)-1), nil
}

// resampleJXRBilinearValue 在原生数值域计算未量化的双线性结果
// 入参: source 原生分量, x0, x1 水平邻点坐标, y0, y1 垂直邻点坐标, fx, fy 双线性插值权重
// 返回: float64 插值结果, error 样本或非有限数值错误
func resampleJXRBilinearValue(source Component, x0, x1, y0, y1 int, fx, fy float64) (float64, error) {
	var values [4]float64
	for index, point := range [4]image.Point{{X: x0, Y: y0}, {X: x1, Y: y0}, {X: x0, Y: y1}, {X: x1, Y: y1}} {
		if index&1 != 0 && fx == 0 || index&2 != 0 && fy == 0 {
			continue
		}
		value, err := resampleJXRNumericSample(source, point)
		if err != nil {
			return 0, err
		}
		if !geometryFinite(value) {
			return 0, fmt.Errorf("%w: nonfinite resampling contribution", ErrInvalid)
		}
		values[index] = value
	}
	top := values[0]*(1-fx) + values[1]*fx
	bottom := values[2]*(1-fx) + values[3]*fx
	return top*(1-fy) + bottom*fy, nil
}

// resampleJXRNumericSample 读取未归一化的整数或原生浮点样本
// 入参: source 原生分量, point 分量网格坐标
// 返回: float64 原始数值样本, error 读取或不支持的格式错误
func resampleJXRNumericSample(source Component, point image.Point) (float64, error) {
	bits, err := source.sampleBitsAt(point.X, point.Y)
	if err != nil {
		return 0, err
	}
	switch source.info.Format {
	case SampleUnsigned:
		return float64(bits), nil
	case SampleFixed:
		if source.info.Precision == 16 {
			return float64(int16(bits)), nil
		}
		return float64(int32(bits)), nil
	case SampleFloat:
		return sampleValue(bits, source.info), nil
	default:
		return 0, fmt.Errorf("%w: resampling numeric format", ErrUnsupported)
	}
}

// makeJXRPixelFormats 构建标准像素格式GUID末字节到内部格式描述的静态查找表
// 返回: [0x57]jxrPixelFormat 按末字节索引的标准像素格式表
func makeJXRPixelFormats() [0x57]jxrPixelFormat {
	var table [0x57]jxrPixelFormat
	for id := byte(0x05); id <= 0x56; id++ {
		f := jxrPixelFormat{channels: 3, color: 7, depth: 1, number: jxrUnsigned}
		switch id {
		case 0x05:
			f.name, f.channels, f.color, f.depth = "BlackWhite", 1, 0, 0
		case 0x08:
			f.name, f.channels, f.color = "8bppGray", 1, 0
		case 0x09:
			f.name, f.depth = "16bppBGR555", 8
		case 0x0a:
			f.name, f.depth = "16bppBGR565", 10
		case 0x0b:
			f.name, f.channels, f.color, f.depth = "16bppGray", 1, 0, 2
		case 0x0c:
			f.name = "24bppBGR"
		case 0x0d:
			f.name = "24bppRGB"
		case 0x0e:
			f.name = "32bppBGR"
		case 0x0f, 0x10:
			f.name, f.alpha = "32bppBGRA", true
			if id == 0x10 {
				f.name, f.premultiply = "32bppPBGRA", true
			}
		case 0x11:
			f.name, f.channels, f.color, f.depth, f.number = "32bppGrayFloat", 1, 0, 7, jxrFloat
		case 0x12, 0x13:
			f.name, f.depth, f.number = "48bppRGBFixedPoint", 3, jxrFixed16
			if id == 0x13 {
				f.name, f.channels, f.color = "16bppGrayFixedPoint", 1, 0
			}
		case 0x14:
			f.name, f.depth = "32bppBGR101010", 9
		case 0x15:
			f.name, f.depth = "48bppRGB", 2
		case 0x16, 0x17:
			f.name, f.depth, f.alpha = "64bppRGBA", 2, true
			if id == 0x17 {
				f.name, f.premultiply = "64bppPRGBA", true
			}
		case 0x18:
			f.name, f.depth, f.number = "96bppRGBFixedPoint", 6, jxrFixed32
		case 0x19, 0x1a, 0x1b:
			f.name, f.depth, f.number, f.alpha = "128bppRGBAFloat", 7, jxrFloat, true
			if id == 0x1a {
				f.name, f.premultiply = "128bppPRGBAFloat", true
			} else if id == 0x1b {
				f.name, f.alpha = "128bppRGBFloat", false
			}
		case 0x1c, 0x1f, 0x2c, 0x2d, 0x43, 0x54, 0x55, 0x56:
			f.name, f.channels, f.color = "32bppCMYK", 4, 4
			if id == 0x1f || id == 0x2d || id == 0x43 || id == 0x55 {
				f.name, f.depth = "64bppCMYK", 2
			}
			if id == 0x2c || id == 0x2d || id == 0x43 || id == 0x56 {
				f.alpha = true
			}
			if id >= 0x43 && id != 0x2c && id != 0x2d {
				f.color = 5
			}
			if f.color == 5 {
				if f.depth == 2 {
					f.name = "64bppCMYKDIRECT"
				} else {
					f.name = "32bppCMYKDIRECT"
				}
			}
			if f.alpha {
				switch {
				case f.color == 4 && f.depth == 1:
					f.name = "40bppCMYKAlpha"
				case f.color == 4:
					f.name = "80bppCMYKAlpha"
				case f.depth == 1:
					f.name = "40bppCMYKDIRECTAlpha"
				default:
					f.name = "80bppCMYKDIRECTAlpha"
				}
			}
		case 0x1d, 0x1e:
			f.name, f.depth, f.number, f.alpha = "64bppRGBAFixedPoint", 3, jxrFixed16, true
			if id == 0x1e {
				f.name, f.depth, f.number = "128bppRGBAFixedPoint", 6, jxrFixed32
			}
		case 0x20, 0x21, 0x22, 0x23, 0x24, 0x25:
			f.color, f.channels = 6, int(id-0x20)+3
			f.name = fmt.Sprintf("%dbpp%dChannels", f.channels*8, f.channels)
		case 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b:
			f.color, f.depth, f.channels = 6, 2, int(id-0x26)+3
			f.name = fmt.Sprintf("%dbpp%dChannels", f.channels*16, f.channels)
		case 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33:
			f.color, f.channels, f.alpha = 6, int(id-0x2e)+3, true
			f.name = fmt.Sprintf("%dbpp%dChannelsAlpha", (f.channels+1)*8, f.channels)
		case 0x34, 0x35, 0x36, 0x37, 0x38, 0x39:
			f.color, f.depth, f.channels, f.alpha = 6, 2, int(id-0x34)+3, true
			f.name = fmt.Sprintf("%dbpp%dChannelsAlpha", (f.channels+1)*16, f.channels)
		case 0x3a, 0x3b:
			f.name, f.depth, f.number = "48bppRGBHalf", 4, jxrHalf
			if id == 0x3a {
				f.name, f.alpha = "64bppRGBAHalf", true
			}
		case 0x3d:
			f.name, f.color, f.number = "32bppRGBE", 8, jxrRGBE
		case 0x3e, 0x3f:
			f.name, f.channels, f.color, f.depth, f.number = "16bppGrayHalf", 1, 0, 4, jxrHalf
			if id == 0x3f {
				f.name, f.depth, f.number = "32bppGrayFixedPoint", 6, jxrFixed32
			}
		case 0x40, 0x41, 0x42:
			f.name, f.depth, f.number = "64bppRGBFixedPoint", 3, jxrFixed16
			if id == 0x41 {
				f.name, f.depth, f.number = "128bppRGBFixedPoint", 6, jxrFixed32
			} else if id == 0x42 {
				f.name, f.depth, f.number = "64bppRGBHalf", 4, jxrHalf
			}
		case 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53:
			if id == 0x44 || id == 0x4c {
				f.name, f.color = "12bppYCC420", 1
			} else if id <= 0x47 || id >= 0x4d && id <= 0x4f {
				f.name, f.color = "16bppYCC422", 2
			} else {
				f.name, f.color = "24bppYCC444", 3
			}
			if id == 0x46 || id == 0x49 || id == 0x4e || id == 0x51 {
				f.depth = 9
			} else if id == 0x47 || id == 0x4a || id == 0x4f || id == 0x52 {
				f.depth = 2
			} else if id == 0x4b || id == 0x53 {
				f.depth, f.number = 3, jxrFixed16
			}
			f.alpha = id >= 0x4c
			switch id {
			case 0x46:
				f.name = "20bppYCC422"
			case 0x47:
				f.name = "32bppYCC422"
			case 0x49:
				f.name = "30bppYCC444"
			case 0x4a:
				f.name = "48bppYCC444"
			case 0x4b:
				f.name = "48bppYCC444FixedPoint"
			case 0x4c:
				f.name = "20bppYCC420Alpha"
			case 0x4d:
				f.name = "24bppYCC422Alpha"
			case 0x4e:
				f.name = "30bppYCC422Alpha"
			case 0x4f:
				f.name = "48bppYCC422Alpha"
			case 0x50:
				f.name = "32bppYCC444Alpha"
			case 0x51:
				f.name = "40bppYCC444Alpha"
			case 0x52:
				f.name = "64bppYCC444Alpha"
			case 0x53:
				f.name = "64bppYCC444AlphaFixedPoint"
			}
		default:
			continue
		}
		table[id] = f
	}
	return table
}
