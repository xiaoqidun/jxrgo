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
	"fmt"
	"math"
)

// jxrCICPDirectSRGB 标识直接输出sRGB而无需额外传递转换的内部代码
const jxrCICPDirectSRGB = 255

// jxrBT2100Red 等常量定义BT.2100亮度权重以及PQ与HLG传递特性常数
const (
	jxrBT2100Red   = 0.2627
	jxrBT2100Green = 0.6780
	jxrBT2100Blue  = 0.0593
	jxrPQC1        = 107.0 / 128
	jxrPQC2        = 2413.0 / 128
	jxrPQC3        = 2392.0 / 128
	jxrPQM1        = 1305.0 / 8192
	jxrPQM2        = 2523.0 / 32
	jxrHLGA        = 0.17883277
	jxrHLGB        = 1 - 4*jxrHLGA
	jxrHLGC        = 0.5599107295295620
)

var (
	// jxrBT2100Luma 保存表5指定的BT.2020场景亮度权重
	jxrBT2100Luma = [3]float64{jxrBT2100Red, jxrBT2100Green, jxrBT2100Blue}
	// jxrBT2020ToSRGB 将方法A输出转换到线性sRGB
	jxrBT2020ToSRGB, _, _, _ = jxrCICPPrimaries(9)
	// jxrSRGBToBT2020 将线性sRGB坐标转换到方法A的输入基色
	jxrSRGBToBT2020 = jxrInvertCompositionMatrix(jxrBT2020ToSRGB)

	// jxrBT2446HDRRho 保存方法A表2中1000 cd/m²输入的ρ减一
	jxrBT2446HDRRho = 32 * math.Pow(1000.0/10000, 1/2.4)
	// jxrBT2446SDRRho 保存方法A表2中100 cd/m²输出的ρ减一
	jxrBT2446SDRRho = 32 * math.Pow(100.0/10000, 1/2.4)
	// jxrBT2446HDRLog 保存HDR的ρ对数
	jxrBT2446HDRLog = math.Log1p(jxrBT2446HDRRho)
	// jxrBT2446SDRLog 保存SDR的ρ对数
	jxrBT2446SDRLog = math.Log1p(jxrBT2446SDRRho)
	// jxrIPTToLMSPrime 反转SMPTE ST 2128式1的非线性LMS到IPT矩阵
	jxrIPTToLMSPrime = jxrInvertCompositionMatrix([3][3]float64{
		{1638.0 / 4096, 1638.0 / 4096, 820.0 / 4096},
		{18248.0 / 4096, -19870.0 / 4096, 1622.0 / 4096},
		{3300.0 / 4096, 1463.0 / 4096, -4763.0 / 4096},
	})
	// jxrIPTLMS2RGB 反转SMPTE ST 2128式2的线性BT.2100 RGB到LMS矩阵
	jxrIPTLMS2RGB = jxrInvertCompositionMatrix([3][3]float64{
		{1747.0 / 4096, 2169.0 / 4096, 180.0 / 4096},
		{673.0 / 4096, 3029.0 / 4096, 394.0 / 4096},
		{50.0 / 4096, 207.0 / 4096, 3839.0 / 4096},
	})
)

// jxrXYZToSRGBD65 保存由BT.709/sRGB标准色度及D65白点反解的D65 XYZ到线性sRGB矩阵
var jxrXYZToSRGBD65 = func() [3][3]float64 {
	xy := [3][2]float64{{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}}
	var xyz [3][3]float64
	for i, p := range xy {
		xyz[i] = [3]float64{p[0], p[1], 1 - p[0] - p[1]}
	}
	var rgbToXYZ [3][3]float64
	for i, p := range xyz {
		u, v := xyz[(i+1)%3], xyz[(i+2)%3]
		x, y, z := u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]
		w := (x*jxrD65White[0] + y*jxrD65White[1] + z*jxrD65White[2]) / (x*p[0] + y*p[1] + z*p[2])
		rgbToXYZ[0][i], rgbToXYZ[1][i], rgbToXYZ[2][i] = w*p[0], w*p[1], w*p[2]
	}
	return jxrInvertCompositionMatrix(rgbToXYZ)
}()

// jxrBradfordForward 保存线性Bradford色适应矩阵及其逆矩阵与D65参考白
var (
	jxrBradfordForward = [3][3]float64{
		{0.8951, 0.2664, -0.1614},
		{-0.7502, 1.7135, 0.0367},
		{0.0389, -0.0685, 1.0296},
	}
	jxrBradfordInverse = jxrInvertCompositionMatrix(jxrBradfordForward)
	jxrD65White        = [3]float64{0.3127 / 0.3290, 1, (1 - 0.3127 - 0.3290) / 0.3290}
)

// jxrCICPColorDisplay 保存H.273信号的范围、矩阵及传递参数
type jxrCICPColorDisplay struct {
	matrix     [3][3]float64
	toBT2020   *[3][3]float64
	offset     [3]float64
	scale      [3]float64
	hlgLuma    [3]float64
	red        float64
	blue       float64
	transfer   uint16
	matrixCode uint16
	ycgco      jxrCICPYCgCo
	constant   jxrCICPConstantLuminance
}

// jxrCICPYCgCo 保存H.273第8.3节的整数分量范围及YCgCo或YCgCo-R反变换参数
type jxrCICPYCgCo struct {
	yMax, cMax, center float64
	rgbMax, offset     float64
	scale              float64
	lifted             bool
}

// jxrCICPConstantLuminance 保存H.273式68至75中按色差符号选择的幅值
type jxrCICPConstantLuminance struct {
	blueNegative, bluePositive float64
	redNegative, redPositive   float64
}

// jxrCICPDisplay 按H.273第8节准备SDR、数字影院XYZ、PQ或HLG到sRGB的转换，PQ和HLG采用BT.2446方法A
// 入参: info 原生分量描述, specification PTM颜色声明
// 返回: jxrDisplayMapping 转换, bool 是否支持声明及样本组合
func jxrCICPDisplay(info Info, specification ColorEncoding) (jxrDisplayMapping, bool) {
	var mapping jxrDisplayMapping
	matrixCode := uint16(specification.Matrix)
	ycc := info.ColorSpace == ColorYCC && info.Channels == 3
	gray := info.ColorSpace == ColorGray && info.Channels == 1
	rgb := info.ColorSpace == ColorRGB && info.Channels == 3
	xyz := specification.Primaries == 10
	layout := rgb && matrixCode == 0 || ycc && (matrixCode != 0 || xyz) || gray && !xyz
	if !layout || len(info.Components) < info.Channels ||
		info.Premultiplied && (!specification.FullRange || ycc) {
		return mapping, false
	}
	transfer := uint16(specification.Transfer)
	if xyz && !jxrCICPXYZEncoding(specification) || transfer == 17 && !xyz {
		return mapping, false
	}
	if info.Linear && (!rgb || transfer != 8 || !specification.FullRange) {
		return mapping, false
	}
	hdr := transfer == 16 || transfer == 18
	hdrMatrix := matrixCode == 14 || matrixCode == 15
	ycgco := matrixCode == 8 || matrixCode == 16 || matrixCode == 17
	constant := matrixCode == 10 || matrixCode == 13
	if hdr && !jxrCICPHDRMatrixSupported(specification) ||
		hdrMatrix && !hdr || (hdrMatrix || ycgco || constant || matrixCode == 11) && !ycc {
		return mapping, false
	}
	switch transfer {
	case 6, 14, 15:
		transfer = 1
	case 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18:
	default:
		return mapping, false
	}
	matrix := jxrXYZToSRGBD65
	var red, blue float64
	var ok bool
	if !xyz {
		matrix, red, blue, ok = jxrCICPPrimaries(uint16(specification.Primaries))
		if !ok {
			return mapping, false
		}
	}
	switch matrixCode {
	case 0, 8, 11, 14, 15, 16, 17:
	case 12, 13:
		if red < 0 || blue < 0 || red+blue >= 1 {
			return mapping, false
		}
	case 1:
		red, blue = 0.2126, 0.0722
	case 4:
		red, blue = 0.30, 0.11
	case 5, 6:
		red, blue = 0.299, 0.114
	case 7:
		red, blue = 0.212, 0.087
	case 9, 10:
		red, blue = jxrBT2100Red, jxrBT2100Blue
	default:
		return mapping, false
	}
	display := &mapping.cicp
	for index, component := range info.Components[:info.Channels] {
		if info.Linear && (component.Format == SampleFloat || component.Format == SampleFixed) {
			display.scale[index] = 1
			continue
		}
		bits := int(component.Precision)
		if component.Format != SampleUnsigned || bits < 1 || ycgco && bits > 32 ||
			!ycgco && bits < 8 && (!specification.FullRange || ycc && matrixCode != 0) {
			return jxrDisplayMapping{}, false
		}
		if ycgco {
			continue
		}
		maximum := math.Ldexp(1, bits) - 1
		offset, span := 0.0, maximum
		colorDifference := ycc && matrixCode != 0 && index != 0
		if specification.FullRange {
			if colorDifference {
				offset = math.Ldexp(1, bits-1)
			}
		} else {
			step := math.Ldexp(1, bits-8)
			offset, span = 16*step, 219*step
			if colorDifference {
				offset, span = 128*step, 224*step
			}
		}
		display.offset[index], display.scale[index] = offset/maximum, maximum/span
	}
	if !ycc && specification.Primaries == 1 && specification.FullRange && (transfer == 8 || transfer == 13) {
		mapping.linear = transfer == 8
		return mapping, true
	}
	if ycc {
		display.red, display.blue = red, blue
	}
	if ycgco {
		display.ycgco, ok = jxrNewCICPYCgCo(info, matrixCode, specification.FullRange)
		if !ok {
			return jxrDisplayMapping{}, false
		}
	}
	if constant {
		display.constant = jxrNewCICPConstantLuminance(red, blue, transfer)
	}
	if hdr {
		if transfer == 18 {
			display.hlgLuma, _ = jxrHLGLuma(specification)
		}
		if specification.Primaries != 9 {
			toBT2020, err := jxrComposeColorMatrices(matrix, jxrSRGBToBT2020)
			if err != nil {
				return jxrDisplayMapping{}, false
			}
			display.toBT2020 = &toBT2020
		}
		matrix = jxrBT2020ToSRGB
	}
	display.matrix, display.transfer, display.matrixCode = matrix, transfer, matrixCode
	if !constant && (gray || specification.Primaries == 1) && transfer == 13 {
		display.transfer = jxrCICPDirectSRGB
	}
	return mapping, true
}

// jxrCICPPrimaries 用H.273表2色度和BT.2380式2.1至2.3求相对XYZ，按线性Bradford适配到sRGB
// 入参: code H.273 RGB基色编号
// 返回: [3][3]float64 线性源RGB到线性sRGB, float64 源红及蓝的亮度权重, bool 是否已定义
func jxrCICPPrimaries(code uint16) ([3][3]float64, float64, float64, bool) {
	const d65x, d65y = 0.3127, 0.3290
	wx, wy := d65x, d65y
	var xy [3][2]float64
	switch code {
	case 1:
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, 87098.0 / 409605, 12673.0 / 175545, true
	case 4:
		xy, wx, wy = [3][2]float64{{0.67, 0.33}, {0.21, 0.71}, {0.14, 0.08}}, 0.310, 0.316
	case 5:
		xy = [3][2]float64{{0.64, 0.33}, {0.29, 0.60}, {0.15, 0.06}}
	case 6, 7:
		xy = [3][2]float64{{0.630, 0.340}, {0.310, 0.595}, {0.155, 0.070}}
	case 8:
		xy, wx, wy = [3][2]float64{{0.681, 0.319}, {0.243, 0.692}, {0.145, 0.049}}, 0.310, 0.316
	case 9:
		xy = [3][2]float64{{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}}
	case 11, 12:
		xy = [3][2]float64{{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}}
		if code == 11 {
			wx, wy = 0.314, 0.351
		}
	case 22:
		xy = [3][2]float64{{0.630, 0.340}, {0.295, 0.605}, {0.155, 0.077}}
	default:
		return [3][3]float64{}, 0, 0, false
	}
	return jxrRGBChromaticityMatrix(xy, [2]float64{wx, wy})
}

// jxrRGBChromaticityMatrix 用原白点求基色权重，再按线性Bradford适配到sRGB
// 入参: xy 红绿蓝基色色度坐标, whiteXY 白点色度坐标
// 返回: [3][3]float64 线性RGB到sRGB矩阵, float64 红基色亮度权重, float64 蓝基色亮度权重, bool 矩阵是否有效
func jxrRGBChromaticityMatrix(xy [3][2]float64, whiteXY [2]float64) ([3][3]float64, float64, float64, bool) {
	wx, wy := whiteXY[0], whiteXY[1]
	if wx <= 0 || wy <= 0 || wx+wy >= 1 {
		return [3][3]float64{}, 0, 0, false
	}
	white := [3]float64{wx / wy, 1, (1 - wx - wy) / wy}
	xyzToRGB := jxrDisplayXYZMatrix(white)
	var xyz, toRGB [3][3]float64
	var weights [3]float64
	for index, primary := range xy {
		xyz[index] = [3]float64{primary[0], primary[1], 1 - primary[0] - primary[1]}
	}
	for index, primary := range xyz {
		u, v := xyz[(index+1)%3], xyz[(index+2)%3]
		x, y, z := u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]
		determinant := x*primary[0] + y*primary[1] + z*primary[2]
		weights[index] = (x*white[0] + y*white[1] + z*white[2]) / determinant
		if !geometryFinite(weights[index]) || weights[index] == 0 {
			return [3][3]float64{}, 0, 0, false
		}
		for out, row := range xyzToRGB {
			toRGB[out][index] = weights[index] * (row[0]*primary[0]/white[0] + row[1]*primary[1] + row[2]*primary[2]/white[2])
			if !geometryFinite(toRGB[out][index]) {
				return [3][3]float64{}, 0, 0, false
			}
		}
	}
	return toRGB, weights[0] * xyz[0][1], weights[2] * xyz[2][1], true
}

// color 按H.273第8.3节逆变换，再应用传递函数和源基色转换
// 入参: info 原生描述, samples 归一化原生样本, alpha 原透明度
// 返回: float64 非预乘红、绿、蓝sRGB值
func (display *jxrCICPColorDisplay) color(info Info, samples []float64, alpha float64) (float64, float64, float64) {
	if info.Premultiplied {
		if !(alpha > 0) {
			return 0, 0, 0
		}
		jxrUnpremultiplyDisplay(samples[:info.Channels], alpha)
	}
	var rgb [3]float64
	for index := range info.Channels {
		rgb[index] = (samples[index] - display.offset[index]) * display.scale[index]
	}
	hdr := display.transfer == 16 || display.transfer == 18
	if info.Channels == 1 && !hdr {
		value := rgb[0]
		if display.transfer != jxrCICPDirectSRGB {
			value = displayLinear(jxrCICPInverseTransfer(value, display.transfer, false))
		}
		return value, value, value
	}
	ycc := info.ColorSpace == ColorYCC
	constant := display.matrixCode == 10 || display.matrixCode == 13
	if ycc {
		switch display.matrixCode {
		case 0:
			rgb = [3]float64{rgb[2], rgb[0], rgb[1]}
		case 8, 16, 17:
			rgb = display.ycgco.rgb(samples)
		case 10, 13:
			rgb = display.constant.linear(rgb, display.red, display.blue, display.transfer)
		case 11:
			rgb = jxrCICPSMPTE2085ToRGB(rgb)
		case 14, 15:
		default:
			rgb = jxrCICPYCCToRGB(rgb, display.red, display.blue)
		}
	}
	if display.transfer == jxrCICPDirectSRGB {
		return rgb[0], rgb[1], rgb[2]
	}
	if hdr {
		if info.Channels == 1 {
			rgb[1], rgb[2] = rgb[0], rgb[0]
		}
		if !constant {
			rgb = jxrBT2100Linear(rgb, display.transfer, display.matrixCode)
		}
		if display.transfer == 18 {
			rgb = jxrHLGDisplayLight(rgb, display.hlgLuma)
		}
		if display.toBT2020 != nil {
			rgb = jxrApplyColorMatrix(*display.toBT2020, rgb)
		}
		rgb = jxrBT2446MethodA(rgb)
	} else if !constant {
		for index, value := range rgb {
			rgb[index] = jxrCICPInverseTransfer(value, display.transfer, ycc)
		}
	}
	var srgb [3]float64
	for index, row := range display.matrix {
		srgb[index] = displayLinear(row[0]*rgb[0] + row[1]*rgb[1] + row[2]*rgb[2])
	}
	return srgb[0], srgb[1], srgb[2]
}

// jxrCICPYCCToRGB 反解H.273式45至47的非恒定亮度颜色矩阵
// 入参: signal 全范围亮度及带符号色差, kr,kb 红及蓝的亮度权重
// 返回: [3]float64 相同传递域中的RGB
func jxrCICPYCCToRGB(signal [3]float64, kr, kb float64) [3]float64 {
	y, cb, cr := signal[0], signal[1], signal[2]
	red := y + 2*(1-kr)*cr
	blue := y + 2*(1-kb)*cb
	green := (y - kr*red - kb*blue) / (1 - kr - kb)
	return [3]float64{red, green, blue}
}

// jxrCICPInverseTransfer 反解H.273表3的SDR传递，分段常数满足8.2的值和斜率连续条件
// 入参: value 非线性信号, transfer 表3编号, ycc 是否采用非零矩阵的有符号sRGB分支
// 返回: float64 线性分量
func jxrCICPInverseTransfer(value float64, transfer uint16, ycc bool) float64 {
	switch transfer {
	case 4:
		return math.Pow(max(0, value), 2.2)
	case 5:
		return math.Pow(max(0, value), 2.8)
	case 8:
		return value
	case 9, 10:
		return math.Pow(10, jxrCICPLogDecades(transfer)*(max(0, value)-1))
	case 17:
		return (52.37 / 48) * math.Pow(colorClamp(value), 2.6)
	}
	alpha, beta, slope, power := jxrCICPPiecewiseTransfer(transfer)
	if transfer == 12 && value < -slope*beta/4 {
		return -math.Pow((alpha-1-4*value)/alpha, power) / 4
	}
	magnitude := value
	if transfer == 11 || transfer == 13 && ycc {
		magnitude = math.Abs(value)
	}
	if magnitude < beta*slope {
		return value / slope
	}
	return math.Copysign(math.Pow((magnitude+alpha-1)/alpha, power), value)
}

// jxrCICPForwardSDR 应用H.273表3的标称SDR传递函数
// 入参: value 零至一的线性分量, transfer 已验证的传递代码
// 返回: float64 编码分量
func jxrCICPForwardSDR(value float64, transfer uint16) float64 {
	switch transfer {
	case 4:
		return math.Pow(value, 1/2.2)
	case 5:
		return math.Pow(value, 1/2.8)
	case 8:
		return value
	case 9, 10:
		return max(0, 1+math.Log10(value)/jxrCICPLogDecades(transfer))
	}
	alpha, beta, slope, power := jxrCICPPiecewiseTransfer(transfer)
	if value < beta {
		return slope * value
	}
	return alpha*math.Pow(value, 1/power) - (alpha - 1)
}

// jxrCICPLogDecades 返回H.273表3的对数动态范围，零信号码反解到对数段起点
// 入参: transfer 对数传递特性代码
// 返回: float64 动态范围十倍程数
func jxrCICPLogDecades(transfer uint16) float64 {
	if transfer == 9 {
		return 2
	}
	return 2.5
}

// jxrCICPPiecewiseTransfer 返回H.273第8.2节连续分段的参数，正反变换共用边界
// 入参: transfer 已选择的分段传递代码
// 返回: float64 增益、线性分段边界、线性斜率及逆传递幂次
func jxrCICPPiecewiseTransfer(transfer uint16) (alpha, beta, slope, power float64) {
	switch transfer {
	case 7:
		return 1.1115721959217313, 0.022821585529445028, 4, 1 / 0.45
	case 13:
		return 1.0550107189475866, 0.003041282560127518, 12.92, 2.4
	default:
		return 1.099296826809442, 0.018053968510807, 4.5, 1 / 0.45
	}
}

// jxrCICPHDRMatrixSupported 核对H.273矩阵和已知HDR传递域、基色的组合
// 入参: color PTM颜色编码声明
// 返回: bool 组合是否受支持
func jxrCICPHDRMatrixSupported(color ColorEncoding) bool {
	if color.Primaries == 10 {
		return color.Transfer == 16 && jxrCICPXYZEncoding(color)
	}
	if color.Transfer == 18 {
		if _, ok := jxrHLGLuma(color); !ok {
			return false
		}
	}
	switch color.Matrix {
	case 0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17:
		return color.Transfer == 16 || color.Transfer == 18
	case 10, 14:
		return color.Primaries == 9
	case 15:
		return color.Primaries == 9 && color.Transfer == 16
	}
	return false
}

// jxrCICPXYZEncoding 识别H.273的XYZ直传和SMPTE ST 2085的PQ色差信号
// 入参: color PTM颜色编码声明
// 返回: bool 是否为支持的XYZ编码组合
func jxrCICPXYZEncoding(color ColorEncoding) bool {
	return color.Matrix == 0 && (color.Transfer == 8 || color.Transfer == 16 || color.Transfer == 17) ||
		color.Matrix == 11 && color.Transfer == 16
}

// jxrHLGLuma 按BT.2390第6.2节取得源基色和源白点的场景亮度权重
// 入参: color PTM颜色编码声明
// 返回: [3]float64 RGB场景亮度权重, bool 权重是否有效
func jxrHLGLuma(color ColorEncoding) ([3]float64, bool) {
	if color.Primaries == 9 {
		return jxrBT2100Luma, true
	}
	_, red, blue, ok := jxrCICPPrimaries(uint16(color.Primaries))
	green := 1 - red - blue
	return [3]float64{red, green, blue}, ok && red >= 0 && green >= 0 && blue >= 0
}

// jxrBT2100Linear 依BT.2100-3表4、5及7恢复相对1000 cd/m²的PQ显示光或HLG场景光
// 入参: signal 非线性RGB或归一化ICtCp/IPT, transfer H.273的PQ或HLG编号, matrix H.273矩阵编号
// 返回: [3]float64 源基色的线性RGB，ICtCp和IPT使用BT.2020基色
func jxrBT2100Linear(signal [3]float64, transfer, matrix uint16) [3]float64 {
	switch matrix {
	case 14:
		i, ct, cp := signal[0], signal[1], signal[2]
		var difference, short float64
		if transfer == 16 {
			difference = (1112064.0/129174029)*ct + (14342144.0/129174029)*cp
			short = i + (72341504.0/129174029)*ct - (41416704.0/129174029)*cp
		} else {
			difference = (6144.0/390875)*ct + (16384.0/78175)*cp
			short = i + (1197568.0/1172625)*ct - (141952.0/234525)*cp
		}
		signal = [3]float64{i + difference, i - difference, short}
	case 15:
		signal = jxrApplyColorMatrix(jxrIPTToLMSPrime, signal)
	}
	for index, value := range signal {
		signal[index] = jxrHDRInverseTransfer(value, transfer)
	}
	switch matrix {
	case 14:
		l, m, s := signal[0], signal[1], signal[2]
		signal = [3]float64{
			(1074053.0/312533)*l - (783349.0/312533)*m + (21829.0/312533)*s,
			-(1236583.0/1562665)*l + (3099703.0/1562665)*m - (60091.0/312533)*s,
			-(40551.0/1562665)*l - (154569.0/1562665)*m + (351557.0/312533)*s,
		}
	case 15:
		signal = jxrApplyColorMatrix(jxrIPTLMS2RGB, signal)
	}
	return signal
}

// jxrHDRInverseTransfer 将PQ转换为相对1000 cd/m²的显示光或将HLG转换为场景光
// 入参: value 非线性HDR分量, transfer H.273传递特性代码
// 返回: float64 线性光强度
func jxrHDRInverseTransfer(value float64, transfer uint16) float64 {
	value = colorClamp(value)
	if transfer == 16 {
		root := math.Pow(value, 1/jxrPQM2)
		return 10 * math.Pow(max(root-jxrPQC1, 0)/(jxrPQC2-jxrPQC3*root), 1/jxrPQM1)
	}
	if value <= 0.5 {
		return value * value / 3
	}
	return (math.Exp((value-jxrHLGC)/jxrHLGA) + jxrHLGB) / 12
}

// jxrHLGDisplayLight 按BT.2100-3表5和BT.2390第6.2节将场景光转换为参考显示光
// 入参: rgb 线性场景RGB, weights 场景亮度权重
// 返回: [3]float64 参考显示光RGB
func jxrHLGDisplayLight(rgb, weights [3]float64) [3]float64 {
	luma := weights[0]*rgb[0] + weights[1]*rgb[1] + weights[2]*rgb[2]
	gain := math.Pow(max(0, luma), 1.2-1)
	for i := range rgb {
		rgb[i] *= gain
	}
	return rgb
}

// jxrBT2446MethodA 按BT.2446-1第4.1节将1000 cd/m²显示光映射到100 cd/m²
// 输入显示光先裁到方法A的定义域，输出按零黑位BT.1886线性化
// 入参: rgb BT.2020线性显示RGB，相对1000 cd/m²
// 返回: [3]float64 BT.2020线性SDR显示RGB
func jxrBT2446MethodA(rgb [3]float64) [3]float64 {
	for index, value := range rgb {
		rgb[index] = math.Pow(colorClamp(value), 1/2.4)
	}
	luma := jxrBT2100Red*rgb[0] + jxrBT2100Green*rgb[1] + jxrBT2100Blue*rgb[2]
	if !(luma > 0) {
		return [3]float64{}
	}
	perceptual := math.Log1p(jxrBT2446HDRRho*luma) / jxrBT2446HDRLog
	var compressed float64
	switch {
	case perceptual <= 0.7399:
		compressed = 1.0770 * perceptual
	case perceptual < 0.9909:
		compressed = (-1.1510*perceptual+2.7811)*perceptual - 0.6302
	default:
		compressed = 0.5*perceptual + 0.5
	}
	sdr := math.Expm1(jxrBT2446SDRLog*compressed) / jxrBT2446SDRRho
	gain := sdr / (1.1 * luma)
	cb, cr := gain*(rgb[2]-luma)/1.8814, gain*(rgb[0]-luma)/1.4746
	rgb = jxrCICPYCCToRGB([3]float64{sdr - max(0.1*cr, 0), cb, cr}, jxrBT2100Red, jxrBT2100Blue)
	for index, value := range rgb {
		rgb[index] = math.Pow(max(0, value), 2.4)
	}
	return rgb
}

// jxrNewCICPYCgCo 按H.273表4及式23至26推导矩阵8、16、17的RGB位数与量化范围
// 入参: info 已校验的三个无符号分量, matrix 矩阵编号, full 是否为全范围
// 返回: jxrCICPYCgCo 转换参数, bool 是否为已定义的位深组合
func jxrNewCICPYCgCo(info Info, matrix uint16, full bool) (jxrCICPYCgCo, bool) {
	y, c := info.Components[0].Precision, info.Components[1].Precision
	if c != info.Components[2].Precision {
		return jxrCICPYCgCo{}, false
	}
	bits, lifted := int(y), c != y || matrix != 8
	switch matrix {
	case 8:
		if c != y && c != y+1 {
			return jxrCICPYCgCo{}, false
		}
	case 16, 17:
		if c != y {
			return jxrCICPYCgCo{}, false
		}
		bits -= 18 - int(matrix)
	}
	if bits < 1 || !full && bits < 8 {
		return jxrCICPYCgCo{}, false
	}
	quant := jxrCICPYCgCo{
		yMax: float64((uint64(1) << y) - 1), cMax: float64((uint64(1) << c) - 1),
		center: float64(uint64(1) << (c - 1)), rgbMax: float64((uint64(1) << bits) - 1),
		lifted: lifted,
	}
	quant.scale = 1 / quant.rgbMax
	if !full {
		unit := float64(uint64(1) << (bits - 8))
		quant.offset, quant.scale = 16*unit, 1/(219*unit)
	}
	return quant, true
}

// rgb 按H.273式54至57或62至65恢复非线性RGB，再撤销RGB量化
// YCgCo-R对已插值的分量先四舍五入到码值，整数源保留可逆提升的有符号右移
// 入参: samples 归一化的Y、Cg、Co码值
// 返回: [3]float64 非线性RGB
func (quant *jxrCICPYCgCo) rgb(samples []float64) [3]float64 {
	y, cg, co := samples[0]*quant.yMax, samples[1]*quant.cMax, samples[2]*quant.cMax
	var rgb [3]float64
	if quant.lifted {
		y, cg, co = math.Round(y), math.Round(cg)-quant.center, math.Round(co)-quant.center
		t := y - math.Floor(cg/2)
		b := min(quant.rgbMax, max(0, t-math.Floor(co/2)))
		rgb = [3]float64{b + co, t + cg, b}
	} else {
		cg, co = cg-quant.center, co-quant.center
		t := y - cg
		rgb = [3]float64{t + co, y + cg, t - co}
	}
	for index, code := range rgb {
		rgb[index] = (min(quant.rgbMax, max(0, code)) - quant.offset) * quant.scale
	}
	return rgb
}

// jxrCICPSMPTE2085ToRGB 反解H.273式76至78的亮度及色差信号
// 入参: signal 归一化YDzDx信号
// 返回: [3]float64 线性三刺激分量
func jxrCICPSMPTE2085ToRGB(signal [3]float64) [3]float64 {
	y, dz, dx := signal[0], signal[1], signal[2]
	return [3]float64{2*dx + 0.991902*y, y, (2*dz + y) / 0.986566}
}

// jxrNewCICPConstantLuminance 为矩阵10或13计算所选传递函数对应的四个色差幅值
// 入参: kr,kb 红蓝亮度权重, transfer 已支持的H.273传递代码
// 返回: jxrCICPConstantLuminance 色差系数
func jxrNewCICPConstantLuminance(kr, kb float64, transfer uint16) jxrCICPConstantLuminance {
	encode := func(value float64) float64 {
		switch transfer {
		case 16:
			power := math.Pow(value, jxrPQM1)
			return math.Pow((jxrPQC1+jxrPQC2*power)/(1+jxrPQC3*power), jxrPQM2)
		case 18:
			if value <= 1.0/12 {
				return math.Sqrt(3 * value)
			}
			return jxrHLGA*math.Log(12*value-jxrHLGB) + jxrHLGC
		}
		return jxrCICPForwardSDR(value, transfer)
	}
	return jxrCICPConstantLuminance{
		blueNegative: 2 * encode(1-kb), bluePositive: 2 * (1 - encode(kb)),
		redNegative: 2 * encode(1-kr), redPositive: 2 * (1 - encode(kr)),
	}
}

// linear 从恒定亮度信号恢复线性RGB，绿色由线性亮度方程求得
// 入参: signal 已撤销量化的亮度及色差, kr,kb 红蓝权重, transfer 已支持的SDR传递代码
// 返回: [3]float64 线性RGB，PQ以1000 cd/m²参考显示光为单位，HLG为场景光
func (constant *jxrCICPConstantLuminance) linear(signal [3]float64, kr, kb float64, transfer uint16) [3]float64 {
	y, cb, cr := signal[0], signal[1], signal[2]
	b, r := constant.bluePositive, constant.redPositive
	if cb < 0 {
		b = constant.blueNegative
	}
	if cr < 0 {
		r = constant.redNegative
	}
	linear := [3]float64{y + r*cr, y, y + b*cb}
	for i, value := range linear {
		if transfer == 16 || transfer == 18 {
			linear[i] = jxrHDRInverseTransfer(value, transfer)
		} else {
			linear[i] = jxrCICPInverseTransfer(value, transfer, true)
		}
	}
	linear[1] = (linear[1] - kr*linear[0] - kb*linear[2]) / (1 - kr - kb)
	return linear
}

// jxrApplyColorMatrix 将三个颜色分量变换到目标坐标系
// 入参: matrix 三阶线性变换矩阵, color 输入三通道向量
// 返回: [3]float64 变换后的三通道向量
func jxrApplyColorMatrix(matrix [3][3]float64, color [3]float64) [3]float64 {
	var out [3]float64
	for i, row := range matrix {
		out[i] = row[0]*color[0] + row[1]*color[1] + row[2]*color[2]
	}
	return out
}

// jxrComposeColorMatrices 按应用顺序合成两个线性颜色变换并校验输出系数
// 入参: first 先应用的变换矩阵, second 后应用的变换矩阵
// 返回: [3][3]float64 复合变换矩阵, error 非有限系数错误
func jxrComposeColorMatrices(first, second [3][3]float64) ([3][3]float64, error) {
	var result [3][3]float64
	for i := range result {
		for j := range result[i] {
			for k := range first {
				result[i][j] += second[i][k] * first[k][j]
			}
			if !geometryFinite(result[i][j]) {
				return [3][3]float64{}, fmt.Errorf("%w: primary conversion matrix", ErrInvalid)
			}
		}
	}
	return result, nil
}

// jxrDisplayXYZMatrix 计算由源白点归一化XYZ经线性Bradford适配至D65并转换到线性sRGB的矩阵
// 入参: white 源参考白点XYZ坐标
// 返回: [3][3]float64 归一化XYZ到线性sRGB矩阵
func jxrDisplayXYZMatrix(white [3]float64) [3][3]float64 {
	srcCone := jxrApplyColorMatrix(jxrBradfordForward, white)
	dstCone := jxrApplyColorMatrix(jxrBradfordForward, jxrD65White)
	scale := [3][3]float64{
		{dstCone[0] / srcCone[0], 0, 0},
		{0, dstCone[1] / srcCone[1], 0},
		{0, 0, dstCone[2] / srcCone[2]},
	}
	step, _ := jxrComposeColorMatrices(jxrBradfordForward, scale)
	adapt, _ := jxrComposeColorMatrices(step, jxrBradfordInverse)
	mapped, _ := jxrComposeColorMatrices(adapt, jxrXYZToSRGBD65)
	for i := range mapped {
		mapped[i][0] *= white[0]
		mapped[i][1] *= white[1]
		mapped[i][2] *= white[2]
	}
	return mapped
}

// jxrInvertCompositionMatrix 求三阶可逆矩阵的逆矩阵
// 入参: m 三阶可逆矩阵
// 返回: [3][3]float64 逆矩阵
func jxrInvertCompositionMatrix(m [3][3]float64) [3][3]float64 {
	c00 := m[1][1]*m[2][2] - m[1][2]*m[2][1]
	c01 := -(m[1][0]*m[2][2] - m[1][2]*m[2][0])
	c02 := m[1][0]*m[2][1] - m[1][1]*m[2][0]
	c10 := -(m[0][1]*m[2][2] - m[0][2]*m[2][1])
	c11 := m[0][0]*m[2][2] - m[0][2]*m[2][0]
	c12 := -(m[0][0]*m[2][1] - m[0][1]*m[2][0])
	c20 := m[0][1]*m[1][2] - m[0][2]*m[1][1]
	c21 := -(m[0][0]*m[1][2] - m[0][2]*m[1][0])
	c22 := m[0][0]*m[1][1] - m[0][1]*m[1][0]
	det := m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02
	return [3][3]float64{
		{c00 / det, c10 / det, c20 / det},
		{c01 / det, c11 / det, c21 / det},
		{c02 / det, c12 / det, c22 / det},
	}
}
