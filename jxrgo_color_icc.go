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
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
)

// jxrICCDisplayWork 保存ICC显示映射解析基础预算字节数
const jxrICCDisplayWork = uint64(512)

// jxrICCMaxChannels 等常量定义ICC整数查表的通道上限、网格轴上限及基础工作预算
const (
	jxrICCMaxChannels  = 15
	jxrICCMaxGridAxes  = 16
	jxrICCLUTWork      = uint64(1536)
	jxrICCMultiLUTWork = uint64(4096)
)

// jxrICCFloatWork 保存ICC浮点管线基础结构工作内存估算字节数
const jxrICCFloatWork = uint64(2048)

// jxrICCSampledSegment 标识ICC浮点分段曲线中的采样查表段类型
const jxrICCSampledSegment = uint16(3)

// jxrICCTag 保存ICC目录中标签的签名及其有效负载范围
type jxrICCTag struct {
	signature uint32
	start     uint32
	length    uint32
}

// jxrICCTransform 保存媒体相对色度的矩阵曲线、整数查表或浮点管线及目标转换
type jxrICCTransform struct {
	curves   [3]jxrICCCurve
	matrix   [3][3]float64
	channels int
	grayLab  bool
	lut      *jxrICCLUT
	floating *jxrICCFloatTransform
}

// jxrDisplayMapping 保存已选择的内建显示转换及其源数值解释
type jxrDisplayMapping struct {
	icc    jxrICCTransform
	cicp   jxrCICPColorDisplay
	linear bool
}

// jxrICCCurve 保存ICC曲线函数或引用原配置中按大端存储的等距样本
type jxrICCCurve struct {
	function  uint16
	params    [7]float64
	threshold float64
	table     []byte
}

// jxrICCLUT 保存到三分量PCS的查表模型，表值引用原配置
type jxrICCLUT struct {
	grid                        jxrICCGrid
	input, output               []byte
	inputEntries, outputEntries int
	lab                         bool
	legacyLab                   bool
	stages                      *jxrICCStages
}

// jxrICCGrid 保存交错网格及各输入轴的字节跨度
type jxrICCGrid struct {
	data      []byte
	strides   [jxrICCMaxGridAxes]int
	points    [jxrICCMaxGridAxes]byte
	precision int
}

// jxrICCGridAxis 保存单元内的小数位置及相邻顶点的字节跨度
type jxrICCGridAxis struct {
	fraction float64
	stride   int
}

// jxrICCFloatTransform 保存DToB浮点管线与单次显示操作重用的通道工作区
type jxrICCFloatTransform struct {
	stages  []jxrICCFloatStage
	scratch []float64
	lab     bool
}

// jxrICCFloatStage 保存已校验的浮点处理元素及其输出通道数
type jxrICCFloatStage struct {
	kind    string
	outputs int
	matrix  []byte
	curves  []jxrICCSegmentedCurve
	grid    *jxrICCGrid
}

// jxrICCPosition 保存嵌套ICC元素的物理范围、逻辑顺序与共享来源
type jxrICCPosition struct {
	start, end    int
	index, shared int
}

// jxrICCSegmentedCurve 保存ICC浮点曲线的有序分段
type jxrICCSegmentedCurve struct {
	segments []jxrICCFloatSegment
}

// jxrICCFloatSegment 保存分段边界及对原始公式参数或样本的引用
type jxrICCFloatSegment struct {
	lower, upper float64
	left         float64
	data         []byte
	function     uint16
}

// jxrICCStages 保存mAB查表元素的曲线与仿射矩阵
type jxrICCStages struct {
	a      [jxrICCMaxChannels]jxrICCCurve
	b, m   [3]jxrICCCurve
	matrix [12]float64
	affine bool
}

// jxrICCStageExtent 保存一个已验证元素及其对齐填充范围
type jxrICCStageExtent struct {
	start, end int
	curve      bool
}

// jxrICCStageLayout 跟踪元素范围，允许曲线数据共享
type jxrICCStageLayout struct {
	data    []byte
	extents [jxrICCMaxChannels + 8]jxrICCStageExtent
	count   int
}

// jxrReadICCTransform 在预算内读取ICC设备到PCS的矩阵曲线或查表转换
// 入参: ctx 已规范化Context, profile 原始ICC, info 解码分量, maximum 活动工作内存
// 返回: jxrICCTransform 转换, error 配置格式、转换能力、资源或取消错误
func jxrReadICCTransform(ctx context.Context, profile []byte, info Info, maximum uint64) (jxrICCTransform, error) {
	var result jxrICCTransform
	if err := ctx.Err(); err != nil {
		return result, err
	}
	profile, err := jxrICCProfileView(profile)
	if err != nil {
		return result, err
	}
	class, device, pcs := string(profile[12:16]), string(profile[16:20]), string(profile[20:24])
	switch class {
	case "scnr", "mntr", "prtr", "spac":
	default:
		return result, fmt.Errorf("%w: ICC profile class", ErrUnsupported)
	}
	result.channels = jxrICCDeviceChannels(device, info)
	if result.channels == 0 || len(info.Components) < result.channels || pcs != "XYZ " && pcs != "Lab " {
		return jxrICCTransform{}, fmt.Errorf("%w: ICC device channels or connection space", ErrUnsupported)
	}
	unsigned := true
	for _, component := range info.Components[:result.channels] {
		if component.Format == SampleUnsigned {
			continue
		}
		if component.Format != SampleFixed && component.Format != SampleFloat {
			return jxrICCTransform{}, fmt.Errorf("%w: ICC input sample encoding", ErrUnsupported)
		}
		unsigned = false
	}
	tags, err := jxrReadICCDirectory(ctx, profile, maximum)
	if err != nil {
		return jxrICCTransform{}, err
	}
	white, err := jxrICCXYZ(jxrICCData(profile, tags, 0x77747074))
	if err != nil || white[0] <= 0 || white[1] <= 0 || white[2] <= 0 {
		return jxrICCTransform{}, fmt.Errorf("%w: ICC media white point", ErrInvalid)
	}
	available := maximum - uint64(len(tags))*16
	if profile[8] == 4 {
		if data := jxrICCData(profile, tags, 0x44324231); data != nil {
			result.floating, err = jxrReadICCFloatTransform(ctx, data, result.channels, pcs == "Lab ", available)
			if err == nil {
				result.matrix = jxrICCXYZToSRGB()
				return result, ctx.Err()
			}
			if !errors.Is(err, ErrUnsupported) {
				return jxrICCTransform{}, err
			}
		}
	}
	if !unsigned {
		return jxrICCTransform{}, fmt.Errorf("%w: ICC extended samples require a floating-point transform", ErrUnsupported)
	}
	for _, id := range [...]uint32{0x41324231, 0x41324230} {
		if data := jxrICCData(profile, tags, id); data != nil {
			result.lut, err = jxrReadICCLUT(data, result.channels, pcs == "Lab ", available)
			if err != nil {
				return jxrICCTransform{}, err
			}
			result.matrix = jxrICCXYZToSRGB()
			return result, ctx.Err()
		}
	}
	if class == "spac" || device != "GRAY" && (device != "RGB " || pcs != "XYZ " || class == "prtr") {
		return jxrICCTransform{}, fmt.Errorf("%w: ICC profile transform", ErrUnsupported)
	}
	result.grayLab = pcs == "Lab "
	if result.channels == 1 {
		result.curves[0], err = jxrReadICCCurve(jxrICCData(profile, tags, 0x6b545243))
		return result, err
	}
	for col, id := range [...]uint32{0x72545243, 0x67545243, 0x62545243} {
		result.curves[col], err = jxrReadICCCurve(jxrICCData(profile, tags, id))
		if err != nil {
			return jxrICCTransform{}, err
		}
	}
	colorants, err := jxrICCColorants(profile, tags)
	if err != nil {
		return jxrICCTransform{}, err
	}
	result.matrix = jxrICCsRGBMatrix(colorants)
	return result, ctx.Err()
}

// jxrICCColorants 读取三个PCS XYZ色度矩阵列
// 入参: profile ICC配置字节, tags 已排序标签目录
// 返回: [3][3]float64 RGB到PCS XYZ矩阵, error 标签或数值错误
func jxrICCColorants(profile []byte, tags []jxrICCTag) ([3][3]float64, error) {
	var colorants [3][3]float64
	for col, id := range [...]uint32{0x7258595a, 0x6758595a, 0x6258595a} {
		value, err := jxrICCXYZ(jxrICCData(profile, tags, id))
		if err != nil {
			return [3][3]float64{}, err
		}
		colorants[col] = value
	}
	return colorants, nil
}

// jxrICCProfileView 校验ICC头并将读取范围限制在配置声明的大小内
// 入参: profile 原始ICC配置字节
// 返回: []byte 有效ICC配置切片, error 头部、长度或版本错误
func jxrICCProfileView(profile []byte) ([]byte, error) {
	if len(profile) < 132 || string(profile[36:40]) != "acsp" {
		return nil, fmt.Errorf("%w: ICC profile header", ErrInvalid)
	}
	size := uint64(binary.BigEndian.Uint32(profile))
	if size < 132 || size > uint64(len(profile)) || size%4 != 0 {
		return nil, fmt.Errorf("%w: ICC profile size", ErrInvalid)
	}
	if profile[8] != 2 && profile[8] != 4 {
		return nil, fmt.Errorf("%w: ICC profile version", ErrUnsupported)
	}
	return profile[:size], nil
}

// jxrICCDeviceChannels 核对已知设备通道顺序及ICC xCLR的声明数量
// 入参: device ICC设备色彩签名, info 原生分量描述
// 返回: int 匹配的通道数，零表示无可用解释
func jxrICCDeviceChannels(device string, info Info) int {
	count := 0
	switch device {
	case "GRAY":
		if info.ColorSpace == ColorGray || info.ColorSpace == ColorMultiChannel {
			count = 1
		}
	case "RGB ":
		if info.ColorSpace == ColorRGB || info.ColorSpace == ColorMultiChannel {
			count = 3
		}
	case "CMYK":
		if info.ColorSpace == ColorCMYK || info.ColorSpace == ColorCMYKDirect || info.ColorSpace == ColorMultiChannel {
			count = 4
		}
	default:
		if info.ColorSpace == ColorMultiChannel && device[1:] == "CLR" {
			switch prefix := device[0]; {
			case prefix >= '2' && prefix <= '9':
				count = int(prefix - '0')
			case prefix >= 'A' && prefix <= 'F':
				count = int(prefix - 'A' + 10)
			}
		}
	}
	if count != info.Channels {
		return 0
	}
	return count
}

// jxrReadICCDirectory 校验标签边界与重叠，允许多个签名引用同一完整标签
// 入参: ctx 已规范化Context, profile 已限定的ICC范围, maximum 工作额度
// 返回: []jxrICCTag 按签名排序的目录, error 格式、资源或取消错误
func jxrReadICCDirectory(ctx context.Context, profile []byte, maximum uint64) ([]jxrICCTag, error) {
	count := uint64(binary.BigEndian.Uint32(profile[128:]))
	size := uint64(len(profile))
	if count > (size-132)/12 {
		return nil, fmt.Errorf("%w: ICC tag count", ErrInvalid)
	}
	if maximum < jxrICCDisplayWork || count > (maximum-jxrICCDisplayWork)/16 || count > uint64(^uint(0)>>1)/16 {
		return nil, fmt.Errorf("%w: ICC tag directory storage", ErrLimit)
	}
	first := 132 + 12*count
	tags := make([]jxrICCTag, int(count))
	for i := range tags {
		if i&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		at := 132 + i*12
		tag := jxrICCTag{binary.BigEndian.Uint32(profile[at:]), binary.BigEndian.Uint32(profile[at+4:]), binary.BigEndian.Uint32(profile[at+8:])}
		start, length := uint64(tag.start), uint64(tag.length)
		if start%4 != 0 || start < first || start > size || length < 8 || length > size-start {
			return nil, fmt.Errorf("%w: ICC tag extent", ErrInvalid)
		}
		tags[i] = tag
	}
	slices.SortFunc(tags, func(a, b jxrICCTag) int {
		return cmp.Compare(a.start, b.start)
	})
	for i := 1; i < len(tags); i++ {
		if i&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		a, b := tags[i-1], tags[i]
		if a.start == b.start && a.length == b.length {
			continue
		}
		if uint64(a.start)+uint64(a.length) > uint64(b.start) {
			return nil, fmt.Errorf("%w: overlapping ICC tags", ErrInvalid)
		}
	}
	slices.SortFunc(tags, func(a, b jxrICCTag) int {
		return cmp.Compare(a.signature, b.signature)
	})
	for i := 1; i < len(tags); i++ {
		if tags[i-1].signature == tags[i].signature {
			return nil, fmt.Errorf("%w: duplicate ICC tag", ErrInvalid)
		}
	}
	return tags, ctx.Err()
}

// jxrICCData 查找已验证的指定标签，切片引用原始配置
// 入参: profile ICC数据, tags 已排序目录, signature 标签签名
// 返回: []byte 原始标签，缺省时为nil
func jxrICCData(profile []byte, tags []jxrICCTag, signature uint32) []byte {
	index, found := slices.BinarySearchFunc(tags, signature, func(tag jxrICCTag, value uint32) int {
		return cmp.Compare(tag.signature, value)
	})
	if !found {
		return nil
	}
	tag := tags[index]
	return profile[tag.start : uint64(tag.start)+uint64(tag.length)]
}

// jxrICCXYZ 读取一个ICC XYZType三刺激值
// 入参: data 单一XYZ标签
// 返回: [3]float64 有符号定点XYZ, error 标签格式错误
func jxrICCXYZ(data []byte) ([3]float64, error) {
	var xyz [3]float64
	if len(data) != 20 || string(data[:4]) != "XYZ " || binary.BigEndian.Uint32(data[4:]) != 0 {
		return xyz, fmt.Errorf("%w: ICC XYZ tag", ErrInvalid)
	}
	for i := range xyz {
		xyz[i] = jxrICCFixed(data[8+4*i:])
	}
	return xyz, nil
}

// jxrICCsRGBMatrix 合并源色度矩阵与ICC注册的sRGB2014色度矩阵之逆
// 入参: columns 源配置的D50 PCS色度列
// 返回: [3][3]float64 源线性分量到目标线性sRGB的矩阵
func jxrICCsRGBMatrix(columns [3][3]float64) [3][3]float64 {
	inverse := jxrICCXYZToSRGB()
	var result [3][3]float64
	for row := range 3 {
		for col := range 3 {
			for k := range 3 {
				result[row][col] += inverse[row][k] * columns[col][k]
			}
		}
	}
	return result
}

// jxrICCXYZToSRGB 返回ICC注册的sRGB2014色度矩阵之逆
// 返回: [3][3]float64 D50 PCS到线性sRGB的矩阵
func jxrICCXYZToSRGB() [3][3]float64 {
	return [3][3]float64{
		{3.134112153323845, -1.6173924595518705, -0.49063340456472027},
		{-0.9787872961391832, 1.9162795864266209, 0.03345471423138195},
		{0.07198304438569493, -0.22898585024754495, 1.4053851315844992},
	}
}

// color 应用ICC转换及目标sRGB传递函数
// 入参: ctx Context, values 已解除预乘的源通道
// 返回: float64 非预乘红、绿、蓝sRGB值, error 数值或取消错误
func (transform *jxrICCTransform) color(ctx context.Context, values []float64) (float64, float64, float64, error) {
	if transform.lut == nil && transform.floating == nil && transform.channels == 1 {
		gray := transform.curves[0].value(values[0])
		if transform.grayLab {
			gray = jxrICCLabInverse((100*gray + 16) / 116)
		}
		gray = displayLinear(gray)
		return gray, gray, gray, nil
	}
	var linear [3]float64
	switch {
	case transform.floating != nil:
		var err error
		linear, err = transform.floating.xyz(ctx, values[:transform.channels])
		if err != nil {
			return 0, 0, 0, err
		}
	case transform.lut != nil:
		linear = transform.lut.xyz(values[:transform.channels])
	default:
		for index := range linear {
			linear[index] = transform.curves[index].value(values[index])
		}
	}
	var rgb [3]float64
	for i, row := range transform.matrix {
		rgb[i] = displayLinear(row[0]*linear[0] + row[1]*linear[1] + row[2]*linear[2])
	}
	return rgb[0], rgb[1], rgb[2], nil
}

// jxrBasicRasterDisplay 选择原有的基本显示解释，保留显式色彩变换入口的能力界限
// 入参: info 原生元数据
// 返回: jxrDisplayMapping 基本显示转换, error 颜色能力错误
func jxrBasicRasterDisplay(info Info) (jxrDisplayMapping, error) {
	if len(info.ICCProfile) != 0 {
		return jxrDisplayMapping{}, fmt.Errorf("%w: display color interpretation", ErrUnsupported)
	}
	return jxrDisplayColorEncoding(info)
}

// jxrManagedRasterDisplay 启用支持的ICC模型，缺少ICC时沿用基本显示策略
// 入参: ctx 已规范化Context, info 原生元数据, maximum 色彩转换工作额度
// 返回: jxrDisplayMapping 显示转换, error 颜色、格式、资源或取消错误
func jxrManagedRasterDisplay(ctx context.Context, info Info, maximum uint64) (jxrDisplayMapping, error) {
	if len(info.ICCProfile) == 0 {
		return jxrBasicRasterDisplay(info)
	}
	icc, err := jxrReadICCTransform(ctx, info.ICCProfile, info, maximum)
	return jxrDisplayMapping{icc: icc}, err
}

// color 在输入设备域解除预乘，再应用ICC或已有基本显示策略
// 入参: ctx Context, info 原生描述, samples 可复用颜色工作值, alpha 原透明度
// 返回: float64 非预乘红、绿、蓝sRGB值, error 数值或取消错误
func (mapping *jxrDisplayMapping) color(ctx context.Context, info Info, samples []float64, alpha float64) (float64, float64, float64, error) {
	if mapping.cicp.transfer != 0 {
		red, green, blue := mapping.cicp.color(info, samples, alpha)
		return red, green, blue, nil
	}
	if mapping.icc.channels == 0 {
		red, green, blue := displayJXRColor(info, samples, alpha, mapping.linear)
		return red, green, blue, nil
	}
	if info.Premultiplied {
		jxrUnpremultiplyDisplay(samples[:info.Channels], alpha)
	}
	return mapping.icc.color(ctx, samples)
}

// jxrICCCurveSize 确定独立或嵌入曲线的有效字节长度
// 入参: data 曲线起始位置的可用字节
// 返回: int 曲线长度, error 格式或函数能力错误
func jxrICCCurveSize(data []byte) (int, error) {
	if len(data) < 12 || binary.BigEndian.Uint32(data[4:]) != 0 {
		return 0, fmt.Errorf("%w: ICC tone curve header", ErrInvalid)
	}
	var size uint64
	switch string(data[:4]) {
	case "curv":
		size = 12 + 2*uint64(binary.BigEndian.Uint32(data[8:]))
	case "para":
		function := binary.BigEndian.Uint16(data[8:])
		if binary.BigEndian.Uint16(data[10:]) != 0 {
			return 0, fmt.Errorf("%w: ICC parametric curve reserved field", ErrInvalid)
		}
		if function > 4 {
			return 0, fmt.Errorf("%w: ICC parametric curve function", ErrUnsupported)
		}
		size = uint64(12 + 4*[...]int{1, 3, 4, 5, 7}[function])
	default:
		return 0, fmt.Errorf("%w: ICC tone curve type", ErrUnsupported)
	}
	if size > uint64(len(data)) {
		return 0, fmt.Errorf("%w: ICC tone curve extent", ErrInvalid)
	}
	return int(size), nil
}

// jxrReadICCCurve 读取ICC的curveType及parametricCurveType函数0至4
// 入参: data 指定曲线的完整标签
// 返回: jxrICCCurve 已校验曲线, error 格式或函数能力错误
func jxrReadICCCurve(data []byte) (jxrICCCurve, error) {
	var curve jxrICCCurve
	size, err := jxrICCCurveSize(data)
	if err != nil {
		return curve, err
	}
	if size != len(data) {
		return curve, fmt.Errorf("%w: ICC tone curve extent", ErrInvalid)
	}
	if string(data[:4]) == "curv" {
		curve.params[0] = 1
		switch {
		case size == 14:
			curve.params[0] = float64(binary.BigEndian.Uint16(data[12:])) / 256
		case size > 14:
			curve.table = data[12:]
		}
		return curve, nil
	}
	curve.function = binary.BigEndian.Uint16(data[8:])
	for i := range (size - 12) / 4 {
		curve.params[i] = jxrICCFixed(data[12+4*i:])
	}
	if err := curve.checkDomain(); err != nil {
		return jxrICCCurve{}, err
	}
	return curve, nil
}

// checkDomain 验证参数化曲线幂函数在实际输入区间中有定义
// 返回: error 参数域错误
func (curve *jxrICCCurve) checkDomain() error {
	g, a, b := curve.params[0], curve.params[1], curve.params[2]
	if curve.function == 0 {
		if g < 0 {
			return fmt.Errorf("%w: ICC negative power at zero", ErrInvalid)
		}
		return nil
	}
	if curve.function < 3 {
		if a == 0 {
			return fmt.Errorf("%w: ICC parametric curve division", ErrInvalid)
		}
		curve.threshold = -b / a
	} else {
		curve.threshold = curve.params[4]
	}
	if curve.threshold > 1 {
		return nil
	}
	start := a*max(0, curve.threshold) + b
	if curve.function < 3 && curve.threshold >= 0 {
		start = 0
	}
	base := min(start, a+b)
	if base < 0 || base == 0 && g < 0 {
		return fmt.Errorf("%w: ICC parametric curve power domain", ErrInvalid)
	}
	return nil
}

// value 按已校验的ICC函数或一维查表解释归一化设备值
// 入参: input 已解除预乘的设备分量
// 返回: float64 裁剪到单位区间的曲线结果
func (curve *jxrICCCurve) value(input float64) float64 {
	x := colorClamp(input)
	if math.IsNaN(x) {
		x = 0
	}
	if len(curve.table) != 0 {
		end := len(curve.table)/2 - 1
		position := x * float64(end)
		lo := min(int(position), end)
		hi := min(lo+1, end)
		weight := position - float64(lo)
		a := float64(binary.BigEndian.Uint16(curve.table[2*lo:]))
		b := float64(binary.BigEndian.Uint16(curve.table[2*hi:]))
		return (a + (b-a)*weight) / 65535
	}
	p := &curve.params
	if curve.function == 0 {
		if p[0] == 1 {
			return x
		}
		return math.Pow(x, p[0])
	}
	if x < curve.threshold {
		switch curve.function {
		case 1:
			return 0
		case 2:
			return colorClamp(p[3])
		case 3:
			return colorClamp(p[3] * x)
		default:
			return colorClamp(p[3]*x + p[6])
		}
	}
	y := math.Pow(max(0, p[1]*x+p[2]), p[0])
	switch curve.function {
	case 2:
		y += p[3]
	case 4:
		y += p[5]
	}
	return colorClamp(y)
}

// jxrICCFixed 将ICC的s15Fixed16字段转为浮点值
// 入参: data 起始于字段的大端字节
// 返回: float64 有符号数值
func jxrICCFixed(data []byte) float64 {
	return float64(int32(binary.BigEndian.Uint32(data))) / 65536
}

// jxrReadICCLUT 读取ICC中A到B方向的三种整数查表模型
// 入参: data 完整标签, channels 源设备通道数, lab 是否为Lab PCS, maximum 查表工作额度
// 返回: *jxrICCLUT 已验证查表, error 格式、能力或额度错误
func jxrReadICCLUT(data []byte, channels int, lab bool, maximum uint64) (*jxrICCLUT, error) {
	if len(data) < 12 || binary.BigEndian.Uint32(data[4:]) != 0 || int(data[8]) != channels || data[9] != 3 {
		return nil, fmt.Errorf("%w: ICC lookup header or channels", ErrInvalid)
	}
	switch string(data[:4]) {
	case "mAB ":
		if maximum < jxrICCMultiLUTWork {
			return nil, fmt.Errorf("%w: ICC multi-stage lookup storage", ErrLimit)
		}
		return jxrReadICCStages(data, channels, lab)
	case "mft1":
		if !lab {
			return nil, fmt.Errorf("%w: ICC 8-bit XYZ lookup encoding", ErrUnsupported)
		}
	case "mft2":
	default:
		return nil, fmt.Errorf("%w: ICC lookup type", ErrUnsupported)
	}
	if maximum < jxrICCLUTWork {
		return nil, fmt.Errorf("%w: ICC lookup storage", ErrLimit)
	}
	return jxrReadICCLegacyLUT(data, channels, lab)
}

// jxrReadICCLegacyLUT 校验传统输入表、网格和输出表的精度及精确字节范围
// 入参: data 完整传统标签, channels 源设备通道数, lab 是否为Lab PCS
// 返回: *jxrICCLUT 已验证查表, error 格式错误
func jxrReadICCLegacyLUT(data []byte, channels int, lab bool) (*jxrICCLUT, error) {
	header, precision, inputs, outputs := 48, 1, 256, 256
	if string(data[:4]) == "mft2" {
		header, precision = 52, 2
		if len(data) < header {
			return nil, fmt.Errorf("%w: ICC 16-bit lookup header", ErrInvalid)
		}
		inputs = int(binary.BigEndian.Uint16(data[48:]))
		outputs = int(binary.BigEndian.Uint16(data[50:]))
	}
	if len(data) < header || data[11] != 0 || inputs < 2 || inputs > 4096 || outputs < 2 || outputs > 4096 {
		return nil, fmt.Errorf("%w: ICC lookup table header", ErrInvalid)
	}
	for index := range 9 {
		var expected uint32
		if index%4 == 0 {
			expected = 65536
		}
		if binary.BigEndian.Uint32(data[12+4*index:]) != expected {
			return nil, fmt.Errorf("%w: ICC device lookup matrix", ErrInvalid)
		}
	}
	inputEnd := header + channels*inputs*precision
	outputStart := len(data) - 3*outputs*precision
	if inputEnd > outputStart {
		return nil, fmt.Errorf("%w: ICC lookup table extent", ErrInvalid)
	}
	var points [jxrICCMaxGridAxes]byte
	for index := range channels {
		points[index] = data[10]
	}
	grid, err := jxrReadICCGrid(data[inputEnd:outputStart], points[:channels], 3, precision)
	if err != nil {
		return nil, err
	}
	if len(grid.data) != outputStart-inputEnd {
		return nil, fmt.Errorf("%w: ICC lookup grid extent", ErrInvalid)
	}
	return &jxrICCLUT{
		grid: grid, input: data[header:inputEnd], output: data[outputStart:],
		inputEntries: inputs, outputEntries: outputs, lab: lab, legacyLab: lab && precision == 2,
	}, nil
}

// jxrReadICCGrid 从最后一轴计算跨度，每次乘法前检查表长
// 入参: data 可用网格字节, points 各轴的网格点数, outputs 输出通道数, precision 每分量的字节数
// 返回: jxrICCGrid 已验证网格, error 格式错误
func jxrReadICCGrid(data []byte, points []byte, outputs, precision int) (jxrICCGrid, error) {
	grid := jxrICCGrid{precision: precision}
	copy(grid.points[:], points)
	stride := outputs * precision
	for index := len(points) - 1; index >= 0; index-- {
		count := int(points[index])
		if count < 2 || stride > len(data)/count {
			return jxrICCGrid{}, fmt.Errorf("%w: ICC lookup grid dimensions", ErrInvalid)
		}
		grid.strides[index] = stride
		stride *= count
	}
	grid.data = data[:stride]
	return grid, nil
}

// value 对多维网格执行分轴线性与单纯形插值并写入输出工作区
// 入参: input 已裁剪的单位设备分量, result 输出工作区
func (grid *jxrICCGrid) value(input, result []float64) {
	var axes [jxrICCMaxGridAxes]jxrICCGridAxis
	var offsets [16]int
	var weights [16]float64
	weights[0] = 1
	outer := min(4, max(0, len(input)-3))
	at, count, corners := 0, 0, 1
	for index, value := range input {
		if math.IsNaN(value) {
			value = 0
		}
		value = colorClamp(value)
		last := int(grid.points[index]) - 1
		position := value * float64(last)
		base := int(position)
		stride := grid.strides[index]
		at += base * stride
		fraction := position - float64(base)
		if fraction == 0 {
			continue
		}
		if index < outer {
			for corner := range corners {
				offsets[corners+corner] = offsets[corner] + stride
				weights[corners+corner] = weights[corner] * fraction
				weights[corner] *= 1 - fraction
			}
			corners *= 2
		} else {
			axes[count] = jxrICCGridAxis{fraction: fraction, stride: stride}
			count++
		}
	}
	slices.SortFunc(axes[:count], func(a, b jxrICCGridAxis) int {
		return cmp.Compare(b.fraction, a.fraction)
	})
	clear(result)
	for corner := range corners {
		vertex := at + offsets[corner]
		weight := 1.0
		for _, axis := range axes[:count] {
			grid.addVertex(result, vertex, weights[corner]*(weight-axis.fraction))
			vertex += axis.stride
			weight = axis.fraction
		}
		grid.addVertex(result, vertex, weights[corner]*weight)
	}
}

// addVertex 将一个网格顶点的各通道值累加到插值结果
// 入参: result 插值累加值, at 已校验顶点字节位置, weight 顶点权重
func (grid *jxrICCGrid) addVertex(result []float64, at int, weight float64) {
	for index := range result {
		result[index] += weight * jxrICCGridSample(grid.data, at+index*grid.precision, grid.precision)
	}
}

// jxrICCGridSample 读取整数或浮点网格值
// 入参: data 已校验表值, at 字节位置, precision 样本字节数
// 返回: float64 网格值
func jxrICCGridSample(data []byte, at, precision int) float64 {
	if precision == 4 {
		return jxrICCFloat32(data[at:])
	}
	return jxrICCNormalized(data, at, precision)
}

// jxrICCNormalized 按标签精度读取一个归一化整数样本
// 入参: data 已校验表值, at 样本字节位置, precision 每分量的字节数
// 返回: float64 归一化值
func jxrICCNormalized(data []byte, at, precision int) float64 {
	if precision == 1 {
		return float64(data[at]) / 255
	}
	return float64(binary.BigEndian.Uint16(data[at:])) / 65535
}

// jxrICCTableValue 按相邻均匀样本对传统单通道表进行线性插值
// 入参: table 依次排列的单通道表, channel 表索引, entries 每表样本数, precision 每样本字节数, value 单位输入值
// 返回: float64 归一化结果
func jxrICCTableValue(table []byte, channel, entries, precision int, value float64) float64 {
	if math.IsNaN(value) {
		value = 0
	}
	last := entries - 1
	position := colorClamp(value) * float64(last)
	low := int(position)
	base := channel * entries
	a := jxrICCNormalized(table, (base+low)*precision, precision)
	b := jxrICCNormalized(table, (base+min(low+1, last))*precision, precision)
	return a + (b-a)*(position-float64(low))
}

// xyz 解释查表输出的PCS编码并将其转换为D50 XYZ
// 入参: values 已解除预乘的源通道
// 返回: [3]float64 未量化的D50 XYZ
func (lut *jxrICCLUT) xyz(values []float64) [3]float64 {
	var pcs [3]float64
	if lut.stages != nil {
		pcs = lut.stages.value(&lut.grid, values)
	} else {
		var inputs [jxrICCMaxChannels]float64
		precision := lut.grid.precision
		for index, value := range values {
			inputs[index] = jxrICCTableValue(lut.input, index, lut.inputEntries, precision, value)
		}
		lut.grid.value(inputs[:len(values)], pcs[:])
		for index := range pcs {
			pcs[index] = jxrICCTableValue(lut.output, index, lut.outputEntries, precision, pcs[index])
		}
	}
	if !lut.lab {
		for index := range pcs {
			pcs[index] *= 65535.0 / 32768.0
		}
		return pcs
	}
	if lut.legacyLab {
		for index := range pcs {
			pcs[index] = min(1, pcs[index]*(65535.0/65280.0))
		}
	}
	return jxrICCLabXYZ(100*pcs[0], 255*pcs[1]-128, 255*pcs[2]-128)
}

// jxrICCLabXYZ 将实际CIE Lab分量转换为D50 XYZ
// 入参: l,a,b 实际Lab数值
// 返回: [3]float64 未量化的D50 XYZ
func jxrICCLabXYZ(l, a, b float64) [3]float64 {
	fy := (l + 16) / 116
	fx := fy + a/500
	fz := fy - b/200
	return [3]float64{0.9642 * jxrICCLabInverse(fx), jxrICCLabInverse(fy), 0.8249 * jxrICCLabInverse(fz)}
}

// jxrICCLabInverse 反演CIE Lab分段非线性
// 入参: value Lab的中间分量
// 返回: float64 相对白点的线性分量
func jxrICCLabInverse(value float64) float64 {
	const delta = 6.0 / 29.0
	if value > delta {
		return value * value * value
	}
	return 3 * delta * delta * (value - 4.0/29.0)
}

// jxrReadICCFloatTransform 读取ICC多处理元素媒体相对色度管线
// 入参: ctx Context, data 完整mpet标签, channels 设备输入数, lab 是否为Lab PCS, maximum 工作额度
// 返回: *jxrICCFloatTransform 已验证模型, error 结构、能力、资源或取消错误
func jxrReadICCFloatTransform(ctx context.Context, data []byte, channels int, lab bool, maximum uint64) (*jxrICCFloatTransform, error) {
	if len(data) < 16 || string(data[:4]) != "mpet" || binary.BigEndian.Uint32(data[4:]) != 0 ||
		int(binary.BigEndian.Uint16(data[8:])) != channels || binary.BigEndian.Uint16(data[10:]) != 3 {
		return nil, fmt.Errorf("%w: ICC floating-point transform header", ErrInvalid)
	}
	p := jxrParser{ctx: ctx, limit: maximum}
	if err := p.reserve(jxrICCFloatWork); err != nil {
		return nil, err
	}
	count := uint64(binary.BigEndian.Uint32(data[12:]))
	positions, err := jxrReadICCPositions(&p, data, 16, count)
	if err != nil {
		return nil, err
	}
	width := channels
	for index, position := range positions {
		if index&1023 == 0 {
			if err := p.check(); err != nil {
				return nil, err
			}
		}
		element := data[position.start:position.end]
		switch string(element[:4]) {
		case "cvst", "matf", "clut", "bACS", "eACS":
		default:
			return nil, fmt.Errorf("%w: ICC floating-point element type", ErrUnsupported)
		}
		if binary.BigEndian.Uint32(element[4:]) != 0 || int(binary.BigEndian.Uint16(element[8:])) != width || binary.BigEndian.Uint16(element[10:]) == 0 {
			return nil, fmt.Errorf("%w: ICC floating-point element channels", ErrInvalid)
		}
		width = int(binary.BigEndian.Uint16(element[10:]))
	}
	if width != 3 {
		return nil, fmt.Errorf("%w: ICC floating-point output channels", ErrInvalid)
	}
	if err := p.reserve(count * 96); err != nil {
		return nil, err
	}
	stages := make([]jxrICCFloatStage, len(positions))
	width = channels
	for index, position := range positions {
		if index&1023 == 0 {
			if err := p.check(); err != nil {
				return nil, err
			}
		}
		if position.shared < index {
			stages[index] = stages[position.shared]
		} else {
			stages[index], err = jxrReadICCFloatStage(&p, data[position.start:position.end])
			if err != nil {
				return nil, err
			}
		}
		width = max(width, stages[index].outputs)
	}
	p.used -= count * 32
	if err := p.reserve(uint64(width) * 16); err != nil {
		return nil, err
	}
	return &jxrICCFloatTransform{stages: stages, scratch: make([]float64, width*2), lab: lab}, nil
}

// jxrReadICCPositions 读取位置表，按物理位置验证重叠后恢复逻辑顺序
// 入参: p 工作预算, data 含位置表及数据的元素, begin 位置表起始字节, count 元素数
// 返回: []jxrICCPosition 已验证范围, error 格式、资源或取消错误
func jxrReadICCPositions(p *jxrParser, data []byte, begin int, count uint64) ([]jxrICCPosition, error) {
	if count == 0 || count > uint64(len(data)-begin)/8 {
		return nil, fmt.Errorf("%w: ICC element position count", ErrInvalid)
	}
	if count > uint64(^uint(0)>>1)/32 {
		return nil, fmt.Errorf("%w: ICC element position storage", ErrLimit)
	}
	if err := p.reserve(count * 32); err != nil {
		return nil, err
	}
	positions := make([]jxrICCPosition, int(count))
	first := uint64(begin) + 8*count
	for index := range positions {
		if index&1023 == 0 {
			if err := p.check(); err != nil {
				return nil, err
			}
		}
		record := data[begin+8*index:]
		start := uint64(binary.BigEndian.Uint32(record))
		size := uint64(binary.BigEndian.Uint32(record[4:]))
		if start < first || start%4 != 0 || start > uint64(len(data)) || size < 12 || size > uint64(len(data))-start {
			return nil, fmt.Errorf("%w: ICC element position", ErrInvalid)
		}
		end := int(start + size)
		if !jxrICCZero(data[end : end+min(int(-size&3), len(data)-end)]) {
			return nil, fmt.Errorf("%w: ICC element padding", ErrInvalid)
		}
		positions[index] = jxrICCPosition{start: int(start), end: end, index: index, shared: index}
	}
	slices.SortFunc(positions, func(a, b jxrICCPosition) int {
		if a.start == b.start {
			return cmp.Compare(a.index, b.index)
		}
		return cmp.Compare(a.start, b.start)
	})
	if uint64(positions[0].start) != first {
		return nil, fmt.Errorf("%w: ICC first element position", ErrInvalid)
	}
	for index := 1; index < len(positions); index++ {
		if index&1023 == 0 {
			if err := p.check(); err != nil {
				return nil, err
			}
		}
		previous, current := positions[index-1], &positions[index]
		if current.start == previous.start && current.end == previous.end {
			current.shared = previous.shared
		} else if current.start < previous.end {
			return nil, fmt.Errorf("%w: overlapping ICC elements", ErrInvalid)
		}
	}
	slices.SortFunc(positions, func(a, b jxrICCPosition) int { return cmp.Compare(a.index, b.index) })
	return positions, p.check()
}

// jxrReadICCFloatStage 校验一个已知浮点元素的字节范围及系数
// 入参: p 工作预算, data 单个已定位元素
// 返回: jxrICCFloatStage 已验证元素, error 格式、能力、资源或取消错误
func jxrReadICCFloatStage(p *jxrParser, data []byte) (jxrICCFloatStage, error) {
	inputs := int(binary.BigEndian.Uint16(data[8:]))
	stage := jxrICCFloatStage{kind: string(data[:4]), outputs: int(binary.BigEndian.Uint16(data[10:]))}
	switch stage.kind {
	case "cvst":
		if inputs != stage.outputs {
			return stage, fmt.Errorf("%w: ICC curve-set channels", ErrInvalid)
		}
		curves, err := jxrReadICCSegmentedCurves(p, data, inputs)
		stage.curves = curves
		return stage, err
	case "matf":
		if uint64(len(data)-12) != 4*uint64(stage.outputs)*uint64(inputs+1) {
			return stage, fmt.Errorf("%w: ICC floating-point matrix extent", ErrInvalid)
		}
		stage.matrix = data[12:]
		return stage, jxrCheckICCFloat32(p, stage.matrix)
	case "clut":
		if inputs > jxrICCMaxGridAxes || len(data) < 28 || !jxrICCZero(data[12+inputs:28]) {
			return stage, fmt.Errorf("%w: ICC floating-point grid header", ErrInvalid)
		}
		grid, err := jxrReadICCGrid(data[28:], data[12:12+inputs], stage.outputs, 4)
		if err != nil {
			return stage, err
		}
		if len(grid.data) != len(data)-28 {
			return stage, fmt.Errorf("%w: ICC floating-point grid extent", ErrInvalid)
		}
		if err := jxrCheckICCFloat32(p, grid.data); err != nil {
			return stage, err
		}
		if err := p.reserve(256); err != nil {
			return stage, err
		}
		stage.grid = &grid
	case "bACS", "eACS":
		if inputs != stage.outputs || len(data) != 16 {
			return stage, fmt.Errorf("%w: ICC placeholder element", ErrInvalid)
		}
	}
	return stage, nil
}

// jxrCheckICCFloat32 校验ICC大端单精度浮点字节并拒绝非规格化数、无穷大和NaN
// 入参: p 取消状态, data 对齐的IEEE二进制32位数值
// 返回: error 数值编码或取消错误
func jxrCheckICCFloat32(p *jxrParser, data []byte) error {
	for at := 0; at < len(data); at += 4 {
		if at&16383 == 0 {
			if err := p.check(); err != nil {
				return err
			}
		}
		bits := binary.BigEndian.Uint32(data[at:]) & 0x7fffffff
		if bits != 0 && bits < 0x00800000 || bits >= 0x7f800000 {
			return fmt.Errorf("%w: ICC float32Number", ErrInvalid)
		}
	}
	return nil
}

// jxrICCFloat32 解释一个已验证的大端ICC浮点数
// 入参: data 始于浮点数的可用字节
// 返回: float64 浮点数值
func jxrICCFloat32(data []byte) float64 {
	return float64(math.Float32frombits(binary.BigEndian.Uint32(data)))
}

// xyz 执行浮点元素并按实际PCS数值转换到D50 XYZ
// 入参: ctx Context, input 解除预乘后的设备通道
// 返回: [3]float64 D50 XYZ, error 数值或取消错误
func (transform *jxrICCFloatTransform) xyz(ctx context.Context, input []float64) ([3]float64, error) {
	var xyz [3]float64
	for _, value := range input {
		if !(math.Abs(value) <= math.MaxFloat32) {
			return xyz, fmt.Errorf("%w: ICC floating-point input", ErrInvalid)
		}
	}
	width := len(transform.scratch) / 2
	output, spare := transform.scratch[:width], transform.scratch[width:]
	for _, stage := range transform.stages {
		if err := ctx.Err(); err != nil {
			return xyz, err
		}
		result := output[:stage.outputs]
		switch {
		case stage.curves != nil:
			for index, curve := range stage.curves {
				if index&255 == 0 {
					if err := ctx.Err(); err != nil {
						return xyz, err
					}
				}
				result[index] = curve.value(input[index])
			}
		case stage.matrix != nil:
			matrix, offsets := stage.matrix, stage.matrix[len(input)*stage.outputs*4:]
			for row := range result {
				if row&63 == 0 {
					if err := ctx.Err(); err != nil {
						return xyz, err
					}
				}
				value := jxrICCFloat32(offsets[4*row:])
				for _, component := range input {
					value += jxrICCFloat32(matrix) * component
					matrix = matrix[4:]
				}
				result[row] = value
			}
		case stage.grid != nil:
			var normalized [jxrICCMaxGridAxes]float64
			for index, value := range input {
				normalized[index] = colorClamp(value)
			}
			stage.grid.value(normalized[:len(input)], result)
		default:
			continue
		}
		for _, value := range result {
			if !(math.Abs(value) <= math.MaxFloat32) {
				return xyz, fmt.Errorf("%w: ICC floating-point result", ErrInvalid)
			}
		}
		input, output, spare = result, spare, output
	}
	copy(xyz[:], input)
	if transform.lab {
		xyz = jxrICCLabXYZ(xyz[0], xyz[1], xyz[2])
	}
	return xyz, nil
}

// jxrReadICCSegmentedCurves 按曲线集的位置表读取各通道，复用共享曲线
// 入参: p 工作预算, data 完整cvst元素, count 曲线数
// 返回: []jxrICCSegmentedCurve 已验证曲线, error 格式、能力、资源或取消错误
func jxrReadICCSegmentedCurves(p *jxrParser, data []byte, count int) ([]jxrICCSegmentedCurve, error) {
	positions, err := jxrReadICCPositions(p, data, 12, uint64(count))
	if err != nil {
		return nil, err
	}
	defer func() { p.used -= uint64(count) * 32 }()
	if err := p.reserve(uint64(count) * 32); err != nil {
		return nil, err
	}
	curves := make([]jxrICCSegmentedCurve, count)
	for index, position := range positions {
		if index&1023 == 0 {
			if err := p.check(); err != nil {
				return nil, err
			}
		}
		if position.shared < index {
			curves[index] = curves[position.shared]
			continue
		}
		curves[index], err = jxrReadICCSegmentedCurve(p, data[position.start:position.end])
		if err != nil {
			return nil, err
		}
	}
	return curves, nil
}

// jxrReadICCSegmentedCurve 读取ICC浮点分段曲线的分段定义及隐含采样端点
// 入参: p 工作预算, data 完整curf数据
// 返回: jxrICCSegmentedCurve 已验证曲线, error 格式、能力、资源或取消错误
func jxrReadICCSegmentedCurve(p *jxrParser, data []byte) (jxrICCSegmentedCurve, error) {
	var curve jxrICCSegmentedCurve
	if len(data) < 12 || string(data[:4]) != "curf" || binary.BigEndian.Uint32(data[4:]) != 0 || binary.BigEndian.Uint16(data[10:]) != 0 {
		return curve, fmt.Errorf("%w: ICC segmented curve header", ErrInvalid)
	}
	count := int(binary.BigEndian.Uint16(data[8:]))
	if count == 0 || count > (len(data)-8)/16 {
		return curve, fmt.Errorf("%w: ICC curve segment count", ErrInvalid)
	}
	header := 8 + 4*count
	if err := jxrCheckICCFloat32(p, data[12:header]); err != nil {
		return curve, err
	}
	if err := p.reserve(uint64(count) * 64); err != nil {
		return curve, err
	}
	segments := make([]jxrICCFloatSegment, count)
	lower, at := math.Inf(-1), header
	for index := range segments {
		upper := math.Inf(1)
		if index+1 < count {
			upper = jxrICCFloat32(data[12+4*index:])
		}
		if upper < lower {
			return curve, fmt.Errorf("%w: ICC curve breakpoint order", ErrInvalid)
		}
		segment, size, err := jxrReadICCFloatSegment(p, data[at:], lower, upper)
		if err != nil {
			return curve, err
		}
		if segment.function == jxrICCSampledSegment {
			if index == 0 || index+1 == count {
				return curve, fmt.Errorf("%w: ICC curve requires formula at unbounded endpoint", ErrInvalid)
			}
			segment.left = segments[index-1].value(lower)
			if !(math.Abs(segment.left) <= math.MaxFloat32) {
				return curve, fmt.Errorf("%w: ICC sampled curve boundary value", ErrInvalid)
			}
		}
		segments[index] = segment
		lower, at = upper, at+size
	}
	if at != len(data) {
		return curve, fmt.Errorf("%w: ICC segmented curve extent", ErrInvalid)
	}
	curve.segments = segments
	return curve, nil
}

// jxrReadICCFloatSegment 读取一个公式或采样分段及其实际字节数
// 入参: p 取消状态, data 当前分段起始的字节, lower,upper 分段边界
// 返回: jxrICCFloatSegment 分段, int 字节数, error 格式、能力或取消错误
func jxrReadICCFloatSegment(p *jxrParser, data []byte, lower, upper float64) (jxrICCFloatSegment, int, error) {
	segment := jxrICCFloatSegment{lower: lower, upper: upper}
	if len(data) < 12 || binary.BigEndian.Uint32(data[4:]) != 0 {
		return segment, 0, fmt.Errorf("%w: ICC curve segment header", ErrInvalid)
	}
	var count uint64
	switch string(data[:4]) {
	case "parf":
		segment.function = binary.BigEndian.Uint16(data[8:])
		if binary.BigEndian.Uint16(data[10:]) != 0 {
			return segment, 0, fmt.Errorf("%w: ICC formula segment reserved field", ErrInvalid)
		}
		switch segment.function {
		case 0:
			count = 4
		case 1, 2:
			count = 5
		default:
			return segment, 0, fmt.Errorf("%w: ICC formula segment function", ErrUnsupported)
		}
	case "samf":
		segment.function = jxrICCSampledSegment
		count = uint64(binary.BigEndian.Uint32(data[8:]))
	default:
		return segment, 0, fmt.Errorf("%w: ICC curve segment type", ErrUnsupported)
	}
	if count == 0 || count > uint64(len(data)-12)/4 {
		return segment, 0, fmt.Errorf("%w: ICC curve segment extent", ErrInvalid)
	}
	size := 12 + 4*int(count)
	segment.data = data[12:size]
	return segment, size, jxrCheckICCFloat32(p, segment.data)
}

// value 查找包含输入的首个曲线分段，分段上界归前段
// 入参: input 有限输入通道
// 返回: float64 未裁剪曲线结果
func (curve jxrICCSegmentedCurve) value(input float64) float64 {
	index := sort.Search(len(curve.segments), func(index int) bool {
		return input <= curve.segments[index].upper
	})
	return curve.segments[index].value(input)
}

// value 计算公式或在含隐含左端点的均匀样本之间插值
// 入参: input 分段内的有限输入
// 返回: float64 未裁剪分段结果
func (segment jxrICCFloatSegment) value(input float64) float64 {
	data := segment.data
	if segment.function == jxrICCSampledSegment {
		count := len(data) / 4
		if input >= segment.upper {
			return jxrICCFloat32(data[4*(count-1):])
		}
		position := (input - segment.lower) / (segment.upper - segment.lower) * float64(count)
		low := int(position)
		if low >= count {
			return jxrICCFloat32(data[4*(count-1):])
		}
		left := segment.left
		if low > 0 {
			left = jxrICCFloat32(data[4*(low-1):])
		}
		right := jxrICCFloat32(data[4*low:])
		return left + (right-left)*(position-float64(low))
	}
	a, b, c, d := jxrICCFloat32(data), jxrICCFloat32(data[4:]), jxrICCFloat32(data[8:]), jxrICCFloat32(data[12:])
	switch segment.function {
	case 0:
		return math.Pow(b*input+c, a) + d
	case 1:
		return b*math.Log10(c*math.Pow(input, a)+d) + jxrICCFloat32(data[16:])
	default:
		return a*math.Pow(b, c*input+d) + jxrICCFloat32(data[16:])
	}
}

// jxrReadICCStages 读取mAB的四种合法元素组合，PCS输出固定为三个通道
// 入参: data 完整mAB标签, channels 源设备分量数, lab 是否为Lab PCS
// 返回: *jxrICCLUT 已校验模型, error 标签格式或曲线能力错误
func jxrReadICCStages(data []byte, channels int, lab bool) (*jxrICCLUT, error) {
	if len(data) < 32 || binary.BigEndian.Uint16(data[10:]) != 0 {
		return nil, fmt.Errorf("%w: ICC multi-stage lookup header", ErrInvalid)
	}
	var offsets [5]int
	for index := range offsets {
		offset := binary.BigEndian.Uint32(data[12+4*index:])
		if offset != 0 && (offset < 32 || offset%4 != 0 || uint64(offset) >= uint64(len(data))) {
			return nil, fmt.Errorf("%w: ICC lookup element offset", ErrInvalid)
		}
		offsets[index] = int(offset)
	}
	b, matrix, m, clut, a := offsets[0], offsets[1], offsets[2], offsets[3], offsets[4]
	if b == 0 || (m == 0) != (matrix == 0) || (clut == 0) != (a == 0) || clut == 0 && channels != 3 {
		return nil, fmt.Errorf("%w: ICC lookup element combination", ErrInvalid)
	}
	stages := &jxrICCStages{affine: matrix != 0}
	layout := jxrICCStageLayout{data: data}
	if err := layout.readCurves(b, stages.b[:]); err != nil {
		return nil, err
	}
	if matrix != 0 {
		if _, err := layout.add(matrix, 48, false); err != nil {
			return nil, err
		}
		for index := range stages.matrix {
			stages.matrix[index] = jxrICCFixed(data[matrix+4*index:])
		}
		if err := layout.readCurves(m, stages.m[:]); err != nil {
			return nil, err
		}
	}
	lut := &jxrICCLUT{stages: stages, lab: lab}
	if clut != 0 {
		table := data[clut:]
		if len(table) < 20 || table[16] < 1 || table[16] > 2 || !jxrICCZero(table[channels:16]) || !jxrICCZero(table[17:20]) {
			return nil, fmt.Errorf("%w: ICC multi-stage grid header", ErrInvalid)
		}
		grid, err := jxrReadICCGrid(table[20:], table[:channels], 3, int(table[16]))
		if err != nil {
			return nil, err
		}
		if _, err := layout.add(clut, 20+len(grid.data), false); err != nil {
			return nil, err
		}
		if err := layout.readCurves(a, stages.a[:channels]); err != nil {
			return nil, err
		}
		lut.grid = grid
	}
	return lut, nil
}

// add 校验元素边界和零填充，并拒绝除完整曲线共享以外的重叠
// 入参: start 元素偏移, size 实际字节数, curve 元素是否为曲线
// 返回: int 对齐后的下一元素位置, error 标签范围错误
func (layout *jxrICCStageLayout) add(start, size int, curve bool) (int, error) {
	if size > len(layout.data)-start {
		return 0, fmt.Errorf("%w: ICC lookup element extent", ErrInvalid)
	}
	end := start + size
	padding := min(-size&3, len(layout.data)-end)
	if !jxrICCZero(layout.data[end : end+padding]) {
		return 0, fmt.Errorf("%w: ICC lookup element padding", ErrInvalid)
	}
	end += padding
	for _, previous := range layout.extents[:layout.count] {
		if start == previous.start && end == previous.end && curve && previous.curve {
			return end, nil
		}
		if start < previous.end && previous.start < end {
			return 0, fmt.Errorf("%w: overlapping ICC lookup elements", ErrInvalid)
		}
	}
	layout.extents[layout.count] = jxrICCStageExtent{start, end, curve}
	layout.count++
	return end, nil
}

// readCurves 读取一组相邻且对齐的嵌入ICC曲线
// 入参: start 第一条曲线偏移, curves 已限定数量的目标曲线
// 返回: error 标签格式或曲线能力错误
func (layout *jxrICCStageLayout) readCurves(start int, curves []jxrICCCurve) error {
	for index := range curves {
		data := layout.data[start:]
		size, err := jxrICCCurveSize(data)
		if err != nil {
			return err
		}
		start, err = layout.add(start, size, true)
		if err != nil {
			return err
		}
		curves[index], err = jxrReadICCCurve(data[:size])
		if err != nil {
			return err
		}
	}
	return nil
}

// jxrICCZero 判断保留字段或对齐填充是否全部为零
// 入参: data 已限定字节
// 返回: bool 是否为零
func jxrICCZero(data []byte) bool {
	for _, value := range data {
		if value != 0 {
			return false
		}
	}
	return true
}

// value 按A曲线、网格、M曲线、仿射矩阵、B曲线的顺序计算
// 入参: grid 可选颜色网格, values 已解除预乘的源设备分量
// 返回: [3]float64 归一化PCS
func (stages *jxrICCStages) value(grid *jxrICCGrid, values []float64) [3]float64 {
	var pcs [3]float64
	if len(grid.data) == 0 {
		copy(pcs[:], values)
	} else {
		var inputs [jxrICCMaxChannels]float64
		for index, value := range values {
			inputs[index] = stages.a[index].value(value)
		}
		grid.value(inputs[:len(values)], pcs[:])
	}
	if stages.affine {
		for index := range pcs {
			pcs[index] = stages.m[index].value(pcs[index])
		}
		x, y, z := pcs[0], pcs[1], pcs[2]
		for index := range pcs {
			row := stages.matrix[3*index:]
			pcs[index] = row[0]*x + row[1]*y + row[2]*z + stages.matrix[9+index]
		}
	}
	for index := range pcs {
		pcs[index] = stages.b[index].value(pcs[index])
	}
	return pcs
}
