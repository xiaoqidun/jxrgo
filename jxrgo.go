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

// Package jxrgo 高性能、零依赖、纯 Go 语言 JXR 图像编解码库
package jxrgo

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
)

const (
	// FormatUnknown 表示未知封装格式
	FormatUnknown Format = iota
	// FormatJXR 表示JPEG XR容器文件格式
	FormatJXR
	// FormatCodestream 表示独立JPEG XR图像码流格式
	FormatCodestream
)

// rasterMaxWork 与 rasterMaxComponents 保存默认最大工作内存上限与最大颜色分量数
const (
	rasterMaxWork       = 1 << 27
	rasterMaxComponents = (1 << 12) + 15
)

const (
	// ColorUnknown 表示未知颜色空间
	ColorUnknown ColorSpace = iota
	// ColorGray 表示灰度单通道颜色空间
	ColorGray
	// ColorRGB 表示红绿蓝三通道颜色空间
	ColorRGB
	// ColorCMYK 表示经变换编码的CMYK颜色空间
	ColorCMYK
	// ColorCMYKDirect 表示直接存储的四通道CMYK颜色空间
	ColorCMYKDirect
	// ColorYCC 表示亮度色度YCC颜色空间
	ColorYCC
	// ColorMultiChannel 表示通用多通道颜色空间
	ColorMultiChannel
)

const (
	// SampleUnknown 表示未知样本数值编码
	SampleUnknown SampleFormat = iota
	// SampleUnsigned 表示无符号整数样本编码
	SampleUnsigned
	// SampleFixed 表示有符号定点样本编码
	SampleFixed
	// SampleFloat 表示IEEE半精度或单精度浮点样本编码
	SampleFloat
	// SampleRGBE 表示共享指数RGBE高动态范围样本编码
	SampleRGBE
)

const (
	// ColorHintNone 表示文件未声明COLOR_SPACE标签
	ColorHintNone ColorHint = 0
	// ColorHintSRGB 表示文件声明sRGB颜色空间标签
	ColorHintSRGB ColorHint = 1
	// ColorHintUnspecified 表示文件显式声明未指定颜色空间标签
	ColorHintUnspecified ColorHint = 0xffff
)

const (
	// BandAll 表示包含全部频带及Flexbits精细位
	BandAll Band = iota
	// BandNoFlexbits 表示省略Flexbits精细位频带
	BandNoFlexbits
	// BandNoHighpass 表示省略高通与Flexbits频带
	BandNoHighpass
	// BandDC 表示仅保留直流频带
	BandDC
)

const (
	// LayoutSpatial 表示按空间宏块顺序分包布局
	LayoutSpatial Layout = iota
	// LayoutFrequency 表示按频带渐进顺序分包布局
	LayoutFrequency
)

const (
	// OverlapNone 表示禁用重叠变换滤波
	OverlapNone Overlap = iota
	// OverlapOne 表示启用一级重叠变换滤波
	OverlapOne
	// OverlapTwo 表示启用两级重叠变换滤波
	OverlapTwo
)

const (
	// OrientationNone 表示无旋转无翻转方向
	OrientationNone Orientation = iota
	// OrientationFlipY 表示垂直翻转方向
	OrientationFlipY
	// OrientationFlipX 表示水平翻转方向
	OrientationFlipX
	// OrientationFlipXY 表示旋转180度方向
	OrientationFlipXY
	// OrientationRotate90 表示顺时针转置方向
	OrientationRotate90
	// OrientationRotate90FlipY 表示转置后垂直翻转方向
	OrientationRotate90FlipY
	// OrientationRotate90FlipX 表示转置后水平翻转方向
	OrientationRotate90FlipX
	// OrientationRotate90FlipXY 表示转置后双向翻转方向
	OrientationRotate90FlipXY
)

const (
	// ImageTypeDefault 表示主图像类型
	ImageTypeDefault ImageType = 0
	// ImageTypePreview 表示预览图像标志
	ImageTypePreview ImageType = 1
	// ImageTypePage 表示页序列图像标志
	ImageTypePage ImageType = 2
)

const (
	// IntentRelative 表示ICC媒体相对色度渲染意图
	IntentRelative RenderingIntent = iota
	// IntentPerceptual 表示ICC感知渲染意图
	IntentPerceptual
	// IntentSaturation 表示ICC饱和度渲染意图
	IntentSaturation
	// IntentAbsolute 表示ICC绝对色度渲染意图
	IntentAbsolute
)

const (
	// SubsamplingNone 表示不额外执行内部色度下采样
	SubsamplingNone Subsampling = iota
	// Subsampling422 表示编码端执行内部YUV422色度下采样
	Subsampling422
	// Subsampling420 表示编码端执行内部YUV420色度下采样
	Subsampling420
)

// jxrDimensionLimit 保存长图像头中减一编码可表示的最大可见轴长
const jxrDimensionLimit uint64 = 1 << 32

// jxrPixelPrefix 保存JPEG XR标准像素格式GUID的前十五字节固定前缀
const jxrPixelPrefix = "\x24\xc3\xdd\x6f\x03\x4e\xfe\x4b\xb1\x85\x3d\x77\x76\x8d\xc9"

// jxrUnsigned 等常量标识像素格式的内部样本数值编码类别
const (
	jxrUnsigned = iota
	jxrFixed16
	jxrHalf
	jxrFixed32
	jxrFloat
	jxrRGBE
)

// ErrInvalid 表示文件结构、码流或参数不符合要求
var ErrInvalid = errors.New("invalid image data")

// ErrUnsupported 表示当前操作不支持的格式或处理能力
var ErrUnsupported = errors.New("unsupported image feature")

// ErrLimit 表示输入、输出、像素或工作内存超过资源或表示容量上限
var ErrLimit = errors.New("image resource limit exceeded")

// ErrClosed 表示解码器已经关闭
var ErrClosed = errors.New("decoder is closed")

// jxrSRGB8Threshold 保存二十位定点线性光到八位sRGB码值的单调递增临界阈值表
var jxrSRGB8Threshold = func() [256]uint32 {
	var table [256]uint32
	for k := 1; k < 256; k++ {
		s := (float64(k) - 0.5) / 255.0
		var lin float64
		if s <= 0.04045 {
			lin = s / 12.92
		} else {
			lin = math.Pow((s+0.055)/1.055, 2.4)
		}
		t := uint32(math.Ceil(lin * (1 << 20)))
		for t > 0 && displayLinear(float64(t-1)/(1<<20))*255+0.5 >= float64(k) {
			t--
		}
		for t < 1<<20 && displayLinear(float64(t)/(1<<20))*255+0.5 < float64(k) {
			t++
		}
		table[k] = t
	}
	return table
}()

// Limits 限制输入字节、像素数量及工作内存，零值采用默认上限
// MaxInputBytes限制压缩输入字节数，MaxPixels限制可见像素总数，MaxMemoryBytes限制受宿主地址容量约束的工作内存字节数
type Limits struct {
	MaxInputBytes  uint64
	MaxPixels      uint64
	MaxMemoryBytes uint64
}

// DecodeOptions 指定解码并发上限、频带截断、方向策略、资源上限及兼容选项
// Workers限制并发Goroutine上限，零值按CPU核数和分块规模自动选择；KeepOrientation为真时保留原始编码方向不旋转翻转
// Band指定全分辨率及降采样解码的频带上限，零值BandAll保留全部频带，其余值跳过或置零更高频带以支持渐进预览
// 默认按反量化前低通系数确定高通预测方向，LegacyHighpassPrediction启用按反量化后低通系数预测的旧版模式
type DecodeOptions struct {
	Workers                  int
	Limits                   Limits
	Band                     Band
	LegacyHighpassPrediction bool
	KeepOrientation          bool
}

// ProfileLevel 表示JPEG XR码流与容器声明的档次及级别对
type ProfileLevel struct {
	Profile uint8
	Level   uint8
}

// RenderingIntent 表示ICC色彩转换的渲染意图
type RenderingIntent uint8

// Subsampling 表示有损编码时可选的内部色度下采样模式
type Subsampling uint8

// WindowMargins 指定编码端图像头的四个裁剪边距（0至63），对应WINDOWING_FLAG窗口参数
// 启用时编码网格扩展为可见尺寸加上对应边距，边缘冗余区域以边界像素复制填充
type WindowMargins struct {
	Top    uint8
	Left   uint8
	Bottom uint8
	Right  uint8
}

// ImageDescription 保存ITU-T T.832 Annex A.5图像描述元数据标签
// 文本字段采用UTF-8字符串，DateTime非空时须符合YYYY:MM:DD HH:MM:SS格式（写入0x0132标签），PageNumber为可选的页号与总页数对（非nil时写入0x0129标签）
type ImageDescription struct {
	DocumentName        string
	ImageDescription    string
	EquipmentMake       string
	EquipmentModel      string
	PageName            string
	PageNumber          *[2]uint16
	SoftwareNameVersion string
	DateTime            string
	ArtistName          string
	HostComputer        string
	CopyrightNotice     string
}

// EncodeOptions 指定编码格式、布局、精度、并发、元数据与资源上限，零值采用无损JXR及全部频带
// TileSize以编码网格像素计且须为十六倍数，仅Lossy允许精度损失与Subsampling下采样，SeparateAlpha写入独立透明度码流
// Profiles指定档次与级别声明，ICCProfile、ColorHint、ColorEncoding、EXIF、GPS、XMP、IPTC与Description非空时优先于源Info写入容器；Uncompressed启用0xBC03无压缩容器模式
// Window指定编码窗口边距，ChromaCenteringX与ChromaCenteringY指定色度采样相位（0至4或7）；Workers零值自动，MaxOutputBytes零值采用64MiB
type EncodeOptions struct {
	Format                   Format
	Layout                   Layout
	Overlap                  Overlap
	Orientation              Orientation
	Subsampling              Subsampling
	TileSize                 image.Point
	HardTiles                bool
	Lossy                    bool
	LegacyHighpassPrediction bool
	Workers                  int
	Band                     Band
	Trim                     uint8
	Scaled                   bool
	Shift                    uint8
	Float                    *FloatOptions
	Quantization             *Quantization
	Alpha                    *PlaneOptions
	SeparateAlpha            bool
	MirrorSeparateAlpha      bool
	Profiles                 []ProfileLevel
	ICCProfile               []byte
	ColorHint                *ColorHint
	ColorEncoding            *ColorEncoding
	EXIF                     []byte
	GPS                      []byte
	XMP                      []byte
	IPTC                     []byte
	Description              *ImageDescription
	Uncompressed             bool
	Window                   *WindowMargins
	ChromaCenteringX         *uint8
	ChromaCenteringY         *uint8
	Scratch                  io.ReadWriteSeeker
	MaxOutputBytes           uint64
	Limits                   Limits
}

// PlaneOptions 指定透明度平面的频带、数值和量化参数，不改变样本自身类型
// Band控制保留频带，Scaled启用变换预缩放，Shift指定整数右移位数，Float指定单精度浮点参数，Quantization指定逐带量化
type PlaneOptions struct {
	Band         Band
	Scaled       bool
	Shift        uint8
	Float        *FloatOptions
	Quantization *Quantization
}

// FloatOptions 指定单精度浮点码流的尾数长度及指数偏置，不用于半精度或整数
// Mantissa为尾数位数（最大二十三），Bias为有符号指数偏置；未指定时选择有界指数偏置，无损二十三位、有损十六位
type FloatOptions struct {
	Mantissa uint8
	Bias     int8
}

// Quantization 指定各频带的逐分量量化参数，空切片采用零，单值共用，否则按内部颜色分量排列
// RGB内部采用亮度及两个色度分量，透明度仅有一个分量，非零量化索引会引入精度损失
// Tiles为空时整平面共用，否则须按最终编码网格顺序逐块指定，调用期间不得修改配置
type Quantization struct {
	DC       []uint8
	Lowpass  []uint8
	Highpass []uint8
	Tiles    []TileQuantization
}

// TileQuantization 指定单个编码分块的量化集合及核心宏块索引，空值继承平面默认参数
// Lowpass和Highpass各允许一至十六组，UseDC与UseLowpass分别复用直流或低通集合，不得重复指定被复用集合或HighpassIndex
// 索引为空时全取零，否则长度须等于编码方向填充后的十六乘十六核心宏块数并按行排列
type TileQuantization struct {
	DC            []uint8
	Lowpass       [][]uint8
	Highpass      [][]uint8
	UseDC         bool
	UseLowpass    bool
	LowpassIndex  []uint8
	HighpassIndex []uint8
}

// Band 表示编码写入的频带或解码截断上限，编码精确模式下删减频带仅允许被省略系数全零
type Band uint8

// Layout 表示空间分包或按频带分包
type Layout uint8

// Overlap 表示无重叠、一级重叠或两级重叠变换
type Overlap uint8

// Orientation 表示标准八种编码网格到显示网格的方向变换
type Orientation uint8

// ColorSpace 表示解码分量的颜色解释，不以通道数量代替颜色空间
type ColorSpace uint8

// ColorHint 表示文件COLOR_SPACE标签，零值表示未指定而非未知颜色通道
type ColorHint uint16

// ColorEncoding 保存PTM_COLOR_INFO的基色、传递特性、矩阵及范围声明，原始分量与ICC配置保持独立
// Primaries、Transfer、Matrix及FullRange采用CICP标准编码标识，保留值保持未解释状态
type ColorEncoding struct {
	Primaries uint8
	Transfer  uint8
	Matrix    uint8
	FullRange bool
}

// Format 区分JPEG XR文件容器与独立图像码流
type Format uint8

// ImageType 保存IMAGE_TYPE低两位标志，可组合预览图与页序列标记
// 预览图按目录顺序与非预览图一一对应，与样本颜色或数值格式无关
type ImageType uint8

// SampleFormat 表示样本位模式的数值编码，定点和浮点不按无符号整数解释
type SampleFormat uint8

// Resolution 保存每英寸水平及垂直像素数，X为水平DPI，Y为垂直DPI
type Resolution struct {
	X float64
	Y float64
}

// Info 保存图像范围、像素格式、原始方向及颜色与容器元数据
// Channels不含透明度，Premultiplied保持原样不解除预乘，Linear表示线性光扩展范围数值格式，Orientation为原始声明方向
// ICCProfile、ColorHint、ColorEncoding、EXIF、GPS、XMP、IPTC及Description保留容器元数据，独立码流不携带这些字段
type Info struct {
	Format        Format
	ImageType     ImageType
	Orientation   Orientation
	Bounds        image.Rectangle
	PixelFormat   string
	Channels      int
	ColorSpace    ColorSpace
	ColorHint     ColorHint
	ColorEncoding *ColorEncoding
	Alpha         bool
	Premultiplied bool
	Linear        bool
	Resolution    Resolution
	ICCProfile    []byte
	EXIF          []byte
	GPS           []byte
	XMP           []byte
	IPTC          []byte
	Description   *ImageDescription
	Components    []ComponentInfo
}

// ComponentInfo 保存分量网格及其在主图像中的采样位置
// XOffset、YOffset以主网格像素为单位，XPositionKnown、YPositionKnown分别表示位置是否明确
// Precision表示位模式长度，FractionBits为定点小数位数，Inverted表示二值黑白反向编码
type ComponentInfo struct {
	Bounds         image.Rectangle
	XStep          uint8
	YStep          uint8
	XOffset        float64
	YOffset        float64
	XPositionKnown bool
	YPositionKnown bool
	Format         SampleFormat
	Precision      uint8
	FractionBits   uint8
	Inverted       bool
}

// Raster 保存原生分量，无符号整数通过单位区间读取，定点及浮点保留扩展范围
// 通过Component读取未经ICC转换或解除预乘的分量，不把多通道图像猜测为RGB
// 不修改样本时可并发读取，写入时须由调用方协调并发访问
type Raster struct {
	native *imageComponents
	planes []imageComponents
	info   Info
}

// Component 提供共享原生存储的单个颜色或透明度通道视图
type Component struct {
	source *imageComponents
	index  int
	info   ComponentInfo
}

// jxrPixelFormat 保存标准像素格式与码流颜色、位深及透明度的对应关系
type jxrPixelFormat struct {
	name        string
	channels    int
	color       uint8
	depth       uint8
	number      uint8
	alpha       bool
	premultiply bool
}

// imageComponents 保存原生样本位模式，数值解释由分量信息确定
type imageComponents struct {
	bounds   image.Rectangle
	channels int
	alpha    bool
	values   []uint16
	words    []uint32
}

// DefaultLimits 返回默认输入、像素及工作内存上限
// 返回: Limits 资源上限
func DefaultLimits() Limits {
	return Limits{MaxInputBytes: 64 << 20, MaxPixels: 1 << 24, MaxMemoryBytes: rasterMaxWork}
}

// Info 返回图像信息副本，不共享容器元数据、描述标签、颜色声明及分量网格切片
// 返回: Info 图像信息
func (r *Raster) Info() Info {
	if r == nil {
		return Info{}
	}
	return cloneInfo(r.info)
}

// cloneBytes 按精确容量复制可选字节切片，空切片统一返回nil
// 入参: src 原始字节切片
// 返回: []byte 独立副本
func cloneBytes(src []byte) []byte {
	if len(src) == 0 {
		return nil
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}

// cloneInfo 独立复制可变元数据，不共享ICC配置、容器元数据、颜色声明、描述标签或分量描述
// 入参: info 原信息
// 返回: Info 独立元数据副本
func cloneInfo(info Info) Info {
	info.ICCProfile = cloneBytes(info.ICCProfile)
	info.EXIF = cloneBytes(info.EXIF)
	info.GPS = cloneBytes(info.GPS)
	info.XMP = cloneBytes(info.XMP)
	info.IPTC = cloneBytes(info.IPTC)
	info.Description = cloneImageDescription(info.Description)
	info.Components = cloneComponentInfo(info.Components)
	info.ColorEncoding = cloneColorEncoding(info.ColorEncoding)
	return info
}

// cloneImageDescription 深拷贝图像描述标签结构，含PageNumber独立副本
// 入参: desc 原图像描述，nil返回nil
// 返回: *ImageDescription 独立图像描述副本
func cloneImageDescription(desc *ImageDescription) *ImageDescription {
	if desc == nil {
		return nil
	}
	copied := *desc
	if desc.PageNumber != nil {
		page := *desc.PageNumber
		copied.PageNumber = &page
	}
	return &copied
}

// jxrDescriptionEmpty 判断图像描述标签结构是否全为空，空结构不写入目录字段
// 入参: desc 图像描述指针
// 返回: bool 是否无有效描述字段
func jxrDescriptionEmpty(desc *ImageDescription) bool {
	if desc == nil {
		return true
	}
	return desc.DocumentName == "" && desc.ImageDescription == "" && desc.EquipmentMake == "" && desc.EquipmentModel == "" &&
		desc.PageName == "" && desc.PageNumber == nil && desc.SoftwareNameVersion == "" && desc.DateTime == "" &&
		desc.ArtistName == "" && desc.HostComputer == "" && desc.CopyrightNotice == ""
}

// jxrDescriptionWork 计算图像描述标签结构在内存中占用的保守字节数
// 入参: desc 图像描述指针
// 返回: uint64 保守内存字节数
func jxrDescriptionWork(desc *ImageDescription) uint64 {
	if desc == nil {
		return 0
	}
	work := uint64(256)
	for _, s := range [10]string{desc.DocumentName, desc.ImageDescription, desc.EquipmentMake, desc.EquipmentModel, desc.PageName, desc.SoftwareNameVersion, desc.DateTime, desc.ArtistName, desc.HostComputer, desc.CopyrightNotice} {
		work += uint64(len(s))
	}
	return work
}

// jxrInfoLinear 判断标准像素格式是否采用线性光编码，定点YCC为扩展色域非线性编码
// 入参: format 标准像素格式
// 返回: bool 是否为线性光数值格式
func jxrInfoLinear(format jxrPixelFormat) bool {
	return format.number != jxrUnsigned && format.color != 1 && format.color != 2 && format.color != 3
}

// cloneComponentInfo 按精确容量复制分量描述，保持空切片语义，不引入追加增长的额外容量
// 入参: components 原分量描述
// 返回: []ComponentInfo 独立且容量等于长度的描述
func cloneComponentInfo(components []ComponentInfo) []ComponentInfo {
	if components == nil {
		return nil
	}
	result := make([]ComponentInfo, len(components))
	copy(result, components)
	return result
}

// Bounds 返回方向变换后的像素范围
// 返回: image.Rectangle 图像范围
func (r *Raster) Bounds() image.Rectangle {
	if r == nil {
		return image.Rectangle{}
	}
	if r.native == nil {
		return r.info.Bounds
	}
	return r.native.bounds
}

// ReadRowNRGBA 针对原生八位或十六位无符号RGB、灰度及RGBE图像，在整数域直接解预乘并写入紧凑八位非预乘NRGBA扫描行
// 不执行ICC或CICP色彩变换，也不经float64绕路，RGBE按线性光经sRGB传递量化且不支持预乘，dst长度须不少于图像宽度乘以四；其余颜色空间或样本编码返回ErrUnsupported
// 入参: y 图像行坐标, dst 目标八位RGBA字节缓冲
// 返回: error 参数、坐标、缓冲或格式能力错误
func (r *Raster) ReadRowNRGBA(y int, dst []byte) error {
	if err := r.readRowNRGBA(y, dst); err != nil {
		return wrapJXRError("Raster.ReadRowNRGBA", err)
	}
	return nil
}

// readRowNRGBA 执行原生八位或十六位RGB、灰度与RGBE的纯整数行读取及解预乘，由公开入口归一化错误
// 入参: y 图像行坐标, dst 目标八位RGBA字节缓冲
// 返回: error 参数、坐标、缓冲或格式能力错误
func (r *Raster) readRowNRGBA(y int, dst []byte) error {
	if r == nil || r.Bounds().Empty() {
		return fmt.Errorf("%w: nil or empty raster", ErrInvalid)
	}
	bounds := r.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	row := y - bounds.Min.Y
	if row < 0 || row >= height || len(dst) < width*4 {
		return fmt.Errorf("%w: raster row or destination", ErrInvalid)
	}
	info := r.info
	stride := info.Channels
	if info.Alpha {
		stride++
	}
	if len(info.Components) != stride {
		return fmt.Errorf("%w: raster components", ErrInvalid)
	}
	isGray := info.ColorSpace == ColorGray && info.Channels == 1
	isRGB := info.ColorSpace == ColorRGB && info.Channels == 3
	if !isGray && !isRGB {
		return fmt.Errorf("%w: integer NRGBA row color space", ErrUnsupported)
	}
	if isRGB && info.Components[0].Format == SampleRGBE {
		return r.readRowRGBENRGBA(row, width, stride, dst)
	}
	precision := info.Components[0].Precision
	if precision != 8 && precision != 16 {
		return fmt.Errorf("%w: integer NRGBA row bit depth", ErrUnsupported)
	}
	for _, comp := range info.Components {
		if comp.Format != SampleUnsigned || comp.Inverted || comp.Precision != precision || comp.XStep != 1 || comp.YStep != 1 || comp.Bounds != bounds {
			return fmt.Errorf("%w: integer NRGBA row component layout", ErrUnsupported)
		}
	}
	interleaved, err := r.checkIntegerRowStorage(bounds, info.Channels, info.Alpha, stride)
	if err != nil {
		return err
	}
	dst = dst[:width*4]
	rowBase := row * width
	if precision == 8 {
		r.readRowNRGBA8(rowBase, width, isGray, info.Alpha, info.Premultiplied, interleaved, dst)
	} else {
		r.readRowNRGBA16(rowBase, width, isGray, info.Alpha, info.Premultiplied, interleaved, dst)
	}
	return nil
}

// checkIntegerRowStorage 校验纯整数行读取所需的十六位紧凑交织或分平面底层存储布局
// 入参: bounds 图像范围, channels 颜色通道数, alpha 是否含透明度, stride 分量总数
// 返回: bool 是否交织存储, error 底层存储不匹配时返回无效错误
func (r *Raster) checkIntegerRowStorage(bounds image.Rectangle, channels int, alpha bool, stride int) (bool, error) {
	pixels := bounds.Dx() * bounds.Dy()
	if r.native != nil && len(r.planes) == 0 {
		if r.native.words != nil || r.native.channels != channels || r.native.alpha != alpha || r.native.bounds != bounds || len(r.native.values) < pixels*stride {
			return false, fmt.Errorf("%w: integer row backing storage", ErrInvalid)
		}
		return true, nil
	}
	if r.native == nil && len(r.planes) == stride {
		for index := range stride {
			p := &r.planes[index]
			if p.words != nil || p.channels != 1 || p.alpha || p.bounds != bounds || len(p.values) < pixels {
				return false, fmt.Errorf("%w: integer row plane storage", ErrInvalid)
			}
		}
		return false, nil
	}
	return false, fmt.Errorf("%w: integer row storage layout", ErrInvalid)
}

// readRowRGBENRGBA 使用二十位整数阈值表将原生共享指数RGBE行直接转换为紧凑八位NRGBA
// 入参: row 零起点行号, width 图像宽度, stride 分量总数, dst 目标八位RGBA切片
// 返回: error 布局或存储错误
func (r *Raster) readRowRGBENRGBA(row, width, stride int, dst []byte) error {
	info := r.info
	bounds := r.Bounds()
	if info.Premultiplied {
		return fmt.Errorf("%w: premultiplied RGBE row", ErrUnsupported)
	}
	for index := range 3 {
		comp := info.Components[index]
		if comp.Format != SampleRGBE || comp.XStep != 1 || comp.YStep != 1 || comp.Bounds != bounds {
			return fmt.Errorf("%w: RGBE row component layout", ErrUnsupported)
		}
	}
	alphaPrecision := uint8(0)
	if info.Alpha {
		comp := info.Components[3]
		if comp.Format != SampleUnsigned || comp.Inverted || (comp.Precision != 8 && comp.Precision != 16) || comp.XStep != 1 || comp.YStep != 1 || comp.Bounds != bounds {
			return fmt.Errorf("%w: RGBE row alpha layout", ErrUnsupported)
		}
		alphaPrecision = comp.Precision
	}
	interleaved, err := r.checkIntegerRowStorage(bounds, 3, info.Alpha, stride)
	if err != nil {
		return err
	}
	dst = dst[:width*4]
	rowBase := row * width
	if interleaved {
		src := r.native.values[rowBase*stride : (rowBase+width)*stride]
		for x := range width {
			si, di := x*stride, x*4
			a := byte(0xff)
			if info.Alpha {
				if alphaPrecision == 16 {
					a = jxrScale16To8(src[si+3])
				} else {
					a = byte(src[si+3])
				}
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrRGBEByte(uint32(src[si])), jxrRGBEByte(uint32(src[si+1])), jxrRGBEByte(uint32(src[si+2])), a
		}
		return nil
	}
	r0, g0, b0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width], r.planes[2].values[rowBase:rowBase+width]
	var a0 []uint16
	if info.Alpha {
		a0 = r.planes[3].values[rowBase : rowBase+width]
	}
	for x := range width {
		di := x * 4
		a := byte(0xff)
		if info.Alpha {
			if alphaPrecision == 16 {
				a = jxrScale16To8(a0[x])
			} else {
				a = byte(a0[x])
			}
		}
		dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrRGBEByte(uint32(r0[x])), jxrRGBEByte(uint32(g0[x])), jxrRGBEByte(uint32(b0[x])), a
	}
	return nil
}

// readRowNRGBA8 在纯整数域完成八位无符号灰度或RGB行的非预乘展开与解预乘输出
// 入参: rowBase 行起像素偏移, width 行宽, isGray 是否灰度, alpha 是否含透明度, premul 是否预乘, interleaved 是否交织存储, dst 目标字节切片
func (r *Raster) readRowNRGBA8(rowBase, width int, isGray, alpha, premul, interleaved bool, dst []byte) {
	if isGray {
		if !alpha {
			var src []uint16
			if interleaved {
				src = r.native.values[rowBase : rowBase+width]
			} else {
				src = r.planes[0].values[rowBase : rowBase+width]
			}
			for x, v := range src {
				b := byte(v)
				di := x * 4
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, 0xff
			}
			return
		}
		if interleaved {
			src := r.native.values[rowBase*2 : (rowBase+width)*2]
			for x := range width {
				si, di := x*2, x*4
				v, a := src[si], src[si+1]
				if premul {
					if a == 0 {
						dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
						continue
					}
					b := jxrUnpremultiplyByte(v, a)
					dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, byte(a)
					continue
				}
				b := byte(v)
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, byte(a)
			}
			return
		}
		g0, a0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width]
		for x := range width {
			di := x * 4
			v, a := g0[x], a0[x]
			if premul {
				if a == 0 {
					dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
					continue
				}
				b := jxrUnpremultiplyByte(v, a)
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, byte(a)
				continue
			}
			b := byte(v)
			dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, byte(a)
		}
		return
	}
	if interleaved {
		if !alpha {
			src := r.native.values[rowBase*3 : (rowBase+width)*3]
			for x := range width {
				si, di := x*3, x*4
				dst[di], dst[di+1], dst[di+2], dst[di+3] = byte(src[si]), byte(src[si+1]), byte(src[si+2]), 0xff
			}
			return
		}
		src := r.native.values[rowBase*4 : (rowBase+width)*4]
		for x := range width {
			si, di := x*4, x*4
			rv, gv, bv, av := src[si], src[si+1], src[si+2], src[si+3]
			if premul {
				if av == 0 {
					dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
					continue
				}
				dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrUnpremultiplyByte(rv, av), jxrUnpremultiplyByte(gv, av), jxrUnpremultiplyByte(bv, av), byte(av)
				continue
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = byte(rv), byte(gv), byte(bv), byte(av)
		}
		return
	}
	r0, g0, b0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width], r.planes[2].values[rowBase:rowBase+width]
	if !alpha {
		for x := range width {
			di := x * 4
			dst[di], dst[di+1], dst[di+2], dst[di+3] = byte(r0[x]), byte(g0[x]), byte(b0[x]), 0xff
		}
		return
	}
	a0 := r.planes[3].values[rowBase : rowBase+width]
	for x := range width {
		di := x * 4
		rv, gv, bv, av := r0[x], g0[x], b0[x], a0[x]
		if premul {
			if av == 0 {
				dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
				continue
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrUnpremultiplyByte(rv, av), jxrUnpremultiplyByte(gv, av), jxrUnpremultiplyByte(bv, av), byte(av)
			continue
		}
		dst[di], dst[di+1], dst[di+2], dst[di+3] = byte(rv), byte(gv), byte(bv), byte(av)
	}
}

// readRowNRGBA16 在纯整数域完成十六位无符号灰度或RGB行的精确缩位与解预乘输出
// 入参: rowBase 行起像素偏移, width 行宽, isGray 是否灰度, alpha 是否含透明度, premul 是否预乘, interleaved 是否交织存储, dst 目标字节切片
func (r *Raster) readRowNRGBA16(rowBase, width int, isGray, alpha, premul, interleaved bool, dst []byte) {
	if isGray {
		if !alpha {
			var src []uint16
			if interleaved {
				src = r.native.values[rowBase : rowBase+width]
			} else {
				src = r.planes[0].values[rowBase : rowBase+width]
			}
			for x, v := range src {
				b := jxrScale16To8(v)
				di := x * 4
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, 0xff
			}
			return
		}
		if interleaved {
			src := r.native.values[rowBase*2 : (rowBase+width)*2]
			for x := range width {
				si, di := x*2, x*4
				v, a := src[si], src[si+1]
				if premul {
					if a == 0 {
						dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
						continue
					}
					b := jxrUnpremultiplyByte(v, a)
					dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, jxrScale16To8(a)
					continue
				}
				b := jxrScale16To8(v)
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, jxrScale16To8(a)
			}
			return
		}
		g0, a0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width]
		for x := range width {
			di := x * 4
			v, a := g0[x], a0[x]
			if premul {
				if a == 0 {
					dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
					continue
				}
				b := jxrUnpremultiplyByte(v, a)
				dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, jxrScale16To8(a)
				continue
			}
			b := jxrScale16To8(v)
			dst[di], dst[di+1], dst[di+2], dst[di+3] = b, b, b, jxrScale16To8(a)
		}
		return
	}
	if interleaved {
		if !alpha {
			src := r.native.values[rowBase*3 : (rowBase+width)*3]
			for x := range width {
				si, di := x*3, x*4
				dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(src[si]), jxrScale16To8(src[si+1]), jxrScale16To8(src[si+2]), 0xff
			}
			return
		}
		src := r.native.values[rowBase*4 : (rowBase+width)*4]
		for x := range width {
			si, di := x*4, x*4
			rv, gv, bv, av := src[si], src[si+1], src[si+2], src[si+3]
			if premul {
				if av == 0 {
					dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
					continue
				}
				dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrUnpremultiplyByte(rv, av), jxrUnpremultiplyByte(gv, av), jxrUnpremultiplyByte(bv, av), jxrScale16To8(av)
				continue
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(rv), jxrScale16To8(gv), jxrScale16To8(bv), jxrScale16To8(av)
		}
		return
	}
	r0, g0, b0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width], r.planes[2].values[rowBase:rowBase+width]
	if !alpha {
		for x := range width {
			di := x * 4
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(r0[x]), jxrScale16To8(g0[x]), jxrScale16To8(b0[x]), 0xff
		}
		return
	}
	a0 := r.planes[3].values[rowBase : rowBase+width]
	for x := range width {
		di := x * 4
		rv, gv, bv, av := r0[x], g0[x], b0[x], a0[x]
		if premul {
			if av == 0 {
				dst[di], dst[di+1], dst[di+2], dst[di+3] = 0, 0, 0, 0
				continue
			}
			dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrUnpremultiplyByte(rv, av), jxrUnpremultiplyByte(gv, av), jxrUnpremultiplyByte(bv, av), jxrScale16To8(av)
			continue
		}
		dst[di], dst[di+1], dst[di+2], dst[di+3] = jxrScale16To8(rv), jxrScale16To8(gv), jxrScale16To8(bv), jxrScale16To8(av)
	}
}

// ReadRowCMYK 针对原生八位或十六位非预乘CMYK与CMYKDirect图像，在整数域直接输出紧凑交织八位CMYK扫描行
// dst长度须不少于图像宽度乘以四；预乘CMYK、非CMYK或非八位/十六位无符号格式返回ErrUnsupported
// 入参: y 图像行坐标, dst 目标八位CMYK字节缓冲
// 返回: error 参数、坐标、缓冲或格式能力错误
func (r *Raster) ReadRowCMYK(y int, dst []byte) error {
	if err := r.readRowCMYK(y, dst, nil); err != nil {
		return wrapJXRError("Raster.ReadRowCMYK", err)
	}
	return nil
}

// ReadRowCMYKAlpha 针对原生八位或十六位非预乘CMYK与CMYKDirect图像，同时输出紧凑交织八位CMYK扫描行及可选独立八位Alpha扫描行
// cmyk长度须不少于图像宽度乘以四，alpha非nil时长度须不少于图像宽度（源无Alpha时填充0xff）；不支持预乘CMYK
// 入参: y 图像行坐标, cmyk 目标八位CMYK字节缓冲, alpha 可选独立八位Alpha字节缓冲
// 返回: error 参数、坐标、缓冲或格式能力错误
func (r *Raster) ReadRowCMYKAlpha(y int, cmyk, alpha []byte) error {
	if err := r.readRowCMYK(y, cmyk, alpha); err != nil {
		return wrapJXRError("Raster.ReadRowCMYKAlpha", err)
	}
	return nil
}

// readRowCMYK 在纯整数域完成八位或十六位CMYK及可选独立Alpha扫描行读取，由公开入口归一化错误
// 入参: y 图像行坐标, cmyk 目标八位CMYK字节缓冲, alpha 可选独立八位Alpha字节缓冲
// 返回: error 参数、坐标、缓冲或格式能力错误
func (r *Raster) readRowCMYK(y int, cmyk, alpha []byte) error {
	if r == nil || r.Bounds().Empty() {
		return fmt.Errorf("%w: nil or empty raster", ErrInvalid)
	}
	bounds := r.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	row := y - bounds.Min.Y
	if row < 0 || row >= height || len(cmyk) < width*4 || alpha != nil && len(alpha) < width {
		return fmt.Errorf("%w: CMYK row or destination", ErrInvalid)
	}
	info := r.info
	if (info.ColorSpace != ColorCMYK && info.ColorSpace != ColorCMYKDirect) || info.Channels != 4 {
		return fmt.Errorf("%w: integer CMYK row color space", ErrUnsupported)
	}
	if info.Premultiplied {
		return fmt.Errorf("%w: premultiplied CMYK row", ErrUnsupported)
	}
	stride := 4
	if info.Alpha {
		stride = 5
	}
	if len(info.Components) != stride {
		return fmt.Errorf("%w: CMYK row components", ErrInvalid)
	}
	precision := info.Components[0].Precision
	if precision != 8 && precision != 16 {
		return fmt.Errorf("%w: integer CMYK row bit depth", ErrUnsupported)
	}
	for _, comp := range info.Components {
		if comp.Format != SampleUnsigned || comp.Inverted || comp.Precision != precision || comp.XStep != 1 || comp.YStep != 1 || comp.Bounds != bounds {
			return fmt.Errorf("%w: integer CMYK row component layout", ErrUnsupported)
		}
	}
	interleaved, err := r.checkIntegerRowStorage(bounds, 4, info.Alpha, stride)
	if err != nil {
		return err
	}
	cmyk = cmyk[:width*4]
	if alpha != nil {
		alpha = alpha[:width]
	}
	rowBase := row * width
	if interleaved {
		src := r.native.values[rowBase*stride : (rowBase+width)*stride]
		if precision == 8 {
			for x := range width {
				si, di := x*stride, x*4
				cmyk[di], cmyk[di+1], cmyk[di+2], cmyk[di+3] = byte(src[si]), byte(src[si+1]), byte(src[si+2]), byte(src[si+3])
				if alpha != nil {
					if info.Alpha {
						alpha[x] = byte(src[si+4])
					} else {
						alpha[x] = 0xff
					}
				}
			}
			return nil
		}
		for x := range width {
			si, di := x*stride, x*4
			cmyk[di], cmyk[di+1], cmyk[di+2], cmyk[di+3] = jxrScale16To8(src[si]), jxrScale16To8(src[si+1]), jxrScale16To8(src[si+2]), jxrScale16To8(src[si+3])
			if alpha != nil {
				if info.Alpha {
					alpha[x] = jxrScale16To8(src[si+4])
				} else {
					alpha[x] = 0xff
				}
			}
		}
		return nil
	}
	c0, m0, y0, k0 := r.planes[0].values[rowBase:rowBase+width], r.planes[1].values[rowBase:rowBase+width], r.planes[2].values[rowBase:rowBase+width], r.planes[3].values[rowBase:rowBase+width]
	var a0 []uint16
	if info.Alpha {
		a0 = r.planes[4].values[rowBase : rowBase+width]
	}
	if precision == 8 {
		for x := range width {
			di := x * 4
			cmyk[di], cmyk[di+1], cmyk[di+2], cmyk[di+3] = byte(c0[x]), byte(m0[x]), byte(y0[x]), byte(k0[x])
			if alpha != nil {
				if info.Alpha {
					alpha[x] = byte(a0[x])
				} else {
					alpha[x] = 0xff
				}
			}
		}
		return nil
	}
	for x := range width {
		di := x * 4
		cmyk[di], cmyk[di+1], cmyk[di+2], cmyk[di+3] = jxrScale16To8(c0[x]), jxrScale16To8(m0[x]), jxrScale16To8(y0[x]), jxrScale16To8(k0[x])
		if alpha != nil {
			if info.Alpha {
				alpha[x] = jxrScale16To8(a0[x])
			} else {
				alpha[x] = 0xff
			}
		}
	}
	return nil
}

// jxrScale16To8 将十六位无符号样本在整数域精确舍入到八位无符号字节
// 入参: v 十六位无符号样本
// 返回: byte 八位无符号字节
func jxrScale16To8(v uint16) byte {
	return byte((uint32(v) + 128) / 257)
}

// jxrUnpremultiplyByte 在纯整数域按非零透明度解除八位或十六位预乘并舍入为八位无符号字节
// 入参: c 预乘颜色样本, a 非零透明度样本
// 返回: byte 非预乘八位无符号字节
func jxrUnpremultiplyByte(c, a uint16) byte {
	if c >= a {
		return 255
	}
	return byte((uint32(c)*255 + uint32(a)/2) / uint32(a))
}

// jxrRGBEByte 使用二十位定点阈值表将原生RGBE位模式在纯整数域转换为八位sRGB字节
// 入参: bits 低八位尾数与高八位共享指数
// 返回: byte 八位sRGB字节
func jxrRGBEByte(bits uint32) byte {
	e := bits >> 8
	m := bits & 255
	if m == 0 || e < 116 {
		return 0
	}
	if e >= 136 {
		return 255
	}
	lin := m << (e - 116)
	if lin >= 1<<20 {
		return 255
	}
	lo, hi := uint32(0), uint32(255)
	for lo < hi {
		mid := (lo + hi + 1) >> 1
		if lin >= jxrSRGB8Threshold[mid] {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return byte(lo)
}

// Component 返回指定通道的共享视图，透明度通道位于颜色通道之后
// 入参: index 从零开始的通道序号
// 返回: Component 通道视图, error 序号错误
func (r *Raster) Component(index int) (Component, error) {
	component, err := r.component(index)
	if err != nil {
		return Component{}, wrapJXRError("Raster.Component", err)
	}
	return component, nil
}

// component 返回指定通道的共享视图而不包装公开操作名，供公开入口与内部调用共用
// 入参: index 从零开始的通道序号
// 返回: Component 通道视图, error 序号错误
func (r *Raster) component(index int) (Component, error) {
	if r == nil {
		return Component{}, fmt.Errorf("%w: nil raster", ErrInvalid)
	}
	if len(r.planes) != 0 {
		if index < 0 || index >= len(r.planes) || index >= len(r.info.Components) {
			return Component{}, fmt.Errorf("%w: component index out of range", ErrInvalid)
		}
		return Component{source: &r.planes[index], info: r.info.Components[index]}, nil
	}
	if r.native == nil {
		return Component{}, fmt.Errorf("%w: uninitialized raster", ErrInvalid)
	}
	count := r.info.Channels
	if r.native.alpha {
		count++
	}
	if index < 0 || index >= count || index >= len(r.info.Components) {
		return Component{}, fmt.Errorf("%w: component index out of range", ErrInvalid)
	}
	return Component{source: r.native, index: index, info: r.info.Components[index]}, nil
}

// Info 返回分量范围、采样间隔及定位信息
// 返回: ComponentInfo 分量信息
func (c Component) Info() ComponentInfo {
	return c.info
}

// Bounds 返回分量自身网格范围，不把子采样分量扩展为图像尺寸
// 返回: image.Rectangle 分量范围
func (c Component) Bounds() image.Rectangle {
	if c.source == nil {
		return image.Rectangle{}
	}
	return c.source.bounds
}

// Sample 返回最接近的float64分量数值，无符号整数归一到单位区间，定点及浮点保留原范围
// 入参: x, y 分量网格坐标
// 返回: float64 分量值, error 视图或坐标错误
func (c Component) Sample(x, y int) (float64, error) {
	value, err := c.sampleAt(x, y)
	if err != nil {
		return 0, wrapJXRError("Component.Sample", err)
	}
	return value, nil
}

// sampleAt 返回未包装公开操作名的分量浮点值，供内部显示与重采样共用
// 入参: x, y 分量网格坐标
// 返回: float64 分量值, error 视图或坐标错误
func (c Component) sampleAt(x, y int) (float64, error) {
	at, err := c.sampleIndex(x, y)
	if err != nil {
		return 0, err
	}
	return sampleValue(c.source.sampleBits(at), c.info), nil
}

// SampleBits 返回原生样本位模式，不经过显示量化或浮点数值转换
// 定点以补码表示，浮点保留IEEE位模式；RGBE低八位为尾数，其后八位为共享指数
// 入参: x, y 分量网格坐标
// 返回: uint32 样本位模式，未使用的高位为零, error 视图或坐标错误
func (c Component) SampleBits(x, y int) (uint32, error) {
	bits, err := c.sampleBitsAt(x, y)
	if err != nil {
		return 0, wrapJXRError("Component.SampleBits", err)
	}
	return bits, nil
}

// sampleBitsAt 返回未包装公开操作名的原生样本位模式，供公开入口与内部路径共用
// 入参: x, y 分量网格坐标
// 返回: uint32 样本位模式, error 视图或坐标错误
func (c Component) sampleBitsAt(x, y int) (uint32, error) {
	at, err := c.sampleIndex(x, y)
	if err != nil {
		return 0, err
	}
	return c.source.sampleBits(at), nil
}

// ReadRowBits 批量读取指定行的原生样本位模式，dst长度不得小于分量宽度
// 入参: y 分量网格行坐标, dst 目标位模式缓冲
// 返回: error 视图、行号或缓冲区长度错误
func (c Component) ReadRowBits(y int, dst []uint32) error {
	base, stride, width, err := c.rowAccess("Component.ReadRowBits", y, len(dst))
	if err != nil {
		return err
	}
	dst = dst[:width]
	if words := c.source.words; words != nil {
		if stride == 1 {
			copy(dst, words[base:base+width])
			return nil
		}
		for x := range width {
			dst[x] = words[base+x*stride]
		}
		return nil
	}
	values := c.source.values
	if stride == 1 {
		src := values[base : base+width]
		for x, v := range src {
			dst[x] = uint32(v)
		}
		return nil
	}
	for x := range width {
		dst[x] = uint32(values[base+x*stride])
	}
	return nil
}

// ReadRow 批量读取指定行的float64分量数值，无符号整数归一到单位区间，定点及浮点保留原范围
// 入参: y 分量网格行坐标, dst 目标数值缓冲，长度不得小于分量宽度
// 返回: error 视图、行号或缓冲区长度错误
func (c Component) ReadRow(y int, dst []float64) error {
	base, stride, width, err := c.rowAccess("Component.ReadRow", y, len(dst))
	if err != nil {
		return err
	}
	dst = dst[:width]
	info := c.info
	words, values := c.source.words, c.source.values
	switch info.Format {
	case SampleFixed:
		scale := math.Ldexp(1, -int(info.FractionBits))
		if info.Precision == 16 {
			if words != nil {
				for x := range width {
					dst[x] = float64(int16(words[base+x*stride])) * scale
				}
				return nil
			}
			for x := range width {
				dst[x] = float64(int16(values[base+x*stride])) * scale
			}
			return nil
		}
		if words != nil {
			for x := range width {
				dst[x] = float64(int32(words[base+x*stride])) * scale
			}
			return nil
		}
		for x := range width {
			dst[x] = float64(int32(values[base+x*stride])) * scale
		}
		return nil
	case SampleFloat:
		if info.Precision == 16 {
			if words != nil {
				for x := range width {
					dst[x] = jxrHalfValue(uint16(words[base+x*stride]))
				}
				return nil
			}
			for x := range width {
				dst[x] = jxrHalfValue(values[base+x*stride])
			}
			return nil
		}
		if words != nil {
			for x := range width {
				dst[x] = float64(math.Float32frombits(words[base+x*stride]))
			}
			return nil
		}
		for x := range width {
			dst[x] = float64(math.Float32frombits(uint32(values[base+x*stride])))
		}
		return nil
	case SampleRGBE:
		for x := range width {
			bits := c.source.sampleBits(base + x*stride)
			if bits>>8 == 0 {
				dst[x] = 0
			} else {
				dst[x] = math.Ldexp(float64(bits&255), int(bits>>8)-136)
			}
		}
		return nil
	default:
		denom := float64((uint64(1) << info.Precision) - 1)
		if info.Inverted {
			if words != nil {
				for x := range width {
					dst[x] = 1 - float64(words[base+x*stride])/denom
				}
				return nil
			}
			for x := range width {
				dst[x] = 1 - float64(values[base+x*stride])/denom
			}
			return nil
		}
		if words != nil {
			for x := range width {
				dst[x] = float64(words[base+x*stride]) / denom
			}
			return nil
		}
		if stride == 1 {
			src := values[base : base+width]
			for x, v := range src {
				dst[x] = float64(v) / denom
			}
			return nil
		}
		for x := range width {
			dst[x] = float64(values[base+x*stride]) / denom
		}
		return nil
	}
}

// rowAccess 校验行号与目标缓冲长度并返回起始索引、步长与行宽
// 入参: op 公开操作名, y 行坐标, dstLen 目标缓冲长度
// 返回: int 起始索引, int 像素步长, int 分量宽度, error 参数错误
func (c Component) rowAccess(op string, y, dstLen int) (int, int, int, error) {
	if c.source == nil {
		return 0, 0, 0, &FormatError{Op: op, Detail: "component view is uninitialized"}
	}
	width := c.source.bounds.Dx()
	if y < c.source.bounds.Min.Y || y >= c.source.bounds.Max.Y {
		return 0, 0, 0, &FormatError{Op: op, Detail: "component row out of range"}
	}
	if dstLen < width {
		return 0, 0, 0, &FormatError{Op: op, Detail: "destination row buffer too short"}
	}
	stride := c.source.channels
	if c.source.alpha {
		stride++
	}
	if stride <= 0 || c.index < 0 || c.index >= stride {
		return 0, 0, 0, &FormatError{Op: op, Detail: "component channel out of range"}
	}
	base := (y-c.source.bounds.Min.Y)*width*stride + c.index
	total := len(c.source.values)
	if c.source.words != nil {
		total = len(c.source.words)
	}
	if width > 0 && (base < 0 || base+(width-1)*stride >= total) {
		return 0, 0, 0, &FormatError{Op: op, Detail: "component row backing storage out of range"}
	}
	return base, stride, width, nil
}

// checkJXRDimensions 分别校验标准尺寸、宿主坐标和像素预算，在乘法前拒绝不能表示的面积
// 入参: width, height 正数轴长, maxPixels 像素预算
// 返回: error 标准范围或资源容量错误
func checkJXRDimensions(width, height, maxPixels uint64) error {
	if width == 0 || height == 0 || width > jxrDimensionLimit || height > jxrDimensionLimit {
		return fmt.Errorf("%w: JPEG XR image dimension fields", ErrInvalid)
	}
	maxInt := uint64(^uint(0) >> 1)
	if width > maxInt || height > maxInt {
		return &LimitError{Resource: "max dimension", Limit: maxInt, Required: max(width, height), Detail: "JPEG XR image dimension capacity"}
	}
	if width > maxPixels/height {
		return &LimitError{Resource: "max pixels", Limit: maxPixels, Required: jxrSaturatedMul(width, height), Detail: "JPEG XR image pixel count"}
	}
	return nil
}

// jxrWorkTotal 在指定额度内累计同时驻留容量，避免求和溢出后误判为较小工作量
// 入参: maximum 工作额度, amounts 各项活动容量
// 返回: uint64 总容量，超出额度时为零, bool 是否满足额度
func jxrWorkTotal(maximum uint64, amounts ...uint64) (uint64, bool) {
	work := uint64(0)
	for _, amount := range amounts {
		if amount > maximum-work {
			return 0, false
		}
		work += amount
	}
	return work, true
}

// normalized 补齐默认资源上限，并将工作内存额度限制在宿主可寻址范围内
// 返回: Limits 完整资源上限
func (l Limits) normalized() Limits {
	defaults := DefaultLimits()
	if l.MaxInputBytes == 0 {
		l.MaxInputBytes = defaults.MaxInputBytes
	}
	if l.MaxPixels == 0 {
		l.MaxPixels = defaults.MaxPixels
	}
	if l.MaxMemoryBytes == 0 {
		l.MaxMemoryBytes = defaults.MaxMemoryBytes
	}
	l.MaxMemoryBytes = min(l.MaxMemoryBytes, uint64(^uint(0)))
	return l
}

// readJXRPixelFormat 解释标准像素格式标识，不把未知格式当作损坏数据或任意RGB
// 入参: data 按文件顺序排列的十六字节标识
// 返回: jxrPixelFormat 像素格式, error 数据或格式能力错误
func readJXRPixelFormat(data []byte) (jxrPixelFormat, error) {
	if len(data) != 16 {
		return jxrPixelFormat{}, fmt.Errorf("%w: JPEG XR pixel format length", ErrInvalid)
	}
	if string(data[:15]) != jxrPixelPrefix {
		return jxrPixelFormat{}, fmt.Errorf("%w: JPEG XR pixel format identifier", ErrUnsupported)
	}
	id := data[15]
	if format, ok := jxrLookupPixelFormatID(id, 0); ok {
		return format, nil
	}
	return jxrPixelFormat{}, fmt.Errorf("%w: JPEG XR pixel format %x", ErrUnsupported, id)
}

// jxrLookupPixelFormatID 从标准GUID表中按末字节检索像素格式，并在二值反向时对齐位深十五
// 入参: id 标准GUID末字节, targetDepth 目标位深，仅在十五且名称为BlackWhite时调整
// 返回: jxrPixelFormat 标准像素格式, bool 是否为已知标准标识
func jxrLookupPixelFormatID(id, targetDepth uint8) (jxrPixelFormat, bool) {
	if int(id) >= len(jxrStandardPixelFormats) {
		return jxrPixelFormat{}, false
	}
	format := jxrStandardPixelFormats[id]
	if format.name == "" {
		return jxrPixelFormat{}, false
	}
	if format.name == "BlackWhite" && targetDepth == 15 {
		format.depth = 15
	}
	return format, true
}

// jxrPixelFormatMatches 比较两种像素格式的通道、颜色、位深、数值编码和透明度属性
// 入参: a, b 待比较像素格式
// 返回: bool 核心编码属性是否完全一致
func jxrPixelFormatMatches(a, b jxrPixelFormat) bool {
	return a.channels == b.channels && a.color == b.color && a.depth == b.depth && a.number == b.number && a.alpha == b.alpha && a.premultiply == b.premultiply
}

// jxrColorSpace 将标准输出颜色格式映射为分量颜色解释
// 入参: color 标准输出颜色格式
// 返回: ColorSpace 分量颜色空间
func jxrColorSpace(color uint8) ColorSpace {
	switch color {
	case 0:
		return ColorGray
	case 1, 2, 3:
		return ColorYCC
	case 4:
		return ColorCMYK
	case 5:
		return ColorCMYKDirect
	case 6:
		return ColorMultiChannel
	case 7, 8:
		return ColorRGB
	default:
		return ColorUnknown
	}
}

// NewRaster 创建样本全零的原生图像并复制元数据，输入元数据不计入新图像内存预算
// 缺省的Channels、采样间隔、分量范围、全采样定位及像素格式名称均由分量与数值编码自动补齐
// 入参: info 图像及完整数值编码信息, limits 新图像资源限制，零值采用默认上限
// 返回: *Raster 独立原生图像, error 元数据或资源错误
func NewRaster(info Info, limits Limits) (*Raster, error) {
	raster, err := newRasterContext(context.Background(), info, limits)
	if err != nil {
		return nil, wrapJXRError("NewRaster", err)
	}
	return raster, nil
}

// newRasterContext 在分配前检查全部样本及元数据容量，分配和复制期间检查取消状态
// 入参: ctx 已规范化Context, info 图像信息, limits 新图像资源限制
// 返回: *Raster 独立原生图像, error 元数据、资源或取消错误
func newRasterContext(ctx context.Context, info Info, limits Limits) (*Raster, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limits = limits.normalized()
	if len(info.Components) == 0 || len(info.Components) > rasterMaxComponents+1 {
		return nil, fmt.Errorf("%w: raster component count", ErrInvalid)
	}
	memory := uint64(512) + uint64(len(info.Components))*256 + uint64(len(info.ICCProfile)) + uint64(len(info.EXIF)) + uint64(len(info.GPS)) + uint64(len(info.XMP)) + uint64(len(info.IPTC)) + jxrDescriptionWork(info.Description)
	if info.ColorEncoding != nil {
		memory += 4
	}
	if memory > limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: memory, Detail: "raster metadata storage"}
	}
	if info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || uint64(info.Bounds.Max.X) > 1<<32 || uint64(info.Bounds.Max.Y) > 1<<32 {
		return nil, fmt.Errorf("%w: raster image bounds", ErrInvalid)
	}
	width, height := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
	if width > limits.MaxPixels/height {
		return nil, &LimitError{Resource: "max pixels", Limit: limits.MaxPixels, Required: jxrSaturatedMul(width, height), Detail: "raster image pixels"}
	}
	info.Components = cloneComponentInfo(info.Components)
	if err := normalizeRasterInfo(&info); err != nil {
		return nil, err
	}
	for _, component := range info.Components {
		count := uint64(component.Bounds.Dx()) * uint64(component.Bounds.Dy())
		bytes := component.storageBytes()
		if count > uint64(^uint(0)>>1)/bytes {
			return nil, &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: jxrSaturatedMul(count, bytes), Detail: "raster sample storage"}
		}
		if count > (limits.MaxMemoryBytes-memory)/bytes {
			return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(memory, jxrSaturatedMul(count, bytes)), Detail: "raster sample storage"}
		}
		memory += count * bytes
	}
	raster := &Raster{info: info, planes: make([]imageComponents, len(info.Components))}
	raster.info.ColorEncoding = cloneColorEncoding(info.ColorEncoding)
	raster.info.EXIF = cloneBytes(info.EXIF)
	raster.info.GPS = cloneBytes(info.GPS)
	raster.info.XMP = cloneBytes(info.XMP)
	raster.info.IPTC = cloneBytes(info.IPTC)
	raster.info.Description = cloneImageDescription(info.Description)
	if len(info.ICCProfile) != 0 {
		raster.info.ICCProfile = make([]byte, len(info.ICCProfile))
		for at := 0; at < len(info.ICCProfile); {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := at + min(32768, len(info.ICCProfile)-at)
			copy(raster.info.ICCProfile[at:end], info.ICCProfile[at:end])
			at = end
		}
	} else {
		raster.info.ICCProfile = nil
	}
	for index, component := range info.Components {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		plane := imageComponents{bounds: component.Bounds, channels: 1}
		count := component.Bounds.Dx() * component.Bounds.Dy()
		if component.Precision > 16 {
			plane.words = make([]uint32, count)
		} else {
			plane.values = make([]uint16, count)
		}
		raster.planes[index] = plane
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return raster, nil
}

// Clone 复制元数据及原生样本，原图内存不计入副本资源限制
// 入参: ctx Context，nil采用Background, limits 副本资源限制
// 返回: *Raster 独立图像副本, error 图像、资源或取消错误
func (r *Raster) Clone(ctx context.Context, limits Limits) (*Raster, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	result, err := r.clone(ctx, limits)
	if err != nil {
		return nil, wrapJXRError("Raster.Clone", err)
	}
	return result, nil
}

// clone 执行原生图像元数据及样本的完整深拷贝，由公开入口统一归一化错误
// 入参: ctx 已规范化Context, limits 副本资源限制
// 返回: *Raster 独立图像副本, error 图像、资源或取消错误
func (r *Raster) clone(ctx context.Context, limits Limits) (*Raster, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil {
		return nil, fmt.Errorf("%w: nil raster", ErrInvalid)
	}
	result, err := newRasterContext(ctx, r.info, limits)
	if err != nil {
		return nil, err
	}
	for index := range result.info.Components {
		source, err := r.component(index)
		if err != nil {
			return nil, err
		}
		target := &result.planes[index]
		if source.source.channels == 1 && !source.source.alpha && (source.source.words != nil) == (target.words != nil) {
			count := len(target.values) + len(target.words)
			for at := 0; at < count; {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				end := at + min(16384, count-at)
				if target.words != nil {
					copy(target.words[at:end], source.source.words[at:end])
				} else {
					copy(target.values[at:end], source.source.values[at:end])
				}
				at = end
			}
			continue
		}
		if source.Bounds() != target.bounds {
			return nil, fmt.Errorf("%w: raster copy component bounds", ErrInvalid)
		}
		stride := source.source.channels
		if source.source.alpha {
			stride++
		}
		count := len(target.values) + len(target.words)
		for at := 0; at < count; {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := at + min(16384, count-at)
			for index := at; index < end; index++ {
				target.storeBits(index, source.source.sampleBits(index*stride+source.index))
			}
			at = end
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// newImageComponents 检查通道、样本宽度和存储预算后分配连续样本
// 入参: bounds 图像范围, channels 颜色通道数, alpha 是否包含透明度, width 每个样本的存储字节数, maxPixels 剩余RGBA64等价容量
// 返回: *imageComponents 原生分量图像, error 结构或资源错误
func newImageComponents(bounds image.Rectangle, channels int, alpha bool, width uint64, maxPixels uint64) (*imageComponents, error) {
	if bounds.Empty() || bounds.Min.X != 0 || bounds.Min.Y != 0 || channels < 1 || channels > rasterMaxComponents || width != 2 && width != 4 {
		return nil, fmt.Errorf("%w: image component layout", ErrInvalid)
	}
	stride := channels
	if alpha {
		stride++
	}
	units := (uint64(stride)*width + 7) / 8
	if err := checkJXRDimensions(uint64(bounds.Dx()), uint64(bounds.Dy()), maxPixels/units); err != nil {
		return nil, err
	}
	count := uint64(bounds.Dx()) * uint64(bounds.Dy())
	if count > uint64(^uint(0)>>1)/(width*uint64(stride)) {
		return nil, fmt.Errorf("%w: image component storage", ErrLimit)
	}
	output := &imageComponents{bounds: bounds, channels: channels, alpha: alpha}
	if width == 4 {
		output.words = make([]uint32, int(count)*stride)
	} else {
		output.values = make([]uint16, int(count)*stride)
	}
	return output, nil
}

// sampleBits 读取已校验索引处的原生样本位模式
// 入参: at 连续样本索引
// 返回: uint32 未使用高位为零的位模式
func (c *imageComponents) sampleBits(at int) uint32 {
	if c.words != nil {
		return c.words[at]
	}
	return uint32(c.values[at])
}

// storeBits 保存已校验索引处的原生样本位模式
// 入参: at 连续样本索引, bits 与存储宽度一致的位模式
func (c *imageComponents) storeBits(at int, bits uint32) {
	if c.words != nil {
		c.words[at] = bits
	} else {
		c.values[at] = uint16(bits)
	}
}

// copySample 在编码相同的分量之间复制完整原生样本
// 入参: at 目标索引, source 源分量存储, from 源索引
func (c *imageComponents) copySample(at int, source *imageComponents, from int) {
	c.storeBits(at, source.sampleBits(from))
}

// storageBytes 返回每个原生样本占用的字节数
// 返回: uint64 单样本存储字节数
func (c ComponentInfo) storageBytes() uint64 {
	if c.Precision > 16 {
		return 4
	}
	return 2
}

// normalizeRasterInfo 补充默认网格并校验图像的颜色、数值编码和采样布局
// 入参: info 已独立复制的图像信息
// 返回: error 元数据错误
func normalizeRasterInfo(info *Info) error {
	count := len(info.Components)
	if info.Alpha {
		count--
	}
	if info.Channels == 0 {
		info.Channels = count
	}
	if count < 1 || count > rasterMaxComponents || info.Channels != count || info.Premultiplied && !info.Alpha || info.Format > FormatCodestream || info.ImageType&^3 != 0 || info.Orientation > OrientationRotate90FlipXY {
		return fmt.Errorf("%w: raster channels or format", ErrInvalid)
	}
	for _, resolution := range []float64{info.Resolution.X, info.Resolution.Y} {
		if !geometryFinite(resolution) || resolution < 0 || resolution > 0 && (float32(resolution) == 0 || math.IsInf(float64(float32(resolution)), 0)) {
			return fmt.Errorf("%w: raster resolution", ErrInvalid)
		}
	}
	for index := range info.Components {
		component := &info.Components[index]
		if component.XStep == 0 {
			component.XStep = 1
		}
		if component.YStep == 0 {
			component.YStep = 1
		}
		if component.XStep > 2 || component.YStep > 2 || info.Bounds.Dx()%int(component.XStep) != 0 || info.Bounds.Dy()%int(component.YStep) != 0 {
			return fmt.Errorf("%w: raster sampling interval", ErrInvalid)
		}
		bounds := image.Rect(0, 0, info.Bounds.Dx()/int(component.XStep), info.Bounds.Dy()/int(component.YStep))
		if component.Bounds == (image.Rectangle{}) {
			component.Bounds = bounds
		} else if component.Bounds != bounds {
			return fmt.Errorf("%w: raster component bounds", ErrInvalid)
		}
		if !geometryFinite(component.XOffset) || !geometryFinite(component.YOffset) {
			return fmt.Errorf("%w: raster sampling position", ErrInvalid)
		}
		if component.XStep == 1 && component.XOffset != 0 || component.YStep == 1 && component.YOffset != 0 {
			return fmt.Errorf("%w: raster full-rate sampling position", ErrInvalid)
		}
		if component.XStep == 1 {
			component.XPositionKnown = true
		}
		if component.YStep == 1 {
			component.YPositionKnown = true
		}
		offsets := [2]float64{component.XOffset, component.YOffset}
		steps := [2]uint8{component.XStep, component.YStep}
		for axis, known := range [2]bool{component.XPositionKnown, component.YPositionKnown} {
			offset := offsets[axis]
			if known && (offset < 0 || offset > float64(steps[axis]-1) || offset*4 != math.Trunc(offset*4)) {
				return fmt.Errorf("%w: raster chroma position", ErrInvalid)
			}
		}
	}
	format, err := rasterPixelFormat(*info)
	if err != nil {
		return err
	}
	if err := checkJXRColorMetadata(format, info.ColorHint, info.ColorEncoding); err != nil {
		return err
	}
	linear := jxrInfoLinear(format)
	if info.Linear && !linear {
		return fmt.Errorf("%w: raster linear representation", ErrInvalid)
	}
	info.Linear = linear
	name := codedPixelFormatName(format)
	if info.PixelFormat == "" {
		info.PixelFormat = name
	} else if info.PixelFormat != name && !rasterFormatNameMatches(info.PixelFormat, format) {
		return fmt.Errorf("%w: raster pixel format name", ErrInvalid)
	}
	return nil
}

// rasterPixelFormat 按原生分量描述确定标准输出格式，不根据通道数量猜测颜色空间
// 入参: info 已补充默认网格的图像信息
// 返回: jxrPixelFormat 标准输出格式, error 颜色、数值或采样错误
func rasterPixelFormat(info Info) (jxrPixelFormat, error) {
	format := jxrPixelFormat{channels: info.Channels, alpha: info.Alpha, premultiply: info.Premultiplied}
	first := info.Components[0]
	switch info.ColorSpace {
	case ColorGray:
		format.color = 0
		if info.Channels != 1 {
			return format, fmt.Errorf("%w: grayscale channel count", ErrInvalid)
		}
	case ColorRGB:
		format.color = 7
		if info.Channels != 3 {
			return format, fmt.Errorf("%w: RGB channel count", ErrInvalid)
		}
		if first.Format == SampleRGBE {
			format.color = 8
		}
	case ColorCMYK, ColorCMYKDirect:
		format.color = 4
		if info.ColorSpace == ColorCMYKDirect {
			format.color = 5
		}
		if info.Channels != 4 {
			return format, fmt.Errorf("%w: CMYK channel count", ErrInvalid)
		}
	case ColorYCC:
		if info.Channels != 3 {
			return format, fmt.Errorf("%w: YCC channel count", ErrInvalid)
		}
		chroma, other := info.Components[1], info.Components[2]
		sameGrid := chroma.Bounds == other.Bounds && chroma.XStep == other.XStep && chroma.YStep == other.YStep
		if !sameGrid || !jxrSharedChromaPhase(chroma, other) {
			return format, fmt.Errorf("%w: raster chroma grid agreement", ErrInvalid)
		}
		format.color = 3
		if chroma.XStep == 2 && chroma.YStep == 2 {
			format.color = 1
		} else if max(chroma.XStep, chroma.YStep) > 1 {
			format.color = 2
		}
	case ColorMultiChannel:
		format.color = 6
	default:
		return format, fmt.Errorf("%w: raster color space", ErrInvalid)
	}
	switch first.Format {
	case SampleUnsigned:
		switch first.Precision {
		case 1:
			if first.Inverted {
				format.depth = 15
			}
		case 8:
			format.depth = 1
		case 16:
			format.depth = 2
		case 5:
			format.depth = 8
			if info.Channels == 3 && info.Components[1].Precision == 6 {
				format.depth = 10
			}
		case 10:
			format.depth = 9
		default:
			return format, fmt.Errorf("%w: raster unsigned precision", ErrInvalid)
		}
	case SampleFixed:
		format.depth, format.number = 3, jxrFixed16
		if first.Precision == 32 {
			format.depth, format.number = 6, jxrFixed32
		}
	case SampleFloat:
		format.depth, format.number = 4, jxrHalf
		if first.Precision == 32 {
			format.depth, format.number = 7, jxrFloat
		}
	case SampleRGBE:
		format.depth, format.number = 1, jxrRGBE
	default:
		return format, fmt.Errorf("%w: raster sample encoding", ErrInvalid)
	}
	if format.number == jxrRGBE && format.color != 8 {
		return format, fmt.Errorf("%w: raster RGBE color space", ErrInvalid)
	}
	if jxrOutputDepth[format.color]&(1<<format.depth) == 0 {
		return format, fmt.Errorf("%w: raster color and precision combination", ErrInvalid)
	}
	for index, component := range info.Components {
		expected := jxrSampleInfo(format, index)
		if !jxrSameSampleEncoding(component, expected) {
			return format, fmt.Errorf("%w: raster component encoding", ErrInvalid)
		}
		if info.ColorSpace != ColorYCC || index == 0 || index == info.Channels {
			if component.XStep != 1 || component.YStep != 1 {
				return format, fmt.Errorf("%w: raster full-rate component", ErrInvalid)
			}
		}
	}
	return format, nil
}

// jxrSharedChromaPhase 检查两个分量能否使用同一JPEG XR码流采样相位字段
// 入参: a, b 待比较的两个色度分量描述
// 返回: bool 相位是否一致
func jxrSharedChromaPhase(a, b ComponentInfo) bool {
	return a.XPositionKnown == b.XPositionKnown && a.YPositionKnown == b.YPositionKnown &&
		(!a.XPositionKnown || a.XOffset == b.XOffset) && (!a.YPositionKnown || a.YOffset == b.YOffset)
}

// jxrSameSampleEncoding 比较不包含空间位置的数值表示
// 入参: a, b 待比较的两个分量描述
// 返回: bool 样本数值编码是否完全一致
func jxrSameSampleEncoding(a, b ComponentInfo) bool {
	return a.Format == b.Format && a.Precision == b.Precision && a.FractionBits == b.FractionBits && a.Inverted == b.Inverted
}

// rasterFormatNameMatches 校验像素格式名称与实际分量信息一致，支持文件及独立码流名称
// 入参: name 像素格式名称, format 实际标准输出格式
// 返回: bool 名称是否匹配
func rasterFormatNameMatches(name string, format jxrPixelFormat) bool {
	if name == codedPixelFormatName(format) {
		return true
	}
	for id := byte(5); id <= 0x56; id++ {
		candidate, ok := jxrLookupPixelFormatID(id, format.depth)
		if !ok || candidate.name != name {
			continue
		}
		return jxrPixelFormatMatches(candidate, format)
	}
	return false
}

// geometryFinite 检查有限数值
// 入参: values 浮点数列表
// 返回: bool 所有数值均非NaN或无穷时为真
func geometryFinite(values ...float64) bool {
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// cloneColorEncoding 复制可选颜色声明，不共享调用方可修改的结构
// 入参: encoding 颜色声明，nil表示未提供
// 返回: *ColorEncoding 独立副本
func cloneColorEncoding(encoding *ColorEncoding) *ColorEncoding {
	if encoding == nil {
		return nil
	}
	value := *encoding
	return &value
}

// checkJXRColorMetadata 校验文件颜色声明的像素格式与矩阵约束
// 入参: format 原生像素格式, hint 文件颜色提示, encoding 可选PTM声明
// 返回: error 颜色元数据错误
func checkJXRColorMetadata(format jxrPixelFormat, hint ColorHint, encoding *ColorEncoding) error {
	if hint != ColorHintNone && hint != ColorHintSRGB && hint != ColorHintUnspecified || hint == ColorHintSRGB && format.number != jxrUnsigned {
		return fmt.Errorf("%w: JPEG XR color space declaration", ErrInvalid)
	}
	if encoding == nil {
		return nil
	}
	if format.color == 7 || format.color == 8 {
		switch encoding.Matrix {
		case 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17:
			return fmt.Errorf("%w: JPEG XR RGB color matrix", ErrInvalid)
		}
	}
	return nil
}

// jxrDisplayColorEncoding 选择标记文件PTM颜色声明，并保留无声明图像的基本显示策略
// 入参: info 原始图像信息, peakNits 可选HDR峰值亮度，零值采用标准标称峰值
// 返回: jxrDisplayMapping 到sRGB的转换, error 当前显示颜色解释不受支持
func jxrDisplayColorEncoding(info Info, peakNits float64) (jxrDisplayMapping, error) {
	if info.ColorEncoding != nil {
		if mapping, supported := jxrCICPDisplay(info, *info.ColorEncoding, peakNits); supported {
			return mapping, nil
		}
		return jxrDisplayMapping{}, fmt.Errorf("%w: display CICP color interpretation", ErrUnsupported)
	}
	if info.ColorHint == ColorHintUnspecified {
		return jxrDisplayMapping{}, fmt.Errorf("%w: unspecified display color space", ErrUnsupported)
	}
	if info.ColorSpace != ColorGray && info.ColorSpace != ColorRGB && info.ColorSpace != ColorCMYK && info.ColorSpace != ColorCMYKDirect {
		return jxrDisplayMapping{}, fmt.Errorf("%w: display color interpretation", ErrUnsupported)
	}
	return jxrDisplayMapping{linear: info.Linear}, nil
}

// SetSampleBits 写入原生位模式，拒绝越界坐标及超过位深的高位，不改变其他分量
// RGBE颜色分量须通过Raster.SetRGBE同时写入，共享指数不能逐分量修改
// 入参: x, y 分量网格坐标, bits 原生样本位模式
// 返回: error 坐标、位深或分量能力错误
func (c Component) SetSampleBits(x, y int, bits uint32) error {
	at, err := c.sampleIndex(x, y)
	if err != nil {
		return wrapJXRError("Component.SetSampleBits", err)
	}
	if c.info.Format == SampleRGBE {
		return &UnsupportedError{Op: "Component.SetSampleBits", Detail: "use Raster.SetRGBE for shared exponent samples"}
	}
	if uint64(bits) >= uint64(1)<<c.info.Precision {
		return &FormatError{Op: "Component.SetSampleBits", Detail: "sample bit width"}
	}
	c.source.storeBits(at, bits)
	return nil
}

// SetSample 写入数值并按目标精度舍入，整数和定点拒绝非有限值及范围越界
// 浮点位模式或NaN载荷需精确保留时应使用SetSampleBits；RGBE须使用Raster.SetRGBE
// 入参: x, y 分量网格坐标, value 分量数值，无符号整数采用单位区间
// 返回: error 坐标、数值范围或分量能力错误
func (c Component) SetSample(x, y int, value float64) error {
	at, err := c.sampleIndex(x, y)
	if err != nil {
		return wrapJXRError("Component.SetSample", err)
	}
	bits, err := sampleValueBits(value, c.info)
	if err != nil {
		return wrapJXRError("Component.SetSample", err)
	}
	c.source.storeBits(at, bits)
	return nil
}

// SetRGBE 同时写入三颜色尾数和共享指数，不修改透明度分量
// 入参: x, y 图像坐标, red, green, blue 八位颜色尾数, exponent 八位共享指数
// 返回: error 图像格式或坐标错误
func (r *Raster) SetRGBE(x, y int, red, green, blue, exponent uint8) error {
	if r == nil || r.info.ColorSpace != ColorRGB || r.info.Channels != 3 {
		return &FormatError{Op: "Raster.SetRGBE", Detail: "RGBE raster required"}
	}
	var components [3]Component
	var positions [3]int
	for index := range components {
		component, err := r.component(index)
		if err != nil || component.info.Format != SampleRGBE {
			return &FormatError{Op: "Raster.SetRGBE", Detail: "RGBE component encoding"}
		}
		position, err := component.sampleIndex(x, y)
		if err != nil {
			return wrapJXRError("Raster.SetRGBE", err)
		}
		components[index], positions[index] = component, position
	}
	for index, mantissa := range [3]uint8{red, green, blue} {
		components[index].source.storeBits(positions[index], uint32(mantissa)|uint32(exponent)<<8)
	}
	return nil
}

// sampleIndex 校验分量坐标并计算对应原生存储索引
// 入参: x, y 分量网格坐标
// 返回: int 连续样本索引, error 视图或坐标错误
func (c Component) sampleIndex(x, y int) (int, error) {
	if c.source == nil {
		return 0, fmt.Errorf("%w: component view is uninitialized", ErrInvalid)
	}
	if !image.Pt(x, y).In(c.source.bounds) {
		return 0, fmt.Errorf("%w: component sample coordinate out of range", ErrInvalid)
	}
	stride := c.source.channels
	if c.source.alpha {
		stride++
	}
	return (y*c.source.bounds.Dx()+x)*stride + c.index, nil
}

// sampleValueBits 按目标数值格式生成样本位模式，采用最近值及中点取偶舍入
// 入参: value 分量数值, info 样本编码信息
// 返回: uint32 原生位模式, error 数值范围或分量能力错误
func sampleValueBits(value float64, info ComponentInfo) (uint32, error) {
	switch info.Format {
	case SampleFloat:
		if info.Precision == 16 {
			return uint32(float64HalfBits(value)), nil
		}
		return math.Float32bits(float32(value)), nil
	case SampleFixed:
		if !geometryFinite(value) {
			return 0, fmt.Errorf("%w: nonfinite signed sample", ErrInvalid)
		}
		value = math.Ldexp(value, int(info.FractionBits))
		low, high := -float64(uint64(1)<<(info.Precision-1)), float64((uint64(1)<<(info.Precision-1))-1)
		if value < low || value > high {
			return 0, fmt.Errorf("%w: signed sample range", ErrInvalid)
		}
		return uint32(int64(math.RoundToEven(value))) & uint32((uint64(1)<<info.Precision)-1), nil
	case SampleUnsigned:
		if !geometryFinite(value) || value < 0 || value > 1 {
			return 0, fmt.Errorf("%w: unsigned sample range", ErrInvalid)
		}
		if info.Inverted {
			value = 1 - value
		}
		return uint32(math.RoundToEven(value * float64((uint64(1)<<info.Precision)-1))), nil
	case SampleRGBE:
		return 0, fmt.Errorf("%w: use Raster.SetRGBE for shared exponent samples", ErrUnsupported)
	default:
		return 0, fmt.Errorf("%w: sample encoding", ErrInvalid)
	}
}

// sampleValue 解释原生位模式，保留扩展数值范围，不进行显示颜色转换
// 入参: bits 样本位模式, info 样本编码信息
// 返回: float64 样本数值，无符号整数归一到单位区间
func sampleValue(bits uint32, info ComponentInfo) float64 {
	switch info.Format {
	case SampleFixed:
		value := int32(bits)
		if info.Precision == 16 {
			value = int32(int16(bits))
		}
		return math.Ldexp(float64(value), -int(info.FractionBits))
	case SampleFloat:
		if info.Precision == 16 {
			return jxrHalfValue(uint16(bits))
		}
		return float64(math.Float32frombits(bits))
	case SampleRGBE:
		if bits>>8 == 0 {
			return 0
		}
		return math.Ldexp(float64(bits&255), int(bits>>8)-136)
	default:
		value := float64(bits) / float64((uint64(1)<<info.Precision)-1)
		if info.Inverted {
			value = 1 - value
		}
		return value
	}
}

// jxrSampleInfo 为标准输出分量确定编码、位深及定点标度
// 入参: format 标准像素格式, index 颜色或透明度通道序号
// 返回: ComponentInfo 数值编码信息，不含采样网格
func jxrSampleInfo(format jxrPixelFormat, index int) ComponentInfo {
	info := ComponentInfo{Format: SampleUnsigned, Precision: 8}
	number := format.number
	if index == format.channels && number == jxrRGBE {
		number = jxrUnsigned
	}
	switch number {
	case jxrFixed16:
		info.Format, info.Precision, info.FractionBits = SampleFixed, 16, 13
	case jxrFixed32:
		info.Format, info.Precision, info.FractionBits = SampleFixed, 32, 24
	case jxrHalf:
		info.Format, info.Precision = SampleFloat, 16
	case jxrFloat:
		info.Format, info.Precision = SampleFloat, 32
	case jxrRGBE:
		info.Format, info.Precision = SampleRGBE, 16
	default:
		switch format.depth {
		case 0, 15:
			info.Precision, info.Inverted = 1, format.depth == 15
		case 2:
			info.Precision = 16
		case 8, 10:
			info.Precision = 5
			if format.depth == 10 && index == 1 {
				info.Precision = 6
			}
		case 9:
			info.Precision = 10
		}
	}
	return info
}

// sampleBytes 返回单个原生样本的实际存储字节数，RGBE分量采用尾数及指数对
// 返回: uint64 样本存储宽度
func (s *jxrRasterSource) sampleBytes() uint64 {
	number := s.container.format.number
	if number == jxrFixed32 || number == jxrFloat {
		return 4
	}
	return 2
}
