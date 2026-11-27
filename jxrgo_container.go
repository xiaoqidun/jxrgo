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
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"math"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"
)

// jxrMissingImageBandTag 与 jxrMissingAlphaBandTag 标记目录中未显式给出主图或透明度频带标签
const (
	jxrMissingImageBandTag = 1 << iota
	jxrMissingAlphaBandTag
)

// jxrReaderAtWork 保存随机读取窗及结构体常驻预算字节数
const jxrReaderAtWork = uint64(4224)

// jxrDefaultProfile 与 jxrDefaultLevel 保存未显式声明档位和级别时的默认值
const (
	jxrDefaultProfile = uint8(111)
	jxrDefaultLevel   = uint8(255)
)

// jxrMaxSubIFDDepth 与 jxrMaxSubIFDNodes 限制EXIF及GPS子目录嵌套深度与总节点数
const (
	jxrMaxSubIFDDepth = 4
	jxrMaxSubIFDNodes = 64
)

// jxrTIFFTypeWidth 保存TIFF字段类型对应的单元素字节宽度
var jxrTIFFTypeWidth = [13]uint64{0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8}

// jxrContainer 保存一个JPEG XR图像目录中的元数据和有界码流，切片或随机视图不混用
type jxrContainer struct {
	width           uint64
	height          uint64
	dpi             Resolution
	format          jxrPixelFormat
	pixelFormat     [16]byte
	transform       uint32
	orientation     Orientation
	image           []byte
	alpha           []byte
	profile         []byte
	exif            []byte
	gps             []byte
	xmp             []byte
	iptc            []byte
	description     *ImageDescription
	profiles        []jxrProfileLevel
	colorHint       ColorHint
	encoding        *ColorEncoding
	imageBands      uint32
	alphaBands      uint32
	missingBandTags uint8
	imageType       ImageType
	uncompressed    bool
	imageRemainder  bool
	imageInput      *jxrReaderAt
	alphaInput      *jxrReaderAt
}

// jxrDirectory 保存首个JPEG XR图像目录及原始字节，不读取后续目录
type jxrDirectory struct {
	data    []byte
	order   binary.ByteOrder
	entries []byte
	next    uint32
	input   *jxrReaderAt
	parser  *jxrParser
}

// jxrOutputField 保存待写入目录字段、内联值、独立数据位置或子目录节点
type jxrOutputField struct {
	tag    uint16
	typ    uint16
	count  uint32
	value  uint32
	data   []byte
	subIFD *jxrSubIFDNode
}

// jxrSubIFDNode 保存已校验的Sub-IFD原始条目表、外部负载偏移与递归嵌套子目录
type jxrSubIFDNode struct {
	table      []byte
	count      uint64
	entries    []jxrSubIFDEntry
	totalBytes uint64
}

// jxrSubIFDEntry 保存单条Sub-IFD外部负载或嵌套子目录的来源与布局信息
type jxrSubIFDEntry struct {
	index      int
	typ        uint16
	srcOffset  uint64
	size       uint64
	subIFD     bool
	swapEndian bool
	children   []*jxrSubIFDNode
}

// jxrSubIFDOrder 保存重排Sub-IFD目录项时的标签值与原槽位序号
type jxrSubIFDOrder struct {
	tag   uint16
	index int
}

// jxrSubIFDScan 保存递归扫描Sub-IFD时的边界、深度、节点计数与防环路状态
type jxrSubIFDScan struct {
	d            *jxrDirectory
	raw          []byte
	order        binary.ByteOrder
	parser       *jxrParser
	size         uint64
	minOffset    uint64
	allowInterop bool
	strictNext   bool
	nodes        int
	totalBytes   uint64
	tempWork     uint64
	ancestors    []uint64
}

// jxrContainerOffsets 保存实际文件前缀中主图长度及独立透明度地址和长度的回填位置
type jxrContainerOffsets struct {
	imageSize   uint64
	alphaOffset uint64
	alphaSize   uint64
}

// jxrReaderAt 持有一个有界输入视图及固定读取窗，不持有完整压缩文件
type jxrReaderAt struct {
	source io.ReaderAt
	ctx    context.Context
	base   uint64
	size   uint64
	start  uint64
	count  int
	buffer [4096]byte
}

// jxrLevel 保存级别及尺寸、分块和缓冲上限
type jxrLevel struct {
	level                              uint8
	dimension, tiles, tileSize, buffer uint64
}

// separateAlpha 判断目录是否声明独立透明度，兼容切片和随机码流视图
// 返回: bool 是否存在独立透明度码流
func (c *jxrContainer) separateAlpha() bool {
	return c.alpha != nil || c.alphaInput != nil
}

// readJXRDirectory 读取小端JPEG XR目录，检查顺序并忽略后续图像目录
// 入参: data 完整文件字节
// 返回: *jxrDirectory 只读图像目录, error 目录或资源错误
func readJXRDirectory(data []byte) (*jxrDirectory, error) {
	if len(data) < 8 || string(data[:3]) != "II\xbc" {
		return nil, fmt.Errorf("%w: JPEG XR file header", ErrInvalid)
	}
	if data[3] != 1 {
		return nil, fmt.Errorf("%w: JPEG XR file version", ErrUnsupported)
	}
	if uint64(len(data)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: JPEG XR file size", ErrLimit)
	}
	offset := uint64(binary.LittleEndian.Uint32(data[4:]))
	d, err := readJXRDirectoryAt(data, offset)
	if err != nil {
		return nil, err
	}
	return &d, nil
}

// readJXRDirectoryAt 按完整文件中的绝对偏移校验目录，后续链接只取值不递归解析
// 入参: data 完整文件, offset 已知目录的绝对位置
// 返回: jxrDirectory 原始字节视图, error 目录或资源错误
func readJXRDirectoryAt(data []byte, offset uint64) (jxrDirectory, error) {
	if offset < 8 || offset&1 != 0 || offset > uint64(len(data)) || uint64(len(data))-offset < 2 {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory offset", ErrInvalid)
	}
	count := uint64(binary.LittleEndian.Uint16(data[offset:]))
	if count == 0 || count*12+6 > uint64(len(data))-offset {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory size", ErrInvalid)
	}
	d := jxrDirectory{data: data, order: binary.LittleEndian, entries: data[offset+2 : offset+2+count*12], next: binary.LittleEndian.Uint32(data[offset+2+count*12:])}
	if err := validateJXRDirectoryEntries(d.entries, d.next); err != nil {
		return jxrDirectory{}, err
	}
	return d, nil
}

// validateJXRDirectoryEntries 校验目录项标签严格递增顺序及后续目录偶数对齐
// 入参: entries 十二字节步长的目录项字节切片, next 下一目录偏移
// 返回: error 标签顺序或对齐错误
func validateJXRDirectoryEntries(entries []byte, next uint32) error {
	previous := -1
	for at := 0; at < len(entries); at += 12 {
		tag := int(binary.LittleEndian.Uint16(entries[at:]))
		if tag <= previous {
			return fmt.Errorf("%w: JPEG XR directory tag order", ErrInvalid)
		}
		previous = tag
	}
	if next&1 != 0 {
		return fmt.Errorf("%w: JPEG XR next directory alignment", ErrInvalid)
	}
	return nil
}

// jxrField 按标签允许的类型及数量取得数据，未知组合按JPEG XR要求忽略
// 入参: tag 字段标识, count 所需数量，零表示任意数量, limit 字节上限, types 允许的字段类型
// 返回: []byte 字段视图，缺省或组合未知时为nil, error 偏移或资源错误
func (d *jxrDirectory) jxrField(tag uint16, count uint32, limit uint64, types ...uint16) ([]byte, error) {
	entry := d.entryFor(tag)
	if entry == nil {
		return nil, nil
	}
	typ := d.order.Uint16(entry[2:])
	allowed := false
	for _, candidate := range types {
		allowed = allowed || typ == candidate
	}
	if !allowed || typ > 12 || typ == 0 {
		return nil, nil
	}
	n := d.order.Uint32(entry[4:])
	if count != 0 && n != count {
		return nil, nil
	}
	width := jxrTIFFTypeWidth[typ]
	size := uint64(n) * width
	if size > limit {
		return nil, &LimitError{Resource: fmt.Sprintf("JPEG XR field %x budget", tag), Limit: limit, Required: size}
	}
	if size <= 4 {
		return entry[8 : 8+size], nil
	}
	offset := uint64(d.order.Uint32(entry[8:]))
	if offset&1 != 0 || offset > d.dataSize() || size > d.dataSize()-offset {
		return nil, fmt.Errorf("%w: JPEG XR field %x offset", ErrInvalid, tag)
	}
	if d.input != nil {
		if size > uint64(^uint(0)>>1) {
			return nil, fmt.Errorf("%w: JPEG XR field %x address capacity", ErrLimit, tag)
		}
		if err := d.parser.reserve(size); err != nil {
			return nil, err
		}
		data := make([]byte, int(size))
		if err := d.input.copy(offset, data); err != nil {
			return nil, err
		}
		return data, nil
	}
	return d.data[offset : offset+size], nil
}

// jxrNumber 读取允许为BYTE、USHORT或ULONG的标量字段，不采用错误组合中的数值
// 入参: tag 字段标识, required 是否为必要字段, fallback 缺省值
// 返回: uint32 无符号值, error 字段错误
func (d *jxrDirectory) jxrNumber(tag uint16, required bool, fallback uint32) (uint32, error) {
	data, err := d.jxrField(tag, 1, 4, 1, 3, 4)
	if err != nil {
		return 0, err
	}
	if data == nil {
		if required {
			return 0, fmt.Errorf("%w: missing JPEG XR field %x", ErrInvalid, tag)
		}
		return fallback, nil
	}
	switch len(data) {
	case 1:
		return uint32(data[0]), nil
	case 2:
		return uint32(d.order.Uint16(data)), nil
	default:
		return d.order.Uint32(data), nil
	}
}

// jxrStreamMin 按给定最小长度检查码流或无压缩像素数据的文件范围
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许零长度延伸, minSize 最小字节数
// 返回: []byte 数据视图, bool 是否由零长度延伸至文件末尾, error 范围或资源错误
func (d *jxrDirectory) jxrStreamMin(offsetTag, countTag uint16, remainder bool, minSize uint64) ([]byte, bool, error) {
	offset, size, usedRemainder, err := d.jxrStreamRangeMin(offsetTag, countTag, remainder, minSize)
	if err != nil {
		return nil, false, err
	}
	return d.data[offset : offset+size], usedRemainder, nil
}

// jxrStreamRangeMin 按给定最小长度校验完整文件中的码流或无压缩数据范围
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许零长度延伸, minSize 最小字节数
// 返回: uint64 全文件偏移, uint64 数据长度, bool 是否零长度延伸, error 范围或资源错误
func (d *jxrDirectory) jxrStreamRangeMin(offsetTag, countTag uint16, remainder bool, minSize uint64) (uint64, uint64, bool, error) {
	offset, err := d.jxrNumber(offsetTag, true, 0)
	if err != nil {
		return 0, 0, false, err
	}
	count, err := d.jxrNumber(countTag, true, 0)
	if err != nil {
		return 0, 0, false, err
	}
	if uint64(offset) > d.dataSize() {
		return 0, 0, false, fmt.Errorf("%w: JPEG XR codestream offset", ErrInvalid)
	}
	size, usedRemainder := uint64(count), false
	if count == 0 && remainder {
		size, usedRemainder = d.dataSize()-uint64(offset), true
	}
	if size < minSize || size > d.dataSize()-uint64(offset) {
		return 0, 0, false, fmt.Errorf("%w: JPEG XR codestream size", ErrInvalid)
	}
	return uint64(offset), size, usedRemainder, nil
}

// jxrStreamViewMin 取得零复制切片或独立有界随机视图，分配固定窗前检查共用预算
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许延伸至文件末尾, minSize 最小字节数
// 返回: []byte 可选切片, *jxrReaderAt 可选随机视图, bool 是否零长度延伸, error 范围、资源或读取错误
func (d *jxrDirectory) jxrStreamViewMin(offsetTag, countTag uint16, remainder bool, minSize uint64) ([]byte, *jxrReaderAt, bool, error) {
	if d.input == nil {
		data, usedRemainder, err := d.jxrStreamMin(offsetTag, countTag, remainder, minSize)
		return data, nil, usedRemainder, err
	}
	offset, size, usedRemainder, err := d.jxrStreamRangeMin(offsetTag, countTag, remainder, minSize)
	if err != nil {
		return nil, nil, false, err
	}
	if err := d.parser.reserve(jxrReaderAtWork); err != nil {
		return nil, nil, false, err
	}
	return nil, &jxrReaderAt{source: d.input.source, ctx: d.input.ctx, base: d.input.base + offset, size: size}, usedRemainder, nil
}

// container 在共享解析预算内读取首个图像的元数据及码流范围，常规切片输入下不复制码流字节，Sub-IFD与描述标签独立解析
// 入参: data 完整文件字节, maxPixels 剩余像素预算
// 返回: jxrContainer 图像容器, error 数据、能力或资源错误
func (parser *jxrParser) container(data []byte, maxPixels uint64) (jxrContainer, error) {
	d, err := readJXRDirectory(data)
	if err != nil {
		return jxrContainer{}, err
	}
	return parser.directoryContainer(d, maxPixels)
}

// directoryContainer 在共享预算内读取已定位目录，所有资源偏移仍相对于完整文件
// 入参: d 已校验的原始目录视图, maxPixels 图像像素上限
// 返回: jxrContainer 当前图像容器, error 元数据、数据或资源错误
func (parser *jxrParser) directoryContainer(d *jxrDirectory, maxPixels uint64) (jxrContainer, error) {
	c, err := parser.directoryFields(d, maxPixels)
	if err != nil {
		return c, err
	}
	if c.uncompressed {
		return c, nil
	}
	return c, c.checkProfile(parser, maxPixels)
}

// directoryFields 读取容器目录标签与码流位置，供已解析码流的入口共用
// 入参: d 已校验原始目录视图, maxPixels 图像像素上限
// 返回: jxrContainer 解析得到的容器元数据, error 标签、格式或资源错误
func (parser *jxrParser) directoryFields(d *jxrDirectory, maxPixels uint64) (jxrContainer, error) {
	c := jxrContainer{dpi: Resolution{96, 96}, missingBandTags: jxrMissingImageBandTag | jxrMissingAlphaBandTag}
	d.parser = parser
	if err := parser.reserve(512); err != nil {
		return c, err
	}
	pixelFormat, err := d.jxrField(0xbc01, 16, 16, 1)
	if err != nil {
		return c, err
	}
	if pixelFormat == nil {
		return c, fmt.Errorf("%w: missing JPEG XR pixel format", ErrInvalid)
	}
	c.format, err = readJXRPixelFormat(pixelFormat)
	if err != nil {
		return c, err
	}
	copy(c.pixelFormat[:], pixelFormat)
	width, err := d.jxrNumber(0xbc80, true, 0)
	if err != nil {
		return c, err
	}
	height, err := d.jxrNumber(0xbc81, true, 0)
	if err != nil {
		return c, err
	}
	c.width, c.height = uint64(width), uint64(height)
	if err := checkJXRDimensions(c.width, c.height, maxPixels); err != nil {
		return c, err
	}
	for i, tag := range []uint16{0xbc82, 0xbc83} {
		field, e := d.jxrField(tag, 1, 4, 11)
		if e != nil {
			return c, e
		}
		if field != nil {
			value := float64(math.Float32frombits(d.order.Uint32(field)))
			if value > 0 && geometryFinite(value) {
				if i == 0 {
					c.dpi.X = value
				} else {
					c.dpi.Y = value
				}
			}
		}
	}
	c.transform, err = d.jxrNumber(0xbc02, false, 0)
	if err != nil {
		return c, err
	}
	if c.transform > 7 {
		c.transform = 0
	}
	c.orientation = Orientation(c.transform)
	uncompressed, err := d.jxrNumber(0xbc03, false, 0)
	if err != nil {
		return c, err
	}
	if uncompressed > 1 {
		return c, fmt.Errorf("%w: JPEG XR uncompressed mode value", ErrInvalid)
	}
	c.uncompressed = uncompressed == 1
	imageType, err := d.jxrField(0xbc04, 1, 4, 4)
	if err != nil {
		return c, err
	}
	if imageType != nil {
		c.imageType = ImageType(d.order.Uint32(imageType) & 3)
	}
	minStreamSize := uint64(8)
	if c.uncompressed {
		minStreamSize = 1
	}
	c.image, c.imageInput, c.imageRemainder, err = d.jxrStreamViewMin(0xbcc0, 0xbcc1, true, minStreamSize)
	if err != nil {
		return c, err
	}
	alphaOffset, err := d.jxrField(0xbcc2, 1, 4, 1, 3, 4)
	if err != nil {
		return c, err
	}
	alphaCount, err := d.jxrField(0xbcc3, 1, 4, 1, 3, 4)
	if err != nil {
		return c, err
	}
	if (alphaOffset != nil) != (alphaCount != nil) || alphaOffset != nil && !c.format.alpha {
		return c, fmt.Errorf("%w: JPEG XR separate alpha fields", ErrInvalid)
	}
	if alphaOffset != nil {
		c.alpha, c.alphaInput, _, err = d.jxrStreamViewMin(0xbcc2, 0xbcc3, false, minStreamSize)
		if err != nil {
			return c, err
		}
	}
	for i, tag := range []uint16{0xbcc4, 0xbcc5} {
		field, e := d.jxrField(tag, 1, 1, 1)
		if e != nil {
			return c, e
		}
		if field != nil {
			if c.uncompressed || field[0] > 3 || i == 1 && !c.separateAlpha() {
				return c, fmt.Errorf("%w: JPEG XR band presence", ErrInvalid)
			}
			if i == 0 {
				c.imageBands = uint32(field[0])
				c.missingBandTags &^= jxrMissingImageBandTag
			} else {
				c.alphaBands = uint32(field[0])
				c.missingBandTags &^= jxrMissingAlphaBandTag
			}
		}
	}
	c.description, err = jxrReadDescription(d, parser)
	if err != nil {
		return c, err
	}
	c.xmp, err = jxrReadContainerBlob(d, parser, 0x02bc, 1, 7)
	if err != nil {
		return c, err
	}
	c.iptc, err = jxrReadContainerBlob(d, parser, 0x83bb, 1, 7, 4)
	if err != nil {
		return c, err
	}
	if entry := d.entryFor(0x8769); entry != nil {
		typ, count := d.order.Uint16(entry[2:]), d.order.Uint32(entry[4:])
		if typ == 4 && count == 1 {
			c.exif, err = jxrReadSubIFD(d, parser, uint64(d.order.Uint32(entry[8:])), true)
			if err != nil {
				return c, err
			}
		}
	}
	if entry := d.entryFor(0x8825); entry != nil {
		typ, count := d.order.Uint16(entry[2:]), d.order.Uint32(entry[4:])
		if typ == 4 && count == 1 {
			c.gps, err = jxrReadSubIFD(d, parser, uint64(d.order.Uint32(entry[8:])), false)
			if err != nil {
				return c, err
			}
		}
	}
	c.profile, err = jxrReadContainerBlob(d, parser, 0x8773, 7)
	if err != nil {
		return c, err
	}
	colorSpace, err := d.jxrField(0xa001, 1, 2, 3)
	if err != nil {
		return c, err
	}
	if colorSpace != nil {
		switch d.order.Uint16(colorSpace) {
		case uint16(ColorHintSRGB):
			c.colorHint = ColorHintSRGB
		case uint16(ColorHintUnspecified):
			c.colorHint = ColorHintUnspecified
		default:
			c.colorHint = ColorHintNone
		}
	}
	encoding, err := d.jxrField(0xbc05, 4, 4, 1)
	if err != nil {
		return c, err
	}
	if encoding != nil {
		if err := parser.reserve(4); err != nil {
			return c, err
		}
		c.encoding = &ColorEncoding{Primaries: encoding[0], Transfer: encoding[1], Matrix: encoding[2], FullRange: encoding[3]&1 != 0}
	}
	if err := checkJXRColorMetadata(c.format, c.colorHint, c.encoding); err != nil {
		return c, err
	}
	declarations, err := d.jxrField(0xbc06, 0, math.MaxUint32, 1)
	if err != nil {
		return c, err
	}
	if declarations != nil {
		if c.uncompressed {
			return c, fmt.Errorf("%w: JPEG XR profile declarations in uncompressed mode", ErrInvalid)
		}
		b := jxrBits{data: declarations, parser: parser}
		c.profiles, err = parser.profiles(&b, uint64(len(declarations)))
		if err != nil {
			return c, err
		}
	}
	return c, nil
}

// jxrReadContainerBlob 读取可选可变长容器元数据并在随机输入内联值时独立计费拷贝
// 入参: d 已校验目录, parser 共享解析预算, tag 字段标识, types 允许的TIFF字段类型
// 返回: []byte 元数据字节切片，缺省时为nil, error 数据、偏移或资源错误
func jxrReadContainerBlob(d *jxrDirectory, parser *jxrParser, tag uint16, types ...uint16) ([]byte, error) {
	blob, err := d.jxrField(tag, 0, math.MaxUint32, types...)
	if err != nil {
		return nil, err
	}
	if blob != nil && len(blob) == 0 {
		return nil, fmt.Errorf("%w: empty JPEG XR metadata field %x", ErrInvalid, tag)
	}
	if d.input != nil && len(blob) > 0 && len(blob) <= 4 {
		if err := parser.reserve(uint64(len(blob))); err != nil {
			return nil, err
		}
		copied := make([]byte, len(blob))
		copy(copied, blob)
		return copied, nil
	}
	return blob, nil
}

// jxrExifTypeWidth 返回EXIF或GPS子目录中TIFF及Exif3字段类型的单元素字节宽度
// 入参: typ TIFF字段类型编号
// 返回: uint64 字节宽度，未知或不支持类型返回零
func jxrExifTypeWidth(typ uint16) uint64 {
	if typ >= 1 && typ <= 12 {
		return jxrTIFFTypeWidth[typ]
	}
	if typ == 13 {
		return 4
	}
	if typ == 129 {
		return 1
	}
	return 0
}

// jxrIsNestedSubIFD 判断子目录项是否指向递归嵌套的子IFD
// 入参: tag 字段标签, typ 字段类型, count 元素数量, allowInterop 是否识别0xA005互操作IFD指针
// 返回: bool 是否为嵌套子IFD指针项
func jxrIsNestedSubIFD(tag, typ uint16, count uint64, allowInterop bool) bool {
	if typ == 13 && count > 0 {
		return true
	}
	return allowInterop && tag == 0xa005 && typ == 4 && count == 1
}

// jxrSwapTIFFEndian 按TIFF字段类型原地翻转多字节元素字节序，单字节类型保持不变
// 入参: data 待转换元素字节切片, typ TIFF字段类型编号
func jxrSwapTIFFEndian(data []byte, typ uint16) {
	switch typ {
	case 3, 8:
		for i := 0; i+2 <= len(data); i += 2 {
			data[i], data[i+1] = data[i+1], data[i]
		}
	case 4, 5, 9, 10, 11, 13:
		for i := 0; i+4 <= len(data); i += 4 {
			data[i], data[i+1], data[i+2], data[i+3] = data[i+3], data[i+2], data[i+1], data[i]
		}
	case 12:
		for i := 0; i+8 <= len(data); i += 8 {
			data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7] = data[i+7], data[i+6], data[i+5], data[i+4], data[i+3], data[i+2], data[i+1], data[i]
		}
	}
}

// jxrReadSubIFD 将容器内由0x8769或0x8825偏移指向的Sub-IFD及其嵌套子目录重定位为自包含小端TIFF字节块
// 入参: d 所属目录, parser 共享解析预算, offset 子目录在文件中的绝对偏移, allowInterop 是否识别EXIF互操作子目录
// 返回: []byte 自包含小端TIFF字节块, error 偏移、格式、嵌套、读取或资源错误
func jxrReadSubIFD(d *jxrDirectory, parser *jxrParser, offset uint64, allowInterop bool) ([]byte, error) {
	state := jxrSubIFDScan{d: d, order: binary.LittleEndian, parser: parser, size: d.dataSize(), minOffset: 8, allowInterop: allowInterop}
	defer func() {
		if state.tempWork != 0 {
			parser.used -= state.tempWork
		}
	}()
	root, err := state.scan(offset, 0)
	if err != nil {
		return nil, err
	}
	if root.totalBytes > math.MaxUint32-8 {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD payload capacity", ErrLimit)
	}
	total := uint64(8) + root.totalBytes
	if total > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD address capacity", ErrLimit)
	}
	if err := parser.reserve(total); err != nil {
		return nil, err
	}
	out := make([]byte, int(total))
	copy(out, "II\x2a\x00")
	binary.LittleEndian.PutUint32(out[4:], 8)
	if _, err := jxrEmitSubIFD(out, 8, root, func(srcOffset uint64, dst []byte) error {
		if d.input != nil {
			d.input.ctx = parser.ctx
			return d.input.copy(srcOffset, dst)
		}
		copy(dst, d.data[srcOffset:srcOffset+uint64(len(dst))])
		return nil
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// jxrParseSubIFDBytes 校验待写入的EXIF或GPS字节块并解析其递归子目录树，支持小端与大端TIFF及纯小端Sub-IFD
// 入参: parser 共享解析预算与取消状态, raw 原始EXIF或GPS字节, allowInterop 是否识别EXIF互操作子目录
// 返回: *jxrSubIFDNode 已校验的子目录树, error 格式、偏移、嵌套、取消或容量错误
func jxrParseSubIFDBytes(parser *jxrParser, raw []byte, allowInterop bool) (*jxrSubIFDNode, error) {
	if err := parser.check(); err != nil {
		return nil, err
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD bytes", ErrInvalid)
	}
	order := binary.ByteOrder(binary.LittleEndian)
	rootOffset, minOffset := uint64(0), uint64(0)
	if string(raw[:4]) == "MM\x00\x2a" {
		order, rootOffset, minOffset = binary.BigEndian, uint64(binary.BigEndian.Uint32(raw[4:8])), 8
	} else if string(raw[:4]) == "II\x2a\x00" {
		rootOffset, minOffset = uint64(binary.LittleEndian.Uint32(raw[4:8])), 8
	}
	state := jxrSubIFDScan{raw: raw, order: order, parser: parser, size: uint64(len(raw)), minOffset: minOffset, allowInterop: allowInterop, strictNext: true}
	defer func() {
		if state.tempWork != 0 {
			parser.used -= state.tempWork
		}
	}()
	return state.scan(rootOffset, 0)
}

// scan 递归扫描单个Sub-IFD及其所有外部值和嵌套子目录，执行边界、环路、深度与总容量检查
// 入参: offset 当前Sub-IFD相对源起点的偏移, depth 当前递归深度
// 返回: *jxrSubIFDNode 已校验子目录节点, error 偏移、类型、环路、读取或资源错误
func (s *jxrSubIFDScan) scan(offset uint64, depth int) (*jxrSubIFDNode, error) {
	if err := s.parser.check(); err != nil {
		return nil, err
	}
	if depth > jxrMaxSubIFDDepth {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD nesting depth", ErrLimit)
	}
	if s.nodes >= jxrMaxSubIFDNodes {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD node count", ErrLimit)
	}
	if offset < s.minOffset || offset&1 != 0 || offset > s.size || s.size-offset < 2 {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD offset", ErrInvalid)
	}
	if slices.Contains(s.ancestors, offset) {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD cycle", ErrInvalid)
	}
	swap := s.order == binary.BigEndian
	s.nodes++
	s.ancestors = append(s.ancestors, offset)
	defer func() { s.ancestors = s.ancestors[:len(s.ancestors)-1] }()
	var countBytes [2]byte
	if err := s.readBytes(offset, countBytes[:]); err != nil {
		return nil, err
	}
	count := uint64(s.order.Uint16(countBytes[:]))
	tableLen := count*12 + 6
	if count == 0 || tableLen > s.size-offset {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD size", ErrInvalid)
	}
	table, err := s.sliceBytes(offset, tableLen)
	if err != nil {
		return nil, err
	}
	if s.strictNext && s.order.Uint32(table[2+count*12:]) != 0 {
		return nil, fmt.Errorf("%w: JPEG XR Sub-IFD next offset", ErrInvalid)
	}
	if err := s.addTotal(tableLen); err != nil {
		return nil, err
	}
	outTable := table
	if swap {
		if err := s.reserveTemp(tableLen); err != nil {
			return nil, err
		}
		outTable = make([]byte, len(table))
		binary.LittleEndian.PutUint16(outTable[0:2], uint16(count))
	}
	node := &jxrSubIFDNode{table: outTable, count: count, totalBytes: tableLen}
	prevTag, ordered := -1, true
	for i := range count {
		if err := s.parser.check(); err != nil {
			return nil, err
		}
		entry := table[2+i*12 : 2+(i+1)*12]
		tag, typ, n := s.order.Uint16(entry), s.order.Uint16(entry[2:]), uint64(s.order.Uint32(entry[4:]))
		if int(tag) <= prevTag {
			ordered = false
		}
		prevTag = int(tag)
		width := jxrExifTypeWidth(typ)
		if width == 0 {
			return nil, fmt.Errorf("%w: JPEG XR Sub-IFD field type", ErrInvalid)
		}
		if n > math.MaxUint32/width {
			return nil, fmt.Errorf("%w: JPEG XR Sub-IFD field length", ErrInvalid)
		}
		fieldBytes := n * width
		nested := jxrIsNestedSubIFD(tag, typ, n, s.allowInterop)
		if swap {
			outEntry := outTable[2+i*12 : 2+(i+1)*12]
			binary.LittleEndian.PutUint16(outEntry[0:2], tag)
			binary.LittleEndian.PutUint16(outEntry[2:4], typ)
			binary.LittleEndian.PutUint32(outEntry[4:8], uint32(n))
			if !nested && fieldBytes <= 4 {
				copy(outEntry[8:8+fieldBytes], entry[8:8+fieldBytes])
				jxrSwapTIFFEndian(outEntry[8:8+fieldBytes], typ)
			}
		}
		if nested {
			if n == 1 {
				childOffset := uint64(s.order.Uint32(entry[8:]))
				child, err := s.scan(childOffset, depth+1)
				if err != nil {
					return nil, err
				}
				node.totalBytes += child.totalBytes
				node.entries = append(node.entries, jxrSubIFDEntry{index: int(i), subIFD: true, children: []*jxrSubIFDNode{child}})
			} else {
				if n > uint64(jxrMaxSubIFDNodes-s.nodes) {
					return nil, fmt.Errorf("%w: JPEG XR Sub-IFD node count", ErrLimit)
				}
				arrayOffset := uint64(s.order.Uint32(entry[8:]))
				if arrayOffset < s.minOffset || arrayOffset > s.size || fieldBytes > s.size-arrayOffset {
					return nil, fmt.Errorf("%w: JPEG XR Sub-IFD pointer array offset", ErrInvalid)
				}
				if err := s.addTotal(fieldBytes); err != nil {
					return nil, err
				}
				node.totalBytes += fieldBytes
				rawOffsets, err := s.sliceBytes(arrayOffset, fieldBytes)
				if err != nil {
					return nil, err
				}
				children := make([]*jxrSubIFDNode, int(n))
				for ci := range int(n) {
					childOffset := uint64(s.order.Uint32(rawOffsets[ci*4:]))
					child, err := s.scan(childOffset, depth+1)
					if err != nil {
						return nil, err
					}
					node.totalBytes += child.totalBytes
					children[ci] = child
				}
				node.entries = append(node.entries, jxrSubIFDEntry{index: int(i), subIFD: true, children: children})
			}
		} else if fieldBytes > 4 {
			fieldOffset := uint64(s.order.Uint32(entry[8:]))
			if fieldOffset < s.minOffset || fieldOffset > s.size || fieldBytes > s.size-fieldOffset {
				return nil, fmt.Errorf("%w: JPEG XR Sub-IFD field offset", ErrInvalid)
			}
			padded := (fieldBytes + 1) &^ 1
			if err := s.addTotal(padded); err != nil {
				return nil, err
			}
			node.totalBytes += padded
			node.entries = append(node.entries, jxrSubIFDEntry{index: int(i), typ: typ, srcOffset: fieldOffset, size: fieldBytes, swapEndian: swap})
		}
	}
	if !ordered {
		if err := s.reserveTemp(count*24 + tableLen); err != nil {
			return nil, err
		}
		orderList := make([]jxrSubIFDOrder, int(count))
		for i := range int(count) {
			if i&1023 == 0 {
				if err := s.parser.check(); err != nil {
					return nil, err
				}
			}
			orderList[i] = jxrSubIFDOrder{tag: binary.LittleEndian.Uint16(outTable[2+i*12:]), index: i}
		}
		slices.SortFunc(orderList, func(a, b jxrSubIFDOrder) int {
			return cmp.Compare(a.tag, b.tag)
		})
		for i := 1; i < len(orderList); i++ {
			if i&1023 == 0 {
				if err := s.parser.check(); err != nil {
					return nil, err
				}
			}
			if orderList[i].tag == orderList[i-1].tag {
				return nil, fmt.Errorf("%w: JPEG XR Sub-IFD duplicate tag", ErrInvalid)
			}
		}
		newIndex := make([]int, int(count))
		sortedTable := make([]byte, len(outTable))
		binary.LittleEndian.PutUint16(sortedTable[0:2], uint16(count))
		for newPos, item := range orderList {
			if newPos&1023 == 0 {
				if err := s.parser.check(); err != nil {
					return nil, err
				}
			}
			newIndex[item.index] = newPos
			copy(sortedTable[2+newPos*12:2+(newPos+1)*12], outTable[2+item.index*12:2+(item.index+1)*12])
		}
		for ei := range node.entries {
			node.entries[ei].index = newIndex[node.entries[ei].index]
		}
		slices.SortFunc(node.entries, func(a, b jxrSubIFDEntry) int {
			return cmp.Compare(a.index, b.index)
		})
		node.table = sortedTable
	}
	return node, nil
}

// sliceBytes 按指定偏移与长度取得当前数据源的字节视图，随机输入按临时预算分配读取
// 入参: offset 相对源起点的偏移, length 需读取的字节长度
// 返回: []byte 数据视图或临时拷贝, error 预算、读取或取消错误
func (s *jxrSubIFDScan) sliceBytes(offset, length uint64) ([]byte, error) {
	if s.d != nil && s.d.input != nil {
		if err := s.reserveTemp(length); err != nil {
			return nil, err
		}
		buf := make([]byte, int(length))
		if err := s.readBytes(offset, buf); err != nil {
			return nil, err
		}
		return buf, nil
	}
	if s.d != nil {
		return s.d.data[offset : offset+length], nil
	}
	return s.raw[offset : offset+length], nil
}

// readBytes 从当前数据源的指定偏移读取字节到目标切片
// 入参: offset 相对源起点的偏移, dst 目标切片
// 返回: error 范围、读取或取消错误
func (s *jxrSubIFDScan) readBytes(offset uint64, dst []byte) error {
	if s.d != nil && s.d.input != nil {
		s.d.input.ctx = s.parser.ctx
		return s.d.input.copy(offset, dst)
	}
	if s.d != nil {
		copy(dst, s.d.data[offset:offset+uint64(len(dst))])
		return nil
	}
	copy(dst, s.raw[offset:offset+uint64(len(dst))])
	return nil
}

// reserveTemp 为扫描期临时切片预留解析内存并记录以便结束后归还
// 入参: amount 新增字节数
// 返回: error 资源限制错误
func (s *jxrSubIFDScan) reserveTemp(amount uint64) error {
	if err := s.parser.reserve(amount); err != nil {
		return err
	}
	s.tempWork += amount
	return nil
}

// addTotal 累加整棵子目录树重定位后的总字节数并检查32位偏移容量上限
// 入参: amount 新增字节数
// 返回: error 超出容量上限时返回资源错误
func (s *jxrSubIFDScan) addTotal(amount uint64) error {
	if amount > math.MaxUint32 || s.totalBytes > math.MaxUint32-amount {
		return fmt.Errorf("%w: JPEG XR Sub-IFD payload capacity", ErrLimit)
	}
	s.totalBytes += amount
	return nil
}

// jxrEmitSubIFD 将已校验的Sub-IFD树序列化至目标缓冲的start位置并重定位所有外部与嵌套偏移
// 入参: dst 已清零且容量充足的目标缓冲, start 当前节点写入起点, node 子目录节点, copySrc 源数据拷贝回调
// 返回: uint64 写入结束位置, error 拷贝或取消错误
func jxrEmitSubIFD(dst []byte, start uint64, node *jxrSubIFDNode, copySrc func(uint64, []byte) error) (uint64, error) {
	tableLen := node.count*12 + 6
	copy(dst[start:start+2+node.count*12], node.table[:2+node.count*12])
	binary.LittleEndian.PutUint32(dst[start+2+node.count*12:], 0)
	cursor := start + tableLen
	for _, item := range node.entries {
		entry := dst[start+2+uint64(item.index)*12 : start+2+uint64(item.index+1)*12]
		if !item.subIFD {
			binary.LittleEndian.PutUint32(entry[8:], uint32(cursor))
			if err := copySrc(item.srcOffset, dst[cursor:cursor+item.size]); err != nil {
				return 0, err
			}
			if item.swapEndian {
				jxrSwapTIFFEndian(dst[cursor:cursor+item.size], item.typ)
			}
			cursor += (item.size + 1) &^ 1
		} else if len(item.children) == 1 {
			binary.LittleEndian.PutUint32(entry[8:], uint32(cursor))
			next, err := jxrEmitSubIFD(dst, cursor, item.children[0], copySrc)
			if err != nil {
				return 0, err
			}
			cursor = next
		} else {
			arrayPos := cursor
			binary.LittleEndian.PutUint32(entry[8:], uint32(arrayPos))
			cursor += uint64(len(item.children)) * 4
			for ci, child := range item.children {
				binary.LittleEndian.PutUint32(dst[arrayPos+uint64(ci)*4:], uint32(cursor))
				next, err := jxrEmitSubIFD(dst, cursor, child, copySrc)
				if err != nil {
					return 0, err
				}
				cursor = next
			}
		}
	}
	return cursor, nil
}

// relocateJXRSubIFD 在多图容器重定位期间递归平移Sub-IFD及其嵌套子目录内全部外部偏移
// 入参: data 单图独占前缀缓冲, offset 当前Sub-IFD在缓冲内的本地偏移, delta 全文件偏移增量, allowInterop 是否识别0xA005, depth 当前嵌套深度
// 返回: error 结构、深度或容量错误
func relocateJXRSubIFD(data []byte, offset, delta uint64, allowInterop bool, depth int) error {
	if depth > jxrMaxSubIFDDepth {
		return fmt.Errorf("%w: JPEG XR relocated Sub-IFD depth", ErrLimit)
	}
	if offset < 8 || offset&1 != 0 || offset > uint64(len(data)) || uint64(len(data))-offset < 2 {
		return fmt.Errorf("%w: JPEG XR relocated Sub-IFD offset", ErrInvalid)
	}
	count := uint64(binary.LittleEndian.Uint16(data[offset:]))
	tableLen := count*12 + 6
	if count == 0 || tableLen > uint64(len(data))-offset {
		return fmt.Errorf("%w: JPEG XR relocated Sub-IFD size", ErrInvalid)
	}
	for i := range count {
		entry := data[offset+2+i*12 : offset+2+(i+1)*12]
		tag, typ, n := binary.LittleEndian.Uint16(entry), binary.LittleEndian.Uint16(entry[2:]), uint64(binary.LittleEndian.Uint32(entry[4:]))
		width := jxrExifTypeWidth(typ)
		if width == 0 || n > math.MaxUint32/width {
			return fmt.Errorf("%w: JPEG XR relocated Sub-IFD field", ErrInvalid)
		}
		fieldBytes := n * width
		if jxrIsNestedSubIFD(tag, typ, n, allowInterop) {
			if n == 1 {
				localChild := uint64(binary.LittleEndian.Uint32(entry[8:]))
				if err := relocateJXRSubIFD(data, localChild, delta, allowInterop, depth+1); err != nil {
					return err
				}
				if localChild > math.MaxUint32-delta {
					return fmt.Errorf("%w: JPEG XR relocated Sub-IFD pointer", ErrLimit)
				}
				binary.LittleEndian.PutUint32(entry[8:], uint32(localChild+delta))
			} else {
				arrayOffset := uint64(binary.LittleEndian.Uint32(entry[8:]))
				if arrayOffset > uint64(len(data)) || fieldBytes > uint64(len(data))-arrayOffset {
					return fmt.Errorf("%w: JPEG XR relocated Sub-IFD array", ErrInvalid)
				}
				for ci := range int(n) {
					slot := data[arrayOffset+uint64(ci)*4:]
					localChild := uint64(binary.LittleEndian.Uint32(slot))
					if err := relocateJXRSubIFD(data, localChild, delta, allowInterop, depth+1); err != nil {
						return err
					}
					if localChild > math.MaxUint32-delta {
						return fmt.Errorf("%w: JPEG XR relocated Sub-IFD pointer", ErrLimit)
					}
					binary.LittleEndian.PutUint32(slot, uint32(localChild+delta))
				}
				if arrayOffset > math.MaxUint32-delta {
					return fmt.Errorf("%w: JPEG XR relocated Sub-IFD array offset", ErrLimit)
				}
				binary.LittleEndian.PutUint32(entry[8:], uint32(arrayOffset+delta))
			}
		} else if fieldBytes > 4 {
			localOffset := uint64(binary.LittleEndian.Uint32(entry[8:]))
			if localOffset > math.MaxUint32-delta {
				return fmt.Errorf("%w: JPEG XR relocated Sub-IFD field offset", ErrLimit)
			}
			binary.LittleEndian.PutUint32(entry[8:], uint32(localOffset+delta))
		}
	}
	return nil
}

// jxrValidDateTime 校验ITU-T T.832 Clause A.7.13要求的DATE_TIME格式（YYYY:MM:DD HH:MM:SS）
// 入参: value 不含末尾NUL的日期时间文本
// 返回: bool 是否完全符合规范格式与日期时间分量范围
func jxrValidDateTime(value string) bool {
	if len(value) != 19 || value[4] != ':' || value[7] != ':' || value[10] != ' ' || value[13] != ':' || value[16] != ':' {
		return false
	}
	for i := range 19 {
		if i == 4 || i == 7 || i == 10 || i == 13 || i == 16 {
			continue
		}
		if value[i] < '0' || value[i] > '9' {
			return false
		}
	}
	month := int(value[5]-'0')*10 + int(value[6]-'0')
	day := int(value[8]-'0')*10 + int(value[9]-'0')
	hour := int(value[11]-'0')*10 + int(value[12]-'0')
	minute := int(value[14]-'0')*10 + int(value[15]-'0')
	second := int(value[17]-'0')*10 + int(value[18]-'0')
	return month >= 1 && month <= 12 && day >= 1 && day <= 31 && hour <= 23 && minute <= 59 && second <= 59
}

// jxrReadDescription 解析ITU-T T.832 Annex A.5定义的图像描述标签，全部描述标签缺省或均为空时返回nil
// 入参: d 已校验目录, parser 共享解析预算
// 返回: *ImageDescription 图像描述副本，全部描述标签缺省或均为空时为nil, error 编码、格式或资源错误
func jxrReadDescription(d *jxrDirectory, parser *jxrParser) (*ImageDescription, error) {
	var desc ImageDescription
	readText := func(tag uint16, target *string) error {
		text, err := jxrReadUTF8Field(d, parser, tag)
		if err != nil {
			return err
		}
		*target = text
		return nil
	}
	for _, item := range []struct {
		tag    uint16
		target *string
	}{
		{0x010d, &desc.DocumentName},
		{0x010e, &desc.ImageDescription},
		{0x010f, &desc.EquipmentMake},
		{0x0110, &desc.EquipmentModel},
		{0x011d, &desc.PageName},
		{0x0131, &desc.SoftwareNameVersion},
		{0x013b, &desc.ArtistName},
		{0x013c, &desc.HostComputer},
		{0x8298, &desc.CopyrightNotice},
	} {
		if err := readText(item.tag, item.target); err != nil {
			return nil, err
		}
	}
	if raw, err := d.jxrField(0x0132, 20, 20, 2); err != nil {
		return nil, err
	} else if raw != nil && raw[19] == 0 {
		text := string(raw[:19])
		if jxrValidDateTime(text) {
			if err := parser.reserve(19); err != nil {
				return nil, err
			}
			desc.DateTime = text
		}
	}
	if raw, err := d.jxrField(0x0129, 2, 4, 3); err != nil {
		return nil, err
	} else if raw != nil {
		if err := parser.reserve(4); err != nil {
			return nil, err
		}
		page := [2]uint16{d.order.Uint16(raw[0:2]), d.order.Uint16(raw[2:4])}
		desc.PageNumber = &page
	}
	if jxrDescriptionEmpty(&desc) {
		return nil, nil
	}
	if err := parser.reserve(256); err != nil {
		return nil, err
	}
	return &desc, nil
}

// jxrReadUTF8Field 读取并校验单个NUL终止的UTF-8描述标签
// 入参: d 已校验目录, parser 共享解析预算, tag 字段标签
// 返回: string 去掉末尾NUL的文本，缺省时为空字符串, error 编码、终止符或资源错误
func jxrReadUTF8Field(d *jxrDirectory, parser *jxrParser, tag uint16) (string, error) {
	raw, err := d.jxrField(tag, 0, math.MaxUint32, 2)
	if err != nil {
		return "", err
	}
	if raw == nil {
		return "", nil
	}
	if len(raw) == 0 || raw[len(raw)-1] != 0 {
		return "", fmt.Errorf("%w: JPEG XR UTF-8 field %x termination", ErrInvalid, tag)
	}
	body := raw[:len(raw)-1]
	if !utf8.Valid(body) || bytes.Contains(body, []byte{0, 0}) || len(body) > 0 && body[len(body)-1] == 0 {
		return "", fmt.Errorf("%w: JPEG XR UTF-8 field %x encoding", ErrInvalid, tag)
	}
	if err := parser.reserve(uint64(len(body))); err != nil {
		return "", err
	}
	return string(body), nil
}

// checkJXRDescription 校验待写入容器的Annex A.5描述结构体中全部文本与日期时间字段
// 入参: desc 图像描述指针
// 返回: error DateTime（0x0132）不符合YYYY:MM:DD HH:MM:SS格式，或文本非合法UTF-8、含非法NUL或超长时返回错误
func checkJXRDescription(desc *ImageDescription) error {
	if desc == nil {
		return nil
	}
	if desc.DateTime != "" && !jxrValidDateTime(desc.DateTime) {
		return fmt.Errorf("%w: JPEG XR description date time", ErrInvalid)
	}
	for _, value := range []string{
		desc.DocumentName,
		desc.ImageDescription,
		desc.EquipmentMake,
		desc.EquipmentModel,
		desc.PageName,
		desc.SoftwareNameVersion,
		desc.ArtistName,
		desc.HostComputer,
		desc.CopyrightNotice,
	} {
		if value == "" {
			continue
		}
		if uint64(len(value)) >= math.MaxUint32 {
			return fmt.Errorf("%w: JPEG XR description field length", ErrLimit)
		}
		if !utf8.ValidString(value) || strings.Contains(value, "\x00\x00") || strings.HasSuffix(value, "\x00") {
			return fmt.Errorf("%w: JPEG XR description UTF-8 encoding", ErrInvalid)
		}
	}
	return nil
}

// jxrUTF8Bytes 为非空描述文本追加单个NUL终止字节
// 入参: value 已校验的UTF-8字符串
// 返回: []byte NUL终止字节切片
func jxrUTF8Bytes(value string) []byte {
	out := make([]byte, len(value)+1)
	copy(out, value)
	return out
}

// dataSize 返回目录所在完整文件的逻辑长度，不将随机输入误当成目录局部字节
// 返回: uint64 全文件范围
func (d *jxrDirectory) dataSize() uint64 {
	if d.input != nil {
		return d.input.size
	}
	return uint64(len(d.data))
}

// storageWork 返回随机目录的独立临时容量，普通目录只引用已计入预算的完整切片
// 返回: uint64 目录工作容量
func (d *jxrDirectory) storageWork() uint64 {
	if d.input != nil {
		return uint64(134 + len(d.entries))
	}
	return 0
}

// entryFor 在已校验的有序目录原始字节中二分定位字段，不为标签建立额外索引
// 入参: tag 字段标识
// 返回: []byte 十二字节目录项，不存在时返回nil
func (d *jxrDirectory) entryFor(tag uint16) []byte {
	index := sort.Search(len(d.entries)/12, func(n int) bool { return d.order.Uint16(d.entries[n*12:]) >= tag })
	if index == len(d.entries)/12 || d.order.Uint16(d.entries[index*12:]) != tag {
		return nil
	}
	return d.entries[index*12 : (index+1)*12]
}

// setInfoMetadata 将图像信息中的分辨率、颜色声明及可选文件元数据复制到容器结构，并按方向变换调整分辨率轴向
// 入参: info 已合并选项覆盖的图像信息
func (c *jxrContainer) setInfoMetadata(info Info) {
	c.dpi, c.colorHint, c.encoding, c.imageType = info.Resolution, info.ColorHint, info.ColorEncoding, info.ImageType
	if c.transform&4 != 0 {
		c.dpi.X, c.dpi.Y = c.dpi.Y, c.dpi.X
	}
	if len(info.ICCProfile) != 0 {
		c.profile = info.ICCProfile
	}
	if len(info.EXIF) != 0 {
		c.exif = info.EXIF
	}
	if len(info.GPS) != 0 {
		c.gps = info.GPS
	}
	if len(info.XMP) != 0 {
		c.xmp = info.XMP
	}
	if len(info.IPTC) != 0 {
		c.iptc = info.IPTC
	}
	if !jxrDescriptionEmpty(info.Description) {
		c.description = info.Description
	}
}

// writeJXRContainer 写入单图像标准目录及主图、透明度和明确提供的元数据，不附加应用信息
// 入参: b 独占存储且不与输入共享的空缓冲, c 容器数据, maxPixels 输出像素上限
func writeJXRContainer(b *jxrBitWriter, c *jxrContainer, maxPixels uint64) {
	if b.err != nil {
		return
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return
	}
	if b.bit != 0 || len(b.data) != 0 {
		b.err = fmt.Errorf("%w: JPEG XR output file start", ErrInvalid)
		return
	}
	exifNode, gpsNode, err := checkJXRContainer(b, c, maxPixels)
	if err != nil {
		b.err = err
		return
	}
	writeJXRContainerBytes(b, c, false, c.alpha != nil, nil, exifNode, gpsNode)
}

// writeJXRContainerBytes 共用标准目录和元数据序列化，前缀模式不复制码流并返回主图长度字段的位置
// 入参: b 已校验的独占空缓冲, c 已校验元数据, prefix 是否只写主图之前的目录和元数据, separate 是否存在独立透明度, offsets 可选回填位置, exifNode 已解析EXIF子目录树, gpsNode 已解析GPS子目录树
func writeJXRContainerBytes(b *jxrBitWriter, c *jxrContainer, prefix, separate bool, offsets *jxrContainerOffsets, exifNode, gpsNode *jxrSubIFDNode) {
	var fields [36]jxrOutputField
	count := 0
	if c.description != nil {
		d := c.description
		if d.DocumentName != "" {
			raw := jxrUTF8Bytes(d.DocumentName)
			fields[count] = jxrOutputField{tag: 0x010d, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.ImageDescription != "" {
			raw := jxrUTF8Bytes(d.ImageDescription)
			fields[count] = jxrOutputField{tag: 0x010e, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.EquipmentMake != "" {
			raw := jxrUTF8Bytes(d.EquipmentMake)
			fields[count] = jxrOutputField{tag: 0x010f, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.EquipmentModel != "" {
			raw := jxrUTF8Bytes(d.EquipmentModel)
			fields[count] = jxrOutputField{tag: 0x0110, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.PageName != "" {
			raw := jxrUTF8Bytes(d.PageName)
			fields[count] = jxrOutputField{tag: 0x011d, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.PageNumber != nil {
			p := *d.PageNumber
			fields[count] = jxrOutputField{tag: 0x0129, typ: 3, count: 2, value: uint32(p[0]) | uint32(p[1])<<16}
			count++
		}
		if d.SoftwareNameVersion != "" {
			raw := jxrUTF8Bytes(d.SoftwareNameVersion)
			fields[count] = jxrOutputField{tag: 0x0131, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.DateTime != "" {
			raw := jxrUTF8Bytes(d.DateTime)
			fields[count] = jxrOutputField{tag: 0x0132, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.ArtistName != "" {
			raw := jxrUTF8Bytes(d.ArtistName)
			fields[count] = jxrOutputField{tag: 0x013b, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
		if d.HostComputer != "" {
			raw := jxrUTF8Bytes(d.HostComputer)
			fields[count] = jxrOutputField{tag: 0x013c, typ: 2, count: uint32(len(raw)), data: raw}
			count++
		}
	}
	if c.xmp != nil {
		fields[count] = jxrOutputField{tag: 0x02bc, typ: 1, count: uint32(len(c.xmp)), data: c.xmp}
		count++
	}
	if c.description != nil && c.description.CopyrightNotice != "" {
		raw := jxrUTF8Bytes(c.description.CopyrightNotice)
		fields[count] = jxrOutputField{tag: 0x8298, typ: 2, count: uint32(len(raw)), data: raw}
		count++
	}
	if c.iptc != nil {
		fields[count] = jxrOutputField{tag: 0x83bb, typ: 1, count: uint32(len(c.iptc)), data: c.iptc}
		count++
	}
	if exifNode != nil {
		fields[count] = jxrOutputField{tag: 0x8769, typ: 4, count: 1, data: c.exif, subIFD: exifNode}
		count++
	}
	if c.profile != nil {
		fields[count] = jxrOutputField{tag: 0x8773, typ: 7, count: uint32(len(c.profile)), data: c.profile}
		count++
	}
	if gpsNode != nil {
		fields[count] = jxrOutputField{tag: 0x8825, typ: 4, count: 1, data: c.gps, subIFD: gpsNode}
		count++
	}
	if c.colorHint != ColorHintNone {
		fields[count] = jxrOutputField{tag: 0xa001, typ: 3, count: 1, value: uint32(c.colorHint)}
		count++
	}
	fields[count] = jxrOutputField{tag: 0xbc01, typ: 1, count: 16, data: c.pixelFormat[:]}
	count++
	if c.transform != 0 {
		fields[count] = jxrOutputField{tag: 0xbc02, typ: 1, count: 1, value: c.transform}
		count++
	}
	if c.uncompressed {
		fields[count] = jxrOutputField{tag: 0xbc03, typ: 4, count: 1, value: 1}
		count++
	}
	if c.imageType != ImageTypeDefault {
		fields[count] = jxrOutputField{tag: 0xbc04, typ: 4, count: 1, value: uint32(c.imageType)}
		count++
	}
	if c.encoding != nil {
		value := uint32(c.encoding.Primaries) | uint32(c.encoding.Transfer)<<8 | uint32(c.encoding.Matrix)<<16
		if c.encoding.FullRange {
			value |= 1 << 24
		}
		fields[count] = jxrOutputField{tag: 0xbc05, typ: 1, count: 4, value: value}
		count++
	}
	if len(c.profiles) != 0 {
		fields[count] = jxrOutputField{tag: 0xbc06, typ: 1, count: uint32(len(c.profiles) * 4)}
		count++
	}
	fields[count], fields[count+1] = jxrOutputField{tag: 0xbc80, typ: 4, count: 1, value: uint32(c.width)}, jxrOutputField{tag: 0xbc81, typ: 4, count: 1, value: uint32(c.height)}
	count += 2
	for n, resolution := range [2]float64{c.dpi.X, c.dpi.Y} {
		if resolution != 0 && resolution != 96 {
			fields[count] = jxrOutputField{tag: uint16(0xbc82 + n), typ: 11, count: 1, value: math.Float32bits(float32(resolution))}
			count++
		}
	}
	imageField := count
	fields[count], fields[count+1] = jxrOutputField{tag: 0xbcc0, typ: 4, count: 1}, jxrOutputField{tag: 0xbcc1, typ: 4, count: 1, value: uint32(len(c.image))}
	count += 2
	alphaField := -1
	if separate {
		alphaField = count
		fields[count], fields[count+1] = jxrOutputField{tag: 0xbcc2, typ: 4, count: 1}, jxrOutputField{tag: 0xbcc3, typ: 4, count: 1, value: uint32(len(c.alpha))}
		count += 2
	}
	if c.imageBands != 0 {
		fields[count] = jxrOutputField{tag: 0xbcc4, typ: 1, count: 1, value: c.imageBands}
		count++
	}
	if separate && c.alphaBands != 0 {
		fields[count] = jxrOutputField{tag: 0xbcc5, typ: 1, count: 1, value: c.alphaBands}
		count++
	}
	position := uint64(8 + 2 + count*12 + 4)
	for n := range count {
		field := &fields[n]
		if field.subIFD != nil {
			if position > math.MaxUint32 || field.subIFD.totalBytes > math.MaxUint32-position {
				b.err = fmt.Errorf("%w: JPEG XR output metadata size", ErrLimit)
				return
			}
			field.value = uint32(position)
			position = (position + field.subIFD.totalBytes + 1) &^ 1
			continue
		}
		size := uint64(len(field.data))
		if field.tag == 0xbc06 {
			size = uint64(field.count)
		}
		if size > 4 {
			if position > math.MaxUint32 || size > math.MaxUint32-position {
				b.err = fmt.Errorf("%w: JPEG XR output metadata size", ErrLimit)
				return
			}
			field.value = uint32(position)
			position += size
			position = (position + 1) &^ 1
		}
	}
	if position > math.MaxUint32 {
		b.err = fmt.Errorf("%w: JPEG XR output metadata size", ErrLimit)
		return
	}
	fields[imageField].value = uint32(position)
	position += uint64(len(c.image))
	if alphaField >= 0 {
		position = (position + 1) &^ 1
		if position > math.MaxUint32 {
			b.err = fmt.Errorf("%w: JPEG XR output alpha offset", ErrLimit)
			return
		}
		fields[alphaField].value = uint32(position)
		position += uint64(len(c.alpha))
	}
	if prefix {
		position = uint64(fields[imageField].value)
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return
	}
	if position > math.MaxUint32 || !b.reserve(position) {
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR output file size", ErrLimit)
		}
		return
	}
	for offset := 0; offset < len(b.data); offset += 16384 {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
		clear(b.data[offset:min(offset+16384, len(b.data))])
	}
	order := binary.LittleEndian
	copy(b.data, "II\xbc\x01")
	order.PutUint32(b.data[4:], 8)
	order.PutUint16(b.data[8:], uint16(count))
	for n := range count {
		field, entry := &fields[n], b.data[10+n*12:10+(n+1)*12]
		order.PutUint16(entry, field.tag)
		order.PutUint16(entry[2:], field.typ)
		order.PutUint32(entry[4:], field.count)
		if field.subIFD != nil {
			order.PutUint32(entry[8:], field.value)
			srcRaw := field.data
			if _, err := jxrEmitSubIFD(b.data, uint64(field.value), field.subIFD, func(srcOffset uint64, dst []byte) error {
				for len(dst) != 0 {
					if err := b.ctx.Err(); err != nil {
						return err
					}
					part := min(16384, len(dst))
					copy(dst[:part], srcRaw[srcOffset:srcOffset+uint64(part)])
					srcOffset += uint64(part)
					dst = dst[part:]
				}
				return nil
			}); err != nil {
				b.err = err
				return
			}
		} else if field.tag == 0xbc06 {
			target := entry[8:12]
			if field.count > 4 {
				order.PutUint32(entry[8:], field.value)
				target = b.data[uint64(field.value) : uint64(field.value)+uint64(field.count)]
			}
			for index, profile := range c.profiles {
				if err := b.ctx.Err(); err != nil {
					b.err = err
					return
				}
				at := target[index*4 : index*4+4]
				at[0], at[1] = profile.profile, profile.level
				if index == len(c.profiles)-1 {
					at[3] = 1
				}
			}
		} else if len(field.data) != 0 && len(field.data) <= 4 {
			copy(entry[8:12], field.data)
		} else {
			order.PutUint32(entry[8:], field.value)
			if len(field.data) > 4 {
				copyJXRContainerData(b, uint64(field.value), field.data)
			}
		}
	}
	order.PutUint32(b.data[10+count*12:], 0)
	if !prefix {
		copyJXRContainerData(b, uint64(fields[imageField].value), c.image)
		if alphaField >= 0 {
			copyJXRContainerData(b, uint64(fields[alphaField].value), c.alpha)
		}
	}
	if b.err == nil {
		b.bit = position * 8
		if offsets != nil {
			offsets.imageSize = uint64(10 + (imageField+1)*12 + 8)
			if alphaField >= 0 {
				offsets.alphaOffset = uint64(10 + alphaField*12 + 8)
				offsets.alphaSize = uint64(10 + (alphaField+1)*12 + 8)
			}
		}
	}
}

// checkJXRContainer 校验文件目录的尺寸、标准格式标识、数值元数据及码流绑定，并返回已解析的EXIF与GPS子目录树
// 入参: b 输出器及共享剩余预算, c 容器数据, maxPixels 输出像素上限
// 返回: *jxrSubIFDNode EXIF子目录树, *jxrSubIFDNode GPS子目录树, error 参数、资源、取消或码流错误
func checkJXRContainer(b *jxrBitWriter, c *jxrContainer, maxPixels uint64) (*jxrSubIFDNode, *jxrSubIFDNode, error) {
	if c != nil && (c.imageInput != nil || c.alphaInput != nil) {
		return nil, nil, fmt.Errorf("%w: JPEG XR output requires encoded byte buffers", ErrUnsupported)
	}
	if uint64(cap(b.data)) > b.maxMemory {
		return nil, nil, &LimitError{Resource: "JPEG XR container validation buffer memory", Limit: b.maxMemory, Required: uint64(cap(b.data))}
	}
	available := b.maxMemory - uint64(cap(b.data))
	if b.budget != nil {
		if b.budget.used > b.budget.limit {
			return nil, nil, &LimitError{Resource: "JPEG XR container validation memory", Limit: b.budget.limit, Required: b.budget.used}
		}
		available = min(available, b.budget.limit-b.budget.used)
	}
	exifNode, gpsNode, err := checkJXRContainerMetadataLayout(b.ctx, c, maxPixels, available, c != nil && c.separateAlpha())
	if err != nil {
		return nil, nil, err
	}
	parser := jxrParser{ctx: b.ctx, limit: available}
	source, err := parser.containerSource(c, maxPixels)
	if err != nil {
		return nil, nil, err
	}
	if c.uncompressed {
		return exifNode, gpsNode, nil
	}
	channels := c.format.channels
	if c.format.alpha {
		channels++
	}
	if err := source.primary.checkProfileList(c.profiles, c.format.alpha, channels); err != nil {
		return nil, nil, err
	}
	return exifNode, gpsNode, nil
}

// checkJXRContainerMetadataLayout 校验独立透明度布局的目录声明，不用虚构压缩字节表示待写出的平面，并返回已解析的EXIF与GPS子目录树
// 入参: ctx 已规范化Context, c 目录元数据, maxPixels 像素上限, maxMemory 可用解析内存上限, separate 是否存在独立透明度码流
// 返回: *jxrSubIFDNode EXIF子目录树, *jxrSubIFDNode GPS子目录树, error 元数据、格式、取消或容量错误
func checkJXRContainerMetadataLayout(ctx context.Context, c *jxrContainer, maxPixels, maxMemory uint64, separate bool) (*jxrSubIFDNode, *jxrSubIFDNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if c == nil || c.width == 0 || c.height == 0 || c.transform > 7 || c.imageBands > 3 || c.alphaBands > 3 || !separate && c.alphaBands != 0 || c.imageType&^3 != 0 {
		return nil, nil, fmt.Errorf("%w: JPEG XR output container fields", ErrInvalid)
	}
	if c.uncompressed && (c.imageBands != 0 || c.alphaBands != 0 || len(c.profiles) != 0) {
		return nil, nil, fmt.Errorf("%w: JPEG XR uncompressed container fields", ErrInvalid)
	}
	if c.width > math.MaxUint32 || c.height > math.MaxUint32 || uint64(len(c.image)) > math.MaxUint32 || uint64(len(c.alpha)) > math.MaxUint32 ||
		uint64(len(c.profile)) > math.MaxUint32 || uint64(len(c.exif)) > math.MaxUint32 || uint64(len(c.gps)) > math.MaxUint32 || uint64(len(c.xmp)) > math.MaxUint32 || uint64(len(c.iptc)) > math.MaxUint32 {
		return nil, nil, fmt.Errorf("%w: JPEG XR output container field capacity", ErrLimit)
	}
	if err := checkJXRDimensions(c.width, c.height, maxPixels); err != nil {
		return nil, nil, err
	}
	if c.profile != nil && len(c.profile) == 0 || c.exif != nil && len(c.exif) == 0 || c.gps != nil && len(c.gps) == 0 || c.xmp != nil && len(c.xmp) == 0 || c.iptc != nil && len(c.iptc) == 0 || c.alpha != nil && len(c.alpha) == 0 {
		return nil, nil, fmt.Errorf("%w: JPEG XR output empty container data", ErrInvalid)
	}
	var exifNode, gpsNode *jxrSubIFDNode
	if c.exif != nil || c.gps != nil {
		parser := jxrParser{ctx: ctx, limit: maxMemory}
		if c.exif != nil {
			var err error
			exifNode, err = jxrParseSubIFDBytes(&parser, c.exif, true)
			if err != nil {
				return nil, nil, err
			}
		}
		if c.gps != nil {
			var err error
			gpsNode, err = jxrParseSubIFDBytes(&parser, c.gps, false)
			if err != nil {
				return nil, nil, err
			}
		}
	}
	if err := checkJXRDescription(c.description); err != nil {
		return nil, nil, err
	}
	f, err := readJXRPixelFormat(c.pixelFormat[:])
	if err != nil {
		return nil, nil, err
	}
	if f.name == "BlackWhite" && c.format.depth == 15 {
		f.depth = 15
	}
	if c.format != f {
		return nil, nil, fmt.Errorf("%w: JPEG XR output container pixel format", ErrInvalid)
	}
	if separate && !f.alpha {
		return nil, nil, fmt.Errorf("%w: JPEG XR separate alpha pixel format", ErrInvalid)
	}
	if err := checkJXRColorMetadata(f, c.colorHint, c.encoding); err != nil {
		return nil, nil, err
	}
	for _, resolution := range [2]float64{c.dpi.X, c.dpi.Y} {
		value := float64(float32(resolution))
		if !geometryFinite(resolution) || resolution < 0 || resolution != 0 && (!geometryFinite(value) || value <= 0) {
			return nil, nil, fmt.Errorf("%w: JPEG XR output resolution", ErrInvalid)
		}
	}
	if err := checkJXRProfileDeclarations(c.profiles); err != nil {
		return nil, nil, err
	}
	return exifNode, gpsNode, nil
}

// copyJXRContainerData 分段复制已确定范围的数据，复制期间检查取消
// 入参: b 已分配的文件缓冲, offset 数据起点, data 原始数据
func copyJXRContainerData(b *jxrBitWriter, offset uint64, data []byte) {
	if b.err != nil {
		return
	}
	for len(data) != 0 {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
		count := min(16384, len(data))
		copy(b.data[offset:offset+uint64(count)], data[:count])
		offset += uint64(count)
		data = data[count:]
	}
}

// NewDecoderAt 按需读取JPEG XR容器或独立码流，size受MaxInputBytes约束，不持有完整压缩输入
// r在解码器存续期间须保持稳定可读且Close不关闭r，构造时在内存预算内校验固定读取窗、头部、索引及单份Info副本
// 入参: ctx Context，nil采用Background, r 随机输入, size 码流长度, opts 解码选项，nil采用默认上限
// 返回: *Decoder 可复用解码器, error 结构、能力、读取、资源或取消错误
func NewDecoderAt(ctx context.Context, r io.ReaderAt, size int64, opts *DecodeOptions) (*Decoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	decoder, err := newDecoderAt(ctx, r, size, opts)
	if err != nil {
		return nil, wrapJXRError("NewDecoderAt", err)
	}
	return decoder, nil
}

// newDecoderAt 执行NewDecoderAt的核心解析逻辑，由公开入口统一归一化错误
// 入参: ctx 已规范化Context, r 随机输入, size 码流长度, opts 解码选项
// 返回: *Decoder 可复用解码器, error 解析或资源错误
func newDecoderAt(ctx context.Context, r io.ReaderAt, size int64, opts *DecodeOptions) (*Decoder, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || size < 8 {
		return nil, fmt.Errorf("%w: JPEG XR random input", ErrInvalid)
	}
	limits := DefaultLimits()
	workers := 0
	band := BandAll
	legacyHighpass := false
	keepOrientation := false
	if opts != nil {
		if opts.Workers < 0 {
			return nil, fmt.Errorf("%w: negative decode workers", ErrInvalid)
		}
		if opts.Band > BandDC {
			return nil, fmt.Errorf("%w: decode band", ErrInvalid)
		}
		limits = opts.Limits.normalized()
		workers = opts.Workers
		band = opts.Band
		legacyHighpass = opts.LegacyHighpassPrediction
		keepOrientation = opts.KeepOrientation
	}
	if uint64(size) > limits.MaxInputBytes {
		return nil, &LimitError{Resource: "JPEG XR random input bytes", Limit: limits.MaxInputBytes, Required: uint64(size)}
	}
	if uint64(size) > ^uint64(0)/8 {
		return nil, &LimitError{Resource: "JPEG XR random input bit capacity", Limit: ^uint64(0) / 8, Required: uint64(size)}
	}
	if limits.MaxMemoryBytes < 8 {
		return nil, &LimitError{Resource: "JPEG XR random input memory", Limit: limits.MaxMemoryBytes, Required: 8}
	}
	parser := jxrParser{ctx: ctx, limit: limits.MaxMemoryBytes - 8, legacyHighpassPrediction: legacyHighpass}
	if err := parser.reserve(jxrReaderAtWork); err != nil {
		return nil, err
	}
	input := &jxrReaderAt{source: r, ctx: ctx, size: uint64(size)}
	defer func() { input.ctx = nil }()
	var signature [8]byte
	if err := input.copy(0, signature[:4]); err != nil {
		return nil, err
	}
	first := binary.BigEndian.Uint32(signature[:4])
	if first>>8 == 0x4949bc {
		if first&255 != 1 {
			return nil, fmt.Errorf("%w: JPEG XR file version", ErrUnsupported)
		}
		decoder, err := fileDecoderAt(ctx, input, &parser, limits, keepOrientation)
		if err != nil {
			return nil, err
		}
		decoder.setWorkers(workers)
		decoder.setDecodeBand(band)
		return decoder, nil
	}
	if err := input.copy(4, signature[4:]); err != nil {
		return nil, err
	}
	second := binary.BigEndian.Uint32(signature[4:])
	if first != 0x574d5048 || second != 0x4f544f00 {
		return nil, fmt.Errorf("%w: JPEG XR signature", ErrInvalid)
	}
	window := min(uint64(len(input.buffer)), input.size)
	copy(input.buffer[:8], signature[:])
	if err := input.copy(8, input.buffer[8:window]); err != nil {
		return nil, err
	}
	input.count = int(window)
	b := jxrBits{input: input, parser: &parser}
	stream, err := parser.streamReader(b, limits.MaxPixels)
	if err != nil {
		return nil, err
	}
	source := jxrCodedSource(stream, nil)
	if keepOrientation {
		source.container.transform = 0
	}
	info, err := parser.info(&source)
	if err != nil {
		return nil, err
	}
	decoder := &Decoder{data: []byte("WMPHOTO\x00"), source: source, info: info, limits: limits, legacyHighpassPrediction: legacyHighpass, keepOrientation: keepOrientation}
	decoder.setWorkers(workers)
	decoder.setDecodeBand(band)
	if err := decoder.checkInfoCopy(); err != nil {
		return nil, err
	}
	return decoder, nil
}

// bytes 返回当前读取窗内的连续字节，检查ReaderAt协议并保留非EOF底层错误
// 入参: at 视图内绝对偏移, count 至多五个字节
// 返回: []byte 窗口视图，在下一次填充前有效, error 范围、取消、底层或协议错误
func (r *jxrReaderAt) bytes(at uint64, count uint) ([]byte, error) {
	return r.boundedBytes(at, count, 0)
}

// boundedBytes 保留随机读取窗并在声明上界处停止物理预取
// 入参: at 视图内绝对偏移, count 至多五个字节, limit 可选声明上界，零表示视图末尾
// 返回: []byte 窗口视图, error 范围、取消、底层或协议错误
func (r *jxrReaderAt) boundedBytes(at uint64, count uint, limit uint64) ([]byte, error) {
	if r.ctx != nil {
		if err := r.ctx.Err(); err != nil {
			return nil, err
		}
	}
	if count > 5 || at > r.size || uint64(count) > r.size-at || r.base > math.MaxInt64 || at > math.MaxInt64-r.base {
		return nil, fmt.Errorf("%w: JPEG XR random input range", ErrInvalid)
	}
	if at >= r.start && at-r.start <= uint64(r.count) && uint64(count) <= uint64(r.count)-(at-r.start) {
		return r.buffer[at-r.start : at-r.start+uint64(count)], nil
	}
	end := r.size
	if limit != 0 {
		end = min(end, limit)
		if at > end || uint64(count) > end-at {
			return nil, fmt.Errorf("%w: JPEG XR bounded input range", ErrInvalid)
		}
	}
	requested := int(min(uint64(len(r.buffer)), end-at))
	r.count = 0
	n, err := r.source.ReadAt(r.buffer[:requested], int64(r.base+at))
	if n < 0 || n > requested || n < requested && err == nil {
		return nil, fmt.Errorf("%w: JPEG XR random input read count", ErrInvalid)
	}
	if err != nil && err != io.EOF {
		return nil, err
	}
	r.start, r.count = at, n
	if n < int(count) {
		return nil, fmt.Errorf("JPEG XR random input truncation: %w", io.ErrUnexpectedEOF)
	}
	if r.ctx != nil {
		if err := r.ctx.Err(); err != nil {
			return nil, err
		}
	}
	return r.buffer[:count], nil
}

// copy 将已预留容量的元数据分段读入独立目标，不复用位读取窗或分配整文件缓冲
// 入参: at 视图内绝对偏移, data 已分配目标
// 返回: error 范围、取消、底层或读取协议错误
func (r *jxrReaderAt) copy(at uint64, data []byte) error {
	if at > r.size || uint64(len(data)) > r.size-at || r.base > math.MaxInt64 || at > math.MaxInt64-r.base || uint64(len(data)) > math.MaxInt64-r.base-at {
		return fmt.Errorf("%w: JPEG XR random metadata range", ErrInvalid)
	}
	for len(data) != 0 {
		if r.ctx != nil {
			if err := r.ctx.Err(); err != nil {
				return err
			}
		}
		part := data[:min(len(data), 32768)]
		n, err := r.source.ReadAt(part, int64(r.base+at))
		if n < 0 || n > len(part) || n < len(part) && err == nil {
			return fmt.Errorf("%w: JPEG XR random metadata read count", ErrInvalid)
		}
		if err != nil && err != io.EOF {
			return err
		}
		if n != len(part) {
			return fmt.Errorf("JPEG XR random metadata truncation: %w", io.ErrUnexpectedEOF)
		}
		at += uint64(n)
		data = data[n:]
	}
	if r.ctx != nil {
		return r.ctx.Err()
	}
	return nil
}

// inputWork 计算当前解码器实际持有的输入缓冲，不把随机源的完整地址范围当作内存切片
// 返回: uint64 输入缓冲及随机读取窗容量
func (d *Decoder) inputWork() uint64 {
	work := uint64(cap(d.data)) + d.inputExtra
	if d.fileInput != nil {
		work += jxrReaderAtWork
	}
	primaryInput := d.source.primary.input
	if primaryInput == nil {
		primaryInput = d.source.container.imageInput
	}
	if primaryInput != nil {
		work += primaryInput.storageWork()
	}
	var alphaInput *jxrReaderAt
	if d.source.alpha != nil {
		alphaInput = d.source.alpha.input
	}
	if alphaInput == nil {
		alphaInput = d.source.container.alphaInput
	}
	if alphaInput != nil && alphaInput != primaryInput {
		work += alphaInput.storageWork()
	}
	return work
}

// storageWork 计算读取窗所占预算
// 返回: uint64 常驻读取资源容量
func (r *jxrReaderAt) storageWork() uint64 {
	return jxrReaderAtWork
}

// bindInputContext 为独占操作设置随机读取Context，nil释放已结束的Context引用
// 入参: ctx 当前操作Context或nil
func (d *Decoder) bindInputContext(ctx context.Context) {
	if d.fileInput != nil {
		d.fileInput.ctx = ctx
	}
	d.source.bindLogicalContext(ctx)
}

// bindLogicalContext 为逻辑样本视图绑定或释放操作Context，不修改共享文件的Context
// 入参: ctx 当前操作Context或nil
func (source *jxrRasterSource) bindLogicalContext(ctx context.Context) {
	if source.primary.input != nil {
		source.primary.input.ctx = ctx
	}
	if source.container.imageInput != nil {
		source.container.imageInput.ctx = ctx
	}
	if source.alpha != nil && source.alpha.input != nil {
		source.alpha.input.ctx = ctx
	}
	if source.container.alphaInput != nil {
		source.container.alphaInput.ctx = ctx
	}
}

// directoryAt 在分配原始目录项前检查剩余预算，偏移始终相对于完整文件
// 入参: input 完整文件随机视图, offset 目录全文件位置
// 返回: jxrDirectory 独立目录项和共享随机源, error 格式、范围、资源、读取或取消错误
func (parser *jxrParser) directoryAt(input *jxrReaderAt, offset uint64) (jxrDirectory, error) {
	if offset < 8 || offset&1 != 0 || offset > input.size || input.size-offset < 2 {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory offset", ErrInvalid)
	}
	input.ctx = parser.ctx
	field, err := input.bytes(offset, 2)
	if err != nil {
		return jxrDirectory{}, err
	}
	count := uint64(binary.LittleEndian.Uint16(field))
	length := count*12 + 6
	if count == 0 || length > input.size-offset {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory size", ErrInvalid)
	}
	if err := parser.reserve(128 + length); err != nil {
		return jxrDirectory{}, err
	}
	data := make([]byte, int(length))
	if err := input.copy(offset, data); err != nil {
		return jxrDirectory{}, err
	}
	if uint64(binary.LittleEndian.Uint16(data)) != count {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR unstable directory input", ErrInvalid)
	}
	directory := jxrDirectory{order: binary.LittleEndian, entries: data[2 : 2+count*12], next: binary.LittleEndian.Uint32(data[2+count*12:]), input: input, parser: parser}
	if err := validateJXRDirectoryEntries(directory.entries, directory.next); err != nil {
		return jxrDirectory{}, err
	}
	return directory, nil
}

// fileDecoderAt 解析容器首目录及双流有界视图，不持有目录临时缓冲或完整压缩文件
// 入参: ctx Context, input 已预留固定窗的完整文件视图, parser 共享解析预算, limits 解码上限, keepOrientation 是否保留原始编码方向
// 返回: *Decoder 独立元数据及随机源, error 容器、码流、读取或资源错误
func fileDecoderAt(ctx context.Context, input *jxrReaderAt, parser *jxrParser, limits Limits, keepOrientation bool) (*Decoder, error) {
	if input.size > math.MaxUint32 {
		return nil, fmt.Errorf("%w: JPEG XR file size", ErrLimit)
	}
	field, err := input.bytes(4, 4)
	if err != nil {
		return nil, err
	}
	var header [8]byte
	copy(header[:4], "II\xbc\x01")
	copy(header[4:], field)
	directory, err := parser.directoryAt(input, uint64(binary.LittleEndian.Uint32(header[4:])))
	if err != nil {
		return nil, err
	}
	container, err := parser.directoryContainer(&directory, limits.MaxPixels)
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	extra := uint64(cap(container.profile)) + uint64(cap(container.exif)) + uint64(cap(container.gps)) + uint64(cap(container.xmp)) + uint64(cap(container.iptc))
	decoder := &Decoder{data: header[:], source: source, info: info, limits: limits, fileInput: input, inputExtra: extra, legacyHighpassPrediction: parser.legacyHighpassPrediction, keepOrientation: keepOrientation}
	decoder.bindInputContext(nil)
	if err := decoder.checkInfoCopy(); err != nil {
		return nil, err
	}
	return decoder, nil
}

// imageDirectoryAt 取得已经通过文件头校验的目录，随机输入分配受调用方剩余容量约束
// 入参: ctx Context, offset 全文件偏移, maximum 本次目录可用临时容量
// 返回: jxrDirectory 原始目录视图, error 目录、资源或读取错误
func (d *Decoder) imageDirectoryAt(ctx context.Context, offset, maximum uint64) (jxrDirectory, error) {
	if d.fileInput == nil {
		return readJXRDirectoryAt(d.data, offset)
	}
	parser := jxrParser{ctx: ctx, limit: maximum}
	return parser.directoryAt(d.fileInput, offset)
}

// codedSource 在共享预算内为独立码流建立输出描述，不伪造分辨率或ICC配置
// 入参: data 独立码流, maxPixels 图像像素上限
// 返回: jxrRasterSource 已校验图像源, error 结构、能力或资源错误
func (parser *jxrParser) codedSource(data []byte, maxPixels uint64) (jxrRasterSource, error) {
	s, err := parser.stream(data, maxPixels)
	if err != nil {
		return jxrRasterSource{}, err
	}
	return jxrCodedSource(s, data), nil
}

// jxrCodedSource 根据已校验的独立码流生成原生源，随机输入不需要完整码流切片
// 入参: s 已校验独立码流, data 可选完整输入切片
// 返回: jxrRasterSource 原生图像源
func jxrCodedSource(s jxrStream, data []byte) jxrRasterSource {
	f := jxrCodedFormat(&s)
	f.name = codedPixelFormatName(f)
	h := &s.header
	return jxrRasterSource{format: FormatCodestream, primary: s, container: jxrContainer{width: h.width, height: h.height, format: f, transform: uint32(h.transform), orientation: Orientation(h.transform), image: data, imageBands: uint32(s.primary.bands)}}
}

// jxrCodedFormat 根据独立码流的输出语法确定颜色及数值格式
// 入参: s 已解析码流
// 返回: jxrPixelFormat 输出格式
func jxrCodedFormat(s *jxrStream) jxrPixelFormat {
	h := &s.header
	channels := s.primary.channels
	switch h.color {
	case 0:
		channels = 1
	case 1, 2, 3, 7, 8:
		channels = 3
	case 4, 5:
		channels = 4
	}
	number := uint8(jxrUnsigned)
	switch h.depth {
	case 3:
		number = jxrFixed16
	case 4:
		number = jxrHalf
	case 6:
		number = jxrFixed32
	case 7:
		number = jxrFloat
	}
	if h.color == 8 {
		number = jxrRGBE
	}
	return jxrPixelFormat{channels: channels, color: h.color, depth: h.depth, number: number, alpha: h.alpha, premultiply: h.alpha && h.premultiply}
}

// codedPixelFormatName 生成独立码流的标准颜色及位深名称，不包含文件GUID元数据
// 入参: format 标准输出格式
// 返回: string 原生格式名称
func codedPixelFormatName(format jxrPixelFormat) string {
	colors := [...]string{"Gray", "YCC420", "YCC422", "YCC444", "CMYK", "CMYKDIRECT", "NComponent", "RGB", "RGBE"}
	depths := [...]string{"1", "8", "16", "Fixed16", "Half", "", "Fixed32", "Float", "5", "10", "565", "", "", "", "", "1"}
	return fmt.Sprintf("%s/%s", colors[format.color], depths[format.depth])
}

// info 根据输出采样、裁剪及方向生成图像与分量网格，不推断未指定的色度中心
// 返回: Info 图像及分量信息
func (s *jxrRasterSource) info() Info {
	c := &s.container
	width, height := int(c.width), int(c.height)
	dpi := c.dpi
	if c.transform&4 != 0 {
		width, height, dpi.X, dpi.Y = height, width, dpi.Y, dpi.X
	}
	info := Info{Format: s.format, ImageType: c.imageType, Orientation: c.orientation, Bounds: image.Rect(0, 0, width, height), PixelFormat: c.format.name, Channels: c.format.channels, ColorSpace: jxrColorSpace(c.format.color), Alpha: c.format.alpha, Premultiplied: c.format.premultiply, Linear: jxrInfoLinear(c.format), Resolution: dpi, ICCProfile: c.profile, EXIF: c.exif, GPS: c.gps, XMP: c.xmp, IPTC: c.iptc, Description: cloneImageDescription(c.description)}
	info.ColorHint, info.ColorEncoding = c.colorHint, cloneColorEncoding(c.encoding)
	count := info.Channels
	if info.Alpha {
		count++
	}
	info.Components = make([]ComponentInfo, count)
	for index := range info.Components {
		info.Components[index] = s.componentInfo(index)
	}
	return info
}

// componentInfo 计算单分量的采样间隔、范围及方向变换后的首个样本位置
// 入参: index 颜色或透明度通道序号
// 返回: ComponentInfo 分量网格
func (s *jxrRasterSource) componentInfo(index int) ComponentInfo {
	c, p := &s.container, &s.primary.primary
	width, height := int(c.width), int(c.height)
	sx, sy := uint8(1), uint8(1)
	ox, oy := 0.0, 0.0
	knownX, knownY := true, true
	if index > 0 && index < c.format.channels && (c.format.color == 1 || c.format.color == 2) {
		chromaX, chromaY := p.chromaX, p.chromaY
		sx = 2
		knownX = chromaX <= 4
		if knownX {
			ox = float64(chromaX) / 4
		}
		if c.format.color == 1 {
			sy = 2
			knownY = chromaY <= 4
			if knownY {
				oy = float64(chromaY) / 4
			}
		}
	}
	pw, ph := width/int(sx), height/int(sy)
	if c.transform&4 != 0 {
		ox, oy = float64(height-1)-oy-float64(ph-1)*float64(sy), ox
		knownX, knownY = knownY, knownX
		sx, sy, pw, ph, width, height = sy, sx, ph, pw, height, width
	}
	if c.transform&2 != 0 {
		ox = float64(width-1) - ox - float64(pw-1)*float64(sx)
	}
	if c.transform&1 != 0 {
		oy = float64(height-1) - oy - float64(ph-1)*float64(sy)
	}
	if !knownX {
		ox = 0
	}
	if !knownY {
		oy = 0
	}
	info := jxrSampleInfo(c.format, index)
	info.Bounds, info.XStep, info.YStep = image.Rect(0, 0, pw, ph), sx, sy
	info.XOffset, info.YOffset = ox, oy
	info.XPositionKnown, info.YPositionKnown = knownX, knownY
	return info
}

// sampleCount 计算原生颜色及透明度样本数，像素、分量或累加溢出时立即饱和，不缩小子采样容量
// 返回: uint64 样本总数，任一步溢出时返回最大值
func (s *jxrRasterSource) sampleCount() uint64 {
	c := &s.container
	if c.height != 0 && c.width > ^uint64(0)/c.height {
		return ^uint64(0)
	}
	count := uint64(c.width) * uint64(c.height)
	var samples uint64
	switch c.format.color {
	case 1:
		if count > ^uint64(0)-count/2 {
			return ^uint64(0)
		}
		samples = count + count/2
	case 2:
		if count > ^uint64(0)/2 {
			return ^uint64(0)
		}
		samples = count * 2
	default:
		channels := uint64(c.format.channels)
		if channels != 0 && count > ^uint64(0)/channels {
			return ^uint64(0)
		}
		samples = count * channels
	}
	if c.format.alpha {
		if count > ^uint64(0)-samples {
			return ^uint64(0)
		}
		samples += count
	}
	return samples
}

// jxrWritableProfileLevel 检查档位与级别组合是否允许编码输出，拒绝保留值
// 入参: declaration 待输出声明
// 返回: bool 是否为已定义且保留位全零的输出组合
func jxrWritableProfileLevel(declaration jxrProfileLevel) bool {
	if declaration.reserved != 0 {
		return false
	}
	switch declaration.profile {
	case 44, 55, 66, jxrDefaultProfile:
	default:
		return false
	}
	switch declaration.level {
	case 4, 8, 16, 32, 64, 128, jxrDefaultLevel:
		return true
	default:
		return false
	}
}

// profiles 在输入范围与共享预算内预扫描并去重档次级别声明，失败时归还新增预留
// 入参: b 码流读取器, end 声明区域的字节结束位置
// 返回: []jxrProfileLevel 声明列表, error 数据或资源错误
func (parser *jxrParser) profiles(b *jxrBits, end uint64) ([]jxrProfileLevel, error) {
	if b.err != nil {
		return nil, b.err
	}
	if b.bit%8 != 0 || end > b.length() || end > ^uint64(0)/8 || b.bit/8 > end {
		return nil, fmt.Errorf("%w: JPEG XR profile declaration range", ErrInvalid)
	}
	scan, count := *b, uint64(0)
	for {
		if err := parser.check(); err != nil {
			return nil, err
		}
		if scan.bit/8 > end || end-scan.bit/8 < 4 {
			return nil, fmt.Errorf("%w: JPEG XR profile declaration size", ErrInvalid)
		}
		if count >= uint64(^uint(0)>>1)/4 {
			return nil, fmt.Errorf("%w: JPEG XR profile declaration address capacity", ErrLimit)
		}
		if parser.used > parser.limit || count >= (parser.limit-parser.used)/8 {
			return nil, fmt.Errorf("%w: JPEG XR profile declaration memory", ErrLimit)
		}
		_ = scan.read(31)
		last := scan.flag()
		if scan.err != nil {
			return nil, scan.err
		}
		count++
		if last {
			break
		}
	}
	work := count * 8
	if err := parser.reserve(work); err != nil {
		return nil, err
	}
	defer func() { parser.used -= work }()
	profiles, keys := make([]jxrProfileLevel, count), make([]uint32, count)
	for index := range profiles {
		if err := parser.check(); err != nil {
			return nil, err
		}
		profile, level, reserved := b.read(8), b.read(8), b.read(15)
		_ = b.flag()
		if b.err != nil {
			return nil, b.err
		}
		profiles[index] = jxrProfileLevel{uint8(profile), uint8(level), uint16(reserved)}
		keys[index] = profile<<23 | level<<15 | reserved
	}
	slices.Sort(keys)
	for index := 1; index < len(keys); index++ {
		if err := parser.check(); err != nil {
			return nil, err
		}
		if keys[index] == keys[index-1] {
			return nil, fmt.Errorf("%w: duplicate JPEG XR profile declaration", ErrInvalid)
		}
	}
	if err := parser.check(); err != nil {
		return nil, err
	}
	work = count * 4
	return profiles, nil
}

// checkProfile 按声明校验已定义的档次及级别，缺省声明采用高级档次和级别255
// 返回: error 档次或级别约束错误
func (s *jxrStream) checkProfile() error {
	format := jxrCodedFormat(s)
	hasAlpha := format.alpha || s.alpha != nil
	count := format.channels
	if hasAlpha {
		count++
	}
	if len(s.profiles) == 0 {
		return s.checkLevel(jxrDefaultLevel, count)
	}
	return s.checkProfileList(s.profiles, hasAlpha, count)
}

// checkProfileList 校验给定档次与级别列表是否满足主码流约束
// 入参: profiles 档次级别声明列表, alpha 是否存在内置或独立透明度, channels 有效分量数
// 返回: error 档次或级别约束错误
func (s *jxrStream) checkProfileList(profiles []jxrProfileLevel, alpha bool, channels int) error {
	for _, declaration := range profiles {
		if !s.profileAllowed(declaration.profile, alpha) {
			return fmt.Errorf("%w: JPEG XR profile constraints", ErrInvalid)
		}
		if err := s.checkLevel(declaration.level, channels); err != nil {
			return err
		}
	}
	return nil
}

// profileAllowed 检查标准已定义档次的像素格式约束，不将保留档次推断为未知图像格式
// 入参: profile 档次标识, alpha 是否存在内置或独立透明度平面
// 返回: bool 已知档次约束是否成立，保留档次交由实际语法及能力检查
func (s *jxrStream) profileAllowed(profile uint8, alpha bool) bool {
	h := &s.header
	if profile == 44 && (h.longWord || h.overlap == 2) {
		return false
	}
	switch profile {
	case 44, 55:
		if alpha {
			return false
		}
		if h.color == 0 {
			return h.depth == 0 || h.depth == 15 || h.depth == 1 || profile == 55 && (h.depth == 2 || h.depth == 3)
		}
		if h.color == 7 {
			return h.depth == 1 || h.depth == 8 || h.depth == 9 || h.depth == 10 || profile == 55 && (h.depth == 2 || h.depth == 3)
		}
		return false
	case 66:
		switch h.color {
		case 0:
			return !alpha
		case 4:
			return true
		case 7:
			return !alpha || h.depth != 8 && h.depth != 9 && h.depth != 10
		case 6:
			return s.primary.channels >= 3 && s.primary.channels <= 8
		case 8:
			return !alpha
		default:
			return false
		}
	default:
		return true
	}
}

// checkLevel 检查级别对应的尺寸、分块与缓冲约束，保留级别按不低于它的最近已定义级别处理
// 入参: level 声明或推定的级别, channels 所属声明规定的分量数，包含透明度而不包含填充
// 返回: error 尺寸、分块或缓冲约束错误
func (s *jxrStream) checkLevel(level uint8, channels int) error {
	limits := [...]jxrLevel{
		{4, 1 << 10, 1 << 4, 1 << 10, 1 << 22},
		{8, 1 << 11, 1 << 5, 1 << 11, 1 << 24},
		{16, 1 << 12, 1 << 6, 1 << 12, 1 << 26},
		{32, 1 << 13, 1 << 7, 1 << 12, 1 << 28},
		{64, 1 << 14, 1 << 8, 1 << 12, 1 << 30},
		{128, 1 << 16, 1 << 10, 1 << 12, 1 << 32},
		{255, 1 << 32, 1 << 12, 1 << 32, 0},
	}
	for _, limit := range limits {
		if level > limit.level {
			continue
		}
		h := &s.header
		width := h.width + uint64(h.left) + uint64(h.right)
		height := h.height + uint64(h.top) + uint64(h.bottom)
		if width > limit.dimension || height > limit.dimension || uint64(len(h.columns)-1) > limit.tiles || uint64(len(h.rows)-1) > limit.tiles {
			return fmt.Errorf("%w: JPEG XR level dimensions", ErrInvalid)
		}
		for _, boundaries := range [][]uint32{h.columns, h.rows} {
			for i := 1; i < len(boundaries); i++ {
				if uint64(boundaries[i]-boundaries[i-1])*16 >= limit.tileSize {
					return fmt.Errorf("%w: JPEG XR level tile dimensions", ErrInvalid)
				}
			}
		}
		if limit.buffer != 0 {
			size := width * height * uint64(channels)
			switch h.depth {
			case 0, 15:
				size = width * height / 8
			case 2, 3, 4:
				size *= 2
			case 6, 7:
				size *= 4
			case 8, 10:
				size = width * height * 2
			case 9:
				if h.color == 7 {
					size = width * height * 4
				} else {
					size *= 2
				}
			}
			if size >= limit.buffer {
				return fmt.Errorf("%w: JPEG XR level image buffer", ErrInvalid)
			}
		}
		return nil
	}
	return nil
}

// checkProfile 校验容器声明，缓冲分量数按像素格式计算而非沿用主码流分量数
// 入参: parser 共享解析预算, maxPixels 输出像素预算
// 返回: error 档次、级别或图像头错误
func (c *jxrContainer) checkProfile(parser *jxrParser, maxPixels uint64) error {
	if len(c.profiles) == 0 {
		return nil
	}
	retained := parser.used
	defer func() { parser.used = retained }()
	s, err := parser.streamView(c.image, c.imageInput, maxPixels)
	if err != nil {
		return err
	}
	count := c.format.channels
	if c.format.alpha {
		count++
	}
	return s.checkProfileList(c.profiles, c.format.alpha, count)
}
