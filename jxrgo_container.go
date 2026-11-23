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
	"slices"
	"sort"
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
	image           []byte
	alpha           []byte
	profile         []byte
	profiles        []jxrProfileLevel
	colorHint       ColorHint
	encoding        *ColorEncoding
	imageBands      uint32
	alphaBands      uint32
	missingBandTags uint8
	imageType       ImageType
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

// jxrOutputField 保存待写入目录字段、内联值及独立数据位置
type jxrOutputField struct {
	tag   uint16
	typ   uint16
	count uint32
	value uint32
	data  []byte
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

// jxrLevel 保存T.832表B.1的级别及尺寸、分块和缓冲上限
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
	d := jxrDirectory{data: data, order: binary.LittleEndian, entries: data[offset+2 : offset+2+count*12]}
	previous := -1
	for i := range count {
		entry := data[offset+2+i*12 : offset+2+(i+1)*12]
		tag := d.order.Uint16(entry)
		if int(tag) <= previous {
			return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory tag order", ErrInvalid)
		}
		previous = int(tag)
	}
	d.next = d.order.Uint32(data[offset+2+count*12:])
	if d.next&1 != 0 {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR next directory alignment", ErrInvalid)
	}
	return d, nil
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

// jxrStream 检查码流的完整文件范围，主码流的零长度表示延伸到文件末尾
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许零长度延伸
// 返回: []byte 码流视图, error 范围或资源错误
func (d *jxrDirectory) jxrStream(offsetTag, countTag uint16, remainder bool) ([]byte, error) {
	if d.input != nil {
		return nil, fmt.Errorf("%w: JPEG XR random stream requires bounded view", ErrUnsupported)
	}
	offset, size, err := d.jxrStreamRange(offsetTag, countTag, remainder)
	if err != nil {
		return nil, err
	}
	return d.data[offset : offset+size], nil
}

// jxrStreamRange 校验完整文件中的码流范围，不根据相邻资源推测长度
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许零长度延伸
// 返回: uint64 全文件偏移, uint64 码流长度, error 范围或资源错误
func (d *jxrDirectory) jxrStreamRange(offsetTag, countTag uint16, remainder bool) (uint64, uint64, error) {
	offset, err := d.jxrNumber(offsetTag, true, 0)
	if err != nil {
		return 0, 0, err
	}
	count, err := d.jxrNumber(countTag, true, 0)
	if err != nil {
		return 0, 0, err
	}
	if uint64(offset) > d.dataSize() {
		return 0, 0, fmt.Errorf("%w: JPEG XR codestream offset", ErrInvalid)
	}
	size := uint64(count)
	if count == 0 && remainder {
		size = d.dataSize() - uint64(offset)
	}
	if size < 8 || size > d.dataSize()-uint64(offset) {
		return 0, 0, fmt.Errorf("%w: JPEG XR codestream size", ErrInvalid)
	}
	return uint64(offset), size, nil
}

// jxrStreamView 取得零复制切片或独立有界随机视图，分配固定窗前检查共用预算
// 入参: offsetTag 偏移标签, countTag 长度标签, remainder 是否允许延伸至文件末尾
// 返回: []byte 可选切片, *jxrReaderAt 可选随机视图, error 范围、资源或读取错误
func (d *jxrDirectory) jxrStreamView(offsetTag, countTag uint16, remainder bool) ([]byte, *jxrReaderAt, error) {
	if d.input == nil {
		data, err := d.jxrStream(offsetTag, countTag, remainder)
		return data, nil, err
	}
	offset, size, err := d.jxrStreamRange(offsetTag, countTag, remainder)
	if err != nil {
		return nil, nil, err
	}
	if err := d.parser.reserve(jxrReaderAtWork); err != nil {
		return nil, nil, err
	}
	return nil, &jxrReaderAt{source: d.input.source, ctx: d.input.ctx, base: d.input.base + offset, size: size}, nil
}

// container 在共享解析预算内读取首个图像的元数据及码流范围，不复制资源字节
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
	return c, c.checkProfile(parser, maxPixels)
}

// directoryFields 读取附录A标签与码流位置，供已解析码流的入口共用
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
	imageType, err := d.jxrField(0xbc04, 1, 4, 4)
	if err != nil {
		return c, err
	}
	if imageType != nil {
		c.imageType = ImageType(d.order.Uint32(imageType) & 3)
	}
	c.image, c.imageInput, err = d.jxrStreamView(0xbcc0, 0xbcc1, true)
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
		c.alpha, c.alphaInput, err = d.jxrStreamView(0xbcc2, 0xbcc3, false)
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
			if field[0] > 3 || i == 1 && !c.separateAlpha() {
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
	c.profile, err = d.jxrField(0x8773, 0, math.MaxUint32, 7)
	if err != nil {
		return c, err
	}
	if c.profile != nil && len(c.profile) == 0 {
		return c, fmt.Errorf("%w: empty JPEG XR ICC profile", ErrInvalid)
	}
	if d.input != nil && len(c.profile) > 0 && len(c.profile) <= 4 {
		if err := parser.reserve(uint64(len(c.profile))); err != nil {
			return c, err
		}
		profile := make([]byte, len(c.profile))
		copy(profile, c.profile)
		c.profile = profile
	}
	colorSpace, err := d.jxrField(0xa001, 1, 2, 3)
	if err != nil {
		return c, err
	}
	if colorSpace != nil {
		c.colorHint = ColorHintUnspecified
		if d.order.Uint16(colorSpace) == uint16(ColorHintSRGB) {
			c.colorHint = ColorHintSRGB
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
		b := jxrBits{data: declarations, parser: parser}
		c.profiles, err = parser.profiles(&b, uint64(len(declarations)))
		if err != nil {
			return c, err
		}
	}
	return c, nil
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
	if err := checkJXRContainer(b, c, maxPixels); err != nil {
		b.err = err
		return
	}
	writeJXRContainerBytes(b, c, false, c.alpha != nil, nil)
}

// writeJXRContainerBytes 共用标准目录和元数据序列化，前缀模式不复制码流并返回主图长度字段的位置
// 入参: b 已校验的独占空缓冲, c 已校验元数据, prefix 是否只写主图之前的目录和元数据, separate 是否存在独立透明度, offsets 可选回填位置
func writeJXRContainerBytes(b *jxrBitWriter, c *jxrContainer, prefix, separate bool, offsets *jxrContainerOffsets) {
	var fields [17]jxrOutputField
	count := 0
	if c.profile != nil {
		fields[count] = jxrOutputField{tag: 0x8773, typ: 7, count: uint32(len(c.profile)), data: c.profile}
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
		size := uint64(len(field.data))
		if field.tag == 0xbc06 {
			size = uint64(field.count)
		}
		if size > 4 {
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
		if field.tag == 0xbc06 {
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

// checkJXRContainer 校验文件目录的尺寸、标准格式标识、数值元数据及码流绑定
// 入参: b 输出器及共享剩余预算, c 容器数据, maxPixels 输出像素上限
// 返回: error 参数、资源或码流错误
func checkJXRContainer(b *jxrBitWriter, c *jxrContainer, maxPixels uint64) error {
	if c != nil && (c.imageInput != nil || c.alphaInput != nil) {
		return fmt.Errorf("%w: JPEG XR output requires encoded byte buffers", ErrUnsupported)
	}
	if err := checkJXRContainerMetadataLayout(c, maxPixels, c != nil && c.separateAlpha()); err != nil {
		return err
	}
	if uint64(cap(b.data)) > b.maxMemory {
		return &LimitError{Resource: "JPEG XR container validation buffer memory", Limit: b.maxMemory, Required: uint64(cap(b.data))}
	}
	available := b.maxMemory - uint64(cap(b.data))
	if b.budget != nil {
		if b.budget.used > b.budget.limit {
			return &LimitError{Resource: "JPEG XR container validation memory", Limit: b.budget.limit, Required: b.budget.used}
		}
		available = min(available, b.budget.limit-b.budget.used)
	}
	parser := jxrParser{ctx: b.ctx, limit: available}
	source, err := parser.containerSource(c, maxPixels)
	if err != nil {
		return err
	}
	channels := c.format.channels
	if c.format.alpha {
		channels++
	}
	return source.primary.checkProfileList(c.profiles, c.format.alpha, channels)
}

// checkJXRContainerMetadataLayout 校验独立透明度布局的目录声明，不用虚构压缩字节表示待写出的平面
// 入参: c 目录元数据, maxPixels 像素上限, separate 是否存在独立透明度码流
// 返回: error 元数据、格式或容量错误
func checkJXRContainerMetadataLayout(c *jxrContainer, maxPixels uint64, separate bool) error {
	if c == nil || c.width == 0 || c.height == 0 || c.transform > 7 || c.imageBands > 3 || c.alphaBands > 3 || !separate && c.alphaBands != 0 || c.imageType&^3 != 0 {
		return fmt.Errorf("%w: JPEG XR output container fields", ErrInvalid)
	}
	if c.width > math.MaxUint32 || c.height > math.MaxUint32 || uint64(len(c.image)) > math.MaxUint32 || uint64(len(c.alpha)) > math.MaxUint32 || uint64(len(c.profile)) > math.MaxUint32 {
		return fmt.Errorf("%w: JPEG XR output container field capacity", ErrLimit)
	}
	if err := checkJXRDimensions(c.width, c.height, maxPixels); err != nil {
		return err
	}
	if c.profile != nil && len(c.profile) == 0 || c.alpha != nil && len(c.alpha) == 0 {
		return fmt.Errorf("%w: JPEG XR output empty container data", ErrInvalid)
	}
	f, err := readJXRPixelFormat(c.pixelFormat[:])
	if err != nil {
		return err
	}
	if f.name == "BlackWhite" && c.format.depth == 15 {
		f.depth = 15
	}
	if c.format != f {
		return fmt.Errorf("%w: JPEG XR output container pixel format", ErrInvalid)
	}
	if separate && !f.alpha {
		return fmt.Errorf("%w: JPEG XR separate alpha pixel format", ErrInvalid)
	}
	if err := checkJXRColorMetadata(f, c.colorHint, c.encoding); err != nil {
		return err
	}
	for _, resolution := range [2]float64{c.dpi.X, c.dpi.Y} {
		value := float64(float32(resolution))
		if !geometryFinite(resolution) || resolution < 0 || resolution != 0 && (!geometryFinite(value) || value <= 0) {
			return fmt.Errorf("%w: JPEG XR output resolution", ErrInvalid)
		}
	}
	return checkJXRProfileDeclarations(c.profiles)
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

// NewDecoderAt 按需读取JPEG XR附录A文件或独立码流，不持有完整压缩输入
// size指定从零开始的输入字节范围，仍受MaxInputBytes约束；ICC等元数据在预算内独立复制
// r须在解码器存续期间保持稳定且可读取，Close不关闭r；解码器串行访问自己的固定读取窗
// 内存预算包含固定窗、头部和索引，后续区域解码只读取所需包及最多4096字节的预取
// 构造时另检查常驻存储与单份Info副本的总容量，不计调用方持有的既往返回值
// 入参: ctx 构造阶段上下文，nil采用Background, r 随机输入, size 码流长度, opts 解码资源限制
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
// 入参: ctx 已规范化上下文, r 随机输入, size 码流长度, opts 解码选项
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
	legacyHighpass := false
	if opts != nil {
		if opts.Workers < 0 {
			return nil, fmt.Errorf("%w: negative decode workers", ErrInvalid)
		}
		limits = opts.Limits.normalized()
		workers = opts.Workers
		legacyHighpass = opts.LegacyHighpassPrediction
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
		decoder, err := fileDecoderAt(ctx, input, &parser, limits)
		if err != nil {
			return nil, err
		}
		decoder.setWorkers(workers)
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
	info, err := parser.info(&source)
	if err != nil {
		return nil, err
	}
	decoder := &Decoder{data: []byte("WMPHOTO\x00"), source: source, info: info, limits: limits, legacyHighpassPrediction: legacyHighpass}
	decoder.setWorkers(workers)
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
	if d.source.primary.input != nil {
		work += d.source.primary.input.storageWork()
	}
	if d.source.alpha != nil && d.source.alpha.input != nil && d.source.alpha.input != d.source.primary.input {
		work += d.source.alpha.input.storageWork()
	}
	return work
}

// storageWork 计算读取窗所占预算
// 返回: uint64 常驻读取资源容量
func (r *jxrReaderAt) storageWork() uint64 {
	return jxrReaderAtWork
}

// bindInputContext 为独占操作设置随机读取上下文，nil释放已经结束的上下文引用
// 入参: ctx 当前操作上下文或nil
func (d *Decoder) bindInputContext(ctx context.Context) {
	if d.fileInput != nil {
		d.fileInput.ctx = ctx
	}
	d.source.bindLogicalContext(ctx)
}

// bindLogicalContext 为逻辑样本视图绑定或释放操作上下文，不修改共享文件的上下文
// 入参: ctx 当前操作上下文或nil
func (source *jxrRasterSource) bindLogicalContext(ctx context.Context) {
	if source.primary.input != nil {
		source.primary.input.ctx = ctx
	}
	if source.alpha != nil && source.alpha.input != nil {
		source.alpha.input.ctx = ctx
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
	previous := -1
	for index := range count {
		tag := int(binary.LittleEndian.Uint16(directory.entries[index*12:]))
		if tag <= previous {
			return jxrDirectory{}, fmt.Errorf("%w: JPEG XR directory tag order", ErrInvalid)
		}
		previous = tag
	}
	if directory.next&1 != 0 {
		return jxrDirectory{}, fmt.Errorf("%w: JPEG XR next directory alignment", ErrInvalid)
	}
	return directory, nil
}

// fileDecoderAt 解析附录A首目录及双流有界视图，不持有目录临时缓冲或完整压缩文件
// 入参: ctx 构造上下文, input 已预留固定窗的完整文件视图, parser 共享解析预算, limits 解码上限
// 返回: *Decoder 独立元数据及随机源, error 容器、码流、读取或资源错误
func fileDecoderAt(ctx context.Context, input *jxrReaderAt, parser *jxrParser, limits Limits) (*Decoder, error) {
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
	info, err := parser.info(&source)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	decoder := &Decoder{data: header[:], source: source, info: info, limits: limits, fileInput: input, inputExtra: uint64(cap(container.profile)), legacyHighpassPrediction: parser.legacyHighpassPrediction}
	decoder.bindInputContext(nil)
	if err := decoder.checkInfoCopy(); err != nil {
		return nil, err
	}
	return decoder, nil
}

// imageDirectoryAt 取得已经通过文件头校验的目录，随机输入分配受调用方剩余容量约束
// 入参: ctx 当前操作上下文, offset 全文件偏移, maximum 本次目录可用临时容量
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
	return jxrRasterSource{format: FormatCodestream, primary: s, container: jxrContainer{width: h.width, height: h.height, format: f, transform: uint32(h.transform), image: data, imageBands: uint32(s.primary.bands)}}
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

// opacityFormat 返回独立单分量透明度的原生格式，交织透明度沿用对应颜色码流的编码
// 返回: jxrPixelFormat 透明度的数值及通道格式
func (s *jxrRasterSource) opacityFormat() jxrPixelFormat {
	if s.alpha != nil && s.alpha.alpha == nil {
		return jxrCodedFormat(s.alpha)
	}
	return s.container.format
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
	info := Info{Format: s.format, ImageType: c.imageType, Bounds: image.Rect(0, 0, width, height), PixelFormat: c.format.name, Channels: c.format.channels, ColorSpace: jxrColorSpace(c.format.color), Alpha: c.format.alpha, Premultiplied: c.format.premultiply, Linear: c.format.number != jxrUnsigned, Resolution: dpi, ICCProfile: c.profile}
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
		chromaX, chromaY := p.chromaPhase(index)
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
	format := c.format
	if index == format.channels && s.alpha != nil && s.alpha.alpha == nil {
		format = s.opacityFormat()
		index = 0
	}
	info := jxrSampleInfo(format, index)
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

// jxrWritableProfileLevel 只允许附录B明列的输出档位和级别，解码端保留值解释不授权编码端写出
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

// profiles 按8.6及A.7.22预扫描声明，以输入范围、可寻址容量和共享预算约束分配
// 去重键包含保留位但不含终止位，成功仅保留结果容量，失败归还全部新增预留
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
	if err := parser.check(); err != nil {
		return nil, err
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
	count := s.primary.channels
	if s.alpha != nil {
		count++
	}
	if len(s.profiles) == 0 {
		return s.checkLevel(jxrDefaultLevel, count)
	}
	return s.checkProfileList(s.profiles, s.alpha != nil, count)
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
		case 4, 7:
			return true
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

// checkLevel 检查表B.1及表B.2的级别约束，保留级别按标准取不低于它的最近已定义级别
// 入参: level 声明或推定的级别, channels 所属声明规定的分量数，包含透明度而不包含填充
// 返回: error 尺寸、分块、头部或缓冲约束错误
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
		if width > limit.dimension || height > limit.dimension || uint64(len(h.columns)-1) > limit.tiles || uint64(len(h.rows)-1) > limit.tiles || limit.level <= 128 && !h.short {
			return fmt.Errorf("%w: JPEG XR level dimensions or header", ErrInvalid)
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
