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
	"slices"
)

// jxrCoefficientLimit 限制有符号整数重建的计算容量，为后续变换保留运算余量
const jxrCoefficientLimit int64 = 1 << 50

// jxrOutputDepth 保存允许的逐输出颜色位深组合
var jxrOutputDepth = [9]uint16{
	1<<0 | 1<<1 | 1<<2 | 1<<3 | 1<<4 | 1<<6 | 1<<7 | 1<<15,
	1 << 1,
	1<<1 | 1<<2 | 1<<9,
	1<<1 | 1<<2 | 1<<3 | 1<<9,
	1<<1 | 1<<2,
	1<<1 | 1<<2,
	1<<1 | 1<<2,
	1<<1 | 1<<2 | 1<<3 | 1<<4 | 1<<6 | 1<<7 | 1<<8 | 1<<9 | 1<<10,
	1 << 1,
}

// jxrFirstWriteCode 保存首系数前缀表的反向查找结果
var jxrFirstWriteCode = [5][16]jxrWriteCode{
	jxrMakeWriteVLC(jxrFirstCode[0]), jxrMakeWriteVLC(jxrFirstCode[1]),
	jxrMakeWriteVLC(jxrFirstCode[2]), jxrMakeWriteVLC(jxrFirstCode[3]), jxrMakeWriteVLC(jxrFirstCode[4]),
}

// jxrIndexWriteCode 保存后续系数前缀表的反向查找结果
var jxrIndexWriteCode = [4][16]jxrWriteCode{
	jxrMakeWriteVLC(jxrIndexCode[0]), jxrMakeWriteVLC(jxrIndexCode[1]),
	jxrMakeWriteVLC(jxrIndexCode[2]), jxrMakeWriteVLC(jxrIndexCode[3]),
}

// jxrProfileLevel 保存声明的档次、级别及保留值，不用未知声明替代实际能力检查
type jxrProfileLevel struct {
	profile  uint8
	level    uint8
	reserved uint16
}

// jxrStream 保存完整头部和平面的解析结果，包偏移仍相对于当前独立码流
type jxrStream struct {
	header    jxrHeader
	primary   jxrPlane
	alpha     *jxrPlane
	profiles  []jxrProfileLevel
	packets   []uint64
	dataStart uint64
	input     *jxrReaderAt
	workers   int
}

// jxrParser 在解析分配发生前共用容量与取消状态，不把输入缓冲区容量误作可用空间
type jxrParser struct {
	ctx                      context.Context
	limit, used              uint64
	legacyHighpassPrediction bool
}

// jxrHeader 保存输出尺寸、宏块边界、窗口及码流控制标志
type jxrHeader struct {
	width       uint64
	height      uint64
	columns     []uint32
	rows        []uint32
	top         uint32
	left        uint32
	bottom      uint32
	right       uint32
	color       uint8
	depth       uint8
	overlap     uint8
	transform   uint8
	hardTiles   bool
	frequency   bool
	index       bool
	short       bool
	longWord    bool
	trim        bool
	alpha       bool
	redBlue     bool
	premultiply bool
	planeStart  int
}

// jxrPlane 保存单个颜色或透明度平面的格式、精度和图像级量化参数
type jxrPlane struct {
	color                    uint8
	legacyHighpassPrediction bool
	channels                 int
	bands                    uint8
	scaled                   bool
	chromaX                  uint8
	chromaY                  uint8
	itemChroma               [2][2]uint8
	hasItemChroma            bool
	shift                    uint8
	mantissa                 uint8
	exponent                 int8
	dcUniform                bool
	lpUniform                bool
	hpUniform                bool
	dcQuant                  []uint8
	lpQuant                  []uint8
	hpQuant                  []uint8
}

// jxrWriteCode 保存一个符号对应的高位优先前缀及其长度
type jxrWriteCode struct {
	value uint8
	bits  uint8
}

// headerWork 计算已解析头部、索引及量化参数容量，不包含输入字节或系数图像
// 返回: uint64 头部存储预算
func (s *jxrStream) headerWork() uint64 {
	work := uint64(1024) + uint64(cap(s.header.columns)+cap(s.header.rows))*4 + uint64(cap(s.profiles))*4 + uint64(cap(s.packets))*8 + s.primary.quantWork()
	if s.alpha != nil {
		work += s.alpha.quantWork() + 256
	}
	return work
}

// stream 在共用预算内解析平面、量化、索引及声明，分配前检查容量
// 入参: data 独立码流, maxPixels 输出像素预算
// 返回: jxrStream 解析结果, error 数据、能力或资源错误
func (parser *jxrParser) stream(data []byte, maxPixels uint64) (jxrStream, error) {
	return parser.streamReader(jxrBits{data: data, parser: parser}, maxPixels)
}

// streamView 解析目录已校验的切片或随机码流，两个来源共用相同位语法与容量检查
// 入参: data 可选完整码流切片, input 可选随机视图, maxPixels 输出像素预算
// 返回: jxrStream 已解析码流, error 数据、读取、能力或资源错误
func (parser *jxrParser) streamView(data []byte, input *jxrReaderAt, maxPixels uint64) (jxrStream, error) {
	return parser.streamReader(jxrBits{data: data, input: input, parser: parser}, maxPixels)
}

// streamReader 共用切片及随机输入解析头部、索引和平面，包边界按逻辑范围校验而不下载包数据
// 入参: b 独立码流起点读取器, maxPixels 输出像素预算
// 返回: jxrStream 解析结果, error 数据、读取、能力或资源错误
func (parser *jxrParser) streamReader(b jxrBits, maxPixels uint64) (jxrStream, error) {
	s, err := parser.imageHeaders(&b, maxPixels)
	if err != nil {
		return s, err
	}
	return parser.streamIndex(b, s)
}

// streamIndex 从明确的索引起点解析后续声明和包偏移，容器扩展字节仍计入逻辑偏移
// 入参: b 索引起点读取器, s 已解析图像及平面头
// 返回: jxrStream 完整码流索引, error 数据、读取、能力或资源错误
func (parser *jxrParser) streamIndex(b jxrBits, s jxrStream) (jxrStream, error) {
	var err error
	count := uint64(len(s.header.columns)-1) * uint64(len(s.header.rows)-1)
	if s.header.frequency {
		count *= uint64(4 - s.primary.bands)
	}
	if count > uint64(^uint(0)>>1)/8 {
		return s, &LimitError{Resource: "JPEG XR packet index budget", Limit: uint64(^uint(0)>>1) / 8, Required: count}
	}
	if s.header.index {
		start := b.read(16)
		if b.err != nil {
			return s, b.err
		}
		if start != 1 {
			return s, fmt.Errorf("%w: JPEG XR index start code", ErrInvalid)
		}
		if count > b.length()-b.bit/8 {
			return s, fmt.Errorf("%w: JPEG XR truncated packet index", ErrInvalid)
		}
	}
	if err := parser.reserve(count * 8); err != nil {
		return s, err
	}
	s.packets = make([]uint64, int(count))
	if s.header.index {
		for i := range s.packets {
			if i&1023 == 0 {
				if err := parser.check(); err != nil {
					return s, err
				}
			}
			s.packets[i] = b.variableWord()
			if b.err != nil {
				return s, b.err
			}
		}
	}
	subsequent := b.variableWord()
	if b.err != nil {
		return s, b.err
	}
	if subsequent > b.length()-b.bit/8 {
		return s, fmt.Errorf("%w: JPEG XR subsequent header bytes", ErrInvalid)
	}
	end := b.bit/8 + subsequent
	if subsequent != 0 {
		s.profiles, err = parser.profiles(&b, end)
		if err != nil {
			return s, err
		}
	}
	s.dataStart = end
	if len(s.packets) == 1 && s.packets[0] != 0 {
		return s, fmt.Errorf("%w: JPEG XR single packet offset", ErrInvalid)
	}
	for index, offset := range s.packets {
		if index&1023 == 0 {
			if err := parser.check(); err != nil {
				return s, err
			}
		}
		if offset > b.length()-s.dataStart || b.length()-s.dataStart-offset < 4 {
			return s, fmt.Errorf("%w: JPEG XR packet offset", ErrInvalid)
		}
	}
	if err := s.checkProfile(); err != nil {
		return s, err
	}
	return s, nil
}

// reserve 在新增解析存储分配前检查剩余预算及累计地址容量，失败时不增加已保留容量
// 入参: amount 新增活动存储字节数
// 返回: error 取消或资源错误
func (p *jxrParser) reserve(amount uint64) error {
	if p == nil {
		return nil
	}
	if err := p.check(); err != nil {
		return err
	}
	limit := min(p.limit, uint64(^uint(0)))
	if p.used > limit || amount > limit-p.used {
		return &LimitError{Resource: "JPEG XR parsing memory", Limit: limit, Required: jxrSaturatedAdd(p.used, amount)}
	}
	p.used += amount
	return nil
}

// check 在较长索引及声明解析循环中响应取消，不修改内存预算
// 返回: error Context取消错误
func (p *jxrParser) check() error {
	if p != nil && p.ctx != nil {
		return p.ctx.Err()
	}
	return nil
}

// info 在构造分量信息及复制颜色声明前检查共享解析预算
// 入参: source 已绑定的图像源
// 返回: Info 原生图像信息, error 取消或资源错误
func (p *jxrParser) info(source *jxrRasterSource) (Info, error) {
	work := uint64(source.container.format.channels) * 128
	if source.container.format.alpha {
		work += 128
	}
	if source.container.encoding != nil {
		work += 4
	}
	if err := p.reserve(work); err != nil {
		return Info{}, err
	}
	info := source.info()
	if err := p.check(); err != nil {
		return Info{}, err
	}
	return info, nil
}

// headerReader 从有界位输入读取图像头，切片及随机输入共用相同标准字段与分配预检
// 入参: b 独立码流起点读取器, maxPixels 输出像素预算
// 返回: jxrHeader 图像头, error 数据、版本、读取或资源错误
func (parser *jxrParser) headerReader(b jxrBits, maxPixels uint64) (jxrHeader, error) {
	var h jxrHeader
	first, second := b.read(32), b.read(32)
	if b.err != nil {
		return h, b.err
	}
	if first != 0x574d5048 || second != 0x4f544f00 {
		return h, fmt.Errorf("%w: JPEG XR codestream signature", ErrInvalid)
	}
	version := b.read(4)
	h.hardTiles = b.flag()
	_ = b.read(3)
	tiled := b.flag()
	h.frequency = b.flag()
	h.transform = uint8(b.read(3))
	h.index = b.flag()
	h.overlap = uint8(b.read(2))
	h.short = b.flag()
	h.longWord = b.flag()
	window := b.flag()
	h.trim = b.flag()
	_ = b.read(1)
	h.redBlue = b.flag()
	h.premultiply = b.flag()
	h.alpha = b.flag()
	h.color, h.depth = uint8(b.read(4)), uint8(b.read(4))
	dimensionBits, tileBits := uint(32), uint(16)
	if h.short {
		dimensionBits, tileBits = 16, 8
	}
	h.width, h.height = uint64(b.read(dimensionBits))+1, uint64(b.read(dimensionBits))+1
	columns, rows := uint32(1), uint32(1)
	if tiled {
		columns, rows = b.read(12)+1, b.read(12)+1
	}
	if b.err != nil {
		return h, b.err
	}
	if version != 1 {
		return h, fmt.Errorf("%w: JPEG XR codestream version", ErrUnsupported)
	}
	if h.overlap == 3 || h.color > 8 || h.depth == 5 || h.depth >= 11 && h.depth <= 14 {
		return h, fmt.Errorf("%w: JPEG XR header format", ErrUnsupported)
	}
	if jxrOutputDepth[h.color]&(1<<h.depth) == 0 {
		return h, fmt.Errorf("%w: JPEG XR output color and depth combination", ErrInvalid)
	}
	if h.color != 7 || h.depth != 8 && h.depth != 9 && h.depth != 10 {
		h.redBlue = false
	}
	if !h.alpha {
		h.premultiply = false
	}
	if err := checkJXRDimensions(h.width, h.height, maxPixels); err != nil {
		return h, err
	}
	if (h.frequency || columns > 1 || rows > 1) && !h.index {
		return h, fmt.Errorf("%w: JPEG XR required tile index", ErrInvalid)
	}
	if err := parser.reserve(1024 + uint64(columns+rows+2)*4); err != nil {
		return h, err
	}
	h.columns, h.rows = make([]uint32, columns+1), make([]uint32, rows+1)
	for i := uint32(1); i < columns; i++ {
		if i&63 == 0 {
			if err := parser.check(); err != nil {
				return h, err
			}
		}
		width := b.read(tileBits)
		if width == 0 {
			return h, fmt.Errorf("%w: JPEG XR empty tile column", ErrInvalid)
		}
		h.columns[i] = h.columns[i-1] + width
	}
	for i := uint32(1); i < rows; i++ {
		if i&63 == 0 {
			if err := parser.check(); err != nil {
				return h, err
			}
		}
		height := b.read(tileBits)
		if height == 0 {
			return h, fmt.Errorf("%w: JPEG XR empty tile row", ErrInvalid)
		}
		h.rows[i] = h.rows[i-1] + height
	}
	if window {
		h.top, h.left, h.bottom, h.right = b.read(6), b.read(6), b.read(6), b.read(6)
	} else {
		h.bottom, h.right = uint32((16-h.height%16)%16), uint32((16-h.width%16)%16)
	}
	if b.err != nil {
		return h, b.err
	}
	extendedWidth := h.width + uint64(h.left) + uint64(h.right)
	extendedHeight := h.height + uint64(h.top) + uint64(h.bottom)
	if extendedWidth%16 != 0 || extendedHeight%16 != 0 {
		return h, fmt.Errorf("%w: JPEG XR macroblock window alignment", ErrInvalid)
	}
	if err := checkJXRDimensions(extendedWidth, extendedHeight, ^uint64(0)); err != nil {
		return h, err
	}
	if (h.color == 1 || h.color == 2) && (h.width%2 != 0 || h.left%2 != 0 || h.right%2 != 0) || h.color == 1 && (h.height%2 != 0 || h.top%2 != 0 || h.bottom%2 != 0) {
		return h, fmt.Errorf("%w: JPEG XR subsampled window alignment", ErrInvalid)
	}
	h.columns[columns], h.rows[rows] = uint32(extendedWidth/16), uint32(extendedHeight/16)
	if h.columns[columns-1] >= h.columns[columns] || h.rows[rows-1] >= h.rows[rows] {
		return h, fmt.Errorf("%w: JPEG XR tile partition extent", ErrInvalid)
	}
	h.planeStart = int(b.bit / 8)
	return h, nil
}

// imageHeaders 共用图像及平面头解析，不读取包索引、后续字节或像素包
// 入参: b 有界读取器, maxPixels 输出像素预算
// 返回: jxrStream 头部和平面描述, error 数据、读取、能力、资源或取消错误
func (parser *jxrParser) imageHeaders(b *jxrBits, maxPixels uint64) (jxrStream, error) {
	var s jxrStream
	var err error
	s.input = b.input
	s.header, err = parser.headerReader(*b, maxPixels)
	if err != nil {
		return s, err
	}
	b.bit, b.parser = uint64(s.header.planeStart)*8, parser
	s.primary, err = readJXRPlane(b, &s.header, false)
	if err != nil {
		return s, err
	}
	s.primary.legacyHighpassPrediction = parser.legacyHighpassPrediction
	if s.header.alpha {
		if err := parser.reserve(256); err != nil {
			return s, err
		}
		alpha, err := readJXRPlane(b, &s.header, true)
		if err != nil {
			return s, err
		}
		if alpha.bands < s.primary.bands {
			return s, fmt.Errorf("%w: JPEG XR alpha frequency bands", ErrInvalid)
		}
		alpha.legacyHighpassPrediction = parser.legacyHighpassPrediction
		s.alpha = &alpha
	}
	return s, nil
}

// checkJXRImageHeaders 校验图像和平面描述，不要求已有包索引或档次声明
// 入参: s 待输出的头部和平面, maxPixels 输出像素预算
// 返回: error 参数、布局、资源或平面对应关系错误
func checkJXRImageHeaders(s *jxrStream, maxPixels uint64) error {
	if s == nil {
		return fmt.Errorf("%w: JPEG XR output stream header", ErrInvalid)
	}
	h := &s.header
	if err := checkJXRHeader(h, maxPixels); err != nil {
		return err
	}
	if err := checkJXRPlaneHeader(h, &s.primary, false); err != nil {
		return err
	}
	if h.alpha != (s.alpha != nil) {
		return fmt.Errorf("%w: JPEG XR output alpha plane presence", ErrInvalid)
	}
	if s.alpha != nil {
		if err := checkJXRPlaneHeader(h, s.alpha, true); err != nil {
			return err
		}
		if s.alpha.bands < s.primary.bands {
			return fmt.Errorf("%w: JPEG XR output alpha bands", ErrInvalid)
		}
	}
	return nil
}

// writeJXRHeader 写入规范图像头及必要窗口字段，保留字段采用标准规定值
// 入参: b 位写入器, h 图像头, maxPixels 输出像素上限
func writeJXRHeader(b *jxrBitWriter, h *jxrHeader, maxPixels uint64) {
	if b.err != nil {
		return
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return
	}
	if err := checkJXRHeader(h, maxPixels); err != nil {
		b.err = err
		return
	}
	if b.bit%8 != 0 {
		b.err = fmt.Errorf("%w: JPEG XR output header alignment", ErrInvalid)
		return
	}
	for _, value := range []byte("WMPHOTO\x00") {
		b.write(uint32(value), 8)
	}
	b.write(1, 4)
	b.flag(h.hardTiles)
	b.write(1, 3)
	tiled := len(h.columns) > 2 || len(h.rows) > 2
	window := h.top != 0 || h.left != 0 || h.bottom != uint32((16-h.height%16)%16) || h.right != uint32((16-h.width%16)%16)
	b.flag(tiled)
	b.flag(h.frequency)
	b.write(uint32(h.transform), 3)
	b.flag(h.index)
	b.write(uint32(h.overlap), 2)
	b.flag(h.short)
	b.flag(h.longWord)
	b.flag(window)
	b.flag(h.trim)
	b.flag(false)
	b.flag(h.redBlue)
	b.flag(h.premultiply)
	b.flag(h.alpha)
	b.write(uint32(h.color), 4)
	b.write(uint32(h.depth), 4)
	dimensionBits, tileBits := uint(32), uint(16)
	if h.short {
		dimensionBits, tileBits = 16, 8
	}
	b.write(uint32(h.width-1), dimensionBits)
	b.write(uint32(h.height-1), dimensionBits)
	if tiled {
		b.write(uint32(len(h.columns)-2), 12)
		b.write(uint32(len(h.rows)-2), 12)
	}
	for _, boundary := range [][]uint32{h.columns, h.rows} {
		for n := 1; n < len(boundary)-1; n++ {
			b.write(boundary[n]-boundary[n-1], tileBits)
		}
	}
	if window {
		for _, margin := range [4]uint32{h.top, h.left, h.bottom, h.right} {
			b.write(margin, 6)
		}
	}
}

// checkJXRHeader 校验待写入头部的字段范围、宏块窗口及分块分区
// 入参: h 图像头, maxPixels 输出像素上限
// 返回: error 参数、布局或资源错误
func checkJXRHeader(h *jxrHeader, maxPixels uint64) error {
	if h == nil || h.width == 0 || h.height == 0 || h.color > 8 || h.depth > 15 || h.overlap > 2 || h.transform > 7 {
		return fmt.Errorf("%w: JPEG XR output header fields", ErrInvalid)
	}
	if jxrOutputDepth[h.color]&(1<<h.depth) == 0 || h.premultiply && !h.alpha || h.redBlue && (h.color != 7 || h.depth < 8 || h.depth > 10) {
		return fmt.Errorf("%w: JPEG XR output header format", ErrInvalid)
	}
	if err := checkJXRDimensions(h.width, h.height, maxPixels); err != nil {
		return err
	}
	if h.short && (h.width > 1<<16 || h.height > 1<<16) || h.top > 63 || h.left > 63 || h.bottom > 63 || h.right > 63 {
		return fmt.Errorf("%w: JPEG XR output header field width", ErrInvalid)
	}
	if len(h.columns) < 2 || len(h.rows) < 2 || len(h.columns) > 4097 || len(h.rows) > 4097 {
		return fmt.Errorf("%w: JPEG XR output tile partition count", ErrInvalid)
	}
	if (h.frequency || len(h.columns) > 2 || len(h.rows) > 2) && !h.index {
		return fmt.Errorf("%w: JPEG XR output required tile index", ErrInvalid)
	}
	width, height := h.width+uint64(h.left)+uint64(h.right), h.height+uint64(h.top)+uint64(h.bottom)
	if width%16 != 0 || height%16 != 0 {
		return fmt.Errorf("%w: JPEG XR output macroblock window alignment", ErrInvalid)
	}
	if err := checkJXRDimensions(width, height, ^uint64(0)); err != nil {
		return err
	}
	if (h.color == 1 || h.color == 2) && (h.width%2 != 0 || h.left%2 != 0 || h.right%2 != 0) || h.color == 1 && (h.height%2 != 0 || h.top%2 != 0 || h.bottom%2 != 0) {
		return fmt.Errorf("%w: JPEG XR output subsampled window alignment", ErrInvalid)
	}
	maximum := uint32(65535)
	if h.short {
		maximum = 255
	}
	for axis, boundary := range [][]uint32{h.columns, h.rows} {
		extent := width / 16
		if axis != 0 {
			extent = height / 16
		}
		if boundary[0] != 0 || uint64(boundary[len(boundary)-1]) != extent {
			return fmt.Errorf("%w: JPEG XR output tile partition extent", ErrInvalid)
		}
		for n := 1; n < len(boundary); n++ {
			if boundary[n] <= boundary[n-1] || n < len(boundary)-1 && boundary[n]-boundary[n-1] > maximum {
				return fmt.Errorf("%w: JPEG XR output tile partition field", ErrInvalid)
			}
		}
	}
	return nil
}

// writeJXRStreamHeader 写入图像、主平面、透明度、包索引及档次级别声明，不写入像素包
// 入参: b 位写入器, s 完整头部及相对包偏移, maxPixels 输出像素上限, wideIndex 是否采用固定六十四位索引表示
// 返回: uint64 首个索引字节的位置，无索引或失败时为零
func writeJXRStreamHeader(b *jxrBitWriter, s *jxrStream, maxPixels uint64, wideIndex bool) uint64 {
	if b.err != nil {
		return 0
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return 0
	}
	if err := checkJXRStreamOutput(s, maxPixels); err != nil {
		b.err = err
		return 0
	}
	indexAt := uint64(0)
	h := &s.header
	writeJXRHeader(b, h, maxPixels)
	writeJXRPlane(b, h, &s.primary, false)
	if s.alpha != nil {
		writeJXRPlane(b, h, s.alpha, true)
	}
	if h.index {
		b.write(1, 16)
		indexAt = b.bit / 8
		for _, offset := range s.packets {
			if wideIndex {
				b.write(0xfc, 8)
				b.write(uint32(offset>>32), 32)
				b.write(uint32(offset), 32)
			} else {
				b.variableWord(offset)
			}
		}
	}
	b.variableWord(uint64(len(s.profiles)) * 4)
	for n, profile := range s.profiles {
		b.write(uint32(profile.profile), 8)
		b.write(uint32(profile.level), 8)
		b.write(0, 15)
		b.flag(n == len(s.profiles)-1)
	}
	if b.err != nil {
		return 0
	}
	return indexAt
}

// checkJXRStreamOutput 校验输出头部、平面、包索引及档次级别，不分配输出缓冲
// 入参: s 完整码流头部, maxPixels 输出像素上限
// 返回: error 参数、资源或声明错误
func checkJXRStreamOutput(s *jxrStream, maxPixels uint64) error {
	if err := checkJXRImageHeaders(s, maxPixels); err != nil {
		return err
	}
	h := &s.header
	count := uint64(len(h.columns)-1) * uint64(len(h.rows)-1)
	if h.frequency {
		count *= uint64(4 - s.primary.bands)
	}
	if count > uint64(^uint(0)>>1)/8 {
		return fmt.Errorf("%w: JPEG XR output packet count", ErrLimit)
	}
	if uint64(len(s.packets)) != count || count == 1 && s.packets[0] != 0 {
		return fmt.Errorf("%w: JPEG XR output packet index", ErrInvalid)
	}
	if err := checkJXRProfileDeclarations(s.profiles); err != nil {
		return err
	}
	return s.checkProfile()
}

// checkJXRProfileDeclarations 检查显式输出的档次级别及重复声明，不采用缺省推导
// 入参: profiles 待输出的声明，可为空
// 返回: error 声明或容量错误
func checkJXRProfileDeclarations(profiles []jxrProfileLevel) error {
	if len(profiles) > 4096 {
		return fmt.Errorf("%w: JPEG XR output profile count", ErrLimit)
	}
	var seen [1024]uint64
	for _, profile := range profiles {
		key := uint16(profile.profile)<<8 | uint16(profile.level)
		mask := uint64(1) << (key % 64)
		if !jxrWritableProfileLevel(profile) || seen[key/64]&mask != 0 {
			return fmt.Errorf("%w: JPEG XR output profile declaration", ErrInvalid)
		}
		seen[key/64] |= mask
	}
	return nil
}

// chromaPhase 返回指定色度分量的四分之一样本相位，优先采用项目中已核实的相位
// 入参: component 色度分量索引
// 返回: uint8 水平色度相位, uint8 垂直色度相位
func (p *jxrPlane) chromaPhase(component int) (uint8, uint8) {
	if p.hasItemChroma {
		phase := p.itemChroma[component-1]
		return phase[0], phase[1]
	}
	return p.chromaX, p.chromaY
}

// chromaPhasesKnown 检查两个色度分量的全部子采样轴是否有明确相位
// 返回: bool 色度采样相位是否均已明确
func (p *jxrPlane) chromaPhasesKnown() bool {
	for _, component := range [...]int{1, 2} {
		x, y := p.chromaPhase(component)
		if x > 4 || p.color == 1 && y > 4 {
			return false
		}
	}
	return true
}

// readJXRPlane 读取有界图像平面头，透明度平面独立校验而不沿用主平面的颜色参数
// 入参: b 码流读取器, h 所属图像头, alpha 是否为透明度平面
// 返回: jxrPlane 平面参数, error 数据、能力或资源错误
func readJXRPlane(b *jxrBits, h *jxrHeader, alpha bool) (jxrPlane, error) {
	p := jxrPlane{color: uint8(b.read(3)), scaled: b.flag(), bands: uint8(b.read(4))}
	if b.err != nil {
		return p, b.err
	}
	if p.color == 5 || p.color == 7 || p.bands > 3 {
		return p, fmt.Errorf("%w: JPEG XR plane format", ErrUnsupported)
	}
	switch p.color {
	case 0:
		p.channels = 1
	case 1, 2, 3:
		p.channels = 3
		if p.color != 3 {
			_ = b.read(1)
			p.chromaX = uint8(b.read(3))
			if p.chromaX == 5 || p.chromaX == 6 {
				p.chromaX = 7
			}
		} else {
			_ = b.read(4)
		}
		if p.color == 1 {
			_ = b.read(1)
			p.chromaY = uint8(b.read(3))
			if p.chromaY == 5 || p.chromaY == 6 {
				p.chromaY = 7
			}
		} else {
			_ = b.read(4)
		}
	case 4:
		p.channels = 4
	case 6:
		count := b.read(4)
		if count == 15 {
			p.channels = int(b.read(12)) + 16
		} else {
			p.channels = int(count) + 1
			_ = b.read(4)
		}
	}
	if b.err != nil {
		return p, b.err
	}
	if alpha && p.color != 0 {
		return p, fmt.Errorf("%w: JPEG XR alpha color format", ErrInvalid)
	}
	if !alpha && !jxrPlaneColorAllowed(h.color, h.depth, p.color) {
		return p, fmt.Errorf("%w: JPEG XR color conversion combination", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(&p); err != nil {
		return p, err
	}
	if p.color == 1 || p.color == 2 {
		if h.overlap == 2 && h.columns[len(h.columns)-1] < 2 {
			return p, fmt.Errorf("%w: JPEG XR chroma overlap width", ErrInvalid)
		}
		if h.overlap == 2 && h.hardTiles {
			for i := 1; i < len(h.columns); i++ {
				if h.columns[i]-h.columns[i-1] < 2 {
					return p, fmt.Errorf("%w: JPEG XR chroma overlap tile width", ErrInvalid)
				}
			}
		}
	}
	if h.depth == 2 || h.depth == 3 || h.depth == 6 {
		p.shift = uint8(b.read(8))
	} else if h.depth == 7 {
		p.mantissa, p.exponent = uint8(b.read(8)), int8(b.read(8))
	}
	p.dcUniform = b.flag()
	if p.dcUniform {
		p.dcQuant = readJXRQuant(b, p.channels)
	}
	if p.bands != 3 {
		_ = b.read(1)
		p.lpUniform = b.flag()
		if p.lpUniform {
			p.lpQuant = readJXRQuant(b, p.channels)
		}
		if p.bands != 2 {
			_ = b.read(1)
			p.hpUniform = b.flag()
			if p.hpUniform {
				p.hpQuant = readJXRQuant(b, p.channels)
			}
		}
	}
	b.align(true)
	return p, b.err
}

// jxrPlaneColorAllowed 检查允许的采样与颜色转换组合
// 入参: output 输出颜色格式, depth 输出位深, internal 内部颜色格式
// 返回: bool 是否为标准定义的转换组合
func jxrPlaneColorAllowed(output, depth, internal uint8) bool {
	switch output {
	case 0, 1, 2, 6:
		return internal == output
	case 3:
		return internal <= 3
	case 4, 5:
		return internal == 4
	case 7:
		if depth == 4 || depth == 7 {
			return internal == 3
		}
		return internal <= 3
	case 8:
		return internal == 3
	default:
		return false
	}
}

// checkJXRPlaneComponents 校验内部颜色格式与分量数量，包括标准规定的扩展分量范围
// 入参: p 平面格式
// 返回: error 颜色或分量数量错误
func checkJXRPlaneComponents(p *jxrPlane) error {
	if p.channels < 1 || p.channels > rasterMaxComponents {
		return fmt.Errorf("%w: JPEG XR plane component count", ErrInvalid)
	}
	count := p.channels
	switch p.color {
	case 0:
		count = 1
	case 1, 2, 3:
		count = 3
	case 4:
		count = 4
	case 6:
	default:
		return fmt.Errorf("%w: JPEG XR plane color", ErrInvalid)
	}
	if p.channels != count {
		return fmt.Errorf("%w: JPEG XR plane color component count", ErrInvalid)
	}
	return nil
}

// readJXRQuant 读取共用、亮色分离或逐分量量化参数，单分量不读取模式位
// 入参: b 码流读取器, channels 已校验的分量数
// 返回: []uint8 逐分量量化参数，错误保存在读取器中
func readJXRQuant(b *jxrBits, channels int) []uint8 {
	if channels < 1 || channels > rasterMaxComponents {
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR quantization component count", ErrInvalid)
		}
		return nil
	}
	if b.err != nil {
		return nil
	}
	if err := b.parser.reserve(uint64(channels)); err != nil {
		b.err = err
		return nil
	}
	values := make([]uint8, channels)
	mode := uint32(0)
	if channels != 1 {
		mode = b.read(2)
	}
	switch mode {
	case 0:
		value := uint8(b.read(8))
		for i := range channels {
			values[i] = value
		}
	case 1:
		values[0] = uint8(b.read(8))
		value := uint8(b.read(8))
		for i := 1; i < channels; i++ {
			values[i] = value
		}
	case 2:
		for i := range channels {
			values[i] = uint8(b.read(8))
		}
	default:
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR quantization component mode", ErrUnsupported)
		}
	}
	return values
}

// cloneJXRQuant 复制量化数组并使用精确切片容量，保留未声明参数的空引用
// 入参: values 量化参数
// 返回: []uint8 独立量化存储
func cloneJXRQuant(values []uint8) []uint8 {
	if values == nil {
		return nil
	}
	result := make([]uint8, len(values))
	copy(result, values)
	return result
}

// equal 比较已声明的平面语义，不将未启用频带的量化存储视为有效参数
// 入参: other 待比较平面
// 返回: bool 参数是否一致
func (p jxrPlane) equal(other jxrPlane) bool {
	if p.color != other.color || p.channels != other.channels || p.bands != other.bands || p.scaled != other.scaled || p.chromaX != other.chromaX || p.chromaY != other.chromaY || p.shift != other.shift || p.mantissa != other.mantissa || p.exponent != other.exponent || p.dcUniform != other.dcUniform || p.lpUniform != other.lpUniform || p.hpUniform != other.hpUniform {
		return false
	}
	return (!p.dcUniform || equalJXRQuant(p.dcQuant, other.dcQuant, p.channels)) && (!p.lpUniform || equalJXRQuant(p.lpQuant, other.lpQuant, p.channels)) && (!p.hpUniform || equalJXRQuant(p.hpQuant, other.hpQuant, p.channels))
}

// equalJXRQuant 比较逐分量量化值，空参数表示全部为零
// 入参: a, b 量化参数, channels 分量数
// 返回: bool 参数是否一致
func equalJXRQuant(a, b []uint8, channels int) bool {
	if len(a) != 0 && len(a) != channels || len(b) != 0 && len(b) != channels {
		return false
	}
	if len(a) != 0 && len(b) != 0 {
		return slices.Equal(a, b)
	}
	for _, values := range [][]uint8{a, b} {
		for _, value := range values {
			if value != 0 {
				return false
			}
		}
	}
	return true
}

// quantWork 计算平面已持有的量化数组容量
// 返回: uint64 量化存储字节数
func (p *jxrPlane) quantWork() uint64 {
	work := uint64(0)
	for _, values := range [][]uint8{p.dcQuant, p.lpQuant, p.hpQuant} {
		amount := uint64(cap(values))
		if amount > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += amount
	}
	return work
}

// writeJXRPlane 写入内部颜色、采样、数值格式及图像级量化字段
// 入参: b 位写入器, h 图像头, p 平面参数, alpha 是否为透明度平面
func writeJXRPlane(b *jxrBitWriter, h *jxrHeader, p *jxrPlane, alpha bool) {
	if b.err != nil {
		return
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return
	}
	if err := checkJXRPlaneHeader(h, p, alpha); err != nil {
		b.err = err
		return
	}
	b.write(uint32(p.color), 3)
	b.flag(p.scaled)
	b.write(uint32(p.bands), 4)
	if p.color >= 1 && p.color <= 3 {
		if p.color != 3 {
			b.flag(false)
			b.write(uint32(p.chromaX), 3)
		} else {
			b.write(0, 4)
		}
		if p.color == 1 {
			b.flag(false)
			b.write(uint32(p.chromaY), 3)
		} else {
			b.write(0, 4)
		}
	} else if p.color == 6 {
		b.write(uint32(min(p.channels-1, 15)), 4)
		if p.channels >= 16 {
			b.write(uint32(p.channels-16), 12)
		} else {
			b.write(0, 4)
		}
	}
	if h.depth == 2 || h.depth == 3 || h.depth == 6 {
		b.write(uint32(p.shift), 8)
	} else if h.depth == 7 {
		b.write(uint32(p.mantissa), 8)
		b.write(uint32(uint8(p.exponent)), 8)
	}
	b.flag(p.dcUniform)
	if p.dcUniform {
		writeJXRPlaneQuant(b, p.dcQuant, p.channels)
	}
	if p.bands != 3 {
		b.flag(false)
		b.flag(p.lpUniform)
		if p.lpUniform {
			writeJXRPlaneQuant(b, p.lpQuant, p.channels)
		}
		if p.bands < 2 {
			b.flag(false)
			b.flag(p.hpUniform)
			if p.hpUniform {
				writeJXRPlaneQuant(b, p.hpQuant, p.channels)
			}
		}
	}
	b.align()
}

// checkJXRPlaneHeader 校验平面颜色、字段适用范围及抽样重叠条件
// 入参: h 已校验的图像头, p 平面参数, alpha 是否为透明度平面
// 返回: error 参数、布局或容量错误
func checkJXRPlaneHeader(h *jxrHeader, p *jxrPlane, alpha bool) error {
	if h == nil || p == nil {
		return fmt.Errorf("%w: JPEG XR output plane header", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if p.dcUniform && len(p.dcQuant) != 0 && len(p.dcQuant) != p.channels || p.lpUniform && len(p.lpQuant) != 0 && len(p.lpQuant) != p.channels || p.hpUniform && len(p.hpQuant) != 0 && len(p.hpQuant) != p.channels {
		return fmt.Errorf("%w: JPEG XR output plane quantization count", ErrInvalid)
	}
	if p.bands > 3 || alpha && p.color != 0 || !alpha && !jxrPlaneColorAllowed(h.color, h.depth, p.color) {
		return fmt.Errorf("%w: JPEG XR output plane color or bands", ErrInvalid)
	}
	if p.chromaX > 7 || p.chromaX == 5 || p.chromaX == 6 || p.chromaY > 7 || p.chromaY == 5 || p.chromaY == 6 || p.color != 1 && p.chromaY != 0 || p.color != 1 && p.color != 2 && p.chromaX != 0 {
		return fmt.Errorf("%w: JPEG XR output chroma centering", ErrInvalid)
	}
	if p.shift != 0 && h.depth != 2 && h.depth != 3 && h.depth != 6 || (p.mantissa != 0 || p.exponent != 0) && h.depth != 7 {
		return fmt.Errorf("%w: JPEG XR output numeric format fields", ErrInvalid)
	}
	if (p.color == 1 || p.color == 2) && h.overlap == 2 {
		if len(h.columns) < 2 || h.columns[len(h.columns)-1] < 2 {
			return fmt.Errorf("%w: JPEG XR output chroma overlap width", ErrInvalid)
		}
		if h.hardTiles {
			for n := 1; n < len(h.columns); n++ {
				if h.columns[n]-h.columns[n-1] < 2 {
					return fmt.Errorf("%w: JPEG XR output chroma overlap tile width", ErrInvalid)
				}
			}
		}
	}
	return nil
}

// writeJXRPlaneQuant 写入图像级逐分量量化参数，空参数按共用零值写入而不分配数组
// 入参: b 位写入器, values 量化参数, channels 已校验的分量数
func writeJXRPlaneQuant(b *jxrBitWriter, values []uint8, channels int) {
	if len(values) != 0 {
		writeJXRQuant(b, values)
		return
	}
	if channels != 1 {
		b.write(0, 2)
	}
	b.write(0, 8)
}

// writeJXRQuant 写入最短可用的共用、亮色分离或逐分量量化表示
// 入参: b 位写入器, values 单组逐分量量化参数
func writeJXRQuant(b *jxrBitWriter, values []uint8) {
	if b.err != nil {
		return
	}
	if len(values) < 1 || len(values) > rasterMaxComponents {
		b.err = fmt.Errorf("%w: JPEG XR output quantization component count", ErrInvalid)
		return
	}
	if len(values) == 1 {
		b.write(uint32(values[0]), 8)
		return
	}
	mode := uint32(0)
	if values[0] != values[1] {
		mode = 1
	}
	for _, value := range values[2:] {
		if value != values[1] {
			mode = 2
		}
	}
	b.write(mode, 2)
	count := 1
	if mode == 1 {
		count = 2
	} else if mode == 2 {
		count = len(values)
	}
	for _, value := range values[:count] {
		b.write(uint32(value), 8)
	}
}

// jxrQuantScale 根据量化参数和移位计算量化倍率，零量化值对应精确重建
// 入参: qp 量化参数, scaled 是否使用缩放模式, shift 当前频带及分量的缩放位数
// 返回: int64 正量化倍率
func jxrQuantScale(qp uint8, scaled bool, shift uint8) int64 {
	if qp == 0 {
		return 1
	}
	value := int64(qp)
	mantissa, exponent := int64(0), uint8(0)
	if scaled {
		mantissa, exponent = value, shift
		if qp >= 16 {
			mantissa = 16 + value%16
			exponent = (qp>>4 - 1) + shift
		}
	} else if qp < 32 {
		mantissa = (value + 3) >> 2
	} else if qp < 48 {
		mantissa, exponent = (16+value%16+1)>>1, qp>>4-2
	} else {
		mantissa, exponent = 16+value%16, qp>>4-3
	}
	return mantissa << exponent
}

// dequantJXRLowComponents 组合实际分量数的DC及低通系数并缩放，不分配宏块缓冲
// 入参: p 平面格式, dc, lp 预测后的系数, q 分块量化参数, index 低通集合索引, longWord 是否使用三十二位范围, result 输出，可与低通输入使用同一切片
// 返回: error 布局、索引或数值范围错误，仅修改有效分量，省略频带时lp可为空
func dequantJXRLowComponents(p *jxrPlane, dc []int64, lp [][16]int64, q *jxrTileQuant, index int, longWord bool, result [][16]int64) error {
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if q == nil || len(dc) < p.channels || len(result) < p.channels || len(lp) < p.channels && !(p.bands == 3 && len(lp) == 0) || index < 0 || q.lpCount < 1 || q.lpCount > 16 || index >= q.lpCount || index >= len(q.lp) || len(q.dc) != p.channels || len(q.lp[index]) != p.channels {
		return fmt.Errorf("%w: JPEG XR lowpass quantization index or storage", ErrInvalid)
	}
	for i := range p.channels {
		shift := uint8(0)
		if i == 0 {
			shift = 1
		}
		value, err := scaleJXRWord(dc[i], jxrQuantScale(q.dc[i], p.scaled, shift), longWord)
		if err != nil {
			return err
		}
		result[i][0] = value
		if len(lp) == 0 {
			clear(result[i][1:])
			continue
		}
		count := 16
		if i != 0 && p.color == 1 {
			count = 4
		} else if i != 0 && p.color == 2 {
			count = 8
		}
		factor := jxrQuantScale(q.lp[index][i], p.scaled, shift)
		for j := 1; j < count; j++ {
			value, err = scaleJXRWord(lp[i][j], factor, longWord)
			if err != nil {
				return err
			}
			result[i][j] = value
		}
		clear(result[i][count:])
	}
	return nil
}

// jxrAbs 取得已限制计算容量的系数绝对值
// 入参: value 有界系数
// 返回: int64 绝对值
func jxrAbs(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}

// quantJXRValue 将有界系数量化为整数，非精确模式采用中点取偶舍入并校验重建值不越界
// 入参: value 原始系数, factor 正量化倍率, longWord 是否使用三十二位范围, exact 是否要求精确恢复
// 返回: int64 量化系数, error 参数、范围或精度错误
func quantJXRValue(value, factor int64, longWord, exact bool) (int64, error) {
	if factor < 1 {
		return 0, fmt.Errorf("%w: JPEG XR quantization factor", ErrInvalid)
	}
	if err := checkJXRWords([]int64{value}, longWord); err != nil {
		return 0, err
	}
	if factor == 1 {
		return value, nil
	}
	magnitude := jxrAbs(value)
	quotient, remainder := magnitude/factor, magnitude%factor
	if exact && remainder != 0 {
		return 0, fmt.Errorf("%w: JPEG XR quantization precision", ErrUnsupported)
	}
	if !exact && (remainder > factor-remainder || remainder == factor-remainder && quotient&1 != 0) {
		quotient++
	}
	result := quotient
	if value < 0 {
		result = -result
	}
	if _, err := scaleJXRWord(result, factor, longWord); err != nil {
		if exact {
			return 0, err
		}
		if result < 0 {
			result++
		} else if result > 0 {
			result--
		}
	}
	return result, nil
}

// quantJXRLow 按分量量化宏块DC及低通系数，保留子采样网格中未使用的位置
// 入参: ctx Context, p 平面格式, values 分量步长为16的系数缓冲, dc, lp 逐分量量化参数, longWord 是否使用三十二位范围, exact 是否要求精确恢复
// 返回: error 格式、缓冲、精度、范围或取消错误
func quantJXRLow(ctx context.Context, p *jxrPlane, values []int64, dc, lp []uint8, longWord, exact bool) error {
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if len(values) < p.channels*16 || len(dc) < p.channels || len(lp) < p.channels {
		return fmt.Errorf("%w: JPEG XR lowpass quantization buffers", ErrInvalid)
	}
	if err := checkJXRWordsContext(ctx, values[:p.channels*16], longWord); err != nil {
		return err
	}
	for index := range p.channels {
		if err := ctx.Err(); err != nil {
			return err
		}
		count, shift := 16, uint8(0)
		if index == 0 {
			shift = 1
		} else if p.color == 1 {
			count = 4
		} else if p.color == 2 {
			count = 8
		}
		dcFactor := jxrQuantScale(dc[index], p.scaled, shift)
		lpFactor := jxrQuantScale(lp[index], p.scaled, shift)
		if dcFactor == 1 && lpFactor == 1 {
			continue
		}
		for at := range count {
			factor := lpFactor
			if at == 0 {
				factor = dcFactor
			}
			if factor == 1 {
				continue
			}
			value, err := quantJXRValue(values[index*16+at], factor, longWord, exact)
			if err != nil {
				return err
			}
			values[index*16+at] = value
		}
	}
	return ctx.Err()
}

// quantJXRHigh 量化各有效块的高通残差，不修改块直流位置或子采样填充位置
// 入参: ctx Context, p 平面格式, values 分量步长为256的残差缓冲, hp 逐分量量化参数, longWord 是否使用三十二位范围, exact 是否要求精确恢复
// 返回: error 格式、缓冲、精度、范围或取消错误
func quantJXRHigh(ctx context.Context, p *jxrPlane, values []int64, hp []uint8, longWord, exact bool) error {
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if len(values) < p.channels*256 || len(hp) < p.channels {
		return fmt.Errorf("%w: JPEG XR highpass quantization buffers", ErrInvalid)
	}
	if err := checkJXRWordsContext(ctx, values[:p.channels*256], longWord); err != nil {
		return err
	}
	for index := range p.channels {
		if err := ctx.Err(); err != nil {
			return err
		}
		factor := jxrQuantScale(hp[index], p.scaled, 1)
		if factor == 1 {
			continue
		}
		blocks := 16
		if index != 0 && p.color == 1 {
			blocks = 4
		} else if index != 0 && p.color == 2 {
			blocks = 8
		}
		for block := range blocks {
			for at := 1; at < 16; at++ {
				position := index*256 + block*16 + at
				value, err := quantJXRValue(values[position], factor, longWord, exact)
				if err != nil {
					return err
				}
				values[position] = value
			}
		}
	}
	return ctx.Err()
}

// writeJXRPacket 写入分块包头、截断量及逐平面分块量化字段，不写入宏块系数
// 入参: b 位写入器, s 所属码流, band 频带索引，空间模式为-1, q 完整分块量化集合, trim 精细位截断量
func writeJXRPacket(b *jxrBitWriter, s *jxrStream, band int, q *[2]jxrTileQuant, trim uint8) {
	if b.err != nil {
		return
	}
	if err := b.ctx.Err(); err != nil {
		b.err = err
		return
	}
	if s == nil || q == nil || band < -1 || band > 3 || s.header.frequency != (band >= 0) || trim > 15 || !s.header.trim && trim != 0 || b.bit%8 != 0 {
		b.err = fmt.Errorf("%w: JPEG XR output packet header", ErrInvalid)
		return
	}
	if s.primary.bands > 3 || band >= 4-int(s.primary.bands) || s.header.alpha != (s.alpha != nil) || s.alpha != nil && s.alpha.bands < s.primary.bands {
		b.err = fmt.Errorf("%w: JPEG XR output packet bands or alpha", ErrInvalid)
		return
	}
	planes := [2]*jxrPlane{&s.primary, s.alpha}
	for n, p := range planes {
		if p == nil {
			continue
		}
		if err := checkJXRTileQuant(p, &q[n]); err != nil {
			b.err = err
			return
		}
	}
	b.write(1, 24)
	b.write(0, 8)
	if s.header.trim && (band == -1 || band == 3) {
		b.write(uint32(trim), 4)
	}
	for n, p := range planes {
		if p == nil {
			continue
		}
		if (band == -1 || band == 0) && !p.dcUniform {
			writeJXRQuant(b, q[n].dc[:p.channels])
		}
		if p.bands != 3 && (band == -1 || band == 1) && !p.lpUniform {
			b.flag(q[n].useDC)
			if !q[n].useDC {
				b.write(uint32(q[n].lpCount-1), 4)
				for index := range q[n].lpCount {
					writeJXRQuant(b, q[n].lp[index][:p.channels])
				}
			}
		}
		if p.bands < 2 && (band == -1 || band == 2) && !p.hpUniform {
			b.flag(q[n].useLP)
			if !q[n].useLP {
				b.write(uint32(q[n].hpCount-1), 4)
				for index := range q[n].hpCount {
					writeJXRQuant(b, q[n].hp[index][:p.channels])
				}
			}
		}
	}
}

// checkJXRTileQuant 校验分块量化集合数量、图像级共用参数及频带复用关系
// 入参: p 平面参数, q 分块量化集合
// 返回: error 参数、复用关系或容量错误
func checkJXRTileQuant(p *jxrPlane, q *jxrTileQuant) error {
	if p == nil || q == nil || p.bands > 3 {
		return fmt.Errorf("%w: JPEG XR output tile quantization", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if len(q.dc) != p.channels {
		return fmt.Errorf("%w: JPEG XR output DC quantization component count", ErrInvalid)
	}
	count := p.channels
	if p.dcUniform && !equalJXRQuant(q.dc, p.dcQuant, count) {
		return fmt.Errorf("%w: JPEG XR output uniform DC quantization", ErrInvalid)
	}
	if p.bands != 3 {
		if q.lpCount < 1 || q.lpCount > 16 || len(q.lp) < q.lpCount {
			return fmt.Errorf("%w: JPEG XR output lowpass quantization count", ErrInvalid)
		}
		for n := range q.lpCount {
			if len(q.lp[n]) != count {
				return fmt.Errorf("%w: JPEG XR output lowpass quantization component count", ErrInvalid)
			}
		}
		if p.lpUniform && (q.lpCount != 1 || q.useDC || !equalJXRQuant(q.lp[0], p.lpQuant, count)) {
			return fmt.Errorf("%w: JPEG XR output uniform lowpass quantization", ErrInvalid)
		}
		if q.useDC && (q.lpCount != 1 || !slices.Equal(q.lp[0][:count], q.dc[:count])) {
			return fmt.Errorf("%w: JPEG XR output DC quantization reuse", ErrInvalid)
		}
	}
	if p.bands < 2 {
		if q.hpCount < 1 || q.hpCount > 16 || len(q.hp) < q.hpCount {
			return fmt.Errorf("%w: JPEG XR output highpass quantization count", ErrInvalid)
		}
		for n := range q.hpCount {
			if len(q.hp[n]) != count {
				return fmt.Errorf("%w: JPEG XR output highpass quantization component count", ErrInvalid)
			}
		}
		if p.hpUniform && (q.hpCount != 1 || q.useLP || !equalJXRQuant(q.hp[0], p.hpQuant, count)) {
			return fmt.Errorf("%w: JPEG XR output uniform highpass quantization", ErrInvalid)
		}
		if q.useLP {
			if q.hpCount != q.lpCount {
				return fmt.Errorf("%w: JPEG XR output lowpass quantization reuse count", ErrInvalid)
			}
			for n := range q.lpCount {
				if !slices.Equal(q.hp[n][:count], q.lp[n][:count]) {
					return fmt.Errorf("%w: JPEG XR output lowpass quantization reuse", ErrInvalid)
				}
			}
		}
	}
	return nil
}

// jxrMakeWriteVLC 将可信标准前缀表转换为逐符号查找表，不接受文件提供的自定义码表
// 入参: table 标准八位前缀查找表，符号编号小于16
// 返回: [16]jxrWriteCode 逐符号编码
func jxrMakeWriteVLC(table [256]jxrCode) [16]jxrWriteCode {
	var result [16]jxrWriteCode
	for prefix, code := range table {
		if code.bits != 0 {
			result[code.value] = jxrWriteCode{value: uint8(prefix >> (8 - code.bits)), bits: code.bits}
		}
	}
	return result
}

// writeJXRVLC 写入已定义符号的标准前缀码，未定义符号按参数错误处理
// 入参: b 位写入器, table 逐符号编码表, value 符号编号
func writeJXRVLC(b *jxrBitWriter, table *[16]jxrWriteCode, value uint8) {
	if b.err != nil {
		return
	}
	if value < 16 {
		if code := table[value]; code.bits != 0 {
			b.write(uint32(code.value), uint(code.bits))
			return
		}
	}
	b.err = fmt.Errorf("%w: JPEG XR output VLC symbol", ErrInvalid)
}

// writeFirst 写入首系数索引并累计相邻编码表的判别值
// 入参: b 位写入器, value 首系数索引
func (v *jxrMultiVLC) writeFirst(b *jxrBitWriter, value uint8) {
	if b.err != nil {
		return
	}
	if v.table >= 5 || v.delta[0] >= 4 || v.delta[1] >= 4 {
		b.err = fmt.Errorf("%w: JPEG XR output first VLC state", ErrInvalid)
		return
	}
	if value >= 12 {
		b.err = fmt.Errorf("%w: JPEG XR output VLC symbol", ErrInvalid)
		return
	}
	code := jxrFirstWriteCode[v.table][value]
	b.write(uint32(code.value), uint(code.bits))
	if b.err == nil {
		v.discriminant[0] += jxrFirstDelta[v.delta[0]][value]
		v.discriminant[1] += jxrFirstDelta[v.delta[1]][value]
	}
}

// writeIndex 写入后续系数索引并累计相邻编码表的判别值
// 入参: b 位写入器, value 后续系数索引
func (v *jxrMultiVLC) writeIndex(b *jxrBitWriter, value uint8) {
	if b.err != nil {
		return
	}
	if v.table >= 4 || v.delta[0] >= 3 || v.delta[1] >= 3 {
		b.err = fmt.Errorf("%w: JPEG XR output index VLC state", ErrInvalid)
		return
	}
	if value >= 6 {
		b.err = fmt.Errorf("%w: JPEG XR output VLC symbol", ErrInvalid)
		return
	}
	code := jxrIndexWriteCode[v.table][value]
	b.write(uint32(code.value), uint(code.bits))
	if b.err == nil {
		v.discriminant[0] += jxrIndexDelta[v.delta[0]][value]
		v.discriminant[1] += jxrIndexDelta[v.delta[1]][value]
	}
}
