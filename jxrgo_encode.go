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
	"crypto/sha256"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"math/bits"
	"slices"
)

// jxrInputHeaderReserve 为编码头、临时布局、量化描述及验证状态预留固定容量
const jxrInputHeaderReserve uint64 = 32768

// errJXRInputHeaderMemory 预构造固定容量错误，预算不足时不再为格式化诊断分配堆存储
var errJXRInputHeaderMemory = fmt.Errorf("%w: JPEG XR input header fixed memory", ErrLimit)

// errJXREncodingPlanMemory 预构造原图扣减后的额度错误，资源拒绝路径不再临时分配诊断
var errJXREncodingPlanMemory = fmt.Errorf("%w: encoding plan retained source memory", ErrLimit)

// jxrNativeOutput 保存原生编码的文件类型、可选格式标识、容器元数据及完整码流输出参数
type jxrNativeOutput struct {
	format        Format
	pixelFormat   [16]byte
	resolution    *Resolution
	iccProfile    []byte
	colorHint     *ColorHint
	colorEncoding *ColorEncoding
	exif          []byte
	gps           []byte
	xmp           []byte
	iptc          []byte
	description   *ImageDescription
	stream        jxrStreamOutput
}

// jxrForwardPacket 保存首遍得到的包偏移、长度及内容校验摘要
type jxrForwardPacket struct {
	offset uint64
	length uint64
	digest [sha256.Size]byte
}

// jxrForwardPlan 保存有界非定位输出所需的包目录及固定头部身份，不驻留任何完整压缩包
type jxrForwardPlan struct {
	packets     []jxrForwardPacket
	prefix      [sha256.Size]byte
	prefixBytes uint64
	total       uint64
	ready       bool
}

// jxrCountingSeeker 仅计算现有逐包编码器的实际写入和回填位置，不保存压缩字节
type jxrCountingSeeker struct {
	position int64
	length   int64
}

// jxrForwardWriter 为已规划索引的编码路径提供只允许查询当前位置的顺序输出
type jxrForwardWriter struct {
	writer   io.Writer
	position int64
}

// Encode 编码标准图像，opts为nil时使用无损JXR
// 入参: w 输出流, m 标准图像, opts 编码选项
// 返回: error 图像、编码或写入错误
func Encode(w io.Writer, m image.Image, opts *EncodeOptions) error {
	return wrapJXRError("Encode", encodeContextCore(context.Background(), w, m, opts))
}

// EncodeContext 编码标准图像并保留颜色精度与预乘状态，特化支持YCbCr、Paletted及Alpha类型，其余自定义图像转为十六位预乘色，输出原点归零
// opts为nil时默认无损JXR，工作内存计入内部原生副本但不含调用方图像存储，不关闭输出流
// 入参: ctx Context，nil采用Background, w 输出流, m 标准图像, opts 编码选项
// 返回: error 图像、资源、精度、取消或写入错误
func EncodeContext(ctx context.Context, w io.Writer, m image.Image, opts *EncodeOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	return wrapJXRError("EncodeContext", encodeContextCore(ctx, w, m, opts))
}

// encodeContextCore 执行标准图像编码核心流程，由外层公开入口统一包装错误操作名
// 入参: ctx 已规范化Context, w 输出流, m 标准图像, opts 编码选项
// 返回: error 图像、资源、精度、取消或写入错误
func encodeContextCore(ctx context.Context, w io.Writer, m image.Image, opts *EncodeOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil || m == nil {
		return fmt.Errorf("%w: nil output or image", ErrInvalid)
	}
	limits := DefaultLimits()
	preserveYCbCr := false
	if opts != nil {
		if err := checkJXREncodeOptions(*opts); err != nil {
			return err
		}
		limits = opts.Limits.normalized()
		preserveYCbCr = opts.PreserveYCbCr
	}
	raster, err := standardImageRasterWithOptions(ctx, m, limits, preserveYCbCr)
	if err != nil {
		return err
	}
	return encodeRasterCore(ctx, w, raster, opts)
}

// standardImageRasterWithOptions 将标准图像读取为独立原生样本，preserveYCbCr为真时保留对齐YCbCr原生子采样分量
// 入参: ctx Context, m 标准图像, limits 内部原生图像资源限制, preserveYCbCr 是否保留原生YCbCr子采样分量
// 返回: *Raster 原点为零的原生图像, error 颜色、资源、能力或取消错误
func standardImageRasterWithOptions(ctx context.Context, m image.Image, limits Limits, preserveYCbCr bool) (*Raster, error) {
	bounds := m.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("%w: standard image bounds", ErrInvalid)
	}
	if err := checkJXRDimensions(uint64(width), uint64(height), limits.MaxPixels); err != nil {
		return nil, err
	}
	if preserveYCbCr {
		if src, ok := m.(*image.YCbCr); ok {
			return standardYCbCrNativeRaster(ctx, src, bounds, limits)
		}
	}
	precision, count, space, alpha, premultiply := uint8(16), 4, ColorRGB, true, true
	var paletteTable [][4]uint16
	switch src := m.(type) {
	case *image.Gray:
		precision, count, space, alpha, premultiply = 8, 1, ColorGray, false, false
	case *image.Gray16:
		count, space, alpha, premultiply = 1, ColorGray, false, false
	case *image.CMYK:
		precision, space, alpha, premultiply = 8, ColorCMYK, false, false
	case *image.NRGBA:
		precision, premultiply = 8, false
	case *image.NRGBA64:
		premultiply = false
	case *image.RGBA:
		precision = 8
	case *image.Alpha:
		precision = 8
	case *image.Alpha16:
	case *image.YCbCr:
		precision, count, space, alpha, premultiply = 8, 3, ColorRGB, false, false
	case *image.Paletted:
		var err error
		precision, count, alpha, premultiply, paletteTable, err = prepareStandardPalette(src.Palette)
		if err != nil {
			return nil, err
		}
	}
	info := Info{Bounds: image.Rect(0, 0, width, height), ColorSpace: space, Alpha: alpha, Premultiplied: premultiply, Components: make([]ComponentInfo, count)}
	for index := range info.Components {
		info.Components[index] = ComponentInfo{Format: SampleUnsigned, Precision: precision}
	}
	raster, err := newRasterContext(ctx, info, limits)
	if err != nil {
		return nil, err
	}
	switch src := m.(type) {
	case *image.Gray:
		if err := copyStandardGray8(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.Gray16:
		if err := copyStandardGray16(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.RGBA:
		if err := copyStandardRGBA8(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.NRGBA:
		if err := copyStandardRGBA8(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.CMYK:
		if err := copyStandardRGBA8(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.NRGBA64:
		if err := copyStandardRGBA16(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.RGBA64:
		if err := copyStandardRGBA16(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.Alpha:
		if err := copyStandardAlpha8(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.Alpha16:
		if err := copyStandardAlpha16(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.YCbCr:
		if err := copyStandardYCbCr(ctx, src, bounds, raster); err != nil {
			return nil, err
		}
		return raster, nil
	case *image.Paletted:
		if err := copyStandardPaletted(ctx, src.Pix, src.Stride, src.PixOffset(bounds.Min.X, bounds.Min.Y), width, height, paletteTable, count, raster); err != nil {
			return nil, err
		}
		return raster, nil
	}
	p0, p1, p2, p3 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rowBase := y * width
		for x := range width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			c := m.At(bounds.Min.X+x, bounds.Min.Y+y)
			if c == nil {
				return nil, fmt.Errorf("%w: nil image colour", ErrInvalid)
			}
			r, g, b, a := c.RGBA()
			if r > 65535 || g > 65535 || b > 65535 || a > 65535 {
				return nil, fmt.Errorf("%w: image colour channel range", ErrInvalid)
			}
			at := rowBase + x
			p0[at], p1[at], p2[at], p3[at] = uint16(r), uint16(g), uint16(b), uint16(a)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return raster, nil
}

// checkStandardPix 校验标准图像像素切片步长与有效边界，防止畸形输入触发越界
// 入参: pixLen 像素字节切片长度, stride 行字节步长, offset0 首像素偏移, width, height 图像尺寸, bpp 每像素字节数
// 返回: error 步长或缓冲范围错误
func checkStandardPix(pixLen, stride, offset0, width, height, bpp int) error {
	rowBytes := int64(width) * int64(bpp)
	if offset0 < 0 || stride < 0 || int64(stride) < rowBytes || int64(offset0) > int64(pixLen)-rowBytes {
		return fmt.Errorf("%w: standard image pixel buffer", ErrInvalid)
	}
	if height > 1 && int64(height-1) > (int64(pixLen)-int64(offset0)-rowBytes)/int64(stride) {
		return fmt.Errorf("%w: standard image pixel buffer", ErrInvalid)
	}
	return nil
}

// copyStandardGray8 按行直接读取八位灰度像素写入单通道原生存储
// 入参: ctx Context, pix 源像素字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardGray8(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 1); err != nil {
		return err
	}
	p0 := raster.planes[0].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start:end]
			d0 := p0[rowBase+start : rowBase+end]
			_ = src[n-1]
			_ = d0[n-1]
			for i := range n {
				d0[i] = uint16(src[i])
			}
		}
	}
	return ctx.Err()
}

// copyStandardGray16 按行直接读取大端十六位灰度像素写入单通道原生存储
// 入参: ctx Context, pix 源像素字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardGray16(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 2); err != nil {
		return err
	}
	p0 := raster.planes[0].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start*2 : end*2]
			d0 := p0[rowBase+start : rowBase+end]
			_ = src[n*2-1]
			_ = d0[n-1]
			for i := range n {
				o := i * 2
				d0[i] = uint16(src[o])<<8 | uint16(src[o+1])
			}
		}
	}
	return ctx.Err()
}

// copyStandardRGBA8 按行直接读取四通道八位像素写入原生平面存储
// 入参: ctx Context, pix 源像素字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardRGBA8(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 4); err != nil {
		return err
	}
	p0, p1, p2, p3 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start*4 : end*4]
			d0 := p0[rowBase+start : rowBase+end]
			d1 := p1[rowBase+start : rowBase+end]
			d2 := p2[rowBase+start : rowBase+end]
			d3 := p3[rowBase+start : rowBase+end]
			_ = src[n*4-1]
			_ = d0[n-1]
			_ = d1[n-1]
			_ = d2[n-1]
			_ = d3[n-1]
			for i := range n {
				o := i * 4
				d0[i] = uint16(src[o])
				d1[i] = uint16(src[o+1])
				d2[i] = uint16(src[o+2])
				d3[i] = uint16(src[o+3])
			}
		}
	}
	return ctx.Err()
}

// copyStandardRGBA16 按行直接读取大端四通道十六位像素写入原生平面存储
// 入参: ctx Context, pix 源像素字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardRGBA16(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 8); err != nil {
		return err
	}
	p0, p1, p2, p3 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start*8 : end*8]
			d0 := p0[rowBase+start : rowBase+end]
			d1 := p1[rowBase+start : rowBase+end]
			d2 := p2[rowBase+start : rowBase+end]
			d3 := p3[rowBase+start : rowBase+end]
			_ = src[n*8-1]
			_ = d0[n-1]
			_ = d1[n-1]
			_ = d2[n-1]
			_ = d3[n-1]
			for i := range n {
				o := i * 8
				d0[i] = uint16(src[o])<<8 | uint16(src[o+1])
				d1[i] = uint16(src[o+2])<<8 | uint16(src[o+3])
				d2[i] = uint16(src[o+4])<<8 | uint16(src[o+5])
				d3[i] = uint16(src[o+6])<<8 | uint16(src[o+7])
			}
		}
	}
	return ctx.Err()
}

// prepareStandardPalette 分析调色板精度与透明状态并生成紧凑查表，优先选用八位三通道或四通道格式
// 入参: palette 标准调色板
// 返回: uint8 样本位深, int 分量数, bool 是否含透明度, bool 是否预乘, [][4]uint16 查表项, error 调色板非法错误
func prepareStandardPalette(palette color.Palette) (uint8, int, bool, bool, [][4]uint16, error) {
	if len(palette) == 0 {
		return 0, 0, false, false, nil, fmt.Errorf("%w: empty image palette", ErrInvalid)
	}
	hasAlpha, allEight, allNRGBA := false, true, true
	raw := make([][4]uint32, len(palette))
	for i, c := range palette {
		if c == nil {
			return 0, 0, false, false, nil, fmt.Errorf("%w: nil palette colour", ErrInvalid)
		}
		if _, ok := c.(color.NRGBA); !ok {
			allNRGBA = false
		}
		r, g, b, a := c.RGBA()
		if r > 65535 || g > 65535 || b > 65535 || a > 65535 {
			return 0, 0, false, false, nil, fmt.Errorf("%w: palette colour channel range", ErrInvalid)
		}
		raw[i] = [4]uint32{r, g, b, a}
		if a < 65535 {
			hasAlpha = true
		}
		if r != (r>>8)*257 || g != (g>>8)*257 || b != (b>>8)*257 || a != (a>>8)*257 {
			allEight = false
		}
	}
	table := make([][4]uint16, len(palette))
	if allNRGBA && hasAlpha {
		for i, c := range palette {
			n := c.(color.NRGBA)
			table[i] = [4]uint16{uint16(n.R), uint16(n.G), uint16(n.B), uint16(n.A)}
		}
		return 8, 4, true, false, table, nil
	}
	if allEight {
		for i, v := range raw {
			table[i] = [4]uint16{uint16(v[0] >> 8), uint16(v[1] >> 8), uint16(v[2] >> 8), uint16(v[3] >> 8)}
		}
		if !hasAlpha {
			return 8, 3, false, false, table, nil
		}
		return 8, 4, true, true, table, nil
	}
	for i, v := range raw {
		table[i] = [4]uint16{uint16(v[0]), uint16(v[1]), uint16(v[2]), uint16(v[3])}
	}
	if !hasAlpha {
		return 16, 3, false, false, table, nil
	}
	return 16, 4, true, true, table, nil
}

// copyStandardAlpha8 将八位Alpha单通道图像快速展开为八位四通道预乘原生存储
// 入参: ctx Context, pix 源透明度字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardAlpha8(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 1); err != nil {
		return err
	}
	p0, p1, p2, p3 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start:end]
			d0, d1, d2, d3 := p0[rowBase+start:rowBase+end], p1[rowBase+start:rowBase+end], p2[rowBase+start:rowBase+end], p3[rowBase+start:rowBase+end]
			_ = src[n-1]
			_ = d0[n-1]
			_ = d1[n-1]
			_ = d2[n-1]
			_ = d3[n-1]
			for i := range n {
				v := uint16(src[i])
				d0[i], d1[i], d2[i], d3[i] = v, v, v, v
			}
		}
	}
	return ctx.Err()
}

// copyStandardAlpha16 将十六位Alpha16单通道图像快速展开为十六位四通道预乘原生存储
// 入参: ctx Context, pix 源大端透明度字节, stride 行字节步长, offset0 首像素偏移, width, height 尺寸, raster 目标原生图像
// 返回: error 缓冲校验或取消错误
func copyStandardAlpha16(ctx context.Context, pix []byte, stride, offset0, width, height int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 2); err != nil {
		return err
	}
	p0, p1, p2, p3 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values, raster.planes[3].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for start := 0; start < width; start += 1024 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(start+1024, width)
			n := end - start
			src := row[start*2 : end*2]
			d0, d1, d2, d3 := p0[rowBase+start:rowBase+end], p1[rowBase+start:rowBase+end], p2[rowBase+start:rowBase+end], p3[rowBase+start:rowBase+end]
			_ = src[n*2-1]
			_ = d0[n-1]
			_ = d1[n-1]
			_ = d2[n-1]
			_ = d3[n-1]
			for i := range n {
				o := i * 2
				v := uint16(src[o])<<8 | uint16(src[o+1])
				d0[i], d1[i], d2[i], d3[i] = v, v, v, v
			}
		}
	}
	return ctx.Err()
}

// copyStandardYCbCr 将标准YCbCr图像直接转为八位三通道RGB原生存储，避免膨胀为十六位预乘RGBA
// 入参: ctx Context, src 源YCbCr图像, bounds 有效矩形, raster 目标三通道八位原生图像
// 返回: error 缓冲边界或取消错误
func copyStandardYCbCr(ctx context.Context, src *image.YCbCr, bounds image.Rectangle, raster *Raster) error {
	width, height := bounds.Dx(), bounds.Dy()
	p0, p1, p2 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		sy := bounds.Min.Y + y
		rowBase := y * width
		for x := range width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			sx := bounds.Min.X + x
			yi, ci := src.YOffset(sx, sy), src.COffset(sx, sy)
			if yi < 0 || yi >= len(src.Y) || ci < 0 || ci >= len(src.Cb) || ci >= len(src.Cr) {
				return fmt.Errorf("%w: standard YCbCr pixel buffer", ErrInvalid)
			}
			r, g, b := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
			at := rowBase + x
			p0[at], p1[at], p2[at] = uint16(r), uint16(g), uint16(b)
		}
	}
	return ctx.Err()
}

// standardYCbCrNativeRaster 将边界对齐的4:4:4、4:2:2或4:2:0标准YCbCr图像直接保留为原生YCC分量存储
// 入参: ctx Context, src 源YCbCr图像, bounds 有效矩形, limits 内部原生图像资源限制
// 返回: *Raster 原点为零的原生YCC图像, error 格式、缓冲、资源或取消错误
func standardYCbCrNativeRaster(ctx context.Context, src *image.YCbCr, bounds image.Rectangle, limits Limits) (*Raster, error) {
	width, height := bounds.Dx(), bounds.Dy()
	var sx, sy uint8
	switch src.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		sx, sy = 1, 1
	case image.YCbCrSubsampleRatio422:
		sx, sy = 2, 1
	case image.YCbCrSubsampleRatio420:
		sx, sy = 2, 2
	default:
		return nil, fmt.Errorf("%w: native YCbCr subsample ratio", ErrUnsupported)
	}
	if width%int(sx) != 0 || height%int(sy) != 0 || bounds.Min.X%int(sx) != 0 || bounds.Min.Y%int(sy) != 0 {
		return nil, fmt.Errorf("%w: native YCbCr subsample alignment", ErrUnsupported)
	}
	encoding := ColorEncoding{Primaries: 1, Transfer: 13, Matrix: 5, FullRange: true}
	info := Info{
		Bounds:        image.Rect(0, 0, width, height),
		Channels:      3,
		ColorSpace:    ColorYCC,
		ColorEncoding: &encoding,
		Components: []ComponentInfo{
			{Format: SampleUnsigned, Precision: 8, XStep: 1, YStep: 1},
			{Format: SampleUnsigned, Precision: 8, XStep: sx, YStep: sy},
			{Format: SampleUnsigned, Precision: 8, XStep: sx, YStep: sy},
		},
	}
	raster, err := newRasterContext(ctx, info, limits)
	if err != nil {
		return nil, err
	}
	p0 := raster.planes[0].values
	for y := range height {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		syPos := bounds.Min.Y + y
		rowBase := y * width
		for x := range width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			yi := src.YOffset(bounds.Min.X+x, syPos)
			if yi < 0 || yi >= len(src.Y) {
				return nil, fmt.Errorf("%w: standard YCbCr luma buffer", ErrInvalid)
			}
			p0[rowBase+x] = uint16(src.Y[yi])
		}
	}
	cw, ch := width/int(sx), height/int(sy)
	p1, p2 := raster.planes[1].values, raster.planes[2].values
	for cy := range ch {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		syPos := bounds.Min.Y + cy*int(sy)
		rowBase := cy * cw
		for cx := range cw {
			if cx&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			ci := src.COffset(bounds.Min.X+cx*int(sx), syPos)
			if ci < 0 || ci >= len(src.Cb) || ci >= len(src.Cr) {
				return nil, fmt.Errorf("%w: standard YCbCr chroma buffer", ErrInvalid)
			}
			p1[rowBase+cx], p2[rowBase+cx] = uint16(src.Cb[ci]), uint16(src.Cr[ci])
		}
	}
	return raster, ctx.Err()
}

// copyStandardPaletted 通过预计算调色板查找表高速写入三通道或四通道原生存储
// 入参: ctx Context, pix 索引字节, stride 行步长, offset0 首像素偏移, width, height 尺寸, table 颜色查表, count 通道数, raster 目标原生图像
// 返回: error 缓冲、调色板越界或取消错误
func copyStandardPaletted(ctx context.Context, pix []byte, stride, offset0, width, height int, table [][4]uint16, count int, raster *Raster) error {
	if err := checkStandardPix(len(pix), stride, offset0, width, height, 1); err != nil {
		return err
	}
	p0, p1, p2 := raster.planes[0].values, raster.planes[1].values, raster.planes[2].values
	var p3 []uint16
	if count == 4 {
		p3 = raster.planes[3].values
	}
	for y := range height {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := pix[offset0+y*stride:]
		rowBase := y * width
		for x := range width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			idx := int(row[x])
			if idx >= len(table) {
				return fmt.Errorf("%w: palette index out of range", ErrInvalid)
			}
			entry := table[idx]
			at := rowBase + x
			p0[at], p1[at], p2[at] = entry[0], entry[1], entry[2]
			if count == 4 {
				p3[at] = entry[3]
			}
		}
	}
	return ctx.Err()
}

// jxrContainerMetadataRetained 累加编码选项覆盖容器元数据及描述标签引入的常驻内存增量
// 入参: retained 当前已计入的常驻内存, limit 总内存上限, base 原始图像信息, opts 编码选项
// 返回: uint64 累加后的常驻内存, uint64 超限时的需求总量, bool 是否处于上限内
func jxrContainerMetadataRetained(retained, limit uint64, base Info, opts *EncodeOptions) (uint64, uint64, bool) {
	for _, pair := range [5][2][]byte{{opts.ICCProfile, base.ICCProfile}, {opts.EXIF, base.EXIF}, {opts.GPS, base.GPS}, {opts.XMP, base.XMP}, {opts.IPTC, base.IPTC}} {
		if len(pair[0]) != 0 && uint64(cap(pair[0])) > uint64(cap(pair[1])) {
			delta := uint64(cap(pair[0])) - uint64(cap(pair[1]))
			if retained > limit || delta > limit-retained {
				return retained, jxrSaturatedAdd(retained, delta), false
			}
			retained += delta
		}
	}
	if opts.ColorEncoding != nil && base.ColorEncoding == nil {
		if retained > limit || 4 > limit-retained {
			return retained, jxrSaturatedAdd(retained, 4), false
		}
		retained += 4
	}
	if !jxrDescriptionEmpty(opts.Description) {
		nextWork, prevWork := jxrDescriptionWork(opts.Description), jxrDescriptionWork(base.Description)
		if nextWork > prevWork {
			delta := nextWork - prevWork
			if retained > limit || delta > limit-retained {
				return retained, jxrSaturatedAdd(retained, delta), false
			}
			retained += delta
		}
	}
	return retained, retained, true
}

// jxrWithContainerMetadata 合并编码选项覆盖的分辨率、色彩、EXIF、GPS、XMP、IPTC及描述标签，不修改原Info
// 入参: info 原生图像元数据, opts 可选编码选项
// 返回: Info 合并后的图像元数据
func jxrWithContainerMetadata(info Info, opts *EncodeOptions) Info {
	if opts == nil {
		return info
	}
	if opts.Resolution != nil {
		info.Resolution = *opts.Resolution
	}
	if len(opts.ICCProfile) != 0 {
		info.ICCProfile = opts.ICCProfile
	}
	if opts.ColorHint != nil {
		info.ColorHint = *opts.ColorHint
	}
	if opts.ColorEncoding != nil {
		info.ColorEncoding = opts.ColorEncoding
	}
	if len(opts.EXIF) != 0 {
		info.EXIF = opts.EXIF
	}
	if len(opts.GPS) != 0 {
		info.GPS = opts.GPS
	}
	if len(opts.XMP) != 0 {
		info.XMP = opts.XMP
	}
	if len(opts.IPTC) != 0 {
		info.IPTC = opts.IPTC
	}
	if !jxrDescriptionEmpty(opts.Description) {
		info.Description = opts.Description
	}
	return info
}

// jxrQualityQP 将有损质量预设（1至99）映射为标准ITU-T T.832量化索引
// 入参: quality 有损质量值（1至99）
// 返回: uint8 标准T.832量化索引
func jxrQualityQP(quality int) uint8 {
	if quality >= 80 {
		return uint8(2 + (99-quality)*13/19)
	}
	return uint8(15 + (80-quality)*95/79)
}

// jxrQualityQuantization 按质量预设构造标准T.832三频带量化参数，低频保留更高精度减少块效应
// 入参: quality 有损质量值（1至99）
// 返回: *Quantization 独占量化配置
func jxrQualityQuantization(quality int) *Quantization {
	qp := jxrQualityQP(quality)
	dc := qp
	if qp > 2 {
		dc = max(1, qp/2)
	}
	return &Quantization{DC: []uint8{dc}, Lowpass: []uint8{qp}, Highpass: []uint8{qp}}
}

// jxrResolveEncodeOptions 展开EncodeOptions中的Quality质量预设至标准量化表与有损标志，不修改调用方原配置
// 入参: o 已校验的编码选项值副本
// 返回: EncodeOptions 展开质量预设后的选项副本
func jxrResolveEncodeOptions(o EncodeOptions) EncodeOptions {
	if o.Quality > 0 && o.Quality < 100 {
		if o.Quantization == nil {
			o.Quantization = jxrQualityQuantization(o.Quality)
		}
		o.Lossy = true
	}
	o.Quality = 0
	return o
}

// jxrInputTransform 按原生四二二采样方向确定编码网格的转置位，其他颜色保留明确请求的方向
// 入参: info 已校验原生描述, color 编码输出颜色, transform 请求的标准方向
// 返回: uint8 实际编码网格到原生显示网格的方向
func jxrInputTransform(info Info, color, transform uint8) uint8 {
	if color == 2 {
		transform &^= 4
		if info.Components[1].XStep == 1 && info.Components[1].YStep == 2 {
			transform |= 4
		}
	}
	return transform
}

// newJXRInputStream 从原生图像及布局建立编码头部与包索引空间，未指定的内部颜色采用对应可逆格式
// 入参: ctx 已规范化Context, raster 显示方向的原生图像, layout 编码布局, maxPixels 像素上限, maxMemory 新头部内存上限
// 返回: jxrStream 独立头部与包索引空间, error 格式、布局、资源或取消错误
func newJXRInputStream(ctx context.Context, raster *Raster, layout jxrStream, maxPixels, maxMemory uint64) (jxrStream, error) {
	if raster == nil {
		return jxrStream{}, fmt.Errorf("%w: JPEG XR native input image", ErrInvalid)
	}
	source, err := newJXRInputHeader(ctx, raster.info, layout, maxPixels, maxMemory)
	if err != nil {
		return jxrStream{}, err
	}
	for index := range raster.info.Components {
		if err := ctx.Err(); err != nil {
			return jxrStream{}, err
		}
		component, err := raster.component(index)
		if err != nil {
			return jxrStream{}, err
		}
		if _, err := component.sampleBitsAt(component.Bounds().Dx()-1, component.Bounds().Dy()-1); err != nil {
			return jxrStream{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return jxrStream{}, err
	}
	return source, nil
}

// newJXRInputHeader 按已明确来源的完整元数据建立编码头，不触碰不存在于逐块源中的整图样本
// 入参: ctx 已规范化Context, info 原生元数据, layout 编码布局, maxPixels 像素上限, maxMemory 新头部内存上限
// 返回: jxrStream 独立头部及包目录, error 元数据、布局、资源或取消错误
func newJXRInputHeader(ctx context.Context, info Info, layout jxrStream, maxPixels, maxMemory uint64) (jxrStream, error) {
	if err := ctx.Err(); err != nil {
		return jxrStream{}, err
	}
	if info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || len(info.Components) == 0 {
		return jxrStream{}, fmt.Errorf("%w: JPEG XR native input image", ErrInvalid)
	}
	components := info.Channels
	if info.Alpha {
		components++
	}
	if info.Channels < 1 || len(info.Components) != components {
		return jxrStream{}, fmt.Errorf("%w: JPEG XR native input components", ErrInvalid)
	}
	if maxMemory < jxrInputHeaderReserve {
		return jxrStream{}, errJXRInputHeaderMemory
	}
	f, err := rasterPixelFormat(info)
	if err != nil {
		return jxrStream{}, err
	}
	s := layout
	h := &s.header
	h.transform = jxrInputTransform(info, f.color, h.transform)
	h.width, h.height = uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
	if h.transform&4 != 0 {
		h.width, h.height = h.height, h.width
	}
	if err := checkJXRDimensions(h.width, h.height, maxPixels); err != nil {
		return jxrStream{}, err
	}
	h.color, h.depth, h.alpha, h.premultiply = f.color, f.depth, f.alpha, f.premultiply
	h.longWord = h.longWord || f.color == 8 || f.depth != 0 && f.depth != 1 && f.depth != 15
	h.planeStart = 0
	internalColor := s.primary.color
	if s.primary.channels == 0 {
		internalColor = f.color
		if f.color == 7 || f.color == 8 {
			internalColor = 3
		} else if f.color == 5 {
			internalColor = 4
		}
	}
	if !h.explicitWindow && h.top == 0 && h.left == 0 && h.bottom == 0 && h.right == 0 {
		h.right, h.bottom = uint32((16-h.width%16)%16), uint32((16-h.height%16)%16)
		if (internalColor == 1 || internalColor == 2) && h.overlap == 2 && h.width+uint64(h.left)+uint64(h.right) < 32 {
			h.right += 16
		}
	}
	if h.columns == nil {
		extent := (h.width + uint64(h.left) + uint64(h.right)) / 16
		h.columns = []uint32{0, uint32(extent)}
		if extent == jxrDimensionLimit/16 {
			h.columns = []uint32{0, 1, uint32(extent)}
		}
	}
	if h.rows == nil {
		extent := (h.height + uint64(h.top) + uint64(h.bottom)) / 16
		h.rows = []uint32{0, uint32(extent)}
		if extent == jxrDimensionLimit/16 {
			h.rows = []uint32{0, 1, uint32(extent)}
		}
	}
	h.index = h.index || h.frequency || len(h.columns) > 2 || len(h.rows) > 2
	short := h.width <= 65536 && h.height <= 65536
	for _, boundary := range [][]uint32{h.columns, h.rows} {
		for n := 1; n < len(boundary)-1; n++ {
			if boundary[n] > boundary[n-1] && boundary[n]-boundary[n-1] > 255 {
				short = false
			}
		}
	}
	h.short = h.short || short
	if err := checkJXRHeader(h, maxPixels); err != nil {
		return jxrStream{}, err
	}
	if s.primary.channels == 0 {
		s.primary.color, s.primary.channels = internalColor, f.channels
	}
	if s.alpha != nil {
		alpha := *s.alpha
		alpha.color, alpha.channels, alpha.chromaX, alpha.chromaY = 0, 1, 0, 0
		s.alpha = &alpha
	}
	quantMemory := uint64(0)
	for _, p := range []*jxrPlane{&s.primary, s.alpha} {
		if p == nil {
			continue
		}
		if err := checkJXRPlaneComponents(p); err != nil {
			return jxrStream{}, err
		}
		for n, target := range []*[]uint8{&p.dcQuant, &p.lpQuant, &p.hpQuant} {
			if len(*target) != 0 && len(*target) != p.channels {
				return jxrStream{}, fmt.Errorf("%w: JPEG XR input quantization component count", ErrInvalid)
			}
			uniform := p.dcUniform
			if n == 1 {
				uniform = p.lpUniform
			} else if n == 2 {
				uniform = p.hpUniform
			}
			amount := uint64(len(*target))
			if uniform && *target == nil {
				amount = uint64(p.channels)
			}
			if amount > maxMemory-quantMemory {
				return jxrStream{}, fmt.Errorf("%w: JPEG XR input quantization memory", ErrLimit)
			}
			quantMemory += amount
			if uniform && *target == nil {
				*target = make([]uint8, p.channels)
			} else {
				*target = cloneJXRQuant(*target)
			}
		}
	}
	if f.color == 6 && s.primary.channels != f.channels {
		return jxrStream{}, fmt.Errorf("%w: JPEG XR native input component count", ErrInvalid)
	}
	if f.color == 1 || f.color == 2 {
		component := info.Components[1]
		derivedX, derivedY := uint8(7), uint8(0)
		if f.color == 1 {
			derivedY = 7
		}
		x, y := component.XOffset, component.YOffset
		knownX, knownY := component.XPositionKnown, component.YPositionKnown
		if h.transform&2 != 0 && component.XStep == 2 {
			x = 1 - x
		}
		if h.transform&1 != 0 && component.YStep == 2 {
			y = 1 - y
		}
		if h.transform&4 != 0 {
			x, y, knownX, knownY = y, x, knownY, knownX
			if f.color == 1 {
				y = 1 - y
			}
		}
		if knownX {
			derivedX = uint8(x * 4)
		}
		if f.color == 1 && knownY {
			derivedY = uint8(y * 4)
		}
		if !s.primary.explicitChromaX {
			s.primary.chromaX = derivedX
		}
		if !s.primary.explicitChromaY {
			s.primary.chromaY = derivedY
		}
	}
	if err := checkJXRPlaneHeader(h, &s.primary, false); err != nil {
		return jxrStream{}, err
	}
	if f.alpha {
		alpha := s.primary
		if s.alpha != nil {
			alpha = *s.alpha
		}
		alpha.color, alpha.channels, alpha.chromaX, alpha.chromaY = 0, 1, 0, 0
		if layout.alpha == nil {
			amount := uint64(min(1, len(alpha.dcQuant)) + min(1, len(alpha.lpQuant)) + min(1, len(alpha.hpQuant)))
			if amount > maxMemory-quantMemory {
				return jxrStream{}, &LimitError{Resource: "JPEG XR input alpha quantization memory", Limit: maxMemory, Required: jxrSaturatedAdd(quantMemory, amount)}
			}
			quantMemory += amount
			alpha.dcQuant, alpha.lpQuant, alpha.hpQuant = cloneJXRQuant(alpha.dcQuant[:min(1, len(alpha.dcQuant))]), cloneJXRQuant(alpha.lpQuant[:min(1, len(alpha.lpQuant))]), cloneJXRQuant(alpha.hpQuant[:min(1, len(alpha.hpQuant))])
		}
		s.alpha = &alpha
		if err := checkJXRPlaneHeader(h, s.alpha, true); err != nil {
			return jxrStream{}, err
		}
		if alpha.bands < s.primary.bands {
			return jxrStream{}, fmt.Errorf("%w: JPEG XR native input alpha bands", ErrInvalid)
		}
	} else if s.alpha != nil {
		return jxrStream{}, fmt.Errorf("%w: JPEG XR native input missing alpha", ErrInvalid)
	}
	count := uint64(len(h.columns)-1) * uint64(len(h.rows)-1)
	if h.frequency {
		count *= uint64(4 - s.primary.bands)
	}
	if count > uint64(^uint(0)>>1)/8 {
		return jxrStream{}, &LimitError{Resource: "JPEG XR input header packet count", Limit: uint64(^uint(0)>>1) / 8, Required: count}
	}
	if len(s.profiles) > 4096 {
		return jxrStream{}, &LimitError{Resource: "JPEG XR input header profile count", Limit: 4096, Required: uint64(len(s.profiles))}
	}
	memory := jxrInputHeaderReserve + count*8 + uint64(len(h.columns)+len(h.rows)+len(s.profiles))*4
	if memory > maxMemory || quantMemory > maxMemory-memory {
		return jxrStream{}, &LimitError{Resource: "JPEG XR input header memory", Limit: maxMemory, Required: jxrSaturatedAdd(memory, quantMemory)}
	}
	if err := ctx.Err(); err != nil {
		return jxrStream{}, err
	}
	h.columns, h.rows = slices.Clone(h.columns), slices.Clone(h.rows)
	s.profiles = slices.Clone(s.profiles)
	s.packets, s.dataStart = make([]uint64, int(count)), 0
	if err := checkJXRStreamOutput(&s, maxPixels); err != nil {
		return jxrStream{}, err
	}
	if err := ctx.Err(); err != nil {
		return jxrStream{}, err
	}
	return s, nil
}

// makeJXREncodingPlan 校验公开选项，在扣除原图后的剩余预算内生成内部编码参数，不修改调用方配置
// 入参: ctx Context, raster 原生图像, opts 编码选项
// 返回: jxrStream 编码头部, jxrNativeOutput 输出参数, error 选项、资源或取消错误
func makeJXREncodingPlan(ctx context.Context, raster *Raster, opts *EncodeOptions) (jxrStream, jxrNativeOutput, error) {
	var o EncodeOptions
	if opts != nil {
		o = *opts
	}
	if err := checkJXREncodeOptions(o); err != nil {
		return jxrStream{}, jxrNativeOutput{}, err
	}
	retained, err := jxrRasterWork(raster, o.Limits.normalized().MaxMemoryBytes)
	if err != nil {
		return jxrStream{}, jxrNativeOutput{}, err
	}
	return makeJXREncodingPlanRetained(ctx, raster, &o, retained)
}

// makeJXREncodingPlanRetained 共用完整原图和有界分块源的布局规划，选项及常驻源容量由对应入口先行校验
// 入参: ctx 已规范化Context, raster 原图或仅含完整元数据的内部视图, opts 已校验编码选项, retained 常驻源和元数据容量
// 返回: jxrStream 编码头部, jxrNativeOutput 输出参数, error 选项、资源或取消错误
func makeJXREncodingPlanRetained(ctx context.Context, raster *Raster, opts *EncodeOptions, retained uint64) (jxrStream, jxrNativeOutput, error) {
	var o EncodeOptions
	if opts != nil {
		o = jxrResolveEncodeOptions(*opts)
	}
	limits := o.Limits.normalized()
	if o.Format == FormatUnknown {
		o.Format = FormatJXR
	}
	output := jxrNativeOutput{format: o.Format, stream: jxrStreamOutput{exact: !o.Lossy, workers: normalizeJXRWorkers(o.Workers), maxPixels: limits.MaxPixels, maxBytes: o.MaxOutputBytes, maxMemory: limits.MaxMemoryBytes}}
	if o.Format == FormatJXR {
		merged := jxrWithContainerMetadata(raster.info, &o)
		if o.Resolution != nil {
			res := *o.Resolution
			output.resolution = &res
		}
		output.iccProfile, output.colorEncoding = merged.ICCProfile, merged.ColorEncoding
		if o.ColorHint != nil {
			hint := *o.ColorHint
			output.colorHint = &hint
		}
		output.exif, output.gps, output.xmp, output.iptc, output.description = merged.EXIF, merged.GPS, merged.XMP, merged.IPTC, merged.Description
		nextRetained, _, ok := jxrContainerMetadataRetained(retained, limits.MaxMemoryBytes, raster.info, &o)
		if !ok {
			return jxrStream{}, output, errJXREncodingPlanMemory
		}
		retained = nextRetained
	}
	if output.stream.maxBytes == 0 {
		output.stream.maxBytes = 64 << 20
	}
	if retained > limits.MaxMemoryBytes {
		return jxrStream{}, output, errJXREncodingPlanMemory
	}
	available := limits.MaxMemoryBytes - retained
	if available < jxrInputHeaderReserve {
		return jxrStream{}, output, errJXREncodingPlanMemory
	}
	layout := jxrStream{header: jxrHeader{frequency: o.Layout == LayoutFrequency, overlap: uint8(o.Overlap), transform: uint8(o.Orientation), hardTiles: o.HardTiles, trim: o.Trim != 0}}
	if o.Window != nil {
		layout.header.explicitWindow = true
		layout.header.top, layout.header.left, layout.header.bottom, layout.header.right = uint32(o.Window.Top), uint32(o.Window.Left), uint32(o.Window.Bottom), uint32(o.Window.Right)
	}
	if len(o.Profiles) != 0 {
		layout.profiles = make([]jxrProfileLevel, len(o.Profiles))
		for index, item := range o.Profiles {
			layout.profiles[index] = jxrProfileLevel{profile: item.Profile, level: item.Level}
		}
	}
	main := PlaneOptions{Band: o.Band, Scaled: o.Scaled, Shift: o.Shift, Float: o.Float, Quantization: o.Quantization}
	p, err := makeJXREncodePlane(ctx, raster, main, false, o.Alpha == nil, o.Lossy, o.Overlap)
	if err != nil {
		return layout, output, err
	}
	if o.Subsampling != SubsamplingNone {
		f, err := rasterPixelFormat(raster.info)
		if err != nil {
			return layout, output, err
		}
		if f.color != 3 && (f.color != 7 || f.depth == 4 || f.depth == 7) {
			return layout, output, fmt.Errorf("%w: encode subsampling requires non-floating RGB or YCC444", ErrUnsupported)
		}
		p.channels, p.chromaX = 3, 2
		if o.Subsampling == Subsampling422 {
			p.color, p.chromaY = 2, 0
			if o.ChromaCenteringY != nil {
				return layout, output, fmt.Errorf("%w: encode YCC422 does not allow chroma centering Y", ErrInvalid)
			}
		} else {
			p.color, p.chromaY = 1, 2
		}
	} else if o.ChromaCenteringX != nil || o.ChromaCenteringY != nil {
		f, err := rasterPixelFormat(raster.info)
		if err != nil {
			return layout, output, err
		}
		if f.color != 1 && f.color != 2 || f.color == 2 && o.ChromaCenteringY != nil {
			return layout, output, fmt.Errorf("%w: encode chroma centering requires YCC420 or YCC422", ErrInvalid)
		}
	}
	if o.ChromaCenteringX != nil {
		p.chromaX, p.explicitChromaX = *o.ChromaCenteringX, true
	}
	if o.ChromaCenteringY != nil {
		p.chromaY, p.explicitChromaY = *o.ChromaCenteringY, true
	}
	p.legacyHighpassPrediction = o.LegacyHighpassPrediction
	layout.primary = p
	if o.Alpha != nil {
		if !raster.info.Alpha {
			return layout, output, fmt.Errorf("%w: alpha options without alpha", ErrInvalid)
		}
		alpha, err := makeJXREncodePlane(ctx, raster, *o.Alpha, true, false, o.Lossy, o.Overlap)
		if err != nil {
			return layout, output, err
		}
		alpha.legacyHighpassPrediction = o.LegacyHighpassPrediction
		layout.alpha = &alpha
	}
	s, err := newJXRInputHeader(ctx, raster.info, layout, limits.MaxPixels, available)
	if err != nil {
		return layout, output, err
	}
	if o.TileSize.X != 0 || o.TileSize.Y != 0 {
		adjustCols := s.header.hardTiles && s.header.overlap == 2 && (s.primary.color == 1 || s.primary.color == 2)
		columns, err := jxrTileColumnBoundaries(s.header.columns[len(s.header.columns)-1], uint32(o.TileSize.X/16), adjustCols, available)
		if err != nil {
			return layout, output, err
		}
		rows, err := jxrTileColumnBoundaries(s.header.rows[len(s.header.rows)-1], uint32(o.TileSize.Y/16), false, available)
		if err != nil {
			return layout, output, err
		}
		s.header.columns, s.header.rows = columns, rows
		s.header.short = false
		s, err = newJXRInputHeader(ctx, raster.info, s, limits.MaxPixels, available)
		if err != nil {
			return layout, output, err
		}
	}
	output.stream.tiles, output.stream.quantWork, err = planJXRQuantization(ctx, &s, o, available)
	if err != nil {
		return layout, output, err
	}
	output.stream.validated = true
	return s, output, nil
}

// checkJXREncodeOptions 在图像复制及样本扫描前校验与图像无关的公开选项
// 入参: o 编码选项
// 返回: error 参数或容量错误
func checkJXREncodeOptions(o EncodeOptions) error {
	if o.Workers < 0 {
		return fmt.Errorf("%w: negative encode workers", ErrInvalid)
	}
	if o.Quality < 0 || o.Quality > 100 {
		return fmt.Errorf("%w: encode quality", ErrInvalid)
	}
	if o.Resolution != nil {
		for _, resolution := range [2]float64{o.Resolution.X, o.Resolution.Y} {
			if !geometryFinite(resolution) || resolution < 0 || resolution > 0 && (float32(resolution) == 0 || math.IsInf(float64(float32(resolution)), 0)) {
				return fmt.Errorf("%w: encode resolution", ErrInvalid)
			}
		}
	}
	if o.MirrorSeparateAlpha && !o.SeparateAlpha {
		return fmt.Errorf("%w: mirrored alpha requires separate alpha", ErrInvalid)
	}
	if o.SeparateAlpha && o.Format == FormatCodestream {
		return fmt.Errorf("%w: separate alpha requires a file container", ErrInvalid)
	}
	effectiveLossy := o.Lossy || o.Quality > 0 && o.Quality < 100
	if o.Format > FormatCodestream || o.Band > BandDC || o.Layout > LayoutFrequency || o.Overlap > OverlapTwo || o.Orientation > OrientationRotate90FlipXY || o.Subsampling > Subsampling420 || o.Subsampling != SubsamplingNone && !effectiveLossy || o.Trim > 15 || o.Alpha != nil && o.Alpha.Band > BandDC {
		return fmt.Errorf("%w: encode options", ErrInvalid)
	}
	if o.ColorHint != nil && *o.ColorHint != ColorHintNone && *o.ColorHint != ColorHintSRGB && *o.ColorHint != ColorHintUnspecified {
		return fmt.Errorf("%w: encode color hint", ErrInvalid)
	}
	if o.Uncompressed {
		if o.Format == FormatCodestream || o.Layout != LayoutSpatial || o.Overlap != OverlapNone || o.Subsampling != SubsamplingNone || o.TileSize != (image.Point{}) || o.HardTiles || o.Lossy || o.Quality != 0 || o.LegacyHighpassPrediction || o.Band != BandAll || o.Trim != 0 || o.Scaled || o.Shift != 0 || o.Float != nil || o.Quantization != nil || o.Alpha != nil || o.MirrorSeparateAlpha || len(o.Profiles) != 0 || o.Window != nil || o.ChromaCenteringX != nil || o.ChromaCenteringY != nil {
			return fmt.Errorf("%w: uncompressed mode conflicts with compression options", ErrInvalid)
		}
	}
	if o.Window != nil && (o.Window.Top > 63 || o.Window.Left > 63 || o.Window.Bottom > 63 || o.Window.Right > 63) {
		return fmt.Errorf("%w: encode window margins", ErrInvalid)
	}
	if o.ChromaCenteringX != nil && (*o.ChromaCenteringX > 4 && *o.ChromaCenteringX != 7) {
		return fmt.Errorf("%w: encode chroma centering X", ErrInvalid)
	}
	if o.ChromaCenteringY != nil && (*o.ChromaCenteringY > 4 && *o.ChromaCenteringY != 7) {
		return fmt.Errorf("%w: encode chroma centering Y", ErrInvalid)
	}
	if len(o.Profiles) != 0 {
		if len(o.Profiles) > 4096 {
			return &LimitError{Resource: "encode profile declaration count", Limit: 4096, Required: uint64(len(o.Profiles))}
		}
		profiles := make([]jxrProfileLevel, len(o.Profiles))
		for index, item := range o.Profiles {
			profiles[index] = jxrProfileLevel{profile: item.Profile, level: item.Level}
		}
		if err := checkJXRProfileDeclarations(profiles); err != nil {
			return err
		}
	}
	if (o.TileSize.X != 0 || o.TileSize.Y != 0) && (o.TileSize.X <= 0 || o.TileSize.Y <= 0 || o.TileSize.X%16 != 0 || o.TileSize.Y%16 != 0 || o.TileSize.X/16 > 65535 || o.TileSize.Y/16 > 65535) {
		return fmt.Errorf("%w: encode tile dimensions", ErrInvalid)
	}
	if o.Float != nil && o.Float.Mantissa > 23 {
		return fmt.Errorf("%w: encode floating mantissa", ErrInvalid)
	}
	if o.Alpha != nil && o.Alpha.Float != nil && o.Alpha.Float.Mantissa > 23 {
		return fmt.Errorf("%w: encode alpha floating mantissa", ErrInvalid)
	}
	return nil
}

// jxrTileColumnBoundaries 构建分块列或行宏块边界，并在子采样二级重叠硬分块下同步调整尾列宽度
// 入参: extent 轴向总宏块数, size 单块宏块尺寸, adjustChromaTail 是否执行子采样尾列调整, available 可用内存预算
// 返回: []uint32 单调递增分块边界, error 分块数量或内存预算错误
func jxrTileColumnBoundaries(extent, size uint32, adjustChromaTail bool, available uint64) ([]uint32, error) {
	count := (uint64(extent) + uint64(size) - 1) / uint64(size)
	if count == 0 || count > 4096 {
		return nil, &LimitError{Resource: "encode tile partition count", Limit: 4096, Required: count}
	}
	if count*4 > available {
		return nil, &LimitError{Resource: "encode tile partition memory", Limit: available, Required: count * 4}
	}
	boundary := make([]uint32, int(count)+1)
	for n := uint64(1); n <= count; n++ {
		boundary[n] = uint32(min(n*uint64(size), uint64(extent)))
	}
	if adjustChromaTail && count >= 2 && extent-boundary[count-1] < 2 {
		if size >= 3 {
			boundary[count-1]--
		} else if size == 2 {
			boundary[count-1] = extent
			boundary = boundary[:count]
		}
	}
	return boundary, nil
}

// jxrFloatGainBits 按平面通道、缩放与重叠滤波计算单精度浮点变换域增益位数
// 入参: channels 当前平面通道数, scaled 是否启用预缩放, overlap 重叠滤波级别, lossy 是否允许精度损失
// 返回: int 变换域最大增益位数
func jxrFloatGainBits(channels int, scaled bool, overlap Overlap, lossy bool) int {
	gain := 4
	if scaled {
		gain += 3
	}
	if channels == 3 && (lossy || !scaled) {
		gain++
	}
	if lossy {
		gain += int(overlap)
	} else if overlap != OverlapNone {
		gain++
	}
	return gain
}

// jxrFloatSampleBound 计算非负单精度浮点位模式在指定偏置下的无损整数映射幅值
// 入参: raw 清除符号位后的IEEE754单精度位模式, bias 候选指数偏置
// 返回: int64 整数映射幅值，超出范围时返回最大整数
func jxrFloatSampleBound(raw uint32, bias int8) int64 {
	value, err := jxrFloatInputBits(raw, 7, 23, bias, false)
	if err != nil {
		return math.MaxInt64
	}
	return value
}

// jxrVerifyFloatLosslessCapacity 在极端跨指数区间按宏块核验实际无损单精度像素是否超出三十二位变换字长
// 入参: ctx Context, raster 原生图像, base 首分量下标, count 需覆盖的分量数, isRGB 前三通道是否执行RGB正向色彩变换, scaled 是否启用预缩放, overlap 重叠滤波级别, bias 候选指数偏置
// 返回: error 无法无损容纳或取消错误
func jxrVerifyFloatLosslessCapacity(ctx context.Context, raster *Raster, base, count int, isRGB, scaled bool, overlap Overlap, bias int8) error {
	var comps [4]Component
	for index := range count {
		component, err := raster.Component(base + index)
		if err != nil {
			return err
		}
		comps[index] = component
	}
	src := comps[0].source
	stride := src.channels
	if src.alpha {
		stride++
	}
	w, h := src.bounds.Dx(), src.bounds.Dy()
	words := src.words
	mbW, mbH := (w+15)/16, (h+15)/16
	ovNum, ovDen := int64(1), int64(1)
	if overlap == OverlapOne {
		ovNum, ovDen = 5, 4
	} else if overlap == OverlapTwo {
		ovNum, ovDen = 3, 2
	}
	for my := range mbH {
		if err := ctx.Err(); err != nil {
			return err
		}
		for mx := range mbW {
			if mx&15 == 15 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			var sums [4]int64
			var maxAbs [4]int64
			for dy := range 16 {
				cy := min(my*16+dy, h-1)
				rowOff := cy * w * stride
				for dx := range 16 {
					cx := min(mx*16+dx, w-1)
					pixOff := rowOff + cx*stride
					var vals [4]int64
					for i := range count {
						v, err := jxrFloatInputBits(words[pixOff+comps[i].index], 7, 23, bias, false)
						if err != nil {
							return fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
						}
						if scaled {
							v <<= 3
							if v < jxrLongWordMin || v > jxrLongWordMax {
								return fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
							}
						}
						vals[i] = v
					}
					if isRGB {
						v := vals[2] - vals[0]
						t := vals[0] - vals[1] + ((v + 1) >> 1)
						yv := vals[1] + (t >> 1)
						vals[0], vals[1], vals[2] = yv, -t, v
					}
					for i := range count {
						val := vals[i]
						if val < jxrLongWordMin || val > jxrLongWordMax {
							return fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
						}
						sums[i] += val
						if a := jxrAbs(val); a > maxAbs[i] {
							maxAbs[i] = a
						}
					}
				}
			}
			for i := range count {
				dcShift := uint(4)
				if scaled && isRGB && (i == 1 || i == 2) {
					dcShift = 5
				}
				dcEst := (jxrAbs(sums[i]) >> dcShift) * ovNum / ovDen
				acEst := (maxAbs[i] << 2) * ovNum / ovDen
				if dcEst > jxrLongWordMax || acEst > jxrLongWordMax {
					return fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
				}
			}
		}
	}
	return ctx.Err()
}

// makeJXREncodePlane 按完整变换增益建立图像级共用参数，量化切片只读取不共享
// 入参: ctx Context, raster 原生图像, o 平面选项, alpha 透明度平面, includeAlpha 自动浮点偏置是否同时覆盖透明度, lossy 是否允许精度损失, overlap 重叠滤波级别
// 返回: jxrPlane 平面参数, error 选项、精度或取消错误
func makeJXREncodePlane(ctx context.Context, raster *Raster, o PlaneOptions, alpha, includeAlpha, lossy bool, overlap Overlap) (jxrPlane, error) {
	p := jxrPlane{bands: uint8(o.Band), scaled: o.Scaled, shift: o.Shift, dcUniform: true, lpUniform: true, hpUniform: true}
	if o.Band > BandDC {
		return p, fmt.Errorf("%w: encode plane bands", ErrInvalid)
	}
	count, base := raster.info.Channels, 0
	if alpha {
		count, base = 1, raster.info.Channels
	}
	if count < 1 || count > rasterMaxComponents {
		return p, fmt.Errorf("%w: encode component count", ErrInvalid)
	}
	first := raster.info.Components[base]
	p.dcQuant, p.lpQuant, p.hpQuant = make([]uint8, count), make([]uint8, count), make([]uint8, count)
	if o.Float != nil {
		if first.Format != SampleFloat || first.Precision != 32 || o.Float.Mantissa > 23 {
			return p, fmt.Errorf("%w: encode floating parameters", ErrInvalid)
		}
		p.mantissa, p.exponent = o.Float.Mantissa, o.Float.Bias
	} else if first.Format == SampleFloat && first.Precision == 32 {
		if lossy {
			if err := ctx.Err(); err != nil {
				return p, err
			}
			gainBits := jxrFloatGainBits(count, p.scaled, overlap, true)
			p.mantissa, p.exponent = uint8(min(16, max(1, 23-gainBits))), 127
		} else {
			hasNormal, hasSubnormal := false, false
			maxExp, maxRaw := uint32(0), uint32(0)
			needBias := -128
			n := count
			if includeAlpha && raster.info.Alpha {
				n++
			}
			for index := range n {
				component, err := raster.Component(base + index)
				if err != nil {
					return p, err
				}
				src := component.source
				stride := src.channels
				if src.alpha {
					stride++
				}
				w, h := src.bounds.Dx(), src.bounds.Dy()
				words := src.words
				for y := range h {
					if err := ctx.Err(); err != nil {
						return p, err
					}
					rowBase := y*w*stride + component.index
					for x := range w {
						if x&1023 == 1023 {
							if err := ctx.Err(); err != nil {
								return p, err
							}
						}
						raw := words[rowBase+x*stride] & 0x7fffffff
						if raw == 0 {
							continue
						}
						maxRaw = max(maxRaw, raw)
						exp := raw >> 23
						if exp == 0 {
							hasSubnormal = true
							needBias = max(needBias, 127-bits.TrailingZeros32(raw))
							continue
						}
						hasNormal = true
						maxExp = max(maxExp, exp)
						tz := min(23, bits.TrailingZeros32(raw&0x7fffff))
						needBias = max(needBias, 128-int(exp)-tz)
					}
				}
			}
			p.mantissa = 23
			if !hasNormal && !hasSubnormal {
				p.exponent = 127
			} else {
				if needBias > 127 {
					return p, fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
				}
				gainBits := jxrFloatGainBits(count, p.scaled, overlap, false)
				safeLimit := (int64(1) << 31) >> gainBits
				hardGain := 2
				if p.scaled {
					hardGain += 3
				}
				needVal := jxrFloatSampleBound(maxRaw, int8(needBias))
				if needVal > int64(jxrLongWordMax)>>hardGain {
					return p, fmt.Errorf("%w: JPEG XR lossless float32 dynamic range", ErrUnsupported)
				}
				bias := needBias
				if needVal <= safeLimit {
					candidate := needBias
					if hasNormal {
						candidate = max(128-int(maxExp), needBias)
					}
					candidate = min(127, max(-128, candidate))
					if jxrFloatSampleBound(maxRaw, int8(candidate)) <= safeLimit {
						bias = candidate
					} else {
						for b := candidate - 1; b >= needBias; b-- {
							if jxrFloatSampleBound(maxRaw, int8(b)) <= safeLimit {
								bias = b
								break
							}
						}
					}
				} else if err := jxrVerifyFloatLosslessCapacity(ctx, raster, base, n, !alpha && count == 3, p.scaled, overlap, int8(needBias)); err != nil {
					return p, err
				}
				p.exponent = int8(bias)
			}
		}
	}
	if o.Quantization != nil {
		for n, target := range [][]uint8{p.dcQuant, p.lpQuant, p.hpQuant} {
			values := o.Quantization.DC
			if n == 1 {
				values = o.Quantization.Lowpass
			} else if n == 2 {
				values = o.Quantization.Highpass
			}
			if len(values) != 0 && len(values) != 1 && len(values) != count {
				return p, fmt.Errorf("%w: encode quantization component count", ErrInvalid)
			}
			for index := range count {
				if len(values) == 1 {
					target[index] = values[0]
				} else if len(values) != 0 {
					target[index] = values[index]
				}
			}
		}
	}
	return p, nil
}

// jxrQuantizationWork 计算公开量化配置持有的数组容量，不把裁短切片视为释放底层数据
// 入参: ctx Context, q 可选量化配置, maximum 活动容量上限
// 返回: uint64 配置存储容量, error 资源或取消错误
func jxrQuantizationWork(ctx context.Context, q *Quantization, maximum uint64) (uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if q == nil {
		return 0, nil
	}
	if uint64(cap(q.Tiles)) > maximum/128 {
		return 0, fmt.Errorf("%w: quantization tile metadata", ErrLimit)
	}
	work, ok := jxrWorkTotal(maximum, 96, uint64(cap(q.DC)), uint64(cap(q.Lowpass)), uint64(cap(q.Highpass)), uint64(cap(q.Tiles))*128)
	if !ok {
		return 0, fmt.Errorf("%w: retained quantization parameters", ErrLimit)
	}
	for _, tile := range q.Tiles[:cap(q.Tiles)] {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		work, ok = jxrWorkTotal(maximum, work, uint64(cap(tile.DC)), uint64(cap(tile.LowpassIndex)), uint64(cap(tile.HighpassIndex)))
		if !ok {
			return 0, fmt.Errorf("%w: retained quantization indices", ErrLimit)
		}
		for _, groups := range [][][]uint8{tile.Lowpass, tile.Highpass} {
			if uint64(cap(groups)) > (maximum-work)/24 {
				return 0, fmt.Errorf("%w: quantization collection metadata", ErrLimit)
			}
			work += uint64(cap(groups)) * 24
			for _, values := range groups[:cap(groups)] {
				if uint64(cap(values)) > maximum-work {
					return 0, fmt.Errorf("%w: retained quantization collection", ErrLimit)
				}
				work += uint64(cap(values))
			}
		}
	}
	return work, nil
}

// snapshotJXRQuantization 按实际长度复制公开配置，可为继承透明度只保留每组首分量
// 入参: ctx Context, q 原始配置, maximum 副本额度, first 是否提取首分量
// 返回: *Quantization 独占配置, uint64 副本存储容量, error 资源或取消错误
func snapshotJXRQuantization(ctx context.Context, q *Quantization, maximum uint64, first bool) (*Quantization, uint64, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if q == nil {
		return nil, 0, ctx.Err()
	}
	work := uint64(96)
	if uint64(len(q.Tiles)) > maximum/128 || uint64(len(q.Tiles)) > uint64(^uint(0)>>1)/128 {
		return nil, 0, fmt.Errorf("%w: quantization snapshot tile count", ErrLimit)
	}
	work += uint64(len(q.Tiles)) * 128
	for _, values := range [][]uint8{q.DC, q.Lowpass, q.Highpass} {
		count := len(values)
		if first {
			count = min(count, 1)
		}
		if work > maximum || uint64(count) > maximum-work {
			return nil, 0, fmt.Errorf("%w: quantization snapshot vectors", ErrLimit)
		}
		work += uint64(count)
	}
	for _, tile := range q.Tiles {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		count := len(tile.DC)
		if first {
			count = min(count, 1)
		}
		var ok bool
		work, ok = jxrWorkTotal(maximum, work, uint64(count), uint64(len(tile.LowpassIndex)), uint64(len(tile.HighpassIndex)))
		if !ok {
			return nil, 0, fmt.Errorf("%w: quantization snapshot indices", ErrLimit)
		}
		for _, groups := range [][][]uint8{tile.Lowpass, tile.Highpass} {
			if uint64(len(groups)) > (maximum-work)/24 || uint64(len(groups)) > uint64(^uint(0)>>1)/24 {
				return nil, 0, fmt.Errorf("%w: quantization snapshot collections", ErrLimit)
			}
			work += uint64(len(groups)) * 24
			for _, values := range groups {
				count := len(values)
				if first {
					count = min(count, 1)
				}
				if uint64(count) > maximum-work {
					return nil, 0, fmt.Errorf("%w: quantization snapshot values", ErrLimit)
				}
				work += uint64(count)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	result := &Quantization{}
	if q.Tiles != nil {
		result.Tiles = make([]TileQuantization, len(q.Tiles))
	}
	for index, values := range [][]uint8{q.DC, q.Lowpass, q.Highpass} {
		if first {
			values = values[:min(len(values), 1)]
		}
		copy := cloneJXRQuant(values)
		switch index {
		case 0:
			result.DC = copy
		case 1:
			result.Lowpass = copy
		case 2:
			result.Highpass = copy
		}
	}
	for index, tile := range q.Tiles {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		target := &result.Tiles[index]
		*target = tile
		values := tile.DC
		if first {
			values = values[:min(len(values), 1)]
		}
		target.DC = cloneJXRQuant(values)
		var err error
		target.LowpassIndex, err = copyJXRQuantIndex(ctx, tile.LowpassIndex)
		if err != nil {
			return nil, 0, err
		}
		target.HighpassIndex, err = copyJXRQuantIndex(ctx, tile.HighpassIndex)
		if err != nil {
			return nil, 0, err
		}
		for band, groups := range [][][]uint8{tile.Lowpass, tile.Highpass} {
			var copy [][]uint8
			if groups != nil {
				copy = make([][]uint8, len(groups))
			}
			for group, values := range groups {
				if first {
					values = values[:min(len(values), 1)]
				}
				copy[group] = cloneJXRQuant(values)
			}
			if band == 0 {
				target.Lowpass = copy
			} else {
				target.Highpass = copy
			}
		}
	}
	return result, work, ctx.Err()
}

// planJXRQuantization 按最终分块网格统一预检并复制双平面量化，默认透明度继承主平面首分量
// 入参: ctx Context, source 已校验最终网格, options 公开选项, maximum 包含布局和设置的额度
// 返回: []jxrTileSetting 独占设置, uint64 仍活动的公开配置容量, error 参数、资源或取消错误
func planJXRQuantization(ctx context.Context, source *jxrStream, options EncodeOptions, maximum uint64) ([]jxrTileSetting, uint64, error) {
	configuration := [2]*Quantization{options.Quantization, nil}
	if source.alpha != nil {
		configuration[1] = options.Quantization
		if options.Alpha != nil {
			configuration[1] = options.Alpha.Quantization
		}
	}
	active := false
	for _, q := range configuration {
		active = active || q != nil && len(q.Tiles) != 0
	}
	if !active {
		quant := [2]jxrTileQuant{newJXRTileQuant(&source.primary)}
		if source.alpha != nil {
			quant[1] = newJXRTileQuant(source.alpha)
		}
		return []jxrTileSetting{{quant: quant, trim: options.Trim}}, 0, nil
	}
	count := (len(source.header.columns) - 1) * (len(source.header.rows) - 1)
	if uint64(count) > uint64(^uint(0)>>1)/jxrTileSettingBytes || uint64(count) > maximum/jxrTileSettingBytes {
		return nil, 0, fmt.Errorf("%w: adaptive quantization tile storage", ErrLimit)
	}
	retained := uint64(0)
	for index, q := range configuration {
		if q == nil || len(q.Tiles) == 0 || index == 1 && q == configuration[0] {
			continue
		}
		if len(q.Tiles) != count {
			return nil, 0, fmt.Errorf("%w: adaptive quantization tile count", ErrInvalid)
		}
		amount, err := jxrQuantizationWork(ctx, q, maximum-retained)
		if err != nil {
			return nil, 0, err
		}
		retained += amount
	}
	memory, ok := jxrWorkTotal(maximum, retained, source.headerWork(), jxrInputHeaderReserve, uint64(count)*jxrTileSettingBytes)
	if !ok {
		return nil, 0, fmt.Errorf("%w: adaptive quantization planning memory", ErrLimit)
	}
	var flags [2]uint8
	columns := len(source.header.columns) - 1
	for tile := range count {
		x, y := tile%columns, tile/columns
		macros := uint64(source.header.columns[x+1]-source.header.columns[x]) * uint64(source.header.rows[y+1]-source.header.rows[y])
		for index, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
			if plane == nil {
				continue
			}
			var values TileQuantization
			if q := configuration[index]; q != nil && len(q.Tiles) != 0 {
				values = q.Tiles[tile]
			}
			channels := plane.channels
			if index == 1 && options.Alpha == nil {
				channels = source.primary.channels
			}
			work, fields, err := checkJXRQuantTile(ctx, values, plane, channels, macros)
			if err != nil {
				return nil, 0, err
			}
			if work > maximum-memory {
				return nil, 0, fmt.Errorf("%w: adaptive quantization collections", ErrLimit)
			}
			memory += work
			flags[index] |= fields
		}
	}
	for index, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
		if plane != nil {
			plane.dcUniform = flags[index]&1 == 0
			plane.lpUniform = flags[index]&2 == 0
			plane.hpUniform = flags[index]&4 == 0
		}
	}
	result := make([]jxrTileSetting, count)
	for tile := range result {
		result[tile].trim = options.Trim
		for index, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
			if plane == nil {
				continue
			}
			var values TileQuantization
			q := configuration[index]
			if q != nil && len(q.Tiles) != 0 {
				values = q.Tiles[tile]
			}
			value, err := copyJXRQuantTile(ctx, values, plane, q != nil && len(q.Tiles) != 0)
			if err != nil {
				return nil, 0, err
			}
			if err := checkJXRTileQuant(plane, &value); err != nil {
				return nil, 0, err
			}
			result[tile].quant[index] = value
		}
	}
	return result, retained, ctx.Err()
}

// checkJXRQuantTile 校验单块集合、复用及核心索引，并计算精确容量的私有参数存储
// 入参: ctx Context, values 公开分块参数, plane 目标平面, channels 来源向量允许的分量数, macros 核心宏块数
// 返回: uint64 私有参数容量, uint8 改写的频带标志, error 参数或取消错误
func checkJXRQuantTile(ctx context.Context, values TileQuantization, plane *jxrPlane, channels int, macros uint64) (uint64, uint8, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	if len(values.Lowpass) > 16 || len(values.Highpass) > 16 || values.UseDC && len(values.Lowpass) != 0 || values.UseLowpass && (len(values.Highpass) != 0 || len(values.HighpassIndex) != 0) {
		return 0, 0, fmt.Errorf("%w: adaptive quantization collection or reuse", ErrInvalid)
	}
	if plane.bands == 3 && (len(values.Lowpass) != 0 || len(values.LowpassIndex) != 0 || values.UseDC) || plane.bands >= 2 && (len(values.Highpass) != 0 || len(values.HighpassIndex) != 0 || values.UseLowpass) {
		return 0, 0, fmt.Errorf("%w: adaptive quantization omitted band", ErrInvalid)
	}
	if n := len(values.DC); n != 0 && n != 1 && n != channels {
		return 0, 0, fmt.Errorf("%w: adaptive DC component count", ErrInvalid)
	}
	for _, groups := range [][][]uint8{values.Lowpass, values.Highpass} {
		for _, vector := range groups {
			if len(vector) != 0 && len(vector) != 1 && len(vector) != channels {
				return 0, 0, fmt.Errorf("%w: adaptive collection component count", ErrInvalid)
			}
		}
	}
	lp, hp := max(1, len(values.Lowpass)), max(1, len(values.Highpass))
	if values.UseLowpass {
		hp = lp
	}
	for band, indices := range [][]uint8{values.LowpassIndex, values.HighpassIndex} {
		if len(indices) != 0 && uint64(len(indices)) != macros {
			return 0, 0, fmt.Errorf("%w: adaptive macroblock index count", ErrInvalid)
		}
		count := lp
		if band == 1 {
			count = hp
		}
		for index, value := range indices {
			if index&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return 0, 0, err
				}
			}
			if int(value) >= count {
				return 0, 0, fmt.Errorf("%w: adaptive macroblock index", ErrInvalid)
			}
		}
	}
	flags := uint8(0)
	if len(values.DC) != 0 {
		flags |= 1
	}
	if len(values.Lowpass) != 0 || len(values.LowpassIndex) != 0 || values.UseDC {
		flags |= 2
	}
	if len(values.Highpass) != 0 || len(values.HighpassIndex) != 0 || values.UseLowpass {
		flags |= 4
	}
	work := uint64(plane.channels) + uint64(lp+hp)*uint64(24+plane.channels) + uint64(len(values.LowpassIndex)) + uint64(len(values.HighpassIndex))
	return work, flags, nil
}

// copyJXRQuantTile 在已完成容量预检后复制单块参数，宏块索引只属于核心而不包含软重叠邻域
// 入参: ctx Context, values 公开分块参数, plane 目标平面, adaptive 是否来自公开分块配置
// 返回: jxrTileQuant 独占参数, error 取消错误
func copyJXRQuantTile(ctx context.Context, values TileQuantization, plane *jxrPlane, adaptive bool) (jxrTileQuant, error) {
	q := jxrTileQuant{adaptive: adaptive, useDC: values.UseDC, useLP: values.UseLowpass}
	q.dc = expandJXRQuantVector(values.DC, plane.dcQuant, plane.channels)
	q.lpCount, q.hpCount = max(1, len(values.Lowpass)), max(1, len(values.Highpass))
	if q.useDC {
		q.lp = [][]uint8{q.dc}
	} else {
		q.lp = make([][]uint8, q.lpCount)
		for index := range q.lp {
			var vector []uint8
			if len(values.Lowpass) != 0 {
				vector = values.Lowpass[index]
			}
			q.lp[index] = expandJXRQuantVector(vector, plane.lpQuant, plane.channels)
			if len(values.Lowpass) != 0 && len(vector) == 0 {
				clear(q.lp[index])
			}
		}
	}
	if q.useLP {
		q.hpCount, q.hp = q.lpCount, q.lp
	} else {
		q.hp = make([][]uint8, q.hpCount)
		for index := range q.hp {
			var vector []uint8
			if len(values.Highpass) != 0 {
				vector = values.Highpass[index]
			}
			q.hp[index] = expandJXRQuantVector(vector, plane.hpQuant, plane.channels)
			if len(values.Highpass) != 0 && len(vector) == 0 {
				clear(q.hp[index])
			}
		}
	}
	var err error
	q.lpIndex, err = copyJXRQuantIndex(ctx, values.LowpassIndex)
	if err != nil {
		return jxrTileQuant{}, err
	}
	q.hpIndex, err = copyJXRQuantIndex(ctx, values.HighpassIndex)
	return q, err
}

// expandJXRQuantVector 将已校验的单值或分量向量展开为精确长度存储，透明度取来源首分量
// 入参: values 可选覆盖值, fallback 默认向量, channels 目标分量数
// 返回: []uint8 独占逐分量参数
func expandJXRQuantVector(values, fallback []uint8, channels int) []uint8 {
	if len(values) == 0 {
		values = fallback
	}
	result := make([]uint8, channels)
	for index := range result {
		if len(values) == 1 || channels == 1 && len(values) != 0 {
			result[index] = values[0]
		} else if len(values) != 0 {
			result[index] = values[index]
		}
	}
	return result
}

// copyJXRQuantIndex 分段复制已预留额度的宏块索引，避免大表复制长时间忽略取消
// 入参: ctx Context, source 来源索引
// 返回: []uint8 精确容量副本, error 取消错误
func copyJXRQuantIndex(ctx context.Context, source []uint8) ([]uint8, error) {
	if source == nil {
		return nil, ctx.Err()
	}
	result := make([]uint8, len(source))
	for at := 0; at < len(source); at += 16384 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		copy(result[at:min(at+16384, len(source))], source[at:min(at+16384, len(source))])
	}
	return result, ctx.Err()
}

// checkJXRQuantizationGeometry 无分块参数分配地预检多图配置，保证后图错误不在首图提交后才发现
// 入参: ctx Context, info 已规范化原生描述, options 含明确分块尺寸的选项
// 返回: error 网格、集合、索引或取消错误
func checkJXRQuantizationGeometry(ctx context.Context, info Info, options EncodeOptions) error {
	configuration := [2]*Quantization{options.Quantization, nil}
	alphaBand := options.Band
	if info.Alpha {
		configuration[1] = options.Quantization
		if options.Alpha != nil {
			configuration[1], alphaBand = options.Alpha.Quantization, options.Alpha.Band
		}
	}
	active := false
	for _, q := range configuration {
		active = active || q != nil && len(q.Tiles) != 0
	}
	if !active {
		return ctx.Err()
	}
	if options.TileSize.X <= 0 || options.TileSize.Y <= 0 {
		return fmt.Errorf("%w: adaptive tile geometry", ErrInvalid)
	}
	format, err := rasterPixelFormat(info)
	if err != nil {
		return err
	}
	width, height := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
	if jxrInputTransform(info, format.color, uint8(options.Orientation))&4 != 0 {
		width, height = height, width
	}
	internalColor := format.color
	if options.Subsampling == Subsampling420 {
		internalColor = 1
	} else if options.Subsampling == Subsampling422 {
		internalColor = 2
	}
	if options.Window != nil {
		width += uint64(options.Window.Left) + uint64(options.Window.Right)
		height += uint64(options.Window.Top) + uint64(options.Window.Bottom)
		if width%16 != 0 || height%16 != 0 {
			return fmt.Errorf("%w: adaptive window macroblock alignment", ErrInvalid)
		}
		width, height = width/16, height/16
	} else {
		width, height = (width+15)/16, (height+15)/16
		if (internalColor == 1 || internalColor == 2) && options.Overlap == OverlapTwo && width < 2 {
			width = 2
		}
	}
	sx, sy := uint64(options.TileSize.X/16), uint64(options.TileSize.Y/16)
	if sx == 0 || sy == 0 {
		return fmt.Errorf("%w: adaptive tile macroblock geometry", ErrInvalid)
	}
	if width > math.MaxUint32 || height > math.MaxUint32 || sx > math.MaxUint32 || sy > math.MaxUint32 {
		return fmt.Errorf("%w: adaptive tile partition geometry", ErrLimit)
	}
	adjustCols := options.HardTiles && options.Overlap == OverlapTwo && (internalColor == 1 || internalColor == 2)
	colEdges, err := jxrTileColumnBoundaries(uint32(width), uint32(sx), adjustCols, ^uint64(0))
	if err != nil {
		return fmt.Errorf("%w: adaptive tile partition geometry", ErrLimit)
	}
	rowEdges, err := jxrTileColumnBoundaries(uint32(height), uint32(sy), false, ^uint64(0))
	if err != nil {
		return fmt.Errorf("%w: adaptive tile partition geometry", ErrLimit)
	}
	columns, rows := uint64(len(colEdges)-1), uint64(len(rowEdges)-1)
	for _, q := range configuration {
		if q != nil && len(q.Tiles) != 0 && uint64(len(q.Tiles)) != columns*rows {
			return fmt.Errorf("%w: adaptive quantization tile count", ErrInvalid)
		}
	}
	for tile := uint64(0); tile < columns*rows; tile++ {
		col, row := tile%columns, tile/columns
		w, h := uint64(colEdges[col+1]-colEdges[col]), uint64(rowEdges[row+1]-rowEdges[row])
		for index, q := range configuration {
			if q == nil || len(q.Tiles) == 0 {
				continue
			}
			plane := jxrPlane{channels: info.Channels, bands: uint8(options.Band)}
			channels := info.Channels
			if index == 1 {
				plane.channels, plane.bands = 1, uint8(alphaBand)
				if options.Alpha != nil {
					channels = 1
				}
			}
			if _, _, err := checkJXRQuantTile(ctx, q.Tiles[tile], &plane, channels, w*h); err != nil {
				return err
			}
		}
	}
	return ctx.Err()
}

// encodeJXRNative 完成原生图像的头部、变换、分块编码及文件封装，独立码流不携带分辨率或ICC
// 入参: ctx 已规范化Context, raster 原生图像, layout 编码布局, options 输出参数，内存上限包含原图存储
// 返回: []byte 完整文件或独立码流, error 格式、精度、资源或取消错误
func encodeJXRNative(ctx context.Context, raster *Raster, layout jxrStream, options jxrNativeOutput) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.format != FormatJXR && options.format != FormatCodestream || options.stream.budget != nil {
		return nil, fmt.Errorf("%w: JPEG XR native output format or budget", ErrInvalid)
	}
	rasterWork, err := jxrRasterWork(raster, options.stream.maxMemory)
	if err != nil {
		return nil, err
	}
	if options.stream.maxBytes == 0 {
		return nil, &LimitError{Resource: "JPEG XR native output bytes", Limit: 0, Required: 1}
	}
	if rasterWork >= options.stream.maxMemory {
		return nil, &LimitError{Resource: "JPEG XR native output memory", Limit: options.stream.maxMemory, Required: jxrSaturatedAdd(rasterWork, 1)}
	}
	var identifier [16]byte
	var pixelFormat jxrPixelFormat
	if options.format == FormatJXR {
		identifier, pixelFormat, err = jxrRasterIdentifier(raster.info, options.pixelFormat)
		if err != nil {
			return nil, err
		}
	} else if options.pixelFormat != ([16]byte{}) {
		return nil, fmt.Errorf("%w: JPEG XR identifier in bare codestream", ErrInvalid)
	}
	available := options.stream.maxMemory - rasterWork
	s, err := newJXRInputStream(ctx, raster, layout, options.stream.maxPixels, available)
	if err != nil {
		return nil, jxrPlanLimitError(err, options.stream.maxMemory, rasterWork)
	}
	if options.format == FormatJXR {
		validation := uint64(32768) + uint64(len(s.packets))*8 + uint64(len(s.header.columns)+len(s.header.rows)+len(s.profiles))*4
		if validation >= available {
			return nil, &LimitError{Resource: "JPEG XR file validation memory", Limit: available, Required: jxrSaturatedAdd(validation, 1)}
		}
		available -= validation
	}
	coeff, err := forwardJXRNative(ctx, raster, &s, &options.stream, available)
	if err != nil {
		return nil, err
	}
	settingsWork, err := options.stream.tileWork(ctx, available)
	if err != nil {
		return nil, err
	}
	work, ok := jxrWorkTotal(available, coeff.outputWork(&s), uint64(len(s.packets))*8, settingsWork)
	if !ok || work >= available {
		required := jxrSaturatedAdd(coeff.outputWork(&s), jxrSaturatedAdd(uint64(len(s.packets))*8, jxrSaturatedAdd(settingsWork, 1)))
		return nil, &LimitError{Resource: "JPEG XR native stream working memory", Limit: available, Required: required}
	}
	budget := jxrWriteBudget{limit: available - work}
	options.stream.maxMemory, options.stream.budget = available, &budget
	data, err := coeff.writeStream(ctx, &s, options.stream)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.format == FormatCodestream {
		return data, nil
	}
	meta := raster.info
	if options.resolution != nil {
		meta.Resolution = *options.resolution
	}
	if len(options.iccProfile) != 0 {
		meta.ICCProfile = options.iccProfile
	}
	if options.colorHint != nil {
		meta.ColorHint = *options.colorHint
	}
	if options.colorEncoding != nil {
		meta.ColorEncoding = options.colorEncoding
	}
	meta.EXIF, meta.GPS, meta.XMP, meta.IPTC, meta.Description = options.exif, options.gps, options.xmp, options.iptc, options.description
	c := jxrContainer{width: s.header.width, height: s.header.height, transform: uint32(s.header.transform), format: pixelFormat, pixelFormat: identifier, image: data, imageBands: uint32(s.primary.bands), profiles: s.profiles}
	c.setInfoMetadata(meta)
	w := newJXRBitWriter(ctx, options.stream.maxBytes, budget.limit)
	w.budget = &budget
	writeJXRContainer(&w, &c, options.stream.maxPixels)
	if w.err != nil {
		return nil, w.err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return w.data, nil
}

// forwardJXRNative 共用原生编码的量化预检、样本装载及正变换，不生成完整压缩输出
// 入参: ctx 已规范化Context, raster 原生图像, s 已校验头部, options 精度及分块参数, available 除原图外可用容量
// 返回: jxrCoefficientImage 完整系数图像, error 格式、精度、资源或取消错误
func forwardJXRNative(ctx context.Context, raster *Raster, s *jxrStream, options *jxrStreamOutput, available uint64) (jxrCoefficientImage, error) {
	var planes [2]*jxrCoefficients
	return forwardJXRNativeWindowStorage(ctx, raster, s, options, available, &planes, nil)
}

// forwardJXRNativeWindowStorage 共用样本容量及参数预检，支持显式样本窗装载并按窗口核心执行完整变换或局部软重叠变换
// 入参: ctx Context, raster 原生输入, s 已校验布局, options 编码精度, available 活跃容量, planes 独占缓冲, window 非nil时启用window.sample显式样本窗，为nil或window.core.Empty()时执行完整变换
// 返回: jxrCoefficientImage 当前系数视图, error 参数、容量、数值或取消错误
func forwardJXRNativeWindowStorage(ctx context.Context, raster *Raster, s *jxrStream, options *jxrStreamOutput, available uint64, planes *[2]*jxrCoefficients, window *jxrForwardWindow) (jxrCoefficientImage, error) {
	if s == nil || options == nil || planes == nil || s.alpha == nil && planes[1] != nil {
		return jxrCoefficientImage{}, fmt.Errorf("%w: JPEG XR forward storage layout", ErrInvalid)
	}
	raster = jxrSyncRasterChroma(raster, s)
	if options.tiles == nil {
		q := [2]jxrTileQuant{newJXRTileQuant(&s.primary)}
		if s.alpha != nil {
			q[1] = newJXRTileQuant(s.alpha)
		}
		options.tiles = []jxrTileSetting{{quant: q}}
	}
	tiles := (len(s.header.columns) - 1) * (len(s.header.rows) - 1)
	if len(options.tiles) != 1 && len(options.tiles) != tiles {
		return jxrCoefficientImage{}, fmt.Errorf("%w: JPEG XR native tile settings", ErrInvalid)
	}
	settingsWork, err := options.tileWork(ctx, available)
	if err != nil {
		return jxrCoefficientImage{}, err
	}
	working, limit := s.coefficientWork(), available-settingsWork
	if working >= limit {
		return jxrCoefficientImage{}, &LimitError{Resource: "JPEG XR native coefficient memory", Limit: limit, Required: jxrSaturatedAdd(working, 1)}
	}
	for _, setting := range options.tiles {
		if err := ctx.Err(); err != nil {
			return jxrCoefficientImage{}, err
		}
		if setting.trim > 15 || !s.header.trim && setting.trim != 0 {
			return jxrCoefficientImage{}, fmt.Errorf("%w: JPEG XR native trim", ErrInvalid)
		}
		for n, p := range [2]*jxrPlane{&s.primary, s.alpha} {
			if p != nil {
				if err := checkJXRTileQuant(p, &setting.quant[n]); err != nil {
					return jxrCoefficientImage{}, err
				}
			}
		}
	}
	count := uint64(s.header.columns[len(s.header.columns)-1]) * uint64(s.header.rows[len(s.header.rows)-1])
	for index, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
		storage := planes[index]
		if plane == nil || storage == nil {
			continue
		}
		if plane.channels < 1 || !storage.plane.equal(*plane) || storage.longWord != s.header.longWord {
			return jxrCoefficientImage{}, fmt.Errorf("%w: reused JPEG XR transform plane", ErrInvalid)
		}
		for _, term := range [3][3]uint64{
			{uint64(cap(storage.low)), count * uint64(plane.channels) * 16, 8},
			{uint64(cap(storage.high)), count * uint64(plane.channels) * 256, 8},
			{uint64(cap(storage.info)), count, 6},
		} {
			if term[0] < term[1] {
				return jxrCoefficientImage{}, &LimitError{Resource: "reused JPEG XR transform capacity", Limit: term[0], Required: term[1]}
			}
			extra := term[0] - term[1]
			if extra > (limit-working)/term[2] {
				return jxrCoefficientImage{}, &LimitError{Resource: "retained JPEG XR transform capacity", Limit: limit, Required: jxrSaturatedAdd(working, jxrSaturatedMul(extra, term[2]))}
			}
			working += extra * term[2]
			if working >= limit {
				return jxrCoefficientImage{}, &LimitError{Resource: "retained JPEG XR transform capacity", Limit: limit, Required: jxrSaturatedAdd(working, 1)}
			}
		}
	}
	allocated := uint64(0)
	for n, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p == nil {
			continue
		}
		c := planes[n]
		if c == nil {
			var err error
			c, err = newJXRTransformPlane(ctx, &s.header, p, available-allocated)
			if err != nil {
				return jxrCoefficientImage{}, err
			}
		} else {
			if err := ctx.Err(); err != nil {
				return jxrCoefficientImage{}, err
			}
			c.low, c.high, c.info = c.low[:int(count)*p.channels*16], c.high[:int(count)*p.channels*256], c.info[:int(count)]
			c.plane = *p
			clear(c.low)
			clear(c.high)
			clear(c.info)
		}
		planes[n] = c
		c.workers = options.workers
		amount := uint64(cap(c.low)+cap(c.high))*8 + uint64(cap(c.info))*6
		if amount > available-allocated {
			return jxrCoefficientImage{}, &LimitError{Resource: "retained JPEG XR transform capacity", Limit: available, Required: jxrSaturatedAdd(allocated, amount)}
		}
		allocated += amount
		var sample jxrSampleWindow
		if window != nil {
			sample = window.sample
		}
		if err := c.loadSamplesWindow(ctx, raster, &s.header, n != 0, options.exact, sample); err != nil {
			return jxrCoefficientImage{}, err
		}
		var err error
		if window == nil || window.core.Empty() {
			err = c.forward(ctx, &s.header)
		} else {
			err = c.forwardWindow(ctx, &s.header, *window)
		}
		if err != nil {
			return jxrCoefficientImage{}, err
		}
	}
	return jxrCoefficientImage{header: s.header, primary: *planes[0], alpha: planes[1]}, nil
}

// jxrRasterWork 计算原生图像已持有的样本、平面及元数据容量，不仅计算可见样本长度
// 入参: raster 有效原生图像, maximum 总工作容量上限
// 返回: uint64 原图存储预算, error 图像状态或资源错误
func jxrRasterWork(raster *Raster, maximum uint64) (uint64, error) {
	if raster == nil || raster.Bounds().Empty() || len(raster.info.Components) == 0 || raster.native == nil && len(raster.planes) == 0 || raster.native != nil && len(raster.planes) != 0 {
		return 0, fmt.Errorf("%w: JPEG XR input raster storage", ErrInvalid)
	}
	work := uint64(512)
	if raster.info.ColorEncoding != nil {
		work += 4
	}
	for _, size := range []uint64{uint64(cap(raster.info.Components)) * 128, uint64(cap(raster.planes)) * 128, uint64(cap(raster.info.ICCProfile)), uint64(cap(raster.info.EXIF)), uint64(cap(raster.info.GPS)), uint64(cap(raster.info.XMP)), uint64(cap(raster.info.IPTC)), jxrDescriptionWork(raster.info.Description)} {
		if work > maximum || size > maximum-work {
			return 0, &LimitError{Resource: "JPEG XR retained raster metadata", Limit: maximum, Required: jxrSaturatedAdd(work, size)}
		}
		work += size
	}
	for n := -1; n < len(raster.planes); n++ {
		plane := raster.native
		if n >= 0 {
			plane = &raster.planes[n]
		}
		if plane == nil {
			continue
		}
		for _, size := range []uint64{uint64(cap(plane.values)) * 2, uint64(cap(plane.words)) * 4} {
			if work > maximum || size > maximum-work {
				return 0, &LimitError{Resource: "JPEG XR retained raster samples", Limit: maximum, Required: jxrSaturatedAdd(work, size)}
			}
			work += size
		}
	}
	return work, nil
}

// jxrRasterIdentifier 选择与原生描述一致的标准GUID，优先保留已指定名称并采用非填充RGB格式
// 入参: info 图像信息, specified 明确指定的标识，零值自动选择
// 返回: [16]byte 标准标识, jxrPixelFormat 文件格式, error 格式或绑定错误
func jxrRasterIdentifier(info Info, specified [16]byte) ([16]byte, jxrPixelFormat, error) {
	format, err := rasterPixelFormat(info)
	if err != nil {
		return [16]byte{}, jxrPixelFormat{}, err
	}
	if specified != ([16]byte{}) {
		candidate, err := readJXRPixelFormat(specified[:])
		if err != nil {
			return [16]byte{}, jxrPixelFormat{}, err
		}
		if candidate.name == "BlackWhite" && format.depth == 15 {
			candidate.depth = 15
		}
		if !jxrPixelFormatMatches(candidate, format) {
			return [16]byte{}, jxrPixelFormat{}, fmt.Errorf("%w: JPEG XR requested output pixel format", ErrInvalid)
		}
		return specified, candidate, nil
	}
	var fallback [16]byte
	var selected jxrPixelFormat
	for id := byte(5); id <= 0x56; id++ {
		candidate, ok := jxrLookupPixelFormatID(id, format.depth)
		if !ok || !jxrPixelFormatMatches(candidate, format) {
			continue
		}
		var identifier [16]byte
		copy(identifier[:], jxrPixelPrefix)
		identifier[15] = id
		if candidate.name == info.PixelFormat {
			return identifier, candidate, nil
		}
		if fallback == ([16]byte{}) || id == 0x0d && format.color == 7 && format.depth == 1 && !format.alpha {
			fallback, selected = identifier, candidate
		}
	}
	if fallback == ([16]byte{}) {
		return fallback, selected, fmt.Errorf("%w: JPEG XR container cannot describe native pixel format", ErrUnsupported)
	}
	return fallback, selected, nil
}

// jxrPlanLimitError 将零分配规划预检哨兵转换为包含真实额度的资源上限错误
// 入参: err 规划阶段返回的错误, limit 当前总内存上限, retained 已占用常驻容量
// 返回: error 原错误或带真实额度的资源错误
func jxrPlanLimitError(err error, limit, retained uint64) error {
	if err == errJXREncodingPlanMemory || err == errJXRInputHeaderMemory {
		return &LimitError{Resource: "encoding plan memory", Limit: limit, Required: jxrSaturatedAdd(retained, jxrInputHeaderReserve)}
	}
	return err
}

// EncodeRaster 编码原生分量，不进行ICC转换、解除预乘或隐式重采样
// 编码失败不写入输出流，底层写入失败可能已输出部分字节；不关闭调用方输出流
// 入参: ctx Context，nil采用Background, w 输出流, raster 原生图像, opts 编码选项，nil采用无损JXR
// 返回: error 格式、精度、资源、取消或写入错误
func EncodeRaster(ctx context.Context, w io.Writer, raster *Raster, opts *EncodeOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	return wrapJXRError("EncodeRaster", encodeRasterCore(ctx, w, raster, opts))
}

// encodeRasterCore 执行原生图像编码与写入核心逻辑，由外层公开入口统一包装错误操作名
// 入参: ctx 已规范化Context, w 输出流, raster 原生图像, opts 编码选项
// 返回: error 格式、精度、资源、取消或写入错误
func encodeRasterCore(ctx context.Context, w io.Writer, raster *Raster, opts *EncodeOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if w == nil || raster == nil {
		return fmt.Errorf("%w: nil output or raster", ErrInvalid)
	}
	if raster.info.ImageType&ImageTypePreview != 0 && (opts == nil || opts.Format != FormatCodestream) {
		return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	data, err := encodeRasterData(ctx, raster, opts)
	if err != nil {
		return err
	}
	return writeEncoderOutput(ctx, w, data)
}

// encodeRasterData 编码一幅完整原生图像，是否为多图文件中的预览图由外层容器操作校验
// 入参: ctx 已规范化Context, raster 原生图像, opts 编码选项
// 返回: []byte 独占编码缓冲, error 格式、精度、资源或取消错误
func encodeRasterData(ctx context.Context, raster *Raster, opts *EncodeOptions) ([]byte, error) {
	if opts != nil && opts.Uncompressed {
		return encodeUncompressedJXR(ctx, raster, *opts)
	}
	if opts != nil && opts.SeparateAlpha {
		return encodeJXRSeparateAlpha(ctx, raster, *opts)
	}
	layout, output, err := makeJXREncodingPlan(ctx, raster, opts)
	if err != nil {
		if err == errJXREncodingPlanMemory || err == errJXRInputHeaderMemory {
			retained, _ := jxrRasterWork(raster, output.stream.maxMemory)
			return nil, jxrPlanLimitError(err, output.stream.maxMemory, retained)
		}
		return nil, err
	}
	return encodeJXRNative(ctx, raster, layout, output)
}

// writeEncoderOutput 分段写入完整编码数据，保留底层错误并拒绝无错误的短写入
// 入参: ctx Context, w 输出流, data 完整编码数据
// 返回: error 写入协议、底层或取消错误
func writeEncoderOutput(ctx context.Context, w io.Writer, data []byte) error {
	for len(data) != 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		part := data[:min(len(data), 32768)]
		n, err := w.Write(part)
		if n < 0 || n > len(part) {
			return fmt.Errorf("%w: output write count", ErrInvalid)
		}
		if err != nil {
			return err
		}
		if n != len(part) {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return ctx.Err()
}

// sameInterfaceTarget 比较两个Interface值是否指向同一底层对象，不可比较类型安全返回false
// 入参: a, b Interface值
// 返回: bool 是否为同一对象
func sameInterfaceTarget(a, b any) (same bool) {
	defer func() {
		if recover() != nil {
			same = false
		}
	}()
	return a == b
}

// encodeJXRCodestreamTilesSequential 从可重读提供器两遍编码独立码流到普通Writer，不持有整幅原图或拼接完整压缩结果
// 入参: ctx 已规范化Context, w 顺序输出, info 原生描述, read 可重读逐块提供器, opts 明确分块参数
// 返回: error 参数、能力、精度、资源、源、取消或写入错误
func encodeJXRCodestreamTilesSequential(ctx context.Context, w io.Writer, info Info, read RasterTileReader, opts *EncodeOptions) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if settings.Format != FormatUnknown && settings.Format != FormatCodestream || settings.SeparateAlpha || settings.TileSize.X == 0 || settings.TileSize.Y == 0 {
		return fmt.Errorf("%w: sequential tile input requires explicit bare-stream tiles", ErrUnsupported)
	}
	settings.Format = FormatCodestream
	info, retained, err := prepareJXRTileInfo(ctx, info, settings)
	if err != nil {
		return err
	}
	metadata := Raster{info: info}
	source, output, err := makeJXREncodingPlanRetained(ctx, &metadata, &settings, retained)
	if err != nil {
		return jxrPlanLimitError(err, output.stream.maxMemory, retained)
	}
	plan, options, err := newJXRForwardPlan(ctx, &source, output.stream, retained)
	if err != nil {
		return err
	}
	options.maxMemory -= retained
	return encodeJXRForwardPasses(ctx, w, plan, func(destination io.WriteSeeker) error {
		return encodeJXRProvidedTilesTo(ctx, destination, info, &source, options, read)
	})
}

// newJXRForwardPlan 共用整图及提供器输入的有界包目录规划，并预留仍活动的源描述和原编码参数
// 入参: ctx Context, source 完整编码几何, options 当前输出参数, retained 源图或描述容量
// 返回: *jxrForwardPlan 两遍共享目录, jxrStreamOutput 扣除规划后的参数, error 几何或容量错误
func newJXRForwardPlan(ctx context.Context, source *jxrStream, options jxrStreamOutput, retained uint64) (*jxrForwardPlan, jxrStreamOutput, error) {
	if source == nil || len(source.header.columns) < 2 || len(source.header.rows) < 2 || len(source.header.columns) > 4097 || len(source.header.rows) > 4097 || source.primary.bands > 3 {
		return nil, options, fmt.Errorf("%w: sequential planning geometry", ErrInvalid)
	}
	count := uint64(len(source.header.columns)-1) * uint64(len(source.header.rows)-1)
	if source.header.frequency {
		count *= uint64(4 - source.primary.bands)
	}
	if count == 0 || count > uint64(^uint(0)>>1)/48 {
		return nil, options, &LimitError{Resource: "sequential packet directory", Limit: uint64(^uint(0)>>1) / 48, Required: count}
	}
	settingsWork, err := options.tileWork(ctx, options.maxMemory)
	if err != nil {
		return nil, options, err
	}
	reserved := uint64(512) + count*48
	for _, amount := range []uint64{source.headerWork(), settingsWork, retained} {
		if reserved > options.maxMemory || amount > options.maxMemory-reserved {
			return nil, options, &LimitError{Resource: "sequential retained planning memory", Limit: options.maxMemory, Required: jxrSaturatedAdd(reserved, amount)}
		}
		reserved += amount
	}
	reserved -= retained
	plan := &jxrForwardPlan{packets: make([]jxrForwardPacket, int(count))}
	options.maxMemory -= reserved
	options.forward = plan
	return plan, options, nil
}

// encodeJXRForwardPasses 先测量并校验完整包目录，再以只前进的Writer提交相同内容，不重用上一遍的系数或包缓存
// 入参: ctx Context, w 顺序输出, plan 共享目录, encode 每遍重新创建独占状态的编码器
// 返回: error 任一遍资源、数据、源、取消或输出错误
func encodeJXRForwardPasses(ctx context.Context, w io.Writer, plan *jxrForwardPlan, encode func(io.WriteSeeker) error) error {
	counting := jxrCountingSeeker{}
	if err := encode(&counting); err != nil {
		return err
	}
	if plan.total > math.MaxInt64 || counting.position != int64(plan.total) || counting.length != int64(plan.total) {
		return fmt.Errorf("%w: sequential encoded extent", ErrInvalid)
	}
	plan.ready = true
	forward := jxrForwardWriter{writer: w}
	if err := ctx.Err(); err != nil {
		return err
	}
	return encode(&forward)
}

// Write 累加本次虚拟写入范围，不分配输出缓冲或忽略有符号地址溢出
// 入参: data 当前头部、包或回填字节
// 返回: int 计入长度, error 地址错误
func (w *jxrCountingSeeker) Write(data []byte) (int, error) {
	if w.position < 0 || int64(len(data)) > math.MaxInt64-w.position {
		return 0, fmt.Errorf("%w: sequential count address", ErrLimit)
	}
	w.position += int64(len(data))
	w.length = max(w.length, w.position)
	return len(data), nil
}

// Seek 模拟已经写入范围内的索引回填，不允许创建空洞或无效位置
// 入参: offset 位移, whence 定位基准
// 返回: int64 新位置, error 定位错误
func (w *jxrCountingSeeker) Seek(offset int64, whence int) (int64, error) {
	base := int64(0)
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		base = w.position
	case io.SeekEnd:
		base = w.length
	default:
		return 0, fmt.Errorf("%w: sequential count seek origin", ErrInvalid)
	}
	if offset < -base || offset > w.length-base {
		return 0, fmt.Errorf("%w: sequential count seek range", ErrInvalid)
	}
	w.position = base + offset
	return w.position, nil
}

// Write 原样转发顺序写入并仅累计合法的实际写入量，协议错误仍交由共用写入检查传播
// 入参: data 当前输出字节
// 返回: int 实际写入量, error 原始写入错误
func (w *jxrForwardWriter) Write(data []byte) (int, error) {
	count, err := w.writer.Write(data)
	if count >= 0 && count <= len(data) {
		w.position += int64(count)
	}
	return count, err
}

// Seek 只允许查询当前位置，任何需要回退或跳过的生产路径都属于非定位输出能力错误
// 入参: offset 须为零, whence 须为SeekCurrent
// 返回: int64 当前位置, error 不支持的定位请求
func (w *jxrForwardWriter) Seek(offset int64, whence int) (int64, error) {
	if offset != 0 || whence != io.SeekCurrent {
		return 0, fmt.Errorf("%w: sequential output cannot seek", ErrUnsupported)
	}
	return w.position, nil
}

// jxrSyncRasterChroma 当原生子采样图像显式指定色度相位时构造浅拷贝视图以匹配编码平面声明，不修改调用方元数据
// 入参: raster 原始图像, s 已规划的编码码流头部
// 返回: *Raster 原始图像或同步相位后的浅拷贝视图
func jxrSyncRasterChroma(raster *Raster, s *jxrStream) *Raster {
	if raster == nil || s.header.color != 1 && s.header.color != 2 || !s.primary.explicitChromaX && !s.primary.explicitChromaY {
		return raster
	}
	format, err := rasterPixelFormat(raster.info)
	if err != nil {
		return raster
	}
	view := *raster
	view.info.Components = cloneComponentInfo(raster.info.Components)
	src := jxrRasterSource{container: jxrContainer{width: s.header.width, height: s.header.height, transform: uint32(s.header.transform), format: format}, primary: *s}
	for i := 1; i < 3; i++ {
		expected := src.componentInfo(i)
		view.info.Components[i].XOffset = expected.XOffset
		view.info.Components[i].YOffset = expected.YOffset
		view.info.Components[i].XPositionKnown = expected.XPositionKnown
		view.info.Components[i].YPositionKnown = expected.YPositionKnown
	}
	return &view
}

// encodeUncompressedJXR 按HD Photo / JPEG XR无压缩容器扩展标签0xBC03=1将原生图像打包写入无压缩JXR容器
// 入参: ctx 已规范化Context, raster 原生图像, options 已请求无压缩模式的编码选项
// 返回: []byte 完整无压缩JXR容器字节, error 选项、格式、样本、资源或取消错误
func encodeUncompressedJXR(ctx context.Context, raster *Raster, options EncodeOptions) ([]byte, error) {
	if err := checkJXREncodeOptions(options); err != nil {
		return nil, err
	}
	limits := options.Limits.normalized()
	maxBytes := options.MaxOutputBytes
	if maxBytes == 0 {
		maxBytes = 64 << 20
	}
	rasterWork, err := jxrRasterWork(raster, limits.MaxMemoryBytes)
	if err != nil {
		return nil, err
	}
	if options.SeparateAlpha && !raster.info.Alpha {
		return nil, fmt.Errorf("%w: separate alpha without source alpha", ErrInvalid)
	}
	meta := jxrWithContainerMetadata(raster.info, &options)
	retained, required, ok := jxrContainerMetadataRetained(rasterWork, limits.MaxMemoryBytes, raster.info, &options)
	if !ok {
		return nil, &LimitError{Resource: "uncompressed container metadata memory", Limit: limits.MaxMemoryBytes, Required: required}
	}
	identifier, format, err := jxrRasterIdentifier(raster.info, [16]byte{})
	if err != nil {
		return nil, err
	}
	transform := jxrInputTransform(raster.info, format.color, uint8(options.Orientation))
	codedW, codedH := uint64(raster.Bounds().Dx()), uint64(raster.Bounds().Dy())
	if transform&4 != 0 {
		codedW, codedH = codedH, codedW
	}
	if err := checkJXRDimensions(codedW, codedH, limits.MaxPixels); err != nil {
		return nil, err
	}
	layout, err := jxrMakeUncompressedLayout(identifier[15], format, codedW, codedH, options.SeparateAlpha)
	if err != nil {
		return nil, err
	}
	if format.name == "BlackWhite" && format.depth == 15 {
		format.depth = 0
	}
	rawTotal := layout.mainBytes + layout.alphaBytes
	if rawTotal > maxBytes {
		return nil, &LimitError{Resource: "max output bytes", Limit: maxBytes, Required: rawTotal, Detail: "uncompressed image payload"}
	}
	if retained > limits.MaxMemoryBytes || rawTotal > limits.MaxMemoryBytes-retained || rawTotal > uint64(^uint(0)>>1) {
		return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, rawTotal), Detail: "uncompressed image payload memory"}
	}
	imageBytes, alphaBytes, err := jxrPackUncompressedImage(ctx, raster, &layout, transform)
	if err != nil {
		return nil, err
	}
	c := jxrContainer{width: codedW, height: codedH, format: format, pixelFormat: identifier, transform: uint32(transform), image: imageBytes, alpha: alphaBytes, uncompressed: true}
	c.setInfoMetadata(meta)
	w := newJXRBitWriter(ctx, maxBytes, limits.MaxMemoryBytes-retained-rawTotal)
	writeJXRContainer(&w, &c, limits.MaxPixels)
	if w.err != nil {
		return nil, w.err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return w.data, nil
}

// jxrPackUncompressedImage 按给定无压缩布局与方向变换打包原生图像的主区及可选独立透明度字节
// 入参: ctx Context, raster 原生图像, layout 已校验无压缩布局, transform 编码到显示方向变换
// 返回: []byte 主图像无压缩字节, []byte 可选独立透明度无压缩字节, error 样本范围或取消错误
func jxrPackUncompressedImage(ctx context.Context, raster *Raster, layout *jxrUncompressedLayout, transform uint8) ([]byte, []byte, error) {
	mainData := make([]byte, int(layout.mainBytes))
	var alphaData []byte
	if layout.separateAlpha {
		alphaData = make([]byte, int(layout.alphaBytes))
	}
	codedW, codedH := int(layout.width), int(layout.height)
	totalComp := layout.channels
	if layout.hasAlpha {
		totalComp++
	}
	components := make([]Component, totalComp)
	for ci := range totalComp {
		comp, err := raster.component(ci)
		if err != nil {
			return nil, nil, err
		}
		components[ci] = comp
	}
	if layout.planar {
		step := int(layout.sampleBytes)
		for ci := range totalComp {
			sx, sy, baseOffset, dstRowBytes, inAlpha := layout.planeSpec(ci)
			dst := mainData
			if inAlpha {
				dst = alphaData
			}
			cw, ch := codedW/sx, codedH/sy
			comp := components[ci]
			for cy := range ch {
				if err := ctx.Err(); err != nil {
					return nil, nil, err
				}
				rowDst := dst[baseOffset+uint64(cy)*dstRowBytes:]
				for cx := range cw {
					if cx&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, nil, err
						}
					}
					dx, dy := jxrRasterPosition(cx, cy, cw, ch, transform)
					val, err := comp.sampleBitsAt(dx, dy)
					if err != nil {
						return nil, nil, err
					}
					if err := jxrPackUncompressedSample(rowDst[cx*step:], val, layout.sampleBytes, layout.tenBit); err != nil {
						return nil, nil, err
					}
				}
			}
		}
		return mainData, alphaData, ctx.Err()
	}
	var samples [9]uint32
	for cy := range codedH {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		rowDst := mainData[uint64(cy)*layout.rowBytes:]
		var alphaRow []byte
		if layout.separateAlpha {
			alphaRow = alphaData[uint64(cy)*layout.alphaRowBytes:]
		}
		for cx := range codedW {
			if cx&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, nil, err
				}
			}
			dx, dy := jxrRasterPosition(cx, cy, codedW, codedH, transform)
			for ci := range totalComp {
				val, err := components[ci].sampleBitsAt(dx, dy)
				if err != nil {
					return nil, nil, err
				}
				samples[ci] = val
			}
			if err := jxrPackUncompressedPixel(layout, rowDst, alphaRow, cx, samples[:totalComp]); err != nil {
				return nil, nil, err
			}
		}
	}
	return mainData, alphaData, ctx.Err()
}
