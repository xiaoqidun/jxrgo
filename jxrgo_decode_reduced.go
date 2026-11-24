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
	"io"
	"math"
	"slices"
	"sort"
)

// jxrReducedRows 缓存两条已归一化粗宏块行，允许不同分量及独立透明度以原编码中心定位邻行
type jxrReducedRows struct {
	low      *jxrLowGridRows
	half     *jxrReconstructedRows
	geometry []jxrLowGrid
	offsets  []int
	cache    [2][]int64
	count    int
	height   int
	next     int
}

// jxrLowFrequencyPlan 保存原始包数和独立低带选择，排序副本仅为读取边界服务，不改变原索引或平面声明
type jxrLowFrequencyPlan struct {
	source   *jxrStream
	physical []uint64
	count    int
	selected int
}

// jxrLowFrequencyRows 只持有DC及可选LP的预测状态与当前低通行，不分配高通系数或恢复空间样本
type jxrLowFrequencyRows struct {
	bits    [2]jxrBits
	quant   [2]jxrTileQuant
	rows    [2][2]*jxrTileRow
	scratch [2]*jxrDecodedMacro
	dc      [2]jxrDC
	lp      [2]jxrLP
	image   jxrCoefficientImage
	width   int
	height  int
	next    int
	bands   int
}

// jxrLowGrid 将低频阶段映射为带原编码坐标中心的粗网格，代表值采用正方向平局舍入
type jxrLowGrid struct {
	data       []int64
	columns    int
	width      int
	height     int
	blockWidth int
	blockRows  int
	stride     int
	offset     int
	group      int
	shift      uint8
	gain       int64
	xStep      int
	yStep      int
	xCenter    float64
	yCenter    float64
	dc         bool
	longWord   bool
}

// jxrLowGridRows 将一次低频熵遍历的固定邻行结果逐行投影到粗代表值网格，描述数组原地复用
type jxrLowGridRows struct {
	inverse *jxrLowInverseRows
	primary []jxrLowGrid
	alpha   [1]jxrLowGrid
	divisor int
}

// jxrLowInverseRows 只在DC加LP模式保留首级原始及工作各三邻行，不执行二级逆变换或细网格滤波
type jxrLowInverseRows struct {
	rows     *jxrLowTileRows
	pipeline jxrInverseRows
	height   int
	width    int
	read     int
	nextRow  int
	dcOnly   bool
}

// jxrLowSpatialRows 消费空间包全部交错语法，仅以一个宏块保留高通熵工作，结果只持有当前低通行
type jxrLowSpatialRows struct {
	bits    jxrBits
	quant   [2]jxrTileQuant
	rows    [2]*jxrTileRow
	scratch [2]*jxrDecodedMacro
	dc      [2]jxrDC
	lp      [2]jxrLP
	hp      [2]jxrHP
	macro   jxrCoefficientImage
	image   jxrCoefficientImage
	width   int
	height  int
	next    int
	bands   int
	trim    uint8
}

// jxrLowTileReader 保存一种布局的当前列状态，具体指针避免每行装箱低通视图
type jxrLowTileReader struct {
	frequency *jxrLowFrequencyRows
	spatial   *jxrLowSpatialRows
}

// jxrLowTileRows 按原始块行调度全部列的选带读取，只保留当前行低通及当前块行熵状态
type jxrLowTileRows struct {
	source    jxrStream
	data      []byte
	plan      *jxrLowFrequencyPlan
	readers   []jxrLowTileReader
	image     jxrCoefficientImage
	maximum   uint64
	bands     int
	next      int
	tileRow   int
	activeRow int
}

// DecodeReduced 按二、四、八或十六缩减分母从必要重建阶段生成全采样独立原生粗图，保留原生数值、颜色、ICC及预乘状态
// 未知色度相位按ITU-T T.832默认居中相位处理，四倍及以上仅重建低频带并忽略未选频率包，逐行有界工作内存不含输出粗图，NewDecoderAt按需读取
// 入参: ctx Context，nil采用Background, divisor 缩减分母（二、四、八或十六）
// 返回: *Raster 独立完整采样原生粗图, error 参数、能力、数据、资源、关闭、读取或取消错误
func (d *Decoder) DecodeReduced(ctx context.Context, divisor int) (*Raster, error) {
	if d == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	raster, err := d.decodeReducedLocked(ctx, divisor)
	if err != nil {
		return nil, wrapJXRError("Decoder.DecodeReduced", err)
	}
	return raster, nil
}

// jxrReducedInfo 预检完整采样粗图及复制元数据所需容量，清除旧像素格式名，由原生粗图构造器生成完整采样格式名
// 入参: original 原图信息, divisor 缩减分母, maximum 可用输出额度
// 返回: Info 粗图描述, uint64 输出存储额度, error 几何或资源错误
func jxrReducedInfo(original Info, divisor int, maximum uint64) (Info, uint64, error) {
	if original.Bounds.Empty() || len(original.Components) == 0 || divisor != 2 && divisor != 4 && divisor != 8 && divisor != 16 {
		return Info{}, 0, fmt.Errorf("%w: reduced output geometry", ErrInvalid)
	}
	width, height := (original.Bounds.Dx()+divisor-1)/divisor, (original.Bounds.Dy()+divisor-1)/divisor
	work := uint64(512) + uint64(len(original.Components))*256 + uint64(len(original.ICCProfile)) + uint64(len(original.EXIF)) + uint64(len(original.GPS)) + uint64(len(original.XMP)) + uint64(len(original.IPTC)) + jxrDescriptionWork(original.Description)
	if original.ColorEncoding != nil {
		work += 4
	}
	if work > maximum {
		return Info{}, 0, fmt.Errorf("%w: reduced output metadata budget", ErrLimit)
	}
	for _, c := range original.Components {
		bytes := c.storageBytes()
		pixels := uint64(width) * uint64(height)
		if pixels > (maximum-work)/bytes || pixels > uint64(^uint(0)>>1)/bytes {
			return Info{}, 0, fmt.Errorf("%w: reduced output sample budget", ErrLimit)
		}
		work += pixels * bytes
	}
	info := original
	info.Bounds, info.PixelFormat = image.Rect(0, 0, width, height), ""
	info.Resolution.X = scaleJXRResolution(info.Resolution.X, width, original.Bounds.Dx())
	info.Resolution.Y = scaleJXRResolution(info.Resolution.Y, height, original.Bounds.Dy())
	info.Components = cloneComponentInfo(original.Components)
	for i := range info.Components {
		c := &info.Components[i]
		c.Bounds, c.XStep, c.YStep = info.Bounds, 1, 1
		c.XOffset, c.YOffset = 0, 0
		c.XPositionKnown, c.YPositionKnown = true, true
	}
	return info, work, nil
}

// decodeReducedLocked 在独占锁内预检总活动额度，再以各自两条粗行配对主流及独立透明度
// 入参: ctx 已规范化Context, divisor 二、四、八或十六
// 返回: *Raster 独立原生粗图, error 参数、数据、资源、能力或生命周期错误
func (d *Decoder) decodeReducedLocked(ctx context.Context, divisor int) (*Raster, error) {
	if d.closed {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(d.data) == 0 || divisor != 2 && divisor != 4 && divisor != 8 && divisor != 16 {
		return nil, fmt.Errorf("%w: reduced decoder parameters", ErrInvalid)
	}
	if d.source.container.uncompressed {
		return d.decodeUncompressedReducedLocked(ctx, divisor)
	}
	source := &d.source
	for _, stream := range [2]*jxrStream{&source.primary, source.alpha} {
		if stream == nil {
			continue
		}
		for _, p := range [2]*jxrPlane{&stream.primary, stream.alpha} {
			if p == nil {
				continue
			}
			if err := checkJXRModelComponents(p); err != nil {
				return nil, err
			}
		}
	}
	remaining := min(d.limits.MaxMemoryBytes, uint64(^uint(0)>>1))
	metadataExtra := uint64(cap(d.info.ICCProfile)) + uint64(cap(d.info.EXIF)) + uint64(cap(d.info.GPS)) + uint64(cap(d.info.XMP)) + uint64(cap(d.info.IPTC)) + jxrDescriptionWork(d.info.Description)
	for _, amount := range []uint64{d.inputWork(), d.retainedHeaderWork(), metadataExtra, 4096 + uint64(len(d.info.Components))*128} {
		if amount > remaining {
			return nil, fmt.Errorf("%w: reduced retained storage", ErrLimit)
		}
		remaining -= amount
	}
	var rowWork [2]uint64
	for i, stream := range [2]*jxrStream{&source.primary, source.alpha} {
		if stream == nil {
			continue
		}
		rowWork[i] = jxrReducedRowWork(stream, divisor)
		if rowWork[i] > remaining {
			return nil, fmt.Errorf("%w: reduced reconstruction budget", ErrLimit)
		}
		remaining -= rowWork[i]
	}
	info, outputWork, err := jxrReducedInfo(d.info, divisor, remaining)
	if err != nil {
		return nil, err
	}
	output, err := newRasterContext(ctx, info, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: outputWork})
	if err != nil {
		return nil, err
	}
	d.bindInputContext(ctx)
	defer d.bindInputContext(nil)
	var readers [2]*jxrReducedRows
	for i, stream := range [2]*jxrStream{&source.primary, source.alpha} {
		if stream == nil {
			continue
		}
		data := source.container.image
		if i == 1 {
			data = source.container.alpha
		}
		readers[i], err = newJXRReducedRows(ctx, data, stream, divisor, rowWork[i])
		if err != nil {
			return nil, err
		}
	}
	h := &source.primary.header
	width, height := (int(h.width)+divisor-1)/divisor, (int(h.height)+divisor-1)/divisor
	channels := max(3, source.primary.primary.channels, info.Channels)
	var posStorage [2][17]int
	var valStorage [16]int64
	var codeStorage [16]uint32
	positions := [2][]int{posStorage[0][:], posStorage[1][:]}
	for i, r := range readers {
		if r != nil && r.count > len(positions[i]) {
			positions[i] = make([]int, r.count)
		}
	}
	values, codes := valStorage[:], codeStorage[:]
	if channels > len(values) {
		values, codes = make([]int64, channels), make([]uint32, channels)
	} else {
		values, codes = values[:channels], codes[:channels]
	}
	opacityNumber := source.container.format.number
	for y := range height {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		visibleY := min(float64(h.height-1), float64(y*divisor)+float64(divisor-1)/2)
		for i, stream := range [2]*jxrStream{&source.primary, source.alpha} {
			if stream == nil {
				continue
			}
			r := readers[i]
			needed := 0
			for c := range r.count {
				_, py, err := r.position(c, float64(stream.header.left), float64(stream.header.top)+visibleY)
				if err != nil {
					return nil, err
				}
				positions[i][c] = py
				needed = max(needed, py/r.geometry[c].height)
			}
			if err := r.ensure(ctx, needed); err != nil {
				return nil, err
			}
		}
		for x := range width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			visibleX := min(float64(h.width-1), float64(x*divisor)+float64(divisor-1)/2)
			for c := range source.primary.primary.channels {
				px, _, err := readers[0].position(c, float64(h.left)+visibleX, float64(h.top)+visibleY)
				if err != nil {
					return nil, err
				}
				values[c], err = readers[0].at(c, px, positions[0][c])
				if err != nil {
					return nil, err
				}
			}
			count, err := jxrRasterBitsComponents(h, &source.primary.primary, source.container.format.number, values[:], false, codes[:])
			if err != nil || count != info.Channels {
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("%w: reduced output channel count", ErrInvalid)
			}
			px, py := jxrRasterPosition(x, y, width, height, uint8(source.container.transform))
			at := py*info.Bounds.Dx() + px
			for c := range count {
				output.planes[c].storeBits(at, codes[c])
			}
			if info.Alpha {
				stream, reader, pass := &source.primary, readers[0], 0
				if source.alpha != nil {
					stream, reader, pass = source.alpha, readers[1], 1
				}
				component, plane := stream.primary.channels, stream.alpha
				ax, _, err := reader.position(component, float64(stream.header.left)+visibleX, float64(stream.header.top)+visibleY)
				if err != nil {
					return nil, err
				}
				value, err := reader.at(component, ax, positions[pass][component])
				if err != nil {
					return nil, err
				}
				alphaValues, alphaCodes := [1]int64{value}, [1]uint32{}
				if _, err := jxrRasterBitsComponents(&stream.header, plane, opacityNumber, alphaValues[:], true, alphaCodes[:]); err != nil {
					return nil, err
				}
				output.planes[count].storeBits(at, alphaCodes[0])
			}
		}
	}
	for _, r := range readers {
		if r != nil {
			if err := r.ensure(ctx, r.height-1); err != nil {
				return nil, err
			}
		}
	}
	return output, ctx.Err()
}

// jxrReducedRowWork 计算所需重建阶段、两条粗代表值行及固定描述的合计，不包含最终图像和调用方输入
// 入参: source 完整流, divisor 二、四、八或十六
// 返回: uint64 工作额度，无效布局或溢出返回最大值
func jxrReducedRowWork(source *jxrStream, divisor int) uint64 {
	if source == nil || len(source.header.columns) < 2 {
		return ^uint64(0)
	}
	work := jxrLowGridRowWork(source, divisor)
	if divisor == 2 {
		work = jxrStreamingRowWork(source)
	}
	if work > ^uint64(0)-4096 {
		return ^uint64(0)
	}
	perMacro := uint64(0)
	totalChannels := uint64(0)
	for _, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
		if plane == nil {
			continue
		}
		if plane.channels < 1 || plane.channels > rasterMaxComponents {
			return ^uint64(0)
		}
		totalChannels += uint64(plane.channels)
		for component := range plane.channels {
			cells := uint64(1)
			if divisor != 16 {
				cells = uint64((16 / divisor) * (16 / divisor))
				if component > 0 && (plane.color == 1 || plane.color == 2) {
					cells /= 2
					if plane.color == 1 {
						cells /= 2
					}
				}
			}
			perMacro += cells
		}
	}
	overhead := uint64(4096)
	if totalChannels > 17 {
		overhead += totalChannels * 128
	}
	if work > ^uint64(0)-overhead {
		return ^uint64(0)
	}
	width := uint64(source.header.columns[len(source.header.columns)-1])
	if perMacro == 0 || width > (^uint64(0)-work-overhead)/16/perMacro {
		return ^uint64(0)
	}
	return work + overhead + width*perMacro*16
}

// newJXRReducedRows 预检总额度后建立必要频带的重建游标及两行缓存
// 入参: ctx Context, data 可选原始流切片, source 完整流, divisor 缩减分母, maximum 总阶段额度
// 返回: *jxrReducedRows 固定邻行缓存, error 资源、结构、能力或取消错误
func newJXRReducedRows(ctx context.Context, data []byte, source *jxrStream, divisor int, maximum uint64) (*jxrReducedRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	work := jxrReducedRowWork(source, divisor)
	if work > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: reduced representative row budget", ErrLimit)
	}
	lowWork := jxrLowGridRowWork(source, divisor)
	if divisor == 2 {
		return newJXRHalfReducedRows(ctx, data, source, maximum)
	}
	low, err := newJXRLowGridRows(ctx, data, source, divisor, lowWork)
	if err != nil {
		return nil, err
	}
	totalChannels := source.primary.channels
	if source.alpha != nil {
		totalChannels += source.alpha.channels
	}
	r := &jxrReducedRows{low: low, height: int(source.header.rows[len(source.header.rows)-1]), geometry: make([]jxrLowGrid, totalChannels), offsets: make([]int, totalChannels)}
	coeff := &low.inverse.rows.image
	samples := 0
	for _, c := range [2]*jxrCoefficients{&coeff.primary, coeff.alpha} {
		if c == nil {
			continue
		}
		if c.plane.channels > len(r.geometry)-r.count {
			return nil, fmt.Errorf("%w: reduced representative component capacity", ErrLimit)
		}
		views, err := c.lowGridViews(&coeff.header, divisor, r.geometry[r.count:r.count])
		if err != nil {
			return nil, err
		}
		for _, v := range views {
			r.offsets[r.count] = samples
			samples += v.width * v.height
			r.count++
		}
	}
	overhead := uint64(4096)
	if totalChannels > 17 {
		overhead += uint64(totalChannels) * 128
	}
	if uint64(samples)*16+overhead > work-lowWork {
		return nil, fmt.Errorf("%w: reduced representative storage accounting", ErrInvalid)
	}
	for i := range r.cache {
		r.cache[i] = make([]int64, samples)
	}
	return r, ctx.Err()
}

// ensure 单向推进到所需宏块行并复制已归一化值，不在移动低频窗口后继续引用旧系数
// 入参: ctx Context, row 包含在图内的目标宏块行
// 返回: error 行序、熵、数值或取消错误
func (r *jxrReducedRows) ensure(ctx context.Context, row int) error {
	if row < 0 || row >= r.height || row < r.next-2 {
		return fmt.Errorf("%w: reduced representative row sequence", ErrInvalid)
	}
	for r.next <= row {
		if r.half != nil {
			if err := r.readHalf(ctx); err != nil {
				return err
			}
			continue
		}
		primary, alpha, index, err := r.low.next(ctx)
		if err != nil {
			return err
		}
		if index != r.next {
			return fmt.Errorf("%w: reduced representative source sequence", ErrInvalid)
		}
		component := 0
		for _, views := range [2][]jxrLowGrid{primary, alpha} {
			for _, v := range views {
				if err := ctx.Err(); err != nil {
					return err
				}
				at := r.offsets[component]
				for y := range v.height {
					for x := range v.width {
						if x&1023 == 0 {
							if err := ctx.Err(); err != nil {
								return err
							}
						}
						value, err := v.value(x, y)
						if err != nil {
							return err
						}
						r.cache[index%2][at+y*v.width+x] = value
					}
				}
				component++
			}
		}
		r.next++
	}
	return ctx.Err()
}

// position 根据原编码支持中心选择最近粗样点，平局选较大坐标，图外采用端点延拓
// 入参: component 实际内部颜色或透明度分量, x, y 原编码坐标
// 返回: int 粗列坐标, int 粗行坐标, error 分量序号错误
func (r *jxrReducedRows) position(component int, x, y float64) (int, int, error) {
	if component < 0 || component >= r.count {
		return 0, 0, fmt.Errorf("%w: reduced representative component", ErrInvalid)
	}
	v := &r.geometry[component]
	px := int(math.Floor((x-v.xCenter)/float64(v.xStep) + 0.5))
	py := int(math.Floor((y-v.yCenter)/float64(v.yStep) + 0.5))
	return min(v.width-1, max(0, px)), min(r.height*v.height-1, max(0, py)), nil
}

// at 从已完成的两条粗行取得值，不允许访问已覆盖或尚未读取的行
// 入参: component 内部分量, x, y 粗样点坐标
// 返回: int64 归一化代表值, error 存储或坐标错误
func (r *jxrReducedRows) at(component, x, y int) (int64, error) {
	if component < 0 || component >= r.count {
		return 0, fmt.Errorf("%w: reduced representative component", ErrInvalid)
	}
	v := &r.geometry[component]
	row := y / v.height
	if x < 0 || x >= v.width || y < 0 || row >= r.next || row < r.next-2 {
		return 0, fmt.Errorf("%w: reduced representative cache coordinate", ErrInvalid)
	}
	return r.cache[row%2][r.offsets[component]+y%v.height*v.width+x], nil
}

// newJXRLowFrequencyPlan 在既有频率索引上建立选择性读取计划，跳过未选频带
// 入参: ctx Context, source 完整原始频率流, bands 一或二，分别选择DC及DC加LP, maximum 排序目录及源头额度
// 返回: *jxrLowFrequencyPlan 独立操作计划, error 参数、索引、能力、资源或取消错误
func newJXRLowFrequencyPlan(ctx context.Context, source *jxrStream, bands int, maximum uint64) (*jxrLowFrequencyPlan, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil || !source.header.frequency || source.primary.bands > 3 || bands < 1 || bands > 2 || len(source.header.columns) < 2 || len(source.header.rows) < 2 {
		return nil, fmt.Errorf("%w: low frequency operation plan", ErrInvalid)
	}
	count := 4 - int(source.primary.bands)
	tiles := uint64(len(source.header.columns)-1) * uint64(len(source.header.rows)-1)
	if tiles*uint64(count) != uint64(len(source.packets)) || len(source.packets) == 0 {
		return nil, fmt.Errorf("%w: low frequency original packet count", ErrInvalid)
	}
	if len(source.packets) > 1 && !source.header.index {
		return nil, fmt.Errorf("%w: low frequency indexed access", ErrUnsupported)
	}
	work := source.headerWork()
	if work > maximum || uint64(len(source.packets)) > (maximum-work)/8 || 256 > maximum-work-uint64(len(source.packets))*8 {
		return nil, fmt.Errorf("%w: low frequency packet directory budget", ErrLimit)
	}
	physical := make([]uint64, 0, len(source.packets))
	for index, offset := range source.packets {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if offset != jxrPacketEscape {
			physical = append(physical, offset)
		}
	}
	slices.Sort(physical)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &jxrLowFrequencyPlan{source: source, physical: physical, count: count, selected: min(bands, count)}, nil
}

// packet 为选中包建立严格物理上界的独立窗口，索引可非单调，预取也不能越过下一物理包
// 入参: ctx Context, data 可选原始切片, index 原逻辑包序号
// 返回: jxrBits 包内读取器, error 索引、范围或取消错误
func (p *jxrLowFrequencyPlan) packet(ctx context.Context, data []byte, index int) (jxrBits, error) {
	if err := ctx.Err(); err != nil {
		return jxrBits{}, err
	}
	s := p.source
	if index < 0 || index >= len(s.packets) || s.packets[index] == jxrPacketEscape {
		return jxrBits{}, fmt.Errorf("%w: low frequency packet index", ErrInvalid)
	}
	length := uint64(len(data))
	if s.input != nil {
		length = s.input.size
	}
	if s.dataStart > length || s.packets[index] > length-s.dataStart {
		return jxrBits{}, fmt.Errorf("%w: low frequency packet offset", ErrInvalid)
	}
	start, end := s.dataStart+s.packets[index], length
	if next := sort.Search(len(p.physical), func(at int) bool { return p.physical[at] > s.packets[index] }); next < len(p.physical) {
		if p.physical[next] > length-s.dataStart {
			return jxrBits{}, fmt.Errorf("%w: low frequency next packet offset", ErrInvalid)
		}
		end = s.dataStart + p.physical[next]
	}
	if end-start < 4 {
		return jxrBits{}, fmt.Errorf("%w: low frequency packet extent", ErrInvalid)
	}
	if s.input == nil {
		return jxrBits{data: data[start:end]}, nil
	}
	if s.input.base > math.MaxInt64 || start > math.MaxInt64-s.input.base {
		return jxrBits{}, fmt.Errorf("%w: low frequency input address", ErrInvalid)
	}
	return jxrBits{input: &jxrReaderAt{source: s.input.source, ctx: ctx, base: s.input.base + start, size: end - start}}, nil
}

// rowWork 计算一次低频行读取的系数、预测、量化和独立包窗，源及排序目录由调用方另计
// 入参: tile 原始分块编号
// 返回: uint64 行状态与目标合计额度，非法几何返回最大值
func (p *jxrLowFrequencyPlan) rowWork(tile int) uint64 {
	s := p.source
	columns := len(s.header.columns) - 1
	if tile < 0 || tile >= columns*(len(s.header.rows)-1) {
		return ^uint64(0)
	}
	x := tile % columns
	width := uint64(s.header.columns[x+1] - s.header.columns[x])
	if width == 0 {
		return ^uint64(0)
	}
	work := uint64(32768)
	for _, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
		if plane == nil {
			continue
		}
		if plane.channels < 1 || plane.channels > rasterMaxComponents || plane.bands > 3 {
			return ^uint64(0)
		}
		stride := uint64(plane.channels)*128 + 6
		if width > (^uint64(0)-work)/stride {
			return ^uint64(0)
		}
		work += width * stride
		for _, amount := range []uint64{plane.quantWork(), uint64(plane.channels)*170 + 256, jxrPredictionRowWork(width, plane.channels) * uint64(min(p.selected, 4-int(plane.bands)))} {
			if amount > ^uint64(0)-work {
				return ^uint64(0)
			}
			work += amount
		}
	}
	if s.input != nil {
		windows := uint64(p.selected) * jxrReaderAtWork
		if windows > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += windows
	}
	return work
}

// newJXRLowFrequencyRows 在总额度内创建当前块的低频行状态，原始每块包数始终用于索引
// 入参: ctx Context, data 可选切片, plan 低频操作计划, tile 原块序号, maximum 行状态及目标额度
// 返回: *jxrLowFrequencyRows 连续读取器, error 参数、包、模型、资源或取消错误
func newJXRLowFrequencyRows(ctx context.Context, data []byte, plan *jxrLowFrequencyPlan, tile int, maximum uint64) (*jxrLowFrequencyRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if plan.rowWork(tile) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: low frequency row working budget", ErrLimit)
	}
	s := plan.source
	columns := len(s.header.columns) - 1
	x, y := tile%columns, tile/columns
	r := &jxrLowFrequencyRows{width: int(s.header.columns[x+1] - s.header.columns[x]), height: int(s.header.rows[y+1] - s.header.rows[y]), bands: plan.selected}
	r.image.header = s.header
	for index, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
		if plane == nil {
			continue
		}
		if err := checkJXRModelComponents(plane); err != nil {
			return nil, err
		}
		current := jxrCoefficients{plane: *plane, longWord: s.header.longWord, low: make([]int64, r.width*plane.channels*16), info: make([]jxrMacroInfo, r.width)}
		if index == 0 {
			r.image.primary = current
		} else {
			r.image.alpha = &current
		}
		r.quant[index] = newJXRTileQuant(plane)
		var err error
		r.scratch[index], err = newJXRDecodedMacro(ctx, plane.channels, maximum)
		if err != nil {
			return nil, err
		}
		for band := range min(r.bands, 4-int(plane.bands)) {
			r.rows[band][index], err = newJXRTileRow(ctx, r.width, plane.channels, maximum)
			if err != nil {
				return nil, err
			}
		}
		r.dc[index], r.lp[index] = newJXRDC(), newJXRLP()
	}
	for band := range r.bands {
		var err error
		r.bits[band], err = plan.packet(ctx, data, tile*plan.count+band)
		if err != nil {
			return nil, err
		}
		if _, err := readJXRPacket(&r.bits[band], s, band, &r.quant); err != nil {
			return nil, err
		}
	}
	return r, ctx.Err()
}

// readRow 读取一行选中频带并恢复真实预测及反量化，目标只含低通，未选频带不被解释或验证
// 入参: ctx Context
// 返回: error 包、数值、序列或取消错误，结果位于读取器的当前image行
func (r *jxrLowFrequencyRows) readRow(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.next >= r.height {
		return fmt.Errorf("%w: low frequency row sequence", ErrInvalid)
	}
	planes := [2]*jxrCoefficients{&r.image.primary, r.image.alpha}
	for _, plane := range planes {
		if plane != nil {
			clear(plane.low)
			clear(plane.info)
		}
	}
	for band := range r.bands {
		b := &r.bits[band]
		for _, row := range r.rows[band] {
			if row != nil {
				row.startRow()
			}
		}
		for x := range r.width {
			if err := ctx.Err(); err != nil {
				return err
			}
			for index, plane := range planes {
				if plane == nil || band >= 4-int(plane.plane.bands) {
					continue
				}
				row, q := r.rows[band][index], &r.quant[index]
				m := jxrMacroContext{at: x, firstColumn: x == 0, firstRow: r.next == 0, reset: x%16 == 0, adapt: x == r.width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
				current := &row.current
				var err error
				if band == 0 {
					err = plane.readDC(b, &r.dc[index], q, m, &current.low, r.scratch[index])
				} else {
					if q.lpCount > 1 && !q.useDC {
						plane.info[x].lpIndex = uint8(readJXRQPIndex(b, q.lpCount))
					}
					err = plane.readLP(b, &r.lp[index], q, m, &current.low, r.scratch[index])
				}
				if err != nil {
					return err
				}
				if err := row.advance(x, current); err != nil {
					return err
				}
			}
		}
	}
	if r.bands == 1 {
		for index, plane := range planes {
			if plane == nil || plane.plane.bands == 3 {
				continue
			}
			for x := range r.width {
				for channel := range plane.plane.channels {
					shift := uint8(0)
					if channel == 0 {
						shift = 1
					}
					at := (x*plane.plane.channels + channel) * 16
					value, err := scaleJXRWord(plane.low[at], jxrQuantScale(r.quant[index].dc[channel], plane.plane.scaled, shift), plane.longWord)
					if err != nil {
						return err
					}
					plane.low[at] = value
				}
			}
		}
	}
	r.next++
	if r.next == r.height {
		for band := range r.bands {
			r.bits[band].align(true)
			if r.bits[band].err != nil {
				return r.bits[band].err
			}
		}
	}
	return ctx.Err()
}

// lowGridViews 建立四、八或十六倍粗网格的借用视图，DC输入必须未经首级逆变换，LP输入必须已完成首级阶段
// 入参: h 当前低频缓冲的编码几何, divisor 四或八选择LP代表值，十六选择DC代表值, storage 可复用描述容量
// 返回: []jxrLowGrid 逐原生分量网格及中心, error 参数、几何或存储错误
func (c *jxrCoefficients) lowGridViews(h *jxrHeader, divisor int, storage []jxrLowGrid) ([]jxrLowGrid, error) {
	if c == nil || h == nil || divisor != 4 && divisor != 8 && divisor != 16 || len(h.columns) < 2 || len(h.rows) < 2 {
		return nil, fmt.Errorf("%w: reduced grid parameters", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(&c.plane); err != nil {
		return nil, err
	}
	columns, rows := uint64(h.columns[len(h.columns)-1]), uint64(h.rows[len(h.rows)-1])
	stride := uint64(c.plane.channels) * 16
	if columns == 0 || rows == 0 || columns > uint64(^uint(0)>>1)/4 || rows > uint64(^uint(0)>>1)/4 || columns > uint64(len(c.low))/stride/rows {
		return nil, fmt.Errorf("%w: reduced grid storage", ErrInvalid)
	}
	views := storage
	if cap(views) < c.plane.channels {
		views = make([]jxrLowGrid, c.plane.channels)
	} else {
		views = views[:c.plane.channels]
	}
	for i := range views {
		sx, sy, bw, bh := 1, 1, 4, 4
		phaseX, phaseY := 0.0, 0.0
		if i > 0 && (c.plane.color == 1 || c.plane.color == 2) {
			chromaX, chromaY := c.plane.chromaPhase(i)
			sx, bw, phaseX = 2, 2, 0.5
			if chromaX <= 4 {
				phaseX = float64(chromaX) / 4
			}
			if c.plane.color == 1 {
				sy, bh, phaseY = 2, 2, 0.5
				if chromaY <= 4 {
					phaseY = float64(chromaY) / 4
				}
			}
		}
		v := jxrLowGrid{data: c.low, columns: int(columns), blockWidth: bw, stride: int(stride), offset: i * 16, gain: 1, longWord: h.longWord}
		if divisor == 16 {
			v.dc, v.group, v.blockRows, v.shift = true, 1, 1, 4
			v.width, v.height = int(columns), int(rows)
			v.xStep, v.yStep = 16, 16
			v.xCenter, v.yCenter = float64(16-sx)/2+phaseX, float64(16-sy)/2+phaseY
			if sx != 1 {
				v.shift = 3
			}
			if i > 0 && c.plane.scaled {
				v.gain = 2
			}
		} else {
			v.group, v.shift = divisor/4, 2
			if v.group == 2 {
				v.shift += 2
			}
			v.blockRows = bh / v.group
			v.width, v.height = int(columns)*(bw/v.group), int(rows)*v.blockRows
			v.xStep, v.yStep = divisor*sx, divisor*sy
			v.xCenter, v.yCenter = float64((divisor-1)*sx)/2+phaseX, float64((divisor-1)*sy)/2+phaseY
		}
		views[i] = v
	}
	return views, nil
}

// value 返回变换域归一化后的粗代表值，不进行颜色转换、码值映射或边缘补点
// 入参: x, y 当前借用网格的有效坐标
// 返回: int64 带符号代表值, error 坐标或系数范围错误
func (v *jxrLowGrid) value(x, y int) (int64, error) {
	if x < 0 || y < 0 || x >= v.width || y >= v.height {
		return 0, fmt.Errorf("%w: reduced grid coordinate", ErrInvalid)
	}
	perWidth := v.width / v.columns
	at := ((y/v.blockRows)*v.columns+x/perWidth)*v.stride + v.offset
	sum := int64(0)
	if v.dc {
		sum = v.data[at]
		if err := checkJXRWords([]int64{sum}, v.longWord); err != nil {
			return 0, err
		}
		sum *= v.gain
	} else {
		at += (y%v.blockRows*v.group)*v.blockWidth + x%perWidth*v.group
		for dy := range v.group {
			for dx := range v.group {
				value := v.data[at+dy*v.blockWidth+dx]
				if err := checkJXRWords([]int64{value}, v.longWord); err != nil {
					return 0, err
				}
				sum += value
			}
		}
	}
	return (sum + 1<<(v.shift-1)) >> v.shift, nil
}

// jxrLowGridRowWork 计入低频游标及固定粗网格描述，不计调用方持有的原始输入或最终图像
// 入参: source 完整原始流, divisor 四、八或十六
// 返回: uint64 活动工作额度，参数无效或溢出返回最大值
func jxrLowGridRowWork(source *jxrStream, divisor int) uint64 {
	if source == nil || divisor != 4 && divisor != 8 && divisor != 16 {
		return ^uint64(0)
	}
	bands := 2
	if divisor == 16 {
		bands = 1
	}
	overhead := uint64(4096)
	if source.primary.channels > 16 {
		overhead += uint64(source.primary.channels) * 128
	}
	work := jxrLowInverseRowWork(source, bands)
	if work > ^uint64(0)-overhead {
		return ^uint64(0)
	}
	return work + overhead
}

// newJXRLowGridRows 在任何输入读取前预留全部阶段工作额度，粗网格中心仍采用原编码坐标
// 入参: ctx Context, data 可选原始流切片, source 完整流, divisor 四、八或十六, maximum 总阶段额度
// 返回: *jxrLowGridRows 可重试建立的独立游标, error 参数、资源、能力或取消错误
func newJXRLowGridRows(ctx context.Context, data []byte, source *jxrStream, divisor int, maximum uint64) (*jxrLowGridRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if divisor != 4 && divisor != 8 && divisor != 16 {
		return nil, fmt.Errorf("%w: reduced row divisor", ErrInvalid)
	}
	if source == nil {
		return nil, fmt.Errorf("%w: reduced row source", ErrInvalid)
	}
	if source.primary.channels < 1 || source.primary.channels > rasterMaxComponents || source.alpha != nil && source.alpha.channels > 1 {
		return nil, fmt.Errorf("%w: reduced row component capacity", ErrLimit)
	}
	if jxrLowGridRowWork(source, divisor) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: reduced grid row budget", ErrLimit)
	}
	bands := 2
	if divisor == 16 {
		bands = 1
	}
	overhead := uint64(4096)
	if source.primary.channels > 16 {
		overhead += uint64(source.primary.channels) * 128
	}
	inverse, err := newJXRLowInverseRows(ctx, data, source, bands, maximum-overhead)
	if err != nil {
		return nil, err
	}
	return &jxrLowGridRows{inverse: inverse, primary: make([]jxrLowGrid, source.primary.channels), divisor: divisor}, ctx.Err()
}

// next 交出一个宏块行内的粗网格借用视图，下一次调用会覆盖系数及描述，透明度与主平面同步
// 入参: ctx Context
// 返回: []jxrLowGrid 主分量, []jxrLowGrid 交织透明度或nil, int 全图宏块行号, error 结束或读取错误
func (r *jxrLowGridRows) next(ctx context.Context) ([]jxrLowGrid, []jxrLowGrid, int, error) {
	row, index, err := r.inverse.next(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	h := row.header
	rows := [2]uint32{0, 1}
	h.rows = rows[:]
	var output [2][]jxrLowGrid
	for plane, coefficients := range [2]*jxrCoefficients{&row.primary, row.alpha} {
		if coefficients == nil {
			continue
		}
		c := *coefficients
		stride := r.inverse.width * c.plane.channels * 16
		first := int(row.header.top/16) * stride
		c.low = c.low[first : first+stride]
		storage := r.primary[:0]
		if plane == 1 {
			storage = r.alpha[:0]
		}
		views, err := c.lowGridViews(&h, r.divisor, storage)
		if err != nil {
			return nil, nil, 0, err
		}
		for i := range views {
			views[i].yCenter += float64(index * 16)
		}
		output[plane] = views
	}
	return output[0], output[1], index, ctx.Err()
}

// newJXRHalfReducedRows 在完整频带的固定重建邻行之外只分配两条半速代表行，不分配整幅原尺寸样本
// 入参: ctx Context, data 可选码流切片, source 完整码流, maximum 总行工作额度
// 返回: *jxrReducedRows 半速游标, error 结构、能力、资源或取消错误
func newJXRHalfReducedRows(ctx context.Context, data []byte, source *jxrStream, maximum uint64) (*jxrReducedRows, error) {
	work := jxrReducedRowWork(source, 2)
	if work > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: half reduced row budget", ErrLimit)
	}
	base := jxrStreamingRowWork(source)
	reader, err := newJXRStreamRows(ctx, data, source, base)
	if err != nil {
		return nil, err
	}
	rows, err := newJXRReconstructedRows(ctx, source, reader, base)
	if err != nil {
		return nil, err
	}
	totalChannels := source.primary.channels
	if source.alpha != nil {
		totalChannels += source.alpha.channels
	}
	r := &jxrReducedRows{half: rows, height: rows.height, geometry: make([]jxrLowGrid, totalChannels), offsets: make([]int, totalChannels)}
	columns := int(source.header.columns[len(source.header.columns)-1])
	samples := 0
	for _, p := range [2]*jxrPlane{&source.primary, source.alpha} {
		if p == nil {
			continue
		}
		if p.channels < 1 || p.channels > len(r.geometry)-r.count {
			return nil, fmt.Errorf("%w: half reduced component capacity", ErrLimit)
		}
		for c := range p.channels {
			sx, sy := 1, 1
			phaseX, phaseY := 0.0, 0.0
			if c > 0 && (p.color == 1 || p.color == 2) {
				chromaX, chromaY := p.chromaPhase(c)
				sx, phaseX = 2, 0.5
				if chromaX <= 4 {
					phaseX = float64(chromaX) / 4
				}
				if p.color == 1 {
					sy, phaseY = 2, 0.5
					if chromaY <= 4 {
						phaseY = float64(chromaY) / 4
					}
				}
			}
			v := jxrLowGrid{width: columns * 8 / sx, height: 8 / sy, xStep: 2 * sx, yStep: 2 * sy, xCenter: float64(sx)/2 + phaseX, yCenter: float64(sy)/2 + phaseY}
			r.geometry[r.count], r.offsets[r.count] = v, samples
			samples += v.width * v.height
			r.count++
		}
	}
	overhead := uint64(4096)
	if totalChannels > 17 {
		overhead += uint64(totalChannels) * 128
	}
	if uint64(samples)*16+overhead > work-base {
		return nil, fmt.Errorf("%w: half reduced storage accounting", ErrInvalid)
	}
	for i := range r.cache {
		r.cache[i] = make([]int64, samples)
	}
	return r, ctx.Err()
}

// readHalf 将一条完成所有实际逆变换和重叠的原生行直接合并到粗缓存，之后不保留原行视图
// 入参: ctx Context
// 返回: error 行序、熵、数值或取消错误
func (r *jxrReducedRows) readHalf(ctx context.Context) error {
	coeff, index, err := r.half.next(ctx)
	if err != nil {
		return err
	}
	if index != r.next {
		return fmt.Errorf("%w: half reduced source sequence", ErrInvalid)
	}
	var storage [17]jxrComponentView
	component := 0
	for _, plane := range [2]*jxrCoefficients{&coeff.primary, coeff.alpha} {
		if plane == nil {
			continue
		}
		views := plane.componentViewsStorage(&coeff.header, true, storage[:])
		for _, v := range views {
			geometry := &r.geometry[component]
			first := int(coeff.header.top) / (geometry.yStep / 2)
			at := r.offsets[component]
			for y := range geometry.height {
				if err := ctx.Err(); err != nil {
					return err
				}
				for x := range geometry.width {
					if x&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					px, py := x*2, first+y*2
					sum := v.at(px, py) + v.at(px+1, py) + v.at(px, py+1) + v.at(px+1, py+1)
					r.cache[index%2][at+y*geometry.width+x] = (sum + 2) >> 2
				}
			}
			component++
		}
	}
	r.next++
	return ctx.Err()
}

// jxrLowInverseRowWork 计算跨分块低带读取及六行首级邻域，DC模式不分配逆变换环
// 入参: source 完整原始流, bands 一或二
// 返回: uint64 活动工作额度，非法几何或溢出返回最大值
func jxrLowInverseRowWork(source *jxrStream, bands int) uint64 {
	work := jxrLowTileRowsWork(source, bands)
	if work == ^uint64(0) || bands == 1 {
		return work
	}
	channels, planes := uint64(source.primary.channels), uint64(1)
	if source.alpha != nil {
		channels += uint64(source.alpha.channels)
		planes++
	}
	width := uint64(source.header.columns[len(source.header.columns)-1])
	stride := channels*128 + planes*6
	if 2048 > ^uint64(0)-work || width > (^uint64(0)-work-2048)/6/stride {
		return ^uint64(0)
	}
	return work + 2048 + width*stride*6
}

// newJXRLowInverseRows 在读取前建立固定低频窗口，DC输出仍为反量化系数，LP输出为首级逆变换后的粗网格
// 入参: ctx Context, data 可选原流切片, source 完整原始流, bands 一或二, maximum 读取和重建总额度
// 返回: *jxrLowInverseRows 低频阶段游标, error 参数、资源、模型、能力或取消错误
func newJXRLowInverseRows(ctx context.Context, data []byte, source *jxrStream, bands int, maximum uint64) (*jxrLowInverseRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrLowInverseRowWork(source, bands) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: reduced low inverse row budget", ErrLimit)
	}
	rows, err := newJXRLowTileRows(ctx, data, source, bands, maximum)
	if err != nil {
		return nil, err
	}
	frame := jxrRowReconstructionStream(source)
	r := &jxrLowInverseRows{rows: rows, width: int(frame.header.columns[1]), height: int(frame.header.rows[1]), dcOnly: bands == 1}
	if r.dcOnly {
		return r, nil
	}
	r.pipeline.header, r.pipeline.width, r.pipeline.height = source.header, r.width, r.height
	h := frame.header
	h.height, h.top, h.bottom, h.rows = 48, 0, 0, []uint32{0, 3}
	for _, buffer := range []*jxrCoefficientImage{&r.pipeline.raw, &r.pipeline.work} {
		buffer.header = h
		for index, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
			if plane == nil {
				continue
			}
			current := jxrCoefficients{plane: *plane, longWord: h.longWord, low: make([]int64, r.width*3*plane.channels*16), info: make([]jxrMacroInfo, r.width*3)}
			if index == 0 {
				buffer.primary = current
			} else {
				buffer.alpha = &current
			}
		}
	}
	return r, ctx.Err()
}

// next 逐行交出未格式化粗系数，保留真实软硬边界，仅在首级重叠模式需要时读取下一邻行
// 入参: ctx Context
// 返回: *jxrCoefficientImage 借用低通窗口，其top定位目标行, int 全图宏块行号, error 结束时io.EOF
func (r *jxrLowInverseRows) next(ctx context.Context) (*jxrCoefficientImage, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if r.nextRow >= r.height {
		return nil, 0, io.EOF
	}
	row := r.nextRow
	if r.dcOnly {
		if err := r.rows.readRow(ctx); err != nil {
			return nil, 0, err
		}
		r.nextRow++
		return &r.rows.image, row, ctx.Err()
	}
	p := &r.pipeline
	needed := row + 1
	if p.header.overlap == 2 {
		needed++
	}
	for r.read < min(r.height, needed) {
		if err := r.rows.readRow(ctx); err != nil {
			return nil, 0, err
		}
		for index, target := range [2]*jxrCoefficients{&p.raw.primary, p.raw.alpha} {
			if target == nil {
				continue
			}
			from := [2]*jxrCoefficients{&r.rows.image.primary, r.rows.image.alpha}[index]
			first := r.read % 3 * r.width
			stride := target.plane.channels * 16
			copy(target.low[first*stride:(first+r.width)*stride], from.low)
			copy(target.info[first:first+r.width], from.info)
			view := *target
			view.low, view.info = target.low[first*stride:(first+r.width)*stride], target.info[first:first+r.width]
			if err := view.inverseFirstRange(ctx, &p.header, 0, len(view.info)); err != nil {
				return nil, 0, err
			}
		}
		r.read++
	}
	core, edges, err := p.prepareWindow(ctx, row, false)
	if err != nil {
		return nil, 0, err
	}
	if p.header.overlap == 2 {
		for _, plane := range [2]*jxrCoefficients{&p.work.primary, p.work.alpha} {
			if plane != nil {
				if err := plane.filterWindow(ctx, &p.work.header, false, core, edges, false); err != nil {
					return nil, 0, err
				}
			}
		}
	}
	p.work.header.top = uint32(core.Min.Y * 16)
	r.nextRow++
	return &p.work, row, ctx.Err()
}

// jxrLowSpatialRowWork 以既有完整行预算减去整行高通存储并计入独立宏块工作，不将消费语法误报为跳过输入
// 入参: s 原始空间流, tile 原分块编号
// 返回: uint64 目标低通行与全部熵工作额度，非法布局或溢出返回最大值
func jxrLowSpatialRowWork(s *jxrStream, tile int) uint64 {
	work := jxrSpatialRowWork(s, tile)
	if work == ^uint64(0) {
		return work
	}
	channels, planes := s.primary.channels, uint64(1)
	if s.alpha != nil {
		channels += s.alpha.channels
		planes++
	}
	if channels < 1 || channels > rasterMaxComponents+1 {
		return ^uint64(0)
	}
	x := tile % (len(s.header.columns) - 1)
	width := uint64(s.header.columns[x+1] - s.header.columns[x])
	removed := width * uint64(channels) * 256 * 8
	if removed > work {
		return ^uint64(0)
	}
	work -= removed
	extra := uint64(channels)*272*8 + planes*6 + 1024
	if extra > ^uint64(0)-work {
		return ^uint64(0)
	}
	return work + extra
}

// newJXRLowSpatialRows 建立空间交错低带读取器，保留真实主alpha频带声明与单独预测状态
// 入参: ctx Context, data 原始切片, s 原始空间流, tile 原块序号, bands 一或二，选择DC或DC加LP, maximum 合计工作额度
// 返回: *jxrLowSpatialRows 连续行读取器, error 参数、模型、容量、包或取消错误
func newJXRLowSpatialRows(ctx context.Context, data []byte, s *jxrStream, tile, bands int, maximum uint64) (*jxrLowSpatialRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if bands < 1 || bands > 2 {
		return nil, fmt.Errorf("%w: spatial low frequency selection", ErrInvalid)
	}
	if jxrLowSpatialRowWork(s, tile) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: spatial low frequency row budget", ErrLimit)
	}
	if tile >= len(s.packets) {
		return nil, fmt.Errorf("%w: spatial low frequency packet", ErrInvalid)
	}
	columns := len(s.header.columns) - 1
	x, y := tile%columns, tile/columns
	r := &jxrLowSpatialRows{bits: jxrPacketReader(data, s, tile), width: int(s.header.columns[x+1] - s.header.columns[x]), height: int(s.header.rows[y+1] - s.header.rows[y]), bands: bands}
	r.image.header, r.macro.header = s.header, s.header
	if s.input != nil {
		r.bits.input = &jxrReaderAt{source: s.input.source, ctx: ctx, base: s.input.base, size: s.input.size}
	}
	for index, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
		if plane == nil {
			continue
		}
		if err := checkJXRModelComponents(plane); err != nil {
			return nil, err
		}
		current := jxrCoefficients{plane: *plane, longWord: s.header.longWord, low: make([]int64, r.width*plane.channels*16), info: make([]jxrMacroInfo, r.width)}
		macro := newJXRCoefficients(plane, 1, s.header.longWord)
		if index == 0 {
			r.image.primary, r.macro.primary = current, macro
		} else {
			r.image.alpha, r.macro.alpha = &current, &macro
		}
		r.quant[index] = newJXRTileQuant(plane)
		var err error
		r.rows[index], err = newJXRTileRow(ctx, r.width, plane.channels, maximum)
		if err != nil {
			return nil, err
		}
		r.scratch[index], err = newJXRDecodedMacro(ctx, plane.channels, maximum)
		if err != nil {
			return nil, err
		}
		r.dc[index], r.lp[index], r.hp[index] = newJXRDC(), newJXRLP(), newJXRHP()
	}
	packet, err := readJXRPacket(&r.bits, s, -1, &r.quant)
	if err != nil {
		return nil, err
	}
	r.trim = packet.trim
	return r, ctx.Err()
}

// readRow 推进完整空间行语法并丢弃高通结果，不执行高通反量化或空间逆变换，结果仍非最终缩减像素
// 入参: ctx Context
// 返回: error 熵、低带数值、序列或取消错误
func (r *jxrLowSpatialRows) readRow(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.next >= r.height {
		return fmt.Errorf("%w: spatial low frequency row sequence", ErrInvalid)
	}
	macro := [2]*jxrCoefficients{&r.macro.primary, r.macro.alpha}
	target := [2]*jxrCoefficients{&r.image.primary, r.image.alpha}
	for index, row := range r.rows {
		if row != nil {
			row.startRow()
			clear(target[index].low)
			clear(target[index].info)
		}
	}
	for x := range r.width {
		if err := ctx.Err(); err != nil {
			return err
		}
		for index, plane := range macro {
			if plane == nil {
				continue
			}
			clear(plane.low)
			clear(plane.high)
			plane.info[0] = jxrMacroInfo{}
			row, q := r.rows[index], &r.quant[index]
			m := jxrMacroContext{firstColumn: x == 0, firstRow: r.next == 0, reset: x%16 == 0, adapt: x == r.width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
			info := &plane.info[0]
			if plane.plane.bands != 3 {
				if q.lpCount > 1 && !q.useDC {
					info.lpIndex = uint8(readJXRQPIndex(&r.bits, q.lpCount))
				}
				if plane.plane.bands < 2 {
					if q.useLP {
						info.hpIndex = info.lpIndex
					} else if q.hpCount > 1 {
						info.hpIndex = uint8(readJXRQPIndex(&r.bits, q.hpCount))
					}
				}
			}
			current := &row.current
			if err := plane.readDC(&r.bits, &r.dc[index], q, m, &current.low, r.scratch[index]); err != nil {
				return err
			}
			if plane.plane.bands != 3 {
				if err := plane.readLP(&r.bits, &r.lp[index], q, m, &current.low, r.scratch[index]); err != nil {
					return err
				}
			}
			if plane.plane.bands < 2 {
				if err := plane.readHP(&r.bits, &r.hp[index], m, true, r.trim, row.residual, current.pattern); err != nil {
					return err
				}
			}
			if err := row.advance(x, current); err != nil {
				return err
			}
			output := target[index].lowMacro(x)
			if r.bands == 2 {
				copy(output, plane.low)
			} else {
				for channel := range plane.plane.channels {
					output[channel*16] = plane.low[channel*16]
				}
			}
			target[index].info[x] = *info
		}
	}
	r.next++
	if r.next == r.height {
		r.bits.align(true)
	}
	return r.bits.err
}

// jxrLowTileRowsWork 计算当前块行各列状态、连续低通目标及频率物理目录，不随已完成块行累积
// 入参: s 完整原始流, bands 选择一或两个低频带
// 返回: uint64 合计活动额度，无效几何或溢出返回最大值
func jxrLowTileRowsWork(s *jxrStream, bands int) uint64 {
	columns, rows := len(s.header.columns)-1, len(s.header.rows)-1
	if columns < 1 || columns > 4096 || rows < 1 || rows > 4096 || bands < 1 || bands > 2 || s.primary.bands > 3 {
		return ^uint64(0)
	}
	channels, planes := uint64(s.primary.channels), uint64(1)
	if s.alpha != nil {
		channels += uint64(s.alpha.channels)
		planes++
	}
	if channels == 0 || channels > rasterMaxComponents+1 {
		return ^uint64(0)
	}
	width := uint64(s.header.columns[columns])
	stride := channels*128 + planes*6
	if width > (^uint64(0)-1024-uint64(columns)*16)/stride {
		return ^uint64(0)
	}
	work := uint64(1024) + uint64(columns)*16 + width*stride
	plan := jxrLowFrequencyPlan{source: s, selected: min(bands, 4-int(s.primary.bands))}
	if s.header.frequency {
		amount := s.headerWork()
		if amount > ^uint64(0)-work || 256 > ^uint64(0)-work-amount || uint64(len(s.packets)) > (^uint64(0)-work-amount-256)/8 {
			return ^uint64(0)
		}
		work += amount + uint64(len(s.packets))*8 + 256
	}
	for column := range columns {
		amount := jxrLowSpatialRowWork(s, column)
		if s.header.frequency {
			amount = plan.rowWork(column)
		} else if s.input != nil {
			if amount > ^uint64(0)-jxrReaderAtWork {
				return ^uint64(0)
			}
			amount += jxrReaderAtWork
		}
		if amount > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += amount
	}
	return work
}

// newJXRLowTileRows 读取前预检全宽额度，原始索引和当前低通目标分别保留而不创建整图系数
// 入参: ctx Context, data 可选原流切片, source 完整原始流, bands 一或二, maximum 当前行总额度
// 返回: *jxrLowTileRows 跨分块连续低通行, error 资源、模型、能力或取消错误
func newJXRLowTileRows(ctx context.Context, data []byte, source *jxrStream, bands int, maximum uint64) (*jxrLowTileRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrLowTileRowsWork(source, bands) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: reduced tile row working budget", ErrLimit)
	}
	for _, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
		if plane != nil {
			if err := checkJXRModelComponents(plane); err != nil {
				return nil, err
			}
		}
	}
	columns := len(source.header.columns) - 1
	r := &jxrLowTileRows{source: *source, data: data, maximum: maximum, bands: bands, activeRow: -1, readers: make([]jxrLowTileReader, columns)}
	if source.header.frequency {
		var err error
		r.plan, err = newJXRLowFrequencyPlan(ctx, &r.source, bands, maximum)
		if err != nil {
			return nil, err
		}
	} else if len(source.packets) > 1 && !source.header.index {
		return nil, fmt.Errorf("%w: reduced tile indexed access", ErrUnsupported)
	}
	width := int(source.header.columns[columns])
	r.image.header = jxrRowReconstructionStream(source).header
	r.image.header.height, r.image.header.top, r.image.header.bottom, r.image.header.rows = 16, 0, 0, []uint32{0, 1}
	for index, plane := range [2]*jxrPlane{&source.primary, source.alpha} {
		if plane == nil {
			continue
		}
		current := jxrCoefficients{plane: *plane, longWord: source.header.longWord, low: make([]int64, width*plane.channels*16), info: make([]jxrMacroInfo, width)}
		if index == 0 {
			r.image.primary = current
		} else {
			r.image.alpha = &current
		}
	}
	return r, ctx.Err()
}

// readRow 惰性切换原块行并拼接当前所有列，包行状态仍按各原块独立推进，结果不含高通行
// 入参: ctx Context
// 返回: error 序列、包、数值或取消错误
func (r *jxrLowTileRows) readRow(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.next >= int(r.source.header.rows[len(r.source.header.rows)-1]) {
		return fmt.Errorf("%w: reduced tile row sequence", ErrInvalid)
	}
	for r.next >= int(r.source.header.rows[r.tileRow+1]) {
		r.tileRow++
	}
	columns := len(r.readers)
	if r.activeRow != r.tileRow {
		clear(r.readers)
		for column := range columns {
			tile := r.tileRow*columns + column
			var err error
			if r.plan != nil {
				r.readers[column].frequency, err = newJXRLowFrequencyRows(ctx, r.data, r.plan, tile, r.maximum)
			} else {
				r.readers[column].spatial, err = newJXRLowSpatialRows(ctx, r.data, &r.source, tile, r.bands, r.maximum)
			}
			if err != nil {
				return err
			}
		}
		r.activeRow = r.tileRow
	}
	for column := range columns {
		state := &r.readers[column]
		var current *jxrCoefficientImage
		if state.frequency != nil {
			if err := state.frequency.readRow(ctx); err != nil {
				return err
			}
			current = &state.frequency.image
		} else {
			if err := state.spatial.readRow(ctx); err != nil {
				return err
			}
			current = &state.spatial.image
		}
		for index, target := range [2]*jxrCoefficients{&r.image.primary, r.image.alpha} {
			if target != nil {
				from := [2]*jxrCoefficients{&current.primary, current.alpha}[index]
				at := int(r.source.header.columns[column])
				copy(target.low[at*target.plane.channels*16:], from.low)
				copy(target.info[at:], from.info)
			}
		}
	}
	r.next++
	return ctx.Err()
}

// decodeUncompressedReducedLocked 在单行有界工作内存内对无压缩容器图像执行缩减采样重建
// 入参: ctx 已规范化Context, divisor 二、四、八或十六缩减分母
// 返回: *Raster 全采样原生粗图, error 参数、读取、资源或取消错误
func (d *Decoder) decodeUncompressedReducedLocked(ctx context.Context, divisor int) (*Raster, error) {
	c := &d.source.container
	layout, err := jxrMakeUncompressedLayout(c.pixelFormat[15], c.format, c.width, c.height, c.separateAlpha())
	if err != nil {
		return nil, err
	}
	rowWork := layout.rowBufferWork()
	remaining := min(d.limits.MaxMemoryBytes, uint64(^uint(0)>>1))
	metadataExtra := uint64(cap(d.info.ICCProfile)) + uint64(cap(d.info.EXIF)) + uint64(cap(d.info.GPS)) + uint64(cap(d.info.XMP)) + uint64(cap(d.info.IPTC)) + jxrDescriptionWork(d.info.Description)
	for _, amount := range []uint64{d.inputWork(), d.retainedHeaderWork(), metadataExtra, rowWork} {
		if amount > remaining {
			return nil, fmt.Errorf("%w: reduced uncompressed budget", ErrLimit)
		}
		remaining -= amount
	}
	info, outputWork, err := jxrReducedInfo(d.info, divisor, remaining)
	if err != nil {
		return nil, err
	}
	output, err := newRasterContext(ctx, info, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: outputWork})
	if err != nil {
		return nil, err
	}
	d.bindInputContext(ctx)
	defer d.bindInputContext(nil)
	rowBuf := make([]byte, int(layout.rowBytes))
	var alphaBuf []byte
	if !layout.planar && layout.separateAlpha {
		alphaBuf = make([]byte, int(layout.alphaRowBytes))
	}
	codedW, codedH := int(c.width), int(c.height)
	outW, outH := (codedW+divisor-1)/divisor, (codedH+divisor-1)/divisor
	dispW := output.Bounds().Dx()
	transform := uint8(c.transform)
	if layout.planar {
		step := int(layout.sampleBytes)
		for ci := range len(d.info.Components) {
			sx, sy, baseOffset, srcRowBytes, inAlpha := layout.planeSpec(ci)
			planeData, planeInput := c.image, c.imageInput
			if inAlpha {
				planeData, planeInput = c.alpha, c.alphaInput
			}
			cw, ch := codedW/sx, codedH/sy
			buf := rowBuf[:int(srcRowBytes)]
			cachedRow := -1
			offX, offY := float64(sx-1)*0.5, float64(sy-1)*0.5
			for y := range outH {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				visY := min(float64(codedH-1), float64(y*divisor)+float64(divisor-1)/2)
				cy := max(0, min(ch-1, int(math.Floor((visY-offY)/float64(sy)+0.5))))
				if cy != cachedRow {
					if err := jxrReadUncompressedSpan(planeData, planeInput, ctx, baseOffset+uint64(cy)*srcRowBytes, buf); err != nil {
						return nil, err
					}
					cachedRow = cy
				}
				for x := range outW {
					if x&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					visX := min(float64(codedW-1), float64(x*divisor)+float64(divisor-1)/2)
					cx := max(0, min(cw-1, int(math.Floor((visX-offX)/float64(sx)+0.5))))
					val, err := jxrUnpackUncompressedSample(buf[cx*step:], layout.sampleBytes, layout.tenBit)
					if err != nil {
						return nil, err
					}
					px, py := jxrRasterPosition(x, y, outW, outH, transform)
					output.planes[ci].storeBits(py*dispW+px, val)
				}
			}
		}
		return output, ctx.Err()
	}
	totalComp := layout.channels
	if layout.hasAlpha {
		totalComp++
	}
	var samples [9]uint32
	cachedRow := -1
	for y := range outH {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		visY := min(float64(codedH-1), float64(y*divisor)+float64(divisor-1)/2)
		cy := max(0, min(codedH-1, int(math.Floor(visY+0.5))))
		if cy != cachedRow {
			if err := jxrReadUncompressedSpan(c.image, c.imageInput, ctx, uint64(cy)*layout.rowBytes, rowBuf); err != nil {
				return nil, err
			}
			if layout.separateAlpha {
				if err := jxrReadUncompressedSpan(c.alpha, c.alphaInput, ctx, uint64(cy)*layout.alphaRowBytes, alphaBuf); err != nil {
					return nil, err
				}
			}
			cachedRow = cy
		}
		for x := range outW {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			visX := min(float64(codedW-1), float64(x*divisor)+float64(divisor-1)/2)
			cx := max(0, min(codedW-1, int(math.Floor(visX+0.5))))
			if err := jxrUnpackUncompressedPixel(&layout, rowBuf, alphaBuf, cx, samples[:totalComp]); err != nil {
				return nil, err
			}
			px, py := jxrRasterPosition(x, y, outW, outH, transform)
			at := py*dispW + px
			for ci := range totalComp {
				output.planes[ci].storeBits(at, samples[ci])
			}
		}
	}
	return output, ctx.Err()
}
