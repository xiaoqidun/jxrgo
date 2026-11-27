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
)

// jxrScanOrder 保存低通及水平、垂直高通的初始逆扫描顺序
var jxrScanOrder = [2][16]uint8{{0, 4, 1, 5, 8, 2, 9, 6, 12, 3, 10, 13, 7, 14, 11, 15}, {0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 12, 15, 13, 10, 11, 14}}

// jxrQPBits 保存非零量化索引位数
var jxrQPBits = [17]uint8{0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4}

// jxrMacroInfo 保存频率模式跨包使用的宏块方向、量化索引及细化位数
type jxrMacroInfo struct {
	dcMode   uint8
	scanMode uint8
	lpIndex  uint8
	hpIndex  uint8
	bits     [2]uint8
}

// jxrCoefficients 保存单平面逐宏块的低通、高通系数及跨包状态
type jxrCoefficients struct {
	plane    jxrPlane
	longWord bool
	workers  int
	low      []int64
	high     []int64
	info     []jxrMacroInfo
}

// jxrCoefficientImage 保存有界双平面系数，不将系数解析视为像素重建
type jxrCoefficientImage struct {
	header  jxrHeader
	primary jxrCoefficients
	alpha   *jxrCoefficients
}

// jxrTileNeighbor 保存一个宏块的反量化前预测缓存及高通块标志
type jxrTileNeighbor struct {
	low     jxrPrediction
	pattern []uint16
}

// jxrMacroContext 保存宏块位置、状态调整标志及相邻宏块缓存
type jxrMacroContext struct {
	at          int
	firstColumn bool
	firstRow    bool
	reset       bool
	adapt       bool
	left        *jxrTileNeighbor
	top         *jxrTileNeighbor
	corner      *jxrTileNeighbor
}

// jxrScan 保存逐位置累计数量及自适应逆扫描顺序
type jxrScan struct {
	order  [16]uint8
	totals [16]uint16
}

// jxrComponentView 保存按宏块及分量存储的二维系数或样本视图，masks高低四位分别保存垂直和水平块内掩码
type jxrComponentView struct {
	data        []int64
	width       int
	height      int
	columns     int
	blockWidth  int
	blockHeight int
	micro       int
	stride      int
	offset      int
	wordBits    uint8
	xShift      uint8
	yShift      uint8
	masks       uint8
}

// jxrDecodedMacro 持有当前分块复用的DC、低通及逐分量标志，不在宏块间共享输出视图
type jxrDecodedMacro struct {
	dc    []int64
	low   [][16]int64
	flags []bool
}

// jxrTileQuant 保存分块量化集合、频带复用关系及可选的核心宏块索引表
type jxrTileQuant struct {
	dc       []uint8
	lp       [][]uint8
	hp       [][]uint8
	lpCount  int
	hpCount  int
	useDC    bool
	useLP    bool
	adaptive bool
	lpIndex  []uint8
	hpIndex  []uint8
}

// jxrPacket 保存分块包的有效性及精细位截断量
type jxrPacket struct {
	valid bool
	trim  uint8
}

// jxrTileOutput 保存包写入器、量化、残差及同形复用的预测行，各分块仍独立重置预测状态
type jxrTileOutput struct {
	packet    [4]*jxrBitWriter
	quant     [2]jxrTileQuant
	macro     [2]jxrEncodedMacro
	rows      [2]*jxrTileRow
	exact     bool
	validated bool
	trim      uint8
	maxMemory uint64
}

// coefficientWork 计算系数、宏块状态、图像头、分块索引及预测行缓存的总容量
// 返回: uint64 需要的工作字节数
func (s *jxrStream) coefficientWork() uint64 {
	if len(s.header.columns) < 2 || len(s.header.rows) < 2 {
		return ^uint64(0)
	}
	width := uint64(s.header.columns[len(s.header.columns)-1])
	height := uint64(s.header.rows[len(s.header.rows)-1])
	count := width * height
	channels, planes := s.primary.channels, uint64(1)
	if s.alpha != nil {
		channels += s.alpha.channels
		planes++
	}
	quant := s.primary.quantWork() + uint64(channels)*(33+137) + planes*128
	row := jxrPredictionRowWork(width, s.primary.channels)
	if s.alpha != nil {
		quant += s.alpha.quantWork()
		other := jxrPredictionRowWork(width, s.alpha.channels)
		if other > ^uint64(0)-row {
			return ^uint64(0)
		}
		row += other
	}
	stride := uint64(channels)*272*8 + planes*6
	if stride == 0 || count > ^uint64(0)/stride {
		return ^uint64(0)
	}
	work := count * stride
	for _, amount := range []uint64{row, uint64(len(s.packets)) * 8, uint64(len(s.header.columns)+len(s.header.rows)+len(s.profiles)) * 4, quant, 32768} {
		if amount > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += amount
	}
	return work
}

// tileScratchWork 计算单个额外并发分块所需的预测行、宏块缓冲及熵模型状态字节数
// 返回: uint64 单分块临时工作预算，无效布局或溢出返回最大值
func (s *jxrStream) tileScratchWork() uint64 {
	if len(s.header.columns) < 2 || len(s.header.rows) < 2 {
		return ^uint64(0)
	}
	maxWidth := uint64(0)
	for i := 0; i+1 < len(s.header.columns); i++ {
		w := uint64(s.header.columns[i+1] - s.header.columns[i])
		if w > maxWidth {
			maxWidth = w
		}
	}
	channels, planes := s.primary.channels, uint64(1)
	if s.alpha != nil {
		channels += s.alpha.channels
		planes++
	}
	quant := s.primary.quantWork() + uint64(channels)*(33+137) + planes*128
	row := jxrPredictionRowWork(maxWidth, s.primary.channels)
	if s.alpha != nil {
		quant += s.alpha.quantWork()
		other := jxrPredictionRowWork(maxWidth, s.alpha.channels)
		if other > ^uint64(0)-row {
			return ^uint64(0)
		}
		row += other
	}
	work := row
	for _, amount := range []uint64{quant, 32768} {
		if amount > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += amount
	}
	return work
}

// coefficientsContext 从已校验头部解码系数，分配前检查预算，按宏块响应取消
// 入参: ctx Context, data 当前独立码流, maxWork 系数及临时状态的字节容量
// 返回: *jxrCoefficientImage 双平面系数, error 数据、能力、资源或取消错误
func (s *jxrStream) coefficientsContext(ctx context.Context, data []byte, maxWork uint64) (*jxrCoefficientImage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.decodeBand > BandDC {
		return nil, fmt.Errorf("%w: JPEG XR decode band", ErrInvalid)
	}
	limit := min(maxWork, uint64(^uint(0)>>1))
	baseWork := s.coefficientWork()
	if baseWork > limit {
		return nil, &LimitError{Resource: "JPEG XR coefficient working budget", Limit: limit, Required: baseWork}
	}
	count := uint64(s.header.columns[len(s.header.columns)-1]) * uint64(s.header.rows[len(s.header.rows)-1])
	img := &jxrCoefficientImage{header: s.header, primary: newJXRCoefficients(&s.primary, int(count), s.header.longWord)}
	img.primary.workers = s.workers
	if s.alpha != nil {
		alpha := newJXRCoefficients(s.alpha, int(count), s.header.longWord)
		alpha.workers = s.workers
		img.alpha = &alpha
	}
	tiles := (len(s.header.columns) - 1) * (len(s.header.rows) - 1)
	workers := 1
	if tiles > 1 && s.workers > 1 && s.input == nil && s.header.index {
		perWorker := s.tileScratchWork()
		if perWorker != ^uint64(0) {
			workers = resolveJXRWorkers(s.workers, tiles)
			workers = budgetJXRWorkers(workers, jxrSaturatedAdd(limit-baseWork, perWorker), perWorker)
		}
	}
	if err := runJXRWorkers(ctx, workers, tiles, func(runCtx context.Context, tile int) error {
		if s.header.frequency {
			return img.readFrequency(runCtx, data, s, tile)
		}
		return img.readSpatial(runCtx, data, s, tile)
	}); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return img, nil
}

// newJXRCoefficients 分配已通过总容量检查的单平面系数缓冲
// 入参: plane 图像平面, count 宏块数量, longWord 是否使用三十二位数值范围
// 返回: jxrCoefficients 空系数缓冲
func newJXRCoefficients(plane *jxrPlane, count int, longWord bool) jxrCoefficients {
	return jxrCoefficients{plane: *plane, longWord: longWord, low: make([]int64, count*plane.channels*16), high: make([]int64, count*plane.channels*256), info: make([]jxrMacroInfo, count)}
}

// jxrEmptyFlexPacket 判断频率模式精细位包是否因逃逸值或尾余不足四字节而为空包
// 入参: s 已校验并完成空包归一化的码流头, data 独立码流, index 逻辑包编号
// 返回: bool 为空精细位包时为真
func jxrEmptyFlexPacket(s *jxrStream, data []byte, index int) bool {
	offset := s.packets[index]
	if offset == jxrPacketEscape {
		return true
	}
	length := uint64(len(data))
	if s.input != nil {
		length = s.input.size
	}
	return s.dataStart > length || offset > length-s.dataStart || length-s.dataStart-offset < 4
}

// jxrPacketReader 按已校验的包偏移定位并返回码流位读取器
// 入参: data 独立码流, s 已校验的码流头, index 包编号
// 返回: jxrBits 包起点读取器
func jxrPacketReader(data []byte, s *jxrStream, index int) jxrBits {
	return jxrBits{data: data, bit: (s.dataStart + s.packets[index]) * 8, input: s.input}
}

// lowMacro 取得当前宏块的逐分量低通视图
// 入参: at 宏块编号
// 返回: []int64 分量步长为16的系数视图
func (c *jxrCoefficients) lowMacro(at int) []int64 {
	stride := c.plane.channels * 16
	return c.low[at*stride : (at+1)*stride]
}

// highMacro 取得当前宏块的逐分量高通视图
// 入参: at 宏块编号
// 返回: []int64 分量步长为256的系数视图
func (c *jxrCoefficients) highMacro(at int) []int64 {
	stride := c.plane.channels * 256
	return c.high[at*stride : (at+1)*stride]
}

// storeDCComponents 按当前分块量化步长反量化并保存宏块DC系数
// 入参: at 宏块编号, q 分块量化参数, dc 逐分量预测后DC系数
// 返回: error 数值范围或计算容量错误
func (c *jxrCoefficients) storeDCComponents(at int, q *jxrTileQuant, dc []int64) error {
	low := c.lowMacro(at)
	for ch := range c.plane.channels {
		shift := uint8(0)
		if ch == 0 {
			shift = 1
		}
		value, err := scaleJXRWord(dc[ch], jxrQuantScale(q.dc[ch], c.plane.scaled, shift), c.longWord)
		if err != nil {
			return err
		}
		low[ch*16] = value
	}
	return nil
}

// storeLowComponents 保存宏块反量化系数，仅复制实际分量
// 入参: at 宏块编号, values 宏块系数
// 返回: error 数值范围错误
func (c *jxrCoefficients) storeLowComponents(at int, values [][16]int64) error {
	if err := c.checkLow(values); err != nil {
		return err
	}
	low := c.lowMacro(at)
	for i := range c.plane.channels {
		copy(low[i*16:(i+1)*16], values[i][:])
	}
	return nil
}

// finishHighComponents 完成高通反量化及宏块内预测，不修改熵解码阶段的扫描方向
// 入参: at 宏块编号, q 分块量化参数, low 逐分量低通方向工作缓冲
// 返回: error 索引或计算容量错误
func (c *jxrCoefficients) finishHighComponents(at int, q *jxrTileQuant, low [][16]int64) error {
	if len(low) < c.plane.channels {
		return fmt.Errorf("%w: JPEG XR highpass direction storage", ErrInvalid)
	}
	values := c.highMacro(at)
	if err := dequantJXRHigh(&c.plane, values, q, int(c.info[at].hpIndex), c.longWord); err != nil {
		return err
	}
	mode := c.info[at].scanMode
	if c.plane.legacyHighpassPrediction {
		view := c.lowMacro(at)
		for i := range c.plane.channels {
			copy(low[i][:], view[i*16:(i+1)*16])
		}
		var err error
		mode, err = jxrHighpassModeChecked(&c.plane, low[:c.plane.channels])
		if err != nil {
			return err
		}
	}
	if err := applyJXRHighPredictionChecked(&c.plane, values, mode, false); err != nil {
		return err
	}
	if !c.longWord && mode != 2 {
		return checkJXRWords(values, false)
	}
	return nil
}

// readSpatial 按宏块及平面顺序解析空间模式系数，状态不跨越分块
// 入参: ctx Context, data 独立码流, s 码流头, tile 分块编号
// 返回: error 数据、量化、计算容量或取消错误
func (img *jxrCoefficientImage) readSpatial(ctx context.Context, data []byte, s *jxrStream, tile int) error {
	r, err := newJXRSpatialRows(ctx, data, s, tile, s.coefficientWork())
	if err != nil {
		return err
	}
	columns := len(s.header.columns) - 1
	column, row := tile%columns, tile/columns
	leftEdge, topEdge := int(s.header.columns[column]), int(s.header.rows[row])
	stride := int(s.header.columns[len(s.header.columns)-1])
	for y := range r.height {
		if err := r.readRow(ctx, img, (topEdge+y)*stride+leftEdge); err != nil {
			return err
		}
	}
	return nil
}

// newJXRScan 初始化指定方向的逆扫描顺序及累计数量
// 入参: vertical 是否采用垂直高通的初始顺序
// 返回: jxrScan 初始状态
func newJXRScan(vertical bool) jxrScan {
	index := 0
	if vertical {
		index = 1
	}
	s := jxrScan{order: jxrScanOrder[index]}
	s.reset()
	return s
}

// reset 重置累计数量而不恢复已自适应的扫描顺序
func (s *jxrScan) reset() {
	for i := 1; i < 16; i++ {
		s.totals[i] = uint16(34 - i*2)
	}
}

// place 取得系数位置后更新扫描顺序，位置一不与未参与扫描的位置零交换
// 入参: position 已校验的一至十五扫描位置
// 返回: uint8 当前系数的实际位置
func (s *jxrScan) place(position int) uint8 {
	at := s.order[position]
	s.totals[position]++
	if position > 1 && s.totals[position] > s.totals[position-1] {
		s.totals[position], s.totals[position-1] = s.totals[position-1], s.totals[position]
		s.order[position], s.order[position-1] = s.order[position-1], s.order[position]
	}
	return at
}

// gather 按当前扫描顺序收集非零变长系数，并更新对应位置的累计数量
// 入参: values 已校验的十六项系数, bits 已校验的细化位数
// 返回: [15]jxrRunValue 游程系数, int 非零数量
func (s *jxrScan) gather(values []int64, bits uint8) ([15]jxrRunValue, int) {
	var runs [15]jxrRunValue
	count, run := 0, uint8(0)
	for k := 1; k < 16; k++ {
		value := values[s.order[k]]
		coarse := jxrAbs(value) >> bits
		if coarse != 0 {
			s.place(k)
			if value < 0 {
				coarse = -coarse
			}
			runs[count] = jxrRunValue{run: run, value: coarse}
			count++
			run = 0
		} else {
			run++
		}
	}
	return runs, count
}

// componentViews 创建实际分量的二维视图，不另行复制整幅图像
// 入参: h 图像头, samples 是否为二级重建样本
// 返回: []jxrComponentView 逐分量视图
func (c *jxrCoefficients) componentViews(h *jxrHeader, samples bool) []jxrComponentView {
	return c.componentViewsStorage(h, samples, nil)
}

// componentViewsStorage 优先使用调用方的临时描述容量，分量较多时仍按实际数量分配，不限制原生通道数
// 入参: h 图像头, samples 是否为二级重建样本, storage 可复用描述存储
// 返回: []jxrComponentView 逐分量视图，不复制系数或样本
func (c *jxrCoefficients) componentViewsStorage(h *jxrHeader, samples bool, storage []jxrComponentView) []jxrComponentView {
	columns := int(h.columns[len(h.columns)-1])
	rows := int(h.rows[len(h.rows)-1])
	data, micro, perComponent := c.low, 1, 16
	if samples {
		data, micro, perComponent = c.high, 4, 256
	}
	views := storage
	if cap(views) < c.plane.channels {
		views = make([]jxrComponentView, c.plane.channels)
	} else {
		views = views[:c.plane.channels:c.plane.channels]
	}
	wordBits := uint8(16)
	if h.longWord {
		wordBits = 32
	}
	for i := range views {
		blockWidth, blockHeight := 4, 4
		xShift, yShift := uint8(2), uint8(2)
		if i != 0 && (c.plane.color == 1 || c.plane.color == 2) {
			blockWidth = 2
			xShift = 1
			if c.plane.color == 1 {
				blockHeight = 2
				yShift = 1
			}
		}
		if samples {
			xShift, yShift = xShift+2, yShift+2
		}
		views[i] = jxrComponentView{data: data, width: columns * blockWidth * micro, height: rows * blockHeight * micro, columns: columns, blockWidth: blockWidth, blockHeight: blockHeight, micro: micro, stride: c.plane.channels * perComponent, offset: i * perComponent, wordBits: wordBits, xShift: xShift, yShift: yShift, masks: uint8(1<<xShift-1) | uint8(1<<yShift-1)<<4}
	}
	return views
}

// index 将已校验的二维坐标映射到宏块、分量、块及块内位置
// 入参: x, y 视图内坐标
// 返回: int 存储索引
func (v *jxrComponentView) index(x, y int) int {
	var macro, pos int
	if xs := v.xShift; xs != 0 && x|y >= 0 {
		macro = (y>>v.yShift)*v.columns + x>>xs
		x, y = x&int(v.masks&15), y&int(v.masks>>4)
		if xs > 2 {
			pos = (y&^3)<<xs | (x&^3|y&3)<<2 | x&3
		} else {
			pos = y<<xs | x
		}
	} else {
		m := v.micro
		w, h := v.blockWidth*m, v.blockHeight*m
		macro = (y/h)*v.columns + x/w
		x, y = x%w, y%h
		pos = ((y/m)*v.blockWidth+x/m)*m*m + (y%m)*m + x%m
	}
	return macro*v.stride + v.offset + pos
}

// at 取得已校验坐标的系数或样本值
// 入参: x, y 视图内坐标
// 返回: int64 当前值
func (v *jxrComponentView) at(x, y int) int64 {
	return v.data[v.index(x, y)]
}

// set 修改已校验坐标的系数或样本值
// 入参: x, y 视图内坐标, value 新值
func (v *jxrComponentView) set(x, y int, value int64) {
	v.data[v.index(x, y)] = value
}

// readFrequency 按独立频带包解析系数，预测缓存及编码状态分别属于当前分块
// 入参: ctx Context, data 独立码流, s 码流头, tile 分块编号
// 返回: error 数据、量化、计算容量或取消错误
func (img *jxrCoefficientImage) readFrequency(ctx context.Context, data []byte, s *jxrStream, tile int) error {
	planes := [2]*jxrCoefficients{&img.primary, img.alpha}
	q := [2]jxrTileQuant{newJXRTileQuant(&s.primary)}
	if s.alpha != nil {
		q[1] = newJXRTileQuant(s.alpha)
	}
	columns := len(s.header.columns) - 1
	column, row := tile%columns, tile/columns
	leftEdge, topEdge := int(s.header.columns[column]), int(s.header.rows[row])
	width := int(s.header.columns[column+1]) - leftEdge
	height := int(s.header.rows[row+1]) - topEdge
	stride := int(s.header.columns[len(s.header.columns)-1])
	var rows [2]*jxrTileRow
	var scratch [2]*jxrDecodedMacro
	for i, p := range planes {
		if p != nil {
			var err error
			rows[i], err = newJXRTileRow(ctx, width, p.plane.channels, s.coefficientWork())
			if err != nil {
				return err
			}
			scratch[i], err = newJXRDecodedMacro(ctx, p.plane.channels, s.coefficientWork())
			if err != nil {
				return err
			}
		}
	}
	dc := [2]jxrDC{newJXRDC(), newJXRDC()}
	lp := [2]jxrLP{newJXRLP(), newJXRLP()}
	hp := [2]jxrHP{newJXRHP(), newJXRHP()}
	streamCount := 4 - int(s.primary.bands)
	if streamCount < 1 || tile >= len(s.packets)/streamCount {
		return fmt.Errorf("%w: JPEG XR frequency packet", ErrInvalid)
	}
	count := min(streamCount, 4-int(s.decodeBand))
	for band := range count {
		if err := ctx.Err(); err != nil {
			return err
		}
		var b jxrBits
		var packet jxrPacket
		var err error
		packetIndex := tile*streamCount + band
		if band == 3 && jxrEmptyFlexPacket(s, data, packetIndex) {
			packet.trim = 15
		} else {
			b = jxrPacketReader(data, s, packetIndex)
			packet, err = readJXRPacket(&b, s, band, &q)
			if err != nil {
				return err
			}
			if !packet.valid {
				packet.trim = 15
			}
		}
		for _, row := range rows {
			if row != nil {
				row.reset()
			}
		}
		for y := range height {
			for _, row := range rows {
				if row != nil {
					row.startRow()
				}
			}
			for x := range width {
				if err := ctx.Err(); err != nil {
					return err
				}
				for i, p := range planes {
					if p == nil || band >= 4-int(p.plane.bands) {
						continue
					}
					row := rows[i]
					m := jxrMacroContext{at: (topEdge+y)*stride + leftEdge + x, firstColumn: x == 0, firstRow: y == 0, reset: x%16 == 0, adapt: x == width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
					info := &p.info[m.at]
					current := &row.current
					switch band {
					case 0:
						err = p.readDC(&b, &dc[i], &q[i], m, &current.low, scratch[i])
						if err == nil && count == 1 && p.plane.bands != 3 {
							err = p.storeDCComponents(m.at, &q[i], scratch[i].dc)
						}
					case 1:
						if q[i].lpCount > 1 && !q[i].useDC {
							info.lpIndex = uint8(readJXRQPIndex(&b, q[i].lpCount))
						}
						err = p.readLP(&b, &lp[i], &q[i], m, &current.low, scratch[i])
					case 2:
						if q[i].useLP {
							info.hpIndex = info.lpIndex
						} else if q[i].hpCount > 1 {
							info.hpIndex = uint8(readJXRQPIndex(&b, q[i].hpCount))
						}
						err = p.readHP(&b, &hp[i], m, false, 0, row.residual, current.pattern)
						if err == nil && (p.plane.bands == 1 || p.plane.bands == 0 && count == 3) {
							err = refineJXRHigh(&b, &p.plane, info.bits, 15, p.highMacro(m.at))
							if err == nil {
								err = p.finishHighComponents(m.at, &q[i], scratch[i].low)
							}
						}
					case 3:
						err = refineJXRHigh(&b, &p.plane, info.bits, packet.trim, p.highMacro(m.at))
						if err == nil {
							err = p.finishHighComponents(m.at, &q[i], scratch[i].low)
						}
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
		if packet.valid {
			b.align(true)
			if b.err != nil {
				return b.err
			}
		}
	}
	return nil
}

// readDC 读取并预测宏块DC，只有缺失低通频带时立即完成反量化
// 入参: b 码流读取器, decoder 分块DC状态, q 分块量化参数, m 当前宏块状态, cache 独立预测缓存, scratch 分块工作缓冲
// 返回: error 数据、缓存或计算容量错误
func (c *jxrCoefficients) readDC(b *jxrBits, decoder *jxrDC, q *jxrTileQuant, m jxrMacroContext, cache *jxrPrediction, scratch *jxrDecodedMacro) error {
	residual := scratch.dc
	decoder.readChecked(b, &c.plane, m.adapt, residual)
	if b.err != nil {
		return b.err
	}
	if err := checkJXRWords(residual[:c.plane.channels], c.longWord); err != nil {
		return err
	}
	mode, err := applyJXRDCChecked(&c.plane, residual[:c.plane.channels], &m.left.low, &m.top.low, &m.corner.low, m.firstColumn, m.firstRow, false)
	dc := residual
	if err != nil {
		return err
	}
	if err := checkJXRWords(dc[:c.plane.channels], c.longWord); err != nil {
		return err
	}
	c.info[m.at].dcMode = mode
	storeJXRLowChecked(&c.plane, dc[:c.plane.channels], nil, 0, cache)
	if c.plane.bands == 3 {
		values := scratch.low
		if err := dequantJXRLowComponents(&c.plane, dc, nil, q, 0, c.longWord, values); err != nil {
			return err
		}
		if err := c.storeLowComponents(m.at, values); err != nil {
			return err
		}
	} else {
		low := c.lowMacro(m.at)
		for i := range c.plane.channels {
			low[i*16] = dc[i]
		}
	}
	return nil
}

// readLP 读取并预测低通系数，先保存扫描方向和预测缓存，再反量化存储
// 入参: b 码流读取器, decoder 分块低通状态, q 分块量化参数, m 当前宏块状态, cache 独立预测缓存, scratch 分块工作缓冲
// 返回: error 数据、缓存或计算容量错误
func (c *jxrCoefficients) readLP(b *jxrBits, decoder *jxrLP, q *jxrTileQuant, m jxrMacroContext, cache *jxrPrediction, scratch *jxrDecodedMacro) error {
	values := scratch.low
	decoder.readChecked(b, &c.plane, m.reset, m.adapt, values, scratch.flags)
	if b.err != nil {
		return b.err
	}
	if err := c.checkLow(values); err != nil {
		return err
	}
	info := &c.info[m.at]
	if err := applyJXRLPChecked(&c.plane, values[:c.plane.channels], info.dcMode, info.lpIndex, &m.left.low, &m.top.low, false); err != nil {
		return err
	}
	if err := c.checkLow(values); err != nil {
		return err
	}
	dc := scratch.dc
	low := c.lowMacro(m.at)
	for i := range c.plane.channels {
		dc[i] = low[i*16]
	}
	storeJXRLowChecked(&c.plane, dc[:c.plane.channels], values[:c.plane.channels], info.lpIndex, cache)
	mode, err := jxrHighpassModeChecked(&c.plane, values[:c.plane.channels])
	if err != nil {
		return err
	}
	info.scanMode = mode
	if err := dequantJXRLowComponents(&c.plane, dc[:], values[:], q, int(info.lpIndex), c.longWord, values[:]); err != nil {
		return err
	}
	return c.storeLowComponents(m.at, values)
}

// readHP 重建高通块标志并读取高通系数，保留供频率精细位包使用的模型位数
// 入参: b 码流读取器, decoder 分块高通状态, m 宏块状态, spatial 是否为空间模式, trim 精细位截断量, residual, pattern 逐分量残差及重建标志缓冲
// 返回: error 数据或缓冲错误
func (c *jxrCoefficients) readHP(b *jxrBits, decoder *jxrHP, m jxrMacroContext, spatial bool, trim uint8, residual, pattern []uint16) error {
	decoder.cbp.readChecked(b, &c.plane, residual)
	if b.err != nil {
		return b.err
	}
	decoder.cbp.predictChecked(&c.plane, residual, m.left.pattern, m.top.pattern, m.firstColumn, m.firstRow, pattern)
	info := &c.info[m.at]
	control := jxrHPControl{vertical: info.scanMode == 1, reset: m.reset, adapt: m.adapt, spatial: spatial, trim: trim}
	info.bits = decoder.readChecked(b, &c.plane, pattern, control, c.highMacro(m.at))
	return b.err
}

// newJXRDecodedMacro 在预算内分配实际分量工作缓冲，三类存储仅在分块开始时创建
// 入参: ctx Context, channels 分量数, maximum 本缓冲可用的字节预算
// 返回: *jxrDecodedMacro 工作缓冲, error 参数、资源或取消错误
func newJXRDecodedMacro(ctx context.Context, channels int, maximum uint64) (*jxrDecodedMacro, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if channels < 1 || channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: JPEG XR decoded macro components", ErrInvalid)
	}
	if uint64(channels)*137+128 > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR decoded macro budget", ErrLimit)
	}
	scratch := &jxrDecodedMacro{dc: make([]int64, channels), low: make([][16]int64, channels), flags: make([]bool, channels)}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return scratch, nil
}

// newJXRTileQuant 初始化分块量化集合，图像级共用参数只保留一组
// 入参: p 图像平面
// 返回: jxrTileQuant 初始分块量化参数
func newJXRTileQuant(p *jxrPlane) jxrTileQuant {
	storage := make([]uint8, p.channels*3)
	q := jxrTileQuant{dc: storage[:p.channels:p.channels], lpCount: 1, hpCount: 1}
	groups := make([][]uint8, 2)
	q.lp, q.hp = groups[:1:1], groups[1:]
	q.lp[0], q.hp[0] = storage[p.channels:p.channels*2:p.channels*2], storage[p.channels*2:]
	copy(q.dc, p.dcQuant)
	copy(q.lp[0], p.lpQuant)
	copy(q.hp[0], p.hpQuant)
	return q
}

// quantWork 计算分块量化集合及宏块索引的保守存储容量，共用数组按各引用分别计入
// 返回: uint64 量化存储预算
func (q *jxrTileQuant) quantWork() uint64 {
	work, ok := jxrWorkTotal(^uint64(0), uint64(cap(q.dc)), uint64(cap(q.lpIndex)), uint64(cap(q.hpIndex)))
	if !ok {
		return ^uint64(0)
	}
	for _, groups := range [][][]uint8{q.lp, q.hp} {
		if uint64(cap(groups)) > (^uint64(0)-work)/24 {
			return ^uint64(0)
		}
		work += uint64(cap(groups)) * 24
		for _, values := range groups[:cap(groups)] {
			amount := uint64(cap(values))
			if amount > ^uint64(0)-work {
				return ^uint64(0)
			}
			work += amount
		}
	}
	return work
}

// readJXRPacket 读取分块包头及量化参数，损坏的精细位包按标准视为全零
// 入参: b 包起点读取器, s 所属码流, band 频带索引，空间模式为-1, q 主平面及透明度平面的分块参数
// 返回: jxrPacket 包头, error 数据或能力错误
func readJXRPacket(b *jxrBits, s *jxrStream, band int, q *[2]jxrTileQuant) (jxrPacket, error) {
	var packet jxrPacket
	if band < -1 || band > 3 || s.header.frequency != (band >= 0) {
		return packet, fmt.Errorf("%w: JPEG XR packet band", ErrInvalid)
	}
	if band == 3 && (b.bit/8 > b.length() || b.length()-b.bit/8 < 4) {
		packet.trim = 15
		return packet, nil
	}
	start := b.read(24)
	_ = b.read(8)
	if b.err != nil {
		return packet, b.err
	}
	if start != 1 {
		if band == 3 {
			packet.trim = 15
			return packet, nil
		}
		return packet, fmt.Errorf("%w: JPEG XR tile start code", ErrInvalid)
	}
	packet.valid = true
	if s.header.trim && (band == -1 || band == 3) {
		packet.trim = uint8(b.read(4))
	}
	planes := [2]*jxrPlane{&s.primary, s.alpha}
	for i, p := range planes {
		if p == nil {
			continue
		}
		if band == -1 || band == 0 {
			if !p.dcUniform {
				q[i].dc = readJXRQuant(b, p.channels)
			}
		}
		if p.bands != 3 && (band == -1 || band == 1) {
			if !p.lpUniform {
				q[i].useDC = b.flag()
				if q[i].useDC {
					q[i].lpCount, q[i].lp = 1, [][]uint8{q[i].dc}
				} else {
					q[i].lpCount = int(b.read(4)) + 1
					q[i].lp = make([][]uint8, q[i].lpCount)
					for j := range q[i].lpCount {
						q[i].lp[j] = readJXRQuant(b, p.channels)
					}
				}
			}
		}
		if p.bands < 2 && (band == -1 || band == 2) {
			if !p.hpUniform {
				q[i].useLP = b.flag()
				if q[i].useLP {
					q[i].hpCount, q[i].hp = q[i].lpCount, q[i].lp
				} else {
					q[i].hpCount = int(b.read(4)) + 1
					q[i].hp = make([][]uint8, q[i].hpCount)
					for j := range q[i].hpCount {
						q[i].hp[j] = readJXRQuant(b, p.channels)
					}
				}
			}
		}
	}
	return packet, b.err
}

// readJXRQPIndex 读取宏块量化索引，保留值不用于访问量化集合
// 入参: b 码流读取器, count 已声明的量化集合数量
// 返回: int 量化集合索引，错误保存在读取器中
func readJXRQPIndex(b *jxrBits, count int) int {
	if count < 2 || count > 16 {
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR quantization set count", ErrInvalid)
		}
		return 0
	}
	if !b.flag() {
		return 0
	}
	index := int(b.read(uint(jxrQPBits[count]))) + 1
	if index >= count && b.err == nil {
		b.err = fmt.Errorf("%w: JPEG XR reserved quantization index", ErrInvalid)
	}
	return index
}

// jxrDisplayTileIndex 将显示行优先序号反映射到编码分块序号，不分配排列或修改逻辑索引顺序
// 入参: h 已校验的编码几何及方向, ordinal 有效的显示遍历序号
// 返回: int 编码方向的分块序号
func jxrDisplayTileIndex(h *jxrHeader, ordinal int) int {
	columns, rows := len(h.columns)-1, len(h.rows)-1
	displayColumns, displayRows := columns, rows
	if h.transform&4 != 0 {
		displayColumns, displayRows = rows, columns
	}
	inverse := [8]uint8{0, 1, 2, 3, 7, 5, 6, 4}
	x, y := jxrRasterPosition(ordinal%displayColumns, ordinal/displayColumns, displayColumns, displayRows, inverse[h.transform])
	return y*columns + x
}

// writeTile 按分块及宏块顺序写入系数，频率模式将各频带写入独立包缓冲
// 入参: ctx Context, s 所属码流头, tile 分块编号, output 包写入器及独立工作缓冲
// 返回: [4]uint64 各频带包在对应缓冲中的字节起点, error 数据、资源、精度或取消错误
func (img *jxrCoefficientImage) writeTile(ctx context.Context, s *jxrStream, tile int, output *jxrTileOutput) ([4]uint64, error) {
	var offsets [4]uint64
	if err := ctx.Err(); err != nil {
		return offsets, err
	}
	if img == nil || s == nil || output == nil {
		return offsets, fmt.Errorf("%w: JPEG XR tile output", ErrInvalid)
	}
	for n := range output.macro {
		output.macro[n].cache = jxrPrediction{}
	}
	defer func() {
		for n := range output.macro {
			output.macro[n].cache = jxrPrediction{}
		}
	}()
	h := &s.header
	if !output.validated {
		if err := img.checkOutput(s); err != nil {
			return offsets, err
		}
	}
	columns, rows := len(h.columns)-1, len(h.rows)-1
	if tile < 0 || tile >= columns*rows {
		return offsets, fmt.Errorf("%w: JPEG XR output tile", ErrInvalid)
	}
	planes := [2]*jxrCoefficients{&img.primary, img.alpha}
	for n, p := range planes {
		if p == nil {
			continue
		}
		if len(output.macro[n].high) < p.plane.channels*256 {
			return offsets, fmt.Errorf("%w: JPEG XR output residual capacity", ErrInvalid)
		}
		if !output.validated {
			if err := checkJXRTileQuant(&p.plane, &output.quant[n]); err != nil {
				return offsets, err
			}
		}
	}
	bands := 1
	if h.frequency {
		bands = 4 - int(s.primary.bands)
	}
	column, row := tile%columns, tile/columns
	leftEdge, topEdge := int(h.columns[column]), int(h.rows[row])
	width, height := int(h.columns[column+1])-leftEdge, int(h.rows[row+1])-topEdge
	for n, p := range planes {
		if p != nil && output.quant[n].adaptive {
			for _, indices := range [][]uint8{output.quant[n].lpIndex, output.quant[n].hpIndex} {
				if len(indices) != 0 && uint64(len(indices)) != uint64(width)*uint64(height) {
					return offsets, fmt.Errorf("%w: JPEG XR output macroblock selection size", ErrInvalid)
				}
			}
		}
	}
	stride := int(h.columns[len(h.columns)-1])
	for n, cached := range output.rows {
		p := planes[n]
		if cached != nil && (p == nil || len(cached.top) != width || len(cached.residual) != p.plane.channels) {
			output.rows[n] = nil
		}
	}
	work := img.outputWork(s)
	for n, p := range planes {
		if p != nil {
			var minimum jxrEncodedMacro
			extra := output.macro[n].payloadWork(p.plane.channels) - minimum.payloadWork(p.plane.channels)
			if extra > ^uint64(0)-work {
				return offsets, &LimitError{Resource: "JPEG XR output scratch capacity", Limit: output.maxMemory, Required: ^uint64(0)}
			}
			work += extra
		}
	}
	if work == ^uint64(0) || work > output.maxMemory {
		return offsets, &LimitError{Resource: "JPEG XR tile output working budget", Limit: output.maxMemory, Required: work}
	}
	remaining := output.maxMemory - work
	var budget *jxrWriteBudget
	for band := range bands {
		writer := output.packet[band]
		if writer == nil || writer.bit%8 != 0 {
			return offsets, fmt.Errorf("%w: JPEG XR tile output packet buffer", ErrInvalid)
		}
		if writer.err != nil {
			return offsets, writer.err
		}
		for previous := 0; previous < band; previous++ {
			if output.packet[previous] == writer {
				return offsets, fmt.Errorf("%w: JPEG XR aliased frequency output buffers", ErrInvalid)
			}
		}
		if band == 0 {
			budget = writer.budget
		}
		if writer.budget != budget {
			return offsets, fmt.Errorf("%w: JPEG XR output packet budget ownership", ErrInvalid)
		}
		if budget == nil {
			if writer.maxMemory > remaining {
				return offsets, &LimitError{Resource: "JPEG XR tile output packet budget", Limit: remaining, Required: writer.maxMemory}
			}
			remaining -= writer.maxMemory
		} else if band == 0 {
			if budget.used > budget.limit {
				return offsets, &LimitError{Resource: "JPEG XR shared packet budget", Limit: budget.limit, Required: budget.used}
			}
			if budget.limit > remaining {
				return offsets, &LimitError{Resource: "JPEG XR shared packet budget", Limit: remaining, Required: budget.limit}
			}
			remaining -= budget.limit
		}
		offsets[band] = writer.bit / 8
	}
	for band := range bands {
		kind := band
		if !h.frequency {
			kind = -1
		}
		if output.validated {
			writeJXRPacketChecked(output.packet[band], s, kind, &output.quant, output.trim)
		} else {
			writeJXRPacket(output.packet[band], s, kind, &output.quant, output.trim)
		}
		if output.packet[band].err != nil {
			return offsets, output.packet[band].err
		}
	}
	dcWriter, lpWriter, hpWriter, flexWriter := output.packet[0], output.packet[0], output.packet[0], output.packet[0]
	if h.frequency {
		lpWriter, hpWriter, flexWriter = output.packet[1], output.packet[2], output.packet[3]
	}
	var cacheRows [2]*jxrTileRow
	for n, p := range planes {
		if p != nil {
			if err := output.macro[n].prepareLowStorage(p.plane.channels); err != nil {
				return offsets, err
			}
			if output.rows[n] == nil {
				var err error
				output.rows[n], err = newJXRTileRow(ctx, width, p.plane.channels, output.maxMemory)
				if err != nil {
					return offsets, err
				}
			} else {
				output.rows[n].reset()
			}
			cacheRows[n] = output.rows[n]
		}
	}
	dc, lp, hp := [2]jxrDC{newJXRDC(), newJXRDC()}, [2]jxrLP{newJXRLP(), newJXRLP()}, [2]jxrHP{newJXRHP(), newJXRHP()}
	for y := range height {
		for _, row := range cacheRows {
			if row != nil {
				row.startRow()
			}
		}
		for x := range width {
			if err := ctx.Err(); err != nil {
				return offsets, err
			}
			for n, p := range planes {
				if p == nil {
					continue
				}
				row := cacheRows[n]
				m := jxrMacroContext{at: (topEdge+y)*stride + leftEdge + x, firstColumn: x == 0, firstRow: y == 0, reset: x%16 == 0, adapt: x == width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
				q, scratch, info := &output.quant[n], &output.macro[n], &p.info[m.at]
				lpIndex, hpIndex := int(info.lpIndex), int(info.hpIndex)
				if q.adaptive {
					lpIndex, hpIndex = 0, 0
					if len(q.lpIndex) != 0 {
						lpIndex = int(q.lpIndex[y*width+x])
					}
					if q.useLP {
						hpIndex = lpIndex
					} else if len(q.hpIndex) != 0 {
						hpIndex = int(q.hpIndex[y*width+x])
					}
				}
				scratch.cache = row.current.low
				if err := p.prepareMacroChecked(ctx, q, lpIndex, hpIndex, m, output.exact, scratch); err != nil {
					return offsets, err
				}
				if p.plane.bands != 3 {
					if !q.useDC && q.lpCount > 1 {
						writeJXRQPIndex(lpWriter, q.lpCount, lpIndex)
						if lpWriter.err != nil {
							return offsets, lpWriter.err
						}
					}
					if p.plane.bands < 2 && !q.useLP && q.hpCount > 1 {
						writeJXRQPIndex(hpWriter, q.hpCount, hpIndex)
						if hpWriter.err != nil {
							return offsets, hpWriter.err
						}
					}
				}
				dc[n].writeChecked(dcWriter, &p.plane, scratch.dc, m.adapt)
				if dcWriter.err != nil {
					return offsets, dcWriter.err
				}
				current := jxrTileNeighbor{low: scratch.cache, pattern: row.current.pattern}
				if p.plane.bands != 3 {
					lp[n].writeChecked(lpWriter, &p.plane, scratch.lp, m.reset, m.adapt, scratch.flags)
					if lpWriter.err != nil {
						return offsets, lpWriter.err
					}
				}
				if p.plane.bands < 2 {
					pattern := current.pattern
					if err := makeJXRHighPattern(&p.plane, scratch.high, hp[n].model.bits, pattern); err != nil {
						return offsets, err
					}
					residual := row.residual
					hp[n].cbp.residualChecked(&p.plane, pattern, row.left.pattern, row.top[x].pattern, m.firstColumn, m.firstRow, residual)
					hp[n].cbp.writeChecked(hpWriter, &p.plane, residual)
					if hpWriter.err != nil {
						return offsets, hpWriter.err
					}
					control := jxrHPControl{vertical: scratch.scanMode == 1, spatial: !h.frequency, reset: m.reset, adapt: m.adapt, trim: output.trim}
					bits := hp[n].writeChecked(hpWriter, &p.plane, pattern, control, scratch.high, output.exact)
					if hpWriter.err != nil {
						return offsets, hpWriter.err
					}
					if h.frequency && p.plane.bands == 0 {
						writeJXRHighFlexChecked(flexWriter, &p.plane, bits, output.trim, scratch.high, output.exact)
					}
					current.pattern = pattern
				}
				for band := range bands {
					if output.packet[band].err != nil {
						return offsets, output.packet[band].err
					}
				}
				if err := row.advance(x, &current); err != nil {
					return offsets, err
				}
			}
		}
	}
	for band := range bands {
		output.packet[band].align()
		if output.packet[band].err != nil {
			return offsets, output.packet[band].err
		}
	}
	return offsets, ctx.Err()
}
