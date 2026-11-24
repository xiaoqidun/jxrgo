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

// jxrPrediction 保存反量化前的一项DC及六项低通预测系数和量化索引
type jxrPrediction struct {
	values [][7]int64
	index  uint8
}

// jxrTileRow 持有当前平面一行的独立预测缓存，宏块更新只复制有效分量
type jxrTileRow struct {
	top      []jxrTileNeighbor
	left     jxrTileNeighbor
	corner   jxrTileNeighbor
	current  jxrTileNeighbor
	residual []uint16
}

// jxrEncodedMacro 保存供熵编码使用的宏块残差、相邻预测缓存及两种高通方向
type jxrEncodedMacro struct {
	dc       []int64
	lp       [][16]int64
	workLow  []int64
	decoded  [][16]int64
	high     []int64
	flags    []bool
	cache    jxrPrediction
	dcMode   uint8
	scanMode uint8
	highMode uint8
}

// applyJXRDC 加入或移除同一规范性DC预测，缓存只读且不跨越分块
// 入参: p 图像平面, values DC系数, left, top, corner 相邻缓存, firstColumn, firstRow 分块边缘, residual 是否移除预测
// 返回: uint8 预测方向, error 缓冲、缓存或数值错误
func applyJXRDC(p *jxrPlane, values []int64, left, top, corner *jxrPrediction, firstColumn, firstRow, residual bool) (uint8, error) {
	mode := uint8(3)
	if p == nil || len(values) < p.channels {
		return mode, fmt.Errorf("%w: JPEG XR DC prediction buffer", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return mode, err
	}
	if err := checkJXRWords(values[:p.channels], true); err != nil {
		return mode, err
	}
	if !firstColumn && (left == nil || len(left.values) < p.channels) || !firstRow && (top == nil || len(top.values) < p.channels) || !firstColumn && !firstRow && (corner == nil || len(corner.values) < p.channels) {
		return mode, fmt.Errorf("%w: JPEG XR DC prediction cache", ErrInvalid)
	}
	for i := range p.channels {
		var sample [3]int64
		if !firstColumn {
			sample[0] = left.values[i][0]
		}
		if !firstRow {
			sample[1] = top.values[i][0]
		}
		if !firstColumn && !firstRow {
			sample[2] = corner.values[i][0]
		}
		if err := checkJXRWords(sample[:], true); err != nil {
			return mode, err
		}
	}
	if firstColumn && !firstRow {
		mode = 1
	} else if !firstColumn && firstRow {
		mode = 0
	} else if !firstColumn && !firstRow {
		horizontal := jxrAbs(corner.values[0][0] - left.values[0][0])
		vertical := jxrAbs(corner.values[0][0] - top.values[0][0])
		if p.color != 0 && p.color != 6 {
			scale := int64(2)
			if p.color == 1 {
				scale = 8
			} else if p.color == 2 {
				scale = 4
			}
			horizontal *= scale
			vertical *= scale
			for i := 1; i <= 2; i++ {
				horizontal += jxrAbs(corner.values[i][0] - left.values[i][0])
				vertical += jxrAbs(corner.values[i][0] - top.values[i][0])
			}
		}
		if err := checkJXRWords([]int64{horizontal, vertical}, true); err != nil {
			return mode, err
		}
		mode = 2
		if horizontal*4 < vertical {
			mode = 1
		} else if vertical*4 < horizontal {
			mode = 0
		}
	}
	for i := range p.channels {
		predicted := int64(0)
		switch mode {
		case 0:
			predicted = left.values[i][0]
		case 1:
			predicted = top.values[i][0]
		case 2:
			round := int64(0)
			if i != 0 && (p.color == 1 || p.color == 2) {
				round = 1
			}
			predicted = (left.values[i][0] + top.values[i][0] + round) >> 1
		}
		if residual {
			values[i] -= predicted
		} else {
			values[i] += predicted
		}
		if err := checkJXRWords(values[i:i+1], true); err != nil {
			return mode, err
		}
	}
	return mode, nil
}

// applyJXRLP 加入或移除低通预测，YUV422链式预测在逆过程先于相邻宏块预测移除
// 入参: p 图像平面, values 低通系数, dcMode DC方向, index 量化索引, left, top 相邻缓存, residual 是否移除预测
// 返回: error 缓冲、缓存、方向或数值错误
func applyJXRLP(p *jxrPlane, values [][16]int64, dcMode, index uint8, left, top *jxrPrediction, residual bool) error {
	if p == nil || len(values) < p.channels || dcMode > 3 {
		return fmt.Errorf("%w: JPEG XR lowpass prediction buffer or mode", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	mode := uint8(2)
	if dcMode == 0 {
		if left == nil || len(left.values) < p.channels {
			return fmt.Errorf("%w: JPEG XR left lowpass prediction cache", ErrInvalid)
		}
		if index == left.index {
			mode = 0
		}
	} else if dcMode == 1 {
		if top == nil || len(top.values) < p.channels {
			return fmt.Errorf("%w: JPEG XR top lowpass prediction cache", ErrInvalid)
		}
		if index == top.index {
			mode = 1
		}
	}
	sign := int64(1)
	if residual {
		sign = -1
	}
	for i := range p.channels {
		if err := checkJXRWords(values[i][1:], true); err != nil {
			return err
		}
		if mode == 0 {
			if err := checkJXRWords(left.values[i][:], true); err != nil {
				return err
			}
		} else if mode == 1 {
			if err := checkJXRWords(top.values[i][:], true); err != nil {
				return err
			}
		}
		if residual && i != 0 && p.color == 2 && dcMode == 1 {
			values[i][5] -= values[i][1]
		}
		if i == 0 || p.color != 1 && p.color != 2 {
			if mode == 0 {
				values[i][4] += sign * left.values[i][4]
				values[i][8] += sign * left.values[i][5]
				values[i][12] += sign * left.values[i][6]
			} else if mode == 1 {
				for j := 1; j <= 3; j++ {
					values[i][j] += sign * top.values[i][j]
				}
			}
		} else if p.color == 1 {
			if mode == 0 {
				values[i][2] += sign * left.values[i][2]
			} else if mode == 1 {
				values[i][1] += sign * top.values[i][1]
			}
		} else {
			if mode == 0 {
				for _, j := range [3]int{4, 2, 6} {
					values[i][j] += sign * left.values[i][j]
				}
			} else if mode == 1 {
				values[i][4] += sign * top.values[i][4]
				values[i][1] += sign * top.values[i][5]
			}
			if !residual && dcMode == 1 {
				values[i][5] += values[i][1]
			}
		}
		if err := checkJXRWords(values[i][1:], true); err != nil {
			return err
		}
	}
	return nil
}

// storeJXRLow 将反量化前的系数写入已有独立缓存，不分配或替换缓存存储
// 入参: p 图像平面, dc DC系数, lp 低通系数，nil表示全零, index 量化索引, dst 目标缓存
// 返回: error 缓冲容量错误
func storeJXRLow(p *jxrPlane, dc []int64, lp [][16]int64, index uint8, dst *jxrPrediction) error {
	if p == nil || dst == nil || len(dc) < p.channels || lp != nil && len(lp) < p.channels || len(dst.values) != p.channels {
		return fmt.Errorf("%w: JPEG XR prediction storage", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	clear(dst.values)
	dst.index = index
	for i := range p.channels {
		dst.values[i][0] = dc[i]
		if lp == nil {
			continue
		}
		if i == 0 || p.color != 1 && p.color != 2 {
			for j := 1; j <= 4; j++ {
				dst.values[i][j] = lp[i][j]
			}
			dst.values[i][5], dst.values[i][6] = lp[i][8], lp[i][12]
		} else if p.color == 1 {
			dst.values[i][1], dst.values[i][2] = lp[i][1], lp[i][2]
		} else {
			for _, j := range [5]int{1, 2, 4, 5, 6} {
				dst.values[i][j] = lp[i][j]
			}
		}
	}
	return nil
}

// jxrHighpassMode 按低通系数强度计算高通方向，多分量格式只用第一分量选择方向
// 入参: p 图像平面, values 当前阶段的DC及低通系数
// 返回: uint8 左、上或不预测的方向, error 缓冲或数值错误
func jxrHighpassMode(p *jxrPlane, values [][16]int64) (uint8, error) {
	if p == nil || len(values) < p.channels {
		return 2, fmt.Errorf("%w: JPEG XR highpass direction buffer", ErrInvalid)
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return 2, err
	}
	for i := range p.channels {
		if err := checkJXRWords(values[i][1:], true); err != nil {
			return 2, err
		}
	}
	horizontal := jxrAbs(values[0][1]) + jxrAbs(values[0][2]) + jxrAbs(values[0][3])
	vertical := jxrAbs(values[0][4]) + jxrAbs(values[0][8]) + jxrAbs(values[0][12])
	if p.color != 0 && p.color != 6 {
		for i := 1; i <= 2; i++ {
			horizontal += jxrAbs(values[i][1])
			if p.color == 1 {
				vertical += jxrAbs(values[i][2])
			} else if p.color == 2 {
				vertical += jxrAbs(values[i][2]) + jxrAbs(values[i][6])
				horizontal += jxrAbs(values[i][5])
			} else {
				vertical += jxrAbs(values[i][4])
			}
		}
	}
	if err := checkJXRWords([]int64{horizontal, vertical}, true); err != nil {
		return 2, err
	}
	if horizontal*4 < vertical {
		return 0, nil
	}
	if vertical*4 < horizontal {
		return 1, nil
	}
	return 2, nil
}

// residualJXRHigh 按逆块顺序移除宏块内高通预测，不在相邻子块被修改后使用其值
// 入参: p 平面格式, values 分量步长为256的系数缓冲, mode 左、上或不预测的方向
// 返回: error 缓冲、方向或数值错误
func residualJXRHigh(p *jxrPlane, values []int64, mode uint8) error {
	return applyJXRHighPrediction(p, values, mode, true)
}

// dequantJXRHigh 按高通量化集合缩放有效块的非DC系数
// 入参: p 图像平面, values 分量步长为256的高通系数缓冲, q 分块量化参数, index 当前量化集合索引, longWord 是否使用三十二位范围
// 返回: error 索引、缓冲或数值范围错误
func dequantJXRHigh(p *jxrPlane, values []int64, q *jxrTileQuant, index int, longWord bool) error {
	if p == nil || q == nil || p.channels < 1 || p.channels > rasterMaxComponents || index < 0 || index >= q.hpCount || index >= len(q.hp) || len(values) < p.channels*256 || len(q.hp[index]) != p.channels {
		return fmt.Errorf("%w: JPEG XR highpass quantization buffer or index", ErrInvalid)
	}
	for i := range p.channels {
		blocks := 16
		if i != 0 && p.color == 1 {
			blocks = 4
		} else if i != 0 && p.color == 2 {
			blocks = 8
		}
		factor := jxrQuantScale(q.hp[index][i], p.scaled, 1)
		for block := range blocks {
			for j := 1; j < 16; j++ {
				at := i*256 + block*16 + j
				value, err := scaleJXRWord(values[at], factor, longWord)
				if err != nil {
					return err
				}
				values[at] = value
			}
		}
	}
	return nil
}

// applyJXRHighPrediction 执行或移除宏块内高通预测，反向处理先访问子块以保留原始邻块值
// 入参: p 平面格式, values 系数缓冲, mode 预测方向, residual 是否移除预测
// 返回: error 格式、缓冲、方向或数值错误
func applyJXRHighPrediction(p *jxrPlane, values []int64, mode uint8, residual bool) error {
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if len(values) < p.channels*256 || mode > 2 {
		return fmt.Errorf("%w: JPEG XR highpass prediction buffer or mode", ErrInvalid)
	}
	if err := checkJXRWords(values[:p.channels*256], true); err != nil {
		return err
	}
	if mode == 2 {
		return nil
	}
	for i := range p.channels {
		width, height := 4, 4
		if i != 0 && (p.color == 1 || p.color == 2) {
			width = 2
			if p.color == 1 {
				height = 2
			}
		}
		for order := range width * height {
			block := order
			if residual {
				block = width*height - 1 - order
			}
			neighbor, step := block-1, 4
			if mode == 0 && block%width == 0 {
				continue
			}
			if mode == 1 {
				if block < width {
					continue
				}
				neighbor, step = block-width, 1
			}
			for j := 1; j <= 3; j++ {
				at, from := i*256+block*16+j*step, i*256+neighbor*16+j*step
				value := values[at] + values[from]
				if residual {
					value = values[at] - values[from]
				}
				if err := checkJXRWords([]int64{value}, true); err != nil {
					return err
				}
				values[at] = value
			}
		}
	}
	return nil
}

// jxrPredictionRowWork 计算预测行及三个工作缓存的全平台存储上限，溢出返回最大值
// 入参: width 分块宽度，单位为宏块, channels 分量数
// 返回: uint64 预测行存储预算
func jxrPredictionRowWork(width uint64, channels int) uint64 {
	if channels < 1 || channels > rasterMaxComponents || width > (^uint64(0)-256)/64 || width > ^uint64(0)-4 {
		return ^uint64(0)
	}
	work := uint64(256) + width*64
	if uint64(channels)*2 > ^uint64(0)-work {
		return ^uint64(0)
	}
	work += uint64(channels) * 2
	size := uint64(channels) * 58
	if width+3 > (^uint64(0)-work)/size {
		return ^uint64(0)
	}
	return work + (width+3)*size
}

// newJXRTileRow 在预算内一次分配预测行，所有缓存持有互不重叠的分量存储
// 入参: ctx Context, width 宏块宽度, channels 分量数, maximum 新行存储上限
// 返回: *jxrTileRow 预测行, error 参数、资源或取消错误
func newJXRTileRow(ctx context.Context, width, channels int, maximum uint64) (*jxrTileRow, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if width < 1 || channels < 1 || channels > rasterMaxComponents {
		return nil, fmt.Errorf("%w: JPEG XR prediction row dimensions", ErrInvalid)
	}
	work := jxrPredictionRowWork(uint64(width), channels)
	if work > maximum || work > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: JPEG XR prediction row memory", ErrLimit)
	}
	row := &jxrTileRow{top: make([]jxrTileNeighbor, width)}
	storage := make([][7]int64, (width+3)*channels)
	patterns := make([]uint16, (width+4)*channels)
	row.residual = patterns[(width+3)*channels : (width+4)*channels : (width+4)*channels]
	for n := range width + 3 {
		if n&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		cache := &row.current
		if n < width {
			cache = &row.top[n]
		} else if n == width {
			cache = &row.left
		} else if n == width+1 {
			cache = &row.corner
		}
		cache.low.values = storage[n*channels : (n+1)*channels : (n+1)*channels]
		cache.pattern = patterns[n*channels : (n+1)*channels : (n+1)*channels]
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return row, nil
}

// startRow 清空分块首列不使用的工作缓存，不重置上一行的有效数据
func (r *jxrTileRow) startRow() {
	for _, cache := range []*jxrTileNeighbor{&r.left, &r.corner, &r.current} {
		clear(cache.low.values)
		cache.low.index = 0
		clear(cache.pattern)
	}
	clear(r.residual)
}

// reset 清空频带独立的预测值并保留已分配存储
func (r *jxrTileRow) reset() {
	r.startRow()
	for n := range r.top {
		clear(r.top[n].low.values)
		r.top[n].low.index = 0
		clear(r.top[n].pattern)
	}
}

// advance 保存当前宏块和原上方宏块，先复制到独立左缓存以避免源切片别名覆盖
// 入参: x 当前列, current 当前宏块缓存
// 返回: error 缓存容量或列序号错误
func (r *jxrTileRow) advance(x int, current *jxrTileNeighbor) error {
	if x < 0 || x >= len(r.top) || current == nil || len(current.low.values) != len(r.left.low.values) || len(r.top[x].low.values) != len(r.left.low.values) || len(r.corner.low.values) != len(r.left.low.values) {
		return fmt.Errorf("%w: JPEG XR prediction row update", ErrInvalid)
	}
	count := len(r.left.low.values)
	if len(r.left.pattern) != count || len(r.top[x].pattern) != count || len(r.corner.pattern) != count || current.pattern != nil && len(current.pattern) != count {
		return fmt.Errorf("%w: JPEG XR prediction row pattern storage", ErrInvalid)
	}
	copy(r.left.low.values, current.low.values)
	r.left.low.index = current.low.index
	if current.pattern == nil {
		clear(r.left.pattern)
	} else {
		copy(r.left.pattern, current.pattern)
	}
	copy(r.corner.low.values, r.top[x].low.values)
	r.corner.low.index = r.top[x].low.index
	copy(r.corner.pattern, r.top[x].pattern)
	copy(r.top[x].low.values, r.left.low.values)
	r.top[x].low.index = r.left.low.index
	copy(r.top[x].pattern, r.left.pattern)
	return nil
}

// prepareMacro 按解码逆过程在复用输出缓冲中准备宏块残差，不修改原始变换系数
// 入参: ctx Context, q 分块量化集合, lpIndex, hpIndex 当前量化索引, m 宏块及相邻缓存, exact 是否要求精确恢复, output 独立且可复用的残差缓冲
// 返回: error 布局、容量、索引、精度、数值或取消错误
func (c *jxrCoefficients) prepareMacro(ctx context.Context, q *jxrTileQuant, lpIndex, hpIndex int, m jxrMacroContext, exact bool, output *jxrEncodedMacro) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	p := &c.plane
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if p.bands > 3 || m.at < 0 || m.at >= len(c.info) || output == nil || len(output.high) < p.channels*256 {
		return fmt.Errorf("%w: JPEG XR encoder macro layout", ErrInvalid)
	}
	if uint64(m.at)+1 > uint64(len(c.low))/uint64(p.channels*16) || uint64(m.at)+1 > uint64(len(c.high))/uint64(p.channels*256) {
		return fmt.Errorf("%w: JPEG XR encoder macro coefficient storage", ErrInvalid)
	}
	if q == nil || q.lpCount < 1 || q.lpCount > 16 || q.hpCount < 1 || q.hpCount > 16 || len(q.lp) < q.lpCount || len(q.hp) < q.hpCount || lpIndex < 0 || lpIndex >= q.lpCount || hpIndex < 0 || hpIndex >= q.hpCount {
		return fmt.Errorf("%w: JPEG XR encoder quantization index", ErrInvalid)
	}
	if len(q.dc) != p.channels || len(q.lp[lpIndex]) != p.channels || len(q.hp[hpIndex]) != p.channels {
		return fmt.Errorf("%w: JPEG XR encoder quantization component count", ErrInvalid)
	}
	if p.bands != 3 && q.useDC {
		if q.lpCount != 1 || !slices.Equal(q.lp[0][:p.channels], q.dc[:p.channels]) {
			return fmt.Errorf("%w: JPEG XR encoder DC quantization reuse", ErrInvalid)
		}
	}
	if p.bands < 2 && q.useLP {
		if q.hpCount != q.lpCount || hpIndex != lpIndex || !slices.Equal(q.hp[hpIndex][:p.channels], q.lp[lpIndex][:p.channels]) {
			return fmt.Errorf("%w: JPEG XR encoder lowpass quantization reuse", ErrInvalid)
		}
	}
	low, high := c.lowMacro(m.at), c.highMacro(m.at)
	if err := checkJXRWordsContext(ctx, low, c.longWord); err != nil {
		return err
	}
	if err := checkJXRWordsContext(ctx, high, c.longWord); err != nil {
		return err
	}
	if err := output.prepareLowStorage(p.channels); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	values := output.workLow
	copy(values, low)
	copy(output.high, high)
	for index := range p.channels {
		count, blocks := 16, 16
		if index != 0 && p.color == 1 {
			count, blocks = 4, 4
		} else if index != 0 && p.color == 2 {
			count, blocks = 8, 8
		}
		if p.bands == 3 {
			for at := 1; at < count; at++ {
				if exact && values[index*16+at] != 0 {
					return fmt.Errorf("%w: JPEG XR omitted lowpass precision", ErrUnsupported)
				}
				values[index*16+at] = 0
			}
		}
		if p.bands >= 2 {
			for block := range blocks {
				for at := 1; at < 16; at++ {
					position := index*256 + block*16 + at
					if exact && output.high[position] != 0 {
						return fmt.Errorf("%w: JPEG XR omitted highpass precision", ErrUnsupported)
					}
					output.high[position] = 0
				}
			}
		}
	}
	if err := quantJXRLow(ctx, p, values, q.dc[:p.channels], q.lp[lpIndex][:p.channels], c.longWord, exact); err != nil {
		return err
	}
	clear(output.dc)
	clear(output.lp)
	for index := range p.channels {
		output.dc[index] = values[index*16]
		copy(output.lp[index][1:], values[index*16+1:(index+1)*16])
	}
	if output.cache.values == nil {
		output.cache.values = make([][7]int64, p.channels)
	}
	if err := storeJXRLow(p, output.dc[:p.channels], output.lp[:p.channels], uint8(lpIndex), &output.cache); err != nil {
		return err
	}
	output.scanMode, output.highMode = 2, 2
	if p.bands < 2 {
		if err := c.prepareHighpass(ctx, q, lpIndex, hpIndex, exact, output); err != nil {
			return err
		}
	}
	var left, top, corner *jxrPrediction
	if m.left != nil {
		left = &m.left.low
	}
	if m.top != nil {
		top = &m.top.low
	}
	if m.corner != nil {
		corner = &m.corner.low
	}
	mode, err := applyJXRDC(p, output.dc[:p.channels], left, top, corner, m.firstColumn, m.firstRow, true)
	if err != nil {
		return err
	}
	output.dcMode = mode
	if p.bands != 3 {
		if err := applyJXRLP(p, output.lp[:p.channels], mode, uint8(lpIndex), left, top, true); err != nil {
			return err
		}
	}
	if err := checkJXRWords(output.dc[:p.channels], c.longWord); err != nil {
		return err
	}
	for index := range p.channels {
		if err := checkJXRWords(output.lp[index][1:], c.longWord); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// prepareHighpass 计算熵扫描和重建方向，并按对应解码过程准备有界残差
// 入参: ctx Context, q 分块量化集合, lpIndex 低通量化索引, hpIndex 高通量化索引, exact 是否精确恢复, output 宏块编码输出缓冲
// 返回: error 预测、量化、残差范围或取消错误
func (c *jxrCoefficients) prepareHighpass(ctx context.Context, q *jxrTileQuant, lpIndex, hpIndex int, exact bool, output *jxrEncodedMacro) error {
	p := &c.plane
	mode, err := jxrHighpassMode(p, output.lp[:p.channels])
	if err != nil {
		return err
	}
	output.scanMode = mode
	hp := q.hp[hpIndex]
	if !p.legacyHighpassPrediction {
		output.highMode = mode
		if err := quantJXRHigh(ctx, p, output.high, hp, c.longWord, exact); err != nil {
			return err
		}
		if err := residualJXRHigh(p, output.high, mode); err != nil {
			return err
		}
		return c.checkQuantizedHighResidual(ctx, output.high, hp)
	}
	if err := dequantJXRLowComponents(p, output.dc, output.lp, q, lpIndex, c.longWord, output.decoded); err != nil {
		return err
	}
	mode, err = jxrHighpassMode(p, output.decoded)
	if err != nil {
		return err
	}
	output.highMode = mode
	if err := residualJXRHigh(p, output.high, mode); err != nil {
		return err
	}
	return quantJXRHigh(ctx, p, output.high, hp, c.longWord, exact)
}

// checkQuantizedHighResidual 确认参考顺序的残差可在声明的字长内反量化
// 入参: ctx Context, values 高通量化残差, hp 各分量高通量化参数
// 返回: error 残差越界或取消错误
func (c *jxrCoefficients) checkQuantizedHighResidual(ctx context.Context, values []int64, hp []uint8) error {
	low, high := jxrWordBounds(c.longWord)
	for channel := range c.plane.channels {
		factor := jxrQuantScale(hp[channel], c.plane.scaled, 1)
		minimum, maximum := low/factor, high/factor
		for _, value := range values[channel*256 : (channel+1)*256] {
			if value < minimum || value > maximum {
				return fmt.Errorf("%w: JPEG XR dequantized highpass residual range", ErrInvalid)
			}
		}
	}
	return ctx.Err()
}

// prepareLowStorage 首次为实际分量数建立独立残差和临时存储，后续宏块复用且不扩大已有缓冲
// 入参: channels 已校验的分量数
// 返回: error 已有缓冲布局错误
func (m *jxrEncodedMacro) prepareLowStorage(channels int) error {
	if channels < 1 || channels > rasterMaxComponents || (m.dc == nil) != (m.workLow == nil) || (m.lp == nil) != (m.decoded == nil) || m.dc != nil && (len(m.dc) != channels || len(m.workLow) != channels*16) || m.lp != nil && (len(m.lp) != channels || len(m.decoded) != channels) {
		return fmt.Errorf("%w: JPEG XR macro residual storage", ErrInvalid)
	}
	if m.flags != nil && len(m.flags) != channels {
		return fmt.Errorf("%w: JPEG XR macro lowpass flag storage", ErrInvalid)
	}
	if m.dc == nil {
		storage := make([]int64, channels*17)
		m.dc, m.workLow = storage[:channels:channels], storage[channels:channels*17:channels*17]
	}
	if m.lp == nil {
		storage := make([][16]int64, channels*2)
		m.lp, m.decoded = storage[:channels:channels], storage[channels:channels*2:channels*2]
	}
	if m.flags == nil {
		m.flags = make([]bool, channels)
	}
	return nil
}

// payloadWork 计算宏块持有或首次使用时需要的系数缓冲，包含切片隐藏容量，溢出返回最大值
// 入参: channels 分量数
// 返回: uint64 高通、DC、低通及双工作缓冲的字节容量，不重复计入预测行持有的缓存
func (m *jxrEncodedMacro) payloadWork(channels int) uint64 {
	if channels < 1 || channels > rasterMaxComponents {
		return ^uint64(0)
	}
	work := uint64(0)
	for _, term := range [][2]uint64{
		{uint64(max(cap(m.high), channels*256)), 8},
		{uint64(max(cap(m.dc), channels)), 8},
		{uint64(max(cap(m.workLow), channels*16)), 8},
		{uint64(max(cap(m.lp), channels)), 128},
		{uint64(max(cap(m.decoded), channels)), 128},
		{uint64(max(cap(m.flags), channels)), 1},
	} {
		if term[0] > (^uint64(0)-work)/term[1] {
			return ^uint64(0)
		}
		work += term[0] * term[1]
	}
	return work
}
