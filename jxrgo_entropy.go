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
	"math/bits"
)

// jxrLongWordMin 与 jxrLongWordMax 保存三十二位有符号系数的允许上下界
const (
	jxrLongWordMin = -int64(1) << 31
	jxrLongWordMax = int64(1)<<31 - 1
)

// jxrLevelBase 保存绝对值编码初值
var jxrLevelBase = [6]int64{2, 3, 4, 6, 10, 14}

// jxrLevelBits 保存绝对值初值对应的细化位数
var jxrLevelBits = [6]uint8{0, 0, 1, 2, 2, 2}

// jxrDCFlagsWriteCode 保存DC联合非零标志前缀码
var jxrDCFlagsWriteCode = jxrMakeWriteVLC(jxrMakeVLC([]string{"10", "001", "00001", "0001", "11", "010", "00000", "011"}))

// jxrAbsoluteWriteCode 保存双绝对值前缀码
var jxrAbsoluteWriteCode = [2][16]jxrWriteCode{
	jxrMakeWriteVLC(jxrMakeVLC([]string{"01", "10", "11", "001", "0001", "00000", "00001"})),
	jxrMakeWriteVLC(jxrMakeVLC([]string{"1", "01", "001", "0001", "00001", "000000", "000001"})),
}

// jxrLPTranspose 保存各分量尺寸对应的低通细化顺序
var jxrLPTranspose = [3][16]uint8{{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, {0, 2, 1, 3, 4, 6, 5, 7}, {0, 2, 1, 3}}

// jxrLPChromaRemap 保存抽样色度联合编码的系数映射
var jxrLPChromaRemap = [7]uint8{4, 1, 2, 3, 5, 6, 7}

// jxrLPPatternWriteCode 保存低通联合非零标志前缀码
var jxrLPPatternWriteCode = [2][16]jxrWriteCode{
	jxrMakeWriteVLC(jxrMakeVLC([]string{"0", "100", "1010", "1011", "1100", "1101", "1110", "1111"})),
	jxrMakeWriteVLC(jxrMakeVLC([]string{"0", "10", "110", "111"})),
}

// jxrHPBlockOrder 保存四乘四高通块的分组扫描到行优先位置的映射
var jxrHPBlockOrder = [16]uint8{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}

// jxrCBPGroupCode 保存高通组及非抽样单分量块的双编码表
var jxrCBPGroupCode = [2][256]jxrCode{jxrMakeVLC([]string{"1", "01", "001", "0000", "0001"}), jxrMakeVLC([]string{"1", "000", "001", "010", "011"})}

// jxrCBPChromaCode 保存亮色联合高通块的双编码表
var jxrCBPChromaCode = [2][256]jxrCode{
	jxrMakeVLC([]string{"010", "00000", "0010", "00001", "00010", "1", "011", "00011", "0011"}),
	jxrMakeVLC([]string{"1", "001", "010", "0001", "000001", "011", "00001", "0000000", "0000001"}),
}

// jxrCBPGroupDelta 保存单分量高通组及块的编码表判别增量
var jxrCBPGroupDelta = [5]int{0, -1, 0, 1, 1}

// jxrCBPChromaDelta 保存亮色联合高通块的编码表判别增量
var jxrCBPChromaDelta = [9]int{2, 2, 1, 1, -1, -2, -2, -2, -3}

// jxrCBPRefineBits 保存块组位置细化的位数
var jxrCBPRefineBits = [6]uint8{0, 2, 1, 2, 2, 0}

// jxrCBPRefineOffset 保存块组位置细化的初始索引
var jxrCBPRefineOffset = [6]uint8{0, 4, 2, 8, 12, 1}

// jxrCBPRefineValue 保存块组非零位置的位掩码
var jxrCBPRefineValue = [16]uint8{0, 15, 3, 12, 1, 2, 4, 8, 5, 6, 9, 10, 7, 11, 13, 14}

// jxrCBP422Shift 保存4:2:2色度块组的行优先位偏移
var jxrCBP422Shift = [4]uint8{0, 1, 4, 5}

// jxrCBPGroupWriteCode 保存高通组及独立分量块数量的逐符号编码
var jxrCBPGroupWriteCode = [2][16]jxrWriteCode{jxrMakeWriteVLC(jxrCBPGroupCode[0]), jxrMakeWriteVLC(jxrCBPGroupCode[1])}

// jxrCBPChromaWriteCode 保存亮色联合高通块标志的逐符号编码
var jxrCBPChromaWriteCode = [2][16]jxrWriteCode{jxrMakeWriteVLC(jxrCBPChromaCode[0]), jxrMakeWriteVLC(jxrCBPChromaCode[1])}

// jxrThirdWriteCode 保存三符号前缀码
var jxrThirdWriteCode = jxrMakeWriteVLC(jxrMakeVLC([]string{"1", "01", "00"}))

// jxrChromaCountWriteCode 保存色度非零块数量前缀码
var jxrChromaCountWriteCode = jxrMakeWriteVLC(jxrMakeVLC([]string{"1", "01", "000", "001"}))

// jxrCBPRefineWriteCode 保存双非零块位置前缀码
var jxrCBPRefineWriteCode = [16]jxrWriteCode{3: {0, 2}, 5: {1, 2}, 6: {4, 3}, 9: {5, 3}, 10: {6, 3}, 12: {7, 3}}

// jxrCBPNibbleClass 保存四位掩码对应的标准位置细化类别
var jxrCBPNibbleClass = [16]uint8{0, 1, 1, 2, 1, 3, 3, 4, 1, 3, 3, 4, 2, 4, 4, 5}

// jxrCBPNibbleRef 保存四位掩码在对应细化类别中的索引
var jxrCBPNibbleRef = [16]uint8{0, 0, 1, 0, 2, 0, 1, 0, 3, 2, 3, 1, 1, 2, 3, 0}

// jxrRunBase 保存游程初值
var jxrRunBase = [15]uint8{1, 2, 3, 5, 7, 1, 2, 3, 5, 7, 1, 2, 3, 4, 5}

// jxrRunBits 保存游程细化位数
var jxrRunBits = [15]uint8{0, 0, 1, 1, 3, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1}

// jxrRunBin 保存按最大游程选择的编码区间
var jxrRunBin = [15]int8{-1, -1, -1, -1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0}

// jxrRunWriteCode 保存游程索引前缀码
var jxrRunWriteCode = jxrMakeWriteVLC(jxrMakeVLC([]string{"1", "01", "001", "0000", "0001"}))

// jxrLastIndexWriteCode 保存倒数第二位置索引前缀码
var jxrLastIndexWriteCode = jxrMakeWriteVLC(jxrMakeVLC([]string{"0", "110", "10", "111"}))

// jxrRunCombinedCode 保存按最大游程与游程值直接索引的合并游程编码表
var jxrRunCombinedCode = jxrMakeRunWriteTable()

// jxrPairVLC 保存双编码表选择及累积判别值
type jxrPairVLC struct {
	table        uint8
	discriminant int
}

// jxrDC 保存分块DC系数的独立归一化和亮色编码状态
type jxrDC struct {
	model  jxrModel
	luma   jxrPairVLC
	chroma jxrPairVLC
}

// jxrLP 保存分块低通系数的编码、归一化、扫描及零块计数状态
type jxrLP struct {
	block     jxrBlock
	model     jxrModel
	scan      jxrScan
	countZero int
	countMax  int
}

// jxrHPControl 保存宏块高通的扫描方向、状态调整及空间模式控制
type jxrHPControl struct {
	vertical bool
	reset    bool
	adapt    bool
	spatial  bool
	trim     uint8
}

// jxrHP 保存高通频带的独立编码、归一化、双向扫描及块标志状态
type jxrHP struct {
	block jxrBlock
	model jxrModel
	scan  [2]jxrScan
	cbp   jxrCBP
}

// jxrCBP 保存高通块标志的编码表与逐分量组预测状态
type jxrCBP struct {
	group  jxrPairVLC
	block  jxrPairVLC
	state  [2]uint8
	ones   [2]int
	zeroes [2]int
}

// jxrRunValue 保存一个非零系数前的游程及有符号值
type jxrRunValue struct {
	run   uint8
	value int64
}

// jxrBlock 保存低通或高通频带独立的亮色编码状态
type jxrBlock struct {
	first [2]jxrMultiVLC
	index [2][2]jxrMultiVLC
	level [2]jxrPairVLC
}

// newJXRDC 初始化分块DC系数的熵编解码状态
// 返回: jxrDC 初始状态
func newJXRDC() jxrDC {
	return jxrDC{model: newJXRModel(0)}
}

// readChecked 在平面静态布局已校验前提下读取DC残差并更新模型
// 入参: b 读取器, p 已校验平面格式, adapt 是否调整编码表, values 残差缓冲
func (c *jxrDC) readChecked(b *jxrBits, p *jxrPlane, adapt bool, values []int64) {
	if b.err != nil {
		return
	}
	mean := c.readResidualChecked(b, p, values)
	if b.err == nil {
		c.model.update(mean, p, 0)
		if adapt {
			c.luma.adapt()
			c.chroma.adapt()
		}
	}
}

// readResidualChecked 在平面分量已校验前提下读取逐分量DC残差语法
// 入参: b 读取器, p 已校验平面格式, values 调用方提供的残差缓冲
// 返回: [2]int 亮度及其他分量的非零计数，错误保存在读取器中
func (c *jxrDC) readResidualChecked(b *jxrBits, p *jxrPlane, values []int64) [2]int {
	var mean [2]int
	if len(values) < p.channels || c.model.bits[0] > 15 || c.model.bits[1] > 15 || c.luma.table > 1 || c.chroma.table > 1 {
		b.err = fmt.Errorf("%w: JPEG XR DC input storage or state", ErrInvalid)
		return mean
	}
	clear(values[:p.channels])
	if p.color == 0 || p.color == 4 || p.color == 6 {
		for n := range p.channels {
			group := 0
			if n != 0 {
				group = 1
			}
			present := b.flag()
			if present {
				mean[group]++
			}
			values[n] = readJXRDCValue(b, &c.luma, c.model.bits[group], present)
			if b.err != nil {
				return mean
			}
		}
	} else {
		flags := readJXRDCFlags(b)
		for n := range 3 {
			group, level := 0, &c.luma
			if n != 0 {
				group, level = 1, &c.chroma
			}
			present := flags&(4>>n) != 0
			if present {
				mean[group]++
			}
			values[n] = readJXRDCValue(b, level, c.model.bits[group], present)
		}
	}
	return mean
}

// readJXRDCFlags 读取亮度和双路色度的联合非零标志
// 入参: b 码流读取器
// 返回: uint8 三分量标志，错误保存在读取器中
func readJXRDCFlags(b *jxrBits) uint8 {
	switch b.read(2) {
	case 2:
		return 0
	case 3:
		return 4
	case 1:
		return 5 + uint8(b.read(1))*2
	default:
		if b.flag() {
			return 1
		}
		if b.flag() {
			return 3
		}
		if b.flag() {
			return 2
		}
		return 6
	}
}

// readJXRDCValue 组合绝对值编码、细化位及符号位，使用宽整数避免32位平台溢出
// 入参: b 码流读取器, level 绝对值编码状态, bits 细化位数, present 是否存在变长编码部分
// 返回: int64 有符号DC残差，错误保存在读取器中
func readJXRDCValue(b *jxrBits, level *jxrPairVLC, bits uint8, present bool) int64 {
	value := int64(0)
	if present {
		value = level.absolute(b) - 1
	}
	if bits != 0 {
		value = value<<bits | int64(b.read(uint(bits)))
	}
	if value != 0 && b.flag() {
		value = -value
	}
	return value
}

// absolute 读取系数绝对值并累计编码表选择的判别值
// 入参: b 码流读取器
// 返回: int64 系数绝对值，错误保存在读取器中
func (v *jxrPairVLC) absolute(b *jxrBits) int64 {
	index := uint8(0)
	if v.table == 0 {
		if b.flag() {
			index = 1 + uint8(b.read(1))
		} else if b.flag() {
			index = 0
		} else if b.flag() {
			index = 3
		} else if b.flag() {
			index = 4
		} else {
			index = 5 + uint8(b.read(1))
		}
	} else {
		for index < 5 && !b.flag() {
			index++
		}
		if index == 5 {
			index += uint8(b.read(1))
		}
	}
	if b.err != nil {
		return 0
	}
	if index == 0 {
		v.discriminant++
	} else if index > 1 {
		v.discriminant--
	}
	if index < 6 {
		return jxrLevelBase[index] + int64(b.read(uint(jxrLevelBits[index])))
	}
	bits := b.read(4) + 4
	if bits == 19 {
		bits += b.read(2)
		if bits == 22 {
			bits += b.read(3)
		}
	}
	return 2 + (int64(1) << bits) + int64(b.read(uint(bits)))
}

// adapt 根据判别值切换双编码表，未切换时限制累积判别值
func (v *jxrPairVLC) adapt() {
	if v.discriminant < -8 && v.table != 0 {
		v.table, v.discriminant = 0, 0
	} else if v.discriminant > 8 && v.table != 1 {
		v.table, v.discriminant = 1, 0
	} else {
		v.discriminant = min(64, max(-64, v.discriminant))
	}
}

// writeChecked 在平面静态布局已校验前提下写入宏块DC残差并更新模型
// 入参: b 位写入器, p 已校验平面格式, values DC残差, adapt 是否在宏块后调整编码表
func (c *jxrDC) writeChecked(b *jxrBitWriter, p *jxrPlane, values []int64, adapt bool) {
	if b.err != nil {
		return
	}
	mean := c.writeResidualChecked(b, p, values)
	if b.err == nil {
		c.model.update(mean, p, 0)
		if adapt {
			c.luma.adapt()
			c.chroma.adapt()
		}
	}
}

// writeResidualChecked 在平面分量和Context已校验前提下写入逐分量DC残差语法
// 入参: b 写入器, p 已校验平面格式, values 逐分量残差
// 返回: [2]int 亮度及其他分量的非零计数，错误保存在写入器中
func (c *jxrDC) writeResidualChecked(b *jxrBitWriter, p *jxrPlane, values []int64) [2]int {
	var mean [2]int
	if len(values) < p.channels {
		b.err = fmt.Errorf("%w: JPEG XR DC output component storage", ErrInvalid)
		return mean
	}
	if c.model.bits[0] > 15 || c.model.bits[1] > 15 || c.luma.table > 1 || c.chroma.table > 1 {
		b.err = fmt.Errorf("%w: JPEG XR DC output model state", ErrInvalid)
		return mean
	}
	if err := checkJXRWords(values[:p.channels], true); err != nil {
		b.err = err
		return mean
	}
	if p.color == 0 || p.color == 4 || p.color == 6 {
		for index := range p.channels {
			if !b.noCancel && index&63 == 0 {
				if err := b.ctx.Err(); err != nil {
					b.err = err
					return mean
				}
			}
			group := 0
			if index != 0 {
				group = 1
			}
			present := jxrAbs(values[index])>>c.model.bits[group] != 0
			b.flag(present)
			if present {
				mean[group]++
			}
			writeJXRDCValue(b, &c.luma, c.model.bits[group], values[index])
		}
	} else {
		flags := uint8(0)
		for index := range 3 {
			group := 0
			if index != 0 {
				group = 1
			}
			if jxrAbs(values[index])>>c.model.bits[group] != 0 {
				flags |= 4 >> index
				mean[group]++
			}
		}
		writeJXRVLC(b, &jxrDCFlagsWriteCode, flags)
		for index := range 3 {
			group, level := 0, &c.luma
			if index != 0 {
				group, level = 1, &c.chroma
			}
			writeJXRDCValue(b, level, c.model.bits[group], values[index])
		}
	}
	return mean
}

// writeJXRDCValue 写入DC绝对值、细化位及符号位，零系数不写符号
// 入参: b 位写入器, level 绝对值编码状态, modelBits 细化位数, value DC残差
func writeJXRDCValue(b *jxrBitWriter, level *jxrPairVLC, modelBits uint8, value int64) {
	if b.err != nil {
		return
	}
	if modelBits > 15 {
		b.err = fmt.Errorf("%w: JPEG XR DC refinement width", ErrInvalid)
		return
	}
	if value < jxrLongWordMin || value > jxrLongWordMax {
		b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
		return
	}
	magnitude := jxrAbs(value)
	coarse := magnitude >> modelBits
	if coarse != 0 {
		level.writeAbsolute(b, coarse+1)
	}
	if modelBits != 0 {
		b.write(uint32(magnitude&(int64(1)<<modelBits-1)), uint(modelBits))
	}
	if magnitude != 0 {
		b.flag(value < 0)
	}
}

// writeAbsolute 写入标准绝对值前缀、逃逸长度及细化位，再累计编码表判别值
// 入参: b 位写入器, value 不小于2的绝对值
func (v *jxrPairVLC) writeAbsolute(b *jxrBitWriter, value int64) {
	if b.err != nil {
		return
	}
	if v.table > 1 || value < 2 {
		b.err = fmt.Errorf("%w: JPEG XR output absolute level", ErrInvalid)
		return
	}
	if value > 1<<30+1 {
		b.err = fmt.Errorf("%w: JPEG XR absolute level escape range", ErrUnsupported)
		return
	}
	index := uint8(6)
	for at := range 6 {
		if value >= jxrLevelBase[at] && value-jxrLevelBase[at] < int64(1)<<jxrLevelBits[at] {
			index = uint8(at)
			break
		}
	}
	writeJXRVLC(b, &jxrAbsoluteWriteCode[v.table], index)
	if index < 6 {
		b.write(uint32(value-jxrLevelBase[index]), uint(jxrLevelBits[index]))
	} else {
		width := uint(bits.Len64(uint64(value-2)) - 1)
		if width < 19 {
			b.write(uint32(width-4), 4)
		} else {
			b.write(15, 4)
			if width < 22 {
				b.write(uint32(width-19), 2)
			} else {
				b.write(3, 2)
				b.write(uint32(width-22), 3)
			}
		}
		b.write(uint32(value-2-(int64(1)<<width)), width)
	}
	if b.err == nil {
		if index == 0 {
			v.discriminant++
		} else if index > 1 {
			v.discriminant--
		}
	}
}

// writeJXRQPIndex 写入非零标志及量化索引细化字段，不生成保留索引值
// 入参: b 位写入器, count 量化集合数量，范围为2到16, index 集合索引
func writeJXRQPIndex(b *jxrBitWriter, count, index int) {
	if b.err != nil {
		return
	}
	if count < 2 || count > 16 || index < 0 || index >= count {
		b.err = fmt.Errorf("%w: JPEG XR output quantization index", ErrInvalid)
		return
	}
	b.flag(index != 0)
	if index != 0 {
		b.write(uint32(index-1), uint(jxrQPBits[count]))
	}
}

// newJXRLP 初始化分块低通系数的独立状态
// 返回: jxrLP 初始状态
func newJXRLP() jxrLP {
	return jxrLP{block: newJXRBlock(), model: newJXRModel(1), scan: newJXRScan(false), countZero: 1, countMax: 1}
}

// readChecked 在平面静态布局已校验前提下读取低通残差并更新模型
// 入参: b 读取器, p 已校验平面格式, reset, adapt 扫描及编码表控制, values 系数缓冲, flags 逐分量标志工作缓冲
func (c *jxrLP) readChecked(b *jxrBits, p *jxrPlane, reset, adapt bool, values [][16]int64, flags []bool) {
	if b.err != nil {
		return
	}
	mean := c.readResidualChecked(b, p, reset, values, flags)
	if b.err == nil {
		c.model.update(mean, p, 1)
		if adapt {
			c.block.adapt()
		}
	}
}

// readResidualChecked 在平面分量已校验前提下读取低通残差与细化语法
// 入参: b 读取器, p 已校验平面格式, reset 是否重置扫描累计, values 系数缓冲, flags 标志工作缓冲
// 返回: [2]int 亮度及其他分量的非零计数，错误保存在读取器中
func (c *jxrLP) readResidualChecked(b *jxrBits, p *jxrPlane, reset bool, values [][16]int64, flags []bool) [2]int {
	var mean [2]int
	if len(values) < p.channels || len(flags) < p.channels || c.model.bits[0] > 15 || c.model.bits[1] > 15 || c.countZero < -8 || c.countZero > 7 || c.countMax < -8 || c.countMax > 7 {
		b.err = fmt.Errorf("%w: JPEG XR lowpass input storage or state", ErrInvalid)
		return mean
	}
	clear(values[:p.channels])
	clear(flags[:p.channels])
	if reset {
		c.scan.reset()
	}
	subsampled := p.color == 1 || p.color == 2
	planes := p.channels
	if subsampled {
		planes = 2
	}
	pattern := uint32(0)
	if p.color >= 1 && p.color <= 3 {
		maximum := planes*4 - 5
		if c.countZero <= 0 || c.countMax < 0 {
			pattern = readJXRLPPattern(b, subsampled)
			if c.countMax < c.countZero {
				pattern = uint32(maximum) - pattern
			}
		} else {
			pattern = b.read(uint(planes))
		}
		c.countZero++
		if pattern == 0 {
			c.countZero -= 4
		}
		c.countMax++
		if pattern == uint32(maximum) {
			c.countMax -= 4
		}
		c.countZero = max(-8, min(7, c.countZero))
		c.countMax = max(-8, min(7, c.countMax))
		for n := range planes {
			flags[n] = pattern>>n&1 != 0
		}
	} else {
		for n := range p.channels {
			flags[n] = b.flag()
		}
	}
	for n := range planes {
		group := 0
		if n != 0 {
			group = 1
		}
		if flags[n] {
			start := 1
			if n != 0 && subsampled {
				start = 2
				if p.color == 1 {
					start = 10
				}
			}
			runs, count := c.block.read(b, n != 0, start)
			if b.err != nil {
				return mean
			}
			mean[group] += count
			position := 1
			if n != 0 && subsampled {
				position = 0
			}
			for _, run := range runs[:count] {
				position += int(run.run)
				if n != 0 && subsampled {
					offset, transpose := 0, 1
					if p.color == 1 {
						offset, transpose = 1, 2
					}
					at := jxrLPTranspose[transpose][jxrLPChromaRemap[(position>>1)+offset]]
					values[(position&1)+1][at] = run.value
				} else {
					values[n][c.scan.place(position)] = run.value
				}
				position++
			}
		}
		bits := c.model.bits[group]
		if bits != 0 {
			if n != 0 && subsampled {
				count, transpose := 8, 1
				if p.color == 1 {
					count, transpose = 4, 2
				}
				for k := 1; k < count; k++ {
					at := jxrLPTranspose[transpose][k]
					values[1][at] = refineJXRLP(b, values[1][at], bits)
					values[2][at] = refineJXRLP(b, values[2][at], bits)
				}
			} else {
				for k := 1; k < 16; k++ {
					at := jxrLPTranspose[0][k]
					values[n][at] = refineJXRLP(b, values[n][at], bits)
				}
			}
		}
		if b.err != nil {
			return mean
		}
	}
	return mean
}

// readJXRLPPattern 读取联合低通非零标志
// 入参: b 码流读取器, subsampled 是否采用抽样色度的二组编码
// 返回: uint32 联合标志，错误保存在读取器中
func readJXRLPPattern(b *jxrBits, subsampled bool) uint32 {
	if !b.flag() {
		return 0
	}
	if subsampled {
		if !b.flag() {
			return 1
		}
		return 2 + b.read(1)
	}
	prefix := b.read(2)
	if prefix == 0 {
		return 1
	}
	return prefix*2 + b.read(1)
}

// refineJXRLP 组合低通变长编码值及固定长度细化位，零值单独读取符号
// 入参: b 码流读取器, value 变长编码值, bits 细化位数
// 返回: int64 完整有符号系数，错误保存在读取器中
func refineJXRLP(b *jxrBits, value int64, bits uint8) int64 {
	ref := int64(b.read(uint(bits)))
	if value > 0 {
		return value<<bits + ref
	}
	if value < 0 {
		return value<<bits - ref
	}
	if ref != 0 && b.flag() {
		return -ref
	}
	return ref
}

// writeChecked 在平面静态布局已校验前提下写入低通残差并更新模型
// 入参: b 位写入器, p 已校验平面格式, values 低通残差, reset, adapt 扫描及编码表控制, flags 逐分量标志工作缓冲
func (c *jxrLP) writeChecked(b *jxrBitWriter, p *jxrPlane, values [][16]int64, reset, adapt bool, flags []bool) {
	if b.err != nil {
		return
	}
	mean := c.writeResidualChecked(b, p, values, reset, flags)
	if b.err == nil {
		c.model.update(mean, p, 1)
		if adapt {
			c.block.adapt()
		}
	}
}

// writeResidualChecked 在平面分量和Context已校验前提下写入低通残差语法
// 入参: b 写入器, p 已校验平面格式, values 逐分量残差, reset 是否重置扫描累计, flags 标志工作缓冲
// 返回: [2]int 亮度及其他分量非零数量，错误保存在写入器中
func (c *jxrLP) writeResidualChecked(b *jxrBitWriter, p *jxrPlane, values [][16]int64, reset bool, flags []bool) [2]int {
	var mean [2]int
	if len(values) < p.channels || len(flags) < p.channels {
		b.err = fmt.Errorf("%w: JPEG XR lowpass output component storage", ErrInvalid)
		return mean
	}
	if c.model.bits[0] > 15 || c.model.bits[1] > 15 || c.countZero < -8 || c.countZero > 7 || c.countMax < -8 || c.countMax > 7 {
		b.err = fmt.Errorf("%w: JPEG XR lowpass output model state", ErrInvalid)
		return mean
	}
	subsampled := p.color == 1 || p.color == 2
	pattern := uint32(0)
	clear(flags[:p.channels])
	for n := range p.channels {
		if !b.noCancel && n&63 == 0 {
			if err := b.ctx.Err(); err != nil {
				b.err = err
				return mean
			}
		}
		if err := checkJXRWords(values[n][:], true); err != nil {
			b.err = err
			return mean
		}
		group, count := 0, 16
		if n != 0 {
			group = 1
			if p.color == 1 {
				count = 4
			} else if p.color == 2 {
				count = 8
			}
		}
		if values[n][0] != 0 {
			b.err = fmt.Errorf("%w: JPEG XR lowpass output DC slot", ErrInvalid)
			return mean
		}
		for k := 1; k < 16; k++ {
			if k >= count && values[n][k] != 0 {
				b.err = fmt.Errorf("%w: JPEG XR lowpass output component layout", ErrInvalid)
				return mean
			}
			coarse := jxrAbs(values[n][k]) >> c.model.bits[group]
			if coarse > 1<<30+1 {
				b.err = fmt.Errorf("%w: JPEG XR lowpass absolute level escape range", ErrUnsupported)
				return mean
			}
			if coarse != 0 {
				at := n
				if subsampled && n != 0 {
					at = 1
				}
				flags[at] = true
			}
		}
	}
	if reset {
		c.scan.reset()
	}
	planes := p.channels
	if subsampled {
		planes = 2
	}
	if p.color >= 1 && p.color <= 3 {
		for n := range planes {
			if flags[n] {
				pattern |= 1 << n
			}
		}
		maximum := uint32(planes*4 - 5)
		if c.countZero <= 0 || c.countMax < 0 {
			encoded := pattern
			if c.countMax < c.countZero {
				encoded = maximum - pattern
			}
			table := 0
			if subsampled {
				table = 1
			}
			writeJXRVLC(b, &jxrLPPatternWriteCode[table], uint8(encoded))
		} else {
			b.write(pattern, uint(planes))
		}
		c.countZero++
		if pattern == 0 {
			c.countZero -= 4
		}
		c.countMax++
		if pattern == maximum {
			c.countMax -= 4
		}
		c.countZero = max(-8, min(7, c.countZero))
		c.countMax = max(-8, min(7, c.countMax))
	} else {
		for n := range p.channels {
			b.flag(flags[n])
		}
	}
	for n := range planes {
		group := 0
		if n != 0 {
			group = 1
		}
		modelBits := c.model.bits[group]
		var runs [15]jxrRunValue
		count, run, start := 0, uint8(0), 1
		if n != 0 && subsampled {
			limit, transpose, offset := 14, 1, 0
			start = 2
			if p.color == 1 {
				limit, transpose, offset, start = 6, 2, 1, 10
			}
			for k := range limit {
				at := jxrLPTranspose[transpose][jxrLPChromaRemap[(k>>1)+offset]]
				value := values[(k&1)+1][at]
				coarse := jxrAbs(value) >> modelBits
				if coarse != 0 {
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
		} else {
			runs, count = c.scan.gather(values[n][:], modelBits)
		}
		if count != 0 {
			c.block.write(b, n != 0, start, runs[:count])
		}
		mean[group] += count
		if modelBits != 0 {
			if n != 0 && subsampled {
				limit, transpose := 8, 1
				if p.color == 1 {
					limit, transpose = 4, 2
				}
				for k := 1; k < limit; k++ {
					at := jxrLPTranspose[transpose][k]
					writeJXRLPRefinement(b, values[1][at], modelBits)
					writeJXRLPRefinement(b, values[2][at], modelBits)
				}
			} else {
				for k := 1; k < 16; k++ {
					writeJXRLPRefinement(b, values[n][jxrLPTranspose[0][k]], modelBits)
				}
			}
		}
		if b.err != nil {
			return mean
		}
	}
	return mean
}

// writeJXRLPRefinement 写入低通系数的固定长度细化位，变长部分为零时另写符号
// 入参: b 位写入器, value 完整有符号系数, width 细化位数，范围为1到15
func writeJXRLPRefinement(b *jxrBitWriter, value int64, width uint8) {
	if b.err != nil {
		return
	}
	if width < 1 || width > 15 {
		b.err = fmt.Errorf("%w: JPEG XR lowpass output refinement width", ErrInvalid)
		return
	}
	if value < jxrLongWordMin || value > jxrLongWordMax {
		b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
		return
	}
	magnitude := jxrAbs(value)
	b.write(uint32(magnitude&(int64(1)<<width-1)), uint(width))
	if magnitude != 0 && magnitude>>width == 0 {
		b.flag(value < 0)
	}
}

// newJXRHP 初始化分块高通系数的独立状态
// 返回: jxrHP 初始状态
func newJXRHP() jxrHP {
	return jxrHP{block: newJXRBlock(), model: newJXRModel(2), scan: [2]jxrScan{newJXRScan(false), newJXRScan(true)}, cbp: newJXRCBP()}
}

// readChecked 在平面、块标志与系数缓冲布局已校验前提下读取宏块高通系数并更新模型
// 入参: b 码流读取器, p 已校验图像平面, pattern 已校验逐分量块标志, control 宏块读取控制, values 系数缓冲
// 返回: [2]uint8 亮度及其他分量的细化位数，错误保存在读取器中
func (c *jxrHP) readChecked(b *jxrBits, p *jxrPlane, pattern []uint16, control jxrHPControl, values []int64) [2]uint8 {
	bits := c.model.bits
	if b.err != nil {
		return bits
	}
	clear(values[:p.channels*256])
	if control.reset {
		c.scan[0].reset()
		c.scan[1].reset()
	}
	direction := 0
	if control.vertical {
		direction = 1
	}
	scan := &c.scan[direction]
	trim := control.trim
	if p.bands != 0 {
		trim = 15
	}
	var mean [2]int
	for i := range p.channels {
		group, blocks := 0, 16
		if i != 0 {
			group = 1
			if p.color == 1 {
				blocks = 4
			} else if p.color == 2 {
				blocks = 8
			}
		}
		width := bits[group]
		chBase := i * 256
		pat := pattern[i]
		shift := uint8(0)
		if control.spatial && width <= trim {
			shift = width
		}
		readFlex := control.spatial && width > trim
		remaining := uint(0)
		if readFlex {
			remaining = uint(width - trim)
		}
		for block := range blocks {
			mapped := block
			if blocks == 16 {
				mapped = int(jxrHPBlockOrder[block])
			}
			blockValues := values[chBase+mapped*16 : chBase+mapped*16+16]
			if pat>>block&1 != 0 {
				runs, count := c.block.read(b, i != 0, 1)
				if b.err != nil {
					return bits
				}
				mean[group] += count
				position := 1
				for _, run := range runs[:count] {
					position += int(run.run)
					at := scan.place(position)
					blockValues[at] = run.value << shift
					position++
				}
			}
			if readFlex {
				for n := 1; n < 16; n++ {
					at := jxrLPTranspose[0][n]
					value := blockValues[at]
					ref := int64(b.read(remaining))
					if value < 0 || value == 0 && ref != 0 && b.flag() {
						ref = -ref
					}
					blockValues[at] = value<<width + ref<<trim
				}
				if b.err != nil {
					return bits
				}
			}
		}
	}
	c.model.update(mean, p, 2)
	if control.adapt {
		c.block.adapt()
		c.cbp.adapt()
	}
	return bits
}

// refineJXRHigh 读取一个宏块的精细位并原地组合高通系数，缺失精细位时仅保留变长部分
// 入参: b 精细位读取器, p 图像平面, bits 当前宏块的双组细化位数, trim 包截断量，缺失精细位时为15, values 分量步长为256的高通系数
// 返回: error 数据或缓冲错误
func refineJXRHigh(b *jxrBits, p *jxrPlane, bits [2]uint8, trim uint8, values []int64) error {
	if len(values) < p.channels*256 {
		return fmt.Errorf("%w: JPEG XR flex coefficient buffer", ErrInvalid)
	}
	for i := range p.channels {
		group, blocks := 0, 16
		if i != 0 {
			group = 1
			if p.color == 1 {
				blocks = 4
			} else if p.color == 2 {
				blocks = 8
			}
		}
		width := bits[group]
		if width == 0 {
			continue
		}
		chBase := i * 256
		if width <= trim {
			for block := range blocks {
				mapped := block
				if blocks == 16 {
					mapped = int(jxrHPBlockOrder[block])
				}
				blockValues := values[chBase+mapped*16 : chBase+mapped*16+16]
				for n := 1; n < 16; n++ {
					blockValues[n] <<= width
				}
			}
			continue
		}
		remaining := uint(width - trim)
		for block := range blocks {
			mapped := block
			if blocks == 16 {
				mapped = int(jxrHPBlockOrder[block])
			}
			blockValues := values[chBase+mapped*16 : chBase+mapped*16+16]
			for n := 1; n < 16; n++ {
				at := jxrLPTranspose[0][n]
				value := blockValues[at]
				ref := int64(b.read(remaining))
				if value < 0 || value == 0 && ref != 0 && b.flag() {
					ref = -ref
				}
				blockValues[at] = value<<width + ref<<trim
			}
			if b.err != nil {
				return b.err
			}
		}
	}
	return b.err
}

// writeChecked 在平面、模型与刚生成的非零块标志已校验前提下写入宏块高通残差，跳过重复全量重扫
// 入参: b 位写入器, p 已校验平面格式, pattern 已校验逐分量非零块标志, control 扫描及空间模式控制, values 完整高通残差, exact 是否拒绝细化位截断造成的精度损失
// 返回: [2]uint8 当前宏块的亮度及其他分量细化位数，错误保存在写入器中
func (c *jxrHP) writeChecked(b *jxrBitWriter, p *jxrPlane, pattern []uint16, control jxrHPControl, values []int64, exact bool) [2]uint8 {
	bits := c.model.bits
	if b.err != nil {
		return bits
	}
	trim := control.trim
	if p.bands != 0 {
		trim = 15
	}
	if exact && trim != 0 && (control.spatial || p.bands != 0) && (bits[0] != 0 || bits[1] != 0) {
		for n := range p.channels {
			group := 0
			if n != 0 {
				group = 1
			}
			mask := int64(1)<<min(bits[group], trim) - 1
			if mask == 0 {
				continue
			}
			for _, value := range values[n*256 : (n+1)*256] {
				if value&mask != 0 {
					b.err = fmt.Errorf("%w: JPEG XR omitted highpass refinement precision", ErrUnsupported)
					return bits
				}
			}
		}
	}
	if control.reset {
		c.scan[0].reset()
		c.scan[1].reset()
	}
	direction := 0
	if control.vertical {
		direction = 1
	}
	scan := &c.scan[direction]
	var mean [2]int
	for n := range p.channels {
		group, blocks := 0, 16
		if n != 0 {
			group = 1
			if p.color == 1 {
				blocks = 4
			} else if p.color == 2 {
				blocks = 8
			}
		}
		width := bits[group]
		chBase := n * 256
		pat := pattern[n]
		writeFlex := control.spatial && width > trim
		remaining := uint(0)
		refMask := int64(0)
		if writeFlex {
			remaining = uint(width - trim)
			refMask = int64(1)<<width - 1
		}
		for block := range blocks {
			mapped := block
			if blocks == 16 {
				mapped = int(jxrHPBlockOrder[block])
			}
			coefficients := values[chBase+mapped*16 : chBase+mapped*16+16]
			if pat>>block&1 != 0 {
				runs, count := scan.gather(coefficients, width)
				c.block.write(b, n != 0, 1, runs[:count])
				mean[group] += count
			}
			if writeFlex {
				for k := 1; k < 16; k++ {
					value := coefficients[jxrLPTranspose[0][k]]
					magnitude := value
					if magnitude < 0 {
						magnitude = -magnitude
					}
					ref := (magnitude & refMask) >> trim
					b.write(uint32(ref), remaining)
					if magnitude>>width == 0 && ref != 0 {
						b.flag(value < 0)
					}
				}
			}
			if b.err != nil {
				return bits
			}
		}
	}
	c.model.update(mean, p, 2)
	if control.adapt {
		c.block.adapt()
		c.cbp.adapt()
	}
	return bits
}

// makeJXRHighPattern 按当前细化位数生成层次扫描顺序的非零块标志，并校验系数布局与数值范围
// 入参: p 平面格式, values 完整高通残差, widths 亮度及其他分量细化位数, pattern 调用方提供的逐分量标志缓冲
// 返回: error 布局、数值、缓冲或逃逸范围错误，仅修改有效分量的输出
func makeJXRHighPattern(p *jxrPlane, values []int64, widths [2]uint8, pattern []uint16) error {
	if err := checkJXRPlaneComponents(p); err != nil {
		return err
	}
	if p.channels > len(pattern) {
		return fmt.Errorf("%w: JPEG XR highpass pattern component storage", ErrInvalid)
	}
	if widths[0] > 15 || widths[1] > 15 || len(values) < p.channels*256 {
		return fmt.Errorf("%w: JPEG XR highpass output coefficient layout", ErrInvalid)
	}
	if err := checkJXRWords(values[:p.channels*256], true); err != nil {
		return err
	}
	clear(pattern[:p.channels])
	for n := range p.channels {
		group, blocks := 0, 16
		if n != 0 {
			group = 1
			if p.color == 1 {
				blocks = 4
			} else if p.color == 2 {
				blocks = 8
			}
		}
		width := widths[group]
		threshold := int64(1) << width
		chValues := values[n*256 : (n+1)*256]
		var pat uint16
		for block := range 16 {
			mapped := block
			if blocks == 16 {
				mapped = int(jxrHPBlockOrder[block])
			}
			blockValues := chValues[mapped*16 : mapped*16+16]
			if block >= blocks {
				for _, value := range blockValues {
					if value != 0 {
						return fmt.Errorf("%w: JPEG XR highpass output reserved coefficient", ErrInvalid)
					}
				}
				continue
			}
			if blockValues[0] != 0 {
				return fmt.Errorf("%w: JPEG XR highpass output reserved coefficient", ErrInvalid)
			}
			if width == 0 {
				for k := 1; k < 16; k++ {
					value := blockValues[k]
					if value == 0 {
						continue
					}
					magnitude := value
					if magnitude < 0 {
						magnitude = -magnitude
					}
					if magnitude > 1<<30+1 {
						return fmt.Errorf("%w: JPEG XR highpass absolute level escape range", ErrUnsupported)
					}
					pat |= 1 << block
				}
				continue
			}
			for k := 1; k < 16; k++ {
				value := blockValues[k]
				if value >= threshold || value <= -threshold {
					pat |= 1 << block
					break
				}
			}
		}
		pattern[n] = pat
	}
	return nil
}

// newJXRCBP 初始化高通块标志的编码及预测计数
// 返回: jxrCBP 初始状态
func newJXRCBP() jxrCBP {
	return jxrCBP{ones: [2]int{-4, -4}, zeroes: [2]int{4, 4}}
}

// readChecked 在平面、存储与编码表状态已校验前提下读取高通块标志残差
// 入参: b 码流读取器, p 已校验图像平面, values 逐分量块标志残差缓冲
func (c *jxrCBP) readChecked(b *jxrBits, p *jxrPlane, values []uint16) {
	if b.err != nil {
		return
	}
	clear(values[:p.channels])
	components := 1
	if p.color == 4 || p.color == 6 {
		components = p.channels
	}
	combined := p.color >= 1 && p.color <= 3
	for i := range components {
		count := readJXRVLC(b, &jxrCBPGroupCode[c.group.table])
		c.group.discriminant += jxrCBPGroupDelta[count]
		groups := refineJXRCBP(b, count)
		for block := range 4 {
			if groups>>block&1 == 0 {
				continue
			}
			table := &jxrCBPGroupCode[c.block.table]
			if combined {
				table = &jxrCBPChromaCode[c.block.table]
			}
			count = readJXRVLC(b, table)
			if combined {
				c.block.discriminant += jxrCBPChromaDelta[count]
			} else {
				c.block.discriminant += jxrCBPGroupDelta[count]
			}
			value := count + 1
			pattern := uint8(0)
			if value >= 6 {
				pattern = (readJXRThird(b) + 1) << 4
				if value >= 9 {
					value += readJXRThird(b)
				}
				value -= 6
			}
			index := jxrCBPRefineOffset[value] + uint8(b.read(uint(jxrCBPRefineBits[value])))
			pattern |= jxrCBPRefineValue[index]
			switch p.color {
			case 3:
				values[0] |= uint16(pattern&15) << (block * 4)
				for k := range 2 {
					if pattern>>(k+4)&1 != 0 {
						count := uint8(0)
						if !b.flag() {
							count = 1
							if !b.flag() {
								count = 2 + uint8(b.read(1))
							}
						}
						values[k+1] |= uint16(refineJXRCBP(b, count+1)) << (block * 4)
					}
				}
			case 2:
				values[0] |= uint16(pattern&15) << (block * 4)
				for k := range 2 {
					if pattern>>(k+4)&1 != 0 {
						mask := jxrCBP422Shift[readJXRThird(b)+1]
						values[k+1] |= uint16(mask) << jxrCBP422Shift[block]
					}
				}
			case 1:
				values[0] |= uint16(pattern&15) << (block * 4)
				values[1] |= uint16(pattern>>4&1) << block
				values[2] |= uint16(pattern>>5&1) << block
			default:
				values[i] |= uint16(pattern) << (block * 4)
			}
			if b.err != nil {
				return
			}
		}
	}
}

// predictChecked 在平面、相邻缓冲与预测状态已校验前提下重建高通块标志并更新模型
// 入参: p 已校验图像平面, residual 残差, left, top 相邻标志, firstColumn, firstRow 分块边缘, values 输出切片
func (c *jxrCBP) predictChecked(p *jxrPlane, residual, left, top []uint16, firstColumn, firstRow bool, values []uint16) {
	for i := range p.channels {
		group, count, leftBit, topBit := 0, 16, uint(5), uint(10)
		if i != 0 {
			group = 1
			if p.color == 2 {
				count, leftBit, topBit = 8, 1, 6
			} else if p.color == 1 {
				count, leftBit, topBit = 4, 1, 2
			}
		}
		value := residual[i]
		if c.state[group] == 0 {
			if firstColumn {
				if firstRow {
					value ^= 1
				} else {
					value ^= top[i] >> topBit & 1
				}
			} else {
				value ^= left[i] >> leftBit & 1
			}
			switch count {
			case 16:
				value ^= 2 & (value << 1)
				value ^= 16 & (value << 3)
				value ^= 32 & (value << 1)
				value ^= (value & 0x33) << 2
				value ^= (value & 0xcc) << 6
				value ^= (value & 0x3300) << 2
			case 8:
				value ^= (value & 1) << 1
				value ^= (value & 3) << 2
				value ^= (value & 12) << 2
				value ^= (value & 48) << 2
			case 4:
				value ^= 2 & (value << 1)
				value ^= (value & 3) << 2
			}
		} else if c.state[group] == 2 {
			value ^= uint16(1<<count - 1)
		}
		values[i] = value
		c.update(group, value, count)
	}
}

// update 按重建块标志的非零数量更新预测模型，抽样分量归一化到十六块
// 入参: group 亮度或其他分量组, value 重建块标志, count 分量的实际块数
func (c *jxrCBP) update(group int, value uint16, count int) {
	ones := bits.OnesCount16(value) * (16 / count)
	c.ones[group] = min(15, max(-16, c.ones[group]+ones-3))
	c.zeroes[group] = min(15, max(-16, c.zeroes[group]+16-ones-3))
	c.state[group] = 0
	if c.ones[group] < 0 {
		c.state[group] = 2
		if c.ones[group] < c.zeroes[group] {
			c.state[group] = 1
		}
	} else if c.zeroes[group] < 0 {
		c.state[group] = 2
	}
}

// adapt 更新高通块标志的双编码表
func (c *jxrCBP) adapt() {
	c.group.adapt()
	c.block.adapt()
}

// refineJXRCBP 按非零块数量读取四块组的位置标志
// 入参: b 码流读取器, count 非零块数量
// 返回: uint8 四块位置标志，错误保存在读取器中
func refineJXRCBP(b *jxrBits, count uint8) uint8 {
	switch count {
	case 1:
		return 1 << b.read(2)
	case 2:
		prefix := b.read(2)
		if prefix < 2 {
			return 3 + uint8(prefix)*2
		}
		return [4]uint8{6, 9, 10, 12}[(prefix-2)*2+b.read(1)]
	case 3:
		return 15 ^ (1 << b.read(2))
	case 4:
		return 15
	default:
		return 0
	}
}

// readJXRThird 读取三符号前缀码
// 入参: b 码流读取器
// 返回: uint8 零至二的符号，错误保存在读取器中
func readJXRThird(b *jxrBits) uint8 {
	if b.flag() {
		return 0
	}
	return 2 - uint8(b.read(1))
}

// writeChecked 在平面、残差与编码表状态已校验前提下写入高通块标志残差
// 入参: b 位写入器, p 已校验平面格式, values 已校验逐分量块标志残差
func (c *jxrCBP) writeChecked(b *jxrBitWriter, p *jxrPlane, values []uint16) {
	if b.err != nil {
		return
	}
	components := 1
	if p.color == 4 || p.color == 6 {
		components = p.channels
	}
	combined := p.color >= 1 && p.color <= 3
	for n := range components {
		if !b.noCancel && n&63 == 0 {
			if err := b.ctx.Err(); err != nil {
				b.err = err
				return
			}
		}
		var layout [4][3]uint8
		groups := uint8(0)
		for block := range 4 {
			layout[block][0] = uint8(values[n] >> (block * 4) & 15)
			if combined {
				for k := range 2 {
					switch p.color {
					case 1:
						layout[block][k+1] = uint8(values[k+1] >> block & 1)
					case 2:
						layout[block][k+1] = uint8(values[k+1] >> jxrCBP422Shift[block] & 5)
					case 3:
						layout[block][k+1] = uint8(values[k+1] >> (block * 4) & 15)
					}
				}
			}
			if layout[block] != [3]uint8{} {
				groups |= 1 << block
			}
		}
		count := uint8(bits.OnesCount8(groups))
		writeJXRVLC(b, &jxrCBPGroupWriteCode[c.group.table], count)
		if b.err == nil {
			c.group.discriminant += jxrCBPGroupDelta[count]
		}
		writeJXRCBPRefinement(b, groups)
		for block := range 4 {
			if groups>>block&1 != 0 {
				c.writeGroup(b, p.color, layout[block])
			}
		}
		if b.err != nil {
			return
		}
	}
}

// writeGroup 写入单个四块组的亮度掩码及存在的色度细化字段
// 入参: b 位写入器, color 内部颜色格式, pattern 亮度或独立分量、U及V的组掩码
func (c *jxrCBP) writeGroup(b *jxrBitWriter, color uint8, pattern [3]uint8) {
	if b.err != nil {
		return
	}
	class := jxrCBPNibbleClass[pattern[0]]
	chroma := uint8(0)
	if pattern[1] != 0 {
		chroma |= 1
	}
	if pattern[2] != 0 {
		chroma |= 2
	}
	symbol := class - 1
	if chroma != 0 {
		symbol = min(class, 3) + 5
	}
	combined := color >= 1 && color <= 3
	if combined {
		writeJXRVLC(b, &jxrCBPChromaWriteCode[c.block.table], symbol)
		if b.err == nil {
			c.block.discriminant += jxrCBPChromaDelta[symbol]
		}
	} else {
		writeJXRVLC(b, &jxrCBPGroupWriteCode[c.block.table], symbol)
		if b.err == nil {
			c.block.discriminant += jxrCBPGroupDelta[symbol]
		}
	}
	if chroma != 0 {
		writeJXRVLC(b, &jxrThirdWriteCode, chroma-1)
		if class >= 3 {
			writeJXRVLC(b, &jxrThirdWriteCode, class-3)
		}
	}
	b.write(uint32(jxrCBPNibbleRef[pattern[0]]), uint(jxrCBPRefineBits[class]))
	for k := range 2 {
		if pattern[k+1] == 0 {
			continue
		}
		if color == 3 {
			count := uint8(bits.OnesCount8(pattern[k+1]))
			writeJXRVLC(b, &jxrChromaCountWriteCode, count-1)
			writeJXRCBPRefinement(b, pattern[k+1])
		} else if color == 2 {
			symbol := uint8(0)
			if pattern[k+1] == 4 {
				symbol = 1
			} else if pattern[k+1] == 5 {
				symbol = 2
			}
			writeJXRVLC(b, &jxrThirdWriteCode, symbol)
		}
	}
}

// residualChecked 在平面、相邻缓冲与预测状态已校验前提下计算高通块标志残差并更新模型
// 入参: p 已校验平面格式, values 原始标志, left, top 相邻标志, firstColumn, firstRow 分块边缘, residual 输出切片
func (c *jxrCBP) residualChecked(p *jxrPlane, values, left, top []uint16, firstColumn, firstRow bool, residual []uint16) {
	for n := range p.channels {
		group, count, leftBit, topBit := 0, 16, uint(5), uint(10)
		if n != 0 {
			group = 1
			if p.color == 2 {
				count, leftBit, topBit = 8, 1, 6
			} else if p.color == 1 {
				count, leftBit, topBit = 4, 1, 2
			}
		}
		value := values[n]
		if c.state[group] == 0 {
			switch count {
			case 16:
				value ^= (value & 0x3300) << 2
				value ^= (value & 0xcc) << 6
				value ^= (value & 0x33) << 2
				value ^= 32 & (value << 1)
				value ^= 16 & (value << 3)
				value ^= 2 & (value << 1)
			case 8:
				value ^= (value & 48) << 2
				value ^= (value & 12) << 2
				value ^= (value & 3) << 2
				value ^= (value & 1) << 1
			case 4:
				value ^= (value & 3) << 2
				value ^= 2 & (value << 1)
			}
			if firstColumn {
				if firstRow {
					value ^= 1
				} else {
					value ^= top[n] >> topBit & 1
				}
			} else {
				value ^= left[n] >> leftBit & 1
			}
		} else if c.state[group] == 2 {
			value ^= uint16(1<<count - 1)
		}
		c.update(group, values[n], count)
		residual[n] = value
	}
}

// writeJXRCBPRefinement 按非零块数量写入四块组的实际位置
// 入参: b 位写入器, pattern 四块位置掩码
func writeJXRCBPRefinement(b *jxrBitWriter, pattern uint8) {
	if b.err != nil {
		return
	}
	if pattern > 15 {
		b.err = fmt.Errorf("%w: JPEG XR output CBP refinement pattern", ErrInvalid)
		return
	}
	switch bits.OnesCount8(pattern) {
	case 1:
		b.write(uint32(bits.TrailingZeros8(pattern)), 2)
	case 2:
		writeJXRVLC(b, &jxrCBPRefineWriteCode, pattern)
	case 3:
		b.write(uint32(bits.TrailingZeros8(15^pattern)), 2)
	}
}

// newJXRBlock 初始化一个频带的亮色及自适应编码表
// 返回: jxrBlock 初始状态
func newJXRBlock() jxrBlock {
	var c jxrBlock
	for i := range 2 {
		c.first[i] = newJXRMultiVLC()
		for j := range 2 {
			c.index[i][j] = newJXRMultiVLC()
		}
	}
	return c
}

// read 读取不超过十五个非零系数的游程编码，非法位置不用于访问系数数组
// 入参: b 码流读取器, chroma 是否使用色度编码状态, start 起始扫描位置
// 返回: [15]jxrRunValue 游程与系数, int 非零数量，错误保存在读取器中
func (c *jxrBlock) read(b *jxrBits, chroma bool, start int) ([15]jxrRunValue, int) {
	var values [15]jxrRunValue
	if start < 1 || start > 15 {
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR block scan start", ErrInvalid)
		}
		return values, 0
	}
	group := 0
	if chroma {
		group = 1
	}
	first := c.first[group].first(b)
	sign := b.flag()
	runZero, next := first&1, first>>2
	context := runZero & next
	value := int64(1)
	if first&2 != 0 {
		value = c.level[context].absolute(b)
	}
	if sign {
		value = -value
	}
	run := uint8(0)
	if runZero == 0 {
		run = readJXRRun(b, 15-start)
	}
	values[0] = jxrRunValue{run, value}
	indexTables := &c.index[group]
	location, count := start+int(run)+1, 1
	for b.err == nil && next != 0 {
		if location >= 16 || count >= len(values) {
			b.err = fmt.Errorf("%w: JPEG XR block coefficient count", ErrInvalid)
			break
		}
		run = 0
		if next&1 == 0 {
			run = readJXRRun(b, 15-location)
		}
		location += int(run) + 1
		index := uint8(0)
		if location < 15 {
			index = indexTables[context].index(b)
		} else if location == 15 {
			if b.flag() {
				index = 2
				if b.flag() {
					index = 1
					if b.flag() {
						index = 3
					}
				}
			}
		} else if location == 16 {
			if b.flag() {
				index = 1
			}
		} else if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR block coefficient position", ErrInvalid)
		}
		next = index >> 1
		context &= next
		sign = b.flag()
		value = 1
		if index&1 != 0 {
			value = c.level[context].absolute(b)
		}
		if sign {
			value = -value
		}
		values[count] = jxrRunValue{run, value}
		count++
	}
	return values, count
}

// adapt 更新一个频带的全部编码表，亮度和色度状态保持独立
func (c *jxrBlock) adapt() {
	c.first[0].adapt(4)
	c.first[1].adapt(4)
	c.index[0][0].adapt(3)
	c.index[0][1].adapt(3)
	c.index[1][0].adapt(3)
	c.index[1][1].adapt(3)
	c.level[0].adapt()
	c.level[1].adapt()
}

// readJXRRun 读取非零游程并校验范围
// 入参: b 码流读取器, maximum 当前扫描位置允许的最大游程
// 返回: uint8 游程，错误保存在读取器中
func readJXRRun(b *jxrBits, maximum int) uint8 {
	if maximum < 1 || maximum > 14 {
		if b.err == nil {
			b.err = fmt.Errorf("%w: JPEG XR run range", ErrInvalid)
		}
		return 0
	}
	run := uint8(1)
	if maximum < 5 {
		for int(run) < maximum && !b.flag() {
			run++
		}
	} else {
		index := 0
		for index < 3 && !b.flag() {
			index++
		}
		if index == 3 && b.flag() {
			index = 4
		}
		index += 5 * int(jxrRunBin[maximum])
		run = jxrRunBase[index]
		if bits := jxrRunBits[index]; bits != 0 {
			run += uint8(b.read(uint(bits)))
		}
	}
	if int(run) > maximum && b.err == nil {
		b.err = fmt.Errorf("%w: JPEG XR run exceeds block", ErrInvalid)
	}
	return run
}

// jxrMakeRunWriteTable 预计算全部有效游程的前缀与细化位合并编码
// 返回: [15][15]jxrWriteCode 按最大游程与游程索引的编码表
func jxrMakeRunWriteTable() [15][15]jxrWriteCode {
	var table [15][15]jxrWriteCode
	for maximum := 1; maximum <= 14; maximum++ {
		for run := uint8(1); int(run) <= maximum; run++ {
			if maximum < 5 {
				width := run
				value := uint8(1)
				if int(run) == maximum {
					width--
					value = 0
				}
				table[maximum][run] = jxrWriteCode{value: value, bits: width}
				continue
			}
			base := 5 * int(jxrRunBin[maximum])
			for index := range 5 {
				at := base + index
				if run >= jxrRunBase[at] && int(run-jxrRunBase[at]) < 1<<jxrRunBits[at] {
					prefix := jxrRunWriteCode[index]
					extraBits := jxrRunBits[at]
					extraValue := run - jxrRunBase[at]
					table[maximum][run] = jxrWriteCode{value: prefix.value<<extraBits | extraValue, bits: prefix.bits + extraBits}
					break
				}
			}
		}
	}
	return table
}

// write 写入非零系数的游程、符号及幅值，并累计自适应编码表状态
// 入参: b 位写入器, chroma 是否使用色度状态, start 起始扫描位置, values 非空游程系数序列
func (c *jxrBlock) write(b *jxrBitWriter, chroma bool, start int, values []jxrRunValue) {
	if b.err != nil {
		return
	}
	if !b.noCancel {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
	}
	if start < 1 || start > 15 || len(values) < 1 || len(values) > 16-start {
		b.err = fmt.Errorf("%w: JPEG XR output block size", ErrInvalid)
		return
	}
	location := start
	for _, value := range values {
		location += int(value.run) + 1
		if location > 16 || value.value == 0 {
			b.err = fmt.Errorf("%w: JPEG XR output block coefficient position", ErrInvalid)
			return
		}
		if value.value < jxrLongWordMin || value.value > jxrLongWordMax {
			b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
			return
		}
		magnitude := value.value
		if magnitude < 0 {
			magnitude = -magnitude
		}
		if magnitude > 1<<30+1 {
			b.err = fmt.Errorf("%w: JPEG XR block absolute level escape range", ErrUnsupported)
			return
		}
	}
	group := 0
	if chroma {
		group = 1
	}
	indexTables := &c.index[group]
	location, context := start, uint8(0)
	for index, value := range values {
		next := uint8(0)
		if index+1 < len(values) {
			next = 1
			if values[index+1].run != 0 {
				next = 2
			}
		}
		magnitude := value.value
		if magnitude < 0 {
			magnitude = -magnitude
		}
		if index == 0 {
			symbol := next << 2
			if value.run == 0 {
				symbol |= 1
				context = next & 1
			}
			if magnitude > 1 {
				symbol |= 2
			}
			c.first[group].writeFirst(b, symbol)
			b.flag(value.value < 0)
			if magnitude > 1 {
				c.level[context].writeAbsolute(b, magnitude)
			}
			if value.run != 0 {
				writeJXRRun(b, 15-location, value.run)
			}
			location += int(value.run) + 1
		} else {
			if value.run != 0 {
				writeJXRRun(b, 15-location, value.run)
			}
			location += int(value.run) + 1
			symbol := next << 1
			if magnitude > 1 {
				symbol |= 1
			}
			if location < 15 {
				indexTables[context].writeIndex(b, symbol)
			} else if location == 15 {
				writeJXRVLC(b, &jxrLastIndexWriteCode, symbol)
			} else {
				b.flag(magnitude > 1)
			}
			context &= next
			b.flag(value.value < 0)
			if magnitude > 1 {
				c.level[context].writeAbsolute(b, magnitude)
			}
		}
		if b.err != nil {
			return
		}
	}
}

// writeJXRRun 写入非零游程的前缀及细化位，单一可选游程不占码位
// 入参: b 位写入器, maximum 当前位置允许的最大游程, run 非零游程
func writeJXRRun(b *jxrBitWriter, maximum int, run uint8) {
	if b.err != nil {
		return
	}
	if maximum < 1 || maximum > 14 || run == 0 || int(run) > maximum {
		b.err = fmt.Errorf("%w: JPEG XR output run range", ErrInvalid)
		return
	}
	if code := jxrRunCombinedCode[maximum][run]; code.bits != 0 {
		b.write(uint32(code.value), uint(code.bits))
	}
}

// writeJXRHighFlexChecked 在平面与系数缓冲静态布局已校验前提下写入宏块精细位，保留全部字长与精度检查
// 入参: b 位写入器, p 已校验平面格式, bits 当前宏块的双组细化位数, trim 包截断量, values 完整高通残差, exact 是否要求精确恢复
func writeJXRHighFlexChecked(b *jxrBitWriter, p *jxrPlane, bits [2]uint8, trim uint8, values []int64, exact bool) {
	if b.err != nil {
		return
	}
	for n := range p.channels {
		group, blocks := 0, 16
		if n != 0 {
			group = 1
			if p.color == 1 {
				blocks = 4
			} else if p.color == 2 {
				blocks = 8
			}
		}
		width := bits[group]
		chBase := n * 256
		if width <= trim {
			truncMask := int64(0)
			if exact && width != 0 {
				truncMask = int64(1)<<width - 1
			}
			for block := range blocks {
				mapped := block
				if blocks == 16 {
					mapped = int(jxrHPBlockOrder[block])
				}
				blockValues := values[chBase+mapped*16 : chBase+mapped*16+16]
				if truncMask == 0 {
					for k := 1; k < 16; k++ {
						value := blockValues[k]
						if value < jxrLongWordMin || value > jxrLongWordMax {
							b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
							return
						}
					}
					continue
				}
				for k := 1; k < 16; k++ {
					value := blockValues[jxrLPTranspose[0][k]]
					if value < jxrLongWordMin || value > jxrLongWordMax {
						b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
						return
					}
					if value&truncMask != 0 {
						b.err = fmt.Errorf("%w: JPEG XR refinement truncation precision", ErrUnsupported)
						return
					}
				}
			}
			continue
		}
		remaining := uint(width - trim)
		refMask := int64(1)<<width - 1
		truncMask := int64(0)
		if exact && trim != 0 {
			truncMask = int64(1)<<trim - 1
		}
		for block := range blocks {
			mapped := block
			if blocks == 16 {
				mapped = int(jxrHPBlockOrder[block])
			}
			blockValues := values[chBase+mapped*16 : chBase+mapped*16+16]
			for k := 1; k < 16; k++ {
				value := blockValues[jxrLPTranspose[0][k]]
				if value < jxrLongWordMin || value > jxrLongWordMax {
					b.err = fmt.Errorf("%w: JPEG XR 32-bit coefficient range", ErrInvalid)
					return
				}
				if truncMask != 0 && value&truncMask != 0 {
					b.err = fmt.Errorf("%w: JPEG XR refinement truncation precision", ErrUnsupported)
					return
				}
				magnitude := value
				if magnitude < 0 {
					magnitude = -magnitude
				}
				ref := (magnitude & refMask) >> trim
				b.write(uint32(ref), remaining)
				if magnitude>>width == 0 && ref != 0 {
					b.flag(value < 0)
				}
			}
			if b.err != nil {
				return
			}
		}
	}
}
