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
	"sync/atomic"
)

// jxrSafeShortLimit 与 jxrSafeLongLimit 限制十六位和三十二位零溢出快路径的安全输入上界
const (
	jxrSafeShortLimit = uint64(1 << 10)
	jxrSafeLongLimit  = uint64(1 << 26)
)

// jxrFirstCode 保存五组首系数前缀码
var jxrFirstCode = [5][256]jxrCode{
	jxrMakeVLC([]string{"00001", "000001", "0000000", "0000001", "00100", "010", "00101", "1", "00110", "0001", "00111", "011"}),
	jxrMakeVLC([]string{"0010", "00010", "000000", "000001", "0011", "010", "00011", "11", "011", "100", "00001", "101"}),
	jxrMakeVLC([]string{"11", "001", "0000000", "0000001", "00001", "010", "0000010", "011", "100", "101", "0000011", "0001"}),
	jxrMakeVLC([]string{"001", "11", "0000000", "00001", "00010", "010", "0000001", "011", "00011", "100", "000001", "101"}),
	jxrMakeVLC([]string{"010", "1", "0000001", "0001", "0000010", "011", "00000000", "0010", "0000011", "0011", "00000001", "00001"}),
}

// jxrIndexCode 保存四组后续系数前缀码
var jxrIndexCode = [4][256]jxrCode{
	jxrMakeVLC([]string{"1", "00000", "001", "00001", "01", "0001"}),
	jxrMakeVLC([]string{"01", "0000", "10", "0001", "11", "001"}),
	jxrMakeVLC([]string{"0000", "0001", "01", "10", "11", "001"}),
	jxrMakeVLC([]string{"00000", "00001", "01", "1", "0001", "001"}),
}

// jxrFirstDelta 保存首系数编码表选择的四组判别增量
var jxrFirstDelta = [4][12]int{
	{1, 1, 1, 1, 1, 0, 0, -1, 2, 1, 0, 0},
	{2, 2, -1, -1, -1, 0, -2, -1, 0, 0, -2, -1},
	{-1, 1, 0, 2, 0, 0, 0, 0, -2, 0, 1, 1},
	{0, 1, 0, 1, -2, 0, -1, -1, -2, -1, -2, -2},
}

// jxrIndexDelta 保存后续系数编码表选择的三组判别增量
var jxrIndexDelta = [3][6]int{{-1, 1, 1, 1, 0, 1}, {-2, 0, 0, 2, 0, 0}, {-1, -1, 0, 1, -2, 0}}

// jxrModelLumaWeight 保存DC、低通及高通频带的亮度归一化权重
var jxrModelLumaWeight = [3]int{240, 12, 1}

// jxrModelComponentWeight 保存非抽样格式的其他分量归一化权重
var jxrModelComponentWeight = [3][16]int{
	{0, 240, 120, 80, 60, 48, 40, 34, 30, 27, 24, 22, 20, 18, 17, 16},
	{0, 12, 6, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1},
	{0, 16, 8, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1},
}

// jxrModelChromaWeight 保存4:2:0及4:2:2格式的色度归一化权重
var jxrModelChromaWeight = [2][3]int{{120, 37, 2}, {120, 18, 1}}

// jxrBits 保存高位优先的JPEG XR读取位置及首次错误，后续读取不越过输入
type jxrBits struct {
	data   []byte
	bit    uint64
	end    uint64
	err    error
	parser *jxrParser
	input  *jxrReaderAt
}

// jxrSharedWriteBudget 在并发分块编码期间以原子操作共享并约束全部输出缓冲的瞬时总容量
type jxrSharedWriteBudget struct {
	limit    uint64
	used     atomic.Uint64
	exceeded atomic.Bool
}

// jxrWriteBudget 记录同次编码各输出缓冲共享的容量上限及已占容量
type jxrWriteBudget struct {
	limit  uint64
	used   uint64
	shared *jxrSharedWriteBudget
}

// jxrBitWriter 保存高位优先的编码缓冲、位数、容量限制及首次错误
type jxrBitWriter struct {
	ctx              context.Context
	data             []byte
	bit              uint64
	maxBytes         uint64
	maxMemory        uint64
	budget           *jxrWriteBudget
	err              error
	checks           uint32
	noCancel         bool
	maxBytesExceeded bool
}

// jxrCode 保存前缀查找表中的符号及实际编码位数
type jxrCode struct {
	value uint8
	bits  uint8
}

// jxrMultiVLC 保存多编码表选择的索引、判别值及相邻增量表索引
type jxrMultiVLC struct {
	table        uint8
	delta        [2]uint8
	discriminant [2]int
}

// jxrModel 保存系数归一化的双组状态及固定长度细化位数
type jxrModel struct {
	state [2]int
	bits  [2]uint8
}

// jxrArithmetic 保存整数运算局部变量的数值范围及字长标志
type jxrArithmetic struct {
	minimum, maximum int64
	longWord         bool
}

// read 读取不超过三十二位的无符号值，截断时记录错误并停止推进
// 入参: count 位数
// 返回: uint32 读取值，发生错误时为零
func (b *jxrBits) read(count uint) uint32 {
	if b.err == nil && b.input == nil && count <= 32 {
		limit := uint64(len(b.data))
		if b.end != 0 {
			limit = b.end
		}
		if limit <= uint64(len(b.data)) && limit <= ^uint64(0)/8 {
			if count == 0 {
				if b.bit <= limit<<3 {
					return 0
				}
				return b.readSlow(0)
			}
			at := b.bit
			start := at >> 3
			if start+8 <= limit {
				window := binary.BigEndian.Uint64(b.data[start : start+8])
				b.bit = at + uint64(count)
				return uint32((window << (at & 7)) >> (64 - count))
			}
		}
	}
	return b.readSlow(count)
}

// readSlow 处理流尾部、显式边界及随机读取窗的精确读取
// 入参: count 位数
// 返回: uint32 读取值，发生错误时为零
func (b *jxrBits) readSlow(count uint) uint32 {
	if b.err != nil {
		return 0
	}
	total := b.length()
	if count > 32 || total > ^uint64(0)/8 || b.input == nil && total > uint64(len(b.data)) || b.bit > total*8 || uint64(count) > total*8-b.bit {
		b.err = fmt.Errorf("%w: JPEG XR bitstream end", ErrInvalid)
		return 0
	}
	if count == 0 {
		return 0
	}
	at := b.bit
	start, shift := at>>3, uint(at&7)
	if b.input == nil && start+8 <= uint64(len(b.data)) {
		window := binary.BigEndian.Uint64(b.data[start : start+8])
		b.bit = at + uint64(count)
		return uint32((window << shift) >> (64 - count))
	}
	bytes := (shift + count + 7) >> 3
	data := b.data
	if b.input != nil {
		var err error
		data, err = b.input.boundedBytes(start, bytes, b.end)
		if err != nil {
			b.err = err
			return 0
		}
		start = 0
	}
	var window uint64
	for i := range bytes {
		window = window<<8 | uint64(data[start+uint64(i)])
	}
	b.bit = at + uint64(count)
	return uint32((window << (64 - bytes*8 + shift)) >> (64 - count))
}

// length 返回独立码流的逻辑字节范围，随机输入不依赖完整切片的存在
// 返回: uint64 当前输入长度
func (b *jxrBits) length() uint64 {
	if b.end != 0 {
		return b.end
	}
	if b.input != nil {
		return b.input.size
	}
	return uint64(len(b.data))
}

// flag 读取单个标志位，读取错误仍保留在当前读取器中
// 返回: bool 标志值
func (b *jxrBits) flag() bool {
	if b.err == nil && b.input == nil {
		at := b.bit
		bytePos := at >> 3
		if bytePos < uint64(len(b.data)) && (b.end == 0 || b.end <= ^uint64(0)/8 && bytePos < b.end) {
			b.bit = at + 1
			return (b.data[bytePos]>>(7-(at&7)))&1 != 0
		}
	}
	return b.readSlow(1) != 0
}

// align 移到字节边界，必要时检查规范要求的零填充
// 入参: zero 是否要求填充位为零
func (b *jxrBits) align(zero bool) {
	count := uint((8 - b.bit&7) & 7)
	if value := b.read(count); zero && value != 0 && b.err == nil {
		b.err = fmt.Errorf("%w: JPEG XR byte alignment", ErrInvalid)
	}
}

// variableWord 读取标准的可变长度整数，显式返回逃逸字节区分偏移零与逃逸值并保留八字节宽度
// 返回: uint64 整数值，逃逸时为零, uint8 逃逸字节，非逃逸时为零
func (b *jxrBits) variableWord() (uint64, uint8) {
	first := b.read(8)
	if b.err != nil {
		return 0, 0
	}
	switch {
	case first < 0xfb:
		return uint64(first<<8 | b.read(8)), 0
	case first == 0xfb:
		return uint64(b.read(32)), 0
	case first == 0xfc:
		high, low := b.read(32), b.read(32)
		return uint64(high)<<32 | uint64(low), 0
	default:
		return 0, uint8(first)
	}
}

// tryAcquire 在共享上限内原子申请扩容所需瞬时容量，优先满足desired，不足时退化至required
// 入参: required 最低所需字节数, desired 期望扩容字节数
// 返回: uint64 实际申请到的容量, bool 是否申请成功
func (s *jxrSharedWriteBudget) tryAcquire(required, desired uint64) (uint64, bool) {
	for {
		cur := s.used.Load()
		if cur >= s.limit {
			s.exceeded.Store(true)
			return 0, false
		}
		avail := s.limit - cur
		grant := desired
		if grant > avail {
			grant = required
		}
		if grant > avail {
			s.exceeded.Store(true)
			return 0, false
		}
		if s.used.CompareAndSwap(cur, cur+grant) {
			return grant, true
		}
	}
}

// release 归还旧缓冲释放出的共享容量
// 入参: amount 归还字节数
func (s *jxrSharedWriteBudget) release(amount uint64) {
	if amount != 0 {
		s.used.Add(^amount + 1)
	}
}

// newJXRBitWriter 创建有界位写入器，缓冲增长同时计入旧、新容量
// 入参: ctx 已规范化Context, maxBytes 输出字节上限, maxMemory 缓冲内存上限
// 返回: jxrBitWriter 空写入器，错误保存在err字段中
func newJXRBitWriter(ctx context.Context, maxBytes, maxMemory uint64) jxrBitWriter {
	noCancel := ctx == context.Background() || ctx == context.TODO()
	b := jxrBitWriter{ctx: ctx, maxBytes: min(maxBytes, uint64(^uint(0)>>1), ^uint64(0)/8), maxMemory: maxMemory, noCancel: noCancel}
	if !noCancel {
		if err := ctx.Err(); err != nil {
			b.err = err
			return b
		}
	}
	if maxBytes == 0 {
		b.maxBytesExceeded = true
		b.err = &LimitError{Resource: "max output bytes", Limit: 0, Required: 1, Detail: "JPEG XR output buffer budget"}
	} else if maxMemory == 0 {
		b.err = &LimitError{Resource: "max memory bytes", Limit: 0, Required: 1, Detail: "JPEG XR output buffer budget"}
	}
	return b
}

// reset 归还当前缓冲在共享预算中的已占容量并重置写入器状态以便安全回退
func (b *jxrBitWriter) reset() {
	if b.budget != nil {
		if old := uint64(cap(b.data)); old != 0 {
			if b.budget.shared != nil {
				b.budget.shared.release(old)
			}
			if b.budget.used >= old {
				b.budget.used -= old
			} else {
				b.budget.used = 0
			}
		}
	}
	b.data = nil
	b.bit = 0
	b.err = nil
	b.checks = 0
	b.maxBytesExceeded = false
}

// withinFastBudget 检查所需长度与当前容量是否同时满足输出上限、内存上限与共享预算
// 入参: needed 目标长度字节数, capacity 当前切片容量字节数
// 返回: bool 是否可直接走无扩容快路径
func (b *jxrBitWriter) withinFastBudget(needed, capacity uint64) bool {
	return needed <= capacity && needed <= b.maxBytes && capacity <= b.maxMemory && (b.budget == nil || b.budget.used <= b.budget.limit && capacity <= b.budget.used)
}

// write 写入不超过三十二位的无符号值，不静默截断超出字段宽度的有效位
// 入参: value 字段值, count 字段位数
func (b *jxrBitWriter) write(value uint32, count uint) {
	if b.err == nil && count <= 32 && uint64(value) < uint64(1)<<count && (b.noCancel || b.checks >= 4 && b.checks&255 != 0) {
		pos := b.bit
		if pos <= ^uint64(0)-39 {
			needed := (pos + uint64(count) + 7) >> 3
			if b.withinFastBudget(needed, uint64(cap(b.data))) {
				if !b.noCancel {
					b.checks++
				}
				b.data = b.data[:needed]
				if count == 0 {
					return
				}
				at, shift := pos>>3, uint(pos&7)
				b.bit = pos + uint64(count)
				if total := shift + count; total <= 8 {
					b.data[at] |= byte(value << (8 - total))
					return
				}
				jxrPutBits(b.data[at:], value, shift+count)
				return
			}
		}
	}
	b.writeSlow(value, count)
}

// writeSlow 处理扩容、取消检查、边界校验及跨字节写入
// 入参: value 字段值, count 字段位数
func (b *jxrBitWriter) writeSlow(value uint32, count uint) {
	if b.err != nil {
		return
	}
	if count > 32 || uint64(value) >= uint64(1)<<count {
		b.err = fmt.Errorf("%w: JPEG XR output bit field", ErrInvalid)
		return
	}
	if !b.noCancel {
		if b.checks < 4 || b.checks&255 == 0 {
			if err := b.ctx.Err(); err != nil {
				b.err = err
				return
			}
		}
		b.checks++
	}
	pos := b.bit
	if pos > ^uint64(0)-uint64(count)-7 {
		b.maxBytesExceeded = true
		b.err = &LimitError{Resource: "max output bytes", Limit: b.maxBytes, Required: ^uint64(0), Detail: "JPEG XR output bit count"}
		return
	}
	if !b.reserve((pos + uint64(count) + 7) >> 3) {
		return
	}
	if count == 0 {
		return
	}
	at, shift := pos>>3, uint(pos&7)
	b.bit = pos + uint64(count)
	if total := shift + count; total <= 8 {
		b.data[at] |= byte(value << (8 - total))
		return
	}
	jxrPutBits(b.data[at:], value, shift+count)
}

// jxrPutBits 将跨越多个字节的高位优先码值写入已清零的新扩缓冲区
// 入参: data 起始字节切片, value 字段值, total 起始位偏移与字段位数之和
func jxrPutBits(data []byte, value uint32, total uint) {
	bits := uint64(value) << (64 - total)
	data[0] |= byte(bits >> 56)
	switch {
	case total <= 16:
		data[1] = byte(bits >> 48)
	case total <= 24:
		data[2] = byte(bits >> 40)
		data[1] = byte(bits >> 48)
	case total <= 32:
		data[3] = byte(bits >> 32)
		data[2] = byte(bits >> 40)
		data[1] = byte(bits >> 48)
	default:
		data[4] = byte(bits >> 24)
		data[3] = byte(bits >> 32)
		data[2] = byte(bits >> 40)
		data[1] = byte(bits >> 48)
	}
}

// flag 写入单个标志位
// 入参: value 标志值
func (b *jxrBitWriter) flag(value bool) {
	if b.err == nil && (b.noCancel || b.checks >= 4 && b.checks&255 != 0) {
		pos := b.bit
		if pos <= ^uint64(0)-8 {
			needed := (pos + 8) >> 3
			if b.withinFastBudget(needed, uint64(cap(b.data))) {
				if !b.noCancel {
					b.checks++
				}
				b.data = b.data[:needed]
				if value {
					b.data[pos>>3] |= 1 << (7 - (pos & 7))
				}
				b.bit = pos + 1
				return
			}
		}
	}
	bit := uint32(0)
	if value {
		bit = 1
	}
	b.writeSlow(bit, 1)
}

// align 以零填充到下一个字节边界，不在失败后推进位置
func (b *jxrBitWriter) align() {
	if b.err != nil {
		return
	}
	if !b.noCancel {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
	}
	count := uint((8 - b.bit&7) & 7)
	pos := b.bit
	if pos > ^uint64(0)-uint64(count)-7 {
		b.maxBytesExceeded = true
		b.err = &LimitError{Resource: "max output bytes", Limit: b.maxBytes, Required: ^uint64(0), Detail: "JPEG XR output bit count"}
		return
	}
	needed := (pos + uint64(count) + 7) >> 3
	if b.withinFastBudget(needed, uint64(cap(b.data))) {
		b.data = b.data[:needed]
	} else if !b.reserve(needed) {
		return
	}
	b.bit = pos + uint64(count)
}

// variableWord 按标准选择十六位、三十二位或六十四位可变长度整数表示
// 入参: value 整数值，不生成保留的逃逸标识
func (b *jxrBitWriter) variableWord(value uint64) {
	if value < 0xfb00 {
		b.write(uint32(value), 16)
	} else if value <= 1<<32-1 {
		b.write(0xfb, 8)
		b.write(uint32(value), 32)
	} else {
		b.write(0xfc, 8)
		b.write(uint32(value>>32), 32)
		b.write(uint32(value), 32)
	}
}

// reserve 在分配前检查输出及增长期间的缓冲容量，不改变已有编码数据
// 入参: required 所需字节数
// 返回: bool 是否可继续写入，失败原因保存在err字段中
func (b *jxrBitWriter) reserve(required uint64) bool {
	if required > b.maxBytes {
		b.maxBytesExceeded = true
		b.err = &LimitError{Resource: "max output bytes", Limit: b.maxBytes, Required: required, Detail: "JPEG XR encoded output budget"}
		return false
	}
	if required > b.maxMemory || uint64(cap(b.data)) > b.maxMemory {
		if b.budget != nil && b.budget.shared != nil {
			b.budget.shared.exceeded.Store(true)
		}
		b.err = &LimitError{Resource: "max memory bytes", Limit: b.maxMemory, Required: max(required, uint64(cap(b.data))), Detail: "JPEG XR encoded output budget"}
		return false
	}
	if b.budget != nil && (b.budget.used > b.budget.limit || uint64(cap(b.data)) > b.budget.used) {
		if b.budget.shared != nil {
			b.budget.shared.exceeded.Store(true)
		}
		b.err = &LimitError{Resource: "max memory bytes", Limit: b.budget.limit, Required: max(b.budget.used, uint64(cap(b.data))), Detail: "JPEG XR shared output budget"}
		return false
	}
	if required > uint64(cap(b.data)) {
		old := uint64(cap(b.data))
		capacity := max(required, min(b.maxBytes, max(uint64(512), old*2)))
		available := b.maxMemory - old
		if b.budget != nil && b.budget.shared != nil {
			if capacity > available {
				capacity = required
			}
			if capacity > available {
				b.budget.shared.exceeded.Store(true)
				b.err = &LimitError{Resource: "max memory bytes", Limit: b.maxMemory, Required: jxrSaturatedAdd(old, required), Detail: "JPEG XR output growth memory"}
				return false
			}
			granted, ok := b.budget.shared.tryAcquire(required, capacity)
			if !ok {
				b.err = &LimitError{Resource: "max memory bytes", Limit: b.budget.shared.limit, Required: jxrSaturatedAdd(b.budget.shared.used.Load(), required), Detail: "JPEG XR output growth memory"}
				return false
			}
			next := make([]byte, int(required), int(granted))
			copy(next, b.data)
			b.data = next
			b.budget.shared.release(old)
			b.budget.used += granted - old
			return true
		}
		if b.budget != nil {
			available = min(available, b.budget.limit-b.budget.used)
		}
		if capacity > available {
			capacity = required
		}
		if capacity > available {
			limit := b.maxMemory
			used := old
			if b.budget != nil && b.budget.limit-b.budget.used < b.maxMemory-old {
				limit = b.budget.limit
				used = b.budget.used
			}
			b.err = &LimitError{Resource: "max memory bytes", Limit: limit, Required: jxrSaturatedAdd(used, required), Detail: "JPEG XR output growth memory"}
			return false
		}
		next := make([]byte, int(required), int(capacity))
		copy(next, b.data)
		b.data = next
		if b.budget != nil {
			b.budget.used += capacity - old
		}
	} else {
		b.data = b.data[:int(required)]
	}
	return true
}

// appendBytes 追加独立且按字节对齐的数据，在复制期间响应取消
// 入参: data 不与输出缓冲共享存储的数据
func (b *jxrBitWriter) appendBytes(data []byte) {
	if b.err != nil {
		return
	}
	if b.bit&7 != 0 {
		b.err = fmt.Errorf("%w: JPEG XR output byte alignment", ErrInvalid)
		return
	}
	if !b.noCancel {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
	}
	start := b.bit >> 3
	if uint64(len(data)) > b.maxBytes || start > b.maxBytes-uint64(len(data)) {
		b.maxBytesExceeded = true
		b.err = &LimitError{Resource: "max output bytes", Limit: b.maxBytes, Required: jxrSaturatedAdd(start, uint64(len(data))), Detail: "JPEG XR output byte count"}
		return
	}
	if !b.reserve(start + uint64(len(data))) {
		return
	}
	if b.noCancel {
		copy(b.data[start:start+uint64(len(data))], data)
		b.bit += uint64(len(data)) * 8
		return
	}
	for len(data) != 0 {
		if err := b.ctx.Err(); err != nil {
			b.err = err
			return
		}
		count := min(16384, len(data))
		copy(b.data[start:start+uint64(count)], data[:count])
		start += uint64(count)
		b.bit += uint64(count) * 8
		data = data[count:]
	}
}

// jxrMakeVLC 展开标准固定前缀码，不接受来自文件的编码表定义
// 入参: codes 按符号编号排列的不超过八位的标准前缀码
// 返回: [256]jxrCode 八位前缀查找表
func jxrMakeVLC(codes []string) [256]jxrCode {
	var table [256]jxrCode
	for value, code := range codes {
		prefix := 0
		for _, bit := range code {
			prefix = prefix<<1 | int(bit-'0')
		}
		count := 1 << (8 - len(code))
		for at := prefix * count; at < (prefix+1)*count; at++ {
			table[at] = jxrCode{uint8(value), uint8(len(code))}
		}
	}
	return table
}

// readJXRVLC 查看最多八位前缀，仅推进实际编码位数，不以补零替代截断数据
// 入参: b 码流读取器, table 标准前缀查找表
// 返回: uint8 符号，错误保存在读取器中
func readJXRVLC(b *jxrBits, table *[256]jxrCode) uint8 {
	if b.err == nil && b.input == nil {
		limit := uint64(len(b.data))
		if b.end != 0 {
			limit = b.end
		}
		if limit <= uint64(len(b.data)) && limit <= ^uint64(0)/8 {
			pos := b.bit
			at := pos >> 3
			if at+2 <= limit {
				code := table[uint8(binary.BigEndian.Uint16(b.data[at:at+2])>>(8-(pos&7)))]
				if code.bits != 0 {
					b.bit = pos + uint64(code.bits)
					return code.value
				}
				b.err = fmt.Errorf("%w: JPEG XR variable-length code", ErrInvalid)
				return 0
			}
		}
	}
	return readJXRVLCSlow(b, table)
}

// readJXRVLCSlow 处理流末尾、显式边界及随机读取窗的前缀解码
// 入参: b 码流读取器, table 标准前缀查找表
// 返回: uint8 符号，错误保存在读取器中
func readJXRVLCSlow(b *jxrBits, table *[256]jxrCode) uint8 {
	if b.err != nil {
		return 0
	}
	total := b.length()
	if total > ^uint64(0)/8 || b.input == nil && total > uint64(len(b.data)) || b.bit >= total*8 {
		b.err = fmt.Errorf("%w: JPEG XR variable-length code end", ErrInvalid)
		return 0
	}
	pos := b.bit
	at, shift := pos>>3, uint(pos&7)
	data := b.data
	if b.input != nil {
		var err error
		data, err = b.input.boundedBytes(at, uint(min(uint64(2), total-at)), b.end)
		if err != nil {
			b.err = err
			return 0
		}
		at = 0
	}
	window := uint16(data[at]) << 8
	if at+1 < uint64(len(data)) && (b.input != nil || at+1 < total) {
		window |= uint16(data[at+1])
	}
	code := table[uint8(window>>(8-shift))]
	if code.bits == 0 {
		b.err = fmt.Errorf("%w: JPEG XR variable-length code", ErrInvalid)
		return 0
	}
	if uint64(code.bits) > total*8-pos {
		b.err = fmt.Errorf("%w: JPEG XR bitstream end", ErrInvalid)
		return 0
	}
	b.bit = pos + uint64(code.bits)
	return code.value
}

// newJXRMultiVLC 初始化多编码表选择状态
// 返回: jxrMultiVLC 初始状态
func newJXRMultiVLC() jxrMultiVLC {
	return jxrMultiVLC{table: 1, delta: [2]uint8{0, 1}}
}

// first 读取首系数索引并累计相邻编码表的判别值
// 入参: b 码流读取器
// 返回: uint8 首系数索引，错误保存在读取器中
func (v *jxrMultiVLC) first(b *jxrBits) uint8 {
	value := readJXRVLC(b, &jxrFirstCode[v.table])
	if b.err == nil {
		v.discriminant[0] += jxrFirstDelta[v.delta[0]][value]
		v.discriminant[1] += jxrFirstDelta[v.delta[1]][value]
	}
	return value
}

// index 读取后续系数索引并累计相邻编码表的判别值
// 入参: b 码流读取器
// 返回: uint8 后续系数索引，错误保存在读取器中
func (v *jxrMultiVLC) index(b *jxrBits) uint8 {
	value := readJXRVLC(b, &jxrIndexCode[v.table])
	if b.err == nil {
		v.discriminant[0] += jxrIndexDelta[v.delta[0]][value]
		v.discriminant[1] += jxrIndexDelta[v.delta[1]][value]
	}
	return value
}

// adapt 自适应调整编码表，切换时重新指定相邻判别增量表
// 入参: maximum 最大编码表索引
func (v *jxrMultiVLC) adapt(maximum uint8) {
	changed := false
	if v.discriminant[0] < -8 && v.table != 0 {
		v.table--
		changed = true
	} else if v.discriminant[1] > 8 && v.table != maximum {
		v.table++
		changed = true
	}
	if changed {
		v.discriminant = [2]int{}
		v.delta[0], v.delta[1] = v.table, v.table
		if v.table != 0 {
			v.delta[0]--
		}
		if v.table == maximum {
			v.delta[1]--
		}
	} else {
		v.discriminant[0] = min(64, max(-64, v.discriminant[0]))
		v.discriminant[1] = min(64, max(-64, v.discriminant[1]))
	}
}

// newJXRModel 初始化频带归一化状态
// 入参: band 已校验的DC、低通或高通频带索引
// 返回: jxrModel 初始状态
func newJXRModel(band uint8) jxrModel {
	return jxrModel{bits: [2]uint8{(2 - band) * 4, (2 - band) * 4}}
}

// update 按宏块非零系数数量更新归一化状态，算术右移保留标准的负数舍入规则，支持至四千一百一十一通道
// 入参: mean 亮度及其他分量的非零数量, p 图像平面, band 已校验的频带索引
func (m *jxrModel) update(mean [2]int, p *jxrPlane, band uint8) {
	mean[0] *= jxrModelLumaWeight[band]
	if p.color == 1 || p.color == 2 {
		mean[1] *= jxrModelChromaWeight[p.color-1][band]
	} else if p.channels <= len(jxrModelComponentWeight[0]) {
		mean[1] *= jxrModelComponentWeight[band][p.channels-1]
		if band == 2 {
			mean[1] >>= 4
		}
	} else {
		others := p.channels - 1
		mean[1] = (mean[1] * jxrModelLumaWeight[band]) / others
	}
	count := 2
	if p.color == 0 {
		count = 1
	}
	for j := range count {
		state := m.state[j]
		delta := (mean[j] - 70) >> 2
		if delta <= -8 {
			delta = max(delta+4, -16)
			state += delta
			if state < -8 {
				if m.bits[j] == 0 {
					state = -8
				} else {
					state = 0
					m.bits[j]--
				}
			}
		} else if delta >= 8 {
			delta = min(delta-4, 15)
			state += delta
			if state > 8 {
				if m.bits[j] >= 15 {
					m.bits[j], state = 15, 8
				} else {
					state = 0
					m.bits[j]++
				}
			}
		}
		m.state[j] = state
	}
}

// newJXRArithmetic 创建十六位或三十二位的局部变量校验器
// 入参: longWord 是否使用三十二位范围
// 返回: jxrArithmetic 运算校验状态
func newJXRArithmetic(longWord bool) jxrArithmetic {
	return jxrArithmetic{longWord: longWord}
}

// value 记录赋给局部变量的数值范围，不裁剪或回绕；空接收者不记录范围
// 入参: value 运算结果
// 返回: int64 原始运算结果
func (c *jxrArithmetic) value(value int64) int64 {
	if c != nil {
		c.minimum = min(c.minimum, value)
		c.maximum = max(c.maximum, value)
	}
	return value
}

// error 检查整个算子的局部变量范围，空接收者不参与校验
// 返回: error 数值范围错误
func (c *jxrArithmetic) error() error {
	if c != nil {
		low, high := jxrWordBounds(c.longWord)
		if c.minimum < low || c.maximum > high {
			return fmt.Errorf("%w: JPEG XR local variable range", ErrInvalid)
		}
	}
	return nil
}

// jxrSafeBlock16 判断十六个输入值是否均处于零溢出快路径的安全上界内
// 入参: v 待检查的十六个输入值, longWord 是否使用三十二位范围
// 返回: bool 全部值处于安全上界内时为真
func jxrSafeBlock16(v *[16]int64, longWord bool) bool {
	offset := jxrSafeShortLimit
	if longWord {
		offset = jxrSafeLongLimit
	}
	acc := (uint64(v[0]) + offset) | (uint64(v[1]) + offset) | (uint64(v[2]) + offset) | (uint64(v[3]) + offset) |
		(uint64(v[4]) + offset) | (uint64(v[5]) + offset) | (uint64(v[6]) + offset) | (uint64(v[7]) + offset) |
		(uint64(v[8]) + offset) | (uint64(v[9]) + offset) | (uint64(v[10]) + offset) | (uint64(v[11]) + offset) |
		(uint64(v[12]) + offset) | (uint64(v[13]) + offset) | (uint64(v[14]) + offset) | (uint64(v[15]) + offset)
	return acc < offset<<1
}

// jxrSafeValues4 判断四个输入值是否均处于零溢出快路径的安全上界内
// 入参: a, b, c, d 待检查的四个输入值, longWord 是否使用三十二位范围
// 返回: bool 全部值处于安全上界内时为真
func jxrSafeValues4(a, b, c, d int64, longWord bool) bool {
	offset := jxrSafeShortLimit
	if longWord {
		offset = jxrSafeLongLimit
	}
	acc := (uint64(a) + offset) | (uint64(b) + offset) | (uint64(c) + offset) | (uint64(d) + offset)
	return acc < offset<<1
}

// jxrSafeValues2 判断两个输入值是否均处于零溢出快路径的安全上界内
// 入参: a, b 待检查的两个输入值, longWord 是否使用三十二位范围
// 返回: bool 两个值处于安全上界内时为真
func jxrSafeValues2(a, b int64, longWord bool) bool {
	offset := jxrSafeShortLimit
	if longWord {
		offset = jxrSafeLongLimit
	}
	return ((uint64(a) + offset) | (uint64(b) + offset)) < offset<<1
}

// jxrWordBounds 取得LONG_WORD_FLAG规定的有符号数组范围
// 入参: longWord 是否使用三十二位范围
// 返回: int64 最小值, int64 最大值
func jxrWordBounds(longWord bool) (int64, int64) {
	width := uint(16)
	if longWord {
		width = 32
	}
	low := -int64(1) << (width - 1)
	return low, -low - 1
}

// scaleJXRWord 先检查反量化结果的声明范围，再执行不会溢出的乘法
// 入参: value 系数, factor 正量化倍率, longWord 是否使用三十二位范围
// 返回: int64 反量化系数, error 数值范围错误
func scaleJXRWord(value, factor int64, longWord bool) (int64, error) {
	low, high := jxrWordBounds(longWord)
	if factor < 1 || value < low/factor || value > high/factor {
		return 0, fmt.Errorf("%w: JPEG XR dequantized coefficient range", ErrInvalid)
	}
	return value * factor, nil
}

// checkJXRWords 检查LONG_WORD_FLAG规定的数组值范围，越界按数据错误处理
// 入参: values 待校验的数组值, longWord 是否使用三十二位数值范围
// 返回: error 数值范围错误
func checkJXRWords(values []int64, longWord bool) error {
	bits := uint(16)
	if longWord {
		bits = 32
	}
	low, high := jxrWordBounds(longWord)
	for _, value := range values {
		if value < low || value > high {
			return fmt.Errorf("%w: JPEG XR %d-bit coefficient range", ErrInvalid, bits)
		}
	}
	return nil
}

// checkJXRWordsContext 分段校验大型系数缓冲，避免整幅扫描阻塞取消响应
// 入参: ctx Context, values 待校验的数组值, longWord 是否使用三十二位数值范围
// 返回: error 数值范围或取消错误
func checkJXRWordsContext(ctx context.Context, values []int64, longWord bool) error {
	for len(values) != 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(len(values), 16384)
		if err := checkJXRWords(values[:n], longWord); err != nil {
			return err
		}
		values = values[n:]
	}
	return ctx.Err()
}

// checkLow 检查实际分量中保存的DC及低通系数范围
// 入参: values 待校验的逐分量系数
// 返回: error 数值范围错误
func (c *jxrCoefficients) checkLow(values [][16]int64) error {
	if len(values) < c.plane.channels {
		return fmt.Errorf("%w: JPEG XR lowpass component storage", ErrInvalid)
	}
	for i := range c.plane.channels {
		if err := checkJXRWords(values[i][:], c.longWord); err != nil {
			return err
		}
	}
	return nil
}
