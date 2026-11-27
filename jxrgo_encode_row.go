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
	"sync"
)

// jxrRowStageCopyBytes 定义顺序输出暂存拷贝块大小
const jxrRowStageCopyBytes = 32768

// jxrMemWriteSeeker 提供受输出字节上限、内存上限与Context取消约束的有界内存暂存定位写入器
type jxrMemWriteSeeker struct {
	ctx       context.Context
	data      []byte
	pos       int64
	maxBytes  uint64
	maxMemory uint64
	budget    *jxrWriteBudget
}

// jxrOffsetReadWriteSeeker 将外部暂存的偏移子区间映射为零起点独立io.ReadWriteSeeker，支持单暂存分区复用
type jxrOffsetReadWriteSeeker struct {
	target io.ReadWriteSeeker
	base   int64
	pos    int64
	length int64
}

// RowEncoder 提供面向顺序io.Writer与io.WriteSeeker的推式单遍流式JPEG XR行编码器
// 内部采用与EncodeRows相同的有界环形行缓存实时编码并提交已就绪分块，非转置顺序无压缩流直写目标Writer实现常量行内存，Close幂等安全且不关闭底层输出流
type RowEncoder struct {
	mu           sync.Mutex
	finalWriter  io.Writer
	stageWriter  io.WriteSeeker
	stageReader  io.ReadSeeker
	memStage     *jxrMemWriteSeeker
	info         Info
	file         bool
	separate     bool
	uncompressed bool
	uncDirect    bool
	closed       bool
	failed       error
	height       int
	width        int
	step         int
	pushed       int
	ring         *Raster
	rowView      Raster
	rowStorage   Raster
	alphaStorage jxrAlphaRowStorage
	separateIn   jxrSeparateTileInput
	separateOff  jxrContainerOffsets
	mainStart    uint64
	fileSizeAt   uint64
	fileMaxBytes uint64
	packetStream *jxrPacketStream
	streamPlan   jxrHardTileStreamPlan
	coders       []jxrTileCoder
	sharedBudget jxrSharedWriteBudget
	nextOrdinal  int
	nextTile     int
	nextPlan     jxrHardTileInput
	uncLayout    jxrUncompressedLayout
	uncTransform uint8
	uncHeader    []byte
	uncRowBuf    []byte
	uncAlphaBuf  []byte
	uncMainData  []byte
	uncAlphaData []byte
	uncMainStart uint64
	uncAlphaAt   uint64
	uncTotal     uint64
	uncPadMain   bool
	uncNeedSeek  bool
}

// NewRowEncoder 初始化推式单遍流式行编码器并完成元数据、方向、预算及输出头预检
// w实现io.WriteSeeker时原地直写回填且须位于目标起点；普通顺序io.Writer在非转置无压缩且未提供Scratch时（无需定位回填布局）直写目标Writer，否则优先借用opts.Scratch或有界内存暂存回填后流式拷贝，不关闭底层输出流
// 入参: ctx Context，nil采用Background, w 目标输出流, info 完整原生图像元数据, opts 可选编码选项
// 返回: *RowEncoder 推式行编码器, error 参数、格式、预算、取消或输出错误
func NewRowEncoder(ctx context.Context, w io.Writer, info Info, opts *EncodeOptions) (*RowEncoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	enc, err := newRowEncoder(ctx, w, info, opts)
	if err != nil {
		return nil, wrapJXRError("NewRowEncoder", err)
	}
	return enc, nil
}

// Info 返回编码器规范化后的图像元数据副本，关闭后或接收者为nil时返回零值
// 返回: Info 图像元数据副本
func (e *RowEncoder) Info() Info {
	if e == nil {
		return Info{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return cloneInfo(e.info)
}

// RowStep 返回每次调用WriteRow须提供的最小行步长，全采样图像为1，纵向二倍子采样图像为2
// 返回: int 最小行步长
func (e *RowEncoder) RowStep() int {
	if e == nil {
		return 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.step
}

// WrittenRows 返回已成功推入编码器的显示行数
// 返回: int 已推入行数
func (e *RowEncoder) WrittenRows() int {
	if e == nil {
		return 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pushed
}

// WriteRow 将原生行带按步长推入流式编码器并立即编码刷出全部已就绪分块
// src宽度须等于图像宽度、高度为RowStep正整数倍且不超过剩余未写行数，颜色空间、分量编码及采样步长须严格一致
// 入参: ctx Context，nil采用Background, src 源原生行带或条带
// 返回: error 参数、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRow(ctx context.Context, src *Raster) error {
	if e == nil {
		return wrapJXRError("RowEncoder.WriteRow", ErrClosed)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.writeRowRaster(ctx, src); err != nil {
		return wrapJXRError("RowEncoder.WriteRow", err)
	}
	return nil
}

// WriteRowNRGBA 将单行八位非预乘RGBA扫描行零分配推入八位或十六位非预乘RGB流式编码器
// 入参: ctx Context，nil采用Background, src 源八位RGBA字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowNRGBA(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowNRGBA", func(row *Raster) error {
		return row.writeRowRGBABytes(0, src, false)
	})
}

// WriteRowNRGBA64 将单行大端十六位非预乘RGBA扫描行零分配推入十六位非预乘RGB流式编码器
// 入参: ctx Context，nil采用Background, src 源大端十六位RGBA字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowNRGBA64(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowNRGBA64", func(row *Raster) error {
		return row.writeRowRGBA64Bytes(0, src, false)
	})
}

// WriteRowGray 将单行八位灰度扫描行零分配推入八位或十六位无Alpha灰度流式编码器
// 入参: ctx Context，nil采用Background, src 源八位灰度字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowGray(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowGray", func(row *Raster) error {
		return row.writeRowGray(0, src)
	})
}

// WriteRowGray16 将单行大端十六位灰度扫描行零分配推入十六位无Alpha灰度流式编码器
// 入参: ctx Context，nil采用Background, src 源大端十六位灰度字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowGray16(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowGray16", func(row *Raster) error {
		return row.writeRowGray16(0, src)
	})
}

// WriteRowCMYK 将单行八位CMYK扫描行零分配推入八位或十六位非预乘CMYK流式编码器
// 入参: ctx Context，nil采用Background, src 源八位CMYK字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowCMYK(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowCMYK", func(row *Raster) error {
		return row.writeRowCMYK(0, src, nil)
	})
}

// WriteRowCMYKAlpha 将单行八位CMYK及可选独立八位Alpha扫描行零分配推入八位或十六位非预乘CMYK流式编码器
// 入参: ctx Context，nil采用Background, cmyk 源八位CMYK字节缓冲, alpha 可选源八位Alpha字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowCMYKAlpha(ctx context.Context, cmyk, alpha []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowCMYKAlpha", func(row *Raster) error {
		return row.writeRowCMYK(0, cmyk, alpha)
	})
}

// WriteRowRGB 将单行紧凑二十四位八位RGB扫描行零分配推入八位或十六位非预乘RGB流式编码器
// 入参: ctx Context，nil采用Background, src 源紧凑八位RGB字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowRGB(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowRGB", func(row *Raster) error {
		return row.writeRowRGB(0, src)
	})
}

// WriteRowRGBA 将单行八位预乘RGBA扫描行零分配推入八位或十六位预乘RGB流式编码器
// 入参: ctx Context，nil采用Background, src 源八位预乘RGBA字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowRGBA(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowRGBA", func(row *Raster) error {
		return row.writeRowRGBABytes(0, src, true)
	})
}

// WriteRowRGBA64 将单行大端十六位预乘RGBA扫描行零分配推入十六位预乘RGB流式编码器
// 入参: ctx Context，nil采用Background, src 源大端十六位预乘RGBA字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowRGBA64(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowRGBA64", func(row *Raster) error {
		return row.writeRowRGBA64Bytes(0, src, true)
	})
}

// WriteRowBlackWhite 将单行高位优先一位打包黑白扫描行零分配推入原生一位黑白流式编码器
// 入参: ctx Context，nil采用Background, src 源一位打包字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowBlackWhite(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowBlackWhite", func(row *Raster) error {
		return row.writeRowBlackWhite(0, src)
	})
}

// WriteRowCMYK16 将单行大端十六位CMYK扫描行零分配推入十六位非预乘CMYK流式编码器
// 入参: ctx Context，nil采用Background, src 源大端十六位CMYK字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowCMYK16(ctx context.Context, src []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowCMYK16", func(row *Raster) error {
		return row.writeRowCMYK16(0, src, nil)
	})
}

// WriteRowCMYK16Alpha 将单行大端十六位CMYK及可选独立大端十六位Alpha扫描行零分配推入十六位非预乘CMYK流式编码器
// 入参: ctx Context，nil采用Background, cmyk 源大端十六位CMYK字节缓冲, alpha 可选源大端十六位Alpha字节缓冲
// 返回: error 参数、格式、状态、预算、精度、取消或写入错误
func (e *RowEncoder) WriteRowCMYK16Alpha(ctx context.Context, cmyk, alpha []byte) error {
	return e.writeRowFast(ctx, "RowEncoder.WriteRowCMYK16Alpha", func(row *Raster) error {
		return row.writeRowCMYK16(0, cmyk, alpha)
	})
}

// Close 完成剩余分块编码、索引回填、顺序输出拷贝并释放全部内部缓冲，重复调用幂等安全
// 返回: error 未写满行、编码、预算或写入错误
func (e *RowEncoder) Close() error {
	return e.closeWithOp(context.Background(), "RowEncoder.Close")
}

// CloseContext 在Context取消约束下完成剩余分块编码、索引回填与顺序输出拷贝并释放内部缓冲，重复调用幂等安全
// 入参: ctx Context，nil采用Background
// 返回: error 未写满行、编码、预算、取消或写入错误
func (e *RowEncoder) CloseContext(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	return e.closeWithOp(ctx, "RowEncoder.CloseContext")
}

// closeWithOp 共用关闭收尾流程并按公开入口操作名包装错误
// 入参: ctx 已规范化Context, op 公开操作名
// 返回: error 未写满行、编码、预算、取消或写入错误
func (e *RowEncoder) closeWithOp(ctx context.Context, op string) error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	err := e.finish(ctx)
	e.release()
	if err != nil {
		return wrapJXRError(op, err)
	}
	return nil
}

// availableLimit 计算内存暂存当前允许达到的最大总容量，结合共享写缓冲预算防止超额
// 返回: uint64 当前允许的最大总字节容量
func (m *jxrMemWriteSeeker) availableLimit() uint64 {
	limit := m.maxMemory
	if m.budget != nil {
		free := uint64(0)
		if m.budget.limit > m.budget.used {
			free = m.budget.limit - m.budget.used
		}
		limit = min(limit, jxrSaturatedAdd(uint64(cap(m.data)), free))
	}
	return limit
}

// Write 将字节写入有界内存暂存并检查Context、输出上限与共享内存上限
// 入参: p 待写入字节切片
// 返回: int 实际写入字节数, error 取消或上限错误
func (m *jxrMemWriteSeeker) Write(p []byte) (int, error) {
	if m.ctx != nil {
		if err := m.ctx.Err(); err != nil {
			return 0, err
		}
	}
	if m.pos < 0 || uint64(len(p)) > math.MaxInt64-uint64(m.pos) {
		return 0, fmt.Errorf("%w: sequential row stage position", ErrLimit)
	}
	end := uint64(m.pos) + uint64(len(p))
	if end > m.maxBytes {
		return 0, &LimitError{Resource: "max output bytes", Limit: m.maxBytes, Required: end, Detail: "sequential row output stage"}
	}
	if end > uint64(cap(m.data)) {
		limit := m.availableLimit()
		maxInt := uint64(^uint(0) >> 1)
		if end > maxInt || end > limit {
			return 0, &LimitError{Resource: "max memory bytes", Limit: limit, Required: end, Detail: "sequential row output stage memory"}
		}
		target := uint64(cap(m.data)) * 2
		if target < 4096 {
			target = 4096
		}
		if target < end {
			target = end
		}
		target = min(target, limit, m.maxBytes, maxInt)
		oldCap := uint64(cap(m.data))
		grown := make([]byte, len(m.data), int(target))
		copy(grown, m.data)
		m.data = grown
		if m.budget != nil {
			m.budget.used += uint64(cap(m.data)) - oldCap
		}
	}
	if int(end) > len(m.data) {
		m.data = m.data[:int(end)]
	}
	copy(m.data[m.pos:int(end)], p)
	m.pos = int64(end)
	return len(p), nil
}

// Seek 在已分配内存暂存范围内移动读写游标
// 入参: offset 偏移量, whence 定位基准
// 返回: int64 新偏移量, error 定位错误
func (m *jxrMemWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	var base int64
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		base = m.pos
	case io.SeekEnd:
		base = int64(len(m.data))
	default:
		return 0, fmt.Errorf("%w: sequential row stage seek origin", ErrInvalid)
	}
	if offset < -base || offset > math.MaxInt64-base {
		return 0, fmt.Errorf("%w: sequential row stage seek range", ErrInvalid)
	}
	next := base + offset
	limit := m.availableLimit()
	if uint64(next) > m.maxBytes || uint64(next) > limit {
		return 0, &LimitError{Resource: "max memory bytes", Limit: min(m.maxBytes, limit), Required: uint64(next), Detail: "sequential row stage seek"}
	}
	m.pos = next
	return m.pos, nil
}

// Read 从内存暂存当前游标读取数据
// 入参: p 目标切片
// 返回: int 实际读取字节数, error EOF错误
func (m *jxrMemWriteSeeker) Read(p []byte) (int, error) {
	if m.pos >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(p, m.data[m.pos:])
	m.pos += int64(n)
	return n, nil
}

// Read 在子区间偏移内从底层暂存读取字节
// 入参: p 目标切片
// 返回: int 读取字节数, error 读取或定位错误
func (s *jxrOffsetReadWriteSeeker) Read(p []byte) (int, error) {
	if s.pos >= s.length {
		return 0, io.EOF
	}
	if _, err := s.target.Seek(s.base+s.pos, io.SeekStart); err != nil {
		return 0, err
	}
	limit := s.length - s.pos
	if int64(len(p)) > limit {
		p = p[:int(limit)]
	}
	n, err := s.target.Read(p)
	if n > 0 {
		s.pos += int64(n)
	}
	return n, err
}

// Write 在子区间偏移内向底层暂存写入字节
// 入参: p 待写入字节
// 返回: int 写入字节数, error 写入或定位错误
func (s *jxrOffsetReadWriteSeeker) Write(p []byte) (int, error) {
	if _, err := s.target.Seek(s.base+s.pos, io.SeekStart); err != nil {
		return 0, err
	}
	n, err := s.target.Write(p)
	if n > 0 {
		s.pos += int64(n)
		if s.pos > s.length {
			s.length = s.pos
		}
	}
	return n, err
}

// Seek 在子区间零起点坐标系内定位读写游标
// 入参: offset 相对偏移, whence 定位基准
// 返回: int64 逻辑位置, error 定位错误
func (s *jxrOffsetReadWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	var origin int64
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		origin = s.pos
	case io.SeekEnd:
		origin = s.length
	default:
		return 0, fmt.Errorf("%w: scratch partition seek origin", ErrInvalid)
	}
	if offset < -origin || offset > math.MaxInt64-origin-s.base {
		return 0, fmt.Errorf("%w: scratch partition seek range", ErrInvalid)
	}
	next := origin + offset
	if _, err := s.target.Seek(s.base+next, io.SeekStart); err != nil {
		return 0, err
	}
	s.pos = next
	return s.pos, nil
}

// copyJXRStageToWriter 将已完成回填的暂存流从零起点按块流式拷贝至最终顺序输出流
// 入参: ctx 已规范化Context, dst 最终顺序输出, src 可定位暂存源, length 待拷贝字节数
// 返回: error 定位、读取、写入或取消错误
func copyJXRStageToWriter(ctx context.Context, dst io.Writer, src io.ReadSeeker, length int64) error {
	if length < 0 {
		return fmt.Errorf("%w: staged output length", ErrInvalid)
	}
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return err
	}
	var buf [jxrRowStageCopyBytes]byte
	remaining := length
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := min(int64(len(buf)), remaining)
		if err := readJXRAlphaRowStorage(ctx, src, buf[:int(chunk)]); err != nil {
			return err
		}
		if err := writeEncoderOutput(ctx, dst, buf[:int(chunk)]); err != nil {
			return err
		}
		remaining -= chunk
	}
	return ctx.Err()
}

// encodeJXRRowsSequential 借助Scratch或有界内存暂存支持EncodeRows写入普通顺序io.Writer
// 入参: ctx 已规范化Context, w 顺序输出流, info 原生描述, read 顺序行源, opts 编码选项, file 是否写JXR容器文件
// 返回: error 参数、资源、源、暂存、取消或写入错误
func encodeJXRRowsSequential(ctx context.Context, w io.Writer, info Info, read RasterRowReader, opts *EncodeOptions, file bool) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if file {
		settings.Format = FormatJXR
	} else {
		settings.Format = FormatCodestream
	}
	if settings.Scratch == nil {
		enc, err := newRowEncoder(ctx, w, info, &settings)
		if err != nil {
			return err
		}
		defer enc.release()
		for y := 0; y < enc.height; y += enc.step {
			if err := ctx.Err(); err != nil {
				return err
			}
			height := min(enc.step, enc.height-y)
			view := enc.bindRingRowView(height)
			if err := read(ctx, y, view); err != nil {
				return err
			}
			if err := enc.commitStep(ctx, height); err != nil {
				return err
			}
		}
		return enc.finish(ctx)
	}
	limits := settings.Limits.normalized()
	if sameInterfaceTarget(w, settings.Scratch) {
		return fmt.Errorf("%w: row output scratch aliases output", ErrInvalid)
	}
	if limits.MaxMemoryBytes <= jxrRowStageCopyBytes {
		return &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrRowStageCopyBytes + 1, Detail: "sequential row copy buffer"}
	}
	var stage io.ReadWriteSeeker
	if settings.SeparateAlpha && !settings.Uncompressed {
		alphaLen, err := jxrAlphaRowByteLength(info)
		if err != nil {
			return err
		}
		origScratch := settings.Scratch
		settings.Scratch = &jxrOffsetReadWriteSeeker{target: origScratch, base: 0}
		stage = &jxrOffsetReadWriteSeeker{target: origScratch, base: int64(alphaLen)}
	} else {
		if err := seekJXRImageOutput(ctx, settings.Scratch, 0); err != nil {
			return err
		}
		stage = settings.Scratch
		settings.Scratch = nil
	}
	limits.MaxMemoryBytes -= jxrRowStageCopyBytes
	settings.Limits = limits
	if err := encodeJXRRows(ctx, stage, info, read, &settings, file); err != nil {
		return err
	}
	endPos, err := stage.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	return copyJXRStageToWriter(ctx, w, stage, endPos)
}

// jxrAlphaRowByteLength 计算单通道透明度原始行样本暂存所需的精确字节长度
// 入参: info 原始图像描述
// 返回: uint64 精确暂存字节数, error 元数据或容量错误
func jxrAlphaRowByteLength(info Info) (uint64, error) {
	count := len(info.Components)
	if count < 1 || count > rasterMaxComponents+1 || info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || uint64(info.Bounds.Max.X) > 1<<32 || uint64(info.Bounds.Max.Y) > 1<<32 {
		return 0, fmt.Errorf("%w: row source geometry", ErrInvalid)
	}
	if !info.Alpha || count < 2 || info.Channels != 0 && info.Channels != count-1 {
		return 0, fmt.Errorf("%w: row alpha scratch metadata", ErrInvalid)
	}
	comp := info.Components[count-1]
	sampleBytes := uint64(2)
	if comp.Precision == 32 {
		sampleBytes = 4
	}
	dx, dy := uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy())
	if dx > uint64(math.MaxInt64)/sampleBytes/dy {
		return 0, &LimitError{Resource: "row alpha scratch extent", Limit: uint64(math.MaxInt64), Required: jxrSaturatedMul(jxrSaturatedMul(dx, dy), sampleBytes)}
	}
	return dx * dy * sampleBytes, nil
}

// newRowEncoder 校验并初始化推式流式行编码器的全部状态、环形行缓存、输出暂存及码流前缀
// 入参: ctx 已规范化Context, w 目标输出流, info 完整原生图像元数据, opts 可选编码选项
// 返回: *RowEncoder 推式行编码器, error 参数、格式、预算、取消或输出错误
func newRowEncoder(ctx context.Context, w io.Writer, info Info, opts *EncodeOptions) (*RowEncoder, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, fmt.Errorf("%w: nil row encoder output", ErrInvalid)
	}
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return nil, err
	}
	file := settings.Format != FormatCodestream
	if file && info.ImageType&ImageTypePreview != 0 {
		return nil, fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	if file {
		settings.Format = FormatJXR
	} else {
		settings.Format = FormatCodestream
	}
	limits := settings.Limits.normalized()
	maxBytes := settings.MaxOutputBytes
	if maxBytes == 0 {
		maxBytes = 64 << 20
	}
	if file {
		maxBytes = min(maxBytes, uint64(math.MaxUint32))
	}
	settings.MaxOutputBytes = maxBytes
	count := len(info.Components)
	metadata := uint64(4096) + uint64(count)*512
	if count < 1 || count > rasterMaxComponents+1 || info.Bounds.Empty() || info.Bounds.Min != (image.Point{}) || uint64(info.Bounds.Max.X) > 1<<32 || uint64(info.Bounds.Max.Y) > 1<<32 {
		return nil, fmt.Errorf("%w: row encoder geometry", ErrInvalid)
	}
	if metadata >= limits.MaxMemoryBytes {
		return nil, &LimitError{Resource: "row encoder descriptors", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(metadata, 1)}
	}
	if uint64(info.Bounds.Dx()) > limits.MaxPixels/uint64(info.Bounds.Dy()) {
		return nil, &LimitError{Resource: "row encoder pixels", Limit: limits.MaxPixels, Required: jxrSaturatedMul(uint64(info.Bounds.Dx()), uint64(info.Bounds.Dy()))}
	}
	cleanInfo := jxrStripContainerMetadata(info)
	if file {
		cleanInfo = jxrStripContainerMetadata(jxrWithContainerMetadata(info, &settings))
	}
	cleanInfo.Components = cloneComponentInfo(info.Components)
	cleanInfo.ColorEncoding = cloneColorEncoding(cleanInfo.ColorEncoding)
	if err := normalizeRasterInfo(&cleanInfo); err != nil {
		return nil, err
	}
	formatInfo, err := rasterPixelFormat(cleanInfo)
	if err != nil {
		return nil, err
	}
	enc := &RowEncoder{
		finalWriter:  w,
		info:         cleanInfo,
		file:         file,
		separate:     settings.SeparateAlpha,
		uncompressed: settings.Uncompressed,
		width:        cleanInfo.Bounds.Dx(),
		height:       cleanInfo.Bounds.Dy(),
		step:         1,
	}
	defer enc.unbindContext()
	for _, comp := range cleanInfo.Components {
		enc.step = max(enc.step, int(comp.YStep))
	}
	if ws, ok := w.(io.WriteSeeker); ok {
		enc.stageWriter = ws
	} else if settings.Scratch != nil {
		if sameInterfaceTarget(w, settings.Scratch) {
			return nil, fmt.Errorf("%w: row encoder scratch aliases output", ErrInvalid)
		}
		if limits.MaxMemoryBytes <= jxrRowStageCopyBytes+metadata {
			return nil, &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrRowStageCopyBytes + metadata + 1, Detail: "row encoder copy buffer"}
		}
		limits.MaxMemoryBytes -= jxrRowStageCopyBytes
		if settings.SeparateAlpha && !settings.Uncompressed {
			alphaLen, err := jxrAlphaRowByteLength(cleanInfo)
			if err != nil {
				return nil, err
			}
			origScratch := settings.Scratch
			settings.Scratch = &jxrOffsetReadWriteSeeker{target: origScratch, base: 0}
			stage := &jxrOffsetReadWriteSeeker{target: origScratch, base: int64(alphaLen)}
			enc.stageWriter, enc.stageReader = stage, stage
		} else {
			if err := seekJXRImageOutput(ctx, settings.Scratch, 0); err != nil {
				return nil, err
			}
			enc.stageWriter, enc.stageReader = settings.Scratch, settings.Scratch
			settings.Scratch = nil
		}
	} else {
		mem := &jxrMemWriteSeeker{ctx: ctx, maxBytes: maxBytes, maxMemory: limits.MaxMemoryBytes}
		enc.memStage = mem
		enc.stageWriter, enc.stageReader = mem, mem
	}
	settings.Limits = limits
	if settings.Uncompressed {
		if err := enc.initUncompressed(ctx, info, settings); err != nil {
			return nil, err
		}
		return enc, nil
	}
	settings = jxrResolveEncodeOptions(settings)
	if settings.TileSize == (image.Point{}) {
		tileSize, err := jxrDefaultRowTileSize(cleanInfo, formatInfo, settings)
		if err != nil {
			return nil, err
		}
		settings.TileSize = tileSize
	}
	if err := enc.initCompressed(ctx, info, formatInfo, settings, metadata); err != nil {
		return nil, err
	}
	return enc, nil
}

// initUncompressed 初始化无压缩模式的推式行编码器，非转置顺序流无需定位时直接向最终Writer写出以达成常量行内存
// 入参: ctx 已规范化Context, origInfo 含容器元数据的原始描述, settings 已校验无压缩选项
// 返回: error 格式、资源、取消或I/O错误
func (e *RowEncoder) initUncompressed(ctx context.Context, origInfo Info, settings EncodeOptions) error {
	info, retained, err := attachJXRTileMetadata(ctx, e.info, origInfo, settings)
	if err != nil {
		return err
	}
	e.info = info
	limits := settings.Limits.normalized()
	maxBytes := min(settings.MaxOutputBytes, uint64(math.MaxUint32))
	prefix, err := prepareJXRUncompressedPrefix(ctx, info, settings, maxBytes, limits.MaxMemoryBytes-retained)
	if err != nil {
		return err
	}
	rowInfo, rowRasterWork, err := jxrRegionInfo(jxrStripContainerMetadata(info), image.Rect(0, 0, e.width, e.step))
	if err != nil {
		return err
	}
	baseHeld := retained + uint64(cap(prefix.header))
	e.uncLayout, e.uncTransform = prefix.layout, prefix.transform
	e.uncMainStart, e.uncAlphaAt, e.uncTotal, e.uncPadMain = prefix.mainLocal, prefix.alphaLocal, prefix.totalLocal, prefix.padMain
	if prefix.transform&4 != 0 {
		position, err := e.stageWriter.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		if position != 0 {
			return fmt.Errorf("%w: uncompressed row encoder output position", ErrInvalid)
		}
		e.uncHeader = prefix.header
		rawTotal := prefix.layout.mainBytes + prefix.layout.alphaBytes
		if rawTotal > uint64(^uint(0)>>1) {
			return &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: rawTotal, Detail: "uncompressed transposed payload capacity"}
		}
		if _, ok := jxrWorkTotal(limits.MaxMemoryBytes, baseHeld, rowRasterWork, rawTotal); !ok {
			return &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(baseHeld, jxrSaturatedAdd(rowRasterWork, rawTotal)), Detail: "uncompressed transposed row encoder memory"}
		}
		ring, err := newRasterContext(ctx, rowInfo, Limits{MaxPixels: limits.MaxPixels, MaxMemoryBytes: rowRasterWork})
		if err != nil {
			return err
		}
		e.ring = ring
		e.initRowView(rowInfo)
		e.uncMainData = make([]byte, int(prefix.layout.mainBytes))
		if prefix.layout.separateAlpha {
			e.uncAlphaData = make([]byte, int(prefix.layout.alphaBytes))
		}
		return ctx.Err()
	}
	bufWork := prefix.layout.rowBufferWork()
	if bufWork > uint64(^uint(0)>>1) {
		return &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: bufWork, Detail: "uncompressed row encoder buffer capacity"}
	}
	held, ok := jxrWorkTotal(limits.MaxMemoryBytes, baseHeld, rowRasterWork, bufWork)
	if !ok {
		return &LimitError{Resource: "max memory bytes", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(baseHeld, jxrSaturatedAdd(rowRasterWork, bufWork)), Detail: "uncompressed row encoder memory"}
	}
	e.uncNeedSeek = prefix.layout.planar || prefix.layout.separateAlpha || prefix.transform&1 != 0
	if e.memStage != nil && !e.uncNeedSeek {
		e.uncDirect = true
		e.memStage = nil
		e.stageWriter = nil
		e.stageReader = nil
	} else {
		position, err := e.stageWriter.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		if position != 0 {
			return fmt.Errorf("%w: uncompressed row encoder output position", ErrInvalid)
		}
		if e.memStage != nil {
			e.memStage.maxMemory = limits.MaxMemoryBytes - held
			if prefix.totalLocal > e.memStage.maxMemory {
				return &LimitError{Resource: "max memory bytes", Limit: e.memStage.maxMemory, Required: prefix.totalLocal, Detail: "sequential row output stage memory"}
			}
		}
	}
	ring, err := newRasterContext(ctx, rowInfo, Limits{MaxPixels: limits.MaxPixels, MaxMemoryBytes: rowRasterWork})
	if err != nil {
		return err
	}
	e.ring = ring
	e.initRowView(rowInfo)
	e.uncRowBuf = make([]byte, int(prefix.layout.rowBytes))
	if !prefix.layout.planar && prefix.layout.separateAlpha {
		e.uncAlphaBuf = make([]byte, int(prefix.layout.alphaRowBytes))
	}
	dst := io.Writer(e.stageWriter)
	if e.uncDirect {
		dst = e.finalWriter
	}
	if err := writeEncoderOutput(ctx, dst, prefix.header); err != nil {
		return err
	}
	if e.uncNeedSeek {
		var zeros [32768]byte
		for remaining := prefix.totalLocal - uint64(len(prefix.header)); remaining != 0; {
			chunk := min(remaining, uint64(len(zeros)))
			if err := writeEncoderOutput(ctx, e.stageWriter, zeros[:int(chunk)]); err != nil {
				return err
			}
			remaining -= chunk
		}
	}
	return ctx.Err()
}

// initCompressed 初始化压缩模式的推式行编码器，分配环形行缓存、码流前缀与可复用工作槽
// 入参: ctx 已规范化Context, origInfo 含容器元数据的原始描述, formatInfo 像素格式, settings 已解析编码选项, metadata 固定描述额度
// 返回: error 布局、预算、取消或I/O错误
func (e *RowEncoder) initCompressed(ctx context.Context, origInfo Info, formatInfo jxrPixelFormat, settings EncodeOptions, metadata uint64) error {
	limits := settings.Limits.normalized()
	cleanInfo := e.info
	if settings.SeparateAlpha {
		var held uint64
		var err error
		e.alphaStorage, held, err = makeJXRAlphaRowStorage(cleanInfo, settings.Scratch, limits.MaxMemoryBytes-metadata)
		if err != nil {
			return err
		}
		if e.alphaStorage.storage != nil {
			if sameInterfaceTarget(e.stageWriter, e.alphaStorage.storage) {
				return fmt.Errorf("%w: row alpha scratch aliases output", ErrInvalid)
			}
			if err := seekJXRImageOutput(ctx, e.alphaStorage.storage, 0); err != nil {
				return err
			}
		}
		limits.MaxMemoryBytes -= held
	}
	ringHeight, err := jxrRingHeight(cleanInfo, formatInfo.color, settings)
	if err != nil {
		return err
	}
	cacheInfo, ringWork, err := jxrRegionInfo(cleanInfo, image.Rect(0, 0, e.width, ringHeight))
	if err != nil {
		return err
	}
	if ringWork >= limits.MaxMemoryBytes-metadata {
		return &LimitError{Resource: "row ring storage", Limit: limits.MaxMemoryBytes, Required: jxrSaturatedAdd(metadata, jxrSaturatedAdd(ringWork, 1))}
	}
	limits.MaxMemoryBytes -= ringWork + metadata
	settings.Limits = limits
	info, retained, err := attachJXRTileMetadata(ctx, cleanInfo, origInfo, settings)
	if err != nil {
		return err
	}
	e.info = info
	var source jxrStream
	var streamOpts jxrStreamOutput
	var codedInfo Info
	if settings.SeparateAlpha {
		input, err := makeJXRSeparateTileInput(ctx, info, settings, retained)
		if err != nil {
			return err
		}
		e.separateIn = input
		header, offsets, err := makeJXRFilePrefixParts(ctx, input.file.info, &input.file.layout[0], &input.file.layout[1], input.file.maxPixels, min(input.file.maximum, uint64(math.MaxUint32)), input.file.prefixMemory)
		if err != nil {
			return err
		}
		position, err := e.stageWriter.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		if position != 0 {
			return fmt.Errorf("%w: incremental separate alpha output position", ErrInvalid)
		}
		if uint64(len(header)) >= min(input.file.maximum, uint64(math.MaxUint32)) {
			return &LimitError{Resource: "incremental separate alpha file capacity", Limit: min(input.file.maximum, uint64(math.MaxUint32)), Required: jxrSaturatedAdd(uint64(len(header)), 1)}
		}
		if e.memStage != nil {
			e.memStage.maxMemory = input.file.prefixMemory
		}
		if err := writeEncoderOutput(ctx, e.stageWriter, header); err != nil {
			return err
		}
		e.separateOff = offsets
		e.mainStart = uint64(len(header))
		e.fileMaxBytes = min(input.file.maximum, uint64(math.MaxUint32))
		source = input.file.layout[0]
		streamOpts = input.output[0]
		streamOpts.maxBytes = e.fileMaxBytes - e.mainStart
		streamOpts.rowOrder = true
		codedInfo = input.info[0]
	} else {
		metadataRaster := Raster{info: info}
		planSource, output, err := makeJXREncodingPlanRetained(ctx, &metadataRaster, &settings, retained)
		if err != nil {
			return jxrPlanLimitError(err, output.stream.maxMemory, retained)
		}
		source = planSource
		streamOpts = output.stream
		streamOpts.rowOrder = true
		codedInfo = info
		if e.file {
			settingsWork, err := streamOpts.tileWork(ctx, streamOpts.maxMemory)
			if err != nil {
				return err
			}
			working := source.headerWork() + settingsWork
			if retained >= streamOpts.maxMemory || working >= streamOpts.maxMemory-retained {
				return fmt.Errorf("%w: incremental file metadata memory", ErrLimit)
			}
			maxFileBytes := min(streamOpts.maxBytes, uint64(math.MaxUint32))
			prefixMemory := streamOpts.maxMemory - retained - working
			header, sizeAt, err := makeJXRFilePrefix(ctx, info, &source, streamOpts.maxPixels, maxFileBytes, prefixMemory)
			if err != nil {
				return err
			}
			if uint64(len(header)) >= maxFileBytes {
				return fmt.Errorf("%w: incremental file codestream capacity", ErrLimit)
			}
			position, err := e.stageWriter.Seek(0, io.SeekCurrent)
			if err != nil {
				return err
			}
			if position != 0 {
				return fmt.Errorf("%w: incremental file output position", ErrInvalid)
			}
			if e.memStage != nil {
				e.memStage.maxMemory = prefixMemory
			}
			if err := writeEncoderOutput(ctx, e.stageWriter, header); err != nil {
				return err
			}
			e.fileSizeAt = sizeAt
			e.mainStart = uint64(len(header))
			e.fileMaxBytes = maxFileBytes
			streamOpts.maxBytes = maxFileBytes - e.mainStart
		}
		streamOpts.maxMemory -= retained
	}
	if err := e.initStreamState(ctx, codedInfo, &source, streamOpts); err != nil {
		return err
	}
	ring, err := newRasterContext(ctx, cacheInfo, Limits{MaxPixels: uint64(cacheInfo.Bounds.Dx()) * uint64(cacheInfo.Bounds.Dy()), MaxMemoryBytes: ringWork})
	if err != nil {
		return err
	}
	e.ring = ring
	e.initRowView(cacheInfo)
	return ctx.Err()
}

// initRowView 初始化复用于单行写入的零分配切片视图骨架
// 入参: info 行或环形缓存元数据
func (e *RowEncoder) initRowView(info Info) {
	e.rowStorage.info = info
	e.rowStorage.info.Components = cloneComponentInfo(info.Components)
	e.rowStorage.planes = make([]imageComponents, len(info.Components))
}

// initStreamState 规划码流全部块的最大工作预算、写出码流头与占位索引并预分配复用的单块或并发工作槽
// 入参: ctx Context, codedInfo 当前主码流原生描述, source 主码流头部, options 码流输出预算与参数
// 返回: error 布局、预算、取消或I/O错误
func (e *RowEncoder) initStreamState(ctx context.Context, codedInfo Info, source *jxrStream, options jxrStreamOutput) error {
	streamPlan, err := planJXRHardTileStream(ctx, codedInfo, source, options.maxMemory)
	if err != nil {
		return err
	}
	ps, _, err := beginJXRPacketStream(ctx, e.stageWriter, source, options, streamPlan.retained+streamPlan.tileMax)
	if err != nil {
		return err
	}
	rowTiles := len(source.header.columns) - 1
	if source.header.transform&4 != 0 {
		rowTiles = len(source.header.rows) - 1
	}
	freePacket := ps.budget.limit
	if ps.budget.used < freePacket {
		freePacket -= ps.budget.used
	} else {
		freePacket = 0
	}
	slots := jxrTileCoderSlots(options.workers, rowTiles, streamPlan.tileMax, freePacket)
	if slots > 1 {
		ps.budget.limit -= uint64(slots-1) * streamPlan.tileMax
		e.sharedBudget.limit = ps.budget.limit
		if e.memStage != nil {
			e.memStage.maxMemory = min(e.memStage.maxMemory, ps.budget.limit)
		}
	}
	coders := make([]jxrTileCoder, slots)
	for slot := range slots {
		var budget *jxrWriteBudget
		if slots == 1 {
			budget = &ps.budget
		} else {
			coders[slot].budget = jxrWriteBudget{limit: ps.budget.limit, shared: &e.sharedBudget}
		}
		if err := coders[slot].init(ctx, codedInfo, &streamPlan, &ps.stream, &options, ps.maximum-ps.total, budget); err != nil {
			return err
		}
	}
	e.packetStream = ps
	e.streamPlan = streamPlan
	e.coders = coders
	e.nextOrdinal = 0
	return e.prepareNextTilePlan(codedInfo)
}

// prepareNextTilePlan 预计算下一个显示序待刷出分块的编码序号与所需显示窗口
// 入参: codedInfo 主码流原生描述
// 返回: error 几何规划错误
func (e *RowEncoder) prepareNextTilePlan(codedInfo Info) error {
	if e.nextOrdinal >= e.packetStream.tiles {
		return nil
	}
	tile := e.nextOrdinal
	if e.packetStream.stream.header.transform&5 != 0 {
		tile = jxrDisplayTileIndex(&e.packetStream.stream.header, e.nextOrdinal)
	}
	plan, err := planJXRHardTile(codedInfo, &e.packetStream.stream, tile)
	if err != nil {
		return err
	}
	e.nextTile, e.nextPlan = tile, plan
	return nil
}

// bindRingRowView 将环形缓存当前槽位映射为零起点可写单步长临时视图并清零
// 入参: height 本次写入行高（须不超过e.step）
// 返回: *Raster 指向环形缓存当前槽位的临时视图
func (e *RowEncoder) bindRingRowView(height int) *Raster {
	return jxrBindRingRowView(e.ring, &e.rowStorage, &e.rowView, e.pushed, height)
}

// writeRowFast 共用十二种零分配单行快路径写入，校验状态后直写环形缓存并立即刷出已就绪分块
// 入参: ctx Context, op 公开操作名, fill 单行写入回调
// 返回: error 状态、格式、参数、编码、取消或写入错误
func (e *RowEncoder) writeRowFast(ctx context.Context, op string, fill func(*Raster) error) error {
	if e == nil {
		return wrapJXRError(op, ErrClosed)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return wrapJXRError(op, ErrClosed)
	}
	if e.failed != nil {
		return wrapJXRError(op, e.failed)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.step != 1 {
		return wrapJXRError(op, fmt.Errorf("%w: fast row write requires full-rate sampling", ErrUnsupported))
	}
	if e.pushed >= e.height {
		return wrapJXRError(op, fmt.Errorf("%w: row encoder exceeded image height", ErrInvalid))
	}
	row := e.bindRingRowView(1)
	if err := fill(row); err != nil {
		return wrapJXRError(op, err)
	}
	if err := e.commitStep(ctx, 1); err != nil {
		e.failed = err
		return wrapJXRError(op, err)
	}
	return nil
}

// writeRowRaster 校验并逐行步将原生行带或条带拷贝至环形缓存，每步立即触发已就绪分块编码刷出
// 入参: ctx 已规范化Context, src 源原生行带
// 返回: error 参数、格式、状态、编码、取消或写入错误
func (e *RowEncoder) writeRowRaster(ctx context.Context, src *Raster) error {
	if e.closed {
		return ErrClosed
	}
	if e.failed != nil {
		return e.failed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if src == nil || src.Bounds().Empty() {
		return fmt.Errorf("%w: nil or empty row raster", ErrInvalid)
	}
	srcInfo := src.info
	if srcInfo.ColorSpace != e.info.ColorSpace || srcInfo.Channels != e.info.Channels || srcInfo.Alpha != e.info.Alpha || srcInfo.Premultiplied != e.info.Premultiplied || len(srcInfo.Components) != len(e.info.Components) {
		return fmt.Errorf("%w: row raster format mismatch", ErrInvalid)
	}
	bounds := src.Bounds()
	rows := bounds.Dy()
	if bounds.Dx() != e.width || rows%e.step != 0 || rows > e.height-e.pushed {
		return fmt.Errorf("%w: row raster dimensions", ErrInvalid)
	}
	for index, comp := range srcInfo.Components {
		want := e.info.Components[index]
		if !jxrSameSampleEncoding(comp, want) || comp.XStep != want.XStep || comp.YStep != want.YStep || comp.Bounds.Dx() != want.Bounds.Dx() || comp.Bounds.Dy() != rows/int(want.YStep) {
			return fmt.Errorf("%w: row raster component mismatch", ErrInvalid)
		}
	}
	var srcStorage [9]Component
	srcComps := srcStorage[:]
	if len(srcInfo.Components) > len(srcStorage) {
		srcComps = make([]Component, len(srcInfo.Components))
	} else {
		srcComps = srcComps[:len(srcInfo.Components)]
	}
	for index := range srcComps {
		var err error
		srcComps[index], err = src.component(index)
		if err != nil {
			return err
		}
	}
	for offset := 0; offset < rows; offset += e.step {
		if err := ctx.Err(); err != nil {
			if offset > 0 {
				e.failed = err
			}
			return err
		}
		dst := e.bindRingRowView(e.step)
		for index, comp := range dst.info.Components {
			srcComp := srcComps[index]
			srcY0 := srcComp.info.Bounds.Min.Y + offset/int(comp.YStep)
			srcX0 := srcComp.info.Bounds.Min.X
			if err := copyJXRComponentRect(ctx, &dst.planes[index], srcComp, srcX0, srcY0, comp.Bounds.Dx(), comp.Bounds.Dy()); err != nil {
				if offset > 0 {
					e.failed = err
				}
				return err
			}
		}
		if err := e.commitStep(ctx, e.step); err != nil {
			e.failed = err
			return err
		}
	}
	return nil
}

// commitStep 提交环形缓存中刚写入的单个行步长，记录独立透明度并立即刷出全部已就绪分块或无压缩行
// 入参: ctx 已规范化Context, rows 本步写入行数
// 返回: error 编码、暂存、取消或写入错误
func (e *RowEncoder) commitStep(ctx context.Context, rows int) error {
	defer e.unbindContext()
	if e.memStage != nil {
		e.memStage.ctx = ctx
	}
	if e.uncompressed {
		if err := e.commitUncompressedRow(ctx); err != nil {
			return err
		}
		e.pushed += rows
		return nil
	}
	if e.separate {
		if err := e.alphaStorage.transfer(ctx, &e.rowStorage.planes[e.info.Channels], true); err != nil {
			return err
		}
	}
	e.pushed += rows
	return e.flushReadyTiles(ctx)
}

// commitUncompressedRow 打包并写出或散射当前单步无压缩原生行
// 入参: ctx 已规范化Context
// 返回: error 打包、取消或写入错误
func (e *RowEncoder) commitUncompressedRow(ctx context.Context) error {
	cleanInfo := jxrStripContainerMetadata(e.info)
	if e.uncTransform&4 != 0 {
		return jxrScatterUncompressedRowTransposed(ctx, cleanInfo, e.rowStorage.planes, &e.uncLayout, e.uncMainData, e.uncAlphaData, e.uncTransform, e.pushed, e.step)
	}
	dst := io.Writer(e.stageWriter)
	if e.uncDirect {
		dst = e.finalWriter
	}
	return jxrWriteUncompressedRowStreamStep(ctx, dst, cleanInfo, e.rowStorage.planes, &e.uncLayout, e.uncRowBuf, e.uncAlphaBuf, e.uncTransform, e.pushed, e.step, e.uncMainStart, e.uncAlphaAt, e.uncNeedSeek)
}

// flushReadyTiles 按显示行序编码并写出全部所需显示行均已到达的分块，多列就绪且预算充足时按批并发编码并按序提交
// 入参: ctx 已规范化Context
// 返回: error 变换、编码、预算、取消或写入错误
func (e *RowEncoder) flushReadyTiles(ctx context.Context) error {
	codedInfo := e.info
	if e.separate {
		codedInfo = e.separateIn.info[0]
	}
	ps := e.packetStream
	slots := len(e.coders)
	capacity := e.ring.Bounds().Dy()
	var batchBuf [16]jxrTileJob
	batch := batchBuf[:0]
	if slots > len(batchBuf) {
		batch = make([]jxrTileJob, 0, slots)
	}
	for e.nextOrdinal < ps.tiles {
		if e.pushed < e.nextPlan.region.Max.Y {
			break
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		batch = batch[:0]
		for len(batch) < slots && e.nextOrdinal < ps.tiles && e.pushed >= e.nextPlan.region.Max.Y {
			if e.nextPlan.region.Min.Y < max(0, e.pushed-capacity) || e.nextPlan.region.Max.Y > e.height || e.nextPlan.region.Dy() > capacity {
				return fmt.Errorf("%w: row encoder ring window order", ErrInvalid)
			}
			batch = append(batch, jxrTileJob{tile: e.nextTile, plan: e.nextPlan})
			e.nextOrdinal++
			if err := e.prepareNextTilePlan(codedInfo); err != nil {
				return err
			}
		}
		if slots > 1 && e.memStage != nil {
			memCap := uint64(cap(e.memStage.data))
			if memCap >= ps.budget.limit {
				return &LimitError{Resource: "max memory bytes", Limit: ps.budget.limit, Required: jxrSaturatedAdd(memCap, 1), Detail: "sequential row output stage memory"}
			}
			sharedLimit := ps.budget.limit - memCap
			if e.sharedBudget.used.Load() > sharedLimit {
				for slot := range slots {
					e.coders[slot].resetPackets(e.streamPlan.bands)
				}
			}
			e.sharedBudget.limit = sharedLimit
			for slot := range slots {
				e.coders[slot].setPacketLimit(e.streamPlan.bands, sharedLimit)
			}
		}
		packetMax := ps.maximum - ps.total
		if len(batch) == 1 {
			coder := &e.coders[0]
			local, err := resizeJXRHardTile(ctx, codedInfo, batch[0].plan, e.streamPlan.largest.rawWork, coder.local)
			if err != nil {
				return err
			}
			coder.local = local
			if err := jxrProjectRingRegion(ctx, e.ring, local, batch[0].plan.region); err != nil {
				return err
			}
			if err := coder.encodeLocal(ctx, &ps.stream, &ps.options, &e.streamPlan, batch[0].tile, &batch[0].plan, packetMax); err != nil {
				return err
			}
		} else {
			if err := runJXRWorkers(ctx, len(batch), len(batch), func(runCtx context.Context, i int) error {
				coder := &e.coders[i]
				local, err := resizeJXRHardTile(runCtx, codedInfo, batch[i].plan, e.streamPlan.largest.rawWork, coder.local)
				if err != nil {
					return err
				}
				coder.local = local
				if err := jxrProjectRingRegion(runCtx, e.ring, local, batch[i].plan.region); err != nil {
					return err
				}
				return coder.encodeLocal(runCtx, &ps.stream, &ps.options, &e.streamPlan, batch[i].tile, &batch[i].plan, packetMax)
			}); err != nil {
				return err
			}
		}
		if slots > 1 && e.memStage != nil {
			ps.budget.used = uint64(cap(e.memStage.data)) + e.sharedBudget.used.Load()
		}
		for i := range batch {
			if err := ps.commitTile(ctx, e.stageWriter, batch[i].tile, &e.coders[i].packets); err != nil {
				return err
			}
		}
	}
	return nil
}

// finish 校验写满整帧、回填索引和容器长度字段、写出可选独立透明度并流式拷贝至顺序输出
// 入参: ctx 已规范化Context
// 返回: error 未写满行、回填、透明编码、取消或I/O错误
func (e *RowEncoder) finish(ctx context.Context) error {
	defer e.unbindContext()
	if e.failed != nil {
		return e.failed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.pushed != e.height {
		return fmt.Errorf("%w: incomplete row encoder input", ErrInvalid)
	}
	if e.memStage != nil {
		e.memStage.ctx = ctx
	}
	if e.uncompressed {
		if e.uncTransform&4 != 0 {
			dst := io.Writer(e.stageWriter)
			if e.stageReader != nil {
				dst = e.finalWriter
			}
			if err := writeEncoderOutput(ctx, dst, e.uncHeader); err != nil {
				return err
			}
			if err := writeEncoderOutput(ctx, dst, e.uncMainData); err != nil {
				return err
			}
			if e.uncPadMain {
				if err := writeEncoderOutput(ctx, dst, []byte{0}); err != nil {
					return err
				}
			}
			if e.uncLayout.separateAlpha {
				if err := writeEncoderOutput(ctx, dst, e.uncAlphaData); err != nil {
					return err
				}
			}
			if e.stageReader != nil {
				return nil
			}
		} else if e.uncNeedSeek {
			if err := seekJXRImageOutput(ctx, e.stageWriter, e.uncTotal); err != nil {
				return err
			}
		}
		return e.flushStagedToFinal(ctx)
	}
	if err := e.flushReadyTiles(ctx); err != nil {
		return err
	}
	if err := e.packetStream.finish(ctx, e.stageWriter); err != nil {
		return err
	}
	if e.separate {
		if err := e.finishSeparateAlpha(ctx); err != nil {
			return err
		}
		return e.flushStagedToFinal(ctx)
	}
	if e.file {
		endPos, err := e.stageWriter.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		if endPos < 0 || uint64(endPos) < e.mainStart || uint64(endPos)-e.mainStart > e.packetStream.options.maxBytes {
			return fmt.Errorf("%w: incremental file output extent", ErrInvalid)
		}
		if err := patchJXRFileSize(ctx, e.stageWriter, e.fileSizeAt, uint64(endPos)-e.mainStart, uint64(endPos)); err != nil {
			return err
		}
	}
	return e.flushStagedToFinal(ctx)
}

// finishSeparateAlpha 在主图码流写完后按行读回已暂存的透明度通道编码独立透明度码流并回填双流容器字段
// 入参: ctx 已规范化Context
// 返回: error 暂存、编码、回填、取消或I/O错误
func (e *RowEncoder) finishSeparateAlpha(ctx context.Context) error {
	mainEnd, err := e.stageWriter.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if mainEnd < 0 || uint64(mainEnd) < e.mainStart || uint64(mainEnd) > e.fileMaxBytes {
		return fmt.Errorf("%w: incremental separate alpha main extent", ErrInvalid)
	}
	e.coders = nil
	e.packetStream = nil
	e.sharedBudget = jxrSharedWriteBudget{}
	mainSize := uint64(mainEnd) - e.mainStart
	cursor := uint64(mainEnd)
	if cursor&1 != 0 {
		if cursor == e.fileMaxBytes {
			return &LimitError{Resource: "incremental separate alpha alignment", Limit: e.fileMaxBytes, Required: jxrSaturatedAdd(cursor, 1)}
		}
		if err := writeEncoderOutput(ctx, e.stageWriter, []byte{0}); err != nil {
			return err
		}
		cursor++
	}
	alphaStart := cursor
	if alphaStart >= e.fileMaxBytes {
		return &LimitError{Resource: "incremental separate alpha stream capacity", Limit: e.fileMaxBytes, Required: jxrSaturatedAdd(alphaStart, 1)}
	}
	if e.alphaStorage.position != e.alphaStorage.length {
		return fmt.Errorf("%w: incomplete row alpha scratch", ErrInvalid)
	}
	if e.alphaStorage.storage != nil {
		if err := seekJXRImageOutput(ctx, e.alphaStorage.storage, 0); err != nil {
			return err
		}
	}
	e.alphaStorage.position = 0
	alphaSource := jxrRowInput{
		info:   e.ring.info,
		cache:  e.ring,
		row:    e.rowStorage,
		height: e.height,
		step:   e.step,
		read: func(cbCtx context.Context, _ int, target *Raster) error {
			return e.alphaStorage.transfer(cbCtx, &target.planes[e.info.Channels], false)
		},
	}
	selected := e.separateIn.output[1]
	selected.maxBytes, selected.rowOrder = e.fileMaxBytes-alphaStart, true
	if e.memStage != nil {
		e.memStage.budget = nil
		memCap := uint64(cap(e.memStage.data))
		if memCap >= selected.maxMemory {
			return &LimitError{Resource: "max memory bytes", Limit: selected.maxMemory, Required: jxrSaturatedAdd(memCap, 1), Detail: "sequential separate alpha output stage memory"}
		}
		e.memStage.maxMemory = selected.maxMemory
		selected.maxMemory -= memCap
	}
	if err := encodeJXRProjectedTilesTo(ctx, e.stageWriter, e.separateIn.file.info, e.separateIn.info[1], &e.separateIn.file.layout[1], selected, alphaSource.load, true, e.separateIn.mirror); err != nil {
		return err
	}
	alphaEnd, err := e.stageWriter.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if alphaEnd < 0 || uint64(alphaEnd) < alphaStart || uint64(alphaEnd) > e.fileMaxBytes {
		return fmt.Errorf("%w: incremental separate alpha output extent", ErrInvalid)
	}
	return patchJXRSeparateAlphaFields(ctx, e.stageWriter, 0, e.separateOff, mainSize, alphaStart, uint64(alphaEnd)-alphaStart, uint64(alphaEnd))
}

// flushStagedToFinal 当底层目标为普通顺序Writer时将暂存数据流式拷贝至最终输出流
// 入参: ctx 已规范化Context
// 返回: error 拷贝、取消或写入错误
func (e *RowEncoder) flushStagedToFinal(ctx context.Context) error {
	if e.stageReader == nil {
		return ctx.Err()
	}
	if e.memStage != nil {
		return writeEncoderOutput(ctx, e.finalWriter, e.memStage.data)
	}
	endPos, err := e.stageWriter.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	return copyJXRStageToWriter(ctx, e.finalWriter, e.stageReader, endPos)
}

// unbindContext 清空暂存与包写入器绑定的Context，防止编码器滞留已结束请求的Context
func (e *RowEncoder) unbindContext() {
	if e.memStage != nil {
		e.memStage.ctx = nil
	}
	for index := range e.coders {
		e.coders[index].unbindContext(e.streamPlan.bands)
	}
}

// release 标记编码器关闭并彻底释放元数据、外部流对象、环形行缓存、局部变换缓冲与暂存引用
func (e *RowEncoder) release() {
	e.closed = true
	e.failed = nil
	e.finalWriter = nil
	e.stageWriter = nil
	e.stageReader = nil
	e.uncDirect = false
	if e.memStage != nil {
		e.memStage.ctx = nil
		e.memStage.data = nil
		e.memStage = nil
	}
	e.info = Info{}
	e.ring = nil
	e.rowView = Raster{}
	e.rowStorage = Raster{}
	e.alphaStorage = jxrAlphaRowStorage{}
	e.separateIn = jxrSeparateTileInput{}
	e.packetStream = nil
	e.streamPlan = jxrHardTileStreamPlan{}
	e.coders = nil
	e.sharedBudget = jxrSharedWriteBudget{}
	e.uncHeader = nil
	e.uncRowBuf = nil
	e.uncAlphaBuf = nil
	e.uncMainData = nil
	e.uncAlphaData = nil
}
