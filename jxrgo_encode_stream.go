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
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

// jxrTileSettingBytes 为双平面量化集合及宏块索引描述保留全平台容量
const jxrTileSettingBytes uint64 = 296

// jxrTileSetting 保存一个分块的双平面量化参数及精细位截断量
type jxrTileSetting struct {
	quant [2]jxrTileQuant
	trim  uint8
}

// jxrStreamOutput 保存码流写入的精度策略、分块参数和资源上限
type jxrStreamOutput struct {
	tiles     []jxrTileSetting
	rowOrder  bool
	exact     bool
	validated bool
	workers   int
	maxPixels uint64
	maxBytes  uint64
	maxMemory uint64
	budget    *jxrWriteBudget
	forward   *jxrForwardPlan
	quantWork uint64
}

// jxrPacketStream 管理可定位码流输出的头部写出、包偏移记录、前向摘要校验与索引表回填
type jxrPacketStream struct {
	stream   jxrStream
	options  jxrStreamOutput
	position int64
	indexAt  uint64
	prefix   uint64
	total    uint64
	maximum  uint64
	bands    int
	tiles    int
	budget   jxrWriteBudget
}

// writeStream 编码完整系数图像并生成实际包索引，不修改输入头部、量化参数或系数
// 入参: ctx 已规范化Context, source 图像及平面头部, options 分块参数及系数、临时缓冲、输出容量预算
// 返回: []byte 完整独立码流, error 参数、资源、精度或取消错误
func (img *jxrCoefficientImage) writeStream(ctx context.Context, source *jxrStream, options jxrStreamOutput) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if img == nil || source == nil {
		return nil, fmt.Errorf("%w: JPEG XR image output", ErrInvalid)
	}
	s := *source
	if !options.validated {
		if err := checkJXRImageHeaders(&s, options.maxPixels); err != nil {
			return nil, err
		}
		if err := img.checkOutput(&s); err != nil {
			return nil, err
		}
	}
	tiles := (len(s.header.columns) - 1) * (len(s.header.rows) - 1)
	bands := 1
	if s.header.frequency {
		bands = 4 - int(s.primary.bands)
	}
	if uint64(tiles) > uint64(^uint(0)>>1)/uint64(bands)/8 {
		return nil, fmt.Errorf("%w: JPEG XR image output packet count", ErrLimit)
	}
	if len(options.tiles) != 1 && len(options.tiles) != tiles {
		return nil, fmt.Errorf("%w: JPEG XR image output tile settings", ErrInvalid)
	}
	settingsWork, err := options.tileWork(ctx, options.maxMemory)
	if err != nil {
		return nil, err
	}
	if options.maxBytes == 0 {
		return nil, &LimitError{Resource: "max output bytes", Limit: 0, Required: 1, Detail: "JPEG XR stream output working budget"}
	}
	work, ok := jxrWorkTotal(options.maxMemory, img.outputWork(&s), uint64(tiles*bands)*8, settingsWork)
	if !ok || work >= options.maxMemory {
		required := jxrSaturatedAdd(jxrSaturatedAdd(img.outputWork(&s), uint64(tiles*bands)*8), settingsWork)
		if required <= options.maxMemory {
			required = jxrSaturatedAdd(options.maxMemory, 1)
		}
		return nil, &LimitError{Resource: "max memory bytes", Limit: options.maxMemory, Required: required, Detail: "JPEG XR stream output working budget"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.packets = make([]uint64, tiles*bands)
	if !options.validated {
		for _, setting := range options.tiles {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if setting.trim > 15 || !s.header.trim && setting.trim != 0 {
				return nil, fmt.Errorf("%w: JPEG XR image output trim", ErrInvalid)
			}
			for n, p := range [2]*jxrPlane{&s.primary, s.alpha} {
				if p != nil {
					if err := checkJXRTileQuant(p, &setting.quant[n]); err != nil {
						return nil, err
					}
				}
			}
		}
	}
	budget := &jxrWriteBudget{limit: options.maxMemory - work}
	if options.budget != nil {
		budget = options.budget
		if budget.used > budget.limit || budget.limit > options.maxMemory-work {
			return nil, &LimitError{Resource: "max memory bytes", Limit: options.maxMemory - work, Required: max(budget.used, budget.limit), Detail: "JPEG XR retained stream output budget"}
		}
	}
	writer := newJXRBitWriter(ctx, options.maxBytes, budget.limit)
	writer.budget = budget
	if writer.err != nil {
		return nil, writer.err
	}
	if !options.validated {
		if err := checkJXRProfileDeclarations(s.profiles); err != nil {
			return nil, err
		}
		if err := s.checkProfile(); err != nil {
			return nil, err
		}
	}
	if tiles > 1 && options.workers > 1 {
		if data, done, err := img.writeStreamParallel(ctx, &s, options, &writer, budget, tiles, bands); err != nil {
			return nil, err
		} else if done {
			return data, nil
		}
	}
	var packets [4]jxrBitWriter
	output := jxrTileOutput{exact: options.exact, validated: true, maxMemory: options.maxMemory}
	for band := range bands {
		packets[band] = newJXRBitWriter(ctx, options.maxBytes, budget.limit)
		packets[band].budget = budget
		if packets[band].err != nil {
			return nil, packets[band].err
		}
		output.packet[band] = &packets[band]
	}
	for n, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p != nil {
			output.macro[n].high = make([]int64, p.channels*256)
		}
	}
	for tile := range tiles {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		setting := &options.tiles[0]
		if len(options.tiles) != 1 {
			setting = &options.tiles[tile]
		}
		output.quant, output.trim = setting.quant, setting.trim
		offsets, err := img.writeTile(ctx, &s, tile, &output)
		if err != nil {
			return nil, err
		}
		copy(s.packets[tile*bands:(tile+1)*bands], offsets[:bands])
	}
	base := uint64(0)
	for band := range bands {
		if uint64(len(packets[band].data)) > writer.maxBytes-base {
			return nil, &LimitError{Resource: "max output bytes", Limit: writer.maxBytes, Required: jxrSaturatedAdd(base, uint64(len(packets[band].data))), Detail: "JPEG XR stream packet byte budget"}
		}
		for tile := range tiles {
			s.packets[tile*bands+band] += base
		}
		base += uint64(len(packets[band].data))
	}
	writeJXRStreamHeaderChecked(&writer, &s, false)
	if writer.err != nil {
		return nil, writer.err
	}
	if base > writer.maxBytes || writer.bit/8 > writer.maxBytes-base || !writer.reserve(writer.bit/8+base) {
		if writer.err != nil {
			return nil, writer.err
		}
		return nil, &LimitError{Resource: "max output bytes", Limit: writer.maxBytes, Required: jxrSaturatedAdd(writer.bit/8, base), Detail: "JPEG XR complete stream byte budget"}
	}
	for band := range bands {
		writer.appendBytes(packets[band].data)
		if writer.err != nil {
			return nil, writer.err
		}
	}
	return writer.data, nil
}

// tileWriteScratchWork 计算单个额外并发编码Goroutine所需的残差、预测行及熵状态字节数
// 返回: uint64 单并发工作区开销，无效几何或溢出返回最大值
func (s *jxrStream) tileWriteScratchWork() uint64 {
	base := s.tileScratchWork()
	if base == ^uint64(0) {
		return ^uint64(0)
	}
	work := base
	for _, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p != nil {
			var macro jxrEncodedMacro
			extra := macro.payloadWork(p.channels) + uint64(p.channels)*256*8 + 2048
			if extra > ^uint64(0)-work {
				return ^uint64(0)
			}
			work += extra
		}
	}
	return work
}

// writeStreamParallel 在共享原子输出预算内并行编码多个分块，并按原始分块和频带顺序拼接完整码流
// 入参: ctx 已规范化Context, s 已校验码流头部, options 写入选项, writer 最终流写入器, budget 共享输出预算, tiles 分块总数, bands 频带数
// 返回: []byte 完整码流, bool 是否由并行路径完成, error 非并行内存回退类编码或取消错误
func (img *jxrCoefficientImage) writeStreamParallel(ctx context.Context, s *jxrStream, options jxrStreamOutput, writer *jxrBitWriter, budget *jxrWriteBudget, tiles, bands int) ([]byte, bool, error) {
	if budget.used >= budget.limit {
		return nil, false, nil
	}
	avail := budget.limit - budget.used
	perWorker := s.tileWriteScratchWork()
	if perWorker == ^uint64(0) {
		return nil, false, nil
	}
	highWork := uint64(0)
	for _, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p != nil {
			highWork += uint64(p.channels) * 256 * 8
		}
	}
	tableWork := uint64(tiles) * (uint64(bands)*64 + 128)
	headerReserve := uint64(tiles*bands)*8 + 4096
	baseFixed, ok := jxrWorkTotal(avail, tableWork, highWork, 256, headerReserve)
	minOutputReserve := uint64(bands) * 4096
	if !ok || minOutputReserve >= avail-baseFixed {
		return nil, false, nil
	}
	perExtraWorker := perWorker + 256
	workers := resolveJXRWorkers(options.workers, tiles)
	workers = budgetJXRWorkers(workers, jxrSaturatedAdd(avail-baseFixed-minOutputReserve, perExtraWorker), perExtraWorker)
	if workers <= 1 {
		return nil, false, nil
	}
	fixedWork, ok := jxrWorkTotal(avail, tableWork, highWork, uint64(workers)*256, uint64(workers-1)*perWorker, headerReserve)
	if !ok || fixedWork >= avail {
		return nil, false, nil
	}
	sharedLimit := (avail - fixedWork) / 2
	if sharedLimit < uint64(bands)*1024 {
		return nil, false, nil
	}
	slotOutputs := make([]jxrTileOutput, workers)
	for slot := range workers {
		slotOutputs[slot] = jxrTileOutput{exact: options.exact, validated: true, maxMemory: options.maxMemory}
		for n, p := range [2]*jxrPlane{&s.primary, s.alpha} {
			if p != nil {
				slotOutputs[slot].macro[n].high = make([]int64, p.channels*256)
			}
		}
	}
	shared := &jxrSharedWriteBudget{limit: sharedLimit}
	tileData := make([][4][]byte, tiles)
	err := runJXRWorkerSlots(ctx, workers, tiles, func(runCtx context.Context, slot, tile int) error {
		out := &slotOutputs[slot]
		setting := &options.tiles[0]
		if len(options.tiles) != 1 {
			setting = &options.tiles[tile]
		}
		out.quant, out.trim = setting.quant, setting.trim
		tileBudget := jxrWriteBudget{limit: sharedLimit, shared: shared}
		var localPackets [4]jxrBitWriter
		committed := false
		defer func() {
			if !committed {
				for band := range bands {
					if old := uint64(cap(localPackets[band].data)); old != 0 {
						shared.release(old)
					}
				}
			}
		}()
		for band := range bands {
			localPackets[band] = newJXRBitWriter(runCtx, options.maxBytes, sharedLimit)
			localPackets[band].budget = &tileBudget
			if localPackets[band].err != nil {
				return localPackets[band].err
			}
			out.packet[band] = &localPackets[band]
		}
		if _, err := img.writeTile(runCtx, s, tile, out); err != nil {
			return err
		}
		for band := range bands {
			tileData[tile][band] = localPackets[band].data
		}
		committed = true
		return nil
	})
	slotOutputs = nil
	if err != nil {
		if !isJXRMaxOutputBytesError(err) && shared.exceeded.Load() && errors.Is(err, ErrLimit) && ctx.Err() == nil {
			return nil, false, nil
		}
		return nil, true, err
	}
	base := uint64(0)
	for band := range bands {
		for tile := range tiles {
			s.packets[tile*bands+band] = base
			length := uint64(len(tileData[tile][band]))
			if length > writer.maxBytes-base {
				return nil, true, &LimitError{Resource: "max output bytes", Limit: writer.maxBytes, Required: jxrSaturatedAdd(base, length), Detail: "JPEG XR stream packet byte budget"}
			}
			base += length
		}
	}
	tileHeld := tableWork + shared.used.Load()
	if tileHeld > budget.limit-budget.used {
		clear(s.packets)
		return nil, false, nil
	}
	budget.used += tileHeld
	defer func() { budget.used -= tileHeld }()
	writeJXRStreamHeaderChecked(writer, s, false)
	if writer.err != nil {
		if !writer.maxBytesExceeded && errors.Is(writer.err, ErrLimit) && ctx.Err() == nil {
			writer.reset()
			clear(s.packets)
			return nil, false, nil
		}
		return nil, true, writer.err
	}
	if base > writer.maxBytes || writer.bit/8 > writer.maxBytes-base {
		return nil, true, &LimitError{Resource: "max output bytes", Limit: writer.maxBytes, Required: jxrSaturatedAdd(writer.bit/8, base), Detail: "JPEG XR complete stream byte budget"}
	}
	if !writer.reserve(writer.bit/8 + base) {
		if writer.err != nil {
			if !writer.maxBytesExceeded && errors.Is(writer.err, ErrLimit) && ctx.Err() == nil {
				writer.reset()
				clear(s.packets)
				return nil, false, nil
			}
			return nil, true, writer.err
		}
		return nil, true, &LimitError{Resource: "max output bytes", Limit: writer.maxBytes, Required: jxrSaturatedAdd(writer.bit/8, base), Detail: "JPEG XR complete stream byte budget"}
	}
	for band := range bands {
		for tile := range tiles {
			writer.appendBytes(tileData[tile][band])
			tileData[tile][band] = nil
			if writer.err != nil {
				if !writer.maxBytesExceeded && errors.Is(writer.err, ErrLimit) && ctx.Err() == nil {
					writer.reset()
					clear(s.packets)
					return nil, false, nil
				}
				return nil, true, writer.err
			}
		}
	}
	return writer.data, true, nil
}

// tileWork 计算公开配置、分块设置及量化和索引数组，包含切片容量范围内未使用的设置
// 入参: ctx Context, maximum 工作内存上限
// 返回: uint64 设置存储预算, error 资源或取消错误
func (o *jxrStreamOutput) tileWork(ctx context.Context, maximum uint64) (uint64, error) {
	if o.quantWork > maximum || uint64(cap(o.tiles)) > (maximum-o.quantWork)/jxrTileSettingBytes {
		return 0, fmt.Errorf("%w: JPEG XR tile settings budget", ErrLimit)
	}
	work := o.quantWork + uint64(cap(o.tiles))*jxrTileSettingBytes
	for _, setting := range o.tiles[:cap(o.tiles)] {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		for n := range setting.quant {
			amount := setting.quant[n].quantWork()
			if amount > maximum-work {
				return 0, fmt.Errorf("%w: JPEG XR retained tile quantization", ErrLimit)
			}
			work += amount
		}
	}
	return work, nil
}

// outputWork 计算系数、头部、预测行、残差及已持有切片多余容量的编码工作字节数
// 入参: s 已校验的图像和平面头部
// 返回: uint64 不包含输出包缓冲和外部原始图像的工作容量
func (img *jxrCoefficientImage) outputWork(s *jxrStream) uint64 {
	work := s.coefficientWork()
	for _, p := range [2]*jxrCoefficients{&img.primary, img.alpha} {
		if p != nil {
			var macro jxrEncodedMacro
			for _, term := range [][2]uint64{
				{macro.payloadWork(p.plane.channels), 1}, {2048, 1},
				{uint64(cap(p.low) - len(p.low)), 8},
				{uint64(cap(p.high) - len(p.high)), 8},
				{uint64(cap(p.info) - len(p.info)), 6},
			} {
				if term[0] > (^uint64(0)-work)/term[1] {
					return ^uint64(0)
				}
				work += term[0] * term[1]
			}
		}
	}
	for _, term := range [][2]uint64{
		{uint64(cap(s.header.columns) - len(s.header.columns)), 4},
		{uint64(cap(s.header.rows) - len(s.header.rows)), 4},
		{uint64(cap(s.packets) - len(s.packets)), 8},
		{uint64(cap(s.profiles) - len(s.profiles)), 4},
	} {
		if term[0] > (^uint64(0)-work)/term[1] {
			return ^uint64(0)
		}
		work += term[0] * term[1]
	}
	return work
}

// checkOutput 校验变换系数与输出头部的布局，允许独立选择包布局及显示方向
// 入参: s 所属码流头部
// 返回: error 参数、字段或系数容量错误
func (img *jxrCoefficientImage) checkOutput(s *jxrStream) error {
	if img == nil || s == nil {
		return fmt.Errorf("%w: JPEG XR output coefficient image", ErrInvalid)
	}
	h := &s.header
	if err := checkJXRHeader(h, ^uint64(0)); err != nil {
		return err
	}
	old := &img.header
	if old.width != h.width || old.height != h.height || old.top != h.top || old.left != h.left || old.bottom != h.bottom || old.right != h.right || old.color != h.color || old.depth != h.depth || old.overlap != h.overlap || old.hardTiles != h.hardTiles || old.longWord != h.longWord || old.redBlue != h.redBlue || !slices.Equal(old.columns, h.columns) || !slices.Equal(old.rows, h.rows) {
		return fmt.Errorf("%w: JPEG XR output coefficient header", ErrInvalid)
	}
	if h.alpha != (s.alpha != nil) || (img.alpha != nil) != (s.alpha != nil) || s.alpha != nil && s.alpha.bands < s.primary.bands {
		return fmt.Errorf("%w: JPEG XR output coefficient alpha", ErrInvalid)
	}
	count := uint64(h.columns[len(h.columns)-1]) * uint64(h.rows[len(h.rows)-1])
	descriptors := [2]*jxrPlane{&s.primary, s.alpha}
	for n, p := range [2]*jxrCoefficients{&img.primary, img.alpha} {
		if p == nil {
			continue
		}
		if err := checkJXRPlaneHeader(h, &p.plane, n != 0); err != nil {
			return err
		}
		if !p.plane.equal(*descriptors[n]) || p.longWord != h.longWord || uint64(len(p.info)) != count || uint64(len(p.low)) != count*uint64(p.plane.channels)*16 || uint64(len(p.high)) != count*uint64(p.plane.channels)*256 {
			return fmt.Errorf("%w: JPEG XR output coefficient layout", ErrInvalid)
		}
	}
	return nil
}

// encodeJXRCodestreamTo 共用独立码流和文件内逐包写出，输入布局与选项已由公开入口校验
// 入参: ctx 已规范化Context, w 当前位置为码流起点的输出, raster 原生图像, layout 编码布局, options 含原图的资源限制
// 返回: error 精度、资源、取消、定位或写入错误
func encodeJXRCodestreamTo(ctx context.Context, w io.WriteSeeker, raster *Raster, layout jxrStream, options jxrStreamOutput) error {
	retained, err := jxrRasterWork(raster, options.maxMemory)
	if err != nil {
		return err
	}
	if retained >= options.maxMemory {
		return &LimitError{Resource: "max memory bytes", Limit: options.maxMemory, Required: jxrSaturatedAdd(retained, 1), Detail: "incremental source memory"}
	}
	available := options.maxMemory - retained
	s, err := newJXRInputStream(ctx, raster, layout, options.maxPixels, available)
	if err != nil {
		return err
	}
	if len(s.header.columns) > 2 || len(s.header.rows) > 2 {
		options.maxMemory = available
		return encodeJXRHardTileSourceTo(ctx, w, raster.info, &s, options, func(plan jxrHardTileInput, maximum uint64, local *Raster) (*Raster, error) {
			return copyJXRHardTile(ctx, raster, plan, maximum, local)
		})
	}
	state, err := forwardJXRNative(ctx, raster, &s, &options, available)
	if err != nil {
		return err
	}
	options.maxMemory = available
	return state.writeStreamTo(ctx, w, &s, options)
}

// writeStreamTo 共用实际分块包编码并回填固定宽度索引，完整系数保持只读且包缓冲逐块复用
// 入参: ctx 已规范化Context, w 可定位输出, source 图像头部, options 分块参数及除原图外的容量限制
// 返回: error 格式、精度、资源、取消、定位或写入错误
func (state *jxrCoefficientImage) writeStreamTo(ctx context.Context, w io.WriteSeeker, source *jxrStream, options jxrStreamOutput) error {
	if state == nil || source == nil {
		return fmt.Errorf("%w: incremental stream parameters", ErrInvalid)
	}
	if !options.validated {
		if err := state.checkOutput(source); err != nil {
			return err
		}
	}
	ps, cleanup, err := beginJXRPacketStream(ctx, w, source, options, state.outputWork(source))
	if err != nil {
		return err
	}
	defer cleanup()
	var packets [4]jxrBitWriter
	output := jxrTileOutput{exact: options.exact, validated: true, maxMemory: options.maxMemory}
	for band := range ps.bands {
		packets[band] = newJXRBitWriter(ctx, ps.maximum-ps.total, ps.budget.limit)
		packets[band].budget = &ps.budget
		if packets[band].err != nil {
			return packets[band].err
		}
		output.packet[band] = &packets[band]
	}
	for index, plane := range [2]*jxrPlane{&ps.stream.primary, ps.stream.alpha} {
		if plane != nil {
			output.macro[index].high = make([]int64, plane.channels*256)
		}
	}
	for ordinal := range ps.tiles {
		if err := ctx.Err(); err != nil {
			return err
		}
		tile := ordinal
		if options.rowOrder && ps.stream.header.transform&5 != 0 {
			tile = jxrDisplayTileIndex(&ps.stream.header, ordinal)
		}
		setting := &options.tiles[0]
		if len(options.tiles) != 1 {
			setting = &options.tiles[tile]
		}
		output.quant, output.trim = setting.quant, setting.trim
		for band := range ps.bands {
			packet := &packets[band]
			clear(packet.data)
			packet.data, packet.bit, packet.maxBytes = packet.data[:0], 0, ps.maximum-ps.total
		}
		if _, err := state.writeTile(ctx, source, tile, &output); err != nil {
			return err
		}
		if err := ps.commitTile(ctx, w, tile, &packets); err != nil {
			return err
		}
	}
	return ps.finish(ctx, w)
}

// beginJXRPacketStream 校验预算并写出码流头与占位索引表，返回可复用的增量包提交状态
// 入参: ctx 已规范化Context, w 可定位输出, source 完整头部, options 输出限制, working 不含设置和输出缓冲的活动容量
// 返回: *jxrPacketStream 增量包提交状态, func() 退出清理函数, error 预检、预算、取消或I/O错误
func beginJXRPacketStream(ctx context.Context, w io.WriteSeeker, source *jxrStream, options jxrStreamOutput, working uint64) (*jxrPacketStream, func(), error) {
	if source == nil || w == nil || options.budget != nil {
		return nil, nil, fmt.Errorf("%w: incremental stream parameters", ErrInvalid)
	}
	s := *source
	if !options.validated {
		if err := checkJXRStreamOutput(&s, options.maxPixels); err != nil {
			return nil, nil, err
		}
	}
	tiles := (len(s.header.columns) - 1) * (len(s.header.rows) - 1)
	bands := 1
	if s.header.frequency {
		bands = 4 - int(s.primary.bands)
	}
	count := tiles * bands
	if options.forward != nil && len(options.forward.packets) != count {
		return nil, nil, fmt.Errorf("%w: sequential planned packet count", ErrInvalid)
	}
	if len(options.tiles) != 1 && len(options.tiles) != tiles {
		return nil, nil, fmt.Errorf("%w: incremental tile settings", ErrInvalid)
	}
	settingsWork, err := options.tileWork(ctx, options.maxMemory)
	if err != nil {
		return nil, nil, err
	}
	if settingsWork > ^uint64(0)-uint64(count)*8 {
		return nil, nil, fmt.Errorf("%w: incremental settings capacity", ErrLimit)
	}
	extra := settingsWork + uint64(count)*8
	if working >= options.maxMemory || extra >= options.maxMemory-working {
		return nil, nil, fmt.Errorf("%w: incremental output working memory", ErrLimit)
	}
	if !options.validated {
		for _, setting := range options.tiles {
			if setting.trim > 15 || !s.header.trim && setting.trim != 0 {
				return nil, nil, fmt.Errorf("%w: incremental trim", ErrInvalid)
			}
			for index, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
				if plane != nil {
					if err := checkJXRTileQuant(plane, &setting.quant[index]); err != nil {
						return nil, nil, err
					}
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, nil, err
	}
	if position < 0 {
		return nil, nil, fmt.Errorf("%w: incremental output position", ErrInvalid)
	}
	maximum := min(options.maxBytes, uint64(math.MaxInt64-position))
	if maximum == 0 {
		return nil, nil, fmt.Errorf("%w: incremental output address capacity", ErrLimit)
	}
	ps := &jxrPacketStream{
		stream:   s,
		options:  options,
		position: position,
		maximum:  maximum,
		bands:    bands,
		tiles:    tiles,
		budget:   jxrWriteBudget{limit: options.maxMemory - working - extra},
	}
	cleanup := func() {}
	if mem, ok := w.(*jxrMemWriteSeeker); ok {
		prevBudget, prevMax := mem.budget, mem.maxMemory
		memCap := uint64(cap(mem.data))
		if memCap >= ps.budget.limit {
			return nil, nil, &LimitError{Resource: "max memory bytes", Limit: ps.budget.limit, Required: jxrSaturatedAdd(memCap, 1), Detail: "incremental output stage memory"}
		}
		ps.budget.used = memCap
		mem.budget = &ps.budget
		mem.maxMemory = min(prevMax, ps.budget.limit)
		cleanup = func() {
			mem.budget, mem.maxMemory = prevBudget, prevMax
		}
	}
	ps.stream.packets = make([]uint64, count)
	header := newJXRBitWriter(ctx, maximum, ps.budget.limit)
	header.budget = &ps.budget
	indexAt := writeJXRStreamHeaderChecked(&header, &ps.stream, true)
	if header.err != nil {
		cleanup()
		return nil, nil, header.err
	}
	if plan := options.forward; plan != nil {
		identity := sha256.Sum256(header.data)
		if plan.ready {
			if plan.prefix != identity || plan.prefixBytes != uint64(len(header.data)) {
				cleanup()
				return nil, nil, fmt.Errorf("%w: sequential header changed", ErrInvalid)
			}
			if ps.stream.header.index {
				if indexAt > uint64(len(header.data)) || uint64(count)*9 > uint64(len(header.data))-indexAt {
					cleanup()
					return nil, nil, fmt.Errorf("%w: sequential index header capacity", ErrInvalid)
				}
				for index, packet := range plan.packets {
					at := indexAt + uint64(index)*9
					header.data[at] = 0xfc
					binary.BigEndian.PutUint64(header.data[at+1:], packet.offset)
				}
			}
		} else {
			plan.prefix, plan.prefixBytes = identity, uint64(len(header.data))
		}
	}
	ps.indexAt = indexAt
	ps.total, ps.prefix = uint64(len(header.data)), uint64(len(header.data))
	if err := writeEncoderOutput(ctx, w, header.data); err != nil {
		cleanup()
		return nil, nil, err
	}
	if ps.budget.used >= uint64(cap(header.data)) {
		ps.budget.used -= uint64(cap(header.data))
	}
	header.data = nil
	return ps, cleanup, nil
}

// commitTile 校验联合包预算、记录偏移及可选前向摘要并按频带顺序写出单块包字节
// 入参: ctx 已规范化Context, w 可定位输出, tile 编码分块序号, packets 单块已编码频带包
// 返回: error 容量、一致性、取消或I/O错误
func (ps *jxrPacketStream) commitTile(ctx context.Context, w io.WriteSeeker, tile int, packets *[4]jxrBitWriter) error {
	encoded := uint64(0)
	for band := range ps.bands {
		length := uint64(len(packets[band].data))
		if length > ps.maximum-ps.total-encoded {
			return fmt.Errorf("%w: incremental combined packet byte budget", ErrLimit)
		}
		encoded += length
	}
	for band := range ps.bands {
		ps.stream.packets[tile*ps.bands+band] = ps.total - ps.prefix
		if plan := ps.options.forward; plan != nil {
			current := jxrForwardPacket{offset: ps.total - ps.prefix, length: uint64(len(packets[band].data)), digest: sha256.Sum256(packets[band].data)}
			if plan.ready {
				if plan.packets[tile*ps.bands+band] != current {
					return fmt.Errorf("%w: sequential encoded packet changed", ErrInvalid)
				}
			} else {
				plan.packets[tile*ps.bands+band] = current
			}
		}
		if err := writeEncoderOutput(ctx, w, packets[band].data); err != nil {
			return err
		}
		ps.total += uint64(len(packets[band].data))
	}
	return nil
}

// finish 校验前向计划总长度并在需要时回填码流包偏移索引表
// 入参: ctx 已规范化Context, w 可定位输出
// 返回: error 一致性、定位、写入或取消错误
func (ps *jxrPacketStream) finish(ctx context.Context, w io.WriteSeeker) error {
	if plan := ps.options.forward; plan != nil {
		if plan.ready && plan.total != ps.total {
			return fmt.Errorf("%w: sequential encoded size changed", ErrInvalid)
		}
		if !plan.ready {
			plan.total = ps.total
		}
	}
	if ps.stream.header.index && (ps.options.forward == nil || !ps.options.forward.ready) {
		if err := patchJXRStreamIndexTable(ctx, w, ps.position, ps.indexAt, ps.total, ps.stream.packets); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// patchJXRStreamIndexTable 在可定位输出中回填固定九字节宽度的码流包偏移索引表并恢复至码流末尾
// 入参: ctx 已规范化Context, w 可定位输出, streamPos 码流起始位置, indexAt 索引表相对偏移, total 码流总字节数, packets 各包相对偏移切片
// 返回: error 定位、写入或取消错误
func patchJXRStreamIndexTable(ctx context.Context, w io.WriteSeeker, streamPos int64, indexAt, total uint64, packets []uint64) error {
	if err := seekJXRImageOutput(ctx, w, uint64(streamPos)+indexAt); err != nil {
		return err
	}
	var patch [9 * 256]byte
	for at := 0; at < len(packets); {
		count := min(len(packets)-at, 256)
		for index := range count {
			patch[index*9] = 0xfc
			binary.BigEndian.PutUint64(patch[index*9+1:], packets[at+index])
		}
		if err := writeEncoderOutput(ctx, w, patch[:count*9]); err != nil {
			return err
		}
		at += count
	}
	return seekJXRImageOutput(ctx, w, uint64(streamPos)+total)
}

// encodeJXRFileTilesSequential 从可重放有界块源顺序写出单图JXR容器文件到普通Writer，不要求Seek或完整原图及压缩结果
// 入参: ctx 已规范化Context, w 顺序输出, info 原生描述, read 可重放提供器, opts 文件分块参数
// 返回: error 元数据、能力、资源、精度、源、取消或写入错误
func encodeJXRFileTilesSequential(ctx context.Context, w io.Writer, info Info, read RasterTileReader, opts *EncodeOptions) error {
	var settings EncodeOptions
	if opts != nil {
		settings = *opts
	}
	if err := checkJXREncodeOptions(settings); err != nil {
		return err
	}
	if settings.Format != FormatUnknown && settings.Format != FormatJXR || settings.TileSize.X == 0 || settings.TileSize.Y == 0 {
		return fmt.Errorf("%w: sequential file requires explicit Annex A tiles", ErrUnsupported)
	}
	if info.ImageType&ImageTypePreview != 0 {
		return fmt.Errorf("%w: JPEG XR preview requires corresponding image", ErrInvalid)
	}
	settings.Format = FormatJXR
	info, retained, err := prepareJXRTileInfo(ctx, info, settings)
	if err != nil {
		return err
	}
	var input jxrSeparateTileInput
	if settings.SeparateAlpha {
		input, err = makeJXRSeparateTileInput(ctx, info, settings, retained)
		if err != nil {
			return err
		}
	} else {
		metadata := Raster{info: info}
		source, output, err := makeJXREncodingPlanRetained(ctx, &metadata, &settings, retained)
		if err != nil {
			return err
		}
		settingsWork, err := output.stream.tileWork(ctx, output.stream.maxMemory)
		if err != nil {
			return err
		}
		working := source.headerWork() + settingsWork
		if retained >= output.stream.maxMemory || working >= output.stream.maxMemory-retained {
			return fmt.Errorf("%w: sequential file retained metadata", ErrLimit)
		}
		input.info[0], input.file.info, input.file.layout[0] = info, info, source
		input.file.maximum, input.file.maxPixels = output.stream.maxBytes, output.stream.maxPixels
		input.file.prefixMemory = output.stream.maxMemory - retained - working
		output.stream.maxMemory -= retained
		input.output[0] = output.stream
	}
	return writeJXRForwardTileFile(ctx, w, &input, read, settings.SeparateAlpha)
}

// writeJXRForwardTileFile 共用交织与双流的先规划后顺序提交，目录和固定宽度索引均在首次外部写入前确定
// 入参: ctx Context, w 顺序输出, input 已校验的常驻布局和描述, read 稳定窗口源, separate 是否独立alpha
// 返回: error 目录、容量、源、精度、取消或I/O错误
func writeJXRForwardTileFile(ctx context.Context, w io.Writer, input *jxrSeparateTileInput, read RasterTileReader, separate bool) error {
	count := 1
	var alpha *jxrStream
	if separate {
		count, alpha = 2, &input.file.layout[1]
	}
	maximum := min(input.file.maximum, uint64(math.MaxUint32))
	header, offsets, err := makeJXRFilePrefixParts(ctx, input.file.info, &input.file.layout[0], alpha, input.file.maxPixels, maximum, input.file.prefixMemory)
	if err != nil {
		return err
	}
	var plans [2]*jxrForwardPlan
	var options [2]jxrStreamOutput
	var planWork [2]uint64
	for index := range count {
		source := &input.file.layout[index]
		packets := uint64(len(source.header.columns)-1) * uint64(len(source.header.rows)-1)
		if source.header.frequency {
			packets *= uint64(4 - source.primary.bands)
		}
		if packets == 0 || packets > uint64(^uint(0)>>1)/48 {
			return fmt.Errorf("%w: sequential file packet directory", ErrLimit)
		}
		planWork[index] = 512 + packets*48
	}
	for index := range count {
		selected := input.output[index]
		held := uint64(cap(header)) + planWork[1-index]
		if held >= selected.maxMemory {
			return fmt.Errorf("%w: sequential file prefix and other plan memory", ErrLimit)
		}
		selected.maxMemory -= held
		plans[index], options[index], err = newJXRForwardPlan(ctx, &input.file.layout[index], selected, 0)
		if err != nil {
			return err
		}
	}
	encode := func(index int, destination io.WriteSeeker) error {
		if separate {
			return encodeJXRProjectedTilesTo(ctx, destination, input.file.info, input.info[index], &input.file.layout[index], options[index], read, index == 1, input.mirror)
		}
		return encodeJXRProvidedTilesTo(ctx, destination, input.info[0], &input.file.layout[0], options[0], read)
	}
	total := uint64(len(header))
	var starts [2]uint64
	for index := range count {
		if total >= maximum {
			return fmt.Errorf("%w: sequential file stream byte capacity", ErrLimit)
		}
		starts[index], options[index].maxBytes = total, maximum-total
		counter := jxrCountingSeeker{}
		if err := encode(index, &counter); err != nil {
			return err
		}
		plan := plans[index]
		if plan.total > maximum-total || counter.position != int64(plan.total) || counter.length != int64(plan.total) {
			return fmt.Errorf("%w: sequential file stream extent", ErrInvalid)
		}
		total += plan.total
		if separate && index == 0 && total&1 != 0 {
			if total == maximum {
				return fmt.Errorf("%w: sequential file alpha alignment capacity", ErrLimit)
			}
			total++
		}
		plan.ready = true
	}
	binary.LittleEndian.PutUint32(header[offsets.imageSize:], uint32(plans[0].total))
	if separate {
		binary.LittleEndian.PutUint32(header[offsets.alphaOffset:], uint32(starts[1]))
		binary.LittleEndian.PutUint32(header[offsets.alphaSize:], uint32(plans[1].total))
	}
	if err := writeEncoderOutput(ctx, w, header); err != nil {
		return err
	}
	forward := jxrForwardWriter{writer: w, position: int64(len(header))}
	header = nil
	for index := range count {
		if err := encode(index, &forward); err != nil {
			return err
		}
		if separate && index == 0 && forward.position&1 != 0 {
			if err := writeEncoderOutput(ctx, &forward, []byte{0}); err != nil {
				return err
			}
		}
	}
	if forward.position != int64(total) {
		return fmt.Errorf("%w: sequential file final extent", ErrInvalid)
	}
	return ctx.Err()
}

// writeJXRFileTo 共用已知布局的单图文件前缀和长度回填，具体输入方式由独立编码器处理
// 入参: ctx Context, w 起点为零的可定位输出, info 原生元数据, source 完整编码布局, options 含输入的额度, retained 输入常驻额度, encode 主图码流编码器
// 返回: error 元数据、资源、取消、定位或写入错误
func writeJXRFileTo(ctx context.Context, w io.WriteSeeker, info Info, source *jxrStream, options jxrStreamOutput, retained uint64, encode func(jxrStreamOutput) error) error {
	_, _, err := writeJXRFileAt(ctx, w, info, source, options, retained, 8, encode)
	return err
}

// writeJXRFileAt 在指定目录位置逐包写当前图像，非首图不重复文件头并将目录地址转换为全文件坐标
// 入参: ctx Context, w 当前输出, info 原生信息, source 编码布局, options 含输入的额度, retained 常驻输入额度, start 当前目录起点，首图为八, encode 码流编码器
// 返回: uint64 当前图像结束位置, uint64 当前目录链接位置, error 元数据、资源、取消、定位或写入错误
func writeJXRFileAt(ctx context.Context, w io.WriteSeeker, info Info, source *jxrStream, options jxrStreamOutput, retained, start uint64, encode func(jxrStreamOutput) error) (uint64, uint64, error) {
	if start < 8 || start&1 != 0 || start > math.MaxUint32 {
		return 0, 0, fmt.Errorf("%w: incremental file directory position", ErrInvalid)
	}
	settingsWork, err := options.tileWork(ctx, options.maxMemory)
	if err != nil {
		return 0, 0, err
	}
	working := source.headerWork() + settingsWork
	if retained >= options.maxMemory || working >= options.maxMemory-retained {
		return 0, 0, fmt.Errorf("%w: incremental file metadata memory", ErrLimit)
	}
	delta := start - 8
	options.maxBytes = min(options.maxBytes, uint64(math.MaxUint32)-delta)
	header, sizeAt, err := makeJXRFilePrefix(ctx, info, source, options.maxPixels, options.maxBytes, options.maxMemory-retained-working)
	if err != nil {
		return 0, 0, err
	}
	prefix := uint64(len(header))
	if prefix >= options.maxBytes {
		return 0, 0, fmt.Errorf("%w: incremental file codestream capacity", ErrLimit)
	}
	link, err := relocateJXRImageDirectory(header, start)
	if err != nil {
		return 0, 0, err
	}
	position, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, 0, err
	}
	wanted := int64(start)
	if start == 8 {
		wanted = 0
	} else {
		header = header[8:]
	}
	if position != wanted {
		return 0, 0, fmt.Errorf("%w: incremental file output position", ErrInvalid)
	}
	if err := writeEncoderOutput(ctx, w, header); err != nil {
		return 0, 0, err
	}
	header = nil
	options.maxBytes -= prefix
	if err := encode(options); err != nil {
		return 0, 0, err
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	end, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, 0, err
	}
	streamStart := delta + prefix
	if end < 0 || uint64(end) < streamStart || uint64(end)-streamStart > options.maxBytes {
		return 0, 0, fmt.Errorf("%w: incremental file output extent", ErrInvalid)
	}
	if err := patchJXRFileSize(ctx, w, delta+sizeAt, uint64(end)-streamStart, uint64(end)); err != nil {
		return 0, 0, err
	}
	return uint64(end), link, nil
}

// patchJXRFileSize 回填容器目录中的主图码流字节长度字段并恢复输出位置至码流末尾
// 入参: ctx 已规范化Context, w 可定位输出, sizeAt 长度字段绝对偏移, streamSize 主码流字节数, endPos 回填后恢复的流末尾
// 返回: error 定位、写入或取消错误
func patchJXRFileSize(ctx context.Context, w io.WriteSeeker, sizeAt, streamSize, endPos uint64) error {
	if err := seekJXRImageOutput(ctx, w, sizeAt); err != nil {
		return err
	}
	var count [4]byte
	binary.LittleEndian.PutUint32(count[:], uint32(streamSize))
	if err := writeEncoderOutput(ctx, w, count[:]); err != nil {
		return err
	}
	if err := seekJXRImageOutput(ctx, w, endPos); err != nil {
		return err
	}
	return ctx.Err()
}

// makeJXRFilePrefix 从同一已校验编码布局产生目录及元数据，不构造占位码流或假样本
// 入参: ctx Context, info 原生信息, source 编码布局, maxPixels 像素上限, maxBytes 文件上限, maxMemory 前缀容量
// 返回: []byte 独占文件前缀, uint64 主图长度字段位置, error 元数据、容量或取消错误
func makeJXRFilePrefix(ctx context.Context, info Info, source *jxrStream, maxPixels, maxBytes, maxMemory uint64) ([]byte, uint64, error) {
	data, offsets, err := makeJXRFilePrefixParts(ctx, info, source, nil, maxPixels, maxBytes, maxMemory)
	return data, offsets.imageSize, err
}

// makeJXRFilePrefixParts 从明确的主流和可选独立透明度头部生成目录，不需要完整压缩结果
// 入参: ctx Context, info 原生元数据, source 主图布局, alpha 可选独立透明度布局, maxPixels 像素上限, maxBytes 文件上限, maxMemory 前缀容量
// 返回: []byte 文件前缀, jxrContainerOffsets 全部待回填字段位置, error 元数据、布局、容量或取消错误
func makeJXRFilePrefixParts(ctx context.Context, info Info, source, alpha *jxrStream, maxPixels, maxBytes, maxMemory uint64) ([]byte, jxrContainerOffsets, error) {
	identifier, format, err := jxrRasterIdentifier(info, [16]byte{})
	if err != nil {
		return nil, jxrContainerOffsets{}, err
	}
	c := jxrContainer{width: source.header.width, height: source.header.height, transform: uint32(source.header.transform), format: format, pixelFormat: identifier, imageBands: uint32(source.primary.bands), profiles: source.profiles}
	c.setInfoMetadata(info)
	if alpha != nil {
		if !info.Alpha || source.header.alpha || !alpha.header.alpha || alpha.alpha == nil || source.header.width != alpha.header.width || source.header.height != alpha.header.height || source.header.depth != alpha.header.depth || alpha.primary.channels != 1 {
			return nil, jxrContainerOffsets{}, fmt.Errorf("%w: incremental separate alpha stream layout", ErrInvalid)
		}
		c.alphaBands = uint32(alpha.alpha.bands)
	}
	exifNode, gpsNode, err := checkJXRContainerMetadataLayout(ctx, &c, maxPixels, maxMemory, alpha != nil)
	if err != nil {
		return nil, jxrContainerOffsets{}, err
	}
	channels := format.channels
	if format.alpha {
		channels++
	}
	if err := source.checkProfileList(source.profiles, format.alpha, channels); err != nil {
		return nil, jxrContainerOffsets{}, err
	}
	b := newJXRBitWriter(ctx, maxBytes, maxMemory)
	if b.err != nil {
		return nil, jxrContainerOffsets{}, b.err
	}
	var offsets jxrContainerOffsets
	writeJXRContainerBytes(&b, &c, true, alpha != nil, &offsets, exifNode, gpsNode)
	if b.err != nil {
		return nil, jxrContainerOffsets{}, b.err
	}
	return b.data, offsets, nil
}
