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
)

// RasterStripWriter 同步消费按编码宏块行产生的原生行带，region为完整显示位置，source为零起点只读临时图像
// 方向变换后行带可能为竖带或逆序，仅回调期间可读取source，不得修改、保留引用或重入当前Decoder（如需保留请显式Clone）
// 接收Context、显示位置及临时行带并返回消费者错误，返回非nil错误时立即中止且不回滚已消费行带
type RasterStripWriter func(ctx context.Context, region image.Rectangle, source *Raster) error

// jxrStripRowOutput 复用一条显示行带及其独立描述，将最终样本窗口投影给同步消费者
type jxrStripRowOutput struct {
	output     jxrRegionRowOutput
	write      RasterStripWriter
	cache      *Raster
	view       Raster
	components []ComponentInfo
	planes     []imageComponents
	alpha      *jxrStripAlphaRows
}

// jxrStripAlphaRows 保留独立透明度的最终行游标，按主图可见坐标同步配对，不要求两流分块或裁剪边缘相同
type jxrStripAlphaRows struct {
	source *jxrStream
	rows   *jxrReconstructedRows
	state  *jxrCoefficientImage
	row    int
	number uint8
}

// jxrStripPlan 保存已完成网格和总容量预检的行带输出布局，不读取熵包
type jxrStripPlan struct {
	info      Info
	cacheWork uint64
	alphaWork uint64
	remaining uint64
	total     uint64
}

// jxrSpatialRows 持有单个空间包的连续熵状态和预测行，系数目标可在每行后复用
type jxrSpatialRows struct {
	bits     jxrBits
	quant    [2]jxrTileQuant
	rows     [2]*jxrTileRow
	scratch  [2]*jxrDecodedMacro
	dc       [2]jxrDC
	lp       [2]jxrLP
	hp       [2]jxrHP
	channels [2]int
	width    int
	height   int
	next     int
	trim     uint8
}

// jxrFrequencyRows 持有各频带连续位位置、独立预测行及共享的当前行系数工作，不保留整块系数
type jxrFrequencyRows struct {
	bits      [4]jxrBits
	quant     [2]jxrTileQuant
	rows      [3][2]*jxrTileRow
	scratch   [2]*jxrDecodedMacro
	dc        [2]jxrDC
	lp        [2]jxrLP
	hp        [2]jxrHP
	channels  [2]int
	bands     [2]uint8
	width     int
	height    int
	next      int
	count     int
	trim      uint8
	flexValid bool
}

// jxrReconstructedRow 定义只读消费已完成两级逆变换及重叠滤波宏块行的回调函数类型
type jxrReconstructedRow func(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error

// jxrReconstructedRows 以固定邻行状态按需交出最终样本行，允许不同裁剪及分块的主流与透明度流同步推进
type jxrReconstructedRows struct {
	reader   jxrCoefficientRows
	raw      jxrCoefficientImage
	pipeline *jxrInverseRows
	height   int
	read     int
	nextRow  int
	flushed  bool
}

// jxrInverseRows 保留首级系数、二级样本及独立工作窗口各三行，避免后滤波覆盖下一行所需的原始邻域
type jxrInverseRows struct {
	header  jxrHeader
	raw     jxrCoefficientImage
	samples jxrCoefficientImage
	work    jxrCoefficientImage
	width   int
	height  int
	rowEnds [2]uint32
}

// jxrTileRows 按原分块行组合各列独立熵状态，少列复用固定描述容量，完整网格仅按当前列数分配
type jxrTileRows struct {
	source    jxrStream
	data      []byte
	readers   []jxrCoefficientRows
	storage   [5]jxrCoefficientRows
	maximum   uint64
	width     int
	height    int
	next      int
	tileRow   int
	activeRow int
}

// jxrUpsampleRows 保存最多三行已完成后滤波的原生样本，垂直插值不会读取仍待后滤波的邻行
type jxrUpsampleRows struct {
	image  jxrCoefficientImage
	rows   [2]uint32
	width  int
	height int
	count  int
}

// jxrSampleRowOutput 以具体目标描述分派最终样本，避免Interface调用使每行临时分量视图逃逸到堆
type jxrSampleRowOutput struct {
	region *jxrRegionRowOutput
	strip  *jxrStripRowOutput
}

// planDecodeRows 只在行工作量可能更小时预检完整输出并计入调用方额外输出预留容量，不修改解码器上限
// 入参: ctx Context, full 原整幅路径的总工作容量, extra 额外预留字节数
// 返回: jxrStripPlan 可采用的行布局, uint64 完整输出容量，零表示不采用, error 取消错误
func (d *Decoder) planDecodeRows(ctx context.Context, full, extra uint64) (jxrStripPlan, uint64, error) {
	var empty jxrStripPlan
	rows, coefficients := uint64(0), uint64(0)
	for _, stream := range [2]*jxrStream{&d.source.primary, d.source.alpha} {
		if stream == nil {
			continue
		}
		var ok bool
		rows, ok = jxrWorkTotal(^uint64(0), rows, jxrStreamingRowWork(stream))
		if !ok {
			return empty, 0, nil
		}
		coefficients, ok = jxrWorkTotal(^uint64(0), coefficients, stream.coefficientWork())
		if !ok {
			coefficients = ^uint64(0)
		}
	}
	if rows >= coefficients {
		return empty, 0, nil
	}
	_, output, err := jxrRegionInfo(d.info, d.info.Bounds)
	if err != nil {
		return empty, 0, err
	}
	if extra > ^uint64(0)-output {
		return empty, 0, nil
	}
	plan, err := d.planStrips(ctx, output+extra)
	if err != nil {
		if canceled := ctx.Err(); canceled != nil {
			return empty, 0, canceled
		}
		return empty, 0, nil
	}
	if plan.total-extra >= full {
		return empty, 0, nil
	}
	return plan, output, nil
}

// decodeRows 用已预算的行布局填充完整原生图像，全部包成功后才返回独立结果
// 入参: ctx Context, plan 包含完整目标容量的有效布局, work 完整图像存储容量
// 返回: *Raster 完整原生图像, error 存储、数据、读取或取消错误
func (d *Decoder) decodeRows(ctx context.Context, plan jxrStripPlan, work uint64) (*Raster, error) {
	target, err := newRasterContext(ctx, d.info, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: work})
	if err != nil {
		return nil, err
	}
	if done, err := d.decodeRowsHardTiles(ctx, target, plan); err != nil {
		return nil, err
	} else if done {
		return target, ctx.Err()
	}
	if done, err := d.decodeRowsFullParallel(ctx, target, plan); err != nil {
		return nil, err
	} else if done {
		return target, ctx.Err()
	}
	if err := d.walkStrips(ctx, func(ctx context.Context, region image.Rectangle, strip *Raster) error {
		return copyJXRDecodedStrip(ctx, target, region, strip)
	}, plan); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return target, nil
}

// decodeRowsHardTiles 在全内存多独立分块且预算充足时按分块并行重建并写入互不相交的目标区域
// 入参: ctx Context, target 独占完整原生输出图像, plan 已预检的行带预算布局
// 返回: bool 是否已由多核分块路径完成, error 解码、数值或取消错误
func (d *Decoder) decodeRowsHardTiles(ctx context.Context, target *Raster, plan jxrStripPlan) (bool, error) {
	if d.workers <= 1 || d.source.primary.input != nil || d.source.alpha != nil && d.source.alpha.input != nil {
		return false, nil
	}
	sources := [2]jxrRasterSource{d.source}
	sources[0].alpha = nil
	sources[0].container.alpha = nil
	sources[0].container.alphaInput = nil
	sources[0].container.format.alpha = d.source.primary.alpha != nil
	if !sources[0].container.format.alpha {
		sources[0].container.format.premultiply = false
	}
	passes := 1
	if d.source.alpha != nil {
		passes++
		sources[1] = jxrCodedSource(*d.source.alpha, d.source.container.alpha)
		sources[1].container.transform = 0
	}
	available := plan.remaining + plan.cacheWork + plan.alphaWork
	var tileCounts, passWorkers [2]int
	var maxWorks [2]uint64
	for pass := range passes {
		source := &sources[pass]
		s := &source.primary
		h := &s.header
		tiles := (len(h.columns) - 1) * (len(h.rows) - 1)
		if tiles < 1 || pass == 0 && tiles <= 1 || tiles > 1 && (!h.index || !h.hardTiles && h.overlap != 0) {
			return false, nil
		}
		if (s.primary.color == 1 || s.primary.color == 2) && (h.color == 3 || h.color == 7) {
			return false, nil
		}
		maxWork := uint64(0)
		for tile := range tiles {
			if err := ctx.Err(); err != nil {
				return true, err
			}
			selected, coded, err := source.regionTile(tile)
			if err != nil || coded.Empty() {
				return false, nil
			}
			if color := source.container.format.color; color == 1 || color == 2 {
				if coded.Min.X%2 != 0 || coded.Max.X%2 != 0 || color == 1 && (coded.Min.Y%2 != 0 || coded.Max.Y%2 != 0) {
					return false, nil
				}
			}
			tileWork := selected.regionDecodeWork()
			if tileWork == ^uint64(0) {
				return false, nil
			}
			if tileWork > maxWork {
				maxWork = tileWork
			}
		}
		if maxWork > ^uint64(0)-4096 {
			return false, nil
		}
		perWorker := maxWork + 4096
		if perWorker > available {
			return false, nil
		}
		workers := budgetJXRWorkers(resolveJXRWorkers(d.workers, tiles), available, perWorker)
		if pass == 0 && workers <= 1 {
			return false, nil
		}
		tileCounts[pass], passWorkers[pass], maxWorks[pass] = tiles, workers, maxWork
	}
	transform := uint8(d.source.container.transform)
	bounds := target.Bounds()
	for pass := range passes {
		source := &sources[pass]
		h := &source.primary.header
		width, height := int(h.width), int(h.height)
		alphaOnly := pass != 0
		tileLimit := maxWorks[pass]
		if err := runJXRWorkers(ctx, passWorkers[pass], tileCounts[pass], func(runCtx context.Context, tile int) error {
			selected, coded, err := source.regionTile(tile)
			if err != nil {
				return err
			}
			selected.setWorkers(1)
			return selected.copyRegionTile(runCtx, target, coded.Min, bounds, width, height, transform, alphaOnly, tileLimit)
		}); err != nil {
			return true, err
		}
	}
	return true, nil
}

// decodeRowsFullParallel 在预算可容纳完整系数时执行并行熵解码、两级重建与行分片像素写入
// 入参: ctx Context, target 独占完整原生输出图像, plan 已预检的行带预算布局
// 返回: bool 是否已由多核整图系数路径完成, error 解码、数值或取消错误
func (d *Decoder) decodeRowsFullParallel(ctx context.Context, target *Raster, plan jxrStripPlan) (bool, error) {
	if d.workers <= 1 || d.source.primary.input != nil || d.source.alpha != nil && d.source.alpha.input != nil {
		return false, nil
	}
	c := &d.source.container
	if c.format.color == 1 || c.format.color == 2 {
		return false, nil
	}
	h := &d.source.primary.header
	if len(h.columns) < 2 || len(h.rows) < 2 {
		return false, nil
	}
	mbCount := uint64(h.columns[len(h.columns)-1]) * uint64(h.rows[len(h.rows)-1])
	if mbCount < 256 {
		return false, nil
	}
	available := plan.remaining + plan.cacheWork + plan.alphaWork
	coeffWork := d.source.primary.coefficientWork()
	if coeffWork == ^uint64(0) || coeffWork > available {
		return false, nil
	}
	if d.source.alpha != nil {
		alphaWork := d.source.alpha.coefficientWork()
		if alphaWork == ^uint64(0) || alphaWork > available-coeffWork {
			return false, nil
		}
		coeffWork += alphaWork
	}
	slack := available - coeffWork
	state, err := d.source.primary.coefficientsContext(ctx, c.image, jxrWorkerBudgetAllowance(slack, d.source.primary.coefficientWork()))
	if err != nil {
		return true, err
	}
	views, alphaView, err := state.reconstructContext(ctx)
	if err != nil {
		return true, err
	}
	alphaHeader, alphaPlane := &state.header, state.alpha
	if d.source.alpha != nil {
		alphaState, err := d.source.alpha.coefficientsContext(ctx, c.alpha, jxrWorkerBudgetAllowance(slack, d.source.alpha.coefficientWork()))
		if err != nil {
			return true, err
		}
		_, interleaved, err := alphaState.reconstructContext(ctx)
		if err != nil {
			return true, err
		}
		alphaHeader, alphaPlane, alphaView = &alphaState.header, alphaState.alpha, interleaved
	}
	p := &state.primary.plane
	if err := checkJXRPlaneComponents(p); err != nil {
		return true, err
	}
	if len(views) != p.channels {
		return true, fmt.Errorf("%w: JPEG XR sample component storage", ErrInvalid)
	}
	channels := max(3, p.channels, c.format.channels)
	upsample := (p.color == 1 || p.color == 2) && (state.header.color == 3 || state.header.color == 7)
	left, top := int(state.header.left), int(state.header.top)
	srcWidth, srcHeight := int(c.width), int(c.height)
	dstWidth := target.Bounds().Dx()
	transform := uint8(c.transform)
	opacityNumber := c.format.number
	outChannels := c.format.channels
	hasAlpha := c.format.alpha
	workers, chunkRows, tasks := jxrMacroChunkPlan(srcHeight, 16, d.workers)
	err = runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
		startY := task * chunkRows
		endY := min(startY+chunkRows, srcHeight)
		var valStorage [8]int64
		var codeStorage [8]uint32
		values, components := valStorage[:], codeStorage[:]
		if channels > len(values) {
			values, components = make([]int64, channels), make([]uint32, channels)
		} else {
			values, components = values[:channels], components[:channels]
		}
		for y := startY; y < endY; y++ {
			if err := runCtx.Err(); err != nil {
				return err
			}
			sy := y + top
			for x := range srcWidth {
				if x&1023 == 0 {
					if err := runCtx.Err(); err != nil {
						return err
					}
				}
				sx := x + left
				if !upsample {
					for i := range views {
						values[i] = views[i].at(sx, sy)
					}
				} else {
					values[0] = views[0].at(sx, sy)
					for i := 1; i < len(views); i++ {
						values[i] = sampleJXRChroma(&views[i], p, i, sx, sy)
					}
				}
				n, err := jxrRasterBitsComponents(&state.header, p, c.format.number, values, false, components)
				if err != nil {
					return err
				}
				if n != outChannels {
					return fmt.Errorf("%w: JPEG XR output component count", ErrInvalid)
				}
				alpha := uint32(0)
				if alphaView != nil {
					sample := [1]int64{alphaView.at(x+int(alphaHeader.left), y+int(alphaHeader.top))}
					var codes [1]uint32
					_, err := jxrRasterBitsComponents(alphaHeader, &alphaPlane.plane, opacityNumber, sample[:], true, codes[:])
					if err != nil {
						return err
					}
					alpha = codes[0]
				}
				px, py := jxrRasterPosition(x, y, srcWidth, srcHeight, transform)
				at := py*dstWidth + px
				for i := range n {
					target.planes[i].storeBits(at, components[i])
				}
				if hasAlpha {
					target.planes[n].storeBits(at, alpha)
				}
			}
		}
		return runCtx.Err()
	})
	if err != nil {
		return true, err
	}
	return true, nil
}

// copyJXRDecodedStrip 按显示方向及各分量网格复制借用行带，不保留临时样本引用
// 入参: ctx Context, target 独占完整图像, region 行带显示区域, strip 临时平面样本
// 返回: error 取消错误
func copyJXRDecodedStrip(ctx context.Context, target *Raster, region image.Rectangle, strip *Raster) error {
	for index, info := range target.info.Components {
		source, destination := &strip.planes[index], &target.planes[index]
		x, y := region.Min.X/int(info.XStep), region.Min.Y/int(info.YStep)
		width, height := source.bounds.Dx(), source.bounds.Dy()
		stride := destination.bounds.Dx()
		for row := range height {
			for column := 0; column < width; {
				if err := ctx.Err(); err != nil {
					return err
				}
				count := min(16384, width-column)
				from, to := row*width+column, (row+y)*stride+x+column
				if source.words != nil {
					copy(destination.words[to:to+count], source.words[from:from+count])
				} else {
					copy(destination.values[to:to+count], source.values[from:from+count])
				}
				column += count
			}
		}
	}
	return ctx.Err()
}

// DecodeStrips 单遍推进首图有效包并以固定邻行与单条可复用原生行带输出，不分配整幅系数或样本图
// 成功返回前校验全部有效包且不触发ICC转换，已回调行带在后续出错或取消时不回滚，NewDecoderAt按需读取
// 入参: ctx Context，nil采用Background, write 同步行带消费者，不得重入当前Decoder
// 返回: error 数据、能力、资源、关闭、读取、取消或消费者错误
func (d *Decoder) DecodeStrips(ctx context.Context, write RasterStripWriter) error {
	if d == nil {
		return ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if write != nil {
		userWrite := write
		write = func(cbCtx context.Context, region image.Rectangle, source *Raster) error {
			if err := userWrite(cbCtx, region, source); err != nil {
				return jxrCallbackError{err: err}
			}
			return nil
		}
	}
	return wrapJXRError("Decoder.DecodeStrips", d.decodeStripsLocked(ctx, write))
}

// decodeStripsLocked 在持锁状态下预检总容量和原生网格，创建一次熵行状态并同步推进最终样本消费者
// 入参: ctx 已规范化Context, write 同步消费者
// 返回: error 数据、能力、资源、关闭、读取、取消或消费者错误
func (d *Decoder) decodeStripsLocked(ctx context.Context, write RasterStripWriter) error {
	if d.closed {
		return ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if write == nil || len(d.data) == 0 {
		return fmt.Errorf("%w: JPEG XR strip consumer", ErrInvalid)
	}
	if d.source.container.uncompressed {
		return d.decodeUncompressedStripsLocked(ctx, write)
	}
	plan, err := d.planStrips(ctx, 0)
	if err != nil {
		return err
	}
	return d.walkStrips(ctx, write, plan)
}

// planStrips 在读取熵包前校验行网格，并将调用方额外保留容量纳入同一预算
// 入参: ctx Context, retained 本次操作另行持有的存储字节数
// 返回: jxrStripPlan 行带布局及剩余额度, error 网格、资源、能力或取消错误
func (d *Decoder) planStrips(ctx context.Context, retained uint64) (jxrStripPlan, error) {
	var plan jxrStripPlan
	if err := ctx.Err(); err != nil {
		return plan, err
	}
	source := &d.source
	s := &source.primary
	h := &s.header
	for _, stream := range [2]*jxrStream{s, source.alpha} {
		if stream != nil {
			if len(stream.header.columns) < 2 || len(stream.header.rows) < 2 {
				return plan, fmt.Errorf("%w: JPEG XR strip geometry", ErrUnsupported)
			}
			if (len(stream.header.columns) != 2 || len(stream.header.rows) != 2) && !stream.header.index {
				return plan, fmt.Errorf("%w: JPEG XR strip packet index", ErrUnsupported)
			}
			for _, plane := range [2]*jxrPlane{&stream.primary, stream.alpha} {
				if plane != nil {
					if err := checkJXRModelComponents(plane); err != nil {
						return plan, err
					}
				}
			}
		}
	}
	if color := source.container.format.color; color == 1 || color == 2 {
		if s.primary.color != color || h.left%2 != 0 || color == 1 && h.top%2 != 0 {
			return plan, fmt.Errorf("%w: JPEG XR native strip sampling alignment", ErrUnsupported)
		}
	}
	for _, component := range d.info.Components {
		if component.XStep == 0 || component.YStep == 0 || d.info.Bounds.Dx()%int(component.XStep) != 0 || d.info.Bounds.Dy()%int(component.YStep) != 0 {
			return plan, fmt.Errorf("%w: JPEG XR strip sampling grid", ErrUnsupported)
		}
	}
	remaining := min(d.limits.MaxMemoryBytes, uint64(^uint(0)>>1))
	auxiliary := uint64(len(d.info.Components))*256 + uint64(max(3, s.primary.channels, source.container.format.channels))*12 + 4096
	if source.alpha != nil {
		auxiliary += 4096
	}
	for _, amount := range []uint64{d.inputWork(), d.retainedHeaderWork(), auxiliary, retained} {
		if amount > remaining {
			return plan, fmt.Errorf("%w: JPEG XR strip retained memory", ErrLimit)
		}
		remaining -= amount
	}
	transform := uint8(source.container.transform)
	region := jxrRegionPosition(image.Rect(0, 0, int(h.width), min(16, int(h.height))), int(h.width), int(h.height), transform)
	info, cacheWork, err := jxrRegionInfo(d.info, region)
	if err != nil {
		return plan, err
	}
	if cacheWork > remaining {
		return plan, fmt.Errorf("%w: JPEG XR strip sample memory", ErrLimit)
	}
	remaining -= cacheWork
	alphaWork := uint64(0)
	if source.alpha != nil {
		alphaWork = jxrStreamingRowWork(source.alpha)
		if alphaWork > remaining {
			return plan, fmt.Errorf("%w: JPEG XR strip independent alpha memory", ErrLimit)
		}
		remaining -= alphaWork
	}
	work := jxrStreamingRowWork(s)
	if work > remaining {
		return plan, fmt.Errorf("%w: JPEG XR strip reconstruction memory", ErrLimit)
	}
	plan.info, plan.cacheWork, plan.alphaWork, plan.remaining = info, cacheWork, alphaWork, remaining
	plan.total = min(d.limits.MaxMemoryBytes, uint64(^uint(0)>>1)) - remaining + work
	return plan, ctx.Err()
}

// walkStrips 按已预算的行布局推进所有有效包，不重复预检或在失败后切换解码策略
// 入参: ctx Context, write 同步行带消费者, plan 同一解码器的有效行布局
// 返回: error 数据、资源、读取、取消或消费者错误
func (d *Decoder) walkStrips(ctx context.Context, write RasterStripWriter, plan jxrStripPlan) error {
	source := &d.source
	s := &source.primary
	h := &s.header
	cache, err := newRasterContext(ctx, plan.info, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: plan.cacheWork})
	if err != nil {
		return err
	}
	channels := max(3, s.primary.channels, source.container.format.channels)
	output := jxrStripRowOutput{write: write, cache: cache, components: make([]ComponentInfo, len(plan.info.Components)), planes: make([]imageComponents, len(plan.info.Components)), output: jxrRegionRowOutput{source: source, width: int(h.width), height: int(h.height), transform: uint8(source.container.transform), values: make([]int64, channels), codes: make([]uint32, channels)}}
	if source.alpha != nil {
		primary := *source
		primary.alpha = nil
		primary.container.format.alpha = false
		output.output.source = &primary
	}
	d.bindInputContext(ctx)
	defer d.bindInputContext(nil)
	reader, err := newJXRStreamRows(ctx, source.container.image, s, plan.remaining)
	if err != nil {
		return err
	}
	if source.alpha != nil {
		alphaReader, err := newJXRStreamRows(ctx, source.container.alpha, source.alpha, plan.alphaWork)
		if err != nil {
			return err
		}
		rows, err := newJXRReconstructedRows(ctx, source.alpha, alphaReader, plan.alphaWork)
		if err != nil {
			return err
		}
		output.alpha = &jxrStripAlphaRows{source: source.alpha, rows: rows, row: -1, number: source.container.format.number}
	}
	emit := jxrReconstructedRow(output.copy)
	var samples *jxrUpsampleRows
	frame := jxrRowReconstructionStream(s)
	if jxrUpsampleRowWork(&frame) != 0 {
		samples, err = newJXRUpsampleRows(ctx, &frame, plan.remaining)
		if err != nil {
			return err
		}
		emit = func(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error {
			return samples.accept(ctx, jxrSampleRowOutput{strip: &output}, h, views, alpha, row)
		}
	}
	if err := walkJXRReconstructedRows(ctx, s, reader, plan.remaining, emit); err != nil {
		return err
	}
	if samples != nil {
		if err := samples.finish(ctx, jxrSampleRowOutput{strip: &output}); err != nil {
			return err
		}
	}
	if output.alpha != nil {
		for output.alpha.rows.nextRow < output.alpha.rows.height {
			if _, _, err := output.alpha.rows.next(ctx); err != nil {
				return err
			}
		}
	}
	return ctx.Err()
}

// newJXRStreamRows 根据完整流几何选择单块或跨块熵状态，单块构造会读取有效初始包头
// 入参: ctx Context, data 可选完整流切片, s 已校验码流, maximum 可用工作容量
// 返回: jxrCoefficientRows 连续状态, error 资源、结构或取消错误
func newJXRStreamRows(ctx context.Context, data []byte, s *jxrStream, maximum uint64) (jxrCoefficientRows, error) {
	if len(s.header.columns) != 2 || len(s.header.rows) != 2 {
		return newJXRTileRows(ctx, data, s, maximum)
	}
	if s.header.frequency {
		return newJXRFrequencyRows(ctx, data, s, 0, maximum)
	}
	return newJXRSpatialRows(ctx, data, s, 0, maximum)
}

// copy 复用独立行带描述及样本容量，转换完成后才调用消费者，并在返回后撤销临时Raster外壳
// 入参: ctx Context, h 最终样本窗口, views 主分量样本, alpha 透明度, row 原始宏块行号
// 返回: error 转换、网格、取消或消费者错误
func (output *jxrStripRowOutput) copy(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s := &output.output.source.primary
	first, last := max(0, row*16-int(s.header.top)), min(int(s.header.height), (row+1)*16-int(s.header.top))
	if first >= last {
		return ctx.Err()
	}
	region := jxrRegionPosition(image.Rect(0, first, int(s.header.width), last), output.output.width, output.output.height, output.output.transform)
	output.view = *output.cache
	output.view.info.Bounds = image.Rect(0, 0, region.Dx(), region.Dy())
	output.view.info.Components = output.components
	output.view.planes = output.planes
	for index, component := range output.cache.info.Components {
		component.Bounds = image.Rect(0, 0, region.Dx()/int(component.XStep), region.Dy()/int(component.YStep))
		output.components[index] = component
		plane := output.cache.planes[index]
		count := component.Bounds.Dx() * component.Bounds.Dy()
		if count > len(plane.values)+len(plane.words) {
			return fmt.Errorf("%w: JPEG XR strip view capacity", ErrLimit)
		}
		clear(plane.values)
		clear(plane.words)
		plane.bounds = component.Bounds
		if len(plane.values) != 0 {
			plane.values = plane.values[:count]
		} else {
			plane.words = plane.words[:count]
		}
		output.planes[index] = plane
	}
	output.output.target, output.output.region = &output.view, region
	if err := output.output.copy(ctx, h, views, alpha, row); err != nil {
		output.view = Raster{}
		return err
	}
	if output.alpha != nil {
		if err := output.alpha.copy(ctx, &output.output, first, last); err != nil {
			output.view = Raster{}
			return err
		}
	}
	err := output.write(ctx, region, &output.view)
	output.view = Raster{}
	if err != nil {
		return err
	}
	return ctx.Err()
}

// copy 将独立透明度按其自身裁剪、位移和量化元数据转换到主流当前行带，最多保留一条借用最终行
// 入参: ctx Context, output 已转换主流行带, first, last 主图可见纵向范围
// 返回: error 透明度数据、数值、资源或取消错误
func (a *jxrStripAlphaRows) copy(ctx context.Context, output *jxrRegionRowOutput, first, last int) error {
	var storage [1]jxrComponentView
	for y := first; y < last; y++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		wanted := (y + int(a.source.header.top)) / 16
		for a.row < wanted {
			state, row, err := a.rows.next(ctx)
			if err != nil {
				return err
			}
			a.state, a.row = state, row
		}
		if a.row != wanted || a.state == nil {
			return fmt.Errorf("%w: JPEG XR strip alpha pairing", ErrInvalid)
		}
		plane := a.state.alpha
		views := plane.componentViewsStorage(&a.state.header, true, storage[:])
		sy := (y+int(a.source.header.top))%16 + int(a.state.header.top)
		for x := range output.width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			values := [1]int64{views[0].at(x+int(a.source.header.left), sy)}
			var codes [1]uint32
			if _, err := jxrRasterBitsComponents(&a.state.header, &plane.plane, a.number, values[:], true, codes[:]); err != nil {
				return err
			}
			px, py := jxrRasterPosition(x, y, output.width, output.height, output.transform)
			at := (py-output.region.Min.Y)*output.region.Dx() + px - output.region.Min.X
			output.target.planes[len(output.target.planes)-1].storeBits(at, codes[0])
		}
	}
	return ctx.Err()
}

// jxrSpatialRowWork 计算单块一行系数、连续熵状态及独立预测行的容量，不随块高增长
// 入参: s 已校验空间流, tile 分块序号
// 返回: uint64 工作预算，无效几何返回最大值
func jxrSpatialRowWork(s *jxrStream, tile int) uint64 {
	columns, rows := len(s.header.columns)-1, len(s.header.rows)-1
	if s.header.frequency || columns < 1 || rows < 1 || tile < 0 || tile >= columns*rows {
		return ^uint64(0)
	}
	x := tile % columns
	width := s.header.columns[x+1] - s.header.columns[x]
	local := *s
	local.header.columns, local.header.rows = []uint32{0, width}, []uint32{0, 1}
	return local.coefficientWork()
}

// newJXRSpatialRows 在总额度内建立空间包的连续读取状态，不分配整块系数
// 入参: ctx Context, data 码流字节视图, s 已校验空间流, tile 分块序号, maximum 行处理预算
// 返回: *jxrSpatialRows 有状态行读取器, error 参数、数据、资源或取消错误
func newJXRSpatialRows(ctx context.Context, data []byte, s *jxrStream, tile int, maximum uint64) (*jxrSpatialRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrSpatialRowWork(s, tile) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR spatial row working budget", ErrLimit)
	}
	if tile >= len(s.packets) {
		return nil, fmt.Errorf("%w: JPEG XR spatial row packet", ErrInvalid)
	}
	if err := checkJXRModelComponents(&s.primary); err != nil {
		return nil, err
	}
	if s.alpha != nil {
		if err := checkJXRModelComponents(s.alpha); err != nil {
			return nil, err
		}
	}
	columns := len(s.header.columns) - 1
	x, y := tile%columns, tile/columns
	r := &jxrSpatialRows{bits: jxrPacketReader(data, s, tile), width: int(s.header.columns[x+1] - s.header.columns[x]), height: int(s.header.rows[y+1] - s.header.rows[y])}
	r.quant[0] = newJXRTileQuant(&s.primary)
	if s.alpha != nil {
		r.quant[1] = newJXRTileQuant(s.alpha)
	}
	packet, err := readJXRPacket(&r.bits, s, -1, &r.quant)
	if err != nil {
		return nil, err
	}
	r.trim = packet.trim
	for i, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p == nil {
			continue
		}
		r.channels[i] = p.channels
		r.rows[i], err = newJXRTileRow(ctx, r.width, p.channels, maximum)
		if err != nil {
			return nil, err
		}
		r.scratch[i], err = newJXRDecodedMacro(ctx, p.channels, maximum)
		if err != nil {
			return nil, err
		}
		r.dc[i], r.lp[i], r.hp[i] = newJXRDC(), newJXRLP(), newJXRHP()
	}
	return r, nil
}

// readRow 按原包顺序读取一宏块行，保留未反量化预测值并覆盖调用方指定的系数行
// 入参: ctx Context, target 单行或整图系数目标, start 目标首宏块下标
// 返回: error 数据、目标容量、数值或取消错误，末行同时校验包尾对齐
func (r *jxrSpatialRows) readRow(ctx context.Context, target *jxrCoefficientImage, start int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.next >= r.height || start < 0 {
		return fmt.Errorf("%w: JPEG XR spatial row sequence", ErrInvalid)
	}
	planes := [2]*jxrCoefficients{&target.primary, target.alpha}
	for i, p := range planes {
		if p == nil {
			if r.channels[i] != 0 {
				return fmt.Errorf("%w: JPEG XR spatial row missing plane", ErrInvalid)
			}
			continue
		}
		if p.plane.channels < 1 || p.plane.channels != r.channels[i] || start > len(p.info)-r.width || len(p.low)/(16*p.plane.channels) < start+r.width || len(p.high)/(256*p.plane.channels) < start+r.width {
			return fmt.Errorf("%w: JPEG XR spatial row coefficient storage", ErrInvalid)
		}
		clear(p.info[start : start+r.width])
		clear(p.low[start*16*p.plane.channels : (start+r.width)*16*p.plane.channels])
		clear(p.high[start*256*p.plane.channels : (start+r.width)*256*p.plane.channels])
		r.rows[i].startRow()
	}
	for x := range r.width {
		if err := ctx.Err(); err != nil {
			return err
		}
		for i, p := range planes {
			if p == nil {
				continue
			}
			row, q := r.rows[i], &r.quant[i]
			m := jxrMacroContext{at: start + x, firstColumn: x == 0, firstRow: r.next == 0, reset: x%16 == 0, adapt: x == r.width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
			info := &p.info[m.at]
			if p.plane.bands != 3 {
				if q.lpCount > 1 && !q.useDC {
					info.lpIndex = uint8(readJXRQPIndex(&r.bits, q.lpCount))
				}
				if p.plane.bands < 2 {
					if q.useLP {
						info.hpIndex = info.lpIndex
					} else if q.hpCount > 1 {
						info.hpIndex = uint8(readJXRQPIndex(&r.bits, q.hpCount))
					}
				}
			}
			current := &row.current
			if err := p.readDC(&r.bits, &r.dc[i], q, m, &current.low, r.scratch[i]); err != nil {
				return err
			}
			if p.plane.bands != 3 {
				if err := p.readLP(&r.bits, &r.lp[i], q, m, &current.low, r.scratch[i]); err != nil {
					return err
				}
			}
			if p.plane.bands < 2 {
				if err := p.readHP(&r.bits, &r.hp[i], m, true, r.trim, row.residual, current.pattern); err != nil {
					return err
				}
				if err := p.finishHighComponents(m.at, q, r.scratch[i].low); err != nil {
					return err
				}
			}
			if err := row.advance(x, current); err != nil {
				return err
			}
		}
	}
	r.next++
	if r.next == r.height {
		r.bits.align(true)
	}
	return r.bits.err
}

// jxrFrequencyRowWork 计算单行系数、各有效预测频带和全部独立随机读取窗的总工作，不随分块高度增长
// 入参: s 已校验频率流, tile 分块序号
// 返回: uint64 总工作字节数，无效布局或溢出返回最大值
func jxrFrequencyRowWork(s *jxrStream, tile int) uint64 {
	if !s.header.frequency || s.primary.bands > 3 {
		return ^uint64(0)
	}
	local := *s
	local.header.frequency = false
	work := jxrSpatialRowWork(&local, tile)
	if work == ^uint64(0) {
		return work
	}
	column := tile % (len(s.header.columns) - 1)
	width := uint64(s.header.columns[column+1] - s.header.columns[column])
	for _, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p == nil {
			continue
		}
		if p.bands > 3 {
			return ^uint64(0)
		}
		row := jxrPredictionRowWork(width, p.channels)
		for extra := 1; extra < min(3, 4-int(p.bands)); extra++ {
			if row > ^uint64(0)-work {
				return ^uint64(0)
			}
			work += row
		}
	}
	if s.input != nil {
		windows := uint64(4-s.primary.bands) * jxrReaderAtWork
		if windows > ^uint64(0)-work {
			return ^uint64(0)
		}
		work += windows
	}
	return work
}

// newJXRFrequencyRows 在总额度内建立独立频带状态，按DC到精细位的顺序解析量化复用关系
// 入参: ctx Context, data 码流字节视图, s 已校验频率流, tile 分块序号, maximum 总工作额度
// 返回: *jxrFrequencyRows 连续行状态, error 数据、模型、资源或取消错误
func newJXRFrequencyRows(ctx context.Context, data []byte, s *jxrStream, tile int, maximum uint64) (*jxrFrequencyRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrFrequencyRowWork(s, tile) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR frequency row working budget", ErrLimit)
	}
	count := 4 - int(s.primary.bands)
	if tile >= len(s.packets)/count {
		return nil, fmt.Errorf("%w: JPEG XR frequency row packet", ErrInvalid)
	}
	for _, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p != nil {
			if err := checkJXRModelComponents(p); err != nil {
				return nil, err
			}
		}
	}
	columns := len(s.header.columns) - 1
	x, y := tile%columns, tile/columns
	r := &jxrFrequencyRows{width: int(s.header.columns[x+1] - s.header.columns[x]), height: int(s.header.rows[y+1] - s.header.rows[y]), count: count}
	r.quant[0] = newJXRTileQuant(&s.primary)
	if s.alpha != nil {
		r.quant[1] = newJXRTileQuant(s.alpha)
	}
	for band := range count {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		packetIndex := tile*count + band
		if band == 3 && jxrEmptyFlexPacket(s, data, packetIndex) {
			r.trim = 15
			continue
		}
		r.bits[band] = jxrPacketReader(data, s, packetIndex)
		if s.input != nil {
			r.bits[band].input = &jxrReaderAt{source: s.input.source, ctx: ctx, base: s.input.base, size: s.input.size}
		}
		packet, err := readJXRPacket(&r.bits[band], s, band, &r.quant)
		if err != nil {
			return nil, err
		}
		if band == 3 {
			r.trim, r.flexValid = packet.trim, packet.valid
			if !packet.valid {
				r.trim = 15
			}
		}
	}
	for i, p := range [2]*jxrPlane{&s.primary, s.alpha} {
		if p == nil {
			continue
		}
		r.channels[i] = p.channels
		r.bands[i] = p.bands
		for band := range min(3, 4-int(p.bands)) {
			var err error
			r.rows[band][i], err = newJXRTileRow(ctx, r.width, p.channels, maximum)
			if err != nil {
				return nil, err
			}
		}
		var err error
		r.scratch[i], err = newJXRDecodedMacro(ctx, p.channels, maximum)
		if err != nil {
			return nil, err
		}
		r.dc[i], r.lp[i], r.hp[i] = newJXRDC(), newJXRLP(), newJXRHP()
	}
	return r, ctx.Err()
}

// readRow 依次恢复当前宏块行的各频带，各频带预测缓存不与其他频带或反变换后的目标混用
// 入参: ctx Context, target 单行或环形系数目标, start 目标首宏块下标
// 返回: error 数据、目标容量、数值或取消错误，末行校验所有包尾对齐
func (r *jxrFrequencyRows) readRow(ctx context.Context, target *jxrCoefficientImage, start int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if target == nil || r.next >= r.height || start < 0 {
		return fmt.Errorf("%w: JPEG XR frequency row sequence", ErrInvalid)
	}
	planes := [2]*jxrCoefficients{&target.primary, target.alpha}
	for i, p := range planes {
		if p == nil {
			if r.channels[i] != 0 {
				return fmt.Errorf("%w: JPEG XR frequency row missing plane", ErrInvalid)
			}
			continue
		}
		if p.plane.channels < 1 || p.plane.channels != r.channels[i] || p.plane.bands != r.bands[i] || start > len(p.info)-r.width || len(p.low)/(16*p.plane.channels) < start+r.width || len(p.high)/(256*p.plane.channels) < start+r.width {
			return fmt.Errorf("%w: JPEG XR frequency row coefficient storage", ErrInvalid)
		}
		clear(p.info[start : start+r.width])
		clear(p.low[start*16*p.plane.channels : (start+r.width)*16*p.plane.channels])
		clear(p.high[start*256*p.plane.channels : (start+r.width)*256*p.plane.channels])
	}
	for band := range r.count {
		b := &r.bits[band]
		if band < 3 {
			for _, row := range r.rows[band] {
				if row != nil {
					row.startRow()
				}
			}
		}
		for x := range r.width {
			if err := ctx.Err(); err != nil {
				return err
			}
			for i, p := range planes {
				if p == nil || band >= 4-int(p.plane.bands) {
					continue
				}
				at, q := start+x, &r.quant[i]
				info := &p.info[at]
				if band == 3 {
					if err := refineJXRHigh(b, &p.plane, info.bits, r.trim, p.highMacro(at)); err != nil {
						return err
					}
					if err := p.finishHighComponents(at, q, r.scratch[i].low); err != nil {
						return err
					}
					continue
				}
				row := r.rows[band][i]
				m := jxrMacroContext{at: at, firstColumn: x == 0, firstRow: r.next == 0, reset: x%16 == 0, adapt: x == r.width-1 || x%16 == 0, left: &row.left, top: &row.top[x], corner: &row.corner}
				current := &row.current
				var err error
				switch band {
				case 0:
					err = p.readDC(b, &r.dc[i], q, m, &current.low, r.scratch[i])
				case 1:
					if q.lpCount > 1 && !q.useDC {
						info.lpIndex = uint8(readJXRQPIndex(b, q.lpCount))
					}
					err = p.readLP(b, &r.lp[i], q, m, &current.low, r.scratch[i])
				case 2:
					if q.useLP {
						info.hpIndex = info.lpIndex
					} else if q.hpCount > 1 {
						info.hpIndex = uint8(readJXRQPIndex(b, q.hpCount))
					}
					err = p.readHP(b, &r.hp[i], m, false, 0, row.residual, current.pattern)
					if err == nil && p.plane.bands == 1 {
						err = refineJXRHigh(b, &p.plane, info.bits, 15, p.highMacro(at))
						if err == nil {
							err = p.finishHighComponents(at, q, r.scratch[i].low)
						}
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
	r.next++
	if r.next == r.height {
		for band := range r.count {
			if band == 3 && !r.flexValid {
				continue
			}
			r.bits[band].align(true)
			if r.bits[band].err != nil {
				return r.bits[band].err
			}
		}
	}
	return ctx.Err()
}

// walkJXRReconstructedRows 从已预检的连续熵状态单遍恢复最终原生行，不构造整幅样本图或进行色度上采样
// 入参: ctx Context, s 原始分块几何, reader 连续熵读取器, maximum 重建额度, emit 同步行消费者
// 返回: error 数据、数值、资源、取消或消费者错误，已消费行不回滚
func walkJXRReconstructedRows(ctx context.Context, s *jxrStream, reader jxrCoefficientRows, maximum uint64, emit jxrReconstructedRow) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if reader == nil || emit == nil {
		return fmt.Errorf("%w: JPEG XR reconstructed row consumer", ErrInvalid)
	}
	if jxrStreamingRowWork(s) > min(maximum, uint64(^uint(0)>>1)) {
		return fmt.Errorf("%w: JPEG XR reconstructed row memory", ErrLimit)
	}
	rows, err := newJXRReconstructedRows(ctx, s, reader, maximum)
	if err != nil {
		return err
	}
	var storage [17]jxrComponentView
	for rows.nextRow < rows.height {
		row, macroY, err := rows.next(ctx)
		if err != nil {
			return err
		}
		views := row.primary.componentViewsStorage(&row.header, true, storage[:16:16])
		var alpha *jxrComponentView
		if row.alpha != nil {
			alphaViews := row.alpha.componentViewsStorage(&row.header, true, storage[16:17:17])
			alpha = &alphaViews[0]
		}
		if err := emit(ctx, &row.header, views, alpha, macroY); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// newJXRReconstructedRows 在熵读取前检查额度并创建固定重建窗口，不提前读取任何包
// 入参: ctx Context, s 原始几何及平面, reader 连续熵状态, maximum 重建额度
// 返回: *jxrReconstructedRows 按需行状态, error 资源、参数或取消错误
func newJXRReconstructedRows(ctx context.Context, s *jxrStream, reader jxrCoefficientRows, maximum uint64) (*jxrReconstructedRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if reader == nil {
		return nil, fmt.Errorf("%w: JPEG XR reconstructed row reader", ErrInvalid)
	}
	if jxrStreamingRowWork(s) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR reconstructed row cursor memory", ErrLimit)
	}
	frame := jxrRowReconstructionStream(s)
	r := &jxrReconstructedRows{reader: reader, height: int(frame.header.rows[1])}
	if s.header.overlap != 0 {
		pipeline, err := newJXRInverseRows(ctx, &frame, maximum)
		if err != nil {
			return nil, err
		}
		pipeline.header = s.header
		r.pipeline = pipeline
	} else {
		h := frame.header
		h.height, h.top, h.bottom, h.rows = 16, 0, 0, []uint32{0, 1}
		r.raw = jxrCoefficientImage{header: h, primary: newJXRCoefficients(&s.primary, int(h.columns[1]), h.longWord)}
		if s.alpha != nil {
			alpha := newJXRCoefficients(s.alpha, int(h.columns[1]), h.longWord)
			r.raw.alpha = &alpha
		}
	}
	return r, ctx.Err()
}

// next 推进到下一最终行，借用结果仅在下次推进前有效，最后两行只执行一次真实末边界刷新
// 入参: ctx Context
// 返回: *jxrCoefficientImage 最终样本窗口, int 原宏块行号, error 完结时为io.EOF，其他错误立即停止
func (r *jxrReconstructedRows) next(ctx context.Context) (*jxrCoefficientImage, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if r.nextRow >= r.height {
		return nil, 0, io.EOF
	}
	row := r.nextRow
	if p := r.pipeline; p != nil {
		for r.read < min(r.height, row+3) {
			y := r.read
			if err := r.reader.readRow(ctx, &p.raw, (y%3)*p.width); err != nil {
				return nil, 0, err
			}
			if err := p.inverseFirst(ctx, y); err != nil {
				return nil, 0, err
			}
			if y > 0 {
				if err := p.inverseSecond(ctx, y-1); err != nil {
					return nil, 0, err
				}
			}
			r.read++
		}
		if row >= r.height-2 && !r.flushed {
			if err := p.inverseSecond(ctx, r.height-1); err != nil {
				return nil, 0, err
			}
			r.flushed = true
		}
		state, err := p.outputRow(ctx, row)
		if err != nil {
			return nil, 0, err
		}
		r.nextRow++
		return state, row, ctx.Err()
	}
	if err := r.reader.readRow(ctx, &r.raw, 0); err != nil {
		return nil, 0, err
	}
	for _, plane := range [2]*jxrCoefficients{&r.raw.primary, r.raw.alpha} {
		if plane != nil {
			if err := plane.inverseFirstRange(ctx, &r.raw.header, 0, len(plane.info)); err != nil {
				return nil, 0, err
			}
			if err := plane.inverseSecondRange(ctx, &r.raw.header, 0, len(plane.info)); err != nil {
				return nil, 0, err
			}
		}
	}
	r.nextRow++
	return &r.raw, row, ctx.Err()
}

// jxrInverseRowWork 计入连续熵状态和九行系数容量，工作量不随分块高度增长
// 入参: s 单分块码流
// 返回: uint64 工作字节数，无效布局或溢出返回最大值
func jxrInverseRowWork(s *jxrStream) uint64 {
	base := jxrSpatialRowWork(s, 0)
	if s.header.frequency {
		base = jxrFrequencyRowWork(s, 0)
	}
	if len(s.header.columns) != 2 || len(s.header.rows) != 2 || s.primary.channels < 1 || s.primary.channels > rasterMaxComponents {
		return ^uint64(0)
	}
	channels, planes := uint64(s.primary.channels), uint64(1)
	if s.alpha != nil {
		if s.alpha.channels < 1 || s.alpha.channels > rasterMaxComponents {
			return ^uint64(0)
		}
		channels += uint64(s.alpha.channels)
		planes++
	}
	stride := channels*272*8 + planes*6
	extra := uint64(s.header.columns[1]) * stride * 8
	if extra > ^uint64(0)-base {
		return ^uint64(0)
	}
	return base + extra
}

// newJXRInverseRows 在额度内建立独立系数及样本环，色度低通及样本按各自网格处理
// 入参: ctx Context, s 单分块码流, maximum 行重建工作额度
// 返回: *jxrInverseRows 行重建状态, error 布局、资源或取消错误
func newJXRInverseRows(ctx context.Context, s *jxrStream, maximum uint64) (*jxrInverseRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrInverseRowWork(s) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR inverse row working budget", ErrLimit)
	}
	if s.header.overlap < 1 || s.header.overlap > 2 {
		return nil, fmt.Errorf("%w: JPEG XR inverse row overlap format", ErrUnsupported)
	}
	r := &jxrInverseRows{header: s.header, width: int(s.header.columns[1]), height: int(s.header.rows[1])}
	h := s.header
	h.height, h.top, h.bottom, h.rows = 48, 0, 0, []uint32{0, 3}
	for _, buffer := range []*jxrCoefficientImage{&r.raw, &r.samples, &r.work} {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		buffer.header = h
		buffer.primary = newJXRCoefficients(&s.primary, r.width*3, h.longWord)
		if s.alpha != nil {
			alpha := newJXRCoefficients(s.alpha, r.width*3, h.longWord)
			buffer.alpha = &alpha
		}
	}
	return r, ctx.Err()
}

// jxrCoefficientRow 建立已预算系数的连续行视图，变换只触及该行而不改相邻预测缓存
// 入参: source 系数缓冲, first 首宏块, count 连续宏块数
// 返回: jxrCoefficients 共享存储的行视图
func jxrCoefficientRow(source *jxrCoefficients, first, count int) jxrCoefficients {
	row := *source
	low, high := source.plane.channels*16, source.plane.channels*256
	row.low = source.low[first*low : (first+count)*low]
	row.high = source.high[first*high : (first+count)*high]
	row.info = source.info[first : first+count]
	return row
}

// inverseFirst 仅对刚读入的原始环形行执行首级逆变换，低通结果保留给相邻行后滤波
// 入参: ctx Context, row 全块宏块行号
// 返回: error 数值范围、行号或取消错误
func (r *jxrInverseRows) inverseFirst(ctx context.Context, row int) error {
	if row < 0 || row >= r.height {
		return fmt.Errorf("%w: JPEG XR inverse first row", ErrInvalid)
	}
	h := r.raw.header
	h.height, h.rows = 16, []uint32{0, 1}
	for _, plane := range [2]*jxrCoefficients{&r.raw.primary, r.raw.alpha} {
		if plane != nil {
			view := jxrCoefficientRow(plane, (row%3)*r.width, r.width)
			if err := view.inverseFirstRange(ctx, &h, 0, len(view.info)); err != nil {
				return err
			}
		}
	}
	return ctx.Err()
}

// prepareWindow 复制目标行及相邻原始值到独立工作区，人工上下边缘不会冒充真实图像边缘
// 入参: ctx Context, row 全块宏块行号, samples 是否复制二级样本而非首级系数
// 返回: image.Rectangle 局部目标宏块行, [4]bool 左上右下真实边缘, error 行号或取消错误
func (r *jxrInverseRows) prepareWindow(ctx context.Context, row int, samples bool) (image.Rectangle, [4]bool, error) {
	if row < 0 || row >= r.height {
		return image.Rectangle{}, [4]bool{}, fmt.Errorf("%w: JPEG XR inverse row window", ErrInvalid)
	}
	first, last := max(0, row-1), min(r.height, row+2)
	top, bottom := 0, r.height
	if r.header.hardTiles {
		for tileRow := 0; tileRow+1 < len(r.header.rows); tileRow++ {
			if row < int(r.header.rows[tileRow+1]) {
				top, bottom = int(r.header.rows[tileRow]), int(r.header.rows[tileRow+1])
				break
			}
		}
		first, last = max(first, top), min(last, bottom)
		r.work.header.columns = r.header.columns
	}
	r.work.header.height, r.work.header.top, r.work.header.bottom = uint64((last-first)*16), 0, 0
	r.rowEnds[1] = uint32(last - first)
	r.work.header.rows = r.rowEnds[:]
	for i, target := range [2]*jxrCoefficients{&r.work.primary, r.work.alpha} {
		if target == nil {
			continue
		}
		source := [2]*jxrCoefficients{&r.raw.primary, r.raw.alpha}[i]
		if samples {
			source = [2]*jxrCoefficients{&r.samples.primary, r.samples.alpha}[i]
		}
		for y := first; y < last; y++ {
			if err := ctx.Err(); err != nil {
				return image.Rectangle{}, [4]bool{}, err
			}
			stride := source.plane.channels * 16 * r.width
			from, to := source.low, target.low
			if samples {
				stride *= 16
				from, to = source.high, target.high
			}
			copy(to[(y-first)*stride:(y-first+1)*stride], from[(y%3)*stride:(y%3+1)*stride])
		}
	}
	return image.Rect(0, row-first, r.width, row-first+1), [4]bool{true, first == top, true, last == bottom}, nil
}

// inverseSecond 在首级邻行可用后恢复目标行样本，结果保留到独立样本环等待二级后滤波
// 入参: ctx Context, row 全块宏块行号
// 返回: error 邻域、数值范围或取消错误
func (r *jxrInverseRows) inverseSecond(ctx context.Context, row int) error {
	core, edges, err := r.prepareWindow(ctx, row, false)
	if err != nil {
		return err
	}
	h := r.work.header
	for i, plane := range [2]*jxrCoefficients{&r.work.primary, r.work.alpha} {
		if plane == nil {
			continue
		}
		if r.header.overlap == 2 {
			if err := plane.filterWindow(ctx, &h, false, core, edges, false); err != nil {
				return err
			}
		}
		raw := [2]*jxrCoefficients{&r.raw.primary, r.raw.alpha}[i]
		view := jxrCoefficientRow(plane, core.Min.Y*r.width, r.width)
		original := jxrCoefficientRow(raw, (row%3)*r.width, r.width)
		view.high, view.info = original.high, original.info
		rowHeader := h
		rowHeader.height, rowHeader.rows = 16, []uint32{0, 1}
		if err := view.inverseSecondRange(ctx, &rowHeader, 0, len(view.info)); err != nil {
			return err
		}
		target := [2]*jxrCoefficients{&r.samples.primary, r.samples.alpha}[i]
		stride := r.width * plane.plane.channels * 256
		copy(target.high[(row%3)*stride:(row%3+1)*stride], view.high)
	}
	return ctx.Err()
}

// outputRow 在二级邻行可用后执行样本后滤波，返回的窗口视图仅在下一次状态操作前有效
// 入参: ctx Context, row 全块宏块行号
// 返回: *jxrCoefficientImage 已重建样本窗口，header.top指向目标行, error 邻域、数值范围或取消错误
func (r *jxrInverseRows) outputRow(ctx context.Context, row int) (*jxrCoefficientImage, error) {
	core, edges, err := r.prepareWindow(ctx, row, true)
	if err != nil {
		return nil, err
	}
	for _, plane := range [2]*jxrCoefficients{&r.work.primary, r.work.alpha} {
		if plane != nil {
			if err := plane.filterWindow(ctx, &r.work.header, true, core, edges, false); err != nil {
				return nil, err
			}
		}
	}
	r.work.header.top = uint32(core.Min.Y * 16)
	return &r.work, ctx.Err()
}

// jxrNeighborhoodRowWork 计算当前分块行各列熵状态、合计一行系数和目录描述，不按分块行累积熵缓存
// 入参: s 至多五乘五分块的局部码流
// 返回: uint64 行工作字节数，无效几何或溢出返回最大值
func jxrNeighborhoodRowWork(s *jxrStream) uint64 {
	columns, rows := len(s.header.columns)-1, len(s.header.rows)-1
	if columns < 1 || columns > 5 || rows < 1 || rows > 5 {
		return ^uint64(0)
	}
	return jxrTileRowsWork(s)
}

// jxrTileRowsWork 计算当前分块整行的独立熵容量，不按分块行累计已结束状态
// 入参: s 已校验的完整或局部码流
// 返回: uint64 描述及熵状态容量，无效几何或溢出返回最大值
func jxrTileRowsWork(s *jxrStream) uint64 {
	columns, rows := len(s.header.columns)-1, len(s.header.rows)-1
	if columns < 1 || columns > 4096 || rows < 1 || rows > 4096 {
		return ^uint64(0)
	}
	work := uint64(1024)
	if columns > 5 {
		work += uint64(columns) * 16
	}
	for column := range columns {
		amount := jxrSpatialRowWork(s, column)
		if s.header.frequency {
			amount = jxrFrequencyRowWork(s, column)
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

// jxrRowReconstructionStream 建立用于行容量与连续存储布局的全宽行几何视图，不改写包字节或原始分块边界
// 入参: s 已校验的局部码流，至少各有两个列行端点
// 返回: jxrStream 共享原码流描述及全宽连续行几何
func jxrRowReconstructionStream(s *jxrStream) jxrStream {
	local := *s
	if len(s.header.columns) != 2 || len(s.header.rows) != 2 {
		local.header.columns = []uint32{0, s.header.columns[len(s.header.columns)-1]}
		local.header.rows = []uint32{0, s.header.rows[len(s.header.rows)-1]}
	}
	return local
}

// newJXRNeighborhoodRows 在读取任何包前检查合计工作，实际各列状态延迟至对应原分块行才创建
// 入参: ctx Context, data 可选完整字节, s 已校验局部码流, maximum 合计行额度
// 返回: *jxrTileRows 连续局部行读取器, error 几何、模型、资源或取消错误
func newJXRNeighborhoodRows(ctx context.Context, data []byte, s *jxrStream, maximum uint64) (*jxrTileRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrNeighborhoodRowWork(s) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR neighborhood row memory", ErrLimit)
	}
	return newJXRTileRows(ctx, data, s, maximum)
}

// newJXRTileRows 在完整网格行额度内创建惰性熵调度器，构造期间不读取包或分配所有分块行状态
// 入参: ctx Context, data 可选完整字节, s 已校验完整或局部流, maximum 当前行总额度
// 返回: *jxrTileRows 连续行读取器, error 几何、模型、资源或取消错误
func newJXRTileRows(ctx context.Context, data []byte, s *jxrStream, maximum uint64) (*jxrTileRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if jxrTileRowsWork(s) > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR tile row memory", ErrLimit)
	}
	for _, plane := range [2]*jxrPlane{&s.primary, s.alpha} {
		if plane != nil {
			if err := checkJXRModelComponents(plane); err != nil {
				return nil, err
			}
		}
	}
	r := &jxrTileRows{source: *s, data: data, maximum: maximum, width: int(s.header.columns[len(s.header.columns)-1]), height: int(s.header.rows[len(s.header.rows)-1]), activeRow: -1}
	columns := len(s.header.columns) - 1
	r.readers = r.storage[:min(columns, len(r.storage))]
	if columns > len(r.storage) {
		r.readers = make([]jxrCoefficientRows, columns)
	}
	return r, nil
}

// readRow 保持每个原始分块内的行和频带顺序，将各列系数写入同一连续目标行的独立区间
// 入参: ctx Context, target 单行或环形系数缓冲, start 目标首宏块下标
// 返回: error 行序列、容量、包、数值或取消错误，失败后不可原地复用
func (r *jxrTileRows) readRow(ctx context.Context, target *jxrCoefficientImage, start int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if target == nil || start < 0 || r.next >= r.height {
		return fmt.Errorf("%w: JPEG XR neighborhood row sequence", ErrInvalid)
	}
	for index, plane := range [2]*jxrCoefficients{&target.primary, target.alpha} {
		expected := [2]*jxrPlane{&r.source.primary, r.source.alpha}[index]
		if plane == nil && expected == nil {
			continue
		}
		if plane == nil || expected == nil || plane.plane.channels != expected.channels || plane.plane.bands != expected.bands || start > len(plane.info)-r.width || len(plane.low)/(16*expected.channels) < start+r.width || len(plane.high)/(256*expected.channels) < start+r.width {
			return fmt.Errorf("%w: JPEG XR neighborhood row storage", ErrInvalid)
		}
	}
	for r.next >= int(r.source.header.rows[r.tileRow+1]) {
		r.tileRow++
	}
	columns := len(r.source.header.columns) - 1
	if r.activeRow != r.tileRow {
		clear(r.readers)
		for column := range columns {
			local := r.source
			var reader jxrCoefficientRows
			var err error
			if local.header.frequency {
				reader, err = newJXRFrequencyRows(ctx, r.data, &local, r.tileRow*columns+column, r.maximum)
			} else {
				if local.input != nil {
					local.input = &jxrReaderAt{source: local.input.source, ctx: ctx, base: local.input.base, size: local.input.size}
				}
				reader, err = newJXRSpatialRows(ctx, r.data, &local, r.tileRow*columns+column, r.maximum)
			}
			if err != nil {
				return err
			}
			r.readers[column] = reader
		}
		r.activeRow = r.tileRow
	}
	for column := range columns {
		if err := r.readers[column].readRow(ctx, target, start+int(r.source.header.columns[column])); err != nil {
			return err
		}
	}
	r.next++
	return ctx.Err()
}

// copy 将最终样本同步交给唯一具体目标，不保存借用的行视图
// 入参: ctx Context, h 最终窗口, views 主分量, alpha 透明度, row 原宏块行号
// 返回: error 输出数据、数值或取消错误
func (output jxrSampleRowOutput) copy(ctx context.Context, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error {
	if output.strip != nil {
		return output.strip.copy(ctx, h, views, alpha, row)
	}
	if output.region == nil {
		return fmt.Errorf("%w: JPEG XR finalized sample target", ErrInvalid)
	}
	return output.region.copy(ctx, h, views, alpha, row)
}

// jxrUpsampleRowWork 计算420全速输出的三行最终样本缓存，其他输出不需要该容量
// 入参: s 单个局部块的已校验码流
// 返回: uint64 附加容量字节数，几何或分量无效时返回最大值
func jxrUpsampleRowWork(s *jxrStream) uint64 {
	if s.primary.color != 1 || s.header.color != 3 && s.header.color != 7 {
		return 0
	}
	if len(s.header.columns) != 2 || len(s.header.rows) != 2 || s.header.columns[0] != 0 || s.header.rows[0] != 0 || s.header.rows[1] == 0 || s.primary.channels < 1 || s.primary.channels > rasterMaxComponents {
		return ^uint64(0)
	}
	channels := uint64(s.primary.channels)
	if s.alpha != nil {
		if s.alpha.channels != 1 {
			return ^uint64(0)
		}
		channels++
	}
	stride := channels * 256 * 8 * 3
	width := uint64(s.header.columns[1])
	if width == 0 || width > (^uint64(0)-1024)/stride {
		return ^uint64(0)
	}
	return width*stride + 1024
}

// jxrCoefficientRowWork 合并熵、必要重叠窗口及最终色度插值邻行的全部容量
// 入参: s 单个局部块码流
// 返回: uint64 行工作字节数，溢出或无效布局返回最大值
func jxrCoefficientRowWork(s *jxrStream) uint64 {
	if len(s.header.columns) > 6 || len(s.header.rows) > 6 {
		return ^uint64(0)
	}
	return jxrStreamingRowWork(s)
}

// jxrStreamingRowWork 计算完整分块网格的熵状态、固定逆变换及最终色度行缓存，不分配整幅系数
// 入参: s 已校验完整或局部码流
// 返回: uint64 总行容量，无效几何或溢出返回最大值
func jxrStreamingRowWork(s *jxrStream) uint64 {
	if len(s.header.columns) < 2 || len(s.header.rows) < 2 {
		return ^uint64(0)
	}
	multiple := len(s.header.columns) != 2 || len(s.header.rows) != 2
	local := jxrRowReconstructionStream(s)
	base := jxrSpatialRowWork(&local, 0)
	if local.header.frequency {
		base = jxrFrequencyRowWork(&local, 0)
	}
	work := base
	if local.header.overlap != 0 {
		work = jxrInverseRowWork(&local)
	}
	if multiple {
		entropy := jxrTileRowsWork(s)
		if work < base || entropy > ^uint64(0)-(work-base) {
			return ^uint64(0)
		}
		work = entropy + (work - base)
	}
	extra := jxrUpsampleRowWork(&local)
	if extra > ^uint64(0)-work {
		return ^uint64(0)
	}
	return work + extra
}

// newJXRUpsampleRows 在附加预算内创建独立最终样本缓存，不分配低通系数或宏块预测元数据
// 入参: ctx Context, s 单块420全速输出码流, maximum 当前块总额度
// 返回: *jxrUpsampleRows 最终样本状态, error 格式、资源或取消错误
func newJXRUpsampleRows(ctx context.Context, s *jxrStream, maximum uint64) (*jxrUpsampleRows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	work := jxrUpsampleRowWork(s)
	if work == 0 {
		return nil, fmt.Errorf("%w: JPEG XR vertical sample rows", ErrInvalid)
	}
	if work > min(maximum, uint64(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: JPEG XR vertical sample row memory", ErrLimit)
	}
	r := &jxrUpsampleRows{width: int(s.header.columns[1]), height: int(s.header.rows[1])}
	r.rows[1] = 3
	r.image.header = s.header
	r.image.header.height, r.image.header.top, r.image.header.bottom = 48, 0, 0
	r.image.header.rows = r.rows[:]
	r.image.primary = jxrCoefficients{plane: s.primary, high: make([]int64, r.width*s.primary.channels*256*3), longWord: s.header.longWord}
	if s.alpha != nil {
		r.image.alpha = &jxrCoefficients{plane: *s.alpha, high: make([]int64, r.width*256*3), longWord: s.header.longWord}
	}
	return r, ctx.Err()
}

// accept 保存连续最终样本行并在下一行已完成时输出前一行，始终保留真实上下插值邻域
// 入参: ctx Context, output 区域输出, h 当前样本窗口, views 主平面样本, alpha 透明度样本, row 原块行号
// 返回: error 行序列、容量、数值或取消错误
func (r *jxrUpsampleRows) accept(ctx context.Context, output jxrSampleRowOutput, h *jxrHeader, views []jxrComponentView, alpha *jxrComponentView, row int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if row != r.count || row >= r.height || len(views) != r.image.primary.plane.channels || h.top%16 != 0 {
		return fmt.Errorf("%w: JPEG XR vertical sample row sequence", ErrInvalid)
	}
	slot := min(r.count, 2)
	for i, p := range [2]*jxrCoefficients{&r.image.primary, r.image.alpha} {
		if p == nil {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		view := &views[0]
		if i == 1 {
			view = alpha
		}
		stride := r.width * p.plane.channels * 256
		start := int(h.top/16) * stride
		if view == nil || view.columns != r.width || view.micro != 4 || view.stride != p.plane.channels*256 || view.offset != 0 || start > len(view.data)-stride {
			return fmt.Errorf("%w: JPEG XR vertical sample row storage", ErrInvalid)
		}
		if r.count >= 3 {
			copy(p.high[:2*stride], p.high[stride:3*stride])
		}
		copy(p.high[slot*stride:(slot+1)*stride], view.data[start:start+stride])
	}
	r.count++
	if r.count < 2 {
		return ctx.Err()
	}
	return r.emit(ctx, output, r.count-2, min(r.count, 3)-2)
}

// emit 在已完成的邻域中按原始扩展网格插值，局部top只定位当前行而不改变可见裁剪相位
// 入参: ctx Context, output 区域输出, row 原块行号, local 缓存内目标行号
// 返回: error 数值、输出或取消错误
func (r *jxrUpsampleRows) emit(ctx context.Context, output jxrSampleRowOutput, row, local int) error {
	r.rows[1] = uint32(min(r.count, 3))
	r.image.header.height, r.image.header.top = uint64(r.rows[1])*16, uint32(local)*16
	var storage [17]jxrComponentView
	views := r.image.primary.componentViewsStorage(&r.image.header, true, storage[:16:16])
	var alpha *jxrComponentView
	if r.image.alpha != nil {
		alphaViews := r.image.alpha.componentViewsStorage(&r.image.header, true, storage[16:17:17])
		alpha = &alphaViews[0]
	}
	return output.copy(ctx, &r.image.header, views, alpha, row)
}

// finish 仅在全部原始行已校验后刷新最后一行，末边界由真实编码网格长度延拓
// 入参: ctx Context, output 区域输出
// 返回: error 缺行、数值、输出或取消错误
func (r *jxrUpsampleRows) finish(ctx context.Context, output jxrSampleRowOutput) error {
	if r.count != r.height || r.count < 1 {
		return fmt.Errorf("%w: JPEG XR incomplete vertical sample rows", ErrInvalid)
	}
	return r.emit(ctx, output, r.count-1, min(r.count, 3)-1)
}

// decodeUncompressedStripsLocked 按至多十六编码行一批读取无压缩容器图像并同步回调可复用行带
// 入参: ctx 已规范化Context, write 同步行带消费者
// 返回: error 数据、读取、资源、取消或消费者错误
func (d *Decoder) decodeUncompressedStripsLocked(ctx context.Context, write RasterStripWriter) error {
	d.bindInputContext(ctx)
	defer d.bindInputContext(nil)
	c := &d.source.container
	layout, err := jxrMakeUncompressedLayout(c.pixelFormat[15], c.format, c.width, c.height, c.separateAlpha())
	if err != nil {
		return err
	}
	codedW, codedH := int(c.width), int(c.height)
	transform := uint8(c.transform)
	firstHeight := min(16, codedH)
	firstRegion := jxrRegionPosition(image.Rect(0, 0, codedW, firstHeight), codedW, codedH, transform)
	firstInfo, stripWork, err := jxrRegionInfo(d.info, firstRegion)
	if err != nil {
		return err
	}
	rowWork := layout.rowBufferWork()
	retained := d.inputWork() + d.retainedHeaderWork()
	if _, ok := jxrWorkTotal(d.limits.MaxMemoryBytes, retained, stripWork, rowWork); !ok {
		return &LimitError{Resource: "max memory bytes", Limit: d.limits.MaxMemoryBytes, Required: jxrSaturatedAdd(retained, jxrSaturatedAdd(stripWork, rowWork)), Detail: "uncompressed strip memory budget"}
	}
	if rowWork > uint64(^uint(0)>>1) {
		return &LimitError{Resource: "max memory bytes", Limit: uint64(^uint(0) >> 1), Required: rowWork, Detail: "uncompressed strip row capacity"}
	}
	cache, err := newRasterContext(ctx, firstInfo, Limits{MaxPixels: d.limits.MaxPixels, MaxMemoryBytes: stripWork})
	if err != nil {
		return err
	}
	rowBuf := make([]byte, int(layout.rowBytes))
	var alphaBuf []byte
	if !layout.planar && layout.separateAlpha {
		alphaBuf = make([]byte, int(layout.alphaRowBytes))
	}
	view := Raster{planes: make([]imageComponents, len(cache.planes))}
	viewComponents := make([]ComponentInfo, len(cache.info.Components))
	for y0 := 0; y0 < codedH; y0 += 16 {
		if err := ctx.Err(); err != nil {
			return err
		}
		y1 := min(y0+16, codedH)
		dispRect := jxrRegionPosition(image.Rect(0, y0, codedW, y1), codedW, codedH, transform)
		copy(viewComponents, cache.info.Components)
		view.info = cache.info
		view.info.Bounds = image.Rect(0, 0, dispRect.Dx(), dispRect.Dy())
		view.info.Components = viewComponents
		for ci := range view.planes {
			comp := &viewComponents[ci]
			comp.Bounds = image.Rect(0, 0, dispRect.Dx()/int(comp.XStep), dispRect.Dy()/int(comp.YStep))
			count := comp.Bounds.Dx() * comp.Bounds.Dy()
			plane := cache.planes[ci]
			plane.bounds = comp.Bounds
			if plane.words != nil {
				plane.words = plane.words[:count]
			} else {
				plane.values = plane.values[:count]
			}
			view.planes[ci] = plane
		}
		if err := d.decodeUncompressedInto(ctx, &view, dispRect, &layout, rowBuf, alphaBuf); err != nil {
			return err
		}
		if err := write(ctx, dispRect, &view); err != nil {
			return err
		}
	}
	return ctx.Err()
}
