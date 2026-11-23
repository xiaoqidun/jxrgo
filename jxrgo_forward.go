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
)

// jxrForwardWindow 指定局部网格的目标宏块和真实图像四边，人工邻域边缘不执行图像边缘滤波
type jxrForwardWindow struct {
	core  image.Rectangle
	edges [4]bool
}

// newJXRTransformPlane 校验宏块布局并在预算内分配编码侧低通、高通及状态缓冲
// 入参: ctx 已规范化取消上下文, h 图像头, p 平面信息, maxWork 新缓冲内存预算
// 返回: *jxrCoefficients 空变换平面, error 布局、资源或取消错误
func newJXRTransformPlane(ctx context.Context, h *jxrHeader, p *jxrPlane, maxWork uint64) (*jxrCoefficients, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := checkJXRPlaneComponents(p); err != nil {
		return nil, err
	}
	if len(h.columns) < 2 || len(h.rows) < 2 || h.overlap > 2 || h.width == 0 || h.height == 0 || h.width > 1<<32 || h.height > 1<<32 {
		return nil, fmt.Errorf("%w: JPEG XR transform plane layout", ErrInvalid)
	}
	width, height := h.width+uint64(h.left)+uint64(h.right), h.height+uint64(h.top)+uint64(h.bottom)
	if width%16 != 0 || height%16 != 0 || (h.color == 1 || h.color == 2) && (h.width%2 != 0 || h.left%2 != 0 || h.right%2 != 0) || h.color == 1 && (h.height%2 != 0 || h.top%2 != 0 || h.bottom%2 != 0) {
		return nil, fmt.Errorf("%w: JPEG XR transform window alignment", ErrInvalid)
	}
	if width > uint64(^uint(0)>>1) || height > uint64(^uint(0)>>1) {
		return nil, &LimitError{Resource: "JPEG XR transform extent", Limit: uint64(^uint(0) >> 1), Required: max(width, height)}
	}
	for index, edges := range [][]uint32{h.columns, h.rows} {
		extent := width / 16
		if index == 1 {
			extent = height / 16
		}
		if edges[0] != 0 || uint64(edges[len(edges)-1]) != extent || len(edges) > 4097 {
			return nil, fmt.Errorf("%w: JPEG XR transform tile extent", ErrInvalid)
		}
		for at := 1; at < len(edges); at++ {
			if edges[at] <= edges[at-1] {
				return nil, fmt.Errorf("%w: JPEG XR transform tile order", ErrInvalid)
			}
		}
	}
	if (p.color == 1 || p.color == 2) && h.overlap == 2 {
		if width < 32 {
			return nil, fmt.Errorf("%w: JPEG XR transform chroma overlap width", ErrInvalid)
		}
		if h.hardTiles {
			for at := 1; at < len(h.columns); at++ {
				if h.columns[at]-h.columns[at-1] < 2 {
					return nil, fmt.Errorf("%w: JPEG XR transform chroma tile width", ErrInvalid)
				}
			}
		}
	}
	count := width / 16 * (height / 16)
	stride := uint64(p.channels)*272*8 + 6
	metadata := uint64(512) + uint64(p.channels)*512
	if metadata > maxWork || count > (maxWork-metadata)/stride || count > uint64(^uint(0)>>1)/(uint64(p.channels)*256*8) {
		return nil, &LimitError{Resource: "JPEG XR transform working storage", Limit: maxWork, Required: jxrSaturatedAdd(metadata, jxrSaturatedMul(count, stride))}
	}
	plane := newJXRCoefficients(p, int(count), h.longWord)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &plane, nil
}

// forward 将已填充的完整样本网格转换为两级系数，不执行预测、量化或熵编码
// 入参: ctx 已规范化取消上下文, h 图像头
// 返回: error 变换、资源或取消错误
func (c *jxrCoefficients) forward(ctx context.Context, h *jxrHeader) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if h.overlap != 0 {
		if err := c.filterGrid(ctx, h, true, true); err != nil {
			return err
		}
	}
	if err := c.forwardFirst(ctx, h); err != nil {
		return err
	}
	if h.overlap == 2 {
		if err := c.filterGrid(ctx, h, false, true); err != nil {
			return err
		}
	}
	return c.forwardSecond(ctx, h)
}

// forwardFirst 对每个四乘四样本块执行正变换，将块直流值归入宏块低通网格
// 入参: ctx 取消上下文, h 图像头
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardFirst(ctx context.Context, h *jxrHeader) error {
	return c.forwardFirstRegion(ctx, h, image.Rect(0, 0, int(h.columns[len(h.columns)-1]), int(h.rows[len(h.rows)-1])))
}

// forwardFirstRegion 只变换已预滤波的宏块矩形，不对邻域人工边缘计算无关变换
// 入参: ctx 当前上下文, h 已校验布局, region 宏块坐标矩形
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardFirstRegion(ctx context.Context, h *jxrHeader, region image.Rectangle) error {
	total := region.Dx() * region.Dy()
	if c.workers > 1 && total >= 64 {
		workers, chunkSize, tasks := jxrMacroChunkPlan(total, 32, c.workers)
		if workers > 1 {
			return runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
				start := task * chunkSize
				return c.forwardFirstRange(runCtx, h, region, start, min(start+chunkSize, total))
			})
		}
	}
	return c.forwardFirstRange(ctx, h, region, 0, total)
}

// forwardFirstRange 对矩形内指定宏块区间执行首级四乘四正变换
// 入参: ctx 当前上下文, h 已校验布局, region 宏块坐标矩形, start 起始位置, end 结束位置
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardFirstRange(ctx context.Context, h *jxrHeader, region image.Rectangle, start, end int) error {
	columns := int(h.columns[len(h.columns)-1])
	width := region.Dx()
	for position := start; position < end; position++ {
		at := (region.Min.Y+position/width)*columns + region.Min.X + position%width
		if err := ctx.Err(); err != nil {
			return err
		}
		low, high := c.lowMacro(at), c.highMacro(at)
		for index := range c.plane.channels {
			if err := ctx.Err(); err != nil {
				return err
			}
			blocks := 16
			if index != 0 && c.plane.color == 1 {
				blocks = 4
			} else if index != 0 && c.plane.color == 2 {
				blocks = 8
			}
			for block := range blocks {
				var values [16]int64
				offset := index*256 + block*16
				copy(values[:], high[offset:offset+16])
				if err := checkJXRWords(values[:], h.longWord); err != nil {
					return err
				}
				check := newJXRArithmetic(h.longWord)
				jxrForward4(&values, &check)
				if err := check.error(); err != nil {
					return err
				}
				low[index*16+block], values[0] = values[0], 0
				copy(high[offset:offset+16], values[:])
			}
		}
	}
	return nil
}

// forwardSecond 对宏块低通网格执行正变换，按标准色度编码布局组织系数
// 入参: ctx 取消上下文, h 图像头
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardSecond(ctx context.Context, h *jxrHeader) error {
	return c.forwardSecondRegion(ctx, h, image.Rect(0, 0, int(h.columns[len(h.columns)-1]), int(h.rows[len(h.rows)-1])))
}

// forwardSecondRegion 只变换输出所需的宏块低通值，仍执行全部标准字长检查
// 入参: ctx 当前上下文, h 已校验布局, region 宏块坐标矩形
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardSecondRegion(ctx context.Context, h *jxrHeader, region image.Rectangle) error {
	total := region.Dx() * region.Dy()
	if c.workers > 1 && total >= 64 {
		workers, chunkSize, tasks := jxrMacroChunkPlan(total, 32, c.workers)
		if workers > 1 {
			return runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
				start := task * chunkSize
				return c.forwardSecondRange(runCtx, h, region, start, min(start+chunkSize, total))
			})
		}
	}
	return c.forwardSecondRange(ctx, h, region, 0, total)
}

// forwardSecondRange 对矩形内指定宏块区间执行二级低通正变换
// 入参: ctx 当前上下文, h 已校验布局, region 宏块坐标矩形, start 起始位置, end 结束位置
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) forwardSecondRange(ctx context.Context, h *jxrHeader, region image.Rectangle, start, end int) error {
	columns := int(h.columns[len(h.columns)-1])
	width := region.Dx()
	for position := start; position < end; position++ {
		at := (region.Min.Y+position/width)*columns + region.Min.X + position%width
		if err := ctx.Err(); err != nil {
			return err
		}
		low := c.lowMacro(at)
		for index := range c.plane.channels {
			if err := ctx.Err(); err != nil {
				return err
			}
			var values [16]int64
			copy(values[:], low[index*16:(index+1)*16])
			count := 16
			if index != 0 && c.plane.color == 1 {
				count = 4
			} else if index != 0 && c.plane.color == 2 {
				count = 8
			}
			if err := checkJXRWords(values[:count], h.longWord); err != nil {
				return err
			}
			if index != 0 && c.plane.scaled {
				for at := range count {
					values[at] >>= 1
				}
			}
			check := newJXRArithmetic(h.longWord)
			if count == 4 {
				values[1], values[2] = values[2], values[1]
				values[0], values[1], values[2], values[3] = jxrT2x2(values[0], values[1], values[2], values[3], 0, &check)
			} else if count == 8 {
				values[5], values[6] = values[6], values[5]
				values[4], values[6], values[5], values[7] = jxrT2x2(values[4], values[6], values[5], values[7], 0, &check)
				values[1], values[2] = values[2], values[1]
				values[0], values[1], values[2], values[3] = jxrT2x2(values[0], values[1], values[2], values[3], 0, &check)
				values[0], values[4] = jxrForward2(values[0], values[4], &check)
			} else {
				jxrForward4(&values, &check)
			}
			if err := check.error(); err != nil {
				return err
			}
			copy(low[index*16:index*16+count], values[:count])
		}
	}
	return ctx.Err()
}

// loadSamples 将显示方向的原生图像反映射到编码网格，窗口外采用相邻边缘样本填充
// 不隐式进行RGB与子采样YUV间的有损抽样，原生子采样图像使用自己的分量网格
// 少分量描述使用局部临时容量，超过容量仍按实际分量数分配，不改变原生通道或模型能力边界
// 入参: ctx 已规范化取消上下文, raster 原生图像, h 图像头, alpha 是否加载独立透明度, exact 是否要求输入精确恢复
// 返回: error 格式、采样、数值或取消错误
func (c *jxrCoefficients) loadSamples(ctx context.Context, raster *Raster, h *jxrHeader, alpha, exact bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if raster == nil || h == nil || c == nil || len(raster.info.Components) == 0 || h.transform > 7 {
		return fmt.Errorf("%w: JPEG XR source image", ErrInvalid)
	}
	width, height := int(h.width), int(h.height)
	if h.transform&4 != 0 {
		width, height = height, width
	}
	if raster.Bounds() != image.Rect(0, 0, width, height) {
		return fmt.Errorf("%w: JPEG XR source dimensions", ErrInvalid)
	}
	format, err := rasterPixelFormat(raster.info)
	if err != nil {
		return err
	}
	if format.depth != h.depth || !alpha && format.color != h.color || alpha && (!raster.info.Alpha || c.plane.color != 0) {
		return fmt.Errorf("%w: JPEG XR source pixel format", ErrInvalid)
	}
	count := raster.info.Channels
	base := 0
	if alpha {
		count, base = 1, raster.info.Channels
	}
	if !alpha && ((c.plane.color == 1 || c.plane.color == 2) && format.color != c.plane.color || (format.color == 1 || format.color == 2) && format.color != c.plane.color) {
		return fmt.Errorf("%w: explicit JPEG XR source chroma resampling required", ErrUnsupported)
	}
	var componentStorage [4]Component
	components := componentStorage[:]
	if count > len(components) {
		components = make([]Component, count)
	} else {
		components = components[:count]
	}
	source := jxrRasterSource{container: jxrContainer{width: h.width, height: h.height, transform: uint32(h.transform), format: format}, primary: jxrStream{primary: c.plane}}
	for index := range components {
		components[index], err = raster.component(base + index)
		if err != nil {
			return err
		}
		expected := source.componentInfo(base + index)
		actual := components[index].info
		if actual.Bounds != expected.Bounds || actual.XStep != expected.XStep || actual.YStep != expected.YStep {
			return fmt.Errorf("%w: JPEG XR source sampling grid", ErrInvalid)
		}
		if !alpha && index > 0 && (c.plane.color == 1 || c.plane.color == 2) {
			if actual.XPositionKnown != expected.XPositionKnown || actual.YPositionKnown != expected.YPositionKnown || actual.XPositionKnown && actual.XOffset != expected.XOffset || actual.YPositionKnown && actual.YOffset != expected.YOffset {
				return fmt.Errorf("%w: JPEG XR source chroma centering", ErrInvalid)
			}
		}
	}
	var viewStorage [4]jxrComponentView
	views := c.componentViewsStorage(h, true, viewStorage[:])
	if !alpha && (c.plane.color == 1 || c.plane.color == 2) {
		color, _, bias, precision, shift, scale, err := jxrIntegerInputConfig(h, &c.plane, false)
		if err != nil {
			return err
		}
		for index, component := range components {
			view := &views[index]
			sx, sy := 1, 1
			if index > 0 {
				sx = 2
				if c.plane.color == 1 {
					sy = 2
				}
			}
			cw, ch := int(h.width)/sx, int(h.height)/sy
			for y := range view.height {
				if err := ctx.Err(); err != nil {
					return err
				}
				py := min(max(y-int(h.top)/sy, 0), ch-1)
				for x := range view.width {
					if x&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					px := min(max(x-int(h.left)/sx, 0), cw-1)
					dx, dy := jxrRasterPosition(px, py, cw, ch, h.transform)
					sampleBits, err := component.sampleBitsAt(dx, dy)
					if err != nil {
						return err
					}
					value, err := jxrIntegerInputSample(h, color, bias, precision, shift, scale, index, sampleBits, exact)
					if err != nil {
						return err
					}
					view.set(x, y, value)
				}
			}
		}
		return ctx.Err()
	}
	mbRows := views[0].height / 16
	if c.workers > 1 && mbRows >= 8 && views[0].height%16 == 0 && views[0].width*views[0].height >= 16384 {
		workers, chunkRows, tasks := jxrMacroChunkPlan(mbRows, 4, c.workers)
		if workers > 1 {
			sharedComponents := make([]Component, len(components))
			copy(sharedComponents, components)
			sharedViews := make([]jxrComponentView, len(views))
			copy(sharedViews, views)
			return c.loadFullSamplesParallel(ctx, h, sharedComponents, sharedViews, alpha, exact, workers, chunkRows, tasks)
		}
	}
	var bitStorage [4]uint32
	var valueStorage [4]int64
	bits, values := bitStorage[:], valueStorage[:]
	if count > len(bits) {
		bits, values = make([]uint32, count), make([]int64, count)
	} else {
		bits, values = bits[:count], values[:count]
	}
	return c.loadFullSamplesRange(ctx, h, components, views, alpha, exact, 0, views[0].height, bits, values)
}

// loadFullSamplesParallel 将大尺寸全采样图像按宏块行分片并行装载到编码视图
// 入参: ctx 取消上下文, h 图像头, components 源分量视图, views 目标编码视图, alpha 是否加载透明度, exact 是否精确恢复, workers 并发槽位数, chunkRows 单片宏块行数, tasks 分片数
// 返回: error 格式、采样、数值或取消错误
func (c *jxrCoefficients) loadFullSamplesParallel(ctx context.Context, h *jxrHeader, components []Component, views []jxrComponentView, alpha, exact bool, workers, chunkRows, tasks int) error {
	count := len(components)
	return runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
		var bStorage [4]uint32
		var vStorage [4]int64
		rowBits, rowValues := bStorage[:], vStorage[:]
		if count > len(rowBits) {
			rowBits, rowValues = make([]uint32, count), make([]int64, count)
		} else {
			rowBits, rowValues = rowBits[:count], rowValues[:count]
		}
		startY := task * chunkRows * 16
		endY := min((task+1)*chunkRows*16, views[0].height)
		return c.loadFullSamplesRange(runCtx, h, components, views, alpha, exact, startY, endY, rowBits, rowValues)
	})
}

// loadFullSamplesRange 将指定行范围的全采样原始分量反映射并执行正向色彩变换写入视图
// 入参: ctx 取消上下文, h 图像头, components 源分量视图, views 目标编码视图, alpha 是否加载透明度, exact 是否精确恢复, startY 起始行, endY 结束行, rowBits 临时位模式缓冲, rowValues 临时样本缓冲
// 返回: error 格式、采样、数值或取消错误
func (c *jxrCoefficients) loadFullSamplesRange(ctx context.Context, h *jxrHeader, components []Component, views []jxrComponentView, alpha, exact bool, startY, endY int, rowBits []uint32, rowValues []int64) error {
	cw, ch := int(h.width), int(h.height)
	integerMode := h.depth != 4 && h.depth != 7 && (h.color != 8 || alpha)
	var color, precision, shift, scale uint8
	var count int
	var bias int64
	if integerMode {
		var err error
		color, count, bias, precision, shift, scale, err = jxrIntegerInputConfig(h, &c.plane, alpha)
		if err != nil {
			return err
		}
		if len(rowBits) < count || len(rowValues) < count {
			return fmt.Errorf("%w: JPEG XR integer input component count", ErrInvalid)
		}
	}
	for y := startY; y < endY; y++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		py := min(max(y-int(h.top), 0), ch-1)
		for x := range views[0].width {
			if x&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			px := min(max(x-int(h.left), 0), cw-1)
			dx, dy := jxrRasterPosition(px, py, cw, ch, h.transform)
			for index, component := range components {
				v, err := component.sampleBitsAt(dx, dy)
				if err != nil {
					return err
				}
				rowBits[index] = v
			}
			if integerMode {
				for index := range count {
					value, err := jxrIntegerInputSample(h, color, bias, precision, shift, scale, index, rowBits[index], exact)
					if err != nil {
						return err
					}
					rowValues[index] = value
				}
			} else if _, err := jxrInputSamples(h, &c.plane, rowBits, rowValues, alpha, exact); err != nil {
				return err
			}
			if actual, err := jxrForwardColor(h, &c.plane, rowValues, alpha); err != nil || actual != len(views) {
				if err != nil {
					return err
				}
				return fmt.Errorf("%w: JPEG XR loaded component count", ErrInvalid)
			}
			for index := range views {
				views[index].set(x, y, rowValues[index])
			}
		}
	}
	return ctx.Err()
}

// jxrInputSamples 按原生位模式选择整数、浮点或共享指数预处理，不执行颜色变换
// 入参: h 图像头, p 目标平面, bits 原生位模式, values 样本缓冲, alpha 是否为透明度, exact 是否要求输入精确恢复
// 返回: int 输入分量数, error 格式、精度或数值范围错误
func jxrInputSamples(h *jxrHeader, p *jxrPlane, bits []uint32, values []int64, alpha, exact bool) (int, error) {
	if h.depth == 4 || h.depth == 7 {
		return jxrFloatInput(h, p, bits, values, alpha, exact)
	}
	if h.color == 8 && !alpha {
		return jxrRGBEInput(h, p, bits, values, exact)
	}
	return jxrIntegerInput(h, p, bits, values, alpha, exact)
}

// jxrForwardColor 执行规范性输出颜色转换的可逆正向运算，不处理偏置、数值缩放或色度抽样
// RGB先转换到完整采样的YUV，子采样由后续图像处理执行；灰度目标拒绝不可逆的颜色丢弃
// 入参: h 图像头, p 目标平面, values 已去偏置及预缩放的颜色分量，原地替换, alpha 是否为透明度平面
// 返回: int 目标分量数, error 颜色、分量或数值范围错误
func jxrForwardColor(h *jxrHeader, p *jxrPlane, values []int64, alpha bool) (int, error) {
	color := h.color
	if alpha {
		color = 0
	}
	if !jxrPlaneColorAllowed(color, h.depth, p.color) || p.channels < 1 || p.channels > rasterMaxComponents {
		return 0, fmt.Errorf("%w: JPEG XR forward color combination", ErrInvalid)
	}
	count := p.channels
	switch color {
	case 0:
		count = 1
	case 1, 2, 3, 7, 8:
		count = 3
	case 4, 5:
		count = 4
	}
	internal := p.channels
	switch p.color {
	case 0:
		internal = 1
	case 1, 2, 3:
		internal = 3
	case 4:
		internal = 4
	}
	if p.channels != internal || len(values) < count {
		return 0, fmt.Errorf("%w: JPEG XR forward component count", ErrInvalid)
	}
	if err := checkJXRWords(values[:count], true); err != nil {
		return 0, err
	}
	check := newJXRArithmetic(true)
	switch color {
	case 0, 1, 2, 6:
	case 3:
		if p.color == 0 && (values[1] != 0 || values[2] != 0) {
			return 0, fmt.Errorf("%w: nonzero chroma for reversible grayscale input", ErrUnsupported)
		}
	case 4:
		v := check.value(values[0] - values[2])
		u := check.value(values[0] - values[1] - (v >> 1))
		y := check.value(values[3] - values[1] - (u >> 1))
		k := check.value(values[3] - (y >> 1))
		values[0], values[1], values[2], values[3] = y, u, v, k
	case 5:
		values[0], values[1], values[2], values[3] = values[3], values[0], values[1], values[2]
	case 7, 8:
		if p.color == 0 {
			if values[0] != values[1] || values[0] != values[2] {
				return 0, fmt.Errorf("%w: unequal RGB for reversible grayscale input", ErrUnsupported)
			}
			break
		}
		if !h.redBlue && (h.depth == 8 || h.depth == 9 || h.depth == 10) {
			values[0], values[2] = values[2], values[0]
		}
		v := check.value(values[2] - values[0])
		t := check.value(values[0] - values[1] + ((v + 1) >> 1))
		y := check.value(values[1] + (t >> 1))
		values[0], values[1], values[2] = y, check.value(-t), v
	}
	if err := check.error(); err != nil {
		return 0, err
	}
	return internal, nil
}

// forwardWindow 对软重叠邻域仅执行核心及其必要依赖，子采样低通使用自身网格及真实角部规则
// 入参: ctx 当前上下文, h 已校验局部布局, window 核心宏块及左上右下真实边缘
// 返回: error 布局、能力、数值范围或取消错误
func (c *jxrCoefficients) forwardWindow(ctx context.Context, h *jxrHeader, window jxrForwardWindow) error {
	bounds := image.Rect(0, 0, int(h.columns[len(h.columns)-1]), int(h.rows[len(h.rows)-1]))
	if h.hardTiles || h.overlap < 1 || h.overlap > 2 || window.core.Empty() || !window.core.In(bounds) {
		return fmt.Errorf("%w: JPEG XR forward window", ErrInvalid)
	}
	first := window.core
	if h.overlap == 2 {
		first = first.Inset(-1).Intersect(bounds)
	}
	if err := c.filterWindow(ctx, h, true, first, window.edges, true); err != nil {
		return err
	}
	if err := c.forwardFirstRegion(ctx, h, first); err != nil {
		return err
	}
	if h.overlap == 2 {
		if err := c.filterWindow(ctx, h, false, window.core, window.edges, true); err != nil {
			return err
		}
	}
	return c.forwardSecondRegion(ctx, h, window.core)
}

// filterWindow 按分量网格处理所需矩形的滤波片，边缘片仅在真实图像边缘执行
// 子采样低通及普通网格均按各自片尺寸保留预滤波或后滤波依赖
// 入参: ctx 当前上下文, h 局部布局, samples 样本级而非低通级, region 所需宏块矩形, edges 左上右下真实边缘, pre 是否执行预滤波
// 返回: error 邻域不足、数值范围或取消错误
func (c *jxrCoefficients) filterWindow(ctx context.Context, h *jxrHeader, samples bool, region image.Rectangle, edges [4]bool, pre bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	values := c.low
	if samples {
		values = c.high
	}
	if err := checkJXRWordsContext(ctx, values, h.longWord); err != nil {
		return err
	}
	var storage [len(jxrModelComponentWeight[0])]jxrComponentView
	views := c.componentViewsStorage(h, samples, storage[:])
	for index := range views {
		v := &views[index]
		sx, sy := v.blockWidth*v.micro, v.blockHeight*v.micro
		needed := image.Rect(region.Min.X*sx, region.Min.Y*sy, region.Max.X*sx, region.Max.Y*sy)
		chroma := !samples && index != 0 && (c.plane.color == 1 || c.plane.color == 2)
		if h.hardTiles && (len(h.columns) > 2 || len(h.rows) > 2) {
			for row := 0; row+1 < len(h.rows); row++ {
				for column := 0; column+1 < len(h.columns); column++ {
					left, right := int(h.columns[column])*sx, int(h.columns[column+1])*sx
					top, bottom := int(h.rows[row])*sy, int(h.rows[row+1])*sy
					part := needed.Intersect(image.Rect(left, top, right, bottom))
					if part.Empty() {
						continue
					}
					local := *v
					local.width, local.height = right-left, bottom-top
					local.offset += (int(h.rows[row])*v.columns + int(h.columns[column])) * v.stride
					localEdges := [4]bool{column > 0 || edges[0], row > 0 || edges[1], column+2 < len(h.columns) || edges[2], row+2 < len(h.rows) || edges[3]}
					if err := local.filterComponentWindow(ctx, part.Sub(image.Pt(left, top)), localEdges, chroma, pre); err != nil {
						return err
					}
				}
			}
		} else if err := v.filterComponentWindow(ctx, needed, edges, chroma, pre); err != nil {
			return err
		}
	}
	return checkJXRWordsContext(ctx, values, h.longWord)
}

// filterComponentWindow 在同一真实滤波分区内处理所需片，视图可保留原宏块行步幅而裁短水平范围
// 入参: ctx 取消上下文, needed 分量局部所需范围, edges 左上右下真实分区边缘, chroma 是否为子采样低通, pre 是否预滤波
// 返回: error 邻域、数值范围或取消错误
func (v *jxrComponentView) filterComponentWindow(ctx context.Context, needed image.Rectangle, edges [4]bool, chroma, pre bool) error {
	margin := 2
	if chroma {
		margin = 1
	}
	if !edges[0] && needed.Min.X < margin || !edges[1] && needed.Min.Y < margin || !edges[2] && needed.Max.X > v.width-margin || !edges[3] && needed.Max.Y > v.height-margin {
		return fmt.Errorf("%w: JPEG XR forward window halo", ErrInvalid)
	}
	if chroma {
		return v.filterChromaWindow(ctx, needed, edges, pre)
	}
	patch := func(x, y, width, height int) error {
		if image.Rect(x, y, x+width, y+height).Overlaps(needed) {
			return v.filterPatch(ctx, x, y, width, height, pre)
		}
		return ctx.Err()
	}
	for y := 2; y < v.height-2; y += 4 {
		for x := 2; x < v.width-2; x += 4 {
			if err := patch(x, y, 4, 4); err != nil {
				return err
			}
		}
		for edge, x := range [4]int{0, 1, v.width - 2, v.width - 1} {
			if edges[(edge/2)*2] {
				if err := patch(x, y, 1, 4); err != nil {
					return err
				}
			}
		}
	}
	for x := 2; x < v.width-2; x += 4 {
		for edge, y := range [4]int{0, 1, v.height - 2, v.height - 1} {
			if edges[(edge/2)*2+1] {
				if err := patch(x, y, 4, 1); err != nil {
					return err
				}
			}
		}
	}
	for row, y := range [2]int{0, v.height - 2} {
		for column, x := range [2]int{0, v.width - 2} {
			if edges[column*2] && edges[row*2+1] {
				if err := patch(x, y, 2, 2); err != nil {
					return err
				}
			}
		}
	}
	return ctx.Err()
}

// filterChromaWindow 执行所需色度低通预滤波或后滤波片及真实角部残差，人工邻域边缘不参与边缘运算
// 入参: ctx 当前上下文, needed 当前分量的所需低通矩形, edges 左上右下真实边缘, pre 是否执行预滤波
// 返回: error 邻域、数值或取消错误
func (v *jxrComponentView) filterChromaWindow(ctx context.Context, needed image.Rectangle, edges [4]bool, pre bool) error {
	if v.width < 4 || needed.Empty() || !needed.In(image.Rect(0, 0, v.width, v.height)) {
		return fmt.Errorf("%w: JPEG XR chroma filter window", ErrInvalid)
	}
	if err := v.chromaWindowCorners(ctx, needed, edges, -1); err != nil {
		return err
	}
	patch := func(x, y, width, height int) error {
		if image.Rect(x, y, x+width, y+height).Overlaps(needed) {
			return v.filterChromaPatch(ctx, x, y, width, height, pre)
		}
		return ctx.Err()
	}
	for y := 1; y < v.height-1; y += 2 {
		for x := 1; x < v.width-1; x += 2 {
			if err := patch(x, y, 2, 2); err != nil {
				return err
			}
		}
		for edge, x := range [2]int{0, v.width - 1} {
			if edges[edge*2] {
				if err := patch(x, y, 1, 2); err != nil {
					return err
				}
			}
		}
	}
	for x := 1; x < v.width-1; x += 2 {
		for edge, y := range [2]int{0, v.height - 1} {
			if edges[edge*2+1] {
				if err := patch(x, y, 2, 1); err != nil {
					return err
				}
			}
		}
	}
	return v.chromaWindowCorners(ctx, needed, edges, 1)
}

// chromaWindowCorners 仅在所需真实角部转换或恢复相邻系数残差，保持标准有符号字长保护
// 入参: ctx 当前上下文, needed 所需低通矩形, edges 真实边缘, direction 减去相邻值为负一，恢复为一
// 返回: error 数值范围或取消错误
func (v *jxrComponentView) chromaWindowCorners(ctx context.Context, needed image.Rectangle, edges [4]bool, direction int64) error {
	for row, y := range [2]int{0, v.height - 1} {
		for column, x := range [2]int{0, v.width - 1} {
			if err := ctx.Err(); err != nil {
				return err
			}
			if !edges[column*2] || !edges[row*2+1] || !image.Pt(x, y).In(needed) {
				continue
			}
			neighbor := x + 1
			if column != 0 {
				neighbor = x - 1
			}
			value := v.at(x, y) + direction*v.at(neighbor, y)
			if err := checkJXRWords([]int64{value}, v.wordBits == 32); err != nil {
				return err
			}
			if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
				return jxrCoefficientRangeError("JPEG XR chroma corner coefficient range", value)
			}
			v.set(x, y, value)
		}
	}
	return nil
}
