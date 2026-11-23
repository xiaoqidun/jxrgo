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

// jxrOverlap4x4 执行四乘四重叠后滤波，输入及输出均采用行优先位置
// 入参: v 有界的十六个输入值，原地替换为滤波结果, check 局部变量校验状态
func jxrOverlap4x4(v *[16]int64, check *jxrArithmetic) {
	if check == nil || jxrSafeBlock16(v, check.longWord) {
		jxrOverlap4x4Fast(v)
		return
	}
	v[0], v[3], v[12], v[15] = jxrT2x2(v[0], v[3], v[12], v[15], 0, check)
	v[1], v[2], v[13], v[14] = jxrT2x2(v[1], v[2], v[13], v[14], 0, check)
	v[4], v[7], v[8], v[11] = jxrT2x2(v[4], v[7], v[8], v[11], 0, check)
	v[5], v[6], v[9], v[10] = jxrT2x2(v[5], v[6], v[9], v[10], 0, check)
	v[13], v[12] = jxrInverseRotate(v[13], v[12], check)
	v[9], v[8] = jxrInverseRotate(v[9], v[8], check)
	v[7], v[3] = jxrInverseRotate(v[7], v[3], check)
	v[6], v[2] = jxrInverseRotate(v[6], v[2], check)
	v[10], v[11], v[14], v[15] = jxrInverseOddOdd(v[10], v[11], v[14], v[15], true, check)
	v[0], v[15] = jxrInverseScale(v[0], v[15], check)
	v[1], v[14] = jxrInverseScale(v[1], v[14], check)
	v[4], v[11] = jxrInverseScale(v[4], v[11], check)
	v[5], v[10] = jxrInverseScale(v[5], v[10], check)
	v[0], v[3], v[12], v[15] = jxrT2x2Post(v[0], v[3], v[12], v[15], check)
	v[1], v[2], v[13], v[14] = jxrT2x2Post(v[1], v[2], v[13], v[14], check)
	v[4], v[7], v[8], v[11] = jxrT2x2Post(v[4], v[7], v[8], v[11], check)
	v[5], v[6], v[9], v[10] = jxrT2x2Post(v[5], v[6], v[9], v[10], check)
}

// jxrOverlap4 执行边缘或角部使用的四点重叠后滤波
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个滤波值
func jxrOverlap4(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	if check == nil || jxrSafeValues4(a, b, c, d, check.longWord) {
		return jxrOverlap4Fast(a, b, c, d)
	}
	a = check.value(a + d)
	b = check.value(b + c)
	d = check.value(d - ((a + 1) >> 1))
	c = check.value(c - ((b + 1) >> 1))
	a, d = jxrInverseScale(a, d, check)
	b, c = jxrInverseScale(b, c, check)
	a = check.value(a + ((d*3 + 4) >> 3))
	b = check.value(b + ((c*3 + 4) >> 3))
	d = check.value(d - (a >> 1))
	c = check.value(c - (b >> 1))
	a = check.value(a + d)
	b = check.value(b + c)
	d, c = check.value(-d), check.value(-c)
	c, d = jxrInverseRotate(c, d, check)
	d = check.value(d + ((a + 1) >> 1))
	c = check.value(c + ((b + 1) >> 1))
	a = check.value(a - d)
	b = check.value(b - c)
	return a, b, c, d
}

// jxrOverlap2x2 执行抽样色度分量的二乘二重叠后滤波
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个滤波值
func jxrOverlap2x2(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	if check == nil || jxrSafeValues4(a, b, c, d, check.longWord) {
		return jxrOverlap2x2Fast(a, b, c, d)
	}
	a = check.value(a + d)
	b = check.value(b + c)
	d = check.value(d - ((a + 1) >> 1))
	c = check.value(c - ((b + 1) >> 1))
	a, b = jxrOverlap2(a, b, check)
	d = check.value(d + ((a + 1) >> 1))
	c = check.value(c + ((b + 1) >> 1))
	a = check.value(a - d)
	b = check.value(b - c)
	return a, b, c, d
}

// jxrOverlap2 执行抽样色度边缘使用的两点重叠后滤波
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个滤波值
func jxrOverlap2(a, b int64, check *jxrArithmetic) (int64, int64) {
	if check == nil || jxrSafeValues2(a, b, check.longWord) {
		return jxrOverlap2Fast(a, b)
	}
	b = check.value(b + ((a + 2) >> 2))
	a = check.value(a + ((b + 1) >> 1))
	a = check.value(a + (b >> 5))
	a = check.value(a + (b >> 9))
	a = check.value(a + (b >> 13))
	b = check.value(b + ((a + 2) >> 2))
	return a, b
}

// jxrInverseRotate 执行重叠后滤波中的两点逆旋转
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrInverseRotate(a, b int64, check *jxrArithmetic) (int64, int64) {
	a = check.value(a - ((b + 1) >> 1))
	b = check.value(b + ((a + 1) >> 1))
	return a, b
}

// jxrInverseScale 执行重叠后滤波中的两点逆缩放
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrInverseScale(a, b int64, check *jxrArithmetic) (int64, int64) {
	a = check.value(a + b)
	b = check.value((a >> 1) - b)
	a = check.value(a + ((b * 3) >> 3))
	b = check.value(b + ((a * 3) >> 4))
	b = check.value(b + (a >> 7))
	b = check.value(b - (a >> 10))
	return a, b
}

// jxrT2x2Post 执行重叠后滤波中的四点整数变换
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrT2x2Post(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	b = check.value(b - c)
	a = check.value(a + ((d*3 + 4) >> 3))
	d = check.value(d - (b >> 1))
	c = check.value(((a - b) >> 1) - c)
	c, d = d, c
	a = check.value(a - d)
	b = check.value(b + c)
	return a, b, c, d
}

// jxrOverlap4x4Fast 执行无校验的四乘四重叠后滤波快路径
// 入参: v 处于安全范围的十六个输入值，原地替换为滤波结果
func jxrOverlap4x4Fast(v *[16]int64) {
	v0, v1, v2, v3 := v[0], v[1], v[2], v[3]
	v4, v5, v6, v7 := v[4], v[5], v[6], v[7]
	v8, v9, v10, v11 := v[8], v[9], v[10], v[11]
	v12, v13, v14, v15 := v[12], v[13], v[14], v[15]
	v0 += v15
	v3 -= v12
	t := (v0 - v3) >> 1
	oldC := v12
	v12, v15 = t-v15, t-oldC
	v0 -= v15
	v3 += v12
	v1 += v14
	v2 -= v13
	t = (v1 - v2) >> 1
	oldC = v13
	v13, v14 = t-v14, t-oldC
	v1 -= v14
	v2 += v13
	v4 += v11
	v7 -= v8
	t = (v4 - v7) >> 1
	oldC = v8
	v8, v11 = t-v11, t-oldC
	v4 -= v11
	v7 += v8
	v5 += v10
	v6 -= v9
	t = (v5 - v6) >> 1
	oldC = v9
	v9, v10 = t-v10, t-oldC
	v5 -= v10
	v6 += v9
	v13 -= (v12 + 1) >> 1
	v12 += (v13 + 1) >> 1
	v9 -= (v8 + 1) >> 1
	v8 += (v9 + 1) >> 1
	v7 -= (v3 + 1) >> 1
	v3 += (v7 + 1) >> 1
	v6 -= (v2 + 1) >> 1
	v2 += (v6 + 1) >> 1
	v15 += v10
	v14 -= v11
	t1, t2 := v15>>1, v14>>1
	v10 -= t1
	v11 += t2
	v10 -= (v11*3 + 6) >> 3
	v11 += (v10*3 + 2) >> 2
	v10 -= (v11*3 + 4) >> 3
	v11 -= t2
	v10 += t1
	v14 += v11
	v15 -= v10
	v0 += v15
	v15 = (v0 >> 1) - v15
	v0 += (v15 * 3) >> 3
	v15 += (v0 * 3) >> 4
	v15 += v0 >> 7
	v15 -= v0 >> 10
	v1 += v14
	v14 = (v1 >> 1) - v14
	v1 += (v14 * 3) >> 3
	v14 += (v1 * 3) >> 4
	v14 += v1 >> 7
	v14 -= v1 >> 10
	v4 += v11
	v11 = (v4 >> 1) - v11
	v4 += (v11 * 3) >> 3
	v11 += (v4 * 3) >> 4
	v11 += v4 >> 7
	v11 -= v4 >> 10
	v5 += v10
	v10 = (v5 >> 1) - v10
	v5 += (v10 * 3) >> 3
	v10 += (v5 * 3) >> 4
	v10 += v5 >> 7
	v10 -= v5 >> 10
	v3 -= v12
	v0 += (v15*3 + 4) >> 3
	v15 -= v3 >> 1
	v12 = ((v0 - v3) >> 1) - v12
	v12, v15 = v15, v12
	v0 -= v15
	v3 += v12
	v2 -= v13
	v1 += (v14*3 + 4) >> 3
	v14 -= v2 >> 1
	v13 = ((v1 - v2) >> 1) - v13
	v13, v14 = v14, v13
	v1 -= v14
	v2 += v13
	v7 -= v8
	v4 += (v11*3 + 4) >> 3
	v11 -= v7 >> 1
	v8 = ((v4 - v7) >> 1) - v8
	v8, v11 = v11, v8
	v4 -= v11
	v7 += v8
	v6 -= v9
	v5 += (v10*3 + 4) >> 3
	v10 -= v6 >> 1
	v9 = ((v5 - v6) >> 1) - v9
	v9, v10 = v10, v9
	v5 -= v10
	v6 += v9
	v[0], v[1], v[2], v[3] = v0, v1, v2, v3
	v[4], v[5], v[6], v[7] = v4, v5, v6, v7
	v[8], v[9], v[10], v[11] = v8, v9, v10, v11
	v[12], v[13], v[14], v[15] = v12, v13, v14, v15
}

// jxrOverlap4Fast 执行无校验的四点重叠后滤波快路径
// 入参: a, b, c, d 处于安全范围的输入值
// 返回: int64 四个滤波值
func jxrOverlap4Fast(a, b, c, d int64) (int64, int64, int64, int64) {
	a += d
	b += c
	d -= (a + 1) >> 1
	c -= (b + 1) >> 1
	a += d
	d = (a >> 1) - d
	a += (d * 3) >> 3
	d += (a * 3) >> 4
	d += a >> 7
	d -= a >> 10
	b += c
	c = (b >> 1) - c
	b += (c * 3) >> 3
	c += (b * 3) >> 4
	c += b >> 7
	c -= b >> 10
	a += (d*3 + 4) >> 3
	b += (c*3 + 4) >> 3
	d -= a >> 1
	c -= b >> 1
	a += d
	b += c
	d, c = -d, -c
	c -= (d + 1) >> 1
	d += (c + 1) >> 1
	d += (a + 1) >> 1
	c += (b + 1) >> 1
	a -= d
	b -= c
	return a, b, c, d
}

// jxrOverlap2x2Fast 执行无校验的二乘二重叠后滤波快路径
// 入参: a, b, c, d 处于安全范围的输入值
// 返回: int64 四个滤波值
func jxrOverlap2x2Fast(a, b, c, d int64) (int64, int64, int64, int64) {
	a += d
	b += c
	d -= (a + 1) >> 1
	c -= (b + 1) >> 1
	b += (a + 2) >> 2
	a += (b + 1) >> 1
	a += b >> 5
	a += b >> 9
	a += b >> 13
	b += (a + 2) >> 2
	d += (a + 1) >> 1
	c += (b + 1) >> 1
	a -= d
	b -= c
	return a, b, c, d
}

// jxrOverlap2Fast 执行无校验的两点重叠后滤波快路径
// 入参: a, b 处于安全范围的输入值
// 返回: int64 两个滤波值
func jxrOverlap2Fast(a, b int64) (int64, int64) {
	b += (a + 2) >> 2
	a += (b + 1) >> 1
	a += b >> 5
	a += b >> 9
	a += b >> 13
	b += (a + 2) >> 2
	return a, b
}

// jxrCoefficientRangeError 构造带准确幅值上限与实测值的系数超限错误
// 入参: resource 超限资源描述, value 触发超限的有符号系数值
// 返回: error 包含上限与实测绝对值的资源错误
func jxrCoefficientRangeError(resource string, value int64) error {
	required := uint64(value)
	if value < 0 {
		required = uint64(-(value + 1)) + 1
	}
	return &LimitError{Resource: resource, Limit: uint64(jxrCoefficientLimit), Required: required}
}

// filterChromaRegion 在两乘二网格上预滤波或后滤波，422的纵向半宏块边界同样参与滤波
// 入参: ctx 取消上下文, left, top, right, bottom 区域边界, pre 是否执行预滤波
// 返回: error 错误信息
func (v *jxrComponentView) filterChromaRegion(ctx context.Context, left, top, right, bottom int, pre bool) error {
	if right-left < 4 {
		return fmt.Errorf("%w: JPEG XR chroma filter region width", ErrInvalid)
	}
	if err := v.chromaCornerResidual(left, top, right, bottom, -1); err != nil {
		return err
	}
	for y := top + 1; y < bottom-1; y += 2 {
		if err := ctx.Err(); err != nil {
			return err
		}
		for x := left + 1; x < right-1; x += 2 {
			if err := v.filterChromaPatch2x2(x, y, pre); err != nil {
				return err
			}
		}
		for _, x := range [2]int{left, right - 1} {
			if err := v.filterChromaPatch2(x, y, 1, 2, pre); err != nil {
				return err
			}
		}
	}
	for x := left + 1; x < right-1; x += 2 {
		if (x-left-1)&127 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for _, y := range [2]int{top, bottom - 1} {
			if err := v.filterChromaPatch2(x, y, 2, 1, pre); err != nil {
				return err
			}
		}
	}
	return v.chromaCornerResidual(left, top, right, bottom, 1)
}

// chromaCornerResidual 在角部系数和水平方向相邻系数间转换残差或恢复系数
// 入参: left, top, right, bottom 区域边界, direction 减去相邻系数时为负一，恢复时为一
// 返回: error 计算容量错误
func (v *jxrComponentView) chromaCornerResidual(left, top, right, bottom int, direction int64) error {
	data := v.data
	for _, y := range [2]int{top, bottom - 1} {
		for _, pair := range [2][2]int{{left, left + 1}, {right - 1, right - 2}} {
			at := v.index(pair[0], y)
			value := data[at] + direction*v.at(pair[1], y)
			if v.wordBits != 0 {
				low, high := jxrWordBounds(v.wordBits == 32)
				if value < low || value > high {
					return checkJXRWords([]int64{value}, v.wordBits == 32)
				}
			}
			if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
				return jxrCoefficientRangeError("JPEG XR chroma corner coefficient range", value)
			}
			data[at] = value
		}
	}
	return nil
}

// filterChromaPatch 执行子采样色度内部的四点滤波或边缘的两点滤波
// 入参: ctx 取消上下文, x, y 区域起点, width, height 区域尺寸, pre 是否执行预滤波
// 返回: error 计算容量错误
func (v *jxrComponentView) filterChromaPatch(ctx context.Context, x, y, width, height int, pre bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if width*height == 4 {
		return v.filterChromaPatch2x2(x, y, pre)
	}
	return v.filterChromaPatch2(x, y, width, height, pre)
}

// filterChromaPatch2x2 执行子采样色度内部二乘二区域滤波，复用坐标索引并优先走无校验快路径
// 入参: x, y 区域起点, pre 是否执行预滤波
// 返回: error 数值范围或容量错误
func (v *jxrComponentView) filterChromaPatch2x2(x, y int, pre bool) error {
	data := v.data
	i0, i1, i2, i3 := v.index(x, y), v.index(x+1, y), v.index(x, y+1), v.index(x+1, y+1)
	a, b, c, d := data[i0], data[i1], data[i2], data[i3]
	safeLong := v.wordBits == 0 || v.wordBits == 32
	if jxrSafeValues4(a, b, c, d, safeLong) {
		if pre {
			a, b, c, d = jxrPrefilter2x2Fast(a, b, c, d)
		} else {
			a, b, c, d = jxrOverlap2x2Fast(a, b, c, d)
		}
		data[i0], data[i1], data[i2], data[i3] = a, b, c, d
		return nil
	}
	var arithmetic jxrArithmetic
	var check *jxrArithmetic
	if v.wordBits != 0 {
		arithmetic = newJXRArithmetic(v.wordBits == 32)
		check = &arithmetic
	}
	if pre {
		a, b, c, d = jxrPrefilter2x2(a, b, c, d, check)
	} else {
		a, b, c, d = jxrOverlap2x2(a, b, c, d, check)
	}
	if err := check.error(); err != nil {
		return err
	}
	for _, value := range [4]int64{a, b, c, d} {
		if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
			return jxrCoefficientRangeError("JPEG XR chroma 2x2 overlap coefficient range", value)
		}
	}
	data[i0], data[i1], data[i2], data[i3] = a, b, c, d
	return nil
}

// filterChromaPatch2 执行子采样色度边缘两点区域滤波，复用坐标索引并优先走无校验快路径
// 入参: x, y 区域起点, width, height 区域尺寸, pre 是否执行预滤波
// 返回: error 数值范围或容量错误
func (v *jxrComponentView) filterChromaPatch2(x, y, width, height int, pre bool) error {
	data := v.data
	i0, i1 := v.index(x, y), v.index(x+width-1, y+height-1)
	a, b := data[i0], data[i1]
	safeLong := v.wordBits == 0 || v.wordBits == 32
	if jxrSafeValues2(a, b, safeLong) {
		if pre {
			a, b = jxrPrefilter2Fast(a, b)
		} else {
			a, b = jxrOverlap2Fast(a, b)
		}
		data[i0], data[i1] = a, b
		return nil
	}
	var arithmetic jxrArithmetic
	var check *jxrArithmetic
	if v.wordBits != 0 {
		arithmetic = newJXRArithmetic(v.wordBits == 32)
		check = &arithmetic
	}
	if pre {
		a, b = jxrPrefilter2(a, b, check)
	} else {
		a, b = jxrOverlap2(a, b, check)
	}
	if err := check.error(); err != nil {
		return err
	}
	for _, value := range [2]int64{a, b} {
		if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
			return jxrCoefficientRangeError("JPEG XR chroma 2-point overlap coefficient range", value)
		}
	}
	data[i0], data[i1] = a, b
	return nil
}

// filterGrid 按分量及硬分块边界应用预滤波或后滤波，区域处理期间检查取消状态
// 入参: ctx 取消上下文, h 图像头, samples 是否为完整样本网格, pre 是否执行预滤波
// 返回: error 数值范围、资源或取消错误
func (c *jxrCoefficients) filterGrid(ctx context.Context, h *jxrHeader, samples, pre bool) error {
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
	views := c.componentViews(h, samples)
	if c.workers > 1 && len(values) >= 16384 {
		if h.hardTiles {
			tileCols, tileRows := len(h.columns)-1, len(h.rows)-1
			tiles := tileCols * tileRows
			total := len(views) * tiles
			workers := resolveJXRWorkers(c.workers, total)
			if workers > 1 {
				if err := runJXRWorkers(ctx, workers, total, func(runCtx context.Context, task int) error {
					comp := task / tiles
					tile := task % tiles
					tx, ty := tile%tileCols, tile/tileCols
					v := &views[comp]
					chroma := !samples && comp != 0 && (c.plane.color == 1 || c.plane.color == 2)
					width, height := v.blockWidth*v.micro, v.blockHeight*v.micro
					return v.filterComponentRegion(runCtx, int(h.columns[tx])*width, int(h.rows[ty])*height, int(h.columns[tx+1])*width, int(h.rows[ty+1])*height, chroma, pre)
				}); err != nil {
					return err
				}
				return checkJXRWordsContext(ctx, values, h.longWord)
			}
		} else if len(views) > 1 {
			workers := resolveJXRWorkers(c.workers, len(views))
			if workers > 1 {
				if err := runJXRWorkers(ctx, workers, len(views), func(runCtx context.Context, i int) error {
					v := &views[i]
					chroma := !samples && i != 0 && (c.plane.color == 1 || c.plane.color == 2)
					return v.filterComponentRegion(runCtx, 0, 0, v.width, v.height, chroma, pre)
				}); err != nil {
					return err
				}
				return checkJXRWordsContext(ctx, values, h.longWord)
			}
		}
	}
	for i := range views {
		if err := ctx.Err(); err != nil {
			return err
		}
		v := &views[i]
		chroma := !samples && i != 0 && (c.plane.color == 1 || c.plane.color == 2)
		if !h.hardTiles {
			if err := v.filterComponentRegion(ctx, 0, 0, v.width, v.height, chroma, pre); err != nil {
				return err
			}
			continue
		}
		width, height := v.blockWidth*v.micro, v.blockHeight*v.micro
		for y := 0; y+1 < len(h.rows); y++ {
			for x := 0; x+1 < len(h.columns); x++ {
				if err := v.filterComponentRegion(ctx, int(h.columns[x])*width, int(h.rows[y])*height, int(h.columns[x+1])*width, int(h.rows[y+1])*height, chroma, pre); err != nil {
					return err
				}
			}
		}
	}
	return checkJXRWordsContext(ctx, values, h.longWord)
}

// filterComponentRegion 按当前级别的分量网格选择普通或子采样色度滤波
// 入参: ctx 取消上下文, left, top, right, bottom 区域边界, chroma 是否为子采样色度的低通网格, pre 是否执行预滤波
// 返回: error 错误信息
func (v *jxrComponentView) filterComponentRegion(ctx context.Context, left, top, right, bottom int, chroma, pre bool) error {
	if chroma {
		return v.filterChromaRegion(ctx, left, top, right, bottom, pre)
	}
	return v.filterRegion(ctx, left, top, right, bottom, pre)
}

// filterRegion 在四乘四块网格上预滤波或后滤波，软分块不形成额外边缘
// 入参: ctx 取消上下文, left, top, right, bottom 滤波区域边界, pre 是否执行预滤波
// 返回: error 错误信息
func (v *jxrComponentView) filterRegion(ctx context.Context, left, top, right, bottom int, pre bool) error {
	for y := top + 2; y < bottom-2; y += 4 {
		if err := ctx.Err(); err != nil {
			return err
		}
		for x := left + 2; x < right-2; x += 4 {
			if err := v.filterPatch4x4(x, y, pre); err != nil {
				return err
			}
		}
		for _, x := range [4]int{left, left + 1, right - 2, right - 1} {
			if err := v.filterPatch4(x, y, 1, 4, pre); err != nil {
				return err
			}
		}
	}
	for x := left + 2; x < right-2; x += 4 {
		if (x-left-2)&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for _, y := range [4]int{top, top + 1, bottom - 2, bottom - 1} {
			if err := v.filterPatch4(x, y, 4, 1, pre); err != nil {
				return err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, y := range [2]int{top, bottom - 2} {
		for _, x := range [2]int{left, right - 2} {
			if err := v.filterPatch4(x, y, 2, 2, pre); err != nil {
				return err
			}
		}
	}
	return nil
}

// filterPatch 预滤波或后滤波一个内部、边缘或角部区域，先校验结果再写回
// 入参: ctx 取消上下文, x, y 区域起点, width, height 区域尺寸, pre 是否执行预滤波
// 返回: error 错误信息
func (v *jxrComponentView) filterPatch(ctx context.Context, x, y, width, height int, pre bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if width == 4 && height == 4 {
		return v.filterPatch4x4(x, y, pre)
	}
	return v.filterPatch4(x, y, width, height, pre)
}

// filterPatch4x4 预滤波或后滤波一个内部四乘四块，按四个二乘二微块基址批量读写
// 入参: x, y 区域起点, pre 是否执行预滤波
// 返回: error 错误信息
func (v *jxrComponentView) filterPatch4x4(x, y int, pre bool) error {
	data := v.data
	if (v.micro == 4 || (v.micro == 1 && v.blockWidth == 4 && v.blockHeight == 4)) && x >= 0 && y >= 0 && x&3 == 2 && y&3 == 2 {
		i00 := v.index(x, y)
		i01 := v.index(x+2, y)
		i10 := v.index(x, y+2)
		i11 := v.index(x+2, y+2)
		s00 := data[i00 : i00+6]
		s01 := data[i01 : i01+6]
		s10 := data[i10 : i10+6]
		s11 := data[i11 : i11+6]
		values := [16]int64{
			s00[0], s00[1], s01[0], s01[1],
			s00[4], s00[5], s01[4], s01[5],
			s10[0], s10[1], s11[0], s11[1],
			s10[4], s10[5], s11[4], s11[5],
		}
		if err := v.applyPatch4x4(&values, pre); err != nil {
			return err
		}
		s00[0], s00[1], s00[4], s00[5] = values[0], values[1], values[4], values[5]
		s01[0], s01[1], s01[4], s01[5] = values[2], values[3], values[6], values[7]
		s10[0], s10[1], s10[4], s10[5] = values[8], values[9], values[12], values[13]
		s11[0], s11[1], s11[4], s11[5] = values[10], values[11], values[14], values[15]
		return nil
	}
	var idx [16]int
	var values [16]int64
	for j := range 4 {
		for i := range 4 {
			at := v.index(x+i, y+j)
			idx[j*4+i] = at
			values[j*4+i] = data[at]
		}
	}
	if err := v.applyPatch4x4(&values, pre); err != nil {
		return err
	}
	for k := range 16 {
		data[idx[k]] = values[k]
	}
	return nil
}

// applyPatch4x4 执行四乘四块滤波运算，安全上界内走无校验快路径，越界回退精确校验
// 入参: values 原地替换的十六个系数, pre 是否执行预滤波
// 返回: error 数值范围或容量错误
func (v *jxrComponentView) applyPatch4x4(values *[16]int64, pre bool) error {
	safeLong := v.wordBits == 0 || v.wordBits == 32
	if jxrSafeBlock16(values, safeLong) {
		if pre {
			jxrPrefilter4x4Fast(values)
		} else {
			jxrOverlap4x4Fast(values)
		}
		return nil
	}
	var arithmetic jxrArithmetic
	var check *jxrArithmetic
	if v.wordBits != 0 {
		arithmetic = newJXRArithmetic(v.wordBits == 32)
		check = &arithmetic
	}
	if pre {
		jxrPrefilter4x4(values, check)
	} else {
		jxrOverlap4x4(values, check)
	}
	if err := check.error(); err != nil {
		return err
	}
	for _, value := range values {
		if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
			return jxrCoefficientRangeError("JPEG XR 4x4 overlap coefficient range", value)
		}
	}
	return nil
}

// filterPatch4 预滤波或后滤波一个边缘或角部四点区域，复用坐标索引并优先走无校验快路径
// 入参: x, y 区域起点, width, height 区域尺寸, pre 是否执行预滤波
// 返回: error 数值范围或容量错误
func (v *jxrComponentView) filterPatch4(x, y, width, height int, pre bool) error {
	var i0, i1, i2, i3 int
	aligned := (v.micro == 4 || (v.micro == 1 && v.blockWidth == 4 && v.blockHeight == 4)) && x >= 0 && y >= 0
	if width == 1 && height == 4 {
		if aligned && y&3 == 2 {
			i0 = v.index(x, y)
			i1 = i0 + 4
			i2 = v.index(x, y+2)
			i3 = i2 + 4
		} else {
			i0, i1, i2, i3 = v.index(x, y), v.index(x, y+1), v.index(x, y+2), v.index(x, y+3)
		}
	} else if width == 4 && height == 1 {
		if aligned && x&3 == 2 {
			i0 = v.index(x, y)
			i1 = i0 + 1
			i2 = v.index(x+2, y)
			i3 = i2 + 1
		} else {
			i0, i1, i2, i3 = v.index(x, y), v.index(x+1, y), v.index(x+2, y), v.index(x+3, y)
		}
	} else {
		if aligned && x&1 == 0 && y&1 == 0 {
			i0 = v.index(x, y)
			i1 = i0 + 1
			i2 = i0 + 4
			i3 = i0 + 5
		} else {
			i0, i1, i2, i3 = v.index(x, y), v.index(x+1, y), v.index(x, y+1), v.index(x+1, y+1)
		}
	}
	data := v.data
	a, b, c, d := data[i0], data[i1], data[i2], data[i3]
	safeLong := v.wordBits == 0 || v.wordBits == 32
	if jxrSafeValues4(a, b, c, d, safeLong) {
		if pre {
			a, b, c, d = jxrPrefilter4Fast(a, b, c, d)
		} else {
			a, b, c, d = jxrOverlap4Fast(a, b, c, d)
		}
		data[i0], data[i1], data[i2], data[i3] = a, b, c, d
		return nil
	}
	var arithmetic jxrArithmetic
	var check *jxrArithmetic
	if v.wordBits != 0 {
		arithmetic = newJXRArithmetic(v.wordBits == 32)
		check = &arithmetic
	}
	if pre {
		a, b, c, d = jxrPrefilter4(a, b, c, d, check)
	} else {
		a, b, c, d = jxrOverlap4(a, b, c, d, check)
	}
	if err := check.error(); err != nil {
		return err
	}
	for _, value := range [4]int64{a, b, c, d} {
		if value < -jxrCoefficientLimit || value > jxrCoefficientLimit {
			return jxrCoefficientRangeError("JPEG XR 4-point overlap coefficient range", value)
		}
	}
	data[i0], data[i1], data[i2], data[i3] = a, b, c, d
	return nil
}

// jxrPrefilter4x4 执行规范性重叠后滤波的可逆预滤波，输入及输出均为行优先排列
// 入参: v 有界的十六个输入值，原地替换为预滤波结果, check 局部变量校验状态
func jxrPrefilter4x4(v *[16]int64, check *jxrArithmetic) {
	if check == nil || jxrSafeBlock16(v, check.longWord) {
		jxrPrefilter4x4Fast(v)
		return
	}
	v[0], v[3], v[12], v[15] = jxrT2x2Pre(v[0], v[3], v[12], v[15], check)
	v[1], v[2], v[13], v[14] = jxrT2x2Pre(v[1], v[2], v[13], v[14], check)
	v[4], v[7], v[8], v[11] = jxrT2x2Pre(v[4], v[7], v[8], v[11], check)
	v[5], v[6], v[9], v[10] = jxrT2x2Pre(v[5], v[6], v[9], v[10], check)
	v[0], v[15] = jxrForwardScale(v[0], v[15], check)
	v[1], v[14] = jxrForwardScale(v[1], v[14], check)
	v[4], v[11] = jxrForwardScale(v[4], v[11], check)
	v[5], v[10] = jxrForwardScale(v[5], v[10], check)
	v[13], v[12] = jxrForwardRotate(v[13], v[12], check)
	v[9], v[8] = jxrForwardRotate(v[9], v[8], check)
	v[7], v[3] = jxrForwardRotate(v[7], v[3], check)
	v[6], v[2] = jxrForwardRotate(v[6], v[2], check)
	v[10], v[11], v[14], v[15] = jxrForwardOddOdd(v[10], v[11], v[14], v[15], true, check)
	v[0], v[12], v[3], v[15] = jxrT2x2(v[0], v[12], v[3], v[15], 0, check)
	v[1], v[2], v[13], v[14] = jxrT2x2(v[1], v[2], v[13], v[14], 0, check)
	v[4], v[7], v[8], v[11] = jxrT2x2(v[4], v[7], v[8], v[11], 0, check)
	v[5], v[6], v[9], v[10] = jxrT2x2(v[5], v[6], v[9], v[10], 0, check)
}

// jxrPrefilter4 执行边缘或角部使用的四点重叠预滤波
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个预滤波值
func jxrPrefilter4(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	if check == nil || jxrSafeValues4(a, b, c, d, check.longWord) {
		return jxrPrefilter4Fast(a, b, c, d)
	}
	a = check.value(a + d)
	b = check.value(b + c)
	d = check.value(d - ((a + 1) >> 1))
	c = check.value(c - ((b + 1) >> 1))
	c, d = jxrForwardRotate(c, d, check)
	d, c = check.value(-d), check.value(-c)
	a = check.value(a - d)
	b = check.value(b - c)
	d = check.value(d + (a >> 1))
	c = check.value(c + (b >> 1))
	a = check.value(a - ((d*3 + 4) >> 3))
	b = check.value(b - ((c*3 + 4) >> 3))
	a, d = jxrForwardScale(a, d, check)
	b, c = jxrForwardScale(b, c, check)
	d = check.value(d + ((a + 1) >> 1))
	c = check.value(c + ((b + 1) >> 1))
	a = check.value(a - d)
	b = check.value(b - c)
	return a, b, c, d
}

// jxrPrefilter2x2 执行子采样色度分量的二乘二重叠预滤波
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个预滤波值
func jxrPrefilter2x2(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	if check == nil || jxrSafeValues4(a, b, c, d, check.longWord) {
		return jxrPrefilter2x2Fast(a, b, c, d)
	}
	a = check.value(a + d)
	b = check.value(b + c)
	d = check.value(d - ((a + 1) >> 1))
	c = check.value(c - ((b + 1) >> 1))
	a, b = jxrPrefilter2(a, b, check)
	d = check.value(d + ((a + 1) >> 1))
	c = check.value(c + ((b + 1) >> 1))
	a = check.value(a - d)
	b = check.value(b - c)
	return a, b, c, d
}

// jxrPrefilter2 执行子采样色度边缘的两点重叠预滤波
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个预滤波值
func jxrPrefilter2(a, b int64, check *jxrArithmetic) (int64, int64) {
	if check == nil || jxrSafeValues2(a, b, check.longWord) {
		return jxrPrefilter2Fast(a, b)
	}
	b = check.value(b - ((a + 2) >> 2))
	a = check.value(a - (b >> 13))
	a = check.value(a - (b >> 9))
	a = check.value(a - (b >> 5))
	a = check.value(a - ((b + 1) >> 1))
	b = check.value(b - ((a + 2) >> 2))
	return a, b
}

// jxrForwardRotate 执行表D.15的两点整数正旋转
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrForwardRotate(a, b int64, check *jxrArithmetic) (int64, int64) {
	b = check.value(b - ((a + 1) >> 1))
	a = check.value(a + ((b + 1) >> 1))
	return a, b
}

// jxrForwardScale 执行表D.16的两点整数正缩放
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrForwardScale(a, b int64, check *jxrArithmetic) (int64, int64) {
	b = check.value(b - ((a * 3) >> 4))
	b = check.value(b - (a >> 7))
	b = check.value(b + (a >> 10))
	a = check.value(a - ((b * 3) >> 3))
	b = check.value((a >> 1) - b)
	a = check.value(a - b)
	return a, b
}

// jxrT2x2Pre 执行表D.14的四点预滤波变换，对应规范性T2x2hPOST的逆运算
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrT2x2Pre(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	a = check.value(a + d)
	b = check.value(b - c)
	t1, t2 := d, c
	c = check.value(((a - b) >> 1) - t1)
	d = check.value(t2 + (b >> 1))
	b = check.value(b + c)
	a = check.value(a - ((d*3 + 4) >> 3))
	return a, b, c, d
}

// jxrPrefilter4x4Fast 执行无校验的四乘四重叠预滤波快路径
// 入参: v 处于安全范围的十六个输入值，原地替换为预滤波结果
func jxrPrefilter4x4Fast(v *[16]int64) {
	v0, v1, v2, v3 := v[0], v[1], v[2], v[3]
	v4, v5, v6, v7 := v[4], v[5], v[6], v[7]
	v8, v9, v10, v11 := v[8], v[9], v[10], v[11]
	v12, v13, v14, v15 := v[12], v[13], v[14], v[15]
	v0 += v15
	v3 -= v12
	t1, t2 := v15, v12
	v12 = ((v0 - v3) >> 1) - t1
	v15 = t2 + (v3 >> 1)
	v3 += v12
	v0 -= (v15*3 + 4) >> 3
	v1 += v14
	v2 -= v13
	t1, t2 = v14, v13
	v13 = ((v1 - v2) >> 1) - t1
	v14 = t2 + (v2 >> 1)
	v2 += v13
	v1 -= (v14*3 + 4) >> 3
	v4 += v11
	v7 -= v8
	t1, t2 = v11, v8
	v8 = ((v4 - v7) >> 1) - t1
	v11 = t2 + (v7 >> 1)
	v7 += v8
	v4 -= (v11*3 + 4) >> 3
	v5 += v10
	v6 -= v9
	t1, t2 = v10, v9
	v9 = ((v5 - v6) >> 1) - t1
	v10 = t2 + (v6 >> 1)
	v6 += v9
	v5 -= (v10*3 + 4) >> 3
	v15 -= (v0 * 3) >> 4
	v15 -= v0 >> 7
	v15 += v0 >> 10
	v0 -= (v15 * 3) >> 3
	v15 = (v0 >> 1) - v15
	v0 -= v15
	v14 -= (v1 * 3) >> 4
	v14 -= v1 >> 7
	v14 += v1 >> 10
	v1 -= (v14 * 3) >> 3
	v14 = (v1 >> 1) - v14
	v1 -= v14
	v11 -= (v4 * 3) >> 4
	v11 -= v4 >> 7
	v11 += v4 >> 10
	v4 -= (v11 * 3) >> 3
	v11 = (v4 >> 1) - v11
	v4 -= v11
	v10 -= (v5 * 3) >> 4
	v10 -= v5 >> 7
	v10 += v5 >> 10
	v5 -= (v10 * 3) >> 3
	v10 = (v5 >> 1) - v10
	v5 -= v10
	v12 -= (v13 + 1) >> 1
	v13 += (v12 + 1) >> 1
	v8 -= (v9 + 1) >> 1
	v9 += (v8 + 1) >> 1
	v3 -= (v7 + 1) >> 1
	v7 += (v3 + 1) >> 1
	v2 -= (v6 + 1) >> 1
	v6 += (v2 + 1) >> 1
	v15 += v10
	v14 -= v11
	t1, t2 = v15>>1, v14>>1
	v10 -= t1
	v11 += t2
	v10 += (v11*3 + 4) >> 3
	v11 -= (v10*3 + 2) >> 2
	v10 += (v11*3 + 6) >> 3
	v11 -= t2
	v10 += t1
	v14 += v11
	v15 -= v10
	v0 += v15
	v12 -= v3
	t := (v0 - v12) >> 1
	oldC := v3
	v3, v15 = t-v15, t-oldC
	v0 -= v15
	v12 += v3
	v1 += v14
	v2 -= v13
	t = (v1 - v2) >> 1
	oldC = v13
	v13, v14 = t-v14, t-oldC
	v1 -= v14
	v2 += v13
	v4 += v11
	v7 -= v8
	t = (v4 - v7) >> 1
	oldC = v8
	v8, v11 = t-v11, t-oldC
	v4 -= v11
	v7 += v8
	v5 += v10
	v6 -= v9
	t = (v5 - v6) >> 1
	oldC = v9
	v9, v10 = t-v10, t-oldC
	v5 -= v10
	v6 += v9
	v[0], v[1], v[2], v[3] = v0, v1, v2, v3
	v[4], v[5], v[6], v[7] = v4, v5, v6, v7
	v[8], v[9], v[10], v[11] = v8, v9, v10, v11
	v[12], v[13], v[14], v[15] = v12, v13, v14, v15
}

// jxrPrefilter4Fast 执行无校验的四点重叠预滤波快路径
// 入参: a, b, c, d 处于安全范围的输入值
// 返回: int64 四个预滤波值
func jxrPrefilter4Fast(a, b, c, d int64) (int64, int64, int64, int64) {
	a += d
	b += c
	d -= (a + 1) >> 1
	c -= (b + 1) >> 1
	d -= (c + 1) >> 1
	c += (d + 1) >> 1
	d, c = -d, -c
	a -= d
	b -= c
	d += a >> 1
	c += b >> 1
	a -= (d*3 + 4) >> 3
	b -= (c*3 + 4) >> 3
	d -= (a * 3) >> 4
	d -= a >> 7
	d += a >> 10
	a -= (d * 3) >> 3
	d = (a >> 1) - d
	a -= d
	c -= (b * 3) >> 4
	c -= b >> 7
	c += b >> 10
	b -= (c * 3) >> 3
	c = (b >> 1) - c
	b -= c
	d += (a + 1) >> 1
	c += (b + 1) >> 1
	a -= d
	b -= c
	return a, b, c, d
}

// jxrPrefilter2x2Fast 执行无校验的二乘二重叠预滤波快路径
// 入参: a, b, c, d 处于安全范围的输入值
// 返回: int64 四个预滤波值
func jxrPrefilter2x2Fast(a, b, c, d int64) (int64, int64, int64, int64) {
	a += d
	b += c
	d -= (a + 1) >> 1
	c -= (b + 1) >> 1
	b -= (a + 2) >> 2
	a -= b >> 13
	a -= b >> 9
	a -= b >> 5
	a -= (b + 1) >> 1
	b -= (a + 2) >> 2
	d += (a + 1) >> 1
	c += (b + 1) >> 1
	a -= d
	b -= c
	return a, b, c, d
}

// jxrPrefilter2Fast 执行无校验的两点重叠预滤波快路径
// 入参: a, b 处于安全范围的输入值
// 返回: int64 两个预滤波值
func jxrPrefilter2Fast(a, b int64) (int64, int64) {
	b -= (a + 2) >> 2
	a -= b >> 13
	a -= b >> 9
	a -= b >> 5
	a -= (b + 1) >> 1
	b -= (a + 2) >> 2
	return a, b
}
