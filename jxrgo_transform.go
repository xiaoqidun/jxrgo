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

import "context"

// jxrInversePermutation 保存T.832表164规定的逆变换系数位置
var jxrInversePermutation = [16]uint8{0, 8, 4, 13, 2, 15, 3, 14, 1, 12, 5, 9, 7, 11, 6, 10}

// jxrInverse4 执行四乘四整数逆变换，阶段内各子变换使用独立系数
// 入参: values 有界的十六个变换系数，原地替换为重建值, check 局部变量校验状态
func jxrInverse4(values *[16]int64, check *jxrArithmetic) {
	if check == nil || jxrSafeBlock16(values, check.longWord) {
		jxrInverse4Fast(values)
		return
	}
	var v [16]int64
	for i, at := range jxrInversePermutation {
		v[at] = check.value(values[i])
	}
	v[0], v[1], v[4], v[5] = jxrT2x2(v[0], v[1], v[4], v[5], 1, check)
	v[2], v[3], v[6], v[7] = jxrInverseOdd(v[2], v[3], v[6], v[7], check)
	v[8], v[12], v[9], v[13] = jxrInverseOdd(v[8], v[12], v[9], v[13], check)
	v[10], v[11], v[14], v[15] = jxrInverseOddOdd(v[10], v[11], v[14], v[15], false, check)
	v[0], v[3], v[12], v[15] = jxrT2x2(v[0], v[3], v[12], v[15], 0, check)
	v[5], v[6], v[9], v[10] = jxrT2x2(v[5], v[6], v[9], v[10], 0, check)
	v[1], v[2], v[13], v[14] = jxrT2x2(v[1], v[2], v[13], v[14], 0, check)
	v[4], v[7], v[8], v[11] = jxrT2x2(v[4], v[7], v[8], v[11], 0, check)
	*values = v
}

// jxrT2x2 执行带指定舍入量的四点整数变换
// 入参: a, b, c, d 有界输入值, round 零或一的舍入量, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrT2x2(a, b, c, d, round int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	a = check.value(a + d)
	b = check.value(b - c)
	t := check.value((a - b + round) >> 1)
	oldC := c
	c, d = check.value(t-d), check.value(t-oldC)
	a = check.value(a - d)
	b = check.value(b + c)
	return a, b, c, d
}

// jxrInverseOdd 执行单奇频分量的四点整数逆变换
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrInverseOdd(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	b = check.value(b + d)
	a = check.value(a - c)
	d = check.value(d - (b >> 1))
	c = check.value(c + ((a + 1) >> 1))
	a = check.value(a - ((3*b + 4) >> 3))
	b = check.value(b + ((3*a + 4) >> 3))
	c = check.value(c - ((3*d + 4) >> 3))
	d = check.value(d + ((3*c + 4) >> 3))
	c = check.value(c - ((b + 1) >> 1))
	d = check.value(((a + 1) >> 1) - d)
	b = check.value(b + c)
	a = check.value(a - d)
	return a, b, c, d
}

// jxrInverseOddOdd 执行双奇频分量的逆变换或重叠后滤波变换
// 入参: a, b, c, d 有界输入值, post 是否采用重叠后滤波的舍入和符号规则, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrInverseOddOdd(a, b, c, d int64, post bool, check *jxrArithmetic) (int64, int64, int64, int64) {
	d = check.value(d + a)
	c = check.value(c - b)
	t1, t2 := check.value(d>>1), check.value(c>>1)
	a = check.value(a - t1)
	b = check.value(b + t2)
	r1, r2 := int64(3), int64(3)
	if post {
		r1, r2 = 6, 2
	}
	a = check.value(a - ((b*3 + r1) >> 3))
	b = check.value(b + ((a*3 + r2) >> 2))
	a = check.value(a - ((b*3 + 4) >> 3))
	b = check.value(b - t2)
	a = check.value(a + t1)
	c = check.value(c + b)
	d = check.value(d - a)
	if !post {
		b, c = check.value(-b), check.value(-c)
	}
	return a, b, c, d
}

// jxrT2 执行两点整数逆变换
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrT2(a, b int64, check *jxrArithmetic) (int64, int64) {
	a = check.value(a - ((b + 1) >> 1))
	b = check.value(b + a)
	return a, b
}

// jxrInverse4Fast 执行无校验的四乘四整数逆变换快路径
// 入参: values 处于安全范围的十六个变换系数，原地替换为重建值
func jxrInverse4Fast(values *[16]int64) {
	v0, v1, v2, v3 := values[0], values[8], values[4], values[6]
	v4, v5, v6, v7 := values[2], values[10], values[14], values[12]
	v8, v9, v10, v11 := values[1], values[11], values[15], values[13]
	v12, v13, v14, v15 := values[9], values[3], values[7], values[5]
	v0 += v5
	v1 -= v4
	t := (v0 - v1 + 1) >> 1
	oldC := v4
	v4, v5 = t-v5, t-oldC
	v0 -= v5
	v1 += v4
	v3 += v7
	v2 -= v6
	v7 -= v3 >> 1
	v6 += (v2 + 1) >> 1
	v2 -= (3*v3 + 4) >> 3
	v3 += (3*v2 + 4) >> 3
	v6 -= (3*v7 + 4) >> 3
	v7 += (3*v6 + 4) >> 3
	v6 -= (v3 + 1) >> 1
	v7 = ((v2 + 1) >> 1) - v7
	v3 += v6
	v2 -= v7
	v12 += v13
	v8 -= v9
	v13 -= v12 >> 1
	v9 += (v8 + 1) >> 1
	v8 -= (3*v12 + 4) >> 3
	v12 += (3*v8 + 4) >> 3
	v9 -= (3*v13 + 4) >> 3
	v13 += (3*v9 + 4) >> 3
	v9 -= (v12 + 1) >> 1
	v13 = ((v8 + 1) >> 1) - v13
	v12 += v9
	v8 -= v13
	v15 += v10
	v14 -= v11
	t1, t2 := v15>>1, v14>>1
	v10 -= t1
	v11 += t2
	v10 -= (v11*3 + 3) >> 3
	v11 += (v10*3 + 3) >> 2
	v10 -= (v11*3 + 4) >> 3
	v11 -= t2
	v10 += t1
	v14 += v11
	v15 -= v10
	v11, v14 = -v11, -v14
	v0 += v15
	v3 -= v12
	t = (v0 - v3) >> 1
	oldC = v12
	v12, v15 = t-v15, t-oldC
	v0 -= v15
	v3 += v12
	v5 += v10
	v6 -= v9
	t = (v5 - v6) >> 1
	oldC = v9
	v9, v10 = t-v10, t-oldC
	v5 -= v10
	v6 += v9
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
	values[0], values[1], values[2], values[3] = v0, v1, v2, v3
	values[4], values[5], values[6], values[7] = v4, v5, v6, v7
	values[8], values[9], values[10], values[11] = v8, v9, v10, v11
	values[12], values[13], values[14], values[15] = v12, v13, v14, v15
}

// jxrT2x2Fast 执行无校验的带舍入量四点整数变换快路径
// 入参: a, b, c, d 处于安全范围的输入值, round 零或一的舍入量
// 返回: int64 四个变换值
func jxrT2x2Fast(a, b, c, d, round int64) (int64, int64, int64, int64) {
	a += d
	b -= c
	t := (a - b + round) >> 1
	oldC := c
	c, d = t-d, t-oldC
	a -= d
	b += c
	return a, b, c, d
}

// jxrT2Fast 执行无校验的两点整数逆变换快路径
// 入参: a, b 处于安全范围的输入值
// 返回: int64 两个变换值
func jxrT2Fast(a, b int64) (int64, int64) {
	a -= (b + 1) >> 1
	b += a
	return a, b
}

// inverseFirstContext 执行首级逆变换并恢复缩放色度，按宏块和分量响应取消
// 入参: ctx 取消上下文, h 图像头
// 返回: []jxrComponentView 首级重建系数视图, error 数值范围或取消错误
func (c *jxrCoefficients) inverseFirstContext(ctx context.Context, h *jxrHeader) ([]jxrComponentView, error) {
	if err := c.inverseFirstValuesContext(ctx, h); err != nil {
		return nil, err
	}
	return c.componentViews(h, false), nil
}

// inverseFirstValuesContext 原地执行首级数值步骤，不为不使用视图的连续行处理分配描述数组
// 入参: ctx 取消上下文, h 图像头
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) inverseFirstValuesContext(ctx context.Context, h *jxrHeader) error {
	total := len(c.info)
	if c.workers > 1 && total >= 64 {
		workers, chunkSize, tasks := jxrMacroChunkPlan(total, 32, c.workers)
		if workers > 1 {
			return runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
				start := task * chunkSize
				return c.inverseFirstRange(runCtx, h, start, min(start+chunkSize, total))
			})
		}
	}
	return c.inverseFirstRange(ctx, h, 0, total)
}

// inverseFirstRange 原地执行指定宏块区间的首级逆变换并恢复缩放色度
// 入参: ctx 取消上下文, h 图像头, start 起始宏块, end 结束宏块
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) inverseFirstRange(ctx context.Context, h *jxrHeader, start, end int) error {
	longWord := h.longWord
	channels := c.plane.channels
	color := c.plane.color
	scaled := c.plane.scaled
	for at := start; at < end; at++ {
		if (at-start)&63 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		low := c.lowMacro(at)
		for i := range channels {
			src := (*[16]int64)(low[i*16 : i*16+16])
			if jxrSafeBlock16(src, longWord) {
				if i != 0 && color == 1 {
					src[0], src[1], src[2], src[3] = jxrT2x2Fast(src[0], src[1], src[2], src[3], 0)
					src[1], src[2] = src[2], src[1]
					if scaled {
						src[0] *= 2
						src[1] *= 2
						src[2] *= 2
						src[3] *= 2
					}
				} else if i != 0 && color == 2 {
					src[0], src[4] = jxrT2Fast(src[0], src[4])
					src[0], src[1], src[2], src[3] = jxrT2x2Fast(src[0], src[1], src[2], src[3], 0)
					src[1], src[2] = src[2], src[1]
					src[4], src[6], src[5], src[7] = jxrT2x2Fast(src[4], src[6], src[5], src[7], 0)
					src[5], src[6] = src[6], src[5]
					if scaled {
						for j := range 8 {
							src[j] *= 2
						}
					}
				} else {
					jxrInverse4Fast(src)
					if i != 0 && scaled {
						for j := range 16 {
							src[j] *= 2
						}
					}
				}
				continue
			}
			values := *src
			if err := checkJXRWords(values[:], longWord); err != nil {
				return err
			}
			check := newJXRArithmetic(longWord)
			count := 16
			if i != 0 && color == 1 {
				values[0], values[1], values[2], values[3] = jxrT2x2(values[0], values[1], values[2], values[3], 0, &check)
				values[1], values[2] = values[2], values[1]
				count = 4
			} else if i != 0 && color == 2 {
				values[0], values[4] = jxrT2(values[0], values[4], &check)
				values[0], values[1], values[2], values[3] = jxrT2x2(values[0], values[1], values[2], values[3], 0, &check)
				values[1], values[2] = values[2], values[1]
				values[4], values[6], values[5], values[7] = jxrT2x2(values[4], values[6], values[5], values[7], 0, &check)
				values[5], values[6] = values[6], values[5]
				count = 8
			} else {
				jxrInverse4(&values, &check)
			}
			if err := check.error(); err != nil {
				return err
			}
			if i != 0 && scaled {
				for j := range count {
					values[j] *= 2
				}
			}
			if err := checkJXRWords(values[:count], longWord); err != nil {
				return err
			}
			copy(low[i*16:i*16+count], values[:count])
		}
	}
	return ctx.Err()
}

// inverseSecondContext 组合首级重建值与高通系数，原地执行二级逆变换并响应取消
// 入参: ctx 取消上下文, h 图像头
// 返回: []jxrComponentView 二级重建样本视图, error 数值范围或取消错误
func (c *jxrCoefficients) inverseSecondContext(ctx context.Context, h *jxrHeader) ([]jxrComponentView, error) {
	if err := c.inverseSecondValuesContext(ctx, h); err != nil {
		return nil, err
	}
	return c.componentViews(h, true), nil
}

// inverseSecondValuesContext 原地组合低通及高通执行二级数值步骤，调用方可复用既有样本视图
// 入参: ctx 取消上下文, h 图像头
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) inverseSecondValuesContext(ctx context.Context, h *jxrHeader) error {
	total := len(c.info)
	if c.workers > 1 && total >= 64 {
		workers, chunkSize, tasks := jxrMacroChunkPlan(total, 32, c.workers)
		if workers > 1 {
			return runJXRWorkers(ctx, workers, tasks, func(runCtx context.Context, task int) error {
				start := task * chunkSize
				return c.inverseSecondRange(runCtx, h, start, min(start+chunkSize, total))
			})
		}
	}
	return c.inverseSecondRange(ctx, h, 0, total)
}

// inverseSecondRange 原地执行指定宏块区间的二级逆变换
// 入参: ctx 取消上下文, h 图像头, start 起始宏块, end 结束宏块
// 返回: error 数值范围或取消错误
func (c *jxrCoefficients) inverseSecondRange(ctx context.Context, h *jxrHeader, start, end int) error {
	longWord := h.longWord
	channels := c.plane.channels
	color := c.plane.color
	for at := start; at < end; at++ {
		if (at-start)&63 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		low, high := c.lowMacro(at), c.highMacro(at)
		for i := range channels {
			blocks := 16
			if i != 0 && color == 1 {
				blocks = 4
			} else if i != 0 && color == 2 {
				blocks = 8
			}
			lowBase := i * 16
			highBase := i * 256
			for block := range blocks {
				from := highBase + block*16
				dst := (*[16]int64)(high[from : from+16])
				dc := low[lowBase+block]
				saved := dst[0]
				dst[0] = dc
				if jxrSafeBlock16(dst, longWord) {
					jxrInverse4Fast(dst)
					continue
				}
				dst[0] = saved
				values := *dst
				values[0] = dc
				if err := checkJXRWords(values[:], longWord); err != nil {
					return err
				}
				check := newJXRArithmetic(longWord)
				jxrInverse4(&values, &check)
				if err := check.error(); err != nil {
					return err
				}
				if err := checkJXRWords(values[:], longWord); err != nil {
					return err
				}
				*dst = values
			}
		}
	}
	return ctx.Err()
}

// jxrForward4 执行四乘四可逆整数正变换，输出排列对应解码端的系数顺序
// 入参: values 有界的十六个样本，原地替换为变换系数, check 局部变量校验状态
func jxrForward4(values *[16]int64, check *jxrArithmetic) {
	if check == nil || jxrSafeBlock16(values, check.longWord) {
		jxrForward4Fast(values)
		return
	}
	v := *values
	v[0], v[3], v[12], v[15] = jxrT2x2(v[0], v[3], v[12], v[15], 0, check)
	v[5], v[6], v[9], v[10] = jxrT2x2(v[5], v[6], v[9], v[10], 0, check)
	v[1], v[2], v[13], v[14] = jxrT2x2(v[1], v[2], v[13], v[14], 0, check)
	v[4], v[7], v[8], v[11] = jxrT2x2(v[4], v[7], v[8], v[11], 0, check)
	v[0], v[1], v[4], v[5] = jxrT2x2(v[0], v[1], v[4], v[5], 1, check)
	v[2], v[3], v[6], v[7] = jxrForwardOdd(v[2], v[3], v[6], v[7], check)
	v[8], v[12], v[9], v[13] = jxrForwardOdd(v[8], v[12], v[9], v[13], check)
	v[10], v[11], v[14], v[15] = jxrForwardOddOdd(v[10], v[11], v[14], v[15], false, check)
	for index, from := range jxrInversePermutation {
		values[index] = check.value(v[from])
	}
}

// jxrForwardOdd 执行表D.9的单奇频四点整数正变换
// 入参: a, b, c, d 有界输入值, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrForwardOdd(a, b, c, d int64, check *jxrArithmetic) (int64, int64, int64, int64) {
	b = check.value(b - c)
	a = check.value(a + d)
	c = check.value(c + ((b + 1) >> 1))
	d = check.value(((a + 1) >> 1) - d)
	b = check.value(b - ((3*a + 4) >> 3))
	a = check.value(a + ((3*b + 4) >> 3))
	d = check.value(d - ((3*c + 4) >> 3))
	c = check.value(c + ((3*d + 4) >> 3))
	d = check.value(d + (b >> 1))
	c = check.value(c - ((a + 1) >> 1))
	b = check.value(b - d)
	a = check.value(a + c)
	return a, b, c, d
}

// jxrForwardOddOdd 执行双奇频正变换或重叠预滤波中的对应变换
// 入参: a, b, c, d 有界输入值, pre 是否采用重叠预滤波的舍入和符号规则, check 局部变量校验状态
// 返回: int64 四个变换值
func jxrForwardOddOdd(a, b, c, d int64, pre bool, check *jxrArithmetic) (int64, int64, int64, int64) {
	if !pre {
		b, c = check.value(-b), check.value(-c)
	}
	d = check.value(d + a)
	c = check.value(c - b)
	t1, t2 := check.value(d>>1), check.value(c>>1)
	a = check.value(a - t1)
	b = check.value(b + t2)
	r1, r2 := int64(3), int64(3)
	if pre {
		r1, r2 = 6, 2
	}
	a = check.value(a + ((b*3 + 4) >> 3))
	b = check.value(b - ((a*3 + r2) >> 2))
	a = check.value(a + ((b*3 + r1) >> 3))
	b = check.value(b - t2)
	a = check.value(a + t1)
	c = check.value(c + b)
	d = check.value(d - a)
	return a, b, c, d
}

// jxrForward2 执行表163两点整数逆变换的可逆正向运算
// 入参: a, b 有界输入值, check 局部变量校验状态
// 返回: int64 两个变换值
func jxrForward2(a, b int64, check *jxrArithmetic) (int64, int64) {
	b = check.value(b - a)
	a = check.value(a + ((b + 1) >> 1))
	return a, b
}

// jxrForward4Fast 执行无校验的四乘四整数正变换快路径
// 入参: values 处于安全范围的十六个样本，原地替换为变换系数
func jxrForward4Fast(values *[16]int64) {
	v0, v1, v2, v3 := values[0], values[1], values[2], values[3]
	v4, v5, v6, v7 := values[4], values[5], values[6], values[7]
	v8, v9, v10, v11 := values[8], values[9], values[10], values[11]
	v12, v13, v14, v15 := values[12], values[13], values[14], values[15]
	v0 += v15
	v3 -= v12
	t := (v0 - v3) >> 1
	oldC := v12
	v12, v15 = t-v15, t-oldC
	v0 -= v15
	v3 += v12
	v5 += v10
	v6 -= v9
	t = (v5 - v6) >> 1
	oldC = v9
	v9, v10 = t-v10, t-oldC
	v5 -= v10
	v6 += v9
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
	v0 += v5
	v1 -= v4
	t = (v0 - v1 + 1) >> 1
	oldC = v4
	v4, v5 = t-v5, t-oldC
	v0 -= v5
	v1 += v4
	v3 -= v6
	v2 += v7
	v6 += (v3 + 1) >> 1
	v7 = ((v2 + 1) >> 1) - v7
	v3 -= (3*v2 + 4) >> 3
	v2 += (3*v3 + 4) >> 3
	v7 -= (3*v6 + 4) >> 3
	v6 += (3*v7 + 4) >> 3
	v7 += v3 >> 1
	v6 -= (v2 + 1) >> 1
	v3 -= v7
	v2 += v6
	v12 -= v9
	v8 += v13
	v9 += (v12 + 1) >> 1
	v13 = ((v8 + 1) >> 1) - v13
	v12 -= (3*v8 + 4) >> 3
	v8 += (3*v12 + 4) >> 3
	v13 -= (3*v9 + 4) >> 3
	v9 += (3*v13 + 4) >> 3
	v13 += v12 >> 1
	v9 -= (v8 + 1) >> 1
	v12 -= v13
	v8 += v9
	v11, v14 = -v11, -v14
	v15 += v10
	v14 -= v11
	t1, t2 := v15>>1, v14>>1
	v10 -= t1
	v11 += t2
	v10 += (v11*3 + 4) >> 3
	v11 -= (v10*3 + 3) >> 2
	v10 += (v11*3 + 3) >> 3
	v11 -= t2
	v10 += t1
	v14 += v11
	v15 -= v10
	values[0], values[1], values[2], values[3] = v0, v8, v4, v13
	values[4], values[5], values[6], values[7] = v2, v15, v3, v14
	values[8], values[9], values[10], values[11] = v1, v12, v5, v9
	values[12], values[13], values[14], values[15] = v7, v11, v6, v10
}
