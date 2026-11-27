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
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
)

// normalizeJXRWorkers 将公开选项中的并发数规范化为正整数上限
// 入参: requested 调用方请求的并发上限，零或负值取当前CPU核数
// 返回: int 至少为一的有效并发上限
func normalizeJXRWorkers(requested int) int {
	if requested == 1 {
		return 1
	}
	if requested <= 0 {
		requested = runtime.GOMAXPROCS(0)
	}
	if requested < 1 {
		return 1
	}
	return requested
}

// resolveJXRWorkers 根据请求上限与任务数量确定实际并发度，单任务或单核请求直接返回一
// 入参: requested 请求并发数，非正数取当前CPU核数, tasks 待执行任务数
// 返回: int 最终并发工作数
func resolveJXRWorkers(requested, tasks int) int {
	if tasks <= 1 {
		return 1
	}
	return min(tasks, normalizeJXRWorkers(requested))
}

// budgetJXRWorkers 根据可用内存预算与单个工作区开销收缩并发度，最小保底为一
// 入参: workers 目标并发数, available 可用字节预算, perWorker 单个并发工作区字节开销
// 返回: int 预算安全范围内的并发数
func budgetJXRWorkers(workers int, available, perWorker uint64) int {
	if workers <= 1 || perWorker == 0 {
		return workers
	}
	allowed := available / perWorker
	if allowed <= 1 {
		return 1
	}
	if allowed < uint64(workers) {
		return int(allowed)
	}
	return workers
}

// jxrMacroChunkPlan 计算宏块数组并行的有效并发数、每片宏块数与分片总数
// 入参: total 宏块总数, minChunk 每片最小宏块数, requested 请求并发上限
// 返回: int 并发数, int 每片大小, int 分片数
func jxrMacroChunkPlan(total, minChunk, requested int) (int, int, int) {
	if requested <= 1 || minChunk <= 0 || total < minChunk*2 {
		return 1, total, 1
	}
	maxChunks := total / minChunk
	workers := resolveJXRWorkers(requested, maxChunks)
	if workers <= 1 {
		return 1, total, 1
	}
	chunks := min(maxChunks, workers*4)
	chunkSize := (total + chunks - 1) / chunks
	tasks := (total + chunkSize - 1) / chunkSize
	return min(workers, tasks), chunkSize, tasks
}

// runJXRWorkers 使用有界并发池执行独立任务，单核或单任务时在当前Goroutine零分配顺序执行
// 入参: ctx 已规范化Context, workers 最大并发数, tasks 任务总数, fn 单任务回调
// 返回: error 首个真实错误或Context取消错误
func runJXRWorkers(ctx context.Context, workers, tasks int, fn func(context.Context, int) error) error {
	if workers <= 1 || tasks <= 1 {
		for task := range tasks {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := fn(ctx, task); err != nil {
				return err
			}
		}
		return ctx.Err()
	}
	return runJXRWorkerSlots(ctx, workers, tasks, func(runCtx context.Context, _ int, task int) error {
		return fn(runCtx, task)
	})
}

// runJXRWorkerSlots 使用有界并发池为各工作Goroutine分配独占槽位执行任务，子Goroutine发生panic时取消其余任务并在等待退出后重抛
// 入参: ctx 已规范化Context, workers 最大并发数, tasks 任务总数, fn 接收Context、槽位号与任务号的回调
// 返回: error 首个真实错误或Context取消错误
func runJXRWorkerSlots(ctx context.Context, workers, tasks int, fn func(context.Context, int, int) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if tasks <= 0 {
		return nil
	}
	if workers <= 1 || tasks <= 1 {
		for task := range tasks {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := fn(ctx, 0, task); err != nil {
				return err
			}
		}
		return ctx.Err()
	}
	if workers > tasks {
		workers = tasks
	}
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var next atomic.Int64
	var errMu sync.Mutex
	var firstErr error
	var panicVal any
	firstTask, panicTask := tasks, tasks
	errRank, hasPanic := 0, false
	recordErr := func(task int, err error) {
		if err == nil {
			return
		}
		rank := 3
		if errors.Is(err, context.DeadlineExceeded) {
			rank = 2
		} else if errors.Is(err, context.Canceled) {
			rank = 1
		}
		errMu.Lock()
		if firstErr == nil || rank > errRank || rank == errRank && task < firstTask {
			firstErr = err
			firstTask = task
			errRank = rank
		}
		errMu.Unlock()
		cancel()
	}
	recordPanic := func(task int, recovered any) {
		errMu.Lock()
		if !hasPanic || task < panicTask {
			panicVal = recovered
			panicTask = task
			hasPanic = true
		}
		errMu.Unlock()
		cancel()
	}
	runSlot := func(slot int) {
		task := tasks
		completed := false
		defer func() {
			if !completed {
				recordPanic(task, recover())
			}
		}()
		for {
			if runCtx.Err() != nil {
				completed = true
				return
			}
			idx := int(next.Add(1) - 1)
			if idx >= tasks {
				completed = true
				return
			}
			task = idx
			if err := fn(runCtx, slot, idx); err != nil {
				recordErr(idx, err)
				completed = true
				return
			}
		}
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for slot := 1; slot < workers; slot++ {
		s := slot
		go func() {
			defer wg.Done()
			runSlot(s)
		}()
	}
	runSlot(0)
	wg.Wait()
	if hasPanic {
		panic(panicVal)
	}
	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}
