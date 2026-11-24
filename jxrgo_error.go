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
	"fmt"
	"strings"
)

// FormatError 表示码流、容器或参数格式错误
// Op说明触发错误的操作名，Detail描述具体格式原因
type FormatError struct {
	Op     string
	Detail string
	err    error
}

// UnsupportedError 表示尚不支持的格式特性或处理能力
// Op说明触发错误的操作名，Detail描述具体不支持的特性
type UnsupportedError struct {
	Op     string
	Detail string
	err    error
}

// LimitError 表示输入、输出、像素或工作内存超出上限
// Resource说明受限资源名称，Limit和Required在非零时分别表示上限与需求量，单位按Resource语义为字节、像素或数量
// Detail补充具体超限操作或场景说明
type LimitError struct {
	Resource string
	Limit    uint64
	Required uint64
	Detail   string
	err      error
}

// jxrCallbackError 标记来自外部回调的原始错误以便在公开入口原样透传
type jxrCallbackError struct {
	err error
}

// Error 返回外部回调原始错误的描述
// 返回: string 原始错误描述
func (e jxrCallbackError) Error() string {
	return e.err.Error()
}

// Unwrap 返回外部回调的原始错误
// 返回: error 原始错误
func (e jxrCallbackError) Unwrap() error {
	return e.err
}

// Error 返回格式错误描述
// 返回: string 错误描述
func (e *FormatError) Error() string {
	if e == nil {
		return ErrInvalid.Error()
	}
	if e.Op != "" && e.Detail != "" {
		return e.Op + ": invalid format: " + e.Detail
	}
	if e.Op != "" {
		return e.Op + ": invalid format"
	}
	if e.Detail != "" {
		return "invalid format: " + e.Detail
	}
	return ErrInvalid.Error()
}

// Unwrap 返回底层原因错误或通用无效数据错误以便通过errors.Is匹配
// 返回: error 底层错误或ErrInvalid
func (e *FormatError) Unwrap() error {
	if e == nil {
		return nil
	}
	if e.err != nil {
		return e.err
	}
	return ErrInvalid
}

// Error 返回不支持特性的错误描述
// 返回: string 错误描述
func (e *UnsupportedError) Error() string {
	if e == nil {
		return ErrUnsupported.Error()
	}
	if e.Op != "" && e.Detail != "" {
		return e.Op + ": unsupported: " + e.Detail
	}
	if e.Op != "" {
		return e.Op + ": unsupported"
	}
	if e.Detail != "" {
		return "unsupported: " + e.Detail
	}
	return ErrUnsupported.Error()
}

// Unwrap 返回底层原因错误或通用不支持错误以便通过errors.Is匹配
// 返回: error 底层错误或ErrUnsupported
func (e *UnsupportedError) Unwrap() error {
	if e == nil {
		return nil
	}
	if e.err != nil {
		return e.err
	}
	return ErrUnsupported
}

// Error 返回资源超限错误描述
// 返回: string 错误描述
func (e *LimitError) Error() string {
	if e == nil {
		return ErrLimit.Error()
	}
	if e.Limit != 0 || e.Required != 0 {
		if e.Detail != "" && e.Resource != "" {
			return fmt.Sprintf("%s limit exceeded: need %d, limit %d (%s)", e.Resource, e.Required, e.Limit, e.Detail)
		}
		if e.Resource != "" {
			return fmt.Sprintf("%s limit exceeded: need %d, limit %d", e.Resource, e.Required, e.Limit)
		}
		if e.Detail != "" {
			return fmt.Sprintf("limit exceeded: need %d, limit %d (%s)", e.Required, e.Limit, e.Detail)
		}
		return fmt.Sprintf("limit exceeded: need %d, limit %d", e.Required, e.Limit)
	}
	if e.Detail != "" && e.Resource != "" {
		return e.Detail + ": " + e.Resource + " limit exceeded"
	}
	if e.Resource != "" {
		return e.Resource + " limit exceeded"
	}
	if e.Detail != "" {
		return e.Detail + ": limit exceeded"
	}
	return ErrLimit.Error()
}

// Unwrap 返回底层原因错误或通用资源上限错误以便通过errors.Is匹配
// 返回: error 底层错误或ErrLimit
func (e *LimitError) Unwrap() error {
	if e == nil {
		return nil
	}
	if e.err != nil {
		return e.err
	}
	return ErrLimit
}

// wrapJXRError 将公开入口返回的哨兵错误统一包装为指针结构化错误，并保留取消、关闭与I/O语义
// 入参: op 公开操作名, err 待归一化错误
// 返回: error 归一化后的错误
func wrapJXRError(op string, err error) error {
	if err == nil {
		return nil
	}
	var cbErr jxrCallbackError
	if errors.As(err, &cbErr) {
		return cbErr.err
	}
	var formatErr *FormatError
	if errors.As(err, &formatErr) {
		return err
	}
	var unsupportedErr *UnsupportedError
	if errors.As(err, &unsupportedErr) {
		return err
	}
	var limitErr *LimitError
	if errors.As(err, &limitErr) {
		return err
	}
	if errors.Is(err, ErrClosed) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	if errors.Is(err, ErrLimit) {
		detail := jxrErrorDetail(err, ErrLimit)
		if detail == "" {
			return &LimitError{Resource: op, err: err}
		}
		return &LimitError{Resource: detail, Detail: op, err: err}
	}
	if errors.Is(err, ErrUnsupported) {
		return &UnsupportedError{Op: op, Detail: jxrErrorDetail(err, ErrUnsupported), err: err}
	}
	if errors.Is(err, ErrInvalid) {
		return &FormatError{Op: op, Detail: jxrErrorDetail(err, ErrInvalid), err: err}
	}
	return err
}

// jxrSaturatedAdd 计算两个无符号整数的和并在溢出时饱和为最大值
// 入参: a, b 无符号加数
// 返回: uint64 饱和求和结果
func jxrSaturatedAdd(a, b uint64) uint64 {
	if a > ^uint64(0)-b {
		return ^uint64(0)
	}
	return a + b
}

// jxrSaturatedMul 计算两个无符号整数的积并在溢出时饱和为最大值
// 入参: a, b 无符号乘数
// 返回: uint64 饱和求积结果
func jxrSaturatedMul(a, b uint64) uint64 {
	if a == 0 || b == 0 {
		return 0
	}
	if a > ^uint64(0)/b {
		return ^uint64(0)
	}
	return a * b
}

// isJXRMaxOutputBytesError 判断错误是否为输出字节上限超限
// 入参: err 待判断错误
// 返回: bool 是否由输出字节上限触发
func isJXRMaxOutputBytesError(err error) bool {
	var limitErr *LimitError
	return errors.As(err, &limitErr) && limitErr.Resource == "max output bytes"
}

// jxrErrorDetail 从已包装哨兵的错误文本中剥离重复的哨兵前缀
// 入参: err 原始错误, sentinel 匹配的哨兵错误
// 返回: string 精简后的原因描述
func jxrErrorDetail(err, sentinel error) string {
	msg := err.Error()
	prefix := sentinel.Error()
	if msg == prefix {
		return ""
	}
	return strings.TrimPrefix(msg, prefix+": ")
}
