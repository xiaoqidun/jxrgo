# JXRGo [![PkgGoDev](https://pkg.go.dev/badge/github.com/xiaoqidun/jxrgo)](https://pkg.go.dev/github.com/xiaoqidun/jxrgo)
高性能、零依赖、纯 Go 语言 JXR 图像编解码库

# 安装指南
```shell
go get -u github.com/xiaoqidun/jxrgo
```

# 解码图像
```go
package main

import (
	"image/png"
	"log"
	"os"

	"github.com/xiaoqidun/jxrgo"
)

func main() {
	file, err := os.Open("test.jxr")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	img, err := jxrgo.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	out, err := os.Create("test.png")
	if err != nil {
		log.Fatal(err)
	}
	if err := png.Encode(out, img); err != nil {
		out.Close()
		log.Fatal(err)
	}
	if err := out.Close(); err != nil {
		log.Fatal(err)
	}
}
```

# 标准用法
```go
package main

import (
	"image"
	"image/png"
	"log"
	"os"

	_ "github.com/xiaoqidun/jxrgo"
)

func main() {
	file, err := os.Open("test.jxr")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	out, err := os.Create("test.png")
	if err != nil {
		log.Fatal(err)
	}
	if err := png.Encode(out, img); err != nil {
		out.Close()
		log.Fatal(err)
	}
	if err := out.Close(); err != nil {
		log.Fatal(err)
	}
}
```

# 编码图像
```go
package main

import (
	"image/png"
	"log"
	"os"

	"github.com/xiaoqidun/jxrgo"
)

func main() {
	file, err := os.Open("test.png")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	img, err := png.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	out, err := os.Create("test.jxr")
	if err != nil {
		log.Fatal(err)
	}
	if err := jxrgo.Encode(out, img, nil); err != nil {
		out.Close()
		log.Fatal(err)
	}
	if err := out.Close(); err != nil {
		log.Fatal(err)
	}
}
```

# 授权协议
本项目使用 [Apache License 2.0](https://github.com/xiaoqidun/jxrgo/blob/main/LICENSE) 授权协议