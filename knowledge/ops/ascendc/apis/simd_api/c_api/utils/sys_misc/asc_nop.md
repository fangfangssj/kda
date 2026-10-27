---
type: API
title: asc_nop
description: 生成一条空操作指令，不执行实际计算。每调用一次阻塞一拍流水。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_misc/asc_nop.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/utils/sys_misc.h
- include/c_api/asc_simd.h
---

# asc_nop

## 功能说明

头文件路径为：`"c_api/utils/sys_misc.h"`。

生成一条空操作指令，不执行实际计算。每调用一次阻塞一拍流水。

## 函数原型

```c
__aicore__ inline void asc_nop()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

如果需要精确阻塞n拍流水，需要在n个`asc_nop`接口调用前后插入[`asc_sync`](../../sync/intra_core_sync/asc_sync.md)，例如：

```c
// 阻塞3拍流水
asc_sync();
asc_nop();
asc_nop();
asc_nop();
asc_sync();
```

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <iostream>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

namespace {
__global__ __vector__ void asc_nop_kernel()
{
    asc_init();
    asc_sync();
    asc_nop();
    asc_nop();
    asc_nop();
    asc_sync();
}
} // namespace

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    asc_nop_kernel<<<1, 0>>>();
    const aclError ret = aclrtSynchronizeDevice();
    const bool passed = ret == ACL_SUCCESS;
    std::cout << (passed ? "[Success] asc_nop passed." : "[Failed] asc_nop failed.") << std::endl;
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
