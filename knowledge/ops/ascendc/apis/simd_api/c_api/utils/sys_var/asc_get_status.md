---
type: API
title: asc_get_status
description: 获取状态信息。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_get_status.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
- include/c_api/asc_simd.h
---

# asc_get_status

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

读取当前核上下文中STATUS特殊寄存器的64bit整体值并按`int64_t`返回。STATUS是64bit状态寄存器，记录核运行过程中各类告警事件的发生情况，各状态位一旦因告警行为置位即保持，直到复位或被读取清除。

## 函数原型

```c
__aicore__ inline int64_t asc_get_status()
```

## 参数说明

无

## 返回值说明

状态信息。各bit含义如下：

**表1** 状态寄存器比特位说明

| bit范围    | 含义 |
| ----------- |:----|
| 5 | 浮点运算溢出。SIMD指令int16_t和int32_t算术运算溢出也会上报到该位。 |
| 6 | 浮点运算下溢（结果浮点数小于非规格化数能表示的最小值，则结果为0）。 |
| 7 | 将任意浮点数转换为无符号整数时，输入为负数。 |
| 8 | 从L0C到UB的数据搬运过程中发生溢出（float->half、int32_t->half）。 |
| 9 | 从L0C到UB的数据搬运过程中发生下溢（float->half）。 |
| 10 | CUBE累加运算溢出（可能是float、half、int32_t）。 |
| 11 | CUBE累加运算下溢（可能是float、half）。 |
| 13 | 标量指令输入为NaN/INF。 |
| 14 | 向量指令输入为NaN/INF。 |
| 15 | CUBE指令输入为NaN/INF。 |
| 61 | 数据搬运类指令输入为NaN/INF |
| 其它bit位 | 保留位。 |

## 流水类型

PIPE_S

## 约束说明

- 本接口为只读查询接口，仅读取状态寄存器当前值，可在AIC与AIV中调用。
- 本接口不支持纯SIMT编译模式。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
constexpr uint32_t ELEMENTS = 16;
constexpr uint32_t BYTES = ELEMENTS * sizeof(uint64_t);

void print_data(const char* label, const std::vector<uint64_t>& data)
{
    std::cout << label << ":";
    for (uint32_t i = 0; i < 8; ++i) std::cout << ' ' << data[i];
    std::cout << " ..." << std::endl;
}

void print_hex(const char* label, const std::vector<uint64_t>& data)
{
    std::cout << label << ":" << std::hex;
    for (uint32_t i = 0; i < 4; ++i) std::cout << " 0x" << data[i];
    std::cout << std::dec << std::endl;
}

__global__ __vector__ void asc_get_status_kernel(__gm__ uint64_t* output)
{
    asc_init();
    output[0] = static_cast<uint64_t>(asc_get_status());
    output[1] = static_cast<uint64_t>(asc_get_status());
    asc_dcci_single(output);
}

} // namespace

int main()
{
    std::vector<uint64_t> output(ELEMENTS, ~0ULL);
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint64_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(output_device, BYTES, output.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_get_status_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BYTES, output_device, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Observed", output);
    print_hex("Observed hex", output);
    const bool passed = output[0] == output[1];
    std::cout << (passed ? "[Success] asc_get_status passed." : "[Failed] asc_get_status failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
