---
type: API
title: asc_copy_ub2ub
description: 将数据从Unified Buffer（UB）搬运到UB。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2ub.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
- include/c_api/asc_simd.h
---

# asc_copy_ub2ub

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

将数据从Unified Buffer（UB）搬运到UB。

本接口支持连续数据搬运和高维切分数据搬运。连续数据搬运按`size`指定的字节数在UB内连续搬运，如图1所示；高维切分数据搬运按`burst_count`指定搬运的数据块个数，每个数据块按`burst_len`指定的DataBlock个数搬运，并通过`src_gap`和`dst_gap`指定源操作数和目的操作数相邻数据块之间的间隔，如图2所示。

**图1**  连续搬运示意图
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/c_api/figures/asc_copy_ub2ub_continuous.png "连续搬运示意图")

**图2**  高维切分搬运示意图
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/c_api/figures/asc_copy_ub2ub_highdim_split.png "高维切分搬运示意图")

本接口仅在AIV上生效。

## 函数原型

### 连续数据搬运

```c
__aicore__ inline void asc_copy_ub2ub(__ubuf__ void* dst,
                                      __ubuf__ void* src,
                                      uint32_t size)
```

### 高维切分数据搬运

```cpp
__aicore__ inline void asc_copy_ub2ub(__ubuf__ void* dst, __ubuf__ void* src, uint16_t burst_count, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap)
```

## 参数说明

### 连续数据搬运

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| dst | 输出 | 目的UB的起始地址。需要32字节对齐。 |
| src | 输入 | 源UB的起始地址。需要32字节对齐。 |
| size | 输入 | 待搬运数据大小，单位字节。取值范围：[32, 65535×32]，且需为32的整数倍。 |

### 高维切分数据搬运

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| dst | 输出 | 目的UB的起始地址。需要32字节对齐。 |
| src | 输入 | 源UB的起始地址。需要32字节对齐。 |
| burst_count | 输入 | 待搬运的连续数据块个数。取值范围：[1, 65535]。 |
| burst_len | 输入 | 每个连续数据块的长度，单位为DataBlock（32字节）。取值范围：[1, 65535]。 |
| src_gap | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔），单位为DataBlock（32字节）。取值范围：[0, $2^{16}−1$]。 |
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔），单位为DataBlock（32字节）。取值范围：[0, $2^{16}−1$]。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

### 通用约束

- 本接口在非AIV上调用直接返回。
- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 源操作数与目的操作数的有效搬运DataBlock要求不重叠。高维切分搬运模式下，若`src_gap`和`dst_gap`使源、目的的实际搬运DataBlock互不重叠，则允许`src`和`dst`覆盖的整体地址跨度存在交叠。

- Ascend 950PR&950DT系列产品的`PIPE_V`内部同步由硬件自动保证，多条本接口指令之间无需额外插入同步。


- 如果本指令与其他指令存在数据依赖，需要插入[同步指令](../sync/system_sync_overview.md)，保证多个指令串行化，防止出现异常数据。


### 连续搬运模式约束

- `size`需为32的整数倍，否则`size/32`截断后实际搬运字节数不足，多出非整32字节部分被丢弃。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```c
#include <cstdint>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {

constexpr uint32_t INPUT_BYTES = 256;
constexpr uint32_t OUTPUT_BYTES = 256;

__global__ __vector__ void asc_copy_ub2ub_kernel(__gm__ uint8_t* output, __gm__ uint8_t* input)
{
    asc_init();
    __ubuf__ uint8_t source[INPUT_BYTES], copied[INPUT_BYTES];
    asc_copy_gm2ub_align(source, input, INPUT_BYTES);
    asc_sync_mte2(0);
    // Copy INPUT_BYTES within UB on PIPE_V before the MTE3 consumer starts.
    asc_copy_ub2ub(copied, source, INPUT_BYTES);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, copied, INPUT_BYTES);
    asc_sync_mte3(0);
}

void print_data(const char* name, const std::vector<uint8_t>& data)
{
    std::cout << name << ":";
    const uint32_t count = data.size() < 32 ? data.size() : 32;
    for (uint32_t i = 0; i < count; ++i) std::cout << ' ' << +data[i];
    if (data.size() > count) std::cout << " ...";
    std::cout << std::endl;
}
} // namespace

int main()
{
    std::vector<uint8_t> input(INPUT_BYTES), output(OUTPUT_BYTES, 0), golden(OUTPUT_BYTES, 0);
    for (uint32_t i = 0; i < INPUT_BYTES; ++i) input[i] = static_cast<uint8_t>(i + 1);
    for (uint32_t i = 0; i < 256; ++i) golden[i] = input[i];
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint8_t *input_device = nullptr, *output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&input_device), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&output_device), OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, INPUT_BYTES, input.data(), INPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_device, OUTPUT_BYTES, output.data(), OUTPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_copy_ub2ub_kernel<<<1, 0>>>(output_device, input_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), OUTPUT_BYTES, output_device, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = output == golden;
    std::cout << (passed ? "[Success] asc_copy_ub2ub passed." : "[Failed] asc_copy_ub2ub failed.") << std::endl;
    aclrtFree(input_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
