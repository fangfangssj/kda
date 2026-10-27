---
type: API
title: asc_float2int32
description: 将float类型的标量转换为int32_t类型并返回。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/scalar_compute/scalar_convert/asc_float2int32.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/scalar_compute/compute/scalar_convert.h
- include/c_api/asc_simd.h
---

# asc_float2int32

## 功能说明

头文件路径为：`"c_api/scalar_compute/compute/scalar_convert.h"`。

将`float`类型的标量转换为`int32_t`类型并返回。

关于舍入模式和饱和/非饱和模式的详细说明，请参见[舍入模式](../../reg_compute/reg_convert/rounding_mode.md)。

## 函数原型

```c
// 占位符形式
__aicore__ inline int32_t asc_float2int32_<round_mode>(float value)
```

`<round_mode>`支持`rd`（FLOOR）、`rn`（RINT）、`rna`（ROUND）和`ru`（CEIL）。

### 函数原型典型示例

```c
__aicore__ inline int32_t asc_float2int32_rn(float value)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :------ | :----- | :----------------- |
| value | 输入 | 源操作数（标量），数据类型为`float`。 |

## 返回值说明

返回`value`精度转换成`int32_t`的结果。

## 约束说明

- 使用饱和或非饱和模式时，需要配置`ctrl`寄存器，详细说明请参见[asc_set_ctrl](../../spr/asc_set_ctrl.md)。
- 输入值超出`int32_t`的表示范围[−2^31, 2^31-1]时，饱和模式下，结果钳位到`int32_t`的最大值或最小值；非饱和模式下，结果截断为低32位。
- 输入为nan或inf时返回0。
- 本接口运行在标量流水（`PIPE_S`）上，同一流水内的数据依赖由指令执行顺序保证，无需额外同步。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
template <typename T>
void print_data(const char* label, const std::vector<T>& values)
{
    std::cout << label << ":";
    const size_t count = values.size() < 8 ? values.size() : 8;
    for (size_t i = 0; i < count; ++i) std::cout << ' ' << +values[i];
    if (values.size() > count) std::cout << " ...";
    std::cout << std::endl;
}

template <typename T>
bool compare_data(const std::vector<T>& actual, const std::vector<T>& expected, double tolerance = 0.0)
{
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == expected[i]) continue;
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        if (diff > tolerance || diff < -tolerance) return false;
    }
    return true;
}

constexpr uint32_t ELEMENTS = 8;

__global__ __vector__ void asc_float2int32_kernel(__gm__ int64_t* output)
{
    asc_init();
    constexpr float value = 1.25f;
    output[0] = asc_float2int32_rn(value);
    output[1] = asc_float2int32_rna(value);
    output[2] = asc_float2int32_rd(value);
    output[3] = asc_float2int32_ru(value);
    asc_sync();
}
} // namespace

int main()
{
    std::vector<int64_t> input = {0, 0, 0, 0};
    std::vector<int64_t> golden = {1, 1, 1, 2};
    input.resize(ELEMENTS, 0);
    golden.resize(ELEMENTS, 0);
    std::vector<int64_t> output(ELEMENTS, -1);
    aclInit(nullptr);
    aclrtSetDevice(0);
    int64_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), (ELEMENTS) * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(output_device, input.size() * sizeof(int64_t), input.data(), input.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    asc_float2int32_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), output.size() * sizeof(int64_t), output_device, output.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = compare_data(output, golden);
    std::cout << (passed ? "[Success] asc_float2int32_rn/asc_float2int32_rna/asc_float2int32_rd/asc_float2int32_ru passed." : "[Failed] asc_float2int32_rn/asc_float2int32_rna/asc_float2int32_rd/asc_float2int32_ru failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
