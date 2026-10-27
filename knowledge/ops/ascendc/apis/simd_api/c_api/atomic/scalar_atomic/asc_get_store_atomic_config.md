---
type: API
title: asc_get_store_atomic_config
description: 获取原子操作启用位与原子操作类型的值，可用于校验asc_set_store_atomic_config_v1和asc_set_store_atomic_config_v2（废弃）设置的原子操作启用位和原子操作类型的值是否符合预期。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/atomic/scalar_atomic/asc_get_store_atomic_config.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/atomic/scalar_atomic.h
- include/c_api/asc_simd.h
---

# asc_get_store_atomic_config

## 功能说明

头文件路径为：`"c_api/atomic/scalar_atomic.h"`。

获取原子操作启用位与原子操作类型的值，可用于校验[asc_set_store_atomic_config_v1](asc_set_store_atomic_config_v1.md)和[asc_set_store_atomic_config_v2（废弃）](../../deprecated_interface/asc_set_store_atomic_config_v2_deprecated.md)设置的原子操作启用位和原子操作类型的值是否符合预期。


**Ascend 950PR&950DT系列产品上该接口已废弃。原子加操作可以直接使用[asc_atomic_add](asc_atomic_add.md)实现。无需再对原子操作启用位和原子操作类型的值进行配置和校验。**


## 函数原型

```cpp
__aicore__ inline void asc_get_store_atomic_config(asc_store_atomic_config& config)
```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输出 | 用于获取原子操作启用位和原子操作类型的值，详细说明请参考[asc_store_atomic_config](../../defs/union/asc_store_atomic_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

将以下代码保存为`example.asc`后，执行如下编译运行命令：


以[NPU架构版本2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-2201 && ./main
```


```cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
constexpr uint32_t ELEMENTS = 64;
constexpr uint32_t BYTES = ELEMENTS * sizeof(int32_t);

template <typename T>
void PrintData(const char* label, const std::vector<T>& data)
{
    std::cout << label << ":";
    const size_t count = data.size() < 8 ? data.size() : 8;
    for (size_t i = 0; i < count; ++i) std::cout << ' ' << +data[i];
    if (data.size() > count) std::cout << " ...";
    std::cout << std::endl;
}

__global__ __vector__ void AscGetStoreAtomicConfigKernel(__gm__ int32_t* output, __gm__ int32_t* input,
    __gm__ uint64_t* raw_config)
{
    asc_init();
    // 开启float类型的原子求和操作，再读取原子配置寄存器的原始值。
    asc_set_store_atomic_config_v1(1, 0);
    asc_store_atomic_config config;
    asc_get_store_atomic_config(config);
    raw_config[0] = config.config;
    asc_dcci_single(raw_config);
    asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
    asc_disable_dma_atomic();
}

} // namespace

int main()
{
    std::vector<int32_t> input(ELEMENTS), output(ELEMENTS, 10), golden(ELEMENTS);
    for (uint32_t i = 0; i < ELEMENTS; ++i) {
        input[i] = static_cast<int32_t>(i % 8 + 1);
        golden[i] = 10;
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    int32_t *input_device = nullptr, *output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&input_device), BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&output_device), BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    uint64_t* config_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&config_device), sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, BYTES, input.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_device, BYTES, output.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    AscGetStoreAtomicConfigKernel<<<1, 0>>>(output_device, input_device, config_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BYTES, output_device, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<uint64_t> raw_config(1);
    aclrtMemcpy(raw_config.data(), sizeof(uint64_t), config_device, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    PrintData("Input", input);
    PrintData("Output", output);
    PrintData("Golden", golden);
    PrintData("Raw atomic config", raw_config);
    aclrtFree(config_device);
    bool passed = output == golden && (raw_config[0] == 1);
    std::cout << (passed ? "[Success] asc_get_store_atomic_config passed." : "[Failed] asc_get_store_atomic_config failed.") << std::endl;
    aclrtFree(input_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
