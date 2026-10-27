---
type: Guide
title: Printf：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。SIMD和SIMT编程调试场景下提供格式化输出功能，在算子Kernel侧打印日志信息。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/utils_api/tuning_interface/printf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/debug-api.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
- a2
- a3
aliases:
- SIMT printf
---

# printf

头文件路径为："utils/debug/asc_printf.h"。

本接口提供SIMD和SIMT编程调试场景下的格式化输出功能。

在算子Kernel侧的实现代码中，需要输出日志信息时，调用printf接口打印相关内容。

> [!CAUTION]注意
>printf接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC_DUMP=0的方式关闭打印功能。

函数原型：template <class... Args>；__aicore__ static __attribute__((noinline)) void printf( const __gm__ char* fmt, Args&&... args)；template <class... Args> 等6种重载

参数：fmt（输入）：格式控制字符串，包含两种类型的对象：普通字符和转换说明。<br><br>普通字符将原样不动地打印输出。<br>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型。<br>支持的转换类型包括：<br>%d、%ld、%lld、%i、%li、%lli：输出十进制数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t<br>%f、%F：输出浮点数，支持打印的数据类型：float、half、bfloat16_t<br>%x、%lx、%llx：输出十六进制整数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t、uint8_t、uint16_t、uint32_t、uint64_t<br>%s：输出字符串<br>%u、%lu、%llu：输出unsigned类型数据，支持打印的数据类型：uint8_t、uint16_t、uint32_t、uint64_t<br>%p：输出指针地址<br><br>注意：上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。；args（输入）：附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。

返回值：无

平台支持：Ascend 950PR/950DT、Atlas A3 训练/推理、Atlas A2 训练/推理系列支持，其他产品不支持。
约束：-   本接口不支持打印除换行符之外的其他转义字符。

-   SIMT场景的支持情况如下：

    - Ascend 950PR/Ascend 950DT：支持。


    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持。
支持 SIMT 与 SIMD/SIMT 混合编程场景。

## 固定 SIMT VF 声明与缓存范围

950PR/950DT 的固定 SIMT 调试 reference 具体给出以下声明；它补充前文摘要省略的 SIMT 调用形式，不替代其他 SIMD/主 Scalar 重载：

```cpp
template <class... Args>
__attribute__((always_inline)) inline __simt_callee__ void printf(const __gm__ char* fmt, Args&&... args);
```

本重载返回 void，在 SIMT VF 内使用，格式字符串位于 GM，头文件为 `utils/debug/asc_printf.h`。不能把 C 标准库 printf 的返回计数或前文 __aicore__ 重载修饰符自动套到该声明。格式转换与参数个数按前文 NPU 类型表匹配；CPU 域调试的 C/C++ 规则不证明 NPU 支持其全部格式。

调用会消耗 SIMT 栈；数据另缓存在 Global Memory。固定资料给出 `acl.json` 的 `simt_printf_fifo_size` 配置键、默认 2MB 和 1MB–64MB 范围，实际 JSON 数值编码需核对运行时，本文不生成未经核实的配置值。大量线程/次数和仿真打印会增加运行时间，增加 FIFO 也不能防止栈溢出。资源与缺日志的解释见 [SIMT 调试输出](../simt/debug_output_resource_cost.md)。

这些 SIMT 专项合同只由固定资料中的 950 产品范围支持，不把本卡其他 SIMD 平台的支持范围自动扩展为 SIMT 支持。与 [AICPU printf](aicpu_printf_reference_contract.md) 的头文件、函数修饰符和 Dump 容量不同；本卡未执行编译、实际打印或配置写入。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [Printf](../../apis/utils_api/tuning_interface/printf.md)
