---
type: Guide
title: printf（AI CPU API）：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。
tags:
- task.lookup
- topic.implementation.api_layer.aicpu_api
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/ai_cpu_api/printf.md
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
- AICPU printf
---


# printf（AI CPU API）

该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。

函数原型：void printf(const char* fmt, ...)

参数：
- fmt（输入）：格式控制字符串，包含两种类型的对象：普通字符和转换说明。
普通字符将原样不动地打印输出。转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，，以转换说明结束，从而说明输出数据的类型。

约束：
- 该接口用于通过 `<<<...>>>` 启动的 AI CPU Kernel，并在异构编译场景使用；启动语法不修饰 printf 调用本身。
- 该接口不支持打印除换行符之外的其他转义字符。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu_api为Device链接libaicpu_api.a，通过--cce-aicpu-L指定libaicpu_api.a的库路径。

支持：Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品。不支持：Atlas 200I/500 A2 推理产品、Atlas 推理系列产品AI Core、Atlas 推理系列产品Vector Core、Atlas 训练系列产品。

## 格式转换、头文件与调用域

固定 950PR/950DT AICPU 调试 reference 使用 `#include "aicpu_api.h"`。其格式表如下；不能直接套用 SIMD/SIMT printf 的数据类型扩展：

| 转换说明 | 来源列出的数据类型与含义 |
|---|---|
| %d、%i | 十进制：bool、int8_t、int16_t、int32_t、int64_t |
| %f | 实数：float、half |
| %x | 十六进制：int8_t/int16_t/int32_t/int64_t 与对应 uint 类型 |
| %s | 字符串 |
| %u | unsigned 显示：bool、uint8_t、uint16_t、uint32_t、uint64_t |
| %p | 指针地址 |

普通字符原样输出，转换说明消耗对应附加参数，参数数量应与需要参数的转换标签相匹配，类型按目标实现支持确认。bool 出现在本来源的 %d/%i 和 %u 范围，不能仅因通用 Utils printf 表未列 bool 就抹去此 AICPU 合同；也不能从 SIMD/SIMT 的 bfloat16、%F 或长度修饰符支持反推本接口全部支持。来源没有给出完整格式语法解析器，不能将本表提升为 C 标准库的全功能保证。

`<<<...>>>` 描述 kernel 启动方式，printf 调用位于该 AICPU Kernel 中；它不是 `printf<<<...>>>`。异构编译、除换行外转义字符限制、每核全部 Dump 接口总量不超过来源所称 1M，以及 bisheng 手动链接/CMake 自动链接的既有约束仍适用。--cce-aicpu-laicpu_api 链接 libaicpu_api.a，--cce-aicpu-L 选择库目录；实际安装前缀不能固定为样例 root 路径。

该补充范围仅来自固定 950 资料；本卡其他平台的既有归档证据保留，不能用这一份资料单独证明新增细节在全部平台相同。返回 void，未执行 SDK 编译或打印验证；对缺少输出应同时检查执行路径与共享 Dump 上限，而非只根据屏幕日志判定 kernel 未执行。

## 相关

- [kernel 源码改动未生效因 binary 缓存](../../runbooks/compilation/kernel_binary_cache_printf_hash_toggle.md)（`exemplifies`）：“该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。”规定 printf 探针的输出上限；“用上述 PRINTF 对照实验——加 PRINTF 后输出变化即确诊缓存未刷新。”展示该接口作为 kernel 二进制缓存生效探针的具体用法。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [printf（AI CPU API）](../../apis/ai_cpu_api/printf.md)
