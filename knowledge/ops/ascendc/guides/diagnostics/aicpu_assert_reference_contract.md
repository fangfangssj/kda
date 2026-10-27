---
type: Guide
title: assert（AI CPU API）：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。该接口实现AI CPU算子Kernel调试场景下的assert断言功能。 算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。
tags:
- task.lookup
- topic.implementation.api_layer.aicpu_api
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/ai_cpu_api/assert.md
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
- AICPU assert
---


# assert（AI CPU API）

该接口实现AI CPU算子Kernel调试场景下的assert断言功能。 算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。

参数：
- expr（输入）：assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。

约束：
- 该接口用于通过 `<<<...>>>` 启动的 AI CPU Kernel，并在异构编译场景使用；启动语法不修饰 assert 宏本身。
- kernel开发不要包含系统的assert.h，会导致宏定义冲突。
- assert接口调用形式与C语言一致，不需要使用AscendC命名空间。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu_api为Device链接libaicpu_api.a，通过--cce-aicpu-L指定libaicpu_api.a的库路径。

支持：Ascend 950PR/Ascend 950DT、Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品。不支持：Atlas 200I/500 A2 推理产品、Atlas 推理系列产品AI Core、Atlas 推理系列产品Vector Core、Atlas 训练系列产品。

## 断言形式与 AICPU 编译合同

固定 950PR/950DT reference 使用 `#include "aicpu_api.h"`，调用形式为 `assert(expr)`，与 C 语言断言调用形式一致，不需要 AscendC 命名空间。expr 为 true 时继续执行，为 false 时终止并输出条件、触发文件及行号信息；返回值为无。

`<<<...>>>` 是包含断言的 AICPU Kernel 的启动形式，不是对 assert 宏执行模板启动。固定资料限定异构编译；不要将 [SIMD/SIMT assert](../../apis/utils_api/tuning_interface/assert.md) 的融合编译与头文件合同直接套入 AICPU。AICPU 开发不包含系统 assert.h，以免发生宏冲突。

既有每核所有 Dump 接口合计 1M 上限同样覆盖此断言输出，超过时可能不打印，不能由缺少断言文字推出条件一定为真。bisheng 的静态库链接与 CMake 行为按前文执行；相关 [AICPU printf](aicpu_printf_reference_contract.md) 与 assert 共用 Dump 预算，不是各自获得 1M。

来源示例的文件路径、行号和条件只是输出形态，不是固定错误位置。本补充来自 950 固定资料，未通过它扩展其他产品的断言能力，也未运行故障断言或编译测试。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [assert（AI CPU API）](../../apis/ai_cpu_api/assert.md)
