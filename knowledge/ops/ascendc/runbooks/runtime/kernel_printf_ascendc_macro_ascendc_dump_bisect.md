---
type: "Runbook"
title: kernel 侧打印定位崩溃：AscendC::PRINTF 宏
description: "NPU kernel 无源码级调试器，用 AscendC::PRINTF 宏（大写、命名空间限定、无需单独 include）埋点 + ASCENDC_DUMP=1 运行输出；配逐段禁用二分定位异步队列故障。"
tags: [task.debugging, topic.implementation.debugging, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# kernel 侧打印定位崩溃：AscendC::PRINTF 宏


## 现象

在 AscendC kernel（`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h` 的 Phase 2 主循环）里加运行时打印来定位一个稳定设备崩溃（ST 用例 G6 `aclrtSynchronizeStream 失败`，无任何 kernel 侧报错）。依次尝试了四种写法，前三种全部编译失败，报错原文各不相同：

1. `AscendC::Printf(...)` →
   ```
   thnn_fused_gru_cell_grad_group.h:269:18: error: no member named 'Printf' in namespace 'AscendC'; did you mean 'printf'?
   ```
2. `Printf(...)`（去掉命名空间）→
   ```
   thnn_fused_gru_cell_grad_group.h:269:9: error: use of undeclared identifier 'Printf'
   ```
3. 按工程内指南加 `#include "kernel_printf.h"` 并用 `PRINTF(...)` →
   ```
   thnn_fused_gru_cell_grad_group.h:48:10: fatal error: 'kernel_printf.h' file not found
   ```
   去掉 include 后裸 `PRINTF(...)` →
   ```
   thnn_fused_gru_cell_grad_group.h:296:9: error: use of undeclared identifier 'PRINTF'; did you mean 'AscendC::PRINTF'?
   ```
4. `AscendC::PRINTF(...)` → 编译通过。

之后用 `ASCENDC_DUMP=1` 运行 ST 才能真正看到输出；不加该环境变量时 printf 内容不落盘。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke，AscendC kernel 上板联调）——kernel 内无源码级调试器，需要观测核内执行进度/参数值时的标准手段。
- API 路径：AscendC kernel 侧 `PRINTF` 宏（`kernel_operator.h` 传递引入）；运行时开关 `ASCENDC_DUMP=1`。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.0.0-beta.2（宏定义位于 `/home/<user>/Ascend/cann-9.0.0-beta.2/x86_64-linux/asc/impl/basic_api/kernel_operator_dump_tensor_intf_impl.h:350` 的 `#define PRINTF(...)`，经 `kernel_operator.h` 的包含链可用，无需也不能 include `kernel_printf.h`）。与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（CANN 基础 API 行为，跨算子；跨算子可用性为推断——仅在单一算子 kernel 内实测）。
- 路线优先级：recommended_fastpath（该 CANN 版本下 kernel 侧打印的唯一可用形式）。
- 明确不适用/未验证：其他 CANN 版本宏名可能不同（工程内指南称"部分版本也可用小写 printf"）；格式符精度/宽度修饰符不支持等细节未在本 trace 实测（仅文档记载，见坏实践）。

## 判别方法

- 编译错误即判别：报 `no member named 'Printf' in namespace 'AscendC'` / `use of undeclared identifier 'Printf'` / `use of undeclared identifier 'PRINTF'` 都指向同一个修复：`AscendC::PRINTF`（大写宏 + 命名空间限定）。
- 找宏本体：`grep -rn "define PRINTF" /home/<user>/Ascend/cann-<ver>/` → 定位到 `kernel_operator_dump_tensor_intf_impl.h`，确认它已在 `kernel_operator.h` 的包含链里，因此**不要**再单独 include。
- 运行时无输出时检查是否设置了 `ASCENDC_DUMP=1`。

## 解决配方

正确写法（可直接套用的埋点模板，逐字取自本 trace 的调试代码）：

```cpp
        AscendC::PRINTF("P2 blk=%d before ReduceSum aLenUb=%d rGroup=%d\n",
            static_cast<int>(blockIdx), static_cast<int>(aLenUb), static_cast<int>(rGroupCnt_));
        // ... 被观测的操作 ...
        AscendC::PRINTF("P2 blk=%d after ReduceSum\n", static_cast<int>(blockIdx));
```

运行命令：

```bash
cd operators/ThnnFusedGruCellGrad_package/tests/aclnn && ASCENDC_DUMP=1 ./run.sh 2>&1 | grep -E "P2 blk|G6|Branch-1 ST summary"
```

本例观测到的输出形态（崩溃 case）：

```
P2 blk=0 loop=0 aOff=0 aLen=136
P2 blk=0 before ReduceSum aLenUb=136 rGroup=22
P2 blk=0 after ReduceSum
P2 blk=1 loop=1 aOff=136 aLen=136
P2 blk=1 before ReduceSum aLenUb=136 rGroup=22
P2 blk=1 after ReduceSum
P2 blk=2 loop=2 aOff=272 aLen=136
P2 blk=2 before ReduceSum aLenUb=136 rGroup=22
P2 blk=2 after ReduceSum
P2 blk=3 loop=3 aOff=408 aLen=136
P2 blk=3 before ReduceSum aLenUb=136 rGroup=22
（blk=3 无 after——崩溃点附近）
```

## 设计决策

- 为什么用 PRINTF 而不是别的：NPU kernel 侧没有源码级调试器；host 侧日志对该故障完全静默（`aclrtSynchronizeStream 失败`，无错误码）；PRINTF 是唯一能拿到核内执行进度的低成本手段。
- 为什么最终不依赖 PRINTF 定结论：PRINTF 会插入同步/改变时序（Heisenbug），且 MTE2/MTE3 是异步队列——"打印停在 X"只能说明"X 附近的某异步操作出事"，不能说明 X 本身崩溃。本例 PRINTF 曾把嫌疑指向 ReduceSum（blk=3 before 后无 after），而逐段禁用二分最终证明崩溃在之后的 MTE3 拷出。**结论：PRINTF 缩小范围，二分锁定根因。**
- 当前方案属 recommended_fastpath；备选（GM 标志位写点）因需要额外 GM buffer 未采用。

## 系统信息

- CANN：9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`）；目标 ascend950。
- 宏定义位置：`x86_64-linux/asc/impl/basic_api/kernel_operator_dump_tensor_intf_impl.h:350` `#define PRINTF(...)`（多行宏，带续行符）。
- 埋点文件：`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h`（Phase 2 主循环内）；调试后 PRINTF 全部移除。
- 工程内参考：`.opencode/skills/ascendc-kernel-printf-dump/references/printf-guide.md`（记载 `PRINTF` 宏、格式符限制——仅 `%d/%u/%x/%p/%f/%s`，不支持精度/宽度修饰符、`%e/%g/%lf`；本 trace 未逐一实测这些格式符限制，仅文档记载）。

## 完整代码

调试埋点的完整使用单元（插入位置：Phase 2 每个 A chunk 的 ReduceSum 前后；含崩溃定位所需的块号与形状参数）：

```cpp
        AscendC::PRINTF("P2 blk=%d loop=%d aOff=%d aLen=%d\n",
            static_cast<int>(blockIdx), static_cast<int>(aLoop),
            static_cast<int>(aOff), static_cast<int>(aLen));
        // CopyIn workspace + MTE2_V 同步 ...
        AscendC::PRINTF("P2 blk=%d before ReduceSum aLenUb=%d rGroup=%d\n",
            static_cast<int>(blockIdx), static_cast<int>(aLenUb), static_cast<int>(rGroupCnt_));
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
            Base::cacheBuf_.template Get<float>()[0],
            Base::preRegion_.template Get<float>()[static_cast<uint32_t>(Base::biasElemOff_)],
            Base::preRegion_.template Get<uint8_t>()[static_cast<uint32_t>(Base::tailByteOff_)],
            srcShape, /*srcInnerPad=*/true);
        AscendC::PRINTF("P2 blk=%d after ReduceSum\n", static_cast<int>(blockIdx));
```

复现命令：

```bash
# 构建（确认 AscendC::PRINTF 编译通过）
( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package ) 2>&1 | grep -cE "SUCCESS"
## 带 dump 运行
cd operators/ThnnFusedGruCellGrad_package/tests/aclnn && ASCENDC_DUMP=1 ./run.sh 2>&1 | grep -E "P2 blk" | head -30
```

## 预防

- 记住三要素：**大写宏 + AscendC:: 前缀 + ASCENDC_DUMP=1**；工程内旧指南写的 `#include "kernel_printf.h"` 在 CANN 9.0.0-beta.2 不存在该头，不要照抄。
- 埋点参数带 `blk`（GetBlockIdx）与关键形状量（aOff/aLen/rGroup），多核并发时输出会交错，没有块号无法对齐。
- 移除调试宏用编辑工具整段删，不要 `sed '/AscendC::PRINTF/d'` 按行删——宏调用是多行语句（定义带续行），单行删除会留下孤立续行导致编译错（本 trace 实测踩过）。

## 验证信号

本卡验证来自该 group kernel 崩溃定位轮（构建与 ST 运行输出）：

- `AscendC::Printf` / `Printf` / 裸 `PRINTF` / `#include "kernel_printf.h"` 四种写法分别报上文编译错误（来源平台：950；证据等级：tool_observed）。
- `sed 's/\bPRINTF(/AscendC::PRINTF(/g'` 后构建输出 `SUCCESS`（grep -cE "SUCCESS" → 1）（来源平台：950；证据等级：tool_observed）。
- `ASCENDC_DUMP=1 ./run.sh` 输出上述 `P2 blk=N ...` trace，观测到 blk=3 打印 before 后无 after（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 时序敏感故障（异步 MTE 队列、跨核竞态）中 PRINTF 的结论可能是错位的——本例 PRINTF 指向 ReduceSum 而真因在其后的 MTE3 拷出；PRINTF 只能缩小范围。
- 宏名/可用性验证于 CANN 9.0.0-beta.2 / ascend950；其他版本未验证（推断小写 `printf` 在部分版本可用，未验证）。
- 格式符限制（不支持 `%e/%g/%lf`、精度/宽度修饰符）来自工程内文档记载，本 trace 未实测。

## 坏实践

- 照搬 GM→UB 时代遗留的 `#include "kernel_printf.h"` 写法（该头在本版本 CANN 安装中不存在，fatal error）。
- 用 `sed` 按行删除 `AscendC::PRINTF` 调用（多行宏留孤立续行 → 编译错）。
- 把 PRINTF 输出当成崩溃点的直接证据（异步队列时序错位）。

## 误诊

- 无（本卡为 API 用法闭环；关联故障的误诊见[b16 DataCopyPad UB→GM 源非 32B 对齐卡](../compilation/b16_datacopypad_ub_gm_32b_fault.md)的误诊节）。


