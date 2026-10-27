---
type: "Runbook"
title: 非连续 TilingKey 死代码：三层连续化拦载
description: "自定义算子即使打包了非连续 kernel 路径（TilingKey=2），op_def 的 .AutoContiguous()、L2 API 的 l0op::Contiguous() 与 tiling 硬编码 isContiguous=1 三层会让 kernel 永远只收到连续输入，非连续分支成为死代码；且\"非连续 ST 用例 PASS\"测到的是框架连续化路径，不构成自有路径的覆盖证据。"
tags: [task.debugging, topic.implementation.operator_registration, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-16T10:22:16Z'
platforms: ["950"]
---

# 非连续 TilingKey 死代码：三层连续化拦载

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上开发 BenchNoHarnessSwiGlu 算子的非连续输入路径：算子按设计打包了 TilingKey=1（连续）/TilingKey=2（非连续）双路径 kernel（安装包 `binary_info_config.json` 的 simplifiedKey 为 `d=*/p=*/1,2/1,2`，3 个 dtype binary 各含两个 TILING_PATH 变体），但验收发现 **TilingKey=2 分支从未被选中**——ST 全部用例走的都是连续路径。随后构造的非连续 ST 用例（转置 stride 输入）也全部 PASS，看似"非连续已支持"，实际是输入在进 kernel 前就被转成了连续。

排查时在代码里发现三层拦载：

1. `op_host/bench_no_harness_swi_glu_def.cpp`（算子定义）：

```cpp
        this->Input("input")                                   // 输入定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();                                 // 内存自动连续化
```

2. `op_api/aclnn_bench_no_harness_swi_glu.cpp`（L2 aclnn 接口）：调用 `l0op::Contiguous(input, ...)` 把输入复制为连续后再进 L0，输出侧 `l0op::ViewCopy(...)` 回写。

3. `op_host/arch35/bench_no_harness_swi_glu_tiling.cpp` line 218：`uint32_t isContiguous = 1;` 硬编码，TilingKey 恒为连续路径。

## 适用场景

- 工程形态：AscendC 自定义算子多 TilingKey（连续/非连续双路径）开发与验收覆盖。
- API 路径：op_def `Input().AutoContiguous()`、L2 `l0op::Contiguous/ViewCopy`（`aclnn_kernels/contiguous.h`）、Host Tiling 的 `InputIsView/GetInputStride/GetInputOffset`（`exe_graph/runtime/tiling_context.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 图引擎 + aclnn 框架下的一切双路径算子。
- 路线优先级：recommended_fastpath（做非连续路径前必查，否则写的是死代码）。
- 明确不适用 / 未验证场景：**有意采用框架统一连续化**的算子（不自带非连续 kernel，靠 AutoContiguous/Contiguous 保证正确性）——这是合法设计（也是 CANN 9.1.0 上推荐形态，见下文修复方向说明），本卡只针对"想触发自有非连续路径却发现不触发"的情形。

## 判别方法

按三层依次核查（均为只读检查，几分钟内完成）：

1. **op_def 层**：`grep -n "AutoContiguous" op_host/*_def.cpp`——输入定义带 `.AutoContiguous()` 则图引擎在进 kernel 前自动连续化。
2. **L2 层**：`grep -n "l0op::Contiguous\\|l0op::ViewCopy" op_api/aclnn_*.cpp`——L2 显式做连续化拷贝则 kernel 永远收到连续张量。
3. **Tiling 层**：查 tiling 里 `isContiguous` 是否硬编码常量（本例 line 218 `uint32_t isContiguous = 1;`）。
4. **旁证**：查安装包 `opp/vendors/<vendor>/op_impl/.../config/ascend950/binary_info_config.json` 的 simplifiedKey——双路径 binary 存在 + 运行时永远选中 path=1，即非连续分支为死代码。

## 根因

三层机制都在"保护" kernel 不见非连续输入：图引擎的 `AutoContiguous()`（注册级开关）、L2 aclnn 的 `l0op::Contiguous()`（接口级拷贝）、tiling 的硬编码（分支级常量）。任何一层生效，kernel 收到的都是连续数据——TilingKey=2 的 kernel binary 虽然编译打包了，但永远不会被 tiling 选中。三层叠加时，只改其中一层（例如只把 tiling 的硬编码改成动态判定）不会有任何效果。

## 解决配方（修复方向及其边界）

要启用自有非连续路径，直觉方案是三层**同时**放开（op_def 移除 `AutoContiguous()`、L2 移除 `l0op::Contiguous()` 调用、Host Tiling 用 `TilingContext` 的视图信息动态判定并设置 TilingKey）。op_def 侧的放开写法（verbatim 取自该开发线写入事件）：

```cpp
        this->Input("input")                                   // 输入定义
            .ParamType(REQUIRED)                               // 必选输入
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            // 移除 AutoContiguous, 由 Host Tiling 动态检测连续性 (TilingKey=1/2)
```

**但必须注意：该"三层放开 + tiling 动态检测"路线在 CANN 9.1.0 上已被同一开发线的后续实测否定**——`gert::TilingContext::GetInputStride(0)` 返回未填充的 `gert::Stride`（`GetDimNum()==0`、全 0），tiling 层根本拿不到输入 stride，动态判定失去输入信号，非连续输入被当连续按 C-order 读，12 条非连续用例 MERE 0.09~0.20 全错。完整失败-回退闭环见 [非连续输入结果错：tiling 无 stride 须 L2 Contiguous 卡](../precision/noncontig_tiling_no_stride_l2_contiguous.md)。因此在 CANN 9.1.0 上，**保留三层框架连续化是已验证的正确形态**；自有非连续路径须等框架提供有效 stride 信号（或另寻 stride 来源，如 L2 层手动读 aclTensor 描述，未验证）后再评估。

## 设计决策

- 为什么必须三层同改才能启用自有路径：任一层残留都会把输入重新连续化，tiling 的动态判定永远得到"连续"。
- 为什么 tiling 判定拟用 `InputIsView/GetInputStride` 而不是比较 storageShape：视图 shape 与存储 shape 相同仍可能是非连续视图（stride 不匹配），`InputIsView` + 逐维 stride 与期望值（`expectedStride *= viewShape.GetDim(i)` 累乘）比对才是权威判定——但该方案依赖 GetInputStride 返回有效数据，CANN 9.1.0 不满足（见上）。
- 框架统一连续化（保留三层）与自有非连续路径（三层放开）是二选一的设计：前者省 kernel 工作量但多一次拷贝，后者省拷贝但要自己处理 stride——CANN 9.1.0 上 stride 不可得，前者是当前唯一可行动路线。
- 当前方案路线档位：recommended_fastpath（针对"确实要实现非连续路径"的算子，先查三层开关与 stride 可用性再动工）。

## 预防

- 立项"非连续 kernel 路径"时，第一件事跑上面 4 步判别，确认三层开关状态与 `GetInputStride` 是否返回有效数据，再写 kernel 代码。
- 验收非连续覆盖时，不能只看"非连续输入的 ST 用例 PASS"——必须确认 TilingKey=2 binary 真的被选中（tiling 日志/分支计数），否则测的是框架连续化路径。

## 验证信号

以下信号来自该算子验收与后续开发会话（同一条开发线的两个阶段）。

- 触发：验收记录 TilingKey=2 分支 0 覆盖（来源平台：950；证据等级：tool_observed）。
- 探针：`op_def` 读到 `.AutoContiguous()`（line 36/43）；`aclnn_*.cpp` 读到 `l0op::Contiguous` 调用（L2 参数检查/Contiguous/ViewCopy 处理）；`tiling.cpp` 读到 line 218 `uint32_t isContiguous = 1;` 硬编码；安装包 `binary_info_config.json` 显示 `d=*/p=*/1,2/1,2` 双路径 binary 存在（来源平台：950；证据等级：tool_observed——逐文件读取原文）。
- 观察：非连续 ST 用例（转置 stride 构造）在三层拦载下全部 PASS——走的是框架连续化路径，TilingKey=2 未被选中（来源平台：950；证据等级：tool_observed）。
- 后续反证：三层放开 + tiling 动态检测后，12/12 条非连续用例 Real 模式 FAIL（MERE 0.09~0.20），`GetInputStride(0)` 返回未填充 Stride（GetDimNum()==0、全 0），恢复 L2 `l0op::Contiguous()` + op_def `AutoContiguous()` 后 115/115 全过——自有路径在 CANN 9.1.0 不可达（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "三层放开 + tiling 动态检测"的完整闭环未成立：编译与 ST 验证在后续会话完成且结果为失败-回退（见上），本卡不作为"已验证修复"引用该配方。
- `InputIsView/GetInputStride/GetInputOffset` 的语义以 CANN 9.1.0 头文件为准；其他版本 API 形态未验证。
- 保留框架连续化（不实现自有路径）是合法替代——本卡不是"必须去掉 AutoContiguous"的建议。
- 其他平台未验证，推断可迁移。

**未决项**：

- 【优先项】复现并追根因：`gert::TilingContext::GetInputStride` 在 CANN 9.1.0 返回空 Stride 的机制层原因，以及更高版本是否已填充——决定自有非连续路径何时可重新评估。
- TilingKey=2 kernel 路径（workspace 连续化拷贝 + 多核 `SyncAll` 同步方案）的正确性与性能验证在 CANN 9.1.0 上不可达，挂起。
## 坏实践

- 只把 tiling 的 `isContiguous` 从 1 改成"动态计算"就以为启用了非连续路径——上层两层照拦，改动无效。
- 用"非连续输入用例 PASS"当非连续路径已验证——测到的可能只是 `l0op::Contiguous` 的正确性。
- 在未确认 tiling 层能拿到 stride 之前投入 kernel 双路径开发——整条路径返工。

## 误诊

- 非连续覆盖缺失曾一度被当作"用例没写够"，实际是路径被三层拦载（本卡即该误诊的修正）。
