---
type: "Runbook"
title: 非连续输入结果错：tiling 无 stride 须 L2 Contiguous
description: "CANN 9.1.0 的 gert::TilingContext::GetInputStride 返回未填充 Stride，tiling 层检测不到非连续输入，非连续用例被当连续按 C-order 读致 MERE 0.09~0.2 全错；正确处置是在 L2 API 层恢复 l0op::Contiguous() + op_def AutoContiguous()，kernel 内非连续路径（TilingKey 分流）当前版本不可用。"
tags: [task.debugging, topic.precision, topic.data_movement.layout_transform, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-16T10:22:16Z'
platforms: ["950"]
---

# 非连续输入结果错：tiling 无 stride 须 L2 Contiguous

## 现象

在 Ascend950PR（CANN 9.1.0）上联调 BenchNoHarnessSwiGlu：此前一轮已用 stride 有效性回退修掉"全量 507035 死锁"（所有输入被误判非连续触发 TilingKey=2），重跑 Real 全量 115 条后剩 12 条失败——**恰好是全部 12 条非连续（2D/3D transpose）用例**，且误差巨大（远超舍入量级），报错原文：

```
[Real] noncont_fp16_2d_trans_[8,4]_dim-1
  shape=[8,4], dtype=float16, dim=-1
  [FAIL] MERE=1.65e-01 (>= 9.77e-04), MARE=3.54e-01 (>= 9.77e-03)
[Real] noncont_bf16_2d_trans_[4,8]_dim0
  shape=[4,8], dtype=bfloat16, dim=0
  [FAIL] MERE=1.81e-01 (>= 7.81e-03), MARE=6.10e-01 (>= 7.81e-02)
...（共 12 条 noncont_* 用例全 FAIL，MERE 0.09~0.20）
```

其余 103 条连续输入用例全部 PASS。当时的代码状态：op_def 输入已移除 `AutoContiguous()`、L2 API 已移除 `l0op::Contiguous()`（意图是让非连续输入直达 kernel，由 TilingKey=2 的 kernel 内拷贝路径处理），但 DetectContiguity 的 stride 回退使所有输入都走 TilingKey=1（连续路径，按 C-order 读 GM）。

## 适用场景

- 工程形态：aclnn 自定义算子开发，输入可能非连续（stride 传入 aclCreateTensor / transpose view）的场景。
- API 路径：op_def `Input("input").AutoContiguous()`；L2 API `l0op::Contiguous(input, executor)`；`gert::TilingContext::GetInputStride`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 上自定义算子非连续输入的处置层选择。
- 路线优先级：recommended_fastpath（L2 API 层 Contiguous 为当前版本唯一可行动路线）。
- 明确不适用 / 未验证场景：CANN 更高版本若 GetInputStride 已返回有效 stride，kernel 内非连续路径（TilingKey 分流）可重新评估（未验证）；纯图模式（GE）下 AutoContiguous 的行为未单独验证。

## 根因

CANN 9.1.0 的 tiling 上下文**根本拿不到输入的逻辑 stride**：`GetInputStride(0)` 返回未填充的 `gert::Stride`（`GetDimNum()==0`、全 0）。因此"在 Host Tiling 检测非连续并分流到 kernel 内拷贝路径"的整个方案失去输入信号——非连续输入与连续输入在 tiling 层不可区分，只能全部按连续处理（TilingKey=1 按 C-order 线性读 GM）。对 transpose 生成的非连续输入，C-order 线性读到的元素排列与逻辑视图不同，输出随之错乱，产生 0.1 量级的 MERE/MARE（远超 dtype 舍入 1e-3~1e-2 阈值）。

结论：非连续输入的连续化必须放在**框架可感知的层**——L2 API 的 `l0op::Contiguous()`（或 op_def 的 `AutoContiguous()`）——而不是依赖 tiling 层 stride 检测的 kernel 内路径。

## 解决配方

修复前（`op_api/aclnn_bench_no_harness_swi_glu.cpp`，verbatim，非连续输入直达 L0 算子）：

```cpp
    // 迭代三: 移除 l0op::Contiguous(), 由 Host Tiling 动态检测连续性
    // 非连续输入通过 TilingKey=2 路径在 Kernel 中拷贝到 workspace
    // 调用 L0 算子 (input 可能非连续, Kernel 内部处理)
    const aclTensor* opResult = l0op::BenchNoHarnessSwiGlu(input, dim, uniqueExecutor.get());
```

修复后（verbatim，恢复 L2 API 层连续化）：

```cpp
    // 迭代三: 恢复 l0op::Contiguous() 处理非连续输入
    // 原因: CANN 9.1.0 的 gert::TilingContext::GetInputStride() 不返回有效 stride 信息
    //        (dim_num=0, 全 0), 无法在 Tiling 层检测非连续性。
    //        保留 DetectContiguity() 和 TilingKey=2 代码路径供未来框架支持后启用。
    //        Contiguous() 后输入必为连续, DetectContiguity() 正确返回 isContiguous=1。
    const aclTensor* contiguousInput = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 算子 (input 已通过 Contiguous 转为连续)
    const aclTensor* opResult = l0op::BenchNoHarnessSwiGlu(contiguousInput, dim, uniqueExecutor.get());
```

同时恢复 op_def 输入侧（`op_host/bench_no_harness_swi_glu_def.cpp`，verbatim）：

```cpp
        this->Input("input")                                   // 输入定义
            .ParamType(REQUIRED)                               // 必选输入
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();  // 迭代三: 恢复 AutoContiguous (框架自动处理非连续输入)
```

（修复前该处为无 AutoContiguous 的裸链 + 注释"移除 AutoContiguous, 由 Host Tiling 动态检测连续性"。）输出的 `AutoContiguous()` 与 `l0op::ViewCopy()`（连续 kernel 输出写回可能非连续的用户输出）保持不变。

## 设计决策

- 备选一（采用）：恢复 `l0op::Contiguous()` + `AutoContiguous()`，TilingKey=2 kernel 路径保留为死代码。理由：与已被 115/115 验证的形态一致，风险最低；stride API 一旦可用可再启用。
- 备选二（放弃）：继续在 tiling 层检测非连续——被 plog 实测否定（stride 恒 0，无信号可用）。
- 备选三（放弃，记录为未尝试）：L2 API 层手动读 `aclTensor` 的 stride 描述（如 `aclGetTensorStride` 一类接口）在调 L0 前自行分流——未验证该接口在 9.1.0 的可用性，仅作为后续建议。
- AutoContiguous 与 Contiguous() 双恢复：前者作用于图编译路径、后者作用于 aclnn 直调路径；ST 测试走 aclnn 直调，两者都恢复最稳。

## 预防

- CANN 9.1.0 上写自定义算子：非连续输入默认按"框架层连续化"设计（AutoContiguous/Contiguous()），不要假设 tiling 上下文能拿到 stride。
- 若确实要验证 tiling 层 stride 可用性，先写最小 tiling 探针（OP_LOGI 打印 GetDimNum/GetStride）再决定架构，避免整条 kernel 双路径返工。
- 非连续用例的失败画像（MERE 0.1 量级、仅 noncont 用例错）与 dtype 舍入问题（MERE 1e-3~1e-2 量级）可分辨：前者是读错数据，后者是精度模型差异。

## 坏实践

- 移除 `AutoContiguous()`/`Contiguous()` 后仅凭 Mock（CPU golden）测试通过就认为非连续处理完成——Mock 不经过 tiling/kernel，本轨迹 Mock 113/113 全过而 Real 全崩，非连续路径必须 Real 模式验证。

## 验证信号

来源：该算子非连续路径联调验证会话，12 条非连续用例失败 → 恢复 Contiguous()/AutoContiguous → 全量回归。

- 修复前：12/12 条 noncont 用例 FAIL（MERE 1.65e-01~9.34e-02、MARE 3.54e-01~6.54e-01），其余 103 条 PASS（来源平台：950；证据等级：tool_observed）。
- 根因探针：`gert::TilingContext::GetInputStride(0)` 返回未填充 `gert::Stride`（`GetDimNum()==0`、全 0），tiling 层无非连续判定信号（来源平台：950；证据等级：tool_observed）。
- 修复后 Real 全量：Exit 0，测试报告`总计: 115 通过: 115 失败: 0 跳过: 0`（2 条 inf-only 用例 WARN 属预期）（来源平台：950；证据等级：tool_observed）。
- 修复后非连续用例精度：fp16 MERE 8.24e-06~2.00e-05、fp32 0~1.85e-09、bf16 1.16e-04~1.80e-04，全部远低于阈值（如 `noncont_fp16_2d_trans_[8,4]_dim-1 MERE=8.24e-06, MARE=4.43e-05 (threshold=9.77e-04)`）（来源平台：950；证据等级：tool_observed）。
- Mock 回归 113/113、UT 46/46 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论绑定 CANN 9.1.0 的 stride API 行为；更高版本若填充 stride，kernel 内非连续路径（更省一次 host 侧拷贝）值得重评（未验证，推断可迁移）。
- `l0op::Contiguous()` 对已是连续的输入为 no-op 直通（本轨迹 103 条连续用例回归无性能异常）；其开销代价未单独计量。
- 未验证场景：广播维（stride=0）输入、负 stride 输入的 Contiguous 行为。
- 其他平台未验证，推断可迁移。

**未决项**：

- 【优先项】复现并追根因：`GetInputStride` 返回空 Stride 的机制层原因（框架未填充 or 接口形态不符），并确认更高 CANN 版本是否已修复——决定 kernel 内非连续路径的可行性窗口。
- 在框架支持 stride 后对比"L2 API Contiguous"与"kernel 内拷贝（TilingKey=2）"两条路线的成对相对开销，确定是否有必要启用后者。
- 评估 L2 API 层手动读取 aclTensor stride 描述做分流的可行性（未尝试）。
## 误诊

- 无（plog stride 探针直接定位 tiling 层无信号，修复前后全量对照闭环）。
