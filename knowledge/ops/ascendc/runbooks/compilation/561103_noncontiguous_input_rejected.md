---
type: "Runbook"
title: "561103 — 非连续输入经 Contiguous 仍被拒"
description: "stride≠contiguous 的输入 tensor 传入自定义 aclnn 算子 GetWorkspaceSize 返回 561103，即使 aclnn 层已调用 l0op::Contiguous；具体拒绝点未定位，规避为上层 x.contiguous() 前置。"
tags: ["task.debugging", "topic.data_movement", "topic.implementation.api_layer"]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-16T10:29:17Z'
platforms: ["950"]
---

# 561103 — 非连续输入经 Contiguous 仍被拒

## 现象

去重算子（对标 `torch.unique`，ascend950，CANN 9.1.0）迭代三联调跑全量 C++ ST（Real 模式，121 用例）。其中 4 个迭代三新增的"非连续 tensor"用例全部失败，失败形态是 `aclnnBenchNoHarnessUniqueGetWorkspaceSize` 直接返回错误码 561103，kernel 根本没执行：

```
[I3-044] dtype=int32, ri=1, shape=[4,4], dedup=all_diff, special=none, sort=random
  strides=[8,1] (非连续)
  描述: 非连续 rank2 int32 stride=[8,1] ri=true
  [FAIL] GetWorkspaceSize: 561103

[I3-045] dtype=float16, ri=0, shape=[4,4], ... strides=[8,1] (非连续)
  [FAIL] GetWorkspaceSize: 561103

[I3-046] dtype=bfloat16, ri=1, shape=[4,4,4], ... strides=[32,8,1] (非连续)
  [FAIL] GetWorkspaceSize: 561103

[I3-047] dtype=float32, ri=0, shape=[4,4,4], ... strides=[32,8,1] (非连续)
  [FAIL] GetWorkspaceSize: 561103
```

测试侧用 `aclCreateTensor` 显式传非默认 stride（如 shape [4,4] 配 stride [8,1]）构造输入，物理存储因 stride 大于逻辑大小。其余 117 个连续 tensor 用例全部通过，说明算子本体功能正常。该算子的 aclnn 实现里本已按模板写了连续化处理：

```cpp
auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

即预期 `l0op::Contiguous` 为非连续输入生成连续副本——但 561103 仍然发生。

## 适用场景

- 工程形态：自定义算子 aclnn 接口（op_api）开发与 ST 测试，需要评估/支持非连续输入 tensor。
- API 路径：`aclnnXxxGetWorkspaceSize` 第一段接口；`l0op::Contiguous`；测试侧 `aclCreateTensor` 带 stride。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0；仅 950 有实测证据。
- 适用粒度：`operator_family`——任何按 aclnn 模板开发、依赖 l0op::Contiguous 处理非连续输入的自定义算子。
- 路线优先级：`safety_fallback`——当前以"文档化为已知限制 + 上层保证连续输入（如 PyTorch 适配层先 x.contiguous()）"规避；不要假设 l0op::Contiguous 一定能兜住非连续输入。
- 明确不适用/未验证场景：其他 CANN 版本 / 其他 SoC 未实测；经 aclnn 之外路径（GE 图模式）传入非连续 tensor 的行为未测；本卡不含 kernel 侧非连续支持方案。非连续张量经 OpParamDef 的 AutoContiguous()/IgnoreContiguous() 声明缺失而报 561000 的另一根因见 [ACLNN 561000 非连续张量需 AutoContiguous 卡](aclnn_561000_autocontiguous.md)；其他 561103 根因（[签名冲突](561103.md)、[vendor 未注册](561103_vendor_load_priority_not_registered.md)等）与本卡输入输出关系不同，按各自判别法区分。

## 判别方法（根因/解决配方）

**已证事实（单句归因）**：stride≠contiguous 的输入 tensor 让 `aclnnXxxGetWorkspaceSize` 返回 561103（ACLNN_ERR_INNER_NULLPTR），即使 aclnn 层已调用 `l0op::Contiguous`——拒绝发生在框架/l0 层，不是算子自己的 `CheckParams` 逻辑（自己的参数检查会返回 561002 类参数错误且仅在本层消息中体现）。

**判别步骤**：

1. 全量 ST 中精确定位失败子集：只有显式传非默认 stride 的用例失败、同 shape 同 dtype 的连续用例通过 → 输入格式问题而非数值问题。
2. 核对错误码段：561xxx 为 ACLNN 错误段，561103 为 ACLNN_ERR_INNER_NULLPTR。
3. 核对自己的 aclnn 代码是否已按模板调用 `l0op::Contiguous`——已调用仍报 561103，说明拒绝点在 Contiguous/框架执行器创建/L0 接口某一层，算子侧无可修代码。

**当前规避（verbatim，PyTorch 适配层前置连续化）**：

```cpp
static std::tuple<torch::Tensor, torch::Tensor> forward_npu(
    const torch::Tensor& x, bool return_inverse) {
    // 确保连续 (已知限制: l0op::Contiguous 对非连续返回 561103)
    auto x_contig = x.contiguous();
```

（即：上层框架在调用 aclnn 前自行把输入变成连续，绕过该拒绝点；C++ ST 侧则把这 4 个用例记为已知限制。）

**判别 561103 的分层入口**：本卡形态是"仅非连续输入用例失败 + 已调用 Contiguous"；若全部用例统一 561103，先查 vendor 注册（load_priority）；若同名内置算子签名冲突，见对应卡。

## 设计决策

- **为何选择上层连续化而不是继续追 561103 来源**：非连续支持是迭代三新增需求而非核心验收项，连续 tensor 的功能与精度已 100% 通过；在联调时间窗内把 4 个用例文档化为已知限制 + PyTorch 适配层 `x.contiguous()` 兜底，保主线交付。
- **当前方案属哪档路线**：`safety_fallback`——上层保证连续输入；算子原生非连续支持待根因澄清后另行实现。

## 预防

- 设计阶段就明确"非连续输入是否在验收范围"；若在，先用最小 demo（一个 `l0op::Contiguous` + 非连续 aclTensor）验证框架行为，再写实现。
- 适配层（torch adapter 等）入口一律 `x.contiguous()`，把非连续处理收敛到一层。

## 验证信号

段首来源说明：本卡结论来自该去重算子迭代三联调验证会话（ascend950，CANN 9.1.0）。

- 全量 ST 121 用例：非连续 4/4 返回 `[FAIL] GetWorkspaceSize: 561103`（stride [8,1] 与 [32,8,1]，int32/fp16/bf16/fp32 四种 dtype），其余 107 通过（来源平台：950；证据等级：tool_observed）。
- PyTorch ST 适配层前置 `x.contiguous()` 后 151/151 bit-exact 通过（其中含随机生成的各种输入）（来源平台：950；证据等级：tool_observed）。
- "aclnn 层已调用 l0op::Contiguous"为 op_api 源码读取事实（op_api/aclnn_bench_no_harness_unique.cpp）（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- 本卡不解释 561103 的产生机制（Contiguous/框架/L0 三者未区分），只锁定了"非连续输入 + 已调用 Contiguous 仍被拒"的输入输出关系。
- 其他 CANN 版本行为可能不同（未验证，推断可迁移需复现）。
- stride 模式仅覆盖 [8,1]（rank2）与 [32,8,1]（rank3）两类；其他 stride 形态（如广播式 stride=0、负 stride）未测。

**未决项**：

- **优先项：复现并追根因到机制层**——区分 `l0op::Contiguous`、框架执行器创建、L0 接口三者中哪一层返回 561103；可用的最小实验是直接调用 `l0op::Contiguous` 观察返回。
- 若确认为框架限制，评估 aclnn 层手动拷贝连续化的成本与正确性。
- 本算子最终以"非连续不支持"交付；GE 图模式下非连续输入行为未验证。
## 坏实践

- 假设"aclnn 模板里有 l0op::Contiguous 就自动支持非连续输入"——本轨迹证明该假设在 950/CANN 9.1.0 下不成立。
- 把 561103 用例直接从测试集删掉了事——应保留用例并标注已知限制，否则后续修复无回归锚点。

## 误诊

- 无显著误诊轮次；定位较快。注意 561103 的错误码含义（ACLNN_ERR_INNER_NULLPTR）在本轨迹中经 `op_errno.h` 核对；"非连续输入与 561103 的绑定关系"是实测事实。
