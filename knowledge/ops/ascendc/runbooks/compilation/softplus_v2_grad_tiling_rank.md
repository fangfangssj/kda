---
type: "Runbook"
title: "softplus_v2_grad tiling rank 校验拒标量广播"
description: "softplus_v2_grad 自定义算子（arch35/Ascend950PR）的 Host tiling GetShapeAttrsInfo 在 PadAndSqueeze 之前插入 raw rank 一致性校验（raw_input_shapes_[0].size()!=raw_input_shapes_[1].size()），对 grad_output 标量(rank=0) vs self [2,3] 等合法 PyTorch 风格广播输入直接 GRAPH_FAILED；与同算子 InferShape 的 numpy 广播实现及 aclnn API broadcast 契约自相矛盾，且 PadAndSqueeze 本身已左填充 1 处理不同 rank，该校验冗余。判别：跑含标量-vs-tensor 用例的 torch adapter 测试集，命中 rank(gradOutput)!=rank(self) 报错即为本根因。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T15:37:56Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# softplus_v2_grad tiling rank 校验拒标量广播

## 现象

在 Ascend950PR（arch35 / dav-3510）上为 `softplus_v2_grad` 自定义算子做 bug 定位。清理 vendor 目录、`bash build.sh --soc=ascend950` 编译安装、构建 torch adapter 后，运行 `tests/st/torch/test.py --lib ./build/libtorch_adapter.so` 复现：总计 581 个用例，通过 463，失败 118，跳过 0。其中最先跳出来的 3 个 `[ERROR]` 用例是 L0 模板「标量 rank=0 vs [2,3]」三 dtype：

```
[16/581] [ERROR] 标量 rank=0 vs [2,3] [float32]: softplus_v2_grad: rank(gradOutput)=0 != rank(self)=2
[17/581] [ERROR] 标量 rank=0 vs [2,3] [float16]: softplus_v2_grad: rank(gradOutput)=0 != rank(self)=2
[18/581] [ERROR] 标量 rank=0 vs [2,3] [bfloat16]: softplus_v2_grad: rank(gradOutput)=0 != rank(self)=2
```

报错来自算子自身（`softplus_v2_grad:` 前缀 + `OP_LOGE` 打印），不是框架侧。报错文本直接点出 `rank(gradOutput)=0 != rank(self)=2`——即 grad_output 是 rank=0 标量、self 是 rank=2 的 `[2,3]`，算子在 tiling 阶段就以 rank 不一致为由返回 `GRAPH_FAILED`。这 3 个用例在测试套件里作为**功能用例（L0）**登记，即测试期望它通过 PyTorch 风格广播正常工作。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（自定义算子，op_kernel + op_host + op_api + tests/st/torch adapter），kernel 直调路径经 torch adapter 调 `torch.ops.softplus_v2_grad.forward`。
- **API 路径**：`SoftplusV2Grad` Host tiling `GetShapeAttrsInfo`（`op_host/arch35/softplus_v2_grad_tiling_arch35.cpp`）；同算子 `InferShape`（`op_host/softplus_v2_grad_infershape.cpp`）已按 numpy 广播右对齐前缀补 1。
- **验证过的平台与 CANN 版本**：Ascend950PR（`--soc=ascend950` 编译 + NPU 实跑复现报错）。CANN 版本未在 trace 中显式记录。
- **适用粒度（transfer_scope）**：`operator_family`——任何在 Host tiling 阶段对双输入做 raw rank 一致性校验、而 InferShape/API 契约要求 numpy 广播的逐元素/二元算子都可能踩同一坑。
- **路线优先级**：`recommended_fastpath`——移除 tiling 中冗余的 raw rank 一致性校验、依赖 PadAndSqueeze 的左填充 1 机制处理不同 rank（本 trace 仅诊断到这一方向，未应用/未验证）。
- **不适用或未验证场景**：a3/a2 未验证（推断可迁移，见 `## 失效边界`）；旧 CANN TBE 实现按其历史行为确实对 rank 不同返回 GRAPH_FAILED，那条**历史限制**不适用本卡修复方向。

## 判别方法

- **触发（tool_observed 失败）**：调用 `softplus_v2_grad` 时 grad_output 与 self 的 raw rank 不同（典型：一方为 rank=0 标量、另一方为 rank≥1 张量），或更一般地任何右对齐后可广播但 raw rank 不同的双输入。torch adapter 测试集在 L0 模板「标量 rank=0 vs [2,3]」三 dtype 上均报 `[ERROR] softplus_v2_grad: rank(gradOutput)=0 != rank(self)=2`。
- **探针（可执行检查）**：算子仓编译安装后，用 torch adapter 跑含标量-vs-tensor 广播用例的测试集并过滤算子侧报错：

  ```bash
  cd <workspace>/tests/st/torch && python3 test.py --lib ./build/libtorch_adapter.so 2>&1 \
    | grep -E "\[ERROR\] 标量 rank=0"
  ```

  若命中 `softplus_v2_grad: rank(gradOutput)=0 != rank(self)=2`，即为本校验触发。进一步定位根因点：`grep -n "raw_input_shapes_\[0\].size() != raw_input_shapes_\[1\].size()" op_host/arch35/softplus_v2_grad_tiling_arch35.cpp`，命中行即为冗余校验插入点（trace 中为第 231-234 行）。
- **观测（artifact_verified 代码取证）**：`GetShapeAttrsInfo` 中 raw rank 一致性校验紧跟在 dtype 校验之后、`PadAndSqueeze` 调用之前（verbatim 自 file-read 事件 seq 40，行号对应原文件）：

  ```cpp
  // rank 一致性校验（spec shape_constraints.global_constraints）
  OP_CHECK_IF(raw_input_shapes_[0].size() != raw_input_shapes_[1].size(),
      OP_LOGE(ctx_->GetNodeName(), "rank(grad_output) != rank(self)"),
      return ge::GRAPH_FAILED);

  PadAndSqueeze(raw_input_shapes_, raw_output_shapes_,
                max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
  ```

  同函数内 `PadAndSqueeze` 的 `pad` lambda 已对短 rank 左填充 1（verbatim 自同次 file-read）：

  ```cpp
  auto pad = [&](const std::vector<int64_t>& s) {
      std::vector<int64_t> p;
      p.assign(max_rank - (int64_t)s.size(), 1);
      p.insert(p.end(), s.begin(), s.end());
      return p;
  };
  ```

  即 tiling 流程本就具备处理不同 rank 的能力——raw rank 一致性校验放在 `PadAndSqueeze` 之前，把合法输入提前判死。
- **判读（interpretation，根因分析）**：raw rank 一致性校验是**冗余且自相矛盾**的插入。三处证据都要求允许不同 rank：(1) aclnn API 文档要求 gradOutput 与 self 满足 broadcast 关系（PyTorch/numpy 风格右对齐广播，允许不同 rank）；(2) 同算子 InferShape 已按 numpy 广播右对齐、前缀补 1 推导输出 shape；(3) `PadAndSqueeze` 的 `pad` lambda 已用 `max_rank - s.size()` 个 1 对短 rank 左填充。`docs/req_analysis.md` §4.6.2「两输入 rank 不同 → GRAPH_FAILED」描述的是**旧 CANN TBE 实现**的历史限制（实现限制，非算子需求），新 Ascend C 实现应遵循 aclnn API 文档的 broadcast 规则。该校验把历史限制当成需求硬编码进新实现，造成对标量/低 rank 广播输入的错误拒绝。判据：报错文本含 `rank(gradOutput)=0 != rank(self)=2` + 代码定位到 `raw_input_shapes_[0].size() != raw_input_shapes_[1].size()` 紧随 `PadAndSqueeze` 之前 → 命中本根因。

## 验证信号

- 复现报错 3 例（L0 模板「标量 rank=0 vs [2,3]」float32/float16/bfloat16 三 dtype 全部 `[ERROR]`，报错文本含 `rank(gradOutput)=0 != rank(self)=2`，由算子 `OP_LOGE` 打印）（来源平台：950；证据等级：tool_observed）。
- 同次跑全量 581 用例：通过 463、失败 118、跳过 0（其中 3 例即上述 rank ERROR，余 117 例为 L1 精度 FAIL，属另一独立现象，另卡处理）（来源平台：950；证据等级：tool_observed）。
- `GetShapeAttrsInfo` 中 raw rank 一致性 `OP_CHECK_IF` 校验紧随其后即 `PadAndSqueeze(...)` 调用，verbatim 取自 `op_host/arch35/softplus_v2_grad_tiling_arch35.cpp` 第 231-236 行（file-read 事件 seq 40）（来源平台：950；证据等级：artifact_verified）。
- `PadAndSqueeze` 的 `pad` lambda 已 `p.assign(max_rank - s.size(), 1)` 对短 rank 左填充 1，证明 tiling 流程本就支持不同 rank、raw rank 校验是冗余插入（verbatim 自同次 file-read）（来源平台：950；证据等级：artifact_verified）。

## 预防

在为「双输入 + numpy 广播」类算子写 Host tiling 时，不要把 `raw_input_shapes_[i].size()` 的 raw rank 一致性当成需求级硬校验——广播契约本就允许不同 rank。若要保留防御性校验，应校验**右对齐后逐维兼容**（一维为 1 或两两相等），而不是 raw rank 相等。注意区分「旧 CANN TBE 实现的历史限制」（可在 req_analysis 里以「实现限制」措辞登记）与「新实现的需求」——不要把历史限制硬编码进新实现。

## 失效边界

- Ascend950PR 实测复现并定位到该 code path；a3/a2 **未验证，推断可迁移**（同一 tiling_arch35.cpp 在其他 arch 若保留相同 raw rank 校验也会触发同类现象，但本 trace 未在 a3/a2 编译/跑过）。
- 本卡仅诊断根因与修复方向（移除冗余 raw rank 校验、依赖 PadAndSqueeze），**未应用代码修复、未验证修复后通过**——修复闭环留给后续 trace/curator。
- 旧 CANN TBE 实现（`aclnnSoftplusBackward`）按其历史行为对 rank 不同返回 GRAPH_FAILED，那条历史限制不适用本卡的修复方向。

## 坏实践

把旧实现的历史限制（req_analysis §4.6.2「两输入 rank 不同 → GRAPH_FAILED」）当作算子需求，原样硬编码进新 Ascend C 实现的 tiling 校验，导致新实现无法满足 aclnn API 文档的 broadcast 契约、且与同算子 InferShape 自相矛盾。

## 设计决策

- tiling 选 `PadAndSqueeze` + `CheckBroadcastShape` 两段式：先左填充 1 对齐到 max_rank、再逐维校验兼容性——这本就是处理不同 rank 广播的正解，raw rank 一致性校验是与该设计自相矛盾的冗余插入。
- per_buf 按 fp32 粒度（`dtype_size_ = 4`）固定，与 rank 校验无关；rank 校验是独立插入的防御性检查，非 tiling 切分逻辑必需。