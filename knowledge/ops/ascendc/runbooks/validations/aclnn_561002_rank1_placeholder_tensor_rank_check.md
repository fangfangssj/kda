---
type: "Runbook"
title: "ACLNN 561002：占位 1D 张量过不了 rank 校验"
description: "aclnn ST 最小调用通路把全部 aclTensor 建成 rank-1 {1} 占位张量，Host 侧输入校验按 dtype→format→维度→attr→shape 拦截，GetWorkspaceSize 报 561002；按 proto 声明的 rank/shape 建张量即通过，可选输入传 nullptr 不是成因。"
tags: [task.testing, topic.architecture]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# ACLNN 561002：占位 1D 张量过不了 rank 校验

## 现象

为一个基于 ACLNN 算子仓框架生成的自定义算子（arch35/Ascend 950，CANN 9.2.0）编写 aclnn 集成测试（ST）的"最小调用通路"用例：用例的目的是只验证 `aclnn<Op>GetWorkspaceSize → aclnn<Op>` 调用链能走通，不做数值校验。旧版用例为了省事，把全部 10 个 aclTensor（8 个输入 + 2 个输出）都建成 rank-1、dims={1} 的 1 元素占位张量，可选输入 conv_state 传 nullptr。

在 `tests/aclnn` 目录执行 `./run.sh`（内部 cmake + make + 运行测试可执行文件），测试本体编译链接正常，但运行时在 aclnn 两段式调用的**第一段**就报错退出：

```
ACLNN error 561002 at .../operators/engram_gate_fusion_package/tests/aclnn/test_aclnn.cpp:120: aclnnBenchHarnessEngramGateFusionGetWorkspaceSize( keys, hiddenStates, value, norm1Weight, norm2Weight, convNormWeight, convWeight, nullptr, hcMult, hiddenSize, kernelSize, dilation, normEps, output, convStateOut, &workspaceSize, &executor)

==== ACLNN BenchHarnessEngramGateFusion summary: 0/1 cases PASSED ====
```

错误发生在 GetWorkspaceSize 阶段，尚未进入 kernel 执行；报错点正是旧用例里发起 GetWorkspaceSize 调用的那一行。

## 适用场景

- 工程形态：aclnn ST/集成测试（宿主 C++ 测试程序，经 aclnn 两段式接口调用自定义算子，vendor 包已 `./build.sh <pkg>` 安装、`LD_LIBRARY_PATH` 指向 `opp/vendors/<Op>/op_api/lib`）。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`；`aclCreateTensor(dims, dimCount, dtype, strides, ...)`。
- 验证过的平台与 CANN 版本：950 + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有走 aclnn 两段式调用、Host 侧带输入校验的自定义算子测试均适用。
- 路线优先级：recommended_fastpath（按算子 proto 声明的 rank/shape 构造 aclTensor 是唯一正路）。
- 明确不适用 / 未验证场景：错误发生在第二段（`aclnn<Op>` 执行）或 kernel 内部（越界/NaN/崩溃）时不适用本卡；Graph（GE 构图）模式下占位张量是否报同一错误码未验证；其他 CANN 版本的校验顺序未验证。

## 根因

单句归因：aclnn 两段式调用的第一段会在 Host 侧执行算子的输入校验（顺序 dtype → format → 维度 rank → attr → shape，任一失败返回 GRAPH_FAILED 并在 aclnn 边界以 561002 报出），占位 rank-1 `{1}` 张量不满足算子 proto 声明的 rank 约束（keys/hidden_states 4D、value 3D、norm*_weight 2D、conv_weight 3D），校验在"维度"一步即失败。

机制链（不跳步）：

1. `aclnn<Op>GetWorkspaceSize(...)` 进入 Host Tiling 公共链：`GetPlatformInfo → GetInputShape + 读 attr → 异常值校验（dtype→format→维度→attr→shape）`。
2. 旧用例所有张量都是 `aclCreateTensor(dims /*={1}*/, 1, ...)`，dimCount=1：keys 需要 rank==4，rank 校验失败 → Host 侧返回 GRAPH_FAILED 早退（与设计文档中"异常 case 在 Host Tiling 阶段 GRAPH_FAILED 早退、不进 kernel"的声明一致）。
3. aclnn 层把该 Host 校验失败以错误码 561002 返回给测试进程，调用链在 GetWorkspaceSize 即中断，测试进程打印 `ACLNN error 561002` 并以 0/1 通过收场。

## 解决配方

按算子 proto/DESIGN 为每个张量给出真实 rank/shape（元素总数可以取最小值，但 rank 与各维语义必须正确），并让 attr 值与 shape 一致（`hc_mult == keys.shape[2]`、`hidden_size == keys.shape[3]`，否则同样会被 attr-shape 一致性校验拦截）。可选输入（本例 conv_state）可以继续传 nullptr——修后用例仍传 nullptr 且通过，证明它不是 561002 的成因。

修前（旧最小调用通路用例，全部张量 rank-1 占位）：

```cpp
// --- Tensor shapes: 1-element tensors for all inputs/outputs ---
// BF16 tensors: keys, hidden_states, value, conv_state(optional), output, conv_state_out
// FP32 tensors: norm1_weight, norm2_weight, conv_norm_weight, conv_weight
const int64_t dims[]   = {1};
const int64_t strides[] = {1};
const int64_t storage[] = {1};
...
aclTensor *keys         = aclCreateTensor(dims, 1, ACL_BF16,  strides, 0, ACL_FORMAT_ND, storage, 1, devKeys);
aclTensor *hiddenStates = aclCreateTensor(dims, 1, ACL_BF16,  strides, 0, ACL_FORMAT_ND, storage, 1, devHiddenStates);
aclTensor *value        = aclCreateTensor(dims, 1, ACL_BF16,  strides, 0, ACL_FORMAT_ND, storage, 1, devValue);
// FP32 inputs 同样全部 dims={1}、dimCount=1
// conv_state is OPTIONAL — pass nullptr (Prefill mode)
```

修后（辅助函数按 dims 逐维推导 stride，rank 由 dims.size() 决定）：

```cpp
static TensorInfo createTensor(const std::vector<int64_t>& dims, aclDataType dtype, const void* hostData = nullptr)
{
    TensorInfo ti;
    int64_t totalElems = 1;
    for (auto d : dims) totalElems *= d;
    size_t elemBytes = (dtype == ACL_BF16) ? 2 : 4;
    ti.bytes = static_cast<size_t>(totalElems) * elemBytes;
    int n = static_cast<int>(dims.size());
    std::vector<int64_t> strides(n);
    if (n > 0) {
        strides[n-1] = 1;
        for (int i = n-2; i >= 0; i--) strides[i] = strides[i+1] * dims[i+1];
    }
    size_t allocBytes = std::max(ti.bytes, size_t{1});
    if (aclrtMalloc(&ti.devPtr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return ti;
    if (hostData && ti.bytes > 0)
        aclrtMemcpy(ti.devPtr, ti.bytes, hostData, ti.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ti.tensor = aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0,
                                ACL_FORMAT_ND, dims.data(), dims.size(), ti.devPtr);
    return ti;
}
```

最小调用通路用例按 proto 声明的 rank/shape 建张量（B=1, L=4, HC=2, D=8, K=4, dil=3 的 Prefill 场景，无 conv_state）：

```cpp
const int64_t B=1, L=4, HC=2, D=8, K=4, dil=3;
const int64_t CD=HC*D, SL=(K-1)*dil;
...
TensorInfo tdKeys   = createTensor({B,L,HC,D}, ACL_BF16, keysHost.data());
TensorInfo tdHidden = createTensor({B,L,HC,D}, ACL_BF16, hiddenHost.data());
TensorInfo tdValue  = createTensor({B,L,D},    ACL_BF16, valueHost.data());
TensorInfo tdW1     = createTensor({HC,D},     ACL_FLOAT, w1Host.data());
TensorInfo tdW2     = createTensor({HC,D},     ACL_FLOAT, w2Host.data());
TensorInfo tdWc     = createTensor({HC,D},     ACL_FLOAT, wcHost.data());
TensorInfo tdWConv  = createTensor({CD,1,K},   ACL_FLOAT, wConvHost.data());
TensorInfo tdOutput = createTensor({B,L,HC,D}, ACL_BF16);
TensorInfo tdCSOut  = createTensor({B,CD,SL},  ACL_BF16);
...
CHECK_ACLNN(aclnnBenchHarnessEngramGateFusionGetWorkspaceSize(
    tdKeys.tensor, tdHidden.tensor, tdValue.tensor,
    tdW1.tensor, tdW2.tensor, tdWc.tensor, tdWConv.tensor, nullptr,   // conv_state 仍传 nullptr
    HC, D, K, dil, 1.0e-5,
    tdOutput.tensor, tdCSOut.tensor, &workspaceSize, &executor));
```

## 设计决策

- 为什么不设法"放过占位张量"：Host 输入校验是算子接口契约的一部分（异常输入在 Host Tiling 阶段 GRAPH_FAILED 早退是设计行为），测试侧的正确做法是提供符合 proto 契约的真实 rank/shape 张量，而不是绕过或放松校验。
- conv_state 保持 nullptr：修后用例仍传 nullptr 且通过，说明可选输入缺省与 561002 无关；为没有 conv_state 的场景强行构造 dummy 张量反而掩盖语义（对比：Host Tiling 单元测试的 MakeCase 因 builder 会解引用张量指针而给 conv_state 塞 dummy `{1}`，见 [TilingFunc直调UT传nullptr输入段错误](tiling_gtest_optional_input_nullptr_segfault_placeholder_tensor.md)——tiling UT 直调层 nullptr 即崩、aclnn 层 nullptr 合法，层间行为相反，勿把两层的构造要求混用）。
- 本修法属于 recommended_fastpath：一次修正即恢复调用链，无额外代价。

## 预防

- 写 aclnn ST 最小调用通路用例前，先从 proto.md/DESIGN 抄出每个输入输出的 rank 与 shape 语义，再写 `createTensor`；不要复制"全 1 元素张量"的占位模板。
- 出现 561002 时先核对每个 aclTensor 的 dimCount 与各维大小是否满足算子校验链（dtype→format→维度→attr→shape），再怀疑可选输入指针。
- 561002 已知根因速查（各为独立闭环，判别维度不同）：[561002：TilingParse 未生效，CompileInfo 为空](../compilation/561002_tilingparse_compileinfo.md)（注册失效，查构建系统注册链）；[561002：torch_npu GE 初始化干扰 aclnn tiling](../compilation/561002_torch_npu_ge_init_tiling_interference.md)（同进程 GE 劫持，查宿主进程环境）；[标量输入被赋非标量 shape 致 561002](../compilation/shape_561002.md)（spec role:scalar 输入被赋大 shape——与本卡同为用例侧张量构造违反接口契约，但其判别维度是 spec role，本卡是 proto rank）。

## 验证信号

以下信号均为本开发会话（编写该算子 Branch-0 Kernel ST 红基线）内的实测输出：

- 修前实跑：`ACLNN error 561002 at .../test_aclnn.cpp:120: aclnnBenchHarnessEngramGateFusionGetWorkspaceSize(...)`，`0/1 cases PASSED`（来源平台：950；证据等级：tool_observed）。
- 修后实跑（同一 run.sh 流程）：`Test minimal_call_path: OK`，`==== ACLNN BenchHarnessEngramGateFusion summary: 1/23 cases PASSED ====`（22 个数值用例按 kernel stub 阶段预期失败，属测试先行红基线；最小调用通路恢复通过），进程退出码 0（来源平台：950；证据等级：tool_observed）。
- 独立验收会话重跑 `./run.sh` 复核：`Test minimal_call_path: OK`、`1/23 cases PASSED`、`RUN_EXIT=0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖 aclnn 两段式调用**第一段**的 Host 输入校验失败；若错误码出现在第二段执行或 kernel 内部，结论不适用。
- "精确命中 rank 检查（而非后续 attr-shape 检查）"依据设计文档声明的校验顺序（维度在 attr/shape 之前）推断，本会话未做单变量隔离实验；其他算子校验顺序不同时报错码可能不同。
- Graph（GE 构图）模式、其他 CANN 版本未验证；推断凡 Host 侧按 proto 校验 rank 的算子行为一致（未验证，推断可迁移）。

**未决项**：

- 优先项：复现并做单变量隔离（只改 rank、不改 attr 值），把失败精确定位到"维度 rank 检查"这一步，并抓取 Host 侧日志确认 GRAPH_FAILED → 561002 的映射链。
- 561002 与 Host GRAPH_FAILED 的映射关系是依调用链行为推断，未在会话内用独立日志证实。
## 坏实践

- 把全部 aclTensor 建成 dims={1} 的 rank-1 占位张量来"最小化"调用通路用例：必被 rank 校验拦截，得到 561002。
- 把 561002 直接归因于"可选输入传了 nullptr"并为此构造 dummy 张量：本会话 before/after 对照已否定该归因。

## 误诊

- "conv_state 传 nullptr 导致 561002"——修前修后两版用例都传 nullptr，只有 rank/shape 正确的版本通过，属误诊（agent 诊断消息亦明确指向 1D `[1]` 形状）。

## 后续增量（2026-09-17，并行策展批：新增 TilingFunc CheckShape 后旧冒烟成批回归 561002——测试维护债而非实现回归）

另一独立轨迹并入 apply_rotary_pos_emb 算子（Ascend950PR，CANN 9.2.0）的对称场景：主卡是"占位 rank-1 张量撞既有校验"，本增量是"**新加校验撞旧测试的越界输入**"——同一个指纹（GetWorkspaceSize 阶段成批 561002）的另一个方向。

**现象**：给算子实现公共 TilingFunc（含 CheckShape 七项 shape/dtype/attr 校验）之后，回归跑历史遗留的三 dtype 冒烟测试（`tests/aclnn/run.sh`），三个用例全部在第一阶段报同一个错误码：

```
ACLNN error 561002 at .../tests/aclnn/test_aclnn.cpp:63: aclnnBenchHarnessApplyRotaryPosEmbGetWorkspaceSize( query, key, cos, sin, 0, const_cast<char*>("half"), queryOut, keyOut, &workspaceSize, &executor)
ACLNN error 561002 at ...（fp32/fp16/bf16 三个 dtype 用例同样报错）

==== Smoke summary: 0/3 cases PASSED (fp32, fp16, bf16) ====
RUNSH_EXIT=255
```

此前（TilingFunc 还是不过校验的 stub 时）同一冒烟测试 3/3 通过。冒烟输入是老接口时代的 2 维小张量：`const int64_t dims[] = {2, 4};`、`aclCreateTensor(dims, 2, ...)`——query/key/cos/sin 全部 rank-2。而新 TilingFunc 的 CheckShape 首项校验即 `rank(query)==4 且 rank(key)==4`（算子 spec 要求 4D BSND/BNSD 输入），rank-2 输入被判 `RANK_NOT_4` → `return ge::GRAPH_FAILED` → aclnn 在 GetWorkspaceSize 阶段向调用方报 561002。

**判别增量**：

1. 多个 dtype/shape 不同的用例**成批**同样失败在 GetWorkspaceSize 调用行 → 优先查 host 侧校验（本例在 TilingFunc），而不是 aclnn 胶水/OpDef 注册/动态库链接。
2. 输入合法性物证：读测试源码核对输入 rank/shape 是否满足算子 spec（`const int64_t dims[] = {2, 4};` + `aclCreateTensor(dims, 2, ...)`）。**旧测试在"校验缺失的 stub 时代"通过 ≠ 输入合法**。
3. 对照证据：形式化 TilingUT 33/33 通过（含同公式的 CheckShape 独立 oracle），佐证失败来自输入侧而非公式/实现侧。

**处置口径（与主卡一致：改测试数据，不放松校验）**：校验拒绝是 spec 语义的正确行为；该回归暴露的是**旧测试输入不合规**（测试维护债），不是实现回归。正确处置是把旧用例数据更新为合法 4D 输入。为让旧测试变绿而删掉/放松 CheckShape 属测试倒逼实现退化，禁止。

**复现增量验证信号**（公共 Tiling 实现落地后的验收会话）：

- 冒烟回归实测：`ACLNN error 561002 at ...: aclnnBenchHarnessApplyRotaryPosEmbGetWorkspaceSize(...)` ×3，`==== Smoke summary: 0/3 cases PASSED ====`，`RUNSH_EXIT=255`（来源平台：950；证据等级：tool_observed）。
- 冒烟输入 rank-2 的物证：测试源码 `const int64_t dims[] = {2, 4};`、`aclCreateTensor(dims, 2, ...)`（来源平台：950；证据等级：artifact_verified）。
- TilingFunc 拒绝路径的物证：glue 中 `if (CheckShape(csIn) != CheckShapeStatus::OK) { return ge::GRAPH_FAILED; }`，CheckShape 首项 `if (in.qRank != 4 || in.kRank != 4) return St::RANK_NOT_4;`（来源平台：950；证据等级：artifact_verified）。
- 对照：同会话形式化 TilingUT 33/33 通过（实现正确），佐证失败来自输入侧（来源平台：950；证据等级：tool_observed）。

**复现增量边界**：561002 在本开发会话 观测为 GetWorkspaceSize 阶段与 TilingFunc GRAPH_FAILED 的绑定码值，其官方错误码语义未在本会话查证（同码可由不同根因触发，本卡只覆盖"TilingFunc 校验拒绝"一支——速查其它 561002 根因见主卡预防节）；"判定测试维护问题而非实现回归"依据 spec rank 约束与 UT 全绿，不同团队对"前序已过用例回归不退化"的口径可能更严；旧冒烟用例按 4D 口径更新的闭环落在后续任务（本会话未执行）。