---
type: "Runbook"
title: "advance_step in-place kernel FATBIN wrap"
description: "ascend950 FATBIN/TPL wrapper 按算子注册的输入+输出数量与顺序生成调用（+workspace+tiling），kernel 入口签名必须与之严格对齐：in-place 算子缺 output 形参、或把标量属性写进签名（GM_ADDR k），都会 no matching function 编译失败、.o 缺失、FATBIN 报 cannot open .o。属性不在 wrapper 传参序列中，必须经 TilingData 传递。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-18T03:30:00Z"
updated_at: "2026-09-16T11:05:00Z"
platforms: ["950"]
aliases: ["add_kernel_sources", "MulAddDst"]
---

# advance_step in-place kernel FATBIN wrap

## 现象

在 Ascend950PR（CANN 9.0.0）上为 `advance_step` 算子补齐 `op_host/advance_step_def.cpp` 的 4 个 `Output(...)` 声明后，执行 `bash build.sh --pkg --soc=ascend950 --ops=advance_step -j16`，FATBIN 阶段直接失败。报错原文（来自 build 日志 tool 输出）：

```
[ERROR] ... Exception: An error occurred during compile phases of CompileStage.FATBIN,
msg is ld.lld: error: cannot open /workspace/cann/ops-nn/scripts/kernel/binary_script/
kernel_meta_AdvanceStep_52f00e7e9250348bc94f9de4b551b94e_high_performance/kernel_meta/
AdvanceStep_52f00e7e9250348bc94f9de4b551b94e_high_performance_1.o: No such file or directory
ld.lld: error: cannot open .../AdvanceStep_52f00e7e9250348bc94f9de4b551b94e_high_performance_2.o: No such file or directory
[ERROR] ... Op[AdvanceStep] of index[0] compile failed,
kernelName: AdvanceStep_52f00e7e9250348bc94f9de4b551b94e_high_performance.
```

即 `.o` 中间产物从未生成，ld.lld 在 FATBIN 链接阶段找不到它。进一步查 CCEC 编译日志（`build/binary/ascend950/bin/build_logs/advance_step_binary_high_performance.json_0.log`）发现根因：FATBIN 生成的 wrapper kernel 调用 `advance_step()` 时传了 14 个实参（8 input + 4 output + workspace + tiling），但 kernel 函数只声明 10 个形参（8 input + workspace + tiling，没有 output 参数）。

`advance_step` 是 in-place 算子（输出覆写输入 buffer），原 kernel 作者据此省略了 output 形参；但 ascend950 FATBIN wrapper 是按 `OpDef` 声明的 input+output 总数自动生成调用点的，不会为 in-place 语义省略 output 参数。

对偶方向（独立复现，CANN 9.1.0，registry-invoke + ASCENDC_TPL 模板入口）：融合算子定义了标量属性 `k`（topK 数量，`this->Attr("k").Int()`），kernel 入口按"参数齐全"的直觉把属性也写进了签名（`GM_ADDR k`）。构建推进到 FATBIN 链接阶段先看到误导性的 `ld.lld: error: cannot open ..._27.o/_283.o`；用 `bash build.sh --make_clean` 后 `-j1` 重跑并抓生成文件，才看到真实错误——所有 6 个 tiling key 变体的自动生成入口全部编译失败：

```
kernel_meta_BenchNoHarnessMoeGatingTopKSoftmax_..._2949239_kernel.cpp:39:5:
  error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_256_tilingkey'
kernel_meta_...:39:5: error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_0_tilingkey'
kernel_meta_...:39:5: error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_27_tilingkey'
kernel_meta_...:39:5: error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_283_tilingkey'
kernel_meta_...:39:5: error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_257_tilingkey'
kernel_meta_...:39:5: error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_1_tilingkey'
```

展开宏可见框架生成入口的调用形态（`#define bench_no_harness_moe_gating_top_k_softmax bench_no_harness_moe_gating_top_k_softmax_1_tilingkey` 后）：

```cpp
bench_no_harness_moe_gating_top_k_softmax<TEMPLATE_PARAMS>(x_in__, finished_in__, y_out_, expert_idx_out_, row_idx_out_, usrWorkspace, tiling);
```

调用侧传的是 7 个参数（x, finished, y, expert_idx, row_idx, usrWorkspace, tiling）——**不含属性 k**；而自己写的函数定义是 8 个参数，宏重命名后同名但形参列表不匹配，模板实例化失败，`.o` 一个都没生成，FATBIN 自然报"cannot open .o"。

## 适用场景

- **工程形态**：构建（ops-nn `build.sh --pkg --soc=ascend950`，FATBIN/CCEC kernel 预编译；及 registry-invoke ASCENDC_TPL 模板入口的 per-tilingkey wrapper 生成）。
- **API 路径**：`op_kernel/<op>.cpp` 的 `extern "C" __global__ __aicore__ void <op>(GM_ADDR ...)` / 模板 `__global__ __aicore__` kernel 入口签名 + `op_host/<op>_def.cpp` 的 `OpDef` Input/Output/Attr 声明。
- **验证过的平台与 CANN 版本**：Ascend950PR / CANN 9.0.0（`ascend950` AddConfig）；属性方向为 CANN 9.1.0（soc=ascend950，ASCENDC_TPL 模板参数体系）。
- **适用粒度**：`operator_family`（所有在 ascend950 上走 FATBIN/TPL wrapper 预编译的算子；in-place 省略 output 形参、或入口签名混入属性形参均适用）。
- **路线优先级**：`recommended_fastpath`（按 OpDef 声明与 wrapper 固定序列对齐签名是 build 系统的契约要求，非可选）。
- **不适用或未验证场景**：aclnn 两段式通路（不经 FATBIN wrapper 直接调 kernel）不在本卡范围；老式 TILING_KEY_IS 入口或其他 wrapper 生成逻辑未验证；ascend910b / ascend910_93 的 FATBIN 行为本开发会话 未验证（推断同构，标注"未验证，推断可迁移"于失效边界）。

## 根因

ascend950 FATBIN 阶段为 kernel 自动生成一个 wrapper（TPL 体系为每个 tiling key 生成 `_<key>_tilingkey` 入口），wrapper 的调用实参 = `OpDef` 声明的 input 数 + output 数 + workspace + tiling，**按注册顺序固定、属性不在其中**。当 `OpDef` 声明了 4 个 Output、而 kernel 函数签名因 in-place 语义省略了 output 形参时，或 kernel 入口签名多写一个属性形参时，CCEC 编译 wrapper 时出现 `no matching function for call` / `candidate function not viable`，CCEC 编译失败 → `.o` 不生成 → ld.lld 报 `cannot open ...o`。根因是**kernel 签名必须与 wrapper 固定传参序列（输入序列 + 输出序列 + usrWorkspace + tiling）严格对齐**：in-place 语义不能作为省略 output 形参的理由，属性也不能挤进签名。

标量属性的正确通道是 host 侧 TilingFunc 把值写进 TilingData、kernel 侧从 tiling 指针读（`GET_TILING_DATA_WITH_STRUCT` 一次带出全部标量）。

## 解决配方

### Before（原版 10 参数，无 output 形参）

`op_kernel/advance_step.cpp`（verbatim 取自 trace file_read 事件）：

```cpp
extern "C" __global__ __aicore__ void advance_step(GM_ADDR input_tokens, GM_ADDR sampled_token_ids,
                                                   GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
                                                   GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

    if (TILING_KEY_IS(1)) {
        KernelAdvanceStep<int64_t> op;
        op.Init(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, userWs,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        TPipe Pipe;
        KernelAdvanceStepSpec<int64_t> op;
        op.Init(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, spec_token,
                accepted_num, userWs, &tilingData, &Pipe);
        op.Process();
    }
}
```

### After（14 参数，补 4 个 `GM_ADDR *_out` 形参）

`op_kernel/advance_step.cpp`（verbatim 取自 trace file_read 事件；output 形参插入在 `accepted_num` 与 `workspace` 之间）：

```cpp
extern "C" __global__ __aicore__ void advance_step(GM_ADDR input_tokens, GM_ADDR sampled_token_ids,
                                                   GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
                                                   GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num,
                                                   GM_ADDR input_tokens_out, GM_ADDR input_positions_out,
                                                   GM_ADDR seq_lens_out, GM_ADDR slot_mapping_out,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

    int64_t numSeqs = tilingData.numSeqs;
    auto* inTokens = (__gm__ int64_t*)input_tokens;
    auto* outTokens = (__gm__ int64_t*)input_tokens_out;
    auto* inPositions = (__gm__ int64_t*)input_positions;
    auto* outPositions = (__gm__ int64_t*)input_positions_out;
    auto* inSeqLens = (__gm__ int64_t*)seq_lens;
    auto* outSeqLens = (__gm__ int64_t*)seq_lens_out;
    auto* inSlotMapping = (__gm__ int64_t*)slot_mapping;
    auto* outSlotMapping = (__gm__ int64_t*)slot_mapping_out;
    for (int64_t i = GetBlockIdx(); i < numSeqs; i += GetBlockNum()) {
        outTokens[i] = inTokens[i];
        outPositions[i] = inPositions[i];
        outSeqLens[i] = inSeqLens[i];
        outSlotMapping[i] = inSlotMapping[i];
    }
    PipeBarrier<PIPE_ALL>();

    if (TILING_KEY_IS(1)) {
        KernelAdvanceStep<int64_t> op;
        op.Init(input_tokens_out, sampled_token_ids, input_positions_out, seq_lens_out, slot_mapping_out,
                block_tables, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        TPipe Pipe;
        KernelAdvanceStepSpec<int64_t> op;
        op.Init(input_tokens_out, sampled_token_ids, input_positions_out, seq_lens_out, slot_mapping_out,
                block_tables, spec_token, accepted_num, userWs, &tilingData, &Pipe);
        op.Process();
    }
}
```

注意 after 版本同时把 `op.Init` 的前四个 buffer 参数从 input 切换为 `*_out`（另一独立修复），本卡只覆盖签名参数对齐这一根因。

### 对偶方向：从模板入口签名删除属性参数（verbatim，CANN 9.1.0）

before（`op_kernel/bench_no_harness_moe_gating_top_k_softmax_arch35.cpp`，属性 k 混入签名）：

```cpp
#include "arch35/bench_no_harness_moe_gating_top_k_softmax.h"

template <typename D_T_X, bool HAS_FINISHED>
__global__ __aicore__ void bench_no_harness_moe_gating_top_k_softmax(
    GM_ADDR x, GM_ADDR finished, GM_ADDR k, GM_ADDR y,
    GM_ADDR expertIdx, GM_ADDR rowIdx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData, tilingData, tiling);

    NsBenchMoeGating::BenchNoHarnessMoeGatingTopKSoftmax<D_T_X, HAS_FINISHED> op;
    op.Init(x, finished, y, expertIdx, rowIdx, &tilingData);
    op.Process();
}
```

after（删除 `GM_ADDR k`，形参序列与 wrapper 调用一致）：

```cpp
#include "arch35/bench_no_harness_moe_gating_top_k_softmax.h"

template <typename D_T_X, bool HAS_FINISHED>
__global__ __aicore__ void bench_no_harness_moe_gating_top_k_softmax(
    GM_ADDR x, GM_ADDR finished, GM_ADDR y,
    GM_ADDR expertIdx, GM_ADDR rowIdx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData, tilingData, tiling);

    NsBenchMoeGating::BenchNoHarnessMoeGatingTopKSoftmax<D_T_X, HAS_FINISHED> op;
    op.Init(x, finished, y, expertIdx, rowIdx, &tilingData);
    op.Process();
}
```

配套（host 侧把属性写入 TilingData，该算子首次版本即已正确、无需改动）：

```cpp
    tiling->k = k;   // TilingFunc 内，k 取自 attrs
```

排障时抓生成入口的命令（构建失败会清理生成物，需 make_clean 后单线程重跑再立刻查看）：

```bash
cd <op> && bash build.sh --make_clean 2>&1 | tail -1 && bash build.sh --soc=ascend950 -j1 2>&1 | tail -5
find build/op_kernel -name "*_kernel.cpp" | head -1 | xargs head -60
```

`kernel_meta_*_kernel.cpp` 里的调用语句就是契约的机器可读形式，一眼定案。

## 设计决策

- wrapper 传参序列（输入、输出、workspace、tiling）是注册信息决定的固定契约，kernel 侧无从扩展；in-place 语义、属性传递都不能改变签名形态。
- 属性走 TilingData 而不是 GM 参数：TilingData 是 host→kernel 传标量的正规通道，且 `GET_TILING_DATA_WITH_STRUCT` 一次带出全部标量。
- 排障顺序上选择"抓生成入口源码对照"而不是"反复猜签名"。

## 验证信号

- 补齐 4 个 output 形参后，`build.sh --pkg --soc=ascend950` FATBIN 阶段通过、CCEC 生成 `.o`，`[SUCCESS] Build package success!`（来源平台：950；证据等级: tool_observed）。
- custom module 安装返回 `SUCCESS`（来源平台：950；证据等级: tool_observed）。
- 对偶方向修复前：6 个 tiling key 全部 `error: no matching function for call to 'bench_no_harness_moe_gating_top_k_softmax_{0,1,27,256,257,283}_tilingkey'`，FATBIN 报 `ld.lld: cannot open ..._27.o/_283.o`（来源平台：950；证据等级：tool_observed）
- 对偶方向修复后：`bash build.sh --soc=ascend950 -j8` 成功生成 3 个 kernel 二进制（覆盖全部 6 个 tiling key）与算子包；最终复跑 `Build completed successfully!`，`exit code: 0`（来源平台：950；证据等级：tool_observed）

## 失效边界

- ascend910b / ascend910_93 的 FATBIN wrapper 行为：未验证，推断可迁移（同一 build 系统按 OpDef 生成 wrapper 的机制应一致，但本开发会话 仅在 ascend950 实测）。
- aclnn 两段式通路不经 FATBIN wrapper，本卡不适用。
- 若 OpDef 的 Output 声明数与 kernel output 形参数再次不一致（例如后续新增 output），会以同样的 `requires N arguments, but M were provided` 复现。
- 属性方向结论基于 ASCENDC_TPL 模板参数体系（dynamicCompileStatic）+ CANN 9.1.0；老式 TILING_KEY_IS 入口或其他 wrapper 生成逻辑未验证；后续版本若允许属性进签名（未验证），此约束可能变化。

## 预防

- kernel 入口签名模板固定为 `(全部输入按注册顺序, 全部输出按注册顺序, GM_ADDR workspace, GM_ADDR tiling)`；写完入口后数一遍参数个数 = 输入数 + 输出数 + 2。
- 属性一律经 TilingData：host TilingFunc 从 `context->GetAttrs()` 取值写入 tiling 结构，kernel 从 tiling 读。
- 任何使用 `SetTilingKey(N>0)` 的算子，struct.h 的 `ASCENDC_TPL_SEL` 必须为 0..N 全部 KEY 显式列出 SEL 项（见关联卡《ACLNN 361001 — tilingKey 与 funcEntry 映射失》）。

## 坏实践

- 看到 FATBIN 的 "cannot open .o" 就去清缓存/重装工具链——真实编译错误在 kernel_meta 生成入口里，且默认输出被清理，需 make_clean 后 -j1 重跑抓取。
- 把"属性"当"输入"写进签名——属性在注册信息里不属于输入序列，wrapper 不会传。
- 为 in-place 语义省略 output 形参。

## 误诊

- 排查初期曾以为只有 bf16 两个变体（_27/_283）失败、fp16/fp32 正常——抓到生成入口后确认 6 个 key 全部失败，ld.lld 报缺 .o 的先后顺序造成了误导。
