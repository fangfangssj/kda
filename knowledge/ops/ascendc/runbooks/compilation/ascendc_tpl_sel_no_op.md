---
type: "Runbook"
title: "缺 ASCENDC_TPL_SEL 致运行时分发静默 no-op 跳过"
description: "euclidean_norm kernel 入口用非模板 extern C + 运行时 if/else + PipeBarrier fall-through 兜底，偏离 proto.md §3 / DESIGN §5.6.1 的 ASCENDC_TPL_SEL 编译期分发；未命中分支静默 no-op，Group+tail-A special-value 用例被 harness 以 SKIP (no golden) 跳过而非报错。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T08:56:02Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 缺 ASCENDC_TPL_SEL 致运行时分发静默 no-op 跳过

## 现象

编写 EuclideanNorm（euclidean_norm）算子、对齐 `docs/euclidean_norm` 规范时发现：kernel 入口实现与 `proto.md §3` / `DESIGN.md §5.6.1` 规定的 Kernel Entry 不一致。规范要求模板化入口 `template <bool templateType, bool isEmptyTensor, bool isTailR> __global__ __aicore__ void euclidean_norm(...)`，由 `ASCENDC_TPL_SEL` 在 `*_struct.h` 内声明 5 个 sub-kernel 实例、host 用 `SetTilingKey(GET_TPL_TILING_KEY(...))` 运行期选择。

实现侧（修复前 `op_kernel/euclidean_norm_apt.cpp`）却是一个**非模板** `extern "C"` 函数，内部按 `tilingData.axisNum / usedCoreNum / rGroupCnt` 运行时 `if/else` 分发，未命中三个 `if` 分支的输入 fall through 到末尾 `AscendC::PipeBarrier<PIPE_ALL>()`（no-op stub）后返回：

```cpp
extern "C" __global__ __aicore__ void euclidean_norm(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(EuclideanNormTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.axisNum == 0) {
        // EMPTY branch
    } else if (tilingData.usedCoreNum > 0 && tilingData.rGroupCnt > 0) {
        // Group branch (dispatch by axisNum parity)
    } else if (tilingData.usedCoreNum > 0 && tilingData.rGroupCnt == 0) {
        // Normal branch (dispatch by axisNum parity)
    }
    // Unmatched cases fall through to PipeBarrier (no-op).
    (void)workspace;
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

同时 `op_kernel/arch35/euclidean_norm_tiling_struct.h` 仅是 include shim，**没有任何 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 宏**；`CMakeLists.txt` 用 `npu_op_kernel_options(... OPTIONS "--tiling_key=0,1,2,3,4")` **手动**注册 5 个 tiling key（注释自承 "The kernel dispatches internally"），而非由 `ASCENDC_TPL_SEL` 自动枚举 sub-kernel；`op_host/euclidean_norm_def.cpp` 的 `TilingFunc` 与 `FillEmptyTilingData` 均**未调用 `SetTilingKey`**（注释明示 "tilingKey not set (defaults to 0)"）。

构建能通过，但 `tests/aclnn/run.sh` 跑出 `73 pass / 0 fail / 7 skip / 80 total`，其中 7 个 skip 全是 Group+tail-A 形状上的 special-value 用例（NaN / all-zero / FP16 overflow / ±inf / mixed NaN+inf / all-negative），harness 标注 `SKIP (no golden)`。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（op_kernel + op_host + CMakeLists 全链路）；kernel 入口 / tiling 注册 / host `SetTilingKey` 三层联动。
- **API 路径**：`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_BOOL_DECL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_BOOL_SEL`（`ascendc/host_api/tiling/template_argument.h`，CMake 构建期解析）；host 侧 `context->SetTilingKey(GET_TPL_TILING_KEY(...))`；CMake `npu_op_kernel_sources` / `npu_op_kernel_options` / `npu_op_kernel_library`。
- **验证过的平台与 CANN 版本**：Ascend950（proto.md `AddConfig("ascend950")`，kernel 安装路径 `opp/vendors/EuclideanNorm/.../kernel/ascend950/`），CANN 9.1.0；本机 `build.sh` 退出 0、`run.sh` 实际执行。
- **适用粒度**：`operator_family` —— 同一类「N bool 模板参数 + 多 tilingKey 编译期分发」的多分支归约/规约类算子（scale_custom、softplus_v2_grad、adam_apply_one_assign 等同族参考实现均用同一机制）。
- **路线优先级**：凡规范/DESIGN 明确 `ASCENDC_TPL_SEL` + `GET_TPL_TILING_KEY` 的算子，必须用编译期模板分发；`--tiling_key=` 手动注册 + 运行时 if/else 分发是 `banned` 的偏离形态（见 `## 坏实践`）。
- **不适用或未验证场景**：非归约/非多分支类、单 sub-kernel 算子无需 TPL 机制；a2/a3 等其他 SoC 未在本 trace 实测，推断可迁移但未验证（见 `## 失效边界`）。

## 判别方法

### 触发（tool_observed）

Ascend950 / CANN 9.1.0 上 `bash run.sh`（sourced CANN env）输出 `==== ST summary: 73 pass / 0 fail / 7 skip / 0 err / 80 total ====`，7 个 skip 集中在 Group+tail-A 形状的 special-value 用例（`b4_29_nan_fp32` / `b4_30_all_zero_fp32` / `b4_31_fp16_overflow` / `b4_32_pos_inf_fp32` / `b4_33_neg_inf_fp32` / `b4_34_nan_plus_inf_fp32` / `b4_35_negative_fp32`），逐条标注 `SKIP (no golden)`。注意 `0 fail`——这些用例不是断言失败，而是被 harness 静默跳过，故不会触发常规运行时错误码。

### 可执行探针

1. **读 kernel 入口签名**：打开 `op_kernel/euclidean_norm_apt.cpp`，看 kernel 函数是 `template <bool ...> __global__ __aicore__ void euclidean_norm(...)` 还是 `extern "C" __global__ __aicore__ void euclidean_norm(...)`。后者即偏离规范的运行时分发路线。
2. **看末尾兜底**：在 kernel 入口末尾搜 `AscendC::PipeBarrier<PIPE_ALL>();`。若存在且其前是 `if/else` 链，未命中分支即 fall-through 到此 no-op——这是 silent skip 的直接代码征兆。
3. **查 TPL 宏**：`grep -n "ASCENDC_TPL_ARGS_DECL\|ASCENDC_TPL_SEL" op_kernel/arch35/euclidean_norm_tiling_struct.h`。无命中说明 sub-kernel 未由 TPL 声明。
4. **查 CMake 手动注册**：`grep -n "tiling_key" CMakeLists.txt`。若出现 `npu_op_kernel_options(... OPTIONS "--tiling_key=...")`，说明 tilingKey 被手动注册指向同一 binary entry，绕过了 `ASCENDC_TPL_SEL` 自动枚举。
5. **查 host 是否下发 key**：`grep -n "SetTilingKey\|GET_TPL_TILING_KEY" op_host/euclidean_norm_def.cpp`。若 `TilingFunc` / `FillEmptyTilingData` 均无命中（或注释自承 "tilingKey not set"），说明路由权威被塞进 kernel 内部 if/else，与 `DESIGN §5.4.10` 的 host `SetTilingKey` 职责相反。
6. **构建后核对 sub-kernel 数**：`python3 .../inspect_build.py <package>`，看每个 `.o` 的 sub-kernel 数是否等于 `ASCENDC_TPL_SEL` 行数。运行时路线只有一个 binary entry，sub-kernel 数远少于预期。

### 原始观测（tool_observed）

- 探针 1/2 命中：kernel 入口为非模板 `extern "C"`，内部三层 `if (axisNum==0) / else if (Group) / else if (Normal)` 之后 fall through 到 `AscendC::PipeBarrier<PIPE_ALL>();`（no-op）。该 before-state 代码在本轨迹中以非 redacted 的文件读取 tool_result 逐字保留。
- 探针 3 命中：`euclidean_norm_tiling_struct.h` 仅为 `#include "euclidean_norm_tiling_data.h"` 的 shim，无 `ASCENDC_TPL_*` 宏。
- 探针 4 命中：`CMakeLists.txt` 含 `npu_op_kernel_options(ascendc_kernels EuclideanNorm OPTIONS "--tiling_key=0,1,2,3,4")`，注释自承 "The kernel dispatches internally" / "Branch-2/3/4 fall through to PipeBarrier stub"。
- 探针 5 命中：`euclidean_norm_def.cpp` 的 `TilingFunc` 有 `ChooseTilingKey(ctx)` 校验但**不下发 key**，注释 "tilingKey is NOT set on the framework"；`FillEmptyTilingData` 同样不 `SetTilingKey`。
- 触发信号：`run.sh` 7 个 Group+tail-A special-value 用例 `SKIP (no golden)`，`0 fail / 0 err`。

### 解释（interpretation，根因分析）

规范要求的编译期路线与实现的运行时路线有三处关键差别，第三处即 silent skip 的根因：

1. **sub-kernel 枚举来源不同**。编译期路线由 `ASCENDC_TPL_SEL` 每行 `ASCENDC_TPL_ARGS_SEL(...)` 在构建期生成一个 sub-kernel（`.o` 内 sub-kernel 数 = `ASCENDC_TPL_SEL` 行数）；运行时路线只有一个 `extern "C"` entry，`--tiling_key=0,1,2,3,4` 仅在框架层注册 5 个 key 指向同一 binary entry，kernel 内部自己 if/else——`ASCENDC_TPL_SEL` 完全缺失。
2. **路由权威错位**。规范要求 host 用 `GET_TPL_TILING_KEY(templateType, isEmptyTensor, isTailR)` 生成 key 并 `SetTilingKey` 下发，kernel 按模板参数 `if constexpr` 编译期分叉；运行时路线把路由权威塞进 kernel 自身的 `axisNum/usedCoreNum/rGroupCnt` 判定，host 反而不 `SetTilingKey`，二者职责与 DESIGN §5.4.10/§5.5.1 相反。
3. **未匹配 fall-through 是 silent no-op**（根因）。运行时路线的 `PipeBarrier<PIPE_ALL>()` 兜底使任何不命中三个 `if` 分支的输入（如某些 Group+tail-A 特殊值形状）直接 no-op 返回，kernel 不产生有效输出，harness 以 "no golden" 跳过——这**掩盖了路由未覆盖**，而非真正缺失参考数据。因此表现为 `7 skip / 0 fail`：用例既不通过也不失败，而是被静默吞掉。

判别结论：凡规范规定 `ASCENDC_TPL_SEL` + `GET_TPL_TILING_KEY` 的多分支算子，若 kernel 入口改走 `extern "C"` + 运行时 if/else + `PipeBarrier` fall-through，未命中分支会静默 no-op，在 special-value 形状上表现为 harness `SKIP (no golden)` 而非失败。诊断方向是改实现侧路由（模板化入口 + `ASCENDC_TPL_SEL` + host `SetTilingKey`），而非补 golden 文件。

## 验证信号

来源：本次开发会话对 `euclidean_norm_package` 的 `build.sh` + `inspect_build.py` + `tests/aclnn/run.sh` 实测（Ascend950 / CANN 9.1.0）。

- 修复前 `bash run.sh`：`==== ST summary: 73 pass / 0 fail / 7 skip / 0 err / 80 total ====`，7 skip 为 `b4_29_nan_fp32` / `b4_30_all_zero_fp32` / `b4_31_fp16_overflow` / `b4_32_pos_inf_fp32` / `b4_33_neg_inf_fp32` / `b4_34_nan_plus_inf_fp32` / `b4_35_negative_fp32`，逐条 `SKIP (no golden)`（来源平台：950；证据等级：tool_observed）
- 修复前 kernel 入口 `euclidean_norm_apt.cpp` 为非模板 `extern "C"` + 运行时 if/else + 末尾 `AscendC::PipeBarrier<PIPE_ALL>()` fall-through no-op 兜底（before-state 代码以非 redacted 文件读取 tool_result 逐字保留）（来源平台：950；证据等级：tool_observed）
- `tiling_struct.h` 修复前为 include shim 无 `ASCENDC_TPL_*` 宏；`CMakeLists.txt` 含 `npu_op_kernel_options(... OPTIONS "--tiling_key=0,1,2,3,4")` 手动注册；`op_host/euclidean_norm_def.cpp` 的 `TilingFunc`/`FillEmptyTilingData` 未调用 `SetTilingKey`（来源平台：950；证据等级：artifact_verified）
- 对齐规范（模板化入口 + `ASCENDC_TPL_SEL` + host `SetTilingKey(GET_TPL_TILING_KEY)` + 删 `--tiling_key`）后 `bash run.sh`：`==== ST summary: 80 pass / 0 fail / 0 skip / 0 err / 80 total ====`，原 7 skip 全部转 pass（来源平台：950；证据等级：tool_observed）
- 修复后 `inspect_build.py`：`dtype combinations: 3`、`.o files found: 3`、`Total sub-kernels: 15`、`✅ Pipeline status: COMPLETE (3/3 .o files)`，sub-kernel 数 = 外层 dtype(3) × 内层 `ASCENDC_TPL_SEL` 行数(5)（来源平台：950；证据等级：tool_observed）

## 失效边界

- **a2 / a3 等其他 SoC**：本 trace 仅在 Ascend950 实测（build + run.sh）。`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` / `GET_TPL_TILING_KEY` 是平台无关的构建期宏机制，推断在 a2/a3 可迁移，但**未验证，推断可迁移**，迁移前须在目标平台重跑 `build.sh` + `inspect_build.py`（确认 sub-kernel 数）+ `run.sh`。
- **单 sub-kernel 算子**：若算子只有一个分支（无 TPL 分叉），`ASCENDC_TPL_SEL` 只需 1 行，本卡的 5 行结构与 `ChooseTilingKey` 映射不适用。
- **非归约/非多分支类算子**：本卡的 (templateType, isEmptyTensor, isTailR) 三 bool 语义是 euclidean_norm 的分支模型；其他算子的 TPL 参数语义不同（如 scale_custom 用 `RANK` int 模板参数 4/8），`GET_TPL_TILING_KEY` 的参数须按各自 DESIGN 重新映射，不能照搬。
- **`ChooseTilingKey` 与 `GET_TPL_TILING_KEY` 必须自洽**：若 host 改 `ChooseTilingKey` 返回值顺序而不改 `GET_TPL_TILING_KEY` 的 (templateType, isEmptyTensor, isTailR) 合成，会下发错误 key 路由到错误 sub-kernel。
- **本卡不证实 "no-op 命中" 的单点复现**：因果链由 before/after 代码 diff + `73→80` 信号联合支撑，未单独抓取修复前 7 个 skip 用例的 kernel 执行日志以坐实其确实命中 `PipeBarrier` no-op 分支。

## 预防

- 写多分支归约/规约算子前，先读规范是否规定 `template <bool ...>` 入口与 `ASCENDC_TPL_SEL`；若规定，必须三层联动：`*_struct.h` 声明 TPL → `*_apt.cpp` 模板 + `if constexpr` → `*_def.cpp` `SetTilingKey(GET_TPL_TILING_KEY(...))` → `CMakeLists.txt` 删 `--tiling_key` 手动注册。
- 不要用 `PipeBarrier<PIPE_ALL>()` no-op 作为 kernel 入口的 fall-through 兜底——它会掩盖路由未覆盖，表现为 harness "no golden" skip 而非真实失败。
- 若 `inspect_build.py` 报 sub-kernel 数 < `ASCENDC_TPL_SEL` 行数，说明 TPL 枚举没生效（多为 `--tiling_key` 手动注册未删，或 `*_struct.h` 未 include `template_argument.h`）。

## 坏实践

- `CMakeLists.txt` 用 `npu_op_kernel_options(... OPTIONS "--tiling_key=0,1,2,3,4")` 绕过 `ASCENDC_TPL_SEL`，注释自承 "The kernel dispatches internally" / "Branch-2/3/4 fall through to PipeBarrier stub"——把编译期分发降级为运行时 if/else 的标志性坏实践。
- `*_def.cpp` 注释 "tilingKey not set (defaults to 0)" 同时存在 `ChooseTilingKey` 校验却不下发 key——host 把路由权威交给 kernel 内部 `if/else`，与 DESIGN §5.4.10 的 `SetTilingKey(GET_TPL_TILING_KEY(...))` 职责相反。
- 把 7 个 `SKIP (no golden)` 当成"测试侧缺 golden 文件"去补 golden——会落入反投机取巧的"改测试/补期望"陷阱；正确路径是改实现侧路由，7 个 skip 自然转 pass。