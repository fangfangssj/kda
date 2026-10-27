---
type: "Runbook"
title: "ACLNN 507035 — opType 与内置 ops_legacy 同名碰"
description: "CANN C++ 运行时 AppendDynBin/HashAndInsert 在同名时优先加载 built-in ops_legacy kernel binary 而非 vendor，导致 vendor TilingData 与内置 kernel 结构不匹配触发 vector core exception 507035；修复为 opType 重命名加 Custom 后缀。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-18T03:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["built_in_collision"]
---

# ACLNN 507035 — opType 与内置 ops_legacy 同名碰

## 现象

在 Ascend950PR 上以 ops-registry-invoke 方式开发 `softplus_v2_grad` 算子（ACLNN 接口 `aclnnSoftplusBackward`）。联调验证阶段，通过 ST 测试工程调用 ACLNN 两段式接口把算子下发到 NPU 执行时，38 个用例中 34 个失败，报错 `ret=507035`（stream sync failure = vector core exception）。编译通过、tiling 注册正常，仅在 NPU 实际执行 kernel 时崩溃。

设置 `ASCEND_GLOBAL_LOG_LEVEL=0` + plog 后，在 `/root/ascend/log/debug/plog/plog-180999_20260706093138574.log` 中观察到关键证据链：
- Line 1382-1383：框架先查 built-in 的 op desc，`[WARNING] Get op SoftplusV2Grad json info failed!` / `[WARNING] No builtin op desc info [SoftplusV2Grad]`
- Line 1390：但随后框架从 **built-in ops_legacy** 路径找到了 kernel binary：`End for modify resultJsonPath ascend950/ops_legacy/softplus_v2_grad/SoftplusV2Grad_a813aece931d655069c8bb16e60978cd_high_performance.json`
- Line 1399：最终加载的 binary 路径是 built-in 而非 vendor：`bin file path[/home/developer/Ascend/cann-9.0.0-beta.2/opp/built-in/op_impl/ai_core/tbe//kernel/ascend950/ops_legacy/softplus_v2_grad/SoftplusV2Grad_a813..._high_performance.o]`
- Line 52567：`retcode:507035 (vector core exception)`

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（vendor 包安装到 `opp/vendors/<vendor_name>/`），通过 ACLNN 两段式接口在 NPU 上联调。
- **API 路径**：AscendCL `aclnnSoftplusBackward` → CANN 运行时算子调度器 → opType 名查找 kernel binary。
- **验证平台与版本**：Ascend950PR（deviceId=0, Health=OK, HBM 5407/131072 MB），CANN cann-9.0.0-beta.2。
- **适用粒度**：`operator_family` — 任何与 CANN built-in `ops_legacy` 同名注册的 vendor 算子在 ACLNN 联调时均可能命中。
- **路线优先级**：`recommended_fastpath`（重命名 opType 是当前环境下唯一可靠的规避方式；卸载 built-in 是破坏性操作不宜在生产部署执行）。
- **不适用或未验证场景**：若目标环境 built-in 中不存在同名算子，重命名非必要；通过 ops-test-kit（Python 侧）走 `registries.py` 主动注册的路径不触发此问题（ops-test-kit 的 custom>vendor>builtin 优先级与 C++ 运行时相反，见根因）。a2/a3 平台未验证，推断可迁移但需确认该平台 built-in 是否同样注册了同名 legacy 算子。

## 根因

CANN C++ 运行时（`op_kernel.cpp` 的 `AppendDynBin` / `HashAndInsert` 方法）在按 opType 名查找 kernel binary 时，**built-in 路径优先于 vendor 路径**。这与 ops-test-kit（Python 侧）的优先级策略相反——ops-test-kit 通过 `_env.py` 的 `_get_vendor_impl_parent_paths` 读取 `vendors/config.ini` 的 `load_priority`，按 custom → vendor → builtin 顺序 prepend PYTHONPATH，`registries.py` 的 `load_op_registries` 按同顺序注册 tiling SO（first-registered-wins），`OpInfoKeeper`/`OpApiInfoKeeper` 都是 built-in 先入 dict、vendor 后入覆盖。

但 ST 测试工程直接调 `aclnnSoftplusBackward` 走 CANN C++ 运行时，后者是否遵循 `config.ini` 的 `load_priority` 取决于 CANN 框架自身实现——实测结果是 built-in legacy 优先命中。框架加载了 built-in `ops_legacy/softplus_v2_grad` 的 TBE 旧实现 kernel binary，但 TilingData 是 vendor 包的自定义结构，两者不匹配 → vector core exception 507035。

## 解决配方

将 opType 从 `SoftplusV2Grad` 重命名为 `SoftplusV2GradCustom`，涉及 6 处注册点同步修改。ACLNN 接口名 `aclnnSoftplusBackward` 保持不变（用户无感知），仅算子内部注册类型名改为 `SoftplusV2GradCustom` 以彻底避免碰撞。

before（修复前，opType 与 built-in 同名，6 处均为 `SoftplusV2Grad`）：

```cpp
// op_host/softplus_v2_grad_def.cpp
class SoftplusV2Grad : public OpDef {
    explicit SoftplusV2Grad(const char* name) : OpDef(name)
OP_ADD(SoftplusV2Grad);

// op_host/softplus_v2_grad_infershape.cpp
IMPL_OP_INFERSHAPE(SoftplusV2Grad)

// op_host/arch35/softplus_v2_grad_tiling.cpp
IMPL_OP_OPTILING(SoftplusV2Grad).Tiling(TilingFuncSoftplusV2Grad).TilingParse<SoftplusV2GradCompileInfo>(TilingPrepareForSoftplusV2Grad);

// op_api/softplus_backward.cpp
OP_TYPE_REGISTER(SoftplusV2Grad);
auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftplusV2Grad, ...);

// op_kernel/arch35/softplus_v2_grad_struct.h
ASCENDC_TPL_ARGS_DECL(SoftplusV2Grad, ...);

// op_kernel/CMakeLists.txt
        OP_TYPE SoftplusV2Grad
```

after（修复后，6 处同步改为 `SoftplusV2GradCustom`）：

```cpp
// op_host/softplus_v2_grad_def.cpp
class SoftplusV2GradCustom : public OpDef {
    explicit SoftplusV2GradCustom(const char* name) : OpDef(name)
OP_ADD(SoftplusV2GradCustom);

// op_host/softplus_v2_grad_infershape.cpp
IMPL_OP_INFERSHAPE(SoftplusV2GradCustom)

// op_host/arch35/softplus_v2_grad_tiling.cpp
IMPL_OP_OPTILING(SoftplusV2GradCustom).Tiling(TilingFuncSoftplusV2Grad).TilingParse<SoftplusV2GradCompileInfo>(TilingPrepareForSoftplusV2Grad);

// op_api/softplus_backward.cpp
OP_TYPE_REGISTER(SoftplusV2GradCustom);
auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftplusV2GradCustom, ...);

// op_kernel/arch35/softplus_v2_grad_struct.h
ASCENDC_TPL_ARGS_DECL(SoftplusV2GradCustom, ...);

// op_kernel/CMakeLists.txt
        OP_TYPE SoftplusV2GradCustom
```

同时 kernel 入口函数名也需同步：`op_kernel/softplus_v2_grad_apt.cpp` 中 `__global__ __aicore__ void softplus_v2_grad(...)` 改为 `softplus_v2_grad_custom(...)`，因为 CANN 编译系统的 `dfx_for_func_name` 期望 kernel 函数名为 OP_TYPE 的 snake_case 形式（`SoftplusV2GradCustom` → `softplus_v2_grad_custom`），否则报 `kernel entry 'softplus_v2_grad' not implement in 'softplus_v2_grad_apt.cpp'`。

## 验证信号

本卡验证来自 opType 碰撞复现实验（改回 `SoftplusV2Grad` 复现 → 改回 `SoftplusV2GradCustom` 验证），证据取自 plog 文件与 ST 日志：

- 修复前 ST：34/38 用例 `ret=507035` 失败（来源平台：Ascend950；证据等级: tool_observed）
- 修复前 plog Line 1399：`bin file path` 指向 `built-in/.../ops_legacy/softplus_v2_grad/SoftplusV2Grad_a813..._high_performance.o`（来源平台：Ascend950；证据等级: tool_observed）
- 修复后 ST：38/38 用例 PASS（来源平台：Ascend950；证据等级: tool_observed）
- 修复后 plog Line 1374：`opFile` 从 `softplus_v2_grad.json`（built-in 命名）变为 `softplus_v2_grad_apt.json`（vendor 命名）；`bin file path` 指向 `vendors/softplus_v2_grad_custom/.../softplus_v2_grad_apt/SoftplusV2Grad_9ef9...json`（来源平台：Ascend950；证据等级: tool_observed）
- 修复后 plog：507035 错误 0 次；`HashAndInsert` 使用 vendor binary（来源平台：Ascend950；证据等级: tool_observed）

## 系统信息

- **工具链**：CANN cann-9.0.0-beta.2，`bash build.sh --soc=ascend950` 编译。
- **目标平台**：Ascend950PR（`npu-smi info` 确认 deviceId=0, Health=OK）。
- **vendor 安装位置**：`${ASCEND_HOME_PATH}/opp/vendors/softplus_v2_grad_custom/`，已注册到 `config.ini` 的 `load_priority`。
- **built-in 碰撞源**：`opp/built-in/.../config/ascend950/aic-ascend950-ops-info-legacy.json` 注册了内置 `SoftplusV2Grad`（TBE 旧算子），同目录 `ops_legacy/dynamic/softplus_v2_grad.py`。
- **相关文件**：`op_host/softplus_v2_grad_def.cpp`、`op_host/softplus_v2_grad_infershape.cpp`、`op_host/arch35/softplus_v2_grad_tiling.cpp`、`op_api/softplus_backward.cpp`、`op_kernel/arch35/softplus_v2_grad_struct.h`、`op_kernel/CMakeLists.txt`、`op_kernel/softplus_v2_grad_apt.cpp`。

## 预防

在新建 vendor 算子前，先检查 `opp/built-in/.../config/<soc>/aic-<soc>-ops-info-legacy.json` 与 `aic-<soc>-ops-info.json` 中是否已存在同名 opType。若存在，应在初始注册时即用 `<OpName>Custom` 后缀或其他唯一名，避免到联调阶段才发现碰撞。

## 设计决策

- **为何选择重命名 opType 而非卸载 built-in**：临时移除 built-in `ops-info-legacy.json` 条目或移动 `ops_legacy/softplus_v2_grad` 目录虽然能让 vendor 命中，但这是破坏性操作，会污染 CANN 安装环境，不宜在生产部署中执行。重命名 opType 是自包含、可移植的规避方式。
- **为何 ACLNN 接口名保持不变**：`aclnnSoftplusBackward` 是面向用户的接口名，与 opType 内部注册名解耦；改名仅影响算子内部注册，用户无感知。
- **被放弃的方案**：在 ST `run.sh` 中设置 `ASCEND_CUSTOM_OPP_PATH` 指向 vendor 目录——理论上应让 C++ 运行时优先查 vendor，但实测 C++ 运行时与 Python 侧 ops-test-kit 的优先级机制不同，仅设 `LD_LIBRARY_PATH`（op_api/lib 层）不足以影响 kernel binary 查找路径。
- **当前方案属哪档路线**：`recommended_fastpath`——在 built-in 同名算子无法卸载的环境下，重命名是唯一可靠规避。

## 失效边界

- a2 与 a3 平台未验证：若该平台 built-in `ops-info-legacy.json` 中未注册同名 `SoftplusV2Grad`，则重命名非必要（未验证，推断可迁移）。
- 若未来 CANN 版本修复了 C++ 运行时的 vendor 优先级（使 `config.ini` 的 `load_priority` 在 C++ 侧也生效），则重命名可回退为原名。

## 坏实践

- 在联调时为快速通过而直接移除 built-in `ops-info-legacy.json` 条目或移动 `ops_legacy/` 目录——这会污染 CANN 安装环境，且复现后忘记恢复会导致其他依赖 built-in 的流程出错。

## 误诊

初始诊断曾怀疑"vendor 包安装不完整、binary 未正确生成、或 tiling SO 注册失败"。plog 复现后证实 built-in 确实被优先加载（`bin file path` 指向 built-in），根因是 C++ 运行时优先级策略，非 vendor 安装问题。