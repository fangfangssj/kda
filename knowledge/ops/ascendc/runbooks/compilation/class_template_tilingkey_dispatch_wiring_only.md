---
type: "Runbook"
title: 模板 kernel + tilingKey 派发：分支启用只改 apt 接线
description: 多 tilingKey 分支算子的 kernel 工程模式：单一类模板 EuclideanNormKernel<DType, isTailR> 用 if constexpr(isTailR) 内联特化分支差异，tail-R 分支启用时 kernel 代码零改动，只在 apt.cpp 把 TILING_KEY_IS(1) 从桩改为模板实例 Init+Process，18/18 上板通过且 tail-A 无回归。复现增量：正交多维推广——2 个 kernel 模板类（half/interleaved 计算链）× 2 个正交布尔参数（kIsBnsd 坐标反算 × kCos3d 源基址）覆盖 8 个 tilingKey 分支，新分支 = 分发链加一个 else-if 实例化；差异点集中在坐标反算与 NDDMA 源基址两处，未接入键落 else PipeBarrier 静默空跑。
tags: [task.implementation, topic.implementation.operator_registration, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# 模板 kernel + tilingKey 派发：分支启用只改 apt 接线

## 现象

EuclideanNorm（L2 范数 reduce）按合轴 pattern 编码 5 个 tilingKey（normal tail-A=0 / normal tail-R=1 / empty=2 / group tail-A=4 / group tail-R=5），每个 key 对应一份编译产物。开发采用 TDD 分支推进：先实现 tail-A 分支（key=0）kernel 时，就把 kernel 写成单一类模板 `EuclideanNormKernel<DType, isTailR>`，所有分支差异（CopyIn 尾块、二分缓存树、CopyOut 路径）用 `if constexpr(isTailR)` 在紧内循环内联特化。到 tail-R 分支（key=1）启用时，kernel 文件 1175 行零改动——ST 基线 0/18 全 0（key=1 在 apt.cpp 里还是桩），只把派发处从 `(void)tilingData;` 改成模板实例化 + `Init/Process` 两行，重编后 18/18 用例上板全过，tail-A 5/5 无回归。

## 适用场景

- 工程形态：多 tilingKey 分支的 AscendC 自定义算子（同算子多 pattern/特化分支，逐分支 TDD 推进）。
- API 路径：`op_kernel/<op>_apt.cpp` 的 `REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY` / `GET_TILING_DATA_WITH_STRUCT` / `TILING_KEY_IS(n)` 派发宏 + kernel 类模板（`__aicore__ inline` 成员、`if constexpr` 编译期分支）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（dav-3510 连续 Vector 计算链用 `__simd_vf__` + `asc_vf_call` 实现）。
- 适用粒度：operator_family——"多分支共用一个 kernel 类、按 tilingKey 派发"的自定义算子工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：分支间数据流差异大到无法共用类骨架（如 group 的 A×R 二维分核 + Phase 2 mini-kernel，本轨迹中 group 仍为桩，是否复用同类待后续验证）；dtype 差异走 `DTYPE_X` 模板实例化（fp16/bf16/fp32/int32 共 4 份），与该模式正交。

## 解决配方

**先写"一般"再启用分支**：kernel 类模板以 `isTailR` 为编译期参数，分支差异内联在紧内循环：

```cpp
// op_kernel/arch35/euclidean_norm.h（既有结构，本阶段零改动）
// class EuclideanNormKernel { ... };
//   __aicore__ inline void Process();
//   __aicore__ inline void ProcessGroup();
// 分支差异示例（紧内循环编译期特化）：
//   if constexpr (isTailR) { ... }   // CopyIn 尾块 / 缓存树 / CopyOut 路径 1
```

apt 派发链（启用前，key=1 是桩）：

```cpp
extern "C" __global__ __aicore__ void euclidean_norm(
    GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();

    REGISTER_TILING_DEFAULT(EuclideanNormTilingData);
    if (TILING_KEY_IS(0)) {
        // Normal Tail-A (DESIGN §5.5.1 else 分支, isTailR=false)
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 0", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormKernel<DTYPE_X, /*isTailR=*/false> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;   // Task 28: Normal Tail-R kernel
    } else if (TILING_KEY_IS(2)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;   // Task 32: EMPTY kernel
    } else if (TILING_KEY_IS(4)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;   // Task 36: Group Tail-A kernel
    } else if (TILING_KEY_IS(5)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;   // Task 40: Group Tail-R kernel
    }

    (void)axes;
    (void)workspace;
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

启用 tail-R 分支（唯一的代码改动——两行换成模板实例）：

```cpp
    } else if (TILING_KEY_IS(1)) {
        // Normal Tail-R (DESIGN §5.5.1 else 分支, isTailR=true, AR pattern)
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormKernel<DTYPE_X, /*isTailR=*/true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
```

（注释块同步更新派发表：key 1 从 "stub" 改为 "EuclideanNormKernel<DTYPE_X, true> Init + Process"。）

## 系统信息

- 平台/工具链：Ascend950PR（arch35 / dav-3510），CANN 9.1.0；ST 用 npu-smi 确认设备在线后经 aclnn 实跑。
- 相关文件：`op_kernel/euclidean_norm_apt.cpp`（派发）、`op_kernel/arch35/euclidean_norm.h`（类模板，1175 行，含 10 处 `__simd_vf__` + 10 处 `asc_vf_call`）、`op_kernel/arch35/euclidean_norm_tiling_key.h` 同源的模板参数声明（templateType/isEmptyTensor/isTailR 三 bool → 5 组合 × 4 dtype = 20 份 binary）、`tests/aclnn/test_aclnn_branch1.cpp`（20 用例 ST，2 个 key=0 用例由独立 oracle 标 DEFERRED）。
- 验证命令：`operators/build.sh euclidean_norm_package` → `tests/aclnn/run.sh`（或直接跑 `build/test_aclnn_branch1`，注意 run.sh 里该二进制带 `|| true`，须直查其退出码）。

## 设计决策

- **单类模板 + `if constexpr` 而非每分支一个类**：tail-A/tail-R 的 CopyIn→CastSquare→缓存树→ReduceSum→PostElewise→CopyOut 骨架完全同构，差异只在尾块与 CopyOut 路径；`if constexpr` 让两份实例在编译期各自展开、零运行时分支开销，后启用分支时 kernel 侧零改动、零回归面。
- **逐分支 TDD 的红利**：ST 用例先行（key=1 用例 0/18 红基线锁定期望），接线后 18/18 直接绿——若 kernel 与接线同时新写，失败时无法区分是派发错还是 kernel 错。
- **派发处保留每个 key 的 `REGISTER_TILING_FOR_TILINGKEY` + `GET_TILING_DATA_WITH_STRUCT`**：key 与 TilingData struct 的注册关系是按 key 声明的，桩分支也保留这两行，启用时 diff 最小。
- 放弃的备选：每分支独立 kernel 类 + 各自 CopyIn/Compute/CopyOut——代码量翻倍且骨架漂移风险大；运行时 if/else 分支（非编译期）——紧内循环每次迭代的分支开销不可接受。

## 验证信号

验证来自 tail-R 分支启用闭环（Ascend950PR，CANN 9.1.0）：

- 接线前基线：Branch-1 ST key=1 精度 **0/18**，全部 `dev=0.000000`（桩不计算）（来源平台：950；证据等级：tool_observed）。
- 接线后：Branch-1 ST key=1 精度 **18/18 PASSED**（maxRel≤8.467e-08，覆盖 fp16/bf16/fp32/int32-bitwise、int32/int64 axes dtype、`[1024] axes=[]` full reduce、`[17] axes=[-1]` 非对齐尾块；2 个真实 key=0 用例由独立 oracle 正确 DEFERRED）（来源平台：950；证据等级：tool_observed）。
- 回归：Branch-0 ST key=0 精度 5/5 PASSED、两个测试二进制退出码均 0；TilingUT 全绿 19/19 + 9/9 + 9/9；`grep -c '__simd_vf__'`/`grep -c 'asc_vf_call'` 于 `op_kernel/arch35/euclidean_norm.h` 各 10（≥1 硬标准）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 模式成立的前提是分支差异能被 `if constexpr` 内联表达（同构数据流）；数据流异构的分支（本轨迹的 group 二维分核即待验证案例）可能仍需独立类。
- `DTYPE_X` 由 aclnn 层 dtype 决定实例化份数（本算子 4 dtype × 5 key 组合 = 20 份 binary），分支/dtype 组合多时编译产物体积线性增长。
- 桩分支 `(void)tilingData;` 阶段调用该 key 会得到全 0 输出（无报错）——上板前必须确认所有 key 都已接线或调用方不触达未接线 key。

**未决项**：

- group 分支（key=4/5，A×R 二维分核 + Phase 2 RA mini-kernel）是否能继续复用同一类模板未验证（其数据流差异更大）。
- empty 分支（key=2）的 kernel 接线在本轨迹该阶段仍为桩，待后续启用。
## 坏实践

- run.sh 对分支测试二进制统一 `|| true` 兜底（测试先行阶段的权宜），启用后若不直查退出码会掩盖真实失败——验收时应直接运行二进制并检查 `$?`（本轨迹验收即如此兜底核查）。

## 增量（正交多维推广：2 个模板类 × 2 个布尔参数覆盖 8 分支）

另一算子族（RoPE 类 BenchHarnessApplyRotaryPosEmb，Ascend950PR + CANN 9.2.0，`extern "C"` 入口 + `GET_TILING_DATA` 后按 `tilingData.tilingKey` 位编码分发，regbase 单 binary 范式）把"单一类模板 + if constexpr"推广到组合分支空间：

**结构**：8 个 tilingKey 分支 = rotaryMode（half/interleaved 两条计算链）× layout（BSND/BNSD）× cos.ndim（2d/3d）。实现只需 **2 个 kernel 模板类 + 每类 2 个正交布尔模板参数**：

- `Branch0Kernel<T, kIsBnsd, kCos3d>` 覆盖 half 系（key 0/2/4/6）；
- `Branch1Kernel<T, kIsBnsd, kCos3d>` 覆盖 interleaved 系（key 1/3/5/7）。

两个维度的差异点各自只落一处，**集中在两个切点而不是散布运行时 if**：

- `kIsBnsd` 切坐标反算：BSND `s=(row/N)%S`（N-broadcast）vs BNSD `s=row%S`（s-stride）；
- `kCos3d` 切 cos/sin NDDMA 源基址：cos2d `srcBase=s*Dh`（B 广播）vs cos3d `srcBase=(b*S+s)*Dh`（真实 B 轴）。

```cpp
// kIsBnsd=false (默认) → BSND: s=(row/N)%S；kIsBnsd=true → BNSD: s=row%S。
// kCos3d=false (默认) → cos2d: srcBase=s*Dh（B 广播）；kCos3d=true → cos3d: srcBase=(b*S+s)*Dh（B 轴真实数据不广播）。
// 其余计算链（rotate_half 分解、VF MulAdd 融合、升精度 Cast、CopyOut）各系内逐指令相同。
template <typename T, bool kIsBnsd = false, bool kCos3d = false>
class Branch0Kernel {
```

**新分支接入 = 分发链加一个 else-if 实例化**（保持"kernel 代码零改动、只改接线"的本卡主旨）：

```cpp
    } else if (tilingData.tilingKey == 7u) {
        // Branch-7 (interleaved-BNSD-cos3d): 复用 Branch1Kernel 计算链，
        // kIsBnsd=true && kCos3d=true 同时切换坐标反算 s=row%S 与 cos/sin 源基址 (b*S+s)*Dh。
        Branch1Kernel<DTYPE_QUERY, true, true> kernel;
        kernel.Init(query, key, cos, sin, queryOut, keyOut, &tilingData);
        kernel.Process();
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
```

**未接入键的静默空跑坑**：tilingKey 落入 `else`（仅 `PipeBarrier`）时编译/运行都不报错，输出不写——配合 ST 的输出毒化表现为全 `-nan` 红基线（判别与接线修复见库内 stub_branch_sentinel_output_entry_wiring 卡）；交付前必须补齐分发或确认调用方不触达未接线 key（与本卡失效边界第 3 条同源）。

**分发注释纪律**：分发注释必须写明"模板参数 ↔ 行为"映射与复用依据分支（kIsBnsd 控制 s 反算、kCos3d 控制 srcBase 真实 b 偏移），防止后人误改参数组合。

验证信号（复现增量，同算子 branch-2/3/4/5/6/7 六轮分支接入会话，ST 上板 Ascend950PR）：

- 每轮接入前红基线：新分支 ST numeric 0/11 全败（`actual=-nan`，毒化输出未被写入）、空张量用例 1/1 结构通过、前序分支各 12/12 无回归（来源平台：950；证据等级：tool_observed）。
- 每轮接入后：新分支 ST 12/12（numeric 11/11 + returns_empty 1/1）；branch-7 收官时八分支 combined 全 rc=0（来源平台：950；证据等级：tool_observed）。
- 改动量：branch-4（cos3d 首个分支）kernel 单文件 `38 insertions(+) / 9 deletions(-)`；branch-7 仅分发链一个 else-if + 类头注释（来源平台：950；证据等级：tool_observed）。
- UT 随分支增长 99→114→131→148 全绿（tiling 字段正确性独立验证）；`grep -cE '__simd_vf__|asc_vf_call'` 主 kernel 文件恒 18（计算链未稀释）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 正交性成立的前提是"布局维只改坐标反算/stride、编码维只改 cos/sin 源基址"这类可分解耦合；某变体改动计算链本身（不同数学公式或不同 buffer 数量/UB 预算）时须新增模板参数或新类，不能硬套。
- 坐标反算/源基址公式与 tiling 的 `cosSrcStrideDh`/`cosB` 取值必须配对正确——配错时 ST 以有限值错位（非 NaN）暴露，属另一类现象（配对语义见库内 rope_cos_sin_bsnd_bnsd_stride_pairing 卡）。

坏实践（复现增量）：

- 为每个 (mode × layout × cos.ndim) 组合各复制一个 kernel 类——组合数按维度乘积增长，回归面不可控（8 组合仅 2 类 + 2 参数覆盖）。
- 把 layout/cos.ndim 差异以运行时 if 散布进计算链代码——应集中到坐标反算与源基址两处并由模板参数编译期切换。