---
type: "Runbook"
title: "ACLNN L2 缓存键漏 scalar 致批跑误命中"
description: "DFX_IN 遗漏影响 TilingData 的 scalar 属性时，ACLNN L2 cache key 不含该属性，同 shape 不同属性值的调用复用旧 TilingData，出现\"单跑 PASS、批跑 FAIL\"的精度漂移。"
tags: ["task.debugging", "topic.precision", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-09-17T02:30:52Z'
platforms: ["950"]
---

# ACLNN L2 缓存键漏 scalar 致批跑误命中

## 现象

在调试 `softplus_v2_grad` 算子（arch35/DAV_3510，Ascend950PR）的 PyTorch ST 时，C++ ST 57/57 全过，但 PyTorch ST 仅 463/578 通过、115 失败。失败用例呈现一个明显模式：**同一用例单独跑通过、放进批量用例集里跑失败**，且失败数随用例顺序波动。设置环境变量 `DISABLE_L2_CACHE=1` 重跑，失败数从 115 降到 78，证明部分失败由 L2 缓存误命中导致。

检查 `op_api/aclnnSoftplusV2Grad.cpp` 第 123 行：

```cpp
L2_DFX_PHASE_1(aclnnSoftplusV2Grad, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
```

`DFX_IN` 只列了 `gradOutput`、`self` 两个 tensor，**漏掉了 `beta`、`threshold` 两个 scalar 属性**。而算子的 `OP_ATTR(...)` 里 `beta`、`threshold` 作为 scalar 传入，并参与 TilingData 计算。

## 适用场景

- **工程形态**：aclnn 算子仓开发（ops-registry-invoke），op_api L2 接口实现层。
- **API 路径**：`L2_DFX_PHASE_1` / `DFX_IN(...)` / `DFX_OUT(...)` 宏，以及 `OP_ATTR(...)` scalar 注册。
- **验证平台与版本**：Ascend950PR（soc=ascend950，arch35/DAV_3510），CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family`（所有通过 `OP_ATTR` 传 scalar 属性并影响 TilingData 的 aclnn L2 算子）。
- **路线优先级**：`recommended_fastpath`（补全 `DFX_IN` 保留缓存性能收益，优于直接关缓存）。
- **不适用或未验证场景**：无 scalar 属性的纯 tensor 算子（`DFX_IN` 只需列 tensor 时无此问题）；非 L2 aclnn 框架（如 kernel 直调）不经过此缓存。

## 根因

ACLNN L2 的 cache key = hash(**API 名 + op config + `DFX_IN(...)` 参数 + `DFX_OUT(...)` 参数**）。命中时直接返回已缓存的 `aclOpExecutor`（含已计算的 TilingData），**跳过 Tiling 函数**。**若 `DFX_IN` 遗漏了某个影响 TilingData 的 scalar（如 `beta`/`threshold`），该 scalar 不参与 cache key hash**：同 shape 不同属性值的调用命中同一缓存项、复用旧 TilingData，拿旧属性参与计算 → 结果错误。机制链：`OP_ATTR` 的 scalar 进入 TilingData → `DFX_IN` 漏 scalar → cache key 不含 scalar → 同 shape 异属性命中同缓存 → 复用旧 TilingData（旧 beta/threshold）→ 批跑时后序用例拿到前序用例的属性 → 精度漂移；单跑时缓存未污染故通过。

## 解决配方

**before**（`op_api/aclnnSoftplusV2Grad.cpp:123`，DFX_IN 漏 scalar）：

```cpp
L2_DFX_PHASE_1(aclnnSoftplusV2Grad, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
```

**after**（DFX_IN 补全所有影响 TilingData 的输入：tensor + scalar）：

```cpp
L2_DFX_PHASE_1(aclnnSoftplusV2Grad, DFX_IN(gradOutput, self, beta, threshold), DFX_OUT(gradInput));
```

硬性约束：`OP_ATTR(...)` 中的每个 scalar 必须同时在 L2 的 `DFX_IN(...)` 中出现；`DFX_IN` 应覆盖输入 tensor + scalar + 普通 C++ 类型参数的全部输入。

## 验证信号

- 初始 PyTorch ST 463/578 通过、115 失败（来源平台：950；证据等级：tool_observed）。
- `DISABLE_L2_CACHE=1` 后失败数 115→78（来源平台：950；证据等级：tool_observed）。
- `DFX_IN` 补全 `beta, threshold` 后重建重测：499/578 通过（463→499，+36），残留 79 失败为另一独立 kernel bug（来源平台：950；证据等级：tool_observed）。

## 预防

- 同类排查：凡 `OP_ATTR` 传 scalar 的 L2 接口，批量审查 `DFX_IN` 完整性——`DFX_IN` 列表必须与 `OP_ATTR` 的 scalar 一一对应。
- 调试单跑 PASS/批跑 FAIL 的精度漂移时，先用 `DISABLE_L2_CACHE=1` 快速确认是否缓存相关，再查 `DFX_IN` 完整性（这是 quick-confirm 手段，不是最终修复；最终修复仍是补全 `DFX_IN`）。

## 坏实践

- 用 `DISABLE_L2_CACHE=1` 作为"修复"部署到生产——这放弃了 L2 缓存的性能收益，且本例 `DISABLE_L2_CACHE` 在该运行时未必完全生效（78 失败仍存在说明还有独立 kernel bug）。正确做法是补全 `DFX_IN`。

## 失效边界

- Ascend950PR/arch35 实测；其他平台（a2/a3）走相同 ACLNN L2 框架，机制一致但未验证，推断可迁移。
- `DFX_IN` 完整性规则对所有走 L2_DFX_PHASE 的 aclnn 算子通用；非 aclnn 框架（kernel 直调）不经过此缓存。

## 复现增量：修复路线二（L2_DFX_PHASE_1_WITHOUT_CACHE）+ tiling 跳过探针 + 两层缓存关系

另一独立轨迹并入 cross entropy loss 算子（arch35/CANN 9.1.0）的实例：`reduction` 以 `const char*`/整型传入（非 scalar 张量属性），同 shape 的 mean 与 sum（输出同为标量 shape）缓存键相同，第二次调用直接复用第一次的 executor（含 tiling 结果与 TilingKey），**tiling 函数根本不会被执行**。相对本卡原实例（softplus_v2_grad 的 beta/threshold 为 OP_ATTR scalar，可补进 DFX_IN），本实例给出不可/不宜入 DFX_IN 时的替代修复路线与判别手段：

**1. 框架侧机制证据（CANN 9.1.0 头文件原文）**——`op_dfx.h` 两个宏与 `op_cache.h` 的 GetFromCache 键构成：

```cpp
#define L2_DFX_PHASE_1(APIName, IN, OUT)                                              \
    static_assert(op::ValidDfxName(#APIName "GetWorkspaceSize", __func__),            \
                  "Invalid DFX:" #APIName "GetWorkspaceSize");                        \
    if (CheckPhase1Params(executor, workspaceSize) != ACLNN_SUCCESS) {                \
        return ACLNN_ERR_PARAM_NULLPTR;                                               \
    }                                                                                 \
    InitL2Phase1Context(#APIName, executor);                                          \
    L2_OP_PROF_PHASE_1(#IN, #OUT, IN, OUT);                                           \
    do {                                                                              \
        if (op::internal::GetFromCache(executor, workspaceSize, #APIName, IN, OUT)) { \
            return ACLNN_SUCCESS;                                                     \
        }                                                                             \
    } while (false)
```

```cpp
template <typename INPUT_TUPLE = void*, typename OUTPUT_TUPLE = void*>
bool GetFromCache(aclOpExecutor** executor, uint64_t* workspaceSize, const char* api, const INPUT_TUPLE& in,
                  const OUTPUT_TUPLE& out)
{
    // Collecting Cache Data
    InitExecutorCacheThreadLocal();
    AddParamToBuf(api);
    AddOpConfigInfoToBuf();
    CalculateHashKey(in);
    CalculateHashKey(out);
    if (!CheckCacheable()) {
        return false;
    }
    return GetFromCache(executor, workspaceSize);
}
```

缓存键 = hash(API 名 + op config + IN/OUT 张量元组)，算子属性不在其中。

**2. tiling 跳过探针（区分 executor 缓存命中与 TilingKey 编码错误）**：在 tiling 函数末尾加 stderr 打印（`OP_LOGI` 在该场景不可见，须 `fprintf(stderr, ...)`），单进程重跑——失败的属性变体调用**没有打印** = tiling 未执行 = executor 缓存命中：

```cpp
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, isHardLabelVal, reductionMode);
    fprintf(stderr, "[TilingKeyDebug] reduction=%s mode=%u dtype=%u isHard=%u tilingKey=%lu\n",
            reductionStr, reductionMode, dTypeX, isHardLabelVal,
            static_cast<uint64_t>(dTypeX) | (static_cast<uint64_t>(isHardLabelVal) << 8) |
            (static_cast<uint64_t>(reductionMode) << 9));
```

**3. 修复路线二：整算子禁用执行缓存**（本卡原路线"补全 DFX_IN"保留缓存的替代，适用于非张量属性）：

```cpp
    // Disable executor cache: the cache key (L2_DFX_PHASE_1) is based on input/output
    // tensor properties only, NOT attributes. mean and sum produce the same scalar output
    // shape, so the cache would reuse the wrong tiling result (wrong TilingKey/reduction).
    // L2_DFX_PHASE_1_WITHOUT_CACHE ensures the tiling function is always called.
    L2_DFX_PHASE_1_WITHOUT_CACHE(aclnnBenchNoHarnessCrossEntropyLoss, DFX_IN(input, target), DFX_OUT(loss));
```

注意：`L2_DFX_PHASE_1_WITHOUT_CACHE` 宏体内**没有** `CheckPhase1Params(executor, workspaceSize)` 空指针检查，调用方需自带等价检查。代价：同参数重复调用失去 host 侧 executor/tiling 复用，tiling 函数每次重跑（本例 tiling 计算很轻，正确性优先）。备选被放弃的原因：把属性塞进 `DFX_IN(...)` 元组——`AddParamToBuf(const char*)` 重载虽能写入哈希缓冲，但 `IN` 同时被 `L2_OP_PROF_PHASE_1` 用于 profiling 打点，非张量参数兼容性未验证；宏调用前往哈希缓冲手动注入——`GetFromCache` 第一步 `InitExecutorCacheThreadLocal()` 会重置缓冲，外部注入被覆盖。

**4. 两层缓存的独立性与修复顺序**：本实例先修 TilingKey 编码（reduction 入键），单进程 ST 仍失败 19 例；探针显示失败的 sum 用例 tiling 根本没执行——TilingKey 修复必要但不充分，executor 级缓存是独立的一层（TilingKey 缓存层另见库内《TilingKey 缓存 correction 致 tilingdata 复用》卡）。误差指纹：sum-after-mean 的 MERE 呈 (N−1)/N（N=16 → 9.37e-01）。

**独立 trace 新增验证信号**（均为 tool_observed）：

- 修复前单进程 ST：105 用例 19 失败；sum 用例 MERE=9.37e-01（N=16）/9.69e-01（N=32），呈 (N−1)/N 指纹；1 例 ignore 变体流同步失败 507035（来源平台：950；证据等级：tool_observed）。
- 探针输出：`[TilingKeyDebug] reduction=none ... tilingKey=256`、`reduction=mean ... tilingKey=768` 逐调用打印，失败 sum 用例无打印（来源平台：950；证据等级：tool_observed）。
- 修复后（WITHOUT_CACHE）单进程 ST：105 用例 104 通过（唯一失败为空 tensor 边界，另卡）；与 TilingKey 修复联合后单进程 ST 105/105、UT 74/74；PyTorch 适配层同进程批量验收全部 reduction/ignore 用例通过（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：属性影响输出张量 shape/dtype/format 时不适用（缓存键含张量属性，天然不命中——none→向量输出与 mean/sum→标量的 shape 不同，none 模式基本不受影响）；多线程并发调用（GetFromCache 用 thread_local 哈希缓冲）与 GEIR 图路径未验证。坏实践补充：只修 TilingKey 不查 executor 缓存；用分进程执行测试长期"规避"（掩盖真实缺陷且显著拖慢测试）。

## 复现增量：五个新 trace——第三条规避路线 ACLNN_CACHE_LIMIT=0、float 属性漏序列化、线程局部缓存与新线程隔离

另一独立轨迹并入五个不同算子（resize 双线性、grid_sampler_3d、AdamW 类、指数族、conv2d 类 PyTorch 适配层）的独立复现，补齐本卡规避路线与机制边界：

**1. 触发指纹（批跑互踩的形态判别）**：同 shape 的第一个用例永远正确，后续同 shape 不同属性的用例整片错成第一个用例的模式——MERE/MARE 呈"整片错"量级（MARE=1.0 量级或畸高），而非小数值偏差；失败用例随执行顺序非确定转移（谁后跑谁错），单跑全部通过。kernel printf（plog `ParsePrintToLog`）与 host tiling `OP_LOGI` 探针：8 用例仅 3 次 tiling 调用、属性恒为首例值。L2 入口打印 N 次而函数体内部打印少于 N 次（函数被调用、返回 ACL_SUCCESS，但函数体未执行）是缓存短路的直接证据。

**2. 第三条规避路线：环境变量 ACLNN_CACHE_LIMIT=0（测试侧，不动 op_api）**：`strings libnnopbase.so` 可见 `MM_ENV_ACLNN_CACHE_LIMIT` 与告警文案（该变量必须为数字，非法值框架告警）。`export ACLNN_CACHE_LIMIT=0` 后每次调用都重新 tiling，kernel 逐用例收到正确属性（通过率 3/8→6/8，剩余失败为独立的精度问题）；终验 8/8、全量 40/40，对照组（缓存启用）3/8。适用于 op_api 无法变更时的 ST run.sh 规避（仅 Real 模式注入）。

**3. float 属性漏序列化（artifact_verified 的机制边界）**：op cache 以 `InitOpCacheKey()` 序列化的 key 命中缓存，该序列化未正确处理 float 类型的属性值——同 shape/dtype、不同 base/scale/shift（float 属性）的调用生成相同 key，tiling 只执行首次一次（plog：8 次调用仅 1 次 tiling），后续输出恒为首次属性的结果。规避：给每条 ST 用例分配互不相同的输入 shape（按维度 ±1、±2 生成"近似形状"，保持规模量级），使 key 因 shape 不同而 miss；用例集自检 `grep -oP 'shape=\[([0-9, ]+)\]' | sort | uniq -d` 为空。非 float 类型属性（int/bool/string）是否同样漏序列化未验证。

**4. 线程局部缓存与"重复签名新线程"隔离（闭合先前增量遗留的"多线程未验证"边界）**：CANN 运行时用**线程局部**缓存（`InitExecutorCacheThreadLocal`）存储 tiling 数据，键为输入/输出 shape+dtype、不含 strides/pads/dilations 等属性。测试编排层维护已执行用例的 shape 签名集合，当前签名重复时放入**新线程**执行（线程局部缓存天然隔离，tiling 被重新调用；线程内重新 `aclrtSetDevice`/`torch.npu.set_device` 并用独立 stream）。PyTorch 与 C++ 双工程同款实现：L0 60/60、L0+L1 全量 187/187。签名不需要包含属性参数：缓存键本身不含属性，只要 shape+dtype 重复就可能命中。若未来版本缓存改为进程级/设备级，新线程不再隔离（未验证）。

**5. 层次证据（这些手段都清不掉该缓存）**：`OP_COMPILER_CACHE_MODE=disable`、`aclSetCompileopt(ACL_OP_COMPILER_CACHE_MODE, "disable")`、`ACL_OP_JIT_COMPILE` 关编译缓存全部无效（op 编译缓存与 aclnn executor/OpExecCache 是两层）；`aclFinalize()` + `aclInit()` 全量重置也清不掉。`DFX_IN` 哈希缓冲有 8192 字节上限（op_cache.h `K_HASH_BUF_SIZE`，超限置 `K_INVALID_HASH_OFFSET` 禁用缓存）——属性极多时缓存自动退化为不命中，正确性不受影响。测试侧 FORK_MODE 每用例独立进程作 fallback 的代价：串行跑 92 用例 15 分钟超时。

**独立 trace 新增验证信号**（五个独立 trace，均为 tool_observed，另注明者除外）：

- resize 双线性：4 条用例仅 2 次 tiling 调用；交换用例顺序后失败集合跟随顺序移动（先跑的对、后跑的错）；L2 入口打印 4 次而函数体内部仅 2 次；换 `L2_DFX_PHASE_1_WITHOUT_CACHE` + alignCorners 编入 TilingKey 后 4/4、UT 25/25、全量 ST 23/23 两轮、PyTorch ST 39/39（含多条同 shape 不同属性用例）（来源平台：950；证据等级：tool_observed）。
- grid_sampler_3d：kernel printf 显示同 shape 全部用例 padMode=0 alignCorners=0 而 shape/stride 字段正确；host tiling 8 用例仅 3 次调用；单用例隔离 border 用例 PASS（MERE=1.825e-07）；`ACLNN_CACHE_LIMIT=0` 后 3/8→6/8，终验 8/8、全量 40/40，对照组 3/8、3/15（来源平台：950；证据等级：tool_observed）。
- AdamW 类：修复前全量 92 用例失败 5 个（MARE 最高 1.99e+00 超阈 5.00e-02），失败用例单跑全过；tiling 调用 38 次/19 唯一组合，日志属性恒为首个用例参数；`DFX_IN` 补 7 个标量属性后 92/92（FP32 65/65、FP16 14/14、BF16 13/13），tiling 调用升至 83 次，PyTorch 接入同包 92/92（来源平台：950；证据等级：tool_observed）。
- 指数族（float 属性）：issue 工件记载 8 次同 shape 不同属性调用 tiling 仅执行 1 次、后续复用首次 tiling data（来源平台：950；证据等级：artifact_verified——issue 文件为轨迹内已提交工件）；规避自检 29 条用例 shape 去重为空（来源平台：950；证据等级：tool_observed）；唯一 shape 用例集 NPU Real 全量通过 16/16（MERE=0）与 29/29 ALL PASS（FP32 MERE≤1.14e-08）（来源平台：950；证据等级：tool_observed）。
- conv2d 类 PyTorch 适配层：L0 批跑 60 条 2 条失败（MERE=5.94、MARE=1.55e+4，单跑全过）；成对复现失败随顺序转移（触发条件恒为同签名连续执行）；修复（seenShapes 签名集合 + 重复签名新线程执行）后 L0 60/60、L0+L1 187/187（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：`ACLNN_CACHE_LIMIT=0` 是测试侧规避、不修复生产侧（真实业务同进程同 shape 连续变 attr 仍会复用旧 tiling，框架侧待修）；`L2_DFX_PHASE_1_WITHOUT_CACHE`（另一独立 trace 的路线）、`DFX_IN` 补属性（主卡路线）、`ACLNN_CACHE_LIMIT=0`（本增量路线）、唯一 shape（float 属性形态）、新线程隔离（线程局部形态）五条路线按"能否改 op_api / 属性是否可入 DFX_IN / 宿主形态"选择。坏实践补充：把批跑偶发全错归因为"内存复用/未同步"，在适配层反复加 `synchronize()`/`empty_cache()`/加长冷却时间（实测全部无效，缓存与时间无关）；用加长用例间冷却时间规避；因单用例全过就放行批跑失败（失败用例随顺序漂移，重跑恰好通过不代表没有缓存互踩）；删缓存目录/重装算子包排查（进程内 executor cache 与磁盘产物无关）。
