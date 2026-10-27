---
type: "Runbook"
title: BiSheng 不重编 TPL_UI_LIST 第二实例化
description: ASCENDC_TPL_UI_LIST 定义多值模板参数（如 HAS_SCALES 0/1）时，源码变更后 BiSheng 只重编第一个实例化（_1），第二实例化（_257）沿用旧代码（--make_clean 干净重建仍复现）；叠加 AscendCL OP CACHE 跨进程持久化 TilingKey 选择，修复无法应用到对应执行路径。判别靠 +1000 行为标记探针，二进制文件名 hash 双向不可作重编判据。
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# BiSheng 不重编 TPL_UI_LIST 第二实例化

## 现象

BenchNoHarnessMoeReRouting 用 `ASCENDC_TPL_UINT_DECL(HAS_SCALES, 8, ASCENDC_TPL_UI_LIST, 0, 1)` 定义两值模板参数（有/无 per_token_scales），BiSheng 生成的 `.o` 里含两个 kernel 函数（`_1`=HAS_SCALES=true、`_257`=HAS_SCALES=false）。迭代一联调中，无scales用例（L0_002/L0_007）的 `permute_token_idx` 输出为 float 位模式垃圾值：

```
不匹配 [0]: 期望=0, 实际=1058048516      （0x3F13F324，float 位模式）
不匹配 [1]: 期望=1, 实际=1059669090
不匹配 [2]: 期望=2, 实际=1007391513
[FAIL] int32 mismatch count=64 / 64
```

对 kernel 做多轮修复（移除 `if constexpr(HAS_SCALES)` 守卫、始终分配 scalesBuf 等）并重编译安装——**有scales路径逐轮改善直至全对，无scales路径的垃圾值跨重编逐位不变**（含 `bash build.sh --make_clean` 干净重建、kernel 二进制文件名 hash 从 ea226e 变为 bc44b2 之后仍不变）。配套现象：tiling 函数里修改 `ASCENDC_TPL_SEL_PARAM` 的选择参数也不生效——运行期仍按旧 TilingKey 选 kernel。

## 适用场景

- 工程形态：算子仓开发，`ASCENDC_TPL_UI_LIST` 多值模板参数（同一 kernel 多实例化）+ tiling 侧动态选择。
- API 路径：`ASCENDC_TPL_UINT_DECL(..., ASCENDC_TPL_UI_LIST, ...)`、`ASCENDC_TPL_SEL_PARAM`、`bash build.sh --soc=ascend950 [--make_clean]`、AscendCL 运行时 OP CACHE（op_cache.cpp）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（BiSheng 编译器 + AscendCL 运行时行为，与算子无关）。
- 路线优先级：safety_fallback——以下判别/规避手段是工具链修复前的兜底；确认命中后正确的路线是换单实例化或运行时参数，见遗留问题。fallback 声明：本卡不修复编译器缓存本身，只提供判别与绕行；结构性规避（单实例化/运行时参数）未实施验证。
- 明确不适用 / 未验证场景：单值模板参数（`ASCENDC_TPL_UI_SINGLE`）模式（issue 后续计划验证，本轨迹未执行）；其他 CANN 版本未验证。

## 根因

**编译器层未完全定位到缓存 key 的实现**，已证实的行为链（见判别方法）：

1. BiSheng 对 `ASCENDC_TPL_UI_LIST` 的模板实例化使用增量编译缓存：源码变更后 `_1`（第一实例化）被正确重编，`_257`（第二实例化）沿用旧代码——`rm -rf build` 后从零构建仍复现，说明缓存在编译器内部而非 cmake 层。
2. AscendCL 运行时 OP CACHE 跨进程持久化 operator executor 的 TilingKey 选择结果，tiling 函数修改 `ASCENDC_TPL_SEL_PARAM` 参数不会使缓存失效。
3. 两者叠加：kernel 代码修复 + tiling 选择修正都无法通过常规「编译→安装→运行」流程应用到第二实例化的执行路径。

## 判别方法

**触发**：某条 TilingKey 路径的行为在多轮源码修复+重编译安装后逐位不变（垃圾值指纹恒定），而其他路径正常响应修复。

**探针（+1000 标记法）**：

1. 在 kernel 代码里给 idx 输出值加 +1000 偏移标记；
2. 重新编译、安装、运行 ST；
3. 对比两类用例的输出变化——`_1` 路径（L0_001，HAS_SCALES=true）idx 输出变化 +1000（已更新）；`_257` 路径（L0_002，HAS_SCALES=false）idx 输出完全不变（未更新）。

证据级别：该实验结果为 issue 文件记录（agent_reported 级，非本会话复现）——由修复子任务执行并记录于 issue 文件，本卡以 issue 文件读取与本会话实测信号（垃圾值跨重编不变、hash 双向误导）交叉锚定，不计入「验证信号」。

**辅助观测**：

```bash
# 源码确实包含修改（排除「忘保存」）
grep -n "tokenBuf\|idxBuf\|PipeBarrier\|GetValue" op_kernel/arch35/*.h | head
# 干净重建后对比 kernel 二进制文件名 hash
bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8 2>&1 | grep "Kernel binary:"
```

**解释**：第二实例化的二进制未随源码更新（编译器缓存），且运行时 TilingKey 选择被 OP CACHE 固化——修复代码从未在该路径执行。

**重要误诊警示**：kernel 二进制文件名 hash **不能**作为重编判据——修改 `.h` 后 hash 可保持 ea226e 不变但行为已变（第一实例化已更新）；给入口 `.cpp` 加一行注释即可让 hash 变为 bc44b2（hash 明显由入口文件派生），但第二实例化仍旧是旧代码。判断「某路径是否吃到修复」唯一可靠的是行为标记探针，不是文件名。

## 解决配方

工具链问题本身未在本轨迹内修复（已登记 issue 向编译器团队报告）。可复用的规避手段（按代价排序）：

1. **确认命中后，不要继续在第二实例化路径上调试**——先跑 +1000 标记探针区分「代码错」与「二进制旧」，避免对着旧二进制改代码（本会话在无scales路径上消耗了多轮无效修复）。
2. **ST 层规避（本会话实际采用，性质是测试侧绕过而非修复）**：无scales用例提供 dummy scales tensor，使运行时选择已正确重编的 `_1` 路径——该做法掩盖了 `_257` 路径未验证的事实，不得当作正修复引用。
3. **结构性规避（issue 后续计划，未执行）**：改 `ASCENDC_TPL_UI_SINGLE` 单实例化；或将 HAS_SCALES 从编译期模板参数改为运行时参数（`if` 替代 `if constexpr`）。

```cpp
// 触发形态（多值模板参数 + 条件化 buffer 布局，本会话的翻车组合）
// tiling_key.h
ASCENDC_TPL_UINT_DECL(HAS_SCALES, 8, ASCENDC_TPL_UI_LIST, 0, 1)
// kernel .h —— 两个实例化的 UB 布局随 if constexpr 分叉
if constexpr (HAS_SCALES) {
    uint32_t scalesSize = tileTokens_ * sizeof(float);
    scalesSize = (scalesSize + BENCH_MOE_32B_ALIGN - 1) / BENCH_MOE_32B_ALIGN * BENCH_MOE_32B_ALIGN;
    pipe.InitBuffer(scalesBuf, scalesSize);      // _257（无scales）实例不分配 → 布局与 tiling 预算不一致
}
```

## 设计决策

- 为什么把「hash 不变」降级为辅助观测：实测 hash 与「内容是否更新」双向解耦（不变≠没更新，变了≠全更新），行为标记才是金标准。
- 结构性规避优先级的理由：单实例化消除缓存分叉面；运行时参数消除的是编译期分叉本身；两者都比「每次 touch 入口文件」可靠（touch 只影响文件名 hash，不保证第二实例化重编）。

## 预防

- 多值 TilingKey 模板参数的算子，每次「修复后验证」前先跑行为标记探针确认目标路径二进制已更新。
- 条件化 `pipe.InitBuffer`（`if constexpr` 分叉 UB 布局）时，确保 tiling 的 UB 预算与**所有**实例化的实际布局一致（本会话的布局不一致是无scales路径输出垃圾的近因，缓存是它无法被修复的原因）。

## 验证信号

来源说明：BenchNoHarnessMoeReRouting 迭代一联调修复与复查会话。以下仅列本会话内实测的信号；+1000 标记实验为跨会话 issue 文件记录（agent_reported 级，非本会话复现），已归入「判别方法」。

- 触发信号：L0_002 idx 垃圾值 1058048516/1059669090/1007391513 跨多轮重编（含 --make_clean、hash ea226e→bc44b2）逐位不变（来源平台：950；证据等级：tool_observed）
- hash 误导性实测：`.h` 修改后 hash 保持 ea226e 但第一路径行为已更新；入口 `.cpp` 加注释后 hash 变 bc44b2（来源平台：950；证据等级：tool_observed）
- ST 层规避后 12/12 通过（性质为绕过，不构成 `_257` 路径修复的验证）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 限定 CANN 9.1.0 随附 BiSheng + AscendCL OP CACHE 组合；其他版本未验证（未验证，推断同类机制可能延续）。
- OP CACHE 的清除/禁用方法未找到（issue 后续计划项，未执行）。
- 本卡不宣称缓存 key 的实现细节（编译器内部机制未定位）。

**未决项**：

- **优先项：复现并追根因到机制层**——定位 BiSheng 增量缓存的 key 构成与 OP CACHE 持久化位置/失效条件；向编译器团队报告（issue 已登记，未闭环）。
- `ASCENDC_TPL_UI_SINGLE` 单实例化与运行时参数两条结构性规避路线均未实施验证。
## 坏实践

- 对着不变的行为反复改 kernel 代码——若二进制未更新，代码修改零效果（本会话多轮无效修复的真实损失）。
- 用 kernel 二进制文件名 hash 判断重编状态（双向误导，见判别方法）。

## 误诊

- 「hash 不变 = 没重编」：被 +1000 实验推翻（第一实例化在 hash 不变时已更新）；真正粒度是**按实例化**缓存，不是按文件。
- 「--make_clean 就能解决」：实测干净重建后第二实例化仍旧旧代码。
