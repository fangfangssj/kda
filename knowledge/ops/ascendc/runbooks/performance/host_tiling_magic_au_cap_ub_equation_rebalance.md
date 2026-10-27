---
type: "Runbook"
title: "tiling魔数au上限与UB方程欠计：均衡au求解"
description: "一般路径 tiling 的 au 上限来自 tileLimit 魔数（16384/sizeof(float)）且 UB 合法性方程按 3*pre+4*post+16384 欠计 buf3 实占（目标 case 欠 16256B）；改为真实字节清单方程 + coreNum 均衡 au 扫描，仅 host 改动即使目标 case 24→48 核，同算子同协议相对原始基线 1.79x——host-only 手段的天花板对照（safety_fallback）。"
tags: ["task.optimization", "topic.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# tiling魔数au上限与UB方程欠计：均衡au求解

## 现象

在 Ascend950PR（coreNum=56、UB=253952B，CANN 9.2.0）上优化融合算子 FusedRmsNormSigmoidMul：目标 case 为 bf16 `(128,6,128)` 类形状（A=768、R=128、ra=128）。原始融合基线相对非融合五算子链约 2.43x 慢（同协议 10 组中位，判别见 [mte2_issue_overhead_vs_bandwidth_bound](mte2_issue_overhead_vs_bandwidth_bound.md)），且 tiling 探针显示目标 case 只用 blockDim=24（au=32、ao=24）——56 个 AIV 只喂了 24 个核。检查 host tiling（`ComputeTilingImpl`，约 141 行起）发现两处问题：

1. au 上限是魔数推导：`const int64_t tileLimit = 16384 / sizeof(float);`（=4096）→ `au = max(1, tileLimit / ra)`。目标 case ra=128 → au 上限 32，但实际因 `3*pre+4*post+16384 > ubSize` 方程约束只取到 32；而 A=768 用 au=32 时 ao=24（768=32×24，无尾 tile 但只用 24 核），用 au=16 则 48 核整除同样无尾 tile——魔数上限既不反映真实 UB 容量也不反映核数均衡。
2. UB 合法性方程口径错误：`if (3 * pre + 4 * post + 16384 > int64_t>(ubSize)) return false;` 其中 pre/post 按 4B/元素预留。设备侧 InitBuffer 实际是 `buf0~buf3 各实拿 preBufSize`（buf3 并非 4*post 字节）+ `cache_ 拿 cacheBufUbSize`：原方程把 buf3 按 `4*post` 计，目标 case 欠计 16256B——方程既可能误判可行（欠计）也可能过度保守。

该算子是 host tiling 与 device kernel 分离的直调工程，device 侧（kernel.h 等）此前已冻结不允许改动，本卡是"只改 host tiling"的低风险杠杆对照方案。

## 适用场景

- 工程形态：AscendC 算子 host tiling（直调 `ComputeTilingImpl` 或注册工程 `TilingFuncPublic`）调优；device kernel 冻结或希望零设备侧改动的场合。
- API 路径：host tiling 的 au/pre/post 求解与 UB 合法性方程；`PubCeil/PubUp` 对齐宏；coreNum 平衡扫描。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——按行 tile 切分、行主序连续、base+group 双路径的行归约类算子。
- 路线优先级：safety_fallback——本卡是无设备侧改动权限或需要快速收益时的保底手段；其相对收益天花板明确（见"性能影响"的成对对比），追求更高加速比必须叠加搬运/计算通路手段。fallback 声明：本方案单独使用不改变逐行搬运与 VF 调用形态，只改善核数利用与 UB 判定正确性。
- 明确不适用 / 未验证场景：group 路径触发域（超长 R）本卡刻意保持原始形态不动；设备 kernel 存在逐行搬运 bound 时本卡收益有限（对照实测）；非 bf16 输入的 2B tile 字节口径未验证。

## 根因

au 的可行域被两处非物理约束压制：上界来自 `tileLimit=16384/sizeof(float)` 魔数（与该 kernel 实际 buffer 布局无关），UB 合法性判据 `3*pre+4*post+16384<=ubSize` 与设备真实分配清单（buf0-3 各 preBufSize + cache cacheBufUbSize）不一致（buf3 被按 4*post 计，欠计）。两者叠加使目标 case 落在 au=32/24 核，既没用满 56 核也没让 au 在真实 UB 约束下取均衡解。

## 解决配方

修复前（魔数 au + 欠计 UB 方程）：

```cpp
    int64_t blockElems = blockSize / 2, cacheElems = PubDown(cacheLineSize / 2, blockElems);
    if (blockElems <= 0 || cacheElems <= 0 || vectorSize <= 0) return false;
    const int64_t tileLimit = 16384 / sizeof(float);
    int64_t ru = std::min(R, tileLimit), ra = PubUp(ru, blockElems);
    int64_t au = std::max<int64_t>(1, tileLimit / ra);
    int64_t post = PubUp(au * 4, blockSize), pre = PubUp(au * ra * 4, blockSize);
    if (3 * pre + 4 * post + 16384 > int64_t(ubSize)) return false;
```

修复后（死常量清除 + 均衡 au 求解 + 真实清单 UB 方程）：

```cpp
    // 清除 cacheElems/vectorSize 死常量（原实现计算后从未参与求解）；
    // blockElems（=blockSize/2=16，bf16 的 32B 对齐元素数）仍用于 ra 对齐
    int64_t blockElems = blockSize / 2;
    if (blockElems <= 0) return false;
    // ru/ra 维持原公式：group 触发域（ro=PubCeil(R,ru) 输入 ChooseGroupTiling）
    // 与设备归约段 R chunk（rUbFactor）语义保持原始形态——超长 R case 仍走
    // group 路径且参数不变
    const int64_t tileLimit = 16384 / sizeof(float);
    int64_t ru = std::min(R, tileLimit), ra = PubUp(ru, blockElems);
    // au 由 coreNum 均衡扫描求解（取代魔数 au=tileLimit/ra）
    const int64_t cacheBufUbSize = 16384;
    int64_t au = SolveBalancedAu(A, ra, static_cast<int64_t>(coreNum),
                                 static_cast<int64_t>(ubSize), cacheBufUbSize);
    if (au < 1) return false;
    int64_t post = PubUp(au * 4, blockSize), pre = PubUp(au * ra * 4, blockSize);
    // UB 合法性方程修正为设备真实 buffer 清单：buf0-3 各实拿 preBufSize
    // （原方程把 buf3 按 4*post 计，目标 case 欠计 16256B），cache 按
    // cacheBufUbSize。pre/post 公式形态不变（仍按 4B/元素保守上界建模并喂
    // ChooseGroupTiling/P2CapacityLegal：base 路径 buf0-3 为 bf16 实占 2B/元素，
    // group 路径 buf0 作 FP32 归并缓冲按 4B/元素精确；保守口径保持 kernel 侧
    // capacity=preBufSize/sizeof(bfloat16_t) 的多 chunk 切分语义兼容）。
    if (4 * pre + cacheBufUbSize > int64_t(ubSize)) return false;
```

新增 `SolveBalancedAu`（放在 `ChooseGroupTiling` 之后；注释与实现完整）：

```cpp
// ===== 一般路径 au 均衡求解（coreNum 纳入） =====
// 在修正后的 UB 合法性方程（4*pre + cacheBufUbSize <= ubSize，pre=PubUp(au*ra*
// blockSize) 按 4B/元素保守上界建模：base 路径 buf0-3 实为 bf16（2B/元素），
// group 路径 buf0 作 FP32 归并缓冲按 4B/元素精确，保守口径同时覆盖两者，且保持
// kernel 侧 capacity=preBufSize/sizeof(bfloat16_t) 与 P2CapacityLegal 的
// pre/post 语义兼容）约束下扫描 au 候选（2 的幂 ∪ 16 的倍数 ∪ UB 上界本身，
// 均不超过 aTotal），按 (关键路径行数上界 crit=Ceil(Ceil(A/au)/usedC)*au asc,
// A%au==0 整除 desc, au desc（更大 tile/更少 RstdVF 段数）, usedCoreNum desc)
// 择优；tile 单 tensor 字节（au*ra*2B，bf16 真实口径）不低于约 2KiB 为软过滤
// （存在满足者时强制）。消除魔数 tileLimit 对 au 的上限；au 上界
// 改由真实 UB 方程给出。目标 case A=768/R=128/ra=128：au=16
// （768=16×48，48 核整除无尾 tile；crit=16 与 au=8 持平后由整除+au desc 胜出）。
// 返回 0 表示连 au=1 都不满足 UB 方程（调用方 return false，与原方程失败一致）。
int64_t SolveBalancedAu(int64_t aTotal, int64_t raAlign, int64_t coreNum,
                        int64_t ubSize, int64_t cacheBufUbSize)
{
    constexpr int64_t elemBytes = sizeof(float);  // pre 的保守建模口径（4B/元素）
    constexpr int64_t minTileBytes = 2048;  // tile 单 tensor 字节软下限（约 2KiB）
    if (aTotal < 1 || raAlign < 1 || coreNum < 1 || ubSize < 1 || cacheBufUbSize < 1) return 0;
    auto preBytes = [raAlign](int64_t au) { return PubUp(au * raAlign * elemBytes, 32); };
    auto tileBytes = [raAlign](int64_t au) { return PubUp(au * raAlign * 2, 32); };
    // UB 上界：4*preBytes(au)+cache <= ubSize 关于 au 单调；公式求解后回退校验
    // PubUp 到 32B 的舍入（至多回退 1~2 步）
    int64_t upper = std::min(aTotal, (ubSize - cacheBufUbSize) / (4 * raAlign * elemBytes));
    while (upper >= 1 && 4 * preBytes(upper) + cacheBufUbSize > ubSize) --upper;
    if (upper < 1) return 0;
    // 候选集：2 的幂 ∪ 16 的倍数 ∪ UB 上界本身（去重，截断到 upper）
    int64_t cands[300];
    int32_t candNum = 0;
    for (int64_t p = 1; p <= upper && candNum < 300; p <<= 1) cands[candNum++] = p;
    for (int64_t m = 16; m <= upper && candNum < 300; m += 16) {
        bool dup = false;
        for (int32_t i = 0; i < candNum; ++i) {
            if (cands[i] == m) { dup = true; break; }
        }
        if (!dup) cands[candNum++] = m;
    }
    {
        bool dup = false;
        for (int32_t i = 0; i < candNum; ++i) {
            if (cands[i] == upper) { dup = true; break; }
        }
        if (!dup && candNum < 300) cands[candNum++] = upper;
    }
    bool hasMinTile = false;
    for (int32_t i = 0; i < candNum; ++i) {
        if (cands[i] <= aTotal && tileBytes(cands[i]) >= minTileBytes) hasMinTile = true;
    }
    int64_t bestAu = 0, bestCrit = 0, bestExact = 0, bestUsed = 0;
    for (int32_t i = 0; i < candNum; ++i) {
        const int64_t au = cands[i];
        if (au > aTotal) continue;  // 超过总行数的候选无意义（候选恒含 1）
        if (hasMinTile && tileBytes(au) < minTileBytes) continue;
        const int64_t tiles = PubCeil(aTotal, au);
        const int64_t used = std::min(tiles, coreNum);
        const int64_t bigLoop = PubCeil(tiles, used);
        const int64_t crit = bigLoop * au;  // 最慢核行数上界
        const int64_t exact = (aTotal % au == 0) ? 1 : 0;
        const bool better = bestAu == 0 || crit < bestCrit ||
            (crit == bestCrit && (exact > bestExact ||
                (exact == bestExact && (au > bestAu ||
                    (au == bestAu && used > bestUsed)))));
        if (better) {
            bestAu = au;
            bestCrit = crit;
            bestExact = exact;
            bestUsed = used;
        }
    }
    return bestAu;
}
```

尾部 TilingData 填充同步把 `t->cacheBufUbSize = 16384;` 改为 `t->cacheBufUbSize = cacheBufUbSize;`（单一事实源）。

## 设计决策

- 为什么 UB 方程改保守 4B/元素口径而不是 bf16 精确 2B：`pre` 同时喂 `ChooseGroupTiling`/`P2CapacityLegal`（group 路径 buf0 作 FP32 归并缓冲按 4B 精确）与 kernel 侧 `capacity=preBufSize/sizeof(bfloat16_t)` 的多 chunk 切分语义；保守口径同时覆盖 base/group 两路径且不破坏既有语义，仅快路（另有字节精确方程）才需要精确口径。
- 为什么 ru/ra 与 group 触发域保持原公式：超长 R case（如 R≈9.37M）必须仍走 group 路径且参数不变，否则 18 条 L0 中的 group case 语义漂移；本卡只动 base 路径内的 au。
- 为什么候选集取"2 的幂∪16 的倍数∪上界"而不是全扫描：A 可能极大（百万级），全扫描 O(A) 不可接受；对齐友好的稀疏候选集已覆盖整除解与上界解，目标 case 的 48 核整除解在集合内。
- 为什么 crit（最慢核行数上界）作为第一排序键：blockDim 提升不等于加速——尾 tile 或不均匀切分会让最慢核拖尾；`crit=Ceil(Ceil(A/au)/usedC)*au` 直接刻画最慢核负载。
- 当前方案路线档位：safety_fallback（host-only 杠杆对照；终选方案在其上叠加搬运合并与 VF 收拢）。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0（coreNum=56、UB=253952B），CANN 9.2.0。
- 相关文件（候选隔离副本内，仅 1 个文件与基线不同）：`csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp`（SolveBalancedAu + ComputeTilingImpl 修改）；device 侧全部文件（kernel.h/tiling_struct.h/entry/launch）与基线逐字节一致（SHA 复核）。
- 验收入口：`python3 run_acceptance.py --work-dir round1/optimized_s3-host-tiling-rebalance --cases functional-acceptance-cases.csv --golden .../golden.py --out-manifest .../run-manifest.json --report .../report.md`（strict 门禁 18/18）。

## 性能影响

- 相对原始融合基线（同算子、同协议、同设备，参照路线=优化前 tiling）：有效 case 几何加速比 1.79x（来源平台：950）。
- 与同会话其他候选成对对比（参照路线=本 host-only 方案）：tile 级合并 DMA 快路约 3.45x 于本卡、叠加 VF 收拢的终选约 5.12x 于本卡（成对，来源平台：950）——成对量化说明：不解除逐行搬运 bound 时，仅靠 tiling 均衡（24→48 核）的天花板约 1.8x。
- 本卡单独未达成本会话的链内目标线（链参考/1.2），作为杠杆隔离对照保留。

## 预防

- 写 tiling 时把"魔数推导的尺寸上限"换成"由真实 buffer 字节清单导出的方程"，并在注释中列全清单（哪个 buf 实拿多少字节、按什么口径）。
- UB 方程的每个项必须能对应到设备侧 `InitBuffer` 的实际调用；buf 分配改动时方程同步改。
- au 求解把 coreNum 纳入均衡目标（整除解优先、无尾 tile、crit 最小），不要只看"UB 装得下"。

## 验证信号

以下信号来自该融合算子仅 host tiling 重平衡候选的实施会话（Ascend950PR / CANN 9.2.0）：改码前先做 Python 仿真对账，构建后真库 tiling 身份检查，再走唯一验收入口与同协议性能采集。

- 仿真对账：18 条 L0 仿真 blockDim/tilingKey 与基线 manifest 一致性核对——组用例（超长 R）参数一致，目标 case au=16→bd=48，仅 1 条广播边界 case 填充提升（bd 1→3）（来源平台：950；证据等级：tool_observed）。
- 构建：5/5 步骤成功，AIVEC patch 6 sites、CUBE magic remaining 0，VF 信号与基线完全一致（asc_vf_call=6、cast_trait=2）（来源平台：950；证据等级：tool_observed）。
- 真库身份检查：目标 case blockDim=48/tilingKey=0，17/18 case 与基线逐字节一致（来源平台：950；证据等级：tool_observed）。
- 功能验收（精度不回归）：18 条 L0 strict 18/18 PASS，golden/cases/runner SHA 冻结一致（来源平台：950；证据等级：tool_observed）。
- 修改面：设备文件 SHA 与基线一致（仅 kernel.cpp 修改），死标识符清除后无残留引用（来源平台：950；证据等级：tool_observed）。
- 性能：目标 case 10 组中位采集 validation 30/30；有效 case 几何加速比相对原始融合基线 1.79x（同算子同协议成对，差异远超抖动阈值）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡不触碰逐行搬运与 VF 调用形态：device 侧存在搬运/发射 bound 时收益封顶约 1.8x（成对对照实测），追求更高必须叠加设备侧手段（见 [tile_merged_dma_eliminate_mte2_issue_bound](tile_merged_dma_eliminate_mte2_issue_bound.md)）。
- UB 方程的保守 4B 口径在 base 路径（buf 实占 2B/元素）下偏保守——需要更大 au 时应像快路那样按 bf16 字节精确建模，但精确口径必须重新核算 group 路径兼容性。
- `SolveBalancedAu` 的候选集为稀疏集合：极端 A 值下可能错过非 2 幂/非 16 倍数的最优整除解（如 A=质数×大素数场景退化为 crit 次优）；上界本身在候选集内兜底。
- 超长 R 的 group 路径触发域刻意未动：其参数优化需另行分析（未验证）。
- 同主题的 UB 预算反解设计与 aUbFactor 钳制路由翻转反面案例见 [ub_budget_backsolve_dual_axis_split_clamp_ring](../runtime/ub_budget_backsolve_dual_axis_split_clamp_ring.md) 与 [aub_factor_cachebuf_clamp_flips_base_group_route](../runtime/aub_factor_cachebuf_clamp_flips_base_group_route.md)；按可用核数切分恢复并行度的同族结论见 [row_reduction_min_rows_floor_underutilizes_aiv](row_reduction_min_rows_floor_underutilizes_aiv.md)。

**未决项**：

- 广播边界 case 的填充提升（bd 1→3）是 SolveBalancedAu 的副作用收益，其对广播路径性能的影响未单独测量。
- 保守 4B 口径与 bf16 精确口径的通用切换条件（何算子形态可安全收紧）未系统化。
## 坏实践

- 用 `tileLimit/ra` 这类与 buffer 布局无关的魔数限制 tile 尺寸——它既可能虚假放行（越界）也可能压制核数利用（本例 24/56）。
- UB 方程里混用"元素数×错误字节"的项（如 buf3 按 4*post 计而实拿 preBufSize）——欠计 16KB 量级足以让越界写入逃过判定。
- 只验证目标 case 的 blockDim 提升就收工——必须逐 case 对账冻结验收集的 bd/tilingKey，确认 group/广播/tiny case 零漂移（本会话 17/18 逐字节一致 + 1 条预期内填充提升）。

## 误诊

- 曾把"24/56 欠利用"归因于核数硬限制：实测同 shape 下快路候选可稳定用 48 核，证明欠利用来自 au 求解而非调度限制。
