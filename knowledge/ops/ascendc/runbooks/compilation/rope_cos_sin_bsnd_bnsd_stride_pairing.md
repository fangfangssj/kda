---
type: "Runbook"
title: RoPE cos/sin BSND 与 BNSD 反算配对
description: RoPE 类算子按行搬 cos/sin 时 layout 决定 row 序与坐标反算：BSND 行序 s=(row/N)%S 连续 N 行复用同一 cos[s]（N-broadcast，tiling cosSrcStrideDh=0）；BNSD 行序 s=row%S 连续行取不同 cos[s]（s-stride，tiling cosSrcStrideDh=Dh）。tiling 字段与 kernel 反算必须配对，否则数值错位。复现增量：cos.ndim=3 真实 B 轴维度（cosB=B、b=row//(S*N) 两布局同形、srcBase=(b*S+s)*Dh 由 kernel 反算不进 TilingData stride 字段），以及多分支 Tiling 特化架构——公共切分复用 ComputeTiling 单一真值源，分支只填 cos* 五元组差异字段与 tilingKey。
tags: [task.implementation, topic.tiling, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# RoPE cos/sin BSND 与 BNSD 反算配对

## 现象

在 BenchHarnessApplyRotaryPosEmb（RoPE 类算子，query/key 形状 (B,N,S,D)，cos/sin 为 (S,Dh) 二维表，Dh=D/2）里，kernel 需要逐行把该行对应的 cos[s]（Dh 个元素）经 NDDMA 搬进 UB 并 tile 成 D 维。同一份 kernel 计算链要同时服务 BSND（layout=0，行序 row=b*(S*N)+s*N+n）与 BNSD（layout=1，行序 row=b*(N*S)+n*S+s）两种布局：branch-1（interleaved-BSND）已经用 `s = ((rowStart + r) / N) % S` 反算；实现 branch-3（interleaved-BNSD）时行序改变，反算必须换成 `s = (rowStart + r) % S`，且 tiling 侧配对的 NDDMA 源 stride 字段从 `cosSrcStrideDh=0`（N-broadcast）变为 `cosSrcStrideDh=Dh`（s-stride）。两分支共存于一个 kernel 类，用 `if constexpr (kIsBnsd)` 编译期切换。落地后 aclnn ST（含 B 广播、最小 D、非对齐 D=10、大 shape 多核满切等 12 用例）与 TilingUT（含 stride 字段断言）全绿，验证配对正确。

## 适用场景

- 工程形态：aclnn 自定义算子的 AscendC kernel——cos/sin（或任意按位置索引的旁表）按行 NDDMA 搬运、同一计算链服务多种内存布局（BSND/BNSD 对偶）。
- API 路径：`AscendC::DataCopy<T, 1, cfg>`（NdDmaParams 的 loopSrcStride/loopSize）；tiling 的 cos/sin NDDMA stride 字段（cosB/cosS/cosLoopSize/cosSrcStrideDh/cosDstStrideD）；kernel 类模板参数 + `if constexpr` 布局切换。
- 验证过的平台与 CANN 版本：Ascend950PR（npu-smi 25.7.rc1），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何"行线性切分 + 按行取旁表"的布局敏感算子（RoPE、位置编码类）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：cos/sin 本身按 (B,S,Dh) 三维索引（cosB>1，每 b 取不同表）的场景——反算还需含 b，本卡公式不覆盖；非行线性的切分（如按 D 轴切块）；其他 NPU 架构（机制推断可迁移，未验证）。

## 根因/机制（布局↔反算↔stride 的配对推导）

行线性切分把 (B,N,S) 三个维展平成行号 row，两种布局只是展平顺序不同：

- **BSND（layout=0）**：`row = b*(S*N) + s*N + n`。行号连续递增时 n 先走——**连续 N 行的 s 不变**，即同一 `cos[s]`（Dh 个元素）被连续 N 行复用：N-broadcast。反算 `s = (row / N) % S`（整除吃掉 n 与 b 的贡献，模 S 吃掉 s）。tiling 配对：NDDMA 源在 N 行间不前进，`cosSrcStrideDh = 0`（重读同一 cos[s]）。
- **BNSD（layout=1）**：`row = b*(N*S) + n*S + s`。行号连续递增时 s 先走——**连续行的 s 递增**，每行取不同的 `cos[s]`：s-stride。反算 `s = row % S`（低位就是 s）。tiling 配对：NDDMA 外层 s 行源 stride = Dh（每行源前进 Dh 个元素），`cosSrcStrideDh = Dh`。

配对错误的后果是数值错位（有限值 mismatch 而非 NaN）：若 BNSD 用了 `(row/N)%S`，取到的 cos[s] 与该行真实 s 不符，旋转角错位——小 shape 下部分用例可能巧合通过（S=1 时两公式同值），大 shape/多核必错。

两布局下其余字段不变：`cosB=1`（cos.ndim=2，B 轴广播）、`cosS=S`、`cosLoopSize=Dh`（最内维一次搬 Dh 元素）、`cosDstStrideD=D`（Dh→D tile 因子 2，行 stride=D）。即 layout 只改"源怎么走"，不改"搬多少、写到哪"。

## 解决配方

### tiling 侧（分支特化字段，verbatim）

BSND 分支（branch-0/1 口径）与 BNSD 分支（branch-2/3 口径）的差异字段：

```cpp
    // ---- BNSD（layout=1）分支：s-stride ----
    td.cosB = 1u;             // cos.ndim=2 → B 轴广播
    td.cosS = in.S;           // cos 的 S 维 = query S
    td.cosLoopSize = in.Dh;   // NDDMA 最内维一次搬运 Dh 元素
    td.cosSrcStrideDh = in.Dh; // **s-stride**：外层 s 行源 stride=Dh，每行取不同 cos[s]
                               // （BNSD row 序连续 s 递增；BSND 分支此字段为 0 = N-broadcast）
    td.cosDstStrideD = in.D;  // Dh→D tile 因子 2，行 stride = D
    td.tilingKey = 3u;        // bit-encoding：layout 位 2 + rotaryMode 位 1
```

### kernel 侧（CopyInCosSin 坐标反算，verbatim）

同一 kernel 类用模板参数 + `if constexpr` 服务两布局（fp32 与非 fp32 两条搬运路径同形）：

```cpp
    __aicore__ inline void CopyInCosSin(uint32_t rowStart, uint32_t cnt)
    {
        const uint32_t H = H_;
        const uint32_t N = td_->N;
        const uint32_t S = td_->S;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        AscendC::NdDmaParams<T, 1> params;
        params.loopInfo.loopSrcStride[0] = 1;
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopLpSize[0] = 0;
        params.loopInfo.loopRpSize[0] = 0;
        params.constantValue = T(0);
        if constexpr (kIsFp32) {
            auto cosBuf = bufCos_.Get<T>();
            auto sinBuf = bufSin_.Get<T>();
            for (uint32_t r = 0; r < cnt; ++r) {
                // 坐标反算：
                //   BSND (kIsBnsd=false): row=b*(S*N)+s*N+n → s=(row/N)%S (N-broadcast: 连续 N 行同 cos[s])
                //   BNSD (kIsBnsd=true ): row=b*(N*S)+n*S+s → s=row%S        (s-stride: 连续行取不同 cos[s])
                uint32_t s;
                if constexpr (kIsBnsd) {
                    s = (rowStart + r) % S;
                } else {
                    s = ((rowStart + r) / N) % S;
                }
                AscendC::DataCopy<T, 1, cfg>(cosBuf[static_cast<int64_t>(r) * paddedH_],
                    gmCos_[static_cast<int64_t>(s) * H], params);
                AscendC::DataCopy<T, 1, cfg>(sinBuf[static_cast<int64_t>(r) * paddedH_],
                    gmSin_[static_cast<int64_t>(s) * H], params);
            }
        } else {
            auto cosBuf = bufCosIn_.Get<T>();
            auto sinBuf = bufSinIn_.Get<T>();
            for (uint32_t r = 0; r < cnt; ++r) {
                uint32_t s;
                if constexpr (kIsBnsd) {
                    s = (rowStart + r) % S;
                } else {
                    s = ((rowStart + r) / N) % S;
                }
                AscendC::DataCopy<T, 1, cfg>(cosBuf[static_cast<int64_t>(r) * paddedH_],
                    gmCos_[static_cast<int64_t>(s) * H], params);
                AscendC::DataCopy<T, 1, cfg>(sinBuf[static_cast<int64_t>(r) * paddedH_],
                    gmSin_[static_cast<int64_t>(s) * H], params);
            }
        }
    }
```

kernel 类声明（布局作为编译期参数，默认 BSND 保持既有实例兼容）：

```cpp
// kIsBnsd=false (默认) → BSND: s=(row/N)%S（N-broadcast：连续 N 行复用同 cos[s]）；
// kIsBnsd=true  → BNSD: s=row%S（s-stride：连续行取不同 cos[s]）。
// 其余计算链两分支完全一致，仅 layout 影响 cos/sin 对齐的坐标反算。
template <typename T, bool kIsBnsd = false>
class Branch1Kernel {
    using ComputeT = float;
```

## 设计决策

- 为什么坐标反算在 kernel 内推导、stride 在 tiling 下发：s 由 row 现场可算（一次模/除），下发是冗余；而 NDDMA 搬运参数（源 stride）是硬件配置，必须随 tiling 进 TilingData。两者描述同一事实（row 序），一个走代码、一个走数据，改动时必须**成对**核对。
- 为什么用 `if constexpr` 模板参数而不是运行时 if：布局在算子调用期固定（layout attr → tilingKey → 模板实例），编译期消除分支开销；且两布局的计算链（rotate_half、MulAdd、Cast、CopyOut）逐指令相同，模板复用避免复制漂移。
- 为什么 BSND 分支的 `cosSrcStrideDh` 是 0 而不是省略搬运：N-broadcast 语义 = 连续 N 行重读同一 cos[s]，源不前进即 stride 0；字段保留使多分支共用同一 TilingData 结构。
- 未采用的替代：在 tiling 里下发 `isBnsd` 布尔让 kernel 运行时分支——多一次每行判断且无法按布局特化搬运路径，被放弃。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 新增布局变体时先写 row 序展开式（`row = b*? + ?*? + ?`），再推导反算公式与源 stride，最后把三者写进同一处注释（如本例的 `// BSND: row=b*(S*N)+s*N+n → s=(row/N)%S` / `// BNSD: row=b*(N*S)+n*S+s → s=row%S`）——注释即配对合同。
- TilingUT 必须断言 `cosSrcStrideDh` 等配对字段（本算子族 UT 对 18 字段逐一 EXPECT_EQ），布局配错在 UT 层就能拦截。
- ST 用例至少覆盖一个"S 与 N 都 >1 且互不相同"的 shape（S=1 或 N=1 时两种反算公式同值，无法暴露配对错误）。

## 验证信号

以下信号来自该算子 branch-3（BNSD 布局）实现会话：布局配对落地后 ST/UT 全量验证。

- branch-3 ST 12/12（`numeric 11/11 PASSED, returns_empty 1/1 OK`），用例含 B 广播（B=2, cosB=1）、最小 D=2（Dh=1）、非对齐 D=10、大 shape fp16 多核满切+尾块+S 子块拆分（S 与 N 均大于 1，能区分两种反算）（来源平台：950；证据等级：tool_observed）。
- 全量回归：四分支（half/interleaved × BSND/BNSD）各 12/12、`run.sh` 退出码 0——四种布局-模式组合的 cos/sin 对齐互相独立验证（来源平台：950；证据等级：tool_observed）。
- TilingUT 85/85，其中 `cosSrcStrideDh=Dh`（BNSD 分支）与 `=0`（BSND 分支）的字段断言分别通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 公式只覆盖"行线性切分 + cos.ndim=2（B 轴广播）"的形态；cos.ndim=3（cosB>1，每 b 一张表）时反算需含 `b = row//(N*S)`（BNSD）并按 b 维索引源基址，本卡字段组合不适用（cos3d 场景由文末 复现增量覆盖）。
- `cosSrcStrideDh` 的单位是元素数（非字节），与 NDDMA params 的元素口径一致；若搬运层换字节口径接口须换算（未验证）。
- ST 判别依赖"S、N 均 >1"的用例存在；若用例集只有 S=1/N=1 的 shape，配对错误不可见（本算子族用例集已含大 shape 用例，无此问题）。

**未决项**：

- 无（row 序展开、反算、stride 配对三方均有代码与测试双向锚定）。
## 坏实践

- 照抄相邻分支的坐标反算公式只改 tilingKey——BSND 与 BNSD 的反算公式互斥，抄错一侧数值全错。
- 只改 kernel 反算不改 tiling 的 `cosSrcStrideDh`（或反之）——两处描述同一 row 序，单改一处即配对断裂。
- 在 S=1 或 N=1 的用例上"验证"布局正确性——两种公式在这些退化 shape 下同值，验证不了配对。

## 误诊

- 无（本会话 BNSD 反算一次配对正确；配错时的现象为有限值错位，可据 ST mismatch 的分布（按 s 分带错位）与 UT 的 stride 字段断言区分）。

## 增量（cos3d 真实 B 轴与多分支 Tiling 特化架构）

同一算子后续分支（branch-3/4/6，CANN 9.2.0）把配对推导扩展到第三维度并沉淀分支特化架构：

**1) cos.ndim=3（真实 B 轴）补齐本卡声明的缺口**：cos/sin 为 (B,S,Dh) 3D 张量、每 batch 独立位置编码时，与 cos2d 的差异**只影响 NDDMA 源基址**，loop stride 完全不变——

- cos2d（B 广播）：`srcBase = s*Dh`（同一 (S,Dh) 表对所有 b 复用）；
- cos3d（真实 B 轴）：`srcBase = (b*S+s)*Dh`（b 偏移 = 每 batch 独立表）；
- b 的坐标反算：BSND `row=b*(S*N)+s*N+n` → `b = row/(S*N)`；BNSD `row=b*(N*S)+n*S+s` → `b = row/(N*S)`——两布局 b 的系数都是 S*N（乘法交换律），**反算公式对两种 layout 通用**（layout 差异只体现在 s 的反算）。

tiling 侧只把 `cosB` 从 1 改成真实 B、`tilingKey` 置对应值（源基址由 kernel 坐标反算产出，**不进 TilingData stride 字段**——loop stride 与 srcBase 作用于 NDDMA 不同维度，正交无耦合）。kernel 侧差异点：

```cpp
    int64_t srcBase;
    if constexpr (kCos3d) {
        uint32_t b = absRow / (S * N);  // B 轴真实索引（BSND/BNSD 同形 S*N==N*S）
        srcBase = (static_cast<int64_t>(b) * S + s) * H;
    } else {
        srcBase = static_cast<int64_t>(s) * H;
    }
```

**2) 多分支 Tiling 特化架构（单一真值源 + 差异字段）**：8 个 tilingKey 分支共用一个 TilingData 结构（18 字段），公共切分（多核 `perCoreRows` 对齐 512 + UB `ubRows` 256B 对齐、`rowBytes=32*D`）全部复用公共 `ComputeTiling` 单一真值源；分支特化函数只填 cos/sin NDDMA 五元组与 tilingKey——layout（BSND/BNSD）只影响 row 序与坐标反算、不影响切分量，rotaryMode（half/interleaved）不影响 cos/sin 对齐（interleaved 的 stride=2 在 kernel 内推导，不进 TilingData）。正交组合分支的字段差异即"两个维度的叠加"：

| 字段 | half-BNSD-cos2d | half-BSND-cos3d | half-BNSD-cos3d（=前两者正交合并） |
|---|---|---|---|
| cosB | 1（B 广播） | B（真实 B 轴） | B |
| cosSrcStrideDh | Dh（s-stride） | 0（N 广播） | Dh |
| cosS / cosLoopSize / cosDstStrideD | S / Dh / D | 同 | 同 |
| tilingKey | 2 | 4 | 6 |

分支特化函数骨架（verbatim）：

```cpp
uint32_t ComputeBranch6Tiling(const Branch6Inputs& in, BenchHarnessApplyRotaryPosEmbTilingData& td)
{
    CommonTilingInputs tin{};
    tin.B = in.B; tin.S = in.S; tin.N = in.N; tin.D = in.D; tin.Dh = in.Dh;
    tin.coreNum = in.coreNum;
    tin.usableUB = in.usableUB;
    tin.dtypeBytes = in.dtypeBytes;
    uint32_t blockDim = ComputeTiling(tin, td);  // 公共 12 字段 + blockDim（单一真值源）

    td.cosB = in.B;             // cos.ndim=3 → B 轴真实数据不广播
    td.cosS = in.S;             // cos 的 S 维 = query S
    td.cosLoopSize = in.Dh;     // NDDMA 最内维一次搬运 Dh 元素
    td.cosSrcStrideDh = in.Dh;  // s-stride（取自 BNSD 维度）
    td.cosDstStrideD = in.D;    // Dh→D tile 因子 2
    td.tilingKey = 6u;          // half ∧ BNSD ∧ cos.ndim=3（bit-encoding: 4|2|0）
    return blockDim;
}
```

host glue 按 `ChooseTilingKey` 的 bit-encoding（`cosDim 位 | layout 位 | rotaryMode 位`）以 `else if (tilingKey == N)` 链派发到各特化函数。

验证信号（复现增量，同算子 branch-2/3/4/6 的 tiling TDD 会话，宿主 gtest / ST 上板 Ascend950PR）：

- branch-2：红基线 14 例全 FAILED（stub 不填 td）→ 实现 71/71；`./build.sh` 编译打包成功、无 tiling 未注册告警（来源平台：950；证据等级：tool_observed）。
- branch-3：红基线 14 例（`act.tilingKey 0 vs 3`）→ 85/85，构建退出码 0、无 stub 残留（来源平台：950；证据等级：tool_observed）。
- branch-4（cos3d）：TilingUT 红基线 0/14 → 实现后 99/99；ST 12/12（含 B=3 无广播与跨 b 边界大 shape 用例），前序四分支各 12/12 无回归（来源平台：950；证据等级：tool_observed）。
- branch-6（BNSD×cos3d 正交合并）：红基线 17 例（`act.cosB 0 vs B`、`act.cosSrcStrideDh 0 vs Dh`、`act.tilingKey 0 vs 6`）→ 131/131；kernel 分发接通后 ST 12/12、其余分支回归各 12/12（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- cos3d 的 cos 形状契约 (B,S,Dh) 由 host 侧校验保证；上游给 (1,S,Dh) 广播 cos 会走 cos2d 分支（tilingKey 不同）。
- b 反算 `row/(S*N)` 用 uint32 乘法，S*N 需不溢出 uint32（实测最大 B*S*N≈52 万安全；超大规模须换 int64 或分步）。
- 特化复用的前提是"公共切分同形"（layout/mode/cos.ndim 不改变多核与 UB 切分量）；分支引入不同 UB 峰值或切分策略时 `ComputeTiling` 复用失效，须单独推导并在 oracle 独立重算。
- interleaved∧BNSD∧cos3d 组合（tilingKey=7）字段预期同形，实现时按本表叠加验证。

坏实践（复现增量）：

- 每个分支整段复制公共切分公式再改几个字段——切分参数改动时各分支必然漂移；公共字段一律复用公共函数。
- 分支特化函数照抄相邻分支时忘了改 `tilingKey` 值——host 会按错误 key 派发到别的 kernel 分支，产生难定位的精度错误；UT 的 tilingKey 字段断言正是为此。
- 把 `srcBase=(b*S+s)*Dh` 编码进 TilingData stride 字段——源基址随行变化，由 kernel 坐标反算产出，混入 stride 字段会把两个正交差异耦合。