---
type: "Runbook"
title: tiling UB 预算不留 headroom 输出全 0
description: tiling 侧按 ubAvailable-fixedCost 全额反解 rUbFactor、不留栈/开销余量，ARA 的 padded 2D 块布局使实际分配贴满 UB 而越界，kernel 不报错但输出全 0（softmax sum_to_one=0.000000）；avail 再减 32KB headroom 修复，配三步单变量对照（删 PipeBarrier / 强制 AR 路径 / headroom）定位。
tags: [task.debugging, topic.tiling, topic.memory, topic.compute.normalization]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# tiling UB 预算不留 headroom 输出全 0

## 现象

在为一个 softmax 基准算子（opbuild 工程）的分块分支（R 轴超 UB、按 chunk 在线处理）补 ARA（归约轴非末轴，`[A0, R, A1]` 合轴布局）路径时：kernel 编译通过、无任何运行时报错，但 C++ ST 里三条 ARA 复现用例（`((50000, 4),)` dim=0、`((4, 50000, 8),)` dim=1/-2，fp16）全部输出全 0：

```text
[WB_125] FAIL (invariant sum_to_one: sum=0.000000)
[WB_126] FAIL (invariant sum_to_one: sum=0.000000)
[WB_129] FAIL (invariant sum_to_one: sum=0.000000)
```

（softmax 每个归约切片输出和应为 1，`sum_to_one` 不变量探针直接给出 0，说明输出根本没写出，而不是算错。）

当时代码状态：ARA 路径刚实现完（`CopyInChunkAra`/`CopyOutChunkAra`/`ProcessOneBlockAra` 在线两趟 + kernel 内按 `axisNum==2/3` 分发 AR/ARA），tiling host 的 `ComputeChunkedRTiling` 新加了 ARA 分支按 UB 预算反解每 chunk 行数 `rUbFactor`。AR（末轴）路径的既有用例全部正常。

## 适用场景

- 工程形态：opbuild/算子仓工程的 tiling host（`op_host/*_tiling_host.cpp`）UB 容量预算计算 + kernel `InitBuffer` 分配；kernel 直调复现同样适用。
- API 路径：`TBuf/TPipe InitBuffer` 尺寸来自 tiling 下发的 `cacheBufUbSize`/`rUbFactor` 等 tiling 字段；ARA 的 `DataCopyPad` 逐行 32B 对齐 padded 2D 块布局。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR）+ CANN 9.1.0（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`platform_wide`（UB 容量预算原则与布局无关，对任何留满预算的 tiling 计算都适用）。
- 路线优先级：`recommended_fastpath`（headroom 是零成本硬保护，无替代方案）。
- 明确不适用/未验证场景：本卡只覆盖"输出全 0 + 无报错"这一形态；UB 越界若表现为 `memory_oob` 报错或随机数值错，需另行定位（本会话未遇到）。

## 根因

tiling host 的 ARA 预算分支把 UB 可用量全额反解成 chunk 尺寸，没有为设备侧栈与隐式开销留余量：

```cpp
int64_t avail = in.ubAvailable - fixedCost;   // fixedCost 只含 max/sum buffer + 16KB carryBuf
```

`avail` 直接除以每元素成本得到 `rUbFactor`，kernel `Init` 再按该尺寸分配 xQue(双缓冲)/yQue/xF32Buf/eBuf/maxBuf/sumBuf/carryBuf——各 buffer 之和恰好贴满 `ubAvailable`。ARA 布局的 per-row 32B 对齐 padding 让实际字节数进一步放大，叠加设备侧栈/隐式开销后真实分配超出可用 UB，kernel 不报任何错误但输出全 0。

## 判别方法

本例的定位链是三步单变量对照，可整体复用为"输出全 0 类"问题的排障模板：

1. **不变量探针先行**：用 `sum_to_one`（每切片和=1）这类语义不变量把"没写出"与"算错"分开——`sum=0.000000` 直接锁定前者。
2. **删同步重跑**：把 kernel 里全部 `AscendC::PipeBarrier<PIPE_V>()` 注释掉重建重跑 → 仍 `sum=0.000000`，排除向量 RAW 依赖/同步问题。
3. **强制走已验证路径**：把分发临时改为 ARA shape 也走 AR 路径（`ProcessOneRow(a); // TEMP: test AR path for ARA`）→ 输出变非 0（`sum=0.250750`/`0.498999`，值不对但非零），证明分发与 kernel 主干正常，问题隔离到 ARA 专用的资源/代码。
4. **资源预算减 headroom 重跑**：`avail` 减 32KB 后 ARA 路径输出非 0——锁定 UB 容量越界。

（第 3 步之后记得把临时分发改回来，再验一次。）

## 解决配方

**before（tiling host，预算全额反解）：**

```cpp
        // ARA pattern：per-chunk [rChunkLen, A1] 2D 块，per-row padded 到 32B。
        //   预算：2×rF×aAlignD×d (xBuf DB=2) + rF×aAlignF32×4 (xF32Buf) + rF×aAlignF32×4 (eBuf)
        //         + rF×aAlignD×d (yBuf) + 2×CeilAlign(aAlignF32×4, bs) (max+sum) + 16KB ≤ ubAvailable
        int64_t aAlignD = CeilAlign(in.aBundle, alignFactor);           // D_T 行 stride（padded）
        int64_t aAlignF32 = CeilAlign(in.aBundle, in.blockSize / FP32_BYTES); // fp32 行 stride（padded）
        int64_t perElemCost = 2 * aAlignD * in.elemBytes                 // xBuf DB=2 per elem
                            + 2 * aAlignF32 * FP32_BYTES;               // xF32Buf + eBuf per elem
        perElemCost += aAlignD * in.elemBytes;                           // yBuf per elem
        int64_t fixedCost = 2 * CeilAlign(aAlignF32 * FP32_BYTES, in.blockSize)  // max + sum
                          + CACHE_BUF_16KB;
        int64_t avail = in.ubAvailable - fixedCost;
        if (avail <= 0) return false;
        int64_t rImax = avail / perElemCost;  // 反解最大 rUbFactor
        int64_t cap = std::min(rImax, in.rTotal);
        rUbFactor = FloorAlign(cap, alignFactor);
```

**after（减 32KB headroom）：**

```cpp
        int64_t avail = in.ubAvailable - fixedCost - 32 * 1024; // 32KB stack/overhead headroom
```

配套加固（同一次修复中一并落到 kernel `Init`，防 VF 尾轮越界污染相邻 buffer）：所有经 `asc_vf_call` 按 `kRepF32`(64) 一轮处理的 buffer，尺寸先 `CeilAlign` 到 64 元素：

```cpp
    // VF 函数以 kRepF32(64) 元素为一轮处理，buffer 须 CeilAlign 到 64 防止尾轮越界写入
    // 污染相邻 buffer（尤其 eBuf → carryBuf）。
    const int64_t rAlign64 = CeilAlignKernel(rUbFactorAlign_, static_cast<int64_t>(kRepF32));
```

## 设计决策

- headroom 取 32KB：覆盖设备侧栈与隐式开销的经验量级；取值偏保守（相对 UB 总量占比小），不追求精确刻画。
- 修在 tiling 侧而不是 kernel 侧砍 buffer：预算责任在 tiling（它决定 rUbFactor），kernel 只是按 tiling 分配；两侧都改会造成口径分裂。
- 当前方案属 `recommended_fastpath`。

## 预防

- tiling 预算模板统一保留 headroom：`avail = ubAvailable - fixedCost - HEADROOM`，HEADROOM 显式命名并注释用途。
- 新增路径（尤其带 per-row padding 的 2D 布局）时，先在 C++ ST 加语义不变量探针（sum_to_one）再上板。
- 复用本文判别方法的三步对照，避免一上来就猜同步问题。

## 验证信号

以下信号来自同一次开发会话的 C++ ST 运行记录：

- 修复前：`WB_125/126/129 FAIL (invariant sum_to_one: sum=0.000000)`（来源平台：950；证据等级：tool_observed）。
- 排除性对照：删除全部 `PipeBarrier<PIPE_V>` 后仍 `sum=0.000000`（来源平台：950；证据等级：tool_observed）。
- 隔离性对照：强制 ARA shape 走 AR 路径后输出非 0（sum=0.250750/0.498999）（来源平台：950；证据等级：tool_observed）。
- 修复后（恢复 ARA 分发 + 32KB headroom）：ARA 三条用例输出非 0，mare/mere 进入 1e-6/1e-3 量级（WB_125 mare=4.5011e-06 mere=9.1743e-03）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 32KB 是经验值：若某算子的隐式开销超过 32KB（极深调用栈/大局部数组），同款全 0 会复现——headroom 需按工程实测放大（未验证场景）。
- 本卡只解决"贴满预算 → 全 0"；`memory_oob_status` 报越界的显式检测路径未涉及。
- 其他 NPU 平台（未验证，推断可迁移）：UB 预算不留余量导致静默失败的模式预期同样存在。

**未决项**：

- UB overflow 的硬件层机制未追查：具体哪个 buffer 越界、越界后为何表现为静默全 0 而非越界报错/脏数据——复现并追根因到机制层是接续优先项。
- 32KB headroom 的精确构成（栈 + 哪些隐式开销）未定量验证。
## 坏实践

- tiling 预算把 `ubAvailable` 精确除尽用满（"能装多少装多少"）——任何未建模的隐式开销都会变成越界，且不报错。
- 排障时先入为主怀疑向量同步（PipeBarrier），删完重跑才发现无关——同步排除法成本高，应配合强制路径对照一起做。

## 误诊

- 第一反应"ARA 向量代码有 RAW 依赖/缺同步"：删除全部 `PipeBarrier<PIPE_V>()` 后输出仍是全 0，单变量对照排除同步归因；真因在 tiling 资源预算。
- "输出全 0 = 数据没搬进来"：强制 AR 路径输出非 0 证明 CopyIn 主干与分发都正常，进一步把范围收窄到 ARA 专用资源。
