---
type: "Runbook"
title: "SoftMax srcK须等于缓冲区行步长"
description: "UB 行按 topkInner=CeilAlign(EAlign,32) 排布后，SoftMax shapeInfo.srcK 仍为 EAlign，按 416B 步长读 512B 布局的行全错位；srcK/tiling srcShape 改用 topkInner（oriSrcK=E）修复非对齐 E。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# SoftMax srcK须等于缓冲区行步长

## 现象

Softmax+TopK kernel 修完行步长契约（见同批 `topk_row_stride_misalignment_507035`）后 ST 到 29/45，剩余失败里有一组稳定的"非对齐 E"精度错：E=33/100 的 fp16/bf16/fp32 用例 y 失败，MERE 从 2.9e-02 到 6.7e+02 不等。失败指纹很有辨识度——**TopK 选错了元素**而非数值不准：

```
（fp32, E=100, k=16 的用例）
  不匹配 [60]: 期望=0.028322, 实际=0.026905   ← actual[60] == expected[62]
  不匹配 [62]: 期望=0.026905, 实际=0.021105   ← actual[62] 不在期望 top-16 内
```

行内前段（第 0 行）正确、错位集中在后续行，且对齐 E 的用例全部通过——指向"第 2 行起 SoftMax 读的不是本行数据"。

## 适用场景

- 工程形态：kernel 内 `SoftMax`（adv_api）+ `TopK` 串联，输入行按对齐宽度 `topkInner`（32 的倍数）排布在 UB 的算子。
- API 路径：`SoftMax<COMPUTE_T, ...>(dst, src, sharedTmp, tiling, shapeInfo)`；`SoftMaxShapeInfo{srcM, srcK, oriSrcM, oriSrcK}`（`asc/include/adv_api/activation/softmax_utils.h`）；`SoftMaxTilingFunc`/`GetSoftMaxMaxTmpSize`（tiling 侧 srcShape）。
- 验证过的平台与 CANN 版本：Ascend950PR (arch35) + CANN 9.1.0。
- 适用粒度：`operator_family`（SoftMax shapeInfo 与自管 UB 布局解耦的通病）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：SoftMax src/dst 由 API 自己管理布局（不经用户 UB 重排）的用法不涉及本卡；`oriSrcK` 语义细节见失效边界。

## 根因

为满足 TopK `inner % 32 == 0` 约束，缓冲区行步长扩大为 `topkInner = CeilAlign(EAlign, 32)`（元素），但 SoftMax 的 `shapeInfo.srcK` 仍写 `EAlign`。**SoftMax 以 srcK 为行步长读取**：以 E=100（EAlign=104、topkInner=128、COMPUTE_T=float）为例，行数据实际按 `128*4=512` 字节排布，SoftMax 却按 `104*4=416` 字节跳行——第 0 行恰好正确，第 1 行起读到上一行尾 + 下一行头的拼接数据，SoftMax 输出随之错乱，TopK 在错乱数据上"正确地"选出了错误元素。

## 解决配方

让 SoftMax 的行步长与缓冲区实际布局一致，用 `oriSrcK` 表达有效宽度：

kernel 侧 `Init()`/`ComputeBatch()` 的 shapeInfo（before → after）：

```cpp
// 修复前：srcK=EAlign（与 topkInner 行布局不一致）
    // SoftMax shape info (初始值，实际执行时按 curBatch 更新)
    softmaxShapeInfo_.srcM = static_cast<uint32_t>(batchRows_);
    softmaxShapeInfo_.srcK = static_cast<uint32_t>(EAlign_);
    softmaxShapeInfo_.oriSrcM = static_cast<uint32_t>(batchRows_);
    softmaxShapeInfo_.oriSrcK = static_cast<uint32_t>(E_);
```

```cpp
// 修复后：srcK = topkInner_ (行步长, 匹配缓冲区布局), oriSrcK = E_ (仅归一化有效元素)
    // SoftMax shape info (初始值，实际执行时按 curBatch 更新)
    // srcK = topkInner_ (行步长, 匹配缓冲区布局), oriSrcK = E_ (仅归一化有效元素)
    softmaxShapeInfo_.srcM = static_cast<uint32_t>(batchRows_);
    softmaxShapeInfo_.srcK = static_cast<uint32_t>(topkInner_);
    softmaxShapeInfo_.oriSrcM = static_cast<uint32_t>(batchRows_);
    softmaxShapeInfo_.oriSrcK = static_cast<uint32_t>(E_);
```

（`ComputeBatch` 内对尾批的 `srcM/oriSrcM = curBatch` 更新保持不变。）

tiling 侧同步——SoftMax 的 tmp/tiling 计算所用 srcShape 也必须按 topkInner（before → after）：

```cpp
// 修复前（估算与重算都用 EAlign）
    ge::Shape estShape({1, EAlign});
    uint32_t softmaxMaxTmp = GetSoftMaxMaxTmpSize(estShape, computeDtypeSize, false);
    ...
    ge::Shape srcShape({static_cast<int64_t>(batchRows), EAlign});
    softmaxMinTmp = GetSoftMaxMinTmpSize(srcShape, computeDtypeSize, false);
    softmaxMaxTmp = GetSoftMaxMaxTmpSize(srcShape, computeDtypeSize, false);
```

```cpp
// 修复后（估算与重算都用 topkInner）
    // SoftMax srcK = topkInner (行步长), TopK inner = topkInner
    ge::Shape estShape({1, topkInner});
    uint32_t softmaxMaxTmp = GetSoftMaxMaxTmpSize(estShape, computeDtypeSize, false);
    uint32_t softmaxMinTmp = GetSoftMaxMinTmpSize(estShape, computeDtypeSize, false);
    ...
    // 6. 使用实际 batchRows 重新计算 SoftMax/TopK tiling
    // SoftMax srcK = topkInner, TopK inner = topkInner
    ge::Shape srcShape({static_cast<int64_t>(batchRows), topkInner});
    softmaxMinTmp = GetSoftMaxMinTmpSize(srcShape, computeDtypeSize, false);
    softmaxMaxTmp = GetSoftMaxMaxTmpSize(srcShape, computeDtypeSize, false);
```

配套：TopK 侧 `topKInfo_.inner = topkInner_`，-inf padding 覆盖 `E..topkInner-1`（含 EAlign..topkInner 的未初始化区），保证 TopK 只在有效元素里选。

## 设计决策

- 选 `srcK=topkInner + oriSrcK=E`（行步长与有效宽度分离）而非把 EAlign 也提到 32 对齐：后者会无差别放大所有用例的缓冲区（E=100 → 128），代价更高；topkInner 方案只影响 SoftMax 读行方式。
- 该决策是被实验逼出来的：先试过"EAlign 强制 32 对齐"（E=33→64、E=100→128），结果 16 个失败用例**全部**变成流同步失败 507035（缓冲区放大引发别的问题），回退后才转向 srcK 方案。
- 同批还试验并**回退**了另一个方案（见坏实践）：SoftMax 前填 -1e30 + `oriSrcK=EAlign`。

## 预防

- `SoftMaxShapeInfo.srcK` 的语义是**行步长**（含 padding），不是有效宽度——任何自管 UB 行布局的改动（为其他 API 对齐而扩大步长）都必须同步 srcK 与 tiling srcShape。
- "第 0 行对、后续行错"是行步长错位的固定指纹，出现时直接查两个宽度是否同源。

## 验证信号

来源说明：问题与验证来自同一 NPU ST 联调会话（srcK=EAlign 时非对齐 E 用例失败；`srcK=topkInner` + tiling srcShape 同步后同批重建对比）。

- 修复前：非对齐 E 用例 y 失败（MERE 2.9e-02~6.7e+02，actual[60]=expected[62] 型选错）（来源平台：950；证据等级：tool_observed）。
- 机制推导（本例实测数值）：E=100 时行布局 512B vs SoftMax 读步长 416B，`(128*4 - 104*4)/32 = 3` 个 32B 块差（来源平台：950；证据等级：artifact_verified，事件内推导算式）。
- 修复后：ST 29/45 → 33/45，四个非对齐 E 用例全部转 PASS（其余失败为 k=E/大规模/fp16 大E 独立根因，见相关卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡解决"行步长与布局不一致"；`oriSrcK=E` 时 SoftMax 对 padding 区（值 0）的处理语义（是否入 sum/max）本轨迹未在 API 层验证——fp16 大 E 近阈值的残留问题与此相关（见同批 `fp16_large_e_threshold_margin_unresolved`）。
- topkInner 方案对 E 已 32 对齐的用例无行为变化（topkInner==EAlign）。

**未决项**：

- `SoftMax` 在 `oriSrcK < srcK` 时对 padding 元素的内部处理（sum/max 是否包含）未验证；这直接影响 fp16 大 E 近阈值问题（见相关卡），建议用最小 kernel 实验锚定。
## 坏实践

- **EAlign 强制 32 对齐方案**（把非对齐 E 的对齐宽度直接提到 32 的倍数）：实测使 16 个失败用例全部转为流同步失败 507035（含原本只是精度错的用例），整体更糟，已回退。
- **SoftMax 前填 -1e30 + oriSrcK=EAlign 方案**（想让 padding 经 exp 归零）：实测同样导致全部失败用例变 507035，回退。两个方案共享同一教训：改 SoftMax 的 shape/padding 语义前先确认 API 对极端值的内部处理，-inf/-1e30 类填充在本环境会引发 kernel 异常。

## 误诊

- 曾把非对齐 E 失败归因为"TopK tie-breaking/排序顺序不同"（值近似但顺序不同），对照 actual[60]==expected[62] 后确认为选错元素，进而定位到 SoftMax 行错位（该转向过程为推理叙述；最终机制有 416/512 字节算式支撑）。
