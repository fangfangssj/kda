---
type: "Runbook"
title: aclnn ST 大 shape 宿主 oracle 超时，同路径缩 shape
description: 测试先行 ST 的期望值由宿主侧逐元素 oracle 现算，1e8 元素级用例把单次全量运行拖过 120s 工具超时；把大 shape 降为保持同一 tiling/广播路径覆盖的小 shape（并注明原始 shape）后，整套 48 用例在 90s 内完整跑完。复现增量：毒化+回读+转换的数据搬运成本模型（~4GB 设备侧移动）与 banned/safety_fallback 档位——顶格形只进性能验收，代表形按覆盖口径缩形。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T01:05:00Z'
platforms: ["950"]
---

# aclnn ST 大 shape 宿主 oracle 超时，同路径缩 shape

## 现象

在为 softplus_v2_grad 编写 RANK_4 分支的 aclnn ST（测试先行，48 用例、12 类场景，kernel 为 stub，期望 0/48 基线）时：首次运行用 `./build/test_aclnn 2>&1 | head -80` 能看到用例逐条输出全 mismatch（C01/C02/C03…）；紧接着用 `| tail -30` 取完整汇总输出时，命令被工具终止：

```text
(no output)

<bash_metadata>
bash tool terminated command after exceeding timeout 120000 ms.
```

注意首次"能跑完"是假象——`head -80` 读满行数后关闭管道，测试进程被提前掐断；换 `tail` 需要等全量输出，真实运行时间远超 120s。

## 适用场景

- 工程形态：aclnn 自定义算子的白盒/分支 ST（自建 `test_aclnn.cpp`，宿主生成数据 + 宿主算期望 + 上板取回比较），尤其测试先行阶段（kernel 未实现，全部耗时在宿主侧）。
- API 路径：aclrtMemcpy H2D/D2H、aclnn 两段式调用；测试进程内逐元素 oracle（宿主 libm）。
- 验证过的平台与 CANN 版本：Ascend950PR（HBM 128GB、宿主内存约 773GB——内存不是瓶颈，时间才是），CANN 9.1.0（与 frontmatter 一致）。
- 适用粒度：operator_family——ST 用例 shape 选取策略，与具体算子无关。
- 路线优先级：recommended_fastpath（路径覆盖型 ST 的默认 shape 策略）。
- 明确不适用 / 未验证场景：性能/吞吐验证（必须用目标规模大 shape + 更长超时/后台运行，本卡缩减策略不适用）；用例目的是压测边界规模（如逼近 int32 索引上限）时不能缩。

## 根因

ST 的数值期望由宿主侧逐元素 oracle 现场计算，而用例 shape 按"最终目标场景"（1e8 元素）而非"路径覆盖所需"选取，宿主计算加搬运的耗时远超执行工具超时。

机制链（以本例实测 shape 演算）：主块满载用例 `[32,64,224,224]` = 32×64×224×224 = 102,760,448 元素（约 1e8）。该规模用例共 12 个：C01 主块满载 3 个 dtype 变体 + C06 多维广播 9 个 dtype 组合。每元素宿主要做一次 `expf` + 一次除法 + 一次乘法 + 比较与 dtype 舍入往返，合计约 12 × 1.03e8 ≈ 1.2×10^9 次宿主浮点计算；外加每用例 3 个张量（2 入 1 出）× 1e8 元素 × 2–4 字节的 H2D/D2H 搬运（fp32 单用例约 1.2GB）。宿主串行算下来即分钟级，超 120s 工具上限。会话内的耗时定位为分析推算（每元素成本为量级估计），但"12 个 1e8 级用例 → 超时"与"缩减后 90s 内跑完"两端均为实测。

## 解决配方

修复前（用例按目标场景规模选 shape，`tests/aclnn/test_aclnn.cpp`）：

```cpp
    // ---- Class 1: 主块满载 [32,64,224,224] (rank=4, ~1e8) ----
    for (int i : kSameDtypeIdx) {
        CaseDesc c = makeRandom("C01_main_block [32,64,224,224]",
                                {32,64,224,224}, {32,64,224,224}, kDtypeCombos[i]);
        cases.push_back(c);
    }
```

```cpp
    // ---- Class 6: 多维广播 [1,64,1,224]⊗[32,64,224,224] (rank=4) — all 9 combos ----
    for (int i = 0; i < kNumDtypeCombos; ++i) {
        CaseDesc c = makeRandom("C06_multi_broadcast [1,64,1,224]⊗[32,64,224,224]",
                                {1,64,1,224}, {32,64,224,224}, kDtypeCombos[i]);
        cases.push_back(c);
    }
```

修复后（缩为同路径小 shape，注释保留原始 shape 与缩减理由，verbatim）：

```cpp
    // ---- Class 1: 主块满载 (rank=4, multi-core main-block path) ----
    // §9 shape [32,64,224,224] (~1e8) reduced to [8,16,56,56] (~400K) for
    // practical ST runtime; still exercises rank=4 main-block a_i path and
    // multi-core tiling (DESIGN-BRANCH §9 class 1).
    for (int i : kSameDtypeIdx) {
        CaseDesc c = makeRandom("C01_main_block [8,16,56,56]",
                                {8,16,56,56}, {8,16,56,56}, kDtypeCombos[i]);
        cases.push_back(c);
    }
```

```cpp
    // ---- Class 6: 多维广播 (rank=4, 4D NDDMA broadcast) ----
    // §9 shape [1,64,1,224]⊗[32,64,224,224] (~1e8) reduced to
    // [1,16,1,56]⊗[8,16,56,56] (~400K) for practical ST runtime; still
    // exercises 4D broadcast with NDDMA single-coverage (§9 class 6).
    for (int i = 0; i < kNumDtypeCombos; ++i) {
        CaseDesc c = makeRandom("C06_multi_broadcast [1,16,1,56]ot[8,16,56,56]",
                                {1,16,1,56}, {8,16,56,56}, kDtypeCombos[i]);
        cases.push_back(c);
    }
```

缩减原则：

1. **保路径**：缩的是元素量，不是用例类。`[8,16,56,56]`（约 4e5 元素）仍满足 rank=4、主块 a_i 切分、多核负载均衡的触发条件；`[1,16,1,56]⊗[8,16,56,56]` 仍触发 4D 广播 + 广播轴 stride=0。判断标准回到该用例类存在的目的（覆盖哪条 tiling/数据流路径），与"路径覆盖所需最小规模"对齐。
2. **留痕**：用例注释里保留 §9 原始 shape、缩减后 shape 与理由，后续性能阶段可还原。
3. **保判据**：不删用例类、不放宽数值比较——期望 0/48 基线本身不变。

## 设计决策

- 为什么缩 shape 而不是加大工具超时：48 用例 × 1e8 的宿主 oracle 是分钟级纯浪费（stub 阶段结果恒 mismatch，规模不增加信息量）；路径覆盖所需规模 4e5 已足够，运行时间从 >120s 降到秒级，反馈环更快。
- 为什么不砍 dtype 组合：dtype 组合（9 种）是独立的覆盖维度（混合精度提升路径），与 shape 规模正交，砍组合丢覆盖、缩 shape 不丢。
- 放弃的候选：全部用原 shape + 后台跑 + 轮询（对测试先行基线而言过度工程）；只跑 head 截断的前几条（掩盖真实运行时间，且汇总输出拿不到）。
- 当前方案路线档位：recommended_fastpath（路径覆盖型 ST）；性能验证场景为 banned（不得用缩减 shape 冒充性能结论）。

## 预防

- 写 ST 用例表时按"每用例类的路径目的"给 shape，默认给路径覆盖级规模，目标场景级大 shape 只进性能/压测用例集并单独配超时。
- 估算宿主成本：用例数 × 元素数 ≈ 宿主 oracle 调用次数，超过 1e8 量级就要警惕工具超时（120s 默认）。
- 取完整输出用 `tail`/重定向到文件再读，别用 `head`——管道提前关闭会制造"跑完了"的假象。

## 验证信号

本次开发会话在 Ascend950PR + CANN 9.1.0 上的实测：

- 修复前：全量输出运行超过 120000ms 被工具终止（来源平台：950；证据等级：tool_observed）。
- 缩减后重编译（增量 make 通过）+ `timeout 90 ./build/test_aclnn`：48 用例完整跑完，`==== ST summary: 0/48 PASSED, 48 FAILED ====`（stub 基线预期），运行时间进入 90s 限时以内（来源平台：950；证据等级：tool_observed）。
- 覆盖不缩水：12 类用例齐全（C01–C12）、48 条（3+3+3+9+9+9+1+3+2+2+3+1），dtype 组合全保留（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "同路径缩 shape"的前提是能用明确的路径判据论证等价（本例：rank=4、主块切分、多核、4D 广播）；无法给出路径论证的用例（如规模触发的边界行为：溢出、地址回绕、索引上限）不能缩。
- stub/测试先行阶段宿主 oracle 是瓶颈；kernel 实现后计算移到 NPU 侧，宿主只剩搬运+比较（仍随 shape 线性），1e8 级用例需重新实测耗时并配超时，本卡的量级结论不直接外推。
- 具体阈值（120s）是执行工具默认超时，量级推算按宿主串行浮点估算，不同宿主性能下数值不同——判据应落在"估算宿主 oracle 总调用次数"而非具体秒数。

**未决项**：

- 未尝试：把宿主 oracle 向量化/并行化以支撑 1e8 级用例的常驻运行（当前用不到，性能阶段再评估）。
## 坏实践

- 为凑运行时间删用例类或砍 dtype 组合（丢覆盖维度）。
- 用 `head` 截断的输出当作"测试跑过"的证据（管道关闭的假完成）。
- 在性能验证场景沿用缩减 shape 冒充目标规模结论（banned）。

## 增量（数据搬运成本模型与 banned 档位）

另一条独立轨迹（RoPE 类算子分支 ST，测试先行，Ascend950PR / CANN 9.2.0）以"输出毒化 + 全量回读 + 逐点转换比对"框架复现同一约束并补充两点：

**1) 数据搬运成本模型（宿主 oracle 之外的第二个超时源）**：设计文档顶格形（536,870,912 元素/张量，fp16）单用例要走 H2D 上传、输出毒化 H2D、D2H 回读，host 侧还要把实际值与期望值各转一遍 float 再逐点比对——6 张量合计约 4GB 设备侧移动 + 约 8GB host 侧转换/比对内存流量（成本模型为推算，实测锚点是超时与缩形后跑完两端）。同覆盖口径代表形（[8,1024,64,128]=67M 元素）后全量 12 用例跑完并打印逐用例结果。

**2) 路线档位与缩形纪律**：对"顶格形进每轮 ST 精度回归"标 **banned**（顶格形留性能/极限验收单独跑，可调大超时）；代表形为 safety_fallback（牺牲极限规模验证换迭代速度）。缩形不缩覆盖：逐项对齐原用例意图（本例 totalRows=524288 仍 ≥32 核满切、保留 512 对齐尾块、dtype 不变）——随手取整（如缩到 totalRows 不足以满切多核）会丢掉该用例存在的意义。

验证信号（复现增量）：

- 顶格形实测：`[gen] b4_large_fp16 ... (q 536870912 elems)` 后测试运行被终止：`shell tool terminated command after exceeding timeout 120000 ms`，无任何用例结果输出（来源平台：950；证据等级：tool_observed）。
- 代表形实测：`q=[8, 1024, 64, 128] ... (q 67108864 elems)`，全量 12 用例跑完并打印逐用例结果与总结；覆盖口径对照：totalRows=524288 仍满足 ≥32 核满切与 512 对齐尾块条件（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- ~4GB/8GB 数据流量为成本模型推算（agent_reported），实测锚点仅为超时发生与缩形后跑完两件事。
- 该轨迹的瓶颈在数据搬运而非宿主 oracle 计算（kernel 为 stub 仍超时，移除大用例后整套秒级跑完）——与主卡的宿主 oracle 计算瓶颈互补：估算单用例成本时按"宿主逐元素计算 + 三段数据搬运 + host 侧转换比对"三项合计。
- 若框架改为抽样比对或 device 侧比对，成本模型变化需重新评估（未验证）。