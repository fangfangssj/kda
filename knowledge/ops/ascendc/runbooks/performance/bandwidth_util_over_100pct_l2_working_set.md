---
type: "Runbook"
title: 中等 shape 带宽利用率超 100% — L2 命中假象
description: "工作集小于 L2（950PR Server 128MB）时表观 HBM 带宽可超峰值（96MB 用例实测 151.4%），测得的是 L2 有效带宽；真实 HBM 带宽须用超出 L2 的 shape 测量。"
tags: [task.optimization, topic.memory, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# 中等 shape 带宽利用率超 100% — L2 命中假象

## 现象

在 Ascend950PR（CANN 9.1.0，Server 形态，HBM 峰值带宽取 1600 GB/s——harness 沿用 npu-arch 技能文档常数的**未验证假设**，同平台 copy-only 探针实测已 >1.0x 于 1.64 TB/s 候选峰值、多源推断峰值 ~3.28 TB/s，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)；L2 128 MB）上对 BenchNoHarnessSwiGlu（带宽受限的 elementwise 算子）做性能验收，用 CPU 侧计时（每次迭代 GetWorkspaceSize + aclnnXxx + aclrtSynchronizeStream，warmup 5 + 测量 50）按 `memory_traffic / avg_time / 峰值常数` 计算带宽利用率。小/中/大三档 shape 的结果出现反物理值（原文口径，绝对时延/带宽数值为单机单次数据，此处只列工作集与利用率）：

```
TestCase                    Op           Input(MB)  Output(MB)      BW_Util(%)
medium_[4096,8192]_FP16     Custom           64.00       32.00          151.4
medium_[4096,8192]_BF16     Custom           64.00       32.00          132.2
medium_[4096,8192]_FP32     Custom          128.00       64.00           91.3
large_[8192,16384]_FP16     Custom          256.00      128.00           80.1
large_[8192,16384]_FP32     Custom          512.00      256.00           82.8
small_[1024,2048]_FP16      Custom            4.00        2.00           21.8
```

2 字节 dtype 的 medium 用例带宽利用率 151.4%/132.2%，超过 HBM 物理峰值 100%——不是测出"超规格带宽"，而是计量口径失效。

## 适用场景

- 工程形态：aclnn 自定义算子性能验收 / benchmark（以带宽利用率为达标指标的 elementwise、归一化等访存受限算子）。
- API 路径：aclrtSynchronizeStream + chrono 计时；msprof 辅助（op_summary 的 aiv_mte2_ratio 等）。
- 验证过的平台与 CANN 版本：Ascend950PR（Server，L2 128MB；HBM 峰值常数 1.6TB/s 为未验证假设），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——按 L2 容量分档选择测量 shape 的方法（阈值 128MB 为 950PR Server 值，其他芯片按各自 L2 换算）。
- 路线优先级：recommended_fastpath（作为测量方法规则）。
- 明确不适用 / 未验证场景：计算受限算子（瓶颈不在搬运，带宽口径本身不适用）；其他芯片形态（L2 容量不同，需按硬件参数重算分档阈值，未验证）。

## 判别方法

触发信号：带宽利用率 = memory_traffic / avg_time / peak_HBM > 100%。

解释（基于硬件参数的推断，非 L2 计数器直接实测）：medium FP16/BF16 单次迭代 memory_traffic = 64MB 输入 + 32MB 输出 = 96MB，**小于 128MB L2**——50 次重复测量同一输入时，输入侧（乃至输出写回）大量命中 L2，实际 HBM 流量远小于名义 96MB，分母时间随之缩短，名义带宽被高估。对照证据：medium FP32 的 traffic 为 192MB > 128MB L2，无法整段驻留，利用率回落到 91.3%（物理合理）；large 档（384MB~768MB）全部在 80~83%；small 档 6MB 因启动开销与并行度不足仅 21.8%。

因此**测真实 HBM 带宽利用率的 shape 必须满足：单迭代 memory_traffic（输入+输出）> L2 容量**，本平台即 >128MB（取 2 倍余量更稳，如本例 large 档 384MB+）。

峰值口径敏感性：上表利用率均按 1.6TB/s 未验证常数折算；若改按同平台推断的 ~3.28 TB/s 口径（见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)），各档数值约减半（medium FP16 约 74%、BF16 约 65%、large 档约 39~40%）——">100% 反物理"这一触发信号依赖分母取值，但"工作集 < L2 时口径失效、测得的是 L2 有效带宽"的机制结论与分母取值无关，且 large 档相对 medium 档的回落趋势在两种口径下一致。

## 根因

计时口径假设全部 memory_traffic 都走 HBM；当工作集可驻留 L2 时，重复测量的稳态流量大部分在 L2 完成，`名义流量/时间` 得到的是 L2 有效带宽（可高于 HBM 峰值），带宽容收指标失去意义。

## 预防

- 带宽验收用例设计分三档：small（启动开销观察）、L2 内（观察缓存收益，不用于 HBM 带宽结论）、**超过 L2 的 large（唯一用于 HBM 带宽利用率达标的档位）**。
- 报告中带宽利用率 >100% 的条目必须标注"L2 命中口径失效"，不得当作性能达标证据。
- 硬件参数（L2 容量、HBM 峰值）以芯片形态为准（Server/板卡不同），先查 npu 硬件参数再定档。

## 坏实践

- 用 medium 档的 151% 带宽利用率作为"超过 HBM 峰值/远超达标线"的验收证据。
- 用同一输入反复迭代测带宽而不检查工作集与 L2 的相对大小。

## 误诊

- 初始把 medium 151% 解读为"计时含 GetWorkspaceSize 开销导致"——方向反了：额外开销只会让表观带宽更低，解释不了超 100%。

## 验证信号

来源：性能验收会话，9 用例×2 算子计时数据与 msprof 交叉验证。

- 触发观测：medium_[4096,8192]_FP16 BW_Util 151.4%、BF16 132.2%（名义 traffic 96MB < 128MB L2）；同档 FP32（192MB > L2）仅 91.3%，large 档 80.1~82.8%（均按 1.6TB/s 未验证峰值常数折算；按 ~3.28TB/s 推断口径约减半，见 getcoremembw 卡）（来源平台：950；证据等级：tool_observed）。
- 交叉验证（large 档 FP16，msprof op_summary）：kernel Task Duration 与 CPU 计时同量级（互差约 5%，排除计时口径错位）；aiv_mte2_ratio≈97.8%（MTE2 搬运引擎近满载），佐证 large 档确为 HBM 带宽受限、计时口径有效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 128MB L2 阈值绑定 Ascend950PR Server 形态；其他芯片/形态按实际 L2 容量换算（未验证，推断可迁移的是"工作集须超 L2"的方法本身）。
- "96MB<128MB 命中 L2"的机制解释为基于硬件参数文档的推断，未提取 L2 命中率计数器直接验证（遗留问题首项）。
- warmup 后输出缓冲若被复用，输出侧也可能驻留 L2；保守做法是按 输入+输出 总量都计入工作集（本卡口径）。
- 未验证场景：多算子交替执行破坏 L2 驻留时的测量行为。

**未决项**：

- 用 msprof 的 L2 命中率/缓存计数器直接证实该解释，把"推断"升级为"实测"。
