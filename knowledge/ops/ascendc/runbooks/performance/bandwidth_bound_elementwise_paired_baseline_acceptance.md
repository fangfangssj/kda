---
type: "Runbook"
title: 带宽受限算子验收画像：对标官方 aclnn 基线
description: "950PR 上自研 elementwise kernel 对标官方 aclnnSwiGlu 为 0.99x~1.00x（大 shape）；msprof 画像 aiv_mte2_ratio≈98%、aiv_vec_ratio≈30% 即“计算藏于访存”的健康形态，验收应成对对标基线。"
tags: [task.optimization, topic.compute.elementwise.activation, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# 带宽受限算子验收画像：对标官方 aclnn 基线

## 现象

在 Ascend950PR（CANN 9.1.0，Server 形态；核数口径以实测为准：msprof op_summary Block Num=56（VectorCore）、npu-smi AICore Count 28 + 平台查询 AIV 56，即 28 AICore + 56 AIV；HBM 峰值取 harness 常数 1600 GB/s——源自 npu-arch 技能文档的**未验证假设**，同平台 copy-only 探针实测已 >1.0x 于 1.64 TB/s 候选峰值、多源推断峰值 ~3.28 TB/s，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）上为自定义算子 BenchNoHarnessSwiGlu（output = silu(x0)·x1，x0/x1=input.chunk(2,dim)，内部 fp32 升精度计算）做性能达标验收。验收方法：C++ 原生 harness 同 shape、同 dtype、同流成对测量自研 aclnnBenchNoHarnessSwiGlu 与官方基线 aclnnSwiGlu（libopapi 内置），warmup 5 + 测量 50；另用 msprof 采集 AI Core 指标画像。测量结果（成对相对对比，参照路线=官方 aclnnSwiGlu；绝对时延为单机单次数据不列入）：

```
TestCase                        Speedup      Custom BW%    Baseline BW%
small_[1024,2048]_FP16            0.76x            21.8            28.8
small_[1024,2048]_FP32            0.73x            41.5            57.0
medium_[4096,8192]_FP16           1.05x           151.4*          144.5*
large_[8192,16384]_FP16           0.99x            80.1            80.6
large_[8192,16384]_FP32           1.00x            82.8            82.6
large_[8192,16384]_BF16           0.99x            80.1            80.6
（*medium 档 BW% 因工作集 <128MB L2 口径失效，见[带宽利用率超 100% — L2 命中假象](bandwidth_util_over_100pct_l2_working_set.md)；Speedup 列不受影响。BW% 列的分母为未验证的 1.6TB/s 峰值常数，仅作画像参照，达标结论以 Speedup 成对对比与 MTE2 近满载为准）
```

msprof 大 shape FP16 画像（op_summary/op_statistic，15 次采样）：Block Num=56（VectorCore）；aiv_mte2_ratio≈97.8%（GM→UB 搬运引擎占用）、aiv_mte3_ratio≈72.7%（UB→GM）、aiv_vec_ratio≈29.5%（向量计算）、aiv_scalar_ratio≈12.3%、cube_utilization=0%。

## 适用场景

- 工程形态：aclnn 自定义算子性能验收（对标官方同功能算子）；带宽受限 elementwise 算子的 profiling 判读。
- API 路径：aclnn 两段式调用 harness；msprof（`--application=... --output=...` 采集，mindstudio_profiler_output/op_summary CSV + device_0/sqlite 的 ai_core_metrics 表）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——带宽受限 elementwise（激活/逐元素算术）类自研 kernel 的验收画像与达标判据。
- 路线优先级：recommended_fastpath（成对基线对标为验收首选口径）。
- 明确不适用 / 未验证场景：计算受限算子（vec_ratio 高、mte 低，画像判据不同）；Cube/MIX 类算子（本画像纯 Vector）；小 shape 绝对时延结论（受 GetWorkspaceSize 与启动开销影响，见失效边界）。

## 判别方法

带宽受限 elementwise 算子健康画像（本轨迹实测值，作判读参照）：

- **aiv_mte2_ratio ≈ 98%（近满载）**：符合带宽受限预期，是"搬运是瓶颈"的**必要但不充分**判据——MTE2 泳道占比高同样出现在"逐条小搬运的发射/描述符开销 bound"形态（有效带宽距 HBM 能力低 1~2 个数量级），须结合有效带宽/搬运粒度区分，见[MTE2占比高≠带宽bound：发射开销判别](mte2_issue_overhead_vs_bandwidth_bound.md)；
- **aiv_vec_ratio ≈ 30%**：计算与访存充分重叠，计算藏于访存之后——这是"带宽受限算子的理想状态"，不是"算力浪费"；
- **cube_utilization = 0%**：纯 Vector 算子正常；
- 计时侧 HBM 带宽利用率（large 档，按未验证的 1.6TB/s 峰值常数折算）80~83%，与 mte2 近满载互相印证——该列只作辅助画像，不作为达标主证据。

"AI Core 利用率"在此类算子上没有单一指标，应看引擎分项：搬运引擎占用率（mte2/mte3）+ 带宽利用率共同构成达标证据。

## 根因

自研 kernel 与官方 aclnnSwiGlu 在大 shape 下同受 HBM 带宽约束（双方 BW% 均在 80~83%、互差 <1%——同一未验证分母下的相互一致性，非对绝对峰值的证明），时延由搬运决定、计算实现差异被完全掩盖，故成对 Speedup≈1.0x 是"实现已达平台带宽上限"的表征；小 shape 下 host 侧调用开销（GetWorkspaceSize + launch）占比升高，自研路径每次迭代完整重建上下文（host 侧两段式调用的必要开销）略多于官方路径调用成本，成对 0.73~0.76x。

## 解决配方

验收口径（可直接复用的方法）：

1. 同 shape×dtype×stream 成对测量自研与官方基线（基线从 `aclnnop/level2/` 头 + libopapi.so 链接），Speedup=Baseline_time/Custom_time 作为主指标；
2. large 档（工作集超 L2）的 HBM 带宽利用率 = (输入字节+输出字节)/avg_time/峰值带宽 作为辅助指标（峰值常数须注明来源与验证状态）；
3. msprof 采样一轮取 aiv_mte2/mte3/vec_ratio 画像佐证瓶颈类型；
4. 达标判据示例（本轨迹采用）：对标基线 ≥0.95x（主判据）、搬运引擎占用 >80%、带宽利用率（大 shape，按注明口径的峰值常数折算）>70%（辅助指标，结论依赖分母口径）。

## 系统信息

- 平台：Ascend950PR（9579，Server 形态），arch35/DAV_3510；CANN 9.1.0。
- 硬件参数（npu-arch 技能文档，**未经本轨迹实测验证**）：HBM 128GB/峰值 1.6TB/s，L2 128MB，Vector 算力 54 TOPS。核数口径以实测为准：msprof op_summary Block Num=56（VectorCore）、npu-smi AICore Count 28、平台查询 AIV 56（即 28 AICore + 56 AIV）；技能文档的"32 Cube + 64 Vector"与本平台实测口径不符，引用时须以实测为准。HBM 峰值 1.6TB/s 同为未验证常数——同平台 copy-only 探针实测已 >1.0x 于 1.64 TB/s 候选峰值、多源推断峰值 ~3.28 TB/s，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)。
- 相关文件：`tests/perf/perf_test.cpp`、`tests/perf/CMakeLists.txt`（链接 libascendcl/libopapi/libnnopbase + vendors cust_opapi）、`tests/perf/run.sh`、`tests/perf/msprof_op_summary_large_fp16.csv`、`tests/perf/msprof_op_statistic_large_fp16.csv`、`docs/performance-report.md`。
- 环境要点：运行前 `export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:.../bench_no_harness_swi_glu_custom/op_api/lib:$LD_LIBRARY_PATH`。

## 设计决策

- 选官方 aclnnSwiGlu 为基线而非理论峰值单指标：成对对比能同时暴露实现差距与口径问题（如 medium 档双方 BW% 同时失效反而互相印证）；理论带宽只作分母。
- 三档 shape 设计：small 看启动开销、medium 观察 L2 效应（不用于带宽结论）、large 用于 HBM 带宽达标——档位语义先于数值。
- msprof 用独立精简 profile_test（仅大 shape FP16 循环）采集，避免全用例混入导致 op_summary 噪声。

## 性能影响

- 自研相对官方 aclnnSwiGlu 基线（成对，来源平台：950）：大 shape 0.99x~1.00x、medium FP16 1.05x / BF16 0.93x、small 0.73x~0.76x。小 shape 差距来自每迭代重建调用上下文的必要开销，非 kernel 实现差异。

## 坏实践

- 只报自研绝对时延/带宽而不带同条件基线——无法区分"平台上限"与"实现差距"。
- 把 aiv_vec_ratio≈30% 当作"AI Core 利用率不达标"去优化计算——带宽受限算子 vec_ratio 低是健康形态，优化方向应确认 mte2 是否已近满载。

## 验证信号

来源：性能验收会话，harness 全量计时 + msprof 采样。

- 成对对比数据：9 用例 × 2 算子全量出数（上表，Speedup 0.73x~1.05x）（来源平台：950；证据等级：tool_observed）。
- msprof 实测（大 shape FP16，15 次采样）：aiv_vec_ratio 0.294~0.298、aiv_mte2_ratio 0.966~0.982、aiv_mte3_ratio 0.677~0.748、Block Num 56（op_statistic）（来源平台：950；证据等级：tool_observed）。
- 精度未退化：同算子 ST Real 115/115（含 12 非连续）通过，本性能路径未改 kernel（来源平台：950；证据等级：tool_observed）。
注：绝对 ms/GB/s 数值为单次会话单机数据，仅作画像参照；跨机器复用只取成对相对比值与引擎比例。

## 失效边界

- 画像判据（mte2≈98% + vec≈30%）绑定带宽受限 elementwise 类；计算受限算子应看 vec/mac 比例，不可套用。
- 本轨迹 harness 计时窗口含每迭代 GetWorkspaceSize（毫秒以下量级），对 small 档绝对值有抬高；引用 small 档数据须注明口径。
- 未验证场景：其他 CANN 版本/芯片的 msprof 字段（aiv_* 列名与语义）；BF16 medium 的 0.93x 差距是否为官方内核特化。

**未决项**：

- small shape 下自研路径相对官方的每迭代 host 开销差距（成对 0.73~0.76x 的主要构成）能否通过官方批量/缓存调用机制消除，未探索。
