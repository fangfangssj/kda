---
type: Guide
title: msprof CSV 字段、单位与诊断边界
description: 保存八类 CSV 的可用指标及交叉分析方式，纠正单阈值根因判定。
tags:
- task.debugging
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/profiling_reference/INDEX.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/profiling_reference/csv_fields_reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/profiling_reference/optimization_quickref.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/references/csv_fields_reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/references/optimization_quickref.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
---

# msprof CSV 字段、单位与诊断边界

来源描述MindStudio8.3.0/A2A3导出的八类CSV，实际名称和字段取决于采集版本/模式，缺列不能当0。OpBasicInfo含Op Name/Type、Task Duration(us)、Block Dim/Mix Block Dim、Device ID、PID、Current/Rated Freq；block_id是行所属block标识而非该行字段本身的总核数，sub_block_id标子执行单元。Task与每核aic/aiv_time、total_cycles的范围不能混合。

PipeUtilization的aiv_vec、aic_cube、ai*_scalar、aic_fixpipe、aic_mte1、ai*_mte2/mte3各有time/ratio，部分模式还导出active_bw及icache_miss_rate。MTE2是GM到核内对应存储的搬入，不是来源标签表的“Scalar→UB”。各pipe活动可重叠，ratio之和不必等于1；active_bw分母与整task带宽不同，不能直接相加各核活跃峰值当整卡吞吐。

ArithmeticUtilization记录Cube FP16/INT8比例、fops、total/fp/int指令数，Vector FP32/FP16/INT32/INT16/misc比例及fops。高FP32比例并不证明都是FP16→FP32 Cast，也不独立证明计算满吞吐。Memory有GM↔UB、L1读写、主存读写bandwidth，MTE指令数与ratio，主存/GM↔L1/GM↔UB/L0C→L1/GM的数据量和bw_usage_rate；estimate列须保留估算标志。

MemoryL0区分L0A/L0B读写和Cube侧L0C读写；MemoryUB区分Vector/Scalar访问UB的读写。L2Cache含写hit/miss_allocate、分读通道hit/miss_allocate及read/write/total hit rate，权重需原始计数而非各百分比简单平均。低命中对单次流式读取可正常，高命中也不证明没有瓶颈，CacheMode策略需要具体数据复用与版本API。

ResourceConflictRatio在来源采集描述中为上板指标，含vec_total、bankgroup、bank、resc、mte冲突占比和Cube/Vec/MTE等待。总冲突不等于全部bank冲突，也不能推断不同原因占比互斥。地址padding/block或repeat stride改变前后都需重验数据映射，源“48 bank/16group/每bank32B”未固定足够平台证据，不作为通用UB结构。

将KB转换为字节需确认导出采用1024还是1000，除Byte/s得到秒，与us比较还需乘1e6。请求平均字节数=同范围数据量/非零指令数，不能跨task或把read_main_memory与某核GM→UB指令数随意配对。峰值整卡/单核、HBM/L2有效流量应一致，热点资源可能局部饱和而全局平均偏低。

vec>50%、scalar>30%、mte2>50%、fixpipe>15%、icache>15%、L2>80%或<50%、total conflict5/15%、bank/group/mte3%、resource5%、小DMA16KiB及重叠5/30%等是来源诊断阈值，不是充分必要判据。高Fixpipe不能唯一证明GM未512B对齐，高MTE2+低L2不唯一证明CacheMode错误，低BlockDim+高Duration不一定核数不足。Current<Rated是频率差证据，具体DVFS/温控/功耗原因需更多记录。

核间imbalance=(max-min)/max要求非空、max>0并选同类型可比核；负载、内存争用、启动差和等待都可造成差异。DB有效性要看真实生命周期与时间线，不能仅凭BUFFER_NUM>1就推断已重叠或配置错。Counter模式也不是任意Scalar分支的直接替代，AIC:AIV1:2启动含义需实际架构合同。

来源GroupedMatmul218.1→154.2→131.8→128.1us对应累计时延降低约41.3%，不是各阶段百分比相加，也不证明单项独立贡献。其Matmul4.75x、FlashAttention Fixpipe80→55%、MC2 32.7%及A2满核头开销20–21us均保留为来源案例，未复现且不普遍。八类数据的关联原则见[Profiler口径](profiler_metrics_and_bottleneck_hypotheses.md)。

## 原始字段名与上述含义的对应

以下保留固定 MindStudio 8.3.0/A2A3 来源中的字段拼写，用于按导出列名定位；它们不增加新的跨版本 schema 保证。`ai*` 是来源对 aic/aiv 的占位表示，实际 CSV 应使用对应核型列名，不把星号当真实列名。

| 所属表 | 原始字段 | 解释与单位边界 |
|---|---|---|
| ArithmeticUtilization | `aic_cube_fops`、`aiv_vec_fops` | Cube/Vector 浮点运算量，与指令条数分开 |
| ArithmeticUtilization | `aic_cube_total_instr_number`、`aic_cube_fp_instr_number`、`aic_cube_int_instr_number` | Cube 总指令、浮点指令、整数指令计数 |
| ArithmeticUtilization | `aic_cube_fp16_ratio`、`aic_cube_int8_ratio` | Cube 对应精度指令占比 |
| ArithmeticUtilization | `aiv_vec_fp32_ratio`、`aiv_vec_fp16_ratio`、`aiv_vec_int32_ratio`、`aiv_vec_int16_ratio`、`aiv_vec_misc_ratio` | Vector 对应指令类别占比，不等同 Cast 数量 |
| Memory | `read_main_memory_datas(KB)`、`write_main_memory_datas(KB)` | 主存读/写数据量，KB 换算遵循实际导出单位 |
| Memory | `GM_to_L1_datas(KB)`、`L1_to_GM_datas(KB)(estimate)` | GM→L1 与估算的 L1→GM 数据量，estimate 不可丢弃 |
| Memory | `L0C_to_L1_datas(KB)`、`L0C_to_GM_datas(KB)` | 两种 Fixpipe 目的路径的数据量，不能互换 |
| Memory | `GM_to_UB_datas(KB)`、`UB_to_GM_datas(KB)` | GM↔UB 两个方向的数据量 |
| MemoryL0 | `aic_l0a_read_bw(GB/s)`、`aic_l0a_write_bw(GB/s)`、`aic_l0b_read_bw(GB/s)`、`aic_l0b_write_bw(GB/s)` | L0A/L0B 读写带宽，分别保留方向 |
| MemoryL0 | `aic_l0c_read_bw_cube(GB/s)`、`aic_l0c_write_bw_cube(GB/s)` | Cube 对 L0C 的读写带宽 |
| MemoryUB | `aiv_ub_read_bw_vector(GB/s)`、`aiv_ub_write_bw_vector(GB/s)` | Vector 访问 UB 的读写带宽 |
| MemoryUB | `aiv_ub_read_bw_scalar(GB/s)`、`aiv_ub_write_bw_scalar(GB/s)` | Scalar 访问 UB 的读写带宽，不替代 Vector 计数 |
| ResourceConflictRatio | `aiv_vec_total_cflt_ratio`、`aiv_vec_bankgroup_cflt_ratio`、`aiv_vec_bank_cflt_ratio`、`aiv_vec_resc_cflt_ratio`、`aiv_vec_mte_cflt_ratio` | 总阻塞及 bank group、bank、执行资源、MTE 冲突分类 |

按字段名召回本卡后，仍需使用前文的任务身份、分母、缺列/零值和诊断边界；精确列名匹配不证明当前算子的根因。
