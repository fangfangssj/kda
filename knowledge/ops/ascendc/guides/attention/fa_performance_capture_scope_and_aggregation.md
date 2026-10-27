---
type: Guide
title: FA 性能采集范围与汇总口径
description: 区分全 forward、Event 与 device task，按已声明完整段选择同源整版；防止有利窗拼接、汇总指标替换及归档指针漂移。
aliases:
- whole-version selection
- complete episodes
- favorable-window mix
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources: []
created_at: '2026-09-11T05:21:50Z'
updated_at: "2026-10-15T00:00:00Z"
platforms: []
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# FA 性能采集范围与汇总口径

性能比较需同公式、输入、dtype、layout、设备/争用窗口和计时范围。device kernel执行、同stream事件区间及包含Host调度/布局转换的端到端时长是不同指标。Host初始化销毁和executor重建可显著扩大后两者，不能据纯device结果解释为Host框架开销带来的加速。

来源torch进程ctypes launch逐次event取min出现约1.6–1.7us恒值和391倍假加速，是事件未覆盖实际外来stream/操作等问题的线索。可用实际launch stream上的ACL事件组回放并与profiler核对；不能推广为全部torch event永远无法量外来kernel。分组3*10取min/均值/median都要标明聚合顺序，某一最小值尤其易放大漏测。

ATB executor重建例event4061us/device76us说明区间可能含Host造成的GPU/NPU空闲，选择profiler纯device可隔离该部分；端到端吞吐仍可合法用同步包围的多次回放，不能称墙钟一律无效。NNAL ATB环境脚本和torch_npu导出解析触发属于来源安装/版本要求，不是所有安装路径固定或仅export_chrome_trace一种导出。

profiler过滤只留AI_VECTOR_CORE/AI_CORE会漏MIX_AIC融合kernel；类型字段及名称按导出版本核验。磁盘满、trace截断、进程OOM和解析未完成都可造成异常短时长，不能只凭15us对72–78us差异认定工具盲区。独立进程逐case可降低累积内存影响，也会改变缓存/初始化条件。

多kernel链逐项device duration之和忽略Host间隙，但若核间/stream间重叠，它也不是必然的端到端公平下界；需区分工作量总和与时间线span。分解链与融合基线应显式命名，source33.5倍墙钟对1.16倍device不能当同一加速指标。

gate可用exp(mean(log(tRef/tCustom)))，要求各时长正且有限、逐case配对及权重明示；失败/缺测/0/0不能算通过或静默排除。算术均值对大ratio敏感，几何均值也不替代逐case回退/尾部风险。来源1.025、3.047、0.107–0.364等不同工程历史值不能作新实现门槛。

争用时将memset和kernel分事件区间可定位排队成本，但队列等待不保证只发生在首操作；若算法每launch确实需要清workspace，端到端仍应计入。kfc只清64KiB取决于实际系统区合同。两侧同窗口交替、多轮、同配置可降低环境偏差，不能保证完全相同争用。torch.cuda.Event并非NPU事件替代；同步墙钟均摊仍含Host部分，应与device口径分列。

## 多段完整测量下的整版选择与归档

若验收合同已经声明多个完整 episode 和保守选优策略，先在每个 source/wrapper/DSO 版本自己的原分母上独立完成全部精度、负控及有效性能段，再计算该版的合同分数。例：合同指定每 case 取两腿重复值的各自中位数、case 等权几何均值，并以所有完整段的最小值选优，则 `Q[v,e]=exp(mean_i(log(median(tRef[v,e,i])/median(tCustom[v,e,i]))))`，`score[v]=min_e Q[v,e]`。不能将所有 pair ratio 的 GM、两腿中位数之和的 ratio 或只挑第二段替代这个已声明指标。两段/min 是此合同选择，并非适用于全部任务的统计定律。

应保留一整版，而不是逐 case/逐段拼最优；只在所有必需精度与测量谓词合格的整版之间选优。失败版的较快时长为诊断，不进入选优。缺例、窗口无效或被中断的一段保持缺测，不能与另一窗口余例拼成完整段。即使全部版均低于性能目标，也可按合同保留当前最好的合格整版，同时明确尚未达标。

报告、可部署源码/构建/DSO 包、完整原始 archive 和 source/wrapper/consumer 身份应同指选定整版；旧合格包和失败历史分别保留，不能用新摘要覆盖旧包。归档完整性、独立审读/发布状态、数值合格和性能达标仍是不同结论。

已保存观察的脱敏重放包含两版各49例×2段×5交替 Event pairs，共980 pairs；三次同量 warmup/leg，全部 forward helpers 在 Event 内，profile 另采。版6 Q A/B=1.054184212451175/1.053647720406219，版7=1.0333457141572764/1.0614512602026265，均通过该链声明的全部精度和零输出控制。只取较快 B 会选版7，而已声明 min 选版6。原两段是在同一进程的不同完整循环，不是独立进程重复。固定来源中的 `replay.py` 可离线复算；它不是新冷生成或 NPU 复测，不证明知识检索减少迭代或带来任何倍数的性能收益。

相关方法：[性能归因与内存证据](fa_optimization_attribution_and_memory_evidence.md)用于区分测量结果与瓶颈因果；本卡只定义已经声明的计时、汇总与整版选择范围。
