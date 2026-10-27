---
type: Guide
title: 深度池化 Direct Buffer 路径与同步修订
description: 保存行地址和复用条件，消除双重屏障必需的矛盾。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/ub-management.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/pooling-anti-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/reduce-d-fastpath-implementation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/tilelang-translation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/PoolingPatternIndex.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/pipe_barrier_isasi.md
  role: constraint
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# 深度池化 Direct Buffer 路径与同步修订

KH=KW=1时可针对每输出(n,od,oh)取ih=oh*SH-PH，遍历有效id=od*SD-PD+kd，搬入连续NDHWC的W*C行，再对每ow用iw=ow*SW-PW选择C段累加。元素行基址为((n*D+id)*H+ih)*W*C；SH/SW和padding仍生效，不能因KH/KW为1把整个W不经索引直接相加。

该路径可预分配TBuf避免每kd重复queue管理，但仍需输入原始行、FP32 cast、OW*C accumulator及输出/scratch的完整UB预算。源uint32(W*C)*sizeof及InitBuffer(W*C)会随大W/C增长，需宽整数检查与分片；不能声称direct TBuf自动解决所有大shape容量。TBuf<>真实默认位置需看模板声明，不是“无任何物理约束”。

无效ih/id/iw应不读输入，但仍须完成输出初始化和按合同归一化，goto normalize只是结构选择，也可用等价分支。源码Normalization省略，因此不是独立可运行完整Kernel。AvgPool空贡献的0与MaxPool中性值不同，不能复用sum骨架表示MaxPool。

同一kd内若不同ow操作互不重叠、只读相应输入C段，可把必要V管线屏障放在一组操作之后，减少每ow屏障；跨kd对同一acc的依赖及下一MTE2覆盖原buffer前的V读取完成仍须建立。同一PIPE_V barrier本身不普遍约束MTE2，需相应V→MTE2事件或完整pipe等待；MTE2写后V读同理需要MTE2→V。

来源ub-management/anti-patterns要求PIPE_MTE2加PIPE_ALL双屏障，fastpath末尾修订又明确单PIPE_ALL即可等待先前各pipe接口。已有KB API也描述PIPE_ALL等待本核全部先前流水接口，因此可将双屏障说法限定为多余保守写法，而非正确性必需；精确HardEvent可减少全pipe等待。PIPE_ALL不能推导为其他AIC/AIV核或整个设备barrier，source称含其他AIC的全局完成过宽。

TQue可管理这些生命周期，窗口小也可正确使用；TBuf性能取决于实际同步而非天然比queue快。来源BUFFER_NUM=1只能无参DeQue把buffer数与模板depth混淆，引用式/返回值形式应按真实static_assert，depth0也不是运行时队列为空。每个句柄从分配到最后消费者再释放，不能跨buffer Free。

KH/KW1、KD≤5、W32/C32是来源候选，c2的222.03→145.78µs、每ow屏障209.68→145.78µs及FP16 170.06µs缺本批原始profile，保持历史案例。固定24核应替换为与真实AIV launch模型一致的可用资源和任务分配，空输出不能产生除零或非法0核启动。

来源TileLang0.1.4 AUTO_SYNC约3.4µs/op、PipelinePlanner非SeqStmt、3D Parallel限制及0.007–0.010倍记录只适用于其后端/代码/环境，不能证明所有Pooling在TileLang永远不可达目标；2维示意也不足以证明3维编译问题。可提炼为细粒度操作的同步成本需要测量，向Ascend C转换时按真实RAW/WAR/WAW依赖选择事件，不能机械将所有AUTO_SYNC换单PipeBarrier。流程轮数、后台跑法和报告审批不迁入。
