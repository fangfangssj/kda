---
type: Guide
title: DumpTensor 的阶段身份与 Attention 参考语义
description: 保存生产消费观测法，修正同步探针、尾块和局部 softmax 参考的偏差。
tags:
- task.debugging
- topic.implementation.debugging
- topic.attention
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-precision-tuning/references/api-reference.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-precision-tuning/references/debug-workflow-complex.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-precision-tuning/references/debug-workflow-simple.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-precision-tuning/references/error-patterns.md
  role: supporting
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# DumpTensor 的阶段身份与 Attention 参考语义

DumpTensor的desc是uint32标签，dumpSize是元素数；LocalTensor与GlobalTensor重载可观察不同存储，具体平台/存储支持见[API卡](../../apis/simd_api/basic_api/debug_interface/onboard_print/dump_tensor.md)。ND GM/Workspace容易同CPU逻辑数组对照，但它的值不直接证明NZ/L0内部加载正确。只取前32个元素也要确保有效长度足够且不会把padding当输出。

简单Vector可分别观察CopyIn后、计算中间、CopyOut前；Cube+Vector链可观察Q/K/V→MM1的S→Vector读S→P/状态→MM2的O_tmp→最终归一化输出。生产侧正确而消费侧错误提示地址、slot、完成与可见性问题，生产侧已错则回查输入、layout、MMAD、scale，但这些是候选解释，不是唯一根因。固定常数误差、周期错、NaN/Inf、随机散错、全零、等于输入均不能直接判定某单一原因；增加BUFFER_NUM不会自动修复竞态。

探针需要标识launch、核、batch、head、query块、KV迭代、物理slot、阶段和有效span，desc=300+slot不能区分slot轮转后的版本，也可能与310+slot重叠。首块正确不能代表尾块、后续状态、GQA映射或全部核；应对准失败元素的逻辑索引与实际地址，而非仅对照相同标签。

Fixpipe/DataCopy调用后的源码位置不证明DMA已完成；DumpTensor自身访问也需正确pipeline依赖。SetFlag必须在生产完成路径发布，WaitFlag只解决对应跨核事件，MTE2→V、V→MTE3、MTE2→MTE3及reuse仍需各自协议。插入dump可能改变时序或隐含同步，不能把插桩通过当原始Kernel正确；输出到屏幕还是被运行器收集/离线解析由环境决定，“不会生成日志”不是普遍API合同。

Attention尾KV的无效score通常按合同mask为-Inf，再处理全mask行、有效行数和跨行广播；BLOCK_N长度mask不能未经布局证明覆盖整个BLOCK_M×BLOCK_N tile。dimAlign/actualDim不同时，来源把dealRows*actualDim连续搬出不能自动跳过每行padding，需要逐行或二维stride搬运。输出正确但GM错误可由这一压缩布局问题引起。

来源CPU示例q_len//BLOCK_M漏掉Q尾，kv_len//BLOCK_N+1在整除时多出空块；应使用ceildiv及有效切片。逐块torch.softmax(S)不等于online softmax中间P：P通常相对跨块running max且尚未做最终全局归一化，需要同样的m/l/输出重缩放状态。不同Q/KV head数量需明确GQA映射，不能用同一by索引泛化。参考[状态递推](../attention/online_softmax_state_and_partition_merge.md)和[槽版本](../attention/online_state_slots_and_final_selection.md)。

CPU对照需要相同输入字节、shape、dtype、转置/scale、mask、运算次序与观测阶段；公式级高精度参考与量化/舍入仿真承担不同验证职责。运行器命令、固定desc编号和用例文件管理属于Skill流程。
