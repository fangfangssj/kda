---
type: Guide
title: CoC Matmul Tile、累加器与 Workspace 交接
description: 补全计算分块、SplitK、Bias/Fixpipe 及通信计算槽位轮换的前提。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-optim/references/compute-optimization.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/fused-compute-phase-gates.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/design-template.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/mmad_compute/mmad.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate.md
  role: constraint
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# CoC Matmul Tile、累加器与 Workspace 交接

CoC 来源用AIC CATLASS BlockMmad和AIV通信Epilogue组合计算与通信。CATLASS计算策略不代替SHMEM symmetric buffer、远端完成和跨PE调度合同。Host aclnnMatmul启动设备计算再接独立通信是另一执行结构，三重标量循环也与Cube路径性能不同；L0/L1/L2/L3是来源交付分级，不是Ascend C禁止合法标量或向量算法的能力声明。

常规MN分块数B=ceil(M/TM)*ceil(N/TN)，理想同成本轮数约ceil(B/C)。目标通常是提供足够并行且不过量资源/搬运，不能要求B必等于C或整除于C：多轮有时更利于负载均衡与复用，tail成本也不相同。来源1024×576由128×256得24块、改256×128得20块；20×6144由128×256得24、改32×320得20；1×768由128×128得6、改16×32得24。块数成立不证明1.49/1.30/2.34倍收益，本批未复现实测。

TileShape对齐与L0/L1尺寸关系由实际dtype、指令及CATLASS特化决定，不是各轴永远16倍数，也不是L0.M/N必须恒等L1且K恒为1/4。Atlas A2不固定20 AIC。Pingpong、Preload及ShuffleK的具体支持从所用源码确认；仅策略名不能证明一定减bank冲突。GemmIdentityBlockSwizzle的direction/offset影响块遍历与复用，M>N选M方向等只是候选启发式，源40.6→35.3us是特定case记录。

SplitK用多个核计算同一输出的不同K区间，再合并partial；需要额外workspace、归约同步与累加精度。MN块少且K大是候选情形，不是仅这种情形可能有收益。并行度收益与归约/存储成本一起评估，首贡献初始化、最后合并、tail-K与每PE输出ownership须闭合。

将多次矩阵乘结果留L0C累加可减少GM往返，但需保持同一输出tile及累加器生命周期。来源用“enAtomic=1”描述L0C累加没有给出真实重载，不能据此与GM原子写混用。固定KB Iterate说明enPartialSum对应已有L0C累加，且该高阶接口要求singleCoreM==baseM与singleCoreN==baseN；手动Mmad有自身初始化/累加合同，不能跨API借同名参数。

BT/C2存Bias可用于支持的Mmad初始化，不能说Bias是在Mmad CopyOut时无条件自动融合。FP/量化参数可在支持的Fixpipe输出转换中消除部分GM→UB处理，但需按真实scale粒度、舍入、饱和、类型与平台能力核对。L1驻留小矩阵需要与其他同时存活数据共同预算，≤L1一半只是来源启发式。

Iterate<false>是高阶Matmul异步协议选择，不能当成CATLASS DispatchPolicy的另一名称。它需要配套workspace、结果获取和完成方法；来源“只首轮发消息”是某MIX实现的性能解释，不能推出消费者可在任意轮提前读取或所有模式可用。固定KB还限定MixDualMaster不支持该Iterate接口。

compute/comm共享stage=idx%STAGES时，每槽需先前消费者释放、当前生产完成、当前消费者访问完成三种状态。来源只给comm循环：等flagFinishCompute后写workspace并发布flagFinishComm；还缺compute侧对应等待/释放、初始token、末轮排空及远端读者释放。STAGES=2本身不能防止覆盖。

Matmul→CopyOut→Comm、CopyOut直接落symmetric后通知、Comm→本地归约→Matmul具有不同数据依赖；减少GM中转不能跳过真实空间可达与完成关系。实现常量、历史收益与固定五轮优化流程均不能替代正确性和同条件性能证据。
