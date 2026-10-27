---
type: Guide
title: Fixpipe 直达 UB 的镜像布局与复用槽合同
description: 区分完整输出镜像和紧凑复用槽的地址、容量、路由及通知协议，约束 Cube/Vector 融合中的 GM 往返消除。
tags:
- task.optimization
- topic.data_movement.data_copy
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-aic.md
  role: primary
created_at: '2026-09-14T04:19:37Z'
platforms: []
updated_at: '2026-09-14T04:19:37Z'
---

# Fixpipe 直达 UB 的镜像布局与复用槽合同

当同一混合 Kernel 的 AIC 将 matmul 中间结果写入 GM，配对 AIV 又将同一有效区域读回做后处理时，目标硬件和 SDK 若支持 L0C→UB Fixpipe 路径，可消除这份中间结果的一次 GM 写和一次 GM 读。AIV 仍可能读取其他 GM 输入，最终输出仍需按接口发布；MTE2 比例下降不能独立证明正确路由或全部往返消失。来源将该路径限定在 dav-3510/Ascend 950 的示例环境，不能推广到所有 SoC。

完整输出镜像保留输出行距。以实际接收的 M×N 元素区域为例，宽方向第 j 个 tile 的起点是 j*tileN，第 r 行起点是 r*N+j*tileN；接收 UB 必须为这个物理跨度分配空间，Fixpipe 目的 stride 与 AIV 读取时的行距一致。只把 dstStride 改成 tileN，却继续按 N 寻址，会把行写到错误位置。尾 tile 的宽度、GM 有效界限和 UB 物理容量需分别处理，不能用 floor(N/tileN) 次循环遗漏最后一片。

来源还配置独立 scratch，把镜像中的 strided tile 搬成紧凑 UB 数据后计算。容量账本包括镜像、scratch、槽数和其他同时存活的缓冲；每个 AIV 应分别核算其真实接收区域，不能把两个 AIV 的 UB 当成单块共享容量。来源用两份 cSize 分配是其示例设计，不是所有融合算子的最小空间。

紧凑 tile 槽改用接收 tile 的物理行距，只有地址公式、Fixpipe stride、AIV 读取和槽编号同时改变才正确。复用槽必须建立生产完成后消费者可读、消费完成后生产者可覆盖的双向依赖；双缓冲也需要每个槽独立的生命周期。完整镜像若各 tile 写入区域确实不重叠，在单次写入期间可以不需要用于“释放该数据槽”的反向确认，但同一事件标识的反复 set/wait 仍需符合其消费和重用合同。数据区域不复用，不代表通知资源可以无限重复发布而不等待。

来源的单目标示例通过 dualDstCtl=0 和 subBlockId 选择某一个 AIV 接收；SPLIT_M 示例通过另一模式把同一 tile 的行分给两个 AIV。两者具有不同的工作归属、接收尺寸、偏移和通知次数，不能只改一个开关。模式值、行数限制、Flag 路由偏移和支持的重载应以实际 SDK 声明为准；来源中的 +16 不是跨版本的通用事件算术。

通知必须在 Fixpipe 数据可见后发出，AIV 第一个实际消费者执行前完成等待。来源示例在 PIPE_S 上门控后续派发；这说明该示例的等待位置，不能证明其他合法同步方案一概不可用。增加反向确认本身不会必然死锁；死锁需检查初始化、路由、生产消费计数和依赖环。scratch 经计算并被 MTE3 消费后也须等待完成才能重用，不能只关注镜像的安全性。

验证应覆盖不同路由、多个 tile、N 尾部、每个槽重复使用以及最终输出，并核对对应路径的真实搬运量、地址、事件和总时延。低 MTE2 活动与非零 CUBE/VEC 重叠只是辅助信号。示例没有在本轮 SDK/NPU 上编译或运行；其配置和同步片段作为版本线索，不作为可直接替换的已验证模板。工作归属还可参阅[混合改写的资源所有权](mix_rewrite_work_mapping_and_resource_ownership.md)。
