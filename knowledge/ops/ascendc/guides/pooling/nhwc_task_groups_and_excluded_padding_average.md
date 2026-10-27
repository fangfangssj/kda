---
type: Guide
title: NHWC 池化任务分组与排除 Padding 的平均值
description: 保存真实窗口采样和分组地址，修正一次加载与均匀分核的误述。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_process_example_for_lowering.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# NHWC 池化任务分组与排除 Padding 的平均值

来源为NHWC、方形kernel/stride/padding、floor输出的AvgPool2d。OH=floor((H+2P-K)/S)+1、OW同理只在参数合法域成立；unsigned直接H+2P-K会在负结果时回绕，stride0先除零，int64属性强转uint32前需校验。输出[B,OH,OW,C]不能沿用输入InferShape，非对称参数、ceil、dilation、includePad或override需另述合同。

每task对应(b,oh,owGroup)，G=ceil(OW/Wg)，totalTasks=B*OH*G；b=task//(OH*G)，oh=(task%(OH*G))//G，ow0=(task%G)*Wg。Host选Wg=max(1,tileW/strideW)，tileW18或max(6*strideW,kernelW)只是分组启发式，不是UB最优保证，kernel与Host的max处理要一致。

来源每次实际上只加载一个位置的C通道，不是一次加载tileW*C并共享所有窗口；因此windows_per_load更准确地说是每task输出窗口数。若真正批量加载Wg个相邻输出窗口，非dilated连续输入跨度为(Wg-1)*strideW+kernelW，不能总用tileW；Wg18/S1/K3需要20输入位置。以融合载入优化时需补左右边界和halo，并保持窗口计数。

连续任务分核tpc=ceil(totalTasks/nUsed)，第p核[p*tpc,min((p+1)*tpc,totalTasks))能覆盖全部，但负载差不保证≤1，也可能有空核。例如10任务6核得到2,2,2,2,2,0。需要更均匀时可用商余pivot或其他可证明完整调度；totalTasks0须先处理。task数和所有地址乘积用足够宽整数。

有效采样ih=oh*SH-PH+kh、iw=ow*SW-PW+kw；仅合法点累加sum并使cnt加1，GM元素基址=((b*H+ih)*W+iw)*C。来源signed ih/iw表达式混有unsigned操作，先在宽有符号域计算再界限检查可避免回绕转换依赖。对同窗口各channel几何计数相同，可用每窗口标量count再广播，前提是没有逐channel mask。

sum/cnt实现count_include_pad=false；边界不能改为固定K*K，includePad另有除数。cnt0在源代码Div整组时产生0/0，未使用的尾窗口虽不写回也不代表有效空窗口已定义。只对有效窗口求值或显式约定空窗口结果，不能静默假定0。浮点count大于精确整数域也需考虑，以整数几何计数可避免累加count舍入。

该Init输入BF16+FP32 staging为6C字节，输出FP32/BF16加sum/cnt为14*Wg*C，总有效预算6C+14WgC再计padding。按w*C切FP32段需要相应对齐；普通BF16DataCopy搬C元素通常要求对应32B粒度，C非16倍数可用合适Pad/物理pitch实现，不是算子数学拒绝条件。BF16升FP32后求和，降BF16用目标支持的RINT，并保留中间Vector依赖和队列生命周期。
