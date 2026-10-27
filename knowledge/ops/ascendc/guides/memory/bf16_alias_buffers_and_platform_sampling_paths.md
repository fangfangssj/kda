---
type: Guide
title: BF16 别名缓冲与平台采样路径
description: 保存半区Cast布局和310P采样，补齐覆盖、Mask及原子条件。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_05_bf16_special_handling.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# BF16 别名缓冲与平台采样路径

ClippedSwiGLU的16位数据放buffer后半、FP32从前部写，是容量与别名设计候选；xQueSpace/2/sizeof(bfloat16_t)为16位元素offset，转成FP32视图后单位改变。先按byte区间计算源/目标实际长度与重叠，再核对Cast是否允许该方向重叠；地址错开不等于Cast具有原子性，也不自动保证分块执行不覆盖尚未读源。

Modulate把ubLength减半并从xLocal[ubLength]扩为float，需要证明handleL*alignedD的FP32目标与BF16源区均在总allocation内，以及scale/shift/output各自布局。ReinterpretCast只生成视图，容量减半不代表原先任何tile都仍容纳。该例x*scale+shift与baseline的x*(scale+1)+shift语义不同，不能据BF16路径名判前者正确。

InplaceAddRMS的FP16片段先在half做Add再Cast，BF16片段先升FP32 Add再舍回BF16；二者都可能在RMS前有低精度残差舍入点，不能概括为所有计算始终FP32。若规范要求保存低精度residual并基于它归一化，去掉舍入点会改变结果；若规范要求未舍入和，则应继续使用FP32和。别名输出不可提前覆盖后续仍需要的输入。

Pooling overlap的FP32 workspace须先清零并确保所有更新之后再ProcessCast；普通GM GetValue+SetValue并非atomic，需唯一写者、真实原子或分阶段归并。升精不代替同步，“双精度累加”在该例实际只是FP32。

310P可变形注意力来源专用Kernel用CAL_H_W_BLOCK512、CHANNEL_BLOCK32、mask scratch及NHWC→NCHW Transpose，这些是版本分块候选而非设备普适最优。TransposeParams各stride、临时区与源别名需真实重载允许，tail通道/有效坐标和输出layout不能只按常量整块访问。

CoordinatesFrameRange/ClipCoordinates以Mins上界upBound再Maxs0，正文却说[0,upBound-1]；须明确调用者传的是size还是size-1，空维度不能减1。范围mask应由原采样坐标生成，clip后的合法地址只用于安全加载，不应把越界贡献变成边缘值。X/Y掩码应完整组合，数据移动安全与数学零填充分别验证。

SetAtomicAdd<float>→GM写回→SetAtomicNone定义一段原子写模式，仍需输出零初始化完成、正确参与者和边界，未证明的原子支持不能从平台名字推出。无baseline的专用路径只提供实现设计，不具备自动性能对照证据。
