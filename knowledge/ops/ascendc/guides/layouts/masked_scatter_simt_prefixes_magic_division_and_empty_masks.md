---
type: Guide
title: MaskedScatter SIMT 前缀、快速除法与空 Mask
description: 保存线程分发，补齐前缀基数和未修改输出语义。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_13_special_algorithms.md
  role: primary
created_at: '2026-09-11T06:02:18Z'
updated_at: '2026-09-11T06:02:18Z'
platforms: []
---

# MaskedScatter SIMT 前缀、快速除法与空 Mask

来源SIMT grid-stride循环i=blockIdx*threadNum+threadIdx，步长blockNum*threadNum，适合按mask选择稀疏updates读取。LAUNCH_BOUND(THREAD_LAUNCH)是编译/调用约束，不能仅从标注推定实际启动1024个线程；真实调用线程数、每核范围与大整数乘积应一致，步长必须正且i递增不回绕。

AB折叠row=i/xInner、col=i-row*xInner，mask[row]为真时输出来自对应updates位置。若position为每row inclusive true计数，则该row的首update序号应为(position[row]-1)*xInner，再加col；BA为inner mask重复到各outer时，还需前面outer的总true数与inner前缀。源updates[prefixSum]未给公式，不能直接让整行使用同一update或漏减1。掩码false时必须保持原x；若输出独立buffer，则还需copy原值，只有正确inplace/已初始化输出才可跳过写。

GetUintDivMagicAndShift与Simt::UintDiv必须同一位宽/无符号域，xInner正且在helper支持范围内，magic/shift及乘高位/修正由真实实现决定。不能把所有常数除法简化为普通低位mul+shr，也不能把有符号负索引交给UintDiv。验证商q与余r满足i=q*d+r、0≤r<d，含d1、非2次幂、最大i；10–20倍固定收益未实测。

position最后值0仅在前缀可信、非负单调、覆盖全部mask且定义一致时说明无true。AB读取xOuter-1、BA读取xInner-1之前须检查长度非零；空Tensor不等于非空全false。early return还需已验证updates/shape/dtype合同，且不能让其他核等待本核参与的barrier。

全false可以省SIMT更新，但不免除独立输出复制、状态或其他可观察输出；读position的GM完成/一致性及Scalar访问协议须满足。越界position、重复运行和同Tensor多核片段是必要边界，不能用一个全false性能点宣称所有稀疏场景数量级收益。
