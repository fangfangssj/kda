---
type: Guide
title: Dequant-GLU 的列映射与 Scale 合同
description: 保留 mode 列置换、累加类型与量化收尾，限制软件路径唯一性。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dequant_kernel_patterns.md
  role: primary
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# Dequant-GLU 的列映射与 Scale 合同

来源dequant链为INT32输入转FP32、乘逐列weight_scale与逐行activation_scale，再激活/门控、乘smooth quant_scale，最后按每行amax量化INT8并输出scale。INT32超过FP32精确整数范围会舍入；乘法次序、zero-point/bias、gate形式及量化规则需由实际算子定义。不能仅因名字dequant就假定这组输入和顺序。

standard mode将两半列作为gate/linear，另一mode把交织列view(M,H,2)→permute(0,2,1)→contiguous→view(M,N)，逐列weight_scale必须做相同置换，否则第j列乘错scale。N必须能按2拆分且真实stride适配；quant_scale对应激活后H列，不能盲目复用原N列置换。该mode的alpha/clamp等也要按公式，不能用同一silu表达所有变体。

quant_scale可能FP16/BF16/FP32，Host wrapper转连续FP32可统一Kernel类型，但增加设备转换/分配；也可合法dtype分发。主文禁止Host预处理与此文允许.to属于Skill策略差异，应以实际任务授权和端到端成本判断，不能把Python调用认成CPU算术或把reinterpret当dtype转换。

来源amax=0时q=0且scale=0，与动态量化另一文scale下界1e-10不同，需按本算子合同保存。scale计算发生在每行数据产生之后，文中“host-side scalar”若指Device Scalar不能搬到真正Host而漏D2H/同步成本。分片N必须用整行amax；scale reciprocal与Div可有舍入差，先FP32入整再窄化和先half再入整也不等价。来源FP32→half RINT、half→int8 RINT须核对真实重载支持，不能直接沿用。

每核收集scale再一次DataCopyPad可减少逐行Scalar写，适合该核负责连续输出区域；round-robin行时不能把离散scale当连续直接写。GlobalTensor::SetValue与DMA均可在满足各自可见性/完成合同下实现，不存在“永远不合法”和“总是快”的统一结论。普通DataCopy有效INT8长仍需对齐，非整尾用适用路径。

来源PR139、mode1 vendor错误和50个benchmark通过只是该reference历史叙述；未给可核验实现/版本时，不应据此宣布某oracle有bug或软件FP32是唯一正确路径。硬件API、改进近似或其他数学实现均可能满足合同，精度必须独立测量；改写reference/阈值以适配候选不能建立原合同正确性。
