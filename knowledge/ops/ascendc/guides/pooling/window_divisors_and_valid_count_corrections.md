---
type: Guide
title: 池化窗口除数与有效元素计数修正
description: 保存 ceil/padding/override 优先级及前后向一致性。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/precision-patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/pooling-anti-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/PoolingPatternIndex.md
  role: supporting
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# 池化窗口除数与有效元素计数修正

AvgPool前向对每输出窗口的有效输入求和，再除该窗口合同规定的divisor；反向把该输出梯度按同一divisor分给有效输入。MaxPool依赖最大值/argmax而不是均匀求和，AdaptivePool窗口由输入输出比例决定，不能直接套固定KD/KH/KW。索引里所有pooling“必须FP32累加”只适用于相应浮点sum路径，整数Max或保留原值比较需按类型合同。

对于正kernel/stride、非dilated固定窗口及padding区间，原始起点a=o*s-pLeft、原始终点b=a+k；有效个数为max(0,min(size,b)-max(0,a))。来源先把a clamp到0再令b=a+k会多算左padding，例如a=-1,k=3应保留[0,2)两项，错误算成[0,3)三项。3D除数为各轴有效计数乘积，需宽整数防溢出。

includePad时计入约定padding区间[-pLeft,size+pRight)，每轴count=max(0,min(size+pRight,b)-max(-pLeft,a))；来源对正常输出a≥-pLeft时可省左侧max。floor模式若输出shape按正常固定窗口规则生成且窗口完全在padding区，可用KD*KH*KW；ceil模式末端可能截断，需每位置padded count，不能无条件用常数。dilation或其他padding语义需按实际采样点数重新计算。

显式divisor_override按接口优先使用，但来源>0判断隐含只接受正值；未提供值、0及负值的合法性应由真实API校验，不能把非法值静默当未指定。有效count0时是输出0、NaN还是拒绝也需合同，来源乘0是一种选择，并非数学定义。用乘倒数替代除法可能改变舍入。

来源ceil=True、includePad=True在某边界应除18却用27，得到正确值的2/3；“偏小1.5倍”应明确比值方向。标准用例全部ceil=False会漏此分支，需覆盖ceil与includePad四组合、override、非对称kernel/stride/padding和空边界，前后向共享同一窗口定义。

FP16/BF16升FP32做sum、末尾转目标dtype有助于累加精度，Cast模式仍需目标重载。PrepareInput对FP32 ReinterpretCast只是同址视图，必须保持源存活；FinalizeOutput对FP32分支无操作要求调用前out已正确别名/绑定，不能用未初始化out写回。
