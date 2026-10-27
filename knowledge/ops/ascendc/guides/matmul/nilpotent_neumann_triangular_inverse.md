---
type: Guide
title: 幂零矩阵的有限 Neumann 三角逆
description: 保留乘积恒等式、层数与累加实现，纠正符号和低精度误差的无条件断言。
tags:
- task.design
- topic.compute.composite
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P103_cube_neumann_triangular_solve.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P103.yaml
  role: supporting
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# 幂零矩阵的有限 Neumann 三角逆

对于C×C严格下三角A，令X=-A，则X^C=0。解(I+A)x=b可先构造R=(I-X)^(-1)=sum_{j=0}^{C-1}X^j，再乘b。若L=ceil(log2(C))，乘积prod_{i=0}^{L-1}(I+X^(2^i))展开为sum_{j=0}^{2^L-1}X^j，因幂零等于R。上三角对应相同幂零推导，C非二次幂也可取覆盖到C的层数。C=0/1单独定义空矩阵/单位阵行为。

来源代码注释称输出(I+X)^(-1)，与定义X=-A相反；实际乘积对应(I-X)^(-1)。严格三角是有限截断前提，含非零对角的一般三角矩阵不能直接套用。对角可逆时可能先做适当归一化得到I加严格三角，但需另外处理对角缩放和条件数；不能只换一个mask保留同公式。

C=64需要幂X²、X⁴、X⁸、X¹⁶、X³²，共五次平方。若F0=I+X已备好，可用五次R←R+R*P完成其余五个因子；源额外F0*I载入L0C作为初始化，合计五平方加六次乘法共11次Mmad。这是该实现计数，并非代数必需恰11次；C=128再增加一层，并重新计算workspace与资源。

实现R+R*P时，旧R必须同时作为输入和原L0C累加基值，首轮初始化、后续更新及每轮重新加载的R版本一致。L0C长期保持FP32、输入R16为已舍入half时，是特定混合精度递推，并非逐步精确乘积；不能把“留在L0C”视为数值证明。平方结果经GM转换到half再加载是该通路选择，不是防止精度失控的普适必要条件；支持的更高精度或片上通路应按误差与布局选择。

单位矩阵可由Host准备固定slot，也可由另一个合法生产者生成，只要dtype、形状、初始化顺序和生命周期正确。每轮幂、R与输出的物理slot不应覆盖仍被消费的输入；队列/事件、Mmad→Fixpipe及Fixpipe→下轮读取需完整，单个PIPE_ALL名字不替代跨核协议。

有限幂零公式避免一般无限Neumann级数的收敛半径条件，但浮点误差仍受矩阵幅值、系统条件、层数及中间dtype影响，可能溢出或放大，不能承诺half误差约1.2e-4。求单RHS的前代约O(C²)，显式稠密逆乘积有更多算术；Cube并行、多RHS复用和实际硬件成本决定是否划算。

验证严格三角、非零对角反例、C=1/3/64/128、b的不同列数、多chunk状态与残差||(I+A)R-I||及||(I+A)x-b||。来源98→3微秒约32.67倍而非其字段30倍，误差/速度案例未在本批设备复现；数学恒等式与工程性能应分别评估。
