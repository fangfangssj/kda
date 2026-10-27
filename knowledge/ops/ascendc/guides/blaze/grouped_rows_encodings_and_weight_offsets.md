---
type: Guide
title: Grouped Matmul 的行编码与逐组权重地址
description: 补齐 cumsum/count、空组、完整 Golden 写回及独立 group/batch 物理映射。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/group-matmul-delta.md
  role: primary
created_at: '2026-09-11T03:07:47Z'
updated_at: '2026-09-11T03:07:47Z'
platforms: []
---

# Grouped Matmul 的行编码与逐组权重地址

来源GMM以group i的半开行区间[start_i,end_i)读取共同A的不同行，使用自己的B_i，写回C对应行。其groupListType=0表示累计端点：start取前一端点、end取当前值、totalM取最后端点；1表示count：start是前缀和、end=start+count、totalM为总和。这是参考ABI枚举，不自动适用所有Grouped算子。

正维执行需group数和数组长度一致，cumsum非负非递减且末值=M，count非负且求和=M，前缀和/地址算术不溢出。重复端点或零count表示空组，跳过计算但保留后续prefix与专家索引；不能静默排序错误端点，也不能对cumsum再次求和。

M=0/E=0仅在算子合同允许时走单独空路径，不解引用不存在的group buffer；仍要校验该空路径规定的metadata、shape和输出。合法空组不意味着任意E=0都能进入正常tiler。全零数值行仍是实际工作，不能按空Tensor跳过。

连续ND基线中A组基址为A+start_i*lda，B组基址为B+i*weightGroupStride，C组基址为C+start_i*ldc，偏移按对应Tensor元素解释；字节接口只转换一次。转置、NZ、独立Tensor列表、低比特packed或paired权重各自推导物理地址，group编码本身不定义stride。metadata改变后必须同步Copy真正消费的pitch。

Golden要覆盖完整M行，可先分配完整mm[M,N]并逐非空组填对应slice，再做后处理；也可对行独立后处理逐组写最终对应slice。不能循环覆盖一个局部mm后在循环外只量化最后一组。GLU用每组两路权重并写对应M区间，未参与数学合同的x_scale/weight_scale不能因旧签名存在而加入。

测试首/中/末及连续空组、不均匀group、专家各不相同、前缀非零、多M tile与非法编码拒绝；分别比较每组C及融合结果，避免所有组误用同一B却被常量数据掩盖。来源设备507015是具体故障，不能作为逻辑K误写baseK的唯一识别信号。
