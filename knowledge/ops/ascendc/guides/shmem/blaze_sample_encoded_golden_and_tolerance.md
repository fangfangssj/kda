---
type: Guide
title: Blaze-SHMEM 编码 Golden 与精度判定
description: Blaze-SHMEM 编码 Golden 与精度判定的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/gen_data.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/verify_result.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/run.sh
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 编码 Golden 与精度判定

gen_data seed42，A各rank独立FP8 E4M3、B所有rank复制且物理[N,K]，scale e8m0来自编码后的随机值；先每32元素重复scale反量化，再按target M段交换、按source拼K，CPU float32 GEMM最终转BF16。验证器将文件uint16位视图解释BF16；Host hostOutput虽声明half，仅回拷/原样写字节不做half算术，在2字节存储合同下不改变BF16编码，不能单凭Host类型名判断结果为FP16。

verifier逐rank对独立golden判断，不要求各rank结果相同，打印的“All ranks same”不符数学合同。torch.allclose实参为(golden,npu)，相对阈值按第二个npu幅值计算，含atol=rtol=1e-2且equal_nan=True；交换实参会改变非对称阈值。大矩阵summary的count(absErr>0.01)不是实际allclose失败数，不能把它当越阈值统计。

1%只是来源验收设置，不是FP8输入天然引入1% kernel误差的定律，因为golden已用同一编码输入；应依据输出舍入和累加顺序及任务合同选阈值。equal_nan可能让双方非法NaN同时通过，正常有限输入域宜另外检查有限性。缺文件/错长度按rank记录失败，rankNum0时循环空可能all_pass=True，独立调用脚本须验证正rank/shape。

默认正数均匀1..8缺少负值、抵消、零、极值及编码group边界，不能覆盖所有E4M3/E5M2组合；main只调用E4M3E4M3，其他三个入口存在不等于有造数/精度证据。增加热点source、独立rank/列/group尺度、Ka96等布局分歧和多轮复用探针；记录输入哈希与实际artifact身份，不把源码中的历史288%错误或“已fixed”注释当本次实测。
