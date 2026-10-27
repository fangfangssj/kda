---
type: Guide
title: Blaze 性能采样的 Rank 身份与聚合
description: Blaze 性能采样的 Rank 身份与聚合的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/extract_perf.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 性能采样的 Rank 身份与聚合

extract_perf遍历排序PROF_*目录，以枚举编号当rank，仅取每目录glob第一个op_summary CSV，按Op Name精确过滤目标kernel，取文件最后N条Task Duration算均值，再对rank均值取max。这是脚本实际统计定义，目录顺序不等于真实rank/设备身份；多个CSV可能被漏，文件行序也未必时间顺序。

应从采集元数据匹配rank/device/PID、kernel实例与轮次，检查预期rank数及每rank有效记录，排除heavy和不同配置混合。last_n必须正，0会除0、负数可产生无意义负均值；duration须有限非负且单位us，当前float解析不拒绝NaN/Inf。只有1个PROF目录也会给overall，不能证明4卡采样完整。

max_r mean_i(t_ri)不等于mean_i max_r(t_ri)，前者是最慢rank的平均，后者是逐轮整体最慢耗时平均；rank身份/迭代配对缺失时无法推算后者。目标kernel Task Duration不含全部Host/bootstrap/flush成本，是否等于通信算子整体耗时取决于kernel内部wait与跨卡起止定义。无外部采集时run.sh perf退出0没有产生可信性能报告。

用相同shape/编码/设备/版本/冷热条件比较tileCnt和bufferSize，记录重复波动及慢rank尾部；不能只看AIC利用率或主导pipe得出融合收益。来源最后5次/10轮是本脚本选择，不是普遍统计标准；“距理论>50%即FAIL”留在Skill，不作为硬件知识。
