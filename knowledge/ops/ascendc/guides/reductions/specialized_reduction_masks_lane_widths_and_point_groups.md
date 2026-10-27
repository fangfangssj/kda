---
type: Guide
title: 特化归约 Mask、Lane 宽度与采样点组
description: 保存CLZ二分、分层平方和及INT64辅助路径。
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

# 特化归约 Mask、Lane 宽度与采样点组

MX非尾轴优化用floor(log2(rowsSingleLoop))构造2次幂主体，clz输入0须单独处理，源1<<loopSize使用int常量且rows/expOffset为uint16，较大移位或rows*dataLen可能先溢出/截断。必须在宽域计算与验证后窄化；非2次幂额外rows还需先折叠或独立合并，所示循环未展示，不能自动丢掉尾rows。

二分Max每层VEC_STORE→VEC_LOAD的LocalMemBar仅保护局部存取；DataCopyUnAlignPre/UnAlign的状态u0、真实地址、maskNum和寄存器有效lane需遵守API。树深O(logN)，完整数据读取/比较总工作仍O(N)，不能按深度宣称算法只需对数次标量比较。

InplaceAddRMS来源reduceNum≤VL、≤2VL、≤2VL²及更大四类平方和路径。常量VL须对应实际FP32向量宽度，binaryAddQuotient/Remainder和尾mask覆盖全部原值；示例循环每次写sumReg和vMean但没展示跨r累计/保存，不能据函数名认为全归约完成。先平方再加与先加后平方不同，归一化时还需完整除N/epsilon/rstd。

MaxPool Argmax按sizeof(index)/sizeof(value)选择mask扩展，ratio1直接Select、ratio2 MaskUnPack是部分路径；INT64索引配FP16值时ratio4未在片段覆盖，需额外扩展或专用ComputeMultiBatchForInt64。helper buffer容量、寄存器lane映射、value/argmax共同更新与tie/NaN政策必须一致，MaskUnPack不是数值转换或安全窄化。

可变形Attention对2/4/8采样点组采用树形Add，offset4*alignedD、2*alignedD、alignedD按对应组逐层合并，再加到输出累积。num_points3等不在这些完整二分组中，不能只按>=2就认为第三点已被消费；应由Host分成受支持组并处理尾组。四角插值权重应先正确应用，点组归并与每点四角归并是不同层。

SetVectorMask类状态需覆盖真实有效channel并按调用者合同恢复，output[outOffset]首次加前初始化，别名/边界和全部点组完成再写回。原文没有baseline的路径仅提供设计，不能标成已证明最佳性能。
