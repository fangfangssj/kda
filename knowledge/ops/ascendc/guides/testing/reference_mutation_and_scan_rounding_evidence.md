---
type: Guide
title: 参考污染与 Scan 舍入证据
description: 保留 torch 算子加载和 scan 差异诊断，排除改写基准来匹配候选。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/AscendCVerification.md
  role: primary
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# 参考污染与 Scan 舍入证据

仅包含TORCH_LIBRARY静态注册而无Python扩展入口的共享库，可由torch.ops.load_library加载，包__init__.py也可封装该调用；不能据此称所有.so都不能import，具有PyInit入口的扩展另有机制。候选forward应保留参数位置、默认值和输出结构，注册namespace、包名和OUTPUT_NAME/_kernel.<plat>.so是该构建模板的命名约定，不是PyTorch普遍要求。

来源声称某NPU版本FP16二维dim0 cumsum重复运行波动、小tensor约0.1–0.5%、8192×16384约10%，但无本批绑定硬件/版本/日志；dim1确定性串行、BF16无问题也不能推广。需要区分重复执行不一致、合法并行归约顺序差和数学误差，记录相同输入、stride、dtype、设备/库版本以及独立高精度输出。

模块加载时monkey-patch全局torch.cumsum会改变之后调用的reference，因此候选与被改写基准一致不能证明原合同正确。来源包装还丢弃dtype/out等args/kwargs，input.T通常只是非连续视图，改dim并不自动证明走连续物理扫描。转置等价只在相同数学轴和完整参数合同成立时有效，需要独立测试；cumprod也不能因为同属scan就改用cumsum逻辑。

FP16逐步舍入与FP32累加末尾转FP16可能不同，应由明确合同或要复现的固定参考定义。来源L≤512选FP16、L>512选FP32并借大值rtol通过是特定经验，不能作为通用算法切换或精度依据。CAST_NONE不是通用向零截断，实际舍入依赖源/目的dtype和平台重载，CAST_TRUNC与CAST_ROUND/RINT应分别核对。BF16输出不能直接继承只针对FP16转换的建议。

构建成功、包import成功及torch.ops存在分别是不同证据；最终仍需实际调用、所有输出、有效shape和未被污染的独立参考。统一评测脚本入口和Skill步骤不迁入为知识库操作流程。
