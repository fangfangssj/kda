---
type: Guide
title: 算子测试的行为断点与属性交叉覆盖
description: 从实现分支和接口合法域识别尾块、资源轮换及路由边界，避免随机样本数量冒充覆盖。
tags:
- task.testing
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/case-coverage-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/design-and-harness.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/blaze-acceptance.md
  role: supporting
created_at: '2026-09-11T03:15:06Z'
updated_at: '2026-09-11T03:15:06Z'
platforms: []
---

# 算子测试的行为断点与属性交叉覆盖

测试特征应覆盖实际接口的 dtype、format、transpose、packing、量化方式、可选 scale/offset/bias、groupSize，以及它们对数学、地址、资源和路由的影响。接口支持约束、性能建议和当前实现路由分别描述：推荐对齐不能删除合法非对齐功能测试，现有测试名称也不能反推完整支持域。

Shape 断点来自真实行为：最小/最大合法值，tile-1/tile/tile+1，group tail，单核到多核切换，buffer 数量或 ping-pong 并发切换，workspace/TilingKey/fallback 切换。所谓小、中、大应对应单 tile、稳定复用、多轮资源压力等实际条件，而不是跨算子套固定数值。空输入及合法零维单独按合同处理。

非对齐用例可打破内部 tile 整除，却仍需满足 packing、group、scale shape、地址和公开接口的必要约束。记录具体破坏了哪种整除关系，以及为何仍合法，避免把非法形状的拒绝当成 tail 支持。

当属性改变 Kernel、layout、地址、同步、转换/累加顺序，或与尺寸/tail/多核分支耦合时，需要相应属性与 Shape 交叉覆盖。只有证明相关行为等价才能共享代表用例；无需机械构造所有轴的笛卡尔积，也不能仅因用例多而压掉语义不同的路径。

随机数适合在已确定的合法域和行为类内填充具体值，不能代替覆盖结构。保存最终 Shape、生成器版本、seed、输入字节和实际路由；seed 单独不足以重现变化后的生成器。重抽次数及原因可解释候选筛选，不能通过重抽失败输入掩盖错误。

覆盖由每个约束/行为对应哪些已执行 case 推导，包含合法组合、非法拒绝、fallback、tail、资源复用与多输出。用例数、每属性出现次数和整体通过率都不能独立证明覆盖；来源固定 10~30 正式用例是 Skill 流程约定，不是知识结论。
