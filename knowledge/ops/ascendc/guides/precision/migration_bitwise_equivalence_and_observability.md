---
type: Guide
title: 迁移比较的逐字节等价与可观测范围
description: 区分数值正确性、确定性和逐位迁移等价，覆盖动态输出、原地状态和真实执行。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/blaze-acceptance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/design-and-harness.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/original-baseline.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/serial-workflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/workspace-and-evidence.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/references/non_compute.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/scripts/bitwise_match.py
  role: implementation
created_at: '2026-09-11T03:15:06Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 迁移比较的逐字节等价与可观测范围

逐字节等价与容差等价是不同合同。来源 Blaze 迁移场景选择以前实现的稳定输出为基线，目标有效输出字节、必要动态元数据和 inplace after-state 全部一致；这种合同下 rtol/atol、平均误差和独立 Golden 不能替代字节比较。它是该迁移场景的验收选择，不能提升为所有 Ascend C 算子必须逐位一致。

逐位比较保留 +0/-0、NaN payload、Inf 的真实表示；仅对定义的有效输出范围比较，不比较未定义 padding。动态 shape、length、index、count 与最终数据共同决定结果，不能只比固定最大 buffer。原地算子需比较修改后的对象，并在重复测量前恢复规定输入状态，否则各轮执行的已不是同一问题。

原始实现重复输出稳定只证明这些条件下的确定性，不证明数学正确；两侧同错也能逐字节相等。非确定性算子不能在未确认支持条件、执行模式和累加顺序时直接套稳定二进制基线，确定性要求与误差要求分别确认。文档支持但基线失败的 case 仍暴露待定位问题，失败不能自动成为缩小公开支持域的依据。

退出码 0、目标符号存在、输出文件存在均不证明目标 Kernel 本轮实际执行。需对应输入、路由、执行次数和真实产物；合法 fallback 检查其输出与实际路线，非法输入检查拒绝阶段和行为。完整结果包括全部输出与失败/未执行项，不能只汇总通过样本。

ACLNN runner 只能直接检查自己控制的 Host buffer canary、输出初始化、字节数、动态元数据及可观测 after-state；不能凭外层 guard 断言内部 padding/workspace 无越界。设备内部访问另需对应分析或检测证据。本文来源未在本批进行设备复现。

## 数值相等不等于逐字节相等

固定来源 `bitwise_match.py` 实际调用 np.array_equal；该函数的数值相等语义不能直接称作 bitwise。NumPy 1.26.4 的可复现反例包括：float32 的 +0.0 与 -0.0 返回相等，int8 的 [1] 与 int64 的 [1] 返回相等，而字节相同的 NaN 默认返回不相等。比较器名称和返回字段 bitwise_match 都不能改变这些语义。

字节合同比较需先约定 dtype、shape、元素顺序、字节序、有效输出范围和是否要求相同物理布局，再比较规定的表示；不能把非连续视图的整个底层 buffer 或未定义 padding 一并比较。Cast 等算子的输出合同还取决于目标类型、舍入和特殊值，不能把所有“非计算”类别都要求输入输出原始字节不变。
