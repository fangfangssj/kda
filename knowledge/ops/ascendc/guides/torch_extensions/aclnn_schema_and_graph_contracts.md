---
type: Guide
title: PyTorch 自定义 ACLNN 算子的四层接口契约
description: 对齐 ACLNN 参数、GE OpDef、PyTorch schema 与 Meta/图转换，避免将不同接口层的名称和顺序直接互换。
tags:
- task.implementation
- topic.implementation.operator_registration
- topic.graph_execution
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ascendc-op-extension/references/aclnn/schema_mapping.md
  role: primary
created_at: '2026-09-10T12:45:59Z'
updated_at: '2026-09-10T12:45:59Z'
platforms: []
---

# PyTorch 自定义 ACLNN 算子的四层接口契约

| 接口层 | 决定的内容 |
|---|---|
| 实际 ACLNN 头文件 | C 接口参数类型、顺序、输出参数及工作空间/执行器协议 |
| GE OpDef 与推导实现 | 图输入、属性和输出名称，支持的类型组合及输出推导 |
| PyTorch schema | Python 调用签名、可选项、默认值、返回值与修改/别名语义 |
| Meta 和图后端 converter | 不执行设备计算时的元信息，以及 schema 到图节点的映射 |

这几层共同描述同一算子，但参数名称和顺序不必相同。converter 构造图节点时应匹配 OpDef，不能直接照搬 ACLNN 的 C 参数名称。NPU 包装则应匹配实际 ACLNN 声明；辅助宏可能负责附加工作空间和 executor，不能重复传入。

可选 tensor 缺省时，图节点应按接口语义省略或表示该输入；把不存在的输入变成一个普通空 tensor 可能改变算子行为。可选属性、默认值及输出数量也应在 eager、Meta 和 converter 中一致。

若 Python 接口接收预分配输出或修改输入，schema 的修改和别名标注必须描述真实行为，并使用当前 PyTorch 支持的语法。Meta 应返回与 NPU 路径一致的 shape、dtype 和结构，不能固定复用输入元信息来掩盖真正的输出规则。

验证至少覆盖可选项存在/缺省、非默认属性、不同 dtype、动态 shape 和实际输出结构。eager 测试只验证其中一条执行路径，不替代图转换验证。
