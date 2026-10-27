---
type: Guide
title: Count 分组 Matmul 的参考语义与 API 模式边界
description: 保留连续 token 分组、Bias 与输出拼接，限定简化 torch_npu 描述的适用范围。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/reference-generation/references/grouped_matmul_api_description.md
  role: primary
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-11T03:46:00Z'
platforms: []
---

# Count 分组 Matmul 的参考语义与 API 模式边界

来源BF16 benchmark的数学语义为每组Yi=Xi@Wi+Bi，输入X=[M,K]、W=[G,K,N]、可选Bias=[G,N]，按count_i连续切分X的M轴，最后按组顺序拼接为[M,N]。定义offset_0=0、offset_(i+1)=offset_i+count_i，需count非负整数、组数匹配且sum(count)=M；不能仅依赖slice自动截短来验证合同。

这是明确的count模式参考，不是所有group_list的默认定义。来源函数签名简写x/weight为Tensor、返回List[Tensor]，约束又称输入列表最多128，golden则返回单Tensor；实际torch_npu接口的list/split/group模式、累计边界与count、group_type及输出结构必须由目标版本确认。group_list=None的多Tensor模式不能由当前单Tensor循环自动覆盖，K<65536和列表128也不迁为通用硬件限制。

每组独立可合并调度以减少launch开销，但收益需实测；均匀/不均匀token、不同专家矩阵、Bias可选与组顺序是功能维度。来源测试既有[16,16]，也有[8,24,16,32]与[200,56,128,128]等非均匀分配；这些用于说明分片，不证明所标模型完整结构。

空组count=0、G=0、M=0以及shape不一致需定义结果。torch.cat空results会失败，不能用其异常代替算子空输出合同；来源在每组matmul后再加bias的工作dtype舍入点，也可能不同于FP32累加/Bias融合后一次Cast。Golden若提升精度需明确是数学参考还是迁移等价参考，不能静默改变链路。

Pythongroup_list[i].item()可能触发Host取值及同步；适用于参考构造不代表Device性能方案。FP16/BF16/FP32及INT8/INT4量化支持要按真实接口模式核对，Cube具备某计算能力不证明此算子暴露相同类型。更一般的组尺寸、地址和反量化坐标见[group形状合同](group_shapes_and_dequant_coordinates.md)。
