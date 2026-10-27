---
type: Guide
title: 测试约束图的语义与求解完整性
description: 区分推导、匹配、广播、存在性和类型兼容，保留可执行依赖模型及缺口。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-st-design/references/constraint-examples.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-st-design/references/dependency-yaml-spec.md
  role: supporting
created_at: '2026-09-11T03:52:52Z'
updated_at: '2026-09-11T03:52:52Z'
platforms: []
---

# 测试约束图的语义与求解完整性

测试因子节点parameter.attribute表达exist/dtype/format/dimensions/shape、length_ranges/length、shape_list/value_range/value；input/output/intermediate区分角色。静态枚举和支持范围属于节点域，跨参数关系属于约束。rank只确定shape长度，不唯一决定每维大小；dtype也只给编码范围，不决定算子合法数值域。来源derive_shape/value等名称代表生成器钩子，不是已给出的算法。

calculate按sources计算target，match表达双向相等关系，二者不能因结果有时相同而合并。来源要求calculate的sources非空是该schema约束；自引用修改grid.shape要先有输入值/显式生成阶段，不能按普通无环图直接拓扑排序。sources[0][:3]+[2]在rank不足时并不能保证最后shape为4维，要先证明rank约束并在修改后重验关联关系。

broadcast_dim核对指定合法维度索引，broadcast_shape核对右对齐形状。单向模式在此schema中是target广播到sources，字段方向与计算约束不同；“双向”表示可产生共同结果，并不表示两者都能原样扩展到彼此形状。0与1合并结果为0，不是逐维max。派生[M,N]可先calculate到intermediate，再检查bias/self广播；source_shape_expr未在来源schema支持。

conditional在条件成立时应用then/else；beta=0忽略self的数值只有目标算子如此规定才成立，不能自动推导self可缺席或shape/dtype无需检查。expression: '== 0'是相对factor的片段，并非独立可执行Python表达式，解析器须明确绑定。existential用于Optional未出现时将其属性置NA，NA不能与空Tensor/空列表或实际dtype混淆；必选输入缺席应按负例合同而非伪造Optional规则处理。

convertible来源约定target dtype可转换到sources dtype，方向与字段直觉相反。可转换、无损转换、可隐式提升以及算子允许的混合类型是不同关系；“整数可转浮点”不代表无损，浮点大小也不能决定所有复数/有符号混合类型。tensor_tensor与tensor_scalar须使用具体框架规则，弱标量与张量的值/类型影响不可只按较大类型。

来源tensor_scalar文字说Tensor优先，示例标第一列为Tensor却把(f32,f16)结果写f16、(f16,f32)写f32，彼此矛盾；inferable又将“是否兼容”和“推导结果赋值给target”混写。不能按这张表迁成类型事实，应将兼容谓词、计算结果、候选原始dtype分别表达并核对实际promotion表。

inferable_filter从target_domain筛与已定sources兼容的候选，逐层锚点采样可构造合法组合，但贪心路径不保证遍历全部合法组合，也可能较晚遇到空域。求解器需处理回溯/枚举、环、多约束交集、缺失节点、空候选、输出推导与最终全约束复验；记录可行覆盖分母，避免把未采到或被顺序剪掉当不支持。

来源完整YAML示例多次重复factors键可能被解析器静默覆盖，末尾例子还引用未声明batch1/batch2；equal类型占位与实际类型清单不一致。落地数据模型需唯一键和唯一constraint id、节点引用完整、索引合法、表达式允许集合、明确函数实现及解析错误报告。表达式可计算不等于应对任意文本直接eval。命名序号、产出文件顺序和用例数量配额留在Skill。
