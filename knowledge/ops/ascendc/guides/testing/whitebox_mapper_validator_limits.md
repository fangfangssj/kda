---
type: Guide
title: Mapper 模板的校验能力与遗漏
description: 从完整模板提取结构校验边界及变体追踪，排除生成脚本流程。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/s5_case_mapper_template.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/S5_mapped_case_schema.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# Mapper 模板的校验能力与遗漏

来源模板校验7个顶层key精确集合、容器类型、tensor/list的字段集合、非空dtype/format和list count，禁止meta.supported_data_ranges。map_path_case/map_network_case固定source，并把id转字符串；shape变体使用low_case_N、meta.base_id和variant_kind，再检查唯一ID。它们建立描述结构与溯源，不能替代算子支持性或真实路径求值。

_validate_tensor_shape使用isinstance(dim,int)，Python bool也满足该条件；该函数没有排除负维或溢出。count检查同样会接受bool，而True与长度1相等。param_type/data_range只要求非空字符串，不是枚举验证；列表外层和每个子项range相等也未被这个模板证明。rank0、负数动态标记、合法dtype/format元组仍依赖目标schema/接口。

validate_mapped_case先校验再normalize_input_ranges，将输入及已有data_range的字典子项范围强制normal；这不是保留传入分布，也不自动补齐缺字段。若normal生成器违反数学域，结构通过仍会造出无效用例。outputs只做描述校验，没有检查推导是否随shape/attrs变化。

tensor构造helper的extra字典通过update可覆盖先前kind/dtype等字段；精确key检查能发现多余字段，不能发现所有语义性覆盖。列表helper保存原tensors引用，后续修改可能联动；clone_case的deepcopy才提供嵌套隔离。network_record的浅复制和setdefault也不能修复已有None id，路径ID用or回退可能把有效但假值标签改写。

build_low_base_case、make_path_shape_case、make_network_shape_case、derive_outputs均是NotImplementedError扩展点，模板本身不是任何算子的完整mapper。静态main没有独立强制调用derive_outputs，因此仅实现同名函数不保证变体实际用了它；每条最终记录须携带与该输入一致的输出。来源模板并未在本轮执行，以上是固定版本代码审阅结果。
