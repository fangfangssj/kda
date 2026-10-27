---
type: Guide
title: TTK Kernel CSV 的 TensorList 编码
description: 保留26列位置与嵌套范围，限制版本特有的广播和别名规则。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/01-csv-common.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/02-kernel-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/ttk-converter/03-kernel-mode.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# TTK Kernel CSV 的 TensorList 编码

来源Kernel CSV固定列顺序为 testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries。公共9项加Kernel17项的定义表排序不等于最终CSV物理顺序。

CSV含逗号/引号/换行的单元格需要正确CSV转义；内部Python literal单元素tuple需尾逗号，(12289,)是单轴shape而(12289)是整数。((a,b))与((a,b),)不同层次，若loader兼容二者要明确归一化，而不能称数学上等价。来源禁用repr/强制str是生成策略，不是CSV语法要求；序列化器必须保持字符串、None、嵌套及数值类型。

TensorList shapes按每个子tensor展开，外层按算子参数槽组织。来源TTK版本对dtype string/scalar允许len==1广播，因此混合示例 input_shapes=(((1,2),(3,4),(5,6)),(7,8)) 与 input_dtypes=(('float16',),'float32') 可表示一个3成员列表加一个普通tensor。异构列表不能压成单dtype；具体_normalize_field_by_dist/_flatten_by_distribution行为需与实际TTK版本绑定，来源未附当前外部源码核验。

range/tolerance本身就是pair，压缩嵌套可能被当作更深容器，不能照搬dtype广播。通用表达为input_data_ranges=(((-10,10),(-10,10),(-10,10)),(-1,1))，TensorList输出tolerance同样逐成员展开。扁平单range在单列表场景可工作也不证明混合输入可工作。input/output_formats及ori字段的扁平回退是该TTK适配规则，异构格式要确保消费者能无损表达，不能忽略子项。

null形状转None仍保留参数位置、dtype与format；静态正shape转动态-1的框架推导不能代替真实动态约束。ori留空只在回退值与原始逻辑/存储格式一致时成立。来源将output_shape_unknown_indexes固定()，不适合所有数据依赖输出；manual binary为空也不表示已经有独立golden。

attributes与const_inputs用{**attrs,**consts}合并时，同名键会被const覆盖，需证明无冲突或明确合同。output_inplace_indexes=()据来源触发op_info同名输入/输出的自动推导，这是具体TTK实现约定，不是普遍“空即inplace”规则；真实别名非同名或op_info陈旧时应显式保真，不能凭shape或算子名猜。

is_enabled控制是否实际执行，soc_series据来源支持以-前缀排除，priority是筛选标签；统计CSV行数时不能等同实际执行数量。缺省network/remark/dump/ori等字段的语义与None字符串要分开。26列能被解析不保证算子名字与OP_ADD映射一致或参数顺序合法。
