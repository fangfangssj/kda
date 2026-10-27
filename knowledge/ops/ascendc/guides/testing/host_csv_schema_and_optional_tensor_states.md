---
type: Guide
title: Host UT CSV 的字段合同与 Optional 状态
description: 保留来源 CSV loader 的编码、实例和解析关系，区分框架约定与 Tensor 本体语义。
tags:
- task.testing
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/csv-format-spec.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/csv-refactor-workflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/ops-transformer.md
  role: supporting
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# Host UT CSV 的字段合同与 Optional 状态

来源 ops-transformer Host CSV loader 以 case_name/expectResult 标识用例，Tiling/InferShape 的每个 Tensor 用 shape、dtype、format 三列，InferDataType 通常只需 dtype；属性列按真实 OpDef 名称和类型读取。shape 维度用空格分隔，字段用英文逗号；首行是表头，其余为数据。这是特定 loader 的格式，不是所有 CSV 工具的限制。

空字段与额外字段必须按表头数量解析。来源一面禁止“尾随逗号”，一面示例用 ,,, 表达末尾 Optional 三个空列；合法末尾空字段不能机械删除，否则列数错误。空行、注释行、引号/逗号转义和空字段支持要按实际 loader 确认，通用 CSV 规范不等于它已实现。

来源 Tiling Optional 缺省时三列空，InferDataType 的缺省 dtype 空，InferShape 约定用 shape=0 并保留 dtype/format；但同文失败输出又允许 0 或空，存在歧义。应由 GetTensorGE/GetDataTypeGE 返回的实例标记与 Faker 消费关系确认“缺省”。不能把字符串 0、shape=[0]、空 shape=[] 与 nullptr 合为同一 Tensor 状态；DYNAMIC 的实例数量也不能压成单个存在 bool。

HostUtParamBase 已有 case_name、expectResult、inputInstance、outputInstance。来源派生示例又声明同名 vector<int>，会隐藏基类 vector<uint32_t>，导致两套状态；应保持唯一且类型一致的实例数据，并按真实输入输出顺序传入测试上下文。TD_DEFAULT/ID_DEFAULT 和 DT_UNDEFINED 只是初值，不会自动保证缺省语义。

属性解析对应 int64→stoll、float→stof、string→ReadMap、bool→规定的 0/1；应校验缺失、完整消费、范围和非法布尔文本。默认值来自接口合同，不因构造器字段有默认就允许对空字符串无条件转换。expectResult=FAILED 时来源允许不填 key/hash，而示例无条件 stoull 会抛异常，需先判字段与用例类型。

来源 token 包含 FLOAT/FLOAT16/BF16、INT4/8/16/32/64、UINT8/16/32/64、BOOL/DOUBLE/STRING；format 示例 ND/NCHW/NHWC/HWCN/NC1HWC0。它们映射 GE 枚举的前缀，不证明目标算子支持这些组合。Tiling 期望 key 为 uint64，不能经过浮点表格数值损失精度；hash 为文本。Shape/dtype 输出期望和状态分别比较，失败用例不能凭任意输出占位推断成功输出语义。

GetCasesFromCsv 与 TEST_P/ValuesIn 在运行时加载参数。来源 PrintCaseInfoString 使用 testing::TestParamInfo，若其头文件未自行包含 gtest，使用者需先提供 gtest 声明；这是头文件依赖缺口，不是 C++ 要求 gtest 永远第一。ReplaceFileExtension2Csv(__FILE__) 将源文件名关联 CSV；BUILD_PATH/工作目录与打包位置须让运行程序读到实际数据。_shape_/_dtype_ 文件名是否参与构建选择取决于本仓 CMake。
