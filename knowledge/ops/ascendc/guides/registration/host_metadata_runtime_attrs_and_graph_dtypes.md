---
type: Guide
title: Host 元数据、运行属性与图输出类型
description: 保留描述符查询、属性类型和实际注册绑定，避免用工程命名规则推导 API 禁令。
tags:
- task.implementation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-op-conventions.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-general.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/mc2-specific.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# Host 元数据、运行属性与图输出类型

Host 仅需 dtype/shape 时优先使用相应描述符和 shape context 接口，不能假定 GetInputTensor 总有实际数据。来源 GetInputDesc()->GetShape() 示例未给出匹配的 context 类型，不应据此发布统一原型；读取值依赖输入时仍可能需要真实 Tensor。返回的 descriptor、shape、attrs 及 GetAttrPointer 指针都要按目标合同确认存在后使用。

来源属性映射例为 Int→int64_t、bool→bool、float→float、string→const char*（GetAttrPointer<char>），模板类型须与实际 IR 属性存储一致；不能把 int64_t 属性当 int32_t 指针解引用。当前运行属性来自当前 context，CompileInfo 主要承载编译信息，跨调用缓存运行属性可能让新输入使用旧值。真正编译期常量可以保留，不能以名称禁令替代生命周期分析。

图注册需要让每个输出 dtype 按接口规则可推导。InferShape 与 InferDataType 是不同能力，函数定义存在还需被正确 OpName 的注册链消费。来源给出 IMPL_OP(OpName).InferDataType(fn) 及 IMPL_OP_INFERSHAPE(OpName).InferShape(shapeFn).InferDataType(dtypeFn) 两种接线；多输出不能默认全部复制输入 0 dtype，例如值与索引输出可能不同。是否有框架默认推导应依目标路径确认。

TilingKey 编码与模板实例要一致，但禁止 dtype 模板参数、禁止 TILING_KEY_IS、要求 _apt 改名均属于来源工程规范。dtype 特化本身是有效设计轴；源代码存在 legacy 分发并不证明技术上不可执行。命名、目录或整数 schMode 也不能自动避免编码冲突，需核对 Host 选择与生成实例。

宏注册回调、导出入口、Kernel entry、测试框架入口可能没有普通 C++ 调用者；Host 写而 Device 读的字段也不能按单侧搜索删除。删除“重复调用”前应确认它们同一路径实际重复执行、输入不变且没有状态/错误处理作用；不同互斥分支出现相同函数名不等于重复执行。

多线程 Tiling 的共享可变状态需隔离或同步；每调用独立对象的成员并不自动构成竞争。量化模式按当前算子实际支持子集校验，库新增枚举不要求该算子全部支持。worldSize、rank、group 等范围由真实合同决定，不能把所有上限检查都删成 !=0。
