---
type: Guide
title: Host C++ 所有权、对象表示与声明边界
description: 补齐异步捕获、智能指针、宏展开、字节复制和 ABI 扩展的适用条件。
tags:
- task.implementation
- topic.memory
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-general.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-secure.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-red-line.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-style.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# Host C++ 所有权、对象表示与声明边界

资源应有明确所有者：new/delete、new[]/delete[]、malloc/free 以及文件/stream/锁各自匹配，异常和部分初始化失败也需回收。普通 throwing new 失败走异常，nothrow new 和 C 分配接口才相应检查空返回。RAII 可管理 Host 资源，也可封装设备侧资源协议；“Kernel 没有 RAII 场景”并不成立，但具体 Device C++ 能力要依编译上下文确认。

独占所有权适合 unique_ptr，共享才需要 shared_ptr；make_shared 可统一创建对象与控制块，不能强制把所有转移都变为 shared_ptr+move。不涉及所有权的引用/指针由调用合同保证寿命，不得擅自 delete。置空一个已释放指针不会修复其他悬空别名。延迟 lambda 按引用捕获局部量时，局部量必须活到执行结束；值捕获对象也要检查内部借用指针。

std::move 只是转换表达式值类别，不搬数据；对 const 对象操作常仍选择复制，或因无匹配构造失败。显式拷贝/移动/析构要共同维护资源不变量；无独占资源时可优先默认操作。override 检查覆盖关系，final 限制后续覆盖；虚函数默认实参按静态类型绑定，动态实现与默认值可能来自不同类。

explicit 用于控制隐式转换和相应列表初始化，多参数构造也可合法声明 explicit，来源“多参数禁止”是风格约定。const 引用可避免某些对象复制，但小标量按值并不错误；可选输出指针与必需引用表达不同合同。void* 或 reinterpret_cast 需要明确类型、对齐、地址空间和生命周期，不是换成强类型名字就完成验证。

sizeof(pointer) 返回指针对象大小，不是所指数组容量；LocalTensor 对象的 sizeof 也不等于 UB 区大小。字节复制需满足对象生存期与 trivially-copyable 等条件，memset 全零并不等于任意 C++ 对象的合法初始化。Ascend C 模板类不全是 POD，Duplicate 不能泛化为初始化任何类对象。TilingData 末尾加字段可能保留部分旧字段偏移，却会改变 sizeof、对齐、版本和消费者读范围，不能单凭“尾加”宣称 ABI 兼容。

优先复用正式头声明而不是手写 extern 猜签名；extern 本身合法。把含 C++ 声明的头放进 extern "C" 或 namespace 会改变 linkage/声明归属；Kernel 入口是否 extern "C" 依实际启动 ABI，不是所有模板入口的统一语法。using namespace 改变名字查找，不把后续 include 的声明自动搬进该 namespace；命名空间内 using 的影响还可经 qualified lookup/ADL 等路径扩散，不能按目录作无风险豁免。

宏参数需括号和明确求值次数，局部临时名仍可能与实参冲突；仅加 _macro_ 前缀不能保证卫生。GNU ({...}) 是扩展，不能当可移植 C++。多语句宏和无花括号 if 容易改变控制流；避免同一表达式中不清晰的自增/读取顺序。文件命名、缩进、比较左右位置、TODO 和注释格式继续作为 Skill 风格规则维护。
