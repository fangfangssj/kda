---
type: API
title: TPipe
description: 构造用来管理内存和同步的TPipe对象。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/TPipe_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tpipe.h
---

# TPipe构造函数<a name="ZH-CN_TOPIC_0000002027238497"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

构造用来管理内存和同步的TPipe对象。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline TPipe()
```

## 约束说明<a name="section633mcpsimp"></a>

-   避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化，具体原理请参考[避免TPipe在对象内创建和初始化](../../../../../guides/operator_practice/simd_operator_optimization/overhead_optimization/avoid_tpipe_init_in_object.md)。
-   TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡死等随机行为。如果需要使用多个TPipe时，请先调用[Destroy](destroy.md)接口释放前一个TPipe。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(..., TPipe* pipeIn)
    {
        ...
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        ...
    }
private:
    ...
    TPipe* pipe;
    ...
};
extern "C" __global__ __aicore__ void example_kernel(...) {
    ...
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    ...
}
```
