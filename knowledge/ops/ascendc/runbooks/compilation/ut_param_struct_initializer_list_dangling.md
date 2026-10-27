---
type: "Runbook"
title: UT 参数结构体 initializer_list 悬垂致维度垃圾
description: "gtest 参数化用例的参数结构体用 std::initializer_list 存 shape，底层数组随临时对象销毁而悬垂；由它构造 gert::StorageShape 时 dim_num 正确但 dims 全是指针垃圾，tiling 输出超大数值。改 std::vector + AppendDim 构造即恢复。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T03:16:34Z'
updated_at: '2026-09-17T03:16:34Z'
platforms: ["950"]
---

# UT 参数结构体 initializer_list 悬垂致维度垃圾

## 现象

在自定义 argmax 算子（Ascend950PR，CANN 9.1.0）op_host UT 的段错误修复（compileInfo 传非空）之后，tiling 用例能跑通 TilingFunc 了，但断言大面积失败，且失败值非常反常——TilingData 字段全部是"指针量级"的数：

```
[DEBUG] TilingFunc: rank=2 dim[0]=94115993547440 dim[1]=94115993542016
[DEBUG] TilingFunc: a1=94115993547440 r=94115993542016 a0=1 inputTotal=8007811680211519488 dim=1
[DEBUG] TilingFunc: done. tiling ptr=0x55991662ae20 aSize=94115993547440 rSize=94115993542016 a0=1 blockDim=483664
Expected equality of these values:
  tiling->aSize   Which is: 94559231636144
  param.expectA1  Which is: 4
```

期望 aSize=4、rSize=8，实际是 9.4e13（十六进制 0x55…，堆指针）。dump raw tiling 前 32 字节也是两个堆指针 + 垃圾：

```
dataPtr=0x55afce447e20, first 32 bytes: b0 c2 3e ce af 55 00 00 80 ad 3e ce af 55 00 00 01 00 00 00 ...
```

当时代码状态：测试参数结构体 `ArgMaxTestParam` 的 `inputShape` 字段声明为 `std::initializer_list<int64_t>`，用 `ArgMaxTestParam{"float_ar_basic_2d", {4, 8}, ...}` 构造后交给 `testing::Values(...)`；用例内用 `gert::StorageShape shape = {param.inputShape, param.inputShape};` 构造输入 shape。

## 适用场景

- 工程形态：自定义算子 op_host UT（gtest 参数化用例 + gert::StorageShape/TilingContextFaker）。
- API 路径：`gert::StorageShape` / `gert::Shape`（构造函数吃 `const std::initializer_list<int64_t>&`，头文件 exe_graph/runtime/storage_shape.h、shape.h）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——C++ 标准库语义 + gert shape 构造组合，与算子和平台无关。
- 路线优先级：recommended_fastpath（正确写法）。
- 明确不适用 / 未验证场景：直接在用例函数体内就地写 `StorageShape shape = {{4, 8}, {4, 8}}`（字面量数组生命周期覆盖整个全表达式，构造完成即拷贝进 Shape）不踩此坑；本坑专门针对"initializer_list 被存进长生命周期对象再延迟使用"。

## 根因

`std::initializer_list` 不拥有底层数组：数组是构造全表达式里的临时对象，`initializer_list` 只是 (指针, 长度) 二元组。把它作为字段存进 `ArgMaxTestParam` 并经 `testing::Values(...)` 拷贝传递后，原临时数组早已销毁，字段成为悬垂引用。`Shape(const std::initializer_list<int64_t>&)` 构造时按悬垂指针逐元素拷贝，于是 `dim_num_`（从 size 拷贝，正确=2）没问题，而 `dims_[i]`（从已释放内存读，值=堆地址）全是指针。TilingFunc 拿这些"维度"做乘除，得到 9.4e13 量级的 aSize/rSize。

## 解决配方

修复分两步（`tests/ut/op_host/test_<op>_tiling.cpp`）。

**before**：

```cpp
struct ArgMaxTestParam {
    std::string caseName;
    std::initializer_list<int64_t> inputShape;   // 悬垂：底层数组随临时对象销毁
    ge::DataType inputDtype;
    int64_t dim;
    bool keepdim;
    ge::graphStatus expectStatus;
    // Expected values (for GRAPH_SUCCESS cases)
    uint64_t expectA1;       // A1: product of dims before reduce axis
    uint64_t expectR;        // R:  reduce axis length
    uint64_t expectA0;       // A0: product of dims after reduce axis
    uint32_t expectMode;     // 0=AR, 1=ARA
};

// 用例内：
    gert::StorageShape shape = {param.inputShape, param.inputShape};
```

**after**：

```cpp
struct ArgMaxTestParam {
    std::string caseName;
    std::vector<int64_t> inputShape;             // 拥有数据的容器
    ge::DataType inputDtype;
    int64_t dim;
    bool keepdim;
    ge::graphStatus expectStatus;
    // Expected values (for GRAPH_SUCCESS cases)
    uint64_t expectA1;       // A1: product of dims before reduce axis
    uint64_t expectR;        // R:  reduce axis length
    uint64_t expectA0;       // A0: product of dims after reduce axis
    uint32_t expectMode;     // 0=AR, 1=ARA
};

// 用例内（vector 不能隐式转 initializer_list，用 AppendDim 逐维填）：
    // Build StorageShape from vector (avoid initializer_list dangling reference)
    gert::StorageShape shape;
    for (auto d : param.inputShape) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
```

注意中间尝试过的 `gert::StorageShape shape = {std::vector<int64_t>(param.inputShape), std::vector<int64_t>(param.inputShape)};` 不可行——`Shape` 构造函数收 `initializer_list`，`std::vector` 不会隐式转换过去；正确落点是默认构造 + `MutableOriginShape()/MutableStorageShape().AppendDim(d)`（origin 在前、storage 在后，StorageShape 构造函数参数顺序也是 (origin_shape, storage_shape)）。

## 设计决策

- 选 `std::vector` 字段 + `AppendDim` 而不是"让参数结构体持 array 引用"：vector 拥有数据、可拷贝，是 gtest `Values(...)` 安全传递的唯一简单解；AppendDim 路径绕开 initializer_list 生命周期问题。
- 先用 sizeof 探针排除了"UT 可执行与 .so 结构体布局不一致"（sizeof(StorageShape)=536、sizeof(Shape)=248、sizeof(Tensor)=752 两侧一致），又用地址探针确认 `Tensor[0]` 地址即 `StorageShape` 地址、`dim_num=2` 正确而 `dim0/dim1` 为指针，才把根因压到"构造入参已是垃圾"。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；gtest 参数化框架；`StorageShape`/`Shape` 布局：Shape = size_t dim_num_ + int64_t dims_[25] + reserved（248B），StorageShape = origin + storage + reserved（536B），Tensor 内含 StorageShape 成员（752B）。
- 相关文件：`tests/ut/op_host/test_<op>_tiling.cpp`（参数结构体与 StorageShape 构造）、`tests/ut/common/tiling_case_executor.cpp`（TensorDescription→builder 链路）。

## 预防

- 参数化测试的参数结构体一律用拥有型容器（`std::vector<int64_t>`）存序列数据，禁止存 `std::initializer_list`。
- 看到"数值是指针量级（0x55…/0x7f… 开头的十进制大数）"的断言失败，第一反应查"数据从悬垂/已释放内存读出"，而不是查算子逻辑。

## 验证信号

来源说明：该算子 op_host UT 调试会话的前后对照（gtest/bash 输出）。

- 修复前：`dim[0]=94115993547440 dim[1]=94115993542016`、aSize=94559231636144（期望 4）、raw tiling 首字节为堆指针对 `b0 c2 3e ce af 55 00 00 …`（来源平台：950；证据等级：tool_observed）
- 地址探针：`Tensor[0]=0x556a92887e60, StorageShape at 0x556a92887e60, dim_num=2, dim0=93916212945584 dim1=93916212940160`——dim_num 正确、dims 为指针（来源平台：950；证据等级：tool_observed）
- 布局排除：两侧 sizeof(StorageShape)=536 / sizeof(Shape)=248 / sizeof(Tensor)=752 一致（来源平台：950；证据等级：tool_observed）
- 修复后：op_host UT 27/27 通过（含 tiling 10、infershape 15、tilingkey 1、workspace 1），op_api 10/10，共 37/37（来源平台：950；证据等级：tool_observed）

## 失效边界

- 结论绑定"initializer_list 存入长生命周期对象再延迟使用"这一形态；就地字面量构造 StorageShape 不受影响。
- 未在其他编译器/标准库实现上复现（理论上所有符合标准的实现都会悬垂，但未逐一验证）。

**未决项**：

- 无（根因已追到语言语义层并闭环验证）。
## 坏实践

- 用 `std::initializer_list` 做 gtest 参数结构体的字段类型——它只是视图，永远不要当数据成员存。
- 试图用"套一层 std::vector 临时对象"修 initializer_list 构造——vector 不会隐式转 initializer_list，编译都过不了。

## 误诊

- "UT 可执行与 op_host .so 的 StorageShape/Shape 结构体布局不一致（ABI/编译选项差异）"——sizeof 三探针证明两侧一致（中间假设，被 tool_observed 推翻）。
