---
type: "Runbook"
title: "tiling UT 标量输入须带 isConst/constValue"
description: tiling UT 里标量输入的 TensorDescription 只给 shape/dtype 不设 isConst/constValue 时，faker 张量数据指针为 nullptr，tiling 的 OP_CHECK_NULL_WITH_CONTEXT 返回 GRAPH_FAILED，全部用例 Tiling failed；push_back 时补 `true, &scalarVal` 即解。
tags: [task.testing, topic.memory, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# tiling UT 标量输入须带 isConst/constValue

## 现象

自定义算子（tiling 阶段需要读入一个标量输入张量的值来决定 TilingKey/范数类型）的 op_host UT：迭代一遗留的 3 条 tiling 用例在升级后的算子上全部失败，gtest 输出：

```text
[ RUN      ] ForeachNormTilingTests/ForeachNormTilingTest.tiling_test/0
[TEST_CASE] single_tensor_1024
test_bench_no_harness_foreach_norm_tiling.cpp:309: Failure
Value of: success
  Actual: false
Expected: true
Tiling failed for case: single_tensor_1024
[  FAILED  ] ...tiling_test/0 ...
（另两条用例同样 Tiling failed）
```

被测 tiling 代码用 `GetRequiredInputTensor(1)` 读标量并做了空指针防护：

```cpp
    const gert::Tensor* scalarTensor = context->GetRequiredInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalarTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalarTensor->GetAddr());
    float scalarVal = *(scalarTensor->GetData<float>());
```

## 适用场景

- 工程形态：op_host tiling UT（x86 gtest + `tiling_context_faker`/`tiling_case_executor` 桩），被测 tiling 需要在 tiling 阶段读取输入张量的**数据**（而非仅 shape/dtype）。
- API 路径：`gert::TilingContextPara::TensorDescription(shape, dtype, format, isConst, constValue)`；executor 以 `gert::TensorPlacement::kOnHost` + `constValue` 指针构造张量；`GetRequiredInputTensor(i)->GetData<T>()`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（arch35 op_host 源码）+ CANN 9.1.0，UT 在 x86 主机运行。
- 适用粒度：`platform_wide`（UT faker 的数据注入契约与算子无关）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证场景：只读 shape/dtype 的 tiling（不注入数据也能过）；设备侧数据在 UT 桩里本就不存在，无需模拟。

## 根因

UT 用例构造输入张量描述时，标量输入没有设置 `isConst=true` 和 `constValue`（默认 `isConst_=false, constValue_=nullptr`）；`tiling_case_executor` 对非 const 输入传 `nullptr` 作为数据指针，于是 `scalarTensor->GetAddr()` 为空，tiling 的 `OP_CHECK_NULL_WITH_CONTEXT` 防护直接返回 `GRAPH_FAILED`，表现为 "Tiling failed"。

## 解决配方

executor 侧机制（`tiling_case_executor.cpp`，读取已有实现，说明 constValue 如何变成张量数据）：

```cpp
for (size_t index = 0; index < inputNum; index++) {
    std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
        tilingContextPara.inputTensorDesc_[index].shape_,
        gert::StorageFormat(...),
        gert::TensorPlacement::kOnHost,
        tilingContextPara.inputTensorDesc_[index].dtype_,
        tilingContextPara.inputTensorDesc_[index].isConst_ ?
        tilingContextPara.inputTensorDesc_[index].constValue_:
        nullptr);
    ...
}
```

修复前（标量按普通 shape/dtype 描述 push，无数据）：

```cpp
    gert::StorageShape scalarShape = MakeStorageShape({1});
    inputTensorDesc.push_back({scalarShape, ge::DT_FLOAT, ge::FORMAT_ND});   // 无 isConst/constValue
```

修复后（标量输入必须带 `isConst=true` 与实际值指针）：

```cpp
    // scalar input (1-element tensor) — 必须设置 isConst=true 并传入实际值
    // tiling.cpp 通过 GetRequiredInputTensor(1)->GetData<float>() 读取 scalar
    gert::StorageShape scalarShape = MakeStorageShape({1});
    float scalarVal = param.inputScalar;
    inputTensorDesc.push_back({scalarShape, ge::DT_FLOAT, ge::FORMAT_ND, true, &scalarVal});
```

同时注意 IR 实例数要与 DYNAMIC 输入对齐（x 为 L 个实例、scalar 1 个、y 为 L 个输出）：

```cpp
    std::vector<uint32_t> inputInstanceNum = {static_cast<uint32_t>(L), 1};
    std::vector<uint32_t> outputInstanceNum = {static_cast<uint32_t>(L)};
```

## 设计决策

- 数据注入走 `TensorDescription` 的 const 值通道而不是给 faker 另开"host 数据"接口：const 通道是 executor 既有契约（kOnHost placement + constValue 即 `GetAddr()` 返回值），零改动复用。
- 标量存活期：`float scalarVal` 是栈变量，push 进 desc 后在同函数内立即执行 tiling，指针在生命周期内有效；若用例参数化延迟执行需保证数据存活期覆盖 ExecuteTiling 调用。

## 预防

- 写"tiling 需要读输入数据"的算子 UT 时，逐输入对照一张表：shape、dtype、format、isConst、constValue 五项全部显式给值，不留默认参数。
- UT 首跑前先读一遍 `tiling_case_executor.cpp` 的张量构造循环，确认数据指针从哪来。

## 验证信号

来自该算子迭代二 UT 会话的实测：

- 修复前：3/3 用例 `Tiling failed`（single_tensor_1024 / single_tensor_1000000 / multi_tensor_2 全部 Actual: false）（来源平台：950；证据等级：tool_observed）。
- 修复后：同一套 faker 框架上 24/24 用例通过（含 3 dtype × 6 范数类型 TilingKey 精确断言与多核用例），`[  PASSED  ] 24 tests.`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `constValue` 指针必须指向 `ge::DT_FLOAT` 对应类型的实际内存；传错类型指针 tiling 读到的是位型错乱的值，不会报错只会算错。
- 该方案只覆盖标量/小张量的 host 数据注入；大块输入数据走 UT 桩不在此卡范围。

**未决项**：

- 无。
## 坏实践

- 只给 shape/dtype 就期待 `GetData<float>()` 有值——faker 不会替你造数据，缺 constValue 时数据指针就是 nullptr。
- 把 "Tiling failed" 当成 tiling 算法 bug 去改 tiling 代码——先确认 UT 桩是否把输入数据给足了。

## 误诊

无（读 executor 张量构造循环后直接定位 constValue 缺失）。
