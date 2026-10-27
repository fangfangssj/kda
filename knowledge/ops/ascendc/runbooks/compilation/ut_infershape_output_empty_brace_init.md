---
type: "Runbook"
title: "UT InferShape 空占位聚合初始化编译失败"
description: "op_host UT 里给 InferShape 输出占位写 {{}, {}} 聚合初始化，编译报 could not convert ... to std::vector<gert::InfershapeContextPara::TensorDescription>；改为默认构造的命名 gert::StorageShape 对象后编译通过、用例全过。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# UT InferShape 空占位聚合初始化编译失败

## 现象

在 x86_64 主机上为 Ascend950PR 自定义算子（CANN 9.1.0）编写 op_host InferShape 单元测试，用 faker 框架构造 `gert::InfershapeContextPara`。输入 tensor 的 shape 用命名 `gert::StorageShape` 对象构造都正常，输出 shape 按"待推导"语义想给一个空占位，写成内联花括号：

```cpp
// Output shape is empty (to be inferred)
std::vector<InfershapeContextPara::TensorDescription> outputDesc = {
    {{}, {}}, // empty shape, will be set by InferShape
};
```

编译 `cmake --build . --target <op>_op_host_ut` 直接报错：

```
test_bench_no_harness_roi_align_infershape.cpp:63:5: error: could not convert '{{<brace-enclosed initializer list>(), <brace-enclosed initializer list>()}}' from '<brace-enclosed initializer list>' to 'std::vector<gert::InfershapeContextPara::TensorDescription>'
```

## 适用场景

- 工程形态：AscendC 自定义算子 op_host UT（infershape_context_faker / `InfershapeContextPara::TensorDescription`）。
- API 路径：`gert::StorageShape`、`gert::InfershapeContextPara`（`tests/ut/common/infershape_context_faker.h` 链路）。
- 验证过的平台与 CANN 版本：Ascend950PR（UT 目标 SOC_VERSION=ascend950，x86_64 主机执行），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 faker API 用法与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：TilingContextPara（TilingCasePara）的 shape 构造未测同款写法；其他 CANN 版本的 TensorDescription 构造签名未验证。

## 根因

`InfershapeContextPara::TensorDescription` 需要一个真实的 `gert::StorageShape` 对象作成员，`{{}, {}}` 双层空花括号无法聚合转换成 `StorageShape`（空 brace-init list 到该类型不可转换），整个 vector 初始化列表因此编译失败。

注意：模板 add_example 的测试里出现过 `{{{ {}, {} }}, ge::DT_FLOAT16, ge::FORMAT_ND}` 形态的内联空 shape——照抄到本工程的 TensorDescription 上即触发上述编译错，不能直接搬。

## 解决配方

before：

```cpp
    gert::StorageShape xShape({B, C, H, W}, {B, C, H, W});
    gert::StorageShape boxesShape({numBoxes, 5}, {numBoxes, 5});

    std::vector<InfershapeContextPara::TensorDescription> inputDesc = {
        {xShape, dtype, ge::FORMAT_ND},
        {boxesShape, dtype, ge::FORMAT_ND}
    };
    // Output shape is empty (to be inferred)
    std::vector<InfershapeContextPara::TensorDescription> outputDesc = {
        {{}, {}}, // empty shape, will be set by InferShape
    };
```

after（先默认构造命名对象，再传入 TensorDescription）：

```cpp
    gert::StorageShape xShape({B, C, H, W}, {B, C, H, W});
    gert::StorageShape boxesShape({numBoxes, 5}, {numBoxes, 5});
    gert::StorageShape yShape;  // empty shape, will be set by InferShape

    std::vector<InfershapeContextPara::TensorDescription> inputDesc = {
        {xShape, dtype, ge::FORMAT_ND},
        {boxesShape, dtype, ge::FORMAT_ND}
    };
    std::vector<InfershapeContextPara::TensorDescription> outputDesc = {
        {yShape, dtype, ge::FORMAT_ND}
    };
```

## 设计决策

- 选"默认构造命名对象"而不是"给输出塞一个哑 shape 再断言被覆盖"：`gert::StorageShape yShape;` 语义就是空 shape，InferShape 执行后由 faker 回填，断言直接读回填结果即可。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 在 faker 测试里构造输出占位时，一律先声明命名的空 `gert::StorageShape`，不要写内联空花括号。
- 从 add_example 抄 UT 构造代码时，注意其 `InfershapeContextPara` 构造形态与本工程 wrapper 的 `TensorDescription` 形态不同，逐字段核对。

## 验证信号

以下信号来自该算子迭代一 UT 会话（x86_64 主机，SOC_VERSION=ascend950，CANN 9.1.0）：

- 修复前：`test_<op>_infershape.cpp:63:5: error: could not convert '{{{}, {}}}' ... to 'std::vector<gert::InfershapeContextPara::TensorDescription>'`（来源平台：950；证据等级：tool_observed）。
- 修复后重编：无 error/warning，`[100%] Built target bench_no_harness_roi_align_op_host_ut`（来源平台：950；证据等级：tool_observed）。
- 运行 `--gtest_filter="*Infershape*"`：4/4 通过（EXIT CODE: 0），全量 op_host 15/15 PASSED（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `gert::StorageShape` 默认构造可用性在 CANN 9.1.0 实测；若后续版本删除默认构造函数，此写法需改为显式 `StorageShape({}, {})`（未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- 照抄 add_example UT 里 `{{{ {}, {} }}, ...}` 的内联空 shape 形态到自研 wrapper。

## 误诊

- 无。
