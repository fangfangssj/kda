---
type: "Runbook"
title: "optiling 内调 TilingFunc 须引 AscendC 空间"
description: "SoftMax/TopK 高阶 API 的 host 侧 tiling 辅助（SoftMaxTilingFunc/TopKTilingFunc/TopKMode/GetTopKMaxMinTmpSize）全部声明在 namespace AscendC 内；namespace optiling 里裸调报 not declared in this scope（note 指向 AscendC::），在 optiling 开头补 using namespace AscendC; 解决。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.api_layer.high_level_api"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# optiling 内调 TilingFunc 须引 AscendC 空间

## 现象

接续同工程 tiling 头 include 路径修复（见 [高阶 API tiling 头传递依赖卡](highlevel_api_tiling_header_adv_api_include.md)）之后重建，Softmax+TopK 融合算子（Ascend950PR，CANN 9.1.0，registry-invoke 工程）的 `op_host/arch35/<op>_tiling.cpp` 已能找到 `softmax/softmax_tiling.h` 与 `topk/topk_tiling.h`，但 `bash build.sh --soc=ascend950` 仍在 host 编译阶段报一批"未声明"错误：

```text
op_host/arch35/<op>_tiling.cpp:215:5:
  error: 'SoftMaxTilingFunc' was not declared in this scope; did you mean 'AscendC::SoftMaxTilingFunc'?
op_host/arch35/<op>_tiling.cpp:220:40:
  error: 'TopKMode' has not been declared
op_host/arch35/<op>_tiling.cpp:223:5:
  error: 'TopKTilingFunc' was not declared in this scope; did you mean 'AscendC::TopKTilingFunc'?
```

编译器 note 同时给出了正确位置：`asc/include/tiling/softmax/softmax_tiling.h:54:6: note: 'AscendC::SoftMaxTilingFunc' declared here`。当时代码在 `namespace optiling` 里直接裸调这些函数，并且已经用 `using Ops::Base::CeilDiv;` 引入了工具函数——只引入了 Ops 命名空间，漏了 AscendC。

## 适用场景

- 工程形态：registry-invoke 自定义算子包的 op_host tiling 实现（`IMPL_OP_OPTILING` 注册的 TilingFunc 内）。
- API 路径：`AscendC::SoftMaxTilingFunc` / `AscendC::TopKTilingFunc` / `AscendC::GetTopKMaxMinTmpSize` / `AscendC::GetSoftMaxMaxMinTmpSize` / `AscendC::TopKMode`（高阶 API 的 host 侧 tiling 辅助，声明于 `asc/include/tiling/{softmax,topk}/*.h` 的 `namespace AscendC`）。
- 验证过的平台与 CANN 版本：Ascend950PR（soc=ascend950），CANN 9.1.0。
- 适用粒度：platform_wide——所有在 `namespace optiling` 中调用高阶 API TilingFunc 的算子工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 侧（device 编译单元本就在 AscendC 语境）；host 侧若工程选择全程 `AscendC::` 前缀写法则不需要 using 声明（同仓 TopK 工程即用前缀写法）。

## 根因

SoftMax/TopK 高阶 API 的 host 侧 tiling 辅助函数（TilingFunc、MaxMinTmpSize、TopKMode 枚举）全部声明在 `namespace AscendC` 内；自定义算子的 tiling 实现写在 `namespace optiling` 中，两个命名空间之间没有自动引入，裸调即报 not declared。

## 解决配方

在 tiling 实现文件的 `namespace optiling {` 开头补 `using namespace AscendC;`。

before（首次写入的 `op_host/arch35/<op>_tiling.cpp` 命名空间开头）：

```cpp
namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
```

after：

```cpp
namespace optiling {

using namespace AscendC;
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
```

调用处无需改动（修复后以下裸调即可解析到 `AscendC::` 下声明）：

```cpp
    GetTopKMaxMinTmpSize(ascendcPlatform, static_cast<int32_t>(EAlign), 1,
                         false, false, TopKMode::TOPK_NORMAL, true, computeDtypeSize,
                         topkMaxTmp, topkMinTmp);
    ...
    SoftMaxTilingFunc(srcShape, computeDtypeSize, softmaxMinTmp, softmaxTiling);
    ...
    TopKTilingFunc(ascendcPlatform, static_cast<int32_t>(EAlign),
                   static_cast<int32_t>(batchRows), static_cast<int32_t>(k),
                   computeDtypeSize, false, TopKMode::TOPK_NORMAL, true, topkTiling);
```

## 设计决策

- 选 `using namespace AscendC;`（置于 optiling 内）而不是逐个 `using AscendC::SoftMaxTilingFunc;`：tiling 代码要调用的 AscendC 符号多（两个 TilingFunc、两个 MaxMinTmpSize、TopKMode、SoftMaxTiling/TopkTiling 结构），逐个引入冗长；optiling 是算子实现专属命名空间，与 AscendC 冲突风险低。
- 备选"全部加 `AscendC::` 前缀"被放弃：改动点多、可读性差，且与模板参数结构体 `AscendC::tiling::SoftMaxTiling` 混用容易漏。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 在 optiling 里用高阶 API 的 tiling 辅助时，模板固定带 `using namespace AscendC;`；报错文本里 `did you mean 'AscendC::xxx'` 即是命名空间问题的直接提示，优先按 note 修而不是怀疑头文件没包含对。

## 验证信号

来源说明：命名空间修复前后连续两次构建（Ascend950PR + CANN 9.1.0）。

- 修复前：`'SoftMaxTilingFunc' was not declared in this scope`、`'TopKMode' has not been declared`、`'TopKTilingFunc' was not declared in this scope` 等错误（来源平台：950；证据等级：tool_observed）。
- 修复后：host 侧三个目标（cust_optiling/cust_opapi/cust_op_proto 相关编译单元）编译通过，构建推进到 opbuild 阶段（报下一阶段的 `The dtype size of input[0] ... is 0`，属另一根因），"未声明"错误消失（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `using namespace AscendC;` 放在 optiling 内是局部引入；若工程在头文件或全局作用域这样写可能污染其他翻译单元（未验证，不建议）。
- CANN 9.1.0 下 SoftMax/TopK tiling 辅助的命名空间为 AscendC；其他 API 族（如 Reduce）的归属命名空间未逐一验证，报错时仍以编译器 note 为准。

**未决项**：

- 无（该根因已闭环到编译通过；后续 opbuild 阶段错误见同批其他卡）。
## 坏实践

- 只引入 `Ops::Base::` 工具函数就以为 tiling 头"已可用"——头文件解析成功（include 路径修复）和符号可解析（命名空间引入）是两个独立前提，需分别满足。

## 误诊

- 无。
