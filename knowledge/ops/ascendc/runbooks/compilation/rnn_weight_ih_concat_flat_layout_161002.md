---
type: "Runbook"
title: "多层 weight_ih 须 flat 无补零拼接防 161002"
description: ST 测试把多层/双向 weight_ih 拼成 (rows, maxInputDim) 2D 补零布局时，元素总数多于 aclnn CheckShape 的 flat 变长 K 校验与 kernel GetWeightIhOffset 寻址期望，GetWorkspaceSize 返回 161002；改 1D flat 无 padding 拼接后多层与非对齐 inputSize 用例全部通过。
tags: [task.testing, topic.compilation, topic.compute.indexing]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# 多层 weight_ih 须 flat 无补零拼接防 161002

## 现象

在 GRU 算子（多层/双向扩展，Ascend950PR / arch35，CANN 9.2.0）的 Real 模式联调中，两类用例在 `aclnnBenchNoHarnessGruGetWorkspaceSize` 阶段即失败，报错原文 `[FAIL] GetWorkspaceSize 失败: 161002`（ACLNN_ERR_PARAM_INVALID）：

- 多层单向用例（2 层 / 3 层，inputSize=128，hiddenSize=64）：layer 0 输入维 128、layer 1+ 输入维 numDirs×H=64；
- 非对齐 inputSize=63 的单层用例（hiddenSize=64）。

而 inputSize=127 的单层用例 PASS——127 与 64 取 max 仍是 127，不产生补齐差；63 与 64 取 max 变 64，产生补齐差。失败模式与"拼接维度是否被补齐"精确相关。

## 适用场景

- 工程形态：aclnn 自定义算子 ST 测试工程（C++），为单 Tensor 接口拼装多层/多方向 TensorList 权重；aclnn L2 参数校验。
- API 路径：aclnn `GetWorkspaceSize` → `CheckShape`（权重元素总数校验）；kernel `GetWeightIhOffset`（flat 变长 K 布局寻址）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——多层/双向 RNN 类算子（GRU/LSTM/RNN）把逐层权重拼成单 Tensor 传递的测试侧与接口侧布局契约。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：接口直接接收 TensorList（无需拼接）的算子；权重布局契约不同的实现（须以各自 kernel 寻址公式为准）。

## 根因

ST 测试的 `ConcatenateWeightIh` 把各层/方向权重拼成 `(numLayers*numDirs*3H, maxInputDim)` 的 2D 张量，列数不足 maxInputDim 的层（layer 0 的 inputSize < numDirs×H，或 layer k>0 的 numDirs×H < inputSize）补零；而 aclnn `CheckShape` 按 flat 元素总数校验（layer 0：numDirs×3H×inputSize；layer k>0：numDirs×3H×(numDirs×H)），kernel `GetWeightIhOffset` 也按变长 K 的 1D flat 布局寻址（`wGm[wOff + n*K + k]`，K 为该层实际输入维）。补零后元素总数多于期望值，校验拒绝并返回 161002。

## 解决配方

修复前（`tests/st/test_aclnn_<op>.cpp`，2D maxInputDim 补齐布局，verbatim）：

```cpp
static void ConcatenateWeightIh(
    const std::vector<GruTensor>& tensors,
    int64_t hiddenSize, int64_t numLayers, int numDirs, int64_t inputSize,
    std::vector<float>& concatData, std::vector<int64_t>& concatShape) {

    int64_t gateSize = 3 * hiddenSize;
    int64_t totalRows = numLayers * numDirs * gateSize;
    int64_t maxInputDim = std::max(inputSize, (int64_t)(numDirs * hiddenSize));

    concatData.resize(totalRows * maxInputDim, 0.0f);

    for (int64_t layer = 0; layer < numLayers; ++layer) {
        int64_t layerInputDim = (layer == 0) ? inputSize : (int64_t)(numDirs * hiddenSize);
        for (int dir = 0; dir < numDirs; ++dir) {
            int64_t idx = layer * numDirs + dir;
            int64_t rowOffset = idx * gateSize;

            // 拷贝实际权重数据（列数 = layerInputDim），剩余列保持 0
            for (int64_t r = 0; r < gateSize; ++r) {
                for (int64_t c = 0; c < layerInputDim; ++c) {
                    concatData[(rowOffset + r) * maxInputDim + c] =
                        tensors[idx].data[r * layerInputDim + c];
                }
            }
        }
    }

    concatShape = {totalRows, maxInputDim};
}
```

修复后（1D flat 变长 K 拼接，无 padding，与 kernel `GetWeightIhOffset` 布局一致）：

```cpp
static void ConcatenateWeightIh(
    const std::vector<GruTensor>& tensors,
    int64_t hiddenSize, int64_t numLayers, int numDirs, int64_t inputSize,
    std::vector<float>& concatData, std::vector<int64_t>& concatShape) {

    int64_t gateSize = 3 * hiddenSize;

    // 1D flat packing (no padding): matches kernel's GetWeightIhOffset
    // Layer 0: (3H, inputSize) per direction
    // Layer k>0: (3H, numDirs*H) per direction
    // Total elements = numDirs*gateSize*inputSize + (numLayers-1)*numDirs*gateSize*(numDirs*H)
    int64_t totalElements = 0;
    for (int64_t layer = 0; layer < numLayers; ++layer) {
        int64_t layerInputDim = (layer == 0) ? inputSize : (int64_t)(numDirs * hiddenSize);
        totalElements += numDirs * gateSize * layerInputDim;
    }

    concatData.resize(totalElements, 0.0f);

    int64_t offset = 0;
    for (int64_t layer = 0; layer < numLayers; ++layer) {
        int64_t layerInputDim = (layer == 0) ? inputSize : (int64_t)(numDirs * hiddenSize);
        for (int dir = 0; dir < numDirs; ++dir) {
            int64_t idx = layer * numDirs + dir;
            int64_t layerElements = gateSize * layerInputDim;
            for (int64_t i = 0; i < layerElements; ++i) {
                concatData[offset + i] = tensors[idx].data[i];
            }
            offset += layerElements;
        }
    }

    concatShape = {totalElements};
}
```

（weight_hh 各层恒为 (3H, H)，stride 一致，无需改动。）

## 设计决策

- 为什么选 flat 变长布局而不是统一 padded 布局：aclnn CheckShape 与 kernel 寻址已按"逐层 K=layer 0 输入维 / numDirs×H"的 flat 语义实现并经 UT 固化，测试侧向接口契约对齐成本最低；padded 布局需同时改 kernel 寻址与校验公式，收益仅是形状规整。
- 拼接 shape 用 1D `{totalElements}`：校验按元素总数，1D 形状最不易与逐层 2D 形状产生二义。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 测试侧为单 Tensor 接口拼装 TensorList 数据时，先读 kernel 的 offset 寻址公式（本例 `GetWeightIhOffset`），按其布局契约拼接，不按"取 max 补齐"直觉写。
- 多层用例矩阵里保留"inputSize < numDirs×H"与"inputSize > numDirs×H"两个方向的最小用例（本例 63 与 127/128），任何一侧被补齐都会触发本问题。
- 161002 出现在 GetWorkspaceSize 阶段且用例参数仅维度组合不同时，优先对照元素总数校验公式，而不是怀疑 kernel。

## 验证信号

来源说明：该 GRU 算子联调验证会话（Ascend950PR，CANN 9.2.0）；定位拼接布局与校验公式不符后改为 flat 拼接，重编译上板复测。

- 修复前：多层单向（2/3 层）与 inputSize=63 用例 `GetWorkspaceSize 失败: 161002`；inputSize=127 用例 PASS（无补齐差）（来源平台：950；证据等级：tool_observed）。
- 修复后：inputSize=63 用例 PASS（y MERE=3.03e-06、MARE=3.49e-03）；inputSize=127 用例 PASS（MERE=2.29e-06）（来源平台：950；证据等级：tool_observed）。
- 修复后：2 层 GRU PASS（MERE=5.35e-06）、3 层 GRU PASS（MERE=1.54e-05）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡布局契约（layer 0 K=inputSize、layer k>0 K=numDirs×H、方向内连续、层间按方向递进）以该算子实现定义为准；其他算子的 GetWeightIhOffset 公式不同时须按其公式重推。
- 校验公式若未来放宽为"元素数 ≥ 期望"（允许 padding），本失败形态消失但 kernel 读 padded 数据仍会算错——布局契约本身不能松。
- 其他平台未验证，机制推断可迁移（未验证，推断可迁移）。

**未决项**：

- 无（根因已追到拼接布局与校验/寻址契约不一致机制层并闭环验证）。
## 坏实践

- 用 `(rows, maxInputDim)` 统一列宽拼装变长 K 数据——既通不过元素总数校验，即使放过校验也会让 kernel 按错误 K 读权重。
- 把 161002 当作"不支持该 shape"直接跳过用例，不对照校验公式排查。

## 误诊

- 无（inputSize=127 与 63 的对照直接把失败面收敛到补齐差）。
