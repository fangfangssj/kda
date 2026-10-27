---
type: "Runbook"
title: "Select CMPMASK_SPR 模式致混合分支误选"
description: "arch35(dav_c310) 高层 Select 用 VSEL_CMPMASK_SPR 时 set_cmpmask 不随 repeat 偏移 mask 指针，count 超过一个 repeat 的 64 个 fp32 元素时每 repeat 复用首 64bit mask 致混合分支误选；改 VSEL_TENSOR_TENSOR_MODE 后失败数由 116 降至 3 判明根因。"
tags: ["task.debugging", "topic.compute.elementwise.comparison_logic", "topic.implementation.api_layer.basic_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T08:28:51Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Select CMPMASK_SPR 模式致混合分支误选

## 现象

在 Ascend950PR（`__NPU_ARCH__==3510` → dispatcher 选中 `dav_c310` 变体，CANN cann-9.0.0-beta.2）上调试 `softplus_v2_grad` 反向算子 PyTorch ST。基线 `bash build.sh --soc=ascend950 -j8` 编译安装后跑 `python3 test.py --lib ./build/libtorch_adapter.so`，出现批量失败，形态集中在 L1 大 shape 用例：对应当取线性分支（`z > threshold`，应返回 grad_output）的元素，NPU 返回 `0.5*grad_output`（即 `sigmoid(0)*grad_output`），反之亦然。失败沿 tile 内 dim3 维度呈结构化分布（如 d3=8 仅 element0 错、d3=9 全错、d3=10 element1-8 错），形如 mask 数据在 repeat 间未偏移的"移位/复用"特征。

## 适用场景

- 工程形态：ops-registry-invoke（算子仓开发），kernel 直调 + aclnn L2。
- API 路径：`AscendC::Select<float, uint8_t>(dst, maskUb, src0, src1, SELMODE, count)` 高层向量选择 API；`AscendC::Compare` 生成 mask。
- 验证过的平台与 CANN 版本：Ascend950PR（`__NPU_ARCH__==3510` → `dav_c310`），CANN cann-9.0.0-beta.2。
- 适用粒度：`operator_family`（所有在 dav_c310/arch35 上用 `Select(...,count)` 高层 API 处理 count 超过一个 repeat 的混合分支的算子）。
- 不适用/未验证场景：dav_c100 / dav_c220 / dav_m200 等其他 `__NPU_ARCH__` 变体未验证。

## 判别方法

**trigger（触发）**：PyTorch ST 批量失败，失败形态为混合分支误选——大 shape 用例中本应取线性分支的元素返回了 sigmoid 分支值（`0.5*grad_output`），或反之；失败沿 dim3 呈结构化"移位/复用"分布。基线 `VSEL_CMPMASK_SPR` 模式下 116/578 失败（精确计数见 `## 验证信号`）。

**probe（可执行探针）**：将 kernel `ComputeFp32Core` 中 `Select` 调用的 SELMODE 常量从 `VSEL_CMPMASK_SPR` 改为 `VSEL_TENSOR_TENSOR_MODE`（仅 1 行常量替换），撤销此前任何手动分段，重跑 PyTorch ST：

```
python3 test.py --lib ./build/libtorch_adapter.so
```

若失败数从约 116 骤降至个位数（与本根因无关的极值 FTZ 残留），则判明根因为 CMPMASK_SPR 路径的 mask 偏移缺陷。

**observation（工具观测）**：改 MODE 后 PyTorch ST 失败数 116 降至 3（575/578 通过），C++ ST 57/57 通过，L0 功能用例 51/51 通过——模式替换直接消除了结构化混合分支误选，证实根因在 SELMODE 而非计算公式或数据布局。

**interpretation（根因解释）**：dav_c310 高层 `Select(...,count)` 在 `VSEL_CMPMASK_SPR` 路径中，`VselIntrinsicsImpl` 调用 `set_cmpmask(sel)` 把整段 mask 装入 CMPMASK SPR，但分块循环对 `sel`（mask 指针）不做偏移；因此当 count 超过一个 repeat 的 fp32 元素数（`ONE_REPEAT_ELE_NUM_FP32=64`）时，每个 repeat 复用首 64-bit mask，混合分支（部分 `z>thr`、部分 `z<=thr`）误选。改用 `VSEL_TENSOR_TENSOR_MODE` 后，mask 作为 tensor 参数传入，`VselImpl` 每 repeat 用 `sel + i*selOffset` 正确偏移，对任意 count 正确。注意变体查证：Ascend950PR 即 `__NPU_ARCH__==3510` 经 dispatcher 映射到 `dav_c310/`（支持 mode 2），而非 `dav_c100/`；误读变体会得到"arch35 仅支持 CMPMASK_SPR"的错误前提。

## 验证信号

- 基线（VSEL_CMPMASK_SPR）：PyTorch ST 578 用例 116 失败，混合分支误选（来源平台：950；证据等级：tool_observed）
- 改 MODE 为 VSEL_TENSOR_TENSOR_MODE（仅改 1 行常量）：PyTorch ST 575/578 通过（来源平台：950；证据等级：tool_observed）
- C++ ST（`bash run.sh --real`）：57/57 通过（来源平台：950；证据等级：tool_observed）
- L0 功能用例（`python3 test.py --level L0`）：51/51 通过（来源平台：950；证据等级：tool_observed）

## 失效边界

- dav_c100 / dav_c220 / dav_m200 等其他变体未验证；改 MODE 前须先确认目标变体经 dispatcher 映射且支持 `VSEL_TENSOR_TENSOR_MODE`（mode 2），不支持时改 MODE 会 assert false。
- 剩余 3 个失败（极值 `grad=+-FLT_MAX`、`z` 落次正规 exp 区间）与本根因无关，属硬件 FTZ，另行声明。

## 预防

- 开发期对照 `DESIGN.md` 检查"实现↔设计"的 SELMODE 偏差：DESIGN 指定 `VSEL_TENSOR_TENSOR_MODE` 而实现用 `VSEL_CMPMASK_SPR` 即高优先级假设。
- 在 dav_c310/arch35 上优先用 `VSEL_TENSOR_TENSOR_MODE`（mask 作为 tensor 参数，每 repeat 自动偏移）；仅当变体不支持 mode 2 时才退回 CMPMASK_SPR 并手动按 `ONE_REPEAT_ELE_NUM_FP32`(=64) 分段 Compare+Select。