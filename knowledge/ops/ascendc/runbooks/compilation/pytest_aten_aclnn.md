---
type: "Runbook"
title: "白盒pytest走aten内置绕过自定义aclnn"
description: "自定义 AscendC 算子的白盒 pytest 若通过 torch.ops.aten.<op> 调用，在 torch_npu 环境下会分派到 CANN 内置同名算子而非自定义 aclnn，导致自定义 kernel/TilingKey 路径从未被命中、用例全 PASS 却没测到自定义 kernel。给出 aten vs aclnn 调用通路的判别方法。"
tags: ["task.testing", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-17T03:04:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 白盒pytest走aten内置绕过自定义aclnn

## 现象

在 Ascend950（dav-3510 / arch35，DAV_3510，CANN 9.0.0，torch/torch_npu 2.10.0）上为自定义算子 `smooth_l1_loss_grad`（SmoothL1LossGrad，registry-invoke 上库流程，已 build+install 为 `smooth_l1_loss_grad_custom` vendor）生成白盒 pytest（`tests/whitebox/S6_test_smooth_l1_loss_grad.py`）。low 档门禁用例全 PASS，精度双千双万达标。但 Step 6c 自动从 plog 提取的 `S6_tilingkey_coverage.json` 显示期望的 TilingKey（KEY_FP16/KEY_BF16/KEY_FP32）全未命中自定义算子。plog 里出现的 kernel 名是 CANN 内置的 `SmoothL1LossGradV2`，而非自定义的 `SmoothL1LossGrad`。即：白盒用例"全过"了，但过的不是自定义 kernel——测的是 torch_npu 已注册的 aten 内置实现。

触发场景还原：白盒 pytest 的算子调用写成了 `torch.ops.aten.smooth_l1_loss_backward(dout, predict, label, reduction=0, beta=sigma)`（`reduction=0` 对应 `'none'`，`beta` 等价算子属性 `sigma`）。这条路径在 torch_npu 环境下会被 aten dispatch 路由到 GE IR 已注册的内置 `SmoothL1LossGradV2`，**完全绕过** `smooth_l1_loss_grad_custom` vendor 注册的自定义 aclnn 两段式接口（`aclnnSmoothL1LossGradGetWorkspaceSize` + `aclnnSmoothL1LossGrad`）。自定义 kernel 二进制（按 TilingKey 分派的 arch35 `.o`）从未被加载执行。

## 适用场景

- **工程形态**：白盒 pytest（`tests/whitebox/S6_test_<op>.py`）在真实 NPU 上验证自定义 AscendC 算子；属于测试工程/调用通路问题，非 kernel 精度或编译问题。
- **API 路径**：`torch.ops.aten.<builtin_op>` vs `torch.ops.ascend.aclnn<CustomOp>`（自定义算子 aclnn 两段式：`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`，经 `libtorch_adapter.so` 注册到 `torch.ops.ascend` 命名空间）。
- **验证过的平台与 CANN 版本**：Ascend950 / dav-3510 / arch35（DAV_3510），CANN 9.0.0，torch 2.10.0+cpu / torch_npu 2.10.0，davinci0 真实 NPU。自定义算子已 `custom_opp` build+install 到 `~/Ascend/cann/cann-9.0.0/vendors/<op>_custom/`。
- **适用粒度**：`operator_family`——对所有"自定义算子与 CANN 内置同名/同语义算子共存"的 registry-invoke 上库白盒测试均适用（不是本算子特例）。
- **路线优先级**：`recommended_fastpath`——白盒 pytest 必须走自定义算子的 aclnn 接口（经 `libtorch_adapter.so` 注册的 `torch.ops.ascend.aclnn<Op>`），而非 `torch.ops.aten.<builtin>`；后者是 banned 的"假绿"路径。
- **明确不适用或未验证场景**：① 算子在 CANN 内置无同名/同语义实现时（aten 无对应 op，调用必然走自定义路径，本问题不会出现）；② 走 C++ ST（`test_aclnn_<op>.cpp` 直接 `#include` aclnn 头 + 链 `libcust_opapi.so` 调 `aclnn<Op>`）的场景不受影响（C++ ST 本就走 aclnn）；③ 推断可迁移到 a3/a2 等 AscendC 平台但本 trace 未实测——见 `## 失效边界`。

## 判别方法

本卡为 `diagnostic_rule`：给出触发信号 → 可执行探针 → 原始观测 → 机制解释；不依赖未恢复的修复后 after 代码。

**触发信号**：白盒 pytest 源码中出现 `torch.ops.aten.<op>` 调用（即走 aten 内置 dispatch 而非自定义 aclnn），且被测自定义算子与 CANN 内置存在同名/同语义算子。触发信号为 tool_observed——直接读取 pytest 源码即可确认调用通路。

**可执行探针**：

1. 读白盒 pytest 源码，定位算子调用语句——若出现 `torch.ops.aten.<op>` 即命中本判别规则。本 trace 的 before 源码（`S6_test_smooth_l1_loss_grad.py`，修复前读取的完整文件，第 9 行说明 + 第 100-127 行 `test_smooth_l1_loss_grad` 函数）verbatim 如下：

```python
def test_smooth_l1_loss_grad(p):
    if not HAS_NPU:
        pytest.skip("NPU not available")

    tensors = p["tensors"]
    params = p["params"]

    # a. 构造输入 tensor
    inputs = {}
    for name, spec in tensors["inputs"].items():
        if spec is None:
            inputs[name] = None
            continue
        dr = spec.get("_data_range", "normal")
        dtype = DTYPE_MAP[spec["dtype"]]
        shape = spec["shape"]
        cpu_tensor = make_data(shape, dtype, dr)
        inputs[name] = cpu_tensor.npu()

    # b. 提取标量参数
    sigma = float(params.get("sigma", 1.0))

    # c. 调用算子 (NPU) — aten::smooth_l1_loss_backward(grad_output, self, target, reduction, beta)
    #    reduction=0 ('none'), beta=sigma
    gradient_out = torch.ops.aten.smooth_l1_loss_backward(
        inputs["dout"], inputs["predict"], inputs["label"],
        0, sigma
    )
```

（文件头说明同样标注了调用方式：`调用方式：torch.ops.aten.smooth_l1_loss_backward(dout, predict, label, reduction=0, beta=sigma)`，第 9 行。）

2. 跑白盒 pytest 时开启 plog，grep 自定义算子名与 TilingKey：

```bash
ASCEND_CUSTOM_OPP_PATH=~/Ascend/cann/cann-9.0.0/vendors/smooth_l1_loss_grad_custom \
pytest S6_test_smooth_l1_loss_grad.py --cases-file=S5_mapped_cases_low.json -v -s
# 跑完后在 tilingkey_logs/ 下找 plog-*.log
grep -i "tilingKey\|SmoothL1\|smooth_l1" tilingkey_logs/plog-*.log
```

若 grep 结果只出现内置 `SmoothL1LossGradV2`、无自定义 `SmoothL1LossGrad`、且无 `tilingKey=` 命中自定义 key，即确诊。

**原始观测**：白盒 pytest 源码 `S6_test_smooth_l1_loss_grad.py` 第 124-127 行确实调用 `torch.ops.aten.smooth_l1_loss_backward(...)`（artifact_verified，文件完整读取）。对照同仓参考算子 `relu6_d` 的白盒 pytest `S6_test_relu6_d.py`——它通过 `tests/st/torch/build/libtorch_adapter.so` 注册的 `torch.ops.ascend.aclnnRelu6D` 调用自定义 kernel，文件头说明与调用语句 verbatim 如下（第 1-6 行说明 + 第 180-185 行调用）：

```python
# S6_test_relu6_d.py — Relu6D 白盒 pytest（真实 NPU 执行）
#
# 通路说明：relu6_d 对外走 GE IR，aclnn 为 aclnn_inner（内部 UT/ST 精度验收），
# torch_npu 无 npu_relu6_d 绑定。本测试通过 tests/st/torch/build/libtorch_adapter.so
# 注册的 torch.ops.ascend.aclnnRelu6D 在真实 NPU（Ascend950 / dav-3510 / arch35，
# davinci0）上调用自定义 kernel（conftest.py 负责 load_library + ASCEND_CUSTOM_OPP_PATH）。

    # c. NPU 调用（真实 kernel，预分配输出）
    x_npu = x_cpu.npu()
    out_npu = torch.empty_like(x_npu)
    torch.ops.ascend.aclnnRelu6D(x_npu, scale, out_npu)
    torch.npu.synchronize()
    npu_out = out_npu.cpu()
```

两相对照：relu6_d 走 `torch.ops.ascend.aclnn<Op>`（自定义），smooth_l1_loss_grad 错走 `torch.ops.aten.<builtin>`（内置）——这就是自定义 kernel 路径未被命中的机制层证据。

**解释**：`torch.ops.aten.<op>` 是 torch_npu 的 aten dispatch 入口。当自定义算子与 CANN 内置算子**同语义同名**（本例 `smooth_l1_loss_backward` ↔ `SmoothL1LossGradV2`）时，aten 路径优先路由到 GE IR 已注册的内置实现 `SmoothL1LossGradV2`，**不会**落到 `smooth_l1_loss_grad_custom` vendor 注册的自定义 aclnn 两段式接口。自定义 kernel 二进制（按 TilingKey 分派的 arch35 `.o`）只在 aclnn 接口被调用时才由 runtime 按 TilingKey 选中加载。aten 路径完全不经自定义 vendor，故 plog 无自定义 kernel 名、TilingKey 恒全未命中。pytest 全 PASS 是因为内置 `SmoothL1LossGradV2` 本身就是正确实现——这掩盖了"没测到自定义 kernel"的事实，形成"假绿"。

机制链（不跳步）：aten dispatch → GE IR 查找 `SmoothL1LossGradV2`（内置，已注册）→ 调内置 kernel → PASS；自定义 vendor 的 aclnn `aclnnSmoothL1LossGrad(GetWorkspaceSize)` 从未被调用 → runtime 不查 `<op>_custom` 的 TilingKey→kernel 映射 → 自定义 TilingKey 全未命中。

## 验证信号

- 白盒 pytest `S6_test_smooth_l1_loss_grad.py` 第 124-127 行调用 `torch.ops.aten.smooth_l1_loss_backward(inputs["dout"], inputs["predict"], inputs["label"], 0, sigma)`，即走 aten 内置 dispatch 而非自定义 aclnn（来源平台：950；证据等级：artifact_verified）。
- 参考算子 `relu6_d` 白盒 pytest `S6_test_relu6_d.py` 第 183 行调用 `torch.ops.ascend.aclnnRelu6D(x_npu, scale, out_npu)`，经 `tests/st/torch/build/libtorch_adapter.so` 注册到 `torch.ops.ascend` 命名空间，命中自定义 kernel；文件头第 4-6 行明确说明此通路（来源平台：950；证据等级：artifact_verified）。
- 自定义算子已安装为 `smooth_l1_loss_grad_custom` vendor，`libcust_opapi.so` 存在于 `~/Ascend/cann/cann-9.0.0/vendors/smooth_l1_loss_grad_custom/op_api/lib/`（来源平台：950；证据等级：tool_observed）。

## 性能影响

无性能故事（本卡为测试调用通路判别，不涉及 kernel 性能）。

## 失效边界

- 本 trace 仅在 Ascend950/arch35 实测。a3、a2 等 AscendC 平台推断可迁移（aten dispatch 与 aclnn vendor 注册机制是 CANN 通用行为，不依赖 arch35 特性），但**未验证，推断可迁移**。
- 仅当"自定义算子与 CANN 内置同语义同名算子共存"时本问题才出现。若算子是全新增、aten 无同名入口，则 `torch.ops.aten.<op>` 调用根本不成立，本判别规则不适用。
- 本卡只判别"走 aten 内置"这一根因；修复侧（建 `torch_adapter.cpp` + `libtorch_adapter.so` 注册 `torch.ops.ascend.aclnn<Op>`、改 pytest 调用）的完整 verbatim 代码未在本 trace 归一化事件流中恢复，不作为本卡配方给出。

## 坏实践

- 白盒 pytest 图省事直接调 `torch.ops.aten.<op>`（与 PyTorch CPU 侧写法一致），不构建 torch 适配层——结果用例全 PASS 却没测到自定义 kernel，TilingKey/路径覆盖声明全部失效。这种"假绿"比报错更危险：它会通过 workflow validator 的精度对账（因为内置算子精度也对），却把自定义 kernel 的 TilingKey 分派、尾块/空 Tensor 短路、cast-compute 路径等全部漏测。

## 误诊

- 主 agent 口述"修复后 TilingKey 从全未命中变为全部命中（fp32/fp16/bf16 均命中），用例全 PASS"。该结论为 `agent_reported`：修复后 pytest 输出与 plog grep 结果被 markdown_transcript 适配器截断（`_see below_`），归一化事件流中未保留 tool_observed 原文，故不进 `## 验证信号`。下个 agent 若要据此确认修复有效，需在真实 NPU 上重跑白盒 pytest + plog grep 复现。
