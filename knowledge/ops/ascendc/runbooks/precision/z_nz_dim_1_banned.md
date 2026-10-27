---
type: "Runbook"
title: "输出 z NZ 物理布局致 dim=1 间歇错位 banned"
description: "自定义 kernel 以 flat 1D 行序写入 torch::empty(sizes,input.options()) 分配的输出 z，当 NPU 分配器选 FRACTAL_NZ 物理布局时 PyTorch 按 NZ 读回致特定行整行错位；空张量上 _npu_format_cast 仅改元数据不重排物理内存，作为修复手段 banned。"
tags: ["task.debugging", "topic.data_movement.layout_transform", "topic.implementation.operator_registration", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T08:39:16Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["ACL_FORMAT_FRACTAL_NZ"]
---

# 输出 z NZ 物理布局致 dim=1 间歇错位 banned

## 现象

`swi_glu` 算子（`output = SiLU(x0) * x1`，其中 `(x0,x1)=chunk(input,2,dim)`）在 cann-bench 评测框架下完成 kernel 与 op_plugin 后，精度复现：

- 小 shape（8~64 元素、4x64 等）、大 shape **FP16/BF16**、以及 FP32 的 `dim=0`/`dim=-1` 多数 PASS。
- **唯独大 shape FP32、`dim=1`、2D 输入**（如 `(1024,2048)`、`(8192,4096)`）间歇性出现巨大 `max_diff`，同 shape 连续重跑 5 次：前 1~2 次 `max_diff=3.5~39`（FAIL），后 3~4 次 `max_diff=0.00000143`（PASS）。失败率约 2%。
- 失败用例的错误模式很稳定：**特定行整行错、其余全对**，如 `(512,2048) dim=1` 下 `First error at flat idx=196672, row=192, col=64`，`Bad rows: [192]`，`Total bad elements: 64 out of 524288`。

环境：CANN 9.0.0、torch 2.10.0 + torch_npu、ascend910b（A2）。

## 适用场景

- **工程形态**：op_plugin 层用 `torch::empty(sizes, input.options())` 为自定义 kernel 分配输出 `z`、kernel 以 flat 1D 行序写入 `z` 的 GM。
- **API 路径**：`torch::empty` / `torch::empty_like` / `at_npu::native::custom_ops::_npu_format_cast` / `aclrtSynchronizeStream` / `aclrtSynchronizeDevice`。
- **验证过的平台与 CANN 版本**：a2 / CANN 9.0.0。a3、950 未验证。
- **适用粒度**：`platform_wide` —— 任何在 A2 上以 flat 1D 写法集成自定义 kernel 的 op_plugin 都可能触发，与具体算子无关。
- **路线优先级**：`banned` —— 把"在空张量上调用 `_npu_format_cast(z, NCHW)` 作为修复"判为 banned 路线（对已分配张量仅改元数据、不重排物理内存，无法对冲 NZ 物理布局）。本卡未给已验证的稳定修复，落地时按本卡探针自行复现后再采用候选缓解。
- **明确不适用或未验证场景**：FP16/BF16 大 shape、FP32 dim=0/dim=-1、1D 输入在本 trace 内未复现此失败；a3/950 未验证。

## 失败路线

尝试：在 op_plugin 层用 `torch::empty(sizes, input.options())` 分配输出 `z`、kernel 以 flat 1D 行序写入 `z`，期望 NPU 分配器给出与 kernel 行序一致的 NCHW/ND 物理布局。以下 6 条修复尝试在本 trace 内**均未稳定消除大 shape FP32 dim=1 的间歇失败**（均为实测，来源平台：a2；证据等级: tool_observed）：

- `z = torch::empty_like(x0)`（期望继承 x0 的 ND 格式）—— 仍间歇失败，新张量仍可能被分配器选为 NZ。
- `z = at_npu::native::custom_ops::_npu_format_cast(z, ACL_FORMAT_NCHW)`（kernel 前对 z 做格式转换）—— **空张量上 no-op，data_ptr 不变，无效**。此路线 banned。
- `aclrtSynchronizeStream(getCurrentNPUStream())` 在 `.contiguous()` 之后、kernel 之前 —— 缓解（失败率下降）但未归零；`aclrtSynchronizeDevice()` 同样不能完全消除。
- `aclrtSynchronizeStream` 在 kernel 之后 —— 不解决（问题不在 kernel 完成度，在物理布局错位）。
- `x0 = chunks[0].clone()` / `input.contiguous()` 先行 —— 新张量仍可能被分配为 NZ，无效。
- 绕过 `OpCommand::RunOpApi`、直接 `aclrtLaunchKernel` 到 `getCurrentNPUStream()` —— 失败率从约 2% 飙升至约 77%（OpCommand 提供的 stream 编排部分掩盖了 bug，但非根治）。

客观失败：6 条尝试无一稳定消除间歇失败；会话在调试中结束，未确认最终稳定修复。

边界与回退：把"空张量上 `_npu_format_cast` 作为物理布局修复"判为 banned 路线；回退到"不信任分配器给出的物理布局、由调用方显式强制 NCHW 物理内存或让 kernel 适配 NZ 布局"——但本 trace 内未找到稳定强制 NCHW 物理布局的 API，须继续。

根因（机制链，不跳步）：kernel 把 `z` 当作 flat 1D 行序数组写入 GM 物理地址；但 NPU 内存分配器对 2D+ 大张量倾向选 `FRACTAL_NZ` 物理布局（分块 Z 字形），PyTorch 读回 `z` 时按 `z` 的格式元数据解读。当物理布局是 NZ、kernel 按 NCHW/ND 行序写、PyTorch 按 NZ 解读时，写入位置与读回位置错位 → 特定行整行错、其余全对。曾一度归因为"kernel 计算逻辑错"，但小 shape 逐元素手工 trace 与参考一致，排除；进一步怀疑 `.contiguous()` 后 data_ptr 变了应已修复，但实测新张量仍可能被分配为 NZ，排除。

支撑诊断的判别探针（trace 内实际运行过的判别命令，逐字保留，来源平台：a2；证据等级: tool_observed）：

探针 A — 张量格式与物理布局：

```python
import torch, torch_npu
x = torch.randn(1024, 2048, device='npu')
print("input format:", x.npu_format())          # 2 = ACL_FORMAT_ND? 看运行期分配
x0 = torch.chunk(x, 2, dim=1)[0].contiguous()
print("x0 format:", x0.npu_format())
z = torch.empty((1024,1024), x.options())
print("z format:", z.npu_format())              # 若为 29 (FRACTAL_NZ) 则触发本问题
```

探针 B — `_npu_format_cast` 是否改物理布局（data_ptr 对比）：

```python
import torch, torch_npu
x = torch.randn(512, 2048, device='npu')
print("before data_ptr:", x.data_ptr())
y = torch_npu.npu_format_cast(x, 0)            # ACL_FORMAT_NCHW
print("after  data_ptr:", y.data_ptr())
print("same data_ptr:", x.data_ptr() == y.data_ptr())   # True ⇒ 仅改元数据, 未重排物理内存
```

探针 C — 同 shape 不同 ACL 格式 A/B（控制变量）：

```python
import torch, torch_npu
for fmt in [0, 2, 29]:   # NCHW, ND, FRACTAL_NZ
    x = torch.randn(1024, 2048, device='npu')
    x = torch_npu.npu_format_cast(x, fmt)
    out = cann_bench.swi_glu(x, dim=1)
    ref = (torch.nn.functional.silu(x[:,:1024].float()) * x[:,1024:].float()).to(out.dtype)
    print(f"Format {fmt}: max_diff={(out.cpu()-ref.cpu()).abs().max().item()}")
```

## 验证信号

本卡为 `negative_result`（尝试 + 客观失败 + 边界），按形态合同不强制 repair AFTER。以下为本 trace 内实测（来源平台：a2）：

- 大 shape FP32 dim=1 间歇失败 signature 稳定：`(512,2048) FP32 dim=1` 失败时 `Bad rows=[192]`、`Total bad elements: 64 out of 524288`，符合物理布局错位特征而非精度逻辑错误（来源平台：a2；证据等级：tool_observed）。
- 同 shape 连续重跑 5 次：前 1~2 次 `max_diff=3.5~39`（FAIL），后 3~4 次 `max_diff=0.00000143`（PASS），失败率约 2%（来源平台：a2；证据等级：tool_observed）。
- 探针 B：`before/after data_ptr` 相同，`same data_ptr: True` —— `_npu_format_cast` 对已分配张量只改格式元数据、不触发物理重排（来源平台：a2；证据等级：tool_observed）。
- 探针 C 同 shape 重跑，不同 seed/不同分配轮次下 `Format 0(NCHW): max_diff=3.15`、`Format 2(ND): max_diff=8.10`、`Format 29(FRACTAL_NZ): max_diff=6.21`，另一次同 shape 重跑 `ACL_FORMAT_ND(2): max_diff=3.67`、`ACL_FORMAT_NCHW(0): max_diff=0.00000143 PASS` —— 结果与 NPU 内存分配器当轮选的物理布局有关，与元数据标签不完全一致，故间歇（来源平台：a2；证据等级：tool_observed）。
- 格式探测确认 NPU 对 2D+ 大张量可能分配 `FRACTAL_NZ(29)`，ACL_FORMAT 枚举 `NCHW=0, NHWC=1, ND=2, FRACTAL_NZ=29`（来源平台：a2；证据等级：tool_observed）。
- 小 shape kernel 计算逻辑逐元素正确：`(8,8)` 等小 shape `max_diff=0.00000000`，手工 trace 输出与期望一致，排除 kernel 计算逻辑根因（来源平台：a2；证据等级：artifact_verified）。
- 绕过 OpCommand 直接 stream launch 使失败率从约 2% 升至约 77%，反证 OpCommand 的 stream 编排部分掩盖而非根治 bug（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- 本结论在 a2 / CANN 9.0.0 上成立。a3、950 的 NPU 内存分配器选格式行为未实测，**推断可迁移（未验证）**——只要分配器对 2D+ 大张量可能选非 NCHW 物理布局、且 kernel 按 flat 1D 写，就适用本结论。
- 本卡只覆盖"输出 `z` 物理布局错位"这一个根因；其他精度现象（如 FP16 cast-back 舍入、UB 尾块对齐）是独立根因，不在本卡。
- 本 trace 未定位 NPU 内存分配器在什么条件下对 `torch::empty` 的 2D+ 张量选 NCHW vs FRACTAL_NZ（看起来与张量 size、分配池状态、是否首次分配有关），故间歇性未在机制层闭环。
