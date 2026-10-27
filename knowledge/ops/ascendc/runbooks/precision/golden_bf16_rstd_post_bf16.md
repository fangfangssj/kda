---
type: "Runbook"
title: "golden BF16 rstd 归约误取 post-BF16 量化值"
description: "BF16 路径 rstd 在小 numCol 用例超 FP32 rtol，判别=读 golden.py compute_golden 的 BF16 分支，看 rstd 归约输入是 post-BF16 的 x.float()（FP32→BF16→FP32 round-trip）还是 pre-BF16 的 x1.float()+x2.float()；前者注入 2^-8 量化噪声、小 numCol 均值不抵消致 ~1.7e-3 超 9.77e-4。"
tags: ["task.debugging", "topic.precision", "topic.compute.normalization", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T08:11:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# golden BF16 rstd 归约误取 post-BF16 量化值

## 现象

在为 `add_rms_norm_cast`（Add + RMSNorm + Cast 融合算子）做 BF16 路径精度验收时，rstd 输出（FP32，`rsqrt(mean(x²)+eps)`）在两个小 numCol 用例上超出 FP32 容差，其余 78/80 组合通过：

- `B5: bf16 [2,16] SingleN 极小 shape` — rstd FAIL，`max_error=1.703024e-03`，`matched_ratio=0.500000`（标准 `rtol=9.77e-04`、`atol=1.53e-05`、`required_matched_ratio=0.99`）；同用例的 y1/y2/x 全 PASS。
- `B9: bf16 [64,64] 小方阵` — rstd FAIL，`max_error=1.691937e-03`，`matched_ratio=0.984375`；同用例的 y1/y2/x 全 PASS。

验收在 Ascend910B3（`--npu-arch=dav-2201`，`__CCE_AICORE__==220`，CANN 9.1.0/9.2.0）上独立运行 `scripts/precision_acceptance.py` 触发；编译通过、kernel 不崩溃、y2/x 与 golden 完全一致，唯有 BF16 小 numCol 的 rstd 偏差约 1.7e-3 超出 9.77e-4 的 rtol。FP16 全部用例与 BF16 大 numCol（≥1024）用例的 rstd 均通过，说明问题与 BF16 量化噪声在小归约规模下的相对放大强相关。

## 适用场景

- **工程形态**：kernel 直调（`<<<>>>` 内核调用 + op_extension PyTorch 注册），精度验收用独立 `precision_acceptance.py` 脚本 + `golden.py` CPU 参考实现比对。
- **API 路径**：kernel 侧 `Cast(x1,FP32)+Cast(x2,FP32) → Add → xFp32`（pre-BF16 FP32 和），`Mul(sqx,xFp32,xFp32)` 归约，`Sqrt → Div` 输出 1/rstd；golden 侧 `torch.rsqrt(x_fp32.pow(2).mean(-1)+eps)`。
- **验证过的平台与 CANN 版本**：Ascend910B3 / DAV_2201 / `--npu-arch=dav-2201`，CANN 9.1.0（验收）与 9.2.0（编译）。仅 a2 实测；A3 共享 DAV_2201 但未在本轨迹验证。
- **适用粒度**：`operator_family` —— 适用于一切“BF16 输入、内部 FP32 归约、同时把 BF16 量化值作为输出”的归一化/归约融合算子（RMSNorm/LayerNorm/RMSNormCast 类）的 golden 参考实现编写，不限于 add_rms_norm_cast 单体。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；修复实现的终版代码与修后重跑验收的 closure 未从 trace 完整恢复。
- **不适用或未验证场景**：FP16 路径不受影响（FP16 Add 在 FP16 完成，归约本就用 post-FP16 量化值，kernel 与 golden 同源）；BF16 大 numCol 因均值平均化使偏差可忽略，不会超差但根因仍在。A3（Ascend910_93 族）共享 DAV_2201 但未验证，推断可迁移（见失效边界）。

## 判别方法

**触发条件**（tool_observed 失败症状）：BF16 路径 rstd 在小 numCol（如 `[2,16]`、`[64,64]`）用例上 FAIL，`max_error` 约 1.7e-3 量级、超过 FP32 `rtol=9.77e-04`；同用例 y1/y2/x 全 PASS、FP16 全部 rstd PASS、BF16 大 numCol（≥1024）rstd PASS。分层特征：默认/大 numCol 全过、仅小 numCol 集中超差，且超差只出现在采用严格 FP32 容差的 rstd 输出上。

**可执行探针**（按序）：

1. **定位 golden 归约取值点**：读 `scripts/golden.py` 的 `compute_golden` 函数，查 BF16 分支里 `rstd = rsqrt(mean(x_fp32²)+eps)` 中的 `x_fp32` 来源。待识别的坏模式（BEFORE verbatim，取自 trace file_read 事件 `golden.py:48-59`，artifact_verified）：

```python
    # 1. x = x1 + x2（FP16 直接 Add；BF16 先 Cast FP32 再 Add 再 Cast 回 BF16）
    if dtype == torch.bfloat16:
        x = (x1.float() + x2.float()).to(torch.bfloat16)
    else:  # float16
        x = x1 + x2

    # 2. xFp32 = float(x)
    x_fp32 = x.float()

    # 3. rstd = rsqrt(mean(xFp32²) + eps)  [注意是 rsqrt，不是 sqrt]
    x_fp32_2d = x_fp32.reshape(numRow, numCol)
    rstd = torch.rsqrt(x_fp32_2d.pow(2).mean(-1, keepdim=True) + epsilon)  # [numRow, 1]
```

   关键判据：BF16 分支里 `x = (x1.float()+x2.float()).to(bfloat16)` 后 `x_fp32 = x.float()`——归约输入是 **post-BF16** 的 `x.float()`，即先得 pre-BF16 FP32 和、再量化回 BF16、再回 FP32 的 round-trip。若 `x_fp32` 直接取 `x1.float()+x2.float()`（pre-BF16），则不是本判别命中的坏模式。

2. **对照 kernel 数据流**：读 `DESIGN.md §1.3` 与 kernel CopyIn（`add_rms_norm_cast.h` BF16 分支），确认 kernel 归约 `Mul(sqx,xFp32,xFp32)` 用的是 CopyIn 阶段 `Cast(x1,FP32)+Cast(x2,FP32)` 的 pre-BF16 FP32 和，`Cast(xLocal, xFp32, RINT)` 回 BF16 仅用于输出 x、不回灌归约。若 golden 走 post-BF16 而 kernel 走 pre-BF16，即取值点分歧，命中本判别。

3. **rsqrt/sqrt 笔误排查**：确认 golden 的 rstd 用 `torch.rsqrt`（`1/sqrt`，对应 kernel `Sqrt → Div` 输出 1/rstd），而非 `torch.sqrt`。若文档/golden 误用 `torch.sqrt`，与归约取值点同一处片段一并记为待更正项（本判别的次级标志，不单独成立）。

**原始观测**（Ascend910B3 / DAV_2201 / CANN 9.1.0，`precision_acceptance.py` 输出）：

- BF16 `[2,16]` rstd FAIL：`max_error=1.703024e-03`，`matched_ratio=0.500000`（标准 `rtol=9.77e-04` `atol=1.53e-05` `required_matched_ratio=0.99`）；同用例 y1/y2/x 全 PASS。
- BF16 `[64,64]` rstd FAIL：`max_error=1.691937e-03`，`matched_ratio=0.984375`；同用例 y1/y2/x 全 PASS。
- FP16 全部 rstd PASS；BF16 大 numCol（≥1024）rstd PASS。
- `golden.py` `compute_golden` BF16 分支的归约输入为 post-BF16 `x.float()`（round-trip），见上文 BEFORE verbatim 代码块。

**机制解释**（根因分析，属 interpretation）：golden 与 kernel 在 BF16 路径归约输入的取值点上不一致。golden 先 `x = (x1.float()+x2.float()).to(bfloat16)` 把 FP32 和量化回 BF16，再 `x_fp32 = x.float()`，归约用这个 post-BF16 的 `x.float()`——相当于在归约前多走一次 `FP32 → BF16 → FP32` round-trip，把 BF16 的量化误差（约 2^-8 量级截断/舍入）注入归约输入；kernel 归约直接用 pre-BF16 的 `xFp32 = x1.float()+x2.float()`，不经过此次量化。rstd = `rsqrt(mean(x²)+eps)`，`mean` 对 numCol 个元素求平均：BF16 量化使每个 `x²` 带约 2^-8 相对误差，numCol 大（≥1024）时正负误差在均值中相互抵消，`mean(x²)` 相对误差被平均化到远低于 9.77e-4；numCol 小（16、64）时样本少、抵消不充分，`mean(x²)` 相对误差被放大到约 1.7e-3，超过 rstd 的 FP32 rtol=9.77e-4。FP16 路径 kernel 与 golden 都用 post-FP16 量化值（同源），故不受影响。BF16 的 y2/x 容差是 1.56e-2（远宽于 rstd 的 9.77e-4），同样 post-BF16 量化误差在 y2/x 上仍在容差内，只有严格 FP32 容差的 rstd 暴露了 golden 取值点错误。（注：上述“纯 FP32 rstd 与 kernel 一致、post-BF16 rstd 与 golden 一致”的数值对照由精度子 agent 叙述、原始 probe 输出未在轨迹保留，属 agent_reported，不列入 `## 验证信号`，仅作 interpretation 参考。）

## 验证信号

- 修复前 BF16 `[2,16]` rstd FAIL：`max_error=1.703024e-03`，`matched_ratio=0.500000`（标准 `rtol=9.77e-04` `atol=1.53e-05` `required_matched_ratio=0.99`）（来源平台：a2；证据等级：tool_observed）
- 修复前 BF16 `[64,64]` rstd FAIL：`max_error=1.691937e-03`，`matched_ratio=0.984375`（标准同上）（来源平台：a2；证据等级：tool_observed）
- 修复前同用例 y1/y2/x 均 PASS、FP16 全部 rstd PASS、BF16 大 numCol（≥1024）rstd PASS（来源平台：a2；证据等级：tool_observed）
- `golden.py` `compute_golden` BF16 分支归约取 post-BF16 `x.float()` 的 BEFORE verbatim 代码（`golden.py:48-59`，见 `## 判别方法`）（来源平台：a2；证据等级：artifact_verified）

## 失效边界

- 仅在 a2（Ascend910B3/DAV_2201）实测验证。A3（Ascend910_93 族）共享 DAV_2201 架构，根因属软件侧 golden 取值点错误、不依赖具体型号，推断可迁移，但未验证（推断可迁移）。
- 仅适用于 “BF16 输入 + 内部 FP32 归约 + BF16 输出” 的路径形态。若 kernel 本身就把归约放在 BF16 域完成（非本算子设计），则 golden 应与该路径同源，本卡的 pre-BF16 取值点结论不适用。
- FP16 路径不适用（FP16 Add 在 FP16 完成，归约本就用 post-FP16 值，kernel/golden 同源）。