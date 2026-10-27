---
type: "Runbook"
title: "Exp 默认 INTRINSIC FTZ 致 subnormal 残留失配"
description: "arch35 Exp 默认 ExpAlgo::INTRINSIC 将 subnormal 输入 FTZ 为 0；softplus_v2_grad 的 sigmoid 分支在 z∈(-104,-87)（exp(z) 为 fp32 subnormal）+ gradOutput≈FLT_MAX 时，aten 顺序 grad*exp(z) 的中间 subnormal 被 FTZ→0 致 actual=0，而 aten golden（CPU 保留 subnormal）得有限值。aten 顺序（默认 INTRINSIC）575/578（3 失败，subnormal 区间）与 spec 除法形式 567/578（11 失败，z>0 溢出区间）失败集不重叠，无单一顺序在所有极值下匹配 golden，需按 golden 来源选顺序。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.math_function"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Exp 默认 INTRINSIC FTZ 致 subnormal 残留失配

## 现象

在 `softplus_v2_grad` 算子（arch35/DAV_3510，Ascend950PR，CANN 9.0.0-beta.2）修复 DFX_IN 缓存键、Select 掩码错配、`Maxs(-50)` 下限 clamp 之后，PyTorch ST 提升到 565-568/578，仍残留 3-13 例失败（固定 seed=42 后稳定为 3-4）。残留失败均位于 sigmoid 分支的极值区间：gradOutput≈FLT_MAX（3.4e38）、z=beta*self 落在 (-104,-87)（exp(z) 为 fp32 subnormal）。逐元素探针观测到 `actual=0`、`golden=有限值`（grad×subnormal）。典型 `aclnnSoftplusV2Grad_L1_018`：`[FAIL] Golden=2.152225 Actual=0.000000`。

## 适用场景

- **工程形态**：kernel 实现（`op_kernel/arch35/*.h`），sigmoid 类算子的数值分支。
- **API 路径**：`AscendC::Exp<T, config>`（template 参数 `ExpConfig`），`AscendC::ExpAlgo::{INTRINSIC, PRECISION_1ULP_FTZ_TRUE, PRECISION_1ULP_FTZ_FALSE}`。
- **验证平台与版本**：Ascend950PR（arch35/DAV_3510），CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family`（所有含 sigmoid/softplus 类分支、exp 输入可能落入 subnormal 区间、且 gradOutput 可达 FLT_MAX 量级的 kernel）。
- **路线优先级**：`diagnostic_only`——本卡记录 INTRINSIC FTZ 的失配区间与两种等价公式的失败集差异，不推荐按 golden 来源切换公式，也不把 clamp 作为通用修复。
- **不适用或未验证场景**：gradOutput 量级正常（非 FLT_MAX）时 subnormal×grad 本身下溢为 0，FTZ 与否无差；a2/a3 的 `Exp` FTZ 行为未实测；`PRECISION_1ULP_FTZ_FALSE` 是否在 arch35 运行时真正改变 `Exp`/`Mul` 的 subnormal 处理未在 trace 内确认。

## 判别方法

**触发（tool_observed）**：sigmoid 类分支 + gradOutput 大（FLT_MAX 量级）+ exp 输入落入 subnormal 区间，修复其他机制（缓存键、Select 掩码、clamp）后仍残留少量 `actual=0 / golden=有限值` 的失败。典型 `aclnnSoftplusV2Grad_L1_018`（`beta≈17.01, thr≈0.004, self∈[-100,100], gradOutput=±FLT_MAX`）单独跑即 `[FAIL] mismatch`，最大差异位置 `Actual=0.000000` 而 `Golden=2.152225`。

**探针**：逐元素比对并统计 z 分布（`/tmp/debug_l18d.py`，固定 `torch.manual_seed(0)` 隔离复现）：

```python
import torch
torch.ops.load_library("./build/libtorch_adapter.so")
from golden import compute_golden

beta=17.009491618554122
threshold=0.004012797157477129
go_shape=(3,10,2,7,7,10,5,3)
sf_shape=(3,10,2,7,7,1,5,3)
torch.manual_seed(0)
sf=torch.empty(sf_shape,dtype=torch.float64).uniform_(-100,100).to(torch.float32)
go=torch.full(go_shape,3.4028235e+38,dtype=torch.float64).to(torch.float32)
sf_b=torch.broadcast_to(sf,go_shape)
res=torch.ops.softplus_v2_grad.forward(go.to("npu"),sf.to("npu"),beta,threshold).cpu()
golden=compute_golden(go,sf,beta,threshold)
rf=res.flatten().float()
gf=golden.flatten().float()
diff=(rf-gf).abs()
print("diff nan count=",torch.isnan(diff).sum().item(),"inf count=",torch.isinf(diff).sum().item())
print("diff>1 count=",(diff>1).sum().item())
mismatch = (rf!=gf) & ~torch.isnan(rf)
print("mismatch(!=) count=",mismatch.sum().item())
idx=mismatch.nonzero()[:10].flatten()
for i in idx:
    print(f"  idx={i.item()} sf={sf_b.flatten()[i].item():.4f} actual={rf[i].item()} golden={gf[i].item()}")
```

判定要点：确认残留失败的 `Actual` 是否为 `0`、`Golden` 是否为 `sigmoid(大负 z)×FLT_MAX` 量级（≈2，非整段错、非 inf/nan 伪影），且失败位置 z 落在 `exp(z)` 为 subnormal 的区间（z∈(-104,-87)）。若 `Actual=0` 量级与 subnormal/FTZ 签名吻合且单独跑可复现，指向 Exp 的 subnormal 处理而非控制流分支错。

**原始观测（tool_observed）**：残留失败的 z 值集中在 (-104,-87)；`actual=0`、`golden=有限值`（grad×subnormal，如 `Golden=2.152225 Actual=0.000000`）。aten 顺序（默认 INTRINSIC）PyTorch ST 3 次重复 `通过: 575 / 失败: 3`；spec 除法形式 `grad/(1+exp(-z))` `通过: 567 / 失败: 11`，失败集与 aten 顺序不重叠；aten 顺序 + `PRECISION_1ULP_FTZ_FALSE` `通过: 567`（568→567，未改善）。

**判别与解释（interpretation，agent 根因分析，非 API 规范主张）**：arch35 `Exp` API 的 `ExpConfig` 默认 `ExpAlgo::INTRINSIC`（API 文档 `ascendc-precision-debug/references/asc-devkit-api/basic-api/Exp.md` 记载：`INTRINSIC`、`PRECISION_1ULP_FTZ_TRUE` 使用单指令计算，所有 Subnormal 被近似为 0；`PRECISION_1ULP_FTZ_FALSE` 支持 Subnormal 数据计算）。机制链（sigmoid 分支，z=beta*self 极负）：`z∈(-104,-87)` → `exp(z)` 为 fp32 subnormal（最小正常数约 `1.18e-38`，`exp(-87)≈8.5e-39` 已 subnormal）→ 默认 `INTRINSIC` FTZ 为 0 → `grad_sig=grad*0=0`；aten golden（CPU fp32）保留 subnormal → `FLT_MAX×8.5e-39≈2.9`（有限值）→ `actual=0≠golden=有限值`。算式：`exp(-87.5)≈5.4e-39`（subnormal），`FLT_MAX(3.4e38)×5.4e-39≈1.8`（有限），FTZ 后 `3.4e38×0=0`。与此交织的另一层根因是**公式顺序**：aten `softplus_backward` 用 `grad*exp(z)/(1+exp(z))`（先乘后除），spec.yaml 用 `grad/(1+exp(-z))`（除法形式）。两顺序在极值下失效区间不同且不重叠：z 极负（exp(z) subnormal）时 aten 形式 `grad*exp(z)` FTZ→0（错），spec 形式 `exp(-z)` 为大正常数→`grad/large` 正确收敛；z 极正且 exp(z) 不溢出时 aten 形式中间溢出为 inf（与 aten golden 一致），spec 形式得有限值（与 golden 不符）。`PRECISION_1ULP_FTZ_FALSE` 未改善（568→567）的推断是 `Mul` 同样对 subnormal 输入做 FTZ，故仅改 `Exp` 配置无法救回 `grad*exp_subnormal`——该 Mul-flush 推断未独立隔离验证。单点归因：残留 `actual=0/golden=有限值` 失败的 NPU 侧根因是 Exp/Mul 对 subnormal 的 FTZ 行为，叠加公式顺序在极值下的失效区间差异，非控制流分支错、非全为 golden 极值伪影。

## 验证信号

- 残留失败位置 `Actual=0.000000` vs `Golden=2.152225`（z≈-88, gradOutput=±FLT_MAX，golden 期望 `sigmoid(-88)×FLT_MAX≈2.06`）（来源平台：950；证据等级：tool_observed）
- aten 顺序（默认 INTRINSIC，FTZ）PyTorch ST 3 次重复：`通过: 575 / 失败: 3`（575/578，残留 subnormal 区间失败）（来源平台：950；证据等级：tool_observed）
- spec 除法形式 `grad/(1+exp(-z))` PyTorch ST：`通过: 567 / 失败: 11`（567/578，z>0 溢出区间失败，失败集与 aten 顺序不重叠）（来源平台：950；证据等级：tool_observed）
- aten 顺序 + `PRECISION_1ULP_FTZ_FALSE`：`通过: 567`（568→567，未改善）（来源平台：950；证据等级：tool_observed）

## 坏实践

- 在 arch35 上对 sigmoid 分支用 aten 顺序 `grad*exp(z)/(1+exp(-z))` 且 gradOutput 可达 FLT_MAX、z 可极负，而不处理 `exp(z)` 的 subnormal FTZ——`grad*0=0` 残留失配。该路线在 subnormal 区间标记为 `banned`（除非显式确认 `Mul` 不 flush subnormal 输入）。
- 放宽 `numerical_tolerance` 把残留极值失败掩盖为通过——测试侧放松捷径，不解决根因，掩盖 Exp/Mul 的 FTZ 行为。
- 未确认 `PRECISION_1ULP_FTZ_FALSE` 在运行时生效就当作已修复——本 trace 中该配置未改善（568→567），且 `Mul` 是否同样 flush subnormal 输入未独立验证。

## 误诊

- 早期曾把残留失败全归因为"aten golden 极值溢出伪影（inf/nan），kernel 计算正确"——部分 case 确如此（z 正且 exp(z) 溢出），但 z 负 subnormal 区间的 `actual=0≠golden=有限值` 是 NPU FTZ 行为，不能全归为 golden 伪影。用户指出应检查 kernel 公式顺序是否与 golden 一致，agent 浅尝辄止过早收尾。

## 失效边界

- 仅 Ascend950PR/arch35（CANN 9.0.0-beta.2）实测；`PRECISION_1ULP_FTZ_FALSE` 在该平台运行时是否真正改变 `Exp`/`Mul` 的 subnormal 处理未验证，推断可迁移但需先确认配置生效。
- a2/a3 的 `Exp`/`Mul` FTZ 行为未实测，结论仅对 arch35 成立。
- gradOutput 始终为常规量级（非 ±FLT_MAX）时 subnormal×grad 本身下溢为 0，FTZ 与否无差，本卡诊断仅在极端 gradOutput 下必要。