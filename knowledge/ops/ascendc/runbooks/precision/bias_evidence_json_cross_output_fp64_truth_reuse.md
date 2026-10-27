---
type: "Runbook"
title: bias 噪声证据失真：write_evidence 复用 input 真值
description: 证据 JSON 中 grad_hidden_bias 噪声数百量级值系生成脚本对两路 bias 输出复用同一 input-gates fp64 真值所致伪影；位流 diff+口径复算+生成脚本审计三步判别。
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# bias 噪声证据失真：write_evidence 复用 input 真值


## 现象

在开发 ThnnFusedGruCellGrad 算子（对标 torch aten `_thnn_fused_gru_cell_backward` 的 GRU Cell 反向算子，走 aclnn 算子仓开发流程）的 Branch-0 ST 精度验收时，Validation.md「精度验收标准」刚完成一次用户裁定的修订：两路归约主导的 bias 输出（grad_input_bias / grad_hidden_bias）mare 预算由固定 10×th 改为 max(10×th, 10×golden fp64 噪声 mare)，噪声须在期望值生成时逐用例现算。为实现该口径，我在期望值生成器 `operators/ThnnFusedGruCellGrad_package/tests/aclnn/gen_st_oracle.py` 中新增了噪声现算（fp64 真值 = 同语义五条梯度链 + 两路 bias 沿 batch 归约，在 fp64 域全链计算），先在临时目录跑生成器核验噪声数值时发现：**现算结果与既有证据文件 `docs/ThnnFusedGruCellGrad/develop/checks/branch0_bias_mare_evidence.json` 的 `mare_golden_vs_fp64` 显著不符**。

证据 JSON（前次失败定界时生成，被 Validation.md 修订记录与 REQUIREMENTS §8.2.4 指向引用）：

| 用例 | input 噪声（证据） | hidden 噪声（证据） |
|---|---|---|
| P7（B=32,H=2048,fp32） | 4.3034e-4 | 394.28 |
| B0-a（B=64,H=2176,fp32） | 1.9138e-4 | 2315.45 |
| B0-c（B=96,H=4160,fp32） | 5.7857e-4 | 1751.86 |
| B0-e（B=256,H=2176,fp32） | 4.9882e-3 | 217.93 |

本次按规范口径现算（生成器输出 `bias_mare_noise(in/hid)`）：

| 用例 | input 噪声（现算） | hidden 噪声（现算） |
|---|---|---|
| P7 | 3.232e-4 | 3.232e-4 |
| B0-a | 8.863e-4 | 7.578e-5 |
| B0-c | 6.055e-4 | 6.055e-4 |
| B0-e | 1.135e-3 | 1.135e-3 |

判别信号有两点：①证据中 hidden 噪声比 input 噪声大 3~6 个数量级（如 P7：394.28 vs 4.3e-4），且证据同时登记 `rms_abs_golden_vs_fp64`≈0.93~1.30——对 B=32~256 行 fp32 求和而言，O(1) 的绝对偏差在量纲上不可能是舍入噪声（应为 ~1e-6 量级）；②证据目录下并存多份脚本（threeway.py / final_evidence.py / write_evidence.py），"证据数值"到底出自哪一份、口径是否一致需要逐份核对，而 Validation.md 规范文本只声明"与证据 threeway.py 口径一致"。

## 适用场景

- 工程形态：aclnn 算子仓 ST 精度验收与精度证据链复核（ops-registry-invoke 流程）；多输出算子（输出同时含归约主导路 + 逐元素路）的精度口径落地与噪声证据验证。
- API 路径：`tests/aclnn/gen_st_oracle.py`（期望值/噪声生成）、`tests/aclnn/build/st_data/<case>/`（位流与 meta.json）、`docs/ThnnFusedGruCellGrad/develop/checks/`（证据脚本与证据 JSON）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN cann-9.0.0-beta.2（npu-smi 25.7.rc1.6 实测在位；期望值生成、位流比对与噪声复算均在该环境执行）。
- 适用粒度（transfer_scope 最窄档）：operator_family——多输出 RNN 类梯度算子的精度证据方法学；判别规则与三步探针配方跨算子可迁移。
- 路线优先级：recommended_fastpath——任何把证据数值当预算/阈值输入的场景，消费前先复算 + 量纲自检。
- 明确不适用/未验证场景：单输出算子（无跨输出真值复用形态）；非数值类证据；final_evidence.py 等其余证据脚本的口径一致性未逐一审计（未验证场景）。

## 判别方法（diagnostic_rule：触发 / 探针 / 原始观测 / 解释）

**触发（何时怀疑证据失真）**

- 复算值与证据登记值不符，且不符呈"某一路输出系统性大几个数量级"的形态；
- 量纲自检不过：golden-vs-fp64 的 rms_abs 达 O(1)（本例 0.93~1.30），而 B=32~256 行 fp32 求和的舍入误差绝对量级应为 ~1e-6 级——差 6 个数量级，必是真值/口径错位而非舍入。

**探针（三步，全部可执行）**

第一步：排除数据漂移——新旧期望值位流逐文件比对（确定性种子下应逐位一致）：

```bash
for c in P7 B0-a B0-c B0-e E01; do
  for f in grad_hy.bin storage.bin exp_grad_input_bias.bin exp_grad_hidden_bias.bin; do
    old=operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/st_data/$c/$f
    new=/tmp/opencode/st_data_noisecheck/$c/$f
    if cmp -s "$old" "$new"; then echo "$c/$f SAME"; else echo "$c/$f DIFF"; fi
  done
done
```

第二步：用证据声称的口径（threeway.py 的 chains + rel 公式）在同一份 st_data 上独立复算（探针全文见「完整代码」第 3 段）：

```bash
python3 /tmp/opencode/noise_method_check.py
```

第三步：审计真正生成证据 JSON 的脚本源码（`/tmp/opencode/dump/write_evidence.py`，见「完整代码」第 1 段）——重点看 fp64 真值 `t64` 如何构建、是否逐输出。

**原始观测（verbatim）**

- 第一步输出：4 用例 × 4 文件（含 `exp_grad_hidden_bias.bin`）全部 `SAME`——数据无漂移，差异只能在计算本身。
- 第二步输出（threeway 口径在旧 st_data 上复算）：

```text
P7: input  mare=3.231617e-04 at col 1684 (|t|=5.518e-05, |g-t|=1.786e-08)
P7: hidden mare=3.231617e-04 at col 1684 (|t|=5.518e-05, |g-t|=1.786e-08)
B0-a: input  mare=8.863379e-04 at col 4568 (|t|=2.860e-04, |g-t|=2.536e-07)
B0-a: hidden mare=7.577723e-05 at col 5208 (|t|=7.360e-04, |g-t|=5.578e-08)
B0-c: input  mare=6.054563e-04 at col 5978 (|t|=4.557e-04, |g-t|=2.759e-07)
B0-c: hidden mare=6.054563e-04 at col 5978 (|t|=4.557e-04, |g-t|=2.759e-07)
B0-e: input  mare=1.135488e-03 at col 3775 (|t|=6.978e-05, |g-t|=7.934e-08)
B0-e: hidden mare=1.135488e-03 at col 3775 (|t|=6.978e-05, |g-t|=7.934e-08)
```

与生成器现算逐值一致，与证据 JSON 的 hidden 值（394.28/2315.45/1751.86/217.93）不符。

- 第三步观测：`write_evidence.py` 的 `load()` 只构建 **input-gates** 矩阵并返回；主循环对 `["grad_input_bias", "grad_hidden_bias"]` 两路输出**复用同一个 `t64`**（见「完整代码」第 1 段）。

**解释（机制链，含算式）**

- 结构：`input_gates = [grg | gig | gin]`，`hidden_gates = [grg | gig | ghn]`，其中 `ghn = gin * r`。前 2H 列两路完全相同；第 3 段（列 2H..3H-1）一边是 gin、一边是 ghn=gin·r，逐行不同。
- write_evidence.py 的 `t64` 是 input-gates 真值。用它去判 hidden golden 时，第 3 段每列：

  rel = |Σgin·(r−1)| / (|Σgin| + 1e-7)

  r∈(0,1) 逐行随机，Σgin·(1−r) 与 Σgin 同量级 → 一般列 rel = O(1)；Σgin 的相消列（|Σgin|≈0）上 rel 放大到数百~数千。这正是证据 hidden 值 394.28/2315.45/1751.86/217.93 与 `rms_abs_golden_vs_fp64`≈0.93~1.30（|Σgin·(1−r)| 的典型量级）的来源——**hidden golden 与 input 真值错位比对的伪影，不是 hidden 输出的真实噪声**。
- input 路数值也有偏差（证据 P7 4.3034e-4 vs 复算 3.2316e-4；B0-a 1.9138e-4 vs 8.863e-4）：write_evidence.py 的 t64 = **fp32 链值** `.astype(np.float64)` 后求和（"fp32 链 + fp64 归约"），而规范口径 = 输入位流升 fp64 后**全链 fp64**（"fp64 链 + fp64 归约"，threeway.py 口径）——两种真值定义的口径差。
- 结论：证据 JSON 中 grad_hidden_bias 的 `mare_golden_vs_fp64` / `mare_kernel_vs_fp64` / `rms_abs_*` 以及 `candidate_mare_vs_golden`（sequential / pure_pairwise_tree / blocks48_seq / fp64_exact / torch_casc16，均为 input-gates 求和对 hidden golden 的 mare）**全部失真**；Validation.md 规范文本（"fp64 真值 = 同一语义公式在 fp64 域全链计算，与证据 threeway.py 口径一致"）指向 threeway.py 的**逐输出全 fp64 链**真值，生成器实现与之逐值吻合（4 用例 × input/hidden 共 8 值）。

## 设计决策

- 为何按"规范文本 + threeway 口径"实现而不采信证据数值：预算消费的是噪声值——采信 394.28 会得到 hidden 预算 = 10×394.28 ≈ 3942.8，实质等于取消该输出的 mare 约束；量纲自检（rms_abs 达 O(1)）一票否决。规范口径的 hidden 噪声（7.578e-5~1.135e-3）仍全部 > fp32 th=1e-5（噪声主导形态成立），预算 10×noise = 7.578e-4~1.135e-2 仍有实际约束力。
- 放弃的候选：在本任务内修正 write_evidence.py / 重生成证据 JSON——超出任务范围（本任务只改判读链路）；改为在开发日志登记"证据 hidden 数百值系 write_evidence 以 input 真值代算的口径差"，规范性文本优先。
- 当前方案属哪档路线：recommended_fastpath（证据数值消费前先复算 + 量纲自检 + 核对生成脚本本体）。

## 系统信息

- 环境：Ascend950PR NPU（npu-smi 25.7.rc1.6），CANN cann-9.0.0-beta.2；torch 2.7.1+cpu、numpy 2.5.1。
- 目录布局：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/`（gen_st_oracle.py、build/st_data/\<case\>/、st_cases_branch0.json、test_aclnn.cpp）；`docs/ThnnFusedGruCellGrad/develop/checks/`（branch0_bias_mare_evidence.json、branch0_bias_mare_dumps/threeway.py）。
- 相关文件清单（与本次诊断实际读写一一对应）：读 `branch0_bias_mare_evidence.json`、`branch0_bias_mare_dumps/threeway.py`、`branch0_bias_mare_dumps/final_evidence.py`、`/tmp/opencode/dump/write_evidence.py`、`build/st_data/<case>/{meta.json,*.bin}`；写 `/tmp/opencode/noise_method_check.py`（复算探针）与 `/tmp/opencode/st_data_noisecheck/`（临时生成目录）。

## 完整代码

第 1 段——证据生成脚本 write_evidence.py 的 bug 现场（verbatim，fp64 真值构建与两输出复用处）：

```python
def rel(a, g):
    return np.abs(a - g) / (np.abs(g) + 1e-7)


def load(case):
    meta = json.loads((ST / case / "meta.json").read_text())
    B, H = meta["B"], meta["H"]
    go = np.fromfile(ST / case / "grad_hy.bin", dtype=np.float32).reshape(B, H)
    st = np.fromfile(ST / case / "storage.bin", dtype=np.float32).reshape(B, 5 * H)
    r_ = st[:, 0*H:1*H]; z = st[:, 1*H:2*H]; n_ = st[:, 2*H:3*H]
    hx = st[:, 3*H:4*H]; hn = st[:, 4*H:5*H]
    gin = (go * (1 - z) * (1 - n_ * n_)).astype(np.float32)
    gig = (go * (hx - n_) * (1 - z) * z).astype(np.float32)
    grg = (gin * hn * (1 - r_) * r_).astype(np.float32)
    return B, H, np.concatenate((grg, gig, gin), axis=1)   # ← 只返回 input-gates 矩阵（fp32 链值）


for case in CASES:
    B, H, X = load(case)
    t64 = X.astype(np.float64).sum(axis=0)                  # ← input-gates 的 fp64 和，被两输出共用
    t64f = t64.astype(np.float32)
    aUb, rUb, r_loop = sk.tiling(B, H)
    c = {
        "B": B, "H": H, "aUbFactor": aUb, "rUbFactor": rUb, "rLoopCntTotal": r_loop,
        "orders": {},
    }
    for gname in ["grad_input_bias", "grad_hidden_bias"]:
        g = np.fromfile(ST / case / f"exp_{gname}.bin", dtype=np.float32)
        k = np.fromfile(ACT / f"{case}.act.{gname}.bin", dtype=np.float32)
        sim, _ = sk.simulate_case(case, gname)
        entry = {
            "mere_kernel_vs_golden": float(rel(k, g).mean()),
            "mare_kernel_vs_golden": float(rel(k, g).max()),
            "mare_golden_vs_fp64": float(rel(g, t64).max()),    # ← hidden 时 g 是 hidden golden、t64 仍是 input 真值
            "mare_kernel_vs_fp64": float(rel(k, t64).max()),
            "rms_abs_kernel_vs_fp64": float(np.sqrt(((k - t64) ** 2).mean())),
            "rms_abs_golden_vs_fp64": float(np.sqrt(((g - t64) ** 2).mean())),
            "simulator_bitwise_match_kernel": bool(np.array_equal(sim, k)),
            # ……（candidate_mare_vs_golden 各求和序同样基于 X 的 input-gates 求和，
            #     对 grad_hidden_bias 一并失真，略）……
        }
        c["orders"][gname] = entry
    evidence["cases"][case] = c
```

第 2 段——规范口径参照 threeway.py 的逐输出真值核心（verbatim）：

```python
def chains(go, st, H, dtype):
    r = st[:, 0 * H:1 * H]
    z = st[:, 1 * H:2 * H]
    n = st[:, 2 * H:3 * H]
    hx = st[:, 3 * H:4 * H]
    hn = st[:, 4 * H:5 * H]
    gin = go * (1 - z) * (1 - n * n)
    gig = go * (hx - n) * (1 - z) * z
    grg = gin * hn * (1 - r) * r
    ghn = gin * r
    in_gates = np.concatenate((grg, gig, gin), axis=1)
    hid_gates = np.concatenate((grg, gig, ghn), axis=1)
    return in_gates.astype(dtype), hid_gates.astype(dtype)


def rel(a, g):
    return np.abs(a - g) / (np.abs(g) + 1e-7)


# —— 主循环内：fp64 真值 = 输入升 fp64 → 全链 fp64 → 逐输出拼接 → fp64 归约 ——
go64 = go.astype(np.float64)
st64 = st.astype(np.float64)
in64, hid64 = chains(go64, st64, H, np.float64)
t_in = in64.sum(axis=0)
t_hid = hid64.sum(axis=0)
for name, k, g, t in [("grad_input_bias", k_in, gin_b, t_in),
                      ("grad_hidden_bias", k_hid, hin_b, t_hid)]:
    ...
```

第 3 段——复算探针 noise_method_check.py 全文（verbatim，可独立执行）：

```python
import numpy as np, json
from pathlib import Path
ST = Path("operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/st_data")

def load_inputs(case):
    meta = json.loads((ST / case / "meta.json").read_text())
    B, H = meta["B"], meta["H"]
    go = np.fromfile(ST / case / "grad_hy.bin", dtype=np.float32).reshape(B, H)
    st = np.fromfile(ST / case / "storage.bin", dtype=np.float32).reshape(B, 5 * H)
    return go, st, B, H

def chains(go, st, H, dtype):
    r = st[:, 0*H:1*H]; z = st[:, 1*H:2*H]; n = st[:, 2*H:3*H]
    hx = st[:, 3*H:4*H]; hn = st[:, 4*H:5*H]
    gin = go * (1 - z) * (1 - n * n)
    gig = go * (hx - n) * (1 - z) * z
    grg = gin * hn * (1 - r) * r
    ghn = gin * r
    in_gates = np.concatenate((grg, gig, gin), axis=1)
    hid_gates = np.concatenate((grg, gig, ghn), axis=1)
    return in_gates.astype(dtype), hid_gates.astype(dtype)

def rel(a, g):
    return np.abs(a - g) / (np.abs(g) + 1e-7)

for case in ["P7", "B0-a", "B0-c", "B0-e"]:
    go, st, B, H = load_inputs(case)
    gin_b = np.fromfile(ST / case / "exp_grad_input_bias.bin", dtype=np.float32)
    hin_b = np.fromfile(ST / case / "exp_grad_hidden_bias.bin", dtype=np.float32)
    go64 = go.astype(np.float64); st64 = st.astype(np.float64)
    in64, hid64 = chains(go64, st64, H, np.float64)
    t_in = in64.sum(axis=0); t_hid = hid64.sum(axis=0)
    rg_in = rel(gin_b, t_in); rg_hid = rel(hin_b, t_hid)
    # where is the max?
    i_in = int(np.argmax(rg_in)); i_hid = int(np.argmax(rg_hid))
    print(f"{case}: input  mare={rg_in.max():.6e} at col {i_in} (|t|={abs(t_in[i_in]):.3e}, |g-t|={abs(gin_b[i_in]-t_in[i_in]):.3e})")
    print(f"{case}: hidden mare={rg_hid.max():.6e} at col {i_hid} (|t|={abs(t_hid[i_hid]):.3e}, |g-t|={abs(hin_b[i_hid]-t_hid[i_hid]):.3e})")
```

## 预防

- 噪声/容差类证据落盘时附量纲自校验：golden-vs-fp64 的 rms_abs 应在 fp32 求和舍入量级（~1e-6 级），O(1) 即报警。
- 多输出算子的证据脚本**逐输出构建 fp64 真值**——每个输出的真值按该输出的 gate 矩阵单独计算，禁止跨输出复用同一真值矩阵。
- 预算/阈值消费侧（ST 判读、容差设定）对引用的证据数值先独立复算再采用；复算口径先对照规范文本锁定。
- 证据 JSON 内登记生成脚本路径与真值口径说明，避免多脚本（threeway / write_evidence / final_evidence）口径混淆。

## 验证信号

以下信号来自 ThnnFusedGruCellGrad Branch-0 ST 判读口径更新开发会话中对噪声证据链的复核：期望值生成器现算噪声 → 与证据 JSON 对比发现不符 → 三步探针（位流 diff / threeway 口径复算 / 生成脚本审计）定位，全部为该会话内工具执行实测。

1. 期望值位流确定性：4 用例（P7/B0-a/B0-c/B0-e）× 4 文件（grad_hy.bin / storage.bin / exp_grad_input_bias.bin / exp_grad_hidden_bias.bin）新旧生成逐位 SAME（来源平台：950；证据等级：tool_observed）。
2. threeway 口径独立复算（同一份 st_data）：input/hidden mare = 3.231617e-4/3.231617e-4（P7）、8.863379e-4/7.577723e-5（B0-a）、6.054563e-4/6.054563e-4（B0-c）、1.135488e-3/1.135488e-3（B0-e），与生成器现算一致，与证据 JSON hidden 值（394.2761159297635 / 2315.4469959549256 / 1751.86 / 217.93）不符（来源平台：950；证据等级：tool_observed）。
3. 证据生成脚本源码审计：write_evidence.py 的 load() 仅构建并返回 `np.concatenate((grg, gig, gin), axis=1)`（input-gates、fp32 链值），主循环对两路 bias 输出复用同一 `t64 = X.astype(np.float64).sum(axis=0)`（来源平台：950；证据等级：tool_observed）。
4. 规范口径落地后 ST 全量重跑 20/20 PASSED（4 条原超差用例的预算来自与复算一致的噪声值，详见同会话[噪声自适应预算卡](bias_reduction_mare_noise_adaptive_budget.md)）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本判别针对"证据脚本跨输出复用真值矩阵 / 真值口径漂移"类失真；若证据由唯一脚本生成且口径自洽，三步探针中第二步即可吻合、无需第三步。
- 复算口径必须先对照规范文本锁定：若规范本身定义的是"fp32 链 + fp64 归约"（write_evidence 式），则 input 路数值差异属口径差而非失真——本例规范文本明确为全 fp64 链（threeway 口径）。
- 相消殆尽列（fp64 真值≈0）上 mare_noise 本身可达大值是真实数学形态（本例 input 路最差列 |t|=5.518e-05 即其弱形态，rel=1.786e-8/(5.518e-5+1e-7)≈3.23e-4）；"数百量级"特指真值错位伪影，两者必须区分。
- 未验证场景：final_evidence.py（fp32 链 + fp64 归约口径）与 threeway.py 的口径统一；fp16/bf16 大 B 用例的 hidden 噪声形态（本批半精度用例均为小 B）。

## 坏实践

- 直接采信证据 JSON 数值设置预算：hidden 预算将变成 10×394.28≈3942.8，实质等于取消该输出的 mare 约束。
- 只读方法描述（"与 threeway.py 口径一致"）不核对生成脚本本体——证据数值可能出自另一份口径不同的脚本（本例出自 write_evidence.py 而非 threeway.py）。
- 证据脚本一输出计算真值、多输出套用（跨输出复用真值矩阵）。

## 误诊

- 既有记载（Validation.md 修订记录「相消殆尽列的预期形态」条目）称"实测 grad_hidden_bias 达数百量级（见证据文件）"——本次复算证明该数百量级系证据伪影（hidden golden 与 input 真值错位比对），hidden 真实噪声为 7.578e-5~1.135e-3；该描述性文字本会话未修正（仅开发日志登记），引用时须注意（来源：文档记载为前会话产物，本次以 tool_observed 复算对照修正）。
- 排查顺序参考：先怀疑自身实现（fp64 真值函数切片/宽度错误）→ 位流 diff 排除数据因素 → 证据声称口径复算排除方法因素 → 最后审计生成脚本定位。前两步的"嫌疑"为合理排查路径而非实际误诊。

## 未决事项

- write_evidence.py 本体未修正，其 JSON（已被 Validation.md 修订记录与 REQUIREMENTS §8.2.4 指向引用）中 grad_hidden_bias 相关数值持续失真；后续应修正脚本重生成证据，或在替代证据中作废该批 hidden 数值。
- Validation.md 修订记录引用的 input 噪声范围（1.9e-4~5.0e-3）亦为 write_evidence 口径（fp32 链 + fp64 归约）数值；全 fp64 链口径下为 3.2e-4~1.1e-3——"噪声 > th 一个量级以上"的结论在两种口径下均成立，但具体数值引用时应区分口径。
- final_evidence.py 与 threeway.py 两套真值口径并存的统一未处理。
