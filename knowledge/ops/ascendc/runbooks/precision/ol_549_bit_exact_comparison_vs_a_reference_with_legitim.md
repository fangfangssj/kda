---
type: Runbook
title: "bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + nan_to_n"
description: "参考输出合法含 NaN 时，位精确判定要拆成 NaN 掩码位置一致 + nan_to_num 后位比较；torch.equal 遇 NaN 永假会造假阴性"
tags: [task.debugging, topic.compilation, topic.implementation.masking, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-549"]
---

# bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + nan_to_n

**OL-549** — **bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + `nan_to_num` bit-compare**: `torch.equal` alone fails on ANY NaN (manufactures a fake "0/N bit-exact"); an all-NaN fallback clause silently mis-grades MIXED NaN/finite outputs (~half-NaN reference with bit-identical finite parts fails BOTH arms and reads as a kernel bug when the predicate is the bug). soc=all, verifier/runner-side. 17_AdamW 310p 2026-08-15: fake 0/18 scare → corrected predicate (`代码片段 17_AdamW/pass_a_runner.py.snippet.py（文末附）`) → 13/18 + 605/645 bit-exact, residuals root-caused as hardware classes ([`PB-120`](../compilation/pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md) saturation, [`OL-147`](ol_147_a5_subnormal_handling_via_lnconfig_algo_for_ln_sqr.md) FTZ). Cross-ref [`OL-109`](ol_109_two_tier_precision_verdict.md), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md), OL-548, [`PB-120`](../compilation/pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md). backend=all.

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (verifier/runner-side)`
<!-- applies_to_backend: all -->

**Principle**: when a reference legitimately produces NaN in a SUBREGION of the output (Sqrt of negatives, 0/0, log of ≤0, inf−inf — e.g. an optimizer update whose v_hat < 0 feeds a Sqrt), a bit-exactness predicate must check TWO facts independently: (1) the NaN MASKS agree positionally, and (2) the finite values are bit-identical after `nan_to_num`. Two broken forms:
- `torch.equal(out, ref)` returns False if even ONE element is NaN on BOTH sides — an any-NaN reference can never pass, manufacturing a fake "0/N bit-exact" result that reads as a kernel precision bug.
- An "all-NaN everywhere" fallback clause (`torch.equal(o, r) or (o.isnan().all() and r.isnan().all())`) is silent poison for MIXED outputs: a reference that is ~half NaN with bit-identical finite parts fails BOTH arms, and the failure again reads as a kernel bug when the predicate is the bug.

**Concrete anchor** (correct form):
```python
an, bn = torch.isnan(a.float()), torch.isnan(b.float())
return torch.equal(an, bn) and bool(
    torch.equal(torch.nan_to_num(a.float(), 0.0),
                torch.nan_to_num(b.float(), 0.0)))
```
(`代码片段 17_AdamW/pass_a_runner.py.snippet.py（见文末）`; same predicate in `pass_b_runner.py`.)

**Evidence**: 17_AdamW 310p (2026-08-15, benchmark): the broken predicate manufactured a fake "0/18 bit-exact" scare (failures_ledger iter-5); corrected predicate → pass_a 13/18, pass_b 605/645 bit-exact, with every residual class root-caused as hardware classes (fp16 overflow saturation per PB-120, denormal FTZ per OL-147), NOT predicate artifacts (`17_AdamW/failures_ledger.md:16`).

**Other instances (predicted)**: any verifier/runner over references with legitimate NaN subregions — Sqrt/log-domain ops, 0/0 divisions, attention masks chaining −inf→NaN; also mixed inf/saturation outputs where mask-then-compare beats scalar equality.

**Cross-ref**: [`OL-109`](ol_109_two_tier_precision_verdict.md) (two-tier T1/T2 judging this predicate feeds), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) (logic-first — predicate bugs are harness bugs, fixed harness-side), OL-548 (sibling "grade what you think you graded" class — here the grader itself lied), [`PB-120`](../compilation/pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md) (the NaN→65504 saturation that makes NaN-region outputs legitimate in the first place). backend=all.

## 内嵌代码片段

### 17_AdamW/pass_a_runner.py.snippet.py

```python
    return mod

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_dir", nargs="?", default=".")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    so_dir = _find_so_dir()
    sys.path.insert(0, str(so_dir))
    sys.path.insert(0, str(HERE))

    ref_mod = _load_module(HERE / "model.py", "ref_model")
    new_mod = _load_module(HERE / "model_new_ascendc.py", "new_model")

    torch.manual_seed(0)
    input_groups = ref_mod.get_input_groups()
    total = len(input_groups)

    ref_model = ref_mod.Model().eval()
    new_model = new_mod.ModelNew().eval()

    n_pass = 0
    per_case = []
    per_dtype_pass = {}
    per_dtype_total = {}

    for case_idx, inputs in enumerate(input_groups):
```

## 适用场景

- **工程形态**：参考输出合法含 NaN 时，位精确判定要拆成 NaN 掩码位置一致 + nan_to_num 后位比较；torch.equal 遇 NaN 永假会造假阴性。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
