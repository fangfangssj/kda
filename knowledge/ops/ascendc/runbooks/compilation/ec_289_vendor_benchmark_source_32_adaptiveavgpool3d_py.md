---
type: Runbook
title: "Vendor benchmark source `32_AdaptiveAvgPool3d.py get_input_groups()` crashes on every load — `for idx, case in cases:` unpacks single-key-per-line JSON dicts"
description: "厂商 benchmark 文件 `get_input_groups()` 对单行单键 dict 的解包假设错误（ValueError unpack）——解析 jsonl 输入的坑。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-289"]
---

# Vendor benchmark source `32_AdaptiveAvgPool3d.py get_input_groups()` crashes on every load — `for idx, case in cases:` unpacks single-key-per-line JSON dicts

`applies_to: soc=Ascend310P3 (vendor file defect, chip-independent by construction); cann=9.0.0; bisheng=n/a; op_class=32_AdaptiveAvgPool3d (vendor benchmark source)`
- **Symptom**: importing the vendor benchmark file raises `ValueError: not enough values to unpack (expected 2, got 1)` from `get_input_groups()` — the loop `for idx, case in cases:` treats each JSON dict entry as an `(idx, case)` tuple, but the cases list holds single-key-per-line dicts (`[{"shape": ...}, ...]`), so each element unpacks as one value, not two.
- **Fix**: `enumerate(cases)` — the evident intent (the input recipe branches on `idx` parity). Workspace `代码片段 32_AdaptiveAvgPool3d/model.py.snippet.py（见文末）` uses the corrected form; verbatim vendor file kept only as reference manifest. Anchor: 内部路径（不公开）vendor/AscendOpGenAgent/benchmarks/NPUKernelBench/level1/32_AdaptiveAvgPool3d.py:69 (defect) vs `代码片段 32_AdaptiveAvgPool3d/model.py.snippet.py（见文末）` (fix).
- **Rule**: a vendor benchmark file that cannot be imported is a reference-source defect, not a hardware finding — fix in the workspace `model.py` (document the delta in its docstring) and keep the vendor file untouched for provenance; do NOT patch the vendor tree in place. Same handling class as the vendor input-recipe defects documented under OL-87's `get_input_groups` family (there the bug was semantic OOB int ranges; here it is a Python-level crash — both are "reference never loads / loads wrong" defects).
- **Evidence**: 32_AdaptiveAvgPool3d 310p (2026-08-21, benchmark): vendor file failed on every load attempt; corrected `model.py` drove 100/100 Pass A + 168/168 Pass B. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`; the defect is source-level, visible without a device.) backend=ascendc.

## 内嵌代码片段

### 32_AdaptiveAvgPool3d/model.py.snippet.py

```python
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        output_size_info = inputs[1]
```

## 适用场景

- **工程形态**：厂商 benchmark 文件 `get_input_groups()` 对单行单键 dict 的解包假设错误（ValueError unpack）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
