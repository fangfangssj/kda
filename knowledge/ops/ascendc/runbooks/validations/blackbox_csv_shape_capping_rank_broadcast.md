---
type: "Runbook"
title: 黑盒CSV转ST清单：shape capping保rank广播
description: TTK 黑盒 CSV 用例形状可达 2.1B 元素无法上板；manifest 生成器做 per-dim + 总量双重封顶（1M/1024），保 rank/广播模式/dtype/值域，输出 JSON 清单驱动 ST。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:41:27Z'
updated_at: '2026-09-17T04:41:27Z'
platforms: ["950"]
---

# 黑盒CSV转ST清单：shape capping保rank广播

## 现象

把 softplus_v2_grad 的黑盒 TTK 用例（三层 CSV：L0 精选 40 条、L1 随机 500 条、L2 异常 9 条，共 549 条）纳入 aclnn ST 框架（[黑盒CSV用例数据驱动aclnn ST框架](blackbox_csv_data_driven_st_framework.md)）时，第一个拦路虎是**形状规模**：逐条统计元素数发现

```
L0: 40 cases, min=6, median=75547320, max=2103307908    >1M: 26, >10M: 25, >100M: 19
L1: 500 cases, min=2, median=212942730, max=2103307908  >1M: 316, >10M: 288, >100M: 262
L2: 9 cases, min=6, median=245616, max=8573040
Total cases: 549    （>1M 元素共 343 条）
```

最大 2,103,307,908 元素（fp32 即 ~8.4GB 单张量），远超设备内存，无法直接上板执行；但直接砍掉这些用例又会丢掉它们覆盖的因子（rank、广播模式、dtype 组合、值域）。需要在不破坏用例设计意图的前提下把形状缩到可执行规模。

## 适用场景

- 工程形态：黑盒/随机生成的 ST 用例集（TTK CSV）纳入可执行 aclnn ST 框架前的预处理；任何"用例按因子覆盖设计、形状随机膨胀"的场景。
- API 路径：CSV 列结构 `testcase_name, api_name, tensor_view_shapes, tensor_dtypes, scalar_dtypes, attributes, output_tensor_indexes, precision_tolerances, ..., input_data_ranges`；输出 `aclnn_cpp_cases.json` manifest。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0。
- 适用粒度：`workflow_only` 偏 `platform_wide` 的测试基建配方（capping 本身与芯片无关，上限取值与设备内存相关）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证：白盒用例（形状本就手工设计无需封顶）；L2 异常用例的输出形状会被替换为 cap 后广播形状（见失效边界）。

## 根因
黑盒随机生成器按"因子覆盖"造用例：rank 取 1–8、每维尺寸随机（可达数千），元素总量随维数乘积膨胀到 1e9 量级；而用例的**验证价值在因子组合**（rank、哪些维是 1 的广播模式、dtype、值域含 inf/nan/denormal、属性），不在具体尺寸。因此正确做法是 **shape capping**（CANN ST 框架对黑盒超大 shape 用例的既有实践）：封顶尺寸但保因子——

1. `CAP_TOTAL = 1_048_576`（单用例广播后总元素上限 1M）、`CAP_DIM = 1024`（单维上限）；
2. 先对广播 shape 逐维 `min(d, 1024)`（原始为 1 的维保持 1，保广播模式）；
3. 总量仍超限时**循环把最大维减半**直到达标（保 rank：不减维数）；
4. 从 cap 后广播 shape **反向重构**各输入 shape：原始为 1 的维仍为 1，其余维取 cap 后对应值（保对齐广播关系）；
5. dtype、值域、属性原样透传（保其余因子）。

## 完整代码

`gen_blackbox_manifest.py` 终版全文（verbatim 取自事件流中的终版文件读取；入口 `main` 合并目录下全部 `*_test_cases.csv` 为单一 manifest，并统计 expect_error 条数）：

```python
#!/usr/bin/env python3
"""Merge blackbox TTK CSVs (L0/L1/L2) into one ACLNN ST manifest with shape capping.

Reads all *_test_cases.csv in the blackbox testcases dir, converts each row to a
manifest case, caps oversized shapes to practical ST limits (preserving rank +
broadcast pattern + dtype + value ranges), marks L2 exception cases (empty
output_tensor_indexes) as expect_error, and writes a single merged
aclnn_cpp_cases.json.

Shape capping is standard CANN ST practice for blackbox cases with impractically
large shapes (2B+ elements): cap per-dim size and total element count while
preserving the rank / broadcast / dtype / value-range coverage that the case is
designed to exercise.
"""
import ast
import csv
import glob
import json
import math
import os
import sys

CAP_TOTAL = 1_048_576   # max broadcasted elements per case (1M)
CAP_DIM   = 1024        # max per-dimension size after capping

# Safe namespace for eval fallback (nan/inf are not Python literals).
_EVAL_NS = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _parse_cell(raw, default):
    """Parse a TTK cell (tuple/dict literal), handling nan/inf/-inf."""
    if raw is None:
        return default
    s = str(raw).strip()
    if not s:
        return default
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            return eval(s, {"__builtins__": {}}, dict(_EVAL_NS))  # noqa: S307
        except Exception:
            return default


def _parse_attributes(raw):
    """Parse the attributes JSON cell, converting 'nan'/'inf' strings to floats."""
    s = str(raw or "").strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return {}
    for k, v in d.items():
        if isinstance(v, str):
            low = v.lower()
            if low == "nan":
                d[k] = float("nan")
            elif low == "inf":
                d[k] = float("inf")
            elif low == "-inf":
                d[k] = float("-inf")
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
    return d


def _elems(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _broadcast(a, b):
    """Broadcast two shapes (right-aligned, numpy rules)."""
    na, nb = len(a), len(b)
    n = max(na, nb)
    out = []
    for i in range(n):
        da = a[na - n + i] if i >= n - na else 1
        db = b[nb - n + i] if i >= n - nb else 1
        if da != db and da != 1 and db != 1:
            out.append(max(da, db))  # mismatch (error case); keep larger
        else:
            out.append(max(da, db))
    return out


def _cap_broadcast(bshape):
    """Cap a broadcast shape: per-dim cap then total cap (halve largest)."""
    capped = [min(d, CAP_DIM) if d > 1 else 1 for d in bshape]
    while _elems(capped) > CAP_TOTAL and any(d > 1 for d in capped):
        max_idx, max_val = -1, 1
        for i, d in enumerate(capped):
            if d > max_val:
                max_val, max_idx = d, i
        if max_idx < 0:
            break
        capped[max_idx] = max(1, capped[max_idx] // 2)
    return capped


def _reconstruct(orig, bcapped):
    """Map capped broadcast shape back to a tensor, keeping dim=1 where original was 1."""
    if not orig:
        return orig  # scalar stays scalar
    n_orig, n_bc = len(orig), len(bcapped)
    result = []
    for i in range(n_orig):
        bc_idx = n_bc - n_orig + i
        if orig[i] == 1:
            result.append(1)
        else:
            result.append(bcapped[bc_idx] if bc_idx >= 0 else orig[i])
    return result


def cap_case_shapes(input_shapes, output_shapes):
    """Cap shapes preserving rank + broadcast pattern.

    For softplus backward: input[0]=grad, input[1]=self, output=broadcast(grad,self).
    Returns (capped_input_shapes, capped_output_shapes).
    """
    if len(input_shapes) < 2:
        return input_shapes, output_shapes

    grad_s = list(input_shapes[0]) if input_shapes[0] else []
    self_s = list(input_shapes[1]) if input_shapes[1] else []
    g = grad_s if grad_s else [1]
    s = self_s if self_s else [1]
    bshape = _broadcast(g, s)

    if _elems(bshape) <= CAP_TOTAL:
        return input_shapes, output_shapes

    capped_b = _cap_broadcast(bshape)
    new_grad = _reconstruct(grad_s, capped_b)
    new_self = _reconstruct(self_s, capped_b)
    new_inputs = [new_grad, new_self] + list(input_shapes[2:])

    new_outputs = []
    if output_shapes:
        new_outputs = [list(capped_b)]
    return new_inputs, new_outputs


def row_to_case(row):
    """Convert one CSV row to a manifest case dict."""
    shapes = list(_parse_cell(row.get("tensor_view_shapes"), ()))
    dtypes = list(_parse_cell(row.get("tensor_dtypes"), ()))
    out_idx_raw = _parse_cell(row.get("output_tensor_indexes"), ())
    out_idx = set(out_idx_raw) if out_idx_raw else set()

    # L2 exception cases have empty output_tensor_indexes: default last tensor as output.
    if not out_idx and len(shapes) >= 3:
        out_idx = {len(shapes) - 1}

    in_ranges = list(_parse_cell(row.get("input_data_ranges"), ()))
    attrs = _parse_attributes(row.get("attributes"))

    input_shapes, output_shapes = [], []
    input_dtypes, output_dtypes = [], []
    for i, sh in enumerate(shapes):
        shape = list(sh) if isinstance(sh, (list, tuple)) else [sh]
        dt = dtypes[i] if i < len(dtypes) else ""
        if i in out_idx:
            output_shapes.append(shape)
            output_dtypes.append(dt)
        else:
            input_shapes.append(shape)
            input_dtypes.append(dt)

    # Normalize data ranges to [[lo, hi], ...] for each input.
    norm_ranges = []
    for r in in_ranges:
        if isinstance(r, (list, tuple)):
            vals = list(r)
        else:
            vals = [r]
        norm_ranges.append([float(v) for v in vals])

    # Pad ranges if fewer than inputs.
    while len(norm_ranges) < len(input_shapes):
        norm_ranges.append([-1.0, 1.0])

    # Shape capping (preserves rank + broadcast pattern).
    input_shapes, output_shapes = cap_case_shapes(input_shapes, output_shapes)

    # Detect expect_error: L2 exception cases (name contains 'exception' or
    # original output_tensor_indexes was empty).
    name = row.get("testcase_name", "")
    expect_error = ("exception" in name.lower()) or not _parse_cell(
        row.get("output_tensor_indexes"), ()
    )

    return {
        "case_id": name,
        "op_name": row.get("api_name", ""),
        "input_shapes": input_shapes,
        "input_dtypes": input_dtypes,
        "input_formats": ["ND"] * len(input_shapes),
        "output_shapes": output_shapes,
        "output_dtypes": output_dtypes,
        "output_formats": ["ND"] * len(output_shapes),
        "attributes": attrs,
        "input_data_ranges": norm_ranges,
        "precision_tolerances": [],
        "absolute_precision": None,
        "is_enabled": True,
        "expect_error": expect_error,
        "remark": "",
    }


def main(argv):
    if len(argv) < 3:
        sys.stderr.write(
            "用法: gen_blackbox_manifest.py <blackbox_testcases_dir> <out_manifest.json> [golden.py]\n"
        )
        return 2

    cases_dir = argv[1]
    out_path = argv[2]
    golden_path = argv[3] if len(argv) > 3 else None

    csv_files = sorted(glob.glob(os.path.join(cases_dir, "*_test_cases.csv")))
    if not csv_files:
        sys.stderr.write(f"FATAL: no *_test_cases.csv in {cases_dir}\n")
        return 1

    cases = []
    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cases.append(row_to_case(row))

    manifest = {
        "schema_version": 1,
        "source": "blackbox L0+L1+L2",
        "op_name": "aclnnSoftplusBackward",
        "golden_path": golden_path,
        "case_count": len(cases),
        "cases": cases,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=True, ensure_ascii=False)
        f.write("\n")

    # Stats
    n_err = sum(1 for c in cases if c["expect_error"])
    print(f"wrote {len(cases)} cases ({n_err} expect_error) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

## 设计决策

- **cap 广播 shape 再反向重构输入，而不是各自独立 cap**：独立 cap 两输入会破坏广播对齐关系（对应维不再相等或为 1）；以广播 shape 为锚重构，保证 cap 后输入仍严格可广播到 cap 后输出。
- **"最大维减半"而不是"按比例缩"**：整数域上减半保持整除性与 2 的幂倾向，且自然终止（任一维到 1 后停止）；按比例缩会引入取整误差破坏广播维相等。
- **`allow_nan=True` 写 JSON**：值域因子含 inf/nan（如 `[[inf, inf], [0.0, 0.0]]`），Python `json.dumps` 会产出 `Infinity/NaN` 字面量——下游 C++ 解析器必须能吃这两个 token（本框架的自研 JSON parser 已处理）。
- **L2 异常用例 `output_tensor_indexes` 为空时默认取最后一张张量为输出**：CSV 里这类行没有显式输出索引，取末张是黑盒生成器的隐含约定。
- **CAP_TOTAL=1M 的取值**：兼顾设备内存（549 条全量连跑不 OOM）与覆盖率（尾块/非对齐仍有足够元素），非硬性最优。
- **当前方案路线档位**：`recommended_fastpath`。

## 系统信息

- CANN 9.1.0 / Ascend 950；黑盒 CSV 位于 `tests/blackbox/testcases/`（L0 41 行含头、L1 501 行、L2 10 行）。
- 产物：`tests/aclnn/aclnn_cpp_cases.json`（约 573KB，549 条）。
- 运行：`python3 gen_blackbox_manifest.py ../blackbox/testcases aclnn_cpp_cases.json <golden.py 路径>`。

## 验证信号

本卡信号来自该黑盒用例纳入框架的执行闭环：

- 形状规模统计：343 条 >1M 元素、最大 2,103,307,908（来源平台：950；证据等级：tool_observed）。
- 生成器输出：`wrote 549 cases (9 expect_error) -> aclnn_cpp_cases.json`，manifest 条数与 CSV 数据行数（40+500+9）一致、零丢失（来源平台：950；证据等级：tool_observed）。
- capping 抽查：原始 2.1B 元素的 7D 广播用例封顶后为 6,144 元素，rank=7 与广播维模式（哪些维为 1）保持不变（来源平台：950；证据等级：tool_observed）。
- cap 后全量上板：549/549 PASSED、0 FAILED、0 SKIPPED、退出码 0（3 次重复运行一致）（来源平台：950；证据等级：tool_observed）。

## 预防

- 接收外部（黑盒/随机）用例集先跑一遍元素数分布统计（min/median/max、>1M/>10M/>100M 计数），超限规模在设计阶段就规划 capping，而不是跑到第 N 条 OOM 才发现。

## 失效边界

- **已知 wart**：`cap_case_shapes` 对 L2 异常用例会把输出形状替换为 cap 后的广播形状（原始第 3 张量形状被丢弃）。对"期待拒绝"的用例无碍（输出形状不影响拒绝判定），但若某异常用例被算子接受，数值验证按广播 shape 做而非原始输出 shape——本批 3 条被接受用例的经验证结果一致，未构成问题，但语义上不严谨。
- 广播不相容的对（L2 中 mismatch 维）按"取较大维"近似处理，capping 不修复也不检测不相容性。
- `CAP_TOTAL`/`CAP_DIM` 针对本机设备内存与全量连跑时长选取；换设备或用例集更大时需重估。
- 三输入以上算子：`cap_case_shapes` 只 cap 前两输入与输出（本算子为双输入），多输入算子需扩展。
- 其他平台/CANN 版本未实测，capping 机制与芯片无关（上限取值与设备内存相关），推断可迁移（未验证）。

**未决项**：

- L2 异常用例输出形状被广播 shape 覆盖的 wart 未修（见失效边界第一条）；若后续出现"被接受且输出 shape 敏感"的异常用例需回头修 `cap_case_shapes` 保留原始输出。
- capping 对尾块/非对齐覆盖率的影响未量化（1M 上限下非对齐尾巴占比变化未测量）。
## 坏实践

- 为让用例能跑而删掉大 shape 用例——因子覆盖直接丢失。
- 只 cap 不保广播模式（如全维等比缩）——cap 后输入不再可广播，用例失去设计意义。

## 误诊

无。
