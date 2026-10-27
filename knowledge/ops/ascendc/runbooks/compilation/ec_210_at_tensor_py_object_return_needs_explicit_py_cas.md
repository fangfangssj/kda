---
type: Runbook
title: "at::Tensor → py::object return needs explicit py::cast"
description: "pybind 返回 `py::object` 时每个 return 点都要显式 `py::cast(t)`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-210"]
---

# at::Tensor → py::object return needs explicit py::cast

**EC-210** — **`at::Tensor` → `py::object` return needs explicit `py::cast`**: a pybind11 wrapper declared `py::object run_X(...)` (variant returns: tensor / tuple / differently-constructed empty) gets "no viable conversion" at return statements — wrap every return site with `return py::cast(t);`. 22_Nonzero 310p build iter-2 2026-08-15 (2 sites). Predicted: `as_tuple` variants, empty-rank early-exits, optional aux outputs. Cross-ref [`OL-36`](../precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md). backend=ascendc.

`applies_to: soc=all (host-side pybind11 wrapper concern, silicon-independent); cann=n/a; bisheng=n/a; op_class=any pybind wrapper with a `py::object`-typed entry point (variant returns: tensor / tuple / differently-constructed empty)`

- **Error**: no viable conversion from `at::Tensor` to `py::object` at the return sites of a wrapper declared `py::object run_X(...)` — needed when the function must return either a real tensor or an empty tensor built via a different path.
- **Fix**: `return py::cast(t);` at every return site.
- **Evidence**: 22_Nonzero 310p build iter-2 (2026-08-15): both return sites wrapped (pre-fix 内部路径（不公开）pybind11.cpp:87,183), build PASS (`代码片段 22_Nonzero/pybind11.cpp.snippet.cpp（见文末）` — the shipped `py::cast` sites; `22_Nonzero/failures_ledger.md` iter 2).
- **Other instances (predicted)**: any wrapper returning polymorphic results — `as_tuple` variants, empty-rank early-exits, optional auxiliary outputs.
- Cross-ref [`OL-36`](../precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md) (pybind is metadata+alloc+launch only — a boundary typing detail of that layer). backend=ascendc.

## 内嵌代码片段

### 22_Nonzero/pybind11.cpp.snippet.cpp

```cpp
                "nonzero kernel int32 position math requires numel < 2^31, got ", numel);

    int32_t dims[4] = {1, 1, 1, 1};
    for (int32_t i = 0; i < rank; ++i) {
        dims[i] = static_cast<int32_t>(x.size(i));
    }
    const int64_t dLast = dims[rank - 1];
            } else if (isFp16) {
                NZ_TABLE_CALL(nonzero_count_fp16);
            } else if (isBf16) {
                NZ_TABLE_CALL(nonzero_count_bf16);
            } else if (isI32) {
                NZ_TABLE_CALL(nonzero_count_i32);
            } else {
```

## 适用场景

- **工程形态**：pybind 返回 `py::object` 时每个 return 点都要显式 `py::cast(t)`。。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
