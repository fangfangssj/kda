---
type: Runbook
title: "libtorch C++ API has no `at::kUInt8` — the uint8 dtype enum is `at::kByte`"
description: "libtorch C++ API has no `at::kUInt8` — the uint8 dtype enum is `at::kByte`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["at::kByte", "at::kUInt8"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-199 — EC-199: A5Ops additive extension

> 关联：同一 EC-199 主题已由既有卡 [ec_199_libtorch_c_api_has_no_at_kuint8_the_uint8_dtype.md](ec_199_libtorch_c_api_has_no_at_kuint8_the_uint8_dtype.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3；CANN 9.0.0；op_class=all（libtorch API 常量，预期与 arch 无关，但仅在 Ascend310P3 实测）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（moe_init_routing_v2 Phase C iter 3，2026-08-10，Ascend310P3）。

**EC-199** — libtorch C++ API has **no `at::kUInt8`**; the uint8 dtype enum is `at::kByte` (matches `torch.uint8`/legacy `ByteTensor`). `at::zeros(n, at::kUInt8)` fails to compile in the pybind layer → use `at::kByte`. Most other enums follow `k<Name>` (`kFloat32`,`kInt64`,`kBool`); `kByte` is the ATen-inherited outlier. libtorch API constant (expected arch-independent; verified 310P). moe_init_routing_v2 2026-08-10.

- Former A5Ops ID: `EC-199`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all`
`unverified_on: expected arch-independent (libtorch API constant, not a chip behavior) — verified only on Ascend310P3 so far; not yet measured on Ascend950PR / Ascend910_220x`

- **Error pattern**:
  ```
  error: 'kUInt8' is not a member of 'at'
  // or: no member named 'kUInt8' in namespace 'at'
  ```
- **Root cause**: the libtorch / torch_npu C++ tensor dtype enum does NOT expose `at::kUInt8`. The 8-bit-unsigned dtype is named `at::kByte` (matching `torch.uint8` / the legacy `torch.ByteTensor` on the Python side). Authors expecting "dtype name == enum name" write `at::kUInt8` and hit a compile error in the pybind layer.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  at::Tensor t = at::zeros({(int64_t)n}, at::kUInt8);
  // AFTER (compiles):
  at::Tensor t = at::zeros({(int64_t)n}, at::kByte);
  ```
  Same for the option-set form: `at::TensorOptions().dtype(at::kByte)`.
- **Detection**: grep `pybind11.cpp` / host shim for `kUInt8` → replace with `kByte`. (Most other dtype enums DO follow the `k<Name>` spelling — `kFloat32`, `kInt64`, `kBool` — `kByte` is the one historical outlier inherited from ATen.)
- **Evidence**: moe_init_routing_v2 (2026-08-10, Ascend310P3, CANN 9.0.0) Phase C iter 3 — `at::kUInt8` rejected; `at::kByte` compiled. Failures-ledger: `iter 3 | compile | novel`.
- **Cross-ref**: none (libtorch API surface, not AscendC). backend=ascendc (pybind layer).

---