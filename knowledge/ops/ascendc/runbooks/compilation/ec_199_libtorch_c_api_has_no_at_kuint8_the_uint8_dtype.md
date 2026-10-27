---
type: Runbook
title: "libtorch C++ API has no at::kUInt8; the uint8 dtype enum is at::kByte (matches torch.uint8/legacy ByteTensor)."
description: "libtorch 没有 `at::kUInt8`，uint8 的枚举是 `at::kByte`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-199"]
---

# libtorch C++ API has no at::kUInt8; the uint8 dtype enum is at::kByte (matches torch.uint8/legacy ByteTensor).

**EC-199** — libtorch C++ API has **no `at::kUInt8`**; the uint8 dtype enum is `at::kByte` (matches `torch.uint8`/legacy `ByteTensor`). `at::zeros(n, at::kUInt8)` fails to compile in the pybind layer → use `at::kByte`. Most other enums follow `k<Name>` (`kFloat32`,`kInt64`,`kBool`); `kByte` is the ATen-inherited outlier. libtorch API constant (expected arch-independent; verified 310P). moe_init_routing_v2 2026-08-10.

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
