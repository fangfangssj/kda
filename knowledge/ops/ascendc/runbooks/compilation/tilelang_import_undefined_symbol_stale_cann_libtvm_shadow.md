---
type: Runbook
title: "tilelang-ascend source build: `import tilelang` fails `undefined symbol: TVMGetLastPythonError` — a stale CANN-bundled `libtvm.so` shadows the fresh build"
description: "tilelang-ascend source build: `import tilelang` fails `undefined symbol: TVMGetLastPythonError` — a stale CANN-bundled `libtvm.so` shadows the fresh build"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-84"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-84 — EC-84: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all；CANN 9.0.0+；kernel_type=tilelang-ascend（从源码构建 tile-ai/tilelang-ascend（route-a 编译器）的场景）。
- 验证来源：a3 / Ascend910_9382 / CANN 9.1.0（route-a IL-PoC，2026-07-14）。

**EC-84** — building `tile-ai/tilelang-ascend` (route-a compiler) from source then `import tilelang` fails `undefined symbol: TVMGetLastPythonError` because a stale CANN-bundled `libtvm.so` shadows the fresh build. Fix: `export TVM_LIBRARY_PATH=<repo>/build/tvm`. Anchor: route-a IL-PoC a3 Ascend910_9382 / CANN 9.1.0 2026-07-14. Cross-ref CAND-TILELANG-ASCEND-SOURCE-BUILD-A3. backend=ascendc (kernel_type=tilelang-ascend).

- Former A5Ops ID: `EC-84`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=9.0.0+; bisheng=n/a; op_class=all; kernel_type=tilelang-ascend`
`verified_on: a3 / Ascend910_9382 / CANN 9.1.0 (route-a IL-PoC 2026-07-14)`

- **Symptom**: after building `tile-ai/tilelang-ascend` from source, the first `python3 -c "import tilelang"` (or running any example) aborts with `ImportError: ... undefined symbol: TVMGetLastPythonError`.
- **Root cause**: the process resolves a DIFFERENT (older, CANN-bundled) `libtvm.so` before the freshly-built one — e.g. a `cann-9.0.0`-era TVM on the default library path — whose ABI predates the symbol the new tilelang expects.
- **Fix**: `export TVM_LIBRARY_PATH=<repo>/build/tvm` (the directory holding the just-built `libtvm.so`) before importing, forcing tilelang's loader to the fresh TVM. Verify: `import tilelang` then a trivial `@tilelang.jit` compile succeeds.
- **Detection**: `python3 -c "import tvm; print(tvm.__file__)"` resolving under a CANN install (not the repo `build/`) while a fresh build tree exists = stale-lib shadow.
- **Cross-ref**: `patterns/unverified/candidates.md` CAND-TILELANG-ASCEND-SOURCE-BUILD-A3 (the full source-build recipe this env-var is part of).
