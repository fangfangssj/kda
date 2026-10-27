---
type: Runbook
title: "catlass 坐标/形状等基础类型的头文件在库顶层（catlass/gemm_coord.hpp），不在 catlass/gemm/ 子目录下"
description: "GemmCoord/GemmShape 等坐标/形状基础小类型定义在 catlass include 树的顶层头文件（镜像 CUTLASS 的 cutlass/gemm_coord.hpp 布局）；catlass/gemm/ 子树放的是大型模板机制——猜子路径会在 include 处失败。拿不准符号在哪个头时，从 catlass 示例的 include 块抄，不要凭目录结构直觉。"
tags:
- task.debugging
- topic.compilation
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- agnostic
aliases:
- EC-93
---

# catlass 坐标/形状等基础类型的头文件在库顶层（catlass/gemm_coord.hpp），不在 catlass/gemm/ 子目录下

Catlass coordinate/shape vocabulary headers live at the library TOP LEVEL (`catlass/gemm_coord.hpp`), NOT under `catlass/gemm/` — a guessed subpath fails at the include site.

- **Symptom**: compile fails on `#include "catlass/gemm/gemm_coord.hpp"` (or similar guessed subpath) — no such header. The small vocabulary types (`GemmCoord`, `GemmShape`, and siblings) are TOP-LEVEL headers in the catlass include tree, mirroring CUTLASS's `cutlass/gemm_coord.hpp` layout; the `catlass/gemm/` subtree holds the larger template machinery, not these types.
- **Fix**: include from the top level. The include list in `examples/00_basic_matmul/basic_matmul.cpp` is the authoritative layout — when unsure which header a catlass symbol lives in, copy from the example's include block, not from intuition about the directory structure.
- **Concrete anchor**:
  ```cpp
  #include "catlass/gemm_coord.hpp"
  // ...
  Catlass::GemmCoord problemShape(m_, n_, kShard_);
  ```
- **Cross-ref**: OL-297 (catlass Resource memory model — the other catlass-integration rule), EC-92 (catlass Resource lifetime trap).
