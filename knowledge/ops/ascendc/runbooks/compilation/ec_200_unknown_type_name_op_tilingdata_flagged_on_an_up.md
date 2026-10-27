---
type: Runbook
title: "unknown type name '<Op>TilingData' flagged on an upstream op_kernel/.h line means the tiling struct is defined"
description: "报 `unknown type name '<Op>TilingData'` 说明 tiling 结构体定义太晚（在 .cpp 里、include 之后）；要提成独立 `_tiling.h` 前置包含。"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-200"]
---

# unknown type name '<Op>TilingData' flagged on an upstream op_kernel/.h line means the tiling struct is defined

**EC-200** — `unknown type name '<Op>TilingData'` flagged on an upstream `op_kernel/*.h` line means the tiling struct is **defined too late** (in the .cpp, after the include). When wrapping 220x headers that reference it (Mode-B / port_a3 authoring), the struct must be VISIBLE (fully defined) at the include site — a forward-decl is insufficient. Fix: lift the struct into its own `<op>_tiling.h` `#include`d BEFORE the upstream header. Orthogonal to OL-77 (GM read) / OL-178 (layout) / EC-40 (POD size) — this is the include-ORDER rule. moe_init_routing_v2 2026-08-10.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any port that wraps upstream op_kernel/*.h referencing a host-authored tiling struct)`

- **Error pattern**:
  ```
  error: unknown type name 'MoeInitRoutingV2TilingData'   // (or '<Op>TilingData')
  ```
  flagged on an upstream `op_kernel/*.h` line that uses the tiling struct, even though the struct IS defined later in the same `.cpp`.
- **Root cause**: when porting by wrapping upstream 220x `op_kernel/*.h` headers directly (Mode-B / `port_a3` authoring), those headers reference the op's `TilingData` struct in their function signatures. C++ requires the type to be VISIBLE (fully defined, not merely forward-declared) at the point a header uses it by value or complete type. If the struct definition lives in the `.cpp` AFTER the `#include` of the upstream header, the header sees an undefined type → `unknown type name`.
- **Fix**: move the tiling struct definition into its own small header and `#include` it BEFORE any upstream header that references it:
  ```cpp
  // moe_init_routing_v2_tiling.h  — define FIRST
  struct MoeInitRoutingV2TilingData { /* ...fields... */ };

  // then the .cpp / kernels.cpp:
  #include "moe_init_routing_v2_tiling.h"          // before 220x headers
  #include "op_kernel/moe_v2_init_routing_fullload.h"  // references the struct
  ```
  A bare forward declaration (`struct X;`) is NOT sufficient where the header needs the complete type.
- **Detection**: `unknown type name '<Op>TilingData'` pointing at an upstream `.h` line → the struct is defined too late; lift it into a header included before the offender.
- **Evidence**: moe_init_routing_v2 (2026-08-10, Ascend310P3, CANN 9.0.0) Phase C iter 1 — 内部路径（不公开）moe_v2_init_routing_fullload.h:29 unknown type MoeInitRoutingV2TilingData`; fixed by moving tiling structs to `moe_init_routing_v2_tiling.h` included before the 220x headers. Failures-ledger: `iter 1 | compile | novel`.
- **Cross-ref**: [`OL-77`](ol_77_gm_tiling_struct_cannot_be_copied_wholesale_must_s.md) (GM tiling struct field-by-field READ — kernel-side rule), [`OL-178`](ol_178_host_device_tiling_struct_layout_mismatch_from_mix.md) (host↔device tiling struct LAYOUT / `pragma pack`), [`EC-40`](ec_40_cube_tiling_host_pod_size_mismatch_tcubetiling_is.md) (cube tiling host POD size mismatch). This entry is the DEFINITION-ORDER / include-order rule, orthogonal to those. backend=ascendc.

**孪生卡 ec-268 合并的独有内容**（ec-268-op-tilingdata-must-be-defined-before-any-header.md 已删除）：

> error: unknown type name 'MoeInitRoutingV2TilingData' // (or '<Op>TilingData')

> #include "moe_init_routing_v2_tiling.h" // before 220x headers

> #include "op_kernel/moe_v2_init_routing_fullload.h" // references the struct

## 适用场景

- **工程形态**：报 `unknown type name '<Op>TilingData'` 说明 tiling 结构体定义太晚（在 .。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
