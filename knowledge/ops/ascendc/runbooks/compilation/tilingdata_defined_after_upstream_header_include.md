---
type: Runbook
title: "`unknown type name '<Op>TilingData'` on an upstream `op_kernel/*.h` line means the tiling struct is defined too late — lift it into a header included first"
description: "`unknown type name '<Op>TilingData'` on an upstream `op_kernel/*.h` line means the tiling struct is defined too late — lift it into a header included first"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["TilingData"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-200 — EC-200: A5Ops additive extension

> 关联：同一 EC-200 主题已由既有卡 [ec_200_unknown_type_name_op_tilingdata_flagged_on_an_up.md](ec_200_unknown_type_name_op_tilingdata_flagged_on_an_up.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3；CANN 9.0.0；op_class=all（任何包装引用宿主侧 tiling struct 的上游 op_kernel/*.h 的 Mode-B / port_a3 移植）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（moe_init_routing_v2 Phase C iter 1，2026-08-10）。

**EC-200** — `unknown type name '<Op>TilingData'` flagged on an upstream `op_kernel/*.h` line means the tiling struct is **defined too late** (in the .cpp, after the include). When wrapping 220x headers that reference it (Mode-B / port_a3 authoring), the struct must be VISIBLE (fully defined) at the include site — a forward-decl is insufficient. Fix: lift the struct into its own `<op>_tiling.h` `#include`d BEFORE the upstream header. Orthogonal to OL-77 (GM read) / OL-178 (layout) / EC-40 (POD size) — this is the include-ORDER rule. moe_init_routing_v2 2026-08-10.

- Former A5Ops ID: `EC-200`
- Fusion disposition: `additive_new`

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
- **Cross-ref**: OL-77 (GM tiling struct field-by-field READ — kernel-side rule), OL-178 (host↔device tiling struct LAYOUT / `pragma pack`), EC-40 (cube tiling host POD size mismatch). This entry is the DEFINITION-ORDER / include-order rule, orthogonal to those. backend=ascendc.
