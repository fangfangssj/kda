---
type: Runbook
title: "A compiler flag consumed in the wrong language mode is silently ignored and the build still emits a plausible object — probe option acceptance only in the production mode, with a rejection control"
description: "A compiler flag consumed in the wrong language mode is silently ignored and the build still emits a plausible object — probe option acceptance only in the production mode, with a rejection control"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-314"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-314 — OL-314: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=clang-15.0.5; op_class=all。
- 验证来源：soc=Ascend950PR; cann=9.1.0。

**OL-314** — **A compiler flag consumed in the WRONG LANGUAGE MODE is silently ignored, and the build still emits a plausible object.** Measured on the A5 PR box: `bisheng --cce-aicore-arch=dav-3510 -c x.cpp` exits 0 with a 1008-byte object, while the same arch under the real device mode (`--cce-aicore-only -x cce`) errors `Unsupported CCE architecture: dav-3510`. Compiled as plain C++ the flag means nothing and is dropped — so **the failure presents as a success**, and re-reading the command cannot reveal it; only exercising the production mode can. That false positive survived several rounds of downstream work. ⭐ **A5's CCE arch tokens are the `dav-c310` family** (`dav-c310`/`-cube`/`-vec`), NOT `dav-3510` ⇒ `__NPU_ARCH__ == 3510` (used INSIDE the source) and `--cce-aicore-arch=dav-c310` (used BY the compiler) are **two naming systems for the same chip**, not interchangeable; in-repo `regbase_matmul.h` says "C310 regbase" while its sources branch on `__NPU_ARCH__ >= 3510`. **How to apply**: probe option-acceptance only in the mode the real build uses (same `-x`/mode flags), and pair every acceptance probe with a **rejection control** — push deliberately broken source through the identical command and confirm it fails with no artefact; without it, "an object appeared" cannot distinguish a working config from one where the interesting flags were discarded. Records the working A5 device-compile line, incl. two include roots in DIFFERENT trees (`kernel_tiling/` lives under `ascendc/include/highlevel_api`) and a forwarding shim for `lib/matmul_intf.h` (CANN B103 layout) → `adv_api/matmul_intf.h` (9.1.0) — shim rather than edit, so a currently-passing header is not forked.

- Former A5Ops ID: `OL-314`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=clang-15.0.5; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.0`
`unverified_on: soc=Ascend910_9382 (220x) — the arch tokens differ there, the SHAPE of the trap is what transfers`
**Principle.** When a toolchain has more than one language mode, a mode-specific flag handed to
the *wrong* mode is not an error — it is **ignored**. The build then succeeds, emits an object
of a believable size, and the object is not what was asked for. Because the failure surfaces as
a **success**, no amount of re-reading the command reveals it; only exercising the real mode does.
**Measured instance (2026-07-31, the only A5 PR box).** Probing which arch `bisheng` accepts:
```
bisheng --cce-aicore-arch=dav-3510 -c probe.cpp -o probe.o          -> exit 0, 1008-byte object
bisheng --cce-aicore-arch=dav-3510 --cce-aicore-only -x cce ...     -> error: Unsupported CCE
                                                                       architecture: dav-3510
```
The first form compiles as ordinary C++, where `--cce-aicore-arch` means nothing, so it is
dropped. Reading that exit-0 as "this arch is supported" is a **false positive that survived
several rounds of downstream work** before the real mode exposed it.
⇒ **A5's CCE arch tokens are the `dav-c310` family** (`dav-c310`, `dav-c310-cube`,
`dav-c310-vec`) — **not** `dav-3510`. Note the consequence: `__NPU_ARCH__ == 3510` (the token
used *inside* the source) and `--cce-aicore-arch=dav-c310` (the token used *by the compiler*)
are **two different naming systems for the same chip**, and neither can be substituted for the
other. In-repo corroboration: `regbase_matmul.h` heads its load helpers "C310 regbase L1->L0
loads" while the same project's sources branch on `__NPU_ARCH__ >= 3510`.
**How to apply.** Probing whether a toolchain accepts an option is only valid **in the mode the
real build uses**. Concretely: pass the same language-mode flags the production build passes
(`-x cce --cce-aicore-only` here), and pair every acceptance probe with a **rejection control** —
feed deliberately broken source through the identical command and confirm it fails and emits no
artefact. Without that control, "it produced an object" cannot distinguish a working
configuration from one where the interesting flags were discarded.
**Working A5 device-compile configuration** (recorded because it is otherwise expensive to
rediscover, and because two of its include roots live in *different trees*):
```
-x cce --cce-aicore-only -std=c++17 --cce-aicore-arch=dav-c310-cube   (or -vec)
-I<asc>                                  # headers reach impl/ via ../../impl
-I<asc>/include{,/basic_api,/interface,/adv_api,/tensor_api,/utils,/tiling}
-I<...>/ascendc/include/highlevel_api    # kernel_tiling/kernel_tiling.h — a DIFFERENT tree
```
⚠️ Older in-repo kernels include `"lib/matmul_intf.h"` (the CANN B103 layout); cann-9.1.0 ships
it at `adv_api/matmul_intf.h`. Prefer a forwarding shim over editing headers that are currently
passing — editing them forks a file whose value is that it is known-good.
