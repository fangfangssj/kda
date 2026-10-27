---
type: Runbook
title: "MIX_AIC_1_2 kernel 必须 include lib/matmul_intf.h——compile_op.py 注入的 clearWorkspace 在 alias-only 的 matmul.h 下无法解析"
description: "每个 MIX_AIC_1_2（cube+vec）kernel 构建都会因 compile_op.py 无条件注入的 matmul::clearWorkspace(workspace) 而失败，真实错误（no type named 'clearWorkspace' in namespace 'AscendC'）藏在 .build kernel_0/_error.log，ld.lld 缺 .o 只是症状；修法是 include lib/matmul_intf.h——它经 kernel_kfc.h 引入官方 inline 定义，alias-only 的 matmul.h 不带。"
tags:
- task.debugging
- topic.compilation
- topic.programming_model
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- a2
- a3
- "950"
aliases:
- EC-87
---

# MIX_AIC_1_2 kernel 必须 include lib/matmul_intf.h——compile_op.py 注入的 clearWorkspace 在 alias-only 的 matmul.h 下无法解析

Every MIX_AIC_1_2 kernel needs `#include "lib/matmul_intf.h"` — CANN `compile_op.py` injects `matmul::clearWorkspace(workspace)` into the AIC branch, and the alias-only `matmul.h` include leaves the name unresolved.

- **Symptom**: bisheng build of a MIX_AIC_1_2 kernel fails. The opc wrapper surfaces only `ld.lld: error: cannot open .../MatmulAllreduce_..._mix_aic_0.o` and `..._mix_aiv_0.o` (the per-core .o files were never produced), so the **real** bisheng compile error is invisible in stdout. The actual error lives in `.build/.../kernel_0/_error.log`: `error: no type named 'clearWorkspace' in namespace 'AscendC'` at the generated `_534124_kernel.cpp` line 50: `matmul::clearWorkspace(workspace);`. A TU that additionally lacks the `matmul` namespace alias fails with the alternate signature `matmul undeclared` — same root cause, a different missing link in the resolution chain.
- **Root cause**: `tbe/tikcpp/compile_op.py:446` unconditionally injects into every non-c310 MIX kernel's prelude:
  ```cpp
  #ifdef <MIX_CORE_MACRO>
      if constexpr (g_coreType == AscendC::AIC) {
          matmul::clearWorkspace(workspace);
      }
  #endif
  ```
  `matmul.h` declares only `namespace matmul = AscendC;` (the alias; 8.5.1 `lib/matmul/matmul.h:384`), so the call resolves to `AscendC::clearWorkspace` — whose official inline definition ships in `impl/adv_api/detail/kfc/kernel_kfc.h`, pulled in by `#include "lib/matmul_intf.h"` and NOT by `matmul.h`. In a TU that includes only `matmul.h`, the injected call fails name lookup.
- **Fix**: `#include "lib/matmul_intf.h"` in the kernel header. It provides BOTH the `namespace matmul = AscendC` alias AND the official `AscendC::clearWorkspace` inline definition (via `kernel_kfc.h`) — the injected call resolves to the intended framework implementation.
- **How to surface the hidden bisheng error**: the opc wrapper swallows the CCE compiler stderr and only emits the downstream ld.lld "cannot open .o" message. To see the real error, read `.build/op_kernel/<Op>_<soc>/kernel_0/_error.log` (written by `dump_build_log`) — grep for `: error:` (the ld.lld lines are the symptom; the `: error:` line above them is the cause). Do NOT waste cycles chasing the ld.lld message.
- **Scope**: fires on EVERY MIX_AIC_1_2 kernel (cube+vec), regardless of whether the op even uses matmul — `compile_op.py:443` gates only on `is_mix and not is_c310()`. Single-core (AIV-only or AIC-only) kernels are unaffected.
- **Cross-ref**: EC-39 (MatmulApiStaticTiling MM_CFG — the other MIX_AIC cube compile blocker), PB-42 (fp32-L0C→fp16-output Fixpipe precision — the precision companion to this compile fix), EC-58 (matmul_intf.h non-transitivity — the sibling include rule).
