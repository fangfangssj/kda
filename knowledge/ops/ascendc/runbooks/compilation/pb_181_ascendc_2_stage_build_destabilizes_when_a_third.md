---
type: Runbook
title: "ascendc 2-stage build destabilizes when a THIRD kernel entry is appended to a `*_kernels.cpp`"
description: "ascendc 两段式构建在 `*_kernels.cpp` 追加第三个 kernel 入口时不稳定：make 规则产出 HOST ELF 而 flags.make 重放同命令产 DEVICE ELF，链接即炸。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-181"]
---

# ascendc 2-stage build destabilizes when a THIRD kernel entry is appended to a `*_kernels.cpp`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=any *_kernels.cpp with >=3 kernel entries`
**PB-181 — ascendc 2-stage build destabilizes when a THIRD kernel entry is appended to a `*_kernels.cpp`** — the make-rule m200_obj compile emits a HOST ELF (358464B) where the byte-identical command replayed from flags.make emits the DEVICE ELF (568992B); `ld.lld -m aicorelinux` fails "unknown file type"; clean-prefix rebuild does not resolve; reverting to two entries restores a byte-identical working binary. 15_Pad ko-8 Opt8 bf16-int16 (2026-09-01, reverted). Candidate DEBT for the build-chain owners. backend=ascendc.
