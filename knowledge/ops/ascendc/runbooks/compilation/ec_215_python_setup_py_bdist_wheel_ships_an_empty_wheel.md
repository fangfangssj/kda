---
type: Runbook
title: "python setup.py bdist_wheel ships an EMPTY wheel unless the cmake-built .so is staged into build/lib.<plat>"
description: "`setup.py bdist_wheel` 不打 cmake 产的 .so，装出来是空 wheel——要在 build_ext.run 里把 .so 拷进 build_lib。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-215"]
---

# python setup.py bdist_wheel ships an EMPTY wheel unless the cmake-built .so is staged into build/lib.<plat>

**EC-215** — **`python setup.py bdist_wheel` ships an EMPTY wheel unless the cmake-built.so is staged into `build/lib.<plat>`** — install_lib packages its own tree, not `kernel/build/`; wheel builds "successfully" with no.so inside. Fix: copy the.so into `self.build_lib` in build_ext.run before super().run(). Detection: unzip -l the wheel. sort 310p 2026-08-18 (build iter 2). Cross-ref (the copy-only rule this is the staging half of). backend=ascendc.

`applies_to: soc=all (host-side wheel packaging); cann=all; bisheng=n/a; mode=benchmark-acceptance standalone-pybind lane; op_class=all`

- **Symptom**: `python setup.py bdist_wheel` completes "successfully" but the produced wheel contains NO `.so`; `pip install` of it then fails at import (or silently lacks the extension) even though the cmake build produced a working library.
- **Root cause**: setuptools' `install_lib` packages the `build/lib.<plat>` tree. A `build_ext` that leaves the cmake-built `.so` only in `kernel/build/` never puts it on that tree, so the wheel is built from nothing.
- **Fix**: in the custom `build_ext.run`, COPY the cmake-built `.so` into `self.build_lib` BEFORE calling `super().run()` (and keep setup.py copy-only — recompiling from setup.py double-builds and breaks the cmake↔wheel pairing, per CAND-BENCH-STANDALONE-KERNEL-PACKAGING).
- **Detection**: unzip -l the wheel — an empty or source-only file list after a "successful" bdist_wheel is this bug, not a cmake failure.
- **Evidence**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 2): empty-wheel trap hit and fixed by staging into `self.build_lib`; kernel shipped 26/26 bit-exact.
- **Cross-ref**: (the copy-only rule this EC is the staging-location half of). backend=ascendc.
