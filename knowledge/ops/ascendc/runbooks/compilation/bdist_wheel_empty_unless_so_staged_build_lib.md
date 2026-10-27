---
type: Runbook
title: "`python setup.py bdist_wheel` ships an empty wheel unless the cmake-built `.so` is staged into `build/lib.<plat>`"
description: "`python setup.py bdist_wheel` ships an empty wheel unless the cmake-built `.so` is staged into `build/lib.<plat>`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-215 — EC-215: A5Ops additive extension

> 关联：同一 EC-215 主题已由既有卡 [ec_215_python_setup_py_bdist_wheel_ships_an_empty_wheel.md](ec_215_python_setup_py_bdist_wheel_ships_an_empty_wheel.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（宿主侧 wheel 打包）；CANN=all；mode=benchmark-acceptance standalone-pybind 通道；op_class=all。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（sort 310p，2026-08-18，build iter 2）。

**EC-215** — **`python setup.py bdist_wheel` ships an EMPTY wheel unless the cmake-built .so is staged into `build/lib.<plat>`** — install_lib packages its own tree, not `kernel/build/`; wheel builds "successfully" with no .so inside. Fix: copy the .so into `self.build_lib` in build_ext.run before super().run(). Detection: unzip -l the wheel. sort 310p 2026-08-18 (build iter 2). Cross-ref CAND-BENCH-STANDALONE-KERNEL-PACKAGING (the copy-only rule this is the staging half of). backend=ascendc.

- Former A5Ops ID: `EC-215`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side wheel packaging); cann=all; bisheng=n/a; mode=benchmark-acceptance standalone-pybind lane; op_class=all`

- **Symptom**: `python setup.py bdist_wheel` completes "successfully" but the produced wheel contains NO `.so`; `pip install` of it then fails at import (or silently lacks the extension) even though the cmake build produced a working library.
- **Root cause**: setuptools' `install_lib` packages the `build/lib.<plat>` tree. A `build_ext` that leaves the cmake-built `.so` only in `kernel/build/` never puts it on that tree, so the wheel is built from nothing.
- **Fix**: in the custom `build_ext.run`, COPY the cmake-built `.so` into `self.build_lib` BEFORE calling `super().run()` (and keep setup.py copy-only — recompiling from setup.py double-builds and breaks the cmake↔wheel pairing, per CAND-BENCH-STANDALONE-KERNEL-PACKAGING).
- **Detection**: unzip -l the wheel — an empty or source-only file list after a "successful" bdist_wheel is this bug, not a cmake failure.
- **Evidence**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 2): empty-wheel trap hit and fixed by staging into `self.build_lib`; kernel shipped 26/26 bit-exact.
- **Cross-ref**: CAND-BENCH-STANDALONE-KERNEL-PACKAGING (the copy-only rule this EC is the staging-location half of). backend=ascendc.
