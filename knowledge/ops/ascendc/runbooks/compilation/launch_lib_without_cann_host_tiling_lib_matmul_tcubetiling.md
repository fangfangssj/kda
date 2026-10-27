---
type: Runbook
title: "When the launch shared lib links only the device kernel + framework libs (not the CANN host tiling lib), a matmul-library cube kernel cannot compute its host TCubeTiling at the launch layer — scope-corrected 2026-09-05: environment-conditional, not structural"
description: "When the launch shared lib links only the device kernel + framework libs (not the CANN host tiling lib), a matmul-library cube kernel cannot compute its host TCubeTiling at the launch layer — scope-corrected 2026-09-05: environment-conditional, not structural"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-235"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-235 — OL-235: When the launch shared lib links only the device kernel + framework libs (not the CANN host tiling lib), a matmul-library cube kernel cannot compute its host TCubeTiling at the launch layer

## 适用场景

- 适用平台与范围：soc=Ascend950PR/V351; cann=9.1.T500; bisheng=n/a; op_class=CUBE_MIX (matmul/attention/conv/rnn/gmm); build=launch-shared-lib-harness (pybind/ACLRT_LAUNCH_KERNEL with a restricted link line)。
- 验证来源：soc=Ascend950PR; cann=9.1.T500 — deformable_conv2d port_a3 2026-06-20 (empirical pybind link-line probe + build_ascendc.py source read)。

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR/V351; cann=9.1.T500; bisheng=n/a; op_class=CUBE_MIX (matmul/attention/conv/rnn/gmm); build=launch-shared-lib-harness (pybind/ACLRT_LAUNCH_KERNEL with a restricted link line)`
`verified_on: soc=Ascend950PR; cann=9.1.T500 — deformable_conv2d port_a3 2026-06-20 (empirical pybind link-line probe + build_ascendc.py source read)`

> ⚠️ **SCOPE CORRECTION (main, 2026-09-05, Ascend950DT_9582 / CANN 9.1.0 / `build_ascendc.py` pybind harness). The CONCLUSION below still stands as the default, but the stated REASON is outdated and the word "structurally" is wrong. Verified by compile + device run, not by citation. Full entry: OL-508.**
>
> 1. **The pybind link line is not fixed.** At repo HEAD `4b8941b`, `build_ascendc.py:356` emits `kernels torch_npu m dl{link_libs_block}` (block built at `:307` from `merged.link_libs`), and `src/scripts/patches/build_capabilities/baseline_cann_host_tiling.py` (DEBT-110) appends `${CANN_NNOPBASE} ${CANN_OPAPI}` to `pybind11_lib` under `if(CANN_NNOPBASE AND CANN_OPAPI)`. That file's own docstring states the firing condition: *"On a clean-runtime lane the libs are absent and the DEBT-110 stub headers are used"*. **The discriminator is whether the CANN host libs are installed on the BUILD host** — not the harness. The 2026-06-20 anchor ran on a clean-runtime lane, so the original observation was correct *for that lane*; it is environment-conditional, not structural.
> 2. **A host-computed `TCubeTiling` is not required at all** — a kernel-local on-stack `TCubeTiling` drives the library, so "cannot compute its `TCubeTiling` at the launch layer" does not by itself imply the path is unbuildable.
>    ⛔ **Corrected citation.** An earlier draft cited `fa_matmul_poc_kernel.h:146-148` here. Read at ref `3638ca27`, `fa_matmul_poc/fa_matmul_poc_kernel.h:146` is the comment `// We pass nullptr tiling and rely on SetOrgShape + SetSingleShape + IterateAll.` and `:147-148` are `bmm1.Init(nullptr, pipe); bmm2.Init(nullptr, pipe)` — **nullptr tiling, the opposite of what was claimed**; that file does NOT support this point and the citation is withdrawn rather than reworded. The support is a standalone probe (fp16 vs `torch.matmul`, `max_abs 3.814697265625e-06`, verdict `CORRECT`) whose source is **not in this repo** — main lane `main_op42_fix_20260905/` (not in this repo): run receipt `MMRUN5.log` sha256 `8c348c8744395778`, driver `_mmrun5.sh` sha256 `228476c5da8a2c8b`, probe source `scripts/mmprobe.cpp` sha256 `87ee8cdcfd0f0d3e` (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` :19, on-stack `TCubeTiling tiling{}` :38, `IterateAll<true>` :58) and binding `scripts/mmprobe_bind.cpp` sha256 `7a4590cf3f92e40c`. ⚠️ Those hashes are of the **local staging copies**; the container copy at run time (2026-09-05 04:53Z) was never hashed, so byte-identity between the two is **not established** — the run's own log is the receipt, not the disk match. **In-repo evidence for this sub-claim is DANGLING** until that source is imported under a fixed hash.
> 3. **Measured refutation on this host.** A `matmul::Matmul<>` kernel BUILT (rc=0) and RAN through this harness: 64×64×64 fp16 vs torch, `max_abs 3.814697265625e-06`. The same pattern is in production on the same host in Triton's op-29 `29_FlashAttentionBwd/kernel/flashattentionbwd_kernel.h`.
> 4. **Where the real boundary sits.** `AIC_ONLY` + synchronous `IterateAll<true>` uses no KFC and runs here. `MIX` probes faulted (507035 / 507015), but the fault MECHANISM was NOT established — it must not be equated with the July 507014 KFC-deadlock anchor (that equation was an inference and is withdrawn). The KFC-bootstrap wall remains separately recorded as `CAND-KFC-standalone-bootstrap-teardown`.
> 5. **What replaces the withdrawn reason — and what does NOT.** The linkage justification is withdrawn. It is NOT replaced by my measurement: ported into 4-42 (CoTAttention) the library path was both less accurate (39/50 vs the manual candidate's 50/50 under the selected profile) and slower (full-50 event GM 0.149592 all-GEMM, 0.250848 mm1-only, vs manual 0.256397) — but that is **one op, one port, one host**, and it supports exactly one conclusion: *this port was not adopted for 4-42*. It cannot carry an op-wide default on its own. If the manual-`Mmad` default for cube-class ports is kept, it must stand on **P-P102's own independent, scoped evidence**, which this entry did not re-verify and does not vouch for. Treat the choice as **per-op, per-shape, and measured** (OL-508).

### Principle

A `matmul::Matmul<>` + `REGIST_MATMUL_OBJ` cube kernel needs a host-computed `TCubeTiling`, produced by the CANN host tiling lib (`matmul_tiling::MatmulApiTiling::GetTiling` from `tiling/tiling_api.h`). **When** a harness's launch shared library links ONLY the device kernel object + framework libs, the host tiling lib's symbols are not on that link line, so the `TCubeTiling` host computation cannot be performed at the launch layer and the matmul-library cube path is unbuildable **in that configuration**. ⚠️ **That condition is not a property of the harness — read the scope correction above before using this sentence.** `build_ascendc.py` appends `${CANN_NNOPBASE} ${CANN_OPAPI}` under `if(CANN_NNOPBASE AND CANN_OPAPI)` (DEBT-110, 2026-06-29), so on a build host where those libs are installed the link line **does** carry real tiling and this sentence does not apply. ⛔ **The rest of this paragraph previously read: "it generalizes beyond the FA numerical/deadlock evidence to ALL CUBE_MIX ports built through this harness. The fix is not 'find the tiling lib' — it is to use manual `Mmad`." That imperative is WITHDRAWN, not annotated** (OL-508 removed its premise: the library does build here). What is supported instead: **choose manual `Mmad` vs the matmul library per op and per shape, by measurement.** Whatever independent basis P-P102 has for its default is untouched; this entry no longer supplies one and does not vouch for it.

### Concrete anchor — the **libs-absent** side only (clean-runtime lane)

⚠️ **This block shows the `else()` branch, not the only branch.** It is what the autogen CMake emits on a
build host where the CANN host matmul-tiling libs are absent. Thirteen lines below it in the real
generated file sits the conditional that appends them when they are present. Do not read this snippet as
the harness's fixed link line — that misreading is what this card's 2026-09-05 scope correction exists to
prevent, and it has since caused at least one live architecture decision to rest on the withdrawn claim
(see the 2026-09-26 instance under Evidence).

`build_ascendc.py`'s autogen CMake, **on a clean-runtime lane**, links the pybind module against a
restricted set:
```cmake
add_library(pybind11_lib SHARED ".../pybind11.cpp")
target_link_libraries(pybind11_lib PRIVATE
  kernels torch_npu m dl)          # no host tiling lib (no MatmulApiTiling / libtiling)
```
### Which branch am I on? — the discriminator, and the wrong way to check it

⛔ **Do not answer this by grepping the generated `CMakeLists.txt`.** Both `message()` strings —
`"… found … CUBE precision enabled"` and `"… NOT found … CUBE precision deferred"` — are literals in that
file, so a grep returns **both** regardless of which branch ran, and both still carry unexpanded
`${ASCEND_CANN_PACKAGE_PATH}` / `${CANN_NNOPBASE}`. A reader who greps the source concludes nothing, and may
conclude the opposite of the truth.

✅ **Read `CMakeCache.txt` in the build dir and check whether `find_library` resolved:**

```
CANN_NNOPBASE:FILEPATH=/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so   <- resolved  => real tiling linked
CANN_OPAPI:FILEPATH=/usr/local/Ascend/cann-9.1.0/lib64/libopapi.so         <- resolved  => CUBE precision enabled
```

`…:FILEPATH=…-NOTFOUND` on either variable means the `else()` branch ran and the snippet above applies.
A `stat` of the two `.so` paths is a useful cross-check but is **not** the verdict: the verdict is what
`find_library` recorded at configure time for that build tree.

**Measured instance, 2026-09-26** (`Ascend950DT_9582`, CANN 9.1.0, container `…-164-d7-…`): both variables
resolved to `<pkg>/lib64`, so the conditional fired and **real tiling was linked** — i.e. this host is on the
libs-present side and the restricted-link-line snippet did **not** describe it. Recorded because the same day a
`port_a3` conv-class design cited this card's uncorrected half to reject the matmul-library route and take the
vector path instead; on `Ascend950DT` vector FP32 is 23–30 TFLOPS against cube TF32 189–243 (≈8x), so the cost
of that misreading is roughly an order of magnitude of headroom. **Falsifying the premise is not the same as
proving the library route builds and passes here — that was not demonstrated and is not claimed.**

`host_config.cmake` / `host_intf.cmake` are pulled in by `ascendc.cmake` for the DEVICE `ascendc_library(kernels)` target, not for the pybind target. Remote probe (CANN 9.1.T500): `tiling_api.h` + `libplatform.so` are present, but no host `libtiling` exporting `MatmulApiTiling` is on the pybind link path.

### Scope distinction vs OL-220 (read before assuming a contradiction)

OL-220 shows host `MultiCoreMatmulTiling` / `TCubeTiling` (from `tiling/matrix/bmm_tiling.h`) building clean (chunk_gated_delta_rule, 122/122) — but through the **hand-authored `ascendc_library` flow with a separately-linked `op_host` target** that DOES get the host tiling include/link path. The two are not in conflict: host matmul tiling CAN link when you author the host target's link line yourself (OL-220 flow); it does NOT link when the launch lib's link line is fixed by `build_ascendc.py` to `kernels torch_npu m dl` (this entry). The discriminator is WHERE the `TCubeTiling` computation lives and whether that target's link line includes the tiling lib.

### Evidence
- deformable_conv2d (modulated DCNv2, CUBE_MIX) port_a3 2026-06-20 (Ascend950PR / CANN 9.1.T500): worker ran an empirical pybind-harness linkage probe BEFORE committing to a matmul-library port (Probe-Before-Fix), found `MatmulApiTiling` absent from the pybind link line, and routed to the manual-Mmad cube + bilinear-deform vec MIX scaffold (FA-class structural_rewrite verdict). Confirmed against source: `src/scripts/patches/build_ascendc.py` link line is literally `kernels torch_npu m dl`.

### Other instances (predicted)
- Every cube-required `port_a3_to_a5` op (matmul / attention / conv / rnn / gmm / ffn family, tagged CUBE_MIX) built through the `build_ascendc.py` pybind/ACLRT_LAUNCH harness — the restricted link line is harness-level, not op-specific, so the manual-Mmad default applies uniformly.
- Cross-ref: [P-P102](../../examples/cube_vector_fusion.md) (cube-MIX manual-Mmad scaffold — this entry is its harness-linkage root cause), OL-188 (cube-required ≠ pure-vec), OL-200 (MIX staggered cube/vec schedule), OL-220 (the `ascendc_library` flow where host tiling DOES link — scope distinction above).
