---
type: Runbook
title: "torch_npu FFT is DEAD on Ascend310P3 — `torch.fft.rfft/c2c` hits the complex-dtype gate, and `set_compile_mode(jit_compile=False)` only opens the gate before FFTc2c fails at aclnnMatmul 161002 (deterministic, all devices) — route FFT/DFT-class ops to the Path-A cpu-truth lane"
description: "torch_npu FFT 在 310P3 上整体不可用：complex dtype 门禁 + jit_compile=False 也只在 FFTc2c 处撞 aclnnMatmul 161002。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-128", set_compile_mode]
---

# torch_npu FFT is DEAD on Ascend310P3 — `torch.fft.rfft/c2c` hits the complex-dtype gate, and `set_compile_mode(jit_compile=False)` only opens the gate before FFTc2c fails at aclnnMatmul 161002 (deterministic, all devices) — route FFT/DFT-class ops to the Path-A cpu-truth lane

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9 (aarch64); op_class=fft/dft/spectral ops`
`verified_on: soc=Ascend310P3; cann=9.0.0; torch_npu 2.9 (23_HyenaFftSizePaddingRfft, 2026-09-05 — 4-device probe sweep, devices 0/1/2/3, AICore idle per shared-NPU check)`
`unverified_on: other CANN / torch_npu builds; soc=Ascend950PR, Ascend910 (FFT availability there unmeasured in this KB)`

- **Severity**: HIGH for reference routing (not a kernel bug — the vendor reference path is unusable on this SoC/build).
- **Symptom ladder**: (1) `torch.fft.rfft(x_npu)` raises the complex-dtype gate ("Current settings do not support Complex dtype"); (2) the error message's own suggested mitigation `torch.npu.set_compile_mode(jit_compile=False)` EXISTS and opens the gate — but FFT then fails deeper inside `_exec_fft` (the FFTc2c NPU-op-api wrapper): "NPU function error: call aclnnMatmul failed, error code 161002" — verbatim stack frame in the probe table; (3) identical 161002 on devices 0/1/2/3 — deterministic, not device-state noise. (`torch.npu.config.jit_compile` does not exist in this build — `_npuConfig` AttributeError; only the `set_compile_mode` form is real.)
- **Action**: do NOT burn iterations trying to make the NPU FFT reference run — the ladder above IS the C20 probe result (the error message's own suggested mitigation was tried; that is where it ends). Route FFT/DFT-class ops to the Path-A cpu-truth lane per PB-75 / OL-68 Case A, and declare direct-DFT AscendC kernels per OL-188 when no CANN NPU FFT reference exists to mirror.
- **Evidence**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05): 4-device probe table in `23_HyenaFftSizePaddingRfft/self_critic_report.md:16-18` (§Empirical probes); shipped Path-A kernel 49/49 PASS_T1 strict, beats the pocketfft-fp32 reference on 25/49 cases (`23_HyenaFftSizePaddingRfft/verification.json`).
- **Other instances (predicted)**: stft/irfft/fft2/hfft and spectral-window ops on this SoC; any op whose torch reference lowers through FFTc2c.
- **Cross-ref**: PB-75 (the nonzero sibling — same NPU-ref-dead → Path-A methodology), [`OL-68`](ol_68_some_torch_npu_fused_ops_are_unavailable_on_ascend.md) Case A, [`OL-89`](../precision/ol_89_when_the_active_reference_is_non_deterministic_loo.md)/[`OL-90`](../precision/ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) (CPU-truth topology), [`OL-188`](ol_188_op_family_architecture_requirement_classification.md) (VEC-only direct DFT declaration). backend=ascendc.
