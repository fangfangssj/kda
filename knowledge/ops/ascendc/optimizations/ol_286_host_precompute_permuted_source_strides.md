---
type: Optimization
title: "Precompute permuted source strides on host"
description: "Encode a static permutation as permuted source strides so the device directly ravels output coordinates without an inverse-permutation search."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-286-host-precompute-permuted-source-strides.md
    role: primary
created_at: "2026-07-29T00:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: []
---

# Precompute permuted source strides on host

> 原 OKF v1 卡号：**OL-286**。

`applies_to: soc=all; cann=all; op_class=permute/transpose/layout materialization; backend=ascendc`

## Principle

For output coordinate `o`, precompute `permuted_src_stride[k] = src_stride[dims[k]]` on the host. The kernel then evaluates `src_flat = sum(o[k] * permuted_src_stride[k])` directly. This avoids a per-element inverse-permutation lookup and keeps the device path independent of a variable-length `dims` tuple.

Validate that `dims` is a bijection of `[0, rank)`, combine with OL-285 when rank is padded, and use zero strides only for synthetic dimensions that must not contribute. The rule covers forward layout ops and backward transforms of gradient layouts.

**Evidence / provenance**: derived from historical card TR-OL-14. This form passed 149/149 12_Permute cases on 2026-05-17 across fp32/fp16/bf16 and rank-2/3/4 inputs. The stride equivalence is mathematical; the coverage is measured.
