---
type: Runbook
title: "`<c10/npu/NPUStream.h>` is not on the pybind TU include path on this build chain — use `<torch_npu/csrc/core/npu/NPUStream.h>`"
description: "`<c10/npu/NPUStream.h>` 的 c10 子路径没导出到扩展 include 路径——要用 torch_npu 全路径 include。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-288"]
---

# `<c10/npu/NPUStream.h>` is not on the pybind TU include path on this build chain — use `<torch_npu/csrc/core/npu/NPUStream.h>`

`applies_to: soc=Ascend310P3 (dav-m200 benchmark lane); cann=9.0.0; bisheng=n/a; op_class=all (pybind TU compile)`
`unverified_on: other build chains (A5 acceptance lane) — the c10 path may resolve there; check before reuse`
- **Symptom**: pybind TU fails to compile with a not-found include on `<c10/npu/NPUStream.h>` (the header exists in the torch_npu source tree but its `c10/...` subpath is not exported to the extension include path on this build chain).
- **Fix**: include the cousin-proven full path `<torch_npu/csrc/core/npu/NPUStream.h>` (plus `pybind11/pybind11.h`, `torch/extension.h`, `acl/acl_rt.h` for parity with prior lane kernels).
- **Evidence**: 32_AdaptiveAvgPool3d 310p (2026-08-21, benchmark, failures_ledger build iter 1): iter-1 build failed on the `c10/` include; switched to the `torch_npu/csrc/core/npu/` path and the TU compiled clean — same path every prior dav-m200 benchmark op used (e.g. `代码片段 26_AvgPool3d/pybind11.core.snippet.cpp（见文末）`). Fixed form at `代码片段 32_AdaptiveAvgPool3d/pybind11.cpp.snippet.cpp（见文末）`. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`; this is a compile-level fact.)
- **Other instances (predicted)**: any new pybind TU on this lane that copies an A5-era template's include block verbatim. backend=ascendc.

## 内嵌代码片段

### 26_AvgPool3d/pybind11.core.snippet.cpp

```cpp
}

PYBIND11_MODULE(_avgpool3d_ext, m) {
    m.def("run_avg_pool3d", &run_avg_pool3d,
          py::arg("input"), py::arg("kernel_size"),
          py::arg("stride") = py::none(), py::arg("padding") = 0,
          py::arg("ceil_mode") = false, py::arg("count_include_pad") = true,
          py::arg("divisor_override") = py::none(),
          "AvgPool3d AscendC kernel (Ascend310P3, dav-m200)");
}
```

### 32_AdaptiveAvgPool3d/pybind11.cpp.snippet.cpp

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
```
