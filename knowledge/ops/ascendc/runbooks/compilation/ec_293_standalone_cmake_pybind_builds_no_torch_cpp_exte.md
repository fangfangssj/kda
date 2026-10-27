---
type: Runbook
title: "standalone-cmake pybind builds (no torch cpp_extension env) leave `TORCH_EXTENSION_NAME` undefined — `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` registers a literal module name the wheel cannot import"
description: "cmake 产 .so 装上但 import 失败：PyInit 模块名不匹配——用字面量模块名，别依赖未定义的宏。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-293", "TORCH_EXTENSION_NAME", "PYBIND11_MODULE"]
---

# standalone-cmake pybind builds (no torch cpp_extension env) leave `TORCH_EXTENSION_NAME` undefined — `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` registers a literal module name the wheel cannot import

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (benchmark-acceptance lane standalone `kernel/CMakeLists.txt` + `setup.py` packaging, CAND-BENCH-STANDALONE-KERNEL-PACKAGING)`
`unverified_on: other lanes / CANN versions (any cmake path that bypasses torch.utils.cpp_extension has the same gap)`

- **Symptom**: the cmake-built .so installs but import fails with `PyInit_TORCH_EXTENSION_NAME` missing / module-name mismatch — the wheel looks for the configured extension name while the .so exports a module literally derived from the undefined macro.
- **Root cause**: `TORCH_EXTENSION_NAME` is a compile definition injected by `torch.utils.cpp_extension`; the lane's standalone `kernel/CMakeLists.txt` (vendor-template `ascendc_library` flow) never defines it.
- **Fix**: use the literal module name — `PYBIND11_MODULE(<module_name>, m)` matching the name setup.py/wheel expects (e.g. `PYBIND11_MODULE(instance_norm_grad_ext, m)`), or pass `-DTORCH_EXTENSION_NAME=<module_name>` in cmake compile definitions.
- **Evidence**: instance_norm_grad 310p (2026-09-02, failures_ledger build iter 2 `instance_norm_grad/failures_ledger.md`; PROGRESS.md Phase C): macro undefined under raw cmake → literal-name fix at `代码片段 instance_norm_grad/pybind11.cpp.snippet.cpp（见文末）`; rebuild exported `PyInit_instance_norm_grad_ext` and the wheel imported clean.
- **Other instances (predicted)**: every benchmark-acceptance-lane op packaging a pybind extension via the standalone cmake pair — any template copied from a torch-extension example carries this macro. backend=ascendc.

## 内嵌代码片段

### instance_norm_grad/pybind11.cpp.snippet.cpp

```cpp
                     double scale_in,
                     c10::optional<torch::Tensor> probe_workspace_in)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "query must be on NPU");
    TORCH_CHECK(query.dtype() == torch::kFloat16,
                "kw-2 supports fp16 only; got ", query.dtype());
```
