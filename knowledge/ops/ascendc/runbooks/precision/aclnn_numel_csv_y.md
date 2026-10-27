---
type: "Runbook"
title: "aclnn 黑盒 numel 不符：输出取自 CSV y 不被算子覆盖"
description: "TTK aclnn 黑盒把 CSV tensor_view_shapes 的 y 当作输出 numel，算子 infershape 在该路径不覆盖调用方输出 shape，致全量用例 numel 与 golden 系统性不符。"
tags: ["task.debugging", "topic.precision", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aclnn 黑盒 numel 不符：输出取自 CSV y 不被算子覆盖

## 现象

在 Ascend950（dav-3510, arch35）上对自研 `euclidean_norm`（L2 归约，`sqrt(sum(x^2 over axes))`）算子跑 TTK aclnn 黑盒全量 621 用例（`ttk aclnn -i blackbox_all.csv`，无 `--quick`/`--max-cases` 子集捷径），结果 `total=621 pass=0 fail=621 skip=0`，逐用例精度列几乎全部是 `<actual> vs <golden>` 形如 `36 vs 24` 的**元素数不符**（parse_result.py 归类 NUMERIC_MISMATCH 604、GOLDEN_FAILURE 7、NO_OUTPUT 4、PARAM_COUNT_MISMATCH 6）。算子侧 C++ ST（`tests/aclnn/run.sh`，自带独立 CPU oracle、手工给正确输出 shape）全 80+ 用例 PASSED，kernel 对 ST 用例正确；唯有走 TTK 黑盒的 aclnn 路径全量失败。

以 L1_237 为例：`x shape=(1,4,1,2,3)` fp32，axes 数据范围 `(0,0)` 即 `axes=[0]`，keepDims=false，正确降维输出 shape 应为 `(4,1,2,3)=24` 元素。CSV 的 `tensor_view_shapes` y 字段却是 `(2,1,1,3,2,3)=36`。TTK `full_run.log` 打印 `View Shape: ((1, 4, 1, 2, 3), (1,), (2, 1, 1, 3, 2, 3))` 与 `GOLD: 36 vs 24`——即 TTK 把 CSV y 的 36 元素当 device 输出 numel，golden 算出 24，36≠24 直接判 precision 为 `36 vs 24`。

## 适用场景

- **工程形态**：aclnn 算子工程化开发（ops-registry-invoke / aclnn 单算子 API 路径），用 TTK（ops-test-kit）aclnn 模式做黑盒精度验收。
- **API 路径**：`aclnnEuclideanNormGetWorkspaceSize(x, axes, keepDims, out, &ws, &ex)` + `aclnnEuclideanNorm(...)`；输出 tensor 由调用方（TTK）按 CSV `tensor_view_shapes` 的 y 项创建；TTK 比对经 `aclnn` framework_api → `is_close.py::_isclose`。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510, arch35, kernel binary `ascend950/euclidean_norm/*.o`）。CANN/TTK 具体版本号未在 trace 内定位（未验证场景见失效边界）。
- **适用粒度**：`operator_family`（适用于一切走 TTK aclnn 黑盒、由 CSV y-shape 驱动输出 buffer 的归约/shape-changing 算子，不限于 euclidean_norm）。
- **路线优先级**：`recommended_fastpath`（判别 aclnn 黑盒"全量 numel 不符"类失败时先查此机制，再查 kernel 数值）。
- **明确不适用或未验证场景**：kernel 直调（`kernel_direct_call`，调用方自填正确输出 shape）不受此机制约束；ST 用例因调用方手工给正确 shape 也不会暴露此问题；未验证 TTK 非 aclnn 模式（E2E/Torch）与其它 CANN 大版本下 `aclGetViewShape` 是否同样返回调用方 shape。

## 根因

TTK aclnn 黑盒的输出 buffer 与比对 numel **取自 CSV `tensor_view_shapes` 的 y 项**，而非算子 infershape 推导的降维 shape；且在本 aclnn framework 路径下**算子 infershape 不覆盖调用方提供的输出 shape**（`aclGetViewShape` 返回调用方创建 tensor 时填入的 shape，未被算子改写）。于是当 CSV y-shape ≠ 正确降维 shape（本 trace 中全量 621 条 CSV y-shape 均为占位/无意义值，0/621 与正确降维 shape 吻合），`is_close._isclose` 走到 `output_size != golden_size` 分支，直接产出 `<csv_y_count> vs <golden_count>` 的 precision 串并判 FAIL——这是**全量系统性 numel 不符**，与 kernel 数值正确与否无关。

补充：本算子 `euclidean_norm_infershape.cpp` 还是未替换的 AddCustom 桩（见下"判别方法"），即便框架调用了它，它也只是 `*y_shape = *x1_shape`（把 x shape 复制给 y），对归约算子是错的；但实测 `aclGetViewShape` 仍返回调用方 shape，说明该 infershape 在此 aclnn 路径下既未生效也未传播——**核心机制是"aclnn 不让算子覆盖调用方输出 shape"**，infershape 桩错只是旁证。

## 判别方法

**触发**：TTK aclnn 黑盒全量用例 precision 列形如 `<N> vs <M>`（两个整数），且 N 恰等于 CSV `tensor_view_shapes` y 项乘积、M 恰等于按 (x shape, axes 实际值, keepDims) 算出的正确降维元素数；C++ ST（调用方自填正确输出 shape）却全绿。

**可执行探针**：以 standalone C++ 复现程序（链接安装好的 vendor `libcust_opapi.so`，`aclrtSetDevice(0)` 上板 Ascend950）调用 `aclnnEuclideanNorm`，刻意以 CSV 风格的**错误 out shape** 创建 `aclTensor* out`，执行后查询 `aclGetViewShape(out)` 并读 device 输出与"正确降维 golden 元素数"比对。关键：算子不会把 out 的 shape 改回正确值——下为复现程序 `runOne` 的输出查询与比对核心（verbatim 取自 trace 中 `tests/aclnn/test_repro.cpp`）：

```cpp
    // The output element count TTK would read = product of the shape we provided
    // (the op does not resize the caller's output buffer).
    int64_t reportedCount=prod(outShapeProvided);
    fprintf(stdout,"[%s] out providedShape=",tag);
    for(auto d:outShapeProvided) fprintf(stdout,"%lld ",(long long)d);
    fprintf(stdout,"(count=%lld)\n",(long long)reportedCount);

    std::vector<uint8_t> act(outTotal*4);
    aclrtMemcpy(act.data(),outTotal*4,devOut,outTotal*4,ACL_MEMCPY_DEVICE_TO_HOST);
    // Print first few output values
    fprintf(stdout,"[%s] actual[0..min(8,n)]:",tag);
    for(int64_t i=0;i<std::min<int64_t>(8,outTotal);++i){float v;std::memcpy(&v,act.data()+i*4,4);fprintf(stdout," %.3f",v);}
    fprintf(stdout,"\n");
```

`reportedCount` 始终是调用方填入的 `outShapeProvided` 乘积，算子未改写。

**原始观测**（探针运行输出，`COMPILE OK` 后）：

```
[A2_axes0_csvOut] GetWorkspaceSize ret=0 ws=0
[A2_axes0_csvOut] aclGetViewShape ret=0 repNum=6 repDims=2 1 1 3 2 3 (reportedCount=36)
[A2_axes0_csvOut] out providedShape=2 1 1 3 2 3 (count=36)
[A2_axes0_csvOut] actual[0..min(8,n)]: 15.146 14.387 13.872 0.000 0.000 0.000 0.000 0.000
[A2_axes0_csvOut] goldenShape=4 1 2 3 (count=24)
[A2_axes0_csvOut] SHAPE_MATCH=NO (providedCount=36 goldenCount=24)
```

`aclGetViewShape` 返回的 `repDims=2 1 1 3 2 3`（count=36）就是调用方给的 CSV y shape，未被算子改写成正确的 `(4,1,2,3)=24`。

**机制解释（TTK 比对源，verbatim 取自 `/workspace/ops-test-kit/ttk/core_modules/comparison/is_close.py` 完整 `_isclose` 方法）**：

```python
    def _isclose(self) -> EachCompareResult:
        rtol = self._get_rtol(self.output.dtype)
        atol = self._get_atol(self.output.dtype)
        diff_indices, log = None, ""
        output_size = self.output.numel() if self.is_torch_output else self.output.size
        golden_size = self.golden.numel() if self.is_torch_golden else self.golden.size
        if golden_size <= 0:
            if not self.output.any():
                precision = 1
                log += f"Output {self.output_idx} is full of zero while it has a zero golden.\n"
            else:
                precision = 0
                diff_indices = [idx for idx in range(output_size)]
                if self.is_torch_output:
                    import torch
                    diff_indices = torch.tensor(diff_indices)
                log += f"Output {self.output_idx} is not full of zero while it has a zero golden.\n"
        elif output_size != golden_size:
            precision = f"{output_size} vs {golden_size}"
            log += f"Output {self.output_idx} size is different with golden size: {precision}\n"
        else:
            output = self._normalize_dtype(self.output)
            golden = self._normalize_dtype(self.golden)
            if not self.is_torch_output and not self.is_torch_golden:
                precision, diff_indices = self._numpy_isclose(output, golden, rtol, atol, golden_size)
            else:
                import torch
                if self.is_torch_output and self.is_torch_golden:
                    try:
                        promote_dtype = torch.promote_types(output.dtype, golden.dtype)
                        diff_results = torch.isclose(output.to(dtype=promote_dtype).view([-1]),
                                                     golden.to(dtype=promote_dtype).view([-1]),
                                                     rtol=rtol, atol=atol, equal_nan=True)
                        diff_indices = torch.where(diff_results != True)[0]
                        del diff_results
                        precision = (golden_size - diff_indices.numel()) / golden_size
                    except (TypeError, RuntimeError):
                        output = output.detach().cpu().numpy()
                        golden = golden.detach().cpu().numpy()
                        precision, diff_indices = self._numpy_isclose(output, golden, rtol, atol, golden_size)
                else:
                    if self.is_torch_output:
                        output = output.detach().cpu().numpy()
                    if self.is_torch_golden:
                        golden = golden.detach().cpu().numpy()
                    precision, diff_indices = self._numpy_isclose(output, golden, rtol, atol, golden_size)
        return EachCompareResult(precision, diff_indices, False, log)
```

`output_size = self.output.numel()` 中的 `self.output` 是 TTK 从 device 读回的输出（按 CSV y storage 的 `np_storage.nbytes` 字节数读回，再 reshape 到 `aclGetViewShape` 返回的 shape）；`golden_size = self.golden.numel()` 是 golden.py 按正确降维算出的元素数。`output_size != golden_size` 直接产出 `f"{output_size} vs {golden_size}"`——这就是黑盒 `result.csv` 里满屏 `<N> vs <M>` 的来源。

TTK 重查输出 shape 的入口（verbatim 取自 `/workspace/ops-test-kit/ttk/core_modules/aclnn/acl_interface.py`）：

```python
    def get_view_shape(self, tensor: ctypes.c_void_p) -> tuple:
        c_view_dims_ptr = ctypes.POINTER(ctypes.c_int64)()
        c_view_dims_ptr_ptr = ctypes.byref(c_view_dims_ptr)
        c_view_dims_num = ctypes.c_uint64()
        c_view_dims_num_ptr = ctypes.byref(c_view_dims_num)
        self._opbase_api_call("aclGetViewShape", None,
                              tensor, c_view_dims_ptr_ptr, c_view_dims_num_ptr)
        view_shape = tuple([c_view_dims_ptr[i]
                            for i in range(c_view_dims_num.value)])
        delete_array_func = self._get_opbase_dll('_ZdaPv')
        if delete_array_func:
            delete_array_func.argtypes = [ctypes.c_void_p]
            delete_array_func.restype = None
            delete_array_func(c_view_dims_ptr)
        return view_shape
```

`aclGetViewShape` 在算子执行后重查输出 tensor 的 view shape；本 trace 实测它返回的是调用方创建 tensor 时填入的 shape（CSV y），而非算子 infershape 改写后的值——说明算子侧 infershape 在此 aclnn 路径下不传播。

**未替换的 AddCustom infershape 桩**（verbatim 取自 `operators/euclidean_norm_package/op_host/euclidean_norm_infershape.cpp`，完整函数）：

```cpp
static ge::graphStatus InferShapeForEuclideanNorm(gert::InferShapeContext* context)
{
    // Get the shape of the first input (x).
    const gert::Shape* x1_shape = context->GetInputShape(0);

    // Get a mutable reference to the output shape (z).
    gert::Shape* y_shape = context->GetOutputShape(0);

    // Copy x's shape to z: output shape = input shape.
    *y_shape = *x1_shape;

    return GRAPH_SUCCESS;
}
```

注意上方注释甚至仍写着 "EuclideanNorm is element-wise addition (x + y = z)"——是从 AddCustom 模板复制后未替换的桩。对归约算子它本是错的（应按 axes 计算降维 shape），但因 infershape 在此 aclnn 路径不传播，这个错误桩反而没造成 shape 被改写成 x shape 的副作用——真正决定输出 numel 的是调用方填的 CSV y。以上"infershape 不传播、输出 numel 取自调用方 CSV y"是本判别方法的**解释（interpretation）**：观测锚是 `aclGetViewShape` 返回值与 `SHAPE_MATCH` 输出，根因归因（框架在该 aclnn 路径不调用/不采纳算子 infershape 改写）为解释层结论。

## 验证信号

本卡信号来自一次 euclidean_norm 黑盒失败用例复现会话：用 standalone C++ 复现程序（链接 vendor `libcust_opapi.so`，`aclrtSetDevice(0)` 上板 Ascend950）跑 aclnn 路径并查询 `aclGetViewShape`，对照 TTK 黑盒全量结果与 TTK 比对源码。

- `COMPILE OK`，`aclnnEuclideanNormGetWorkspaceSize ret=0`，算子成功执行并产出 device 输出（来源平台：950；证据等级：tool_observed）
- 用例 A2（axes=[0]、CSV 风格错误 out shape (2,1,1,3,2,3)）：`aclGetViewShape` 返回 `repDims=2 1 1 3 2 3 (reportedCount=36)`，`SHAPE_MATCH=NO (providedCount=36 goldenCount=24)`（来源平台：950；证据等级：tool_observed）
- TTK 黑盒 `full_run.log` 对 L1_237 打印 `GOLD: 36 vs 24` 与 `View Shape: ((1, 4, 1, 2, 3), (1,), (2, 1, 1, 3, 2, 3))`（来源平台：950；证据等级：tool_observed）
- 黑盒执行报告统计 `total=621 pass=0 fail=621 skip=0`（L0:23 fail / L1:592 fail / L2:6 fail），`GATE: FAIL`（来源平台：950；证据等级：tool_observed）

## 失效边界

- **未验证，推断可迁移**：同 CANN aclnn framework 下其它 shape-changing 算子（cast/broadcast/reduce_sums 等）若用 TTK aclnn 黑盒验收、CSV y-shape 与正确输出 shape 不一致，应同样全量命中 `output_size != golden_size` 分支；未在其它算子上实测。
- **未验证**：TTK 非 aclnn 模式（E2E / Torch 路径）是否同样以 CSV y 为输出 buffer；未实测。
- **未验证**：其它 CANN 大版本 / 非 Ascend950 平台下 `aclGetViewShape` 是否同样返回调用方 shape 而不被算子 infershape 覆盖；本卡 `platforms` 仅保留实测的 `950`。
- 本卡不提供修复配方（trace 内修复 turn 在完成前结束、未验证全量全绿）；落库后需由后续 trace 的 `repair_recipe` 补全。

## 坏实践

- **把"放宽容差/改 golden/改 CSV y"当修复**：本任务的失败是 numel 不符（连数值比对都没进入），调松 `rtol/atol` 无意义；改 golden.py 的输出 shape 或改 CSV y-shape 让 count 凑齐则是"测试侧放松"，掩盖真因，属禁止捷径。
- **靠 C++ ST 全绿推断 kernel 对黑盒也对**：ST 调用方自填正确输出 shape，绕开了"aclnn 不覆盖调用方 shape"这一机制，ST 全绿不能代替黑盒全绿。

## 误诊

- 早期一度怀疑"infershape 桩错（y=x）是直接根因"——但实测 `aclGetViewShape` 返回的是调用方 shape 而非 x shape（24），说明该 infershape 既未生效也未传播；真正的驱动是 CSV y-shape。判别时先查 `aclGetViewShape` 返回值是否等于调用方填入 shape，勿先动 infershape 桩或 compute。
