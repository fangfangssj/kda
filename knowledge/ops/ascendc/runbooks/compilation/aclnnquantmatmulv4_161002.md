---
type: "Runbook"
title: "aclnnQuantMatmulV4 直调 161002 根因判别"
description: "Ascend 950/CANN 9.1.0 上从 C++ op_plugin 用 dlsym 手构 aclTensor 直调 aclnnQuantMatmulV4，GetWorkspaceSize 返 161002；以 nm 符号确认与同一 NPU torch_npu.npu_quant_matmul 成功为对照，判别 161002 为手构 aclTensor 的参数/格式校验失败而非算子缺失，机制层未追尽。"
tags: ["task.debugging", "topic.compute.quantization", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-18T07:46:04Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aclnnQuantMatmulV4 直调 161002 根因判别

## 现象

在 cann_bench `direct_launch_example` 工程里实现 `quant_matmul`（desc.md 指明 CANN 底层实现为 `aclnnQuantMatmulV4`，输入 int8、输出 float16/bfloat16），平台 SoC=Ascend 950 / NpuArch=dav-3510 / CANN 9.1.0。Agent 写 C++ `op_plugin/quant_matmul_plugin.cpp`，用 `dlopen`+`dlsym` 从 `libopapi.so`（`libcust_opapi.so` fallback 后）解析 `aclnnQuantMatmulV4GetWorkspaceSize` / `aclnnQuantMatmulV4`，自行用 `aclCreateTensor` 手构 `aclTensor` 后调用 `aclnnQuantMatmulV4GetWorkspaceSize`。

`--soc=ascend950` 构建通过；`nm -D libopapi.so` 确认 `aclnnQuantMatmulV4` 与 `aclnnQuantMatmulV4GetWorkspaceSize` 均为导出符号（`T`）。但运行时 `aclnnQuantMatmulV4GetWorkspaceSize(...)` 直接返回 `161002`，插件抛：

```
RuntimeError: aclnnQuantMatmulV4GetWorkspaceSize failed: 161002
[ERROR] ... ERR99999 UNKNOWN applicaiton exception
```

随后官方评测 20 例中 11 例报同一 `161002`（均为 float16 输出、基础 scale、无 pertoken_scale 的组合），仅 9 例通过。

## 适用场景

- **工程形态**：direct_launch / op_plugin C++ 插件 + Python `cann_bench` wrapper；Host 侧调 ACLNN 算子（非自定义 Ascend C kernel）。
- **API 路径**：`aclnnQuantMatmulV4` / `aclnnQuantMatmulV4GetWorkspaceSize`（手构 `aclTensor` + `dlsym` 直调）；对照路径 `torch_npu.npu_quant_matmul`。
- **验证平台与版本**：Ascend 950（dav-3510）/ CANN 9.1.0。frontmatter `platforms: [950]` 与本场景严格一致。
- **适用粒度**：`operator_family`（适用于在 950 stock CANN 上经手构 `aclTensor` + `dlsym` 直调 aclnn 量化 matmul 系列的算子工程）。
- **路线优先级**：`recommended_fastpath`——用 `torch_npu.npu_quant_matmul`；`banned`——在 950 上从 C++ 插件手构 `aclTensor` + `dlsym` 直调 `aclnnQuantMatmulV4`。
- **不适用/未验证场景**：a2 / a3 平台未验证；自定义 Ascend C kernel 路径不在本卡范围（本算子未走 kernel 路线）。

## 判别方法

判定一次 `aclnnQuantMatmulV4GetWorkspaceSize` 返回 `161002` 是否属"手构 `aclTensor` 校验失败"而非"算子缺失/不支持"：

1. **触发条件（trigger）**：在 950 / CANN 9.1.0 上以 `dlsym` 从 `libopapi.so` 解析并直调 `aclnnQuantMatmulV4GetWorkspaceSize`，自行用 `aclCreateTensor` 手构 `aclTensor` 传入；调用返回 `161002`（含 `ERR99999 UNKNOWN applicaiton exception`），失败为 RuntimeError。原始输出：
   ```
   RuntimeError: aclnnQuantMatmulV4GetWorkspaceSize failed: 161002
   ```

2. **可执行探针（probe）**：同一 NPU 会话内执行两条独立检查：
   - 符号存在性：`nm -D libopapi.so | grep -E 'aclnnQuantMatmulV4(GetWorkspaceSize)?$'`
   - 算子可用性对照：在同一 NPU 上调 `torch_npu.npu_quant_matmul(x1, x2, scale, output_dtype=torch.float16)`（底层走同一 `aclnnQuantMatmulV4`）。

3. **原始观测（observation）**：
   - `nm -D` 命中 `T aclnnQuantMatmulV4` 与 `T aclnnQuantMatmulV4GetWorkspaceSize`（符号存在、已导出）。
   - `torch_npu.npu_quant_matmul(...)` 正常返回 `torch.Size([32, 512])`（同一 NPU、同一底层算子成功）。

4. **解释（interpretation）**：符号存在 + 同一 NPU 上经 torch_npu 内部构造 `aclTensor` 调同一算子成功 → `161002` 不是"算子不存在/不支持"，而是 ACLNN 算子对**手构 `aclTensor` 的参数/格式校验失败**。本判别依赖的手构 `aclTensor` 机制单元（verbatim 取自 trace 写入事件）：

```cpp
static aclTensor *CreateAclTensor(const at::Tensor &tensor) {
    if (!tensor.defined()) return nullptr;
    static const auto aclCreateTensor = reinterpret_cast<_aclCreateTensor>(
        GetOpApiFuncAddr("aclCreateTensor"));
    if (!aclCreateTensor) return nullptr;

    static constexpr int type_map[] = {
        [static_cast<int>(at::ScalarType::Byte)] = 6,
        [static_cast<int>(at::ScalarType::Char)] = 3,
        [static_cast<int>(at::ScalarType::Short)] = 4,
        [static_cast<int>(at::ScalarType::Int)] = 5,
        [static_cast<int>(at::ScalarType::Long)] = 2,
        [static_cast<int>(at::ScalarType::Half)] = 1,
        [static_cast<int>(at::ScalarType::Float)] = 7,
        [static_cast<int>(at::ScalarType::Double)] = 8,
        [static_cast<int>(at::ScalarType::BFloat16)] = 29,
    };

    at::ScalarType scalar_type = tensor.scalar_type();
    int acl_dtype = type_map[static_cast<int>(scalar_type)];

    auto itemsize = tensor.itemsize();
    c10::SmallVector<int64_t, 5> storageDims;
    storageDims.push_back(tensor.storage().nbytes() / itemsize);

    return aclCreateTensor(
        tensor.sizes().data(), tensor.sizes().size(), acl_dtype,
        tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND,
        storageDims.data(), storageDims.size(),
        const_cast<void *>(tensor.storage().data())
    );
}
```

   由此将怀疑面收敛到：手写 `type_map[]` 把 `at::ScalarType` 映射到本地 `ACL_*` 整数常量、`tensor.storage().nbytes()/itemsize` 单元素 `storageDims`、`ACL_FORMAT_ND` 这一组合，绕过了 torch_npu 内部的平台感知构造。具体是 `aclDataType` 取值、`stride`/`storage_dims` 维度还是 `ACL_FORMAT_ND` 与该算子期望 format 不符——本 trace 未追到机制层。若 `nm` 命中 0 个符号或 `torch_npu` 同样失败，则不属本判别（前者为符号/库定位问题，后者为算子在 950 真不支持）。

## 验证信号

本 trace 内实测（同一 Ascend 950 / CANN 9.1.0 会话）：

- `aclnnQuantMatmulV4GetWorkspaceSize` 直调返回 `161002`（RuntimeError，含 `ERR99999 UNKNOWN applicaiton exception`）（来源平台：950；证据等级：tool_observed）
- `nm -D libopapi.so` 命中 `T aclnnQuantMatmulV4` 与 `T aclnnQuantMatmulV4GetWorkspaceSize`，证明 `161002` 非符号缺失（来源平台：950；证据等级：tool_observed）
- 同一 NPU `torch_npu.npu_quant_matmul(x1, x2, scale, output_dtype=torch.float16)` 返回 `torch.Size([32, 512])`，底层走同一 `aclnn` 算子成功（来源平台：950；证据等级：tool_observed）
- C++ 插件仍在时官方评测 11/20 用例报 `161002`（9/20 通过）（来源平台：950；证据等级：tool_observed）

## 失效边界

- `torch_npu.npu_quant_matmul` 的可迁移性：a2 / a3 平台未验证，推断可迁移（torch_npu 跨平台绑定，底层调同一 `aclnn` 算子），但未实测，不得据此刻入 frontmatter `platforms`。
- raw ACL 直调的 `161002` 是否为 950 特有、是否在 a2/a3 上以同样方式复现——未验证。
- C++ `TORCH_LIBRARY_IMPL(... PrivateUse1 ...)` 注册的 `torch.ops.cann_bench.quant_matmul` 遮蔽 Python wrapper 的判据：本 trace 内观测到（C++ 插件在→分发到失败 C++ 路径；删 C++ 插件→分发到 `torch_npu`），但属 torch dispatch 通用机制 + cann_bench harness 表现，未在库层确认是否所有 cann_bench 算子受此影响。

## 预防

- 在 stock Ascend 950 / CANN 9.1.0 上调 `aclnnQuantMatmulV4` 时，优先用 `torch_npu.npu_quant_matmul`；如确需 C++ 直调，不要手构 `aclTensor`，改用 `aclnn_launch_example` 的 `ACLNN_CMD`/`ConvertTypes` 完整通路（它内部处理 `at::Tensor`→`aclTensor*` 的平台正确构造）。
- 同一算子名不要同时注册 C++ `torch.ops` 实现与 Python wrapper 期望的 fallback 路径；若 Python wrapper 是期望入口，就不要让 C++ `TORCH_LIBRARY_IMPL(PrivateUse1)` 抢占同名 op。

## 坏实践

- 手写 `type_map[]` 把 `at::ScalarType` 映射到本地 `ACL_*` 整数常量、并用 `tensor.storage().nbytes()/itemsize` 作 `storageDims`、`ACL_FORMAT_ND` 去构造 `aclTensor`——这种"裸数值映射 + 裸 storage 描述"绕过了 torch_npu 内部的平台感知构造，在 950 上被 `GetWorkspaceSize` 以 `161002` 拒绝。

## 误诊

- agent 一度怀疑 `161002` 是"算子在 950 不支持"或"aclDataType 枚举值不对"，并反复改 dtype 映射试图消除，均未解决；最终用 `torch_npu.npu_quant_matmul` 反证"算子可用、问题在直调方式"。该怀疑属 agent_reported，不进验证信号（验证信号只保留 tool_observed 的 `161002` 与 `torch_npu` 成功对照）。
- agent 初期用 `--soc=ascend910b` 构建（与目标 950 不符），属配置错误，改 `--soc=ascend950` 即解，不构成本卡的根因。