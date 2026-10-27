---
type: "Runbook"
title: "Gather 非 last-dim 转置+索引换算预处理占比高"
description: "Ascend950 上 Gather 算子当 dim != last 时，plugin 端 permute+contiguous 转置与 Cast/Mul/Cast 三段索引 byte-offset 换算合计占 case 总耗时的 58-83%，kernel 本身仅占 17-41%。"
tags: ["task.debugging", "topic.data_movement.layout_transform", "topic.compute.indexing", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Gather 非 last-dim 转置+索引换算预处理占比高

## 现象

在 Ascend950（dav-3510, CANN 9.1.0）上用 ops-direct-invoke-flash 工作流实现 Gather 算子：plugin（`gather_plugin.cpp`）在 `dim != ndim-1`（非最后一维）时，先把 `x` 与 `index` `permute(actual_dim, ndim-1).contiguous()` 转置成最后一维 gather，再用三段 on-device 换算把元素索引转成 byte-offset uint32：

```cpp
auto xWork = x.contiguous();
auto indexWork = index.contiguous();
if (needTranspose) {
    std::vector<int64_t> perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) perm[i] = i;
    std::swap(perm[actual_dim], perm[ndim - 1]);
    xWork = xWork.permute(perm).contiguous();
    indexWork = indexWork.permute(perm).contiguous();
}
// ...
auto indexUint32 = (indexWork.to(torch::kInt64) * (int64_t)dtypeSize).to(torch::kInt32).contiguous();
```

跑 `./scripts/run_evaluation.sh`（20 case）精度 20/20 All Pass（`max_diff=0.0`），但 dim!=last 的 4 个大 case（gather_1/3/4/10）speedup 均 < 1.0x。评测报告 `op_times.device_kernels` 把总耗时拆成 `aclnnInplaceCopy_TransposeAiCore_Transpose`、`aclnnInplaceCopy_CastAiCore_Cast`、`aclnnMuls_MulAiCore_Mul` 和自定义 `gather_kernel_entry*` 四段，前三段（预处理）合计在 4 个慢 case 里占 58-83%，kernel 本身只占 17-41%。1D / last-dim 的 case（gather_7/13/20 等）没有 Transpose，kernel 占比 57-99%，speedup 1.15-16.24x。

## 适用场景

- **工程形态**：ops-direct-invoke-flash Kernel 直调（自定义 AscendC kernel + torch_npu plugin），非 aclnn spec 算子仓。
- **API 路径**：plugin 侧 `torch::Tensor::permute/contiguous` + `aclnnInplaceCopy_TransposeAiCore`、`aclnnInplaceCopy_CastAiCore`、`aclnnMuls_MulAiCore`；kernel 侧 `AscendC::Gather` / `AscendC::DataCopyPad`。
- **验证平台与版本**：Ascend950（SoC Ascend 950, NpuArch dav-3510, Ascend950PR_9579），CANN 9.1.0, PyTorch 2.10.0+cpu, torch_npu 2.10.0.post2。`platforms: [950]`。
- **适用粒度**：`operator_family`（Gather/Scatter/IndexSelect 等索引收集类算子的非 last-dim 实现路径）。
- **路线优先级**：`safety_fallback`（转置+预处理是本 trace 唯一完成精度与计时验证的实现；未实现的 stride-aware 方案不作为推荐路线）。
- **不适用或未验证场景**：dim == last-dim 的 gather（无转置开销，本诊断不适用）；Ascend950 以外平台未验证，推断 a2/a3 同类转置 aclnn 亦有开销但未实测。

## 判别方法

- **触发（tool_observed）**：Gather/索引收集类算子，`dim != ndim-1`，评测 speedup < 1.0x 或 `below_compute_capacity`。
- **可执行探针**：跑评测后读评测报告 JSON 的 `operators[].cases[].op_times.device_kernels`，把 `aclnnInplaceCopy_TransposeAiCore_Transpose` + `aclnnInplaceCopy_CastAiCore_Cast` + `aclnnMuls_MulAiCore_Mul` 三段耗时相加，与 `gather_kernel_entry*`（或对应自定义 kernel）比较。
- **原始观测（tool_observed，Ascend950 / CANN 9.1.0）**：
  - gather_1（f16, 2D, dim=0）：预处理 3 段合计占 case 总耗时约 81%，kernel 约 19%；相对 baseline 约 0.92x。
  - gather_3（bf16, 4096×4096, dim=0）：预处理约占 76%，kernel 约 24%；相对 baseline 约 0.62x。
  - gather_4（int32, 8192×8192, dim=0）：预处理约占 83%，kernel 约 17%；相对 baseline 约 0.73x。
  - gather_10（f16, 3D, dim=1）：预处理约占 59%，kernel 约 41%；相对 baseline 约 0.51x。
  - 对照 1D/last-dim case（gather_7/13/20）：无 Transpose，kernel 占 57-99%，相对 baseline 1.15-16.24x。
- **解释（interpretation）**：plugin 把非 last-dim gather 归约为 last-dim gather 以便用 `AscendC::Gather` 在 UB fiber 上做连续收集，代价是 1 次 `permute+contiguous`（落地为 `aclnnInplaceCopy_TransposeAiCore_Transpose`）+ 索引 byte-offset 三段换算（`to(kInt64)` → `Mul(dtypeSize)` → `to(kInt32)`，分别对应 `Cast` / `Mul` / 第二次 `Cast`，每次都是独立 aclnn kernel launch）。对大 shape 这 3-4 个预处理 kernel 的 launch+搬运开销远超 gather kernel 自身。索引换算三段链还额外占 12-24%（gather_3/4/10 的 Cast+Mul 段）。

## 验证信号

问题与验证来源：在 Ascend950 上对 Gather 算子跑 20-case 评测，读回评测报告 JSON（`gather_eval_20260623_063200.json`）的逐 case `op_times.device_kernels` 与 `accuracy` 字段得出下列相对占比与精度结论。

- 精度 20/20 All Pass，所有 case `mismatch_count=0`、`max_diff=0.0`、`passed=true`（checker=`relative_error`）（来源平台：950；证据等级：artifact_verified）。
- dim!=last 的 4 个慢 case 预处理（Transpose+Cast+Mul）占 case 总耗时 58-83%，kernel 占 17-41%（来源平台：950；证据等级：artifact_verified）。
- dim!=last 慢 case 相对 baseline 路线约 0.51-0.93x；1D/last-dim case 相对 baseline 1.15-16.24x（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- Ascend950 / CANN 9.1.0 实测；a2/a3 同类转置与 Cast/Mul aclnn 是否有同等占比未验证，推断可迁移但未实测。
- 本诊断只覆盖「plugin 端 permute+contiguous + 三段索引换算」这一实现路径；若改用 stride-aware kernel（无转置）则本诊断的占比结论不再适用。
- 本卡只建立现有实现的耗时归因；任何去转置或 stride-aware 改写都未形成可核的精度与性能闭环。

## 坏实践

- 把 `index.to(kInt64) * dtypeSize` 再 `.to(kInt32)` 写成三段独立算子链，在 plugin 侧额外产生 2 个 Cast + 1 个 Mul 的 aclnn kernel launch，对大 shape 占 12-24% 耗时；终版 plugin 仍保留三段链（下沉进 kernel 的 `AscendC::Muls` 单条换算未实现）。

## 误诊

- 早期曾把慢 case 归因为「UB buffer overflow 触发 slow path」与「单线程并行不足」，但 perf 报告 `op_times` 显示 dim!=last 慢 case 的 kernel 时间占比仅 17-41%，真正瓶颈是预处理 aclnn kernel，不是 kernel 内 UB/并行（来源平台：950；证据等级：artifact_verified）。