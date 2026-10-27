---
type: "Runbook"
title: "golden 参考精度不足：fp32 中间计算升 fp64"
description: "PyTorch fallback golden 以 fp32 做双线性插值中间计算时，参考侧舍入与 kernel 侧 fp32 舍入在 abs_err 里叠加（3e-5 > abs_floor 2e-5），近零 golden 元素 MARE 放大至 7.2e-3 假失败；参考实现精度必须高于被测对象——中间计算升 fp64（含 ROI 轴分块控内存）后转 PASS。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# golden 参考精度不足：fp32 中间计算升 fp64

## 现象

PyTorch ST 测试（BenchNoHarnessRoiAlign，fp32 kernel，Ascend950PR）在判定口径已对齐（mixed abs+rel，abs_floor=2e-5）并用逐用例独立进程消除 tiling 缓存后，36 条用例仍剩 1 条失败——大 shape fp32 用例（x=1x64x512x512，oh=ow=14，sr=4）：

```
MERE: 1.909039e-05 (阈值: 1.220703e-04) PASS
MARE: 7.216146e-03 (阈值: 1.220703e-03) FAIL
最大差异位置: [319309] Golden=0.001195, Actual=0.001165
超阈值元素数: 21763 / 627200
  [1] Golden=-0.025200, Actual=-0.025205, abs_err=4.910e-06, rel_err=1.741e-04
  [42] Golden=-0.043791, Actual=-0.043783, abs_err=7.860e-06, rel_err=1.680e-04
```

同一个用例在 C++ ST（golden 用 double 中间计算）通过。差异点：环境里 torchvision 不可用，PyTorch 测试走了纯 Python fallback golden，而该 fallback 此前以**输入 dtype（fp32）**做中间计算；C++ golden 一直是 double。golden=0.001195 的近零元素上，abs_err=3e-5 超过了 abs_floor=2e-5——这个 3e-5 里有一半是参考自己带来的：fp32 golden 与 fp32 kernel 各自舍入，误差叠加。

## 适用场景

- 工程形态：算子精度测试的 CPU golden 参考实现（PyTorch fallback / 任何手写参考），被测 kernel 为 fp32。
- API 路径：`torchvision.ops.roi_align` 的纯 Python 等价 fallback（`_roi_align_fallback` + `_bilinear_interpolate`）；参考实现精度策略。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，torch 2.7.1（fp64 中间计算在 CPU 上执行）。
- 适用粒度：platform_wide（参考实现精度纪律，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 与参考都已是 fp64 的场景；fp16 kernel（其噪声界远大，参考 fp32 已足够高时另议）；torchvision 原生可用时应直接用官方参考（其内部精度由官方保证）。

## 根因

golden 参考与被测 kernel 使用同档精度（fp32）做中间计算时，参考侧舍入与 kernel 侧舍入在 abs_err 里叠加（本例把噪声界从 C++ 参考下的 ~1.17e-5 推高到 3e-5），超过按"单侧 fp32 舍入"校准的 abs_floor；参考实现精度必须高于被测对象，否则测量的是两份舍入而非 kernel 误差。

## 解决配方

`golden.py::_roi_align_fallback` 中间计算整体升 fp64（含大 shape 的 ROI 分块控内存），修后核心形态（verbatim 关键段）：

```python
def _roi_align_fallback(
    x, boxes, outputHeight, outputWidth,
    spatial_scale, sampling_ratio, aligned,
):
    """Pure-Python fallback mirroring torchvision.ops.roi_align's reference path.

    中间计算使用 fp64 (double), 与 C++ ST 测试 (ComputeGoldenRoiAlignFp32) 的
    double 中间精度一致。原因: torchvision 不可用时使用本 fallback, 若以 fp32
    计算 golden, fp32 golden vs fp32 kernel 的 abs_err 包含双方的 fp32 舍入,
    对近零 golden 值 (如 0.001) 会放大相对误差 (L1_010 数据依赖问题)。
    升 fp64 后 abs_err 仅含 kernel 侧 fp32 舍入, 与 C++ golden 一致。
    """
    orig_dtype = x.dtype
    x_fp64 = x.double()
    boxes_fp64 = boxes.double()
    _, channels, height, width = x_fp64.size()
    roi_batch_ind = boxes_fp64[:, 0].long()
    offset = 0.5 if aligned else 0.0
    roi_start_w = boxes_fp64[:, 1] * spatial_scale - offset
    roi_start_h = boxes_fp64[:, 2] * spatial_scale - offset
    roi_end_w = boxes_fp64[:, 3] * spatial_scale - offset
    roi_end_h = boxes_fp64[:, 4] * spatial_scale - offset
    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0)
        roi_height = roi_height.clamp(min=1.0)
    bin_size_h = roi_height / outputHeight
    bin_size_w = roi_width / outputWidth
    exact_sampling = sampling_ratio > 0
    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / outputHeight)
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / outputWidth)
    ...  # 采样网格 y/x_pos 构造（同式 fp64）

    # _bilinear_interpolate materializes [n, C, oh, gh, ow, gw]; for auto-sampling
    # (gh,gw span the full feature map) this reaches ~18 GiB on case 16 and OOMs the
    # NPU. Per-ROI reductions are independent, so chunking the ROI axis is bit-exact
    # and caps the peak val tensor to ~1 GiB (fp64).
    n_rois = y.shape[0]
    elems_per_roi = max(
        1, x_fp64.shape[1] * y.shape[1] * y.shape[2] * x_pos.shape[1] * x_pos.shape[2]
    )
    # fp64: 8 bytes/elem, budget ~1 GiB → 128M elems per chunk
    chunk = max(1, (128 * 1024 * 1024) // elems_per_roi)
    parts = []
    for st in range(0, n_rois, chunk):
        en = st + chunk
        val = _bilinear_interpolate(
            x_fp64, roi_batch_ind[st:en], y[st:en], x_pos[st:en],
            None if ymask is None else ymask[st:en],
            None if xmask is None else xmask[st:en],
        )
        ...  # 掩码置零 + sum((-1,-2)) + 除 count（逐 ROI 独立）
        parts.append(out_chunk)
    output = torch.cat(parts, dim=0)

    return output.to(orig_dtype)
```

要点：

1. 入口 `x.double()` / `boxes.double()`，出口 `output.to(orig_dtype)`——只在参考内部用 fp64，比对仍按原 dtype。
2. fp64 使中间张量内存翻倍：按 ROI 轴分块（fp64 每元素 8 字节、预算 ~1GiB → 每块 128M 元素）；逐 ROI 归约相互独立，分块对结果逐位无损。
3. 修完必须清 `__pycache__` 再跑（本会话实测旧缓存会让修复"看似无效"）。

## 设计决策

- 为什么升参考精度而不是再放大 abs_floor：abs_floor 按"单侧 fp32 舍入"校准（2e-5）；参考叠加后 3e-5 超界，正解是消除参考侧噪声，让测量回到"kernel 单侧误差"；继续放大阈值会连真实回归一起豁免。
- 为什么与 C++ golden 对齐到 double：两套 harness 的参考精度必须一致，否则同算子在不同 harness 上的"通过性"取决于参考质量而非 kernel 质量。
- 分块预算取 1GiB/8B=128M 元素：fp64 化后无分块会在大 shape（自动采样展开到全图）上 OOM；逐 ROI 归约独立性保证分块逐位等价。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

来源说明：本结论来自最终精度验收会话中大 shape 用例的参考精度修复，前后同环境实测（Ascend950PR，CANN 9.1.0）。

- 修复前：该用例 `MERE=1.909039e-05 PASS, MARE=7.216146e-03 FAIL`，最大差异 `Golden=0.001195, Actual=0.001165`（abs_err=3e-5 > abs_floor 2e-5），超阈值 21763/627200（来源平台：950；证据等级：tool_observed）
- 修复后：该用例 `结果: PASS`（通过率 1/1），golden 自测 PASS（来源平台：950；证据等级：tool_observed）
- 全量回归：`bash run.sh --torch` 全量 36/36（`汇总: 通过 36 / 失败 0 / 总计 36`）；C++ ST 同样 36/36（来源平台：950；证据等级：tool_observed）

## 失效边界

- fp64 参考只保证"参考侧舍入可忽略"，不能修复 kernel 自身的真实精度缺陷；若 abs_err 仍超 abs_floor，问题在 kernel。
- fp64 中间张量内存约 2×；不分块时大 shape（自动采样）会 OOM——分块参数按宿主内存预算调整。
- 本例在 CPU 上算 golden；若参考放到 NPU 上计算，fp64 支持与性能须另行验证（本卡未验证）。
- 其他平台未实测，未验证，推断可迁移（参考精度纪律与平台无关）。

**未决项**：

- 无（该闭环已收敛：单用例与全量均通过；参考精度纪律已体现在两套 harness）。
## 坏实践

- 参考实现与被测对象同精度（fp32 golden 验 fp32 kernel）：abs_err 变成双侧舍入之和，噪声界漂移、判定失真——这是本卡要根除的形态。
- 修完 Python 参考不清 `__pycache__` 直接复跑：旧字节码让修复"不生效"，误判方案无效。

## 误诊

- 最初归因"随机数生成器不同（std::mt19937 vs torch.Generator）导致数据不同"：数据确实不同，但这只解释分布差异，不解释 abs_err 超 noise 界；真正的差异在参考精度（C++ double vs Python fp32）。
- 一度提议继续放大 abs_threshold（2e-5→5e-5）：属测试侧放松方向，被"升参考精度"的根因解替代。
