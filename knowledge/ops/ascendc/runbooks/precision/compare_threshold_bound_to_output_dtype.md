---
type: "Runbook"
title: "比对阈值随 golden dtype 漂移：按算子输出 dtype 取阈"
description: "精度阈值的语义是\"允许算子输出相对参考的偏差上限\"，应由算子输出 dtype 决定；比较器按 golden.dtype 取阈时，golden 上提 fp32（或未按输出 dtype 返回）会把 fp16 用例的阈值静默漂到 2^-13（收严 8 倍）整批假 FAIL。两条独立轨迹修复：golden 按契约返回目标 dtype / 比较器显式 threshold_dtype 参数。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# 比对阈值随 golden dtype 漂移：按算子输出 dtype 取阈

## 现象

两条独立轨迹（均为 Ascend950PR、fp16 输入输出的自定义算子 PyTorch ST）命中同一误判形态：**fp16 用例被按 fp32 阈值 1.22e-04（2^-13）判定，而 fp16 应为 9.77e-04（2^-10）**，偏差本身只有 1 个 fp16 ulp 量级，属正常舍入差——整批假 FAIL。

**形态一（golden 未按输出 dtype 返回）**：BenchNoHarnessMoeFinalizeRouting 算子（mode0 drop_less 按列、全可选输入、fp16）NPU 首跑门槛用例即失败：

```
golden 输出: shape=(1024, 512), dtype=torch.float32, 范围=[-2.2584, 2.3271]
  [FAIL] MERE=1.76e-04 (threshold=1.22e-04), MARE=6.06e-04 (threshold=1.22e-03)
  最大差异位置: [41013] Golden=2.327150, Actual=2.328125
```

三个矛盾点：① 用例是 fp16，golden 输出却是 `dtype=torch.float32`；② 阈值用了 1.22e-04（fp32 档）；③ 最大差异点相差 0.000975 ≈ 1 个 fp16 ulp——数值在 fp16 表示精度内，不构成真实缺陷。当时代码状态：golden.py 的 `_round_to_dtype` 把数据预舍入到目标 dtype 后又转回 float32 返回；算子输出为 fp16。

**形态二（golden 为规避近零 MARE 上提 fp32）**：Conv3D filter 梯度反向算子的 compare.py 原按 `get_threshold(golden.dtype)` 取阈；为规避 golden 近零值 MARE 爆炸，把 golden 与 NPU 结果都 upcast 到 float32 再比对。改完后 MERE ≈ 0.8e-3~1.0e-3（与之前 fp16 比对时同量级），但每条都判 FAIL，输出阈值显示 `1.220703e-04`——golden 提精度后阈值跟着 golden 的 dtype 漂移、被收严 8 倍。

## 适用场景

- 工程形态：算子仓开发的 PyTorch 接入 ST（torch_npu + 自编译 torch_adapter 调 aclnn）与自研 MERE/MARE 比较器；任何"golden 计算精度 ≠ 算子输出精度"的混合比对。
- API 路径：CANN 生态算子精度标准各 dtype Threshold（fp32 2^-13=1.22e-04 / fp16 2^-10=9.77e-04 / bf16 2^-7=7.81e-03）；PyTorch `Tensor.to(dtype)` 舍入语义；自研 compare.py 的 `get_threshold`/`compare_results`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0 + torch 2.7.1 + torch_npu（两条独立轨迹均实测）。
- 适用粒度：platform_wide（golden 建模契约与 dtype 阈值选择，不绑定算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：算子契约本身输出 fp32（golden 应保持 fp32，不套用形态一修复）；golden 与算子输出同 dtype 直接比对的场景不会触发（但显式传参同样无害）。

## 根因

阈值选择依据错了对象：精度阈值的语义是"允许算子输出（fp16）相对高精度参考的偏差上限"，应由**算子输出 dtype** 决定（fp16→2^-10）；而比较器把阈值绑定到了 **golden 张量的 dtype**。golden 一旦为规避近零 MARE 上提到 float32（形态二），或因 `_round_to_dtype` 返回 `values.to(dtype).float()` 使低精度检测失效而输出 fp32（形态一），阈值就漂移到 float32 档（2^-13，比 2^-10 严 8 倍），fp16 输出相对 fp32 golden 的固有偏差（MERE≈1e-3 量级）必然超阈，造成整批假 FAIL。换算：2^-10 = 9.77e-4，2^-13 = 1.22e-4，9.77e-4 / 1.22e-4 = 8。

形态一的机制链：`_round_to_dtype` 返回 `values.to(dtype).float()`——值已舍入到目标 dtype 精度，但张量 dtype 是 float32；golden 函数以输入张量的 dtype 检测低精度（检测到 float32 → `_low_prec=False`），累加后输出不再转回 fp16。于是 golden 输出 fp32、算子输出 fp16，比较器按 golden 的 dtype 取阈值。

## 解决配方

两条互补修复（按触发面选择；也可并用）：

**修复 A（golden 侧：按契约返回目标 dtype）**——适用于"golden 应与算子输出同 dtype"的场景（形态一）：

```python
# 修复前：值舍入到目标 dtype 但张量是 float32，golden 低精度检测失效
def _round_to_dtype(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return values.clone()
    return values.to(dtype).float()

# 修复后：返回目标 dtype 的 tensor，使 golden 的低精度检测、fp32 上抛累加、
# 输出转回三个环节全部对齐算子输出 dtype
def _round_to_dtype(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """将 float 值舍入到目标 dtype（返回目标 dtype 的 tensor）。

    返回目标 dtype 而非 float32，使 golden 函数能正确检测低精度并执行
    fp32 累加 + 输出转回原 dtype 的流程，与 C++ RoundToFp16/RoundToBf16 一致。
    """
    return values.to(dtype)
```

修复后数据流：`generate_inputs` 产出 fp16 张量 → golden 检测 `_low_prec=True` → 上抛 fp32 累加 → `out.to(original_dtype)` 输出 fp16 → 比较器按 fp16 Threshold 9.77e-04 比较。混合精度（bf16 主路径 + fp32 scales）天然正确。

**修复 B（比较器侧：显式 threshold_dtype 参数）**——适用于"golden 刻意保持高精度（fp32 比对空间）"的场景（形态二）：

```python
# 修复前：阈值随 golden dtype 漂移
def compare_results(golden: torch.Tensor, actual: torch.Tensor, verbose: bool = True) -> bool:
    # ... 浮点 MERE/MARE 比对 ...
    threshold = get_threshold(golden.dtype)   # <-- 阈值随 golden dtype 漂移
    mare_threshold = 10 * threshold
    ...

# 修复后：增加 threshold_dtype 参数（函数级关键段）
def compare_results(golden: torch.Tensor, actual: torch.Tensor, verbose: bool = True,
                    threshold_dtype: torch.dtype = None) -> bool:
    """
    比对 golden 与实际结果，按 MERE/MARE 阈值判定

    threshold_dtype: 阈值依据的 dtype（默认为 golden.dtype）。
        当 golden 用高精度（float32）计算、actual 为低精度（float16）时，
        应传入算子输出 dtype（float16）以使用对应阈值。
    """
    # ... shape/dtype 校验与整型分支、误差度量略 ...
    threshold = get_threshold(threshold_dtype if threshold_dtype is not None else golden.dtype)
```

调用侧（test.py）显式传算子输出 dtype：

```python
    # 取回结果，upcast 到 float32 做比对（避免 fp16 比对精度损失）
    result_cpu = result.cpu().float()
    golden_f32 = golden.float()

    # 精度比对（混合误差度量，fp16 阈值）
    passed = compare_results(golden_f32, result_cpu, verbose=True, threshold_dtype=tc["dtype"])
```

判别信号（下次遇到先看这两点再动 kernel）：① fp16/bf16 用例的 FAIL 行出现 1.22e-04 等 fp32 阈值；② 日志"golden 输出"的 dtype=torch.float32 而用例 dtype 是低精度。

## 设计决策

- **修复 A vs 修复 B 的取舍**：A 与 C++ ST 的 RoundToFp16/RoundToBf16 + golden 低精度检测流程一致，比较器无需新增参数，混合精度无需特判；B 保留 compare_results 的独立可测性（不传参时行为向后兼容），让"阈值依据"在调用点可见，且允许"fp32 比对空间 + fp16 阈值"这一规避近零 MARE 的组合成立（fp16 比对空间有自身的近零/MARE 问题，参见近零 MARE 判别相关卡）。两条路线都把阈值绑定回算子输出 dtype，语义等价。
- **为什么不反过来把 golden 留在 fp16（形态二场景）**：fp16 比对空间有自身的近零/MARE 问题；float32 比对空间 + fp16 阈值是精度语义正确的组合。
- **为什么不放宽容差**：阈值该是多少是由 dtype 契约决定的，不是调参项。

## 验证信号

来源说明：以下信号来自两条独立轨迹的 PyTorch ST 修复闭环（均为 NPU 实测，Ascend950PR + CANN 9.1.0）。

形态一（golden 返回目标 dtype 修复）：

- 修复前：fp16 门槛用例 `MERE=1.76e-04 (threshold=1.22e-04) FAIL`，golden 输出 dtype=torch.float32，最大差异 1 个 fp16 ulp（来源平台：950；证据等级：tool_observed）
- 修复后同用例：`golden 输出: dtype=torch.float16`，`[PASS] MERE=1.46e-08, MARE=9.02e-04 (threshold=9.77e-04)`（来源平台：950；证据等级：tool_observed）
- 修复后全量 57 用例（4 L0 + 53 L1）在 NPU 全部 PASS，阈值随 dtype 正确变化：fp32 用例 threshold=1.22e-04、bf16 用例 threshold=7.81e-03、混合精度 bf16+fp32 scales 用例 threshold=7.81e-03；golden 手工自测 11/11 不受影响（来源平台：950；证据等级：tool_observed）

形态二（threshold_dtype 参数修复）：

- 修复前假 FAIL（golden 已上提 fp32）：`MERE: 9.937900e-04 (阈值: 1.220703e-04) FAIL`，L0 全部按 1.22e-4 判 FAIL（来源平台：950；证据等级：tool_observed）
- 阈值行为探针：`fp16 threshold: 0.0009765625` / `fp32 threshold: 0.0001220703125`；`compare_results(..., threshold_dtype=torch.float16)` 打印阈值 9.765625e-04，不传时 1.220703e-04（来源平台：950；证据等级：tool_observed）
- 修复后单用例：`L0-01 结果: PASS (MERE=3.125183e-04, MARE=3.027344e-02, 阈值=9.765625e-04)`；全量 L0 5/5 PASS（MERE 2.5e-4~3.3e-4 < 9.77e-4）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 阈值表本身必须与算子精度规格一致（fp16 2^-10 / bf16 2^-7 / fp32 2^-13）；规格变更时同步。
- threshold_dtype 传错（例如传了 golden 的 dtype）会回到原问题，调用点需评审。
- 修复 A 仅适用于"参考实现以算子输出 dtype 为契约"的 golden 建模；若算子契约就是输出 fp32（如某些混合精度输出），golden 应保持 fp32，本修复不适用。
- 阈值仍按输出 dtype 选取的前提是算子 dtype 校验已保证输出 dtype 与输入主 dtype 一致；若算子支持输出 dtype 独立指定，比较器需按输出 dtype 显式取阈值（未覆盖）。
- 两条轨迹均为 950 + CANN 9.1.0 + torch 2.7.1 实测；其他环境未验证，推断可迁移（阈值-dtype 绑定语义与平台无关）。

**未决项**：

- 无（两条轨迹的机制与修复均已闭环）。
## 坏实践

- 比对阈值隐式地从某个张量的 dtype 推导——golden 提精度的重构会静默改变判定口径。
- 把这类 FAIL 当作算子精度缺陷直接去改 kernel——最大差异恰为一个 fp16 ulp 时应先核对 golden/阈值 dtype。
- 用放宽容差掩盖 dtype 建模错误——阈值该是多少是由 dtype 契约决定的，不是调参项。
- 排查此类问题时先怀疑算子精度而不先打印实际生效的阈值。

## 误诊

- 形态二修复前一度把全量 FAIL 当成"fp32 golden 与 fp16 kernel 偏差真的大到超阈"；打印阈值后发现是阈值档位漂移，偏差本身没有变化（MERE 数值与 fp16 比对时同量级）。

## 预防

- 比对器接口从设计上把"阈值 dtype"作为显式参数，调用点传算子输出 dtype；CI 里加一条阈值自检（构造已知偏差的合成张量验证判定）。
- 为 fp16/bf16 算子写 torch golden 时，先查 spec 的输出 dtype 契约：golden 的输入 dtype 与算子一致、比对在高精度空间、阈值按算子输出 dtype，三件套写成模板。
