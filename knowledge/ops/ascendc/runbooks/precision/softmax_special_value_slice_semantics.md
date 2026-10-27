---
type: "Runbook"
title: "softmax 特殊值切片语义：含 +inf 全 NaN 非 one-hot"
description: "max-shift 稳定 softmax 的特殊值切片契约（经 torch 交叉验证）：切片含 NaN/+inf 或全 -inf 输出整切片全 NaN（非 one-hot、非全 0）；混合 -inf 时 -inf 位恰为精确 0 且有限位归一化；全零与 fp16 溢出输入正常归一化为 1/n。golden 须直调参考 API 并配值级不变量自检。"
tags: [task.testing, topic.precision, topic.compute.normalization]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["agnostic"]
---

# softmax 特殊值切片语义：含 +inf 全 NaN 非 one-hot

## 现象

为 softmax 类算子（沿 `dim` 归约的 max-shift 数值稳定 softmax，输入/输出 dtype float16/float32/bfloat16）准备黑盒 ST 的 golden 参考实现（对标 `torch.nn.functional.softmax`）时，特殊值切片语义与两种常见直觉相反。用 torch 直接探测（CPU，`dim=-1`）得到以下实测行为：

| 输入切片 | torch 输出 | 手写 max-shift 公式输出 |
|---|---|---|
| `[nan, 1..7]`（含 NaN） | 整切片全 NaN | 一致 |
| `[inf, 1..7]`（单个 +inf 与有限值共存） | 整切片全 NaN（**非 one-hot**） | 一致 |
| `[1, inf, inf, 4]`（多 +inf） | 整切片全 NaN | 一致 |
| `[-inf]*8`（全 -inf） | 整切片全 NaN（**非全 0**） | 一致 |
| `[inf, -inf, 1.]`（±inf 共存） | 整切片全 NaN | 一致 |
| `[-inf, 1..7]`（混合 -inf + 有限） | `-inf` 位**恰为精确 0.0**，有限位正常归一化（如 `[0., 0.0015683, 0.00426308, ...]`） | 一致 |
| `[0.]*8`（全零） | 均匀 `1/8 = 0.125` | 一致 |
| `[60000]*8`（fp16，溢出量级） | `[0.125]*8`（max-shift 后无溢出，sum==1） | 一致 |
| rank-0 标量 | `1.0` | 一致 |
| 空 Tensor（shape `[0,4]`） | 同 shape 空 Tensor | — |

两种被证伪的直觉：①"NaN 逐元素传播，只有 NaN 位置输出 NaN"；②"+inf 当大数处理，最大元素拿到全部概率（one-hot）"。若 golden 或 kernel 按直觉实现，在含 inf/NaN 的测试切片（本例黑盒用例中 nan 切片约占 2858/10000、+inf 476、-inf 952）上会整批失配。

## 适用场景

- 工程形态：AscendC 算子开发中为 softmax/归约复合类算子编写 TTK 黑盒 ST 的 golden 参考实现（TestSpec 插件），或调试此类算子在 inf/NaN 输入切片上的精度失配。
- API 路径：参考 oracle `torch.nn.functional.softmax(x, dim=...)`；TTK TestSpec golden 插件（`__spec__` 注册 + `golden()` 返回 list）；numpy/ml_dtypes bf16 位桥。
- 验证过的平台与 CANN 版本：特殊值契约按 IEEE 754 重结合语义成立（platforms: agnostic）；验证在开发 host 的 torch CPU 参考实现 + 手写公式对照完成。NPU kernel 侧对该契约的符合性属后续联调阶段。
- 适用粒度：operator_family（softmax/单轴归约复合算子族的精度契约与 oracle 配方）。
- 路线优先级：recommended_fastpath（golden 直调参考 oracle API，而非手写公式重实现特殊值分支）。
- 明确不适用 / 未验证场景：多轴（axes 列表）softmax 未验证；非 max-shift 的朴素 `exp(x)/Σexp(x)` 实现在含 +inf 时表现为 `inf/inf=NaN` 或溢出差异，不可混用本表期望值；masked softmax / 带 -inf 白名单重归一化的变体不适用；NPU 上板一致性未验证；torch 版本变更后特殊值语义需重新探测。

## 根因/解决配方

特殊值契约的机制链（max-shift 稳定 softmax `m = max_j x_j; e_j = exp(x_j - m); y_j = e_j / Σ e_k`）：

- **含 NaN**：`m = max(...)` 中 NaN 参与比较即传播 → `m = NaN` → `x_j - m` 全 NaN → `e` 全 NaN → 分母 NaN → 整切片 NaN。（另一表述：`v > m` 比较对 NaN 恒为 false，max 不传播 NaN，但 `exp(NaN - m) = NaN` → `s = NaN` → 全切片 `y = e/NaN = NaN`。）
- **含 +inf（与有限值共存）**：`m = +inf`；inf 元素处 `x_j - m = inf - inf = NaN`；其余元素 `x_j - m = -inf → exp = 0`；分子含 NaN、分母 `Σ = NaN` → 整切片 NaN。直觉的 one-hot 不成立，因为分母被 NaN 毒化。
- **全 -inf**：`m = -inf`，每个元素 `x_j - m = -inf - (-inf) = NaN` → `e` 全 NaN → 整切片 NaN（0/0 形态，而非 0）。
- **混合 -inf + 有限**：`m` = 有限最大值 → -inf 位置 `-inf - m = -inf → e = exp(-inf) = 0`（**精确 0，非下溢**）→ 该位 `y = 0/s = 0` 恰为 0；有限位正常归一化。
- **全零**：`m = 0`，`e_j = exp(0) = 1`，`s = n` → `y = 1/n` 精确。
- **fp16 溢出量级（如全 60000）**：朴素 `exp(60000)` 会溢出为 +inf；max-shift 后 `x - m = 0 → e = 1`、`s = 8` → `y = 0.125`，无任何溢出——这正是 max-subtraction 的数值稳定意义。
- **rank-0 标量**：归约切片只有一个元素，`y = exp(x-x)/exp(x-x) = 1`。

golden 三原则（解决配方）：

1. golden **直调参考 oracle API**（`torch.nn.functional.softmax`）而不是手写 max-shift 公式——特殊值语义由参考实现背书，杜绝自行发挥分支；
2. 用 numpy 直实现（`_numpy_softmax`：fp32 上 `m = np.max(x, axis, keepdims); e = np.exp(x-m); y = e/e.sum(...)`）做**交叉校验**（fp32 上 allclose 1e-5/1e-6），确保"直调 API"没有 silently 换语义；
3. 落一段**值级不变量自检脚本**随 golden 交付，在随机合法输入上跑 `nonneg (y >= 0)`、`bounded_above (y <= 1)`、`sum_to_one`，外加特殊值场景逐一与 torch 对拍（`equal_nan=True`），全过（EXIT=0）后才允许当 oracle 使用。

golden 参考实现核心（verbatim，直调 torch API + bf16 int16 位桥）：

```python
__spec__ = {"bench_harness_softmax": "BenchHarnessSoftmaxTestSpec"}

def _to_torch(x):
    """numpy ndarray → torch.Tensor，保 dtype；bf16 走 int16 位桥（ml_dtypes ↔ torch.bfloat16）。
    numpy 无原生 bf16：ml_dtypes.bfloat16 → view int16 → torch.from_numpy → view torch.bfloat16，
    按 bit 保真，不经 fp32 中转（避免丢精度）。"""
    if isinstance(x, torch.Tensor):
        return x
    arr = np.asarray(x)
    if arr.dtype.name == "bfloat16":
        arr = np.ascontiguousarray(arr)
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)

def softmax(x, dim=-1, **kwargs):
    dim = _norm_dim(dim)          # TTK 可能以 numpy 0-d array 传标量属性
    xt = _to_torch(x)
    # spec.reference_oracle: torch.nn.functional.softmax（含 NaN/+inf/-inf 切片语义）
    yt = torch.nn.functional.softmax(xt, dim=dim)
    return _to_np(yt)

class BenchHarnessSoftmaxTestSpec:
    def golden(x, dim=-1, **kwargs):
        return [softmax(x, dim=dim)]
    tolerance = {
        "float16":  {"standard": "stat_rel_err"},
        "float32":  {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
```

特殊值对拍探针（自检脚本核心，`rtol=0, atol=0, equal_nan=True` 逐位判定）：

```python
SPECIAL_CASES = [
    ("single NaN -> whole slice NaN",
     np.array([[1., 2., np.nan, 4., 5.]], dtype=np.float32), -1),
    ("single +inf -> whole slice NaN",
     np.array([[1., 2., np.inf, 4., 5.]], dtype=np.float32), -1),
    ("all -inf -> whole slice NaN",
     np.array([[-np.inf, -np.inf, -np.inf]], dtype=np.float32), -1),
]
for name, x, dim in SPECIAL_CASES:
    y = softmax(x, dim=dim)
    y_torch = torch.nn.functional.softmax(torch.from_numpy(x), dim=dim).numpy()
    ok = bool(np.allclose(y, y_torch, equal_nan=True, rtol=0, atol=0)) and bool(np.all(np.isnan(y)))
```

## 预防

- 写 softmax 族 golden 前，先用一组最小探针（单 NaN / 单 +inf / 全 -inf / 混合 -inf / 全零 / 标量）确认参考 oracle 的特殊值行为，再写实现。
- golden 落地必须配不变量自检脚本并跑 EXIT=0 才当 oracle；自检须含特殊值场景（`equal_nan=True` 对拍）与 numpy 公式交叉校验。
- 含 inf/NaN 的黑盒用例集是对拍时最先暴露契约分歧的入口，不要因为"特殊值难对齐"而删减这类用例。
- 比对口径配套：NaN 切片用 NaN 感知比对（isnan 同位）、混合 -inf 的 -inf 位用**精确 0** 逐位判定（相对误差会放过非零下溢值）。

## 验证信号

本卡契约与实现来自 softmax 类算子黑盒 ST golden 开发会话：先用 torch 探针确认特殊值行为，随后实现 golden 与不变量自检并全量跑通，验收阶段独立复跑通过。

- 特殊值探针实测：单 NaN 切片 → 全 NaN；单 +inf → 全 NaN；多 +inf → 全 NaN；全 -inf → 全 NaN；±inf 共存 → 全 NaN；混合 -inf 时 -inf 位精确 0.0、有限位归一化；全零（长度 8）→ 每元素 0.125；标量 → 1.0；空 Tensor shape 保持 `[0,4]`；bf16 沿 dim 求和 = 1（来源平台：agnostic—host CPU 上 torch 参考实现与手写公式交叉验证；证据等级：tool_observed）
- 正常 fp32 随机输入 [4,8]：torch 与手写 max-shift 公式逐元素 max diff = 2.9802322e-08（fp32 舍入量级），每行和≈1（来源平台：agnostic—host CPU；证据等级：tool_observed）
- 值级不变量自检 74/74 PASS、EXIT=0：10 组随机 shape×dtype×dim（含负索引 dim=-2、归约轴长度 1、rank-0 标量）的 shape/dtype 保持、nonneg、bounded_above、sum_to_one；fp32 上与 numpy 公式直实现交叉校验一致；11 组特殊值场景与 torch 逐位一致（rtol=0, atol=0, equal_nan=True）；fp16 60000 上溢边界无 NaN/Inf 且 sum==1；bf16 NaN 场景一致（来源平台：agnostic—host CPU torch/numpy/ml_dtypes；证据等级：tool_observed）
- golden.py 可独立加载（importlib 加载退出码 0），import 仅 numpy/torch/ml_dtypes，无任何被测 kernel 产物引用（来源平台：agnostic—host CPU；证据等级：tool_observed）
- 验收阶段独立复跑不变量自检全部通过（来源平台：agnostic—host CPU；证据等级：tool_observed）

## 失效边界

- **NPU 上板一致性未验证**：本卡只在 CPU 参考实现侧验证契约；kernel 与该契约的对拍属后续联调，若 kernel 在特殊值用例失配，应按本卡机制链排查 kernel 的 max/exp/sum 实现，而不是改 golden。
- 契约绑定 `torch.nn.functional.softmax` 的当前行为；参考框架升级后特殊值语义若变化需重新探测。
- 仅验证单轴 `dim`（int，支持负索引）；axes 列表型多轴 softmax 未验证。
- 空 Tensor（shape 含 0）契约由 golden 保证，kernel 侧行为属另一闭环。
- 推断可迁移（未验证）：任何严格按 max-shift 链实现的 softmax 在 IEEE 语义下应自然得到同契约；fp16/bf16 下 kernel 的 exp/max 精度差异可能引入非 NaN 切片上的正常容差内偏差。

**未决项**：

- NPU kernel 实现与本契约的对拍结果待联调阶段验证；若失配，优先按机制链逐环节（max 归约 / x−m / exp / sum / div）定位首个引入偏差的环节。
- bf16 位桥通路在 TTK 框架真实字段流转（CSV → 框架 → golden）中的端到端保真性待黑盒执行阶段确认。
- 多轴（axes 列表）softmax 的同构契约未探测。
## 坏实践

- golden 手写 `exp(x)/sum(exp(x))` 不减最大值：fp16 大值切片（如 60000）直接上溢为 NaN/Inf。
- golden 手写公式并自行处理特殊值分支（把 +inf 当 finite、全 -inf 切片输出 0、NaN 只逐元素传播）：与 torch 参考对拍时含特殊值的整批用例误报，或更糟——错误 oracle 把正确 kernel 判为失败。
- 对含 +inf 切片写 one-hot 期望——与 torch 口径不符，会把正确实现误判为失败。
- 对混合 -inf 切片只用 rtol/atol 相对误差比对——-inf 位置期望精确 0（`y == 0` 逐位判定）。
- 只誊抄 spec 公式不做随机输入不变量自检：特殊值语义偏差不可见，直到上板对拍才暴露且难以定位是 golden 还是 kernel 的错。
- 为"对齐"而删减/跳过含 inf/NaN 的测试用例：属测试侧放松，掩盖真实契约分歧。

## 误诊

- 无（本闭环内未出现错误归因；特殊值行为均为探针实测后按机制链解释）。
