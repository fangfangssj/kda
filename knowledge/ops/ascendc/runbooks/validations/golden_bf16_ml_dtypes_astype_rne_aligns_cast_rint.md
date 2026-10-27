---
type: "Runbook"
title: "golden bf16 输出用 ml_dtypes 而非 torch"
description: "torch bf16 张量无法转 numpy（TypeError: Got unsupported ScalarType BFloat16）；CPU golden 用 ml_dtypes.bfloat16 的 astype 做就近偶数舍入，与 kernel 侧 CAST_RINT 舍入语义一致。"
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# golden bf16 输出用 ml_dtypes 而非 torch

## 现象

为一个 bf16 输出算子（bf16 激活输入 / fp32 权重、内部全程 fp32、输出经就近舍入截断回 bf16）编写 TTK TestSpec 的 CPU 参考实现（golden.py，numpy 实现）。编写前探测 bf16 的 CPU 侧桥接能力：

```bash
python3 -c "import ml_dtypes; print('ml_dtypes available:', ml_dtypes.bfloat16)"
python3 -c "import torch; t = torch.tensor([1.0], dtype=torch.bfloat16); print('bf16 to numpy:', t.numpy().dtype)"
```

实测输出（逐字）：

```text
ml_dtypes available: <class 'ml_dtypes.bfloat16'>
---
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; print('torch available:', torch.__version__); t = torch.tensor([1.0], dtype=torch.bfloat16); print('bf16 to numpy:', t.numpy().dtype)
                                                                                                                                        ~~~~~~~^^
TypeError: Got unsupported ScalarType BFloat16
torch available: 2.7.1+cpu
[ERROR] 2026-08-26-00:48:15 (PID:2751301, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

即 torch（2.7.1+cpu）侧 **bf16 张量不能 `.numpy()`**——想在 golden 里用 torch 算完再交 numpy 比对，这条链在 bf16 上是断的。

## 适用场景

- 工程形态：TTK TestSpec golden（numpy CPU 参考实现），以及一切需要在 CPU 侧产出/比对 bf16 数组的算子测试脚手架。
- API 路径：`ml_dtypes.bfloat16` + `numpy.ndarray.astype`；输入侧 torch.Tensor → fp32 的绑定方法桥接；kernel 侧对应 `Reg::Cast<bfloat16_t, float, ...RoundMode::CAST_RINT>`（fp32→bf16 就近舍入）。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0，Python 3.13 + numpy 2.4.6 + torch 2.7.1+cpu + ml_dtypes。
- 适用粒度：`operator_family`——"bf16 输出 + fp32 内部计算"的算子 golden。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证：golden 全程 fp32/fp16 输出的算子（无 bf16 桥接问题）；kernel 侧非 CAST_RINT（如截断 RTZ）舍入模式的算子——那需要匹配不同舍入，未验证。

## 判别方法

两条探针即可判定（均为同会话实测）：

1. torch 桥接探针（上文现象节命令）：bf16 `.numpy()` 抛 `TypeError: Got unsupported ScalarType BFloat16` → torch 路线不通。
2. ml_dtypes 舍入行为探针：

```python
from ml_dtypes import bfloat16 as bf16
import numpy as np
# Test round-to-nearest-even for fp32->bf16
x = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, -0.1, 0.0], dtype=np.float32)
y = x.astype(bf16)
z = y.astype(np.float32)  # back to fp32 for display
```

实测输出（逐字）：

```text
fp32: [ 0.1      0.5      1.       1.5       2.       3.14159 -0.1      0.     ]
bf16->fp32: [ 0.10009766  0.5         1.          1.5         2.          3.140625
  -0.10009766  0.        ]
bf16 dtype name: bfloat16
bf16 itemsize: 2
```

0.1 → 0.10009766、3.14159 → 3.140625（就近舍入而非截断：截断应得 0.099609375 与 3.125）、1.5/2.0 精确不变——确认 `astype(bf16)` 是**就近偶数舍入（round-to-nearest-even）**，与 kernel 侧 fp32→bf16 的 `RoundMode::CAST_RINT` 舍入语义一致，golden 与 kernel 的终点 cast 不会因舍入模式不同而系统性偏差。

## 解决配方

golden 采用"numpy fp32 全程 + ml_dtypes 终点 cast"，输入输出两个桥接函数（逐字）：

```python
def _to_fp32(t):
    """Convert a torch.Tensor *or* numpy.ndarray to numpy float32.

    For bfloat16 tensors the conversion zero-extends the mantissa bits
    (torch ``.float()`` and numpy ``.astype(float32)`` both do this
    correctly), preserving the exact fp32 value represented by the bf16
    pattern — the "bit-bridge fidelity" required for binary_equal.
    """
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t.astype(np.float32) if t.dtype != np.float32 else t
    # torch.Tensor — call bound methods (no torch import needed in this module)
    if hasattr(t, "float") and callable(t.float):
        t32 = t.float()
        if hasattr(t32, "numpy") and callable(t32.numpy):
            return t32.numpy()
        return np.asarray(t32, dtype=np.float32)
    return np.asarray(t, dtype=np.float32)


def _cast_bf16(x_fp32):
    """Cast a float32 numpy array to bfloat16 (round-to-nearest-even)."""
    return x_fp32.astype(_bf16)
```

要点：

1. 输入桥：bf16 → fp32 只做尾数零扩展（无损），torch.Tensor 用**绑定方法** `.float().numpy()`（fp32 的 `.numpy()` 是支持的），全程不在模块级 `import torch`。
2. 输出桥：统一 `x_fp32.astype(ml_dtypes.bfloat16)`，就近偶数舍入对齐 kernel CAST_RINT。
3. golden 主体全部 numpy fp32 运算（`math`/`numpy`/`ml_dtypes` 三个模块级 import）。

## 设计决策

- 备选一：golden 里用 torch 全程计算再转 numpy——被排除，两个原因：torch bf16→numpy 直接 TypeError（本卡现象）；TestSpec 隔离契约禁止 golden 模块级 import torch（竞品 API 必须走 `third_party` 声明、由框架在隔离子进程执行）。
- 备选二：手写 bf16 舍入（位操作模拟）——被排除：ml_dtypes 的 `astype` 已提供就近偶数舍入且实测与 CAST_RINT 语义一致，无需重造。
- 采用：numpy fp32 全程 + ml_dtypes 终点 cast；属 `recommended_fastpath` 路线（无额外依赖成本，ml_dtypes 为环境既有包）。
- 邻接卡区分：[fp16/bf16 位流转 float 零依赖转换](cxx_fp16_bf16_bitstream_to_float_conversion.md) 与本卡重叠的只有"torch bf16 不能 `.numpy()`"这一事实——其场景是位流落盘走 uint16 位桥，本卡是数值语义 golden 产出（astype + RNE 对齐 + 隔离桥接），工程形态与配方不同；kernel 侧 RoundMode 按方向区分的判别见 [bf16 扩精度 Cast 误用 CAST_RINT 产垃圾值](../precision/bf16_cast_cast_rint.md)（扩精度一律 CAST_NONE、缩精度才 CAST_RINT）；fp16 侧窄化 RNE oracle 锚定见 [fp32 窄化舍入模式不一致触发稀疏 MARE 越线](../precision/fp32_to_fp16_cast_matches_rne_oracle.md)；golden 函数签名/调用契约（不带 self、未绑定调用）见 [TTK TestSpec golden 函数不得带 self 形参](ttk_testspec_golden_no_self_unbound_call.md)。

## 验证信号

以下信号来自同一次开发会话中该算子 golden.py 的编写与验收（环境探针、修复后冒烟与门禁检查）：

- torch bf16→numpy 探针抛 `TypeError: Got unsupported ScalarType BFloat16`（torch 2.7.1+cpu）；ml_dtypes 可用（来源平台：950；证据等级：tool_observed）
- ml_dtypes 舍入探针：fp32 [0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, -0.1, 0.0] → bf16→fp32 回显 [0.10009766, 0.5, 1.0, 1.5, 2.0, 3.140625, -0.10009766, 0.0]，dtype 名 bfloat16、itemsize 2——就近偶数舍入（来源平台：950；证据等级：tool_observed）
- 修复后 golden 以 C 头顺序未绑定调用冒烟：2 个输出 dtype 均 bfloat16、shape 正确，Prefill / Empty(L=0) / K=1 三场景 PASS（来源平台：950；证据等级：tool_observed）
- TTK 官方 validator PASS；SPEC-ORACLE-1 值级不变量（RMSNorm 保范性 4 case × 3 张量）全 PASS、退出码 0；黑盒 L0 CSV 多用例（含 D=769 非对齐、非空 conv_state）golden 绑定 PASS（来源平台：950；证据等级：tool_observed）
- golden.py 的 `_to_fp32`/`_cast_bf16` 实现与模块级 import 清单（math/numpy/ml_dtypes，无 torch/aclnn）按当次源码逐字核对（来源平台：950；证据等级：artifact_verified）

## 失效边界

- torch 后续版本若支持 bf16→numpy，"torch 路线不通"一条失效，但"就近偶数舍入对齐 CAST_RINT"的终点 cast 结论仍适用。
- 本卡验证的是 ml_dtypes `astype` 的默认舍入与 CAST_RINT 一致；若 kernel 改用其它 RoundMode（截断/饱和等），需另配舍入实现，本配方不覆盖。
- kernel 侧 CAST_RINT 对应关系在本会话 kernel 源码中核对（`Reg::CastTrait` 的 `RoundMode::CAST_RINT`），未做 golden↔kernel 的逐值对拍（逐值一致由该算子 ST 的 bitwise 通过用例间接佐证）。
- 仅 950 + CANN 9.2.0 环境实测；其它 CANN 版本的 CAST_RINT 语义推断一致（未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- golden 里 `import torch` 后试图 `torch_tensor(bf16).numpy()` —— 直接 TypeError，且违反 TestSpec 的 golden 隔离契约。
- 用截断式转换（如先乘 256 取整再除）模拟 bf16 —— 舍入模式与 CAST_RINT 不一致，会产生系统性偏差。

## 误诊

- 无。
