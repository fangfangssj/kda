---
type: "Runbook"
title: TTK 采样将 nan→0、inf→±max，CSV 非有限值域不物化
description: TTK CSV input_data_ranges 保留 nan/inf 字面量，但 RandomData._digitize_inf_nan 采样时数字化为有限值，NaN/Inf 传播用例在 TTK 执行层永不触发。
tags: [task.testing, topic.compute.random_sampling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["agnostic"]
---
# TTK 采样将 nan→0、inf→±max，CSV 非有限值域不物化


## 现象

验收 Branch-0 白盒用例（TTK kernel 模式）时，为核实特殊值用例是否真实物化，逐一检查用例 CSV 与 TTK 源码：用例 CSV 的 `input_data_ranges` 列里**保留了 nan/inf 字面量**——结构扫描用例有 `((nan, nan), (nan, nan))`（bf16/fp16/fp32 各档）与 `((-inf, inf), (-inf, inf))` 行，黑盒 aclnn 模式 CSV 同样含 `(nan,nan)` 行——表面上 NaN/Inf 边界场景已被用例覆盖。

追查 TTK 仓（/workspace/ops-test-kit）源码发现：输入采样器 `RandomData`（ttk/utilities/data.py:51）在采样时把非有限 range 边界**数字化**——`_digitize_inf_nan` 把 nan 替换为 0、±inf 替换为该 dtype 取值域边界（如 fp16 ±65504）。即 `(nan, nan)` 用例实际生成全 0 输入（退化为全零面用例），`(-inf, inf)` 用例实际生成 ±dtype_max 大幅值**有限**输入（是有意义的幅值边界测试，但不是 Inf）。整套 TTK 用例（白盒 kernel 模式 + 黑盒 aclnn 模式）执行时不会产生任何非有限输入值，NaN/Inf 的 IEEE 传播验证在 TTK 执行层全部不触发。

旁证：白盒统一入口（ascendc-whitebox-design 的 whitebox_designer.py）在 **e2e 模式**导出时也主动把非有限 range 映射为有限（`_finite_range`：nan→0.0、inf→±0.4×dtype-max），其 docstring 自述 "TTK's RandomData cannot sample inf/nan, so inf/nan edge scenarios become large-magnitude boundary tests instead of breaking the CSV float parser"——两条路径殊途同归：TTK 用例体系的取值域通道不会物化非有限输入。

## 适用场景

- 工程形态：白盒/黑盒用例生成与执行（TTK kernel / aclnn / e2e 模式）、覆盖度审计中对 valclass=nan/inf 用例的解读。
- API 路径：`ttk/utilities/data.py` 的 `RandomData`（`_digitize_inf_nan`）；`ttk/core_modules/npu/op/input_generation.py`（按 data_range 构造 RandomData）；CSV `input_data_ranges` 列。
- 验证过的平台与 CANN 版本：宿主机 TTK 仓源码级检视，未上板；输入生成在宿主机 numpy 完成、结论按定义与 NPU 型号无关（frontmatter `platforms: [agnostic]`）。CANN 版本未涉及。
- 适用粒度（transfer_scope）：platform_wide（TTK 采样器语义，跨算子）。
- 路线优先级：recommended_fastpath——NaN/Inf 传播验证路由到 C++ ST 代码内构造；±dtype_max 幅值边界可继续由 (-inf,inf) range 用例承担并如实解读其语义。
- 明确不适用 / 未验证场景：本卡为源码级静态结论，(nan,nan)→全 0、(-inf,inf)→±dtype_max 的实际物化未跑用例核对；`_digitize_inf_nan` 调用点未逐行定位；e2e 模式 `_finite_range` 映射未实际生成 CSV 验证；TTK 版本差异未查证。

## 根因

TTK `RandomData._digitize_inf_nan` 在输入采样时把 range 中的非有限边界替换为有限值（isinf → dtype 取值域边界 ±max、isnan → 0），使 CSV 中保留的 nan/inf 字面量在执行层永不物化为非有限输入。

## 判别方法

（本会话实际执行的探针链）

1. 确认 kernel 模式 CSV 字面量被保留（导出不映射）：

```bash
grep -n "nan" operators/ThnnFusedGruCellGrad_package/tests/whitebox/branch-0/ttk_design_doc_cases.csv | head -3
```

观测：case00029 行 input_data_ranges 为 `"((nan, nan), (nan, nan))"`、case00026 为 `"((-inf, inf), (-inf, inf))"`。

2. 定位采样器实现：

```bash
grep -rn "class RandomData" /workspace/ops-test-kit/ttk --include=*.py | head -3
```

观测：`/workspace/ops-test-kit/ttk/utilities/data.py:51:class RandomData:`。

3. 读采样器源码确认数字化语义（见 ## 完整代码 verbatim）。

4. e2e 路径旁证（统一入口自文档）：

```bash
grep -n "_finite_range\|cannot sample inf/nan" .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py | head
```

## 完整代码

（verbatim，TTK 仓 `ttk/utilities/data.py` 51–114 行核心段，本会话 read 工具读取原文）

```python
class RandomData:
    def __init__(self, dtype: str, shape: Union[list, tuple],
                 data_range: Union[list, tuple]):
        self._dtype = resolve_custom_numpy_dtypes([dtype])[0]
        self._shape = list(shape)
        self._data_range = self._replace_none_in_data_range(data_range)

    @property
    def data_range(self):
        return self._data_range

    def generate(self, distribution: str = 'uniform'):
        if self._dtype == "uint1":
            np_uint8 = self._random('uint8', self._shape, distribution=distribution)
            np_bool = np_uint8.astype('bool', copy=False)
            np_array = numpy.packbits(np_bool)
        elif "complex" in str(self._dtype):
            if self._dtype == "complex32":
                shape = list(self._shape) + [1]
                real = self._random('float16', shape, distribution=distribution)
                imag = self._random('float16', shape, distribution=distribution,
                                    is_complex_imag=True)
                np_array = numpy.concatenate((real, imag), axis=-1)
            else:  # complex64 / complex128
                bits = eval(str(self._dtype)[7:])
                fp_dtype = f"float{bits // 2}"
                real = self._random(fp_dtype, self._shape, distribution=distribution)
                imag = self._random(fp_dtype, self._shape, distribution=distribution,
                                    is_complex_imag=True)
                np_array = numpy.array(real + imag * 1j)
        elif "e8m0" in str(self._dtype):
            # float8_e8m0 only represents positive scale powers; generate a positive range.
            low = float(self._data_range[0])
            high = float(self._data_range[1])
            if not numpy.isfinite(low) or low <= 0:
                low = 2 ** -3
            if not numpy.isfinite(high) or high <= low:
                high = 2 ** 7
            f32 = numpy.random.uniform(low, high, self._shape).astype('float32')
            from .dtypes import numpy_float8_e8m0
            np_array = f32.astype(numpy_float8_e8m0())
        else:
            np_array = self._random(self._dtype, self._shape, distribution=distribution)
        return np_array

    @staticmethod
    def _replace_none_in_data_range(data_range: Union[list, tuple]) -> list:
        low, high = get(data_range, 0), get(data_range, 1)
        if low is None:
            low = DEFAULT_LOW
        if high is None:
            high = DEFAULT_HIGH
        return list([low, high, *data_range[2:]])

    @staticmethod
    def _digitize_inf_nan(r, dtype: str):
        tmp = numpy.array([r], dtype="float64")
        if numpy.isinf(tmp):
            dtype_range = get_dtype_range(dtype)
            return dtype_range[0] if numpy.isneginf(tmp) else dtype_range[1]
        elif numpy.isnan(tmp):
            return 0
        else:
            return tmp[0]
```

数字化语义（机制推导点破）：`_digitize_inf_nan(nan)` → `numpy.isnan` 命中 → 返回 `0`；`_digitize_inf_nan(-inf, 'float16')` → `numpy.isneginf` 命中 → 返回 `get_dtype_range('float16')[0]` = −65504.0；`+inf` → +65504.0。故 `(nan, nan)` 的均匀采样区间实际为 [0, 0]（全零），`(-inf, inf)` 为 [−65504, 65504]（大幅值有限值）——非有限值域字面量与实际采样语义不一致。

（旁证，verbatim 摘自 whitebox_designer.py 内嵌 `_finite_range` 模板源码，去掉字符串转义后；该映射仅用于 e2e 行转换）

```python
def _finite_range(pair, dtype: str):
    """Map a (lo, hi) range to finite, TTK-parseable floats.

    Strings/inf → ±0.4×dtype-max (preserving sign); nan → 0.0. Finite numbers pass through.
    TTK's RandomData cannot sample inf/nan, so inf/nan edge scenarios become large-magnitude
    boundary tests instead of breaking the CSV float parser.
    """
    ...
    def _fin(x, default):
        if isinstance(x, str):
            xl = x.strip().lower()
            if "nan" in xl:
                return 0.0
            if "inf" in xl:
                return -0.4 * m if xl.startswith("-") else 0.4 * m
        ...
```

## 设计决策

- 特殊值覆盖分层：TTK 层保留 range 用例承担「可物化取值形态」（(-inf,inf) 物化后的 ±dtype_max 仍是有效幅值边界测试，与显式大值用例同族），非有限值传播交 C++ ST——不采用「删除 nan/inf range 用例」（幅值边界价值保留），也不夸大其覆盖主张。
- 本卡路线属 recommended_fastpath（分层承载 + 如实解读）。

## 系统信息

- TTK（ops-test-kit）仓：宿主机 `/workspace/ops-test-kit`（环境中另有多份 checkout：`/workspace/ttk-dev/ttk` 等，检视取 `/workspace/ops-test-kit`）；关键文件 `ttk/utilities/data.py`、`ttk/core_modules/npu/op/input_generation.py`（按 `input_data_ranges` 逐输入构造 `RandomData(dtype, ori_shape, data_range)`，另有手工输入文件读取通路）。
- 白盒统一入口：`.opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py`（e2e 路径 `_finite_range`，约 2960–3035 行内嵌模板段）。
- 用例产物：`operators/ThnnFusedGruCellGrad_package/tests/whitebox/branch-0/ttk_design_doc_cases.csv`（含 (nan,nan)/(-inf,inf) 行）；黑盒 `tests/blackbox/testcases/aclnnThnnFusedGruCellGrad_l{0,1,2}_*.csv`（同款 range 格式，l0 首行 input_data_ranges 含 (nan,nan)）。

## 预防

- 设计特殊值覆盖时先判物化通道：TTK 层只规划可物化形态；NaN/Inf 传播交 C++ ST 代码内构造。
- 解读覆盖度审计：valclass=nan/inf 签名 ≠ 非有限值已测；(-inf,inf) 用例如实解读为 ±dtype_max 幅值边界。

## 验证信号

本卡证据来自一次白盒用例验收会话中对 TTK 仓与统一入口源码的静态检视（grep/read 工具输出），未上板、未执行用例。

- 用例 CSV（kernel 模式）input_data_ranges 保留非有限字面量：主表含 "((nan, nan), (nan, nan))"（case00029 等）与 "((-inf, inf), (-inf, inf))"（case00026 等）行（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）
- TTK 采样器源码 ttk/utilities/data.py:51 起 RandomData，`_digitize_inf_nan` 语义为 isinf→dtype 取值域边界（±max）、isnan→0（来源平台：agnostic（宿主机 TTK 仓源码检视）；证据等级：tool_observed）
- TTK 输入生成链按 data_range 逐输入构造 RandomData（input_generation.py），并存在手工输入文件读取通路（来源平台：agnostic（宿主机 TTK 仓源码检视）；证据等级：tool_observed）
- 黑盒 aclnn 模式 CSV 同为 input_data_ranges 格式且含 (nan,nan) 行（l0 45 行 / l1 516 行 / l2 50 行），走同一 RandomData 采样链（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）
- 白盒统一入口 e2e 路径 `_finite_range` docstring 自述 "TTK's RandomData cannot sample inf/nan"（来源平台：agnostic（宿主机源码检视）；证据等级：tool_observed）

## 失效边界

- 本卡为源码级静态结论：实际物化（全 0 / ±dtype_max）未跑用例核对；`_digitize_inf_nan` 调用点未逐行定位（读到函数定义与 must-contain 边界值构造开头）。
- ±dtype_max 幅值边界测试本身有效（如 fp16 ±65504 与显式大值用例同族），但不得当作 Inf 传播覆盖主张。
- e2e 模式 `_finite_range` 映射为源码读取 + docstring 自文档，未实际生成 e2e CSV 验证。
- TTK 版本差异未查证（结论基于当前 /workspace/ops-test-kit 源码）。

## 坏实践

- 在 TTK CSV 里写 (nan,nan)/(-inf,inf) range 并在覆盖度主张中计为「NaN/Inf 已覆盖」。
- 把（假如物化的）全 NaN 输入→全 NaN 输出弱形式当作位置逐位对齐断言的等价物。

## 误诊

- 「CSV 里能写 nan 字面量 = TTK 支持 NaN 输入」——字面量只通过 CSV 解析，采样时被数字化。
- 「(nan,nan) 用例的通过/失败反映 NaN 行为」——实际输入是 0，与全零面用例同类。
- 「(-inf,inf) 用例测的是 Inf 传播」——实际是 ±dtype_max 幅值边界（fp16 即 ±65504）。


