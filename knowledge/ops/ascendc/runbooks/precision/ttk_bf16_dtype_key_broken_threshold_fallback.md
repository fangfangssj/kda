---
type: "Runbook"
title: TTK bf16阈值键破碎回退fp32档2^-13
description: TTK numpy_bfloat16() 返回 ml_dtypes 类型对象而非 numpy dtype，_dtype_str 产生 "bfloat16'>" 破碎键查不中 THRESHOLDS["bfloat16"]=2^-7，静默回退 DEFAULT_THRESHOLD=2^-13（fp32 档，过严 64 倍）致 bf16 用例批量假失败；判别=看 result.csv bf16 行 threshold 是否 0.0001220703125，修复=golden 插件 __spec__ 双键覆盖。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# TTK bf16阈值键破碎回退fp32档2^-13

## 现象

在 Ascend950PR（CANN 9.2.0）上对已通过白盒验证的自研 softmax 基准算子（BenchHarnessSoftmax，aclnn 形态，fp16/fp32/bf16 三 dtype）跑 TTK aclnn 模式黑盒全量精度测试：535 条正向用例里 **60 条 precision FAIL——59 条 bf16 + 1 条 fp16**。失败高度集中：bf16 dim=0 `shape(2,14,1,1,1,1,6)` range(1,10) ×29 条、bf16 dim=1 `shape(18847,7,3,1,5)` range(-10,-1) ×30 条；而同一 kernel 的白盒 TTK（kernel 模式）与 C++ aclnn ST（独立双精度累加 oracle，181 例）全部通过——kernel 本身没有任何改动。负向门禁正常（6/6 非法输入被拒绝）。

拆 result.csv 的 `precision_metrics` 发现决定性信号：**所有 bf16 用例（含 PASS 的）`threshold: 0.0001220703125`（=2^-13，fp32/默认阈值），而不是 bf16 应有的 2^-7=0.0078125**；fp16 用例 threshold=0.0009765625（2^-10）正确。bf16 FAIL 例的实际误差 mere=6.07e-4~1.25e-3、mare=7.4e-3~7.8e-3——若按 2^-7（mere 阈）与 10×2^-7=0.078125（mare 阈）判定全部应 PASS。bf16 PASS 例多为 `mere: None`（空 tensor / 全 NaN 匹配等平凡通过路径，未真正算指标）。即 59 条 bf16 失败全部是阈值误配造成的假失败，非 kernel 精度问题。

## 适用场景

- 工程形态：TTK aclnn/kernel 模式黑盒/白盒精度评测的失败分诊；测试 CSV 的 `precision_tolerances` 字段为空（走默认阈值解析）的 bf16 用例。
- API 路径：`ttk/core_modules/comparison/resolve.py` 的 `THRESHOLDS`/`DEFAULT_THRESHOLD`/`_dtype_str`/`resolve_tolerance`；`ttk/utilities/dtypes.py` 的 `numpy_bfloat16()`/`resolve_custom_numpy_dtypes`；golden_plugin 的 `__spec__` tolerance 覆盖机制。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579），CANN 9.2.0，ops-test-kit Rev e559b7f（依赖该版本的 `_dtype_str` 实现）。
- 适用粒度：platform_wide——所有经 TTK stat_rel_err 默认阈值判定 bf16 输出的算子评测（numpy 无原生 bf16、靠 ml_dtypes 模拟时高发）。
- 路线优先级：recommended_fastpath（见到 bf16 大面积失败先查 threshold 字段，一轮定位；不要先入为主调 kernel）。
- 明确不适用 / 未验证场景：CSV 显式填了 `precision_tolerances` 的用例（不走默认解析）；fp32/fp16-only 算子（fp16 的 `_dtype_str` 键命中正常，本例 fp16 阈值 2^-10 正确生效）；非 TTK 评测（自建 C++ oracle 自管容差）；TTK 修复 `_dtype_str`/`numpy_bfloat16` 之后的行为（坏键覆盖变冗余但无害，见失效边界）。

## 判别方法

**第 1 步（trigger）**：bf16 用例成批失败 + 失败聚集在固定 pattern + 同 kernel 白盒/C++ ST 已绿 ⇒ 查 `result.csv` 每条 bf16 用例 `precision_metrics` 的 `threshold` 字段。等于 `0.0001220703125`（2^-13）即中招；等于 `0.0078125`（2^-7）则另查。

**第 2 步（机制探针，可直接复跑）**：

```python
cd $TTK && python3 -c "
from ttk.utilities.dtypes import resolve_custom_numpy_dtypes
from ttk.core_modules.comparison.resolve import _dtype_str, resolve_tolerance, THRESHOLDS, DEFAULT_THRESHOLD

flat_output_dtypes = ('bfloat16',)
output_dtypes = resolve_custom_numpy_dtypes(flat_output_dtypes)
print('output_dtypes:', output_dtypes)          # (<class 'ml_dtypes.bfloat16'>,)  ← type 对象而非 numpy dtype
print('type:', [type(d) for d in output_dtypes])  # [<class 'type'>]
for d in output_dtypes:
    s = _dtype_str(d)
    print(f'_dtype_str({d!r}) = {s!r}')          # "bfloat16'>"  ← 破碎键
    print(f'  THRESHOLDS.get({s!r}, DEFAULT) = {THRESHOLDS.get(s, DEFAULT_THRESHOLD)}')  # 0.0001220703125
"
```

**第 3 步（修复有效性探针）**：tolerance 覆盖用**破碎键**同样生效（`resolve_tolerance` 对 spec 里的每个键独立 resolve）：

```python
tolerance = {"bfloat16'>": {'standard': 'stat_rel_err', 'threshold': 2**-7}}
standards = resolve_tolerance(tolerance, None, None, output_dtypes, None)
# params={'threshold': 0.0078125, ...}
```

## 根因

**单句根因：TTK `numpy_bfloat16()` 返回 `ml_dtypes.bfloat16` 类型对象（type，而非 `np.dtype` 对象），`_dtype_str` 对其做 `str(type).split(".")[-1]` 得到破碎键 `"bfloat16'>"`（来自 `"<class 'ml_dtypes.bfloat16'>"` 的尾段），查不中 `THRESHOLDS["bfloat16"] = 2**-7`，回退 `DEFAULT_THRESHOLD = 2**-13`（fp32 阈值），bf16 用例被按 fp32 精度要求误判。** THRESHOLDS 表本身的 bf16 取值是对的，仅查表键破碎。

机制链（为什么必然批量误报）：

- bf16 显式尾数 7 位 → 数值量化的相对误差量级为 `2^-7`（0.0078125），设计文档据此规定 bf16 阈值 `2^-7`；
- 错误套用 fp32 档 `2^-13`（0.0001220703125）后，阈值收紧 `2^-7 / 2^-13 = 64` 倍；
- 凡输出相对误差落在 `(2^-13, 2^-7)` 区间的 bf16 用例（输入 range 与 shape 使量化误差落进该窗口的两个 pattern 共 59 条）即被误判 FAIL——不是逐元素算错，而是判尺错了 64 倍。

## 解决配方

**before**（blackbox `golden_plugin.py` 原状态，无 tolerance spec，阈值全部走 TTK 默认解析）：

```python
__golden__ = {"aclnn": {"aclnnBenchHarnessSoftmax": "_wrap"}}
```

**after**（增加 `__spec__` + TestSpec 类，同时覆盖正确键与破碎键；fp16/fp32 阈值显式写成与默认一致，防止解析行为漂移）：

```python
__golden__ = {"aclnn": {"aclnnBenchHarnessSoftmax": "_wrap"}}

# ── Tolerance spec per DESIGN.md §4.3 ──
# TTK's numpy_bfloat16() returns the ml_dtypes.bfloat16 *type* (not a numpy
# dtype), so _dtype_str produces "bfloat16'>" instead of "bfloat16".  This
# misses the THRESHOLDS table and falls back to DEFAULT_THRESHOLD=2^-13 (fp32),
# applying the wrong (too-strict) threshold to bf16 cases.  The tolerance below
# overrides the threshold for both the correct key "bfloat16" and the broken
# key "bfloat16'>" to the DESIGN §4.3 value of 2^-7.
__spec__ = {"aclnnBenchHarnessSoftmax": "BenchHarnessSoftmaxTestSpec"}

class BenchHarnessSoftmaxTestSpec:
    """Precision thresholds per docs/BenchHarnessSoftmax/DESIGN.md §4.3."""
    tolerance = {
        "float16": {"standard": "stat_rel_err", "threshold": 2**-10},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 2**-7},
        "float32": {"standard": "stat_rel_err", "threshold": 2**-13},
        # Workaround for TTK dtype resolution bug (see comment above)
        "bfloat16'>": {"standard": "stat_rel_err", "threshold": 2**-7},
    }
```

要点：这是**修正误配**而不是放松——bf16 阈值从错误的 2^-13 恢复为 THRESHOLDS 表自己声明的 2^-7；fp16/fp32 保持原判据不变。若修改 TTK 源码可行，正解是 `numpy_bfloat16()` 返回 `np.dtype(ml_dtypes.bfloat16)` 或 `_dtype_str` 兼容 type 对象；从算子包侧只能走 spec 覆盖。

## 设计决策

- **双键覆盖而非只补坏键**：同时写 `"bfloat16"` 与 `"bfloat16'>"`——TTK 若修复 `_dtype_str`，正确键已就位；未修复时坏键兜底。冗余键无害。
- **双键都写而非只写正确键**：默认解析路径 `resolve_tolerance` 收到的是破碎键（实测探针），只写正确键在本 bug 下不生效，必须两个键都写。
- **不改 TTK 本体、不放宽容差**：`_dtype_str` 是 TTK 内部实现，算子包侧不可修；统一放大阈值到超过 2^-7 会违反设计的分 dtype 阈值契约（fp32 仍须 2^-13）。TestSpec 覆盖是设计文档 §4.3 明文规定的正确值，验收时专门核对过"非人为放宽"（阈值与 DESIGN §4.3 逐项一致）。
- **修复落在包装层而非 kernel**：白盒 TTK 与 C++ aclnn ST（双精度累加 oracle）全过 + kernel 无改动，两条独立证据都指向判尺（容差）而不是被测对象。

## 预防

- bf16 用例首次接入 TTK 评测后，先抽一行 result.csv 核对 threshold 列 == 设计规定的 bf16 档值，再跑全量。
- 黑盒 bf16 批量失败而白盒/C++ ST 全过时，先查容差套用与 golden 口径，再动 kernel。

## 验证信号

本卡来源说明：一次黑盒 bf16 批量精度失败的修复及其独立验收（黑盒全量复跑 + 验收器 parse_result.py 独立复核双门禁 + 对照设计文档容差契约 + git 溯源确认 golden.py 与 kernel 未被修改 + 构建回归）。逐条实测信号（Ascend950PR，CANN 9.2.0）：

- 失败信号：黑盒正向 535 条中 60 条 FAIL（59 bf16 + 1 fp16，集中两个 bf16 pattern：dim=0 shape(2,14,1,1,1,1,6) range(1,10)×29、dim=1 shape(18847,7,3,1,5) range(-10,-1)×30）（来源平台：950；证据等级：tool_observed）。
- 误配证据：全部 bf16 用例 threshold=0.0001220703125；FAIL 例 mere 6.07e-4~1.25e-3、mare 7.4e-3~7.8e-3（按 2^-7/10×2^-7 判应全 PASS）；bf16 PASS 例 mere=None（平凡通过路径）（来源平台：950；证据等级：tool_observed）。
- 机制探针：`resolve_custom_numpy_dtypes(('bfloat16',))` → `(<class 'ml_dtypes.bfloat16'>,)`；`_dtype_str(...)` → `"bfloat16'>"`；`THRESHOLDS.get("bfloat16'>", DEFAULT)` → `0.0001220703125`；破碎键 tolerance 覆盖 → threshold=0.0078125（来源平台：950；证据等级：tool_observed）。
- 修复后黑盒复跑：正向 total=535 PASS=535 FAIL=0（GATE: PASS）；负向 total=6 被拒绝 6（GATE: PASS）（来源平台：950；证据等级：tool_observed）。
- 独立复核：验收器 parse_result.py 复核 FORWARD total=535 pass=535 fail=0 skip=0 EXIT=0；REVERSE NEGATIVE-GATE: PASS EXIT=0（来源平台：950；证据等级：tool_observed）。
- 阈值契约核对：修复后 result.csv 中 bf16 行 threshold=0.0078125（=2^-7）、fp16 行 0.0009765625（=2^-10）、fp32 行 0.0001220703125（=2^-13），与 DESIGN §4.3 逐项一致——是纠正错误套用，非放宽容差（来源平台：950；证据等级：tool_observed）。
- 无回归：git 溯源确认 golden.py（外部 golden）与 op_kernel/op_host 均未被本次修复触碰；修复后 `./build.sh` 退出码 0；C++ aclnn ST 181/181 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 破碎键字面量 `"bfloat16'>"` 依赖 ops-test-kit Rev e559b7f 的 `_dtype_str = str(x).split(".")[-1]` 实现；TTK 改动该函数（或 `numpy_bfloat16` 返回真正的 dtype）后，破碎键覆盖变多余但无害，正确键 `"bfloat16"` 覆盖继续生效。
- 本修复只在 golden_plugin 所在目录被 `--plugin` 指定时生效（TTK 的 spec 搜索按 plugin_path）；换执行入口须同步迁移 spec。
- spec 的 `standard` 字段只接受注册值（`stat_rel_err` 等）；试图设为未注册名会令全部用例输入生成失败（本开发会话实测 535/535 全灭后回退）。
- 阈值修正的前提是判据本身（设计文档 §4.3 的 2^-7）正确；若算子 spec 另有约定，应覆盖为 spec 值而不是照抄本卡数值。
- fp8 等其他非标准 dtype 键路径仅 bf16 实证命中（未验证）。

**未决项**：

- TTK 侧根治（`numpy_bfloat16` 返回 dtype 或 `_dtype_str` 兼容 type 对象）未提报/未验证；spec 双键覆盖是临时 workaround。
- 修复前 59 条 bf16 失败用例的逐用例 mare/mere 数值未留存（仅有 pattern 汇总），无法进一步刻画误报幅值分布。
## 坏实践

- 见 bf16 大面积失败直接归因 kernel 精度缺陷并改 kernel 累加/舍入——本例 kernel 全程正确（fp32 累加符合 spec），改 kernel 才会引入真缺陷。
- 用"统一放大阈值"或放宽容差让用例过——掩盖错误套用的同时破坏分 dtype 阈值契约，属测试侧放松。
- 直接修改 TTK 安装目录源码修 `_dtype_str`——越界且不可移植；spec 双键覆盖是算子包内的正道。
- bf16 PASS 例被当作"bf16 路径基本正常"的证据——其 `mere: None` 说明是平凡通过，不能佐证阈值正确。

## 误诊

- 初始直觉"fp16/fp16-quantization 类 golden 误差"或"kernel bf16 路径缺陷"：白盒 TTK、C++ aclnn ST（双精度累加 oracle）与 kernel 无改动三条证据共同排除了 kernel 侧；黑盒独有失败 + 失败集中在特定输入 range 的 bf16 pattern，指向评测层的 dtype 相关判尺。
- pattern 聚集初看像"某 shape 路径的 kernel bug"——pattern 聚集其实是「同一 dtype 同一阈值」的伪装；dtype 分布（59 bf16 + 1 fp16）才是第一线索。
