---
type: "Runbook"
title: TTK stat_rel_err 默认阈值 fp32 2^-13
description: golden 不写 threshold 时 TTK 按 dtype 默认表取阈值（fp32=2^-13≈1.22e-4）并以 mere<th AND mare<10×th 判定，spec numerical_tolerance 不直接生效。R2（CANN 9.2.0）：CSV precision_tolerances/absolute_precision 以 legacy rtol/ptol/atol 注入、不参与 stat_rel_err 判定链（源码确认）；mere 达标仅 mare 超阈多为近零 golden 被 FLOOR 分母放大，--compare close 可交叉验证。R5：大 R fp16 subnormal 输出在 stat_rel_err 硬编码 |g|+1e-7 分母下近零假失败（mare 10%~27%）；CSV absolute_precision 被生成阶段填默认 1e-08 而非 spec 逐 dtype atol，修正为 spec 值 + --compare close 后 130/130。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# TTK stat_rel_err 默认阈值 fp32 2^-13

## 现象

在用 TTK kernel 模式对 AscendC 算子（本例为 GRU Cell 反向融合算子的开发分支成品 kernel）跑白盒精度套件、判读失败用例时，发现 `result.csv` 的 `precision_metrics` 里实际生效的比对阈值是 `0.0001220703125`（即 2^-13），而算子 spec.yaml 的 `numerical_tolerance` 声明 fp32 rtol/atol=1e-5、用例 CSV 的 `precision_tolerances` 列也写 1e-5——两者都与实测阈值对不上。当时 golden（TestSpec 格式 `golden.py`）的容差声明只写了判据名、没写数值：

```python
    # 计算类算子（Elementwise/Reduction/FusedComposite 数值计算范式）浮点输出 → stat_rel_err
    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
```

同一套件里还能观测到：某路输出 mare=7.33e-4，超过了单阈值 th=1.22e-4，但 TTK 判 `pass=true`——因为实际判据是 `mere < th AND mare < 10×th` 的双指标结构，不是单阈值。若按 spec 的 1e-5 去推"mare 预算 = 10×1e-5 = 1e-4"，与 TTK 实际预算 10×2^-13≈1.22e-3 会差 12 倍，判读结论直接错。

## 适用场景

- 工程形态：TTK kernel / GEIR 模式白盒精度评测（`ttk kernel --plugin <golden.py>`），判读 `result.csv` / `result_oom.csv` 的精度列。
- API 路径：TTK 比对框架 `ttk/core_modules/comparison/resolve.py`（阈值解析）+ `ttk/core_modules/comparison/stat_rel_err.py`（判据实现）；golden TestSpec 的 `tolerance` 字段。
- 验证过的平台与 CANN 版本：Ascend950PR_9579 + CANN cann-9.0.0-beta.2（TTK 检出于 /tmp/ops-test-kit；与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（TTK 工具行为，跨算子适用）。
- 路线优先级：recommended_fastpath（判读规则，无替代路线）。
- 明确不适用 / 未验证场景：CLI `--compare` 显式传阈值、或 golden `tolerance` 显式声明 `threshold` 字段的场景（两者优先级高于默认表，本卡未验证其具体覆盖行为）；`close`/`cosine`/`binary` 等其他比对 standard。

## 判别方法

机制：TTK 的浮点比对阈值按三级优先级解析——CLI > golden TestSpec `tolerance` 声明 > `stat_rel_err` 默认表。golden 只声明 `{"standard": "stat_rel_err"}` 而不给 `threshold` 时，落到默认表，与 spec.yaml 的 `numerical_tolerance`、用例 CSV 的 `precision_tolerances` 列均无关（后两者不参与该判定链）。

默认表源码（`ttk/core_modules/comparison/resolve.py`，verbatim）：

```python
# 2.1 stat_rel_err 阈值表（resolve_tolerance 是唯一解析点——比对类不查表）
THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
}
DEFAULT_THRESHOLD = 2**-13
```

判据实现源码（`ttk/core_modules/comparison/stat_rel_err.py` 的 `compare_impl` 核心，verbatim）：

```python
    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        # threshold 是 resolve_tolerance 解析好的最终值（经 params → options 传入），不查表
        actual, golden = self.output, self.golden
        T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
        a = actual.astype(T)
        g = golden.astype(T)
        th = self.tol_options["threshold"]

        with np.errstate(invalid="ignore", divide="ignore"):
            rel_err = np.abs(a - g) / (np.abs(g) + FLOOR)
            # ……（NaN/Inf 一致性短路，略）……
            mere = float(np.mean(rel_err[mask]))
            mare = float(np.max(rel_err[mask]))
            passed = mere < th and mare < MARE_RATIO * th
```

其中 `MARE_RATIO = 10`、`FLOOR = 1e-7`（文件顶部算法常量）。即：
- `mere` = 全部元素相对误差的均值，`mare` = 最大值；
- 相对误差分母带 1e-7 的下限保护：`rel_err = |a − g| / (|g| + 1e-7)`；
- 通过条件是双指标：`mere < th` **且** `mare < 10 × th`。

判读操作：拿到失败用例后直接看 `result.csv` 该行的 `precision_metrics`（dict 形式含 `standard/mere/mare/threshold/pass/reason`），用其中的 `threshold` 字段作为实际生效阈值，不要用 spec/CSV 里的容差列推算。可用一小段脚本跨 dtype 抽查 threshold 字段验证默认表生效（fp32 用例 0.0001220703125、bf16 用例 0.0078125、fp16 用例 0.0009765625）。

## 设计决策

本卡为评测判读规则，无代码设计取舍，无证据。需要提醒的取舍点是：若希望 TTK 判定与 spec 数值容差严格一致，应在 golden 的 `tolerance` 里显式写 threshold（走第二优先级），而不是依赖默认表——但显式覆盖行为本开发会话未验证，属未验证场景。

## 验证信号

来源说明：以下信号来自一次 AscendC 算子开发分支的白盒精度套件执行（TTK kernel 模式、上板 Ascend950PR_9579、CANN 9.0.0-beta.2），判读失败用例时从 result.csv 与 TTK 源码双侧核对得出。

- 同一套件三个 dtype 用例的 `precision_metrics.threshold` 实测：fp32 用例 0.0001220703125 = 2^-13、bfloat16 用例 0.0078125 = 2^-7、float16 用例 0.0009765625 = 2^-10，与 `resolve.py` 的 THRESHOLDS 默认表逐一吻合（来源平台：950；证据等级：tool_observed）。
- 判据结构实测：某 fp32 用例一路输出 mare=7.33e-4 > th=1.22e-4 但 `pass=true`，与其并列的另一路输出 mare=1.24e-3 ≥ 10×th=1.22e-3 时 `pass=false` 且 reason 为 `threshold exceeded: mare(1.24e-03>=0.001220703125)`——证实预算是 10×th 而非 th（来源平台：950；证据等级：tool_observed）。
- golden `tolerance` 无 threshold 声明的源码事实（上卡内联代码块），与实测阈值来自默认表的推断互洽（来源平台：950；证据等级：tool_observed）。

- （R2 增补，Ascend950PR_9579 + CANN 9.2.0）全量结果 total=32 pass=19 fail=13 skip=0，13 例 FAIL 全 fp32、全 perf_status=PASS；Group A 11 例 mere 1e-8~1e-5 达标而 mare 1.87e-3~2.87 超阈，Group B case00006 mere=4.9888e-04、mare=7.8893 双超阈（来源平台：950；证据等级：tool_observed）。
- （R2 增补）失败报告实测阈值 mere ≤ 1.220703125e-04（2^-13）、mare ≤ 1.220703125e-03（10×2^-13）（来源平台：950；证据等级：tool_observed）。
- （R2 增补）stat_rel_err/resolve/is_close 公式与"CSV 容差不参与 stat_rel_err 判定"来自 ttk 源码文件阅读（rel_err=|a-g|/(|g|+FLOOR=1e-7)；th 按 dtype THRESHOLDS 表；isclose 才用 CSV rtol/atol）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 默认表数值属于 TTK 检出版本的行为，TTK 升级后可能变化；本卡锚定实测版本（CANN cann-9.0.0-beta.2 环境下的 /tmp/ops-test-kit 检出，无精确版本号事件）。
- `resolve.py` 源码注释提示存在 dtype 字符串解析的鲁棒性处理（`_dtype_str` 优先 `numpy.dtype(...).name`，异常时回退字符串切分；回退路径可能令 bfloat16 查表 miss 而回落 DEFAULT=2^-13 使阈值过严）——该坑在本开发会话中**未实际发生**（bf16 用例正确切到 2^-7），仅作边界提示，未验证。
- spec `numerical_tolerance` 与 TTK 默认阈值的口径差（fp32: 1e-5 vs 1.22e-4）是"哪边该改"的验收口径问题，本卡只记录实际生效行为，不裁决口径。

- （R2 增补）"仅 mare 超阈 ⇒ 非真实偏差"是高概率线索而非证明：极端元素处也可能存在真实小概率数值错误，下结论前应抽定位极端元素核对绝对误差。
- （R2 增补）FLOOR=1e-7 分母与阈值表是 TTK 实现细节；CANN 9.0.0-beta.2（本卡原始证据）与 9.2.0（增补证据）行为一致，升级仍以当次源码/失败报告为准。

## 坏实践

- 拿 spec.yaml 的 `numerical_tolerance`（如 fp32 1e-5）推算 TTK 的 mare 预算（10×1e-5=1e-4），再拿它判读 TTK 的 FAIL/PASS——与实际预算 1.22e-3 差 12 倍，会把边缘超差误判为大幅超差（或反向）。
- 只看 mare 是否超 th 就断言 FAIL，忽略 `mere < th AND mare < 10×th` 的双指标结构（mare 在 th 与 10×th 之间时 TTK 判 PASS）。
- 以为用例 CSV 的 `precision_tolerances` 列会传给比对器——实测不参与 stat_rel_err 判定链。

- （R2 增补）见 mare 爆炸就断定 kernel 精度错误并开始改 kernel 数值路径——先做失败模式分层（仅 mare 超阈组的均值精度已被 mere 证明正确）。
- （R2 增补）为"变绿"给 stat_rel_err 分母加 floor、改用宽松 metric 或放大 CSV 容差——评测侧放松，禁止；近零敏感只用于正确解读与定向排查。

## 误诊

- 把"threshold=1.22e-4 与 spec 1e-5 不一致"误诊为 TTK bug 或 golden 配置错误：实际是设计好的三级优先级 + dtype 默认表机制，golden 不写 threshold 就该落到默认表。
- 把 mare 超出 th（但小于 10×th）的输出当成失败逐个排查：TTK 此时判 pass，失败清单应以 `pass`/`reason` 字段为准。

## 未决事项

- 用例 CSV `precision_tolerances` 列在 TTK kernel 模式下是否完全不进判定链，只做了"实测未生效"的观测，未逐行验证其消费路径。
- golden `tolerance` 显式声明 `threshold` 时的覆盖行为未验证；如需让判定与 spec 对齐，可作后续实验。

## 增补：mare-only 失败分层与 CSV legacy 参数路径（R2，2026-09，CANN 9.2.0）

TTK kernel 模式全量跑 BenchHarnessApplyAdamW 32 例白盒（13 例 FAIL 全 fp32、全部 perf_status=PASS）时总结的解读框架，与本卡阈值机制配合使用：

1. **先分层失败模式**：解析每例 FAIL 的 mere/mare 与阈值比较，分成"仅 mare 超阈（mere 达标）"与"mere+mare 双超阈"两组——实测 Group A 11 例仅 mare 超阈（mere 1.39e-8~1e-5，mare 1.87e-3~2.87），Group B 2 例双超阈（case00006 mere=4.9888e-4 / mare=7.8893）。
2. **仅 mare 超阈组**：`rel_err = |a−g|/(|g|+FLOOR=1e-7)` 的分母在近零 golden（|g|≪FLOOR）时被 FLOOR 主导，微小绝对误差放大成大相对误差（如 |g|≈1e-10、|a−g|≈1e-9 时 rel_err≈1e-2）——度量对近零参考值的固有敏感，不等于 kernel 均值层面算错；优先怀疑宽动态范围/穿零输出用例，定位极端元素处的 |g| 与 |a−g|。
3. **双超阈组**：mere 也超阈说明存在实质数值偏差，优先排查 kernel 数值路径。
4. **换判据交叉验证**（不改实现只换度量）：`--compare close` 走 `np.isclose(rtol=CSV[0], atol=absolute_precision, equal_nan=True)` 且允许 ptol 比例元素失败——close 过而 stat_rel_err 不过，佐证近零敏感而非真实偏差；注意 ptol 很小时 close 也可能过不了。
5. **CSV 容差的消费路径（源码确认）**：`precision_tolerances`/`absolute_precision` 只作为 legacy 参数（rtol/ptol/atol）注入、不参与 stat_rel_err 判定——本卡未决事项"CSV 列是否完全不进判定链"由此闭合。


## R4 增量（自研框架 defaultTolerances 严于设计文档致成批误判 + 空置率探针）

R4 并入 BesselI1e 算子黑盒批量验证（数据驱动 ST 框架：CSV → manifest → C++ 框架，Ascend950PR / CANN 9.0.0-beta.2）的实例，把"默认表是空容差用例的实际验收标准"推进到自研框架形态并给出对齐修复：

**1. 现象（成批误判）**：521 条黑盒用例全量基线 `PASS: 274 | SKIP: 85 | FAIL: 162 | ERROR: 0`。排查两个关键事实：① 逐条解析 manifest，521 条用例 `precision_tolerances` **全部为空**（`With explicit tolerances: 0, Without: 521`），即全部走框架 `defaultTolerances(aclDataType, rtol&, atol&)` 默认表；② 默认表与设计验收标准脱节——默认表 fp16 `atol=1e-4`、bf16 `atol=1e-3`，而设计文档 §4.3.1（源于 spec.yaml `numerical_tolerance.per_dtype`）规定 fp16 `1e-3/1e-3`、bf16 `4e-3/4e-3`（metric max_relative：`pass = all(|y_i - g_i| ≤ atol + rtol × max(|y_i|, |g_i|))`）——默认表比设计阈值**严 4~10 倍**。kernel 实现（fp16/bf16 上溯 fp32 计算后下舍）误差量级在 1e-4~1e-3 档，落在设计阈值内但超出错误默认表，成批被误判 FAIL。

**2. 探针（容差来源分层）**：

```python
import json
from collections import Counter
data = json.load(open('<op>/tests/aclnn/aclnn_cpp_cases.json'))
cases = data['cases']
dtypes = Counter(c['input_dtypes'][0] for c in cases)
with_tol = sum(1 for c in cases if len(c.get('precision_tolerances', [])) >= 2)
print(f'Total cases: {len(cases)}')
print(f'Dtype distribution: {dict(dtypes)}')
print(f'With explicit tolerances: {with_tol}, Without: {len(cases) - with_tol}')
# 本例输出：Total cases: 521 / dtype 分布 fp16 182/bf16 176/fp32 162/float64 1
#           With explicit tolerances: 0, Without: 521
```

**3. 修复（默认表整表对齐设计文档，kernel 不动）**：

```cpp
// 修复前（框架默认表，atol 严于设计验收标准）：
static void defaultTolerances(aclDataType t, float& rtol, float& atol) {
    if (t == ACL_FLOAT)          { rtol = 1e-5f; atol = 1e-5f; }
    else if (t == ACL_FLOAT16)   { rtol = 1e-3f; atol = 1e-4f; }
    else if (t == ACL_BF16)      { rtol = 4e-3f; atol = 1e-3f; }
    else                         { rtol = 1e-5f; atol = 1e-5f; }
}

// 修复后（对齐 DESIGN §4.3.1）：
static void defaultTolerances(aclDataType t, float& rtol, float& atol) {
    if (t == ACL_FLOAT)          { rtol = 1e-5f; atol = 1e-5f; }
    else if (t == ACL_FLOAT16)   { rtol = 1e-3f; atol = 1e-3f; }
    else if (t == ACL_BF16)      { rtol = 4e-3f; atol = 4e-3f; }
    else                         { rtol = 1e-5f; atol = 1e-5f; }
}
```

**与"放水"的界线**：本修复的合法性在于**对齐设计验收标准这一既存事实源**（设计文档明确写在 §4.3.1 判定逻辑里），不是为让失败用例通过而临时调松；对齐后同一张表用于全量重跑与 3 次确定性验收，kernel 真有超阈值误差仍会 FAIL；改完后由验收方按设计文档逐项复核确认。为让失败用例通过而把 atol/rtol 调到"恰好够大"且无文档依据仍是放水（banned）。

**4. 分层归因纪律**：精度失败归因时先分层——显式容差 > 框架默认表 > 设计文档阈值，三层对不齐先修脚手架再查 kernel。黑盒用例接入时先统计 `precision_tolerances` 空置率：空置率高时默认表就是实际验收标准，它的正确性决定整批结论。

**R4 验证信号**：

- 对齐前基线：全量 521 条 `PASS: 274 | SKIP: 85 | FAIL: 162 | ERROR: 0`；容差探针 521 条全空、dtype 分布 fp16 182/bf16 176/fp32 162/float64 1（来源平台：950；证据等级：tool_observed）。
- 仅改默认表（kernel 的 inf 修复尚未做）：46 条 mini 集中常规值用例全部 PASS，仅剩 4 条 ±inf 用例 FAIL——容差误判与 inf 根因相互独立（来源平台：950；证据等级：tool_observed）。
- 默认表对齐 + inf 修复后：437 条全量（84 条超大 shape 另行过滤）`PASS: 437 | SKIP: 0 | FAIL: 0 | ERROR: 0`，3 次复跑 bit-wise 一致；验收方复核确认精度满足 DESIGN §4.3（来源平台：950；证据等级：tool_observed）。

**R4 边界**：本增量仅覆盖"默认表与设计阈值脱节"这一脚手架根因；设计文档本身阈值变更时默认表需再次同步（无自动机制）；CSV 自带显式 `precision_tolerances` 的用例不走默认表不受影响；误差超出设计阈值的真 kernel 缺陷不会被对齐后的默认表放行。默认表与 spec/design 之间尚无自动一致性校验（lint/生成脚本），漂移只能靠人工发现——值得补一个构建期检查（遗留）。

## R5 增量（大 R fp16 subnormal 输出的近零假失败与 CSV absolute_precision 生成默认值坑）

softmax 基准算子分块分支（TTK kernel 模式 130 条白盒，Ascend950PR + CANN 9.1.0）把 R2 的"仅 mare 超阈"形态推到 **fp16 subnormal 输出区间**，并闭合 CSV `absolute_precision` 的生成默认值坑：

**1. 现象与口径核对**：8 条失败全部 fp16——5 条 AR 用例 mere=1.7e-6~1.5e-4（阈 9.77e-04）远低于阈而 mare=1.02e-2~2.72e-1（阈 9.77e-03）超阈；同批用例在 C++ ST 用 `allclose(rtol=atol=9.77e-3)` 判定却全部 PASS——两个执行入口结论相反。把 C++ 比较器复刻成"golden 经存储 dtype round-trip 后算 mare/mere"后 8/8 复现失败，确认失败信号真实、差异在**比对口径**。

**2. 机制（subnormal 放大）**：TTK 默认 `stat_rel_err` 分母硬编码 `|g|+1e-7`（FLOOR，不消费 CSV 的 `absolute_precision`）；大 R fp16 softmax 输出量级 ~1/R——R=2M 时实测输出分布 0/subnormal/normal = 23976/2073176/0、均值 4.77e-7（全部落入 subnormal/0 区间）。golden≈5e-7、|a-g|≈1e-7 时 stat_rel_err 分母≈1.5e-7 → rel_err≈0.1~0.27（与实测 mare=0.272 吻合）；spec 口径分母 `|g|+atol`（fp16 atol=9.77e-3）→ rel_err≈1e-5——同一组数值两口径差约 5 个数量级。CPU 数值探针：`torch.nn.functional.softmax(x_fp16)` 与 `softmax(x_fp16.float()).half()` 逐位相同（torch 内部 fp32 上抛再回写），与朴素 fp32 参考 max diff 5.96e-8（fp16 的 1 个量化级）——kernel 与 golden 各自的 fp32 中间量只要差 1e-7 量级，回写 fp16 就可能落在不同 subnormal 档。

**3. CSV `absolute_precision` 生成默认值坑**：白盒用例 CSV 的 `absolute_precision`（`--compare close` 的 atol 来源）被生成阶段填成工具默认 `1e-08`（实测原值 `(1e-08,)`），与 spec `numerical_tolerance` 逐 dtype atol 不一致——修正为 spec 值（fp16 9.765625e-3 / fp32 1.220703125e-3 / bf16 7.8125e-2）+ TTK 执行加 `--compare close`（`np.isclose(a, g, rtol, atol)`，rtol/atol 经 `resolve_tolerance` 取自 CSV 即 spec 值，等价 spec 的 allclose 口径）。**修正纪律：修正值必须逐 dtype 等于 spec 原值，不得大于**；rtol/ptol 一律不动。

**4. 处置结果**：8 条原失败用例 `--compare close` 全部 PASS（isclose 100.0%），全量 130/130，独立验收复跑 130/130（REPRODUCIBLE）、GATE PASS；容差审计逐 dtype atol/rtol 与 spec 完全相等、VIOLATION_COUNT=0，git diff 确认仅 `absolute_precision` 从 1e-08 变更为 spec 值。注意 ARA 用例的 mere 2-3e-3 是独立的 kernel 真实缺陷（在线两趟重缩放），**不能**靠口径修正掩盖。

**5. 转写警告**：TTK 失败报告/转写文档里 mere/mare 的展开名可能写反（TTK 代码里 mere=mean、mare=max）——报告解读以比较器源码为准。

### R5 验证信号

- 失败画像：TTK 全量 total=130 pass=122 fail=8，8 条全 fp16（AR 5 条 mere=1.7e-6~1.5e-4 正常、mare=1.02e-2~2.72e-1 超阈；F1 [1,2097152] mare=2.719e-1）；同批 8 条 C++ ST allclose 全 PASS；C++ 复刻 mare/mere 口径后 8/8 复现失败（来源平台：950；证据等级：tool_observed）。
- 口径探针：R=2M fp16 输出分布 0/subnormal/normal = 23976/2073176/0、y mean=4.77e-7；torch fp16 softmax 与 fp32 上抛再 half 逐位一致；vs 朴素 fp32 参考 max diff 5.96e-8（来源平台：950；证据等级：tool_observed）。
- CSV 原值 `absolute_precision` 为 `(1e-08,)`（TTK 生成默认）≠ spec 逐 dtype atol（来源平台：950；证据等级：tool_observed）。
- 处置后：8 条 `--compare close` 全 PASS（isclose 100.0%）、全量 130/130、独立复跑 130/130 REPRODUCIBLE、GATE PASS；容差审计 VIOLATION_COUNT=0（来源平台：950；证据等级：tool_observed）。

### R5 边界与坏实践

- 本形态限定"输出全 subnormal/0 的大 R（≈1e5~2e6）fp16"场景；R 较小（输出落 normal 区）时 mare 超阈大概率是真实精度问题，不适用。
- `--compare close` 的 `equal_nan=True` 与 spec 正常用例 `equal_nan=False` 存在差异：NaN 输入用例依赖 golden 与 kernel 同时 NaN 才通过，特殊值场景未验证。
- stat_rel_err 在输出为 normal 区间时与 close 口径结论一致（本例 122 条原 PASS 用例两口径均过）。
- 坏实践增补：生成白盒/黑盒用例 CSV 时让 `absolute_precision` 落工具默认值 1e-08（验收清单应加"CSV atol/rtol == spec，无一大于"）；调试中计划的"把失败用例 rtol 放宽到 0.039~0.51 以通过 mere/mare"属测试侧放松（会掩盖 ARA 真缺陷），被明确放弃；把 `absolute_precision` 改到 spec 值时顺手把 rtol 也调大——任何大于 spec 的值都是违规。
