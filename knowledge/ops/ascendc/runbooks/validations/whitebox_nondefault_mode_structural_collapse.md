---
type: "Runbook"
title: 白盒非默认 mode 分支结构覆盖塌缩：mode_struct_k 恢复
description: "whitebox-designer design-to-ttk 为非默认 mode 分支（如 layout=1 BNSD）生成用例时，engine 对 enum_int 类 attr 恒并 set(range(5))（opspec values=[1] 不收敛枚举），build_model 只给排序首位的 default_mode 全结构笛卡尔，非默认 mode 仅得 pool[:K]（默认 K=16）——236 例塌缩成 55 例且无报错。修复：design_facts 显式设 mode_struct_k=池大小（12 valclass×3 align×3 size×1 rank=108）。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:11:06Z'
updated_at: '2026-09-17T04:11:06Z'
platforms: ["950"]
---

# 白盒非默认 mode 分支结构覆盖塌缩：mode_struct_k 恢复

## 现象

为 RoPE 算子 Branch-2（half-BNSD-cos2d，layout=1）生成 TTK kernel 模式白盒用例：按 branch-0/1 的先例构造 `design_facts.json`（dtype=[fp32,fp16,bf16]、rank=[4]、attrs 限定 `layout=[1], rotaryMode=["half"]`）与 `opspec.json`，经统一入口运行：

```
python3 $PY design-to-ttk $OUT/_tmp_design_facts.json $OUT/_tmp_opspec.json \
    $OUT/design_doc_cases.json $OUT/ttk_design_doc_cases.csv
```

输出只有一行汇总，**没有任何失败报错**：

```
[wb_scenario_gen] bench_harness_apply_rotary_pos_emb: dropped 15 unrealizable/invalid targets
wrote 55 generated cases -> ...
```

而 branch-0/1 同流程产出 236/238 例。用例集统计确认塌缩：`target_sigs: 48`（branch-0 为 324）、`cases: 55`、结构例仅 33（branch-0 为 216），`modes: {'layout=1-rotaryMode=half': 33}`——覆盖从 3 dtype×2 size×3 align×12 valclass 掉到每 dtype 只剩 11 条。

## 适用场景

- 工程形态：TTK 白盒用例生成（whitebox-designer `design-to-ttk` 子命令），AscendC 算子多分支开发流程中的白盒阶段。
- API 路径：`skills/ascendc-whitebox-design/references/whitebox_designer.py` 的 `design-to-ttk`；输入 `design_facts.json`（`scenario_dimensions`/`mode_struct_k`）与 `opspec.json`（`interface.attrs`）。
- 验证过的平台与 CANN 版本：生成器为 host 侧 Python（无 NPU 依赖）；产出的 238 例用例集随后在 Ascend950PR_9579 + CANN 9.2.0 上全量执行（tiling 数据 tilingKey=2 共 229 条，分支真实生效）。
- 适用粒度：workflow_only（whitebox-designer 驱动语义；对同一 engine 的后续分支生成直接适用——凡目标 mode 非"排序首个 mode"的分支都会踩，例如后续 interleaved-BNSD 分支 mode `layout=1-rotaryMode=interleaved`）。
- 路线优先级：recommended_fastpath（配置一行即恢复全结构覆盖）。
- 明确不适用 / 未验证场景：默认 mode 分支（如 branch-0 `layout=0-rotaryMode=half`、branch-1 `layout=0-rotaryMode=interleaved`——它们的 mode 恰为排序首位，天然吃满全笛卡尔，无需此配置）；engine 版本更新后枚举语义若变化需复核。

## 根因

engine 的两个机制叠加，把非默认 mode 的结构覆盖砍到 K=16/dtype：

1. **enum_int 枚举不收敛**：`_attr_kind` 把 int 型 attr（如 `layout`）归为 `enum_int`；`_candidates` 对 enum_int 恒做并集（engine 源码，覆盖 begin_mask/dilation 等"desc 数字 ∪ 小整数 0..4 ∪ default"场景）：

```python
if kind == "enum_int":
    # desc 数字 ∪ 小整数集(位掩码/小枚举/倍数常见 0..4)∪ default,覆盖 begin_mask/dilation/hc_mult 等
    vals = set(spec.get("int_choices") or []) | set(range(5))
    if isinstance(dflt, int): vals.add(dflt)
    return sorted(vals)
```

opspec 里 `"values": [1]` 只进 `int_choices`，挡不住 `set(range(5))`——layout 仍枚举 {0,1,2,3,4}，attr_modes 展开 5 个 mode（`layout=0..4 × rotaryMode=half`）。
2. **全笛卡尔只给 default_mode**：`build_model` 把完整结构池（每 dtype 的 `|valclasses|×|aligns|×|sizes|×|ranks|` 笛卡尔）只分配给排序首位的 default_mode（`layout=0-rotaryMode=half`，layout=0 恰是 branch-0 的 mode）；其余 mode 每 dtype 只取 `pool[:K]`，`K = dims.get("mode_struct_k", 16)`。

branch-2 的 mode（layout=1）非默认 → 每 dtype 只有 16 个结构点 → 3 dtype×16=48 target_sigs → 33 例（再丢 15 unrealizable）。branch-0/1 没踩坑纯属"其 mode 恰好排序第一"。

## 解决配方

在 `design_facts.json` 显式设 `mode_struct_k` = 单 dtype 结构池大小，让非默认 mode 也吃满整个 pool。池大小按维度乘积算：`|valclasses(12)| × |aligns(3)| × |sizes(3)| × |ranks(1)| = 108`。

修复前 → 修复后（design_facts.json 片段，verbatim）：

```json
// 修复前（无该键，K 默认 16）：
  "golden_path": null,
  "op_dir": "operators/BenchHarnessApplyRotaryPosEmb_package/tests/whitebox/branch-2",

// 修复后：
  "golden_path": null,
  "mode_struct_k": 108,
  "op_dir": "operators/BenchHarnessApplyRotaryPosEmb_package/tests/whitebox/branch-2",
```

重跑 `design-to-ttk` 后：L-size 目标（36/dtype）被 `max_shape_elems=60Mi` 丢弃（dropped 108），余 72/dtype×3=216 结构例 + 22 边界/极值/误差 = **238 例**，与 branch-1 一致。

判别式：生成后核对三处计数——`target_sigs`（应=全笛卡尔 324）、结构例数（应=216）、`modes` 分布（应只有本分支 mode）。计数不符即命中本坑，无任何报错可依赖。

## 设计决策

- 备选一：把 opspec 的 layout 改成 enum_str（`values: ["1"]`）可绕开 enum_int 并集——被否：TTK CSV 的 attributes 会变字符串 `'layout': '1'`，而 aclnn 原型要求 int64，下游类型错。
- 备选二：不收敛枚举、纯靠 `_filter_target_sigs` 过滤（opspec attrs 已把 scenario 过滤到 layout=1）——被否：过滤发生在结构分配之后，非默认 mode 已经只分到 K=16，过滤救不回覆盖。
- 选定：`mode_struct_k=108` 覆盖全 pool，L-size 由 max_shape_elems 自然丢弃——与 branch-0 的有效覆盖（216 结构）精确对齐。
- 当前方案路线档位：recommended_fastpath（驱动侧配置，不改 engine）。

## 系统信息

- 工具：`skills/ascendc-whitebox-design/references/whitebox_designer.py`（`design-to-ttk` 子命令；engine 内嵌 `wb_scenario_model` 的 `_candidates`/`build_model`）。
- 输入产物：`tests/whitebox/branch-2/_tmp_design_facts.json` + `_tmp_opspec.json`（临时，产出后清理）；输出 `design_doc_cases.json` + `ttk_design_doc_cases.csv`。
- 约束 sidecar：`whitebox_constraints.yaml`（dim_even 强制 D 偶）；`max_shape_elems=60Mi`。

## 完整代码

配方核心 = 上述 design_facts.json 一行配置；生成与复核命令（verbatim）：

```bash
PY=skills/ascendc-whitebox-design/references/whitebox_designer.py
OUT=operators/BenchHarnessApplyRotaryPosEmb_package/tests/whitebox/branch-2
python3 $PY design-to-ttk $OUT/_tmp_design_facts.json $OUT/_tmp_opspec.json \
  $OUT/design_doc_cases.json $OUT/ttk_design_doc_cases.csv
# 复核覆盖（target_sigs 应 324、cases 应 238、modes 应仅本分支）：
python3 -c "
import json
from collections import Counter
d = json.load(open('$OUT/design_doc_cases.json'))
print('target_sigs:', len(d['target_sigs'])); print('cases:', len(d['cases']))
modes = Counter(c.get('sig',['?',]*6)[5] for c in d['cases'] if isinstance(c.get('sig'), list) and len(c['sig'])==6)
print('modes:', dict(modes))
"
```

## 验证信号

以下信号来自 Branch-2 白盒用例生成会话（首跑 55 例 → 定位 engine 枚举语义 → 加 mode_struct_k=108 → 重跑复核）。

- 修复前：`wrote 55 generated cases`（dropped 15），target_sigs=48，结构例 33（来源平台：950 开发环境（host 侧生成，产物随后上 950 执行）；证据等级：tool_observed）。
- 修复后：`wrote 238 generated cases`（dropped 108 = 3 dtype×36 L-size），target_sigs=324，结构例 216（3 dtype×2 size×3 align×12 valclass），modes 仅 `{'layout=1-rotaryMode=half': 216}`，size 分布 {M:108, S:108}、align 三类各 72、dtype 三类各 72（来源平台：950 开发环境；证据等级：tool_observed）。
- 三项校验全过：validate-ttk PASS（238 行）、coverage-audit ok=true missing={}（size/dtype/align 全覆盖）、validate-shapes ok=true（0 violations，230 checked）（来源平台：950 开发环境；证据等级：tool_observed）。
- 下游上板佐证：该 238 例集在 Ascend950PR_9579 全量执行，229 条成功编译用例 tiling 数据末字段 tilingKey=2（branch-2 真实调度），证明修复后的用例集有效命中目标分支（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `mode_struct_k` 取值依赖本分支的维度组合（valclass/align/size/rank 数量）；换算式为乘积，维度变化须重算，硬编码 108 不可迁移。
- engine 若修改 enum_int 并集语义（如尊重 `values` 声明）则本坑消失、本卡过期。
- L-size 结构覆盖天然被 max_shape_elems 丢弃，"全 pool"只到 M 档；L 档须靠 shape_boundary 单点（如 B4 代表例）补。

**未决项**：

- engine 侧 `enum_int ∪ range(5)` 与"default_mode 独占全笛卡尔"的语义未改动（工具行为），本卡只给驱动侧配方；若后续分支持续受累，应考虑给 engine 提"尊重 values 声明"的改进。
## 坏实践

- 以为 opspec `attrs.values=[1]` 能把枚举收敛到单值——enum_int 恒并 range(5)，声明值只是并集的一支。
- 只看"wrote N cases"不核对 target_sigs/结构例数/modes 分布——本坑静默发生，验收侧"total==应有条数"是最后一道网。

## 误诊

无（首轮分析曾怀疑 opspec `values` 字段写法未被读取，读 engine 源码后确认 `_opspec_attr_spec` 确实读了 values 进 int_choices，问题在 `_candidates` 的恒并集）。
