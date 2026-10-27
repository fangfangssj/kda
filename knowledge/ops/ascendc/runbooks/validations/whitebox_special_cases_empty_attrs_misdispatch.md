---
type: "Runbook"
title: 白盒 special 用例 attrs 空致误路由
description: 白盒生成引擎的 extreme/empty/min-shape 特殊用例注入路径硬编码 params={}（不注入 scenario attrs），进入条件含非默认属性（如 layout=1/rotaryMode=interleaved）的分支其特殊用例会按默认属性静默派发到错误分支——用例仍在跑、跑批门禁不暴露；须生成后对全部非 error 用例断言/注入本分支 attrs 再 cases-to-ttk 重导 CSV。
tags: [task.testing, topic.tiling, topic.attention]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# 白盒 special 用例 attrs 空致误路由

## 现象

为 RoPE 类多分支自定义算子（BenchHarnessApplyRotaryPosEmb，分支判定 `tilingKey = f(layout, rotaryMode, cos.ndim)`——layout/rotaryMode 来自用例属性、cos.ndim 来自输入 shape）逐分支生成 TTK 白盒用例（统一入口 design-to-ttk，每分支一目录，238 例/分支）。为 Branch-2（half-BNSD-cos2d，`layout=1`）生成用例后做 attrs 审计：216 条结构用例都带 `params={'layout': 1, 'rotaryMode': 'half'}`，但 8 条非 error 特殊用例（empty-tensor×3、min-shape×1、extreme C1-C4×4）的 `params={}`，TTK CSV 的 `attributes` 列同样为 `{}`：

```
Cases with empty params (no layout/rotaryMode):
  case00218 scenario=empty-tensor params={}
  case00219 scenario=empty-tensor params={}
  case00220 scenario=empty-tensor params={}
  case00221 scenario=min-shape params={}
  case00222 scenario=extreme:C1-query-NaN-produces_nan params={}
  case00223 scenario=extreme:C2-all-zero-equals_0 params={}
  case00224 scenario=extreme:C3-query-pos-inf-matches_oracle params={}
  case00225 scenario=extreme:C4-large-value-10000-matches_oracle params={}
```

对照 branch-0（默认 mode：layout=0/half）同样有空 attrs——但它默认 `layout=0` 恰是 branch-0 自己，无害。branch-2 是 `layout=1`：attrs 空意味着这 8 条上板后会按默认 layout=0 调度进 **branch-0（BSND 寻址）kernel**，用 BNSD 形状/期望去测另一个分支的实现——结果要么假失败、要么静默测错对象，白盒覆盖出现 8 条黑洞且无任何报错。

跨分支核对进一步确认缺口形态：branch-1（interleaved-BSND）与 branch-4（half-BSND-cos3d）的 extreme/empty/min-shape 用例同样 `params={}`（结构用例 A5/A7/B2 属性齐全）；而 branch-2/branch-3 的同类 special 用例属性已被后处理注入补齐（`params={"layout": 1, "rotaryMode": "half"/"interleaved"}`）。即该缺口系统性影响一切"进入条件含非默认属性"的分支，注入过的分支是前一轮修复的产物，不是引擎行为。

## 适用场景

- 工程形态：TTK 白盒用例生成（统一入口 design-to-ttk → design_doc_cases.json + ttk CSV），多分支算子（attrs 决定分支调度）的白盒阶段。
- API 路径：生成引擎 `_special`/shape_boundary 注入路径（`build_fast_case` 硬编码 `attrs={}`）；结构用例属性来自 mode2attrs 反查表，shape_boundaries 属性来自 `sb.get("attrs")`；TTK CSV `attributes` 列 → aclnn attrs → host tiling 分支判定。
- 验证过的平台与 CANN 版本：生成器 host 侧；修复后的用例集在 Ascend950PR + CANN 9.2.0 全量执行，229 条成功编译用例 tiling 数据末字段 tilingKey=2（含极值/空张量用例），确认命中 branch-2。
- 适用粒度：workflow_only（whitebox 生成器驱动语义；凡"本分支 mode ≠ attrs 默认值"的分支生成均适用）。
- 路线优先级：recommended_fastpath（后处理一段脚本补齐，生成侧探针把关）。
- 明确不适用/未验证场景：分支判定只依赖输入 shape（无属性维度）的算子无此问题；默认 mode 分支（attrs 空恰好落到本分支，无需处理）；error 用例（expect_error 类保留原 attrs——D5 layout=2、D6 rotaryMode=full 本身就是要喂非法值）；黑盒用例生成路径未验证。

## 根因

生成引擎的 special 用例注入路径（extreme/empty/min-shape）构造用例时硬编码 `attrs={}`，属性只对结构扫描用例（mode2attrs）和 shape_boundaries（sb.attrs）填充。TTK CSV 的 `attributes` 由 `case["params"]` 直译（`{k: v for k, v in case["params"].items() if k in attr_names}`），params 空 → attributes 空 → aclnn 侧按算子注册的属性默认值调度。默认属性恰好等于分支进入条件的分支（如 half+layout0 系）**碰巧正确**；进入条件含非默认属性的分支（如 rotaryMode=interleaved、layout=1）其 special 用例被派发到**默认属性分支**——用例仍在跑、精度仍可能过（极值/空用例的语义对模式不敏感），但实际没测到目标分支，形成静默覆盖缺口。

## 判别方法

生成后对 `design_doc_cases.json` 逐例核对 special 场景的属性（判据：special 场景的 `params` 为 `{}` 而结构/边界用例非空，即命中本缺口；再对照该分支进入条件是否含非默认属性判断是否实际派发错误）。可执行探针（逐分支核对，可接入生成后检查；退出码非 0 即有违规）：

```python
#!/usr/bin/env python3
# coding=utf-8
# 用法: python3 check_special_attrs.py <whitebox_root> <branch> [<branch> ...]
# 核对各分支 design_doc_cases.json 的 special 用例（extreme/empty/min-shape）params 是否为空。
import json
import sys

SPECIAL_MARKS = ('extreme:', 'empty-tensor', 'min-shape')

def check(whitebox_root, branch):
    path = f'{whitebox_root}/{branch}/design_doc_cases.json'
    d = json.load(open(path))
    empty, populated = [], 0
    for c in d['cases']:
        sc = c.get('_scenario', '')
        if not any(m in sc for m in SPECIAL_MARKS):
            continue
        if c.get('params', {}):
            populated += 1
        else:
            empty.append((c.get('id', '?'), sc))
    return path, populated, empty

def main():
    whitebox_root, branches = sys.argv[1], sys.argv[2:]
    bad = 0
    for br in branches:
        path, populated, empty = check(whitebox_root, br)
        print(f'=== {br} ({path}) ===')
        print(f'  special 用例属性已注入: {populated}; params={{}}: {len(empty)}')
        for cid, sc in empty:
            print(f'    MISSING-ATTRS {cid} {sc}')
            bad += 1
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
```

## 解决配方

对**全部非 error 用例**断言/补齐本分支 attrs，再 `cases-to-ttk` 重导 CSV（verbatim 后处理脚本）：

```python
import json
path = 'operators/BenchHarnessApplyRotaryPosEmb_package/tests/whitebox/branch-2/design_doc_cases.json'
with open(path) as f:
    d = json.load(f)
# Fix: inject layout=1, rotaryMode=half for non-error cases with empty params
fixed = 0
for c in d['cases']:
    if c.get('expect_error'):
        continue  # error cases keep their (possibly invalid) attrs
    p = c.get('params', {})
    if not p or (p.get('layout') != 1 or p.get('rotaryMode') != 'half'):
        c['params'] = {'layout': 1, 'rotaryMode': 'half'}
        fixed += 1
print(f'fixed {fixed} non-error cases with missing/incorrect params')
with open(path, 'w') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
    f.write('\n')
```

重导与复核：

```bash
python3 $PY cases-to-ttk $OUT/design_doc_cases.json $OUT/_tmp_opspec.json $OUT/ttk_design_doc_cases.csv
# 复核：非 error 行 layout 分布应只剩本分支值
#   non-error layout values: {1: 230}
#   case00218 attrs={'layout': 1, 'rotaryMode': 'half'}（8 条特殊用例逐条核对）
```

复核口径 = 非 error 行 attributes 只允许本分支取值（如 230 例 layout 全 `{1}`）。

## 设计决策

- 选"后处理注入"而不是"改引擎 `_special` 传 attrs"：数据侧一段脚本、与既有后处理修复同轨同序，可一次脚本合并执行；改引擎须动特殊用例生成路径（所有 `_special`/shape_boundary 调用点），工具侧成本与回归面大。
- 注入范围取"全部非 error"而非"仅空 params"：把"空"与"错"一并兜住（极端情况下 shape_boundary 用例可能带错值），一次归一。
- 判定"哪些分支受影响"用分支判定表反推：默认属性（如 layout=0、rotaryMode=half）+ 各分支条件——默认属性恰好同分支的分支碰巧正确，进入条件含非默认属性的分支必然错路。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 工具：白盒统一生成入口（`whitebox_designer.py` 的 design-to-ttk / cases-to-ttk 子命令）。
- 数据流：`design_doc_cases.json`（cases[].params）→ `cases-to-ttk` → `ttk_design_doc_cases.csv`（attributes 列）→ TTK kernel 模式按 attrs 调 aclnn → host tiling 分支判定（tilingKey）。
- 目录布局：`operators/<Op>_package/tests/whitebox/branch-<N>/`（design_doc_cases.json + ttk_design_doc_cases.csv + whitebox_constraints.yaml）。
- 执行环境：修复后用例集 TTK kernel 模式执行于 Ascend950PR（SOC Ascend950PR_9579），CANN 9.2.0。

## 预防

- 为"进入条件含非默认属性"的分支生成白盒用例后，固定跑一遍 special attrs 探针；special 用例 params={} 即注入本分支属性再重导 CSV。
- 用例集评审时把"special 用例的属性齐整性"列入清单（与 expect_error 标记、rank/shape 校验同级）——该缺口**不会被跑批门禁暴露**（极值/空用例即使派发错分支也可能精度通过），只能靠生成侧探针发现。
- 注入的 attrs 取值须与本分支进入条件逐字一致；attrs 名写错会被 `attr_names` 过滤静默丢弃。

## 验证信号

以下信号来自 Branch-2 白盒生成会话（attrs 审计发现 8 条空 params → 注入修复 → 重导复核 → 下游上板）与为 interleaved 分支准备配置时的跨分支历史用例核对：

- 修复前：Branch-2 的 8 条非 error 特殊用例 params={}、CSV attributes={}；branch-0 对照同样为空但默认值恰为其 mode（无害）；branch-1/branch-4 的 extreme/empty/min-shape 用例同样 params={}，而 branch-2/3 同类用例属性已注入（前一轮修复产物）（来源平台：950 开发环境（host 侧生成，产物随后上 950 执行）；证据等级：tool_observed）。
- 修复后：`fixed 8 non-error cases`；非 error layout 分布 `{1: 230}`、rotaryMode `{'half': 230}`；case00218-00225 逐条 `attrs={'layout': 1, 'rotaryMode': 'half'}`；error 用例保留原 attrs（D5 layout=2、D6 rotaryMode=full 为故意的非法值）（来源平台：950 开发环境；证据等级：tool_observed）。
- validate-ttk PASS / coverage-audit ok=true / validate-shapes 0 violations（230 checked），CSV 238 行重导（来源平台：950 开发环境；证据等级：tool_observed）。
- 下游上板佐证：修复后用例集全量执行，成功编译用例 tiling 数据末字段 tilingKey=2 共 229 条（含极值/空张量用例），branch-2 BNSD 路径真实生效（来源平台：950；证据等级：tool_observed）。
- "缺省属性仍派发到正确分支"的旁证：half+layout0+cos3d 分支的白盒全量执行中，tilingKey=4 命中全部 228 个非 error 用例（含 params={} 的 special 用例）——默认属性与该分支进入条件一致，碰巧正确（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 注入只补 `params`；用例的其他生成属性（数据域、shape）不受影响。
- 默认属性值是算子注册决定的：默认属性不同于本卡假设（layout=0/rotaryMode=half）的算子，"哪些分支碰巧正确"的结论需按其 spec 重推（未验证，按算子 spec 判定）。
- 对"attrs 影响输出 shape"的算子（mode 改变输出形状），注入后还须复核输出 shape 一致性，本例 RoPE 输出恒同形不涉及。
- 极值/空用例即使派发错分支也可能精度通过（语义对模式不敏感），因此该缺口不会被跑批门禁暴露——只能靠生成侧探针发现。
- 引擎若改为 `_special` 自动继承 scenario attrs，本坑与配方同时过期。

**未决项**：

- 引擎 `_special`/shape_boundary 不继承 scenario attrs 的行为未改（工具侧）；连续多分支生成时，"注入本分支 attrs"应固化为标准后处理序。
- branch-1 既有用例集的 special 用例派发缺口（其极值/空/最小用例实际测的是 half 分支）未回补；是否重生成由该算子工程侧决策。
- 引擎侧是否可在 `_special` 注入时继承 default_mode 属性（治本），未验证。
## 坏实践

- 只核对结构用例的 attrs（它们由 mode 反查表自动带对）而漏掉特殊用例组——特殊用例是唯一被静默改道的集合。
- 把注入做成"仅当 params 为空才补"——带错值的用例会漏网；应按本分支期望值做全量断言/归一。
- 只看结构覆盖度审计（coverage_audit 的 size/dtype/align 轴）就认为用例集有效——special 用例的派发正确性不在其审计范围。

## 误诊

- 初判时曾以为"params={} 是打印方式问题（attrs 藏在 params.attrs 子键里）"；复查确认结构用例的 `params` 本身就是属性字典、special 用例确实为空 dict（两次探针交叉确认）。
- 曾以为 branch-0 也有同样问题；对照后确认 branch-0 的空 attrs 是默认 mode 下的无害巧合，坑只对非默认分支成立——这正是它容易漏检的原因。
