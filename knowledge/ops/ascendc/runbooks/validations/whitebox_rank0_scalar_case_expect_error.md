---
type: "Runbook"
title: 白盒 rank0-scalar 用例须补 expect_error
description: whitebox-designer 的 _special([], "rank0-scalar") 特例只造用例、不结合算子 rank_range 判合法性且不设 expect_error；固定 rank 算子（rank_range 排除 0，如 [4,4]）生成后 validate-shapes 报 5-6 条违规 exit=1。修复为后处理置 expect_error=True（_scenario 改 rank0-scalar-error）并用 cases-to-ttk 重导出 CSV；design-to-ttk 重跑会覆盖补丁，须重放。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:12:25Z'
updated_at: '2026-09-17T04:12:25Z'
platforms: ["950"]
---

# 白盒 rank0-scalar 用例须补 expect_error

## 现象

为 RoPE 类自定义算子（query/key rank_range=[4,4]，个别变体 cos/sin rank_range=[3,3]）逐分支生成白盒 TTK 用例后跑三项校验（validate-ttk / coverage-audit / validate-shapes），前两项通过而 validate-shapes 退出码 1，违规全部集中在同一个自动注入的 rank0-scalar 特殊用例（各输入 shape 均为 `[]`）。以其中一个分支为例（6 条违规，case00217）：

```
=== validate-shapes ===
{
  "checked": 231,
  "num_violations": 6,
  "violations": [
    {"case": "case00217", "tensor": "query", "kind": "rank_range", "detail": "rank 0 not in [4,4]"},
    {"case": "case00217", "tensor": "key",    "kind": "rank_range", "detail": "rank 0 not in [4,4]"},
    {"case": "case00217", "tensor": "cos",    "kind": "rank_range", "detail": "rank 0 not in [4,4]"},
    {"case": "case00217", "tensor": "sin",    "kind": "rank_range", "detail": "rank 0 not in [4,4]"},
    {"case": "case00217", "tensor": "cos", "kind": "shape_rule_eval",
     "detail": "rule not evaluable: cos.shape = [1, query.shape[2], 1, query.shape[3] // 2]"},
    {"case": "case00217", "tensor": "sin", "kind": "shape_rule_eval",
     "detail": "rule not evaluable: sin.shape = [1, query.shape[2], 1, query.shape[3] // 2]"}
  ],
  "ok": false
}
```

后两条 `shape_rule not evaluable` 是派生噪音（rank0 输入没有 shape[2]/shape[3] 可取），真正事实只有一条：rank 0 违反 rank_range——这个用例本该是 error 用例。查用例数据：`case00217 _scenario: rank0-scalar, expect_error: None`；而同算子 branch-0 的同名用例是 `_scenario: rank0-scalar-error, expect_error: True`。同一根因在该算子的多个生成分支上重复命中（branch-1 的 case00281 报 5 条违规、branch-4 的 case00281 报 6 条违规且含 cos/sin rank 0 not in [3,3]），每个分支生成都要重放同一修复。

## 适用场景

- 工程形态：CANN 自定义算子白盒用例生成（whitebox-designer 统一入口 `design-to-ttk`）+ 三项校验收尾，**接口为固定 rank**（`rank_range` 上下界相同或不含 0）的算子。
- API 路径：engine `_special([], "rank0-scalar")` 特例注入（`allow_scalar` 默认开）；`design_doc_cases.json` 用例的 `expect_error`/`_scenario` 字段；`validate-shapes` 对 expect_error 用例的跳过逻辑；`cases-to-ttk` 子命令（JSON→CSV 重导出）。
- 验证过的平台与 CANN 版本：生成/校验在 950 开发环境 host 侧完成，用例集随后在 Ascend950PR_9579 + CANN 9.2.0 上执行。
- 适用粒度：workflow 级——whitebox-designer 流程内、所有固定 rank 算子的分支用例生成。
- 路线优先级：recommended_fastpath（每分支生成后的固定后处理，脚本可复用）。
- 明确不适用 / 未验证场景：接口本就接受标量/变 rank（rank_range 含 0）的算子——rank0 用例是合法正例覆盖，不应标 expect_error。

## 根因

engine 的 rank0-scalar 特殊用例只做"存在性覆盖"，不携带合法性判定：

```python
# engine 生成路径（whitebox_designer.py）：
if (dims or {}).get("allow_scalar", True): _special([], "rank0-scalar")
# _special 只设 _scenario=tag，不设 expect_error
```

validate-shapes 对普通用例执行 `rank_range` 检查并跳过 `expect_error: True` 的用例；rank0-scalar 用例恰卡在中间——它既没有被标 error，又必然违反排他 rank_range（rank 0 ∉ [4,4]，且 rank0 上 `query.shape[2]`/`query.shape[-1]` 不存在使 shape_rule 不可求值，派生出额外违规）。生成器不结合算子 rank_range 判定该特例是否违规，是每个分支生成都会复现的固定后处理点。

## 解决配方

两步：后处理把 rank0-scalar 用例标为 expect_error，再从 JSON 重导出 CSV（先前 design-to-ttk 导出的 CSV 里该用例无 expect_error 标记）。

后处理脚本（对生成产物 design_doc_cases.json 原位修改，verbatim）：

```python
import json
path = 'operators/<OP>_package/tests/whitebox/branch-2/design_doc_cases.json'
with open(path) as f:
    d = json.load(f)
fixed = 0
for c in d['cases']:
    if c.get('_scenario') == 'rank0-scalar':
        c['expect_error'] = True
        c['_scenario'] = 'rank0-scalar-error'
        c['_note'] = 'rank0-scalar | expect_error: rank0 scalar violates rank_range [4,4] -> shape_mismatch'
        fixed += 1
print(f'fixed {fixed} rank0-scalar case(s)')
with open(path, 'w') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
    f.write('\n')
```

（`_note` 中的 rank_range 值是本算子取值，复用时按目标算子改写。）

重导出与复验命令序列（`$PY` 为 whitebox_designer.py 脚本入口，`$OUT` 为分支产物目录）：

```bash
python3 $PY cases-to-ttk $OUT/design_doc_cases.json $OUT/_tmp_opspec.json $OUT/ttk_design_doc_cases.csv
python3 $PY validate-ttk     $OUT/ttk_design_doc_cases.csv
python3 $PY coverage-audit   $OUT/design_doc_cases.json $OUT/_tmp_design_facts.json $OUT/_tmp_opspec.json
python3 $PY validate-shapes  $OUT/design_doc_cases.json $OUT/_tmp_design_facts.json $OUT/_tmp_opspec.json
```

关键约束与替代方案：

- 必须用 `cases-to-ttk` 重导出而不是重跑 `design-to-ttk`——后者从 design_facts 重新生成 JSON，会覆盖 expect_error 后处理并重新引入未标记的 rank0-scalar 用例；每次再生成后补丁须重新应用。
- 替代方案：`design_facts` 设 `allow_scalar=false` 直接关闭注入，对固定 rank 算子语义等价且少一步补丁；本轨迹选补丁方案是为保留 rank0 这条 error 用例覆盖（算子应在 host/编译期拒绝它），并与同算子前序分支的产物口径保持一致（前序分支的 rank0-scalar-error 用例即 expect_error=True）。

判别要点：validate-shapes 的违例若集中在单一 rank0-scalar 用例且伴随 `shape_rule not evaluable` 派生噪音，先查该用例 `expect_error` 是否缺失，而不是去修 shape_rule；违规分散在多个结构用例上才是真 shape 契约缺陷。

## 设计决策

- 选"后处理标 error"而不是"allow_scalar=false 关掉该用例"：rank0 输入是值得保留的 error 覆盖点（shape_mismatch 错误注入路径）；关掉会丢一条 error 用例。
- `_scenario` 同步改名 `rank0-scalar-error` 与前序分支命名对齐，便于跨分支对账与 provenance 统计。
- CSV 中该用例 `is_enabled=True` + `remark=expect_error`：TTK gate 按 precision_status≠PASS 记 FAIL——含 expect_error 用例时"pass==total"门禁永不可过，这是 gate 口径而非缺陷，验收按"全量执行+如实汇报"而非 pass==total。

## 验证信号

以下信号来自该算子多个分支的白盒生成会话（validate-shapes 违规 → 用例字段对照 → 后处理修复 → 复验）。

- 修复前：validate-shapes exit 1，branch-2 共 6 条违规全在 case00217（rank 0 not in [4,4] ×4 + cos/sin shape_rule not evaluable ×2），该用例 expect_error=None（来源平台：950 开发环境（host 侧生成，产物随后上板）；证据等级：tool_observed）。
- 同因复现：branch-1 的 case00281 报 5 条违规（4 tensor rank_range + 1 条 shape_rule 不可求值，checked 294）、branch-4 的 case00281 报 6 条违规（query/key rank 0 not in [4,4] + cos/sin rank 0 not in [3,3] + shape_rule 派生噪音），均为 expect_error=None（来源平台：950 开发环境（host 侧）；证据等级：tool_observed）。
- 修复后（branch-2）：`fixed 1 rank0-scalar case(s)`；validate-shapes ok=true、0 violations（230 checked）；validate-ttk PASS（238 行）+ coverage-audit ok=true 同轮全过（来源平台：950 开发环境；证据等级：tool_observed）。
- 修复后（branch-4）：`fixed 1 scalar cases -> expect_error=True`，cases-to-ttk 重导出 301 行；validate-ttk PASS + validate-shapes ok=True checked=293 num_violations=0 + coverage-audit ok=true missing={}（来源平台：950 开发环境（host 侧）；证据等级：tool_observed）。
- 下游上板佐证：case00217 在 Ascend950PR_9579 执行为 DYN_COMPILE_FAILURE（rank0 标量输入被算子在编译期正确拒绝，TTK 按 FAIL 记账属工具口径）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅当 rank0-scalar 用例确实非法（rank_range 排除 0）才标 expect_error；rank_range 含 0 的算子是合法正例，不应套用。
- 后处理改的是生成产物 JSON；后续分支重新 design-to-ttk 必须重放该后处理（或改用 allow_scalar=false 一劳永逸）。
- `_note` 文案与违规计数中的 rank_range 取值绑定本算子，复用时按目标算子改写。
- engine 若改为按 rank_range 自动判定 expect_error，本卡过期。

**未决项**：

- engine `_special` 不按 rank_range 自动判定 expect_error 的行为未改（工具侧）；连续分支生成建议把该修复与分支其他生成产物后处理合并为统一后处理脚本，并向工具维护方反馈（rank0 不在算子 rank_range 时应自动置 expect_error）。
## 坏实践

- 看到 `shape_rule not evaluable` 就去改 shape_rule 表达式——那是 rank0 输入的派生噪音，根因在 expect_error 缺失。
- 用 design-to-ttk 重跑来"刷新"CSV——重新生成会丢 expect_error 后处理，又回到违规状态。
- 为了让校验过而删掉标量用例——丢错误路径覆盖，且掩盖生成器缺口。
- 手改 CSV 的 remark 列而不改 JSON——下次 cases-to-ttk 重导出即丢失。
