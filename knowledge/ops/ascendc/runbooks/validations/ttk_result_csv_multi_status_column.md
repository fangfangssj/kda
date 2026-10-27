---
type: "Runbook"
title: "TTK result.csv 多状态列：聚合计数取错列"
description: "TTK 白盒 result.csv 有 perf_status/precision_status/memory_oob_status 三个状态列且 perf 在前；聚合计数取\"首个含 status 的列\"会数成 perf_status，产出全 PASS 假象；必须显式按 precision_status 列聚合并与开发日志交叉核对。"
tags: [task.debugging, topic.implementation, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T04:17:42Z'
updated_at: '2026-09-17T04:17:42Z'
platforms: ["950"]
---

# TTK result.csv 多状态列：聚合计数取错列

## 现象

为自定义 softmax 算子（BenchHarnessSoftmax，Ascend950PR，CANN 9.2.0）编写最终开发进展汇报时，需要汇总五个白盒分支的 TTK `result.csv` 精度通过率。先用一段快速 Python 聚合脚本按"**取第一个名字含 `status` 的列**"计数，得到：branch-0 74/74、branch-1 22/22、branch-2 **82/82 全 PASS**、branch-3 80/2、branch-4 24/24——而开发日志明明记录 branch-2 终态是 **80 PASS / 2 FAIL**（case00076、case00063 两条 fp16 1-ULP 用例）。按这份错数写汇报，就会把有 FAIL 的分支报成全绿。逐用例核对后确认：这两条用例 `precision_status=FAIL` 而 `perf_status=PASS`、`memory_oob_status=PASS`——脚本数的其实是 perf_status 列。改成显式按 `precision_status` 聚合后，branch-2 80/2、branch-3 79/3、branch-4 22/2，与开发日志逐分支完全一致。

## 适用场景

- 工程形态：TTK 白盒（kernel 模式）/黑盒（aclnn 模式）评测产物 `result.csv` 的离线聚合、门禁统计、进展汇报。
- API 路径：TTK result.csv 状态列族（白盒：`perf_status`、`precision_status`、`memory_oob_status`；黑盒：仅 `precision_status`）；Python `csv.DictReader` + `collections.Counter`。
- 验证过的平台与 CANN 版本：Ascend950PR_9579，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——一切对 TTK result.csv 做计数/门禁的脚本与人工解读。
- 路线优先级：recommended_fastpath（列名显式化零成本，汇报门禁必做）。
- 明确不适用 / 未验证场景：黑盒 result.csv（只有 `precision_status` 一个状态列，取首列碰巧正确——正因如此同一脚本在黑盒上不出错，缺陷被掩盖）；TTK 版本更换后的列族变化（须重新 `head -1` 核对表头）。

## 判别方法

白盒 result.csv 的列序中 `perf_status` 在 `precision_status` 之前（状态列族共三列：perf/precision/memory_oob）。触发本坑的原始聚合脚本（verbatim，问题行是 `c=Counter(r[st[0]] ...)` —— `st[0]` 取到的是 `perf_status`）：

```bash
for b in 0 1 2 3 4; do
  f=whitebox/branch-$b/ttk/result.csv
  if [ -f "$f" ]; then
    python3 -c "
import csv
from collections import Counter
rows=list(csv.DictReader(open('$f')))
total=len(rows)
keys=rows[0].keys() if rows else []
st=[k for k in keys if 'status' in k.lower()]
c=Counter(r[st[0]] for r in rows) if st else {}
print(f'branch-$b: total={total} {dict(c)}')
"
  fi
done
```

**探针 1（逐用例打印全部状态列，定位矛盾）**：

```python
import csv
rows=list(csv.DictReader(open('whitebox/branch-2/ttk/result.csv')))
for r in rows:
    cid=r.get('case_id') or r.get('id') or r.get('name') or ''
    if 'case00076' in str(r) or 'case00063' in str(r):
        st=[k for k in r if 'status' in k.lower()]
        print(cid, {k:r[k] for k in st})
```

实际输出：`{'perf_status': 'PASS', 'precision_status': 'FAIL', 'memory_oob_status': 'PASS'}`。

**探针 2（显式列名聚合，修正口径）**——列不存在时报出全部列名防静默错列：

```bash
cd <package>/tests
for b in 0 1 2 3 4; do
  python3 -c "
import csv
from collections import Counter
rows=list(csv.DictReader(open('whitebox/branch-$b/ttk/result.csv')))
total=len(rows)
# use precision_status explicitly; if absent, list cols
col='precision_status'
if col not in rows[0]:
    print('branch-$b: NO precision_status col, cols=',list(rows[0].keys())); raise SystemExit
c=Counter(r[col] for r in rows)
print(f'branch-$b: total={total} pass={c.get(\"PASS\",0)} fail={c.get(\"FAIL\",0)} skip={c.get(\"SKIP\",0)} other={ {k:v for k,v in c.items() if k not in (\"PASS\",\"FAIL\",\"SKIP\")} }')
"
done
```

判读规则：**精度门禁只认 `precision_status`**；`perf_status`（性能）与 `memory_oob_status`（越界）是独立维度，三者可同时出现 PASS/FAIL 分裂（如性能达标但精度失败的 1-ULP 用例）。聚合结果必须与开发日志的记录终态交叉核对，不一致时先怀疑自己数错列。

## 设计决策

- 显式列名 + 列缺失即报错（列出全部列名后 `SystemExit`），而不是静默回退到别的列：表头变化时宁可失败也不给错数。
- 保留逐用例探针作为第二步：聚合数与记录矛盾时，先看矛盾用例的完整状态列，再下"哪份是对的"的结论——本例正是靠逐用例打印发现 `precision_status=FAIL` 与"全 PASS"计数并存，才定位到取错列。
- 汇报口径以开发日志/任务状态为事实源、result.csv 为佐证：磁盘上的 result.csv 可能被后续复跑覆盖（mtime 晚于日志条目），单看文件当前态并不可靠。

## 系统信息

- 环境：Ascend950PR_9579，CANN 9.2.0；TTK v3.0.0 系（白盒 kernel 模式 / 黑盒 aclnn 模式产物）。
- 相关文件：`operators/<package>/tests/whitebox/branch-{0..4}/ttk/result.csv`（三状态列）；`tests/blackbox/ttk/result.csv`、`result_l2.csv`（仅 precision_status）；开发日志（记录各分支终态，作交叉核对基准）。

## 预防

- 写 TTK result.csv 聚合/门禁脚本时，状态列一律写死 `precision_status`（或先 `head -1` 打印表头人工确认），禁止"找第一个含 status 的列"。
- 任何聚合统计上汇报前，与开发日志记录的通过数逐分支对账。

## 验证信号

本卡来源说明：一次开发进展汇报的数据对账中，同一批白盒 result.csv 先后被"取首状态列"与"显式 precision_status"两种口径聚合，并与开发日志终态核对。逐条实测信号：

- 错误口径（来源平台：950；证据等级：tool_observed）：取首状态列计数得 branch-0 74/74、branch-1 22/22、branch-2 82/82（全 PASS 假象）、branch-3 80/2、branch-4 24/24。
- 逐用例观测（来源平台：950；证据等级：tool_observed）：branch-2 case00076/case00063 均为 `perf_status=PASS, precision_status=FAIL, memory_oob_status=PASS`；branch-3 三条失败用例中两条 perf_status=FAIL、一条 perf_status=PASS，但三条 precision_status 全 FAIL（解释了为何 branch-3 首列计数 80/2 而真实精度是 79/3）。
- 修正口径（来源平台：950；证据等级：tool_observed）：显式 precision_status 聚合得 branch-2 80/2、branch-3 79/3、branch-4 22/2（branch-0 74/74、branch-1 22/22 不变），与开发日志各分支终态逐项一致。
- 黑盒对照（来源平台：950；证据等级：tool_observed）：黑盒 result.csv 状态列族仅 `precision_status` 一个（535 全 PASS）、result_l2.csv 6 条（3 FAIL + 3 PARAM_COUNT_MISMATCH）——同一"取首列"脚本在黑盒上结果正确，印证缺陷只在多状态列的白盒 CSV 上暴露。

## 失效边界

- 列族与列序是 TTK 当前版本（CANN 9.2.0 配套）的白盒产物实测；TTK 升级增删状态列后须重新核对表头，"precision_status 优先于 perf_status"的列序不可假设恒定。
- 其他平台/TTK 版本组合的列族未验证，推断可迁移（用 `head -1` 重新核对表头即可确认）。
- 本卡只解决"聚合计数取错列"；磁盘 result.csv 被后续复跑覆盖导致与日志时间线不一致的问题（本例同时出现，靠 mtime 与日志对账处理）属通用文件取证，不在本卡范围。
- 未验证：TTK 其它模式（如 perf 专项产物）是否引入更多状态列。

**未决项**：

- 无（方法卡；本例错数在写入汇报前已当场纠正，未外流）。
## 坏实践

- 聚合脚本用 `[k for k in keys if 'status' in k.lower()][0]` 之类"动态找状态列"：白盒三状态列下列序决定数的是谁，产出静默错数。
- 不与开发日志对账就把聚合计数写进汇报/门禁结论：错数全 PASS 假象会直接漏报精度失败。

## 误诊

- 发现"branch-2 全 PASS 与日志 80/2 矛盾"时的第一反应曾是"result.csv 被更晚的复跑覆盖成全绿了"（mtime 确实晚于日志条目）——逐用例打印状态列后才发现计数口径就错了；两件事（复跑覆盖、取错列）都可能造成类似矛盾，先验状态列再验文件时间线。
