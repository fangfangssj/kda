---
type: "Runbook"
title: "L2 负向意外 PASS：合法输入误归类，非洗绿"
description: "黑盒修复后负向门禁报 UNEXPECTED-PASS 时，先按 dim 值域/校验代码/同参正向覆盖三步判别真假阴性——dim=0∈[0,rank) 合法而 remark 标'维度下溢'属用例误归类，不是放松校验的洗绿；门禁把崩溃也计 FAIL，旧 NEGATIVE-GATE PASS 里可能混假绿。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# L2 负向意外 PASS：合法输入误归类，非洗绿

## 现象

在 Ascend950PR（CANN 9.2.0）上验收一个 scatter 类自定义算子的黑盒失败用例修复（blackbox-fix）：kernel 精度问题修复后，正向套件 399/399 全 PASS（`parse_result.py result.csv --expect-total 399` 输出 `total=399 pass=399 fail=0 skip=0, GATE: PASS`），但负向（L2 异常输入，预期算子全部拒绝）门禁报：

```
total=25 pass=1 fail=24 skip=0
  [UNEXPECTED-PASS 算子未拒绝非法输入] aclnnBenchHarnessScatter_L2_exc_dimval_019
NEGATIVE-GATE: FAIL(存在未被拒绝的非法用例)
```

即 25 条 L2 负向用例中有 1 条（`aclnnBenchHarnessScatter_L2_exc_dimval_019`）在修复后反而 PASS 了。该用例的 remark 写的是"维度下溢; expect ACL_ERROR_INVALID_DIMENSION"——按字面理解，这是"算子没有拒绝一个非法维度"。此时面临两种相反的定性：**修复者放松了校验（洗绿，验收必须 FAIL）**，还是**这条负向用例本身把合法输入误归类为异常（假阴性，正确实现的必然结果）**。

查看用例定义（L2 exception CSV，原文）：

```
CASE: aclnnBenchHarnessScatter_L2_exc_dimval_019
  tensor_view_shapes: ((5,11,8,9,8,7,7,),(2,5,4,4,4,3,3,),(2,5,4,4,4,3,3,),(5,11,8,9,8,7,7,),)
  tensor_dtypes: ('float32','int32','float32','float32',)
  tensor_formats: ('ND','ND','ND','ND',)
  attributes: {"dim": 0, "reduceOptional": "update"}
  input_data_ranges: ((-10,-2,),(0,4,),(2147483647,2147483647,),)
  remark: 维度下溢; expect ACL_ERROR_INVALID_DIMENSION
```

data 是 rank=7 的张量，dim=0，dtypes 四元组完全一致，index 值域 (0,4) < data.shape[0]=5。

## 适用场景

- 工程形态：aclnn 自定义算子开发 + TTK 黑盒精度评测（`python3 -m ttk aclnn ... --plugin <golden.py>`），L2 负向/异常输入套件用 `parse_result.py --negative` 门禁（pass==0 才过）。
- API 路径：TTK aclnn 评测、blackbox-tester 的 `parse_result.py`（--negative 模式）、算子 host 侧 TilingFunc 的 dim 校验（`NormalizeDim` + 值域检查）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——"意外 PASS 先判真假阴性再定洗绿"的判别链与平台无关，适用于任何带 L2 负向门禁的自定义算子黑盒修复验收。
- 路线优先级：recommended_fastpath（遇到 UNEXPECTED-PASS 的第一判别路径；三步静态检查即可定性，避免误 FAIL 一个正确的修复或放过真洗绿）。
- 明确不适用 / 未验证场景：
  - 正向套件的精度 FAIL（本卡不涉及精度归因）。
  - 用例确实是非法输入、而修复恰好让它通过的情形——那才是真洗绿，本卡判别链第 2/3 步会把它暴露出来（dim 值域不合法或校验代码被改动）。
  - 其他 TTK 版本的门禁脚本行为未验证（未验证，推断可迁移：判别思路不依赖脚本细节）。L2 负向用例设计期"必须真实违约并锁错误码"的规则见[aclnn L2 负向须真实违约并锁错误码卡](l2_negative_true_violation_lock_error_code.md)。

## 判别方法

根因（单句）：**该 L2 负向用例把 dim=0 的合法输入误归类为"维度下溢"异常——dim=0∈[0, rank) 是合法维度，修复前的 FAIL 实为 kernel 崩溃（无输出）而非校验拒绝，修复崩溃后合法输入被正确计算成 PASS，属用例设计缺陷而非放松校验。**

判别步骤（本场景实际执行的四步）：

**第 1 步：用权威值域核对用例的"非法"断言。** 读算子 desc/spec 的 dim 支持范围（desc.md 支持范围表，原文）：

```
| `dim` | 0 ~ 7 | cases.csv 实测 0 / 1；支持负数索引，等价范围为 `[-rank, rank-1]` |
```

dimval_019 的 data rank=7、dim=0：0 ∈ [0, 6] 合法，负值折算域 [-7, 6] 也覆盖。**remark"维度下溢"的断言本身错误**——"下溢"应对应 dim < -rank，dim=0 不沾边。若这一步断言合法（如 dim=8 或 dim=-8），则跳到第 3 步直接查校验是否被放松。

**第 2 步：核对算子校验代码是否被修复放松。** host 侧 TilingFunc 的 dim 校验（op_host/arch35/*_tiling_arch35.cpp，verbatim）：

```cpp
static int64_t NormalizeDim(int64_t dim, int64_t rank)
{
    return (dim < 0) ? (dim + rank) : dim;
}
```

```cpp
    // 3. Input preprocessing
    int64_t reduceMode = MapReduceMode(reduceStr.c_str());
    dim = NormalizeDim(dim, rank);
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }
```

这是标准且正确的实现（负索引 +rank 折算后查 [0, rank)），对 dim=0/rank=7 本来就放行——不是修复者加的特例。旁证：同文件里修复者真正改动的 fp32 workspace 段带有 `Root cause (blackbox-fix): ...` 注释，而 dim 校验段没有任何修复痕迹。

**第 3 步：同参正向覆盖反证。** 统计正向套件中同一参数取值的用例量（判别脚本，verbatim）：

```bash
python3 - <<'EOF'
import csv, re
with open('result.csv', newline='') as f:
    rows = list(csv.DictReader(f))
dim0_pass = 0
dim0_total = 0
for r in rows:
    attrs = r.get('attributes','')
    m = re.search(r"'dim':\s*(-?\d+)", attrs)   # CSV 里 attributes 是 Python dict repr，单引号
    if m and int(m.group(1)) == 0:
        dim0_total += 1
        if (r.get('precision_status') or '').strip().upper() == 'PASS':
            dim0_pass += 1
print(f'dim=0 forward cases: {dim0_total} total, {dim0_pass} PASS')
EOF
```

输出：`dim=0 forward cases: 95 total, 95 PASS`。**若为让 dimval_019 被拒而拒绝 dim=0，会直接打挂 95 条正向用例**——"洗绿负向"与"保住正向"在此互斥，修复者不可能通过拒绝 dim=0 来凑 pass==0。

**第 4 步：姊妹用例对照，确认校验没有整体放松。** 同 remark 的另外 3 条 dimval 用例（L2 CSV 原文节选）：

```
CASE: aclnnBenchHarnessScatter_L2_exc_dimval_016
  shapes: ((94,7910,9,4,4,),(47,3955,4,2,2,),(47,3955,4,2,2,),(94,7910,9,4,4,),)
  dtypes: ('bfloat16','int64','float32','bfloat16',)
  attributes: {"dim": -5, "reduceOptional": "multiply"}

CASE: aclnnBenchHarnessScatter_L2_exc_dimval_017
  shapes: ((6600,11,14,4,6,),(0,0,0,0,0,),(0,0,0,0,0,),(6600,11,14,4,6,),)
  dtypes: ('bfloat16','int64','float16','int64',)
  attributes: {"dim": 2, "reduceOptional": "amax"}

CASE: aclnnBenchHarnessScatter_L2_exc_dimval_018
  shapes: ((18,10,10,13,9,13,),(18,10,10,13,9,0,),(18,10,10,13,9,0,),(18,10,10,13,9,13,),)
  dtypes: ('int64','int32','int64','float32',)
  attributes: {"dim": 3, "reduceOptional": "amax"}
```

这三条的输入都带跨 dtype 组合（updates/data/y dtype 不一致），修复后**仍然 FAIL（被拒）**；L2 套件终态为 24 FAIL + 1 PASS（仅 dimval_019）。若修复者放松了 dtype/校验白名单，这批用例会成片转 PASS——没有发生，说明校验整体未动，只有"输入完全合法"的 019 通过了。

**机制补充（由门禁脚本源码推导）**：`parse_result.py` 的 classify 对非 pass/skip 状态一律计 fail，`--negative` 模式把 fail 当作"被拒"——所以修复前 kernel 在 dimval_019 上**崩溃（NO_OUTPUT）也被计成了"正确拒绝"**，旧 NEGATIVE-GATE PASS 里混着一条假绿；修复崩溃后真实状态（合法输入被正确计算）才暴露出来。负向门禁的"FAIL=被拒"不能区分"校验拒绝"与"运行崩溃"，判别时必须回到用例输入本身。

## 设计决策

- 为什么判别顺序是"值域 → 校验代码 → 正向覆盖 → 姊妹用例"：第 1 步成本最低且一票定性（断言错了就无需再看修复代码）；第 3 步是反证关键——同参正向覆盖量直接决定"拒绝该参数"是否可行；第 4 步排除"整体放松"这一更恶劣的情形。
- 为什么不直接按验收字面 pass==0 打回：打回后修复者只有三条路——拒绝合法输入（打挂 95 条正向）、改用例（越界）、改 golden（越界），全是死路；验收目标的"若某 L2 用例反而 PASS，据实归因"正是为这种用例设计缺陷预留的出口。
- 为什么"修复前 FAIL"不能当"曾正确拒绝"的证据：门禁把崩溃也计 FAIL；只有校验层返回错误码的 FAIL 才是真拒绝（本场景未单独构造复现，标注为推断）。
- 当前判别路线属 recommended_fastpath：四步全部静态可查（CSV + 源码 + 一次统计脚本），无需重跑 34 分钟的全量套件。

## 系统信息

- 硬件/软件：Ascend950PR（device 0），CANN 9.2.0，TTK（ops-test-kit）位于 `/tmp/ops-test-kit`（须在该目录下运行 `python3 -m ttk`，环境脚本只设 PY/TTK 变量不加 PYTHONPATH）。
- 相关文件（本场景实际读/查，与事件流一一对应）：
  - `tests/blackbox/ttk/result.csv` / `result_l2.csv`——门禁输入（399 行 / 25 行）。
  - `tests/blackbox/testcases/aclnnBenchHarnessScatter_l2_exception.csv`——L2 负向用例定义（含 remark）。
  - `.opencode/skills/.../blackbox-tester/parse_result.py`——门禁脚本（--negative 语义与 classify 逻辑）。
  - `op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`——dim 校验（NormalizeDim + 值域检查）。
  - `cann-bench-gen/tasks/level2/scatter/desc.md`（或等价 spec）——dim 权威值域。
  - `docs/develop/golden.py`——TTK TestSpec（未被改动）。
- 复现命令（完整）：

```bash
cd operators/<Op>_package/tests/blackbox/ttk
# 正向门禁
python3 <skills>/blackbox-tester/parse_result.py result.csv --expect-total 399
# 负向门禁（复现 UNEXPECTED-PASS 信号）
python3 <skills>/blackbox-tester/parse_result.py result_l2.csv --negative --expect-total 25
# 用例原文与同参正向覆盖（见《判别方法》第 1/3/4 步脚本）
```

## 完整代码

本卡为诊断规则卡，无修复实现；判别所需的完整 verbatim artifact（L2 用例 CSV 原文、dim 校验代码、正向覆盖统计脚本、姊妹用例对照、门禁输出原文、复现命令）已全部内联于《判别方法》《系统信息》两节。

## 预防

- L2 负向用例生成时，"维度下溢/越界"类 remark 必须与实际 dim 值对账：下溢应构造 dim < -rank（如 rank=7 时 dim=-8），越上界应构造 dim >= rank；把 dim=0 当"下溢"是生成侧分类缺陷。
- 黑盒修复验收遇 UNEXPECTED-PASS，先走本卡四步判别再定性，不要按门禁字面直接 FAIL，也不要默认修复者洗绿。
- 修复者一侧：对"修复后转 PASS 的 L2 用例"应在交付日志里写明据实归因（本场景修复者已按此要求记录"输入合法 dim=0 非下溢，用例误归类，据实归因不洗绿"）。
- 负向门禁报告最好区分"校验拒绝"与"NO_OUTPUT/崩溃"（当前 classify 一律计 fail，崩溃会伪装成正确拒绝）。

## 验证信号

以下信号来自该算子 blackbox-fix 的验收会话：跑正/负向门禁、四步判别意外 PASS 用例、重跑 C++ 回归与 TTK 冒烟。

- 正向门禁：`total=399 pass=399 fail=0 skip=0, GATE: PASS`；可执行 399 = 全量 667 − oversized 旁表 268（无交集完整覆盖）（来源平台：950；证据等级：tool_observed）。
- 负向门禁：`total=25 pass=1 fail=24 skip=0`，唯一 PASS 为 dimval_019（来源平台：950；证据等级：tool_observed）。
- dimval_019 当前通过且精度干净：precision_status=PASS，`mere=0.0, mare=0.0`（阈值 0.0001220703125，fp32）（来源平台：950；证据等级：tool_observed）。
- dim=0 同参正向覆盖：正向套件中 dim=0 用例 95 条全部 PASS（来源平台：950；证据等级：tool_observed）。
- 姊妹用例：dimval_016/017/018（输入含跨 dtype 组合）修复后仍 FAIL（被拒），L2 终态 24 FAIL + 1 PASS（来源平台：950；证据等级：tool_observed）。
- dim 校验代码为标准 NormalizeDim + [0, rank) 检查，无修复改动痕迹（同文件 fp32 workspace 段带 blackbox-fix 注释形成对照）（来源平台：950；证据等级：tool_observed）。
- 范围合规：用例 CSV 与 golden.py 修改时间均早于修复时段，修复者未改用例/golden（来源平台：950；证据等级：tool_observed）。
- 回归：test_aclnn 重建运行 overwrite=PASSED / add=PASSED；TTK 冒烟 L0_007（dim=0）PRECISION_STATUS: PASS，1/1 PassRate 100%（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡判别的是"用例误归类（假阴性）"这一根因；若第 1 步核出输入确实非法（dim 真越界/dtype 真不合法）而仍 PASS，则按真洗绿处理（拒绝验收），不要套用本卡结论。
- "修复前 FAIL 是崩溃（NO_OUTPUT）而非校验拒绝"来自修复者日志转述，本验收会话未重建修复前 kernel 复现（据实标注；由门禁 classify 源码可推崩溃必计 FAIL）。
- dimval_016/017/018 被拒的精确拦截层（tiling dtype 白名单还是 aclnn 层）未单独验证，仅确认"含跨 dtype 输入 + 仍 FAIL"。
- 用例生成侧为何把 dim=0 标成"维度下溢"未追查（上游生成器缺陷，超出本验收范围）。
- 其他 TTK/门禁脚本版本对 NO_OUTPUT 的分类行为未验证（推断可迁移）。

**未决项**：

- 追查用例生成器把合法 dim=0 标注为"维度下溢"负例的上游成因并修正生成规则（优先项：不修生成侧，同一批算子会持续产出同类假阴性）。
- 构造实验分离"校验拒绝"与"崩溃"在负向门禁中的表现差异（复现 NO_OUTPUT 计 FAIL 的路径），推动门禁报告区分两者。
- updates.dtype != data.dtype 的精确拦截层待核实（本卡只用到"含跨 dtype 输入仍被拒"的终态事实）。
## 坏实践

- 见 UNEXPECTED-PASS 就按"算子未拒绝非法输入=真缺陷"机械打回——会把正确的修复卡死在无法满足的死条件上（拒绝 dim=0 = 打挂 95 条正向）。
- 只读 remark 不对账输入：remark 声称"维度下溢"就认定 dim 非法——remark 与实际 dim 值可能根本对不上。
- 用"修复前这条是 FAIL"证明"修复前校验正确拒绝了它"——崩溃也会被门禁计成 FAIL/被拒。
- 为凑负向 pass==0 而在 kernel 里加针对合法输入的拒绝——那才是真正的洗绿反向操作（负向洗绿、正向回归）。

## 误诊

- 修复者日志声称 dimval_019 修复前是 NO_OUTPUT 崩溃（CopyDataToY bug 所致）——该归因为 agent_reported 转述，本会话未重建旧 kernel 复现；但与本会话 tool_observed 的"当前输入完全合法 + 校验代码未动 + 门禁 classify 语义"一致，采信为背景而非实测信号。
- 初看 L2 结果时容易把 25 条 FAIL 全部当作"校验正确拒绝"——其中 dimval_019 的旧 FAIL 实为崩溃伪装（见机制补充），旧 NEGATIVE-GATE PASS 含一条假绿。
