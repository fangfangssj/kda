---
type: "Runbook"
title: RoPE interleaved链白盒精度劣于half链
description: 同一 RoPE 算子白盒 TTK 成对实测：interleaved（DeInterleave+Muls+Interleave）链在 fp16/bf16 normal 值域相对 half（slice+neg）链新增 mare 超阈（fp16 72/72→52/72、bf16 60/72→44/72；BNSD 分支 mere 达 1e2~1e9 的值级错误），fp32 失配跨分支同构为既存现象；机制未定论，精度敏感场景回落 half 路线。复现增量：第三分支（interleaved-BSND-cos2d）D=88 严格分界证据——60 条失败集中两 D=88 形状、三 dtype 全失败，fp16 在 D=44/42/32/96 全过。
tags: [task.debugging, topic.precision, task.testing]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T01:05:00Z'
platforms: ["950"]
---

# RoPE interleaved链白盒精度劣于half链

## 现象

对自定义 RoPE 算子（BenchHarnessApplyRotaryPosEmb）的两条 interleaved 分支做 TTK kernel 模式白盒全量执行（独立 golden 为 torch 实现经 numpy 桥接插件，未改 kernel/tiling/用例，vendor 源、Ascend950PR_9579 上板）：

**分支一（interleaved-BSND-cos3d，tilingKey=5，rotaryMode=interleaved ∧ layout=0(BSND) ∧ cos rank=3）**：237 条用例（216 结构 + 边界/极值/空张量/错误注入）：

```
total=237 pass=136 fail=101 skip=0
GATE: FAIL (exit 1)
```

对 101 条失败逐用例分类（按合并 CSV 的 `remark=expect_error` 标记、`input_data_ranges` 极值 token、CRASH 状态三级判别）：

| 类别 | float16 | bfloat16 | float32 | 合计 | 说明 |
|------|---------|----------|---------|------|------|
| normal-range mare 超阈 | 16 | 16 | 32 | 64 | 真精度失败（mare 超容差） |
| 极值 ±65504/±inf/nan（NaN/Inf 不一致） | 4 | 12 | 12 | 28 | golden↔kernel NaN/Inf 不一致 |
| expect_error 正确拒绝 | 0 | 0 | 8 | 8 | 算子正确拒绝畸形输入（记 FAIL） |
| PROFILE_CRASH | 0 | 0 | 1 | 1 | 大 shape profiling 崩溃 |

**分支二（interleaved-BNSD-cos2d，branch-3，rotaryMode=interleaved ∧ layout=1）**：238 条用例 `total=238 pass=136 fail=102 skip=0`，按 `dyn_precision` 分类后真精度失败 93 例（FAIL,FAIL 90 + PASS,FAIL 3），其中 fp16 全部落在 normal-range 且失配幅度是值级错误而非边界容差问题（threshold 0.0009765625）：

```
case00050 dt=float16 rng=((-0.1, 0.2),...) mere=374.8, mare=2.0e6
case00058 dt=float16 rng=((-10, 10),...) mere=3.4e5, mare=2.0e9
case00068 dt=float16 rng=((-200, 200),...) reason: 'NaN/Inf mismatch'
```

FAIL,FAIL 90 = fp32 42（extreme 6 + normal 36）+ bf16 28（extreme 6 + normal 22）+ fp16 20（全部 normal-range，0 extreme）。

与同 harness 的 half 分支白盒存档成对对比（half-BSND-cos3d，tilingKey=4，238 条 pass=172 fail=66：normal-range mare 超阈 32 条全 fp32、极值 24=bf16 12+fp32 12、expect_error 9、crash 1，fp16 无任何失败）：

| dtype | half 分支 pass/total | interleaved 分支 pass/total | interleaved 新增失败 |
|------|---------------------|---------------------------|---------------------|
| float16 | 72/72 | 52/72 | normal 16 + extreme 4 |
| bfloat16 | 60/72 | 44/72 | normal 16 |
| float32 | 40/94 | 40/93 | 无（32 normal + 12 extreme 同构） |

失败 fp16 normal-range 用例的 shape/值域分布（可见明细样本）：`(7,55,403,72)` 值域 (-50,200)、(-1,2)、(-1,1)，`(7,54,395,76)` 值域 (-0.1,0.2)、(-0.1,0.1) 等——非单一值域或单一 shape 触发。当时 kernel 的 half 分支计算链为 slice+neg，interleaved 分支为 DeInterleave+Muls+Interleave，二者共用同一 NDDMA 搬运与 VF MulAdd 融合。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 精度验证——TTK kernel 模式白盒全量执行 + 跨路线（同算子相邻分支）成对对比；为 RoPE/双模式（half/interleaved）算子做路线选型与精度排查。
- API 路径：kernel 内 interleaved rotate_half 链（DeInterleave+Muls+Interleave + VF MulAdd）对照 half 链（slice+neg + VF MulAdd）；TTK `result.csv` 的 `precision_status`/`dyn_precision` 列与 `parse_result.py --expect-total` 门禁。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——RoPE/双模式算子的精度路线对比与失败分类方法（分类法适用于任何走 TTK kernel 模式的算子）。
- 路线优先级：safety_fallback——**fallback 声明**：在 interleaved 链 fp16/bf16 精度根因修复前，精度敏感场景应回落 half 路线；成对相对代价：同算子同 harness 白盒实测，interleaved 相对 half 在 fp16 normal 值域多 16 条 mare 超阈（72/72→52/72）、bf16 normal 值域多 16 条（60/72→44/72）、fp16 极值多 4 条，fp32 无差异（来源平台：950）。
- 明确不适用 / 未验证场景：fp32 normal/extreme mare 失配为跨分支既存现象（两分支各 32/12 条），不属本卡结论（见同域 fp32 跨分支同构卡）；极值类 NaN/Inf 不一致在两分支均恒定存在，归极值分诊卡；TTK aclnn 模式的结果列不同，未验证；分类结论依赖 golden 插件已被单例验证正常（若 golden 崩溃会全量 GOLDEN_FAILURE）。

## 判别方法

**第一步：对 gate FAIL 的全量结果先分类剥离非精度失败**。按 `dyn_precision` 分五类（Python 读 `result.csv`）：

1. `FAIL,FAIL` / `PASS,FAIL`：真精度失败候选（mere/mare 超阈或 NaN/Inf mismatch）；
2. `INVALID_TILING`：逐例看 attributes/input_shapes——畸形输入（layout 不支持、rotaryMode 不支持、cos 与 query 的 S/D/Dh 失配、D 奇数、attr 空等）是 host 侧 CheckShape/optiling 的**正确拒绝**，不是缺陷；
3. `DYN_COMPILE_FAILURE`：如 rank0 标量空 shape 的编译正确拒绝（expect_error 类）；
4. `PROFILE_CRASH`：超大 shape（约 538M 元素/张量）profiling 阶段崩溃，属基建/资源问题；
5. `PASS,PASS`/`100%,100%` 均为 pass（两种格式）。

分类探针（verbatim 取自事件流中写入的 `_gen_artifacts.py`）：

```python
EXTREME_TOKENS = ("3.3895e+38", "3.4e+38", "65504.0", "inf", "nan")
def is_extreme(r):
    rng = r.get("input_data_ranges","")
    return any(t in rng for t in EXTREME_TOKENS)

def first_dtype(s):
    m = re.search(r"'(float\d+|bfloat16|float16|float32)'", s)
    return m.group(1) if m else "?"

# --- failures categorization ---
cats = {}
fail_rows = []
for r in rows:
    name = r["testcase_name"]
    status = (r.get("precision_status") or "").strip().upper()
    mr = mrows.get(name, {})
    dtype = first_dtype(r.get("input_dtypes",""))
    remark = mr.get("remark","")
    if status == "PASS":
        continue
    if "expect_error" in remark:
        cat = "expect_error_reject"
    elif "CRASH" in status:
        cat = "profile_crash"
    elif is_extreme(mr):
        cat = "extreme_value"
    else:
        cat = "normal_range"
    cats[(dtype,cat)] = cats.get((dtype,cat),0)+1
    fail_rows.append((name,dtype,cat,status,r.get("dyn_precision",""),
                      mr.get("input_shapes",""),mr.get("input_data_ranges",""),remark))
```

**第二步：真精度失败按 dtype × 值域二维透视**——观察回归集中在哪些 dtype；mere 量级定性质（mere 1e2~1e9 与 threshold 1e-3 差 5~12 个数量级是输出值级错误，容差讨论无意义；mere ~1e-8 且 mare 超阈才可能是相对误差放大类问题）。

**第三步：与同 harness 相邻路线（half 分支）按 dtype 成对对比通过率**——fp32 同构 + fp16/bf16 新增 → 路线（计算链）相关而非搬运/公共路径相关。

## 解释

单一结论：interleaved rotate_half（DeInterleave+Muls+Interleave）链在 fp16/bf16 normal 值域的精度劣于 half（slice+neg）链——fp32 两分支失配完全同构（32 normal + 12 extreme 各自恒定）说明显微差异与 dtype 强相关；BNSD 分支 fp16 在 normal 数据范围 mere 即达 1e2~1e9，说明输出值级错误而非舍入。机制层根因（链中间精度、是否升 fp32 计算、DeInterleave/Interleave 取址、与 half 链升精度策略差异）**未追**，白盒执行契约只跑不改，失败交 developer 修复。

## 解决配方

本卡为诊断规则（允许没有成功修复）：分类完成后产出三件套交后续修复——`per_case_precision.csv`（逐例精度）、`whitebox_failures.md`（失败清单+分类表）、LOG 条目（如实报 gate FAIL 与分类计数）。修复动作以 FAIL,FAIL/PASS,FAIL 的最小复现 shape（如 (3,7,19,32) fp32）为入口，从 interleaved 链路取址与 DeInterleave/Interleave 对齐查起。路线处置：根因修复前精度敏感场景回落 half 路线（见适用场景 fallback 声明）。

## 设计决策

- 为什么先分类再汇报：gate 的 fail=N 把正确拒绝、基建崩溃与真精度失败混在一起，直接报数字会误导优先级；expect_error 用例计入 gate fail 是 harness 计数口径，不代表回归。
- 为什么用 mere 量级分性质：mere 1e2~1e9 与 threshold 1e-3 差 5 个数量级以上，任何容差讨论都无意义，只能按"输出错误"处理——排查方向是取址/对齐/坐标反算，而不是数值精度策略。
- 判读顺序上先排除既存机制（fp32 跨分支同构、极值 NaN/Inf 不一致）再下"路线回归"结论——混在一起会误判成实现 bug 或掩盖新增回归。
- 放弃的候选：逐例深挖 93 例各自的失配模式（任务边界不允许修复，深挖投入不成比例；留给 developer 以最小复现切入）。
- 当前方案属 safety_fallback（half 路线回落）；interleaved 路线本身的缺陷未修复前不能视为可用。

## 系统信息

- 硬件/软件：Ascend950PR_9579，CANN 9.2.0；TTK（ops-test-kit，`ttk kernel` + golden 插件 + parse_result gate）。
- 被测算子：BenchHarnessApplyRotaryPosEmb（vendor 安装源）；interleaved-BSND-cos3d 分支 tilingKey=5、interleaved-BNSD-cos2d 分支 branch-3；对照 half-BSND-cos3d 分支 tilingKey=4 白盒存档。
- 产物：`result.csv`、`per_case_precision.csv`、`whitebox_failures.md`、分类脚本 `_gen_artifacts.py`。

## 验证信号

来源说明：以下信号来自同一算子两条 interleaved 分支的白盒全量执行会话与 half 分支白盒存档的成对对比（同一白盒 harness、同源 golden、同执行环境；half 分支数值来自存档失败清单文件读取）。

- interleaved-BSND-cos3d 分支：`total=237 pass=136 fail=101 skip=0`（GATE: FAIL, exit 1），分类分布 normal-range 64=fp16 16+bf16 16+fp32 32、极值 28、expect_error 8、crash 1（来源平台：950；证据等级：tool_observed）。
- interleaved-BNSD-cos2d 分支：`total=238 pass=136 fail=102 skip=0`（GATE: FAIL）；dyn_precision 分类 FAIL,FAIL 90（fp32 42/bf16 28/fp16 20；extreme 12/normal 78）+ PASS,FAIL 3 + INVALID_TILING 7 + DYN_COMPILE_FAILURE 1 + PROFILE_CRASH 1；PASS,PASS 132 + 100%,100% 4 = 136 pass（来源平台：950；证据等级：tool_observed）。
- BNSD 分支 fp16 normal-range 失配幅度：mere=374.8/3.4e5、mare=2.0e6/2.0e9、NaN/Inf mismatch，threshold 0.0009765625；PASS,FAIL 3 例形态为 fp32、query 侧 mere 1e-8 pass、key 侧超阈（来源平台：950；证据等级：tool_observed）。
- INVALID_TILING 7 例逐例 attributes/shapes 核对为 expect_error 畸形输入（layout=2、rotaryMode=full、S 失配、D=5 奇、Dh 失配、D 失配、attr 空）（来源平台：950；证据等级：tool_observed）。
- half-BSND-cos3d 分支存档：238 条 pass=172 fail=66，normal-range mare 超阈 32 条全 fp32（fp16 0 条）、极值 24=bf16 12+fp32 12（来源平台：950；证据等级：tool_observed，存档失败清单文件读取）。
- 成对对比：fp16 72/72→52/72、bf16 60/72→44/72、fp32 40/94→40/93（dtype 分档通过率中 half 分支数值转述自存档统计）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论基于同一算子、同一白盒 harness、同源 golden 的分支间成对对比；跨算子、跨 harness 外推未验证（未验证，推断可迁移——成对对比判别方法本身与算子无关）。
- fp32 normal/extreme 失配为跨分支既存现象，本卡不解释其机制；极值 NaN/Inf 不一致与 expect_error 正确拒绝两分支均存在且数量恒定，不构成路线差异。
- interleaved 链缺陷归因未到机制层，不排除 golden 自身对 interleaved 的实现差异贡献了部分失配（golden 直调 sane 已验，但逐位对照未做）。
- 若 interleaved 链的 fp16/bf16 根因被修复（如补升精度），本卡对比结论需复测后失效。
- PROFILE_CRASH（(8,128,4096,128)）的成因（OOM/超时）未诊断，属基建边界。
- 分类法依赖 golden 插件先被验证正常（否则全量 GOLDEN_FAILURE 会吞掉一切信号）。

**未决项**：

- **优先项：复现并追根因到机制层**——在 interleaved 链（DeInterleave+Muls+Interleave）中定位 fp16/bf16 精度损失点（中间 cast 精度、MulAdd 累加顺序、取址与对齐、与 half 链升精度策略的差异），以最小复现 shape（如 (3,7,19,32) fp32）切入，给出修复或确认链固有精度特性；修复后重跑白盒全量并与 half 分支复测成对对比，确认回归消除。
- 失败 fp16 用例 shape 集中于非对齐 D（72/76）的观察仅基于截断明细样本（约 10 条），完整 shape-D 相关性未统计。（复现增量已用第三分支的 dtype×shape 全量分组补上严格分界证据，见下。）
- BNSD 分支 extreme-range 12 例与 golden 自身极值溢出的贡献比例未拆分。
## 坏实践

- 把 gate 的 fail=101/102 直接当 kernel 缺陷总数——expect_error 是正确拒绝、crash 非精度、极值 NaN/Inf 不一致归因存疑；真 normal-range 失败须分类后才知道。
- 只看总通过率不看 dtype × 值域分布——fp32 既存失配（两分支同构）会掩盖 fp16/bf16 的新增路线回归。
- 用 mere 1e2~1e9 的失败去讨论 rtol/atol 容差——量级差 5 个数量级以上，容差不是问题。
- 白盒执行阶段顺手"修" kernel 或放宽容差——白盒契约只跑不改，修复应回到 C++ 工程按红绿循环闭环。

## 误诊

- 分类前曾按相邻 half 分支的经验预期失败模式（fp32 为主）；分类后证实 fp32 部分确实同构，但 fp16/bf16 normal 值域各 16 条新增回归超出经验预期，为成对对比的新发现（分类前的预期本身构成一次轻量误诊方向，已被 dtype 透视纠正）。
- INVALID_TILING=正确拒绝 的判别要逐例核对 attributes/shapes：若合法输入也 INVALID_TILING，则是 tiling 实现缺陷（另一类问题），不能套用本卡结论。

## 增量（第三分支 interleaved-BSND-cos2d：D=88 严格分界）

第三条分支轨迹（interleaved-BSND-cos2d，tilingKey=1，BSND + interleaved + cos.ndim=2，同算子 TTK kernel 模式白盒 238 条，Ascend950PR_9579 / CANN 9.2.0）补上完整的 shape-D 相关性证据：**60 条失败严格集中于两个 D=88 形状**（(8,57,427,88) 与 (8,57,430,88)，Dh=44），三 dtype 各 20 条（每形状 12 条中 10 失败 + 2 平凡通过）；fp16 在 D=44/42/32/96 形状全部 12/12 通过、唯 D=88 失败——失败按 D=88 严格分界且跨 dtype。失配为正确性级别（fp32 case00107 双输出 mere=8986/9117、mare=2.0e+7，阈值 1.22e-4）；同形状仅有的通过用例是全零/NaN 平凡输入（mere=0.0），说明 kernel 执行链没崩（无 507035/INVALID_TILING），是数据通路按内容出错。

D=44（fp16 通过）与 D=88（三 dtype 全失败）的分界说明边界不在"任意非对齐 D"（44×2B=88B 同样非 32B 对齐）；候选方向：Dh=44 偶奇拆分粒度、padded 平面布局（paddedD=alignUp(D,16)）的填充/回读算术、小 D 自定义 `__simd_vf__` InterleaveVF 路径选择（机制未追，D%16=8 待最小形复现，如 (1,2,1,88) fp16）。

dtype×shape 分组统计探针（判别主工具，与主卡分类法同构）：

```python
import csv, ast
from collections import defaultdict
rows = list(csv.DictReader(open('<pkg>/tests/whitebox/ttk/result.csv')))
combo = defaultdict(lambda: [0, 0])
for r in rows:
    shape = ast.literal_eval(r['input_shapes'])[0]
    dt = ast.literal_eval(r['input_dtypes'])[0]
    ok = r.get('precision_status', '').upper() == 'PASS'
    combo[(str(shape), dt)][0 if ok else 1] += 1
for (shape, dt), (p, f) in sorted(combo.items()):
    print(f'{shape:32s} {dt:10s} pass={p} fail={f}')
```

验证信号（复现增量）：

- 汇总门禁：`total=238 pass=138 fail=100 skip=0`、`GATE: FAIL`（exit 1）；按 dtype bf16 44/72、fp16 52/72、fp32 42/94（来源平台：950；证据等级：tool_observed）。
- D=88 失败组：两形状 × 三 dtype = 60 条；每 dtype pass=4 fail=20（4 条平凡通过 = 全 0/NaN 范围 × 2 形状）；失败分类统计 D=88_correctness: 60（来源平台：950；证据等级：tool_observed）。
- 失配量级：case00107（fp32，(8,57,427,88)，范围 (-50,200)）双输出 mere=8986.15/9116.53、mare=20000000.0；同形状全零用例 mere=0.0、mare=0.0 通过（来源平台：950；证据等级：tool_observed）。
- 对照组：fp16 在 (3,7,17,44)/(3,7,18,42)/(3,7,19,32)/(7,55,411,96) 全部 12/12 通过，唯 D=88 两形状各失 10 条（来源平台：950；证据等级：tool_observed）。
- 失败报告产物：`whitebox_failures.md` + `per_case_precision.csv` 随会话落盘（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 本增量结论限于 D=88（及"非 16 倍数大 D"推断类形状，未验证）；该分支 100 条失败中的其余三类另有归因：8 条 expect_error 为算子正确拒绝（INVALID_TILING/DYN_COMPILE_FAILURE）、16 条极值范围 NaN/Inf 不一致、15 条 fp32 常规范围 mare 超阈、1 条 PROFILE_CRASH——均不属 D=88 组。
- 初看 238 条中 108 条 `in_D != out_D`（CSV 输出 D 保留修复前奇值）曾疑形状不一致；实测该不一致为良性（108 条中 65 条通过），与 D=88 失败无因果。
- "interleaved 路径特有"的方向性判断基于本切片 fp16 D=44 通过 vs D=88 全 dtype 失败的分界；与 half 链的成对对比沿用主卡存档结论。