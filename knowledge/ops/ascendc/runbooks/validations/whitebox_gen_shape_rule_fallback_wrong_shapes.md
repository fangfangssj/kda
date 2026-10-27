---
type: "Runbook"
title: "白盒生成输出形状全错：golden 探测静默回退 shape_rule"
description: "白盒统一入口设 golden_path 后，golden 形状探测对 TestSpec 类 golden 不生效，output_shapes 静默落入 broadcast/placeholder 占位且 CSV 格式校验全过；改用 spec outputs[].shape_rule 并以 _shape_source 审计交付。复现增量：golden_path 的逐例子进程探测对 torch 实现 golden 必然失败（numpy probe 触发 .to() AttributeError）且每例 ~1s torch import 纯开销，全量结构目标（300+ 例）线性膨胀超时；输出形状可由 shape_rule 精确表达时置 design_facts.golden_path=null 跳过探测。"
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["agnostic"]
---

# 白盒生成输出形状全错：golden 探测静默回退 shape_rule

## 现象

为 ThnnFusedGruCellGrad（GRU Cell 反向算子）的 empty 空张量短路分支（Branch-2，tilingKey=2）生成 TTK kernel 模式白盒用例：scaffold 用 `--preset empty-explicit` 生成的 opspec.json 骨架**自带 5 条 `outputs[].shape_rule`**（从 spec.yaml 抽取，含 `grad_input_bias.shape = (3 * grad_hy.shape[1],) if has_bias else (0,)` 这样的属性条件式）。随后按统一入口的字段互斥规则"已有 golden_path 时不要再写 shape_rule"，在 opspec.json 里设了 `golden_path`（指向独立 oracle golden.py，期望输出形状从 golden 现算探测）、**删除了全部 outputs shape_rule**，再跑 `design-to-ttk` 生成 12 条空张量边界用例。

生成结果**表面全绿**：CSV 的 11 项格式校验全 PASS（含"tuple 字段长度与 output_shapes 一致"），无任何 warning。但逐条核对发现 **12 条用例的 output_shapes 全部错误**——五路输出都被填成输入形状的占位/广播值：

- B=0、H=8、has_bias=false 的用例：output_shapes = `((0,8),(0,8),(0,8),(0,8),(0,8))`，而正确值（golden 实跑）应为 `((0,24),(0,24),(0,8),(0,),(0,))`；
- has_bias=true 时两路 bias 输出应为 `(3H,)` 即 `(24,)`，生成值仍是 `(0,8)`；
- 检查 design_doc_cases.json 逐用例的 `_shape_source` 字段：全部为 `placeholder` / `broadcast`——golden 形状探测根本没生效，静默落入兜底占位。

派生关系算式（以 H=8、B=0 为例）：门梯度输出 (B,3H) = (0, 3×8) = (0,24)；`grad_hx.shape = grad_hy.shape` = (0,8)；bias 输出 has_bias=true → (3H,) = (24,)、false → (0,)；输入耦合 storage = (B,5H) = (0, 5×8) = (0,40)。占位形状 `((0,8)×5)` 把 3H/5H 的派生关系全部丢失——这样的用例集交付下游执行必然形状不匹配。

## 适用场景

- 工程形态：白盒 TTK 用例生成（ascendc-whitebox-design 统一入口：scaffold → opspec/design_facts → validate-inputs → design-to-ttk → coverage-audit / validate-shapes / validate-ttk）。
- API 路径：opspec.json `golden_path` 与 `interface.outputs[].shape_rule`（`shape_rule_kind: numpy_expr`）；design_doc_cases.json 逐用例 `_shape_source` 字段；golden 为 TTK TestSpec 类形态（类体内无 self 的 golden 函数）。
- 验证过的平台与 CANN 版本：agnostic——本卡全部证据为宿主侧生成管线行为（统一入口 + torch CPU golden 探测），按定义与硬件平台无关；生成产物目标为 950（ascend950 算子包）kernel 模式，但本会话未上板执行。与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（生成工具行为，跨算子）。
- 路线优先级：recommended_fastpath（spec `outputs[].shape_rule` 作为形状事实源；golden 保持运行期期望值 oracle，职责分离）。
- 明确不适用 / 未验证场景：plain-function 形态 golden 的探测行为未验证（本例为 TestSpec 类形态）；`shape_rule` 更复杂的跨输出相互依赖未验证。

## 根因

白盒统一入口的 golden 形状探测对 TTK TestSpec 类形态 golden 不生效，形状回退链静默落到 broadcast/placeholder 占位（无 warning、格式校验不拦截），叠加"设 golden_path 时按互斥规则删除 shape_rule"使 shape_rule 兜底缺失，生成语义错误的 output_shapes。（探测不生效的 helper 内部机制未穿透到源码层——统一入口按 skill 规则为黑盒；见「遗留问题」。）

## 判别方法

生成后立即审计 `_shape_source`（交付前的必检项）：

```bash
python3 - <<'EOF'
import json
from collections import Counter
d = json.load(open('<out_dir>/design_doc_cases.json'))
print('shape_source:', Counter(c['_shape_source'] for c in d['cases']))
# 期望：全 golden 或全 shape_rule；出现 broadcast/placeholder 即形状无据，禁止交付
EOF
```

再用独立 golden 探测逐形状核对（**必须类访问**，勿用实例访问——见「坏实践」）：

```bash
python3 -c "
import sys; sys.path.insert(0, 'docs/<op>/develop')
import numpy as np, golden
G = golden.<Op>TestSpec                      # 类访问
outs = G.golden(np.zeros((B,H), np.float32), np.zeros((B,5*H), np.float32), has_bias=True)
print([tuple(np.asarray(o).shape) for o in outs])
"
```

判别顺序：`_shape_source` 出现 placeholder/broadcast → 确认 golden_path 探测是否真的生效（可用非空结构边界 + golden_path 做一次黑盒探针生成，仍 broadcast/placeholder 即证伪）→ 改用 spec.yaml `outputs[].shape_rule` 兜底（管线文档的合法形状源：优先 golden，求值失败依次退回 spec shape_rule → 广播 → 首输入占位）。

## 解决配方

**before**（错误形状版本，opspec.json 完整 verbatim——设 golden_path、无 outputs shape_rule）：

```json
{
  "op_name": "ThnnFusedGruCellGrad",
  "category": "FusedComposite",
  "golden_path": "docs/ThnnFusedGruCellGrad/develop/golden.py",
  "interface": {
    "inputs": [
      {
        "name": "grad_hy",
        "param_type": "required",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "rank_range": [
          2,
          2
        ]
      },
      {
        "name": "storage",
        "param_type": "required",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "rank_range": [
          2,
          2
        ],
        "shape_rule": "storage.shape = (grad_hy.shape[0], 5 * grad_hy.shape[1])",
        "shape_rule_kind": "numpy_expr"
      }
    ],
    "outputs": [
      {
        "name": "grad_input_gates",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ]
      },
      {
        "name": "grad_hidden_gates",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ]
      },
      {
        "name": "grad_hx",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ]
      },
      {
        "name": "grad_input_bias",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ]
      },
      {
        "name": "grad_hidden_bias",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ]
      }
    ],
    "attrs": [
      {
        "name": "has_bias",
        "type": "bool",
        "kind": "enum",
        "values": [
          false,
          true
        ],
        "default": false
      }
    ]
  },
  "numerical_tolerance": {
    "source": "REQUIREMENTS.md §8.2.4 数值口径分析（spec.yaml numerical_tolerance；VALIDATION.md「精度验收标准」）",
    "per_dtype": {
      "float32": {
        "rtol": 1e-05,
        "atol": 1e-05,
        "metric": "max_relative"
      },
      "float16": {
        "rtol": 0.001,
        "atol": 0.001,
        "metric": "max_relative"
      },
      "bfloat16": {
        "rtol": 0.004,
        "atol": 0.004,
        "metric": "max_relative"
      }
    }
  }
}
```

**after**（修复版，opspec.json 完整——由一次 write + 两处 edit 逐字组装，终版未整体重写）：

```json
{
  "op_name": "ThnnFusedGruCellGrad",
  "category": "FusedComposite",
  "interface": {
    "inputs": [
      {
        "name": "grad_hy",
        "param_type": "required",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "rank_range": [
          2,
          2
        ]
      },
      {
        "name": "storage",
        "param_type": "required",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "rank_range": [
          2,
          2
        ],
        "shape_rule": "storage.shape = (grad_hy.shape[0], 5 * grad_hy.shape[1])",
        "shape_rule_kind": "numpy_expr"
      }
    ],
    "outputs": [
      {
        "name": "grad_input_gates",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "shape_rule": "grad_input_gates.shape = (grad_hy.shape[0], 3 * grad_hy.shape[1])",
        "shape_rule_kind": "numpy_expr"
      },
      {
        "name": "grad_hidden_gates",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "shape_rule": "grad_hidden_gates.shape = (grad_hy.shape[0], 3 * grad_hy.shape[1])",
        "shape_rule_kind": "numpy_expr"
      },
      {
        "name": "grad_hx",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "shape_rule": "grad_hx.shape = grad_hy.shape",
        "shape_rule_kind": "numpy_expr"
      },
      {
        "name": "grad_input_bias",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "shape_rule": "grad_input_bias.shape = (3 * grad_hy.shape[1],) if has_bias else (0,)",
        "shape_rule_kind": "numpy_expr"
      },
      {
        "name": "grad_hidden_bias",
        "dtype": [
          "bfloat16",
          "float16",
          "float32"
        ],
        "shape_rule": "grad_hidden_bias.shape = (3 * grad_hy.shape[1],) if has_bias else (0,)",
        "shape_rule_kind": "numpy_expr"
      }
    ],
    "attrs": [
      {
        "name": "has_bias",
        "type": "bool",
        "kind": "enum",
        "values": [
          false,
          true
        ],
        "default": false
      }
    ]
  },
  "numerical_tolerance": {
    "source": "REQUIREMENTS.md §8.2.4 数值口径分析（spec.yaml numerical_tolerance；VALIDATION.md「精度验收标准」）",
    "per_dtype": {
      "float32": {
        "rtol": 1e-05,
        "atol": 1e-05,
        "metric": "max_relative"
      },
      "float16": {
        "rtol": 0.001,
        "atol": 0.001,
        "metric": "max_relative"
      },
      "bfloat16": {
        "rtol": 0.004,
        "atol": 0.004,
        "metric": "max_relative"
      }
    }
  }
}
```

design_facts.json 同步修改（verbatim edit，两处）：

```json
// before（design_facts.json 头部节选）
  "golden_path": "docs/ThnnFusedGruCellGrad/develop/golden.py",
  "scenario_dimensions": {
// after（删除该行）
  "scenario_dimensions": {
```

```json
// before
  "gaps": [
    "确定性必检项（同输入连跑 ≥2 次五路输出位级一致）：TTK CSV 单条用例不表达重复执行语义，由下游执行 / 验收口径复核；本分支输出为常量 0 / 空张量、无浮点舍入与拓扑选择，DESIGN-BRANCH-2 §7 判定平凡满足（bitwise_reproducible）",
// after
  "gaps": [
    "输出形状来源：golden.py（TTK TestSpec 形态）不可被统一入口执行期形状探测（实测 _shape_source 回退 broadcast/placeholder），按管线兜底顺序改用 spec.yaml outputs[].shape_rule（numpy_expr，含 has_bias 条件分支）推导五输出 shape——已与 golden 对空 shape 的实跑输出逐形状核对一致（(0,24)/(0,8)/(24,)/(0,)/(4,0) 等）；期望值仍由 golden.py 在执行期现算（运行时经 TTK plugin_path + TestSpec 约定名 ThnnFusedGruCellGradTestSpec 解析，非被测 kernel）",
    "确定性必检项（同输入连跑 ≥2 次五路输出位级一致）：TTK CSV 单条用例不表达重复执行语义，由下游执行 / 验收口径复核；本分支输出为常量 0 / 空张量、无浮点舍入与拓扑选择，DESIGN-BRANCH-2 §7 判定平凡满足（bitwise_reproducible）",
```

修后验证（全部实测通过）：重新 `design-to-ttk` → `shape_source: Counter({'shape_rule': 12})`；12 条 output_shapes 与 golden 类访问探测逐条一致（如 B=0,H=8,bias=true → `((0,24),(0,24),(0,8),(24,),(24,))`）；属性条件式 shape_rule 求值正确（has_bias=true → `(24,)`，false → `(0,)`）——求值命名空间包含输入张量与 attrs。

## 设计决策

- 为何用 spec `outputs[].shape_rule` 而非坚持 golden 探测：管线文档规定形状源优先级"golden → spec shape_rule → 广播 → 首输入占位"；golden 探测被黑盒探针证伪（非空结构边界 + golden_path 仍 broadcast/placeholder），shape_rule 是 spec.yaml 单一事实源的确定性推导，scaffold 骨架本来就带。
- 备选（被放弃）：design_facts.json 保留 `golden_path` 仅作溯源——放弃原因：避免再次触发探测歧义；golden 路径改记 WORKFLOW_PROVENANCE.json 的 `inputs.golden_path` + `golden_role`（含运行期解析路径说明）。
- oracle 职责分离：golden 仍为运行期期望值 oracle（下游 TTK 经 plugin_path + TestSpec 约定名解析现算），本修复只改"形状事实源"，不动期望值独立性与容差——不是测试侧放松。
- 当前方案属 recommended_fastpath 路线。

## 系统信息

- 统一入口：`.opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py`（按 skill 规则黑盒使用，未读源码）；repo revision `cf5a9a82142099b1dbec604724f2237400ea79df`（用例产物为 worktree 未提交状态）。
- golden 探测环境：torch 2.7.1+cpu（宿主侧）。
- 相关文件：`operators/ThnnFusedGruCellGrad_package/tests/whitebox/branch-2/{opspec.json, design_facts.json, design_doc_cases.json, ttk_design_doc_cases.csv, WORKFLOW_PROVENANCE.json}`；`docs/ThnnFusedGruCellGrad/spec.yaml`（shape_rule 事实源）；`docs/ThnnFusedGruCellGrad/develop/golden.py`（独立 oracle）。

## 完整代码

生成与审计命令集（verbatim，`<out_dir>` = 用例输出目录）：

```bash
python3 .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py validate-inputs \
  <out_dir>/design_facts.json <out_dir>/opspec.json        # 期望 exit 0、无 error/warning

python3 .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py design-to-ttk \
  <out_dir>/design_facts.json <out_dir>/opspec.json \
  <out_dir>/design_doc_cases.json <out_dir>/ttk_design_doc_cases.csv

python3 - <<'EOF'
import json
from collections import Counter
d = json.load(open('<out_dir>/design_doc_cases.json'))
print('shape_source:', Counter(c['_shape_source'] for c in d['cases']))
EOF

python3 .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py coverage-audit \
  <out_dir>/design_doc_cases.json <out_dir>/design_facts.json <out_dir>/opspec.json

python3 .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py validate-shapes \
  <out_dir>/design_doc_cases.json <out_dir>/design_facts.json <out_dir>/opspec.json

python3 .opencode/skills/ascendc-whitebox-design/references/whitebox_designer.py validate-ttk \
  <out_dir>/ttk_design_doc_cases.csv <out_dir>/opspec.json
```

golden 独立探测（类访问，交付前逐形状核对）：

```bash
python3 -c "
import sys; sys.path.insert(0, 'docs/ThnnFusedGruCellGrad/develop')
import numpy as np, golden
G = golden.ThnnFusedGruCellGradTestSpec
outs = G.golden(np.zeros((0,8), np.float32), np.zeros((0,40), np.float32), has_bias=True)
print('B=0,H=8,bias=true :', [tuple(np.asarray(o).shape) for o in outs])
outs = G.golden(np.zeros((0,8), np.float32), np.zeros((0,40), np.float32), has_bias=False)
print('B=0,H=8,bias=false:', [tuple(np.asarray(o).shape) for o in outs])
outs = G.golden(np.zeros((4,0), np.float32), np.zeros((4,0), np.float32), has_bias=True)
print('B=4,H=0,bias=true :', [tuple(np.asarray(o).shape) for o in outs])
"
```

## 预防

- 白盒用例生成后**必检** `_shape_source` 全部 ∈ {golden, shape_rule}——出现 broadcast/placeholder 禁止交付（CSV 格式校验不查形状语义，占位形状照样 11 项全 PASS）。
- 设 `golden_path` 前先用一次小规模黑盒探针（非空结构边界）验证该 golden 形态可被探测；不可探测就不要依赖它，也别按互斥规则删掉 shape_rule 兜底。
- 输出形状依赖属性（如 bias 分支）时，用 numpy_expr 条件式 shape_rule 表达（`... if has_bias else (0,)`），求值命名空间含 attrs。

## 验证信号

本卡结论来自同一次 ThnnFusedGruCellGrad empty 分支白盒用例生成会话：统一入口修复前后各一次完整生成、一次黑盒探针实验、torch CPU 独立 golden 探测，以及三项审计（coverage-audit / validate-shapes / validate-ttk）。逐条实测信号：

- 修复前：首次生成 12 条用例 `_shape_source` 全为 placeholder/broadcast，output_shapes 全错（B=0,H=8,has_bias=false 用例为 `((0,8)×5)`，正确应为 `((0,24),(0,24),(0,8),(0,),(0,))`），而 CSV 11 项格式校验全 PASS。（来源平台：宿主侧生成管线（agnostic）；证据等级：tool_observed）
- 黑盒探针：保留 golden_path + 非空结构边界（empty_only=false）重新生成，`_shape_source` 仍为 `Counter({'broadcast': 5, 'placeholder': 2})`——对 TestSpec 类形态 golden 探测不生效。（来源平台：宿主侧生成管线（agnostic）；证据等级：tool_observed）
- 修复后：`shape_source: Counter({'shape_rule': 12})`；12 条 output_shapes 与 golden 类访问探测逐条一致（B=0,H=8,bias=true → `((0,24),(0,24),(0,8),(24,),(24,))`；B=4,H=0 → `((4,0),(4,0),(4,0),(0,),(0,))`；H=3000 fp16 → `((0,9000),(0,9000),(0,3000),(9000,),(9000,))`）；validate-shapes 12 checked / 0 violations（exit 0）；coverage-audit exit 0（empty_only 显式边界联合义务 12/12、0 违规）；validate-ttk PASS。（来源平台：宿主侧生成管线（agnostic）；证据等级：tool_observed）
- golden 独立探测（torch CPU，类访问）：四种空形状组合全部产出正确五路形状（`(0,24)/(0,8)/(24,)/(0,)/(4,0)` 等），与 shape_rule 推导一致。（来源平台：宿主侧 torch CPU（agnostic）；证据等级：tool_observed）

## 失效边界

- 本卡锚定 ascendc-whitebox-design 统一入口在 repo revision `cf5a9a8…` 的行为；其他 helper 版本未验证（推断可迁移）。
- plain-function 形态 golden 的探测行为未验证（本例为 TestSpec 类形态）。
- shape_rule 的 numpy_expr 求值命名空间已验证含输入张量与 attrs（has_bias）；更复杂的跨输出依赖未验证。
- 生成的用例集目标为 950 kernel 模式，但本会话未上板执行（属下游白盒执行任务）——形状正确性为生成期+golden 探测双重核对，非执行期实测。

## 坏实践

- 只看 CSV 格式校验（11 项 PASS）就交付白盒用例——格式校验不查形状语义，placeholder/broadcast 占位形状照样全过。
- 用**实例访问**探测 TestSpec golden（`spec = G(); spec.golden(a, b, has_bias=True)`）：类体内无 self 的普通函数经实例访问会把实例绑成第一个位置参数，报 `TypeError: ...golden() got multiple values for argument 'has_bias'`（本会话实测踩过）；必须类访问 `G.golden(a, b, has_bias=True)`。
- 设 `golden_path` 同时按互斥规则删 shape_rule，却不验证探测是否生效——等于拆掉兜底又没换来探测。

## 误诊

- "设了 golden_path，输出形状就会从 golden 现算来"——黑盒探针证伪：TestSpec 类形态 golden 下探测不产生 golden 来源形状（agent 当时的推断，后经探针 tool_observed 证据修正）。
- "输出形状错是 op_name 命名（snake_case/CamelCase）问题"——无关：命名只影响运行期 golden 解析，不影响生成期形状推导（见配套的 TTK golden 解析诊断知识）。



## 后续增量（2026-09-17，并行策展批：golden_path 逐例探测的成本维度与置 null 修复）

RoPE 算子 interleaved 分支（宿主 x86_64 生成环境，无 NPU 参与）补齐本卡的两个未验证点并新增超时形态：

**1. plain-function torch golden 的探测行为（补齐"plain-function 形态未验证"边界）**：torch 实现的 golden（内部 `.to(compute_dtype)` 等张量方法）同样**不可被探测**——引擎对每个用例 spawn 子进程 `infer_output_shapes_via_golden`：读 stdin JSON → `import golden` → 造 numpy zeros probe 入参 → 调 golden → 打印输出形状；probe 传入的是 numpy 数组，golden 内 `.to(compute_dtype)` 抛 AttributeError 被捕获后打印 "null" 退出——探测**必然失败并回退 shape_rule**。

**2. 成本维度（新失败形态：全量结构目标超时）**：探测虽必然失败，成本照付——每次子进程 `import torch` 约 0.6s（实测 `torch 2.7.1+cpu in 0.6 s`），约 1s/例；opspec 的 attr values 收窄到本分支后结构目标恢复全量（约 324 个 rank-4 目标）时，`design-to-ttk` 在 120s shell 超时被杀、无任何输出（此前小规模 70 例可完成——耗时随用例数线性膨胀）。

**3. 修复（golden_path=null）**：输出形状可由 opspec `outputs[].shape_rule` 精确表达（如 `queryOut.shape = query.shape`）时，`design_facts.json` 置 `"golden_path": null` 跳过逐例探测（golden 路径仍记入 `WORKFLOW_PROVENANCE.json` 的 `inputs.golden_path` 供下游执行侧算期望值）；缺省时输出形状依次回退 `spec/opspec outputs[].shape_rule` → 广播 → 首输入占位，期望值现算发生在下游 TTK 执行侧（golden 插件），与生成期 `golden_path` 无关——置 null 无信息损失。

### 复现增量验证信号

- golden_path 指向 torch golden + 全量结构目标：`design-to-ttk` 超过 120000ms 被 shell 终止、无输出（来源平台：宿主 x86_64 生成环境，无 NPU 参与（agnostic）；证据等级：tool_observed）。
- `golden_path: null` 后同命令退出码 0，输出 `wrote 301 generated cases`，耗时数秒（来源平台：宿主 x86_64（agnostic）；证据等级：tool_observed）。
- 生成的 `design_doc_cases.json` 中 301（终版 238）条用例 `_shape_source` 全部为 `shape_rule`、无 placeholder——形状信息无损失（来源平台：宿主 x86_64（agnostic）；证据等级：tool_observed）。
- torch import 单次成本探针：`torch 2.7.1+cpu in 0.6 s`；324 例 × ~1s/例 ≈ 324-648s（来源平台：宿主 x86_64（agnostic）；证据等级：tool_observed）。

### 复现增量边界与坏实践

- 置 null 的前提是 `outputs[].shape_rule` 能**精确**表达输出形状；形状不显然（非显然的推导/广播归约形状）的算子仍须 golden 探测，应改用更大的执行超时而不是置 null。
- 本增量只解决生成期探测成本；下游 TTK 执行侧的 golden 桥接（numpy↔torch）是另一独立问题，见 [TTK kernel模式torch golden的numpy桥接插件](ttk_kernel_golden_plugin_numpy_torch_bridge.md)。
- 引擎若改为批量/缓存探测，本卡耗时结论过期；"每用例一个子进程"基于当前实现。
- 坏实践增补：golden_path 指向 torch golden 却指望它给出输出形状（探测注定失败回退 shape_rule，白付每例约 1s 的 import 成本）；遇超时只加大 timeout 重跑（能跑完但浪费数分钟在注定失败的子进程上）。
