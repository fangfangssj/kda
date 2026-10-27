---
type: "Runbook"
title: TTK 白盒 CSV op_name 蛇形与 golden 注册键不一致
description: TTK kernel 模式 CSV op_name 必须是 OpType 蛇形（驼峰/aclnn 名会 SOC_NOT_SUPPORT），而 golden TestSpec 的 __spec__ 若只注册驼峰/aclnn 键则 golden 查找不到，精度全 FAIL。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TTK 白盒 CSV op_name 蛇形与 golden 注册键不一致


## 现象

在一个 ACLNN 算子仓（对标 torch 的 GRU Cell 反向算子）里为自定义算子生成 TTK kernel 模式白盒用例 CSV 时，需要确定 CSV 的 `op_name` 列写什么。仓库里留有一次先期 TTK 环境验证（冒烟）的产物：`tests/whitebox/ttk/result_smoke.csv` 只有 1 行，`op_name=thnn_fused_gru_cell_grad`，`precision_status: FAIL`；对应的 `smoke_run.log` 里有两条关键记录：

```
52:2026-09-06 01:30:12,825 [WARNING] [996077 thnn_fused_gru_cell_grad_smoke_01 OnGenGolden] [output_generation.py:272]: Golden function for operator thnn_fused_gru_cell_grad is not provided!
89:Golden Shape: ('UNSUPPORTED',)
90:Golden Dtype: ('UNSUPPORTED',)
```

即：TTK 找不到该算子的 golden 函数，golden shape/dtype 判 UNSUPPORTED，精度比对 FAIL。追查名字注册链路后发现两套组件的命名口径互相矛盾：

- 下游执行技能的静态预检 `preflight.py` 硬性要求 CSV `op_name` 必须是 `<OpType>` 的**蛇形**（驼峰 `ThnnFusedGruCellGrad` 或 aclnn 名 `aclnnThnnFusedGruCellGrad` → `SOC_NOT_SUPPORT`），依据是"kernel 模式按 snake_case 查算子"；
- 而工程参考 golden（`docs/<OP>/develop/golden.py`，TestSpec 格式）的 `__spec__` 注册表**只有驼峰键和 aclnn 键，没有蛇形键**，其 docstring 声明"kernel / GEIR 模式按 CSV op_name 查找"；
- GEIR 测试套件为此自建了一份 golden 副本 `geir_spec.py`，专门注册了蛇形键。

## 适用场景

- **工程形态**：TTK（ops-test-kit）kernel 模式 / GEIR 模式的白盒或 GEIR 用例 CSV 编写；TestSpec 格式 golden 插件（`--plugin`）编写与排障。
- **API 路径**：`ttk kernel -i <csv> --plugin <golden.py>`；whitebox-tester 的 `preflight.py` 静态预检；TestSpec 的 `__spec__` 注册表（loader AST 静态扫描，值为类名字符串）。
- **验证过的平台与 CANN 版本**：950（实机 Ascend950PR_9589）+ CANN 9.0.0-beta.2（TTK 冒烟日志中 vendor op info 从 `config/ascend950/aic-ascend950-ops-info.json` 加载，CANN 安装路径为 `cann-9.0.0-beta.2`）。
- **适用粒度**：platform_wide（该契约由 TTK 框架与 preflight 脚本决定，适用于该环境下任何算子的 kernel 模式 CSV，与具体算子无关）。
- **路线优先级**：recommended_fastpath（CSV 用蛇形 op_name + golden __spec__ 补注册蛇形键，是同时满足两套查找的唯一解）。
- **明确不适用 / 未验证场景**：aclnn 模式按 CSV `api_name` 查找（应注册 aclnn 键，本卡未验证 aclnn 通路）；其他 CANN/TTK 版本是否对 op_name 做驼峰/蛇形归一化未验证；本卡证据来自 TDD 红相位会话（kernel 未实现），golden 查找成功后的精度结果不在本卡范围。

## 判别方法

**触发条件**：TTK kernel 模式运行后所有用例 `precision_status=FAIL`，或日志出现 `Golden function for operator <name> is not provided!`、`Golden Shape: ('UNSUPPORTED',)`。

**可执行探针**（按序执行）：

1. 确认运行命令是否带了 `--plugin`。冒烟日志第 1 行是 `Command: ttk op -i .../smoke_kernel.csv`——**没有 `--plugin`**，此时 "Golden function not provided" 是"根本没给 golden 插件"，不是名字不匹配。先排除这个再查名字。
2. grep 运行日志找 golden 报错原文：
   ```bash
   grep -n -i "golden" <run.log> | head -20
   ```
3. 核对 CSV `op_name` 与 OpType 的蛇形是否一致（preflight 本体就是探针）：
   ```bash
   python3 <whitebox-tester skill 目录>/preflight.py <merged.csv> --optype <OpType> [--opp $ASCEND_OPP_PATH]
   ```
   非 0 退出时逐条打印不通过项（op_name 应为 `<OpType>` 的蛇形名）。
4. 核对 golden `__spec__` 注册键是否覆盖 CSV `op_name`（kernel/GEIR 模式查找键 = CSV op_name）：
   ```bash
   grep -n "__spec__" docs/<OP>/develop/golden.py
   ```

**原始观测**（本卡证据，来源见 `## 验证信号`）：冒烟结果 CSV `precision_status: FAIL`；日志 WARNING `Golden function for operator thnn_fused_gru_cell_grad is not provided!`；`develop/golden.py` 的 `__spec__` 只注册 `ThnnFusedGruCellGrad` / `aclnnThnnFusedGruCellGrad` 两键；GEIR 套件 `geir_spec.py` 注册蛇形键 `thnn_fused_gru_cell_grad`。

**解释**：TTK kernel 模式有两个按名字查找的注册点——算子本体按 CSV `op_name`（必须蛇形，否则 SOC_NOT_SUPPORT）和 golden 插件按 CSV `op_name` 查 `__spec__` 键。CSV 只有一个 `op_name` 列，因此 golden 的 `__spec__` **必须注册与 CSV 相同的蛇形键**；只注册驼峰/aclnn 键的 golden 在 kernel 模式下永远查不到，表现为 golden UNSUPPORTED + 精度全 FAIL（非算子精度问题）。

## 设计决策

- **为何 CSV 取蛇形而不是驼峰**：preflight 与 TTK 算子查找都以蛇形为准（驼峰/aclnn → SOC_NOT_SUPPORT，直接跑不起来）；golden 侧补一个蛇形键只是注册表改动，不触碰数学实现，代价最小。golden 作者 docstring 里"kernel 模式 op_name 用驼峰"的意图与 preflight 契约冲突时，以 preflight/TTK 查找口径为准。
- **GEIR 套件先例**：GEIR 模式同样按 CSV op_name 查 golden，该套件的解法是自建 `geir_spec.py` 副本注册蛇形键——佐证 TTK 查找是精确匹配、不做名字归一化（此点为类比推断，未实测，见 `## 未决事项`）。
- **本卡所在路线**：recommended_fastpath（蛇形 CSV + golden 补蛇形键）；不建议走"CSV 写驼峰迁就 golden"——会被 preflight 以 SOC_NOT_SUPPORT 拦下。

## 系统信息

- 工具链：TTK（ops-test-kit）kernel 模式；CANN 9.0.0-beta.2；实机 Ascend950PR_9589（冒烟日志 vendor op info：`opp/vendors/<OpType>/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json`）。
- 相关文件（与本卡证据一一对应）：
  - `operators/<OP>_package/tests/whitebox/ttk/smoke_run.log`、`result_smoke.csv`、`smoke_kernel.csv`（冒烟产物）
  - `.opencode/skills/whitebox-tester/preflight.py`（蛇形契约的执行者）
  - `.opencode/skills/whitebox-tester/SKILL.md`（步骤 2 定位 golden：`--plugin "$GOLDEN"`；步骤 2.5 用 `--ti=1` 单例冒烟提前暴露 golden/部署/环境问题）
  - `docs/<OP>/develop/golden.py`（工程参考 golden，TestSpec 格式）
  - `operators/<OP>_package/tests/geir/geir_spec.py`（GEIR 自建蛇形键 golden 副本）
  - 生成侧 `opspec.json` 的 `op_name` 字段（决定 CSV op_name 列）

## 完整代码

**preflight.py 的判定逻辑**（verbatim，含驼峰转蛇形与逐条核验）：

```python
def _camel_to_snake(name):
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def _op_names(path):
    with open(path, newline="") as f:
        return sorted({r["op_name"] for r in csv.DictReader(f) if r.get("op_name")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--optype", required=True)
    ap.add_argument("--opp", default=os.environ.get("ASCEND_OPP_PATH", ""))
    a = ap.parse_args(argv)

    ok = True
    expected = _camel_to_snake(a.optype)
    for op in _op_names(a.csv):
        if op != expected:
            ok = False
            print(f"[preflight] FAIL: op_name '{op}' 应为 <OpType> 的蛇形名 '{expected}'"
                  f"（kernel 模式按 snake_case 查算子，勿用驼峰/aclnn 名）")

    impl = os.path.join(a.opp, "vendors", a.optype, "op_impl", "ai_core", "tbe", f"{a.optype}_impl")
    if a.opp and not os.path.isdir(impl):
        ok = False
        print(f"[preflight] FAIL: 找不到 python impl 目录 {impl}"
              f"（确认算子已部署到 vendors/{a.optype}）")

    print("[preflight] PASS" if ok else "[preflight] 有不通过项，修复后重试")
    return 0 if ok else 1
```

**工程 golden 的注册表（问题形态，verbatim）**——只有驼峰/aclnn 键，kernel 模式按 CSV op_name（蛇形）查找时miss：

```python
# TestSpec 注册（loader AST 静态扫描，值为类名字符串）：
##   kernel / GEIR 模式按 CSV op_name 查找；aclnn 模式按 CSV api_name 查找。
__spec__ = {
    "ThnnFusedGruCellGrad": "ThnnFusedGruCellGradTestSpec",
    "aclnnThnnFusedGruCellGrad": "ThnnFusedGruCellGradTestSpec",
}
```

**修复方向（注册性修改，不改数学）**——在 `__spec__` 补蛇形键：

```python
__spec__ = {
    "thnn_fused_gru_cell_grad": "ThnnFusedGruCellGradTestSpec",   # kernel/GEIR 模式查找键 = CSV op_name（蛇形）
    "ThnnFusedGruCellGrad": "ThnnFusedGruCellGradTestSpec",
    "aclnnThnnFusedGruCellGrad": "ThnnFusedGruCellGradTestSpec",  # aclnn 模式查找键 = CSV api_name
}
```

（GEIR 套件 `geir_spec.py` 的等价注册：`__spec__ = {'thnn_fused_gru_cell_grad': 'ThnnFusedGruCellGradTestSpec'}`。）

## 预防

生成/编写 TTK kernel 模式 CSV 前的检查清单：

1. CSV `op_name` = `_camel_to_snake(OpType)`；生成侧把 `opspec.op_name` 设为蛇形，让它流入 CSV（不要写驼峰 OpType 或 aclnn 名）。
2. golden `__spec__` 必须含与 CSV `op_name` 相同的蛇形键（kernel/GEIR 模式）；aclnn 模式另注册 aclnn 键。
3. 全量跑之前先用 `--ti=1` 单例冒烟（开 CANN debug 日志）暴露 golden/部署/环境问题——golden 缺键这类基础设施问题在这里就会现形，精度 FAIL 不拦但 golden UNSUPPORTED 要先修。
4. 交付白盒用例集时把 `design_facts.json` / `opspec.json` 与 cases 一起留档在套件目录——下游 st-verifier 的 `coverage_audit.py` / `shape_audit.py` 三件缺一即判 FAILED（黑盒套件 `design/` 子目录是同款先例）。
5. 读到 "Golden function ... is not provided!" 先确认运行命令是否带 `--plugin`，再查名字注册。

## 验证信号

以下信号来自一次白盒用例设计会话中对仓库留存的 TTK 冒烟产物与工具链源码的实测读取（冒烟本身是先期环境验证任务在实机上运行的记录，本会话以工具读取其日志/结果原文），以及生成侧对蛇形 op_name 的校验实跑：

- 冒烟结果 CSV 单行 `{'testcase_name': 'thnn_fused_gru_cell_grad_smoke_01', 'op_name': 'thnn_fused_gru_cell_grad', 'precision_status': 'FAIL'}`（来源平台：950，TTK 实机运行结果 CSV；证据等级：tool_observed）
- 冒烟日志 WARNING `Golden function for operator thnn_fused_gru_cell_grad is not provided!`（output_generation.py:272）与 `Golden Shape: ('UNSUPPORTED',)`；同时确认冒烟命令 `ttk op -i .../smoke_kernel.csv` 未带 `--plugin`（来源平台：950，TTK 实机运行日志；证据等级：tool_observed）
- preflight.py 源码确认蛇形契约：`op_name != _camel_to_snake(optype)` 即 FAIL 并提示"SOC_NOT_SUPPORT"（来源平台：950 开发环境；证据等级：tool_observed）
- `develop/golden.py` `__spec__` 仅注册 `ThnnFusedGruCellGrad`/`aclnnThnnFusedGruCellGrad` 两键（无蛇形键）；GEIR `geir_spec.py` 注册蛇形键（来源平台：950 开发环境；证据等级：tool_observed）
- 生成侧采用蛇形 `opspec.op_name=thnn_fused_gru_cell_grad` 后，`validate-ttk` 校验 `[PASS] op_name 格式正确: {'thnn_fused_gru_cell_grad'}`、13 项校验全 PASS 退出码 0（来源平台：950 开发环境宿主侧；证据等级：tool_observed）

## 失效边界

- "TTK 不做名字归一化"是按 GEIR 模式旁证 + preflight 文案的推断，kernel 模式带 `--plugin` 的实际查找行为未在本轨迹内实测（TDD 红相位，kernel 未实现未跑全量）；其他 CANN/TTK 版本可能改变查找口径——未验证，推断可迁移。
- aclnn 模式按 `api_name` 查找（注册 aclnn 键），本卡结论不适用；GEIR 模式若套件自带蛇形键 spec 副本则天然满足。
- golden 补蛇形键的修复动作属于下游执行侧（whitebox-tester 单例冒烟会拦截并按"修 golden.py"处置）；本卡只锁定契约与判别方法，不宣称 golden 修复后精度通过。

## 坏实践

- kernel 模式 CSV 写驼峰 `OpType` 或 `aclnn` 名 → preflight SOC_NOT_SUPPORT，全量根本起不来。
- 看到 "Golden function not provided" 就断定名字注册不匹配，不先查运行命令是否带 `--plugin`——会把"没给插件"误诊成"键不匹配"。
- golden 只注册作者习惯的驼峰键就交付，依赖下游"反正会报错"兜底——单例冒烟能拦，但全量前不跑单例就是带病全量。

## 误诊

- **误诊一（本轨迹实际发生）**：先入为主认为冒烟 FAIL 是"蛇形 op_name 未匹配 golden 注册"，随后核对冒烟命令发现其根本未带 `--plugin`——该次 FAIL 的直接原因是无插件（且属预期的红相位冒烟），不是键不匹配。名字契约结论由 preflight 源码与 `__spec__` 注册表独立支撑，与该误诊无关。
- **误诊二（潜在）**：把 golden UNSUPPORTED 导致的全 FAIL 当成 kernel 精度问题去调 kernel——先看日志里 golden 是否找到，再谈精度。


