---
type: "Runbook"
title: TTK CSV attributes 需 Python 字面量，JSON 被拒
description: TTK aclnn 黑盒执行的 CSV attributes 字段由 process_dict 以 eval 解析，JSON 布尔值 false/true 触发 ValueError；对合并执行 CSV 做 JSON→Python 字面量归一化后全量执行恢复。 复现增量：生成源定位 ascendc-st-design 的 format_ttk_dict（json.dumps，重生成后必须重跑归一化）；int 形（maximize=0）可解析的判别信号；ast.literal_eval 解析 input_data_ranges（裸 nan/inf/±0 token）必报错与裸字符串替换 false→False 会误伤两条坏实践；第二轨迹全量 642/642 单元格解析通过、ttk aclnn --validate L0/L1/L2 全 valid。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T04:06:11Z'
platforms: ["950"]
---

# TTK CSV attributes 需 Python 字面量，JSON 被拒

## 现象

在对一个自研 aclnn 自定义算子（对标 torch 的 GRU Cell 反向算子）做 TTK aclnn 模式黑盒精度验收时：用例 CSV 由上游用例设计阶段产出（标准 L0/L1/L2 三张 CSV，`attributes` 列写的是 JSON 语法，如 `{"hasBias":false}`），执行前用合并脚本按列名合并成正向套件（559 条）与负向套件（49 条）两张执行 CSV，`attributes` 原样透传。golden 以 `--plugin` 直接指向 TestSpec 文件。按流程先跑单条冒烟：

```bash
cd "$TTK"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
env PYTHONPATH="$TTK:$(dirname "$GOLDEN"):$ASCEND_TOOLKIT_HOME/python/site-packages" \
    "$PY" -m ttk aclnn -i "$MERGED" -t "$SMOKE_CASE" \
    --pc=1 --task-prof false --run 1 --warmup false --device-whitelist 0 \
    --plugin "$GOLDEN" -o "$OUTDIR/result_smoke.csv"
```

TTK 启动后在用例预解析阶段（TestcasePreParsing）直接崩溃退出（`smoke exit=1`），**一条用例都没执行**，卡上没有任何 kernel/精度输出。报错原文：

```text
2026-09-07 13:08:07,272 [ERROR] [2576998 TestcaseManager TestcasePreParsing] [testcase_manager.py:444]: Failed to process header [attributes] of row [0] with value [{"hasBias":false}]
Traceback (most recent call last):
  File "/tmp/ops-test-kit/ttk/core_modules/testcase_manager/testcase_manager.py", line 440, in _process_over_result
    result[current_header_name] = self.testcase_instance.get_header_func(current_header_name)(value)
  File "/tmp/ops-test-kit/ttk/core_modules/testcase_manager/field_parser.py", line 618, in process_dict
    raise ValueError(f"Invalid value {value} for dict") from None
ValueError: Invalid value {"hasBias":false} for dict
```

外层 traceback 沿 `_parse_testcases → _load_case_from_table → UniversalTestcaseFactory.from_path → _parse_testcase → _process_over_result` 一路到同一个 `ValueError`，即整个 profile 在读表阶段就终止。当时产物状态：合并执行 CSV 刚由合并脚本生成（`attributes` 与上游原始 CSV 逐字一致），preflight 自检通过（preflight 不校验该字段格式），golden/算子部署/环境变量均正常。

## 适用场景

- **工程形态**：TTK aclnn 模式黑盒精度验收（评测执行侧）；TTK 以源码方式使用（`cd $TTK` 后 `python3 -m ttk aclnn ...`）。
- **API 路径**：CSV `attributes`（dict 类）字段 → `ttk/core_modules/testcase_manager/field_parser.py::process_dict`（eval 求值）；入口命令 `python3 -m ttk aclnn -i <CSV> --plugin <golden.py>`。
- **验证过的平台与 CANN 版本**：Ascend 950（Ascend950PR 单卡上板实测）+ CANN 9.0.0-beta.2 + TTK 源码仓（`/tmp/ops-test-kit`）。与 frontmatter `platforms` 一致。
- **适用粒度（transfer_scope）**：`platform_wide`——TTK 主机侧 CSV 字段解析契约，与具体算子无关，任何带 JSON 语法 `attributes` 的 aclnn/kernel 用例 CSV 都会触发。
- **路线优先级**：`recommended_fastpath`——冒烟阶段发现即归一化合并 CSV 再全量（冒烟单条本就是为拦截这类基础设施问题设计的）。
- **明确不适用 / 未验证场景**：无 `attributes` 字段的 CSV 不涉及；`input_data_ranges` 等字段里的裸 `inf`/`nan`/`-0` 写法有专门正则预处理、不受本卡影响；其他芯片平台未验证（本失败与修复均为主机侧 Python 解析行为，与芯片无关——未验证，推断可迁移）；TTK 若未来改为 `json.loads` 解析 dict 字段则本卡失效。

## 根因

TTK 对 dict 类 CSV 字段（`attributes`）的解析函数 `process_dict` 是**正则预处理 + `eval()`**，eval 求值的是 **Python 字面量**，而上游用例生成产出的是 **JSON 序列化**。JSON 与 Python 字面量在布尔/空值上不兼容：JSON 的 `false`/`true`/`null` 在 Python 里不是合法名字，eval 直接抛 `NameError: name 'false' is not defined`，被 `except` 统一转成 `ValueError: Invalid value {"hasBias":false} for dict`。`process_dict` 源码（field_parser.py，verbatim）：

```python
def process_dict(value: str) -> dict:
    """
    dictionary
    :param value:
    :return:
    """
    try:
        # convert -0.0 or -0 to -float(0) for powf & atan operator
        value = re.sub(r"(-0+\.0*|-0+\.?)(?!(\d|\.))", r"-float(0)", value)

        # convert inf to float(inf), nan to float(nan)
        # but skip when nan/inf is inside single/double quotes (dict key)
        def _replace_nan_inf(m):
            start = m.start()
            if start > 0 and value[start - 1] in ("'", '"'):
                return m.group(0)
            return f'float("{m.group(1)}")'

        value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))', _replace_nan_inf, value)
        result = eval(value)
    except Exception:
        raise ValueError(f"Invalid value {value} for dict") from None
    else:
        if isinstance(result, dict):
            return result
        elif isinstance(result, type(None)):
            return result
        else:
            raise TypeError(f"Value {value} is not a dict")
```

可见它只为 `-0`/`inf`/`nan` 做了适配，**没有任何 JSON 布尔/空值转换分支**。TTK 官方示例 CSV 的 `attributes` 全部是 Python 字面量写法，佐证其字段契约就是 Python 字面量而非 JSON：

```csv
--- ./examples/case_store/aclnn/aclnn_cat.csv ---
aclnnCat_float,aclnnCat,"(((3,3),(3,2)),(3,5),)","(('float32','float32'),'float32')",{'dim': -1},"(1,)"
--- ./examples/case_store/aclnn/aclnn_convolution.csv ---
aclnnConvolution_00,aclnnConvolution,"((2, 2, 2, 2),(1, 2, 1, 1), None, (2, 1, 4, 4))","('float16', 'float16', None, 'float16')","('NCHW', 'NCHW', None, 'NCHW')","{'stride': [1, 1], 'padding': [1, 1], 'outputPadding': [0, 0], 'dilation': [1, 1],'transposed': False,'groups': 1,'cubeMathType':0}"
```

行为复核（同机 python3 实测）：

```text
eval JSON fails: name 'false' is not defined
eval Python: {'hasBias': False}
eval Python single: {'hasBias': False}
json parsed: {'hasBias': False}
```

即 `eval('{"hasBias":false}')` 必失败、`eval("{'hasBias':False}")` 与 `json.loads('{"hasBias":false}')` 均成功——格式契约不匹配是确定性触发，不是偶发。

## 解决配方

对**合并执行 CSV**（执行侧自己的产物）的 `attributes` 列做 JSON→Python 字面量归一化；**不改**上游交付的原始用例 CSV，**不改** TTK 源码，**不包装** golden。

before（归一化前，CSV 单元格解析后的值，JSON 语法，TTK 拒收）：

```text
{"hasBias":false}
{"hasBias":true}
```

after（归一化后，Python 字面量，TTK 正常解析）：

```text
{'hasBias': False}
{'hasBias': True}
```

转换规则：`false→False`、`true→True`、`null→None`、双引号键→单引号键（`json.loads` 后 `repr` 一步完成）。完整脚本见 `## 完整代码

> 注：正文判别演示输出为同轨迹未锚定佐证（核心机制 artifact 的角色锚点见 manifest 证据合同）。`，用法 `python3 - <正向合并CSV> <负向合并CSV>`，幂等（已是 Python 字面量的行原样保留）。

## 设计决策

- **为何改合并执行 CSV 而不是改 TTK 源码或上游生成器**：TTK 是源码方式使用的共享工具仓，改动影响面不可控；上游用例 CSV 是别的阶段的交付产物，回写会污染交付物并破坏复跑一致性；合并执行 CSV 是本次执行自己生成的产物，改动边界最小、可随产物重建而重放。
- **为何 `ast.literal_eval` 先行、`json.loads→repr` 兜底的双轨设计**：保证幂等与向后兼容——已经是 Python 字面量的行不被重写；两种解析都失败的行原样保留、交给 TTK 报错（不吞掉真正的脏数据）。
- **被否决的备选**：在 golden.py 侧包装/代写属性处理——验收任务明令 golden 直接 `--plugin` 指向、不包装不代写；sed/正则手工替换布尔字面量——对嵌套结构不稳健，`json.loads→repr` 是结构化转换。
- **路线档位**：`recommended_fastpath`——冒烟阶段（`-t` 单条）先确认基础设施 OK 再全量，正是拦截此类格式/环境问题的设计意图；本修复不触碰用例语义（`hasBias` 取值逐字保留，仅表示法转换）。

## 系统信息

- **toolchain**：CANN cann-9.0.0-beta.2（`ASCEND_TOOLKIT_HOME=/home/<user>/Ascend/cann-9.0.0-beta.2`）；TTK 源码方式使用（`/tmp/ops-test-kit`，`cd $TTK && python3 -m ttk aclnn`）；python3；环境变量来自算子仓 `docs/<OP>/develop/env.sh`（提供 `$PY`/`$TTK`/`ASCEND_*`）。
- **目标平台**：Ascend950PR 单卡（device 0；TTK 日志 `Device Platform: Ascend950PR_9579`）。
- **目录布局**：`operators/<OP>_package/tests/blackbox/testcases/`（上游 L0/L1/L2 用例 CSV，只读）、`operators/<OP>_package/tests/blackbox/ttk/`（合并执行 CSV 与结果产物）。
- **相关文件清单**（与本次读写一一对应）：
  - `tests/blackbox/testcases/aclnn*_{l0_functional,l1_functional,l2_exception}.csv`——上游产物，只读（`attributes` 为 JSON，未改动）；
  - `tests/blackbox/ttk/blackbox_all.csv`（正向合并 559 条）、`tests/blackbox/ttk/blackbox_l2_neg.csv`（负向合并 49 条）——归一化作用对象；
  - `tests/blackbox/ttk/result_smoke.csv`、`result.csv`、`result_l2.csv`、`smoke_debug.log`——执行产物；
  - `docs/<OP>/develop/golden.py`（TestSpec，`--plugin` 直指）、`docs/<OP>/develop/env.sh`；
  - TTK 侧：`ttk/core_modules/testcase_manager/field_parser.py`（`process_dict`，约 L597-622）、`ttk/core_modules/testcase_manager/testcase_manager.py`（`_process_over_result` L440、预解析报错点 L444）。

## 完整代码

归一化脚本（verbatim，曾对 559+49 行实测执行）：

```python
import ast, csv, json, sys

def norm_attr(v):
    v = (v or '').strip()
    if not v:
        return v
    # already Python literal?
    try:
        ast.literal_eval(v)
        return v
    except (ValueError, SyntaxError):
        pass
    # try JSON
    try:
        d = json.loads(v)
        return repr(d)
    except (ValueError, json.JSONDecodeError):
        pass
    return v

for path in sys.argv[1:]:
    rows = []
    with open(path, newline='', encoding='utf-8') as h:
        reader = csv.DictReader(h)
        fieldnames = reader.fieldnames
        changed = 0
        for row in reader:
            old = row.get('attributes', '')
            new = norm_attr(old)
            if new != old:
                row['attributes'] = new
                changed += 1
            rows.append(row)
    with open(path, 'w', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: normalized {changed} attributes rows")
```

调用方式（对两张合并执行 CSV）：

```bash
python3 - "$MERGED" "$NEG" <<'EOF'
<上述脚本>
EOF
```

## 预防

- 拿到上游生成的黑盒用例 CSV 后、全量执行前，先看一眼 `attributes` 列的写法：`{"k":false}`（JSON，双引号键、小写布尔）还是 `{'k': False}`（Python 字面量，单引号键、大写布尔）——前者必挂。
- 上游用例生成器若输出 JSON 序列化，就在合并/执行链路统一挂一道归一化（幂等脚本见 `## 完整代码`），不要依赖逐次手工救火。
- 冒烟阶段（`-t` 单条 + debug 日志）就是为了拦截这类基础设施问题：冒烟不过先分诊格式/环境/部署，**不要**直接进 kernel/tiling 修复循环。
- 报错栈里先看是哪个 header、哪个 parser 挂了（本例 `header [attributes]` → `field_parser.py ... in process_dict`），再决定排查方向。

## 验证信号

以下信号来自一次 aclnn 自定义算子黑盒精度验收会话：冒烟首跑在用例预解析阶段失败，定位为 CSV `attributes` 的 JSON/Python 字面量契约不匹配，对合并执行 CSV 归一化后重跑冒烟与全量。

- 修复前冒烟：TTK 在 TestcasePreParsing 报 `Failed to process header [attributes] of row [0] with value [{"hasBias":false}]`，traceback 终止于 `field_parser.py line 618 in process_dict → ValueError: Invalid value {"hasBias":false} for dict`，进程 exit=1、零用例执行（来源平台：950，TTK 主机侧预解析；证据等级：tool_observed）。
- eval 行为复核：`eval('{"hasBias":false}')` → `NameError: name 'false' is not defined`；`eval("{'hasBias':False}")` → `{'hasBias': False}`；`json.loads('{"hasBias":false}')` → `{'hasBias': False}`（来源平台：950 验证机主机侧 Python；证据等级：tool_observed）。
- 归一化执行结果：正向合并 CSV `normalized 559 attributes rows`、负向合并 CSV `normalized 49 attributes rows`，归一化后全表 `attributes` 取值仅 `{'hasBias': False}` / `{'hasBias': True}`（来源平台：950 验证机主机侧；证据等级：tool_observed）。
- 修复后冒烟：preflight PASS；TTK 单条冒烟 `total=1 pass=1 fail=0 skip=0`（SMOKE-GATE: PASS）；check_op_binpath 确认 kernel bin 来自 `opp/vendors/<OP>/op_impl/ai_core/tbe/kernel/ascend950/...`，跑的确实是自定义算子而非内置（来源平台：950；证据等级：tool_observed）。
- 修复后全量：正向 559 条全部完成执行（TTK Test Summary：Total 559 / PASS 557 / FAIL 2 / SKIP 0，PassRate 99.64%——剩余 2 条失败为该算子 bias 输出精度问题，与本格式修复无关）；负向 49 条全部执行并按预期被拒（NEGATIVE-GATE: PASS）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖 TTK 以 `eval` 解析的 dict 类 CSV 字段（`attributes`）；裸 `inf`/`nan`/`-0` 写法已有专门正则预处理，不受影响。
- 上游生成器若已改输出 Python 字面量，则无需再归一化；TTK 若未来给 `process_dict` 加 JSON 回退（`json.loads` 优先），本卡失效。
- 归一化只应作用于**执行侧合并产物**；回写上游交付 CSV 是坏实践（见下）。
- 仅在 Ascend 950（Ascend950PR）+ CANN 9.0.0-beta.2 + 源码 TTK 环境实测；其他芯片平台未验证——本失败与修复均为主机侧 Python 解析行为、与芯片平台无关（未验证，推断可迁移）。

## 坏实践

- 看到冒烟零用例执行就按"算子未部署 / golden 报错 / api_name 不符"盲查——那是另外几类基础设施失败的表象；先读报错栈定位挂掉的 header 与 parser，本例一眼可见是 `attributes` 字段格式问题。
- 为过执行把 `attributes` 单元格删掉或置空——会丢掉用例语义（如 `hasBias` 分支选择），等于偷改用例。
- 回写修改上游交付的原始用例 CSV——污染他人产物、破坏复跑一致性；只改自己会话生成的执行产物。

## 误诊

- 本会话未发生实际误诊。记录一个易混点：同一次冒烟输出里 `Input testcase file is not a valid zip file, switch to normal csv mode...` 是 INFO 级的 zip→csv 双格式探测回退，属正常日志，勿与后面的 ValueError 混为一谈（来源平台：950；证据等级：tool_observed）。



## 后续增量（2026-09-17，并行策展批：生成源定位 ascendc-st-design + ast.literal_eval 坏实践 + 第二轨迹全量信号）

同形逐元素 AdamW 优化器算子（4 输入张量 + 7 个标量属性，含布尔属性 maximize）的黑盒 TTK 用例生成轨迹（Ascend 950 环境，CANN 9.2.0 安装 + TTK ops-test-kit 源码仓）在同一契约上补四点：

**1. 生成源定位与"重生成必须重跑"边界**：本轨迹的 JSON 风格 attributes 来自 ascendc-st-design 技能 `generate_test_cases.py` 的 `format_ttk_dict`（用 `json.dumps` 序列化属性字典，双引号键 + 小写 `true`/`false`）。产物侧归一化（`repr(json.loads(cell))`，与原卡 json.loads→repr 同型、幂等——已是 Python 字面量的单元格 `json.loads` 失败被跳过）**每次重新生成用例后必须重跑**：生成器会再次输出 JSON 格式（本轨迹在重生成 L1 后重新归一化了全部新单元格）。

**2. 判别信号（int 形可解析）**：`{"lr": 0.001, "maximize": 0}`（int 0）可被 `process_dict` 正常解析——失败只出现在小写布尔字面量 `true`/`false`，可据此快速分诊是格式问题还是脏数据。

**3. 相邻坏实践（同契约的字段族）**：TTK CSV 各字段普遍是 Python 表达式语义——`input_data_ranges` 里的裸 `nan`/`inf`/`+0`/`-0` token 是 Python 名而非字面量，用 `ast.literal_eval` 解析必报 `ast.Name` 类错误（验收侧聚合脚本踩过）；解析这类字段要么复用 TTK 自己的 field_parser（内部 eval + 正则替换），要么用受控 `eval`，不要用 `ast.literal_eval`。

**4. 归一化手段取舍**：用 `repr(json.loads(cell))` 而不是字符串替换 `false→False`——`json.loads` 保证只接受合法 JSON 且布尔保型（`type(r['maximize']) == bool` 已验证）；裸字符串替换在键名/字符串值含 `false` 子串时会误伤。

配套探针（不依赖完整 TTK 执行，直接用 TTK 解析器验证格式，可纳入固定验收步骤）：

```python
import sys
sys.path.insert(0, "<ops-test-kit 路径>")
from ttk.core_modules.testcase_manager.field_parser import process_dict
samples = ['{"lr": 0.001, "maximize": true}', "{'lr': 0.001, 'maximize': True}"]
for s in samples:
    try:
        process_dict(s); print("OK  ->", s)
    except Exception as e:
        print("FAIL->", type(e).__name__, e)
```

### 复现增量验证信号

- 修复前探针：3 个样本中 2 个含 JSON 小写布尔的全部 `ValueError: Invalid value ... for dict`，int 形 `{"lr": 0.001, "maximize": 0}` 解析通过（来源平台：950；证据等级：tool_observed）。
- 归一化验证：`repr(json.loads(orig))` 后同一单元格解析 OK，`maximize` 类型为 `bool`（来源平台：950；证据等级：tool_observed）。
- 全量归一化后：L0 21/21、L0 超规格 1/1、L1 567/567、L1 超规格 53/53 attributes 单元格经 TTK `process_dict` 解析通过，TOTAL 642/642（来源平台：950；证据等级：tool_observed）。
- 最终校验：`ttk aclnn --validate` L0 20/20 valid、L1 586/586 valid、L2 5/5 valid（"All testcases passed validation"）（来源平台：950；证据等级：tool_observed）。
