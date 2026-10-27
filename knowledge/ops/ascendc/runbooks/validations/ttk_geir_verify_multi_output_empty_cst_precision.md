---
type: "Runbook"
title: GEIR 多输出精度串无法过 verify 严格 PASS 判定
description: 多输出+空输出算子 cst_precision 为逐输出逗号串（如 PASS,PASS,PASS,100%,100%），verify 要求严格 PASS；TestSpec compare 钩子返回聚合单结果并重生成冻结插件后通过。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GEIR 多输出精度串无法过 verify 严格 PASS 判定


## 现象

一个 5 输出自定义算子（3 路常规输出 + 2 路由 `has_bias` 开关的 bias 输出，`has_bias=false` 时为 `(0,)` 空张量）在 TTK GEIR 图模式下跑完正向源码静态套件：44 个用例**全部真实执行、逐输出精度比对全部通过**（结果 CSV 中 `precision_status=PASS`），但验证脚本 `verify_ttk_geir.py check-result` 直接抛异常终止：

```
ValueError: GEIR source enabled modes are not PASS for aclnnThnnFusedGruCellGrad_L0_001: {'cst_precision': 'PASS,PASS,PASS,100%,100%'}
```

即：用例实际全对，卡在**结果格式**——`cst_precision` 列的值是 5 个输出的逗号拼接串 `PASS,PASS,PASS,100%,100%`（3 路门梯度输出 `PASS` + 2 路空输出 `100%`），而 verify 的 `_check_enabled_modes` 对该列做严格相等判定 `any(value != "PASS" for value in values)`。当时代码状态：判读走的是 TTK **默认比较器**，因为 GEIR 模式的 `--plugin` 指向的 `tests/geir/geir_spec.py` 是一份**冻结快照**，生成于 `docs/<OP>/develop/golden.py` 增加 `compare` 判读钩子（早前 aclnn 黑盒阶段为 bias 输出 NaN/Inf 处理所加）之前——快照里没有 `compare` 方法。

## 适用场景

- 工程形态：AscendC 算子仓开发中的 GEIR 图模式验证（TTK `geir` 六套件 + `verify_ttk_geir.py` 结果门禁）；多输出（尤其含空输出/可选输出）算子。
- API 路径：TTK TestSpec 判读钩子契约——`compare(*outputs, *goldens, compare_context=None, **kwargs)`；TTK 比较链 `compare_with_hooks`/`try_custom_compare`/`ComparisonBase.compare`；冻结机制 `verify_ttk_geir.py freeze`/`check-frozen`（FROZEN_FILES 含 `geir_spec.py`）。
- 验证过的平台与 CANN 版本：Ascend950 + CANN 9.0.0-beta.2（GEIR 套件上板执行；比较器行为属 TTK 工具链，平台无关）。
- 适用粒度：`operator_family`（同工作流下任何多输出自定义算子）。
- 路线优先级：**recommended_fastpath**。
- 明确不适用/未验证：单输出算子无此问题（默认比较器即返回单个 `PASS`）；TTK 版本号未在轨迹内读取，比较链行为以当轮安装为准。

## 根因

TTK 默认比较器把逐输出精度用逗号拼接写入 `cst_precision`，空输出的精度是数值 `1` 而被格式化为 `"100%"`，于是任何"多输出 + 空输出"算子的 `cst_precision` 都不可能等于严格的 `"PASS"`；而 GEIR 模式本应用的自定义 `compare` 钩子（可控制返回格式）因 `geir_spec.py` 冻结快照滞后于 `golden.py` 而缺失，比较回退到默认比较器。

机制链（逐条对应 TTK 源码实测）：

1. **逐输出拼接**：`compare_with_hooks` 走默认 `compare` 时返回 `",".join(total_precision)`——N 个输出就是 N 段逗号串。
2. **空输出 → "100%"**：`ComparisonBase.compare` 里 `if isinstance(compare_result.precision, str): precision = compare_result.precision else: precision = f"{compare_result.precision * 100}%"`；而空输出在 `_check_empty` 短路返回 `EachCompareResult(1, is_pass=True)`——`precision=1`（**数值**）→ `f"{1*100}%"` = `"100%"`。所以 `has_bias=false` 的两路 `(0,)` bias 输出贡献两个 `"100%"`。
3. **verify 严格判定**：`_check_enabled_modes` 对 `cst_precision` 列 `strip().upper()` 后要求**整个值**等于 `"PASS"`；`"PASS,PASS,PASS,100%,100%" != "PASS"` → ValueError。
4. **自定义钩子缺位**：GEIR 运行命令 `--plugin $GEIR/geir_spec.py`；TTK 用 `get_spec_attr(op_name, "compare", plugin_path)` 从该文件加载判读钩子，加载不到就回退默认比较器。`geir_spec.py` 是 `golden.py` 的生成别名（生成脚本 `build_geir_plugin.py` 的自述：Create a GEIR TestSpec alias without modifying the public golden file），且其哈希被 `baseline.json` 冻结——`golden.py` 后加的 `compare` 钩子不会自动出现在旧快照里。`golden.py` 自带的 `_stat_judge` 对空输出返回**字符串** `"PASS"`（`actual.size==0 and golden.size==0 → {"pass": True, "precision": "PASS"}`），但即便钩子在场，返回 list 时 TTK 仍会逐输出拼接成 `"PASS,PASS,PASS,PASS,PASS"`——还是过不了严格判定。因此修复必须同时做两件事：**钩子在场** + **钩子返回单聚合结果**。

## 判别方法

- 看结果 CSV：`precision_status=PASS` 而 `cst_precision` 含逗号或 `%`（如 `PASS,PASS,PASS,100%,100%`）→ 格式问题，不是精度问题（别去调 kernel/精度阈值）。
- 空输出对应段为 `100%`、常规输出段为 `PASS`/`FAIL` 的组合，可反推哪些输出是空张量。
- 检查 GEIR 插件快照是否滞后：对比 `tests/geir/geir_spec.py` 与 `docs/<OP>/develop/golden.py` 的 TestSpec 类（有无 `compare` 方法）。

## 设计决策

- **聚合而不放判定**：钩子聚合为单 dict `{"pass": all_pass, "precision": "PASS"/"FAIL", "error_info": "; ".join(errors)}`，其中 `all_pass = all(r.get("pass") for r in results)`——逐输出的通过语义原样保留（all 语义），只是把 N 段报告折叠成一段；不触碰 mere/mare 阈值、不改 golden 计算语义。
- **代价**：聚合后 CSV 里看不到逐输出精度分段；失败时靠 `error_info` 拼接保留定位信息（`Output 0: ...; Output 1: ...`）。
- **快照同步而非改冻结机制**：`geir_spec.py` 定义上就是 `golden.py` 的生成别名，`golden.py` 更新判读钩子后重生成别名并用 `verify_ttk_geir.py freeze` 重新冻结 baseline 是该工作流的既定通道（freeze 命令即为此存在）；不是绕过冻结——`check-frozen` 与 projection-check 在重冻结后照常通过（44/49/0）。
- **双 `__spec__` 的坑**：`golden.py` 内含两个 `__spec__` 条目（kernel 模式与 aclnn 模式各一），官方生成脚本 `build_geir_plugin.py` 按"恰好一个 `__spec__`"实现，直接跑会 Traceback；用内联 AST 脚本替换 `__spec__` 为 GEIR op_name 映射完成重生成。

## 系统信息

- 工具链：TTK GEIR runner（`run_ttk_geir.sh` → `python -m ttk geir -i <cases> --plugin $GEIR/geir_spec.py -c=true -d=false -b=false ...`）+ `verify_ttk_geir.py`（check-result/freeze/check-frozen）；CANN 9.0.0-beta.2；Ascend950。
- 相关文件（与事件流实际读写一一对应）：
  - `docs/<OP>/develop/golden.py`——判读钩子 `compare`（本卡修改点：返回值聚合）
  - `operators/<OP>_package/tests/geir/geir_spec.py`——冻结的 GEIR TestSpec 别名（重生成）
  - `operators/<OP>_package/tests/geir/baseline.json`——冻结哈希（重新 freeze）
  - `.opencode/skills/geir-route-tester/scripts/build_geir_plugin.py`——官方别名生成脚本（双 `__spec__` 时失败）
  - TTK 比较链（只读参照）：`core_modules/comparison/custom.py`、`comparison.py`、`registry.py`、`stat_rel_err.py`；`core_modules/geir/profiling.py`

## 完整代码

**before**（`docs/<OP>/develop/golden.py` 的 `compare` 钩子返回逐输出 list——TTK 逐段拼接）：

```python
        except Exception:  # 审计日志失败不影响判定
            pass
        return results
```

（上文 `results` 为 5 个 `_stat_judge(...)` 结果 dict 的列表，每个形如 `{"pass": bool, "precision": "PASS"/"FAIL", "error_info": ...}`。）

**after**（聚合为单 dict 返回，判定语义不变）：

```python
        except Exception:  # 审计日志失败不影响判定
            pass
        # 聚合为单结果（多输出算子：TTK try_custom_compare 会对 list 逐输出
        # 拼 precision 字符串，导致 cst_precision="PASS,PASS,…"，无法通过 GEIR
        # verify 的严格 "PASS" 判定。此处聚合为单 dict 返回，保持判定语义不变。）
        all_pass = all(r.get("pass", False) for r in results)
        errors = [r["error_info"] for r in results if r.get("error_info")]
        return {
            "pass": all_pass,
            "precision": "PASS" if all_pass else "FAIL",
            "error_info": "; ".join(errors) if errors else None,
        }
```

**重生成 `geir_spec.py` 并重新冻结**（`golden.py` 含双 `__spec__` 条目时 `build_geir_plugin.py` 会 Traceback，用 AST 替换绕过；单条目时直接用官方脚本）：

```bash
# 官方通道（golden.py 只有一个 __spec__ 条目时）：
python3 .opencode/skills/geir-route-tester/scripts/build_geir_plugin.py \
  --golden docs/<OP>/develop/golden.py --out operators/<OP>_package/tests/geir/geir_spec.py

## 双 __spec__ 时的等价替换（内联 AST 脚本，verbatim 自联调过程，路径按需替换）：
python3 -c "
import ast
from pathlib import Path

source = Path('docs/<OP>/develop/golden.py').read_text('utf-8')
tree = ast.parse(source)

## Find __spec__ assignment
for i, node in enumerate(tree.body):
    target = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        continue
    if isinstance(target, ast.Name) and target.id == '__spec__':
        break

## Replace __spec__ with GEIR op_name
replacement = \"__spec__ = {'<snake_case_op_name>': '<OpName>TestSpec'}\n\"
lines = source.splitlines(keepends=True)
lines[node.lineno - 1:node.end_lineno] = [replacement]
result = ''.join(lines)

## Verify it parses
ast.parse(result)

## Write
out = Path('operators/<OP>_package/tests/geir/geir_spec.py')
out.write_text(result, 'utf-8')
print('OK: regenerated geir_spec.py')
"

## 重新冻结并自检：
python3 .opencode/skills/geir-route-tester/scripts/verify_ttk_geir.py freeze \
  --directory operators/<OP>_package/tests/geir
python3 .opencode/skills/geir-route-tester/scripts/verify_ttk_geir.py check-frozen \
  --directory operators/<OP>_package/tests/geir
python3 skills/geir-route-tester/scripts/project_blackbox_cases.py check \
  --manifest operators/<OP>_package/tests/geir/projection.json
```

## 预防

- 多输出算子上 GEIR 验证前，先确认 TestSpec 插件里 `compare` 钩子**返回单聚合 dict**（`pass`/`precision`/`error_info`），不要返回逐输出 list。
- `golden.py` 每次变更（尤其加/改判读钩子）后，**立即重生成 `geir_spec.py` 并 re-freeze**——冻结的是生成别名的哈希，别名不会自己跟上源文件；三条自检（geir-freeze / check-frozen / projection-check）跑完再进六套件。
- 结果 CSV 出现 `PASS,...`/`100%` 形态的 `cst_precision` 时，先判格式问题再查精度——`precision_status=PASS` 而 verify 报错就是典型信号。

## 验证信号

以下信号来自该算子 GEIR 图模式联调的连续两轮验证运行（同一隔离 staging 机制、TTK GEIR 正向源码静态套件）：修复前一轮复现格式报错，修复（钩子聚合 + 别名重生成 + 重冻结）后一轮确认格式报错消失。

- 修复前：44 用例执行完毕且逐输出精度全过（`precision='PASS,PASS,PASS,100%,100%'`、`precision_status='PASS'`），verify 抛 `ValueError: GEIR source enabled modes are not PASS for ...L0_001: {'cst_precision': 'PASS,PASS,PASS,100%,100%'}`（来源平台：Ascend950；证据等级：tool_observed）
- 重生成+重冻结自检：`PASS geir-freeze files=5`、`PASS projection-check positive=44 negative=49 not_applicable=0`（来源平台：Ascend950；证据等级：tool_observed）
- 修复后：verify 不再报格式错误——同一套件首轮失败推进到第 6 个用例的真实精度 FAIL（`cst_precision='FAIL'`，错误信息 `Output 0: NaN/Inf mismatch; NaN/Inf mismatch; NaN/Inf mismatch`），即前 5 个用例 `cst_precision` 已严格 `PASS`；再后一轮推进到第 20 个用例的真实精度 FAIL，前 19 个用例严格 `PASS`（来源平台：Ascend950；证据等级：tool_observed）
- 机制源码证据（只读）：`ComparisonBase.compare` 的数值精度百分比格式化、`_check_empty` 对双空输出返回数值 `precision=1`、`compare_with_hooks` 的 `",".join(precisions)`（来源平台：Ascend950（TTK 工具链源码，行为与平台无关）；证据等级：tool_observed，源码读取）

## 失效边界

- 仅对"多输出 + 判读走 TestSpec 插件"的 TTK GEIR 工作流实测（5 输出、含 `(0,)` 空输出）；单输出算子默认比较器即返回单个 `PASS`，无需本卡。
- 空输出 `"100%"` 的格式化细节属 TTK 比较器实现，TTK 版本未在轨迹内读取——升级 TTK 后需复核该行为。
- 聚合后 `cst_precision` 不再保留逐输出分段（失败定位依赖 `error_info` 拼接）；若工作流后续要求逐输出精度分段落盘，需改 verify 判定而不是钩子聚合。
- 本卡只解决**结果表示格式**；修后暴露的 NaN/Inf 判定与数值精度失败是独立问题，不在本卡范围。

## 坏实践

- 见到 verify 报 "not PASS" 就去调 kernel 或放宽容差——先看 `cst_precision` 的值形态：`PASS,PASS,...,100%` 是格式问题，用例本身已全过。
- `golden.py` 加了判读钩子却不动 `geir_spec.py`——冻结别名继续用旧快照，钩子形同虚设（GEIR 模式根本加载不到）。
- 为过严格判定去改 verify 脚本或 tolerance——验证脚本是门禁资产，判读格式的正确收敛点是算子侧 TestSpec 钩子。
- **明确禁入**：为了压掉后续暴露的含 Inf/NaN 输入用例失败，把比对钩子的 NaN/Inf 位置差异判定对所有输出整体放宽（如 `_stat_judge(..., is_bias_reduce=True)` 一律传 true）——那是测试侧放松，掩盖真实差异模式，不属于本卡配方；本轨迹中该放宽未做任何机制验证（无逐位置 NaN/Inf mask 对比、无 finite 值单独比对），其安全性未证明。

## 误诊

- 一度怀疑是 TTK/verify 的 bug 且"不可修改（属技能框架）"而准备放弃——实际收敛点在算子侧：钩子返回格式 + 冻结别名同步，两处都是算子仓自己的文件。
- 一度以为只要钩子在場（list 返回）即可——`try_custom_compare` 对 list 结果同样逐输出拼接（`PASS,PASS,PASS,PASS,PASS`），仍过不了严格判定；必须聚合。


