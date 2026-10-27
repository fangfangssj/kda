---
type: "Runbook"
title: 白盒精度失败为过期产物：TTK compare 钩子生效判别
description: golden.py 已实现修订容差 compare 钩子但 result.csv 仍报精度失败——结果是旧 golden 生成的过期产物；四判别法确认钩子是否实调。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 白盒精度失败为过期产物：TTK compare 钩子生效判别


## 现象

在 ThnnFusedGruCellGrad 算子（AscendC ops-registry-invoke 算子仓流程）Branch-0 白盒精度验证中：golden.py 的 TestSpec **已经实现**了修订容差的 `compare` 比对钩子（bias 输出噪声预算），但白盒 `result.csv` 仍报多条 bias mare 精度失败（release 套件 case00034/55：`threshold exceeded: mare(1.24e-03>=0.001220703125)`），看起来像容差修订没有生效。当时代码状态：golden.py 的 compare 钩子在前一开发会话已写入工作区文件（后提交为 commit c28d6d2a），而 result.csv / result_oom.csv 是更早时间生成的旧产物。

排查中还有两个迷惑信号：TTK 日志的 `Command:` 行只显示 `ttk op -i .../whitebox_all.csv`，看不到 `--plugin` 旗标——一度被误判为「--plugin 没传、钩子没加载」；而 TTK 框架源码（`comparison/custom.py`）显示钩子机制本身要求 `custom_compare is not None` 才会走自定义路径，否则静默回落默认 `stat_rel_err` 比对器。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发 · TTK kernel 模式白盒验证（`--plugin` 挂 TestSpec golden.py；aclnn/e2e 通路共用同一钩子入口）。
- **API 路径**：`python3 -m ttk kernel -i <cases.csv> ... --plugin <golden.py> -o <result.csv>`；TTK `comparison/custom.py` 的 `compare_with_hooks` / `try_custom_compare` / `_compare_context`；TestSpec 加载器 `test_spec` 的 `get_spec_attr(op_name, "compare", plugin_path)`。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册平台 `950`；`TTK_FULL_SOC_VERSION=Ascend950PR_9579`，Mode `ASCEND_ONBOARD` 实机）+ CANN `cann-9.0.0-beta.2`——与 frontmatter `platforms` 严格一致。
- **适用粒度**：`platform_wide`——TTK 验证工具链上的比对钩子诊断，与具体算子无关。
- **路线优先级**：`recommended_fastpath`（遇到「实现已改但结果没变」的精度失败时，先按本判别法确认钩子实调状态，再动 kernel 或容差）。
- **明确不适用 / 未验证场景**：aclnn / e2e 通路未实测（钩子入口同源，见 ## 失效边界）；非 TTK 的比对框架不适用。

## 判别方法

**根因（单句）**：result.csv / result_oom.csv 是 golden.py 尚无 `compare` 方法时生成的过期产物——原跑 `--plugin` 已传入（golden 函数被找到、期望值正常算出），但当时 TestSpec 类没有 `compare` 方法，TTK 比回落默认 `stat_rel_err`（mare 预算 = 10×th），故 bias mare 失败；须用新 golden.py 重跑。

**四判别法 + 两探针**（按排查顺序）：

1. **result.csv `precision_metrics` 键集指纹**：默认 `stat_rel_err` 比对器产出的指标为 `{mere, mare, threshold, pass, reason}`（reason 形如 `threshold exceeded: mare(...)`）；TestSpec 自定义 `compare` 钩子产出的判定为 `{pass, precision, error_info}`。失败行若是前者格式，说明该结果由默认比对器判定。
2. **`Command:` 日志行不可作插件判据**：TTK `bridge.py:392` 的日志只记 `ttk {test_mode} -i {input_file}`，**不含 `--plugin` 等其余旗标**——不能据此断言插件未传（本 trace 的误诊源）。
3. **正判据 = OnGenInput 阶段的插件管理器行**：full_run.log 中 `CustomPluginManager initialized with plugin_path: (PosixPath('.../develop/golden.py'),)` 出现即证明 `--plugin` 已生效、golden.py 已被加载。
4. **时间戳 / git 核对**：result.csv 生成时间（文件时间 16:22）vs golden.py 修改时间 / git 提交时间（17:46 修改、21:19 提交 c28d6d2a）——结果早于实现则必为过期产物，须重跑。命令：`git log --oneline -- <golden.py>` + `git diff <旧commit>..<新commit> -- <golden.py>`（diff 显示 +248 行插入即 compare 钩子加入）。
5. **重跑后确认钩子实调**：OnComparison 阶段出现钩子内审计日志行（如 `[golden.py:406] [ThnnFusedGruCellGrad][case00049] compare: bias mare noise ...`）= 自定义钩子确实被调用。
6. **离线探针（不跑板，验证钩子可加载）**：

```bash
# ① 纯 import 检查：TestSpec 类上 compare/golden 方法存在
python3 -c "
import sys
sys.path.insert(0, '<golden.py 所在目录>')
import golden
cls = golden.ThnnFusedGruCellGradTestSpec
print('has compare:', hasattr(cls, 'compare'))
print('has golden:', hasattr(cls, 'golden'))
print('__spec__:', golden.__spec__)
"
## ② TTK 加载器检查：按 CSV 的 snake_case op_name 走真实解析路径
python3 -c "
from ttk.test_spec import get_spec_attr
golden_path = '<...>/develop/golden.py'
golden_fn = get_spec_attr('thnn_fused_gru_cell_grad', 'golden', golden_path)
compare_fn = get_spec_attr('thnn_fused_gru_cell_grad', 'compare', golden_path)
print('golden_fn:', golden_fn)
print('compare_fn:', compare_fn)
"
```

探针②要点：CSV 的 `op_name` 是 snake_case（`thnn_fused_gru_cell_grad`），而 golden.py `__spec__` 注册键是 PascalCase（`ThnnFusedGruCellGrad`）——直接查表不命中，加载器走**命名约定回退**（snake→Pascal + `TestSpec`）找到类，`golden`/`compare` 均可解析（本 trace 实测返回函数对象）。

**TTK 钩子机制背景**（`/tmp/ops-test-kit/ttk/core_modules/comparison/custom.py`，路径参数化、判读逻辑逐字 关键路径）：`compare_with_hooks` 在 `hooks_enabled = testcase is not None and (pre_compare is not None or custom_compare is not None)` 且输出已落地为真实数组时，先 `try_custom_compare`——钩子返回非 None 即采信其判定并**跳过标准判据**；返回 None 表示不接管、回落默认 compare。钩子未找到（`get_spec_attr` 返回 None）时静默走默认 `stat_rel_err`，**无任何报错或告警**——这是「实现了钩子但结果看起来没变」却不易察觉的机制根源。

## 设计决策

- **判别顺序为何把 precision_metrics 键集放第一**：它是产物级指纹，不需要 git/日志即可从 result.csv 本身判断该行由哪个比对器产出，最快排除「钩子没生效」。
- **为何强调 Command: 行的误导性**：本 trace 实际排查路径先误判「--plugin 未传」（差点往重跑命令加参数了事），后在 full_run.log 第 53 行找到 `CustomPluginManager` 证据才纠正——把该误诊路径记入 ## 误诊，防止复踩。
- **当前方案属哪档路线**：`recommended_fastpath`（诊断路径本身无成本权衡）。

## 系统信息

- **toolchain**：CANN `cann-9.0.0-beta.2`（`develop/env.sh`）；TTK `/tmp/ops-test-kit`（`python3 -m ttk kernel ...`）；`npu-smi 25.7.rc1.6`。
- **目标平台**：Ascend950PR（`TTK_FULL_SOC_VERSION=Ascend950PR_9579`，Mode `ASCEND_ONBOARD`）。
- **相关文件**：
  - `docs/ThnnFusedGruCellGrad/develop/golden.py`——TestSpec（golden / compare / tolerance / oob_exemptions；commit c28d6d2a 加入 compare 钩子）；
  - `operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/`——`whitebox_all.csv`（33 条）、`result.csv` / `result_oom.csv`、`full_run.log` / `oom_debug.log` / `smoke_run.log`；
  - `/tmp/ops-test-kit/ttk/core_modules/comparison/custom.py`——钩子公共入口；
  - `/tmp/ops-test-kit/ttk/cli/bridge.py`——`--plugin` 应用（`_apply_plugin_rerun_args`）与 Command 日志行（line 392）。

## 完整代码

重跑命令（本 trace 实测使 compare 钩子生效的完整命令，路径参数化、判读逻辑逐字）——release 精度套件：

```bash
source docs/ThnnFusedGruCellGrad/develop/env.sh 2>/dev/null

PKG=<算子仓>/operators/ThnnFusedGruCellGrad_package
GOLDEN=<算子仓>/docs/ThnnFusedGruCellGrad/develop/golden.py
SOC=Ascend950PR_9579
OUTDIR=$PKG/tests/whitebox/ttk
MERGED=$OUTDIR/whitebox_all.csv
VENDOR="$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad"

cd "$TTK"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=0

env PYTHONPATH="$TTK:$(dirname "$GOLDEN"):$ASCEND_TOOLKIT_HOME/python/site-packages" \
    TTK_FULL_SOC_VERSION="$SOC" \
    "$PY" -m ttk kernel -i "$MERGED" -d false -b release --pc=1 --task-prof false --run 1 --warmup false \
    --plugin "$GOLDEN" -o "$OUTDIR/result.csv" 2>&1 | tee "$OUTDIR/full_run.log" | tail -20
```

OOM 插桩套件（差异：`-d true -c false -b false --compile-opts op_debug_config=oom`，PYTHONPATH 追加 `$VENDOR/op_impl/ai_core/tbe`，结果写 result_oom.csv / oom_debug.log，结束打印 `TTK-OOM-EXIT status=$?`）。

单用例冒烟（钩子实调确认）：同命令加 `--testcase=case00049`，grep `compare: bias|PRECISION_STATUS`。

## 预防

- 改 golden.py / TestSpec 钩子后，**必须重跑**两套白盒再下结论——result.csv 不会因源文件更新而失效或重生成。
- 重跑后第一眼查 OnComparison 阶段的钩子审计日志行（钩子内 `logging.info`），确认钩子实调而非静默回落。
- 不要凭 `Command:` 日志行判断旗标——它不记完整命令行。

## 验证信号

本卡信号来自该算子 Branch-0 白盒修复任务：对「钩子已实现但 result.csv 仍报失败」现象做溯源诊断，随后用带 `--plugin` 的 TTK kernel 模式命令重跑两套全量白盒（Ascend950PR 实机）。

- 离线探针②输出：`golden_fn: <function ThnnFusedGruCellGradTestSpec.golden ...>`、`compare_fn: <function ThnnFusedGruCellGradTestSpec.compare ...>`——CSV snake_case op_name 经命名约定回退可解析钩子（来源平台：950；证据等级：tool_observed）。
- 原跑 full_run.log 第 53 行（OnGenInput 阶段）：`CustomPluginManager initialized with plugin_path: (PosixPath('.../develop/golden.py'),)`——原跑 `--plugin` 已传入（来源平台：950；证据等级：tool_observed）。
- git diff `45caf3c5..c28d6d2a -- golden.py`：+248 行插入即 compare 钩子（`_STAT_TH` 阈值表镜像、`_mare_noise`、`compare` 方法等）——旧版本无 compare 方法（来源平台：950；证据等级：artifact_verified）。
- 重跑后冒烟 case00042 / case00049 的 OnComparison 日志出现 `compare: bias mare noise ...` 行——钩子实调（来源平台：950；证据等级：tool_observed）。
- 重跑 release 套件：33 条 32 PASS / 1 FAIL（仅负向用例 case00038）；此前 bias mare 失败的 case00034/55 五路 bin_precision 全 PASS（来源平台：950；证据等级：tool_observed）。
- 重跑 OOM 套件：33/33 memory_oob PASS；此前 bias 失败的 case00034/35/36/54/55 全部 precision PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- TTK kernel 模式 + `--plugin` 挂 TestSpec 的场景实测；aclnn / e2e / GEIR 通路共用 `compare_with_hooks` 公共入口（框架注释明示各通路统一），但本 trace 未在那些通路实测——**未验证，推断可迁移**。
- TTK 版本为 CANN `cann-9.0.0-beta.2` 配套的 `/tmp/ops-test-kit`；其他版本的日志行号（bridge.py:392、golden.py:406）与 precision_metrics 键集可能不同，判别法第 1/3/5 条需按版本核对。
- 命令行回显截断是 TTK 日志行为，非 shell 层问题——本判别法不适用于其他测试框架。

## 坏实践

- 凭 `Command:` 日志行断言「`--plugin` 没传」——该行不记完整命令行（本 trace 实际踩过，见 ## 误诊）。
- golden.py 更新后不重跑、直接拿旧 result.csv 下「修复无效」结论。
- 钩子未实调时不去确认加载路径（`get_spec_attr` 探针），而直接怀疑钩子实现有 bug 或去改 kernel。

## 误诊

- 本 trace 的实际误诊路径：先据 `bridge.py:392` 的 `Command: ttk op -i ...` 行判断「--plugin 未传、compare 钩子未加载」（当时还推理了 TestSpecManager 按 plugin_path 查找失败的机制）；后在 full_run.log 第 53 行找到 `CustomPluginManager initialized with plugin_path` 证据推翻——原跑插件正常传入，真因是结果文件早于钩子代码生成。教训：**日志里看不到旗标 ≠ 旗标没传**。
- 「result.csv 里 mere/mare/threshold 有值，说明 golden 找到了；所以 compare 钩子一定也在跑」——错误推理：golden 函数找到与 compare 方法存在是两个独立查找（`get_spec_attr(op_name, "golden", ...)` vs `get_spec_attr(op_name, "compare", ...)`），旧版 golden.py 有前者无后者。


