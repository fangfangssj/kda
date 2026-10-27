---
type: "Runbook"
title: TTK --warmup false 批量假失败（banned）
description: 为绕开 TTK warmup 竞态给 ttk kernel 加 --warmup false，实测全量 980 条中出现 144 条 dyn_precision=RTS_BINARY_FAILURE 假失败，其中 119 条在 warmup=true 对照轮为 PASS——该开关下跑批的精度结论不可信，属 banned 绕法；正确做法是子集 --pc 1 重跑。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# TTK --warmup false 批量假失败（banned）

## 现象

在一个 softmax 基准算子（opbuild 工程，Ascend 950，CANN 9.1.0）的白盒验收阶段，TTK kernel 模式全量复跑（默认多进程 + warmup=true）出现 35 条由 warmup.o 重编译竞态造成的整行 FAILURE 假失败（详见同批诊断卡）。为"干净地绕开竞态"，按 TTK 源码推理给全量复跑加了 `--warmup false`：

- `ttk/cli/device.py:17`：`parser.add_argument("--warmup", default=True, ...)`，类型为 `lambda x: x.lower() != "false"`；
- `ttk/cli/bridge.py:93-94`：`if hasattr(args, "warmup") and not args.warmup: sw.warmup = False`——确认开关确实生效；
- `rts_sequence.do()` 中 warmup 只是第 124 行的 helper kernel 启动，算子 kernel 的 register/launch/输出收集（第 126-131 行）与精度比对照常进行。

推理结论是"关 warmup 即无 warmup.o 访问、无竞态、精度信号保留"。实测却更糟：`--warmup false` 全量跑（980 条，进度 100% 980/980，TTK_EXIT=0）解析结果 `total=980 pass=836 fail=144 skip=0`（GATE FAIL），144 条失败的 `dyn_precision` 全部是 `RTS_BINARY_FAILURE`（`dyn_tiling_key='0 (0x0000000000000000)'`、`dyn_block_dim=56`、`memory_oob_status=PASS` 均正常，即 tiling 成功、二进制注册失败）。与本 kernel 已验证的 980/980 PASS 直接矛盾。

## 适用场景

- 工程形态：TTK kernel 模式（`ttk kernel --csv ...`）全量精度跑批；遇到 warmup 竞态假失败后想通过 CLI 开关规避的场景。
- API 路径：`ttk kernel` 的 `--warmup` 开关（`ttk/cli/device.py` argparse 定义、`bridge.py` 开关映射、`rts_interface.warmup`/`rts_sequence.do`）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR）+ CANN 9.1.0 + TTK（ops-test-kit，Rev e559b7f）（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`workflow_only`（评测工具链行为）。
- 路线优先级：`banned`（以 `--warmup false` 跑精度套件获取精度结论的路线被本轨迹成对实测否定）。fallback 声明：正确替代是"识别假失败行 → 子集 `--pc 1` 单进程重跑 → 合并"（见同批诊断卡），`--warmup false` 不在任何情况下充当精度跑批的 fallback；该开关的预期用途（性能计时前控制预热）不在本卡判定范围。
- 明确不适用/未验证场景：`--warmup` 的设计用途（性能测量场景控制预热）未在本轨迹使用；TTK 修复 warmup 依赖后的行为未验证。

## 根因/解决配方

**本卡是负结果（negative_result）：该路线无解方，唯一正确处置是放弃 `--warmup false`、改用子集 `--pc 1` 单进程重跑**（配方见同批诊断卡"TTK kernel 模式多进程 warmup 竞态致整行假 FAILURE"）。

成对证据链（同一 kernel、同一 980 条 CSV）：

- Run 2（默认 warmup=true，多进程）：`pass=945 fail=35`，35 条全部为整行 FAILURE 的 warmup 竞态行，其余 945 行真实 PASS；
- Run 3（`--warmup false`，多进程）：`pass=836 fail=144`，144 条全部 `dyn_precision=RTS_BINARY_FAILURE`；
- 交叉比对（逐 case 名对齐）：Run 3 的 144 条失败中，**119 条在 Run 2 是 PASS**、25 条是 Run 2 的竞态行、0 条不在 Run 2 中——warmup=false 把 119 条真实通过的用例变成了 RTS_BINARY_FAILURE；
- 子集重跑（35 条竞态行，`--pc 1`，默认 warmup=true）：35/35 PASS——同一批 case 在"warmup=true + 单进程"下全部通过，进一步排除 kernel 自身缺陷。

`RTS_BINARY_FAILURE` 的触发机制（`rts_sequence._register_kernel`：kernel 符号不存在/二进制注册失败时返回）在 warmup 关闭后为何大面积出现，本轨迹未追到底（见误诊与遗留问题）；但"该开关使跑批结果失真"这一结论由 119 条成对反差实测钉死，不依赖机制解释。

## 设计决策

- **为何当时选它**：源码推理链完整（argparse 类型 → bridge 映射 → rts_sequence 流程），且看似零成本（不动 TTK、不重跑多轮）；**为何被否**：推理没有覆盖 `warmup()` 可能的隐性副作用（helper kernel 启动对设备/驱动上下文的初始化），实测一票否决——本卡的存在价值就是给下个 agent 省掉这次"合理推理 → 实测翻车"的往返。
- **正确替代**：抽假失败行子集 + `--pc 1` 重跑 + 合并（同批诊断卡），实测 35/35 PASS。
- **当前方案路线档位**：`banned`。

## 系统信息

- 工具链：TTK（ops-test-kit 安装根 `$TTK`，Rev e559b7f）、CANN 9.1.0、SOC Ascend950PR；`ttk kernel` CLI；解析 `skills/whitebox-tester/parse_result.py`。
- 相关文件：`tests/whitebox/ttk/result.csv`（每轮覆盖生成）、`$TTK/ttk/cli/device.py`、`$TTK/ttk/cli/bridge.py`、`$TTK/ttk/core_modules/npu/op/rts_sequence.py`、`$TTK/ttk/core_modules/runtime/rts_interface.py`。

## 预防

- 需要精度结论的 TTK kernel 模式跑批：保持 `--warmup` 默认 true；遇到 warmup 竞态假失败按"识别 → 子集 `--pc 1` 重跑 → 合并"处置。
- 任何"改一个 CLI 开关让结果变干净"的推理，必须与开关默认值下的成对结果交叉比对（本例的 119 条反差就是靠 Run2/Run3 逐 case 对齐发现的）。

## 验证信号

问题与验证来源：一次白盒验收中为绕开 warmup 竞态试用 `--warmup false` 全量复跑，与默认参数轮次逐 case 对比。以下均为本会话实测：

- Run 3（`--warmup false`）：`total=980 pass=836 fail=144 skip=0`，144 条全部 `dyn_precision=RTS_BINARY_FAILURE`（tk=0/bd=56/oob=PASS 正常，输出未产生）（来源平台：950；证据等级：tool_observed）。
- Run 2（默认 warmup=true）同批对照：945 行真实 PASS + 35 行整行 FAILURE 竞态行（来源平台：950；证据等级：tool_observed）。
- 成对反差：`of Run3 failures, were PASS in Run2: 119`、`were FAIL in Run2 (race-corrupted): 25`、`not in Run2: 0`（来源平台：950；证据等级：tool_observed）。
- 替代路线验证：35 条竞态行子集 `--pc 1` 重跑 `total=35 pass=35 fail=0 skip=0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论边界：`--warmup false` 下**跑批的精度/通过率结论不可信**（119/144 与 warmup=true 相反）；不宣称该开关在一切场景（如纯性能计时）下都错。
- 失败形态边界：本轨迹的假失败形态是 `RTS_BINARY_FAILURE`；warmup=false 是否还引入其他形态未测。
- 平台/版本边界：结论绑定本 TTK 版本（Rev e559b7f）+ CANN 9.1.0 + Ascend950PR；TTK 修复 warmup 依赖后未验证（未验证，推断该行为随版本消失）。

**未决项**：

- **优先项：复现并追根因到机制层**——单 case 复现 `--warmup false` 下的 RTS_BINARY_FAILURE，读 `rts_interface._launch_helper_kernel`/`register_device_binary_kernel` 确认 warmup 的隐性初始化副作用，给 TTK 侧提修复或文档化。
- 144 条失败的 case 特征聚类（是否偏向特定 rank/dtype/shape）未做。
## 坏实践

- 把 `--warmup false` 当竞态规避手段写进跑批脚本——本卡即该实践的实测否定。
- 只看单轮 GATE 结论不做成对交叉——Run 3 单看是"144 条二进制注册失败"，极易误判为 kernel/tiling 二进制缺陷去查 `REGISTER_TILING_DEFAULT` 或 tiling key 路由。

## 误诊

- 对 RTS_BINARY_FAILURE 成因的机制假设（agent 转述，未验证，不得当实测事实）：验收者推测 `warmup()` 启动 helper kernel 可能附带设备/驱动上下文初始化副作用，缺失后部分 case 的 `register_kernel` 失败；该假设未读 `rts_interface` 实现层验证。结论"该开关结果不可信"不依赖此假设。
