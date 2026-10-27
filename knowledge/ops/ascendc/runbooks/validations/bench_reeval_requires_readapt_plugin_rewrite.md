---
type: "Runbook"
title: "改kernel后bench复评须重跑适配并重生成op_plugin"
description: "bench 评测编译安装的是适配目录（output-dir）里的源码快照，算子工程内改动不自动同步；且适配脚本每次运行会把 op_plugin/<op>_plugin.cpp 与 cann_bench/__init__.py 重置为模板。改 kernel 后的复评必须『重跑适配→重写两文件→再评测』；改动后评测结果与改动前逐位相同即测到旧快照。"
tags: [task.debugging, topic.implementation.operator_registration, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# 改kernel后bench复评须重跑适配并重生成op_plugin

## 现象

在 Ascend 950 自定义算子工程内完成 kernel 修改，执行 `bash build.sh --soc=ascend950 -j8` 编译、安装 custom_opp 包全部成功后，直接运行 cann-bench 评测（run_evaluation.sh，`--source-dir` 指向此前生成的适配目录）：

```
[1/1] level3/depthwise_conv_2d_5: ❌ 精度不达标:
  - float32[y]: ❌ MERE=2.657557e-07, MARE=0.138477 (threshold=1.220703e-04, ...)
```

MARE=0.138477 与修改前**逐位相同**——刚改的 kernel 像根本没编译进去一样。排查发现两个机制叠加：

1. bench 评测的 `--source-dir` 是适配脚本生成的独立目录快照（评测框架自行在其中编译 whl/run 包并安装到自己的 `custom_ops` vendor），算子工程目录里的修改不会自动进入这个快照；
2. 重跑适配脚本（adapt_to_bench.sh）同步源码后，它会把需要人工生成的文件重置：`csrc/ops/<op_snake>/op_plugin/<op_snake>_plugin.cpp` 变回模板/缺失、`cann_bench/__init__.py` 变回模板，并在输出里明确打印：

```
Files copied:
  op_kernel: 2 files
  op_host:   3 files
  op_api:    2 files

TODO (requires AI agent):
  1. Generate csrc/ops/<op_snake>/op_plugin/<op_snake>_plugin.cpp
     Reference: .../templates/op_plugin_template.cpp
  2. Update cann_bench/__init__.py with correct function signatures
  3. Run: cd <output-dir> && bash build.sh
```

重新写入这两个文件并重建后，同一评测的 MARE 变为 0.051986——证明前一次评测跑的确实是旧代码。该"重跑适配→重写 op_plugin/__init__.py"的循环在本次联调中被执行了至少 4 轮，每轮漏掉任一步都会得到"改动未生效"或"算子调用失败"的假象。

## 适用场景

- 工程形态：自定义算子（ops-registry-invoke 类工程）接入 cann-bench 精度评测的联调迭代：改 kernel/op_host → 期望 bench 复评反映改动。首次适配（骨架生成、op_plugin 写法、schema 对齐）见 [cann-bench op_plugin 算子适配配方卡](cann_bench_op_plugin_adaptation_recipe.md)；本卡只覆盖"复评为何测到旧代码"的刷新循环。
- API 路径：`adapt_to_bench.sh --operator-dir ... --op-pascal ... --op-snake ... --bench-root ... --output-dir ...`；`run_evaluation.sh --source-dir <适配目录> --operator <名> --device-id 0 --no-perf [--case-id N]`；op_plugin（torch library 注册 + ACLNN_CMD 调用）与 `cann_bench/__init__.py`（Python 包装）。
- 验证过的平台与 CANN 版本：Ascend 950（arch35），CANN 9.2.0。
- 适用粒度：workflow 级工具链诊断——cann-bench 适配评测流程，不涉及 kernel 行为本身。
- 路线优先级：recommended_fastpath（该循环是复评改动的唯一正确路径）。
- 明确不适用/未验证场景：不经适配目录、直连算子包的调测脚本（如直接 import 自装 whl 的 python 探针）不经过本快照机制；其他评测框架不适用。

## 根因

bench 评测编译并安装的是适配目录（output-dir）中的源码快照，算子工程内的修改不自动同步；而刷新快照的适配脚本固定以模板覆盖 op_plugin 与 `cann_bench/__init__.py` 并要求人工重新生成，二者叠加使"改 kernel→直接评测"必然测到旧代码。

## 解决配方

改 kernel 后的 bench 复评完整循环（每轮缺一不可）：

```bash
# 1. 算子工程内构建安装
cd <算子工程> && bash build.sh --soc=ascend950 -j8 && bash build/custom_opp_ubuntu_x86_64.run --quiet

# 2. 重跑适配，同步最新源码到适配目录
bash <bench-adaptation 脚本路径>/adapt_to_bench.sh \
  --operator-dir operators/<op_snake>/ --op-pascal <OpPascal> --op-snake <op_snake> \
  --bench-root <cann-bench根> --output-dir <适配输出目录>

# 3. 重新写入被模板覆盖的两个文件（内容见适配配方卡的完整代码）
#    - <适配目录>/csrc/ops/<op_snake>/op_plugin/<op_snake>_plugin.cpp
#    - <适配目录>/cann_bench/__init__.py

# 4. 评测（框架会在适配目录内自行编译 whl/run 包并安装到 custom_ops vendor）
cd <cann-bench根> && bash scripts/run_evaluation.sh \
  --source-dir <适配目录>/ --operator <Operator名> \
  --device-id 0 --no-perf [--case-id N]
```

判别信号：改动后评测结果与改动前**逐位相同**（同 MERE/MARE/mismatch）→ 几乎必然测的是旧快照，回到第 2 步。两个被重置文件的写法（torch library 注册、ACLNN_CMD 两段式调用、schema 逐字对齐 proto、`__init__.py` 签名替换）见 [cann-bench op_plugin 算子适配配方卡](cann_bench_op_plugin_adaptation_recipe.md)——首次生成后应把终版内容随算子工程存档，每轮适配后直接覆写，避免每次重新手写。

## 设计决策

- 评测框架为何用独立快照而非直连算子工程：适配目录是自包含评测包（whl + run 包一体、安装到评测专用 `custom_ops` vendor），保证评测可复现、不污染工程环境——代价就是每次改动必须显式刷新快照。
- 适配脚本为何重置 op_plugin：该文件含算子专属的 torch library 注册与 aclnn 调用签名，脚本只放模板并打印 TODO 交由开发者生成；因此两个文件的内容应在首次生成后存档，每轮适配后直接覆写。

## 验证信号

本卡验证信号来自一次"改动未生效→按配方刷新→生效"的对照闭环及后续多轮重复（Ascend 950 + CANN 9.2.0）：

- 触发：kernel 已改并安装，直接评测得 MARE=0.138477，与改动前逐位相同（来源平台：950；证据等级：tool_observed）。
- 适配脚本行为：每次运行输出 "Files copied: op_kernel 2 / op_host 3 / op_api 2" + "TODO (requires AI agent): 1. Generate ... op_plugin ... 2. Update cann_bench/__init__.py"；`__init__.py` 实际内容为模板（来源平台：950；证据等级：tool_observed）。
- 修复生效：重跑适配 + 重写两文件 + 评测后，同用例 MARE=0.051986（数值变化证明评测确实换用了新代码）（来源平台：950；证据等级：tool_observed）。
- 流程稳定性：同一循环在后续每轮改动复评中重复执行（≥4 轮），均一次生效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"评测为何测到旧代码/插件为何被重置"的流程层；kernel 本身的精度行为不在本卡范围。
- 直连式探针（python 直接调自装 whl 走 `<op>_custom` vendor）不经适配快照，其结果与 bench 评测可能来自不同 vendor 的二进制——两边对照时须意识到这一点。
- 适配脚本参数（--op-pascal/--op-snake 等）以当次脚本版本为准，本卡命令形态对应本次联调所用版本。
- 其它评测框架的快照/刷新机制未验证（未验证，推断可迁移——"独立评测快照 + 显式刷新"是常见模式）。

**未决项**：

- 适配脚本输出 TODO 后由人工回写两个文件属流程空隙；可评估把 op_plugin/__init__.py 的终版内容随算子工程存档并在适配后自动覆写。
## 坏实践

- 改完 kernel 只做"build.sh + 安装"就跑 bench 评测：必然测旧快照，白白浪费一轮编译评测时间，还容易诱发"算法没效果"的错误结论。
- 重跑适配后忘记重写 op_plugin/__init__.py：评测编译/调用阶段才失败，报错形态与本卡现象不同但同源。
- 把"评测结果逐位不变"当成"改动无效"的证据去回滚算法——先证伪快照新鲜度再下结论（构建层的同类陷阱见 [kernel 头改动后增量编译沿用旧二进制卡](../compilation/incremental_build_stale_kernel_binary.md)）。

## 误诊

- "MARE 一模一样说明算法改动对这个算子无效"：误；该轮评测根本没有包含新代码，"逐位相同"是快照未刷新的指纹（后经刷新后 MARE 变化证实）。
