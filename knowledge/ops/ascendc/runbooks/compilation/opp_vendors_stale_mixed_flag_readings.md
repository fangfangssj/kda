---
type: "Runbook"
title: "flag 改了不生效：opp/vendors 有 stale 同名算子安装"
description: "修改 OpDef 标志后 route 日志对同一算子出现新旧两种读数（InitOpItemValueByMaping 混读），说明 CANN OPP 下存在多个同名算子安装；按 vendor 逐一排查并做条目级清理后读数归一，实验结论才可信。"
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# flag 改了不生效：opp/vendors 有 stale 同名算子安装

## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessScatter 做 GEIR 验收。为控制图编译路径，把 `op_host/bench_harness_scatter_def.cpp` 里的 `aicoreConfig.DynamicCompileStaticFlag(true)` 改成了 `false`，重新 stage 并跑 route 探测。预期日志只应出现一种读数，但 route 日志（plog 调试级别）对同一算子给出了三次 `InitOpItemValueByMaping` 读数，新旧并存：

```
[DEBUG] FE(3034791,geir_route):... [op_kernel_info_constructor.cc:1399] InitOpItemValueByMaping:The [dynamicCompileStatic][flag] of op [BenchHarnessScatter] is [false].
[DEBUG] FE(3034791,geir_route):... [op_kernel_info_constructor.cc:1399] InitOpItemValueByMaping:The [dynamicCompileStatic][flag] of op [BenchHarnessScatter] is [true].
[DEBUG] FE(3034791,geir_route):... [op_kernel_info_constructor.cc:1399] InitOpItemValueByMaping:The [dynamicCompileStatic][flag] of op [BenchHarnessScatter] is [true].
```

同时日志里能看到加载失败的旧路径残留：`LoadOpMasterSo:Get path with op_master path [/workspace/.../cann-9.2.0/opp/vendors/BenchHarnessScatter/op_impl/ai_core/tbe/op_master/lib/linux/x86_64/] failed`，以及 `ResolveBinaryPath` 的 `op.binary.custom` 列表里挂着一串 vendor kernel 目录（BenchHarnessScatter、customize、ScaleCustom、custom_ops、EuclideanNorm、AddCustom、mla_prolog_custom）。也就是说：本会话的隔离 staging 包（`/tmp/geir_opp.*/vendors/BenchHarnessScatter`）并不是系统眼里唯一的算子来源。

## 适用场景

- 工程形态：自定义算子开发/验收，曾在 CANN OPP（`$ASCEND_HOME_PATH/opp/vendors/`）安装过算子包，之后改源码并用隔离 staging 验证。
- API 路径：op_build 产物（`aic-<soc>-ops-info.json`、`binary_info_config.json`、`config.ini` 的 `load_priority`）与 GE/FE 的 vendor 加载（`InitOpItemValueByMaping`、`LoadOpMasterSo`、`ResolveBinaryPath`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何"OPP 安装 + 隔离 staging 并存"的算子开发环境。重装前清理旧安装残留的通用做法另见[重装算子包前清理 OPP vendors 旧安装残留卡](cleanup_vendors_before_reinstall.md)。
- 路线优先级：recommended_fastpath（出现混读时先做环境排查，再继续任何源码实验）。
- 明确不适用 / 未验证场景：单一安装来源的环境（无 stale 可言）；本卡不解决 staging 包与 OPP 包谁优先的语义问题（未验证）。

## 判别方法

1. **触发信号**：改了 OpDef/ops-info 相关标志后，route 日志中同一 flag 的 `InitOpItemValueByMaping` 读数出现多种值（一次 false + 多次 true 即典型混读）。
2. **探针**——逐一列出系统实际会读的安装来源：

```bash
# CANN OPP vendors 目录下有哪些算子 vendor（含 config.ini 的加载顺序）
ls -la "$ASCEND_HOME_PATH/opp/vendors/"
cat "$ASCEND_HOME_PATH/opp/vendors/config.ini"        # load_priority=...

# 每个 vendor 的 ops-info 里是否登记了你的算子
for f in "$ASCEND_HOME_PATH"/opp/vendors/*/op_impl/ai_core/tbe/config/ascend*/aic-*-ops-info.json; do
  python3 - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
if 'BenchHarnessScatter' in d:   # 换成你的算子名
    print(sys.argv[1], '->', d['BenchHarnessScatter'].get('aicore', {}).get('dynamicCompileStatic'))
EOF
done

# binary_info_config.json 里是否有你的算子的 kernel 条目
grep -l "BenchHarnessScatter" "$ASCEND_HOME_PATH"/opp/vendors/*/op_impl/ai_core/tbe/kernel/config/ascend*/binary_info_config.json 2>/dev/null
```

3. **观测**：本例命中两个 stale 来源——`opp/vendors/BenchHarnessScatter/`（整包直接安装，ops-info 为 true）和 `opp/vendors/custom_ops/`（打包进了别的聚合 vendor：kernel 目录、`binary_info_config.json` 条目、`aic-ascend950-ops-info.json` 条目、op_api 头文件）。

## 解决配方

清理 stale 安装（直接 vendor 整删；聚合 vendor 里做条目级删除，别误伤其他算子）：

```bash
# 1) 直接安装的同名 vendor 整目录删除（root 权限不足时会剩 scripts/uninstall.sh，见失效边界）
rm -rf "$ASCEND_HOME_PATH/opp/vendors/BenchHarnessScatter"

# 2) 聚合 vendor（如 custom_ops）里按文件/条目删除
rm -rf "$ASCEND_HOME_PATH/opp/vendors/custom_ops/op_impl/ai_core/tbe/kernel/ascend950/bench_harness_scatter_apt"
python3 - <<'EOF'
import json
# 从 binary_info_config.json 删除算子条目（原地重写）
p = "$ASCEND_HOME_PATH/opp/vendors/custom_ops/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json"
d = json.load(open(p))
d.pop('BenchHarnessScatter', None)
json.dump(d, open(p, 'w'), indent=2)
# 从 ops-info 删除算子条目
p2 = "$ASCEND_HOME_PATH/opp/vendors/custom_ops/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json"
d2 = json.load(open(p2))
d2.pop('BenchHarnessScatter', None)
json.dump(d2, open(p2, 'w'), indent=2)
EOF
rm -f "$ASCEND_HOME_PATH/opp/vendors/custom_ops/op_api/include/aclnn_bench_harness_scatter.h"
```

清理后重跑同一条 route 探测，验证 `InitOpItemValueByMaping` 只剩 staging 包的单一读数（本例为 `[false]`，与源码一致），且 stale 路径的 `LoadOpMasterSo ... failed` 告警消失。

## 设计决策

- 为什么条目级删除而不是删整个聚合 vendor：custom_ops 里还装着其他算子（AddCustom 等独立目录除外，聚合目录共享 config 文件），整删会破坏无关算子。
- 为什么清理后仍要重跑 route 而不是直接采信：清理只证明"读数归一"，不证明图路径恢复（本例读数归一后 RunGraph 仍失败，属另一问题，见同批 GE 图路径 TBE 动态编译判别知识）。
- `rm` 删不掉 `scripts/uninstall.sh`（Permission denied）时：该残留只剩卸载脚本、无 ops-info/kernel 配置，不再参与加载，可容忍；不必强求删净。
- 当前方案档位：recommended_fastpath。

## 预防

- 装过 `.run` 包又切换到隔离 staging 工作流时，先 `ls $ASCEND_HOME_PATH/opp/vendors/` 检查同名算子是否残留；改任何 ops-info 相关标志的实验前先确认读数唯一。
- 聚合 vendor（custom_ops/customize 等）是 stale 的常见藏身处，只看顶层 `vendors/<OpName>/` 会漏。

## 验证信号

以下信号来自该算子 GEIR 验收会话的 route 日志与文件系统实测。

- 混读证据：`InitOpItemValueByMaping` 对 BenchHarnessScatter 的 dynamicCompileStatic 三次读数 false/true/true（stale 未清理时）（来源平台：950；证据等级：tool_observed）。
- stale 安装证据：`opp/vendors/BenchHarnessScatter/`（Jul/Sep 旧时间戳目录）与 `opp/vendors/custom_ops/` 的 ops-info/binary_info_config/kernel 目录均含 BenchHarnessScatter（来源平台：950；证据等级：tool_observed）。
- 清理后验证：重跑 route，`InitOpItemValueByMaping` 仅剩 `[false]` 单一读数（与源码 DynamicCompileStaticFlag(false) 一致），stale 路径 LoadOpMasterSo 告警消失（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只闭合"读数混叠"这一环境层问题；清理后图路径 RunGraph 仍失败（预编译二进制不命中）是独立问题，本卡不解决。
- "多 vendor 读数合并、先到/后到谁覆盖谁"的优先级语义未验证；不要据本卡推断 OPP 安装会覆盖 staging（本会话只有混读观测，无优先级结论）。
- 其他 CANN 版本的日志组件名（FE/op_kernel_info_constructor）可能不同（未验证，推断可迁移）。

**未决项**：

- `config.ini` 的 `load_priority` 与多 vendor 同名算子的覆盖顺序语义未查证（本会话只做了清理，未研究优先级规则）。
- `rm` 无法删除的 `scripts/uninstall.sh`（权限）在后续 `.run` 安装时是否冲突未验证。
## 坏实践

- 在存在 stale 安装的环境里做标志/注册实验并采信结果——读数混叠会让任何 A/B 对照失效。
- 只删顶层 `vendors/<OpName>/` 就认为清理完成——聚合 vendor 里的 ops-info/binary_info_config 条目仍会被读取。

## 误诊

- 无（本卡结论全部有日志/文件系统直接观测支撑）。
