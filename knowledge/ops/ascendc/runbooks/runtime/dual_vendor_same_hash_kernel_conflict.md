---
type: "Runbook"
title: "多 vendor 同 hash kernel 冲突的检测与排除"
description: "同一自定义算子可被安装到 $ASCEND_OPP_PATH/vendors 下多个 vendor 目录且 kernel .o 同名同 hash；用体积/时间戳普查 + hash 集合比对 + 临时改名重跑识别存根与真实二进制冲突，并确认它是否为当前故障根因。"
tags: [task.debugging, topic.implementation.operator_registration, topic.architecture]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# 多 vendor 同 hash kernel 冲突的检测与排除

## 现象

在 Ascend950PR（CANN 9.1.0）上排查一个自定义 scatter 算子"TTK kernel 直调模式全 shape VEC_ERROR"时，怀疑安装目录里存在多份该算子的 kernel 二进制、TTK 加载到了错误的一份。对 `$ASCEND_OPP_PATH/vendors/` 做普查后发现：

- 两个 vendor 目录（`custom_ops` 与算子同名 vendor `BenchHarnessScatter`）都注册了该算子（各自的 `op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json` 均含该算子条目）；
- 两个 vendor 的 `op_impl/ai_core/tbe/kernel/ascend950/<op>_apt/` 目录下各有一套 **10 个同名同 hash** 的 `BenchHarnessScatter_<hash>.o` 文件，但内容完全不同：
  - `custom_ops` 侧：每个 6208 字节，时间戳 2026-09-03 15:31（存根/占位二进制，与该 vendor 的 liboptiling.so 同刻）；
  - `BenchHarnessScatter` 侧：358088–386352 字节，时间戳 2026-09-04 20:28（真实编译产物，与该 vendor 的 liboptiling.so 同刻）；
- TTK 每次运行都会加载全部 vendor 的 op info（日志可见逐 vendor 的 `Loading vendor op info from: ...`）。

即：**同 hash 名、不同内容的多份 kernel 二进制跨 vendor 并存**，实际解析到哪一份由 vendor 加载/解析顺序决定，且不报任何冲突错误。

## 适用场景

- 工程形态：自定义算子安装/部署后的行为排查（kernel 加载结果与预期不符、怀疑执行了旧版或存根二进制）。
- API 路径：`$ASCEND_OPP_PATH/vendors/<vendor>/op_impl/ai_core/tbe/kernel/<soc>/<op>_apt/*.o` 与 `.../config/<soc>/aic-<soc>-ops-info.json`、`.../op_tiling/liboptiling.so`；TTK/aclnn 的 vendor op info 加载日志。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——多 vendor 安装布局是 CANN opp 目录机制，与具体算子无关。
- 路线优先级：safety_fallback（本检测是排查手段而非修复；修复取决于安装策略，见失效边界的 fallback 声明）。
- 明确不适用 / 未验证场景：单一 vendor 的正常安装（无冲突可查）；builtin（built-in）算子路径；其他 CANN 版本的目录结构（未验证，推断可迁移）。重装前清理旧安装残留的通用做法见[重装算子包前清理 OPP vendors 旧安装残留卡](../compilation/cleanup_vendors_before_reinstall.md)，vendor 装载链路验证探针见[vendor 包 aclnn 装载链路验证探针清单卡](../compilation/vendor_package_aclnn_loading_four_probes.md)。

## 判别方法

三步普查 + 一次改名实验（verbatim，路径按实际算子/soc 替换）：

1. **体积/时间戳普查**——同名 .o 跨 vendor 体积差数量级即是存根特征：

```bash
V0=/workspace/…/Ascend/cann-9.2.0/opp/vendors/custom_ops
V1=/workspace/…/Ascend/cann-9.2.0/opp/vendors/BenchHarnessScatter
ls -la --time-style=+%Y-%m-%d_%H:%M $V0/op_impl/ai_core/tbe/kernel/ascend950/bench_harness_scatter_apt/*.o | head -3
ls -la --time-style=+%Y-%m-%d_%H:%M $V1/op_impl/ai_core/tbe/kernel/ascend950/bench_harness_scatter_apt/*.o | head -3
```

2. **hash 集合比对**——确认冲突是"同 hash 双份"而非"各装各的"：

```bash
comm -3 <(ls $V0/op_impl/ai_core/tbe/kernel/ascend950/bench_harness_scatter_apt/ | grep '\.o$' | sort) \
        <(ls $V1/op_impl/ai_core/tbe/kernel/ascend950/bench_harness_scatter_apt/ | grep '\.o$' | sort) && echo "IDENTICAL HASH SETS"
```

3. **确认哪些 vendor 注册了该算子**（op info 存在性）：

```bash
find /workspace/…/Ascend/cann-9.2.0/opp/vendors -name 'aic-ascend950-ops-info.json' | while read f; do
  grep -q BenchHarnessScatter "$f" 2>/dev/null && echo "HAS OP: $f"
done
```

4. **改名实验定位实际加载来源**（可逆，务必恢复）——临时移走一个 vendor，重跑最小用例，从日志的 `Kernel Name: BenchHarnessScatter_<hash>` 确认剩余 vendor 的二进制被加载：

```bash
V=/workspace/…/Ascend/cann-9.2.0/opp/vendors/custom_ops
mv "$V" "${V}_disabled_test" && echo "renamed -> ${V}_disabled_test"
# 重跑最小用例（本例 kernel 模式 tiny），观察 Kernel Name 与结果
cd /tmp/ops-test-kit && timeout 200 "$PY" -m ttk kernel -i /tmp/tiny.csv -t tiny_int32_ovw \
  -d false -b release --pc=1 --task-prof false --run 1 --seed 42 \
  --plugin <golden.py> -o /tmp/perf_ktest/r.csv 2>&1 | grep -iE "VEC_ERROR|Kernel Name|STATUS:"
mv "${V}_disabled_test" "$V" && echo "restored"; ls -d "$V" && echo "vendor back in place"
```

观测（本例）：改名后 kernel 模式日志显示加载 `BenchHarnessScatter_d0a8473acb396193d1eb4813c6aad0e5`（BenchHarnessScatter vendor 的 359232 字节真实 kernel），但 VEC_ERROR 依旧——**冲突存在，却不是本故障的根因**。据此把排查方向从"二进制解析"移开。

## 设计决策

- 为什么用"体积 + 时间戳"做存根指纹：AscendC 真实 kernel .o 通常数十至数百 KB；个位数 KB 的 .o 与同 vendor liboptiling.so 同刻生成，是构建/安装过程留下的占位产物。体积差一个数量级以上即可判定，无需反汇编。
- 为什么用改名而不是删除：改名完全可逆，实验后一条 `mv` 即恢复，不会破坏安装环境；同时日志里 `Kernel Name` 的 hash 能直接指认实际加载的二进制。
- 为什么实验后必须复查恢复：多 vendor 环境常被其他算子共享（本例 custom_ops 还注册了其他算子），残留禁用状态会引入新故障。
- 当前方案路线档位：safety_fallback——检测/排除手段本身无代价，但"清理冲突 vendor"属于环境治理动作，需按安装策略决定（删除存根或统一安装目标），不在本卡给出一刀切修复。

## 系统信息

- 硬件/软件：Ascend950PR_9579，CANN 9.1.0（目录名 cann-9.2.0）；TTK commit b31db58。
- 关键目录：`$ASCEND_OPP_PATH/vendors/<vendor>/op_impl/ai_core/tbe/{kernel/ascend950/<op>_apt, config/ascend950, op_tiling}` 与 `.../custom_ops_impl/dynamic/`（动态编译源码树）。
- 本例涉及 vendor：`custom_ops`（6KB 存根 + 旧动态源码）、`BenchHarnessScatter`（375KB 真实 kernel + 新 host tiling）、`customize`（另有 liboptiling，未深查）。

## 预防

- 安装/升级自定义算子后，跑一次三步普查（体积/时间戳、hash 集合、op info 注册方），确认没有跨 vendor 的同 hash 双份残留。
- 构建脚本以同一 vendor 名为安装目标，避免一次装 `custom_ops`、一次装算子名 vendor 造成双注册。
- 排查"kernel 行为与源码不符"类问题时，先确认实际加载的二进制（日志 Kernel Name 的 hash + 体积），再读代码。

## 验证信号

以下信号来自该 scatter 算子 kernel 模式 VEC_ERROR 排查会话中的安装目录普查与改名实验。

- custom_ops vendor：10 个 kernel .o 均 6208 字节、时间戳 2026-09-03 15:31（与该 vendor liboptiling.so 同刻）（来源平台：950；证据等级：tool_observed）。
- BenchHarnessScatter vendor：10 个 kernel .o 为 358088–386352 字节、时间戳 2026-09-04 20:28（与该 vendor liboptiling.so 同刻）（来源平台：950；证据等级：tool_observed）。
- 两 vendor 的 .o 文件名 hash 集合经 `comm -3` 比对完全一致（IDENTICAL HASH SETS）；两 vendor 的 op info json 均注册该算子（来源平台：950；证据等级：tool_observed）。
- 改名禁用 custom_ops 后重跑：日志加载 `BenchHarnessScatter_d0a8473a...`（真实 kernel），kernel 模式仍 VEC_ERROR——冲突被排除为该故障根因（来源平台：950；证据等级：tool_observed）。
- 实验后 custom_ops vendor 已恢复原位、无残留禁用目录（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只解决"怎么发现与定位冲突、怎么确认它是不是根因"，不决定"该删哪一份"——清理策略取决于该环境的安装流程（本轨迹未执行清理）。
- 存根判定基于体积指纹（KB 级 vs 百 KB 级）；若两个 vendor 都是真实但版本不同的二进制，需改用 hash/时间戳/内容 diff 判别新旧。
- 改名实验期间其他进程若依赖被禁用 vendor 会受影响；应在无并发任务时执行并尽快恢复。
- 观测基于 CANN 9.1.0 + 950；其他版本目录布局未验证（推断可迁移）。

**未决项**：

- 双 vendor 同 hash 并存时，TTK/CANN 实际解析顺序的规则（vendor 遍历顺序、是否可配置）未查证。
- custom_ops vendor 的 6KB 存根与旧版动态源码树是否仍在其他路径（如 aclnn 缓存、GE 编译缓存）被引用，未排查。
- 存根的产生链路（哪次构建/安装写入了 6KB .o）不在本轨迹内，无法归因。
## 坏实践

- 只看文件名 hash 相同就认为两个 vendor 的二进制等价——本例同 hash 体积差 60 倍。
- 直接 `rm -rf` 疑似存根 vendor 而不做改名验证与恢复预案——破坏共享安装环境且不可逆。
- 把"发现冲突"直接当"找到根因"——本例禁用存根后故障依旧，冲突只是并存事实；必须做改名实验闭环归因。

## 误诊

- "kernel 模式 VEC_ERROR 是因为加载了 6KB 存根"：改名实验证伪——加载真实 375KB kernel 后仍然 VEC_ERROR。冲突是真实存在的安装问题，但与该故障无因果。
