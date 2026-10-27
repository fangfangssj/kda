---
type: "Runbook"
title: 自定义算子host的OP_LOGI不进stdout与plog
description: CANN 9.0.0-beta.2 + Ascend950PR 实测：opp/vendors 安装的自定义算子 host 侧 OP_LOGI（INFO 级 tiling 日志）在 ASCEND_GLOBAL_LOG_LEVEL=0 + ASCEND_SLOG_PRINT_TO_STDOUT=1 下不出现在 stdout（22.5 万行全为 GE WARNING），测试进程也无 plog 文件；R2（CANN 9.1.0）：可行观测路线为 TilingFunc 内临时 fprintf(stderr) 直打切分实参——TTK 控制台原样可见，用后必须移除并 grep 验证残留 0。
tags: [task.debugging, topic.architecture]
status: stable
sources: []
created_at: "2026-09-09T11:49:41Z"
updated_at: "2026-09-17T03:25:37Z"
platforms: ["950"]
---

# 自定义算子host的OP_LOGI不进stdout与plog

## 现象

在排查一个自定义算子（ops-registry-invoke 算子仓流程，经 `build.sh` 安装到 `${ASCEND_OPP_PATH}/vendors/<op>/`，aclnn 调用链实跑）的 tiling 分支路由问题时，想直接读 host TilingFunc 里打的 INFO 日志来查看运行时 tiling 字段与分支判定结果。host 源码里确实有这些日志点（`#include "op_common/log/log.h"`，`OP_LOGI(context, "ThnnFusedGruCellGrad tiling: axisNum=%d isTailR=0 usedCoreNum=%d isGroup=%d", ...)`、`"ThnnFusedGruCellGrad inputs: ..."`、`"ThnnFusedGruCellGrad shapes: ..."`、`"Set currentWorkspace size:%zu, ..."` 等）。

按惯常的 slog 打开方式做两轮实测，均拿不到这些行：

**第一轮**（单用例清单实跑）：

```bash
ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn /tmp/opencode/manifest_n2.json ../st_data_branch1 \
  > /tmp/opencode/n2_probe.log 2>&1; echo "exit=$?"; grep -c "" /tmp/opencode/n2_probe.log; grep -i "thnn" /tmp/opencode/n2_probe.log | head -10
```

```text
exit=255
199
[smoke] thnn_fused_gru_cell_grad fp32 PASSED
==== ThnnFusedGruCellGrad aclnn smoke summary: 1/1 cases OK ====
```

stdout 共 199 行，只有冒烟与汇总输出，grep 算子名无任何 host 日志行。（事后确认 `ASCEND_GLOBAL_LOG_LEVEL=3` 是 **ERROR-only**，不是 INFO——级别语义为 0=debug、1=info、2=warning、3=error。）

**第二轮**（换成 debug 级重跑同一探针）：

```bash
ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn /tmp/opencode/manifest_n2.json ../st_data_branch1 \
  > /tmp/opencode/n2_probe.log 2>&1; echo "exit=$?"; \
  grep -E "ThnnFusedGruCellGrad tiling|isGroup" /tmp/opencode/n2_probe.log | head -5; wc -l /tmp/opencode/n2_probe.log
```

```text
exit=255
225002 /tmp/opencode/n2_probe.log
```

stdout 洪泛到 225002 行，但 grep `"ThnnFusedGruCellGrad tiling|isGroup"` 命中的全部是 GE 的 WARNING 注册类日志，一条 OP_LOGI 的 tiling/inputs/shapes/workspace 行都没有：

```text
[WARNING] GE(1711617,test_aclnn):2026-09-06-16:53:52.249.515 [op_impl_space_registry_v2_impl.cc:166]1711617 \
MergeFunctions:op type ThnnFusedGruCellGrad tiling_parse func has been registered.
[WARNING] GE(1711617,test_aclnn):2026-09-06-16:53:52.249.522 [op_impl_space_registry_v2_impl.cc:169]1711617 \
MergeFunctions:op type ThnnFusedGruCellGrad tiling func has been registered.
（…同型行反复出现，共 22.5 万行）
```

**plog 侧三路搜索同样落空**：

- 本次 ST 运行生成的 run/plog（`~/ascend/log/run/plog/plog-1708298_20260906164955809.log`，36259 字节、178 行）grep 算子名/tiling 零命中；
- 全 `~/ascend/log` 树 grep `"ThnnFusedGruCellGrad tiling"` 仅命中**历史** GE ERROR 行（`op_tiling_py.cc:1796 TbeOptilingPyInterfaceNew: ErrorNo: 4294967295(failed)Op ThnnFusedGruCellGrad tiling failed`，属早前 python/GE 进程），无任何 INFO 行；
- 按本次测试进程 PID（1711617）`find / -name "plog-1711617*"` → **文件不存在**（该 aclnn 测试进程根本没有生成 plog）。

结论：在这套环境里，想用 slog/plog 读自定义算子 host 的 OP_LOGI 来获取运行时 tiling 值是一条死路；tiling 运行时值调试须改走行为探测（见同问题域正卡「GetCoreNumAiv 运行时少于 INI 核数」的边界扫描法）。

## 适用场景

- **工程形态**：aclnn 算子仓自定义算子（opp/vendors 安装）host TilingFunc 的运行时值调试（tiling 字段、分支判定、workspace 尺寸）。
- **API 路径**：`op_common/log/log.h` 的 `OP_LOGI`/`OP_LOGE`/`OP_CHECK_*`；环境变量 `ASCEND_GLOBAL_LOG_LEVEL` / `ASCEND_SLOG_PRINT_TO_STDOUT`；日志目录 `~/ascend/log/{run,debug}/plog/`。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册词表 `950`）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致；仅此组合实测）。
- **适用粒度（transfer_scope）**：`platform_wide`——该 CANN 版本下自定义算子 host 日志的可观测性。
- **路线优先级**：`banned`——"设 slog 环境变量/翻 plog 读自定义算子 host OP_LOGI"这条调试路线实测不可行；替代路线（行为探测）为 recommended_fastpath。
- **明确不适用 / 未验证场景**：GE/runtime 自身的 WARNING 日志不受影响（它们确实进了 stdout）；`OP_LOGE`（ERROR 级）能否出现在 plog 未单独验证（历史 plog 中的 ERROR 行来自 GE 层而非算子 host 宏）；其他 CANN 版本（含 9.x 正式版）未验证。

## 根因

**未追到机制层**（如实承认）。已确证的观测事实：自定义算子 host 侧 `OP_LOGI`（INFO 级）在 `ASCEND_GLOBAL_LOG_LEVEL=0`（按 0=debug 语义应包含 INFO）+ `ASCEND_SLOG_PRINT_TO_STDOUT=1` 下不出现在 stdout；aclnn 测试进程没有生成自己的 plog 文件；历史 plog 中只有 GE 层 ERROR 行。推测方向（未验证）：vendor 自定义算子的 host 日志走独立 sink，或被某层默认级别/开关过滤。机制层根因列为「遗留问题」优先项。

## 判别方法

- **快速判别你的场景是否命中**：`ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1` 跑一次最小用例，grep 你的算子名——若只见 GE 注册 WARNING（`op_impl_space_registry_v2_impl.cc ... has been registered`）、不见你 host 源码里的 OP_LOGI 格式串，即命中本卡场景。
- **日志级别语义**：`ASCEND_GLOBAL_LOG_LEVEL`：0=debug、1=info、2=warning、3=error。**3 不是 INFO**——设 3 只会得到 ERROR-only 输出（本开发会话 第一轮即犯此错）。

## 解决配方

本开发会话 内**没有找到日志侧解法**。可复用的处置：

1. 止损判据：两轮 slog 实测 + plog 三路搜索（按内容、按时间、按进程 PID）均为空后，停止在日志侧继续投入，切换到行为探测。
2. 替代路线（本开发会话 实测有效）：放弃"读日志看运行时 tiling 值"，改为构造只扫目标谓词变量的 shape 序列，用"未实现分支桩毒化输出 FAIL / 已实现分支 PASS"作二值路由信号，反推运行时参数——完整配方见同问题域正卡（边界扫描定界法）。
3. 若后续必须拿日志：未尝试方向见「遗留问题」（`=1` 级、`ASCEND_SLOG_PRINT_TO_FILE`、slog 配置文件、按 host 进程查全部日志目录）。

## 设计决策

- **为何把这条路线判为 banned 而不是"再试几个环境变量"**：两次实测（级别 3 与 0）+ plog 三路搜索均为空，且级别 0 时 stdout 洪泛 22.5 万行 GE WARNING 使 grep 成本很高；继续在日志侧投入的边际收益低于直接行为探测（后者一轮 7 个 shape 即完成定界）。
- **当前方案属哪档路线**：日志路线 `banned`；替代的行为探测 `recommended_fastpath`。

## 系统信息

- 设备：Ascend950PR（npu-smi 25.7.rc1.6）；CANN 9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`）。
- 被调试对象：ops-registry-invoke 算子仓自定义算子（host 源码 `op_host/arch35/<op>_tiling_arch35.cpp`，`#include "op_common/log/log.h"` 使用 OP_LOGI/OP_LOGE/OP_CHECK_*），经 `build.sh` 安装至 `${ASCEND_OPP_PATH}/vendors/<op>/`，由 `tests/aclnn/build/test_aclnn` 以 aclnn 两段调用链实跑。
- 日志位置实测：`~/ascend/log/run/plog/`（本次 ST 运行有 36259 字节 plog 但无算子 host 行）、`~/ascend/log/debug/plog/`（仅历史文件）；测试进程自身无 plog。

## 完整代码

> 注：以下命令块为「命令 + 结果注记」的蒸馏呈现（命令与输出均真实存在于开发轨迹，合并排版便于阅读）。

**① host 源码中的 OP_LOGI 日志点（grep 输出，事件流 verbatim）**：

```text
170:    OP_LOGI(context, "ThnnFusedGruCellGrad inputs: grad_hy dtype=%d, storage dtype=%d, has_bias=%d",
201:    OP_LOGI(context, "ThnnFusedGruCellGrad shapes: grad_hy (%ld, %ld), storage (%ld, %ld)",
230:    OP_LOGI(context, "Set currentWorkspace size:%zu, usrSize:%lu", currentWorkspace[0],
274:        OP_LOGI(context, "EMPTY_R tiling: usedCoreNum=%d aTotal=%ld aUbFactor=%ld aBigCoreCnt=%d"
280:        OP_LOGI(context, "EMPTY_A tiling: usedCoreNum=0 (all cores early-exit)");
422:    OP_LOGI(context, "ThnnFusedGruCellGrad tiling: axisNum=%d isTailR=0 usedCoreNum=%d isGroup=%d",
```

**② 两轮 slog 实测命令（事件流 verbatim）**：

```bash
# 第一轮（3 = ERROR-only，误当 INFO 用）
ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn /tmp/opencode/manifest_n2.json ../st_data_branch1 \
  2>&1 | grep -E "ThnnFusedGruCellGrad (tiling|shapes)|isGroup|aUbFactor=|rLoopCntTotal=|aLoopCntTotal=|usedCoreNum=|Set ws size" | head -20
## → 无输出

## 第二轮（0 = debug）
ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn /tmp/opencode/manifest_n2.json ../st_data_branch1 \
  > /tmp/opencode/n2_probe.log 2>&1
grep -E "ThnnFusedGruCellGrad (tiling|shapes|inputs)|Set currentWorkspace" /tmp/opencode/n2_probe.log | head -10
## → 仅 GE WARNING 注册行（op_impl_space_registry_v2_impl.cc MergeFunctions ... has been registered），无一条 OP_LOGI 行
```

**③ plog 三路搜索（事件流 verbatim）**：

```bash
## 按内容（本次 ST 运行的 run/plog）
grep -i "thnn\|tiling" ~/ascend/log/run/plog/plog-1708298_20260906164955809.log | head -10   # → 无输出（178 行总计）

## 按内容（全树，只命中历史 GE ERROR）
grep -rl "ThnnFusedGruCellGrad" ~/ascend/log/ | head
grep -rh "ThnnFusedGruCellGrad tiling" ~/ascend/log/ | tail -5
## → [ERROR] GE(...,python3):... [op_tiling_py.cc:1796]... TbeOptilingPyInterfaceNew: ErrorNo: 4294967295(failed)Op ThnnFusedGruCellGrad tiling failed

## 按进程 PID（本次测试进程 1711617）
find ~/ascend/log -newermt "2026-09-06 16:52" -name "plog*" | head; find / -name "plog-1711617*" | head -3
## → 无输出（该进程无 plog 文件）
```

## 预防

- 调试自定义算子 host tiling 值时，不要把"打开 slog"当第一手段；先评估行为探针（shape 扫描 + 分支二值信号）可行性。
- 若必须保留日志后路：开发期在 host 侧准备一个不依赖 slog 的可观测通道（如调试模式下把关键字段写入 workspace 或输出 tensor 供用例侧读回）。
- 记住 `ASCEND_GLOBAL_LOG_LEVEL` 语义（0=debug/1=info/2=warning/3=error），避免拿 3 当 INFO 浪费一轮。

## 验证信号

以下信号来自一次自定义算子 group 分支 ST 红基线的实机排查（Ascend950PR，CANN 9.0.0-beta.2）：开发者在分支路由排查中先后用两档日志级别与三路 plog 搜索尝试读取 host OP_LOGI，全部落空后改走行为探测。

- `ASCEND_GLOBAL_LOG_LEVEL=3 + ASCEND_SLOG_PRINT_TO_STDOUT=1` 单用例运行：stdout 共 199 行，grep 算子名仅冒烟/汇总行，无 host 日志（来源平台：950；证据等级：tool_observed）
- `ASCEND_GLOBAL_LOG_LEVEL=0 + ASCEND_SLOG_PRINT_TO_STDOUT=1`：stdout 共 225002 行，全部为 GE WARNING 注册类日志（`op_impl_space_registry_v2_impl.cc` MergeFunctions 行），无一条 OP_LOGI tiling/inputs/shapes/workspace 行（来源平台：950；证据等级：tool_observed）
- 本次 ST 运行生成的 run/plog（36259 字节/178 行）grep 算子名与 tiling 零命中（来源平台：950；证据等级：tool_observed）
- 全 `~/ascend/log` 树 grep `"ThnnFusedGruCellGrad tiling"` 仅命中历史 GE ERROR 行（op_tiling_py.cc，早前 python 进程），无 INFO 行（来源平台：950；证据等级：tool_observed）
- 按测试进程 PID find plog 文件：不存在（该 aclnn 测试进程未生成 plog）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅实测 CANN 9.0.0-beta.2 + Ascend950PR + `${ASCEND_OPP_PATH}/vendors` 安装的自定义算子 host 代码；其他 CANN 版本（含 9.x 正式版）行为可能不同，未验证。
- 本卡只覆盖"通常见两条打开方式"（级别 3/0 + PRINT_TO_STDOUT）与 plog 常规搜索；`ASCEND_GLOBAL_LOG_LEVEL=1`、`ASCEND_SLOG_PRINT_TO_FILE`、slog 配置文件、host 日志等级单独配置等路径**本开发会话 未尝试**，不能宣称其不可行。
- GE 自身 WARNING 日志确实进 stdout，不受本卡结论影响。

**未决项**：

- **优先项：复现并追根因到机制层**——自定义算子 host `OP_LOGI` 到底路由到哪个 sink、被哪一层（级别/开关/域配置）过滤，未验证。接续点：试 `ASCEND_GLOBAL_LOG_LEVEL=1`（INFO 档）、`ASCEND_SLOG_PRINT_TO_FILE=1`、检查 slog 配置文件与 vendor 算子日志开关，或向 CANN 侧确认自定义算子 host 日志的默认路由。
- 未尝试：检索 CANN 安装目录内其他日志位置（debug plog 之外的自定义 sink）。
## 坏实践

- 把 `ASCEND_GLOBAL_LOG_LEVEL=3` 当 INFO 用（实为 ERROR-only），浪费一整轮探测。
- 在 22.5 万行 GE WARNING 的 stdout 里反复 grep 期望中的 host 日志行，而不是及时换路线。
- 在没有先验证日志可达性的情况下，把"读 tiling 日志"写进排障计划的关键路径。

## 误诊

- 先入为主假设"设了 `ASCEND_SLOG_PRINT_TO_STDOUT=1`，host 侧 OP_LOGI 就该到 stdout"——该假设无工具证据支撑（agent 假设），已被两轮实测否定。
- 假设"plog 会记录本次运行"——本次 aclnn 测试进程没有 plog 文件；历史 plog 属早前 python/GE 进程（如实标注：此判断由 find 按时间/PID 搜索的空结果支撑，tool_observed）。

## 后续增量（2026-09-17，并行策展批：R2——TilingFunc 内 fprintf(stderr) 直打切分实参的可行观测路线）

R2（euclidean_norm 黑盒 float32 大 N 用例失败排查，Ascend950PR，CANN 9.1.0）：本卡遗留的"tiling 运行时值调试"缺口由 fprintf(stderr) 直打路线补上——**host TilingFunc 跑在评测进程内，stderr 无缓冲、直混入 TTK 控制台流**，按用例顺序可读，无需任何日志框架。相对本卡正卡的行为探测法，它拿到的是切分实参原值（key/isGroup/rUbF/rLoopCntTotal 等），适合"疑似走错分支/切分异常"类黑盒失败的第一手定位；静态推算切分参数是常见误诊源（本例曾据 65536 阈值误推 group 分支 rGroupCnt 超 uint16）。

**配方**：在 TilingFunc 的 `SetWorkspaceSize(context, ctx);` 与 `return ge::GRAPH_SUCCESS;` 之间插入临时打印块（重编译跑 TTK 后按用例 grep）：

```cpp
    SetWorkspaceSize(context, ctx);

    // [DBG-TILING] temporary tiling dump to stderr
    fprintf(stderr, "[DBG-TILING] coreNum=%d axisNum=%d isTailR=%d isGroup=%d isEmpty=%d key=%llu blockDim=%d\n",
            ctx.coreNum, ctx.axisNum, (int)ctx.isTailR, (int)ctx.isGroup, (int)ctx.isEmptyTensor,
            (unsigned long long)tilingKey, blockDim);
    fprintf(stderr, "[DBG-TILING] aUbF=%lld rUbF=%lld rLoopCntTotal=%lld usedCoreNum=%d rGroupCnt=%lld\n",
            (long long)ctx.aUbFactor, (long long)ctx.rUbFactor,
            (long long)ctx.rLoopCntTotal, ctx.usedCoreNum, (long long)ctx.rGroupCnt);

    return ge::GRAPH_SUCCESS;
```

字段选"分支判定 + 切分几何"全集（key/isGroup/isTailR/isEmpty + axisShape/stride/R + UbF/LoopCnt/GroupCnt），一次加打即可回答"走了哪条分支、切成几块、每块多大"。**用后清理**：整块还原后重编译，并 `grep -c "DBG-TILING"` 确认 0 残留（本例同时核对 kernel 侧修复标记仍在）。多进程（--pc>1）时多用例打印交错，须配合单进程复现；fprintf 属调试旁路，不应留在正式交付；device 侧（`__aicore__` kernel 内）无 stdio，不适用本路线（kernel 侧打印见 runtime 域 printf 系列卡）。

R2 验证信号：

- 打印生效：TTK 控制台输出 `[DBG-TILING] coreNum=56 axisNum=3 isTailR=0 isGroup=1 isEmpty=0 key=4 blockDim=36`、`rUbF=927 rUbFA=927`、`rLoopCntTotal=142`（两组大 N 用例）（来源平台：950；证据等级：tool_observed）。
- 打印值被下游消费：以 rLoopCntTotal=142/rGroupCnt=48/rPerCore=3 为输入的 numpy 结构模拟与 kernel 实跑输出吻合 7.6e-06%（来源平台：950；证据等级：tool_observed）。
- 清理验证：还原编辑后重编译 exit 0，`grep -c "DBG-TILING"`（tiling）== 0、kernel 侧修复标记 == 2（来源平台：950；证据等级：tool_observed）。