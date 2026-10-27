---
type: "Runbook"
title: "TTK kernel 模式 VEC_ERROR 而 aclnn 正常的判别"
description: "TTK kernel 直调模式对自定义算子全 shape（含最小 shape 与动态编译）报 VEC_ERROR，而同一 kernel 二进制在 ttk aclnn 路径精度 PASS 时，优先判别为 kernel-mode launch 路径问题而非 kernel 缺陷；用 tiny 复跑、动态编译、aclnn 交叉验证三步快速隔离。"
tags: [task.debugging, topic.architecture, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# TTK kernel 模式 VEC_ERROR 而 aclnn 正常的判别

## 现象

在 Ascend950PR（CANN 9.1.0，ops-test-kit commit b31db58）上为一个已通过黑盒精度验收（aclnn 模式 399/399 PASS）的 scatter 类自定义算子采集上板性能时，按规定命令走 TTK kernel 直调模式：

```bash
python3 -m ttk kernel -i perf_scatter_cases.csv -d false -b release --pc=1 --task-prof true --run 3 --plugin golden.py -o result_perf.csv
```

6 条性能 shape 全部失败：1 条被 tiling 拒绝（`perf_status=INVALID_TILING`），其余 5 条执行期报错，日志原文：

```
2026-09-04 20:42:01,620 [ERROR] [2889362 perf_scatter_03_int32_none OnBinProfiling] [rts_interface.py:848]: VEC_ERROR encountered

Performance result:                Comparison result:                 Memory check:
DYN_PERF: DYN_OFF                  DYN_GOLD: DYN_OFF                  DYN_MEM: UNKNOWN
CST_PERF: CST_OFF                  CST_GOLD: CST_OFF                  CST_MEM: UNKNOWN
BIN_PERF: UNKNOWN                  BIN_GOLD: NO_OUTPUT                BIN_MEM: UNKNOWN
STATUS: FAIL                       PRECISION_STATUS: FAIL
```

关键特征：

- 编译/tiling 均成功（`Compilation Result: SUCC`、`BlockDim: 1`、`Workspace Bytes: (0,)`），VEC_ERROR 发生在 OnBinProfiling/OnDynProfiling 执行期；
- **二进制模式（`-b release`）与动态编译（`-d true`）同样失败**——排除"陈旧 release 二进制"假设；
- **缩小到最小 shape（data (8,4)、indices/updates (3,4)、int32、dim=0、reduce=update）仍然 VEC_ERROR**——排除 shape/规模/dtype 相关性；
- **同一 kernel 二进制在 `python3 -m ttk aclnn` 路径下正常**：tiny 用例精度 PASS，6 条性能 shape 中 5 条精度 PASS。

当时代码状态：kernel 为修复精度问题后的新版（fp32 workspace 累加路径，tiling 结构体含 `useFp32Acc`/`dataNumel` 字段），黑盒精度已全绿，本任务为只采集不修复。

## 适用场景

- 工程形态：TTK（ops-test-kit）kernel 直调模式跑自定义算子用例（性能采集或功能验证）；对照物为 ttk aclnn 模式。
- API 路径：`python3 -m ttk kernel -i <csv> [-d true|false] [-b release|false]` 与 `python3 -m ttk aclnn -i <csv>`；被测算子为已安装到 `$ASCEND_OPP_PATH/vendors/` 的自定义算子。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.1.0，TTK commit b31db58（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——判别规则作用于测试/采集工作流（不改 kernel 代码），对任何用 TTK kernel 直调模式验证自定义算子的会话适用。
- 路线优先级：recommended_fastpath（判别链本身是隔离问题的最短路径，先判别再决定是否动 kernel）。
- 明确不适用 / 未验证场景：aclnn 模式也失败的场景（那是 kernel/注册/环境问题，本判别规则不适用）；VEC_ERROR 只出现在个别 shape 的场景（应优先怀疑该 shape 的越界/对齐，而非 launch 路径）；其他 TTK 版本与其他芯片（未验证，推断判别思路可迁移）。

## 判别方法

判别链按"每步排除一个假设"组织，从便宜到贵排序：

1. **tiny 复跑**（排除 shape/规模/dtype 相关）：构造最小 shape 用例重跑。若 tiny 也 VEC_ERROR，问题与数据规模无关。
2. **动态编译复跑**（排除陈旧二进制）：`-d true -b false` 重新从安装目录源码编译执行。若动态编译同样 VEC_ERROR，不是 release 二进制过期。
3. **aclnn 交叉验证**（隔离 kernel 与 launch 路径）：用同一 kernel 二进制走 `ttk aclnn` 模式跑同一 tiny 用例。**aclnn PASS 而 kernel 模式 VEC_ERROR ⇒ kernel 代码本身可正确执行，问题限定在 TTK kernel-mode 直接 launch 路径（tiling 数据编排 / GM 缓冲布局 / launch 机制与 kernel 的交互）**，此时不要急于改 kernel。
4. （可选）**双 vendor 二进制普查**：确认 kernel 模式加载的是不是预期 vendor 的真实二进制（见同批多 vendor 同 hash kernel 冲突检测知识）。

tiny 复跑探针（kernel 模式，最小 int32 用例，verbatim）：

```bash
cat > /tmp/tiny.csv <<'EOF'
testcase_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,attributes,input_data_ranges
tiny_int32_ovw,bench_harness_scatter,"((8,4),(3,4),(3,4))","('int32','int32','int32')","('ND','ND','ND')","((8,4),)","('int32',)","('ND',)","{'dim': 0, 'reduce': 'update'}","((-1000,1000),(0,7),(-1000,1000))"
EOF
cd /tmp/ops-test-kit && timeout 200 "$PY" -m ttk kernel -i /tmp/tiny.csv -t tiny_int32_ovw \
  -d true -b false --pc=1 --task-prof false --run 1 --seed 42 \
  --plugin <golden.py> -o /tmp/perf_tiny3/r.csv > /tmp/perf_tiny3/full.log 2>&1
grep -nE "Tiling Data Parsed Dict|Tiling Key|VEC_ERROR|Kernel Name|BlockDim|Workspace" /tmp/perf_tiny3/full.log | head -40
```

aclnn 交叉验证探针（同一二进制、同一 tiny shape；注意 aclnn CSV 的 `tensor_view_shapes` 必须含输出张量共 4 个、属性名用 aclnn 层的 `reduceOptional`，verbatim）：

```bash
cat > /tmp/tiny_aclnn.csv <<'EOF'
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,tensor_formats,attributes,output_tensor_indexes,input_data_ranges
tiny_aclnn_int32,aclnnBenchHarnessScatter,"((8,4),(3,4),(3,4),(8,4))","('int32','int32','int32','int32')","('ND','ND','ND','ND')","{""dim"": 0, ""reduceOptional"": ""update""}","(3,)","((-1000,1000),(0,7),(-1000,1000))"
EOF
cd /tmp/ops-test-kit && timeout 300 "$PY" -m ttk aclnn -i /tmp/tiny_aclnn.csv \
  --pc=1 --task-prof false --run 1 --warmup false \
  --plugin <golden.py> -o /tmp/aclnn_smoke/r.csv 2>&1 | grep -iE "Summary|PASS|FAIL|PRECISION_STATUS"
```

观测结果对照（本例）：

| 探针 | kernel 模式 | aclnn 模式 |
|---|---|---|
| tiny int32 (8,4) | `VEC_ERROR encountered`（OnDynProfiling，DYN_PERF: VEC_ERROR） | `PRECISION_STATUS: PASS`，PassRate 100% |
| 性能 shape（-b release） | 5/5 FAIL（BIN_PERF UNKNOWN） | 5/6 PASS（1 条 tiling 拒绝，与 VEC_ERROR 无关） |

## 设计决策

- 为什么先跑 tiny 再跑动态编译再跑 aclnn：三步成本递增（tiny 秒级、动态编译分钟级、aclnn 需要另写 CSV），且每步独立排除一个假设（shape 相关 / 二进制陈旧 / kernel 缺陷），任一步命中即可短路后续排查。
- 为什么用 aclnn 作为"kernel 功能正常"的判据：aclnn API（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）与 kernel 直调最终执行同一 vendor 二进制，aclnn 精度 PASS 即证明二进制本身可正确执行；差异只剩 launch 路径（tiling 编排、GM 布局、launch 机制）。
- 为什么不据此改 kernel：本判别规则的产出是"问题不在 kernel 代码"，改 kernel（如绕开某类 tiling 字段）属于在未定位机制前掩盖 harness 问题的投机修复；正确出口是改用可用的采集/验证路径并在报告中透明记录偏离（同批 aclnn 采集配方知识）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR_9579（V100）单卡，CANN 9.1.0（安装目录名 cann-9.2.0，`compiler/version.info: Version=9.1.0`）；TTK `/tmp/ops-test-kit` commit b31db586d760ca50dddc91df17122941366de4e4。
- 被测算子：已安装的自定义 scatter 算子（kernel 含 fp32 workspace 累加路径，tiling 结构体 38 个 int64 字段）。
- 相关文件/目录：性能用例 CSV（kernel 模式与 aclnn 模式各一份）、TTK 运行日志（run.log）、`$ASCEND_OPP_PATH/vendors/` 下两个注册该算子的 vendor 目录。

## 预防

- 在 kernel 直调模式跑批量用例前，先用一条最小 shape 用例冒烟——tiny 也失败时立即进入判别链，不要在批量结果里逐条找规律。
- 看到 kernel 模式 VEC_ERROR 时，先问"aclnn 模式过不过"，再决定是否进入 kernel 代码调试；两者表现相反时把精力放在 launch 路径差异上。
- 保留两份模式各自的 CSV（kernel 口径 attributes 用 `reduce`、aclnn 口径用 `reduceOptional` 且张量组含输出），交叉验证时可直接复用。

## 验证信号

以下信号来自该 scatter 算子上板性能采集会话：kernel 模式批量失败后按判别链逐级复跑、并以 aclnn 模式交叉验证同一二进制。

- kernel 模式（-d false -b release）冒烟 int32 用例：编译 SUCC、执行期 `VEC_ERROR encountered`（rts_interface.py:848）、STATUS FAIL / PRECISION_STATUS FAIL（来源平台：950；证据等级：tool_observed）。
- kernel 模式全 6 条性能 shape：1 条 INVALID_TILING + 5 条 FAIL（bin_perf_us=UNKNOWN，VEC_ERROR）（来源平台：950；证据等级：tool_observed）。
- 动态编译（-d true -b false）同一用例：`VEC_ERROR encountered`（OnDynProfiling），排除陈旧 release 二进制（来源平台：950；证据等级：tool_observed）。
- 最小 tiny 用例（(8,4) int32/fp32/fp16 reduce=update）kernel 模式全部 VEC_ERROR（dyn_op_tiny_int32_ovw，DYN_PERF: VEC_ERROR）（来源平台：950；证据等级：tool_observed）。
- 同一二进制 aclnn 模式 tiny 用例：PRECISION_STATUS PASS、PassRate 100.00%（来源平台：950；证据等级：tool_observed）。
- aclnn 模式 6 条性能 shape：5 条精度 PASS（1 条 tiling 拒绝另计）（来源平台：950；证据等级：tool_observed）。
- 临时禁用存根 vendor 后 kernel 模式仍 VEC_ERROR（加载真实 375KB kernel），排除存根冲突（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本判别规则只覆盖"kernel 模式全 shape 一致失败 + aclnn 全过"的形态；若 kernel 模式仅个别 shape VEC_ERROR，应优先排查该 shape 的越界/对齐/tiling 尾块，而非 launch 路径。
- "aclnn PASS ⇒ kernel 无缺陷"只在该 aclnn 用例确实覆盖了同一 dtype/reduce/dim 组合时成立；交叉验证要用与失败用例同参数的 tiny 版本。
- 观测基于 CANN 9.1.0 + TTK b31db58 + 950 单卡；其他版本/芯片未验证（判别思路推断可迁移，结论本身未验证）。
- VEC_ERROR 的机制层根因未追到（见遗留问题），不排除特定 kernel 写法（如依赖 tiling buffer 特定字段语义）与 kernel-mode launch 的交互贡献。

**未决项**：

- 【优先项】复现并把 VEC_ERROR 追到机制层：TTK 日志的 `Tiling Data Parsed Dict` 中 `aI`/`perBufBytes` 两个字段为垃圾值（约 7.9e12 / 3.2e13）而其余 36 个 int64 字段全部正确——重点验证 kernel 侧 `pipe_.InitBuffer(buf_[i], td_->perBufBytes)` 读到巨额字节是否直接触发向量核异常，以及这两个字段在 kernel-mode launch 的 tiling buffer 编排中为何损坏。
- custom_ops vendor 的 `custom_ops_impl/dynamic` 安装源码落后于仓库（缺 `useFp32Acc`/`dataNumel`）与 host tiling 新结构体的布局错位，是否解释 `-d true` 路径的 VEC_ERROR 未确证。
- TTK kernel-mode 与 aclnn launch 路径在 tiling 编排/GM 布局上的具体差异清单未梳理（需读 TTK 源码 `rts_interface.py`/launch 模块）。
## 坏实践

- 看到 kernel 模式全量 VEC_ERROR 就开始改 kernel 代码（如删字段、绕路径）——本例中 kernel 在 aclnn 下功能精度均正常，改 kernel 属于在未定位机制前乱动。
- 只跑 `-b release` 就断定"二进制陈旧"并反复重装——动态编译 `-d true` 一步即可排除该假设。
- 用 `--sim-report` 等仿真产物冒充实测继续"验证"——仿真不能证明上板行为。

## 误诊

- "安装目录动态源码陈旧（tiling 结构体比仓库少字段）导致 host tiling 与 kernel 布局错位"：该假设被 `-b release` 路径否定——release 二进制与 host tiling 同代一致仍然 VEC_ERROR；且 TTK 日志按 38 字段新结构解读 tiling 数据，除 2 个字段外其余全部正确。结构体错位在本例中未确证（见遗留问题）。
- "6KB 存根 vendor 抢占了二进制解析"：禁用存根 vendor 后加载真实 kernel 仍 VEC_ERROR，排除（该冲突真实存在但不是本现象根因，检测方法见同批多 vendor 同 hash kernel 冲突检测知识）。
