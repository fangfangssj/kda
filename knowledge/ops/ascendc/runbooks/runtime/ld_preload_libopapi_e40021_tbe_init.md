---
type: "Runbook"
title: LD_PRELOAD libopapi 致 GE TBE 初始化失败
description: 为让 torch 测试解析 l0op 符号而 export LD_PRELOAD=libopapi.so 后，geir 图模式 RunGraph 报 Compilation_Error(E40021) "Pre-compile failed"（errormsg 为空，IGNORE_INFER_ERROR 无效）；开 ASCEND_SLOG_PRINT_TO_STDOUT=1 才见真错链：TEFUSION py_decouple "Launch dynamic-handle failed" → InitTbeFunc 失败 → GEInitialize failed。unset LD_PRELOAD 后恢复。
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# LD_PRELOAD libopapi 致 GE TBE 初始化失败

## 现象

在 BenchNoHarnessScatter 算子完成 InferDataType 修复、移除 run.sh 里的 `IGNORE_INFER_ERROR=1` 之后做全量回归：先跑 PyTorch ST（为解析 libcust_opapi.so 的 `l0op::Contiguous` 符号，shell 里设置了 `export LD_PRELOAD=$ASCEND_HOME_PATH/lib64/libopapi.so`），接着在同一环境跑 geir 图模式示例，RunGraph 失败：

```
....[XIR]: RunGraph failed, ret=-1
GE error message: [PID: ...] Compilation_Error(E40021): Failed to compile Op scatter1. oppath is $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/op_impl/ai_core/tbe/.../dynamic/bench_no_harness_scatter.py Pre-compile failed with errormsg/stack:  and optype is BenchNoHarnessScatter.
        Solution: See the host log for details, and then check the Python stack where the error log is reported.
TraceBack (most recent call last):
        Pre-compile op[scatter1] failed, oppath[...], optype[BenchNoHarnessScatter], taskID[2]. ...
```

特征：`Pre-compile failed with errormsg/stack:` 后面是**空的**（TBE 预编译异常信息没被带出来）；stdout 里还夹着 Python 3.13 multiprocessing forkserver 的 `TypeError: signal handler must be signal.SIG_IGN...` 噪音。值得注意的是 `export IGNORE_INFER_ERROR=1` 再跑一次**同样失败**（说明与 dtype 推导无关，此时 InferDataType 已注册）。开 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 把 GE 日志打到 stdout 后才看到完整错误链（GEInitialize 阶段就崩了）：

```
[ERROR] TEFUSION(...): [py_decouple.cc:51] Initialize Launch dynamic-handle failed.
[ERROR] TEFUSION(...): [python_adapter_manager.cc:43] Initialize Failed to initialize handle manager.
[ERROR] FE(...): [tbe_op_store_adapter.cc:1980] InitializeTeFusion:[GraphOpt][InitializeInner][InitTbeFunc] Failed to init tbe.
[ERROR] FE(...): [tbe_op_store_adapter.cc:1947] InitializeInner:[GraphOpt][InitializeInner][InitTeFusion]: Failed to initialize TeFusion.
[ERROR] FE(...): [op_store_adapter_manager.cc:80] InitializeAdapter:[SubGraphOpt][PreCompileOp][InitAdapter] InitializeAdapter adapter [tbe_op_adapter] failed! Ret [4294967295]
[ERROR] GE(...): [ops_kernel_manager.cc:98] Initialize: ErrorNo: 1343250441(There is no valid so about OpsKernelInfoStore or GraphOptimizer.)PluginManager InvokeAll failed.
...
[ERROR] GE(...): [ge_api.cc:219] GEInitialize: ... initial failed.
[XIR]: GEInitialize failed
```

`unset LD_PRELOAD` 后重跑：`GEInitialize success` + `RunGraph success` + `精度校验: PASS`；经 `bash run.sh --graph` 复跑同样 PASS。

## 适用场景

- 工程形态：同一 shell/CI 会话里先后跑"PyTorch/dlopen 类测试"与"GE 图模式（geir）示例"的自定义算子联调——前者常引入 `LD_PRELOAD=libopapi.so` 来补 l0op 符号解析，后者被其破坏。
- API 路径：环境变量 `LD_PRELOAD`、`ASCEND_SLOG_PRINT_TO_STDOUT`、`IGNORE_INFER_ERROR`；GE `GEInitialize`/`RunGraph`；TEFUSION `py_decouple` Python dynamic-handle（TBE 预编译通道）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0。
- 适用粒度：platform_wide——CANN 9.1.0 图模式 TBE 初始化对进程级 preload 敏感，与算子种类无关。
- 路线优先级：recommended_fastpath（跑图模式前 unset LD_PRELOAD 是零成本一步）。
- 明确不适用 / 未验证场景：eager/aclnn 路径（观测中不受此 preload 影响——同一环境下 PyTorch ST 795/795 照常通过）；其它被 preload 的库（只验证了 libopapi.so）。

## 根因

图模式（GE IR）编译期需要 TEFUSION 的 Python decouple 通道（`py_decouple.cc` 的 dynamic-handle，用于加载执行 vendor 包里的 TBE 适配器 py 文件做 op 预编译）。`LD_PRELOAD=libopapi.so` 把宿主 opapi 库强行注入到**所有**动态链接对象（包括 GE/FE/TEFUSION 自身的 so）之前，破坏了该 dynamic-handle 的初始化（符号/初始化顺序被抢占），`Initialize Launch dynamic-handle failed` 连锁导致 `InitTbeFunc` → OpsKernelInfoStore → `GEInitialize` 整体失败；RunGraph 阶段表现为 `Compilation_Error(E40021) Pre-compile failed` 且 errormsg 为空（错误发生在比 op 编译更早的 TBE 通道初始化，没有被 errormsg 通道捕获）。eager 路径不走 TBE 预编译，故同一 preload 下 PyTorch 测试仍全绿——形成"测试全过、图模式挂"的隐蔽组合。

## 解决配方

修复前后是同一命令在两种环境下的对照（before：带 LD_PRELOAD；after：unset 后）：

```bash
# before（失败）：为 torch 测试补符号而设置的 preload 残留
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/op_api/lib/:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH}
export LD_PRELOAD=$ASCEND_HOME_PATH/lib64/libopapi.so
cd operators/bench_no_harness_scatter/examples/build-geir && ./test_geir_bench_no_harness_scatter 0
# → RunGraph failed, ret=-1（E40021 Pre-compile failed，errormsg 为空）

# after（通过）：跑图模式前清掉 preload
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/op_api/lib/:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH}
unset LD_PRELOAD
cd operators/bench_no_harness_scatter/examples/build-geir && ./test_geir_bench_no_harness_scatter 0
# → GEInitialize success / RunGraph success / 精度校验: PASS
```

配套判别（失败时把真实错误链打出来）：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
./test_geir_bench_no_harness_scatter 0 2>/tmp/geir_stderr.log 1>/tmp/geir_stdout.log
grep -iE "error|exception|failed|cannot|import|TypeError|Pre-compile|compile" /tmp/geir_stdout.log | grep -v "RUNTIME\|task_fail"
# 命中 TEFUSION py_decouple "Launch dynamic-handle failed" 即本卡根因
```

## 设计决策

- 为什么用 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 而不是继续翻 plog：此故障下 plog 只剩 RUNTIME INFO，GE 的 ERROR 链在 stdout 通道；一次环境变量就能拿到全链路（TEFUSION → FE → GE），比猜日志位置快。
- 为什么没去深挖 preload 破坏 dynamic-handle 的符号级机制：`unset LD_PRELOAD` 后行为完全恢复且 torch 侧符号解析改由 LD_LIBRARY_PATH 承担即可，根因定位到"preload 注入破坏 TBE 通道初始化"已足够支撑处置决策；机制层留作遗留问题。
- 被放弃的尝试：`IGNORE_INFER_ERROR=1`（无效，证明与 dtype 推导无关）、`MULTIPROCESSING_DEFAULT_METHOD=fork`（无效，forkserver TypeError 只是伴生噪音）。
- 当前方案属 recommended_fastpath。

## 预防

- 会话内先跑 dlopen/torch 类测试再跑图模式示例时，跑图模式前固定 `unset LD_PRELOAD`；或者在两个干净 shell 里分别跑。
- 符号解析尽量用 LD_LIBRARY_PATH（把 CANN lib64 纳入搜索路径）而不是 LD_PRELOAD——preload 是全局注入，对 GE/TEFUSION 这类自带 so 插件体系的进程风险高（l0op 符号解析的环境配置见 [l0op::Contiguous 未定义符号卡](../compilation/libcust_opapi_l0op_contiguous_undefined_symbol.md)）。
- geir/GE 故障排查第一步开 `ASCEND_SLOG_PRINT_TO_STDOUT=1`，拿到 TEFUSION/FE 层错误再决定下一步。

## 验证信号

问题与验证来源：InferDataType 修复后的全量回归阶段，geir 图模式在带 preload 的环境下失败，同环境 unset 后恢复。

- 失败复现：带 `LD_PRELOAD=libopapi.so` 时 RunGraph failed，`Compilation_Error(E40021) Pre-compile failed with errormsg/stack:`（空），加 `IGNORE_INFER_ERROR=1` 复跑同样失败（排除 dtype 推导因素）（来源平台：950；证据等级：tool_observed）
- 诊断探针：`ASCEND_SLOG_PRINT_TO_STDOUT=1` 捕获 `TEFUSION py_decouple.cc:51 Initialize Launch dynamic-handle failed` → `InitTbeFunc Failed to init tbe` → `ErrorNo 1343250441 There is no valid so about OpsKernelInfoStore` → `GEInitialize failed`（来源平台：950；证据等级：tool_observed）
- 对照（preload 下 eager 正常）：同一 LD_PRELOAD 环境 PyTorch ST 795/795 PASS（TBE 通道仅图模式需要）（来源平台：950；证据等级：tool_observed）
- 修复验证：`unset LD_PRELOAD` 后 `GEInitialize success` + `RunGraph success` + `精度校验: PASS`；`bash run.sh --graph` 路径同样 `执行结果: PASS`（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本卡结论限于 `LD_PRELOAD=libopapi.so` 与 CANN 9.1.0 图模式 TBE 通道的组合；其它 preload 目标库未验证。
- torch 测试侧若无 LD_LIBRARY_PATH 兜底，直接 unset preload 可能让 l0op 符号再次解析失败——两侧要一起切（见 l0op::Contiguous 未定义符号卡的运行期配置）。
- 未验证、推断可迁移：CANN 8.x 的 py_decouple 行为；CI 全局 export 场景。

**未决项**：

- 优先项：LD_PRELOAD 破坏 py_decouple dynamic-handle 的符号级机制（冲突符号清单/初始化顺序）未深挖——如复现可对比 preload 前后 `dl_iterate_phdr` 与 TEFUSION so 的符号解析。
- "仅 LD_LIBRARY_PATH（不 preload）能否同时满足 torch 侧符号解析"未单独验证（与 l0op::Contiguous 未定义符号卡的遗留问题同源）。
## 坏实践

- 把 `LD_PRELOAD=libopapi.so` 写进 run.sh/CI 全局环境并长期保留——本卡即为该做法的直接反例。
- 看到 `Pre-compile failed with errormsg/stack:`（空）就去逐行读 vendor 包里的 TBE 适配器 py 源码——错误源头在更早的 TBE 通道初始化，与 py 文件内容无关。
- 被 stdout 里的 multiprocessing `TypeError: signal handler ...` 带偏去修 Python 环境——那是 forkserver 噪音，与本根因无关（fork 方式复跑仍失败即为反证）。

## 误诊

- 初判一："移除 IGNORE_INFER_ERROR 后暴露了新的编译错误，可能是 InferDataType 修复引入的回归"——带 `IGNORE_INFER_ERROR=1` 复跑同样失败，排除与 dtype 推导的因果关系（推断，被对照实验推翻）。
- 初判二："Pre-compile 失败是 vendor 包里 TBE 适配器内容问题"——打开该文件排查方向错误；真实错误在 GEInitialize 阶段的 dynamic-handle（推断，被 SLOG 输出修正）。
