---
type: "Runbook"
title: TTK 多进程 warmup 竞态整行假 FAILURE
description: TTK kernel 模式全量跑用例时，多 worker 进程并发"删除-重编译" warmup.o 的竞态会让撞上窗口的用例整行 FAILURE（traceback 指向 kernel_meta/warmup.json 缺失）；判别特征是全字段 FAILURE + 空 precision_metrics，正确补救是抽子集以 --pc 1 单进程重跑后合并，不要用 --warmup false。
tags: [task.testing, topic.implementation.debugging, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# TTK 多进程 warmup 竞态整行假 FAILURE

## 现象

在一个 softmax 基准算子（opbuild 工程，Ascend 950，CANN 9.1.0）的 Branch-0 白盒验收阶段：developer 已修复 kernel 并自测 TTK kernel 模式全量 980 条用例 `total=980 pass=980 fail=0 skip=0`（GATE PASS）。随后验收者重建 kernel（`$ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/.../<op>.o` 时间戳晚于上一轮 result.csv）并复跑同一全量命令：

```
ttk kernel --csv <branch-0 的 980 条用例 CSV> ...（默认参数，多进程）
```

复跑结果却变成 `total=980 pass=945 fail=35 skip=0`（GATE FAIL）——kernel 代码没有变，35 条用例"新增失败"。逐行检查 result.csv 发现这 35 行与真实精度失败完全不同：`dyn_tiling_key=FAILURE`、`dyn_block_dim=FAILURE`、`memory_oob_status=FAILURE`、`dyn_precision=FAILURE`、`precision_metrics` 为空 dict——**整行所有字段都是 FAILURE 字面量**，且 `dyn_tiling_time_us` 字段里内嵌完整 Python traceback：

```
Traceback (most recent call last):
  File "$TTK/ttk/core_modules/npu/op/profiling.py", line 665, in do_profiling
    result = rts_profiling(device, param)
  File "$TTK/ttk/core_modules/npu/op/rts_sequence.py", line 52, in rts_profiling
    return online_obj.do()
  File "$TTK/ttk/core_modules/npu/op/rts_sequence.py", line 124, in do
    self._device.warmup(self._switches)
  File "$TTK/ttk/core_modules/runtime/rts_interface.py", line 595, in warmup
    self._launch_helper_kernel(os.path.join(switches.kernel_meta, "warmup.o"), args=(0,))
FileNotFoundError: [Errno 2] No such file or directory: '$TTK/kernel_meta/warmup.json'
RuntimeError: Profiling Sequence of mode dynamic failed
```

（`$TTK` 为 ops-test-kit 安装根。）

## 适用场景

- 工程形态：TTK kernel 模式（`ttk kernel --csv ...`）全量跑算子白盒/精度用例的执行与验收；任何用 `parse_result.py` 解析 result.csv 做 GATE 判定的评测流程。
- API 路径：`ttk kernel` CLI（`--pc/--process-count`、`--warmup`、`--no-prof/--co`）；TTK 内部 `instance_refactor._compile_help_kernels` / `tbe_interface.compile_warmup_kernel` / `rts_sequence.do`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR）+ CANN 9.1.0 + TTK（ops-test-kit，Rev e559b7f）（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`workflow_only`（评测工具链行为，不涉算子 kernel 语义）。
- 路线优先级：`recommended_fastpath`（对假失败行的处置配方：识别 → 子集单进程重跑 → 合并；无更优替代）。
- 明确不适用/未验证场景：单进程 `--pc 1` 本身不触发该竞态；`--no-prof/--co`（compile_only）会整体跳过设备执行，不能用于拿精度信号，与本问题无关；其他 TTK 版本未验证。

## 判别方法

对 result.csv 的 FAIL 行做三步判别（Python `csv.DictReader` 即可）：

1. **整行字段检查**：若该行 `dyn_tiling_key`/`dyn_block_dim`/`memory_oob_status`/`dyn_workspaces`/`dyn_precision` 等全部等于字面量 `FAILURE`、`precision_metrics` 为空 dict——是 TTK 基础设施错误，不是 kernel 精度失败（真实精度失败行的 tiling_key/`block_dim`/oob 均为正常值，只有 `dyn_precision=NO_OUTPUT/FAILURE` 等少数字段异常）。
2. **traceback 提取**：读该行 `dyn_tiling_time_us`（或 `dyn_tiling_data`）长字段，若含 `FileNotFoundError: ... kernel_meta/warmup.json` + `RuntimeError: Profiling Sequence of mode dynamic failed`——命中 warmup 竞态。
3. **失败行占比 sanity**：与上一轮同 kernel 的结果比对，"新增失败"且分布在随机 case 区间（与 shape/dtype/dim 无规律相关）——竞态是时间窗口性的，不挑用例特征。

## 根因

TTK 源码级机制（本次实读 ops-test-kit 的 `ttk/core_modules/operator/tbe_interface.py`）：

```python
    def compile_warmup_kernel(self):
        switches = get_global_storage()
        opc = self._tbe_opc
        if switches.warmup:
            obj_file = os.path.join(switches.kernel_meta, "warmup.o")
            if os.path.exists(obj_file):
                os.remove(obj_file)
            ...
```

`compile_warmup_kernel` 在 `switches.warmup` 为真时**先 `os.remove` 已存在的 warmup.o 再重编译**；而每个 worker 进程 setup 时都会调 `_compile_help_kernels()`（`instance_refactor.py:87`：`if not self.switches.compile_only: self._compile_help_kernels()`）。多进程跑全量用例时，某 worker 中途执行 `rts_sequence.do()` 第 124 行 `self._device.warmup(self._switches)` 要启动 `kernel_meta/warmup.o`，恰好撞上另一 worker 的"删除-重编译"窗口 → `FileNotFoundError` → `RuntimeError: Profiling Sequence of mode dynamic failed` → 整 case 级联标 FAILURE。时间窗口性解释了为什么 980 条里只有 35 条中招、且与用例特征无关。

## 解决配方

1. **抽取假失败行名单**：按判别方法筛出全部整行 FAILURE 的 testcase_name（本例 35 条）。
2. **构造子集 CSV**：从原输入 CSV 按 testcase_name 过滤出子集（本例 35 条 + 表头）。
3. **单进程重跑子集**（关键参数 `--pc 1`，warmup 保持默认 true）：

```
ttk kernel --csv <子集 CSV> --pc 1 ...（其余参数与全量一致，输出到独立 result 文件）
```

4. **合并**：以全量 result.csv 为底，用子集重跑的真实行替换对应假失败行；再跑 `parse_result.py <merged> --expect-total <N>` 复核。

本例实测：35 条子集 `--pc 1` 重跑 `total=35 pass=35 fail=0 skip=0`（GATE PASS）；合并后 `total=980 pass=980 fail=0 skip=0`，且确认无任何行残留 FAILURE/NO_OUTPUT/INVALID_TILING 字样。

**禁止用 `--warmup false` 规避**：见同批负知识卡（TTK `--warmup false` 会引入更大面积的 RTS_BINARY_FAILURE 假失败，其结果不可作精度信号）。

## 设计决策

- **为何选"子集 + `--pc 1` 重跑"而不是全量单进程重跑**：全量 980 条耗时约 20+ 分钟，单进程更慢；假失败行已可精确识别，只重跑受影响子集（2 分多钟）再合并，效率最高且保留全量其余行的真实信号。
- **为何不用 `--warmup false`**：曾按源码推理"关 warmup 即无 warmup.o 访问、无竞态"（`bridge.py:93` 确认 `--warmup false` 能正确置 `sw.warmup=False`），实测全量复跑出现 144 条 `RTS_BINARY_FAILURE`，其中 119 条在 warmup=true 时是 PASS——推理被实测否定，该路线 banned。
- **为何不预创建 warmup 文件挡删除**：`compile_warmup_kernel` 无条件先 remove 再重编，挡不住；且不改 TTK 源码的前提下无旋钮可关单 worker 的重编译。
- **当前方案路线档位**：`recommended_fastpath`。

## 系统信息

- 工具链：TTK（ops-test-kit 安装根 `$TTK`，Rev e559b7f）、CANN 9.1.0、SOC Ascend950PR；`source docs/<op>/develop/env.sh` 取得 `$TTK`/`$PY`/`PYTHONPATH=$TTK:$ASCEND_TOOLKIT_HOME/python/site-packages`。
- 相关文件：`operators/<pkg>/tests/whitebox/ttk/result.csv`（解析对象）、`skills/whitebox-tester/parse_result.py`（GATE 解析）、`$TTK/ttk/core_modules/operator/tbe_interface.py`（竞态源）、`$TTK/ttk/core_modules/npu/op/rts_sequence.py`（warmup 调用点）、`$TTK/kernel_meta/`（warmup.o/json 所在）。

## 预防

- 全量 TTK kernel 模式跑批后的 FAIL 行，先做"整行 FAILURE + traceback"判别再下结论，避免把基础设施假失败当成 kernel 回归（本例若直接采信复跑结果，会把已修好的 kernel 误判为回归）。
- 验收/复跑前若刚重建过 kernel `.o`，预期 result.csv 需重新生成；对比两轮结果时注意 kernel 二进制时间戳与 result.csv 时间戳的先后关系。
- 长套件（约 20+ 分钟）超过单命令超时限制时，后台 `setsid`/`nohup` 启动 + 轮询 result.csv 行数（每 2 分钟约 80-90 行）。

## 验证信号

问题与验证来源：一次算子白盒验收复跑中出现"kernel 未变却新增 35 条失败"，验收者对 result.csv 逐行取证并在 TTK 源码定位机制后，以子集单进程重跑验证。以下均为本会话实测：

- 复跑（多进程默认参数）：`expect-total=980 total=980 pass=945 fail=35 skip=0`，35 行全部整行 FAILURE，`dyn_tiling_time_us` 内嵌 `FileNotFoundError: kernel_meta/warmup.json` traceback（来源平台：950；证据等级：tool_observed）。
- 竞态源码取证：`tbe_interface.py` `compile_warmup_kernel` 中 `if os.path.exists(obj_file): os.remove(obj_file)`；`instance_refactor.py:87` 每个 worker 进程 setup 调 `_compile_help_kernels`（来源平台：950；证据等级：tool_observed）。
- 子集 `--pc 1` 重跑：`total=35 pass=35 fail=0 skip=0`（GATE: PASS, EXIT=0），35 条全 PASS（来源平台：950；证据等级：tool_observed）。
- 合并后全量复核：`total=980 pass=980 fail=0 skip=0`，`rows with any FAILURE/NO_OUTPUT/INVALID_TILING in precision fields: 0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该判别与补救只针对 warmup 竞态这一种基础设施假失败；真实精度失败（部分字段正常、仅 precision 异常、有具体 max_diff/mismatch）不适用，必须走 kernel 修复。
- `--pc 1` 重跑前提是 kernel 二进制与输入 CSV 不变；若期间重建过 kernel，子集结果与全量其余行不属同一被测对象。
- 并发窗口大小与 worker 数、机器负载相关：失败行数不固定（本例 35/980），也可能 0 条。
- 未验证，推断可迁移：同 TTK 版本族的其他 `ttk kernel` 多进程全量跑批；TTK 修复 `compile_warmup_kernel` 的 remove-rebuild 竞态后本卡失效。

**未决项**：

- TTK 侧根治（`compile_warmup_kernel` 加文件锁/跳过已存在 warmup、或 worker 间共享一次编译）未尝试，本卡只给使用侧规避。
- 竞态窗口的精确定量（多少 worker/何种负载必现）未测。
## 坏实践

- 看到 GATE FAIL 就当 kernel 回归去改代码——本例 35 条全是基础设施假失败，kernel 本身 980/980 通过。
- 用 `--warmup false` "绕开"竞态——实测引入 144 条 RTS_BINARY_FAILURE 假失败（119 条与 warmup=true 结果相反），比竞态本身更不可信（详见同批负知识卡）。
- 用 `--no-prof/--co` 拿精度——`compile_only=True` 会走 `prof_compile_only_end` 直接跳过设备执行与精度比对，根本没有精度信号。

## 误诊

- 初见 35 条失败时曾怀疑"kernel 重建后 result.csv 未刷新"或"新引入的 kernel 回归"：取 kernel `.o` 时间戳（23:40）晚于 result.csv（23:37）只能证明需要重跑，重跑后仍 35 条失败才排除该解释；随后曾按"TTK 基础设施偶发 flakiness"猜测，直到读源码找到 `os.remove` 竞态才闭环。
- `--warmup false` 方案一度被源码推理论证为可行（warmup 与精度检查解耦），实测 144 条失败推翻——源码推理必须过实测（该推理过程为 agent 转述，实测信号见验证信号）。
