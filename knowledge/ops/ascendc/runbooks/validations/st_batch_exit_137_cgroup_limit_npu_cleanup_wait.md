---
type: "Runbook"
title: ST 批量 exit 137：容器 cgroup 限额与 NPU 清理等待
description: 容器内单进程批量跑 torch_npu ST 被 SIGKILL（exit 137）且 free 显示内存充足时，先查 cgroup 限额（cgroup v1 读 memory.limit_in_bytes/oom_control，free 是宿主假象）；修复为多进程分批 + 批间等待 NPU 异步清理 + 失败减半重试，无法完成的批次如实上报并由 C++ ST 全量覆盖。
tags: [task.testing, topic.memory, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T02:20:40Z'
updated_at: '2026-09-17T02:20:40Z'
platforms: ["950"]
---

# ST 批量 exit 137：容器 cgroup 限额与 NPU 清理等待

## 现象

在 Ascend950PR 容器内执行自定义算子 BenchNoHarnessCummin 的 PyTorch ST 最终精度验收：`python3 test.py --npu --lib ./build/libtorch_adapter.so --level l1`（600 条用例，最大单条 5.57 亿元素 fp16）。进程运行约 4.5 分钟、跑到约第 183 条时被系统杀死：

```
START: Wed Jul 15 09:57:22 AM CST 2026
/bin/bash: line 1: 149551 Killed                  python3 test.py --npu ... --level l1 > l1_run.log 2>&1
EXIT_CODE=137
```

日志文件只有 1 行 torch_npu 的 TensorFactories 警告。更迷惑的是：被 kill 后立即重启的进程在 NPU 初始化阶段（连 golden 自测都没跑）就再次被杀；而单用例执行完全正常，L0（30 条）此前也能单进程跑完。`free -h` 显示总内存 754GB、可用 685GB，看上去毫无内存压力；`npu-smi` 显示 HBM 仅 5.4GB/128GB。

## 适用场景

- 工程形态：自定义算子的 PyTorch ST（torch_npu）批量执行 / 最终精度验收；任何在 Docker 容器内大批量跑 NPU 测试的场合。
- API 路径：`python3 test.py --npu`（torch.ops.load_library 加载适配层）；诊断用 `npu-smi info`、`/sys/fs/cgroup/memory/*`、`ps aux`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，cgroup v1 容器（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——容器化 NPU 测试环境的执行策略，与具体算子无关。
- 路线优先级：recommended_fastpath（环境受限时的标准跑法）。
- 明确不适用 / 未验证场景：单条用例本身超过容器限额（分批无解，须换环境或裁剪用例）；cgroup v2 环境（限额文件路径为 `/sys/fs/cgroup/memory.max`，机制相同但未在本轨迹验证）；非容器裸机。

## 根因

进程受容器 cgroup v1 内存限额约束，与 `free` 显示的宿主机内存无关。实测证据链：

```
$ cat /sys/fs/cgroup/memory/memory.limit_in_bytes
34359738368            # 32GB
$ cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes
34359738368            # 峰值恰好打满限额
$ cat /sys/fs/cgroup/memory/memory.oom_control
oom_kill_disable 0
under_oom 0
oom_kill 36            # 该容器累计已发生 36 次 OOM kill
```

单进程顺序执行 600 条用例时，torch_npu 侧张量与 native 缓存逐步累积越过 32GB（第 183 条用例本身只有 26x29=754 元素，是累积效应不是单条大用例）。另有加剧因素：同一 NPU/容器上还挂着他人遗留的测试进程 `test_aclnn_bench_no_harness_conv_2d`（占 803MB RSS / 285MB HBM、NPU Util 100%），压缩可用内存并拖慢 kill 后的 NPU runtime 异步清理——所以 kill 后立即重启会在 init 阶段再次被杀，等待 10-15 秒让清理完成即可恢复。

## 解决配方

诊断三步（复现 exit 137 后先做）：

```bash
# 1. 容器真实限额（不要看 free）
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
cat /sys/fs/cgroup/memory/memory.oom_control        # oom_kill 计数
# 2. NPU 残留/他人进程
npu-smi info                                         # Process id / HBM-Usage / Util
ps aux | grep -E "test_aclnn|python3.*test.py" | grep -v grep
# 3. 单用例对照（确认不是用例本身的问题）
python3 test.py --npu --lib ./build/libtorch_adapter.so --case <单条小用例>
```

修复：多进程分批执行器（两个脚本配合，verbatim；个人安装路径已泛化为占位符）。`run_batches.sh` 负责分批、批间等待与失败减半重试：

```bash
#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# 分批执行 L1 测试, 每批独立进程, 批间等待 NPU 清理。
# ----------------------------------------------------------------------------------------------------------
set -u

BATCH_SIZE=25
BATCH_WAIT=8
RETRY_WAIT=15
MAX_RETRIES=2
LEVEL="${1:-l1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="<算子包安装根>/op_api/lib/:${LD_LIBRARY_PATH}"      # 例：$ASCEND_OPP_PATH/vendors/<OpType>_custom/op_api/lib
export ASCEND_HOME_PATH="<CANN 安装根>"                                     # 例：~/Ascend/cann-9.1.0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# 获取总用例数
TOTAL=$(python3 -c "
import csv, os
path = os.path.join('..', 'testcases', f'aclnnBenchNoHarnessCummin_${LEVEL}_test_cases.csv')
with open(path) as f:
    r = csv.reader(f); next(r)
    print(sum(1 for _ in r))
")
echo "总用例数: $TOTAL, 批大小: $BATCH_SIZE"
echo "START: $(date)"

RESULT_DIR="_batch_results"
rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

BATCH_NUM=0
START=0
while [ "$START" -lt "$TOTAL" ]; do
    COUNT=$BATCH_SIZE
    REMAINING=$((TOTAL - START))
    if [ "$COUNT" -gt "$REMAINING" ]; then
        COUNT=$REMAINING
    fi
    BATCH_NUM=$((BATCH_NUM + 1))
    END=$((START + COUNT))
    OUTPUT="$RESULT_DIR/batch_${BATCH_NUM}.json"

    echo ""
    echo "=================================================="
    echo "批次 $BATCH_NUM: 用例 $((START+1))~$END (共 $COUNT 条)"
    echo "=================================================="

    # 重试逻辑
    SUCCESS=0
    for RETRY in $(seq 0 $MAX_RETRIES); do
        if [ "$RETRY" -gt 0 ]; then
            echo "  重试 $RETRY/$MAX_RETRIES (等待 ${RETRY_WAIT}s)..."
            sleep "$RETRY_WAIT"
            # 重试时减半批大小
            HALF=$((COUNT / 2))
            if [ "$HALF" -gt 0 ]; then
                COUNT=$HALF
                END=$((START + COUNT))
                echo "  减半批大小: $COUNT"
            fi
        fi

        python3 _worker.py \
            --lib ./build/libtorch_adapter.so \
            --level "$LEVEL" \
            --start "$START" \
            --count "$COUNT" \
            --output "$OUTPUT" 2>&1 | tail -5

        EXIT_CODE=${PIPESTATUS[0]}
        if [ "$EXIT_CODE" -eq 0 ] && [ -f "$OUTPUT" ]; then
            SUCCESS=1
            echo "  批次 $BATCH_NUM 成功"
            break
        else
            echo "  批次 $BATCH_NUM 失败 (exit=$EXIT_CODE)"
            rm -f "$OUTPUT"
        fi
    done

    if [ "$SUCCESS" -eq 0 ]; then
        echo "  [ERROR] 批次 $BATCH_NUM 所有重试均失败, 跳过"
        # 记录失败批次
        echo "{\"batch\": $BATCH_NUM, \"start\": $START, \"count\": $COUNT, \"status\": \"BATCH_FAILED\"}" > "$OUTPUT"
    fi

    START=$END
    echo "  等待 ${BATCH_WAIT}s (NPU 清理)..."
    sleep "$BATCH_WAIT"
done

echo ""
echo "=================================================="
echo "所有批次完成, 汇总结果..."
echo "=================================================="
echo "END: $(date)"

# 汇总结果
python3 -c "
import json, os, glob
result_dir = '$RESULT_DIR'
all_results = []
batch_failures = []
for f in sorted(glob.glob(os.path.join(result_dir, 'batch_*.json'))):
    with open(f) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        all_results.extend(data)
    elif isinstance(data, dict) and data.get('status') == 'BATCH_FAILED':
        batch_failures.append(data)

passed = sum(1 for r in all_results if r.get('result') == 'PASS')
failed = sum(1 for r in all_results if r.get('result') == 'FAIL')
skipped = sum(1 for r in all_results if r.get('result') == 'SKIP')

print(f'总计: {len(all_results)}')
print(f'通过: {passed}')
print(f'失败: {failed}')
print(f'跳过: {skipped}')
if batch_failures:
    lost = sum(b['count'] for b in batch_failures)
    print(f'批次失败: {len(batch_failures)} (丢失 {lost} 条)')
    for b in batch_failures:
        print(f'  批次 {b[\"batch\"]}: 用例 {b[\"start\"]+1}~{b[\"start\"]+b[\"count\"]}')

# 保存完整结果
with open('test_results.json', 'w') as f:
    json.dump({'total': len(all_results), 'passed': passed, 'failed': failed,
               'skipped': skipped, 'batch_failures': batch_failures,
               'results': all_results}, f, indent=2)
print(f'结果已保存: test_results.json')

if failed > 0:
    print('\\n失败用例:')
    for r in all_results:
        if r.get('result') == 'FAIL':
            print(f'  - {r[\"name\"]}: {r.get(\"error\", \"\")}')
'
```

`_worker.py` 每批一个独立进程，逐用例强制回收（完整，verbatim）：

```python
#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Worker: 执行指定范围的测试用例, 输出 JSON 结果。
# 由 run_batches.sh 调用。
# ----------------------------------------------------------------------------------------------------------
import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden_cummin
from compare import compare_results
from test import get_all_test_cases, generate_input


def run_one(tc: dict) -> dict:
    """执行单个测试用例, 返回结果字典。执行后立即释放内存。"""
    name = tc["name"]
    shape = tc["shape"]
    dtype = tc["dtype"]
    dim = tc["dim"]
    data_range = tc["data_range"]

    seed = 20260715 + hash(name) % 100000
    try:
        input_cpu = generate_input(shape, dtype, data_range, seed=seed)
    except Exception as e:
        return {"name": name, "result": "FAIL", "error": f"data_gen: {e}",
                "shape": "x".join(str(d) for d in shape),
                "dtype": tc["dtype_str"], "dim": dim}

    try:
        golden_values, golden_indices = compute_golden_cummin(input_cpu, dim)
    except Exception as e:
        del input_cpu
        return {"name": name, "result": "FAIL", "error": f"golden: {e}",
                "shape": "x".join(str(d) for d in shape),
                "dtype": tc["dtype_str"], "dim": dim}

    if not torch.npu.is_available():
        del input_cpu, golden_values, golden_indices
        return {"name": name, "result": "SKIP", "error": "NPU unavailable",
                "shape": "x".join(str(d) for d in shape),
                "dtype": tc["dtype_str"], "dim": dim}

    passed = False
    input_npu = actual_v = actual_i = av_cpu = ai_cpu = None
    try:
        input_npu = input_cpu.to("npu")
        actual_v, actual_i = torch.ops.bench_no_harness_cummin.forward(input_npu, dim)
        av_cpu = actual_v.cpu()
        ai_cpu = actual_i.cpu()
        passed = compare_results(golden_values, av_cpu, golden_indices, ai_cpu)
    except Exception as e:
        return {"name": name, "result": "FAIL", "error": f"npu: {e}",
                "shape": "x".join(str(d) for d in shape),
                "dtype": tc["dtype_str"], "dim": dim}
    finally:
        del input_npu, actual_v, actual_i, av_cpu, ai_cpu
        del input_cpu, golden_values, golden_indices
        gc.collect()
        torch.npu.empty_cache()

    return {"name": name, "result": "PASS" if passed else "FAIL",
            "shape": "x".join(str(d) for d in shape),
            "dtype": tc["dtype_str"], "dim": dim}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--level", default="l1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.ops.load_library(args.lib)

    st_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = get_all_test_cases(st_dir, args.level)
    end = args.start + args.count if args.count > 0 else len(cases)
    batch = cases[args.start:end]

    results = []
    for i, tc in enumerate(batch):
        idx = args.start + i
        r = run_one(tc)
        r["index"] = idx
        status = r["result"]
        shape_str = r.get("shape", "?")
        dtype_str = r.get("dtype", "?")
        print(f"[{idx+1}] {r['name']}: {status} ({shape_str}, {dtype_str})", flush=True)
        if status == "FAIL" and "error" in r:
            print(f"    error: {r['error']}", flush=True)
        results.append(r)
        time.sleep(0.02)

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"结果已保存: {args.output} ({len(results)} 条)", flush=True)


if __name__ == "__main__":
    main()
```

结果按批写 `_batch_results/batch_N.json`，父进程只做聚合，不持有任何 NPU 上下文。

## 设计决策

- 为什么多进程而不是单进程内清理：单进程内 `del + gc.collect() + torch.npu.empty_cache()` 不足以阻止累积（首版 batch_runner 在 init 阶段就被杀）；子进程退出是唯一可靠的内存复位手段，且某批被杀只损失该批。
- 为什么批间固定等待 8s、重试前等 15s：kill 后 NPU runtime 的清理是异步的，立即重启会在初始化阶段再次被杀（实测多次复现）；等待后单进程 NPU init 探针即恢复。
- 为什么失败时批大小减半：失败批内可能混有一条触发限额的大用例，减半把它隔离到小子批；减半到 6 条仍失败的批说明该批用例在本容器内无法完成，跳过并如实记录（本轨迹 L1 有 358 条属此类，验收报告明确标注"未完成，由 C++ ST 全量验证覆盖"）。
- 尝试过 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`：单独使用不能解决（仍被 kill），仅作为配套保留。
- 当前方案路线档位：recommended_fastpath（容器内存受限环境下的标准执行策略）。

## 系统信息

- 硬件/软件：Ascend950PR（HBM 128GB，本测试仅用 ~5.4GB），CANN 9.1.0，cgroup v1 容器（限额 32GB），torch_npu。
- 目录：`operators/bench_no_harness_cummin/tests/st/torch/`（test.py / golden.py / compare.py / batch_runner.py / mp_runner.py / run_batches.sh / _worker.py / _batch_results/）。
- 干扰项：同机他人并行评测任务遗留的 `test_aclnn_bench_no_harness_conv_2d` 进程（npu-smi 可见，PID/进程名/占用一目了然）。

## 预防

- 容器内跑 NPU 大批量测试前先 `cat /sys/fs/cgroup/memory/memory.limit_in_bytes` 摸清真实限额，不要用 `free -h` 评估。
- 批量执行器一律按"每批独立子进程 + 批间 sleep + 失败减半重试 + 结果按批落盘"的结构写。
- 遇到 exit 137 先查 oom_kill 计数与 npu-smi 残留进程，再考虑代码问题。

## 验证信号

以下信号来自该算子最终精度验收会话（多轮单进程失败 → 多进程分批收敛），Ascend950PR + CANN 9.1.0：

- 修复前：单进程 L1 三次被杀（分别跑到 0 条 init 期 / 183 条 / init 期），EXIT_CODE=137；单用例正常（来源平台：950；证据等级：tool_observed）。
- 根因证据：`memory.limit_in_bytes=34359738368`、`max_usage_in_bytes=34359738368`、`oom_kill=36`；`free` 显示 754GB（宿主）与容器实际限额无关（来源平台：950；证据等级：tool_observed）。
- 干扰证据：npu-smi 显示他人遗留进程占 285MB HBM / 803MB RSS、Util 100%；等待清理后单进程 NPU init 探针恢复（来源平台：950；证据等级：tool_observed）。
- 中间方案：mp_runner（50 条/批、无减半重试）12 批中 8 批被 kill，仅完成 150/600——固定大批不足（来源平台：950；证据等级：tool_observed）。
- 修复后：run_batches.sh（25 条/批 + 8s 批间等待 + 减半重试）聚合已完成批 242/242 PASS；L0 30/30 PASS（单进程完整跑完）；L1 剩余 358 条在本容器内无法完成，验收报告如实记录"未完成（NPU 内存限制），由 C++ ST 全量验证（633/633，Real NPU + CPU golden）覆盖"（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 32GB 是本容器配置；其他环境限额不同，批大小需按限额与单用例峰值重调。
- 减半到 6 条/批仍失败的批被跳过——本配方不解决单条用例超过限额的场景。
- 多进程每批都要重新初始化 torch_npu（本环境每批约 40s 开销），总耗时显著增加；用例数少时不如单进程。
- cgroup v2 的限额文件为 `/sys/fs/cgroup/memory.max`，机制相同但未验证，推断可迁移。
- 批间等待时长（8s/15s）依赖 NPU runtime 清理速度，机器繁忙时可能需要加大。

**未决项**：

- torch_npu 单进程内存累积的确切来源（allocator 缓存策略 vs 引用泄漏）未深挖；若能定位，或可在进程内回收从而省掉多进程开销。
- 本容器 oom_kill=36 说明是长期共性问题，建议环境侧调高限额或限制并行评测任务数。
## 坏实践

- 用 `free -h` 判断容器内可用内存并据此设计批量策略。
- 进程被 kill 后不等待 NPU 清理立即重跑（init 阶段再被杀，误判为"NPU 坏了"）。
- 把 OOM kill 当 NPU 设备/驱动故障排查（HBM 充足、单用例正常时应先怀疑宿主侧 cgroup）。
- 验收报告隐瞒未完成用例数（本工程做法是如实列出 358 条未完成并给出 C++ ST 全量覆盖的替代证据）。

## 误诊

- 曾怀疑第 183 条用例本身过大：查 CSV 该用例 shape 仅 26x29=754 元素，排除（是累积效应）。
- 曾怀疑 NPU HBM 不足：npu-smi 显示 5.4/128GB，排除。
- 曾怀疑内存引用泄漏可在进程内回收：首版 batch_runner（进程内 del + gc + empty_cache）在 init 阶段即被杀，证明必须进程隔离。
