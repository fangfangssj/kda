---
type: "Runbook"
title: "TTK 黑盒失败最小复现：子集 CSV+dump"
description: "从黑盒全量 CSV 用 python csv 模块抽取失败用例子集（手写 CSV 会被引号/字段语义打回），--warmup=false 绕过预热，--dump full + NPU_DUMP_PATH 拿 input/output/golden .npy，numpy 三方对比定位精度根因的完整工作流。"
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# TTK 黑盒失败最小复现：子集 CSV+dump

## 现象

euclidean_norm 黑盒全量 636 例跑完出 70 个 FAIL 后，需要定位每个失败用例的根因。全量重跑一次约 9 分钟且大用例（820MB 级）生成输入极慢，无法逐例调试；直接在全量结果上推理又缺少 kernel 实际输出数据。需要一个"抽取失败用例子集 → 复现 → 拿到 input/output/golden 三方数据 → numpy 对比"的最小复现闭环。过程中踩了三个坑：手写 CSV 被 TTK 解析器打回（`Failed to process header [tensor_view_shapes] of row [0] with value [((5]`）、`--dump-on-fail` 没有产出任何 dump 文件、`precision_tolerances` 填 `1e-08` 被当作 shape 解析报错。

## 适用场景

- 工程形态：aclnn 自定义算子黑盒（TTK aclnn + golden plugin）失败用例的调试复现。
- API 路径：`ttk aclnn` CLI（`-i` 子集 CSV、`--dump full`、`--dump-format npy`、`--warmup=false`、`--pc=1 --device-whitelist 0`、`--plugin`、`-o`）；环境变量 `NPU_DUMP_PATH`。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.1.0（TTK 仓库本地部署）。
- 适用粒度：platform_wide（TTK 黑盒调试工作流，算子无关）。
- 路线优先级：recommended_fastpath（黑盒精度排障的标准第一步）。
- 明确不适用/未验证场景：非 TTK 的评测框架；需要计时性能的复现（须保留 warmup）；zip 打包用例模式（本流程走普通 CSV 模式）。

## 解决配方

完整工作流（四步，命令 verbatim 取自会话）：

**第 1 步：用 python csv 模块从全量 CSV 抽子集**（不要手写/手拼 CSV）：

```bash
cd <pkg>/tests/blackbox/ttk
python3 -c "
import csv
with open('blackbox_all.csv') as f:
    r = csv.DictReader(f); fn = r.fieldnames; rows = list(r)
wanted = ['aclnnEuclideanNorm_L1_567']        # 失败用例名列表
with open('one.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
    for row in rows:
        if row['testcase_name'] in wanted: w.writerow(row)
print('wrote one.csv')
"
```

抽取行的所有字段（含 `precision_tolerances` 空串、`input_data_ranges` 带引号逗号）原样保留。手写 CSV 的两类打回：tuple 字段内逗号未加引号 → `Failed to process header [tensor_view_shapes] of row [0] with value [((5]`；自造 `precision_tolerances=1e-08` → `Failed to process header [precision_tolerances] of row [0]`（全量 CSV 中该字段实际全为空串，容差走比较器默认）。

**第 2 步：带 dump 参数复现**（`--dump full` + `NPU_DUMP_PATH` + `--warmup=false`）：

```bash
source <docs>/<op>/develop/env.sh
VENDOR="$ASCEND_OPP_PATH/vendors/<Op>"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$TTK:$PWD/hdr_shim:$(dirname golden_plugin.py):$PYTHONPATH"
export NPU_DUMP_PATH="$PWD/repro_dump"
rm -rf repro_dump; mkdir -p repro_dump
timeout 200 python3 -m ttk aclnn -i one.csv --pc=1 --device-whitelist 0 \
  --plugin golden_plugin.py -o one_result.csv \
  --dump full --dump-format npy --warmup=false 2>&1 | grep -iE "PRECISION_STATUS|GOLD:|FAIL|PASS" | head
ls -la repro_dump/
```

产出四类 .npy：`<case>_input_tensor_0.npy`、`<case>_input_tensor_1.npy`（axes）、`<case>_output_0.npy`（kernel）、`<case>_golden_0.npy`。注意：`--dump-on-fail` 在该版本未产出文件（repro_dump 为空），须用 `--dump full`；dump 路径由 `NPU_DUMP_PATH` 控制。

**第 3 步：numpy 三方对比**：

```bash
cd repro_dump
python3 -c "
import numpy as np
x = np.load('<case>_input_tensor_0.npy')
g = np.load('<case>_golden_0.npy')
k = np.load('<case>_output_0.npy')
print('input x dtype/shape:', x.dtype, x.shape)
print('kernel dtype/shape:', k.dtype, k.shape)
print('golden dtype/shape:', g.dtype, g.shape)
print('kernel values:', k.ravel()[:12])
print('golden values:', g.ravel()[:12])
"
```

keepDims=false 时 kernel 输出可能是压缩 shape（如 `(12,)`）而 golden 保留维（如 `(1,2,1,1,6)`）——对比前先 ravel 或 reshape 对齐（本例曾报 `operands could not be broadcast together with shapes (12,) (1,2,1,1,6)`）。

**第 4 步（进阶）**：同型修改 shape 做 N 扫描/阈值二分——用第 1 步脚本改 `tensor_view_shapes` 字段后批量生成用例，一次跑出 PASS/FAIL 边界：

```bash
python3 -c "
import csv
fn=['testcase_name','api_name','tensor_view_shapes','tensor_dtypes','scalar_dtypes',
    'attributes','output_tensor_indexes','precision_tolerances','absolute_precision',
    'input_data_ranges','scalar_data_ranges','tensor_list_distribution']
# 按 fn 从源行取值，替换 tensor_view_shapes/output shape 为目标 N，
# precision_tolerances 保持源行原值（空串），写出 fp32_thr.csv 等扫描文件
"
timeout 200 python3 -m ttk aclnn -i fp32_thr.csv --pc=1 --device-whitelist 0 \
  --plugin golden_plugin.py -o fp32_thr_result.csv \
  --dump full --dump-format npy --warmup=false
```

## 设计决策

- 子集生成选 python `csv.DictWriter` 而非手拼字符串：TTK 的 CSV 解析器（`testcase_manager.py`）对带引号 tuple 字段严格，DictWriter 自动处理 quoting 且保留原字段语义（尤其空 tolerance）。
- dump 选 `--dump full` 而非 `--dump-on-fail`：实测 on-fail 不落文件；full 一次拿齐 input/output/golden，配合 `NPU_DUMP_PATH` 指到工作目录便于反复对比。
- 加 `--warmup=false`：复跑场景 warmup kernel 重编会崩，且精度复现不需要预热。
- 单进程 `--pc=1 --device-whitelist 0`：多进程复跑时 dump 与日志交错难读，单进程保证用例输出顺序可 grep。

## 系统信息

- Toolchain/SOC：Ascend950PR（Ascend950PR_9579）+ CANN 9.1.0；TTK（ops-test-kit，`ttk/cli/common.py` 定义 `--dump`/`--dump-format`；dump 落点由 `NPU_DUMP_PATH`/switches.root_path 决定）。
- 目录布局：`<pkg>/tests/blackbox/ttk/`（blackbox_all.csv、golden_plugin.py、result.csv、repro_dump/）；`env.sh` 提供 source 后的完整执行环境。

## 预防

- 任何"从全量结果转调试"的动作先做子集抽取 + dump 复现，再读数归因——不要在全量日志上直接推理数值根因。
- 改用例字段时只改目标字段（shape/dtype/range），tolerance 与属性字段原样复制源行。

## 验证信号

来源说明：euclidean_norm 黑盒失败修复会话——该工作流贯穿 70 例失败的归因（int32 溢出 dump 对比、float32 大 N 扫描、numpy 三方对比）。

- 子集复现成功：5 例失败用例 repro 运行输出逐例 `GOLD: 0.0% / PRECISION_STATUS: FAIL`，与全量结果一致（来源平台：950；证据等级：tool_observed）。
- dump 产出：`repro_dump/` 含 `<case>_golden_0.npy`(176B)/`<case>_input_tensor_0.npy`(368B)/`<case>_input_tensor_1.npy`(132B)/`<case>_output_0.npy`(176B)（来源平台：950；证据等级：tool_observed）。
- 手写 CSV 打回证据：`Failed to process header [tensor_view_shapes] of row [0] with value [((5]`（无引号逗号）与 `Failed to process header [precision_tolerances]`（自造 1e-08）（来源平台：950；证据等级：tool_observed）。
- 全量 tolerance 字段实测 `Counter({'': 636})` 全空（来源平台：950；证据等级：tool_observed）。
- N 扫描闭环：一次生成 7 个 N 值用例跑出 65536 PASS/131072+ FAIL 边界（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `--dump-on-fail` 不落文件为该 TTK 版本实测行为，后续版本可能修复；用前先 `ls` 确认。
- 820MB 级大用例 dump 极慢且占盘（输入生成 + npy 落盘双份），大 shape 用例优先改小 shape 同 dtype 扫描，只在最终确认时跑原 shape。
- keepDims 语义导致 kernel/golden shape 不一致时，须先对齐再比（ravel），否则 numpy 报 broadcast 错误。
- golden plugin 路径与 hdr_shim 的 PYTHONPATH 前缀是本工程布局约定，其他工程须按其 env.sh 调整。
- 其他 TTK 版本的 CLI 参数与 CSV 解析行为未验证（推断可迁移）。

**未决项**：

- `--dump-on-fail` 未产出文件的 TTK 侧原因未查证（本流程以 `--dump full` 规避）。
## 坏实践

- 手写/echo 拼 CSV——tuple 逗号与字段引号必错；用 csv 模块从源行复制。
- 只看 `precision` 百分比不看原始数值——0.0% 的成因（溢出单值差 vs 全量错位 vs shape 不匹配）必须靠 dump 数值区分。

## 误诊

- 无（流程卡点均由报错原文即时纠正）。
