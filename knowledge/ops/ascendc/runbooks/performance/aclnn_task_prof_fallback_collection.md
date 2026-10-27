---
type: "Runbook"
title: "TTK kernel 模式不可用时 aclnn 采集 AiCore 实测耗时"
description: "kernel 直调模式对某算子全 shape 失败时，改用 ttk aclnn --task-prof 采集真实上板耗时的完整配方：aclnn 口径 CSV 写法（含输出张量与 ABI 属性名）、采集命令、op_perf_us/api_perf_us 指标口径与 msprof 限制。"
tags: [task.testing, topic.performance.analysis, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# TTK kernel 模式不可用时 aclnn 采集 AiCore 实测耗时

## 现象

在 Ascend950PR（CANN 9.1.0，TTK commit b31db58）上为一个已通过黑盒精度验收的 scatter 类自定义算子做上板性能采集，任务规定的路径是 TTK kernel 直调模式（`python3 -m ttk kernel ... -d false -b release --task-prof true`）。但 kernel 模式在该算子全部 shape 上执行期 VEC_ERROR（含最小 shape 与动态编译，且已用 aclnn 交叉验证排除 kernel 缺陷——判别过程见同批 TTK kernel 模式 VEC_ERROR 判别知识），`bin_perf_us`/`dyn_perf_us` 均为 UNKNOWN，一条耗时都采不到。任务又禁止改 kernel/tiling/用例、禁止伪造数据。当时需要一条能拿到**真实上板耗时**的采集路径。

## 适用场景

- 工程形态：自定义算子上板性能采集/目标对照（性能报告、达成率计算）；前提是 kernel 模式路径不可用而 aclnn 路径可用（同一 kernel 二进制在 aclnn 下功能精度正常）。
- API 路径：`python3 -m ttk aclnn -i <csv> --pc=1 --task-prof true --run 3 --warmup true --seed 42 --plugin <golden.py> -o result.csv`；结果 CSV 的 `api_perf_us`/`op_perf_us`/`precision_status` 列与 run.log 的 profiling 表。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.1.0，TTK b31db58（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——纯采集工作流配方，不改任何 kernel/tiling 代码。
- 路线优先级：safety_fallback——这是 kernel 模式不可用时的替代采集路径；相对代价：拿不到 msprof 逐 pipe 占比（见失效边界的 fallback 声明），bound 对照只能标注 N/A 或以设计+时延特征推断。
- 明确不适用 / 未验证场景：kernel 在 aclnn 模式也失败（先修 kernel/环境）；需要逐 pipe 占比（aiv_mte2_ratio 等）作为硬性产物的任务（aclnn --task-prof 不产出）；其他 TTK 版本（未验证，参数与列名可能变化）。aclnn CSV attributes 键须用部署头 ABI 参数名的一般规则见[aclnn CSV attributes 键用 ABI 名卡](../validations/ttk_aclnn_csv_keys_follow_deployment_header_abi.md)。

## 解决配方

四步：改写 CSV 为 aclnn 口径 → 校验 → 采集 → 按口径取数。

1. **CSV 改写要点**（相对 kernel 模式 CSV 的差异）：
   - `api_name` = aclnn 接口名（如 `aclnnBenchHarnessScatter`），不再用 `op_name`；
   - `tensor_view_shapes`/`tensor_dtypes`/`tensor_formats` 必须包含**全部输入 + 全部输出**（3 输入 1 输出的算子就是 4 个张量组），输出位置由 `output_tensor_indexes`（如 `"(3,)"`）指示；
   - `attributes` 用 **aclnn 层的属性名**（本例 `reduceOptional`，不是算子定义层的 `reduce`）——用错属性名会导致精度错配或校验失败；
   - `input_data_ranges` 与输入一一对应（范围元组带尾逗号的形式照黑盒 CSV 抄）。
2. **校验**：`"$PY" -m ttk aclnn -i <csv> --validate`，确认 `N total, N valid, 0 invalid`。
3. **采集**：`--task-prof true --run 3 --warmup true`，完整命令见完整代码。
4. **按口径取数**：
   - `op_perf_us` ＝ **设备侧 AiCore 任务时长**（run.log profiling 表 `<Op>_AiCore` 行的 Avg），对标 kernel 模式的 Task Duration/DYN_PERF 口径，用作实测耗时；
   - `api_perf_us` ＝ host 侧 aclnn API 时长（profiling 表 `aclnnXxx` 行 Avg），只作参考，**不要**当 kernel 耗时报达成率；
   - `--run 3` 下 profiling 表 `# of Calls=2`、Min≈Max 即稳态，取 Avg。

## 设计决策

- 为什么允许从 kernel 模式切换到 aclnn 模式：任务的硬约束是"真实上板数据 + 不改 kernel/用例/阈值"，Approach 里的 kernel 模式是建议路径；同一 kernel 二进制在 aclnn 路径下于 NPU 上真实执行（AiCore 任务时长即设备侧执行时间），满足"上板实测"的本质要求。切换必须在报告里透明声明（模式、口径、切换原因），不能默默替换。
- 为什么不用 `--sim-report` 补 bound：仿真报告（trace_core*.json）不是 msprof 实测，用它充当"实测 bound"违背如实记录原则；bound 列记 N/A 并注明采集路径限制。
- 为什么保留 kernel 口径 CSV：把两种口径的 CSV 都留在交付目录，便于 kernel 模式修复后直接复跑对照。
- 当前方案路线档位：safety_fallback（fallback 声明：相对 kernel 模式 + msprof 的完整采集，本路径损失逐 pipe bound 数据，仅有 AiCore 总时长与 host API 时长）。

## 系统信息

- 硬件/软件：Ascend950PR_9579（V100）单卡 device 0，AiCore 1650MHz；CANN 9.1.0（安装目录名 cann-9.2.0，`compiler/version.info: Version=9.1.0`）；TTK `/tmp/ops-test-kit` commit b31db586d760ca50dddc91df17122941366de4e4；执行环境 `source docs/develop/env.sh`（含 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`）。
- golden：复用算子包的 TestSpec 纯 numpy golden（`--plugin docs/develop/golden.py`），未改。
- 相关文件：`operators/<Op>_package/tests/perf/perf_scatter_aclnn.csv`（aclnn 口径用例）、`perf_scatter_cases.csv`（kernel 口径，供复跑对照）、`result_perf.csv`（结果）、`perf_run.log`（含 profiling 表的运行日志）。

## 完整代码

aclnn 口径性能用例 CSV（6 条关键 shape，3 输入 + 1 输出，verbatim）：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,tensor_formats,attributes,output_tensor_indexes,input_data_ranges
perf_scatter_01_fp16_add,aclnnBenchHarnessScatter,"((65536,128),(8192,128),(8192,128),(65536,128))","('float16','int32','float16','float16')","('ND','ND','ND','ND')","{""dim"": 0, ""reduceOptional"": ""add""}","(3,)","((-1,1,),(0,65535,),(-1,1,))"
perf_scatter_02_fp32_add,aclnnBenchHarnessScatter,"((131072,64),(524288,64),(524288,64),(131072,64))","('float32','int64','float32','float32')","('ND','ND','ND','ND')","{""dim"": 0, ""reduceOptional"": ""add""}","(3,)","((-1,1,),(0,131071,),(-1,1,))"
perf_scatter_03_int32_none,aclnnBenchHarnessScatter,"((1024,1024),(1024,1),(1024,1),(1024,1024))","('int32','int64','int32','int32')","('ND','ND','ND','ND')","{""dim"": 1, ""reduceOptional"": ""update""}","(3,)","((-1000,1000,),(0,1023,),(-1000,1000,))"
perf_scatter_04_bf16_amax,aclnnBenchHarnessScatter,"((2048,512),(512,512),(512,512),(2048,512))","('bfloat16','int32','bfloat16','bfloat16')","('ND','ND','ND','ND')","{""dim"": 0, ""reduceOptional"": ""amax""}","(3,)","((-1,1,),(0,2047,),(-1,1,))"
perf_scatter_05_fp16_none,aclnnBenchHarnessScatter,"((1024,1024),(1024,512),(1024,512),(1024,1024))","('float16','int32','float16','float16')","('ND','ND','ND','ND')","{""dim"": 1, ""reduceOptional"": ""update""}","(3,)","((-1,1,),(0,1023,),(-1,1,))"
perf_scatter_06_fp16_add_r5,aclnnBenchHarnessScatter,"((256,64,8,4,4),(32,64,8,4,4),(32,64,8,4,4),(256,64,8,4,4))","('float16','int32','float16','float16')","('ND','ND','ND','ND')","{""dim"": 0, ""reduceOptional"": ""add""}","(3,)","((-1,1,),(0,255,),(-1,1,))"
```

校验 + 采集 + 取数（verbatim）：

```bash
source docs/develop/env.sh; export TTK=/tmp/ops-test-kit
cd /tmp/ops-test-kit
# 1) 校验
"$PY" -m ttk aclnn -i <Op>_package/tests/perf/perf_scatter_aclnn.csv --validate
# 2) 采集（结果与日志落盘）
rm -rf /tmp/perf_aclnn_run && mkdir -p /tmp/perf_aclnn_run
timeout 900 "$PY" -m ttk aclnn -i <Op>_package/tests/perf/perf_scatter_aclnn.csv \
  --pc=1 --task-prof true --run 3 --warmup true --seed 42 \
  --plugin docs/develop/golden.py -o /tmp/perf_aclnn_run/result_perf.csv \
  > /tmp/perf_aclnn_run/run.log 2>&1
# 3) 结果表（精度 + 两个口径的耗时）
"$PY" -c "
import csv
rows=list(csv.DictReader(open('/tmp/perf_aclnn_run/result_perf.csv')))
for r in rows:
    print(r['testcase_name'], r['precision_status'], r['perf_status'],
          r['api_perf_us'], r['op_perf_us'])
"
# 4) 稳态与设备侧明细：profiling 表（每 case 一段）
grep -nE "<Op>_AiCore|aclnn<Op> |aclrtLaunchKernelWithHostArgs" /tmp/perf_aclnn_run/run.log
```

profiling 表样例（本例某 case，注意三行三个口径）：

```
                                    Name   Total/us     Avg/us     Min/us     Max/us # of Calls
                aclnnBenchHarnessScatter      78.48      39.24      14.56      63.92          2
aclnnBenchHarnessScatterGetWorkspaceSize      40.34      20.17       5.59      34.75          2
              aclrtLaunchKernelWithHostArgs      36.17      18.09       8.43      27.74          2
              BenchHarnessScatter_AiCore     443.38     221.69     221.02     222.36          2
```

## 预防

- 写 aclnn 口径 CSV 时先抄一份已验证的黑盒 CSV 当模板（张量组含输出、属性名用 aclnn 层名称），再替换 shape/dtype/属性值——本例首版漏了输出张量，校验直接报 `Field [tensor_dtypes] of [case] is invalid.`（CASE_FIELD_AMBIGUOUS），报错字段名（dtypes）与真实缺失（输出 shape）不一致，容易误导。
- 采集前先用一条 tiny 用例走通 aclnn `--task-prof`，确认 `op_perf_us`/`api_perf_us` 列与 profiling 表都出数，再上批量。
- 性能报告里固定记录：模式（kernel/aclnn）、口径（AiCore 任务时长）、`--run` 次数、计时调用数与 Min/Max、命令行与结果文件路径。

## 验证信号

以下信号来自该 scatter 算子上板性能采集会话：kernel 模式确认不可用后，按本配方完成 aclnn 口径采集与报告。

- tiny 用例 aclnn `--task-prof`：`precision_status=PASS`、`api_perf_us=50.071`、`op_perf_us=4.11`、`perf_status=PASS`（来源平台：950；证据等级：tool_observed）。
- CSV 校验 6/6 通过（`Validate Complete: 6 total, 6 valid, 0 invalid`）（来源平台：950；证据等级：tool_observed）。
- 全量采集 `--run 3`：Total 6 / PASS 5 / FAIL 1（唯一 FAIL 为 tiling 拒绝用例，非采集问题）；5 条可测 shape 精度全 PASS、`op_perf_us` 全部出数（来源平台：950；证据等级：tool_observed）。
- profiling 表逐 case 输出 `<Op>_AiCore` Avg/Min/Max（# of Calls=2，Min≈Max），`op_perf_us` 与表中 Avg 一致（来源平台：950；证据等级：tool_observed）。
- msprof 限制确认：输出目录无 PROF_* 目录、run.log 中 msprof/pipe ratio 提及 0 次——aclnn `--task-prof` 不产出逐 pipe 占比（来源平台：950；证据等级：tool_observed）。
- 交付验收：性能报告非空（142 行）通过 `test -s`，报告内逐条给出实测/目标/达成率/口径/tilingKey 与方法偏离透明声明（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本配方的前提是 kernel 二进制在 aclnn 路径可正常执行；若 aclnn 也失败（如 tiling 拒绝、注册缺失），本路径同样采不到数。
- `op_perf_us`/`api_perf_us` 列名、`# of Calls=2` 的计数行为基于 TTK b31db58；其他版本列名或计数可能不同（未验证）。
- **fallback 声明（相对代价）**：相对 kernel 模式 + msprof 的完整采集，本路径**拿不到逐 pipe 占比（aiv_mte2_ratio/aiv_vec_ratio 等）**，bound 对照只能记 N/A 或以"kernel 设计 + 每元素时延特征"推断并显式标注；需要逐 pipe 实测的任务不适用（参照路线：kernel 模式 --task-prof 的 msprof 产物；来源平台：950）。
- 异步 launch 下 `api_perf_us`（host 侧）可远小于 `op_perf_us`（设备侧），两者不可互换口径。

**未决项**：

- 通过 aclnn 路径获得 msprof 逐 pipe 占比的手段未找到（`--sim-report` 为仿真，未采用）；若后续 TTK 支持 aclnn 模式落 msprof 产物，可补齐 bound 对照。
- `--run 3` 下 profiling 只计 2 次调用的计数规则（warmup 是否 excluded）未查 TTK 源码确认。
## 坏实践

- aclnn CSV 漏写输出张量（shapes 3 个、dtypes 4 个）——校验报 `Field [tensor_dtypes] ... is invalid`，报错字段与真实缺项不一致，别按字面改 dtypes。
- 用 `api_perf_us` 当 kernel 实测耗时算达成率——那是 host 侧 aclnn API 时长。
- 用 `--sim-report` 仿真产物冒充 msprof 实测 bound。
- 切换采集路径后不在报告里声明——模式与口径的偏离必须透明，否则实测数据失去可信度。

## 误诊

- "首版 aclnn CSV 报 tensor_dtypes 无效是 dtype 写法问题"：实际是 `tensor_view_shapes` 少了输出张量（3 个 shape 对 4 个 dtype），补上第 4 个输出 shape 后同一 dtype 写法即通过。
