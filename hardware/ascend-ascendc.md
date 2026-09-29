# Hardware Profile: `Ascend · Ascend C`

昇腾 NPU 上的 Ascend C 核函数开发（直调算子）。配套 starter prompt：`prompts/ascendc-flow.md`。将本档案复制到独立任务工作区，命名为 `hardware-profile.md`。

## 1. 目标硬件与工具链

- 器件：Atlas A2/A3 训练系列（910B / 910C 等），以 `npu-arch` 技能确认 SoC Version 与 NpuArch 代际。
- 基线：CANN ≥ 8.x（`cat $ASCEND_HOME/version.cfg` 或 `npu-smi info`），Python + torch_npu。
- 环境清单：开工前先跑 `cann-env-setup`、`ascendc-env-check` 技能的检查项，结果记录进 `docs/draft.md`。

## 2. 实现语言与 API

- Ascend C 核函数（`extern "C" __global__ __aicore__`），工程从 `ascendc-direct-invoke-template` 技能脚手架起步。
- Tiling 结构、API 选型遵循 `ascendc-api-best-practices`；写法约束遵循 `ascendc-code-review` 技能的检视规则。
- 禁止绕过 tiling 直接写死 shape（除非契约声明固定 shape 且记录代价）。

## 3. 正确性验证约定

- 参考实现：torch_npu（首选同算子 ACLNN 实现）或 CPU golden data。
- 判定：按 `ops-precision-standard` 技能的精度标准选择比对方式（逐元素阈值 / 双精度统计量 / ulp）。
- 必须覆盖：tiling 分支全覆盖（多核切分边界、UB 切分余数分支）——参考 `ascendc-tiling-design` 技能的分支覆盖清单。
- 验证命令：ST 用例按 `ascendc-st-design` 技能生成；需要的 UT 按 `ascendc-ut-develop` 技能。
- 精度不达标时走 `ascendc-precision-debug` 技能定位（CPU/NPU 逐级比对）。

## 4. 性能测量约定

- 计时与采集统一走 `ops-profiling` 技能（`msprof_profile_run.sh` 标准采集 / `--compare` 对比 / `--quick` 快速对比），输出 `performance.json` 与 markdown 对比报告。
- 基线集合：ACLNN 同名算子 / 被替换的现有实现。
- `benchmark.csv`：`candidate,algo_variant,tiling,shape,dtype,aiv_time_us,bandwidth_gbps,baseline_us,speedup,date`。

## 5. Profiling 与归因

- 算子级瓶颈定位：`ops-profiling` 技能 + `msprof_perf_summary.py` 瓶颈分析。
- Device 侧日志 / 黑匣子：`msnpureport-toolkit` 技能；故障 / 崩溃场景用 `msaicerr`（如已安装）与 `ascendc-crash-debug` 技能。
- 运行时报错与错误码：`ascendc-runtime-debug` 技能；SetFlag/WaitFlag 同步问题用 `ascendc-sync-audit` 技能审计。

## 6. 技能映射

| 需求 | 技能 | 位置 |
|---|---|---|
| 芯片型号 / 架构代际 / 条件编译 | npu-arch | `skills/ascend/npu-arch` |
| Tiling 设计方法论 | ascendc-tiling-design | `skills/ascend/ascendc-tiling-design` |
| API 用法与最佳实践 | ascendc-api-best-practices | `skills/ascend/ascendc-api-best-practices` |
| 代码检视 | ascendc-code-review | `skills/ascend/ascendc-code-review` |
| 性能优化手法 | ascendc-performance-best-practices, ascendc-perf-optimize | `skills/ascend/` |
| 工程脚手架 | ascendc-direct-invoke-template | `skills/ascend/ascendc-direct-invoke-template` |
| 测试用例 | ascendc-st-design, ascendc-ut-develop | `skills/ascend/` |
| 精度标准与调试 | ops-precision-standard, ascendc-precision-debug | `skills/ascend/` |
| 性能采集 | ops-profiling | `skills/ascend/ops-profiling` |
| 日志 / 崩溃 / 运行时调试 | msnpureport-toolkit, asys-toolkit, ascendc-crash-debug, ascendc-runtime-debug | `skills/ascend/` |
| 环境检查 | cann-env-setup, ascendc-env-check | `skills/ascend/` |

## 7. 自动调优纪律

1. **tiling 即算法**：多核切分 / UB 切分 / Buffer 规划方案是"算法候选"；tiling 内的参数（块大小、buffer 深度）是"配置候选"，分组推进。
2. 候选必须声明目标 NpuArch，跨代际候选分开记录。
3. 性能结论必须来自 `ops-profiling` 的对比模式输出，不接受"感觉变快"。

## 8. 常见坑

- SoC 版本不同导致指令集差异——条件编译分支漏写（用 `npu-arch` 技能核对）。
- UB 溢出 / double buffer 深度过大只在特定 tiling 参数组合下触发。
- 忘记同步（SetFlag/WaitFlag 配对）导致偶发数据竞争，验证"偶然通过/失败"。
- AI Core 与 Vector Core 负载不均：`msnpureport-toolkit` 可用于核查 core 占用。
