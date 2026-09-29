# Hardware Profile: `Ascend · Triton (triton_ascend)`

昇腾 NPU 上的 triton_ascend DSL kernel 开发。配套 starter prompt：`prompts/triton-flow.md`。将本档案复制到独立任务工作区，命名为 `hardware-profile.md`。

## 1. 目标硬件与工具链

- 器件：Atlas A2/A3 训练系列（910B / 910C 等），架构能力用 `npu-arch` 技能确认。
- 基线：CANN ≥ 8.x + torch_npu + triton_ascend 插件；具体 DSL API 以 `triton-op-coding` 技能为准。
- 环境清单：先跑 `cann-env-setup`、`ascendc-env-check` 技能检查项，结果记录进 `docs/draft.md`。

## 2. 实现语言与 API

- triton_ascend DSL（kernel 语法与 OpenAI Triton 同源，backend=ascend），宿主 Python + torch_npu。
- 算法设计从 `triton-op-designer` 技能起步（算法草图 → 代码生成 → 验证的既定链路）。
- 契约未特批时禁止：引入 Ascend C 手写核函数替代（那是 `ascend-ascendc.md` 档案的范围）、引入外部编译产物。

## 3. 正确性验证约定

- 参考实现：torch_npu（同名 ACLNN 算子）或 CPU golden。
- 判定：按 `ops-precision-standard` 技能选比对方式与阈值；调试走 `triton-precision-debug` 技能。
- 形状覆盖：非整除 tile、单元素、大 shape；mask 分支必须触发。
- 验证链路：用 `triton-op-verifier` 技能的标准流程（创建验证工程 → `scripts/verify.py` 功能验证 → `scripts/benchmark.py` 性能采集）。

## 4. 性能测量约定

- 采集与对比统一走 `ops-profiling` 技能：`msprof_profile_run.sh --compare / --quick` 输出 `performance.json` 与 markdown 对比报告；批量候选用 `--batch` 并行采集。
- 板卡不可用时可用 `triton-simulator-optimizer` 技能做仿真级预筛，但**晋升结论必须来自上板数据**。
- 基线集合：torch_npu eager / ACLNN 同名算子 / 被替换的现有实现。
- `benchmark.csv`：`candidate,algo_variant,config,shape,dtype,median_us,baseline_us,speedup,date`。

## 5. Profiling 与归因

- 算子级：`ops-profiling` 技能（msprof 采集 + 瓶颈分析）。
- 延迟专项优化：`triton-latency-optimizer` 技能。
- Device 侧日志 / 维测：`msnpureport-toolkit`、`asys-toolkit` 技能。

## 6. 技能映射

| 需求 | 技能 | 位置 |
|---|---|---|
| 芯片架构 / 条件编译 | npu-arch | `skills/ascend/npu-arch` |
| 算法草图设计 | triton-op-designer | `skills/ascend/triton-op-designer` |
| DSL 编码规范 | triton-op-coding | `skills/ascend/triton-op-coding` |
| 验证 + 基准流程 | triton-op-verifier | `skills/ascend/triton-op-verifier` |
| 精度调试 | triton-precision-debug | `skills/ascend/triton-precision-debug` |
| 延迟优化 | triton-latency-optimizer | `skills/ascend/triton-latency-optimizer` |
| 仿真预筛 | triton-simulator-optimizer | `skills/ascend/triton-simulator-optimizer` |
| 性能采集与分析 | ops-profiling | `skills/ascend/ops-profiling` |
| 精度标准 | ops-precision-standard | `skills/ascend/ops-precision-standard` |
| 日志 / 维测 | msnpureport-toolkit, asys-toolkit | `skills/ascend/` |
| 环境检查 | cann-env-setup, ascendc-env-check | `skills/ascend/` |

## 7. 自动调优纪律

1. **先算法变体，后配置扫描**（同 `nvidia-triton.md` 第 7 节）：候选标注"算法 / 配置"分组。
2. autotune `key` 覆盖影响最优配置的 shape 语义；报告搜索空间大小与逐配置耗时。
3. 晋升证据 = `ops-profiling` 对比模式输出（加速比 + 绝对耗时），仿真结果只能作为方向性参考。

## 8. 常见坑

- OpenAI Triton 语法 ≠ triton_ascend 全集：部分高级特性（TMA、特定 reduce 模式）支持度以 `triton-op-coding` 技能说明为准，写之前先查，不要照搬 NVIDIA 教程代码。
- CPU golden 与 NPU 参考不一致时优先怀疑参考实现（dtype 提升路径），再怀疑 kernel。
- 上板前漏跑环境检查，把环境问题误判为 kernel 问题。
- 仿真通过 ≠ 上板达标：仿真只用于预筛。
