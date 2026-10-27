---
type: "Runbook"
title: "flat 多核切分 elementwise scalar 主导诊断"
description: "flat 多核切分的 elementwise 算子在 msprof PipeUtilization 上呈现 scalar_ratio 主导、核间严重不均衡、MTE2 带宽远未饱和的诊断方法与机制解释。"
tags: ["task.debugging", "topic.performance.analysis", "topic.tiling", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T07:49:33Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# flat 多核切分 elementwise scalar 主导诊断

## 现象

在为 add（Broadcast 类，out = in1 + in2）算子做上板性能验收时，采用 flat 多核切分策略：Host 侧做 DimensionCollapse（补维 + 合轴）、计算 strides（广播轴 stride=0）、按总元素数均匀切分到多核、填充 TilingData；Kernel 侧按 computePath（NATIVE / CAST_FP32 / PROMOTE_HALF / BOOL / COMPLEX）走 `DataCopyPad(GM→UB) → Broadcast(若需) → Add → DataCopyPad(UB→GM)` 流水。在 910B3（DAV_2201）上用 `msprof op --warm-up=10` 采集 1M fp32 flat same-shape 场景后，归档的 `summary.txt` 显示流水线结构性失衡：标量单元（scalar）占比异常高、矢量单元（vec）占比极低、核间负载严重不均、GM→UB 带宽利用率仅约一成。这不是单纯的"算得慢"，而是搬运粒度 + 流水编排没让 MTE2/VEC 吃满。

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），Kernel 侧 `DataCopyPad` + `Add` + `Broadcast` 流水，Host 侧 flat 多核切分 + DimensionCollapse。
- **API 路径**：`AscendC::DataCopyPad`（GM↔UB 搬运）、`AscendC::Add<T>`（vec 加法）、`AscendC::Broadcast<T>`（UB 内广播）、`msprof op`（性能采集）+ `PipeUtilization.csv`（逐核流水占比）。
- **验证平台与版本**：Ascend910B3（DAV_2201，aarch64-linux），CANN 9.0.0；msprof op 在真实硬件采集（40 核、满频 1800/1800MHz）。frontmatter `platforms: [a2]` 严格对应此实测。
- **适用粒度**：`operator_family`——适用于任何采用 flat 多核切分 + 逐 tile 元数据驱动的 elementwise / broadcast 算子（不限于 add）。
- **路线优先级**：`safety_fallback`（本例 flat 切分可作为正确性基线；本卡没有实现或验证替代性能路线）。
- **不适用/未验证场景**：Cube/Contraction 算子（瓶颈结构不同）；非 flat 切分（已有 tiling 合轴 + 大块搬运的实现不适用本诊断结论）；a3 / 950 平台未在本 trace 实测，推断可迁移但未验证。

## Profile 判读

flat 多核切分把总元素数均匀分到 40 核，每核再按固定 tile 字节数切多次 `DataCopyPad` 搬运。问题出在"逐 tile 粒度过小 + 逐 tile 都走一遍 scalar 驱动的元数据处理"：

1. **搬运粒度过小**：`summary.txt` 的 Memory 段显示 avg MTE2 transfer 仅 7.70KB（共 1064 条搬运指令），远低于 MTE2 推荐的 ≥16KB 单次搬运粒度。粒度小 → 搬运指令数膨胀 → 每次搬运的启动/对齐开销占比高 → MTE2 带宽利用率仅 10.54%（GM→UB）/ 6.21%（UB→GM），L2Cache total_hit avg 0.2%（几乎全 miss）。
2. **scalar 主导**：SCALAR 子类耗时中 `mte2_stall` 占大头（标量单元在等 MTE2 搬入完成而非做有效标量计算）；同时 scalar_ratio avg 45.75% / max 62.60% 远超 vec_ratio avg 8.79% / max 10.03%——即"算单元几乎没算、标量单元在搬运间隙空转/驱动元数据"。
3. **核间不均衡**：核间 aiv_time max/min 差异 58.7%。flat 均分总元素数后，前若干核分到多一个 tile（7 tiles）、其余核 6 tiles、末核只 4 tiles——余数分配不均直接放大核间等待。

这些指标共同证明当前实现没有把 MTE2/VEC 吃满，并存在小搬运与核间时长不均。`scalar_ratio` 的具体组成、把 tile 放大后能改善多少，仍需受控修改与复测；现有 profile 不能单独证明某一个 tiling 参数是唯一根因。

## 判别方法

- **触发（tool_observed）**：elementwise / broadcast 算子 flat 多核切分实现，上板 `msprof op` 采集后 `PipeUtilization.csv` 呈 scalar_ratio 主导且 vec_ratio 极低，或 `Memory.csv` 显示 GM→UB 带宽利用率远低于峰值（本 trace 仅约一成）。
- **探针（executable check）**：
  ```bash
  export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
  source $ASCEND_HOME_PATH/set_env.sh
  cd <operator_build_dir>
  # 用足够大的数据集（≥1M 元素）避免头开销占比失真
  msprof op --warm-up=10 --output=./msprof_output ./<operator_bin>
  # 归档 + 生成逐核摘要
  python3 <ops-profiling>/scripts/perf_summary.py \
    $(ls -td ./msprof_output/OPPROF_* | head -1) operators/<name>
  ```
  核心是读 `PipeUtilization.csv`（逐核 aiv_time / vec_ratio / scalar_ratio / mte2_ratio / mte3_ratio）与 `Memory.csv`（GM→UB 带宽利用率、avg MTE2 transfer、L2Cache 命中率）。
- **观测（tool_observed，本 trace 实测 a2）**：
  - `scalar_ratio` avg 45.75% / max 62.60%，远超 `vec_ratio` avg 8.79% / max 10.03%——标量单元主导，矢量单元几乎闲置。
  - `mte2_ratio` avg 72.52% / max 86.66%，但 GM→UB 带宽利用率仅 10.54%——MTE2 "忙"但忙的是大量小搬运（avg transfer 7.70KB / 1064 instructions），不是带宽吃满。
  - SCALAR 子类 `mte2_stall` 占比最大，`vec_wait` 76.87% / `mte2_wait` 74.14%——vec/mte2 大量时间在等。
  - 核间 aiv_time max/min 差异 58.7%——余数 tile 分配不均。
  - L2Cache total_hit avg 0.2%——搬运粒度小且无复用，几乎全 miss。
- **解释边界（interpretation）**：`scalar_ratio` 主导、带宽利用率低和 avg transfer 仅 7.70KB 可把问题定位为供数/流水效率不足，核间 aiv_time 差异则提示切分不均。是否由 tile 大小单独造成、以及具体优化方向，必须通过一次只改变单变量的实现对照确认。

## 验证信号

- msprof `PipeUtilization`：`scalar_ratio` avg 45.75% / max 62.60% 主导，`vec_ratio` avg 8.79% / max 10.03% 极低（来源平台：a2；证据等级：artifact_verified）。
- msprof `Memory`：GM→UB 带宽利用率 10.54%，avg MTE2 transfer 7.70KB（1064 instructions），L2Cache total_hit avg 0.2%（来源平台：a2；证据等级：artifact_verified）。
- msprof `PipeUtilization` 核间：aiv_time max/min 差异 58.7%；SCALAR 子类 `mte2_stall` 占比最大、`vec_wait` 76.87%（来源平台：a2；证据等级：artifact_verified）。

## 失效边界

- 本诊断结论在 a2（910B3 / DAV_2201）上以 40 核 / 1800MHz 实测；a3、950 平台未在本 trace 实测，**未验证，推断可迁移**（同属 AiCore 流水线模型，scalar/mte2/vec 占比语义一致，但核数、频率、带宽峰值不同，具体阈值需重测）。
- 非 flat 切分实现（已做 tiling 合轴 + ≥16KB 大块搬运 + DoubleBuffer 流水编排）不适用本诊断——其瓶颈结构不同。
- Cube/Contraction、Reduction 类算子的瓶颈结构不同，本卡的"scalar 主导 + MTE2 未饱和"判别不适用。
- 本 trace 未尝试任何优化（无 round_002、无代码改动），故本卡只给诊断与机制解释，不给修复后信号——优化效果未验证。

## 坏实践

- 用过小数据集（如 96 元素）做 msprof 采集——头开销占比失真，掩盖真实瓶颈；本 trace 性能验收子 agent 显式指出"小数据量头开销占比高，不能反映真实性能"并改用 1M 元素重采。
- 只看 Task Duration 单一数值判定"慢"——会漏掉"scalar 主导 + MTE2 未饱和"的结构性失衡；必须读 `PipeUtilization.csv` 逐核占比 + `Memory.csv` 带宽利用率联合定位。
