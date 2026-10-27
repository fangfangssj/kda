---
type: "Runbook"
title: "MLA Prolog decode 950 tile 调优配方"
description: "950/Ascend950 上 mla_prolog_quant_d decode 算子通过 4 项独立 tile_config 旋钮调优 (pre_quant_cube_tile k=256 / q_vec/k_vec 翻倍 / cube_wuk_tile k=64 / stitch_function_max_num=128) 使 E2E 相对初始 baseline 约 0.85x，精度全 PASS。"
tags: ["task.optimization", "topic.tiling", "topic.implementation.tiling", "topic.programming_model.execution_unit.cube", "topic.performance.optimization.optimization_overview"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# MLA Prolog decode 950 tile 调优配方

## 现象

在 950 (DAV_3510, Ascend950) CANN 9.1.0 上对 `mla_prolog_quant_d` (DeepSeek V3.2 MLA Prolog decode 非量化，t=8, b=4, s=2, s2=64K, n1=128, h=7168, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512, block_size=128, cache_mode=PA_BSND) 算子做性能调优。初始 baseline（impl 默认 options_list + 测试默认 tile_config）bubble analysis 显示：AICore E2E 处于高位，利用率很低，Total Task 含大量 fake task，主要瓶颈是 AIV_49-56 核等待 Matmul_qa 前驱，大量空闲（绝对时序仅作 provenance，不进正文）。逐参数 A/B 测试（改一项→跑测试→看 E2E+利用率+精度→保留或回退）找到 4 项有效 tile 旋钮，组合后 E2E 相对初始约 0.85x，精度全 PASS。

## 适用场景

- **工程形态**: kernel 直调（PyPTO `@pypto.frontend.jit` 算子，通过 `MlaTileConfig` 字段与 `options_list()` runtime_options 调 tile）。
- **API 路径**: `tile_config.pre_quant_cube_tile` / `cube_qb_tile` / `cube_wuk_tile` (6 元组 [m,m,k,n,base_n,base_k])、`tile_config.q_vec_tile1` / `k_vec_tile1` (vector tile)、`tile_config.unroll_list`、`options_list()["runtime_options"]["stitch_function_max_num"]`。
- **验证过的平台与 CANN 版本**: Ascend950 (950/DAV_3510)，CANN 9.1.0（来源平台：950）。
- **适用粒度**: `same_operator`（针对 mla_prolog_quant_d decode 的具体 K/q_lora_rank 规模；调优方向可外推到同类大-K decode cube 算子，但具体 tile 数值是本算子形状的解）。
- **路线优先级**: `recommended_fastpath`（tile shape 调优是本算子在 950 上最直接有效的路线，先于 Mix合图/vf_options/sg_set_tunevf_mode 等会被 banned 的 pass 选项路线）。
- **不适用或未验证场景**: prefill 版 `mla_prolog_quant_p` (t>64) 的 pass_options 与 tile 不同（见 impl 中 t>64 分支），本配方数值不可直接套用；非 950 平台 (a3/a2) 的 L1/L0 上限与 cube 单元不同，tile 上限不同，未验证。

## 根因

初始 baseline 的低利用率与高 E2E 根因是 tile_config 的 k/n 分量偏小导致 cube matmul 迭代次数过多、且 L0 缓冲压力未优化：

- `pre_quant_cube_tile` k=64（初始）对 K=3584 的 matmul 需 56 次 K 迭代；调到 k=256（L1a 上限）只需 14 次，迭代次数 1/4，E2E 降幅最大（相对 stitch baseline 约 0.88x）。k=512 因超 kL1a=256 报 INVALID_TILE（见相关 build 卡）。
- `q_vec_tile1`=32 / `k_vec_tile1`=512（初始）对 RoPE/scatter 的 vector 段迭代偏多；翻倍到 64/1024 减少 vector 迭代，E2E 相对约 0.86x。再翻倍到 128/2048 反而回归（L0 压力增大）。
- `cube_wuk_tile` k=128（初始）L0 压力偏大；减半到 k=64 释放 L0、E2E 相对约 0.85x。再减半到 k=32 过小、迭代增多回归。
- `stitch_function_max_num=128`（初始 DAV_3510 options_list 缺失，补上）启用 memory-driven stitch mode（estimated stitch func num 约 152），小幅改善 CV 间搬运。

4 项独立根因（迭代次数 / L0 压力 / vector 迭代 / stitch 搬运），每项单独 A/B 验证，合并为同一调优配方。

## 完整实现

调优分 4 步，每步只改一项 tile、跑一次测试、看 E2E+利用率+精度、保留或回退：

1. impl `options_list()` DAV_3510 分支 runtime_options 加 `stitch_function_max_num: 128`（其余 device_sched_mode=3、max_workspace_kb=131072 不变）。相对初始约 0.99x，保留。
2. test `tile_config.pre_quant_cube_tile` k 分量 64→256（即 `[m,m,256,256,128,128]`）。相对 stitch baseline 约 0.88x，精度 PASS，保留。
3. test `tile_config.q_vec_tile1` 32→64、`k_vec_tile1` 512→1024。相对约 0.86x，精度 PASS，保留。再翻倍回归，回退。
4. test `tile_config.cube_wuk_tile` k 分量 128→64（即 `[m,m,64,128,128,128]`）。相对约 0.85x（最终最优），精度 PASS，保留。再减半 k=32 回归，回退。

每步失败/回退的 tile 旋钮（同 trace 内 A/B 测过、回归的，作边界）：`cube_qb_tile` k=128/256 回归、`cube_qb_tile` n=128 回归、`cube_wuk_tile` n=64/256 回归、`pre_quant_cube_tile` n=512/128 回归、`unroll_list=[8,1]`/`[4,2,1]` 回归、`q_vec_tile0=2` 回归、`k_vec_tile0=4` 回归、`mv_tile=4` 回归、`device_sched_mode=2` 回归、`cube_l1_reuse_setting -1=8` 回归、`cube_nbuffer_setting -1=8` 回归、`ooo_sched_mode='GAPMIN'/'HLF'` 回归。这些是本算子形状下的 tile/pass 死区，不要重复试。

## 系统信息

- **工具链**: PyPTO (`@pypto.frontend.jit`)，CANN 9.1.0，`pypto.platform.npuarch == 'DAV_3510'`。
- **目标平台**: Ascend950 (950/DAV_3510)，2 NPU。
- **目录布局**: impl: `src/pypto_gym/ops/pypto_tensor/deepseek_v32_exp/mla_prolog_quant_impl.py`；test: `tests/ops/deepseek_v32_exp/test_mla_prolog_quant.py`，目标用例 `test_b4_s64k2_pa_nd_bf16_d`（`@pytest.mark.soc("950")`）。
- **相关文件清单**（与 trace 实际读写一一对应）:
  - `mla_prolog_quant_impl.py` — `options_list()` (DAV_3510 runtime_options)、`mla_prolog_quant_d` JIT 装饰器、`mla_prolog_quant_compute` (compute 主体)、`pre_compute_2d` (Matmul_qa→RmsNorm_qa→Matmul_qb)。
  - `test_mla_prolog_quant.py` — `test_b4_s64k2_pa_nd_bf16_d` 的 `MlaTileConfig` 赋值段。
  - `cannbot-skills/ops/pypto-op-perf-tune/` — 调优 skill（Mix合图 / vf_options / sg_set_tunevf_mode 等）。
- **性能采集**: `debug_options={"runtime_debug_mode": 1}` 加到 JIT 装饰器，输出 bubble_analysis.log / dyn_topo.txt / 泳道图；E2E 取 AICore End-to-End Time，利用率取 AICore Utilization，精度取 qNope/qRope/qNorm/kv/kr 五路比对。

## 完整代码

impl `options_list()` 终态（DAV_3510 分支加 `stitch_function_max_num: 128`，verbatim 取自 trace 事件上下文 + 后续 edit 确认）：

```python
def options_list():
    if pypto.platform.npuarch == 'DAV_3510':
        return {
            "runtime_options": {
                "device_sched_mode": 3,
                "stitch_function_max_num": 128,
                "max_workspace_kb": 131072},
            }
    else:
        return {
            "runtime_options": {
                "device_sched_mode": 2,
                "stitch_function_max_num": 128,
                "max_workspace_kb": 131072,
            },
        }
```

`mla_prolog_quant_d` JIT 装饰器终态（加 `debug_options` 用于性能采集）：

```python
@pypto.frontend.jit(
    runtime_options=options_list()["runtime_options"],
    debug_options={"runtime_debug_mode": 1},
)
def mla_prolog_quant_d(
    ...
```

test `test_b4_s64k2_pa_nd_bf16_d` 的 `MlaTileConfig` 赋值段终态（4 项有效优化后的最优配置）：

```python
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 8

    c0 = 16
    m_tile_value = (min(32, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 256, 256, 128, 128]
    tile_config.cube_qb_tile = [m_tile_value, m_tile_value, 64, 256, 256, 256]
    tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 64, 128, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 1
    tile_config.q_vec_tile1 = 64
    tile_config.k_vec_tile0 = 2
    tile_config.k_vec_tile1 = 1024
    tile_config.unroll_list = [8, 4, 2, 1]
```

`mla_prolog_quant_d` compute 体内 DAV_3510 t<=64 分支的 pass_options（终态）：

```python
        if pypto.platform.npuarch == 'DAV_3510' and t <= 64:
            pypto.set_pass_options(cube_l1_reuse_setting={-1: 4, 0: 1, 1: 1, 2: 1})
            pypto.set_pass_options(cube_nbuffer_setting={-1: 4, 0: 1, 1: 1, 2: 3})
            pypto.set_pass_options(vec_nbuffer_setting={-2: 1, -1: 4})
```

## 设计决策

1. **pre_quant_cube_tile k 顶到 256（L1a 上限）而非更大**：K=3584 的 matmul，k 越大迭代越少，但 k 受 kL1a=256 硬约束（k=512 报 INVALID_TILE）。k=256 是合法上限、迭代次数 1/4，是本算子最大单步收益（相对 stitch baseline 约 0.88x）。被放弃的候选：k=512（硬件约束拒绝）、k=128（收益不足，回归）。
2. **cube_wuk_tile k 减半到 64 而非更大**：与 pre_quant_cube_tile 相反方向——cube_wuk 的 k 不是迭代瓶颈，k=128 时 L0 缓冲压力大，减到 64 释放 L0、E2E 降。被放弃的候选：k=32（过小、迭代增多回归）、k=256（L0 压力大回归）。这说明同算子内不同 cube 的 k 最优点不同，不能一刀切。
3. **q_vec/k_vec tile 翻倍到 64/1024 后不再翻倍**：vector tile 翻倍减少 vector 迭代，但再翻倍到 128/2048 反而回归（L0/L1 压力）。
4. **stitch_function_max_num=128 补到 DAV_3510 options_list**：DAV_3510 分支原本缺此项（其他平台分支有），补上启用 memory-driven stitch mode（estimated stitch func num 约 152 > 128 threshold），小幅改善 CV 间搬运。收益小但稳定、无回归，保留。
5. **unroll_list 保持 [8,4,2,1]**：trace 测过 [8,1]/[4,2,1]/[2,1]/[1]，减 unroll 均回归或仅用于 Mix合图实验（Mix合图本身 banned）。unroll_list 不是本算子的有效优化旋钮。
6. **当前方案属 recommended_fastpath**：tile shape 调优是本算子在 950 上最直接有效的路线；Mix合图/vf_options/sg_set_tunevf_mode/ooo_sched_mode 等 pass 选项路线均 banned 或回归。

## 验证信号

- 4 项 tile 调优组合后 E2E 相对初始 baseline 约 0.85x（最终最优），利用率相对提升（来源平台：950；证据等级: tool_observed）
- 回退 Mix合图 改动后验证复测 E2E 在最优值附近（运行时正常波动），精度 qNope/qRope/qNorm/kv/kr 全 PASS（来源平台：950；证据等级: tool_observed）
- 每步单旋钮 A/B：pre_quant k=256 相对 stitch baseline 约 0.88x；q_vec/k_vec 翻倍约 0.86x；cube_wuk k=64 约 0.85x；stitch_function_max_num=128 约 0.99x（来源平台：950；证据等级: tool_observed）

## 性能影响

相对代价（参照路线：初始 baseline，来源平台 950）：
- 最优组合相对初始 baseline E2E 约 0.85x。
- 最大单步收益：pre_quant_cube_tile k 64→256，相对 stitch baseline 约 0.88x。
- 其余 pass 选项路线（Mix合图/vf_options/sg_set_tunevf_mode）均回归或 banned，不可作为替代路线。

## 坏实践

- 把 AICore 利用率当唯一判据：vf_options 让利用率相对提升但 E2E 回归，若只看利用率会误保留。
- Mix合图 一超时就反复缩小 scope/unroll 重试：单 mix kernel 即超时，缩小无效，累计浪费大量编译时间。
- 一刀切同算子所有 cube 的 k：pre_quant_cube_tile k 要顶到 256（迭代瓶颈），cube_wuk_tile k 要减到 64（L0 压力），方向相反，不能同向调。
- 把本算子形状的 tile 数值（如 pre_quant k=256）当通用解外推到其他算子——这些值依赖 K=3584、q_lora_rank=1536 等本算子形状。

## 失效边界

- 已验证：950/Ascend950/DAV_3510 + CANN 9.1.0 + `mla_prolog_quant_d` decode (t=8, 非量化, PA_BSND) 的具体形状。
- 未验证，推断可迁移：同类大-K decode cube 算子（K 在千级、cube matmul K 迭代是瓶颈）的调优方向（顶 k 到 L1a 上限、减 cube_wuk k 释放 L0、vector tile 翻倍到回归点前）大概率适用，但具体数值须按目标算子形状重测。
- 未验证：a3/a2 平台 L1/L0 上限与 cube 单元不同，k/n 上限数值（256/64）不可直接外推。
- 不适用：prefill 版 `mla_prolog_quant_p` (t>64) 走 impl 中 t>64 分支的 pass_options，本 decode 配方数值不可套用。