---
type: "Runbook"
title: "跨 CANN 代移植归约树：改严格更准，不做树匹配"
description: "为逐位匹配某一代 torch_npu/CANN 求和树而调出的归约核，移植到另一代后在实测类上发散（bf16 1-ulp 骑跨集中于归约输出）；新一代的树用 serial/pairwise/block/lane 等简单候选无法逐位复现（实测 16+37 候选均不匹配）。实测有效的方向是把归约换成严格更准的树（全向量 pairwise 折半 + pow2 尾巴保持任务初零 + pad 行重零守卫），使残差被 golden 自身树误差 bound 住。"
tags: ["task.implementation", "topic.precision", "topic.compute.reduction"]
status: draft
sources: []
created_at: "2026-09-24T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。

# 跨 CANN 代移植归约树：改严格更准，不做树匹配

## 适用场景

移植（port_a3_to_a5 等）一个归约树曾为逐位匹配特定 torch_npu/CANN 代而调参的 kernel，面向另一代 SoC/CANN（实测形态：V220 来源 → Ascend950DT_9582 / arch35 / CANN 9.2.0，scan/reduction-heavy 算子）。

## 触发信号

精度比对出现确定性、量子步长级的逐元素发散（bf16 1-ulp 骑跨），集中在归约输出（如 ddt_bias/dA 类）的长序列/大累加量 case；候选与 golden 都是"合理实现"，但 fp32 累加序不同。

## 判据与探针

1. **先证误差归属再动手**：用 fp64 准真值在失败元素上分别量 |ours−truth| 与 |golden−truth|。若 ours 显著更大，问题在我们的树，不是 golden 非规范（实测：|ours−truth|=1.015e-3 vs |golden−truth|=2.25e-5）。
2. **不要反向工程新代的树**：torch_npu/CANN 的内层连续 fp32 求和树不匹配任何简单候选（serial/pairwise/block{32..256}/lanes{8..64}/whole32 实测 ~95/128 行 bitwise 不匹配；第二次独立探针 37 候选同样证伪）。继续试候选是已证伪的方向（两代 lineage 独立得到同一结论）。
3. **换成严格更准的树**：全向量 pairwise 折半树覆盖 next_pow2(N) 区间——pow2 尾巴区间 `[N2/2,N2)` 永不被树写入并保持任务初零；shape 非 pow2 整除时 pad 行须重零守卫。此时残差 = golden 自己的树误差（实测中位 ~2e-5/头），落在 `|ours−golden| ≤ atol+rtol·|golden|` 的比较器内。

## 验证方式（原始锚）

- run：`4-34 Mamba2Bwd` A3→A5 port，target-model 参考策略，run_id `9c763d2c-4327-4774-9897-9386d117112a`（r2）；host=950DT（内部匿名形），container=该 lane 的已证工作容器。
- 比较器：`eco_open_source_v1`，逐输出 `matched_ratio≥0.99` under `atol+rtol·|golden|`，golden=A5 目标端冻结 model.py 输出（a5_native）。
- 失败案例锚：case 31（B=2, T=53, H=64，B·T=106 项/头）：修复前单 cancellation 头失败，|ours−golden|=1.04e-3 超 bound；fp64 准真值归属 |ours−truth|=1.015e-3 vs |golden−truth|=2.25e-5。
- 修复后：固定 50 例 50/50 PASS（同 run 同 DSO 同比较器）。
- 树证伪锚：tree_probe.py 16 候选（serial/pairwise/block{32..256}/lanes{8..64}/whole32）~95/128 行 bitwise 不匹配；第二次独立探针 37 候选同结论（r3 failures_ledger W2）。

## 失效边界

- 不适用回移旧代：旧代（V220）的树对旧代是对的，不要把新树回移。
- "优雅整除"假设必须写下 pow2 推导：例如 PN=64·24 时 48→24 折半会让 WholeReduceSum(32) 读 8 条陈旧 lane（实测 case 垃圾值 + device 507035）。
- 逐元素（非归约）通路不存在此问题：纯 elementwise 递推里归约误差不复利（dS 型轨迹），单次换树即足。
