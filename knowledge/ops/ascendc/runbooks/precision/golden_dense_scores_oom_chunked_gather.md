---
type: "Runbook"
title: "golden 全量物化 scores OOM 改分块 gather"
description: "GQA 稀疏注意力的 CPU golden 用 einsum 一次性物化 [B,N2,G,S1,S2] fp32 全量分数张量（N1=256 时 8.6GB），内存受限环境下用例进程被杀、结果缺失；按 (b,n2) 头分块 gather 并对 G 批处理后，大 N1 用例内存可控且数值等价。"
tags: [task.testing, topic.attention, topic.memory]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# golden 全量物化 scores OOM 改分块 gather

## 现象

在给自定义稀疏 FlashAttention 算子（GQA：N1 个 query 头 / N2 个 KV 头、topK 稀疏索引、BSND/BNSD、fp16/bf16）做 PyTorch ST 精度验收时，183 个用例里有 5 个（N1=256 或 N2=64 的组合，S1=1024、S2=8192、topK=512）在并行 runner 里表现为 **NO RESULT**——用例进程无任何输出直接被杀，连 PASS/FAIL 行都打不出来。逐个手工重跑同样无输出。而其余 178 个用例全部 PASS，C++ ST 阶段（开发早期）这些 shape 也是通过的——问题只出在 PyTorch 测试的 CPU golden 一侧。

用 python 内存公式对失败用例逐一核算（以 B=1, S1=1024, S2=8192, N1=256, N2=8, Dk=Dv=128, topK=512 为例）：

```
mask: 67M bools = 67MB
scores fp32: 8.6GB
query NPU: 67MB = 0.1GB
output: 67MB
```

```
N1=128 N2=8  G=16 scores_fp32=4.3GB
N1=256 N2=8  G=32 scores_fp32=8.6GB   （3 个用例）
N1=128 N2=64 G=2  scores_fp32=4.3GB
```

当时 golden.py 的 `compute_golden` 直接调用评测标杆的 scatter 路径 `sparse_flash_attention`，其核心是 `torch.einsum('bngsd,bnkd->bngsk', q_g, k)`——为利用 GQA 把 N1 reshape 成 (N2, G) 避免物化 K/V 头扩展，但代价是**把完整 [B, N2, G, S1, S2] 分数张量物化出来再 mask**。注意 S2 维是全量 KV 长度（8192），而算子语义只需要 topK=512 个被选中的位置：`B*N2*G*S1*S2*4 = 1*8*32*1024*8192*4 ≈ 8.6GB`（fp32 中间精度计算，golden 注释明确要求 fp32 累加），在 32GB cgroup 限制下叠加 torch 本身占用即被杀。这是 **golden 的内存设计问题，不是算子缺陷**。

## 适用场景

- 工程形态：PyTorch ST 精度验收的 CPU golden 设计/修复（torch adapter 调 aclnn，CPU golden 与 NPU 输出比对）。
- API 路径：`tests/st/torch/golden.py` 的 `compute_golden` / `sparse_flash_attention`（scatter-mask 路径）与 `_compute_golden_gather`（gather 分块路径）；`torch.einsum` / `Tensor.gather`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35），CANN 9.2.0（golden 跑在 CPU，平台约束来自运行环境 cgroup 32GB；与 frontmatter 一致）。
- 适用粒度：operator_family——任何 GQA/多头注意力类算子的 CPU golden（凡 scores 形如 [.., S1, S2] 全量物化再 mask 的实现，S2 远大于实际需要的 topK 时同样爆炸）；对非注意力算子的启示是"golden 不要物化全量中间张量再筛"。
- 路线优先级：recommended_fastpath（大 shape golden 的唯一可行 CPU 路径；scatter 路径保留给小 shape）。
- 明确不适用 / 未验证场景：N2 很大（如 64）的用例——分块路径内存可行但耗时超限（见失效边界），须配合并行度/超时策略或另寻 golden 实现；GPU/NPU 侧 golden（内存带宽与容量模型不同，未验证）；非 fp32 中间精度要求（若 golden 允许 fp16 中间，内存减半但精度语义变化，未验证）。

## 根因

CPU golden 的 scatter-mask 路径把"只在 topK 个稀疏位置上有定义"的注意力分数按稠密 [B, N2, G, S1, S2] 全量物化：GQA 分组（G=N1/N2）使该张量元素数 = `B*N1*S1*S2`（与 G 的展开等价），fp32 下 `N1=256, S1=1024, S2=8192` 时达 8.6GB；golden 又要求 fp32 中间精度（fp16/bf16 输入先 `.float()`），叠加进程内 torch 常驻内存后超过 32GB cgroup 限制，用例进程被 OOM kill，表现为"无输出的 NO RESULT"。算子 kernel 侧按 sparseIndices 逐行 gather（只碰 topK），无此问题。

机制链（数值演算）：物化量 `= B*N2*G*S1*S2*4B = B*N1*S1*S2*4B`（因 `N2*G=N1`）；代入 `1*256*1024*8192*4 = 8.59e9 B ≈ 8.6GB`。而真正需要的稀疏 scores 仅 `B*N1*S1*topK*4B = 1*256*1024*512*4 ≈ 0.5GB`——**物化量 / 需求量 = S2/topK = 8192/512 = 16 倍**。

## 解决配方

修复分两层：`compute_golden` 按 scatter 物化量估算分派（>2GB 走 gather 路径），`_compute_golden_gather` 逐 (b, n2) 头分块 gather、块内对 G 个 query 头批处理。

修复前（`compute_golden` 直接走 scatter 全量物化路径；`sparse_flash_attention` 内部的 einsum 见现象描述）：

```python
def compute_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str,
    is_causal: bool,
) -> torch.Tensor:
    """
    计算 CPU golden 结果。

    为保证 golden 精度（中间 scores 累加用 fp32），将输入转为 fp32 计算后
    再转回原始 dtype 输出。这与 C++ ST 测试的 ComputeGolden（内部 fp32）一致。
    """
    orig_dtype = query.dtype
    # fp32 中间精度计算，避免 fp16/bf16 大 topK 求和溢出与精度损失
    q_f32 = query.float()
    k_f32 = key.float()
    v_f32 = value.float()
    out = sparse_flash_attention(
        q_f32, k_f32, v_f32, sparseIndices,
        scaleValue, inputLayout, is_causal,
    )
    return out.to(orig_dtype)
```

修复后（终版，含内存估算分派 + 分块 gather；两函数均 verbatim 取自该会话最终写入的 golden.py）：

```python
def compute_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str,
    is_causal: bool,
) -> torch.Tensor:
    """
    计算 CPU golden 结果。

    为保证 golden 精度（中间 scores 累加用 fp32），将输入转为 fp32 计算后
    再转回原始 dtype 输出。这与 C++ ST 测试的 ComputeGolden（内部 fp32）一致。

    当 N1*S1*S2 过大时（scores [B,N2,G,S1,S2] 超出内存），改用 gather 路径计算，
    仅物化 [B,N1,S1,topK] 的 scores，数学上与 scatter mask 路径等价（golden 注释
    明确：同一 (b,n2,s1) 下 sparseIndices 无重复，mask 与 gather 语义等价）。
    """
    orig_dtype = query.dtype
    # fp32 中间精度计算，避免 fp16/bf16 大 topK 求和溢出与精度损失
    q_f32 = query.float()
    k_f32 = key.float()
    v_f32 = value.float()

    # 估算 scatter 路径 scores 内存：[B, N2, G, S1, S2] * 4 bytes
    if inputLayout == "BSND":
        B, S1, N1, Dk = q_f32.shape
        S2 = k_f32.shape[1]
        N2 = k_f32.shape[2]
    else:
        B, N1, S1, Dk = q_f32.shape
        S2 = k_f32.shape[2]
        N2 = k_f32.shape[1]
    G = N1 // N2
    scatter_scores_bytes = B * N2 * G * S1 * S2 * 4

    # 阈值：scores 超过 2GB 时改用 gather 路径（避免 CPU OOM）
    if scatter_scores_bytes > 2 * 1024 * 1024 * 1024:
        out = _compute_golden_gather(
            q_f32, k_f32, v_f32, sparseIndices,
            scaleValue, inputLayout, is_causal,
        )
    else:
        out = sparse_flash_attention(
            q_f32, k_f32, v_f32, sparseIndices,
            scaleValue, inputLayout, is_causal,
        )
    return out.to(orig_dtype)


def _compute_golden_gather(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str,
    is_causal: bool,
) -> torch.Tensor:
    """
    内存高效的 gather 路径 golden 计算（分 head 逐块处理）。

    与 sparse_flash_attention (scatter mask 路径) 数学等价，但逐 N2 head 处理，
    每次仅需 [B, 1, G, S1, topK, Dk] 内存，适用于大 N1/S1/S2 场景。
    """
    if inputLayout == "BSND":
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        si = sparseIndices.permute(0, 2, 1, 3)
    else:
        q, k, v, si = query, key, value, sparseIndices

    B, N1, S1, Dk = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    Dv = v.shape[-1]
    topK = si.shape[-1]
    G = N1 // N2

    out = torch.empty(B, N1, S1, Dv, dtype=query.dtype, device=query.device)

    for b in range(B):
        for n2 in range(N2):
            si_n2 = si[b, n2].long()             # [S1, topK]
            k_n2 = k[b, n2]                       # [S2, Dk]
            v_n2 = v[b, n2]                       # [S2, Dv]

            idx_k = si_n2.reshape(S1 * topK).unsqueeze(-1).expand(-1, Dk)  # [S1*topK, Dk]
            idx_v = si_n2.reshape(S1 * topK).unsqueeze(-1).expand(-1, Dv)  # [S1*topK, Dv]
            k_sel = k_n2.gather(0, idx_k).reshape(S1, topK, Dk)  # [S1, topK, Dk]
            v_sel = v_n2.gather(0, idx_v).reshape(S1, topK, Dv)  # [S1, topK, Dv]

            # Batch all G heads sharing this n2's k_sel/v_sel
            q_g = q[b, n2*G:(n2+1)*G]            # [G, S1, Dk]
            scores = torch.einsum('gsd,skd->gsk', q_g, k_sel) * scaleValue  # [G, S1, topK]

            if is_causal:
                s1_idx = torch.arange(S1, device=scores.device).unsqueeze(-1)  # [S1, 1]
                causal_mask = si_n2 > (s1_idx + (S2 - S1))  # [S1, topK]
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

            scores_max = scores.max(dim=-1, keepdim=True).values
            all_masked = torch.isinf(scores_max) & (scores_max < 0)
            scores = scores - scores_max
            scores.exp_()
            scores_sum = scores.sum(dim=-1, keepdim=True)
            scores = torch.where(all_masked, torch.zeros_like(scores), scores / scores_sum)

            out_g = torch.einsum('gsk,skd->gsd', scores, v_sel)  # [G, S1, Dv]
            out[b, n2*G:(n2+1)*G] = out_g

    if inputLayout == "BSND":
        return out.permute(0, 2, 1, 3)
    return out
```

数值等价依据（golden 源码注释声明的契约）：同一 (b, n2, s1) 下 sparseIndices 无重复且值域 [0, S2)，故 scatter 建 mask 再全量算、与先 gather topK 行再算，二者在每个选中位置上的 score 相同；未选中位置 scatter 路径被 `-inf` mask 后不参与 softmax，gather 路径根本不产生这些位置。全屏蔽行（causal 下可能整行被 mask）两种路径都输出 0。

## 设计决策

- **保留 scatter 路径作为小 shape 默认**：评测标杆 golden（禁止改写计算逻辑的契约）保持原样，只在估算物化量超 2GB 时切换——改动最小、且小 shape 上 scatter 更快（向量化好）。
- **第一版 gather 被放弃（坏实践，见下）**：直接按 [B, N1, S1, topK, Dk] 物化 k_sel/v_sel，把 GQA 的头共享展开成了显式复制，内存 68.7GB×2 反而更糟；教训是 **gather 的正确粒度必须利用"k_sel/v_sel 在 G 个 query 头间共享"**——按 (b, n2) 外层分块、块内 G 头共享同一份 k_sel/v_sel。
- **G 批处理 vs 逐 g 循环**：逐 g 内层循环版本正确但慢（Python 循环 B*N2*G 次）；改为块内一次 `einsum('gsd,skd->gsk')` 批处理所有 G 头后，N2=8 的用例 golden 耗时下降一至两个数量级，N2=64 用例仍超时（循环次数由 B*N2*G 降为 B*N2，但 N2=64 时外层循环本身仍是 64 次 Python 迭代）。
- **阈值取 2GB**：按运行环境 cgroup 32GB 与 torch 常驻占用经验取的保守值；其他环境可调（环境变量未做，写死为常量）。
- 当前方案路线档位：recommended_fastpath（大 shape 下 CPU golden 的必要路径；scatter 路径仍是小 shape 的 fastpath）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0，torch 2.7.1+cpu（golden 计算在 CPU）；宿主内存充足但进程受 cgroup 32GB 限制。
- 相关文件：`operators/bench_no_harness_sparse_flash_attention/tests/st/torch/golden.py`（`compute_golden`/`_compute_golden_gather`/`sparse_flash_attention`/`build_inputs`）、`.../tests/st/torch/test.py`（调用方）、`.../tests/st/testcases/L1_test_cases.csv`（用例 shape）。

## 预防

- 在内存受限环境给注意力 golden 写 `einsum('...->bngsk')` 全量物化 scores 再 mask——物化量是需求量的 S2/topK 倍；先用内存公式（B*N1*S1*S2*中间精度字节数）核算再选路径。
- 把"用例无输出被杀"当成算子或 NPU 问题去查——先核 golden 侧内存公式。

## 验证信号

以下信号来自该算子最终精度验收会话：golden 改为分块 gather 后先跑 golden 自检，再逐个复跑此前被杀的 5 个用例，最后全量回归。

- 修复前：5 个大 shape 用例在逐用例独立进程下仍无任何输出被杀（NO RESULT）；内存公式核算 scores fp32 = 4.3GB（N1=128 或 N2=64）~ 8.6GB（N1=256）（来源平台：950；证据等级：tool_observed）。
- 修复后 golden 自检（`test.py --golden-only`）：GQA non-causal / causal mask / Dk!=Dv 切片 / BSND 布局 4 个手造用例 ALL PASS（来源平台：950；证据等级：tool_observed）。
- 修复后 5 个用例全部 PASS：MERE 4.99e-07~8.45e-07（阈值 9.77e-04），smallVal 均 0 失败（来源平台：950；证据等级：tool_observed）。
- 分块路径内存形态：每 (b, n2) 块仅物化 [S1, topK, Dk]/[S1, topK, Dv]（1024×512×128×4B ≈ 268MB/张量）与 [G, S1, topK] scores，无全量 [B, N2, G, S1, S2] 物化（来源平台：950；证据等级：tool_observed，依据代码结构与用例不再被杀的运行结果）。

## 失效边界

- **大 N2 用例耗时超限**：分块路径的 Python 外层循环次数 = B*N2，N2=64 的用例 golden 耗时约 505s（G 批处理后仍如此），超过常规单用例超时（120~300s）；该场景本卡不闭合，须向量化外层循环、改 NPU 侧 golden 或抽样比对，未实施前不得以"跳过比对仍计 PASS"的口径收口（属测试信号污染反模式）。
- 分派阈值 2GB 是按 32GB cgroup 环境经验取值；内存更宽松的环境可以调大阈值让更多用例走向量化更好的 scatter 路径（更快）。
- gather 路径依赖"同一 (b, n2, s1) 下 sparseIndices 无重复、值域 [0, S2)"的输入契约（golden 的 `get_input` 已保证）；若测试输入不满足该契约，scatter 与 gather 语义不再等价。
- 仅验证过 CPU fp32 中间精度路径；GPU golden、fp16 中间精度未验证。

**未决项**：

- 大 N2（64）用例的分块 golden 耗时未优化到可用区间（向量化外层循环 / NPU 侧 golden / 降低中间精度均未尝试）。
- 分派阈值与块大小均写死（2GB、逐 n2 块），未做成可配置项；更细粒度的 S1 分块（控制 [S1, topK, Dk] 块大小）未尝试。
## 坏实践

- 第一版 gather 修复（已被替换）：按 [B, N1, S1, topK, Dk] 一次性物化 k_sel/v_sel——把 GQA 头共享展开成显式复制，`k_sel 68.7GB + v_sel 68.7GB + scores 0.5GB = 138GB`，比 scatter 路径更糟。正确做法是利用头共享按 (b, n2) 分块。
- 在内存受限环境给注意力 golden 写全量物化 scores 再 mask 的实现。
- 把"用例无输出被杀"当成算子或 NPU 问题去查。

## 误诊

- 曾怀疑这 5 个用例是 NPU OOM 或算子缺陷（N1=256 是"max heads"边界用例）：单独跑 NPU 执行路径无异常，内存公式核算证明是 CPU golden 的 einsum 物化被 cgroup 杀；属于测试 oracle 的可扩展性问题，不是算子缺陷。
- 曾试图用"gather 路径只算 topK 位置"直接替换（第一版）：没意识到 gather 索引展开 [B, N1, S1, topK, Dk] 本身就是海量复制，反而放大内存——gather 的省内存前提是不展开头共享维度。
