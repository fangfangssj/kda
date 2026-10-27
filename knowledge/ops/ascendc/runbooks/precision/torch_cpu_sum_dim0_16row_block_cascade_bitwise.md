---
type: "Runbook"
title: numpy复刻torch sum序：16行块级联bitwise等golden
description: torch CPU fp32 sum(dim=0) 可用 16 行块内顺序累加+块间顺序合并 bitwise 复刻；用于证明 golden 求和序与 mare 门限可达性。
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# numpy复刻torch sum序：16行块级联bitwise等golden


## 现象

在定界"fp32 跨 B 归约输出的 mare 门限是否数学可达"时（见同批定界卡），必须回答一个前置问题：**golden 到底是按什么顺序求和的、能不能在 numpy 里逐位复刻出来**。若能 bitwise 复刻 golden，就能严格证明"唯一能过 mare 门限的实现 = 逐位克隆 golden 的求和序"；若不能，连这个结论都无法下。本卡给出实测有效的复刻配方：对 torch 2.7.1+cpu 的 fp32 `sum(dim=0)`（列向求和、行数 B 为 16 的倍数、行数 ≥ 数千），用 numpy 做 **16 行块内顺序累加 + 块间顺序合并**，与 golden 期望文件**逐位相等（bitwise equal）**，四个大 B 用例 × 两路 bias 输出共 8 路，全部 True（mare if not = 0.000e+00）：

```text
P7    grad_input_bias   casc16==golden bitwise: True  (mare if not: 0.000e+00)
P7    grad_hidden_bias  casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-a  grad_input_bias   casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-a  grad_hidden_bias  casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-c  grad_input_bias   casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-c  grad_hidden_bias  casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-e  grad_input_bias   casc16==golden bitwise: True  (mare if not: 0.000e+00)
B0-e  grad_hidden_bias  casc16==golden bitwise: True  (mare if not: 0.000e+00)
```

## 适用场景

- **工程形态**：aclnn ST 精度定界 / golden 序探查（ops-registry-invoke 算子仓的精度排障工作流）；任何"NPU kernel fp32 归约输出 vs torch CPU fp32 golden"需要判定门限可达性或解释系统性分叉的场景。
- **API 路径**：被复刻对象 = torch（2.7.1+cpu）`Tensor.sum(dim=0)` 的 fp32 CPU 实现；复刻工具 = numpy fp32 逐步累加。
- **验证过的平台与 CANN 版本**：Ascend 950（arch35）+ CANN cann-9.0.0-beta.2 的 ST 工作流内验证（golden 数据由该 ST 的 oracle 产出；与 frontmatter `platforms` 一致）。复刻本身是 CPU/numpy 行为，但知识的使用场景绑定 NPU-vs-torch 精度定界。
- **适用粒度（transfer_scope）**：`operator_family`——凡"fp32 列归约 golden 对拍"的排障皆可试；对"行数为 16 的倍数、行数 ≥ 数千、torch 2.7.1+cpu"这一形态实测成立。
- **路线优先级**：`recommended_fastpath`（证明门限可达性 / 解释 golden 分叉的第一步探针；成本极低，纯 CPU）。
- **明确不适用 / 未验证场景**：行数非 16 倍数的尾块处理未验证（实测四用例 B 均为 16 的倍数）；其他 torch 版本、GPU/其他后端的 sum；N<6K 行（已提交探针记载 torch 可能走不同内核，本 trace 未复验）；dim≠0 或多轴归约；fp16/bf64 等 dtype。

## 判别方法

用途定位：当 fp32 归约输出与 golden 出现**稳定的系统性分叉**（mere 达标、mare 超阈、逐元素链 bitwise 一致）时，用本复刻做两件事：

1. **可达性判定**：若复刻序 == golden（bitwise），则任何不逐位克隆该序的实现都过不了 mare=0 之外的严格门限 → 门限不可达的证明闭环；
2. **分叉解释**：把"kernel vs golden 的差异"翻译成"两套求和序对同一组 fp32 列向量的不同结合"，配合 fp64 真值即可定量归属误差（见同批定界卡）。

操作：取 golden 的输入矩阵 X（B×cols，fp32），跑 `blocks(X, 16)`（下述实现），与 golden 期望 `np.array_equal` 对拍。**注意 per-output**：多路输出各有各的被求和矩阵，不能共用（见同批错参考卡）。

## 设计决策

- **为何选 16 行块**：依据已提交探针（`torch_order.py`/`torch_order2.py`，ATen CPU `multi_row_sum` 级联实现：4 级自适应、`level_step = 2^max(4, CeilLog2(N)/4)`，对 N≥6K 与线程数 1/8/64 无关——此为入库探针结论，本 trace 未复跑）。本 trace 的贡献是直接实测：对 4 个用例（B=32/64/96/256），16 行块级联与 golden **8/8 bitwise 相等**。
- **为何块内与块间都用顺序累加**：fp32 加法不满足结合律，任何重排（两两树、并行归约）都会改变结果；实测中 sequential / 纯 pairwise 树 / 48 行块等候选序的 mare 均不为 0（候选序 mare 数值来自已提交证据 JSON 记载，本 trace 未逐一复算）。
- **备选实现**：完整的 ATen 4 级自适应级联（`multi_row_sum_cascade`，见「## 完整代码」）——对非 16 整除块数同样按 ATen 逻辑处理尾块；本 trace 实测用的是简化版 `blocks(X,16)`（B 均为 16 倍数时二者等价）。简化版未验证尾块行为，若 B 非 16 倍数应使用完整版并自行对拍。
- **当前方案档位**：`recommended_fastpath`（定界探针首选；仅当需要预测 golden 逐位结果时才值得付出复刻成本）。

## 系统信息

- 场景：Ascend 950（arch35）+ CANN cann-9.0.0-beta.2 的 aclnn ST 精度定界；golden = torch 2.7.1+cpu fp32 `sum(dim=0)`（ST oracle 产出 `exp_*.bin`）。
- 数据布局：被求和矩阵 X 为 B×(3H) fp32（两路 bias 各一份，列 = concat(grg,gig,gin) / concat(grg,gig,ghn)）；golden 期望为 3H 维 fp32 向量。
- 相关文件：复验侧内联复刻脚本（下述）；已提交的 `final_evidence.py`（含 `multi_row_sum_cascade` 完整实现与序枚举）、`torch_order.py`/`torch_order2.py`（torch 序探针，入库）。
- 复刻环境：python3 + numpy（CPU 即可，无需 NPU）。

## 完整代码

**实测版（本 trace 逐位验证 8/8 通过，B 为 16 的倍数时用这个）**：

```python
import numpy as np, json
from pathlib import Path
ST = Path("operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/st_data")

def seq_cols(X):
    acc = X[0].copy()
    for i in range(1, X.shape[0]):
        acc = (acc + X[i]).astype(np.float32)
    return acc

def blocks(X, bs):
    B = X.shape[0]
    parts = [seq_cols(X[s:s+bs]) for s in range(0, B, bs)]
    out = parts[0]
    for p in parts[1:]:
        out = (out + p).astype(np.float32)
    return out

def rel(a, g):
    return np.abs(a - g) / (np.abs(g) + 1e-7)

for case in ["P7", "B0-a", "B0-c", "B0-e"]:
    meta = json.loads((ST / case / "meta.json").read_text())
    B, H = meta["B"], meta["H"]
    go = np.fromfile(ST / case / "grad_hy.bin", dtype=np.float32).reshape(B, H)
    st = np.fromfile(ST / case / "storage.bin", dtype=np.float32).reshape(B, 5*H)
    r_ = st[:, 0*H:1*H]; z = st[:, 1*H:2*H]; n_ = st[:, 2*H:3*H]
    hx = st[:, 3*H:4*H]; hn = st[:, 4*H:5*H]
    gin = (go * (1 - z) * (1 - n_*n_)).astype(np.float32)
    gig = (go * (hx - n_) * (1 - z) * z).astype(np.float32)
    grg = (gin * hn * (1 - r_) * r_).astype(np.float32)
    ghn = (gin * r_).astype(np.float32)
    Xs = {"grad_input_bias": np.concatenate((grg, gig, gin), axis=1),
          "grad_hidden_bias": np.concatenate((grg, gig, ghn), axis=1)}
    for name, X in Xs.items():
        g = np.fromfile(ST / case / f"exp_{name}.bin", dtype=np.float32)
        c16 = blocks(X, 16)
        print(f"{case:5s} {name:17s} casc16==golden bitwise: {np.array_equal(c16, g)}  (mare if not: {rel(c16,g).max():.3e})")
```

**完整版（ATen 4 级自适应级联，处理任意 N 的尾块；取自已提交 final_evidence.py，本 trace 未重跑）**：

```python
def multi_row_sum_cascade(X, nrows=1):
    """ATen multi_row_sum cascade, nrows=1, num_levels=4 (fp32, sequential within
    level_step, level partials combined sequentially at the end)."""
    size = X.shape[0]
    num_levels = 4
    clog2 = (size - 1).bit_length()
    level_power = max(4, clog2 // num_levels)
    level_step = 1 << level_power
    level_mask = level_step - 1
    acc = [np.zeros(X.shape[1], dtype=np.float32) for _ in range(num_levels)]
    i = 0
    while i + level_step <= size:
        for j in range(level_step):
            acc[0] = (acc[0] + X[i]).astype(np.float32)
            i += 1
        for j in range(1, num_levels):
            acc[j] = (acc[j] + acc[j - 1]).astype(np.float32)
            acc[j - 1] = np.zeros_like(acc[0])
            mask = level_mask << (j * level_power)
            if (i & mask) != 0:
                break
    while i < size:
        acc[0] = (acc[0] + X[i]).astype(np.float32)
        i += 1
    out = acc[0]
    for j in range(1, num_levels):
        out = (out + acc[j]).astype(np.float32)
    return out
```

## 预防

- 对 fp32 归约输出设严格 mare 门限前，先用本复刻判定 golden 序的可克隆性——若 golden 序可 bitwise 复刻，说明门限实际上在强迫"克隆 or 不可达"的二选一。
- 任何"kernel vs golden 归约分叉"的排障，先跑一次 `blocks(X,16)` 对拍：相等 → 分叉纯由求和序引起；不等 → 先查数据布局/列序（含错参考风险）。
- 写序枚举探针时保持每一步 `astype(np.float32)`——漏掉一次 cast 就变成 fp64 累加，结论全废。

## 验证信号

以下信号来自同一次机械复验会话中对已提交 golden 期望数据的内联复刻对拍（CPU numpy 重算 vs ST oracle 产出的 golden 文件）。逐条标注来源平台与证据等级：

- 16 行块级联复刻（`blocks(X,16)`，块内顺序累加 + 块间顺序合并）与 golden 期望文件 bitwise 相等 **8/8**：P7/B0-a/B0-c/B0-e × grad_input_bias/grad_hidden_bias，全部 True，`mare if not = 0.000e+00`（来源平台：950；证据等级：tool_observed）。
- 同批三方对比显示 kernel（按设计 ReduceSum 树序）与 golden 的 mare = 8.085e-04~5.112e-03 而 mere 全过——即"另一套合法求和序与 casc16 的分叉量级"，佐证复刻序确实捕捉到了 golden 的实际结合顺序（来源平台：950；证据等级：tool_observed）。

> torch 序的线程无关性（1/8/64 线程一致）与 N≥6K 边界**不是本 trace 的实测**（来自已提交探针脚本与证据 JSON 的记载，本 trace 未复跑）——已按 hearsay 处理，移入「## 设计决策」「## 失效边界」如实标注，不作为验证信号。

## 失效边界

- **B 非 16 的倍数**：简化版 `blocks(X,16)` 的尾块行为未验证（尾块按 `seq_cols` 截断处理，但是否仍 bitwise 等 golden 未测）；应改用完整版 `multi_row_sum_cascade` 并自行对拍。
- **torch 版本/后端**：仅对 torch 2.7.1+cpu 实测；其他版本或 GPU/自动微分路径的 sum 序未验证（未验证，推断不可迁移）。
- **N<6K 行**：已提交探针记载 torch 可能走不同内核，未复验。
- **dim≠0、多轴、keepdim、非 fp32**：未验证。
- 推断可迁移：同为 fp32 CPU 列归约的相近 torch 小版本（未验证，推断可迁移）。

## 坏实践

- 用 `X.sum(axis=0)`（numpy 默认 pairwise 累加）去模拟 torch golden——两家的累加序不同，永远对不上，还以为是 kernel 错。
- 复刻脚本中漏掉逐步 `.astype(np.float32)`（变成 fp64 累加后舍入），得出的"复刻失败"是假阴性。
- 把"复刻成功"直接当成"kernel 应该改成这个序"的论据——复刻只证明门限的可达性结构，是否偏离设计求和序是另行裁定的工程决策（见同批定界卡「## 坏实践」）。

## 误诊

- 无（本卡为正向配方；相关误诊见同批定界卡：把求和序分叉误诊为 kernel 归约缺陷）。

## 未决事项

- 非 16 倍数 B 的尾块 bitwise 等价性未测（接续点：构造 B=17/33 用例跑 `multi_row_sum_cascade` 对拍）。
- 其他 torch 版本 / N<6K / 多线程下的序稳定性未复验（接续点：重跑入库探针 torch_order.py 的线程扫描）。
