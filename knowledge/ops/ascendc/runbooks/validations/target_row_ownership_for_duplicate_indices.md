---
type: Runbook
title: 重复索引更新的目标行所有权
description: 在 dense 唯一 AIV 编号前提下，用目标行取模归属保证同一目标的更新由同一 owner 按输入顺序处理。
tags:
- task.implementation
- topic.implementation.tiling
- topic.precision
status: draft
sources: []
created_at: '2026-10-18T08:14:49Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 重复索引更新的目标行所有权

适用于 out-of-place 索引行累加：稳定输入 `var[M,N]`、`value[K,N]`、`indices[K]` 和标量 `alpha` 生成新分配的输出。重复索引的处理顺序要求明确；本结构按 `i=0..K-1` 逐项更新同一目标行，不要求索引已经排序。

## 目标行归属

在所有参与单元使用同一个正整数 `aivNum`，且其 `aivIdx` **恰好唯一、稠密地覆盖 `[0,aivNum)`** 的前提下，目标行 `r` 归 `r % aivNum`。源码 `Process` 先令每个 owner 按 `row=aivIdx; row<M; row+=aivNum` 初始化自己的行，再扫描全部索引，仅处理合法且余数等于本 owner 的目标。于是每行只由一个 owner 初始化及更新；同一行的重复索引按输入序列中的 `i` 递增执行。其他 owner 是否已初始化自己的行，不影响当前行的结构依赖。

这个结论来自行划分和逐项遍历；它不证明实际 MIX 模式下 `GetBlockIdx()` 与 `GetBlockNum()` 满足上述前提。源码注释关于任意 MIX 拓扑的说法未纳入本卡。若编号稀疏、重复、各核总数不同或存在另一个写入者，必须重新分工或使用符合目标平台合同的归并机制。

`CopyVarRowToOut` 和 `ProcessRow` 的队列、独立 VECOUT 写端及 `PIPE_ALL` 随原结构保留；本卡不从静态代码宣布这些语句构成充分或最小的设备同步配方。原源码使用整 32 字节块搬运行，迁移时另核对实际行长、索引与 alpha 的 GM padding、UB 容量及接口计数宽度，不能仅由分工证明搬运安全。

## 检查与边界

CPU 模型覆盖重复目标、非法索引跳过、没有更新的行、owner 数大于行数以及倒序执行 owner。非法索引跳过是该源码的行为，是否符合调用方语义仍须单独确定。模型显式拒绝非稠密或重复 owner 集合；这是模型对前提的断言，原设备 `Process` 没有跨核检查 owner 唯一性的代码。

模型采用小整数验证结构，不证明浮点舍入、原子性、launch 拓扑或多核可见性。实际迁移需额外绑定 launcher、设备编号语义和同一输入的重复执行证据。

同一主题的[Scatter 冲突写入边界](../../guides/vector/transpose_register_loops_and_scatter_accumulation.md)说明 Gather/Add/Scatter 不是原子累加；本卡补充一种有明确前提的目标行独占分工。

## 可独立执行的 CPU 结构检查

将下面完整代码保存为 `target_row_owner.py` 后执行 `python3 target_row_owner.py`。它检验明示前提下的结构模型；输出中的 `device_test` 恒为 `false`。

```python
"""CPU model of target-row ownership; no device scheduling or numeric claim."""
import itertools
import json


def row_owner(var, values, indices, alpha, aiv_num, aiv_ids=None):
    m = len(var)
    assert aiv_num > 0
    aiv_ids = list(range(aiv_num)) if aiv_ids is None else list(aiv_ids)
    assert sorted(aiv_ids) == list(range(aiv_num))  # dense, unique owners
    assert len(values) == len(indices)
    out = [None] * m
    visits = [[] for _ in range(m)]
    for aiv_idx in aiv_ids:
        for row in range(aiv_idx, m, aiv_num):
            out[row] = list(var[row])
        for i, idx in enumerate(indices):
            if 0 <= idx < m and idx % aiv_num == aiv_idx:
                assert out[idx] is not None  # same owner already initialized it
                visits[idx].append(i)
                out[idx] = [x + alpha * y for x, y in zip(out[idx], values[i])]
    return out, visits


def check():
    cases = 0
    for m in range(1, 5):
        var = [[10 * r, 10 * r + 1] for r in range(m)]
        for indices in itertools.product(range(-1, m + 1), repeat=3):
            values = [[i + 1, -i - 2] for i in range(3)]
            expected = [list(r) for r in var]
            expected_visits = [[] for _ in range(m)]
            for i, idx in enumerate(indices):
                if 0 <= idx < m:
                    expected[idx] = [x + 2 * y for x, y in zip(expected[idx], values[i])]
                    expected_visits[idx].append(i)
            for aiv_num in range(1, 7):
                out, visits = row_owner(var, values, indices, 2, aiv_num,
                                        reversed(range(aiv_num)))
                assert out == expected and visits == expected_visits
                cases += 1
    rejected = 0
    for ids in ([0, 0], [0], [0, 2]):
        try:
            row_owner([[1]], [[2]], [0], 1, 2, ids)
        except AssertionError:
            rejected += 1
    assert rejected == 3
    result = {"model": "target_row_owner", "cases": cases,
              "invalid_owner_sets_rejected": rejected,
              "device_test": False}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    check()
```

## 来源定位与证据强度

本卡是人工蒸馏的静态结构材料，来源为独立归档源码。下述 CPU 程序完整保留本卡需要的索引、存储或消费关系，可直接复制运行；它不是原设备算子的实现，也不执行 CANN 接口。没有为本卡运行新真机实验，没有采用归档注释中的历史成功、误差或性能数字。芯片与 SDK 范围尚未由本次证据确认，保留 `draft`、空平台范围与未新增 `verified`；公开附件提供静态核验入口，不把原件 SHA-256 伪装为远端 Git revision。

| 源 ID | 归档内相对文件 | 完整原文件 SHA-256 |
| --- | --- | --- |
| S1 | `level5/005_inplace_index_add_with_sorted/kernel/inplace_index_add_kernel.h` | `1c0024ef0ff88522220cc952981e8960b671ec74aedf93826a2b327d987c4380` |
| S2 | `level5/005_inplace_index_add_with_sorted/kernel/pybind11.cpp` | `b021707ca17d91759fdf5d2a437ce82c5d69c3abd7f9e139b5091bee67bf8103` |

| 源 ID | 本卡相关单元 | 原文件行号 |
| --- | --- | --- |
| S1 | Process | 47–90 |
| S1 | CopyVarRowToOut | 122–141 |
| S1 | ProcessRow | 143–174 |

源 ID 与 SHA-256 绑定已读完整本地文件；行号用于人工对照，不把 SHA-256 当远程 Git revision。对外代码摘录只保留上述相关区段并删除注释，逐段变换在外带 `source_map.json` 中记录；摘录不是完整原文件的字节副本。正文内完整 CPU 模型是本卡的独立可执行检查，不能替代实际 SDK 编译、精度、并发、同步或性能验收。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/C022/source_map.json`；`cannbot_non_level4_split_review_20261019/evidence_public/C022/source_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/C022/target_row_owner.py`；`cannbot_non_level4_split_review_20261019/evidence_public/C022/cpu_check_result.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
