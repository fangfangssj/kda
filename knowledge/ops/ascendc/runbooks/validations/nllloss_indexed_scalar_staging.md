---
type: Runbook
title: NLLLoss 索引取值与单元素暂存容量
description: 从 NLLLoss 的有效 target、ignore 分支和物理地址公式推导单元素读取，限定 input/weight 两个队列的暂存容量。
tags:
- task.implementation
- topic.memory
- topic.data_movement
status: draft
sources: []
created_at: '2026-10-18T08:14:49Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# NLLLoss 索引取值与单元素暂存容量

适用于已知逻辑存储为连续 `(N,C,d1,...,dk)` 的 NLL 单位置取值，令 `D=prod(d1,...,dk)`，二维时 `D=1`。本卡只分析选中 input 与可选 weight 的读取，以及这两个队列的暂存容量，不认证完整 loss、sum/mean、空分母或原工程的端到端精度。

## 索引与有效性

目标位置线性号是 `rowIdx=nIdx*D+dIdx`。在 `0<=rowIdx<N*D`、`D>0`、`C>0`、`0<=classIdx<C` 且乘积不溢出的前提下，对应输入元素的物理偏移为：

```text
(nIdx*C+classIdx)*D+dIdx
```

当 `D>1` 时，单纯把 shape 改成 `[-1,C]` 不会把原始 `input[n,:,d]` 变成连续一行。归档 wrapper 确实使用 reshape；Host 另经 view-base 或 caller frame 推断 `extraDim`。本卡只采用 Kernel 在 **D 和原物理布局已正确绑定** 时的地址式，不把这两种元数据推断策略推广为可靠接口。新实现宜显式传入真实布局元数据；若数据已经转置或复制成另一种物理排列，应重新推导地址。

地址式还假定 input、target 和有效 weight 的 GM 指针均指向各自逻辑首元素。归档 Host 的 launch 对这三者使用 `storage().data()`；已经连续或 dtype 已满足的视图，不会仅因 reshape、contiguous 或 to 被调用就必然消除非零 `storage_offset`。因此不能据这条读取路径认定任意连续视图都正确；采用该 Host 绑定方式时需确认三个存储偏移为零，或确实物化新缓冲并核对逻辑起点。下面 CPU 模型的数组均从逻辑首元素开始，未复现该 Host 的视图取址行为。

`FetchRowContribution` 先读取 target，等于 `ignoreIndex` 时直接返回零贡献，再读取 input 和可选 weight。对非 ignore target，源码没有额外的范围钳制；合法 `classIdx` 是先决条件。模型主动拒绝非法 target 是为了检查这个前提，不能写成原设备已有校验。target 转为 int32 前也应确认值域，避免窄化后改变语义。

## 单元素暂存容量

`ReadInputElem` 的有效 GM 搬运长度是 `sizeof(DataType)`，`ReadWeightElem` 同样只读取 `weight[classIdx]`；各自的 UB 队列槽分配 `AlignBytes(sizeof(DataType))`。对本源码的 2 或 4 字节数据类型，每个选中槽为 32 字节，两槽名义 payload 合计 64 字节，不随 `C` 增长。这不等于整核 UB 只有 64 字节：target/out 队列、队列管理和其他分配还须另外计入，也不表示物理总线交易恰好只有一个元素。

原源码保留 DataCopyPad 描述符、padding、队列及 barrier。CPU 模型只证明下标关系、ignore 跳过及选中槽的容量式；不认证相应 API 在任意 SDK、芯片或 dtype 上的支持，也不继承旧错误码与修复成功。完整实现的 target/weight 长度、空 tensor 和 reduction 行为仍需另验。

## 可独立执行的 CPU 结构检查

将下面完整代码保存为 `nll_scalar_address.py` 后执行 `python3 nll_scalar_address.py`。它检验明示前提下的结构模型；输出中的 `device_test` 恒为 `false`。

```python
"""CPU address and scalar staging model; no full NLLLoss implementation."""
import json


def fetch(flat_input, target, weight, n, c, d, ignore_index):
    assert n > 0 and c > 0 and d > 0
    assert len(flat_input) == n * c * d and len(target) == n * d
    assert weight is None or len(weight) == c
    out, reads = [], []
    for row_idx, class_idx in enumerate(target):
        if class_idx == ignore_index:
            out.append(0)
            continue  # no input or weight read for ignored positions
        assert 0 <= class_idx < c  # required precondition; model rejects invalids
        n_idx, d_idx = divmod(row_idx, d)
        offset = (n_idx * c + class_idx) * d + d_idx
        reads.append((row_idx, class_idx, offset))
        w = 1 if weight is None else weight[class_idx]
        out.append(-flat_input[offset] * w)
    return out, reads


def check():
    cases = 0
    for n in (1, 2, 3):
        for c in (1, 3, 9):
            for d in (1, 2, 5):
                # Each coordinate has a distinct value, independent of flat offset.
                logical = [[[1000 * i + 10 * j + k for k in range(d)]
                            for j in range(c)] for i in range(n)]
                flat = [x for batch in logical for cls in batch for x in cls]
                targets = [-100 if r % 4 == 0 else (r * 2 + 1) % c
                           for r in range(n * d)]
                for weight in (None, [j + 1 for j in range(c)]):
                    out, reads = fetch(flat, targets, weight, n, c, d, -100)
                    expected = [0 if t == -100 else
                                -logical[r // d][t][r % d] *
                                (1 if weight is None else weight[t])
                                for r, t in enumerate(targets)]
                    assert out == expected
                    assert len(reads) == sum(t != -100 for t in targets)
                    assert all(0 <= off < len(flat) for _, _, off in reads)
                    cases += 1
    rejected = 0
    for target in (-1, 3):
        try:
            fetch([1, 2, 3], [target], None, 1, 3, 1, -100)
        except AssertionError:
            rejected += 1
    assert rejected == 2
    # The two selected queues reserve one 32-byte aligned element each.
    for element_bytes in (2, 4):
        each = ((element_bytes + 31) // 32) * 32
        assert each == 32 and 2 * each == 64
    result = {"model": "nll_scalar_address", "cases": cases,
              "invalid_nonignored_targets_rejected": rejected,
              "selected_queue_payload_bytes": 64, "device_test": False}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    check()
```

## 来源定位与证据强度

本卡是人工蒸馏的静态结构材料，来源为独立归档源码。下述 CPU 程序完整保留本卡需要的索引、存储或消费关系，可直接复制运行；它不是原设备算子的实现，也不执行 CANN 接口。没有为本卡运行新真机实验，没有采用归档注释中的历史成功、误差或性能数字。芯片与 SDK 范围尚未由本次证据确认，保留 `draft`、空平台范围与未新增 `verified`；公开附件提供静态核验入口，不把原件 SHA-256 伪装为远端 Git revision。

| 源 ID | 归档内相对文件 | 完整原文件 SHA-256 |
| --- | --- | --- |
| S1 | `level1/025_NLLLoss/kernel/nllloss_kernel.h` | `fee2fe4051032b8477b2143fe1878ed051a94961892ef298964eae3bffa6cf10` |
| S2 | `level1/025_NLLLoss/kernel/pybind11.cpp` | `4ba814c320eda621adc4a4dc8e4dcbb7763b246931fb08d81625777ee07de0a9` |
| S3 | `level1/025_NLLLoss/model_new_ascendc.py` | `bb21ddd9942b11d2209e617b627d545efe28a3530105f03cf25b78aacb72e910` |

| 源 ID | 本卡相关单元 | 原文件行号 |
| --- | --- | --- |
| S1 | ReadInputElem | 99–122 |
| S1 | ReadWeightElem | 127–144 |
| S1 | FetchRowContribution | 150–166 |
| S1 | selected queue allocation | 37–49 |
| S2 | launch GM storage-base pointers | 228–237 |

源 ID 与 SHA-256 绑定已读完整本地文件；行号用于人工对照，不把 SHA-256 当远程 Git revision。对外代码摘录只保留上述相关区段并删除注释，逐段变换在外带 `source_map.json` 中记录；摘录不是完整原文件的字节副本。正文内完整 CPU 模型是本卡的独立可执行检查，不能替代实际 SDK 编译、精度、并发、同步或性能验收。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/C031/source_map.json`；`cannbot_non_level4_split_review_20261019/evidence_public/C031/source_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/C031/nll_scalar_address.py`；`cannbot_non_level4_split_review_20261019/evidence_public/C031/cpu_check_result.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
