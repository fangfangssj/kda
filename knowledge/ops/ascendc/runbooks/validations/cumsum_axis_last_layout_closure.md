---
type: Runbook
title: Cumsum 扫描轴移至末轴的布局闭包
description: 按 wrapper、Host 物理重排、Kernel 连续扫描和逆置换追踪 Cumsum 指定轴的完整坐标映射。
tags:
- task.implementation
- topic.data_movement.layout_transform
- topic.implementation.tiling
status: draft
sources: []
created_at: '2026-10-18T08:14:49Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# Cumsum 扫描轴移至末轴的布局闭包

适用于非空、连续、所有维度为正且 `ndim>=1` 的输入，以及 `-ndim<=dim<ndim` 的扫描轴；传入 Kernel 的 GM 指针还必须绑定逻辑首元素。归档 Host 使用 `x_t.storage().data()`，所以只有最终 `x_t.storage_offset()==0`，或确实物化到满足该条件的新缓冲后，才能沿用这里的坐标闭包。仅有 contiguous 属性不能排除非零偏移视图。形状乘积、索引和 `P`、`S` 等 tiling 字段必须在各自整数类型内可表示；下述闭包不覆盖数值舍入策略。

## Wrapper → Host → Kernel → 输出

wrapper 的 `ModelNew.forward(x,dim)` 原样调用 `run_cumsum(x,dim)`。Host 对负轴加一次 `ndim` 后检查范围，令 `S=shape[dim]`，`P` 为其余轴长度乘积，构造 `perm=[其余轴原序...,dim]`。对多维输入，关键步骤是 `x.permute(perm).contiguous().reshape({P,S})`：真正发生重排且结果不连续时需要物化数据；若结果已经连续，`contiguous()` 可能沿用原存储，不能据函数名推定新分配或零偏移。单维输入直接 reshape 为 `[1,S]`，同样不能消除原有 storage offset。

Kernel 的 `ProcessColumn(col)` 从 `xGM_[col*S]` 读取 `S` 个有效元素，并将相同行写到 `yGM_[col*S]`。只有 Host 提供了正确基址和实际连续的 `[P,S]` 行布局时，该物理行才与指定扫描轴一致。Host 把输出 reshape 为按 `perm` 排列的形状，使用 `inv_perm[perm[i]]=i` 构造逆置换，再 `permute(inv_perm).contiguous()` 恢复原坐标。不能只验证输出 shape；元素坐标和逻辑起点也必须恢复。

归档 Kernel 的实际核号分工、DMA helper、队列同步与浮点累加都在这条坐标证明之外。它们需要独立绑定 launch 和目标平台验证。本卡不采用源码注释中的黄金结果波动、bit-exact、成功率或性能结论，也不推荐按这些历史注释更改参考算子。

## 检查与失效边界

下面的完整 CPU 模型逐坐标构造 axis-last 物理排列、连续行扫描和逆置换，并与直接沿原始逻辑轴求和的独立表达式比较。用小整数避免把 FP16/BF16 的舍入争议混入布局检验，覆盖单维、不同轴、负轴、单元素维及非方形张量。额外的 storage-base 模型拒绝非零偏移，并给出前缀哨兵被错读的反例：逻辑 `[1,2,3]` 应得到 `[1,3,6]`，误传含前缀99的存储基址则得到 `[99,100,102]`。这是静态 CPU 结构反例，未执行原 NPU 算子。非法维度和空维也在模型中拒绝；原 Host 对 `P=0` 会得到 `usedCoreNum=0` 并可能除零，不能据此宣称支持空输入。

[运行张量逻辑视图与物理存储](../../guides/registration/runtime_tensor_view_and_storage.md)提供通用 shape/stride 边界；[参考污染与 Scan 舍入证据](../../guides/testing/reference_mutation_and_scan_rounding_evidence.md)讨论数值与参考合同。本卡只补充上述完整布局路径。

## 可独立执行的 CPU 结构检查

将下面完整代码保存为 `axis_last_scan_layout.py` 后执行 `python3 axis_last_scan_layout.py`。它检验明示前提下的结构模型；输出中的 `device_test` 恒为 `false`。

```python
"""CPU layout closure of an axis-last scan; exact small integers only."""
import itertools
import json
import math


def flat_index(coord, shape):
    offset = 0
    for x, size in zip(coord, shape):
        offset = offset * size + x
    return offset


def axis_last_scan(flat, shape, dim):
    assert shape and all(s > 0 for s in shape)
    ndim = len(shape)
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim
    assert len(flat) == math.prod(shape)
    perm = [d for d in range(ndim) if d != dim] + [dim]
    perm_shape = [shape[d] for d in perm]
    inverse = [perm.index(d) for d in range(ndim)]
    rows = []
    for rest in itertools.product(*(range(s) for s in perm_shape[:-1])):
        row = []
        acc = 0
        for s in range(shape[dim]):
            perm_coord = rest + (s,)
            coord = tuple(perm_coord[inverse[d]] for d in range(ndim))
            acc += flat[flat_index(coord, shape)]
            row.append(acc)
        rows.extend(row)  # contiguous materialization of [P, S]
    restored = [None] * len(flat)
    for perm_coord in itertools.product(*(range(s) for s in perm_shape)):
        coord = tuple(perm_coord[inverse[d]] for d in range(ndim))
        restored[flat_index(coord, shape)] = rows[flat_index(perm_coord, perm_shape)]
    return restored, perm, inverse


def storage_base_scan(storage, storage_offset, shape, dim):
    """Model the archived storage().data() launch; reject a shifted view."""
    assert storage_offset == 0, "the archived launch must bind the logical first element"
    return axis_last_scan(storage[:math.prod(shape)], shape, dim)[0]


def check():
    cases = 0
    for shape in ((1,), (7,), (2, 3), (2, 3, 5), (2, 1, 3, 2)):
        flat = list(range(1, math.prod(shape) + 1))
        for dim in range(-len(shape), len(shape)):
            actual, perm, inv = axis_last_scan(flat, shape, dim)
            axis = dim % len(shape)
            expected = []
            for coord in itertools.product(*(range(s) for s in shape)):
                expected.append(sum(flat[flat_index(coord[:axis] + (i,) +
                                                    coord[axis + 1:], shape)]
                                    for i in range(coord[axis] + 1)))
            assert actual == expected
            assert all(perm[inv[d]] == d for d in range(len(shape)))
            cases += 1
    rejected = 0
    for shape, dim in (((0, 2), 0), ((2, 3), 2), ((2, 3), -3)):
        try:
            axis_last_scan([], shape, dim)
        except AssertionError:
            rejected += 1
    assert rejected == 3
    # A 1D contiguous view can begin inside a larger storage. Reshape does
    # not imply allocation; passing storage base reads the prefix instead.
    storage = [99, 1, 2, 3]
    logical_view = storage[1:]
    correct = axis_last_scan(logical_view, (3,), 0)[0]
    wrong_base = axis_last_scan(storage[:3], (3,), 0)[0]
    assert correct == [1, 3, 6] and wrong_base == [99, 100, 102]
    offset_rejected = 0
    try:
        storage_base_scan(storage, 1, (3,), 0)
    except AssertionError:
        offset_rejected += 1
    assert offset_rejected == 1
    assert storage_base_scan(logical_view.copy(), 0, (3,), 0) == correct
    result = {"model": "axis_last_scan_layout", "cases": cases,
              "invalid_layouts_rejected": rejected,
              "nonzero_storage_offset_rejected": offset_rejected,
              "storage_base_counterexample": {"logical": correct, "wrong_base": wrong_base},
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
| S1 | `level1/005_Cumsum/model_new_ascendc.py` | `4df9ffa76eadb139d512b9f5144ce927946152bef70b3a050cb37737011d0612` |
| S2 | `level1/005_Cumsum/kernel/pybind11.cpp` | `0899dd942c2a5e3b6c72ade0a1d78f361f7af602127bd0dc8e525d90420bf027` |
| S3 | `level1/005_Cumsum/kernel/cumsum_kernel.h` | `1f4dacabf5ee3c7d6c6291b082578727b797b3346d84f92498e9d82c0f82cf59` |
| S4 | `level1/005_Cumsum/kernel/cumsum_tiling.h` | `e391c565747bb0e49d47ac8b9ee2485bac81c404396b5f2aabc61d49560f410b` |

| 源 ID | 本卡相关单元 | 原文件行号 |
| --- | --- | --- |
| S1 | forward | 14–19 |
| S2 | run_cumsum input/axis/materialization | 25–68 |
| S2 | run_cumsum inverse layout | 121–142 |
| S2 | launch storage-base pointers | 104–118 |
| S3 | ProcessColumn contiguous input | 108–127 |
| S3 | ProcessColumn contiguous output | 175–185 |

源 ID 与 SHA-256 绑定已读完整本地文件；行号用于人工对照，不把 SHA-256 当远程 Git revision。对外代码摘录只保留上述相关区段并删除注释，逐段变换在外带 `source_map.json` 中记录；摘录不是完整原文件的字节副本。正文内完整 CPU 模型是本卡的独立可执行检查，不能替代实际 SDK 编译、精度、并发、同步或性能验收。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/C038/source_map.json`；`cannbot_non_level4_split_review_20261019/evidence_public/C038/source_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/C038/axis_last_scan_layout.py`；`cannbot_non_level4_split_review_20261019/evidence_public/C038/cpu_check_result.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
