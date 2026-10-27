---
type: Runbook
title: 融合 Q/K 仿射参数的独立持久存储
description: 追踪四组仿射参数的独立分配、临时区复用及 query/key 消费分发，并提供可独立执行的 CPU 结构检查。
tags:
- task.implementation
- topic.memory
- topic.precision
status: draft
sources: []
created_at: '2026-10-17T19:12:37Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 融合 Q/K 仿射参数的独立持久存储

适用于同一融合核先加载多组仿射参数，再依次处理 query 与 key 的结构。临时加载区可以顺序复用，但后续各消费者仍要读取的参数必须保留到消费完成。`LocalTensor` 变量名不同或两次 `Get<float>()` 本身不证明底层存储不同，应追踪实际分配和每个消费者。

## 分配、物化与消费

归档 `Init` 分别为 `queryWeightFp32Buf_`、`queryBiasFp32Buf_`、`keyWeightFp32Buf_` 和 `keyBiasFp32Buf_` 分配 `D*sizeof(float)`。四次加载分别借用临时 `weightBuf_` / `biasBuf_`，随后 `PrepareParam` 把每组参数复制或转换到各自独立的 FP32 目的缓冲。FP32 分支也执行复制，不能仅把 query 句柄绑定到会被 key 加载覆盖的临时区。

| 句柄 | 持久目的分配 | `ProcessRow` 分发 |
| --- | --- | --- |
| `queryWeight_` | `queryWeightFp32Buf_` | query 的 `ProcessQueryOrKey` |
| `queryBias_` | `queryBiasFp32Buf_` | query 的 `ProcessQueryOrKey` |
| `keyWeight_` | `keyWeightFp32Buf_` | key 的 `ProcessQueryOrKey` |
| `keyBias_` | `keyBiasFp32Buf_` | key 的 `ProcessQueryOrKey` |

`ProcessQueryOrKey` 将相应 weight/bias 传给 `ComputeNorm`；affine LayerNorm 分支消费 weight 与 bias，affine RMSNorm 分支只消费 weight。不能因统一函数签名含 bias 就宣称每种 norm 模式都使用它。以下独立模型把 norm 统计阶段缩成“已归一化行”，只检查这条 affine 数据流，不对原源码的均值、方差、RMS 或 RoPE 作数值认证。

四份参数的名义 FP32 容量合计 `4*D*sizeof(float)`，不包含队列、行临时区、转换缓冲、对齐和管理开销。是否可以进一步复用某块存储，取决于生产与最后消费的生命周期证明；不应由参数名字或一次局部成功决定。

## 可公开复核的机制单元

归档中 `PrepareParam` 的完整函数如下。其目的区在此前已独立分配；这里保留原 count 和同步语句，仅删除解释注释。该函数依赖工程中的 Ascend C 类型，不是一份独立编译或最小同步认证样例。

```cpp
    __aicore__ inline void PrepareParam(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &fp32Buf,
        int32_t count)
    {
        dst = fp32Buf.Get<float>();
        if constexpr (std::is_same_v<dataType, float>) {
            AscendC::DataCopy(dst, src.template ReinterpretCast<float>(), count);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
```

完整 CPU 程序中四组参数使用互异数值，加载 key 后故意覆盖两个临时区，验证四个持久目的与相应消费者仍正确，并构造共享临时句柄的反例。Python 的 float 在此只承载结构哨兵，不模拟 FP32 转换和设备队列。小型程序没有额外依赖，足以复核存储分离与分发关系；源文件的关键摘录及逐段映射同时放入公开审阅材料。

迁移到实际 `D`、dtype、SDK 或芯片时，需单独检查 `InitBuffer` 对齐、容量、DataCopy/Cast count 合同和生产消费依赖。模型对 `D=33` 的结构检查不表示该 Ascend C 搬运配置支持不对齐长度。[Norm 融合的 Affine 重排与流水边界](../../guides/vector/norm_fusion_affine_rounding_and_pipeline_limits.md)已有统计域和舍入讨论，本卡补充独立存储这一具体数据流。

## 可独立执行的 CPU 结构检查

将下面完整代码保存为 `independent_qk_affine_storage.py` 后执行 `python3 independent_qk_affine_storage.py`。它检验明示前提下的结构模型；输出中的 `device_test` 恒为 `false`。

```python
"""CPU model of separate affine storage and Q/K dispatch; no norm numeric claim."""
import json


def prepare(src, persistent):
    assert len(src) == len(persistent)
    persistent[:] = [float(x) for x in src]
    return persistent


def load_parameters(params):
    d = len(params["query_weight"])
    assert d > 0 and all(len(x) == d for x in params.values())
    storage = {name: [0.0] * d for name in params}
    temporary_weight, temporary_bias = [0] * d, [0] * d
    handles = {}
    for name in ("query_weight", "query_bias", "key_weight", "key_bias"):
        temporary = temporary_weight if name.endswith("weight") else temporary_bias
        temporary[:] = params[name]
        handles[name] = prepare(temporary, storage[name])
    return handles, temporary_weight, temporary_bias


def consume_already_normalized(row, weight, bias, norm_type):
    # Only the affine consumer is modeled; input is an already normalized row.
    if norm_type == "layer_affine":
        return [x * w + b for x, w, b in zip(row, weight, bias)]
    if norm_type == "rms_affine":
        return [x * w for x, w in zip(row, weight)]  # no bias consumer
    return list(row)


def check():
    cases = 0
    for d in (1, 3, 8, 33):
        params = {"query_weight": [i + 1 for i in range(d)],
                  "query_bias": [10 + i for i in range(d)],
                  "key_weight": [100 + i for i in range(d)],
                  "key_bias": [-100 - i for i in range(d)]}
        handles, tw, tb = load_parameters(params)
        assert len({id(x) for x in handles.values()}) == 4
        tw[:] = [999] * d
        tb[:] = [-999] * d
        assert all(handles[k] == [float(x) for x in v] for k, v in params.items())
        row = [2] * d
        for mode in ("layer_affine", "rms_affine", "none"):
            q = consume_already_normalized(row, handles["query_weight"], handles["query_bias"], mode)
            k = consume_already_normalized(row, handles["key_weight"], handles["key_bias"], mode)
            if mode == "layer_affine":
                assert q == [2 * (i + 1) + 10 + i for i in range(d)]
                assert k == [2 * (100 + i) - 100 - i for i in range(d)]
            elif mode == "rms_affine":
                assert q == [2 * (i + 1) for i in range(d)]
                assert k == [2 * (100 + i) for i in range(d)]
            else:
                assert q == k == row
            cases += 1
        # Negative fixture: two handles into one reused staging buffer alias.
        shared = list(params["query_weight"])
        wrong_q = shared
        shared[:] = params["key_weight"]
        assert wrong_q != params["query_weight"]
    result = {"model": "independent_qk_affine_storage", "cases": cases,
              "alias_negative_fixtures_detected": 4,
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
| S1 | `level7/004_NormRopeConcat/kernel/norm_rope_concat_kernel.h` | `4b126fbd457465bed9998ceaf6d4f23848be339f9a8c563ea133ad72c7e018b8` |
| S2 | `level7/004_NormRopeConcat/kernel/pybind11.cpp` | `d29a93027bee07f537162ce7f5ff72cdef21018128ca599b8b6f2417daa394d9` |
| S3 | `level7/004_NormRopeConcat/kernel/norm_rope_concat.cpp` | `6cfa915c77164999ec08a195c8e0e150988d8984a662679c040d334d7907ee49` |
| S4 | `level7/004_NormRopeConcat/model_new_ascendc.py` | `1aa593edc43f537babdb67275621c6316175de953b6a2a073dacf746352b5624` |

| 源 ID | 本卡相关单元 | 原文件行号 |
| --- | --- | --- |
| S1 | PrepareParam | 215–229 |
| S1 | ProcessRow | 506–513 |
| S1 | distinct persistent allocation | 76–80 |
| S1 | parameter loads | 90–115 |
| S1 | LayerNorm affine consumer | 315–318 |
| S1 | RMSNorm affine consumer | 333–335 |
| S1 | ProcessQueryOrKey parameter signature | 432–440 |
| S1 | ComputeNorm forwarding | 467–469 |
| S1 | persistent buffers | 543–546 |
| S1 | parameter handles | 557–560 |

源 ID 与 SHA-256 绑定已读完整本地文件；行号用于人工对照，不把 SHA-256 当远程 Git revision。对外代码摘录只保留上述相关区段并删除注释，逐段变换在外带 `source_map.json` 中记录；摘录不是完整原文件的字节副本。正文内完整 CPU 模型是本卡的独立可执行检查，不能替代实际 SDK 编译、精度、并发、同步或性能验收。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/C030/source_map.json`；`cannbot_non_level4_split_review_20261019/evidence_public/C030/source_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/C030/independent_qk_affine_storage.py`；`cannbot_non_level4_split_review_20261019/evidence_public/C030/cpu_check_result.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
