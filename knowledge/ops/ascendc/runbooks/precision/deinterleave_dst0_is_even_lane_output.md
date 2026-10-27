---
type: "Runbook"
title: "DeInterleave 两路写反——按偶索引到 dst0 修正"
description: "Reg 交织拆分把 dst0/dst1 接反会让 interleaved 路线整行错；按 dst0=偶索引、dst1=奇索引接线。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.layout_transform"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DeInterleave 两路写反——按偶索引到 dst0 修正

## 现象

在 Ascend950 Reg 直调 kernel 中，half 配对路线已经能得到正确结果，但 interleaved 路线在合法搬运宽度下仍出现显著数值不匹配。工具探针报告 float32 interleaved 的 `max diff=9.220378875732422`，实际输出与 golden 不呈现单纯的尾块或舍入误差。

当时代码用 `AscendC::DeInterleave(qO, qE, src, D)` 拆分交织行，随后却把 `qE` 当作偶索引流、`qO` 当作奇索引流送进旋转。只交换 `DeInterleave` 的两个目标参数后，同类 float32 探针 `max diff=0.0`。

## 适用场景

- 工程形态：Ascend C kernel 直调；
- API 路径：`AscendC::DeInterleave` → Reg 计算 → `AscendC::Interleave`；
- 验证过的平台与 CANN 版本：950（Ascend950PR / dav-3510）+ CANN 9.0.1；
- 适用粒度：`operator_family`，适用于偶/奇位置拆分后执行成对计算再交织回写的 kernel；
- 路线优先级：`recommended_fastpath`，正确接线后的 Reg 数据重排是本轨迹验证过的主路线；
- 明确不适用 / 未验证场景：不同 API 版本、LocalTensor 非连续视图、低于 API 最小处理粒度的输入未由本卡覆盖。

## 根因

该版本 `DeInterleave(dst0, dst1, src, count)` 的 `dst0` 接收 `src[0], src[2], ...`，`dst1` 接收 `src[1], src[3], ...`。把参数写成 `(qO, qE, ...)` 后，变量语义与实际数据流相反：旋转公式把奇索引值当成 even 分量，把偶索引值当成 odd 分量；后续 `Interleave` 再按 even/odd 顺序重组，因此整个 interleaved 分支错误。

## 解决配方

在变量命名与 API 目标顺序之间建立一一对应：`qE` 必须放在 `dst0`，`qO` 必须放在 `dst1`。以下为轨迹逐字恢复的完整 `ProcessOneTile` before/after；唯一语义修复是 `DeInterleave` 两个目标的顺序。

### before

代码证据：revision=`trace pre-fix snapshot (uncommitted)`；path=`apply_rotary_pos_emb.asc`；symbol=`ProcessOneTile<T>`；lines=`L259-L330`。

```cpp
template <typename T>
__aicore__ inline void ProcessOneTile(AscendC::TPipe &pipe,
                                     AscendC::GlobalTensor<T> &inGm, AscendC::GlobalTensor<T> &outGm,
                                     AscendC::GlobalTensor<T> &cosGm, AscendC::GlobalTensor<T> &sinGm,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &inQueue,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &cosQueue,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &sinQueue,
                                     AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> &outQueue,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsE,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsO,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsOutE,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsOutO,
                                     int64_t r, int64_t curTile,
                                     int64_t D, int64_t H, int64_t S, int64_t N,
                                     int64_t layout, int64_t mode)
{
    int64_t rGroupStart = (layout == 0) ? (r / N) : 0;
    int64_t sStart = (layout == 0) ? (rGroupStart % S) : (r % S);
    int64_t sCount = (layout == 0) ? (((r + curTile - 1) / N) - rGroupStart + 1) : curTile;

    AscendC::DataCopyExtParams rowCopy{1, static_cast<uint32_t>(curTile * D * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> rowPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams csCopy{1, static_cast<uint32_t>(sCount * H * sizeof(T)), 0, 0, 0};

    auto inLocal = inQueue.AllocTensor<T>();
    auto cosLocal = cosQueue.AllocTensor<T>();
    auto sinLocal = sinQueue.AllocTensor<T>();
    AscendC::DataCopyPad(inLocal, inGm[r * D], rowCopy, rowPad);
    AscendC::DataCopyPad(cosLocal, cosGm[sStart * H], csCopy, rowPad);
    AscendC::DataCopyPad(sinLocal, sinGm[sStart * H], csCopy, rowPad);
    inQueue.EnQue(inLocal);
    cosQueue.EnQue(cosLocal);
    sinQueue.EnQue(sinLocal);
    inLocal = inQueue.DeQue<T>();
    cosLocal = cosQueue.DeQue<T>();
    sinLocal = sinQueue.DeQue<T>();

    auto outLocal = outQueue.AllocTensor<T>();

    for (int64_t j = 0; j < curTile; ++j) {
        int64_t localS = (layout == 0) ? (((r + j) / N) - rGroupStart) : j;
        int64_t csOff = localS * H;
        int64_t rowOff = j * D;
        if (mode == 0) {
            // half: a = row[:H], b = row[H:], outA -> out[:H], outB -> out[H:].
            ComputeRopePair<T>(outLocal[rowOff], outLocal[rowOff + H],
                               inLocal[rowOff], inLocal[rowOff + H],
                               cosLocal[csOff], sinLocal[csOff], H);
        } else {
            // interleaved: DeInterleave q -> (qEven=even idx, qOdd=odd idx); rotate; Interleave -> out.
            auto qE = wsE.Get<T>();
            auto qO = wsO.Get<T>();
            auto outPacked = wsOutE.Get<T>();
            AscendC::DeInterleave(qO, qE, inLocal[rowOff], static_cast<int32_t>(D));
            AscendC::PipeBarrier<PIPE_V>();
            ComputeInterleave<T>(outPacked, qE, qO, cosLocal[csOff], sinLocal[csOff], H);
            AscendC::PipeBarrier<PIPE_V>();
            // Interleave(dst0,dst1, src0=outEven, src1=outOdd, count=H): reconstructs
            // [out0,out1,...] split into dst0 (first H) / dst1 (last H).
            AscendC::Interleave(outLocal[rowOff], outLocal[rowOff + H],
                                outPacked, outPacked[H], static_cast<int32_t>(H));
        }
    }

    outQueue.EnQue(outLocal);
    outLocal = outQueue.DeQue<T>();
    AscendC::DataCopyPad(outGm[r * D], outLocal, rowCopy);
    outQueue.FreeTensor(outLocal);
    inQueue.FreeTensor(inLocal);
    cosQueue.FreeTensor(cosLocal);
    sinQueue.FreeTensor(sinLocal);
}
```

### after

代码证据：revision=`trace post-fix snapshot (uncommitted)`；path=`apply_rotary_pos_emb.asc`；symbol=`ProcessOneTile<T>`；lines=`L259-L331`。

```cpp
template <typename T>
__aicore__ inline void ProcessOneTile(AscendC::TPipe &pipe,
                                     AscendC::GlobalTensor<T> &inGm, AscendC::GlobalTensor<T> &outGm,
                                     AscendC::GlobalTensor<T> &cosGm, AscendC::GlobalTensor<T> &sinGm,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &inQueue,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &cosQueue,
                                     AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> &sinQueue,
                                     AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> &outQueue,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsE,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsO,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsOutE,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &wsOutO,
                                     int64_t r, int64_t curTile,
                                     int64_t D, int64_t H, int64_t S, int64_t N,
                                     int64_t layout, int64_t mode)
{
    int64_t rGroupStart = (layout == 0) ? (r / N) : 0;
    int64_t sStart = (layout == 0) ? (rGroupStart % S) : (r % S);
    int64_t sCount = (layout == 0) ? (((r + curTile - 1) / N) - rGroupStart + 1) : curTile;

    AscendC::DataCopyExtParams rowCopy{1, static_cast<uint32_t>(curTile * D * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> rowPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams csCopy{1, static_cast<uint32_t>(sCount * H * sizeof(T)), 0, 0, 0};

    auto inLocal = inQueue.AllocTensor<T>();
    auto cosLocal = cosQueue.AllocTensor<T>();
    auto sinLocal = sinQueue.AllocTensor<T>();
    AscendC::DataCopyPad(inLocal, inGm[r * D], rowCopy, rowPad);
    AscendC::DataCopyPad(cosLocal, cosGm[sStart * H], csCopy, rowPad);
    AscendC::DataCopyPad(sinLocal, sinGm[sStart * H], csCopy, rowPad);
    inQueue.EnQue(inLocal);
    cosQueue.EnQue(cosLocal);
    sinQueue.EnQue(sinLocal);
    inLocal = inQueue.DeQue<T>();
    cosLocal = cosQueue.DeQue<T>();
    sinLocal = sinQueue.DeQue<T>();

    auto outLocal = outQueue.AllocTensor<T>();

    for (int64_t j = 0; j < curTile; ++j) {
        int64_t localS = (layout == 0) ? (((r + j) / N) - rGroupStart) : j;
        int64_t csOff = localS * H;
        int64_t rowOff = j * D;
        if (mode == 0) {
            // half: a = row[:H], b = row[H:], outA -> out[:H], outB -> out[H:].
            ComputeRopePair<T>(outLocal[rowOff], outLocal[rowOff + H],
                               inLocal[rowOff], inLocal[rowOff + H],
                               cosLocal[csOff], sinLocal[csOff], H);
        } else {
            // interleaved: DeInterleave q -> (qEven=even idx, qOdd=odd idx); rotate; Interleave -> out.
            auto qE = wsE.Get<T>();
            auto qO = wsO.Get<T>();
            auto outPacked = wsOutE.Get<T>();
            // Card: DeInterleave(dst0,dst1,src,D) -> dst0=even-index(src[0,2,..]), dst1=odd-index(src[1,3,..]).
            AscendC::DeInterleave(qE, qO, inLocal[rowOff], static_cast<int32_t>(D));
            AscendC::PipeBarrier<PIPE_V>();
            ComputeInterleave<T>(outPacked, qE, qO, cosLocal[csOff], sinLocal[csOff], H);
            AscendC::PipeBarrier<PIPE_V>();
            // Interleave(dst0,dst1, src0=outEven, src1=outOdd, count=H): reconstructs
            // [out0,out1,...] split into dst0 (first H) / dst1 (last H).
            AscendC::Interleave(outLocal[rowOff], outLocal[rowOff + H],
                                outPacked, outPacked[H], static_cast<int32_t>(H));
        }
    }

    outQueue.EnQue(outLocal);
    outLocal = outQueue.DeQue<T>();
    AscendC::DataCopyPad(outGm[r * D], outLocal, rowCopy);
    outQueue.FreeTensor(outLocal);
    inQueue.FreeTensor(inLocal);
    cosQueue.FreeTensor(cosLocal);
    sinQueue.FreeTensor(sinLocal);
}
```

## 设计决策

- 变量名直接表达数据语义：`qE` 只接收 `dst0`，`qO` 只接收 `dst1`；
- 保留 DeInterleave → Reg 旋转 → Interleave 路线，没有回退到标量逐元素配对；
- 放弃依据“odd-position/even-position”自然语言猜参数顺序的做法，因为一基序号与零基索引会产生歧义；
- 当前方案属于 `recommended_fastpath`，是本轨迹完成精度验证的 Reg 数据重排路线。

## 预防

- 使用 DeInterleave 时，在调用点写出 `dst0=src[0,2,...]`、`dst1=src[1,3,...]`；
- 先用递增序列做最小拆分探针，确认两路内容，再接入计算公式；
- 若 half 正确而 interleaved 全面错误，优先核对拆分/重组顺序，不要先改旋转算式或 cast。

## 验证信号

来源说明：Ascend950PR 上先以合法搬运宽度运行 float32 interleaved 探针，再仅交换 `DeInterleave` 的两个目标参数并干净重建。

- 修复前 float32 interleaved `max_diff=9.220378875732422`（来源平台：950；证据等级：tool_observed）；
- 交换目标参数后相同路线 `max_diff=0.0`（来源平台：950；证据等级：tool_observed）；
- 后续完整 dtype/mode 组合探针中，float32 两条路线均为 `max_diff=0.0`，B16 路线也在各自精度阈值内（来源平台：950；证据等级：tool_observed）；
- 最终完整评测集 20/20 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只修正两路语义，不解决 `DeInterleave` 的最小处理粒度、非连续 LocalTensor、barrier 或 UB 容量问题；
- a2/a3 与其他 CANN 版本未验证，推断可迁移前须用递增序列重新确认 dst 顺序；
- 若 API 输入输出的排列定义不同，必须以目标版本头文件和上板探针为准。

## 坏实践

- 根据“第一个/第二个、奇位置/偶位置”这类一基自然语言直接猜 `dst0/dst1`；
- 只验证输出 shape 或 kernel 无报错，不对拆分后的两路做内容探针；
- 看到整行错就先调容差或修改测试期望。

## 误诊

同一开发过程中曾把 interleaved 故障归因于 stale build 和 API 最小粒度；这些因素能解释特定小搬运宽度的运行错误，却不能解释合法宽度下稳定的数值错位。交换 dst 的 A/B 探针才闭合了当前根因。