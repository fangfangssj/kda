---
type: "Runbook"
title: "ACL 507035 — 空 TQue DeQue 触发 B16 路径异常"
description: "B16 分支只入队原始 cos/sin，却从未入队的 fp32 队列 DeQue 作为 scratch；按 dtype 区分 DeQue 与 AllocTensor 后消除大规模路径 507035。"
tags: ["task.debugging", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["TQue::DeQue"]
---

# ACL 507035 — 空 TQue DeQue 触发 B16 路径异常

## 现象

在 Ascend950 上对一个采用 TQue 双缓冲、B16→fp32 Reg Cast 的 Flash kernel 做最终评测时，小规模 `pytest` 已显示 `52 passed`，但通过远端 helper 运行 cann-bench 后，首个大规模用例在 `npuSynchronizeDevice` 报 `device error type 3, error code is 507035`，并导致后续设备操作级联失败；首轮报告没有 correctness pass。

进一步用同一 `.so` 分别探测大规模 float32 与 fp16 路径：float32 通过，fp16 失败。问题位于 `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc` 的 `ApplyRotaryPosEmbKernel<T>::Compute`：B16 路径的生产者只向 `cosRawQ_`/`sinRawQ_` 入队，消费者却无条件从未入队的 `cosfQ_`/`sinfQ_` 调用 `DeQue<float>()`。

## 适用场景

- 工程形态：Ascend C kernel 直调；TQue 流水按 dtype 分支选择“直接消费 fp32”或“B16 搬入后 Cast 到 fp32 scratch”。
- API 路径：`TQue::AllocTensor` / `EnQue` / `DeQue` / `FreeTensor`，以及 B16→fp32 的 AscendC::Reg Cast。
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1，与 frontmatter `platforms` 一致。
- 适用粒度：`operator_family`；适用于生产者在不同 dtype 分支向不同 TQue 入队，但消费者共享一段 Compute 逻辑的 kernel。
- 路线优先级：`recommended_fastpath`；按真实生产者选择 `DeQue`，仅对未入队的计算 scratch 使用 `AllocTensor`。
- 明确不适用 / 未验证场景：若所有 dtype 都对同一队列执行了 `AllocTensor→EnQue`，消费者正常 `DeQue` 不属于此问题；A2/A3 与其他 CANN 版本未验证。

## 根因

TQue 的 `DeQue` 是消费者动作，必须对应此前生产者对同一队列的 `EnQue`。这里 float32 分支的 `LoadCosTile` 会填充并 `EnQue` `cosfQ_`/`sinfQ_`，所以 `DeQue<float>()` 合法；B16 分支则只 `EnQue` `cosRawQ_`/`sinRawQ_`，`cosfQ_`/`sinfQ_` 只是 Cast 目标 scratch，从未入队。

旧实现无条件对 `cosfQ_`/`sinfQ_` 调用 `DeQue`，等价于从空消费队列取 Tensor，再把返回空间当作 `AllocTensor` 使用。小 tile 可能因 UB/队列状态偶然未暴露，进入大规模多 tile B16 路径后则触发 Vector Core 异常并以 ACL 507035 上报。修复必须在队列获取处按 dtype 分支：float32 对已经入队的 fp32 队列 `DeQue`；B16 对 fp32 scratch 队列 `AllocTensor`，同时从已入队的 raw B16 队列 `DeQue`，Cast 完成后再计算。

## 解决配方

把逐行计算抽为 `ComputeRows`，使 queue 生命周期只在 `Compute` 中按 dtype 分支一次。代码均为轨迹事件流中的实际写入，未改写。

### before

代码证据：revision=`0bab7a2`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbol=`ApplyRotaryPosEmbKernel<T>::Compute`；lines=`274-356`。该完整函数证明 `cosfQ_`/`sinfQ_` 在 dtype 判断前被无条件 DeQue。

```cpp
    __aicore__ inline void Compute(uint32_t rowOff, uint32_t rows)
    {
        auto qIn = inQueueQ_.DeQue<T>();
        auto kIn = inQueueK_.DeQue<T>();
        auto qOut = outQueueQ_.AllocTensor<T>();
        auto kOut = outQueueK_.AllocTensor<T>();
        uint32_t dU = static_cast<uint32_t>(d_);
        uint32_t d2U = static_cast<uint32_t>(halfD_);
        auto cosL = cosfQ_.DeQue<float>();
        auto sinL = sinfQ_.DeQue<float>();
        __ubuf__ float *cBase = (__ubuf__ float *)cosL.GetPhyAddr();
        __ubuf__ float *sBase = (__ubuf__ float *)sinL.GetPhyAddr();

        if constexpr (!std::is_same<T, float>::value) {
            auto cR = cosRawQ_.DeQue<T>();
            auto sR = sinRawQ_.DeQue<T>();
            // cast the whole tile's cos/sin (half) -> fp32 cosfQ/sinfQ buffer
            CastB16ToF32<T>(cBase, (__ubuf__ T *)cR.GetPhyAddr(), rows * d2U);
            CastB16ToF32<T>(sBase, (__ubuf__ T *)sR.GetPhyAddr(), rows * d2U);
            cosRawQ_.FreeTensor(cR);
            sinRawQ_.FreeTensor(sR);
            AscendC::PipeBarrier<PIPE_V>();
        }

        auto pp = partnerBuf_.Get<uint32_t>();
        auto sg = signBuf_.Get<float>();
        auto ex = expandBuf_.Get<uint32_t>();
        __ubuf__ uint32_t *ppA = (__ubuf__ uint32_t *)pp.GetPhyAddr();
        __ubuf__ float *sgA = (__ubuf__ float *)sg.GetPhyAddr();
        __ubuf__ uint32_t *exA = (__ubuf__ uint32_t *)ex.GetPhyAddr();

        for (uint32_t i = 0; i < rows; ++i) {
            __ubuf__ float *cRow = cBase + i * d2U;
            __ubuf__ float *sRow = sBase + i * d2U;
            if constexpr (std::is_same<T, float>::value) {
                __ubuf__ float *qInRow = (__ubuf__ float *)qIn.GetPhyAddr() + i * dU;
                __ubuf__ float *kInRow = (__ubuf__ float *)kIn.GetPhyAddr() + i * dU;
                __ubuf__ float *qOutRow = (__ubuf__ float *)qOut.GetPhyAddr() + i * dU;
                __ubuf__ float *kOutRow = (__ubuf__ float *)kOut.GetPhyAddr() + i * dU;
                if (mode_ == 0) {
                    ComputeRopeHalf(qOutRow, qInRow, cRow, sRow, d2U);
                    ComputeRopeHalf(kOutRow, kInRow, cRow, sRow, d2U);
                } else {
                    ComputeRopeInter(qOutRow, qInRow, cRow, sRow, ppA, sgA, exA, dU);
                    ComputeRopeInter(kOutRow, kInRow, cRow, sRow, ppA, sgA, exA, dU);
                }
            } else {
                auto qf = qfRowBuf_.Get<float>();
                auto kf = kfRowBuf_.Get<float>();
                auto qof = qofRowBuf_.Get<float>();
                auto kof = kofRowBuf_.Get<float>();
                __ubuf__ T *qInRow = (__ubuf__ T *)qIn.GetPhyAddr() + i * dU;
                __ubuf__ T *kInRow = (__ubuf__ T *)kIn.GetPhyAddr() + i * dU;
                __ubuf__ T *qOutRow = (__ubuf__ T *)qOut.GetPhyAddr() + i * dU;
                __ubuf__ T *kOutRow = (__ubuf__ T *)kOut.GetPhyAddr() + i * dU;
                __ubuf__ float *qfA = (__ubuf__ float *)qf.GetPhyAddr();
                __ubuf__ float *kfA = (__ubuf__ float *)kf.GetPhyAddr();
                __ubuf__ float *qofA = (__ubuf__ float *)qof.GetPhyAddr();
                __ubuf__ float *kofA = (__ubuf__ float *)kof.GetPhyAddr();
                CastB16ToF32<T>(qfA, qInRow, dU);
                CastB16ToF32<T>(kfA, kInRow, dU);
                AscendC::PipeBarrier<PIPE_V>();
                if (mode_ == 0) {
                    ComputeRopeHalf(qofA, qfA, cRow, sRow, d2U);
                    ComputeRopeHalf(kofA, kfA, cRow, sRow, d2U);
                } else {
                    ComputeRopeInter(qofA, qfA, cRow, sRow, ppA, sgA, exA, dU);
                    ComputeRopeInter(kofA, kfA, cRow, sRow, ppA, sgA, exA, dU);
                }
                AscendC::PipeBarrier<PIPE_V>();
                CastF32ToB16<T>(qOutRow, qofA, dU);
                CastF32ToB16<T>(kOutRow, kofA, dU);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        cosfQ_.FreeTensor(cosL);
        sinfQ_.FreeTensor(sinL);
        outQueueQ_.EnQue(qOut);
        outQueueK_.EnQue(kOut);
        inQueueQ_.FreeTensor(qIn);
        inQueueK_.FreeTensor(kIn);
    }
```

### after

代码证据：revision=`31fdcd2`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbols=`ApplyRotaryPosEmbKernel<T>::ComputeRows, ApplyRotaryPosEmbKernel<T>::Compute`；lines=`274-376`。两个完整函数共同证明逐行算法未改变，只有 queue 获取和 Cast 生命周期按 dtype 分开。

```cpp
    __aicore__ inline void ComputeRows(const AscendC::LocalTensor<T> &qIn,
                                       const AscendC::LocalTensor<T> &kIn,
                                       AscendC::LocalTensor<T> &qOut,
                                       AscendC::LocalTensor<T> &kOut,
                                       __ubuf__ float *cBase, __ubuf__ float *sBase,
                                       uint32_t rows, uint32_t dU, uint32_t d2U,
                                       __ubuf__ uint32_t *ppA, __ubuf__ float *sgA,
                                       __ubuf__ uint32_t *exA)
    {
        for (uint32_t i = 0; i < rows; ++i) {
            __ubuf__ float *cRow = cBase + i * d2U;
            __ubuf__ float *sRow = sBase + i * d2U;
            if constexpr (std::is_same<T, float>::value) {
                __ubuf__ float *qInRow = (__ubuf__ float *)qIn.GetPhyAddr() + i * dU;
                __ubuf__ float *kInRow = (__ubuf__ float *)kIn.GetPhyAddr() + i * dU;
                __ubuf__ float *qOutRow = (__ubuf__ float *)qOut.GetPhyAddr() + i * dU;
                __ubuf__ float *kOutRow = (__ubuf__ float *)kOut.GetPhyAddr() + i * dU;
                if (mode_ == 0) {
                    ComputeRopeHalf(qOutRow, qInRow, cRow, sRow, d2U);
                    ComputeRopeHalf(kOutRow, kInRow, cRow, sRow, d2U);
                } else {
                    ComputeRopeInter(qOutRow, qInRow, cRow, sRow, ppA, sgA, exA, dU);
                    ComputeRopeInter(kOutRow, kInRow, cRow, sRow, ppA, sgA, exA, dU);
                }
            } else {
                auto qf = qfRowBuf_.Get<float>();
                auto kf = kfRowBuf_.Get<float>();
                auto qof = qofRowBuf_.Get<float>();
                auto kof = kofRowBuf_.Get<float>();
                __ubuf__ T *qInRow = (__ubuf__ T *)qIn.GetPhyAddr() + i * dU;
                __ubuf__ T *kInRow = (__ubuf__ T *)kIn.GetPhyAddr() + i * dU;
                __ubuf__ T *qOutRow = (__ubuf__ T *)qOut.GetPhyAddr() + i * dU;
                __ubuf__ T *kOutRow = (__ubuf__ T *)kOut.GetPhyAddr() + i * dU;
                __ubuf__ float *qfA = (__ubuf__ float *)qf.GetPhyAddr();
                __ubuf__ float *kfA = (__ubuf__ float *)kf.GetPhyAddr();
                __ubuf__ float *qofA = (__ubuf__ float *)qof.GetPhyAddr();
                __ubuf__ float *kofA = (__ubuf__ float *)kof.GetPhyAddr();
                CastB16ToF32<T>(qfA, qInRow, dU);
                CastB16ToF32<T>(kfA, kInRow, dU);
                AscendC::PipeBarrier<PIPE_V>();
                if (mode_ == 0) {
                    ComputeRopeHalf(qofA, qfA, cRow, sRow, d2U);
                    ComputeRopeHalf(kofA, kfA, cRow, sRow, d2U);
                } else {
                    ComputeRopeInter(qofA, qfA, cRow, sRow, ppA, sgA, exA, dU);
                    ComputeRopeInter(kofA, kfA, cRow, sRow, ppA, sgA, exA, dU);
                }
                AscendC::PipeBarrier<PIPE_V>();
                CastF32ToB16<T>(qOutRow, qofA, dU);
                CastF32ToB16<T>(kOutRow, kofA, dU);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Compute(uint32_t /*rowOff*/, uint32_t rows)
    {
        auto qIn = inQueueQ_.DeQue<T>();
        auto kIn = inQueueK_.DeQue<T>();
        auto qOut = outQueueQ_.AllocTensor<T>();
        auto kOut = outQueueK_.AllocTensor<T>();
        uint32_t dU = static_cast<uint32_t>(d_);
        uint32_t d2U = static_cast<uint32_t>(halfD_);

        auto pp = partnerBuf_.Get<uint32_t>();
        auto sg = signBuf_.Get<float>();
        auto ex = expandBuf_.Get<uint32_t>();
        __ubuf__ uint32_t *ppA = (__ubuf__ uint32_t *)pp.GetPhyAddr();
        __ubuf__ float *sgA = (__ubuf__ float *)sg.GetPhyAddr();
        __ubuf__ uint32_t *exA = (__ubuf__ uint32_t *)ex.GetPhyAddr();

        if constexpr (std::is_same<T, float>::value) {
            // cosfQ/sinfQ were EnQue'd (filled) by LoadCosTile
            auto cosL = cosfQ_.DeQue<float>();
            auto sinL = sinfQ_.DeQue<float>();
            __ubuf__ float *cBase = (__ubuf__ float *)cosL.GetPhyAddr();
            __ubuf__ float *sBase = (__ubuf__ float *)sinL.GetPhyAddr();
            ComputeRows(qIn, kIn, qOut, kOut, cBase, sBase, rows, dU, d2U, ppA, sgA, exA);
            cosfQ_.FreeTensor(cosL);
            sinfQ_.FreeTensor(sinL);
        } else {
            // cosfQ/sinfQ are scratch fp32 buffers: Alloc, cast the raw B16 cos/sin tile into them
            auto cosL = cosfQ_.AllocTensor<float>();
            auto sinL = sinfQ_.AllocTensor<float>();
            auto cR = cosRawQ_.DeQue<T>();
            auto sR = sinRawQ_.DeQue<T>();
            __ubuf__ float *cBase = (__ubuf__ float *)cosL.GetPhyAddr();
            __ubuf__ float *sBase = (__ubuf__ float *)sinL.GetPhyAddr();
            CastB16ToF32<T>(cBase, (__ubuf__ T *)cR.GetPhyAddr(), rows * d2U);
            CastB16ToF32<T>(sBase, (__ubuf__ T *)sR.GetPhyAddr(), rows * d2U);
            cosRawQ_.FreeTensor(cR);
            sinRawQ_.FreeTensor(sR);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeRows(qIn, kIn, qOut, kOut, cBase, sBase, rows, dU, d2U, ppA, sgA, exA);
            cosfQ_.FreeTensor(cosL);
            sinfQ_.FreeTensor(sinL);
        }

        outQueueQ_.EnQue(qOut);
        outQueueK_.EnQue(kOut);
        inQueueQ_.FreeTensor(qIn);
        inQueueK_.FreeTensor(kIn);
    }
```

## 设计决策

- 选择把纯逐行计算抽到 `ComputeRows`，让队列生命周期集中在 `Compute`；这样 dtype 分支不会复制整段数学逻辑。
- float32 保留 `DeQue`，因为生产者已经把 fp32 tile 入队；B16 改为 `AllocTensor` 获取 Cast 目标 scratch，再从真正入队的 raw 队列 `DeQue`。
- 没有用降低 tile、改单缓冲或全局同步掩盖问题；这些尝试不能修复生产者/消费者不匹配。
- 当前方案属于 `recommended_fastpath`：修正队列语义后保留原 Reg 计算和流水结构。

## 预防

- 为每个 TQue 画出逐 dtype 的 `AllocTensor → EnQue → DeQue → FreeTensor` 状态链；任何 `DeQue` 都必须能在同一分支找到对应 `EnQue`。
- 当小规模通过、大规模 B16 失败而 float32 通过时，优先比较 dtype 分支的队列生产者，不要只调 tile 大小。
- scratch 缓冲必须用 `AllocTensor` 或 TBuf 获取；不得把空队列上的 `DeQue` 当作隐式分配。

## 验证信号

来源是一次 Ascend950 Flash kernel 的最终评测与大规模故障隔离；修复前后均通过远端 helper 在同一物理平台执行。

- 修复前 cann-bench 首轮报告无 correctness pass，首个失败包含 `device error type 3, error code is 507035`（来源平台：950；证据等级：tool_observed）。
- 故障隔离中，大规模 float32 路径通过而大规模 fp16 路径失败，锁定 B16 专属 queue/Cast 分支（来源平台：950；证据等级：tool_observed）。
- 修复后构建目标成功，pytest 为 `52 passed`（来源平台：950；证据等级：tool_observed）。
- 修复后跨 fp16、bf16、fp32，half/interleaved 与两种布局的大规模探针及连续多调用探针全部输出 `OK`（来源平台：950；证据等级：tool_observed）。
- 修复后 cann-bench 20 个用例全部通过，失败数为 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本配方针对“不同 dtype 分支生产不同队列，而共享消费者无条件 DeQue”的生命周期错误；正常成对的 TQue 不应改成 AllocTensor。
- 507035 是上层汇总错误码，UB 越界、非法地址等其他根因也可能报同码；必须先用 dtype/规模对照确认本卡信号。
- A2、A3 未验证，推断队列契约可迁移，但错误码和触发规模未验证（推断可迁移）。

## 坏实践

- 看到 507035 就反复缩小 `tileRows`、关闭双缓冲或增加设备同步；这些操作可能改变复现概率，但没有建立队列生产者与消费者的对应关系。
- 因小规模测试通过就认定队列生命周期正确；空队列消费可能依赖偶然 UB 状态，必须覆盖大规模多 tile 和连续多调用。

## 误诊

轨迹中先后怀疑 UB 预算、block 数、逐行 DMA、stream 生命周期和 pipeline depth。对照实验中 float32 大规模通过、B16 大规模失败，最终才把差异收敛到 B16 独有的 `cosRawQ_ → Cast → cosfQ_` 路径。