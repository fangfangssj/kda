---
type: "Runbook"
title: "DataCopyPad stride 表示 block 间隙"
description: "二维跨步 DataCopyPad 中 local stride 是相邻 block 间额外 gap；把对齐补齐字节再次作为 gap 会二次跳过 UB 空间并导致后续行读写垃圾。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# DataCopyPad stride 表示 block 间隙

## 现象

共享库已构建并可加载，但在上板运行二维跨步与短行输入时，输出第一位置往往正确，后续 values 和 int64 indices 变为随机大数；边界测试出现多项 mismatch。此时 kernel 把 GM 行间隔填到 `srcStride/dstStride`，同时又把 `padRowBytes - rowBytes` 填为 UB 侧 stride。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`DataCopyPad`、`DataCopyExtParams`、`srcStride`、`dstStride` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

这里的 stride 字段是相邻 block 有效 payload 之间的额外 gap。DataCopyPad 已按硬件块粒度组织 UB 行；再写入 `padRowBytes-rowBytes` 会把对齐差重复计入地址推进。以有效字节数 `L`、对齐后行宽 `P` 为例，正确下一行地址是当前行起点加 `P`；错误配置又加 `(P-L)`，实际推进变成 `P+(P-L)`，因此从第二行开始访问错误位置。

## 解决配方

### Before

```cpp
            uint32_t tg = (uint32_t)tg64;
            uint32_t rowBytesT = tg * B;
            uint32_t padRowBytesT = roundUp32(rowBytesT);     // UB row stride for in/outVal (native)
            uint32_t padRowElemsT = padRowBytesT / B;          // index stride for in/outVal
            // init running for this j-group
            for (uint32_t k = 0; k < tg; ++k) {
                runF[k] = posInf;
                runI[k] = 0;
            }

            for (int64_t r0 = 0; r0 < R; r0 += Tr) {
                int64_t tr64 = R - r0;
                if (tr64 > Tr) tr64 = Tr;
                uint32_t tr = (uint32_t)tr64;
                uint32_t tileElem = tr * tg;
                int64_t srcOff = base + r0 * inner + g0;

                // ---- load inTile [tr][tg] (GM->UB), 2D strided padded ----
                AscendC::DataCopyExtParams inParams{
                    (uint16_t)tr, rowBytesT,
                    (uint32_t)((inner - tg64) * (int64_t)B),   // GM srcStride gap (bytes)
                    padRowBytesT - rowBytesT,                  // UB dstStride gap (pad)
                    0};
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(inLocal, inGm[srcOff], inParams, padParams);
                AscendC::PipeBarrier<PIPE_ALL>();

                // ---- scalar scan (compare in fp32 via scalar cast) ----
                for (uint32_t r = 0; r < tr; ++r) {
                    int64_t gi = r0 + (int64_t)r;
                    __ubuf__ T *inRow = inPtr + (uint64_t)r * padRowElemsT;
                    __ubuf__ T *outRow = outValPtr + (uint64_t)r * padRowElemsT;
                    __ubuf__ int32_t *idx32Row = idx32Ptr + (uint64_t)r * tg; // contiguous
                    for (uint32_t k = 0; k < tg; ++k) {
                        T xn = inRow[k];
                        float xf;
                        if constexpr (std::is_same<T, bfloat16_t>::value) {
                            // bf16 -> fp32 is exact bit shift (shared sign/exp layout);
                            // bisheng backend does not support scalar (float)bf16.
                            uint16_t raw = ((__ubuf__ uint16_t *)inRow)[k];
                            uint32_t fbits = ((uint32_t)raw) << 16;
                            union { uint32_t u; float f; } cv; cv.u = fbits; xf = cv.f;
                        } else if constexpr (std::is_same<T, float>::value) {
                            xf = xn;
                        } else {
                            xf = (float)xn;   // half / int32
                        }
                        bool runNan = (runF[k] != runF[k]);
                        bool upd = (xf != xf) || (!runNan && xf <= runF[k]);
                        if (upd) {
                            runF[k] = xf;
                            runV[k] = xn;
                            runI[k] = (int32_t)gi;
                        }
                        outRow[k] = runV[k];
                        idx32Row[k] = runI[k];
                    }
                }

                AscendC::PipeBarrier<PIPE_V>(); // idx32 visible to asc_int322int64
                // ---- int32 -> int64 ----
                asc_int322int64(idx64Ptr, idx32Ptr, tileElem);
                AscendC::PipeBarrier<PIPE_ALL>(); // V -> MTE3

                // ---- store outVal (native, padded 2D) ----
                AscendC::DataCopyExtParams outValParams{
                    (uint16_t)tr, rowBytesT,
                    padRowBytesT - rowBytesT,                // UB srcStride gap
                    (uint32_t)((inner - tg64) * (int64_t)B), // GM dstStride gap
                    0};
                AscendC::DataCopyPad(valGm[srcOff], outValLocal, outValParams);

                // ---- store idx64 (contiguous UB -> 2D strided GM, int64) ----
                uint32_t rowBytes64 = tg * B64;
                AscendC::DataCopyExtParams outIdxParams{
                    (uint16_t)tr, rowBytes64,
                    0,                                       // UB contiguous (srcStride gap 0)
                    (uint32_t)((inner - tg64) * (int64_t)B64), // GM dstStride gap
                    0};
                AscendC::DataCopyPad(idxGm[srcOff], idx64Local, outIdxParams);

                AscendC::PipeBarrier<PIPE_ALL>(); // MTE3 -> V (free out buffers, WAR)
            }
```

### After

```cpp
            uint32_t tg = (uint32_t)tg64;
            uint32_t rowBytesT = tg * B;
            uint32_t padRowBytesT = roundUp32(rowBytesT);     // UB row stride for in/outVal (native)
            uint32_t padRowElemsT = padRowBytesT / B;          // index stride for in/outVal
            uint32_t rowBytes64 = tg * B64;
            uint32_t padRowBytes64 = roundUp32(rowBytes64);    // UB row stride for idx64
            uint32_t padRowElems64 = padRowBytes64 / B64;
            // init running for this j-group
            for (uint32_t k = 0; k < tg; ++k) {
                runF[k] = posInf;
                runI[k] = 0;
            }

            for (int64_t r0 = 0; r0 < R; r0 += Tr) {
                int64_t tr64 = R - r0;
                if (tr64 > Tr) tr64 = Tr;
                uint32_t tr = (uint32_t)tr64;
                int64_t srcOff = base + r0 * inner + g0;

                // ---- load inTile [tr][tg] (GM->UB), 2D strided; UB rows packed (gap 0) ----
                AscendC::DataCopyExtParams inParams{
                    (uint16_t)tr, rowBytesT,
                    (uint32_t)((inner - tg64) * (int64_t)B),   // GM srcStride gap (bytes)
                    0,                                         // UB dstStride gap 0 (rows packed at padRowBytesT)
                    0};
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(inLocal, inGm[srcOff], inParams, padParams);
                AscendC::PipeBarrier<PIPE_ALL>();

                // ---- scalar scan (compare in fp32) + direct int64 index write ----
                for (uint32_t r = 0; r < tr; ++r) {
                    int64_t gi = r0 + (int64_t)r;
                    __ubuf__ T *inRow = inPtr + (uint64_t)r * padRowElemsT;
                    __ubuf__ T *outRow = outValPtr + (uint64_t)r * padRowElemsT;
                    __ubuf__ int64_t *idx64Row = idx64Ptr + (uint64_t)r * padRowElems64;
                    for (uint32_t k = 0; k < tg; ++k) {
                        T xn = inRow[k];
                        float xf;
                        if constexpr (std::is_same<T, bfloat16_t>::value) {
                            // bf16 -> fp32 exact bit shift (shared sign/exp); scalar cast unsupported.
                            uint16_t raw = ((__ubuf__ uint16_t *)inRow)[k];
                            uint32_t fbits = ((uint32_t)raw) << 16;
                            union { uint32_t u; float f; } cv; cv.u = fbits; xf = cv.f;
                        } else if constexpr (std::is_same<T, float>::value) {
                            xf = xn;
                        } else {
                            xf = (float)xn;   // half / int32
                        }
                        bool runNan = (runF[k] != runF[k]);
                        bool upd = (xf != xf) || (!runNan && xf <= runF[k]);
                        if (upd) {
                            runF[k] = xf;
                            runV[k] = xn;
                            runI[k] = (int32_t)gi;
                        }
                        outRow[k] = runV[k];
                        idx64Row[k] = (int64_t)runI[k];
                    }
                }

                AscendC::PipeBarrier<PIPE_ALL>(); // V -> MTE3

                // ---- store outVal (native, padded 2D; UB rows packed gap 0) ----
                AscendC::DataCopyExtParams outValParams{
                    (uint16_t)tr, rowBytesT,
                    0,                                         // UB srcStride gap 0
                    (uint32_t)((inner - tg64) * (int64_t)B),   // GM dstStride gap
                    0};
                AscendC::DataCopyPad(valGm[srcOff], outValLocal, outValParams);

                // ---- store idx64 (padded UB -> 2D strided GM) ----
                AscendC::DataCopyExtParams outIdxParams{
                    (uint16_t)tr, rowBytes64,
                    0,                                         // UB srcStride gap 0
                    (uint32_t)((inner - tg64) * (int64_t)B64), // GM dstStride gap
                    0};
                AscendC::DataCopyPad(idxGm[srcOff], idx64Local, outIdxParams);

                AscendC::PipeBarrier<PIPE_ALL>(); // MTE3 -> V (free out buffers, WAR)
            }
```

## 系统信息

CANN 9.2.0 Ascend C Basic API；目标 dav-2201；相关文件为 `operators/cummin/cummin.asc` 与 `operators/cummin/test_cummin.py`。

## 设计决策

GM 继续使用真实跨行 gap；UB 端使用 gap=0，并用对齐后的元素跨度寻址各行。相邻 int64 输出同时按其自身对齐宽度寻址，避免复用 native dtype 的行跨度。

## 性能影响

本轨迹未对这一局部修复做独立性能归因；它只声明解除构建或正确性阻塞。

## 预防

在目标 CANN/SoC 上为相关 API、dtype 或搬运形态保留最小编译与上板用例，不以静态推断替代。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：定向探针的 f32 值在多个元素上偏离参考，索引同时出现大幅垃圾值；源任务的具体向量只保留在内部 provenance，不迁移。（来源平台：a3；证据等级: tool_observed）。
- 修复后：修正 stride 后 clean build 完成。（来源平台：a3；证据等级: tool_observed）。
- 修复后：ties、NaN 与边界矩阵均通过。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

该卡验证二维跨步 tile 的 gap 语义；超大 1D 连续路径仍需单独分支处理。950/a2 未验证，推断可迁移前需重验 SDK 参数定义。

## 坏实践

把 stride 当成“下一行完整跨度”而不是“有效块之间的额外间隔”，会在已对齐 UB 行上重复计算 padding。

## 误诊

只盯 int64 转换逻辑会误诊；values 与 indices 同时从第二位置开始损坏，指向共享的搬运/行寻址层。