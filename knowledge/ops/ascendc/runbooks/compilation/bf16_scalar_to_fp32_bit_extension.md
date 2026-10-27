---
type: "Runbook"
title: "bisheng 不支持 bf16 标量强转时用位布局精确提升到 fp32"
description: "bisheng 后端拒绝 bf16↔fp32 cast（dav-2201 标量强转与 950 SIMT 的 Reg::Cast/LocalTensor Cast/标量隐式转换三写法实测同句报错，950 classic op_kernel 路径另实测 bf16 标量有序比较 setcc setolt 不可选）；利用 bf16 与 fp32 共享符号位和指数布局，用位运算完成双向转换（反向 RINT 已在 950 闭合）。R4（950 arch35，CANN 9.2.0）：half 同样不可 static_cast，且 bfloat16_t/half 是编译器内建类型（__bf16）——既无转换运算符也无 .value 成员（vec.h 的 struct 定义不生效），只能 __builtin_memcpy 位视图转换。R5（，950 arch35，CANN 9.1.0）：GM 标量路径 static_cast<float>(bfloat16_t) 存在可编译但静默不转换的形态——输出退化为同通道常数（≈bias）、MERE≈1.0，正解是官方标量转换 AscendC::ToFloat/Cast（GetScalarBitcodeValue 位双关 + 左移 16，与本卡位布局同机制）；输出写回的函数式 cast 触发 exit 70 改 Duplicate→Cast(CAST_ROUND)；弱构建检查（find|head -1）会掩盖部分模板编译失败成垃圾输出。后续增量：调试 printf/诊断代码把 bf16 标量拉进 cast 的实例族（exit 70 + ld.lld _27.o 连锁，half 同写法不报错，union/__builtin_memcpy 位重解释规避）；纯 Vector kernel 的 count=1 向量 Cast 与 bf16 标量运算全景（setcc/fmul 不可选，编译期裁剪 bf16 注册的缓发路线）；模板形式 Cast<T>(float) 解析到向量重载——编译过、运行期 507035，标量转换必须用 kernel_scalar_convert.h 的非模板 Cast 重载。复现增量：标量 fadd 实测同样 Cannot select（的'推测 fadd 同类'落定，fp16/fp32 同代码通过）；kernel 内引用 host 侧 op:: 命名空间（aclnn/opdev/bfloat16.h 的 op::bfloat16）报 undeclared identifier 'op'；判型可正向枚举 half/float（else 即 bf16）避免直接命名类型，bf16 占位分支保三目标可编译、正确实现须向量 Cast。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-09-17T03:16:41Z"
platforms: ["a3", "950"]
---

# bisheng 不支持 bf16 标量强转时用位布局精确提升到 fp32

## 现象

在 dav-2201 上构建包含多 dtype 模板的 Ascend C prefix-scan kernel 时，half、float 与 int32 分支可通过前端，但 bf16 实例中的 `float xf = (float)xn` 触发 bisheng 后端致命错误：`not support bf16 type cast`，clang 以 exit code 70 退出。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`bfloat16_t`、`bf16`、`fp32` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，仅迁移到相同 BF16 位布局和设备侧标量提升路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。
- 平台与路线扩展（独立复现 R2，BenchHarnessEngramGateFusion，CANN 9.2.0）：950（arch35）+ bisheng/clang 15.0.5 的 SIMT 路径（`__simd_vf__` + `asc_vf_call` + `AscendC::Reg::*`）实测同一句后端错误 `not support bf16 type cast`（exit 70）——VF 体内 `AscendC::Reg::Cast`、VF 体外 LocalTensor `AscendC::Cast`、标量 `GetValue/SetValue` 隐式转换三种写法错误串逐字一致，最小化用例（仅从 bf16 张量读元素并转换）仍复现，确认为后端能力缺口而非调用方式问题。
- 入口/出口使用模式（950 实测）：GM 搬入后按 `__ubuf__ uint16_t*` 位流解释，`bf16_to_fp32`（16 位左移补零）展开到 fp32 工作缓冲、计算全程 fp32、出口一次 `fp32_to_bf16_rint`；转换函数标 `__aicore__ inline`。
- 平台与路线扩展（独立复现 R3，Cummin 算子开发会话，CANN 9.1.0）：950 classic op_kernel（`op_kernel/arch35/*.h` + LocalTensor GetValue/SetValue 标量扫描）路径实测后端缺口不止 cast——bf16 模板实例的标量 `<` 比较报 `Cannot select: i64 = setcc ... setolt:ch`（exit 70）；改 `static_cast<float>` 到 float 域比较后又报 `not support bf16 type cast`。即该后端既不能选 bf16 标量有序比较指令，也不支持 bf16↔float 标量强转，标量逻辑必须整体放进 float 域、进出存储用位级转换（完整往返模板见解决配方 R3 补充）。
- 类型系统扩展（独立复现 R4，BenchHarnessScatter，CANN 9.2.0 arch35 常规 vector kernel）：`half` 与 `bfloat16_t` 同样不可 `static_cast<float>`（报同一 `not support bf16 type cast`）；且两者是编译器**内建类型**（`bfloat16_t` 即 `__bf16`）——既无转换运算符也无 `.value` 之类结构成员（按"vec.h 里的 struct bfloat16_t"假设写 `v.value` 报 `member reference base type 'bfloat16_t' (aka '__bf16') is not a structure or union`），标量上下文只能 `__builtin_memcpy` 取位再按 IEEE 754 布局换算（三件套见解决配方 R4 补充）。
- 静默无效形态与官方标量转换（独立复现 R5，先前增量 三条轨迹，CANN 9.1.0 arch35）：GM 标量路径 `static_cast<float>(bfloat16_t)`（`GlobalTensor::GetValue` 返回值直接强转）存在**可编译但不做位级转换**的形态——读出的标量对乘加贡献为 0，输出退化为同通道常数（≈bias）、MERE≈1.0、fp16/fp32 同路径正常；输出写回方向的函数式 cast（`SetValue(0, D_T_X(finalResult))`）则触发既有硬错误 exit 70。正解是 AscendC 官方标量转换 `AscendC::ToFloat(const bfloat16_t&)` / `AscendC::Cast(const float&)`（`kernel_scalar_convert.h:158`，其 `Bf16ToFp32` 即 `GetScalarBitcodeValue<bfloat16_t,uint32_t>(bVal) << 16` 位双关——与本卡手写位布局同机制）或向量 `Cast` API；另实测"弱构建检查掩盖部分模板编译失败"会把硬错误伪装成精度垃圾（见解决配方 R5 补充）。

## 根因

该后端不支持 UB bf16 标量经 C++ cast 直接生成 fp32。bf16 的 16 位编码正好是 fp32 的高 16 位，因此对原始位 `raw` 做 `fbits = uint32_t(raw) << 16`，再按 float 重解释即可保持符号位、指数和 7 位尾数，不引入算术舍入。

R3 补充（950/CANN 9.1.0）：同一后端对 bf16 标量有序比较（`<`，setcc setolt）同样无指令可选——凡"bf16 标量算术/比较/转换"都应视为该后端的禁区，标量路径的合法形态是"位级转换 ↔ float 域计算"。

R4 补充（950/CANN 9.2.0）：CANN 9.2.0 的 kernel 编译器把 `bfloat16_t` 实现为内建 `__bf16`、`half` 为内建半精度类型——头文件 `vec.h` 里的 `struct bfloat16_t { uint16_t value; }` 定义并不是 kernel 编译时生效的类型；两者既不支持 C++ 算术/转换运算符，也没有结构成员可访问位。

## 解决配方

### Before

```cpp
template <typename T>
__global__ __aicore__ __vector__ void cummin_kernel(GM_ADDR input, GM_ADDR values, GM_ADDR indices,
    int64_t totalLanes, int64_t R, int64_t inner, int64_t numBlocks,
    int64_t Tr, int64_t Tg, int32_t dtypeKind)
{
    using UT = typename std::conditional<std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value, uint16_t, T>::type;
    (void)dtypeKind;
    if (totalLanes <= 0 || R <= 0) return;

    int64_t bid = AscendC::GetBlockIdx();
    int64_t lpb = (totalLanes + numBlocks - 1) / numBlocks;
    int64_t lStart = bid * lpb;
    if (lStart >= totalLanes) return;
    int64_t lEnd = lStart + lpb;
    if (lEnd > totalLanes) lEnd = totalLanes;

    int64_t oStart = lStart / inner;
    int64_t jStart = lStart % inner;
    int64_t oEnd = (lEnd - 1) / inner;
    int64_t jEnd = (lEnd - 1) % inner + 1; // exclusive end for last outer block

    const uint32_t B = (uint32_t)sizeof(T);            // native element bytes
    const uint32_t B32 = sizeof(int32_t);              // 4
    const uint32_t B64 = sizeof(int64_t);              // 8
    // Max tile element count (full group): Tr * Tg. Tail groups are <= this.
    const uint32_t tileElemsMax = (uint32_t)(Tr * Tg);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN>  inBuf;        // native, padded rows
    AscendC::TBuf<AscendC::TPosition::VECCALC> outValBuf;  // native, padded rows
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx32Buf;   // int32, contiguous (for asc_int322int64)
    AscendC::TBuf<AscendC::TPosition::VECOUT> idx64Buf;    // int64, contiguous
    AscendC::TBuf<AscendC::TPosition::VECCALC> runFBuf, runVBuf, runIBuf;

    pipe.InitBuffer(inBuf, tileElemsMax * B);
    pipe.InitBuffer(outValBuf, tileElemsMax * B);
    pipe.InitBuffer(idx32Buf, tileElemsMax * B32);
    pipe.InitBuffer(idx64Buf, tileElemsMax * B64);
    pipe.InitBuffer(runFBuf, Tg * sizeof(float));
    pipe.InitBuffer(runVBuf, Tg * B);
    pipe.InitBuffer(runIBuf, Tg * B32);

    auto inLocal = inBuf.Get<T>();
    auto outValLocal = outValBuf.Get<T>();
    auto idx32Local = idx32Buf.Get<int32_t>();
    auto idx64Local = idx64Buf.Get<int64_t>();

    AscendC::GlobalTensor<T> inGm, valGm;
    AscendC::GlobalTensor<int64_t> idxGm;
    inGm.SetGlobalBuffer((__gm__ T *)input);
    valGm.SetGlobalBuffer((__gm__ T *)values);
    idxGm.SetGlobalBuffer((__gm__ int64_t *)indices);

    __ubuf__ T *inPtr = (__ubuf__ T *)inLocal.GetPhyAddr();
    __ubuf__ T *outValPtr = (__ubuf__ T *)outValLocal.GetPhyAddr();
    __ubuf__ int32_t *idx32Ptr = (__ubuf__ int32_t *)idx32Local.GetPhyAddr();
    __ubuf__ int64_t *idx64Ptr = (__ubuf__ int64_t *)idx64Local.GetPhyAddr();
    __ubuf__ float *runF = (__ubuf__ float *)runFBuf.Get<float>().GetPhyAddr();
    __ubuf__ T *runV = (__ubuf__ T *)runVBuf.Get<T>().GetPhyAddr();
    __ubuf__ int32_t *runI = (__ubuf__ int32_t *)runIBuf.Get<int32_t>().GetPhyAddr();

    const float posInf = __builtin_inff();

    // helper: round up bytes to 32
    auto roundUp32 = [](uint32_t b) -> uint32_t { return (b + 31u) & ~31u; };

    for (int64_t o = oStart; o <= oEnd; ++o) {
        int64_t jLo = (o == oStart) ? jStart : 0;
        int64_t jHi = (o == oEnd) ? jEnd : inner;
        int64_t base = o * (R * inner);

        for (int64_t g0 = jLo; g0 < jHi; g0 += Tg) {
            int64_t tg64 = jHi - g0;
            if (tg64 > Tg) tg64 = Tg;
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
                        float xf = (float)xn;          // scalar cast (test: works on 2201 for half/bf16/int32)
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
        }
    }
}
```

### After

```cpp
template <typename T>
__global__ __aicore__ __vector__ void cummin_kernel(GM_ADDR input, GM_ADDR values, GM_ADDR indices,
    int64_t totalLanes, int64_t R, int64_t inner, int64_t numBlocks,
    int64_t Tr, int64_t Tg, int32_t dtypeKind)
{
    using UT = typename std::conditional<std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value, uint16_t, T>::type;
    (void)dtypeKind;
    if (totalLanes <= 0 || R <= 0) return;

    int64_t bid = AscendC::GetBlockIdx();
    int64_t lpb = (totalLanes + numBlocks - 1) / numBlocks;
    int64_t lStart = bid * lpb;
    if (lStart >= totalLanes) return;
    int64_t lEnd = lStart + lpb;
    if (lEnd > totalLanes) lEnd = totalLanes;

    int64_t oStart = lStart / inner;
    int64_t jStart = lStart % inner;
    int64_t oEnd = (lEnd - 1) / inner;
    int64_t jEnd = (lEnd - 1) % inner + 1; // exclusive end for last outer block

    const uint32_t B = (uint32_t)sizeof(T);            // native element bytes
    const uint32_t B32 = sizeof(int32_t);              // 4
    const uint32_t B64 = sizeof(int64_t);              // 8
    // Max tile element count (full group): Tr * Tg. Tail groups are <= this.
    const uint32_t tileElemsMax = (uint32_t)(Tr * Tg);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN>  inBuf;        // native, padded rows
    AscendC::TBuf<AscendC::TPosition::VECCALC> outValBuf;  // native, padded rows
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx32Buf;   // int32, contiguous (for asc_int322int64)
    AscendC::TBuf<AscendC::TPosition::VECOUT> idx64Buf;    // int64, contiguous
    AscendC::TBuf<AscendC::TPosition::VECCALC> runFBuf, runVBuf, runIBuf;

    pipe.InitBuffer(inBuf, tileElemsMax * B);
    pipe.InitBuffer(outValBuf, tileElemsMax * B);
    pipe.InitBuffer(idx32Buf, tileElemsMax * B32);
    pipe.InitBuffer(idx64Buf, tileElemsMax * B64);
    pipe.InitBuffer(runFBuf, Tg * sizeof(float));
    pipe.InitBuffer(runVBuf, Tg * B);
    pipe.InitBuffer(runIBuf, Tg * B32);

    auto inLocal = inBuf.Get<T>();
    auto outValLocal = outValBuf.Get<T>();
    auto idx32Local = idx32Buf.Get<int32_t>();
    auto idx64Local = idx64Buf.Get<int64_t>();

    AscendC::GlobalTensor<T> inGm, valGm;
    AscendC::GlobalTensor<int64_t> idxGm;
    inGm.SetGlobalBuffer((__gm__ T *)input);
    valGm.SetGlobalBuffer((__gm__ T *)values);
    idxGm.SetGlobalBuffer((__gm__ int64_t *)indices);

    __ubuf__ T *inPtr = (__ubuf__ T *)inLocal.GetPhyAddr();
    __ubuf__ T *outValPtr = (__ubuf__ T *)outValLocal.GetPhyAddr();
    __ubuf__ int32_t *idx32Ptr = (__ubuf__ int32_t *)idx32Local.GetPhyAddr();
    __ubuf__ int64_t *idx64Ptr = (__ubuf__ int64_t *)idx64Local.GetPhyAddr();
    __ubuf__ float *runF = (__ubuf__ float *)runFBuf.Get<float>().GetPhyAddr();
    __ubuf__ T *runV = (__ubuf__ T *)runVBuf.Get<T>().GetPhyAddr();
    __ubuf__ int32_t *runI = (__ubuf__ int32_t *)runIBuf.Get<int32_t>().GetPhyAddr();

    const float posInf = __builtin_inff();

    // helper: round up bytes to 32
    auto roundUp32 = [](uint32_t b) -> uint32_t { return (b + 31u) & ~31u; };

    for (int64_t o = oStart; o <= oEnd; ++o) {
        int64_t jLo = (o == oStart) ? jStart : 0;
        int64_t jHi = (o == oEnd) ? jEnd : inner;
        int64_t base = o * (R * inner);

        for (int64_t g0 = jLo; g0 < jHi; g0 += Tg) {
            int64_t tg64 = jHi - g0;
            if (tg64 > Tg) tg64 = Tg;
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
        }
    }
}
```

### R3 补充：多 dtype 完整往返模板（950/CANN 9.1.0 classic op_kernel 路径，verbatim）

R3 轨迹（Cummin 算子 `op_kernel/arch35/cummin.h`）把该模式推广为统一 `ToFloat<T>`/`FromFloat<T>` 模板：bf16 为左移 16 位（与 After 版一致），half 含 subnormal 规格化分支，int32 走 static_cast（精度边界见失效边界），标量扫描全程 float 域比较、只在读写存储时转换：

```cpp
namespace NsCummin {

using namespace AscendC;

template <typename T>
__aicore__ inline float ToFloat(T val)
{
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<float>(val);
    } else if constexpr (std::is_same_v<T, half>) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&val);
        uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exp = (bits >> 10) & 0x1Fu;
        uint32_t mant = bits & 0x03FFu;
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) { f32 = sign; }
            else {
                exp = 0;
                while ((mant & 0x0400u) == 0) { mant <<= 1; exp++; }
                mant &= 0x03FFu;
                f32 = sign | ((127 - 15 - exp + 1) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f32 = sign | 0x7F800000u | (mant << 13);
        } else {
            f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float result;
        *reinterpret_cast<uint32_t*>(&result) = f32;
        return result;
    } else {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&val);
        uint32_t f32 = static_cast<uint32_t>(bits) << 16;
        float result;
        *reinterpret_cast<uint32_t*>(&result) = f32;
        return result;
    }
}

template <typename T>
__aicore__ inline T FromFloat(float f)
{
    if constexpr (std::is_same_v<T, float>) {
        return f;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<int32_t>(f);
    } else if constexpr (std::is_same_v<T, half>) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&f);
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = bits & 0x007FFFFFu;
        uint16_t h;
        if (((bits >> 23) & 0xFFu) == 0xFFu) { h = static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13)); }
        else if (exp >= 31) { h = static_cast<uint16_t>(sign | 0x7C00u); }
        else if (exp <= 0) {
            if (exp < -10) { h = static_cast<uint16_t>(sign); }
            else { mant = (mant | 0x00800000u) >> (1 - exp); h = static_cast<uint16_t>(sign | (mant >> 13)); }
        } else { h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13)); }
        T result;
        *reinterpret_cast<uint16_t*>(&result) = h;
        return result;
    } else {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&f);
        uint16_t bf = static_cast<uint16_t>(bits >> 16);
        T result;
        *reinterpret_cast<uint16_t*>(&result) = bf;
        return result;
    }
}
}  // namespace NsCummin
```

次正规分支修正说明：这里的 `exp` 实际记录尾数的左移次数，须从 `0` 起计（移位前即规格化完成时 `exp==0`），每左移一次加 1；FP32 指数字段为 `127 - 14 - exp`（与代码中的 `127 - 15 - exp + 1` 等价）。原先从 `1` 起计，会使所有非零 FP16 次正规数的转换结果缩小一半。边界编码可在 host 侧按位型复算：`0x0001`（最小次正规）= 2^-24、`0x0200` = 2^-15、`0x03ff`（最大次正规）= (1023/1024)×2^-14 ≈ 6.09755516×10^-5（紧邻最小正规数 2^-14）、`0x0400`（最小正规）= 2^-14。这与 [FloatToHalf 次正规阈值卡](../precision/floattohalf_subnormal_threshold_exp102.md) 中读方向使用的 `127−14−e` 基指数一致。

```cpp
// Scalar scan - use float for all comparisons to avoid backend issues
float runningMinF = ToFloat<T>(inputBuf.GetValue(0));
int64_t runningIdx = 0;
valBuf.SetValue(0, inputBuf.GetValue(0));
idxBuf.SetValue(0, static_cast<int64_t>(0));

bool nanSeen = false;

for (uint64_t i = 1; i < R; i++) {
    float curF = ToFloat<T>(inputBuf.GetValue(i));
    ...  // 位级 NaN 判定（掩码 0x7FFFFFFF/0x7F800000，不走 != 自比较）

    if (curF < runningMinF) {
        runningMinF = curF;
        runningIdx = static_cast<int64_t>(i);
    }
    valBuf.SetValue(i, FromFloat<T>(runningMinF));
    idxBuf.SetValue(i, runningIdx);
}
```

### R4 补充：__builtin_memcpy 位视图转换三件套（BenchHarnessScatter，CANN 9.2.0，verbatim）

`Bf16ToFloat`（bf16 是 fp32 的高 16 位，直接移位）：

```cpp
__aicore__ inline float Bf16ToFloat(bfloat16_t v)
{
    uint16_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
```

`FloatToBf16`（低 16 位 round-to-nearest-even，inf/nan 直接截断不加偏置）：

```cpp
__aicore__ inline bfloat16_t FloatToBf16(float f)
{
    // bf16 = high 16 bits of fp32, round-to-nearest-even on the low 16 bits.
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    uint32_t expField = (u >> 23) & 0xFFu;
    if (expField == 0xFFu) {                           // inf / nan — truncate, no rounding
        uint16_t h = static_cast<uint16_t>(u >> 16);
        bfloat16_t r;
        __builtin_memcpy(&r, &h, sizeof(r));
        return r;
    }
    uint32_t lsb      = (u >> 16) & 1u;        // LSB of the bf16 mantissa
    uint32_t dropped  = u & 0xFFFFu;            // 16 bits to discard
    uint32_t halfway  = 0x8000u;                // 1 << 15
    // carry (round up) iff dropped > halfway, or (tie and lsb==1)
    uint32_t bias = lsb ? halfway : (halfway - 1u);
    uint16_t h = static_cast<uint16_t>((u + bias) >> 16);
    bfloat16_t r;
    __builtin_memcpy(&r, &h, sizeof(r));
    return r;
}
```

`HalfToFloat`（IEEE 754 binary16 布局换算，次正规正规化基指数 113）：

```cpp
__aicore__ inline float HalfToFloat(half h)
{
    uint16_t hv;
    __builtin_memcpy(&hv, &h, sizeof(hv));
    uint32_t sign = (static_cast<uint32_t>(hv) & 0x8000u) << 16;
    uint32_t exp  = (static_cast<uint32_t>(hv) & 0x7C00u) >> 10;
    uint32_t mant = (static_cast<uint32_t>(hv) & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;                                  // ±0
        } else {
            // subnormal → normalise: shift mant left until bit 10 set.
            // Base exponent for fp16 subnormal is 2^(-14), so after normalisation
            // the fp32 biased exp = 113 + exp (NOT 112 + exp as for normals).
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            f = sign | ((exp + 113) << 23) | ((mant & 0x03FFu) << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);         // inf / nan
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    __builtin_memcpy(&r, &f, sizeof(f));
    return r;
}
```

`FloatToHalf`（float→fp16）同理按布局换算；其舍入与次正规细节另有两个独立坑（round-to-nearest-even 实现、次正规阈值/移位公式），见 [手写 fp32→fp16/bf16 舍入未实现 RNE 卡](../precision/handwritten_float_cast_rne_roundbit_bias.md)与 [FloatToHalf 次正规阈值卡](../precision/floattohalf_subnormal_threshold_exp102.md)——"编译通过"不等于"转换正确"。

### R5 补充（独立 trace，950 arch35，CANN 9.1.0）：静默无效转换、官方标量转换 API 与弱构建检查掩盖

**形态一（静默无效）**：kernel 标量路径用 `static_cast<float>(xGM_.GetValue(...))` 读 bf16 输入——编译通过，但转换不生效，输出退化为同通道常数（≈bias）。判别：bf16 用例"同一 co 通道所有输出恒为同一常数、MERE≈1.0、fp16/fp32 同路径正常"。修复用官方标量转换（`kernel_operator.h` 包含链直接可用，无需额外 include）：

```cpp
// D_T_X → float：bf16 使用 AscendC::ToFloat，其他类型使用 static_cast
template <typename T>
__aicore__ inline float ScalarToFloat(T val)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToFloat(val);
    } else {
        return static_cast<float>(val);
    }
}

// float → D_T_X：bf16 使用 AscendC::Cast(float)，其他类型使用 static_cast
template <typename DstT>
__aicore__ inline DstT FloatToScalar(float val)
{
    if constexpr (std::is_same_v<DstT, bfloat16_t>) {
        return AscendC::Cast(val);
    } else {
        return static_cast<DstT>(val);
    }
}
```

（`AscendC::Cast(const bfloat16_t&)` 在 3510 上经 `FPTranslation::Bf16ToFp32`，实现即 `GetScalarBitcodeValue` 位双关 + 左移 16——与本卡手写位布局等价，half/float 的 static_cast 是正确路径，保持不变。）

**形态二（函数式 cast 硬错误）**：输出写回 `outLocal.SetValue(0, D_T_X(finalResult))`（`D_T_X = bfloat16_t` 的 C++ 函数式转换）触发 `fatal error: error in backend: not support bf16 type cast`。修复按 `if constexpr` 分流——FP32（D_T_X 与 ComputeT 同型）直接 SetValue；FP16/BF16 先 `Duplicate` 标量进 FP32 work buffer 再向量 `Cast` 到输出 dtype：

```cpp
            if constexpr (std::is_same_v<D_T_X, ComputeT>) {
                // FP32: 直接 SetValue（S→V 同步链略）
                outLocal.SetValue(0, finalResult);
            } else {
                // FP16/BF16: Duplicate(float) -> Cast -> D_T_X
                LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
                Duplicate<ComputeT>(workBuf0, finalResult, 8);
                PipeBarrier<PIPE_V>();
                Cast(outLocal, workBuf0, RoundMode::CAST_ROUND, 8);
                PipeBarrier<PIPE_V>();
                // V→MTE3 事件同步链略
            }
```

（`Duplicate(..., 8)` 复制 8 份再 `Cast(..., 8)`：向量 API 按块工作，8 个 FP32=32B 是安全块大小；多余 7 份留在 UB，`DataCopyPad` 只搬第 1 个。）

**形态三（非连续提取）**：stride 提取场景不要在标量路径上转换 BF16——先把整个物理跨度向量 Cast 到 FP32，再在 FP32 缓冲做标量提取（非连续时 Cast 缓冲按物理跨度 `ubFactor * maxStride` 分配）：

```cpp
                    if constexpr (std::is_same_v<D_T_X, ComputeT>) {
                        for (int64_t j = 0; j < chunkElem; j++) {
                            workBuf1.SetValue(j, inputData.GetValue(j * tensorStride));
                        }
                    } else {
                        Cast(workBuf0, inputData, RoundMode::CAST_NONE, physicalCount);
                        PipeBarrier<PIPE_V>();
                        for (int64_t j = 0; j < chunkElem; j++) {
                            workBuf1.SetValue(j, workBuf0.GetValue(j * tensorStride));
                        }
                    }
```

**弱构建检查掩盖**：构建脚本用 `find ... -name "*.o" | head -1` 只校验"存在至少一个 kernel 二进制"时，BF16 模板编译失败被掩盖——包里残留旧 BF16 二进制，而 host tiling 库已按新 TilingData 布局重新打包，kernel 按旧字段偏移读新结构 → 全部 BF16 用例输出同一垃圾常数（如 1.976562）并伴随 507035 流同步失败。判别：某 dtype 全部用例输出同一常数垃圾值、其余 dtype 正常 → 第一步 `--make_clean` 全量重编译看完整 bisheng 报错，再检查构建脚本的二进制存在性检查是否只查了一个 .o。

## 系统信息

CANN 9.2.0 bisheng，目标 dav-2201；相关文件为 `operators/cummin/cummin.asc`。R3：CANN 9.1.0，`bash build.sh --soc=ascend950 -j8`；相关文件 `op_kernel/arch35/cummin.h`（NsCummin 命名空间 ToFloat/FromFloat、ProcessArScan/ProcessAraScan 标量循环）。R4（BenchHarnessScatter，CANN 9.2.0 arch35）：`bash operators/build.sh <pkg>` 构建；相关文件 `op_kernel/arch35/bench_harness_scatter_kernel.h`（ReduceScalar + 转换函数）；CANN 头 `kernel_type.h`/`vec.h`（类型定义核查对象）。

## 设计决策

用 `if constexpr` 只隔离 bf16 实例：float 直接读取，half/int32 延续可编译的标量 cast。没有用测试侧降 dtype 或跳过 bf16。R3 扩展：当比较本身也不可选（setolt）时，"隔离 cast"不够，须让全部标量比较发生在 float 域；half 的 subnormal 分支（exp==0 规格化循环）保证小值语义，只处理规格化数的简化版会对 subnormal 输入产出全 0。

## 性能影响

本轨迹未对这一局部修复做独立性能归因；它只声明解除构建或正确性阻塞。

## 预防

在目标 CANN/SoC 上为相关 API、dtype 或搬运形态保留最小编译与上板用例，不以静态推断替代。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：构建 exit=2，编译器明确报 not support bf16 type cast。（来源平台：a3；证据等级: tool_observed）。
- 修复后：构建 exit=0，libop_scatter.so 链接成功。（来源平台：a3；证据等级: tool_observed）。
- 修复后：NPU 测试 46 passed。（来源平台：a3；证据等级: tool_observed）。

- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：VF 内 `AscendC::Reg::Cast`、VF 外 LocalTensor `AscendC::Cast`、标量 `GetValue/SetValue` 隐式转换三种 cast 写法编译均 `EXIT=1`，错误逐字一致为 `fatal error: error in backend: not support bf16 type cast`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：改为 union 位运算版（含反向 `fp32_to_bf16_rint`）后，同一构建入口 `bash build.sh` 输出 `SUCCESS`，`inspect_build` 显示 Pipeline status: COMPLETE、1/1 .o 生成（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：位运算版 kernel 上板执行 ST：最小调用通路返回 OK；数值用例的 mismatch 属另一独立根因（标量/向量流水线同步），与本转换路径无关（来源平台：950；证据等级：tool_observed）。

- 独立复现 R3（Cummin 算子开发会话，CANN 9.1.0）：bf16 模板实例标量 `<` 比较报 `fatal error: error in backend: Cannot select: ... setcc ... setolt:ch`，bisheng exit 70（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（Cummin 算子开发会话，CANN 9.1.0）：改 `static_cast<float>` 进 float 域比较后，同一构建报 `not support bf16 type cast`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（Cummin 算子开发会话，CANN 9.1.0）：ToFloat/FromFloat 位级模板落地后 `bash build.sh --soc=ascend950 -j8` 无 error、custom_opp 包成功生成；同一标量路径在 NPU 上 bf16 用例（shape [983,5,1,5] dim=-4 的 ARA-scan 与 bf16 1D AR-scan）值/索引全对，全量 L0 ST 21/21 通过（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（BenchHarnessScatter，CANN 9.2.0 arch35 常规 vector kernel）：修复前 `static_cast<float>(bfloat16_t)` 构建报 Kernel Compilation Error（not support bf16 type cast）；按 POD struct 假设改 `.value` 访问报 `member reference base type 'bfloat16_t' (aka '__bf16') is not a structure or union`（half 同）（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4：改 `__builtin_memcpy` 位级转换三件套后构建 exit=0、`bash operators/build.sh` 输出 SUCCESS；上板 3 个复现用例全部正常执行（int32 拷贝 0/100000 失配、重复索引 add 0/32768 失配）（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5（独立 trace，朴素直卷算子，CANN 9.1.0）：第一次"把 bf16 并入 fp16/fp32 浮点累加路径"修复后复测 bf16 子集仍 5/5 FAIL、MERE=1.09e+00~1.29e+00、actual 全同为同通道常数（如全部 -0.088867 / 全部 1.773438），golden 各异；头文件事实 `kernel_scalar_convert.h:158` `ToFloat(const bfloat16_t&)`、`:134` dtype 编译期断言、`Bf16ToFp32 = GetScalarBitcodeValue<bfloat16_t,uint32_t>(bVal) << 16` 再位转 float；改 `ScalarToFloat/FloatToScalar`（AscendC::ToFloat/Cast）后 L1 全量 104/104 PASS（bf16 33/33，MERE=5.08e-04~1.47e-03 < 7.81e-03），L0 50/50 无回归（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5（独立 trace，foreach 范数算子输出写回，CANN 9.1.0）：修复前 BF16 模板编译报 `fatal error: error in backend: not support bf16 type cast`、`.o` 不生成、ld.lld 连锁 cannot open；改 `Duplicate→Cast(CAST_ROUND, 8)` 后 `Build completed successfully!`、kernel binary 计数 = 3；BF16 用例 Real 模式 NPU 执行 PASS（MERE=0.00e+00，threshold=7.81e-03），全量 60/60（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5（独立 trace，foreach 范数算子非连续 stride，CANN 9.1.0）：弱构建检查掩盖形态——增量构建报"成功"后 ST Real 全部 14 个 BF16 用例 FAIL、actual 恒为 1.976562（golden=32.000000，rel_err=9.38e-01）、1 例流同步失败 507035；`--make_clean` 全量重编译暴露 `not support bf16 type cast` exit 70；改"向量 Cast 整段升 FP32 再标量提取"后 UT 37/37、ST Real BF16 用例全部恢复 PASS（MERE=0.00e+00）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- host 侧归约路线未作为 active 推荐；本卡只保留设备侧 BF16 位扩展到 FP32 的已验证机制。

该做法最初只覆盖 bf16→fp32 的精确位扩展；反向 fp32→bf16 的 RINT 位运算已在 950/CANN 9.2.0 实测闭合（见下条）。大小端、不同浮点编码不能照搬。950 已实测（CANN 9.2.0 SIMT 路径，见验证信号）；a2 未验证，推断可迁移前需确认编译器和位布局。

- 反向转换（950/CANN 9.2.0 实测）：fp32→bf16 的 RINT 位运算 `roundingBias = 0x7FFF + ((u >> 16) & 1)` 进位后取高 16 位，与 `RoundMode::CAST_RINT` 语义对齐、服务于与 golden 逐位一致；截断等其他 round mode 须另写 bias 逻辑，NaN/Inf payload 截断与硬件 cast 的逐位一致性未验证。
- R3 half 往返边界：`FromFloat<half>` 的舍入为截断（mant >> 13），与硬件 round-to-nearest 可能不同——fp16 小值域精度对比时 golden 需模拟存储精度，subnormal 区硬件与软件转换的残余差异未在指令级闭合（同批《fp16 小值域 MARE 超标：golden 未模拟存储精度》卡）。
- R3 int32 边界：`ToFloat<int32_t>/FromFloat<int32_t>` 走 static_cast 往返，|v|>2^24 的整数丢精度——整数路径必须原生比较，不可统一走 float 域（同批《int32 经 fp32 中转丢精度，超 2^24 须原生比较》卡）。
- R3 float 自比较边界：该后端版本上连纯 float 标量的 `x != x` NaN 判定也会触发 seto 不可选（同批《标量 NaN 自比较后端 Cannot select seto》卡）；NaN 判定须用位级掩码，不要换回比较惯用法。
- R4 类型系统边界："内建类型无成员/无 cast"事实绑定 CANN 9.2.0 的 kernel 编译器（bisheng/tikcpp）类型系统；其他版本 `bfloat16_t`/`half` 定义可能变化，升级后需重验。R4 转换函数的数值正确性（舍入/次正规）不属本卡——首版位级转换本身带着舍入与次正规两个数值 bug，见相邻卡。
- 相邻卡片：[bf16 扩精度 Cast 误用 CAST_RINT 卡](../precision/bf16_cast_cast_rint.md)——Cast API 可用时的 RoundMode 方向语义（扩精度 CAST_NONE / 缩精度 CAST_RINT）；本卡是 cast 指令整体不可用时的位运算替代，区分条件：后端报 `not support bf16 type cast` 用本卡，Cast 可用但产出垃圾值用那卡。[fp16/bf16 位流转 float 零依赖转换卡](../validations/cxx_fp16_bf16_bitstream_to_float_conversion.md)——宿主/C++ 侧位级转换与 python bf16 位桥的验证路线，与本卡 device 侧位运算互补。[bf16 标量位序键比较卡](bf16_scalar_compare_by_raw_bits.md)——只需要"选择已有元素"而不需要数值算术时，可按原始 uint16 位序键比较、免往返转换。

## 坏实践

把 bf16 用例从测试集中删除或把输入先转成 half 会掩盖后端能力缺口，且改变算子 dtype 语义。

R3 补充：在标量路径直接写 `static_cast<float>(bfloat16_t)` 或 bf16 之间的 `<` 比较——两条都会在后端 fatal error，且 `not support bf16 type cast` 报错不指向具体源行，需靠二分定位；half 位级转换只处理规格化数不处理 subnormal（exp==0 分支）——小值输入会得到全 0。

R4 补充：假设 AscendC 的 half/bf16 是带 `.value` 的 POD struct（头文件 vec.h 的定义会误导）——kernel 编译时生效的是内建 `__bf16`/半精度类型；把 vector 路径的 cast 写法惯性带到标量路径会直接编译失败。

R5 补充：kernel 内对 `bfloat16_t` 使用 `static_cast<float>` / `static_cast<bfloat16_t>` / 函数式 `D_T_X(x)` 转换——"编译通过"不等于"转换正确"，静默无效形态的输出错误值极难归因（本例第一次修复因未核对转换语义而无效）；增量构建 + "至少一个 .o 存在"式弱检查之后直接跑精度回归，把部分模板编译失败当成精度问题去排查 kernel 逻辑；修改 TilingData 布局后不清理构建目录直接重装——host 库与 kernel 二进制布局不一致的中间态必然产生错位读。

## 误诊

无。R3 首次定位时曾把 setolt 归因于"比较操作数类型写错"，改到 float 域后暴露 cast 缺口，两段错误共同指向"标量 bf16 算术/比较/转换整体不可用"这一结论。

### 增量（诊断代码中的 bf16 标量转换，同函数向量化 Cast 反证）

调试 workspace 问题时向 AIV 侧 `CastAndOutput` 插入诊断代码（把 tiling 标量经 UB 写到输出 y、再向 workspace 写 9.0 读回验证），`bash build.sh --soc=ascend950` 编译失败：`fatal error: error in backend: not support bf16 type cast`（exit 70）+ `ld.lld: error: cannot open .../kernel_meta_..._mix_aiv_27.o`——失败的是 bf16 模板实例（TilingKey=27），fp16 实例正常。**关键反证**：同函数生产代码的向量化 `Cast(ubCastDst, ubCastSrc, RoundMode::CAST_RINT, curN)`（float→bf16，count 参数形式）一直编译通过——锁定问题在标量路径（诊断块含 `static_cast<float>(ubCastDst.GetValue(i))` 与 `ubCastDst.SetValue(i, static_cast<D_T_X>(9.0))` 等标量转换）。修复=移除诊断代码、恢复只含向量化 Cast 的原实现（编译通过、包生成 SUCCESS、后续 UT 10/10 含 `tiling_bf16_g1 | TilingKey=27` 用例）。边界：未逐条隔离具体不支持的操作（两条候选均在失败块内），使用时以"bf16 标量转换整体规避"为边界。预防：kernel 内诊断标量观测一律用 float 张量承载、bf16 只经向量化 Cast 进出；模板算子编译失败先看失败的 TilingKey 实例（`_mix_aiv_<tilingKey>.o` 后缀按实例定位 dtype）。

### 增量（调试 printf 的 %f 实参把 bf16 标量拉进 cast：三实例与 union 位重解释）

`AscendC::printf/PRINTF` 的 `%f` 需要 float 实参，窄类型模板参数的 `GetValue()` 返回 bf16 值——`static_cast<float>(...)` / `(float)...` / 隐式转换把 bf16 标量 cast 拉进 kernel，三个独立算子会话同型失败（`not support bf16 type cast` exit 70 + `ld.lld cannot open ..._27.o` 连锁），**half 同写法不报错**。附带发现：kernel 侧无 `std::memcpy`（`error: no member named 'memcpy' in namespace 'std'; did you mean '__memcpy'?`）。三种修复式：

1. 删除调试打印（Transpose 类算子：`(float)xGm.GetValue(0..2)` 打印移除，保留纯 tiling 字段/地址打印）。
2. `if constexpr (std::is_same_v<T, float>)` 分派 + union 位重解释打印原始 16 位 bits（softmax 类算子，verbatim）：

```cpp
if (li == 0 && AscendC::GetBlockIdx() == 0) {
    if constexpr (std::is_same_v<T, float>) {
        auto pb = preInBuf_.Get<float>();
        float y0 = pb.GetValue(0);
        AscendC::PRINTF("fast y0=%f ...\n", y0, ...);
    } else {
        auto pb = preInBuf_.Get<T>();
        T y0 = pb.GetValue(0);
        union { T h; uint16_t u; } cv; cv.h = y0;   // 位重解释，绕开标量 cast 与 memcpy
        AscendC::PRINTF("fast y0_bits=%x ...\n", (int)cv.u, ...);
    }
}
```

host 侧按 IEEE 754 binary16 解码（如 `0x32a8` ≈ 0.208）。要点：float 专用路径必须 `if constexpr` 隔离（普通 `if` 两分支都实例化）；取值必须 `Get<T>()` native dtype——用 `Get<float>()` 读 16 位缓冲是位型错位（两个 16 位值打包成一个 32 位）。
3. `if constexpr` 把调试段限定到 half 实例后内部显式 `static_cast<half>`（conv 类算子：`static_cast<D_T_X>(float)` 在 bf16 实例化失败）。

验证信号（复现增量）：`Get<T>` 隐式转 float 与 `(float)y0` 显式强转两版均编译失败、`std::memcpy` 版失败（`no member named 'memcpy'`）、if constexpr + union 版 `./build.sh` SUCCESS 且上板输出 `y0_bits=32a8` 与 fp32 路径数值同量级（来源平台：950；证据等级：tool_observed）；删除调试打印后构建全部通过、custom_opp .run 包正常生成（来源平台：950；证据等级：tool_observed）。

### 增量（count=1 向量 Cast、bf16 标量运算全景与编译期裁剪路线）

1. **纯 Vector kernel 的 count=1 向量 Cast 也被拒**：`Cast(outUb, accFloat, RoundMode::CAST_RINT, 1)`（`outUb` 为 `LocalTensor<__bf16>`）在后端 lowering 阶段报同一 `not support bf16 type cast`——受限不止标量形态。恢复构建的做法=**编译期裁剪 bf16 注册**（`op_kernel/arch35/*_tiling_key.h` 的 `ASCENDC_TPL_DATATYPE_DECL` 移除 `C_DT_BF16` + `def.cpp` 移除 `ge::DT_BF16`，两个文件同步改），仅适用"仅需 fp16 交付、bf16 可缓发"的场景；业务硬性要求 bf16 时不适用。
2. **float→bf16 方向同样受限**：`static_cast<D_T_X>(float)`（调试写回）与 `static_cast<float>(__bf16)` 双向都被拒绝，与 R2-R4 合并成"bf16 标量 cast 双向受限"。
3. **bf16 标量运算全景（CANN 9.1.0 独立复现，与 R4 的 9.2.0 实例呼应）**：标量聚合函数直接在 `D_T_DATA` 上用 C++ 运算符，bf16 实例相继报三种后端错误——`Cannot select: i64 = setcc`（`<`/`>` 比较）、`not support bf16 type cast`（`static_cast<float>`）、`Cannot select: bf16 = fmul`（乘法，推测 fadd 同类）；`AscendC::DataCopy` 是 Tensor 搬运 API，不能当 2 字节标量 memcpy 用。解法=`if constexpr` 对 `bfloat16_t` 特化 + `__builtin_memcpy` 位级转换（即 R4 三件套）+ 全部运算在 float 域完成。
4. **bfloat16_t 无标量 operator float() 的接口面事实（CANN 9.0.0-beta.2，数学函数类算子 BF16 分支）**：`half` 有隐式 `operator float()` 可直接 `(float)hx`，`bfloat16_t` 反复 grep 安装目录（kernel_operator.h、basic_api/、内置算子源码）只看到模板参数用法与 `Cast<float, bfloat16_t, ...>` 向量模式。union 位级转换实现标量往返（bf16→fp32 左移 16 位**精确**；fp32→bf16 右移 16 位**截断**——注意截断非 RNE，参考实现用 RNE 舍入时直接套用会差 1 ulp）+ 计算链全程 FP32 上溯（VF 流水满足 `__simd_vf__`+`asc_vf_call` 后标量循环覆写正确结果），`TILING_KEY_IS(n)` 多分支入口按 dtype 泛化复用。

### 增量（模板 Cast<T>(float) 解析到向量重载：编译过、运行期 507035）

SwiGlu 类算子混合路径的标量提取/写回（`T=bfloat16_t`）：

```cpp
x0Fp32.SetValue(k, static_cast<float>(rawLocal.GetValue(x0Offset + k)));  // T=bfloat16_t
...
outLocal.SetValue(outOffset + k, Cast<T>(x0Fp32.GetValue(k)));            // 模板形式 Cast<T>(float)
```

两种写法**编译均通过**，但 bf16 用例运行期 `流同步失败: 507035`（同路径 fp16/fp32 用例 PASS）。机制：`bfloat16_t` 的 C++ `static_cast` 在 NPU 标量 ALU 无对应指令；模板形式 `Cast<T>(float)` 解析到的是**向量 Cast 模板**而非 `kernel_scalar_convert.h` 中的**非模板标量重载**——编译期不报错、运行期崩。与 R2-R4 的"编译期 exit 70"形态、先前增量 R5 的"静默不转换"形态并列：同一后端缺口的三种暴露形态，结论一致——bf16 标量一律走官方标量重载或位级转换。

正确标量接口（`ascendc/include/basic_api/impl/kernel_scalar_convert.h`，trace 内 grep 行号）：

```
30: __aicore__ inline bfloat16_t Cast(const float& fVal)
81: // ToBfloat16 has been updated, please use Cast instead.
82: __aicore__ inline bfloat16_t ToBfloat16(const float& fVal)   // 弃用别名
94: __aicore__ constexpr inline U Cast(T bVal)                   // 模板版，另有 static_assert 限制
```

修复（按 dtype `if constexpr` 三分支，verbatim）：

```cpp
            for (uint32_t k = 0; k < VEC_MIN_ELEMS; k++) {
                if (k < chunkSize) {
                    // 使用 AscendC::Cast 进行标量类型转换 (支持 half/bfloat16_t → float)
                    if constexpr (std::is_same_v<T, float>) {
                        x0Fp32.SetValue(k, rawLocal.GetValue(x0Offset + k));
                        x1Fp32.SetValue(k, rawLocal.GetValue(x1Offset + k));
                    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                        x0Fp32.SetValue(k, Cast(rawLocal.GetValue(x0Offset + k)));
                        x1Fp32.SetValue(k, Cast(rawLocal.GetValue(x1Offset + k)));
                    } else {
                        // half: implicit conversion to float
                        x0Fp32.SetValue(k, static_cast<float>(rawLocal.GetValue(x0Offset + k)));
                        x1Fp32.SetValue(k, static_cast<float>(rawLocal.GetValue(x1Offset + k)));
                    }
                } else {
                    x0Fp32.SetValue(k, 0.0f);
                    x1Fp32.SetValue(k, 0.0f);
                }
            }
```

```cpp
            // 标量写回结果
            for (uint64_t k = 0; k < chunkSize; k++) {
                float val = x0Fp32.GetValue(k);
                if constexpr (std::is_same_v<T, float>) {
                    outLocal.SetValue(outOffset + k, val);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    outLocal.SetValue(outOffset + k, Cast(val));
                } else {
                    outLocal.SetValue(outOffset + k, static_cast<T>(val));
                }
            }
```

设计决策：为什么三分支而不是统一 `Cast<T>`——标量 `Cast` 的 float↔bf16 重载是**非模板**的，模板 `Cast<T>(float)` 解析不到它，统一模板写法正是失败版本；`half` 保留 `static_cast`（同混合路径 fp16 用例实测 PASS，不做无必要改写）。误诊：首轮怀疑"BF16 分配超 UB"（与同算子 FP32 根因类比），逐 buffer 演算 BF16 占用恰为 196608B 未超额，排除后才转向转换语义。

验证信号（复现增量，SwiGlu 算子联调）：

- 修复前：bf16 用例 `流同步失败: 507035`（12 pass 1 fail 时点），同路径 fp16/fp32 均 PASS（来源平台：950；证据等级：tool_observed）。
- 头文件证据：`kernel_scalar_convert.h` line 30 `Cast(const float&)→bfloat16_t`、line 82 弃用别名 `ToBfloat16`、line 94 模板版另有限制（来源平台：950；证据等级：tool_observed——grep 头文件原文）。
- 修复后：该 bf16 用例 PASS（MERE=1.88e-04、MARE=5.45e-04，threshold 7.81e-03），ST Real 13/13、UT 24/24（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 结论限定"kernel 标量路径"（GetValue/SetValue 旁路）；向量 `Cast` 指令的用法不受本增量约束。
- 同一 `static_cast<float>(bfloat16_t)` 写法在不同会话分别呈现"编译期 exit 70"（R2-R4）与"编译通过、运行期 507035/静默错值"（本增量、先前增量 R5）——跨会话暴露形态不稳定，但结论一致：一律避免，改用非模板 `Cast` 标量重载（或位级转换）。
- 仅 Ascend950PR + CANN 9.1.0 实测；后续 CANN 版本是否为 static_cast 提供标量 ALU 支持未验证。

### 增量（标量 fadd 实测、op:: 命名空间陷阱与正向枚举判型）

直接卷积 kernel（BenchNoHarnessConv2d，标量循环 `sum = sum + xv * fv` 逐元素累加，Ascend950PR/arch35 + CANN 9.1.0）把 复现增量的"推测 fadd 同类"落定为实测，并补两个新事实：

**1) 标量算术 fadd 实测同样不可选**：bf16 编译目标（kernel_2）首轮报 `fatal error: error in backend: Cannot select: ... bf16 = fadd ...`（bisheng clang frontend exit 70）；fp16/fp32 目标同代码编译通过（half/float 标量算术与标量 cast 可用）。与 R3 的 setcc（比较）、先前增量 的 fmul（乘法）合并成完整图景：**bf16 标量算术/比较/转换全面不可选**。

**2) host 侧 `op::` 命名空间在 kernel 内不存在**：用 `std::is_same_v<D_T_X, op::bfloat16>` 判型报 `error: use of undeclared identifier 'op'`——`op::bfloat16` 定义在 host 侧头文件 `x86_64-linux/include/aclnn/opdev/bfloat16.h`，kernel 代码里没有 `op` 命名空间（与"不要在 kernel 里引用 host 侧类型/命名空间"同族；R4 的 `AscendC::bfloat16_t` 限定名错误是另一变体）。

**3) 正向枚举判型分流（避免直接命名 bf16 类型）**：判型不写 `is_same_v<D_T_X, bfloat16_t>`（直接命名类型在该编译环境不可靠），而写正向枚举 half/float、else 即 bf16：

```cpp
        if constexpr (std::is_same_v<D_T_X, half> || std::is_same_v<D_T_X, float>) {
            // fp16/fp32: 用 float 累加确保精度
            float biasVal = static_cast<float>(biasGM_.GetValue(static_cast<uint64_t>(co)));
            ...
                    float xv = static_cast<float>(xGM_.GetValue(...));
                    float fv = static_cast<float>(filterGM_.GetValue(...));
                    sum += xv * fv;
            ...
                    yGM_.SetValue(..., static_cast<D_T_X>(sum));
        } else {
            // bf16: 标量 float 转换在 kernel 后端不支持，迭代二用 Cast 向量算子实现
            D_T_X biasVal = biasGM_.GetValue(static_cast<uint64_t>(co));
            for (...) { yGM_.SetValue(..., biasVal); }   // 占位：保三目标可编译可运行
        }
```

（需 `#include <type_traits>`。）反向判型仅在 dtype 集合已知为 {fp16, fp32, bf16} 的模板内安全，新增 dtype 时枚举须同步。bf16 占位输出 bias 是阶段性编译解阻（ST 当期仅测 fp16）；正确 bf16 数值路径须 UB 上向量 Cast（bf16→float 批量转换 → float 计算 → Cast 回写），属后续迭代。

验证信号（复现增量，三轮构建演进 + 终态，均 tool_observed/950）：

- 第一轮：bf16 目标 `fatal error: error in backend: Cannot select: ... bf16 = fadd ... bisheng: error: clang frontend command failed with exit code 70`（fp16/fp32 目标无此错）。
- 第二轮：改 `static_cast<float>` 累加后同目标报 `fatal error: error in backend: not support bf16 type cast`；第三轮 `op::bfloat16` 判型报 `error: use of undeclared identifier 'op'`。
- 最终：if constexpr 正向枚举分流后三目标（fp16/fp32/bf16）全部编译通过、`custom_opp_ubuntu_x86_64.run` 打包安装成功。

失效边界（复现增量）：

- bf16 分支为占位（输出 bias，无卷积计算）——本增量只闭合"bf16 目标编译失败"；bf16 正确数值路径（向量 Cast）未实现未验证。
- "half/float 可标量运算、bf16 不可"为 CANN 9.1.0 bisheng 后端实测；后续编译器版本可能放开（未验证）。
