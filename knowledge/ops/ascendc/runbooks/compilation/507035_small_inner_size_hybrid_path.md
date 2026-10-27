---
type: "Runbook"
title: "507035 小 innerSize：整块拷贝+对齐向量混合路径"
description: "arch35 上 innerSize*sizeof(T)<32B 时分块 DataCopyPad 与非对齐向量访问触发向量核异常 507035；改用整块 raw 拷贝 + 标量提取到 32B 对齐补零 FP32 buffer + 向量计算的混合路径。"
tags: [task.debugging, topic.data_movement.data_copy, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 507035 小 innerSize：整块拷贝+对齐向量混合路径

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上为自定义算子 BenchNoHarnessSwiGlu（计算 `output = silu(x0) * x1`，其中 x0/x1 = `input.chunk(2, dim)`）做迭代一联调验证，安装算子包后运行 `bash tests/st/run.sh --real`（Real NPU 模式，CPU golden 比对），13 个用例中 5 个全部报：

```
[Real] builtin_fp16_2d_dim-1_[4,8]
  shape=[4,8], dtype=float16, dim=-1
  [FAIL] 流同步失败: 507035
```

失败的 5 例全部是 shape `[4,8]`、`dim=-1`（fp16 基础/小值域/大值域、fp32、bf16 各一至三例）；而 `[4,8] dim=0`、3D/4D 各 dim、`[1024,2048]` 等用例都能正常执行（部分只有精度问题）。`aclrtSynchronizeStream` 返回 507035，经查 `rt_error_codes.h:125` 为 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`——kernel 在 NPU 向量核上崩溃。

当时 kernel 的搬运方式是：tiling 算出 `innerSize`（dim 半拆后每个 outer block 的输出元素数；`[4,8] dim=-1` 时 `innerSize = (8/2)*1 = 4`，单核、UB tile 16 个输出元素），`DoCopyInContiguous` 把每个 outer block 的 x0/x1 两半**分别**用 `DataCopyPad` 拷入 UB，`copyParams.blockLen = chunkSize * sizeof(T)`（fp16 下即 4*2=8 字节），随后对 UB 里的 x0/x1 做向量 `Cast/Muls/Exp/Adds/Div/Mul`。`innerSize>=16`（fp16）的用例全部正常，只有 `innerSize*sizeof(T) < 32B` 的用例崩。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 开发 + C++ 原生 ST（aclnn Real NPU 模式）联调。
- API 路径：`DataCopyPad`（GM→UB / UB→GM，`DataCopyParams`/`DataCopyPadParams`）、向量指令 `Cast/Muls/Exp/Adds/Div/Mul`、`LocalTensor::GetValue/SetValue`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——所有按 dim 对半拆分（chunk/stack 类布局）且 inner 维可能小于 32B/sizeof(T) 的向量算子。
- 路线优先级：recommended_fastpath（正确性必需，无替代写法在本轨迹中成立）。
- 明确不适用 / 未验证场景：`innerSize*sizeof(T) >= 32B` 时无需本路径（原分块拷贝即可）；tile 内 raw 跨度 `ceil(tileSize/innerSize)*2*innerSize*sizeof(T) < 32B` 的极小总输入（如 shape [2,4] dim=-1，raw 仅 8 字节）本轨迹未出现也未验证；非 arch35 平台的 32B 约束数值未验证。

## 根因

arch35（DAV_3510）上，小 `innerSize` 同时踩两条 32 字节粒度约束：

1. **GM 读 burst 约束**：`CopyGmToUbufAlignV2`（`asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h`）中 `uint32_t unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN; burstLength = blockLen * unitOfBytes;`——`DataCopyPad` 的 `blockLen` 单位是字节，但当有效 GM 读 burst（`blockLen`）小于 32B 时 DMA 出错。`rightPadding` 只能加长 **UB 侧写入**（`padLength`），加长不了 GM 侧读 burst，所以"补 padding 到 32B"救不了 8 字节的读。
2. **向量指令 UB 地址对齐约束**：向量指令（`Cast` 等 `__simd_vf__` 实现）从 UB 取数的源/目的地址需 32B 对齐。整块 raw 拷入后，x1 位于 `rawLocal[innerSize]`（fp16、innerSize=4 时即 8 字节偏移），非 32B 对齐，向量读直接触发向量核异常。

`[4,8] dim=-1` 时 `innerSize=4`：x0/x1 分块拷贝的 burst 只有 8B（违反约束 1）；改为整块拷贝后向量读 x1 偏移 8B（违反约束 2）。两条约束都源自"每个处理单元的有效数据 < 32B"。

## 解决配方

当 `innerSize * sizeof(T) < 32B` 时切换为"整块原始数据拷贝 + 标量提取 + 对齐向量计算"混合路径（`useRawCopy_`）：

1. **整块 raw 拷贝**：tile 覆盖的输入跨度（x0/x1 交错、GM 中连续）一次性 `DataCopyPad` 入 UB——`rawSize = (lastBlock - firstBlock + 1) * 2 * innerSize` 个元素，`[4,8] dim=-1` 为 32 元素 64 字节 ≥ 32B，满足约束 1；UB 目的地址从 buffer 头开始，天然 32B 对齐。
2. **标量提取**：用 `GetValue/SetValue`（S-pipe 标量访问，无 32B 对齐要求）把每个 chunk 的 x0/x1 搬进 **32B 对齐的 FP32 buffer**，并补零到 `VEC_MIN_ELEMS = 32/sizeof(float) = 8` 个元素（32 字节）。
3. **向量计算**：对 8 元素对齐 buffer 跑 `Muls/Exp/Adds/Div/Mul`（补零元素不污染前 chunkSize 个结果）。
4. **标量写回**：只取前 `chunkSize` 个结果 `SetValue` 写输出。

修复前（分块 x0/x1 拷贝，违反两条约束，节选自 `DoCopyInContiguous` 修复前版本）：

```cpp
    uint64_t remaining = tileSize;
    uint64_t currentOut = outOffset;
    uint64_t localOffset = 0;

    while (remaining > 0) {
        uint64_t outerBlock = currentOut / innerSize_;
        uint64_t offsetInBlock = currentOut % innerSize_;
        uint64_t elementsToEnd = innerSize_ - offsetInBlock;
        uint64_t chunkSize = (remaining < elementsToEnd) ? remaining : elementsToEnd;

        uint64_t x0GmOffset = outerBlock * 2 * innerSize_ + offsetInBlock;
        uint64_t x1GmOffset = x0GmOffset + innerSize_;

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = chunkSize * sizeof(T);   // fp16 innerSize=4 时仅 8 字节 → 507035
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(x0Local[localOffset], inputGM[x0GmOffset], copyParams, {false, 0, 0, 0});
        DataCopyPad(x1Local[localOffset], inputGM[x1GmOffset], copyParams, {false, 0, 0, 0});

        remaining -= chunkSize;
        currentOut += chunkSize;
        localOffset += chunkSize;
    }
```

修复后（`Init` 中判定 + `DoCopyInRaw` 整块拷贝 + `Compute` 混合处理；verbatim 取自该会话写入事件）：

```cpp
    // 判断是否使用整块原始数据拷贝策略
    // 当 innerSize * sizeof(T) < 32 时，分块拷贝的 burstLength < 32B，会触发硬件异常
    uint64_t minAlignElems = MIN_BURST_BYTES / sizeof(T);
    useRawCopy_ = (innerSize_ < minAlignElems) && (innerSize_ > 0);
```

```cpp
template <typename T, int TILING_PATH>
__aicore__ inline void KernelBenchNoHarnessSwiGlu<T, TILING_PATH>::DoCopyInRaw(
    LocalTensor<T>& rawLocal,
    uint64_t outOffset, uint64_t tileSize)
{
    // 整块原始数据拷贝: 适用于 innerSize < 32/sizeof(T)
    // 将 tile 对应的原始输入（x0+x1 交错，GM 中连续）一次性拷贝到 UB
    uint64_t firstBlock = outOffset / innerSize_;
    uint64_t lastBlock = (outOffset + tileSize - 1) / innerSize_;
    uint64_t rawStart = firstBlock * 2 * innerSize_;
    uint64_t rawEnd = (lastBlock + 1) * 2 * innerSize_;
    uint64_t rawSize = rawEnd - rawStart;

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = rawSize * sizeof(T);   // 整块 ≥ 32B，满足 GM 读 burst 约束
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(rawLocal, inputGM[rawStart], copyParams, {false, 0, 0, 0});
}
```

```cpp
    if (useRawCopy_) {
        // ===== 整块原始数据路径 (混合标量+向量处理) =====
        // raw buffer 中数据布局: [x0_b0, x1_b0, x0_b1, x1_b1, ...]
        // 策略: 用 GetValue/SetValue (标量) 从 raw buffer 提取 x0/x1 到
        //       32B 对齐的 FP32 buffer, 再用向量指令计算 silu(x0)*x1。
        //       向量指令处理 >= 32B (8 FP32 元素) 的对齐数据。
        LocalTensor<T> rawLocal = inQueueX0.template DeQue<T>();

        uint64_t remaining = tileSize;
        uint64_t currentOut = tileStart;
        uint64_t outOffset = 0;
        uint64_t firstBlock = tileStart / innerSize_;

        constexpr uint32_t VEC_MIN_ELEMS = MIN_BURST_BYTES / sizeof(float);  // 8 FP32 元素

        while (remaining > 0) {
            uint64_t blockIdx = currentOut / innerSize_;
            uint64_t offsetInBlock = currentOut % innerSize_;
            uint64_t elementsToEnd = innerSize_ - offsetInBlock;
            uint64_t chunkSize = (remaining < elementsToEnd) ? remaining : elementsToEnd;

            uint64_t rawBlockOffset = (blockIdx - firstBlock) * 2 * innerSize_;
            uint64_t x0Offset = rawBlockOffset + offsetInBlock;
            uint64_t x1Offset = rawBlockOffset + innerSize_ + offsetInBlock;

            // 标量提取 x0/x1 到 32B 对齐 FP32 buffer, 补零到 VEC_MIN_ELEMS
            LocalTensor<float> x0Fp32 = x0Fp32Buf.Get<float>();
            LocalTensor<float> x1Fp32 = x1Fp32Buf.Get<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();

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

            // 向量计算 silu(x0) * x1 (处理 VEC_MIN_ELEMS 个元素)
            Muls<float>(tmp, x0Fp32, -1.0f, VEC_MIN_ELEMS);
            Exp<float>(tmp, tmp, VEC_MIN_ELEMS);
            Adds<float>(tmp, tmp, 1.0f, VEC_MIN_ELEMS);
            Div<float>(x0Fp32, x0Fp32, tmp, VEC_MIN_ELEMS);
            Mul<float>(x0Fp32, x0Fp32, x1Fp32, VEC_MIN_ELEMS);

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

            remaining -= chunkSize;
            currentOut += chunkSize;
            outOffset += chunkSize;
        }

        inQueueX0.FreeTensor(rawLocal);
    } else {
```

（fp32 残余失败另有 tiling UB 除数根因、bf16 残余失败另有标量转换根因，均见独立卡。）

## 设计决策

- 为什么不继续用 padding 方案：`rightPadding` 补的是 UB 写侧（`padLength = leftPaddingCount*sizeof(T) + rightPaddingCount*sizeof(T)`），GM 读 `burstLength` 仍是 8 字节，实测仍 507035。
- 为什么不把 `blockLen` 直接加到 32B：会越界读 GM（`[4,8]` 输入仅 32 元素，最后一个 block 的 x0 偏移 24 + 16 元素 = 40 > 32）。
- 为什么整块 raw 拷贝可行：连续输入路径下 tile 的原始跨度在 GM 中连续，且 `ceil(tileSize/innerSize)*2*innerSize` 个元素通常 ≥ 32B；UB 侧 x0 buffer 本来就按 `2*ubFormer_` 元素分配（x0/x1 合计），空间不增加。
- 为什么标量提取后还要回向量指令：纯标量路径（`GetValue/SetValue` + `__builtin_expf` 逐元素算 silu）实测仍 507035，且标量数学库可用性存疑；向量指令对 32B 对齐、32B 长度的 buffer 是已验证可用的（大 innerSize 路径一直在用）。
- 补零到 8 个 FP32 元素而非按 chunkSize：向量指令按整 32B 处理最稳；零元素参与 `Muls/Exp/Div/Mul` 不影响前 chunkSize 个有效结果（0 的 silu 链是良定的）。
- 当前方案路线档位：recommended_fastpath（小 innerSize 正确性必需；标量提取的吞吐代价未测量，见遗留问题）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0（`$ASCEND_HOME_PATH` 指向 cann-9.1.0 安装根），arch35（DAV_3510）。
- 关键头文件证据：`asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h`（`CopyGmToUbufAlignV2`：`unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN`）；`include/acl/error_codes/rt_error_codes.h:125`（507035 定义）。
- 目录布局（算子包内）：
  - `op_kernel/arch35/bench_no_harness_swi_glu.h`——kernel 主体（useRawCopy 判定、DoCopyInRaw、混合 Compute）。
  - `op_host/arch35/bench_no_harness_swi_glu_tiling.cpp`——tiling（UB 切分，须与 kernel 布局同步，见独立 tiling 除数卡）。
  - `tests/st/test_aclnn_bench_no_harness_swi_glu.cpp` + `tests/st/run.sh --real`——ST 入口。

## 性能影响

本轨迹未测量混合路径相对纯向量路径的吞吐代价（无 profiling 数据）；直觉上每个 chunk 增加 2×chunkSize 次 GetValue + chunkSize 次 SetValue 标量往返，但小 innerSize 用例本身数据量小。采用前请自行测量。

## 预防

- 写"按 dim 对半拆分"类 kernel 前，先算最坏 `innerSize = (dimSize/2) * strideDim`，若 `innerSize * sizeof(T) < 32` 直接设计 raw 拷贝 + 标量提取路径，不要先写分块拷贝再踩 507035。
- 记住 `DataCopyPad` 的 `blockLen` 单位是**字节**（DataCopy 是 32B 块），且 GM 读 burst 有 32B 下限；`rightPadding/leftPadding` 只作用于 UB 写侧。
- 向量指令的 UB 源/目的地址按 32B 对齐设计 buffer 布局；从 interleaved raw buffer 中按偏移直接做向量读是不可靠的。

## 验证信号

以下信号来自该算子迭代一联调验证会话：先定位 507035 为向量核异常，经四轮失败尝试收敛出混合路径，最后全量 ST 回归。

- 修复前：5 例 `[4,8] dim=-1`（fp16×3、fp32、bf16）全部 `流同步失败: 507035`；`innerSize>=16`（fp16）用例全部可执行（来源平台：950；证据等级：tool_observed）。
- 四轮否定实验：rightPadding 补 32B、alignedStride 对齐目标地址、整块 raw + 非对齐向量读、纯标量 `__builtin_expf`——均仍 507035（来源平台：950；证据等级：tool_observed）。
- 混合路径落地后：fp16 三例 `[4,8] dim=-1` 转 PASS（`MERE=2.22e-05/1.21e-08/1.13e-04`），11 pass 2 fail（剩 fp32/bf16，另卡根因）（来源平台：950；证据等级：tool_observed）。
- 三根因全部修复后最终回归：ST Real 13/13 PASS、UT 24/24 无回归（fp32 `MERE=1.84e-09`、bf16 `MERE=1.88e-04`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.1.0 实测；其他 arch 的最小 burst/对齐常量可能不同（未验证，推断机制可迁移、数值须按架构调整）。
- tile 的 raw 跨度本身 < 32B 时（总输入极小，如 shape [2,4] dim=-1 raw=8B）本路径仍违反 GM 读约束，需要并入相邻数据或走标量 GM 读——本轨迹未出现该 shape，未验证。
- `useRawCopy_` 改变 UB 布局后，host tiling 的 UB 切分除数必须同步（否则 fp32 超 UB 再现 507035，见独立 tiling 除数卡）。
- bf16 在该路径的标量转换必须用 `AscendC::Cast` 标量重载（见独立 BF16 标量转换卡）。

**未决项**：

- 混合路径的标量提取吞吐代价未测量；若成为瓶颈可探索按 32B 对齐分段批量提取（未尝试）。
- GM 读 burst ≥32B 与向量 32B 对齐约束的官方规范文档未查到，当前结论由头文件注释 + 实验消除法支撑，可在下个会话补文档锚点。
## 坏实践

- 对小 chunk 用 `DataCopyPad` + `rightPadding` 试图凑 32B——padding 不加长 GM 读 burst，实测无效。
- 整块 raw 拷贝后直接对 `rawLocal[innerSize]`（非 32B 对齐偏移）做向量 `Cast`——向量核异常。
- 只改 kernel 不改 tiling 的 UB 布局账目——fp32 下 UB 超额仍是 507035。

## 误诊

- 初期把 507035 当作 `ACL_ERROR_RT_STREAM_SYNC_FAIL` 类流问题排查（查 plog、npu-smi、重跑单例），ASCEND_GLOBAL_LOG_LEVEL=1 的 plog 未给出额外信号；实际是向量核异常，定位靠 dav_3510 头文件约束 + 逐方案实验消除。
- "纯标量路径失败 ⇒ `__builtin_expf` 在标量 ALU 不可用"是未验证假设（agent 推断）；混合路径绕开了该问题，该假设从未被单独证实。
