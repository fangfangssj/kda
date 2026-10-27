---
type: "Runbook"
title: DAV_3510 Matmul API 无先例走纯 AIV 回退
description: DAV_3510 上 Matmul 高阶 API 别名随编译宏解析（MatmulImpl/MatmulClient/MatmulServiceAux）且需 host 侧 MatmulApiTiling + REGIST_MATMUL_OBJ 配合，同环境自定义算子无先例；4 次 bf16 矩阵乘改用 Vector 核外积累加（Muls+Add 逐 k 行）纯 AIV 实现编译通过。
tags: [task.implementation, topic.compute.composite, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# DAV_3510 Matmul API 无先例走纯 AIV 回退

## 现象

在 Ascend950PR（DAV_3510 / arch35，CANN 安装目录 cann-9.2.0、version.info 报 9.1.0 基线）上开发 MLA 前处理融合算子 MlaProlog（4 次 bf16 MatMul + 2 次 RMSNorm + 2 次 RoPE，参考 DeepSeek-V2 MLA 的 Query/Key 投影与位置编码）。设计方案规定 4 次 CUBE 矩阵乘使用 `Matmul<aType,bType,cType,biasType>` 高阶 API + `IterateAll`/`IterateBatch`。

动手实现前核查 CANN 环境发现三个信号：

1. 头文件 grep 确认 `GetMatmulApiTiling` 只存在于 `x86_64-linux/asc/include/adv_api/matmul/constant_tiling.h`，且是 `__aicore__ constexpr` 设备侧函数；host 侧 tiling 需要走 `matmul_tiling::MatmulApiTiling` 类（`tiling/matrix/matmul_tiling_base.h`）。
2. `matmul_intf.h` 中 `Matmul` 别名按编译宏解析：`SPLIT_CORE_CUBE + ASCENDC_CUBE_ONLY` → `MatmulImpl`，`SPLIT_CORE_CUBE` 单独 → `MatmulServiceAux`，`SPLIT_CORE_VEC` → `MatmulClient`，`ASCENDC_MATMUL_AICORE` → `MatmulClient`——自定义算子里 CUBE+VEC 融合的编译配置组合复杂且本环境从未验证过。
3. 同一块环境上已安装的同类自定义算子 `bench_no_harness_gru_custom` / `bench_no_harness_lstm_custom`（opp/vendors 下）**均未使用 Matmul 高阶 API**，其 arch35 kernel 源码头部注释明示：`纯 AIV 模式 Kernel 入口 …… 完全不使用 Matmul API, 不需要 REGIST_MATMUL_OBJ，矩阵乘法通过 Mul + ReduceSum 在 Vector 核上实现，coreType = AIV (非 MIX)`。

即：设计指定的 Matmul 高阶 API 路线在本环境没有任何可参照的可编译先例，而纯 AIV 路线有同团队同环境的成熟先例。

## 适用场景

- 工程形态：自定义算子工程（op_host/op_kernel/op_api/op_graph + build.sh，`bash build.sh --soc=ascend950`），算子含 GEMM 类计算且迭代目标是先拿到可编译骨架。
- API 路径：AscendC kernel 侧 `TPipe`/`TBuf<TPosition::VECCALC>`/`DataCopyPad`/`Cast`/`Muls`/`Add`/`Duplicate`/`SetFlag/WaitFlag<HardEvent::MTE2_V>`；TilingKey 用 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 模板机制。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN cann-9.2.0 安装目录（version.info 9.1.0 基线），x86_64 构建主机；**验证级别为编译级**（kernel 二进制 + 算子包生成），本会话切片内未上板运行（与 frontmatter `platforms` 一致，真机验证见遗留问题）。
- 适用粒度：operator_family——需要 bf16 矩阵乘、且优先保证可编译/功能正确性的 DAV_3510 自定义算子骨架。
- 路线优先级：safety_fallback——这是 Matmul 高阶 API（recommended_fastpath，性能上 CUBE 硬件加速远优于 VEC 软实现）不可用或未验证时的回退路线。fallback 声明：放弃 CUBE 硬件加速，接受 VEC 外积累加在大 K 场景的显著性能代价，换取骨架阶段的可编译性与正确性优先；真机性能对比与 Matmul API 可用性验证留待后续迭代。
- 明确不适用 / 未验证场景：性能敏感的生产实现（VEC 外积累加无 CUBE 加速，大 K 时每步都要 DataCopyPad+Cast+Muls+Add）；其他架构（A2/A3 上 Matmul 高阶 API 有文档背书，无需此回退）。

## 根因

DAV_3510 上 Matmul 高阶 API 的可用形态依赖编译宏组合（MatmulImpl/MatmulClient/MatmulServiceAux 三选一）并要求 host 侧 `matmul_tiling::MatmulApiTiling` 配合与 kernel 侧 `REGIST_MATMUL_OBJ` 注册，在本环境自定义算子场景没有任何已验证先例；同环境既有算子全部选择纯 AIV 模式规避该风险。

## 解决配方

矩阵乘 `C[M,N] = A[M,K] @ B[K,N]` 改为外积累加：对每个输出行 m，把 `A[m,:]` 升精度到 fp32 常驻 UB，再对 k=0..K-1 逐行加载 `B[k, nTile]`，`tmp = xVec[k] * wFloat`（标量广播 Muls）后 `result += tmp`，最后一次性 `Cast` 回 bf16 写 GM。B 按 [K,N] 行主序存储，每行连续，`DataCopyPad` 按 `tileN * sizeof(bfloat16_t)` 字节长度加载；N 维按 `mmTileN` 分块（host tiling 按 UB 预算推导）。

最终可执行实现（kernel 侧核心、4 次矩阵乘调用形态、host tiling 配套、TilingKey 模板机制，全部 verbatim）见下文「完整代码」。

## 完整代码

kernel 侧核心实现（verbatim，`op_kernel/arch35/mla_prolog.h`，外积累加 + 输入行升精度）：

```cpp
    __aicore__ inline void VecMatmul(
        int64_t K, int64_t N,
        const GlobalTensor<bfloat16_t>& wGm, int64_t wOff,
        GlobalTensor<bfloat16_t>& dstGm, int64_t dstOff)
    {
        LocalTensor<float> xVec = xVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> wRaw = wRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> wFloat = wFloatBuf_.Get<float>();
        LocalTensor<float> result = resultBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();

        DataCopyExtParams wcp;
        wcp.blockCount = 1;
        wcp.srcStride = 0;
        wcp.dstStride = 0;

        int64_t nProcessed = 0;
        while (nProcessed < N) {
            int64_t tileN = N - nProcessed;
            if (tileN > mmTileN_) tileN = mmTileN_;
            int32_t tileN32 = static_cast<int32_t>(tileN);

            // 初始化 result = 0
            Duplicate<float>(result, 0.0f, tileN32);
            PipeBarrier<PIPE_V>();

            wcp.blockLen = static_cast<uint32_t>(tileN * sizeof(bfloat16_t));

            for (int64_t k = 0; k < K; k++) {
                // 加载权重行 k [tileN] (bf16)
                DataCopyPad(wRaw, wGm[wOff + k * N + nProcessed], wcp, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(eventId_);
                WaitFlag<HardEvent::MTE2_V>(eventId_);

                // Cast bf16 → fp32
                Cast<float>(wFloat, wRaw, RoundMode::CAST_NONE, tileN32);
                PipeBarrier<PIPE_V>();

                // tmp = xVec[k] * wFloat (标量广播)
                float xVal = xVec.GetValue(k);
                Muls<float>(tmp, wFloat, xVal, tileN32);
                PipeBarrier<PIPE_V>();

                // result += tmp
                Add<float>(result, result, tmp, tileN32);
                PipeBarrier<PIPE_V>();
            }

            // Cast fp32 → bf16
            LocalTensor<bfloat16_t> resultBf16 = wRaw;  // 复用 wRaw buffer
            Cast<bfloat16_t>(resultBf16, result, RoundMode::CAST_RINT, tileN32);
            PipeBarrier<PIPE_V>();

            // 写回 GM
            DataCopyExtParams ocp;
            ocp.blockCount = 1;
            ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(bfloat16_t));
            ocp.srcStride = 0;
            ocp.dstStride = 0;
            DataCopyPad(dstGm[dstOff + nProcessed], resultBf16, ocp);

            nProcessed += tileN;
        }
    }

    __aicore__ inline void LoadRowToFp32(
        const GlobalTensor<bfloat16_t>& srcGm, int64_t srcOff, int64_t K)
    {
        LocalTensor<bfloat16_t> xRaw = xRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> xVec = xVecBuf_.Get<float>();

        DataCopyExtParams xcp;
        xcp.blockCount = 1;
        xcp.blockLen = static_cast<uint32_t>(K * sizeof(bfloat16_t));
        xcp.srcStride = 0;
        xcp.dstStride = 0;
        DataCopyPad(xRaw, srcGm[srcOff], xcp, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);

        Cast<float>(xVec, xRaw, RoundMode::CAST_NONE, static_cast<int32_t>(K));
        PipeBarrier<PIPE_V>();
    }
```

4 次矩阵乘的调用形态（Step1/3/4/6，含 per-head 批量乘）：

```cpp
    __aicore__ inline void Step1MatmulDq(int64_t bsIdx)
    {
        LoadRowToFp32(gmTokenX_, bsIdx * He_, He_);
        VecMatmul(He_, Hcq_, gmWDq_, 0, gmCqRaw_, bsIdx * Hcq_);
    }

    __aicore__ inline void Step4MatmulUk(int64_t bsIdx)
    {
        int64_t qrRowOff = bsIdx * N_ * (D_ + Dr_);
        for (int64_t n = 0; n < N_; n++) {
            // 加载 q_c[bs, n, :D] 从 qr workspace
            int64_t qcOff = qrRowOff + n * (D_ + Dr_);
            LoadRowToFp32(gmQr_, qcOff, D_);
            // W_UK[n, :, :] 起始偏移 = n * D * Hckv
            int64_t wOff = n * D_ * Hckv_;
            // query[bs, n, :] 偏移 = bsIdx * N * Hckv + n * Hckv
            int64_t dstOff = bsIdx * N_ * Hckv_ + n * Hckv_;
            VecMatmul(D_, Hckv_, gmWUk_, wOff, gmQuery_, dstOff);
        }
    }
```

host tiling 侧配套（`op_host/arch35/mla_prolog_tiling.cpp`，verbatim）：多核按 B*S 切分、`mmTileN` 按 UB 预算推导、workspace 三段布局：

```cpp
    // 2. 多核切分: B*S 沿 M 轴分配
    int64_t blockDim = (bsTotal >= coreNum) ? coreNum : bsTotal;
    int64_t bsPerCore = CeilDiv(bsTotal, blockDim);
    int64_t bsTail = bsTotal - (blockDim - 1) * bsPerCore;
    if (bsTail <= 0) {
        bsTail = bsPerCore;
    }

    // 3. 设置 TilingKey
    uint32_t schMode = (bsTotal >= coreNum) ? static_cast<uint32_t>(MLA_PROLOG_TPL_SCH_MULTI_CORE)
                                             : static_cast<uint32_t>(MLA_PROLOG_TPL_SCH_FEW_CORE);

    // 4. VEC matmul tiling: 每次处理 mmTileN 个输出列
    // UB 预算: xVec(K*4B) + wRaw(K*2B) + wFloat(K*4B) + mulData(K*4B) + reduceTmp(K*4B) + result(mmTileN*4B)
    // 对于最大 K=He=7680: 5*7680*4 + mmTileN*4 = 153600 + mmTileN*4
    // UB=248KB=253952B, 预留 40% 余量 → 可用 ~152KB
    // mmTileN = (152000 - 153600) ... 负数，需要减小
    // 实际上 K=7680 时 5 个 K-sized buffer = 153600B ≈ 150KB，已接近极限
    // 解决方案: 对于大 K，减小 mmTileN；对于小 K，增大 mmTileN
    int64_t mmTileN = 64;  // 默认 64 列
    // 动态调整: 确保 UB 不超限
    int64_t ubBudget = 200 * 1024;  // 200KB 预算
    int64_t fixedBufSize = 5 * He * 4;  // 5 个 K-sized float buffer
    if (fixedBufSize + mmTileN * 4 > ubBudget) {
        mmTileN = FloorDiv(ubBudget - fixedBufSize, int64_t(4));
    }
    if (mmTileN < 8) {
        mmTileN = 8;  // 最小 8 列
    }
    // 对齐到 8（float 基本块）
    mmTileN = FloorAlign(mmTileN, int64_t(8));

    // 5. Workspace 规划
    // 用户 workspace: c_q_raw + qr + dkv_kr
    int64_t wsCqRaw = bsTotal * Hcq * BF16_SIZE;
    int64_t wsQr = bsTotal * N * (D + Dr) * BF16_SIZE;
    int64_t wsDkvKr = bsTotal * (Hckv + Dr) * BF16_SIZE;
    int64_t usrWsSize = wsCqRaw + wsQr + wsDkvKr;

    // 系统 workspace (Matmul API 等，纯 AIV 模式为 0)
    int64_t sysWsSize = 0;

    // 6. 设置 Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWsSize + usrWsSize);
```

TilingKey 模板机制（`op_kernel/arch35/mla_prolog_tiling_key.h`，参照 GRU 先例的 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 写法）：

```cpp
#define MLA_PROLOG_TPL_SCH_MULTI_CORE  0
#define MLA_PROLOG_TPL_SCH_FEW_CORE    1
ASCENDC_TPL_ARGS_DECL(MlaProlog,
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 1, ASCENDC_TPL_UI_LIST,
        MLA_PROLOG_TPL_SCH_MULTI_CORE, MLA_PROLOG_TPL_SCH_FEW_CORE));
```

## 设计决策

- 为什么放弃 Matmul 高阶 API：别名随编译宏解析（MatmulImpl/MatmulClient/MatmulServiceAux），需要 host 侧 `matmul_tiling::MatmulApiTiling` + kernel 侧 `REGIST_MATMUL_OBJ` + CUBE↔VEC 跨核同步事件配合，首次实现风险高；同环境 GRU/LSTM（同团队产物）全部选择纯 AIV，是环境可用性的最强证据。
- 为什么外积累加而不是 Mul+ReduceSum（GRU 的做法）：本实现把 A 行常驻 UB，逐 k 加载 B 行做 `Muls+Add`，B 行连续加载对 DataCopyPad 友好；两者都是 VEC 软 matmul，本会话未做两者的性能对比。
- 为什么全程 fp32 中间精度：golden（torch）在 fp32 中计算后舍回 bf16，逐 k 在 fp32 累加与 golden 精度模型一致；只在最终 `Cast<bf16_t>(CAST_RINT)` 一次舍入。
- 为什么 `Cast<bf16_t>` 用 `CAST_RINT`、`Cast<float>` 用 `CAST_NONE`：CAST_RINT 为 round-to-nearest-even，对齐 torch `.to(bf16)` 的舍入；升精度无需舍入模式。
- 被放弃的设计：DESIGN.md 的 CUBE+VEC 四阶段融合流水线（Phase1-4 + 跨核同步）——迭代一未实现，待 Matmul API 可用性验证后再做；`tensor_api/arch` 底层通路（DataCopy→LoadData→Mmad→Fixpipe）作为第二备选也未尝试。
- 当前方案路线档位：`safety_fallback`（骨架正确性优先，性能让位于可编译性）。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510 / arch35 / ascend950），npu-smi 25.7.rc1，CANN 安装目录 cann-9.2.0（version.info 报 9.1.0 基线），x86_64 主机，cmake 3.20.5，g++ 13.3.0。
- 构建命令：`export ASCEND_HOME_PATH=... && export ASCEND_OPP_PATH=.../opp && bash build.sh --soc=ascend950 -j8`。
- 目录/文件（与事件流读写一一对应）：
  - `op_kernel/arch35/mla_prolog.h`——kernel 类（VecMatmul/LoadRowToFp32/VecRmsNorm/VecRope + 8 步流水）。
  - `op_kernel/arch35/mla_prolog_tiling_data.h`——TilingData（含 mmTileN、wsOffset*）。
  - `op_kernel/arch35/mla_prolog_tiling_key.h`——TilingKey（SCH_MODE 0/1）。
  - `op_kernel/mla_prolog_arch35.cpp`——kernel 入口。
  - `op_host/arch35/mla_prolog_tiling.cpp`——host tiling（多核切分 + mmTileN + workspace）。
  - `issues/issue_20260723_matmul_api_deviaton_01.md`——设计偏差 issue（待后续迭代处理）。
- 参考先例：`opp/vendors/bench_no_harness_gru_custom/op_impl/ai_core/tbe/.../dynamic/bench_no_harness_gru_arch35.cpp` 及同目录 `arch35/bench_no_harness_gru.h`、`*_tiling_key.h`、`*_tiling_data.h`。

## 性能影响

本会话切片内无 profiling 数据，无 VEC 路线与 CUBE Matmul 路线的成对相对对比可引用。定性判断（未实测）：大 K（He=6144~7680）的 Step1/Step6 每个输出行需 K 次 DataCopyPad+Cast+Muls+Add 循环，无 CUBE 硬件加速，相对 Matmul 高阶 API 路线预期显著偏慢（此为回退路线的固有代价；真机性能对比留待后续迭代，见遗留问题）。

## 预防

- 在 DAV_3510 上启动含 GEMM 的自定义算子前，先检查 `opp/vendors/` 下同团队既有算子的 kernel 源码：若全部为纯 AIV 模式（无 `REGIST_MATMUL_OBJ`），应默认 Matmul 高阶 API 在该环境未验证，骨架迭代优先走 VEC 软 matmul。
- 需要评估 Matmul API 时先确认三件事：`GetMatmulApiTiling` 是设备侧 constexpr（host tiling 用 `matmul_tiling::MatmulApiTiling` 类）、`Matmul` 别名的编译宏解析分支、以及是否需要 `REGIST_MATMUL_OBJ`。

## 验证信号

以下信号来自该算子迭代一（骨架搭建）开发会话：完成上述纯 AIV 实现后用 `bash build.sh --soc=ascend950 -j8` 全量重建。注意：本切片内未上板（ST 仅 Mock 模式、CPU golden），以下均为编译级信号，不构成运行正确性证据。

- 内核编译通过并生成二进制：`[100%] Built target MlaProlog_ascend950`、`[100%] Built target binary`，产物 `build/op_kernel/ascendc_kernels/binary/ascend950/mla_prolog/MlaProlog_157c9d7f5324f6d1edefc24cb87040f5.o`（来源平台：950；证据等级：tool_observed，编译级）。
- 算子包生成：`custom_opp_ubuntu_x86_64.run`（462KB）构建成功，包内含 `op_impl/ai_core/tbe/mla_prolog_custom_impl/dynamic/mla_prolog_arch35.cpp` 与 tiling/proto/opapi 库（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN cann-9.2.0 目录（9.1.0 基线）编译验证；其他架构（A2/A3）有 Matmul 高阶 API 文档背书，不需要此回退（未验证，推断）。
- 编译通过 ≠ 运行正确：本实现的 NPU 数值正确性与精度未在本切片验证（真机 ST 在后续阶段）。
- VEC 软 matmul 的性能上限受 K 循环内 DataCopyPad+Cast+Muls+Add 逐行开销制约，大 K 场景预期显著慢于 CUBE 路线（未实测）。
- TilingData 不含 TCubeTiling 字段（mmTileN 取代之）；若切回 Matmul API 需重写 host tiling 的该部分。

**未决项**：

- 【优先】真机运行正确性与精度验证（ST 真机模式）未做——本切片边界外，下个接续点。
- VEC 路线与 Matmul 高阶 API 路线的真机成对性能对比未测量（后续迭代计划项）。
- `Matmul<MatmulType<GM, ND, bfloat16_t>, ...>` 在 DAV_3510 纯 CUBE 模式下的编译可用性从未尝试验证（issue 文件中的后续迭代计划）。
## 坏实践

- 在未经环境验证的情况下，直接按文档（dtype 表只列 A2/A3）假设 Matmul 高阶 API 在 DAV_3510 自定义算子中可用并投入完整 CUBE+VEC 融合实现——首次编译配置（SPLIT_CORE 宏组合 + MatmulApiTiling + REGIST_MATMUL_OBJ）极易卡死迭代。
- 设计时把 `GetMatmulApiTiling<...>(...)`（设备侧 `__aicore__ constexpr` 函数）当成 host 侧 tiling 入口使用。
