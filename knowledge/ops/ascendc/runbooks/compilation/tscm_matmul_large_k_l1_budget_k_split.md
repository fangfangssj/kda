---
type: "Runbook"
title: TSCM A/B 大 K GetTiling 失败：L1 预算+K 切分
description: A/B 常驻 L1(TSCM) 时 MultiCoreMatmulTiling 对大 K（M=128,N=1728,K=8192,half,L1=512KB）GetTiling 返回 ret=1 → TilingFunc GRAPH_FAILED；host 手动按 L1 预算 SetSingleShape（留 20% 余量），kernel 侧 K 循环 + IterateAll(enAtomic=1) AtomicAdd 累加。
tags: [task.debugging, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# TSCM A/B 大 K GetTiling 失败：L1 预算+K 切分

## 现象

为 arch35（Ascend950PR / DAV_3510，MIX 模式 taskRation 1:2）三维卷积反传 filter 梯度算子实现 L1 常驻 Matmul（A/B 用 `MatmulType<TPosition::TSCM, CubeFormat::NZ, half>`，`MultiCoreMatmulTiling` 计算 cubeTiling）后，跑 op_host 单元测试：标准大 K 用例（M=128, N=1728, K=8192, half）失败，其余小 K 用例全部通过：

```
.../test_*_tiling.cpp:145: Failure
Value of: ok
  Actual: false
Expected: true
Tiling failed (returned GRAPH_FAILED) for case: tiling_fp16_g1_standard
```

在 Host tiling 中加 fprintf 探针后看到：`M=128 N=1728 K=8192 GetTiling ret=1 sM=0 sN=0 sK=0`（其他用例 ret=0）。此前两版写法都失败过：`SetSingleShape(-1, -1, K)` 强制 singleCoreK=K（本用例 GRAPH_FAILED），改为全自动 `SetSingleShape(-1, -1, -1)` 后引擎对 K=8192 仍返回 ret=1。

## 适用场景

- 工程形态：算子仓开发 op_host TilingFunc + op_kernel Matmul；A/B 矩阵 L1 常驻（TSCM + NZ 格式）+ enVecND2NZ 配置。
- API 路径：`matmul_tiling::MultiCoreMatmulTiling`（`SetShape`/`SetSingleShape`/`SetOrgShape`/`GetTiling`）；Kernel 侧 `Matmul::IterateAll(gmAddr, enAtomic)`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510，L1=512KB，MIX taskRation 1:2）+ CANN 9.1.0。
- 适用粒度（transfer_scope）：`operator_family`（所有 L1 常驻 A/B 的 Matmul 类算子，大 K 场景）。
- 路线优先级：recommended_fastpath（K 切分 + 原子累加是 L1 常驻路线下的标准做法）。
- 明确不适用 / 未验证场景：A/B 在 GM（不占 L1）的传统路径不受此 L1 预算约束；bf16 以外 dtype 的字节数需按 `dtypeSize` 代入；L1 容量非 512KB 的平台需替换 `L1_SIZE_BYTES_T`。

## 根因/解决配方

根因：A/B 常驻 L1(TSCM) 时，`(singleCoreM + singleCoreN) * singleCoreK * dtypeBytes` 受 L1=512KB 硬约束。K=8192、half、sM=sN=16 时 `(16+16)*8192*2 = 512KB` 恰好占满 L1 无余量；无论是强制 `singleCoreK=K` 还是让引擎全自动求解，`MultiCoreMatmulTiling::GetTiling` 都找不到合法解，返回 ret=1 → TilingFunc 返回 `GRAPH_FAILED`。引擎全自动路径对 TSCM A/B 大 K 不给任何回退（ret=1 且 sK=0）。

**Host 侧：手动按 L1 预算设置 singleCore 形状**（after，完整片段）：

```cpp
    tiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    // 手动设置 singleCoreM/N/K，确保 L1 不超（A/B 在 TSCM，L1=512KB 共享）
    // 策略：singleCoreM/N 取 Min(shape, 16) 向上对齐 16；singleCoreK 按 L1 余量反推
    // 若 singleCoreK < K，Kernel 侧用 K 循环 + IterateAll(enAtomic=1) 累加
    constexpr int32_t L1_SIZE_BYTES_T = 512 * 1024;
    constexpr int32_t L1_RESERVE_RATIO_T = 8;  // 留 20% 余量给 Matmul 引擎内部
    constexpr int32_t NZ_ALIGN_T = 16;
    int32_t dtypeBytes = static_cast<int32_t>(dtypeSize);
    int32_t sM = std::min(static_cast<int32_t>(M), static_cast<int32_t>(16));
    int32_t sN = std::min(static_cast<int32_t>(N), static_cast<int32_t>(16));
    sM = CeilAlign(static_cast<int64_t>(sM), static_cast<int64_t>(NZ_ALIGN_T));
    sN = CeilAlign(static_cast<int64_t>(sN), static_cast<int64_t>(NZ_ALIGN_T));
    // (sM + sN) * sK * dtypeBytes ≤ L1 * 0.8
    int32_t maxSK = (L1_SIZE_BYTES_T * L1_RESERVE_RATIO_T / 10) / ((sM + sN) * dtypeBytes);
    int32_t sK = std::min(static_cast<int32_t>(K), maxSK);
    sK = (sK / NZ_ALIGN_T) * NZ_ALIGN_T;  // FloorAlign 16
    if (sK < NZ_ALIGN_T) { sK = NZ_ALIGN_T; }
    tiling.SetSingleShape(sM, sN, sK);
    tiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
```

before（两种失败写法）：

```cpp
    // 写法一（失败）：强制 singleCoreK = K，K=8192 时 (sM+sN)*K*2 占满 L1
    tiling.SetSingleShape(static_cast<int32_t>(-1), static_cast<int32_t>(-1), static_cast<int32_t>(K));
    // 写法二（失败）：全自动，引擎对 K=8192 的 TSCM A/B 找不到合法解，GetTiling ret=1
    tiling.SetSingleShape(static_cast<int32_t>(-1), static_cast<int32_t>(-1), static_cast<int32_t>(-1));
```

**Kernel 侧：M×N 外层循环内加 K 方向 kLoops 循环，用 IterateAll 的 enAtomic 参数累加**（`IterateAll` 第二参数 0=写入、1=AtomicAdd 累加；首 tile 写入、后续 tile 累加，避免先清零 wsC）：

```cpp
    int32_t kLoops = static_cast<int32_t>((K_ + singleCoreK - 1) / singleCoreK);
    if (kLoops <= 0) { kLoops = 1; }

    for (int32_t loopIdx = static_cast<int32_t>(blockIdx_); loopIdx < totalBlocks; loopIdx += curCoreNum) {
        // ... mIdx/nIdx/mStart/nStart/curM/curN 按二维切分推导 ...
        for (int32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
            int32_t kStart = kIdx * singleCoreK;
            int32_t curK = Min(singleCoreK, static_cast<int32_t>(K_) - kStart);
            if (curK <= 0) { break; }

            // A/B 矩阵 GM(ND) → L1(NZ)，逐 K tile CopyND2NZOnTheFly
            LocalTensor<D_T_X> aL1 = qL1TQue_.template AllocTensor<D_T_X>();
            Conv3dCopyND2NZOnTheFly<D_T_X>(aL1, gmWsGrad_[static_cast<int64_t>(mStart) * K_ + kStart],
                                            0, 0, curM, curK, static_cast<int32_t>(K_));
            LocalTensor<D_T_X> bL1 = kL1TQue_.template AllocTensor<D_T_X>();
            Conv3dCopyND2NZOnTheFly<D_T_X>(bL1, gmWsIm2col_[static_cast<int64_t>(nStart) * K_ + kStart],
                                            0, 0, curN, curK, static_cast<int32_t>(K_));

            mm_.SetTail(curM, curN, curK);
            int32_t roundK = (curK + CONV3D_BP_BLOCK_NUM_PER_FRACTAL - 1) /
                             CONV3D_BP_BLOCK_NUM_PER_FRACTAL * CONV3D_BP_BLOCK_NUM_PER_FRACTAL;
            mm_.SetOrgShape(roundM, roundN, roundK);
            mm_.SetTensorA(aL1);
            mm_.SetTensorB(bL1, true);  // isTransB=true
            // enAtomic: 首次 K tile=0（写入），后续=1（AtomicAdd 累加）
            uint8_t enAtomic = (kIdx == 0) ? 0 : 1;
            mm_.IterateAll(gmWsC_[static_cast<int64_t>(mStart) * N_ + nStart], enAtomic);
            mm_.End();

            qL1TQue_.FreeTensor(aL1);
            kL1TQue_.FreeTensor(bL1);
        }
    }
```

## 设计决策

- 为什么不用引擎全自动：对 K=8192 的 TSCM A/B，引擎（`SetSingleShape(-1,-1,-1)`）直接 ret=1 失败，不给任何回退。
- 为什么不强制 singleCoreK=K：会占满 L1 且同样失败；把"K 必须整段驻留"的假设去掉后，K 切分 + AtomicAdd 累加是唯一可行路线。
- 为什么留 20% L1 余量（`L1_RESERVE_RATIO_T = 8` 即预算取 L1*0.8）：Matmul 引擎内部（depthA1/depthB1 双缓冲、L0C 等）还需 L1 空间，满预算会被引擎拒绝。
- enAtomic 取值依据 `IterateAll` 文档：0=不开启 Atomic（写入），1=AtomicAdd，2=AtomicMax，3=AtomicMin。
- 被放弃的方案：把 K 循环放在外层（先聚合 K 再 M×N）——会放大 L1 复用次数且与参考实现的 outer loop 结构不一致。

## 系统信息

- 工具链：`bash build.sh --soc=ascend950 -j8`；CANN 9.1.0（`$ASCEND_HOME_PATH` 指向安装根）。
- 目标平台：Ascend950PR（NpuArch DAV_3510 / arch35，MIX 模式 taskRation 1:2，L1=512KB / L0C=256KB / UB=248KB，平台接口已验证正确上报）。
- 相关文件：`op_host/arch35/*_tiling.cpp`（SetSingleShape 预算）、`op_kernel/arch35/*.h`（MatmulCompute K 循环）、`op_kernel/arch35/*_tiling_data.h`（cubeTiling 字段）、UT `tests/ut/op_host/test_*_tiling.cpp`。

## 预防

- L1 常驻 A/B 的设计阶段就把 K 切分 + enAtomic 累加纳入 kernel 骨架，不要假设 singleCoreK=K。
- Host tiling 对 GetTiling 返回值始终 `OP_CHECK_IF(ret != 0, ...)` 并附带 M/N/K 上下文日志（fprintf 探针），失败时可快速定位是引擎无解。

## 验证信号

本验证来自该算子 L1 常驻方案落地过程中的 op_host UT 循环（fprintf 探针输出 cubeTiling 字段）：

- 修复前：标准大 K 用例（M=128 N=1728 K=8192）`Tiling failed (returned GRAPH_FAILED)`；探针 `GetTiling ret=1 sM=0 sN=0 sK=0`（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 10/10 全过，探针 `M=128 N=1728 K=8192 GetTiling ret=0 sM=16 sN=16 sK=6544 usedCore=32`（K=8192 切成 kLoops=2）；其余用例 `ret=0` 且 sK 随 K 自适应（来源平台：950；证据等级：tool_observed）。
- Kernel 侧 K 循环版本编译通过并进入后续 ST 联调（来源平台：950；证据等级：tool_observed）。

## 失效边界

- L1 容量、`NZ_ALIGN_T`、余量比例均按 950/arch35 硬编码；其他平台需重推 `L1_SIZE_BYTES_T`（未验证，推断可迁移）。
- enAtomic=1 的 AtomicAdd 累加在 C 为 GM float 时验证；C 输出到 UB（VECCALC）路径未验证。
- wsC 首 tile 写入依赖 `kIdx==0` 时 enAtomic=0，若多核同一 M×N block 被重复处理会破坏该假设（当前 totalBlocks 按 coreNum 步进保证每 block 单核处理）。

**未决项**：

- `MultiCoreMatmulTiling` 全自动模式为何对 TSCM A/B 大 K 不做 K 切分求解（引擎内部策略），未追根因；如需依赖全自动，须先在引擎层确认。
## 坏实践

- 为"简化 Kernel 逻辑"强制 `SetSingleShape(-1,-1,K)`：大 K 必然 L1 超限，直接把失败推迟到 tiling 阶段。
- Host tiling 代码直接使用 Kernel 侧助手 `Min(...)`：op_host（`namespace optiling`）无此符号，报 `error: 'Min' was not declared in this scope; did you mean 'sin'?`；须用 `std::min` 并确认 `<algorithm>` 可用。
- GetTiling 失败后不打印 M/N/K/sM/sN/sK 上下文，直接返回 GRAPH_FAILED——无法区分"引擎无解"与"参数算错"。

## 误诊

- 初始怀疑"K=8192 太大，引擎即使切分 K 也放不下"——实际引擎根本没给出切分解（ret=1 且 sK=0），不是切分后仍超，而是全自动路径对 TSCM A/B 大 K 无解；手动预算 + 显式 K 切分才解决。
