---
type: "Runbook"
title: "带主元 Thomas 求解器 SIMT 直访 GM kernel"
description: "串行前向消元有行间数据依赖的求解器在昇腾 SIMT 上用\"每线程一列 RHS + grid-stride + 中间量直写 GM workspace（d'd+du'd+du2' 三段）\"模式实现；不用 AIV/UB，不依赖 math 库。"
tags: ["task.implementation", "topic.programming_model.execution_model.simt", "topic.data_movement.memory_access", "topic.compute.composite"]
status: stable
sources: []
created_at: "2026-08-15T09:06:04Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 带主元 Thomas 求解器 SIMT 直访 GM kernel

## 现象

aclsparseSgtsv2（带部分选主元的三对角线性系统 A·X=B 求解器，FP32，Ascend950/arch35）需要一个 kernel 实现。难点是：前向消元带 partial pivoting，**第 i 步依赖第 i-1 步的结果**（行间数据依赖，串行），无法像逐元素算子那样按行/块并行；同时主元交换会产生两类 fill-in（du' 与 du2'），需要跨步保存。如何在昇腾 SIMT 编程模型上落地一个正确且可复用的实现。

## 适用场景

- **工程形态**：kernel 直调（ops-sparse Legacy API，SIMT `__simt_callee__` / `__simt_vf__` kernel，直接读写 GM）。
- **API 路径**：`aclsparseSgtsv2` → `gtsv2_kernel_do` → `Gtsv2Launcher` → `asc_vf_call<Gtsv2SimtCompute>` → `Gtsv2ProcessOneRhs`。
- **验证过的平台与 CANN 版本**：Ascend950PR / arch35，CANN 9.2.0（CP3/CP4 上板实测，69/69 功能通过）。
- **适用粒度**：`operator_family`（带行间串行依赖 + partial pivoting 的带状/三对角直接求解器族通用）。
- **路线优先级**：`recommended_fastpath`（行间串行求解器在 SIMT 上用 per-RHS 并行 + 直访 GM workspace 是与 gtsv2_nopivot/gtsv_interleaved_batch 一致且经验证的选型）。
- **明确不适用/未验证场景**：可按行并行的逐元素算子应走 AIV/UB 向量化，不套本模式；AIV+CR（循环约化）无主元路径（如 gtsv2_strided_batch）结构不同，不直接套用。

## 根因

行间串行依赖决定了**并行维度只能在 RHS 列方向**，不能在行方向。SIMT 模型（每线程一个轻量标量执行体、直访 GM、`blockIdx`/`threadIdx` 语义）天然契合"每线程处理一列 RHS、跨列 grid-stride"的切分；而 AIV 的 block 向量化模型需要相邻行数据已就绪才有收益，前向消元的串行链使其无法有效利用 UB 批量搬运，且主元分支使控制流不规则。因此选 SIMT 直访 GM，把前向/回代的中间量（修改后的 d'、du'、du2'）直接落在 GM workspace，寄存器只持有当前 pivot 窗口（pivot_d/pivot_du/pivot_b 与预读的 cur_d/cur_du/cur_b）。

## 解决配方

### 设计骨架

- **切分**：`numBlocks = min(ceil(n / kGtsv2ThreadsPerBlock), aivCoreNum)`，每 block `kGtsv2ThreadsPerBlock=32` 线程，每线程一个 RHS 列，grid-stride 遍历剩余列。
- **GM layout**：dl/d/du 各 `[m]` 全 RHS 共享只读；B 列主序 `[ldb*n]`；workspace `[3*m]` per RHS = wsD + wsDu + wsDu2（含主元交换产生的两类 fill-in）。
- **前向消元**：寄存器持有 pivot 窗口（row i-1 的 pivot_d/pivot_du/pivot_b）+ 预读 row i（cur_d/cur_du/cur_b）；step i 比较 `|pivot_d|` 与 `|dl[i]|` 决定 swap/noSwap，算出 row i-1 终态 storeD/storeDu/storeDu2/storeB 写回 GM，row i 成为新 pivot。dl[i] 始终读 GM 原始值（step j 只改 dl[j]，不影响 dl[i>j]）。
- **回代**：x[m-1]=b'/d'；i=m-2..0：`x[i]=(b'[i] - du'[i]*x[i+1] - (i+2<m ? du2'[i]*x[i+2] : 0)) / d'[i]`，用 xNext/xNext2 双滞后寄存器。
- **无 math 依赖**：`Gtsv2Fabsf` 内联比较替代 `fabsf`，与仓内 SIMT "无 math 函数依赖"风格一致（消除 SIMT 下 fabsf 可用性的不确定性）。

### 完整实现

（完整实现单元；逐字代码见下 `## 完整代码`。）

## 完整代码

`sparse/gtsv2/arch35/gtsv2_kernel.cpp`（最终版，verbatim 取自轨迹内文件读取事件）：

```cpp
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "gtsv2_kernel.h"

// ---------------------------------------------------------------------------
// 绝对值内联实现（CP2.2 建议：使用内联比较，消除 fabsf 依赖不确定性）
// 与仓内"无 math 函数依赖"的 SIMT 风格一致。
// ---------------------------------------------------------------------------
static __simt_callee__ inline float Gtsv2Fabsf(float x)
{
    return x < 0.0f ? -x : x;
}

__simt_callee__ inline void Gtsv2ProcessOneRhs(
    __gm__ const float *dlGm,
    __gm__ const float *dGm,
    __gm__ const float *duGm,
    __gm__ float *BGm,
    __gm__ float *wsGm,
    int32_t m, int32_t ldb, int32_t rhsIdx)
{
    if (m < 2) {
        if (m == 1) {
            int64_t colOff = static_cast<int64_t>(rhsIdx) * ldb;
            BGm[colOff] = BGm[colOff] / dGm[0];
        }
        return;
    }

    int64_t colOff = static_cast<int64_t>(rhsIdx) * ldb;
    // Workspace per RHS: 3*m floats. wsOff = rhsIdx * 3 * m
    //   wsD[0..m-1]   = ws[wsOff + 0     .. wsOff + m - 1]   (modified d')
    //   wsDu[0..m-1]  = ws[wsOff + m     .. wsOff + 2*m - 1] (modified du',含 fill-in)
    //   wsDu2[0..m-1] = ws[wsOff + 2*m   .. wsOff + 3*m - 1] (2nd super-diagonal du2', swap fill-in)
    int64_t wsOff = static_cast<int64_t>(rhsIdx) * 3 * static_cast<int64_t>(m);

    // ---- Phase 1: Forward elimination with partial pivoting ----
    float pivot_d  = dGm[0];
    float pivot_du = duGm[0];
    float pivot_b  = BGm[colOff];

    float cur_d  = dGm[1];
    float cur_du = duGm[1];
    float cur_b  = BGm[colOff + 1];

    for (int32_t i = 1; i < m; i++) {
        float dl_i = dlGm[i];

        float absPivotD = Gtsv2Fabsf(pivot_d);
        float absDlI    = Gtsv2Fabsf(dl_i);
        bool noSwap = (absPivotD >= absDlI);

        float storeD;
        float storeDu;
        float storeDu2;
        float storeB;
        float nextPivotD;
        float nextPivotDu;
        float nextPivotB;

        if (noSwap) {
            float mult = dl_i / pivot_d;
            storeD  = pivot_d;
            storeDu = pivot_du;
            storeDu2 = 0.0f;
            storeB  = pivot_b;
            nextPivotD  = cur_d - mult * pivot_du;
            nextPivotDu = cur_du;
            nextPivotB  = cur_b - mult * pivot_b;
        } else {
            // Row swap pushes old du[i] onto 2nd super-diagonal (du2) of row i-1.
            float mult = pivot_d / dl_i;
            storeD  = dl_i;
            storeDu = cur_d;
            storeDu2 = cur_du;                      // du2'[i-1] = old du[i] (FILL-IN)
            storeB  = cur_b;
            nextPivotD  = pivot_du - mult * cur_d;
            nextPivotDu = -mult * cur_du;            // du'[i] = FILL-IN (1st super-diagonal)
            nextPivotB  = pivot_b - mult * cur_b;
        }

        wsGm[wsOff + (i - 1)]               = storeD;
        wsGm[wsOff + m + (i - 1)]           = storeDu;
        wsGm[wsOff + 2 * m + (i - 1)]       = storeDu2;
        BGm[colOff + (i - 1)]               = storeB;

        pivot_d  = nextPivotD;
        pivot_du = nextPivotDu;
        pivot_b  = nextPivotB;

        if (i + 1 < m) {
            cur_d  = dGm[i + 1];
            cur_du = duGm[i + 1];
            cur_b  = BGm[colOff + (i + 1)];
        }
    }

    wsGm[wsOff + (m - 1)]         = pivot_d;
    wsGm[wsOff + m + (m - 1)]     = pivot_du;
    wsGm[wsOff + 2 * m + (m - 1)] = 0.0f;

    // ---- Phase 2: Backward substitution ----
    // Note: no singularity guard — zero pivot produces Inf/NaN (by design)
    float xNext = pivot_b / pivot_d;
    float xNext2 = 0.0f;
    BGm[colOff + (m - 1)] = xNext;

    for (int32_t i = m - 2; i >= 0; i--) {
        float bPrime   = BGm[colOff + i];
        float dPrime   = wsGm[wsOff + i];
        float duPrime  = wsGm[wsOff + m + i];
        float du2Prime = wsGm[wsOff + 2 * m + i];
        float numerator = bPrime - duPrime * xNext;
        if (i + 2 < m) {
            numerator -= du2Prime * xNext2;
        }
        float xI = numerator / dPrime;
        xNext2 = xNext;
        xNext = xI;
        BGm[colOff + i] = xI;
    }
}

// SIMT VF 入口: 每个 thread 处理一个 RHS 列 (grid-stride)
__simt_vf__ __aicore__ __launch_bounds__(kGtsv2ThreadsPerBlock) inline void
Gtsv2SimtCompute(
    __gm__ float *dlGm,
    __gm__ float *dGm,
    __gm__ float *duGm,
    __gm__ float *BGm,
    __gm__ float *wsGm,
    int32_t m, int32_t n, int32_t ldb, int32_t gridStrideInRhs)
{
    for (uint32_t rhsIdx = blockIdx.x * kGtsv2ThreadsPerBlock + threadIdx.x;
         rhsIdx < static_cast<uint32_t>(n);
         rhsIdx += static_cast<uint32_t>(gridStrideInRhs)) {
        Gtsv2ProcessOneRhs(dlGm, dGm, duGm, BGm, wsGm, m, ldb,
                           static_cast<int32_t>(rhsIdx));
    }
}
```

Host 侧 launch（`gtsv2_host.cpp` `LaunchGtsv2Kernel`，决定 numBlocks 并填 TilingData）：

```cpp
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSgtsv2", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    // SIMT: numBlocks = ceil(n / kGtsv2ThreadsPerBlock), capped at aivCoreNum
    uint32_t numBlocks = std::min(
        (static_cast<uint32_t>(n) + kGtsv2ThreadsPerBlock - 1u) / kGtsv2ThreadsPerBlock,
        aivCoreNum);
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }
    Gtsv2TilingData tiling{};
    tiling.m = m; tiling.n = n; tiling.ldb = ldb; tiling.numBlocks = static_cast<int32_t>(numBlocks);
    gtsv2_kernel_do(/*GM_ADDRs...*/, tiling, numBlocks, stream);
```

`gtsv2_kernel.h` 常量与 `kernel_do` 声明：

```cpp
constexpr uint32_t kGtsv2ThreadsPerBlock = 32U;
void gtsv2_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
                     GM_ADDR workspace, const Gtsv2TilingData &tiling,
                     uint32_t numBlocks, void *stream);
```

## 系统信息

- **toolchain**：CANN 9.2.0 Ascend C 编译器（`bash build.sh --ops=gtsv2 --soc=ascend950`），CMake ≥3.16，GCC 13.3.0。
- **目标平台**：Ascend950PR / arch35（DAV-3510），AIV 多核（`GetAivCoreCount()` 动态获取）。
- **目录布局**：`sparse/gtsv2/arch35/`（`gtsv2_kernel.cpp` / `gtsv2_kernel.h` / `gtsv2_host.cpp` / `gtsv2_tiling_data.h` / `README.md`）。
- **相关文件清单**（与事件流读写一一对应）：
  - `sparse/gtsv2/arch35/gtsv2_kernel.cpp` — `Gtsv2Fabsf` / `Gtsv2ProcessOneRhs` / `Gtsv2SimtCompute` / `Gtsv2Launcher`
  - `sparse/gtsv2/arch35/gtsv2_kernel.h` — `kGtsv2ThreadsPerBlock`、`gtsv2_kernel_do` 声明
  - `sparse/gtsv2/arch35/gtsv2_host.cpp` — `ValidateGtsv2Params` / `ComputeWorkspaceBytes` / `LaunchGtsv2Kernel` / `aclsparseSgtsv2` / `aclsparseSgtsv2_bufferSizeExt`
  - `sparse/gtsv2/arch35/gtsv2_tiling_data.h` — `Gtsv2TilingData{m,n,ldb,numBlocks,reserved}`
  - 参考算子：`sparse/gtsv2_nopivot/arch35/`（无主元版，SIMT 结构模板）

## 设计决策

- **为何选 SIMT 而非 AIV/UB**：前向消元 `d'[i]` 依赖 `d'[i-1]`，行方向严格串行；AIV 向量化需要相邻数据已就绪才有搬运/计算重叠收益，串行链 + 主元分支使控制流不规则，UB 批量搬运无收益。SIMT"一线程一列 RHS"把并行度放在无依赖的 RHS 列方向，正交于串行链。仓库先例 `gtsv2_nopivot` / `gtsv_interleaved_batch` 同为 SIMT，结构同构可复用。
- **为何中间量直写 GM 而不入 UB**：前向写 row i-1、回代读 row i，访问跨度大且不规则（主元交换打乱行序），UB 搬运收益不足；直访 GM + 寄存器 pivot 窗口最简且与 nopivot 同构。代价是 GM 流量增大（见性能影响）。
- **grid-stride 而非每核固定列**：n 可远大于核数，grid-stride 让任一空闲线程拾取下一列，负载均衡；`numBlocks` 封顶 `aivCoreNum` 避免过度派发。
- **`Gtsv2Fabsf` 内联而非 `fabsf`**：SIMT kernel 对 math 库函数可用性不确定（开发期确认 `fabsf` 在 SIMT 环境的可用性存疑），内联三元比较消除依赖，与仓内"无 math 函数依赖"SIMT 风格一致。
- **被放弃的候选**：① AIV + per-core factorization + batch RHS（Option 2）：串行链使 per-core factorization 无法跨核并行，且 UB 收益不足；② SIMT + shared factorization via GM（Option 3）：所有核共享同一前向分解再各自回代——但 partial pivoting 的行交换使"共享分解"语义复杂且对多 RHS 收益有限。均放弃，选 per-RHS 独立 SIMT（Option 1）。
- **`kGtsv2ThreadsPerBlock=32`**：与 nopivot/gtsv_interleaved_batch 一致，保持同仓 SIMT block 宽度统一。

## 性能影响

相对 gtsv2_nopivot 的成对代价：GM workspace 流量约 3×（gtsv2 每步 6 floats = 3 读 wsD+wsDu+wsDu2 + 3 写；nopivot 每步 2 floats = 1 读 wsD + 1 写）。主元 swap 分支额外 ~3-5 个标量 ops/step，实测开销仅 1.16~1.25%（非瓶颈）。结构性瓶颈是 GM workspace 读写带宽，源自 3 组 fill-in 数组的 GM 读写。参照路线：gtsv2_nopivot（无主元，1×m workspace），来源平台 Ascend950（证据等级: inferred，由代码结构推算；CP4 用 chrono+sync 计时，未采 msprof 带宽/AIU 利用率）。

## 预防

- 行间串行求解器不要试图在行方向 AIV 并行——并行度放 RHS 列方向。
- workspace 段数 = fill-in 数组数 + 1（d'）；改算法时同步改 host `ComputeWorkspaceBytes` 的倍数与 GM layout 注释。

## 验证信号

> 来源：CP3 功能验收 + 4.1 性能采集（Ascend950PR/arch35，CANN 9.2.0）。gtest/perf 由子 agent 在 build/test/gtsv2 运行；原始 bash 输出被 markdown 转写适配器 redact，下列数值取自报告产物与 agent 对输出的引用。

- 功能：69/69 测试通过（50 CSV 数据驱动 + 19 TEST_F 异常），覆盖 diag_dominant/identity/constant（不触发 swap）、random/pivot_all_swap/pivot_first_only/pivot_extreme_dl（触发 swap）、3 条奇异矩阵（Inf/NaN 传播）、尾线程（n 非 32 整数倍）（来源平台：Ascend950；证据等级: tool_observed）。
- 精度：46 条数值比对全过，最大 MERE=0.000003（L1_1024x64_random），阈值 5e-4；最大 MARE=0.003841（阈值 0.01）（来源平台：Ascend950；证据等级: tool_observed）。
## 失效边界

- 本模式对"行间串行 + per-RHS 独立"的直接求解器族（`operator_family`）适用；可按行并行的算子不适用。
- 未验证、推断可迁移：Ascend910B/a2 等 SIMT 平台同构适用，但本 trace 仅 Ascend950 实测。
- 性能瓶颈（3× GM 流量）是 partial pivoting 的结构性代价，n 极大且 RHS 列数远超核数时 GM 带宽可能成为瓶颈；若需绝对带宽数据须补 msprof op 采集（本 trace 未采）。
