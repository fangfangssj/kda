---
type: "Runbook"
title: "507035 崩溃：多核分发未拦截多余核越界"
description: "任务数小于硬件启动核数时多余核以越界任务号访问 GM 触发 507035；修复路线一：Process() 入口加 cid>=uc 早退；修复路线二（R2）：分片并行度以 AscendC::GetBlockNum()（实际启动网格）为准重新切分——图模式编译的 blockDim 与 tiling numCores 之间没有一致性保证，kernel 侧对齐实际网格才是鲁棒做法（对图模式残余失败必要而不充分）。复现增量：plog 核号对比判别形态（blockDim=18 而 core_id=42-44 越界 → 守卫后仅剩 <18 核号报错），CANN 9.1.0 复验；守卫不消除合法核的 tile 超 UB 崩溃（并存独立根因）。"
tags: [task.debugging, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: "2026-09-17T05:05:01Z"
platforms: ["950"]
---

# 507035 崩溃：多核分发未拦截多余核越界

## 现象

在开发基于 ACLNN 算子仓工程生成的 Ascend C 融合算子（BenchHarnessEngramGateFusion，arch35，Ascend950PR，CANN 9.2.0）时，运行 `tests/aclnn/run.sh` 的 23 个 ST 用例：小任务量用例（标准 prefill/decode、边界与异常注入）全部通过，而 4 个大 D + 大 L 组合用例（如 D=1024、L=2048 的 prefill 用例与 D=769 的质数 D 用例）在 `aclrtStreamSynchronize` 处返回 507035（向量核异常）崩溃。当时 kernel 的多核分发逻辑是"任务总数 tt=B*HC 按 tiling 下发的 usedCoreNum 均分给各核"，没有对核号超过实际任务数的核做任何拦截。编译无任何报错，UB 预算检查也在限额内（约 127KB / 192KB），崩溃与用例任务规模强相关：任务数远小于核数且单任务 D 很大时必崩，任务数同样小但 D 小的用例却静默通过。

## 适用场景

- 工程形态：ACLNN 框架/算子仓流程生成的 Ascend C kernel（aclnn 单算子 ST 直调验证）；手写多核任务均分分发的 AIV kernel 同样适用。
- API 路径：`AscendC::GetBlockIdx()`、tiling 侧 `multicore.usedCoreNum`、按 `tt/uc` 均分任务的手写 Process() 分发。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（任何"总任务数可能小于硬件启动核数"的手写多核分发 kernel）。
- 路线优先级：recommended_fastpath（入口早退一次比较，无成本）。
- 明确不适用/未验证：框架自动切分且保证核数≤任务数的场景；tt≥核数但任务内部索引（l/K/dil）自身越界的崩溃不适用；其他 Ascend 平台未验证（推断可迁移，见失效边界）。

## 根因

Process() 多核分发未拦截 cid≥uc 的多余核：硬件启动全部核（本例 56 核），但任务数 tt=B*HC 可能远小于核数（崩溃用例 tt=4），多余核（cid≥4）的 s/e 计算使 f≥tt，`ProcessOneBhc(f/HC_, f%HC_)` 以越界的 (b, hc) 访问 GM，触发向量核异常 507035。

机制链：`uc=min(tt, bd)` 只约束了"每核至少分到任务"的除法安全性，没有约束"核数>任务数"时多余核的空转路径——多余核的 `s` 从 `bc*(tm+1)+(cid-bc)*tm` 计算出来必然 ≥tt，循环体用越界下标读 GM。小 tt 用例（如 tt=2）同样存在多余核越界，但越界地址恰好落入合法 GM 映射区未触发硬件异常，表现为"静默通过"——这是该坑高危的原因：小规模回归全绿，大规模用例才崩。

## 判别方法

用 kernel 侧 printf 带块索引探针定位"哪些核在执行什么任务"（NPU kernel 无源码级调试器，printf 是定位崩溃点的可行手段）：

```cpp
if(l==0 && D_>=769) printf("D=%d dRep=%d SL=%d K=%d\n",(int)D_,(int)dRep_,(int)stateLen_,(int)K_);
```

```bash
source /home/<user>/cann/Ascend/cann-9.2.0/set_env.sh && export ASCENDC_DUMP=1 \
  && cd operators/<op>_package/tests/aclnn && bash run.sh 2>&1 | grep "AIV Block"
```

原始观测（关键证据）：运行时输出前缀显示硬件实际启动了 56 个 AIV Block，且 Block 28/29/30/31…（远超该用例任务总数 4）都在执行 D=1024 的 kernel 主循环：

```text
[AIV Block 0/2] D=769 dRep=13 SL=9 K=4
[AIV Block 28/56] D=1024 dRep=16 SL=9 K=4
[AIV Block 29/56] D=1024 dRep=16 SL=9 K=4
...
[AIV Block 49/56] D=1024 dRep=16 SL=9 K=4
```

判别要点：看到"启动核数（如 56）> 任务总数（B*HC）"且崩溃用例集中在大 D，即可命中本根因；反过来若 tt≥核数仍崩，则查任务内部索引。

同错误码 507035 的三种不同根因（同码互斥判别，先分诊再修复）：

- [507035 workspace 保留区卡](mte_ddr_507035_workspace_reserved_region.md)——根因是 tiling 声明的 workspace 未含平台保留区（950 为 16MB），kernel 对用户工作区的 MTE 写落在分配范围外（plog 错误码 95 MTE DDR 越界）；区分条件：kernel 用 workspace 且 plog 报 MTE DDR 越界→查保留区，printf 见多余核执行主循环→查本卡分发守卫。
- [507035 负模环形索引卡](507035_negative_modulo_ring_index_ub_offset.md)——根因是 C++ 负模产生负索引、标量 GetValue 以负偏移访问 UB；区分条件：崩溃与短序列（L<SL）用例强相关→查负模索引，崩溃集中在大 D+大 L 且任务数<核数→查本卡。

其他相邻卡片与区分条件：

- [多核 ceil 尾块为负卡](../precision/ceil_partition_each_core_must_clamp_remaining.md)——分区算术错误（ceil 分块后末核 tail 可为负、中后核越界），修复是每核按 total-start 截断；本卡是"核数>任务数"的多余核空转路径，修复是入口早退，两者可叠加出现。
- [多核共享首段行区间卡](../precision/multiblock_launch_must_derive_row_range_from_block_index.md)——host 把首个行区间传给所有 block 致重复处理/覆盖漏写；本卡是多余核以越界任务号读 GM，现象是崩溃而非漏写。
- [空 batch 全核早退卡](empty_branch_kernel_empty_a_early_exit_empty_r_zero_fill.md)——usedCoreNum=0/空张量分支的全核早退配方；本卡 usedCoreNum>0 但小于硬件启动核数，同为"GetBlockIdx 自查自退"家族。
- [block 数覆盖可用 Vector 核卡](../performance/block_count_uses_all_available_vector_cores.md)——host 侧 block 上限的性能视角（上限应绑定设备实测核数）；本卡是 kernel 侧守卫，两者互补。

## 解决配方

before（修复前，任务均分无多余核拦截）：

```cpp
template <typename T, int isEmpty>
__aicore__ inline void BenchHarnessEngramGateFusionKernel<T, isEmpty>::Process() {
    int64_t tt=B_*HC_, bd=td_->multicore.usedCoreNum; if(bd<1) bd=1;
    int64_t uc=(tt<bd)?tt:bd; if(uc<1) uc=1;
    int64_t tm=tt/uc, bc=tt%uc, cid=AscendC::GetBlockIdx(), s, e;
    if(cid<bc){s=cid*(tm+1);e=s+tm+1;} else {s=bc*(tm+1)+(cid-bc)*tm;e=s+tm;}
    for(int64_t f=s;f<e;f++) ProcessOneBhc(f/HC_,f%HC_);
}
```

after（Process() 入口早退多余核；Empty kernel 的 Process() 同样加守卫）：

```cpp
template <typename T, int isEmpty>
__aicore__ inline void BenchHarnessEngramGateFusionKernel<T, isEmpty>::Process() {
    int64_t tt=B_*HC_, bd=td_->multicore.usedCoreNum; if(bd<1) bd=1;
    int64_t uc=(tt<bd)?tt:bd; if(uc<1) uc=1;
    int64_t cid=AscendC::GetBlockIdx();
    if(cid>=uc) return;  // extra cores beyond usedCoreNum have no work
    int64_t tm=tt/uc, bc=tt%uc, s, e;
    if(cid<bc){s=cid*(tm+1);e=s+tm+1;} else {s=bc*(tm+1)+(cid-bc)*tm;e=s+tm;}
    for(int64_t f=s;f<e;f++) ProcessOneBhc(f/HC_,f%HC_);
}
```

Empty kernel 内嵌的第二个分发点同样处理：

```cpp
        int64_t tt=B_*CD_, bd=td_->multicore.usedCoreNum; if(bd<1) bd=1;
        int64_t uc=(tt<bd)?tt:bd; if(uc<1) uc=1;
        int64_t cid=AscendC::GetBlockIdx();
        if(cid>=uc) return;
        int64_t tm=tt/uc, bc=tt%uc, s, e;
```

注意：同一 kernel 文件里可能有多个 Process/分发入口（模板主体 + Empty 分支），必须逐处加守卫，漏一处该分支仍崩。

### 修复路线二（独立复现 R2，BenchHarnessScatter，CANN 9.2.0）：以 GetBlockNum() 对齐实际启动网格

同一根因的另一触发形态：kernel 的多核分片用 **tiling 下发的 numCores** 计算区间，而图模式（GEIR）编译实际启动的网格是独立决策（本例启动 6 个 block 而 tiling numCores=1）——`cid≥numCores` 的 block 计算出越界 GM 偏移，图模式 8 用例全部 `RunGraph rc=255 EZ9999`（设备侧 aivec error 95 + MTE error）；aclnn 动态路径（`SetBlockDim` 运行时网格）同样不与 tiling numCores 强绑定。host 侧曾试图用 TilingParse 写 `compileInfo->coreNum=1` 让图编译器产 blockDim=1，实测该传导不可依赖（staging 仍启动 6 block）。

修复原则：**kernel 分片的并行度一律以 `AscendC::GetBlockNum()`（实际启动网格数）为准，tiling 的 numCores 只作缺省值**。`GetBlockNum()` 是公开 API（`asc/include/basic_api/kernel_operator_sys_var_intf.h` 中 `namespace AscendC` 下 `__aicore__ inline int64_t GetBlockNum()`），与既有 `GetBlockIdx()` 同源。

修复前（`CopyDataToY`，三处同型分片函数之一，verbatim）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::CopyDataToY()
{
    int64_t dataSize=1; for(int64_t k=0;k<td_->rank;k++) dataSize*=td_->dataShape[k];
    int64_t numCores=td_->multicore.numCores; if(numCores<1) numCores=1;
    int64_t cpc=dataSize/numCores, ct=dataSize%numCores;
    int64_t cid=AscendC::GetBlockIdx();
    int64_t ms=cid*cpc+(cid<ct?cid:ct), mc=cpc+(cid<ct?1:0);
    if(mc<=0){ return; }
    int64_t maxChunk=td_->perBufBytes/sizeof(T); if(maxChunk<=0) maxChunk=1;
    for(int64_t off=0;off<mc;off+=maxChunk){
        int64_t cnt=(mc-off<maxChunk)?(mc-off):maxChunk;
        AscendC::DataCopyExtParams p; p.blockCount=1; p.blockLen=cnt*sizeof(T); p.srcStride=0; p.dstStride=0;
        AscendC::DataCopyPadExtParams<T> pad{};
        AscendC::DataCopyPad(buf_[B0].Get<T>(), gmData_[ms+off], p, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::DataCopyPad(gmY_[ms+off], buf_[B0].Get<T>(), p);
        // Ensure MTE3 (B0 read→GM write) completes before next iteration's MTE2
        // overwrites B0.  Without this, the next MTE2 corrupts B0 mid-read.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
}
```

修复后（同一函数，仅头部变化，verbatim）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::CopyDataToY()
{
    int64_t dataSize=1; for(int64_t k=0;k<td_->rank;k++) dataSize*=td_->dataShape[k];
    int64_t numCores=td_->multicore.numCores; if(numCores<1) numCores=1;
    // Align split with the actually-launched grid (GEIR graph path may compile
    // blockDim != tiling numCores; mismatch caused cid>=numCores blocks to
    // compute out-of-range GM offsets -> EZ9999 MTE DDR out of range).
    int64_t launchedBlocks=AscendC::GetBlockNum(); if(launchedBlocks>0) numCores=launchedBlocks;
    int64_t cpc=dataSize/numCores, ct=dataSize%numCores;
    int64_t cid=AscendC::GetBlockIdx();
    int64_t ms=cid*cpc+(cid<ct?cid:ct), mc=cpc+(cid<ct?1:0);
    if(mc<=0){ return; }
    // ……循环体与修复前完全一致（见上）
```

另两处同型修复（`CopyDataToAccFp32` 与 `CopyAccToYNative`，函数头 before/after，verbatim；两函数其余循环体不变）：

```cpp
// before（两处相同）：
    int64_t numCores = td_->multicore.numCores; if (numCores < 1) numCores = 1;
    int64_t cid = AscendC::GetBlockIdx();
    int64_t cpc = numel / numCores, ct = numel % numCores;
    int64_t base = cid*cpc + (cid < ct ? cid : ct);
    int64_t cnt  = cpc + (cid < ct ? 1 : 0);
    if (cnt <= 0) return;

// after（两处相同）：
    int64_t numCores = td_->multicore.numCores; if (numCores < 1) numCores = 1;
    // Align split with the actually-launched grid (see CopyDataToY note).
    int64_t launchedBlocks = AscendC::GetBlockNum(); if (launchedBlocks > 0) numCores = launchedBlocks;
    int64_t cid = AscendC::GetBlockIdx();
    int64_t cpc = numel / numCores, ct = numel % numCores;
    int64_t base = cid*cpc + (cid < ct ? cid : ct);
    int64_t cnt  = cpc + (cid < ct ? 1 : 0);
    if (cnt <= 0) return;
```

两条修复路线的选用条件：早退守卫（路线一）适合"多余核本无工作"的场景，一次整型比较零成本；`GetBlockNum()` 重切分（路线二）让全部已启动 block 分摊工作、保持并行度，且对"实际网格 < tiling 预期"的方向也正确（覆盖式重切保证不相交区间仍完整覆盖 `[0, dataSize)`；只单向钳制会保留另一方向的失配）。同一 kernel 的多个分片阶段（搬运/归约/回写）必须共享同一并行度来源。

## 设计决策

- 守卫放在 Process() 入口而不是 ProcessOneBhc() 内做 (b,hc) 边界判断：入口早退让多余核在触碰任何 GM 地址前退出，一次整型比较即可；在任务函数内判断则多余核仍会执行循环框架与部分访存。
- 备选方案"限制启动核数使其≤tt"不可行：核数由硬件/框架启动行为决定，kernel 侧只能通过 GetBlockIdx 自查自退。
- 当前方案属 recommended_fastpath：零额外同步、零额外访存。
- R2 路线补充设计：为什么是「覆盖」而不是「钳制」（`numCores = launchedBlocks` 而非 `min/max`）：无论实际网格比 tiling 预期多还是少，只要所有已启动 block 用同一个 numCores 重新切分，不相交区间仍然完整覆盖 `[0, dataSize)`；只单向钳制会保留另一方向的失配。host 侧对齐（TilingParse 写 coreNum 控制 blockDim）在图模式 staging 下不可依赖，kernel 侧读取运行时事实（GetBlockNum）才是鲁棒做法。SyncAll 参与性核对：函数内 `if (mc<=0) return;` 只退出分片 helper，外层 `Process()` 会继续执行 `AscendC::SyncAll()`——所有已启动 block 都能到达集合点，不会因空区间 block 提前退出而挂死。

## 系统信息

- 工具链：CANN 9.2.0（`~/cann/Ascend/cann-9.2.0/set_env.sh`），cmake 3.28.3，GNU 13.3.0；目标平台 ascend950（arch35）。
- 相关文件：`operators/<op>_package/op_kernel/arch35/*_kernel.h`（Process 分发，本次修复点）；`op_kernel/arch35/*_tiling_struct.h`（multicore.usedCoreNum 来源）；`tests/aclnn/test_aclnn.cpp` + `run.sh`（ST 回归）；门禁脚本 `vf_reachability_check.py`、`non3510_scaffold_check.py`。
- kernel 侧调试依赖 `ASCENDC_DUMP=1` + kernel printf（输出自动带 `[AIV Block i/N]` 前缀）。

## 预防

- 手写多核任务均分分发时， checklist 第一条：`GetBlockIdx()` 之后立即 `if(cid>=uc) return;`；或分片并行度直接取 `AscendC::GetBlockNum()` 覆盖 tiling 的 numCores——不要信任 tiling 下发的 numCores 等于实际启动网格（图模式编译、aclnn 动态 `SetBlockDim`、staging 环境都可能与 tiling 预期不一致）。
- 评审 tiling 的 usedCoreNum 语义时确认"硬件启动核数"与"有效任务数"是两个量；B*HC（或 B*CD）可能为个位数而核数为几十。
- 回归集必须包含"任务数<核数 × 大 D"组合用例，否则越界会静默通过小规模回归。

## 验证信号

以下信号来自该算子 kernel 崩溃修复会话的同一套 23 用例 ST 回归（aclnn 直调，Ascend950PR 实机）：

- 修复前基线：23 用例中 4 个大 D+大 L 用例报 `FAILED (sync error 507035)`，其余小规模用例全部通过（来源平台：950；证据等级：tool_observed）。
- printf 探针观测到 `[AIV Block 28/56]` 等多余核执行 D=1024 主循环，而该用例任务总数仅 B*HC=4（来源平台：950；证据等级：tool_observed）。
- 两处 Process() 加入 `if(cid>=uc) return;` 后，同一套 23 用例 507035 全消除：原 4 个崩溃用例全部运行至完成（转为 bitwise 数值比对），16/23 通过（来源平台：950；证据等级：tool_observed）。
- 修复后 `git diff` 核对：kernel 仅 2 处 Process() 各 +3/-1 行，无其他改动；`vf_reachability_check.py`（15 个 `__simd_vf__` 定义、57 处调用点）与 `non3510_scaffold_check.py` 均 PASS（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2（BenchHarnessScatter，CANN 9.2.0，修复路线二）：修复前图模式（GEIR）8 用例全部 `RunGraph rc=255` EZ9999（aivec error 95 + MTE error info），aclnn 动态路径黑盒正向 399 例中 284 例失败（含本机制贡献，未隔离）；三处 `GetBlockNum()` 对齐 + 重建后，L2 反向回归 25 例形态符合预期（24 正确拒绝 + 1 已知误归类用例 PASS），正向冒烟（前 30 例，--pc=2）30/30 全 PASS、对照修复前基线有 18 条 FAIL→PASS（归因边界：基线采集早于同批黑盒修复变更，转好是「网格修复 + 黑盒修复」叠加结果，未做隔离归因；本卡可归因于网格修复的干净证据是机制算式与重建后无回归）（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：修复后图模式复测仍 8/8 失败（叠加 GEIR 编译缓存复用旧二进制的干扰，清缓存确认新二进制后仍失败）——图模式残余失败属另一层未解问题（TBE 动态编译路径/gridDim 烘焙），**网格对齐修复对图模式必要而不充分**，边界见同批 GEIR 图模式负结果记录（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本守卫只解决"多余核以越界任务号访问 GM"；tt≥核数但任务内部索引（如 l/K/dil/状态回看下标）越界的崩溃不适用，环形索引负偏移形态见 [507035 负模环形索引卡](507035_negative_modulo_ring_index_ub_offset.md)。
- "小 tt 用例因越界恰好落入合法 GM 区而未崩"是修复会话的分析推断（printf 只证明多余核在执行越界任务，未逐地址验证越界落点），不要据此认为小规模用例结果正确——它们可能已被越界写破坏。
- 其他 Ascend 平台（A2/A3 等）的全核启动语义未验证；若同样"启动全核、kernel 自行分发"，推断可迁移（未验证，推断可迁移）。
- R2 边界：本修复只解决「分片并行度与实际网格失配」这一越界机制；GEIR 图模式残余的 aivec 95/263 失败（TBE 动态编译/binary 选择层）未被本修复解决，网格修复后仍全败；若 tiling 的 `perBufBytes` 等 UB 预算与图模式编译配置不一致，分片循环仍可能出问题——本卡未覆盖。

**未决项**：

- 无（根因已由 printf 实证并修复闭环）。多余核越界写是否曾污染小 tt 用例的输出（静默错误）未做逐地址审计，如需可补 GM 越界探针。
## 坏实践

- 把 507035 先入为主归因于"PipeBarrier<PIPE_ALL> 数量过多导致硬件资源耗尽"并据此删 barrier/加大 chunk：本会话单变量实验证伪——仅移除 barrier 后 507035 依旧存在，反而引入精度回归（见同场景 VF barrier 相关记录）。
- 只在模板主体 Process() 加守卫、漏掉 Empty/辅助分支的第二个分发点。

## 误诊

- "UB 超额导致崩溃"：UB 预算检查在限额内（约 127KB / 192KB），排除。
- "barrier 次数随 L 增长到数十万导致资源耗尽"：受控实验（仅减 barrier 数）崩溃不消失，真正原因是越界访存而非同步数量。

## 增量（2026-09-17，先前增量 · plog 核号对比判别形态与并存根因边界，广播多核 RANK_4）

独立复现（softplus_v2_grad 广播范式多核 AIV kernel，dav-3510/ascend950 + CANN 9.1.0；版本覆盖从 9.2.0 扩到 9.1.0，工程形态为 broadcast 范式多核）：RANK_4 分支 kernel 首次跑 aclnn ST，`[1,1024]⊗[256,1024]` 广播用例在 `aclrtSynchronizeStream` 报 ACL error 507035（vector core exception）。查最新 plog（`~/ascend/log/debug/plog/`）发现设备侧异常登记：

```text
[ERROR] RUNTIME(...) [device_error_proc_c.cc:694] AddExceptionRegInfo:add error register: core_id=42, stream_id=61, task_sn=28
... core id is 42, error code = 82, 341 ...
[ERROR] ... GetArgsInfo:tilingKey = 0, ... blockDim=18
```

同批还有 core_id=43、44 的同类异常。AIC_INFO 明确 `blockDim=18`，而报异常核号 42/43/44 超出 BlockDim——与主卡"启动核数超过有效任务数"同机制，但本形态**事后 plog 即可判别**（异常登记里的 core_id 与 AIC_INFO 的 blockDim 直接对比），无须先加 printf 探针重建一轮。当时代码状态：kernel `Process()` 第一行直接进入事件 ID 获取与 `GetCoreRange(GetBlockIdx(), ...)` 计算，没有对 `multicore.num_cores` 的上界守卫。

修复（守卫依赖 tiling 数据 `multicore.num_cores` 正确下发、host `SetBlockDim` 用同一值；放在 Process() 最前——事件 ID 获取之前，越界核连事件 ID 都不应消耗；before/after verbatim）：

```cpp
    // before
    __aicore__ inline void Process()
    {
        // 事件 ID (broadcast/patterns.md sync)
```

```cpp
    // after
    __aicore__ inline void Process()
    {
        // Guard: cores beyond num_cores must return immediately (some platforms
        // launch all physical cores regardless of BlockDim).
        if (AscendC::GetBlockIdx() >= td_->multicore.num_cores) return;

        // 事件 ID (broadcast/patterns.md sync)
```

### 复现增量验证信号

- 修复前 plog：`AddExceptionRegInfo: core_id=42/43/44` 报 aivec error（`error code = 82, 341`），而 AIC_INFO `blockDim=18`，核号超出 BlockDim（来源平台：950；证据等级：tool_observed）。
- 加守卫重建后 plog：报错核号变为 core_id=0/1/2/3/6/7/8/9（均 < 18），越界核号异常消失（来源平台：950；证据等级：tool_observed）。
- 守卫保留在最终交付的 kernel.h Process 首行（来源平台：950；证据等级：tool_observed）。
- 该守卫单独不消除同批 C05 崩溃——BlockDim 之内的合法核仍因 tile 超 UB 报错，需叠加 host 切分轴修复（同一次 507035 输出里两个独立缺陷并存，分别修复、分别验证）（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 判别证据是修前/修后 plog 核号列表对比；"平台是否无条件启动全部物理核、core_id 到物理核如何映射"的机制层细节仍未做受控实验（与主卡遗留问题一致），本增量结论仅基于核号对比。
- 守卫只防"越界核号执行"，不防 num_cores 内核的越界访问（tile 超 UB、GM 越界）；tiling 未按 host `SetBlockDim` 同源填充 `num_cores` 时守卫失效。
- 其他平台（a2 等）是否同样启动 BlockDim 外的核未验证。