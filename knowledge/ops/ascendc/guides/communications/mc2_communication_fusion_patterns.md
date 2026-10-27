---
type: Guide
title: "MC2 Communication (compute-communication fusion) 域模式：HcclResource / CrossRankSync / Window RDMA / MakeGlobalTensor"
description: "MC2 多 rank 计算-通信融合 kernel 通信参考：HCCL context 生命周期、跨 rank 软屏障、window RDMA 与 GM 对齐标量 I/O（通信路径锁定 Path B1；内嵌 P-P110~P-P113 四个模式）。"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["a2", "a3"]
aliases: ["CrossCoreWaitFlag", "Window RDMA"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Domain: MC2 Communication (compute-communication fusion)

> 适用范围：本 HCCL window-RDMA 通信工具的 HcclResource 内部按 devType（HCCL_ASCEND910B / HCCL_ASCEND910C）分支，助手函数适用于 910B/910C（MVP scope per MC2_HARNESS_DESIGN §2.3）；a5(950PR) 路径 deferred（devType 分支已预留，未实测）——platforms 字段与正文以此为准。

## 适用场景

- 适用平台与范围：soc=all（卡内 applies_to 块）——HcclResource 内部按 devType 分支（HCCL_ASCEND910B/910C），helper 适用 910B/910C；MVP 范围为 a2(910B)/a3(910C)，a5(950PR) 推迟但 devType 分支已预留（卡内 reason 行）。
- 验证来源：卡内无 verified_on 行；模式来源与授权边界见正文 Provenance & inline rule 段（sync 原语内联自 ascendc-sync-guide skill 的 cross_rank_sync.hpp，HcclResource 适配自参考算子 util 头）。
- 内嵌模式索引：P-P110（HcclResource 生命周期 + rank/window 拓扑）、P-P111（CrossRankSync 跨 rank 软屏障 + GM 对齐标量 I/O + SyncAll 层间握手）、P-P112（Window RDMA 单侧 put/get 使用模式）、P-P113（MakeGlobalTensor 包装），逐字节保留于正文。

> Reference examples for MC2 multi-rank kernel communication: HCCL context lifecycle, cross-rank soft barrier, window RDMA, GM-aligned scalar I/O.
> Load when: opgen_mode == opgen_mc2 (multi-rank compute-comm fusion kernel); Analyzer detects `GetHcclContext` / `HcclResource` / `CrossRankSync` / window RDMA / a collective fused with compute.
> **Communication path (design §3.1)** — MC2 has two paths; MVP locks to **B1**:
> - **Path A** — high-level HCCL collective APIs (HcclAllReduce/HcclBroadcast/HcclSend/HcclRecv), HCCL server on AICPU. **Not in MVP** (no scaffold/KB wiring).
> - **Path B — custom shared GM**: window RDMA + CrossRankSync soft barrier.
>   - **B1** (MVP default): reuse the HCCL resource via `GetHcclContext`/`HcclResource`; the aclnn wrapper sets `NnopbaseSetHcclServerType(MTE)` (HCCL server on MTE + AICPU hccl offline) to pair with window RDMA. Path-A APIs CONFLICT with this MTE setting.
>   - **B2** (future): Host allocates/shares GM itself — no HCCL resource, no `NnopbaseSetHcclServerType`.
> The scaffold emits B1 unconditionally (comm-path is a harness scaffold choice, not per-op). Path A / B2 tracked for later.
> chip_scope: **all** (910B/910C via devType branch) — see frontmatter.

> **Provenance & inline rule**: the sync primitives (CrossRankSync / GetGmAlign / SetGmAlign / PipeSync / MakeGlobalTensor) are inlined from the `ascendc-sync-guide` skill (`references/inter-card-sync/cross_rank_sync.hpp`) — this is the canonical source and supersedes older copies; the HCCL-resource helpers (P-P110) are inlined from the reference op's util headers (owner-authorized, design §3.1). **Inline these helpers INTO your kernel** — the snippets are self-contained (no external include, no foreign namespace). Sync-primitive names (CrossRankSync / GetGmAlign / SetGmAlign / PipeSync / MakeGlobalTensor) match the sync-guide source verbatim (canonical bare forms — no `Mc2*` prefix); the P-P110 HcclResource is **adapted from** the reference op's util header — namespace self-contained (`mc2_detail` vs upstream `AscendDevUtil::detail`) and hardened with `ascendc_assert` rankId/addr guards absent from the source. All helpers are self-contained (no external include, no foreign namespace to clash with). Copy the helper scaffolding; author your own fusion logic.

---## Patterns

> **Three-layer sync map** — ascendc-sync-guide (design §5 Layer 3). An MC2 `KERNEL_TYPE_MIX_AIC_1_2` kernel coordinates sync at three layers; P-P111 below covers all three:
>
> | layer | scope | API | detail cross-ref |
> |---|---|---|---|
> | **intra-core** (pipe) | one core, MTE2↔V↔MTE3↔S pipes | `SetFlag<HardEvent>` / `WaitFlag` / `PipeSync` (P-P111) | OL-94 (TQue vs TBuf), P-P26 / P-P75 (event-id reuse), P-P95 (A5 `LocalMemBar`), PB-21 (PipeBarrier+TBuf crash) |
> | **inter-core** (AIC↔AIV) | MIX cube ↔ vector cores | `SyncAll<true/false>` count barrier — no flagId (P-P111) | OL-190 (group barrier symmetric), OL-213 (SyncAll→PipeBarrier hang), `fa_class/cross_core_sync.md` §4 (`CrossCoreSetFlag` flag-based alt — FA-class), PB-34 / PB-35 (manual CrossCore deadlock) |
> | **cross-rank** (card) | NPUs in HCCL group | `CrossRankSyncInRandomMemValue` (P-P111) + window RDMA (P-P112) + HcclResource (P-P110) | `atlasascendc_api_07_0867.md` (`Hccl::InitV2` high-level alt) |
>
> Inter-core (`SyncAll`) and cross-rank (`CrossRankSync*`) are **decoupled** layers — see the inter-core note in P-P111.

### P-P110: MC2 HcclResource — HCCL context lifecycle + rank/window topology

**Severity**: High | **Source**: reference op HCCL-resource util header (HcclResource, owner-authorized, design §3.1).

The single entry point for MC2 rank-private state. `AscendC::GetHcclContext<HCCL_GROUP_ID_0>()` returns a `__gm__` pointer whose layout depends on chip (`HcclCombineOpParam` for 910B, reinterpreted as `HcclOpResParam` for 910C). `HcclResource::Init` parses `devType` from the MC2 init tiling and exposes rank id / size / workspace / per-peer window addresses. All other MC2 helpers depend on it.

```cpp
#include <type_traits>   // is_integral_v only (GetGmAlign/SetGmAlign). Do NOT use declval-dependent
                         // traits (is_invocable_r_v etc.) — they fail template matching under __aicore__.

namespace mc2_detail {

constexpr uint32_t HCCL_MAX_RANK_NUM = 32U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;

struct MemDetails {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
};

struct IbVerbsData {
    MemDetails remoteInput;
    MemDetails remoteOutput;
    MemDetails localInput;
    MemDetails localOutput;
    uint8_t res[24];
};

// Layout returned by GetHcclContext<HCCL_GROUP_ID_0>() on 910B (and multi-node).
struct HcclCombineOpParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
    uint8_t  res[8064];
    uint64_t winExpSize;
    uint64_t windowsExp[HCCL_MAX_RANK_NUM];
    uint8_t  multiFlag;
    __gm__ IbVerbsData *data;
};

// 910C layout: reinterpret_cast<HcclCombineOpParam*> as this when devType != 910B.
struct HcclOpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char     hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
};

struct RemoteResPtr {
    HcclRankRelationResV2 *nextHostPtr;
    HcclRankRelationResV2 *nextDevicePtr;
};

// Shape of the mc2InitTiling GM payload the host passes in.
struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t  debugMode;
    uint8_t  preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t  devType;
    char     reserved[17];
};

} // namespace mc2_detail

enum HcclResourceType {
    HCCL_ASCEND910B = 1,
    HCCL_ASCEND910C
};

class HcclResource {
public:
    // mc2InitTiling may be nullptr -> default to 910B.
    __aicore__
    void Init(__gm__ void *mc2InitTiling) {
        if (mc2InitTiling != nullptr) {
            auto p = reinterpret_cast<__gm__ mc2_detail::Mc2InitTilingInner *>(mc2InitTiling);
            devType_ = p->devType;
        } else {
            devType_ = HCCL_ASCEND910B;
        }
        hcclContext_ = (__gm__ mc2_detail::HcclCombineOpParam *)
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
    }

    __aicore__ uint32_t GetRankId()          { return hcclContext_->rankId; }
    __aicore__ uint32_t GetRankSize()        { return hcclContext_->rankNum; }
    __aicore__ HcclResourceType GetDevType() { return static_cast<HcclResourceType>(devType_); }
    __aicore__ uint32_t IsA2MultiNode()      { return devType_ == HCCL_ASCEND910B && hcclContext_->multiFlag == 1U; }
    __aicore__ uint64_t GetWorkspaceSize()   { return hcclContext_->workSpaceSize; }

    __aicore__ uint64_t GetWindowsExpSize() {
        if (devType_ != HCCL_ASCEND910B) {
            auto p = reinterpret_cast<__gm__ mc2_detail::HcclOpResParam *>(hcclContext_);
            return p->winExpSize;
        }
        return hcclContext_->winExpSize;
    }

    __aicore__ uint64_t GetWindowsSize() {
        if (devType_ != HCCL_ASCEND910B) {
            auto p = reinterpret_cast<__gm__ mc2_detail::HcclOpResParam *>(hcclContext_);
            return p->winSize;
        }
        return hcclContext_->winSize;
    }

    __aicore__ GM_ADDR GetWorkspaceAddr() {
        uint64_t addr = hcclContext_->workSpace;
        ascendc_assert(addr != 0UL, "GetWorkspaceAddr failed, addr is nullptr");
        return reinterpret_cast<GM_ADDR>(addr);
    }

    __aicore__ GM_ADDR GetWindowsExp(uint32_t rankId) {
        ascendc_assert(rankId < GetRankSize(),
            "GetWindowsExp failed, rankId[%u], expected less than[%u]", rankId, GetRankSize());
        uint64_t addr = 0UL;
        if (devType_ != HCCL_ASCEND910B) {
            auto p = reinterpret_cast<__gm__ mc2_detail::HcclOpResParam *>(hcclContext_);
            if (rankId == p->rankId) {
                addr = p->localWindowsExp;
            } else {
                auto r = GetRemoteRankAddrs(p, rankId);
                addr = (r != nullptr ? r->windowsExp : 0UL);
            }
        } else if (hcclContext_->multiFlag == 0U) {
            addr = hcclContext_->windowsExp[rankId];
        } else {
            ascendc_assert(false, "GetWindowsExp failed, cannot get exp addr of rankId[%u]", rankId);
        }
        ascendc_assert(addr != 0UL, "GetWindowsExp failed, addr of rankId[%u] is nullptr", rankId);
        return reinterpret_cast<GM_ADDR>(addr);
    }

    // windowsIn addr for rankId. NOTE: "In"/"Out" are just two symmetric memory
    // slabs — naming carries NO send/recv direction (see P-P112). Either slab works
    // as long as sender and receiver use the same one.
    __aicore__ GM_ADDR GetWindowsInAddr(uint32_t rankId) {
        ascendc_assert(rankId < GetRankSize(),
            "GetWindowsInAddr failed, rankId[%u], expected less than[%u]", rankId, GetRankSize());
        uint64_t addr = 0UL;
        if (devType_ != HCCL_ASCEND910B) {
            auto p = reinterpret_cast<__gm__ mc2_detail::HcclOpResParam *>(hcclContext_);
            if (rankId == p->rankId) {
                addr = p->localWindowsIn;
            } else {
                auto r = GetRemoteRankAddrs(p, rankId);
                addr = (r != nullptr ? r->windowsIn : 0UL);
            }
        } else if (hcclContext_->multiFlag == 0U) {
            addr = hcclContext_->windowsIn[rankId];
        } else {
            if (rankId == hcclContext_->rankId) {
                addr = hcclContext_->data[rankId].localInput.addr;
            } else {
                addr = hcclContext_->data[rankId].remoteInput.addr;
            }
        }
        ascendc_assert(addr != 0UL, "GetWindowsInAddr failed, addr of rankId[%u] is nullptr", rankId);
        return reinterpret_cast<GM_ADDR>(addr);
    }

    // windowsOut addr for rankId — the other symmetric slab (see P-P112); equivalent
    // to windowsIn for direction purposes (both are direction-agnostic).
    __aicore__ GM_ADDR GetWindowsOutAddr(uint32_t rankId) {
        ascendc_assert(rankId < GetRankSize(),
            "GetWindowsOutAddr failed, rankId[%u], expected less than[%u]", rankId, GetRankSize());
        uint64_t addr = 0UL;
        if (devType_ != HCCL_ASCEND910B) {
            auto p = reinterpret_cast<__gm__ mc2_detail::HcclOpResParam *>(hcclContext_);
            if (rankId == p->rankId) {
                addr = p->localWindowsOut;
            } else {
                auto r = GetRemoteRankAddrs(p, rankId);
                addr = (r != nullptr ? r->windowsOut : 0UL);
            }
        } else if (hcclContext_->multiFlag == 0U) {
            addr = hcclContext_->windowsOut[rankId];
        } else {
            if (rankId == hcclContext_->rankId) {
                addr = hcclContext_->data[rankId].localOutput.addr;
            } else {
                addr = hcclContext_->data[rankId].remoteOutput.addr;
            }
        }
        ascendc_assert(addr != 0UL, "GetWindowsOutAddr failed, addr of rankId[%u] is nullptr", rankId);
        return reinterpret_cast<GM_ADDR>(addr);
    }

    // 910C only: remote-rank addr array at ctx + rWinStart.
    __aicore__
    const mc2_detail::HcclRankRelationResV2* GetRemoteRankAddrs(
        __gm__ mc2_detail::HcclOpResParam *ctx, uint32_t rankId)
    {
        const mc2_detail::RemoteResPtr *remoteRes = reinterpret_cast<const mc2_detail::RemoteResPtr *>(
            reinterpret_cast<uintptr_t>(ctx) + ctx->rWinStart);
        return remoteRes[rankId].nextDevicePtr;
    }

public:
    __gm__ mc2_detail::HcclCombineOpParam *hcclContext_;
    uint8_t devType_;
};
```

---

### P-P111: MC2 CrossRankSync — cross-rank soft barrier + GM-aligned scalar I/O

**Severity**: High | **Source**: `ascendc-sync-guide` skill `references/inter-card-sync/cross_rank_sync.hpp` (owner-authorized; canonical source, supersedes older copies — design §3.1).

A software barrier for custom MC2 topologies (not a collective API). Every participating rank `atomicAdd(1)` into a GM flag slot, then polls until the observed value advances by `rankSize`. Includes `GetGmAlign`/`SetGmAlign` (single-element GM read/write honoring native b8/b16/b32 alignment), `PipeSync` (intra-core event flag/wait), and the inter-core `SyncAll` handshake pattern (AIC↔AIV in MIX_AIC_1_2 — see end of this entry). Depends on `MakeGlobalTensor` (P-P113).

```cpp
#include <type_traits>   // is_integral_v only (GetGmAlign/SetGmAlign). Do NOT use declval-dependent
                         // traits (is_invocable_r_v etc.) — they fail template matching under __aicore__.

// pipe-internal sync via event flag/wait.
template <AscendC::HardEvent EVENT>
__aicore__ inline void PipeSync() {
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID<EVENT>();
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}
template <AscendC::HardEvent EVENT>
__aicore__ inline void PipeSync(AscendC::TEventID eventId) {
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}
// PipeSync uses FetchEventID: non-loop immediate Set→Wait only (FetchEventID does not lock the
// bitmap bit). eventID rules (sync-guide intra-core-sync): obtain via FetchEventID /
// AllocEventID+ReleaseEventID only — never hard-code values (clash with framework-internal events);
// Set/Wait is a binary flag (NOT a counter), pair 1:1 (two consecutive Set without Wait = undefined);
// loop reuse of one eventID needs AllocEventID outside the loop + ReleaseEventID after.

#define COMPILER_OPTIMIZATION_BARRIER() __asm__ __volatile__("")
constexpr uint64_t SOFT_SYNC_ALIGN_BYTE     = 512;          // flag slot stride in GM
constexpr uint64_t SOFT_SYNC_TIMEOUT_CYCLES = 1500000000;   // ~30s @ 50MHz
constexpr uint64_t POLL_BACKOFF_CYCLES      = 10;           // ~200ns @ 50MHz, backs off the poll to ease bus contention

// busy-wait `cycles` to back off a spin poll (prevents the poll from saturating the bus).
__aicore__
inline void PollBackoff(uint64_t cycles)
{
    uint64_t start = (uint64_t)AscendC::GetSystemCycle();
    while ((uint64_t)AscendC::GetSystemCycle() - start < cycles) {
        COMPILER_OPTIMIZATION_BARRIER();
    }
}

// read one GM element through UB, honoring native align.
template <typename T>
__aicore__ inline
std::enable_if_t<std::is_integral_v<T>, T>
GetGmAlign(const AscendC::GlobalTensor<T> &gm, __ubuf__ void *ubTmpAddr,
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_S>())
{
    auto gmAddr = gm.GetPhyAddr(0);
    if constexpr (sizeof(T) == AscendC::B8_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b8 (ubTmpAddr, gmAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == AscendC::B16_BYTE_SIZE) {
        copy_gm_to_ubuf_align_b16(ubTmpAddr, gmAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else {
        copy_gm_to_ubuf_align_b32(ubTmpAddr, gmAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    }
    PipeSync<AscendC::HardEvent::MTE2_S>(eventId);
    return *reinterpret_cast<__ubuf__ T *>(ubTmpAddr);
}

// write one GM element through UB (S_MTE3 ordered).
template <typename T>
__aicore__ inline
std::enable_if_t<std::is_integral_v<T>, void>
SetGmAlign(const AscendC::GlobalTensor<T> &gm, __ubuf__ void *ubTmpAddr, T data,
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID<AscendC::HardEvent::S_MTE3>())
{
    auto gmAddr = gm.GetPhyAddr(0);
    *reinterpret_cast<__ubuf__ T *>(ubTmpAddr) = data;
    PipeSync<AscendC::HardEvent::S_MTE3>(eventId);
    if constexpr (sizeof(T) == AscendC::B8_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b8 (gmAddr, ubTmpAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == AscendC::B16_BYTE_SIZE) {
        copy_ubuf_to_gm_align_b16(gmAddr, ubTmpAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else {
        copy_ubuf_to_gm_align_b32(gmAddr, ubTmpAddr, 0, 1, sizeof(T), 0, 0, 0, 0);
    }
}

// barrier: every rank atomicAdd(1) to flagGlobal, then poll until delta >= rankSize.
// Only block-0/subblock-0 participates. ubTmpAddr = scratch UB for the 4-byte flag.
// Poll is timeout-guarded: on SOFT_SYNC_TIMEOUT_CYCLES it printf + Trap() (never hangs);
// PollBackoff between reads eases bus contention.
__aicore__ inline
void CrossRankSyncInRandomMemValue(GM_ADDR gmAddr, __ubuf__ void *ubTmpAddr,
    uint32_t rankSize, uint32_t flagId)
{
    if (get_block_idx() == 0 && get_subblockid() == 0) {
        auto flagGlobal = MakeGlobalTensor<uint32_t>(gmAddr + flagId * SOFT_SYNC_ALIGN_BYTE);
        auto eventId = GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_S>();

        // Step1: read old value
        uint32_t oldFlagData = GetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, eventId);
        COMPILER_OPTIMIZATION_BARRIER();

        // Step2: atomic add 1 to notify other ranks
        AscendC::SetAtomicAdd<int32_t>();
        SetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, 1U);
        PipeSync<AscendC::HardEvent::MTE3_S>();
        AscendC::SetAtomicNone();
        COMPILER_OPTIMIZATION_BARRIER();

        // Step3: poll until all ranks arrive (timeout-guarded)
        uint64_t syncStartCycle = (uint64_t)AscendC::GetSystemCycle();
        uint32_t newFlagData;
        while (true) {
            newFlagData = GetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, eventId);
            if (newFlagData - oldFlagData >= rankSize) {
                break;
            } else if ((uint64_t)AscendC::GetSystemCycle() - syncStartCycle > SOFT_SYNC_TIMEOUT_CYCLES) {
                AscendC::printf(
                    "CrossRankSyncInRandomMemValue TIMEOUT!\n%s:%u\nrankSize:%u flagData:%u expectedVal:%u\n",
                    __FILE__, __LINE__, rankSize, newFlagData - oldFlagData, rankSize);
                AscendC::Trap();
            } else {
                PollBackoff(POLL_BACKOFF_CYCLES);
            }
            COMPILER_OPTIMIZATION_BARRIER();
        }
        COMPILER_OPTIMIZATION_BARRIER();

        // Step4: atomic add 1 to release barrier
        AscendC::SetAtomicAdd<int32_t>();
        SetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, 1U);
        PipeSync<AscendC::HardEvent::MTE3_S>();
        AscendC::SetAtomicNone();
        COMPILER_OPTIMIZATION_BARRIER();
    }
}

// deterministic variant: each aicore writes its OWN per-rank flag slot; owning rank polls for expectedVal.
// Replace the GetRankSyncAddr placeholder with your own per-rank sync-GM address lookup.
// Poll is timeout-guarded (printf + Trap on timeout).
__aicore__ inline
void CrossRankSyncInDeterministicMemValue(__ubuf__ void *ubTmpAddr, uint32_t rankSize,
    uint32_t rankId, uint32_t flagId, uint32_t flagValue, uint32_t expectedVal)
{
    uint32_t aicoreId = get_block_idx();
    uint32_t subAivId = get_subblockid();

    if (subAivId == 0 && aicoreId < rankSize) {
        GM_ADDR syncAddr = /* GetRankSyncAddr(aicoreId) */ 0;
        auto flagGlobal = MakeGlobalTensor<uint32_t>(
            syncAddr + flagId * SOFT_SYNC_ALIGN_BYTE);
        AscendC::SetAtomicAdd<int32_t>();
        SetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, flagValue);
        PipeSync<AscendC::HardEvent::MTE3_S>();
        AscendC::SetAtomicNone();
    }

    if (subAivId == 0 && aicoreId == rankId) {
        GM_ADDR syncAddr = /* GetRankSyncAddr(rankId) */ 0;
        auto flagGlobal = MakeGlobalTensor<uint32_t>(
            syncAddr + flagId * SOFT_SYNC_ALIGN_BYTE, false);
        auto eventId = GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_S>();
        uint64_t syncStartCycle = (uint64_t)AscendC::GetSystemCycle();
        uint32_t flagData;
        while (true) {
            flagData = GetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, eventId);
            if (flagData == expectedVal) {
                break;
            } else if ((uint64_t)AscendC::GetSystemCycle() - syncStartCycle > SOFT_SYNC_TIMEOUT_CYCLES) {
                AscendC::printf("CrossRankSyncInDeterministicMemValue TIMEOUT!\n%s:%u\nrankSize:%u flagData:%u expectedVal:%u\n",
                    __FILE__, __LINE__, rankSize, flagData, flagValue * rankSize);
                AscendC::Trap();
            } else {
                PollBackoff(POLL_BACKOFF_CYCLES);
            }
            COMPILER_OPTIMIZATION_BARRIER();
        }
    }
}

// clear-after variant: same as Deterministic, but the owning rank resets the flag to 0 after the
// barrier, so the same flag slot can be reused across iterations.
__aicore__ inline
void CrossRankSyncInDeterministicMemValueWithClear(__ubuf__ void *ubTmpAddr, uint32_t rankSize,
    uint32_t rankId, uint32_t flagId, uint32_t flagValue)
{
    uint32_t aicoreId = get_block_idx();
    uint32_t subAivId = get_subblockid();

    if (subAivId == 0 && aicoreId < rankSize) {
        GM_ADDR syncAddr = /* GetRankSyncAddr(aicoreId) */ 0;
        auto flagGlobal = MakeGlobalTensor<uint32_t>(
            syncAddr + flagId * SOFT_SYNC_ALIGN_BYTE);
        AscendC::SetAtomicAdd<int32_t>();
        SetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, flagValue);
        PipeSync<AscendC::HardEvent::MTE3_S>();
        AscendC::SetAtomicNone();
    }

    if (subAivId == 0 && aicoreId == rankId) {
        GM_ADDR syncAddr = /* GetRankSyncAddr(rankId) */ 0;
        auto flagGlobal = MakeGlobalTensor<uint32_t>(
            syncAddr + flagId * SOFT_SYNC_ALIGN_BYTE, false);
        auto eventId = GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_S>();
        uint64_t syncStartCycle = (uint64_t)AscendC::GetSystemCycle();
        uint32_t flagData;
        while (true) {
            flagData = GetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, eventId);
            if (flagData == flagValue * rankSize) {
                break;
            } else if ((uint64_t)AscendC::GetSystemCycle() - syncStartCycle > SOFT_SYNC_TIMEOUT_CYCLES) {
                AscendC::printf("CrossRankSyncInDeterministicMemValueWithClear TIMEOUT!\n%s:%u\nrankSize:%u flagData:%u expectedVal:%u\n",
                    __FILE__, __LINE__, rankSize, flagData, flagValue * rankSize);
                AscendC::Trap();
            } else {
                PollBackoff(POLL_BACKOFF_CYCLES);
            }
            COMPILER_OPTIMIZATION_BARRIER();
        }
        COMPILER_OPTIMIZATION_BARRIER();

        SetGmAlign<uint32_t>(flagGlobal, ubTmpAddr, 0);
    }
}
```

#### Inter-core sync (AIC ↔ AIV) — `SyncAll` count-based barrier

A `KERNEL_TYPE_MIX_AIC_1_2` kernel runs cube (AIC) + vector/comm (AIV) cores in one binary; the AIC↔AIV handshake uses `AscendC::SyncAll` — **not** `CrossCoreSetFlag`/flagId (the latter is FA-class, see `fa_class/cross_core_sync.md` §4):

- `SyncAll<true>` = AIV-only barrier (AIC does NOT participate — safe when only some AIV blocks reach it).
- `SyncAll<false>` = AIC + AIV all-block barrier (the AIC↔AIV handshake).
- No `flagId`: pairs match by **ordered count** (AIC's Nth `SyncAll<false>` ↔ AIV's Nth `SyncAll<false>`).

Branch via parallel `if ASCEND_IS_AIV { }` / `if ASCEND_IS_AIC { }` (NOT if/else — the compiler prunes per core type):

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if ASCEND_IS_AIV {
    /* quant + dispatch (cross-rank AllToAll via CrossRankSyncInRandomMemValue) */
    AscendC::SyncAll<false>();   // #1 notify AIC: dispatch done
    AscendC::SyncAll<false>();   // #2 wait AIC: GMM done
    /* dequant + swiglu + final quant (AIV-only -> SyncAll<true>) */
}
if ASCEND_IS_AIC {
    AscendC::SyncAll<false>();   // #1 wait AIV: dispatch done
    GmmKernel(gmmParams);        // cube
    AscendC::SyncAll<false>();   // #2 notify AIV: GMM done
}
```

**⚠️ 死锁头号原因 — `SyncAll` 调用次数必须全核匹配**（sync-guide inter-core-sync）：`SyncAll<false>()` 是全 AIC+AIV 屏障，`SyncAll<true>()` 是全 AIV 屏障 —— **所有参与核必须在相同位置、相同次数调用**，任何核少调/多调一次 → 永久死锁。`ASCEND_IS_AIC`/`ASCEND_IS_AIV` 静态分核是最易踩坑场景（AIC、AIV 走完全不同分支，AIV 侧加了同步却忘在 AIC 侧补空同步）：

```cpp
// ❌ AIV 2 次 SyncAll<false>，AIC 只 1 次 → 永久死锁
if ASCEND_IS_AIC {
    RunGMM();
    AscendC::SyncAll<false>();          // AIC #1
}
if ASCEND_IS_AIV {
    RunDequant();
    AscendC::SyncAll<false>();          // AIV #1 ✓
    RunSwiGLU();
    AscendC::SyncAll<false>();          // AIV #2 ✗ AIC 无对应第 2 次 → 死锁
}
// ✅ AIC 侧补空 SyncAll 匹配次数（即使无数据操作也必须调用）
if ASCEND_IS_AIC {
    RunGMM();
    AscendC::SyncAll<false>();          // AIC #1
    AscendC::SyncAll<false>();          // AIC #2（匹配 AIV，空操作也要调）
}
if ASCEND_IS_AIV {
    RunDequant();
    AscendC::SyncAll<false>();          // AIV #1
    RunSwiGLU();
    AscendC::SyncAll<false>();          // AIV #2
}
```

**检视**：数 AIC 分支与 AIV 分支的 `SyncAll<false>()` 次数，必须完全一致（建议注释标序号 `// SyncAll #1 of N`）；`SyncAll` 不可放在 `if`/`for` 条件分支内（除非该条件在所有核上求值相同）；`blockDim` 必须 ≤ 实际 AI 处理器核数，否则多轮调度插入异常同步卡死。详见 sync-guide inter-core-sync.md。

**Decoupled from cross-rank**: during cross-rank AllToAll (AIV block 0), AIC blocks idle at their own `SyncAll<false>` wait — AIC never joins cross-rank communication. The two layers do not nest.

> **Choosing the inter-core mechanism**: `SyncAll` (count-based, used here) vs `CrossCoreSetFlag<0x2,PIPE>(flagId)`/`CrossCoreWaitFlag` (flag-based, FA-class) — pick per the op. See OL-190 (cross-core group barrier symmetric participation), OL-213 (`SyncAll`→`PipeBarrier` multi-core hang #1 cause), PB-34 (manual `CrossCore` V220 deadlock), PB-35 (`event_t` 0..3 vs cross-core flag-id conflict).

---

### P-P112: MC2 Window RDMA — one-sided put/get communication pattern

**Severity**: Medium | **Source**: reference op HCCL-resource util header (window topology, owner-authorized, design §3.1).

The window addresses from P-P110 (`GetWindowsInAddr` / `GetWindowsOutAddr`) are RDMA-registered GM regions pre-allocated per rank-pair by the HCCS framework. No standalone code — this entry is the *usage pattern* for one-sided communication, building on P-P110's methods.

```cpp
// windowsIn / windowsOut are two INDEPENDENT symmetric slabs — "In"/"Out" carry NO
// send/recv direction. Use either slab; the only rule is sender and receiver point at
// the SAME slab. Examples below use windowsIn throughout.
//
// 两种收发模式（效果相同，优先模式 B）—— 区别在 self rank vs peer rank，不在 In/Out:
//   模式 A: 发端写 GetWindowsInAddr(selfRank)，收端读 GetWindowsInAddr(peerRank)
//   模式 B: 发端写 GetWindowsInAddr(peerRank)，收端读 GetWindowsInAddr(selfRank)

// 发端（写完用软同步通知对端）:
auto win = MakeGlobalTensor<T>(hccl.GetWindowsInAddr(/* selfRank(模式A) | peerRank(模式B) */), winSize);
// ... compute into win via AscendC DataCopy / VEC ops ...
CrossRankSyncInRandomMemValue(syncGm, ubScratch, rankSize, flagId);  // signal peers data ready

// 收端（先等软同步，再读）:
CrossRankSyncInRandomMemValue(syncGm, ubScratch, rankSize, flagId);  // wait for peer to have written
auto win = MakeGlobalTensor<T>(hccl.GetWindowsInAddr(/* peerRank(模式A) | selfRank(模式B) */), winSize);
// ... read win ...

// ❌ 反例（永久 hang）: 发端 GetWindowsOutAddr + 收端 GetWindowsInAddr —— 写读落在两片不同
//    内存，收端永远读不到，DataCopy 永久阻塞。根因是收发不对称，不是"用了 windowsOut"。

// Multi-node path (hccl.IsA2MultiNode()==1): per-rank address comes from
// hcclContext_->data[rankId].{remoteInput,remoteOutput,localInput,localOutput}.addr (IbVerbsData,
// RDMA verbs payload) instead of the in-process windowsIn/Out[] arrays. GetWindowsInAddr /
// GetWindowsOutAddr handle this branch internally — caller is agnostic.
```

All window addresses return as `GM_ADDR` (raw `__gm__ void*`); wrap with `MakeGlobalTensor<T>` (P-P113) to obtain an `AscendC::GlobalTensor<T>` for `DataCopy` / `SetAtomicAdd` / `GetPhyAddr`.

---

### P-P113: MC2 MakeGlobalTensor — GM_ADDR → GlobalTensor\<T\> wrapper

**Severity**: Low | **Source**: `ascendc-sync-guide` skill `references/inter-card-sync/cross_rank_sync.hpp` (MakeGlobalTensor, owner-authorized; uses only `AscendC::GlobalTensor` public API).

The unified entry for wrapping any raw `GM_ADDR` (window address, workspace, flag slot) into an `AscendC::GlobalTensor<T>` for DataCopy / VEC / atomic ops. Depended on by P-P111 (CrossRankSync flag slots) and P-P112 (window tensors).

```cpp
// size-bearing overload (preferred for window/workspace tensors).
template <typename T>
__aicore__ inline
AscendC::GlobalTensor<T> MakeGlobalTensor(__gm__ void *addr, uint64_t size,
    bool enableL2CacheHint = true)
{
    AscendC::GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addr), size);
    if (!enableL2CacheHint) {
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    return global;
}

// size-zero overload (caller tracks size externally, e.g. flag slots).
template <typename T>
__aicore__ inline
AscendC::GlobalTensor<T> MakeGlobalTensor(__gm__ void *addr, bool enableL2CacheHint = true)
{
    return MakeGlobalTensor<T>(addr, 0, enableL2CacheHint);
}
```

**Dependency order** when inlining all four: P-P113 (MakeGlobalTensor) → P-P110 (HcclResource) → P-P111 (CrossRankSync) → P-P112 (window RDMA pattern). Cross-refs: P-P110 ↔ P-P112 (window addr methods + usage); P-P111 ↔ P-P113 (flag-slot wrapping).
