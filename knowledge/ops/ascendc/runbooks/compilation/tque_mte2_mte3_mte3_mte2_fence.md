---
type: "Runbook"
title: "TQue MTE2→MTE3 缺 MTE3_MTE2 fence 致回读错位"
description: "rms_norm_rope_k_store_kv_cache Phase2 回读用 TQue<VECIN> 走 MTE2→MTE3 跨迭代复用同一 UB，下一轮 MTE2 覆盖 UB 时上一轮 MTE3 未读完形成 WAR，大数据(max_seq_len=65536) kv_a_out 前段错位、后段正确；判别靠 debug_full.py 逐 batch front/back match + 单核(blockDim=1)排除多核；根因为 TQue event 在该跨迭代路径不可靠且跨 Phase GM 可见性需 Host 同步。R2（CANN 9.2.0）证明纯 TBuf 分块循环只写单向 MTE2_MTE3 同样整块错位，补 MTE3_MTE2 即修复。"
tags: [task.debugging, topic.implementation.pipeline, topic.implementation.synchronization]
status: stable
sources: []
created_at: "2026-08-18T07:57:42Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# TQue MTE2→MTE3 缺 MTE3_MTE2 fence 致回读错位

## 现象

在 `rms_norm_rope_k_store_kv_cache`（MLA Prefill 融合算子，dav-3510 / Ascend950PR V100 / CANN 9.1.0）RMSNorm/RoPE 修复后，小数据（`max_seq_len=1024`，bs=4）全部通过，但 UT 用 `max_seq_len=65536`（大数据）时 `kv_a_out` 前段（Phase 2 回读区段）不匹配、后段（Phase 1 新算）正确。`debug_full.py` 对每个 batch 输出 `front match=False, back match=True`。

## 适用场景

- 工程形态：kernel 直调（AscendC `TBuf` + 显式 `HardEvent` fence + Host 双 launch + `aclrtSynchronizeStream`）；以及算子仓 kernel 的"GM 张量按块中转 UB"分块搬运循环（独立复现 R2 形态）。
- API 路径：`TQue<VECIN>` / `TBuf<TPosition::VECCALC>` / `GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3|MTE3_MTE2)` / `SetFlag`/`WaitFlag` / `DataCopyPad` / `aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend950（dav-3510，Ascend950PR V100）/ CANN 9.1.0；独立复现 R2：Ascend950PR（arch35）/ CANN 9.2.0。
- 适用粒度：`operator_family`（所有"MTE2 读 GM → MTE3 写 GM"跨迭代复用同一 UB 的二阶段/回读/分块搬运算子）。
- 路线优先级：`recommended_fastpath`（跨流水线 WAR 必须显式 fence；跨 Phase GM 可见性必须 Host 同步）。
- 不适用或未验证场景：单 Phase 算子无回读；单块搬运（数据量 ≤ perBufBytes/sizeof(T)）不触发；a2/a3 未验证，推断可迁移（`HardEvent`/`aclrtSynchronizeStream` 为跨架构通用机制）。

## 判别方法

触发（tool_observed）：大数据（`max_seq_len=65536`）UT 下 `kv_a_out` 前段（Phase 2 回读区段）不匹配、后段（Phase 1 新算）正确，`debug_full.py` 逐 batch 输出 `front match=False, back match=True`；小数据（`max_seq_len=1024`）全部通过。

探针：① 跑 `debug_full.py`，对每个 batch 分别报告 `front match` / `back match`，定位错位只在回读区段且只在大数据暴露；② 把 kernel launch 的 `blockDim` 收为 1（单核），重跑同一大数据 UT——前段仍 `match=False`，排除"多核竞争写同一 GM 地址"假说；③ 检查 `ProcessCache` 回读 buffer 是否用 `TQue<VECIN>` 依赖 `EnQue/DeQue` 自动同步、且回读与下一轮搬入复用同一 UB 区。

观测（tool_observed）：大数据下逐 batch `front match=False, back match=True`；单核 `blockDim=1` 仍复现前段错位（非多核问题）；问题只在数据量大、多轮 UB 复用时暴露，小数据不触发。

解释（interpretation，agent 根因分析，非 API 规范主张）：`ProcessCache` 用 `TQue<VECIN>` 做 cache 回读 buffer，`EnQue/DeQue` 的 event 管理在 MTE2→MTE3 跨迭代路径上不可靠——下一轮 MTE2 把新数据搬进同一 UB 区时，上一轮 MTE3（`DataCopyPad` 写 GM）可能还没读完该 UB 区，形成 WAR 竞争，导致回读区段错位；叠加跨 Phase GM 可见性问题（`Process` 写 paged cache、`ProcessCache` 读 paged cache，原在同 kernel 内靠 `pipe_->Reset()`+`SyncAll()` 切换，大数据时跨核 GM 写对读不可见）。修复方向（已验证 3/3 UT 通过）为改 `TBuf` + 显式 `HardEvent::MTE2_MTE3`（MTE2 搬完才让 MTE3 写）+ `HardEvent::MTE3_MTE2`（MTE3 读完 UB 才让下一轮 MTE2 覆盖）双向 fence 消除 WAR，并把 `Process`/`ProcessCache` 拆为两个独立 kernel entry、Host 两次 launch 之间 `aclrtSynchronizeStream` 确保跨 Phase GM 可见性。

### 独立复现 R2：纯 TBuf 分块循环只写单向 MTE2_MTE3 同样整块错位（CANN 9.2.0）

在 BenchHarnessScatter（scatter / scatter-reduce）黑盒修复中，`CopyDataToY` 的 GM→UB→GM 分块循环复用同一 `buf_[B0]`：每块先 `DataCopyPad(B0, gmData[...])`（MTE2），`SetFlag/WaitFlag<MTE2_MTE3>` 后 `DataCopyPad(gmY[...], B0)`（MTE3），**块与块之间没有任何 MTE3→MTE2 的依赖声明**——与 TQue 自动 event 不可靠是同一 WAR 的两种触发形态。判别探针（可复用）：

1. 用 TTK result.csv 做"单块/多块拷贝"相关性矩阵：`single-chunk copy: PASS=36 FAIL=0`、`multi-chunk copy: PASS=78 FAIL=285`——失败与分块强相关即命中本根因；
2. 纯拷贝复现（不含 scatter 语义：int32 data 100000 元素、indices/updates 为空、UB 每 buffer 87360 字节 → maxChunk=21840、共 5 块）：失配 78160/100000，**y[0]=21840 恰为 maxChunk（块 1 首元素）**——"每个块 k 的区间落了块 k+1 的内容"，最后一块无后继故正确（78160 ≈ 前 4 块）。

修复前（`CopyDataToY`，块间无 MTE3→MTE2 依赖，verbatim）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::CopyDataToY()
{
    int64_t dataSize=1; for(int64_t k=0;k<td_->rank;k++) dataSize*=td_->dataShape[k];
    int64_t numCores=td_->multicore.numCores; if(numCores<1) numCores=1;
    int64_t cpc=dataSize/numCores, ct=dataSize%numCores;
    int64_t cid=AscendC::GetBlockIdx();
    int64_t ms=cid*cpc+(cid<ct?cid:ct), mc=cpc+(cid<ct?1:0);
    int64_t maxChunk=td_->perBufBytes/sizeof(T); if(maxChunk<=0) maxChunk=1;
    for(int64_t off=0;off<mc;off+=maxChunk){
        int64_t cnt=(mc-off<maxChunk)?(mc-off):maxChunk;
        AscendC::DataCopyExtParams p; p.blockCount=1; p.blockLen=cnt*sizeof(T); p.srcStride=0; p.dstStride=0;
        AscendC::DataCopyPadExtParams<T> pad{};
        AscendC::DataCopyPad(buf_[B0].Get<T>(), gmData_[ms+off], p, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::DataCopyPad(gmY_[ms+off], buf_[B0].Get<T>(), p);
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
}
```

修复后（块间补 `MTE3_MTE2` 事件，verbatim）：

```cpp
// ─── Phase 1: Copy data → y (multi-core, disjoint flat regions) ───
// Each core copies its assigned contiguous flat range of y.  B0 is reused across
// chunks; the MTE3_MTE2 event guarantees the previous MTE3 (B0→GM) finishes before
// the next MTE2 (GM→B0) overwrites B0 (WAR hazard fix).
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

事件 ID 在 `Init` 中获取一次：`evMTE3_MTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));`（与既有 `evMTE2_MTE3_` 并列）。

## 验证信号

- 大数据(`max_seq_len=65536`) `kv_a_out` 前段(Phase2回读)`front match=False`、后段(Phase1新算)`back match=True`，小数据全部通过（来源平台:Ascend950；证据等级:tool_observed）
- 单核(`blockDim=1`)重跑同一大数据 UT 前段仍 `match=False`，排除多核竞争写同一 GM（来源平台:Ascend950；证据等级:tool_observed）
- 改 `TBuf` + `MTE2_MTE3`/`MTE3_MTE2` 双向 fence + 双 kernel launch + `aclrtSynchronizeStream` 后 3/3 UT 全部通过（来源平台:Ascend950；证据等级:tool_observed）
- 独立复现 R2（BenchHarnessScatter，CANN 9.2.0）：修复前多块纯拷贝复现（int32 100000 元素 / 5 块）失配 78160/100000、y[0]=21840（=maxChunk，块 1 首元素值）；单块用例 36/36 PASS，多块 78P/285F（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：块间补 `MTE3_MTE2` 后同一纯拷贝复现 0/100000 失配；拷贝+少量 update 复现 0/100000 失配；TTK 正向全量从 114 PASS 提升到 388/399（来源平台：950；证据等级：tool_observed）。

## 预防

- AscendC 跨迭代复用同一 UB 的 MTE2→MTE3 路径必须显式 `HardEvent::MTE2_MTE3` + `HardEvent::MTE3_MTE2` 双向 fence，不要依赖 `TQue` 的自动 event 管理（event ID 复用在大数据多轮场景不可靠）；纯 `TBuf` 手写循环只声明单向 `MTE2_MTE3` 同样不安全——它保证不了"下一块的装载晚于本块的写回"。
- 跨 Phase GM 读写（一个 Phase 写、另一个 Phase 读同一 GM）应拆为独立 kernel launch + Host `aclrtSynchronizeStream`，不要靠核内 `pipe_->Reset()` + `SyncAll()` 保证 GM 可见性。
- 搬运正确性测试必须构造超过单块容量（perBufBytes/sizeof(T)）的规模——单块路径根本不经过竞争窗口。

## 失效边界

- 平台：实测仅 dav-3510（Ascend950）/ CANN 9.1.0 与 9.2.0。a2/a3 未验证，推断可迁移（`HardEvent`/`aclrtSynchronizeStream` 为跨架构通用机制）。
- 单 Phase 算子（无回读）不需要双 launch；UB 足够大、单次搬入不跨迭代复用时 WAR 不触发，`TQue` 可用。
- `aclrtSynchronizeStream` 相对单 kernel 的额外同步开销未在本 trace 做成对相对测量。
- 竞争是时序依赖的：R2 的多块单核用例中仍有约 21%（75/288）碰巧通过——不要用"部分多块用例过了"反推同步正确。
- 消费者是 **V 流水线**（非 MTE3）的跨迭代 WAR 更棘手：`MTE3_MTE2` 事件对 V→MTE2 覆盖不足，见 [循环复用 UB buffer 跨迭代 MTE2/V 竞态卡](../runtime/cross_iteration_ub_reuse_mte2_vf_race.md)。

## 坏实践

- 用 `TQue` 自动同步替代跨流水线 WAR 的显式 fence（大数据暴露不可靠）。该路线标记为 `banned`。
- 块循环里只声明 `MTE2_MTE3`（本块装载完成才可写回）就认为 UB 复用安全——它保证不了"下一块的装载晚于本块的写回"。
- 用随机小张量测搬运正确性——单块路径根本不经过竞争窗口，必须构造超过单块容量的规模。
- 在同 kernel 内靠 `pipe_->Reset()` + `SyncAll()` 保证跨 Phase GM 可见性（大数据不可见）。该路线标记为 `banned`。
