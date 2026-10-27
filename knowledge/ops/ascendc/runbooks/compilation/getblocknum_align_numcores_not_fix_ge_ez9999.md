---
type: "Runbook"
title: GetBlockNum对齐numCores未修复GE路径EZ9999
description: kernel 三处以 AscendC::GetBlockNum() 覆盖 tiling numCores 以消除 cid>=numCores 越界后，GE 图路径（TBE 动态编译 blockDim=6）仍在 core 0 报 aivec error code 95 / mte error 0x80000000（EZ9999）；ACLNN 路径同一源码通过——对齐修复不足以作为"图路径 EZ9999 已修复"的验收结论，负结果卡。
tags: [task.debugging, topic.tiling, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:16:44Z'
updated_at: '2026-09-14T12:16:44Z'
platforms: ["950"]
---

# GetBlockNum对齐numCores未修复GE路径EZ9999

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter 验收 GEIR 图模式时（背景见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡：GE 图路径走 TBE 动态编译，kernel 以 blockDim=6 启动并报 EZ9999），此前一轮 kernel 修复已针对"blockDim 与 tiling numCores 不一致"落地：三处多核切分函数（CopyDataToY / CopyDataToAccFp32 / CopyAccToYNative）统一改为以 `AscendC::GetBlockNum()` 的实际网格数覆盖 tiling 传入的 `numCores`，源码注释明示动机——"GEIR graph path may compile blockDim != tiling numCores; mismatch caused cid>=numCores blocks to compute out-of-range GM offsets -> EZ9999 MTE DDR out of range"。

本会话验证该修复能否闭合 GE 图路径失败：确认修复在源码中在场（fresh staging 的 kernel 头 grep 到三处对齐代码，行 267/446/546），清空 TBE 缓存后 TBE 全新动态编译（kernel sha256 由 b3290843... 变为 3e578b16...、`GetKernels length` 2764→2812，证明新代码确实编进了二进制），然后跑 route 用例（`data:[4,3]/DT_FLOAT`、`indices:[2,3]/DT_INT32`、`updates:[2,3]/DT_FLOAT`、`dim=0`、`reduce=update`）：

```
GRAPH-RUN mode=route case=route-target ... rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL
[ERROR] RUNTIME ... ProcessDavidStarsCoreErrorInfo: ... there is an aivec error exception, core id is 0, error code = 95, ... mte error: 0x80000000
[ERROR] RUNTIME ... GetArgsInfo:tilingKey = 0, ... argsSize=40, schemMode=0, blockDim=6
```

仍然失败——且出错核是 **core 0**（首核在任何 numCores 切分下都合法，不属于对齐修复针对的"多余核"）。对照：同一 staging 源码的 ACLNN 直调回归全 PASS（`Test overwrite: PASSED`、`Test add: PASSED`）。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：自定义算子 kernel 在 GE 图路径（TBE 动态编译、blockDim=平台固化值 ≠ tiling numCores）下 EZ9999 的修复方案评估与验收口径。
- API 路径：`AscendC::GetBlockNum()` / `AscendC::GetBlockIdx()`、TilingData `multicore.numCores`、多核不相交 flat 区间切分、DataCopyPad。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"多核不相交区间搬运 + 单核 scatter"形态的 kernel 在图路径 blockDim≠tiling numCores 时的行为。
- 路线优先级：banned（作为"GE 图路径 EZ9999 的修复"被本会话证伪——不完整，不应再以同一修复宣称修好）；`GetBlockNum()` 对齐写法本身作为防御性实践仍合理保留（消除 cid>=numCores 一类越界分量，同族 ACLNN 路径的正例见[507035 多核分发未拦截多余核越界卡](../runtime/507035_unguarded_extra_cores_oob_dispatch.md)），但不能宣称它修复图路径崩溃。
- 明确不适用 / 未验证场景：
  - ACLNN 直调路径（实测 PASS，不受本边界影响；该路径上 cid 早退守卫修复 507035 的正例见上述链接卡）；
  - 图路径 blockDim==tiling numCores 的场景（未出现、未测）；
  - 其他 reduce 模式/dtype 的图路径用例（route 只跑过 f32 update 一例）。

## 根因

GetBlockNum 对齐只消除 cid>=numCores 一类"多余核越界"分量；本用例的 EZ9999 出错点是 core 0（任何切分下都参与的首核），存在对齐未覆盖的第二层机制，未能定位。

推导（机制链，代入本用例数值）：对齐后 core 0 的行为——dataSize=12（[4,3] f32）、numCores=GetBlockNum()=6 ⇒ cpc=12/6=2、ct=0 ⇒ core 0（cid=0）的区间 ms=0、mc=2，即只对 `gmData_[0..1]`/`gmY_[0..1]` 做 DataCopyPad，地址在张量头部、均在分配范围内；core 1..5 各处理 2 个元素同样合法。既然全部访问都在界内，core 0 仍报 `mte error: 0x80000000` 说明崩溃点不在区间计算——候选机制（GE 图路径下 workspace 声明、输出张量分配、`FetchEventID` 事件 ID 跨核语义等）均未验证。前序会话"core1-5 越界"的归因（见误诊）无法解释 core 0 出错。

## 尝试实现（verbatim，取自会话内源码读取）

kernel `CopyDataToY`（含对齐修复，`op_kernel/arch35/bench_harness_scatter_kernel.h`）：

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

另两处同型对齐（grep verbatim，行 446/546，CopyDataToAccFp32 / CopyAccToYNative）：

```cpp
    int64_t launchedBlocks = AscendC::GetBlockNum(); if (launchedBlocks > 0) numCores = launchedBlocks;
```

## 设计决策

- 修复设计（前一轮）选"kernel 侧运行时对齐"而不是"约束编译侧 blockDim"：kernel 无法控制图路径怎么编译，`GetBlockNum()` 是 kernel 内唯一可靠的"实际网格"来源——这个方向本身合理，本卡只证伪其充分性。
- 本会话验证方法选"指纹对照"：清缓存前后比较 kernel sha256 与 `GetKernels length`（2764→2812），先证明新代码已编入、再评估失败——避免"改了没编进去"的假阴性。
- 从错误行抽取 core id 作为分层信号：core 0（首核）与 core1-5（多余核）出错指向不同机制层，前者排除"区间越界"类归因。
- 当前方案路线档位：banned（作为图路径 EZ9999 修复方案）；对齐写法保留为防御性实践。

## 系统信息

- 硬件/软件：Ascend950PR_9579 device 0，CANN 9.2.0；GEIR route 测试 `operators/BenchHarnessScatter_package/tests/geir`（`./run.sh route`）；ACLNN 回归 `tests/aclnn/run.sh`。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（三处对齐修复，行 267/446/546）、`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`（TilingParse coreNum=1、TilingFunc 多核切分）。
- TilingFunc 侧 fallback（同文件，供对照）：`int64_t coreNum = 8;   // ascend950 default AIV cores`，CompileInfo 命中时被 `ci->coreNum`（=1）覆盖。

## 预防

- kernel 多核切分一律以 `AscendC::GetBlockNum()` 对齐实际网格（消除 cid>=numCores 分量），但验收口径上不要把"已对齐"当作图路径 EZ9999 已修复——必须以 route GRAPH-RUN value=PASS 闭环。
- 评估修复时先确认"新代码已编入"（比较编译产物 sha256/length），再看失败是否复现——本会话该流程一次排除"旧二进制"假设。
- 读设备错误时保留完整字段（core id、error code、mte error 值）：core id 是分层归因的关键信号。

## 验证信号

来源说明：以下信号来自该算子 GEIR 验收会话的 fresh staging + 清缓存重编译 + route/ACLNN 对照实测，均为 950 上板日志实测。

- 修复在场（fresh staging 源码三处）：`int64_t launchedBlocks=AscendC::GetBlockNum(); if(launchedBlocks>0) numCores=launchedBlocks;`（行 267/446/546）（来源平台：950；证据等级：tool_observed）。
- 全新编译确认：TBE kernel sha256 b3290843...→`3e578b16ebc75f549a65b651112e583d549c25e8964e246145ab7c5dc3cafb9f`、`GetKernels ... length=2812`（此前 2764）（来源平台：950；证据等级：tool_observed）。
- 修复后图路径仍崩：`GRAPH-RUN ... rungraph_rc=4294967295 outputs=0 ... value=FAIL`；`ProcessDavidStarsCoreErrorInfo: ... aivec error exception, core id is 0, error code = 95, ... mte error: 0x80000000`；`GetArgsInfo:tilingKey = 0, ... argsSize=40, schemMode=0, blockDim=6`（来源平台：950；证据等级：tool_observed）。
- 对照：同一 staging 源码 ACLNN 回归 `Test overwrite: PASSED`、`Test add: PASSED`（`ACLNN BenchHarnessScatter summary: SUCCESS (overwrite=1 add=1)`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只证伪"GetBlockNum 对齐 ⇒ GE 图路径 EZ9999 修复"这一命题（950PR + CANN 9.2.0、f32 update [4,3] 单用例）；不证伪对齐写法本身（它针对的 cid>=numCores 分量在本用例中无法单独观测，保留为防御性实践；ACLNN 路径上同族 cid 早退守卫修复 507035 的正例见[507035 多核分发未拦截多余核越界卡](../runtime/507035_unguarded_extra_cores_oob_dispatch.md)）。
- 推导"全部访问在界内"依赖 `GetBlockNum()` 在图路径确实返回 6（由 blockDim=6 与失败一致性间接支持，kernel 内未直接打印验证）。
- 其他 reduce 模式/dtype、更大 shape 的图路径行为未测；blockDim==numCores 的场景未出现。
- core 0 崩溃的确切机制（workspace/输出分配/事件 ID 等）未定位——本卡是负结果，不是根因闭环。

**未决项**：

- 优先项：core 0 `mte error: 0x80000000` 的机制层定位——候选方向：GE 图路径与 ACLNN 路径的 workspace 声明/分配差异（本用例 f32 update 理论不用 workspace，但 GE 路径是否分配了 TilingFunc 声明的大小未验证）、输出张量（UNKNOWN_RANK→InferShape 推导）在图路径的分配时机、`GetTPipePtr()->FetchEventID()` 事件 ID 在 6 核下的语义。复现入口：route 日志的 `GetArgsInfo` 打印了执行后 5 个参数（args(0..3) 为非零指针、args(4)=0），可与 ACLNN 路径的同类 dump 对照。
- GE 图路径该用例至今无 PASS 样例；本失败最终以"out-of-scope Kernel/Tiling"关闭验收，未修复。
## 坏实践

- 把"kernel 已加 GetBlockNum 对齐"当作图路径 EZ9999 已修复的验收依据——本会话实测同错（core 0 MTE error）。
- 只看"错误消失与否"不看出错 core id：core 0 与 core1-5 出错指向不同机制分层，混看会误归因。
- 修复验证不确认编译产物指纹就下结论——"源码改了"不等于"二进制换了"。

## 误诊

- 前序会话记载的归因"blockDim=6 ≠ numCores=1，cid>=numCores 的 core1-5 算出越界 GM offset 致 EZ9999"（同仓 LOG.md 转述，agent_reported）：本会话实测出错 core id 是 0，对齐修复（正是按该归因落地）后仍崩——该归因即使曾成立也不完整，存在第二层机制（tool_observed 的 core id 与失败复现优先于文档转述）。
- "GetBlockNum 在图路径返回 0 所以对齐没生效"：若返回 0 则 numCores 保持 tiling 值 1、core 0 处理全部 12 个元素同样在界内，仍无法解释 core 0 崩溃；两种取值下失败都无法用区间越界解释。
