---
type: "Runbook"
title: fp16 aUbFactor 被 cacheBuf 钳制，路由随 B 翻转
description: A 向单 chunk 因子 aUbFactor 不是 (dtype,H) 的函数：大 B 下经 cacheBuf 16KB 预算钳制 fp16 128→64，aLoopCntTotal 翻倍并翻转 base/group 路由；复算路由必须带实际 B 走完整钳制环。
tags: [task.testing, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# fp16 aUbFactor 被 cacheBuf 钳制，路由随 B 翻转


## 现象

同一个算子、同一 dtype 与 hidden 维（fp16, H=2048）的两个 ST 用例路由到了不同 kernel 分支：B=32 的性能用例（P3）落 group 分支（当时 group kernel 是占位 stub，红基线 FAIL），B=1024 的用例（N2 形状）落 base 分支（PASS）。验收 group 分支 ST 清单时若按"aUbFactor 只由 dtype 和 UB 容量决定"的直觉类推，会预期两者同路由——实测相反。

机制：fp16 的 aUbFactor 初值 128 在 B=1024 下被 cacheBuf 预算钳制到 64，aLoopCntTotal 从 CeilDiv(2048,128)=16 翻倍到 CeilDiv(2048,64)=32，越过运行时路由分界 28（coreNum/2，运行时核数实测反推 56-57，非 INI 规格 64——见同批验收沉淀的「GetCoreNumAiv 实测≈56-57」知识），于是 N2 落 base 并被剔出 group 分支清单。验收方用探针复证了路由翻转本身（N2 本体 PASS），并按 host tiling 公式手算了完整钳制链（aUbFactor 中间值未在设备侧打印，以实跑路由行为对账）。

## 适用场景

- 工程形态：aclnn 算子仓（ops-registry-invoke 流程）ST 用例的分支归类、白盒用例安全盒声明、host tiling 切分复算。
- API 路径：op_host tiling 的 aUbFactor/rUbFactor 计算与 cacheBuf 钳制环；`PlatformAscendC::GetCoreMemSize(UB)`（实机 ub=253952）。
- 验证过的平台与 CANN 版本：Ascend 950（实机 aclnn 运行）+ CANN 9.0.0-beta.2（与 frontmatter platforms 一致）。
- 适用粒度（transfer_scope）：operator_family——凡 host tiling 因子受 B 经 cacheBuf/归约树深度钳制的算子（具体常数随算子，按各自 tiling 代码代入）。
- 路线优先级：recommended_fastpath（复算 aLoopCntTotal 必须带实际 B 走完整钳制环，不按 (dtype,H) 查表类推）。
- 明确不适用/未验证场景：无钳制环的单因子 tiling；其他算子的钳制公式未验证（常数不同）；bf16 大 B 同型钳制未实测（本批 bf16 用例 B≤128 未触发）；分界值 28 绑定本机运行时核数，换机须重测。

## 根因

A 方向单 chunk 因子 aUbFactor 不是 (dtype, H) 的确定函数——它受 16KB cacheBuf 预算经 `cacheCount = CalLog2(FindNearestPower2(rLoopCntTotal)) + 1`（rLoopCntTotal 随 B 增大）的钳制环约束，大 B 下被减半使 aLoopCntTotal 翻倍并翻转 base/group 路由。

机制推导（N2：fp16 B=1024 H=2048，实机 ub=253952；19/6 为该算子中间域布局系数，仅作演算实例）：

- 初值 aUbFactor = Min(128, CeilAlign(2048,16)) = 128
- postBufSize = CeilAlign(6×128×4, 32) = 3072；bytesPerRElem = 19×128×4 = 9728
- rIMax = (253952 − 16384 − 3072)/9728 = 234496/9728 = 24.1 → 24；rUbFactor = Min(24, 1024) = 24；rLoopCntTotal = CeilDiv(1024, 24) = 43
- cacheCount = CalLog2(FindNearestPower2(43)) + 1 = CalLog2(32) + 1 = 6；6 × CeilAlign(6×128,8) × 4 = 6×768×4 = 18432 > 16384 → 违反 → aUbFactor 减半为 64
- 复算：postBufSize = CeilAlign(1536,32) = 1536；bytesPerRElem = 19×64×4 = 4864；rIMax = (253952−16384−1536)/4864 = 236032/4864 = 48.5 → 48；rUbFactor = 48；rLoopCntTotal = CeilDiv(1024,48) = 22；cacheCount = CalLog2(16)+1 = 5；5 × CeilAlign(384,8) × 4 = 5×1536 = 7680 ≤ 16384 ✓
- aLoopCntTotal = CeilDiv(2048, 64) = 32 > 运行时分界 28 → base

对照（P3：fp16 B=32 H=2048）：rLoopCntTotal = 2 → cacheCount = CalLog2(2)+1 = 2 → 2×3072 = 6144 ≤ 16384 ✓ 不触发钳制 → aUbFactor=128 → aLoopCntTotal = 16 ≤ 28 → group。

B 进入链路的路径：B ↑ → rLoopCntTotal ↑ → FindNearestPower2(rLoopCntTotal) ↑ → cacheCount（Phase 1 二分树缓存层数）↑ → cacheBuf 占用（cacheCount × CeilAlign(6×aUbFactor,8) × 4）↑ → 超过 16384 时 aUbFactor 减半并全链重算（rUbFactor/rLoopCntTotal/cacheCount 随之变化）。

## 判别方法

- 对照实验：同 (dtype,H) 取两个 B（小 B 不触发 vs 大 B 触发钳制），以桩/实现不对称的 PASS/FAIL 或 tiling 输出（usedCoreNum/rGroupCnt）观察路由是否翻转。
- 复算：按钳制环全式代入实际 B（如上演算），不要按 (dtype,H) 查表类推；复算结果与实跑路由对账。

## 设计决策

- 探针把 N2 本体（fp16 B=1024 H=2048）与 fp32 跨界点（H=1856/2048 → aLoop=29/32）放进同一清单：一次实跑同时复证"运行时分界 28"与"钳制后 aLoop=32 落 base"两个独立机制。
- 放弃的候选：只按 dtype 查 aUbFactor 表（被 P3/N2 对照证伪）；不重算直接信设计文档 §10 注记（验收要求独立复算，前序注记只能作对账参照）。
- 当前方案属哪档路线：recommended_fastpath。

## 系统信息

- 工具链：CANN 9.0.0-beta.2，GCC 13.3.0，python3 3.12.9 + torch 2.7.1+cpu（数据生成侧）。
- 目标平台：ascend950（实机 ACLNN 运行）。
- 目录布局：operators/<pkg>/{op_host/arch35, op_kernel/arch35, tests/aclnn}。
- 相关文件（与事件流实际读写一一对应）：op_host/arch35/<op>_tiling_arch35.cpp/.h（因子与钳制环、ctx 结构）、tests/aclnn/st_cases_branch1.json（P3/G5 的钳制注记）、tests/aclnn/{gen_st_oracle.py, test_aclnn.cpp}（探针数据与实跑）、/tmp 临时探针清单（PRB-N2）。

## 完整代码

tiling 上下文结构（op_host/arch35 tiling 头，verbatim——注意 ubAvailable 已扣 16KB cacheBuf）：

```cpp
// ─── 一次 tiling 计算的上下文（§9；字段单位见注释）───
struct ThnnFusedGruCellGradCtx {
    // 平台参数（GetPlatformInfo，§9）
    int64_t coreNum = 0;        // AIV 核数（核数）
    int64_t ubSize = 0;         // UB 字节预算（字节）
    int64_t blockSize = 0;      // UB block 字节数（字节，32B）
    int64_t cacheLineSize = 0;  // cache line 字节数（字节，256B）
    // 输入（GetShapeAttrsInfo，§2）
    int64_t B = 0;              // batch 维（元素）
    int64_t H = 0;              // hidden 维（元素，hsz = storage.shape[1] // 5）
    int64_t dtypeSize = 0;      // 输入元素字节数（字节）
    int32_t isBias = 0;         // attr has_bias
    // UB 切分派生量（§4.2-§4.3）
    int64_t maxDtypeSize = 0;   // max(dtypeSize, 4)（字节，fp32 中间域恒 4）
    int64_t bsElem = 0;         // blockSize / dtypeSize（元素）
    int64_t cachelineTmp = 0;   // FloorAlign(cacheLineSize/dtypeSize, bsElem)（元素）
    int64_t ubAvailable = 0;    // ubSize − 16KB cacheBuf（字节）
    int64_t aUbFactor = 0;      // A(H) 单 chunk 元素数（元素，valid）
    int64_t rUbFactor = 0;      // R(B) 单 chunk 元素数（元素，valid）
    int64_t rLoopCntTotal = 0;  // CeilDiv(B, rUbFactor)（块数）
    // 多核切分（§5.1/§6.2）
    int64_t aSplitChunkCnt = 0;     // CeilDiv(H, aUbFactor)（块数）
    int64_t aLoopCntTotal = 0;      // = aSplitChunkCnt（前导 A=1 不贡献，块数）
    int64_t aSmallCoreLoopCnt = 0;  // 小核块数
    int64_t aBigCoreLoopCnt = 0;    // 大核块数
    int64_t aBigCoreCnt = 0;        // 大核个数（核数）
    int64_t usedCoreNum = 0;        // 实际使用核数（核数）
    bool isGroup = false;           // ShouldUseGroup 判定结果
    int64_t rGroupCnt = 0;          // group Phase 1 分组数（base 为 0）
};
```

探针 N2 本体用例（/tmp 临时产物，verbatim）：

```json
{
  "case_id": "PRB-N2",
  "source": "kernel-verifier probe: N2 本体 B=1024 fp16 H=2048（Validation.md N2 行；执行者报告：经 cacheBuf 钳制 aUbFactor 128→64 → aLoopCntTotal=32 实机落 base，剔出 branch-1 manifest）",
  "B": 1024, "H": 2048, "dtype": "float16", "has_bias": true,
  "inputs": {
    "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
    "storage": { "pattern": "forward_reachable" }
  },
  "assert_mode": "stat_rel_err", "threshold": 1e-3
}
```

对照用例的清单注记（st_cases_branch1.json，verbatim，执行者口径的钳制记载）：

```text
P3: "Validation.md「端到端测试用例矩阵」性能关键 shape #P3（B=32,H=2048,fp16；实算 aUbFactor=128、aLoopCntTotal=16、rLoopCntTotal=2 → usedCoreNum=32、rGroupCnt=2；…）"
G5: "DESIGN-BRANCH-1.md §9 分支验证 #G5（B=2048,H=128,fp16：cacheBuf 钳制环触发 aUbFactor 128→64；实算 rLoopCntTotal=43 → totalOuter>coreNum 的 2D 对齐路径 usedCoreNum=44、rGroupCnt=22（R 组大小核）；…）"
```

## 预防

- 为分支清单/白盒安全盒归类用例时，aLoopCntTotal 一律按实际 B 走完整钳制环复算；清单行注明实算口径。
- 排查"同 (dtype,H) 路由不一致"时，先对照两个 B 的 rLoopCntTotal/cacheCount 是否跨过钳制阈值，再怀疑路由判定本身。
- 涉及 fp16/bf16 且 B 较大的性能 shape，优先实跑确认路由归属（桩/实现不对称或 tiling 输出）。

## 验证信号

来源说明：以下信号来自该 group 分支 Kernel ST 验收会话——既有清单用例的实跑结果与验收方自建探针（含 N2 本体）的实机输出；钳制链算式为验收方按 tiling 公式手算并与实跑路由对账。

- P3（fp16 B=32 H=2048）→ verdict FAIL（group 路由）（来源平台：950；证据等级：tool_observed）
- PRB-N2（fp16 B=1024 H=2048，N2 本体）→ verdict PASS（base 路由），两槽一致（来源平台：950；证据等级：tool_observed）
- G5（fp16 B=2048 H=128，清单注记"cacheBuf 钳制环触发 aUbFactor 128→64、rLoopCntTotal=43"）→ verdict FAIL（group 路由；aLoop=CeilDiv(128,64)=2 ≤ 28）（来源平台：950；证据等级：tool_observed——路由判定为实测，aUbFactor 数值为清单注记口径）

（注：根因小节中的钳制链算式为按 tiling 公式的手算推导，中间值未在设备侧打印，属 agent_reported 口径；本章节只收实跑实测信号。）

## 失效边界

- 钳制环常数（19/6/16384、CalLog2/FindNearestPower2）为源算子 tiling 设计值，跨算子不复用；迁移时按各自 tiling 代码代入。
- "分界 28"绑定本机运行时核数（运行时 GetCoreNumAiv() 实测反推 56-57）；换机器必须重测分界。
- bf16 大 B、非 2 幂 rLoopCntTotal 的钳制边界未系统扫描；钳制减半是否循环执行多次（至满足预算）未逐行核验。
- 探针判读依赖的 FAIL=group 信号仅在"被验分支为桩"阶段成立。

## 坏实践

- 按 (dtype,H) 类推路由归属（P3→N2 即反例：同 fp16 H=2048，B=32 落 group、B=1024 落 base）。
- 白盒安全盒按"最坏 aUbFactor=64 ⇒ aLoopCntTotal≤32=coreNum/2"的规格核数口径推边界——同时忽略运行时核数与钳制的 B 依赖两重漂移。

## 误诊

- "N2 落 base 是 fp16 有专门路由"——路由判定只看 aLoopCntTotal/rLoopCntTotal（dtype 不参与；清单中同形状 fp16/fp32/bf16 切分一致的注记与 ShouldUseGroup 代码一致）。
- "aUbFactor 只取决于 dtype 和 UB 大小"——被 P3/N2 对照证伪（B 经 rLoopCntTotal→cacheCount 介入）。


