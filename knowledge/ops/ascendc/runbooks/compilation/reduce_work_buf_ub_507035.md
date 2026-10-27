---
type: "Runbook"
title: "Reduce work buf 额外占 UB 致 507035"
description: "Ascend950 上为 ReduceSum/ReduceMax 单独开 fp32 work buffer，N>=14336 时 UB 总占用超限，kernel 启动后访问越界 UB 区触发 device error type 3 error code 507035。判别=缩小复现定位 N 阈值 + 逐 InitBuffer 累加字节对照 UB 上限 + 删冗余 work buffer 复用闲置 fp32 buffer 反向证明。根因为 work buffer 与主数据 buffer 同量级，把 UB 从刚好够推到超额；InitBuffer 划分超额编译期不报错，崩在 kernel 启动后。复现增量：反方向的同族陷阱——4 参重载隐式 PopStackBuffer 分配 sharedTmpBuffer 在大 tile 下触发 507035，须切 5 参显式传入独立 tmpBuf（A/B + 双向回退实验验证）；显式缓冲亦不可盲目加倍（加倍直接挤爆 UB 预算）。"
tags: ["task.debugging", "topic.memory", "topic.compute.reduction"]
status: stable
sources: []
created_at: '2026-08-18T15:55:48Z'
updated_at: '2026-10-10T00:00:00Z'
platforms: ["950"]
aliases:
  - "InitBuffer 逐个累加字节 对照 UB 上限"
  - "UB 预算超限 507035 kernel 启动后崩"
  - "buffer 加多了 kernel 起不来"
---

# Reduce work buf 额外占 UB 致 507035

## 现象

在 Ascend950（dav-3510）上优化 `AddRmsNormDynamicQuant` 算子（Add+RmsNorm+AbsMax+DynamicQuant 融合）。首版重写在 `ProcessRow` 里做两次归约（`ReduceSum` 求 meanSq、`ReduceMax` 求 absMax），按 AscendC 归约 API 约定要传一个 work buffer。agent 为此单独开了一个 `bufWork_`（fp32，`paddedD * sizeof(float)`），与 `bufA_/bufB_/bufGammaFp32_` 三个 fp32 buffer 并列。结果：精度上 case1-3（小 N）通过，case4 起报 NPU 崩溃，评测输出：

```
Case 4: status=failed, error=精度不达标: ... RuntimeError: copy_between_host_and_device_opapi:
  NPU function error: device error type 3, error code is 507035
Case 5: status=failed, err=AI算子执行失败: ... device error type 3, error code is 507035
Case 6: status=failed, err=AI算子执行失败: ... device error type 3, error code is 507035
```

之后 case7-20 全部 "device unrecoverable"（一次崩溃后设备进入不可恢复态）。

## 适用场景

- **工程形态**：ops-direct-invoke-flash Kernel 直调核函数构建（`op_kernel/*.cpp` 内 `TPipe::InitBuffer` 静态分配 UB）。
- **API 路径**：`AscendC::ReduceSum<float>(reduceResult, src, workBuf, ck)` / `AscendC::ReduceMax<float>(reduceResult, src, workBuf, ck, false)` 的 work buffer；`AscendC::TPipe::InitBuffer`；`AscendC::TBuf<AscendC::QuePosition::VECCALC>`。
- **验证过的平台与版本**：Ascend950（dav-3510）；CANN 版本未在 trace 中显式记录。
- **适用粒度**：`platform_wide`——Ascend950 上任何按行归约、N 可能达 14K+ 的算子。
- **路线优先级**：`recommended_fastpath`——Reduce 大 tile 场景切 5 参显式 sharedTmpBuffer 重载（verbatim before/after 已并入下方后续增量；显式缓冲按 tile 的 fp32 大小分配，不可盲目加倍）。主场景（`bufWork_` 删除复用闲置 fp32 buffer 的逐字修复 after 未从 normalized trace 恢复）在本卡主体仅作诊断结论与根因解释，不内联完整修复实现。
- **不适用或未验证场景**：a2/a3 平台未验证（推断 reduce work buffer 复用思路可迁移，但 UB 上限不同、阈值 N 不同，见 `## 失效边界`）；N 远小于 UB 容量阈值时无此问题。

## 判别方法

**触发条件**（tool_observed 失败症状）：Ascend950 上跑含按行归约（`ReduceSum`/`ReduceMax`）的算子，小 N case 通过、大 N case 评测报 `device error type 3, error code is 507035`，崩溃后后续 case 级联 `device unrecoverable`。崩溃发生在 kernel 启动后而非编译期，且与 dtype/value_range 无关、只随 N 增大出现。

**可执行探针**：
1. **缩小复现 N**：固定 M，逐步增大 N（如 8192 → 10240 → 14336 → 16384）跑评测，定位崩溃阈值。若小 N 通过、某 N0 起必崩，阈值即在该 N0。
2. **UB 占用核算**：打开 kernel `Init`，逐 `pipe_.InitBuffer(buf, bytes)` 累加字节，对照"fp32 buffer 数量 × `paddedD×4B` + 其余 buffer"。重点查是否为归约 work buffer 单独开了一个与主数据 buffer 同量级的 fp32 buffer。
3. **删冗余 work buffer 反向证明**：把单独开的 reduce work buffer 删掉，归约 work 复用一个在归约阶段已闲置的 fp32 buffer（如 gamma/常量 buffer，前提是归约不破坏其后续用途），重跑 N≥阈值 评测。若崩溃消失即佐证根因为 work buffer 额外占 UB。

**原始观测**（Ascend950，dav-3510，AddRmsNormDynamicQuant）：
- 缩小复现：`M=4 N=8192 bf16` 通过、`N=10240` 不崩、`M=4 N=14336 fp16 range[-1,1]` 崩——阈值在 N≥14336（与 dtype/value_range 无关）。
- UB 占用核算：N=14336 时 `paddedD=14336`（已对齐 `VL_B32=64`），4 个 fp32 buffer（`bufA_`/`bufB_`/`bufGammaFp32_`/`bufWork_`）各 `14336×4=57344B`，4 个共 `229376B ≈ 224KB`；加 `bufNative_`（`paddedD×sizeof(T)≈28KB`）、`bufI8_`（`paddedD×1≈14KB`）、`reduceBuf_`+`bufScalar_`（各 256B），总占用 ≈ 266KB，超 Ascend950 UB 上限。N≤8192 时 4×32768B=128KB + 其余 ≈ 170KB，不超限故不崩。
- 删 `bufWork_`、归约 work 复用 `bufGamma_` 后，`N=14336` 与 `N=16384` 均通过，全 20 case 精度 100% 全过——反向证明越界发生在 `bufWork_` 引入的额外占用上。

**机制解释**（根因分析，属 interpretation）：UB（Unified Buffer）是 Ascend950 上的片上 SRAM，`TPipe::InitBuffer` 静态划分、不能动态扩。`InitBuffer` 划分超额时 AscendC 不一定在编译期报错，常在 kernel 启动后访问越界 UB 区触发硬件异常 → `device error type 3, error code is 507035`。根因为归约 work buffer 单独开了一个与主数据 buffer 同量级的 fp32 buffer，把 UB 占用从"刚好够"推到"超额"：N≤8192 时 4 个 fp32 buffer 共 128KB + 其余 ≈ 170KB 不超限；N≥14336 时 4 个共 ≈ 224KB + 其余 ≈ 266KB 超限。复用 `bufGamma_` 做 reduce work 安全的前提是归约阶段 gamma 数据已被读完、且归约 work 写入不破坏后续还要用的数据（本算子 ReduceSum/ReduceMax 作用在 `bufB`（abs 值），gamma 不再被读，故复用安全）；若 gamma 在归约后仍要被读则不能这样复用。

## 预防

- 写 `TPipe::InitBuffer` 静态 UB 分配时，逐 buffer 算字节并求和，对照目标平台 UB 上限留余量；Ascend950 上 N 可能达 14K+ 的算子，fp32 buffer 数量 × `paddedD×4B` 要控制在 ~128KB 内（留一半给其余 buffer）。
- `ReduceSum`/`ReduceMax` 的 work buffer 不要单独开——复用一个在归约阶段已闲置的 fp32 buffer（如 gamma/常量 buffer，前提是归约不破坏其后续用途）。
- `InitBuffer` 划分超额时 AscendC 不一定在编译期报错，常在 kernel 启动后以 507035 形式崩；小 shape 通过不代表大 shape 安全，要用最大 N case 压测。

## 误诊

agent 早期一度怀疑 case4 是 bfloat16 的 `Cast<float→half>` reinterpret 问题、或 value_range[-10,10] 与 eps=1e-5 的数值问题；缩小复现到 N≥14336（与 dtype/value_range 无关）才定位到 UB 容量（属 agent reasoning，非实测信号，不进 `## 验证信号`）。

## 验证信号

- 修复前 eval stderr：case4/5/6 `NPU function error: device error type 3, error code is 507035`，case7-20 `device unrecoverable`（来源平台：950；证据等级：tool_observed）
- 缩小复现：`M=4 N=8192 bf16` 通过、`N=10240` 不崩、`M=4 N=14336 fp16 range[-1,1]` 崩——阈值在 N≥14336（来源平台：950；证据等级：tool_observed）
- 修复后 eval stdout：删 `bufWork_` 复用 `bufGamma_` 后 `N=14336` 与 `N=16384` 均通过，全 20 case 精度 100% 全过（来源平台：950；证据等级：tool_observed）

## 失效边界

- a2/a3 平台：未验证，推断可迁移（reduce work buffer 复用思路平台无关，但 UB 上限不同、阈值 N 不同）。
- Ascend950 UB 精确上限未在 trace 中由探针确认；本卡阈值 N≥14336 是实测观察值，非由 UB 上限反推。
- 若算子结构使 gamma 在归约后仍要被读，不能复用 `bufGamma_`，需另选闲置 buffer 或回退到单独 work buffer（但要把 N 限制在 UB 内）。

## 后续增量（2026-09-17，并行策展批：4 参隐式 PopStackBuffer 大 tile 崩溃，切 5 参显式缓冲）

同族反方向陷阱：显式 work buffer 会挤爆 UB（本卡主体），而 4 参重载的**隐式**栈分配在大 tile 下同样触发 507035。Ascend950 + CANN 9.2.0 上 softmax 类算子分支 kernel 的 ST：fp32 小 shape 用例 PASS，10 个大 shape / 归约轴超长用例（attention `[2,8,64,512]`、classification `[32,1000]` 三 dtype、large_reduce `[1,131072]`、max_reduce `[1,2097152]` 的 fp16/bf16）稳定 `[ACLNN_FAIL]` + `ACL error 507035 at aclrtSynchronizeStream`。kernel 的行最大值/行求和用模式 Reduce API 的 **4 参形式**（不带 sharedTmpBuffer，由 API 内部 `PopStackBuffer` 从栈上隐式分配临时缓冲），tile 尺寸由 tiling 决定（fp32 下可达 `[24,512]` 即 12288 元素）。

根因：4 参重载依赖 API 内部隐式栈分配（`PopStackBuffer`）获取 sharedTmpBuffer，在大 tile 场景下该隐式分配不可用/不足，触发 AIV 异常 507035；同一调用点改用 5 参重载显式传入一块独立、tile 尺寸对齐的 UB 缓冲后异常消失。两个重载来自 `$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/reduce/reduce.h`。

before/after（verbatim，`DoReduceSum` 同族两重载同样改法）：

```cpp
// before（4 参隐式分配，大 tile 崩溃）
__aicore__ inline void DoReduceMax()
{
    auto srcTensor = GetComputeTensor();
    auto dstTensor = tmpBuf0_.Get<float>();
    uint32_t srcShape[2] = {
        static_cast<uint32_t>(td_->aUbFactorAlign),
        static_cast<uint32_t>(td_->rUbFactorAlign)
    };
    AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, false>(
        dstTensor, srcTensor, srcShape, true);
}

// after（5 参显式 sharedTmpBuffer，修复）
__aicore__ inline void DoReduceMax()
{
    auto srcTensor = GetComputeTensor();
    auto dstTensor = tmpBuf0_.Get<float>();
    auto tmpTensor = tmpBuf1_.Get<uint8_t>();          // 显式独立缓冲
    uint32_t srcShape[2] = {
        static_cast<uint32_t>(td_->aUbFactorAlign),
        static_cast<uint32_t>(td_->rUbFactorAlign)
    };
    AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, false>(
        dstTensor, srcTensor, tmpTensor, srcShape, true);
}
```

判别：同一 kernel 内小 tile 用例 PASS、大 tile 用例 507035，且 PRINTF 打点崩溃点位于 Reduce 调用时，优先怀疑隐式 sharedTmpBuffer，直接切 5 参显式缓冲做 A/B。显式缓冲尺寸按 tile 的 fp32 大小分配（`aUbFactorAlign * rUbFactorAlign * sizeof(float)` 再对齐），与 reduce 输入同量级——**不能盲目加倍**（见增量坏实践）。

### 复现增量验证信号

来源说明：以下信号来自该 softmax 算子分支 ST 调试会话（每轮改码后全量重跑 run.sh，含两次回退实验）：

- 修复前：10 例 `[ACLNN_FAIL]`（507035）（来源平台：950；证据等级：tool_observed）。
- 切 5 参显式缓冲后：`27/46 precision PASSED, 44 aclnn-OK, 2 aclnn-FAIL`——8 例 group 崩溃消失（来源平台：950；证据等级：tool_observed）。
- 回退实验 1：改回 4 参隐式分配 → 10 例 ACLNN_FAIL 复现（`26/46, 36 aclnn-OK`）（来源平台：950；证据等级：tool_observed）。
- 回退实验 2：保持 5 参但把 tmpBuf1 尺寸加倍 → 再次 10 例 ACLNN_FAIL（UB 预算被挤爆，殃及其他用例）；恢复原尺寸显式缓冲 → 回到 `27/46, 44 aclnn-OK, 2 aclnn-FAIL`（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- 极小 tile（如 `[8,8]`）下 4 参形式未见崩溃；4 参崩溃的精确 tile 尺寸阈值未测定；`isReuseSource=true` 路径与其他 pattern（RA）未验证。
- 显式缓冲的最小安全尺寸公式未标定（本轨迹按 tile fp32 大小分配即够用）。
- 坏实践：大 tile 场景沿用 4 参隐式分配写法（照抄小 tile 参考实现）；修复时"宁可加倍"地放大 tmpBuf——加倍直接挤爆 UB 预算，让原本正常的用例也崩溃（本轨迹实测回归，即本卡主体根因的又一实例）。
- 4 参重载隐式 `PopStackBuffer` 的确切分配上限/失败机制未验证（未追到 runtime 实现层），建议最小复现标定崩溃的 tile 尺寸阈值。

## 相关卡片

- [PB-45 `TPipe::Reset()` 释放全局 `g_tpipeImpl` 事件池](pb_45_tpipe_reset_frees_the_global_g_tpipeimpl_event_poo.md)：同一方法（把 Init/InitBuffer 期的声明逐个累加，对照硬上限）作用在**事件 id**而非 UB 字节上；超限后**不报 507035 而是静默挂死**。症状是「加了一个 buffer 就卡死」时先读该卡，按本卡的 507035 思路查下去会查错方向。
- [arch3510 事件池 QUE_MAX_EVENT=8 的事实锚点](../performance/tile_merged_dma_eliminate_mte2_issue_bound.md)：事件 id 侧的预算数字（每核每事件类型 8、每方向独立位图）只写在那张性能卡里；没有这个数就无法像本卡数 UB 字节一样数事件 id。
- [TBuf Get 提前到搬运前致 kernel 间歇挂死](tbuf_get_before_copy_hangs_kernel.md)：同样是「改了 buffer 之后挂死」，但机理是 TPipe 时序而非预算耗尽；两者症状相同、处置相反，勿互相套用。
