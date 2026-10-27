---
type: "Runbook"
title: "507035 errcode 271：TopK 保留 UB 预算漏记"
description: "大 batchRows 用例 aclnn 返回 0 但同步报 507035，plog errcode 271 为标量访问内部缓冲越界；根因是 tiling UB 预算漏记 kernel 每行 buffer 开销且安全余量不足以覆盖 TopK（__ASC_USE_RESERVED_UBUF__）保留 UB，修复为 perRowCost 逐项对账 + 128KB 安全余量。"
tags: [task.debugging, topic.memory, topic.tiling, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035 errcode 271：TopK 保留 UB 预算漏记

## 现象

在 Ascend950PR（DAV_3510/arch35，CANN 9.1.0）上调一个 SoftMax+TopK 门控算子（数据流：DataCopyPad GM→UB → SoftMax → TopK → Cast → GM）。全量 ST 45 例中 6 个"大 M"用例稳定崩溃——aclnn 下发本身返回 0，随后流同步/测试进程报 **507035（向量核异常）**：

- fp16 `[65536,64]`、`[131072,64]`、`[262144,64]`，k=8（M 很大、E=64 小）；
- fp32 3D `[2,4096,256]`，k=32；
- fp16/fp32 `[131072,1024]`，k=128（大 M + 大 E + 大 k）。

小 M 用例全部正常。用单用例 CSV + `ASCEND_SLOG_PRINT_TO_STDOUT=1` 复现大 M 用例，plog 里多核（core 0/1/2/3/6/7…）同时报同一错误：

```
[ERROR] ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), ...
there is an aivec error exception, core id is 0, error code = 271, ...
The extend info: errcode:(271) errorStr: The address for scalar to access the internal
buffer is out of bounds. subErrType: 0x4.
```

当时代码状态：kernel 用 `TPipe::InitBuffer` 按行分配一批 TBuf（inputBuf_/softmaxBuf_/sumBuf_/maxBuf_/yBuf_/idxBuf_/finishBuf_/sharedTmpBuf_/rowIdx*），行数 `batchRows` 由 op_host tiling 按 UB 预算反推；tiling 的安全余量当时只有 32KB。

## 适用场景

- 工程形态：AscendC kernel（向量核）+ host TilingFunc 自管 UB 预算（`batchRows = (ubSize - fixedCost - margin) / perRowCost` 式反推）的自定义算子。
- API 路径：`AscendC::TopK`（adv_api 高阶 API，3510 实现）、`TopKTilingFunc`/`GetTopKMaxMinTmpSize`、`TPipe::InitBuffer`；plog 错误码 271/同步错误码 507035。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，UB 上限 248KB），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——arch35 上任何"tiling 反推 batchRows + 高阶 API（TopK 等）内部使用保留 UB"的算子。
- 路线优先级：recommended_fastpath（正确性必需）。
- 明确不适用 / 未验证场景：不使用 `__ASC_USE_RESERVED_UBUF__` 类高阶 API、只用手写向量指令的 kernel（保留 UB 需求可能不同，未验证）；其他架构的 UB 上限与保留量需重查（未验证，推断可迁移）。

## 根因

tiling 侧 UB 预算没有覆盖两块真实开销，导致反推出的 `batchRows` 过大、TBuf 总分配挤占了高阶 API 的保留 UB，TopK 内部标量访问越界：

1. **每行 buffer 漏记**：kernel 里 `finishBuf_` 按 `batchRows * 32B` 分配（每行 1 个 32B datablock），而 tiling 只在 `fixedCost` 里记了 32B（按"最少 1 行"计），per-row 部分完全没进 `perRowCost`——batchRows 越大，低估越严重（每行差 32B）。
2. **保留 UB 余量不足**：arch35 的 TopK 3510 实现带 `__ASC_USE_RESERVED_UBUF__(3510, "TopK is forbidden when compile option --cce-disable-asc-reserved-ubuf is enabled")` 属性，即 TopK 需要 TBuf 之外的保留 UB（PopStackBuffer 机制 + Sort 内部栈）；`GetTopKMaxMinTmpSize` 返回的 tmp 大小不含这部分，tiling 的 `UB_SAFETY_MARGIN` 必须兜住。实测 32KB、64KB 都不够（大 M 用例仍崩），128KB 才稳定通过。

机制链（错误码对上号）：errcode 271 的字面语义是"标量访问内部缓冲（UB）越界"——不是 MTE 越界、不是 UB 总量超限的编译错，而是运行期某条标量指令的 UB 地址出界；本例出界的是 TopK 内部（MergeSort/GatherMask 一带）的标量访问，其可用保留区被过大的 TBuf 分配挤掉。另注意：TopK 头文件里的约束检查（`1 <= k <= n <= inner`、`inner % 32 == 0`、`inner <= 4096`）是 `ASCENDC_ASSERT`，**真机上是空实现，仅 CPU debug 模式生效**——违反约束不会在真机报参数错，只会以垃圾值/越界的形式暴露。

## 解决配方

修复前（tiling，UB 预算两处低估；DAV_3510 UB 248KB）：

```cpp
constexpr uint32_t UB_SIZE_LIMIT = 248U * 1024U;  // DAV_3510 UB 248KB
constexpr uint32_t UB_SAFETY_MARGIN = 32U * 1024U;  // 32KB 安全余量（reserved UB, PopStackBuffer 等）
...
    // perRowCost 计算
    uint64_t perRowInputCost = static_cast<uint64_t>(topkInner) * computeDtypeSize;  // inputBuf_ (COMPUTE_T sized)
    uint64_t perRowSoftmaxCost = static_cast<uint64_t>(topkInner) * computeDtypeSize; // softmaxBuf_ (COMPUTE_T sized)
    uint64_t perRowSumMaxCost = 64U;  // sum(32B) + max(32B)
    uint64_t perRowYCost = static_cast<uint64_t>(kPad) * computeDtypeSize;
    uint64_t perRowIdxCost = static_cast<uint64_t>(kPadIndex) * sizeof(int32_t);
    uint64_t perRowRowIdxOutCost = static_cast<uint64_t>(kPadIndex) * sizeof(int32_t);
    uint64_t perRowCost = perRowInputCost + perRowSoftmaxCost + perRowSumMaxCost +
                          perRowYCost + perRowIdxCost + perRowRowIdxOutCost;
    //                      ^^^^ 修复前: 没有 finishBuf_ 的每行开销

    // fixedCost
    uint64_t rowIdxBaseBufSize = static_cast<uint64_t>(kPadIndex) * sizeof(int32_t);  // base only
    uint64_t srcIdxBufSize = static_cast<uint64_t>(topkInner) * sizeof(int32_t);  // TopK srcIndex (按 topkInner)
    uint64_t finishBufFixedCost = static_cast<uint64_t>(ALIGN_BYTES);  // 最少 1 行 32B
    uint64_t fixedCost = static_cast<uint64_t>(sharedTmpSize) + rowIdxBaseBufSize + srcIdxBufSize + finishBufFixedCost;

    // batchRows = FloorDiv(FloorAlign(ubSize - fixedCost - safetyMargin, 32), perRowCost)
    int64_t availableUb = static_cast<int64_t>(ubSize) - static_cast<int64_t>(fixedCost) -
                          static_cast<int64_t>(UB_SAFETY_MARGIN);
    ...
    int64_t batchRows = FloorDiv(alignedAvailable, static_cast<int64_t>(perRowCost));
```

（对应 kernel 侧实际分配：`pipe.InitBuffer(finishBuf_, static_cast<uint32_t>(batchRows_ * ALIGN_BYTES));` ——每行 32B，tiling 却按固定 32B 记账。）

修复后（每行开销对账 + 128KB 余量）：

```cpp
constexpr uint32_t UB_SIZE_LIMIT = 248U * 1024U;  // DAV_3510 UB 248KB
constexpr uint32_t UB_SAFETY_MARGIN = 128U * 1024U;  // 128KB 安全余量（TopK __ASC_USE_RESERVED_UBUF__ + Sort 内部栈）
...
    uint64_t perRowRowIdxOutCost = static_cast<uint64_t>(kPadIndex) * sizeof(int32_t);
    // 修复: finishBuf_ 在 kernel 中按 batchRows * 32B 分配, 需计入 perRowCost
    uint64_t perRowFinishCost = static_cast<uint64_t>(ALIGN_BYTES);  // 32B/行
    uint64_t perRowCost = perRowInputCost + perRowSoftmaxCost + perRowSumMaxCost +
                          perRowYCost + perRowIdxCost + perRowRowIdxOutCost + perRowFinishCost;

    // fixedCost
    uint64_t rowIdxBaseBufSize = static_cast<uint64_t>(kPadIndex) * sizeof(int32_t);  // base only
    uint64_t srcIdxBufSize = static_cast<uint64_t>(topkInner) * sizeof(int32_t);  // TopK srcIndex (按 topkInner)
    // 修复: finishBuf_ 的 per-row 成本已移至 perRowCost, fixedCost 仅含非 per-row 部分
    uint64_t fixedCost = static_cast<uint64_t>(sharedTmpSize) + rowIdxBaseBufSize + srcIdxBufSize;
```

定位链（可复用）：单用例 CSV 复现 → `ASCEND_SLOG_PRINT_TO_STDOUT=1` 抓 plog → 找 `error code =` 行拿到 271"scalar … internal buffer … out of bounds" → 检查 kernel 全部 `InitBuffer` 与 tiling 预算逐项对账（重点：按 batchRows 缩放的 buffer 是否都进了 perRowCost）→ 仍崩则加大 `UB_SAFETY_MARGIN`（32K→64K→128K 阶梯）复测。

## 设计决策

- 为什么用固定 128KB 余量而不是精确计算保留 UB：`__ASC_USE_RESERVED_UBUF__` 的保留量没有公开查询接口，`GetTopKMaxMinTmpSize` 也不含它；实测阶梯到 128KB 后 6 个崩溃用例全过且小 M 用例无回归，先用经验值保正确性。
- 为什么把 finishBuf 从 fixedCost 挪到 perRowCost：它的真实大小随 batchRows 线性增长，按固定 32B 记账在"batchRows 大"时必然低估——对账原则是"kernel 每个 InitBuffer 的尺寸公式 = tiling 某个 cost 项"。
- 同一轮还顺带改了 `topKInfo_.n = topkInner_`（TopK 参数语义）与 SoftMax tiling 改用 `softmaxMaxTmp`（精度），这两项对崩溃用例的单独效果未拆分验证（见误诊）。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写 tiling 时把 kernel 每个 `InitBuffer` 的尺寸公式抄成 cost 项清单逐项对账；凡 `batchRows *` 缩放的 buffer 一律进 perRowCost，只有真正固定的进 fixedCost。
- 使用带 `__ASC_USE_RESERVED_UBUF__` 的高阶 API（TopK 等）时，UB 预算留足安全余量（arch35 实测 128KB），不要假设 `Get*MaxMinTmpSize` 覆盖全部内部开销。
- 大 M 崩溃先抓 plog 看错误码：271=标量 UB 越界（查预算对账）、95=MTE DDR 越界（查 GM/workspace 地址），两者修法完全不同。
- 不要依赖 ASCENDC_ASSERT 在真机拦截参数违规——它只在 CPU debug 模式生效。

## 验证信号

来源说明：以下信号来自该算子的两轮修复会话（每轮：build.sh 重建 → 卸载旧包 → 重装 → 全量 45 例 ST）。

- 修复前：全量 34/45，6 个大 M 用例（fp16 [65536,64]/[131072,64]/[262144,64] k=8、fp32 3D [2,4096,256] k=32、fp16/fp32 [131072,1024] k=128）同步 507035（来源平台：950；证据等级：tool_observed）。
- 修复前：单用例 plog 多核报 `errcode:(271) errorStr: The address for scalar to access the internal buffer is out of bounds. subErrType: 0x4`（来源平台：950；证据等级：tool_observed）。
- 修复轮 1（finishBuf 每行 32B 计入 perRowCost + UB 余量 32KB→64KB，同轮另含 n=topkInner/softmaxMaxTmp 两项非 UB 修改）：36/45，`[131072,1024]` 两个大 M+E+k 用例转 PASS，其余 4 个大 M 用例仍 507035（来源平台：950；证据等级：tool_observed）。
- 修复轮 2（UB 余量 64KB→128KB）：40/45，`[65536,64]`/`[131072,64]`/`[262144,64]` k=8 与 fp32 3D 用例全部转 PASS（MERE 8.20e-04/3.00e-03 等在阈值内），此后多轮回归稳定 40/45（来源平台：950；证据等级：tool_observed）。
- 事实锚点：TopK 3510 实现头文件含 `__ASC_USE_RESERVED_UBUF__(3510, ...)`；TopK 约束检查（k<=n<=inner、inner%32==0）在真机为空实现（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 128KB 余量为 arch35/CANN 9.1.0 的实测安全值，非官方规格；E/k 更极端（如 k=1024、E=2048）或叠加更多高阶 API 时是否够用未验证。
- UB 上限 248KB 是 DAV_3510 的取值，其他架构需按各自 UB_SIZE 与保留区重查（未验证，机制推断可迁移）。
- 本卡只闭合"预算低估型" 271/507035；若对账无误仍报 271，需另查 API 约束违规（真机不报参数错）或具体标量访问越界。

**未决项**：

- TopK `__ASC_USE_RESERVED_UBUF__` 保留 UB 的确切大小与计算方式未追到（无公开接口）；复现并量化保留区需求是优先接续点。
- 修复轮 1 各单项修改的独立贡献未拆分验证。
## 坏实践

- tiling 预算只记"代表性" buffer，凭感觉留 32KB 余量就上大 M 用例。
- 把随 batchRows 缩放的 buffer（finishBuf_ 等）记进 fixedCost。
- 看到 507035 只按"向量核异常"去查 mask/对齐/同步原语，而不读 plog 的 errcode 271 明示的标量 UB 越界方向。
- 依赖 ASCENDC_ASSERT 在真机拦参数问题。

## 误诊

- 修复轮 1 是四项修改打包（n=topkInner、64KB 余量、finishBuf 对账、softmaxMaxTmp），其中两个大 M+E+k 用例的修复无法归因到单项；基于崩溃特征推断主因是两项 UB 修复，但未经单项拆分验证。
- 曾逐段排查 kernel 自身标量访问（srcIndex.SetValue/offset.SetValue/finishLocal.GetValue）的越界可能，逐项核算均在界内——出界点在 TopK 内部而非用户代码，用户侧只能通过预算让位。
