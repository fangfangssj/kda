---
type: "Runbook"
title: "双AIV CrossCoreFlag 的 subBlock 契约缺陷"
description: "Ascend950 双 AIV 融合 kernel 中，AIC 为 V0/V1 设置不同 flag 段，而 AIV wait 未按 subBlock 偏移，且空分支可能不回传 flag；源码依赖图可确认同步缺陷，但单独修正后 507015 仍存在，不能把整起异常只归因于该缺陷。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.programming_model.cooperative_groups"]
status: stable
sources: []
created_at: "2026-08-18T08:56:31Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# 双AIV CrossCoreFlag 的 subBlock 契约缺陷

## 现象

在 Ascend 950（DAV_3510，`--npu-arch=dav-3510`）上开发 GroupedMatmul Blaze 算子（G-B 动态量化，FP8_E4M3FN×FP8_E4M3FN→FP32→BF16），使用 `__mix__(1,2)` = 1 AIC + 2 AIV 双 AIV 模式。测试用例 TC1（M=18, K=2048, N=4096, E=32）编译通过，运行可执行文件后 `aclrtSynchronizeStream` 失败，错误码 `507015 = ACL_ERROR_RT_AICORE_EXCEPTION`。

查 plog（`/root/ascend/log/debug/plog/plog-202163_*.log`）关键信息：`sqe_type=0(aic)`，`errorStr: timeout or trap error. subErrType: 0x4`；所有 core 停在同一 PC `0x12004100205c`（典型同步死锁，flag 等不到）；`mte error info`/`l1 error info` 非零，`cube error info: 0`（不是 MMAD 越界，是 MTE 搬运或 flag 链断裂）。

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke Blaze/blaze_custom 路线，DAV_3510/Ascend 950PR）。
- **API 路径**：`AscendC::CrossCoreSetFlag` / `CrossCoreWaitFlag`（MODE_4）、`GetSubBlockIdx()`、`GetTaskRation()`；`cv_sync_constants.h` 的 `AIC_SYNC_AIV_FLAG`/`AIV_SYNC_AIC_FLAG`/`FLAG_ID_MAX`/`COUNT_ID_MAX`/`COUNT_FLAG`。
- **验证过的平台与 CANN 版本**：Ascend 950（DAV_3510），`--npu-arch=dav-3510`。
- **适用粒度**：`operator_family`——所有 `__mix__(1,2)` 双 AIV C+V 融合算子的 CrossCore flag 同步。
- **路线优先级**：`diagnostic_only`——本卡用于审查双 AIV 的 set/wait 对称性；不把该缺陷写成 507015 的完整根因。
- **不适用或未验证场景**：单 AIV 模式（`__mix__(1,1)`）无 V1 sub-block 区分问题，不适用；A3/A2 平台未验证（见 `## 失效边界`）。

## 判别方法

### 触发信号（tool_observed 失败症状）

`__mix__(1,2)` 双 AIV 模式下 kernel 编译通过但运行立即报 `507015 = ACL_ERROR_RT_AICORE_EXCEPTION`，`aclrtSynchronizeStream` 失败。plog 呈现「所有核停在同一 PC + timeout or trap + cube error info=0」三联征——即同步死锁而非 MMAD 越界。失败发生在 AIC↔AIV 的 CrossCore flag 同步链上。

### 可执行探针

1. **定位错误类型**：取 plog（`/root/ascend/log/debug/plog/plog-202163_*.log`），grep `sqe_type|errorStr|subErrType|current: 0x|core id is|mte error info|l1 error info|cube error info`。确认 `sqe_type=0(aic)`、`errorStr: timeout or trap error`、`subErrType: 0x4`。
2. **判断是否同步死锁**：grep `current: 0x` 取所有 core 停留 PC，若**所有核停在同一 PC** 且 `cube error info=0`，倾向同步死锁（flag 等不到）而非 MMAD 越界（后者 `cube error info` 非零）。
3. **区分 AIC/AIV 核**：grep `core id is [0-9]+` 统计 core_id 分布。在 `__mix__(1,2)` 下按 block 内核编号区分 AIC（cube 核）与 AIV（vector 核，含 V0/V1 两个 sub-block），确认是否所有 AIC+AIV 核都卡在同一 wait 点（全核互锁）。
4. **核对数据分布**：读 `gen_data.py` 的 `make_group_list`，确认是否走 `M < E` 分支（前 M 个 group 各 1 token、后 `E-M` 个 group 为空），以及所有非空 group 的 `curM` 是否为奇数（`curM` 奇数时 `halfM=CeilDiv(curM,GetTaskRation()=2)`，V1 `localRows=halfM-1`，curM=1 时 V1 `localRows=0`）。
5. **比对 flag 同步代码**：读 kernel 的 AIV 分支与 epilogue，核对 `CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG + countId)` 是否**未按 `GetSubBlockIdx()*FLAG_ID_MAX` 偏移**区分 V0/V1，以及 epilogue `localRows<=0` early return 路径是否**不发 `CrossCoreSetFlag(AIV→AIC)`** 就 return。

### 原始观测（Ascend 950，DAV_3510）

- 运行可执行文件：`aclrtSynchronizeStream` 失败，错误码 `507015 = ACL_ERROR_RT_AICORE_EXCEPTION`。
- plog：`sqe_type=0(aic)`，`errorStr: timeout or trap error. subErrType: 0x4`；所有 core 停在同一 PC `0x12004100205c`；`mte error info`/`l1 error info` 非零，`cube error info: 0`。
- 修复 sync flag 后重跑：仍 `507015`，同 PC `0x12004100205c`，全核 timeout（serial number 递增，core_id 1/2/4/5… 即 AIV V0/V1 均卡）。
- `gen_data.py`（M=18, E=32）：走 `M < E` 分支，前 18 个 group 各 1 个 token、后 14 个 group 为空；所有非空 group `M=1`（奇数）。

### 同步契约分析

`__mix__(1,2)` 双 AIV 下，AIC 每个 K-block set 两个 flag：`AIC_SYNC_AIV_FLAG + countId`（即 `8+countId`，给 V0）与 `+ FLAG_ID_MAX`（即 `24+countId`，给 V1），间隔 `FLAG_ID_MAX=16`。设计意图是 V0 用 `[base, base+FLAG_ID_MAX)`、V1 用 `[base+FLAG_ID_MAX, ...)` 两段不重叠 flag 池。

但 kernel AIV 分支里 V0 和 V1 都 `CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG + countId)`，**没有按 `GetSubBlockIdx() * FLAG_ID_MAX` 偏移区分**：AIC 只 set 1 次 `8+countId` 被 V0 消费，V1 也 wait `8+countId` 但 AIC 不会再 set 第二次 → V1 永久阻塞；而 `24+countId` 被 AIC set 却从不被 wait（orphan 累积）。V1 阻塞后，AIC 下一轮 `CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX)` 在等 V1 的 flag 也阻塞 → 全核 timeout 死锁。

叠加第二条链：epilogue `localRows<=0`（curM=1 奇数时 V1 `localRows=0`）直接 `return`，**不发 `CrossCoreSetFlag(AIV→AIC)`**，但 AIC 侧在等 V1 的 `AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX` → AIC 也阻塞。两条链互相等待，与 plog「所有核同一 PC + timeout + cube error=0」完全吻合。

修法方向（仅描述，不内联逐字 after 代码）：① `cv_sync_constants.h` 定义 `FLAG_ID_MAX=16` 为双 AIV flag 间距；② kernel AIV 分支 wait flag 按 `subBlockFlagOffset = GetSubBlockIdx() * FLAG_ID_MAX` 区分；③ kernel 传入 epilogue 的 flagId 加 `subBlockFlagOffset`；④ epilogue `localRows<=0` 时仍发 `CrossCoreSetFlag` 再 return；⑤ epilogue 末尾每个 AIV sub-block 只 set 自己的 flag（`AIV_SYNC_AIC_FLAG + countId + subBlockFlagOffset`）。

**证据边界**：flag 编号不对称和 early return 漏回传可由源码依赖图直接识别；但单独修改同步代码后 507015 未消除，且后续日志出现独立的 vector UB 对齐异常。因此本卡只确认同步契约缺陷，不能声称它解释了该次 507015 的全部运行期表现。

## 误诊

初次修改 sync flag 后运行仍报 507015。继续按 core id 检查发现另一个 AIV core 的 PC 与错误类型不同，并有 vector error 信息，说明同一错误码下存在独立异常。修同步契约后必须重新按 core/PC/error-info 分流，不能用错误码未变判断修改是否有效，也不能在没有完整回归时宣布闭环。

## 预防

- `__mix__(1,2)` 双 AIV 模式下，所有 CrossCore flag 的 set/wait 必须按 `GetSubBlockIdx() * FLAG_ID_MAX` 偏移区分 V0/V1，V0 用 `[base, base+FLAG_ID_MAX)`、V1 用 `[base+FLAG_ID_MAX, ...)`。
- epilogue 的 early return 路径（`localRows<=0`、空 group）必须在 return 前发送所有对端等待的 flag，否则对端永久等待（空闲核必须在 return 前发 flag）。
- 测试用例设计应覆盖 `M<E`（空 group）与 `M` 为奇数（curM 奇数致 V1 `localRows=0`）的边界，暴露 V1 不参与同步时的 flag 链断裂。

## 坏实践

- AIV 侧 V0/V1 共用同一 `CrossCoreWaitFlag(flagId)` 不按 sub-block 偏移——AIC 只 set 1 次被 V0 消费，V1 永久阻塞。
- epilogue `localRows<=0` 直接 `return` 不发 `CrossCoreSetFlag`——AIC 侧对应 wait 永远等不到。
- 修第一层 sync flag 后见同症状复发就回滚或认定修复无效——实际可能是多层根因叠加，须按 core_id 分流 aicore/aivec error 再判。

## 验证信号

- 运行可执行文件：`aclrtSynchronizeStream` 失败，错误码 `507015 = ACL_ERROR_RT_AICORE_EXCEPTION`（来源平台：950；证据等级：tool_observed）
- plog：`sqe_type=0(aic)`，`errorStr: timeout or trap error. subErrType: 0x4`，所有核停在同一 PC `0x12004100205c`，`mte/l1 error info` 非零、`cube error info=0`（来源平台：950；证据等级：tool_observed，plog grep）
- 修复 sync flag 后重跑仍 `507015`、同 PC `0x12004100205c`、全核 timeout（来源平台：950；证据等级：tool_observed，复发复现）
- `gen_data.py`（M=18, E=32）走 `M<E` 分支，前 18 个 group 各 1 token、后 14 个为空，所有非空 group `M=1`（来源平台：950；证据等级：artifact_verified，gen_data.py 分支逻辑 + curM 推导）

## 失效边界

- **已验证**：Ascend 950（DAV_3510），`__mix__(1,2)` 双 AIV，TC1（M=18, K=2048, N=4096, E=32）触发 507015 死锁。
- **未验证，推断可迁移**：A3/A2 平台——`__mix__` 双 AIV 模式在 Ascend 系列中行为一致，但 flag 编号与 `FLAG_ID_MAX` 间距需按平台核对，未上板验证，不写入 `platforms`。
- **不适用**：单 AIV 模式（`__mix__(1,1)`）无 V1 sub-block 区分问题。
- **多故障并存**：同 trace 还出现独立的 vector UB 对齐异常；完整 507015 闭环未在本卡建立。