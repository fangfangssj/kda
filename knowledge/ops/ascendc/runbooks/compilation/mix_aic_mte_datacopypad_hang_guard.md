---
type: "Runbook"
title: arch35 MIX：AIC 跑 DataCopyPad 挂起加守卫
description: Ascend950PR arch35 MIX 模式下 AIC（Cube 核）执行 DataCopyPad 等 MTE 操作会整体挂起；所有 AIV-only 阶段函数开头必须加 if ASCEND_IS_AIC { return; } 守卫。
tags: [task.debugging, topic.data_movement, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-16T11:11:42Z'
platforms: ["950"]
---

# arch35 MIX：AIC 跑 DataCopyPad 挂起加守卫

## 现象

在 Ascend950PR（DAV_3510 / arch35，MIX 模式 AIC+AIV，taskRation 1:2，AIC=28 核 + AIV=56 核）上调试自定义 conv3d 反传 filter 算子（im2col + Matmul 三阶段 kernel：AIV 生成 wsGrad/wsIm2col → AIC Matmul → AIV Cast 输出）。ST Real 模式 5 个用例全部"流同步失败: ret=507015"后，为定位崩溃点把 kernel 逐步简化：

- Process() 清空（什么都不做，只有两次核间同步）→ kernel 正常完成（输出全 0、rel_err=1.0，但不挂起不报错）。
- 在 Process 里只放一次 `DataCopyPad(ubGrad, gmGrad_[0], copyParams, padParams)`（GM→UB，32 字节，读的是合法输入 tensor）→ 测试 30 秒超时，kernel 挂死，无任何报错 bitmap。
- 在 GM→UB 与 UB→GM 之间补 `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` 显式同步 → 仍然挂起。

当时 kernel 的三个数据搬运阶段（GenerateWsGrad / GenerateWsIm2col / CastAndOutput）没有任何 AIC/AIV 分工判断，MIX 模式下 AIC 与 AIV 都会执行这些函数。

## 适用场景

- 工程形态：AscendC kernel 直调/算子仓开发，arch35（DAV_3510）MIX 模式（coreType: MIX，AIC Cube + AIV Vector 分离）。
- API 路径：`DataCopyPad` / `Duplicate` / `SetValue` 等 MTE/Vector 类操作、`ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏（`kernel_macros.h`、`sys_macros.h`：`#define ASCEND_IS_AIV (g_coreType == AscendC::AIV)`）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——arch35 MIX 模式下所有包含 MTE/Vector 操作的 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：纯 AIV 算子（无 AIC 参与 launch）不需要守卫；非 MIX（纯 Cube / 纯 Vector）模式未验证；其他 arch 的 DataCopyPad 实现是否自带 AIC 早退未逐行核验。

## 根因

arch35（3510）MIX 模式下 AIC（Cube 核）没有 MTE2/MTE3 搬运硬件，执行 DataCopyPad 即挂起；CANN 9.1.0 的 arch3510 GM→UB DataCopyPad 实现路径缺少 `if ASCEND_IS_AIC { return; }` 早退（会话内头文件对比：非 3510 路径的同类重载带有该检查），因此用户 kernel 必须自行把 MTE/Vector 操作限定在 AIV（单句归因）。

## 解决配方

修复前（三个阶段函数无核型判断，AIC/AIV 都执行；GenerateWsGrad 原始形态，节选）：

```cpp
template <typename D_T_X, int GROUPS_MODE>
__aicore__ inline void Conv3dBpFilterKernel<D_T_X, GROUPS_MODE>::GenerateWsGrad()
{
    // 每核负责的行段
    int64_t mPerCore = CeilDiv(M_, coreNum_);
    int64_t mStart = blockIdx_ * mPerCore;
    int64_t mEnd = Min(mStart + mPerCore, M_);
    ...（DataCopyPad 循环体）
}
```

修复后（每个 AIV-only 阶段函数开头加守卫；GenerateWsGrad 完整形态）：

```cpp
template <typename D_T_X, int GROUPS_MODE>
__aicore__ inline void Conv3dBpFilterKernel<D_T_X, GROUPS_MODE>::GenerateWsGrad()
{
    // arch35: DataCopyPad(MTE2/MTE3) 仅在 AIV 执行，AIC 无 MTE 硬件会挂起
    if ASCEND_IS_AIC {
        return;
    }
    // 每核负责的行段
    int64_t mPerCore = CeilDiv(M_, coreNum_);
    int64_t mStart = blockIdx_ * mPerCore;
    int64_t mEnd = Min(mStart + mPerCore, M_);

    // grad 布局：[N, C_out, D_out, H_out, W_out]
    const int64_t gradNStride = cOut_ * dOut_ * hOut_ * wOut_;
    const int64_t gradCStride = dOut_ * hOut_ * wOut_;
    const int64_t gradDStride = hOut_ * wOut_;
    const int64_t gradHStride = wOut_;
    const int64_t wsKStride = dOut_ * hOut_ * wOut_;

    LocalTensor<D_T_X> ubGrad = ubGradBuf_.Get<D_T_X>();
    int64_t wOutBytes = wOut_ * sizeof(D_T_X);

    for (int64_t m = mStart; m < mEnd; m++) {
        int64_t cOutIdx = m;
        for (int64_t n = 0; n < nBatch_; n++) {
            for (int64_t d_o = 0; d_o < dOut_; d_o++) {
                for (int64_t h_o = 0; h_o < hOut_; h_o++) {
                    int64_t gradOffset = n * gradNStride + cOutIdx * gradCStride +
                                         d_o * gradDStride + h_o * gradHStride;
                    int64_t wsOffset = m * K_ + n * wsKStride + d_o * gradDStride + h_o * gradHStride;

                    // GM → UB → GM（DataCopyPad 不支持 GM→GM）
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(wOutBytes), 0, 0, 0};
                    DataCopyPadExtParams<D_T_X> padParams{false, 0, 0, 0};
                    DataCopyPad(ubGrad, gmGrad_[gradOffset], copyParams, padParams);
                    // arch35: UB→GM DataCopyPad 仅用 3 参数（DataCopyExtParams），无 padParams
                    DataCopyPad(gmWsGrad_[wsOffset], ubGrad, copyParams);
                }
            }
        }
    }
}
```

同守卫同样加在另外两个 AIV-only 阶段：

```cpp
template <typename D_T_X, int GROUPS_MODE>
__aicore__ inline void Conv3dBpFilterKernel<D_T_X, GROUPS_MODE>::GenerateWsIm2col()
{
    // arch35: DataCopyPad/Duplicate/SetValue 仅在 AIV 执行
    if ASCEND_IS_AIC {
        return;
    }
    // 每核负责的列段
    ...
```

```cpp
template <typename D_T_X, int GROUPS_MODE>
__aicore__ inline void Conv3dBpFilterKernel<D_T_X, GROUPS_MODE>::CastAndOutput()
{
    // arch35: DataCopyPad/Cast 仅在 AIV 执行
    if ASCEND_IS_AIC {
        return;
    }
    // 每核负责的行段
    ...
```

配套：阶段之间的核间同步用 `SyncAll<true>()`（MIX 模式 AIC+AIV 都参与；`SyncAll<false>` 只同步同类核）。该同步语义来自 arch35 文档结论，本轨迹内无独立前后信号（见误诊）。

## 设计决策

- 守卫放在每个阶段函数开头而不是 Process 里统一判断：三阶段各自可独立增删，Matmul 阶段（AIC 执行）与搬运阶段（AIV 执行）交错时不需要改 Process 结构。
- 为什么读输入也挂：AIC 上连 GM→UB 读都执行不了（无 MTE 硬件），与地址合法性无关——所以不能靠"只读合法 tensor"规避。

## 系统信息

- Ascend950PR 单卡（npu-smi 25.7.rc1，Device 0 OK，131072 MB HBM），CANN 9.1.0；平台核数 AIC=28 / AIV=56，UB=248KB。
- 相关文件：`op_kernel/arch35/<op>.h`（守卫落点）；CANN 头 `asc/include/basic_api/impl/kernel_operator_data_copy_intf_impl.h`（重载面参照）。

## 预防

- arch35 MIX kernel 的编写清单：先按 AIC/AIV 划分阶段（Cube 计算 → AIC；MTE 搬运/Vector 计算 → AIV），每个 AIV-only 函数开头加 `if ASCEND_IS_AIC { return; }`，AIC-only 逻辑加 `if ASCEND_IS_AIV { return; }`，跨阶段依赖用 `SyncAll<true>()`。
- "kernel 无报错但 stream 同步超时挂死"且逐步清空后只剩一次 DataCopyPad 也挂 → 第一嫌疑是 AIC 执行了 MTE 操作。

## 验证信号

以下信号来自该算子在 Ascend950PR 上的挂起定位闭环（CANN 9.1.0）：

- 空 Process（仅核间同步）：kernel 正常完成，输出全 0（rel_err=1.0），无挂起无报错（来源平台：950；证据等级：tool_observed）。
- 单次 GM→UB DataCopyPad（无守卫，AIC 也执行）：30 秒超时挂死；加 MTE2_MTE3 SetFlag/WaitFlag 同步后仍挂死（来源平台：950；证据等级：tool_observed）。
- 三个阶段函数加 `if ASCEND_IS_AIC { return; }` 守卫后：kernel 不再挂起，120 秒内完成并产出结果（后续失败转为快速报错 ret=507015，为另一独立根因）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "AIC 无 MTE 硬件故挂起"的机制层解释是头文件对比 + 行为实验的推断，未做指令级验证。
- CANN 后续版本若在 arch3510 DataCopyPad 实现里补上 AIC 早退，守卫变为冗余但无害（未验证，推断可迁移）。
- 本卡只解决挂起；守卫后的 ret=507015（workspace 参数错位）与 Matmul AIV 异常是独立根因，另行闭环。

**未决项**：

- 【优先】逐行核验 CANN 9.1.0 arch3510 DataCopyPad 全部重载是否都缺 AIC 早退（只对比了 GM→UB 路径），并向 CANN 侧反馈。
## 坏实践

- 在 GM→UB 与 UB→GM 之间堆 MTE2_MTE3 同步标志试图"修"挂起——挂起根因是核型不是同步。
- MIX kernel 里默认"所有核都能跑所有 AscendC API"。

## 误诊

- 挂起出现时先怀疑数据依赖/同步缺失，加了 SetFlag/WaitFlag 无效后才用"空 kernel / 单 DataCopyPad"二分定位到核型问题。
- 同批落地的 `SyncAll<false>`→`SyncAll<true>` 修改在本轨迹内没有独立的前后信号（改完 ST 仍 507015），其语义依据是 arch35 文档（`SyncAll<true>` 为 MIX 模式 AIC+AIV 同步），不要把 ST 后续变化归因于它。
