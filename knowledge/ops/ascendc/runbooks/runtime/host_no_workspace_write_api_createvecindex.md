---
type: "Runbook"
title: "Host 预写 workspace 不可行：CreateVecIndex"
description: tiling/GetWorkspaceSize 阶段没有公开的 workspace 数据写入接口（TilingContext 仅 GetWorkspaceSizes 设大小、aclOpExecutor 仅运行期读取），设计上"Host 预填 workspace"不可实现；内核需要的等差序列改用设备侧 CreateVecIndex 生成，消除 workspace 依赖。
tags: [task.implementation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T02:30:52Z'
updated_at: '2026-09-17T02:30:52Z'
platforms: ["950"]
---

# Host 预写 workspace 不可行：CreateVecIndex

## 现象

开发一个 DAV_3510（Ascend950PR，CANN 9.1.0）双线性插值自定义算子时，设计文档原方案是："由 L0 API 在 Host 侧把 W 方向索引序列 `[0, 1, ..., W_out-1]` 预先写入 workspace，内核启动后从 workspace 读取该序列再计算坐标"。实现阶段排查 CANN 9.1.0 安装头文件，发现这条通路走不通：

- `exe_graph/runtime/tiling_context.h`：`TilingContext` 上 workspace 相关只有 `size_t *GetWorkspaceSizes(const size_t workspace_count)`（设大小）与 `GetWorkspaceNum()`——**没有任何写 workspace 数据的接口**；
- `aclnn/opdev/op_executor.h`：`aclOpExecutor` 侧只有 `GetWorkspaceAddr()/GetWorkspaceSize()/GetLinearWorkspaceSize()/SetWorkspaceOffsets()`（读地址/大小），`make_op_executor.h` 的 `ADD_TO_LAUNCHER_LIST_AICORE` 宏只组装输入/输出/属性参数列表，不含 workspace 数据写入；
- 结论：GetWorkspaceSize 阶段没有公开的 workspace 数据写入通路，"Host 预填 workspace"不可实现。

随后在 `ascendc/include/basic_api/impl/kernel_operator_vec_createvecindex_intf_impl.h` 找到 `CreateVecIndex`，改为设备侧生成索引序列，内核照常编译通过。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），内核需要"下标等差序列"（坐标映射、位置编码、前缀索引等）的场景。
- API 路径：`AscendC::CreateVecIndex(LocalTensor<T> dst, const T &firstValue, uint32_t count)`（支持 half/int16_t/float/int32_t）；tiling 侧 `gert::TilingContext::GetWorkspaceSizes`（仅设大小）。
- 验证过的平台与 CANN 版本：Ascend950PR 编译通过（3 dtype 二进制）+ CANN 9.1.0。
- 适用粒度：operator_family。
- 路线优先级：recommended_fastpath（凡是只需等差序列的场景；任意数据仍需真实 workspace 或属性/输入张量方案）。
- 明确不适用 / 未验证场景：需要 Host 侧动态数据（非可公式化序列）进 kernel 的场景——本卡不提供通路，需另寻 constant tensor / 追加输入张量方案；DAV_3510 以外架构的 CreateVecIndex 支持未逐一验证（未验证，推断可迁移，同为 ascendc 基础向量 API）。

## 根因

ACLNN/tiling 框架把 workspace 定位为"运行期由框架分配、内核自行使用的暂存区"：tiling 阶段只申报大小（`GetWorkspaceSizes`），二段执行阶段框架分配并传地址；GetWorkspaceSize 阶段 Host 侧拿不到可写、且会随 kernel 下发的数据区，公开头文件未暴露写入接口。设计上假设"Host 预填 workspace"是对该生命周期的误解。

## 解决配方

改为设备侧生成：`CreateVecIndex` 一条指令产生 `[0, 1, ..., W_out-1]` 等差序列，后续坐标由向量运算（Muls/Adds/Maxs/Mins）在设备侧合成；tiling 侧只保留系统 workspace。

内核侧（终版 `InitWDirection` 完整函数，含 CreateVecIndex 与 W 方向坐标/权重/字节偏移全链）：

```cpp
template <typename T, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessResizeBilinear<T, BUFFER_MODE>::InitWDirection()
{
    uint32_t wOutU = static_cast<uint32_t>(wOut_);
    float wInMinus1 = static_cast<float>(wIn_ - 1);

    // 1. Generate index sequence [0, 1, ..., W_out-1] as float
    LocalTensor<float> wFloat = wFloatBuf.Get<float>();
    CreateVecIndex(wFloat, 0.0f, wOutU);

    // 2. w[j] = j * scaleW + biasW
    Muls(wFloat, wFloat, scaleW_, wOutU);
    Adds(wFloat, wFloat, biasW_, wOutU);

    // 3. Clamp to [0, W_in-1]
    Maxs(wFloat, wFloat, 0.0f, wOutU);
    Mins(wFloat, wFloat, wInMinus1, wOutU);

    // 4. w0[j] = floor(w[j]) as int32_t
    LocalTensor<int32_t> w0Int = w0Buf.Get<int32_t>();
    Cast(w0Int, wFloat, RoundMode::CAST_FLOOR, wOutU);

    // 5. w0Float = (float)w0 (in tmpCalcBuf, temporary)
    LocalTensor<float> w0Float = tmpCalcBuf.Get<float>();
    Cast(w0Float, w0Int, RoundMode::CAST_NONE, wOutU);

    // 6. w1Float = w0 + 1, clamp to W_in-1 (reuse tmpCalcBuf)
    LocalTensor<float> w1Float = tmpCalcBuf.Get<float>();
    Adds(w1Float, w0Float, 1.0f, wOutU);
    Mins(w1Float, w1Float, wInMinus1, wOutU);

    // 7. w1[j] = (int32_t)w1Float
    LocalTensor<int32_t> w1Int = w1Buf.Get<int32_t>();
    Cast(w1Int, w1Float, RoundMode::CAST_FLOOR, wOutU);

    // 8. beta[j] = w[j] - w0[j]
    LocalTensor<float> beta = betaBuf.Get<float>();
    Sub(beta, wFloat, w0Float, wOutU);

    // 9. oneMinusBeta[j] = 1.0 - beta[j]
    LocalTensor<float> oneMinusBeta = oneMinusBetaBuf.Get<float>();
    Muls(oneMinusBeta, beta, -1.0f, wOutU);
    Adds(oneMinusBeta, oneMinusBeta, 1.0f, wOutU);

    // 10. offset0 = w0 * sizeof(float), offset1 = w1 * sizeof(float)
    //    (byte offsets for Gather from fp32 data)
    Muls(w0Int, w0Int, static_cast<int32_t>(FLOAT_SIZE), wOutU);
    Muls(w1Int, w1Int, static_cast<int32_t>(FLOAT_SIZE), wOutU);

    // w0Int and w1Int now contain byte offsets as int32_t
    // They will be reinterpreted as uint32_t for Gather in ProcessRow

    PipeBarrier<PIPE_V>();
}
```

tiling 侧对应收缩为"只留系统 workspace"（`BenchNoHarnessResizeBilinearTilingFunc` 内，终版完整片段）：

```cpp
    // 3. Set workspace size (system workspace only; index sequence generated on-device via CreateVecIndex)
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = Ops::Base::GetWorkspaceSize(context);
```

## 判别方法

确认"无写 workspace API"这一事实的探针（可直接复用）：

```bash
# TilingContext 上 workspace 相关的全部接口（只有大小/数量，无数据写入）
grep -n "Workspace" $ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/tiling_context.h
# aclOpExecutor 上的 workspace 接口（地址/大小/偏移，均为运行期读取）
grep -n "workspace\|Workspace" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h
# CreateVecIndex 的定义位置与签名
find $ASCEND_HOME_PATH/x86_64-linux/ascendc/include -name "*createvecindex*"
```

## 设计决策

- 为什么选 CreateVecIndex 而不是保留 workspace + 运行期填充：索引序列是纯等差数列，`CreateVecIndex(wFloat, 0.0f, wOutU)` 一条指令设备侧生成，收益是零 workspace 布局协调、零 H2D 搬运、免 Host 侧生命周期纠缠；这是对设计文档的显式偏离（原设计"由 L0 API 在 Host 侧写入"），偏离原因与替代方案已记录。
- 为什么 workspace 大小仍保留系统份 `GetWorkspaceSize(context)`：框架库 API 可能需要内部暂存，归零有风险；最小化保留。
- 当前方案属 recommended_fastpath（等差序列场景）。

## 预防

- 设计阶段就把"Host 预填 workspace"当数据通路前，先 grep `tiling_context.h`/`op_executor.h` 确认接口面——workspace 是"申报大小、框架分配、内核使用"的暂存区，不是 Host→Device 的数据通道。
- 内核需要的序列先问一句"是否等差/可公式化"：是则 `CreateVecIndex` + 逐元素向量运算（Muls/Adds）在设备侧合成，成本近零。

## 验证信号

来源说明：以下信号来自该双线性插值算子开发会话的接口面排查与编译闭环（头文件核查 + 构建产物）。

- 头文件 grep 证实 `TilingContext` 无 workspace 数据写入接口、`aclOpExecutor` 无公开写入通路（来源平台：950 开发环境（CANN 9.1.0 头文件）；证据等级：tool_observed）。
- 改用 CreateVecIndex 后内核 `bash build.sh --soc=ascend950 -j8` 编译通过，3 个 dtype 二进制生成并打包成功（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `CreateVecIndex` 的 dtype 支持面为 half/int16_t/float/int32_t；int32/uint32 以外的位宽（如 int64）未支持，需先 Cast。
- 上板数值语义未在本卡范围单独验证（编译级证据；同会话后续 ST 全量通过见 L2 缓存键卡的联合验证信号）；DAV_2201 等其他架构未实测，未验证，推断可迁移（同为 ascendc 基础向量 API）。
- 若序列不是等差可合成的（任意 Host 数据），本配方不适用。

**未决项**：

- Host 侧动态数据（不可公式化）进内核的正规通路（constant tensor / 追加输入张量）未在本轨迹展开，如遇该需求需专项验证。
## 坏实践

- 把设计文档里的 workspace 数据通路当既定事实直接实现到一半才发现无 API——先核接口面再定方案。

## 误诊

无。
