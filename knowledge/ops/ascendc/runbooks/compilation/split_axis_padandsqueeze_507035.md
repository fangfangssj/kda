---
type: "Runbook"
title: "split.axis 未随 PadAndSqueeze 偏移致 507035"
description: "DoTilingAndSet 在 PadAndSqueeze 压缩后 shape 上跑 FindSplitAxis 得 split.axis，再前补 1 填充到 R 维时未把 split.axis 偏移 delta；rank<R 时切分轴指向 size-1 前导维，inner_count 爆涨越界 UB 触发 507035。可用跨用例 stride 对比证伪 loopSrcStride≥240 误诊、并用 host split.axis 与 R 维 shape 检查判别。R2（9.1.0 复现轮）补齐修复：DoTilingAndSet 改为先 pad 后 split（host 一处改动、kernel 零改动），修后 ST 47/48→48/48、TilingUT 无回归。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T01:42:07Z"
platforms: ["950"]
---

# split.axis 未随 PadAndSqueeze 偏移致 507035

## 现象

在 Ascend950（dav-3510/950，CANN 9.1.0）上为 `softplus_v2_grad` 算子实现 RANK_4 分支 Kernel 后跑 aclnn ST 用例：38/48 通过、10 个失败。其中 9 个是 C05 形状 `[1,1024]⊗[256,1024]`（grad 广播到 feat），覆盖全部 9 个 dtype 组合，`aclrtSynchronizeStream` 返回 **error 507035（vector core exception）**，kernel 直接崩溃，无 mismatch 输出。开发者日志把根因记为「NDDMA `loopSrcStride≥240` 平台约束致 UB 越界」（2D NDDMA split axis stride 1024 超约束 `[0,240)`），建议改用 1D `DataCopyParams` 分段搬运绕过 NDDMA stride 约束。

复调时发现该诊断 **错误**：同批次 C06 `[1,16,1,56]⊗[8,16,56,56]` 的 `loopSrcStride=3136` 反而通过——若约束真是 `stride<240`，C06 不可能过。真正根因在 host 侧 `DoTilingAndSet` 的坐标系不一致，与 NDDMA stride 无关。

R2（2026-09 独立复现会话，同算子重新实现 RANK_4 分支，kernel 为 NDDMA 随路广播 CopyIn + TBuf 四缓冲 + `__simd_vf__` VF 链）：同一失败画像复现——48 条 aclnn ST 用例 38 通过、10 失败，其中 9 条 `[1,1024]⊗[256,1024]` 广播用例（9 dtype 组合全挂）在 `aclrtSynchronizeStream` 报 507035，plog 显示 `errcode:(82, 341)` "The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds. The address for VEC to access UB is out of bounds."，AIC_INFO 中 `blockDim=18`，异常核 core_id=42/43/44（越界核问题另卡）。其余 1 条失败是 NaN 用例（另一根因，另卡）。当时代码状态：host 侧 `DoTilingAndSet<R>()` 在 PadAndSqueeze 产出的 **squeezed** `maximum_bro_shape_` 上调用 `FindSplitAxis`，之后才前补 1 到 R=4 维填 TilingData 的 shape/stride 字段。本轮完成修复并全量验证（见解决配方）。

## 适用场景

- **工程形态**：op_host 侧 TilingFunc 实现（aclnn 算子仓开发，ops-registry-invoke 工作流；broadcast 范式、TilingData 按 RANK 模板化分档）。
- **API 路径**：`IMPL_OP_OPTILING` → `DoTilingAndSet<kRank>` → 公共 `FindSplitAxis`/`MultiCoreSplit`/`PrecomputeInputStrides`，kernel 侧读 `td->split.axis` 决定 NDDMA `loopSize/loopSrcStride/loopDstStride` 与 `inner_count`。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510/950），CANN 9.1.0（实测 UB=253952 B，非标称 256 KB）。
- **适用粒度**：`operator_family`——任何走「PadAndSqueeze 压缩 → 再前补 1 填充到 R 维」的广播算子 TilingFunc 都可能踩此坑（一切"host 在一个坐标系算切分、kernel 在另一个坐标系消费"的实现）。
- **不适用/未验证场景**：rank==R（delta=0）天然不受影响（C01/C02/C06/C07 等 rank=4 用例即因此通过）；rank<R 但 split.axis 恰好落在真实大维的偶发场景可能不崩但搬运错。a2/a3 未验证（推断可迁移）。

## 判别方法

1. **触发信号**（tool_observed）：aclnn ST 在广播用例上 `aclrtSynchronizeStream` 返回 507035（vector core exception），kernel 直接崩溃、无 mismatch 输出；失败系统性集中在 rank<R（PadAndSqueeze 后被压缩）的广播形状上。
2. **可执行探针**：
   - **跨用例 stride 对比**：取同批次一个通过用例，核对其 NDDMA `loopSrcStride`。若通过用例的 stride 远大于「误诊约束」（如 C06 的 3136 >> 240）却仍 PASS，即可证伪「stride<240 平台约束」假设，排除 NDDMA stride 为根因。
   - **host 坐标检查**：在 `DoTilingAndSet` 内打印 `td->split.axis` 与填充到 R 维后的 `max_bro_shape`，核对 `split.axis` 是否落在 size 为 1 的前导填充维（即 `max_bro_shape[split.axis] == 1` 且 `split.axis < delta`）。
   - **R2 DBG 探针**（在 `DoTilingAndSet` 填 TilingData 处临时加 fprintf，打印 `ub_size / per_buf / split.axis / a_i / a_o / rank`，再跑失败用例）：
     ```text
     [DBG] R=4 ub=253952 per_buf=63488 dt=4 split.axis=0 a_i=15 a_o=18 a_i_tail=1 rank=2
     ```
     广播结果 `[256,1024]`（有效 rank=2）却得到 `split.axis=0`——这是 **squeezed 坐标系**的轴；前补 1 到 4D 后（`[1,1,256,1024]`）正确切分轴应为 axis=2。同时探针确认平台实际 `ub_size=253952` 字节（不是 256KB 整），`per_buf=(253952/4)&~31=63488B=15872` 个 FP32 元素。
3. **原始观测**（tool_observed）：507035 系统性出现在 PadAndSqueeze 后 rank<R 的广播用例（C05 rank=2）上；rank==R 的用例（C06 rank=4）通过。该通过/失败分布与 `loopSrcStride` 数值大小不相关（C06 stride=3136 通过、C05 stride=1024 反崩），直接否定 stride 约束假设。
4. **解释**（root-cause 分析）：`DoTilingAndSet` 坐标管线不一致——先对输入 `PadAndSqueeze` 得压缩后 `max_bro_shape_`（C05 得 `[256,1024]`，rank=2），在**压缩** shape 上 `FindSplitAxis` 得 `split.axis`（C05 得 0），**最后**才前补 1 填充到 4D（`delta = R - rank = 2`），但 **`split.axis` 未同步加 `delta`**。于是 C05 的 `td->split.axis=0` 在填充后 4D `[1,1,256,1024]` 中指向 size-1 前导维；kernel 读 `inner_count = product(max_bro_shape[split.axis+1 .. R-1]) = 1*256*1024 = 262144`（应为 1024），`tile_elems = a_i_seg * inner_count ≈ 15*262144` 远超单 buffer 15872 元素 → UB 大规模越界 → 507035。这同时解释 C06 通过：C06 压缩后 rank=4、`delta=0`，split.axis 无需偏移，坐标系天然一致。关键换算：`per_buf_bytes = (ub/physNodes) & ~31 = (253952/4)&~31 = 63488`；`per_buf_elems = 63488/4 = 15872`；`a_i = 15872/1024 = 15`。为何只有 rank<R 且需真实切分的用例暴露：小 shape 低 rank 用例因 FindSplitAxis 落入 k==0 兜底且整块能装入单个 tile，恰好不触发。

## 解决配方（R2：先 pad 后 split）

host `DoTilingAndSet<R>()` 改为**先 pad 后 split**：先把 `max_bro_shape_`/`normal_in_`/`normal_out_` 前补 1 到 R 维，再在 padded 形状上调用 `FindSplitAxis`/`MultiCoreSplit` 并在 padded 形状上计算 stride（stride 数值与旧实现等价：旧实现对 j<delta 手工填 0，等于 padded 形状上 `Precompute*Strides` 对补 1 维算出的 0）。这与 TilingUT 已覆盖的 `DoRank4Tiling`（先 PadTo4 再 FindSplitAxis）保持一致。修复仅动 host tiling 一处；kernel 的 NDDMA CopyIn 原样保留。

before（`op_host/softplus_v2_grad_host.cpp` 旧版核心段，先 split 后 pad、坐标系混合）：

```cpp
        // §5.4 FindSplitAxis / MultiCoreSplit
        softplus_v2_grad_tiling::FindSplitAxis(
            max_bro_shape_, dtype_size_, ub_size_, kPhysNodes, td->split);
        softplus_v2_grad_tiling::MultiCoreSplit(
            max_bro_shape_, td->split, core_num_, td->multicore);

        td->per_buf_bytes = (ub_size_ / kPhysNodes) & ~31LL;
        td->rank = rank_;
        td->num_inputs = kMaxInputSlots;
        td->num_outputs = kMaxOutputSlots;
        td->beta = beta_;
        td->threshold = threshold_;

        // 前补 1 到 R (delta = R - rank_) 填充 max_bro_shape / input / output 的 shape 与 stride
        const int64_t delta = R - rank_;
        for (int64_t j = 0; j < R; ++j) {
            td->max_bro_shape[j] = (j < delta) ? 1 : max_bro_shape_[j - delta];
        }
        for (int64_t i = 0; i < kMaxInputSlots; ++i) {
            std::vector<int64_t> strides;
            softplus_v2_grad_tiling::PrecomputeInputStrides(normal_in_[static_cast<size_t>(i)], strides);
            for (int64_t j = 0; j < R; ++j) {
                td->input_shapes[i][j]  = (j < delta) ? 1 : normal_in_[static_cast<size_t>(i)][j - delta];
                td->input_strides[i][j] = (j < delta) ? 0 : strides[static_cast<size_t>(j - delta)];
            }
        }
```

after（新版核心段：统一在 padded 坐标系计算，split.axis 与 kernel 解释一致）：

```cpp
        td->per_buf_bytes = (ub_size_ / kPhysNodes) & ~31LL;
        td->rank = rank_;
        td->num_inputs = kMaxInputSlots;
        td->num_outputs = kMaxOutputSlots;
        td->beta = beta_;
        td->threshold = threshold_;

        // 前补 1 到 R 维 (delta = R - rank_)：max_bro_shape / input / output 的 shape 与 stride
        // 统一在 padded 坐标系计算，保证 split.axis 与 kernel 解释一致。
        const int64_t delta = R - rank_;
        auto padLeft = [&](const std::vector<int64_t>& s) {
            std::vector<int64_t> p(static_cast<size_t>(delta), 1);
            p.insert(p.end(), s.begin(), s.end());
            return p;
        };
        std::vector<int64_t> broR = padLeft(max_bro_shape_);

        // §5.4 FindSplitAxis / MultiCoreSplit — 在 padded R 维坐标系上执行
        softplus_v2_grad_tiling::FindSplitAxis(
            broR, dtype_size_, ub_size_, kPhysNodes, td->split);
        softplus_v2_grad_tiling::MultiCoreSplit(
            broR, td->split, core_num_, td->multicore);

        for (int64_t j = 0; j < R; ++j) {
            td->max_bro_shape[j] = broR[static_cast<size_t>(j)];
        }
        for (int64_t i = 0; i < kMaxInputSlots; ++i) {
            std::vector<int64_t> si = padLeft(normal_in_[static_cast<size_t>(i)]);
            std::vector<int64_t> st;
            softplus_v2_grad_tiling::PrecomputeInputStrides(si, st);
            for (int64_t j = 0; j < R; ++j) {
                td->input_shapes[i][j]  = si[static_cast<size_t>(j)];
                td->input_strides[i][j] = st[static_cast<size_t>(j)];
            }
        }
        // 输出侧同理：so = padLeft(normal_out_[i]); PrecomputeOutputStrides(so, st); 逐 j 填 td->output_shapes/strides
```

设计决策：选择「先 pad 后 split」而不是「split.axis 加 delta 偏移」——偏移方案只救 split.axis 一个字段，而 MultiCoreSplit 的 outer 维度乘积同样依赖坐标系；统一在 padded 坐标系重算一并消除两类错位，且与已被 TilingUT 独立 oracle 交叉校验的 `DoRank4Tiling` 实现路径完全一致。修复属 recommended_fastpath：host 一处改动、kernel 零改动、可被 TilingUT 复用覆盖。

## 验证信号

- `cat docs/softplus_v2_grad/develop/LOG.md` 记载首实现 ST 38/48 PASS、10 FAIL，9× C05 `[1,1024]⊗[256,1024]` 报 `aclrtSynchronizeStream` error 507035，并记前序 `loopSrcStride≥240` 误诊（来源平台：950；证据等级：tool_observed）。

R2 轮（独立复现会话，aclnn ST + TilingUT + plog）增量信号：

- R2 修复前 ST 38/48，其中 9× `[1,1024]⊗[256,1024]` 广播用例全部在 `aclrtSynchronizeStream` 报 507035（来源平台：950；证据等级：tool_observed）。
- R2 plog：`errcode:(82, 341)` "The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds"（来源平台：950；证据等级：tool_observed）。
- R2 host DBG 探针：广播结果 `[256,1024]`（有效 rank=2）得到 `split.axis=0 a_i=15 a_o=18`（squeezed 坐标系错位证据），`ub_size=253952 per_buf=63488`（来源平台：950；证据等级：tool_observed）。
- R2 仅应用本修复后 ST 47/48，9× 广播用例全部通过（来源平台：950；证据等级：tool_observed）。
- R2 叠加 NaN 测试数据修复（独立卡）后 run.sh 48/48、退出码 0；公共 TilingUT 27/27 + RANK_4 TilingUT 12/12 无回归；build.sh 退出码 0（来源平台：950；证据等级：tool_observed）。

## 误诊

前序日志把 9× C05 崩溃归因为「NDDMA `loopSrcStride≥240` 平台约束」，建议「改用 1D `DataCopyParams` 分段搬运绕过 NDDMA stride 约束」。该归因 **错误**：C06 的 `loopSrcStride=3136`（远大于 240）却能通过，证伪「stride<240」假设。真正根因是 host 侧 split.axis 坐标系未随 PadAndSqueeze→R 维填充偏移 `delta`，导致 kernel 内 `inner_count` 爆涨、UB 越界。按误诊的 1D 分段搬运方案去改 kernel 属治标（绕开 NDDMA），治本应修 host `DoTilingAndSet` 坐标系。

R2 会话在同一误诊上走了更多弯路：基于「loopSrcStride≥240」假设连续 5 次绕过性修改 kernel 搬运代码（含以 1D 拷贝循环替代 NDDMA 随路广播，见独立卡 [errcode_263_82_nddma_broadcast_1d_loop_banned](errcode_263_82_nddma_broadcast_1d_loop_banned.md)）全部无效后回退；正确路径是先用探针打印 tiling 实际值、再算 tile 字节数与 per_buf 的比值。

## 预防

- broadcast 范式 tiling 的坐标系不变量：PadAndSqueeze 产物（squeezed）与前补 1 到模板 RANK（padded）是两个坐标系；`FindSplitAxis`/`MultiCoreSplit`/stride 计算与 kernel 侧消费必须在**同一**坐标系完成。
- host 内联 tiling 实现易与 TilingUT 覆盖的独立实现分叉——本例 TilingUT 12/12 全绿但 host 带错（UT 测的 `DoRank4Tiling` 不是 host 里那份 `DoTilingAndSet`）。host 应复用同一实现，或增加 host tiling 与 UT oracle 的端到端对拍。
- 平台 UB 大小必须运行时查询（本机 TilingParse `GetCoreMemSize` 返回 253952B，非 256KB 整）；若 tiling 硬编码 256KB，边界 shape 会再次越界。

## 失效边界

- 仅 Ascend950（CANN 9.1.0，UB=253952）实测；a2/a3 未验证，但「PadAndSqueeze 后 split.axis 坐标系偏移」是纯 host 逻辑问题，与平台无关，推断可迁移，落库前需在 a2/a3 复跑 C05 类用例确认。
- rank==R（delta=0）的算子天然不受影响。
- split.axis 深层的 `inner_count`/`tile_elems` 数值（262144、≈3.9M、15872）来自会话内 host debug-print 推算，非 trace 的 tool_result 直采，仅作解释，不作为独立可核信号；R2 轮的 DBG 探针（split.axis=0、a_i=15、per_buf=63488）为 tool_observed 直采，佐证上述机制链数值。
- 探针结论 `ub_size=253952` 是本机 TilingParse 查询值；其他机器/其他 CANN 版本的 UB 保留量未验证。

## 坏实践

- 在 host 侧同时维护「压缩坐标系」与「R 维坐标系」两套 shape，但只对 shape 做填充、忘了对 `split.axis` 做对应 `delta` 偏移——这是坐标管线不一致的典型坑。
- 用「在 kernel 里 1D 分段搬运绕过 NDDMA stride 约束」去掩盖 host tiling 坐标 bug：治标不治本，且会引入不必要的 kernel 复杂度。
- 在未验证假设上连环改 kernel：先入为主认定 stride 约束类假设并连续多轮改搬运方案，而不先做假设证伪与 tile 字节数核算。
