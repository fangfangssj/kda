---
type: "Runbook"
title: "1D 拷贝循环替代 NDDMA 随路广播 CopyIn 失败"
description: "把 NDDMA 随路广播 CopyIn 改成逐行 1D DataCopyPad/DataCopy 循环后，含内层广播维（stride=0）的用例大面积错值（数十万 mismatch）并触发 errcode (263,82) GM 地址越界；已整体回退，NDDMA 保留。"
tags: [task.debugging, topic.data_movement.data_copy, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 1D 拷贝循环替代 NDDMA 随路广播 CopyIn 失败

## 现象

在调试 softplus_v2_grad RANK_4 kernel 的 `[1,1024]⊗[256,1024]` 广播用例崩溃（507035）时，基于（后来被证伪的）"NDDMA loopSrcStride≥240 硬件约束"假设，尝试把 kernel 的 `CopyInOne` 从 NDDMA `DataCopy<T,ND,cfg>`（随路 broadcast）改为**逐行 1D 拷贝循环**。改动后之前通过的用例大面积翻车：`[1,16,1,56]⊗[8,16,56,56]`（C06，多维广播）9 个 dtype 组合全部 FAIL，`394207/401408 mismatches (first at idx 56)` 量级；C05 崩溃依旧；plog 还出现新的 `errcode:(263, 82)`（"The address for scalar to use is unaligned or out of bounds The GM address exceeds 48 bits..."）。整个方向最终回退：恢复 NDDMA 后 C06 立即恢复 9/9 通过，而 C05 的真正修复在 host tiling 侧（另卡），NDDMA 原样保留即全过。

## 适用场景

- 工程形态：AscendC broadcast 范式 kernel 的 CopyIn 搬运方案选型（NDDMA 随路广播 vs 手写 1D 循环）。
- API 路径：`AscendC::DataCopy<T,ND,cfg>`（NDDMA，`MultiCopyParams`/`loopSrcStride=0` 广播）vs `DataCopyPad`/`DataCopy(dst,src,count)` 1D 路径。
- 验证过的平台与 CANN 版本：ascend950（dav-3510）+ cann-9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（一切含广播输入的 CopyIn 实现）。
- 路线优先级：**banned**（对含内层广播维的输入，1D 连续拷贝循环不可用；如必须走 1D，需按广播维展开重算每个源地址，本卡未验证该替代）。
- 明确不适用/未验证场景：输入 inner 维全部非广播且源连续时，1D 拷贝与 NDDMA 等价（本卡不禁止）。

## 尝试的实现（attempt）

**尝试 A：逐行 1D `DataCopyPad` 循环**（替代整个 NDDMA CopyIn，实际写入代码）：

```cpp
    // CopyIn: GM → buf_[slot], with broadcast support (DESIGN-BRANCH §3)
    // Uses 1D DataCopyPad per contiguous inner block. For broadcast dims
    // (stride=0 on split axis), the same GM block is read multiple times,
    // effectively replicating it in UB. This avoids NDDMA platform issues.
    template <typename T>
    __aicore__ inline void CopyInOne(int slot, int64_t a_i_seg, const int64_t* coord,
        int inputIdx, AscendC::GlobalTensor<T>& gmIn,
        AscendC::MultiCopyParams<T, static_cast<uint8_t>(ND)>& baseParams)
    {
        (void)baseParams;  // NDDMA params not used; 1D DataCopyPad path instead
        int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];
        // split axis stride: 0 for broadcast dims, nonzero for contiguous
        int64_t split_stride = td_->input_strides[inputIdx][td_->split.axis];

        AscendC::DataCopyPadExtParams<T> padParams;  // default: no padding
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = static_cast<uint32_t>(inner_count * sizeof(T));
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        for (int64_t i = 0; i < a_i_seg; i++) {
            int64_t src_off = off + i * split_stride;  // stride=0 → broadcast (same block)
            int64_t dst_off = i * inner_count;
            AscendC::DataCopyPad(buf_[slot].Get<T>()[dst_off], gmIn[src_off],
                                 extParams, padParams);
        }
    }
```

**尝试 B：32B 对齐时 1D `DataCopy(dst,src,count)` 循环 + 未对齐回退 NDDMA**（核心段）：

```cpp
        // 1D DataCopy requires 32B alignment on count
        bool aligned = (inner_count > 0) &&
                       ((inner_count * static_cast<int64_t>(sizeof(T))) % 32 == 0);
        if (aligned) {
            for (int64_t i = 0; i < a_i_seg; i++) {
                int64_t src_off = off + i * split_stride;
                int64_t dst_off = i * inner_count;
                AscendC::DataCopy(buf_[slot].Get<T>()[dst_off], gmIn[src_off],
                                  static_cast<uint32_t>(inner_count));
            }
        } else { /* NDDMA for unaligned cases (scalar, small inner dims) */ }
```

（同批还有三个同类尝试：按精确维数的 1/2/3/4D NDDMA 变体、`per_buf_bytes` 预留 8KB、2 活跃维改 3D NDDMA——均未修复 C05，不再展开。）

## 客观失败观测（observation）

- 尝试 A 后：`ACL error 507035` 从 9 条增至 **21 条**，之前通过的 C06 出现大面积 mismatch（来源平台：950；证据等级：tool_observed）
- 尝试 B 后：ST **29/48**；C06 九个 dtype 组合全部 FAIL（`394207/392965/382858/394224/.../401408 mismatches`，首错 idx=56）；C05 仍 9× 507035 崩溃；plog 新增 `errcode:(263, 82)` 标量/GM 地址越界（来源平台：950；证据等级：tool_observed）
- 回退到 NDDMA 后（同一 tiling）：C06 恢复 9/9 PASS（来源平台：950；证据等级：tool_observed）
- 最终：host 切分轴坐标系修复（另卡）+ NDDMA 原样保留 → 48/48（来源平台：950；证据等级：tool_observed）

## 边界与机制（boundary）

1D 循环的隐含假设是"**切分轴以内（inner）的元素在 GM 连续**、只有切分轴方向需要按 stride 跳"。当 inner 范围内含 **stride=0 的广播维**时假设不成立：C06 的 grad `[1,16,1,56]`（padded `[1,16,1,56]`，split axis=1），inner 范围 = d2×d3 = 56×56 = 3136，但 d2 是广播维（stride=0），grad 真实连续数据只有 16×56 = 896 个元素——按 3136 连续读取越过 tensor 实际数据读到垃圾/越界 → 大面积错值 + GM 地址类 errcode。NDDMA 的随路广播（广播轴 `loopSrcStride=0`）正是为该场景设计，1D 路径无法通过"读同一块 a_i_seg 次"复现 inner 广播（那只能复现**切分轴**上的广播）。

另一教训：这批尝试建立在"loopSrcStride≥240"这一未验证假设上（同轨迹 C06 feat 外层 stride=3136 通过，直接证伪）；先证伪假设、再核算 tile 字节数，就不会走这轮弯路。

## 设计决策

- 回退而非继续修补：1D 路径要正确支持 inner 广播必须按每个广播维展开重算源地址（复杂度等同手写 NDDMA），收益为零；NDDMA 是该范式的正确工具。

## 预防

broadcast 范式 CopyIn 选型：凡输入在 NDDMA 覆盖维内存在 size=1（stride=0）维，一律用 NDDMA 随路广播；改 1D 前先用"inner 范围内有没有 stride=0 的维"这一判别式自检。

## 验证信号

本验证来自该算子 RANK_4 分支调试会话的实测（ST + plog）：

- NDDMA 基线：C06 `[1,16,1,56]⊗[8,16,56,56]` 9/9 PASS、C05 9× 507035（来源平台：950；证据等级：tool_observed）
- 尝试 A（1D DataCopyPad 循环）后：507035 错误 9→21 条，C06 大面积 mismatch（来源平台：950；证据等级：tool_observed）
- 尝试 B（对齐 1D DataCopy + NDDMA 混合）后：ST 29/48；C06 9× FAIL（首行 394207/401408，first at idx=56）；C05 仍 9× 507035；新增 errcode (263,82)（来源平台：950；证据等级：tool_observed）
- 回退 NDDMA 后（同一 tiling）：C06 恢复 9/9 PASS（来源平台：950；证据等级：tool_observed）
- 最终：host 切分轴坐标系修复（另卡）+ NDDMA 原样保留 → ST 48/48、run.sh 退出码 0（来源平台：950；证据等级：tool_observed）

## 失效边界

- 尝试 A 中 C01（无广播）也出现错值（~256× 量级）的具体机制未追（该版本已整体回退）；怀疑 `DataCopyPad` MTE2 方向的 `LocalTensor::operator[]` 偏移或 `DataCopyPadExtParams` 默认值在 3510 上有坑，未验证。
- 本卡结论限于"inner 含广播维"场景；纯连续输入的 1D 拷贝未在本轨迹单独验证。

**未决项**：

- 优先项：如确需绕开 NDDMA（例如真遇到 stride 上限），"按广播维展开的 1D 寻址"替代方案未实现未验证，可作接续点。
- 尝试 A 中无广播用例错值的机制未追（已回退，价值有限）。
## 坏实践

- 用 `src_off = off + i*split_stride` 的逐行 1D 拷贝去复现 inner 广播维（只能复现切分轴广播）。
- 在错误根因假设下连续多轮改搬运方案而不先做假设证伪。

## 误诊

本轮所有尝试的前提"NDDMA loopSrcStride≥240 硬件约束"本身是误诊（详见另卡 [split_axis_padandsqueeze_507035](split_axis_padandsqueeze_507035.md) 的误诊段）。
