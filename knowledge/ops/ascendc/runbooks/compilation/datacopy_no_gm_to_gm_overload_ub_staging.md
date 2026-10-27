---
type: "Runbook"
title: DataCopyPad 无 GM→GM 重载需 UB 中转
description: AscendC 的 DataCopyPad/DataCopy 全部重载至少一侧为 LocalTensor（UB），不存在 GM→GM 直接搬运；设计文档按文档示例声称"GM→GM DataCopyPad 已验证"会导致实现无法落地，正确做法是 TBuf<VECCALC> 暂存 + GM→UB→GM 分块中转。
tags: [task.implementation, topic.data_movement.data_copy, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# DataCopyPad 无 GM→GM 重载需 UB 中转

## 现象

为 Ascend950PR（DAV_3510 / arch35，CANN 9.1.0）开发 scatter 型算子的迭代一骨架：设计文档（DESIGN.md §3.6 API 映射方案）把 Phase 1 的 `data → y` 批量拷贝写成 GM→GM 直接搬运，并在 §3.7 标注"已验证"：

```
| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 |
| Phase1: data→y 批量拷贝 | DataCopyPad<T> | (dst GlobalTensor, src GlobalTensor, DataCopyExtParams{1, len, 0, 0, 0}) | ✅ DAV_3510 | GM 侧无对齐约束；连续搬运 blockLen 可任意 |
```

按此写 kernel 时，开发者对 `asc-devkit/include/basic_api/kernel_operator_data_copy_intf.h` 里 DataCopyPad 的全部重载做了 grep 核对，发现**没有任何一个是 (GlobalTensor, GlobalTensor)**：全部重载至少一侧为 LocalTensor。对 DataCopy 的核对同样如此。也就是说设计里的"GM→GM DataCopyPad"这个 API 形态不存在，Phase 1 按原设计写无法编译。设计文档的"✅ 已验证"标注是按 API 文档示例与 add_example 用法推断的，并没有对重载签名做过核对。

## 适用场景

- 工程形态：手写 AscendC kernel（registry-invoke / kernel 直调），需要在 GM 侧做整段拷贝（如 y = data 的输出初始化、workspace 中转）。
- API 路径：`asc-devkit/include/basic_api/kernel_operator_data_copy_intf.h` 的 `DataCopyPad` / `DataCopy` 重载族；`TBuf<TPosition::VECCALC>` + `TPipe::InitBuffer` + `PipeBarrier<PIPE_MTE2>`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（头文件核对 + `--soc=ascend950` 全量编译通过；与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本的 AscendC API 面（重载表）与平台无关地不存在 GM→GM 形态，任何算子都适用。
- 路线优先级：recommended_fastpath（kernel 内整段拷贝的唯一可行形态是经 UB 中转）。
- 明确不适用 / 未验证场景：GM→GM 若经由框架层（如 L2 API 的 Tensor 拷贝、aclrtMemcpy）不在本卡范围；其他 CANN 版本的重载面未核对（未验证）；单元素 GM 读写用 GetValue/SetValue 组合是另一条路线（本算子 Phase 2 采用）。

## 根因

DataCopyPad/DataCopy 是 MTE2（GM→UB）/MTE3（UB→GM）搬运指令的封装，语义上就以 UB（LocalTensor）为一级中转，因此头文件里的全部重载至少一侧是 LocalTensor；GM→GM 直接搬运在 AscendC API 中不存在。设计文档把 API 文档中"(dst, src)"的宽松描述误读为存在 (GlobalTensor, GlobalTensor) 组合并标注"已验证"，属于文档推断未经头文件核实的假验证。

重载核对结果（`kernel_operator_data_copy_intf.h`，grep 原始输出）：

```
314: __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, ...   # GM→UB
320: __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, ...  # UB→GM
325: void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, ...)  # UB↔UB
330/339: __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, ...)    # GM→UB（带 pad）
345/350: __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, ...)   # UB→GM（带 pad）
# DataCopy 同样：所有 (GlobalTensor dst, ...) 重载的 src 都是 LocalTensor
```

## 解决配方

替代实现：GM→UB→GM 分块拷贝。新增 `TBuf<TPosition::VECCALC> copyBuf_`（32KB）作 Phase1 暂存，`DataCopyPad(UB ← GM[data+off])`（MTE2）→ `PipeBarrier<PIPE_MTE2>()` → `DataCopyPad(GM[y+off] ← UB)`（MTE3），按 `COPY_TILE_BYTES=32KB` 分块循环。

落地代码（算子 kernel 头文件中的完整实现，verbatim；分块数按 dtype 自适应：`copyTileNum_ = COPY_TILE_BYTES / sizeof(D_T_DATA)`，在 Init 中计算）：

```cpp
// Phase 1 data→y 拷贝的 UB 分块大小（字节）。32KB 对 fp16=16384 元素，足够迭代一 shape。
constexpr uint32_t COPY_TILE_BYTES = 32U * 1024U;

template <typename D_T_DATA, typename D_T_IDX>
__aicore__ inline void BenchNoHarnessScatter<D_T_DATA, D_T_IDX>::Phase1CopyData()
{
    if (dataTotalNum_ <= 0) {
        return;
    }
    LocalTensor<D_T_DATA> copyLocal = copyBuf_.template Get<D_T_DATA>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<D_T_DATA> padParams{false, 0, 0, 0};

    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t start = blockIdx * copyBlockFactor_;
    int64_t end = start + copyBlockFactor_;
    if (end > dataTotalNum_) {
        end = dataTotalNum_;
    }
    if (start >= end) {
        return;
    }

    for (int64_t off = start; off < end; off += copyTileNum_) {
        int64_t cur = off + copyTileNum_;
        if (cur > end) {
            cur = end;
        }
        int64_t curNum = cur - off;
        copyParams.blockLen = static_cast<uint32_t>(curNum * sizeof(D_T_DATA));
        // GM → UB (MTE2)
        DataCopyPad(copyLocal, dataGm_[off], copyParams, padParams);
        PipeBarrier<PIPE_MTE2>();
        // UB → GM (MTE3)
        DataCopyPad(yGm_[off], copyLocal, copyParams);
    }
}
```

配套 UB 预算调整：原设计假设 Phase1 不占 UB（GM→GM），现需 `ubAvailable = ubSize - 8KB(系统保留) - 32KB(copyBuf) - 32B(yTmpBuf)`，tiling 侧的 `updateTileK` 计算随之收紧（最坏 dtype 组合 int64+int64 仍满足 240KB 约束）。语义不变（仍为 data→y 完整拷贝），Phase1 与 Phase2 之间保留 `SyncAll<true>()` 同步点；多核场景按核均分 `copyBlockFactor` 后同样适用。

## 设计决策

- 为什么选 TBuf<VECCALC> 而不是 TQue：拷贝暂存无队列语义（不需要 Alloc/EnQue/DeQue 配对），TBuf + PipeBarrier 最直接；Phase2 的 indices/updates 载入才用 TQue<VECIN>。
- 为什么 `PipeBarrier<PIPE_MTE2>()` 放在 GM→UB 之后、UB→GM 之前：MTE3 读 copyLocal 前必须确保 MTE2 写完成。注意落地代码里**没有**在每轮 MTE3 存出后再加一道屏障（下一轮 MTE2 重写 copyLocal 依赖指令队列自身保序），该差异记入遗留问题。
- 被放弃的替代：在 L0/L2 层做 data→y 的框架侧拷贝（如 ViewCopy）——设计明确要求 Kernel Phase 1 完成拷贝（y 是 AllocTensor 后未初始化的输出），放弃；逐元素 GetValue/SetValue 拷贝——Phase1 是连续整段拷贝，批量 DataCopyPad 明显更优，放弃。
- 复用 UB 的备选（copyBuf 与 Phase2 的 updatesBuf 共享同一块内存）：可省 32KB 但引入时序耦合，骨架迭代不采用。
- 当前方案路线档位：recommended_fastpath（API 面上唯一可行的 kernel 内整段拷贝形态）。

## 预防

- 设计文档里任何"API 已验证"的行，实现前先 grep 头文件核对重载签名（`grep -n "void DataCopyPad" <intf 头文件>`），不要信文档表格的"✅"。
- 任何 GM→GM 数据搬运动作，直接按"GM→UB→GM + 屏障"设计，把 UB 暂存预算（分块大小）纳入 tiling。

## 验证信号

以下信号来自该算子迭代一骨架的实现与编译验证：实现前对头文件重载做 grep 核对，落地 GM→UB→GM 后全量构建。

- 重载核对：`kernel_operator_data_copy_intf.h` 中 DataCopyPad 全部重载为 MTE2(Local←Global) / MTE3(Global←Local) / UB↔UB，无 (GlobalTensor, GlobalTensor)；DataCopy 的全部 (GlobalTensor dst, ...) 重载 src 均为 LocalTensor（来源平台：950；证据等级：tool_observed）。
- 设计偏差记录：创建 issue 文件明确"DESIGN.md §3.6/§3.8 的 GM→GM DataCopyPad 与实际 API 不符，Phase1 已改为 GM→UB→GM 分块拷贝"（来源平台：950；证据等级：tool_observed）。
- 编译验证：`bash build.sh --soc=ascend950 -j8` EXIT=0、0 error、0 warning，`BenchNoHarnessScatter_06f7009c805e5f3127a4046d28295668.o`（ascend950）生成；屏障链修订后全新重建同样 EXIT=0、0 warning（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 重载表仅对 CANN 9.1.0 的 asc-devkit 头文件核对；其他版本 API 面可能变化（未验证）。
- 本卡验证止于编译级（kernel 二进制生成）；**同轨迹后续上板实测发现：arch35/CANN 9.1.0 上该 GM→UB→GM 路线的 UB→GM 段（MTE3，DataCopyExtParams 形态）存在写入静默失效，最终 Phase1/Phase2 改用 GlobalTensor::GetValue/SetValue 标量通路**——采用本中转骨架前必须先核对「arch35 UB→GM DataCopyExtParams 拷贝写失效」卡的结论与替代路线；GM→UB（MTE2）段实测正常。
- 单核（blockDim=1）场景验证；多核按核均分 copyBlockFactor 的路径后续迭代才启用（未验证，机制推断可迁移）。
- 若 dtype 极大（int64 时 copyTileNum=4096）分块保护已含 `copyTileNum_ <= 0 → 1` 兜底。

**未决项**：

- Phase1 循环里"上一轮 MTE3 读 copyLocal → 下一轮 MTE2 重写 copyLocal"之间无显式屏障，仅依赖 MTE 指令队列保序；同一轨迹中对同类 yTmpBuf 复用曾补 `PipeBarrier<PIPE_ALL>()`，两处口径不一致，上板前应统一并实测验证。
- copyBuf 与 Phase2 缓冲复用（省 32KB UB）未尝试。
- GM→UB→GM 相对理想 GM→GM 的实际性能代价需在性能阶段用 profiling 对比（未验证）。
## 坏实践

- 把 API 文档中宽松的 "(dst, src)" 描述当成存在 GM→GM 重载，并在设计文档里标"已验证"——本轨迹的设计文档正是这样产生的假验证，实现阶段才被头文件推翻。
- 用逐元素 GetValue/SetValue 做整段连续拷贝（功能上可行但把批量 MTE 搬运退化成标量循环）。

## 误诊

- 初看容易怀疑"是不是 DAV_3510 不支持这个重载"（平台问题）：核对重载表后发现是 API 面上根本没有该形态，与平台无关——文档产品列表未列某平台与重载不存在是两回事。
