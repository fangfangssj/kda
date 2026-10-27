---
type: "Runbook"
title: "DataCopyPad UB→GM 四参调用导致编译失败"
description: "DataCopyPad 搬出方向只接受 GlobalTensor、LocalTensor 与 DataCopyExtParams 三参；误带 PadExtParams 会令模板匹配失败。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCopyPad UB→GM 四参调用导致编译失败

## 现象

在为 Ascend950 的 direct-launch 索引收集算子编译 `operators/gather/gather.asc` 时，输入搬运和输出搬运都写成了 `DataCopyPad(..., DataCopyExtParams, PadExtParams)` 四参数形式。BiSheng 在实例化 `gather_kernel<T, IdxT>` 的 18 组 dtype 组合时拒绝输出搬运调用，关键原文为：

```text
candidate template ignored: could not match 'LocalTensor' against 'GlobalTensor'
candidate function template not viable: requires 3 arguments, but 4 were provided
18 errors generated.
gmake[2]: *** [CMakeFiles/op_gather.dir/build.make:76: CMakeFiles/op_gather.dir/gather.asc.o] Error 1
```

当时代码已经完成 GM→UB 的源行、索引搬入以及标量 gather；唯一阻塞构建的位置是把 `padT` 继续传给 `yGm <- outTile` 的 UB→GM 搬出。

## 适用场景

- 工程形态：Ascend C direct-launch 单 `.asc` 工程的 kernel 编译；
- API 路径：`AscendC::DataCopyPad` 在 `LocalTensor <- GlobalTensor`（MTE2）与 `GlobalTensor <- LocalTensor`（MTE3）两个方向的 overload 选择；
- 验证过的平台与 CANN 版本：Ascend950 / dav-3510，CANN 9.0.1；与 frontmatter `platforms: [950]` 一致；
- 适用粒度：`platform_wide`，适用于同一工具链上所有用 `DataCopyPad` 从 UB 搬回 GM 的 kernel，不依赖 gather 数学语义；
- 路线优先级：`recommended_fastpath`，这是当前工具链下已编译并上板验证的正确 store 签名基线，不涉及算法路线性能比较；
- 明确不适用或未验证场景：GM→UB load 仍需要 `PadExtParams`；其他 CANN 版本与 A2/A3 尚未验证。

## 根因

`DataCopyPad` 的参数表按搬运方向不对称：GM→UB 的 MTE2 overload 为 `LocalTensor dst + GlobalTensor src + DataCopyExtParams + DataCopyPadExtParams`，而 UB→GM 的 MTE3 overload 为 `GlobalTensor dst + LocalTensor src + DataCopyExtParams`；把 load 的第四个 padding 实参机械复制到 store 后，编译器找不到四参数且首参为 `GlobalTensor` 的候选。

## 解决配方

不要只看函数名对称地复制参数。先由首参 tensor 类型判断方向：

- `LocalTensor <- GlobalTensor`：四参数 load，可带 `DataCopyPadExtParams`；
- `GlobalTensor <- LocalTensor`：三参数 store，只传 `DataCopyExtParams`。

下面两段均为事件流中的 `operators/gather/gather.asc`、symbol `gather_kernel<T, IdxT>` 函数级完整代码。before 来自未提交的完整写入快照（原文件 lines 45–131）；after 由该快照应用事件中 verbatim `old_string`→`new_string` 替换确定性重建，函数边界不变。修复后轨迹记录了短提交号 `d79bfed`，但没有可恢复的 40 位提交号或 blob URL。

### before

```cpp
template <typename T, typename IdxT>
__global__ __aicore__ void gather_kernel(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                         int64_t M, int64_t blockRows,
                                         int64_t D, int64_t Dout,
                                         int64_t tileSize, int64_t pathA)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<IdxT> idxGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    idxGm.SetGlobalBuffer((__gm__ IdxT *)index);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    int64_t blockId = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t rowStart = blockId * blockRows;
    if (rowStart >= M) {
        return;
    }
    int64_t rowEnd = rowStart + blockRows;
    if (rowEnd > M) {
        rowEnd = M;
    }

    const int64_t szT = static_cast<int64_t>(sizeof(T));
    const int64_t szIdx = static_cast<int64_t>(sizeof(IdxT));
    auto ceil32 = [](int64_t b) -> int64_t { return (b + 31) / 32 * 32; };

    int64_t rowBytesAligned = ceil32(D * szT);   // actual xRowBuf occupancy
    int64_t idxTileBytes = ceil32(tileSize * szIdx);
    int64_t outTileBytes = ceil32(tileSize * szT);
    // Path B keeps a 32B placeholder xRowBuf (never read; ternary skips GetValue).
    int64_t xRowBytes = pathA ? rowBytesAligned : 32;

    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowBuf;
    pipe.InitBuffer(idxBuf, idxTileBytes);
    pipe.InitBuffer(outBuf, outTileBytes);
    pipe.InitBuffer(xRowBuf, xRowBytes);

    auto idxTile = idxBuf.Get<IdxT>();
    auto outTile = outBuf.Get<T>();
    auto xRow = xRowBuf.Get<T>();

    AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<IdxT> padIdx{false, 0, 0, 0};

    for (int64_t r = rowStart; r < rowEnd; ++r) {
        // Path A: load the whole source row once per row.
        if (pathA) {
            AscendC::DataCopyExtParams cpX{1, static_cast<uint32_t>(rowBytesAligned), 0, 0, 0};
            AscendC::DataCopyPad(xRow, xGm[r * D], cpX, padT);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE2 -> S
        }

        for (int64_t t0 = 0; t0 < Dout; t0 += tileSize) {
            int64_t tileLen = tileSize;
            if (t0 + tileLen > Dout) {
                tileLen = Dout - t0;
            }

            // Load index tile (GM -> UB).
            AscendC::DataCopyExtParams cpIdx{1, static_cast<uint32_t>(tileLen * szIdx), 0, 0, 0};
            AscendC::DataCopyPad(idxTile, idxGm[r * Dout + t0], cpIdx, padIdx);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE2 -> S before scalar GetValue

            // Scalar gather: outTile[j] = pathA ? xRow[idx] : xGm[r*D + idx]
            for (int64_t j = 0; j < tileLen; ++j) {
                int64_t idx = static_cast<int64_t>(idxTile.GetValue(static_cast<uint32_t>(j)));
                T v;
                if (pathA) {
                    v = xRow.GetValue(static_cast<uint32_t>(idx));
                } else {
                    v = xGm.GetValue(static_cast<uint64_t>(r * D + idx));
                }
                outTile.SetValue(static_cast<uint32_t>(j), v);
            }

            AscendC::PipeBarrier<PIPE_ALL>();  // S -> MTE3 before DMA-out

            // Store output tile (UB -> GM).
            AscendC::DataCopyExtParams cpY{1, static_cast<uint32_t>(tileLen * szT), 0, 0, 0};
            AscendC::DataCopyPad(yGm[r * Dout + t0], outTile, cpY, padT);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE3 done before next tile reuses buffers
        }
    }
}
```

### after

```cpp
template <typename T, typename IdxT>
__global__ __aicore__ void gather_kernel(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                         int64_t M, int64_t blockRows,
                                         int64_t D, int64_t Dout,
                                         int64_t tileSize, int64_t pathA)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<IdxT> idxGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    idxGm.SetGlobalBuffer((__gm__ IdxT *)index);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    int64_t blockId = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t rowStart = blockId * blockRows;
    if (rowStart >= M) {
        return;
    }
    int64_t rowEnd = rowStart + blockRows;
    if (rowEnd > M) {
        rowEnd = M;
    }

    const int64_t szT = static_cast<int64_t>(sizeof(T));
    const int64_t szIdx = static_cast<int64_t>(sizeof(IdxT));
    auto ceil32 = [](int64_t b) -> int64_t { return (b + 31) / 32 * 32; };

    int64_t rowBytesAligned = ceil32(D * szT);   // actual xRowBuf occupancy
    int64_t idxTileBytes = ceil32(tileSize * szIdx);
    int64_t outTileBytes = ceil32(tileSize * szT);
    // Path B keeps a 32B placeholder xRowBuf (never read; ternary skips GetValue).
    int64_t xRowBytes = pathA ? rowBytesAligned : 32;

    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowBuf;
    pipe.InitBuffer(idxBuf, idxTileBytes);
    pipe.InitBuffer(outBuf, outTileBytes);
    pipe.InitBuffer(xRowBuf, xRowBytes);

    auto idxTile = idxBuf.Get<IdxT>();
    auto outTile = outBuf.Get<T>();
    auto xRow = xRowBuf.Get<T>();

    AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<IdxT> padIdx{false, 0, 0, 0};

    for (int64_t r = rowStart; r < rowEnd; ++r) {
        // Path A: load the whole source row once per row.
        if (pathA) {
            AscendC::DataCopyExtParams cpX{1, static_cast<uint32_t>(rowBytesAligned), 0, 0, 0};
            AscendC::DataCopyPad(xRow, xGm[r * D], cpX, padT);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE2 -> S
        }

        for (int64_t t0 = 0; t0 < Dout; t0 += tileSize) {
            int64_t tileLen = tileSize;
            if (t0 + tileLen > Dout) {
                tileLen = Dout - t0;
            }

            // Load index tile (GM -> UB).
            AscendC::DataCopyExtParams cpIdx{1, static_cast<uint32_t>(tileLen * szIdx), 0, 0, 0};
            AscendC::DataCopyPad(idxTile, idxGm[r * Dout + t0], cpIdx, padIdx);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE2 -> S before scalar GetValue

            // Scalar gather: outTile[j] = pathA ? xRow[idx] : xGm[r*D + idx]
            for (int64_t j = 0; j < tileLen; ++j) {
                int64_t idx = static_cast<int64_t>(idxTile.GetValue(static_cast<uint32_t>(j)));
                T v;
                if (pathA) {
                    v = xRow.GetValue(static_cast<uint32_t>(idx));
                } else {
                    v = xGm.GetValue(static_cast<uint64_t>(r * D + idx));
                }
                outTile.SetValue(static_cast<uint32_t>(j), v);
            }

            AscendC::PipeBarrier<PIPE_ALL>();  // S -> MTE3 before DMA-out

            // Store output tile (UB -> GM); store overload takes 3 args (no pad).
            AscendC::DataCopyExtParams cpY{1, static_cast<uint32_t>(tileLen * szT), 0, 0, 0};
            AscendC::DataCopyPad(yGm[r * Dout + t0], outTile, cpY);
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE3 done before next tile reuses buffers
        }
    }
}
```

## 设计决策

- 选择按数据方向使用不同 overload，而不是为了“签名统一”包装一层伪 padding 参数；编译器头文件已经把 MTE2 load 与 MTE3 store 的能力边界编码到类型和参数个数中。
- 放弃四参数 store：第四个 `DataCopyPadExtParams` 只服务搬入 UB 后的 padding 行为，GM 输出目标没有对应参数位；保留它会在模板匹配阶段失败。
- 当前方案属于 `recommended_fastpath`：三参数 store 是本工具链上已经完成编译、上板正确性与完整评测的正确基线。

## 系统信息

- toolchain：CANN 9.0.1，BiSheng/Clang 15.0.5；
- 目标：Ascend950PR_957b / dav-3510；
- 工程形态：direct-launch 单 `.asc`，构建目标 `libop_gather.so`；
- 相关文件：`operators/gather/gather.asc`；
- 关键头文件诊断位置：`asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h` 的 MTE2/MTE3 `DataCopyPad` overload。

## 预防

- 写 `DataCopyPad` 时先看第一个 tensor 的存储位置：`LocalTensor` 首参是搬入，`GlobalTensor` 首参是搬出；
- 不要假设同名搬入/搬出 API 参数完全对称；直接核对当前 CANN 版本声明；
- 若错误同时出现“`LocalTensor` against `GlobalTensor`”和“requires 3 arguments, but 4 were provided”，优先检查搬运方向与实参数量，不要先改 tensor 类型；
- 对多 dtype 模板，先用一个实例消除 overload 错误再全量分发，避免同一根因膨胀成大量重复 diagnostics。

## 验证信号

本知识来自一次 Ascend950 索引收集 kernel 的首次完整构建及随后的精确单行修复；构建、pytest、干净重建和 cann-bench 均由远程工具在目标平台执行。

- 修复前编译器报告“requires 3 arguments, but 4 were provided”，并以 `18 errors generated` 终止（来源平台：950；证据等级：tool_observed）；
- 删除 store 的第四个 `padT` 实参后，构建输出 `Linking ASC shared library libop_gather.so` 与 `Built target op_gather`（来源平台：950；证据等级：tool_observed）；
- 修复后的实现通过 181 项 pytest；干净重建后再次通过 181 项 pytest（来源平台：950；证据等级：tool_observed）；
- 最终 cann-bench 的 Gather 用例通过 20/20（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：删除 UB→GM `DataCopyPad` 调用的第四个 padding 参数后，工具输出 `CMake build complete`、`Wheel package built successfully` 与 `Build complete`。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 该结论只约束 `GlobalTensor <- LocalTensor` 的 UB→GM `DataCopyPad`；GM→UB 的四参数 load 不应照此删除 padding 参数；
- A2、A3 与其他 CANN 版本未验证；若头文件提供不同 overload，须以固定版本声明为准（未验证，推断可迁移）；
- 本卡不覆盖 `DataCopy`、二维搬运或 queue API 的参数契约。

## 坏实践

看到 load 与 store 都叫 `DataCopyPad` 就机械复制同一参数表。API 名称相同不代表传输方向的附加参数对称；这种写法会让真实的 arity 错误被大量模板实例化诊断淹没。

## 误诊

“could not match `LocalTensor` against `GlobalTensor`”容易被误读为源、目标 tensor 类型写反。该次故障中，后续候选已经明确指出三参数 store 可匹配，真正阻塞点是多传了第四个 `DataCopyPadExtParams`。