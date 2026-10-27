---
type: "Runbook"
title: "Host 缺 SetDataSize 致 TilingData 截断"
description: "ACLNN 算子 op_host TilingFunc 填完自定义 TilingData 字段后未调用 rawTilingData->SetDataSize(sizeof(<TilingData>))，框架按默认小 data_size 从 GM 搬运 tiling，只传结构体头部，kernel 侧 GET_TILING_DATA 解析出 perCoreCount 等关键字段为 0/垃圾，DMA 循环零次执行，输出保持框架初始化状态、ST 全红。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-18T03:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Host 缺 SetDataSize 致 TilingData 截断

## 现象

在 Ascend950（CANN 9.1.0，ops-registry-invoke / ACLNN 流程）上开发 BatchToSpace 算子 Branch-1（tilingKey=1，ubAxis=5，大 depth D≥cacheLineElements）。kernel 编译通过、能被 dispatch（readelf -S 确认 .text 段约 1952 字节、非空），DataCopyPad 调用存在于源码，但跑 ST 全量用例：

```
[FAIL] case00764
[FAIL] case00765
ST: 765 total, 6 passed, 759 failed, 0 skipped
```

765 个用例仅 6 个通过，且这 6 个全是 int64 identity 用例（block_shape=[1,1] crops=0）。输出缓冲区与未跑 kernel 时几乎一致——kernel 实质上没有向输出 GM 写入任何可见数据。

## 适用场景

- **工程形态**：ACLNN 算子仓开发（ops-registry-invoke），op_host 侧自定义 TilingFunc + op_kernel 侧 GET_TILING_DATA 读取 TilingData 结构体。
- **API 路径**：`REGISTER_TILING_DEFAULT(<TilingData>)`（kernel 侧注册）→ host `rawTilingData->GetData()` 填字段 → `rawTilingData->SetDataSize(sizeof(<TilingData>))` → kernel `GET_TILING_DATA(<TilingData>, tiling)` 解析。
- **验证过的平台与 CANN 版本**：Ascend950 / CANN 9.1.0（`/home/developer/Ascend/cann-9.1.0`）。frontmatter `platforms: [950]` 严格对应。
- **适用粒度**：`operator_family`——任何在 op_host 自定义 TilingData 结构体、kernel 用 GET_TILING_DATA 解析的 ACLNN 算子，若 host 未显式 SetDataSize 都会命中同一截断。
- **不适用或未验证场景**：使用框架内置 TilingData（无自定义结构体）或 host 直接用固定大小 workspace 的算子不涉及此路径；Ascend310/910 等其它平台未在本 trace 验证（推断可迁移，见 ## 失效边界）。

## 判别方法

**触发条件**：ACLNN 算子 kernel 编译通过、能被 dispatch（.text 段非空）、DataCopy/DMA 调用存在于源码，但 ST 全量用例几乎全红（本例 765 total 仅 6 passed），且通过的极少数用例恰是 identity（block_shape=[1,1] crops=0）这类不依赖 kernel 真实输出的旁路用例——即 kernel 实质未向输出 GM 写入可见数据。

**可执行探针**：把目标算子的 op_host TilingFunc 与本仓已验证可跑的同类算子（如 `operators/TileWithAxis_package/op_host/tile_with_axis_host.cpp`）做 diff，重点查 TilingData 字段填写完毕、`SetBlockDim`/`SetTilingKey` 之前是否显式调用 `rawTilingData->SetDataSize(sizeof(<TilingData>))`；或在 host 文件内 `grep SetDataSize`，自定义 TilingData 结构体的算子命中 0 次即命中本缺陷。

```cpp
// 探针：diff/grep host TilingFunc 尾部
//   期望（TileWithAxis host，已验证可跑）：
//     rawTilingData->SetDataSize(sizeof(TileWithAxisTilingData));
//     context->SetBlockDim(realCoreNum);
//   命中缺陷（BatchToSpace host，edit oldString 恢复，SetDataSize 缺失）：
//     td->outH = outH;
//     td->outW = outW;
//     context->SetBlockDim(realCoreNum);   // <- 缺 SetDataSize
```

**原始观测**（Ascend950，BatchToSpace Branch-1）：
- ST: 765 total, 6 passed, 759 failed, 0 skipped；通过的 6 个全是 int64 identity（block_shape=[1,1] crops=0），非 kernel 真实写出。
- TilingFunc 尾部（edit oldString 逐字恢复）只到 `td->outW = outW;` 紧接 `context->SetBlockDim(realCoreNum);`，中间无 `SetDataSize` 调用；对比 TileWithAxis host 在此处显式 `rawTilingData->SetDataSize(sizeof(TileWithAxisTilingData))`。
- 修复 edit（newString 逐字恢复）在 `td->outW = outW;` 与 `context->SetBlockDim(realCoreNum);` 之间插入 `rawTilingData->SetDataSize(sizeof(BatchToSpaceTilingData));`——与 TileWithAxis 的写法对齐。

**机制解释**：框架（CANN ops_custom 编译/运行时）依据 `rawTilingData` 的 `data_size` 决定从 GM 搬到 kernel 可读 UB 区域的 tiling 字节数。未显式 SetDataSize 时，框架沿用一个很小的默认值（本例现象对应仅覆盖结构体头部少量字段，确切字节数未在 trace 由工具直接观测到，见 ## 失效边界）。于是 240 字节的 `BatchToSpaceTilingData` 只有头部被传输，kernel 侧 `GET_TILING_DATA` 解析出的 `td_->totalCount`、`td_->perCoreCount`、`td_->bufferSize`、`td_->ubAxis` 等关键字段为 0 或垃圾值：`for (uint64_t i = start; i < end; i++)` 中 `end = start + td_->perCoreCount`，`perCoreCount=0` ⇒ 循环不执行 ⇒ DataCopyPad 永不被调用 ⇒ 输出 GM 保持框架初始化状态。6 个 int64 identity 用例之所以"通过"，是框架对 identity 走了一条不依赖 kernel 输出的旁路，并非 kernel 真正写对。

## 验证信号

- ST: 765 total, 6 passed, 759 failed, 0 skipped（kernel 编译/Dispatch 正常但输出未改写；通过的 6 个为 int64 identity 旁路）（来源平台：Ascend950；证据等级: tool_observed）
- TilingFunc 尾部 edit oldString 逐字恢复：`td->outH = outH; td->outW = outW; context->SetBlockDim(realCoreNum);`，缺 `SetDataSize`；edit newString 在其间插入 `rawTilingData->SetDataSize(sizeof(BatchToSpaceTilingData));`（来源平台：Ascend950；证据等级: artifact_verified）
- 对比 TileWithAxis host（本仓已验证可跑）在 TilingFunc 同位置显式调用 `rawTilingData->SetDataSize(sizeof(TileWithAxisTilingData))`（来源平台：Ascend950；证据等级: artifact_verified）

> 修复后 ST 重跑信号（764 passed / 1 failed 等）原始 tool_result 未被 markdown_transcript adapter 保留（adapter 丢失大量 tool_result），仅在 agent reasoning 中转述，属 agent_reported，不计入本卡验证信号。

## 失效边界

- Ascend950 / CANN 9.1.0 已实测。a2、a3 等其它平台未在本 trace 验证——`rawTilingData->SetDataSize` 是框架层 API，跨平台行为预期一致，**未验证，推断可迁移**；若新平台框架对未 SetDataSize 时改用结构体全量搬运，则该 bug 不复现，但补 SetDataSize 仍是无害的显式声明。
- 使用框架内置 TilingData（无自定义结构体）的算子不涉及此路径。
- 框架默认 data_size 的确切字节数未在 trace 中由工具直接观测到精确值，"只覆盖结构体头部"系由"TilingData 头部字段被正确读取、后续字段为 0/垃圾"的运行表现推断，未追到框架源码层确认。

## 坏实践

- 调试中曾通过 `context->SetTilingKey(0)` 绕过 `REGISTER_TILING_FOR_TILINGKEY` dispatch，使通过数从 6 升到 24（18 个非 Branch-1 框架假通过 + 6 个 B1），造成"修复有效"假象——实为掩盖 tilingKey 路由问题，未触及 SetDataSize 根因，属调试捷径，不可作为解决配方落库。

## 误诊

调试前期曾误判为以下根因（均经验证排除）：

1. **DataCopyPad 同步方式错误**（TBuf+SetFlag/WaitFlag vs TQue+EnQue/DeQue）：两种同步方式结果完全一致（均 6 通过），排除。
2. **kernel 未被 dispatch（tilingKey 注册缺失）**：补 `REGISTER_TILING_FOR_TILINGKEY(...)` 后仍 6 通过；`TILING_KEY_IS(1)` 守卫与无守卫结果一致，排除。
3. **多核 SetBlockDim 问题**：`SetBlockDim(1)` 单核仍 6 通过，排除。
4. **kernel 代码被编译器优化掉**：readelf -S 确认 .text 段 ≈1952 字节非空（虽远小于 TileWithAxis 的 142KB，但足以容纳简单 DataCopyPad 循环），排除。
5. **InferShape stub 返回 identity**：这是真实存在的并列根因（identity InferShape 触发框架 identity-copy 假通过），但其修复使通过数从 24 降到 6（移除了 18 个框架假通过），并未让 Branch-1 通过数上升——故非"kernel 无输出"的主因，SetDataSize 才是。

## 预防

- 任何在 op_host 自定义 TilingData 结构体的 ACLNN 算子，TilingFunc 末尾填完字段后必须 `rawTilingData->SetDataSize(sizeof(<TilingData>))`，再 `SetBlockDim`/`SetTilingKey`。
- 以本仓已验证可跑的算子（AddCustom、TileWithAxis）host 为参考锚点，新算子 host 与之 diff，缺失 SetDataSize 即报警。
- TDD 红绿基线：kernel 未实现时 ST 预期全红（0 通过）；若"空 kernel"却出现非零通过，先怀疑框架 identity 旁路而非 kernel 正确。