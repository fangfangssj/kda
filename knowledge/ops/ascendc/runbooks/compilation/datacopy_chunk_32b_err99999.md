---
type: "Runbook"
title: "DataCopy尾chunk非32B对齐触发ERR99999"
description: "Ascend C DataCopy 在 UB↔GM 方向要求 count*sizeof(T) 为 32 字节对齐，否则搬运量对 32B 向下取整。当纯搬运算子尾 chunk 的实际元素数不满足 32B 对齐时，UB→GM 写方向触发 ERR99999、GM→UB 读方向静默丢尾部数据。判别方法：尾部元素数*sizeof(T)%32!=0 即命中。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-18T10:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCopy尾chunk非32B对齐触发ERR99999

## 适用场景

- 工程形态：ops-direct-invoke Kernel 直调（`op_kernel/*.asc` + `op_extension/*_torch.cpp` + CMake 直编 `.so`），非 aclnn 算子仓流程。
- API 路径：`DataCopy(ub, srcGm, count)` / `DataCopy(dstGm, result, count)`；`GlobalTensor<T>::SetGlobalBuffer(ptr, len)`；`TPipe` + `TQue<TPosition::VECIN/VECOUT, DB=2>`；`Adds(out, data, (T)0, count)` 做 passthrough。
- 验证过的平台与 CANN 版本：Ascend950PR（chip 957c，dav-3510），CANN 9.1.0。`platforms: [950]` 仅此平台实测。
- 适用粒度：`operator_family`——任何在 UB↔GM 之间用 `DataCopy` 做纯搬运/拼接、且输出尾部元素数可能非 32B 对齐的向量算子（Slice/Concat/Split/Reshape 搬运路径、Gather 尾块等）均适用。
- 路线优先级：`recommended_fastpath`——尾 chunk 读写双向 32B 向上补齐是搬运类算子默认应采用的写法。
- 不适用或未验证场景：Cube 类算子（`DataCopy` 的 cube 通路对齐约束不同）；GM→GM 的 `DataCopy`（仅 UB 侧约束）；a3/a2 平台未验证，推断可迁移但未实测（见 `## 失效边界`）。

## 判别方法

触发（tool_observed）：在 Ascend950PR 上实现 `torch.slice` 算子的连续搬运 kernel（`op_kernel/Slice_kernel.asc` 的 `ContiguousCopy<T>` / `ContiguousCopyBF16`）。kernel 编译通过、基础精度 9 个 case 全过，但扩展精度测试中当输出尾部元素数 `tailElements` 不满足 32 字节对齐时，UB→GM 方向的 `DataCopy` 触发 `ERR99999 UNKNOWN application exception`。复现命令（Python 侧）：

```python
# 32B 对齐 → PASS
torch.ops.npu_test.slice(x_bool_100, 0, 10, 42, 1)  # 32 元素 → 32 bytes → aligned

# 非 32B 对齐 → FAIL (ERR99999)
# 55 元素 → 55 bytes → NOT aligned
```

报错原文（取自审查报告 `docs/REVIEW.md`，该文件经 file_read tool_result 保留于 trace seq210，非 redacted）：

```
[ERROR] ERR99999 UNKNOWN applicaiton exception
[AIV Block 0/1] [Slice] block=0 dtype=4 myTotal=55 step=1 srcOffset=10 myOffset=0
```

探针：对任一用 `DataCopy` 做 UB↔GM 搬运的 kernel，检查尾 chunk 的搬运量是否满足 32B 对齐——即 `tailCount * sizeof(T) % 32 == 0` 是否成立。可直接用上述非对齐尾部 case（bool、55 元素）运行一次：命中 `ERR99999` 即判为本卡所述的向下取整机制问题；同时检查 `ChkSz` 这类"尾 chunk 取实际剩余"的 helper 返回值是否被直接喂给 `DataCopy`。

触发态代码（verbatim，修复前 `ContiguousCopy<T>`，`op_kernel/Slice_kernel.asc` 第 88-120 行，取自 trace seq305 file_read tool_result，非 redacted）：

```cpp
    template<typename T>
    __aicore__ inline void ContiguousCopy()
    {
        uint64_t nChunks = (myTotal_ + chunkSize_ - 1) / chunkSize_;
        if (nChunks == 0) return;

        __gm__ T* srcPtr = reinterpret_cast<__gm__ T*>(srcRaw_);
        __gm__ T* dstPtr = reinterpret_cast<__gm__ T*>(dstRaw_);

        for (uint64_t c = 0; c < nChunks; c++) {
            uint32_t cs = ChkSz(c, nChunks);
            uint64_t srcOff = srcOffset_ + myOffset_ + c * chunkSize_;
            uint64_t dstOff = myOffset_ + c * chunkSize_;

            GlobalTensor<T> srcGm;
            srcGm.SetGlobalBuffer(srcPtr + srcOff, cs);
            GlobalTensor<T> dstGm;
            dstGm.SetGlobalBuffer(dstPtr + dstOff, cs);

            auto ub = inQ_.AllocTensor<T>();
            DataCopy(ub, srcGm, cs);
            inQ_.EnQue(ub);
            auto data = inQ_.DeQue<T>();

            auto out = outQ_.AllocTensor<T>();
            Adds(out, data, (T)0, cs);
            outQ_.EnQue(out);
            inQ_.FreeTensor(data);
            auto result = outQ_.DeQue<T>();
            DataCopy(dstGm, result, cs);
            outQ_.FreeTensor(result);
        }
    }
```

`ContiguousCopyBF16`（第 122-153 行）结构完全相同，仅 `T=bfloat16_t`，同样用 `cs = ChkSz(c, nChunks)` 做 read/write，无对齐补齐。`ChkSz` 对尾 chunk 返回 `myTotal_ - c*chunkSize_`，对 bool（sizeof=1）尾部 55 元素即 55 字节，不满足 32B 对齐。

观测：bool 输出尾部 55 元素（55 bytes，`55 % 32 = 23 ≠ 0`）命中 `ERR99999`；32B 对齐的 bool case（32 元素）通过。修复后 bool/bf16 非对齐尾部扩展精度通过，9/9 基础精度全过、复审从 FAIL 74/100 改判 PASS 80/100。

判别与解释（interpretation，agent 根因分析）：Ascend C `DataCopy` 对 UB↔GM 方向有硬约束——`count * sizeof(T)` 需要 32 字节对齐，若不对齐，搬运量将对 32 字节做向下取整（API 文档《基础数据搬运.md》第 91 行原文，经 `REVIEW.md` 引用保留于 seq210）。原实现对尾 chunk 用实际元素数 `cs` 直接做 `DataCopy`：当 `cs * sizeof(T) % 32 != 0` 时，搬运量被向下取整到 `floor(cs*sizeof(T)/32)*32`，尾部若干元素根本没被搬，UB→GM 写方向触发 `ERR99999`，GM→UB 读方向则静默丢尾部数据。以 bool, tail=55 为例：`55 bytes → floor(55/32)*32 = 32 bytes`，尾部 23 字节未搬运，写方向报 `ERR99999`、读方向尾部 23 字节为未初始化垃圾。这不是精度逻辑错，而是搬运量计算被 API 向下取整的机制问题。机制判定：尾部元素数 `* sizeof(T) % 32 != 0` 且 `DataCopy` 直接用未补齐的 `count` → 命中本判别规则。

## 验证信号

来源：本 trace 在 Ascend950PR（CANN 9.1.0）上开发并审查 `torch.slice` 连续搬运 kernel，扩展精度测试覆盖 bool/bf16 非 32B 对齐尾部 case；`docs/REVIEW.md` 经 file_read 保留于 trace seq210，记录 `ERR99999` 与 API 约束原文。

- `ERR99999 UNKNOWN application exception`，bool 输出尾部 55 元素（55 bytes，非 32B 对齐）触发；32B 对齐 case 通过。（来源平台：950；证据等级：artifact_verified）
- API 约束原文 `count * sizeof(T) 需要 32 字节对齐，若不对齐，搬运量将对 32 字节做向下取整`（《基础数据搬运.md》第 91 行，经 `REVIEW.md` 引用）。（来源平台：950；证据等级：artifact_verified）
- 修复后扩展精度 bool/bf16 非对齐尾部 case 通过，9/9 基础精度全过、复审从 FAIL 74/100 改判 PASS 80/100（bash 运行测试观测，verbatim stdout 未由 adapter 保留）。（来源平台：950；证据等级：tool_observed）

## 失效边界

- `DataCopy` 的 32B 向下取整约束在 DAV_3510（Ascend950）上实测确认；a2 与 a3 未在本 trace 实测，推断同约束可迁移（同属 Ascend C 向量核 `DataCopy` 语义）但**未验证**。
- 写方向用 `alignedCs` 向上补齐会向输出 GM 多写若干补齐字节（`alignedCs - rawCs` 个垃圾元素）。其安全性依赖 PyTorch NPU 分配器的块状分配——若调用方用极紧凑、无 padding 的自定义 GM 分配器，多写补齐字节可能越界，需额外保证输出 GM 尾部有 `alignedCs - rawCs` 字节余量。
- `DataCopy` 在 Cube 通路或 GM→GM 直搬场景的对齐约束不同，本卡不适用。
- 本卡只判别失败机制（trigger/probe/observation/interpretation），不包含修复后的 verbatim 完整函数。

## 坏实践

- 只对读方向 `SetGlobalBuffer(alignedCs)` 而写方向仍 `DataCopy(dstGm, result, rawCs)`——读补齐了，写还在向下取整，尾部依然丢，`ERR99999` 不消。
- 用 `GetValue/SetValue` 逐元素处理尾 chunk 绕过对齐——违反矢量算子规范且性能极差。
- 把 `SetGlobalBuffer` 长度设为 `rawCs` 而 `DataCopy` 传 `alignedCs`——`SetGlobalBuffer` 长度小于实际搬运量，越界不安全。
- 把 `ChkSz`（尾 chunk 取实际剩余）的返回值直接喂给 `DataCopy` 而不做 32B 补齐——即本卡触发态。