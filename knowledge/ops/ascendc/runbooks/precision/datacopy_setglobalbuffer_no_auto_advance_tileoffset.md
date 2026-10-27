---
type: "Runbook"
title: "DataCopy 不自动按 tile 推进须显式元素 offset"
description: "Kernel 直调下 SetGlobalBuffer 绑定 block 子区间后，DataCopy(local, gm, count) 每轮都从 gm 位置 0 读取；必须在 Process 中维护元素级 tileOffset 并以 gm[tileOffset] 索引，否则除首 tile 外输出全错。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-18T03:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# DataCopy 不自动按 tile 推进须显式元素 offset

## 现象

在用 Kernel 直调（ops-direct-invoke）方式开发 FP16 逐元素 add 算子（`add_custom`，C=A+B，升精度 half→FP32→Add→half）时，kernel 编译通过、单核 `blockNum=1` 处理 65536 个 half 元素。运行 `bash run.sh` 后精度验证脚本报：

```
Verification FAILED!
Max diff: 19.828125, Mean diff: 6.253139495849609
Mismatch count: 61406 / 65536
```

输出不是全零、也不是 NaN，而是"前一小段对、后面全错"。后来改写一个只做 `DataCopy(xLocal, xGm, count) → DataCopy(zGm, zHalf, count)` 的纯拷贝 kernel 探针运行，打印前 10 个元素发现：output 前 10 个元素与 golden 完全一致，但越往后越错——说明只有第一个 tile 的数据被正确搬运，后续 tile 全部重复读了第 0 个 tile 的数据。

## 适用场景

- **工程形态**：Kernel 直调（`add_custom_kernel<<<blockNum, ...>>>`），非 aclnn 注册；host 用 `aclrtMalloc/SetGlobalBuffer` 绑定 GM。
- **API 路径**：`AscendC::SetGlobalBuffer` + `AscendC::DataCopy(LocalTensor, GlobalTensor, count)` / `DataCopy(GlobalTensor, LocalTensor, count)`；`GetBlockIdx()` 多核切分下 `xGm.SetGlobalBuffer((__gm__ half*)x + blockIdx*numPerCore, blockLen)`。
- **验证过的平台与 CANN 版本**：Ascend 910B3 / DAV_2201（Atlas A2），CANN 9.0.0，bisheng 编译。
- **适用粒度**：`operator_family`——任何在 `Process` 里循环搬运多个 UB tile 的 1D elementwise kernel（add/sub/mul 等）都适用，不限于 add。
- **路线优先级**：`recommended_fastpath`——元素级 `xGm[tileOffset]` 索引是 AscendC 官方示例（relu 模板 `DataCopyPad(xLocal, xGm, {...})` 同源）的标准推进方式。
- **不适用或未验证场景**：`DataCopyPad` 带 block-pad 参数的形态（其 offset 由参数结构体控制，机制不同）；多核 `blockNum>1` 在本 trace 未实测（单核验证功能）；950/a3 未验证。

## 根因

`GlobalTensor::SetGlobalBuffer(ptr, len)` 把 `xGm` 绑定到当前 block 的子区间后，`xGm` 的"位置 0"就是该子区间起点。`DataCopy(xLocal, xGm, count)` 的三参重载**总是从 `xGm` 的位置 0 读 count 个元素，不会因为你在 `Process` 循环里调了多次就自动推进读指针**。要在循环里依次读后续 tile，必须用带 offset 的索引形式 `xGm[tileOffset]`（元素 offset，不是字节 offset），其中 `tileOffset` 由调用方在每次迭代后 `+= count` 累加。

首版 kernel 还犯了一个伴随错误：把索引写成 `xGm[offset / TILE_LENGTH]`——`offset` 是元素累加量（0, 4096, 8192…），除以 `TILE_LENGTH` 后变成块号（0,1,2…），于是 `xGm[0]`、`xGm[1]`、`xGm[2]` 只读到第 0、1、2 个**元素**而非第 0、1、2 个 tile，数据完全错位。两个错误的共同表象都是"首 tile 对、其余错"，max_diff ≈ 19.9、mismatch ≈ 61406/65536。

## 解决配方

before（`op_kernel/add_custom_kernel.asc`，CopyIn/CopyOut 无 offset，Process 不累加）：

```cpp
    __aicore__ inline void Process() {
        uint64_t offset = 0;
        for (uint32_t i = 0; i < ubLoop - 1; i++) {
            CopyIn(TILE_LENGTH, offset);
            Compute(TILE_LENGTH);
            CopyOut(TILE_LENGTH, offset);
            offset += TILE_LENGTH;
        }
        CopyIn(ubTail, offset);
        Compute(ubTail);
        CopyOut(ubTail, offset);
    }

    __aicore__ inline void CopyIn(uint32_t count, uint64_t offset) {
        AscendC::LocalTensor<half> xHalf = xQueHalf.AllocTensor<half>();
        AscendC::LocalTensor<half> yHalf = yQueHalf.AllocTensor<half>();
        if (count == TILE_LENGTH) {
            AscendC::DataCopy(xHalf, xGm[offset / TILE_LENGTH], count);
            AscendC::DataCopy(yHalf, yGm[offset / TILE_LENGTH], count);
        } else {
            AscendC::DataCopy(xHalf, xGm[offset / TILE_LENGTH], count);
            AscendC::DataCopy(yHalf, yGm[offset / TILE_LENGTH], count);
        }
        xQueHalf.EnQue(xHalf);
        yQueHalf.EnQue(yHalf);
    }

    __aicore__ inline void CopyOut(uint32_t count, uint64_t offset) {
        AscendC::LocalTensor<half> zHalf = zQueHalf.DeQue<half>();
        if (count == TILE_LENGTH) {
            AscendC::DataCopy(zGm[offset / TILE_LENGTH], zHalf, count);
        } else {
            AscendC::DataCopy(zGm[offset / TILE_LENGTH], zHalf, count);
        }
        zQueHalf.FreeTensor(zHalf);
    }
```

after（终版，元素级 `tileOffset` 累加 + `xGm[tileOffset]`）：

```cpp
    __aicore__ inline void Process() {
        if (blockLen == 0) {
            return;
        }
        uint32_t tileOffset = 0;
        for (uint32_t i = 0; i < ubLoop; i++) {
            bool isLastUB = (i == ubLoop - 1);
            uint32_t count = isLastUB ? ubTail : tileLen;
            CopyIn(count, tileOffset);
            Compute(count);
            CopyOut(count, tileOffset);
            tileOffset += count;
        }
    }

    __aicore__ inline void CopyIn(uint32_t count, uint32_t offset) {
        LocalTensor<half> xLocal = xQueHalf.AllocTensor<half>();
        LocalTensor<half> yLocal = yQueHalf.AllocTensor<half>();
        DataCopy(xLocal, xGm[offset], count);
        DataCopy(yLocal, yGm[offset], count);
        xQueHalf.EnQue(xLocal);
        yQueHalf.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t count, uint32_t offset) {
        LocalTensor<half> zHalf = zQueHalf.DeQue<half>();
        DataCopy(zGm[offset], zHalf, count);
        zQueHalf.FreeTensor(zHalf);
    }
```

要点：(1) offset 是**元素个数**，不是字节、不是块号——`xGm[offset]` 直接按 `half` 元素步进；(2) `tileOffset += count`（用本 tile 实际 count，尾 tile 是 `ubTail` 而非 `tileLen`），不要用固定 `TILE_LENGTH`，否则尾块偏移会越界；(3) `xGm`/`yGm`/`zGm` 在 `Init` 里已 `SetGlobalBuffer((__gm__ half*)x + blockIdx*numPerCore, blockLen)` 绑到本 block 子区间，所以 `xGm[tileOffset]` 是相对本 block 起点的偏移，不要再加 `blockIdx*numPerCore`。

## 验证信号

本卡验证来自 2026-06-30 在 Ascend 910B3 / DAV_2201（CANN 9.0.0）上对 `add_custom` 的直调通路实测：修复前 `run.sh` 跑出 `Max diff: 19.828125, Mean diff: 6.253, Mismatch 61406/65536`（探针确认首 10 元素正确、后续错）；改为元素级 `tileOffset` 后 `Verification PASSED! Shape: (65536,), Max diff: 0.0`（来源平台：a2；证据等级: tool_observed）。随后多 shape 精度验收脚本 `precision_test.py` 8 个 shape（256/4096/8192/16384/65536/262144/1048576/4194304）全部 `PASS, max_diff=0.000000e+00`（来源平台：a2；证据等级: tool_observed）。PyTorch 通路 3/3（P1 random / P2 zeros / P3 pos_neg）`PASSED (Max diff=0.0)`（来源平台：a2；证据等级: tool_observed）。

## 预防

- 写多 tile 循环 kernel 时，先把 `CopyIn/CopyOut` 的签名设计成 `CopyIn(count, tileOffset)`，强制每次搬运显式带 offset——无 offset 重载只适合"整段一次搬完"的单 tile 场景。
- 调试"前一段对、后面全错"的搬运问题时，先用一个只做 `DataCopy(xLocal, xGm, count); DataCopy(zGm, zHalf, count)` 的纯拷贝探针跑一遍，打印前 N 与后 N 个元素对比 golden，可快速定位是否"读指针没推进"。
- 索引 GlobalTensor 时单位是**元素**；任何 `/TILE_LENGTH`、`*sizeof(T)` 之类的换算都要在 review 时逐项点破，避免块号/字节/元素混用。

## 坏实践

- 首版试图用 `xGm[offset / TILE_LENGTH]` "对齐到 tile"，把元素 offset 除以 tile 长度变成块号，反而读到错误元素——这是把"循环里第几个 tile"和"GlobalTensor 内第几个元素"两个概念混淆的典型错误。
- 中间一度把 `DataCopy(xLocal, xGm, count)` 的无 offset 重载当作"会自动按调用次数推进"来用——AscendC 的 DataCopy 是无状态的 DMA，不会记忆上次读到哪里。

## 误诊

- 看到输出"几乎全错、max_diff≈20"时先怀疑过"是不是 Cast 升精度逻辑错了 / 是不是输出全零"，于是退化成"先去掉升精度、直接 FP16 Add"再测——结果还是 max_diff≈19.9，说明根因不在计算精度层，而在搬运层。纯拷贝探针才把根因锁死到 DataCopy offset。

## 失效边界

- 推断可迁移到 Ascend 950 / A3（同属 AscendC Vector 数据搬运模型，`DataCopy` + `SetGlobalBuffer` 语义一致），但**未验证，推断可迁移**——950/a3 上未实测。
- `DataCopyPad` 的带 pad 参数重载 offset 由参数结构体控制，本卡的 `xGm[tileOffset]` 元素索引结论不直接套用。
- 多核 `blockNum>1` 下 `numPerCore`/`tailNumLastCore` 的切分正确性在本 trace 未实测（首版固定单核）。