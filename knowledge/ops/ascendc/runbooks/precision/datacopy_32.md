---
type: "Runbook"
title: "DataCopy 非32字节对齐尾块丢字节"
description: "AscendC::DataCopy 按硬件 32 字节块长搬运，count*sizeof(dtype) 非 32B 倍数时向下取整丢弃尾部字节，UB↔GM 尾块元素被写成 0。判别=逐元素对比定位首个 mismatch index，若集中在尾部且 mismatch 数=count−floor(count*sizeof/32)*32/sizeof 即锁定。修复方向 DataCopyPad 显式 blockLen。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["isPad"]
---

# DataCopy 非32字节对齐尾块丢字节

## 现象

用 Kernel 直调（`<<<>>>`）方式开发 ReLU 算子（`KernelRelu`，`TQue<TPosition::VECIN,1>`/`TQue<TPosition::VECOUT,1>` + Double Buffer `BUFFER_NUM=2`，多核按 512 元素对齐切分，计算 `AscendC::Relu(zLocal, xLocal, count)`）。CopyIn/CopyOut 用 `DataCopy(dst, src, count)`，`count` 为本 UB 块实际元素数。首轮 dim0=32768（4096×8，blockNum=16）standalone 与 PyTorch 通路 5/5 用例均 PASSED（Max diff=0.0）。随后用 PyTorch 通路对非对齐 shape 扫描：

```python
for sz in [100, 1000, 7777, 1]:
    x = torch.randn(sz, dtype=torch.float16).npu()
    y = torch.ops.npu.relu_custom(x)
    expected = torch.relu(x)
    diff = torch.max(torch.abs(y.cpu().to(torch.float32) - expected.cpu().to(torch.float32))).item()
    ok = torch.allclose(y.cpu(), expected.cpu(), atol=0, rtol=0)
    print(f'Shape [{sz}]: {"PASSED" if ok else "FAILED"} (Max diff={diff})')
```

实测输出（来源平台：a2；证据等级: tool_observed）：

```
Shape [100]: FAILED (Max diff=0.66015625)
Shape [1000]: FAILED (Max diff=2.857421875)
Shape [7777]: PASSED (Max diff=0.0)
Shape [1]: FAILED (Max diff=0.6572265625)
```

进一步对 32/64/128/256/512/513/1000/1024/1025/2048/4096 扫描：32、64、128、256、512、1024、2048、4096 全 PASSED；513、1000、1025 FAILED。规律是：当 `dim0 * sizeof(half)` 不是 32 字节倍数时失败。

## 适用场景

- **工程形态**：Kernel 直调（`<<<>>>` / triple-chevron）standalone 与 PyTorch 通路双验证的 elementwise 算子开发。
- **API 路径**：`AscendC::DataCopy(LocalTensor, GlobalTensor, count)` 与 `AscendC::DataCopy(GlobalTensor, LocalTensor, count)` 的 UB↔GM 搬运；判别对照 API 为 `AscendC::DataCopyPad`（显式 `blockLen`）。
- **验证过的平台与 CANN 版本**：Ascend 910B3（NpuArch `dav-2201`，Atlas A2 训练系列），CANN 9.0.0，`__aicore__` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `AscendC::InitSocState()`。与 frontmatter `platforms: [a2]` 严格一致。
- **适用粒度**：`operator_family` —— 任何在 UB↔GM 之间用 `DataCopy` 搬运非 32 字节对齐字节量的 elementwise/vector 算子。
- **路线优先级**：`diagnostic_only` —— 本卡只给判别方法与根因解释；逐字 after 修复代码与修后闭环信号未从归一化 trace 恢复。
- **不适用或未验证场景**：Cube/Matmul 路线、需要跨步(stride)搬运的场景未在本 trace 验证；32 字节倍数字节量的搬运不受此问题影响，可继续用 `DataCopy`。

## 判别方法

**触发条件**（tool_observed 失败症状）：elementwise 算子在 32 字节倍数字节量的 shape（如 dim0=32768/4096/1024/512/256/128/64/32）上 PASSED、Max diff=0.0，但在非 32 字节倍数字节量的 shape（如 dim0=100/1000/1/513/1025，对应字节量 200B/2000B/2B/1026B/2050B）上 FAILED，Max diff 量级与输入正值相当（0.66~2.86）。失败 shape 的共同特征是 `dim0 * sizeof(dtype)` 对 32 取余非零。

**可执行探针**：
1. 对一个失败 shape（推荐 dim0=100，half）固定随机种子做逐元素对比，打印首个/末个 mismatch index、mismatch 总数与每个 mismatch 处的 `got/expected/input`：

   ```python
   import numpy as np, torch
   np.random.seed(42)
   x = torch.tensor(np.random.randn(100).astype(np.float16)).npu()
   y = torch.ops.npu.relu_custom(x)
   xg, yg = x.cpu().to(torch.float32), y.cpu().to(torch.float32)
   mism = (yg != torch.relu(xg)).nonzero().squeeze(1)
   print(f'mismatches={len(mism)} first={mism[0].item() if len(mism) else -1} last={mism[-1].item() if len(mism) else -1}')
   for i in mism: print(f'[{i}]: got={yg[i].item():.6f}, expected={torch.relu(xg)[i].item():.6f}, input={xg[i].item():.6f}')
   ```

2. 检查 mismatch 是否集中在尾部、且 mismatch 数等于 `dim0 - floor(dim0*sizeof(dtype)/32)*32/sizeof(dtype)`：若是，即锁定为 `DataCopy` 32 字节块长丢字节。
3. 打开算子 `op_kernel/*.asc`，grep `DataCopy(`，确认 CopyIn/CopyOut 是否用 `DataCopy(dst, src, count)` 形式（`count` 为元素数而非字节数，且未传 `blockLen`）；若是即与症状一致。

**原始观测**（Ascend 910B3, dav-2201, CANN 9.0.0，dim0=100 half 逐元素对比）：

```
dim0=100, total mismatches:
  [ 96]: got=0.000000, expected=0.296143, input=0.296143
  [ 97]: got=0.000000, expected=0.260986, input=0.260986
  [ 98]: got=0.000000, expected=0.005112, input=0.005112
Total mismatches: 3
First mismatch: index 96
Last mismatch: index 98
```

100 个 half = 200 字节，只有 index 96-99（最后 4 个元素 = 8 字节）被错误写成 0，前 96 个元素完全正确。`mismatch_count(3~4) == dim0 - floor(200/32)*32/2 == 100 - 96 == 4`（探针报告 3 是因 index 99 恰好输入为 0、ReLU 后仍为 0 不触发 mismatch），首 mismatch=96 = `floor(200/32)*32/2`，与丢字节公式一致。

**机制解释**（根因分析，属 interpretation）：`AscendC::DataCopy` 在硬件上按块长（Atlas A2 上为 32 字节）粒度搬运。当传入的 `count * sizeof(dtype)` 不是 32 字节倍数时，硬件只搬运 `floor(字节数 / 32) * 32` 字节，丢弃尾部不足 32 字节的部分。对 dim0=100 的 half：总字节 200B → 块数 `floor(200/32)=6` → 实际搬运 `6×32=192B=96 元素` → 丢弃 `200−192=8B=4 元素`（index 96-99）。CopyIn 与 CopyOut 两端都受影响：CopyIn 没把最后 8 字节搬进 UB（UB 对应位置是未初始化值），CopyOut 把 UB 中对应位置写回 GM 时也只写了 192 字节，最终 GM 输出的 index 96-99 保持为 0（`empty_like` 初始化的零值未被覆盖）。注意 `AscendC::Relu(dst, src, count)` 的 `count` 是"参与计算的元素个数"、不要求 32 字节对齐，LocalTensor 起始地址 32 字节对齐由 `pipe.InitBuffer` 的 UB 分配保证——问题完全出在 `DataCopy` 的搬运粒度，不在 `Relu`。机制链不跳步：`count*sizeof(half)` 非 32B 倍数 → `DataCopy` 向下取整到 32B 块边界 → 尾部不足 32B 的字节不搬运 → CopyIn 漏搬尾元素 / CopyOut 漏写尾元素 → GM 尾元素保持初始零值 → Max diff 量级等于被丢输入的正值。

## 预防

- 任何在 UB↔GM 之间用 `DataCopy(dst, src, count)` 搬运的 elementwise/vector 算子，只要 `count * sizeof(dtype)` 可能非 32 字节倍数，就必须用 `DataCopyPad` 显式传 `blockLen`；不要假设 `count` 会被硬件向上补齐——它会向下取整丢字节。
- 多核切分按 `ELEM_ALIGN_FACTOR=512` 对齐只保证多核边界对齐，不保证 UB 尾块（`ubTailOfFormerBlock`/`ubTailOfTailBlock`）的字节量是 32 的倍数。UB 尾块的 `count` 必须走 `DataCopyPad`。
- 排查此类问题时优先用逐元素对比定位首个 mismatch index：若 mismatch 集中在尾部且 `mismatch_count = dim0 - floor(dim0*sizeof/32)*32/sizeof`，即可锁定为 DataCopy 32 字节块长丢字节。

## 坏实践

- 在测试侧把 `atol/rtol` 放宽以"通过"非对齐 shape——会掩盖尾部丢字节的真实缺陷（本 trace 中 `verify_result.py` 用 `rtol=0, atol=0` 精确匹配，未放松）。
- 只测 32 字节倍数的 shape（如 32768/4096/1024）就放行——会漏掉非对齐尾块；非对齐 shape 扫描须包含 `dim0*sizeof(dtype) % 32 != 0` 的用例（如 100/1000/1/513/1025）。

## 验证信号

- 非对齐 shape dim0=100 FAILED (Max diff=0.66015625)、dim0=1000 FAILED (Max diff=2.857421875)、dim0=1 FAILED (Max diff=0.6572265625)、dim0=513 FAILED (Max diff=0.267639)、dim0=1025 FAILED (Max diff=0.559570)；32B 倍数字节量 shape（32/64/128/256/512/1024/2048/4096）全 PASSED（来源平台：a2；证据等级：tool_observed）
- dim0=100 逐元素对比：index 96-99 被写成 0，前 96 元素正确，`100 - 96 = 4` 个元素 = 8 字节 = 200B − 192B 丢弃，首 mismatch=96 = `floor(200/32)*32/2`（来源平台：a2；证据等级：tool_observed）
- 缺陷存在于源码：`op_kernel/relu_custom_kernel.asc` 的 `KernelRelu::CopyIn`/`KernelRelu::CopyOut` 用 `DataCopy(xLocal, xGm, count)` / `DataCopy(zGm, zLocal, count)`，`count` 为元素数、未传 `blockLen`（来源平台：a2；证据等级：artifact_verified）

## 失效边界

- 平台边界：本 trace 仅在 Ascend 910B3 (dav-2201, Atlas A2) 实测。Atlas A3 (a3)、Ascend950 (950) 未验证，推断可迁移——DataCopy 的 32 字节块长对齐机制是 AscendC 通用约束，但具体块长与 `blockLen` 字段语义在不同 NpuArch 上可能不同，迁移前须在该平台用非对齐 shape 复测。
- dtype 边界：本 trace 只验证 `half`(float16)。`int32_t`/`float` 等其他 dtype 的非对齐尾块同理应使用 DataCopyPad，但 `count * sizeof(dtype)` 的具体取整结果不同，未在本 trace 验证。
- 路线边界：Cube/Matmul 路线、跨步(stride)搬运、`DataCopy` 的带 stride 重载未在本 trace 验证。