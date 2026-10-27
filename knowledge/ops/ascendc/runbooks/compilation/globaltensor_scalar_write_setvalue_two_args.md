---
type: "Runbook"
title: "GM 标量写须 SetValue(index, value) 两参"
description: "AscendC kernel 里 gm[i].SetValue(v) 的链式标量写编译报 too few arguments to function call, expected 2, have 1；正确形式是 gm.SetValue(index, value)，标量读是 gm.GetValue(index)（单参）。"
tags: ["task.implementation", "topic.compilation", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# GM 标量写须 SetValue(index, value) 两参

## 现象

`bench_no_harness_unique` 算子（ascend950，CANN 9.1.0）迭代三 kernel 的输出阶段，为避开小数据量下 `DataCopyPad` 的 32 字节对齐问题，改用标量逐元素写 y。按容器直觉写了 `outputGMY_[offset + i].SetValue(...)`，`bash build.sh --soc=ascend950 -j8` 的 kernel 编译阶段报（四个 TilingKey 变体各一次）：

```
op_kernel/arch35/<op>.h:1194:69: error: too few arguments to function call, expected 2, have 1
[ERROR] TBE(...) ld.lld: error: cannot open .../kernel_meta_<Op>_.../_3.o: No such file or directory
```

报错点即该行链式写法；改成 `GlobalTensor::SetValue(index, value)` 两参形式后编译通过。

## 适用场景

- 工程形态：AscendC kernel 开发，GM 侧标量逐元素读写（小数据量/尾部处理/调试通路）。
- API 路径：`GlobalTensor`（`kernel_tensor.h`）：`__aicore__ inline PrimType GetValue(const uint32_t offset)` 单参读、`SetValue(index, value)` 两参写。
- 验证过的平台与 CANN 版本：Ascend950，CANN 9.1.0；仅 950 有实测证据。
- 适用粒度：platform_wide——CANN 9.1.0 全部 AscendC kernel 的 GM 标量访问。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：`LocalTensor`（UB 侧）的 `SetValue(index, value)` 用法一致但对象不同（参数序见对应卡）；其他 CANN 版本未验证。

## 根因

`GlobalTensor` 的标量写不是"取元素引用再赋值"的容器式接口——`gm[i]` 返回对象上的 `SetValue` 是另一个两参签名，单参链式调用不可用；正确形式是 `gm.SetValue(index, value)`。

## 解决配方

before（编译失败，verbatim，Phase 3 输出的小数据量标量分支）：

```cpp
            // 标量写入（避免 DataCopyPad 32 字节对齐溢出）
            outputQueue_.EnQue(valLocal);
            auto outLocal = outputQueue_.template DeQue<D_T_X>();
            for (int64_t i = 0; i < cnt; i++) {
                outputGMY_[offset + i].SetValue(outLocal.GetValue(i));
            }
            outputQueue_.FreeTensor(outLocal);
```

after（编译通过）：

```cpp
            for (int64_t i = 0; i < cnt; i++) {
                outputGMY_.SetValue(offset + i, outLocal.GetValue(i));
            }
```

配套的标量读同轨迹实测可用（单参）：

```cpp
D_T_X val = inputGMX_.GetValue(blockStart + i);   // GM 标量读
inputLocal.SetValue(i, val);                       // UB 标量写（LocalTensor 同为两参）
```

## 验证信号

来源说明：同一次开发会话的编译闭环（Ascend950 + CANN 9.1.0）；该分支的运行正确性属于另一条未闭合的调试线，不作为本卡验证。

- `outputGMY_[offset + i].SetValue(...)` → `outputGMY_.SetValue(offset + i, ...)` 后 `bash build.sh --soc=ascend950 -j8` 输出 `[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- 同一 kernel 中 `inputGMX_.GetValue(blockStart + i)` 单参标量读、`inputLocal.SetValue(i, val)` UB 两参写在同一次构建中通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"调用形式可编译且该形式在同轨迹被用于 x/y 的 GM 读写"；对 workspace 上的 GlobalTensor 标量访问，同轨迹出现独立的运行期寻址失败（另卡记录），不要把两张卡混为一谈。
- CANN 9.1.0 实测；其他版本未验证（推断可迁移）。

**未决项**：

- 无。
## 预防

- GM/UB 张量的标量访问统一记成"读 `GetValue(index)`、写 `SetValue(index, value)`"，不套用 `std::vector` 的 `v[i] = x` 直觉。
- 遇到 `too few arguments ... expected 2, have 1` 落在张量调用行时，先查该类的真实签名再改调用点。

## 坏实践

- `gm[i].SetValue(v)` 链式写法——本例的直接诱因。

## 误诊

- 无。
