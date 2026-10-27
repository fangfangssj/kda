---
type: "Runbook"
title: Compare/Select 模板参错位编译报错
description: AscendC Compare/Select 的第一个模板参数是源数据类型 T（src0/src1），mask 类型 U 是第二个；写 Compare<uint8_t>(dst,src0,src1,…) 把 mask 类型放到第一位会报 no matching function for call to 'Compare'。去掉显式模板参数由 LocalTensor 实参推导即可。
tags: [task.implementation, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-17T04:12:25Z'
updated_at: '2026-09-17T04:12:25Z'
platforms: ["950"]
---

# Compare/Select 模板参错位编译报错

## 现象

给自定义 argmax 算子（BenchNoHarnessArgMax，Ascend950PR，CANN 9.1.0）迭代二实现 Float-ARA / Int32-ARA 行迭代路径时，kernel 里用"Max + Compare(EQ) + Select"做逐行 first-max 索引更新。写法上把 mask 类型当作第一个显式模板参数：

```cpp
// Float-ARA 路径（错误写法）
Compare<uint8_t>(cmpLocal, newMaxLocal, maxLocal,
                 CMPMODE::EQ, a0Align);
Select<float, uint8_t>(idxLocal, cmpLocal, idxLocal, rowIdxF,
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Align);
```

`bash build.sh --soc=ascend950 -j8` 在 kernel 编译阶段报（TBE/opc 工具链输出）：

```
op_kernel/arch35/bench_no_harness_arg_max.h:577:17: error: no matching function for call to 'Compare'
op_kernel/arch35/bench_no_harness_arg_max.h:780:17: error: no matching function for call to 'Compare'
```

当时代码状态：Float-ARA 与 Int32-ARA 两条路径各有一组 Compare/Select 调用，均按"dst mask 类型在前"的直觉写了显式模板参数。

## 适用场景

- 工程形态：AscendC kernel（`__aicore__`）开发，CCE/opc 工具链编译（build.sh --soc=ascend950）。
- API 路径：`AscendC::Compare` / `AscendC::Compares`（basic_api/impl/kernel_operator_vec_cmpsel_intf_impl.h）、`AscendC::Select`（VSEL_TENSOR_SCALAR_MODE 标量模式，SELMODE 定义于 include/pto/npu/a2a3/TSel.hpp）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0。
- 适用粒度：platform_wide——模板签名语义跨算子通用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：需要覆盖 isSetMask/BinaryConfig 的低阶 Compare 重载（T0,T1,isSetMask,config 四个显式模板参）不在本卡范围；SIMT/pto 路径的 TSel 语义未逐条验证。

## 根因

AscendC Compare 的 Level2 签名是：

```cpp
template <typename T, typename U>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
    CMPMODE cmpMode, uint32_t count)
```

第一个模板参数 T 是**源数据类型**（src0/src1），第二个 U 才是 dst（bit-mask，通常 uint8_t）。`Compare<uint8_t>` 把 T 钉死为 uint8_t，而实参 src0/src1 是 `LocalTensor<float>`/`LocalTensor<int32_t>`，模板推导冲突 → no matching function。Select 的 TENSOR_SCALAR 模式同理：`template <typename T, typename U> Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, T src1, SELMODE, uint32_t count)`，T 是数据类型不是 mask 类型。

## 解决配方

**before**（两条 ARA 路径的错误写法）：

```cpp
// Float-ARA
Compare<uint8_t>(cmpLocal, newMaxLocal, maxLocal,
                 CMPMODE::EQ, a0Align);
Select<float, uint8_t>(idxLocal, cmpLocal, idxLocal, rowIdxF,
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Align);

// Int32-ARA
Compare<uint8_t>(cmpLocal, newMaxLocal, maxLocal,
                 CMPMODE::EQ, a0Align);
Select<float, uint8_t>(idxLocal, cmpLocal, idxLocal, rowIdxF,
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Align);
```

**after**（去掉显式模板参数，交由实参推导）：

```cpp
// Float-ARA
// eq = (newMax == curMax_old) → bit=1 means unchanged (keep old idx)
Compare(cmpLocal, newMaxLocal, maxLocal,
        CMPMODE::EQ, a0Align);
PipeBarrier<PIPE_V>();

// bit=1(unchanged)→keep idxLocal; bit=0(updated)→write rowIdxF
Select(idxLocal, cmpLocal, idxLocal, rowIdxF,
       SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Align);
PipeBarrier<PIPE_V>();

// Int32-ARA
// eq = (newMax == curMax_old) → bit=1 means unchanged
Compare(cmpLocal, newMaxLocal, maxLocal,
        CMPMODE::EQ, a0Align);
PipeBarrier<PIPE_V>();

// bit=1(unchanged)→keep idxLocal; bit=0(updated)→write rowIdxF
Select(idxLocal, cmpLocal, idxLocal, rowIdxF,
       SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Align);
PipeBarrier<PIPE_V>();
```

若确要显式指定，正确顺序是"源类型在前、mask 类型在后"：`Compare<float, uint8_t>(...)`、`Select<float, uint8_t>(...)`——但让 `LocalTensor` 实参自行推导最不易错。向量比较后接 `PipeBarrier<PIPE_V>()` 再进 Select，保证 mask 可见。

## 设计决策

- 选"去显式参数靠推导"而不是"写对显式参数"：显式参数重复了实参类型信息，写错顺序就是本卡故障；推导版本少一处出错面。
- Compares（张量 vs 标量）同名族同样遵守"src 类型在前"约定，简单封装签名为 `template <typename T, typename U> Compares(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE, uint32_t count)`，Int32-AR 路径按 `Compares(maskLocal, inData, chunkMax, CMPMODE::EQ, rAlign_)` 推导调用。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（arch35），CANN 9.1.0；kernel 头文件 `x86_64-linux/ascendc/include/basic_api/impl/kernel_operator_vec_cmpsel_intf_impl.h`（Compare/Compares/Select 实现）。
- 相关文件：`op_kernel/arch35/bench_no_harness_arg_max.h`（ProcessFloatARA / ProcessInt32ARA 的 Compare+Select 行迭代）。

## 预防

- 调 AscendC 二元比较/选择类 API 时记住模板参顺序 = **数据类型在前、mask 类型在后**；默认干脆不写显式模板参数。
- 看到 `no matching function for call to 'Compare'/'Select'` 先查显式模板参数与实参 LocalTensor 的类型是否冲突，再查重载集合。

## 验证信号

以下信号来自迭代二 kernel 开发会话（build.sh 编译输出）：

- 修复前：kernel 编译 7 错误，其中 577/780 两处 `no matching function for call to 'Compare'`（来源平台：950；证据等级：tool_observed）。
- 修复后（同批一并修复同文件其他编译错后）：`[INFO] Build completed successfully!`，5 个 kernel binary（覆盖 10 TilingKey 组合）生成，`grep -iE "[ERROR]|compile failed|error:"` 无输出（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论针对 basic_api 的 Level0/Level2 高层封装；带 BinaryConfig 的底层重载（T0,T1,isSetMask,config）模板语义不同，未在本卡验证。
- NPU 精度验证（ARA 路径上板）不在本卡闭环内，本卡验证止于编译通过。

**未决项**：

- 无。
## 坏实践

- 凭"dst 在参数列表第一位"的直觉把 dst 的类型也放到第一个模板参数位——Compare 的模板参顺序跟随 (src, dst) 语义而非形参顺序。

## 误诊

- 无（本闭环根因唯一且由签名阅读+编译验证确认）。
