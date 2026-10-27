---
type: "Runbook"
title: "dav_3510 核数用 GetBlockNum 非 GetBlockDim"
description: "arch35/DAV_3510 kernel 侧查询核数的 intra-kernel API 是 GetBlockNum()（kernel_operator_group_barrier_impl.h 声明，返回 int64_t 总核数），写 GetBlockDim() 报 use of undeclared identifier；多核切分用 GetBlockNum() 取总核数、GetBlockIdx() 取当前核号。"
tags: [task.debugging, topic.compilation, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-17T03:16:41Z'
updated_at: '2026-09-17T03:16:41Z'
platforms: ["950"]
---

# dav_3510 核数用 GetBlockNum 非 GetBlockDim

## 现象

在 Ascend950PR（DAV_3510/arch35）上给动态量化算子写 kernel 的多核切分：按 token 均分到各核，代码按其他平台/旧文档的写法用 `GetBlockDim()` 取核数：

```cpp
    // 多核切分
    uint32_t blockDim = GetBlockDim();
    uint64_t tokensPerCore = (totalTokens_ + blockDim - 1) / blockDim;
    myStart_ = GetBlockIdx() * tokensPerCore;
    myEnd_ = (myStart_ + tokensPerCore > totalTokens_) ? totalTokens_ : (myStart_ + tokensPerCore);
```

`bash build.sh --soc=ascend950 -j8` 的 kernel 编译阶段（TBE opc CompileStage.INFERCHANNEL）报：

```
op_kernel/arch35/bench_no_harness_dynamic_quant.h:90:25: error: use of undeclared identifier 'GetBlockDim'; did you mean 'GetBlockNum'?
    uint32_t blockDim = GetBlockDim();
                        ^~~~~~~~~~~
                        GetBlockNum
<CANN 安装目录>/x86_64-linux/asc/impl/basic_api/dav_3510/core_mng/roc/kernel_operator_group_barrier_impl.h:26:27: note: 'GetBlockNum' declared here
__aicore__ inline int64_t GetBlockNum();
```

## 适用场景

- 工程形态：AscendC kernel（op_kernel，arch35），kernel 内自切多核（不依赖 host tiling 下发 blockDim 的写法）。
- API 路径：intra-kernel 核管理 API：`GetBlockNum()` / `GetBlockIdx()`（dav_3510 core_mng/roc 头）。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：same_operator 家族以上的 arch35 kernel 通用；属平台 API 事实。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：host 侧（op_host tiling）取核数应使用 `platform_ascendc::PlatformAscendC::GetCoreNumAiv()`，不是本卡 API（本会话 tiling 侧即如此，两处勿混）；其他架构（如 2201）的核数 API 未逐一验证。

## 根因

dav_3510（arch35）的 kernel 内核管理 API 面只有 `GetBlockNum()`（返回总核数，`int64_t`）与 `GetBlockIdx()`（当前核号），没有 `GetBlockDim()` 这个标识符；从其他平台/旧资料迁移多核切分代码时沿用旧名即编译失败。编译器会直接给出正确建议（"did you mean 'GetBlockNum'"）。

## 解决配方

修复前：

```cpp
    // 多核切分
    uint32_t blockDim = GetBlockDim();
    uint64_t tokensPerCore = (totalTokens_ + blockDim - 1) / blockDim;
    myStart_ = GetBlockIdx() * tokensPerCore;
    myEnd_ = (myStart_ + tokensPerCore > totalTokens_) ? totalTokens_ : (myStart_ + tokensPerCore);
```

修复后（GetBlockNum 返回 int64_t，显式收窄到 uint32_t）：

```cpp
    // 多核切分
    uint32_t blockDim = static_cast<uint32_t>(GetBlockNum());
    uint64_t tokensPerCore = (totalTokens_ + blockDim - 1) / blockDim;
    myStart_ = GetBlockIdx() * tokensPerCore;
    myEnd_ = (myStart_ + tokensPerCore > totalTokens_) ? totalTokens_ : (myStart_ + tokensPerCore);
```

配套写法（本会话同 kernel 的空任务早退，避免无任务核空跑循环）：

```cpp
template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::Process()
{
    if (myStart_ >= myEnd_) {
        return;
    }
    for (uint64_t tokenIdx = myStart_; tokenIdx < myEnd_; tokenIdx++) {
        CopyIn(tokenIdx);
        Compute(tokenIdx);
        CopyOut(tokenIdx);
    }
}
```

## 设计决策

- kernel 内自取核数（GetBlockNum）与 host tiling 下发（PlatformAscendC::GetCoreNumAiv → TilingData/SetBlockDim）两条路线本会话并存：tiling 侧负责 blockDim 上报，kernel 侧自行计算 token 均分——两侧数值一致（同为 AIV 核数），kernel 侧自取可少传一个 tiling 字段，采用。
- 当前方案属 recommended_fastpath。

## 验证信号

问题与验证来源：bench_no_harness_dynamic_quant 算子迭代一 kernel 首编译（Ascend950PR、CANN 9.1.0）失败与修复后复编。

- 修复前 kernel 编译报 `use of undeclared identifier 'GetBlockDim'; did you mean 'GetBlockNum'?`，opc compile failed、二进制未生成（来源平台：950；证据等级：tool_observed）。
- 修复后重编译成功，Kernel 二进制生成（来源平台：950；证据等级：tool_observed）。
- 修后多核切分在 ST 真机 5/5 用例（含 [128,256]、[1024,1024]、[2,4,128,256] 等多 token shape）上输出与 golden 对齐（y max_diff=0~1、scale MERE=0/MARE=0），多核切分正确性实测通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `GetBlockNum()`/`GetBlockIdx()` 绑定 dav_3510 的 intra-kernel API 面（CANN 9.1.0）；其他架构代际未验证，迁移前 grep 对应 arch 的 core_mng 头。
- 本卡只覆盖"kernel 内查询"；host 侧取核数用 PlatformAscendC 接口，不适用本卡。

**未决项**：

- 无（该 API 语义清楚，编译器报错即自明；无未追根因）。
## 坏实践

- 从旧平台代码/旧文档照搬 `GetBlockDim()` 到 arch35 kernel——该标识符在 dav_3510 API 面不存在，直接编译失败（虽然编译器会给建议，但首轮编译即失败浪费一轮构建）。

## 误诊

- 无。
