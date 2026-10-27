---
type: "Runbook"
title: "GET_TILING_DATA_WITH_STRUCT 限入口函数"
description: "GET_TILING_DATA_WITH_STRUCT(T, var, tiling) 的合法位置是 __global__ __aicore__ kernel 入口函数体内（与 REGISTER_TILING_DEFAULT 成对）；在 kernel 类 Init 方法内调用会报 does not refer to a value / use of undeclared identifier。类 Init 改收 const T* 指针。"
tags: ["task.implementation", "topic.compilation", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# GET_TILING_DATA_WITH_STRUCT 限入口函数

## 现象

MlaProlog 算子 kernel 类（`op_kernel/arch35/mla_prolog.h`）的 `Init` 方法最初接收 `GM_ADDR tiling` 参数并在类方法体内调用 `GetTilingDataWithStruct(MlaPrologTilingData, tilingData, tiling);`，随后以 `tilingData.B` 等值方式访问字段。kernel 编译报错（原文）：

```
op_kernel/arch35/mla_prolog.h:53:33: error: 'MlaPrologTilingData' does not refer to a value
op_kernel/arch35/mla_prolog.h:53:54: error: use of undeclared identifier 'tilingData'
op_kernel/arch35/mla_prolog.h:55:14: error: use of undeclared identifier 'tilingData'
```

`GET_TILING_DATA_WITH_STRUCT` 是供 kernel 入口函数使用的宏（配合 `REGISTER_TILING_DEFAULT`），在类成员函数里调用展开后类型与变量都不可见。

## 适用场景

- 工程形态：自定义算子 kernel 入口与 kernel 类的 tiling 数据传递（`REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 宏对）。
- API 路径：`REGISTER_TILING_DEFAULT(T)`、`GET_TILING_DATA_WITH_STRUCT(T, var, tiling)`；类 `Init(..., const T* tilingData)`。
- 验证过的平台与 CANN 版本：Ascend950PR 目标（`--soc=ascend950` 构建链），CANN 9.2.0 目录（9.1.0 基线）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本所有手写 kernel 类结构的算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：无需 TilingData 的 kernel；其他 tiling 宏家族（如 GET_TILING_DATA 无 struct 版）未逐一覆盖；宏不接受模板类型、直调工程宏不可用两个相邻 facet 见对应卡。

## 根因

`GET_TILING_DATA_WITH_STRUCT(T, var, tiling)` 宏的合法位置是 `__global__ __aicore__` kernel 入口函数体内（与 `REGISTER_TILING_DEFAULT(T)` 成对），它从 GM tiling 地址解出 `T` 类型的局部变量 `var`；类方法内调用时宏展开依赖的上下文不成立，且 `var` 生命周期也不会进入类成员。

## 解决配方

宏上移到 kernel 入口，类 `Init` 改为接收 `const T*` 指针，字段访问全部改为指针箭头。

kernel 入口（verbatim，`op_kernel/mla_prolog_arch35.cpp`）：

```cpp
template <uint32_t SCH_MODE>
__global__ __aicore__ void mla_prolog(
    GM_ADDR token_x, GM_ADDR w_dq, GM_ADDR w_uq_qr, GM_ADDR w_uk, GM_ADDR w_dkv_kr,
    GM_ADDR rmsnorm_gamma_cq, GM_ADDR rmsnorm_gamma_ckv, GM_ADDR rope_sin, GM_ADDR rope_cos,
    GM_ADDR query, GM_ADDR query_rope, GM_ADDR c_kv, GM_ADDR k_rope,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlaPrologTilingData);
    GET_TILING_DATA_WITH_STRUCT(MlaPrologTilingData, tilingData, tiling);
    NsMlaProlog::MlaPrologKernel<SCH_MODE> op;
    op.Init(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr,
            rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
            query, query_rope, c_kv, k_rope, workspace, &tilingData);
    op.Process();
}
```

类 `Init` 签名与字段访问（verbatim，`op_kernel/arch35/mla_prolog.h`）：

```cpp
    // 修复前：Init(..., GM_ADDR tiling) 内调用
    // GetTilingDataWithStruct(MlaPrologTilingData, tilingData, tiling);
    // B_ = tilingData.B; ...

    // 修复后：
    __aicore__ inline void Init(
        GM_ADDR token_x, GM_ADDR w_dq, GM_ADDR w_uq_qr, GM_ADDR w_uk, GM_ADDR w_dkv_kr,
        GM_ADDR rmsnorm_gamma_cq, GM_ADDR rmsnorm_gamma_ckv, GM_ADDR rope_sin, GM_ADDR rope_cos,
        GM_ADDR query, GM_ADDR query_rope, GM_ADDR c_kv, GM_ADDR k_rope,
        GM_ADDR workspace, const MlaPrologTilingData* tilingData)
    {
        // 获取 TilingData
        B_ = tilingData->B; S_ = tilingData->S; He_ = tilingData->He; N_ = tilingData->N;
        Hcq_ = tilingData->Hcq; D_ = tilingData->D; Dr_ = tilingData->Dr; Hckv_ = tilingData->Hckv;
        epsCq_ = tilingData->epsilonCq; epsCkv_ = tilingData->epsilonCkv;
        blockDim_ = tilingData->blockDim; bsTotal_ = tilingData->bsTotal;
        bsPerCore_ = tilingData->bsPerCore; bsTail_ = tilingData->bsTail;
        mmTileN_ = tilingData->mmTileN;
```

workspace 偏移取用同式改指针（遗留的 `tilingData.` 逐一改 `tilingData->`）：

```cpp
        __gm__ uint8_t* wsBase = reinterpret_cast<__gm__ uint8_t*>(workspace);
        gmCqRaw_.SetGlobalBuffer((__gm__ bfloat16_t*)(wsBase + tilingData->wsOffsetCqRaw));
        gmQr_.SetGlobalBuffer((__gm__ bfloat16_t*)(wsBase + tilingData->wsOffsetQr));
        gmDkvKr_.SetGlobalBuffer((__gm__ bfloat16_t*)(wsBase + tilingData->wsOffsetDkvKr));
```

参照先例（GRU 入口，verbatim）：

```cpp
    REGISTER_TILING_DEFAULT(BenchNoHarnessGruTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGruTilingData, tilingData, tiling);
    ...
    op.Init(x, weight_ih, weight_hh, bias_ih, bias_hh, h0,
            y, hn, workspace, &tilingData);
```

## 预防

- 写 kernel 类时固定套用三段式：入口 `REGISTER_TILING_DEFAULT(T)` + `GET_TILING_DATA_WITH_STRUCT(T, tilingData, tiling)` + `op.Init(..., &tilingData)`；类 `Init` 尾参恒为 `const T*`，成员内只用 `->`。

## 验证信号

来源说明：该算子 kernel 编译修复会话（Ascend950PR 构建链，CANN 9.2.0 目录）。

- 修复前：`'MlaPrologTilingData' does not refer to a value` / `use of undeclared identifier 'tilingData'` 系列错误（来源平台：950 构建链；证据等级：tool_observed）。
- 修复后重建：上述错误全部消除，kernel 编译通过：`[100%] Built target MlaProlog_ascend950`、`[100%] Built target binary`（来源平台：950 构建链；证据等级：tool_observed）。
- 二进制生成：`build/op_kernel/ascendc_kernels/binary/ascend950/mla_prolog/MlaProlog_<hash>.o`（来源平台：950 构建链；证据等级：tool_observed）。

## 失效边界

- 针对 CANN 9.2.0 目录（9.1.0 基线）的宏契约；其他版本宏语义可能调整（未验证）。
- kernel 运行期 tiling 数据正确性未上板验证（本卡为编译级修复）。
- 相邻 facet：宏直传模板类型（带 `<>`）报解析错见 [宏不接受模板类型卡](get_tiling_data_with_struct.md)；`<<<>>>` 直调工程无此宏见 [直调不可用宏卡](direct_invoke_unavailable_get_tiling_data_with_struct.md)。

**未决项**：

- 无（该编译问题已闭环；运行期验证随真机 ST）。
## 坏实践

- 把 `GET_TILING_DATA_WITH_STRUCT` 放进类成员函数、或把 GM `tiling` 裸指针当 `T*` 用——前者编译失败，后者读错内存。

## 误诊

- 无（报错行号直接指向类方法内的宏调用点）。
