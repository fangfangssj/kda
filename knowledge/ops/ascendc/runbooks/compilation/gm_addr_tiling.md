---
type: "Runbook"
title: "GM_ADDR 标量不可解引用为 tiling 结构体指针"
description: "Kernel 直调中 Init 形参以 GM_ADDR 接收 tiling 并直接 tiling->field 解引用时，bisheng 把 GM_ADDR 展开为 __gm__ uint8_t 标量，报 'is not a structure or union'；判别为 GM_ADDR 是 GM 标量字节而非指针，须在入口声明 typed __gm__ 指针形参。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T13:28:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["kernel_entry"]
---

# GM_ADDR 标量不可解引用为 tiling 结构体指针

## 适用场景

- 工程形态：Kernel 直调（`<<<>>>` 调用、`extern "C" __global__` 入口），非 aclnn 注册算子。
- API 路径：kernel 入口 `extern "C" __global__ ... void <op>_kernel(GM_ADDR ..., GM_ADDR tiling)`；类内 `Init` 与 `SetGlobalBuffer`、`InitBuffer`。
- 验证过的平台与 CANN 版本：Ascend 910B3 / DAV_2201 / CANN 9.0.0（platforms: [a2]，bisheng 编译器）。
- 适用粒度：`operator_family` — 适用于所有用 Kernel 直调模板、需要经 GM_ADDR 入口传入 TilingData 结构体的 SIMD Vector 算子。
- 路线优先级：`recommended_fastpath` — 入口直接声明 typed `__gm__` 指针形参是稳定的可编译写法。
- 不适用/未验证场景：aclnn 注册算子（tiling 经 TilingParse/GetTilingData 下沉，不走 GM_ADDR 入口）；RegBase/SIMT 路线（DAV_3510，本 trace 未验证）。

## 判别方法

触发（tool_observed）：在用 Kernel 直调方式开发 relu_custom（输入/输出 float16，目标 Ascend 910B3 / DAV_2201 / CANN 9.0.0）时，初次 `cd build && make -j4` 编译 `op_kernel/relu_custom_kernel.asc` 报 12 条同一根因的错误，错误信息逐字如下（build-error，从 raw_source 行 6718-6790 逐字恢复，normalized 的对应 tool_result 被 markdown_transcript adapter 丢失）：

```
op_kernel/relu_custom_kernel.asc:29:47: error: member reference base type '__gm__ uint8_t' (aka '__gm__ unsigned char') is not a structure or union
        bool isLastBlock = (blockIdx == tiling->blockNum - 1);
                                        ~~~~~~^ ~~~~~~~~
op_kernel/relu_custom_kernel.asc:32:45: error: member reference base type '__gm__ uint8_t' (aka '__gm__ unsigned char') is not a structure or union
        blockDataLen = isLastBlock ? (tiling->dim0 - (int64_t)(tiling->blockNum - 1) * tiling->blockFormer)
                                      ~~~~~~^ ~~~~~~
op_kernel/relu_custom_kernel.asc:45:43: error: member reference base type '__gm__ uint8_t' (aka '__gm__ unsigned char') is not a structure or union
        pipe.InitBuffer(inQueue, 2, tiling->ubFormer * sizeof(half));
                                    ~~~~~~^ ~~~~~~~~
12 errors generated.
```

触发态代码特征（来自错误定位行）：kernel 入口签名 `extern "C" __global__ __aicore__ void relu_custom_kernel(GM_ADDR x, GM_ADDR z, GM_ADDR tiling)`，`Init` 形参也声明为 `GM_ADDR tiling`，函数体内直接 `tiling->blockNum`、`tiling->ubFormer` 等解引用。

探针（normalized trace 中 tool_result 完整保留，非 redacted）：

1. `grep` 官方 add_custom 模板（seq 192，tool_observed），观察模板如何处理 tiling 形参。
2. `rg -n 'GM_ADDR' .../platform_ascendc/platform_define.h`（seq 193，tool_observed），确认 GM_ADDR 在该平台头是否定义。

原始观测（grep 模板，seq 192，verbatim）：

```
37:    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
38:                                const __gm__ AddTilingData* tiling)
40:        this->tiling = tiling;
42:        if (blockIdx < tiling->blockNum - 1) {
130:extern "C" __global__ __vector__ void add_custom_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
134:    op.Init(x, y, z, (__gm__ AddTilingData*)tiling);
```

`rg GM_ADDR .../platform_define.h` 返回 `(no output)`（seq 193，verbatim）——即 GM_ADDR 不在该平台定义头中定义。

观测：错误信息自身显示 `GM_ADDR` 被展开为 `__gm__ uint8_t`（aka `__gm__ unsigned char`），即全局内存地址空间下的**单字节标量**，而非指针类型；模板的 Init tiling 形参是 typed `const __gm__ AddTilingData*` 指针，入口处用 `(__gm__ AddTilingData*)tiling` 显式 cast；GM_ADDR 不在 `platform_define.h` 定义（由 kernel_operator 体系提供，展开为标量）。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：`__gm__` 是地址空间限定符（标记数据位于 Global Memory），修饰在 `uint8_t` 上得到的是"GM 中的一个 byte 值"，不是可解引用的指针。因此 `tiling->blockNum` 在 `__gm__ uint8_t` 上做成员访问，编译器判 "base type is not a structure or union"。本 trace 误把 Init 的 tiling 形参也写成 `GM_ADDR` 且未在入口 cast 成 typed 指针，导致标量被当作结构体指针解引用。模板之所以能编译，是因为 Init 形参用 `const __gm__ AddTilingData*`、入口用 `(__gm__ AddTilingData*)tiling` cast。判别结论：凡经 GM_ADDR 入口传 TilingData 的，不要在 Init 形参里继续用 `GM_ADDR tiling`；入口直接声明 typed `__gm__` 指针形参（或 Init 形参用 typed `__gm__` 指针并在入口 cast），使 tiling 以可解引用的指针类型进入函数体。

## 验证信号

来源：本 trace 在 Ascend 910B3 / DAV_2201 / CANN 9.0.0（platforms: [a2]）上对 `op_kernel/relu_custom_kernel.asc` 做 `make -j4` 编译；模板与头文件探针的 tool_result 在 normalized trace 中完整保留，make 输出被 adapter 丢失后从 raw_source 逐字恢复。

- `make -j4` 报 12 条 `member reference base type '__gm__ uint8_t' (aka '__gm__ unsigned char') is not a structure or union`（行 29/32/33/36/37/40/45/46）。（来源平台：a2；证据等级：tool_observed）
- `grep` add_custom 模板输出证实其 Init 用 `const __gm__ AddTilingData* tiling`、入口用 `(__gm__ AddTilingData*)tiling` cast。（来源平台：a2；证据等级：tool_observed）
- `rg -n 'GM_ADDR' .../platform_define.h` 返回 `(no output)`，即 GM_ADDR 不在该平台头定义。（来源平台：a2；证据等级：tool_observed）
- 改为 typed `__gm__` 指针形参后 `make -j4` 输出 `[100%] Linking ASC executable relu_custom`、`[100%] Built target relu_custom`、`[83%] Built target relu_custom_ops`，编译通过。（来源平台：a2；证据等级：tool_observed）

## 失效边界

- 推断可迁移至 DAV_3510（a3/950）SIMD Vector 路线的同类 Kernel 直调工程——未验证，推断可迁移（`__gm__` / `GM_ADDR` 语义在 SIMD 路线一致）。
- 未验证：RegBase/SIMT 路线（DAV_3510 新编程模型）下 GM_ADDR 是否仍展开为 `__gm__ uint8_t` 标量——SIMT 有独立入口约定，本卡结论不直接外推。
- 未验证：aclnn 注册算子路径（tiling 经 `GetTilingData` 在 device 侧获取，不走 `GM_ADDR` 入口）。

## 坏实践

- 在 Init 形参声明 `GM_ADDR tiling` 后于函数体内 `this->tiling = (__gm__ ReLUTilingData*)tiling` 再解引用——cast 作用于 `__gm__ uint8_t` 标量，bisheng 不接受标量→指针的转换。
- 把 `GM_ADDR` 当通用指针类型重复使用（x/z/tiling 全用 `GM_ADDR`），不区分数据指针与 tiling 结构体指针的语义差异。

## 误诊

- 早期误判为"`__aicore__` 限定符不对、应改用 `__global__ __vector__`"——限定符不是该编译错误的根因（后续验证 `__aicore__` 与 `__vector__` 均能编译，真正根因是 GM_ADDR 标量解引用）。此误判导致一度改限定符，未解决问题。
- 误判为"模板的 `(__gm__ AddTilingData*)tiling` cast 在本工程也会工作"——实测在本工程 CMake/`--npu-arch=dav-2201` 组合下从 `__gm__ uint8_t` 标量 cast 到指针仍报错，需用入口直接声明 typed `__gm__` 形参。
