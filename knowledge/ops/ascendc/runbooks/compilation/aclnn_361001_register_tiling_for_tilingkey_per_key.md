---
type: "Runbook"
title: 361001 非零 tilingKey 须逐 key 注册
description: TilingFunc 下发非零 tilingKey 而 kernel 入口只 REGISTER_TILING_DEFAULT 时，aclnn 启动报 361001（tiling struct 未按 key 注册）；须对每个合法 key 补 REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == N", Struct)，表达式必须是字符串。
tags: [task.debugging, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# 361001 非零 tilingKey 须逐 key 注册

## 现象

在 Ascend950PR（CANN 9.1.0）上开发 EuclideanNorm 自定义算子（算子仓工程，op_host/arch35/euclidean_norm_tiling_arch35.cpp 为 TilingFunc，op_kernel/euclidean_norm_apt.cpp 为 device 入口）。当公共 TilingFunc 落地后，它会按设计公式 `tilingKey = (isGroup << 2) | (isEmptyTensor << 1) | isTailR` 调用 `context->SetTilingKey(tilingKey)`，合法取值集合为 {0,1,2,4,5}；而此时 device 侧 kernel 仍是占位桩，入口只有四行骨架：`AscendC::InitSocState()`、`REGISTER_TILING_DEFAULT(EuclideanNormTilingData)`、`GET_TILING_DATA(tilingData, tiling)`、`AscendC::PipeBarrier<PIPE_ALL>()`。

在这个状态下跑 aclnn 冒烟测试（`tests/aclnn/run.sh`，两段式 aclnn API，8 个 dtype 组合），每个组合都报：

```
ACLNN error 361001 at .../tests/aclnn/test_aclnn.cpp:117: aclnnEuclideanNorm(workspace, workspaceSize, executor, stream)
```

隔离诊断确认了触发条件：把 TilingFunc 换成"`SetBlockDim(1)` + 写 TilingData、不设 tilingKey"的最小版本时，冒烟 8/8 通过；同一最小版本只加一行 `context->SetTilingKey(1)`，立即变成 361001——即**非零 tilingKey 是唯一触发变量**。

## 适用场景

- 工程形态：算子仓自定义算子（AscendC kernel + host TilingFunc），渐进式开发——TilingFunc 先行实现 tilingKey 编码，kernel 仍为桩或只支持部分分支。
- API 路径：host 侧 `gert::TilingContext::SetTilingKey`；kernel 侧 `REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == N", Struct)` / `TILING_KEY_IS(k)`（kernel_utils_macros.h，`TILING_KEY_VAR` 即 `g_tilingKey`）；aclnn 两段式接口 `aclnnEuclideanNormGetWorkspaceSize` + `aclnnEuclideanNorm`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何使用多 tilingKey 分派的 aclnn/AscendC 算子，与具体算子无关。
- 路线优先级：recommended_fastpath（逐 key 注册是正确性必需；"只下发 key=0"是绕开而非解决）。
- 明确不适用 / 未验证场景：kernel 已按 key 各自实现并注册的真分支（本卡是桩阶段共 struct 的注册写法）；纯 kernel 直调（不经 aclnn 派发）路径未验证。

## 根因

kernel 入口只用 `REGISTER_TILING_DEFAULT(EuclideanNormTilingData)` 注册了默认 key（0）对应的 tiling struct；运行时按 (OpType, tilingKey) 做 kernel 派发与 TilingData 布局匹配，TilingFunc 下发了注册表中不存在的 key（1）→ 启动失败报 361001。

机制链（均为本会话实测）：

1. kernel 二进制 json（`$ASCEND_OPP_PATH/vendors/<Vendor>/op_impl/ai_core/tbe/kernel/ascend950/<op>/<Op>_<hash>.json`）里没有任何 tilingKey 匹配字段（`simplifiedKeyMode: 0`，匹配串只含 dtype/format/implMode）——tilingKey **不参与二进制选择**，但在 launch 阶段参与 tiling struct 匹配。
2. `REGISTER_TILING_DEFAULT(Struct)` 展开为 `ENABLE_FEATURE_FOR_TILING(default, Struct)`，只把 struct 布局登记到默认 key；`REGISTER_TILING_FOR_TILINGKEY(expr, Struct)` 才按字符串表达式登记到指定 key（CANN 9.1.0 `x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`）。
3. 隔离 A/B：桩 TilingFunc + 写 TilingData（不 SetTilingKey）→ 8/8 通过；加 `SetTilingKey(1)` → 361001。二者唯一差异就是 key，证明未注册 key 直接导致启动失败。

## 解决配方

修复前（op_kernel/euclidean_norm_apt.cpp，只注册默认 key）：

```cpp
extern "C" __global__ __aicore__ void euclidean_norm(
    GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();

    REGISTER_TILING_DEFAULT(EuclideanNormTilingData);

    GET_TILING_DATA(tilingData, tiling);

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

修复后（对 TilingFunc 会下发的全部 5 个合法 key 逐 key 注册，桩阶段共用同一 struct）：

```cpp
extern "C" __global__ __aicore__ void euclidean_norm(
    GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();

    // Multi-key registration: the public TilingFunc emits the full tilingKey
    // encoding {0,1,2,4,5} so the public TilingUT can exercise key+合轴+ShouldUseGroup.
    // The runtime uses (OpType, tilingKey) for kernel dispatch + TilingData
    // layout — a key produced by TilingFunc but never registered here makes
    // the launch fail (361001). While the kernel is still a stub, every
    // dispatch arm runs the same PipeBarrier; the per-key
    // REGISTER_TILING_FOR_TILINGKEY just declares the shared struct layout
    // so the runtime accepts the launch for any of the 5 keys.
    REGISTER_TILING_DEFAULT(EuclideanNormTilingData);
    if (TILING_KEY_IS(0)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 0", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;
    } else if (TILING_KEY_IS(1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;
    } else if (TILING_KEY_IS(2)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;
    } else if (TILING_KEY_IS(4)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;
    } else if (TILING_KEY_IS(5)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

官方参照写法（CANN 9.1.0 内置算子 `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/avg_pool/avg_pool_apt.cpp`）：`REGISTER_TILING_DEFAULT(AvgPoolTilingData); if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_NCHW)) { REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300002", AvgPoolNCHWSmallKernelTilingData); ... }`——条件分支内逐 key 注册 + `GET_TILING_DATA_WITH_STRUCT` 取数。

判别 / 定位方法（本会话走通的链路，可复用）：

1. 用最小 TilingFunc 做 A/B：`SetBlockDim(1)` + 写 TilingData、不设 key → 通过；加 `SetTilingKey(N)` → 361001，即可锁定"key 未注册"。
2. 查 kernel 二进制 json 确认 tilingKey 不参与二进制选择（排除"选错 binary"方向）。
3. 在 CANN 内置 ascendc 算子源码里 grep `REGISTER_TILING_FOR_TILINGKEY` 取官方写法照抄。

## 设计决策

- 为什么逐 key 注册而不是让 TilingFunc 回退 key=0：公共 TilingUT（19 个用例）断言 tilingKey 的精确编码值（合轴/空张量/分支判定都体现在 key 里），回退 key 会让 UT 失去对编码链路的覆盖；逐 key 注册让桩 kernel 可以承载任何合法 key。
- 为什么 5 个 key 共用一个 struct：该算子的 TilingData 设计就是全分支共用一个汇总结构；每个 key 的 `REGISTER_TILING_FOR_TILINGKEY` 只声明布局，真分支实现落地后可各自换 struct。
- 顺带把桩阶段 `usedCoreNum` 从 `max(coreNum,1)`（=56 满核）降为 1：UT 只断言 `usedCoreNum>=1`，对 UT 等效；桩 kernel 无多核承载能力，满核 blockDim 属于无意义风险（此降级为保守选择，未单独 A/B 验证满核是否致崩）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0；关键头文件 `$ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`（`REGISTER_TILING_DEFAULT`/`REGISTER_TILING_FOR_TILINGKEY`/`TILING_KEY_IS`/`TILING_KEY_VAR` 定义）。
- 相关文件：`op_kernel/euclidean_norm_apt.cpp`（注册处）、`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（SetTilingKey 来源）、`tests/aclnn/run.sh` + `test_aclnn.cpp`（复现入口）。
- 产物核对：`$ASCEND_OPP_PATH/vendors/<Vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/<op>.json` 的 `binList`（dtype/format 匹配串，无 tilingKey 字段）。

## 预防

- TilingFunc 一旦引入非零 tilingKey 编码，同一变更必须同步在 kernel 入口为**每一个会下发的 key** 补 `REGISTER_TILING_FOR_TILINGKEY`，否则 aclnn 冒烟必挂 361001。
- 注册宏第一个参数必须是**字符串匹配表达式**（`"TILING_KEY_VAR == N"`），不是类型名或整数。
- 渐进式开发（tiling 先行、kernel 后行）的算子，建议桩 kernel 一开始就按设计的完整 key 集合注册。

## 验证信号

以下信号来自该算子公共 TilingFunc 落地后的 aclnn 冒烟回归修复会话：先用最小 TilingFunc 隔离锁定触发变量，再补齐 5 key 注册后逐级验证。

- 修复前：桩 TilingFunc + 写 TilingData + `SetTilingKey(1)` → 8 个 dtype 组合逐条报 `ACLNN error 361001 ... aclnnEuclideanNorm(workspace, workspaceSize, executor, stream)`；去掉 SetTilingKey 的同版本 8/8 PASSED（来源平台：950；证据等级：tool_observed）。
- 首次修复尝试被编译器打回：`REGISTER_TILING_FOR_TILINGKEY` 表达式未用字符串 → op 编译失败 `[errCode:EB0500] [message:tiling struct match expression is wrong. lines: auto __enable_custom_tiling EuclideanNormTilingData = 1;]`（来源平台：950；证据等级：tool_observed）。
- 修复后：5 key 注册 + `SetTilingKey(1)` + 最小 TilingData 写入 → 冒烟 8/8 PASSED（来源平台：950；证据等级：tool_observed）。
- 修复后（终态门禁）：`./build.sh` 退出 0、无 "do not registe tiling struct" 告警；aclnn 冒烟 `run.sh` 退出 0、8/8 PASSED；公共 TilingUT 19/19 PASSED；冒烟与 UT 串行互跑均通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡写法是**桩阶段共 struct** 的注册；真分支 kernel 各 key 有独立 struct/实现时，`GET_TILING_DATA_WITH_STRUCT` 要换各自类型，注册模式本身不变。
- key 集合 {0,1,2,4,5} 来自该算子的编码设计；其他算子按自己的 `(isGroup<<2)|(isEmpty<<1)|isTailR` 类似编码取全集。遗漏任何一个会下发的 key 都会 361001。
- 361001 的官方错误码语义（错误码表中 "kernel not found / tiling key mismatch" 的准确措辞）未逐字核对，本卡以实测报错行为为准。
- 只验证了 aclnn 派发路径；GE 图执行路径下未注册 key 的报错形态未验证（未验证，推断同机制）。
- 与"tilingKey 与 funcEntry/SEL 位置映射失配"是 361001 的另一注册层根因（TPL 模板 SEL 占位），判别时先核对用的是哪种注册机制。

**未决项**：

- 运行时对未注册 tilingKey 报 361001 的确切判定代码路径（哪一层做 struct 匹配）未追到源码级；不影响修复有效性。
- 桩阶段 `usedCoreNum=1` 降级后，未单独验证"满核 blockDim + 5 key 注册"是否独立致崩（与另外两个根因耦合，拆开验证成本高）；真分支 kernel 落地时自然覆盖。
## 坏实践

- `REGISTER_TILING_FOR_TILINGKEY` 第一个参数写成类型名/裸表达式（如 `REGISTER_TILING_FOR_TILINGKEY(1, Struct)` 或 `(Struct, 1)` 变体）→ op 编译直接 EB0500 "tiling struct match expression is wrong"。
- 只给 TilingFunc 加 SetTilingKey 而不动 kernel 注册，指望"运行时总能找到默认 struct"——实测找不到，报 361001。

## 误诊

- 最初把 aclnn 冒烟的段错误（退出码 139）归因于"stub kernel 收到真实 tilingKey/blockDim 多核执行崩溃"：隔离测试证明 SetTilingKey(1) 单独只引发 361001 **干净报错**（进程不崩），段错误另有根因（TilingFunc host 侧解引用 device 指针）；361001 与 139 是两个独立缺陷。
- 曾怀疑 kernel 二进制按 tilingKey 选择导致选不到 binary：核对 kernel json 后确认 tilingKey 不在二进制匹配串里，排除。
