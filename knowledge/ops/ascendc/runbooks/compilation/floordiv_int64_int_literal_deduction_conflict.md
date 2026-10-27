---
type: "Runbook"
title: FloorDiv 混用 int64 与 int 字面量推导冲突
description: CANN op_host math_util.h 的 FloorDiv/CeilDiv/CeilAlign 是单模板参数模板（FloorDiv(T,T)），第二参传 int 字面量会 deduction 冲突报 no matching function；字面量需显式 cast 成同类型。复现增量：CeilAlign(int64_t, 16) 形态独立复现，同族约束确认。
tags: [task.implementation, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# FloorDiv 混用 int64 与 int 字面量推导冲突

## 现象

在 x86_64 主机上为 Ascend950PR（arch35，CANN 9.1.0）开发自定义算子的 op_host tiling 实现（`op_host/arch35/transpose_tiling.cpp`，计算 UB 单次搬运元素数 `ubFactor`）后，执行 `bash build.sh --soc=ascend950 -j8` 编译，op_host 侧报编译错误（同一文件两处）：

```
/workspace/.../op_host/arch35/transpose_tiling.cpp:276:47: error: no matching function for call to 'FloorDiv(int64_t, int)'
  276 |         tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / elemSize, 2), ubBlockSize);
      |                                       ~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/workspace/.../aarch64-linux/include/op_common/op_host/util/math_util.h:31:6: note: candidate: 'template<class T> ... Ops::Base::FloorDiv(T, T)'
/workspace/.../op_host/arch35/transpose_tiling.cpp:276:47: note: deduced conflicting types for parameter 'T' ('long int' and 'int')
```

即 `FloorDiv` 的模板参数 T 同时被推导为 `long int`（第一参 `static_cast<int64_t>(ubSize) / elemSize`）与 `int`（第二参字面量 `2`），推导冲突。当时代码刚从模板 add_example 参考工程改写为通用 tiling 计算，`ubSize` 来自平台信息（`int64_t`），除数写成了裸字面量 `2`。

## 适用场景

- 工程形态：AscendC 自定义算子 op_host tiling 开发（ascendc-registry-invoke 算子仓工程，`build.sh --soc=ascend950` 编译）。
- API 路径：`op_common/op_host/util/math_util.h` 的 `Ops::Base::FloorDiv/CeilDiv/FloorAlign/CeilAlign`（namespace `optiling` 内 `using` 引入后直接调用）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（x86_64 主机交叉编译，产出 ascend950 kernel 二进制与自定义算子包）。
- 适用粒度：platform_wide——所有使用 CANN op_host math_util 模板工具的 tiling/host 代码。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：不适用于自定义的同名工具宏/函数（非 CANN math_util.h）；其他 CANN 版本的头文件签名未逐一核对（该单模板参数签名在 9.1.0 实测）。

## 根因

`Ops::Base::FloorDiv` 是单模板参数的函数模板 `template<class T> auto FloorDiv(T x, T y)`，要求两个实参类型完全一致才能推导 T；`int64_t` 表达式配 `int` 字面量触发 `deduced conflicting types for parameter 'T' ('long int' and 'int')`。它不像 `std::min/max` 那样有 `(T, U)` 重载或常规算术提升，字面量必须与另一参同型。

## 解决配方

把第二参字面量显式 cast 为 `int64_t`（两处：SCH_MODE_COPY 与 SCH_MODE_GENERAL_ND 分支的 `ubFactor` 计算）。

修复前：

```cpp
        tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / elemSize, 2), ubBlockSize);
        context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, static_cast<uint64_t>(SCH_MODE_COPY));

    } else if (simplifyResult.schMode == SCH_MODE_2D) {
```

修复后：

```cpp
        tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / elemSize, static_cast<int64_t>(2)), ubBlockSize);
        context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, static_cast<uint64_t>(SCH_MODE_COPY));

    } else if (simplifyResult.schMode == SCH_MODE_2D) {
```

（SCH_MODE_GENERAL_ND 分支同型修改：`FloorDiv(static_cast<int64_t>(ubSize) / elemSize, static_cast<int64_t>(2))`。）

## 设计决策

- 选择 cast 字面量而不是把 `ubSize` 降为 `int`：`ubSize` 是平台 UB 字节数，必须保持 `int64_t` 语义；cast 字面量是最小且不改变量纲的改法。
- 备选：显式模板实参 `FloorDiv<int64_t>(...)` 同样可行（未采用，风格上不如 cast 直观）。

## 系统信息

- 主机：x86_64 Ubuntu，GCC 13.3.0；CANN 9.1.0（op_host 编译用 `$ASCEND_HOME_PATH/aarch64-linux/include` 树）。
- 目标：ascend950（Ascend950PR，arch35），`build.sh --soc=ascend950 -j8`。
- 相关文件：`op_host/arch35/transpose_tiling.cpp`（TilingFunc 的 ubFactor 计算）；头文件 `op_common/op_host/util/math_util.h`。

## 验证信号

段首来源说明：本卡来自该算子 op_host tiling 开发的编译修复循环，修复前后各一次全量构建对照。

- 修复前：`bash build.sh --soc=ascend950 -j8` 报 `error: no matching function for call to 'FloorDiv(int64_t, int)'`，note 指向 `math_util.h:31` 候选模板（来源平台：950（ascend950 交叉编译）；证据等级：tool_observed）。
- 修复后同命令重编：FloorDiv 相关错误消失，编译推进到 op_api 层的下一处独立错误；两处错误都修复后 `bash build.sh --soc=ascend950 -j8` 全量通过，生成 `BenchNoHarnessTranspose_*.o` kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 自定义算子包（来源平台：950；证据等级：tool_observed）。

增量信号（CeilAlign 形态独立复现，Conv2d 基准算子 host tiling，Ascend950PR/arch35 + CANN 9.1.0）：

- 修复前：`build.sh --soc=ascend950` 报 `error: no matching function for call to 'CeilAlign(int64_t&, int)'`（`int64_t nTile = CeilAlign(COut, 16);`，COut 为 int64_t），note：`candidate: 'template<class T> ... Ops::Base::CeilAlign(T, T)' (math_util.h:77)` + `deduced conflicting types for parameter 'T' ('long int' and 'int')`（来源平台：950；证据等级：tool_observed）。
- 修复后（`CeilAlign(COut, int64_t(16))`）：`--make_clean` 全量重编后该报错不再出现，编译推进到下一处独立错误（opdev 头文件 include 路径问题），证明本处修复生效；同段 `CeilDiv(CIn, c0)`/`CeilDiv(COut, nTile)` 因两实参恰好同为 int64_t 未触发——只有带 int 字面量的那一处失败（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖「字面量与 int64 表达式混用」这一触发形态；`FloorAlign/CeilDiv/CeilAlign` 同族模板有同样的单参数约束（同文件内 `FloorAlign(int64_t, int64_t)` 两参同型故未触发；复现增量后 CeilAlign(int64_t, int) 形态已实测触发）。
- 若后续 CANN 版本为这些工具增加 `(T, U)` 重载，本卡失效（以头文件实际签名为准）。
- 报错文本是 "no matching function"，容易被当成"头文件没包含/函数不存在"去查 include；候选 note 已给出真实原因（类型推导冲突），先读 note 再动手（复现增量误诊提示）。

**未决项**：

- 无（该错误一次性闭环；无未追根因项）。
## 坏实践

- 在 tiling 计算里对 `int64_t` 平台参数直接写裸整型字面量做除法/对齐参数——每次都会在同一位置被模板推导拒绝。
- 用 `(int)ubSize` 强转降位绕过——UB 字节数与元素数在超大 shape 下会溢出 int。

## 预防

写 op_host tiling 时遵守：凡与 `int64_t` 变量一起进入 `FloorDiv/CeilDiv/FloorAlign/CeilAlign` 的字面量，一律 `static_cast<int64_t>(...)`；或统一用 `int64_t` 局部变量承接中间结果。
