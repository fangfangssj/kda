---
type: "Runbook"
title: op_host util 头不互含，未声明报错补 include
description: "host tiling 用 platform_ascendc::PlatformAscendC 报 'platform_ascendc' has not been declared、用 Ops::Base::CeilDiv 报 'CeilDiv' has not been declared——op_common 的 util 头（platform_util.h/math_util.h）互不传递包含，须按符号逐个补 include；\"未声明\"报错查源文件 include 清单而非 CMake 路径。"
tags: [task.debugging, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# op_host util 头不互含，未声明报错补 include

## 现象

两个独立算子工程的 op_host tiling 首次构建撞到同一类"符号未声明"错误，且都容易被误判为构建配置问题。

**形态一（platform_ascendc）**：Mish 激活算子（Ascend950PR/arch35，CANN 9.1.0）的 host tiling `op_host/arch35/*_tiling.cpp` 用 `platform_ascendc::PlatformAscendC(platformInfoPtr)` 获取 AIV 核数与 UB 大小，首版 include 清单抄了模板的 `register/op_def_registry.h`、`op_common/log/log.h`、`op_common/op_host/util/math_util.h` 三个头，漏掉 `platform_util.h`。首次 `bash build.sh --soc=ascend950` 的 host autogen 编译报：

```
op_host/arch35/bench_no_harness_mish_tiling.cpp:52:28: error: 'platform_ascendc' has not been declared
   52 |     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
       |                            ^~~~~~~~~~~~~~~~
op_host/arch35/bench_no_harness_mish_tiling.cpp:55:36: error: 'platform_ascendc' has not been declared
   55 |     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
       |                                    ^~~~~~~~~~~~~~~~
```

**形态二（Ops::Base::CeilDiv）**：5D 自适应平均池化算子的 tiling 顶部只包含 `register/op_def_registry.h`、`op_common/log/log.h`、`op_common/op_host/util/platform_util.h`，并在 `namespace optiling` 里写了 `using Ops::Base::CeilDiv;`。`bash build.sh --soc=ascend950 -j8` 在 autogen 的 `ascend_all_ops` 阶段报：

```text
op_host/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling.cpp:30:18:
error: 'CeilDiv' has not been declared in 'Ops::Base'

     30 |         using Ops::Base::CeilDiv;
        |                  ^~~~~~~

op_host/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling.cpp:109:28:
error: 'CeilDiv' was not declared in this scope

    109 |     uint32_t coreTaskNum = CeilDiv(totalNC, blockDim);
        |                            ^~~~~~~~

  RuntimeError: compile ascend all ops failed
```

两种形态的报错字面都是"未声明"，没有"找不到头文件"——说明不是 include 搜索路径问题，而是源文件从未引入定义该符号的头。

## 适用场景

- 工程形态：算子仓工程 op_host tiling（IMPL_OP_OPTILING）开发 + host autogen 编译（npu_op_code_gen / cust_optiling / ascend_all_ops）。
- API 路径：`platform_ascendc::PlatformAscendC`、`GetCoreNumAiv()`、`GetCoreMemSize(CoreMemType::UB, …)`（入口头 `op_common/op_host/util/platform_util.h`）；`Ops::Base::CeilDiv / FloorDiv / FloorAlign / CeilAlign`（入口头 `op_common/op_host/util/math_util.h`）；日志 `op_common/log/log.h`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致；两个独立工程实测）。
- 适用粒度：platform_wide（所有 CANN 9.x 算子仓 host tiling）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 侧（__aicore__）代码不走 op_common 头；其他 CANN 大版本的头文件布局未验证。

## 根因

op_common 的三个 util 头**互不传递包含**，各管各的符号族：

- `platform_ascendc` 命名空间与 `PlatformAscendC` 类定义在 `$ASCEND_HOME_PATH/x86_64-linux/asc/include/utils/tiling/platform/platform_ascendc.h`；host 侧获取它的规范入口是 `op_common/op_host/util/platform_util.h`——该头第 21 行 `#include "tiling/platform/platform_ascendc.h"`。
- `Ops::Base::CeilDiv` 等数学工具函数定义在 `op_common/op_host/util/math_util.h`，不在 `platform_util.h` 或 `log.h` 中。
- 日志在 `op_common/log/log.h`。

只 include 其中一个就引用全部工具符号，即触发"未声明"。

判别特征：报错字面决定排查方向——`'X' has not been declared` = 源文件没引入定义（查 include 清单）；`X file not found` = include 搜索路径不对（查 CMake）。形态二的报错同时出现在 using 声明行（30 行）和调用行（109 行），且都是"未声明"而非"重定义/歧义"——说明符号从头到尾不可见，是缺 include，而不是命名空间或链接问题。

## 解决配方

**形态一**（platform_util.h，修复前后 include 块 verbatim）：

```cpp
// 修复前（缺 platform_util.h）
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "bench_no_harness_mish_tiling.h"
#include "../../op_kernel/arch35/bench_no_harness_mish_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_mish_tiling_key.h"
```

```cpp
// 修复后（只增加一行）
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "bench_no_harness_mish_tiling.h"
#include "../../op_kernel/arch35/bench_no_harness_mish_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_mish_tiling_key.h"
```

受影响的调用代码（修复前后不变，`GetPlatformInfo` 完整函数，供对照平台 API 用法）：

```cpp
// 获取平台信息（ubSize, coreNum）
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

**形态二**（math_util.h，修复前后 include 块 verbatim）：

```cpp
// 修复前（缺 math_util.h）
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling_key.h"
```

```cpp
// 修复后（补入 math_util.h）
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling_key.h"
```

使用现场（TilingFunc 内多核切分段，verbatim，含 CeilDiv 的两个调用点）：

```cpp
    // 5. 多核切分：沿 N*C 维度
    uint32_t totalNC = N * C;
    uint32_t blockDim = static_cast<uint32_t>(coreNum);
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    uint32_t coreTaskNum = CeilDiv(totalNC, blockDim);
    if (coreTaskNum == 0U) {
        coreTaskNum = 1U;
    }
    uint32_t usedCoreNum = CeilDiv(totalNC, coreTaskNum);
    if (usedCoreNum == 0U) {
        usedCoreNum = 1U;
    }
```

定位头文件的可复用探针（形态二实测）：

```bash
find $ASCEND_HOME_PATH/x86_64-linux -name "math_util.h" -path "*/op_common/*"
# $ASCEND_HOME_PATH/x86_64-linux/include/op_common/op_host/util/math_util.h

grep -n "CeilDiv\|CeilAlign\|FloorDiv\|FloorAlign" \
  $ASCEND_HOME_PATH/x86_64-linux/include/op_common/op_host/util/math_util.h
# 25:namespace Ops {
# 26:namespace Base {
# 31:auto FloorDiv(T x, T y) -> ...
# 40:auto FloorAlign(T x, T align) -> ...
```

## 设计决策

- 形态一最终选择补 `#include "op_common/op_host/util/platform_util.h"` 而不是直接 `#include "tiling/platform/platform_ascendc.h"`：platform_util.h 是模板示例与 CANN 共同约定的 host 入口，它还同时提供 `fe::PlatFormInfos` 相关的取用封装；直连底层头在本例可用但偏离惯例。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（arch35），CANN 9.1.0（x86_64-linux 安装布局，头文件位于 `$ASCEND_HOME_PATH/x86_64-linux/include/op_common/...`）。
- 相关文件：`op_host/arch35/*_tiling.cpp`（两处修复点）；`asc/include/utils/tiling/platform/platform_ascendc.h`（platform_ascendc 定义）；`op_common/op_host/util/platform_util.h`、`op_common/op_host/util/math_util.h`、`op_common/log/log.h`。
- 另注（形态一会话）：`asc/include/tiling/platform/platform_ascendc.h` 与 `asc/include/utils/tiling/platform/platform_ascendc.h` 两处副本同存，模板原有的 `asc/include/tiling` include 根也能解析 platform_util.h 的内部 include；补 `asc/include/utils` 到 CMake include 路径属稳妥项而非必要修复。

## 预防

- 写 tiling.cpp 时把 include 清单与所用符号对照：用到 `platform_ascendc::` / `fe::PlatFormInfos` ⟹ 必含 `op_common/op_host/util/platform_util.h`；用到 `CeilDiv/CeilAlign/FloorDiv/FloorAlign` ⟹ 必含 `op_common/op_host/util/math_util.h`；日志 ⟹ `op_common/log/log.h`。三者互不传递包含。
- 区分两类报错再动手：`'X' has not been declared` = 查源文件 include 清单；`X file not found` = 查 CMake。先改错的一侧会白跑一轮几分钟的编译。
- 报错行同时覆盖 using 声明与调用点时，优先怀疑缺 include 而非命名空间写法。

## 验证信号

以下信号来自两个独立算子工程的编译联调会话。

形态一（platform_util.h）：

- 修复前：tiling.cpp:52/55 两处 `'platform_ascendc' has not been declared`，连续三轮编译复现（含仅补 CMake include 路径之后的一轮）（来源平台：950；证据等级：tool_observed）。
- 定义位置探针：`grep -rln "namespace platform_ascendc"` 命中 `asc/include/utils/tiling/platform/platform_ascendc.h`；platform_util.h 第 21 行 include 该头（来源平台：950；证据等级：tool_observed）。
- 修复后：host 侧 `cust_optiling` 编译通过（`[ 75%] Building CXX object op_host/CMakeFiles/cust_optiling.dir/arch35/*_tiling.cpp.o` 后 `Built target`），platform_ascendc 报错消失；全部修复完成后 `bash build.sh --soc=ascend950` exit 0（来源平台：950；证据等级：tool_observed）。

形态二（math_util.h）：

- 修复前：`ascend_all_ops` 阶段报 `'CeilDiv' has not been declared in 'Ops::Base'`（using 声明行 30）+ `'CeilDiv' was not declared in this scope`（调用行 109），`RuntimeError: compile ascend all ops failed`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一条 `build.sh --soc=ascend950` 的下一轮输出中 CeilDiv 报错消失，`[81%] Built target cust_optiling`，报错推进到 op_api 阶段的其他问题；该 tiling 源文件此后再未因 include 问题失败，最终整包构建成功并产出 ascend950 kernel 二进制与 custom_opp_ubuntu_x86_64.run（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 头文件布局以 CANN 9.1.0 为准；其他版本 platform_util.h 的内部 include 路径可能调整（未验证，推断可迁移）。
- 若报错是 "ambiguous" 或 "redefinition" 而非 "not declared"，则不是本卡场景。
- 本卡只闭合"未声明"这一根因；同轮首编的 DT_BFLOAT16、ASCENDC_TPL 报错是独立根因。

**未决项**：

- 无（两形态根因均由"补 include 后编译通过"闭合）。
## 坏实践

- 见"误诊"：报"未声明"却先去改 CMake include 路径，浪费一整轮编译时间。
- include 清单靠记忆补齐而不与模板逐行 diff。
- 凭记忆把 op_common 各 util 头当成一个聚合头使用：只 include 一个就引用全部 `Ops::Base::` 工具。

## 误诊

- 形态一第一轮修复先假设是 CMakeLists 缺 `asc/include/utils` include 根（platform_util.h 的相对 include 解析不了），在 npu_op_code_gen COMPILE_OPTIONS、cust_optiling、cust_op_proto 三处补上该路径后重编——错误原样复现，证明路径不是根因。该误诊的正面价值：确立了"报错字面（未声明 vs 找不到文件）决定排查方向"的判别规则。
