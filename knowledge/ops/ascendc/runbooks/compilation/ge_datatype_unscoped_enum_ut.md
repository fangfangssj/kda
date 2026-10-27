---
type: "Runbook"
title: "ge::DataType 为 unscoped 枚举勿手写副本"
description: 自建 gtest UT 在 namespace ge 里重定义 enum class DataType 与 CANN graph/types.h 的 unscoped enum 冲突编译失败，且双精度枚举名是 DT_DOUBLE 而非 DT_FLOAT64；删除本地副本直接复用真实枚举，UT CMake 补 x86_64-linux include 路径。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# ge::DataType 为 unscoped 枚举勿手写副本

## 现象

为一个自定义算子（逐元素一元数学算子）搭建 host 侧 Tiling 单元测试（gtest，编译进 `./build.sh <package>` 的 `tests/ut` 子目标）。为了让 UT 不直接依赖 CANN 安装目录，测试文件里在 `namespace ge` 下手写了一份"本地副本"枚举，并在参数化用例中用 `ge::DataType::DT_FLOAT64` 表示双精度（不支持 dtype）分支。

但被测实现头 `op_host/bessel_i1e_tiling_host.h` 本身 `#include "graph/types.h"`（解析到 `$ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h`，CANN 9.0.0-beta.2），其中已经定义了非 scoped 的 `enum DataType {...}`。同一翻译单元里两份定义相遇，gcc 13.3.0 编译报错、整个算子包构建失败（`make Error 2`，EXIT_CODE=2）：

```text
tests/ut/test_bessel_i1e_tiling.cpp:21:23: error: scoped/unscoped mismatch in enum 'enum ge::DataType'
   21 | enum class DataType : int {
      |                       ^~~~
In file included from .../op_host/bessel_i1e_tiling_host.h:10,
                 from .../tests/ut/test_bessel_i1e_tiling.cpp:17:
<CANN安装根>/x86_64-linux/include/graph/types.h:81:6: note: previous definition here
   81 | enum DataType {
   |      ^~~~
tests/ut/test_bessel_i1e_tiling.cpp:149:73: error: 'DT_FLOAT64' is not a member of 'ge::DataType'
  149 |     DtypeTilingKeyCase{"DT_FLOAT64" " → -1(unsupported)", ge::DataType::DT_FLOAT64, -1},
  |                                                                         ^~~~~~~~~~
```

## 适用场景

- 工程形态：算子仓（ops-registry-invoke 类工程）host 侧 TilingUT / 任何要编译 `op_host` 源码的 gtest 单测；UT 与被测 host 源码同翻译单元编译。
- API 路径：`graph/types.h` 的 `ge::DataType`（`ge::DT_FLOAT`/`ge::DT_FLOAT16`/`ge::DT_BF16`/`ge::DT_DOUBLE`/`ge::DT_INT32`）；UT CMake 需把 `$ENV{ASCEND_HOME_PATH}/x86_64-linux/include` 加入 include 路径才能解析该头。
- 验证过的平台与 CANN 版本：ascend950 构建目标包内 host 侧 UT（x86_64 编译执行），CANN 9.0.0-beta.2。
- 适用粒度：platform_wide——所有包含 host 侧代码（tiling/host cpp）单测的算子工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不 include `graph/types.h`（或等价 ge 头）的纯 kernel 侧测试；其他 CANN 版本中枚举形态未验证（枚举成员名随版本可能变化，用前先 grep 确认）。

## 根因

CANN `graph/types.h` 中 `ge::DataType` 是 **unscoped enum**（`enum DataType { DT_FLOAT = ::C_DT_FLOAT, ... }`，枚举值直接暴露在 `ge` 命名空间），且**双精度的枚举成员名是 `DT_DOUBLE`（= `::C_DT_DOUBLE` = 11），不存在 `DT_FLOAT64`**。在 UT 里以 `enum class DataType : int` 重新定义同名类型构成 scoped/unscoped 重定义冲突；同时手写副本里自造的 `DT_FLOAT64` 在真实头文件中没有对应成员。两处错误都源于"用记忆里的枚举形态手写副本"而不是复用真实头文件。

## 解决配方

删除 UT 内的整个 `namespace ge { enum class DataType ... }` 副本，直接使用被测头文件已经引入的真实 `ge::DataType`；枚举值改为非 scoped 用法 `ge::DT_FLOAT` 等；双精度不支持分支改用 `ge::DT_DOUBLE`。

修复前（UT 文件头部，手写副本 + 后续全部按 scoped 方式使用）：

```cpp
#include "bessel_i1e_tiling.h"
#include "bessel_i1e_tiling_host.h"

// ge::DataType 定义
namespace ge {
enum class DataType : int {
    DT_FLOAT   = 0,
    DT_FLOAT16 = 1,
    DT_INT32   = 3,
    DT_BF16    = 27,
    DT_FLOAT64 = 12,   // double
};
} // namespace ge

namespace {
```

```cpp
int32_t OracleChooseTilingKey(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT:   return 0;
        case ge::DataType::DT_FLOAT16: return 1;
        case ge::DataType::DT_BF16:    return 2;
        default:                       return -1;
    }
}
```

```cpp
INSTANTIATE_TEST_SUITE_P(Sec55, DtypeTilingKeyTest, testing::Values(
    DtypeTilingKeyCase{"DT_FLOAT"   " → 0", ge::DataType::DT_FLOAT,   0},
    DtypeTilingKeyCase{"DT_FLOAT16" " → 1", ge::DataType::DT_FLOAT16, 1},
    DtypeTilingKeyCase{"DT_BF16"    " → 2", ge::DataType::DT_BF16,    2},
    DtypeTilingKeyCase{"DT_FLOAT64" " → -1(unsupported)", ge::DataType::DT_FLOAT64, -1},
    DtypeTilingKeyCase{"DT_INT32"   " → -1(unsupported)", ge::DataType::DT_INT32,   -1}
));
```

修复后（删副本、注释说明真实枚举来源、全部改非 scoped 用法、`DT_FLOAT64`→`DT_DOUBLE`）：

```cpp
#include "bessel_i1e_tiling.h"
#include "bessel_i1e_tiling_host.h"

// graph/types.h 已通过 bessel_i1e_tiling_host.h 引入，提供 ge::DataType（unscoped enum）
// ge::DT_FLOAT=0, DT_FLOAT16=1, DT_INT32=3, DT_BF16=27, DT_DOUBLE=11

namespace {
```

```cpp
int32_t OracleChooseTilingKey(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:   return 0;
        case ge::DT_FLOAT16: return 1;
        case ge::DT_BF16:    return 2;
        default:             return -1;
    }
}
```

```cpp
INSTANTIATE_TEST_SUITE_P(Sec55, DtypeTilingKeyTest, testing::Values(
    DtypeTilingKeyCase{"DT_FLOAT   → 0",               ge::DT_FLOAT,   0},
    DtypeTilingKeyCase{"DT_FLOAT16 → 1",               ge::DT_FLOAT16, 1},
    DtypeTilingKeyCase{"DT_BF16    → 2",               ge::DT_BF16,    2},
    DtypeTilingKeyCase{"DT_DOUBLE  → -1(unsupported)", ge::DT_DOUBLE, -1},
    DtypeTilingKeyCase{"DT_INT32   → -1(unsupported)", ge::DT_INT32,  -1}
));
```

配套的 UT CMake include 路径（`graph/types.h` 的解析依赖它）：

```cmake
target_include_directories(bessel_i1e_tiling_ut PRIVATE
    ${PKG_ROOT}/op_host
    ${PKG_ROOT}/op_kernel
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
)
```

## 设计决策

- 选择"删副本、复用真实头"而不是"改个命名空间自己定义一份"：被测函数签名就是 `int32_t ChooseTilingKey(ge::DataType dtype)`，UT 必须用同一个类型才能直接调用被测函数；自定义副本即使改名也还要与真实枚举做映射，等于把坑搬到别处。
- 不支持 dtype 用 `ge::DT_DOUBLE`（真实存在的成员）而不是自造成员；`DT_BF16=27`、`DT_DOUBLE=11` 这类值以 grep 真实头文件为准（`grep -n 'DT_FLOAT\|DT_BF16\|DT_DOUBLE' $ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h`）。
- 当前方案属 recommended_fastpath：一次删干净，无需维护副本与真实头的同步。

## 系统信息

- 工具链：gcc 13.3.0（GNU），cmake ≥3.16，gtest（`/usr/include/gtest/gtest.h` + `libgtest`/`libgtest_main`/`pthread`）。
- CANN：CANN 9.0.0-beta.2；`graph/types.h` 位于 `$ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h`。
- 目标平台：ascend950（包构建产物 `aic-ascend950-ops-info.json`、`[ascend950] Generating ...`）。
- 相关文件：`operators/<pkg>/tests/ut/test_bessel_i1e_tiling.cpp`（本次修复对象）、`operators/<pkg>/op_host/bessel_i1e_tiling_host.h`（引入 graph/types.h 的被测头）、`operators/<pkg>/tests/ut/CMakeLists.txt`（include 路径）。

## 验证信号

本卡验证来自该算子一次开发会话中 TilingUT 的搭建与修复：UT 编译失败后按上述配方修复，同日同工程复测。

- 修复后同一命令 `./build.sh <package>` 构建退出码 0（`EXIT_CODE=0`，custom_opp_ubuntu_x86_64.run 正常生成）（来源平台：950 构建目标 host 侧编译；证据等级：tool_observed）。
- 修复后 UT 二进制可编译、可执行并进入 stub 基线：修复当轮 21 用例 3 过 18 失败，随后移除 3 个常量校验类用例（与被测函数无关、stub 阶段就会通过）后 18 用例 0 过 18 失败，符合"tiling 未实现期望通过数=0"的基线（来源平台：950 构建目标 host 侧 UT；证据等级：tool_observed）。
- 后续该 UT 在公共 TilingFunc 实现后 18/18 全部通过，证明修复后的用例集真正驱动了实现（来源平台：950 构建目标 host 侧 UT；证据等级：tool_observed）。

## 失效边界

- 本卡结论绑定 `graph/types.h` 的 unscoped enum 形态与 `DT_DOUBLE` 命名（CANN 9.0.0-beta.2 实测）；若后续 CANN 版本把 `ge::DataType` 改为 scoped enum 或新增 `DT_FLOAT64` 别名，症状与修法会变化——使用前先 grep 真实头确认（未验证，推断可迁移）。
- UT 若通过预编译头/独立 stub 头隔离了 `graph/types.h`，则不会触发本冲突，本卡不适用。

**未决项**：

- 无（本根因已闭合到编译通过 + UT 基线恢复）。
## 坏实践

- 在 UT/测试脚手架里按记忆手写 CANN 系统头的类型副本（枚举成员名、成员值都容易错：`DT_FLOAT64` 不存在、`DT_BF16=27` 而非想当然的 2/3）。
- 用 `enum class` 复刻一个本应是 unscoped 的系统枚举——即使成员名全对，scoped/unscoped 冲突也会直接编译失败。

## 误诊

- 初看容易把 `scoped/unscoped mismatch` 当成"头文件版本冲突/重复 include"去查 include 链；实际就是 UT 自己定义了一份同名 scoped 枚举，与 include 进来的真实 unscoped 枚举冲突（编译器 note 已直接指出 previous definition 位置）。
- `DT_FLOAT64` 报错时容易怀疑 gtest 参数化宏的问题；实际是成员名不存在，真实名是 `DT_DOUBLE`（该分析先由推断得出，随后 grep 真实头文件证实：`Line 92: DT_DOUBLE = ::C_DT_DOUBLE`，grep 证实部分为 tool_observed）。
