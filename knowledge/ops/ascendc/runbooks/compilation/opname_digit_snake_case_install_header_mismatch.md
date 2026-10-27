---
type: "Runbook"
title: 算子名数字段致 aclnn 安装头文件名去下划线
description: "CamelCase 算子名含数字段（如 GridSampler3d）时，CANN 构建系统的 snake_case 转换只在大写字母边界插下划线，数字前不插——安装头文件名/内部名为 grid_sampler3d，与源码目录名 grid_sampler_3d 不一致，ST/CMake 按源码名查找报 include directory not found；aclnn 函数符号名不受影响。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# 算子名数字段致 aclnn 安装头文件名去下划线

## 现象

在一个自定义算子（CamelCase 名以数字段结尾，如 `BenchNoHarnessGridSampler3d`，源码目录/文件名按 `bench_no_harness_grid_sampler_3d` 命名，vendor 包 `bench_no_harness_grid_sampler_3d_custom` 已通过 `custom_opp_ubuntu_x86_64.run` 安装到 `${ASCEND_HOME_PATH}/opp/vendors/`）上运行 C++ 原生 ST 测试，`bash run.sh --real` 在 CMake 配置阶段直接失败，报错原文：

```text
CMake Error at CMakeLists.txt:72 (message):
  Custom operator include directory not found (looking for
  aclnn_bench_no_harness_grid_sampler_3d.h).  Please install the operator
  first.
-- Configuring incomplete, errors occurred!
```

此时 `ls ${ASCEND_HOME_PATH}/opp/vendors/bench_no_harness_grid_sampler_3d_custom/op_api/include/` 里实际只有一个文件：

```text
aclnn_bench_no_harness_grid_sampler3d.h      ← 数字 3 前没有下划线
```

而算子源码目录 `op_api/` 下开发者手写的头文件是 `aclnn_bench_no_harness_grid_sampler_3d.h`（有下划线）。同时安装产物里 kernel 二进制目录、`aic-ascend950-ops-info.json` 的 `opFile`/`value` 字段也都是 `bench_no_harness_grid_sampler3d`（无下划线），说明这不是头文件一处笔误，而是构建系统对算子名的统一规范化。

## 适用场景

- 工程形态：registry-invoke 自定义算子（`npu_op_package` 构建 + `custom_opp_*.run` 安装）+ C++ 原生 ST 测试工程（CMake `find_path` 查找安装的头文件、测试源码 `#include` 安装头）。
- API 路径：aclnn L2 自定义算子头文件（安装于 `${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/include/`）；触发条件是算子 CamelCase 名中含数字段（`...Sampler3d`、`...Pool2d`、`...Add2` 等）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——命名规范化由 CANN 构建系统统一执行，与算子类型无关；本机制对任何"CamelCase 名含数字段"的算子成立。
- 路线优先级：recommended_fastpath（消费侧按安装名查找，一处改名即可）。
- 明确不适用/未验证场景：算子名不含数字段（此时安装名=源码名，无此坑）；aarch64 环境、其他 CANN 版本未验证，但机制推断一致（未验证，推断可迁移）。

## 根因

CANN 构建系统从 CamelCase 算子名生成内部名/安装产物名时做 snake_case 转换，转换在"大写字母边界"插下划线；数字段（如 `3d`）前没有大写字母边界，因此 `GridSampler3d → grid_sampler3d`，数字前的下划线被丢掉，而开发者按可读习惯把源码目录命名为 `bench_no_harness_grid_sampler_3d`。自动生成的 aclnn 安装头文件名（`aclnn_bench_no_harness_grid_sampler3d.h`）、kernel 二进制目录（`ascend950/bench_no_harness_grid_sampler3d/`）、ops-info 注册名全部跟随规范化名，与源码侧命名不一致。

注意：**aclnn 函数符号名不受影响**——安装头里声明的是 `aclnnBenchNoHarnessGridSampler3dGetWorkspaceSize` / `aclnnBenchNoHarnessGridSampler3d`，与源码头一致，只有**头文件名**（以及内部路径名）被规范化。

## 解决配方

修复前（ST 工程 `tests/st/CMakeLists.txt`，按源码名拼安装头文件名，找不到）：

```cmake
    # 查找自定义算子头文件目录
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_${OP_NAME}.h
        PATHS
            ${CANN_ROOT}/opp/vendors/${VENDOR_DIR}/op_api/include
            /usr/local/Ascend/opp/vendors/${VENDOR_DIR}/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/${VENDOR_DIR}/op_api/include
    )

    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found (looking for aclnn_${OP_NAME}.h). "
                            "Please install the operator first.")
    endif()
```

（其中 `set(OP_NAME bench_no_harness_grid_sampler_3d)`、`set(VENDOR_DIR ${OP_NAME}_custom)`。）

修复后（显式定义规范化名，按安装名查找）：

```cmake
    # 查找自定义算子头文件目录
    # 注意：CANN 构建系统会规范化算子名（去除数字前的下划线），
    # bench_no_harness_grid_sampler_3d -> bench_no_harness_grid_sampler3d
    set(OP_NAME_NORMALIZED bench_no_harness_grid_sampler3d)
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_${OP_NAME_NORMALIZED}.h
        PATHS
            ${CANN_ROOT}/opp/vendors/${VENDOR_DIR}/op_api/include
            /usr/local/Ascend/opp/vendors/${VENDOR_DIR}/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/${VENDOR_DIR}/op_api/include
    )

    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found (looking for aclnn_${OP_NAME_NORMALIZED}.h). "
                            "Please install the operator first.")
    endif()
```

配套修复测试源码的 include（函数名不变，只改头文件名）：

```cpp
// 修复前
#include "aclnn_bench_no_harness_grid_sampler_3d.h"
// 修复后
#include "aclnn_bench_no_harness_grid_sampler3d.h"
```

判别命令（确认安装侧的真实头文件名与注册名）：

```bash
find ${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/ -name "*.h"
grep "bench_no_harness_grid_sampler" <op>/build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend950/*-ops-info.json
# ops-info 中 "value": "bench_no_harness_grid_sampler3d" 即规范化内部名
```

## 设计决策

- 为什么改消费侧（ST 工程）而不是把源码目录改名对齐：源码侧目录/文件名已在设计文档、CMake、多处 include 中使用，改动面大；安装侧命名由构建系统生成不可配置，消费侧按安装名查找是一处局部修复。若在工程新建期发现，直接让源码目录名避开数字段结尾（或与规范化名一致）是更干净的方案。
- 为什么用显式 `OP_NAME_NORMALIZED` 变量而不是把 `OP_NAME` 直接改掉：`VENDOR_DIR`（`<op>_custom`）仍按源码名命名，vendor 目录名与头文件名遵循两套命名，必须分开表达，避免后续读者误解。
- 当前方案路线档位：recommended_fastpath（无运行时代价，纯查找名修正）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0（`ASCEND_HOME_PATH` 指向安装根），GCC 13.3.0，CMake 3.20.5。
- 相关文件：`operators/<op>/tests/st/CMakeLists.txt`（find_path 修复）；`operators/<op>/tests/st/test_aclnn_<op>.cpp`（include 修复）；安装产物 `${ASCEND_HOME_PATH}/opp/vendors/bench_no_harness_grid_sampler_3d_custom/op_api/include/aclnn_bench_no_harness_grid_sampler3d.h`（自动生成名）；`build/op_kernel/ascendc_kernels/binary/config/ascend950/bench_no_harness_grid_sampler3d.json`（内部名证据）。

## 预防

- 给 CamelCase 名含数字段的算子写 ST/consumer 工程时，先用 `find ${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/ -name "*.h"` 确认安装头文件名，再写 `find_path`/`#include`。
- 经验规则：`FooBar3d → foo_bar3d`（数字段不产生下划线）；只要 CamelCase 名里数字后跟小写字母，安装名与"源码手写 snake_case 名"就会分歧。
- 把"安装产物命名 = 构建系统规范化名，非源码目录名"写进算子工程 README/测试脚手架检查清单。

## 验证信号

以下信号来自该算子 ST 测试汇合验证会话：ST 工程按安装头文件名修复后，重新执行 Real 模式测试。均为工具实测：

- 修复前：CMake 配置失败，报 `Custom operator include directory not found (looking for aclnn_bench_no_harness_grid_sampler_3d.h). Please install the operator first.`，`Configuring incomplete`（来源平台：950；证据等级：tool_observed）。
- 修复前观测：安装目录下仅存在 `aclnn_bench_no_harness_grid_sampler3d.h`；ops-info 与 binary config 中内部名为 `bench_no_harness_grid_sampler3d`（来源平台：950；证据等级：tool_observed）。
- 修复后：CMake 配置成功（`Found custom operator include dir: .../op_api/include`），编译链接通过（来源平台：950；证据等级：tool_observed）。
- 修复后：ST Real 首轮 3 条用例全部通过（首条 MERE=1.104e-07），后续全量回归不再出现该 CMake 报错（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在"CamelCase 名含数字段"时触发；纯字母算子名安装名与源码名一致。
- 规范化行为在 CANN 9.1.0 / x86_64-linux 上实测；其他 CANN 版本或 aarch64 未验证，机制推断一致（未验证，推断可迁移）。
- 本卡只覆盖**头文件/内部路径名**规范化；aclnn 函数符号名不被规范化（修复前后一致），不要误改函数调用。

**未决项**：

- 无（根因已定位到构建系统命名规范化机制层并闭环验证；规范化算法的源码级实现未读，属框架内部）。
## 坏实践

- 按源码目录名硬拼安装头文件名（`aclnn_${OP_NAME}.h`）而不检查安装目录实际产物——数字结尾算子必踩。
- 发现不一致后改源码 `op_api/` 手写头文件名去"对齐"安装名：手写头是构建输入，改名会连带 autogen 适配器、文档、CMake 一串改动，且下一版构建仍可能生成规范化名。

## 误诊

- "算子没装上/装坏了"：报错文案 `Please install the operator first` 具有误导性，实际 vendor 包已安装且完整，只是查找的文件名不对；先 `ls` 安装目录再下结论。

## 后续增量（2026-09-17，并卡合并：op type 派生口径、四种规范化形态与 CMake 缓存陷阱）

#### 并入自 aclnn_header_name_digit_underscore（op type 小写化派生口径、CamelCase 误写形态与 CMake 缓存陷阱）

- **派生口径定形**：自动生成的 aclnn 头文件名由 **op type 名**小写化派生（`BenchNoHarnessConv2d` → `bench_no_harness_conv2d`，`Conv2d` 作为单词整体小写、数字前不插下划线）；算子目录名是另一套 snake_case 约定（`bench_no_harness_conv_2d`），源码 op_api 头文件名是第三处命名——多套命名不保证一致，按目录名或源码名拼接 `aclnn_${OP_NAME}.h` 就会差一个下划线。
- **vendor 目录名本身不规范化**（本例 vendor 目录仍带 `_3d`/`_2d`）——只有 aclnn 头文件名（及 kernel 内部名）被规范化；安装头文件头注明 `this file was generated automaticlly donot change it`，`build/autogen/` 与安装清单（install_manifest）同为规范化名。
- **修复形态（显式 `ACLNN_HEADER` 变量）**：`set(ACLNN_HEADER aclnn_bench_no_harness_conv2d.h)` 后 `find_path(NAMES ${ACLNN_HEADER} ...)`；测试源码同步 `#include "aclnn_bench_no_harness_conv2d.h"`（Mock 模式仍 include 源码目录头，用 `USE_MOCK_ACLNN` 宏隔离）。
- **CamelCase 误写形态（R4）**：`find_path` 的 NAMES 写了 CamelCase 的 `aclnnBenchNoHarnessApplyAdamW.h`（vendors 里实际存在 snake_case 的 `aclnn_bench_no_harness_apply_adam_w.h`）——生成头一律是 snake_case；该形态下 Mock 模式不暴露（include 位于 `#ifndef USE_MOCK_ACLNN` 块内，Mock 编译根本不包含真实头）。
- **CMake 缓存陷阱（R2 配套事实）**：修改 find_path 的 NAMES 后必须清掉 ST 的 build 缓存目录再重配——find_path 的 NOTFOUND 结果会被 CMakeCache 缓存，改完 NAMES 不清缓存会继续失败。
- **kernel 入口名规则同源**：该命名规则与 kernel 入口函数名规则同源一致（`Conv2d`→`conv2d`，入口名不匹配会报 kernel entry not implement，见 [kernel entry 入口名卡](kernel_entry_snake_case_follows_op_type.md)）。
- 坏实践补充（并入）：按直觉写 CamelCase 头名；只跑 Mock 模式就认为 ST 工程与安装产物对齐；CMake 把整个 `opp/vendors/` 根目录塞进 include 路径"总能找到"（多 vendor 并存时易误选同名头）；改完 find_path 的 NAMES 不清 CMake 缓存。
- 预防补充（并入）：Mock 模式编译通过不代表 Real 模式能编（Mock 不 include 真实头文件）——首次接入 Real 模式时专门过一遍头文件/枚举/函数名；op type 中数字/大小写组合更复杂时（如 `Conv2DBackprop`）生成名的拆分点需以安装产物为准（未验证）。

验证信号（并入，Conv2d/AdaptiveAvgPool3d/Conv3dBackpropFilter/DepthwiseConv2d/ApplyAdamW 五个独立算子，Ascend950PR，CANN 9.1.0 与 9.2.0）：

- Conv2d 修复前：CMake fatal `Custom operator include directory not found`（按目录名拼的 `aclnn_bench_no_harness_conv_2d.h` 不存在）；安装产物事实锚点：`opp/vendors/bench_no_harness_conv_2d_custom/op_api/include/aclnn_bench_no_harness_conv2d.h` 存在且为自动生成；修复后 CMake `Found custom operator include dir`，编译推进到后续错误（来源平台：950；证据等级：tool_observed）。
- 池化算子：vendors include 目录 `ls` 实测只有 `aclnn_bench_no_harness_adaptive_avg_pool3d.h`（无 `_3d`），`build/autogen/` 同名；安装头 grep 确认函数名不变（`aclnnBenchNoHarnessAdaptiveAvgPool3dGetWorkspaceSize`）；按安装名 include 的示例编译通过并在 NPU 运行（输入 [1,2,4,4,4] 全 1.0 fp32，输出 [1,2,2,2,2] 16 个元素全 1.000000）（来源平台：950；证据等级：tool_observed）。
- R2（Conv3d 反传滤波，CANN 9.1.0）：修复前 ST CMake 配置报 `Custom operator include directory not found. Please install the operator first.`，安装目录实际头文件为 `aclnn_bench_no_harness_conv3d_backprop_filter.h`（源头文件名 `conv_3d`，安装名 `conv3d`，宏名同步为 `ACLNN_BENCH_NO_HARNESS_CONV3D_BACKPROP_FILTER_H_`）；修复后 `-- Found custom operator include dir`，测试编译链接成功进入 NPU 执行（来源平台：x86 编译主机（host 侧 ST 配置闭环）；证据等级：tool_observed）。
- R3（DepthwiseConv2d，CANN 9.2.0）：修复前 `bash tests/st/run.sh --real --level L0` CMake 阶段报 `Custom operator include directory not found`，实际头文件为 `aclnn_bench_no_harness_depthwise_conv2d.h`（手写名 `conv_2d`，autogen 名 `conv2d`）；修复后全链修复 ST 8/8 通过（6 NPU 实跑 + 2 strided 按设计跳过）（来源平台：950；证据等级：tool_observed）。
- R4（ApplyAdamW，CANN 9.1.0）：修复前 ST Real 模式 CMake 报 `Custom operator include directory not found`，vendors include 目录实际含 `aclnn_bench_no_harness_apply_adam_w.h` 而 NAMES 写了 CamelCase；修复后 CMake 输出 `-- Found custom operator include dir`，测试程序编译链接成功进入执行（来源平台：950；证据等级：tool_observed）。

失效边界补充（并入）：

- 规范化实测样本含四种形态：`conv_2d`→`conv2d`、`conv_3d`→`conv3d`、`pool_3d`→`pool3d`（数字段跟随前缀字母不切开）与 `BenchNoHarnessApplyAdamW`→`bench_no_harness_apply_adam_w`（纯 CamelCase→snake_case）；名中段 `_2d` 是否同样处理未验证（推断一致）。
- 二进制 JSON/kernel 名不受此规范化影响（vendor 目录、binary JSON 跟随源码名）。

#### 并入自 aclnn_install_header_name_digit_no_underscore（AdaptiveAvgPool3d 联调实例与安装头内容差异）

- **安装头内容与源头文件的差异（旁证其为再生成产物）**：安装头 include `aclnn/acl_meta.h` 而非源头的 `aclnn/aclnn_base.h`，声明 `aclnnBenchNoHarnessAdaptiveAvgPool3dGetWorkspaceSize` / `aclnnBenchNoHarnessAdaptiveAvgPool3d` 两个 C 接口（实测样本 1185 字节，含 `this file was generated automaticlly donot change it` 标记）——即安装头不是源头文件的拷贝，而是按 op type 规则重新生成的。
- 修复形态等价（并入印证）：`set(ACLNN_HEADER_NAME "aclnn_bench_no_harness_adaptive_avg_pool3d.h")` + 测试源码 `#include "aclnn_bench_no_harness_adaptive_avg_pool3d.h"`，Mock 模式仍 include 源码目录头（`USE_MOCK_ACLNN` 隔离）。
- 设计决策（并入）：选择"适配安装名"而非"改源文件名"——源文件名贯穿 op 工程全链路（CMake、proto、autogen），改动面大；安装头名是 CANN 生成规则的单向产物，测试侧适配是最小改动。

验证信号（并入，BenchNoHarnessAdaptiveAvgPool3d 联调，Ascend950PR + CANN 9.1.0）：

- 修复前：`CMake Error ... Custom operator include directory not found. Please install the operator first.`（来源平台：950；证据等级：tool_observed）。
- 安装目录实测：`op_api/include/` 下仅 `aclnn_bench_no_harness_adaptive_avg_pool3d.h`（自动生成，含 `aclnn/acl_meta.h` include 与生成标记）（来源平台：950；证据等级：tool_observed）。
- 修复后：CMake `-- Found custom operator include dir`、`-- Configuring done`，测试程序编译链接成功并执行 37 条用例（来源平台：950；证据等级：tool_observed）。