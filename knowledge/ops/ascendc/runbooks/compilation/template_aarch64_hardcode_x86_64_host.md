---
type: "Runbook"
title: x86_64 主机模板 aarch64 硬编码两处失效
description: "算子工程模板默认 aarch64 服务器，把 CANN 安装布局架构段（aarch64-linux/）与 CPack 包名后缀（custom_opp_ubuntu_aarch64.run）两处硬编码；x86_64 主机上前者致 op_api 找不到 opdev/op_executor.h，后者致打包成功后误报 Package not found。改为按主机架构自适应。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# x86_64 主机模板 aarch64 硬编码两处失效

## 现象

在 x86_64 开发主机（CANN 9.1.0 安装布局为 `x86_64-linux/include/...`，安装根下**没有** `aarch64-linux` 目录）上为 ascend950 编译自定义算子包（工程由 registry-invoke/add_example 类模板生成，`bash build.sh --soc=ascend950 -j8`）。两次构建分别出现两类失败：

第一次，opbuild 代码生成通过后，`cust_opapi` 目标编译直接找不到头文件：

```text
In file included from .../op_api/aclnn_<op>.cpp:19:
.../op_api/<op>.h:18:10: fatal error: opdev/op_executor.h: No such file or directory
   18 | #include "opdev/op_executor.h"
      |          ^~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

排查发现 `op_host/CMakeLists.txt` 里 include 路径全部硬编码为 `$ENV{ASCEND_HOME_PATH}/aarch64-linux/include`（以及 `asc/include/tiling`、`pkg_inc/*` 等），而本机 CANN 的 aclnn opdev 头文件实际位于 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h`——注意 `opdev/` 头文件不在 `x86_64-linux/include/` 顶层，而在其下的 `aclnn/` 子目录（include 路径须含 `x86_64-linux/include/aclnn`，`#include "opdev/op_executor.h"` 才能解析）。

第二次，修完 include 后内核编译、CPack 打包全部成功，但构建脚本最后一步报：

```text
Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created.
...
[ERROR] Package not found or empty
```

`build.sh` 的产物存在性检查找的是 `custom_opp_ubuntu_aarch64.run`，而 CPack 在 x86_64 主机生成的包名是 `custom_opp_ubuntu_x86_64.run`（包实际存在，安装也正常）。这两个症状出自同一来源：工程模板按 aarch64 主机写死了架构前缀，而本机是 x86_64。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的构建工程——`op_host/CMakeLists.txt` + `build.sh`，从 aarch64 导向的参考模板复制到 x86_64 主机上为 ascend950 等芯片编译（仿真/交叉开发环境）。
- API 路径：CANN 安装布局 `$ASCEND_HOME_PATH/<arch>-linux/{include,asc,pkg_inc}`；CPack External 打包产物名 `custom_opp_ubuntu_<arch>.run`；build.sh 的包完整性检查（`[ ! -s "$PKG_PATH" ]`）。
- 验证过的平台与 CANN 版本：x86_64 主机 + Ascend950PR 目标（`--soc=ascend950` 编译+打包通过）+ CANN 9.1.0（两个独立算子工程复现；与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（所有基于该模板的算子工程通用）。
- 路线优先级：recommended_fastpath（架构自适应改造）。
- 明确不适用/未验证场景：aarch64 主机上的原生编译（模板原始假设，路径恰好正确，不触发；未在本轨迹出现）；交叉编译（x86_64 主机产出 aarch64 包）未验证；非 CPack External 打包方式未验证。

## 根因

算子工程模板默认部署目标是 aarch64 服务器，把 CANN 安装布局的架构段（`aarch64-linux/`）与 CPack 自解压包名后缀（`custom_opp_ubuntu_aarch64.run`）两处硬编码为 aarch64；在 x86_64 开发主机上，CANN 9.1.0 按 `<主机架构>-linux/` 组织头文件与库、CPack 按主机架构命名产物，两处硬编码同时失配——前者使 op_api 的 `opdev/*` 头文件不可见（`$ASCEND_HOME_PATH/aarch64-linux` 目录根本不存在），后者使构建脚本在打包成功后误报失败。

判别探针（三条命令即可定位）：

```bash
# 1) 头文件不在 aarch64 前缀，也不在 x86_64-linux/include、include 根下：
ls $ASCEND_HOME_PATH/x86_64-linux/include/opdev/op_executor.h 2>/dev/null || echo "NOT FOUND x86_64"
ls $ASCEND_HOME_PATH/aarch64-linux/include/opdev/op_executor.h 2>/dev/null || echo "NOT FOUND aarch64"
ls $ASCEND_HOME_PATH/include/opdev/op_executor.h 2>/dev/null || echo "NOT FOUND root"
# → 三处均 NOT FOUND

# 2) 全盘找头文件实际位置（注意 aclnn 子目录）：
find $ASCEND_HOME_PATH -name "op_executor.h" 2>/dev/null | head -5
# → $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h

# 3) 确认 aarch64 目录根本不存在：
ls -la $ASCEND_HOME_PATH/aarch64-linux 2>/dev/null
# → 无输出（目录不存在）
```

## 解决配方

两处修复，核心都是把架构段从硬编码改为按主机架构自适应。

**修复一：`op_host/CMakeLists.txt` 顶部按主机架构自适应设置目录变量，后续所有 include 路径引用该变量。**

方案 A（架构自适应变量，同一份工程在 x86_64 开发机和 aarch64 服务器上都可构建）：

before（硬编码版本，节选自模板原文件）：

```cmake
# op_host CMakeLists.txt

set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_no_harness_resize_bilinear_def.cpp
)
...
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)
```

after（架构自适应版本）：

```cmake
# op_host CMakeLists.txt

# Detect host architecture for include paths
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ASCEND_LIB_DIR "${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
else()
    set(ASCEND_LIB_DIR "aarch64-linux")
endif()

set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_no_harness_resize_bilinear_def.cpp
)
...
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/include
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_LIB_DIR}/pkg_inc/graph
)
```

`target_include_directories`（`cust_optiling`/`cust_opapi`/`cust_op_proto`）各处同样把 `aarch64-linux` 替换为 `${ASCEND_LIB_DIR}`，且 `cust_opapi` 须包含 `${ASCEND_LIB_DIR}/include/aclnn`（opdev 头文件在 aclnn 子目录下）。

方案 B（最小改动：对整个文件把 `aarch64-linux` 全量替换为 `x86_64-linux`）——同样有效，但工程回到 aarch64 主机构建时需再改回；未发现构建系统提供现成的架构变量。

**修复二：`build.sh` 的产物检查改为按主机架构动态取名。**

before（模板原样，包名检查写死 aarch64）：

```bash
# Check package
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi
```

after（按主机架构动态取名，两种主机都可移植）：

```bash
# Check package
PKG_ARCH=$(uname -m)
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_${PKG_ARCH}.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi
```

替代写法（双包名兼容检查，先查 x86_64 再回落 aarch64）同样验证有效：

```bash
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_x86_64.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
fi
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi
```

注意：朴素 `find | head -1` 检查产物不可靠——find 递归搜索 `${BUILD_PATH}` 会先命中 `build/_CPack_Packages/Linux/External/` 下的 CPack 暂存路径，对该路径做 `-s` 检查同样失败（详见 [Package not found 误报卡](build_sh_package_check_path_misreport.md)）。

## 设计决策

- 用 `CMAKE_HOST_SYSTEM_PROCESSOR` 运行时探测而不是把模板改成 x86_64 硬编码：同一份工程需要在 x86_64 开发机和 aarch64 服务器上都可构建，探测分支保持模板可移植；未知架构回落 aarch64（模板原始假设）。
- build.sh 用 `uname -m` 动态取名（或双包名白名单检查）而不是硬编码任一架构：CPack 产物名跟随主机架构生成，动态取名让同一脚本在两类主机都正确，避免下次换机器再踩一次。
- 为什么直接替换而不引入 `ASCEND_ARCH_PATH` 之类的 cmake 变量：本轨迹未发现构建系统提供现成架构变量；显式替换最小且可审计。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：x86_64 Ubuntu 主机（g++ 13.3.0、cmake 3.20.5），CANN 9.1.0（`$ASCEND_HOME_PATH` 指向安装根，无 `aarch64-linux` 目录），目标芯片 ascend950（Ascend950PR / arch35 / DAV_3510）。
- 相关文件：`op_host/CMakeLists.txt`（修复一：`npu_op_code_gen` 的 COMPILE_OPTIONS 与各 target 的 include 目录）、`build.sh`（修复二：包名检查）、`op_api/<op>.h`（`#include "opdev/op_executor.h"` 触发点）；头文件锚点 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h`。
- 复现命令：`cd <算子目录> && bash build.sh --soc=ascend950 -j8`。

## 预防

- 把 aarch64 导向模板复制到 x86_64 主机后，先 `echo $ASCEND_HOME_PATH && ls $ASCEND_HOME_PATH` 确认本机安装布局是 `x86_64-linux` 还是 `aarch64-linux`，再统一替换所有含架构段的路径；不要等 `fatal error: ... No such file or directory` 再回头查。
- 构建脚本中引用带架构后缀的产物名时，一律用 `$(uname -m)` 动态拼接或 `ls ${BUILD_PATH}/custom_opp_ubuntu_*.run` 通配/双名检查，不写死架构。

## 验证信号

以下信号来自两个独立算子工程（resize_bilinear 类、gather 类）骨架阶段的编译联调，均为构建工具实测：

- 修复前：`cust_opapi` 编译失败，`op_api/<op>.h` 报 `fatal error: opdev/op_executor.h: No such file or directory`（来源平台：950；证据等级：tool_observed）。
- 探针：`$ASCEND_HOME_PATH` 下 `opdev/op_executor.h` 仅存在于 `x86_64-linux/include/aclnn/opdev/`；`aarch64-linux` 目录不存在（来源平台：950；证据等级：tool_observed）。
- include 修复后：cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels 四个 target 全部编译通过、CPack 产出 `custom_opp_ubuntu_x86_64.run`，但 build.sh 仍报 `[ERROR] Package not found or empty`（按 aarch64 找包）（来源平台：950；证据等级：tool_observed）。
- 包名修复后：`Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`，构建脚本验收检查全部通过；一个工程产出 3 个 dtype 内核二进制与约 468KB 安装包，另一工程产出 18 个 kernel `.o`（= 6 x dtype × 3 index dtype）与 474542 字节安装包（来源平台：950；证据等级：tool_observed）。
- 另一工程验证：修复 include 后同一构建命令通过 cust_opapi 编译（构建日志前进到 op_api 源码的后续错误，头文件解析已成功）；修复 build.sh 后 `bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 完整成功，输出 `[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡验证的是 x86_64 主机 + CANN 9.1.0 安装布局；其他 CANN 版本的 `<arch>-linux` 目录命名（若有变化）未验证。
- CPack 包名中的 `ubuntu` 段随打包系统发行版变化，未验证非 Ubuntu 环境；推断可迁移（用通配检查即可），未实测。
- 交叉编译场景（x86_64 主机产出 aarch64 包）不适用本卡结论（未验证）。
- aarch64 主机上原模板路径即正确，无需修改（属模板原始目标环境，未在本轨迹验证）。

**未决项**：

- 无（该根因已闭环到构建成功，两个独立工程互相印证）。更彻底的修法（如模板层面统一用架构变量并探测包名）属于模板改进，未在本轨迹尝试。
## 坏实践

- 在 CMake 里硬编码任一具体架构段路径（无论 aarch64 还是 x86_64）——模板复用到另一类主机必然踩同一个坑。
- 包名检查硬编码任一架构（aarch64 或 x86_64）——换主机即失效。
- 从模板复制 CMakeLists/build.sh 后不核对主机架构目录，直接构建。

## 误诊

- 无（本闭环内一次定位命中：三条 ls/find 探针直接锁定头文件真实位置与 aarch64 目录缺失）。曾短暂考虑"构建系统可能提供 ASCEND_ARCH_PATH 之类的架构变量"，检索未发现现成变量后放弃，采用显式替换。

## 增量（四条独立 trace 复现、顶层符号链接路线与 EXISTS 探测路线）

另一独立轨迹并入四条独立 trace 的同根因复现（add_example 模板副本/自研工程，CANN 9.1.0，x86_64 主机，目标 ascend950/ascend910b），补两条等价修复路线与布局证据：

**1. 布局证据补强（顶层符号链接）**：`ls $ASCEND_HOME_PATH` 顶层有 `include`、`asc`、`pkg_inc`、`lib64`、`x86_64-linux` 等条目、无 `aarch64-linux`；且顶层 `include/asc/pkg_inc/lib64` 均为指向 `x86_64-linux/` 下对应目录的**符号链接**（`readlink -f lib64/cmake/asc-config.cmake` 解析到 `x86_64-linux/lib64/cmake/asc-config.cmake`）。

**2. 路线 C：顶层架构无关路径**（利用上述符号链接，一条 sed 即可对模板副本验证）：

```bash
sed -i 's|\$ENV{ASCEND_HOME_PATH}/aarch64-linux|$ENV{ASCEND_HOME_PATH}|g' op_host/CMakeLists.txt
```

交付形态把 COMPILE_OPTIONS 与 `target_include_directories` 中的 `aarch64-linux` 前缀整体替换为 `$ENV{ASCEND_HOME_PATH}` 顶层路径（`include`、`asc/include/tiling`、`pkg_inc`…）。取舍：顶层入口对 aarch64/x86_64 两类布局都接近中立、不把工程绑死在一种主机架构；代价是依赖"CANN 顶层必有这些目录"的布局约定（未跨版本验证）。

**3. 路线 D：`if(EXISTS ...)` 探测架构子目录**（比按主机架构判等更稳——CANN 安装布局与编译主机架构并非强绑定，可交叉安装）：

```cmake
# =============================================================================
# 检测 ASCEND 架构子目录（aarch64-linux / x86_64-linux）
# =============================================================================
if(EXISTS "$ENV{ASCEND_HOME_PATH}/aarch64-linux")
    set(ASCEND_ARCH_DIR "aarch64-linux")
elseif(EXISTS "$ENV{ASCEND_HOME_PATH}/x86_64-linux")
    set(ASCEND_ARCH_DIR "x86_64-linux")
else()
    set(ASCEND_ARCH_DIR "")
    message(WARNING "ASCEND_HOME_PATH arch subdir not found, using root")
endif()
set(ASCEND_INC_DIR "$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include")
set(ASCEND_ACLNN_INC_DIR "$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include/aclnn")
set(ASCEND_ASC_INC_DIR "$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include")
set(ASCEND_PKG_INC_DIR "$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc")
```

之后所有手写 include 路径统一引用上述变量。另有轨迹用 `CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"` 二分判等（与本卡方案 A 的 `CMAKE_HOST_SYSTEM_PROCESSOR` 在本场景等效）。勿用 `uname -m` 字符串手工拼出 `x86_64-linux` 后写死——换 aarch64 主机同样翻车；要么顶层路径、要么 EXISTS 探测、要么架构判等。

**4. 双架构交叉编译验证**：同一份架构自适应工程对 `--soc=ascend910b` 与 `--soc=ascend950` 两个目标全量清理重建均成功（`Build completed successfully!` + kernel .o 与 custom_opp run 包生成）。

**5. 同族近邻卡**：并行批次对同 nugget 的独立沉淀（x86_64_template_aarch64_path_hardcode / cmakelists_aarch64_hardcode_x86_adapt / aarch64_hardcoded_paths_x86_64_host 三卡）已并入本卡下方"并卡合并"增量，信号互相补强；build.sh 包名检查面的完整判别见 [Package not found 误报卡](build_sh_package_check_path_misreport.md)。

### 复现增量验证信号

- 修复前：模板原样构建 `cust_opapi` 报 `fatal error: opdev/op_executor.h: No such file or directory`（多条独立 trace 一致复现）（来源平台：950；证据等级：tool_observed）。
- 布局证据：`$ASCEND_HOME_PATH` 下无 `aarch64-linux` 目录；`opdev/op_executor.h` 实际位于 `x86_64-linux/include/aclnn/opdev/`；顶层 `include/asc/pkg_inc/lib64` 为指向 `x86_64-linux` 的符号链接（readlink 验证）（来源平台：950；证据等级：tool_observed）。
- 路线 C：sed 修复后模板副本重建编译错误全部消失，kernel .o 与 `custom_opp_ubuntu_x86_64.run` 生成；自研工程用顶层路径全量构建 EXIT=0、0 error、0 warning（来源平台：950；证据等级：tool_observed）。
- 路线 D：EXISTS 探测版 CMakeLists 写入后重建，`cust_opapi` 编译通过、错误推进到 kernel 阶段、最终全工程编译成功（来源平台：950；证据等级：tool_observed）。
- 架构判等路线：`CMAKE_SYSTEM_PROCESSOR` 检测写法落地后 `bash build.sh --soc=ascend950 -j8` 报 `Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- 双架构交叉编译：同工程 `--soc=ascend910b` 与 `--soc=ascend950` 两目标全量清理重建均成功（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 顶层符号链接布局事实仅对 CANN 9.1.0 x86_64-linux 安装实测；CANN 大版本调整安装布局（顶层目录改名/移除）时路线 C 需重新核对，届时退回 EXISTS 探测式写法。
- `npu_op_code_gen`/`npu_op_library` 导入目标自带的 INTERFACE include 路径是否足以覆盖全部头文件未逐项验证；本增量各轨迹中显式 `-I`/include 路径仍是必要的（opdev 头靠它解析）。

## 后续增量（2026-09-17，并卡合并：三条独立 trace 的报错形态、官方模块证据与修复路线补全）

#### 并入自 x86_64_template_aarch64_path_hardcode（Matmul tiling 报错形态、官方模块动态化证据、环境级符号链接路线）

- **报错形态补充（Host 侧 Matmul tiling）**：含 `MultiCoreMatmulTiling` 的 tiling 实现报 `opbuild run failed!` + `op_host/arch35/<op>_tiling.cpp:29:10: fatal error: matrix/bmm_tiling.h: No such file or directory`。注意双因叠加：Host 侧正确 include 写法是 `"matrix/bmm_tiling.h"`（位于 `x86_64-linux/asc/include/tiling/matrix/`），最初写 `"lib/matmul/bmm_tiling.h"` 本身就是错的；改成正确写法仍找不到才是架构前缀硬编码问题。**关键 include 知识**：Host 侧使用 Matmul tiling（`#include "matrix/bmm_tiling.h"`）时，因该头间接 include `"kernel_tiling/kernel_tiling.h"`，必须同时把 `${ASC_LIB_DIR}/asc/include/tiling` 与 `${ASC_LIB_DIR}/ascendc/include/highlevel_api` 两个 include 路径加入编译，缺一即 fatal error。
- **官方模块动态化证据（判别锚点）**：CANN 官方 CMake 模块本身按 `CMAKE_SYSTEM_PROCESSOR` 动态组装路径——`func.cmake:16` 的 `set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)`、`:1248/:1301` 的 `lib/linux/${CMAKE_SYSTEM_PROCESSOR}`、`:1678/:1699` 的 `--platform=${CMAKE_SYSTEM_PROCESSOR}`；`asc_modules/CMakeDetermineASCCompiler.cmake:16/19` 按 `CMAKE_SYSTEM_PROCESSOR` 设 `ASCEND_CANN_PACKAGE_LINUX_PATH`（x86_64→x86_64-linux，aarch64→aarch64-linux）。构建宏自动适配架构，模板里显式写死的 `-I` 不会被该机制纠正——工程侧跟齐即可。探针：`grep -n "CMAKE_SYSTEM_PROCESSOR" $ASCEND_HOME_PATH/x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake`。
- **路线 E（环境级符号链接，模板零改动）**：`ln -sf x86_64-linux <CANN 安装根>/aarch64-linux` 建链后 `bash build.sh --soc=ascend950 -j8` 直接通过。风险：该链接让"交叉编译路径"在 x86_64 上静默解析到本机目录，若环境将来真切换 aarch64 交叉编译会掩盖配置错误——作为不能改工程时的环境级替代，须记录其指向；"掩盖交叉编译配置错误"的风险未实测。
- **cmake configure 通过不代表 include 正确**：configure 阶段不校验这些 include 路径，模板自带样例（不用 Matmul 头）永不触发，错误推迟到编译期。
- build.sh 修复的等价形态：`HOST_ARCH=$(uname -m)` 二分取名，错误信息带上 `$PKG_PATH`（下次误报能直接看出它在找哪个文件）；CMake 内用 `CMAKE_SYSTEM_PROCESSOR`、build.sh 用 `uname -m`，二者在本环境同值。

验证信号（并入，两个独立算子工程，CANN 9.1.0 与 9.2.0）：

- 触发观测：`ls` 探测显示 `aarch64-linux` 不存在、`x86_64-linux` 存在（两套 CANN 安装一致）；`func.cmake:16`/`:1248/:1301` 官方模块动态化事实锚点（来源平台：950 交叉编译；证据等级：tool_observed）。
- include 修正 + 架构自适应后 `cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` 输出 `-- Configuring done`，`matrix/bmm_tiling.h` fatal error 消失；`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 完整通过，包检查命中 `custom_opp_ubuntu_x86_64.run`（CRC/SHA256 输出 + "Build completed successfully!"）（来源平台：950 交叉编译；证据等级：tool_observed）。
- 另一工程（CANN 9.2.0）：`rm -rf build && cmake ..` configure 通过（0.9s）；全量构建输出 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`（约 460KB），端到端成功（来源平台：950 交叉编译；证据等级：tool_observed）。
- 路线 E：`find <CANN 安装根>/aarch64-linux/ -name "*.h"` 报 `No such file or directory`；建链后 `bash build.sh --soc=ascend950 -j8` 成功，3 个 kernel `.o` 与 run 包生成；`CMakeDetermineASCCompiler.cmake:16/19` 动态化证据；UT 编译命令行可见 `-I.../x86_64-linux/include`、`-I.../x86_64-linux/asc/include/tiling`，UT 19/19（Tiling 11 + InferShape 8）通过（来源平台：950；证据等级：tool_observed）。

#### 并入自 cmakelists_aarch64_hardcode_x86_adapt（intf_pub 事实源与包根变量化写法）

- **intf_pub 事实源（路线 C 的官方佐证）**：构建系统自身就是按无 arch 前缀引用的——`x86_64-linux/lib64/cmake/asc-config.cmake` 链路中的 `fwk_modules/intf.cmake:31`：`target_include_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include)`。常用 host 侧头（`op_common/op_host/util/platform_util.h`、`op_common/log/log.h`、`aclnn/opdev/*`、`aclnn_kernels/contiguous.h`）均在包根 `include` 树内经符号链接可达。
- **包根变量化写法（路线 C 的工程化形态）**：把四个 include 目录集中为变量，全部 target 引用变量：

```cmake
set(CANN_INCLUDE_DIR ${ASCEND_CANN_PACKAGE_PATH}/include)
set(CANN_ASC_INCLUDE_DIR ${ASCEND_CANN_PACKAGE_PATH}/asc/include)
set(CANN_TILING_INCLUDE_DIR ${ASCEND_CANN_PACKAGE_PATH}/asc/include/tiling)
set(CANN_ACLNN_INCLUDE_DIR ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn)
```

- 设计取舍（并入）：include 根集合保持模板原有配置不增不减、只替换架构前缀——避免在适配架构的同时引入路径变量，出问题时无法归因；若安装器不再提供包根符号链接，该路线需改为按 arch 目录拼接。

验证信号（并入，两个独立算子工程，x86_64 + CANN 9.1.0）：

- 路径探测：`find $ASCEND_HOME_PATH/aarch64-linux/include` 报 `No such file or directory`；实际布局 `$ASCEND_HOME_PATH/x86_64-linux/{asc,ascendc,include,lib64,pkg_inc,...}`；`<CANN>/include -> x86_64-linux/include` 等符号链接确认；`intf.cmake:31` intf_pub 事实（来源平台：950；证据等级：tool_observed）。
- 架构变量路线适配后：autogen 的 `ascend_all_ops` 编译阶段 include 全部解析，`cust_optiling`/`cust_opapi`/`cust_op_proto` 目标全部 Built，`libcust_opapi.so`、`libcust_opsproto_rt2.0.so`、`libcust_opmaster_rt2.0.so` 链接成功，整包构建产出 kernel 二进制与 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。
- 包根路线构建成功：`bash build.sh --soc=ascend950 -j8` 输出 `[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。

#### 并入自 aarch64_hardcoded_paths_x86_64_host（dlog_pub.h 嵌套包含形态、UT 工程同根因再现与非递归通配修法）

- **报错形态补充（嵌套包含隔层暴露）**：`$ASCEND_HOME_PATH/include/op_common/log/log.h:25:10: fatal error: dlog_pub.h: No such file or directory`——`log.h` 本身经命令行里唯一有效的 `-I$ASCEND_HOME_PATH/include` 找到，但其内部 `#include "dlog_pub.h"` 需要走 `pkg_inc/base`，恰好落在失效的 aarch64 子路径上，报错点离真实病因（include 路径）隔了一层。`dlog_pub.h` 实际位于 `$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/base/dlog_pub.h`，属 CANN 自身发布物——见该报错去找第三方库安装是错方向。
- **同根因第三处：UT 工程**：`tests/ut/CMakeLists.txt` 沿用 `aarch64-linux/include/base/context_builder` 路径，UT 编译报 `fatal error: op_tiling_context_builder.h / op_infer_shape_context_builder.h: No such file or directory`；同修法（根符号链接 + 补 `include/base/context_builder`）后 UT 编译到 `[100%] Built target <op>_op_host_ut`。
- **build.sh 非递归通配修法（第三种等价写法）**：`PKG_PATH=$(ls ${BUILD_PATH}/custom_opp_ubuntu_*.run 2>/dev/null | head -1)`——ls 不进子目录，不会命中 `build/_CPack_Packages/...` 暂存路径（递归 find 会先命中暂存包导致 `-s` 检查误报）。边界：build 目录残留多个历史包时取字典序第一个，清理 build 目录后再构建更稳妥。
- 编译目标 SoC 名旁证：Ascend950PR 芯片对应 `--soc=ascend950`（`tikcpp/ascendc_kernel_cmake/fwk_modules/util/const_var.py` 中 `'ascend950': 'Ascend950PR_9599'`，`opdesc_parser.py` 把各 ascend950pr_* 型号统一映射到 ascend950），不能写 ascend950pr。

验证信号（并入，两台 x86_64 主机、两个独立算子工程，CANN 9.1.0）：

- 修复前：CMake 配置阶段 `fatal error: dlog_pub.h: No such file or directory`（log.h:25 嵌套包含），Configuring incomplete；UT 编译报 `op_tiling_context_builder.h / op_infer_shape_context_builder.h: No such file or directory`（来源平台：950；证据等级：tool_observed）。
- include 路径修复后，同一命令的 host 侧编译（cust_optiling/cust_opapi/cust_op_proto）全部通过，错误推进到 kernel 阶段；最终 `custom_opp_ubuntu_x86_64.run` 安装包生成（内含 3 个 dtype 的 kernel `.o`/`.json`、op_api/op_host so、aic-ascend950-ops-info.json），`bash build.sh --soc=ascend950 -j8` 退出码 0（来源平台：950；证据等级：tool_observed）。
- 另一工程：修复路径后完整构建 3 个 dtype kernel 二进制全部生成，CPack 产出 `custom_opp_ubuntu_x86_64.run`（约 1488KB，SHA256 49ef44ec...），build.sh 产物检查通过（来源平台：950；证据等级：tool_observed）。
- UT 工程按同一修法后编译到 `[100%] Built target <op>_op_host_ut`（来源平台：950；证据等级：tool_observed）。

坏实践补充（并入）：

- 模板工程在 x86_64 机器上 "cmake configure 能过" 就认为构建配置正确——configure 不校验 include 路径，错误推迟到编译期。
- 只修 CMakeLists 不查 build.sh（或反之）——同一硬编码假设会藏在多处（op_host 编译、UT 工程、打包检查三处）。
- 见到 `dlog_pub.h` 缺失就去找第三方库安装——该头文件属于 CANN 自身发布物（`<arch>-linux/pkg_inc/base/`）。
- 用递归 `find ${BUILD_PATH}` 检查包产物——会命中 CPack 暂存路径造成误报（用非递归 ls 通配）。

失效边界补充（并入）：

- 若主机同时安装交叉与原生两套 CANN 布局，需按目标平台显式选择而非依赖探测顺序。
- 模板上游（add_example）本身仍是硬编码，后续新建工程会重复踩坑；可推动模板修复。
