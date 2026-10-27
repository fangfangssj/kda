---
type: "Runbook"
title: Matmul 高阶 API 头不可见：ini 与 include 补配
description: 自定义算子 kernel 编译默认不含 asc/include，Matmul 高阶 API 头文件找不到；用 custom_compile_options.ini 加 -I 并 configure_file 复制到 autogen，op_host 三目标补 asc/include 与 tiling/matrix。R2（注入前负结果阶段）：该构建链 ascendc_impl_build.py 仅注入两个 -I 且 kernel_operator.h 不传递包含 matmul，照抄内置算子 lib/matmul_intf.h 必挂，host 含 MatmulApiTiling 编译失败移除即恢复；官方 api-matmul.md 口径 950 矩阵乘走 Blaze/tensor_api。R3（op_host 前缀写法）：以 adv_api/matmul/ 前缀 include matmul_tiling.h 时须补 asc/include 根（仅加子目录不足），短名写法有同名头冲突风险。
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T04:05:46Z'
platforms: ["950"]
---

# Matmul 高阶 API 头不可见：ini 与 include 补配

## 现象

在一个 ops-registry-invoke 风格的自定义算子工程（顶层 CMakeLists 用 `npu_op_package(...)` 打包、`bash build.sh --soc=ascend950` 编译）里开发 GQA 前缀注意力算子时，需要把 kernel 中的标量模拟矩阵乘法（Mul+ReduceSum+GetValue+SetValue）替换为 Matmul 高阶 API。此前工程 issue 记录：kernel 代码 `#include "lib/matmul/matmul_intf.h"` 报头文件路径不存在——该 include 路径在自定义算子编译环境里不可见；实际头文件位于 `$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/matmul_intf.h`（`adv_api/matmul_intf.h` 是到 `matmul/matmul_intf.h` 的符号链接）。同时 op_host 侧 tiling 代码要 `#include "matmul_tiling.h"`（位于 `asc/include/tiling/matrix/`），首轮编译还暴露 `cust_op_proto` 目标缺 `asc/include/tiling/matrix` 路径的报错。当时的代码状态：kernel 仅有向量 API 实现，op_host 的 `COMPILE_OPTIONS`/`target_include_directories` 里没有 `asc/include` 相关 `-I`。

## 适用场景

- 工程形态：自定义算子工程（CMake `npu_op_package` + `build.sh --soc=<soc>`），kernel 侧与 op_host 侧分别编译。
- API 路径：kernel `#include "adv_api/matmul_intf.h"`（Matmul 高阶 API）；op_host `#include "matmul_tiling.h"`（`matmul_tiling::MatmulApiTiling` / `TCubeTiling`）；构建扩展点 `custom_compile_options.ini` 与 `npu_op_code_gen` 的 `COMPILE_OPTIONS`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35）+ CANN 9.1.0（x86_64 主机；该环境 `aarch64-linux` 是指向 `x86_64-linux` 的符号链接）。
- 适用粒度：`platform_wide`（同一 CANN 安装布局下引入任何 `asc/include` 高阶 API 头的自定义算子工程通用）。
- 路线优先级：`recommended_fastpath`（使用 Matmul 高阶 API 的前置构建配置）。
- 明确不适用/未验证场景：UT 构建（tests/ut 的 CMake）未同步加该 include，UT 编译仍报 `matmul_tiling.h: No such file or directory`（已知缺口）；其他 CANN 版本的目录布局未验证。C-API 头链对 include 根的类似要求另见 [C-API asc_simd.h 编译缺 include 根卡](c_api_asc_simd_h_include.md)。

## 根因

自定义算子 kernel 编译（ccec）的默认 include 集不含 `asc/include`（adv_api 高阶 API 头文件所在目录），op_host 的 `cust_optiling`/`cust_op_proto` 目标默认也缺 `asc/include` 与 `asc/include/tiling/matrix`；Matmul 高阶 API 的头文件全部在这些目录下，必须显式追加 include 路径。

## 解决配方

分三处修改，全部 verbatim 如下。

（1）算子根目录创建 `custom_compile_options.ini`（单行，给 kernel 编译追加 `-I`）：

```ini
ALL,,-I$ASCEND_HOME_PATH/x86_64-linux/asc/include
```

（2）顶层 `CMakeLists.txt`：在 `npu_op_package(...)` 之前把 ini 复制到构建系统扫描的 autogen 路径（否则 ini 不会被编译流程拾取）：

```cmake
set(package_name bench_no_harness_gqa_custom)

# 复制 custom_compile_options.ini 到 autogen 路径（添加 Matmul 高阶 API include 路径）
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_compile_options.ini
    ${ASCEND_AUTOGEN_PATH}/custom_compile_options.ini
    COPYONLY
)

npu_op_package(${package_name}
```

（3）`op_host/CMakeLists.txt`：给 `cust_optiling` 和 `cust_op_proto` 两个目标都补 `asc/include`（两处 `target_include_directories` 同样改法，`COMPILE_OPTIONS` 块同理）：

```cmake
target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/asc/include/tiling/matrix
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/ascendc/include/highlevel_api
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc/base
)
```

kernel 侧即可正常包含：

```cpp
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"
```

## 设计决策

- 选 `custom_compile_options.ini` 而不是改 op_kernel 的 CMake：ini 是编译流程识别的正式扩展点（参考工程 atvoss_add_example 的用法 `ALL,,-I$ASCEND_HOME_PATH/aarch64-linux/pkg_inc/op_common`），对 kernel 编译命令生效；op_kernel/CMakeLists.txt 本身没有直接的 include 目录注入位。
- `ALL,` 前缀表示对全部 kernel 变体生效；`$ASCEND_HOME_PATH` 由编译环境展开，避免把绝对安装路径写死。
- 首轮失败后才发现 `cust_op_proto` 与 `cust_optiling` 是两个独立目标，各自需要 `target_include_directories`——只改一处时另一个目标仍报 `matmul_tiling.h` 找不到。
- 该路线（Matmul 高阶 API）在本轨迹后续运行阶段被弃用（见失效边界），但构建配置本身闭环可用。

## 系统信息

- 工具链：CANN 9.1.0，x86_64 主机，`ASCEND_HOME_PATH` 指向 cann-9.1.0 安装根，`bash build.sh --soc=ascend950`。
- 目录：算子根（CMakeLists.txt、custom_compile_options.ini、build.sh）、`op_host/`（tiling、CMakeLists.txt）、`op_kernel/arch35/`（kernel 头）。
- 相关文件清单：`custom_compile_options.ini`（新建）、`CMakeLists.txt`（configure_file）、`op_host/CMakeLists.txt`（两目标 include）、`op_kernel/arch35/<op>.h`（include adv_api/matmul_intf.h）、`op_host/arch35/*_tiling.cpp`（include matmul_tiling.h）。

## 完整代码

host 侧使用 Matmul tiling 的完整函数（放在 op_host tiling 实现中，依赖上述 include）：

```cpp
static void CalcMatmulTiling(platform_ascendc::PlatformAscendC& plat,
                             int32_t M, int32_t N, int32_t K,
                             bool isTransB, int32_t* outTiling)
{
    matmul_tiling::MatmulApiTiling tiling(plat);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16, isTransB);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(M, N, K);
    tiling.SetOrgShape(M, N, K);
    tiling.EnableBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling cubeTiling;
    int64_t ret = tiling.GetTiling(cubeTiling);
    if (ret != 0) {
        OP_LOGE("CalcMatmulTiling failed for M=%d N=%d K=%d transB=%d", M, N, K, isTransB);
    }
    // 拷贝 TCubeTiling 到 int32_t 数组
    auto* src = reinterpret_cast<const int32_t*>(&cubeTiling);
    constexpr size_t copyLen = sizeof(optiling::TCubeTiling) / sizeof(int32_t);
    for (size_t i = 0; i < copyLen && i < 64; ++i) {
        outTiling[i] = src[i];
    }
}
```

注意：host 侧用 `optiling::TCubeTiling` 填、kernel 侧 `reinterpret_cast` 回 `AscendC::tiling::TCubeTiling` 的做法在本轨迹后续引发了类型布局不匹配问题（tiling 全零、运行挂起），host 侧应直接使用与 kernel 一致的 `AscendC::tiling::TCubeTiling`——详见本卡「坏实践」。

## 预防

- 在自定义算子工程引入任何 `asc/include` 下的高阶 API（adv_api）或 tiling 头时，先确认三处 include：kernel 编译（ini）、cust_optiling、cust_op_proto。
- ini 只在算子根目录放置不够，必须 configure_file 复制到 `${ASCEND_AUTOGEN_PATH}` 才会被拾取。

## 验证信号

段首来源说明：本卡来自一次 GQA 算子把标量矩阵乘替换为 Matmul 高阶 API 的重写会话——include 路径问题的原始编译失败发生在该会话更早阶段（工程 issue 有记录，非本段验证内容）；本段验证覆盖的是配置修复后的构建结果。

- 修复后完整构建通过：`[INFO] Building project...` 各目标（cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels）编译成功，kernel binary `.o` 与 `BenchNoHarnessGqa_*.json` 生成，`Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`（来源平台：950/Ascend950PR 工具链；证据等级：tool_observed）。
- `[100%] Built target binary`，两个 tiling key 变体的 kernel 均生成（来源平台：950；证据等级：tool_observed）。
- UT 构建仍报 `matmul_tiling.h: No such file or directory`（UT CMake 未同步 include；来源平台：950 主机侧；证据等级：tool_observed）——这是已知缺口，不是本卡配方的失败。

## 失效边界

- 只在 x86_64 主机 + CANN 9.1.0 布局验证；其他版本 `asc/include` 相对位置可能不同（未验证，推断可迁移：ini 中的路径按版本调整即可）。
- build.sh 末尾按 `custom_opp_ubuntu_aarch64.run` 检查包名，x86_64 产物为 `custom_opp_ubuntu_x86_64.run`，会打印 `[ERROR] Package not found or empty`——这是脚本检查误报，产物实际已生成（见 [Package not found 误报卡](build_sh_package_check_path_misreport.md)）。
- 本卡只覆盖构建层闭环；该 Matmul 版 kernel 的运行时行为（507015/挂起）属于另一问题，不在本卡结论内。

**未决项**：

- UT 构建的 include 同步（tests/ut 的 CMake 补 `asc/include/tiling/matrix`）未完成。
- `AscendC::tiling::TCubeTiling` 与 `optiling::TCubeTiling` 的字段级布局差异清单未系统整理（本轨迹只碰到 `K`→`Ka` 一处）。
## 坏实践

- host 侧用 `optiling::TCubeTiling`（`asc/include/tiling/matrix/matmul_tilingdata.h` 的 `BEGIN_TILING_DATA_DEF` 定义）填 tiling，kernel 侧 `reinterpret_cast<TCubeTiling*>`（`adv_api/kernel_tiling.h` 的 `AscendC::tiling::TCubeTiling`）读取——两个类型的字段命名/布局并不保证一致（kernel 侧无 `K` 字段而是 `Ka` 等），错配会导致 tiling 数据全零、Matmul 内部循环失控。正确做法：host 侧也 `using AscendC::tiling::TCubeTiling` 并按其字段名赋值。
- `OP_LOGE` 直接传格式串会编译失败（需要符合其宏参数约定）；`CeilAlign(offset, 512LL)` 在 `int64_t`/`long long` 间有类型不匹配，需 `static_cast<int64_t>(512)`。

## 误诊

- 初次构建失败时先怀疑 `aarch64-linux` 路径不可用；实际该环境 `aarch64-linux` 是指向 `x86_64-linux` 的符号链接，路径本身能解析，缺的只是目标里没写这些 `-I`（`ls -ld` 确认符号链接）。

## 增量（注入前的负结果阶段 + op_host 前缀写法）

### R2：include 注入解法之前的负结果阶段（GQA 工程，CANN 9.1.0）

同一 GQA 工程更早阶段的排查确认了 include 集合的机制层事实（均为 tool_observed）：

1. 自定义 kernel 编译命令（由 `fwk_modules/util/ascendc_impl_build.py` 生成）只追加两个 `-I`：`ascendc/act`（该安装中目录不存在）与 `tikcpp/ascendc_kernel_cmake/ascendc/common`；`ccec` 内置搜索路径也不含任何 ascendc 目录。
2. `kernel_operator.h` 的标准 include 链不传递包含任何 matmul 头（grep 无命中）——标准 kernel include 链天然拿不到 Matmul 高阶 API。
3. CANN 内置算子源码用 `#include "lib/matmul_intf.h"` 能编译，是因为 `lib/` 目录仅在内置算子编译路径中可用——照抄到自定义算子必挂。
4. host 侧 tiling 含 `MatmulApiTiling` 时 `compile ascend_all_ops result: 1` / `opbuild run failed!`；把 Matmul tiling 计算整段移除后同工程 `cmake exit: 0`。
5. 官方最佳实践文档（api-matmul.md）明确：Matmul 高阶 API 适用平台为 Atlas A2/A3（DAV_2201）；Ascend 950（DAV_3510）不适用，矩阵乘开发应使用 Blaze/tensor_api。

该阶段的回退（占位通道）：kernel 移除 Matmul 依赖改用基本向量 API（`Mul` + `ReduceSum` 逐行点积、`GetValue`/`SetValue` 标量循环、输出路径 `Cast` + `DataCopyPad`），host tiling 移除 `MatmulApiTiling` 与 `TCubeTiling` 计算段——向量 API 模拟矩阵乘的性能远低于 Cube 路线，仅作占位/正确性通道（safety_fallback）。后续阶段的 GM/ND + MIX 运行期负结果矩阵与"放弃 Matmul 回退向量"终态见 [950 Matmul 选型卡](950_matmul_api.md) 的 R2 增量。

检索头文件位置与 include 集合的判别命令（可复用）：

```bash
find $ASCEND_HOME_PATH -name "matmul_intf.h" 2>/dev/null
# 结果仅有：x86_64-linux/asc/include/adv_api/matmul_intf.h
#           x86_64-linux/asc/include/adv_api/matmul/matmul_intf.h
find $ASCEND_HOME_PATH -path "*/lib/matmul_intf.h" 2>/dev/null   # 无结果
grep -rn "matmul" $ASCEND_HOME_PATH/x86_64-linux/ascendc/include/basic_api/kernel_operator.h  # 无命中
grep -rn "include.*matmul" $ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_oam/ascendc/*/*.h
# 内置算子用 #include "lib/matmul_intf.h"（lib/ 仅内置编译路径可见）
```

### R3：op_host 以带前缀路径 include matmul_tiling.h 须补 asc/include 根（MoE 分组矩阵乘工程，CANN 9.1.0）

op_host/arch35 的 TilingFunc 源码写 `#include "adv_api/matmul/matmul_tiling.h"` 时，`npu_op_code_gen` 的 `COMPILE_OPTIONS` 只给了 `asc/include/tiling` 与 `asc/include/adv_api/matmul` 两个子目录——后者只够解析短名 `#include "matmul_tiling.h"`，不够解析带 `adv_api/matmul/` 前缀的路径，报：

```text
fatal error: adv_api/matmul/matmul_tiling.h: No such file or directory
   30 | #include "adv_api/matmul/matmul_tiling.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

修复（增加 `asc/include` 根；`target_include_directories(cust_optiling ...)` 与 `target_include_directories(cust_op_proto ...)` 同步增加，并删除原来的 `asc/include/adv_api/matmul` 子目录项）：

```cmake
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/include
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/exe_graph
        -I${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/graph
)

target_include_directories(cust_optiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
)
```

取舍：选"`asc/include` 根 + 源码写全前缀"而不是"子目录 + 短名"——前缀写法与 CANN 官方样例一致，且短名 `matmul_tiling.h` 与 ascendc 树的 `highlevel_api/lib/matmul/matmul_tiling.h` 同名，短名解析易撞同名头。`ASCEND_ARCH_DIR` 由 CMakeLists 顶部按 `aarch64-linux` / `x86_64-linux` 存在性探测生成，避免硬编码架构目录。`COMPILE_OPTIONS` 与 `target_include_directories` 必须同步修改——只改一处会出现"编译过但后续目标解析不到"的不一致。

### 增量验证信号

- R2 全树检索：`matmul_intf.h` 仅存在于 `x86_64-linux/asc/include/adv_api/`（含 `adv_api/matmul/` 子目录）；`*/lib/matmul_intf.h` 与 `*/lib/matmul/matmul_intf.h` 无任何结果（来源平台：950；证据等级：tool_observed）。
- R2 自定义 kernel 编译 include 集合（生成脚本实证）仅含 `ascendc/act`（目录不存在）与 `tikcpp/ascendc_kernel_cmake/ascendc/common`；`kernel_operator.h` include 链无 matmul（来源平台：950；证据等级：tool_observed）。
- R2 host 侧含 `MatmulApiTiling` 时 `compile ascend_all_ops result: 1` / `opbuild run failed!`；移除该段后同工程 `cmake exit: 0`；回退后向量 API 实现可完整编译打包（`build exit: 0`，`custom_opp_ubuntu_x86_64.run` 生成）并在 NPU 上执行（来源平台：950；证据等级：tool_observed）。
- R2 官方文档 api-matmul.md 原文："不适用：Ascend 950（`DAV_3510`）MatMul 开发 → `ascendc-blaze-best-practice` skill（Blaze / tensor_api）"，适用平台为 Atlas A2/A3（DAV_2201）（来源平台：950；证据等级：tool_observed）。
- R3 修复前：`fatal error: adv_api/matmul/matmul_tiling.h: No such file or directory`，`compilation terminated`（来源平台：950；证据等级：tool_observed）。
- R3 修复后：该 fatal error 不再出现，host 编译推进到 tiling.cpp 语法/命名空间阶段；连同后续修复全量回归 `build.sh --soc=ascend950 -j8` 生成 `custom_opp_ubuntu_x86_64.run` 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。

### 增量失效边界

- R2 的 include 集合事实绑定该构建框架（tikcpp/ascendc_kernel_cmake 链）；其他构建框架的 kernel -I 含 ascendc 树 `highlevel_api/lib` 时可直接解析 `lib/matmul/matmul_intf.h`（见 [kernel Matmul 头两棵树卡](kernel_matmul_intf_include_tree.md)），含 `asc/include/adv_api` 时可解析裸 `matmul_intf.h`（见 [TCubeTiling 双定义卡](tcube_tiling_host_kernel_dual_definition.md)）——include 可达性随构建框架而异，采用前先对本工程跑 find/-I 核对。
- 官方文档口径（950 → Blaze/tensor_api）与"注入后可编译"不矛盾：内置算子经内置编译路径使用 Matmul 高阶 API，自定义算子路径的 include 集合与平台适用口径是两层独立事实；注入 include 后的运行期形态见 [950 Matmul 选型卡](950_matmul_api.md) 的 R2 负结果矩阵。
- R3 的目录布局结论验证于 CANN 9.1.0 x86_64-linux；`asc/include`（host 侧新 API 树）与 `include`（GE/图侧旧树）是两棵并行的头文件树，其他版本可能变化（未验证，推断可迁移）。
- R3 只覆盖 op_host 编译的 include 解析；kernel 侧不能 include 宿主 tilingdata 头（会引发 graph/types.h 枚举冲突，见 [TCubeTiling 双定义卡](tcube_tiling_host_kernel_dual_definition.md)）。

### 增量坏实践

- 从 CANN 内置算子（opp/built-in 下源码）抄 `#include "lib/matmul_intf.h"` 到自定义算子——内置编译路径与自定义路径的 include 集合不同，直接不可编译。
- 在 Ascend950 目标上按 A2/A3 的 Matmul 高阶 API 最佳实践照搬实现而不查平台适用范围与 include 可达性。
- 只把 `asc/include/adv_api/matmul` 子目录加进 -I 再在源码里写短名 include——短名易与其他树的同名头冲突，且 `matmul_tiling_base.h` 内部的 `"kernel_tiling/kernel_tiling.h"` 等相对引用仍解析不到。