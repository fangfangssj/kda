---
type: "Runbook"
title: Package not found误报：PKG_PATH检查路径错
description: custom_opp 构建成功后仍报 [ERROR] Package not found or empty；包实际存在，是 build.sh 包名检查硬编码 aarch64（或 find 命中 CPack 暂存路径）。x86_64 宿主上模板同源问题还包括 op_host include 硬编码 aarch64-linux 致 opdev/op_executor.h 找不到；按 CMAKE_SYSTEM_PROCESSOR 探测 ASCEND_CANN_PACKAGE_LINUX_PATH 并双包名探测可一并修复。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-16T10:43:46Z'
platforms: ["950"]
---

# Package not found误报：PKG_PATH检查路径错

## 现象

在 x86_64 主机上执行 `bash build.sh --soc=ascend950 -j8` 编译 Cummin 自定义算子包，构建尾部输出：

```
Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created.
Copy /workspace/.../build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/custom_opp_ubuntu_x86_64.run to /workspace/.../build/
CPack: - package: /workspace/.../build/custom_opp_ubuntu_x86_64.run.json generated.
CPack: - package: /workspace/.../build/custom_opp_ubuntu_x86_64.run generated.
[ERROR] Package not found or empty
```

前一行刚说包已生成，后一行立刻报"包不存在或为空"。该误报在后续每次构建中反复出现（清理重建也不例外），但 `ls -la build/custom_opp_ubuntu_x86_64.run` 始终能看到约 448KB 的实际包文件，安装也一直正常。

另一独立会话（BenchNoHarnessUnique，同样照参考样例 add_example 复制改造）在 x86_64 宿主机上首次构建时，同一"模板假设 aarch64 宿主"的根因还以第二个面目出现——host 侧编译直接失败：

```
/workspace/.../operators/bench_no_harness_unique/op_api/bench_no_harness_unique.h:22:10:
fatal error: opdev/op_executor.h: No such file or directory
   22 | #include "opdev/op_executor.h"
      |          ^~~~~~~~~~~~~~~~~~~~~
compilation terminated.
gmake[2]: *** [op_host/CMakeFiles/cust_opapi.dir/build.make:90: ...Error 1
```

（cust_optiling、cust_op_proto 能编过——它们的 include 路径凑巧可用，只有 cust_opapi 的两个翻译单元报此错。）修好 include 路径后，`build.sh` 又在最后的产物检查处退出 `[ERROR] Package not found or empty`——模板脚本检查的是 `${BUILD_PATH}/custom_opp_ubuntu_aarch64.run`，而 CPack 在 x86_64 宿主机上实际生成的自解压包叫 `custom_opp_ubuntu_x86_64.run`。当时 op_host/CMakeLists.txt 的 include 路径是模板原样照抄的 `$ENV{ASCEND_HOME_PATH}/aarch64-linux/include` 等一组 aarch64 路径，而该机器的 CANN 9.1.0 安装目录下只有 `x86_64-linux` 子目录、没有 `aarch64-linux`。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），使用 custom_opp 模板 build.sh / op_host CMakeLists 打包编译的工程；从 add_example 参考样例复制起步、在 x86_64 宿主机本机编译、目标 SoC 为 ascend950（或其他）。
- API 路径：build.sh 的包完整性检查（`[ ! -s "$PKG_PATH" ]`）；CPack 产物命名 `custom_opp_ubuntu_<arch>.run` 与暂存目录 `build/_CPack_Packages/...`；op_host/CMakeLists.txt 的 `npu_op_code_gen(... COMPILE_OPTIONS -I...)` 与 `target_include_directories(cust_optiling/cust_opapi/cust_op_proto ...)`；CANN 自带 ASC cmake 模块（`CMakeDetermineASCCompiler.cmake` 按 `CMAKE_SYSTEM_PROCESSOR` 设置的 `ASCEND_CANN_PACKAGE_LINUX_PATH`）。
- 验证过的平台与 CANN 版本：x86_64 主机 + Ascend 950 目标 + CANN 9.1.0 模板脚本（后续增量的 uname -m 修复验证亦为 9.1.0 / x86_64 交叉构建）。
- 适用粒度：`workflow_only`（构建脚手架行为；随 build track 落 compilation）。
- 路线优先级：`safety_fallback`（不改脚本时的判别方法：忽略该 ERROR 行，以 `ls build/custom_opp_ubuntu_*.run` 为准）；修复脚本（架构探测 + 双包名探测）为 `recommended_fastpath`。本卡 fallback 声明：不改脚本时按"误报"处置，须先确认包文件真实存在。
- 明确不适用/未验证场景：aarch64 交叉编译环境（模板原始假设）；CANN 包布局不同的其他版本（目录名以实际安装为准）；CPack 命名规则变化（如带 OS 版本号）时双包名探测列表需同步扩展。

## 判别方法

**触发一**：构建尾部出现 `[ERROR] Package not found or empty`，且其上一行是 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`。

**触发二**：x86_64 宿主上 cust_opapi 编译报 `fatal error: opdev/op_executor.h: No such file or directory`。

**探针**（可执行）：

```bash
# 包是否真的存在
ls -la build/custom_opp_ubuntu_x86_64.run
# build.sh 实际检查到了哪个路径（find 会先命中 _CPack_Packages 暂存路径）
find build -name "custom_opp_ubuntu_*.run" -not -name "*.json"
# 头文件真实位置（确认是 include 根选错，不是头缺失）
find $ASCEND_HOME_PATH -name "op_executor.h" -path "*/opdev/*"
# → 命中 $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h
# CANN 的 ASC cmake 模块怎么设路径
grep "ASCEND_CANN_PACKAGE_LINUX_PATH" \
  $ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake/asc_modules/CMakeDetermineASCCompiler.cmake
```

**原始观测**：`build/custom_opp_ubuntu_x86_64.run` 存在（448KB）；`find` 输出的第一条是 `build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run`（CPack 暂存路径，排序在真正的产物之前）；`opdev/op_executor.h` 实际位于 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h`，且该安装下无 `aarch64-linux` 目录。

**解释**：误报有两个叠加根因。其一，模板 build.sh 原始检查写死了 aarch64 包名：`PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"`，在 x86_64 主机上 CPack 产物名是 `custom_opp_ubuntu_x86_64.run`，`[ ! -s "$PKG_PATH" ]` 恒真。其二，把它改成 `PKG_PATH=$(find ${BUILD_PATH} -name "custom_opp_ubuntu_*.run" -not -name "*.json" | head -1)` 后，find 递归搜索 `${BUILD_PATH}` 会先命中 `build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run`（CPack 暂存目录），对该路径做 `-s` 检查同样失败——误报依旧。include 失败则是第三种面目：参考样例模板把 CANN 包路径硬编码为 `aarch64-linux` 子目录，而 x86_64 宿主上 CANN 9.1.0 的头文件位于 `x86_64-linux/include`。三处同源于"模板假设 aarch64 宿主"，应一次修完。

## 解决配方

**修复一（build.sh 包名检查：按存在性探测两种架构）**：

```bash
# 修复前（硬编码 aarch64 包名）
# Check package
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi

# 修复后（detect architecture: x86_64 or aarch64）
PKG_PATH=""
for pkg in "${BUILD_PATH}/custom_opp_ubuntu_x86_64.run" "${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"; do
    if [ -f "$pkg" ] && [ -s "$pkg" ]; then
        PKG_PATH="$pkg"
        break
    fi
done
if [ -z "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi
```

（或指向确切产物路径 `PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_$(uname -m).run"`，或在 find 中排除 `_CPack_Packages`——uname -m 方案已由后续增量落盘验证，见文末。）

**修复二（op_host/CMakeLists.txt：按宿主架构探测 include 根）**——在文件顶部加平台探测，三处 include（`npu_op_code_gen` 的 COMPILE_OPTIONS 与 cust_optiling/cust_opapi/cust_op_proto 的 `target_include_directories`）全部改用 `${ASCEND_CANN_PACKAGE_LINUX_PATH}`：

```cmake
# =============================================================================
# Detect platform-specific include path
# =============================================================================
if(NOT DEFINED ASCEND_CANN_PACKAGE_LINUX_PATH)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/x86_64-linux)
    else()
        set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/aarch64-linux)
    endif()
endif()

npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include/tiling
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/op_common
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/base
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/exe_graph
        -I${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/graph
)

target_include_directories(cust_optiling PRIVATE
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include/tiling
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/op_common
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/pkg_inc/base
)
```

（cust_opapi / cust_op_proto 的 `target_include_directories` 同样替换为 `${ASCEND_CANN_PACKAGE_LINUX_PATH}/...`——例如 cust_opapi 需要 `${ASCEND_CANN_PACKAGE_LINUX_PATH}/include/aclnn` 才能解析 `opdev/op_executor.h`。）

## 设计决策

- 第一版修复选 find 而不是按 `uname -m` 拼路径：当时的判断是 find 对产物名变化更鲁棒；事后看 CPack 暂存路径使朴素 find 失效，需排除条件——权衡点是"路径推断的脆弱性 vs 搜索命中的顺序性"。双包名存在性探测对两种架构命名都成立，不依赖搜索顺序。
- 为什么用 `ASCEND_CANN_PACKAGE_LINUX_PATH` 变量而不是把 aarch64 字符串改成 x86_64-linux 写死：CANN 自带 ASC cmake 模块就是按 `CMAKE_SYSTEM_PROCESSOR` 设置这个变量的，复用同一约定后同一份 CMakeLists 在 x86_64 与 aarch64 交叉环境都成立；模板原样硬编码 aarch64 的根源是样例面向 Atlas 交叉编译场景。
- 修复顺序上先解 include（让编译过）、后解包名检查（打包阶段才暴露），两处同源于"模板假设 aarch64 宿主"，一次修完。
- 当前结论属 safety_fallback：脚本修复未完全落地时，以 ls 探针确认包存在、忽略误报。

## 系统信息

- 工具链：CANN 9.1.0 custom_opp 模板 build.sh + CPack（External 打包器）；x86_64 Ubuntu 24.04（g++ 13.3.0、cmake 3.20.5）。
- 相关文件：`build.sh`（PKG_PATH 检查）、`op_host/CMakeLists.txt`（include 修复）、`build/custom_opp_ubuntu_x86_64.run`（真实产物）、`build/_CPack_Packages/Linux/External/`（暂存目录）、`op_api/<op>.h`（`#include "opdev/op_executor.h"` 触发点）、`$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake/asc_modules/CMakeDetermineASCCompiler.cmake`（路径约定参照）。

## 验证信号

段首来源说明：前两条来自同一开发会话（Cummin）的多次构建观测；中间三条来自另一独立会话（BenchNoHarnessUnique，同一模板家族）在 x86_64 宿主机上的构建修复前后对比；末两条来自后续增量（两个独立 trace 的修复落盘验证）。

- 每次"successfully created"后紧跟 `[ERROR] Package not found or empty`，而 `ls -la build/custom_opp_ubuntu_x86_64.run` 显示包存在（448153 字节），find 探针显示 CPack 暂存路径排在首位；安装 `./build/custom_opp_ubuntu_x86_64.run --install-path=... --force` 全程正常（来源平台：950；证据等级：tool_observed）。
- 修复前（include 面）：`bash build.sh --soc=ascend950 -j8` 在 cust_opapi 两个翻译单元报 `fatal error: opdev/op_executor.h: No such file or directory`，构建失败；`find` 探针命中 `x86_64-linux/include/aclnn/opdev/op_executor.h`，且 `$ASCEND_HOME_PATH` 下无 aarch64-linux 目录（来源平台：950；证据等级：tool_observed）。
- include 修复后：同一构建中 cust_optiling / cust_opapi / cust_op_proto 全部编译通过（构建进入 kernel 编译阶段，仅剩 kernel 侧错误）（来源平台：950；证据等级：tool_observed）。
- 包名检查修复后：完整构建成功，输出 `[INFO] Build completed successfully!`，生成 kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 自解压包（来源平台：950；证据等级：tool_observed）。
- 后续增量（uname -m 动态推导）落盘后重跑 `bash build.sh --soc=ascend950 -j8`：脚本输出 `[INFO] Build completed successfully!` 并给出 kernel 二进制与 `build/custom_opp_ubuntu_x86_64.run` 产物路径，不再出现 `[ERROR] Package not found or empty`（来源平台：950；证据等级：tool_observed）。
- 后续增量（双架构兼容修复：x86_64 优先、aarch64 回退）落盘验证：修复前构建日志末尾 `successfully created` 后紧跟 `[ERROR] Package not found or empty`（实际产物 444474 字节存在）；修复后干净构建输出 `[INFO] Package: .../build/custom_opp_ubuntu_x86_64.run` 与 `[INFO] Build completed successfully!` 全绿（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论基于 CANN 9.1.0 模板脚本与 x86_64 主机；模板更新后路径逻辑可能变化。
- `uname -m` 拼接或排除 `_CPack_Packages` 的修复未在本轨迹验证，属未验证方案；双包名探测已在一个会话落盘验证（uname -m 方案已由后续增量在另一独立 trace 落盘验证）。
- 其他版本若包目录结构调整（如 include 层级变化），探测逻辑仍适用但具体子路径须重新核对（未验证，推断可迁移）。
- aarch64 交叉编译环境（模板原始假设）未实测。

**未决项**：

- ~~`uname -m` 拼接或排除 `_CPack_Packages` 的修复未落盘验证（下个 agent 改脚本时一并处理）。~~（uname -m 方案已由后续增量落盘验证，见下。）
## 坏实践

- 把 `[ERROR] Package not found or empty` 当作构建失败而回滚正确的代码改动——该行是误报，包已生成。
- 朴素 `find | head -1` 检查产物：CPack 暂存目录在构建树内且排序靠前，检查结果不代表最终产物。
- 从参考样例复制 CMakeLists/build.sh 后不检查宿主架构直接在 x86_64 上构建——include 根目录与包名检查双双踩 aarch64 假设。
- 遇到 `opdev/op_executor.h: No such file or directory` 就去改业务头文件的 include 写法（如删掉这个 include 或换成相对路径），而不是核对 CANN 包布局里的真实位置。

## 误诊

- 无（Cummin 会话）；Unique 会话本闭环定位一次命中：find 头文件位置 + 读 ASC cmake 模块即确认根因。

## 后续增量（2026-09-16：uname -m 修复落盘验证）

两个后续独立 trace（均为 x86_64 主机构建 ascend950 目标、CANN 9.1.0）把本卡"未落盘验证"的修复方案之一落地并验证：

**修复方案（`uname -m` 动态推导）**：

```bash
# Check package (name depends on platform architecture)
PKG_ARCH=$(uname -m)
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_${PKG_ARCH}.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty: ${PKG_PATH}"
    exit 1
fi
```

选 `uname -m` 动态推导而不是穷举双分支的理由：包名后缀与 CPack 内部取架构的方式保持同源（CPack 按构建主机架构命名，与目标 SoC 无关），新增主机架构无需再改脚本；错误信息带上完整 `PKG_PATH`，下次失配时一眼看出期望的文件名。落盘后干净构建全绿（`[INFO] Build completed successfully!` + `[INFO] Package: ...`）。

另一 trace 采用"x86_64 优先、aarch64 回退"的双分支兼容（与上文修复一等价形态），同样落盘验证消除误报。预防补充：从模板/样例抄 build.sh 后，把其中所有硬编码的绝对产物名（`custom_opp_ubuntu_*.run`、`.json`）改为按 `uname -m` 推导；判断"构建是否成功"以 CPack 的 `successfully created` 输出与实际产物存在性为准，不要只看脚本退出码——脚本自检逻辑本身可能失配。

## 后续增量（2026-09-16，并行策展批：两个等价修复变体）

另一独立会话（MlaProlog 算子，x86_64 主机，CANN cann-9.2.0 目录（version.info 9.1.0 基线））以同一症状族（先 `fatal error: opdev/op_executor.h: No such file or directory`，修好 include 后再 `[ERROR] Package not found or empty`）落盘了两个与上文等价的修复变体：

**变体一（包检查：`find -maxdepth 1` 限定深度）**——把搜索限制在构建根目录，`_CPack_Packages/` 在更深层级不会被命中，从根上避开"朴素 find 先命中 CPack 暂存路径"的坑（与"排除 `_CPack_Packages`"方案等效）：

```bash
PKG_PATH=$(find ${BUILD_PATH} -maxdepth 1 -name "custom_opp_*.run" 2>/dev/null | head -1)
if [ -z "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi
```

（假定 build 目录同时只有一个包产物；多包并行时回到 `uname -m` 确切路径拼接。）

**变体二（include 根：按目录存在性探测而非 `CMAKE_SYSTEM_PROCESSOR`）**——直接探测 CANN 安装下实际存在哪个架构目录，对"工具链处理器与 CANN 包布局不一致"的场景更稳：

```cmake
set(ASCEND_ARCH_DIR "x86_64-linux")
if(EXISTS "$ENV{ASCEND_HOME_PATH}/aarch64-linux")
    set(ASCEND_ARCH_DIR "aarch64-linux")
endif()
# 此后所有 include 用 $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/...（含 cust_opapi 需要的 include/aclnn）
```

验证信号（来源平台：950 构建链；证据等级：tool_observed）：include 路径修复后重建 `cust_opapi` 编译通过（opdev 头解析成功，后续错误只剩 dtype 枚举问题）；包名检查修复后全量构建成功 `[INFO] Build completed successfully!`，产物 `build/custom_opp_ubuntu_x86_64.run`（462KB）与 kernel 二进制均生成，CPack 输出 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created.`

## 后续增量（2026-09-16，并行策展批：两条修复路线的两个额外独立验证）

两个额外独立会话（CANN 9.1.0，x86_64 主机，不同算子工程）对上述两条修复路线分别再次落盘验证，进一步抬高复现计数：

- **find `-maxdepth 1` 通配路线**（add_example 模板副本修复）：误报复现时 EXIT=1、唯一 error 行 `[ERROR] Package not found or empty`，但 CPack 日志尾部有 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`，`build/custom_opp_ubuntu_x86_64.run`（446211 字节）与 kernel `.o` 均存在；改为 `find ${BUILD_PATH} -maxdepth 1 -name "custom_opp_ubuntu_*.run" | head -1` 后 `bash build.sh --soc=ascend950 -j8` EXIT=0，尾部 `[INFO] Build completed successfully!`、`[INFO] Package: .../custom_opp_ubuntu_x86_64.run`（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- **`uname -m` 动态拼接路线**（另一算子工程，双 SoC 目标验证）：`ARCH_SUFFIX=$(uname -m)` 后 `PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_${ARCH_SUFFIX}.run"`；修复后 `bash build.sh --soc=ascend910b -j8` 与 `--make_clean` 后的 `bash build.sh --soc=ascend950 -j8` 均输出 `[INFO] Build completed successfully!`，kernel 二进制与包均生成、无误报退出（来源平台：950 开发机 x86_64 主机；证据等级：tool_observed）。
- 设计要点复述：`uname -m` 与 CPack 的架构后缀取值同源，一处动态化后两类宿主机都无需再改；错误信息带上 `$PKG_PATH`，失配时直接可见期望文件名。
- 模板上游（ascendc-registry-invoke-template 的 add_example build.sh）未同步修复，后续使用该模板的会话会重复遇到该误报。