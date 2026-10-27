---
type: "Runbook"
title: op_host UT 覆盖率：给 dlopen 的 .so 插桩 gcov
description: op_host UT 通过 dlopen（OpImplSpaceRegistryV2）加载实现 .so，gcov 覆盖率不能重编整个测试工程（全新构建目录段错误）也不能静态链接（注册缺失 50 用例失败）；用 g++ 单独编一个 --coverage 的 _cov.so、备份替换原 .so 后用原测试可执行文件跑，.gcno/.gcda 以 .so 名为前缀生成，跑完恢复。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:16:34Z'
updated_at: '2026-09-17T03:16:34Z'
platforms: ["950"]
---

# op_host UT 覆盖率：给 dlopen 的 .so 插桩 gcov

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上为自定义 argmax 算子的 op_host 单测（75 条 gtest，用例通过 `test_op_host_main.cpp` 里的 `OpImplSpaceRegistryV2::AddSoToRegistry` dlopen 加载 `libbench_no_harness_arg_max_op_host_ut_lib.so` 执行 tiling/infershape）测 gcov 行覆盖率时，两条"直觉路线"都走不通：

1. **重编整个测试工程加 coverage 标志**：`cmake .. -DCMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage -fno-access-control" -DCMAKE_EXE_LINKER_FLAGS="--coverage"` 编译成功，但运行 UT 在第一个 infershape 用例段错误（`Segmentation fault (core dumped)`，exit 139）；换成 `Debug`（-O0 -g、无 coverage）甚至默认参数的**全新构建目录**同样段错误，而**既有 build 目录**同用例 75/75 全过——全新目录构建本身在该环境不稳定（根因未明，见遗留问题）。另外 `./ut | tail -5; echo $?` 会把 tail 的退出码当测试退出码，出现过 "Exit: 0" 假象，须直接重定向到文件再看真实 139。
2. **把 op_host 源静态链进测试可执行文件**（自写 CovMain.cpp 直接 RUN_ALL_TESTS）：50 条用例失败，报 `C++ exception with description "not found BenchNoHarnessArgMax" thrown in the test body`——算子实现注册发生在被 dlopen 的 .so 里，静态链接后 `GetSpaceRegistry()->GetOpImpl(...)` 拿不到实现。

可行路线是**只给被 dlopen 的 .so 插桩**：用 g++ 按原 CMake 的 include/链接配置单独编译一个 `lib..._cov.so`，备份并替换原 .so，用原测试可执行文件跑（dlopen 自然加载 cov 版），结束后恢复原 .so。

## 适用场景

- 工程形态：op_host UT（gtest + dlopen 加载算子实现 .so 的 OpImplSpaceRegistryV2 模式）的代码覆盖率测量。
- API 路径：g++ `-fprofile-arcs -ftest-coverage -fPIC -shared --coverage`；gcov/lcov（Ubuntu 13.3.0 gcov / lcov 2.0-1）；CANN 9.1.0 头文件与 librt2_registry.a/libtiling_api.a 等链接库。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only（工具链测量流程，不涉及 kernel 行为）。
- 路线优先级：recommended_fastpath（在本轨迹环境下是唯一验证可行的路线）。
- 明确不适用 / 未验证场景：全新构建目录段错误的根因未解，若该问题不存在（其他环境）则重编整工程加 coverage 标志也是可行选项；kernel（设备侧）代码覆盖率不适用本流程。

## 系统信息

- 工具链：CANN 9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根），x86_64 交叉性质的头文件路径（aarch64-linux/asc、pkg_inc 等）；gcov 13.3.0、lcov 2.0-1、genhtml。
- 目录布局：`tests/ut/build/op_host/`（既有可用构建目录：测试可执行文件 + libbench_no_harness_arg_max_op_host_ut_lib.so）；op_host 源在算子目录 `op_host/`（def/infershape）与 `op_host/arch35/`（tiling）。
- 相关文件：`tests/ut/op_host/test_op_host_main.cpp`（dlopen 注册入口）；`tests/ut/common/tiling_case_executor.cpp`（GetSpaceRegistry()->GetOpImpl 消费方）；`op_host/bench_no_harness_arg_max_def.cpp`、`op_host/bench_no_harness_arg_max_infershape.cpp`、`op_host/arch35/bench_no_harness_arg_max_tiling.cpp`（覆盖率目标）。

## 完整代码

第 1 步：在既有可用构建目录里单独编译 coverage 版 .so（include/链接库照抄原 CMakeLists；源用仓库内路径，`../../op_host` 相对路径会指错层报 `No such file or directory`）：

```bash
cd tests/ut/build/op_host && \
ASCEND="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH 未设置}" && \
REPO=<算子仓根>/operators/bench_no_harness_arg_max && \
OP_DIR=${REPO}/op_host && \
KERNEL_DIR=${REPO}/op_kernel && \
INCLUDES="-I../common -I${REPO}/tests/ut/common -I${OP_DIR} -I${OP_DIR}/arch35 -I${KERNEL_DIR}/arch35 \
 -I${ASCEND}/include -I${ASCEND}/aarch64-linux/include -I${ASCEND}/aarch64-linux/include/aclnn \
 -I${ASCEND}/aarch64-linux/include/base -I${ASCEND}/aarch64-linux/include/base/context_builder \
 -I${ASCEND}/aarch64-linux/asc/include -I${ASCEND}/aarch64-linux/asc/include/tiling \
 -I${ASCEND}/aarch64-linux/asc/include/tiling/platform -I${ASCEND}/aarch64-linux/pkg_inc \
 -I${ASCEND}/aarch64-linux/pkg_inc/op_common -I${ASCEND}/aarch64-linux/pkg_inc/base \
 -I${ASCEND}/aarch64-linux/pkg_inc/base/registry -I${ASCEND}/aarch64-linux/pkg_inc/exe_graph \
 -I${ASCEND}/aarch64-linux/pkg_inc/graph" && \
# 备份原 .so
cp libbench_no_harness_arg_max_op_host_ut_lib.so libbench_no_harness_arg_max_op_host_ut_lib.so.bak && \
# 编译 coverage 版 .so
g++ -std=c++17 -fprofile-arcs -ftest-coverage -fno-access-control -fPIC -shared \
  -DBUILD_SOC_VERSION=Ascend950 -D_GLIBCXX_USE_CXX11_ABI=0 \
  ${INCLUDES} \
  -Wl,--whole-archive ${ASCEND}/lib64/librt2_registry.a ${ASCEND}/lib64/libtiling_api.a -Wl,--no-whole-archive \
  ${ASCEND}/lib64/libopp_registry.so ${ASCEND}/lib64/libregister.so ${ASCEND}/lib64/libmmpa.so \
  --coverage \
  ${OP_DIR}/bench_no_harness_arg_max_def.cpp \
  ${OP_DIR}/bench_no_harness_arg_max_infershape.cpp \
  ${OP_DIR}/arch35/bench_no_harness_arg_max_tiling.cpp \
  -o libbench_no_harness_arg_max_op_host_ut_lib_cov.so
```

第 2 步：替换并运行（dlopen 会加载 cov 版 .so；.gcno 在编译期已生成、前缀为 .so 名）：

```bash
cp libbench_no_harness_arg_max_op_host_ut_lib_cov.so libbench_no_harness_arg_max_op_host_ut_lib.so && \
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH} && \
./bench_no_harness_arg_max_op_host_ut > /tmp/ut_coverage/cov_run.txt 2>&1; echo "Exit: $?"
# 产物：libbench_no_harness_arg_max_op_host_ut_lib_cov.so-bench_no_harness_arg_max_{def,infershape,tiling}.gcno/.gcda
```

第 3 步：读数（lcov 对该形态报 mismatch 错误，用 `--ignore-errors mismatch,gcov` 或直接 gcov）：

```bash
# 直接 gcov：每个 .gcda 对应源文件的行覆盖
gcov -o . bench_no_harness_arg_max_tiling.gcda | grep "Lines executed"
# 或 lcov 流程（忽略 mismatch）+ 只保留 op_host 源
lcov --capture --directory . --output-file /tmp/ut_coverage/coverage.info \
  --rc lcov_branch_coverage=1 --ignore-errors mismatch,gcov && \
lcov --extract /tmp/ut_coverage/coverage.info '*/bench_no_harness_arg_max/op_host/*' \
  --output-file /tmp/ut_coverage/coverage_filtered.info && \
lcov --list /tmp/ut_coverage/coverage_filtered.info
genhtml /tmp/ut_coverage/coverage_filtered.info --output-directory /tmp/ut_coverage/coverage_html --ignore-errors mismatch
```

第 4 步：恢复原 .so 并清理（勿把 cov .so/备份留在构建目录污染后续构建）：

```bash
cp libbench_no_harness_arg_max_op_host_ut_lib.so.bak libbench_no_harness_arg_max_op_host_ut_lib.so && \
rm -f *.gcno *.gcda *.gcov libbench_no_harness_arg_max_op_host_ut_lib.so.bak libbench_no_harness_arg_max_op_host_ut_lib_cov.so && \
# 最终回归：全套 UT 重跑确认无损
cd .. && bash run.sh
```

## 设计决策

- 为什么插桩 .so 而不是测试可执行文件：算子实现的注册与执行都在 dlopen 的 .so 内，覆盖率目标（tiling/infershape/def）也都在 .so 里；测试 main 本身无测量价值。
- 为什么不用静态链接绕开 dlopen：实测 `GetSpaceRegistry()->GetOpImpl("BenchNoHarnessArgMax")` 返回空、50 用例抛 "not found"——注册依赖 .so 加载路径，静态方案需重造注册流程，成本远高于 .so 插桩。
- 为什么在既有 build 目录操作而不是新目录：全新构建目录在本环境段错误（coverage/Debug/默认配置均复现，根因未明）；既有目录产物已验证可用。
- .gcno/.gcda 前缀是 .so 文件名（`lib..._cov.so-<src>.gcda`）：gcov/lcov 读数时以其为锚点，目录里同时存在无前缀的同名 .gcno 会造成 400%/2.8% 之类的错乱读数，以前缀文件为准。
- 当前方案属 recommended_fastpath（本环境内唯一验证可行）。

## 预防

- 做 .so 替换前必须 `cp` 备份，结束时恢复 + 删 cov 产物，否则后续构建/测试会静默用到插桩版 .so（带运行时开销与 .gcda 副作用）。
- 判断测试真实退出码时不要接管道（`| tail` 会吞退出码），重定向到文件后单独 `echo $?`。

## 验证信号

来源说明：为 op_host UT（dlopen 加载实现 .so 的模式）测 gcov 行覆盖率，两条常规路线失败后落地 .so 插桩方案。

- 重编整工程加 coverage 标志：构建成功但 UT 段错误 exit 139（Debug/默认参数的全新目录同样 139，既有 build 目录 75/75 通过）（来源平台：950；证据等级：tool_observed）
- 静态链接 op_host 源：50 条用例失败 `not found BenchNoHarnessArgMax`（来源平台：950；证据等级：tool_observed）
- .so 插桩方案：替换后 75/75 全过（exit 0），`lib..._cov.so-*.gcno/.gcda` 生成（来源平台：950；证据等级：tool_observed）
- 覆盖率读数：tiling.cpp 97.80%（178/182 行，未覆盖 4 行为 GetDtypeSize default 防御分支与 TilingParse 空 stub）、infershape.cpp 100%（26/26）、def.cpp 7.14%（28 行注册宏）；lcov 汇总（含 def.cpp）88.0%（205/233 行）、函数 60%（6/10）（来源平台：950；证据等级：tool_observed）
- 恢复原 .so 并清理后全套 UT 回归 105/105（75 op_host + 30 op_api）通过（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本流程覆盖 host 侧 .so（tiling/infershape/def）；设备侧 kernel 代码的覆盖率不适用。
- 全新构建目录段错误为环境未解问题：换环境（若不存在该问题）时"整工程 coverage 标志"路线亦可，读数方式相同。
- lcov mismatch 错误属 gcno/gcda 形态差异（-fPIC -shared 插桩），用 `--ignore-errors mismatch,gcov` 绕过或直接 gcov；精确分支覆盖读数未验证。
- 未验证场景：clang/llvm-cov 工具链、非 gtest 的 UT 框架。

**未决项**：

- **优先项：复现并追根因**——全新 cmake 构建目录下 op_host UT 第一个 infershape 用例段错误（coverage/Debug/默认配置均复现、既有目录正常、cmake cache 对比无差异），需最小化复现定位。
- def.cpp 覆盖率 7.14% 属注册宏形态（测试进程内不展开），如需提升要靠专项注册路径测试，本流程无法覆盖。
## 坏实践

- 把 op_host 源静态链进自写 main 期望绕开 dlopen——注册在 .so 加载路径上，静态方案拿不到实现（50 用例 "not found"）。
- 用相对路径 `../../op_host` 引源（构建目录嵌套层级容易差一层，报 `cc1plus: fatal error: ... No such file or directory`），用仓库内明确路径。
- 在同一目录混入无前缀 .gcno 与带 .so 前缀 .gcda 再读 lcov——覆盖率先出现 400%/2.8% 的错乱值。
- `./ut | tail -5; echo "Exit: $?"` 判活——拿到的是 tail 的退出码。

## 误诊

- "段错误是 gcov 插桩与 CANN SDK 冲突"：Debug（无 coverage）全新目录同样段错误，排除插桩因素；真实差异在"全新构建目录 vs 既有构建目录"，机制未明。
