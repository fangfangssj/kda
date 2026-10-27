---
type: "Runbook"
title: "UT 报 not found 算子：单目标 make 漏建 .so"
description: "op_host UT 可执行与注册库 .so 是两个独立 CMake 目标，make 单目标只建可执行不建 .so；UT main 仍打印 initialized successfully，随后 GetOpImpl 抛 C++ 异常 \"not found <OpName>\"（Tiling 用例）或首例段错误 exit 139（Infershape 用例）——先 ls 确认 .so 是否存在，再全量 make。复现增量：add_dependencies 把 .so 挂为可执行前置依赖的持久修复与 run.sh 固定全量构建。复现增量：重建 build 目录后只编单个 UT 目标同样触发；LD_LIBRARY_PATH 对该注册加载机制无效。"
tags: ["task.testing", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: '2026-09-17T04:17:42Z'
platforms: ["950"]
---

# UT 报 not found 算子：单目标 make 漏建 .so

## 现象

自定义算子（Ascend950PR / arch35，CANN 9.1.0）op_host UT 调试中，上一处段错误修好后重新运行 gtest，用例不再崩溃但直接失败，gtest 输出：

```text
[ DBG ] DO_TILING step: GetOpImpl done, opName=BenchNoHarnessTopK
unknown file: Failure
C++ exception with description "not found BenchNoHarnessTopK" thrown in the test body.
[  FAILED  ] BenchNoHarnessTopKTilingTest.tiling_2d_last_dim_basic (0 ms)
```

值得注意的是测试 main 启动时打印的信息看起来一切正常：

```text
Loading op_host .so from: .../tests/ut/build/op_host/lib<op>_op_host_ut_lib.so
OpImplSpaceRegistryV2 initialized successfully
```

此前为了快速迭代，构建用的是单目标命令 `make <op>_op_host_ut -j$(nproc)`（只造可执行文件）。

## 适用场景

- 工程形态：算子仓 op_host/op_api gtest UT（可执行 + dlopen 注册库 .so 的双目标工程），常用 `make <ut名>` 单目标快速迭代时。
- API 路径：`gert::DefaultOpImplSpaceRegistryV2` / `AddSoToRegistry` / `GetOpImpl`；CMake 双目标（`<op>_op_host_ut` 可执行 + `<op>_op_host_ut_lib` 共享库）。
- 验证过的平台与 CANN 版本：950（SOC_VERSION=ascend950）+ CANN 9.1.0，x86_64 host。
- 适用粒度：platform_wide——所有该结构的算子 UT 工程。
- 路线优先级：recommended_fastpath（构建/运行 UT 一律 `make -j$(nproc)` 全量或直接跑 run.sh）。
- 明确不适用/未验证场景：算子注册名拼错、whole-archive 链接丢失等同样表现为 "not found" 的其他根因不在本卡（本卡判别点在 .so 文件缺失）。

## 根因

单句归因：UT 的可执行文件与注册库 .so 是两个独立 CMake 目标（构建序上可执行在 92%、.so 在 96%），`make <可执行目标>` 不会连带构建 .so；可执行 dlopen 一个不存在的 .so 路径时框架仍打印 "Loading ... / initialized successfully"，随后 `GetOpImpl("<OpName>")` 返回 nullptr，执行器抛 `std::invalid_argument("not found " + opName)`。

## 判别方法

- 触发信号：UT 用例抛 `C++ exception with description "not found <OpName>"`，且 main 启动信息"正常"（initialized successfully 不代表 .so 已加载注册成功）。
- 可执行探针（按序）：

```bash
# 1) 先确认 .so 是否存在（注意：不要先跑 nm——文件缺失时 nm 2>/dev/null 静默无输出，会误导）
ls -la <build>/op_host/lib*<op>*_ut_lib.so

# 2) 存在时再查注册符号
nm -D <build>/op_host/lib*<op>*_ut_lib.so | grep -iE "<OpName>|IMPL_OP|Register"

# 3) 确认构建目标顺序（可执行在前、.so 在后，二者无依赖关系）
cd <build> && make -j$(nproc) 2>&1 | grep -E "Built target|Linking"
```

- 原始观测（本轨迹）：`ls` 返回 `No such file or directory`；`find . -name "*.so"` 无结果；全量 make 输出 `[ 92%] Built target ..._op_host_ut` → `[ 96%] Linking CXX shared library lib..._ut_lib.so` → `[ 96%] Built target ..._op_host_ut_lib`，随后 .so 出现（2841416 字节）。

解释：单目标 make 只请求可执行目标，.so 目标不在其依赖闭包里；框架 main 对"加载 .so"只打印固定文案，dlopen 失败不产生可见错误，注册表为空，最终在 GetOpImpl 处才以 "not found" 异常暴露。

## 解决配方

构建 UT 一律全量构建（或使用工程的 run.sh）：

```bash
cd <ut>/build && make -j$(nproc)          # 默认目标 all：可执行 + .so + op_api UT
# 或
cd <ut> && bash run.sh -c false           # run.sh 内部全量构建后运行
```

验证 .so 生成后再跑可执行：

```bash
ls -la <build>/op_host/lib*<op>*_ut_lib.so && \
LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH ./<op>_op_host_ut
```

## 验证信号

来源说明：同一次 op_host UT 联调，"not found" 异常定位到 .so 缺失后全量重建（950 host，CANN 9.1.0）。

- `ls`/`find` 探针确认修复前 `lib<op>_op_host_ut_lib.so` 不存在（来源平台：950 host；证据等级：tool_observed）。
- 全量 make 后 `[ 96%] Linking CXX shared library lib..._ut_lib.so`，.so 生成 2841416 字节（来源平台：950 host；证据等级：tool_observed）。
- 全量构建后 op_host UT 27/27 通过（infershape 13 + tiling 8 + fail 路径 6），后续 op_api UT 13/13、run.sh 全流程 40/40 通过（来源平台：950 host；证据等级：tool_observed）。

## 失效边界

- 若 `ls` 确认 .so 已存在而仍报 "not found"，则是注册符号/链接问题（如 whole-archive 未生效、算子名大小写不一致），本卡不适用。
- 构建顺序与目标命名随 UT 工程模板版本可能变化（本轨迹可执行 92% / .so 96%），判别以"单目标 make 后 .so 是否生成"为准。

**未决项**：

- UT 框架 main 对 dlopen 失败无告警、框架对"单目标构建"无提示，属工程可用性缺口，可向上游反馈（加 file-exists 断言或让可执行目标依赖 .so 目标）。
## 坏实践

- `nm -D <file> 2>/dev/null | grep ...` 在文件不存在时静默输出空——本轨迹先跑 nm 得到"无任何 BenchNoHarness 符号"的假象，差点误判为注册符号被 strip；探针必须先验证目标文件存在。
- 被 "Loading op_host .so from: ..." / "OpImplSpaceRegistryV2 initialized successfully" 的启动文案骗过，以为加载环节没问题——这两行是固定打印，不反映 dlopen 结果。

## 误诊

- 最初怀疑 .so 链接时 whole-archive 未包含注册符号、或 IMPL_OP_OPTILING 注册进了别的 registry——nm 探针"无符号"实际是文件不存在的假输出，ls 之后立即推翻。

## 增量（roi_align 独立复现：Infershape 段错误形态与 add_dependencies 持久修复）

另一独立轨迹并入第二条独立 trace（roi_align 类算子，x86_64 主机 + SOC_VERSION=ascend950，CANN 9.1.0），补齐 Infershape 用例的失败形态与持久修复：

**1. Infershape 段错误形态（补全失败面）**：单目标 `cmake --build . --target bench_no_harness_roi_align_op_host_ut -- -j$(nproc)` 构建显示 `[100%] Built target ..._op_host_ut`，运行时两类失败交替出现：

- `--gtest_filter="*Tiling*"`：每个用例抛 `C++ exception with description "not found BenchNoHarnessRoiAlign" thrown in the test body.`（全部 FAILED）。
- 不过滤直接跑：第一个 Infershape 用例 `test_infershape_basic` 卡住后 `Segmentation fault`（`timeout: the monitored command dumped core`，EXIT CODE: 139）。

日志开头都打印 `Loading op_host .so from: .../build/op_host/libbench_no_harness_roi_align_op_host_ut_lib.so` 与 `OpImplSpaceRegistryV2 initialized successfully`，表面上"加载成功"。

**2. 持久修复（add_dependencies，独立 trace 新增）**——把 .so 挂为可执行的前置依赖，从构建图上消除"只建了 exe"的可能：

```cmake
# before
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)
```

```cmake
# after
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

# Ensure .so is built before executable (executable loads .so at runtime)
add_dependencies(${UT_EXE} ${OP_HOST_SO})
```

选 `add_dependencies` 而不是把 .so 直接 `target_link_libraries` 进可执行：UT 框架的设计就是运行期 dlopen（注册表机制），链接进可执行会改变被测代码的加载方式。配套 `run.sh` 统一 `cmake --build . -- -j$(nproc)`（all 目标）而不是逐目标构建。

**3. 判别探针补充**：`ls -la build/op_host/lib<op>_op_host_ut_lib.so`（实测不存在，`nm` 报 cannot open ... No such file or directory）；`find build -name "*.so" -o -name "*op_host_ut*"`。见到 "not found <OpName>" 或 Infershape 用例段错误，先查 .so 是否构建，再怀疑注册宏/ABI。

### 复现增量验证信号

- 修复前：`--gtest_filter="*Tiling*"` 11 个用例全部 `C++ exception ... "not found BenchNoHarnessRoiAlign"`；不过滤运行 `test_infershape_basic` 段错误，EXIT CODE: 139；`ls build/op_host/` 只有可执行文件，`lib..._op_host_ut_lib.so` 不存在（来源平台：950；证据等级：tool_observed）。
- `cmake --build . -- -j` 构建全部目标：`.so` 产出（约 2.8MB）；`--gtest_filter="*Tiling*"` 11/11 OK，`--gtest_filter="*Infershape*"` 4/4 OK（EXIT CODE: 0），全量 15 tests PASSED（来源平台：950；证据等级：tool_observed）。
- 加 `add_dependencies` 并用 run.sh 全新构建复跑：op_host 15/15、op_api 20/20 PASSED（来源平台：950；证据等级：tool_observed）。

### 复现增量坏实践

- 见到 "not found <OpName>" 就去改注册宏、翻 `_GLIBCXX_USE_CXX11_ABI`、重写 OP_ADD/IMPL_OP_OPTILING——本会话先走了 `nm -D` 查符号的路才定位到 .so 根本没构建。
- "日志打印 OpImplSpaceRegistryV2 initialized successfully，说明 .so 加载成功了"——该日志在 .so 缺失时同样打印；真正的判据是 `.so` 文件存在与否与用例是否还报 not found。

### 复现增量遗留问题

- `AddSoToRegistry` 对缺失 .so 静默通过的确切实现（源码层）未核实，仅有行为观测；如需彻底防呆可在 UT main 加载失败处加显式报错（本增量的 add_dependencies 已在构建图层面闭合该缺口）。

## 后续增量（2026-09-17，并行策展批：第三条独立 trace——重建 build 目录后单目标触发；LD_LIBRARY_PATH 无效）

第三条独立轨迹（MoE routing 类算子 tiling 修改后的 UT 重建，Ascend950PR，CANN 9.1.0）确认另一触发变体并补两个判别细节：

**1. 重建 build 目录后只编单个 UT 目标同样触发**：`cd tests/ut/build && make bench_<op>_op_host_ut -j8` 后运行 UT，7/7 用例（无论 gtest_filter 过滤条件）统一抛 `C++ exception with description "not found BenchNoHarnessMoeReRouting" thrown in the test body`，0 通过；换到 `build/op_host` 目录下直接执行同样失败。`ls build/op_host/libbench_<op>_op_host_ut_lib.so` → `No such file or directory`——重建 build 目录把 .so 一并清掉后，单目标 make 不会重建它。

**2. LD_LIBRARY_PATH 对该注册机制无效**：把 .so 路径放进 LD_LIBRARY_PATH 不能修复——`OpImplSpaceRegistryV2->AddSoToRegistry` 按可执行文件同目录拼路径 dlopen，不是按动态库搜索路径找的；判别时直接 `ls` 同目录 .so，不要走环境变量方向。

**3. UT main 加载逻辑探针**：`grep -n "AddSoToRegistry\|op_host_ut_lib\|LIBRARY_PATH" tests/ut/op_host/test_op_host_main.cpp` 可确认 UT main 把 `lib<op>_op_host_ut_lib.so` 注册进注册表的路径来源；找不到 .so 时 tiling 查询按算子名找不到实现，于是每个用到 tiling 的用例体统一抛 "not found <算子名>"。

**4. 运行目录惯例**：UT main 以可执行文件自身位置解析 .so 路径；在 build 根目录用相对路径 `./op_host/<ut>` 调用虽然能启动，但惯例是进入 op_host 目录再执行，保持与加载日志中的路径一致。

### 复现增量验证信号

- 失败态：`make <op>_op_host_ut` 后运行，7/7 用例抛 `C++ exception "not found BenchNoHarnessMoeReRouting"`，0 通过（来源平台：950；证据等级：tool_observed）。
- 伴随 .so 确认缺失：`ls .../libbench_<op>_op_host_ut_lib.so` → No such file or directory（来源平台：950；证据等级：tool_observed）。
- 修复态：完整 `make -j8` 产出 `Linking CXX shared library libbench_<op>_op_host_ut_lib.so` 后，同目录运行打印 "Loading op_host .so from: .../libbench_<op>_op_host_ut_lib.so / OpImplSpaceRegistryV2 initialized successfully"，7/7 通过；随后全量 op_host UT 75/75 通过（来源平台：950；证据等级：tool_observed）。

### 复现增量坏实践

- 见到 "not found <算子名>" 就去重装 CANN 算子包或检查 `opp/vendors`——该报错发生在 UT 本地脚手架层面，与已安装算子包无关。
- 试图用 LD_LIBRARY_PATH 指到 .so 来"修复"——该注册机制不按动态库搜索路径加载，无效。