---
type: "Runbook"
title: host UT 链接 PlatformAscendC 需静态库+dlog
description: host 侧 UT 调 PlatformAscendC 平台查询时连续三错（头文件名/符号未定义/DSO missing）；符号仅在 libtiling_api.a，链接需前置该静态库并补 unified_dlog。
tags: [task.testing, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# host UT 链接 PlatformAscendC 需静态库+dlog


## 现象

在为自定义算子编写 host 侧 tiling 单元测试时，期望值 oracle 需要查询系统 workspace 大小（`platform_ascendc::PlatformAscendC::GetLibApiWorkSpaceSize()`），于是先写了一个最小探针程序（`probe_plat.cpp`）：构造 `fe::PlatFormInfos`、填 mock 平台资源、构造 `PlatformAscendC` 并调用三个查询（AIV 核数 / UB 大小 / LibApiWorkSpaceSize）。用 CANN 的 include 目录和 lib64 下的共享库编译链接，结果连续三次失败，报错原文如下：

第一次（编译期，头文件不存在）：

```
probe_plat.cpp:5:10: fatal error: platform/platform_infos.h: No such file or directory
    5 | #include "platform/platform_infos.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

第二次（链接期，改用正确头文件后，符号未定义；注意此时输出里混入的 `aiv=64 ub=262144 sysWs=16777216` 来自共享临时目录里前一会话遗留的同名旧二进制，不是本次编译产物）：

```
/usr/bin/ld: /tmp/ccORJIJL.o: in function `main':
probe_plat.cpp:(.text+0x83a): undefined reference to `platform_ascendc::PlatformAscendC::GetCoreNumAiv() const'
/usr/bin/ld: probe_plat.cpp:(.text+0x886): undefined reference to `platform_ascendc::PlatformAscendC::GetCoreMemSize(platform_ascendc::CoreMemType const&, unsigned long&) const'
/usr/bin/ld: probe_plat.cpp:(.text+0x8b4): undefined reference to `platform_ascendc::PlatformAscendC::GetLibApiWorkSpaceSize() const'
collect2: error: ld returned 1 exit status
```

（当时链接行已带 `-lops_base`——`grep -rl` 能在 `libops_base.so` 里搜到 "GetLibApiWorkSpaceSize" 字符串，但 `nm -C --defined-only` 证实 `libops_base.so` 没有任何 `PlatformAscendC::Get*` 定义符号；定义符号只在静态库 `libtiling_api.a` 里。）

第三次（链接期，补上 `libtiling_api.a` 之后，又出现 DSO 传递依赖缺失）：

```
/usr/bin/ld: /home/<user>/Ascend/cann-9.0.0-beta.2/lib64/libopp_registry.so: undefined reference to symbol 'CheckLogLevel'
/usr/bin/ld: /home/<user>/Ascend/cann-9.0.0-beta.2/lib64/libunified_dlog.so: error adding symbols: DSO missing from command line
collect2: error: ld returned 1 exit status
```

补上 `-lunified_dlog` 并把 `libtiling_api.a` 放在 `-l` 共享库**之前**后，探针编译链接运行全部成功，三组 mock 核数（48/64/7）下查询输出符合预期（见「## 验证信号」）。

## 适用场景

- 工程形态：算子仓（ops-registry-invoke 类）自定义算子的 host 侧 gtest 单元测试；任何需要在 x86 宿主上调用 `platform_ascendc::PlatformAscendC` 平台查询的 UT/探针程序。
- API 路径：`#include "platform/platform_infos_def.h"`（fe::PlatFormInfos）+ `#include "tiling/platform/platform_ascendc.h"`（asc/include 目录）；链接 `${CANN}/x86_64-linux/lib64/libtiling_api.a` + `-lunified_dlog` + CANN lib64 各共享库。
- 验证过的平台与 CANN 版本：Ascend 950（编译目标 ascend950；host 侧 UT，平台参数经 mock）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（结论绑定 CANN 安装布局与链接行为，不绑定具体算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：未验证其他 CANN 版本（静态库/共享库布局可能变化）；未验证非 x86_64 宿主交叉编译场景；`GetUbBlockSize`/`GetCacheLineSize` 等 `op_common/op_host/util/platform_util.h` 头文件内联工具不经过本链路（它们是头文件模板，无链接需求）。

## 根因

CANN 9.0.0-beta.2 未把 `platform_ascendc::PlatformAscendC` 的实现导出为任何共享库符号——`nm -C --defined-only` 逐库检查证实定义符号（`GetCoreNumAiv`/`GetCoreMemSize`/`GetLibApiWorkSpaceSize` 等）只存在于静态库 `x86_64-linux/lib64/libtiling_api.a`；`libops_base.so` 里出现的 "GetLibApiWorkSpaceSize" 只是字符串，不是可链接的定义符号，所以 `-lops_base` 无济于事。同时：`fe::PlatFormInfos` 的可包含头文件名是 `platform/platform_infos_def.h`（不是直觉的 `platform/platform_infos.h`）；一旦把 `libtiling_api.a` 放上链接行，`libopp_registry.so` 对 `CheckLogLevel` 的未解析引用就要求 `libunified_dlog.so` 显式出现在命令行（现代 ld 的 DSO 规则），且静态库必须放在使用其符号的共享库之前，否则链接器扫过静态库时符号尚未被引用、会被丢弃。

## 解决配方

修复分三步，每一步的 before/after 如下（命令中 `CANN=/home/<user>/Ascend/cann-9.0.0-beta.2`，头文件与库路径按 x86_64-linux 布局）：

1. 头文件 before → after：

```cpp
// before（编译失败：No such file or directory）
#include "platform/platform_infos.h"
#include "tiling/platform/platform_ascendc.h"

// after（编译通过）
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
```

`platform_ascendc.h` 位于 `${CANN}/x86_64-linux/asc/include/tiling/platform/`，编译时需补 `-I${CANN}/x86_64-linux/asc/include`。

2. 链接行 before → after（探针场景）：

```bash
# before（-lops_base 救不了：符号只在静态库里；两次 undefined reference）
g++ ... -L${CANN}/lib64 -lopp_registry -lmetadef -lplatform -lops_base -lc_sec -ldl -lpthread

## after（静态库前置 + 显式 -lunified_dlog；三组 coreNum 全部链接运行成功）
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fno-access-control probe_plat.cpp -o probe_plat2 \
    -I${CANN}/x86_64-linux/include -I${CANN}/x86_64-linux/pkg_inc -I${CANN}/x86_64-linux/pkg_inc/base \
    -I${CANN}/x86_64-linux/asc/include \
    -L${CANN}/lib64 ${CANN}/x86_64-linux/lib64/libtiling_api.a \
    -lopp_registry -lmetadef -lplatform -lc_sec -ldl -lpthread -lunified_dlog
```

3. UT 的 CMakeLists before → after（正式落盘形态；这是本次任务中 UT 工程的实际修改，逐字取自事件流）：

```cmake
## before（UT 原链接行：PlatformAscendC 符号未定义）
target_link_libraries(${UT_EXE} PRIVATE
    gtest_oldabi
    ${CANN_LIB}/libopp_registry.so
    ${CANN_LIB}/libmetadef.so
    ${CANN_LIB}/libplatform.so
    ${CANN_LIB}/libc_sec.so
    dl
    pthread
)

## after（加 asc/include、静态库 libtiling_api.a、unified_dlog）
set(CANN_ASC_INC ${CANN_HOME}/${ASCEND_HOST_ARCH}-linux/asc/include)   # 与 include/pkg_inc 并列新增

target_include_directories(${UT_EXE} PRIVATE
    ${CANN_COMMON_INCLUDE_DIRS}
    ${CANN_ASC_INC}    # tiling/platform/platform_ascendc.h (oracle sysWs query)
    ${KERNEL_INC}
)
target_link_libraries(${UT_EXE} PRIVATE
    gtest_oldabi
    # libtiling_api.a: platform_ascendc::PlatformAscendC (GetLibApiWorkSpaceSize etc.);
    # unified_dlog: explicit DSO required once libtiling_api.a is on the link line.
    ${CANN_LIB}/libtiling_api.a
    ${CANN_LIB}/libopp_registry.so
    ${CANN_LIB}/libmetadef.so
    ${CANN_LIB}/libplatform.so
    ${CANN_LIB}/libc_sec.so
    unified_dlog
    dl
    pthread
)
```

注意编译宏：CANN 头/库按 `_GLIBCXX_USE_CXX11_ABI=0` 构建，UT 编译定义必须一致（探针与 UT 均显式 `-D_GLIBCXX_USE_CXX11_ABI=0`）。

## 设计决策

- 为何选「UT 直接链接 libtiling_api.a」：oracle 需要 `GetLibApiWorkSpaceSize()` 这类平台查询，而符号只在静态库；把静态库链接进 UT 可执行文件是与算子包构建侧（op_host tiling 库同样静态链接 `-ltiling_api`）一致的做法，避免另造 mock 值引入 oracle 偏差。
- 备选与放弃原因：① 继续在共享库里找符号（`-lops_base`）——被 `nm -C --defined-only` 证据否定（无定义符号）；② 在 oracle 里硬编码 sysWs 常量——放弃：该值按头文件注释随芯片类型变化，硬编码会随平台漂移引入假失败；正确做法是 UT 与被测实现在**同一 mock PlatFormInfos** 上各自查询、比对公式而非比对常量。
- 链接顺序：`libtiling_api.a` 放在 `-l` 共享库**之前**（静态库只在被扫描到时提取被引用的目标文件，放后面会因符号尚未被引用而整体丢弃——第三次报错正是这一类问题的变体）。
- 当前方案属哪档路线：recommended_fastpath（本 CANN 版本下唯一被实测验证可行的 host UT 链接组合）。

## 系统信息

- CANN：`/home/<user>/Ascend/cann-9.0.0-beta.2`（`ASCEND_HOME_PATH` 指向该目录；x86_64-linux 布局；宿主 x86_64 Linux，GNU 13.3.0）。
- 关键路径：头 `${CANN}/x86_64-linux/asc/include/tiling/platform/platform_ascendc.h`、`${CANN}/x86_64-linux/include/`（`platform/platform_infos_def.h` 经 pkg_inc/include 解析）；静态库 `${CANN}/x86_64-linux/lib64/libtiling_api.a`；共享库 lib64 下 `libopp_registry.so`/`libmetadef.so`/`libplatform.so`/`libc_sec.so`/`libunified_dlog.so`。
- 相关文件：探针 `/tmp/opencode/probe_plat.cpp`（诊断用，独立于工程）；UT 工程 `operators/<Op>_package/tests/tiling/CMakeLists.txt`（正式修复落点）；UT 源 `operators/<Op>_package/tests/tiling/test_tiling_gtest.cpp`。
- 目标平台：ascend950（包 CMakeLists `ASCEND_COMPUTE_UNIT=ascend950`；UT 为 host 侧运行，平台参数经 mock）。

## 完整代码

探针程序完整源码（最终可编译运行版；头文件与 main 签名/mock 块为修复后形态，其余自初版，逐字取自事件流）：

```cpp
// Probe 2: verify PlatFormInfos mock plumbing for PlatformAscendC queries
#include <cstdio>
#include <map>
#include <string>
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

int main(int argc, char **argv)
{
    int64_t coreNum = (argc > 1) ? atol(argv[1]) : 48;
    fe::PlatFormInfos pi;
    pi.Init();

    {
        std::map<std::string, std::string> socInfos;
        socInfos["ai_core_cnt"] = std::to_string(coreNum);
        socInfos["cube_core_cnt"] = std::to_string(coreNum / 2);
        socInfos["vector_core_cnt"] = std::to_string(coreNum);
        socInfos["core_type_list"] = "AICore";
        socInfos["l2_size"] = "33554432";
        pi.SetPlatformRes("SoCInfo", socInfos);
        std::map<std::string, std::string> aicoreSpec;
        aicoreSpec["ub_size"] = "262144";
        aicoreSpec["l0_a_size"] = "65536";
        aicoreSpec["l0_b_size"] = "65536";
        aicoreSpec["l0_c_size"] = "131072";
        aicoreSpec["l1_size"] = "524288";
        pi.SetPlatformRes("AICoreSpec", aicoreSpec);
        pi.SetCoreNumByCoreType("AICore");
        std::map<std::string, std::string> socVersion;
        socVersion["Short_SoC_version"] = "Ascend950";
        socVersion["NpuArch"] = "3510";
        pi.SetPlatformRes("version", socVersion);
    }

    platform_ascendc::PlatformAscendC plat(&pi);
    fprintf(stderr, "GetCoreNumAiv=%u\n", plat.GetCoreNumAiv());
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    fprintf(stderr, "UB size=%lu\n", ub);
    fprintf(stderr, "LibApiWorkSpaceSize=%u\n", plat.GetLibApiWorkSpaceSize());
    return 0;
}
```

运行（验证用命令，逐字取自事件流）：

```bash
CANN=/home/<user>/Ascend/cann-9.0.0-beta.2
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fno-access-control probe_plat.cpp -o probe_plat2 \
    -I${CANN}/x86_64-linux/include -I${CANN}/x86_64-linux/pkg_inc -I${CANN}/x86_64-linux/pkg_inc/base \
    -I${CANN}/x86_64-linux/asc/include -L${CANN}/lib64 ${CANN}/x86_64-linux/lib64/libtiling_api.a \
    -lopp_registry -lmetadef -lplatform -lc_sec -ldl -lpthread -lunified_dlog
for n in 48 64 7; do LD_LIBRARY_PATH=${CANN}/lib64 ./probe_plat2 $n; echo "exit: $?"; done
```

UT 工程侧的正式修复即上文「## 解决配方」第 3 步的 CMakeLists 段（含 `${CANN_ASC_INC}` 定义），此处不重复。

## 预防

- 写任何调用 `PlatformAscendC` 的 host UT 前，先确认链接组合三件套：头文件 `platform/platform_infos_def.h`、`asc/include` 头目录、`${CANN_LIB}/libtiling_api.a` + `unified_dlog`。
- 判断"某符号在哪个库"用 `nm -C --defined-only <lib>`，不要用 `grep -rl`（字符串命中 ≠ 定义符号；本例 `libops_base.so` 就是字符串假阳性）。
- 共享临时目录里跑探针前先清掉同名旧二进制，或每次换新输出名（如 `probe_plat2`），避免"编译失败却有输出"的假象干扰判断。

## 验证信号

来源说明：以下信号来自本次自定义算子公共 TilingUT（测试先行）开发会话中，为验证 oracle 平台查询可行性而编写的探针程序实测，以及随后 UT 工程以同一链接组合完成的全量构建运行。

- 三组 mock 核数下探针查询全部成功且值受 mock 控制：`GetCoreNumAiv=48 / 64 / 7`（分别等于传入的 mock `vector_core_cnt`），`UB size=262144`（等于 mock `AICoreSpec.ub_size`），`LibApiWorkSpaceSize=16777216`（三组下恒定），每轮 `exit: 0`（来源平台：950（host 侧探针，mock PlatFormInfos，Short_SoC_version=Ascend950）；证据等级：tool_observed）。
- UT 工程采用同一链接组合（CMakeLists 加 `libtiling_api.a` + `unified_dlog` + `asc/include`）后，全量 `run.sh` 链路（包构建 → cpack → 安装 → gtest 编译 → 运行）走通，gtest 可执行文件产出并运行 33 用例（来源平台：950（host 侧 gtest）；证据等级：tool_observed）。

## 失效边界

- 本结论绑定 CANN 9.0.0-beta.2 的库布局；其他版本若把 PlatformAscendC 实现迁入共享库，`-ltiling_api` 与 `-lunified_dlog` 的必要性会变化（未验证，推断需按版本重新 `nm` 确认）。
- `LibApiWorkSpaceSize=16777216` 是 mock `Short_SoC_version=Ascend950` 下的观测值，按头文件注释随芯片类型变化；换芯片/真机环境不要把它当常量（未验证其他 SoC）。
- 交叉编译（非 x86_64 宿主）场景未验证，推断可迁移（路径模式 `<arch>-linux/...` 一致）。

## 坏实践

- 用 `grep -rl "GetLibApiWorkSpaceSize" *.so` 判断符号归属——字符串命中会给出假阳性（`libops_base.so` 命中但无定义符号），导致 `-lops_base` 的无效尝试。
- 把 `libtiling_api.a` 放在 `-l` 共享库之后——静态库被扫描时符号尚未被引用，整体丢弃，报 undefined reference。
- 只加静态库不补 `-lunified_dlog`——`libopp_registry.so` 的 `CheckLogLevel` 未解析，ld 报 "DSO missing from command line"。

## 误诊

- 初见第二次报错时曾怀疑是探针代码问题；实际上编译已通过、纯链接问题。另：该次失败输出里混入的 `aiv=64 ub=262144 sysWs=16777216 (0x1000000) exit: 0` 来自共享临时目录里前一会话遗留的同名旧探针二进制（本次编译产物未生成），差点被误读为"编译失败但程序能跑"——属环境噪音，与链接问题无关（agent 事后核对时间戳与文件确认）。

## 未决事项

- 无（链接根因已追到符号归属与 DSO 规则层面；探针与 UT 均实测通过）。
