---
type: "Runbook"
title: 裸 make 不编 kernel binary 且默认按 910b 编译
description: CANN 算子工程 cmake 默认 make 不编译 kernel binary，裸 make binary 默认按 ascend910b 编译；改 .h 后 make 全 Built 但 kernel .o 陈旧、ST 结果不变（假修复），包内 kernel 路径也是 910b——必须 build.sh --soc=<soc>（改头文件加 --make_clean）。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# 裸 make 不编 kernel binary 且默认按 910b 编译

## 现象

调试 Ascend950PR（arch35）自定义算子 kernel 时反复出现同一类诡异现象：**改了 kernel/tiling 源码（含 .h）后重建、重装、重跑 ST，测试结果完全不变**，像是修复无效。同一开发流水线上的两类典型观测：

场景一（裸 make/cmake 重建）：`cd build && cmake .. && make -j` 输出全部 `Built target`（cust_optiling、cust_op_proto、cust_opapi 等 host 目标），无任何 kernel 编译动作；`make package` 反而报 `file INSTALL cannot find ".../binary/ascend910b": No such file or directory`；`.run` 与 kernel `.o` 时间戳停在 11:23，而源文件是 18:00/18:04——**构建用了缓存，kernel 根本没重编**。`make binary` 则按 **ascend910b** 编译（目标 SoC 是 ascend950），还暴露出 kernel 头里的编译错。

场景二（改 kernel .h 后 make）：改完 kernel `.h` 后 `cmake .. && make -j` 显示 `[93%] Built target cust_op_proto` 等全部成功，重装重跑 ST 仍与改前逐条一致（34/45，同样 11 个失败）；`find build -name "*.o" -newer <kernel头文件>` 只命中 `autogen/op_proto.cc.o`——**kernel `.o` 未重编，ST 结论在陈旧 kernel 上得出**（据此判断"修复无效"是错误结论）。

配套观测（make help 与包内容）：删掉 build 目录重新 `cmake .. && make` 也一样——`make help` 里 kernel 编译目标全部带 **ascend910b** 前缀（`BenchNoHarnessMoeGatingTopKSoftmax_ascend910b`、`ascendc_kernels_ascendc_bin_ascend910b_gen_ops_config`、`ascendc_kernels_ops_info_gen_ascend910b`），而目标芯片是 ascend950；此时强行 `make binary package` 能打出 run 包，但包内 kernel 路径是错的芯片：`op_impl/ai_core/tbe/kernel/ascend910b/<op>/...o`——把这个包装到 950 环境，kernel 不会按预期更新/运行。

## 适用场景

- 工程形态：CANN 自定义算子工程（CMake + ccec kernel 编译 + CPack .run 包）的构建-验证循环（`npu_op_kernel_sources`/`npu_op_package` CMake 框架）。
- API 路径：顶层 CMakeLists（含 `set(ARCH35_COMPUTE_UNITS ascend950)` 与 kernel binary target 注释）、`build.sh --soc=<soc> [--make_clean]`、`make [binary|package]`、`cmake --build . --target all binary package`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35）+ CANN 9.1.0，同一流水线多次独立复现（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 算子构建系统行为，与算子无关）。
- 路线优先级：recommended_fastpath（统一走 build.sh --soc 是正确构建的唯一路径）。
- 明确不适用 / 未验证场景：`make binary` 在正确配置 soc 变量后的行为未验证；经 IDE/CI 封装好的构建入口（其内部已传对 soc）无此问题；其他 CANN 版本构建系统差异未验证。

## 判别方法

工程 CMakeLists.txt 自带警告注释（构建时 message 也会打印）：

```cmake
# IMPORTANT: cmake's default 'all' target does NOT build 'binary'!
# To build kernel binary, use one of:
#   1. bash build.sh --soc=<soc>           (Recommended)
#   2. cmake --build . --target all binary package
message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")
```

验证 kernel 是否真的重编，用时间戳对比探针（强烈建议每次"修复无效"时先跑）：

```bash
# 源码 vs 产物时间戳
stat -c '%Y %n' op_kernel/arch35/<op>.h op_host/arch35/<op>_tiling.cpp \
    build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o \
    build/custom_opp_ubuntu_x86_64.run
# 是否有比源码新的 .o（只看 op_proto 命中即危险）
find build -name "*.o" -newer op_kernel/arch35/<op>.h | head -5
# make help 的 kernel 目标前缀是什么芯片
cd build && make help | grep -i ascendc
```

判别要点：
- `make`/`make all` 输出里**没有任何 kernel（Bench.../ccec）编译行**、只有 host 目标 Built → kernel 未参与构建。
- 源码 mtime > kernel `.o`/`.run` mtime → 陈旧产物。
- `make binary` 报 ascend910b 路径（如 `binary/ascend910b/....json not generated!`）→ 目标 SoC 配置未进该入口，默认按 910b 编。

## 根因

CANN 算子工程的 kernel binary 是独立于默认 `all` 的构建目标；且 kernel 编译入口的 SoC 选择由 `build.sh --soc=<soc>` 注入，裸 `cmake ..` 配置出的 kernel 编译目标是 ascend910b 而非工程期望的 ascend950（与 `ARCH35_COMPUTE_UNITS ascend950` 的源配置脱节；工程 CMakeLists 里"未定义 ASCEND_COMPUTE_UNIT 时默认 ascend950"的逻辑实测未生效，原因未追查）；增量 make 对 kernel 头文件的依赖跟踪不完整，改 `.h` 后可能不触发 kernel `.o` 重编。三个因素叠加，导致"改码→make→重装→ST 不变"的假修复循环。

## 解决配方

统一走 build.sh 全量重建（多次实证有效）：

```bash
cd <operator_dir>
bash build.sh --soc=ascend950 --make_clean -j8   # 清理
bash build.sh --soc=ascend950 -j8                # 重建
cp build/custom_opp_ubuntu_x86_64.run /tmp/ && cd /tmp && bash custom_opp_ubuntu_x86_64.run --quiet --force
```

等价手工路径（须显式列出 binary/package 目标并确保配置期传对芯片）：

```bash
# cmake .. -DASCEND_COMPUTE_UNIT=ascend950 && cmake --build . --target all binary package
```

构建成功的判别（实测期望输出）：

```
[INFO] Compile for SoC: ascend950
[ascend950] Generating <OpName>_<hash> ... Done
[INFO] Kernel binary: <算子目录>/build/op_kernel/ascendc_kernels/binary/ascend950/<op>/<OpName>_<hash>.o
[INFO] Package: <算子目录>/build/custom_opp_ubuntu_x86_64.run
```

并且 run 包清单里 kernel 路径应为 `op_impl/ai_core/tbe/kernel/ascend950/<op>/...`（不是 ascend910b）。重装前先跑卸载脚本（`$ASCEND_TOOLKIT_HOME/opp/vendors/<vendor>/scripts/uninstall.sh --quiet`）避免旧 kernel 残留，安装后用 `md5sum` 对比 build 与安装目录的 kernel `.o` 确认一致。

错误用法对照（修复前）：

```bash
# 错误 1: make 默认 all，不编 kernel binary
cd <算子目录>/build && make clean && make -j$(nproc)

# 错误 2: 裸 cmake 重新配置后 make，kernel 目标按 ascend910b 生成
cd <算子目录> && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

# 错误 3: 强行 make binary package——能出包，但 kernel 是 ascend910b 的
cd <算子目录>/build && make binary package
```

## 设计决策

- 选择 build.sh 作为唯一构建入口而非手工 `cmake --build . --target all binary package`：build.sh 同时注入 --soc、清 CMakeCache、按 soc 重新配置并按正确顺序构建 `all binary package`，一条命令闭合"配置+kernel+打包"；手工路径要求同时记住 target 名和 `-DASCEND_COMPUTE_UNIT`，漏一个就复现本坑。
- 判定修复是否生效以 **kernel .o 时间戳 + ST 结果变化** 双信号为准，不以 make 退出码为准。
- 判断"kernel 是否真的重编"不要看 `.o` 文件名（文件名 hash 由 TilingKey 决定，源码变了文件名也不变），要看 `build/op_kernel/ascendc_kernels/binary/<soc>/<op>/*.o` 的 mtime/md5，以及安装目录对应文件是否一致。

## 预防

- 算子构建一律 `bash build.sh --soc=<目标soc>`；改 kernel 头文件后加 `--make_clean`（本环境增量依赖不可信）。
- 每轮"改码→验证"前跑一次时间戳探针（见判别方法），确认 kernel `.o` 晚于源码。
- ST/UT 结果与改前完全一致（连失败集合都逐条相同）时，第一怀疑对象是**陈旧产物**而非修复思路错误。
- 重装前先跑卸载脚本，再 `--force` 安装，最后 md5sum 双端对比。

## 验证信号

问题与验证来自同一条构建-联调流水线的多次独立复现（先后用裸 make、裸 cmake、make binary package、build.sh 四种方式重建并对比产物）：

- 裸 `make clean && make`：输出仅 host 库目标（cust_opapi/cust_op_proto/cust_optiling 100% Built），`build/op_kernel/ascendc_kernels/binary/ascend950/` 下无新 `.o`，run 包不存在（来源平台：950；证据等级：tool_observed）。
- 裸 `cmake ..` 后 `make help`：kernel 目标为 `BenchNoHarnessMoeGatingTopKSoftmax_ascend910b`、`ascendc_kernels_ascendc_bin_ascend910b_gen_ops_config`、`ascendc_kernels_ops_info_gen_ascend910b`（来源平台：950；证据等级：tool_observed）。
- `make binary package` 产物清单：kernel 路径 `op_impl/ai_core/tbe/kernel/ascend910b/<op>/<OpName>_<hash>.o`，config 同为 ascend910b（来源平台：950；证据等级：tool_observed）。
- 场景一：源码 18:00/18:04 vs `.o`/`.run` 11:23（make 全 Built 但产物陈旧）；`make package` 报 `cannot find .../binary/ascend910b`；`make binary` 按 ascend910b 编译并暴露 kernel 编译错。`build.sh --soc=ascend950 --make_clean` + 重建后 kernel `.o` 时间戳更新（11:23 → 18:39）、`.run` 重新生成，ST 由 8/45 → 29/45——证明前几轮"无效修复"其实是没编进 kernel（来源平台：950；证据等级：tool_observed）。
- 场景二：改 kernel `.h` 后 `cmake .. && make -j` 仅 `autogen/op_proto.cc.o` 比源码新（`find -newer` 实测），kernel `.o` 未重编，ST 仍 34/45 与改前逐条一致（来源平台：950；证据等级：tool_observed）。
- CMakeLists 原文注释与构建期 WARNING：`cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>`（来源平台：950 工程；证据等级：tool_observed）。
- `bash build.sh --soc=ascend950 -j8`：输出 `[INFO] Compile for SoC: ascend950`，包内 kernel 路径全部变为 `tbe/kernel/ascend950/...`，生成 `custom_opp_ubuntu_x86_64.run`，`[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `--make_clean` 每轮全量重建代价较高；增量构建何时可信（如仅改 op_host）未系统验证，保守做法是 kernel 侧任何改动都 clean。
- `make binary` 的默认 soc（ascend910b）是本环境观察值；其他环境默认值可能不同，但"build.sh --soc 显式指定"的处方不受影响。
- "裸 cmake 默认 910b"是实测现象，其配置来源（环境变量/框架内置默认、为何工程 CMakeLists 的 ascend950 默认值未生效）未定位。
- 本卡只覆盖构建入口选择；构建成功后 kernel 是否被正确加载还取决于卸载/安装顺序（旧 vendor 目录残留会掩盖新 kernel）。

**未决项**：

- 裸 cmake 配置下 kernel 目标按 ascend910b 生成而工程 CMakeLists 的 ascend950 默认值未生效的确切机制（ASCEND_COMPUTE_UNIT 环境变量？框架默认？）未验证——复现并追到配置层是接续点。
- 增量 make 对 kernel 头依赖缺失的具体范围（哪些头会/不会触发重编）未逐一验证；需要时可用 touch 单个头文件做最小实验。
## 坏实践

- 以为 `make`（默认 all）"Built target xxx 100%"就是全量编译——kernel binary 目标根本没进 all。
- 不传 `--soc` 直接 `cmake .. && make binary package`，把 ascend910b 的包装到 950 环境。
- 用 `cmake .. && make -j` 作为 kernel 改动的验证构建——默认目标不含 kernel binary，等于没编。
- 用 `.o` 文件名（`<Op>_<hash>.o`）判断 kernel 是否更新——文件名 hash 由 TilingKey 决定，与源码内容无关。
- 见 ST 结果不变就否定修复方案并继续换方案——本轨迹差点据此误判多项修复无效，实为 kernel 未重编。

## 误诊

- 曾把"改了 kernel 头文件、重编译、重装后测试结果与之前完全相同（同样的 34/45 与逐位相同的失败输出）"误判为"编译缓存导致 kernel 没有更新"，并去删 `~/.atc_data/kernel_cache`、找 `kernel_bank`/`.compile_cache` 等目录（均不存在）。真实原因有两层：一是本卡描述的构建入口问题（kernel 根本没编），二是修复本身对该批失败无效（改用 build.sh 正确重建后结果仍相同才证明是后者）。
- 场景一最初以为 `.run` 里"应该包含新编译的 kernel"（`.o` 存在于 ascend950 目录），未先核对时间戳，多走了一轮安装-测试；以 `ls -la` + `date` 对比后才定位缓存问题。
