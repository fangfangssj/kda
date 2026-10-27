---
type: "Runbook"
title: 多 vendor 环境 find 盲选错 opapi 库致符号错
description: "多算子包共存时 run.sh 用 find …|head -1 盲选 opp/vendors 下的 op_api/lib，拿到别的算子的 libcust_opapi.so，运行报 undefined symbol: aclnnXxx；改为优先精确匹配 ${OP_NAME}_custom/op_api/lib 后端到端 PASS。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# 多 vendor 环境 find 盲选错 opapi 库致符号错

## 现象

自定义 aclnn 算子（示例/测试工程，Ascend950PR / arch35 / DAV_3510，CANN 9.1.0，算子包已安装为 `<op>_custom` vendor）的 `run.sh`（负责 cmake + make + 运行示例二进制）执行 `bash run.sh`：CMake 配置、编译全部成功（`[100%] Built target test_aclnn_<op>`），但运行示例时立刻报：

```
./test_aclnn_bench_no_harness_foreach_addcdiv_scalar: symbol lookup error: ./test_aclnn_bench_no_harness_foreach_addcdiv_scalar: undefined symbol: aclnnBenchNoHarnessForeachAddcdivScalar
```

脚本自身的环境日志暴露了原因——它把错误的算子包目录加进了 `LD_LIBRARY_PATH`：

```
VENDOR_LIB_DIR: <cann 安装根>/opp/vendors/custom_ops/op_api/lib
```

拿到的是另一个算子包（`custom_ops`）的 `op_api/lib`，而本算子的 aclnn 符号在 `libcust_opapi.so` 位于 `opp/vendors/bench_no_harness_foreach_addcdiv_scalar_custom/op_api/lib`。本机 `opp/vendors` 下共存十余个算子包（AddCustom、SoftplusV2Grad、cummin_custom、bench_no_harness_* 系列等），`find | head -1` 的返回顺序取决于目录遍历顺序（文件系统创建顺序）。另一独立算子的示例在 8 个 vendor 共存环境下命中同样故障（错误 vendor 同为 `custom_ops`），`nm -D` 确认符号齐全于本算子 vendor 库。

## 适用场景

- 工程形态：自定义算子示例/测试的运行脚本（run.sh 类），构建产物 + 运行时符号解析。
- API 路径：`opp/vendors/<vendor>/op_api/lib/libcust_opapi.so`、`LD_LIBRARY_PATH`、CMake `CMAKE_SKIP_RPATH`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950，arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（CANN 自定义算子包安装布局通用行为，跨算子成立；两个独立算子/两台会话复现）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：单算子包的干净环境不会触发；aarch64 交叉环境未验证；若不用 `CMAKE_SKIP_RPATH=TRUE` 而保留二进制内嵌 rpath，理论上对 `LD_LIBRARY_PATH` 依赖减弱（未单独验证）。

## 根因

run.sh 用 `find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" -name "lib" -type d | head -1` 取第一个 vendor 的 `op_api/lib` 加入 `LD_LIBRARY_PATH`；多算子包共存时第一个命中目录不是本算子的 vendor 目录，其 `libcust_opapi.so` 不含本算子的 `aclnn<Op>` 符号；叠加 run.sh 以 `CMAKE_SKIP_RPATH=TRUE` 配置（二进制无内嵌 rpath，CMakeLists 里 `-Wl,-rpath,${CUSTOM_OP_LIB_DIR}` 被禁用），运行期符号解析完全依赖这份错误的 `LD_LIBRARY_PATH`。

判别探针（确认符号确实存在于本算子 vendor 库）：

```bash
find "$ASCEND_HOME_PATH/opp/vendors" -name "libcust_opapi.so" -type f
nm -D "$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so" | grep -i "<Op>"
```

实测输出（本算子 vendor 库中两个入口符号齐全）：

```
0000000000006c00 T aclnnBenchNoHarnessExp
00000000000063df T aclnnBenchNoHarnessExpGetWorkspaceSize
```

## 解决配方

修复前（run.sh 环境变量段，盲选第一个 vendor）：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

VENDOR_LIB_DIR=$(find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" -name "lib" -type d 2>/dev/null | grep -v "^${ASCEND_HOME_PATH}/opp/vendors/vendors" | head -1)
if [ -n "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
fi
```

修复后（优先精确匹配本算子 vendor 目录，find 仅作兜底并显式标注）：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

OP_VENDOR_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/${OP_NAME}_custom/op_api/lib"
if [ -d "${OP_VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${OP_VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${OP_VENDOR_LIB_DIR}"
else
    VENDOR_LIB_DIR=$(find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" -name "lib" -type d 2>/dev/null | grep -v "^${ASCEND_HOME_PATH}/opp/vendors/vendors" | head -1)
    if [ -n "${VENDOR_LIB_DIR}" ]; then
        export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
        echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR} (fallback)"
    fi
fi
```

（`OP_NAME` 为脚本中已有的算子名变量；自定义算子包安装目录约定为 `<op_name>_custom`。）

另一会话的等价修复（直接硬编码本算子 vendor 路径，与 CMakeLists `find_library` 用的 vendor 名保持一致）：

```bash
VENDOR_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/bench_no_harness_exp_custom/op_api/lib"
if [ -d "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
fi
```

## 设计决策

- 选择"精确路径优先 + find 兜底"而不是完全删掉 find：兜底分支覆盖算子包以非默认 `<op_name>_custom` 名字安装的场景，且 `(fallback)` 标注让误配可见。
- 未选择"去掉 CMAKE_SKIP_RPATH 让 rpath 生效"作为主修复：CMakeLists 已带 `-Wl,-rpath,${CUSTOM_OP_LIB_DIR}` 链接选项，但 run.sh 显式传了 `CMAKE_SKIP_RPATH=TRUE`（参考模板有意用该组合便于统一控制运行环境）；脚本层修复影响面更小、语义更直白。
- 不用"遍历所有 vendor 目录都加进 LD_LIBRARY_PATH"：多个同名 `libcust_opapi.so` 一起进路径时加载顺序仍不确定，只是把不确定性往后挪；精确单 vendor 才是确定性解。
- 当前方案属 `recommended_fastpath`，无运行成本差异。

## 预防

- 自定义算子的示例/测试脚本一律按 `<op_name>_custom` 精确拼接 vendor 库目录，禁止 `find … | head -1` 盲选。
- 若必须盲选（通用脚手架），至少在日志里打印选中目录并与算子名比对。
- 判别口诀：`undefined symbol: aclnn<OpName>` + 多 vendor 共存环境 → 先看脚本把哪个 `op_api/lib` 放进了 `LD_LIBRARY_PATH`。
- 安装器在安装完成提示里会给出正确的 `LD_LIBRARY_PATH`（`export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib/:${LD_LIBRARY_PATH}`），可直接作为脚本取值依据。

## 验证信号

问题与验证来自两个独立算子的示例端到端联调会话：

- 修复前：`bash run.sh` 编译成功后运行报 `symbol lookup error … undefined symbol: aclnnBenchNoHarnessForeachAddcdivScalar`，且日志显示 `VENDOR_LIB_DIR: …/opp/vendors/custom_ops/op_api/lib`（来源平台：950；证据等级：tool_observed）。
- 修复前（另一算子）：`./test_aclnn_<op>: symbol lookup error: undefined symbol: aclnn<Op>`；`find` 列出 8 个 vendor 的 `libcust_opapi.so`；`nm -D` 在 `<op>_custom` 库中命中 `aclnn<Op>` 与 `aclnn<Op>GetWorkspaceSize` 两个 T 符号（来源平台：950；证据等级：tool_observed）。
- 修复后：`rm -rf build-eager build-geir && bash run.sh` 重新配置编译运行，`VENDOR_LIB_DIR` 正确指向 `…/opp/vendors/bench_no_harness_foreach_addcdiv_scalar_custom/op_api/lib`，结果正确（`Tensor[0][0] = 3.500000`、`Tensor[1][0] = 18.500000`），脚本输出 `执行结果: PASS`（来源平台：950；证据等级：tool_observed）。
- 修复后（另一算子）：`VENDOR_LIB_DIR` 指向本算子 vendor，示例执行结果 PASS，exp 类数值正确（自然指数与自定义底数分支的输出均符合预期）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论基于 CANN 9.1.0 单机多 vendor 共存环境；vendors 目录遍历顺序依文件系统而异，`custom_ops` 恰好排前只是本机现象，其他机器可能命中别的错误目录（失败模式相同：undefined symbol aclnnXxx）。
- 未验证 aarch64 交叉编译部署与容器内安装布局（未验证，推断可迁移）。
- 该修复不解决"算子包根本未安装"的场景（精确目录不存在时走 fallback，仍可能失败——此时应报"未找到算子包"类错误而非符号错误）。
- 未验证去掉 `CMAKE_SKIP_RPATH=TRUE`（纯 rpath 方案）在该环境是否同样可用，可作为备选修复验证。

**未决项**：

- 无（根因与修复在两个会话内闭环）；可选增强：run.sh 在精确目录不存在时直接 fail-fast 并提示"算子包未安装"，而不是静默走 fallback。
- 未追查 `custom_ops` vendor 的 `libcust_opapi.so` 为何排序在前（文件系统创建顺序），属于环境事实，不影响结论。
## 坏实践

- `find ${ASCEND_HOME_PATH}/opp/vendors -path "*/op_api/lib" | head -1` 类盲选写法在多算子包环境必然埋雷。
- 同时使用 `CMAKE_SKIP_RPATH=TRUE` 又不精确设置 `LD_LIBRARY_PATH`，把二进制符号解析交给运气。
- 只看编译链接通过就认定运行环境正确：链接期 `-lcust_opapi` 命中的库与运行期 `LD_LIBRARY_PATH` 加载的库可以不是同一份。
- 报 undefined symbol 后先去改代码/改头文件：应先用 `find` + `nm -D` 确认符号在哪个 `.so` 里，再查加载的是哪个。

## 误诊

- 无重大误诊；本故障的定位信息（错误 VENDOR_LIB_DIR 日志行）在失败输出中直接可见。注意不要把该 `symbol lookup error` 与"算子包未安装/未编译"或安装期共享库校验误报（见 [l0op::Contiguous 未定义符号致安装校验失败](libcust_opapi_l0op_contiguous_undefined_symbol.md)）混淆——本例算子包完好，只是库目录指错了。

## 后续增量（2026-09-17，并卡合并：config.ini 证据、三目录依赖面、链接期形态与 RUNPATH 优先级）

#### 并入自 symbol_lookup_error_wrong_vendor_lib_dir（GRU 实例、load_priority 证据与字母序命中形态）

- **命中顺序的另一种形态**：GRU 算子示例（CANN 9.2.0）的 `find ... | head -1` 按字母序命中 `BenchHarnessApplyRotaryPosEmb`（大写 B 排前）——多 vendor 环境里 head -1 的结果由目录遍历/字母序决定，哪个 vendor 排前是环境事实，失败模式相同（undefined symbol aclnnXxx）。
- **config.ini 旁证**：`opp/vendors/config.ini` 的 `load_priority=bench_no_harness_gru_custom,BenchHarnessApplyRotaryPosEmb,BenchHarnessApplyAdamW,custom_ops` 列出全部共存 vendor——联调环境装多个 vendor 时先 `cat config.ini` 明确"哪些库会同名共存"。
- **修复形态（固定本算子 vendor 名 + 未找到告警兜底）**：

```bash
VENDOR_NAME="bench_no_harness_gru_custom"
VENDOR_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_api/lib"
if [ -d "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
else
    echo "警告: 未找到 vendor 库目录 ${VENDOR_LIB_DIR}"
fi
```

- 设计取舍（并入）：选"硬编码 vendor 名"而不是"把所有 vendor lib 目录都加进 LD_LIBRARY_PATH"——多库并存时符号解析顺序不可控，仍可能先命中别人的同名 `libcust_opapi.so`；点名本算子 vendor 是唯一确定的绑定。保留 `nm -D` 探针作为第一步：symbol lookup error 有两种可能（库没导出 / 加载了错库），先证明"符号在正确库里是 T（导出）"，再查加载路径，避免误修算子包。
- 坏实践补充（并入）：见到 undefined symbol 就重装算子包——先 `nm -D` 分清"没导出"还是"没加载对"。

验证信号（并入，GRU 算子 aclnn eager 示例，Ascend950PR + CANN 9.2.0）：

- 修复前：运行报 `symbol lookup error: undefined symbol: aclnnBenchNoHarnessGruGetWorkspaceSize`，run.sh 日志显示 VENDOR_LIB_DIR 指向 BenchHarnessApplyRotaryPosEmb（来源平台：950；证据等级：tool_observed）。
- `nm -D` 确认 `aclnnBenchNoHarnessGru` / `aclnnBenchNoHarnessGruGetWorkspaceSize` 均在 `bench_no_harness_gru_custom` 的 libcust_opapi.so 中以 T 导出（来源平台：950；证据等级：tool_observed）。
- 修复后同一示例 PASS：y/hn 输出数值正常（S=10 B=4 inputSize=32 hiddenSize=16），执行结果 PASS（来源平台：950；证据等级：tool_observed）。

#### 并入自 multi_vendor_lib_dir_find_head1_mispick（GE 图模式三目录依赖面与 ldd 探针）

- **三目录依赖面（图模式完整运行期依赖）**：`$ASCEND_HOME_PATH/opp/vendors/<vendor_name>/` 下除 `op_api/lib`（aclnn 接口符号所在 libcust_opapi.so）外，GE 图模式运行还需要 `op_proto/lib/linux/x86_64`（REG_OP 注册库 libcust_opsproto_rt2.0.so）与 `op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64`（tiling 库）——只导出 op_api 时图模式找不到算子注册。三个目录都应按本算子 vendor 名确定性拼接（架构段按主机实测 x86_64，aarch64 未验证）：

```bash
VENDOR_PROTO_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_proto/lib/linux/x86_64"
VENDOR_TILING_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64"
```

- **判别方法补充**：`ldd <binary> | grep cust_opapi` 看进程实际解析到哪个库；复核本算子库存在性 `ls "${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/lib/"`。
- 修复后先 `bash run.sh --clean` 清掉旧构建产物再重跑（旧 build 目录里链接的是错误路径时代的产物）。
- 失效边界补充（并入）：`VENDOR_NAME` 与包名硬耦合，换算子/改 vendor 名时须同步修改（通用 harness 应参数化）；依赖 vendor 目录名 = `<op>_custom` 的安装约定；`find` 输出顺序选中无关 vendor 的确切机制（目录字典序 vs inode 顺序）未深究，不影响结论与修复。

验证信号（并入，dynamic_quant 与 sigmoid 两个独立算子工程，Ascend950PR + CANN 9.1.0）：

- 修复前：`VENDOR_LIB_DIR` 分别错指 `.../opp/vendors/BenchGcd/op_api/lib` 与 `.../opp/vendors/custom_ops/op_api/lib`，运行分别报 `symbol lookup error: undefined symbol: aclnnBenchNoHarnessDynamicQuantGetWorkspaceSize` 与 `undefined symbol: aclnnBenchNoHarnessSigmoidGetWorkspaceSize`（编译全绿、头文件目录正确，仅运行期符号解析失败）（来源平台：950；证据等级：tool_observed）。
- 修复后：`VENDOR_LIB_DIR` 按名命中 `.../vendors/<本算子>_custom/op_api/lib`，`bash run.sh --eager` 全流程执行结果 PASS（dynamic_quant：y int8 与 scale fp32 输出正常；sigmoid：aclnn 示例编译运行通过）（来源平台：950；证据等级：tool_observed）。
- 同一脚本图模式：`VENDOR_PROTO_DIR` 按名命中 `op_proto/lib/linux/x86_64`，`bash run.sh --graph` 执行结果 PASS（来源平台：950；证据等级：tool_observed）。

#### 并入自 multi_vendor_libcust_opapi_misdetection（链接期 undefined reference 形态、CMake glob 误选与 RUNPATH 优先级）

- **链接期形态（第三种表现）**：ST 工程 Real 模式链接期即失败——`/usr/bin/ld: ... undefined reference to 'aclnnBenchNoHarnessGcdGetWorkspaceSize' / 'aclnnBenchNoHarnessGcd'`（int/int64 模板实例共 6 处），`collect2: error: ld returned 1 exit status`。成因是 CMake 侧 `file(GLOB .../vendors/*/op_api/lib/libcust_opapi.so)` 取第一个候选（字母序 adaptive_avg_pool_3d 排前）做链接库。
- **ELF 优先级规则（运行期形态的机制补全）**：`LD_LIBRARY_PATH` 的优先级**高于**可执行文件自身的 RUNPATH——即使二进制 RUNPATH 正确指向本算子 vendor，脚本注入的错误路径仍会遮蔽正确库（`readelf -d` 看 RUNPATH、带/不带脚本环境各跑一次 `ldd` 对比解析目录即可证明）。叠加因素：`-DCMAKE_SKIP_RPATH=TRUE` 会把 CMakeLists 中指向正确库目录的 rpath 一并禁用。
- **修复（include 目录推导同 vendor 库，glob 降级为兜底）**：CMake 配置阶段已用头文件把 vendor 唯一化，同 vendor 的 lib 路径只需一次推导：

```cmake
    set(CUSTOM_OP_LIBRARY "")
    if(CUSTOM_OP_INCLUDE_DIR)
        # Derive vendor op_api dir: .../vendors/<vendor>/op_api/include -> .../vendors/<vendor>/op_api
        get_filename_component(_VENDOR_OP_API_DIR "${CUSTOM_OP_INCLUDE_DIR}" DIRECTORY)
        set(_CANDIDATE_LIB "${_VENDOR_OP_API_DIR}/lib/libcust_opapi.so")
        if(EXISTS "${_CANDIDATE_LIB}")
            set(CUSTOM_OP_LIBRARY "${_CANDIDATE_LIB}")
        endif()
    endif()
```

  并对可执行文件设 `BUILD_RPATH`/`INSTALL_RPATH` 指向该 lib 目录（双保险）；run.sh 侧也可按"哪个 vendor 含本算子头文件 `aclnn_<op>.h`"循环定位。统一原则：**include、link、rpath、LD_LIBRARY_PATH 四处绑定同一 vendor**。
- **探针补充**：`LD_DEBUG=libs ./binary 2>&1 | grep -A2 "find library=libcust_opapi.so"` 的 `trying file=` 行逐列出搜索顺序，第一个命中路径即实际加载库（加载源判定的完整方法见 [多 vendor 同名 libcust_opapi 加载源判定卡](ld_debug_libcust_opapi_same_soname_multi_vendor.md)）。
- 坏实践补充（并入）：链接与运行环境用两套不同的定位逻辑（CMake 一套、run.sh 一套），链接对了运行又加载错库；以为设了 rpath 就万事大吉——`LD_LIBRARY_PATH` 优先级更高；只看 `nm -D` 确认符号存在就断定库没问题，不去核对运行时实际加载路径。
- 失效边界补充（并入）：若两个 vendor 都含同名算子头文件（重名算子包），按循环取第一个命中者仍可能选错——彻底方案是按 vendor 包名直接拼路径；若错载的库恰好也导出同名符号（同名算子多版本并存），不会报 undefined symbol 而是行为异常，本判别法不覆盖。

验证信号（并入，三个独立算子工程（resize_bilinear 类、gcd、gather），Ascend950PR + CANN 9.1.0，8+ vendor 共存）：

- 链接期触发：ST 工程链接报 `undefined reference to aclnn<Op>GetWorkspaceSize / aclnn<Op>`（6 处），`ld returned 1 exit status`；环境输出显示 `LD_LIBRARY_PATH` 指向其他算子（adaptive_avg_pool_3d）vendor 的 op_api/lib（来源平台：950；证据等级：tool_observed）。
- 探针：`nm -D` 显示本算子 vendor 库导出 `aclnn<Op>`（T, 0x80f8）与 `aclnn<Op>GetWorkspaceSize`（T, 0x77d8），被命中的其他 vendor 同名库无这些符号；`readelf -d` 显示二进制 RUNPATH 正确；不带脚本环境 ldd 解析到正确库、带 run.sh 环境 ldd 解析到错误 vendor 的同名库（来源平台：950；证据等级：tool_observed）。
- 修复后（ST 工程）：清理重建，CMake 输出 `Found custom operator library: .../<本算子>_custom/op_api/lib/libcust_opapi.so`，`[100%] Built target`，24 条用例在 NPU 上实际执行（来源平台：950；证据等级：tool_observed）。
- 修复后（示例脚本）：`VENDOR_LIB_DIR` 输出变为本算子目录，示例运行输出全部正确结果（如 8 组 gcd/gather 逐值正确），`执行结果: PASS`（两个独立算子工程各验证一次）（来源平台：950；证据等级：tool_observed）。