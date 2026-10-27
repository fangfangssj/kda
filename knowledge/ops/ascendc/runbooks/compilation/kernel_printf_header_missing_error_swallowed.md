---
type: "Runbook"
title: kernel_printf.h 不存在且编译错误被构建吞掉
description: "AscendC kernel 加 #include \"kernel_printf.h\" 后编译 fatal error 被 build.sh 吞掉，.run 仍打包安装旧二进制，printf 调试无输出；CANN 9.1.0 无该头文件，printf 为内建。复现增量：AscendC::Printf 混大小写形态同样编译失败（no member named 'Printf'），正确为小写 printf（__asc_aicore::printf）；浮点格式符仅支持 %f——%.6f/%.2e 从格式符处截断为 \"%.\"（实测闭合，%d/%lld/%f 可用）；输出经 RUNTIME ParsePrintToLog 路由进 plog，ASCEND_SLOG_PRINT_TO_STDOUT=1 + ASCEND_GLOBAL_LOG_LEVEL=0 时 stdout 可见（带 [AIV/AIC Block i/n] 前缀）。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# kernel_printf.h 不存在且编译错误被构建吞掉

## 现象

在 Ascend950PR（CANN 9.1.0）上调试自定义算子（RoiAlign 类）的精度失败用例时，按调试参考在 kernel 头文件 `op_kernel/arch35/bench_no_harness_roi_align.h` 顶部加入 `#include "kernel_printf.h"`，并在 `ProcessOneRoi` 中插入 `printf` 调试输出，随后执行惯常构建安装链：`bash build.sh --soc=ascend950 -j8 2>&1 | tail -20` → `bash build/custom_opp_ubuntu_x86_64.run --force`。构建尾部正常显示 `[INFO] Build completed successfully!` 与 kernel 二进制/包路径，安装 SUCCESS。但以 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 运行 `bash run.sh --case L1_007` 后，`grep -iE "DBG|debug"` 没有任何 kernel printf 输出——调试语句像不存在一样。再试 `ASCEND_SLOG_PATH` 自定义日志目录也找不到新日志。最终对同一构建命令加 `| grep -iE "printf|warning|error"` 重新过滤，才在日志中段暴露真实错误：`fatal error: 'kernel_printf.h' file not found`（TBE 编译异常，`Exception occured during compile phases of CompileStage.PRECOMPILE`）——kernel 编译实际失败，测试一直在跑旧二进制。

## 适用场景

- 工程形态：自定义算子 AscendC kernel 的 printf 调试（ops-registry-invoke 算子仓 build.sh → custom_opp .run → ST/UT 的开发链路）。
- API 路径：kernel 侧 `printf`（`__aicore__` 函数内）；`build.sh --soc=<soc>` 构建脚本；`custom_opp_ubuntu_x86_64.run --force` 安装。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——构建/调试工具链行为与 kernel printf 可用性，不绑定具体算子。
- 路线优先级：recommended_fastpath（先确认构建真实成功再归因调试输出问题）。
- 明确不适用 / 未验证场景：
  - printf 内容的路由（stdout vs plog 文件）与 `ASCEND_SLOG_PRINT_TO_STDOUT` 是否能让 kernel printf 上屏——本会话未在"构建真实成功"的前提下验证（修复后的重编译验证未完成，见失效边界与遗留问题）。
  - 其他 CANN 版本是否提供 `kernel_printf.h` 未验证。

## 判别方法

触发条件：kernel 里加了 printf（或任何 kernel 源码改动）后，构建"成功"、安装 SUCCESS，但运行行为/调试输出与改动前完全一致。

可执行探针（对同一构建命令按错误关键字过滤，而不是只看尾部）：

```bash
cd operators/<op_dir> && bash build.sh --soc=ascend950 -j8 2>&1 | grep -iE "printf|warning|error" | head -20
```

再核对头文件是否真的存在（在 CANN 安装树内全量查找）：

```bash
find $ASCEND_HOME_PATH -name "kernel_printf.h" 2>/dev/null
# 以及更宽的 printf 相关头文件查找：
find $ASCEND_HOME_PATH -name "*printf*" -path "*kernel*" 2>/dev/null
```

参照证据（CANN 内置 AscendC 算子的用法）：

```bash
grep -rn "ASCENDC_PRINTF\|printf(" $ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/sinkhorn/sinkhorn.h
# 输出：宏定义直接调用 printf(__VA_ARGS__)，无任何 kernel_printf.h include
cat $ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/common/atvoss/util/log.h
# 输出：内置算子公共日志头，直接用 printf，注释标明"内核函数不支持'%e'"
```

原始观测：

- `bash build.sh ... | tail -20` 尾部：`[INFO] Build completed successfully!` + kernel 二进制路径 + `CPack: - package: ... .run generated`（看起来成功）。
- 同一构建 `| grep -iE "printf|warning|error"`：`fatal error: 'kernel_printf.h' file not found`、`[ERROR] TBE ... Exception occured, An error occurred during compile phases of CompileStage.PRECOMPILE`（实际 kernel 编译失败）。
- `find $ASCEND_HOME_PATH -name "kernel_printf.h"`：无结果（该头文件在 CANN 9.1.0 中不存在）。
- 内置算子 `sinkhorn.h`/`atvoss/util/log.h`：直接调用 `printf`，不带 kernel_printf.h；log.h 注释标明内核函数不支持 `%e` 格式。

解释（interpretation）：`build.sh` 在 kernel 编译失败时不会让整体构建失败——`tail` 尾部只见 CPack 打包与 "[INFO] Build completed successfully!"，.run 照常生成并可用旧 kernel 二进制安装 SUCCESS；测试因此运行的是改动前的旧二进制，printf 自然不存在。`kernel_printf.h` 在 CANN 9.1.0 安装树中不存在（旧资料引用的该头文件与当前版本不符）；kernel 侧 printf 是内建能力，内置 AscendC 算子均直接调用且无对应 include，使用 `%e` 格式不被支持（应改用 `%f`）。修复方向：删除 `#include "kernel_printf.h"`、格式串避开 `%e`，并以错误关键字过滤的方式确认构建真实成功。

## 设计决策

- 判别顺序为什么是"先验构建、再查路由"：printf 不可见有两种成因（编译失败跑旧二进制 vs 输出路由问题）；先用 `grep -iE error` 过滤构建日志可零成本区分二者，避免在日志路由（SLOG 环境变量、plog 目录权限）上空耗——本会话先在 `ASCEND_SLOG_PRINT_TO_STDOUT`/`ASCEND_SLOG_PATH`/plog 目录权限上排查多轮无果，回头验构建才命中真因。
- 当前方案路线档位：recommended_fastpath（每次 kernel 改动后以错误关键字过滤构建输出，作为固定验证步骤）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0。
- 相关文件：`op_kernel/arch35/bench_no_harness_roi_align.h`（printf 调试插入点与 `#include "kernel_printf.h"` 引入点）；`build.sh`；`build/custom_opp_ubuntu_x86_64.run`；`tests/st/run.sh`。
- 参照源（tool_observed 读取）：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/sinkhorn/sinkhorn.h`、`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/common/atvoss/util/log.h`。
- 默认 plog 目录 `/tmp/plog` 属于其他用户，当前用户不可写；`ASCEND_SLOG_PATH` 可重定向。

## 预防

- kernel 源码改动后，构建验证一律用 `bash build.sh --soc=<soc> -j8 2>&1 | grep -iE "error|fatal"` 这类错误关键字过滤，不要只 `tail` 尾部——"Build completed successfully!"不代表 kernel 编译成功。
- 安装后可用 kernel 二进制的修改时间戳（`stat -c '%y'` 对比安装树内 .o/so 与源文件时间）快速确认新二进制确实落地。
- 在 kernel 里用 printf 前，先 `find $ASCEND_HOME_PATH -name "kernel_printf.h"` 确认头文件是否存在；CANN 9.1.0 中直接 `printf` 即可，不要 include 不存在的头。
- kernel printf 格式串避免 `%e`，用 `%f` 输出浮点。

## 验证信号

以下信号来自该算子精度失败用例调试会话：插入 printf 后构建安装、运行无输出，再以错误关键字过滤复跑构建定位。

- 构建尾部（tail -20）：`[INFO] Build completed successfully!`、`CPack: - package: .../custom_opp_ubuntu_x86_64.run generated`（来源平台：950；证据等级：tool_observed）。
- 安装：`bash custom_opp_ubuntu_x86_64.run --force` → `SUCCESS`（来源平台：950；证据等级：tool_observed）。
- 运行 `run.sh --case L1_007`（ASCEND_SLOG_PRINT_TO_STDOUT=1）后 `grep -iE "DBG|debug"`：无输出（来源平台：950；证据等级：tool_observed）。
- 同一构建以 `grep -iE "printf|warning|error"` 复跑：`bench_no_harness_roi_align.h:32:10: fatal error: 'kernel_printf.h' file not found` + `[ERROR] TBE ... An error occurred during compile phases of CompileStage.PRECOMPILE`（来源平台：950；证据等级：tool_observed）。
- `find $ASCEND_HOME_PATH -name "kernel_printf.h"`：无结果；`find ... -name "*printf*" -path "*kernel*"`：仅 `aicpu_api.h` 等无关命中（来源平台：950；证据等级：tool_observed）。
- 内置算子参照：`sinkhorn.h` 定义 `printf("[%d] " fmt "\n", blockIdx, ##__VA_ARGS__)` 类宏直接调用 printf；`atvoss/util/log.h` 注释"内核函数不支持'%e'"（来源平台：950；证据等级：tool_observed）。
- 修复动作已落地：删除 `#include "kernel_printf.h"` 的 edit 成功应用（该修复的重编译与 printf 上屏验证未在本段轨迹内完成，见失效边界）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "build.sh 吞掉 kernel 编译失败"是 CANN 9.1.0 自定义算子构建脚本（CPack 打包流程）的观测行为，构建脚本内部错误处理逻辑未审计；其他版本/其他构建入口（CMake 直编）行为未验证。
- printf 输出的路由方式（stdout / plog 文件 / 环境变量开关）未在"构建真实成功"前提下验证——本卡的结论止于"编译失败导致 printf 不存在"，不含"printf 正确出现在何处"。
- `%e` 不支持来自 CANN 内置算子源码注释（artifact 证据），未在本会话实测复现。
- 仅 Ascend950PR + CANN 9.1.0 实测；头文件存在性与 printf 内建语义跨版本可能变化（未验证，推断需逐版本确认）。

**未决项**：

- 删除 `kernel_printf.h` include 后的重编译、以及 printf 在 ST 运行中的实际上屏路径（stdout 还是 plog、需要哪些环境变量）未完成验证——接续点：重建 → 安装 → 复跑失败用例确认 DBG 输出可见。
- `build.sh` 对 kernel 编译失败不失败退出的确切机制（脚本哪一层吞了错误码）未审计；若需根治应修构建脚本而非只靠 grep 过滤（未尝试）。
- kernel printf 相对性能/正确性的影响（调试后必须移除）未在本段轨迹验证。
## 坏实践

- 用 `tail -20` / 只看退出码判断自定义算子构建成败——kernel 编译错误被中段日志吞掉，打包与"成功"提示照常出现。
- 调试输出不生效时先怀疑日志路由（环境变量/日志目录权限）而不先复核构建——本会话因此多走数轮弯路。
- 按旧版本资料直接 `#include "kernel_printf.h"`——当前版本无此头文件。

## 误诊

- 初判"printf 输出进了 Ascend 日志系统、需要设置 `ASCEND_SLOG_PRINT_TO_STDOUT=1`/自定义 `ASCEND_SLOG_PATH` 才能看见"——两者都设置后仍无输出，真因是 kernel 编译失败、二进制未更新（该推断由后续构建日志复核证实）。
- 初判默认 plog 目录权限问题是输出不可见的原因——目录确实不可写（属其他用户），但与本次无输出无关。

## 后续增量（2026-09-17，并行策展批：printf API 面、格式符子集实测与输出路由验证）

### R2：AscendC::Printf 混大小写形态失败，正确写法是小写 printf（独立 trace，conv 反传类算子调试）

按常见资料写 `AscendC::Printf("DBG Init: ...", ...)`，编译报（且被本卡 R1 同款"构建吞错"掩盖，调试输出完全不出现）：

```text
op_kernel/arch35/<op>.h:402:18: error: no member named 'Printf' in namespace 'AscendC'; did you mean 'printf'?
note: 'printf' declared here (kernel_operator_dump_tensor_intf.h:52): using __asc_aicore::printf;
```

正确写法为小写 `printf`（经 `using __asc_aicore::printf;` 引入，无需单独 include）。输出走测试进程 stdout、带 `[AIC Block 0/27]` 前缀，plog 里搜不到。配套实践：`if (blockIdx_ == 0)` 限流避免 27/56 核刷屏；最后一行停在哪个 DBG 即崩溃/挂起发生在其后第一个阶段；交付前 `grep -n "printf\|DBG" op_kernel/*.h` 清残留为 0。排查"DBG 没出现"先 `grep "error:" 构建日志` 再怀疑日志路由。

### R3：浮点格式符子集实测——%.6f/%.2e 截断为 "%."（独立 trace，roi_align 类算子精度排障）

RoiAlign 类 kernel printf 埋点中整型正常（`%d` 打出 1、`%lld` 打出 H=32 等），浮点全部残缺为格式串本身：

```text
[AIV Block 0/10] [DBG roi=0] box=[1, %.[AIV Block 0/10] ... roiStartH=%.[AIV Block 0/10] ...
```

即 `%.6f`/`%.2e` 不被支持，遇到不支持的形式**从格式符处截断输出**（残缺为 `%.`）——本卡原"%e 不支持（内置 log.h 注释，未实测）"由本轨迹实测闭合并扩展为 %.Nf/%.Ne 全系。修复（会话内实际执行）：

```bash
sed -i 's/%\.6f/%f/g; s/%\.2e/%f/g' <kernel_header>.h
```

修后浮点完整输出（`box=[1, 36.756367, 44.341740, ...]`，`%f` 默认 6 位小数）。配套纪律：需要更高分辨率时先乘 1e6 再 `%lld` 打定点整数（`printf("%lld", (long long)(v * 1e6))`），不要换格式符；printf 用编译开关关闭（如 `#define DEBUG_L1 0`）并交付前 grep 清零——留在紧同步循环里的 printf 还可能扰动正确性；多核并发输出交错属正常，按 `[AIV Block i/N]` 前缀过滤。误诊提示：初见 `%.[` 容易误读为"浮点变量是垃圾值"——实为格式串截断，值链路正常，对照整型字段正常即可区分。

### R4：kernel_printf.h 不存在的第二独立复现与输出路由验证（grid_sampler_3d 类算子，关闭本卡遗留问题）

第二独立 trace 复现 `fatal error: 'kernel_printf.h' file not found`（kernel 头 :31:10，Opc tool compile failed，kernel binary json 未生成）；对 `${ASCEND_HOME_PATH}/compiler/tikcpp` 全树 `find -iname '*printf*'` / grep AscendCPrintf 均无命中——CANN 9.1.0 安装树确认无该头。删 include 裸调 printf 后构建成功（Build completed successfully + 安装 SUCCESS）。

**输出路由实测（本卡遗留问题的接续闭合）**：printf 输出经 RUNTIME `ParsePrintToLog` 路由进 plog；设 `ASCEND_SLOG_PRINT_TO_STDOUT=1` + `ASCEND_GLOBAL_LOG_LEVEL=0` 后 stdout 出现：

```text
[printf.cc:337] ... ParsePrintToLog: PrintInfo: [AIV Block 0/54] [DBG-TILING] ...
```

即看 kernel printf 不必翻 plog 文件——上述两个环境变量即可在测试进程 stdout 直接观测。收尾：`grep -rn 'DBG-' op_kernel/` 为空后交付。

### 复现增量验证信号

- R2 修复前：`error: no member named 'Printf' in namespace 'AscendC'; did you mean 'printf'?`，build rc=0 掩盖、运行无任何 DBG 输出（来源平台：950；证据等级：tool_observed）。修复后：小写 printf 的 DBG 输出完整出现（stdout `[AIC Block 0/27]` 前缀，M/N/K、workspace 偏移等参数正确），并据此定位崩溃阶段（来源平台：950；证据等级：tool_observed）。
- R3 修复前：浮点值输出残缺为 `%.[`（`%.6f`/`%.2e` 处截断），整型 `%d`/`%lld` 正常（来源平台：950；证据等级：tool_observed）。修复后（统一 `%f`）：浮点完整输出（如 `box=[1, 36.756367, 44.341740, 91.031952, 86.638802]`、`acc[48]=1.490202`）（来源平台：950；证据等级：tool_observed）。
- R4：`fatal error: 'kernel_printf.h' file not found`（第二独立 trace）+ tikcpp 全树 find 无命中（来源平台：950；证据等级：tool_observed）；删 include 裸调 printf 后构建安装成功（来源平台：950；证据等级：tool_observed）；`ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0` 时 stdout 出现 `[printf.cc:337] ... ParsePrintToLog: PrintInfo: [AIV Block 0/54] ...`（来源平台：950；证据等级：tool_observed）；收尾 grep DBG- 为空（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 格式符支持面仅实测 `%f/%d/%lld` 可用、`%.6f/%.2e` 截断；`%x`/`%s`/宽度控制等未逐一验证。`%f` 默认 6 位小数，观测 ~1e-7 量级差异须定点放大。
- printf 大量使用会拖慢 kernel 并可能改变时序敏感行为；经 TTK 等评测框架运行时 stdout 捕获方式可能不同（未验证）。与 [kernel 侧打印定位崩溃：AscendC::PRINTF 宏卡](../runtime/kernel_printf_ascendc_macro_ascendc_dump_bisect.md)（CANN 9.0.0-beta.2 的 `AscendC::PRINTF` 宏 + ASCENDC_DUMP=1 形态）互为版本补充：9.1.0 实测为小写 printf 直调。