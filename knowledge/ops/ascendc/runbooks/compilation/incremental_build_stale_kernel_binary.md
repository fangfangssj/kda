---
type: "Runbook"
title: kernel头改动后增量编译沿用旧二进制
description: 修改 op_kernel 头文件后 build.sh 增量构建复用旧 kernel 二进制/不刷新 build 目录头文件副本，"编译成功"但行为不变甚至打包安装旧实现；rm -rf build 后才暴露真实编译错或更新产物。图模式 TBE 现场编译以安装包内源码为输入，会复现"已在源码修过"的错误。变体：编译失败被吞、旧 .o 照常打包——全量日志 grep + .o 时间戳判别。R8：只改实现头时 .json/算子包照常刷新并报成功（包新二进制旧的静默失效链），.o mtime 探针 + 硬编码常量决定性验证 + --make_clean 修复；多轮"修复无效"的结论全部作废。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-14T12:18:28Z"
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# kernel头改动后增量编译沿用旧二进制

## 现象

在一次 Cummin 算子 kernel 大改（把 `ProcessAraScan()` 从向量 Compare/Select 流水线整体重写为标量处理，代码在 `op_kernel/arch35/cummin.h`，还引入了一个不存在的成员名 `runningIdxFloatBuf`）之后：

1. 执行 `bash build.sh --soc=ascend950 -j8`，输出显示 `[100%] Built target Cummin_ascend950`、kernel binary 生成、CPack 打包成功——构建"成功"。
2. 安装新包后在 NPU 上跑 ST 测试，L0_004 仍是 26 个索引 mismatch、L0_005 仍是 values 大误差——与修复前的失败输出完全相同，好像代码没改过。
3. 执行 `rm -rf build/* && bash build.sh --soc=ascend950 -j8` 清理重建，立刻暴露真实编译错误：

```
error: use of undeclared identifier 'runningIdxFloatBuf'; did you mean 'runningIdxBuf'?
```

（新代码把类成员 `runningIdxBuf` 误写成了 `runningIdxFloatBuf`，这种错误在增量构建下完全被吞掉。）修掉成员名后清理重建成功，NPU 测试结果随即改变（L0_004/L0_005 转为 PASS）。

R2（CANN 9.1.0，独立会话，scatter 算子）：为修精度问题连续修改 kernel 头文件 `op_kernel/arch35/bench_no_harness_scatter.h`（先把 `DataCopyExtParams` 全部换成 `DataCopyParams`，再加 SetValue 诊断代码）。每一轮都执行 `bash build.sh --soc=ascend950 -j8`（输出 `[INFO] Build completed successfully!` + CPack 打包）并重装 `.run`，但 ST 结果与修改前**逐位相同**，多轮修复"全部无效"。直到 `bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 干净重建立刻失败，真实编译错误浮出：

```
RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessScatter Kernel File .../op_kernel/bench_no_harness_scatter_arch35.cpp!
op_kernel/arch35/bench_no_harness_scatter.h:199:9: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_scatter.h:246:13: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_scatter.h:290:9: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_scatter.h:298:9: error: no matching function for call to 'DataCopyPad'
（4 errors generated；随后 ld.lld: error: cannot open ..._mix_aiv_769.o: No such file or directory）
```

即 `DataCopyParams` 与 arch35 的 DataCopyPad 重载根本不匹配，此前的增量构建从未把这些改动编译进二进制——几轮"成功"构建与 ST 测的都是旧缓存 kernel。

R3（CANN 9.1.0，独立会话，回退改动后安装内容陈旧）：修改 `op_kernel/arch35/bench_no_harness_depthwise_conv_2d.h`（删除全部 Two-Product 补偿代码、恢复 Axpy）后执行标准交付链（增量构建 + 安装，两条命令均无任何报错），但验证安装内容时发现异常：安装到 CANN 的头文件 `grep -c "compBuf_\|productBuf_\|xHiBuf_\|Two-Product"` 返回 **22**——仍是含 Two-Product 的旧版本；`ls -la --time-style=full-iso` 对比显示 `build/op_kernel/ascendc_kernels/binary/dynamic/arch35/*.h` 副本时间戳为上一次构建，早于本次修改后的源文件——build.sh 在修改之后运行过，却没有刷新 build/ 目录里已存在的头文件副本，CPack 把陈旧副本打进了 .run 包并安装。此时源码目录 grep 已确认干净、UT/ST 也全部通过（它们链接的是新编译产物），唯独"安装包内的头文件"是旧的——"命令全绿但装的是旧代码"极难察觉。

R4（CANN 9.1.0，独立会话，图模式暴露）：geir 图模式验证时 kernel 编译报 5 个模板错误，报错路径指向**安装包内**的 kernel 头文件：

```
<CANN安装根>/opp/vendors/bench_no_harness_scatter_custom/op_impl/ai_core/tbe/.../dynamic/arch35/bench_no_harness_scatter.h:265:28: error: use 'template' keyword to treat 'ReinterpretCast' as a dependent template name
        auto yGmU16 = yGm_.ReinterpretCast<uint16_t>();
```

而这条 `template` 关键字修复**早已落在源码里**。直接执行 `bash build.sh --soc=ascend950 -j8` 重新打包并 `--force` 安装后，安装包内的头文件**仍然是旧代码**；直到 `bash build.sh --make_clean` 清空构建目录再全量重建，安装包内才出现 `template` 关键字，geir 也随之通过 kernel 编译。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 实现主要在 `op_kernel/arch35/*.h` 头文件中、由 `op_kernel/*.cpp` 包含的工程结构；"改代码→增量构建→安装→验证"的迭代联调，含图模式/geir 验证路径（TBE 现场编译以安装包内源码为输入）。
- API 路径：`bash build.sh --soc=ascend950 -j8`（custom_opp 工程 CMake 增量构建 + opc/bisheng kernel 编译）、`bash build.sh --make_clean` / `rm -rf build`（全量清理）、`.run` 安装；构建产物在 `build/op_kernel/...` 与 `build/custom_opp_ubuntu_x86_64.run`，安装后在 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/`。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0（多个独立会话）。
- 适用粒度：`workflow_only`（构建系统行为；随 build track 落 compilation）。
- 路线优先级：recommended_fastpath（kernel 改动后先清理重建再下结论）。
- 明确不适用/未验证场景：纯 op_host/op_api 改动是否同样受缓存影响未单独验证；UT 工程（tests/ut 独立 cmake，未观察到此行为）；只改 op_host/tiling 或 op_api 的场景未观察到（未验证）。

## 判别方法

**触发**：修改了 `op_kernel` 下的头文件/源文件并重新 `bash build.sh`，构建快速"成功"（提示 Built target 而无实际重编译日志），但 NPU 测试输出与修改前逐字节相同；或安装包/安装树内容（特征符号、锚点行）与源码不一致。

**探针**（可执行）：

```bash
# 1. 观察是否真的重编了 kernel（增量路径下往往直接跳过）
rm -rf build/* && bash build.sh --soc=ascend950 -j8 2>&1 | grep -E "(error|Error|ERROR|Built target|Package|completed)"

# 2. （R2）对比构建产物与安装产物的 MD5，排除"装了旧包"
md5sum build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o \
       $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_impl/.../对应的 .o

# 3. （R3）安装目录特征符号计数（对比源码中的预期）+ 时间戳对比
grep -c "<本次改动引入/删除的特征符号>" $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/.../arch35/<kernel>.h
ls -la --time-style=full-iso op_kernel/arch35/<kernel>.h build/op_kernel/ascendc_kernels/binary/dynamic/arch35/<kernel>.h
grep -rn "<特征符号>" build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/

# 4. （R4）锚点行对比：源码 vs 安装包 vs build 中间产物（以出错的行号为锚）
sed -n '<改动行号>p' operators/<op>/op_kernel/arch35/<kernel>.h
sed -n '<改动行号>p' $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/.../dynamic/arch35/<kernel>.h
sed -n '<改动行号>p' operators/<op>/build/op_kernel/ascendc_kernels/binary/dynamic/arch35/<kernel>.h
# 源码是新形态而后两份是旧形态，即为陈旧缓存
```

**原始观测**：增量构建"成功"→ 测试结果不变；清理重建 → `error: use of undeclared identifier 'runningIdxFloatBuf'` 暴露；修复后清理重建 → 测试行为改变（L0_004 26 个索引 mismatch → 全对）。

**解释**：custom_opp 工程的增量构建对 kernel 头文件改动没有正确触发 kernel 二进制重编译/头文件副本刷新，打包时携带的是 `build/` 下的旧产物；"构建成功 + 行为逐字节不变"是缓存陈旧的判别信号，真正的编译错误被缓存吞掉。R4 补充：图模式（GE）的 TBE 现场编译以**安装包内**的 kernel 源码为输入，安装包陈旧时会把一个已在源码侧修复过的错误再报一遍；整条链上没有任何一步报"缓存陈旧"，构建日志显示 "Build completed successfully"，极难从日志发现。

**区分相邻根因**：若确认构建确实重编了 kernel、而 NPU 仍运行旧二进制，则属运行时 device binary 缓存（主机侧变量不进 binary hash，需 PRINTF 改 hash 强制重载），见 [kernel 源码改动未生效因 binary 缓存卡](kernel_binary_cache_printf_hash_toggle.md)；重编译生成了新 `.run` 但根本没重装（vendors 安装树陈旧），见 [重编未重装 vendors 旧 tiling 卡](rebuild_without_reinstall_561103.md)——判别都用清理重建/时间线/内容对比先行分流。本卡是构建期增量依赖缺口。

## 设计决策

- 处置策略选"kernel 改动一律清理重建"而不是排查构建依赖图：构建系统内部依赖不在算子工程可控范围，清理重建代价（一次全量编译）远低于在被污染的二进制上调试的代价。
- R2 从发现起的行为修正：kernel 每次改动后一律 `rm -rf build && bash build.sh --soc=ascend950 -j8` 干净重建再安装测试，不再信任增量构建的"成功"输出；干净构建成本可控（单算子全量重编），而增量假阳性会浪费整轮"修→测"循环并误导归因。
- R4 排查里先怀疑安装包的依据：图模式 TBE 现场编译的错误信息直接带安装包内绝对路径（`opp/vendors/.../arch35/*.h:<行>`），这是"以安装副本为编译输入"的直接证据；eager 路径不暴露这一点。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0 custom_opp 工程（CMake + opc/bisheng）。
- 相关文件：`op_kernel/arch35/cummin.h`（改动源）、`build/op_kernel/`（陈旧产物）、`build.sh`；R2 `op_kernel/arch35/bench_no_harness_scatter.h`；R3 `op_kernel/arch35/bench_no_harness_depthwise_conv_2d.h`（含 build/ 副本与 CPack 打包目录）；R4 安装树 `$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/op_impl/ai_core/tbe/.../dynamic/arch35/`。

## 验证信号

验证来自同一开发会话的对照实验：同一份含未声明标识符的 kernel 源码，增量 `bash build.sh` 报成功（来源平台：950；证据等级：tool_observed）；`rm -rf build/*` 清理重建同一条命令立即报 `error: use of undeclared identifier 'runningIdxFloatBuf'`（来源平台：950；证据等级：tool_observed）；修复成员名并清理重建后，NPU 测试结果随新代码改变（L0_004/L0_005 由 FAIL 转 PASS）（来源平台：950；证据等级：tool_observed）。

R2（CANN 9.1.0，独立会话）：

- 假阳性构建：`DataCopyParams` 改动后 `bash build.sh --soc=ascend950 -j8` 输出 `CPack: - package: custom_opp_ubuntu_x86_64.run generated` / `[INFO] Build completed successfully!`，重装后 ST 结果与改动前逐位相同（同一用例 MERE=1.85e+00、actual=0.000061/0.000000 不变）（来源平台：950；证据等级：tool_observed）。
- 干净重建暴露真错：`--make_clean` 后 `ascendc_compile_kernel.py` 抛 `RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessScatter` + 4 处 `error: no matching function for call to 'DataCopyPad'` + `ld.lld: cannot open ..._769.o`（来源平台：950；证据等级：tool_observed）。
- 修复编译错误并 `rm -rf build` 干净重建后，SetValue 拷贝改动第一次在 NPU 输出生效：actual 从近零变为真实浮点数据值，证明此前确实在测旧二进制（来源平台：950；证据等级：tool_observed）。

R3（CANN 9.1.0，独立会话）：

- 故障观测：增量构建+安装后，安装目录头文件含 22 处 Two-Product 特征符号（源码已回退为 0）；build/ 副本时间戳早于源文件修改时间；CPack 打包目录副本同样含旧特征（来源平台：950；证据等级：tool_observed）。
- 修复动作：`rm -rf build && bash build.sh --soc=ascend950 -j8` 构建成功，重新生成 custom_opp_ubuntu_x86_64.run（来源平台：950；证据等级：tool_observed）。
- 修复后校验：安装目录头文件 Two-Product 特征符号 = 0，`Axpy<float, float>` = 10 处（与源码一致）；全量重建包上 UT 57/57、ST 53/53 通过（来源平台：950；证据等级：tool_observed）。

R4（CANN 9.1.0，独立会话）：

- 失败复现：图模式 kernel 编译报 5 处 "use 'template' keyword to treat 'ReinterpretCast' as a dependent template name"，路径在安装包内（来源平台：950；证据等级：tool_observed）。
- 判别探针：`sed -n '265p'` 三份对比显示 source=template 版、installed=旧版；build 中间产物与 CPack 包均为旧版（来源平台：950；证据等级：tool_observed）。
- 增量无效证据：普通 `build.sh --soc=ascend950` + `--force` 安装后 installed 第 265 行仍为旧码（来源平台：950；证据等级：tool_observed）。
- 修复验证：`--make_clean` 重建后 build 中间产物与安装包第 265 行均为 `template ReinterpretCast<uint16_t>()`；geir RunGraph success 且精度校验 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 清理重建会丢失全部增量编译缓存，大工程编译时间变长；仅在 kernel 改动后必须采用。
- 该行为由 CANN 9.1.0 构建链的依赖跟踪决定，其他版本可能已修复，未验证，推断可迁移（判别方法同上）。
- R3 触发面：发生在修改 op_kernel 头文件（类成员/内联实现所在文件）的场景；修改 .cpp/tiling 等其他文件是否同样不刷新未验证；编译出的 .o 二进制是否也受陈旧副本影响未单独验证（本例通过 UT/ST 行为一致性侧面确认最终产物正确）。
- R4 增量构建不拷贝新 `op_kernel` 头的确切构建规则（CMake target 依赖缺失 or 显式跳过）未定位到脚本行级；预编译 Kernel 二进制（.o）是否同样存在增量不失效问题（本例 .o 哈希每次都变，间接说明二进制路径有重编）未单独验证。

**未决项**：

- 增量构建依赖图的具体缺口（是 CMake 层还是 opc 层未跟踪头文件）未定位；定位后可推动构建脚本按头文件改动自动清理。
- R2 优先项：复现并定位构建链为何在 kernel TU 编译失败时仍报成功并打包缓存二进制（build.sh 脚本段与 cmake 目标的失败传播路径），向构建模板维护方反馈。
- R3 优先项：追到 build.sh/CPack 拷贝规则的源码级机制，确认"何种修改不触发副本刷新"的精确条件；评估在交付链中加一步自动"源码 vs 安装内容"一致性校验的可行性。
## 坏实践

- 见"构建成功"就认定新代码已进入二进制——本例中被吞掉的甚至是硬编译错误。
- 修复后测试结果不变时，先怀疑自己的修复逻辑而不是构建产物——先用清理重建排除缓存因素，再进入代码调试。
- 修改"无效"就连环堆叠更多修改（R2 中 DataCopyParams → PipeBarrier → SetValue 连环被同一假阳性误导），而不是先验证二进制确实更新了。
- 图模式报了一个"昨天修过"的错误时，去怀疑修复本身写错了而不怀疑安装副本陈旧。
- "UT/ST 都过了所以安装包肯定没问题"：UT/ST 链接的是构建产物中的测试目标，与 CPack 打包进 .run 的头文件副本是两条路径（R3 中前者全新、后者陈旧）。

## 误诊

- 无（本次通过对照实验直接确认；此前浪费的一轮"修复无效"分析即缓存污染本身）。
- R2 连续两轮把"修改无效"归因于修改本身错误（DataCopyParams 假设、SetValue 假设先后被放弃），实际是二进制未更新；`--make_clean` 一次即澄清。
- R4 中间一度把"安装包陈旧"与"`.run` 安装失败（l0op 符号自校验）"混在一起看——后者用 `--force` 解决后第一反应是"装好了"，但锚点行复核显示包内容未变，才把注意力转回构建缓存。

## 后续增量（2026-09-16，并行策展批：ST Real 链路判读、grep 自检探针与并行测试伪信号）

**1. ST Real 链路判读**：ST Real 执行的是 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom` 安装目录里的 op_api 库与 kernel binary；ST 日志开头的"编译成功"编译的是**测试程序**（链接安装目录的 op_api 头/库），与 kernel 是否重编无关。因此"改了 kernel → 结果不变"有三种可能且都必须排除：① 增量构建没重编 kernel（甚至吞掉编译错误）；② 包重编了但没重装；③ 装了但测试跑在装包之前（并行开发时）。任何"修复无效"的结论都必须在"全新构建（--make_clean）+ 无条件 `.run --force` 重装 + 复测仍然不变"之后才能下。

**2. 静态可编译性自检探针（秒级，先于一切时间戳推断）**：

```bash
grep -n "<刚改的成员名>" op_kernel/arch35/<kernel>.h
# 引用处都有声明才能通过编译；"源码根本无法编译而此前 ST 日志却显示编译成功且有运行结果"
# → 该轮结果来自旧 binary/旧包的直接铁证
```

**3. 并行测试伪信号**：实现方完成 int64 路径并构建出新包，但测试方在此之前已经跑了 Real 模式全量——int64 用例 39 条全部 mismatch-rate=1.00（输出全 0），一度像"int64 kernel 全错"；按构建（包生成时间）与测试开始时间的时间线核对确认为旧包，实现本身无此缺陷。并行开发/测试分工时，测试方跑全量前先确认安装目录里是最新包；ST 报告里把"kernel 包构建/安装时间"和"测试开始时间"打进日志头，事后可对账。

**4. 验证信号（argmax 类算子联调会话，三次踩中该陷阱的完整记录）**：

- 增量 build 对含 `redefinition of 'outLocal'` 编译错误的源码显示构建成功（包生成、安装成功），ST 复测 75/101 与修改前相同；`--make_clean` 后同源码立即报 `RuntimeError: Kernel Compilation Error ... redefinition`（来源平台：950；证据等级：tool_observed）。
- 修复会话入场：`grep -n "maxLocal"` 显示该变量被引用 6 处但未声明（源码不可编译），而联调 ST 日志显示"编译成功"且有 91/101 运行结果——该轮结果来自旧安装包；对不可编译源码再跑一次增量 build 仍显示成功（构建缓存未重编 kernel）（来源平台：950；证据等级：tool_observed）。
- 整体重写 ARA 两个函数 + 重编 + `--force` 重装后：ST Real 101/101（100%），行为发生实质变化，确认此前"结果不变"轮次均为旧 binary（来源平台：950；证据等级：tool_observed）。
- 并行场景：测试方在实现方的新包安装前跑 Real 全量，int64 用例 39 条全部 mismatch-rate=1.00、输出全 0；按构建与测试时间线核对确认为旧包，实现本身无此缺陷（来源平台：950；证据等级：tool_observed——失败输出为工具实测，旧包归因为会话时间线分析，已如实标注）。

**5. 误诊补充**：曾据"结果不变"推断 ARA 标量 Cast 修复无效并转向其他假设（padding 等）——实际是编译错误被增量构建吞掉，修复本身有效；clean 构建修掉编译错误后立即恢复 14 条用例。曾把并行测试方的 int64 全错当成 int64 kernel 缺陷展开排查——实际是新包未安装。

## 后续增量（2026-09-16，并行策展批：ASCENDC_TPL 模板维度变更后 kernelList 变体缺失 → 361001）

CANN 9.2.0（Ascend950PR，asc_opc_tool 构建链）独立会话以新形态复现同一根因，并给出**变体计数探针**：

- **现象**：稀疏 top-K 注意力算子把 TilingKey 从 2 维（dtype × KV_MODE）恢复为 3 维（dtype × IS_CAUSAL × KV_MODE，共 8 变体）后，增量构建显示 `Build completed successfully!`；安装运行 ST，24 用例中 21 个通过，唯独 3 个 MLA 用例（Dk=576、Dv=512，即 KV_MODE=1 的 Dk≠Dv 分支）全部报 `aclnn execute failed: 361001`。产物异常：每个 dtype 的二进制 kernelList 只有 2 个 kernel（fp16：`_1`、`_257`；bf16：`_27`、`_283`），而 3 维 TilingKey 应生成 4 个/每 dtype（fp16 应含 `_1`、`_257`、`_65537`、`_65793`）——KV_MODE=1 的两个变体缺失，恰好对应失败用例（运行期按 (OpType, tilingKey) 找不到匹配 kernel 即 361001）。
- **变体计数探针**（跑测试前先核对，可执行）：

```bash
for f in build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json; do
    echo "=== $(basename $f) ==="
    python3 -c "import json; d=json.load(open('$f')); print(d.get('kernelList'))"
done
# kernelList 数量必须 == 预期变体数（本例 2 dtype × 2 causal × 2 kv = 8，即 4/每 dtype）
```

- **clean 构建暴露被掩盖的编译错误**：`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 立即失败，报 `error: use of undeclared identifier 'IS_CAUSAL'`（replaceAll 批量替换模板签名时只改了类限定名、漏改成员函数的模板声明行）；对应修复把声明行补齐第 2 维：

```cpp
// 修复前：template <typename D_T_Q, int KV_MODE>
// 修复后：
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
```

- 修复模板声明行并 clean 构建后产物含 4 kernel/每 dtype（8 变体齐全），重装重跑全量 24 用例 `PASS=24 FAIL=0`，原 361001 用例 `MERE=1.76e-04` 通过（来源平台：950；证据等级：tool_observed）。
- 判读补充：ST/UT 出现"部分变体专属用例报 361001 而其余通过"时，第一时间怀疑产物变体缺失而非 kernel 逻辑；模板维度改动（`*_tiling_key.h` 的 ASCENDC_TPL 声明/选择宏）与 kernel 类模板签名变更后，一律 `--make_clean` 全量重建并核对变体数。

## 后续增量（2026-09-16，并行策展批：conv3d 编译失败被吞变体（R5）与 cmake 手动增量的 copy_kernel_srcs 时间戳守卫）

本节并入两条独立会话的增量（其中 conv3d 变体在并行批次中曾以"R2"标号落稿，为避免与上文 scatter 会话的 R2 冲突，此处改标 R5，内容保持原样）。

**1. R5（conv3d 反传算子，CANN 9.1.0）：编译失败被吞、旧 .o 照常打包**。修改 kernel（含 Process 开头 `SetValue(0, 99.0)` 调试探针）后 build.sh 报 `[INFO] Build completed successfully!` 并安装 SUCCESS，ST 输出与修改前逐位相同（连最粗暴的调试写入都不可见）；全量 grep 编译日志发现 4 条 `error: no member named 'PIPE_MTE2' in namespace 'AscendC'` 位于日志中部（`tail -3` 不可见），安装目录 `.o` 时间戳停留在上次成功编译时刻；`rm -rf build build_out` 清理重建后真实错误暴露、修复后行为恢复随代码变化（来源平台：950；证据等级：tool_observed）。

- R5 变体判别：编译失败未被传播到整体退出码——kernel 二进制编译是独立子步骤，其失败不改变尾部提示，CPack 继续打包 `build/` 下遗留的旧 `.o`，安装后运行的是旧 kernel（新代码"从未上板"）。与"无 error 的增量缓存不失效"不同，本变体可全量 grep 日志抓到 error；判别信号是"日志中部有 error + 尾部 Build completed + 安装目录 .o mtime 陈旧"。
- R5 门禁：每次构建后固定 `bash build.sh ... 2>&1 | grep -iE "error:" | grep -v deprecat` 作全量日志检查，而非信任尾部 `Build completed successfully`；大改（模板参数/头文件/tiling 结构体变动）后一律 `rm -rf build` 再构建；联调前核对安装目录 `.o` 的 mtime 与本次构建时间一致。
- R5 误诊增补：初始判断"kernel 没有执行（TilingKey 不匹配 / op_def 注册问题 / aclnn 调用问题）"——被证伪：真实原因是编译失败被吞、安装的是旧二进制，新 kernel 从未上板；调试代码（如对 bf16 实例化的 `static_cast<D_T_X>(float)`）引入的新编译失败又被同一问题吞掉，形成"改了也没报错"的双重假象。
- R5 遗留增补：build.sh 吞错误的具体机制（哪个子步骤的退出码未被聚合）未追根因；修复构建脚本本身不在轨迹范围内，列为接续项。

**2. cmake 手动增量的机制定位与定向清理（gather 算子迭代二，CANN 9.1.0）**：在已有 build 目录上手动 `cmake --build . -j8` 时，输出全是 host 侧目标（`bench_no_harness_gather_custom_npu_supported_ops`、`ascendc_kernels_copy_kernel_srcs` 等 "Built target"），**没有任何 `Generating BenchNoHarnessGather_*` 的 kernel 生成动作**，kernel binary 保持旧版。机制（补充 R1-R4 未定位的依赖缺口一层）：kernel 源码先由 `ascendc_kernels_copy_kernel_srcs` 目标**拷贝**到 build 目录（以 timestamp 文件守卫），kernel 编译消费的是拷贝副本；该拷贝目标未感知 `op_kernel/arch35/*.h` 的修改时，`cmake --build .` 只重编 host 产物。

定向修复（比 `rm -rf build` 更轻量，保留 CMakeCache 与 host 侧编译产物）：

```bash
cd <op>/build
# 1. 清理源拷贝目标（去掉 timestamp 守卫）
cmake --build . --target ascendc_kernels_copy_kernel_srcs --clean-first
# 2. touch 源文件确保依赖扫描感知变更，再全量构建
touch ../op_kernel/arch35/<kernel>.h ../op_kernel/<kernel>_arch35.cpp
cmake --build . -j8
# 成功标志：输出出现大量 "[ascend950] Generating <OpType>_<hash> ... Done" 行
```

改动 kernel 头文件（`.h`）时同时 touch 对应 `.cpp` 入口（依赖扫描对头文件变更的传播更可靠）。本卡不解决"打包产物未更新"（kernel 重编后还需重新 package/`.run` 安装）。目标名 `ascendc_kernels_copy_kernel_srcs`/`<OpType>_<soc>` 取自本工程 CMake 生成结构（`npu_op_kernel_sources` 首参），其他算子名/版本的目标名会不同。

- 验证信号：修复前改写 kernel 后 `cmake --build . -j8` 仅 host 目标 "Built target"、无 `Generating` 动作、binary 数量/内容不变（来源平台：950 构建链路；证据等级：tool_observed）；清拷贝目标 + touch 后输出连续 `[ascend950] Generating BenchNoHarnessGather_<hash> ... Done`，18 个 kernel binary 全部重新生成（6 dtype × 3 index dtype 组合，每个含单/双缓冲 2 kernel）（来源平台：950 构建链路；证据等级：tool_observed）；后续打包安装 + ST Real 25/25 bitwise 通过，确认新 kernel 真实生效（来源平台：950；证据等级：tool_observed）。
- 增量后自检补充：每次增量编译后核对 kernel binary（`find build/op_kernel/ascendc_kernels/binary -name '*.o' | wc -l` 并抽查时间戳，或 grep 编译输出中的 `Generating` 行数）。

## 后续增量（2026-09-17，并行策展批：R6——三处缓存清理位置与安装/构建 .o 时间戳对账）

R6（CANN 9.1.0，roi_align 类算子）：修改 `op_kernel/arch35/<op>.h`（SetValue 写出改 DataCopyPad + 新埋 printf 调试块）后标准构建安装全"成功"（`[INFO] Build completed successfully!`、安装 SUCCESS），但失败指标几乎不变（MARE 与改前同级）、新 printf 完全不出现。产物对账：**构建目录 `.o` 时间戳（03:18）早于源码修改、安装目录时间戳（03:24）为旧二进制**——打包安装的是旧 `.o`。三处一并清理后重建才暴露被掩盖的真实编译错误：

```bash
rm -rf operators/<op>/build/op_kernel/ascendc_kernels
rm -rf ~/.cache/ascend/kernel_cache 2>/dev/null
rm -rf /tmp/atc_data 2>/dev/null
cd operators/<op> && bash build.sh --soc=ascend950 -j8 2>&1 | grep -iE "error generated|error:|Build completed"
```

- **三处清理位置**：build 内 `op_kernel/ascendc_kernels` + `~/.cache/ascend/kernel_cache` + `/tmp/atc_data`——三处一并清理后错误才显形，未单独归因哪一处是关键失效点（工程上一起清最稳妥）。
- **时间戳/sha 对账作为例行动作**：构建脚本的成功输出无法区分"重编成功"与"未重编"，只有产物指纹能证明；对账命令零侵入、秒级完成（`ls -la` 构建产物 vs 安装产物、`sha256sum` 双端对比）。
- **grep 模式补充**：CANN kernel 编译失败的话术是 `N error generated.`（本例 `1 error generated.` + `error: no matching function for call to 'DataCopyPad'`），不含 `fatal error`——只 grep `fatal error` 会漏检，模式须含 `error generated` 与 `error:`。
- 修正编译错误后全量清理重建：`Build completed successfully!` + 包生成 + 安装成功，原失败用例 PASS（MERE=2.851e-06）、全量 ST 36/36——编辑真正生效。

R6 验证信号：

- 清理前产物对账：构建 `.o` 时间戳 03:18 / 安装 03:24，源码修改晚于 03:18（来源平台：950；证据等级：tool_observed）。
- 清理后重建暴露被掩盖的编译错误：`1 error generated.` 与 `<op>.h:430:17: error: no matching function for call to 'DataCopyPad'` 原文（来源平台：950；证据等级：tool_observed）。
- 修正后全量清理重建 + 重装：原失败用例 PASS（MERE=2.851e-06）、全量 ST 36/36（来源平台：950；证据等级：tool_observed）。

R6 边界补充：R6 未定位增量构建失效的确切环节（`ascendc_kernels_copy_kernel_srcs` 时间戳机制 vs `~/.cache/ascend/kernel_cache` 缓存键），与复现增量的机制定位互补；运行行为不变期间日志可见 `TaskFailCallBackManager` 构造/析构与 `xpu_task_fail_callback_data_manager` 记录，属旧二进制运行的伴随噪声，不构成独立判据。

## 后续增量（2026-09-17，并行策展批：R7——CANN 9.2.0 GRU 实例、FATBIN ld.lld 形态与"输出矛盾直读"）

R7（CANN 9.2.0，独立会话，GRU 类算子 bench_no_harness_gru）：修改 `op_kernel/arch35/bench_no_harness_gru.h` 后运行 `bash build.sh --soc=ascend950 -j8 2>&1 | grep -E "error:|completed|Package" | tail -10`，输出**同时包含**真实 kernel 编译错误与成功标志：

```
/workspace/.../op_kernel/arch35/bench_no_harness_gru.h:202:24: error: no member named 'max' in namespace 'std'; did you mean simply 'max'?
/workspace/.../op_kernel/arch35/bench_no_harness_gru.h:342:13: error: no matching function for call to 'ReduceSum'
...
Copy .../custom_opp_ubuntu_x86_64.run to .../build/
[INFO] Build completed successfully!
[INFO] Package: .../build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run
```

即 kernel 编译已失败，但构建脚本仍走完 cpack 打包并宣布成功。另一轮（代码有语法错误未完全修完）运行测试时发现"改了代码但运行行为与旧版逐位一致"（`There are compilation errors but the build says 'completed successfully' - this means the old binary is being used`）。带 `--make_clean` 的干净构建硬失败并暴露真实错误（FATBIN 阶段形态）：

```
[ERROR] TBE(714305,python3):... [ascend950]  BenchNoHarnessGru_48c9e4f6... ld.lld: error: cannot open /workspace/.../build/op_kernel/BenchNoHarnessGru_ascend950/kernel_0/kernel_meta_BenchNoHarnessGru_48c9.../kernel_meta/BenchNoHarnessGru_48c9..._0.o: No such file or directory
[ERROR] ... An error occurred during compile phases of CompileStage.FATBIN, msg is ld.lld: error: cannot open ...
RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessGru Kernel File .../bench_no_harness_gru_arch35.cpp!
```

- R7 判读补充：`grep -E "error:|completed|Package"` 组合能让"错误行与 SUCCESS 行同屏"直接暴露矛盾——比只 tail 结尾更早发现假成功；`ld.lld: error: cannot open ...kernel_meta_*_*.o` 出现在 FATBIN 阶段时，真错误在更早的编译阶段日志里（.o 从未生成）。
- R7 与 R5 同型（编译失败被吞、旧 .o 照常打包），新增 CANN 9.2.0 复验（R1-R6 均为 9.1.0）与 GRU 类算子形态；`--make_clean` 硬失败 + 修完全部编译错误后 3 个 kernel `Generating ... Done`、安装后 ST 行为随之改变。
- R7 误诊增补：本会话一度把"修改后测试结果逐位不变"归因为二进制缓存过期而排查内核缓存；实际上是构建从未成功、安装的一直是旧包（构建日志回看证实）。

R7 验证信号：

- 增量构建输出同时含 2 类 `error:`（std::max / ReduceSum）与 `[INFO] Build completed successfully!` + Package 行（来源平台：950；证据等级：tool_observed）。
- `--make_clean` 后同代码构建硬失败：`ld.lld: error: cannot open ...kernel_meta_..._0.o / _4.o`、`CompileStage.FATBIN` 异常、`RuntimeError: Kernel Compilation Error`（来源平台：950；证据等级：tool_observed）。
- 修完所有编译错误后干净构建：3 个 kernel `Generating ... Done`，安装后 ST 行为随之改变（来源平台：950；证据等级：tool_observed）。


## 后续增量（2026-09-17，并行策展批：R9——cmake 手动三目标调用形态与 error 计数门禁脚本）

R7（conv 反传类算子调试循环，x86 构建主机 + CANN 9.1.0，独立 trace）：绕过 build.sh 直接 `cd build && cmake .. -DASCEND_COMPUTE_UNIT=ascend950 && cmake --build . --target all binary package -- -j8` 的手动形态下复现同族行为——`build rc=0`、CPack 显示 `custom_opp_ubuntu_x86_64.run successfully created`，但 build 日志中段躺着 kernel 子构建（ascendc_compile_kernel.py 驱动）的编译错误（`error: no member named 'Printf' in namespace 'AscendC'`），安装目录 kernel `.o` 时间戳停在旧值、打包复用旧二进制，新加的 kernel 调试输出一条不出现。机制补充：kernel 子构建（python 驱动）的错误**不会置位 make 失败状态**，日志是唯一可靠信号——"构建成功"的判定须从退出码升级为三条件（退出码 0 + `grep -c 'error:'` 为 0 + 产物时间戳晚于源码 mtime）。

可复用的门禁脚本形态（编译-安装-运行一条龙，带错误计数门禁）：

```bash
rm -rf build && mkdir -p build && cd build && cmake .. -DASCEND_COMPUTE_UNIT=ascend950 > /dev/null 2>&1 && cmake --build . --target all binary package -- -j8 > /tmp/buildB.log 2>&1
echo "build rc=$?, errors=$(grep -c 'error:' /tmp/buildB.log)"
grep "error:" /tmp/buildB.log | head -3
if [ $(grep -c 'error:' /tmp/buildB.log) -eq 0 ]; then
    bash custom_opp_ubuntu_x86_64.run --quiet --force 2>&1 | tail -1
fi
```

相邻形态（同链路踩到）：只构建 `all package` 而漏掉 `binary` 目标时，CPack 在 `op_kernel/cmake_install.cmake:50 (file)` 报 INSTALL 找不到文件——另一形态的同类坑，用全目标构建规避。判读噪声：`grep 'error:'` 会对含 "error" 字样的头文件路径（如 `ge_api_error_codes.h` 的 `In file included from` 行）计噪——看 `error:` 精确串而非裸 "error"。

R7 验证信号：

- 触发：build rc=0 且包生成成功，但 build 日志含 kernel 子构建 `error: no member named 'Printf' ...`，安装 `.o` 时间戳未更新、运行无新调试输出（来源平台：x86 构建主机；证据等级：tool_observed）。
- 处置：修复源码 + `rm -rf build` 全量重建，`.o` 重新生成、安装后调试输出出现（`[AIC Block 0/27] DBG Init: ...`）（来源平台：x86 构建主机 + 950 运行；证据等级：tool_observed）。

R7 边界：kernel 子构建错误不传导退出码的底层机制（python 子进程返回值被吞）未深挖，不影响判别方法有效性；构建框架行为仅验证 CANN 9.1.0 npu_op_package 工程的手动 cmake 三目标路径。

## 后续增量（2026-09-17，并行策展批：R8——包刷新掩盖陈旧 .o 的取证链与"多轮修复无效"作废）

R8 并入 argmax 类算子（Ascend950PR + CANN 9.1.0）的完整取证案例：只修改 kernel 实现头文件（`op_kernel/arch35/<op>.h`，被 `*_arch35.cpp` include）时，增量构建不重编 `.o`，但打包/安装流程照常重新生成 `.json` 与 `.run` 包并报成功——形成"构建成功 + 包是新的 + 二进制是旧的"的**静默失效链**。该会话连续多轮修改 kernel（加 18 处 SetFlag/WaitFlag 同步、输入读取改 GlobalTensor::GetValue 直读 GM、换 TBuf+PipeBarrier、关 DoubleBuffer、输出改硬编码常量）并执行 `build.sh + .run --force`，每次都打印 `[INFO] Build completed successfully!` 并生成新算子包，但 ST 结果逐位不变（同样的错值）——所有"修复无效"的结论全部作废（此前被错杀的流水线竞争、DataCopyPad、GetValue 等假设在陈旧二进制上根本没有被测试过）。

**R8 取证探针链**（怀疑"修改没生效"时按序执行）：

```bash
# 1) 对比构建目录 .o / .json 时间戳：.json 新而 .o 旧 → 陈旧二进制
ls -la build/op_kernel/ascendc_kernels/binary/ascend950/<op_name>/
# 2) 对比构建目录与安装目录 .o 的 MD5（相同只能证明安装拷贝了构建产物，
#    若构建产物本身陈旧，两边一致仍是旧码；须结合时间戳判断）
md5sum build/op_kernel/ascendc_kernels/binary/ascend950/<op_name>/*.o \
       $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_impl/ai_core/tbe/kernel/ascend950/<op_name>/*.o
# 3) 决定性验证：在 kernel 里临时输出一个不可能正确的常量（如把输出 SetValue 成 42），
#    干净重建后跑一条用例——若用例输出随之变化，说明新码在跑；若不变，仍是旧二进制
bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8 && bash build/custom_opp_ubuntu_x86_64.run --force
```

**R8 与既有条目的关系**：与 R1-R7 同根因（增量依赖跟踪不覆盖"只改被 include 的实现头"），新增三条增量事实——(a) 包/安装链不校验 `.o` 新旧、照常刷新 `.json` 与包（时间戳证据：5 个 `.o` 全停在 17:50、同目录 `.json` 为 18:28、安装目录产物 18:29 只是复制了陈旧 `.o`）；(b) 探针选时间戳优先于 MD5：MD5 只能证明"安装=构建"，无法证明"构建=源码"；(c) 修复用 `bash build.sh --make_clean`（语义完整）而非手动删单个 `.o`（`.o` 文件名含 32 位哈希后缀、按 dtype×mode 模板分多个 binary，手工挑选易漏）。干净重建后**第一次**编译可能暴露此前被掩盖的编译错误（本例 18 处 `no matching function for call to 'SetFlag'`）——这不是回归，是被陈旧构建藏住的真实问题。

**R8 验证信号**（argmax 算子联调，Ascend950PR + CANN 9.1.0）：

- 构建目录 5 个 `.o` 时间戳全部停在 17:50，而同目录 `.json` 为 18:28、安装目录产物为 18:29（来源平台：950；证据等级：tool_observed）。
- `--make_clean` 干净重建立刻报出 18 处此前完全静默的 `error: no matching function for call to 'SetFlag'`（来源平台：950；证据等级：tool_observed）。
- 修好编译错并干净重建后，"输出常量 42"的调试改动生效：`[100,50]`、`[8]` int64 AR 用例的实际值从索引变成 42/大整数常量，而此前同样的改动毫无效果（来源平台：950；证据等级：tool_observed）。
- 干净重建后构建目录与安装目录 `.o` 的 MD5 一致，且时间戳同步刷新（来源平台：950；证据等级：tool_observed）。

**R8 边界增补**：

- 本变体机制针对"改 .h 不触发 .o 重编"；若现象是"改 .cpp 也不重编"，属另一类缓存问题（外部构建缓存），未验证。
- 构建目录被 `rm -rf build` 后的首次全量构建不受此影响。
- 坏实践增补：连续多轮"改代码→build→install→跑测试"而不核对产物时间戳，把陈旧二进制上的"无效"当成"方向错误"记录结论；用 `md5sum build/*.o install/*.o` 相同来证明"是新码"——只能证明安装拷贝成功。
- 评测层的同型陷阱（bench 适配目录快照未刷新，评测结果逐位不变）见 [改 kernel 后 bench 复评须重跑适配并重生成 op_plugin 卡](../validations/bench_reeval_requires_readapt_plugin_rewrite.md)——构建层与评测层构成"改动未生效"的两级检查点。