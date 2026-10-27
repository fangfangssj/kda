---
type: "Runbook"
title: "ld.lld cannot open _mix_aiv 报错判别"
description: "cube/vec 双段编译下 ld.lld 报 cannot open _mix_aiv_*.o（或 _mix_aic_*.o）实为上游编译失败、.o 未生成，需过滤链接噪声定位真实编译错误；stdout 与 kernel_meta 日志全空时，清 kernel_meta 缓存强制重编方见真错。单 TU 多变体形态（kernel_N/bin_param）下顶层只报 ld.lld + python traceback，真实 C++ error: 行被淹没——全量日志 grep 或按 .make/bin_param 钻取单变体；-j8 并行构建下前端错误可被完全淹没，build.sh -j1 串行重跑并 grep error 是可靠探针。"
tags: ["task.debugging", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-17T03:42:43Z"
updated_at: "2026-09-17T02:21:59Z"
platforms: ["950"]
---

# ld.lld cannot open _mix_aiv 报错判别

## 现象

在对 `flash_attn_gqa` arch35 的 cube/vec kernel 做重构（抽取 `FlashAttentionLoop` 公共循环、`FALoopState` 状态控制 struct 统一 cube/vec，不经继承）后，运行固定构建命令：

```
bash build.sh --pkg --ops="flash_attn,flash_attn_metadata" --soc=ascend950
```

构建不以"AIC 编译错误"收尾，而是以链接器 `ld.lld` 的一长串 `cannot open ... .o: No such file or directory` 报错收尾，且**全部缺失对象都是 `_mix_aiv_*.o`**（vec 侧产物），`_mix_aic_*.o` 不在缺失列表中。原始报错（verbatim，trace 内 tool_observed）：

```
[2026-06-16 10:00:14] ld.lld: error: cannot open /mnt/workspace/<user>/ops-transformer_cv/build/binary/ascend950/gen/kernel_meta_FlashAttn_7a9b290f0be5028cb35fc6d409953d1d/kernel_meta/FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_mix_aiv_196864.o: No such file or directory
[2026-06-16 10:00:14] ld.lld: error: cannot open /mnt/workspace/<user>/ops-transformer_cv/build/binary/ascend950/gen/kernel_meta_FlashAttn_7a9b290f0be5028cb35fc6d1d/kernel_meta/FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_mix_aiv_512.o: No such file or directory
[2026-06-16 10:00:14] ld.lld: error: cannot open /mnt/workspace/<user>/ops-transformer_cv/build/binary/ascend950/gen/kernel_meta_FlashAttn_7a9b290f0be5028cb35fc6d409953d1d/kernel_meta/FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_mix_aiv_131584.o: No such file or directory
[2026-06-16 10:00:14] ld.lld: error: cannot open /mnt/workspace/<user>/ops-transformer_cv/build/binary/ascend950/gen/kernel_meta_FlashAttn_7a9b290f0be5028cb35fc6d409953d1d/kernel_meta/FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_mix_aiv_66048.o: No such file or directory
[2026-06-16 10:00:14] ld.lld: error: cannot open /mnt/workspace/<user>/ops-transformer_cv/build/binary/ascend950/gen/kernel_meta_FlashAttn_7a9b290f0be5028cb35fc6d409953d1d/kernel_meta/FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_mix_aiv_197120.o: No such file or directory
```

报错看起来像"链接器找不到文件"的环境/路径问题，但实际是上游 AIV 编译阶段失败、`.o` 根本没生成。

## 适用场景

- **工程形态**：构建/编译（`bash build.sh --pkg --soc=ascend950`），非 kernel 直调、非 aclnn、非评测。
- **API 路径**：CANN 算子仓 `build.sh` + `kernel_meta_FlashAttn_<hash>/` 产物布局；cube/vec 双段编译（AIC 产 `_mix_aic_*.o`、AIV 产 `_mix_aiv_*.o`，最终 ld.lld 链接为 FATBIN）。
- **验证过的平台与 CANN 版本**：ascend950（构建命令 `--soc=ascend950` 实跑，trace 内 tool_observed）；CANN 版本未在 trace 中显式记录。
- **适用粒度**：`platform_wide`（同一 build.sh + 双段编译模型下，所有 cube/vec 算子仓构建均适用，不限定 flash_attn）。
- **路线优先级**：`recommended_fastpath`（命中该报错时首选"过滤链接噪声、定位 AIV 编译错误"判别，而非排查路径/权限/重装工具链）。
- **不适用或未验证场景**：单段编译（无 AIV 阶段）的算子仓不适用；非 ascend950 的 SoC 未验证（见 `## 失效边界`）。
- 探针升级分支（独立复现 R2，BenchHarnessEngramGateFusion，CANN 9.2.0）：原探针（`grep -v ld.lld` 滤噪/翻 kernel_meta 日志）存在盲区——stdout 与 kernel_meta 构建日志 `grep -i "error:"`、`grep -i "error|fatal|undefined|failed"` 均无输出、真错被陈旧 kernel_meta 缓存完全吞掉时，须升级为删缓存目录强制全量重编：`rm -rf operators/<pkg>/build/<Op>_<soc>/kernel_0/kernel_meta_<hash>_<hash>` 后 `bash operators/build.sh <pkg>`，重编后真实编译错误直接出现在 stdout；全量重编有额外编译时间成本（大算子包更明显），只应在"错误不可见"时使用。
- 单 TU 多变体分支（独立复现 R3，BenchNoHarnessConv2d，CANN 9.1.0）：非 cube/vec 双段、而是按变体拆 kernel_N 目标的构建（`build/op_kernel/<OpType>_<soc>/<op>.make` 列 `bench_no_harness_conv2d0/1/2` 三个 target，各在 `kernel_N` 目录调 `bin_param/<OpType>-<op>-N.sh`）同样命中本判别——顶层输出只显示 python 异常链（`Exception: ... CompileStage.FATBIN, msg is ld.lld: error: cannot open ..._1.o`），无任何 C++ error: 行；真实错误（如 `no matching function for call to '<op>_0_tilingkey'`）一直埋在同一份构建输出里，被 `tail` 截断与 traceback 淹没。探针一（首选）——全量重建并 grep 错误行，不要只 tail：`bash build.sh --soc=ascend950 -j8 2>&1 | grep -E "error:|Error|INFERCHANNEL|FATBIN|not implement|fatal" | head -30`。探针二（钻取单变体）——读 `.make` 找到每个变体的编译脚本手动执行：`cd build/op_kernel/<OpType>_<soc>/kernel_1 && bash ../bin_param/<OpType>-<op>-1.sh --kernel-src=<op_kernel 源文件路径>`；注意探针二的坑：脱离完整构建流程手动单跑脚本，可能先撞上 python 侧错误而不是 kernel 错误（`ModuleNotFoundError: No module named 'impl.dynamic'`，op_manager.py 的 importlib 链路——`impl.dynamic` 模块由构建流程生成/注入路径，独立执行上下文不可见），遇到它应回到探针一。
- 并行掩盖分支（独立复现 R4，BenchNoHarnessConv3dBackpropFilter 混合双核 kernel，CANN 9.1.0）：`-j8` 并行构建下 `-mix_aic_<N>.o` 缺失形态（AIC 段失败）同样命中本判别，且前端错误在并行输出里**完全不可见**（不止被淹没，是根本没打印）——失败产物不落盘（构建目录内 `find *.log/*.err/*.mk` 全空），翻日志无门。探针（首选）：`bash build.sh --soc=ascend950 -j1 2>&1 | grep -i "error" | head -10`——`-j1` 让失败发生在第一个编译单元，真实错误完整输出；代价是一次慢构建，仅出现该报错形态时使用。判别要点：FATBIN `cannot open ..._mix_aic_*.o`（N 常为 27/283/1/257 等核号）+ 无源码行号错误 = "源码有编译错"的强信号，不要先怀疑构建目录/缓存/磁盘。

## 判别方法

### 触发特征

`bash build.sh ... --soc=<soc>` 末尾出现一长串 `ld.lld: error: cannot open .../kernel_meta_<Op>_<hash>/kernel_meta/<Op>_<hash>_mix_aiv_*.o: No such file or directory`，且缺失对象**仅 `_mix_aiv_*.o`**（vec 侧）。`_mix_aic_*.o` 不在缺失列表 = AIC（cube）段已编译通过。

（增量 R4）对称形态同样成立：缺失对象为 `_mix_aic_<N>.o`（N 为核号，如 27/283）时，是 AIC 段源码编译失败、其 `.o` 未生成——判读逻辑不变（哪段 `.o` 缺失就是哪段失败），且 `-j8` 并行下该形态的前端错误可能完全不被打印，`-j1` 串行重跑是首选探针。

### 可执行探针

重跑构建并过滤掉 ld.lld 链接噪声，只留真实编译错误行。trace 内实际使用的探针命令（verbatim，tool_observed）：

```
bash build.sh --pkg --ops="flash_attn,flash_attn_metadata" --soc=ascend950 2>&1 \
  | grep -E "error|Error|ERROR" \
  | grep -v "ld.lld" \
  | grep -v "Traceback" \
  | grep -v "opc_common" \
  | head -40
```

- `grep -v ld.lld` 是关键——它剥掉"cannot open _mix_aiv_*.o"链接噪声，暴露上游 AIV 编译错误（如 `use of undeclared identifier`、`narrowing`、`static_cast` 相关报错）。
- 若该探针无输出，说明真实 AIV 错误行被 build.sh 的 `tail`/`head` 截断或落在 `opc_common`/`Traceback` 过滤分支里；改用不截断的全量重定向 `2>&1 | tee /tmp/build.log` 后再 `grep -v ld.lld` 检索，或直接翻 `build/binary/ascend950/gen/kernel_meta_<Op>_<hash>/*.log`。

### 观测

构建末尾出现 `ld.lld: error: cannot open .../kernel_meta_FlashAttn_<hash>/kernel_meta/FlashAttn_<hash>_mix_aiv_*.o: No such file or directory`，缺失对象仅 `_mix_aiv_*.o`、`_mix_aic_*.o` 不在缺失列表。探针命令实跑（命令执行本身为 tool_observed；其 grep 输出因 trace redaction 未在事件流恢复）。

### 判读（interpretation）

`_mix_aiv_*.o` 缺失 = AIV（vec）编译阶段失败、未产出 `.o`；链接器在 FATBIN 阶段拿不到 vec 产物，于是报 `cannot open`。报错文本本身**不包含**真实 AIV 编译错误（如 undeclared identifier），真实错误在 ld.lld 行**之前**的编译日志里。`_mix_aic_*.o` 不缺失即说明 cube 段已过。

本判别方法在 trace 中两次命中：

- **命中 A**：把 vec.h 中 `maskInfo.maskValue = negativeIntScalar;`（成员标量）改成 `maskInfo.maskValue = *((uint32_t *)&static_cast<T>(-1000000000000.0));`（对标量取地址做 type-pun）后，构建以 `ld.lld cannot open _mix_aiv_*.o` 收尾。探针 grep 当时未滤出真实错误（截断/过滤所致），据相关性推断为取地址 type-pun 触发 AIV 失败（agent_reported 推断，非直接错误原文佐证），改回 `-3.4028235e38F` 浮点字面量与已有 `constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF` 后构建通过。
- **命中 B**：抽取 `FlashAttentionLoop` 公共循环后，vec.h 使用 `sectionNum_`（Init 中 `sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];`、Process 中 `for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++)`）但类成员块缺失该声明，构建以 `ld.lld cannot open _mix_aiv_*.o` 收尾。探针 grep 滤出 `sectionNum_` / `GetFDSectionInfo` undeclared（agent_reported 转述，原始 grep tool_result 被 redact），补 `uint32_t sectionNum_ = 0;` 成员声明后构建通过。

## 根因

AscendC 算子仓采用 **cube(AIC) / vec(AIV) 双段独立编译**：两段各自产出 `_mix_aic_*.o` 与 `_mix_aiv_*.o`，再由 `ld.lld` 链接为 FATBIN。当 AIV 段编译失败时，其 `.o` 不生成，链接阶段便以 `cannot open ... _mix_aiv_*.o` 收尾——这一链接报错**掩盖**了上游 AIV 段的真实编译错误（undeclared identifier / narrowing / 不支持的 type-pun 等）。`_mix_aic_*.o` 不缺失即说明 cube 段已过。

> 注：双段编译模型由 verbatim ld.lld 报错中的 `_mix_aiv_*.o` 命名 + "No such file" + agent reasoning 共同推断，非直接 spec 工具输出；AIC/AIV 错误原文在会话事件流中被脱敏，未逐字恢复。

## 验证信号

来源说明：本知识来自一次 flash_attn_gqa arch35 cube/vec 重构会话；构建命令 `bash build.sh --pkg --ops="flash_attn,flash_attn_metadata" --soc=ascend950` 实跑，ld.lld 链接报错为 verbatim tool_observed 输出，探针命令实跑但 grep 输出被 redact 未恢复。

- 构建末尾出现 `ld.lld: error: cannot open .../kernel_meta_FlashAttn_<hash>/kernel_meta/FlashAttn_<hash>_mix_aiv_*.o: No such file or directory`，且缺失对象仅 `_mix_aiv_*.o`、`_mix_aic_*.o` 不在缺失列表（来源平台：950；证据等级：tool_observed）
- 探针命令 `bash build.sh ... 2>&1 | grep -E "error|Error|ERROR" | grep -v ld.lld | grep -v Traceback | grep -v opc_common | head -40` 实跑（来源平台：950；证据等级：tool_observed）

> 两次命中的真实 AIV 编译错误原文与两次修后的"构建成功"工具输出在 trace 内被 redact，未落入事件流，故不进 `## 验证信号`；`## 判别方法` 中的转述均标注为 agent_reported。

- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：清缓存前 `build.sh` 失败输出仅含 `ld.lld: error: cannot open ..._0.o/_1.o: No such file or directory`，stdout 与 kernel_meta 构建日志 grep `error`/`fatal`/`undefined`/`failed` 均无输出（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：清 `kernel_meta_<hash>` 缓存目录强制全量重编后，同命令输出真实错误 `op_kernel/arch35/<op>_kernel.h:869:13: error: no matching function for call to 'DataCopyPad'`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（BenchNoHarnessConv2d，CANN 9.1.0，单 TU 多变体）：`cat .../bench_no_harness_conv2d.make` 显示 kernel 编译拆为三 target、各调 `bin_param/...-N.sh`（脚本核心是 `asc_opc ... --main_func=... --soc_version=Ascend950PR_9599 --op_mode=dynamic`）；kernel_1 目录手动执行脚本报 `ModuleNotFoundError: No module named 'impl.dynamic'`（未现出 kernel 错误，属探针二的坑）；全量重建 + `grep -E "error:|..."` 过滤后输出 `no matching function for call to 'bench_no_harness_conv2d_0_tilingkey'`（kernel_1/kernel_2 的自动生成 wrapper）及 FATBIN ld.lld 报错，证明真实错误一直在完整日志里（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（BenchNoHarnessConv3dBackpropFilter，CANN 9.1.0，并行掩盖）：`rm -rf build/ && bash build.sh --soc=ascend950 -j8` 失败输出仅含 `CompileStage.FATBIN` + `ld.lld: error: cannot open ..._mix_aic_27.o / _mix_aic_283.o: No such file or directory`，无任何源码行号错误；构建目录内 `find *.log/*.err/*.mk` 全部为空（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4：同代码 `bash build.sh --soc=ascend950 -j1 2>&1 | grep -i "error"` 输出 7 个源码级 `error:`（`use of undeclared identifier 'optiling'`（tiling_data.h:79）、`use of undeclared identifier 'ASCubeObjConfig'`（kernel 入口:38）、static assertion / constexpr if 非常量等），`7 errors generated`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4：修复源码后 `-j8` 构建成功（`CPack: - package: ... successfully created`），反证此前的 FATBIN 报错确为源码错误所触发（来源平台：950；证据等级：tool_observed）。

## 坏实践

- 把 `ld.lld: cannot open _mix_aiv_*.o` 当成"路径/权限/工具链未装"去排查——命中 A 中 agent 连续跑了 8 条 grep（按 hash、按 `aiv_1189744_65536`、按 `narrowing|overflow` 等）都滤不出真实错误，因为没先 `grep -v ld.lld` 剥链接噪声。正确第一步是过滤 ld.lld 行、看上游编译日志。
- 对标量做 `*((uint32_t *)&static_cast<T>(-1e12))` 取地址 type-pun 来构造掩码常量——在 AscendC vec kernel 上既触发 scalar 寄存器→内存 spill，又可能直接被 AIV 段编译拒绝。应改用浮点字面量（`-3.4028235e38F`）或已有 `constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF` 位模式常量。
- 只 grep 现存 kernel_meta 日志就下结论"没有编译错误"——缓存目录里的日志可能根本不对应本次失败；日志全空时须清缓存强制重编（见适用场景的升级分支）。
- （增量 R3）只看 `tail -60`/`tail -80` 的顶层输出就下结论"工具链坏了/链接器坏了"：真实 error: 行往往在日志中段，必须全量 grep。
- （增量 R3）在未完成一次完整构建流程前就手动单跑 bin_param 脚本：会先撞 `impl.dynamic` 导入错误，浪费一轮排查。
- （增量 R4）看到 `cannot open ... .o: No such file or directory` 就去重建目录、查磁盘、怀疑 `rm -rf build/` 时机——`-j8` 形态下构建失败产物不落盘，翻日志一无所获；应直接 `-j1` 重跑。
- （增量 R4）在 `-j8` 输出里反复搜索源码错误——并行模式下它可能根本没被打印。

## 误诊

- 命中 A 中 agent 一度把 `ld.lld cannot open _mix_aiv_*.o` 误判为"链接器/环境问题"，连跑 8 条 grep 找真实错误无果；后据"插入取地址 type-pun 前构建通过、插入后失败"的相关性，推断为该 type-pun 触发 AIV 失败（agent_reported，非直接错误原文佐证）。
- 命中 B 中 agent 一度推测 `sectionNum_` 缺失是"预存 bug、与本次重构无关"（agent_reported），但未做 git 历史核验，不能排除是 `FlashAttentionLoop` 抽取时遗漏声明。

## 失效边界

- **未验证，推断可迁移**：a2 / a3 等 SoC 同样使用 `build.sh` + cube/vec 双段编译模型，本判别方法（`grep -v ld.lld` 定位 AIV 编译错误）推断可迁移；但本开发会话 仅在 ascend950 实跑构建，a2/a3 未验证，未写入 frontmatter `platforms`。
- 单段编译算子仓（无 AIV 阶段、不产 `_mix_aiv_*.o`）不适用本判别。
- 若构建系统未来改为 AIV 失败即整体 fail-fast、不再进入 ld.lld 链接阶段，本"链接报错掩盖编译错误"判别即失效。
- 升级分支的边界：若清缓存重编后仍只见 .o 缺失，则属真实产物链问题（磁盘满/权限/链接器），本判别不适用。kernel_meta 缓存跳过重编的确切条件未读到实现源码（`ascendc_kernel_cmake`/opc），"陈旧缓存吞掉真错"的机制解释保持行为级推断；遗留：读 opc 相关源码确认缓存判定条件，把行为层结论补成机制层结论。
- （增量 R3）单 TU 多变体分支绑定 CANN 9.1.0 的 `asc_opc`/`asc_op_compile_base` 工具链与 build 目录布局（`build/op_kernel/<OpType>_<soc>/kernel_<N>` + `bin_param/*.sh`）；目录结构变化时探针二的路径需相应调整。变体数量由算子模板参数/tiling key 组合决定，不同算子不同。`impl.dynamic` 手动单跑失败的确切根因（模块由哪个构建步骤生成、正确的独立执行环境如何构造）未追查，只记录现象与绕过方式。
- 相邻卡片：[FATBIN wrapper 参数数量卡](fatbin_wrapper_param_count_matches_opdef_input_plus_output.md)——另一类『.o 缺失的上游因』：in-place kernel 缺输出参数致 CCEC 编译失败，判别入口同为翻真正的 `error:` 行。[kernel 二进制缓存卡](kernel_binary_cache_printf_hash_toggle.md)——运行期侧的缓存变体：源码改动不进 device binary hash、NPU 跑旧 binary，加 PRINTF 改 hash 强制加载；与本卡的编译期 kernel_meta 缓存区分。
