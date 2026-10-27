---
type: "Runbook"
title: 重编未重装 vendors 旧 tiling 致 561103
description: 源码升级多 TilingKey 后只重编了 .run 未重装，vendors 里仍是旧版 tiling（只支持 fp16），ST 全部 fp32/bf16 用例 GetWorkspaceSize 返回 561103 而 UT 全过（UT 直接编译加载源码 tiling，与安装树不是同一份二进制）；plog 错误文案含当前源码已不存在的字样 + stat 时间线即可确诊，重编重装即恢复。
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# 重编未重装 vendors 旧 tiling 致 561103

## 现象

BenchNoHarnessDepthwiseConv2d（Ascend950PR，CANN 9.2.0，8 卡 NPU 环境）多 dtype 扩展联调：源码已完成多 TilingKey 升级（op_host tiling 支持 fp16/fp32/bf16 三 dtype，op_kernel 新增 fp32/bf16 路径），`bash build.sh --soc=ascend950` 编译通过并产出含 3 个 kernel binary 的新 `custom_opp_ubuntu_x86_64.run`，UT 37/37 全过。但 ST 实跑 `bash tests/st/run.sh --real --level L0+L1`（43 条用例）中**所有 fp32/bf16 用例**（共 9 条）在 `aclnnBenchNoHarnessDepthwiseConv2dGetWorkspaceSize` 阶段统一返回 **561103**（ACLNN_ERR_INNER_NULLPTR），fp16 用例则大部分正常（18 过 / 10 败 / 15 跳过，通过率 64.3%）。排查中确认 op_kernel 3 个 binary 都在、binary_info_config.json 的 dtype 映射正确、TilingKey 模板注册正确——静态产物全对，动态行为却像"没升级过"。

## 适用场景

- 工程形态：registry-invoke 自定义算子仓的 ST 联调（NPU 实跑），流程为 build.sh 编译 → `.run` 安装到 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/` → ST 调 aclnn。
- API 路径：`custom_opp_ubuntu_x86_64.run` 安装器；vendors 安装树（op_impl/op_tiling libcust_opmaster_rt2.0.so 等）；`ASCEND_GLOBAL_LOG_LEVEL=1 + ASCEND_SLOG_PRINT_TO_STDOUT=1` 的 plog。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（NPU 实跑验证）。
- 适用粒度：workflow 之上的平台级事实——坑在"编译≠安装"的部署流程，任何走 .run 安装流水的自定义算子都可能踩。
- 路线优先级：recommended_fastpath（重装一次即可解，无代码改动）。
- 明确不适用 / 未验证场景：从未安装过算子包的全新环境（首装不存在新旧错位）；用 CMake 直接加载 build 产物而非 vendors 安装树的调试路径未验证。

## 根因

**重编译生成了新 `.run`，但没有重新执行安装**——`opp/vendors/` 下的 tiling 动态库（`libcust_opmaster_rt2.0.so`）还是上一版旧版：旧 tiling 只认 fp16，遇到 fp32/bf16 在 tiling 阶段报"iter1 only supports fp16"失败，上层 L0 API 返回 nullptr，aclnn 层表现为 561103。UT 全过的原因是 UT 直接编译并加载**源码** tiling，与 vendors 安装树无关——UT 与 ST 的被测对象在这个流程里不是同一份二进制。

确诊证据链（两条互补信号）：

1. **plog 错误文案含当前源码已不存在的字样**：开 plog 重跑，抓到
   `[ERROR] OP(…):[…/op_host/arch35/bench_no_harness_depthwise_conv_2d_tiling.cpp:278][OPS_BASE][BenchNoHarnessDepthwiseConv2dTilingFunc] … iter1 only supports fp16, got dtype=27`
   而当前源码的 tiling.cpp 里早已没有 "iter1 only supports fp16" 这句话（已改为 "iter2 only supports stride=(1,1)…"，且 278 行是 hTile 计算）——报错文案来自旧二进制。
2. **stat 时间线**：`op_host/arch35/…_tiling.cpp` 源码 mtime 19:21:41 → build 产物 `.so` 19:22 / `.run` 19:23，但 vendors 内 `libcust_opmaster_rt2.0.so` mtime 停在 19:07:57（上一版安装时刻）——安装树落后源码两轮修改。

## 解决配方

判别流程（命令级，可直接复用）：

```bash
# 1. 时间线对照：源码/构建产物/安装树三者 mtime
stat -c "%y" operators/<op>/op_host/arch35/<op>_tiling.cpp
stat -c "%y" operators/<op>/build/op_host/libcust_opmaster_rt2.0.so
stat -c "%y" $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so
# 源码 > build 产物 > 安装树 mtime 即存在新旧错位

# 2. plog 铁证：错误文案是否为当前源码所有
export ASCEND_GLOBAL_LOG_LEVEL=1; export ASCEND_SLOG_PRINT_TO_STDOUT=1
bash tests/st/run.sh --real --level L0+L1 2>&1 | tee /tmp/st_full_log.txt
grep -n "ERROR" /tmp/st_full_log.txt | grep -i "tiling\|not support\|iter"
# 报出的文案/行号在当前源码中找不到 → 装的是旧包
```

修复动作（干净重建 + 重装 + 验证安装树已更新）：

```bash
bash build.sh --make_clean                       # 清掉旧构建目录
bash build.sh --soc=ascend950 -j8                # 全量重编
bash build/custom_opp_ubuntu_x86_64.run --quiet  # 重装
stat -c "%y" $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so
# → 应刷新为刚才的安装时间
```

## 设计决策

- 用 `build.sh --make_clean` 全量重建而不是直接重装现有 `.run`：`.run` 虽是新产物可直接装，但既然怀疑构建-安装链路不同步，全量重建+重装可同时排除"构建缓存残留"与"安装树陈旧"两个变量，一次闭环。
- 判别优先走 plog 文案 + stat 时间线而不是反复读源码比对逻辑：两者都是客观产物证据，比"重新审一遍 tiling 代码"快且无歧义。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 把"重装 .run"写进每次源码改动后的联调 checklist：build.sh 成功 ≠ vendors 已更新；改完 op_host/op_kernel 必须重跑 `.run` 安装再跑 ST。
- 联调脚本可在 ST 前加一步安装树 mtime 与源码 mtime 的对比告警（源码更新于安装树即提示重装）。
- UT 全过 + ST 全新 dtype 失败的组合，先查部署一致性（安装树新旧），再查 kernel 逻辑。

## 验证信号

以下信号来自该算子多 dtype 扩展联调（一次被中断后恢复的完整排查链）。

- 失败信号：ST 43 条中全部 9 条 fp32/bf16 用例 GetWorkspaceSize 返回 561103（fp16 大部分正常，18/43 通过率 64.3%），同期 UT 37/37 全过（来源平台：950；证据等级：tool_observed）。
- 探针信号①：plog 抓到 `tiling.cpp:278 … iter1 only supports fp16, got dtype=27`，该文案在当前源码已不存在（来源平台：950；证据等级：tool_observed）。
- 探针信号②：stat 时间线 tiling.cpp 19:21:41 / build .so 19:22 / .run 19:23 / vendors libcust_opmaster_rt2.0.so 19:07:57（来源平台：950；证据等级：tool_observed）。
- 修复信号：`--make_clean` 重编 + 重装后 vendors `.so` mtime 刷新为 20:05:07，重跑 ST 后 fp32/bf16 用例不再返回 561103、进入实际 kernel 执行与精度比对阶段（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只解决"561103 由旧安装树引起"的分支；重装后该算子 fp32/bf16 仍有独立精度失败（另案），不因重装消失。
- `--make_clean` 全量重建成本较高；确认 `.run` 本身已是新产物时直接重装即可（本会话为稳妥走了全量）。
- 多个 vendor 包并存时的安装树相互覆盖行为未验证。

**未决项**：

- 无（部署不一致闭环完整；重装后暴露的精度问题属另一根因）。
## 坏实践

- "build.sh 绿了就直接跑 ST"——编译与安装是两步，中间断链不会有任何报错提示。
- 看到 561103（nullptr 类错误）就直接扎进 kernel/TilingKey 代码查逻辑，跳过部署一致性检查——本会话一度在 binary_info_config.json/simplifiedKey 的静态比对上花了大量时间，而静态产物本来就是对的。

## 误诊

- 第一轮排查曾怀疑 TilingKey 模板注册或 binary_info_config 的 dtype 映射错误：逐项检查 3 个 .o、binary_info_config.json、SEL 注册全部正确——静态产物无误，问题只在"跑的不是这份产物"。
- 曾怀疑 L0 API（autogen `aclnn_bench_…conv2d.cpp`）对 fp32/bf16 未处理：读代码确认 L0 API 无 dtype 限制，排除。
