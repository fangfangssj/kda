---
type: "Runbook"
title: TEFUSION二进制匹配走全局OPP非staging
description: 用 ASCEND_CUSTOM_OPP_PATH 指向 /tmp staging 做 GE 图模式隔离验证时，TEFUSION 的 binary_info_config 搜索仍走 CANN 安装目录下的全局 OPP（opp/vendors/），不随环境变量切换到 staging；staging 隔离只覆盖 GE 插件（SO）加载与 FE DSL 路径——删除全局安装的负向实验证明 staging 从不进入 TEFUSION 二进制搜索，处置是同步重装保持全局与 staging 一致。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:16:44Z'
updated_at: '2026-09-14T12:16:44Z'
platforms: ["950"]
---

# TEFUSION二进制匹配走全局OPP非staging

## 现象

在 Ascend950PR（CANN 9.2.0）上做自定义算子的 GE 图模式（GEIR）隔离验证：测试脚本把算子包 stage 到 `/tmp/geir_opp.XXXXXX/vendors/<Op>` 并 `export ASCEND_CUSTOM_OPP_PATH=<staging>`，期望整条链路只消费本轮 staging 产物。route 测试跑完后翻 route debug 日志，发现同一次编译里来源分裂——GE 侧的 DSL/注册来自 staging，而 TEFUSION 的二进制配置全部指向 CANN 安装目录下的全局 OPP：

```
GetOpDslFilePath:Op dsl file path of op[n_route_target, BenchHarnessScatter] is [/tmp/geir_opp.qrzCmP/vendors/BenchHarnessScatter/op_impl/ai_core/tbe/BenchHarnessScatter_impl]
SetBinaryConfigPaths The current file path is /workspace/<user>/Ascend/cann-9.2.0/opp/vendors/BenchHarnessScatter/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths tbe current imply_type is [4294967298], binary_config_path is [/workspace/<user>/Ascend/cann-9.2.0/opp/vendors/BenchHarnessScatter/op_impl/ai_core/tbe/kernel/config/].
```

当时机器的全局 OPP 里恰有一份早前安装的同名算子（安装时间与本轮 staging 不同），二进制配置与 staging 产物可能漂移。为验证这一点，删除全局安装（`sudo rm -rf $ASCEND_HOME_PATH/opp/vendors/<Op>`）后重跑 route：日志里 TEFUSION 依次遍历其余全局 vendors（customize、ScaleCustom、custom_ops、EuclideanNorm、AddCustom、mla_prolog_custom、built-in），始终不扫 `/tmp` staging 目录，route 行为不变（仍 TBE 动态编译回退）。最终以重装 run 包（`echo "yes" | ./custom_opp_ubuntu_x86_64.run --install`）恢复全局安装，保持全局与 staging 同步。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：GE 图模式（GEIR）隔离验证 / 算子仓开发中 staging 隔离安装（`ASCEND_CUSTOM_OPP_PATH` + /tmp vendor 目录）的覆盖面排查。
- API 路径：`ASCEND_CUSTOM_OPP_PATH` 环境变量、TEFUSION `SetBinaryConfigPaths`/`binary_info_config.json`、GE `GetOppPluginPathNew`/`GetAllCustomOpApiSoPaths` 插件加载、FE `GetOpDslFilePath`、custom_opp run 包安装。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本的 TEFUSION/GE 路径解析行为，与具体算子无关。
- 路线优先级：safety_fallback（处置 = 保持全局 OPP 与 staging 同步重装；无让 TEFUSION 直接读 staging 的办法）。fallback 声明：同步重装是绕行而非根治，接受"GE 图路径二进制匹配以全局安装为准"；它不是行为等价的替代，只是让全局安装内容与本轮构建一致，消除"旧全局安装冒充本轮构建"的风险。
- 明确不适用 / 未验证场景：ACLNN 直调路径的 SO 加载（`GetAllCustomOpApiSoPaths` 遵循 ASCEND_CUSTOM_OPP_PATH，日志可见）；纯 host 侧注册验证（InferShape/InferDataType 等经 plugin_path 前插 staging，隔离有效）。

## 判别方法

**触发**：GE 图模式验证需要确认"本轮下发的 kernel/二进制到底来自哪里"，或怀疑全局旧安装污染 staging 测试。

**探针**（grep route debug 日志，本会话走通）：

```bash
grep -n "SetBinaryConfigPaths\|GetOpDslFilePath\|GetOppPluginPathNew\|ASCEND_CUSTOM_OPP_PATH" "$ROUTE_LOG" | head -20
```

**原始观测**（三个判别点）：

1. 二进制配置路径（TEFUSION）：`SetBinaryConfigPaths ... /workspace/<user>/Ascend/cann-9.2.0/opp/vendors/<Op>/op_impl/ai_core/tbe/kernel/config/`——全局 OPP；
2. DSL/impl 路径（FE）：`GetOpDslFilePath ... /tmp/geir_opp.qrzCmP/vendors/<Op>/op_impl/ai_core/tbe/<Op>_impl`——staging；
3. 插件路径（GE）：`GetAllCustomOpApiSoPaths:value of env ASCEND_CUSTOM_OPP_PATH is /tmp/geir_opp.uWx9bw/vendors/<Op>`；`GetOppPluginPathNew:plugin_path is '/tmp/geir_opp.uWx9bw/vendors/<Op>/op_proto/:/workspace/.../opp/vendors/<Op>/op_proto/:...'`——staging 排在最前，但全局路径仍拼接在链上。

**负向实验**（可选、侵入性大）：删除全局安装后重跑，观察 `SetBinaryConfigPaths` 是否出现 staging 路径——本会话实测不出现（只遍历其余全局 vendors），证明 staging 从不进入 TEFUSION 二进制搜索。

**解释**：`ASCEND_CUSTOM_OPP_PATH` 只把 staging 注入 GE 插件（op_proto/opmaster SO）加载链与 FE 的 DSL/impl 路径，TEFUSION 的 `binary_info_config.json` 搜索仍按 CANN 安装布局（`$ASCEND_HOME_PATH/opp/vendors/`）枚举。因此 staging 隔离对"注册/推导"有效、对"二进制匹配"无效；全局安装与 staging 内容漂移时，GE 图路径二进制匹配的结果不代表本轮构建。

## 设计决策

- 判别手段选择"读日志路径"而非"diff 二进制内容"：`SetBinaryConfigPaths`/`GetOpDslFilePath` 直接打印来源目录，一行日志即可定性，比对比 .o 文件 hash 更快。
- 负向实验（删除全局安装）成本高且有权限障碍（root 属主文件需 sudo），只做一次用于证伪"staging 会被 TEFUSION 扫到"。
- 恢复策略选择重装 run 包保持全局与 staging 同步，而不是删除后置之不理：全局安装还服务于机器上其他路径（ACLNN 回归等）。
- 结论表述上区分三层来源：GE 插件（staging 优先）、DSL/impl（staging）、TEFUSION 二进制配置（全局）。
- 当前方案路线档位：safety_fallback（无根治手段，同步重装是验证过的可用处置）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0（`/workspace/<user>/Ascend/cann-9.2.0`）。
- 相关文件：staging 目录 `/tmp/geir_opp.*/vendors/<Op>`（run.sh stage 产物）、全局 `$ASCEND_HOME_PATH/opp/vendors/<Op>/`（custom_opp_ubuntu_x86_64.run 安装产物）、route debug 日志、`build/custom_opp_ubuntu_x86_64.run` run 包。
- run 包安装命令（EULA 交互用 yes 管道）：

```bash
echo "yes" | operators/<Op>_package/build/custom_opp_ubuntu_x86_64.run --install
```

## 预防

- GE 图模式验证前先 grep 一次 `SetBinaryConfigPaths`：确认二进制配置来源与 staging 是否一致；不要默认 `ASCEND_CUSTOM_OPP_PATH` 覆盖了整条链路。
- 每轮 staging 验证后同步重装全局 OPP（或明确记录本轮二进制匹配将读到哪份全局安装），避免用旧全局安装的匹配结果代表新构建；改了 host 代码只重建 staging 就重跑尤其危险——binary_info 从 installed vendor 读，旧 installed SO 会以 "registered first" 抢先注册回调。
- 结论表述上区分三层来源：GE 插件（staging 优先）、DSL/impl（staging）、TEFUSION 二进制配置（全局）。

## 验证信号

来源说明：以下信号来自该算子 GEIR 图模式实现验证会话：staging 隔离 route 测试的日志路径检查与一次全局安装删除对照实验，均为 950 上板实测。

- 同一次 route 编译中来源分裂：`SetBinaryConfigPaths ... /workspace/<user>/Ascend/cann-9.2.0/opp/vendors/BenchHarnessScatter/.../config/`（全局）与 `GetOpDslFilePath ... /tmp/geir_opp.qrzCmP/vendors/BenchHarnessScatter/...`（staging）并存（来源平台：950；证据等级：tool_observed）。
- 删除全局安装后：route 仍 EXIT=1，`SetBinaryConfigPaths` 只出现 customize/ScaleCustom/custom_ops/EuclideanNorm/AddCustom/mla_prolog_custom/built-in 等全局路径，无任何 /tmp staging 路径（来源平台：950；证据等级：tool_observed）。
- 插件链：`GetAllCustomOpApiSoPaths:value of env ASCEND_CUSTOM_OPP_PATH is /tmp/geir_opp.uWx9bw/vendors/BenchHarnessScatter`，plugin_path 把 staging op_proto 排在全局之前（来源平台：950；证据等级：tool_observed）。
- 重装 run 包后全局 OPP 恢复（安装日志 `create /workspace/<user>/Ascend/cann-9.2.0/opp/vendors/BenchHarnessScatter/op_proto...`），后续验证继续（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.2.0 / Ascend950PR 实测；TEFUSION 路径解析在闭源 .so 内，CANN 升级后须用本卡 grep 方法重验（未验证，机制可能变化）。
- 本卡不回答"prebuild 为什么被跳过"（即使 TEFUSION 扫到正确的 binary_info_config.json，prebuild 仍在匹配之前被 prebuildPattern=Opaque 跳过，见同批《GE图prebuildPattern=Opaque跳过预编译匹配》卡）——两卡现象相关但根因独立。
- 同批另一会话观测到重装到全局 OPP 后 vendor 仍不在 `SetBinaryConfigPaths` 清单（与本会话"本算子全局路径在列"矛盾）——扫描清单的建立机制未定案（可能与安装状态/注册步骤有关），两处观测互补不互斥：本卡证明 staging 永不进入搜索，该卡证明清缓存/重装/改 simplifiedKeyMode 三条修复路线无效。
- 删除全局安装属破坏性操作，会波及机器上依赖全局 OPP 的其他路径（本会话随即重装恢复）；复现负向实验前先确认可恢复。

**未决项**：

- TEFUSION 二进制搜索路径的完整构造规则（是否可经其他环境变量注入 staging）未在 C++ 层确认，仅观察到行为。
- 全局安装与 staging 二进制漂移时，GE 图路径若走到匹配环节会命中哪一份（本会话 prebuild 恒被跳过，未观察到实际匹配发生）。
- `SetBinaryConfigPaths` vendor 清单的建立机制（为何同版本不同会话观测不一致）未追明——同批扫描清单卡已列为该卡优先项，此处不重复展开。
## 坏实践

- 假设 `ASCEND_CUSTOM_OPP_PATH` 让整条 GE 图链路（含 TEFUSION 二进制匹配）都读 staging——实测只覆盖 GE 插件加载与 FE DSL 路径。
- staging 验证得出二进制相关结论时不检查 `SetBinaryConfigPaths` 的实际路径，让旧全局安装的结果冒充本轮构建结果。
- 删除全局安装做实验后不恢复，影响同机其他验证路径。

## 误诊

- 曾把"TEFUSION 读全局路径"当作 prebuild 被跳过的原因：删除全局安装后 prebuild 仍被跳过（prebuildPattern=Opaque 来自 ops-info 判定，与二进制路径无关）——全局路径问题是独立事实，不是 prebuild 跳过的根因（agent_reported 假设，对照实验否定）。
