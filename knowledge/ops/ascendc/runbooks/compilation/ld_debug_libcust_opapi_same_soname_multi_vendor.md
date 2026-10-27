---
type: "Runbook"
title: 多 vendor 同名 libcust_opapi 加载源判定
description: 所有 custom opp vendor 共用同一 SONAME libcust_opapi.so，LD_LIBRARY_PATH 顺序决定进程实际加载哪份；用 LD_DEBUG=libs 的 trying file= 行判定实际加载源，runtime 还按 config.ini 的 load_priority 逐 vendor dlopen。
tags: [task.debugging, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# 多 vendor 同名 libcust_opapi 加载源判定

## 现象

在多用户共享的 CANN 环境（Ascend950PR + CANN 9.2.0，`opp/vendors/` 下 6 个自定义算子 vendor）排查自定义算子 ST 测试崩溃时，需要回答一个前置问题："进程到底加载了哪一份 `libcust_opapi.so`？"——每个 vendor 的 `op_api/lib/` 下都有一份同名同 SONAME 的 `libcust_opapi.so`（各自只导出本算子的 aclnn 符号）。实测用 `LD_DEBUG=libs` 检查，发现某次运行动态链接器把 `libcust_opapi.so` 解析到了**外来用户**的 vendor 路径：

```
   1999277:	find library=libcust_opapi.so [0]; searching
   1999277:	  trying file=/workspace/<other-user>/Ascend/opp/vendors/xlog1py_custom/op_api/lib/libcust_opapi.so
   1999277:	calling init: /workspace/<other-user>/Ascend/opp/vendors/xlog1py_custom/op_api/lib/libcust_opapi.so
   1999277:	calling fini: /workspace/<other-user>/Ascend/opp/vendors/xlog1py_custom/op_api/lib/libcust_opapi.so [0]
```

即当条 shell 的 `LD_LIBRARY_PATH` 里残留了他人 vendor 库路径时，别人的 `libcust_opapi.so`（导出的是 xlog1py 符号）会抢先成为进程内的 `libcust_opapi.so`。

## 适用场景

- 工程形态：自定义算子（custom opp）ST/适配器开发与联调，涉及 `libcust_opapi.so` 加载的任何宿主进程。
- API 路径：动态链接器 `LD_LIBRARY_PATH` 解析；`libnnopbase.so` 的 vendor 自动加载机制（`GetOppVendorsFilePaths` 读取 `<ASCEND_OPP_PATH>/vendors/config.ini`，按 `load_priority` 拼 `<vendor>/op_api/lib/libcust_opapi.so` 逐个 dlopen）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（多用户共享安装）。
- 适用粒度：库解析诊断规则（跨算子的环境级机制，不针对特定 kernel 行为）。
- 路线优先级：recommended_fastpath（怀疑库加载问题时第一优先的探针）。
- 明确不适用/未验证场景：单一 vendor、无外来路径污染的环境（无歧义可判）；其他 CANN 版本的 config.ini 机制细节未验证。

## 判别方法

探针（可执行、可直接复用）：

```bash
# 1) 看动态链接器把 libcust_opapi.so 解析到哪个路径（重点看 trying file= 与 calling init: 行）
LD_DEBUG=libs ./test_aclnn_mla_prolog ../testcases/L0_test_cases.csv 2>&1 | grep -A2 "find library=libcust_opapi.so"

# 2) 纠正后复核：把目标 vendor lib 目录前置，再跑一次探针
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<your_vendor>/op_api/lib:$LD_LIBRARY_PATH
LD_DEBUG=libs ./test_aclnn_mla_prolog ... 2>&1 | grep "cust_opapi" | head -20

# 3) 佐证 runtime 的自动加载机制（strings 证实 libnnopbase 的路径拼装逻辑）
strings $ASCEND_HOME_PATH/lib64/libnnopbase.so | grep -E "load_priority|config.ini|op_api/lib|GetOppVendorsFilePaths"
cat $ASCEND_HOME_PATH/opp/vendors/config.ini   # load_priority= 依次列出所有 vendor
```

判读规则：

- `trying file=` 逐行列出搜索顺序，**第一个命中的路径就是实际加载的库**；若命中路径不是你的 vendor 目录，即为抢占。
- `libcust_opapi.so` 无版本化 SONAME 区分，所有 vendor 同名——解析完全由 `LD_LIBRARY_PATH`（及 rpath）顺序决定。
- 除链接期解析外，`libnnopbase.so` 运行期还会按 `config.ini` 的 `load_priority` 对每个 vendor 拼 `<vendor>/op_api/lib/libcust_opapi.so` 主动 dlopen；由于 SONAME 相同，先加载者占据进程内的 `libcust_opapi.so` 身份。

## 系统信息

- Ascend950PR + CANN 9.2.0，`$ASCEND_HOME_PATH/lib64/libnnopbase.so`（strings 可见 `GetOppVendorsFilePaths`、`/vendors/config.ini`、`/op_api/lib/libcust_opapi.so`、`MM_ENV_ASCEND_OPP_PATH`/`MM_ENV_ASCEND_CUSTOM_OPP_PATH` 等环境变量名）。
- `$ASCEND_HOME_PATH/opp/vendors/config.ini`：`load_priority=mla_prolog_custom,bench_no_harness_lstm_custom,custom_ops,bench_no_harness_gru_custom,BenchHarnessApplyRotaryPosEmb,BenchHarnessApplyAdamW`。

## 预防

- 跑自定义算子测试/适配器前统一用探针核一遍加载路径；run.sh 里显式 `export LD_LIBRARY_PATH=<本算子 vendor>/op_api/lib:$LD_LIBRARY_PATH` 前置，而不是追加。
- 共享机器上避免把他人 vendor 路径泄入自己的 `LD_LIBRARY_PATH`（source 他人 set_env 脚本是常见污染源）。

## 验证信号

来源说明：以下为该次排查的实测探针输出（共享环境下的真实观测）。

- 观测到外来路径抢占：`find library=libcust_opapi.so ... trying file=/workspace/<other-user>/Ascend/opp/vendors/xlog1py_custom/op_api/lib/libcust_opapi.so`，`calling init:`/`calling fini:` 均落在外来路径（来源平台：950；证据等级：tool_observed）。
- 目标 vendor 前置后：链接器按 `LD_LIBRARY_PATH` 顺序在本算子 vendor 目录命中，解析路径正确（来源平台：950；证据等级：tool_observed）。
- 机制佐证：`strings libnnopbase.so` 输出 `GetOppVendorsFilePaths` / `/vendors/config.ini` / `/vendors/` / `/op_api/lib/libcust_opapi.so`；`config.ini` 实测含 6 个 vendor 的 `load_priority` 列表（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本规则只回答"加载了哪份库"；**加载路径正确不等于无其他问题**——本例纠正路径后 ST 二进制的 double free 崩溃依旧（另有独立未定位问题，见同批 double free 卡），不要把"路径已纠正"当成修复完成。
- 探针本身有大量输出，`LD_DEBUG=libs` 需配合 grep 收窄；崩溃极早的进程可用输出重定向落盘再 grep。
- `libnnopbase.so` 运行期按 `load_priority` 逐 vendor dlopen 的完整行为（顺序、失败处理、与链接期解析的交互）仅经 strings 佐证，未逐项实验验证。
- 未验证，推断可迁移：单用户环境同样可用此探针核对 rpath/LD_LIBRARY_PATH 生效顺序。

**未决项**：

- `libnnopbase.so` 运行期按 `load_priority` 逐 vendor dlopen 的完整行为（顺序、失败处理、与链接期解析的交互）仅经 strings 佐证，未逐项实验验证；如需精确控制可进一步实验。
## 坏实践

- 只看 `ldd ./test`（静态视图）就断定加载正确——`ldd` 不反映运行时 `LD_PRELOAD`/config.ini dlopen 与当条 shell 的 `LD_LIBRARY_PATH` 差异。
- 把他人 vendor 路径写进公共 run.sh/set_env 脚本。

## 误诊

- 发现外来路径后一度判定"double free 根因就是它"——前置目标路径后崩溃不变，证伪（见失效边界）；本规则的价值在于独立回答加载源问题。
