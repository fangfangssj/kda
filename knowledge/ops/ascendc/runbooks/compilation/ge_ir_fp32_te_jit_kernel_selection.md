---
type: "Runbook"
title: GE IR fp32 命中 te_ JIT 而非预编译 binary
description: 自定义算子 GE IR 图模式 fp32 输入运行时选中 te_ 前缀 JIT/TBE 动态实现 kernel（GetArgsInfo argsSize=32、blockDim=43）而非预编译 AIC binary（argsSize=136、blockDim=4），JIT kernel 执行越界；同 shape ACLNN 直调正常，示例输入改 fp16 后命中预编译 binary 恢复——判别方法与规避路线。
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-17T01:48:53Z'
updated_at: '2026-09-17T01:48:53Z'
platforms: ["950"]
---

# GE IR fp32 命中 te_ JIT 而非预编译 binary

## 现象

自定义双线性插值算子（ResizeBilinear 类，同时交付预编译 AIC binary 与 TBE 动态实现）的 GE IR（图模式）示例在补齐 InferDataType 注册后，图能编译、能起 session，但 fp32（DT_FLOAT）输入运行到 `Start to run ir compute graph` 后 `Run graph failed`。plog 显示的故障与 ACLNN 直调路径的越界同类：

```
[ERROR] RUNTIME(...):[model_execute_task.cc:538] ReportErrorInfoForModelExecuteTask:Real fault task, device_id=0, ..., type=66[KERNEL_AIVEC].
[ERROR] RUNTIME(...):[davinci_kernel_task.cc:1076] GetArgsInfo:[AIC_INFO] args(0 to 3) after execute:0x12004da10200, 0x12004da00200, 0x12004ca00000, 0.
[ERROR] ... GetArgsInfo:tilingKey = 0, print 1 Times totalLen=(4*8), argsSize=32, schemMode=0, blockDim=43
[ERROR] ... fault kernel_name=te_benchnoharnessresizebilinear_<hash>__kernel0, ...
```

对照 ACLNN 直调（同 shape 正常）的 plog：fault/执行的 kernel 是预编译 AIC binary `BenchNoHarnessResizeBilinear_<hash>_0`，`argsSize=136`（17×8 字节 TilingData）、`blockDim=4`。两条路径选中的根本不是同一个 kernel 实现。安装目录的 binary_info_config.json（`opp/vendors/<op>_custom/op_impl/ai_core/tbe/kernel/config/ascend950/<op>.json`）binList 里预编译 binary 的 simplifiedKey 覆盖 `d=0,p=1`、`d=0,p=0`、`d=1,p=1`、`d=1,p=0` 四种组合。

把示例输入 dtype 从 DT_FLOAT 改成 DT_FLOAT16 后，GE IR 图执行 `Session run ir compute graph success` + `Precision is ok`，恢复正常。

## 适用场景

- 工程形态：自定义算子 GE IR 图模式调用（同一算子 ACLNN 直调正常、GE IR 失败的双路径差分场景）；算子包同时含预编译 AIC binary 与 TBE 动态实现。
- API 路径：GE 运行时 kernel 选择（binary_info_config.json / simplifiedKey 匹配）、预编译 AIC binary 与 `te_*` JIT/TBE 动态实现两条实现来源；plog 的 `GetArgsInfo` 任务信息。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有同时提供预编译 AIC binary 与 TBE 动态实现的算子仓型自定义算子。
- 路线优先级：safety_fallback（换 dtype 是示例层规避；GE 匹配机制层的根因修复未完成，采用前须自知这是绕行，不能据此宣称 fp32 图模式已修复）。
- 明确不适用 / 未验证场景：只交付单一实现（无 TBE 动态实现）的算子包不存在此分叉；fp32 路径为何不匹配的机制层结论未闭合，不能推广为"fp32 必然走 JIT"。

## 判别方法

触发 → 探针 → 观测 → 解释：

1. 触发：GE IR 图模式运行自定义算子失败（`Run graph failed` / KERNEL_AIVEC fault），而 ACLNN 直调同 shape、同 dtype 正常。
2. 探针：对比两条路径的 plog 任务信息——`GetArgsInfo` 的 argsSize 与 blockDim、`fault kernel_name`（或正常执行时的 kernel 名）：
   - `te_<op小写>_<hash>__kernel0` 前缀 + argsSize=32（4×8）+ blockDim=43 → JIT/TBE 动态实现 kernel；
   - `<Op>_<hash>_0`（与安装目录 .o 同名）+ argsSize=136（完整 TilingData 字段数×8）+ blockDim 与 Tiling 一致 → 预编译 AIC binary。

   辅以 `cat .../tbe/kernel/config/ascend950/<op>.json` 查看 binList 的 simplifiedKey 覆盖（d/p 组合）。
3. 观测：fp32（DT_FLOAT）输入 → GE IR 选中 `te_...__kernel0` 并越界；输入改 DT_FLOAT16 → GE IR 命中预编译 binary，图执行成功且精度 OK。`ge.dynamicCompileStaticFlag=0` 的 build option 尝试无效（仍失败）。
4. 解释：GE 运行时对 fp32 输入配置未匹配到预编译 binary（binList 声称覆盖 d=0/d=1×p=0/p=1），回落到 te_ JIT/TBE 动态实现，该实现在所测 shape 上执行越界；fp16 配置可匹配预编译 binary。为何 fp32 不匹配的确切机制（simplifiedKey 的 d/p 语义与输入 dtype 的映射规则）未追到底。

## 设计决策

- 为什么选择"示例改 fp16"而不是继续追根因：当时示例任务的交付目标是给出可运行的调用样例；fp16 是算子支持的 dtype 之一且与 ST 已验证路径一致，示例层换 dtype 成本最小。kernel/JIT 层根因超出示例任务边界。
- 该规避的代价：示例不再覆盖 fp32 的 GE IR 调用形态；fp32 GE IR 用户仍会踩同一越界。因此本卡定位为 safety_fallback，不宣称 fp32 路径已修复。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0，GE 运行时（libge_runner），plog（GE 编译子进程日志，按进程号取 `plog-<pid>_*.log`）。
- 相关文件：`examples/test_geir_*.cpp`（dtype 定义处）、`opp/vendors/<op>_custom/op_impl/ai_core/tbe/kernel/config/ascend950/<op>.json`（binList/simplifiedKey）、预编译 binary 目录 `.../tbe/kernel/ascend950/<op>/`。
- 诊断命令：`ls -t ~/ascend/log/debug/plog/*.log | head -1 | xargs grep -i "fault kernel_name\|GetArgsInfo\|ReportErrorInfoForModelExecuteTask"`。

## 预防

- 自定义算子交付 GE IR 支持时，至少用每一种声明支持的 dtype 各跑一遍图模式用例，不要只用单一 dtype 验证后就宣称"GE IR 可用"。
- 双路径差分（ACLNN 好而 GE IR 坏）时，先对比 plog 的 kernel_name/argsSize/blockDim 确认两条路径是否选中同一实现，再谈 kernel 缺陷。

## 验证信号

以下信号来自该算子 GE IR 示例联调会话：fp32 失败 → plog 差分定位 → fp16 规避 → 双示例交付。

- fp32 触发：GE IR fp32 运行 `Run graph failed`，plog `fault kernel_name=te_benchnoharnessresizebilinear_...__kernel0`、`argsSize=32`、`blockDim=43`、KERNEL_AIVEC fault（含越界）（来源平台：950；证据等级：tool_observed）。
- 路径差分：ACLNN 直调同 shape fp32 正常，执行 kernel 为预编译 `BenchNoHarnessResizeBilinear_<hash>`、argsSize=136（17×8）、blockDim=4（来源平台：950；证据等级：tool_observed）。
- 规避生效：示例输入改 DT_FLOAT16 后 GE IR `Session run ir compute graph success`、输出 dtype 回读正确、`Precision is ok`；最终交付 ACLNN（fp32）与 GE IR（fp16）双示例 PASS（来源平台：950；证据等级：tool_observed）。
- 无效尝试：build options 加 `{"ge.dynamicCompileStaticFlag", "0"}` 后 GE IR 仍失败（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 换 dtype 只是让示例命中可用实现，fp32 的 GE IR 路径问题仍在：任何必须 fp32 图模式调用的场景不能靠本规避。
- "GE 对 fp32 不匹配预编译 binary"的机制解释未验证到 simplifiedKey 匹配规则层；fp16 命中预编译 binary 是观测事实，不保证其他算子/dtype 组合同构（未验证，推断不可外推）。
- 其他平台未验证，推断不可迁移。

**未决项**：

- 【优先项】复现并追根因：GE 运行时 binary 匹配机制——binary_info_config.json 的 simplifiedKey（`d=<dtype 索引>,p=<buffer mode>`）与输入 dtype/编译选项的映射规则，为何声明覆盖 d=0（含 fp32 组合）的 binList 仍让 fp32 走了 te_ JIT。
- te_ JIT/TBE 动态实现本身的越界缺陷未定位（它是另一份实现，不含本仓 AscendC kernel 的修复）。
- fp32 GE IR 在 kernel 侧 UB 修复后是否恢复、以及 ge.opsProtoLibPath 对 kernel 选择有无影响，均未验证。
## 坏实践

- 把"ACLNN 好而 GE IR 坏"直接归因为"图模式本来就不可用/示例写错"，不核对两条路径实际选中的 kernel。
- 用单一 dtype 验证通过就交付全部 dtype 的 GE IR 支持声明。

## 误诊

- 期间曾把 GE IR 失败与小 shape 关联（"4x4→8x8 触发 kernel bug"）：后续 64x64→128x128 fp32 同样失败，证明与 shape 规模无关，真正的分岔是选中的 kernel 实现。
- `ge.dynamicCompileStaticFlag` 调整被寄望于强制静态编译，实测无效。
