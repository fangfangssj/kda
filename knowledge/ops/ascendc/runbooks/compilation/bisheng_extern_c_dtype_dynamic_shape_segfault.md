---
type: "Runbook"
title: "bisheng 动态 shape 编译段错误 — extern C + DTYPE"
description: "Ascend950 上某 extern C + DTYPE_* 宏入口算子在动态 shape TBE 重编译时触发 bisheng 段错误；记录静态/动态、缓存与入口改写的排除实验，不把未完成的 template 迁移当作修复。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# bisheng 动态 shape 编译段错误 — extern C + DTYPE

## 现象

在对照《CANNBot 生成算子标准流程》第五章检查项 5「GEIR 通路多场景补充测试」实跑已上库的 SmoothL1LossGrad 算子（3 输入 predict/label/dout + sigma 属性，`op_kernel/smooth_l1_loss_grad.cpp` 入口为 `extern "C"` + `KernelSmoothL1LossGrad<DTYPE_PREDICT>`，无 `tiling_key.h`，dtype 由编译期 `-DDTYPE_PREDICT=float` 宏注入）的动态 shape 场景时：

- 静态 shape `{4,8}`：GEIR 通路正常，数据精度双万分之一达标。
- 动态 shape `-1`（`{-1,-1}`）：`AddGraph` 成功，`RunGraph` 阶段 TBE 编译 kernel 时主进程段错误，TBE 子进程报 `main process disappeared`，`-2` 场景未测即退出。

对照算子 Relu6D（1 输入 + scale 属性，`op_kernel/arch35/relu6_d_tiling_key.h` 用 `ASCENDC_TPL_ARGS_DECL` + `ASCENDC_TPL_SEL`，kernel 入口为 `template <typename D_T_X> __global__ __aicore__ void relu6_d(...)`，4 TilingKey 按 dtype 分发）在同一环境跑动态 shape `-1`/`-2` 各 5/5 全部 PASSED。

环境：CANN 9.2.0（weekly.20260715.01，子包 9.1.0），SOC=Ascend950DT（dav-3510 / arch35），驱动 7.0.t9.0.B798，davinci0 可用，TBE 编译用系统 Python 3.9.5。崩溃算子已 build+install 到 `custom_nn` vendor（`$CANN/vendors/custom_nn/`），`source $CANN/opp/vendors/custom_nn/bin/set_env.bash` 后 GE 能找到算子并开始 TBE 编译。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发，GEIR 通路动态 shape（`-1`/`-2`）实跑 / TBE 动态编译。
- **API 路径**：`op_kernel/*.cpp` kernel 入口（`extern "C"` + `DTYPE_*` 宏 vs `ASCENDC_TPL` template）+ `op_host/arch35/*_tiling_arch35.cpp` 的 `ASCENDC_TPL_SEL_PARAM` 注册；Python tiling 脚本 `compile_op` 调用 bisheng/ccec。
- **验证过的平台与 CANN 版本**：Ascend950DT（arch35），CANN 9.2.0。仅在此平台实测过。
- **适用粒度**：`exact_task`。已验证对象是 SmoothL1LossGrad 的这一入口与动态编译组合，其他算子需复现实验后才能归并。
- **路线优先级**：`diagnostic_only`。本卡只记录编译器崩溃的隔离过程，不提供 fallback 或已验证迁移方案。
- **明确不适用 / 未验证场景**：静态 shape 不触发该崩溃；其他入口模式、算子与 a2/a3 未验证。

## 已隔离的故障边界

该 trace 能确认崩溃位于 bisheng 的动态 shape TBE 重编译阶段，并与当前 `extern "C"` + `DTYPE_*` 宏入口形态相关；尚未隔离到 bisheng 内部的具体 lowering 指令，也没有证明所有同类入口都会复现。

## 判别方法

**触发条件**：算子 def.cpp 设 `DynamicRankSupportFlag(true)` + `DynamicShapeSupportFlag(true)`，kernel 入口为 `extern "C"`（非 template），kernel 内用 `DTYPE_*` 宏区分 dtype；GEIR 用动态 shape（`-1`/`-2`）调 `RunGraph` 触发 TBE 重编译。

**判别探针**（按下述顺序逐项执行，均可复现）：

1. **静态 PASS / 动态崩溃分离**：跑静态 shape（如 `{4,8}`）应 PASS；跑动态 `-1` 应在 `AddGraph success` 后 `RunGraph` 阶段崩溃，stderr 出现 `timeout: the monitored command dumped core` 与 `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`。
2. **缓存复用确认**：首个动态 shape（如 `{2,3}`）若此前编译过会复用缓存 kernel（GE 日志 `UpdateOpTaskForCompileCache ... can reuse cache of json file path ... don't need to compile`）而看似 PASS；清缓存后所有动态 shape 均崩。此现象即说明崩溃点在「TBE 重编译」而非算子逻辑或 tiling。
3. **dmesg 取铁证**：

```
[3653499.179525] bisheng[3830638]: segfault at 10 ip 000055612915109f sp 00007ffd9b30a570 error 6 in bisheng[5561284c3000+55cb000]
```

`bisheng` 进程段错误，`error 6` = 写入不存在的页面（地址 `0x10`，疑似空指针 + 偏移）。plog / `_bt.log` 为空（崩溃太早无日志写入），故 dmesg 是核心可观测信号。

4. **排除实验链**（在已安装 vendor 目录逐项改动后清缓存重跑动态 shape，逐项仍崩即排除该因素）：
   - 简化 kernel `Compute` 为单 `DataCopy` → 仍崩 → 非 VF 代码复杂度 / 非 5 步 VF 链 / 非 7 寄存器。
   - 移除 Python tiling 脚本 `compile_op` 的 `--cce-no-dcache-flush` 选项（对齐 Relu6D 的 `'{}'`）→ 仍崩 → 非编译选项。
   - 整段替换 Python tiling 脚本为 Relu6D 风格（include 路径解析 + `compile_op` 7 参 + `{'valueDepend': {}}`）→ 仍崩 → 非 Python 脚本。
5. **入口改写实验（缩小范围）**：仅把已安装 kernel 入口改为 `template <typename D_T_PREDICT>` 后，bisheng 不再以原方式段错误，而是在更早阶段报 `expected a qualified name after 'typename'`。该实验说明入口改写改变了失败路径，但由于改写后的源码本身不能编译，不能据此证明 template 入口是可运行修复。

**解释边界**：静态 shape 与缓存复用均能绕开当次 TBE 重编译，动态 shape 清缓存后触发 bisheng segfault，因此可把故障定位到重编译路径。入口改写会改变失败形态，但确切的编译器内部机制仍未验证。

## 预防

- 声明动态 shape 支持的算子必须在清理编译缓存后跑至少一个动态用例，避免缓存复用掩盖 TBE 重编译崩溃。
- 提交编译器问题时附静态/动态对照、缓存状态、dmesg 与最小入口形态；不要用关闭动态 shape 的方式掩盖问题。

## 验证信号

问题与验证来源：本次 GEIR 多场景实跑在 Ascend950DT / CANN 9.2.0 上对 SmoothL1LossGrad 与 Relu6D 做静态 + 动态 shape 跑测；子 agent 在 vendor 安装目录做排除实验后已恢复原状。

- 静态 shape `{4,8}` SmoothL1LossGrad PASS，数据精度双万分之一达标（来源平台：950；证据等级：tool_observed）。
- Relu6D 动态 shape `-1` 5/5 + `-2` 5/5 PASSED，数据精度达标（来源平台：950；证据等级：tool_observed）。
- SmoothL1LossGrad 动态 shape `-1` `RunGraph` 阶段 TBE 编译崩溃，`timeout: the monitored command dumped core` + `main process disappeared`（来源平台：950；证据等级：tool_observed）。
- dmesg：`bisheng[3830638]: segfault at 10 ip 000055612915109f sp 00007ffd9b30a570 error 6 in bisheng[5561284c3000+55cb000]`（来源平台：950；证据等级：tool_observed）。
- 首个动态 shape `{2,3}` 复用缓存 kernel PASS，清缓存后崩——崩点在 TBE 重编译而非算子逻辑（来源平台：950；证据等级：tool_observed）。
- 入口改为 `template` 后 bisheng 不再段错误，转图编译错误 `expected a qualified name after 'typename'`（来源平台：950；证据等级：tool_observed）。
- 恢复原状后静态 shape 仍 PASS、Relu6D 动态仍 PASS（来源平台：950；证据等级：tool_observed）。
## 性能影响

无性能故事（本卡为编译崩溃判别，未涉及两条路线的成对相对性能对比）。

## 失效边界

- **未验证，推断可迁移**：a2 / a3 SoC 上的 bisheng 是否同样在 `extern "C"` + `DTYPE_*` 宏 + 动态 shape 组合下段错误——同编译器族、机制同源，推断可迁移，但未实测，不得写入 `platforms`。
- **不触发**：本 trace 的静态 shape 路径。
- **未形成修复闭环**：入口改写只改变了失败形态，没有得到可编译、可运行、精度通过的动态 shape 实现。

## 坏实践

（本卡无测试侧放松捷径。排除实验链中的「简化 Compute / 改 Python 脚本 / 移除编译选项」均为合法诊断手段，非投机取巧；若有人试图通过「声明 `DynamicShapeSupportFlag(false)` 关闭动态 shape 让用例过」掩盖崩溃，属坏实践——动态 shape 是 should-have 功能，关闭属功能回退而非修复。）

## 误诊

- **3 输入 shape 一致性校验**：早期怀疑 tiling 中 `GetStorageShape() != ` 比较在动态 shape 下返回 -1 致除零或负数 blockNum。被「简化 Compute 仍崩 + Relu6D 单输入也走类似 tiling 不崩」排除。
- **安装包不一致**：旧 ASCENDC_TPL 包与新 DTYPE_PREDICT 包均崩、静态均正常，排除安装包过时。
- **3 输入 / VECIN 队列数量**：怀疑 3 VECIN + 1 VECOUT 触发编译器 bug。被「入口改 template 即不崩」排除（输入数量未变，仅入口模式变）。
- **`--cce-no-dcache-flush` 选项**：怀疑该选项触发 bisheng bug。移除后仍崩，排除。
- **Python tiling 脚本路径解析**：怀疑 `tikcpp_path` 计算错误或缺 `{'valueDepend': {}}`。替换为 Relu6D 风格仍崩，排除。

以上误诊均为 `agent_reported` / `inferred` 的早期假设，已被 `tool_observed` 实验排除，不进验证信号。