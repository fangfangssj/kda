---
type: Runbook
title: arch22→arch35 移植与反向生成算子的 NPU 验证
description: arch22→arch35 移植与反向算子产物的 NPU 验证 runbook：先划真值边界（默认用内容寻址的 KernelBench 风格 task/sidecar，a3_live 源侧实测为显式回退，反向以前向契约与解析梯度为准），再按 R0 构建来源证明到 R5 属性 oracle 的顺序逐项验证。
tags: [task.testing, topic.precision, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/reference/porter/toolchain/npu_ut_validation.md
    role: primary
created_at: "2026-08-05T14:23:52Z"
updated_at: "2026-09-19T18:01:00Z"
platforms: [a3, "950"]
aliases:
  - NPU validation for generated arch22→arch35 and backward operators
  - npu_ut_validation
  - npubench task sidecar golden
  - GM canary out-of-bounds probe
---
# arch22→arch35 移植与反向生成算子的 NPU 验证

## 适用场景

本 Runbook 用于验证当前任务中生成的代码：arch22→arch35 跨代际移植算子与正向→反向生成算子的 NPU 侧产物。

目标压缩包（target archive）、既有实现（prior art）和预置目标分支可以为生成与调研提供参考，但必须按此身份记录：不得原样复制进交付物并宣称成功，不能替代本任务的构建来源证明（build provenance），也绝不能被提升为移植或反向真值。预置分支只有作为选定 arch22 来源的一部分、且可追溯至该来源契约时才可复用。

## 前置条件

### 真值边界（truth boundary）

- 移植默认真值：声明的算子契约 + 不可变的、内容寻址的 KernelBench 风格 task/sidecar 包（task `.py` + 同 stem `.json`/`.jsonl` sidecar 对；`reference.source=npubench`）；持久化状态必须以 digest 绑定 bundle、fixture 和提供方拥有的验证证据。
- 移植 `a3_live` 真值：声明的算子契约 + 本次运行期间从当前用户选定的 arch22 来源在源架构 NPU 上产出的 capture。它是显式回退，不是 npubench 包缺失时的静默替代。
- 旧版 `a3_live` 来源抓取措辞："Migration ordinary truth: source-arch NPU"——其中 "ordinary" 仅限该选定来源抓取提供方；显式选定的 npubench task/sidecar 包仍是其自身不可变的 golden 契约。
- 反向真值：声明的前向契约、saved-tensor 契约、解析梯度公式，以及 CPU fp64 autograd oracle。
- 目标 NPU 执行只用于验证新生成的目标，永远不是答案来源。
- 公开的目标 API 与硬件文档，加上独立编写的探针，可用于确立目标能力。

### 基本规则：当前生成源码必须与被测二进制一致

记录选定 arch22 来源、声明 schema、生成源码树、构建 manifest、设备目标文件和已加载扩展的 digest。只有当被加载的二进制可追溯到这些当前生成输入时，结果才可采信。来源存疑时，删除生成的构建产物、干净重建并复查 digest。

已安装的目标源码、目标文件、元数据、分派记录和归档实验可作为 prior art 参考。这些证据必须与选定来源契约、当前生成二进制 digest 链以及最终验证使用的真值权威分开保存。

### 环境与运行时

诊断 kernel 代码之前，先证明授权 lane 能初始化运行时、分配内存、启动最小公开 API kernel 并同步。记录编译器、运行时、驱动、Python ABI、框架包、设备架构和容器身份。凭证、主机地址和私有路径不进入知识库。

## 验证步骤

按以下判定顺序执行：

1. 验证契约与真值来源（provenance）。
2. 干净构建当前生成源码，并证明被加载二进制的 digest。
3. 运行最小目标公开 API 启动探针。
4. 运行内存与异常控制探针。
5. 确认生成分派覆盖。
6. 与选定的移植参考（默认冻结的 KernelBench 风格 task/sidecar 包，或显式源架构 capture）或 CPU fp64 autograd 真值比对结果。
7. 执行选项与属性分支。
8. 仅在正确性通过之后，运行确定性与性能门禁。

### R0：干净生成源码构建与来源证明

在任务拥有的空目录中，从当前生成源码与 manifest 构建。改动一个任务拥有的探针入口使其发出不同哨兵（sentinel），重建并确认哨兵变化；随后恢复生成源码并再次重建。由此证明测试加载的是本任务的二进制，且未修改或覆盖任何已安装算子。

### R1：GM canary 越界写检测

在声明范围的前后，用任务拥有的 guard region 包住输出或 workspace。初始化唯一哨兵，启动生成的 kernel，同步后验证两侧 guard。覆盖对齐、尾块（tail）、空和最小 shape。guard 存储不得改变逻辑张量契约。

### R2：受控设备异常探针

用同一生成入口分别运行一个合法对照和一个违反契约的触发。launch 与 synchronize 状态分开记录。只使用任务拥有的分配。异常只有在对照成功、且来源证明当前生成二进制确实运行过时才有意义。

### R3：与可采信真值做数值比对

默认移植：与不可变的 npubench 包证据及其声明容差比对；显式 `a3_live`：与同一当前 arch22 源架构 capture 及其声明容差比对。反向生成：归一到声明的输出 dtype 后，每个梯度都与 CPU fp64 autograd 比对。覆盖近零、溢出、归约、非连续、可选输入和 saved-state 用例。

### R4：生成分派确认

为每个来源推导的 dtype、shape、layout、属性、可选输入和反向分支至少构造一个用例。只对当前生成的代码插桩任务拥有的分支计数器或哨兵。最终干净构建之前移除插桩。

### R5：属性与选项 oracle

固定张量，一次只改变一个声明的属性或可选输入。根据契约和 CPU/来源 oracle 预测行为，再确认生成的输出。只有当声明语义表明该选项对所用例惰性（inert）时，无变化才是正确结果。

## 验证信号

- R0：探针哨兵随改动变化、恢复后复原——证明被测二进制来自本任务当前生成源码。（来源平台：a3/950；证据等级：tool_observed）
- R1：kernel 执行并同步后，前后 guard 哨兵均保持初始值。（来源平台：a3/950；证据等级：tool_observed）
- R2：合法对照成功，违反契约的触发在 launch 或 synchronize 阶段按预期报错，且两次状态分别记录。（来源平台：a3/950；证据等级：tool_observed）
- R3：前向输出与全部梯度输出在声明容差内通过比对。（来源平台：a3/950；证据等级：tool_observed）
- R4：分支计数器/哨兵确认每个生成的分派分支至少被执行一次；最终交付构建不含插桩。（来源平台：a3/950；证据等级：tool_observed）
- R5：输出与按契约预测的行为一致；属性无变化仅在语义惰性时可接受。（来源平台：a3/950；证据等级：tool_observed）
- 全部门禁：结果只有在 digest 链证明被加载二进制可追溯到当前生成输入时才可采信。（来源平台：a3/950；证据等级：tool_observed）

## 失效边界

- 在一侧架构上通过的门禁只是方法论证据：每个受支持的架构、编译器、运行时、dtype 和启动模式上都必须重跑构建、启动、内存与分派探针，不能跨架构外推。
- `a3_live` capture 不能静默替代缺失的 npubench 包；两者容差与真值权威各自独立。
- prior art（已安装目标源码、目标文件、元数据、分派记录、归档实验）仅供调研参考，不得计入来源证明链或真值权威。
- 来源存疑的构建结果一律不可采信：删除构建产物、干净重建并复查 digest 后重新验证。
- 对照失败或来源不明的异常探针不构成有效的拒绝证据。

## 更新 (2026-09-19)

私有沉淀仓（不公开，pinned commit cb9c78c85ead3088398f3455051355349298b3cb）`okf/runbooks/operator-optimization/` 下 NPU_UT_VALIDATION.md 的 11 张节选卡经逐张核读为本文对应小节的同 lesson 摘录（英文改写，无内容增量），不单独建卡，锚点对照：source_rebuild_overlay_override_proof → R0；gm_canary_oob_write → R1；device_exception_oob_read → R2；value_diff_cpu_reference → R3；dispatch_confirmation_probe → R4；attr_toggle_oracle → R5；verification_decision_order → 验证步骤判定顺序；installed_vs_scanned 与 cann_overlay_gotchas → 基本规则（digest 链 / 任务拥有、不 overlay 已安装算子树）；aclinit_500000_not_device_lock → 前置条件 §环境与运行时（正文一致；其标题级细节「aclInit 500000 报错并非设备锁」正文未展开，如实记录）；a3_recipes_are_template_on_a5 → 失效边界（每架构重跑门禁、不跨架构外推）。
