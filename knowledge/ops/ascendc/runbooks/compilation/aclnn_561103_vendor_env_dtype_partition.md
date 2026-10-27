---
type: "Runbook"
title: 561103 自定义aclnn非fp32调用环境相关失败
description: 未 source 算子包 set_env.bash 或经 torch_npu 进程调用时，自定义 aclnn 算子非 fp32 dtype 报 GetWorkspaceSize 561103，fp32 正常；带环境 21/21 全过。
tags: [task.debugging, topic.compilation, topic.architecture]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# 561103 自定义aclnn非fp32调用环境相关失败

## 现象

Cummin 自定义算子（fp16/fp32/int32/bf16 四 dtype，C++ L0 ST 已在 NPU 上 21/21 全过）在两条新路径上出现 dtype 分型失败：

1. **PyTorch 适配器路径**：新开发的 PyTorch ST 测试（`tests/st/torch/`，Python ctypes 加载 `libtorch_adapter.so`，内部两段式调用 `aclnnCumminGetWorkspaceSize` + `aclnnCummin`，进程已 `import torch_npu`）在 NPU 上执行：
   - fp32 用例 6/6 全部 PASS（MERE=0.00e+00，索引全对）；
   - fp16 / int32 / bfloat16 用例全部失败，报错原文：`CumminGetWorkspace: GetWorkspaceSize failed, status=561103`。
2. **C++ 二进制直跑路径**：同一个此前 21/21 全过的测试二进制（`tests/st/build-real/test_aclnn_cummin`），在不 source 算子包环境脚本的情况下直接执行（`cd tests/st/build-real && ./test_aclnn_cummin`）：
   - 7/21 通过（全部 fp32 用例 + 2 个"预期报错"用例）；
   - 14/21 失败，int32/bfloat16/float16 的所有真实执行用例均报 `GetWorkspaceSize failed: 561103`。

对照实验：同一二进制在 `source $HOME/Ascend/opp/vendors/cummin_custom/bin/set_env.bash` 之后经 `bash run.sh` 执行，21/21 全过。期间算子包未重装、代码未改动——结果差异只来自运行环境（vendor 环境脚本是否生效）。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的验收/联调阶段，为自定义 aclnn 算子搭建 PyTorch 适配层（torch_npu + ctypes + 两段式 ACLNN）或在裸环境直跑 ST 二进制。
- API 路径：`aclnnCumminGetWorkspaceSize`（自定义算子 ACLNN 入口）、算子包安装目录 `$HOME/Ascend/opp/vendors/<vendor>/bin/set_env.bash`（LD_LIBRARY_PATH/OPP 路径注入）、torch_npu 进程的库加载上下文。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：`workflow_only`（环境/调用上下文问题；随 build track 落 compilation）。
- 路线优先级：safety_fallback（本卡提供判别规则与最小复现对照；根因修复待遗留问题）。fallback 声明：在根因定位前，自定义 aclnn 算子的所有 NPU 调用（含 PyTorch 适配层）都应先 source 算子包的 set_env.bash；无该环境时的非 fp32 结果不可作为算子正确性结论。
- 明确不适用/未验证场景：经算子包环境脚本加载后的调用（不受影响）；561103 的官方定义未查到，其他触发场景未知。

## 判别方法

**触发**：自定义 aclnn 算子在 torch_npu 进程或未 source set_env.bash 的 shell 中调用 `aclnnXxxGetWorkspaceSize`，非 fp32 dtype 返回 561103，fp32 正常。

**探针**（可执行，最小对照）：

```bash
# A. 无 vendor 环境：预期非 fp32 用例 561103、fp32 通过
cd <op_dir>/tests/st/build-real && ./test_aclnn_cummin 2>&1 | grep -E "\[Real\]|\[PASS\]|\[FAIL\]|Total"

# B. 带 vendor 环境：预期全量通过
source $HOME/Ascend/opp/vendors/<vendor>/bin/set_env.bash && \
  rm -rf <op_dir>/tests/st/build-real && cd <op_dir>/tests/st && bash run.sh 2>&1 | grep -E "Total|Passed|Failed"
```

**原始观测**：
- torch 适配器：`cann-bench-01 fp16 [1024,1024] → FAIL 561103`；`L0-02 fp32 [256] → PASS MERE=0`；`--category L0-basic` 汇总 fp32 6/6 通过、fp16/int32/bf16 0/14 全 561103。
- C++ 直跑（无 env）：`L0_017 int32 [1,2] dim=1 → FAIL 561103`；`L0_019 bf16 → FAIL 561103`；fp32 用例（L0_002/012/013/014/015）全 PASS；报告 Total 21 / Passed 7 / Failed 14。
- 同一二进制 + set_env.bash：Total 21 / Passed 21 / Failed 0。

**解释**：561103 失败与"算子实现是否正确"、"适配器代码是否正确"都无关（fp32 走完全相同的调用路径通过；同一二进制带环境全过），是运行环境相关的失败——vendor 算子的注册/查找/符号解析路径在无 set_env.bash 注入的进程里对非 fp32 dtype 不可用。确切机制（如：无 vendor 库路径时 `aclnnCummin` 符号解析命中了系统 libopapi.so 中仅支持 fp32 的内置同名实现，或算子实现/tiling 库查找失败）未定位；561103 的官方错误码定义也未能查到。fp32 通过的原因同样未验证。561103 另有已闭合的同码根因（同名算子签名冲突时链接绑到内置版，其 l0op::Contiguous 返回 nullptr 触发 561103），见 [561103 同名算子签名冲突卡](561103.md)——本卡现象与之同族但机制未闭合；vendor 装载链路的探针方法见 [vendor 包 aclnn 装载链路验证探针清单卡](vendor_package_aclnn_loading_four_probes.md)与[aclnn 测试链接安装目录库卡](aclnn_test_links_installed_vendors_opapi.md)。

## 设计决策

- 处置选择"先判别、后深挖"而不是立即改适配器：dtype 分型 + 环境对照两个证据已把问题排除在 kernel 与适配器代码之外，继续改测试代码只会污染现场；根因属环境/加载链，需要专门的符号解析排查（遗留问题）。
- 当前结论属 safety_fallback。

## 系统信息

- 工具链：CANN 9.1.0（系统侧 `/workspace/.../Ascend/cann-9.1.0`，含 libopapi.so）；算子安装 `$HOME/Ascend/opp/vendors/cummin_custom/`（libcust_opapi.so、op_impl、set_env.bash）。
- 相关文件：`tests/st/torch/torch_adapter.cpp`（两段式封装）、`tests/st/torch/test.py`（ctypes 加载）、`tests/st/build-real/test_aclnn_cummin`（对照二进制）、`tests/st/run.sh`（run.sh 内部 source 环境）。

## 验证信号

验证来自同一开发会话的 NPU 对照实验：PyTorch 适配器路径 fp32 用例 6/6 PASS（MERE=0.00e+00）、非 fp32 用例全部 `GetWorkspaceSize failed, status=561103`（来源平台：950；证据等级：tool_observed）；同一 C++ 测试二进制无 set_env.bash 直跑 7/21（失败全部为非 fp32 真实用例），带 set_env.bash 经 run.sh 执行 21/21 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只证明"环境相关 + dtype 分型"的相关性，未证明因果机制；561103 在其他场景（如 tiling 未实现、参数非法、CheckBroadcastShape 过严，见 [反量化 561103 实为 CheckBroadcastShape 过严卡](561103_checkbroadcastshape.md)）也可能出现，不能把 561103 一律归因于环境。
- torch_npu 版本、进程内 LD_PRELOAD 状态对符号解析的影响未隔离验证。
- 其他 CANN 版本/安装布局未验证。

**未决项**：

- **优先项**：定位 561103 的确切机制——(1) 查证 561103 官方定义；(2) 在无 env 进程中用 `ldd`/`LD_DEBUG=symbols` 或 gdb 断点确认 `aclnnCumminGetWorkspaceSize` 实际解析到 libcust_opapi.so 还是系统 libopapi.so；(3) 解释 fp32 为何通过（是否存在同名内置算子仅支持 fp32）。
- torch_npu 进程内固定注入 vendor 库路径的工程化方案（如 sitecustomize/环境打包）未建立。
## 坏实践

- 拿无 vendor 环境下的非 fp32 失败去修改 kernel 或适配器代码——本例中该失败与代码无关，改代码是缘木求鱼。
- 在验收报告里把 561103 记为"算子/NPU 兼容性问题"而不做环境对照——一个 `source set_env.bash` 的对照实验即可把问题域收窄。

## 误诊

- 子任务报告称"C++ 直连无此问题"（agent_reported）：该说法只在带 set_env.bash 的 run.sh 路径上成立；C++ 二进制裸跑同样复现 561103，对照实验修正了这一表述。
- PyTorch 适配器初查时怀疑 `aclCreateTensor` 的 stride/format 构造错误（agent_reported）：dtype 分型（fp32 过、其余不过）与"张量描述构造"假设矛盾，被对照实验否定。
