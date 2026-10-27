---
type: "Runbook"
title: "__global__ 核函数 extern C 失效致符号未定义"
description: "ASC 编译器对 __global__ __vector__ 核函数忽略 extern \"C\"，host 侧 stub 以 C++ mangled 名链接；torch 侧 extern \"C\" 声明找不到符号，加载 .so 报 undefined symbol。用 nm -D 判别后改 C++ linkage 修复。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# __global__ 核函数 extern C 失效致符号未定义

## 现象

HardSwish 算子按 ops-direct-invoke 双 target（可执行文件 + `libhardswish_ops.so`）开发。Kernel 核函数在 `op_kernel/hardswish_kernel.asc` 中定义为 `extern "C" __global__ __vector__ void hardswish_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)`；PyTorch 接入侧 `op_extension/hardswish_torch.cpp` 按 add_custom 模板照搬 `extern "C" void hardswish_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream, uint8_t *x, uint8_t *y, uint8_t *tiling);` 声明，期望以 C 链接名 `hardswish_kernel` 调用核函数 stub。

编译产出 `libhardswish_ops.so` 后，运行 `python3 ../scripts/test_torch.py` 加载 .so 时报错（agent 在归一化轨迹中逐字转述 test_torch 输出）：

```
undefined symbol: hardswish_kernel
```

## 适用场景

- **工程形态**：ops-direct-invoke Kernel 直调 + PyTorch 接入（TORCH_LIBRARY 注册 `lib<name>_ops.so`，host 侧通过 `<<<>>>` launch 语法调用核函数 stub）。
- **API 路径**：ASC 工具链 (bisheng) 对 `__global__ __vector__` 核函数的 host-side stub 生成与符号导出；`op_extension/*_torch.cpp` 的 `extern "C"` 声明与 `register.cpp` 的 `TORCH_LIBRARY`。
- **验证平台与版本**：Ascend 910B / Atlas A2，CANN 8.5.2，aarch64-linux。`platforms: [a2]` 与此一致。
- **适用粒度**：operator_family（所有走 ops-direct-invoke + PyTorch .so 双 target、且 torch 侧用 `extern "C"` 声明核函数 stub 的算子）。
- **路线优先级**：recommended_fastpath（torch 侧核函数 stub 声明去掉 `extern "C"`，改用与 stub 一致的 C++ linkage，让 mangled 名匹配）。
- **不适用/未验证场景**：aclnn 算子仓开发（ops-registry-invoke，走 op_host 注册 + aclnnAPI 封装）的符号导出路径未在本轨迹覆盖；950/a3 及其它 CANN 版本的 ASC stub 链接行为未验证，推断可迁移但需重验。

## 判别方法

**触发**：`lib<name>_ops.so` 被 `torch.ops` 或 `ctypes`/dlopen 加载时报 `undefined symbol: <name>_kernel`，而 .so 已成功编译生成。

**探针**（轨迹 tool_call 逐字保留的命令）：

```bash
nm -D <项目路径>/build/lib<name>_ops.so | grep <name>
```

对本例即：

```bash
nm -D /mnt/workspace/cann/cannbot-skills/ops/operators/hardswish/build/libhardswish_ops.so | grep -i hardswish
```

**原始观测**（agent 在归一化轨迹中逐字转述 nm 输出）：

```
000000000007ee54 T __device_stub__hardswish_kernel
                 U hardswish_kernel
000000000007ee58 T _Z16hardswish_kerneljPvS_PhS0_S0_
```

三个符号：`__device_stub__hardswish_kernel` (T=已定义，device stub)、`hardswish_kernel` (U=未定义，C 链接名)、`_Z16hardswish_kerneljPvS_PhS0_S0_` (T=已定义，C++ mangled 名)。被引用的 C 链接名 `hardswish_kernel` 标 U（未定义），故 dlopen 报 undefined symbol。

**解释**：ASC 编译器对 `__global__ __vector__` 核函数不尊重 `extern "C"`——核函数定义上的 `extern "C"` 并未生成 C 链接名的 host-side stub，stub 以 C++ mangled 名（`_Z16hardswish_kerneljPvS_PhS0_S0_`，对应 `hardswish_kernel(unsigned int, void*, void*, unsigned char*, unsigned char*, unsigned char*)`，即 `(blockDim, l2Ctrl, stream, x, y, tiling)`）导出。torch 侧 `extern "C"` 声明却去找 C 链接名 `hardswish_kernel`，因而链接未定义。

## 根因

ASC 编译器为 `__global__ __vector__` 核函数生成的 host-side launch stub 使用 C++ 名修饰（name mangling），核函数定义上的 `extern "C"` 对 stub 链接名无效；host 侧若仍用 `extern "C"` 声明该 stub，dlopen 时找不到 C 链接符号，报 undefined symbol。

## 解决配方

修复点在 `op_extension/hardswish_torch.cpp` 的核函数声明：去掉 `extern "C"`，改用与 stub 一致的 C++ linkage 声明（参数保持 blockDim/l2Ctrl/stream/x/y/tiling 六参数，与 mangled 名一致）。核函数定义（`op_kernel/hardswish_kernel.asc`）上的 `extern "C"` 可保留（对 stub 链接名无影响，轨迹终版 kernel 仍为 `extern "C" __global__ __vector__ void hardswish_kernel(...)`）。

```cpp
// after（torch 侧声明，C++ linkage，与 stub mangled 名匹配）
extern void hardswish_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *x, uint8_t *y, uint8_t *tiling);
```

注：该 edit 的 oldString/newString 在归一化轨迹中被 markdown_transcript adapter 折叠为 '_see below_'，未逐字保留；上示 after 声明为 agent 在 reasoning 中明确描述的修复方向（"removed extern \"C\" from torch declaration"、"torch 侧声明改用 C++ linkage"），并经 PLAN.md 开发日志（轨迹 file_read 产物）佐证 "extern C 对 __global__ 无效 | ASC 编译器对 __global__ 函数忽略 extern C，torch 侧声明改用 C++ linkage"。完整逐字 before/after 代码未完全从轨迹恢复，故本卡以 diagnostic_rule 形态发布（判别方法+探针+原始观测+解释为可逐字核验的核心机制单元）。

## 预防

走 ops-direct-invoke + PyTorch .so 双 target 时，torch 侧核函数 stub 声明不要照搬 add_custom 模板的 `extern "C"`；ASC 对 `__global__` stub 用 C++ mangling，host 侧声明应去掉 `extern "C"` 以匹配 mangled 名。若 .so 加载报 `undefined symbol: <func>`，先用 `nm -D lib<name>_ops.so | grep <func>` 看 C 链接名是否标 U、是否存在 `_Z...` mangled 名 (T)，即可判定为本问题。

## 坏实践

照搬 add_custom 模板在 torch 侧声明 `extern "C" void <name>_kernel(...)`，假定 ASC 编译器对 `__global__ __vector__` 函数尊重 `extern "C"`。该假定在 CANN 8.5.2/A2 不成立，stub 以 mangled 名导出，导致 .so 编译成功但运行时 dlopen 失败。

## 验证信号

本次会话为 HardSwish 算子开发轨迹（ops-direct-invoke + PyTorch .so，A2/CANN 8.5.2）；nm 命令为轨迹 tool_call 逐字保留，nm 输出与 test_torch 错误由 agent 在 reasoning 中逐字转述（归一化 adapter 将 bash 输出折叠进 reasoning，见 manifest provenance_missing）。

- `python3 test_torch.py` 加载 `libhardswish_ops.so` 报 `undefined symbol: hardswish_kernel`（来源平台：a2；证据等级: tool_observed）。
- `nm -D libhardswish_ops.so | grep -i hardswish` 命令逐字保留于轨迹 tool_call（来源平台：a2；证据等级: artifact_verified）。
- nm 观测：`__device_stub__hardswish_kernel`(T)、`hardswish_kernel`(U)、`_Z16hardswish_kerneljPvS_PhS0_S0_`(T)（来源平台：a2；证据等级: tool_observed）。
- torch 侧声明去 `extern "C"` 改 C++ linkage 后重编，PyTorch 通路 P1–P8 全部通过（NaN/Inf 用例 P4 经测试侧 NaN-aware 比较修正后通过；8/8 PASS）（来源平台：a2；证据等级: tool_observed）。
- PLAN.md 开发日志（轨迹 file_read 产物）记 "extern C 对 __global__ 无效 | ASC 编译器对 __global__ 函数忽略 extern C，torch 侧声明改用 C++ linkage"（来源平台：a2；证据等级: artifact_verified）。
## 失效边界

- aclnn 算子仓开发（ops-registry-invoke）走 `op_host` 注册 + `aclnn<Op>GetWorkspaceSize`/`aclnn<Op>` API 封装，符号导出路径不同，本判别规则不直接适用。
- 950/a3 平台、其它 CANN 版本 ASC 编译器对 `extern "C"` + `__global__` 的处理未验证，推断可迁移（C++ mangling 是编译器通用行为，但具体 stub 生成策略需重验）。
- 本卡只覆盖 host-side stub 链接名问题；device 侧核函数内部的 `extern "C"` 语义不在本卡范围。
