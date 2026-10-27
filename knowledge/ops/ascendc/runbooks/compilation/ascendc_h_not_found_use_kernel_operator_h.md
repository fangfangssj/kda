---
type: "Runbook"
title: 设备侧头文件用 kernel_operator.h 而非 AscendC.h
description: "op_kernel 下的设备侧头文件 #include \"AscendC.h\" 使 bisheng 前端报 fatal error: 'AscendC.h' file not found（host/UT 目标可编译，仅设备侧 kernel 编译失败）；设备侧统一入口是 kernel_operator.h，已传递包含 AscendC::Reg:: 等 regbase 类型。"
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# 设备侧头文件用 kernel_operator.h 而非 AscendC.h

## 现象

bessel_i1e 算子 Branch-0（FP32）kernel 新增 VF 头文件 `op_kernel/bessel_i1e_kernel_vf.h` 时，按"使用 AscendC 命名空间就 include AscendC.h"的直觉写了：

```cpp
#include "AscendC.h"
```

执行 `./operators/build.sh BesselI1e_package`（CANN cann-9.0.0-beta.2，目标 ascend950），host/UT 目标全部编过，唯独设备侧 kernel 编译（opc/bisheng）失败：

```
In file included from /workspace/.../op_kernel/bessel_i1e_kernel.cpp:13:
/workspace/.../op_kernel/bessel_i1e_kernel_vf.h:14:10: fatal error: 'AscendC.h' file not found
#include "AscendC.h"
         ^~~~~~~~~~~
1 error generated.
[ERROR] TBE(...):[op_compilation.py:190][record_compile_error_info] Op[BesselI1e] of index[0] compile failed, kernelName: BesselI1e_6661d1593d5e838f694e23d2f3db5893.
RuntimeError: Kernel Compilation Error: OpType BesselI1e Kernel File .../bessel_i1e_kernel.cpp!
```

同一包内 `add_custom` 示例 kernel 与本 kernel 的 `.cpp` 入口都只 include `kernel_operator.h`。把 include 改为 `"kernel_operator.h"` 后该错误消失（后续编译进入下一类错误）。

## 适用场景

- 工程形态：算子仓 kernel 开发（op_kernel/ 下设备侧头文件与源文件）。
- API 路径：设备侧统一 include 入口 `kernel_operator.h`（提供 AscendC::TPipe/TQue/TBuf、AscendC::Reg::（regbase）等全部设备侧类型）。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：host 侧（op_host/）代码不 include kernel_operator.h（用 gert/register 头）；纯 x86 UT 代码不受 bisheng 搜索路径约束。

## 根因

设备侧编译（bisheng frontend）的 include 搜索路径里不存在名为 `AscendC.h` 的头文件；`kernel_operator.h` 是设备侧统一入口，且已传递包含 `AscendC::Reg::*`（regbase）等类型——不需要也不应该单独 include `AscendC.h`。

## 解决配方

修复前（verbatim）：

```cpp
#include "AscendC.h"
```

修复后（verbatim 取自修复编辑）：

```cpp
#include "kernel_operator.h"
```

## 设计决策

- 参考同仓已验证 kernel（如 add_custom）的 include 形态：设备侧一律 `#include "kernel_operator.h"`。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：ascend950 目标，CANN cann-9.0.0-beta.2，bisheng clang 15.0.5。
- 相关文件：`operators/BesselI1e_package/op_kernel/bessel_i1e_kernel_vf.h`（新增 VF 头）；对照 `operators/add_custom_package/op_kernel/add_custom_kernel.cpp`。

## 预防

- 设备侧头文件自行拼 include 前先对照包内已验证 kernel（add_custom 等）的 include 形态，不要凭命名空间名猜头文件名。

## 验证信号

- 修复前：`fatal error: 'AscendC.h' file not found`（bisheng，1 error generated，3 个 kernel 变体全部失败）（来源平台：950；证据等级：tool_observed）。
- 修复后：该 fatal error 消失，同文件的下一轮编译错误变为 CMPMODE 未声明（进入 API 签名问题，属另一独立根因），证明 include 问题闭环（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 无特别边界；其他 CANN 版本上该入口名称未逐一验证，推断一致（未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- 凭命名空间名猜测头文件名（`AscendC.h`）；设备侧头文件自行拼 include 而不对照包内已验证 kernel。

## 误诊

- 无。
