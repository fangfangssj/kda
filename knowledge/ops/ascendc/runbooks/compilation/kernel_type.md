---
type: "Runbook"
title: "直调 kernel-type 裸赋值标记无效"
description: "AscendC 直调 kernel 用文件作用域裸赋值 __enable_feature_for_compile_default = KERNEL_TYPE_*; 标记 CUBE+VEC mix 类型时编译失败；该写法只被 extract_host_stub.py 的正则匹配为文本标记，不是合法 C++ 语句，属 banned 路线。"
tags: ["task.debugging", "topic.programming_model.system_variable"]
status: stable
sources:
  - resource: "https://gitcode.com/cann/cann-outreach/blob/8cce8c1d7a332d3c4a0f88e0b4166781a638b8af/2026/CANN-Code-Detective/README.md"
    role: constraint
created_at: "2026-08-18T05:06:14Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["a2"]
---

# 直调 kernel-type 裸赋值标记无效

## 现象

在 Ascend910B3 设备（CANN 9.2.0，`--npu-arch=dav-2201`）上把 rain_fusion_attention 的 P/V matmul 改造成 AscendC 直调工程（`ops-direct-invoke`），kernel `pv_matmul_custom` 需要 CUBE 核做 mmad、VEC 核做跨核 flag 通知，同时启动两类核。首次 build 成功但带 warning：

```
WARNING: kernel type of __global__ func: pv_matmul_custom(unsigned char*, unsigned char*, unsigned char*, unsigned char*, unsigned char*, unsigned char*) is not marked. auto type derivate may be failed.
```

随后运行 `timeout 30 ./pv_matmul_demo Ascend910B3` 直接 `EXIT: 124`（kernel 挂起，无输出）。`extract_host_stub.py::find_kernel_type_by_source` 用正则 `__enable_feature_for_compile_default\s*=\s*([0-9a-zA-Z_]{1,})\s*;` 从源码**文本**推断 kernel 类型，v220/dav-2201 找不到该标记就返回 `None`（推导失败）。于是在 `.asc` 文件顶部、`#include "attn_infra/rain_base_defs.hpp"` 之前插入了一行裸赋值：

```
__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_1;
```

重新 build 报编译错误：

```
/mnt/workspace/gitCode/pv_matmul_direct_invoke/op_kernel/pv_matmul_kernel.asc:33:1: error: a type specifier is required for all declarations
1 error generated.
make[2]: *** [CMakeFiles/pv_matmul_demo.dir/build.make:76: CMakeFiles/pv_matmul_demo.dir/op_kernel/pv_matmul_kernel.asc.o] Error 1
```

## 适用场景

- **工程形态**：AscendC 直调 kernel（`ops-direct-invoke`，`.asc` + CMake + `bash build.sh` + `./demo SOC`），非注册算子、非 aclnn。
- **API 路径**：`__global__ __aicore__` kernel 函数 + `__kfc_workspace__` 参数 + `<<<blockDim, nullptr, stream>>>` launch；构建工具 `extract_host_stub.py::find_kernel_type_by_source`（CANN 9.2.0 `aarch64-linux/tikcpp/ascendc_kernel_cmake/legacy_modules/util/`）。
- **验证过的平台与 CANN 版本**：Ascend910B3（按当前注册口径记为 `a2`），CANN 9.2.0，`--npu-arch=dav-2201`（v220/c220 build mode）。
- **适用粒度**：`operator_family`（适用于所有需要 CUBE+VEC mix 类型的直调 kernel，不限于 pv_matmul）。
- **路线优先级**：`banned`（文件作用域裸赋值 `__enable_feature_for_compile_default = ...;` 不是合法 C++，禁止用作标记手段）。
- **不适用或未验证场景**：`matmul::Matmul` 高阶 API 直调 kernel 不需要本标记（其内部走 KFC 调 CUBE，`AIV_ONLY` 即可，本 trace 未对该路径单独复测）；A2/Ascend950 未验证；c310/m200 build mode 的 `STR_TO_KERNEL_TYPE_*` 词表差异未在本 trace 验证。

## 失败路线

**banned：文件作用域裸赋值 `__enable_feature_for_compile_default = KERNEL_TYPE_*;`。**

目标：在直调 kernel 中显式标记 CUBE+VEC mix 类型，使 `find_kernel_type_by_source` 推导出非 `None` 的 `CodeMode`，从而同时启动两类核。

机制链（不跳步）：

1. `find_kernel_type_by_source` 的判定逻辑（`extract_host_stub.py`，本 trace `Read` 事件逐字恢复）：

```python
def find_kernel_type_by_source(path: str, data: str, build_mode: str) -> CodeMode:
    is_c310 = build_mode == "c310"
    is_m200 = build_mode == "m200"
    is_v220 = build_mode == "c220"
    match = re.search(
        r"__enable_feature_for_compile_default\s*=\s*([0-9a-zA-Z_]{1,})\s*;", data
    )
    if match:
        kernel_type = match.group(1)
        if (kernel_type in STR_TO_KERNEL_TYPE_V200) and is_m200:
            return STR_TO_KERNEL_TYPE_V200[kernel_type]
        elif (kernel_type in STR_TO_KERNEL_TYPE_V220) and is_v220:
            return STR_TO_KERNEL_TYPE_V220[kernel_type]
        elif (kernel_type in STR_TO_KERNEL_TYPE_C310) and is_c310:
            return STR_TO_KERNEL_TYPE_C310[kernel_type]
        else:
            print(f"[WARNING]: {kernel_type} in path: {path} do not support in soc {build_mode}!")
            if is_v220 or is_c310:
                return None
            else:
                return CodeMode.AIC
```

2. 该函数对源码 `data` 做**正则文本匹配**——它只关心源文本里是否**出现** `__enable_feature_for_compile_default = <TOKEN>;` 这个串，并不要求它是合法 C++ 声明。于是产生陷阱：看起来"在 .asc 里写一行 `__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_1;` 就能被识别"，但 bisheng/ccec 编译器把这行当成文件作用域的 C++ 语句解析——`__enable_feature_for_compile_default` 既非类型也非已声明变量，左侧无类型说明符，触发 `error: a type specifier is required for all declarations`（`pv_matmul_kernel.asc:33`）。

3. 修复前先尝试过的相关回退（均未消除 warning、均未解挂起，归入 `## 误诊`）：把 `__kfc_workspace__` 参数从第 5 位移到第 4 位、把 `gSelectIdx` 合并进 workspace 把参数减到 5 个与 matmul 示例对齐——warning 仍在，因为真因是 kernel 类型未标记，不是参数数量/位置。

该路线以编译错误告终，目标（标记 mix 类型以启动 CUBE+VEC）未达成，确认为 banned：正则文本匹配 ≠ 语法合法，不能用裸赋值作 kernel 类型标记。

## 验证信号

本卡验证信号来自 2026-08-15 一条 pv_matmul 直调开发会话（trace b349a3624caf，Ascend910B3 设备，CANN 9.2.0）：

- 修复前 build 输出含 `WARNING: kernel type of __global__ func: pv_matmul_custom(...) is not marked. auto type derivate may be failed.`，运行 `timeout 30 ./pv_matmul_demo Ascend910B3` `EXIT: 124`（kernel 挂起无输出）（来源平台：a2；证据等级：tool_observed）。
- 插入裸赋值后 build 报 `pv_matmul_kernel.asc:33:1: error: a type specifier is required for all declarations` + `1 error generated.` + `make[2]: *** ... Error 1`（来源平台：a2；证据等级：tool_observed）。
- `extract_host_stub.py::find_kernel_type_by_source` 的正则与 `None` 返回路径逐字恢复（来源平台：a2；证据等级：artifact_verified）。

## 性能影响

无性能故事。本卡只涉及编译期 kernel 类型标记，不涉及两条路线的相对性能代价。

## 误诊

- **误诊一（参数数量/位置）**：曾以为 warning 是因为 `__kfc_workspace__` 不在第 4 位或参数数 6 > matmul 的 5，把 workspace 移到第 4 位并把 `gSelectIdx` 合并进 workspace。改完参数签名与 matmul 完全一致后 warning 仍在——真因是 kernel 类型未标记，与参数数量/位置无关。
- **误诊二（CrossCore 自置自等死锁）**：曾把运行挂起（`EXIT 124`）归因为 CUBE 核 `CrossCoreSetFlag` 后自己 `CrossCoreWaitFlag` 等同一个 flag 的"自置自等"死锁，并给 VEC 核加了 `CrossCoreSetFlag(softmaxReady)`。改后仍挂起。真因是 kernel-type 未标记导致 CUBE+VEC 两类核根本没有被同时启动；CrossCore 配对语义仅为 agent 推理，未被 `tool_observed` 验证，不作为本卡结论。
- **误诊三（debug 哨兵写 gO 尾）**：曾把 `*dbgFlag = 0xDEADBEEF` 写到 `gO + 65536`（注释假设输出固定 128×128 float = 65536 字节）。改 embed/shape 后 `oFileSize` 变化，哨兵会污染真实输出。已改为写到 `workspace + 128` 独立 debug buffer。

## 预防

- 直调 kernel 若需要同时启动 CUBE 与 VEC 核（直接在 CUBE 核写 L0A/L0B/L0C mmad、而非用 `matmul::Matmul` 高阶 API 走 KFC），必须在 kernel 函数体内用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_*)` 显式标记 mix 类型，不能依赖文件作用域裸赋值。
- 看到 `kernel type of __global__ func ... is not marked` warning 时，先排查是否缺 `KERNEL_TASK_TYPE_DEFAULT(...)`；不要被 `extract_host_stub.py` 的正则文本匹配误导去写裸赋值。

## 坏实践

- 把 `__enable_feature_for_compile_default = ...;` 当作 C++ 语句写到 `.asc` 文件顶部（banned）。
- 看到 build 工具正则匹配某标识符，就假定该标识符可作为合法 C++ 写进源码——正则文本匹配 ≠ 语法合法。

## 失效边界

- `platforms: [a2]` 只覆盖本 trace 实测的 Ascend910B3 + CANN 9.2.0 + dav-2201。
- A3、Ascend950 未验证，**未验证，推断可迁移**：同一 `extract_host_stub.py` 与 `KERNEL_TASK_TYPE_DEFAULT` 宏机制在其他 build mode 下应当相近，但各自的 `STR_TO_KERNEL_TYPE_*` 词表与 `return None` 分支需复测后再扩展 `platforms`。
- 仅适用于直调（`ops-direct-invoke`）kernel；注册算子（`ops-registry-invoke`）走 op-host/tiling 注册流程，不由此 build 工具链推导 kernel 类型，本卡不适用。
