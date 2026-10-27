---
type: "Runbook"
title: "ASCENDC_CUBE_ONLY 入口须 __aicore__"
description: "定义 `#define ASCENDC_CUBE_ONLY` 后 kernel 入口若用 `__global__ __cube__`，bisheng 报 core_ratio attribute needs to appear with cce kernel func；判别入口属性冲突，改回 `__global__ __aicore__`、宏保留即可过编译。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.execution_unit.cube"]
status: draft
sources: []
created_at: "2026-08-18T08:56:24Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# ASCENDC_CUBE_ONLY 入口须 __aicore__

## 现象

在构建目标名为 `ascend910b` 的环境中开发 aclnn 注册算子 `QuantBatchMatMulV3` 的 kernel。轨迹未采集可确认具体 SoC 的设备探针，因此不把泛化目标名直接映射为 a2；只有明确的 Ascend910B3 才按本库规则映射为 a2。为规避 AIV 读 workspace 的 507015，在 `op_kernel/quant_batch_mat_mul_v3.cpp` 顶部加 `#define ASCENDC_CUBE_ONLY`，让 matmul 与反量化都在 Cube 核执行。kernel 入口写作：

```cpp
__global__ __cube__ void quant_batch_mat_mul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale,
                                                GM_ADDR pertokenScale, GM_ADDR out,
                                                GM_ADDR workspace, GM_ADDR tiling)
```

`cmake --build` 报（verbatim，trace 内 tool_observed，重复多次）：

```
/tmp/qbmm_clean/op_kernel/quant_batch_mat_mul_v3.cpp:14:26: error: __attribute__((aic)),
__attribute__((aiv)), bisheng::core_ratio attribute needs to appear with cce kernel func.
```

报错指向第 14 行 kernel 入口，并未提示「入口属性选错」，容易误判为 ASCENDC_CUBE_ONLY 宏本身不被支持。

## 适用场景

- **工程形态**：aclnn/直调 kernel 工程，kernel 入口用 `ASCENDC_CUBE_ONLY` 宏走 Cube-only 执行模型。
- **API 路径**：kernel 入口属性 `__global__ __aicore__` / `__global__ __cube__` + `#define ASCENDC_CUBE_ONLY`。
- **验证过的平台与 CANN 版本**：精确平台未定位；构建目标包含 `ascend910b`，CANN 9.0.0，bisheng kernel 编译器。
- **适用粒度**：`platform_wide`——所有使用 `ASCENDC_CUBE_ONLY` 的 kernel 入口。
- **路线优先级**：`recommended_fastpath`——`ASCENDC_CUBE_ONLY` + `__global__ __aicore__`；`banned`——`ASCENDC_CUBE_ONLY` + `__global__ __cube__`（属性冲突，编译失败）。
- **不适用/未验证场景**：未定义 `ASCENDC_CUBE_ONLY` 的常规 kernel 入口（`__cube__`/`__aicore__` 的常规选择另有规则，本卡不覆盖）。

## 判别方法

### 触发特征

在已 `#define ASCENDC_CUBE_ONLY` 的 kernel 源文件上运行 `cmake --build`，末尾出现：

```
<file>:<line>:<col>: error: __attribute__((aic)), __attribute__((aiv)),
bisheng::core_ratio attribute needs to appear with cce kernel func.
```

报错锚定 kernel 入口行，且文件顶部存在 `#define ASCENDC_CUBE_ONLY`。

### 可执行探针

1. 核对失败文件的 kernel 入口属性与宏定义（trace 内 verbatim 工具命令，tool_observed 执行）：

```
grep -n "__global__\|__aicore__\|__cube__\|ASCENDC_CUBE" op_kernel/quant_batch_mat_mul_v3.cpp
```

   若同时命中 `#define ASCENDC_CUBE_ONLY` 与 `__global__ __cube__` 入口，即命中本判别。

2. 对照已知可工作的 Cube 算子入口（trace 内 verbatim 工具命令，tool_observed 执行）：

```
grep -n "__global__\|__aicore__\|__cube__\|ASCENDC_CUBE" <参考算子>/op_kernel/<kernel>.cpp
```

   trace 内对照的参考算子（conv2d_solution）入口为 `__global__ __aicore__`，不使用 `__cube__`。

### 观测

构建末尾报 `bisheng::core_ratio attribute needs to appear with cce kernel func`，锚点为 kernel 入口行（第 14 行）；探针 1 同时命中 `ASCENDC_CUBE_ONLY` 宏与 `__global__ __cube__` 入口；探针 2 显示可工作参考算子入口为 `__global__ __aicore__`。探针命令本身为 tool_observed 执行；其 grep 命中文本因 markdown_transcript adapter 重排/折叠未在事件流逐字恢复，结论（参考算子用 `__aicore__`、未用 `__cube__`）由 agent reasoning 转述并标注。

### 判读（interpretation）

`ASCENDC_CUBE_ONLY` 的语义是「执行模型只走 Cube」，由宏在编译期注入 `__attribute__((aic))`/`__attribute__((aiv))` 与 `bisheng::core_ratio` 属性以重定向执行模型到 Cube-only。bisheng 要求这类 core_ratio 属性必须出现在「cce kernel func」上，即带 `__aicore__` 的 kernel 入口。`__cube__` 是另一类核函数属性，与 `ASCENDC_CUBE_ONLY` 注入的属性叠加后不满足「cce kernel func」的归属约束，故报错。判别结论：入口须改回 `__global__ __aicore__`，`#define ASCENDC_CUBE_ONLY` 保留不变——执行模型由宏负责，不由入口属性换成 `__cube__` 负责。

## 根因

AscendC kernel 入口的「cce kernel func」由 `__aicore__` 标记。`ASCENDC_CUBE_ONLY` 宏在预处理期向入口注入 core_ratio/aic/aiv 属性以切到 Cube-only 执行模型；这些注入属性要求宿主入口本身已是 cce kernel func（`__aicore__`）。`__cube__` 与 `__aicore__` 属不同属性体系，`__cube__` 入口不被识别为 cce kernel func，注入属性无合法宿主即报错。该机制由 verbatim 编译错误中的 `core_ratio attribute needs to appear with cce kernel func` 文本与 agent reasoning 共同推断；bisheng 属性注入的逐行宏展开未在 trace 内逐字捕获。

## 验证信号

来源说明：本知识来自一次 QuantBatchMatMulV3 aclnn 搭建会话；构建命令 `cmake --build /tmp/qbmm_build --target all binary package` 实跑，core_ratio 编译错误为 verbatim tool_observed 输出；入口属性 edit 的 old/new 字符串逐字取自 trace 内 edit 事件。

- `cmake --build` 末尾报 `op_kernel/quant_batch_mat_mul_v3.cpp:14:26: error: __attribute__((aic)), __attribute__((aiv)), bisheng::core_ratio attribute needs to appear with cce kernel func`，锚定 `#define ASCENDC_CUBE_ONLY` 下的 `__global__ __cube__` 入口（来源平台：a2；证据等级：tool_observed）
- 入口属性 edit `__global__ __cube__ void quant_batch_mat_mul_v3` → `__global__ __aicore__ void quant_batch_mat_mul_v3`，oldString/newString 逐字取自 trace 内 edit 事件，`#define ASCENDC_CUBE_ONLY` 保留不变（来源平台：a2；证据等级：tool_observed）

> 改 `__aicore__` 后的「构建成功」tool_result 因 markdown_transcript adapter 把多条 build 的 tool_result 重排/折叠，未在事件流逐字恢复；agent 的「Build succeeded」口述为 agent_reported，不进 `## 验证信号`。

## 坏实践

- 误以为 `ASCENDC_CUBE_ONLY` 等价于把入口属性换成 `__cube__`——两者属性体系不同，叠加即冲突。
- 见到 `core_ratio attribute needs to appear with cce kernel func` 即去查 bisheng 版本/重装工具链——报错锚点已指向 kernel 入口行，应先核对该行属性是否与 `ASCENDC_CUBE_ONLY` 宏冲突。

## 误诊

trace 内 agent 一度据报错文本推测「`__cube__` 属性需与 `__global__` 作为 CCE kernel function 配对，ASCENDC_CUBE_ONLY 改变了编译方式」（agent_reported 推断），后经对照参考算子入口（`__aicore__`）与 grep `__cube__`/`__aicore__` 宏定义后定位为入口属性冲突。bisheng 属性注入的逐行宏展开未逐字捕获，机制层推断见 `## 根因`。

## 失效边界

- 构建目标 `ascend910b` 上实测；精确 SoC 未定位。bisheng 属性语义跨平台一致，推断可迁移到 a2/a3/950，但未独立验证，不写入 frontmatter `platforms`。
- 不同 CANN 版本若 `ASCENDC_CUBE_ONLY` 宏语义调整（不再注入 core_ratio 属性），本卡即失效——使用前先按报错信息确认。
- `ASCEND_COMPUTE_UNIT=ascend910b;ascend910_93` 同时包含两个泛化构建目标，且未跑 `npu-smi`，不足以确认具体平台；不能据此把本卡标为 a2。明确的 Ascend910B3 仍按平台概念卡映射为 a2。
