---
type: "Runbook"
title: 标量NaN自比较后端Cannot select seto
description: "kernel 标量路径 x!=x 判 NaN 触发 bisheng 后端 Cannot select: i64 = setcc seto；改为位级指数/尾数判定后编译通过。"
tags: [task.debugging, topic.compilation, topic.precision]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# 标量NaN自比较后端Cannot select seto

## 现象

Cummin 算子 AR-scan 模式（扫描维为最后一维、A0=1）用标量循环逐元素维护运行最小值，需要在 kernel 内做 NaN 判定。代码位于 `op_kernel/arch35/cummin.h`，r=1..R-1 循环内的写法是经典的 `x != x` 惯用法：

```cpp
for (uint64_t i = 1; i < R; i++) {
    T current = inputBuf.GetValue(i);

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
                  std::is_same_v<T, bfloat16_t>) {
        bool currentIsNan = (current != current);
        if (currentIsNan) {
            ...
        }
        bool minIsNan = (runningMin != runningMin);
        ...
    }

    if (current < runningMin) {
```

执行 `bash build.sh --soc=ascend950 -j8`，bisheng 后端直接崩溃：

```
fatal error: error in backend: Cannot select: 0x560a02858ad0: i64 = setcc 0x560a0431d540, 0x560a0431d540, seto:ch, op_kernel/arch35/cummin.h:390:46
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
```

先怀疑是 `bfloat16_t` 类型的 `!=` 不支持，改成先 `static_cast<float>(current)` 再比较：

```cpp
float curF = static_cast<float>(current);
float minF = static_cast<float>(runningMin);
bool currentIsNan = (curF != curF);
```

重新编译后错误只是移到 392:43，仍是 `Cannot select: ... seto:ch`——说明标量 float 的自相等比较同样不可选。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 标量路径（GetValue/SetValue 循环）需要 NaN 判定的算子（cummin/cummax/argmin 等含 NaN 语义的扫描类算子）。
- API 路径：标量比较表达式（`!=`）在 bisheng AICore 后端的指令选择；对照向量路径 `Compare(mask, src, src, CMPMODE::NE, ...)` 可用。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0（bisheng 编译器，exit code 70）。
- 适用粒度：`platform_wide`（arch35/ascend950 后端的标量代码生成行为）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：向量 Compare 的 NE 自比较不受此限制（本算子向量路径曾正常使用）；其他后端版本未验证。

## 根因/解决配方

根因：`x != x` 的 NaN 判定在标量路径会 lower 为有序不等比较（setcc seto）指令，bisheng 的 AICore 后端没有该指令的选择模式，编译期直接 `Cannot select` 崩溃；换成先转 float 再自比较不改变生成的比较指令，同样失败。向量路径的 `Compare(..., CMPMODE::NE, ...)` 走向量指令，不受影响。同族的 bf16 `setuo` 不可选（配合 `if constexpr` 剔除不需要的分支）另见 [bf16 未编译剔除触发 backend Cannot select 卡](if_constexpr_excludes_unsupported_bf16_compare.md)——本卡场景是 NaN 判定必需、不可剔除，须改写判定本身。

解决配方：用位级判定替代比较运算。IEEE 754 中 NaN 当且仅当指数字段全 1 且尾数非 0；对 fp32 位型 `(bits & 0x7FFFFFFF) > 0x7F800000` 即为 NaN。统一做法是先把值转成 float（bf16/fp16 的转换见 [bf16 标量位级转换卡](bf16_scalar_to_fp32_bit_extension.md)），再做 fp32 位判定：

修复前（触发后端崩溃）：

```cpp
if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
              std::is_same_v<T, bfloat16_t>) {
    float curF = static_cast<float>(current);
    float minF = static_cast<float>(runningMin);
    bool currentIsNan = (curF != curF);
    if (currentIsNan) {
        if (!nanSeen) {
            runningIdx = static_cast<int64_t>(i);
            nanSeen = true;
        }
        runningMin = current;
        valBuf.SetValue(i, runningMin);
        idxBuf.SetValue(i, runningIdx);
        continue;
    }
    bool minIsNan = (minF != minF);
    if (minIsNan) {
        valBuf.SetValue(i, runningMin);
        idxBuf.SetValue(i, runningIdx);
        continue;
    }
}
```

修复后（位级判定，消除 seto）：

```cpp
if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
              std::is_same_v<T, bfloat16_t>) {
    uint32_t curBits = *reinterpret_cast<const uint32_t*>(&curF);
    bool currentIsNan = ((curBits & 0x7FFFFFFFu) > 0x7F800000u);
    if (currentIsNan) {
        if (!nanSeen) {
            runningIdx = static_cast<int64_t>(i);
            nanSeen = true;
        }
        runningMinF = curF;
        valBuf.SetValue(i, FromFloat<T>(runningMinF));
        idxBuf.SetValue(i, runningIdx);
        continue;
    }
    uint32_t minBits = *reinterpret_cast<const uint32_t*>(&runningMinF);
    bool minIsNan = ((minBits & 0x7FFFFFFFu) > 0x7F800000u);
    if (minIsNan) {
        valBuf.SetValue(i, FromFloat<T>(runningMinF));
        idxBuf.SetValue(i, runningIdx);
        continue;
    }
}
```

按原始存储类型的位宽变体（无 float 中转时）：fp16 用 `(bits & 0x7FFFu) > 0x7C00u`，bfloat16 用 `(bits & 0x7FFFu) > 0x7F80u`（位段判定：fp16 指数 5 位在 bits>>10，bf16 指数 8 位在 bits>>7）。

## 设计决策

- 为什么选 fp32 位判定而不是分类型三套位判定：统一先转 float（比较与判定都在 float 域，见 [bf16 标量位级转换卡](bf16_scalar_to_fp32_bit_extension.md)）后只需一套 `0x7FFFFFFF/0x7F800000` 掩码逻辑，减少分支；分类型变体保留作为无转换场景的备选。
- 放弃的候选：`static_cast<float>` 后继续用 `!=`（实测仍 seto，失败）；向量 Compare NE 自比较（需要把标量逻辑搬到向量管线，代价大且当时向量管线另有正确性问题）。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0，bisheng（clang frontend，exit code 70），`bash build.sh --soc=ascend950 -j8`。
- 相关文件：`op_kernel/arch35/cummin.h`（AR-scan ProcessArScan 标量循环、ARA-scan 标量循环同款判定）。

## 验证信号

验证来自同一开发会话的编译迭代：位级判定落地后 `Cannot select ... seto` 错误消失，编译推进到下一处独立错误（bf16 标量 `<` 比较的 setolt，见 [bf16 标量位级转换卡](bf16_scalar_to_fp32_bit_extension.md)）（来源平台：950；证据等级：tool_observed）；全部标量路径修复完成后构建成功、包生成，NPU 上 NaN 语义用例随全量 L0 ST 21/21 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 位级判定假定浮点格式为标准 IEEE 754 fp32/fp16/bf16；若后端使用非标准 NaN 载荷（payload 全 0 的 NaN 也可能被判定为非 NaN——标准规定指数全 1 且尾数非 0 才是 NaN，掩码判定与此一致），无影响。
- `reinterpret_cast` 位型访问要求值已在本端寄存器/UB 中；对 GM 直接指针做位型 reinterpret 不适用（未验证）。
- 版本边界：本卡 seto 不可选实测于 CANN 9.1.0 的 classic op_kernel（LocalTensor GetValue/SetValue）标量路径；库内 [bf16 标量位级转换卡](bf16_scalar_to_fp32_bit_extension.md)记录的 950/CANN 9.2.0 SIMT 路径复现代码中保留了 float 标量自比较且构建成功——两例的 CANN 版本与代码路径差异尚未隔离，跨版本套用前先做最小编译探针。

**未决项**：

- seto 在标量路径不可选的官方文档/编译器 issue 佐证未查（本轨迹以实测错误与修复闭环为准）。
- 标量路径是否有其他不可选的浮点比较模式（如 unordered 大于）未系统枚举。
## 坏实践

- 在 AscendC kernel 标量路径沿用 x86/CPU 上的 `x != x` NaN 惯用法而不查后端指令支持——本例第一步和第二步（float 转换）都栽在同一指令选择上。
- 见到 `error in backend: Cannot select` 就怀疑代码逻辑：该错误是编译器后端无法为某 IR 节点选指令，定位方法是看错误里带出的源码行列（本例 390:46 精确指向 `current != current`）。

## 误诊

- 第一次修复假设"bfloat16_t 不支持 != 运算符，转 float 即可"（agent_reported）：转 float 后错误仍在，证伪；真正限制在比较指令本身的标量 seto 选择。
