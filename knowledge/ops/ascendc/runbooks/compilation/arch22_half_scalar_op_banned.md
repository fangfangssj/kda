---
type: "Runbook"
title: "arch22 禁 half 标量运算，NaN 检测须位级"
description: "ascend910b（arch22）kernel 中对 half 做标量比较（含 v != v NaN 自比较）报 \"half precision operation is not allowed in aicore function\"（arch35 无此限制）；NaN 检测须改用 union 位级判断（指数全 1 且尾数非 0），ld.lld cannot open .o 为次生噪声。"
tags: [task.debugging, topic.compilation, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["a2", "950"]
---

# arch22 禁 half 标量运算，NaN 检测须位级

## 现象

在给自定义算子 BenchNoHarnessMaximum 增加 NaN 传播能力（需求：与 torch.maximum 一致，任一输入为 NaN 则输出 NaN；AscendC::Max 文档未写明 NaN 行为）时，第一版实现放在 kernel 头文件（arch22 与 arch35 各一份）里，用经典的 `v != v` 自比较做 NaN 检测：

```cpp
template <typename F>
__aicore__ inline void PropagateNanImpl(AscendC::LocalTensor<F>& yLocal,
                                         const AscendC::LocalTensor<F>& x1Local,
                                         const AscendC::LocalTensor<F>& x2Local,
                                         int64_t count)
{
    if constexpr (std::is_same_v<F, half> || std::is_same_v<F, float>) {
        for (int64_t i = 0; i < count; i++) {
            F v0 = x1Local.GetValue(i);
            F v1 = x2Local.GetValue(i);
            // NaN 检测：v != v 仅对 NaN 为 true（IEEE 754 标准）
            if (v0 != v0 || v1 != v1) {
                // 任一输入为 NaN，输出 NaN（取第一个 NaN 值以保留 NaN 类型）
                yLocal.SetValue(i, (v0 != v0) ? v0 : v1);
            }
        }
    }
}
```

ascend950（arch35）构建一次通过；紧接着构建 ascend910b（arch22）时失败。直接看构建尾部只见一堆 `ld.lld: error: cannot open .../kernel_meta/BenchNoHarnessMaximum_....o: No such file or directory`；执行 `--make_clean` 清理后重建 ascend910b，真实报错才暴露：

```
/workspace/.../op_kernel/arch22/bench_no_harness_maximum.h:83:20: error: half precision operation is not allowed in aicore function
1 error generated.
```

报错行正是 `if (v0 != v0 || v1 != v1)`——对 `half` 类型做标量 `!=` 比较。

## 适用场景

- 工程形态：AscendC kernel 开发，同一份 kernel 代码需要同时编译 ascend910b（arch22）与 ascend950（arch35）。
- API 路径：`LocalTensor::GetValue/SetValue` 标量访问 + half/float 类型上的标量比较运算；IEEE 754 half/float 位格式。
- 验证过的平台与 CANN 版本：ascend910b（arch22，CANN 9.1.0）触发报错，位级改写后编译通过；ascend950（arch35）同代码构建通过（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——arch22 编译器的 half 标量运算禁用约束，与具体算子无关。
- 路线优先级：recommended_fastpath（跨架构一致代码的必经写法）。
- 明确不适用 / 未验证场景：float 上的标量比较在 arch22 本身允许（本卡统一改位级是为代码一致性）；纯 arch35 目标可直接用 `v != v`（本轨迹 arch35 首版即编译通过）；向量级 NaN 判断（Compare/Select API 组合）未在本轨迹尝试。

## 根因

arch22（ascend910b/dav_c220）的 kernel 编译器禁止在 `__aicore__` 函数内对 `half` 类型做标量运算（包括 `operator!=`），报 "half precision operation is not allowed in aicore function"；arch35（dav_3510）无此限制。`v != v` 这种逐元素 NaN 自比较是纯标量路径，恰好踩中该禁用。NaN 判定本身不依赖浮点比较——按 IEEE 754 位格式用整数位检查即可：half 的 NaN = 指数(5bit)全 1 且尾数(10bit)非 0；float 的 NaN = 指数(8bit)全 1 且尾数(23bit)非 0。

## 解决配方

修复前（见上文 `## 现象` 中首版 `PropagateNanImpl`：`v0 != v0 || v1 != v1` 标量自比较，arch22 编译报错）。

修复后（union type punning + 整数位检查，arch22/arch35 通用）：

```cpp
template <typename F>
__aicore__ inline void PropagateNanImpl(AscendC::LocalTensor<F>& yLocal,
                                         const AscendC::LocalTensor<F>& x1Local,
                                         const AscendC::LocalTensor<F>& x2Local,
                                         int64_t count)
{
    if constexpr (std::is_same_v<F, half>) {
        // half NaN 位级检测：exponent(5bit)=0x7C00, mantissa(10bit)!=0
        for (int64_t i = 0; i < count; i++) {
            half v0 = x1Local.GetValue(i);
            half v1 = x2Local.GetValue(i);
            union { half h; uint16_t u; } cvt0, cvt1;
            cvt0.h = v0;
            cvt1.h = v1;
            bool nan0 = ((cvt0.u & 0x7C00U) == 0x7C00U) && ((cvt0.u & 0x03FFU) != 0);
            bool nan1 = ((cvt1.u & 0x7C00U) == 0x7C00U) && ((cvt1.u & 0x03FFU) != 0);
            if (nan0) {
                yLocal.SetValue(i, v0);
            } else if (nan1) {
                yLocal.SetValue(i, v1);
            }
        }
    } else if constexpr (std::is_same_v<F, float>) {
        // float NaN 位级检测：exponent(8bit)=0x7F800000, mantissa(23bit)!=0
        for (int64_t i = 0; i < count; i++) {
            float v0 = x1Local.GetValue(i);
            float v1 = x2Local.GetValue(i);
            union { float f; uint32_t u; } cvt0, cvt1;
            cvt0.f = v0;
            cvt1.f = v1;
            bool nan0 = ((cvt0.u & 0x7F800000U) == 0x7F800000U) && ((cvt0.u & 0x007FFFFFU) != 0);
            bool nan1 = ((cvt1.u & 0x7F800000U) == 0x7F800000U) && ((cvt1.u & 0x007FFFFFU) != 0);
            if (nan0) {
                yLocal.SetValue(i, v0);
            } else if (nan1) {
                yLocal.SetValue(i, v1);
            }
        }
    }
}
```

定位方法（可复用）：arch22 构建失败且尾部只见 `ld.lld: error: cannot open ....o` 时，先 `bash build.sh --make_clean` 再按目标 soc 重建——ld.lld 缺 .o 是"某个 kernel 编译单元先失败、链接器随后找不到产物"的次生噪声，真实报错（带源文件行号）在清理重建后才会出现在日志前部。

## 设计决策

- 为什么位级而不是把 half 先 Cast 到 float 再比较：位级判断对 half/float 各写一个分支即可，无额外 Cast buffer/指令；且 union type punning 在 kernel 编译器下可移植（`__builtin_memcpy`/`reinterpret_cast` 在 aicore 环境的可用性未验证，union 是最保守写法）。
- 为什么 float 分支也改位级：arch22 上 float 标量比较虽允许，但两架构共用同一份头文件，统一写法避免"half 走位级、float 走比较"的双轨维护。
- 为什么不用 `std::isnan`：AscendC kernel 环境下标准库可用性受限（`std::isnan` 不可用，需手工检测）。
- 背景：AscendC::Max 的 NaN 行为文档未明确；若硬件按 `(src0 > src1) ? src0 : src1` 实现，则 Max(NaN, x) 返回 x 而非 NaN。此为设计假设，故在 Max 之后追加逐元素 NaN 传播修正。
- 当前方案路线档位：recommended_fastpath（跨架构一致代码下无更简单的合规写法）。

## 系统信息

- 硬件/软件：构建机 CANN 9.1.0，目标 soc：ascend910b（arch22）与 ascend950（arch35）；NPU 实机为 Ascend950PR。
- 相关文件：`op_kernel/arch22/bench_no_harness_maximum.h`、`op_kernel/arch35/bench_no_harness_maximum.h`（PropagateNanImpl 及 ComputeNormal/ComputeBcastDup 调用点）。
- 构建命令：`bash build.sh --soc=ascend910b -j8` / `bash build.sh --soc=ascend950 -j8`（交叉切换目标 soc 前先 `bash build.sh --make_clean`）。

## 预防

- 跨 arch22/arch35 的 kernel 代码中，凡涉及 half 的逐元素标量判断（NaN 检测、符号判断等）一律预先写成位级整数判断。
- 构建 arch22 失败且只见 ld.lld cannot open .o 时，先清理重建拿到真实报错，不要围绕链接器错误排查。

## 验证信号

以下信号来自 BenchNoHarnessMaximum 算子 NaN 传播功能开发会话：首版在 arch22 编译失败，位级改写后分架构重建验证。

- 修复前：ascend950 构建成功；ascend910b 构建失败，`--make_clean` 重建后暴露 `op_kernel/arch22/bench_no_harness_maximum.h:83:20: error: half precision operation is not allowed in aicore function`，其前首版构建尾部仅有 `ld.lld: error: cannot open ..._1.o` 等次生链接错误（来源平台：a2；证据等级：tool_observed）。
- 修复后：ascend910b 构建成功（`[100%] Built target binary`，6 个 kernel 二进制生成）；ascend950 同代码 `--make_clean` 重建成功（来源平台：a2, 950；证据等级：tool_observed）。
- 修复后算子包 `--force` 安装成功（来源平台：950；证据等级：tool_observed）。

## 失效边界

- NaN 传播的 NPU 功能验证（含 NaN/inf 特殊值的 ST 用例）在本开发阶段尚未执行，仅有编译级验证——位级检测逻辑本身的正确性待上板确认。
- 位掩码基于 IEEE 754 binary16/binary32 标准格式；bfloat16 的 NaN 位格式（指数 8bit）不同，若要支持 bf16 需另行按 `0x7F80/0x007F` 推导（未验证）。
- union type punning 依赖 half 与 uint16_t、float 与 uint32_t 尺寸一致（成立）；C++ 标准层面 union 读非活跃成员属 UB 但在 kernel 编译器实测可用。
- "arch22 禁 half 标量运算"在 CANN 9.1.0 实测；其他版本约束可能变化（未验证，推断可迁移）。

**未决项**：

- 优先项：上板执行含 NaN/inf 的 ST 用例，确认 PropagateNanImpl 的功能正确性（本阶段仅编译验证）。
- 优先项：实测 AscendC::Max 硬件指令的 NaN 行为（返回 NaN 还是另一操作数），若硬件本身已 NaN 传播则该修正可简化。
- 向量化 NaN 判断（Compare/Select 组合替代标量循环）的性能取舍未评估。
## 坏实践

- 在 arch22 目标的 kernel 里对 half 用 `==/!=/<` 等标量运算符（包括 `v != v` NaN 检测）。
- 见到 `ld.lld: error: cannot open *.o` 就去查链接配置或清理策略以外的问题——先 `--make_clean` 重建看真实编译错误。

## 误诊

- 首次 arch22 构建失败时，日志尾部同时出现 `RuntimeError: Kernel Compilation Error: ... Kernel File .../bench_no_harness_maximum_arch35.cpp`（构建 ascend910b 却指向 arch35 入口文件），一度像构建目录串台；`--make_clean` 清理重建后报错正确指向 arch22 头文件的真实 half 运算错误，确认主因是编译失败 + 次生链接噪声，串台指向为构建系统报错文本的误导（未进一步追查该文本来源）。
