---
type: "Runbook"
title: base=1.000001f 落入 |base-1|<1e-6 容差分支
description: TilingFunc 按 fabs(base-1.0f)<1e-6f 路由 base==1 优化分支时，float 字面量 1.000001f 实际值 1.0000009537 距 1.0 仅 9.537e-7，仍路由 base==1 分支；0.999999f 反而落在容差外。边界用例取值必须按 float 实际值计算，不能按十进制直觉。
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# base=1.000001f 落入 |base-1|<1e-6 容差分支

## 现象

给带 float 属性 `base` 的 Elementwise 指数族算子（y = base^(scale*x+shift)）补 tiling 分支 UT：TilingFunc 的分支路由规则是"base≤0 → 自然指数分支；base>0 且 |base-1|≥1e-6 → 自定义底数分支；|base-1.0f|<1e-6f → base==1 优化分支"。设计"容差边界"用例时，按十进制直觉选了 `base=1.000001f`（意图：刚好超出 1e-6 容差、应走自定义底数分支）和 `base=0.999999f`（意图：容差内、应走 base==1 分支）。在 x86 主机上用 g++ 编译最小探针复现分支判定（探针代码见"判别方法"），输出与直觉完全相反：

```
base=1.000001f     actual=1.0000009537  diff=9.5367431641e-07  diff<1e-6=true   branch=3
base=0.999999f     actual=0.9999989867  diff=1.0132789612e-06  diff<1e-6=false  branch=2
base=1.0+1e-7f     actual=1.0000001192  diff=1.1920928955e-07  diff<1e-6=true   branch=3
base=1.0+5e-7f     actual=1.0000004768  diff=4.7683715820e-07  diff<1e-6=true   branch=3
```

`1.000001f` 的 float 最近可表示值是 1.0000009537（距 1.0 仅 9.537e-7 < 1e-6），落入 base==1 优化分支；`0.999999f` 的 float 最近值 0.9999989867 距 1.0 为 1.013e-6 > 1e-6，反而落入自定义底数分支。

## 适用场景

- 工程形态：算子 TilingFunc 按 float 属性做分支路由（TilingKey 选择）+ 对应 op_host UT / ST 边界用例设计；任何"数值属性 + 容差比较"决定执行路径的算子。
- API 路径：op_host TilingFunc 内 `std::fabs(attr - threshold) < eps` 判定 + `ASCENDC_TPL_SEL_PARAM` 分支编码；aclnn 侧 float 属性经 `static_cast<float>(double)` 传入。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（host 侧 tiling 逻辑 + x86 探针；NPU ST 全分支 29/29 佐证，与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——float 表示与容差比较语义与平台无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：double 属性（表示精度不同，边界值需按 double 重算）；容差量级接近或小于 float ulp（约 1.19e-7@1.0）时判定本身不稳定；bf16/fp16 属性先经类型转换的情形未验证。

## 判别方法

探针（x86 g++ 直接编译执行，verbatim，可复用于任何属性边界值选定前的验证）：

```cpp
#include <cstdio>
#include <cmath>
int main() {
    // Test various base values
    float testValues[] = {0.0f, -1.0f, 1.0f, 2.0f, 10.0f, 1.000001f, 0.999999f, 1.001f, 0.999f, 1.0f + 1e-7f, 1.0f + 5e-7f};
    const char* labels[] = {"0.0", "-1.0", "1.0", "2.0", "10.0", "1.000001f", "0.999999f", "1.001f", "0.999f", "1.0+1e-7f", "1.0+5e-7f"};

    for (int i = 0; i < 11; i++) {
        float base = testValues[i];
        float diff = std::fabs(base - 1.0f);
        int branch;
        if (diff < 1e-6f) {
            branch = 3;
        } else if (base > 0.0f) {
            branch = 2;
        } else {
            branch = 1;
        }
        printf("base=%-12s  actual=%.10f  diff=%.10e  diff<1e-6=%s  branch=%d\n",
               labels[i], base, diff, diff < 1e-6f ? "true" : "false", branch);
    }
    return 0;
}
```

判别规则：对每个候选边界值先算 `float 实际值` 与 `|实际值 - 阈值|`，再与容差比较；十进制字面量必须先过这一步才能当作"容差外/内"用例。本算子最终把"容差外"用例值从 1.000001f 改为 1.001f（diff=1.0e-3，远离边界），并保留 1.000001f 作为"float 精度落入容差内"的正向用例。

UT 中的固化写法（verbatim，节选自 tiling 分支边界测试）：

```cpp
// 4. base = 1.000001f → 实际路由到 branch 3（float 精度问题）
// 注意: 1.000001f 的实际 float 值为 1.0000009537, diff = 9.537e-7 < 1e-6
// 这是一个已知的浮点精度行为，与十进制直觉不同
TEST_F(BenchNoHarnessExpTilingBranchTest, branch_1_000001_float_precision_routes_to_branch3)
{
    float base = 1.000001f;
    // 验证 float 精度: 实际 diff < 1e-6
    float diff = std::fabs(base - 1.0f);
    cout << "[INFO] 1.000001f actual=" << base << " diff=" << diff
         << " < 1e-6=" << (diff < 1e-6f ? "true" : "false") << endl;
    RunBranchRoutingTest("float_precision_1_000001", base, DT_FLOAT16, 3, 0.0f);
}

// 5. base = 0.999999f → 路由到 branch 2（超出容差）
// 0.999999f 的实际 float 值为 0.9999989867, diff = 1.013e-6 > 1e-6
TEST_F(BenchNoHarnessExpTilingBranchTest, branch_0_999999_routes_to_branch2)
{
    float base = 0.999999f;
    float diff = std::fabs(base - 1.0f);
    cout << "[INFO] 0.999999f actual=" << base << " diff=" << diff
         << " < 1e-6=" << (diff < 1e-6f ? "true" : "false") << endl;
    RunBranchRoutingTest("float_0_999999", base, DT_FLOAT16, 2, logf(base));
}
```

## 根因

float32 在 1.0 附近的可表示间隔（ulp）约为 1.19e-7：十进制"1.000001"（距 1.0 恰 1e-6）不存在精确的 float 表示，最近可表示值为 1.0000009537，实际距离被"压"到 9.537e-7 < 1e-6，于是落容差内；而"0.999999"的最近可表示值 0.9999989867 距 1.0 为 1.013e-6 > 1e-6，落容差外。即：容差 1e-6 只有约 8 个 ulp 的宽度，十进制上"对称"的 ±1e-6 偏移经 float 舍入后落在容差两侧是常态而非偶然。分支路由因此与十进制直觉相反。

## 设计决策

- 边界用例取值策略改为"远离边界的明确值 + 落入容差的明确值"两档：容差外用 1.001f/0.999f（diff≈1e-3），容差内用 1.0+5e-7f；1.000001f/0.999999f 保留为"float 舍入改变路由"的专项用例而非通用边界用例。
- UT 直接断言 TilingKey（含分支编码）而非只断言输出，保证路由层回归可见。

## 预防

- 任何 `fabs(attr - x) < eps` 型分支判定，写用例前先用主机探针算出 float 实际值与实际 diff；不要用十进制字面量当"刚好在容差外"的值。
- eps 若与属性类型 ulp 同量级（float@1.0 ulp≈1.19e-7），边界判定对字面量舍入敏感，文档与测试需明确写"实际 float 值"语义。

## 验证信号

以下信号来自该算子 tiling 分支 UT 扩展会话（Ascend950PR 工程，CANN 9.1.0）：先以 x86 探针锚定 float 实际行为，再把期望写入 UT 并全量回归。

- 探针：`1.000001f actual=1.0000009537 diff=9.5367431641e-07 <1e-6=true → branch 3`；`0.999999f actual=0.9999989867 diff=1.0132789612e-06 <1e-6=false → branch 2`（x86 g++ -std=c++17 编译执行）（来源平台：950；证据等级：tool_observed）。
- UT：`branch_1_000001_float_precision_routes_to_branch3` 与 `branch_0_999999_routes_to_branch2` 按上述期望通过（TilingKey 分别为 513 与 257），全量 32/32 用例通过（含 11 个分支路由专项）（来源平台：950；证据等级：tool_observed）。
- ST：NPU Real 模式全量 ST 覆盖 TilingKey 1/2/3 分布 14/10/5，29/29 ALL PASS，无因路由误判产生的失配（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 数值针对 float32 属性 + 1e-6 容差 @ base≈1.0；阈值/类型/容差改变后需按同法重算（未验证，方法可迁移）。
- 属性从 aclnn 的 double 入参 `static_cast<float>` 落地：double 侧与 float 侧的值在 1e-6 量级可能不同侧，跨层用例要按最终 float 值断言。

**未决项**：

- 无。
## 坏实践

- 按十进制直觉断言 1.000001f "应在容差外"——UT 会误报，或更糟：为了过测试去改容差常量。
- 把 0.999999f 当作"容差内"用例。

## 误诊

- 无（探针先行，未走弯路）。
