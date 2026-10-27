---
type: "Runbook"
title: TilingUT 基线假通过：stub 常量撞上负向期望
description: 测试先行 tiling UT 中 bool 函数的 stub 恒返回 false，与"非法输入应返回 false"的负向用例期望碰撞产生假 PASS（2/27 而非 0/27）；用实现宏门控 report() 通过计数，恢复 0-pass 基线。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T00:54:40Z'
platforms: ["950"]
---

# TilingUT 基线假通过：stub 常量撞上负向期望

## 现象

在给自定义算子 softplus_v2_grad（AscendC aclnn 工程，op_host 侧）做"测试先行"的公共 tiling 单元测试时：7 个公共 tiling 函数（PadAndSqueeze / CheckBroadcastShape / PrecomputeInputStrides / PrecomputeOutputStrides / FindSplitAxis / MultiCoreSplit / MapRankToBranch）尚未实现，UT 文件 `tests/tiling_ut/tiling_ut.cpp` 里用 `#ifndef SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED` 守卫的临时 stub 提供链接期定义（bool 函数恒返回 `false`、整型返回 `-1`，出参填哨兵值），期望基线是"公共 Tiling 未实现 → UT 期望通过数 = 0，仅要求可编译"。构建通过后直接运行 UT 可执行文件：

```text
=== §5.2 CheckBroadcastShape ===
  [FAIL] bcast_compatible
  [PASS] bcast_incompatible_inputs
  [PASS] bcast_incompatible_in_out
  [FAIL] bcast_all_ones
...
=======================================
Summary: 2 / 27 PASSED
=======================================
```

出现了 2 个不应该存在的 PASS，且都集中在 CheckBroadcastShape 的"不兼容 shape"负向用例上——此时实现代码一行都没写。

## 适用场景

- 工程形态：AscendC 自定义算子 op_host 侧 tiling 的测试先行单元测试（ops-registry-invoke 类工程：先写 UT + 链接期 stub，后写真实实现；UT 为宿主侧纯函数单测，随 `./build.sh <op>_package` 一起编出）。
- API 路径：自建 UT 可执行文件 + CMake 宏切换（`target_compile_definitions(<UT> PRIVATE <OP>_XXX_IMPLEMENTED)`）；被测函数为返回 bool/整型的纯 host tiling 函数。
- 验证过的平台与 CANN 版本：Ascend950PR（包目标平台 ascend950，宿主侧执行），CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：operator_family——任何采用"stub + 宏切换"测试先行模式的 host 侧 tiling UT 都适用，与具体算子无关。
- 路线优先级：recommended_fastpath（测试先行工作流正确性基线的必要保障，无额外成本）。
- 明确不适用 / 未验证场景：返回类型只有单一取值域外的函数（如恒返回 -1 哨兵的整型函数，天然不会撞期望）；非测试先行（实现已存在）的 UT；设备侧 kernel 测试。

## 根因

stub 对 bool 返回值函数恒返回 `false`，而负向用例的期望值恰好也是 `false`，断言 `ret == exp && !ret` 变成 `false == false && !false` = 恒真，造成假通过。

机制链：CheckBroadcastShape 的"不兼容输入"用例期望函数返回 false（校验应拒绝非法 shape）；stub 未实现时也返回 false；两者数值碰撞 → 断言通过。本质上 bool 只有 true/false 两个取值，任何常量返回的 stub 都必然与一部分用例的期望值相等——正向用例期望 true、负向用例期望 false，常量 stub 必撞其一（返回 false 撞负向、返回 true 撞正向）。因此"负向用例 + 常量 bool stub"是测试先行 UT 的结构性假阳性来源，靠改 stub 返回值无法消除。

## 解决配方

修复前（`tests/tiling_ut/tiling_ut.cpp` 的 report 函数，无条件按断言结果计 pass）：

```cpp
static void report(const char* name, bool ok)
{
    g_total++;
    if (ok) { g_pass++; printf("  [PASS] %s\n", name); }
    else    {            printf("  [FAIL] %s\n", name); }
}
```

修复后（用实现宏门控通过计数：宏未定义（stub 阶段）时任何断言结果都只计 FAIL；宏已定义（真实实现已链接）时恢复正常判定）：

```cpp
// When stubs are active (SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED not defined),
// every test is forced to FAIL — even if a stub's constant return value
// happens to match an expected value (e.g. CheckBroadcastShape returning
// false for an incompatible-shape case). This guarantees pass count = 0
// until the real functions are implemented and the macro is defined.
static void report(const char* name, bool ok)
{
    g_total++;
#ifdef SOFTPLUS_V2_GRAD_COMMON_IMPLEMENTED
    if (ok) { g_pass++; printf("  [PASS] %s\n", name); return; }
#endif
    printf("  [FAIL] %s\n", name);
}
```

配套要求：`main` 末尾保留 `return (g_pass == g_total) ? 0 : 1;`（stub 阶段 0/27 时退出码 1，可供脚本判失败）。

判别/定位步骤（本会话走通，可复用）：

1. 首跑基线必须精确核对"预期通过数"：任务口径是 0 就要逐条看是哪些用例 PASS 了，不要只看跑完没崩。
2. 对每个意外 PASS 的用例，写出断言式（如 `ret == exp && !ret`）代入 stub 常量返回值手算一遍——`false == false && !false` 恒真即确认碰撞。
3. 确认碰撞后不要动 stub 返回值（改成 true 只会把假阳性挪到正向用例），也不要删负向用例（丢失覆盖），改计分门控。

## 设计决策

- 为什么选 `#ifdef` 门控 report 而不是其他候选（本会话逐一评估过）：
  - 改 stub 返回 `true`：正向用例（期望 true）会假通过，只是把坑挪了个位置，否决；
  - 负向用例内部再附带校验一个配套正向输入（"函数须能同时区分合法/非法输入"）：单个用例内两次调用、用例间耦合，且 stub 恒 false 时 `ret_compat && !ret_incompat` 虽恒 false 但语义混杂，否决；
  - 新增探针函数 `IsCommonTilingImplemented()`（stub 返回 false、实现返回 true）每条用例先查：等价于宏门控但引入了 DESIGN 之外的接口，否决；
  - `#ifdef` 宏门控：宏本来就用于切换 stub/真实实现的链接，复用它做计分门控零接口侵入、且保证"宏定义 ⇒ 真实实现已链接"（宏定义而实现缺失会直接链接失败），选定。
- 该门控不检验实现正确性，只保证基线语义：stub 阶段 pass 恒 0；实现落地后 pass 由真实断言决定。测试代码本身在两个阶段都被完整编译和执行（调用 stub/实现、跑 oracle、做比较），只有计分口径切换。
- 与哨兵 stub 策略的关系：对整型/结构体出参可以写"与任何期望都不相等"的哨兵值（如 `INT32_MIN`、`0xDEADBEEF`，见同域 [Host Tiling 纯函数分层与 UT 同源编译](tilingfunc_pure_helper_glue_layering.md)的 stub 纪律）；但 **bool 返回值写不出哨兵**——只有 true/false 两个取值，必然撞一侧期望，此时计分门控是唯一消除结构性假阳性的手段。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 测试先行 UT 模板里把"宏门控 report"作为默认组件，连同 stub 宏守卫一起生成，而不是出了假通过再补。
- 验收口径写死"stub 阶段通过数 = 0 且退出码非 0"，把这个数字当机器检查项。
- 出现任何非预期 PASS，先代入 stub 常量手算断言，确认是否为结构性碰撞。

## 验证信号

本次开发会话（softplus_v2_grad 公共 tiling 测试先行阶段）的实测：

- 修复前首跑 `Summary: 2 / 27 PASSED`，两条假 PASS 均为 CheckBroadcastShape 负向用例（来源平台：950 包内宿主 UT；证据等级：tool_observed）。
- 修复后复跑 `Summary: 0 / 27 PASSED`、`exit code: 1`，符合预期基线（来源平台：950 包内宿主 UT；证据等级：tool_observed）。
- 公共 tiling 真实实现落地并定义宏后，同一 UT `Summary: 27 / 27 PASSED`、`tiling_ut_exit=0`（来源平台：950 包内宿主 UT；证据等级：tool_observed）。
- 同一门控模式复刻到分支 UT（`tiling_ut_rank4.cpp`，宏 `SOFTPLUS_V2_GRAD_RANK4_IMPLEMENTED`）后首跑即 `Summary: 0 / 12 PASSED`、`EXIT_CODE=1`，无假通过（来源平台：950 包内宿主 UT；证据等级：tool_observed）。

## 失效边界

- 门控宏与真实实现必须同轮定义：只定义宏不链接实现会编译失败（保护有效）；反之实现落地却忘定义宏，UT 会全 FAIL 掩盖真实结果——接入 CMake `target_compile_definitions` 后由构建系统保证一致性。
- 该机制只保证"基线语义"，不提高用例本身的断言质量；断言写错（正负期望都写反）仍会静默通过（实现阶段）。
- 纯宿主逻辑，不依赖具体 NPU 平台；在其他宿主编译环境同样成立（未验证，推断可迁移）。

**未决项**：

- 无（该假阳性机制在本会话内完全闭合：修复、复跑、实现后 27/27、模式复用 0/12 四级信号齐备）。
## 坏实践

- 为凑 0-pass 基线删掉/跳过负向用例：测试侧放松，丢掉"实现后应拒绝非法输入"的覆盖。
- 把 stub 返回值改成 true/随机值来"绕开碰撞"：bool 常量必撞一侧，随机值让基线不可复现。
- 只看 Summary 数字不看哪些用例 PASS：假通过必须逐条归因才能确认是 stub 碰撞而非实现提前生效。
