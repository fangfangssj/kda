---
type: Guide
title: Tiling 整数范围证明与守卫语义
description: 修正取绝对值、类型提升、ceildiv 和尾部减法示例，区分算术安全与参数合同。
tags:
- task.implementation
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-red-line.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-perf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-secure.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# Tiling 整数范围证明与守卫语义

Shape、字节数、offset 和循环次数需保持整数语义。先用窄类型相乘再赋给 int64_t/uint64_t 已无法恢复丢失值；应在运算前提升，并验证最终宽类型也够用。uint32_t 的 65536*65536 回绕为 0；有符号溢出属于未定义行为。GM 大小并非只能用 int64_t，满足地址 API、符号与可表示范围的宽类型同样可用。

来源将 32*32*8192*128*2 写成 54GB，正确值为 2147483648 字节，即 2GiB。它超过 int32_t 正数上限，尚未超过 uint32_t 上限；不能用算错的例子证明某个更大边界。2^24 以上并非每个整数都能被 FP32 精确表示，尺寸计算转浮点会引入新的舍入风险；整数运算也不天然比浮点更快。

对已有效得到的 int64_t x，来源 static_cast<uint64_t>(-x) 在 x=INT64_MIN 时仍先有符号取反溢出。可用无符号模运算表达幅值：
```cpp
uint64_t ux = static_cast<uint64_t>(x);
uint64_t magnitude = x < 0 ? uint64_t{0} - ux : ux;
```
此处无符号回绕是有意的数学表示；若 x 来自 a-b，a-b 自身仍要先证明不会有符号溢出，不能只保护后面的 abs。

对非负整数 n 与正整数 d，ceildiv 可写 n/d + (n%d != 0)，避免 (n+d-1) 的加法溢出；再做 aligned=ceildiv*d 仍需乘法容量检查。尾块用 offset<=total 后的 min(tile,total-offset)，避免 total-offset 下溢以及 offset+tile 溢出。双缓冲的 (loop-1)%N 需 N>0 且首轮特殊处理；N 为 2 的幂不意味着下溢后得到的槽位语义正确。

size_t 与有符号量混算可能将负值转为大正数，具体依赖 usual arithmetic conversions；零乘积仍为零，来源称“零也回绕成极大值”不成立。非负 shape 合法性需在转换前确定，符号一致也不保证连乘安全。

安全范围来自真实输入合同、生成端校验及实际可达分支，不能仅凭 TilingData、constInfo 字段名或 __aicore__ 参数认定已校验。循环 i<bound 还需 bound<=数组容量。Host 可建立供 Device 消费的不变量，需证明所有入口确实经过它；运行时 GM 序列长度和可选地址另行处理。

guard 必须保护实际除数：actS1Size 非零不证明 actS1Size/mBaseSize 的 mBaseSize 非零。max(div,1) 虽避免除零，却可能改变合法零工作量或错误输入语义；assert 在 Release 消失时不能作为运行保护，moe 目录没有特殊语言豁免。多核早退还需保留必要参与者通知。
