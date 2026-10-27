---
type: Guide
title: CATLASS tile 的 L1、L0 与 UB 分项预算
description: 按输入类型、累计类型和实际缓冲阶段分别计算片上资源，避免混淆 L1 与 UB 或遗漏双缓冲份数。
tags:
- task.design
- topic.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/00-hardware-arch.md
  role: primary
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms:
- a2
- '950'
---

# CATLASS tile 的 L1、L0 与 UB 分项预算

CATLASS 的 L1、L0A、L0B、L0C 与 UB 分别具有独立用途和容量约束。L1 中 A/B 输入的暂存不能记在 Vector UB 预算中；L0C 的累计类型也不能直接使用输入 dtype 的字节数。

对来源中的矩形 tile 方案，记 L1 tile 为 `(m1,n1,k1)`、L0 tile 为 `(m0,n0,k0)`，A/B 元素字节数为 a/b，累计元素字节数为 c，实际缓冲份数分别为 SA1、SB1、SA0、SB0、SC0。基础占用可写为：

```text
L1 输入 = m1*k1*a*SA1 + n1*k1*b*SB1
L0A = m0*k0*a*SA0
L0B = n0*k0*b*SB0
L0C = m0*n0*c*SC0
```

这些是按来源布局推导的基本数据区，仍须追加实际 layout 的补齐、bias、scale、其他常驻区和实现要求。Epilogue 的 UB 另列输入、输出、临时区和缓冲份数。

例如 FP16 输入、FP32 累计，L1 `<128,256,256>` 且 A/B 各双缓冲，输入占用为 384 KiB。L0 `<128,256,64>` 且 A/B 各双缓冲时，L0A 为 32 KiB、L0B 为 64 KiB；单份 L0C 为 128 KiB。左右矩阵的 m/n 不同，不能只计算 A 侧就宣称两侧都满足约束。

源 reference 列出了 AtlasA2 与 Ascend950 的 ArchTag 常量；它们属于该资料描述的库实现，不能替代运行环境或目标版本的资源查询。`m0=m1`、`n0=n1` 也只是所述 Common 模板的约束，不是所有矩阵内核必须满足的等式。

验证时将每一笔预算对应到实际初始化的缓冲和消费者，检查最坏 dtype、tile、尾块与阶段并存情况；数学预算通过后还需编译和运行验证。资源核算与组件选择应一同进行，见 [组件契约](component_contracts.md)。
