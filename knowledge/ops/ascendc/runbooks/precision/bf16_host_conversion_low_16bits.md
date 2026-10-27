---
type: "Runbook"
title: "ST造BF16输入误取fp32低16位致MERE=0.149"
description: C++ 测试造 BF16 输入时 memcpy 取 round 后 fp32 位型的低 16 位而非高 16 位，喂给 kernel 的输入整体错误（MERE=1.49e-01）；改为 rounded>>16 后 BF16 用例通过。
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T00:49:13Z'
platforms: ["950"]
---

# ST造BF16输入误取fp32低16位致MERE=0.149

## 现象

算子扩展多 dtype 支持后跑全量 ST：L0_010（FP16+INT64 硬标签 mean，shape (8,5)）通过，而结构完全相同的 L0_011（BF16+INT64 硬标签 mean）失败：

```
[Real] L0_010 - fp16+int64硬标签 reduction=mean
  [PASS] ...
[Real] L0_011 - bf16+int64硬标签 reduction=mean
  [FAIL] MERE=1.49e-01 (>= 7.81e-03), MARE=1.49e-01 (>= 7.81e-02)
```

此时 kernel 的 FP16 与 BF16 走同一段代码：`GetFp32WorkTensor` 里 `Cast(xFp32, xLocal, RoundMode::CAST_NONE, cAlign_)` 升 fp32 计算、输出侧 `Cast(outLocal, floatSrc, RoundMode::CAST_ROUND, 1)` 舍回——FP16 过、BF16 不过，表面指向"BF16 Cast 路径有问题"。

## 适用场景

- 工程形态：aclnn ST C++ 测试工程的 host 侧数据构造（FP32 生成 → 手工位级转 BF16 喂 NPU）。
- API 路径：测试 harness 的 BF16 位级转换（无 ACL API，纯 memcpy/位操作）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（BF16 位布局是格式事实，与平台无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：FP16 转换分支（半精度位级布局不同，本轨迹该分支正确）；使用框架 API（如 aclFloat16/aclBfloat16 转换工具）的场景不涉及本坑。

## 根因

测试端把 FP32 原数据转 BF16 时位级操作写错。BF16 的定义是 **fp32 位型的高 16 位**（round-to-nearest-even 后截取），而错误代码 `std::memcpy(&host_buf[i * 2], &rounded, 2)` 拷的是 `rounded`（uint32 位型）的**低 16 位**——那是尾数低位，与真实 BF16 值毫无关系。kernel 收到的是逐元素完全错误的"BF16"输入，按错误输入计算出错误输出，误差 15%（数值近乎随机，与精度策略无关）。FP16 分支用的是另一段（正确的）转换代码，所以 L0_010 不受影响。

## 解决配方

修复前（verbatim）：

```cpp
} else { // BF16
    for (size_t i = 0; i < num_elems; ++i) {
        uint32_t f32;
        std::memcpy(&f32, &src[i], sizeof(f32));
        uint32_t lsb = (f32 >> 16) & 1;
        uint32_t rounded = (f32 + 0x7FFF + lsb) & 0xFFFF0000;
        if ((f32 & 0x7F800000) == 0x7F800000) rounded = f32 & 0xFFFF0000;
        std::memcpy(&host_buf[i * 2], &rounded, 2); // 取低16位即BF16
    }
}
```

修复后（verbatim，仅最后取位一步改正）：

```cpp
} else { // BF16
    for (size_t i = 0; i < num_elems; ++i) {
        uint32_t f32;
        std::memcpy(&f32, &src[i], sizeof(f32));
        uint32_t lsb = (f32 >> 16) & 1;
        uint32_t rounded = (f32 + 0x7FFF + lsb) & 0xFFFF0000;
        if ((f32 & 0x7F800000) == 0x7F800000) rounded = f32 & 0xFFFF0000;
        // BF16 is the upper 16 bits of FP32 (round-to-nearest-even)
        uint16_t bf16 = static_cast<uint16_t>(rounded >> 16);
        std::memcpy(&host_buf[i * 2], &bf16, sizeof(bf16));
    }
}
```

round-to-nearest-even 的骨架（`lsb` 进位 + NaN/Inf 保护）原本就正确，只有取位方向错。这是测试脚手架的造数 bug 修复，不涉及任何放宽容差或修改期望值。

## 设计决策

- 判别顺序的教训（见误诊）：同构 dtype 一过一不过时，应先 diff 测试端两条 dtype 的数据构造分支，再怀疑 kernel Cast 路径——本轨迹在 kernel 侧排查了长清单假设后才回到测试端。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

以下信号来自该算子多 dtype 全量 ST 上板运行：

- 修复前：L0_011 BF16 mean MERE=1.49e-01（阈值 7.81e-03）；同期 L0_010 FP16 mean PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：L0_011 通过（阈值内），L0_001~L0_013 全部 L0 用例（含 BF16 none/mean/sum 与大 shape BF16）通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"host 侧手工位级转 BF16"的写法；使用框架转换 API 的代码不适用。
- 修复后的转换对次正规/NaN/Inf 沿用原骨架的保护分支，未逐一单独验证边界值。
- 其他平台未验证，推断可迁移（BF16 位布局是格式事实，未验证，推断可迁移）。

**未决项**：

- 无（本根因已闭环；UB→GM 32 字节写越界的疑点属于另一独立问题，本轨迹未追）。
## 坏实践

- `memcpy(&dst, &src_u32, 2)` 式"取低 16 位"当 BF16——BF16 是高 16 位截取；写完位级转换应与 numpy `astype(bfloat16)`（或等价参考实现）对拍一遍。
- 同构 dtype 用例一过一不过时，长时间停留在 kernel 侧排查（本轨迹的教训）。

## 误诊

- 在定位到测试端之前，先后在 kernel 侧假设并逐一排除：BF16 Cast 方向/模板实例化（`Cast<float, bfloat16_t>` 顺序）、cAlign 32 字节对齐计算、ReduceMax Level 2 对齐要求、DataCopyPad padding 段 Cast 污染、OutputScalar 的 count=1 Cast、UB→GM 32 字节写越界等——全部无果；根因在测试端造数。
