---
type: "Runbook"
title: ST axes输入按目标dtype宽度逐元素序列化
description: aclnn ST 用例构造 axes 这类标量列表输入时，把 int64_t 原始数组按 nAxes×elemSize(目标dtype) 直接 aclrtMemcpy，INT32 多元素时设备侧只拿到第一个元素+零填充（{1,2} 读成 {1,0}）kernel 按错误轴算出错值；修复为按目标 dtype 宽度逐元素写字节的 buildAxesBytes。
tags: [task.testing, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# ST axes输入按目标dtype宽度逐元素序列化

## 现象

EuclideanNorm（L2 范数 reduce）算子的 aclnn ST 里，用例表用 `int64_t axesRaw[kMaxAxes]` 存轴索引、用例的 axes 张量 dtype 可为 ACL_INT32 或 ACL_INT64。Branch-0（tail-A）kernel 与 tiling 修复后重跑 ST，`[2,3,4,5] axes=[1,2] fp16 keep_dims` 用例失败但形态特殊——不是全 0，而是错值：`FAILED: exceeds rtol/atol (elem0 dev=150.375000 golden=115.437500)`。同一轮里 `[2,3,4] axes=[1,2] fp32` 输出 0.000000（期望 25.495098，该例另有归因）。dev=150.375 恰是只按 dim=1（shape 3）reduce、漏掉 dim=2（shape 4）的量级，提示设备侧实际收到的 axes 是 `[1,0]` 而非 `[1,2]`。当时测试代码的 axes 上传是 `aclrtMemcpy(devAxes, axesBytesTotal, tc.axesRaw, axesBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE)`，其中 `axesBytesTotal = nAxes * elemSizeOf(tc.axesDt)`——dtype 是 int32、源头数组却是 int64。

## 适用场景

- 工程形态：aclnn ST / 自建 aclnn 诊断程序，为算子构造 int32/int64 可变 dtype 的标量列表型输入（axes/dims/shape 之类）。
- API 路径：`aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)` + `aclCreateTensor`（ACL_INT32/ACL_INT64 输入张量）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter 一致）。
- 适用粒度：platform_wide——任何"host 侧用宽类型数组 + 目标张量窄 dtype"的字节流构造场景，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：axes dtype 与存放数组 dtype 一致（如 int64 数组 + int64 张量）时无此问题（本轨迹 int64 用例一直正确）；float 等非整型 dtype 未验证。

## 根因

`tc.axesRaw` 是 `int64_t[]`，而上传字节数按目标张量 dtype 计算：`axesBytesTotal = nAxes * elemSizeOf(axesDt)`。INT32 axes、nAxes=2 时拷 8 字节，覆盖的是 int64 数组第一个元素的 8 字节（值 1 的低 4 字节 + 高 4 字节 0），设备侧 int32[2] 读回 `{1, 0}`——第二根轴变成 0，kernel 按 `[1,0]` 计算得到错值（150.375 而非 115.4375）。错误只出现在"多元素 + 窄 dtype"组合；单元素 int32 或 int64 用例都碰巧正确，极易漏检。

## 解决配方

修复前（两处调用点，runPrecisionCase / runErrorCase 同病）：

```cpp
    if (axesBytesTotal > 0) {
        CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, tc.axesRaw,
                              axesBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE));
    }
```

```cpp
    CHECK_ACL(aclrtMemcpy(devX, inBytesTotal, inBytes.data(), inBytesTotal,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, tc.axesRaw, axesBytesTotal,
                          ACL_MEMCPY_HOST_TO_DEVICE));
```

修复后（新增按 dtype 宽度逐元素写字节的 helper，两处调用点统一走它）：

```cpp
// ---------------------------------------------------------------------------
// Build the axes tensor host bytes in the case's axes dtype.
// IMPORTANT: tc.axesRaw is int64_t[], but the axes tensor dtype may be int32 or
// int64. Each axis value must be written in the correct dtype width — copying
// raw bytes from the int64_t[] array would corrupt multi-element int32 axes
// (e.g. {1,2} as int32 would be read back as {1,0} because only the first 8
// bytes — the low half of int64(1) plus zeros — get copied).
// ---------------------------------------------------------------------------
static void buildAxesBytes(const Case& tc, std::vector<uint8_t>& bytes)
{
    size_t es = elemSizeOf(tc.axesDt);
    bytes.resize((size_t)tc.nAxes * es);
    for (int i = 0; i < tc.nAxes; ++i) {
        if (tc.axesDt == ACL_INT64) {
            int64_t v = tc.axesRaw[i];
            std::memcpy(bytes.data() + (size_t)i * es, &v, sizeof(v));
        } else { // ACL_INT32 (and any other width: fall back to int32)
            int32_t v = static_cast<int32_t>(tc.axesRaw[i]);
            std::memcpy(bytes.data() + (size_t)i * es, &v, sizeof(v));
        }
    }
}
```

```cpp
    if (axesBytesTotal > 0) {
        std::vector<uint8_t> axesHostBytes;
        buildAxesBytes(tc, axesHostBytes);
        CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, axesHostBytes.data(),
                              axesBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE));
    }
```

```cpp
    CHECK_ACL(aclrtMemcpy(devX, inBytesTotal, inBytes.data(), inBytesTotal,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    {
        std::vector<uint8_t> axesHostBytes;
        buildAxesBytes(tc, axesHostBytes);
        CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, axesHostBytes.data(),
                              axesBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE));
    }
```

## 设计决策

- 修复方向是"构造正确的输入字节"而不是放宽容差或改期望值——错值是测试自身污染了被测输入，golden 没有错；放宽容差只会掩盖真实回归。
- helper 内 int32 之外的非 int64 dtype 兜底按 int32 写（当前算子 axes 只有这两种 dtype，枚举外的取值本就会被拒绝）。
- 负数轴索引（如 `[-2,-1]`）经 int64 数组 + int32 张量同样走逐元素窄化，行为正确（本轨迹负索引 int64 用例一直通过）。

## 预防

- 构造 aclnn 输入张量字节流时，永远按"目标张量 dtype 宽度逐元素序列化"，不要对宽类型源数组做整块 memcpy——只要 `elemSize(目标) != sizeof(源元素)`，多元素必错。
- 用例表存 `int64_t`（覆盖面最宽）没问题，但序列化必须显式 `static_cast` 到目标宽度。
- 给"多元素窄 dtype 输入"至少留一个用例（单元素用例探测不到这类截断）。

## 验证信号

验证来自 EuclideanNorm Branch-0 回修轮的 aclnn ST 实测（Ascend950PR，CANN 9.1.0）：

- 修复前：`[2,3,4,5] axes=[1,2] fp16 keep_dims` 用例 FAILED，dev=150.375 vs golden=115.4375（错值非全 0）（来源平台：950；证据等级：tool_observed）。
- 修复后同用例：PASSED（maxRel=0.000e+00）；key=0 精度用例合计 5/5 PASSED（来源平台：950；证据等级：tool_observed）。
- 修复后多轮全量回归（build + TilingUT 19/19 + 9/9 + ST run.sh exit 0）保持绿（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖整型 dtype（int32/int64）；其他宽度或 float 型"标量列表"输入未验证，机制（按 dtype 宽度逐元素写）推断可迁移。
- 该修复属于测试脚手架正确性，不改变被测算子任何行为；若设备侧仍见错值，应另寻根因（如 tiling 读 axes 的 placement 问题）。
- 其余平台未验证（未验证，推断可迁移——字节序/宽度问题是宿主侧通用事实）。

**未决项**：

- 无（根因已追到字节层并实测闭环）。
## 误诊

- 该错值一度混在"Branch-0 ST 多用例失败"清单里被整体归因为 kernel/tiling 问题；逐例分类后才发现该用例的错值（非全 0）与全 0 用例根因不同——全 0 是路由到桩 kernel，本用例是测试自身把 axes 传错。
