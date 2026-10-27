---
type: "Runbook"
title: 纯搬运算子 ST 字节级 golden 与 bitwise 比对
description: 纯搬运（gather/scatter/copy/slice）算子的 C++ ST CPU golden 按字节级实现——输入当字节数组逐输出元素 memcpy 对应宽度，全 dtype 共用一条路径且天然 bitwise 精确；fp16/bf16 仅生成数值/打印诊断时用位级转换；比对先 memcmp 位级相等再降级解释误差。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T01:48:53Z'
updated_at: '2026-09-17T01:48:53Z'
platforms: ["950"]
---

# 纯搬运算子 ST 字节级 golden 与 bitwise 比对

## 现象

为多 dtype gather 算子（x 与 index 的 dtype 组合覆盖 fp16/fp32/bf16/int8/int32/int64）开发 C++ ST 测试工程的 Mock 模式（无 NPU 环境下用 CPU golden 自测比对函数与用例）。两个现实问题：

1. golden 需要覆盖全部 dtype。写 fp16 golden 时发现宿主机编译器不支持 `_Float16/__fp16`：

```
g++ (Ubuntu 13.3.0-6ubuntu3~24.04.1) 13.3.0
/tmp/test_fp16.cpp:5:5: error: '__fp16' was not declared in this scope; did you mean '__bf16'?
    5 |     __fp16 h = static_cast<__fp16>(3.14f);
```

2. 若按 dtype 各写一套"解释值→计算→再编码"的 golden，路径多、易错，且浮点解释/重编码会引入与 kernel 位级行为不一致的舍入，导致 golden 自身不 bitwise。

## 适用场景

- 工程形态：AscendC 算子 ST 测试工程（C++ 原生、Mock/CPU golden 模式），尤其纯数据搬运算子（gather/scatter/copy/slice 等）或要求 bitwise 比对的用例。
- API 路径：自建 ST 框架（`test_aclnn_*.cpp` + `run.sh --mock/--real`）；fp16(1+5+10)/bf16(1+8+7) 位级转换；`memcmp` 逐元素比对。
- 验证过的平台与 CANN 版本：x86_64 宿主机（g++ 13.3.0）golden + Ascend950PR（arch35）Real 对照，CANN 9.1.0。
- 适用粒度：operator_family（纯搬运算子的 ST golden 配方）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：含算术计算的算子 golden（必须按语义计算，不能 memcpy）；ARM/Clang 等支持 __fp16 的宿主机（可直接用原生类型，但字节级路径仍兼容）；MSVC 未验证。

## 根因/解决配方

**方案（单句）**：纯搬运算子的 golden 按**字节级**实现——把输入当作字节数组，逐输出元素按索引从输入字节区 `memcpy` 对应元素宽度的字节，全 dtype 共用一条路径；fp16/bf16 仅在需要生成数值/打印诊断时用**位级转换**（不依赖编译器扩展），比对一律 `memcmp` 位级相等或 bitwise 判定。

配方分三部分：① 字节级 golden 主体（全 dtype 通用）；② fp16 位级转换（生成/诊断用）；③ bf16 同构位级转换 + bitwise 优先的比对策略。

## 完整代码

以下均 verbatim 取自事件流终版写入。

字节级 golden 主体（任意 dtype 通用，按输出坐标反解输入物理位置后逐元素 memcpy）：

```cpp
void ComputeGatherGoldenBytes(const uint8_t* xBytes, size_t elemSize,
                               const uint8_t* indexBytes, DType indexDtype,
                               uint8_t* yBytes,
                               const std::vector<int64_t>& xShape,
                               const std::vector<int64_t>& indexShape,
                               int64_t dim) {
    int64_t yNumel = GetShapeSize(indexShape);
    int64_t ndim = static_cast<int64_t>(xShape.size());
    size_t indexElemSize = DTypeToSize(indexDtype);

    auto xStrides = ComputeStrides(xShape);
    auto idxStrides = ComputeStrides(indexShape);

    for (int64_t outFlat = 0; outFlat < yNumel; ++outFlat) {
        int64_t idxVal = ReadIndexValue(indexBytes + outFlat * indexElemSize, indexDtype);

        int64_t srcFlat = 0;
        int64_t remaining = outFlat;
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = remaining / idxStrides[d];
            remaining %= idxStrides[d];
            int64_t srcCoord = (d == dim) ? idxVal : coord;
            srcFlat += srcCoord * xStrides[d];
        }

        std::memcpy(yBytes + outFlat * elemSize,
                    xBytes + srcFlat * elemSize, elemSize);
    }
}
```

fp16 位级转换（float↔bits，round-to-nearest-even，不依赖编译器扩展）：

```cpp
uint16_t FloatToFp16Bits(float fv) {
    uint32_t x;
    std::memcpy(&x, &fv, sizeof(float));
    uint16_t sign = static_cast<uint16_t>(x >> 31);
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 112;  // half-biased: unbiased + 15
    uint32_t frac = x & 0x7FFFFFu;

    if (exp >= 0x1F) {
        // overflow or inf/nan
        if ((x & 0x7F800000u) == 0x7F800000u) {
            if (frac == 0) {
                return static_cast<uint16_t>(sign << 15) | 0x7C00u;  // inf
            }
            uint16_t nanFrac = static_cast<uint16_t>(frac >> 13);
            if (nanFrac == 0) nanFrac = 1;  // 确保是 nan（非零尾数）
            return static_cast<uint16_t>(sign << 15) | 0x7C00u | nanFrac;
        }
        return static_cast<uint16_t>(sign << 15) | 0x7C00u;  // overflow -> inf
    }
    if (exp > 0) {
        // normal: exp in [1, 30]
        uint32_t frac10 = frac >> 13;
        uint32_t rem = frac & 0x1FFFu;
        uint32_t halfWay = 0x1000u;  // 1<<12
        if (rem > halfWay || (rem == halfWay && (frac10 & 1))) {
            frac10 += 1;
            if (frac10 > 0x3FFu) {
                frac10 = 0;
                exp += 1;
                if (exp >= 0x1F) return static_cast<uint16_t>(sign << 15) | 0x7C00u;
            }
        }
        return static_cast<uint16_t>(sign << 15) |
               (static_cast<uint16_t>(exp) << 10) |
               static_cast<uint16_t>(frac10);
    }
    if (exp > -11) {
        // subnormal: exp in [-10, 0]
        uint32_t frac23 = frac | 0x800000u;  // 加上隐含的 1，24-bit
        int shift = 14 - exp;
        uint32_t frac10 = frac23 >> shift;
        uint32_t rem = frac23 & ((1u << shift) - 1);
        uint32_t halfWay = 1u << (shift - 1);
        if (rem > halfWay || (rem == halfWay && (frac10 & 1))) {
            frac10 += 1;
            if (frac10 & 0x400u) {
                // 进位到最小 normal
                return static_cast<uint16_t>(sign << 15) | 0x0400u;
            }
        }
        return static_cast<uint16_t>(sign << 15) | static_cast<uint16_t>(frac10);
    }
    // underflow -> zero
    return static_cast<uint16_t>(sign << 15);
}

float Fp16BitsToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;  // zero
        } else {
            // subnormal: normalize
            exp = 1;
            while (!(frac & 0x400u)) { frac <<= 1; exp--; }
            frac &= 0x3FFu;
            f = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        // inf/nan
        f = (sign << 31) | (0xFFu << 23) | (frac << 13);
    } else {
        // normal
        f = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}
```

bf16 走"float 高 16 位截断 + round-to-nearest-even"的同构位级实现（`FloatToBf16Bits`/`Bf16BitsToFloat`，同文件）；比对端对浮点 dtype 先 `memcmp` 位级相等判 PASS，不等时才用位级转换解释相对误差（MERE/MARE）打印诊断。

## 设计决策

- 字节级而非值级 golden：gather 是纯搬运，字节相等即语义正确；一条路径覆盖全部 dtype，golden 与 kernel 的位级行为天然一致（bitwise 比对不再受 host 舍入差异干扰）。
- 位级 fp16 转换而非 `__fp16`：x86-64 g++ 13.3 实测不支持（编译错误），且位级实现显式控制 round-to-nearest-even 与 subnormal/inf/nan 分支，跨编译器行为确定。
- 浮点比对保留阈值路径（MERE/MARE）作诊断，但判定先走 bitwise：纯搬运算子的正确标准就是位一致。
- 交叉验证：另保留一套按值计算的模板版 golden（`ComputeGatherGolden<T,IndexT>`），Mock 自测里字节级 vs 模板版互相对账，防单一路径自证。

## 系统信息

- 工具链：宿主机 g++ (Ubuntu 13.3.0)；CMake Mock/Real 双模式（`-DUSE_MOCK_ACLNN`）。
- 目标平台：Ascend950PR（arch35/DAV_3510），CANN 9.1.0。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（golden + 比对 + 用例，终版一次完整写入）、`tests/st/run.sh`（--mock/--real）。
- dtype 矩阵：x 与 index 的 dtype 组合覆盖 fp16/fp32/bf16/int8/int32/int64（含跨宽度组合）。

## 验证信号

实现与验证来自该 gather 算子 ST 工程开发会话（迭代二 B）：

- `__fp16` 不可用证据：g++ 13.3.0 编译 `error: '__fp16' was not declared in this scope; did you mean '__bf16'?`（来源平台：x86_64 宿主机；证据等级：tool_observed）。
- Golden 自测（含 fp16 inf/nan、bf16 ±0、int8 边界、int64、空 Tensor、字节级 vs 模板版交叉验证）：通过=12/12（来源平台：x86_64 宿主机 Mock；证据等级：tool_observed）。
- Mock 全量用例 25/25 通过（含 6 穿刺：fp16/bf16/int8/int64/负索引/空 Tensor）（来源平台：x86_64 宿主机 Mock；证据等级：tool_observed）。
- 上板一致性：同套 golden 在 Real 模式下与 NPU 输出比对 25/25 全部 bitwise=YES（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 字节级 golden 只对"纯搬运/纯重排"算子成立；任何含计算（cast/加乘/归约）的算子必须按语义计算 golden。
- 位级转换验证于 IEEE 754 binary16/bfloat16 语义；硬件若采用非默认舍入模式（如 RZ）需调整转换的舍入分支——未验证。
- 空 Tensor（indexNumel=0）用例需比对函数先判 `numel==0` 再解指针（空 vector 的 `data()` 可能为 nullptr）。
- 其他平台未验证，推断可迁移（纯 host 侧方法）。

**未决项**：

- bf16 舍入分支（round-to-nearest-even 截断）未做穷举位样张测试（仅常规值 + ±0 + inf/nan 自测），如需硬化可补 exhaustive fp32→bf16 对照。
## 坏实践

- 在 x86 golden 里依赖 `__fp16`/`_Float16`（编译不过或跨编译器行为不一；_Float16 路线见相邻卡，两条路线择一）。
- 纯搬运算子用"解释→计算→重编码"的值级 golden，引入 host 舍入差异后 bitwise 永远对不齐。
- 只测单一 dtype 就上 Real。

## 误诊

- 无。
