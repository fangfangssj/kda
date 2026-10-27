---
type: "Runbook"
title: "C++ ST fp16/bf16 全 dtype 数据通路适配配方"
description: "按 float 写死的 C++ aclnn ST 测试工程补 fp16/bf16 用例须全链路 dtype 化五步改造：golden 头加位级转换与 pack/unpack、输入先 fp32→dtype→fp32 roundtrip 再算 golden、CreateAclTensor 改 aclDType/testDType 双参数按 dtype 字节流打包、结果按 dtype 解包回 fp32 比对；CANN 9.2.0 的 bf16 枚举名是 ACL_BF16。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T00:51:56Z'
platforms: ["950"]
---

# C++ ST fp16/bf16 全 dtype 数据通路适配配方

## 现象

一个 LSTM 类自定义算子（aclnn 两段式接口）的 C++ ST 测试工程最初只覆盖 fp32（Mock 自洽 + Real NPU 比对双模式全过）。补 fp16/bf16 用例时，测试工程的数据通路全部按 float 写死：

- `CreateAclTensor` / `CreateOutputAclTensor` 用 `sizeof(float)` 计算 device 内存并以 `ACL_FLOAT` 创建 `aclTensor`；
- golden 是纯 fp32 CPU 实现，直接吃原始 fp32 随机数据；
- 结果回读按 `std::vector<float>` + `sizeof(float)` 解包。

而 kernel 对 fp16/bf16 输入是先 Cast 到 fp32 计算再 Cast 回原 dtype 输出——直接加用例会同时踩三个坑：device 内存大小错 2 倍、`aclTensor` 类型与真实数据不符、golden 比对的输入是 kernel 从未见过的未量化 fp32 值（引入与 kernel 行为无关的量化假误差，近零值还会被相对误差分母放大，参见 [ST golden 须用 dtype 转换后数据卡](golden_input_dtype_aligns_npu_input.md)）。

## 适用场景

- 工程形态：自定义算子 aclnn 接口的 C++ ST 测试工程（Real 模式 NPU 执行 + CPU golden 比对，Mock 模式 golden 自洽双轨），从 fp32 单 dtype 扩展 fp16/bf16。
- API 路径：`aclrtMalloc`/`aclrtMemcpy`/`aclCreateTensor`；`aclDataType`（`ACL_FLOAT`/`ACL_FLOAT16`/`ACL_BF16`，定义在 `acl/acl_base_rt.h`）；golden 头文件（独立于设备侧类型系统的纯 C++ 参考实现）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510 / arch35）+ CANN 9.2.0（与 frontmatter 一致）。
- 适用粒度：platform_wide——该 CANN 版本下所有需要给 C++ aclnn ST 工程加 fp16/bf16 用例的算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：PyTorch golden 路线（torch adapter，机制不同）；fp8 等更窄 dtype 未覆盖；其他 CANN 版本的枚举名未验证（未验证，推断可迁移，使用前查 acl_base_rt.h）。

## 解决配方

五步改造（golden 层 → 数据准备层 → tensor 创建层 → 结果回读层 → dtype 映射层）：

1. **golden.h 增加位级转换层**：`DType` 枚举（FP32/FP16/BF16）+ IEEE 754 位操作实现 `FloatToFp16`/`Fp16ToFloat`/`FloatToBf16`（RNE）/`Bf16ToFloat` + `CastRoundtrip`/`CastRoundtripVec` + `DTypeSize` + `PackFloatDataAsDType`/`UnpackDataToFloat`。
2. **输入 roundtrip**：fp16/bf16 用例在算 golden 前对全部输入做 `fp32→dtype→fp32` roundtrip，保证 golden 与 kernel 看到相同的量化输入。
3. **`CreateAclTensor` 签名改双参数 `(aclDataType aclDType, DType testDType)`**：按 `DTypeSize(testDType)` 分配 device 内存，`PackFloatDataAsDType` 打包字节流后 `aclrtMemcpy`，`aclCreateTensor` 用 `aclDType`；所有调用点（含未被 `RunTestCase` 调用的辅助函数）统一替换。
4. **结果按 dtype 回读**：按 `elemSize` 读字节流 → `UnpackDataToFloat` 解包为 fp32 → 与 golden 比对。
5. **dtype 映射**：`ACL_FLOAT`/`ACL_FLOAT16`/`ACL_BF16`——CANN 9.2.0 的 aclDataType 里 bfloat16 名为 `ACL_BF16`（=27），不存在 `ACL_BFLOAT16`。

关键代码（golden.h dtype 转换层，verbatim）：

```cpp
enum class DType {
    FP32,
    FP16,
    BF16
};

// float32 -> fp16 (IEEE 754 half: 1 sign + 5 exp + 10 frac, bias=15)
inline uint16_t FloatToFp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (u >> 13) & 0x3FF;

    if (exp <= 0) {
        // 下溢或非规格化
        if (exp < -10) {
            // 太小，归零（保留符号）
            return static_cast<uint16_t>(sign);
        }
        // 非规格化数: 原始 frac 补上隐含的 1，右移
        uint32_t mantissa = (u & 0x7FFFFF) | 0x800000;
        int shift = 14 - exp;  // exp<=0 时 shift>=14
        frac = mantissa >> shift;
        return static_cast<uint16_t>(sign | frac);
    } else if (exp >= 31) {
        // 上溢或 NaN/Inf，返回 Inf（保留符号）
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | frac);
}

// fp16 -> float32
inline float Fp16ToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t u;

    if (exp == 0) {
        if (frac == 0) {
            // ±0
            u = sign;
        } else {
            // 非规格化: 规范化
            int e = 1;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                e--;
            }
            frac &= 0x3FF;
            u = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        // Inf / NaN
        u = sign | 0x7F800000 | (frac << 13);
    } else {
        // 规格化
        u = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// float32 -> bfloat16 (1 sign + 8 exp + 7 frac, 本质是 fp32 截断低16位)
// 采用 round-to-nearest-even
inline uint16_t FloatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even: 加上 0x7FFF + 最低位（用于偶数舍入）
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding = 0x7FFF + lsb;
    return static_cast<uint16_t>((u + rounding) >> 16);
}

// bfloat16 -> float32
inline float Bf16ToFloat(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// 单值 roundtrip: fp32 -> dtype -> fp32（模拟该 dtype 的精度损失）
inline float CastRoundtrip(float val, DType dt) {
    switch (dt) {
        case DType::FP32:
            return val;
        case DType::FP16:
            return Fp16ToFloat(FloatToFp16(val));
        case DType::BF16:
            return Bf16ToFloat(FloatToBf16(val));
    }
    return val;
}

// 向量 roundtrip: 原地转换
inline void CastRoundtripVec(std::vector<float>& data, DType dt) {
    if (dt == DType::FP32) return;
    for (auto& v : data) {
        v = CastRoundtrip(v, dt);
    }
}

// 返回 dtype 的元素字节大小
inline size_t DTypeSize(DType dt) {
    switch (dt) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
    }
    return 4;
}
```

打包/解包层（verbatim）：

```cpp
// 将 float32 数据打包为指定 dtype 的字节流（用于拷贝到 device 内存）
inline std::vector<uint8_t> PackFloatDataAsDType(const std::vector<float>& data, DType dt) {
    std::vector<uint8_t> bytes;
    bytes.reserve(data.size() * DTypeSize(dt));
    if (dt == DType::FP32) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
        bytes.assign(p, p + data.size() * sizeof(float));
    } else {
        for (float v : data) {
            uint16_t h = (dt == DType::FP16) ? FloatToFp16(v) : FloatToBf16(v);
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&h);
            bytes.push_back(p[0]);
            bytes.push_back(p[1]);
        }
    }
    return bytes;
}

// 将指定 dtype 的字节流解包为 float32 数据（用于从 device 读取结果后比对）
inline std::vector<float> UnpackDataToFloat(const void* data, size_t numel, DType dt) {
    std::vector<float> result(numel);
    if (dt == DType::FP32) {
        const float* p = static_cast<const float*>(data);
        for (size_t i = 0; i < numel; i++) result[i] = p[i];
    } else {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < numel; i++) {
            uint16_t h;
            std::memcpy(&h, bytes + i * 2, 2);
            result[i] = (dt == DType::FP16) ? Fp16ToFloat(h) : Bf16ToFloat(h);
        }
    }
    return result;
}
```

输入 roundtrip（`PrepareTestData` 内，对全部输入张量统一应用）：

```cpp
    // fp16/bf16 用例: 对输入数据做 roundtrip（fp32→dtype→fp32），
    // 模拟输入为 fp16/bf16 的精度损失，对齐 kernel 实际读到的量化输入。
    if (tc.dtype != lstm_golden::DType::FP32) {
        lstm_golden::CastRoundtripVec(td.x, tc.dtype);
        for (auto& w : td.weight_ih) lstm_golden::CastRoundtripVec(w, tc.dtype);
        for (auto& w : td.weight_hh) lstm_golden::CastRoundtripVec(w, tc.dtype);
        // bias/h0/c0/proj 等其余输入同法逐个 roundtrip
    }
```

tensor 创建层修复后（双参数，verbatim；修复前为单 `aclDataType` 参数 + `sizeof(float)` 硬编码）：

```cpp
// 创建单个 aclTensor（含 device 内存分配和数据拷贝）
// hostData 为 float32 数据，根据 testDType 转换为对应字节流后拷贝到 device
int CreateAclTensor(const std::vector<float>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType aclDType,
                    lstm_golden::DType testDType,
                    aclTensor** tensor) {
    int64_t numel = lstm_golden::ShapeNumel(shape);
    size_t elemSize = lstm_golden::DTypeSize(testDType);
    size_t size = static_cast<size_t>(numel) * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    auto bytes = lstm_golden::PackFloatDataAsDType(hostData, testDType);
    ret = aclrtMemcpy(*deviceAddr, size, bytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), static_cast<int>(shape.size()), aclDType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), static_cast<int>(shape.size()), *deviceAddr);
    return ACL_SUCCESS;
}
```

结果回读层修复后（按 dtype 字节流读回再解包，verbatim）：

```cpp
    // 拷贝结果到 host（按 dtype 读取字节流，再解包为 float32 比对）
    {
        size_t elemSize = lstm_golden::DTypeSize(testDType);
        std::vector<uint8_t> yBytes(static_cast<size_t>(td.shapes.y_numel) * elemSize);
        aclrtMemcpy(yBytes.data(), yBytes.size(), yDev, yBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        auto yNpu = lstm_golden::UnpackDataToFloat(yBytes.data(), static_cast<size_t>(td.shapes.y_numel), testDType);
        // hn/cn 等其余输出同法
    }
```

dtype 映射层（修复后；修复前 bf16 一行误写 `ACL_BFLOAT16`，主动核查头文件后在编译前发现）：

```cpp
    aclDataType aclDType;
    lstm_golden::DType testDType = tc.dtype;
    switch (tc.dtype) {
        case lstm_golden::DType::FP32: aclDType = ACL_FLOAT; break;
        case lstm_golden::DType::FP16: aclDType = ACL_FLOAT16; break;
        case lstm_golden::DType::BF16: aclDType = ACL_BF16; break;
    }
```

## 设计决策

- **为什么输入必须 roundtrip**：kernel 对 fp16/bf16 输入先 Cast fp32 计算；若 golden 直接吃原始 fp32 随机数据，比对基准是 kernel 从未见过的输入，量化假误差会淹没真问题（实测 roundtrip 后 fp16 y MERE 稳定在 1.7e-4 量级、bf16 在 1.4e-3 量级，均为该语义下的合理水平）。
- **golden 自实现位级转换而非复用设备侧头**：golden 须独立于设备侧类型系统、纯 C++ 可移植（Mock 模式无 NPU 也要能跑）；bf16 用 fp32 高 16 位截断 + round-to-nearest-even，与硬件舍入语义一致。
- **`aclDType` 与 `testDType` 双参数**：前者描述 aclTensor 类型（ACL 枚举），后者描述数据通路（字节大小/打包方式，golden 的 DType 枚举）——两者一一对应但属两套类型系统，合成一个参数会在 Mock 路径引入不必要的 ACL 头依赖。
- **大 shape 保护性跳过**：CPU golden 朴素循环在大 shape 上过慢，`EstimateFlops(tc) > 1e10` 时跳过并显式打印 `[SKIP] 计算量过大 ... 跳过 golden+NPU 计算`——跳过时 NPU 计算一并跳过、计入"跳过"而非"通过"，不伪造结果。
- **Mock 与 Real 行为一致**：Mock（actual=golden 自洽）同样做输入 roundtrip，两种模式验证同一套语义。

## 预防

- 给 ST 工程加 dtype 前先盘点数据通路上的每个 `sizeof(float)` / `ACL_FLOAT` / `std::vector<float>`——三处（分配/创建/解包）必须同步 dtype 化。
- 改 `CreateAclTensor` 签名时全量排查调用点（含当前未使用的辅助函数）：遗漏后 Real 模式编译报 `error: cannot convert 'aclTensor**' to 'lstm_golden::DType'`。
- dtype 映射写前实读 `acl_base_rt.h` 核对枚举名（bf16 是 `ACL_BF16`）。

## 验证信号

来源说明：本卡信号来自该算子迭代三 ST 全 dtype 适配后的双模式实测（Ascend950PR，CANN 9.2.0）。

- Mock 模式（CPU golden 自洽，非上板）：L0+L1 44 用例 43 通过 0 失败 1 跳过（大 shape FLOP 阈值保护性跳过）（来源平台：950 开发环境（Mock/CPU）；证据等级：tool_observed）。
- Real NPU 全 dtype 43/43 通过（1 跳过）：fp32 y max MERE 5.81e-05 / max MARE 1.23e-02；fp16 y MERE 1.70e-04~1.76e-04、max MARE 1.22e-02；bf16 y MERE 1.39e-03~1.64e-03（阈值 mere<5e-2、mare<5e-1）（来源平台：950；证据等级：tool_observed）。
- 测试工程师独立清理 build-real 重跑 ST Real 仍 43/43 + 1 跳过，L0 回归精度与迭代一逐位一致（来源平台：950；证据等级：tool_observed）。
- 枚举事实源实读：`acl_base_rt.h:140-169` 中 `ACL_FLOAT=0, ACL_FLOAT16=1, …, ACL_BF16=27`（无 ACL_BFLOAT16）（来源平台：CANN 9.2.0 include 树；证据等级：tool_observed）。

## 失效边界

- 位级转换按 IEEE 754 半精度（含非规格化、Inf/NaN 分支）与 bf16 截断+RNE 实现；仅覆盖 fp32/fp16/bf16 三 dtype，fp8 等更窄类型未实现未验证。
- 输入 roundtrip 只对齐"输入即已量化"的语义；若被测算子对 fp16 输入有特殊预处理（如先反量化），roundtrip 位置需相应调整。
- 大 shape 用例（FLOPs 超 1e10 阈值）被保护性跳过，其 NPU 精度在本方案内无法验证（未验证项）。
- `ACL_BF16` 枚举名与取值（=27）是 CANN 9.2.0 事实，其他版本未验证（未验证，推断可迁移，使用前应查 acl_base_rt.h）。

**未决项**：

- 大 shape 用例（约 1.0e10 FLOPs）始终被跳过，NPU 精度未验证——需要向量化 golden 或上板专用比对方案。
## 坏实践

- dtype 映射想当然写 `ACL_BFLOAT16`——CANN 9.2.0 枚举名是 `ACL_BF16`。
- fp16/bf16 用例不做输入 roundtrip 直接比对——量化假误差会把真问题淹没。
- 改 `CreateAclTensor` 签名只改主调用点，不排查辅助函数内的间接调用。
- 大 shape 用例因 CPU golden 慢就标记"通过"或删掉——正确做法是显式 `[SKIP]` 并计入跳过统计。

## 误诊

- 无（本改造过程中无错误归因；`ACL_BFLOAT16` 笔误在编译前由头文件核查发现）。
