---
type: "Runbook"
title: "无 NPU 的 ST Mock 模式：CPU golden 前置验证"
description: "用 USE_MOCK_ACLNN 双模式 CMake + CSV 用例驱动 + 位级 fp16/bf16 转换的 CPU golden（移植 golden.py）+ 属性校验（全排列/re-gather/列和）+ MERE/MARE 阈值比对，在 x86 主机上无 NPU 完成 ST 前置验证；x86_64 gcc13 无 __fp16，golden 须上位级转换。复现增量：Real 模式上板闭环（NPU 20/20）+ NaN/Inf 位置匹配优先于数值的比对规则 + fp32 2^-13 阈值档 + golden 手算期望值锚定自测。复现增量：四条独立轨迹的自测双保险（手算+golden.py 交叉校验，实战抓出期望漏算 padding tap）+ 自测容差与用例阈值分离（fp16 存储量化下限）。 增量 A（MlaProlog，bf16）：Mock MERE=0 是 golden 与自身比较的恒零值、不构成正确性证据——C++ golden 移植必须先过组件级自测 + 与 Python golden 的位级交叉验证（期望值同过 bf16 量化域、uint16 位型终审）。增量 B（GRU，双输出 RNN）：golden 三关门控（手算/零输入/双向）FATAL 前置，实战抓出「逐层循环后 yInternal 未回写」致 y 全 0 而 hn 对的装配 bug；双精度内核 + (S,B,D) 统一内部布局。"
tags: [task.testing, topic.data_movement.layout_transform, topic.mixture_of_experts]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# 无 NPU 的 ST Mock 模式：CPU golden 前置验证

## 现象

MoE 重排自定义算子（token 按专家顺序 gather 重排）进入迭代一，需要一套 C++ ST 测试工程。当时算子 kernel 刚编译出二进制、尚未安装到 NPU 环境，但验收要求测试工程先行可跑。直接写"真机版"测试会卡在环境上；于是搭了一套 **Mock 模式**：`run.sh --mock` 用纯 CPU golden（与官方 golden.py 逐语义对齐）先验证测试框架自身与 golden 正确性，`run.sh --real` 再接真实 aclnn。开发中踩到一个 x86 主机特有报错：

```
.tmp_test/test_fp16.cpp:4:5: error: '__fp16' was not declared in this scope; did you mean '__bf16'?
```

（gcc 13.3 x86_64 默认无 `__fp16` 扩展，CPU golden 无法用它存 fp16 参考数据。）最终 Mock 模式 L0 全部 12 用例 PASS、golden 与 python 参考实现 bit 级一致、`-Wall -Wextra` 零警告。

## 适用场景

- 工程形态：自定义算子 ST 测试工程开发（aclnn 两段式接口），无 NPU/算子包未就绪时的前置验证，以及 golden 框架自测。
- API 路径：aclnn `GetWorkspaceSize + Execute` 两段式调用；CMake `option(USE_MOCK)` 双模式构建；CPU 侧位级 fp16/bf16 转换。
- 验证过的平台与 CANN 版本：目标算子 Ascend950PR（ascend950/arch35）、CANN 9.1.0；Mock 模式本身在 x86_64 主机（g++ 13.3）运行，不依赖 NPU。
- 适用粒度：operator_family——golden 为"字节级搬移/纯数据重排"类算子（gather/scatter/layout transform）可直接复用；含数值计算的算子需按其 golden 公式改写计算部分。
- 路线优先级：recommended_fastpath（ST 工程的先行轨道；Real 模式仍是最终验收）。
- 明确不适用 / 未验证场景：超大 shape（L1 的 A=524288×H=2048×fp16 ≈ 2GB/buffer，Mock 模式受主机内存与 120s 命令超时限制，实测超时；大 shape 走 Real 模式的 NPU 内存）；Mock 模式不验证 kernel 本身，只验证 golden/框架/属性约束。

## 解决配方

三件套结构：

1. **双模式 CMake**（`tests/st/CMakeLists.txt` 核心段）：

```cmake
option(USE_MOCK "Use mock aclnn for testing without real operator (CPU golden only)" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK aclnn + CPU golden - full golden validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden comparison")
endif()

# 算子 vendor 包名（与安装目录名一致）
set(OP_VENDOR_NAME bench_no_harness_moe_re_routing_custom)

if(NOT USE_MOCK)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH environment variable is not set!")
    endif()
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
    set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)
endif()
```

测试主程序里用 `#ifndef USE_MOCK_ACLNN` 把 aclnn 调用与 acl 类型全部隔离；Mock 分支只跑 CPU golden + 属性校验。`run.sh --mock/--real` 分别落独立构建目录（build-mock / build-real）避免缓存互踩。

2. **位级 fp16/bf16 转换**（`common/dtype_util.h`，解决 `__fp16` 不可用；round-to-nearest-even）——见《完整代码》第一节。

3. **CPU golden 移植 + 属性校验**（`common/golden.h`）——见《完整代码》第二节。除逐值比对外，golden 自测增加三个**属性校验**：`permute_token_idx` 是 [0,A) 全排列（`IsPermutation`）、用 gather 索引再 gather 回原序一致性、`expert_token_num` 列和 == A；比对用社区标准 MERE/MARE 阈值（fp16 2^-10、bf16 2^-7，int8 精确匹配）。

## 设计决策

- 为什么手写位级转换而不是 `_Float16`/intrinsics：gcc 13.3 x86_64 下 `__fp16` 默认不可用（实测报错），`__bf16` 可用但 fp16 没有；位级实现可移植且舍入模式明确（round-to-nearest-even，与 NPU 侧语义一致），先用 `f=1.5/-2.25/次正规数` 等探针验证 round-trip 正确再入库。
- 为什么 Mock 模式仍做属性校验而不只 trivial 比对：golden 自身若有 bug，Mock 全 PASS 也无意义；全排列/re-gather/列和三个性质从数学上约束 golden 实现，配合 4 个手算用例 + 与官方 golden.py 的随机交叉验证（100 cases）构成"golden 的 golden"。
- 为什么按 CSV 用例驱动（L0/L1/L2_test_cases.csv）：与测试设计文档的用例分级一致，Mock/Real 两种模式跑同一份用例文件。
- 放弃的方案：在 Mock 里链接真实 aclnn 库做桩——环境依赖重，与"无 NPU 前置验证"目标矛盾。
- 当前方案路线档位：recommended_fastpath（Real 模式为最终验收，Mock 是其低成本前置）。

## 系统信息

- 主机：x86_64，g++ 13.3.0，cmake 3.20；CANN 9.1.0（Real 模式用，`ASCEND_HOME_PATH` 指向 CANN 安装根）；目标算子 Ascend950PR/arch35，vendor 包名 `bench_no_harness_moe_re_routing_custom`。
- 目录布局（`operators/<op>/tests/st/`）：

| 文件 | 行数 | 职责 |
|------|------|------|
| `test_aclnn_<op>.cpp` | 767 | 主程序：数据生成、Mock/Real 执行器、golden 自测、CSV 加载、报告 |
| `common/dtype_util.h` | 207 | fp16/bf16 位级转换（不依赖 `__fp16`）、dtype 枚举映射、阈值 |
| `common/csv_parser.h` | 113 | L0/L1/L2 CSV 用例解析 |
| `common/golden.h` | 152 | CPU golden（SrcOffset/DstOffset 双重求和 + gather 索引 + 属性校验） |
| `common/compare.h` | 188 | MERE/MARE 浮点比对 + int32/int64/int8 精确匹配 |
| `CMakeLists.txt` | 100 | Mock/Real 双模式构建 |
| `run.sh` | 227 | `--mock`/`--real`/`--case` 参数，独立构建目录 |

## 完整代码

`common/dtype_util.h`（verbatim，终版写入；注：Real 分支的 bf16 枚举按 `acl_base_rt.h` 实名写 `ACL_BF16`——`ACL_BFLOAT16` 拼写不存在，见同批 bf16 枚举实名卡）：

```cpp
// ============================================================================
// dtype_util.h - 数据类型工具：fp16/bf16 转换、dtype 枚举映射
// ============================================================================

#ifndef COMMON_DTYPE_UTIL_H
#define COMMON_DTYPE_UTIL_H

#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

// ----------------------------------------------------------------------------
// FP16 (IEEE 754 half) 转换 - 便携实现，不依赖 __fp16 编译器扩展
// ----------------------------------------------------------------------------
inline float Fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // +-0
        } else {
            // subnormal: 归一化
            int e = 0;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3ffu;
            f = (sign << 31) | ((static_cast<uint32_t>(127 - 15 - e)) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);  // inf/nan
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

inline uint16_t FloatToFp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 0x1u;
    int32_t  exp  = (u >> 23) & 0xffu;
    uint32_t mant = u & 0x7fffffu;
    if (exp == 0xff) {
        uint16_t hm = static_cast<uint16_t>(mant >> 13);
        if (mant) hm |= 0x200u;  // nan payload 非零
        return static_cast<uint16_t>((sign << 15) | (0x1fu << 10) | hm);
    }
    int32_t newExp = exp - 127 + 15;
    if (newExp >= 0x1f) {
        return static_cast<uint16_t>((sign << 15) | (0x1fu << 10));  // overflow -> inf
    }
    if (newExp <= 0) {
        if (newExp < -10) {
            return static_cast<uint16_t>(sign << 15);  // underflow -> 0
        }
        uint32_t m = mant | 0x800000u;
        int shift = 14 - newExp;
        uint32_t val = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (val & 1u))) val++;  // round to nearest even
        return static_cast<uint16_t>((sign << 15) | val);
    }
    uint32_t val = (static_cast<uint32_t>(newExp) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    uint32_t half = 0x1000u;
    if (rem > half || (rem == half && (val & 1u))) val++;
    return static_cast<uint16_t>((sign << 15) | val);
}

// ----------------------------------------------------------------------------
// BF16 转换 - BF16 为 FP32 的高 16 位
// ----------------------------------------------------------------------------
inline float Bf16ToFloat(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

inline uint16_t FloatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;  // round to nearest even
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// ----------------------------------------------------------------------------
// dtype 枚举
// ----------------------------------------------------------------------------
enum class TokenDtype { FP16 = 0, BF16 = 1, INT8 = 2, UNKNOWN = -1 };
enum class ExpertDtype { INT32 = 0, INT64 = 1, UNKNOWN = -1 };

inline TokenDtype ParseTokenDtype(const std::string& s) {
    if (s == "float16" || s == "fp16") return TokenDtype::FP16;
    if (s == "bfloat16" || s == "bf16") return TokenDtype::BF16;
    if (s == "int8") return TokenDtype::INT8;
    return TokenDtype::UNKNOWN;
}

inline ExpertDtype ParseExpertDtype(const std::string& s) {
    if (s == "int32") return ExpertDtype::INT32;
    if (s == "int64") return ExpertDtype::INT64;
    return ExpertDtype::UNKNOWN;
}

inline const char* TokenDtypeName(TokenDtype d) {
    switch (d) {
        case TokenDtype::FP16:  return "float16";
        case TokenDtype::BF16:  return "bfloat16";
        case TokenDtype::INT8:  return "int8";
        default: return "unknown";
    }
}

inline const char* ExpertDtypeName(ExpertDtype d) {
    switch (d) {
        case ExpertDtype::INT32: return "int32";
        case ExpertDtype::INT64: return "int64";
        default: return "unknown";
    }
}

// 元素字节数
inline size_t TokenElemSize(TokenDtype d) {
    switch (d) {
        case TokenDtype::FP16: return 2;
        case TokenDtype::BF16: return 2;
        case TokenDtype::INT8: return 1;
        default: return 0;
    }
}

inline size_t ExpertElemSize(ExpertDtype d) {
    switch (d) {
        case ExpertDtype::INT32: return 4;
        case ExpertDtype::INT64: return 8;
        default: return 0;
    }
}

// 将一个元素（按 token dtype 存储的字节）转为 double（用于精度比对）
inline double TokenElemToDouble(const uint8_t* p, TokenDtype d) {
    switch (d) {
        case TokenDtype::FP16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return static_cast<double>(Fp16ToFloat(v));
        }
        case TokenDtype::BF16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return static_cast<double>(Bf16ToFloat(v));
        }
        case TokenDtype::INT8:
            return static_cast<double>(static_cast<int8_t>(*p));
        default:
            return 0.0;
    }
}

// 精度阈值（社区标准 MERE/MARE Threshold）
inline double TokenThreshold(TokenDtype d) {
    switch (d) {
        case TokenDtype::FP16: return 9.765625e-4;   // 2^-10
        case TokenDtype::BF16: return 7.8125e-3;     // 2^-7
        case TokenDtype::INT8: return 0.0;           // 精确匹配
        default: return 0.0;
    }
}

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
// dtype 枚举 -> aclDataType
inline aclDataType TokenAclDtype(TokenDtype d) {
    switch (d) {
        case TokenDtype::FP16: return ACL_FLOAT16;
        case TokenDtype::BF16: return ACL_BF16;  // 实名 ACL_BF16(=27)；ACL_BFLOAT16 不存在
        case TokenDtype::INT8:  return ACL_INT8;
        default: return ACL_DT_UNDEFINED;
    }
}
inline aclDataType ExpertAclDtype(ExpertDtype d) {
    switch (d) {
        case ExpertDtype::INT32: return ACL_INT32;
        case ExpertDtype::INT64: return ACL_INT64;
        default: return ACL_DT_UNDEFINED;
    }
}
#endif

#endif  // COMMON_DTYPE_UTIL_H
```

`common/golden.h` 核心（verbatim，终版写入；gather 索引算法与 golden.py 逐语义对齐——src 顺序 (rank, expert) 行优先，dst 顺序 (expert, rank)）：

```cpp
inline void ComputePermuteTokenIdx(const int64_t* etnpr, int64_t N, int64_t E, int64_t A,
                                   int32_t* permuteTokenIdx) {
    const int64_t NE = N * E;

    // cnt_src: (rank, expert) 行优先，长度 NE
    std::vector<int64_t> cntSrc(NE);
    for (int64_t i = 0; i < NE; ++i) cntSrc[i] = etnpr[i];

    // cnt_dst: (expert, rank) 顺序，cnt_dst[e*N + r] = etnpr[r*E + e]
    std::vector<int64_t> cntDst(NE);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t e = 0; e < E; ++e) {
            cntDst[e * N + r] = etnpr[r * E + e];
        }
    }

    // src_start[b] = exclusive cumsum of cntSrc
    std::vector<int64_t> srcStart(NE, 0);
    for (int64_t i = 1; i < NE; ++i) {
        srcStart[i] = srcStart[i - 1] + cntSrc[i - 1];
    }

    // dst_start_dst[d] = exclusive cumsum of cntDst (in (expert,rank) order)
    std::vector<int64_t> dstStartDst(NE, 0);
    for (int64_t i = 1; i < NE; ++i) {
        dstStartDst[i] = dstStartDst[i - 1] + cntDst[i - 1];
    }

    // 映射回 (rank, expert) 顺序：dstStart[r*E + e] = dstStartDst[e*N + r]
    std::vector<int64_t> dstStart(NE, 0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t e = 0; e < E; ++e) {
            dstStart[r * E + e] = dstStartDst[e * N + r];
        }
    }

    // 计算每个 src 位置的 dst 位置，然后反转得到 gather 索引
    std::vector<int64_t> dstPos(A, 0);
    for (int64_t b = 0; b < NE; ++b) {
        for (int64_t k = 0; k < cntSrc[b]; ++k) {
            int64_t srcIdx = srcStart[b] + k;
            dstPos[srcIdx] = dstStart[b] + k;
        }
    }

    // permute_token_idx[dst_pos[i]] = i  -> gather 索引
    for (int64_t i = 0; i < A; ++i) {
        permuteTokenIdx[dstPos[i]] = static_cast<int32_t>(i);
    }
}

// Gather tokens：permute_tokens[d][:] = tokens[permute_token_idx[d]][:]
//   字节级拷贝，dtype 无关（gather 为精确数据搬移）
inline void GatherTokens(const uint8_t* tokens, int64_t A, int64_t H, size_t elemSize,
                         const int32_t* permuteTokenIdx, uint8_t* permuteTokens) {
    const size_t rowBytes = static_cast<size_t>(H) * elemSize;
    for (int64_t d = 0; d < A; ++d) {
        int64_t src = permuteTokenIdx[d];
        std::memcpy(permuteTokens + d * rowBytes,
                    tokens + src * rowBytes,
                    rowBytes);
    }
}

inline void GatherScales(const float* scales, int64_t A,
                         const int32_t* permuteTokenIdx, float* permuteScales) {
    for (int64_t d = 0; d < A; ++d) {
        permuteScales[d] = scales[permuteTokenIdx[d]];
    }
}

// expert_token_num (count 模式)：expert_token_num[e] = sum_r etnpr[r*E + e]
inline void ComputeExpertTokenNum(const int64_t* etnpr, int64_t N, int64_t E,
                                  ExpertDtype dtype, void* out) {
    if (dtype == ExpertDtype::INT64) {
        int64_t* o = static_cast<int64_t*>(out);
        for (int64_t e = 0; e < E; ++e) {
            int64_t s = 0;
            for (int64_t r = 0; r < N; ++r) s += etnpr[r * E + e];
            o[e] = s;
        }
    } else {
        int32_t* o = static_cast<int32_t*>(out);
        for (int64_t e = 0; e < E; ++e) {
            int64_t s = 0;
            for (int64_t r = 0; r < N; ++r) s += etnpr[r * E + e];
            o[e] = static_cast<int32_t>(s);
        }
    }
}

// 校验 permute_token_idx 是否为 [0, A) 的全排列
inline bool IsPermutation(const int32_t* idx, int64_t A) {
    std::vector<bool> seen(A, false);
    for (int64_t i = 0; i < A; ++i) {
        if (idx[i] < 0 || idx[i] >= A) return false;
        if (seen[idx[i]]) return false;
        seen[idx[i]] = true;
    }
    return true;
}
```

## 预防

- 在 x86 主机写涉及 fp16 的 host golden，先跑 `__fp16` 可用性探针（一行小程序编译），不可用直接上位级转换，不要赌编译器扩展。
- golden 上线前先过"golden 的 golden"：手算小用例 + 官方 golden.py 随机交叉验证 + 数学属性校验，三者齐了再开始比对真机输出。

## 验证信号

以下信号来自该 MoE 重排算子 ST 工程开发会话（目标 Ascend950PR/CANN 9.1.0；Mock 模式在 x86_64 主机运行）：

- `__fp16` 不可用复现：x86_64 gcc 13.3 编译 `__fp16 h = 1.5f;` 报 `error: '__fp16' was not declared in this scope`（来源平台：950；证据等级：tool_observed）。
- 位级转换探针：`f=0/1/1.5/-2.25...` round-trip 输出 `fp16=0x3c00 back=1`、`bf16=0x3f80 back=1` 等全部一致（来源平台：950；证据等级：tool_observed）。
- golden 交叉验证：与官方 golden.py 对同一手算用例输出逐位一致（permute_token_idx=[0,4,5,1,2,3,6,7] 等），随机交叉验证 100 cases `mismatches=0`，`PYTHON GOLDEN OK`（来源平台：950；证据等级：tool_observed）。
- Mock 全量：`bash run.sh --mock` golden 自测 ALL PASS（4 手算用例 + 属性校验）、L0 12/12 PASS（fp16/bf16/int8 × int32/int64 × 有/无 scales），测试报告落盘；`-Wall -Wextra` 编译零警告（来源平台：950；证据等级：tool_observed）。

## 失效边界

- Mock 模式不执行 kernel，不能发现 kernel/tiling 侧问题（本轨迹 kernel 验证在 Real 模式，后续阶段）。
- 主机内存与命令超时限制：L1 大 shape（A=524288、H=2048、fp16 ≈ 2GB/buffer，多缓冲再 gather）在 Mock 模式 120s 超时未完成——大 shape 必须走 Real 模式（NPU 内存），Mock 仅适合 L0 量级。
- 位级 FloatToFp16 的溢出/下溢分支按 IEEE 语义处理（overflow→inf、underflow→0），但未对全部边界值穷举测试（未验证，推断可迁移）。

**未决项**：

- Real 模式（真实 aclnn + NPU）运行与比对在本卡证据范围外，尚未验证（Mock 只证明框架与 golden 正确）。（复现增量已补两条轨迹的 Real 上板闭环证据，见文末增量节。）
- L1 大 shape 的 Mock 性能优化（分块比对/流式 golden）未尝试。
## 坏实践

- 在 x86 主机 golden 里用 `__fp16`/`_Float16` 赌编译器扩展——gcc 13.3 默认没有（`__fp16`），报错浪费一轮。
- Mock golden 只做"输出 == 自己算的值"的 trivial 比对——golden 自身 bug 会让 Mock 假 PASS；属性校验（全排列/re-gather/列和）是必要的第二道闸。

## 误诊

- 无。

## 增量（Real 模式闭环与 NaN/Inf 比对规则）

两条独立轨迹（bench_no_harness 系列旋转位置编码类算子与 sigmoid 类逐元素算子，Ascend950PR / CANN 9.1.0）以同一 `USE_MOCK_ACLNN` 单文件双模式形态跑通全闭环，补齐主卡未覆盖的四点：

**1) Real 模式上板闭环**（主卡遗留问题 #1 的补证）：Mock 验证通过后零改动切 Real（NPU 输出 vs CPU golden，同一份用例与比对器），sigmoid 类算子联调阶段 Real 20/20 PASS、最高 MARE 1.80e-07——Mock/Real 共用用例与比对器的形态在 NPU 侧同样成立。

**2) NaN/Inf 位置匹配优先于数值**（MERE/MARE 比对器完整规则，主流程 verbatim）：

```cpp
    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);

        // NaN 处理: 两者都为 NaN 视为匹配
        if (std::isnan(g)) {
            if (std::isnan(a)) {
                nanMatchCount++;
                continue;
            }
            // golden 为 NaN 但 actual 不为 NaN → 最大误差
            mere += 1.0;
            mare = std::max(mare, 1.0);
            mismatchCount++;
            continue;
        }

        // Inf 处理: 同号 Inf 视为匹配
        if (std::isinf(g)) {
            if (std::isinf(a) && (g > 0) == (a > 0)) {
                infMatchCount++;
                continue;
            }
            mere += 1.0;
            mare = std::max(mare, 1.0);
            mismatchCount++;
            continue;
        }

        // 正常值: 计算相对误差
        double relErr = std::abs(a - g) / (std::abs(g) + 1e-7);
        mere += relErr;
        if (relErr > mare) mare = relErr;
        if (relErr > threshold) mismatchCount++;
    }

    mere /= static_cast<double>(size);
    double mareThreshold = 10.0 * threshold;

    bool pass = (mere < threshold) && (mare < mareThreshold);
```

**3) fp32 阈值档**：阈值常量按 dtype 分级，fp32 档与 fp16/bf16 同样必要（主卡只列 fp16/bf16/int8）：

```cpp
// 精度阈值 (cann-bench 标准)
constexpr double THRESHOLD_FP32 = 1.0 / 8192.0;   // 2^-13 ≈ 1.2207e-4
constexpr double THRESHOLD_FP16 = 1.0 / 1024.0;   // 2^-10 ≈ 9.7656e-4
constexpr double THRESHOLD_BF16 = 1.0 / 128.0;    // 2^-7  ≈ 7.8125e-3
```

**4) golden 自测须用手算期望值锚定**：Mock 模式 actual=golden（拷贝）、MERE/MARE=0 是预期，不能证明 golden 正确——必须用手算小输入锚定语义后再进用例流。旋转位置编码类轨迹的 golden 自测 7 组（half/interleaved × BSND/BNSD × cos 2D/3D × query/key 独立性，如 half 模式 B=S=N=1、D=4、query={1,2,3,4}、cos={0.5,0.3}、sin={0.1,0.2}，期望输出 {0.2, -0.2, 1.6, 1.6}；interleaved 同输入期望 {0.3, 1.1, 0.1, 1.8}）；sigmoid 类轨迹 18 组（基本值/饱和区/±Inf/NaN/±0/±FLT_MAX/批量范围）。与主卡的"属性校验（全排列/re-gather/列和）"互补：属性校验约束结构性 golden，手算锚定约束数值语义。

验证信号（复现增量，两条独立轨迹）：

- sigmoid 轨迹 Mock：编译链接通过（C++17）；CPU golden 自测 18/18 PASS（基本值/饱和/±Inf/NaN/±0/±FLT_MAX/批量范围）；L0 用例 20/20 PASS，MERE/MARE=0.00e+00 或远低于 1.22e-04（来源平台：950；证据等级：tool_observed）。
- sigmoid 轨迹 Real（联调阶段）：NPU 输出对比 CPU golden 20/20 PASS，最高 MARE 1.80e-07（来源平台：950；证据等级：tool_observed）。
- 旋转位置编码类轨迹 Mock：g++ 13.3 + cmake 3.20，`-Wall -Wextra` 零警告；golden 手算自测 7/7 通过；L0 用例 Mock 执行 15/15（fp32/fp16/bf16 × 2 layout × 2 rotaryMode × 2 cos_dim），最大 12.8M 元素用例跑通（来源平台：Mock(CPU) x86 主机；证据等级：tool_observed）。

失效边界（复现增量）：

- Mock 可跑的规模上限与算法复杂度相关：旋转位置编码类逐元素/旋转路径 12.8M 元素在 Mock 内跑通，而主卡的 MoE gather 类 L1（单缓冲 ≈2GB）超时——判据用"宿主逐元素成本 × 元素总量 + 单缓冲字节数"估算，不是元素数单指标。
- Real 信号来自联调阶段的任务级结果（20/20），逐用例 MERE/MARE 分布未在增量内展开。
- Mock(CPU) 信号不锚定 NPU 平台；frontmatter `platforms: [950]` 由 sigmoid 轨迹的 Real 信号支撑。

## 增量（四条独立轨迹：自测双保险与 golden 修正实战）

四条独立算子轨迹（ROIAlign 类双线性插值、TopK 类排序、GQA 注意力、Depthwise 卷积；均为 x86_64 host Mock 实测，目标 Ascend950PR / CANN 9.1.0）以同一 `USE_MOCK_ACLNN` 单文件双模式形态先行交付 ST 工程，补齐主卡与复现增量的三点：

**1) golden 自测双保险（手算用例 + golden.py 交叉校验）**：手算小例锚定算法语义（对齐偏移、采样点数、边界 clamp、padding 边界 tap），golden.py 交叉校验用例（固定输入+期望输出数组，由独立 golden.py 离线生成后嵌入）锚定"移植没写错"——只用其一都会漏一类错误。实战反例：Depthwise 轨迹 golden 自测首跑即抓出手算期望漏算 padding 相邻 tap 贡献（期望 4.5 实际 5.5，golden 算法本身正确，修正期望而非修 golden）；TopK 轨迹自测含"比对函数扰动验证"（完全相等 PASS/微小扰动 PASS/大扰动 FAIL/NaN=NaN PASS），证明比对器对"该过的过、该挂的挂"有区分度。

**2) Real 模式前置约束**：Real 分支的 aclnn 头按部署环境实际存在的 include 写（如 `aclnn/aclnn_base.h` 而非 `aclnn/aclnn.h`，以 CANN 安装目录为准）；TopK 类算子 idx 不做逐元素比对（tie 顺序非确定），改用 gather 校验 `x.gather(dim, idx) == y`，自测与 Real 比对同用。

**3) 自测容差与用例阈值分离**：golden 自测的期望值是高精度手算值而 golden 按 fp16 存储时，自测容差必须 ≥ 存储量化（fp16 [0.5,1) 区间 ulp/2≈2.44e-4，本工程取 2e-3），否则必误报（GQA 轨迹实测 diff=1.02e-04 假 FAIL 并 FATAL 终止）；判别法：实际值 ×2048 贴近整数即最近可表示值，是量化误差不是逻辑错误。该自测容差与 L0 用例的 MERE/MARE 社区标准阈值（fp16 2^-10）是两回事，不能共用。

验证信号（复现增量，四条独立轨迹，均为 host Mock 实测）：

- ROIAlign 轨迹：Mock 构建 `Build mode: MOCK (CPU golden, no NPU)` 编译成功；golden 自测 5/5（手算 3 例 + golden.py 交叉校验 Case A MERE=1.347e-07/MARE=1.743e-06、Case B MERE=2.432e-08/MARE=9.728e-08，紧阈值 1e-5）；Mock 全量 L0 exit code 0（`L0_001 [PASS] MERE=0.000e+00, threshold=1.221e-04, 627200 elems`）（来源平台：x86_64 host（Mock，无 NPU）；证据等级：tool_observed）。
- TopK 轨迹：CPU golden 自测 6 组全 PASS（含 fp16 NaN/Inf topk 与比对函数扰动验证）；L0 用例 PASS（MERE=0.00e+00、gather 校验 PASS）；`bash run.sh --mock` 端到端 PASS；Real 模式在算子包未安装时 cmake 报 `Custom operator include directory not found`（预期前置行为，非缺陷）（来源平台：x86_64 host（Mock，无 NPU）；证据等级：tool_observed）。
- GQA 注意力轨迹：`cmake -DUSE_MOCK=ON` + make 编译通过（零 CANN 依赖）；golden 自测 5/5、L0 用例 6/6 PASS（MERE=0.00e+00、threshold=9.77e-04、每用例 16384~2097152 elems）；`bash run.sh --mock` 端到端复跑全 PASS（来源平台：x86_64 host（Mock，无 NPU）；证据等级：tool_observed）。
- Depthwise 轨迹：首跑 golden 自测 2 FAIL（MERE=2.02e+00，期望 4.5 实际 5.5 等 4 处）——手算期望漏算 padding 边界 tap，修正后期望后自测 5/5 PASS（MERE=0）、L0 用例 8/8 PASS（阈值 9.77e-04，最大 200704 元素）（来源平台：x86_64 host CPU Mock；证据等级：tool_observed）。

失效边界（复现增量）：

- 交叉校验用例绑定 golden.py 的具体版本输出；golden.py 更新后需重新生成嵌入数据（跨版本稳定性未验证）。
- 四条轨迹的 Real 模式执行在各会话切片内未闭环（依赖算子包安装与 NPU 可用）；`_Float16` 依赖 GCC 12+（x86），与本卡先前增量 主卡的位级转换路线二选一。
- 修正期望时的反投机取巧红线：golden 与手写期望不一致时先复核期望数据本身（独立手算/交叉校验），不得反向"修 golden 凑期望"。

## 后续增量（2026-09-17，并行策展批 A：MlaProlog 轨迹——Mock 恒零的证据语义与位级交叉验证）

MlaProlog 算子（MLA 前处理融合，bf16，CANN 9.2.0 工具链环境）的 C++ ST Mock 轨迹补齐本卡的证据语义边界与交叉验证方法：

**1. Mock MERE=0 的证据语义（判读规则）**：Mock 模式下 L0 6/6、L1 21/21 全部 MERE=0.00e+00/MARE=0.00e+00——这是 golden 与自身比较的恒零值，**只证明框架闭环，不构成 golden 正确性证据**；C++ golden 移植若有 bug，会在真机联调时伪装成 kernel 精度缺陷，浪费联调轮次。Mock 全绿的唯一正确性证据来自组件级自测 + 端到端位级交叉验证。

**2. 三步验证方法**：

- **组件级自测**：对 golden 的每个数值组件（类型转换、归一化、旋转、矩阵乘）用已知值小用例验证，覆盖退化路径（全零输入走 epsilon、sin=0/cos=1 恒等、sin=1/cos=0 纯 rotate_half、MatMul 2x2 已知值）。
- **端到端位级交叉验证**：Python 侧 `torch.manual_seed(42)` 生成小规模固定输入，`.bfloat16()` 量化后按 float 打印输入与期望输出，**并打印输出的 bf16 原始 uint16 位型**（`tensor.view(torch.uint16)`）；C++ 侧嵌入同一组输入，先做 `Bf16ToFloat(FloatToBf16(v))` 量化往返（与 `.bfloat16().float()` 一致），跑 C++ golden 后逐元素比较 `Bf16ToFloat(got[i]) != Bf16ToFloat(FloatToBf16(exp[i]))`——**期望值也先过一次 bf16 量化再比较**，保证 bf16 位级一致（十进制文本往返可能引入末位误差，两侧统一"bf16 量化后比较"才是位级判定；uint16 位型打印作人工核对终审）。
- **之后才跑用例集**：golden 可信后，Mock 跑 L0/L1 CSV 用例（此时 MERE=0 只证明框架闭环）。

```cpp
    // --- 测试 7: 端到端 MlaProlog 与 Python golden.py 交叉验证 ---
    // 期望输出 (Python golden.py 的 bf16 输出, 以 float 表示)
    std::vector<float> exp_query      = {1.0234375f, -0.490234375f};
    std::vector<float> exp_query_rope = {0.74609375f, -0.287109375f, 2.328125f, 0.76953125f};
    ...
    // 逐元素精确比对 (bf16 级别)
    auto check = [&](const std::vector<bf16_t>& got, const std::vector<float>& exp, const char* name) {
        if (got.size() != exp.size()) { ... ok = false; return; }
        for (size_t i = 0; i < got.size(); ++i) {
            float got_f = Bf16ToFloat(got[i]);
            float exp_bf16 = Bf16ToFloat(FloatToBf16(exp[i]));
            if (got_f != exp_bf16) { ... ok = false; }
        }
    };
```

**3. 设计要点**：固定量化值文本嵌入而不是共享随机种子（C++ `std::mt19937` 与 torch 随机数生成器不可复现同一序列，固定量化值是最小可靠通道）；`#ifndef USE_MOCK_ACLNN` 守卫 aclnn 头与调用，CPU golden 替代 kernel 输出——只验证 golden 与框架，不验证 kernel。

### 复现增量 A 验证信号

- golden 自测 7/7 通过，其中测试 7 端到端与 Python golden.py 交叉验证：4 个输出（query/query_rope/c_kv/k_rope）bf16 位级一致（如 query=[1.0234, -0.4902] 与 Python 期望逐元素相等）（来源平台：950 环境主机侧 CPU Mock（x86 host，非 NPU）；证据等级：tool_observed）
- 组件级自测通过：bf16 转换往返、RMSNorm 已知值、RMSNorm 全零输入（epsilon 防除零）、RoPE 恒等、RoPE rotate_half、MatMul 2x2 已知值（来源平台：950 环境主机侧 CPU Mock；证据等级：tool_observed）
- Mock 用例集：L0 6/6、L1 21/21 通过（MERE=0/MARE=0，框架闭环证据）（来源平台：950 环境主机侧 CPU Mock；证据等级：tool_observed）

### 复现增量 A 边界

- Mock 通过不证明 kernel 正确（kernel 输出未参与比较）；真机 ST 是独立阶段。
- 固定输入交叉验证只覆盖小规模 shape；大规模 shape 的 golden 正确性由组件自测 + 同公式外推保证（未逐一验证）。

## 后续增量（2026-09-17，并行策展批 B：GRU 轨迹——三关门控实战抓出 yInternal 未回写）

BenchNoHarnessGru（多层/双向 GRU，fp32/fp16/bf16，Ascend950PR / CANN 9.2.0）的 C++ ST Mock 轨迹为本卡补上"golden 三关门控"形态与一个实战抓出的装配 bug：

**1. golden 三关门控（FATAL 前置）**：进入用例前先跑三关自测（手算小用例 S=1,B=1,H=1 / 零输入 / 双向 shape+有限值），任一失败 `[FATAL]` 终止——golden 自身错误不流入用例结果。设计要点：**零输入用例对"输出未写"类 bug 零检出**（期望本来就是 0），必须配手算用例；手算关第一时间暴露下述 bug。

**2. 实战抓出的 bug：RNN 族 golden 逐层循环后未回写最终输出（y 全 0 而 hn 对）**：ComputeGruGolden 的逐层循环里，每层输出只回填了下一层输入（`layerInput = layerOutput;`）；最终输出缓冲 `yInternal(S * B * outputDim, 0.0f)` 自零初始化后从未被写入——y 保持全 0。hn 不受影响（层内循环直接把 hFinal 写进 hnInternal）。修复 = 层循环结束后补一行回写（layerInput 每轮末尾=该层输出，循环结束后即最后一层输出）：

```cpp
        // layerOutput 作为下一层的输入
        layerInput = layerOutput;
    }

    // 最终层输出即为 y
    yInternal = layerInput;
```

回写放循环外一次赋值而非每层内写 yInternal：单层时语义等价，多层时只有最后一层输出该进 y，循环内写会被覆盖成中间层输出。判别路径可复用：双输出（y/hn）算子一个对一个错时，先查两个输出的装配路径差异，而不是怀疑共享计算（hn 正确证明门激活/矩阵乘等共享路径无误）。

**3. 工程要点**：内部统一 (S, B, D) 布局计算、batchFirst 只在输入/输出边界转置（核心循环不感知 layout）；门激活（Sigmoid/Tanh）与累加用 double、入口/出口 cast float（与 torch 的数值差异压到 1e-7 量级，使交叉验证能区分"golden 写错"与"浮点误差"）；MERE 主判 + MARE（阈 10×）辅判，阈值按 dtype 统一 0.05（GRU 含非线性多层累积，来自 TEST.md 精度标准）；交叉验证独立脚本用与 C++ 自测完全相同的固定输入调 Python golden.py 对拍。

### 复现增量 B 验证信号

以下信号来自 BenchNoHarnessGru ST 测试工程的 Mock 模式（CPU 实测；同轨迹后续 PyTorch 适配层在 NPU 上 33/33 通过，见同域适配层配方卡增量）：

- 修复前：手算用例 FAIL（期望 0.600920，实际 0.000000，误差 6.01e-01），hn 同用例 PASS（0.600895，误差 2.47e-05）；双向用例 y[0]/y[2] 全 0.0000（来源平台：950 环境主机侧 CPU Mock（x86 host，非 NPU）；证据等级：tool_observed）
- 修复后（`yInternal = layerInput;` 一行）：自测 3/3 ALL PASS——手算 h=0.600895（误差 2.47e-05）、双向 y[0]=[0.0473,0.0473,0.2524,0.2524]、y[2]=[0.1926,0.1926,0.1903,0.1903]（非零且方向区分可见）（来源平台：950 环境主机侧 CPU Mock；证据等级：tool_observed）
- 用例回归：L0 5/5 PASS（fp32/fp16/bf16/batchFirst/双向）；L1 28/28 PASS（多层 1-3、双向、有无 bias/h0 组合）（来源平台：950 环境主机侧 CPU Mock；证据等级：tool_observed）
- 与 torch golden.py 交叉验证（同输入同权重）：|y_python − y_cpp| = 2.86e-07（float 精度内一致），双向用例 max 逐元素差 < 1e-3（来源平台：950 环境主机侧 CPU Mock；证据等级：tool_observed）

### 复现增量 B 边界

- yInternal bug 修的是 CPU golden 脚手架；kernel 侧 y 全 0 的成因（kernel 未执行、切分层错误、输出搬运缺失）不在本增量。
- y 语义按 PyTorch GRU（y=最后一层全时间步输出）验证；其他框架 y 语义（如返回所有层）不适用。
- 零输入用例 PASS 不能作为唯一自测（对未回写类 bug 零检出）。