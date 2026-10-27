---
type: "Runbook"
title: "x86 host fp16：探测 _Float16，兜底位级转换"
description: "x86_64 g++ 无 __fp16（ARM 侧类型，报未声明且误导性建议 __bf16）；_Float16 在不同会话的 g++ 13.3.0 上可用性观测不一致，写码前必须用 __SIZEOF_FLOAT16__ 探针确认。可用则 _Float16 + uint16_t 位存储（已知位模式验证）；不可用则手写位级 f32↔f16 RNE 软件转换；演示型示例可直接用 float + ACL_FLOAT。复现增量：真需要 fp16 数据时，链 ACL 的代码用官方 aclFloat16(uint16_t) + aclFloatToFloat16/aclFloat16ToFloat；不链 ACL 的代码（GE IR 示例）自带简化位级转换（注意 subnormal 边界）。"
tags: [task.testing, topic.compilation, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T01:58:00Z'
platforms: ["950"]
---

# x86 host fp16：探测 _Float16，兜底位级转换

## 现象

在 x86_64 主机侧为 aclnn 自定义算子写 C++ ST 测试 / UT / 调用示例，需要生成或校验 fp16 数据时，按 ARM 习惯写 `__fp16`（如 `std::vector<__fp16> yData(ySize, 0);` 或 `__fp16 h = 1.5f;`），g++ 直接报未声明，且编译器的建议具有误导性（建议的是 bf16 而不是可用的 fp16 类型）：

```text
error: '__fp16' was not declared in this scope; did you mean '__bf16'?
   45 |     std::vector<__fp16> yData(ySize, 0);
```

模板参数非法后 `yData` 整体退化为 `int`，级联出 `request for member 'data' in 'yData', which is of non-class type 'int'`、`invalid types 'int[int]' for array subscript` 等错误，淹没第一条真实原因。

改用 `_Float16`（GCC/Clang 的 ISO 扩展类型）在部分会话编译通过（带 float→_Float16 转换 warning），但在另一会话的同版本编译器（g++ 13.3.0，`-std=c++17`）上探针输出 `_Float16 NOT available`（`__SIZEOF_FLOAT16__` 未定义）——可用性不能假设，必须探测。

## 适用场景

- 探针通过且仅需类型别名时，按架构条件选型可直接用 `_Float16`（见 [x86_64 主机 __fp16 不可用：按架构选 _Float16](x86_64_host_fp16_use_float16.md)）；本卡的探针+位级转换路线是可用性不一致时的兜底。

- 工程形态：aclnn 自定义算子的 C++ ST/UT 测试、host 侧诊断程序与交付示例（x86_64 主机构造/校验 fp16 数据，喂给 ACL_FLOAT16 张量）。
- API 路径：`aclCreateTensor` + `ACL_FLOAT16`（设备侧 2 字节/元素）、`aclrtMemcpy` 字节流；主机侧 `_Float16`（GCC/Clang 扩展）与 `uint16_t` 位存储；纯 ISO C++ 位运算转换（memcpy 位重解释 + 整数运算）。
- 验证过的平台与 CANN 版本：x86_64 主机（Ubuntu g++ 13.3.0，-std=c++11/-std=c++17 均有实测）+ 目标 Ascend950PR / CANN 9.1.0（与 frontmatter 一致）。
- 适用粒度：platform_wide——所有在 x86 主机侧处理 fp16 数据的测试/示例代码。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：MSVC（无 `_Float16` 同名语义，未验证）；ARM/AArch64 宿主（`__fp16` 可用，不需要本配方）；bf16 host 侧处理（`__bf16` 可用，另一条路）；设备侧 kernel 内 `half`/`bfloat16_t`（AscendC 自有类型）；F16C 硬件指令（`_cvtss_sh`）路线未采用（可移植性差）。

## 根因

`__fp16` 是 ARM/AArch64 的 GCC 类型扩展，x86_64 g++（C++ 模式）不提供；编译器 "did you mean __bf16" 的建议是另一种 brain-float 类型（尾数 8 位），不能当 fp16（尾数 10 位）用。ISO C23 之前的 GCC/Clang 以扩展类型 `_Float16` 提供 IEEE 754 binary16 主机侧支持，但其可用性取决于编译器版本与编译配置——本批多轨迹在同为 g++ 13.3.0 的环境下观测到可用与不可用两种结果，因此任何 host 侧 fp16 代码都应先探测再选型。测试数据在主机侧以 2 字节/元素组织时，用 `uint16_t` 持有位模式 + `_Float16`（或软件转换）做数值转换即可覆盖生成/比对两端。

## 判别方法

**探针 1（`__fp16` 不可用的触发信号）**：

```bash
echo 'int main(){__fp16 a=1.0; return (int)(float)a;}' > /tmp/test_fp16.cpp && g++ -c /tmp/test_fp16.cpp -o /tmp/test_fp16.o
# → error: '__fp16' was not declared in this scope; did you mean '__bf16'?
```

**探针 2（`_Float16` 可用性）**：

```cpp
#if defined(__SIZEOF_FLOAT16__) && __SIZEOF_FLOAT16__ == 2
printf("_Float16 available\n");
#else
printf("_Float16 NOT available\n");
#endif
```

**探针 3（转换链位模式自检，无论走哪条路线都跑一遍）**：已知值快速验证——1.5f 的 binary16 表示是 `0x3e00`、1.0f→`0x3c00`、65504（最大有限）→`0x7bff`、65536→`0x7c00`（Inf）、0.1f→`0x2e66`、NaN→`0x7e00`、bf16 1.5→`0x3fc0`。

## 解决配方

按探针结果三段选型：

**路线 A（`_Float16` 可用，首选）**：`_Float16` 做数值转换 + `uint16_t` 位存储双轨（张量搬运按字节 memcpy，`uint16_t` 视图避免对 `_Float16` 数组做指针双关）：

```cpp
// 探针程序：验证 _Float16 可用性与位模式（可直接嵌入 ST 测试）
#include <cstdio>
#include <cstdint>
#include <cstring>
int main() {
    _Float16 h = static_cast<_Float16>(1.5f);
    float f = static_cast<float>(h);
    printf("h=%f, f=%f\n", (double)h, f);
    _Float16 h2 = static_cast<_Float16>(0.1f);
    printf("h2=%f\n", (double)h2);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    printf("h bits=0x%04x\n", bits);   // 1.5f -> 0x3e00
    return 0;
}
```

工程化要点：数据生成按 fp32 生成值域再 `static_cast<_Float16>` 舍入、memcpy 到 `uint16_t` 缓冲送 `aclCreateTensor(..., ACL_FLOAT16, ...)`；精度比对读回 `uint16_t`、memcpy 到 `_Float16` 后 `static_cast<float>` 参与 MERE/MARE（fp16 阈值 2^-10 ≈ 9.77e-04）；float→_Float16 的隐式转换 warning 用 pragma 局部关闭，不要全局禁。

**路线 B（`_Float16` 不可用，位级软件转换兜底）**——f32↔f16 round-half-to-even 全分支实现（与 NPU Cast 的 CAST_RINT 及 numpy `astype` 舍入对齐）：

```cpp
// fp16 bits → float (exact, fp32 is a strict superset of fp16)
static float f16ToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                  // ±0
        } else {
            // Subnormal fp16 → normalize to fp32 normal
            int e = 0;
            while (!(mant & 0x400)) { mant <<= 1; ++e; }
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e - 1 + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | (0xFFu << 23) | (mant << 13);       // Inf / NaN
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// float → fp16 bits with round-half-to-even (CAST_RINT)
static uint16_t floatToF16Rn(float val)
{
    uint32_t u;
    std::memcpy(&u, &val, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    uint32_t exp_mant = u & 0x7FFFFFFF;

    if (exp_mant > 0x7F800000) {
        // NaN: preserve sign, make quiet NaN (mantissa non-zero)
        uint16_t mant = (uint16_t)((u >> 13) & 0x3FF);
        if (mant == 0) mant = 1;
        return (uint16_t)(sign | 0x7C00u | mant);
    }
    if (exp_mant >= 0x47FFE000) {     // overflow → ±Inf
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp_mant <= 0x33000000) {     // |val| < 2^-24 → ±0
        return (uint16_t)sign;
    }
    if (exp_mant < 0x38800000) {      // subnormal fp16 range
        uint32_t mant = (exp_mant & 0x7FFFFF) | 0x800000;
        int shift = 113 - (int)(exp_mant >> 23);
        if (shift < 1) shift = 1;
        if (shift > 24) shift = 24;
        uint32_t hm  = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) ++hm;
        return (uint16_t)(sign | hm);
    }
    // normal fp16
    uint32_t exp  = (exp_mant >> 23) - 127 + 15;
    uint32_t mant = exp_mant & 0x7FFFFF;
    uint32_t hm   = mant >> 13;
    uint32_t rem  = mant & 0x1FFF;
    uint32_t half = 0x1000;
    if (rem > half || (rem == half && (hm & 1))) {
        ++hm;
        if (hm == 0x400) { hm = 0; ++exp; }
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | hm);
}
```

NaN 分支尾数兜底非零是必须的：fp32 NaN 直接截位可能落到 fp16 的 Inf 位型，NaN 传播用例会假失败。数据生成侧用"0.5 的倍数"构造输入（fp16 精确可表示），避免生成侧自身引入舍入噪声。

**路线 C（演示型示例的最简解）**：交付示例的目的是演示 aclnn 两段式调用模式时，host 侧数据直接用 `float`（`std::vector<float>`）+ `ACL_FLOAT` 配对（GE IR 示例 `DT_FLOAT`），完全避开 host 侧 fp16 类型问题；host 数据类型必须与 aclTensor dtype 严格配对——`aclrtMemcpy` 只按原始字节拷贝，host 4 字节 float 配 2 字节 ACL_FLOAT16 会使设备侧数据解释错乱。

## 设计决策

- 位存储统一 `uint16_t`：任意 x86 编译器行为一致，位模式可直接打印（失配诊断时能看到 `0x3800(0.500000)` 原始位），对精度排障更有用。
- RNE 而不是截断/四舍五入：golden 侧转换必须与 NPU Cast 的 CAST_RINT 和 numpy `astype` 一致，否则 golden 自身引入 ulp 偏差，精度用例假失败。
- 比对策略 bitwise 优先（非计算类/纯搬运算子）：golden 与实现都应逐位一致；bitwise 先行能把"布局错"与"数值舍入差"区分开，MERE/MARE 只留给真正含计算的路径兜底（MARE 上限取 10×MERE 阈值）。
- 放弃的候选：F16C intrinsics（x86 限定、可移植性差）；`half` 等第三方头（引依赖）；升示例工程 `-std`（动构建配置，影响面大于示例自身需要）。

## 预防

- 写 x86 主机侧 fp16 代码前先跑一遍探针（`__fp16` 触发 + `__SIZEOF_FLOAT16__` + 位模式自检），确认工具链类型可用再展开；跨机器/跨编译配置复用时重跑。
- 搜索代码库时注意 `__fp16` 多为 ARM 侧写法，跨平台测试代码统一 `_Float16` 或位级转换。
- 不要顺着编译器建议把 fp16 数据改成 `__bf16`——尾数宽度不同（8 vs 10 位），会静默改变测试语义。

## 验证信号

来源说明：以下信号来自本批 4 个独立开发会话（x86_64 主机 g++ 13.3.0；目标 Ascend950PR / CANN 9.1.0）。

- `__fp16` 触发探针：x86_64 GCC 13.3.0 上报 `error: '__fp16' was not declared in this scope; did you mean '__bf16'?`（ST 测试与交付示例两个独立会话均复现；示例侧并级联模板参数非法/成员访问错误）（来源平台：950 开发环境 host 编译（g++ 13.3.0，-std=c++11）；证据等级：tool_observed）。
- `_Float16` 可用性观测（可用侧）：`_Float16` 版编译通过并输出 `h=1.500000, f=1.500000`、`h2=0.099976`、`h bits=0x3e00`——转换与位模式正确（来源平台：950 开发环境 host（g++ 13.3.0）；证据等级：tool_observed）。
- `_Float16` 可用性观测（不可用侧）：`__SIZEOF_FLOAT16__` 未定义，探针输出 `_Float16 NOT available`（g++ 13.3.0，-std=c++17）——同版本编译器不同会话结论相反，可用性必须逐环境探测（来源平台：950 开发环境 host（g++ 13.3.0）；证据等级：tool_observed）。
- 位级转换逐点验证（路线 B）：f32↔f16 位模式逐点命中——1.0→0x3c00、2.0→0x4000、1.5→0x3e00、60000→0x7b53、65504→0x7bff、65536→0x7c00(Inf)、0.1→0x2e66、NaN→0x7e00、Inf→0x7c00、0.0→0x0000；bf16 1.5→0x3fc0、1.0→0x3f80、fp32 NaN(0x7fc00000)→0x7fc0（来源平台：950 开发环境 host（g++ 13.3.0）；证据等级：tool_observed）。
- 位级转换进入 ST golden 后 16 例（含 fp16/bf16 用例）全量执行无崩溃，四 dtype 用例在后续 kernel 路由修复的中间态中转为通过——转换正确性随 ST 闭环（来源平台：950；证据等级：tool_observed）。
- 路线 C（示例改 FP32）：`[100%] Built target test_aclnn_<op>` 与 GE IR 示例 `[100%] Built target test_geir_<op>`（仅余 array_ops.h 弃用告警，不影响链接）（来源平台：950 开发环境 host 编译（g++ 13.3.0，链接 CANN 9.1.0 vendors 包）；证据等级：tool_observed）。

## 失效边界

- `_Float16` 是 GCC/Clang 扩展（C23 起标准化）；不可移植到 MSVC（未验证替代方案）。本卡两个方向的可用性观测均 tool_observed，但未追到"同版本 g++ 结论相反"的配置层原因（编译 flags/目标架构选项差异未隔离）——使用前以探针为准。
- 位级转换舍入模式仅实现 RNE；RTZ/RNA 需另改比较分支。fp16 次正规区间的移位路径用 [1,24] 钳位实现，极端移位输入已被上/下溢分支截走。
- 路线 C 示例仅做编译验证，未上板运行；示例二进制的运行时正确性（含 FP32 路径输出）未验证。
- 与 numpy `astype` 的全区间随机对拍未执行（只做了已知位模式逐点验证 + 用例级闭环），可作加强项。

**未决项**：

- 同为 g++ 13.3.0 的两个会话 `_Float16` 可用性结论相反的配置层原因（编译 flags/架构选项）未隔离——复现并定位该差异是优先项。
- 与 numpy `astype` 的全区间随机对拍未执行。
## 坏实践

- 在 x86 C++ 代码里直接写 `__fp16` 并期望它等同 fp16——g++ C++ 模式直接报未声明，且模板参数非法后的级联报错会淹没真实原因。
- 顺着编译器建议改成 `__bf16`——bf16 与 fp16 尾数宽度不同，静默改变测试语义。
- 假设 `_Float16` 一定可用（或一定不可用）而不跑探针。
- golden 用"四舍五入"（round-half-away）截位——与 NPU/numpy 的 RNE 在半数处差 1 ulp，精度用例偶发假失败。
- fp32 NaN 直接截位到 fp16 不做尾数兜底——可能得到 Inf 位型，NaN 传播用例假失败。
- host 数据类型与 aclTensor dtype 不配对（host float 4 字节 + 张量 ACL_FLOAT16 2 字节）。

## 误诊

- 先写 `.c` 文件用 gcc 编译探针，报 `fatal error: cstdio: No such file or directory`——是 gcc 按 C 编译 C++ 头的问题，换 `.cpp` + g++ 即可，别据此误判环境缺头文件。
- 交付示例侧曾推断"-std=c++11 不支持 _Float16、需升级 C++14"——该推断未做实验验证，不能当实测结论（路线 C 直接改 FP32 闭合，未走该分支）。

## 增量（真 fp16 数据的两条落地方针：ACL 官方转换 / 自带位级转换）

本增量补"示例必须真用 fp16 数据"（而非路线 C 避开）时的两条路线，来自一个自定义算子 aclnn + GE IR 双示例的交付会话（Ascend950PR + CANN 9.1.0，x86_64 Ubuntu + GCC 13.3.0 host 编译、上板运行）：两个示例的 host 侧代码都用 `__fp16` 准备/打印 FP16 数据，`make` 时同时报错——

```text
examples/test_aclnn_bench_no_harness_swi_glu.cpp:44:17: error: '__fp16' was not declared in this scope; did you mean '__bf16'?
   44 |     std::vector<__fp16> resultData(size, 0);
examples/test_geir_bench_no_harness_swi_glu.cpp:91:9: error: '__fp16' was not declared in this scope; did you mean '__bf16'?
```

**路线 D1（链 ACL 的代码——ACLNN 示例，用官方类型与转换函数）**：CANN 的 host 侧契约是 FP16 数据以 `aclFloat16`（即 `uint16_t` 的位模式 typedef，定义于 `acl_base_rt.h`，经 `acl/acl.h → acl_rt.h → acl_base.h` 链可达）表示，转换用 `aclFloatToFloat16(float)` / `aclFloat16ToFloat(aclFloat16)`（verbatim）：

```cpp
    std::vector<int64_t> inputShape = {4, 8};
    std::vector<aclFloat16> inputHostData(32, aclFloatToFloat16(1.0f));
```

```cpp
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
```

链了 ACL 就用官方位级正确实现，避免自写转换的边界 bug；类型即 uint16_t，与 aclrtMemcpy 字节数计算天然一致。

**路线 D2（不链 ACL 的代码——GE IR 示例只链 GE/proto，自带位级转换）**：拿不到 ACL 转换函数，为一个转换引 ACL 依赖不值；自带一对简化 `FloatToHalf/HalfToFloat`（verbatim）：

```cpp
uint16_t FloatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = (x >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
}

float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    if (exp == 0 && mantissa == 0) {
        float zero;
        uint32_t zeroBits = sign;
        std::memcpy(&zero, &zeroBits, sizeof(zero));
        return zero;
    }
    uint32_t f = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}
```

数据准备与打印按 `uint16_t` 位存储走这对函数（`pData[i] = FloatToHalf(value)` / `HalfToFloat(pData[j])`）。

**边界（复现增量）**：路线 D2 的简化版**未覆盖非正规数（subnormal）路径**（exp==0 且 mantissa!=0 时按隐含逻辑返回带符号零附近值），数值敏感场景须改用本卡路线 B 的全分支 RNE 实现或路线 D1；aclFloat16 转换函数签名以 `acl_base_rt.h` 实际声明为准，其他 CANN 版本未验证（推断可迁移）。

**验证信号（复现增量，双示例编译修复与上板运行）**：

- 报错原文 `'__fp16' was not declared in this scope; did you mean '__bf16'?`（两个 target 各多处）（来源平台：950 开发环境 host（GCC 13.3.0）；证据等级：tool_observed）。
- 头文件确认：`acl/acl.h → acl_rt.h → acl_base.h → acl_base_rt.h` 链上定义 aclFloat16 与转换函数（来源平台：950 开发环境 host；证据等级：tool_observed）。
- 修复后 `make` 两 target `[100%] Built target`；ACLNN 示例上板输出 `result[0] is: 0.730957`（期望 silu(1.0)*1.0≈0.7311，fp16 精度内一致）；GE IR 示例输出 `output[0][0] = 0.730957`，run.sh eager/graph 两模式均 PASS（来源平台：950；证据等级：tool_observed）。