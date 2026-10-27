---
type: "Runbook"
title: fp16/bf16 位流转 float 零依赖转换
description: C++ 侧 fp16/bf16 设备位流→float 的位级转换（含 subnormal 规格化推导）与 python bf16 uint16 位桥，避免链接 CANN 转换库。
tags: [task.testing, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# fp16/bf16 位流转 float 零依赖转换


## 现象

在写自定义算子的 aclnn ST（C++ 测试二进制）时，需要把设备端读回的 fp16/bf16 输出位流与 golden 期望位流都转成 float/double 计算 rel_err。当时的处境：

- CANN 头文件 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/fp16_t.h` 提供了 `op::fp16_t` 类型，但它的 `operator float()` **在头文件里只有声明，没有实现体**——实现位于 CANN 的某个链接库中（参考工程 add_custom 的测试链接 `libascendcl`/`libnnopbase` 后可用 `toFloat/static_cast`，但具体是哪个库未做符号级定位）。
- bf16 侧没有 `aclBFloat16` 这样的 ACL 类型可比照；`bfloat16.h` 存在但同样引入头+库依赖面。
- 测试二进制已经链接 ascendcl/nnopbase/cust_opapi 三库，引入 fp16_t.h 技术上可行，但为了"零额外链接依赖、完全确定性、避免任何 ABI/链接意外"，最终选择手写位级转换（fp16→fp32 约 25 行，bf16→fp32 一行移位）。
- python 侧（oracle 生成器）有对偶问题：numpy 没有 bfloat16 dtype，torch bf16 张量不能直接 `.numpy().tobytes()` 落盘。

## 适用场景

- 工程形态：aclnn ST / 任何需要在 host 侧 C++ 里解释 fp16/bf16 原始位流的测试或诊断工具（设备上载/读回均按字节透传，只有数值解释需要转换）。
- API 路径：CANN `aclnn/opdev/fp16_t.h`（被规避的依赖）、`aclrtMemcpy` 字节流、torch `Tensor.view(torch.uint16)`（python 位桥）。
- 验证过的平台与 CANN 版本：Ascend950PR（npu-smi 25.7.rc1.6）+ CANN 9.0.0-beta.2，x86_64 小端宿主（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`operator_family`——转换与算子无关，是纯位语义；但"验证过的运行链路"是自定义算子 aclnn ST。
- 路线优先级：`recommended_fastpath`（与 `op::fp16_t` 路线相比：两者数值等价，本路线零链接面；若测试已必然链接 CANN 运行库，`op::fp16_t` 是可接受的 `safety_fallback`，代价是多一个头依赖与对其库归属的信任）。
- 明确不适用/未验证场景：设备侧 kernel 内的半精度运算（用 AscendC 内建类型/向量指令，不走本转换）；大端宿主（memcpy 位序需调整）；fp16 subnormal 元素未构造专门用例实测（见遗留问题）。

## 判别方法（位级语义与推导）

**bf16 → fp32（精确，一行）**：bfloat16 就是 fp32 截断高 16 位，两者共享 8 位指数域（偏置 127），因此 `bits = uint32_t(v) << 16` 后按位重释即得 float，Inf/NaN/subnormal 语义自动保持。

**fp16 → fp32（三分支 + subnormal 规格化）**：设 fp16 位型 h = sign(1) | exp(5) | man(10)：

- `exp == 0 && man == 0`：±0 → fp32 位型仅保留符号 `sign << 31`。
- `exp == 0x1F`：Inf/NaN → `sign<<31 | 0x7F800000 | (man << 13)`（尾数左移 13 位补齐 fp32 的 23 位尾数）。
- normal：`sign<<31 | ((exp − 15 + 127) << 23) | (man << 13)`。核对：h=0x3C00（fp16 1.0）→ exp=15 → 偏置域 127 → 0x3F800000 = 1.0f ✓；h=0x7BFF（65504）→ exp=30 → 偏置域 142 → 2^15×1.999…=65504 ✓。
- **subnormal（exp==0 且 man≠0）**：fp16 次正规数值 = man × 2^−24。左移 m=man 直到 bit10 置位，共移 s 次，则 `man × 2^s = 2^10 + frac`（frac = m & 0x3FF），即 `man = 2^(10−s) × (1 + frac/1024)`，故值 = `2^(−14−s) × (1 + frac/1024)`，fp32 指数 = −14−s，**偏置域 = 127 − 14 − s = 113 − s**，尾数 = `frac << 13`。核对：man=1 → s=10 → 偏置域 103 → 2^(103−127) = 2^−24 ✓（最小 fp16 subnormal）。

**float → fp16 / bf16（poison 方向，截断即可）**：只须"非零有限"时无须 round-to-nearest：溢出 clamp 到 0x7BFF（最大有限），下溢 clamp 到 0x0001（最小非零），正常域直接截断尾数（`man >> 13`）；bf16 直接 `bits >> 16` 并把 Inf/NaN 位型改为有限大值（0x7F7F）。数值语义方向（golden 比对）不要用这个截断版。

**python bf16 位桥**：torch bf16 张量 `t.contiguous().view(torch.uint16).numpy().tobytes()`——`view(dtype)` 同元素字节数重释，2 字节小端位流无损落盘；读回侧 numpy `view` 对称恢复。

## 设计决策

- **为何不用 `op::fp16_t`**：`operator float()` 头内声明、库内实现，引入对具体 CANN 库的链接依赖与版本耦合；手写 25 行零依赖、位级确定、可单测。参考工程路线（add_custom 直接用 fp16_t + 链接两库）被评估为可行但被放弃——这是链接面 vs 代码量的权衡，不是对错。
- **为何 bf16 走移位**：bf16 与 fp32 共享指数格式，无 subnormal 规格化复杂度，移位即精确。
- **为何 poison 方向单独写截断版**：见[输出 poison 防 stub 巧合命中卡](st_red_baseline_poison_prefill_exact_zero.md)——哨兵只须非零有限，舍入语义无关紧要；数值方向（本卡主转换）必须精确。
- **两端同源**：device 侧（C++ 读 .bin 位流）与 golden 侧（python 生成 .bin）消费同一份位流，转换只发生在"解释"环节——fp16→fp32 是精确拓宽（无舍入），比较在 fp32/double 域进行，不存在二次舍入。
- 当前方案属 `recommended_fastpath` 路线（`op::fp16_t` 为可接受 fallback，相对成本：+1 头依赖 +对其实现库的链接信任，无性能差异——两者都是 O(1) 位操作）。

## 系统信息

- 实现落点：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（`Fp16BitsToFloat`/`Bf16BitsToFloat`/`FloatToFp16Bits`/`FloatToBf16Bits`/`ElemToDouble`）；python 位桥在 `tests/aclnn/gen_st_oracle.py`（`save_bits`）。
- 被规避的依赖：`/home/<user>/Ascend/cann-9.0.0-beta.2/x86_64-linux/include/aclnn/opdev/fp16_t.h`（`op::fp16_t`，`operator float()` 仅声明）。
- 运行环境：Ascend950PR + CANN 9.0.0-beta.2；宿主 x86_64 小端；测试链 ascendcl/nnopbase/cust_opapi。
- 相关文件：`test_aclnn.cpp`、`gen_st_oracle.py`（均本 trace 写入）、fp16_t.h/bfloat16.h（只读参照）。

## 完整代码

verbatim 取自本 trace 的写入事件。

C++（数值方向，精确）：

```cpp
// fp16 位流 → float（值精确拓宽；含 subnormal / Inf / NaN）
static float Fp16BitsToFloat(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    const uint32_t exp = (h >> 10) & 0x1FU;
    const uint32_t man = h & 0x3FFU;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;  // ±0
        } else {
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400U) == 0) {
                m <<= 1;
                ++s;
            }
            bits = sign | (static_cast<uint32_t>(113 - s) << 23) | ((m & 0x3FFU) << 13);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (man << 13);  // Inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// bf16 位流 → float（bf16 = fp32 高 16 位，直接补零）
static float Bf16BitsToFloat(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
```

poison 方向（截断版，见[输出 poison 防 stub 巧合命中卡](st_red_baseline_poison_prefill_exact_zero.md)完整上下文）：

```cpp
// float → fp16 / bf16 位流（仅用于输出 poison 填充；截断舍入即可，只须非零有限）
static uint16_t FloatToFp16Bits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    const uint32_t expF = (x >> 23) & 0xFFU;
    const uint32_t man = x & 0x7FFFFFU;
    if (expF == 0xFFU) {
        return static_cast<uint16_t>(sign | 0x7BFFU);  // Inf/NaN → 最大有限（poison 保持有限）
    }
    const int32_t exp = static_cast<int32_t>(expF) - 127 + 15;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7BFFU);
    }
    if (exp <= 0) {
        return static_cast<uint16_t>(sign | 0x0001U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

static uint16_t FloatToBf16Bits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t bits = x >> 16;
    if (bits == 0x7F80U || bits == 0xFF80U) {
        bits = 0x7F7FU;  // Inf/NaN → 有限大值
    }
    return static_cast<uint16_t>(bits);
}
```

python（bf16 位桥，oracle 落盘）：

```python
def save_bits(t: torch.Tensor, path: Path) -> None:
    """目标 dtype 原始位流落盘（bf16 经 uint16 位桥保真）。"""
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        path.write_bytes(t.view(torch.uint16).numpy().tobytes())
    else:
        path.write_bytes(t.numpy().tobytes())
```

## 预防

- 在 C++ 测试里解释半精度位流前，先确认转换的三个分支都写了：normal / Inf-NaN / **subnormal**——值域小的随机数据（U(0,1)）不产生 subnormal，漏写后常规用例全对、极小值用例才炸，极难排查。
- bf16 不要找"类型"，认准"fp32 高 16 位"这一位事实，移位即得。
- python 落盘 bf16 一律走 `view(torch.uint16)` 位桥，不要尝试 `.numpy()`（numpy 无 bf16，直接抛错）或 float 中转（丢精度）。

## 验证信号

来源说明：以下信号来自 GRU Cell 反向算子 Branch-0 aclnn ST 的红基线运行（kernel 接口桩、输出为 poison 哨兵），转换器随整套 ST 首次编译运行。

- fp16 用例（E02/S6/P5）与 bf16 用例（E03）全部经转换器完成比对并产出量级自洽的指标：E02 `mere=7.963e+06 mare=2.280e+08`（与 fp16 哨兵 30000 除以 |g|∈O(0.05–2) 的推导量级一致）、E03 `mare=1.136e+35`（bf16 哨兵 3e30 量级一致）——转换链未引入畸变（来源平台：950；证据等级：tool_observed）。
- S6（fp16 全 60000 输入）golden 输出经 python 位桥落盘，其中 bias 输出的 6 个 fp16 溢出 Inf 位置被 C++ 侧 Inf 匹配逻辑逐位检出（`nan_inf_mismatch=6`）——bf16/fp16 的 Inf 位型经转换后语义保持（来源平台：950；证据等级：tool_observed）。
- 独立旁证：fp32 用例 E01 的 golden 期望位流经纯 numpy 从 .bin 复算五路输出 bitwise-equal（转换/生成链正确性的对照实验）（来源平台：950；证据等级：tool_observed）。
- 编译链接通过（C++17，零新增链接依赖）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅覆盖 fp16/bf16 → fp32 与 fp32 → fp16/bf16（截断版）；fp64/int 系不在内。
- fp16 subnormal 分支的推导已按位验证过公式（113−s 偏置域推导见判别方法），但未构造 subnormal 实际数据实测——用例值域 U(−2,2)/U(0,1) 不产生 subnormal。
- 大端宿主未适配（memcpy 位序假设小端；x86_64 实测）。
- 截断版 float→fp16 **不可**用于数值语义（golden 期望生成等）——只用于哨兵填充；数值方向须 round-to-nearest-even（如 torch `.to(torch.float16)`）。

## 坏实践

- 把 uint16 位型直接 `(float)h` 强转——这是整数到浮点的数值转换（h=0x3C00 → 15392.0），不是位重释；必须走位拼接。
- bf16 期望位流经 float 中转落盘（先 `.to(torch.float32).numpy()` 再截断）——两次舍入，与 device 侧位流不再两端同源。
- 忽略 subnormal 分支（见预防节）。

## 误诊

- "CANN 头里的 fp16_t 拿来 static_cast 就行"——`operator float()` 只有声明，实现在库里；不链接对应库直接编译失败或链接失败。参考工程能用是因为它恰好链接了含实现的库。
- "numpy 支持 bfloat16"——不支持；必须走 uint16 位桥。


