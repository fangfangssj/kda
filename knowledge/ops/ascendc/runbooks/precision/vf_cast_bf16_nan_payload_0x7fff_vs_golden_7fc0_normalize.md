---
type: "Runbook"
title: "bf16 输出 NaN 载荷 0x7FFF 需归一化为 0x7FC0"
description: "VF cast fp32→bf16 把全 1 mantissa 的 NaN 截成 0x7FFF，与 CPU golden 的 canonical quiet NaN 0x7FC0 bitwise 不等；cast 后按位归一化两处转换点修复。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# bf16 输出 NaN 载荷 0x7FFF 需归一化为 0x7FC0

## 现象

在 Ascend950PR（CANN 9.2.0）上调试一个 bf16 输入/输出的 VVFusion 算子 ST 精度，测试用 bitwise 比较对比 CPU golden（`op::bfloat16` 转换）。NaN/Inf 注入用例失败，且只有 NaN 元素错：

```text
Test E1_NaNInject: FAILED (bitwise mismatch, output 16/64 elems differ
  golden[0..7]: 7fc0 7fc0 7fc0 7fc0 7fc0 7fc0 7fc0 7fc0
  npu[0..7]:    7fff 7fff 7fff 7fff 7fff 7fff 7fff 7fff
  first mismatch at elem 0 (b=0 l=0 hc=0 d=0): golden=nan npu=nan
Test E8_InfInject: FAILED (bitwise mismatch, output 16/64 elems differ
  golden[0..7]: 7fc0 ... npu[0..7]: 7fff ...
```

注意现象特征：两侧都是 NaN（打印值均为 nan），**仅 16 位位模式不同**——golden 是 `0x7FC0`（canonical quiet NaN），NPU kernel 是 `0x7FFF`。数值语义等价但 bitwise 用例判失败。

## 适用场景

- 工程形态：ACLNN 算子 ST 精度验证（bitwise/严格位级比较 golden 的 bf16 输出），kernel 内部 fp32 计算、输出前 VF cast 到 bf16。
- API 路径：`asc_vf_call` 的 fp32→bf16 向量 cast（`__simd_vf__` 包装）、kernel 侧标量 fp32→bf16 转换函数；CPU 侧 golden 用 `op::bfloat16(nan)` 类转换。
- 验证过的平台与 CANN 版本：Ascend 950（950PR）+ CANN 9.2.0。
- 适用粒度：operator_family（所有 fp32 内算 + bf16 输出 + 位级比对的 AscendC 算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：容差比较（非 bitwise）的测试不受影响，无需归一化；fp16 输出、整数输出的算子未验证。

## 根因

NPU 侧 VF cast 对 NaN 的 mantissa 截取保留了全 1 载荷，而 CPU golden 转换产出 canonical quiet NaN，两者 16 位位模式不同。

机制链（逐步）：bf16 布局 = 1 位符号 + 8 位指数 + 7 位 mantissa；NaN 判定 = 指数全 1 且 mantissa ≠ 0。

1. 注入的 NaN/Inf 经算术（如 Inf 参与 dot/rmsnorm 链）在 fp32 域产生 NaN，fp32 NaN 的 23 位 mantissa 通常为全 1（`0x7FFFFF`）。
2. NPU VF cast fp32→bf16 取高 7 位 mantissa → `0x7FFF`（符号 0 + 指数 11111111 + mantissa 1111111）。
3. CPU 侧 `op::bfloat16(nan)` 转换输出 canonical quiet NaN `0x7FC0`（mantissa 1000000，最高 mantissa 位为 quiet 位）。
4. `0x7FFF ≠ 0x7FC0` → bitwise 比较失败；仅注入的 NaN 元素（本例 16/64）不一致，其余元素 bitwise 全等。

## 解决配方

两个转换出口都要归一化：**向量 cast 后的输出 buffer** 和 **标量转换函数**。只修其一不行（本流先修输出路径后 conv_state_out 仍报 NaN 不匹配，补修标量函数才全过）。

**出口 1：VF cast 后对输出 buffer 做按位 NaN 归一化（before → after）：**

```c
        // S21: Cast→bf16, CopyOut
        CHUNKED_CAST_OUT_F32_BF16((__ubuf__ bfloat16_t*)bufB1_.Get<T>().GetPhyAddr(),
            (__ubuf__ float*)bufB0_.Get<float>().GetPhyAddr());
        // Normalize NaN to 0x7FC0 (quiet NaN) to match golden's op::bfloat16(nan)
        { auto b1=bufB1_.Get<T>();
          for(int64_t d=0;d<D_;d++) {
              T bv=b1.GetValue(d); uint16_t bits; __builtin_memcpy(&bits,&bv,sizeof(bits));
              if((bits&0x7F80)==0x7F80 && (bits&0x007F)!=0) { bits=0x7FC0; __builtin_memcpy(&bv,&bits,sizeof(bits)); b1.SetValue(d,bv); }
          }
        }
```

掩码推导：`(bits&0x7F80)==0x7F80` 命中指数全 1（bf16 指数域为 bit14..bit7，掩码 `0x7F80`）；`(bits&0x007F)!=0` 排除 Inf（mantissa 为 0）；命中即改写为 `0x7FC0`。

**出口 2：标量 fp32→bf16 转换函数加同款归一化：**

```c
__aicore__ inline uint16_t fp32_to_bf16_rint_scalar(float v) {
    union { uint32_t u; float f; } c; c.f = v;
    // Normalize NaN to 0x7FC0 (quiet NaN) to match golden's op::bfloat16(nan)
    if((c.u & 0x7F800000) == 0x7F800000 && (c.u & 0x007FFFFF) != 0) return 0x7FC0;
    return (uint16_t)((c.u + 0x7FFF + ((c.u >> 16) & 1)) >> 16); }
```

fp32 侧掩码：`(c.u & 0x7F800000)==0x7F800000` 指数全 1、`(c.u & 0x007FFFFF)!=0` 非 Inf → 返回 `0x7FC0`；正常值仍走 round-to-nearest-even（`+0x7FFF + lsb`）路径。

## 设计决策

- 为何在 kernel 侧归一化而不是放测试容差：bitwise 用例的语义就是位级一致，golden 的 `0x7FC0` 是 CPU 转换的确定行为；kernel 输出 canonical quiet NaN 是合法且更规范的表达（IEEE 754 推荐 canonical NaN），属 kernel 侧实质修复而非测试侧放松。
- 备选（未采用）：在 VF cast 的 `__simd_vf__` 包装内做掩码归一化——需要额外 VF 指令且改动主链，风险大于输出后一次性标量扫描（NaN 元素占比极小，标量循环开销可忽略）。
- 同族 NaN 位模式卡的选用条件（修哪侧取决于 golden 对 NaN 的处理）：golden 保留原始 payload 时修测试侧——[AscendC::Abs 规范化 NaN payload 致逐位验收误报](ascendc_abs_nan_payload.md)（numpy.abs 保留 0x7e00，用 NaN 感知比对而非改 kernel）；输出契约要求逐位保留原始输入 payload 时修 kernel 数据流——[b16 running 经 fp32 回转会改写 NaN payload](bit_exact_b16_nan_payload_needs_original_dtype_running_value.md)（不经 fp32 回转、且明言不在输出端修补，与本卡"输出端归一化到 canonical NaN"的契约相反）；[Cast SatMode::SAT 饱和致 fp16 NaN/Inf 丢失](cast_satmode_sat_fp16_nan_inf.md) 是饱和把 NaN/Inf 变 65504、机制不同；[MX-FP4 NaN 块 device 写 e8m0=0xFF](mx_fp4_nan_device_e8m0_0xff_fp4_0x0.md) 是量化块级 NaN 表示分歧。本卡 golden 规范化为 0x7FC0，故修 kernel 侧。
- 当前方案属 recommended_fastpath。

## 验证信号

以下信号均来自同一条开发会话内的 ST 实测输出：

- 修复前：E1_NaNInject / E8_InfInject 均 `FAILED (bitwise mismatch, output 16/64 elems differ)`，golden 位模式 `7fc0`、npu 位模式 `7fff`（来源平台：950；证据等级：tool_observed）。
- 仅修输出路径后：两用例主输出 `0/64 elems differ`（NaN 位模式已一致），但 conv_state_out 仍有 NaN 不匹配，用例仍 FAILED——证明两个转换出口必须同时归一化（来源平台：950；证据等级：tool_observed）。
- 补修标量转换函数后：`Test E1_NaNInject: PASSED`、`Test E8_InfInject: PASSED`，全套 ST 由 14/23 升至 16/23，其余用例无退化（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅覆盖 bf16 输出、bitwise 比较、CPU golden 产 canonical quiet NaN 的组合；若 golden 实现产出其他 NaN 载荷（如保留源 mantissa），掩码目标值需相应调整。
- 标量归一化循环遍历整个输出行（O(D)），对全 NaN 大shape 输出性能未测；若成为热点可改 VF 按位选择实现（未验证）。
- fp32 直接输出（无 cast）的场景不受此坑影响。
- 其他 NPU 代/编译器版本的 VF cast NaN 载荷行为未验证（推断同类，需实测）。

**未决项**：

- 无（本卡根因与修复已闭合）。
## 误诊

- 容易误判为"精度误差/容差问题"而放宽容差：现象是位模式差异而非数值误差，打印值两侧都是 nan，放宽 rtol/atol 无意义且属测试侧放松。
- 容易只修主输出路径：conv_state_out 等第二输出走标量转换函数时仍会带 `0x7FFF`，必须逐转换出口排查。
