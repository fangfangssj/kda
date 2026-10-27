---
type: Guide
title: AscUnsqueeze：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据掩码对矢量数据进行解压缩，dst各元素值取决于mask中前序元素为1的数量
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_compress/asc_unsqueeze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_unsqueeze.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_unsqueeze
---
# AscUnsqueeze

根据 mask 生成解压所用的排他前缀计数并写入 dst。解压缩方式：dst中第0个元素置为0，dst中的第i个元素等于mask中从第0个到第(i-1)个元素中1的数量。mask最高位被忽略不参与统计。
dst[i] 由 mask[i-1] 决定是否在 dst[i-1] 上加 1，统计范围是 mask[0..i-1]，见下方递推。

参数：dst(输出): 保存生成计数的目的矢量数据寄存器，不从原 dst 数据解压。；mask(输入): 源操作数（掩码寄存器），用于提供dst解压缩信息。

支持数据类型：int16_t、int32_t、int8_t、uint16_t、uint32_t、uint8_t。

仅 Ascend 950PR/Ascend 950DT 支持。

## C 声明与排他前缀计数

固定 950PR/950DT reference 的六种整数目的寄存器声明如下；返回 void，流水为 PIPE_V：

```cpp
__simd_callee__ inline void asc_unsqueeze(vector_int8_t& dst, vector_bool mask);
__simd_callee__ inline void asc_unsqueeze(vector_uint8_t& dst, vector_bool mask);
__simd_callee__ inline void asc_unsqueeze(vector_int16_t& dst, vector_bool mask);
__simd_callee__ inline void asc_unsqueeze(vector_uint16_t& dst, vector_bool mask);
__simd_callee__ inline void asc_unsqueeze(vector_int32_t& dst, vector_bool mask);
__simd_callee__ inline void asc_unsqueeze(vector_uint32_t& dst, vector_bool mask);
```

递推使用前一位 mask，而非当前位：

```cpp
dst[0] = 0;
for (int i = 1; i < dstLen; ++i) {
    dst[i] = mask[i - 1] ? (dst[i - 1] + 1) : dst[i - 1];
}
```

这是解释算法的伪代码，dstLen 是目的逻辑元素数。例如 mask 前四位为 [1,0,1,0] 时，相应前四个结果是 [0,1,1,2]；输出位置是否为零不由同位置 mask 独立决定。最高掩码位不会进入前缀统计。本操作也不同于给张量增加维度的 unsqueeze。

来源调用示例误写成 asc_squeeze(dst, mask)；实际接口名和参数应以上述 asc_unsqueeze 声明为准。验证应覆盖全无效、全有效和交错掩码，并按所选整数类型与实际向量长度核对可表示范围。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_unsqueeze](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_unsqueeze.md)
