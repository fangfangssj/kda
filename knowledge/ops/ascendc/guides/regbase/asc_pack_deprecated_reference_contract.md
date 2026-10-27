---
type: Guide
title: asc_pack (废弃)：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。该接口已废弃，请使用 asc_pack_to_low 和 asc_pack_to_high
  替代。 将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分或高半部分。 - asc_pack：将源操作数写入目的操作数的低半部分。
  - asc_pack_v2：将源操作数写入目的操作数的高半部分。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_compress/asc_pack_deprecated.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_pack.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_pack
- asc_pack_v2
---
# asc_pack (废弃)

该接口已废弃，请使用 asc_pack_to_low 和 asc_pack_to_high 替代。

将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分或高半部分。

- asc_pack：将源操作数写入目的操作数的低半部分。

- asc_pack_v2：将源操作数写入目的操作数的高半部分。

参数：dst(输出): 目的操作数（矢量数据寄存器/掩码寄存器）。；src(输入): 源操作数（矢量数据寄存器/掩码寄存器）。

历史类型配对包括 int16_t、int32_t、uint16_t、uint32_t、uint8_t 对应寄存器，以及 vector_bool 掩码寄存器重载，见下文完整声明。

仅 Ascend 950PR/Ascend 950DT 支持。

## 历史 C 重载与替代边界

本节仅保存固定旧 reference 的 950PR/950DT 调用合同，用于理解旧代码。接口仍按本卡前文标记为已废弃，应根据目标版本使用 [asc_pack_to_low](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_pack_to_low.md) 或 [asc_pack_to_high](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_pack_to_high.md)；补充历史声明不代表恢复推荐使用。

```cpp
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_uint16_t src);
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_int16_t src);
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_uint32_t src);
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_int32_t src);
__simd_callee__ inline void asc_pack(vector_bool& dst, vector_bool src);
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_uint16_t src);
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_int16_t src);
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_uint32_t src);
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_int32_t src);
__simd_callee__ inline void asc_pack_v2(vector_bool& dst, vector_bool src);
```

asc_pack 写目的低半，asc_pack_v2 写高半。每一形式包括四组整数宽度配对和一组 vector_bool 配对：有符号/无符号 16 位输入压至 vector_uint8_t，32 位输入压至 vector_uint16_t。这里取低位是位模式压缩，不是完整数值的饱和转换；来源未给未写半部分的全部保留规则，不推导其为零。返回 void，流水为 PIPE_V。

替代卡描述了 vector_bool 按每两 bit 选取低位的压缩原理，但旧 reference 这里只给出 bool 重载，不能由新接口描述断言所有旧版本的全部细节相同。概述提到 b64，而本页历史声明未列出 b64 类型配对，也不能据此补造 C 重载。

迁移旧代码时，核对 SDK 中旧符号是否仍可用、对应替代符号、输入输出宽度、写入半区和其余位的有效性。可用高低位不同的整数与交替掩码观察位选择，不把数值 Cast 的验证替代位压缩验证。本卡没有执行实际 SDK 编译或 NPU 对照。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_pack (废弃)](../../apis/simd_api/c_api/deprecated_interface/asc_pack_deprecated.md)
