---
type: API
title: ICachePreLoad
description: 从指令所在GM地址预加载指令到ICache中，支持配置预取长度，提升指令读取效率。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cache_control/ICachePreLoad_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_cache_intf.h
---

# ICachePreLoad\(ISASI\)<a name="ZH-CN_TOPIC_0000001834752537"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_cache_intf.h"`。

开发者手动调用ICachePreLoad接口，能够从指令所在GM地址预加载指令到ICache中。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void ICachePreLoad(const int64_t preFetchLen)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| preFetchLen | 输入 | 预取长度。|



针对如下型号：

- Ascend 950PR&950DT系列产品


- Atlas A3系列产品


- Atlas A2系列产品

preFetchLen参数单位为2K Byte，取值满足如下要求：

$$
\text{preFetchLen}< \dfrac{\text{ICache的大小}}{2K}
$$
 
其中，AIC和AIV的ICache大小分别为32KB和16KB。


针对Atlas推理系列产品AI Core，传入该参数无效，预取长度均为128Byte。
 

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```cpp
int64_t preFetchLen = 2; // 预取指令长度。
AscendC::ICachePreLoad(preFetchLen);
```

- [FlashAttn 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attn_950.md)
- [FusedInferAttentionScore A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/fused_infer_attention_score.md)
- [FusedInferAttentionScore 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/fused_infer_attention_score_950.md)
- [GQA 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/attention/gqa_950.md)
- [MHA 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/attention/mha_950.md)
