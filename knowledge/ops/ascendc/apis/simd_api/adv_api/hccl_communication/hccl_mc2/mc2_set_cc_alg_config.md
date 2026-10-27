---
type: API
title: Mc2SetCcAlgConfig
description: 设置MC2通信任务的通信算法。若不调用本接口或传入空字符串，接口内部会通过算法选择器自动选择算法。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2SetCcAlgConfig.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# Mc2SetCcAlgConfig

## 功能说明

设置MC2通信任务的通信算法。若不调用本接口或传入空字符串，接口内部会通过算法选择器自动选择算法。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2SetCcAlgConfig(void* ccArgs, const char* algConfig)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输入 | [Mc2GetCcArgs](mc2_get_cc_args.md)返回的MC2通信参数对象指针。 |
| algConfig | 输入 | 通信算法配置，以'\0'结尾的字符串，最大有效长度为127字节。支持的配置格式和取值请参考[SetAlgConfig](../hccl_tiling/set_alg_config.md)中`algConfig`参数的说明。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败，例如参数为空指针或算法配置长度非法。

## 约束说明

-   算法配置长度必须小于128字节。
-   使用本接口设置的通信算法必须与指定的通信任务类型匹配，具体配置请参考[SetAlgConfig](../hccl_tiling/set_alg_config.md)中`algConfig`参数的说明。

## 调用示例

```cpp
char algConfig[] = "sole[mesh]";
Mc2Result ret = Mc2SetCcAlgConfig(ccArgs, algConfig);
```
