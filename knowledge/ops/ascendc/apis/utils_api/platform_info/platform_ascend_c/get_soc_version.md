---
type: API
title: GetSocVersion
description: 获取当前硬件平台版本型号。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetSocVersion.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetSocVersion<a name="ZH-CN_TOPIC_0000002078653638"></a>

## 功能说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台版本型号。

## 函数原型<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
SocVersion GetSocVersion(void) const
```

## 参数说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section25791320141317"></a>

当前硬件平台版本型号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下include/tiling/platform/platform\_ascendc.h头文件获取。

AI处理器的型号请通过如下方式获取：


-   针对如下产品：在安装AI处理器的服务器执行`npu-smi info`命令进行查询，获取**Name**信息。实际配置值为AscendName，例如**Name**取值为xxxyy，实际配置值为Ascendxxxyy。

    
    Atlas A2系列产品
    

    
    Atlas 200I/500 A2推理产品
    

    
    Atlas推理系列产品
    

    
    Atlas训练系列产品
    



-   针对Atlas A3系列产品，在安装AI处理器的服务器执行`npu-smi info -t board -i <id> -c <chip_id>`命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为`Chip Name_NPU Name`。例如**Chip Name**取值为Ascendxxx，**NPU Name**取值为1234，实际配置值为Ascendxxx_1234。其中：
    -   `id`：设备id，通过`npu-smi info -l`命令查出的NPU ID即为设备id，命令中表示为`<id>`。
    -   `chip_id`：芯片id，通过`npu-smi info -m`命令查出的Chip ID即为芯片id，命令中表示为`<chip_id>`。



-   针对Ascend 950PR&950DT系列产品，在安装AI处理器的服务器执行`npu-smi info -t board -i <id>`命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为`Chip Name_NPU Name`。例如**Chip Name**取值为Ascendxxx，**NPU Name**取值为1234，实际配置值为Ascendxxx_1234。

    其中，`id`为设备id，通过`npu-smi info -l`命令查出的NPU ID即为设备id，命令中表示为`<id>`。


## 约束说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section19165124931511"></a>

为了满足算子二进制复用，使用Atlas A3系列产品时，调用GetSocVersion接口返回的枚举值与Atlas A2系列产品相同；如果需要区分，可使用接口aclrtGetSocName接口。

## 调用示例<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    // 根据所获得的版本型号自行设计Tiling策略
    // ASCENDXXX请替换为实际的版本型号
    if (socVersion == platform_ascendc::SocVersion::ASCENDXXX) {
        // ...
    }
    return ret;
}
```

### Kernel直调

```cpp
void GetInfoFun()
{
    auto* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return;
    }
    auto socVersion = ascendcPlatform->GetSocVersion();
    // ASCENDXXX请替换为实际的版本型号
    if (socVersion == platform_ascendc::SocVersion::ASCENDXXX) {
        // ...
    }
}
```

- [GroupNormSwish A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/group_norm_swish.md)

## 配套 API 与常见组合

- [PlatformAscendCManager](../platform_ascend_c_manager.md)——Kernel直调场景 GetSocVersion 只能经 PlatformAscendCManager::GetInstance() 取得平台对象后调用
