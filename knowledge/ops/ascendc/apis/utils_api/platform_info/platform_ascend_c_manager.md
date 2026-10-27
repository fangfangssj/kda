---
type: API
title: PlatformAscendCManager
description: Kernel直调场景下获取硬件平台信息的单例管理类，通过GetInstance获取PlatformAscendC指针以访问核数等平台信息。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendCManager.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# PlatformAscendCManager<a name="ZH-CN_TOPIC_0000002078498914"></a>

## 功能说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section36583473819"></a>

基于Kernel Launch算子工程，通过核函数（Kernel）直调（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能：通过该类的GetInstance方法可以获取一个PlatformAscendC类的指针，再通过该指针获取硬件平台相关信息，支持获取的信息可参考[PlatformAscendC](platform_ascend_c/index.md)。

> [!CAUTION]注意 
>-   使用该功能需要包含"tiling/platform/platform\_ascendc.h"头文件，并在编译脚本中链接tiling\_api、platform动态库。
>    -   包含头文件的样例如下：
>        ```
>        #include "tiling/platform/platform_ascendc.h"
>        ```
>    -   链接动态库的样例如下:
>        ```
>        add_executable(main main.cpp)
>        target_link_libraries(main PRIVATE
>          kernels
>          tiling_api
>          platform
>        )
>        ```

## 函数原型<a name="zh-cn_topic_0000001796358754_section7979556121414"></a>

```
class PlatformAscendCManager {
public:
    static PlatformAscendC* GetInstance();
    // 在仅有CPU环境、无对应的NPU硬件环境时，需要传入customSocVersion来指定对应的AI处理器型号。注意：因为GetInstance实现属于单例模式，仅在第一次调用时传入的customSocVersion生效。
    static PlatformAscendC* GetInstance(const char *customSocVersion);
private:
...
}
```

## 参数说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section189014013619"></a>

| 参数名 | 描述 |
|:------|:-----|
|customSocVersion|AI处理器型号。 |

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


## 返回值说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section320753512363"></a>

```
GetInfoFun() {
    ...
    auto coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNum();
    ...
    return;
}
```

- [ArgMax 950 Direct Invoke 算子知识](../../../operators/direct/math/math/arg_max_950.md)
- [Sigmoid 950 Direct Invoke 算子知识](../../../operators/direct/nn/activation/sigmoid_950.md)

## 配套 API 与常见组合

- [GetSocVersion](platform_ascend_c/get_soc_version.md)——Kernel直调场景 GetSocVersion 只能经 PlatformAscendCManager::GetInstance() 取得平台对象后调用
