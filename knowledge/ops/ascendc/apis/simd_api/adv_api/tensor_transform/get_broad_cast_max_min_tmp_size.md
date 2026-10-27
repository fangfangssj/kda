---
type: API
title: GetBroadCastMaxMinTmpSize
description: 在host侧获取Broadcast接口计算所需的最大和最小临时空间大小。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.layout_transform
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/tensor_transform/GetBroadCastMaxMinTmpSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/pad/broadcast_tiling.h
---

# GetBroadCastMaxMinTmpSize

## 功能说明

核函数（Kernel）侧Broadcast接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到核函数（Kernel）侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，核函数（Kernel）侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型

```
void GetBroadCastMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const AscendC::TensorShape& srcShape,const AscendC::TensorShape& dstShape, uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 执行的平台信息 |
| srcShape | 输入 | 输入的shape信息，参数类型为[AscendC::TensorShape](../data_structures/tensor_shape.md)。 |
| dstShape | 输入 | 输出的shape信息，参数类型为[AscendC::TensorShape](../data_structures/tensor_shape.md)。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与核函数（Kernel）侧接口一致。 |
| maxValue | 输出 | Broadcast接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，核函数（Kernel）侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>maxValue仅作为参考值，有可能大于Unified Buffer（UB）剩余空间的大小，该场景下，开发者需要根据UB剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Broadcast接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
// 输入shape信息为(1024,1);算子输入的数据类型为half;不允许修改源操作数
auto platformInfo = context_->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
std::vector<int64_t> srcShapeVec = {1024, 1};
std::vector<int64_t> dstShapeVec = {1024, 16};
AscendC::TensorShape srcShape(srcShapeVec);
AscendC::TensorShape dstShape(dstShapeVec);
uint32_t maxValue{0};
uint32_t minValue{0};
AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShape, dstShape, 2, false, maxValue, minValue);
```

- [MatmulAllReduce 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_950.md)
- [QuantMatmulAllReduce 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce_950.md)

## 补充证据（2026-10-09，安装SDK声明与限定shape的scratch查询）

本节记录一个版本/安装限定的接口差异，不替换上文归档来源。执行者将环境报告为CANN 9.1、目标Ascend950DT_9582；包版本元数据与安装前缀未独立核实，不能据此宣称所有9.1构建一致。适用依据是下列实际声明字节、目标及host-only探针。

### 实际声明与调用类型

归档原型、参数表及示例使用`AscendC::TensorShape`；本次读取的公共`broadcast_tiling.h`声明使用`const ge::Shape& srcShape`和`const ge::Shape& dstShape`。调用时先核实际CANN组件根及已安装声明；不要按知识卡所在目录解析`source_headers`，也不要无条件把归档类型套到另一版安装。此安装的host探针使用`tiling/pad/broadcast_tiling.h`；源码组件定位`include/adv_api/pad/broadcast_tiling.h`与安装host头布局需分别核实。

最小调用片段（`platform`来自该安装的`PlatformAscendCManager::GetInstance("Ascend950DT_9582")`，其非空及链接成功已在本次探针核实）：

```cpp
uint32_t maxValue = 0, minValue = 0;
AscendC::GetBroadCastMaxMinTmpSize(*platform,
    ge::Shape({64, 1}), ge::Shape({64, 64}),
    4, false, maxValue, minValue);
```

### 观测与失效边界

该host-only探针成功运行的原始stdout为`max=0 min=0 UB=253952 SOC=4`。这只说明本次float、64×1→64×64、`isReuseSource=false`及目标平台对象的临时空间需求为0；最大/最小值含义沿用上文。`UB=253952`是总容量，不等于扣除已用buffer后的剩余空间。换shape、dtype、复用标志、接口形式、目标或SDK，应重新查询并结合全部buffer核容量；不能把这组0外推为Broadcast永不需要scratch。本探针未执行NPU性能计时，也不构成加速收益证明。

### 可复核原始来源

- 公共声明及许可（内部材料，不公开）。捕获中还包括弃用包装头，不能把整段捕获SHA冒充单个头SHA。
- 实际头文件SHA记录（内部材料，不公开）：安装host头`include/tiling/pad/broadcast_tiling.h` SHA256为`021cbc401951ba475cf5ef112a4e54df4da0df048c3a3084acd1edef2a7d2b7e`。
- 最小探针源（内部材料，不公开）与成功stdout（内部材料，不公开）。这是本次实测证据，归档主来源及其许可保留；未复制另一套优化模板。
