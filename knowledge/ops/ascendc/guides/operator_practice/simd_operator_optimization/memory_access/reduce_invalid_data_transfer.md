---
type: Guide
title: 非对齐场景减少无效数据的搬运
description: '>该性能优化建议适用于如下型号：'
tags:
- task.lookup
- topic.performance.optimization.optimization_overview
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/reduce_invalid_data_transfer.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 非对齐场景减少无效数据的搬运<a name="ZH-CN_TOPIC_0000002497818522"></a>

【优先级】中

>[!NOTE]说明 
>该性能优化建议适用于如下型号：
>
>- Ascend 950PR&950DT系列产品
>

【描述】在非对齐数据搬运场景中，Ascend 950PR&950DT系列产品在基础API层面提供了DataCopyPad接口，该接口支持Normal、Compact（紧凑）两种搬运模式。搬运多块非32B对齐数据块的场景下，使用Compact模式在可以减少搬运的无效数据量，节省带宽。

假设需要搬运三个数据块，每块数据块大小为48B，数据类型为float类型。除了这三个48字节的数据块之外，其他所有数据均为无效数据。

【反例】使用DataCopyPad接口进行Normal模式搬运数据

```
__aicore__ inline void CopyIn(){
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::Duplicate<T>(xLocal, 0, count);
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 3;
    dataCopyParams.blockLen = 48;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    AscendC::DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = 1;
    dataCopyPadParams.leftPadding = 0;
    dataCopyPadParams.rightPadding = 4;
    dataCopyPadParams.paddingValue = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
    inQueueX.EnQue<T>(xLocal);
}
```

搬运后Unified Buffer（UB）内数据如下：

```
[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.....]
```

**图1**  Normal模式搬运<a name="fig1316440112511"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/norm_copy.png "Normal模式搬运")

如图所示，由于每块数据块为48B，非32B对齐，因此搬运每块数据块时需要插入16B大小的padding数据使得数据32B对齐，最终搬运192B大小的数据到UB，其中包含48B的无效数据。

【正例】改用Compact模式搬运进行优化

```
__aicore__ inline void CopyIn(){
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::Duplicate<T>(xLocal, 0, count);
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 3;
    dataCopyParams.blockLen = 48;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    AscendC::DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = 1;
    dataCopyPadParams.leftPadding = 0;
    dataCopyPadParams.rightPadding = 4;
    dataCopyPadParams.paddingValue = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
    inQueueX.EnQue<T>(xLocal);
}
```

搬运后UB内数据如下：

```
[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 0., 0., 0., 0....]
```

**图2**  Compact模式搬运<a name="fig203951741467"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/compact_copy.png "Compact模式搬运")

根据Compact模式搬运的示意图，最终搬运了160B大小的数据，其中包含16B的无效数据。

【总结】通过比较可以发现，搬运多块非32B对齐数据块的场景下，使用Compact模式在可以减少搬运的无效数据量，节省带宽。
