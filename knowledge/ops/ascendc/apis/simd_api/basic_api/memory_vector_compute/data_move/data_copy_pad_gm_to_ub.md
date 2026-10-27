---
type: API
title: DataCopyPad
description: 将数据从GM非对齐搬运至UB，可根据需要自行填充数据。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# DataCopyPad（GMToUB非对齐数据搬运）<a name="ZH-CN_TOPIC_0000001894460401"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

该接口提供将数据从Global Memory非对齐搬运至Unified Buffer（UB）的功能，可以根据开发者的需要自行填充数据。

当每个搬运的数据块长度（blockLen）非32字节对齐时，每一个数据块都需要填充数据至32字节对齐。


特别地，针对Ascend 950PR&950DT系列产品，支持Compact模式，该模式支持单个数据块非32字节对齐，将所有数据块合并成一个连续数据块，在该数据块右侧填充数据至32字节对齐。


具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- Global Memory -> UB
    - GM -> VECIN
    - GM -> VECOUT
    
    - GM -> VECCALC（仅Ascend 950PR&950DT系列产品支持）
    

填充的数据有两种配置方式：

- 配置结构体[DataCopyPadExtParams](#table_gm2ub_pad_5)的参数paddingValue，此时DataCopyPadExtParams的参数isPad需要设置为true。
- 使用寄存器配置接口[SetPadValue](../data_move_aux_config/set_pad_value_isasi.md)在接口外部设置需要填充的数据，此时DataCopyPadExtParams的参数isPad需要设置为false。

## 函数原型<a name="section620mcpsimp"></a>

- 不支持配置设置数据搬运模式mode

    ```cpp
    template <typename T>
    __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams)
    ```



- 支持配置设置数据搬运模式mode（仅Ascend 950PR&950DT系列产品支持）

    ```cpp
    // 该函数原型仅支持Ascend 950PR&950DT系列产品
    template <typename T, PaddingMode mode = PaddingMode::Normal>
    __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams)
    ```



## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| T | 操作数以及paddingValue（待填充数据值）的数据类型。各产品支持的数据类型请参考[数据类型](#section4219135304818)。 |
| mode | 配置数据搬运模式。PaddingMode类型，定义如下：<br>enum class PaddingMode : uint8_t {<br>    Normal = 0,  // 默认模式，与原有数据搬运格式保持一致，每次数据搬运都会补齐至32字节对齐<br>    Compact,     // 紧凑模式，允许单次搬运不对齐，统一在整块数据末尾补齐至32字节对齐<br>}; |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| dst | 输出 | 目的操作数，类型为[LocalTensor](../../data_structures/local_tensor/local_tensor_intro.md)，存储位置为UB，目的地址需要32字节对齐。 |
| src | 输入 | 源操作数，类型为[GlobalTensor](../../data_structures/global_tensor/global_tensor_intro.md)，存储位置为Global Memory，源地址需要1字节对齐。 |
| dataCopyParams | 输入 | 搬运参数。DataCopyExtParams类型，具体参数说明见下文。 |
| padParams | 输入 | 从Global Memory搬运数据至Local Memory时，可以根据开发者需要，在搬运数据左边或右边填充数据。padParams是用于控制数据填充过程的参数。DataCopyPadExtParams类型，具体参数请参考[表5](#table_gm2ub_pad_5)。 |

下文表格中列出的结构体参数定义请参考`${INSTALL_DIR}/asc/include/basic_api/kernel_struct_data_copy.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。



针对Ascend 950PR&950DT系列产品，DataCopyExtParams结构体参数定义请参考[表3](#table_gm2ub_pad_3)。

**表3**  DataCopyExtParams结构体参数定义<a name="table_gm2ub_pad_3"></a>

| 参数名 | 描述 |
| :--- | :--- |
| blockCount | 搬运的数据块个数，数据类型为uint16_t，取值范围：blockCount∈[0, 65535]，默认值为1。 |
| blockLen | 搬运的每个数据块长度，数据类型为uint32_t，取值范围：blockLen∈[0, $2^{21}−1$]，单位：1B。<br>**注：blockLen必须是sizeof(T)的整数倍，需要注意不要超过UB空间大小。** |
| srcStride | 源操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为int64_t，取值范围：srcStride∈[-blockLen, $2^{40}−1$]，单位：1B。当srcStride = -blockLen时，表示每次传输的连续数据块均为同一块，即重复搬运第一个数据块。 |
| dstStride | 目的操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为int64_t，取值范围：dstStride∈[0, 65535]，单位：dataBlock（32B）。若PaddingMode为Compact模式，该参数无效，默认值为0，取默认值即可。<br>**注：需要注意不要超过UB空间大小**。 |
| rsv | 预留参数，当前需设置为0。 |





针对如下型号，DataCopyExtParams结构体参数定义请参考[表4](#table_gm2ub_pad_4)：


- Atlas A3系列产品


- Atlas A2系列产品


- Atlas 200I/500 A2推理产品


**表4**  DataCopyExtParams结构体参数定义<a name="table_gm2ub_pad_4"></a>

| 参数名 | 描述 |
| :--- | :--- |
| blockCount | 搬运的数据块个数，数据类型为uint16_t，取值范围：blockCount∈[0, 4095]，默认值为1。 |
| blockLen | 搬运的每个数据块长度，数据类型为uint32_t，取值范围：blockLen∈[0, $2^{21}−1$]，单位：1B。<br>**注：blockLen必须是sizeof(T)的整数倍，需要注意不要超过UB空间大小。** |
| srcStride | 源操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为uint32_t，取值范围：srcStride∈[0, $2^{32}−1$]，单位：1B。 |
| dstStride | 目的操作数相邻数据块之间的间隔（即前一个数据块**结束地址**与后一个数据块**起始地址**的差值），数据类型为uint32_t，取值范围：dstStride∈[0, $2^{32}−1$]，单位：dataBlock（32B）。若PaddingMode为Compact模式，该参数无效，默认值为0，取默认值即可。<br>**注：需要注意不要超过UB空间大小**。 |
| rsv | 预留参数，当前需设置为0。 |



**表5**  DataCopyPadExtParams&lt;T&gt;结构体参数定义<a name="table_gm2ub_pad_5"></a>

| 参数名 | 描述 |
| :--- | :--- |
| isPad | 是否通过DataCopyPadExtParams结构体中的paddingValue字段设置填充值。<br>true：使用paddingValue字段作为填充值。<br>false：不使用paddingValue字段，根据是否调用寄存器配置接口[SetPadValue](../data_move_aux_config/set_pad_value_isasi.md)分为两种情况。<br>&bull;调用SetPadValue时，在接口外部配置需要填充的数据。<br>&bull;不调用SetPadValue时，硬件自动在**每个数据块右侧**填充dummy数据至32字节对齐。若PaddingMode为Compact模式，则只会在**最后一个数据块右侧**填充dummy数据至32字节对齐。 |
| leftPadding | 连续搬运数据块左侧需要补充的数据范围，单位为元素个数。<br>若PaddingMode为Compact模式，该参数无效，只会在**最后一个数据块右侧**填充数据。<br>**leftPadding、rightPadding所占的字节数均不能超过32字节。** |
| rightPadding | 连续搬运数据块右侧需要补充的数据范围，单位为元素个数。<br>若PaddingMode为Compact模式，该参数无效，只会在**最后一个数据块右侧**填充数据。<br>**leftPadding、rightPadding所占的字节数均不能超过32字节。** |
| paddingValue | 左右两侧需要填充的数据值，需要保证在数据占用字节范围内。<br>数据类型和源操作数保持一致，数据类型为模板参数T。<br>**当数据类型长度为64位时，该参数只能设置为0。** |

- GM -> VECIN/VECOUT配置示例<a name="li73127579197"></a>

    下面通过三个示例展示DataCopyPad在GM -> VECIN/VECOUT通路上的不同配置场景。示例1演示32字节对齐场景下的数据搬运，示例2演示非32字节对齐时的dummy填充机制，示例3（仅Ascend 950PR&950DT系列产品支持）演示负stride实现重复搬运同一数据块的场景。

    - 参数解释
        - 当blockLen+leftPadding+rightPadding满足32字节对齐时，若isPad为false，左右两侧填充的数据值会默认为随机值；否则为paddingValue。
        - 当blockLen+leftPadding+rightPadding不满足32字节对齐时，框架会填充一些假数据dummy，保证左右填充的数据和blockLen、假数据为32字节对齐。若leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值；若leftPadding/rightPadding不为0：isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

    - 示例1：32字节对齐场景

        blockLen为64，每个连续传输数据块包含64字节；srcStride为1，因为源操作数的逻辑位置为GM，srcStride的单位为字节，即源操作数中前一个数据块**结束地址**与后一个数据块**起始地址**的差值为1字节；dstStride为1，因为目的操作数的逻辑位置为VECIN/VECOUT，dstStride的单位为DataBlock数量（每DataBlock为32字节），即目的操作数相邻数据块之间间隔1个dataBlock。

        blockLen+leftPadding+rightPadding满足32字节对齐，isPad为false，左右两侧填充的数据值会默认为随机值；否则为paddingValue。此处示例中，leftPadding、rightPadding均为0，则不填充。

        blockLen+leftPadding+rightPadding不满足32字节对齐时，框架会填充一些假数据dummy，保证左右填充的数据和blockLen、假数据为32字节对齐。leftPadding/rightPadding不为0：若isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/datacopypad1.png)

    - 示例2：非32字节对齐场景

        blockLen为47，每个连续传输数据块包含47字节；srcStride为1，表示源操作数中前一个数据块**结束地址**与后一个数据块**起始地址**的差值为1字节；dstStride为1，表示目的操作数相邻数据块之间间隔1个dataBlock。

        blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding、rightPadding均为0：dummy会默认填充待搬运数据块的第一个元素值。

        blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding/rightPadding不为0：若isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/datacopypad2.png)

    

    - 示例3：负stride重复搬运场景（仅Ascend 950PR&950DT系列产品支持）

        blockLen为48，每个连续传输数据块包含48字节；srcStride为-48，表示源操作数中前一个数据块**结束地址**与后一个数据块**起始地址**的差值为-48字节（一个连续数据块长度为48字节），相当于每次传输的连续数据块都是同一块；dstStride为1，表示目的操作数相邻数据块之间间隔1个dataBlock。

        blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding、rightPadding均为0：dummy会默认填充待搬运数据块的第一个元素值。

        blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding/rightPadding不为0：若isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/datacopypad.png)

    

- 搬运模式的配置示例（仅Ascend 950PR&950DT系列产品支持）

    下面通过两个示例展示Ascend 950PR&950DT系列产品上Normal和Compact两种搬运模式的区别。

    

    - 示例1：Normal模式

        blockLen为48，每个连续传输数据块包含48字节；srcStride为0，源操作数的逻辑位置为GM，srcStride的单位为字节，即源操作数相邻数据块之间紧密排列；dstStride为0，目的操作数的逻辑位置为VECIN/VECOUT，dstStride的单位为DataBlock（32字节），目的操作数相邻数据块之间无间隔，注意数据块包含leftPadding/rightPadding数据。

        blockLen + leftPadding + rightPadding满足32字节对齐，isPad为false，左右两侧填充的数据值会默认为随机值，否则为paddingValue。此处示例中，leftPadding为0，rightPadding为16，每个连续传输数据块都会在右侧填充16字节。目的操作数的总长度为192字节。

    - 示例2：Compact模式

        blockLen为48，每个连续传输数据块包含48字节；srcStride为0，源操作数的逻辑位置为GM，srcStride的单位为字节，即源操作数相邻数据块之间紧密排列；dstStride为0，目的操作数的逻辑位置为VECIN/VECOUT，dstStride的单位为DataBlock（32字节），目的操作数相邻数据块之间紧密排列，不会填充数据。

        blockLen * blockCount + leftPadding + rightPadding满足32字节对齐，isPad为false，左右两侧填充的数据值会默认为随机值，否则为paddingValue。此处示例中，leftPadding为0，rightPadding为16，在最后一个数据块右侧填充16字节。目的操作数的总长度为160字节。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/paddingMode.png)

    

## 数据类型<a name="section4219135304818"></a>



- Ascend 950PR&950DT系列产品，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t、double、complex64。





- Atlas A3系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。





- Atlas A2系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。





- Atlas 200I/500 A2推理产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。



## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 结构体DataCopyPadExtParams的参数取值范围需要考虑不要超过UB空间大小。

- 当DataCopyExtParams结构体参数blockCount、blockLen任意一个值为0时，该接口将被视为NOP（空操作）。该说明针对如下型号生效：
  
  - Atlas A3系列产品
  
  
  - Atlas A2系列产品
  
  
  - Ascend 950PR&950DT系列产品
  



- 针对Ascend 950PR&950DT系列产品，若PaddingMode为Compact模式，则参数dstStride、leftPadding、rightPadding无效。



- 位于Global Memory的源地址必须1字节对齐，位于UB的目的地址必须32字节对齐。
- leftPadding、rightPadding所占字节数均不能超过32B。
- blockLen必须是sizeof\(T\)的整数倍。
- 结构体DataCopyPadExtParams的参数paddingValue数据类型和源操作数保持一致。当数据类型为b64时，paddingValue只能设置为0。

- 针对Ascend 950PR&950DT系列产品，DataCopyExtParams结构体参数的数据类型和取值范围如下：

    **表6**  DataCopyExtParams结构体参数的数据类型和取值范围

    | 参数名 | 数据类型 | 取值范围 |
    | --- | --- | --- |
    | blockCount | uint16_t | [0, 65535] |
    | blockLen | uint32_t | [0, $2^{21}−1$] |
    | srcStride | int64_t | [-blockLen, $2^{40}−1$] |
    | dstStride | int64_t | [0, 65535] |



- 针对如下型号，DataCopyExtParams结构体参数的数据类型和取值范围如下：
  
  - Atlas A3系列产品
  
  
  - Atlas A2系列产品
  
  
  - Atlas 200I/500 A2推理产品
  

    **表7**  DataCopyExtParams结构体参数的数据类型和取值范围

    | 参数名 | 数据类型 | 取值范围 |
    | --- | --- | --- |
    | blockCount | uint16_t | [0, 4095] |
    | blockLen | uint32_t | [0, $2^{21}−1$] |
    | srcStride | uint32_t | [0, $2^{32}−1$] |
    | dstStride | uint32_t | [0, $2^{32}−1$] |



## 调用示例<a name="section177231425115410"></a>

- 场景1：使用SetPadValue进行自定义填充

    ```cpp
    // DataCopyExtParams参数：blockCount=1, blockLen=20*sizeof(half), srcStride=0, dstStride=0, rsv=0
    // DataCopyPadExtParams参数：isPad=false, leftPadding=0, rightPadding=12
    // SetPadValue设置填充值为1，配合isPad=false使用。
    AscendC::DataCopyExtParams copyParams{1, srcCols * sizeof(T), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = dstCols - srcCols;

    AscendC::SetPadValue((T)1);
    AscendC::DataCopyPad(srcLocal, srcGlobal, copyParams, padParams);
    ```

- 场景2：使用rightPadding进行默认填充

    ```cpp
    // DataCopyExtParams参数：blockCount=32, blockLen=59*sizeof(float), srcStride=0, dstStride=0, rsv=0
    // DataCopyPadExtParams参数：isPad=true, leftPadding=0, rightPadding=5
    // isPad=true时填充值为0，不使用SetPadValue。
    AscendC::DataCopyExtParams copyParams{srcRows, srcCols * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = dstCols - srcCols;

    AscendC::DataCopyPad(srcLocal, srcGlobal, copyParams, padParams);
    ```

- 场景3：使用Compact模式进行紧凑填充（仅支持Ascend 950PR&950DT系列产品）

    ```cpp
    // DataCopyExtParams参数：blockCount=3, blockLen=24*sizeof(half), srcStride=0, dstStride=0, rsv=0
    // DataCopyPadExtParams参数：isPad=true, leftPadding=0, rightPadding=16
    // PaddingMode::Compact：将多行数据紧凑排列到一行，填充区域紧凑放置在数据之后。
    // 输入[3, 24]紧凑排列为[1, 80]，前72个元素为3行数据，后8个元素为填充区域。
    AscendC::DataCopyExtParams copyParams{srcRows, srcCols * sizeof(T), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = dstCols - srcCols * srcRows;

    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(srcLocal, srcGlobal, copyParams, padParams);
    ```

- 场景4：使用SetLoopModePara使能loop mode进行数据搬运（仅支持Ascend 950PR&950DT系列产品）

    ```cpp
    // Compact模式：每次内层循环搬运80B后填充16B使其96字节对齐。
    // padding值设为-1
    AscendC::DataCopyExtParams copyParams{BLOCK_COUNT, BLOCK_LEN, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, -1};

    AscendC::LoopModeParams loopParam2Ub {LOOP1_SIZE, LOOP2_SIZE, LOOP1_SRC_STRIDE, LOOP1_DST_STRIDE, LOOP2_SRC_STRIDE, LOOP2_DST_STRIDE};
    AscendC::SetLoopModePara(loopParam2Ub, AscendC::DataCopyMVType::OUT_TO_UB);
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(srcLocal, srcGlobal, copyParams, padParams);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    ```

- 场景5：使用SetLoopModePara使能loop mode进行数据搬运（仅支持Ascend 950PR&950DT系列产品）

    ```cpp
    // Normal模式：每个block搬运40B后填充24B使其64字节对齐。
    // padding值设为-1
    AscendC::DataCopyExtParams copyParams{BLOCK_COUNT, BLOCK_LEN, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, -1};

    AscendC::LoopModeParams loopParam2Ub {LOOP1_SIZE, LOOP2_SIZE, LOOP1_SRC_STRIDE, LOOP1_DST_STRIDE, LOOP2_SRC_STRIDE, LOOP2_DST_STRIDE};
    AscendC::SetLoopModePara(loopParam2Ub, AscendC::DataCopyMVType::OUT_TO_UB);
    AscendC::DataCopyPad<int8_t>(srcLocal, srcGlobal, copyParams, padParams);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    ```

- 场景6：五维数据搬运（仅支持Ascend 950PR&950DT系列产品）

    ```cpp
    // Global Memory[2, 4, 3, 128, 126]int8 -> UB[512, 128]int8
    // Normal模式，使用loop mode
    // 搬运规格：[2, 2, 2, 64, 126]，每个126字节补2字节padding到128字节。
    // 最终UB连续存放为[512, 128]。

    // Stride参数说明：
    // - loop1SrcStride=128*126：第2维stride，每次loop1源地址跳过128*126字节。
    // - loop1DstStride=64*128：每次loop1目的地址跳过64*128字节（64行×128字节/行）。
    // - loop2SrcStride=3*128*126：第1维stride，每次loop2源地址跳过3*128*126字节。
    // - loop2DstStride=2*64*128：每次loop2目的地址跳过2*64*128字节。

    AscendC::LoopModeParams loopParam2Ub {
        2,  // loop1Size=2（第2维搬运2次）
        2,  // loop2Size=2（第1维搬运2次）
        128 * 126,  // loop1SrcStride（第2维stride）
        64 * 128,   // loop1DstStride
        3 * 128 * 126,  // loop2SrcStride（第1维stride）
        2 * 64 * 128    // loop2DstStride
    };
    AscendC::SetLoopModePara(loopParam2Ub, AscendC::DataCopyMVType::OUT_TO_UB);

    AscendC::DataCopyExtParams copyParams{
        64,  // blockCount=64（第3维搬运64次）
        126,  // blockLen=126（第4维搬运126字节，padding到128）
        0, 0, 0  // srcStride=0, dstStride=0, rsv=0
    };
    AscendC::DataCopyPadExtParams<int8_t> padParams{true, 0, 0, 0};

    // 第0维for循环搬运2次。
    for (uint32_t dim0 = 0; dim0 < 2; dim0++) {
        constexpr uint32_t dim0SrcStride = 4 * 3 * 128 * 126;  // 第0维源stride
        constexpr uint32_t dim0DstStride = 2 * 2 * 64 * 128;   // 第0维目的stride
        uint32_t srcOffset = dim0 * dim0SrcStride;
        uint32_t dstOffset = dim0 * dim0DstStride;

        AscendC::DataCopyPad<int8_t>(srcLocal[dstOffset], srcGlobal[srcOffset], copyParams, padParams);
    }

    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    ```

完整样例请参考[DataCopyPad样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_pad_gm2ub_ub2gm)。

## 配套 API 与常见组合

- [GmUbDataMove](../../data_move_guide/gm_ub_data_move.md)
- [DataCopy](data_copy_gm_and_ub_continuous.md)
- [DataCopyPad](data_copy_pad_ub_to_gm.md)
- [DataCopyPad 多行搬运的有效长度、padding 与物理行距](../../../../../guides/data_movement/datacopypad_rows_and_padding.md)
- [DataCopyPad 的版本、重载与尾部真实范围](../../../../../guides/data_movement/datacopypad_versions_overloads_and_tail_extent.md)
- [对齐单位、布局行距与填充值](../../../../../guides/memory/alignment_units_layout_stride_and_padding_values.md)
