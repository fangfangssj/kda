---
type: API
title: Copy
description: 支持UB之间的数据搬运，支持mask操作和DataBlock间隔操作。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/Copy_UBToUB_mask_highdim_split.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# Copy（UBToUB掩码式高维数据搬运）<a name="ZH-CN_TOPIC_0000002575088175"></a>

## 功能说明<a name="zh-cn_topic_0000002567699435_section106841136114319"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

支持Unified Buffer（UB）内部的数据搬运，数据搬运时格式和内容保持不变，支持mask操作和DataBlock间隔操作。

具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- UB -> UB
    - VECIN -> VECCALC
    - VECIN -> VECOUT
    - VECCALC -> VECIN
    - VECCALC -> VECOUT
    - VECOUT -> VECIN
    - VECOUT -> VECCALC

## 函数原型<a name="zh-cn_topic_0000002567699435_section82039854412"></a>

- mask逐bit模式

    ```cpp
    template <typename T, bool isSetMask = true>
    __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
    ```

- mask连续模式 

    ```cpp
    template <typename T, bool isSetMask = true>
    __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask, const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
    ```

## 参数说明<a name="zh-cn_topic_0000002567699435_section16128134420472"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数或者目的操作数的数据类型。支持的数据类型请参考[数据类型](#zh-cn_topic_0000002567699435_section4219135304818)。 |
| isSetMask | 是否在接口内部设置mask。<br>&bull; true，表示在接口内部设置mask。<br>&bull; false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](../mask_operations/set_vector_mask.md)接口设置mask值。这种模式下，接口入参中的mask值设置为占位符`MASK_PLACEHOLDER`，用于占位，无实际含义。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为[LocalTensor](../../data_structures/local_tensor/local_tensor_intro.md)，存储位置为UB，目的地址需要32字节对齐。 |
| src | 输入 | 源操作数，类型为LocalTensor，存储位置为UB，源地址需要32字节对齐。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。详细设置参考[掩码](../simd_compute/mask.md)。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256字节数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。<br>关于该参数的具体描述请参考[高维切分](../simd_compute/high_dim_split.md)。|
| repeatParams | 输入 | 控制操作数地址步长的参数。<br>CopyRepeatParams类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。CopyRepeatParams参数说明请参考表3。<br>具体定义请参考`${INSTALL_DIR}/asc/include/basic_api/kernel_struct_data_copy.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。 |

**表3**  CopyRepeatParams结构体参数定义

| 参数名 | 描述 |
| --- | --- |
| dstStride | 目的操作数同一迭代内相邻DataBlock的地址步长，数据类型为uint16_t，取值范围：dstStride∈[0, 65535]，单位：DataBlock（32B）。 |
| srcStride | 源操作数同一迭代内相邻DataBlock的地址步长，数据类型为uint16_t，取值范围：srcStride∈[0, 65535]，单位：DataBlock（32B） |
| dstRepeatSize | 目的操作数相邻迭代间相同DataBlock的地址步长，数据类型为uint16_t，取值范围：dstRepeatSize∈[0, 4095]，单位：DataBlock（32B）。 |
| srcRepeatSize | 源操作数相邻迭代间相同DataBlock的地址步长，数据类型为uint16_t，取值范围：srcRepeatSize∈[0, 4095]，单位：DataBlock（32B）。 |

## 数据类型<a name="zh-cn_topic_0000002567699435_section4219135304818"></a>

源操作数和目的操作数支持的数据类型保持一致。



- Ascend 950PR&950DT系列产品，支持的数据类型为：int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。





- Atlas A3系列产品，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。





- Atlas A2系列产品，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。





- Atlas 200I/500 A2推理产品，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。





- Kirin X90处理器系列产品，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。





- Kirin 9030处理器系列产品，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。



## 返回值说明

无

## 约束说明<a name="zh-cn_topic_0000002567699435_section2045914466492"></a>

- 位于UB的地址必须32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md#通用地址重叠约束)。
- CopyRepeatParams结构体参数的值需在取值范围内：

    **表4**  CopyRepeatParams结构体参数取值范围

    | 参数名 | 取值范围 |
    | --- | --- |
    | dstStride | [0, 65535] |
    | srcStride | [0, 65535] |
    | dstRepeatSize | [0, 4095] |
    | srcRepeatSize | [0, 4095] |


- 当参数repeatTime取值为0时，该接口的行为如下：
    
  - 针对如下型号，当参数repeatTime取值为0时，不会执行计算操作，不会对目的操作数进行写入，该接口将被视为NOP（空操作）。
     
    - Atlas A3系列产品
     
     
    - Atlas A2系列产品
     
    
    
  - 针对Ascend 950PR&950DT系列产品，该接口通过VF调用[Reg矢量计算](../../reg_vector_compute/index.md)API实现兼容，当参数repeatTime取值为0时，不保证该接口被视为NOP（空操作）。
    


- Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和[通用的Counter模式](../simd_compute/mask.md#接口内设置mask)有一定差异。具体差异如下：
    - 通用的Counter模式：Mask代表**整个矢量计算参与计算的元素个数，迭代次数不生效。**
    - Copy高维切分计算API的Counter模式：Mask代表**每次Repeat中处理的元素个数，迭代次数生效。**
    图1中Copy高维切分计算API的Counter模式时，迭代次数生效，每次迭代参与计算的DataBlock个数可通过Mask计算得出：

    $$
    \text{N}=\text{ceil}(\text{Mask}*\text{sizeof(T)}/\text{datablockSize})
    $$

    **图1**  Copy高维切分计算API的Counter模式  
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/repeat-times-12.png)

## 调用示例<a name="zh-cn_topic_0000002567699435_section088124295117"></a>

```cpp
    if constexpr (scenarioNum == 1) {
        // 场景1：[1, 512]数据搬运，mask连续模式，源和目的空间一样大。
        // mask=64表示每次迭代处理64个int32元素。
        // repeatTime=8表示迭代8次，共处理512个元素。
        // srcStride=1, dstStride=1：同一迭代内DataBlock无间隙。
        // srcRepeatSize=8, dstRepeatSize=8：相邻迭代间步长为8个block（每个block 8个int32元素）。
        uint64_t mask = 64;
        AscendC::Copy(dstLocal, srcLocal, mask, 8, {1, 1, 8, 8});
    } else if constexpr (scenarioNum == 2) {
        // 场景2：从[18, 64]的数据里搬运[18, 8]的数据量。
        // 源数据shape: [18, 64]，共1152个元素。
        // 目的数据shape: [18, 8]，共144个元素。
        // 每次迭代处理一行中的8个元素，共18次迭代。
        // mask=8：每次迭代处理8个元素。
        // repeatTime=18：迭代18次（对应18行）。
        // srcStride=1, dstStride=1：同一迭代内无间隙。
        // srcRepeatSize=8：源每次迭代后跳过64个元素（跳到下一行起始位置）。
        // dstRepeatSize=1：目的每次迭代后跳过8个元素（紧凑排列下一行）。
        uint64_t mask = 8;
        AscendC::Copy(dstLocal, srcLocal, mask, 18, {1, 1, 1, 8});
    } else if constexpr (scenarioNum == 3) {
        // 场景3：从[18, 64]搬运[18, 8]数据，Counter模式。
        // 源数据shape: [18, 64]，共1152个元素。
        // 目的数据shape: [18, 8]，共144个元素。
        // Counter模式下，Mask代表每次Repeat中处理的元素个数。
        // mask=144：每次迭代处理144个元素。
        // repeatTime=1：迭代1次（对应18行）。
        // srcStride=8：源数据两个DataBlock地址步长为8（每次选择每一行的起始DataBlock）。
        // dstStride=1：目的数据连续存放无间隙。
        // srcRepeatSize=8, dstRepeatSize=8：repeatTime=1场景下该参数无效。
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<int32_t, AscendC::MaskMode::COUNTER>(144);
        AscendC::Copy<int32_t, false>(dstLocal, srcLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 8, 8, 8});
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    } else {
        uint64_t mask = 64;
        AscendC::Copy(dstLocal, srcLocal, mask, 8, {1, 1, 8, 8});
    }
```

完整样例请参考[Copy样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/00_data_movement/copy_ub2ub)。
