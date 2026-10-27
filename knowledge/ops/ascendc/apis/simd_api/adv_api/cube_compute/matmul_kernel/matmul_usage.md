---
type: API
title: Matmul使用说明
description: Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Matmul使用说明

Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。

Matmul的计算公式为：C = A \* B + Bias，其示意图如下。

-   A、B为源操作数，A为左矩阵，形状为\[M, K\]；B为右矩阵，形状为\[K, N\]。
-   C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N\]。
-   Bias为矩阵乘偏置，形状为\[1, N\]。对A\*B结果矩阵的每一行都采用该Bias进行偏置。

**图1**  Matmul矩阵乘示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/matmul_matrix_diagram.png "Matmul矩阵乘示意图")

> [!NOTE]说明
>下文中提及的M轴方向，即为A矩阵纵向；K轴方向，即为A矩阵横向或B矩阵纵向；N轴方向，即为B矩阵横向；尾轴，即为矩阵最后一个维度。

核函数（Kernel）侧实现Matmul矩阵乘运算的步骤概括为：

1.  创建Matmul对象。
2.  初始化操作。
3.  设置左矩阵A、右矩阵B、Bias。
4.  完成矩阵乘操作。
5.  结束矩阵乘操作。

使用Matmul API实现矩阵乘运算的具体步骤如下：

1.  创建Matmul对象。

    创建Matmul对象的示例如下：

    -   默认为MIX模式（包含矩阵计算和矢量计算），该场景下通常不定义ASCENDC\_CUBE\_ONLY宏，如果在程序中使用了ASCENDC\_CUBE\_ONLY宏，则必须使用ASCEND\_IS\_AIC宏和ASCEND\_IS\_AIV宏将Cube计算和Vector计算隔离开。
    -   纯Cube模式（只有矩阵计算）场景下，建议在代码中定义ASCENDC\_CUBE\_ONLY宏，避免额外的性能开销。

    ```
    // 纯cube模式（只有矩阵计算）场景下，需要设置该代码宏，并且必须在#include "lib/matmul_intf.h"之前设置
    // #define ASCENDC_CUBE_ONLY
    #include "lib/matmul_intf.h"

    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    AscendC::Matmul<aType, bType, cType, biasType> mm;
    ```

    创建对象时需要传入A、B、C、Bias的参数类型信息，类型信息通过[MatmulType](#table1188045714378)来定义，包括：内存逻辑位置、数据格式、数据类型、数据来源的内存逻辑位置。

    ```
    template <
        AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
        LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false, TPosition SRCPOS = TPosition::GM>
    struct MatmulType {
        constexpr static AscendC::TPosition pos = POSITION;
        constexpr static CubeFormat format = FORMAT;
        using T = TYPE;
        constexpr static bool isTrans = ISTRANS;
        constexpr static LayoutMode layout = LAYOUT;
        constexpr static bool ibShare = IBSHARE;
        constexpr static TPosition srcPos = SRCPOS;
    };
    ```

    **表1**  MatmulType参数说明

    <a name="table1188045714378"></a>

    | 参数 | 说明 |
    | --- | --- |
    | POSITION | 内存逻辑位置。 |
    | FORMAT | 数据的物理排布格式。<br>    <br>关于CubeFormat::NZ格式的A矩阵、B矩阵、C矩阵的对齐约束，请参考[NZ格式的矩阵对齐要求](#table98851538118)。 |
    | TYPE | 数据类型。<br>注意：除fp8_e4m3fn_t/fp8_e5m2_t两种数据类型、B矩阵为int8_t数据类型外，A矩阵和B矩阵数据类型需要一致，具体数据类型组合关系请参考[表2](#table1996113269499)。A矩阵和B矩阵为int4b_t数据类型时，矩阵内轴的数据个数必须为偶数。例如，A矩阵为int4b_t数据类型且不转置时，[singleCoreK](../matmul_tiling/t_cube_tiling_struct.md#p11899125875617)必须是偶数。 |
    | ISTRANS | 是否开启支持矩阵转置的功能。<br>    true：开启支持矩阵转置的功能，运行时可以分别通过[SetTensorA](set_tensor_a.md)和[SetTensorB](set_tensor_b.md)中的isTransposeA、isTransposeB参数设置A、B矩阵是否转置。若设置A、B矩阵转置，Matmul会认为A矩阵形状为[K, M]，B矩阵形状为[N, K]。<br>false：默认值，不开启支持矩阵转置的功能，通过[SetTensorA](set_tensor_a.md)和[SetTensorB](set_tensor_b.md)不能设置A、B矩阵的转置情况。Matmul会认为A矩阵形状为[M, K]，B矩阵形状为[K, N]。<br>    <br>注意，由于L1 Buffer上的矩阵数据有分形对齐的约束，A、B矩阵转置和不转置时所需的L1空间可能不相同，在开启支持矩阵转置功能时，必须保证按照[Matmul Tiling参数](../matmul_tiling/t_cube_tiling_struct.md)申请的L1空间不超过L1 Buffer的规格，判断方式为(depthA1*Ceil(baseM/c0Size)*baseK + depthB1*Ceil(baseN/c0Size)*baseK) * db * sizeof(dtype) < L1Size，db表示L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer），其余参数的含义请参考[表1](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)。 |
    | LAYOUT | 表征数据的排布。<br>    <br>NONE：默认值，表示不使用BatchMatmul；其他选项表示使用BatchMatmul。<br>    <br>NORMAL：BMNK的数据排布格式，具体可参考[IterateBatch](iterate_batch.md#功能说明)中对该数据排布的介绍。<br>    <br>BSNGD：原始BSH shape做reshape后的数据排布，具体可参考[IterateBatch](iterate_batch.md#功能说明)中对该数据排布的介绍。<br>    <br>SBNGD：原始SBH shape做reshape后的数据排布，具体可参考[IterateBatch](iterate_batch.md#功能说明)中对该数据排布的介绍。<br>    <br>BNGS1S2：一般为前两种数据排布进行矩阵乘的输出，S1S2数据连续存放，一个S1S2为一个batch的计算数据，具体可参考[IterateBatch](iterate_batch.md#功能说明)中对该数据排布的介绍。 |
    | IBSHARE<a name="p84551411817"></a> | 是否开启IBShare（IntraBlock Share）。IBShare的功能是能够复用L1 Buffer上相同的A矩阵或B矩阵数据，复用的矩阵必须在L1 Buffer上全载。A矩阵和B矩阵仅有一个开启IBShare的场景，与[IBShare模板](matmul_config.md#template-features)配合使用，具体参数设置详见[表2](matmul_config.md#matmulconfig-params)。<br>    <br>注意，A矩阵和B矩阵同时开启IBShare的场景，表示L1 Buffer上的A矩阵和B矩阵同时复用，需要满足：<br>    同一算子中其它Matmul对象的A矩阵和B矩阵也必须同时开启IBShare； |
    | SRC_POSITION |  |

    

    
    **表2**  Matmul输入输出数据类型的支持列表

    <a name="table1996113269499"></a>

    | A矩阵 | B矩阵 | Bias | C矩阵 | 支持平台 |
    | --- | --- | --- | --- | --- |
    | float | float | float/half | float/half/bfloat16_t |  |
    | half | half | float | float |  |
    | half | half | half | float/bfloat16_t |  |
    | int8_t | int8_t | int32_t | int32_t/half |  |
    | int4b_t | int4b_t | int32_t | int32_t/half |  |
    | bfloat16_t | bfloat16_t | float | float |  |
    | bfloat16_t | bfloat16_t | half | float/half |  |
    | half | half | float | int8_t |  |
    | bfloat16_t | bfloat16_t | float | int8_t |  |
    | int8_t | int8_t | int32_t | int8_t |  |
    | half | half | float | half/bfloat16_t |  |
    | half | half | half | half/bfloat16_t |  |
    | bfloat16_t | bfloat16_t | float | bfloat16_t/half |  |
    | half | int8_t | float | float |  |
    | fp8_e4m3fn_t/fp8_e5m2_t | fp8_e4m3fn_t/fp8_e5m2_t | float/half/bfloat16_t | fp8_e4m3fn_t/half/bfloat16_t/float |  |
    | hifloat8_t | hifloat8_t | float/half/bfloat16_t | hifloat8_t/half/bfloat16_t/float |  |
    | float | float | bfloat16_t | float/half/bfloat16_t |  |
    | bfloat16_t | bfloat16_t | bfloat16_t | float/half/bfloat16_t |  |
    | half | half | bfloat16_t | float/half/bfloat16_t |  |
    | int8_t | int8_t | int32_t | bfloat16_t |  |

    

    

2.  初始化操作。
<a name="li5878185413338"></a>

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化matmul对象，参数含义请参考REGIST_MATMUL_OBJ章节
    ```

3.  设置左矩阵A、右矩阵B、Bias。

    ```
    mm.SetTensorA(gm_a);    // 设置左矩阵A
    mm.SetTensorB(gm_b);    // 设置右矩阵B
    mm.SetBias(gm_bias);    // 设置Bias
    ```

    
    Atlas推理系列产品AI Core上需要额外调用[SetLocalWorkspace](set_local_workspace.md)接口设置计算所需的UB空间。

    ```
    mm.SetLocalWorkspace(usedUbBufLen);
    ```
    

4.  完成矩阵乘操作。

    用户可以选择以下三种调用方式之一。

    -   调用[Iterate](iterate.md#li135771283591)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算，方式比较灵活。

        ```
        // API接口内部会进行循环结束条件判断处理
        while (mm.Iterate()) {
            mm.GetTensorC(gm_c);
        }
        ```

    -   调用[IterateAll](iterate_all.md)完成单核上所有数据的计算。IterateAll方式，无需循环迭代，使用比较简单。

        ```
        mm.IterateAll(gm_c);
        ```

    <a id="user-managed-co1"></a>
    
    -   用户申请用于存放矩阵乘结果的L0C Buffer（CO1）内存，调用一次或多次[Iterate](iterate.md#li4843165185812)完成单次或多次迭代计算，在需要搬出计算结果时，调用[Fixpipe](../../../basic_api/cube_compute_isasi/cube_compute_store/fixpipe_l0_c_to_gm.md)接口完成L0C Buffer（CO1）上计算结果的搬运，然后释放申请的L0C Buffer（CO1）内存。该方式下，用户可以灵活控制计算和搬运的节奏，根据实际需要，一次计算对应一次结果的搬出，或者将多次计算结果缓存在L0C Buffer（CO1）内存中，再一次性搬出计算结果。

        在此种调用方式下，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32\_t。

          
        - Ascend 950PR&950DT系列产品暂不支持该方式。
          
          
        - Atlas推理系列产品AI Core暂不支持该方式。
          
          
        - Atlas 200I/500 A2推理产品暂不支持该方式。
          
        

        ```
        // 定义C矩阵的类型信息
        typedef AscendC::MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, float> cType;
        // 创建Matmul对象
        AscendC::Matmul<aType, bType, cType, biasType> mm;

        // 用户提前申请L0C Buffer（CO1）的内存l0cTensor
        TQue<TPosition::CO1, 1> CO1_;
        // 128 * 1024为申请的L0C Buffer（CO1）内存大小
        GetTPipePtr()->InitBuffer(CO1_, 1, 128 * 1024);
        // L0cT为C矩阵的数据类型。
        // A矩阵数据类型是int8_t或int4b_t时，C矩阵的数据类型是int32_t。
        // A矩阵数据类型是half、float或bfloat16_t时，C矩阵的数据类型是float。
        LocalTensor<L0cT> l0cTensor = CO1_.template AllocTensor<L0cT>();

        // 将l0cTensor作为入参传入Iterate，矩阵乘结果输出到用户申请的l0cTensor上
        mm.Iterate(false, l0cTensor);

        // 调用Fixpipe接口将L0C Buffer（CO1）上的计算结果搬运到GM
        FixpipeParamsV220 params;
        params.nSize = nSize;
        params.mSize = mSize;
        params.srcStride = srcStride;
        params.dstStride = dstStride;
        CO1_.EnQue(l0cTensor);
        CO1_.template DeQue<L0cT>();
        Fixpipe<cType, L0cT, CFG_ROW_MAJOR>(gm[dstOffset], l0cTensor, params);

        // 释放L0C Buffer（CO1）内存
        CO1_.FreeTensor(l0cTensor);
        ```

5.  结束矩阵乘操作。

    ```
    mm.End();
    ```

<a name="table98851538118"></a>
CubeFormat::NZ格式的矩阵对齐要求为：

- A矩阵/B矩阵：外轴为16的倍数，内轴为C0_size的倍数。
- C矩阵：
  
  - 对于如下产品型号：
    
    - Atlas推理系列产品AI Core
    
    
    - Atlas 200I/500 A2推理产品
    
    外轴为16的倍数，内轴为16的倍数。
  
  
  - 对于如下产品型号：
    
    - Ascend 950PR&950DT系列产品
    
    
    - Atlas A3系列产品
    
    
    - Atlas A2系列产品
    
    开启channel_split功能：外轴为16的倍数，内轴为C0_size的倍数。

    不开启channel_split功能：外轴为16的倍数，对于float/int32_t类型内轴为16的倍数，对于half/bfloat16_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t类型内轴为C0_size的倍数。
  
  

上述参数和概念的补充说明如下：

-   float/int32\_t数据类型的C0\_size为8，half/bfloat16\_t数据类型的C0\_size为16，int8\_t/fp8\_e4m3fn\_t/fp8\_e5m2\_t/hifloat8\_t数据类型的C0\_size为32，int4b\_t/fp4x2\_e2m1\_t/fp4x2\_e1m2\_t数据类型的C0\_size为64。
-   channel\_split功能通过[MatmulConfig](matmul_config.md#matmulconfig-params)中的isEnableChannelSplit参数配置，具体内容请参考[MatmulConfig](matmul_config.md#matmulconfig-params)。

## 需要包含的头文件

```
#include "lib/matmul/matmul_intf.h"
```

## 实现原理

以输入矩阵A \(GM, ND, half\)、矩阵B\(GM, ND, half\)，输出矩阵C \(GM, ND, float\)，无Bias场景为例，其中\(GM, ND, half\)表示数据存放在GM上，数据格式为ND，数据类型为half，描述Matmul高阶API典型场景的内部算法框图，如下图所示。

**图2**  Matmul算法框图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/matmul_algorithm_diagram.png "Matmul算法框图")

计算过程分为如下几步：

1.  数据从GM搬到L1 Buffer（A1）：DataCopy每次从矩阵A中搬出一个stepM\*baseM\*stepKa\*baseK的矩阵块a1，循环多次完成矩阵A的搬运；数据从GM搬到L1 Buffer（B1）：DataCopy每次从矩阵B中搬出一个stepKb\*baseK\*stepN\*baseN的矩阵块b1，循环多次完成矩阵B的搬运；
2.  数据从L1 Buffer（A1）搬到L0A Buffer（A2）：LoadData每次从矩阵块a1，搬出一个baseM \* baseK的矩阵块a0；数据从L1 Buffer（B1）搬到L0B Buffer（B2），并完成转置：LoadData每次从矩阵块b1，搬出一个baseK \* baseN的矩阵块，并将其转置为baseN \* baseK的矩阵块b0；
3.  矩阵乘：每次完成一个矩阵块a0 \* b0的计算，得到baseM \* baseN的矩阵块co1；
4.  数据从矩阵块co1搬到矩阵块co2:DataCopy每次搬运一块baseM \* baseN的矩阵块co1到singleCoreM \* singleCoreN的矩阵块co2中；
5.  重复2-4步骤，完成矩阵块a1 \* b1的计算；
6.  数据从矩阵块co2搬到矩阵块C：DataCopy每次搬运一块singleCoreM \* singleCoreN的矩阵块co2到矩阵块C中；
7.  重复1-6步骤，完成矩阵A \* B = C的计算。

## 配套 API 与常见组合

- [MatmulConfig](matmul_config.md)
- [REGIST_MATMUL_OBJ](regist_matmul_obj.md)
- [GetMatmulApiTiling](get_matmul_api_tiling.md)
- [Matmul Tiling类使用说明](../matmul_tiling/matmul_tiling_usage.md)
- [Matmul 高阶 API 的执行模式与 Host 契约](../../../../../guides/matmul/execution_mode_and_host_contracts.md)
- [异步 Matmul 的结果空间与完成等待](../../../../../guides/matmul/async_result_buffers_and_completion.md)
- [Matmul 基本块流量模型与输出对齐](../../../../../guides/matmul/base_block_traffic_model_and_alignment.md)
