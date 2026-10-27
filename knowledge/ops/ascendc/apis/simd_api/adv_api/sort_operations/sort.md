---
type: API
title: Sort
description: 排序函数，按照数值大小进行降序排序，排序结果以score和index结构存储。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.ordering
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/sort_operations/Sort.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/adv_api/sort/sort.h
---

# Sort

## 功能说明

-   对应不带模板参数SortConfig的函数原型

    排序函数，按照数值大小进行降序排序。排序后的数据按照如下排布方式进行保存：

    
    Ascend 950PR&950DT系列产品采用方式一。
    

    
    Atlas A3系列产品采用方式一。
    

    
    Atlas A2系列产品采用方式一。
    

    
    Atlas推理系列产品AI Core采用方式二。
    

    
    Kirin X90处理器系列产品采用方式一。
    

    
    Kirin 9030处理器系列产品采用方式一。
    

-   排布方式一：

    一次迭代可以完成32个数的排序，排序好的score与其对应的index一起以（score, index）的结构存储在dst中。不论score为half还是float类型，dst中的（score, index）结构总是占据8Bytes空间。如下所示：

    -   当score为float，index为uint32类型时，计算结果中index存储在高4Bytes，score存储在低4Bytes。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000002007005918.png)

    -   当score为half，index为uint32类型时，计算结果中index存储在高4Bytes，score存储在低2Bytes，中间的2Bytes保留。

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000002042886661.png)

-   排布方式二：Region Proposal排布

    输入输出数据均为Region Proposal，一次迭代可以完成16个region proposal的排序。每个Region Proposal占用连续8个half/float类型的元素，约定其格式：

    ```
    [x1, y1, x2, y2, score, label, reserved_0, reserved_1]
    ```

    对于数据类型half，每一个Region Proposal占16Bytes，Byte\[15:12\]是无效数据，Byte\[11:0\]包含6个half类型的元素，其中Byte\[11:10\]定义为label，Byte\[9:8\]定义为score，Byte\[7:6\]定义为y2，Byte\[5:4\]定义为x2，Byte\[3:2\]定义为y1，Byte\[1:0\]定义为x1。

    如下图所示，总共包含16个Region Proposals。

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000002043918785.png)

    对于数据类型float，每一个Region Proposal占32Bytes，Byte\[31:24\]是无效数据，Byte\[23:0\]包含6个float类型的元素，其中Byte\[23:20\]定义为label，Byte\[19:16\]定义为score，Byte\[15:12\]定义为y2，Byte\[11:8\]定义为x2，Byte\[7:4\]定义为y1，Byte\[3:0\]定义为x1。

    如下图所示，总共包含16个Region Proposals。

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/image_0000002043797849.png)

-   对应带模板参数SortConfig的函数原型

    
    仅在Ascend 950PR&950DT系列产品上支持。
    

    根据模板参数SortConfig，按其中指定的排序算法，对输入数据排序，排序结果可以指定升序或降序排序。

    当函数原型带有输出索引dstIndexTensor参数，需要输出排序结果数据分别对应的索引；若输入带有索引srcIndexTensor参数，则输出索引即为原输入的索引，若输入不带有索引，则对输入数据从0开始生成所需排序数量的索引，最终输出索引即为对应输入数据的索引。如下两幅图，分别为输入带有索引和输入不带索引的数据排序示意图。

    **图1**  输入带有索引srcIndex的排序样例  
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/sort_with_srcindex.png "输入带有索引srcIndex的排序样例")

    **图2**  输入不带索引srcIndex的排序样例  
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/sort_without_srcindex.png "输入不带索引srcIndex的排序样例")

## 函数原型

-   不带SortConfig

    ```
    template <typename T, bool isFullSort>
    __aicore__ inline void Sort(const LocalTensor<T>& dst, const LocalTensor<T>& concat, const LocalTensor<uint32_t>& index, LocalTensor<T>& tmp, const int32_t repeatTime)
    ```

-   带SortConfig

    
    仅在Ascend 950PR&950DT系列产品上支持。
    

    -   接口框架申请临时空间
        -   不带srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
            ```

        -   不带srcIndexTensor参数，带有dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<uint32_t>& dstIndexTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
            ```

        -   带有srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, typename U, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(const LocalTensor<T>& dstTensor, const LocalTensor<U>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<U>& srcIndexTensor, const uint32_t calCount)
            ```

    -   通过sharedTmpBuffer入参传入临时空间
        -   不带srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

        -   不带srcIndexTensor参数，带有dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<uint32_t>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

        -   带有srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, typename U, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<U>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<U>& srcIndexTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

    由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

    -   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
    -   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

    通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetSortMaxMinTmpSize](get_sort_max_min_tmp_size.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明

-   对应不带SortConfig的函数原型

    **表1**  模板参数说明

    | 参数名 | 含义 |
    | --- | --- |
    | T | 操作数的数据类型。支持的数据类型为：half、float。 |
    | isFullSort | 是否开启全排序模式。全排序模式指将全部输入降序排序，非全排序模式下，排序方式请参考表2中的repeatTime说明。 |

    **表2**  参数说明

    | 参数名称 | 输入/输出 | 含义 |
    | --- | --- | --- |
    | dst | 输出 | 目的操作数，shape为[2n]。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。 |
    | concat | 输入 | 源操作数，即接口功能说明中的score，shape为[n]。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。<br>    <br>此源操作数的数据类型需要与目的操作数保持一致。 |
    | index | 输入 | 源操作数，shape为[n]。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。<br>    <br>此源操作数固定为uint32_t数据类型。 |
    | tmp | 输入 | 临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供，临时空间大小BufferSize的获取方式请参考[GetSortTmpSize](get_sort_tmp_size.md)。数据类型与源操作数保持一致。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。 |
    | repeatTime | 输入 | 重复迭代次数，int32_t类型。 |

-   对应带SortConfig的函数原型

    **表3**  模板参数说明

    | 接口 | 功能 |
    | --- | --- |
    | T | 操作数srcTensor和dstTensor的数据类型。RADIX_SORT排序算法支持的数据类型为：uint8_t、int8_t、uint16_t、int16_t、uint32_t、int32_t、half、bfloat16_t、float、uint64_t、int64_t，MERGE_SORT排序算法支持的数据类型为：half、float。 |
    | U | 操作数srcIndexTensor和dstIndexTensor的数据类型。RADIX_SORT排序算法支持的数据类型为：uint32_t、int32_t、uint64_t、int64_t，MERGE_SORT排序算法支持的数据类型为：uint32_t。 |
    | isReuseSource | 可选参数。是否可以复用输入的Tensor空间。 |
    | config | 可选参数。Sort接口的相应配置：选择的排序算法，排序结果的升降序。数据类型SortConfig，定义如下方代码所示。<br>    <br>Sort提供了两种不同的排序算法，MERGE_SORT归并排序算法和RADIX_SORT基排序算法。两种算法在执行速度、时间复杂度和算法稳定性上表现不同。<br>    MERGE_SORT是一种稳定的排序算法，在所有情况下算法的时间复杂度都是O(nlogn)。<br>RADIX_SORT算法的时间复杂度是O(n)，在处理大量数据时，如果最大数字的位数较少，该算法的效率很高，可以接近线性时间复杂度。但是如果最大数字的位数很大，时间复杂度会接近O(n^2)。<br>    <br>config的默认值DEFAULT_SORT_CONFIG取值如下，使用基排序RADIX_SORT，对排序结果升序排序。<br>    constexpr SortConfig DEFAULT_SORT_CONFIG = {SortType::RADIX_SORT, false}; |

    ```
    enum class SortType {
        RADIX_SORT, // 使用基排序算法实现
        MERGE_SORT  // 使用归并排序算法实现
    };
    struct SortConfig {
        SortType type = SortType::RADIX_SORT; // 排序算法
        bool isDescend = false;               // 是否降序排序，默认值为false，输出结果升序排序
    };
    ```

    **表4**  参数说明

    | 参数名称 | 输入/输出 | 含义 |
    | --- | --- | --- |
    | dstTensor | 输出 | 值目的操作数，shape为[n]。MERGE_SORT算法下输出数据的每个元素需要按8Byte申请空间。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。 |
    | dstIndexTensor | 输出 | 索引目的操作数，shape为[n]。当输入不带srcIndexTensor时，只支持uint32_t类型。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。 |
    | srcTensor | 输入 | 值源操作数，shape为[n]。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。<br>    <br>此源操作数的数据类型需要与值目的操作数保持一致。 |
    | srcIndexTensor | 输入 | 索引源操作数，shape为[n]。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。<br>    <br>此源操作数的数据类型需要与索引目的操作数保持一致。 |
    | sharedTmpBuffer | 输入 | 临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供，临时空间大小BufferSize的获取方式请参考[GetSortMaxMinTmpSize](get_sort_max_min_tmp_size.md)。数据类型为uint8_t。<br>    <br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>    <br>LocalTensor的起始地址需要32字节对齐。 |
    | calCount | 输入 | 需要进行排序的数据元素个数。uint32_t类型。 |

## 返回值说明

无

## 约束说明

-   操作数地址对齐要求请参见[通用地址对齐约束](../../general_description_and_constraints.md#section796754519912)。
-   不带SortConfig的接口：
    -   当存在score\[i\]与score\[j\]相同时，如果i\>j，则score\[j\]将首先被选出来，排在前面，即index的顺序与输入顺序一致。
    -   非全排序模式下，每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。

-   带SortConfig的接口：
    -   基排序RadixSort和归并排序MergeSort都为稳定排序，即相同值在排序后的先后顺序保持不变。
    -   值目的操作数、值源操作数、索引目的操作数、索引源操作数的元素个数相同，且calCount参数值不能超过元素个数。
    -   不支持源操作数与目的操作数地址重叠。
    -   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
    -   使用MERGE\_SORT算法排序时，待排序的元素个数必须是32的倍数。若不是32的倍数，用户需要手动将数据量补齐到32的倍数。

## 调用示例

-   处理128个half类型数据。

    该样例适用于：

    
    Ascend 950PR&950DT系列产品
    

    
    Atlas A2系列产品
    

    
    Atlas A3系列产品
    

    
    Kirin X90处理器系列产品
    

    
    Kirin 9030处理器系列产品
    

    ```
    // sortedLocal：排序结果
    // concatLocal：待排序数据的预处理结果
    // indexLocal：索引数据
    // sortTmpLocal：临时空间
    // 待排序元素数量
    uint32_t m_elementCount = 128;
    // 迭代次数，每次迭代完成32个元素的排序
    uint32_t m_sortRepeatTimes = m_elementCount / 32;
    uint32_t m_extractRepeatTimes = m_elementCount / 32;
    // 预处理
    AscendC::Concat(concatLocal, valueLocal, concatTmpLocal, m_concatRepeatTimes);
    // 执行排序
    AscendC::Sort<T, isFullSort>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, m_sortRepeatTimes);
    AscendC::Extract(dstValueLocal, dstIndexLocal, sortedLocal, m_extractRepeatTimes);
    ```

    ```
    示例结果
    输入数据（srcValueGm）: 128个half类型数据
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输入数据（srcIndexGm）:
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输出数据（dstValueGm）:
    [127 126 125 ... 2 1 0]
    输出数据（dstIndexGm）:
    [127 126 125 ... 2 1 0]
    ```

-   处理64个half类型数据。

    该样例适用于：

    
    Atlas推理系列产品AI Core
    

    ```
    uint32_t m_elementCount = 64;
    uint32_t m_sortRepeatTimes = m_elementCount / 16;
    uint32_t m_extractRepeatTimes = m_elementCount / 16;
    AscendC::Concat(concatLocal, valueLocal, concatTmpLocal, m_concatRepeatTimes);
    AscendC::Sort<T, isFullSort>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, m_sortRepeatTimes);
    AscendC::Extract(dstValueLocal, dstIndexLocal, sortedLocal, m_extractRepeatTimes);
    ```

    ```
    示例结果
    输入数据（srcValueGm）: 64个half类型数据
    [15 14 13 ... 2 1 0
     31 30 29 ... 18 17 16
     47 46 45 ... 34 33 32
     63 62 61 ... 50 49 48]
    输入数据（srcIndexGm）:
    [15 14 13 ... 2 1 0
     31 30 29 ... 18 17 16
     47 46 45 ... 34 33 32
     63 62 61 ... 50 49 48]
    输出数据（dstValueGm）:
    [63 62 61 ... 2 1 0]
    输出数据（dstIndexGm）:
    [63 62 61 ... 2 1 0]
    ```

-   带SortConfig
    -   处理1024个half类型数据，输入索引和输出索引为1024个uint32\_t类型数据。

        该样例适用于：

        
        Ascend 950PR&950DT系列产品
        

        ```
        static constexpr AscendC::SortConfig config = {AscendC::SortType::RADIX_SORT, false};
        Sort<T, false, config>(dstLocal, dstIndexLocal, srcLocal, 1024);
        ```

        ```
        示例结果
        输入数据（srcGm）: 1024个half类型数据
        [1023 1022 ... 2 1 0]
        输入数据（srcIndexGm）: 1024个uint32_t类型数据
        [0 1 2 ... 1022 1023]
        输出数据（dstGm）:
        [0 1 2 ... 1022 1023]
        输出数据（dstIndexGm）:
        [1023 1022 ... 2 1 0]
        ```

- [InplaceIndexFill 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_index_fill_950.md)
- [GroupedBiasAddGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/math/grouped_bias_add_grad.md)
- [IndexFill 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/index_fill_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
- [Sort 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/sort_950.md)
- [IndexFillTensor 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/index_fill_tensor_950.md)
- [MoeDistributeDispatchTeardown 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_teardown_950.md)
- [TopK 950 Direct Invoke 算子知识](../../../../operators/direct/math/math/top_k_950.md)
- [InplaceIndexFillTensor 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_index_fill_tensor_950.md)
- [MoeInitRoutingV3 A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/moe/moe_init_routing.md)
- [ScatterNd 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/scatter_nd_950.md)
