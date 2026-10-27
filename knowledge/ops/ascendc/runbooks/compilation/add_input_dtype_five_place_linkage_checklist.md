---
type: "Runbook"
title: "AscendC 算子新增输入 dtype 五处联动清单"
description: 自定义算子新增输入 dtype（或扩展 dtype×模式矩阵）的联动改动清单：op_def 注册表、tiling_key.h 模板声明、op_host tiling 分发链、kernel Cast 路径与 buffer 尺寸、op_api L0/L2 两张 AICORE_DTYPE_SUPPORT_LIST 白名单；漏任何一处分别表现为缺二进制/UT ExecuteTiling false/561103/161002/精度失配，每处漏改均实测复现。
tags: [task.implementation, topic.implementation.tiling, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# AscendC 算子新增输入 dtype 五处联动清单

## 现象

（本卡为成功实现的完整配方。）bench_no_harness_top_k 迭代三把算子从 5 种 dtype（fp16/fp32/bf16/int8/uint8）扩展到 7 种（新增 int32/int64，走 Cast 路径 int→fp32→TopK→fp32→int，TilingKey 0~13 = 7 dtype × small/large inner）；迭代二还曾把 large inner 分块模式从 fp16/fp32 扩到全部 Cast dtype。扩展过程依次踩了四个"漏改一处"的坑（UT tiling 失败、上板 561103、L2 白名单 161002、int 数据域失配），最终以五处联动改动全部落地后通过全量验收：UT 74/74、C++ ST 110/110（跳过 19 个 dim!=last 已知限制）、PyTorch ST 577/577（跳过 159 个）。本卡沉淀"改哪里、为什么、怎么核对"的完整清单与关键代码。

## 适用场景

- 工程形态：AscendC 自定义算子仓（op_def + op_host tiling + op_kernel 模板 + op_api aclnn 两段式 + UT/ST）新增输入/输出 dtype，或扩展 dtype×模式支持矩阵。
- API 路径：`OpDef::Input/Output().DataType`、`ASCENDC_TPL_DATATYPE_DECL/ASCENDC_TPL_ARGS_SEL`、tiling 的 DTypeGroup 分发、kernel 的 `IsCastDType/GetSortT` 特化与转换、L0/L2 op_api 的 `AICORE_DTYPE_SUPPORT_LIST`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0。
- 适用粒度：operator_family——所有 ASCENDC_TPL 模板化 + aclnn 封装的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非模板算子（无 ASCENDC_TPL）；新增的是复合格式而非 dtype；中间精度非 fp32/fp16 的 Cast 设计（buffer 尺寸推导需按实际中间类型重算）。

## 根因

dtype 支持分散在五处独立配置、无单一事实源，漏改任何一处都在不同阶段以不同错误码暴露：

| # | 改动点 | 漏改症状（均已实测） |
|---|--------|--------------------|
| 1 | `op_host/<op>_def.cpp` dtype 注册表 | 构建成功但该 dtype 无 `.o`/binList 条目（opbuild 按 def 推导实例集；def 与 TPL 实例化双向同步的专项机制见 def_dtype_set_must_match_tpl_instantiation 卡） |
| 2 | `op_kernel/arch35/<op>_tiling_key.h` DECL/SEL | dtype×模式组合未注册——扩展模式矩阵时 tiling 显式拒绝或 561103（见下） |
| 3 | `op_host/arch35/<op>_tiling.cpp` 分发链（枚举/GetDTypeGroup/GetSortDTypeSize/isCast/拒绝分支/UB 计算） | UT `ExecuteTiling Actual: false`（UNSUPPORTED 早退）或合法组合被拒 |
| 4 | `op_kernel/arch35/<op>.h` Cast 路径（IsCastDType/GetSortT 特化、buffer 尺寸、int↔float 转换、pad 值） | kernel 编译失败或运行期越界/精度错误 |
| 5 | `op_api/<op>.cpp` L0 与 `op_api/aclnn_<op>.cpp` L2 的 `AICORE_DTYPE_SUPPORT_LIST`（两张独立白名单） | 上板 ST `GetWorkspaceSize 失败: 561103`（L0）或 `161002`（L2 CheckParams 拒绝，根本不进 kernel 路径） |

外加一条数据域约束：int dtype 经 fp32 Cast 时，ST/UT 数据须在 fp32 精确表示域内（|v| ≤ 2^24，工程取 ±2^23），否则整数精确比对必失配（见 topk_integer_dtype_cast_sort_recipe 卡）。

## 解决配方

**处 2：dtype×模式矩阵的 ASCENDC_TPL_ARGS_SEL 补注册**（扩展 large inner 模式到全部 Cast dtype，verbatim）——修复前只注册 fp16/fp32 的 LARGE 条目：

```cpp
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    )
);
```

修复后补齐 bf16/int8/uint8 的 LARGE 条目（int32/int64 同理）：

```cpp
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT8),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_UINT8),
        ASCENDC_TPL_UINT_SEL(INNER_MODE, ASCENDC_TPL_UI_LIST, BENCH_TOPK_INNER_MODE_LARGE)
    )
);
```

同时删除 tiling.cpp 中该组合的显式拒绝（连同其日志）：

```cpp
    // 删除以下整段（该迭代已支持 large inner + Cast dtype）：
    OP_CHECK_IF(isLargeInner && isCast,
                OP_LOGE(context,
                        "BenchNoHarnessTopK: iter2 does not support large inner + Cast dtype (TK=7), "
                        "got dtype=%d, inner=%ld.", static_cast<int>(dataType), inner),
                return ge::GRAPH_FAILED);
```

两处必须同步：tiling_key.h 的 `ASCENDC_TPL_SEL` 决定编译期生成哪些模板实例与 tilingKey 路由；tiling.cpp 的支持判定决定运行期放行与否。只删拒绝不补注册，tilingKey 无法路由到对应模板；只补注册不删拒绝，仍被拦截。注意：注册扩展后打包产物中 kernel 二进制数量不按 dtype×mode 简单线性增长（binary_info_config 按 simplifiedKey 合并），不能以二进制个数反推注册是否生效，应以 ST 用例能否执行为准。

**处 3：tiling 分发链四处补齐**（修复前为迭代二状态，`GetDTypeSize` 已前瞻性支持 int32/int64 而"半改"未同步其余链）：

```cpp
enum class DTypeGroup {
    NATIVE_FP16, NATIVE_FP32, CAST_BF16, CAST_INT8, UNSUPPORTED,
};

static DTypeGroup GetDTypeGroup(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16: return DTypeGroup::NATIVE_FP16;
        case ge::DT_FLOAT:   return DTypeGroup::NATIVE_FP32;
        case ge::DT_BF16:    return DTypeGroup::CAST_BF16;
        case ge::DT_INT8:
        case ge::DT_UINT8:   return DTypeGroup::CAST_INT8;
        default:             return DTypeGroup::UNSUPPORTED;
    }
}

static uint32_t GetSortDTypeSize(DTypeGroup group)
{
    switch (group) {
        case DTypeGroup::NATIVE_FP16: return 2;
        case DTypeGroup::NATIVE_FP32: return 4;
        case DTypeGroup::CAST_BF16:   return 4;  // bf16 排序 dtype = float
        case DTypeGroup::CAST_INT8:   return 2;
        default:                       return 0;
    }
}

// tiling 函数内：
    bool isCast = (dtypeGroup == DTypeGroup::CAST_BF16 || dtypeGroup == DTypeGroup::CAST_INT8);
```

修复后（4 处编辑，逐字取自修复 edit 事件）：

```cpp
enum class DTypeGroup {
    NATIVE_FP16, NATIVE_FP32, CAST_BF16, CAST_INT8, CAST_INT32, CAST_INT64, UNSUPPORTED,
};
```

```cpp
        case ge::DT_INT8:
        case ge::DT_UINT8:   return DTypeGroup::CAST_INT8;
        case ge::DT_INT32:   return DTypeGroup::CAST_INT32;
        case ge::DT_INT64:   return DTypeGroup::CAST_INT64;
        default:             return DTypeGroup::UNSUPPORTED;
```

```cpp
        case DTypeGroup::CAST_INT8:   return 2;  // int8/uint8 排序 dtype = half
        case DTypeGroup::CAST_INT32:  return 4;  // int32 排序 dtype = float
        case DTypeGroup::CAST_INT64:  return 4;  // int64 排序 dtype = float
        default:                       return 0;
```

```cpp
    bool isCast = (dtypeGroup == DTypeGroup::CAST_BF16 || dtypeGroup == DTypeGroup::CAST_INT8 ||
                   dtypeGroup == DTypeGroup::CAST_INT32 || dtypeGroup == DTypeGroup::CAST_INT64);
```

UB 计算随 buffer 字节变化（srcValueOrigBuf 按 max(dtypeSize, sortDtypeSize) 分配——int64 输入 8 字节 > 排序类型 4 字节）：

```cpp
        if (isCast) {
            uint32_t origBufElemSize = (dtypeSize > sortDtypeSize) ? dtypeSize : sortDtypeSize;
            ubRequired += static_cast<uint64_t>(innerPad) * origBufElemSize;  // srcValueOrigBuf
            uint32_t dstOrigSize = (dtypeSize > sizeof(int16_t)) ? dtypeSize : sizeof(int16_t);
            ubRequired += static_cast<uint64_t>(kPad) * dstOrigSize;
        }
```

排查入口：UT 出现 `ExecuteTiling Actual: false` 时，在 tiling 函数各 `OP_CHECK_IF` 后插 `fprintf(stderr, "[DEBUG] ... OK\n")` 探针跑单用例，看打印停在哪个检查（本例探针实测：GetPlatformInfo/GetShapeAttrsInfo/UB 检查均通过、ubRequired=3872 < ubSize=253952，失败点在 dtype 检查）。

**处 4：kernel Cast 路径**——编译期特征与 buffer 尺寸：

```cpp
// 是否需要 Cast（fp16/fp32=false, bf16/int8/uint8/int32/int64=true）
template <typename T>
struct IsCastDType {
    static constexpr bool value = false;
};
template <>
struct IsCastDType<bfloat16_t> { static constexpr bool value = true; };
template <>
struct IsCastDType<int8_t> { static constexpr bool value = true; };
template <>
struct IsCastDType<uint8_t> { static constexpr bool value = true; };
template <>
struct IsCastDType<int32_t> { static constexpr bool value = true; };
template <>
struct IsCastDType<int64_t> { static constexpr bool value = true; };
```

```cpp
        // Cast 场景需要 srcValueOrigBuf（原 dtype 输入）
        // 注意: int64 的 sizeof(T)=8 > sizeof(SortT)=4，需按较大者分配
        if constexpr (IS_CAST) {
            constexpr uint32_t origBufElemSize = (sizeof(T) > sizeof(SortT)) ? sizeof(T) : sizeof(SortT);
            pipe.InitBuffer(srcValueOrigBuf, innerPad_ * origBufElemSize);
        }
```

int→float 正向转换（CopyIn，逐元素 SetValue，不用 Cast API——arch35 Cast 对不同位宽类型有"按较大类型字节数读取"的位宽 bug，见 arch35_cast_cross_width_oob_read 卡，本算子对各 Cast dtype 均规避 Cast API）：

```cpp
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
            // int32/int64 路径：DataCopyPad 搬入 + SetValue 逐元素 int→float
            LocalTensor<T> srcValueOrig = srcValueOrigBuf.Get<T>();
            if (useLastDimPath) {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(thisBlockLen * sizeof(T));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad<T>(srcValueOrig, gmSrc[rowIdx * effSrcRowStride + blockOffset], copyParams,
                               {false, 0, 0, static_cast<T>(0)});
            } else {
                // 非末维：GetValue + SetValue 逐元素搬入
                int64_t baseOffset = rowIdx * srcRowStride_ + blockOffset * srcDimStride_;
                for (int64_t t = 0; t < thisBlockLen; t++) {
                    srcValueOrig.SetValue(static_cast<uint32_t>(t),
                                          gmX.GetValue(baseOffset + t * srcDimStride_));
                }
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();
            // int32/int64 → float：SetValue 逐元素转换
            for (int64_t i = 0; i < thisBlockLen; i++) {
                T val = srcValueOrig.GetValue(static_cast<uint32_t>(i));
                srcValueSort.SetValue(static_cast<uint32_t>(i), static_cast<float>(val));
            }
            PipeBarrier<PIPE_ALL>();
```

int 极值 pad 值（largest 取负极值不被选、非 largest 取正极值）：

```cpp
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return isLargest ? static_cast<int32_t>(-2147483647 - 1) : static_cast<int32_t>(2147483647);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return isLargest ? static_cast<int64_t>(-9223372036854775807LL - 1) : static_cast<int64_t>(9223372036854775807LL);
    } else {
```

**处 5：L0 与 L2 两张 AICORE_DTYPE_SUPPORT_LIST**——dtype 扩展只同步 def/tiling/L0（或 L2）不够，另一张白名单漏配时：L2 在 `CheckParams` 阶段以 `ACLNN_ERR_PARAM_INVALID (161002)` 拒绝请求（op_api UT 接受性用例 `Expected: (ret) != (161002), actual: 161002 vs 161002`）；L0 入口 `IsAiCoreSupport` 的 `OP_CHECK(CheckType(...))` 拒绝则最终以 561103 形态在 ST 上板报出。两张白名单修复前后同型（verbatim）：

```cpp
// 修复前（扩展前 5 dtype）：
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16,
    DataType::DT_FLOAT,
    DataType::DT_BF16,
    DataType::DT_INT8,
    DataType::DT_UINT8
};

// 修复后（与 def / tiling / TPL 对齐为 7 dtype，L0 与 L2 两个文件都要改）：
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16,
    DataType::DT_FLOAT,
    DataType::DT_BF16,
    DataType::DT_INT8,
    DataType::DT_UINT8,
    DataType::DT_INT32,
    DataType::DT_INT64
};
```

L0 检查上下文（平台与 dtype 两道门，供对照；错误文案同步更新，陈旧文案会把排查引向"安装旧库"）：

```cpp
static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BenchNoHarnessTopK not supported on this platform: npuArch=%d. "
                     "Only DAV_3510 (Ascend950PR) is supported.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
              OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                      "BenchNoHarnessTopK not supported: dtype=%d. Supports FLOAT16/FLOAT/BF16/INT8/UINT8/INT32/INT64.",
                      static_cast<int>(x->GetDataType())),
              return false);
    return true;
}
```

配套 UT：原先"期望新 dtype 被拒绝"的负向用例须替换为仍然不支持的 dtype（如 DT_BOOL/DT_INT16），并新增 `Int32DtypeAccepted`/`Int64DtypeAccepted` 正向用例（断言 `ret != 161002`，不驱动真实 NPU 计算）；原先断言"large inner + Cast dtype 应失败"的 UT 用例须同步改为成功路径用例。

**核对清单（每处改完必查）**：

```bash
# (1) 二进制：.o 计数 == dtype 数；binList dtype 清单含新 dtype
ls build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o | wc -l
# (3) UT：ExecuteTiling 全通过
cd tests/ut && rm -rf build && bash run.sh
# (5) ST Real：无 561103/161002、新 dtype 用例执行到精度比对阶段
cd tests/st && bash run.sh --real
```

## 设计决策

- 为什么 int32/int64 走 fp32 Cast 而不是原生排序：`AscendC::TopK` 仅支持 half/float 排序 dtype；原生 int 排序需自研（成本高）。代价是 2^24 表示域边界（专项卡跟踪）。
- 为什么逐元素 SetValue 而不用 `AscendC::Cast` 向量 API：arch35 Cast API 对不同位宽类型有位宽 bug（硬件按较大类型字节数读取导致元素错位），本算子迭代二起对 bf16（位重排）、int8/uint8（先扩 int16）均规避，int32/int64 沿用逐元素转换保持一致；逐元素版本经全量验证正确。
- 为什么 buffer 尺寸取 `max(sizeof(T), sizeof(SortT))`：int64 输入 8 字节 > 排序类型 4 字节，按较小者分配会越界。
- 为什么五处必须一次联动：每处漏改的症状都出现在不同阶段（编译"成功"、UT 失败、上板 561103、L2 161002、精度失配），单点修复会让排查串行拉长一整轮构建-安装-上板循环。
- 正向接受性 UT 用例只断言 `ret != 161002`（通过 CheckParams），完整数值验证留给 ST Real——op_api UT 可在无 NPU 环境回归 dtype 校验。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 把本清单固化为新增 dtype 的 PR 自查模板：五处改动 + 数据域约束 + 三条核对命令。
- 代码评审时对"新增枚举值"类 diff 强制检查所有 switch/if 链（GetDTypeGroup/GetSortDTypeSize/isCast 是三处显式枚举链）。
- 561103 出现且同维度其他 dtype 通过时，优先 grep tiling.cpp 的显式拒绝与 TPL_SEL 注册表、再查两张白名单；排查前先做"安装链路完好性"核对（binList/simplifiedKey/md5）。

## 验证信号

来源说明：该算子迭代二/迭代三 dtype 扩展联调会话，各"漏改一处"坑的修复前后实测对照（Ascend950PR / CANN 9.1.0），最后全量回归验收。

- 处 2/3 修复前：ST Real 首轮 bf16/int8/uint8 × 大 inner（8192/16384）用例 `GetWorkspaceSize 失败: 561103`（fp32 同 shape 用例 PASS）；补 ASCENDC_TPL_ARGS_SEL 注册并移除拒绝后原 561103 用例全部可执行（IT2-013/015/017/025/027/029 PASS，largest=true 精度问题另卡）（来源平台：950；证据等级：tool_observed）。
- 处 3 修复前：op_host UT 10 个 int32/int64 用例 `ExecuteTiling(para, info) Actual: false`（探针实测 GetPlatformInfo/GetShapeAttrsInfo/UB 检查通过、失败点在 dtype 检查）；补齐四处分发链后 op_host UT 58 tests 全过 + op_api UT 16 tests 全过（来源平台：950；证据等级：tool_observed）。
- 处 5 修复前（L2 漏配）：op_api UT `Int32DtypeAccepted/Int64DtypeAccepted` 均 `Expected: (ret) != (161002), actual: 161002 vs 161002` FAILED，ST Real int32/int64 用例 `[FAIL] GetWorkspaceSize 失败: 161002`；L2 列表补齐后 clean build UT 16/16 PASSED（来源平台：950；证据等级：tool_observed）。
- 处 5 修复前（L0 漏配）：ST Real 129 用例中 44 个 int32/int64 用例全部 `GetWorkspaceSize 失败: 561103`，而安装链路核对正常（7 dtype 齐全、int32/int64 各 4 个 simplifiedKey、.o md5 与 build 一致、tiling 库重装时间戳晚于源码修改）；补 L0 白名单后 561103 计数归零、通过数 70→74（来源平台：950；证据等级：tool_observed）。
- 五处联动全部落地后：UT 74/74；C++ ST Real 129 用例跳过 19（dim!=last 已知限制）、110/110 通过；PyTorch ST 执行 577 全通过（int32 84 个、int64 76 个全 mismatch=0）；构建产物 7 个 `.o`、binList 7 dtype（int32/int64 各 4 个 simplifiedKey）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 清单基于 CANN 9.1.0 的 opbuild/ASCENDC_TPL/aclnn 行为；其他版本的字段或宏语义变化需重核对（未验证，结构性流程推断可迁移）。
- 非末维（dim!=last）场景不随本清单解决——arch35 已知限制另卡跟踪；int32/int64 数据域 |v|>2^24 不在本配方覆盖内（表示域硬边界）。
- 161002 亦覆盖 k/dim/shape 等其它参数错误、561103 亦有二进制缺失/tiling 库旧版等其它成因，需按判别顺序逐一排除后才能落到白名单/注册表根因。

**未决项**：

- 逐元素 SetValue 与向量 Cast 的性能差未测量（若成为瓶颈，Cast API 的位宽 bug 修复/规避是前置课题）。
- 非末维路径（dim!=last）仍为已知限制，见专项卡。
- dtype 多处列表的自动化一致性检查（例如 CI 比对五处清单）值得建设但本次会话未实现。
## 坏实践

- 五处只改三处就上板联调——每漏一处多一轮"构建-安装-上板"排错循环，本迭代四个坑全部串行踩过。
- dtype 扩展只改 def/tiling/kernel 三处、漏掉 op_api 两张白名单——错误被推迟到 UT 161002 或上板 561103 呈现，且文案陈旧时极易误判为安装问题。
- 只删 tiling.cpp 拒绝、忘记补 `ASCENDC_TPL_SEL` 注册（或反之），问题以另一种形式残留。
- 用 kernel 二进制个数判断注册是否生效（binary_info_config 按 simplifiedKey 合并，会误判）。
- 数据生成器沿用旧 dtype 的全范围分布——新 dtype 的中间精度约束（fp32 2^24）必须同步进数据域。
- 负向 UT 用例绑定"当前不支持的 dtype"却不随支持范围更新——支持扩展后这些用例从"验证拦截"变成"验证失败"。

## 误诊

- 迭代过程中"tiling_key.h 是 UT 失败根因""构建缓存导致缺二进制""安装过期导致 561103"三个假设均被逐项排除（`ASCENDC_TPL_SEL_PARAM` 在非 ASCENDC_DEBUG 下走 `FastEncodeTilingKeyDirect` 对 DATATYPE 参数不做存在性校验；安装链路 md5/simplifiedKey 核对完好）——症状出现在哪一层，不等于根因在哪一层。
- L0 白名单误诊轮：先后怀疑 tiling 库安装过期（`strings` 见旧迭代文案——实为源码里未更新的文案，代码路径已支持）、ops-info 安装缺 binList、TilingKey 映射格式问题；`strings` 看到的陈旧文案要先确认其代码路径是否为实际失败点。
