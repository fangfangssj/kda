---
type: "Runbook"
title: "融合算子 TilingData 内嵌 SoftMax/TopkTiling"
description: "高阶 API 融合算子用标准 C++ struct 定义 TilingData，直接内嵌 AscendC::tiling::SoftMaxTiling/TopkTiling plain struct，host TilingFunc 用 SoftMaxTilingFunc/TopKTilingFunc 填充、kernel 用 GET_TILING_DATA_WITH_STRUCT 一次读出，host/kernel 二进制兼容；附完整可复用结构定义与填充/消费代码。"
tags: [task.implementation, topic.tiling, topic.mixture_of_experts]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# 融合算子 TilingData 内嵌 SoftMax/TopkTiling

## 现象

MoE 门控 Softmax+TopK 融合算子（Ascend950PR，CANN 9.1.0，registry-invoke 工程）需要把 SoftMax 与 TopK 两个高阶 API 组合在同一个 kernel 里：host 侧 tiling 要同时算出两套 API 的 tiling 参数，kernel 侧要一次拿到两套参数加自定义切分参数。若用手写 BEGIN_TILING_DATA_DEF 宏逐一登记两套 API 的全部字段，既冗长又容易与 API 内部结构失配。本卡沉淀实测可构建通过的"TilingData 内嵌高阶 API plain struct"完整配方。

## 适用场景

- 工程形态：registry-invoke 自定义算子，融合（composite）多个高阶 API（SoftMax、TopK、以及其他在 `asc/include/tiling/` 提供 tiling 头的 API）。
- API 路径：`AscendC::tiling::SoftMaxTiling`/`AscendC::tiling::TopkTiling`（plain struct，`kernel_tiling/kernel_tiling.h` 体系）；host 侧 `SoftMaxTilingFunc`/`TopKTilingFunc`/`GetSoftMaxMaxMinTmpSize`/`GetTopKMaxMinTmpSize`；kernel 侧 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`。
- 验证过的平台与 CANN 版本：Ascend950PR（soc=ascend950），CANN 9.1.0（验证深度：完整编译打包通过、6 tiling key 二进制生成；上板运行/精度属后续阶段，未验证）。
- 适用粒度：operator_family——所有"高阶 API 融合 + 多核切分 + UB 分批"形态的算子（不限 MoE 门控）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：纯基础 API 手写 kernel（无高阶 API tiling 结构可嵌）；上板精度/性能表现未在本阶段验证。
- 配套前置：host 侧引用高阶 API tiling 头需补 adv_api include 路径、`namespace optiling` 内需 `using namespace AscendC;`（两条独立编译坑另有卡片）。

## 解决配方

三段式：定义（tiling_data.h）→ host 填充（tiling.cpp）→ kernel 消费（入口 + kernel 类）。

### 完整代码

**1. TilingData 定义（`op_kernel/arch35/<op>_tiling_data.h`，verbatim 取自终版写入）**——标准 C++ struct，内嵌两个 plain struct：

```cpp
#ifndef _BENCH_NO_HARNESS_MOE_GATING_TOP_K_SOFTMAX_TILING_DATA_H_
#define _BENCH_NO_HARNESS_MOE_GATING_TOP_K_SOFTMAX_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct BenchNoHarnessMoeGatingTopKSoftmaxTilingData {
    // 多核切分参数
    int64_t M = 0;              // 非尾轴长度乘积（2D: N, 3D: B*N）
    int64_t E = 0;              // 尾轴长度（专家数）
    int64_t k = 0;              // topK 数量
    int64_t blockFactor = 0;    // 每核处理的行数
    int64_t batchRows = 0;      // 单次 UB 处理的行数
    int64_t EAlign = 0;         // E 向上对齐到 32B 的元素数（基于 COMPUTE_T）
    int64_t kPad = 0;           // k 向上对齐到 32B 的元素数（值类型，基于 COMPUTE_T）
    int64_t kPadIndex = 0;      // k 向上对齐到 32B 的元素数（索引类型 int32，8 元素）

    // SoftMax Tiling
    AscendC::tiling::SoftMaxTiling softmaxTilingData;

    // TopK Tiling
    AscendC::tiling::TopkTiling topkTilingData;

    // TopK 信息
    int32_t topkOutter = 0;     // TopK outter（= batchRows）
    int32_t topkInner = 0;      // TopK inner（= EAlign）
    int32_t topkN = 0;          // TopK n（= E，实际长度）

    // 临时空间大小
    uint32_t softmaxTmpSize = 0;
    uint32_t topkTmpSize = 0;
};
#endif // _BENCH_NO_HARNESS_MOE_GATING_TOP_K_SOFTMAX_TILING_DATA_H_
```

**2. host 侧填充（`op_host/arch35/<op>_tiling.cpp`，TilingFunc 内关键段 verbatim）**——先用高阶 API 的 TilingFunc 生成 plain struct，再整体赋值进自定义 TilingData：

```cpp
    // SoftMax tiling
    AscendC::tiling::SoftMaxTiling softmaxTiling;
    SoftMaxTilingFunc(srcShape, computeDtypeSize, softmaxMinTmp, softmaxTiling);

    // TopK tiling
    GetTopKMaxMinTmpSize(ascendcPlatform, static_cast<int32_t>(EAlign),
                         static_cast<int32_t>(batchRows),
                         false, false, TopKMode::TOPK_NORMAL, true, computeDtypeSize,
                         topkMaxTmp, topkMinTmp);
    AscendC::tiling::TopkTiling topkTiling;
    TopKTilingFunc(ascendcPlatform, static_cast<int32_t>(EAlign),
                   static_cast<int32_t>(batchRows), static_cast<int32_t>(k),
                   computeDtypeSize, false, TopKMode::TOPK_NORMAL, true, topkTiling);

    // 7. 设置 TilingData
    BenchNoHarnessMoeGatingTopKSoftmaxTilingData* tiling =
        context->GetTilingData<BenchNoHarnessMoeGatingTopKSoftmaxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BenchNoHarnessMoeGatingTopKSoftmaxTilingData), 0,
                 sizeof(BenchNoHarnessMoeGatingTopKSoftmaxTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->M = M;
    tiling->E = E;
    tiling->k = k;
    tiling->blockFactor = blockFactor;
    tiling->batchRows = batchRows;
    tiling->EAlign = EAlign;
    tiling->kPad = kPad;
    tiling->kPadIndex = kPadIndex;
    tiling->softmaxTilingData = softmaxTiling;
    tiling->topkTilingData = topkTiling;
    tiling->topkOutter = static_cast<int32_t>(batchRows);
    tiling->topkInner = static_cast<int32_t>(EAlign);
    tiling->topkN = static_cast<int32_t>(E);
    tiling->softmaxTmpSize = softmaxMaxTmp;
    tiling->topkTmpSize = topkMaxTmp;
```

（该文件命名空间开头需 `using namespace AscendC;`，include 路径需补 `asc/include/adv_api/{sort,activation}`，见同主题两张编译卡。）

**3. kernel 侧消费（`op_kernel/<op>_arch35.cpp` 入口，终版 verbatim）**：

```cpp
#include "arch35/bench_no_harness_moe_gating_top_k_softmax.h"

template <typename D_T_X, bool HAS_FINISHED>
__global__ __aicore__ void bench_no_harness_moe_gating_top_k_softmax(
    GM_ADDR x, GM_ADDR finished, GM_ADDR y,
    GM_ADDR expertIdx, GM_ADDR rowIdx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessMoeGatingTopKSoftmaxTilingData, tilingData, tiling);

    NsBenchMoeGating::BenchNoHarnessMoeGatingTopKSoftmax<D_T_X, HAS_FINISHED> op;
    op.Init(x, finished, y, expertIdx, rowIdx, &tilingData);
    op.Process();
}
```

kernel 类内直接使用嵌套结构（`op_kernel/arch35/<op>.h` 初始化段节选）：

```cpp
    // SoftMax shape info (初始值，实际执行时按 curBatch 更新)
    ...
    uint32_t sharedTmpSize = (tilingData->softmaxTmpSize > tilingData->topkTmpSize) ?
                             tilingData->softmaxTmpSize : tilingData->topkTmpSize;
    if (sharedTmpSize == 0) {
        sharedTmpSize = 1;  // 兜底
    }
    pipe.InitBuffer(sharedTmpBuf_, sharedTmpSize);
```

## 设计决策

- **为何选 plain struct 内嵌**：`AscendC::tiling::SoftMaxTiling`/`TopkTiling` 是无宏、无虚成员的 POD，直接作为自定义 TilingData 的成员时 host/kernel 两侧按同一内存布局解释，二进制兼容由 struct 定义本身保证；相比 `BEGIN_TILING_DATA_DEF` 逐字段登记，少一层样板且高阶 API 升级字段时无需同步两处。
- **备选与放弃原因**：把 SoftMax/TopK 的 tiling 参数手工拆散成标量再拼回——字段多、易漏、与 API 内部结构耦合；只在 kernel 里重新调用 TilingFunc——kernel 侧拿不到 platform 上下文，不可行。
- **属性 k 的通道**：入口签名不含属性（框架 wrapper 只传输入/输出/workspace/tiling），`k` 经 `tiling->k` 传递。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 环境：x86_64 宿主机交叉编译，CANN 9.1.0，目标 Ascend950PR（DAV_3510/arch35，soc=ascend950）。
- 目录布局：`op_host/arch35/<op>_tiling.cpp`（host 填充）、`op_kernel/arch35/<op>_tiling_data.h`（定义，kernel/host 共享）、`op_kernel/arch35/<op>_tiling_key.h`（模板参数 DECL/SEL）、`op_kernel/<op>_arch35.cpp`（入口）、`op_kernel/arch35/<op>.h`（kernel 类）。
- 相关文件清单（与事件流实际读写一一对应）：上述 5 个源文件 + `op_host/<op>_def.cpp`/`<op>_infershape.cpp` + `op_host/CMakeLists.txt` + `op_api/*` + `build.sh`。

## 验证信号

以下信号来自该融合算子完整构建（编译级验证；上板未在本阶段执行）：

- `bash build.sh --soc=ascend950 -j8` 编译打包成功，生成 3 个 kernel 二进制（`BenchNoHarnessMoeGatingTopKSoftmax_{06a5b7e1...,322daf1c...,23e59e1d...}.o`，覆盖 6 个 tiling key：3 dtype × 2 hasFinished）与 `custom_opp_ubuntu_x86_64.run` 算子包（来源平台：950；证据等级：tool_observed）。
- 最终复跑 `Build completed successfully!`、`exit code: 0`；binary_info_config.json 含 4 条 simplifiedKey（d=0/1 × p=0/1）与完整输入输出 dtype 注册（来源平台：950；证据等级：tool_observed）。
注：kernel 运行时正确性（SoftMax/TopK 数值、finished 语义）未经上板验证，属后续 ST/精度验收阶段。

## 失效边界

- plain struct 二进制兼容前提是 host 与 kernel 包含**同一版本**的 `kernel_tiling/kernel_tiling.h` 与同一份 tiling_data.h；CANN 升级后 `SoftMaxTiling`/`TopkTiling` 字段变化时需重新编译两侧（本工程 host/kernel 同仓同版本，天然满足）。
- 该配方的验证深度为编译/打包层；上板精度与性能表现不在本卡结论内（未验证，推断可迁移）。

**未决项**：

- 上板运行验证（含 finished 语义经 TopK isHasfinish 的实际行为、bf16 Cast 路径精度）留待后续测试阶段闭环；若上板失败应回溯本卡的设计前提。
## 坏实践

- 把高阶 API 的 tiling 参数手工拆散复制进自定义 TilingData——多一份要维护的镜像，API 内部字段变化即失配（本卡方案正是为消除该模式）。

## 误诊

- 无。
