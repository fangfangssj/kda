---
type: "Runbook"
title: elementwise host tiling 切分配方
description: host tiling 完整配方：GetCoreNumAiv 动态核数、blockFormer 对齐 512、ubFormer 按 dtype 精确字节预算（fp16=14/fp32=28/bf16=24 字节每元素）向下对齐 256B、空 tensor SetBlockDim(1)、0 维视作 {1}；op_host UT 12/12 覆盖边界用例。
tags: [task.implementation, topic.implementation.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# elementwise host tiling 切分配方

## 现象

在为 Mish 激活算子（y = x·tanh(softplus(x))，fp16/fp32/bf16，Ascend950PR/arch35，CANN 9.1.0）编写算子仓工程的 host 侧 tiling 时，需要一次性解决：多核切分（核数动态获取）、UB 切分（按 dtype 字节预算 + 对齐）、TilingKey 路由、以及空 tensor / 0 维 scalar / 非法 dtype / shape 不一致等边界。首版 tiling 实现后随工程编译通过，并配套 GoogleTest UT（host x86 执行，mock 平台信息）全量通过 12/12。

## 适用场景

- 工程形态：算子仓工程 op_host tiling（IMPL_OP_OPTILING）+ TilingData 结构 + kernel 侧消费。
- API 路径：`gert::TilingContext`（GetPlatformInfo/GetInputShape/GetTilingData/SetBlockDim/GetWorkspaceSizes）、`platform_ascendc::PlatformAscendC`、`ASCENDC_TPL_SEL_PARAM` TilingKey 路由、op_common 的 CeilDiv/CeilAlign/FloorDiv/FloorAlign。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0；UT 在 x86 宿主跑 mock 平台（与 frontmatter `platforms` 一致，UT 数值基于 mock 的 ubSize/核数）。
- 适用粒度：operator_family（一入一出 elementwise 算子的通用切分骨架）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：广播/双输入不同 shape、需要 workspace 的算子（本例 WS=0）、非 elementwise；真实 NPU 上的 tiling 行为未验证（只有 host UT + 编译）。

## 根因

不适用（本卡为完整配方，validated_pattern）。

## 解决配方

TilingData 结构（kernel 与 host 共享，字段语义即切分契约）：

```cpp
struct BenchNoHarnessMishTilingData {
    uint64_t totalLength = 0;            // 元素总数量（展平后）
    uint32_t coreNum = 0;                // 实际使用的核数
    uint64_t blockFormer = 0;            // 每个核的基础数据量（对齐到 512 元素）
    uint64_t blockNum = 0;               // 虚拟 block 数量
    uint64_t ubFormer = 0;               // UB 单次处理元素数（对齐到 256B）
    uint64_t ubLoopOfFormerBlock = 0;    // 首块 UB 循环次数
    uint64_t ubTailOfFormerBlock = 0;    // 首块尾部元素数
    uint64_t ubLoopOfTailBlock = 0;      // 尾块 UB 循环次数
    uint64_t ubTailOfTailBlock = 0;      // 尾块尾部元素数
    uint32_t tmpBufferSize = 0;          // Tanh 共享临时空间大小（字节）
};
```

host 侧常量：

```cpp
// 多核切分对齐粒度（元素数）：512 元素保证 32B 对齐（FP16: 1024B, FP32: 2048B）
constexpr uint64_t BLOCK_FORMER_ALIGN = 512;
// UB 切分对齐字节数
constexpr uint64_t UB_ALIGN_BYTES = 256;
// UB 安全余量（字节），预留栈空间等
constexpr uint64_t UB_RESERVE = 2048;
// Workspace 数量与大小（Elementwise 算子无需 workspace）
constexpr size_t WORKSPACE_NUM = 1;
constexpr uint64_t WS_SYS_SIZE = 0;
```

tiling 主函数（完整，含边界处理；dtype 的 UB 字节预算见 GetUbParams）：

```cpp
static ge::graphStatus BenchNoHarnessMishTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取 shape/dtype
    int64_t totalLength = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalLength, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. 设置 TilingData
    BenchNoHarnessMishTilingData* tiling = context->GetTilingData<BenchNoHarnessMishTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BenchNoHarnessMishTilingData), 0, sizeof(BenchNoHarnessMishTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);

    // TilingKey 路由参数（对应 tiling_key.h 的 D_T_X）
    uint32_t dTypeX = static_cast<uint32_t>(dataType);

    // 空 tensor：SetBlockDim(1)，TilingData 全 0
    if (totalLength == 0) {
        tiling->totalLength = 0;
        tiling->coreNum = 1;
        tiling->blockNum = 0;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分：blockFormer 对齐到 512 元素
    uint64_t totalLen = static_cast<uint64_t>(totalLength);
    uint64_t blockFormer = CeilAlign(CeilDiv(totalLen, static_cast<uint64_t>(coreNum)), BLOCK_FORMER_ALIGN);
    uint64_t blockNum = CeilDiv(totalLen, blockFormer);
    uint64_t blockTail = totalLen - (blockNum - 1) * blockFormer;  // 尾核元素数

    // UB 切分：按 dtype 计算 ubFormer
    UbParams ubParams = GetUbParams(dataType);
    uint32_t totalBytesPerElem = ubParams.bufferBytesPerElem + ubParams.tmpBytesPerElem;
    uint64_t usableUb = (ubSize > UB_RESERVE) ? (ubSize - UB_RESERVE) : 0;
    uint64_t ubFormer = FloorDiv(usableUb, static_cast<uint64_t>(totalBytesPerElem));
    ubFormer = FloorAlign(ubFormer, ubParams.alignElems);
    // 保证 ubFormer 至少能处理一个对齐块
    if (ubFormer == 0) {
        ubFormer = ubParams.alignElems;
    }
    uint32_t tmpBufferSize = static_cast<uint32_t>(ubFormer * ubParams.tmpBytesPerElem);

    // UB 循环次数与尾部
    uint64_t ubLoopOfFormerBlock = FloorDiv(blockFormer, ubFormer);
    uint64_t ubTailOfFormerBlock = blockFormer - ubLoopOfFormerBlock * ubFormer;
    uint64_t ubLoopOfTailBlock = FloorDiv(blockTail, ubFormer);
    uint64_t ubTailOfTailBlock = blockTail - ubLoopOfTailBlock * ubFormer;

    // 填充 TilingData
    tiling->totalLength = totalLen;
    tiling->coreNum = static_cast<uint32_t>(blockNum);
    tiling->blockFormer = blockFormer;
    tiling->blockNum = blockNum;
    tiling->ubFormer = ubFormer;
    tiling->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling->ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling->ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling->ubTailOfTailBlock = ubTailOfTailBlock;
    tiling->tmpBufferSize = tmpBufferSize;

    // 动态设置核数（禁止硬编码）
    context->SetBlockDim(static_cast<int64_t>(blockNum));

    // 5. 设置 TilingKey（参数顺序与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一致）
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}
```

按 dtype 的 UB 字节预算（队列 double buffer 双份 + 全部计算 buffer + Tanh 临时估算）：

```cpp
static UbParams GetUbParams(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            // inQueue(2*2) + outQueue(2*2) + 3*tmpBuf(3*2) = 14; tmp=2; align=256/2=128
            return {14U, 2U, 128U};
        case ge::DT_FLOAT:
            // inQueue(2*4) + outQueue(2*4) + 3*tmpBuf(3*4) = 28; tmp=4; align=256/4=64
            return {28U, 4U, 64U};
        case ge::DT_BF16:
            // inQueue(2*2) + outQueue(2*2) + bufXF32(4) + 3*tmpBuf(3*4) = 24; tmp=4; align=256/2=128
            return {24U, 4U, 128U};
        default:
            return {0U, 0U, 0U};
    }
}
```

0 维 scalar 归一化与 shape/dtype 校验：

```cpp
static const gert::Shape g_vec_1_shape = {1};

// 0维 tensor 转为 {1}，避免后续除零
static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}
```

注册（含必须存在的 CompileInfo 占位，入图场景依赖）：

```cpp
static ge::graphStatus TilingParseForBenchNoHarnessMish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BenchNoHarnessMishCompileInfo {};  // 必须定义，入图场景依赖

IMPL_OP_OPTILING(BenchNoHarnessMish)
    .Tiling(BenchNoHarnessMishTilingFunc)
    .TilingParse<BenchNoHarnessMishCompileInfo>(TilingParseForBenchNoHarnessMish);
```

## 设计决策

- **blockFormer 对齐 512**：多核均分后向上对齐到 512 元素（fp16 即 1024B），保证每核搬运块 32B 对齐；代价是核数少于物理核（blockNum=ceil(total/blockFormer)），换尾核不规则度低。
- **ubFormer 向下对齐**（FloorDiv 预算后 FloorAlign 到 256B/elemBytes）：UB 必须装得下，宁小勿越界；`ubFormer==0` 时兜底为一个对齐块。
- **字节预算精确记账**：fp16=14、fp32=28、bf16=24 字节/元素，double buffer 的 in/out 队列各按 2 份计，bf16 多出 bufXF32（升精度缓冲 4 字节）；tmp 按 2/4 字节保守估算 Tanh 临时空间（未调 GetTanhMaxMinTmpSize，备选方案，未对比松紧）。
- **空 tensor**：SetBlockDim(1) + TilingData 全 0，kernel 侧 Init/Process 对 totalLength==0 直接早退（不 InitBuffer）。
- **0 维 scalar**：GetStorageShape 为 0 维时视作 {1}，避免后续 CeilDiv 除零。
- **核数动态**：GetCoreNumAiv() 取 AIV 数，SetBlockDim(blockNum)（虚拟核数），禁硬编码。
- **blockNum==1 的语义**（UT 踩过）：ubTailOfFormerBlock 基于 blockFormer（对齐后的每核基量）而非 blockTail——blockNum==1 时"前核字段"仍按 blockFormer 填充但无核消费（kernel 里只有尾核分支生效），写 UT 断言时不要拿 blockTail 当前核尾部预期值。例：total=256、coreNum=48 ⟹ blockFormer=CeilAlign(CeilDiv(256,48),512)=512、blockNum=1、blockTail=256，则 ubTailOfFormerBlock=512（=blockFormer）、ubTailOfTailBlock=256（=blockTail）。
- 当前方案路线档位：recommended_fastpath。

## 完整代码

见上文「解决配方」——TilingData、常量、TilingFunc 主流程、GetUbParams、EnsureNotScalar、注册段全部 verbatim 取自该会话写入事件。

## 系统信息

- 硬件/软件：Ascend950PR（arch35），CANN 9.1.0；构建入口 `bash build.sh --soc=ascend950`（产出 libcust_opmaster_rt2.0.so 等 3 个 host 库）。
- UT 环境：op_host GoogleTest UT 在 x86 宿主执行，平台信息为 mock（ubSize/核数来自 mock fixture，fp16 用例推导 ubFormer=15744）——UT 验证的是切分算术正确性，不是真机参数。
- 相关文件：op_host tiling 实现（`BenchNoHarnessMishTilingFunc`、`GetUbParams`、`EnsureNotScalar`、`TilingParse` 注册段与 `BenchNoHarnessMishTilingData`，verbatim 见「解决配方」）；`tiling_key.h`（D_T_X TilingKey 路由）；op_common（CeilDiv/CeilAlign/FloorDiv/FloorAlign）。

## 预防

- 写 UT 断言先手算一遍 blockFormer/ubFormer 链路（特别是 blockNum==1 时的前核字段），不要拿直觉值。
- 新增 dtype 分支时同步更新 GetUbParams 的字节记账（漏算一个 buffer 就 UB 越界）。
- tiling 的 memset_s 之后逐字段填充，未填充字段即 0 契约（空 tensor 路径依赖它）。

## 验证信号

以下信号来自该 Mish 算子 tiling 的实现与 UT 会话：首版实现 → op_host UT 首轮 11/12（一处断言算错）→ 修正测试断言 → 12/12。

- 编译：`bash build.sh --soc=ascend950` exit 0，libcust_opmaster_rt2.0.so（tiling）等 3 个 host 库随包生成（来源平台：950；证据等级：tool_observed）。
- op_host UT：12/12 PASSED——infershape 5 例（fp16_3d / fp32_1d / bf16_4d / scalar_0dim / max_dims_8d）+ tiling 7 例（fp16_basic_1024 / large_multi_ub_loop / empty_tensor / scalar_0dim / invalid_dtype_double / shape_mismatch / fp16_multidim），宿主执行（来源平台：950；证据等级：tool_observed）。
- 首轮失败用例定位：tiling_fp16_multidim 期望 ubTailOfFormerBlock=256 实得 512——测试自身算术错误（前核字段基于 blockFormer），修正断言后通过（来源平台：950；证据等级：tool_observed）。
真实 NPU 上 tiling 行为（真实 ubSize/核数、入图路由）未验证。

## 失效边界

- UB_RESERVE=2048 与对齐粒度按本算子 buffer 布局定；换 buffer 数量/dtype 集合须重算 GetUbParams。
- UT 的 ubSize/核数来自 mock fixture（fp16 用例推导 ubFormer=15744），与真机值不同——UT 验证的是切分算术正确性，不是真机参数。
- 入图（graph）场景仅注册了 TilingParse 占位；动态 shape 路由未实测。

**未决项**：

- tmpBufferSize 保守估算与 GetTanhMaxMinTmpSize 精确值的差距未评估。
- 真机 tiling（真实平台参数 + 上板运行）待后续验证。
## 坏实践

- 硬编码核数或 ubSize；忽略 0 维/空 tensor（后续除零/空指针）。
- UT 断言失败先怀疑实现——本例实现语义自洽，是断言没按"前核字段基于 blockFormer"的契约推。

## 误诊

- 无。
