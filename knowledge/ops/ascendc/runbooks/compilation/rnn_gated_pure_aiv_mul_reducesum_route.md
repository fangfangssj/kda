---
type: "Runbook"
title: RNN 门控算子纯 AIV 路线：Mul+ReduceSum 矩阵乘
description: LSTM/GRU 类门控算子骨架可用纯 AIV 路线替代 Cube Matmul 主路线：flat 权重 offset 访问、Mul+ReduceSum 逐行矩阵乘、门存储 8 对齐、H/C 常驻 UB、batch-only 多核切分；NPU 实测精度 MERE≈1e-6。复现增量：为何必须走 Mul+ReduceSum——外积法在 k 循环内对 Cast 产物逐元素 GetValue(k) 不可靠（非零输入输出困在输入量级、MERE≈1.0，全零输入全对），标量读只用于 ReduceSum 结果 GetValue(0) 或 DataCopyPad 直载数据；含 5 条实测失败替代路线清单与 host 侧权重转置。复现增量（GRU 救援重写实例）：MIX 路线（SyncAll 死锁 + 原地 Vector 化全 0）失败后完全重写为纯 AIV 一次通过（ST 5/5 + 独立验收），TQue 逐行装权重双缓冲、H_prev 常驻 UB、PreComputeGi 走 GM workspace、binary_info 校验 coreType=VectorCore 再上板。
tags: [task.implementation, topic.compute.composite]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# RNN 门控算子纯 AIV 路线：Mul+ReduceSum 矩阵乘

## 现象

在开发一个 LSTM 自定义算子（4 门 i/f/g/o、多层/双向/投影可选、fp32/fp16/bf16）的迭代一骨架时，面临路线选择：设计方案主路线是 Cube Matmul（`Matmul` 高级 API + MatmulApiTiling），但其集成复杂度被评估为 HIGH（tiling 侧需引入 Matmul tiling 配置、kernel 侧需 REGIST_MATMUL_OBJ）；而同仓已有一个结构相同的 GRU 算子用纯 AIV 模式（Vector 核 `Mul` + `ReduceSum` 实现矩阵乘）验证可行。迭代一的验收标准是"编译通过 + 骨架正确"，因此按设计方案 §5.4 预留的回退条款采用纯 AIV 路线落地：编译一次通过（3 个 kernel `.o` + 575KB 自定义算子包），UT 26/26，ST Real（NPU）4/4 且精度 `MERE=8.74e-07` 量级。

## 适用场景

- 工程形态：自定义算子 kernel 实现（手写 AscendC，`kernel_operator.h`），RNN/门控类复合算子的骨架迭代；aclnn 两段式接口 + TilingKey 模板（DTYPE_ID × HAS_BIAS）。
- API 路径：AscendC Vector API（`Mul`/`ReduceSum`/`Add`/`Sigmoid`/`Tanh`/`Duplicate`/`Adds`/`Cast`）、`DataCopyPad` + `DataCopyExtParams`、`TQue`/`TBuf`、`SetFlag/WaitFlag`（MTE2↔V 事件同步）、`REGISTER_TILING_DEFAULT`/`GET_TILING_DATA_WITH_STRUCT`；不使用 Matmul API、无 REGIST_MATMUL_OBJ。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510 / arch35）+ CANN cann-9.2.0（与 frontmatter `platforms` 一致；含 NPU 上板实测）。
- 适用粒度：`operator_family`（LSTM/GRU/RNN 类门控递归算子族）。
- 路线优先级：safety_fallback（相对 Cube Matmul 主路线的回退路线；适用条件与相对代价见下）。fallback 声明：本路线为迭代骨架/低风险首选，非最终性能路线；大 K（输入/隐藏维数大）时逐行 ReduceSum 的吞吐低于 Cube，性能验收前应切换或对比 Cube Matmul 路线（相对代价未量化——本迭代未做性能 profiling，无成对数据）。
- 明确不适用/未验证场景：大 inputSize/hiddenSize 的性能未评测；projSize>0 投影分支未实测（本迭代用例 projSize=0）；多层/双向代码路径已实现但 L1 用例上板验证属后续迭代；非 RNN 结构的通用 GEMM 不建议用逐行 ReduceSum。

## 根因/解决配方（实现要点）

单句归因：把 4 门线性变换从"Cube Matmul 一次矩阵乘"改为"Vector 核逐行 Mul+ReduceSum"，配合 flat 权重布局与 8 对齐门存储，可在不引入 Matmul tiling 的前提下让 RNN 骨架一次编译通过并达到 fp32 级精度。

机制链：

1. **flat 权重布局**：所有层/方向权重拼接为 1D tensor（proto 单 INPUT），kernel 端 offset 计算：

```cpp
    // weight_ih: 拼接 1D, layer 0: (4H, inputSize), layer k>0: (4H, numDirs*H)
    __aicore__ inline int64_t GetWeightIhOffset(int64_t layer, int64_t dir) {
        int64_t offset = 0;
        for (int64_t l = 0; l < layer; l++) {
            int64_t inDim = (l == 0) ? inputSize_ : numDirections_ * hiddenSize_;
            offset += numDirections_ * gateSize_ * inDim;
        }
        int64_t inDim = (layer == 0) ? inputSize_ : numDirections_ * hiddenSize_;
        return offset + dir * gateSize_ * inDim;
    }
    // weight_hh: 拼接 1D, 每个 (4H, H)  — projSize=0
    __aicore__ inline int64_t GetWeightHhOffset(int64_t layer, int64_t dir) {
        return (layer * numDirections_ + dir) * gateSize_ * hiddenSize_;
    }
    // bias: 拼接 1D, 每个 (4H)
    __aicore__ inline int64_t GetBiasOffset(int64_t layer, int64_t dir) {
        return (layer * numDirections_ + dir) * gateSize_;
    }
```

2. **逐行矩阵乘**：`VectorMatmul` 对权重逐行（每行 = 一个门的输出维 n）做 `DataCopyPad` 加载 → 零填充到 8 对齐 → `Mul` → `ReduceSum` → 按门分段存储（`gateIdx*hiddenSize8 + withinGate`）：

```cpp
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessLstm<D_T, HAS_BIAS>::VectorMatmul(
    LocalTensor<float>& resultBase, int64_t resultOff,
    const LocalTensor<float>& xVecBase, int64_t xVecOff,
    const GlobalTensor<D_T>& wGm, int64_t wOff,
    int64_t N, int64_t K, int64_t hiddenSize, int64_t hiddenSize8)
{
    LocalTensor<float> mulData = mulBuf_.Get<float>();
    LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();

    DataCopyExtParams wcp; wcp.blockCount=1;
    wcp.blockLen=static_cast<uint32_t>(K*sizeof(float));
    wcp.srcStride=0; wcp.dstStride=0;
    int64_t K8 = (K + 7) & ~7;  // 对齐到 8

    for (int64_t n = 0; n < N; n++) {
        LocalTensor<float> wFloat;
        if constexpr (std::is_same_v<D_T, float>) {
            LocalTensor<float> wData = inQueueW_.template AllocTensor<float>();
            DataCopyPad(wData, wGm[wOff + n * K], wcp, {false, 0, 0, 0});
            inQueueW_.EnQue(wData);
            wData = inQueueW_.template DeQue<float>();
            for (int64_t i = K; i < K8; i++) wData.SetValue(i, 0.0f);
            wFloat = wData;
            Mul<float>(mulData, xVecBase[xVecOff], wFloat, static_cast<int32_t>(K8));
            ReduceSum<float, false>(reduceDst, mulData, wFloat, static_cast<int32_t>(K8));
            float sum = reduceDst.GetValue(0);
            int64_t gateIdx = n / hiddenSize;
            int64_t withinGate = n % hiddenSize;
            int64_t storeOff = resultOff + gateIdx * hiddenSize8 + withinGate;
            resultBase.SetValue(storeOff, sum);
            inQueueW_.FreeTensor(wData);
        } else {
            LocalTensor<D_T> wDataRaw = castFloatBuf_.Get<D_T>();
            DataCopyExtParams wcp_d; wcp_d.blockCount=1;
            wcp_d.blockLen=static_cast<uint32_t>(K*sizeof(D_T));
            wcp_d.srcStride=0; wcp_d.dstStride=0;
            DataCopyPad(wDataRaw, wGm[wOff + n * K], wcp_d, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventId_);
            WaitFlag<HardEvent::MTE2_V>(eventId_);
            wFloat = reduceTmpBuf_.Get<float>();
            Cast<float>(wFloat, wDataRaw, RoundMode::CAST_NONE, static_cast<int32_t>(K));
            SetFlag<HardEvent::V_MTE2>(eventId2_);
            WaitFlag<HardEvent::V_MTE2>(eventId2_);
            PipeBarrier<PIPE_V>();
            for (int64_t i = K; i < K8; i++) wFloat.SetValue(i, 0.0f);
            Mul<float>(mulData, xVecBase[xVecOff], wFloat, static_cast<int32_t>(K8));
            ReduceSum<float, false>(reduceDst, mulData, wFloat, static_cast<int32_t>(K8));
            float sum = reduceDst.GetValue(0);
            int64_t gateIdx = n / hiddenSize;
            int64_t withinGate = n % hiddenSize;
            int64_t storeOff = resultOff + gateIdx * hiddenSize8 + withinGate;
            resultBase.SetValue(storeOff, sum);
        }
    }

    // 零填充每门尾部 (hiddenSize 到 hiddenSize8), 确保门间不串数据
    int64_t numGates = N / hiddenSize;  // 4 for LSTM
    for (int64_t g = 0; g < numGates; g++) {
        for (int64_t i = hiddenSize; i < hiddenSize8; i++) {
            resultBase.SetValue(resultOff + g * hiddenSize8 + i, 0.0f);
        }
    }
}
```

3. **时间步循环（H/C 常驻 UB，门控四行式）**：`i=σ(gi_i+gh_i+bhh_i)`、`f=σ(...)`、`g=tanh(...)`、`o=σ(...)`、`c_new=f*c_prev+i*g`、`h=o*tanh(c_new)`；双向时 dir=1 反向遍历 `t=S-1→0`（完整函数，含 bias_hh 门对齐重排与 fp32/fp16 输出分支，核心段 verbatim）：

```cpp
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessLstm<D_T, HAS_BIAS>::TimeStepLoop(int64_t layer, int64_t dir)
{
    int64_t K = hiddenSize_;
    int64_t wOff = GetWeightHhOffset(layer, dir);
    bool isLastLayer = (layer == numLayers_ - 1);
    LocalTensor<float> hPrevUb = hPrevTBuf_.Get<float>();
    LocalTensor<float> cPrevUb = cPrevTBuf_.Get<float>();
```

```cpp
    for (int64_t step = 0; step < seqLen_; step++) {
        int64_t t = (dir == 1) ? (seqLen_ - 1 - step) : step;

        // 加载 gi[t] (使用 gateSize8_ stride)
        LocalTensor<float> giUb = inQueueGi_.template AllocTensor<float>();
        int64_t giOff = t * actualBatch_ * gateSize8_;
        DataCopyPad(giUb, gmGiAll_[giOff], gicp, {false, 0, 0, 0});
        inQueueGi_.EnQue(giUb);
        giUb = inQueueGi_.template DeQue<float>();

        // 计算 gh = U_hh @ h_prev (gate-aware storage: [i(H8), f(H8), g(H8), o(H8)])
        LocalTensor<float> ghUb = ghBuf_.Get<float>();
        for (int64_t b = 0; b < actualBatch_; b++) {
            VectorMatmul(ghUb, b * gateSize8_, hPrevUb, b * hiddenSize8_,
                         gmWhh_, wOff, gateSize_, K, hiddenSize_, hiddenSize8_);
        }

        // 门控计算 — 逐 batch 处理 (门顺序: i, f, g, o)
        LocalTensor<float> iUb=bufI_.Get<float>(), fUb=bufF_.Get<float>();
        LocalTensor<float> gUb=bufG_.Get<float>(), oUb=bufO_.Get<float>();
        LocalTensor<float> tmpUb=bufTmp_.Get<float>(), tanhCUb=bufTanhC_.Get<float>();
        LocalTensor<uint8_t> sigTmp=bufSigmoidTmp_.Get<uint8_t>();

        for (int64_t b = 0; b < actualBatch_; b++) {
            int64_t gOff = b * gateSize8_;   // gi/gh 中该 batch 的门数据起始偏移
            int64_t hOff = b * hiddenSize8_; // h_prev/c_prev 中该 batch 的偏移

            // i = sigmoid(gi_i + gh_i)
            Add<float>(iUb, giUb[gOff], ghUb[gOff], hiddenSize8_);
            Sigmoid<float>(iUb, iUb, sigTmp, hiddenSize8_);
            // f = sigmoid(gi_f + gh_f)
            Add<float>(fUb, giUb[gOff + hiddenSize8_], ghUb[gOff + hiddenSize8_], hiddenSize8_);
            Sigmoid<float>(fUb, fUb, sigTmp, hiddenSize8_);
            // g = tanh(gi_g + gh_g)
            Add<float>(gUb, giUb[gOff + 2*hiddenSize8_], ghUb[gOff + 2*hiddenSize8_], hiddenSize8_);
            Tanh<float>(gUb, gUb, sigTmp, hiddenSize8_);
            // o = sigmoid(gi_o + gh_o)
            Add<float>(oUb, giUb[gOff + 3*hiddenSize8_], ghUb[gOff + 3*hiddenSize8_], hiddenSize8_);
            Sigmoid<float>(oUb, oUb, sigTmp, hiddenSize8_);

            // c_new = f * c_prev + i * g
            Mul<float>(tmpUb, fUb, cPrevUb[hOff], hiddenSize8_);   // fc = f * c_prev
            Mul<float>(tanhCUb, iUb, gUb, hiddenSize8_);            // ig = i * g
            Add<float>(tanhCUb, tmpUb, tanhCUb, hiddenSize8_);      // c_new = fc + ig

            // h = o * tanh(c_new)
            Tanh<float>(tmpUb, tanhCUb, sigTmp, hiddenSize8_);      // tanh_c = tanh(c_new)
            Mul<float>(tmpUb, oUb, tmpUb, hiddenSize8_);            // h = o * tanh_c

            // 更新 hPrev, cPrev (逐 batch 写回常驻 UB)
            Adds<float>(hPrevUb[hOff], tmpUb, 0.0f, hiddenSize8_);
            Adds<float>(cPrevUb[hOff], tanhCUb, 0.0f, hiddenSize8_);

            // 存储输出: 最后一层→gmY_, 中间层→gmLayerOut_（分支略，见源文件）
        }

        inQueueGi_.FreeTensor(giUb);
        SetFlag<HardEvent::MTE3_MTE2>(eventId_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId_);
    }
}
```

4. **主流程（层→方向→时间步，层间 workspace 传递）**：

```cpp
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessLstm<D_T, HAS_BIAS>::Process()
{
    if(actualBatch_<=0) return;

    for (int64_t layer = 0; layer < numLayers_; layer++) {
        int64_t inputDim = (layer == 0) ? inputSize_ : (numDirections_ * hiddenSize_);

        for (int64_t dir = 0; dir < numDirections_; dir++) {
            PreComputeGi(layer, dir, inputDim);
            InitHPrev(layer, dir);
            InitCPrev(layer, dir);
            TimeStepLoop(layer, dir);
            StoreHn(layer, dir);
            StoreCn(layer, dir);
        }

        // 层间数据传递: layer_output → layer_input (非最后一层)
        if (layer < numLayers_ - 1) {
            CopyLayerOutputToLayerInput();
        }
    }
}
```

5. **kernel 入口（TilingKey 模板 dispatch，fp16/bf16 经模板参数实例化、内部 Cast fp32 计算）**：

```cpp
template <uint32_t DTYPE_ID, uint32_t HAS_BIAS>
__global__ __aicore__ void bench_no_harness_lstm(
    GM_ADDR x, GM_ADDR weightIh, GM_ADDR weightHh,
    GM_ADDR biasIh, GM_ADDR biasHh, GM_ADDR weightHr,
    GM_ADDR h0, GM_ADDR c0,
    GM_ADDR y, GM_ADDR hn, GM_ADDR cn,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessLstmTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessLstmTilingData, tilingData, tiling);

    if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_FP32) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<float, HAS_BIAS> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    } else if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_FP16) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<half, HAS_BIAS> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    } else if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_BF16) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<bfloat16_t, HAS_BIAS> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    }
}
```

（`Init`/`PreComputeGi`/`InitHPrev`/`InitCPrev`/`StoreHn`/`StoreCn`/`CopyLayerOutputToLayerInput` 完整实现见源文件 `op_kernel/arch35/<op>.h`，机制同上：8 对齐 buffer 分配、gi 预计算写 workspace、h0/c0 Cast 加载、hn/cn 写回；上文内联的是本路线的核心机制函数。）

## 设计决策

- 为何选纯 AIV 而非 Cube Matmul（4 条依据）：设计方案 §5.4 明确允许的回退方案；迭代计划风险表预登记"Matmul 模块集成失败 → 回退"；同仓 GRU 项目已验证该模式可行；迭代一验收标准为编译通过，AIV 方案风险最低。
- 权重用 flat 拼接单 Tensor 而非 TensorList：对齐同仓 GRU 已验证实现，避免 DYNAMIC_INPUT 的 codegen/tiling 复杂度（TensorList 为后续迭代可选升级）。
- 多核切分选 batch-only（每核处理全部层/方向）：避免跨核同步（时间步与层间有串行依赖，按 batch 切分是唯一无同步切分维度）；`blockDim = min(coreNumAiv, B)`，尾核 actualBatch 单独计算。
- 维度全部 8 对齐（`hiddenSize8_/gateSize8_/K8`，`(x+7)&~7`）：float 基本块大小 8，非对齐调用 Vector API 会崩溃；代价是每门尾部零填充，门存储采用 `[i(H8), f(H8), g(H8), o(H8)]` 分段布局防止串数据。
- fp16/bf16 输入统一 Cast→fp32 内部计算、输出 Cast 回原 dtype（CAST_RINT）：精度按 fp32 通路保障；跨流水线复用 `castFloatBuf_` 用 V_MTE2/MTE2_V 双向事件保护。
- H_prev/C_prev 常驻 UB（TBuf 而非每次往返 GM）：时间步循环的状态依赖只在 UB 内推进，每层每方向重新初始化。
- 当前路线属 safety_fallback（相对 Cube Matmul 主路线）。

## 系统信息

- 工具链：CANN cann-9.2.0、`bash build.sh --soc=ascend950 -j8`（退出码 0，仅 CANN 头 deprecation 警告）。
- 目标平台：Ascend950PR（ascend950 / DAV_3510 / arch35），coreType=AIV（非 MIX），无 REGIST_MATMUL_OBJ。
- 相关文件：`op_kernel/arch35/<op>.h`（核心实现）、`op_kernel/<op>_arch35.cpp`（模板入口）、`op_kernel/arch35/<op>_tiling_data.h`、`op_kernel/arch35/<op>_tiling_key.h`（DTYPE_ID(3)×HAS_BIAS(2)=6 模板组合，IS_BIDIR/HAS_PROJ/IS_BATCH_FIRST 为运行时标志）、`op_host/arch35/<op>_tiling.cpp`（blockDim=min(coreNumAiv,B)、workspace offset）、`op_api/`、`op_graph/<op>_proto.h`（权重单 INPUT 注册）。

## 验证信号

段首来源说明：本验证来自该算子迭代一：AIV 路线实现的编译、UT、ST 全链路。

- 编译：`build.sh --soc=ascend950 -j8` 退出 0，3 个 kernel `.o`（6 种 TilingKey 模板组合按 dtype 去重后的实例化产物），`custom_opp_ubuntu_x86_64.run` 575KB（来源平台：950；证据等级：tool_observed）。
- UT：op_host 26/26 通过（InferShape 8 + Tiling 18）（来源平台：950；证据等级：tool_observed）。
- ST Real（NPU 上板）：L0 4/4 PASS，L0-001（单层单向 bias=true）y `MERE=8.74e-07, MARE=1.32e-04`、hn `MERE=4.52e-07`（阈值 mere<0.05/mare<0.5，取自 proto.yaml），CPU golden 自测 4/4，Mock 回归 4/4（来源平台：950；证据等级：tool_observed）。
性能：未验证——迭代一未做性能验收，无 profiling 数据，本卡不含任何性能结论。

## 失效边界

- 大 inputSize/hiddenSize（大 K）时逐行 ReduceSum 吞吐低于 Cube 路线，性能差距未量化（未验证；性能验收阶段应与 Cube Matmul 路线做同算子成对对比后再定去留）。
- 多层/双向/projSize>0 代码路径已实现但未上板验证（L1 用例属后续迭代）；projSize=0 用例才实测过。
- `hiddenSize8_` 填充使每门多占 `(H8-H)` 元素，H 远大于 8 时浪费可忽略，H=1 等极端值时 padding 占比高（未做该场景性能评估）。
- 推断可迁移：同代 Vector API 语义的其他平台（未验证，推断可迁移）。

**未决项**：

- 与 Cube Matmul 路线的同算子成对性能对比（性能验收阶段的第一个测量点）。
- 多层/双向/proj 的 L1 用例上板验证（后续迭代范围）。
## 坏实践

- 骨架迭代直接上 Cube Matmul 全量配置（MatmulApiTiling + REGIST_MATMUL_OBJ），把编译风险与算法正确性验证耦合在一起——先 AIV 验证数学正确性、再切 Cube，两条风险解耦。
- 非 8 对齐维度直接调 Vector API（Mul/ReduceSum/Sigmoid/Tanh）——非对齐基本块会崩溃，必须 `(x+7)&~7` + 尾部零填充。

## 误诊

无。

## 复现增量：为何必须走 Mul+ReduceSum——外积法逐元素 GetValue(k) 不可靠（CANN 9.2.0）

纯 AIV MLA 前处理算子（4 次 VEC matmul）最初用**外积累加法**实现矩阵乘：把输入行 Cast 到 fp32 的 `xVec` 后，在 k 循环里逐个 `xVec.GetValue(k)` 读标量做广播乘。NPU 实测 21 条非零输入用例精度完全不达标（L0-01：query MERE=1.025、不匹配 65534/65536、actual 困在 [-1,1] 与输入量级一致而 golden 可达数百）——**累加根本没有发生**；2 条全零输入用例独立执行全对（排除框架/数据流问题）；CPU golden 自测 7/7 全过。

**根因**：外积法在 k 循环内对 `Cast` 产生的 V 流水线 LocalTensor 逐元素做标量读取 `xVec.GetValue(k)`，该读法在 ascend950 上不可靠（k>0 时读到 0 或错误值；确切微架构失效模式——返回 0 还是旧值——未单独探测）；同环境参考门控算子的 `VectorMatmul` 从不逐元素 GetValue 于 Cast 输出——逐输出 n 加载权重行、`Mul`（向量乘）+ `ReduceSum` 后只 `GetValue(0)` 读归约结果，其 `GetValue(i)` 仅用于 `DataCopyPad` 直接加载的 bias 数据。标量读的使用纪律：**只用于归约结果的 GetValue(0) 或 DataCopyPad 直接加载的数据，不在循环内逐元素读 Cast/V 流水线产物**（与 S↔V 交接 hazard 家族同族，见 `vector_scalar_ub_handoff_requires_explicit_completion` 卡）。

**修复**：切内积法（上文路线的逐行矩阵乘）+ host 侧权重转置（`w.t().contiguous()`，[K,N]→[N,K] 使每行 W[n,:K] 连续；kernel 内转置需非连续 DataCopyPad，实测非对齐搬运仍出错）。修复后同用例输出值立即逃离 [-1,1]（actual=2.34e+02 等），全部根因修复后 query MERE=8.41e-06、四输出全 PASS 且复跑一致。

**实测失败的替代路线（不要重试）**：逐 k `Adds`+`GetValue(0)`（设备超时 507035）；逐 k mask+`ReduceSum`+`GetValue(0)`（更慢，仍超时）；8 元素块 `GetValue(k&7)`（值变化但仍错）；`bfloat16_t::ToFloat()`（无效）；非对齐偏移 `DataCopyPad` 直读 GM（引入 NaN）。

- 修复前 L0-01：query MERE=1.025051e+00、不匹配 65534/65536、actual 困在 [-1,1]；21 条非零输入用例同签名全失败（来源平台：950；证据等级：tool_observed）。
- 全零输入用例独立执行全 PASS（MERE=0、输出全零）（来源平台：950；证据等级：tool_observed）。
- 切内积法+权重转置后（其余 bug 未修时）：输出值逃离 [-1,1]（actual=2.34e+02/-5.61e+35 等）——累加通路恢复（来源平台：950；证据等级：tool_observed）。
- 全部根因修复后 L0-01：query MERE=8.41e-06、四输出全 PASS、复跑一致（来源平台：950；证据等级：tool_observed）。

边界：GetValue(k) 失效的确切微架构机制未探测（优先项）；大 batch（B=128，bsTotal=2048）下内积法 kernel 超时（507035）——本路线只解决正确性、大规模需 Cube Matmul 路线（成对性能对比未做）；其他 NPU 架构未验证（未验证，推断机制相同）。

## 增量（GRU 救援重写实例：从失败 MIX 路线完全重写一次通过）

同仓 GRU 算子（bench_no_harness_gru，Ascend950PR + CANN 9.2.0）从**失败路线的救援**角度复用本路线：MIX 模式（Matmul API + SyncAll）在 taskRation 1:2 下死锁/崩溃（四种同步变体全败），原地沿用 MIX 骨架做 Vector 化又输出恒 0（十余种变体无解，另见原地 Vector 化卡）——最终以"纯 AIV 完全重写"一次通过：ST 5/5 PASS（y MERE 2.32e-07~6.01e-06、hn MERE 3.72e-07~4.54e-06，阈值 5e-02/5e-01），独立验收重跑 5/5。GRU 实例相对本卡 LSTM 主体的差异化要点：

**1）矩阵乘核心（TQue 逐行装权重双缓冲 + ReduceSum<float, true>）**：

```cpp
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessGru<D_T, HAS_BIAS>::VectorMatmul(
    LocalTensor<float>& resultBase, int64_t resultOff,
    const LocalTensor<float>& xVecBase, int64_t xVecOff,
    const GlobalTensor<D_T>& wGm,
    int64_t N, int64_t K)
{
    LocalTensor<float> mulData = mulBuf_.Get<float>();
    LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();

    DataCopyExtParams wcp; wcp.blockCount=1;
    wcp.blockLen=static_cast<uint32_t>(K*sizeof(float));
    wcp.srcStride=0; wcp.dstStride=0;

    for (int64_t n = 0; n < N; n++) {
        LocalTensor<float> wData = inQueueW_.template AllocTensor<float>();
        DataCopyPad(wData, wGm[n * K], wcp, {false, 0, 0, 0});
        inQueueW_.EnQue(wData);
        wData = inQueueW_.template DeQue<float>();

        Mul<float>(mulData, xVecBase[xVecOff], wData, static_cast<int32_t>(K));
        ReduceSum<float, true>(reduceDst, mulData, reduceTmp, static_cast<int32_t>(K));
        float sum = reduceDst.GetValue(0);
        resultBase.SetValue(resultOff + n, sum);

        inQueueW_.FreeTensor(wData);
    }
}
```

- 权重行经 `TQue<TPosition::VECIN, 2>`（双缓冲）装入，MTE2→V 同步由队列自动完成——与失败版"整块权重缓存进 UB"（挤占容量且与失败骨架耦合）相对。
- 点积结果用 `ReduceSum<float, true>`（每次调用重设 mask）——与失败版的 `isSetMask=false` 是关键差异之一（受控对照未做，见遗留问题）。
- `H_prev` 常驻 `TBuf<TPosition::VECCALC>`：时序循环隐状态每步 `Adds(hPrevUb, tmpUb, 0.0f, n)` 就地更新，不走 GM 往返（GM 往返版实测 507035 超时）。
- `PreComputeGi` 把 gi 全量预计算落 GM workspace（随 S 增长 UB 放不下），时间步循环再经 `TQue<VECIN>` 搬回。

**2）上板前先核编译产物通道**：`build/op_kernel/ascendc_kernels/binary/ascend950/<op>/<Kernel>_<hash>.json` 应为 `coreType: VectorCore`、`core_type: AIV`、`intercoreSync: 0`、`magic: RT_DEV_BINARY_MAGIC_ELF_AIVEC`——确认编译通道真的切到纯 AIV，再上板。

**3）host tiling 配套**：`GetCoreNumAiv()` 切核（纯 Vector 无 AIC 参与）、`blockDim = min(batch × numDirections, coreNum)`、workspace 总量 = 用户区 + `GetLibApiWorkSpaceSize()`（纯 AIV 也需要系统 workspace，ReduceSum 等内部使用）。

设计取舍（GRU 实例）：为何完全重写而不是原地改造——原地沿用 MIX 骨架（整块权重 UB 缓存 + GM workspace 中转 + 逐点 SetValue）输出恒 0 且变体无解；重写数据流一次通过。

验证信号（独立 trace 增补，GRU 纯 AIV 重写后的构建、安装与 ST 回归，Real NPU 模式，Ascend950PR / CANN 9.2.0）：

- 构建安装：干净构建 3 个 tiling key 的 kernel 全部生成，`.run --force` 安装 SUCCESS；`BenchNoHarnessGru_*.json` 显示 `coreType: VectorCore`、`core_type: AIV`、`intercoreSync: 0`、`magic: RT_DEV_BINARY_MAGIC_ELF_AIVEC`（来源平台：950；证据等级：tool_observed）。
- 精度回归：5/5 L0 用例 PASS——y MERE 2.32e-07~6.01e-06、hn MERE 3.72e-07~4.54e-06（阈值 MERE 5e-02 / MARE 5e-01）（来源平台：950；证据等级：tool_observed）。
- 独立验收重跑（`bash run.sh` Real 模式）：5/5 PASS，验收报告通过（来源平台：950；证据等级：tool_observed）。
- 相对代价（定性对照）：首个纯 Vector 版曾因 GM 往返与逐点 SetValue 出现单用例 10 分钟级超时，重写版（TQue 逐行 + H_prev 常驻 UB + workspace 中转 gi）同批用例秒级完成（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 负载形态边界：M=1（每核每步单行向量）× K≤128 实测通过；大 M/大 K 的吞吐经济性未验证，重型矩阵乘不应套用。
- dtype 边界：fp32 实测通过；fp16/bf16 的 Cast 路径（`RoundMode::CAST_RINT`）已实现但未在 ST 验证。
- 功能边界：迭代一范围为单层单向无 bias；多层/双向/bias 的 flat 权重拼接与 TilingKey 扩展属后续迭代（多层 flat 拼接的 161002 坑见同批对应卡）。

遗留问题（独立 trace 增补）：纯 AIV 版的性能画像（IU/带宽利用率、与理论 Cube 路线的相对代价）未测量；`ReduceSum<float, true>` vs `false` 的受控对照未做。