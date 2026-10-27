---
type: "Runbook"
title: "非对齐hiddenSize致507035需全链路8对齐"
description: hiddenSize 非 8 倍数时 Vector API（Mul/ReduceSum/Add/Sigmoid/Tanh 等）按非对齐 count 调用触发 507035 向量核异常；须引入 hiddenSize8/gateSize8/ubElementCount8 全链路对齐并零填充尾部，单一 API 改形态不充分。
tags: [task.debugging, topic.alignment, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 非对齐hiddenSize致507035需全链路8对齐

## 现象

在手写 AscendC（纯 AIV Vector API）GRU 算子的联调验证阶段（Ascend950PR / arch35，CANN 9.2.0），hiddenSize 为 8 非倍数的用例（hiddenSize=31，gateSize=3×31=93；hiddenSize=127，gateSize=381）在 Real 模式直接崩溃：`[FAIL] 流同步失败: 507035`（aclrtSynchronizeStream 返回 507035，向量核异常），无任何输出。同一批 hiddenSize 为 8 倍数（16/32/64/128）的用例全部正常。先单独把 `ReduceSum` 的模板参数从 `true` 改为 `false`（其"支持非对齐 count"的形态）重编译上板，仍然 507035——说明崩溃源不止 ReduceSum 一处。

## 适用场景

- 工程形态：手写 AscendC kernel（AIV），UB 内 Vector API 逐元素/规约计算。
- API 路径：`Mul`/`ReduceSum`/`Add`/`Adds`/`Muls`/`Sigmoid`/`Tanh`/`Duplicate`/`Cast` 等按元素计数调用的 Vector API；`TPipe::InitBuffer`；op_host TilingFunc 的 workspace/UB 规划。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有以隐藏维/任意用户维度作为 Vector API count 且该维度可能非 8 倍数的 kernel（RNN 门计算、任意 H 维向量段）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：hiddenSize 恒为 8 倍数的算子（无此问题）；具体是哪一个 API 触发异常未逐一定位（见遗留问题）；其他硬件代际的向量基本块大小未验证（未验证，推断机制可迁移、块大小须按架构确认）。

## 根因

kernel 内部所有按 hiddenSize 派生的计数（gateSize=3H、ubElementCount=B×H、K=H 的权重行宽）与 UB buffer 尺寸都直接使用原始维度。当 H 非 8 倍数（31/127）时，大量 Vector API 以非 8 倍数 count 执行（fp32 的向量基本块为 8 元素），触发向量核异常（507035）。单一 API 改形态（ReduceSum false）不够——`Add(gateSize)`、`Sigmoid/Mul/Tanh(ubElementCount)`、`Mul/ReduceSum(K)`、`Duplicate`、buffer 分配尺寸等全部需要对齐。

## 解决配方

核心做法：kernel 引入对齐维度三元组并把**计算 count、存储 stride、buffer 分配、host 侧 workspace/UB 规划**统一切到对齐值，尾部零填充。

1. kernel 字段与初始化（对齐维度）：

```cpp
    if(actualBatch_<0) actualBatch_=0;
    ubElementCount_=actualBatch_*hiddenSize_;
    // 对齐到 8（float 基本块大小），避免 Vector API 非对齐崩溃
    hiddenSize8_ = (hiddenSize_ + 7) & ~7;
    gateSize8_ = 3 * hiddenSize8_;
    ubElementCount8_ = actualBatch_ * hiddenSize8_;
```

（成员声明处新增 `int64_t hiddenSize8_=0, gateSize8_=0, ubElementCount8_=0;`，注释注明"用于 Vector API 调用，避免非对齐崩溃"。）

2. buffer 分配与 host 侧规划按对齐值（`InitBuffer` 一律 maxK8/giGh(=actualBatch×gateSize8)/hEl(=ubElementCount8)；tiling 侧 workspace 同步改对齐）：

```cpp
    // 6. 计算 Workspace — giAll + layerInput + layerOutput + 系统 workspace
    //    giAll 使用对齐后的 gateSize8 (kernel 内部使用对齐到8的门大小)
    //    layerInput:   S * batchPerCore * maxInputDim * 4  (层间传递)
    //    layerOutput:  S * batchPerCore * (numDirs * H) * 4  (层间传递)
    int64_t hiddenSize8 = (hiddenSize + 7) & ~7;  // 对齐到 8
    int64_t gateSize8 = NUM_GATES * hiddenSize8;
    int64_t giAllSize = seqLen * batchPerCore * gateSize8 * TYPE_SIZE_FLOAT;
    int64_t layerInputSize = seqLen * batchPerCore * maxInputDim * TYPE_SIZE_FLOAT;
    int64_t layerOutputSize = seqLen * batchPerCore * numDirs * hiddenSize * TYPE_SIZE_FLOAT;
```

3. 逐行 matmul 的 K 维对齐 + 尾部零填充（GM 读按原始 K，计算按 K8；填充用 SetValue 逐元素写，避免引入 V→MTE2 事件依赖）：

```cpp
    DataCopyExtParams wcp; wcp.blockCount=1;
    wcp.blockLen=static_cast<uint32_t>(K*sizeof(float));
    wcp.srcStride=0; wcp.dstStride=0;
    int64_t K8 = (K + 7) & ~7;  // 对齐到 8

    for (int64_t n = 0; n < N; n++) {
        LocalTensor<float> wData = inQueueW_.template AllocTensor<float>();
        DataCopyPad(wData, wGm[wOff + n * K], wcp, {false, 0, 0, 0});
        inQueueW_.EnQue(wData);
        wData = inQueueW_.template DeQue<float>();
        // 零填充尾部元素 (K 到 K8), 处理非对齐 K
        for (int64_t i = K; i < K8; i++) {
            wData.SetValue(i, 0.0f);
        }

        Mul<float>(mulData, xVecBase[xVecOff], wData, static_cast<int32_t>(K8));
        ReduceSum<float, false>(reduceDst, mulData, reduceTmp, static_cast<int32_t>(K8));
        float sum = reduceDst.GetValue(0);
        resultBase.SetValue(resultOff + n, sum);

        inQueueW_.FreeTensor(wData);
    }
```

4. 其余调用点同步：门计算读取与 `Sigmoid/Tanh/Mul/Add/Adds/Muls` 改用 `ubElementCount8_`/`gateSize8_`；gi workspace 存储与读取 stride 改 `gateSize8_`；`InitHPrev`/`StoreHn` 的 `Adds/Duplicate/Cast` count 改对齐值（GM 侧 blockLen 仍按原始 hiddenSize 写出，不污染真实输出）；h0 加载先零填充再 DataCopyPad 实际数据。

## 设计决策

- 为什么选"全链路对齐 + 零填充"而不是逐 API 依赖非对齐支持：实测仅改 `ReduceSum<false>` 无效，说明异常面覆盖多个 API；统一对齐是最小一致解。
- 尾部零填充为什么用 SetValue 逐元素写：早期版本用 `Duplicate + SetFlag/WaitFlag<V_MTE2>` 预填零，引入了额外的 V→MTE2 事件依赖，排查输出错值时被一并简化为 SetValue 循环（K8-K ≤ 7 次，开销可忽略），终版保留 SetValue 写法。
- GM 侧读写保持原始长度（blockLen=K×4 / hiddenSize×4）：对齐只发生在 UB/计算层，输入输出张量布局不变，调用方无感知。
- 当前方案路线档位：recommended_fastpath（正确性优先；对齐带来的尾部冗余计算与零填充开销未测）。

## 预防

- 设计时列出所有以用户维度派生的 Vector API count 与 buffer 尺寸清单；凡是可能非 8 倍数的维度，预先定义 `<dim>8` 并统一替换。
- tiling 侧的 workspace/UB 规划与 kernel 的 InitBuffer 必须用同一套对齐公式，防止 host/kernel 布局漂移。
- ST 用例矩阵必须包含"非对齐维度"组（如 H=31/127、inputSize=63/127），只测对齐 shape 会漏掉整类崩溃。

## 验证信号

来源说明：以下信号来自该 GRU 算子联调验证会话：非对齐 hiddenSize 崩溃定位后实施全链路 8 对齐，重编译安装上板复测。

- 修复前：hiddenSize=31/127 用例 `流同步失败: 507035`，无输出；hiddenSize=16/32/64/128 用例全部 PASS（来源平台：950；证据等级：tool_observed）。
- 中间尝试（仅 `ReduceSum<false>`）：重编译上板仍 507035，单一 API 改形态不充分（来源平台：950；证据等级：tool_observed）。
- 修复后（全链路对齐）：hiddenSize=31/127 不再崩溃（无 507035）；对齐用例回归 5/5 PASS、fp32 子集 14/15 PASS（来源平台：950；证据等级：tool_observed）。
注意：全链路对齐落地后非对齐用例的**数值**仍全错（y MERE=3.97~6.64），那是第二个独立根因（门数据存储布局与 H8 步长访问错位），本卡只闭合"崩溃"根因（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "8 元素基本块"按 fp32 与本硬件（arch35 AIV）成立；其他 dtype（fp16 的 16 元素半块等）或架构的块大小须重新确认对齐粒度（未验证，推断机制可迁移）。
- 零填充假设填充值（0）不参与有效语义（乘法被乘以零权重行、加法门段尾不写出）；若算子对 padding 敏感（如取 max/除法），须换掩码策略。
- 对齐使 UB 占用按 `<dim>8` 上取整增长，接近 UB 上限的大 batch 场景须复核 tiling 的 UB 规划。

**未决项**：

- 优先项：507035 的确切触发 API 未逐一定位（是 Mul/Add/Sigmoid/Tanh 中的哪一个先炸），可在最小 kernel 上按 API 二分复现并追到指令层。
- 对齐引入的尾部冗余计算对性能的影响未测量。
## 坏实践

- 只把"看起来最可疑"的一个 API 改成非对齐支持形态就上板——本例 ReduceSum 单改无效，浪费一轮编译安装。
- 把对齐值同时用于 GM 写出（blockLen 也按 H8 写），会写越界污染相邻输出；对齐只限 UB/计算层。
