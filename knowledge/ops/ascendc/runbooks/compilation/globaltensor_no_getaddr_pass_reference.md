---
type: "Runbook"
title: GlobalTensor 无 GetAddr，辅助函数收引用
description: "CANN 9.1.0 arch35 设备侧 AscendC::GlobalTensor<T> 不提供 GetAddr() 成员，(__gm__ void*)tensor.GetAddr() 报 \"no member named 'GetAddr'\"；需要全局缓冲的辅助函数应直接接收 const GlobalTensor<T>& 并用 gm[offset]（下标返回偏移视图）做 DataCopyPad 源。反向同样成立：DataCopyPad 的 GM 侧重载不接受 __gm__ T* 裸指针，反向转换须用 SetGlobalBuffer(ptr, len) 构造 GlobalTensor。附 GM 标量读取可复用配方。复现增量：混合 dtype workspace 场景的正解变体——Init 保存 GM_ADDR 原始 __gm__ uint8_t* 基址，按字节偏移逐段 SetGlobalBuffer 绑定独立 GlobalTensor<T>（float scores + half 缓冲共存的布局）。复现增量：更轻的直接读法——GlobalTensor::GetValue(const uint32_t offset)（asc/include/basic_api kernel tensor 头，tikcfw 下 grep 不到），低频 GM 标量（bias[c] 等）免一趟 DataCopyPad；half 返回值先 cast float 再比较；高频成片读取仍走向量搬运。"
tags: [task.debugging, topic.compilation, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# GlobalTensor 无 GetAddr，辅助函数收引用

## 现象

自定义算子 kernel 首次设备侧编译（op_host/op_api 已通过，asc_opc 进入 INFERCHANNEL 阶段编译 op_kernel）时，kernel 头文件里多个调用点同时报错。报错原文：

```text
Exception: An error occurred during compile phases of CompileStage.INFERCHANNEL, msg is
In file included from /.../op_kernel/bench_no_harness_moe_finalize_routing_arch35.cpp:34:
/.../op_kernel/arch35/bench_no_harness_moe_finalize_routing.h:329:72:
  error: no member named 'GetAddr' in 'AscendC::GlobalTensor<int>'
    int32_t rowIdx = ReadInt32Scalar((__gm__ void*)expandedSrcToDstGM_.GetAddr(),
                                                   ~~~~~~~~~~~~~~~~~~~ ^
/.../op_kernel/arch35/bench_no_harness_moe_finalize_routing.h:361:66:
  error: no member named 'GetAddr' in 'AscendC::GlobalTensor<int>'
        int32_t expertId = ReadInt32Scalar((__gm__ void*)efsrGM_.GetAddr(),
```

当时 kernel 类里把标量读取辅助函数声明为 `ReadInt32Scalar(GM_ADDR gmBase, int64_t offsetElems)`（接收裸 GM_ADDR），调用点为了把成员 `GlobalTensor` 传进去，写了 `(__gm__ void*)expandedSrcToDstGM_.GetAddr()`——而本版本设备侧 `GlobalTensor<T>` 没有 `GetAddr()` 成员。

## 适用场景

- 工程形态：registry-invoke 自定义算子工程，op_kernel 侧（arch35/AscendC 模板 kernel）编写从 GM 读取单个标量（索引、scale 值）的辅助函数。
- API 路径：`AscendC::GlobalTensor<T>`（无 GetAddr）；`gm[offsetElems]` 下标返回偏移视图；`DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams<T>)`；`LocalTensor::GetValue`；`PipeBarrier<PIPE_ALL>`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——设备侧 GlobalTensor 接口行为，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：host 侧（op_host/op_api 编译上下文）的 GlobalTensor 是否提供 GetAddr 未在轨迹中涉及；其他 CANN 版本的设备侧接口未验证。

## 根因

CANN 9.1.0 的 arch35 设备编译中 `AscendC::GlobalTensor<T>` 不提供 `GetAddr()` 成员，`(__gm__ void*)tensor.GetAddr()` 无法编译。根本设计问题在于辅助函数签名选择了裸 `GM_ADDR`：GM_ADDR 与 GlobalTensor 之间没有公开的转换通道，逼着调用点发明 GetAddr。正确做法是让辅助函数直接接收 `const GlobalTensor<T>&`，用 `gm[offsetElems]`（下标运算返回偏移后的 GlobalTensor 视图）作为 DataCopyPad 的源。

## 解决配方

修复前（辅助函数收 GM_ADDR，调用点用 GetAddr 取基址——编译失败）：

```cpp
template <typename D_T_MAIN, int DROP_PAD_MODE>
__aicore__ inline int32_t BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE>::ReadInt32Scalar(
    GM_ADDR gmBase, int64_t offsetElems)
{
    // 标量读取 int32 索引值: DataCopyPad 4 字节 + GetValue
    LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(int32_t);  // 4 字节
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    GlobalTensor<int32_t> gmTmp;
    gmTmp.SetGlobalBuffer((__gm__ int32_t*)(gmBase));
    DataCopyPad(indexLocal, gmTmp[offsetElems], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
    return indexLocal.GetValue(0);
}

// 调用点（编译失败处）：
//     int32_t rowIdx = ReadInt32Scalar((__gm__ void*)expandedSrcToDstGM_.GetAddr(),
//                                      indexPos);
```

修复后（辅助函数直接接收 GlobalTensor 引用；两个完整实现 verbatim）：

```cpp
    // ===== 辅助方法 =====
    // 标量读取 int32 索引值
    __aicore__ inline int32_t ReadInt32Scalar(const GlobalTensor<int32_t>& gm, int64_t offsetElems);
    // 标量读取主 dtype 值并转 float
    __aicore__ inline float ReadScaleScalar(const GlobalTensor<D_T_MAIN>& gm, int64_t offsetElems);
```

```cpp
template <typename D_T_MAIN, int DROP_PAD_MODE>
__aicore__ inline int32_t BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE>::ReadInt32Scalar(
    const GlobalTensor<int32_t>& gm, int64_t offsetElems)
{
    // 标量读取 int32 索引值: DataCopyPad 4 字节 + GetValue
    LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(int32_t);  // 4 字节
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(indexLocal, gm[offsetElems], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
    return indexLocal.GetValue(0);
}

template <typename D_T_MAIN, int DROP_PAD_MODE>
__aicore__ inline float BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE>::ReadScaleScalar(
    const GlobalTensor<D_T_MAIN>& gm, int64_t offsetElems)
{
    // 标量读取 scale (原 dtype) 并转 float
    LocalTensor<D_T_MAIN> scaleLocal = scaleBuf_.Get<D_T_MAIN>();
    LocalTensor<float> scaleFp32 = scaleBufFp32_.Get<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(D_T_MAIN);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<D_T_MAIN> padParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal, gm[offsetElems], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
    // Cast 到 fp32 (scale 数量=1)
    Cast(scaleFp32, scaleLocal, RoundMode::CAST_NONE, 1);
    PipeBarrier<PIPE_ALL>();
    return scaleFp32.GetValue(0);
}
```

调用点同步改为直接传成员 GlobalTensor：

```cpp
    // 2. 标量读取 expanded_src_to_dst_row[indexPos]
    int32_t rowIdx = ReadInt32Scalar(expandedSrcToDstGM_, indexPos);
```

```cpp
        // 标量读取 expert_for_source_row[i, k]
        int32_t expertId = ReadInt32Scalar(efsrGM_, i * topK_ + k);
```

```cpp
        // 标量读取 scales[i, k] → scaleVal (转 float)
        float scaleVal = ReadScaleScalar(scalesGM_, i * topK_ + k);
```

可复用模式（GM 标量读取配方）：kernel 内读 GM 单个标量不能直接解引用，须 DataCopyPad 到 UB 的 LocalTensor 再 `GetValue(0)`；DataCopyExtParams 用 `blockCount=1, blockLen=sizeof(T), stride=0`；读后 `PipeBarrier<PIPE_ALL>()` 再取值；非 int 类型标量先 Cast 到 fp32（count=1）再 GetValue。标量缓冲用 `TBuf<TPosition::VECCALC>` 固定 64B（满足 32B 对齐）。

## 反向形态：DataCopyPad 不收 __gm__ 裸指针，须 SetGlobalBuffer 构造（独立复现，多张量 TensorList 算子，同平台同版本）

多张量场景下 `Process` 先用 `GetTensorAddr(tensorIdx, x1Ptr_)` 取回**裸指针** `__gm__ T*`，再把裸指针一路传给 `CopyIn/CopyOut`，在 `CopyIn` 里调 `DataCopyPad(x1Local, x1Gm[progress * ubFactor_], copyParams, {false, 0, 0, 0})`。编译算子包时报（原文）：

```text
op_kernel/arch35/<op>.h:142:5: error: no matching function for call to 'DataCopyPad'
asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h:1270:45: note:
  candidate template ignored: could not match 'GlobalTensor<T>' against '__gm__ float'
asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h:1438:24: note:
  candidate template ignored: could not match 'LocalTensor<T>' against '__gm__ float'
...（多条 candidate 均不匹配；另有 requires 3/4 arguments 的重载不匹配）
```

同一文件内 CopyIn×3、CopyOut 共 4 处 DataCopyPad 调用全部报错，随后链接阶段连带 `ld.lld: error: cannot open ..._<hash>.o`（.o 未产出）。

机制：`DataCopyPad` 的全部重载的 GM 侧形参类型是 `GlobalTensor<T>`（或 LocalTensor），模板推导不接受 `__gm__ T*` 裸指针。经典写法里 `x1Gm_[i]` 下标返回 `GlobalTensor<T>&`（子张量引用）所以能编过；把形参类型改成 `__gm__ T*` 后，`x1Gm[i]` 变成裸指针解引用（元素值），类型不匹配直接编不过。即 GM_ADDR/裸指针与 GlobalTensor 之间只有**单向**转换通道：`SetGlobalBuffer(__gm__ T*, len)` 可以从裸指针构造 GlobalTensor；GlobalTensor 自身无 GetAddr() 退回裸指针。

修复（verbatim，分块入口构造 GlobalTensor，形参类型改引用）：

```cpp
template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelForeachAddcdivScalar<T, BUFFER_MODE>::ProcessTensorRange(
    __gm__ T* x1Ptr, __gm__ T* x2Ptr, __gm__ T* x3Ptr, __gm__ T* yPtr,
    int64_t startOffset, int64_t elemCount)
{
    // 从原始指针创建 GlobalTensor（DataCopyPad 需要 GlobalTensor 类型）
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> x3Gm;
    GlobalTensor<T> yGm;
    x1Gm.SetGlobalBuffer(x1Ptr + startOffset, elemCount);
    x2Gm.SetGlobalBuffer(x2Ptr + startOffset, elemCount);
    x3Gm.SetGlobalBuffer(x3Ptr + startOffset, elemCount);
    yGm.SetGlobalBuffer(yPtr + startOffset, elemCount);

    // 单张量内 UB 分块循环
    int64_t loopCount = (elemCount + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1)
            ? (elemCount - ubFactor_ * i) : ubFactor_;
        CopyIn(i, currentNum, x1Gm, x2Gm, x3Gm);
        Compute(currentNum);
        CopyOut(i, currentNum, yGm);
    }
}
```

CopyIn/CopyOut 声明与实现形参同步改为 `const GlobalTensor<T>& x1Gm ...` / `GlobalTensor<T>& yGm`（`DataCopyPad(x1Local, x1Gm[progress * ubFactor_], ...)` 里的 `x1Gm[i]` 恢复为子张量引用语义）。设计取舍：按 (指针, 元素数) 在分块入口一次性构造，边界信息（startOffset/elemCount）就在手边，kernel 内其余路径仍可保留裸指针语义；形参用引用而非值，避免逐分块拷贝 tensor 对象。

## 设计决策

- 选"辅助函数收 `const GlobalTensor<T>&`"而不是"补一个 GM_ADDR 基址参数从入口透传"：入口的 GM_ADDR 在 Init 里已经 `SetGlobalBuffer` 成了带类型的成员 GlobalTensor，直接传引用避免重复转换，且 `gm[offset]` 语义清晰。
- 原实现里函数内 `gmTmp.SetGlobalBuffer((__gm__ int32_t*)(gmBase))` 每次调用重建 GlobalTensor——修复后连这步也省掉，少一次指针转换。

## 预防

- kernel 侧自定义辅助函数避免裸 GM_ADDR 签名：要么传 `const GlobalTensor<T>&`，要么传入口 GM_ADDR 并在 Init 一次性 SetGlobalBuffer 成员化。
- 不要假设 GlobalTensor 有 GetAddr/GetPtr 之类的基址访问器——本版本没有；需要偏移访问用 `operator[]`。

## 验证信号

来源说明：该算子 kernel 首轮设备侧编译失败-修复闭环；反向形态（__gm__ 裸指针直传 DataCopyPad）来自另一独立多张量算子的编译修复会话。

- 修复前：`error: no member named 'GetAddr' in 'AscendC::GlobalTensor<int>'`（kernel 头 329/361 等多调用点，INFERCHANNEL 阶段，TBE traceback）（来源平台：950；证据等级：tool_observed）
- 修复后：同构建命令下该错误消失，错误阶段推进到 FATBIN/模板实例化（另一独立问题）；后续问题解决后全量构建退出码 0、kernel binary 与算子包生成（来源平台：950；证据等级：tool_observed）
- 反向形态修复前：`build.sh --soc=ascend950` 报 4 处 `error: no matching function for call to 'DataCopyPad'`，candidate 全部 `could not match 'GlobalTensor<T>' against '__gm__ float'`，连带 `ld.lld: cannot open ..._<hash>.o`（来源平台：950；证据等级：tool_observed）
- 反向形态修复后：同命令编译成功、kernel 二进制正常产出；重装算子包后多张量用例精度 MERE≈1.7e-09~1.6e-08（16M 元素大 shape 用例），搬运语义正确（来源平台：950；证据等级：tool_observed）

## 失效边界

- "GlobalTensor 无 GetAddr"基于 CANN 9.1.0 arch35 设备编译实测；其他版本未验证（机制推断：若后续版本补齐 GetAddr，本卡配方仍正确可用）。
- 标量读取配方每读一个标量伴随两次全流水屏障，性能非最优——正确性优先的迭代阶段可用，性能敏感路径应改为批量搬运（本轨迹未做性能优化验证）。

**未决项**：

- 标量读取的批量优化（把 K 个索引/scale 一次搬入 UB 再逐个 GetValue）未实现未验证，属迭代二性能方向。
## 坏实践

- `(__gm__ void*)tensor.GetAddr()` 式的"从类型化视图退回裸指针"写法。
- 在标量读取函数里每次 SetGlobalBuffer 重建 GlobalTensor。
- 增量（反向形态）：把 `__gm__ T*` 裸指针直接传给 `DataCopyPad` 做 GM 侧实参（重载集不收，candidate 全部 ignored）；为绕编译错误把 `DataCopyPad` 换成手写 `__gm__` 指针循环搬运——失去硬件搬运指令的掩码/对齐语义。

## 后续增量（2026-09-17，并行策展批：GlobalTensor::GetValue 直接读 GM 标量——比 DataCopyPad 中转更轻的低频路径）

另一独立轨迹并入独立 trace（DepthwiseConv2d 类算子 kernel，Ascend950PR，CANN 9.2.0，编译级验证）：上文"GM 标量读取配方"（DataCopyPad→UB→LocalTensor::GetValue）之外存在**更直接的一条路**——`GlobalTensor` 自带 `GetValue`：

```text
asc/include/basic_api/kernel_*.h:150  __inout_pipe__(S) PrimType GetValue(const uint32_t offset) const;
```

用法（verbatim，kernel 内每 (n,c) 通道读一次 bias、每 (kh,kw) tap 读一次权重）：

```cpp
    // GM 侧：直接按元素偏移取标量，免去为 1 个 half 开 DataCopy
    half biasHalf = biasGm_.GetValue(c);
    float biasFloat = static_cast<float>(biasHalf);
    // UB 侧同构：weightLocal.GetValue(idx)（LocalTensor::GetValue）
    half wScalar = weightLocal.GetValue(static_cast<uint32_t>(kh * KW_ + kw));
    if (static_cast<float>(wScalar) == 0.0f) { continue; }   // half 比较先过 float
```

要点与边界：

- **定位陷阱**：`class GlobalTensor`/`GetValue` 在 tikcfw/ascendc 目录 grep 不到——基础 tensor 接口在 `asc/include/basic_api/`，换目录即命中（同款"搜索范围错了"误诊：初判"AscendC 可能不支持 kernel 内 GM 标量访问"）。
- **只适合低频**：每次一条标量访问，高频/成片读取仍应走 DataCopy 向量搬运（本卡上文配方），标量循环读的代价未测量。
- **half 返回值**参与比较/算术前先 `static_cast<float>`（half 无直接比较的可靠路径）。
- **运行时边界（引用姊妹卡实测）**：GM 标量直接访问对常规输入张量可用，但 **GM workspace 区域的标量访问在 950/arch35 触发 errcode 264 'The address for scalar to access GM is invalid'**（6 种规避全败，见 [errcode 264 workspace 标量 GM 访问卡](errcode_264_workspace_scalar_gm_access.md)）；bf16 标量 GetValue/SetValue 还会丢 nan 位模式（见 [arch35 bf16 标量 GM 读写丢 nan 位模式卡](../precision/arch35_bf16_gm_scalar_nan_lost.md)）；同 kernel 内 SetValue 写回再 GetValue 会读旧值（见 [GM SetValue 写回后 GetValue 读到旧值卡](../precision/gm_setvalue_getvalue_reads_stale_within_kernel.md)）。写方向 SetValue(index, value) 两参签名见 [GM 标量写须 SetValue 两参卡](globaltensor_scalar_write_setvalue_two_args.md)。

复现增量验证信号：

- 头文件事实锚点：`asc/include/basic_api/kernel_*.h:150 __inout_pipe__(S) PrimType GetValue(const uint32_t offset) const;`（GlobalTensor 成员）（来源平台：950；证据等级：tool_observed）。
- 含 `biasGm_.GetValue(c)` 与 `weightLocal.GetValue(...)` 的 kernel 经 `cmake --build . --target all binary package` 编译通过并产出 3 个 ascend950 `.o`（来源平台：950；证据等级：tool_observed）。NPU 实际访存代价/流水行为未测量（编译级验证）。

## 增量（混合 dtype workspace 的原始指针 + 字节偏移布局变体）

另一独立 trace（GQA 算子 kernel，Ascend950PR / arch35，CANN 9.1.0）以第二种面目复现"GlobalTensor 无 GetAddr"：workspace 里要**混放不同 dtype 的段**（float 的 scores 中间结果 + half 的 q/k/v/y 缓冲），初版把整个 workspace 绑定成一个 `GlobalTensor<D_T_Q> wsGm`，之后各处用 `wsGm.GetAddr()` 取基址做字节偏移再转 `( __gm__ half*)` ——每个调用点报 `error: no member named 'GetAddr' in 'AscendC::GlobalTensor<half>'`（共 6 处）；`grep -rn "GetAddr\|GetAddress" $ASCEND_HOME_PATH/x86_64-linux/ascendc/include/` 在 GlobalTensor 类上确认无任何取址成员。

该场景（多 dtype 段共存）不适合"辅助函数收引用"形态，正解是**Init 保存 GM_ADDR 原始 `__gm__ uint8_t*` 基址 + 纯字节偏移布局 + 每段独立 SetGlobalBuffer**（kernel 入口拿到的 `GM_ADDR workspace` 本身就是原始 `__gm__` 指针，不需要从一个已绑定的 GlobalTensor 反取地址）：

```cpp
    GlobalTensor<D_T_Q> queryGm;
    GlobalTensor<D_T_Q> keyGm;
    GlobalTensor<D_T_Q> valueGm;
    GlobalTensor<D_T_Q> yGm;
    __gm__ uint8_t* wsPtr_ = nullptr;  // raw workspace pointer
...
    queryGm.SetGlobalBuffer((__gm__ D_T_Q*)query);
    keyGm.SetGlobalBuffer((__gm__ D_T_Q*)key);
    valueGm.SetGlobalBuffer((__gm__ D_T_Q*)value);
    yGm.SetGlobalBuffer((__gm__ D_T_Q*)y);
    wsPtr_ = (__gm__ uint8_t*)workspace;

    // 每核 workspace 布局（纯字节偏移，与段 dtype 无关）
    int64_t S = tiling_.S;
    int64_t S_kv = tiling_.S_kv;
    int64_t D = tiling_.D;
    int64_t hpc = tiling_.headsPerCore;
    int64_t qBufBytes = S * D * sizeof(D_T_Q);
    int64_t kvBufBytes = S_kv * D * sizeof(D_T_Q);
    int64_t scoresBytes = hpc * S * S_kv * sizeof(float);
    int64_t attnBytes = hpc * S * S_kv * sizeof(D_T_Q);
    int64_t yBufBytes = S * D * sizeof(D_T_Q);

    perCoreWsBytes_ = qBufBytes + kvBufBytes + kvBufBytes + scoresBytes + attnBytes + yBufBytes;
    int64_t base = blockIdx_ * perCoreWsBytes_;
    qBufOff_ = base;
    kBufOff_ = qBufOff_ + qBufBytes;
    vBufOff_ = kBufOff_ + kvBufBytes;
    scoresWsOff_ = vBufOff_ + kvBufBytes;
    attnWsOff_ = scoresWsOff_ + scoresBytes;
    yBufOff_ = attnWsOff_ + attnBytes;
```

使用侧（每段按字节偏移 + 目标 dtype 绑定到独立 GlobalTensor，保持 AscendC API 的类型化用法）：

```cpp
    // 2. Q @ K^T → scores (float)
    GlobalTensor<half> qGm;
    qGm.SetGlobalBuffer((__gm__ half*)(wsPtr_ + qBufOff_));
    GlobalTensor<half> kGm;
    kGm.SetGlobalBuffer((__gm__ half*)(wsPtr_ + kBufOff_));
    int64_t scoresByteOff = scoresWsOff_ + localIdx * S * S_kv * sizeof(float);
    GlobalTensor<float> scoresGm;
    scoresGm.SetGlobalBuffer((__gm__ float*)(wsPtr_ + scoresByteOff));

    mmQK.SetTensorA(qGm);
    mmQK.SetTensorB(kGm, true);
    mmQK.IterateAll(scoresGm);
```

要点：字节偏移与段 dtype 解耦，加减算错时容易对账（每段 bytes 可打印核对）；tiling 侧 `GetWorkspaceSizes` 只声明一块连续工作区，混合 dtype 段在 kernel 内自行按字节切分。判别/定位：报 `no member named 'GetAddr' in 'AscendC::GlobalTensor<T>'` → 全局 grep `GetAddr` 清点调用点；对每处问"这里要的是基址还是元素偏移"，改用保存的 `__gm__ uint8_t*` 基址 + 字节偏移表达。

坏实践（独立 trace 增补）：

- 把 workspace 绑定成单一 dtype 的 GlobalTensor 再"借"它的类型做所有段访问：float 段用 `wsGm[i]` 语义会按绑定 dtype 的元素大小偏移，隐含 dtype 混淆。
- 用 `(void*)` 强转绕过 `__gm__` 修饰：丢失地址空间信息，device 编译器会拒绝或生成错误代码。

验证信号（独立 trace 增补，GQA 算子迭代一编译循环，Ascend950PR / CANN 9.1.0）：

- 修复前：构建在 kernel 头 6 处 GetAddr 调用点报 `error: no member named 'GetAddr' in 'AscendC::GlobalTensor<half>'`（来源平台：950；证据等级：tool_observed）。
- CANN include 全树 grep 确认 GlobalTensor 无 GetAddr/GetAddress 取址成员（来源平台：950；证据等级：tool_observed）。
- 改为 wsPtr_ 原始指针 + 字节偏移方案并重写后重建，GetAddr 类错误全部消失，构建推进到 wrapper 生成阶段（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 本增量只证明编译层面 API 不存在与替代写法编译通过；字节布局的正确性（越界/重叠）属运行期问题，该会话未上板验证（NPU 驱动不可用），采用时须以最小用例核各段偏移。
- `GetPhyAddr`（物理地址，用于 debug/特殊通路）在 9.1.0 是否存在未在本会话验证，不要据此卡推断（见库内 GetPhyAddr const 卡）。