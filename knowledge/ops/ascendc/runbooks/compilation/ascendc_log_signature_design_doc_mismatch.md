---
type: "Runbook"
title: AscendC Log签名与设计文档不符：按已装头文件核对
description: CANN 9.1.0 adv_api 的 Log（自然对数）只有 (dst,src) 与 (dst,src,calCount) 签名、无 sharedTmpBuffer 参数（临时空间框架申请；4 参版本属 Log2）；isReuseSource 是模板参数而非函数参数，原地调用须 Log<T,true>(dst,dst,uint32_t(count))；count 为 uint32_t 与 basic_api 的 int32_t 不同——设计文档 API 映射表不可照抄，按已装头文件核对。
tags: [task.implementation, topic.implementation.api_layer, topic.activation]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# AscendC Log签名与设计文档不符：按已装头文件核对

## 现象

两条独立轨迹在实现含 `Log` 调用的 AscendC kernel（Mish 激活、foreach 范数）时，都发现设计文档的 API 映射表与实际头文件签名不符：

**差异一（sharedTmpBuffer 归属）**：Mish 算子设计文档的 API 映射表写着：

```
| Step5: Log | Log\<half\> | (dst, src, sharedTmpBuffer, calCount) | ✅ DAV_3510 | half/float; **不支持地址重叠**; 需临时空间 |
```

而实际 kernel 代码写的是 3 参调用 `Log<ComputeT>(tmp3, tmp2, countU32)`（无 sharedTmpBuffer）。同一算子里 `Tanh` 却是 4 参 `Tanh<ComputeT>(tmp2, tmp1, sharedTmp, countU32)`。两者必有一个不对：要么代码漏传了 Log 的临时空间参数，要么设计文档写错了 API 签名。若按设计文档"补全"成 4 参，反而会把正确代码改坏。

**差异二（isReuseSource 形态与 count 类型）**：foreach 范数算子设计文档提到 `isReuseSource` 参数（像是函数参数），但核对 CANN 头文件（`asc/include/adv_api/math/log.h`）发现实际签名是**模板参数**；且 LocalTensor 级的 Log 只有 adv_api 这一个版本（basic_api 只有寄存器版 `Log(U& dstReg, U& srcReg, MaskReg& mask)`），basic_api 的 `Exp/Abs/Sqrt` 则是 `int32_t count` 且没有 isReuseSource——count 类型在两套 API 间不同（adv_api 为 `uint32_t`）。

## 适用场景

- 工程形态：AscendC kernel 内使用数学类向量 API（Log/Log2/Log10/Tanh/Exp 等），或按设计文档/API 映射表核对调用签名；原地复用缓冲的元素级数学函数链。
- API 路径：`$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/math/log.h`（Log/Log2/Log10）、`tanh.h`；AscendC 官方 API 文档（asc-devkit docs/api/context/Log.md）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——API 签名属 CANN 公共头文件契约。
- 路线优先级：recommended_fastpath（3 参调用 + 需要原地复用时显式模板参数即正确用法）。
- 明确不适用 / 未验证场景：其他 CANN 版本签名可能增补（以已装头文件为准）；`__NPU_ARCH__` 门控列表之外的架构上 Log 是否可用未验证（log.h 中有 arch 白名单）；basic_api 的其他函数签名不属本卡范围。

## 根因

CANN 9.1.0 中"以 e 为底的 Log"**没有** sharedTmpBuffer 形参，其所需临时空间走"接口框架申请"路径；带 sharedTmpBuffer 的 4 参重载属于 Log2（及文档中 Log10 亦无 4 参）；`isReuseSource` 是模板参数（默认 `false`）而非函数参数；`calCount` 是 `uint32_t`。设计文档把 Log2 的签名与"函数参数式"的写法安到了 Log 头上。头文件证据（`adv_api/math/log.h`，verbatim 关键段）：

```cpp
namespace AscendC {

#pragma begin_pipe(V)
/*
 * @brief dst = log(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] calCount, amount of data to calculate
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
{
    // Only for AI Vector Core.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
    if ASCEND_IS_AIC {
        return;
    }
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 3510 || \
                              __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
    LogImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
#endif
}

/*
 * @brief dst = log(src), calcalate
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Log<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

/*
 * @brief dst = log2(src)
 * @ingroup Log
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] calCount, amount of data to be calculated
 */
```

即 Log 的全部重载：`(dst, src, calCount)` 与 `(dst, src)`；紧接着的 4 参（含 sharedTmpBuffer）版本是 log2。官方文档口径一致："由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间……临时空间支持开发者通过 sharedTmpBuffer 入参传入和接口框架申请两种方式"——Log 走的是后者。

## 判别方法

遇到"设计文档签名与代码不一致"时的核对顺序：

1. 直接读已安装头文件：`find $ASCEND_HOME_PATH -name "log.h" -path "*adv_api*"`，看重载列表（上面的 verbatim 段）。
2. 对照官方 API 文档（asc-devkit `docs/api/context/Log.md`）确认 Log 与 Log2 的参数表分立。
3. 用编译事实收口：3 参 `Log` 在 arch35 编译通过且上板结果正确 → 代码没错、文档错。

## 解决配方

**常规调用保持 3 参**（这是正确形态）：

```cpp
    Log<ComputeT>(tmp3, tmp2, countU32);                           // Step5: tmp3 = log(1+exp(-|x|))
```

**原地调用（dst==src）显式 `isReuseSource=true` 模板参数，count 显式转 uint32_t**（foreach 范数 NORM_TYPE=5 分支 `y = (Σ|x_i|^p)^(1/p)`、`|x|^p = exp(p·log(|x|))`，verbatim 终版）：

```cpp
    } else if constexpr (NORM_TYPE == 5) {
        // 通用 p 范数: y = (sum(|x_i|^p))^(1/p)
        // |x|^p = exp(p * log(|x|))
        // 注意: p<0 时要求输入元素非零 (Host 侧已校验), log(|x|) 对 x≠0 有效
        Abs(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        // in-place Log: isReuseSource=true 允许 API 复用 src 缓冲区, 避免额外临时空间
        Log<ComputeT, true>(workBuf0, workBuf0, static_cast<uint32_t>(chunkElem));
        PipeBarrier<PIPE_V>();
        Muls(workBuf0, workBuf0, scalarVal_, chunkElem);
        PipeBarrier<PIPE_V>();
        Exp(workBuf1, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        ReduceSum<ComputeT>(resultBuf, workBuf1, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
```

修复前（原地 Log 走默认 isReuseSource=false，隐式临时缓冲、浪费 UB）：

```cpp
        Abs(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        Log(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
```

需要显式管理临时空间的场景（UB 紧张、想共享一块 tmp）改用 **Log2 并自行乘 ln2 换底**，或直接用 Log2 的 4 参重载：

```cpp
    // 等价换底：ln(z) = log2(z) * ln(2)——仅当确需显式 sharedTmpBuffer 时使用
    Log2<ComputeT>(tmp3, tmp2, sharedTmp, countU32);
    Muls<ComputeT>(tmp3, tmp3, static_cast<ComputeT>(0.6931471805599453f), countU32);
```

设计文档勘误：把 Log 行的签名改为 `(dst, src[, calCount])`，"需临时空间"注明"框架申请"；isReuseSource 标注为模板参数。配套 host 侧防御（p<0 语义告警，同次迭代补充改动）：

```cpp
    uint32_t normType = GetNormType(scalarVal);

    // p<0 (非 -inf) 时, 要求所有输入元素非零, 否则 log(0)=-inf 导致结果异常
    // Host 侧无法逐元素检查, 仅记录告警; 调用方需保证 p<0 时输入无零元素
    if (normType == 5 && scalarVal < 0.0f) {
        OP_LOGW(context, "scalar p=%f is negative (NORM_TYPE=5), all input elements must be non-zero", scalarVal);
    }
```

## 设计决策

- 选 3 参 Log 而不是换 Log2+换底：少一步 Muls、少占一块 sharedTmp；框架内部申请的 tmp 对本算子 UB 预算无感（实测 UB 切分正常）。
- sharedTmpBuffer 仍按 Tanh 的需求保留并整块传入——`pipe.InitBuffer(sharedTmpBuf, tilingData->tmpBufferSize)` 只服务 Tanh；不能因为"Log 不吃 sharedTmp"就删掉整块分配。
- 原地调用显式 `Log<ComputeT, true>`：isReuseSource=true 允许 API 复用 src 缓冲、避免额外临时空间；非原地调用（dst != src，如 merge 阶段 `Log(workBuf1, workBuf0, 8)`）保持默认 false 语义正确，不改。参考对标：CANN 内置 euclidean_norm 实现中 `Pattern::Reduce` 版 ReduceSum 也用 `/*isReuseSource=*/true`，佐证原地/复用场景显式开启是该 API 家族的预期用法。
- count 类型：adv_api Log 的 `calCount` 是 `uint32_t`，kernel 内 `chunkElem` 为 `int64_t` 时需显式转换；basic_api 的 Exp/Abs/Sqrt 为 `int32_t count` 且无 isReuseSource，两套 API 不可混抄。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 工具链：bisheng（AscendC kernel 编译），CANN 9.1.0 头文件 `asc/include/adv_api/math/log.h`、`basic_api/kernel_operator_vec_unary_intf.h`；AscendC 官方 API 文档（asc-devkit）。
- 目标平台：Ascend950PR / arch3510。
- 相关文件（两条轨迹）：`op_kernel/arch35/bench_no_harness_mish.h`（Mish 九步链的 Log/Tanh 调用）、`op_kernel/arch35/bench_no_harness_foreach_norm.h`（ProcessChunk 的 NORM_TYPE==5 分支）、`op_host/arch35/bench_no_harness_foreach_norm_tiling.cpp`（p<0 告警）。

## 验证信号

来源说明：以下信号来自两条独立轨迹的 kernel 实现核对与上板验证会话（Mish 激活、foreach 范数，均为 Ascend950PR + CANN 9.1.0）。

- 头文件事实：`adv_api/math/log.h` 中 Log 仅有 `(dst,src,calCount)`/`(dst,src)` 两个重载（模板参数 `isReuseSource=false`、`calCount` 为 `uint32_t`），4 参 sharedTmpBuffer 版本出现在紧随其后的 log2 段（来源平台：950；证据等级：tool_observed）。
- 官方文档事实：asc-devkit Log.md 的签名表与头文件一致，并注明临时空间"支持入参传入与框架申请两种方式"（来源平台：950；证据等级：tool_observed）。
- 编译与上板（Mish 轨迹）：3 参 `Log<ComputeT>(tmp3, tmp2, countU32)` 在 arch35 编译通过；FP16/FP32/BF16 三路径 ST 全部通过（FP32 大 shape MERE=1.13e-07），Log 计算正确性实证（来源平台：950；证据等级：tool_observed）。
- 编译与上板（foreach 范数轨迹）：原地 `Log<ComputeT, true>(workBuf0, workBuf0, static_cast<uint32_t>(chunkElem))` 修改后全量重编译、打包、安装成功；ST Real 全量回归中所有 NORM_TYPE=5 用例（p=0.5/-0.5/-1/-2/3/5 等，含 FP16/BF16 dtype）全部 PASS（MERE 0.00e+00 ~ 4.51e-07，阈值 1.22e-04/9.77e-04/7.81e-03）；全量 73/74 通过，唯一失败为非连续 Tensor 用例（与本改动无关）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 签名绑定 CANN 9.1.0；后续版本若为 Log 增补 sharedTmpBuffer 重载，以已装头文件为准。
- Log 在 log.h 内有 `__NPU_ARCH__` 白名单（2201/2002/3510/5102/3003/3113）且 AI Vector Core 限定；白名单外架构的可用性未验证。
- 对非原地调用也开 isReuseSource=true——src 数据可能被复用破坏，只在 dst==src 时使用。
- Log2+换底方案引入一次额外 Muls 的精度/性能影响未实测（仅作备选记录）。

**未决项**：

- 无（Mish 轨迹闭环完整；foreach 轨迹 `isReuseSource=true` 相对默认路径的具体收益未做 profiling 测量，如需量化须补 profiling，不作预期收益宣称）。
## 坏实践

- 按设计文档把 `Log`"补"成 4 参——编译失败或错调 Log2 语义。
- 按设计文档的"函数参数"理解去传 isReuseSource——实际是模板参数，会编译失败或被忽略。
- 因为"Log 不吃 sharedTmp"就删掉整块 sharedTmpBuffer 分配——Tanh 还在用。
- 从 basic_api 照抄 `int32_t count` 的调用习惯到 adv_api Log——count 类型不同，需显式转换。

## 误诊

- 无（两轨迹均为核对一次定位：文档错、代码对）。

## 增量（4 参误用的编译报错原文与两处调用点同批修复）

另一独立轨迹并入同 foreach 范数工程的构建层错误证据（BenchNoHarnessForeachNorm，ascend950/arch35 + CANN 9.1.0，TBE opc 工具链），补齐 4 参误用的实际编译报错与"全局 grep 同一 API 其他调用点"的修复纪律：

**1. 触发形态（no matching function，候选重载明示 2/3 参）**：通用 p-范数分支用 `|x|^p = exp(p·log|x|)` 实现时，两处调用按旧版/其他系列 API 记忆带了第 4 参 `isReuseSource`：

```text
.../op_kernel/arch35/<op_name>.h:208:9: error: no matching function for call to 'Log'
        Log(workBuf0, workBuf0, chunkElem, true);
        ^~~
/workspace/.../x86_64-linux/asc/include/adv_api/math/log.h:37:24: note: candidate function template not viable:
  requires 3 arguments, but 4 were provided
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
/workspace/.../x86_64-linux/asc/include/adv_api/math/log.h:58:24: note: candidate function template not viable:
  requires 2 arguments, but 4 were provided
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
.../op_kernel/arch35/<op_name>.h:444:13: error: no matching function for call to 'Log'
            Log(resultBuf, resultBuf, 1, true);
            ^~~
2 errors generated.
```

**2. 修复（两处同批去掉第 4 参，verbatim before/after）**：

```cpp
// before（ProcessChunk 通用 p 范数分支，逐 chunk 路径）
        Abs(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        Log(workBuf0, workBuf0, chunkElem, true);
        PipeBarrier<PIPE_V>();
        Muls(workBuf0, workBuf0, scalarVal_, chunkElem);
        PipeBarrier<PIPE_V>();
        Exp(workBuf1, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        ReduceSum<ComputeT>(resultBuf, workBuf1, workBuf1, chunkElem);
        PipeBarrier<PIPE_V>();
```

```cpp
// after（同一分支，3 参正确调用）
        Abs(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        Log(workBuf0, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        Muls(workBuf0, workBuf0, scalarVal_, chunkElem);
        PipeBarrier<PIPE_V>();
        Exp(workBuf1, workBuf0, chunkElem);
        PipeBarrier<PIPE_V>();
        ReduceSum<ComputeT>(resultBuf, workBuf1, workBuf1, chunkElem);
        PipeBarrier<PIPE_V>();
```

```cpp
// before（MergeAndOutput 通用 p 最终计算路径）
            Log(resultBuf, resultBuf, 1, true);
            PipeBarrier<PIPE_V>();
            Muls(resultBuf, resultBuf, invP_, 1);
            PipeBarrier<PIPE_V>();
            Exp(resultBuf, resultBuf, 1);
            PipeBarrier<PIPE_V>();
```

```cpp
// after（同一路径，3 参）
            Log(resultBuf, resultBuf, 1);
            PipeBarrier<PIPE_V>();
            Muls(resultBuf, resultBuf, invP_, 1);
            PipeBarrier<PIPE_V>();
            Exp(resultBuf, resultBuf, 1);
            PipeBarrier<PIPE_V>();
```

（该算子后续在原地复用场景升级为显式模板参数 `Log<ComputeT, true>(workBuf0, workBuf0, static_cast<uint32_t>(chunkElem))`，见上文 foreach 轨迹的解决配方——isReuseSource 从"函数第 4 参"误用收敛为模板参数正确用法。）

**3. 预防增补**：从旧代码/旧文档迁移 kernel 代码时，凡带 `isReuseSource` 尾参的调用都要按当前头文件核对——CANN 9.1.0 的 adv_api 向量数学接口普遍只有 2/3 参形态；报错后不要只改报错行，须全局 grep 同一 API 的其他调用点（本例两处 4 参 `Log` 必须同批修）。核对命令：

```bash
grep -n "inline void Log(" $ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/math/log.h
```

### 复现增量验证信号

- 修复前：kernel 编译在 `CompileStage.INFERCHANNEL` 报 `no matching function for call to 'Log'`，候选重载 `requires 3 arguments, but 4 were provided` / `requires 2 arguments, but 4 were provided`（来源平台：950 交叉编译；证据等级：tool_observed）。
- 修复后：两处去掉第 4 参后重构建，kernel 编译通过，生成 `build/op_kernel/ascendc_kernels/binary/ascend950/<op_name>/BenchNoHarnessForeachNorm_....o` 与自定义算子安装包 `custom_opp_ubuntu_x86_64.run`（来源平台：950 交叉编译；证据等级：tool_observed）。

### 复现增量失效边界

- 通用 p 范数分支（NORM_TYPE=5）在该证据点为编译级验证，数值正确性（log→exp 链路的精度）由同工程后续上板 ST 闭环（见上文 foreach 轨迹验证信号）。
- CANN 版本差异未验证：其他版本若提供 4 参重载则该形态不适用（未验证，推断可迁移到 9.x 系列）。