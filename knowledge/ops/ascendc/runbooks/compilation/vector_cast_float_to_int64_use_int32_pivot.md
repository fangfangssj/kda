---
type: "Runbook"
title: 向量Cast不支持float转int64，经int32中转
description: 索引输出用向量 Cast 把 float 转 int64_t 报 no matching function；改 Cast<int32_t,float> + 标量拓宽后编译通过。
tags: [task.implementation, topic.compilation, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# 向量Cast不支持float转int64，经int32中转

## 现象

在开发 Cummin（累积最小值）算子的 ARA-scan 模式（扫描维非最后一维）时，索引输出路径需要把以 float 维护的运行索引（running index）转换成 int64_t 输出。kernel 代码位于 `op_kernel/arch35/cummin.h`，索引回写调用写为：

```cpp
LocalTensor<int64_t> idxOut = idxInt64Buf.template AllocTensor<int64_t>();
LocalTensor<float> runIdxCopy = runningIdxBuf.template AllocTensor<float>();
Cast(idxOut, runIdxCopy, CAST_ROUND::CAST_RINT, tileA0Aligned);
```

执行 `bash build.sh --soc=ascend950 -j8` 编译自定义算子包，clang 报错：

```
op_kernel/arch35/cummin.h:192:46: error: use of undeclared identifier 'CAST_ROUND'
                    Cast(idxOut, runIdxCopy, CAST_ROUND::CAST_RINT, tileA0Aligned);
```

先尝试直接删掉 round mode 参数（`Cast(idxOut, runIdxCopy, tileA0Aligned);`），重新编译后错误变为：

```
op_kernel/arch35/cummin.h:193:21: error: no matching function for call to 'Cast'
op_kernel/arch35/cummin.h:297:21: error: no matching function for call to 'Cast'
```

随后进入 CANN 安装目录头文件查证（`$ASCEND_HOME_PATH` 下 `ascendc/include/basic_api/impl/kernel_operator_vec_vconv_intf_impl.h` 与 `asc/impl/basic_api/utils/kernel_utils_mode.h`），并用 `grep -rn "Cast.*int64" $ASCEND_HOME_PATH/opp/built-in/op_impl/.../ascendc/` 检索既有官方算子的用法。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），自定义 AscendC kernel 需要把 float 中间量转成 int64_t 输出（如 argmin/argmax/cummin/cummax 类索引输出）。
- API 路径：`AscendC::Cast` 向量类型转换（`kernel_operator_vec_vconv_intf_impl.h` Level 2 接口）、`RoundMode` 枚举（`kernel_utils_mode.h`）。
- 验证过的平台与 CANN 版本：Ascend 950（`--soc=ascend950` 编译 + NPU 实测）+ CANN 9.1.0。
- 适用粒度：`operator_family`（所有需要 float→整数索引输出的 AscendC kernel）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：int64_t→float 方向（官方算子存在 `Cast(float, int64, RoundMode::CAST_ROUND, ...)` 用法，本轨迹未实测）；其他 soc 版本未验证。

## 根因/解决配方

根因有两层，同一调用点先后暴露：

1. **枚举用法错误**：`RoundMode` 是 `kernel_utils_mode.h` 中的 `enum class RoundMode : uint8_t`，`CAST_RINT`、`CAST_ROUND` 都是其枚举值；`CAST_ROUND::CAST_RINT` 把枚举值当作用域名，编译期直接报 undeclared identifier。正确写法是 `RoundMode::CAST_RINT`（round to nearest even）。枚举声明位置与命名空间的辨析另见 [Reg CastTrait RoundMode 命名空间卡](casttrait_roundmode_is_in_ascendc.md)。
2. **类型对不支持**：向量 Cast 的 Level 2 签名为 `template <typename T, typename U> __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint32_t count)`。float→int64_t 这一类型对没有可匹配的实现（no matching function），硬件向量转换指令不支持该组合；既有官方算子实现用的是 `Cast<int32_t, float>`（float→int32_t 受支持）。

解决配方：索引值上界是扫描维长度（百万量级），int32 足够承载。先用受支持的 float→int32_t 转换，再用标量循环把 int32 拓宽写入 int64 输出缓冲：

修复前（函数级，r=0 初始化段的索引回写，r=1..R-1 段同型）：

```cpp
// Write back indices[a1, 0, :] = 0
{
    LocalTensor<int64_t> idxOut = idxInt64Buf.template AllocTensor<int64_t>();
    LocalTensor<float> runIdxCopy = runningIdxBuf.template AllocTensor<float>();
    Cast(idxOut, runIdxCopy, tileA0Aligned);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<int64_t> idxOutBuf = outQueueIndices.template AllocTensor<int64_t>();
    DataCopy(idxOutBuf, idxOut, tileA0Aligned);
    outQueueIndices.template EnQue<int64_t>(idxOutBuf);

    LocalTensor<int64_t> idxDeq = outQueueIndices.template DeQue<int64_t>();
    DataCopyExtParams idxCp;
    idxCp.blockCount = 1;
    idxCp.blockLen = tileCount * sizeof(int64_t);
    idxCp.srcStride = 0;
    idxCp.dstStride = 0;
    idxCp.rsv = 0;
    DataCopyPad(indicesGM[dstOffset0], idxDeq, idxCp);
    outQueueIndices.FreeTensor(idxDeq);
    idxInt64Buf.FreeTensor(idxOut);
    runningIdxBuf.FreeTensor(runIdxCopy);
}
```

修复后（新增 `idxInt32Buf` 中转缓冲，`pipe.InitBuffer(idxInt32Buf, tileA0Aligned * sizeof(int32_t));`）：

```cpp
// Write back indices[a1, 0, :] = 0
{
    LocalTensor<float> runIdxCopy = runningIdxBuf.template AllocTensor<float>();
    LocalTensor<int32_t> idxI32 = idxInt32Buf.template AllocTensor<int32_t>();
    Cast(idxI32, runIdxCopy, RoundMode::CAST_RINT, static_cast<uint32_t>(tileA0Aligned));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<int64_t> idxOut = idxInt64Buf.template AllocTensor<int64_t>();
    for (uint64_t ii = 0; ii < tileA0Aligned; ii++) {
        idxOut.SetValue(ii, static_cast<int64_t>(idxI32.GetValue(ii)));
    }

    LocalTensor<int64_t> idxOutBuf = outQueueIndices.template AllocTensor<int64_t>();
    DataCopy(idxOutBuf, idxOut, tileA0Aligned);
    outQueueIndices.template EnQue<int64_t>(idxOutBuf);

    LocalTensor<int64_t> idxDeq = outQueueIndices.template DeQue<int64_t>();
    DataCopyExtParams idxCp;
    idxCp.blockCount = 1;
    idxCp.blockLen = tileCount * sizeof(int64_t);
    idxCp.srcStride = 0;
    idxCp.dstStride = 0;
    idxCp.rsv = 0;
    DataCopyPad(indicesGM[dstOffset0], idxDeq, idxCp);
    outQueueIndices.FreeTensor(idxDeq);
    idxInt64Buf.FreeTensor(idxOut);
    idxInt32Buf.FreeTensor(idxI32);
    runningIdxBuf.FreeTensor(runIdxCopy);
}
```

## 设计决策

- 为什么选 int32 中转而不是维持 float 索引到输出前才转换：向量 Cast 支持的类型对由硬件指令决定，float→int64 无对应指令；int32 是官方算子验证过的类型对，标量拓宽循环每元素一条 `SetValue`，在索引量级（≤对齐 tile 大小）下开销可接受。
- 备选路线：SIMT/Reg 层的 `AscendC::Reg::Cast<int64_t, float>` 配 trait 与受支持舍入模式（如 `CAST_TRUNC`）可编译通过 f322s64 转换，见 [f32→s64 Reg Cast 舍入模式卡](reg_cast_f32_to_s64_uses_supported_rounding.md)——本轨迹在 classic 向量 API 工程内，选 int32 中转改动最小；两卡适用条件不同（Reg 层 vs Level 2 向量 API），套用前按工程体裁选择。RoundMode 的方向语义（扩精度 CAST_NONE / 缩精度 CAST_RINT）另见 [bf16 扩精度 Cast 误用 CAST_RINT 卡](../precision/bf16_cast_cast_rint.md)。
- 放弃的候选：直接删除 round mode 参数（第一步尝试，触发 no matching function，证明问题在类型对而非参数个数）；标量循环直接 float→int64（需先解决 bf16/fp16 标量比较的后端限制，见 [bf16 标量位级转换卡](bf16_scalar_to_fp32_bit_extension.md)）。
- 当前方案属 recommended_fastpath：受支持类型对 + 最小改动。

## 系统信息

- 工具链：CANN 9.1.0（`/workspace/.../Ascend/cann-9.1.0`），bisheng/clang 编译器，构建命令 `bash build.sh --soc=ascend950 -j8`。
- 目标平台：ascend950（NPU 实测设备 Ascend950PR）。
- 相关文件：`op_kernel/arch35/cummin.h`（Cast 调用点、idxInt32Buf/idxInt64Buf 缓冲声明与 InitBuffer）、CANN 头文件 `ascendc/include/basic_api/impl/kernel_operator_vec_vconv_intf_impl.h`（Cast 签名）、`asc/impl/basic_api/utils/kernel_utils_mode.h`（RoundMode 枚举）。

## 验证信号

本卡验证来自 Cummin 算子开发会话的编译与 NPU 联调：修复落地后 `bash build.sh --soc=ascend950 -j8` 不再出现 Cast 相关 error，kernel 二进制与 custom_opp 包正常生成（来源平台：950；证据等级：tool_observed）；后续同一代码路径在 NPU 上通过全量 L0 ST（21/21，含 fp16/fp32/int32/bf16 四 dtype 索引输出）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该配方只解决 float→int64 的向量 Cast 缺口；若索引可能超过 int32 范围（扫描维 > 2^31），标量拓宽前需另做范围处理（未验证场景）。
- `RoundMode::CAST_RINT` 为 round-to-nearest-even；若索引由非精确整数 float（如计算中间量）产生，舍入语义需单独确认（未验证）。
- 其他 CANN 版本的 Cast 支持类型对可能不同，未验证，推断可迁移。

**未决项**：

- 未验证 int64_t→float 反方向 Cast 在本场景的行为（官方算子有先例，按需验证）。
- 标量拓宽循环在极大 tile 下的性能代价未测量（无性能数据，本会话 capabilities 无 metrics）。
## 坏实践

- `CAST_ROUND::CAST_RINT` 这类把枚举值当作用域前缀的写法：报错信息（undeclared identifier 'CAST_ROUND'）容易误导人去猜测命名空间，实际应查 `kernel_utils_mode.h` 的枚举定义。
- 凭记忆写 API 签名：本例第一步"删掉 round mode 参数"就是未查签名的猜测；正确做法是 `grep` 安装目录头文件与 `opp/built-in/op_impl` 下官方算子的实际用法。

## 误诊

- 初见 `error: use of undeclared identifier 'CAST_ROUND'` 时曾怀疑需要 `AscendC::RoundMode::CAST_ROUND` 命名空间前缀（agent_reported，未经查证即被头文件阅读否定）。
