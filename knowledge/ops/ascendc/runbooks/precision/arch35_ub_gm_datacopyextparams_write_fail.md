---
type: "Runbook"
title: arch35 UB→GM DataCopyExtParams 拷贝写失效
description: "Ascend950PR(arch35)+CANN 9.1.0 上 DataCopyPad 的 UB→GM (MTE3) 方向用 DataCopyExtParams 变体写入静默失效/数据损坏：编译与 UT 全过、无运行报错，输出近零（未初始化内存特征）或大面积垃圾值（连单核用例也挂，MARE 最高 9.9e+07）；TQue 管线同步正确也不能救。修复用 GlobalTensor::GetValue/SetValue 标量通路（迭代一全过）或 DataCopyParams 变体 + TQue + Adds 桥接（对齐 tile 实测 bit-exact）。"
tags: [task.debugging, topic.data_movement.data_copy, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# arch35 UB→GM DataCopyExtParams 拷贝写失效

## 现象

bench_no_harness_scatter 算子（Ascend950PR / DAV_3510 / arch35，CANN 9.1.0）ST Real（NPU）联调中，`DataCopyPad` 的 UB→GM（MTE3）方向出现两类客观失效，均无任何报错：

**形态一（迭代一，整块拷贝静默失效）**：507035 崩溃修复后，所有 fp16/int32 用例仍精度失败，且错误形态是**输出近零**而非数值偏差：

```
[Real] PIERCE_MAIN_1D_FP16_I32_NONE: ...
  [FAIL] MERE=1.85e+00 (>= 9.77e-04), MARE=1.43e+02 (>= 9.77e-03), float16
    不匹配 [0]: golden=2.964844, actual=0.000061, rel_err=1.00e+00
    不匹配 [3]: golden=0.968262, actual=0.000000, rel_err=1.00e+00
    不匹配 [8]: golden=3.423828, actual=0.000002, rel_err=1.00e+00
```

`0.000061 ≈ 2^-14`（fp16 最小正规数）——输出基本是未初始化内存，说明 kernel 的 Phase 1（data→y 拷贝，`DataCopyPad` GM→UB→GM，UB→GM 段用 `DataCopyExtParams{1, bytes, 0, 0, 0}`）没有把数据写进 GM。编译零报错、运行零报错；`--make_clean` 干净重建排除缓存二进制后现象不变；Phase 2 的单元素 `DataCopyPad`（同样 `DataCopyExtParams` 形态、对齐 yTmpLocal 中转）UB→GM 写同样不落盘。

**形态二（迭代二，TQue 管线下数据损坏）**：为绕开多核标量写跨核不可见问题，把 Phase1 改为 `TQue` + `DataCopyPad` GM→UB→GM（AllocTensor/EnQue/DeQue 正确配对），UB→GM 仍用 `DataCopyExtParams{1, tileBytes, 0, 0, 0}`。编译通过、UT 17/17 通过（UT 只测 Tiling 参数，不覆盖数据搬运），但 ST Real 全面失败——不仅多核用例挂，**此前一直通过的单核用例（PIERCE3，blockDim=1）也挂**，大张量失配扩大到 MARE≈1e8 量级：

```
[Real] PIERCE3_1D_FP16_I_ADD: [FAIL] MERE=2.75e+00, MARE=1.64e+02
[Real] PIERCE2_2: [FAIL] MERE=2.71e+03, MARE=9.92e+07
```

失败模式（全量挂、失配幅度随张量变大而爆炸、单核也挂）与「跨核可见性」（只有小张量挂）完全不同，指向搬运本身写坏数据。

## 适用场景

- 工程形态：ascendc-registry-invoke 算子仓开发 kernel 侧 UB→GM 数据搬运，ST Real 上板精度验证。
- API 路径：`AscendC::DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams)`（UB→GM 方向）、`TQue::AllocTensor/EnQue/DeQue/FreeTensor`；替代通路 `GlobalTensor::GetValue/SetValue`、`DataCopyParams` 变体 + `Adds` 桥接；arch35 上 `DataCopyExtParams` 的 `srcStride/dstStride` 为 `int64_t`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0。
- 适用粒度：platform_wide（arch35 上所有 UB→GM 的 DataCopyExtParams 用法）。
- 路线优先级：banned（arch35 UB→GM 禁用 DataCopyExtParams 变体）；正确路线见解决配方。DataCopyParams 变体（对齐 tile）= 可用；标量 SetValue/GetValue = 单核内可用、跨核受限（见失效边界）。
- 明确不适用 / 未验证场景：GM→UB（MTE2）方向同参数实测正常，不因 UB→GM 结论而禁用；其他 arch/CANN 版本未验证（未验证，不可推断迁移）。

## 根因

arch35（DAV_3510）后端的 UB→GM `DataCopyPad` 实现对 `DataCopyExtParams` 形态存在缺陷，数据未正确写入 GM；这不是同步问题——形态二用 TQue 队列语义（Alloc/EnQue/DeQue 正确配对）排除了「MTE2→MTE3 顺序未保证」的解释，单核用例（无任何跨核可见性问题）同样损坏也排除了缓存一致性解释。机制层（`CopyUbufToGmAlignV2` 对 ExtParams 分支的 burst 长度换算等，见 `x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h`）未继续下钻。

判别口径：编译通过 + UT 通过 ≠ 搬运正确——UT 只测 Tiling，数据搬运正确性必须靠 NPU 实跑 golden 比对；失败模式对比是关键判别信号：跨核可见性问题 = 小张量挂/大张量过；搬运 API 损坏 = 全量挂、失配随规模爆炸、单核也挂。

## 解决配方

**路线一（迭代一采用的正确性修复，全标量通路）**：Phase 1 与 Phase 2 全部改 `GlobalTensor::GetValue/SetValue` 标量读改写，不再经 UB 中转：

```cpp
template <typename D_T_DATA, typename D_T_IDX>
__aicore__ inline void BenchNoHarnessScatter<D_T_DATA, D_T_IDX>::Phase2ScatterUpdate()
{
    if (indicesTotalNum_ <= 0) {
        return;
    }
    // 单核（迭代一）：直接用 GetValue/SetValue 标量操作，避免 DataCopyPad UB→GM 问题
    for (int64_t k = 0; k < indicesTotalNum_; ++k) {
        int32_t idxVal = static_cast<int32_t>(indicesGm_.GetValue(k));
        int64_t targetOffset = ComputeTargetOffset(k, idxVal);
        auto updateVal = updatesGm_.GetValue(k);

        if (reduceMode_ == 0) {
            // reduce=None: 直接覆盖写
            yGm_.SetValue(targetOffset, updateVal);
        } else {
            // reduce 模式: 读-改-写
            auto oldVal = yGm_.GetValue(targetOffset);
            auto newVal = ScalarOp(oldVal, updateVal, reduceMode_);
            yGm_.SetValue(targetOffset, newVal);
        }
    }
    PipeBarrier<PIPE_ALL>();
}
```

Phase 1 同型：`for (...) { yGm_.SetValue(i, dataGm_.GetValue(i)); }` + `PipeBarrier<PIPE_ALL>()`。两阶段间保持 `PipeBarrier<PIPE_ALL>() + SyncAll<true>()` 跨阶段同步。

**路线二（迭代二实测可用的批量搬运替代，DataCopyParams 变体 + V 操作桥接）**：

```cpp
DataCopyParams copyParams;
copyParams.blockCount = 1;
copyParams.srcStride = 0;
copyParams.dstStride = 0;
DataCopyPadParams padParams{false, 0, 0, 0};
...
copyParams.blockLen = static_cast<uint16_t>(tileElem * sizeof(D_T_DATA));

// CopyIn: GM→UB (MTE2)
LocalTensor<D_T_DATA> inLocal = copyInQueue_.template AllocTensor<D_T_DATA>();
DataCopyPad(inLocal, dataGm_[offset], copyParams, padParams);
copyInQueue_.EnQue(inLocal);

// Compute: V 操作桥接 MTE2→V（dummy Adds +0，等效拷贝）
LocalTensor<D_T_DATA> midLocal = copyInQueue_.template DeQue<D_T_DATA>();
LocalTensor<D_T_DATA> outLocal = copyOutQueue_.template AllocTensor<D_T_DATA>();
Adds(outLocal, midLocal, static_cast<D_T_DATA>(0), static_cast<int32_t>(tileElem));
copyOutQueue_.EnQue(outLocal);
copyInQueue_.FreeTensor(midLocal);

// CopyOut: UB→GM (MTE3)
LocalTensor<D_T_DATA> gmLocal = copyOutQueue_.template DeQue<D_T_DATA>();
DataCopyPad(yGm_[offset], gmLocal, copyParams);
copyOutQueue_.FreeTensor(gmLocal);
```

注意 `Adds` 等向量操作要求元素数对齐 32B/sizeof(T)（fp16=16），非对齐尾部需另行处理；多核标量写的跨核可见性限制见失效边界。

## 设计决策

- 选全标量 `GlobalTensor::GetValue/SetValue`（S 管线标量访存）作迭代一修复：单核、千级元素规模，正确性验证优先；两种 `DataCopyExtParams` 写法（整块直写、对齐 yTmpLocal 中转单元素写）都已实测不落盘，无可调参数空间。
- 迭代二判别口径：先做失败模式对比（全量挂 + 单核挂 vs 规模相关挂）一次定位到搬运 API 变体，与迭代一 issue 记录（`issues/issue_20260718_datacopypad-ub-to-gm-arch35_01.md`）相互印证。
- 被放弃的假设路线：① 加 `PipeBarrier<PIPE_ALL>` 补 MTE3 屏障——无效（排除同步缺失）；② 怀疑 UB 预算/TBuf 与 TQue 冲突——账面核对（32768+141952+70976+32=245728B < 253952B）无超限。
- 当前方案路线档位：DataCopyExtParams UB→GM = banned；DataCopyParams（对齐 tile）= 可用；标量通路 = 单核正确性兜底。

## 预防

- arch35 kernel 用 `DataCopyPad` 做 UB→GM 写时，第一次上板就用"已知非零的 data → 拷贝 → 比对输出"冒烟验证搬运本身，不要默认搬运成功去查下游逻辑。
- 输出全零/近零 + 无任何报错 + 编译干净时，优先怀疑"写未落盘"而不是"算错"（近零 ≈ 未初始化内存特征：fp16 下 0.000061≈2^-14、0.000002 等最小正规数量级）。
- 排查此类问题先做一次 `--make_clean` 干净构建（排除缓存二进制假阳性，见库内「kernel头改动后增量编译沿用旧二进制」卡），再逐通路替换验证。
- 搬运 API 换写法后，必须跑一组覆盖单核/多核、小/大张量的 ST Real 用例再下结论（编译 + UT 不构成搬运正确性证据）。

## 验证信号

以下信号来自该算子迭代一/迭代二 ST Real 联调的多轮"干净构建 → 安装 → run.sh --real"循环。

- 形态一失败：Phase 1 DataCopyExtParams 块拷贝下 PIERCE_MAIN `[FAIL] MERE=1.85e+00, MARE=1.43e+02`，actual=0.000061/0.000000/0.000002（近零）；干净构建（排除缓存）后现象不变（来源平台：950；证据等级：tool_observed）。
- 通路二分（形态一）：Phase 1 改 SetValue 后 actual 变为真实数据值（-3.166016/2.796875 等）——MTE3 块写在替换前未落盘；Phase 2 单元素 DataCopyExtParams（对齐 yTmpLocal 中转）同样在替换前不落盘（来源平台：950；证据等级：tool_observed）。
- 形态一修复后：Phase 1/Phase 2 全标量通路下 PIERCE_MAIN/PIERCE1/PIERCE2/L0_018 全部 `[PASS] MERE=0.00e+00, MARE=0.00e+00`（1024/8192/8192/80 elems, float16）（来源平台：950；证据等级：tool_observed）。
- 形态二尝试（DataCopyExtParams + TQue）：编译 `[INFO] Build completed successfully!`、UT `[PASSED] 17 tests`；ST Real 全量失败——单核 PIERCE3 MERE=2.75e+00 / MARE=1.64e+02，PIERCE2_2 MERE=2.71e+03 / MARE=9.92e+07，PIERCE_MAIN MERE=9.50e-01 / MARE=1.00e+00（来源平台：950；证据等级：tool_observed）。
- 回退对照（形态二）：改回标量拷贝后 ST 恢复「大张量过、小张量挂」的跨核可见性失败模式（规模相关），确认 ExtParams 损坏是独立叠加的根因（来源平台：950；证据等级：tool_observed）。
- 路线二对照：`DataCopyParams` + TQue + Adds 版本下大张量用例（PIERCE2_1/2_2/4，2M~4M 元素）MERE=0.00e+00 / MARE=0.00e+00 全过（来源平台：950；证据等级：tool_observed）。
- 机制边界证据：GM→UB 方向实测正常（Phase 2 加载 indices/updates 后标量读值正确）；arch35 `DataCopyExtParams` 的 `srcStride/dstStride` 为 `int64_t`（`kernel_struct_data_copy.h`，`__NPU_ARCH__ == 3510` 分支）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定"实测环境"：Ascend950PR / arch35 / CANN 9.1.0；其他平台不可推断迁移（未验证）。
- 只覆盖 `DataCopyPad` 的 UB→GM 方向；GM→UB 正常。
- `DataCopyParams` 替代路线实测覆盖对齐 tile（fp16、tileElem 为 16 倍数）场景；非对齐尾部行为未单独验证。
- 标量通路为单核正确性兜底；多核场景 `GlobalTensor::SetValue` 的 S 管线写存在跨核可见性限制（小张量/跨步访问须单核规避），另见同批「多核 S 管线标量写跨核不可见」卡。
- `DataCopyParams` 与 arch35 DataCopyPad 的可用重载组合按调用形态而异：迭代一曾有一次 `DataCopyParams` 替换报 `no matching function for call to 'DataCopyPad'` 编译失败（该轮实验曾被增量构建假成功掩盖），迭代二的四参/三参组合则编译并实测通过——两轮观测未完全对齐，采用前先按头文件核对目标调用形态（见遗留问题）。
- 同日另一独立轨迹（ROIAlign 类算子）实测 `GlobalTensor::SetValue` 逐元素写 GM 亦存在"kernel 内读回正确、host 侧尾部丢写"形态——标量通路与 DMA 通路在 arch35 上各有失效面，写 GM 选型时须双向核对，不能默认任一通路绝对安全。

**未决项**：

- **优先项**：复现并定位 arch35 UB→GM `DataCopyExtParams` 写失效/损坏的微架构机制（`CopyUbufToGmAlignV2` 对 ExtParams 分支的 burst 长度换算：截断？错位？丢弃？），建议用最小用例（单 tile 已知 pattern）复现对比。
- `DataCopyParams` 在 arch35 上与 DataCopyPad 各调用形态（参数个数/方向）的可用重载表未逐一核对（迭代一与迭代二观测不一致），需头文件级梳理。
- 与库内「b16 DataCopyPad UB→GM 源非 32B 对齐致硬件 fault」卡是否同源（同一 MTE3 通路的两种表现）未验证。
- 多核场景恢复批量搬运的正确形态（TQue + DataCopyParams + 非对齐尾部处理）未完成工程化验证。
## 坏实践

- 在 arch35 上用 `DataCopyExtParams{1, bytes, 0, 0, 0}` 做 UB→GM 整块拷贝并依赖「无对齐要求」的便利性。
- 只凭编译通过 + UT 通过就认为搬运路线可交付。
- 在未验证搬运落盘的情况下，把输出近零当成精度/算法问题去查计算逻辑（本例曾先后怀疑流水线同步、UB 预算冲突，均被排除）。
- 用增量构建验证 kernel 搬运改动（假阳性会双重误导，见库内增量构建陈旧二进制卡）。

## 误诊

- 形态一先后误判为：MTE3 后缺 `PipeBarrier`（加了无效）、`DataCopyExtParams` 与 `DataCopyParams` 结构差异（换了无效且一度编译失败）、TBuf/TQue UB 冲突（账面排除）。真正判别手段是"逐通路替换成标量访存看输出是否变化"。
- 形态二无（通过失败模式对比一次定位到搬运 API 变体）。
