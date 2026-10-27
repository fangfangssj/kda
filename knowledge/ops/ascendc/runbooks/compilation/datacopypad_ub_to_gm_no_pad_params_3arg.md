---
type: "Runbook"
title: DataCopyPad UB→GM 不收 padParams，3 参重载
description: AscendC DataCopyPad 的 UB→GM 方向只有 (gm, ub, DataCopyExtParams) 三参重载；照搬 GM→UB 的第四参 DataCopyPadExtParams 报 no matching function 编译错（9.0.0-beta.2 与 9.2.0 重载集实测，头文件两方向签名 verbatim 引用）。R3（9.2.0）：单元素源 `tensor[i]` 引用同样不匹配任何重载——须先 SetValue 到完整 LocalTensor 槽再整体传参；同一循环里读方向 4 参、写方向 3 参并存才是可编译形态。R4/R5：DataCopyParams 形态 blockLen 为 uint16 字节数（narrowing 陷阱）；3510 分支 DataCopyExtParams stride 为 int64。
tags: [task.implementation, topic.data_movement.data_copy, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: "2026-09-09T11:49:41Z"
updated_at: "2026-09-16T11:11:42Z"
platforms: ["950"]
---

# DataCopyPad UB→GM 不收 padParams，3 参重载

## 现象

在实现 ThnnFusedGruCellGrad 算子仓工程的 group 分支 kernel（`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h`，A×R 2D 分核两阶段结构）时，需要在三个位置做 UB→GM 的搬出：Phase 1 末把局部归约树根从 `cacheBuf`（UB）写进 GM workspace（`Phase1OutputToWorkspace`），以及 Phase 2 把 `postBuf`（UB）里的两路 bias 结果写到 GM（`CopyOutBiasDual` 的 part1/part2 两次拷出）。

实现时按 GM→UB 方向的习惯写法给这三处 `AscendC::DataCopyPad` 都传了第四个参数 `DataCopyPadExtParams`（padParams），随后执行标准构建命令：

```bash
( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package )
```

编译失败，报错原文（每个 dtype × tiling 模板实例重复出现）：

```
/workspace/.../op_kernel/arch35/thnn_fused_gru_cell_grad_group.h:212:9: error: no matching function for call to 'DataCopyPad'
/workspace/.../op_kernel/arch35/thnn_fused_gru_cell_grad_group.h:360:9: error: no matching function for call to 'DataCopyPad'
/workspace/.../op_kernel/arch35/thnn_fused_gru_cell_grad_group.h:370:9: error: no matching function for call to 'DataCopyPad'
RuntimeError: Kernel Compilation Error: OpType ThnnFusedGruCellGrad Kernel File .../thnn_fused_gru_cell_grad_apt.cpp!
```

关键判别信号：同一文件里有 4 个 `DataCopyPad` 调用点，报错的 212/360/370 三行**恰好全部是 UB→GM 方向**（dst 为 GlobalTensor、src 为 LocalTensor）；行 332 的 `CopyInWorkspace` 是 GM→UB 方向（dst 为 UB `preRegion_`、src 为 GM `wsGm_`），同样带第四参 padParams 却编译通过。

另注意：构建最先抛出的错误可能是 FATBIN 阶段的 `ld.lld: error: cannot open .../kernel_meta/....o: No such file or directory`——这只是上游 clang 编译失败导致 .o 未生成的下游症状，需要往前翻日志找真正的 C++ 报错行。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke 流程，AscendC 模板 kernel，aclnn ST 联调）；任何需要在 kernel 内把 UB 数据拷出到 GM 的场景。
- API 路径：AscendC 基础 API `AscendC::DataCopyPad`（`kernel_operator.h`），配 `DataCopyExtParams` / `DataCopyPadExtParams`。
- 验证过的平台与 CANN 版本：Ascend 950（构建目标 `ThnnFusedGruCellGrad_ascend950`）+ CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`）。与 frontmatter `platforms` 一致。
- 适用粒度：operator_family——DataCopyPad 重载集合是 CANN 基础 API 语义，跨算子一致。
- 路线优先级：recommended_fastpath（UB→GM 一律 3 参调用）。
- 明确不适用/未验证：GM→UB 方向第四参 `DataCopyPadExtParams` 是合法且必需的（本开发会话 行 332 的用法编译通过并运行正确）；其他 CANN 版本的重载集合未验证。
- 版本与证据扩展（独立复现 R2，BenchHarnessEngramGateFusion，CANN 9.2.0）：950PR + CANN 9.2.0 的 aclnn 包工程（双输出融合算子，kernel 头内三处 CopyOut）复现同错误串、同三参修复；`x86_64-linux/asc/include/basic_api/kernel_operator_data_copy_intf.h`（`__NPU_ARCH__ == 3510` 后端分支）两方向重载签名 verbatim 引用确认不对称在 9.2.0 仍成立——MTE2 方向提供 `DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams&, const DataCopyPadExtParams<T>&)` 四参，MTE3 方向只有 `DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams&)` 三参（3510/5102 分支下另有以 `DataCopyExtParams` 为第三参的 MTE3 override，同样只有三参）。

## 根因

`DataCopyPad` 的重载集合按搬运方向不对称：UB→GM 方向只提供 3 参重载 `DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams)`，不存在带 `DataCopyPadExtParams` 的 4 参形式；padParams（源侧 pad 语义）只在 GM→UB 方向有意义/有重载。R3 补充：单元素引用（`updBuf[i]`，LocalTensor 的 `operator[]` 返回值）也不是任何重载的合法源实参——DataCopy 系 API 收发的是完整的 LocalTensor/GlobalTensor，单元素必须先落到独立 UB 槽再以整个 LocalTensor 传参。

## 解决配方

before（`CopyOutBiasDual` 内，照搬 GM→UB 的 4 参写法，编译错在 360/370 行）：

```cpp
    if (part1 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part1 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPadExtParams<D_T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(Base::gmInBias_[aOff], Base::postBuf_.template Get<D_T>(), ext, padParams);
    }
    if (part2 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part2 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPadExtParams<D_T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(Base::gmHidBias_[aOff - biasBoundary],
                             Base::postBuf_.template Get<D_T>()[static_cast<uint32_t>(part1)], ext, padParams);
    }
```

after（去掉 UB→GM 方向的第四参，保留 `DataCopyExtParams` 承载 blockLen/blockCount/stride；同型修复也应用于 212 行的 `Phase1OutputToWorkspace`）：

```cpp
    if (part1 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part1 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmInBias_[aOff], Base::postBuf_.template Get<D_T>(), ext);
    }
    if (part2 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part2 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmHidBias_[aOff - biasBoundary],
                             Base::postBuf_.template Get<D_T>()[static_cast<uint32_t>(part1)], ext);
    }
```

修复后 `./build.sh ThnnFusedGruCellGrad_package` 输出 `SUCCESS` 并生成 `custom_opp_ubuntu_x86_64.run`。

### 独立复现 R3（BenchHarnessScatter，CANN 9.2.0）：单元素源变体与读写方向并存形态

把逐元素 GM 写从 S-pipe `SetValue` 改成 MTE3 `DataCopyPad` 时，按 MTE2 读方向惯性写了 4 参（dst, src, DataCopyExtParams, DataCopyPadExtParams）：

```cpp
            AscendC::DataCopyPad(gmY_[targetOff], yBuf, p1, pad1);
```

构建报（同错误的更早变体：`AscendC::DataCopyPad(gmY_[targetOff], updBuf[i], p1, pad1);`——`updBuf[i]` 是 LocalTensor 的单元素引用，同样不匹配任何重载）：

```
/workspace/.../op_kernel/arch35/bench_harness_scatter_kernel.h:336:13: error: no matching function for call to 'DataCopyPad'
/workspace/.../op_kernel/arch35/bench_harness_scatter_kernel.h:361:13: error: no matching function for call to 'DataCopyPad'
```

修复后（MTE3 方向 3 参 + 单元素先落独立 UB 槽；verbatim 取自会话中的修正编辑，ScatterChunk 函数体自 `auto yBuf` 起）：

```cpp
    auto yBuf  = buf_[B2].Get<T>();
    AscendC::DataCopyExtParams p1; p1.blockCount=1; p1.blockLen=sizeof(T); p1.srcStride=0; p1.dstStride=0;
    AscendC::DataCopyPadExtParams<T> pad1{};

    if (td_->reduceMode == 0) {
        // Overwrite: y[target] = updates[p] via MTE3 DataCopyPad.
        // S-pipe writes update to B2[0], MTE3 copies B2[0]→gmY[targetOff].
        // Consecutive MTE3 writes are FIFO-ordered → last-writer-wins for duplicates.
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = idxBuf.GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal,
                                                   td_->dataStrides, td_->rank);
            T updVal = updBuf.GetValue(i);
            yBuf.SetValue(0, updVal);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            AscendC::DataCopyPad(gmY_[targetOff], yBuf, p1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evMTE3_S_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evMTE3_S_);
        }
    } else {
        // Reduce: per-element MTE2 read → S reduce → MTE3 write.
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = idxBuf.GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal,
                                                   td_->dataStrides, td_->rank);
            // MTE2: read y[targetOff] → B2[0]
            AscendC::DataCopyPad(yBuf, gmY_[targetOff], p1, pad1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
            // S-pipe: reduce B2[0] with B1[i]
            T oldVal = yBuf.GetValue(0);
            T updVal = updBuf.GetValue(i);
            T newVal = ReduceScalar<T>(td_->reduceMode, oldVal, updVal);
            yBuf.SetValue(0, newVal);
            // S→MTE3 sync: B2[0] written by S-pipe visible to MTE3
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            // MTE3: write B2[0] → y[targetOff]
            AscendC::DataCopyPad(gmY_[targetOff], yBuf, p1);
            // MTE3→MTE2 sync: B2 reusable for next iteration's MTE2 read
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        }
    }
```

要点：**同一循环里读方向 4 参（带 pad1）、写方向 3 参**，两者并存才是可编译形态；单元素源不能直接用 `updBuf[i]`，先 `SetValue` 到独立缓冲槽再以 LocalTensor 整体传参。

## 设计决策

- 判别方法以"报错行号的方向"为准：同文件混用两个方向时，逐行核对 dst/src 是 GlobalTensor 还是 LocalTensor——报错行清一色落在 UB→GM 方向、GM→UB 方向的 4 参调用不报错，即可定位为方向不对称而非拼写问题。
- 参照系：同工程 base 内核 `CopyOutBias` 的 UB→GM 写法本来就是 3 参形式 `AscendC::DataCopyPad(gmInBias_[aOff], postBuf_.Get<D_T>(), cp)`（blockCount=3 + srcStride/dstStride），且 Branch-0 ST 20/20 通过——修复直接向这一已验证写法看齐。
- 当前方案属 recommended_fastpath：UB→GM 需要"裁掉尾部 garbage lane"时，用 `blockLen=valid 字节数` 表达（如 `aLen_ * sizeof(float)`），不依赖 pad 语义。

## 系统信息

- 工程布局：`operators/ThnnFusedGruCellGrad_package/{op_kernel,op_host,tests}`，构建入口 `operators/build.sh <pkg>`；kernel 源 `op_kernel/arch35/thnn_fused_gru_cell_grad_group.h`（本卡改动文件），共享 VF 链 `op_kernel/arch35/thnn_fused_gru_cell_grad_kernel.h`，base 内核 `op_kernel/arch35/thnn_fused_gru_cell_grad_base.h`（3 参参照写法来源）。
- 工具链：CANN 9.0.0-beta.2（编译器 `compiler/tikcpp`，`asc_op_compile_base` 打包链）；目标 ascend950。

## 预防

- 写 UB→GM 搬出时先默写 3 参签名 `(gm, ub, DataCopyExtParams)`；padParams 只出现在 GM→UB。
- 不要把 `tensor[i]`（单元素引用）直接当 DataCopy 源/目的——先落到完整的 LocalTensor/GlobalTensor 上。
- 同文件两个方向混用时，把"方向注释"写在每处 DataCopyPad 上方（本工程 base kernel 即如此），报错时可直接对行核对。
- 见到 `ld.lld: cannot open ....o` 先怀疑上游 C++ 编译失败，用 `grep -iE "error:" | grep -ivE "ld.lld|No such file"` 滤出真实报错。

## 验证信号

本卡验证来自该 group kernel 的首次编译修复轮（开发会话内 `./build.sh` 构建输出）：

- 修复前：`no matching function for call to 'DataCopyPad'` ×3（行 212/360/370）+ `Kernel Compilation Error`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一构建命令输出 `Compile op info cfg successfully` / `SUCCESS`，自解压 run 包生成成功（来源平台：950；证据等级：tool_observed）。

- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0）：修复前 `bash operators/build.sh <package>` 在 kernel 头 869/748/786 三行（主输出、状态输出、空 kernel 清零写三处 CopyOut）报 `error: no matching function for call to 'DataCopyPad'`；删除第四参后同命令重编输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（BenchHarnessScatter，CANN 9.2.0）：修复前 4 参 MTE3 写两次构建失败，报 `no matching function for call to 'DataCopyPad'`（含 `updBuf[i]` 与 `yBuf` 两个变体）；改为写方向 3 参 + 单元素先落独立 UB 槽后构建 SUCCESS，上板复现用例保持 PASS（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3 头文件证据：CANN 9.2.0 `kernel_operator_data_copy_intf.h` 行 365-428 的重载清单——dst 为 GlobalTensor 的 MTE3 重载全部 3 参、dst 为 LocalTensor 的 MTE2 重载才提供 4 参带 `DataCopyPadExtParams`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（ROIAlign 算子 CopyOut，CANN 9.1.0）：四参（DataCopyExtParams+DataCopyPadExtParams）形态报 no matching function；改用 DataCopyParams 形态的 4 字段 brace 组 `{1, blockLen, 0, 0}` 时，`uint32_t` 表达式传入第 2 字段触发 C++11 narrowing 编译错（`non-constant-expression cannot be narrowed from type 'uint32_t' to 'uint16_t'`），显式 `static_cast<uint16_t>(outLen * sizeof(T))` 后编译打包成功、目标用例 PASS（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5（conv3d 反传算子，CANN 9.1.0，头文件逐段读取）：arch3510/5102 分支下 UB→GM 仅 `(GlobalTensor, LocalTensor, DataCopyParams/DataCopyExtParams)` 两种 3 参形态；`DataCopyExtParams` 在 3510 的构造为 `(uint16_t count, uint32_t len, int64_t srcStrideIn, int64_t dstStrideIn, uint32_t rsvIn)`——stride 位宽 int64，与其余架构的 uint32 不同；同一 copyParams 读方向补 padParams、写方向省去的配对写法（花括号初始化自动走 3510 构造重载）在 NPU 上运行正常、CPU Mock 同代码 5/5 用例 MERE=0（来源平台：950；证据等级：artifact_verified——CANN 安装头文件逐段读取；配对写法运行为 tool_observed）。

## 失效边界

- GM→UB 方向带 padParams 合法（本开发会话 中 `CopyInWorkspace` 以 4 参形式编译通过并参与后续全部 ST 通过）。
- 重载集合验证于 CANN 9.0.0-beta.2 / ascend950；其他 CANN 版本未验证（跨版本迁移时以对应版本头文件为准）。
- 版本边界扩展：重载集已于 CANN 9.2.0（ascend950/3510 分支）与 9.1.0（R4/R5）再验证，不对称仍成立；后续版本若为 MTE3 增补四参重载，本卡过期。
- R4 增补：`DataCopyParams` 形态（4 字段 uint16）的第 2 字段 blockLen 语义是**字节数**且接收 `uint16_t`——传 `uint32_t` 表达式触发 narrowing 编译错，须显式收窄。
- R5 增补：跨 arch 移植注意 `DataCopyExtParams` 的 stride 位宽差异：3510/5102 为 int64，其余架构为 uint32；花括号初始化 `{1, len, 0, 0, 0}` 会按 `__NPU_ARCH__` 自动选择对应构造重载，无需手工区分。
- 相邻卡片：[direct-launch 四参变体卡](datacopypad_ub_to_gm_has_no_pad_params.md)——同一 nugget 的直调工程形态记录，现象/根因/修复一致。[DataCopyPad 多行 stride 32B 块单位卡](../precision/datacopypad_stride_fields_use_32_byte_block_units.md)——三参修复后 stride 字段的单位坑：多行搬运的对齐差须 gap_bytes/32。[Vector 结果搬出前须经 VECOUT 队列卡](../precision/vector_results_need_vecout_queue_before_mte3.md)——CopyOut 之前的队列交接约束，输出垃圾值/逐位滞后时先查它。

## 坏实践

- 从 GM→UB 调用复制粘贴后只改指针不改参数个数——本卡 before 即此产物。
- 把 `ld.lld: cannot open *.o` 当作链接/构建系统问题去排查（真实原因在上游编译阶段）。

- R4 增补：给 UB→GM 的 brace 参数组传未收窄的 `uint32_t` blockLen 表达式——narrowing 编译错；记住参数表：DataCopyParams{blockCount, blockLen(uint16, 字节), srcStride, dstStride}。

## 误诊

无（本闭环一次定位一次修复）。

## 未决事项

无。
