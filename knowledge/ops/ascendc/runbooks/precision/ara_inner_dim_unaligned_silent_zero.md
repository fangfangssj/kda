---
type: "Runbook"
title: ARA 归约内维非 32B 对齐静默全 0 输出
description: 归约轴非末轴（ARA）且行内连续维 A1×dtype 非 32B 对齐时，逐行向量算子按未对齐宽度寻址、ReduceMax/ReduceSum 按 srcInnerPad=true 解读 dense 布局，两者与实际 UB 布局错位，kernel 无任何报错静默输出全 0；修复为 UB 按 32B 对齐行 stride 布局、向量/归约全部改对齐宽度并置 srcInnerPad=false，四处必须一致。
tags: [task.debugging, topic.alignment, topic.data_movement.data_copy, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# ARA 归约内维非 32B 对齐静默全 0 输出

## 现象

在开发一个 softmax 归一化基准算子（opbuild 工程 `op_host/ + op_kernel/ + tests/aclnn/`，目标 Ascend 950，CANN 9.1.0）的 Branch-0 全载分支（tilingKey=0）联调阶段：用 TTK kernel 模式对该分支全量跑 980 条白盒用例（CSV `tests/whitebox/branch-0/ttk_design_doc_cases.csv`），结果 `total=980 pass=955 fail=25 skip=0`。25 条失败全部无编译错误、无运行时报错、无越界（`memory_oob_status=PASS`、`dyn_tiling_key=0` 命中正确分支、`dyn_block_dim=56` 多核分配正常），但其中 24 条 `dyn_precision=NO_OUTPUT`——kernel 执行后输出 tensor 未写出。

24 条失败用例的共同特征极其规律：shape 全部是 `(65,63)` 或 `(67,61)`，`dim=0` 或 `dim=-2`（归约轴是行数 65/67，即**归约轴不是末轴**），dtype 覆盖 bfloat16/float16/float32 三种，数据范围只有 `(-1,1)` 与 `(-0.1,0.1)` 两类常规随机范围（全零输入用例反而 PASS，说明不是数值问题而是写出/计算路径问题）。

在 C++ ST 工程 `tests/aclnn/test_aclnn.cpp` 新增复现用例后跑 `./run.sh`：8 条 ARA 非对齐用例（(65,63)/(67,61) × dim=0/-2 × 三 dtype）全部 `[R65_63_bf16_d0] FAIL (rtol/atol fail at 0)`——输出全 0；同时新增的对照用例 `(65,64) dim=0`（A1=64，64×4B=256B 恰好 32B 对齐）`PASS (ok)`。这一个单变量对照直接把根因绑定到"行内连续维 A1 的 32B 对齐性"。

## 适用场景

- 工程形态：ACLNN 自定义算子（opbuild 工程）kernel 联调 / TTK kernel 模式白盒精度执行；归约类算子（softmax/reduce 等）归约轴非末轴的合轴分支。
- API 路径：`AscendC::ReduceMax/ReduceSum`（`Pattern::Reduce::RA`，`srcShape` + `srcInnerPad` 语义）、`DataCopyPad`（`DataCopyExtParams.blockCount` + `DataCopyPadExtParams` 逐行 padding）、`asc_vf_call` 逐行向量算子（Cast/Sub+Exp/Div+Cast）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR）+ CANN 9.1.0（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`operator_family`（ARA 合轴布局 + Reduce API 的对齐约束对同类归约算子一致）。
- 路线优先级：`recommended_fastpath`（UB 内 32B 行 padding 是保持向量+归约主路径的正规解法，无更简单的替代）。
- 明确不适用/未验证场景：AR（归约轴=末轴）路径未改动（原 `rAlignElems_` 逻辑已正确处理）；大 shape 走 ChunkedR 分支（tilingKey=1）未触及；非 max-subtraction 语义（padding 不能用 -inf）的算子未验证。

## 根因

合轴后 ARA 布局为 `[R, A1]`（R=归约轴长度=行数，A1=行内连续维=aBundle），fp32 中间缓冲每行字节数 = A1×4。当 `A1×dtypeBytes % 32 != 0`（如 fp32 A1=63：63×4=252B，252 mod 32=28；bf16 A1=63：126B mod 32=30）时，原实现同时犯两个绑定同一布局的错：

1. 逐行向量算子（`asc_vf_call<CastInVf/SubExpVfRow/DivCastVfRow>`）以未对齐的 `aBundle_` 为每行处理宽度、以 dense 行距 `r*aBundle_` 寻址；
2. `ReduceMax/ReduceSum` 以 `srcShape[1]=aBundle_` 且 `srcInnerPad=true` 调用——`srcInnerPad=true` 向 API 声明"源张量末轴已 pad 到 32B 对齐"，API 据此按 `AlignUp(A1, 32B/4B)` 的行 stride 解读 UB，与实际 dense 布局错位。

两者叠加导致归约/向量硬件路径读错布局，kernel 不报任何错误但输出全 0（C++ ST 表现 `rtol/atol fail at 0`，TTK 表现 `NO_OUTPUT`）。对照用例 (65,64)（256B 对齐）通过，证明触发条件正是 A1 非 32B 对齐。

## 解决配方

思路：UB 内按 **32B 对齐行 stride** 布局，`DataCopyPad` 逐行以 -inf 右填充；向量算子与 Reduce 全部改用对齐宽度 `aBundleAlignElems_`（fp32 缓冲）/`aBundleAlignD_`（D_T 缓冲）；`srcInnerPad=false`（数据在 UB 里**真的**已 pad）；`CopyOut` 逐行只写有效 `aBundle_` 元素回 dense GM。

**before（原实现，ARA 路径关键代码）：**

Init 尺寸计算（ARA 与 AR 共用 dense 整块缓冲）：

```cpp
    rAlignElems_ = CeilAlignKernel(rTotal_, bsElemD);                        // AR：D_T 对齐 padded
    aBundleAlignElems_ = CeilAlignKernel(aBundle_, bsElemF32);               // max/sum buf（fp32 对齐）
    if (aBundleAlignElems_ < bsElemF32) {
        aBundleAlignElems_ = bsElemF32; // 至少 32B
    }
    vfCount_ = isAr_ ? rAlignElems_ : blockElems_; // AR 处理 padded（含 -inf pad）；ARA 处理整块

    // GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));

    // UB buffer 尺寸
    const int64_t xBufElems = isAr_ ? rAlignElems_ : CeilAlignKernel(blockElems_, bsElemD);
    const int64_t xBufBytes = xBufElems * elemBytes_;
    const int64_t xF32Elems = isAr_ ? rAlignElems_ : CeilAlignKernel(blockElems_, bsElemF32);
    const int64_t xF32Bytes = xF32Elems * 4;
    const int64_t reduceBufBytes = aBundleAlignElems_ * 4;
```

CopyIn（整块单段搬入，ARA 行距=dense）：

```cpp
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxKernel<DType>::CopyIn(
    int64_t gmOff, int64_t elems, LocalTensor<D_T>& dst)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(elems * elemBytes_);
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    // 行内非 32B 对齐尾：rightPadding 把 dummy 拉到 paddingValue 路径
    uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = (misalign == 0) ? 0 : static_cast<uint8_t>((kBlockBytes - misalign) / elemBytes_);
    DataCopyPadExtParams<D_T> padParams{/*isPad=*/true, /*leftPadding=*/0, rPad, NegInfD<D_T>()};
    DataCopyPad(dst, xGm_[gmOff], ext, padParams);
}
```

Cast（ARA 整块 flat cast，宽度=blockElems）：

```cpp
    // ── Cast D_T → fp32（V，asc_vf_call）──
    auto xF32 = xF32Buf_.Get<float>();
    __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
    __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
    const uint32_t castCount = static_cast<uint32_t>(vfCount_);
    const uint16_t castReps = static_cast<uint16_t>(CeilDivKernel(castCount, kRepF32));
    asc_vf_call<CastInVf<D_T>>(xF32Ptr, xPtr, castCount, kRepF32, castReps);
```

ReduceMax ARA 分支（`srcShape[1]=aBundle_` + `srcInnerPad=true`，与 dense 布局矛盾）：

```cpp
        } else {
            // ARA：[R, A1]，沿外层 R 归约 → A1 个值
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundle_);
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(
                maxDst, xF32, srcShape, /*srcInnerPad=*/true);
        }
```

Sub+Exp ARA 分支（dense 行距 `r*aBundle_`、未对齐宽度）：

```cpp
    } else {
        // ARA：每行向量 max，循环 R 行
        __ubuf__ float* maxPtr = reinterpret_cast<__ubuf__ float*>(maxDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundle_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t off = r * aBundle_;
            asc_vf_call<SubExpVfRow<D_T>>(ePtr + off, xF32Ptr + off, maxPtr, cnt, kRepF32, reps);
        }
    }
```

ReduceSum ARA 分支（同样 `srcInnerPad=true`）：

```cpp
        } else {
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundle_);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(
                sumDst, eDst, srcShape, /*srcInnerPad=*/true);
        }
```

Div+Cast ARA 分支（dense 行距）：

```cpp
    } else {
        __ubuf__ float* sPtr = reinterpret_cast<__ubuf__ float*>(sumDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundle_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t off = r * aBundle_;
            asc_vf_call<DivCastVfRow<D_T>>(yPtr + off, ePtr + off, sPtr, cnt, kRepF32, reps);
        }
    }
```

**after（修复后，ARA 路径关键代码）：**

成员声明新增 `aBundleAlignD_`：

```cpp
    int64_t rAlignElems_ = 0;   // AR：R padded 到 D_T 32B 对齐（元素）
    int64_t aBundleAlignElems_ = 1; // max/sum buf 元素数（fp32 32B 对齐）；ARA fp32 行 stride
    int64_t aBundleAlignD_ = 1;     // ARA：D_T 32B 对齐行 stride（元素）
    int64_t vfCount_ = 0;       // VF 处理元素数：AR=rAlignElems，ARA 块整体=blockElems（按行调用）
    int32_t elemBytes_ = 0;
```

Init 尺寸计算（ARA 按 padded 行 stride 分配缓冲）：

```cpp
    rAlignElems_ = CeilAlignKernel(rTotal_, bsElemD);                        // AR：D_T 对齐 padded
    aBundleAlignElems_ = CeilAlignKernel(aBundle_, bsElemF32);               // max/sum buf（fp32 对齐）
    if (aBundleAlignElems_ < bsElemF32) {
        aBundleAlignElems_ = bsElemF32; // 至少 32B
    }
    aBundleAlignD_ = CeilAlignKernel(aBundle_, bsElemD);                     // ARA：D_T 行 stride
    if (aBundleAlignD_ < bsElemD) {
        aBundleAlignD_ = bsElemD; // 至少 32B
    }
    vfCount_ = isAr_ ? rAlignElems_ : blockElems_; // AR 处理 padded（含 -inf pad）；ARA 按行调用

    // GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));

    // UB buffer 尺寸
    // AR：单行 padded；ARA：按 padded 行 stride 分配（每行 32B 对齐，保证 per-row VF 对齐）
    const int64_t xBufElems = isAr_ ? rAlignElems_ : (rTotal_ * aBundleAlignD_);
    const int64_t xBufBytes = xBufElems * elemBytes_;
    const int64_t xF32Elems = isAr_ ? rAlignElems_ : (rTotal_ * aBundleAlignElems_);
    const int64_t xF32Bytes = xF32Elems * 4;
    const int64_t reduceBufBytes = aBundleAlignElems_ * 4;
```

CopyIn ARA 分支（单次 DataCopyPad 多块搬运，每行自动 pad 到 32B，UB 行 stride=padded block）：

```cpp
    if (!isAr_) {
        // ARA：单次 DataCopyPad 多块搬运，每行自动 pad 到 32B（-inf），UB 行 stride = padded block
        DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(aBundle_ * elemBytes_);
        ext.blockCount = static_cast<uint16_t>(rTotal_);
        ext.srcStride = 0;  // GM dense（行间无 gap）
        ext.dstStride = 0;  // UB padded block = 行 stride（pad 后无额外 gap）
        ext.rsv = 0;
        uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
        uint8_t rPad = (misalign == 0) ? 0
            : static_cast<uint8_t>((kBlockBytes - misalign) / elemBytes_);
        DataCopyPadExtParams<D_T> padParams{/*isPad=*/true, /*leftPadding=*/0, rPad, NegInfD<D_T>()};
        DataCopyPad(dst, xGm_[gmOff], ext, padParams);
        return;
    }
```

Cast ARA 分支（逐行处理 padded 宽度）：

```cpp
    } else {
        // ARA：逐行 cast padded 宽度（aBundleAlignElems_），padding 元素 -inf 经 cast 后仍 -inf
        const uint32_t cnt = static_cast<uint32_t>(aBundleAlignElems_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t srcOff = r * aBundleAlignD_;
            int64_t dstOff = r * aBundleAlignElems_;
            asc_vf_call<CastInVf<D_T>>(xF32Ptr + dstOff, xPtr + srcOff, cnt, kRepF32, reps);
        }
    }
```

ReduceMax ARA 分支（对齐宽度 + `srcInnerPad=false`）：

```cpp
        } else {
            // ARA：[R, A1_align]，沿外层 R 归约 → A1_align 个值（padding 列 -inf 不影响 valid 列）
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundleAlignElems_);
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(
                maxDst, xF32, srcShape, /*srcInnerPad=*/false);
        }
```

Sub+Exp ARA 分支（padded 宽度 + padded 行距）：

```cpp
    } else {
        // ARA：每行向量 max，循环 R 行，处理 padded 宽度（padding 列 exp(-inf-(-inf))=NaN，不影响 valid 列）
        __ubuf__ float* maxPtr = reinterpret_cast<__ubuf__ float*>(maxDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundleAlignElems_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t off = r * aBundleAlignElems_;
            asc_vf_call<SubExpVfRow<D_T>>(ePtr + off, xF32Ptr + off, maxPtr, cnt, kRepF32, reps);
        }
    }
```

ReduceSum ARA 分支（对齐宽度 + `srcInnerPad=false`）：

```cpp
        } else {
            srcShape[0] = static_cast<uint32_t>(rTotal_);
            srcShape[1] = static_cast<uint32_t>(aBundleAlignElems_);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(
                sumDst, eDst, srcShape, /*srcInnerPad=*/false);
        }
```

Div+Cast ARA 分支（eBuf 用 fp32 padded 行距、yBuf 用 D_T padded 行距）：

```cpp
    } else {
        // ARA：每行向量 s，处理 padded 宽度（padding 列 NaN/NaN=NaN，不写出 GM）
        __ubuf__ float* sPtr = reinterpret_cast<__ubuf__ float*>(sumDst.GetPhyAddr());
        const uint32_t cnt = static_cast<uint32_t>(aBundleAlignElems_);
        const uint16_t reps = static_cast<uint16_t>(CeilDivKernel(cnt, kRepF32));
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t eOff = r * aBundleAlignElems_;
            int64_t yOff = r * aBundleAlignD_;
            asc_vf_call<DivCastVfRow<D_T>>(yPtr + yOff, ePtr + eOff, sPtr, cnt, kRepF32, reps);
        }
    }
```

CopyOut ARA 分支（逐行仅写有效元素回 dense GM）：

```cpp
    if (!isAr_) {
        // ARA：逐行从 padded UB 写出到 dense GM
        for (int64_t r = 0; r < rTotal_; ++r) {
            int64_t gmRowOff = gmOff + r * aBundle_;
            int64_t ubRowOff = r * aBundleAlignD_;
            DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aBundle_ * elemBytes_);
            ext.blockCount = 1;
            ext.srcStride = 0;
            ext.dstStride = 0;
            ext.rsv = 0;
            DataCopyPad(yGm_[gmRowOff], src[ubRowOff], ext);
        }
        return;
    }
```

C++ ST 复现用例（加入 `tests/aclnn/test_aclnn.cpp` 的用例表，含对照与 5D 排除项）：

```cpp
    // ── 复现：ARA 非对齐内维（NO_OUTPUT 失败集）──
    // (65,63) dim=0/-2 → ARA [1,65,63]，A1=63 非 32B 对齐
    { int64_t s[] = {65, 63}; v.push_back(makeCase("R65_63_bf16_d0","R",Dt::Bf16,2,s,0,Kind::Normal)); }
    { int64_t s[] = {65, 63}; v.push_back(makeCase("R65_63_fp16_d0","R",Dt::Fp16,2,s,0,Kind::Normal)); }
    { int64_t s[] = {65, 63}; v.push_back(makeCase("R65_63_fp32_d0","R",Dt::Fp32,2,s,0,Kind::Normal)); }
    { int64_t s[] = {65, 63}; v.push_back(makeCase("R65_63_bf16_dm2","R",Dt::Bf16,2,s,-2,Kind::Normal)); }
    // (67,61) dim=0/-2 → ARA [1,67,61]，A1=61 非 32B 对齐
    { int64_t s[] = {67, 61}; v.push_back(makeCase("R67_61_bf16_d0","R",Dt::Bf16,2,s,0,Kind::Normal)); }
    { int64_t s[] = {67, 61}; v.push_back(makeCase("R67_61_fp16_d0","R",Dt::Fp16,2,s,0,Kind::Normal)); }
    { int64_t s[] = {67, 61}; v.push_back(makeCase("R67_61_fp32_d0","R",Dt::Fp32,2,s,0,Kind::Normal)); }
    { int64_t s[] = {67, 61}; v.push_back(makeCase("R67_61_bf16_dm2","R",Dt::Bf16,2,s,-2,Kind::Normal)); }
    // (65,64) dim=0 → ARA [1,65,64]，A1=64 32B 对齐（对照应通过）
    { int64_t s[] = {65, 64}; v.push_back(makeCase("R65_64_bf16_d0","R",Dt::Bf16,2,s,0,Kind::Normal)); }
```

padding 值语义（机制链，逐列独立）：`-inf` padding 列经 Cast 后仍 `-inf`；ReduceMax 沿 R 逐列取 max，padding 列结果 `-inf` 不影响 valid 列；Sub+Exp 阶段 padding 列 `exp(-inf - max)`：valid 列 max 为有限值时 `exp(-inf)=0`，全 padding 列（max=-inf）时 `exp(-inf-(-inf))=NaN`——但 padding 列不参与 valid 列的逐列求和、最终 `CopyOut` 也不写出，故对 GM 输出无影响。

## 设计决策

- **为何选 UB 内 padding 而非别的层**：保持 tiling 序列化与 GM 布局不动（GM 是 dense 的、由框架决定），只在 kernel 内部 UB 布局上做 32B 行对齐，改动面最小且向量+归约主路径完整保留。
- **CopyIn 两种写法的取舍**：先试了"逐行 for 循环、每行一次 `DataCopyPad`（blockCount=1）"（第一次修复尝试），后改为"单次 `DataCopyPad` + `blockCount=rTotal_` 多块搬运 + 逐行自动 right padding"——后者一条 MTE 指令完成整块搬运，且 `dstStride=0` 时 UB 行距自动等于 padded block 长度，与 `aBundleAlignD_` 一致。两者都编译通过，但第一种配合当时未改宽度的向量/归约调用依然全 0（见坏实践）。
- **被调查后放弃的方向**：非对齐 `Reg::Load/Reg::Store`（kernel_reg_compute_datacopy_intf.h 提供 `Load(U& dstReg, __ubuf__ T* srcAddr)` 无对齐版本、`Store` 带 count 版本）——调查发现 `Reg::Load` 无 mask、总是装载整寄存器（64 个 fp32），且该路径解决不了 Reduce API 的 `srcInnerPad` 语义问题，放弃；改走 padding 路线后不再需要。
- **对照用例方法论**：修复前先加 `(65,64)` 对齐对照用例，单变量证明根因绑定 A1 对齐性，避免在错误方向（多核循环计数、尾块长度）上浪费时间——验收报告原本怀疑的方向是 `aBigCoreLoopCnt/aSmallCoreLoopCnt` 在 aTotal=65/67 非核数整数倍时的循环边界，实测被对照用例否定。
- **当前方案路线档位**：`recommended_fastpath`。

## 系统信息

- 工具链：CANN 9.1.0，SOC Ascend950PR；TTK kernel 模式（ops-test-kit 安装根，环境变量 `$TTK`）；C++ ST 工程 `tests/aclnn`（`./run.sh` 构建+执行）。
- 目录布局：`operators/BenchHarnessSoftmax_package/{op_kernel,op_host,tests/aclnn,tests/whitebox}`。
- 相关文件（与本次读写一一对应）：`op_kernel/bench_harness_softmax.h`（修复点）、`op_kernel/bench_harness_softmax_tiling.h`、`op_host/bench_harness_softmax_host.cpp`、`tests/aclnn/test_aclnn.cpp`（复现用例）、`tests/whitebox/branch-0/ttk_design_doc_cases.csv`（980 条输入）、`tests/whitebox/ttk/result.csv`（TTK 结果）、`tests/whitebox/ttk/whitebox_failures.md`（失败报告）。

## 验证信号

问题与验证来源：在一次算子开发会话的 Branch-0 白盒联调中，TTK kernel 模式全量跑出 24 条 NO_OUTPUT 后，先在 C++ ST 工程复现并加对照用例定位，修复后回归。以下均为本会话实测：

- 修复前 C++ ST 复现：8 条 ARA 非对齐用例（(65,63)/(67,61) × dim=0/-2 × bf16/fp16/fp32）全部 `FAIL (rtol/atol fail at 0)`（输出全 0），对照 `(65,64)` PASS、5D `(2,2,3,4,4) dim=-1` PASS（来源平台：950；证据等级：tool_observed）。
- 修复后 C++ ST：`Branch-0 ST summary: 32/32 cases PASSED`，含全部 ARA 复现用例与既有 S/B/F/E 系列（来源平台：950；证据等级：tool_observed）。
- 修复后 TTK kernel 模式全量白盒：`total=980 pass=980 fail=0 skip=0`、result.csv 全 `precision_status=PASS`（来源平台：950；证据等级：tool_observed）。
- 修复后 C++ ST 回归（该分支 ST 自测全量重跑）：32/32 PASS，无退化（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复只覆盖 ARA（归约轴非末轴、合轴后 `[R, A1]`）路径；AR（末轴归约）与 ChunkedR（大 shape，tilingKey=1）分支未改动。
- padding 用 `-inf` 依赖 max-subtraction 数值稳定算法：`max(-inf, x)=x` 使 padding 列不抬高 valid 列的 max。若算子语义不能用 `-inf` 作 padding neutral 值（如 min 归约、或输入本身可能全 `-inf` 需输出 NaN 的语义已由逐列独立保证），padding 值需另行论证。
- UB 占用增大：ARA 缓冲从 `CeilAlign(R×A1)` 变为 `R×AlignUp(A1)`，A1 很小而 R 很大时 padding 开销比例上升（本例 A1=63→64 开销可忽略）；UB 预算紧张的算子需重算。
- 未验证，推断可迁移：任何以 `Pattern::Reduce::RA` + 逐行向量算子处理非 32B 对齐内维的 AscendC kernel；其他 NPU 平台未验证。

**未决项**：

- **优先项：复现并追根因到机制层**——"输出全 0"的硬件级微观路径未隔离：最终修复一次性变更 CopyIn/Cast/ReduceMax/Sub+Exp/ReduceSum/Div+Cast 六处并一次构建通过，未单独验证"只改 Reduce 的 srcShape[1]/srcInnerPad"或"只改向量算子宽度"各能否独立修复，无法断言具体是哪条指令静默失败。下个 agent 可做单变量消融（逐项回退）定位。
- TTK 首跑那条 `INVALID_TILING`（5D）的 TTK 侧机制未追。
- padding 列在全 `-inf` 输入语义下的 NaN 传播路径已有逐列论证，但未单独构造"整行全 -inf"用例实测。
## 坏实践

- **只 pad 拷贝布局、不改算子宽度**：第一次修复尝试把 CopyIn/CopyOut 改成 padded 行 stride 逐行搬运，但 Cast/Sub+Exp/Div+Cast 仍用 `cnt=aBundle_`、Reduce 仍 `srcShape[1]=aBundle_/srcInnerPad=true`——重建后 8 条 ARA 用例依旧全 0。padding 布局、向量算子宽度、Reduce 的 `srcShape[1]` 与 `srcInnerPad` 四者必须一致，缺一无效。
- **先怀疑构建缓存**：失败后清 `build/` 与 kernel_cache 重试仍失败——本案是代码语义问题，不是缓存陈旧；缓存清理只应作为低成本排除步骤，不应当作修复。
- **按验收报告的"疑似方向"直接改多核循环计数**：报告怀疑 `aBigCoreLoopCnt/aSmallCoreLoopCnt` 非整除边界，但 (65,64) 对照用例（同样 65 行非 56 核整数倍）通过，一次对照即否定该方向。

## 误诊

- TTK 同批 25 条失败中的 1 条 `INVALID_TILING`（5D `(2,2,3,4,4)` bf16 `dim=-1`，`dyn_tiling_key=OPTILING_FAILURE`）一度被疑为 TilingFunc 对 5D 合轴/MAX_PATTERN_RANK=3 的处理缺陷；C++ ST 框架同 shape 复现直接 PASS（修复前即通过），最终 TTK 全量复跑亦通过。该条与 ARA 根因无关，归为 TTK 环境噪音（确切机制未追，属 agent 推断）。
