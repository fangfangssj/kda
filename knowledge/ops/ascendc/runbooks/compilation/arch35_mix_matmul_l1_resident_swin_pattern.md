---
type: "Runbook"
title: arch35 MIX Matmul L1 常驻接入模式（swin 参考）
description: DAV_3510/arch35 MIX 模式下 Matmul 高阶 API 的参考接入模式：kernel 入口 SetSysWorkspace→GetUserWorkspace→REGIST_MATMUL_OBJ，A/B 用 TSCM+NZ，CFG_ENVECND2NZ+SetLocalWorkspace，K 切分 enAtomic 累加；附 AIC/AIV 职责守卫与方案取舍。落地后编译/UT/Mock/507015 消除层验证通过。
tags: [task.implementation, topic.architecture, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# arch35 MIX Matmul L1 常驻接入模式（swin 参考）

## 现象（任务背景）

在 arch35（Ascend950PR / DAV_3510，MIX 模式 AIC+AIV，taskRation 1:2）上开发含 Matmul 阶段的算子（三维卷积反传 filter 梯度，im2col 路线）时，最初设计直接照通用文档写法：A/B 用 `MatmulType<TPosition::GM, CubeFormat::ND, half>` + 默认 `CFG_NORM` 模板。该写法在 arch35 MIX 模式下被解析为 `MatmulServiceAux(AIC) + MatmulClient(AIV)` 服务模型，与本算子 GM 输入结构不兼容，表现为 AIV 异常（ret=507015）或流同步挂起（行为不稳定，负结果矩阵见 950 Matmul 选型卡的 GM 输入增量）。本卡沉淀排查后确定的正确接入模式——以 CANN 内置 `swin_attention_score_quant`（DAV_3510 已验证运行的实现）为参考蓝本，并在本算子上完整落地。

## 适用场景

- 工程形态：算子仓开发（手写 AscendC kernel），Matmul 阶段需在 arch35/DAV_3510 MIX 模式运行，或多 dtype（half/bf16）模板实例化。
- API 路径：`Matmul<>` 高阶 API + `MatmulType` + `GetNormalConfig`；`REGIST_MATMUL_OBJ`；`SetSysWorkspace`/`GetUserWorkspace`/`GetSysWorkSpacePtr`；`TQue<TPosition::A1/B1>`；`SetLocalWorkspace`；`CopyND2NZOnTheFly`；`IterateAll(gm, enAtomic)`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35，MIX taskRation 1:2）+ CANN cann-9.1.0。参考实现为 CANN 内置源码（`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/swin_attention_score_quant/`）。
- 适用粒度：operator_family（所有在 DAV_3510/arch35 MIX 下用 Matmul 高阶 API 的算子）。
- 路线优先级：recommended_fastpath（相对手动 Mmad 与 ASCENDC_CUBE_ONLY 是当前唯一参考级路径）。
- 明确不适用 / 未验证场景：非 MIX（纯 AIV/AIC）形态；A3/A2 等其他架构（参考实现的 MatmulType 语义可能不同，未验证）；本轨迹 ST Real 端到端精度受框架 workspace 传参问题阻塞未通过（见"失效边界"），完整 E2E 结论待该问题解决后补验。

## 根因/解决配方（模式要点）

arch35 MIX 下 Matmul 高阶 API 的服务模型要求业务侧重排数据通路：A/B 不能以 GM(ND) 直入，须业务负责搬运到 L1 并转 NZ 格式，engine 的 ND2NZ 协作 workspace 必须正确供给。完整模式分五部分：

**1）kernel 入口（swin 模式，参考 swin_attention_score_quant.cpp:22-59）**——顺序固定：

```cpp
extern "C" __global__ __aicore__ void swin_attention_score_quant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, /* ... */, GM_ADDR attention_score,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    AscendC::SetMaskNorm();
    AscendC::SetSysWorkspace(workspace);          // 必须在 REGIST_MATMUL_OBJ 前
    GET_TILING_DATA(tiling_data, tiling);

    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);  // 用户数据放 user 区

    if (TILING_KEY_IS(INT8_NOMASK_MODE)) {
        SwinAttentionScoreQuant<int8_t, half, uint64_t, int32_t, half, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkBmm, &tiling_data.qkBmmTilingData, op.pvBmm,
            &tiling_data.pvBmmTilingData);       // kernel 入口、栈上 TPipe、Init 之前
        op.Init(query, /* ... */, attention_score, user, &tiling_data, &tPipe);
        op.Process();
    }
}
```

**2）Matmul 类型与配置**——A/B 常驻 L1(TSCM) + NZ；C 出 GM(ND)；第 5 模板参数启用 vector ND2NZ：

```cpp
constexpr MatmulConfig CFG_ENVECND2NZ = GetNormalConfig(false, false, true);
// A[M,K] 在 L1(TSCM)，NZ 格式，不转置
using aType = MatmulType<TPosition::TSCM, CubeFormat::NZ, D_T_X, false>;
// B 转置存储 [N,K]（ISTRANS=true，Matmul 视为 B[K,N]），在 L1(TSCM)，NZ 格式
using bType = MatmulType<TPosition::TSCM, CubeFormat::NZ, D_T_X, true>;
// C[M,N] float，输出到 GM(ND)（fp32 累加）
using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
Matmul<aType, bType, cType, cType, CFG_ENVECND2NZ> mm_;
```

**3）buffer 布局**——`REGIST_MATMUL_OBJ` 后、InitBuffer 顺序（参考 swin Init 62-72 行与 217-220 行）：

```cpp
    // 初始化 Matmul 对象（必须在 InitBuffer 之前）
    REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_, &tilingData->cubeTiling);
    // cubeShared/vecShared：Matmul ND2NZ workspace + ND→NZ 中转
    pipe_.InitBuffer(cubeSharedTBuf_, tilingData->cubeSharedUbSize);
    pipe_.InitBuffer(vecSharedTBuf_, tilingData->vecSharedUbSize);
    // L1 TQue（A1/B1，NZ 格式）
    pipe_.InitBuffer(qL1TQue_, 1, tilingData->aL1BufSize);
    pipe_.InitBuffer(kL1TQue_, 1, tilingData->bL1BufSize);
    // 其余 UB TBuf（grad/im2col 暂存、Cast src/dst 等）
```

**4）Matmul 计算（SetLocalWorkspace 一次 + M×N outer loop + K 切分 enAtomic 累加）**：

```cpp
    // Process 开头：设置 ND2NZ workspace（仅一次）
    LocalTensor<uint8_t> cubeSharedUb = cubeSharedTBuf_.template Get<uint8_t>();
    mm_.SetLocalWorkspace(cubeSharedUb);

    // M×N outer loop（blockIdx 步进 coreNum），K 方向 kLoops 循环：
    for (int32_t kIdx = 0; kIdx < kLoops; kIdx++) {
        LocalTensor<D_T_X> aL1 = qL1TQue_.template AllocTensor<D_T_X>();
        Conv3dCopyND2NZOnTheFly<D_T_X>(aL1, gmWsGrad_[mStart * K_ + kStart], 0, 0, curM, curK, K_);
        LocalTensor<D_T_X> bL1 = kL1TQue_.template AllocTensor<D_T_X>();
        Conv3dCopyND2NZOnTheFly<D_T_X>(bL1, gmWsIm2col_[nStart * K_ + kStart], 0, 0, curN, curK, K_);
        mm_.SetTail(curM, curN, curK);
        mm_.SetOrgShape(roundM, roundN, roundK);   // 16 对齐后的原始 shape
        mm_.SetTensorA(aL1);
        mm_.SetTensorB(bL1, true);                  // isTransB=true
        uint8_t enAtomic = (kIdx == 0) ? 0 : 1;     // 首写，后续 AtomicAdd 累加
        mm_.IterateAll(gmWsC_[mStart * N_ + nStart], enAtomic);
        mm_.End();
        qL1TQue_.FreeTensor(aL1);
        kL1TQue_.FreeTensor(bL1);
    }
```

**5）AIC/AIV 职责守卫**（MIX 模式硬约束）：

- `DataCopyPad`（MTE2/MTE3）与 `Cast` 仅 AIV 执行——AIC 无 MTE 硬件，误执行会挂起；
- Matmul 的 MMAD 仅 AIC 执行——AIV 无 Cube 硬件；
- `CopyND2NZOnTheFly`（GM→L1 的 DataCopy + 内部 `SetFlag/WaitFlag<MTE2_MTE1>`/`<MTE1_MTE2>` 同步）在业务重排中由 AIC 侧调用（swin 原版在两类核一致的 Process 内由引擎分派；本算子落地为显式 AIC 侧搬运 + AIV 不重复搬运）。其 swin 原版关键段（swin_attention_score_quant.h:195-213）：

```cpp
template <typename DTYPE>
__aicore__ inline void CopyND2NZOnTheFly(const AscendC::LocalTensor<DTYPE> &dst,
    const AscendC::GlobalTensor<DTYPE> &src, const int row,
    const int col, const int height, const int width, const int gCol)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int calcWidth = width / c0Size;
    int64_t dstOffset = 0;
    int64_t srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
    int calcWidthExr = DivUp(width, c0Size);
    int calcHeightExr = DivUp(height, BLOCK_NUM_PER_FRACTAL);
    if (height % BLOCK_NUM_PER_FRACTAL != 0) {
        int64_t repeat = calcWidthExr * calcHeightExr;
        AscendC::LocalTensor<int16_t> tmp = dst.template ReinterpretCast<int16_t>();
        AscendC::InitConstValueParams<int16_t> initConstValueParams;
        initConstValueParams.repeatTimes = (uint16_t)repeat;
        initConstValueParams.initValue = 0;
        InitConstValue(tmp, initConstValueParams);
        AscendC::PipeBarrier<PIPE_MTE2>();   // 注意：PIPE_MTE2 无 AscendC:: 前缀
    }
    /* ...DataCopy 分块搬运 + MTE2_MTE1/MTE1_MTE2 SetFlag/WaitFlag 同步... */
}
```

配套 Host 侧：`SetMatmulConfigParams(0, false, matmul_tiling::ScheduleType::INNER_PRODUCT, matmul_tiling::MatrixTraverse::NOSET, true)`（注意命名空间前缀）；A/B `SetAType/SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, ...)`；**手动补 `transLength`**（`MultiCoreMatmulTiling` 不计算该字段，`transLength = baseM * baseN * sizeof(float)`，作为 cubeSharedUbSize，见 SetLocalWorkspace transLength 卡）；大 K 手动 L1 预算 + K 切分。

## 设计决策

- **方案 A（本模式，L1 常驻 + ND2NZ，swin 风格）vs 方案 B（手动 Mmad + LoadData）vs 方案 C（ASCENDC_CUBE_ONLY）**：
  - B 被放弃：需自管 L1/L0A/L0B/L0C/UB 全链搬运、自实现 ND2NZ/双缓冲/流水，复杂度极高、文档示例稀少，性能未必优。
  - C 被放弃（实测否决）：单 kernel `ASCENDC_CUBE_ONLY` 变 AIC-only，AIV 阶段（生成/Cast）被跳过，输出全 0；改双 kernel 又引入拆分调度复杂度。
  - A 被采用：官方内置实现已验证路径，数据流（im2col + Matmul）契合，改动集中在 Matmul 阶段。
- **AIV 守卫的取舍**：GM+ND 失败路径上实测"无守卫→流同步挂起（timeout 124），加 `if ASCEND_IS_AIV return`→转为 fast-fail（507015）或静默全 0"——守卫不能修复服务模型问题，但把不可诊断的挂起变成可诊断的失败，联调期建议保留。
- **L1 TQue 是否走 EnQue/DeQue**：swin 原版 AllocTensor→CopyND2NZOnTheFly→SetTensorA/B→IterateAll→End→FreeTensor，无 EnQue/DeQue（CopyND2NZOnTheFly 内部已做 MTE2_MTE1 同步）；跟随参考实现，避免冗余同步。
- **SetLocalWorkspace 位置**：swin 在 Process 开头调用一次；放在循环内反复调用无收益。

## 预防

- DAV_3510/arch35 MIX 上用 Matmul 高阶 API，设计期即按本模式立项（勿用 GM(ND) A/B + 默认模板起步）。
- 移植参考实现时保持符号书写（`PIPE_MTE2` 无前缀等）与调用顺序（SetSysWorkspace→REGIST_MATMUL_OBJ→InitBuffer）。

## 验证信号

本验证来自该算子 L1 常驻方案的落地与回归（Ascend950PR）。

- 参考实现事实：swin_attention_score_quant 为 CANN 内置、DAV_3510 上运行的实现（源码逐行读取核对，含 kernel 入口/类型定义/同步细节）（来源平台：950；证据等级：artifact_verified——固定安装源码的逐字读取）。
- 模式落地后：`bash build.sh --soc=ascend950 -j8` 编译通过（清理重建后无 error）；op_host UT 10/10（含大 K 用例 sK=6544/kLoops=2）；ST Mock（CPU golden）5/5 `MERE=0.00e+00`（来源平台：950；证据等级：tool_observed）。
- SetLocalWorkspace + transLength 补齐组合下，ST Real 不再出现 ret=507015 AIV 异常（来源平台：950；证据等级：tool_observed）。
- ST Real 端到端精度未通过：5/5 输出全 0，受独立的框架 workspace 传参问题（DynamicCompileStaticFlag 传参错位）阻塞，见同轨迹对应卡（来源平台：950；证据等级：tool_observed）。

## 性能影响

无性能数据（本轨迹无 profiling 信号）；相对代价仅定性：相对 GM 直入方案新增 GM(ND)→L1(NZ) 转换搬运与 K 切分多次 IterateAll，换取 MIX 服务模型兼容性。

## 失效边界

- 本卡验证范围：编译/UT/Mock/异常消除层；**NPU 端到端精度未验证**（阻塞于框架传参问题，修复后须复验）——采用本模式不能保证 ST Real 通过，只能保证不踩 GM 输入服务模型与 ND2NZ workspace 两类坑。
- GetUserWorkspace 在"测试侧 aclrtMalloc 裸 workspace"配置下返回无效指针（本轨迹实测触发 507015）；该子环节在 workspace 布局未含 system 区时不可用（另见纯 AIV 写 workspace 卡）。
- 模式限定 CANN 9.1.0 的 swin 实现；跨版本 API 语义未验证（未验证，推断需复核）。
- AIV 侧是否必须执行到 SetLocalWorkspace（MatmulClient 协作）未取得机制层证据，守卫布局以试验收敛为准。

**未决项**：

- 框架 workspace 传参问题解决后，复跑 ST Real 验证本模式端到端正确性（最高优先接续项）。
- AIC/AIV 在 ND2NZ 协作中的精确分工（AIV 是否须执行 SetLocalWorkspace）待机制层证据。
- qkBmm（默认模板）与 pvBmm（CFG_ENVECND2NZ）在 swin 中的差异化用法（为何 qk 不带第 5 参数）未深究。
## 坏实践

- 在 arch35 MIX 上用 `MatmulType<TPosition::GM, CubeFormat::ND, ...>` + 默认 CFG_NORM 直入（触发服务模型异常/挂起）。
- 期望用 `ASCENDC_CUBE_ONLY` 一键规避 MIX 协调（AIV 阶段被跳过，输出全 0）。
- 对 `static_cast<D_T_X>(float)` 类调试代码不加 `if constexpr` dtype 约束（bf16 实例化编译失败）。
- 移植 swin 时"统一命名空间前缀"（`AscendC::PIPE_MTE2` 编译错）或自作主张加外部 EnQue/DeQue 同步。

## 误诊

- 前序诊断"Matmul L1 错误 / SetBufferSpace 超容"——被实测证伪（平台正确上报 L1=512KB/L0C=256KB/UB=248KB，cubeTiling 占用均在规格内）；真实问题是服务模型/接入方式。
- "GET_TILING_DATA_WITH_STRUCT 编译期烘焙 tiling"的表述——比对同样使用该宏的内置 conv3d_backprop_filter_v2 后判定为可疑误解（运行时 tiling 正常；烘焙只在 DynamicCompileStaticFlag(true) 下发生，见对应卡）。
