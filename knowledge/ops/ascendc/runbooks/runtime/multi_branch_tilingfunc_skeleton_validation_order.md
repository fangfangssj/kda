---
type: "Runbook"
title: TilingFunc 骨架：校验→空→组→基与收尾三件套
description: 多分支 host TilingFunc 骨架——校验顺序 dtype→rank→shape（rank 先于索引防越界）→空张量短路→group 判定→默认基；收尾 SetTilingKey/SetBlockDim(max(usedCoreNum,1))/ws=sys+usr；含大小核均衡三元组与注册机制选择。
tags: [task.implementation, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingFunc 骨架：校验→空→组→基与收尾三件套


## 现象

ThnnFusedGruCellGrad（GRU 单步融合反向算子，ascend950/arch35，ops-registry-invoke 算子仓流程）的公共 TilingFunc 处于桩阶段：33 条公共 TilingUT 全红（红基线 0 passed / 33 failed——tiling 函数已从安装包 .so 解析成功、失败是真实断言失败：桩不填任何 TilingData 字段）。任务是从零实现完整 TilingFunc 使 UT 全绿，同时不回归既有 build.sh 与 tests/aclnn 最小通路。

骨架要一次性拿对四件事，每一件都有负向 UT 用例盯着：

1. **输入校验顺序**：6 条拒绝用例（fp64 dtype、fp32+fp16 混搭、rank 1、rank 3、storage 宽 ≠5H、batch 不匹配）全部要在 tiling 阶段以 `GRAPH_FAILED` 拒绝，且**任一失败不得进入任何 kernel 分支**；
2. **分支判定顺序**：先校验 → 再空张量 → 后组判定 → 默认基（顺序错了会把非法输入送进切分公式）；
3. **收尾三件套**：SetTilingKey / SetBlockDim（严禁 0）/ SetWorkspaceSize（不用 workspace 也要显式设 sys）；
4. **注册机制**：本工程 tiling 函数经 def.cpp 的 `AICore().SetTiling(optiling::TilingFunc)` 绑定（非 IMPL_OP_OPTILING），实现必须保持该绑定不被破坏。

实现后 33/33 一次转绿（重跑一致），build.sh 退出码 0，aclnn 真机冒烟 1/1 不回归，git 变更仅 3 个 tiling 链路文件（tiling cpp/h + struct.h），UT 的 oracle 与期望值零改动（实现向设计对齐，不反向凑绿）。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（host tiling 函数从桩到完整实现；多分支算子的 TilingFunc 骨架）
- API 路径：`gert::TilingContext`（GetInputShape/GetInputTensor/GetAttrs/GetAttrPointer<bool>/GetPlatformInfo/GetTilingData<T>/SetTilingKey/SetBlockDim/GetWorkspaceSizes）；`platform_ascendc::PlatformAscendC`；OpDef 注册 `AICore().SetTiling(optiling::TilingFunc)`（`op_host/*_def.cpp`）
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致；UT 为 mock 平台、aclnn 为真机）
- 适用粒度：`operator_family`（带输入校验/空张量/分组分支的 host tiling 骨架）
- 路线优先级：`recommended_fastpath`（骨架顺序是负向用例能否全部拦截的决定因素）
- 明确不适用/未验证：`GetCompileInfo<T>()` 路径（设计注明 aclnn 通路不调用 TilingParse、恒返回 null——本会话未直接观测该 null，见误诊/设计决策）；TilingParse/CompileInfo 注册流（本工程未用）

## 根因

骨架的三条硬约束：

1. **rank 校验必须先于维度索引**：`GetDim(1)` 在 rank-1 的 `gert::Shape` 上是越界读——`RejectRankOne` 用例（grad_hy rank 1、storage rank 2）如果先读 dim 再查 rank，行为未定义而非干净拒绝。
2. **校验失败必须短路**：设计（BranchRoute §2）规定"先校验 → 再空 → 后组 → 默认基"，校验失败的输入（dtype/shape 非法）进任何切分公式都会产出垃圾 TilingData，比拒绝更糟。
3. **blockDim 严禁 0**：`SetBlockDim(0)` 是非法调度参数；`aLoopCntTotal` 小于核数时 `usedCoreNum` 可能是"只有大核"的少数，必须 `max(usedCoreNum, 1)`。

## 解决配方

TilingFunc 入口编排（verbatim，取自终版；注释里的 § 编号是源算子设计文档索引）：

```cpp
// 公共 TilingFunc 入口（经 OpDef AICore().SetTiling 绑定）：按「先校验 → 再空 →
// 后组 → 默认基」分叉（BranchRoute.md §2 判定顺序；HostTiling.md §9）
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ThnnFusedGruCellGradCtx ctx;

    // 硬件参数 + 输入解析与校验（校验顺序 dtype → 维度 → shape，任一失败即拒绝，
    // 不进入任何 kernel 分支）
    ge::graphStatus ret = GetPlatformInfo(context, ctx);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetShapeAttrsInfo(context, ctx);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== 空 tensor 短路（先于合轴四步；EMPTY_A 优先于 EMPTY_R）→ tilingKey=2 =====
    const ThnnFusedGruCellGradEmptyKind kind = ClassifyEmptyTensor(ctx);
    if (kind != ThnnFusedGruCellGradEmptyKind::NORMAL) {
        return HandleEmptyTensor(context, ctx, kind);
    }

    // ===== 合轴四步（§3.3 退化式：pattern 恒 [1(A), B(R), H(A)]，aSplitIdx≡2 / rSplitIdx≡1）=====
    // ===== UB 切分三步 + cacheBuf 钳制环（§4.2 → §4.3 → §4.4 → §4.5）=====
    ComputeAUbFactor(ctx);
    ret = ComputeUbSplit(context, ctx);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== 多核切分（§5.1）=====
    ComputeFusedALoopSplit(ctx);

    // ===== Group 判定（先空后组默认基）→ tilingKey=1 / tilingKey=0 =====
    ctx.isGroup = ShouldUseGroup(ctx);
    if (ctx.isGroup) {
        ret = ComputeGroupSplit(context, ctx);  // 含 SetScheduleMode(1)
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    } else {
        ctx.rGroupCnt = 0;
    }

    // ===== 收尾：填 TilingData + SetTilingKey + SetBlockDim + SetWorkspaceSize（§7）=====
    ret = FillAndLogTilingData(context, ctx);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 位编码（§7.3）：isGroup 占 bit0、isEmptyTensor 占 bit1 → base=0 / group=1
    OP_CHECK_IF(context->SetTilingKey(GET_TPL_TILING_KEY(ctx.isGroup ? 1U : 0U, 0U)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Set tiling key failed"), return ge::GRAPH_FAILED);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, int64_t(1))));  // 严禁 0；≤ 物理核数
    // workspace（§6.3）：group 且 isBias=1 → sys + rGroupCnt×BIAS_SEG_CNT×hiddenSize×4
    //（Tuple Reduce 纪律不乘归约路数 2）；base / group&&isBias=0 → 仅 sys
    const int64_t usrWorkspaceBytes = (ctx.isGroup && (ctx.isBias == 1))
                                          ? ctx.rGroupCnt * BIAS_SEG_CNT * ctx.H * FP32_BYTES
                                          : 0;
    return SetWorkspaceSize(context, usrWorkspaceBytes);
}
```

输入解析与校验（verbatim；顺序 dtype → rank → shape，rank 先于 dim 索引）：

```cpp
// ─── 输入解析 + INFO 全量打印 + 异常值校验（§2/§3.4，顺序 dtype → 维度 → shape）───
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, ThnnFusedGruCellGradCtx& ctx)
{
    const gert::StorageShape* gradHyShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHyShape);
    const gert::StorageShape* storageShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Tensor* gradHyTensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHyTensor);
    const gert::Tensor* storageTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageTensor);
    const ge::DataType gradHyDtype = gradHyTensor->GetDataType();
    const ge::DataType storageDtype = storageTensor->GetDataType();

    // attr has_bias：按索引访问（§3.4 校验 4 / §7.4），缺省 false（OpDef 默认）
    const auto* attrs = context->GetAttrs();
    const bool* hasBiasPtr = (attrs != nullptr) ? attrs->GetAttrPointer<bool>(0) : nullptr;
    ctx.isBias = ((hasBiasPtr != nullptr) && *hasBiasPtr) ? 1 : 0;

    OP_LOGI(context, "ThnnFusedGruCellGrad inputs: grad_hy dtype=%d, storage dtype=%d, has_bias=%d",
            static_cast<int>(gradHyDtype), static_cast<int>(storageDtype), ctx.isBias);

    // 校验 1：dtype ∈ {fp32, fp16, bf16} 且双输入一致（dtype_not_supported）
    const bool gradHyDtypeOk =
        (gradHyDtype == ge::DT_FLOAT) || (gradHyDtype == ge::DT_FLOAT16) || (gradHyDtype == ge::DT_BF16);
    const bool storageDtypeOk =
        (storageDtype == ge::DT_FLOAT) || (storageDtype == ge::DT_FLOAT16) || (storageDtype == ge::DT_BF16);
    OP_CHECK_IF(!gradHyDtypeOk || !storageDtypeOk || (gradHyDtype != storageDtype),
                OP_LOGE(context, "dtype not supported: grad_hy dtype=%d, storage dtype=%d, only fp32/fp16/bf16"
                        " with identical dtypes are supported",
                        static_cast<int>(gradHyDtype), static_cast<int>(storageDtype)),
                return ge::GRAPH_FAILED);
    ctx.dtypeSize = (gradHyDtype == ge::DT_FLOAT) ? static_cast<int64_t>(FP32_BYTES) : 2;

    // 校验 3：rank==2 强约束（rank=0 / rank=3 一律拒绝，shape_mismatch）
    const gert::Shape& gradHyDims = gradHyShape->GetStorageShape();
    const gert::Shape& storageDims = storageShape->GetStorageShape();
    OP_CHECK_IF((gradHyDims.GetDimNum() != 2U) || (storageDims.GetDimNum() != 2U),
                OP_LOGE(context, "shape mismatch: grad_hy rank=%zu, storage rank=%zu, rank 2 required",
                        gradHyDims.GetDimNum(), storageDims.GetDimNum()),
                return ge::GRAPH_FAILED);

    // 校验 5：CheckShapeConsistency 型（非 broadcast）——storage.shape == (B, 5H)
    ctx.B = gradHyDims.GetDim(0);
    ctx.H = gradHyDims.GetDim(1);
    OP_CHECK_IF((storageDims.GetDim(0) != ctx.B) || (storageDims.GetDim(1) != 5 * ctx.H),
                OP_LOGE(context, "shape mismatch: storage (%ld, %ld) != (B, 5H) = (%ld, %ld)",
                        storageDims.GetDim(0), storageDims.GetDim(1), ctx.B, 5 * ctx.H),
                return ge::GRAPH_FAILED);

    OP_LOGI(context, "ThnnFusedGruCellGrad shapes: grad_hy (%ld, %ld), storage (%ld, %ld)",
            ctx.B, ctx.H, ctx.B, 5 * ctx.H);
    return ge::GRAPH_SUCCESS;
}
```

大小核均衡三元组（base 多核切分，verbatim）：

```cpp
// ─── base 多核切分（§5.1）：外层 A loop 扁平线性计数、大小核均衡（负载差 ≤ 1）───
void ComputeFusedALoopSplit(ThnnFusedGruCellGradCtx& ctx)
{
    ctx.aSplitChunkCnt = CeilDiv(ctx.H, ctx.aUbFactor);
    ctx.aLoopCntTotal = ctx.aSplitChunkCnt;  // outerAProd = 1（前导 A=1 不贡献）
    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = ctx.aLoopCntTotal % ctx.coreNum;  // 前 aBigCoreCnt 核多处理 1 块
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + ((ctx.aBigCoreCnt > 0) ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? ctx.coreNum : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, int64_t(1));  // blockIdx ≥ usedCoreNum：idle
}
```

大小核均衡机制（数值例，48 核）：
- `aLoopCntTotal=32`：`small=32/48=0`、`bigCnt=32%48=32`、`bigLoop=0+1=1`、`usedCoreNum=(0>0)?48:32=32` → 32 核各 1 块，16 核 idle（`blockIdx ≥ 32`），blockDim=32；
- `aLoopCntTotal=48`：`small=1`、`bigCnt=0`、`bigLoop=1`、`usedCoreNum=(1>0)?48:0=48` → 48 核各 1 块，blockDim=48；
- `aLoopCntTotal=52`（7 核场景）：`small=52/7=7`、`bigCnt=52%7=3`、`bigLoop=8`、`usedCoreNum=7` → 前 3 核 8 块、后 4 核 7 块（负载差 ≤ 1）。

## 设计决策

- **平台信息走 `context->GetPlatformInfo()` / `Ops::Base::Get*Size()`，不依赖 `GetCompileInfo<T>()`**：设计文档与既有桩注释注明 aclnn 通路不调用 TilingParse、GetCompileInfo 恒返回 null（本会话未直接观测该 null——属设计输入；实现全程走 GetPlatformInfo 路径，经 UT mock 与真机 aclnn 双路验证通过）。
- **保留 def.cpp 的 `AICore().SetTiling(optiling::TilingFunc)` 注册，不新增 IMPL_OP_OPTILING**：红基线阶段已实证该绑定可被 UT 从安装包 .so 解析（"tiling kernel resolved"）；参考工程 scale_custom/add_custom 用 `IMPL_OP_OPTILING(...).Tiling(...).TilingParse<...>` 是另一套机制，混用会造成双注册。def.cpp 属注册链文件，本任务范围只有 tiling 链路文件——不动注册。
- **dtype 拒绝日志用 `OP_LOGE` 带具体值**：本 CANN 版本的 log.h 无 `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON` 宏（grep 实证，仅有 `OP_LOGE_WITH_INVALID_INPUT_DTYPE`），按设计"错误日志必须含具体变量值"用 OP_LOGE 手写。
- **attr 按索引访问**：`GetAttrPointer<bool>(0)` 而非按名；`GetAttrs()` 为空或指针为空时取 OpDef 默认 false——attr 缺省不构成失败。
- **INFO 全量打印**：每个输入 shape/dtype/attr 与 TilingData 每字段都 OP_LOGI（设计 §2/§7.4），涉及 workspace 必打、不用也显式设置。
- **当前方案属哪档路线**：`recommended_fastpath`。

## 系统信息

- CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`）；目标 ascend950/arch35
- 算子包：`operators/ThnnFusedGruCellGrad_package/`；相关文件（本任务实际改动，与事件流一一对应）：
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（完整实现，504 行 diff）
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.h`（声明 + 注释更新）
  - `op_kernel/arch35/thnn_fused_gru_cell_grad_struct.h`（TPL 声明）
  - 只读参照：`op_host/thnn_fused_gru_cell_grad_def.cpp`（注册绑定）、`op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`（TilingData 双 struct，不增不删）、`tests/tiling/`（33 条 UT，零改动）
- 注册机制：`thnn_fused_gru_cell_grad_def.cpp` 的 `this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend950")`；UT 经 dlopen 安装 run 包 `libcust_opmaster_rt2.0.so` + op registry 解析该函数
- 验证链路：`tests/tiling/run.sh`（构建 run 包→安装→编 gtest→执行）；`operators/build.sh <pkg>`；`tests/aclnn/run.sh`

## 完整代码

见「解决配方」（TilingFunc + GetShapeAttrsInfo + ComputeFusedALoopSplit 全量 verbatim；空张量/UB 切分/group 三个子模块见同批分卡（[空 batch 归约](empty_branch_kernel_empty_a_early_exit_empty_r_zero_fill.md)、[UB 预算反解](ub_budget_backsolve_dual_axis_split_clamp_ring.md)、[group 2D 分核](group_2d_core_split_numblocks_align_schedule_mode.md)））。单文件编译自检命令（复刻 libascend_all_ops.so 编译）：

```bash
P=<package根>; CANN=/home/<user>/Ascend/cann-9.0.0-beta.2; g++ -g -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI=0 -O0 -I $P -I $P/op_kernel -I $CANN/include \
  -I $CANN/x86_64-linux/include -I $CANN/x86_64-linux/pkg_inc -I $CANN/x86_64-linux/pkg_inc/base \
  -D BUILD_SOC_VERSION=Ascend950 -c $P/op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp -o /tmp/tiling_test.o
```

## 性能影响

无性能故事（骨架正确性知识；分支顺序与校验开销可忽略，未测两路线对比）。

## 预防

- tiling 动工前先写"校验矩阵"：每个负向用例 → 哪条校验拒绝 → 拒绝时返回什么；rank 校验永远排在 dim 索引之前。
- 每个分支（base/group/empty）收尾三件套 checklist：SetTilingKey / SetBlockDim(max(used,1)) / ws[0]=sys+usr（不用 workspace 也显式设 sys）。
- 注册机制沿用工程既有绑定（读 def.cpp 确认），不因参考工程用 IMPL_OP_OPTILING 就跟着换。
- 实现前用独立探针钉死 TilingContext API 面，实现后先单文件编译再跑全链路。

## 验证信号

本卡信号来自 ThnnFusedGruCellGrad 公共 TilingFunc 实现会话（红基线对照 → 实现 → 全链路验证门）：

- 红基线（实现前，同一 UT 链路）：33 用例 0 passed / 33 failed，tiling kernel 从安装包 .so 解析成功（"registered TilingFunc not loaded" 未出现，失败为真实断言失败）（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 实现后 TilingUT：`[  PASSED  ] 33 tests.`，重跑退出码 0 结果一致（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 拒绝用例 6 条全过：RejectUnsupportedDtypeFp64 / RejectDtypeCombinationMismatch / RejectRankOne / RejectRankThree / RejectStorageWidthNotFiveH / RejectBatchMismatch（来源平台：950，TilingUT mock；证据等级：tool_observed）
- 验证门：无 "do not registe tiling struct" 告警；UT 可执行产物存在；oracle 独立性 grep clean（`Oracle*` 函数体无 `optiling::` 调用）；git 变更仅 3 个 tiling 链路文件（来源平台：950；证据等级：tool_observed）
- 全量构建 `build.sh` 退出码 0；真机 aclnn 最小通路 1/1 OK（fp32 冒烟 PASSED，日志无 tiling 告警）（来源平台：950，真机；证据等级：tool_observed）

## 失效边界

- 校验集合（dtype 白名单 {fp32,fp16,bf16}、rank==2、storage==(B,5H)）是源算子契约——迁移时按各自 spec 替换，骨架顺序（dtype→rank→shape、先校验后切分）可迁移。
- `AICore().SetTiling` 注册绑定的 UT 可解析性在本工程（CANN 9.0.0-beta.2 + 该 def.cpp 模板）实证；其他注册模板/版本未验证。
- 本卡只覆盖 host tiling；kernel 侧三分支（`_0/_1/_2` 子核）在 kernel 模板化前不存在，真机只有 base 路径可用。
- GetCompileInfo 恒 null 的说法来自设计文档/桩注释（本会话未直接观测），不要据此在别的工程直接删除 CompileInfo 逻辑——先各自验证。

## 坏实践

- **`OP_CHECK_NULL_WITH_CONTEXT(context, context)`（用 context 检查 context 自身）**：宏内部会对被检指针解引用取日志上下文——context 为 null 时先崩溃，防护完全不生效。本会话实现初稿犯过此写法，自审时移除（终版 TilingFunc 入口无 context 自检，nullptr 由调用方契约保证）。初稿片段（负例，勿抄）：
  ```cpp
  ge::graphStatus TilingFunc(gert::TilingContext* context)
  {
      OP_CHECK_NULL_WITH_CONTEXT(context, context);   // ✗ 若 context==nullptr，宏内先解引用崩溃
      ...
  ```
- 先读 `GetDim(1)` 再查 rank——rank-1 输入越界读，拒绝逻辑形同虚设。
- 校验失败仍继续填 TilingData（软失败）——垃圾切分比拒绝更难排查。
- `SetBlockDim(usedCoreNum)` 不做 `max(…, 1)`——退化 shape 下 0 是非法调度参数。

## 误诊

- "UT 里 tiling 函数解析不到是因为没写 IMPL_OP_OPTILING"——本工程走 def.cpp 的 `AICore().SetTiling` 绑定，红基线实证可解析；解析不到的真实原因更可能是 .so 加载失败（如缺 libascend_hal.so 依赖）而非注册方式。
- "平台参数可以缓存/写死（48 核/256KB）"——UT mock 与真机值不同（真机 950PR 另有 coreNum/ubSize），写死会静默错切分；一律运行时 `GetPlatformInfo()` 查询。


