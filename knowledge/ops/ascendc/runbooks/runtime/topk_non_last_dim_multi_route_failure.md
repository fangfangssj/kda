---
type: "Runbook"
title: "TopK非末维多路线修复失败回退"
description: "arch35 上 TopK 沿非末维（dim=0/1）排序时 GM 源非连续：搬运层三方案（逐元素 DataCopyPad 507035 / blockCount+srcStride 与 GM 标量 get-set 致 idx 越界）与转置层四路线（纯 DataCopyPad、32B+S pipe、Gather 偏移表、GM workspace transpose）全部失败；四个机制未解，整体回退并恢复 ST 跳过。 复现增量：迭代三确认——失败覆盖全部 7 种 dtype 的非末维组合（与 dtype/largest/ndim 无关的全覆盖性失败），Cast 类型非末维 CopyIn baseOffset 修正 + 32B 对齐分块搬入无效；最终 C++ ST 跳过 19 例 110/110、PyTorch ST 跳过 159 例 577/577，issue 跟踪三个候选机制（SetValue/GetValue 非连续访问、CopyOut 黑名单 API、transpose 禁用）。"
tags: [task.debugging, topic.precision, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# TopK非末维多路线修复失败回退

## 现象

TopK 归约类自定义算子（Ascend950PR / arch35，CANN 9.1.0）支持任意 dim 的 topk。当 dim 指向非末维（dim=0 或中间维 dim=1）时，待排序元素在 GM 中按 `srcDimStride` 非连续分布，需要带 stride 或逐元素搬入 UB；末维（dim=-1）用例全部通过（111/111）。非末维用例（fp16，2D `[256,1024]` dim=0、3D `[16,256,1024]` dim=0/1 等，k=10）长期失败，失败形态随实现路线变化：

- 逐元素 `DataCopyPad`（`blockLen = sizeof(T)`）搬运：`[FAIL] 流同步失败: 507035`（kernel 崩溃）；
- 带 stride 的整块搬运 / GM 标量 get-set：不崩但 `idx gather 校验失败：idx 越界`（idx 为垃圾大数如 3715807866285798289）、y 比对 mismatch≈98%~100%（值合法但位置错）；
- 转置重排路线（纯 DataCopyPad 转置 / 32B 块 + S pipe 提取 / Gather 偏移表 / GM workspace transpose）：精度 mismatch≈100%（inner=16 的用例曾通过、inner=256 失败），或启用 GM workspace 后 5 个非末维用例全部流同步失败 507035。

最终把 kernel/tiling/op_api/测试共 10 个文件 `git checkout` 回退到 HEAD，恢复 ST 跳过逻辑收场：基线 ST 61 用例 50 通过/0 失败/11 跳过；后续联调验证 ST 111/129，18 个 dim!=last 失败被标记为 arch35 已知限制。

## 适用场景

- 工程形态：算子仓开发，kernel 非末维（dim ≠ last）数据重排/转置路径的设计与排障。
- API 路径：`DataCopyPad`（`DataCopyExtParams` 的 blockCount/blockLen/srcStride/dstStride；arch35 上 srcStride/dstStride 为 int64 字节单位）、GM `GetValue`/UB `SetValue` 标量访问、`Gather`（V pipe 偏移表）、`PipeBarrier<PIPE_ALL>`、GM workspace（tiling `needTranspose` + `wsInputBytes/wsValueBytes`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0，ST Real 模式 NPU 实测。
- 适用粒度：operator_family（arch35 跨 pipe UB 同步与搬运语义）。
- 路线优先级：banned（本卡记录的多条路线在该平台该场景下均未成功，禁止当作修复配方直接复用；候选方向见遗留问题）。
- 明确不适用/未验证：末维（dim=-1）路径不受影响；非 arch35 平台的 pipe 同步行为未验证。

## 根因（未追到底，四个未解机制）

失败被归档为四个未解机制（均为分析结论、未做探针级验证）：

1. **S pipe → MTE3 同步不完全**：S pipe 大量写 UB 后 MTE3 读 UB，inner=16 通过、inner=256 失败，`PipeBarrier<PIPE_ALL>` 不足以保证大数据量同步；
2. **DataCopyPad blockLen<32 的 pad 语义**：`blockLen=sizeof(T)` 时数据在 UB 中被 pad 到 32 字节块，不连续，后续按连续元素处理即错位；
3. **Gather 偏移单位不确定**：`srcOffset` 是字节偏移还是元素偏移无文档结论，两种取值的实测功能结果都不正确；
4. **手写 L2 API 的 workspace 传递**：`GetWorkspaceSize()` 只返回固定 512，不透传 TilingFunc 的 `currentWorkspace[0]`（本例需 626688）；正确途径 `NnopbaseRunForWorkspace`（ACLNN 内部函数）在手写 L2 下调用段错误（详见同批 `aclnn_l2_tiling_workspace_gap_507035`）。

## 已尝试路线（attempt，核心代码可重建）

**搬运层三方案**（同一 5 用例跨方案对照）：

尝试 1——逐元素 DataCopyPad（初始实现，507035）：

```cpp
            // 非末维：逐元素搬入（失败：507035 流同步失败）
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(sizeof(T));   // fp16 = 2 字节
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            int64_t baseOffset = rowIdx * srcRowStride_;
            for (int64_t t = 0; t < inner_; t++) {
                DataCopyPad<T>(srcIn[t], gmX[baseOffset + t * srcDimStride_], copyParams,
                               {false, 0, 0, static_cast<T>(0)});
            }
```

尝试 2——blockCount + srcStride 整块搬运（idx 越界 + y 98% 不匹配）：

```cpp
            // 非末维：使用 blockCount + srcStride 一次性搬入（失败：idx 越界 + y 98% 不匹配）
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint32_t>(inner_);
            copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
            copyParams.srcStride = (srcDimStride_ - 1) * sizeof(T);   // arch35: 字节单位
            copyParams.dstStride = 0;
            DataCopyPad<T>(srcIn, gmX[rowIdx * srcRowStride_], copyParams, ...);
```

尝试 3——GM GetValue + UB SetValue 逐元素标量搬运（同样 idx 越界）：

```cpp
            // 非末维：使用 GM GetValue + UB SetValue 逐元素搬入（失败：同样 idx 越界）
            int64_t baseOffset = rowIdx * srcRowStride_;
            for (int64_t t = 0; t < inner_; t++) {
                srcIn.SetValue(t, gmX.GetValue(baseOffset + t * srcDimStride_));
            }
```

**转置层四路线**（`TransposeInput`/`TransposeOutput` 重排）：

路线 2 的 `TransposeInput`（DataCopyPad 32B 块搬入 + S pipe 提取 + MTE3 写 GM workspace；inner=16 通过 / inner=256 失败的那一版）：

```cpp
// 输入 transpose：DataCopyPad(MTE2) blockLen=32 读非连续 GM→UB，
// GetValue/SetValue(S pipe) 提取到连续 UB，DataCopyPad(MTE3) 写到 workspace GM。
// 注意：arch35 S pipe 大量写 UB 后 MTE3 读 UB 可能有同步问题（inner>32 时失败）。
template <typename T, int INNER_MODE>
__aicore__ inline void BenchNoHarnessTopK<T, INNER_MODE>::TransposeInput(int64_t rowIdx)
{
    int64_t outerIdx = rowIdx / origSrcDimStride_;
    int64_t afterIdx = rowIdx % origSrcDimStride_;
    int64_t baseOffset = outerIdx * (inner_ * origSrcDimStride_) + afterIdx;
    int64_t wsOffset = rowIdx * inner_;

    // 1. DataCopyPad blockCount=inner, blockLen=32 从非连续 GM 读到 UB（MTE2）
    constexpr int32_t elemsPerBlock = 32 / sizeof(T);
    LocalTensor<T> nonLastDimData = nonLastDimBuf.Get<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(inner_);
    copyParams.blockLen = 32;
    copyParams.srcStride = (origSrcDimStride_ - elemsPerBlock) * static_cast<int64_t>(sizeof(T));
    copyParams.dstStride = 0;
    DataCopyPad<T>(nonLastDimData, gmX[baseOffset], copyParams, {false, 0, 0, static_cast<T>(0)});
    PipeBarrier<PIPE_ALL>();

    // 2. GetValue/SetValue 提取每个 block 的第 1 个元素到连续 UB（S pipe 操作 UB）
    LocalTensor<T> wsContig = srcValueSortBuf.Get<T>();
    for (int64_t t = 0; t < inner_; ++t) {
        wsContig.SetValue(static_cast<uint32_t>(t),
                          nonLastDimData.GetValue(static_cast<uint32_t>(t * elemsPerBlock)));
    }
    PipeBarrier<PIPE_ALL>();

    // 3. DataCopyPad 从连续 UB 写到 workspace GM（MTE3）
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad<T>(gmWsInput[wsOffset], wsContig, outParams);
    PipeBarrier<PIPE_ALL>();
}
```

路线 3 的 Gather 提取段（元素偏移变体；字节偏移变体把 `Arange` 的 step 换成 32）：

```cpp
    // 2. Gather(V pipe) 从 32B-padded UB 提取到连续 UB
    // 偏移表：元素偏移 [0, elemsPerBlock, 2*elemsPerBlock, ...]
    constexpr int32_t elemsPerBlock = 32 / sizeof(T);
    LocalTensor<int32_t> gatherOffsetI32 = gatherOffsetBuf.template Get<int32_t>();
    Arange<int32_t>(gatherOffsetI32, static_cast<int32_t>(0), elemsPerBlock,
                    static_cast<int32_t>(inner_));
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> gatherOffsetU32 = gatherOffsetI32.template ReinterpretCast<uint32_t>();
    LocalTensor<T> wsContig = srcValueSortBuf.Get<T>();
    Gather<T>(wsContig, nonLastDimData, gatherOffsetU32, 0, static_cast<uint32_t>(inner_));
    PipeBarrier<PIPE_V>();
```

路线 4 的 tiling 开关（启用 GM workspace transpose，5 用例全部 507035）：

```cpp
    // GM workspace transpose：dim != last 时，用 workspace 做 GM 层 transpose
    bool needTranspose = !isLastDim;
    int64_t wsInputBytes = 0;
    int64_t wsValueBytes = 0;
    size_t workspaceSize = 0;
    if (needTranspose) {
        wsInputBytes = CeilAlign(outter * inner * static_cast<int64_t>(dtypeSize),
                                 static_cast<int64_t>(32));
        wsValueBytes = CeilAlign(outter * k * static_cast<int64_t>(dtypeSize),
                                 static_cast<int64_t>(32));
        int64_t wsIdxBytes = CeilAlign(outter * k * static_cast<int64_t>(sizeof(int64_t)),
                                       static_cast<int64_t>(32));
        workspaceSize = static_cast<size_t>(wsInputBytes + wsValueBytes + wsIdxBytes);
    }
    tiling->needTranspose = needTranspose;
    tiling->wsInputBytes = wsInputBytes;
    tiling->wsValueBytes = wsValueBytes;
```

回退决策的最终落盘（tiling 侧留档注释）：

```cpp
    // GM workspace transpose：dim != last 时，用 workspace 做 GM 层 transpose
    // 该方案因 arch35 S pipe→MTE3 同步问题未完全解决
    // L2 API workspaceSize 传递机制需要 NnopbaseRunForWorkspace（手写 L2 API 不支持）
    // 暂时禁用，非末维用例走旧路径（inner≤16 通过，inner=256 S→V 同步问题）
    bool needTranspose = false;
```

## 设计决策

- 路线演进逻辑：每次失败后剔除一个嫌疑 pipe——纯 DataCopyPad（无 S pipe）→ 仍错，怀疑 pad 布局；换 SetValue 提取（小数据量过、大数据量败）→ 嫌疑集中到 S pipe 大数据量同步；换 Gather（纯 V pipe）→ 仍全败，嫌疑回到偏移语义；启用 GM transpose 整体绕开 UB 重排 → 507035 崩溃，卡在 L2 workspace 传递。
- 尝试 2 的依据：arch35 `DataCopyExtParams` 的 `srcStride/dstStride` 为 `int64_t`（区别于其他 arch 的 `uint32_t`），且参考实现（reduce_sch.h）用 `srcStride=(stride-count)*sizeof(T)` 字节单位——方向合理但结果错误，说明问题不在单位换算而在更底层的搬运行为。
- 放弃条件：多路线覆盖了 MTE2/MTE3/S/V 四种可用 pipe 的组合后仍无一条通过，且每条的失败机制都无法低成本验证，选择整体回退保基线并把机制问题归档 issue，而不是继续在不确定的假设上叠改。
- "ST 跳过这 5 个用例"是范围管理动作（显式标 SKIP 并记录 issue，属可接受的临时回避），不构成修复；绝不能把 SKIP 计入通过率或宣称支持非末维。

## 系统信息

- CANN：9.1.0；目标平台 ascend950（Ascend950PR，arch35 / DAV_3510）。
- 相关文件：`op_kernel/arch35/bench_no_harness_top_k.h`（`TransposeInput`/`TransposeOutput`/`CopyIn`/`CopyOut`/`GatherFromPadded`）、`op_host/arch35/bench_no_harness_top_k_tiling.cpp`（`needTranspose`/workspace 计算）、`op_api/aclnn_bench_no_harness_top_k.cpp`（L2 API）、`tests/st/test_aclnn_bench_no_harness_top_k.cpp`（dim!=last 用例与跳过逻辑）、算子仓 issues 目录（多轮尝试归档）。
- 复现命令：`bash build.sh --soc=ascend950 -j8 && bash build/custom_opp_ubuntu_x86_64.run --quiet && rm -rf tests/st/build-real && bash tests/st/run.sh`（过滤非末维用例）。

## 验证信号

来源说明：以下信号均为该算子多轮开发会话中的 NPU ST 实测（同一 5 用例跨方案对照 + 全量回归）：

- 尝试 1：`[FAIL] 流同步失败: 507035`（fp16 dim=0/1 的 5 个非末维用例）（来源平台：950；证据等级：tool_observed）。
- 尝试 2：`[FAIL] idx gather 校验失败：idx 越界`、y mismatch=10044/10240（来源平台：950；证据等级：tool_observed）。
- 尝试 3：同样 idx 越界（垃圾 idx 如 3715807866285798289）（来源平台：950；证据等级：tool_observed）。
- 路线 1/2 精度失败：fp16 `[256,1024] k=10 dim=0` mismatch=10096/10240（MERE=5.73e-01）、3D `[16,256,1024] dim=1` mismatch=158343/163840（来源平台：950；证据等级：tool_observed）。
- 路线 2 的规模分界：inner=16 用例通过、inner=256 用例失败（来源平台：950；证据等级：tool_observed）。
- 路线 3 全败：14 个 dim!=last 用例 FAIL（总计 129、通过 70、失败 14）（来源平台：950；证据等级：tool_observed）。
- 路线 4 崩溃：启用 `needTranspose` 后 5 个非末维用例全部 `[FAIL] 流同步失败: 507035`（引入 workspace 前为静默 mismatch，形态升级）（来源平台：950；证据等级：tool_observed）。
- 对照组：同 dtype 末维用例全部通过；回退后基线 ST 61 用例 50 通过/0 失败/11 跳过；后续联调验证 ST 111/129，末维 111/111（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限于 arch35 + CANN 9.1.0 的非末维重排场景；`PipeBarrier<PIPE_ALL>` 在 inner=16 规模下足够、inner=256 不够的量化边界未做系统性测定（16 与 256 之间未扫描）。
- 四个机制结论（S pipe 同步、32B pad、Gather 偏移单位、L2 workspace 传递）均未追到硬件/实现层验证，属强假设；推翻其中任何一个都可能复活对应路线。
- 三条搬运路线的失败样本均为 fp16、`[256,1024]`/`[16,256,1024]` 形状、k=10；其他 dtype/形状未逐一测试（int8 等 1 字节类型预期更差，未验证）。
- 未验证，推断可迁移：同代架构可能存在类似跨 pipe UB 同步问题，迁移前需实测。

**未决项**：

- **优先项：复现并追根因到机制层**——① arch35 S pipe 写 UB 后 MTE3/V pipe 读 UB 的同步语义（建议最小 inner 扫描 16→256 找失败边界）；② `Gather` `srcOffset` 的字节/元素偏移语义（读 `dav_3510/kernel_operator_vec_gather_impl.h` 的 `GatherImpl` 实现）；③ `DataCopyPad` blockLen<32 时 UB 内布局的探针验证；④ 手写 L2 的 workspace 传导（见同批 `aclnn_l2_tiling_workspace_gap_507035`）。
- 候选修复方向（issue 归档建议，均未尝试）：TQue EnQue/DeQue 替代 TBuf 做精确同步；`TransDataTo5HD` 或 DataCopyPad ND2NZ 做非连续→连续转换（纯 MTE2，无 S pipe）；调研 `NnopbaseRunForWorkspace` 正确调用方式或改用 autogen L2 API 解决 workspace 传递。
- 恢复被跳过的非末维用例前，本卡问题未解决。
## 坏实践

- 在机制未验证的情况下连续叠改同一路径（路线 2→3 之间多次改 `TransposeInput` 细节），每轮都要全量重编译+上板，时间成本高且难归因。
- 用 `git checkout -- <多文件>` 在多代理共享 worktree 里做"整体回退"——一次性摧毁了并行开发会话未提交的 int32/int64 kernel 实现与 ST 测试扩展，事后只能靠后续联调验证重新补齐（工作流层负知识，与 kernel 无关但代价真实发生）。

## 误诊

- `needTranspose` 曾被早前修改静默置为 `false`（tiling 注释"禁用 GM workspace transpose"），导致一段时间内调试的是一条根本没启用的路径；调 dim!=last 前先确认 tiling 开关实际下发的值。
- "某用例之前通过、现在也失败"的回归假象：同一份代码在不同 TilingKey/kernel 二进制组合下行为不同，先比对二进制/tiling 再怀疑代码。
- 507035 的第一反应是"workspace 大小不够"，但真正卡点可能在 L2 不透传 TilingFunc workspace（512 vs 626688）——两层的修复位置完全不同（分析结论，未实测分层验证）。

## 后续增量（2026-09-17，并行策展批：迭代三全覆盖确认 + 跳过处置定版 + issue 候选机制）

同算子迭代三（int32/int64 扩展联调，Ascend950PR / arch35 / CANN 9.1.0）补齐本卡的失败覆盖面与最终处置：

**1. 全覆盖性失败确认**：int 数据域与 L0 API 白名单问题解决后，剩余失败全部集中在非末维（dim=0/1/2）用例，覆盖所有 7 种 dtype 的非末维组合（fp16 dim=0、3D dim=1、8D 非末维等），失配率 50%-100%（如 `y 比对: 2522/2560 个元素不匹配`、`MERE=5.68e-01, MARE=1.00e+00, mismatch=10071/10240`、int 域 `MERE=2.78e+06, MARE=1.00e+07, mismatch=10230/10240`）——与 dtype、largest、ndim 无关；而末维（dim=-1/last）的全部 7 dtype × 全 ndim × 全 k 边界 × 全特殊值用例全部通过。

**2. 最后一次尝试（Cast 类型非末维搬入修正，verbatim）**：对 Cast 类型非末维 CopyIn 的 baseOffset 修正 + 32 字节对齐分块搬入，未能改变整体失败形态：

```cpp
        } else {
            // 非末维：正确计算 baseOffset（与非 Cast 路径一致）
            int64_t outerIdx = rowIdx / srcDimStride_;
            int64_t afterIdx = rowIdx % srcDimStride_;
            int64_t baseOffset = outerIdx * (inner_ * srcDimStride_) + afterIdx;
            // DataCopyPad blockLen=32 对齐搬入 + GetValue 提取
            LocalTensor<T> nonLastDimData = nonLastDimBuf.Get<T>();
            constexpr int32_t elemsPerBlock = 32 / sizeof(T);
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(inner_);
            copyParams.blockLen = 32;  // 32 字节对齐
            copyParams.srcStride = (srcDimStride_ - elemsPerBlock) * static_cast<int64_t>(sizeof(T));
            copyParams.dstStride = 0;
            DataCopyPad<T>(nonLastDimData, gmX[baseOffset], copyParams,
                           {false, 0, 0, static_cast<T>(0)});
            PipeBarrier<PIPE_ALL>();
            // 提取每个 block 的第 1 个元素（目标元素）
            for (int64_t t = 0; t < inner_; t++) {
                srcValueOrig.SetValue(static_cast<uint32_t>(t),
                                      nonLastDimData.GetValue(static_cast<uint32_t>(t * elemsPerBlock)));
            }
            PipeBarrier<PIPE_ALL>();
        }
```

**3. 最终处置**：C++ ST 跳过 19 个 dim!=last 用例（110/110 通过）、PyTorch ST 跳过 159 个（577/577 通过），创建 issue 文件跟踪。issue 列出三个候选机制（均未实验隔离验证，引用时须标注为假设）：①非 Cast 路径（fp16/fp32）的 CopyIn 用 `GlobalTensor::SetValue()/GetValue()` 逐元素搬出/搬入，该 API 在 arch35 上对非连续 GM 访问有一致性问题；②CopyOut 所有 dtype 的非末维输出用 `gmY.SetValue()` 逐元素写回，属 arch35 黑名单 API；③tiling 中 `needTranspose = false`（迭代二禁用 GM workspace transpose），非末维路径依赖 UB 层 strided access，arch35 上 strided GM→UB 访问受限。issue 给出的候选修复方向（按优先级，均未验证）：启用 GM workspace transpose 后走已验证的末维路径；TransDataTo5HD 维度转换；全面替换 SetValue/GetValue 为 DataCopyPad + LocalTensor。

**4. 归因纪律**：联调中期曾判定"int32/int64 在 dim=last 时通过、剩余失败全为 dim!=last"，随后核对日志发现 int32/int64 的 dim=last 也失败过（数据域问题，另卡处置）——两类失败曾并存，归因时须逐类核对而不是取最近一轮的印象。

### 复现增量验证信号

- C++ ST Real：dim!=last 用例 y 比对 `2522/2560`、`MERE=5.68e-01, MARE=1.00e+00, mismatch=10071/10240` 等大面积失配，覆盖 7 dtype（来源平台：950；证据等级：tool_observed）
- 排除性观测：失败用例 grep 排除 dim=0/dim=1 后为空（全部失败均为非末维）；末维用例（含 int32/int64、大 inner、特殊值）全通过（来源平台：950；证据等级：tool_observed）
- 处置后：C++ ST 129 用例跳过 19、110/110 通过（`测试结果: PASS`）；PyTorch ST 736 用例跳过 159（dim 分布 45×dim=0、43×dim=1、42×dim=2）、577/577 通过（来源平台：950；证据等级：tool_observed）
- issue 跟踪文件写入（LOW 级、待跟进）（来源平台：950；证据等级：tool_observed，issue 内容本身为 agent 分析）

### 复现增量边界

- 三个候选机制（SetValue/GetValue 非连续访问、黑名单 API、transpose 禁用）均未实验隔离验证，不能作为已证根因引用。
- 跳过处置只覆盖 ST 评测面，不改变算子在非末维输入下产出错误结果的事实——宣称支持非末维前必须先落地并验证修复方向。
- 跳过用例必须挂 issue 跟踪，不得当作长期方案。