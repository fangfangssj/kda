---
type: "Runbook"
title: "kernel直绑aclnn workspace入参即507035"
description: 多条独立 trace 实证：C++ aclnn ST 直调场景下 host 已按非零 workspaceSize 分配并传入，kernel 仍把中间缓冲直绑 workspace GM_ADDR 入参（SetGlobalBuffer/SetValue/DataCopyPad 访问）即触发 507035（errcode 95 MTE DDR 越界 / errcode 264 / 无 plog 标记的向量核异常）；机制各 trace 均未定位，路线 banned；已验证规避=输出别名 y、两遍重算去 workspace、移除 workspace 依赖、GetSysWorkSpacePtr 小规模应急。
tags: [task.debugging, topic.memory, topic.data_movement.data_copy, topic.attention]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# kernel直绑aclnn workspace入参即507035

## 现象

三条独立 trace、四个不同算子（GQA 注意力、softmax Group 分核、unique 去重、GQA 单 Pass 重写），同一经验事实：**kernel 把任何中间缓冲直绑到 aclnn 入参 `workspace`（GM_ADDR）并做 GM 读写，即触发 `aclrtSynchronizeStream` 返回 507035（向量核异常）**——尽管 host 侧 `GetWorkspaceSize` 返回的 workspaceSize 非零、测试侧已 `aclrtMalloc` 分配并传入。各 trace 的失败形态：

- **GQA 向量版（CANN 9.1.0）**：kernel 把 softmax 中间 scores 矩阵存放在 tiling 经 `context->GetWorkspaceSizes(1)` 动态声明的 GM workspace（每核独立区域）。最小用例（B=4, S=1, S_kv=128, fp16）即 `[FAIL] 流同步失败: 507035`；plog 显示**所有核在同一 PC** 上失败：`errcode:(95) The DDR address of the MTE instruction is out of range. subErrType: 0x4`、`mte error: 0x80000000`（bit 31 无效地址）。二分裁剪确认：只要不碰 workspace（只做 Q→Y 拷贝）就不报；一旦加一条 `DataCopy(scoresGm[0], scoresRow, 8)`（8 个 float = 32 字节对齐）写 workspace 就必现。当时 CPU golden 7/7、Mock ST 6/6、UT 12/12 全过，仅真机失败；workspaceSize=29184 非零、ST 已分配。
- **softmax GroupTailA 分支（CANN 9.2.0）**：kernel `Init()` 把跨核部分和缓冲 `gmWs_` 绑到入参 `workspace`（测试侧已按非零大小分配传入）。随机数据用例与 NaN 输入用例全部 `ACL error 507035 at ... aclrtSynchronizeStream`；同期常量特殊值用例（全 +inf/全 -inf/全 0 等）与边界用例 PASS。静态排查 UB 越界、DMA 32B 对齐、workspace 偏移与大小均无问题；plog 对该异常无错误标记（搜 `507035|aic error|vector core|exception` 无命中）。`git stash` 回基线重建同批全过、stash pop 恢复改动复现崩溃——崩溃由"绑定 workspace"这一改动唯一引入。
- **unique 去重 K 回读（CANN 9.1.0）**：kernel `kGM_.SetGlobalBuffer((__gm__ uint32_t*)workspace)` + `kGM_.SetValue(0, K)` 写 4 字节 → 全量向量核异常 507035；去掉该写回并置 wsSize=0 后 `grep -c "507035"` = 0。后续在更大 workspace（数十/上百字节）上做标量访问仍报 `errcode(264,95)`（The address for scalar to access GM is invalid / DDR address of the MTE instruction is out of range）。
- **GQA 单 Pass + GM workspace 缓存 scores（CANN 9.1.0）**：`wsGm.SetGlobalBuffer((__gm__ float*)workspace)` 直接以入口 workspace 为基址、tiling 只声明用户字节数；重写后 MERE 无改善且数据量最大的 MHA 形态触发 507035，路线回退。

## 适用场景

- 工程形态：aclnn 自定义算子开发 + ST 真机直调（aclnn 两段式；workspace 由 tiling 声明或 host 分配后传入）；分组/多阶段 kernel 需要在 kernel 内访问 GM workspace 的场景。
- API 路径：kernel 入参 `GM_ADDR workspace`、`GlobalTensor::SetGlobalBuffer((__gm__ T*)workspace)`、`GlobalTensor::SetValue/GetValue`、`DataCopyPad`（UB↔GM）；`GetSysWorkSpacePtr()`；aclnn `GetWorkspaceSize`（host 分配）→ `aclnnXxx(..., workspace, workspaceSize, stream)`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0 与 9.2.0（与 frontmatter 一致）。
- 适用粒度：operator_family——所有需要在 kernel 内访问 GM workspace 中间缓冲的向量类算子（本批跨 4 算子复现）。
- 路线优先级：banned（kernel 内直用入参 workspace 指针这条路已多 trace 实证崩溃；需 workspace 语义时应改用算法层绕开，或先验证传递链/布局约定）。
- 明确不适用 / 未验证场景：同一批 trace 中也有"非 group 分支绑 workspace 且 ST 通过"的历史记录（见失效边界）——本卡结论是"该参数不可信任、上板对照验证前禁止依赖"，不是"所有路径一律失效"；其他 CANN 版本与调用框架（ge/acl 单算子）未验证。

## 根因（未追到机制层，附候选解释）

各 trace 均止步于"绑定改动 ↔ 崩溃"的因果对照，框架侧机制未逐层验证。已排除的方向：TilingData 大小/参数个数（核对一致）、`KERNEL_TASK_TYPE_DEFAULT` 宏（移除/加回无效，参考算子带宏正常用 workspace）、ReduceSum dst 对齐（改固定位置中转无效）、workspaceSize==0 空指针（实测 29184 非零已分配）、偏移/blockLen 写法（最小 8 float 对齐写也崩）。

**首选候选机制（本批未验证，引自库内卡 `mte_ddr_507035_workspace_reserved_region` 的 9.2.0 实测）**：该平台 workspace 缓冲布局为 [系统保留区（Ascend950 为 16MB / `GetLibApiWorkSpaceSize()`）][用户区]，kernel 收到的 `workspace` 指针已偏移到用户区位置（实测与 host 传入基址相差 0x1000000）；tiling 只声明用户字节数时，kernel 拿到的偏移后指针落在调用方分配范围之外 → MTE 写即 DDR 越界（errcode 95）。该机制可统一解释本卡各 trace 的现象（声明/分配仅按用户量 → 偏移后指针越界），但本批各 trace 未做 kernel 内指针与分配区间的核对，引用时按候选解释处理。正确用法（声明含保留区 + `GetUserWorkspace`/按偏移取址）见库内该卡。

## 解决配方（规避路线，均已上板验证）

**规避 1：两遍算法彻底去掉 GM workspace（GQA，终版 27/27）**——Pass 1 逐行算 Q@K^T 得 rowMax/rowSum 只存 UB；Pass 2 重算 Q@K^T、softmax、attn@V；tiling 侧 `ws[0] = 0`。修复前后 Init 的 workspace 取址（verbatim）：

```cpp
    // 修复前（Init 内 workspace 取址）
    // Per-core workspace: scores [S, S_kv] float
    int64_t wsBase = blockIdx_ * tiling_.wsPerCore;
    auto* wsPtr = (__gm__ uint8_t*)workspace;
    scoresGm.SetGlobalBuffer((__gm__ float*)(wsPtr + wsBase));

    // 修复后（终版 Init，不再使用 workspace；scores 全在 UB）
    queryGm.SetGlobalBuffer((__gm__ D_T_Q*)query);
    keyGm.SetGlobalBuffer((__gm__ D_T_Q*)key);
    valueGm.SetGlobalBuffer((__gm__ D_T_Q*)value);
    yGm.SetGlobalBuffer((__gm__ D_T_Q*)y);

    // 两遍算法不需要 GM workspace（scores 全在 UB）
    // workspace 参数在 arch35 上导致 507035，GetSysWorkSpacePtr 太小
```

终版 Init 的 UB 缓冲布局（scores/attn/rowMax/rowSum 全在 UB，最小 64 字节防 S=1 过小）：

```cpp
    // 最小缓冲区大小 64 bytes（16 float），防止 S=1 时分配过小
    int64_t rowBufSize = S * sizeof(float);
    if (rowBufSize < 64) { rowBufSize = 64; }
    int64_t bcastSize = S_kv * sizeof(float);
    if (bcastSize < 64) { bcastSize = 64; }
    // reduceTmp 需要足够大：max(D, S_kv) * sizeof(float) * 2（ReduceSum 内部需要）
    int64_t reduceSize = ((D > S_kv ? D : S_kv) * 2) * sizeof(float);
    if (reduceSize < 256) { reduceSize = 256; }

    pipe.InitBuffer(copyHalfTbuf_, D * sizeof(D_T_Q));
    pipe.InitBuffer(qRowTbuf_, D * sizeof(float));
    pipe.InitBuffer(kRowTbuf_, D * sizeof(float));
    pipe.InitBuffer(mulTmpTbuf_, D * sizeof(float));
    pipe.InitBuffer(reduceTmpTbuf_, reduceSize);
    pipe.InitBuffer(scoresRowTbuf_, S_kv * sizeof(float));
    pipe.InitBuffer(rowMaxTbuf_, rowBufSize);
    pipe.InitBuffer(rowSumTbuf_, rowBufSize);
    pipe.InitBuffer(bcastTbuf_, bcastSize);
    pipe.InitBuffer(yTbuf_, D * sizeof(float));
    pipe.InitBuffer(yHalfTbuf_, D * sizeof(D_T_Q));
```

Pass 1 对每行 `ComputeScoresRow`（Q·K 行内积）→ `ReduceMax` → `exp(x-mv)` → `ReduceSum` 把 rowMax/rowSum 存 UB 标量；Pass 2 重算 scoresRow → 归一化 → 逐 j 加权累加 V 行得 y。行内归约与逐 j 累加之间用 `PipeBarrier<PIPE_V>()` 串行（UB 手动搬运跨流水线的同步是独立根因，另行处理）。

**规避 2：改绑输出 GM（y 别名，softmax Group 分支，40/40）**——group 模式把 `gmWs_` 绑到输出 `y`：Phase 1 把 m/s 写到输出区开头，Phase 2 在 Pass 3 的 CopyOut 覆盖输出之前读回，`SyncAll` 保证跨核写读顺序。before/after（`Init()`，verbatim）：

```cpp
        // before（绑定入参 workspace——触发 507035 的写法）
        if (td_->rGroupCnt > 0) {
            if constexpr (isTailR) {
                gmWs_.SetGlobalBuffer((__gm__ float*)y);
            } else {
                gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
            }
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }

        // after（终版：group 模式统一 y 别名）
        // Group mode workspace: alias output GM (y) for m/s storage
        // (the framework does not reliably pass the separate workspace
        //  buffer to the kernel's workspace GM_ADDR).  Phase 1 writes m/s
        // to the start of y; Phase 2 reads before CopyOut overwrites.
        if (td_->rGroupCnt > 0) {
            gmWs_.SetGlobalBuffer((__gm__ float*)y);
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }
```

（该算子的 GroupTailR 分支从最初就用 y 别名存 m/s，从未踩坑；同 trace 另有"L2 GetWorkspaceSize 返回 0 → nullptr"的分配侧缺口，见同批 `aclnn_l2_tiling_workspace_gap_507035`——本卡只覆盖"已分配非零仍崩"的指针侧。）

**规避 3：移除 workspace 依赖（unique K 回读，67/74、0 崩溃）**——删除 `kGM_` 成员与 `SetGlobalBuffer`、删除 `kGM_.SetValue` 写回、tiling `wsSize = 0`：

```cpp
    // 回退（rollback）：tiling 侧
    // 7. Workspace 计算（迭代二暂不使用 workspace，K 回读待后续实现）
    int64_t wsKeysOffset = 0;
    int64_t wsSortedKeysOffset = 0;
    int64_t wsKOffset = 0;
    int64_t wsSize = 0;
```

**应急（小规模 only）：`GetSysWorkSpacePtr()` 取系统工作区代替 workspace 参数**——立即不再 507035；但系统工作区容量很小（实测 S=128 时每核 64KB scores 即溢出，表现为大面积精度异常 MARE=0.11~0.75），只可作小规模应急，不可作为正式方案。

## 设计决策

- 为什么最终选两遍算法/y 别名而不是修 workspace 参数：workspace 参数无效的确切机制未定位、无法修；系统工作区容量太小不可靠；两遍算法以 Q@K^T 重算一次的代价换取零 GM workspace 依赖，正确性优先（性能代价未测量，采用前自行 profiling）。
- 向量类算子优先评估"中间矩阵能否留在 UB"（按行/按块在线计算），把 GM workspace 当最后手段；必须用时先在本平台用最小写探针（写 8 个对齐 float）验证 workspace 参数可用性，并核对库内 `mte_ddr_507035_workspace_reserved_region` 的保留区声明约定。
- 崩溃后先做 stash/对照实验（只换缓冲绑定指针重跑最小失败集），再决定绕开方向；不要在崩溃态上连续叠改。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0 与 9.2.0；plog `~/ascend/log/debug/plog/`（配 `ASCEND_GLOBAL_LOG_LEVEL=1`）。
- 相关文件（各 trace）：`op_kernel/arch35/bench_no_harness_gqa.h`（两遍算法 Init/ProcessQueryHead）、`op_kernel/arch35/bench_harness_softmax_kernel.h`（Init/ProcessGroup/Phase1TailA/Phase2TailA）、`op_kernel/arch35/bench_no_harness_unique.h`（kGM_ 写回与回退）、`op_host/arch35/*_tiling.cpp`（workspace 声明）、各自 `tests/` 下 ST harness。

## 验证信号

来源说明：以下信号来自三条独立 trace 的上板实测（GQA 真机联调、softmax Group 分支实现与回退对照、unique K 回读成对回退实验）：

- GQA 修复前：最小用例流同步失败 507035；plog `errcode:(95) The DDR address of the MTE instruction is out of range`、`mte error: 0x80000000`、全核同一 PC（来源平台：950；证据等级：tool_observed）。
- GQA 二分定位：不访问 workspace 的 Q→Y 拷贝版运行成功（MERE=3.25e+01，仅因不是 GQA 计算）；恢复一条写 workspace 的 `DataCopy(scoresGm[0], scoresRow, 8)` 即再 507035（来源平台：950；证据等级：tool_observed）。
- GQA 应急规避：改 `GetSysWorkSpacePtr()` 后不再 507035（S=1 用例成功）；S=128 用例 MARE=1.13e-01~7.47e-01（每核 64KB scores 溢出系统工作区）（来源平台：950；证据等级：tool_observed）。
- GQA 回测确认：换回 workspace 参数后同批用例再次全部 507035，确认该参数在此场景不可用（来源平台：950；证据等级：tool_observed）。
- GQA 两遍算法终版：全部 27 条用例 PASS（MERE 2.18e-06~3.38e-06，含 BF16）（来源平台：950；证据等级：tool_observed）。
- softmax Group 分支：绑定 workspace 后随机数据用例（3 dtype × 5 shape）+ NaN 用例逐条 `ACL error 507035 at ... aclrtSynchronizeStream(stream)`；常量特殊值与边界用例 PASS（来源平台：950；证据等级：tool_observed）。
- softmax 隔离实验：仅改绑定为 y、其余不动，失败用例全 dtype 立即 PASS；终版全量回归四分支 46/37/34/40 全 PASS、0 aclnn-FAIL（来源平台：950；证据等级：tool_observed）。
- softmax stash 对照：绑定独立 workspace 版本 17 条 aclnn-FAIL（全 507035）；`git stash` 回基线重建后同批全 PASS、stash pop 恢复即复现——崩溃由绑定改动唯一引入（来源平台：950；证据等级：tool_observed）。
- unique K 回读：带 kGM_ 写回版本全量 60/74（内部 sync 吞错掩盖，见同批 `aclnn_internal_sync_false_pass`；去内部 sync 后 31/74、int32 回归范围 0/16 全 507035）；回退版本（无 kGM_、wsSize=0）67/74、`grep -c "507035"` = 0、UT 13/13（来源平台：950；证据等级：tool_observed）。
- unique 更大 workspace 标量访问仍报 `errcode(264) The address for scalar to access GM is invalid` / errcode 95——"极小 workspace（4B）导致"的候选解释被削弱（来源平台：950；证据等级：tool_observed）。
- GQA 单 Pass + workspace 缓存 scores：重写后 MERE 无改善、输出值回退到修复前形态、MHA 形态 `aclrtSynchronizeStream` 返回 507035，路线回退并记录 issue（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 「workspace 入参不可用」的结论限定本批实测的 aclnn ST 直调场景与用例集合；同一 unique/softmax 算子的**非 group 分支**绑 workspace 在历史上 ST 通过，不能一概而论所有路径都失效——上板对照验证前既不要依赖也不要直接判定可用。
- 本卡不宣称"workspace 参数一律不可用"（机制未定位，仅限观测场景）；按库内 `mte_ddr_507035_workspace_reserved_region` 的保留区约定正确声明与取址的用法未在本批 trace 验证。
- 同一 unique trace 内另有"DataCopyPad 批量搬 workspace 数据通路可用"的观测，与"指针整体无效"的解释存在未解张力（该观测的 workspace 尺寸/路径前提未单独核对），引用时不要把本卡外推为"workspace 一切访问必崩"。
- 常量特殊值用例通过 / 随机数据崩溃的判别依据未机制级解释，不能作为"workspace 部分有效"的证据。
- 两遍算法 UB 占用约 5-6KB（D=128、S_kv=128 量级），超大 S/S_kv 时 rowMax/rowSum（2×S×4 字节）与 scoresRow（S_kv×4 字节）需重估；相对单遍 + GM workspace 的性能代价未测量。
- 507035 本身是同步/stream 级错误码，还可能由其他原因（内存越界、非法地址）触发；本卡的归因是各 trace 的绑定对照，不是 507035 的通用解码表。

**未决项**：

- **优先项：复现并追根因到机制层**——逐层核对 workspace 传递链（autogen `NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize)` 到 kernel 第 3/5 参）与 kernel 内指针相对分配区间的位置；优先验证库内 `mte_ddr_507035_workspace_reserved_region` 的保留区偏移机制是否即本卡根因（声明含保留区 + 正确取址后 workspace 参数是否恢复可用）。
- 常量特殊值用例为何幸存、同 trace 内 DataCopyPad 通路可用的观测与指针无效解释的关系，均未机制级验证。
- `GetSysWorkSpacePtr()` 的确切可用容量未测定（当前仅由溢出现象反推）。
## 坏实践

- 只看 host `GetWorkspaceSize` 返回非零就断定 kernel 的 workspace 入参可用——host 分配与 kernel 收到是两回事，必须上板对照验证。
- kernel 内直绑 `SetGlobalBuffer((__gm__ float*)workspace)` 当作"框架必然传好"的常规操作——本批 4 个算子、3 条 trace 全部踩坑。
- 崩溃后不做 stash 对照就继续在崩溃态上叠改；或未定位机制前反复试错宏开关/偏移/长度写法（本批证均无效）。
- 把 `GetSysWorkSpacePtr()` 当正式方案——系统工作区容量小，大规模即精度异常。

## 误诊

- GQA 首轮把 507035 归因于 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 宏并移除——无效；参考算子恰恰带着该宏正常使用 workspace（该归因被观测否定）。
- 误诊为 ReduceSum dst 未 32B 对齐而改 `bcast[0]` 固定位置中转——无效、非根因（该写法作为通用模式保留在终版）。
- 怀疑 ST 仅在 `workspaceSize>0` 时分配导致空指针——实测 29184 非零，排除。
- 曾声称"507035 已修复、workspace 参数现在可用"——回测立即 507035，被观测否定。
- softmax 分支一度把错误码读成 361001（那是 kernel 分发未实现阶段的历史错误，出现在 aclnn 调用处）；实际是 507035、出现在 `aclrtSynchronizeStream`——两个错误码、两个调用点，对应两个不同根因。
- "若 workspace 无效则所有用例都应崩溃，特殊值用例通过说明 workspace 有效"——与实验结论矛盾，属推理误诊，勿引用。
