---
type: "Runbook"
title: "输出恒为哨兵位型：入口分支stub未接线"
description: "多分支算子 runtime TILING_KEY_IS 分发的 stub 分支输出保持测试预填哨兵位型（0x41/0xFF 跨 dtype 解码一致）；按「入口未接线 / 路由到别的分支 / kernel 早退」三成因判别后把入口接线到已实现 kernel 类；group 与 EMPTY 两实例接线后全绿。复现增量：host/kernel 分支集不同步形态——host 新增分支下发新 tilingKey 而 kernel 入口 dispatch 未同步加分支时执行落 else 桩（仅 PipeBarrier），aclnn 返回 0 无报错、0xFF 毒化读回全 -nan；四处同步清单（host tiling 函数 → glue 派发 → kernel 入口 → 计算链）。"
tags: [task.debugging, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# 输出恒为哨兵位型：入口分支stub未接线

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上继续开发 euclidean_norm 算子的 group 分支（tilingKey=4，Group Tail-A：A 维用不满核而 R 维有并行度时做 A×R 2D 分核，两阶段执行——Phase 1 各核局部 R 段归约写 workspace fp32 部分和，`SyncAll` 跨核同步后 Phase 2 读全 R 分组做最终归约 + Sqrt + Cast）。此时 host 侧 tiling 已实现（`ShouldUseGroup` 触发、`ComputeGroupSplit` 覆盖核数、`rGroupCnt` 与 workspace 已声明），kernel 类（`InitGroup`/`ProcessGroup` 两阶段实现）也已在 kernel 头文件中完成，ST 用例已按实测路由适配好 shape（如 `[2,1024,3]` `axes=[1]`、`[1,4096,2]` `axes=[1]` 等 8 个精度用例，输出 buffer 预填 0x41 有限哨兵）。跑 ST 基线：8 个精度用例全失败，报错原文：

```
[G1   [2,1024,3] axes=[1] fp32 kd=false (ARA group, Phase1 bisect+Phase2 RA)] FAILED: exceeds rtol/atol (elem0 dev=12.078431 golden=56741.984375)
[G5   [2,1024,3] axes=[1] int32 kd=false (CAST_TRUNC bitwise)] FAILED: int32 bitwise mismatch (elem0 dev=1094795648.000000 golden=56741.000000)
[G6   [2,1024,3] axes=[1] bf16 kd=false (CAST_RINT bit-bridge)] FAILED: exceeds rtol/atol (elem0 dev=12.062500 golden=56832.000000)
[G7   [1,4096,2] axes=[1] fp16 kd=true (keep_dims, fp32 accumulate)] FAILED: golden Inf but device mismatch (elem0 dev=2.626953 golden=inf)
```

关键判别点：dev 值在四种 dtype 下分别**恰为 0x41 字节填充的位型解码**——fp32 `0x41414141`=12.078431；int32 `0x41414141`=1094795585（测试经 float 打印为 1094795648）；bf16 `0x4141`=12.0625；fp16 `0x4141`≈2.626953。kernel 一个元素都没写。逐文件排查发现：kernel 头文件里 group 实现完整，但算子入口 `.apt.cpp` 的 `TILING_KEY_IS(4)` 分支仍是 `(void)tilingData;` no-op stub——kernel 类从未被调用。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC kernel + host TilingFunc），多 tilingKey 分支分发（runtime `TILING_KEY_IS` 判别），kernel 类与入口分步开发/分人开发。
- API 路径：算子入口 `.apt.cpp` 的 `REGISTER_TILING_FOR_TILINGKEY` + `GET_TILING_DATA_WITH_STRUCT` + kernel 类 `InitGroup(x, y, workspace, &td)` / `ProcessGroup()`；ST 侧 `aclrtMemset` 哨兵预填。
- 验证过的平台与 CANN 版本：Ascend950PR（A310-50），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——"输出=哨兵 ⇒ 分支未执行"的判别与入口接线模式适用于该平台任何多分支 tilingKey 算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：没有预填哨兵的测试（零初始化 buffer 会把 no-op 伪装成全 0 输出，判别失效）；kernel 类本身尚未实现的分支（接线不能代替实现，本卡前提是 kernel 类已存在）。

## 根因

算子入口按 tilingKey 的分发分支未接线：tiling 路由正确、kernel 类已实现，但入口 `TILING_KEY_IS(4)` 分支不构造、不调用任何 kernel 对象，输出保持测试预填的哨兵。这是"kernel 类与入口由不同任务/角色分步交付"开发模式下的典型遗漏——编译不报错（stub 合法）、tiling 正常、只有端到端 ST 能暴露。

## 解决配方

修复前（入口，no-op stub）：

```cpp
    } else if (TILING_KEY_IS(4)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        (void)tilingData;   // Group Tail-A kernel 未接线
    } else if (TILING_KEY_IS(5)) {
```

修复后（接线到 group kernel；`isTailR=false` 是 Group Tail-A 的模板实参）：

```cpp
    } else if (TILING_KEY_IS(4)) {
        // Group Tail-A (templateType 分支, isTailR=false, RA pattern).
        //   Phase 1 → SyncAll → Phase 2: A×R 2D 分核局部 R 段 reduce 写 workspace
        //   fp32 部分和，跨核 SyncAll 后 Phase 2 RA mini-kernel 读 workspace 全 R
        //   分组做最终 reduce + Sqrt + Cast → yGm。
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormKernel<DTYPE_X, /*isTailR=*/false> op;
        op.InitGroup(x, y, workspace, &tilingData);
        op.ProcessGroup();
    } else if (TILING_KEY_IS(5)) {
```

判别方法（输出恒为哨兵常数的定位链）：

1. 看到 ST 失败值在多种 dtype 下"各自是同一个常数"（fp32 12.078431、int32 1094795648、bf16 12.0625、fp16 2.626953），先查测试是否预填哨兵及哨兵值；把这些常数按目标 dtype 解码位型（12.078431 → `0x41414141`），与哨兵对上 ⇒ kernel 未写输出。
2. 区分"未写输出"的三种可能：入口分支 stub（本卡）、tilingKey 路由到了别的分支（用路由探针确认）、kernel 早退（查 `usedCoreNum`/早退条件）。本例 tiling 侧 UT 全绿 + 路由探针已确认 key=4，故锁定入口 stub。
3. 接线后重跑 ST：失败值消失、出现真实数值（容差内）即确认。

## 设计决策

- **group 入口用 `InitGroup(x, y, workspace, &tilingData)` 而非普通 `Init`**：两阶段 group 的 Phase 1 要把各核部分和写到 GM workspace、Phase 2 再读回，workspace 指针必须传进 kernel；这是 group 分发与 normal/empty 分发在入口签名上的本质差异。
- **保持 runtime `TILING_KEY_IS` 分发而非模板分发**：dtype 走 `DTYPE_X` 编译期实例化（每 dtype 组合一份 binary），tilingKey 不进 dtype 维度；运行时按 key 分发与既有 key=0/1/2 分支风格一致，改动面最小。
- **`(void)workspace;` 兼容性**：入口尾部对仍为 stub 的 key=5 保留 `(void)workspace;`，对已使用 workspace 的 key=4 无害（已使用变量的 void 转换是 no-op）。
- 当前方案路线档位：recommended_fastpath（正确性必需）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0。
- 相关文件：`op_kernel/euclidean_norm_apt.cpp`（入口接线，本卡修改点）、`op_kernel/arch35/euclidean_norm.h`（group kernel 类 `InitGroup`/`ProcessGroup`/Phase1/Phase2，先于本场景实现）、`tests/aclnn/test_aclnn_branch4.cpp`（ST + 0x41 哨兵 + 独立 oracle）、`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（group tiling，先于本场景实现）。
- 验证命令：`cd operators && ./build.sh euclidean_norm_package`，`cd euclidean_norm_package/tests/aclnn && ./run.sh`。

## 预防

- 多分支算子采用"kernel 类与入口分步交付"时，每个新分支落地必须同时改两处：kernel 类实现 + 入口分发接线；入口文件头的分支状态注释（stub/已实现）随改随更新，作为 checklist。
- ST 输出预填**有限**哨兵（如 0x41）并记录各 dtype 解码值，失败输出可一眼对上位型。
- `grep -n "TILING_KEY_IS\|(void)tilingData" op_kernel/*_apt.cpp` 快速盘点哪些分支还是 stub。

## 验证信号

来源说明：以下信号来自 group 分支 ST 用例在入口接线前后的同套对比（独立 CPU oracle 判分，设备 Ascend950PR）。

- 基线（入口 stub）：Branch-4 精度用例 0/8 通过，全部失败、dev 恒为 0x41 哨兵位型（fp32 12.078431 / int32 1094795648 / bf16 12.0625 / fp16 2.626953）（来源平台：950；证据等级：tool_observed）。
- 接线后：Branch-4 精度用例 8/8 PASSED（fp32 maxRel 3.122e-07/3.097e-07/4.038e-07；int32/bf16/fp16-keepdims 按位 0；NaN 注入用例 NaN 传递通过；int64-axes 变体通过）（来源平台：950；证据等级：tool_observed）。
- 回归：Branch-0 5/5、Branch-1 18/18、Branch-2 14/14 无退化，`run.sh` 退出码 0（来源平台：950；证据等级：tool_observed）。
- 构建门：`build.sh` 退出 0、无 "do not registe tiling struct" 警告、8 份 ascend950 kernel binary 生成、`__simd_vf__`/`asc_vf_call` grep 计数达标（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "输出=哨兵 ⇒ 未执行"只对预填了哨兵的测试成立；无哨兵时 no-op 可能伪装成全 0。
- 哨兵解码表（0x41 → 各 dtype 值）是 IEEE 位型事实，跨平台通用；但"哪个 tilingKey 路由到该分支"由 tiling 侧决定，本卡不覆盖路由判定。
- group kernel 类（Phase1/SyncAll/Phase2）的实现与验证先于本场景，不在本卡证据内；本卡只覆盖入口接线层。

**未决项**：

- 同一入口中 `TILING_KEY_IS(5)`（Group Tail-R）分支仍是 stub，待后续接线（机制同本卡，无需新知识）。
- EMPTY kernel 侧实现配方的细节（早退/常量填充/同步）见库内空 batch 归约 kernel 卡；本卡只覆盖入口接线层。
## 坏实践

- 只实现 kernel 类不改入口（或反之），依赖"编译通过"判断完成度——runtime 分发的 stub 是合法 C++，编译器不会提醒。
- 把 `dev=12.078431` 这类常数失败当成数值精度问题去查计算链——先对哨兵位型再定方向。

## 误诊

- 初判容易走向"group 两阶段算法实现有 bug"（查 workspace 布局/SyncAll 顺序）；判别锚点是失败值恒等于哨兵位型且**逐 dtype 都是同一个填充模式的解码**——数值 bug 不会在四种 dtype 下都精确复现哨兵常数。

## 同根因实例：EMPTY 分支（tilingKey=2，0xFF 哨兵）

同一算子的空 tensor 分支给出第二个实例与另一张哨兵解码表。EMPTY 分支 kernel 未实现时入口同为 `(void)tilingData;` no-op stub；测试把输出 device buffer 预填 **0xFF** 哨兵后，11/14 精度用例失败，`dev=-nan`（fp32）/ `dev=-1`（int32）正是 0xFF 字节填充的位型解码——kernel 一个字节都没写。判别锚点与主实例一致：失败值跨 dtype 恒为同一填充模式的解码，数值 bug 不会在多种 dtype 下精确复现哨兵常数。

两个必须配套的认知：

- **空输出用例对 no-op stub 无检测力**：3 个"通过"用例全是 EMPTY_A（输出 tensor 本身为空，比较 0 个元素恒过）——它们的通过不代表 kernel 正确，覆盖力来自分类正确性而非数值比较。
- **零初始化 buffer 会把 no-op 伪装成全 0 输出**：golden 期望全 0（空输入归约 `sqrt(Σ∅)=0`）时必须预填非零哨兵，否则 stub 假通过。

修复同样是入口接线（EMPTY kernel 类已按"空输入归约 = 常量 0"的数学退化链实现：`y = sqrt(Σ_{i∈∅} x_i²) = sqrt(0) = 0`，且 `Cast(0)=0` 对 fp32/fp16/bf16/int32 全部成立；修复前形态与主实例相同，即 `(void)tilingData;`）：

```cpp
    } else if (TILING_KEY_IS(2)) {
        // EMPTY (isEmptyTensor 分支, EuclideanNormEmptyKernel)
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", EuclideanNormTilingData);
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormEmptyKernel<DTYPE_X> op;
        op.Init(y, &tilingData);
        op.Process();
        (void)x;   // empty kernel 不读输入 x
    } else if (TILING_KEY_IS(4)) {
```

EMPTY kernel 侧的完整实现配方（EMPTY_A 全核早退并跳过 InitBuffer / EMPTY_R Duplicate-0 → DataCopyPad 多核填充）已沉淀于库内《空batch归约kernel：bias零填充+全核早退配方》卡，此处不重复。

增补验证信号（同 trace，独立 CPU oracle 判分，Ascend950PR）：

- 基线（EMPTY 入口 stub）：Branch-2 精度用例 3/14 通过（仅输出为空的 EMPTY_A），11 例失败、dev 为 0xFF 哨兵位型（fp32 `-nan`、int32 `-1`）（来源平台：950；证据等级：tool_observed）。
- EMPTY 接线后：Branch-2 精度用例 14/14 PASSED，全部 `maxRel=0.000e+00`（按位 0，含 int32 逐位比较、fp16/bf16 CAST_RINT、int64-axes 变体、`[2,0,3]` 多轴、`[0]` 一维全归约）（来源平台：950；证据等级：tool_observed）。
- 回归：Branch-0 5/5、Branch-1 18/18 无退化（来源平台：950；证据等级：tool_observed）。
- 空 tensor 的 device 分配按最小 32 字节预留（`aclrtMalloc` 不接受 0 字节）（来源平台：950；证据等级：tool_observed）。

## 增量（host/kernel 分支集不同步形态与四处同步清单）

另一 trace（RoPE 类 BenchHarnessApplyRotaryPosEmb，Ascend950PR + CANN 9.2.0）补齐同一"入口未接线"根因的 **host 侧推进、kernel 侧滞后** 变体：host 侧 tiling 已全绿（`ComputeBranch3Tiling` 下发 `tilingKey=3`，glue 已派发），kernel 入口 dispatch 只到 `tilingKey == 0/1/2`——新增 key 落入 else 桩 `AscendC::PipeBarrier<PIPE_ALL>();`（不写任何输出）。与主实例（入口分支留 `(void)tilingData;` no-op）机制相同，但触发方式是"新增分支时只改了 host 两处、漏改 kernel 入口"，更容易在多人/多任务协作里发生。

**判别要点（补充主卡的三成因判别）**：`aclnn` 调用本身返回 0（下发成功）、无运行时报错、无段错误——kernel "正常跑完"但什么都没写。`actual=-nan` 且**所有** numeric 用例在 idx=0 就 mismatch、而 returns_empty 用例（S=0，0 元素无比对）结构通过，是"分支未派发/未写输出"的签名；若是坐标反算或精度问题，通常只有部分用例错或错值是有限数。毒化 NaN 与输入 NaN 无关——输入含 NaN 的用例在红基线也报 `expected=<有限值>`，说明 mismatch 索引落在期望有限值的元素上。

**修复（入口加分支并复用相邻分支 kernel 模板）**：

```cpp
// 修复前：入口只派发 tilingKey == 0/1/2，tilingKey == 3 落 else 桩（仅 PipeBarrier）。

// 修复后（新增分支 = 入口 else-if + 复用相邻分支 kernel 类，kIsBnsd 编译期切换坐标反算）：
    } else if (tilingData.tilingKey == 3u) {
        // Branch-3 (interleaved-BNSD-cos2d): 复用 Branch1Kernel 计算链（interleaved rotate_half
        // DeInterleave+Muls+Interleave + VF MulAdd），kIsBnsd=true 切换坐标反算（s=row%S，s-stride）。
        Branch1Kernel<DTYPE_QUERY, true> kernel;
        kernel.Init(query, key, cos, sin, queryOut, keyOut, &tilingData);
        kernel.Process();
    } else {
```

（复用相邻分支模板的前提是两分支计算链逐指令相同、只差坐标反算/源基址类编译期差异；计算链不同的分支必须新写 kernel 类，且按模式选对源分支——half 与 interleaved 的 rotate_half 链不同，抄错源分支会引入真正的数值错误。）

**多分支新增分支的四处同步清单**（漏任何一处都会以"静默不写/写错"暴露）：

1. host 特化 tiling 函数（`ComputeBranchNTiling` 填差异字段与 tilingKey）；
2. host TilingFunc glue 的 tilingKey 派发臂；
3. **kernel 入口派发**（本增量漏点）；
4. kernel 计算链（复用时核对坐标反算/源基址口径与 tiling 的 stride 字段配对）。

验证信号（复现增量，同算子 branch-3 kernel 实现阶段，ST 上板 Ascend950PR）：

- 修复前：branch-3 ST `numeric 0/11 PASSED, returns_empty 1/1 OK (total 1/12)`，11 个用例全部 `actual=-nan`（如 a6_fp32 idx=0 actual=-nan expected=0.389963）（来源平台：950；证据等级：tool_observed）。
- 修复后：branch-3 ST 12/12（numeric 11/11 + returns_empty 1/1），含三 dtype、B 广播、最小 D=2、大 shape、非对齐 D=10、NaN/全零/Inf/大值特殊值用例全过（来源平台：950；证据等级：tool_observed）。
- 回归：全量 `run.sh` 四分支各 12/12、脚本退出码 0；UT 85/85（kernel 改动不影响 host tiling）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- `actual=-nan` 全用例签名依赖 ST 框架的 0xFF 毒化；若测试不毒化输出，未派发分支可能表现为"输出为未初始化内存的随机值"或恰好全 0，判别法失效（与主卡"零初始化 buffer 伪装成全 0"同一条边界）。
- else 桩写部分输出（半实现）会污染红基线判别——桩必须保持纯 no-op（仅 PipeBarrier）。

坏实践（复现增量）：

- 只改 host 不改 kernel 入口就跑 ST——新 tilingKey 静默落桩，浪费一轮构建/上板时间。
- 修复时把 half 分支的 kernel 模板误当可复用模板照抄给 interleaved 分支——必须按模式选对源分支。