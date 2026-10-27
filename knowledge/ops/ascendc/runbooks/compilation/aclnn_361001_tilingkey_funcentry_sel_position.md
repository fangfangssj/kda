---
type: "Runbook"
title: "ACLNN 361001 — tilingKey 与 funcEntry 映射失"
description: "Host SetTilingKey(N) 后运行时按 funcEntry=tilingKey 查找内核入口，若 kernel 二进制未通过 ASCENDC_TPL_SEL 在对应 SEL 位置注册该 funcEntry，则 BinaryGetFunctionByEntry 失败并报 361001。给出 funcEntry=SEL 位置=tilingKey 的判别与 SEL 占位修复。 R2（2026-09）：SEL 只实例化部分组合时未列 key 报 361001（plog BinaryGetFunctionByEntry failed）或静默落到已实例化 kernel 产出错值（无报错形态），修复为 UI_LIST 逐值补齐；测试先行新 key 无分发另有 aclnn 整批 361001（launch 处）与 TTK RTS_BINARY_FAILURE+空 metrics 两形态，属预期红基线；稀疏编码键场景 funcEntry 按编码值匹配而非 SEL 位置。复现增量：无任何 ASCENDC_TPL 声明的 regbase 单 binary 算子，host 透传非零 key 必 361001（kernelList 仅 _0、clean rebuild 不变），互斥修复路线 A=host 恒置 0 + kernel 读 TilingData.tilingKey 字段分发；plog 特征 Value N for p / rtsFuncGetByEntry ErrCode=107000。"
tags: ["task.debugging", "topic.tiling", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# ACLNN 361001 — tilingKey 与 funcEntry 映射失

## 现象

在为 TileWithAxis 算子（ascend950 / arch35）实现 Branch-1 kernel（`tilingKey=1, ubAxis=inner`）后，运行 aclnn ST：`aclnnTileWithAxisGetWorkspaceSize` 与 `aclnnTileWithAxis`（launch）均返回 OK，但 `aclrtSynchronizeStream(stream)` 返回 **361001**。所有 Branch-1 ST 用例（10 个）失败，只有 BR1-9（空张量，走 op_api 空快速路径）通过。开启 `ASCEND_GLOBAL_LOG_LEVEL=0` 抓运行时日志，定位到运行时调用 `BinaryGetFunctionByEntry(funcEntry=1)` 找不到入口——kernel 二进制只注册了 `funcEntry_0`，而 Host TilingFunc 调用了 `context->SetTilingKey(1)`。Branch-2（tilingKey=2）与 Branch-3（tilingKey=3）首次联调时同样以 361001 失败，因为 stub kernel 仅有 `funcEntry_0`。

## 适用场景

- **工程形态**：aclnn 算子开发（ops-registry-invoke），Host TilingFunc 使用 `SetTilingKey(N)` 多分支分发，kernel 用 `ASCENDC_TPL` 模板按 KEY 编译多个分支。
- **API 路径**：`ASCENDC_TPL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_ARGS_SEL` / `ASCENDC_TPL_UINT_SEL`（struct.h 中声明 KEY 取值列表）；Host 侧 `context->SetTilingKey()`；运行时 `BinaryGetFunctionByEntry(funcEntry=tilingKey)`。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.1.0。
- **适用粒度**：`operator_family`（任何使用 `SetTilingKey` + `ASCENDC_TPL` 多分支 kernel 分发的 AscendC 算子）。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用或未验证场景**：单分支（无 `SetTilingKey` 或恒为 0）的算子不会触发；a2/a3 平台未验证（推断可迁移，标注于 `## 失效边界`）。

- （R2 增补）RTS_BINARY_FAILURE 不只由"key 未实现"引起——真实的注册问题（tiling key 值非法、二进制损坏、编译失败残留）同样报它；判别闭合条件是确认该 key 在 apt 入口确为 stub/未实现，不能只凭报错文案下结论。
- （R2 增补）"no-op else 分支为何不产生可用二进制"的机制层解释与本卡 funcEntry 机制互补，未做二进制对账实验闭合。

## 判别方法

1. **触发信号**：aclnn launch 返回 OK 但 `aclrtSynchronizeStream` 报 361001；或 ST 全部用例（除走空快速路径的空张量用例）一致失败。
2. **可执行探针**：查看部署后的 kernel 配置 JSON（`$ASCEND_OPP_PATH/opp/vendors/<OpType>/op_impl/ai_core/tbe/kernel/config/ascend950/<op>.json`）的 `binList` 与 funcEntry 列表；再 `cat` 构建产物中的 kernel list 文件，确认注册了哪些 `funcEntry_N`。
3. **原始观测**：运行时 DEBUG 日志显示 `BinaryGetFunctionByEntry(funcEntry=N)` 失败（kernel 二进制仅含 `funcEntry_0`），而 Host 侧 `SetTilingKey(N)` 设置了 N>0。
4. **解释**：funcEntry = `ASCENDC_TPL_SEL` 列表中的**位置（index_value）**；运行时**直接令 `funcEntry = tilingKey`** 查找入口。因此要让 `tilingKey=1` 可达，必须让 `KEY=1` 的 SEL 组合出现在 SEL 列表的 index 1 处——即需要一个 index 0 的占位（dummy KEY=0，运行时永不调用），再放 KEY=1。同理 KEY=2 放 index 2。funcEntry 既不是 tilingKey 的编码值本身，也不是 DECL 列表顺序——它是 SEL 列表中该组合的**位置**；只有当 SEL 按 KEY 升序且无空缺时，位置才恰好等于 KEY。

## 解决配方

修复在 `op_kernel/arch35/<op>_struct.h` 的 `ASCENDC_TPL_SEL` 中显式列出全部 KEY（含 index 0 的占位 KEY_0），使每个 `tilingKey=N` 都能在 SEL 位置 N 找到对应入口；并在 kernel 头里 `#include` struct.h，使 `ASCENDC_TPL` 符号进入 tiling `.so`（kernel-file 链）。

**before**（struct.h 只声明了 KEY=0，SEL 仅一项 → 只有 funcEntry_0）：

```cpp
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_0))
);
```

Host 调 `SetTilingKey(1)` 时运行时查 `funcEntry=1` 不存在 → 361001。

**after**（加入 KEY_0 占位 + KEY_1，使 KEY=1 位于 SEL 位置 1）：

```cpp
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_1))
);
```

Branch-2/3 同理扩展（KEY_2 放位置 2、KEY_3 放位置 3）：

```cpp
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_LIST, TILE_WITH_AXIS_KEY_2))
);
```

kernel 头需 include struct.h（使 ASCENDC_TPL 符号在 tiling .so 中可见，参考 scale_custom 的 kernel-file 链）。

## 验证信号

本次开发会话（TileWithAxis Branch-1 联调）修复后实测：

- 构建产物 kernel 配置 JSON 中出现 `funcEntry_0`（KEY=0 占位）与 `funcEntry_1`（KEY=1 Branch-1）两个入口（来源平台：950；证据等级: tool_observed）。
- Branch-1 aclnn ST：10/10 用例 bitwise 通过，含空张量、特殊值、多核、全部 dtype（来源平台：950；证据等级: tool_observed）。
- 回归：公共 TilingUT 21/21 + Branch-1 TilingUT 10/10 均无退化（来源平台：950；证据等级: tool_observed）。

- （R2 增补，Ascend950PR + CANN 9.2.0，BenchHarnessSoftmax 新增 tilingKey=4 的测试先行）aclnn 形态：stub 期 Branch-4 汇总 9/40 precision PASSED、31 aclnn-FAIL，31 条失败均在 `aclnnBenchHarnessSoftmax(workspace, workspaceSize, executor, stream)` 调用行报 ACLNN error 361001；9 条 PASS 全为设计上路由 key=1 的边界用例（#421/#422/#423）（来源平台：950；证据等级：tool_observed）。
- （R2 增补）配对观测：kernel 分发补上（key=4 与 key=1 合并分发到同一 Kernel<_,false> 模板）后同批用例 361001 不再出现，转入正常执行路径（终态 40/40 PASS）（来源平台：950；证据等级：tool_observed）。
- （R2 增补）TTK 形态：case00084/case00085（bf16 (2049,2047) dim=-2/dim=0）dyn_tiling_key=4、precision_metrics 全空、dyn_prec=RTS_BINARY_FAILURE；执行日志报 `Kernel register failed. Expected symbol dyn_op_case00085 does not exist ... This is usually caused by wrong tiling key.`；apt 入口确认为 `key=4 (GroupTailA): not yet implemented. Stub: no-op.`（来源平台：950；证据等级：tool_observed）。
- （R2 增补）字段级对照：同批 case00076 dyn_tiling_key=0、metrics 含 stat_rel_err 数值（mere/mare/threshold）——两种 FAIL 形态可区分（来源平台：950；证据等级：tool_observed）。

## 失效边界

- Ascend950（arch35）实测；a2/a3 平台未验证，推断可迁移（`ASCENDC_TPL` / `SetTilingKey` 为跨 arch 通用机制），但 funcEntry 编号策略是否随 toolkit 版本变化未验证。
- 若 SEL 列表顺序与 KEY 升序不一致（例如 SEL{2,1}），funcEntry≠tilingKey，本卡的"位置=tilingKey"简化不再成立——需按 SEL 实际位置核对 kernel JSON。
- 本卡仅覆盖 funcEntry 注册缺失导致的 361001；kernel 本身访问非法内存同样可能引发 361001 类（vector core exception），需用 507035/361001 的运行时日志区分。

## 预防

- 任何使用 `SetTilingKey(N>0)` 的算子，struct.h 的 `ASCENDC_TPL_SEL` 必须为 0..N 全部 KEY 显式列出 SEL 项，且 KEY=0 留作占位（不调用）。
- 联调前先 `cat` kernel 配置 JSON 确认 funcEntry 数量 = tilingKey 取值数 + 1（含占位）。

## 坏实践

- 只在 SEL 中列出"用到的 KEY"而忽略 index 0 占位 → funcEntry 编号错位，`tilingKey=1` 命中 index 0 的占位而非 KEY=1。
- 误以为 funcEntry = tilingKey 的编码值（实际 funcEntry = SEL 位置），跳过占位直接放 KEY=1 于 index 0 → `tilingKey=1` 查 `funcEntry_1` 仍不存在。

- （R2 增补）见到 361001 就查环境（重装算子包、查 ASCEND_OPP_PATH、核对 vendor 目录）——测试先行新 key 场景下包和路径都是好的，浪费排障轮次。
- （R2 增补）把 TTK 里 precision_metrics 为空的 FAIL 当精度缺陷派修复——没有比对数据，按精度排查方向从根上就错。

## 误诊

- 起初误判为 kernel 未实现/空 kernel 崩溃（507035 vector core exception），后通过运行时 DEBUG 日志定位到 `BinaryGetFunctionByEntry(funcEntry=1)` 失败，才确认是 funcEntry 注册问题而非 kernel 本身崩溃（agent_reported，非实测信号）。

## 增补：测试先行新 key 无分发的两种失败形态（R2，2026-09，CANN 9.2.0）

同一机制（tilingKey 路由到无 kernel 可执行体的 key）在两条评测路径上的形态与红基线判读：

**形态一：aclnn ST 整批 361001（launch 处）**。host TilingFunc 已实现并路由新 key（分支 TilingUT 通过）、kernel APT 入口分发无该 key 分支（残留 no-op else）时，定向新 key 的用例在 `aclnn<Op>(workspace, workspaceSize, executor, stream)` 调用行整批报 `ACLNN error 361001`；同期路由到已实现 key 的边界用例全部 PASS（同时验证 host 路由没写错）。把该 key 并入 APT 分发（如 `TILING_KEY_IS(1) || TILING_KEY_IS(4)` 共用同一 Kernel 模板）后 361001 立即消失——配对观测。判读要点：361001 出现在 aclnn 调用行（launch 失败），507035 出现在 `aclrtSynchronizeStream`（执行崩溃），必须按错误串里的调用点区分。

**形态二：TTK kernel 模式 RTS_BINARY_FAILURE + 空 metrics**。白盒 result.csv 中 `dyn_prec=RTS_BINARY_FAILURE` 且 `precision_metrics` 全空（`{'dyn': {}, 'cst': {}, 'bin': {}}`）、执行日志报 `Kernel register failed. Expected symbol dyn_op_caseNNN does not exist in .../kernel_meta/dyn_op_caseNNN.o. This is usually caused by wrong tiling key.`——kernel 没有成功注册/执行、根本没产生比对数据，不是精度失败；有 stat_rel_err 数值的 FAIL 才是精度问题。探针：读 result.csv 的 `dyn_tiling_key` 字段拿实际路由 key，对照 apt 入口实现状态（grep `not yet implemented` / `Stub: no-op`）。

**处置**：归因与排期而非改代码——把失败用例按 dyn_tiling_key 分类（精度问题 vs key 未实现），未实现 key 归后续分支任务，实现后重跑转绿。红基线验收口径：定向新 key 的用例全部进入该失败态且边界用例不回归。

## 增补：SEL 部分实例化——未列组合报 361001 或静默落错 kernel（R2·并行策展批次增补）

并行批次第三条 trace 补齐 SEL 视角的同根因形态与一处机制修正：

**SEL 只实例化部分组合**：`ASCENDC_TPL_SEL` 只实例化列表内枚举的模板参数组合；tiling 侧按属性算出的 TilingKey 落在未实例化组合上时，运行时 `aclrtBinaryGetFunctionByEntry` 找不到入口报 361001。plog 报错原文（`ASCEND_GLOBAL_LOG_LEVEL=1`）：

```
[ERROR] RUNTIME ... [api_impl.cc:999] BinaryGetFunctionByEntry:BinaryGetFunctionByEntry failed.
Value 257 for parameter funcEntry is invalid. Reason: The funcHandle cannot be found through funcEntry. ErrorCode=EE1001.
[ERROR] ASCENDCL ... [kernel.cpp:205] aclrtBinaryGetFunctionByEntryImpl:Binary get function by entry Failed, runtime result = 107000
[ERROR] OP ... [op_kernel.cpp:410] errno[361001] aclrtBinaryGetFunctionByEntry failed,
kernel name: BenchNoHarnessMoeFinalizeRouting, tiling key: 257
```

**静默落错 kernel 形态（更隐蔽）**：个别未实例化 key（本例 mode=2，key=513）不报 361001，而是静默落到已实例化的 mode0 kernel 执行——输出与"mode0 公式"的 golden 逐位一致（MERE=4.66e+00 对正确 golden），表现为精度全错而非报错。报 361001 的用例与静默错的用例要分开归因。

**修复：SEL 按 UI_LIST 逐值补齐全部组合**（verbatim）：

```cpp
// before：SEL 只实例化 TK_0
ASCENDC_TPL_SEL(
    // TK_0: fp16 + drop_pad_mode=0 (drop less 按列)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_MAIN, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(DROP_PAD_MODE, ASCENDC_TPL_UI_LIST, 0)
    )
);

// after：SEL 覆盖全部 4 个 mode
ASCENDC_TPL_SEL(
    // TK_0~3: fp16 + drop_pad_mode=0/1/2/3
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_MAIN, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(DROP_PAD_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
    )
);
```

配套检查：重编重装后用 kernel json 的 `kernelList`（本例 key=1/257/513/769）核对实例化是否齐全；tiling 能产生的每个 key 都应在 SEL 内。

**机制修正（对本卡判别方法第 4 条）**：本卡"funcEntry = SEL 列表中的位置"的简化只在 SEL 紧凑按序（0..N 无空缺）时成立；本例 kernelList 含稀疏编码键 257/513/769（UINT 位段编码值，见库内 tilingKey 位打包卡），运行时按**编码后的 TilingKey 值**匹配 funcEntry 而非列表位置——稀疏/编码键场景须以 kernel json 的实际键集对照，不能按位置推算。

验证信号（R2·独立 trace 增补，Ascend950PR + CANN 9.1.0，BenchNoHarnessMoeFinalizeRouting ST 联调）：

- 修复前：L0-002/L0-004 报 361001，plog `aclrtBinaryGetFunctionByEntry failed, tiling key: 257/769`（EE1001，rts 107000）；L0-003（mode2）无报错但 MERE=4.66e+00，DIAG 探针 `actual vs golden(mode0按列) match=1`——kernel 实际执行了 mode0 公式（来源平台：950；证据等级：tool_observed）。
- 修复后（SEL 补齐 0/1/2/3 并全量重编重装）：L0-002（mode1 3D）PASS（MERE=2.78e-08）；4 个 mode 最终全 PASS（MERE 2.13e-08~4.13e-08）（来源平台：950；证据等级：tool_observed）。

失效边界（R2·独立 trace 增补）：

- mode2 静默落到 mode0 kernel 的 fallback 机制是从观测（输出与 mode0 golden 逐位一致）推断的结论，未直接观测运行时选核逻辑。
- 本卡"位置=tilingKey"简化仅适用于紧凑按序 SEL；稀疏编码键场景以 kernel json 实际键集为准。

## 增量（单 binary 无 TPL 的互斥路线 A：恒置 0 + TilingData 字段分发）

两个独立 trace（CANN 9.1.0 与 9.2.0 各一）补齐"**未声明任何 ASCENDC_TPL** 的 regbase 单 binary 算子"视角的同根因形态与互斥修复路线：

**现象**：kernel 入口在单一 `__global__` 函数内按 `TilingData.tilingKey` 字段 if/else 分发（无编译期模板变体），host TilingFunc 把设计文档的"分支号"语义（0=FullloadR、1=ChunkedR、2=Empty 之类）原样透传 `context->SetTilingKey(tilingKey)`——构建通过，但新分支的全部数值用例 aclnn 返回 361001、输出全 0（既有 key=0 分支用例正常）。plog 特征与 R2·独立 trace 增补一致：`BinaryGetFunctionByEntry:Obtaining the kernel function handle based on the function entry failed. Value 1 for p` + `rtsFuncGetByEntry:ErrCode=107000, desc=[invalid value], InnerCode=0x7110001`；kernel 二进制 JSON `kernelList` 仅 `[{"kernelName": "…_<hash>_0"}]`，clean rebuild 后仍只有 `_0`。

**机制**：运行时按 host `SetTilingKey` 的值在 kernelList 里查找后缀 `_<key>` 的入口；opc 不会从 kernel 源码里 `if (tilingData.tilingKey == N)` 这类运行时 if/else 自动发现并产出 `_N` 入口——"tilingKey=分支号"的语义只有在 kernel 侧用 ASCENDC_TPL 模板选择编译出对应变体（本卡主修复/SEL 补齐路线）时才成立。

**修复（路线 A，与 SEL 注册路线互斥）**：单 binary 范式下分支分发全部留在 kernel 内（读 `td->tilingKey`），host 恒置入口键 0：

```cpp
    // 修复前（透传分支语义键）：
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));
    // 修复后（regbase 单 binary：opc 仅产出默认 _0 入口，分支由 kernel 读 TilingData.tilingKey 分发）：
    context->SetTilingKey(0);
```

空 tensor 早退分支同理（原 `SetTilingKey(2)` 也选不到 binary，一并改 0，由 `usedCoreNum==0` 早退）。前提：各 `ComputeBranch*Tiling` 已把真实 tilingKey 写进 TilingData 字段（kernel 分发读 `td->tilingKey`，与框架入口键解耦）。

**两条路线不可混用**：声明了 per-key SEL 后 host 又恒置 0，非 0 入口永远不被命中；反之未声明却传非 0 key 就是本增量故障。判别顺序：先 `cat` kernel 二进制 JSON 的 `kernelList` 看入口后缀集合（只有 `_0` → 必须走路线 A；有 `_0.._N` → host 传真实 key，缺失项走 SEL 补齐），再对照 host 传入值。

**边界注意**：另一张同主题卡（tiling_key_is_runtime_dispatch_single_binary）记载 TILING_KEY_IS 宏路线在单 binary 下 host 置非零 key 可用（kernel 读框架传入的 g_tilingKey）——该路线与"kernel 读 TilingData 字段 + host 恒 0"的路线 A 注册/分发机制不同，两轨迹结论未在同一工程对拍；采用任一路线前以本工程 kernelList 形态实测为准。

**ST 侧掩盖提示**：自建 aclnn ST 在 `s1==0` 时仍跑 golden 对比并覆盖 `ret.detail`，361001 被"输出全 0"的对比信息顶掉——须用最小直连探针还原返回码（见库内 aclnn_st_masks_execute_error_code_probe 卡）。

验证信号（复现增量，两个独立 trace）：

- trace 1（CANN 9.1.0，BenchHarnessSoftmax Branch-1）：修复前 ST 0/9 全失败输出全 0、Branch-0 32/32 PASS；直连探针 `GetWorkspaceSize: 0, wsSize=0 / aclnn: 361001 / nonzero output elements: 0`；`SetTilingKey` 统一 0 后 kernel 实际执行（Branch-1 转 4/9，剩余为独立精度问题），后续精度修复后 Branch-1 9/9 + Branch-0 32/32、连续 3 次运行确定（来源平台：950；证据等级：tool_observed）。
- trace 2（CANN 9.2.0，BenchHarnessApplyRotaryPosEmb interleaved 分支）：修复前 tilingKey=1 的 11 个数值用例全部 `ret=361001`，plog 报 `BinaryGetFunctionByEntry … Value 1 for p` 与 `rtsFuncGetByEntry:ErrCode=107000`；kernelList 仅 `_0`、clean rebuild 后仍仅 `_0`；host 改 `SetTilingKey(0ULL)` 后 361001 全部消失、用例进入 kernel 执行，tilingKey=0 既有分支不受影响（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 把设计文档的"tilingKey=分支号"语义直接透传给 `context->SetTilingKey()` 而 kernel 侧没有对应的编译期模板变体——分支号是设计语义，不是框架 binary 键。
- 构建通过就认为"新增分支已生效"——入口缺失在编译期完全无感知，只在运行期以 361001 暴露。
