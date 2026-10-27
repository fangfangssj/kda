---
type: "Runbook"
title: TTK kernel 冒烟判定 tiling 分发命中 vendor V2
description: vendor 自定义算子在 ascend950 上用最小 kernel 用例冒烟，以 tiling data/blockDim 指纹、同名 V1 注册源排除与标记零计数判定 tiling 分发命中 V2，防 V1 劫持误判。
tags: [task.testing, topic.implementation.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TTK kernel 冒烟判定 tiling 分发命中 vendor V2


## 现象

算子仓开发流程（ops-registry-invoke，ACLNN 自定义算子对标 torch 的 GRU Cell 反向）中，自定义算子的接口桩工程已构建、已安装为 CANN vendor 包（`$ASCEND_OPP_PATH/vendors/<OpName>/`），接下来要跑大批量 TTK（ops-test-kit）白盒/黑盒用例。此时流程要求做一次**目标算子通路预检**：确认 TTK kernel 模式的 **tiling 分发通路命中 vendor V2 tiling**（`gert::TilingContext` 通路），而不是被 CANN 内置 V1 同名 tiling 劫持 / `TbeOpTilingPyInterfaceOld` 阻断——这类框架级问题会以 `OPTILING_FAILURE` 形式炸出来，且和算子自身缺陷难以区分。本次预检在 Ascend950PR + CANN 9.0.0-beta.2 + ops-test-kit MR#57 head（上游 master 尚未合入 ascend950 跳过 legacy 的修复）上执行：造 1 条最小 kernel 模式用例冒烟一次，再从结果 CSV 与日志取证分发路径。注意接口桩 kernel 无计算逻辑，**冒烟精度 FAIL 属预期，只判 tiling 分发阶段**。

结果：op info 命中 vendor 源、`impl_type = asc`（V2）、tiling 数据 `(8,)` 与 blockDim `1` 均只能出自 vendor 桩 TilingFunc、上板执行 OK、日志中 `OPTILING_FAILURE`/`TbeOpTilingPyInterface` 计数为 0——**tiling 分发正常，无需登记框架级 blocker**。

## 适用场景（必填 — 检索前的第一道滤网）

- **工程形态**：ops-registry-invoke 算子仓开发；vendor 包安装后、TTK 白盒/黑盒/GEIR 大批量用例执行前的一次性通路预检；也可用于疑似 V1 劫持 / `OPTILING_FAILURE` 框架级报错时的分诊取证。
- **API 路径**：`python3 -m ttk kernel`（kernel 模式，vendor 源）；`$ASCEND_OPP_PATH/vendors/<OpName>/op_impl/ai_core/tbe/`（vendor 包布局：`config/<soc>/aic-<soc>-ops-info.json`、`<OpName>_impl/dynamic/<op_file>.py`、`op_tiling/lib/.../libcust_opmaster_rt2.0.so`）。
- **验证过的平台与 CANN 版本**：Ascend 950 系列（Ascend950PR_9579 实测）+ CANN 9.0.0-beta.2 + ops-test-kit `1d175ef`（MR#57 head，动态 shape 编译路径）。
- **适用粒度（transfer_scope）**：`platform_wide`——判定方法对任意安装为 vendor 包的自定义算子通用（指纹取算子自身 TilingFunc 常量，天然逐算子适配）；本 trace 仅单算子、单平台、动态编译路径实测。
- **路线优先级**：`recommended_fastpath`（一次最小冒烟换取整条后续测试链路的分发确定性）。
- **明确不适用 / 未验证场景**：静态 shape 编译（`-c`）与二进制执行（`-b release`）路径未验证；非 vendor 安装（ASCEND_CUSTOM_OPP_PATH custom 源）未实测（机制上 custom 优先级更高，推断同理但未验证）；内置算子本身（无 vendor 包）不适用；aclnn 直调的运行时解析不经 TTK，本卡不覆盖。

## 根因（机制：为什么这套判定成立）

TTK 对算子的 tiling 分发解析由三层机制决定（源码均为 TTK 检出 `1d175ef` 的实测读取）：

1. **op info 来源优先级**：`OpInfoKeeper._load_op_info_cfg` 按 built-in → vendors（逆序）→ custom（逆序）收集 ops-info json，**后加载覆盖先加载（custom 优先级最高，vendor 覆盖 built-in）**。源码注释原文：`# Collect all op info config files: built-in → vendors(reversed) → custom(reversed)` / `# Later loading overrides earlier = custom highest priority`。所以只要 vendor 包安装到位，同名时 vendor 注册胜出。
2. **V1/V2 类型判据**：`get_operator_function` 以 `__import__("{vendor}_impl.dynamic.{op_file}")` 加载 vendor 实现 py，`info["_impl_type"] = "asc" if getattr(module, '__version__', '1.0.0') == '2.0.0' else "tbe"`——**vendor 实现 py 的 `__version__ == '2.0.0'` 即 V2（asc）**；注意 import 失败会静默回退 `tbe`（假信号——判读前提：探针必须在 CLI 等价环境（vendor impl base 的 PYTHONPATH + TTK_FULL_SOC_VERSION）下运行）。
3. **ascend950 的 legacy 陷阱（PR#57 的存在理由）**：ascend950 的 CANN built-in 配置目录里有一个 `aic-ascend950-ops-info-legacy.json`（V1/legacy 注册源）；未打 PR#57 补丁的上游 master 会把它排进加载列表**最前**（`sorted(paths, key=lambda x: 0 if 'legacy' in x.lower() else 1)`——legacy 得 key=0 排最前）。PR#57（`1d175ef`）的补丁在 ascend950 上跳过 legacy json：

```
+            skip_legacy = soc_lower == "ascend950"
             paths = [os.path.join(config_dir, f) for f in os.listdir(config_dir)
-                     if f.endswith(".json") and os.path.isfile(os.path.join(config_dir, f))]
+                     if f.endswith(".json") and os.path.isfile(os.path.join(config_dir, f))
+                     and not (skip_legacy and "legacy" in f.lower())]
```

   工作流担心的失败模式（本环境实测**未发生**，仅作为预检动机）：若 legacy json 里存在与自定义算子同名的 V1 注册，tiling 分发可能被劫持到 `TbeOpTilingPyInterfaceOld`（ascend950 上不可用）→ `OPTILING_FAILURE` 类框架级失败。

由此，"分发命中 vendor V2"的**可靠证据链 = ①无同名 V1 注册源（built-in 主配置 + ascend950 legacy json 都无该算子名）②`impl_type = asc` ③冒烟结果 CSV 的 `dyn_tiling_data`/`dyn_block_dim` 与**算子自身 TilingFunc 写入的常量**逐项一致（指纹判据）④日志中 `OPTILING_FAILURE`/`TbeOpTilingPyInterface` 计数为 0**。其中③是决定性证据：tiling 数据的数值只能由该算子自己的 TilingFunc 产出（本例 totalLength = 输入0 shape 元素数 = 2×4 = 8；blockDim = `SetBlockDim(1)` 的字面值 1），任何内置/legacy 同名算子都不可能凑出相同指纹。

## 判别方法（四步配方）

**第 1 步：排除同名 V1 注册源**（劫持可能性的静态排除）

```bash
grep -rn "ThnnFusedGruCellGrad\|thnn_fused_gru_cell_grad" $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ 2>/dev/null | head -5; echo "---search in whole opp---"; find $ASCEND_OPP_PATH -iname "*thnn_fused_gru*" 2>/dev/null | head -10
```

实测：built-in 主配置无命中；全 opp 搜出的 `*thnn_fused_gru*` 全部位于 `vendors/ThnnFusedGruCellGrad/`。再查 ascend950 legacy json（本环境实测 195 个 legacy 算子中无 thnn/gru 同名）：

```bash
python3 -c "
import json
d = json.load(open('/home/<user>/Ascend/cann-9.0.0-beta.2/opp/built-in/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info-legacy.json'))
print('legacy ops total:', len(d))
hits = [k for k in d if 'gru' in k.lower() or 'thnn' in k.lower()]
print('thnn/gru hits in legacy:', hits)
"
```

实测输出：`legacy ops total: 195` / `thnn/gru hits in legacy: []`。

**第 2 步：确认 vendor impl 的 V2 标记**

```bash
grep -n "__version__" $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/ThnnFusedGruCellGrad_impl/dynamic/thnn_fused_gru_cell_grad.py
```

实测输出：`21:__version__ = '2.0.0'` → `_impl_type = "asc"`（V2）。另可确认 vendor 包带 V2 tiling 库：`op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so`。

**第 3 步：造 1 条最小 kernel 模式用例并 CLI 冒烟**（完整 CSV 与命令见「完整代码」）：先 `--validate`，再 `cd $TTK` 后 `python3 -m ttk kernel -i <csv> --pc=1 --device-whitelist 0 --task-prof false --run 1 --seed 42 -o <out>/result.csv`，全量日志留存。

**第 4 步：证据判读**

- 结果 CSV：`dyn_tiling_data`、`dyn_block_dim`、`dyn_tiling_key` 与算子 TilingFunc 常量对照；`dyn_tiling_time_us` 非空（默认 `--tr 3`，3 次 tiling 执行完成）；`dyn_perf_us=OK`（上板执行）；`precision_status` 为 FAIL 不阻塞（桩预期）。
- 日志：`grep -c -i "OPTILING_FAILURE\|TbeOpTilingPyInterface" <log>` 计数为 0；TestcaseManager/OnDynCompilation 从 `vendors/<OpName>/.../aic-<soc>-ops-info.json` 加载 vendor op info 的 INFO 行在位。
- （补强，可选）CLI 等价环境探针 `impl_type_of(op)`：`asc`。裸探针会误报 `tbe`——必须带 vendor impl base 的 `PYTHONPATH` 与 `TTK_FULL_SOC_VERSION` 在 CLI 等价环境中运行。

## 设计决策

- **指纹判据的选择**：不依赖"冒烟没报错"这种弱证据，而是把 `dyn_tiling_data`/`dyn_block_dim` 与算子自身 TilingFunc 的确定性输出对齐——本例 totalLength = grad_hy(2,4) 的元素数 2×4 = 8（写入 `tiling->totalLength`），blockDim = `SetBlockDim(1)` 的字面值。这两个值是算子身份的指纹：内置/legacy 同名算子不可能产出相同值，故"指纹命中 + 同名源排除"双重锁死分发归属。
- **`has_bias: True`**：让 5 个输出全部非空（`has_bias=false` 时两个 bias 输出是 (0,) 空张量），避免空张量在框架侧引入与分发判定无关的变量。
- **`--task-prof false --run 1`**：功能冒烟纪律（不采集性能、单次执行）；性能采集是后续任务的独立步骤（`--task-prof true --run N≥3`），本卡不混入。
- **`--seed 42` + `--pc=1 --device-whitelist 0`**：单卡单进程、输入可复现，把日志噪音降到最小，便于逐行取证。
- **结论不依赖 tiling 库加载痕迹**：日志中 grep `opmaster|rt2.0` 无命中（如实记录）——"tiling 走编译产物库还是动态编译期源码态"无法从本日志区分，但指纹证据已足以判定"命中 vendor V2"；此差异列为遗留问题。
- **不登记 known_blockers.md**：登记条件是"命中 OPTILING_FAILURE/V1 劫持类框架级根因"；本次四项证据全过、未命中条件，结论为"分发正常"，以冒烟产物 + 开发日志为载体即可。
- **当前方案属哪档路线**：`recommended_fastpath`。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`）；TTK：ops-test-kit `/tmp/ops-test-kit` @ `1d175ef`（MR#57 head，branch pr57；上游 master `151886a` 未合入该 MR）。
- 设备：NPU 0 = Ascend950PR（完整 SoC 名 `Ascend950PR_9579`）。
- 执行环境：`source` 落盘的 env.sh（CANN set_env.sh + `PY=python3` + `TTK=/tmp/ops-test-kit` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`）；命令一律 `cd $TTK` 后执行（编译产物写 CWD 的 `./kernel_meta`）。
- 相关文件（与本次实测读写一一对应）：
  - `operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_kernel.csv`（新建，冒烟用例）
  - `operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/result_smoke.csv`、`smoke_run.log`（产物）
  - `operators/ThnnFusedGruCellGrad_package/op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（指纹来源：桩 TilingFunc）
  - `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/`（vendor 包：ops-info / impl py / op_tiling 库）
  - `$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info-legacy.json`（legacy 排查对象）
  - `/tmp/ops-test-kit/ttk/core_modules/operator/op_info_keeper.py`（分发解析源码）

## 完整代码

**指纹来源——桩 TilingFunc**（`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`，verbatim 全函数；冒烟判据就是它的两个确定性输出）：

```cpp
// 桩：1 核、无 workspace、仅写 totalLength 占位字段。
// ⚠️ 芯片平台信息（ubSize/coreNum 等）必须经 context->GetPlatformInfo() 获取，
//    不能依赖 GetCompileInfo<T>()：aclnn 通路不调用 TilingParse，恒返回 null。
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ThnnFusedGruCellGradTilingData* tiling = context->GetTilingData<ThnnFusedGruCellGradTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = 0;
    const auto* gradHyStorageShape = context->GetInputShape(0);
    if (gradHyStorageShape != nullptr) {
        totalLength = static_cast<uint32_t>(gradHyStorageShape->GetOriginShape().GetShapeSize());
    }
    tiling->totalLength = totalLength;

    context->SetBlockDim(1);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

**冒烟用例 CSV**（`tests/whitebox/ttk/smoke_kernel.csv`，verbatim 全文件；最小 shape：B=2、H=4 → grad_hy (2,4)、storage (2,20)，输出 (2,12)/(2,12)/(2,4)/(12,)/(12,)）：

```csv
testcase_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision
thnn_fused_gru_cell_grad_smoke_01,thnn_fused_gru_cell_grad,"((2, 4), (2, 20))","('float32',)","('ND',)","((2, 12), (2, 12), (2, 4), (12,), (12,))","('float32',)","('ND',)",{'has_bias': True},"((-1, 1), (-1, 1))",,0
```

**校验 + 冒烟命令**（verbatim；`--validate` 先行，冒烟全量日志落 `smoke_run.log`）：

```bash
# 校验 CSV（不执行编译/生成流程）
bash -c '
source /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/docs/ThnnFusedGruCellGrad/develop/env.sh
cd /tmp/ops-test-kit
export PYTHONPATH="/tmp/ops-test-kit:$ASCEND_TOOLKIT_HOME/python/site-packages"
python3 -m ttk kernel -i /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_kernel.csv --validate
' 2>&1 | tail -10

## 冒烟（kernel 模式、vendor 源、单卡单进程、不采性能、单次执行）
bash -c '
source /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/docs/ThnnFusedGruCellGrad/develop/env.sh
cd /tmp/ops-test-kit
export PYTHONPATH="/tmp/ops-test-kit:$ASCEND_TOOLKIT_HOME/python/site-packages"
WS=/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier
python3 -m ttk kernel -i $WS/operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_kernel.csv \
  --pc=1 --device-whitelist 0 --task-prof false --run 1 --seed 42 \
  -o $WS/operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/result_smoke.csv \
  > $WS/operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_run.log 2>&1
echo "TTK_EXIT=$?"
' ; tail -5 operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_run.log
```

**证据判读命令**（verbatim）：

```bash
grep -n -i "tiling" operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_run.log | head -30
cat operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/result_smoke.csv; echo "=== error/v1 markers ==="; grep -c -i "OPTILING_FAILURE\|TbeOpTilingPyInterface" operators/ThnnFusedGruCellGrad_package/tests/whitebox/ttk/smoke_run.log
```

**结果 CSV 关键字段**（`result_smoke.csv` 数据行 verbatim 节选——本卡结论所依据的全部字段；完整结果文件另含时延/编译耗时等列，其数值不作为本卡结论，仅存 `.build` 证据链）：

```
op_name                 = thnn_fused_gru_cell_grad
attributes              = {'has_bias': True}
input_shapes            = ((2, 4), (2, 20))
output_ori_shapes       = ((2, 12), (2, 12), (2, 4), (12,), (12,))
dyn_kernel_name         = dyn_op_thnn_fused_gru_cell_grad_smoke_01
dyn_tiling_time_us      = <非空三元组>        # 默认 --tr 3 次 tiling 执行完成（单跑时延数值按纪律不落卡正文，只存 .build 证据链）
dyn_block_dim           = 1                  # = TilingFunc 的 SetBlockDim(1)
dyn_tiling_key          = 0 (0x0000000000000000)
dyn_tiling_data         = (np.int32(8),)     # = totalLength：grad_hy(2,4) 元素数 2×4=8
dyn_perf_us             = OK                 # 上板执行 OK
dyn_oob_result          = OK,OK,OK,OK,OK,OK  # 无越界
memory_oob_status       = PASS
precision_status        = FAIL               # 桩 kernel 无计算，预期内，不阻塞
perf_status             = PASS               # 执行流通路 PASS
soc                     = Ascend950PR_9579
```

**日志关键行**（`smoke_run.log` verbatim 节选）：

```
66:Tiling Data Parsed Dict: (np.int32(8),)
67:Tiling Data Parsed Tuple(int32): (np.int32(8),)
68:Tiling Key: 0 (0x0000000000000000)
```

以及 TestcaseManager / OnDynCompilation 两个阶段均出现 `Loading vendor op info from: /home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json` 的 INFO 行；`grep -c "OPTILING_FAILURE\|TbeOpTilingPyInterface"` 计数为 0。

## 性能影响

本卡不涉及性能结论：冒烟按功能纪律执行（`--task-prof false --run 1`），未采集性能数据，也未验证精度影响（桩 kernel 无计算，精度 FAIL 属预期、不作为判据）。

## 预防

- vendor 包新装/重装后、大批量 TTK 用例开跑前，先做本四步预检（成本 = 1 条最小用例 + 一次冒烟），避免把框架级分发问题混进算子缺陷排查。
- 预检结论落盘：冒烟三件套（用例 CSV / 结果 CSV / 全量日志）留存于算子工程 `tests/whitebox/ttk/`，供后续 run/fix 任务对账。
- ascend950 环境注意 TTK 检出的版本：上游 master 未含"跳过 legacy"修复时，legacy json 会参与 built-in 加载且排在最前——预检第 1 步（同名排除）必须**连 legacy json 一起查**，只查主配置会漏。

## 验证信号

本次开发会话（自定义算子接口桩工程的 TTK 环境预检）实测：

- CSV 校验：`Validate Complete: 1 total, 1 valid, 0 invalid`（来源平台：950；证据等级：tool_observed）。
- 冒烟执行：`TTK_EXIT=0`、`STATUS: PASS`（执行流）、`PRECISION_STATUS: FAIL`（桩 kernel 无计算的预期结果，非缺陷）（来源平台：950；证据等级：tool_observed）。
- tiling 指纹：`dyn_tiling_data = (np.int32(8),)`（= totalLength，grad_hy(2,4) 元素数 2×4=8）、`dyn_block_dim = 1`（= SetBlockDim(1)）、`dyn_tiling_key = 0`、日志 `Tiling Data Parsed Dict: (np.int32(8),)`（来源平台：950；证据等级：tool_observed）。
- tiling 实际执行：`dyn_tiling_time_us` 为非空三元组（默认 `--tr 3` 次 tiling 执行完成）（来源平台：950；证据等级：tool_observed）。
- V1 标记零计数：`grep -c -i "OPTILING_FAILURE\|TbeOpTilingPyInterface" smoke_run.log` = 0（来源平台：950；证据等级：tool_observed）。
- 同名源排除：built-in 主配置 grep 无命中、全 opp `find *thnn_fused_gru*` 仅命中 vendors 目录、ascend950 legacy json（195 个 legacy 算子）无 thnn/gru 同名（来源平台：950；证据等级：tool_observed）。
- vendor 源与 V2 标记：vendor ops-info json 注册本算子（日志 INFO 行）、vendor impl py `__version__ = '2.0.0'`、CLI 等价环境探针 `impl_type = asc` / `operator_func = thnn_fused_gru_cell_grad`（来源平台：950；证据等级：tool_observed）。
- 上板执行：`dyn_perf_us = OK`、`dyn_oob_result` 全 OK、`memory_oob_status = PASS`、结果 CSV `soc = Ascend950PR_9579`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在动态 shape 编译路径（kernel 模式默认）实测；静态编译（`-c`）与二进制执行（`-b release`）的分发行为未验证（推断同理由同一 ops-info/impl_type 机制决定，未验证）。
- 仅 ascend950（Ascend950PR_9579）+ CANN 9.0.0-beta.2 + TTK `1d175ef` 实测；其他 SoC / CANN / TTK 版本未验证，推断可迁移。legacy json 仅在 ascend950 的 built-in 配置目录观察到；PR#57 合入上游后"跳过 legacy"行为成为默认，第 1 步的 legacy 排查在新版本上可能不再必要（需复核）。
- "同名 V1 劫持"分支未在本环境发生（legacy json 无同名算子）——本卡验证的是"无劫持时的确认方法"；真的存在同名 legacy 注册时的行为（覆盖顺序、报错形态）未实测。
- 精度路径不适用：桩 kernel 无计算，`precision_status=FAIL` 在本场景是预期值，不能外推为"精度 FAIL 总是可忽略"。
- 探针取 `impl_type` 必须在 CLI 等价环境（vendor impl base 进 `PYTHONPATH` + `TTK_FULL_SOC_VERSION`），否则得到假 `tbe`（须在 CLI 等价环境中运行）。

## 坏实践

- 不做同名源排除（尤其漏查 ascend950 legacy json）就下"分发正常"结论——同名 V1 注册源的存在会推翻整个判断。
- 把桩 kernel 的预期 `precision_status=FAIL` 当算子缺陷去修——接口桩阶段它只证明"kernel 没写计算逻辑"，与分发无关。
- 只看"冒烟退出码 0"就结论分发正常——退出码只证明流程走完，不证明 tiling 走的是 vendor V2；必须对上 tiling 数据/blockDim 指纹。
- 用裸探针的 `impl_type = tbe` 否定冒烟结论（假信号）。

## 误诊

- **误诊（本 trace 排除过）**：独立 python 探针显示 `impl_type = tbe` → 疑似"V1 通路/被劫持"。正确解读：探针缺 vendor impl 路径导致 import 失败被吞、类型回退默认值（判别锚点：输出中的 `No module named '{Op}_impl'` ERROR 行）；CLI 等价环境复测为 `asc`。
- **误诊（流程性）**：把 `PRECISION_STATUS: FAIL` 读作"环境没配好/算子坏了"。任务口径明确：接口桩冒烟只判 tiling 分发阶段根因，精度/运行失败不构成阻塞。


