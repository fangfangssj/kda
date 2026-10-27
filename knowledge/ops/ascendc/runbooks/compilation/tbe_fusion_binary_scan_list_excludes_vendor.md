---
type: "Runbook"
title: TBE fusion 不扫自定义 vendor：三种修复均无效
description: 清 TBE kernel cache、卸载重装 vendor 到 CANN OPP、手改 binary_info_config.json 的 simplifiedKeyMode=1，均无法让 GEIR 图模式命中预编译 binary——TBE fusion 的 binary 配置扫描清单（SetBinaryConfigPaths）不含该算子 vendor，配置修改根本不被加载。
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# TBE fusion 不扫自定义 vendor：三种修复均无效

## 现象

接自定义 scatter 类算子的 GEIR 图模式验收场景：route kernel 崩溃（EZ9999，根因为 binary matching 未命中后 TBE 动态编译把 blockDim 烘焙为 6，判别见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）。为让图编译命中 op_build 预编译 binary（blockDim=-1，SetBlockDim 运行时生效）从而绕开动态编译，在同一环境（Ascend950PR_9579 + CANN 9.2.0，staging vendor 经 `ASCEND_CUSTOM_OPP_PATH` 注入）依次做了三个修复尝试，每次后重跑同一 route 验证。

尝试 1 — 清 TBE kernel cache 与本地 kernel_meta（排除陈旧编译产物）：

```bash
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter*
rm -rf <pkg>/tests/geir/kernel_meta/
```

尝试 2 — 卸载旧 vendor 并用当前源码重装到 CANN OPP（排除新旧包混用/旧配置抢先）：

```bash
bash <ASCEND_HOME>/opp/vendors/BenchHarnessScatter/scripts/uninstall.sh
# 重新 build 生成 custom_opp_ubuntu_x86_64.run 并安装到 CANN OPP 目录（Install exit: 0）
```

尝试 3 — 手改 installed + staging 两份 binary_info_config.json 的 simplifiedKeyMode 0→1（启用 GenSimplifiedKey 回调做 binary matching）：

```bash
python3 -c "
import json
f='<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json'
d=json.load(open(f))
d['BenchHarnessScatter']['simplifiedKeyMode']=1
json.dump(d, open(f,'w'), indent=2)"
```

三次重跑结果逐位相同：

```
ROUTE op=BenchHarnessScatter path=V2 status=NOT_UNIQUE probe_exit=1
GRAPH-RUN ... rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL
StreamLaunchKernelV2: kernel_name=te_benchharnessscatter_911e..., coreDim=6
SetNodeCompileInfoAttr Op[name=n_route_target]: compile_info_key is empty, returning.
```

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：GEIR/GE 图模式验收（staging vendor 经 `ASCEND_CUSTOM_OPP_PATH` 注入）下排查"预编译 binary 不被使用、总走 `te_` 动态编译"。
- API 路径：TBE fusion 的 binary 配置加载链（`SetBinaryConfigPaths` / `ReadOpBinaryInfoConfig` / fusion_binary_info）+ `binary_info_config.json`（simplifiedKeyMode/binaryList）+ `~/atc_data/kernel_cache`。
- 验证过的平台与 CANN 版本：Ascend950PR_9579（950 系列）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——同一 CANN 图编译/binary 加载机制下的自定义算子。
- 路线优先级：`banned`——三个修复路线在该场景实测无效，不要重复投入；它们不是替代修复，只是排除项。
- 明确不适用 / 未验证场景：vendor 已正式进入扫描清单的安装形态未验证；其他 CANN 版本未验证；本卡不宣称"binary matching 在 CANN 上不可能"，只实测这三条路线无效。缓存确实陈旧的真实场景（源码变更后命中旧 hash、vendor 包更新残留旧产物）不在本卡否定范围。

## 尝试与观测（负结果）

三次尝试的完整命令见「现象」段；客观观测：

1. 清 cache + 清 kernel_meta 后强制重编译：新 `te_` kernel json 仍 `"blockDim":6`，route `value=FAIL`。
2. 卸载旧 vendor + 重装到 CANN OPP（Install exit: 0）后：route `value=FAIL`、`coreDim=6`、compile_info 仍空。
3. simplifiedKeyMode=1（installed + staging 同改，python 校验回读为 1）后：route `value=FAIL`、`coreDim=6`、日志无 GenSimplifiedKey 调用记录。

## 根因

单句归因：TBE fusion 的 binary 配置扫描清单不含该算子 vendor，binary matching 从未被尝试，因此"清缓存 / 换安装位置 / 改 simplifiedKeyMode"都不改变结果。

证据（route 全量日志逐项核对）：

1. TBE fusion 实际注册的 binary 配置路径清单（`SetBinaryConfigPaths`）只含固定 vendor 与 built-in，不含本算子：

```
SetBinaryConfigPaths ... /opp/vendors/customize/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths ... /opp/vendors/ScaleCustom/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths ... /opp/vendors/custom_ops/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths ... /opp/vendors/EuclideanNorm/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths ... /opp/vendors/AddCustom/op_impl/ai_core/tbe/kernel/.
SetBinaryConfigPaths ... /opp/vendors/mla_prolog_custom/op_impl/ai_core/tbe/kernel/.
```

`ReadOpBinaryInfoConfig` 实际读取的 binary_info_config.json 清单（built-in 的 ops_legacy/ops_cv/ops_nn/ops_math/ops_transformer + 上述 vendor）同样不含本算子。

2. 该算子的 binary_info_config.json 实际存在（staging 与 CANN OPP 安装目录各一份；安装目录文件 ls 可见、7632 字节），且其中预编译条目 `blockDim=-1` 可用——文件在，但不被读。
3. 环境变量 `ASCEND_CUSTOM_OPP_PATH` 已指向 staging vendor 且被 GE 侧识别（`GetAllCustomOpApiSoPaths` / `GetPluginPathFromCustomOppPath` 日志确认 op_proto SO 从 staging 加载）——GE 侧认这个路径，TBE fusion 的 binary 配置侧不认。
4. 因此 simplifiedKeyMode 手改无效：该字段所在的 json 根本未被加载；GenSimplifiedKey 回调（运行时注册成功）也无从被调用。
5. 伴生事实：用含 GenSimplifiedKey 注册的源码重建 staging，op_build 产出的 binary_info_config.json 仍是 `simplifiedKeyMode: 0`（构建系统未因回调注册翻转该字段——机制推断）。

## 设计决策

- 尝试顺序按"最便宜先试"设计：cache 清理（无副作用）→ vendor 卸载重装（中等成本）→ 手改配置 json（诊断性对照实验：若 skm 是关键开关，改完应立即改变行为；结果不变即证明该配置未被消费）。
- 该三路线标 banned 的含义：继续叠加同类配置修改（重装到不同路径、清更多缓存目录、改 json 其他字段）无预期收益；正确方向是把 vendor 送进扫描清单或改用不依赖 binary matching 的验证路径。
- 与其他会话观测的关系：后续会话在重装后的环境里观测到 `ReadOpBinaryInfoConfig ... success`（installed vendor 路径读取成功），与本卡的"清单不含 vendor"矛盾——清单的建立可能与安装状态/注册步骤有关，未定案（见遗留问题）。

## 系统信息

- Ascend950PR_9579 + CANN 9.2.0（cann-9.2.0）；staging：`/tmp/geir_opp.*/vendors/<Op>`（`run.sh stage` 产物）；安装目录：`<ASCEND_HOME>/opp/vendors/<Op>`；TBE cache：`/home/<user>/atc_data/kernel_cache/Ascend950PR_9579/`。
- 相关文件：`binary_info_config.json`（staging/installed 两份，含 10 条预编译 binary 键值）、`tests/geir/run.sh`（`GEIR_VENDOR_DIR`/`ASCEND_CUSTOM_OPP_PATH` 注入）、route 全量日志（`/tmp/geir-route-*.log`）、vendor 卸载/安装脚本（`scripts/uninstall.sh`、`build/custom_opp_ubuntu_x86_64.run`）。

## 预防

- 图模式验收若走 staging vendor（`ASCEND_CUSTOM_OPP_PATH`），先 grep route 日志 `SetBinaryConfigPaths`：目标 vendor 不在清单 → 预编译 binary 必不生效，直接进框架层排查，不要在缓存/重装上花时间。

## 验证信号

来源说明：同一次 GEIR 图模式验收会话中对"binary matching 为什么不命中"的三轮修复-重跑对照，每轮重跑同一 route 验证二进制并核对全量日志。

- 清 cache + 清 kernel_meta 后强制重编译：新 `te_` kernel json 仍 `"blockDim":6`，route `value=FAIL`（来源平台：950；证据等级：tool_observed）。
- 卸载旧 vendor + 重装到 CANN OPP（Install exit: 0）后：route `value=FAIL`、`coreDim=6`、compile_info 仍空（来源平台：950；证据等级：tool_observed）。
- simplifiedKeyMode=1（installed + staging 同改，python 校验回读为 1）后：route `value=FAIL`、`coreDim=6`、日志无 GenSimplifiedKey 调用记录（来源平台：950；证据等级：tool_observed）。
- `SetBinaryConfigPaths` / `ReadOpBinaryInfoConfig` 清单不含本算子 vendor，而其 binary_info_config.json 在 staging 与安装目录均存在、预编译条目 blockDim=-1（来源平台：950；证据等级：tool_observed）。
- 用含 GenSimplifiedKey 注册的当前源码重建 staging 后，binary_info_config.json 仍 `simplifiedKeyMode: 0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 负结论限定：staging vendor（`ASCEND_CUSTOM_OPP_PATH` 注入）+ `jit_compile=2` 场景实测；"安装到 CANN OPP 后仍未进入扫描清单"也已实测，但清单的建立机制（何时/由谁注册 vendor 路径）未追明——存在某种安装后注册/索引步骤未做的可能，此可能性未排除。
- 其他会话在同版本观测到 installed vendor 的 binary_info 读取成功（Read ... success），与本卡清单观测矛盾：两轮观测的安装状态不同，不能合并成单一机制结论；本卡只主张"三条修复路线在该场景无效"。
- 未验证，推断可迁移：其他 CANN 版本的 TBE fusion vendor 扫描行为。

**未决项**：

- （优先项）`SetBinaryConfigPaths` vendor 清单的建立机制：是安装期注册（某 index/缓存文件）、还是编译期枚举 OPP 目录？为何安装到 `<ASCEND_HOME>/opp/vendors/` 后仍不入列？需在 CANN 侧（binary_manager / vendor 安装脚本）追到机制层。
- 找到让 staging vendor 进入 binary 扫描清单的官方途径（或让图模式直接用 ACLNN 预编译 binary 的配置）后，本卡结论应更新。
## 坏实践

- 手改 vendor 安装目录里的 binary_info_config.json 期望改变运行时行为：该文件是否被加载先于其内容生效；本例中根本未被读取。
- 反复清 kernel cache 治"kernel 不对"：cache 只存编译产物，不决定编译参数（blockDim 在编译期烘焙，重编译结果相同）。

## 误诊

- 早期假设"旧 vendor 抢先 / 新旧混用导致读了旧配置"：卸载重装后行为不变，且日志显示根本没有任何一份本算子 binary_info_config 被读取（agent_reported 假设 → tool_observed 排除）。
- "simplifiedKeyMode=0 阻止 GenSimplifiedKey 被调用是根因"（前次验收报告的表述）：0→1 后仍不调用——字段未被消费，前次表述把表象当成了根因（agent_reported 旧叙述 → tool_observed 修正）。
