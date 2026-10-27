---
type: "Runbook"
title: 清 TBE 编译缓存不修复 blockDim 固化
description: GE 图路径 kernel 因 TBE 动态编译 blockDim 固化为平台 AIV 核数而 EZ9999 失败时，清除 ~/atc_data/kernel_cache 的 te_* 缓存无效——fresh TBE 编译立即重新生成同样的 blockDim=平台核数产物；blockDim 来自编译器平台默认而非陈旧缓存，且 TilingParse 的 coreNum 因 compile_info 缺失进不了 TBE 编译。
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# 清 TBE 编译缓存不修复 blockDim 固化

## 现象

在 Ascend950PR（CANN 9.2.0）上排查一个自定义 scatter 类算子 GEIR 图模式的 kernel 执行失败（RunGraph 返回 4294967295，设备 EZ9999 error 95 `The DDR address of the MTE instruction is out of range`，`StreamLaunchKernelV2 ... coreDim=6`；判别见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）时，日志里出现：

```
[DEBUG] TEFUSION ... MatchCompileCache Start to match cache of kernel[te_benchharnessscatter_911e...] in cache dir[/home/<user>/atc_data/kernel_cache/Ascend950PR_9579].
[DEBUG] TEFUSION ... GetJsonValueFromJsonFile ... jsonValue: [{"binFileName":"te_benchharnessscatter_...","blockDim":6,"coreType":"MIX","intercoreSync":1,...,"taskRation":"0:1"}].
```

TBE 动态编译缓存 json 里明确写着 `"blockDim": 6`，而算子的 TilingParse 在 op_host 侧填的是 `coreNum=1`。第一反应是"这个 blockDim=6 是以前某次编译留下的陈旧缓存，清掉重编译就好"——于是删除了 `~/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter*` 全部缓存（.json/.o/.lock 多组文件，删除后 `Remaining files: 0`），重跑同一最小 route 用例。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：GE 图模式/TBE 动态编译联调中，怀疑陈旧 kernel 缓存导致 blockDim 或行为异常时的处置决策。
- API 路径：TBE 动态编译缓存 `~/atc_data/kernel_cache/<soc>/te_<op>__<hash>.{json,o,lock}`；op_host TilingParse（CompileInfo coreNum）与 TBE 动态实现 `<Op>_impl/dynamic/<apt>.py`。
- 验证过的平台与 CANN 版本：Ascend950PR（与 frontmatter `platforms` 一致），CANN 9.2.0。
- 适用粒度：`platform_wide`——结论关于 TBE 动态编译器的 blockDim 来源（平台默认核数，非缓存残留），与算子无关。
- 路线优先级：`banned`——作为 EZ9999/blockDim 类图路径故障的修复手段无效且浪费一轮全量重编译；仅可作为排除法的一步。
- 明确不适用 / 未验证场景：缓存确实陈旧的场景（源码变更后命中旧 hash、vendor 包更新后残留旧产物）不在本卡否定范围——本卡只否定"blockDim=平台核数来自缓存残留"这一假设。

## 尝试与观测（负结果）

**尝试（attempt，verbatim 命令）**：

```bash
# 清除该算子的全部 TBE 动态编译缓存
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter*
echo "Cleared TBE cache. Remaining files:"
ls /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter* 2>/dev/null | wc -l
# 输出：Cleared TBE cache. Remaining files: 0
```

随后重跑同一最小 route 用例（`cd tests/geir && ./run.sh route`）。

**原始观测（observation）**：

- `ROUTE_STATUS=1`（失败不变）；
- `StreamLaunchKernelV2` 仍是同一 `te_benchharnessscatter_911e...` kernel、`coreDim=6`；`GRAPH-RUN ... rungraph_rc=4294967295 ... value=FAIL` 逐字不变；
- fresh 编译重新生成了同名缓存文件，blockDim 仍为 6：

```
=== Check new cache file blockDim ===
--- /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_911e...json ---
blockDim: 6
```

- 旁证（解释为何重编译也不会变）：TBE 动态实现 `bench_harness_scatter_apt.py` 中 `grep -n "block_dim\|blockDim\|core_num\|coreNum\|get_core_num\|getCoreNum\|compile_info\|coreNum"` 无任何命中；`aic-ascend950-ops-info.json` 无 compile_info 字段；运行日志 `SetNodeCompileInfoAttr Op[name=n_route_target]: compile_info_key is empty, returning.`——op_host TilingParse 填的 `coreNum=1` 没有任何通路进入 TBE 编译。

**边界（boundary）**：

- `blockDim=6` 不是陈旧缓存，而是每次 TBE 动态编译都用平台默认 AIV 核数固化产出：清缓存后 fresh 编译立即重建同样的 `blockDim=6` 产物；
- "清缓存让 TBE 重新读 coreNum"这条路走不通：coreNum 经 CompileInfo 传递，而 compile_info 未落盘到 ops-info.json、TBE 动态实现 python 也不读它；
- 本结论只覆盖"blockDim 来自 TBE 编译器平台默认"的情形；若失败确实由源码/vendor 包更新后的旧缓存残留引起，清缓存仍是对的第一步——用"删除后重跑、对比 fresh 产物 json 的 blockDim"即可区分两种情形（本探针本身就是区分手段）。

## 根因

blockDim 在 TBE 动态编译期由编译器按平台默认核数固化进产物 json（与缓存无关）；重编译不改变编译参数来源，清除缓存只强制重建同样的产物。TilingParse 的 coreNum 经 CompileInfo 传递，而 compile_info 链路在该路径断裂（`compile_info_key is empty`），故"重编译时读 coreNum"的期望不成立。

## 设计决策

- 把"清缓存"作为单独的否定性结论记录，而不是埋进主诊断卡的误诊：清缓存是 GE/TBE 联调中最高频的第一反应动作，独立的 negative_result 能让下个 agent 直接跳过这一步（或只用它做一次排除法对照）。
- 判别设计为"删除 → 重跑 → 对比 fresh json 的 blockDim"三步：若 fresh 产物与旧缓存逐字段一致，即可证明值来自编译器默认而非历史残留，一次对照同时完成排除与定界。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；GE `ge::Session` 图模式；TBE 动态编译。
- 相关文件：`~/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*`（TBE 缓存）、`vendors/<Op>/op_impl/ai_core/tbe/<Op>_impl/dynamic/bench_harness_scatter_apt.py`（TBE 动态实现）、`vendors/<Op>/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json`、route 原始 plog——与本会话实际读取的文件一一对应。

## 预防

- 遇到 GE 图路径 kernel 异常先判二进制来源（kernel 名 `te_*` 前缀 + `ParseTvmBlockDim` 值），再决定是否值得清缓存：`te_*` + blockDim=平台核数时清缓存注定无效。
- 若确要验证缓存假设，按本卡三步探针做一次对照即可，不要反复清缓存重跑全量。

## 验证信号

来源说明：以下信号来自该算子 GEIR 图模式实现验收会话（Ascend950PR，CANN 9.2.0）清缓存前后的同一最小 route 用例对照；同组其他会话独立重做清缓存对照，结果一致。

- 清缓存前：TBE 缓存 json `blockDim: 6`（`coreType: MIX`、`taskRation: 0:1`），route GRAPH-RUN `rungraph_rc=4294967295` value=FAIL（来源平台：950；证据等级：tool_observed）。
- 清缓存后重跑：`ROUTE_STATUS=1`；fresh 生成的 `te_benchharnessscatter_911e...json` 仍 `blockDim: 6`；`StreamLaunchKernelV2` 仍 `coreDim=6`；GRAPH-RUN 仍 `rungraph_rc=4294967295` value=FAIL（来源平台：950；证据等级：tool_observed）。
- 机制旁证：TBE 动态实现 python 中 block_dim/blockDim/core_num/coreNum/compile_info 相关 grep 无输出；`aic-ascend950-ops-info.json` 无 compile_info 字段；运行日志 `compile_info_key is empty`（来源平台：950；证据等级：tool_observed）。
- 跨会话复现：另一验收会话删除 `te_benchharnessscatter_*` 后强制重编译，kernel sha256 变化（`821f8388...`→`0ecaeef4...`，证明确实是新二进制），`ParseTvmBlockDim: 6.` 与 EZ9999 逐字不变（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论在 CANN 9.2.0 + Ascend950PR（AIV=6）实测；其他平台核数不同但机制推断相同（未验证，推断可迁移）。
- 只否定"清缓存可修复 blockDim 固化类失败"；对 hash 碰撞、vendor 包残留等真实陈旧缓存场景未测试，不构成否定。
- 本卡不含修复：blockDim 固化的根因与修复方向见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡。

**未决项**：

- 无独立遗留项（根因归属与接续点见同批判别卡的遗留问题，其"追 binary matching/compile_info 机制层根因"优先项同样适用）。
## 坏实践

- 把"清 kernel 缓存重跑"当作 EZ9999/blockDim 类图路径故障的修复手段直接交付——本例证明无效，且全量重编译浪费时间；它只能作为排除法的一步并留下对照证据。
- 反复清 kernel cache 治"kernel 不对"：cache 只存编译产物，不决定编译参数（blockDim 在编译期烘焙，重编译结果相同）。

## 误诊

- "缓存是陈旧的，清掉就好"：fresh 编译产物与缓存逐字段一致（blockDim=6），证明 6 来自编译器默认而非历史残留。
