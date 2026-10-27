---
type: "Runbook"
title: "TTK 缓存失败编译结果，修复后不重试"
description: TTK 以 .ttk 文件缓存动态编译失败结果（kernel_name/tiling_op_type = DYN_OPERATOR_BUILD_LOST）并直接返回不重编译；修复源头后重跑命中缓存照样报一模一样的失败。排障第一步先清 kernel_meta/*.ttk（连同 dyn_op_* 产物）。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:32:55Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# TTK 缓存失败编译结果，修复后不重试

## 现象

TTK 白盒动态编译失败后（DYN_OPERATOR_BUILD_LOST），即使已经修复了源头问题（如补装缺失头文件），重跑同一用例仍然报一模一样的失败，且没有任何重新编译的迹象（kernel_meta 里没有新产物、编译日志无更新、重跑耗时异常短）。查看 `<TTK根>/kernel_meta/` 发现一个以动态特征码命名的 .ttk 缓存文件：

```
=== <TTK根>/kernel_meta/<dynamic_characteristic_code>.ttk ===
{
    "func_params": null,
    "kernel_name": "DYN_OPERATOR_BUILD_LOST",
    "compile_info": {},
    "tiling_op_type": "DYN_OPERATOR_BUILD_LOST"
}
```

删除 *.ttk（连同 dyn_op_* 产物）后重跑，编译真实重新执行，失败信息变为真实根因。

## 适用场景

- 工程形态：TTK（ops-test-kit）白盒/kernel 模式动态编译排障。
- API 路径：`ttk/core_modules/npu/op/compilation/dynamic_compilation.py`（.ttk 缓存按 dynamic_characteristic_code 命名读写）。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.2.0。
- 适用粒度：platform_wide（TTK 工具行为，与算子无关）。
- 路线优先级：recommended_fastpath（清缓存是动态编译排障第一步）。
- 明确不适用 / 未验证场景：静态编译缓存、ccec 编译缓存未验证；成功编译的 .ttk 缓存是正常加速机制，不需要清。

## 根因

TTK 按动态特征码缓存编译结果，**失败结果同样入缓存**并立即返回，不触发重编译；修复源后必须清掉对应 .ttk 才能验证修复是否生效。

## 判别方法

1. 触发：动态编译失败（BUILD_LOST / build artifacts not found）在修复源后重跑**逐字不变**，且无编译动作迹象。
2. 探针（可执行）：
   ```bash
   ls <TTK根>/kernel_meta/*.ttk
   for f in <TTK根>/kernel_meta/*.ttk; do echo "=== $f ==="; cat "$f"; done
   ```
3. 原始观测：某个 .ttk 的 JSON 中 kernel_name 与 tiling_op_type 均为 `"DYN_OPERATOR_BUILD_LOST"`。
4. 解释：见根因。

处置命令：

```bash
rm -f <TTK根>/kernel_meta/*.ttk
rm -f <TTK根>/kernel_meta/dyn_op_*
```

与相邻卡的分界：DYN 失败若表现为 `fatal error: ... file not found`（日志里有编译器报错），是部署树缺头文件一类问题（见 [vendor 缺 op_host 头致 DYN 重编译失败卡](vendor_dyn_recompile_missing_op_host_tiling_data_h.md)）；本卡覆盖的是"日志无 fatal error、失败被缓存原样返回"的形态。结构体布局变更引发的 kernel_meta 陈旧（单例过/全量挂分裂）另见 [TilingData 改布局须重编 kernel 并清 kernel_meta 卡](tilingdata_layout_change_rebuild_kernel_meta.md)。

## 验证信号

问题与验证来源：一次白盒修复会话中的缓存失败命中与清除验证（Ascend 950 + CANN 9.2.0）：

- 清缓存前：.ttk 内容为 kernel_name/tiling_op_type = DYN_OPERATOR_BUILD_LOST（来源平台：950；证据等级：tool_observed）
- 清缓存后重跑：编译真实执行（ASCENDC_BUILD_LOG_DIR 日志出现 bisheng 编译命令与链接），失败信息暴露为真实根因（来源平台：950；证据等级：tool_observed）

## 失效边界

- 只验证了动态编译**失败**结果的缓存行为；.ttk 文件名是动态特征码哈希，无法从文件名判断对应哪个用例，排障时全量清即可。
- TTK 版本更新后缓存策略可能变化。

**未决项**：

- 无（本规则已闭环）。
## 预防

- TTK 动态编译排障 SOP 第一步：清 kernel_meta/*.ttk，再做其他排查，避免被缓存失败误导得出"修复无效"的错误结论。

## 坏实践

- 修复源后不清缓存直接重跑，据"失败不变"推断修复无效或编译器有 bug。

## 误诊

- 曾据"失败逐字不变 + 无编译日志"怀疑 TBE 编译器根本没被调用/进程异常——实际是缓存层命中；清缓存后即证伪。
