---
type: "Runbook"
title: "unroll s2 内层循环破坏 flash 在线 softmax 循环态"
description: "对 flash attention 主循环 FLASH_LOOP_L4_s2_SA 把 unroll_list 由 {1} 扩为 {1,2} 会破坏 is_loop_begin/end 的循环态边界，使在线 softmax 的 li/mi 状态在 s2 步间错位、atten_out 大面积精度回归（8306526 误差点/阈值 41943，max_diff 0.066）。该展开路线 banned，性能优化改走 pass_option/TileShape。"
tags: ["task.optimization", "topic.compute.normalization", "topic.implementation.pipeline", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T08:42:48Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["sparse_flash_attention_quant_d_950", "online_softmax"]
---

# unroll s2 内层循环破坏 flash 在线 softmax 循环态

## 现象

在 Ascend950（CANN 9.1.0）上对 `sparse_flash_attention_quant` 的 `_d_950` 变体做 AICore 调优。基线配置精度通过（28 个误差点，阈值 41943，BF16 ULP 级）。为尝试提升 s2 维度的并行度，把 flash attention 主循环 `FLASH_LOOP_L4_s2_SA` 的 `unroll_list` 由 `{1}` 改为 `{1,2}`（即把 s2 的第 2 层循环也纳入展开）。改后单测 `compare` 立即失败：误差点从 28 暴涨到 8306526（远超阈值 41943），max_diff 0.06591797（阈值仅 0.00014496），量级高出约 454×，属于大面积数值回归而非尾块/边界抖动。随即回退 `unroll_list` 到 `{1}`，精度恢复通过。

## 适用场景

- **工程形态**：kernel 直调（pypto.frontend.jit DSL 编写的算子 kernel 实现，经 PyPTO 前端编译为 `_d_950` kernel 后 JIT 调用）。
- **API 路径**：`@pypto.frontend.jit` 装饰器内 `pypto.scoped_range`/循环原语 `FLASH_LOOP_L4_s2_SA`（`idx_name="s2_idx"`，`unroll_list` 参数）；配套 `pypto.is_loop_begin` / `pypto.is_loop_end` 循环态判定。
- **验证过的平台与 CANN 版本**：Ascend950（CANN 9.1.0）。frontmatter `platforms: [950]` 与此一致。
- **适用粒度**：`operator_family`（适用于采用 online-softmax + 循环态更新的 flash attention 类算子，非仅本变体）。
- **路线优先级**：`banned`——对承载在线 softmax 循环态的内层循环做 `unroll_list` 扩展属禁止操作；性能优化应改走 TileShape/pass_option（vec_nbuffer_setting、cube_l1_reuse_setting 等）方向。
- **不适用或未验证场景**：不携带跨迭代状态（li/mi）的纯逐元素循环不受此约束；a2/a3 未验证。

## 失败路线

尝试：为提升 s2 维度并行度，把 `FLASH_LOOP_L4_s2_SA` 的 `unroll_list` 由 `{1}` 扩为 `{1,2}`，使框架对 s2 第 2 层循环做展开。期望增加 s2 维度并行度以提升 AICore 利用率。

banned 路线代码（`FLASH_LOOP_L4_s2_SA` 循环头，`unroll_list={1,2}`，verbatim 取自该 edit 事件的 oldString）：

```python
                        name="FLASH_LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1, 2}):
```

参照路线（回退后的安全配置，`unroll_list={1}`，verbatim 取自该 edit 事件的 newString）：

```python
                        name="FLASH_LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
```

> 注：上述 before/after 为该 `edit` 事件逐字恢复的完整 oldString/newString（单行属性变更，无省略）。循环体内 `li_update[:] = li_new` / `mi_update[:] = mi_new` 的循环态写回逻辑未改动，回归仅由 `unroll_list` 一项引起。

客观失败（来源平台：950；证据等级：tool_observed）：

- 回归态（`unroll_list={1,2}`）：`compare fail: atten_out, max diff: 0.06591797 at (46, 0, 103, 25), error_count: 8306526, error_count_threshold: 41943`，EXIT_CODE=1（误差点从 28 暴涨到 8306526，max_diff 量级高出阈值约 454×）。
- 回退态（`unroll_list={1}`）：atten_out 28 个误差点（阈值 41943），精度通过，EXIT_CODE=0。

边界与回退：`unroll_list` 扩展承载循环态的内层循环 banned；回退到 `unroll_list={1}` 后精度恢复通过。性能优化改走 pass_option/TileShape 旋钮，不动循环态边界。

根因（机制链，属 interpretation/根因分析，非验证信号）：`FLASH_LOOP_L4_s2_SA` 的 s2 内层循环承载 flash attention 在线 softmax 的循环态——每步 s2 用 `is_loop_begin` / `is_loop_end` 判定是否首个/末个 s2 子步，据此决定初始化还是增量更新 `li`（log-sum-exp 的 max）/`mi`（缩放因子），并经 `li_update[:] = li_new` / `mi_update[:] = mi_new` 写回。把 `unroll_list` 从 `{1}` 扩为 `{1,2}` 后，框架对 s2 第 2 层循环做展开，破坏了 `is_loop_begin` / `is_loop_end` 在原单层 s2 上的边界语义——循环态的"初始化 vs 增量更新"判别点错位，`li`/`mi` 在 s2 步间不再按正确顺序传播，softmax 归一化失真，atten_out 大面积出错。

## 验证信号

本卡验证信号来自对该算子在 Ascend950（CANN 9.1.0）上的 JIT 调用 + `compare` 精度校验输出（同一开发会话内实测）。

- 回归态（`unroll_list={1,2}`）：`compare fail: atten_out, max diff: 0.06591797 at (46, 0, 103, 25), error_count: 8306526, error_count_threshold: 41943`，EXIT_CODE=1（来源平台：950；证据等级：tool_observed）
- 回退态（`unroll_list={1}`）：atten_out 28 个误差点（阈值 41943），精度通过，EXIT_CODE=0（来源平台：950；证据等级：tool_observed）

## 失效边界

- 平台：仅 Ascend950 实测；a2/a3 未验证（PyPTO 前端循环态语义应跨平台一致，但 UB/调度差异可能影响边界表现，需独立实测）。
- 算子：结论针对 flash attention 类（online-softmax + 循环态更新）算子；无循环态的逐元素循环不适用。
- 循环：仅 s2 内层循环；外层 s1/batch 循环的 unroll 行为未在本 trace 验证。
- 退化边界：`is_loop_begin`/`is_loop_end` 在 `unroll_list={1,2}` 下的具体退化机制（展开后边界点消失还是被误判为非边界）未在事件中进一步验证。

## 坏实践

- 用 `unroll_list` 扩展承载循环态的内层循环以"提升并行度"——在本算子上直接导致 8306526 误差点回归，既破坏精度又无性能收益。
- 看到 compare fail 误差点暴涨两个量级时，不先核最近一次循环态/控制流改动，而去查计算公式。