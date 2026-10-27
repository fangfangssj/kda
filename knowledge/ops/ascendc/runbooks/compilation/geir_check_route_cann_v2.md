---
type: "Runbook"
title: GEIR check-route标记缺失：CANN日志差异验V2路由
description: CANN 9.0.0-beta.2 GE 日志缺 "registered first" 标记致 check-route 误报 staging 未选中；用 CallInferFunc V2 + RegisterOpImplToRegistry 日志证据判读真实路由。
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GEIR check-route标记缺失：CANN日志差异验V2路由


## 现象

在 ACLNN 算子仓开发流程中给 ThnnFusedGruCellGrad（V2 自定义算子，`IMPL_OP(ThnnFusedGruCellGrad).InferShape(…).InferDataType(…)` 注册）做 GEIR 图模式验收。六套用例全部通过（正向 4×44 PASS、负向 2×49 EXEC_FAILURE）之后，运行路由检查：

```bash
source docs/ThnnFusedGruCellGrad/develop/env.sh \
  && STAGING=$(cat operators/ThnnFusedGruCellGrad_package/tests/geir/runtime/staging.path) \
  && python3 skills/geir-route-tester/scripts/verify_ttk_geir.py check-route \
       --log operators/ThnnFusedGruCellGrad_package/tests/geir/logs/positive-source-static.log \
       --op-type ThnnFusedGruCellGrad --staging "$STAGING" \
       --record operators/ThnnFusedGruCellGrad_package/tests/geir/route.json
```

抛出：

```
ValueError: staging InferShape was not selected first for ThnnFusedGruCellGrad; an old or global callback may be active
```

该检查在日志的 staging op_proto 块内找标记 `MergeTypesToImpl:op type ThnnFusedGruCellGrad infer_shape func registered first.`；在 CANN 9.0.0-beta.2 的 GE 日志里 grep `MergeTypesToImpl.*ThnnFusedGruCellGrad.*registered first` 为 **0 匹配**，而 `MergeTypesToImpl.*ThnnFusedGruCellGrad` 有 **1936 匹配**——实际日志格式是 `MergeTypesToImpl:Merge types to impl, op type ThnnFusedGruCellGrad`（不含 "func registered first" 子串）。即：路由本身正常，标记串是 CANN 版本相关的日志格式差异。

## 适用场景

- **工程形态**：GEIR 图模式验收的路由检查（geir-route-tester `check-route`）；V2 自定义算子（GERT/Runtime 2.0 InferShape 路径）开发与验收。
- **API 路径**：GE 引擎日志语义——`CallInferFunc:Op <node>[<op_type>] Call InferShapeFuncV1|V2`、`RegisterOpImplToRegistry:… register OP_IMPL : [InferShape][InferDataType]`、`AddSoToRegistry:Start to AddSoToRegistry so_path:<staging>/op_proto/...`、`MergeTypesToImpl:Merge types to impl, op type <Op>`。
- **验证过的平台与 CANN 版本**：950（ascend950）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- **适用粒度**：`platform_wide`——同一 CANN 版本下任何 V2 自定义算子的 GE 路由判读（与具体算子无关）。
- **路线优先级**：`recommended_fastpath`——check-route 报"not selected first"时先 grep CallInferFunc 按 op_type 过滤验证 V2，再下结论；不要直接重跑或怀疑 staging。
- **明确不适用/未验证场景**：其他 CANN 版本（日志格式可能又变，但 CallInferFunc V1/V2 判读法不依赖 "registered first" 标记）；V1 注册（OpDef SetInferShape 走 Runtime 1.0）的算子不适用本判读（V1 才是其预期路由）。

## 判别方法

**触发**：`check-route` 抛 `staging InferShape was not selected first for <Op>; an old or global callback may be active`，且六套用例结果本身全部通过。

**探针（对 GE 执行日志逐条 grep，按 op_type 过滤）**：

```bash
LOG=operators/ThnnFusedGruCellGrad_package/tests/geir/logs/positive-source-static.log

# ① 标记串是否存在（确认是格式差异还是真没注册）
grep "MergeTypesToImpl.*ThnnFusedGruCellGrad.*registered first" $LOG | head -5   # 0 匹配
grep -c "MergeTypesToImpl.*ThnnFusedGruCellGrad\|MergeTypesToImpl.*thnn_fused" $LOG  # 1936 匹配

## ② 路由事实：本算子的 InferShape 走 V1 还是 V2（核心判据）
grep "CallInferFunc.*ThnnFusedGruCellGrad" $LOG | head -5

## ③ staging 回调注册了哪些推导函数
grep "RegisterOpImplToRegistry.*ThnnFusedGruCellGrad" $LOG | head -5

## ④ staging so 是否被加载
STAGING=$(cat operators/ThnnFusedGruCellGrad_package/tests/geir/runtime/staging.path)
grep "AddSoToRegistry.*op_proto\|AddSoToRegistry.*Save.*op_proto" $LOG | grep "$STAGING" | head -5
```

**原始观测（CANN 9.0.0-beta.2，日志原文摘录）**：

```
CallInferFunc:Op placeholder0[Data] Call InferShapeFuncV1      ← 框架内置 Data 节点，V1
CallInferFunc:Op placeholder1[Data] Call InferShapeFuncV1      ← 同上
CallInferFunc:Op op1[ThnnFusedGruCellGrad] Call InferShapeFuncV2   ← 本算子，V2 ✓

RegisterOpImplToRegistry:LocalRegistry[21998484978668984] Op type[ThnnFusedGruCellGrad] register OP_IMPL : [InferShape][InferDataType]

AddSoToRegistry:Start to AddSoToRegistry so_path:/tmp/ttk_geir_….…/vendors/ThnnFusedGruCellGrad/op_proto/lib/linux/x86_64//libcust_opsproto_rt2.0.so, package name ThnnFusedGruCellGrad
AddSoToRegistry:Save so symbol and handle in path[.../libcust_opsproto_rt2.0.so] successfully!
```

**解释（判读规则）**：

1. `CallInferFunc` 的格式是 `Op <节点名>[<op_type>] Call InferShapeFuncV<1|2>`——**按方括号内的 op_type 过滤**才是本算子的路由；`placeholder0[Data]`/`placeholder1[Data]` 的 V1 调用属于框架内置 Data 算子，与本算子无关。
2. 本算子仅有 `InferShapeFuncV2` 调用、无 V1 调用 → 走 Runtime 2.0（GERT）路径，符合 V2 注册预期。
3. `RegisterOpImplToRegistry … register OP_IMPL : [InferShape][InferDataType]` → staging 的两个推导回调均已注册；`AddSoToRegistry` 指向 staging 目录的 `libcust_opsproto_rt2.0.so` → 暂存包被加载。
4. 三项证据齐备即可判定路由正确；`registered first` 标记缺失是 CANN 9.0.0-beta.2 日志措辞变化（`Merge types to impl, op type <Op>`），在验收文档登记兼容性备注即可，不需要改代码或重跑。

## 设计决策

- **为何以 CallInferFunc V1/V2 为准而不是 MergeTypesToImpl 标记**：前者是路由调用的直接记录（每次图编译都打印，每用例一条，44×4 套共 176 次），后者是注册阶段的一次性格式化输出且措辞随 CANN 版本变化——判据应绑在事实（调用路径）上，不绑在日志措辞上。
- **为何强调按 op_type 过滤**：GE 日志里 Data/placeholder 节点恒走 V1；不过滤会把框架内置节点的 V1 误读成"我的算子进了 Runtime 1.0"，触发不必要的停线（本工作流规定"当前 OpType 出现 Call InferShapeFuncV1 即停止任务"）。
- **当前方案属哪档路线**：`recommended_fastpath`（判读方法，无代码改动）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2；GEIR 验收脚本 `skills/geir-route-tester/scripts/verify_ttk_geir.py`（check-route 的标记期望来自旧版 CANN 日志格式）；GE 日志经 `ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0` 收敛到套件日志（单文件 13.5M 行，必须 grep 钻取）。
- 目标平台：ascend950。
- 相关文件：`operators/ThnnFusedGruCellGrad_package/tests/geir/logs/positive-source-static.log`（判读对象）、`…/tests/geir/runtime/staging.path`（staging 定位）、`…/tests/geir/route.json`（路由记录）、`op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp`（IMPL_OP V2 注册，只读参照）。

## 完整代码

本卡为判读型知识，无代码修复；可执行探针即"判别方法"中的四条 grep（对任意 V2 自定义算子把 `ThnnFusedGruCellGrad` 换成目标 op_type 即可）。验收文档中的兼容性备注写法（GEIR_VERIFY.md 追加段，摘录）：

```markdown
**路由日志格式备注**：`verify_ttk_geir.py check-route` 的 `MergeTypesToImpl:op type ... func registered first.`
标记在 CANN 9.0.0-beta.2 日志中格式为 `MergeTypesToImpl:Merge types to impl, op type ThnnFusedGruCellGrad`
（不含 "func registered first" 子串）。此为 CANN 版本日志格式差异，非路由异常——OP_IMPL 注册证据
+ V2 调用证据已充分证明 staging InferShape/InferDataType 被加载并调用。
```

## 预防

- V2 算子验收前先确认所用 CANN 版本的 GE 日志措辞，路由判据固定用 `CallInferFunc:Op <node>[<op_type>] Call InferShapeFuncV2`（按 op_type 过滤）。
- 看到 `placeholder*[Data] … InferShapeFuncV1` 不要停线——先确认方括号里的 op_type 是否是自己的算子。
- check-route 失败但用例结果全 PASS 时，优先怀疑检查脚本的标记期望与 CANN 版本不匹配，而非 staging 加载。

## 验证信号

本卡验证来自同一次 GEIR 图模式验收会话（ThnnFusedGruCellGrad，ascend950 + CANN 9.0.0-beta.2），全部为对真实 GE 执行日志的 grep 实测。

- check-route 失败复现：`ValueError: staging InferShape was not selected first for ThnnFusedGruCellGrad; an old or global callback may be active`（来源平台：950；证据等级：tool_observed）。
- 标记串 0 匹配 vs `MergeTypesToImpl.*ThnnFusedGruCellGrad` 1936 匹配；实际格式 `MergeTypesToImpl:Merge types to impl, op type ThnnFusedGruCellGrad`（来源平台：950；证据等级：tool_observed）。
- 路由证据：`CallInferFunc:Op op1[ThnnFusedGruCellGrad] Call InferShapeFuncV2`（本算子仅 V2）；`RegisterOpImplToRegistry … register OP_IMPL : [InferShape][InferDataType]`；`AddSoToRegistry … so_path:<staging>/…/libcust_opsproto_rt2.0.so`（来源平台：950；证据等级：tool_observed）。
- 旁证：同日志对应的六套用例 check-result 全 PASS（正向 4×44、负向 2×49 EXEC_FAILURE）——若路由真未选中，正向用例不可能通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "registered first" 标记缺失仅在 CANN 9.0.0-beta.2 实测；其他版本需重新 grep 确认措辞（判读法本身不依赖该标记）。
- 本判读法只证明"本算子走 V2 路由 + staging 回调已注册"；不覆盖 V1 注册算子（其预期就是 InferShapeFuncV1）。
- 日志级别需开到 DEBUG（ASCEND_GLOBAL_LOG_LEVEL=0）才能看到 CallInferFunc 行；级别不够时该判据不可用。
- 950 + CANN 9.0.0-beta.2 实测；其他芯片未验证（GE 日志语义跨芯片推断可迁移，未验证）。

## 坏实践

- 见到 check-route 报"an old or global callback may be active"就直接重跑整套 600s+ 验证或怀疑 staging 安装——先花 4 条 grep 确认路由事实。
- 不按 op_type 过滤就把日志里任何 `InferShapeFuncV1` 当作本算子进 Runtime 1.0 的证据并停线。

## 误诊

- 检查脚本的报错文案（"an old or global callback may be active"）暗示回调注册异常——实际是脚本期望的日志标记串与 CANN 9.0.0-beta.2 的实际措辞不匹配；文案本身会误导排查方向。

## 未决事项

- 无。标记差异已作为兼容性备注登记进验收文档（GEIR_VERIFY.md 路由证据节）；检查脚本对多 CANN 版本的标记适配属上游工具维护事项。
