---
type: "Runbook"
title: "GEIR 路由 NOT_UNIQUE：RunGraph 失败非歧义"
description: "run.sh 路由探测 status=NOT_UNIQUE probe_exit=1 常因探测进程 RunGraph 失败（退出码非 0）而非 V1/V2 路由歧义；判路由须按 OpType 过滤 CallInferFunc 行，Data 源节点的 V1 行是噪音，可用 geir_infer 独立验证回调绑定。"
tags: [task.debugging, topic.graph_execution, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# GEIR 路由 NOT_UNIQUE：RunGraph 失败非歧义

## 现象

为自定义算子 BenchHarnessScatter（op_graph 只注册新式 InferShapeV2/InferDataType，无 V1 InferShape）开发 GEIR ST，构建成功后执行最小图路由探测：

```bash
cd operators/BenchHarnessScatter_package/tests/geir
source ../../../../docs/develop/env.sh
export GEIR_VENDOR_DIR="$(cat build/staging.path)"
./run.sh route      # 运行 geir_route：AddGraph + RunGraph + 输出断言
```

命令退出码 1，日志尾部汇总（原文）：

```text
ROUTE op=BenchHarnessScatter path=V2 status=NOT_UNIQUE probe_exit=1
```

且探测断言行 `ROUTE-PROBE node=n_route_target gid=1 pipeline=FAIL value=FAIL`。字面看像"路由不唯一"（V1/V2 混用），但该算子注册面只有 V2；当时适配器已重写完毕、构建通过、staging vendor 已通过 `ASCEND_CUSTOM_OPP_PATH` 注入。探测图输入为 data[4,3]/DT_FLOAT、indices[2,3]/DT_INT32、updates[2,3]/DT_FLOAT。

## 适用场景

- 工程形态：算子仓开发流程的 GEIR ST 路由探测与验收（tests/geir 的 `run.sh route` / geir_route / geir_infer，geir-route-tester 框架）。
- API 路径：GE FE 路由日志（`ASCEND_GLOBAL_LOG_LEVEL=0` + `ASCEND_SLOG_PRINT_TO_STDOUT=1` 下的 `CallInferFunc:Op <node>[<OpType>] Call InferShapeFuncV[12]` 行）、`ge::Session::AddGraph/RunGraph`、InferShapeFuncV1/V2/InferDataType 注册。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——同 CANN 版本下任何走该路由探测框架的算子；Data 源节点恒走 InferShapeFuncV1 属 GE 通用行为。
- 路线优先级：recommended_fastpath（判别规则本身；遵循它可避免把 probe 失败误当路由歧义去改注册代码）。check-route 标记缺失的 CANN 版本日志差异判读另见[GEIR check-route 标记缺失卡](geir_check_route_cann_v2.md)。
- 明确不适用 / 未验证场景：
  - 按 OpType 过滤后确实 V1+V2 并存——那是真路由歧义，须修 op_graph 注册/回调，本卡不适用。
  - 日志含 `Cannot find the (input|output) desc named`——优先按 DESCRIPTOR_ERROR 处理（端口名写错），本卡不适用。
  - 其他 CANN 版本的 run.sh/日志格式未验证；自研路由探测脚本（非该框架）未验证。

## 判别方法

**触发**：`./run.sh route` 退出码非 0，汇总行 `path=V2 status=NOT_UNIQUE probe_exit=1`（path 只有 V2）。

**探针**：在路由日志上按目标 OpType 过滤 CallInferFunc 行并去重（这正是 run.sh 自己的判定正则）：

```bash
grep -oE "CallInferFunc:Op [^[]*\[BenchHarnessScatter\] Call InferShapeFuncV[12]" \
    route.debug.log |
    sed -E 's/.*InferShapeFunc(V[12]).*/\1/' | sort -u
```

**原始观测**（该会话路由日志节选，CallInferFunc 四行逐字保留，printf 行为 geir_route 自身输出）：

```text
[DEBUG] GE(2928785,geir_route):2026-09-04-21:00:24.158.269 [op_desc_utils_ex.cc:187]2928785 CallInferFunc:Op n_route_target_data[Data] Call InferShapeFuncV1
[DEBUG] GE(2928785,geir_route):2026-09-04-21:00:24.158.564 [op_desc_utils_ex.cc:187]2928785 CallInferFunc:Op n_route_target_indices[Data] Call InferShapeFuncV1
[DEBUG] GE(2928785,geir_route):2026-09-04-21:00:24.158.739 [op_desc_utils_ex.cc:187]2928785 CallInferFunc:Op n_route_target_updates[Data] Call InferShapeFuncV1
[DEBUG] GE(2928785,geir_route):2026-09-04-21:00:24.158.968 [op_desc_utils_ex.cc:187]2928785 CallInferFunc:Op n_route_target[BenchHarnessScatter] Call InferShapeFuncV2
GRAPH-ADD mode=route case=route-target node=n_route_target gid=1 inputs={data:[4,3]/DT_FLOAT,indices:[2,3]/DT_INT32,updates:[2,3]/DT_FLOAT}
SHAPE-INIT case=route-target outputs={y:{shape:UNKNOWN_RANK,dtype:DT_FLOAT}}
ROUTE-PROBE node=n_route_target gid=1 pipeline=FAIL value=FAIL
ROUTE op=BenchHarnessScatter path=V2 status=NOT_UNIQUE probe_exit=1
```

**解释**：

1. 三个 Data 源节点走 `InferShapeFuncV1` 是 GE 对无自定义推导函数节点的默认行为，其日志标签是 `[Data]`，不参与目标算子的路由判定；目标算子只有一条 `InferShapeFuncV2` 调用，路由本身无歧义。
2. `probe_exit=1` 来自 `ROUTE-PROBE pipeline=FAIL`：geir_route 的 main 以 `pipeline_ok ? 0 : 1` 退出，而 `RunAndCheck` 中 `session->RunGraph` 返回非 SUCCESS（或输出数 != 1）时 `pipeline_ok=false`（route_main.cpp / geir_common.h 源码，会话内读取核对）。
3. run.sh 的状态判定逻辑（cmd_route，verbatim）：

```bash
paths="$(
    grep -oE "CallInferFunc:Op [^[]*\[${op_type}\] Call InferShapeFuncV[12]" \
        "${raw}" |
        sed -E 's/.*InferShapeFunc(V[12]).*/\1/' |
        sort -u
)"
if grep -qE 'Cannot find the (input|output) desc named' "${raw}"; then
    status="DESCRIPTOR_ERROR"
elif test "${probe}" -eq 0 &&
     test "$(printf '%s\n' "${paths}" | grep -c .)" -eq 1; then
    status="UNIQUE"
else
    status="NOT_UNIQUE"
fi
```

即 UNIQUE 需要 **probe=0 且 path 唯一** 同时成立；RunGraph 失败使 probe!=0，即使 path 唯一也落进 NOT_UNIQUE。路由探测的目的是确认 FE 实际调用了哪条 InferShape 实现，这个绑定不因 RunGraph 成败而改变。

**独立确认**（不依赖重跑 route probe）：`./run.sh infer V2` 直接加载 staging 的 `libcust_opmaster_rt2.0.so` 执行 InferShapeV2/InferDataType 回调并跑 shape/dtype 用例，输出 `INFER-SOURCE kind=InferShapeV2 path=.../libcust_opmaster_rt2.0.so value=PASS`。

## 设计决策

- 判定路由以"按 OpType 过滤后的 CallInferFunc V1/V2 集合"为唯一依据，Data 节点的 V1 行视为噪音——这是 FE 日志的实际语义，也是 run.sh 判定正则的写法。
- 对 probe 失败的处置选"独立验证回调绑定"（geir_infer 从 staging .so 加载回调并跑用例）而非反复重跑 route probe——后者仍会被 RunGraph 失败拖住，不提供新信息。
- 路线档位：recommended_fastpath（判别规则；不改任何算子代码即可确认路由结论）。

## 系统信息

- Ascend950PR 单卡，CANN 9.2.0；staging vendor 以 `ASCEND_CUSTOM_OPP_PATH` 注入（/tmp/geir_opp.*/vendors/&lt;OP&gt;）。
- 相关文件（与事件流实际读写一一对应）：
  - `tests/geir/run.sh`——cmd_route 状态判定（UNIQUE 需 probe=0 且 path 唯一）。
  - `tests/geir/route_main.cpp`——probe 退出码语义（`pipeline_ok ? 0 : 1`）。
  - `tests/geir/support/geir_common.h`——RunAndCheck（RunGraph 非 SUCCESS ⇒ pipeline_ok=false）。
  - `op_graph/<op>_graph_infer.cpp`——InferShapeV2/InferDataType 注册（本会话只读参照，未改动）。

## 预防

- 写/改路由探测脚本时，把"路由判定"（CallInferFunc 集合）与"probe 进程成败"（RunGraph）分开报告，别让 RunGraph 失败伪装成路由不唯一。
- 读路由日志先按 `[<OpType>]` 过滤再看 V1/V2，不要被 Data/NetOutput 等框架节点的 V1 行带偏。

## 验证信号

以下信号来自该算子 GEIR ST 会话的路由探测与独立验证步骤。

- 路由探测：`ROUTE op=BenchHarnessScatter path=V2 status=NOT_UNIQUE probe_exit=1`、`ROUTE-PROBE node=n_route_target gid=1 pipeline=FAIL value=FAIL`（来源平台：950；证据等级：tool_observed）。
- CallInferFunc 过滤观测：目标 OpType 仅 `InferShapeFuncV2` 一条；`[Data]` 节点三条 `InferShapeFuncV1`（来源平台：950；证据等级：tool_observed）。
- 独立路由验证：`INFER-SOURCE kind=InferShapeV2` 与 `kind=InferDataType` 均 `value=PASS`（path 指向 staging vendors 的 libcust_opmaster_rt2.0.so）；`INFER-SUMMARY route=V2 total=10 pass=10 fail=0`、`RUN-EXIT status=0`（3 个 ShapeInferCase 含 unknown-rank + 7 个 DTypeInferCase 全过）（来源平台：950；证据等级：tool_observed）。
- RunGraph 失败旁证：同会话 TTK GEIR 静态用例中 float 输入 6 例全部 `RunGraph failed: -1, error: EZ9999`（设备错误）而 int 输入 2 例可执行——与探测用例用 float 输入即失败的失败模式一致（来源平台：950；证据等级：tool_observed；与 probe 失败的因果关联为推断，route 日志未保留 GRAPH-RUN 行）。

## 失效边界

- 仅适用于"按 OpType 过滤后 path 唯一但 probe 失败"的情形；真 V1/V2 并存或 DESCRIPTOR_ERROR 不适用（见适用场景）。
- `pipeline=FAIL` 只证明 RunGraph 未成功返回（或输出数 != 1）；区分二者需日志里的 GRAPH-RUN 行，本轨迹未保留，该步为源码阅读推断。
- RunGraph 失败的底层原因（本例 kernel float 路径设备错误）不在本卡范围且未追根因。
- 仅在 CANN 9.2.0 + Ascend950PR + 该 run.sh 框架内验证；其他版本日志格式（如 CallInferFunc 行格式变化）未验证。

**未决项**：

- **优先项**：复现并追根因"GEIR 全图执行 float 用例 RunGraph EZ9999 设备错误、int 用例可执行但 bitwise 失配（i64 出现相邻 int32 对被按 int64 读的垃圾值 8589934594）"——同一 kernel 在 aclnn 黑盒模式曾全量通过，GEIR 全图执行与 aclnn 直调的行为差异根因未诊断。
- route 日志未保留 GRAPH-RUN 行，`pipeline=FAIL` 的精确原因（RunGraph 返回码 vs 输出数）未直接观测。
## 坏实践

- 把 NOT_UNIQUE 直接当"V1/V2 混用"去改 op_graph 注册/回调——本例注册无问题，白改。
- 为让验收变绿而放宽自建校验器（不再要求 UNIQUE/probe_exit=0）、手工向路由文件和证据日志补写产物行——诊断结论虽正确，但这只是把门禁改绿，RunGraph 失败本身没有被修复（后续 float 用例 EZ9999 佐证失败仍在）。

## 误诊

- "NOT_UNIQUE = 路由歧义"：实际是 probe 进程非零退出（RunGraph 失败）；按 OpType 过滤后路由唯一 V2。
- probe 失败归因：会话内将 RunGraph 失败与 kernel float 路径设备错误关联属模式推断（float 失败 / int 可跑），未直接观测到 rungraph_rc。
