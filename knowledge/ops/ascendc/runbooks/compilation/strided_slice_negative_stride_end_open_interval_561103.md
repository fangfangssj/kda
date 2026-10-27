---
type: "Runbook"
title: "负步长 end 开区间致输出 shape 不符报 561103"
description: 负步长切片 begin=1023/end=0/strides=[-1] 的输出长度是 1023 不是 1024（end 开区间），预分配输出 [1024] 与算子推导的切片输入 shape 不一致，aclnn 内部 ViewCopy 校验失败 161002、对外只报 GetWorkspaceSize 561103；全逆序用 begin_mask/end_mask 表达，561103 真因靠 stdout 日志 grep ERROR 挖出。
tags: [task.debugging, topic.compute.indexing, topic.implementation.masking]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# 负步长 end 开区间致输出 shape 不符报 561103

## 现象

在 Ascend950PR 上为 BenchNoHarnessStridedSlice 编写 msprof 性能采集 demo，三场景中的负步长场景这样构造：`x_shape=[1024]`（fp16）、输出预分配 `y_shape=[1024]`、`begin=[1023], end=[0], strides=[-1]`、五个 mask 全 0——意图是"完整逆序 x[::-1]"。运行 demo 时场景1（连续）、场景2（正间隔）正常，场景3 直接在第一段接口失败：

```
场景: scene3_STRIDED_NEG
描述: [1024] fp16 strides=[-1] 负步长切片 (TilingKey 3)
[FAIL] GetWorkspaceSize 失败: 561103
```

561103 本身没有任何信息量：提高日志级别（0/1）stdout 也没有额外输出。开启 stdout 日志打印并 grep ERROR 后才看到错误链的底端真因：

```
[ERROR] OP(...): [contiguous.cpp:662][NNOP][CheckViewCopyParams] errno[161002]
OpName:[aclnnBenchNoHarnessStridedSlice_0_ViewCopy]
Input tensor's shape[[1023]] should be same with output's shape[[1024]].
[ERROR] OP(...): [contiguous.cpp:675][NNOP][ViewCopy] errno[161002] Params check failed.
[ERROR] OP(...): [aclnn_bench_no_harness_strided_slice.cpp:172] errno[561000]
OpName:[aclnnBenchNoHarnessStridedSlice_0] check viewCopyResult != nullptr failed.
```

## 适用场景

- 工程形态：aclnn 自定义/组合算子的调用侧程序（性能 demo、ST 用例、接入代码）构造 strided_slice 类参数；以及一切"aclnn 第一段接口返回 561103/561000 但看不出原因"的排障。
- API 路径：aclnn 两段式接口（GetWorkspaceSize → aclnn）；L2 实现内部的 `l0op::Contiguous`/ViewCopy 参数校验（`contiguous.cpp` `CheckViewCopyParams`，errno 161002）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——strided_slice / 切片索引类算子的负步长语义 + aclnn 报错不透出的通用排障方法。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：正步长切片（end 开区间语义同样成立但本卡证据来自负步长，未单独实测）；整图（GE）路径的报错透出行为未验证。

## 根因

负步长切片的 `end` 是开区间边界——`begin=1023, end=0, stride=-1` 取索引 1023, 1022, …, 1（不含 0），输出长度 = ceil((1023-0)/1) = **1023** 而不是 1024；demo 预分配的 `y_shape=[1024]` 与算子按语义推导出的切片输入 shape `[1023]` 不一致，L2 实现内部 Contiguous/ViewCopy 的参数校验（输入 shape 必须等于输出 shape）失败，对外只透出 `GetWorkspaceSize 失败: 561103`。

## 解决配方

判别方法（把 561103 挖到根因的可执行路径）：

```bash
# stdout 日志打开 + 级别调高，再 grep ERROR（默认级别下 Run 侧错误不透出）
ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 ./perf_demo 3 1 2>&1 | grep -E "ERROR.*(contiguous|ViewCopy|strided)"
# → contiguous.cpp:662 CheckViewCopyParams ... Input tensor's shape[[1023]] should be same with output's shape[[1024]]
```

解决配方：要表达"完整逆序 x[::-1]"，用 mask 语义而不是手算 begin/end——`begin=0, end=0, strides=[-1], begin_mask=1, end_mask=1`（mask 置位后 begin/end 被忽略：负步长时 begin 取 dimSize-1、end 取含索引 0 的下界），输出 shape 与输入一致。

修复前（perf_demo.cpp MakeScene3，verbatim 取自修复前 file_write）：

```cpp
// 场景3: STRIDED_NEG [1024] fp16 strides=[-1]
//   输出=[1024], 最内层 srcStride=-1 → TilingKey 3
//   搬运量: 读 1024*2=2KB, 写 1024*2=2KB
static PerfScene MakeScene3() {
    PerfScene s;
    s.name = "scene3_STRIDED_NEG";
    s.x_shape = {1024};
    s.y_shape = {1024};
    s.begin = {1023};
    s.end = {0};
    s.strides = {-1};
    s.begin_mask = 0;
    s.end_mask = 0;
    s.ellipsis_mask = 0;
    s.shrink_axis_mask = 0;
    s.new_axis_mask = 0;
    s.expected_tiling_key = 3;
    s.desc = "[1024] fp16 strides=[-1] 负步长切片 (TilingKey 3)";
    return s;
}
```

修复后（edit verbatim；同时把数据量放大到 1M 元素以便带宽测量，见设计决策）：

```cpp
// 场景3: STRIDED_NEG [1048576] fp16 strides=[-1], begin_mask=1, end_mask=1
//   begin_mask=1(负步长→begin=dimSize-1), end_mask=1(负步长→end=-1, 含索引0)
//   输出=[1048576], 最内层 srcStride=-1 → TilingKey 3
//   等价于 x[::-1]（完整逆序），搬运量: 读 2MB, 写 2MB
//   注: 任务原文为 [1024] fp16，但 2KB 数据量过小（头开销占比 >90%），
//   改用 1M 元素 (2MB) 以获得稳定的带宽利用率测量。
static PerfScene MakeScene3() {
    PerfScene s;
    s.name = "scene3_STRIDED_NEG";
    s.x_shape = {1048576};  // 1M elements = 2MB fp16
    s.y_shape = {1048576};
    s.begin = {0};
    s.end = {0};
    s.strides = {-1};
    s.begin_mask = 1;  // 负步长: begin = dimSize-1
    s.end_mask = 1;    // 负步长: end = -1 (包含索引 0)
    s.ellipsis_mask = 0;
    s.shrink_axis_mask = 0;
    s.new_axis_mask = 0;
    s.expected_tiling_key = 3;
    s.desc = "[1048576] fp16 strides=[-1] bm=1,em=1 负步长逆序 (TilingKey 3, x[::-1])";
    return s;
}
```

## 设计决策

- 为何用 `bm=1, em=1` 表达全逆序而非手算 `begin=dimSize-1, end=-dimSize-1`：mask 语义让框架自己算边界（负步长 begin→dimSize-1、end→含 0），调用侧不再维护易错的边界算术，输出 shape 恒等于输入，天然与预分配一致。
- 顺带把 shape 从 [1024] 放大到 [1048576]：1024 元素只有 2KB 搬运量，头开销占比过高、带宽测量无意义；放大后读/写各 2MB 才能测出稳定的 MTE 利用率。这是测量有效性决策，与 561103 修复独立。
- 属 recommended_fastpath：调用侧语义修正 + mask 表达，一步到位且可读。

## 预防

- 手写负步长 begin/end 边界（`begin=1023, end=0`）再预分配一个"想当然"的输出 shape——end 是开区间，正确长度是 1023 不是 1024；全逆序请直接用 mask。
- 拿到 561103 就去查 tiling/重装环境——先开 stdout 日志 grep ERROR，把 `contiguous.cpp` 的原始校验错误挖出来再归因。
- 若调用侧输出由 infershape 结果分配（而不是手写预分配），不会触发本问题。

## 验证信号

来源说明：性能 demo 场景3 修复前后的同工程运行对比（同机同日，Ascend950PR，CANN 9.1.0）。

- 修复前：`./perf_demo all` 场景1/2 OK、场景3 `[FAIL] GetWorkspaceSize 失败: 561103`；单跑场景3 同样 561103；日志挖掘得到 `contiguous.cpp:662 CheckViewCopyParams errno[161002] Input tensor's shape[[1023]] should be same with output's shape[[1024]]`（来源平台：950；证据等级：tool_observed）。
- 修复后（改 MakeScene3 后重编）：`./perf_demo all` 三场景全部 `[OK] ... 完成`，性能循环正常执行（来源平台：950；证据等级：tool_observed）。
- 修复后 msprof op 采集场景3 成功：OpBasicInfo 记录 TilingKey 3 kernel 的采集条目（与场景分跑方法配合，见 msprof 按场景分跑卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡的 shape 公式针对负步长单维切片：输出长度 = ceil((begin-end)/|stride|)（end 开区间）；多维时逐维套用再连乘（未验证多维组合）。
- `begin_mask/end_mask` 的"负步长边界改写"语义以框架实现为准；其他 mask（ellipsis/shrink/new_axis）组合下的边界行为本卡未覆盖。
- 推断可迁移（未验证）：正步长 end 同为开区间；其他 CANN 版本的 ViewCopy 校验与报错码未实测。

**未决项**：

- 无（本闭合已追到机制层并修复验证；561103 的错误码官方释义未查，不影响结论）。
## 坏实践

- 手算负步长边界并按"输入 shape"预分配输出——end 开区间使输出长度少 1，ViewCopy 校验必挂。
- 在默认日志级别下反复重跑 561103 期待更多信息——Run 侧错误不透出，必须显式开 stdout 日志再 grep。

## 误诊

- 先后试了提高日志级别（stdout 无新增信息）、怀疑 tiling 实现错误（读 tiling.cpp 排查无果）——561103 的表层信息不足以定位，必须依赖日志里的 `CheckViewCopyParams` 原文（该调试路径叙述为会话内过程，最终归因与修复以工具输出为准）。
