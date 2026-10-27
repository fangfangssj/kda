---
type: "Runbook"
title: "scatter dim=0 手写期望写反维度：值对调指纹"
description: "手写 scatter ST 期望值时把 dim=0 的目标位置写成\"同行更新\"（y[0][1]=20），正确语义是 y[indices[i][j]][j]（索引值替换 dim 维坐标，即 y[1][1]=20）；期望 [10,20,3,4] 应为 [10,2,3,20]，勿把该 FAIL 误判为 golden 算法 bug。"
tags: [task.testing, topic.compute.indexing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["agnostic"]
---

# scatter dim=0 手写期望写反维度：值对调指纹

## 现象

为 scatter 型算子（updates 按 indices 沿 dim 写入 data，reduce=None/add/multiply/amin/amax）开发 C++ ST 测试工程：CPU golden 按 torch `scatter_` 语义移植，另配 15 个手写期望值的自测用例做 golden 正确性自检（Mock 模式，无 NPU）。首次跑 Mock 自测，15 例中第 6 例 `[2D dim=0 None]` 失败，报错原文：

```
[FAIL] MERE=1.22e+00 (>= 1.22e-04), MARE=4.00e+00 (>= 1.22e-03), float32
  不匹配 [1]: golden=20.000000, actual=2.000000, rel_err=9.00e-01
  不匹配 [3]: golden=4.000000, actual=20.000000, rel_err=4.00e+00
测试 [2D dim=0 None]: FAIL
```

用例输入：`data=[[1,2],[3,4]]`（行主序 [1,2,3,4]）、`indices=[[0,1]]`、`updates=[[10,20]]`、`dim=0`。golden 算出的 actual 是 [10,2,3,20]，而手写期望 exp_vals 是 {10,20,3,4}——即期望值认为 updates[0][1]=20 落在 y[0][1]，golden 把它落在 y[1][1]。其余 14 例（1D 各 reduce、2D dim=1、负 dim、3D、int32/int64）全部 PASS。

## 适用场景

- 工程形态：scatter/scatter_reduce 族算子的 ST 测试开发（CPU golden 自测、手写期望用例），Mock 模式先行验证 golden 正确性。
- API 路径：参照 torch `Tensor.scatter_`/`scatter_reduce_` 语义的 C++ golden 实现；MERE/MARE 社区精度标准（fp32 阈值 2^-13 与 10×2^-13）。
- 验证过的平台与 CANN 版本：本闭环在 CPU Mock 模式验证（语义为 torch scatter 定义，与硬件平台无关）；算子目标平台 Ascend950PR / CANN 9.1.0。
- 适用粒度：operator_family——所有 scatter/gather 类"索引搬运"算子的测试 oracle 编写。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：NPU 实测比对（本片未运行 Real 模式）；torch 与自研 golden 在重复索引 reduce 语义上的顺序差异（另一独立问题，本卡不覆盖）。

## 根因

手写期望值误读了 scatter 的 dim 语义：`dim=0` 时，indices 的位置 (i,j) 的更新写到 `y[indices[i][j]][j]`——**索引值替换第 dim 维（第 0 维，行）坐标，其余坐标（列 j）保持**；错误期望按 `y[i][indices[i][j]]`（索引值替换列）理解，把 (0,1) 的更新 20 写到了 y[0][1] 而不是 y[1][1]。

机制推导（本用例逐位置演算）：
- (0,0)：indices[0][0]=0 → 目标 y[0][0]，updates=10 ⇒ y[0][0]=1→10
- (0,1)：indices[0][1]=1 → 目标 y[1][1]，updates=20 ⇒ y[1][1]=4→20
- 未触及位置保持：y[0][1]=2、y[1][0]=3
- 正确期望（行主序）：[10, 2, 3, 20]；错误期望：[10, 20, 3, 4]

golden 算法本身（移植 torch scatter_）计算正确；错的是手写期望数据。判别特征：FAIL 的 mismatch 恰好是"两个值换了位置"（[1] 期望 20 实际 2、[3] 期望 4 实际 20），且 dim≠0 的用例（dim=1、负 dim）全 PASS——指向 dim 维坐标替换方向的语义误解，而非算法实现错误。

## 解决配方

修复前（ST 测试中该自测用例的期望值块，含错误注释，verbatim）：

```cpp
    // --- 测试 6: 2D dim=0 reduce=None ---
    {
        // data = [[1, 2], [3, 4]], indices = [[0, 1]], updates = [[10, 20]], dim=0
        // y[0][0] = 10, y[0][1] = 20 (被覆盖), y = [[10, 20], [3, 4]]
        HostTensor data(DType::FLOAT32, {2, 2});
        HostTensor indices(DType::INT32, {1, 2});
        HostTensor updates(DType::FLOAT32, {1, 2});
        HostTensor expected(DType::FLOAT32, {2, 2});
        double data_vals[] = {1, 2, 3, 4};
        for (int i = 0; i < 4; ++i) WriteFromDouble(data.ptr(), i, DType::FLOAT32, data_vals[i]);
        WriteIndex(indices.ptr(), 0, DType::INT32, 0);
        WriteIndex(indices.ptr(), 1, DType::INT32, 1);
        WriteFromDouble(updates.ptr(), 0, DType::FLOAT32, 10.0);
        WriteFromDouble(updates.ptr(), 1, DType::FLOAT32, 20.0);
        double exp_vals[] = {10, 20, 3, 4};
        for (int i = 0; i < 4; ++i) WriteFromDouble(expected.ptr(), i, DType::FLOAT32, exp_vals[i]);
        run_test("2D dim=0 None", data, indices, updates, 0, "None", expected);
    }
```

修复后（更正期望值与逐位置注释，verbatim）：

```cpp
    // --- 测试 6: 2D dim=0 reduce=None ---
    {
        // data = [[1, 2], [3, 4]], indices = [[0, 1]], updates = [[10, 20]], dim=0
        // (0,0): y[0][0] = 10
        // (0,1): y[1][1] = 20 (dim=0: 行索引=indices值, 列索引=j)
        // y = [[10, 2], [3, 20]]
        HostTensor data(DType::FLOAT32, {2, 2});
        HostTensor indices(DType::INT32, {1, 2});
        HostTensor updates(DType::FLOAT32, {1, 2});
        HostTensor expected(DType::FLOAT32, {2, 2});
        double data_vals[] = {1, 2, 3, 4};
        for (int i = 0; i < 4; ++i) WriteFromDouble(data.ptr(), i, DType::FLOAT32, data_vals[i]);
        WriteIndex(indices.ptr(), 0, DType::INT32, 0);
        WriteIndex(indices.ptr(), 1, DType::INT32, 1);
        WriteFromDouble(updates.ptr(), 0, DType::FLOAT32, 10.0);
        WriteFromDouble(updates.ptr(), 1, DType::FLOAT32, 20.0);
        double exp_vals[] = {10, 2, 3, 20};
        for (int i = 0; i < 4; ++i) WriteFromDouble(expected.ptr(), i, DType::FLOAT32, exp_vals[i]);
        run_test("2D dim=0 None", data, indices, updates, 0, "None", expected);
    }
```

注意这是**期望数据修正**而非放宽容差：MERE/MARE 阈值（fp32 1.22e-04 / 1.22e-03）未动，golden 算法未动，只把手写期望从违反 scatter 语义的值改为按 `y[indices[i][j]][j]` 逐位置推导的值。

## 设计决策

- 修复方向选择"改期望"而非"改 golden"：先用独立通道交叉验证——golden 是从标杆 golden.py（torch scatter_）逐语句移植的，且其余 14 例（覆盖 dim=1、负 dim、3D、int 索引）全 PASS；唯一失败用例的手写期望未经过语义推导。逐位置演算确认期望错、golden 对。
- 修复后同时把注释从含糊的"被覆盖"改为逐位置坐标推导，防止后续维护者再次误读。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 手写 scatter 族期望值时，对每个更新位置显式写坐标推导注释：`目标 = indices 值替换第 dim 维坐标，其余坐标照抄 indices 的下标`；dim=0 与 dim=1 各写一个 2D 用例互为镜像，写反会立刻交叉暴露。
- golden 与手写期望不一致时，先看 mismatch 的结构（值对调/整体偏移/部分错）再决定改哪边；"两个值换了位置"是 dim 语义写反的指纹。
- 用偶数个不对称值（如 10/20、位置 1/3）构造用例，让对调可被定位（对称值对调不可见）。

## 验证信号

以下信号来自该算子迭代一 ST 测试工程的 Mock 模式自测（期望值修正前后各跑一次完整自测，CPU Mock 模式）：

- 修复前：15 例自测中该用例 FAIL，MERE=1.22e+00（阈值 1.22e-04）、MARE=4.00e+00（阈值 1.22e-03），mismatch [1] golden=20/actual=2（rel_err=9.00e-01）、[3] golden=4/actual=20（rel_err=4.00e+00）（来源平台：CPU Mock 模式；证据等级：tool_observed）。
- 修复后：CPU Golden 自测 15/15 PASS（含该用例 MERE=0.00e+00），L0 用例 28/28 PASS（来源平台：CPU Mock 模式；证据等级：tool_observed）。
- 修复后复核：`-Wall -Wextra` 无警告、Mock 编译通过；后续重跑保持 15/15 + 28/28（来源平台：CPU Mock 模式；证据等级：tool_observed）。

## 失效边界

- 结论在 CPU Mock 模式闭环（golden 自洽性）；NPU Real 模式下 kernel 是否符合同一语义未在本片验证。
- scatter 语义以 torch `scatter_` 为准（reduce=None 覆盖、重复索引非确定顺序）；若目标框架语义不同（如 TensorFlow scatter 的维度约定），期望值推导须按各自语义重做（未验证）。
- 3D+ 的 dim 语义按同一规则类推（索引值替换第 dim 维坐标），本轨迹 3D 用例（dim=1）PASS 佐证但未逐维展开。
- 语义为框架定义、与硬件平台无关；NPU 侧行为未验证（推断可迁移）。

**未决项**：

- 无（本闭环完整：失败→定位→修正→复测通过）。
## 坏实践

- 手写期望值不写坐标推导，靠"看起来对"直觉填数。
- golden 与期望不符时先改 golden 或放宽容差，而不先复核期望数据本身。

## 误诊

- 初判方向是"golden 算法 2D 分支有 bug"：其余 14 例（含 2D dim=1、负 dim、3D dim=1）全 PASS 排除了通用 2D 路径错误；逐位置演算后确认错在手写期望的 dim=0 坐标方向。
