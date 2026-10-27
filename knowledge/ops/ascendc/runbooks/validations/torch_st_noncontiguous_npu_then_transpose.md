---
type: "Runbook"
title: torch ST非连续用例：先npu()再转置才真非连续
description: torch 适配层 ST 的非连续 Tensor 用例须先 .npu()（物化为连续 device 副本）再 .t() 得到真正非连续的 NPU tensor；CPU 转置后 .npu() 得到的是连续副本，双重转置更使用例静默失效且覆盖度误报。
tags: [task.testing, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# torch ST非连续用例：先npu()再转置才真非连续

## 现象

为逐元素最大值自定义算子开发 PyTorch 接入 ST（`torch_adapter.so` 注册 `torch.ops` 调 ACLNN 两段式），其中非连续用例要求输入为非连续 Tensor（2D 转置视图）以验证 ACLNN L2 API 的 Contiguous/ViewCopy 处理路径。代码走查（在排查 bfloat16 问题时顺带发现）暴露：初版 `run_npu_test` 的非连续构造是"CPU 上先 `x1_cpu.t()`，再 `.npu()`，再在 NPU 上 `x1_npu.t()`"——等于转置两次，NPU 上拿到的是恢复连续布局的 tensor，**非连续用例实际没有测到非连续路径**，且用例静默通过（无任何报错），覆盖度报告却显示"非连续=Y"。

## 适用场景

- 工程形态：PyTorch 接入 ST（torch/torch_npu 适配层），需要构造非连续输入验证算子/ACLNN 的 Contiguous 处理。
- API 路径：`torch.Tensor.npu()`（跨设备拷贝会物化连续副本）、`torch.Tensor.t()`（转置视图）；ACLNN L2 的 Contiguous/ViewCopy 路径。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）+ CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——torch 适配层测试的非连续输入构造模式。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：C++ ST 侧的非连续构造（本轨迹 C++ 侧用直传 strides 的 aclCreateTensor，是另一套实现）；rank>2 的非连续视图（本例仅 2D 转置）；stride 含间隙的切片视图未验证。

## 根因/解决配方

**机制（单句）**：`.npu()` 会把 CPU tensor（包括非连续视图）物化为 device 上的连续副本，因此 (a) "CPU 转置后 `.npu()`"得到的是连续 tensor（非连续性不跨设备保留），(b) 再补一次 `.t()` 只是把数据转回去、恰好又回到连续布局——正确做法是先 `.npu()`（连续拷贝）再在 device 上 `.t()`（非连续视图），golden 用 CPU 侧转置版本计算。

完整代码（before/after verbatim 取自本轨迹读取与编辑事件）：

```python
# before：双重转置 —— NPU 输入实际是连续的，非连续用例静默失效
    # 非连续 Tensor: 转置（仅 2D shape 适用）
    if nc:
        x1_cpu = x1_cpu.t()
        x2_cpu = x2_cpu.t()

    # 拷贝到 NPU
    x1_npu = x1_cpu.npu()
    x2_npu = x2_cpu.npu()

    # 非连续: NPU 上也转置
    if nc:
        x1_npu = x1_npu.t()
        x2_npu = x2_npu.t()

    # 执行算子
    try:
        result = torch.ops.bench_no_harness_maximum.forward(x1_npu, x2_npu)
    except Exception as e:
        print(f"  [ERROR] 算子执行异常: {e}")
        return False

    # 取回结果
    result_cpu = result.cpu()

    # CPU golden
    golden = compute_golden_maximum(x1_cpu, x2_cpu)

    # 精度比对
    passed = compare_results(golden, result_cpu)
```

```python
# after：先 .npu()（连续拷贝）再 .t()（非连续视图），golden 用 CPU 转置版
    # 非连续 Tensor: 先拷贝到 NPU（连续），再转置（非连续）
    # 这样 NPU tensor 真正非连续，golden 用 CPU 转置版本计算
    if nc:
        x1_npu = x1_cpu.npu().t()
        x2_npu = x2_cpu.npu().t()
        x1_cpu = x1_cpu.t()
        x2_cpu = x2_cpu.t()
    else:
        x1_npu = x1_cpu.npu()
        x2_npu = x2_cpu.npu()

    # 执行算子 + 精度比对（统一异常处理）
    try:
        result = torch.ops.bench_no_harness_maximum.forward(x1_npu, x2_npu)
        result_cpu = result.cpu()
        golden = compute_golden_maximum(x1_cpu, x2_cpu)
        passed = compare_results(golden, result_cpu)
    except Exception as e:
        print(f"  [ERROR] 异常: {e}")
        return False
```

## 设计决策

- golden 与输入构造同步转置（`x1_cpu` 在 NPU 侧构造完成后才 `.t()`）：保证 CPU golden 的逻辑视图与 NPU 输入一致（同为转置视图），比对语义正确。
- 转置放在 device 上做（`npu().t()`）而不是保留 CPU 视图：`.npu()` 的物化行为使"连续性"只在 device 侧可控。
- 执行+取回+golden+比对统一包进 `try/except`：单用例异常不逃逸（配合 ERR99999 教训，见 [bf16比对.numpy()报错，栈被ERR99999掩盖](err99999_bf16_numpy_compare_double_bridge.md)）。
- 当前方案属 recommended_fastpath。

## 预防

- torch 适配层测试模板：非连续输入固定写 `x_npu = x_cpu.npu().t()`（先拷贝后转置），并断言 `not x_npu.is_contiguous()` 防回归。
- CR 清单：非连续用例必须验证"构造后 `is_contiguous()==False`"，覆盖度报告的"非连续=Y"要以该断言为准。

## 验证信号

来源说明：问题与验证来源：PyTorch ST 开发会话的全量 NPU 回归（修复后重跑），全部为 950 真机实测。

- 修正后全量：PyTorch 接入 72/72 NPU 通过，覆盖度报告 `边界: 空Tensor=Y rank1=Y rank8=Y dim1=Y 大shape=Y 非连续=Y`，浮点用例 MERE=0.00e+00、整数 bitwise exact（含非连续 Tensor 转置用例）（来源平台：950；证据等级：tool_observed）。
- 修正后全量日志落盘复跑：`PYTHON_EXIT_CODE=0`，`总计: 72 通过: 72 失败: 0 跳过: 0`（来源平台：950；证据等级：tool_observed）。
- 修正前（before 形态）无失败信号——这正是本模式的危害：用例静默通过、非连续路径未被覆盖（该判断为代码走查结论，佐证见 before 代码的双重转置逻辑；来源平台：950；证据等级：artifact_verified 的代码证据，无运行期失败观测）。

## 失效边界

- 本模式针对 2D 转置视图；更高维 permute/切片+间隙 stride 的非连续构造未在此闭环验证（原则相同：非连续性必须在 device 侧生成）。
- C++ ST 侧非连续构造（直传 strides 给 aclCreateTensor）是独立实现，本卡不覆盖。

**未决项**：

- 未给非连续用例加 `is_contiguous()` 断言（建议项，本闭环未实施）。
## 坏实践

- CPU 上转置后直接 `.npu()` 当非连续输入（`.npu()` 物化为连续副本，非连续性丢失）。
- 双重转置（CPU 一次 + NPU 一次）：数据转回去、布局恢复连续，用例完全失效且静默通过——比"漏测"更隐蔽，覆盖度报告还会误报已覆盖。

## 误诊

- 无（本问题由代码走查发现，非运行期失败定位）。
