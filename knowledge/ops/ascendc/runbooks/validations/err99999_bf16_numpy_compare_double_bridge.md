---
type: "Runbook"
title: bf16比对.numpy()报错，栈被ERR99999掩盖
description: "torch 的 bfloat16 tensor 调 .numpy() 抛 TypeError: Got unsupported ScalarType BFloat16（numpy 无 bf16 类型）；PyTorch ST 比对模块 golden 侧已 .double() 而 actual 侧裸转触发崩溃，且异常被 torch_npu 包装成 ERR99999 一度掩盖真实 Python 栈——比对前两侧统一 .double() 再转 numpy，且异常须在用例内捕获。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# bf16比对.numpy()报错，栈被ERR99999掩盖

## 现象

为逐元素最大值自定义算子开发 PyTorch ST 接入测试（`torch_adapter.so` 注册 `torch.ops` 调 ACLNN 两段式，NPU Ascend950PR，torch 2.7.1+cpu + torch_npu）时，所有 bfloat16 用例进程异常退出：NPU 执行后无比对输出，torch_npu 打一行 `[ERROR] ... ERR99999 UNKNOWN applicaiton exception`（原文拼写如此）。用 `tail -30` 看完整输出后，真实 Python 栈在 ERR99999 之前（输出原文）：

```
[NPU] L0-004: 标量广播 bfloat16 [2,8,256,256]x[1]
  dtype=bfloat16, x1=(2, 8, 256, 256), x2=(1,)
Traceback (most recent call last):
  File ".../tests/st/torch/test.py", line 535, in run_npu_test
    passed = compare_results(golden, result_cpu)
  File ".../tests/st/torch/compare.py", line 166, in compare_results
    return _compare_float(golden, actual)
  File ".../tests/st/torch/compare.py", line 58, in _compare_float
    a = actual.flatten().numpy()
TypeError: Got unsupported ScalarType BFloat16
[ERROR] ... ERR99999 UNKNOWN applicaiton exception
```

当时代码状态：`compare.py` 的浮点比对里 golden 侧已 `golden.flatten().double().numpy()`，actual 侧却是 `actual.flatten().numpy()`——NPU 返回的 bfloat16 tensor 直接转 numpy 触发 TypeError。同一批用例在 C++ ST Real 模式下 bfloat16 全部通过（MERE=0），算子本身无问题。

## 适用场景

- 工程形态：PyTorch 接入测试（torch/torch_npu 适配层 + CPU golden 比对），覆盖 bfloat16 dtype 的算子。
- API 路径：`torch.Tensor.numpy()`（不支持 bfloat16）、`.double()`；比对模块 `compare.py::_compare_float`；torch_npu 对未捕获 Python 异常的 ERR99999 包装日志。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）+ CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——torch/numpy 混合比对脚手架的 bf16 处理与异常捕获纪律。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：fp16/fp32/int 类（numpy 原生支持）；`torch.Tensor.tolist()`/torch 侧直接比较等不走 numpy 的比对路径。

## 根因/解决配方

**根因（单句）**：numpy 没有 bfloat16 dtype，`torch.Tensor.numpy()` 对 bf16 tensor 抛 `TypeError: Got unsupported ScalarType BFloat16`；比对模块 golden 侧做了 `.double()` 而 actual 侧漏做，且该异常未在用例内捕获，经 torch_npu 包装成 ERR99999 日志后一度掩盖真实栈。

修复（before/after verbatim 取自本轨迹编辑事件）：

```python
# before：actual 侧（NPU 返回的 bf16 tensor）直接 .numpy()
    g = golden.flatten().double().numpy()
    a = actual.flatten().numpy()
```

```python
# after：先转 double 再转 numpy（bfloat16 不支持直接 .numpy()）
    g = golden.flatten().double().numpy()
    a = actual.flatten().double().numpy()
```

配套（同轮修复，用例内统一异常处理，防异常逃逸后被 torch_npu 包装；verbatim）：

```python
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

- 两侧统一 `.double()`（而不是只补 actual 侧）：golden/actual 走同一条转换路径，避免一侧 fp64 一侧原 dtype 的不对称。
- 比对包进 `try/except` 返回 FAIL 而不是让异常杀死进程：保证单用例失败不中断全量批次，且 Python 栈完整打印在前（不再被 ERR99999 淹没）。
- 当前方案属 recommended_fastpath。

## 预防

- torch→numpy 的比对模板：入口处统一 `t.flatten().double().numpy()`（bf16/自定义 dtype 一劳永逸）。
- CR 清单：适配层测试出现 ERR99999 且无 Python 栈时，先看完整 stdout（tail 全量）再归因；不要先怀疑算子/适配层。

## 验证信号

来源说明：问题与验证来源：PyTorch ST 开发会话的逐用例 NPU 运行与全量回归，全部为 950 真机实测。

- 修复前：bf16 用例（L0-004、L1-002）无比对输出即退出，`TypeError: Got unsupported ScalarType BFloat16` + `ERR99999 UNKNOWN applicaiton exception`（来源平台：950；证据等级：tool_observed）。
- 修复后：`L0-004: 标量广播 bfloat16 [2,8,256,256]x[1] → [PASS] bfloat16 MERE=0.00e+00, MARE=0.00e+00 (thr=7.81e-03, 1048576 elems)`、`L1-002: 无广播 bfloat16 [1024,1024] → PASS`（来源平台：950；证据等级：tool_observed）。
- 全量回归：PyTorch 接入 72/72 NPU 通过（6 dtype 全覆盖，bfloat16 10 例）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `.double()` 会把 bf16 提升为 fp64 再比对：对"bf16 原位精度"敏感的比对（要求按 bf16 精度阈值判差）需确认阈值口径（本例阈值按 dtype 给出，不受影响）。要求 bf16 位级无损往返（golden 原生 dtype 计算）的场景须改用 16-bit 视图桥接，见 [TTK kernel模式torch golden的numpy桥接插件](ttk_kernel_golden_plugin_numpy_torch_bridge.md) 与 [golden bf16 输出用 ml_dtypes 而非 torch](./golden_bf16_ml_dtypes_astype_rne_aligns_cast_rint.md)。
- torch 后续版本若原生支持 bf16→numpy，本卡退化为历史背景。

**未决项**：

- torch_npu 把未捕获异常包装成 ERR99999 并置于 Python 栈之后的日志顺序机制未深究（对结论非必需）。
## 坏实践

- 只给 golden 侧做 dtype 提升而 actual 侧裸转（本例原始缺陷形态）。
- 见到 ERR99999 就归因算子/适配层（torch_npu 会把未捕获 Python 异常包装成该日志，真实栈在其之前的 stdout 里）。

## 误诊

- 首见 ERR99999 时曾怀疑适配层创建 bf16 aclTensor 或算子执行异常；通过独立小脚本在多种 shape（1D/2D/[2,8,256,256]/[1024,1024]）直调 `torch.ops` 全部 PASS，排除算子与适配层，锁定 test.py 的 compare 环节（该独立诊断结论由复现实验 + 工具输出佐证）。
- 一次独立诊断脚本因未带算子 lib 的 `LD_LIBRARY_PATH` 报 `undefined symbol: aclnnBenchNoHarnessMaximumGetWorkspaceSize`，属环境变量遗漏，与主线问题无关。
