---
type: "Runbook"
title: TTK 输入无固定 seed，同用例两跑 mare 可不同
description: TTK kernel 模式不带 --seed 时输入走全局 numpy 随机状态，相消敏感的归约输出 mare 强实例依赖，同用例两套/两跑结果可翻转，勿误诊 kernel 非确定性。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TTK 输入无固定 seed，同用例两跑 mare 可不同


## 现象

对同一个 AscendC 算子开发分支（本例为 GRU Cell 反向融合算子）的白盒用例集，先后跑两套 TTK kernel 模式套件——release 精度套件与动态 OOM 插桩套件（同一用例 CSV、同一 golden）——发现同一测试用例的精度结果在两套之间翻转：某大 B fp32 用例（B=96、H=4160、输入范围 (-1,1)）的 bias 归约输出在 release 套件上 mare=3.07e-3 判 FAIL，而在 OOM 套件上同用例五路输出全部 PASS；反向也有三例（含 B=2043 的大 B 用例）在 release 套件全 PASS、在 OOM 套件的 bias 输出上 mare 1.3e-3~3.7e-2 判 FAIL。两套跑的是完全相同的用例定义与 kernel 二进制逻辑，唯一差别是两套是两次独立进程、各自重新生成了随机输入。

## 适用场景

- 工程形态：TTK kernel / GEIR 模式白盒精度评测（release 精度套件、OOM 插桩套件、以及任何多次独立执行）；对照重跑判读精度波动。
- API 路径：TTK 输入生成 `ttk/core_modules/framework_api/input_generation.py`（`generate_np_storages`）+ `ttk/utilities/data.py`（`RandomData`）；CLI `--seed` 参数。
- 验证过的平台与 CANN 版本：Ascend950PR_9579 + CANN cann-9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（TTK 工具行为 + 浮点归约数值规律，跨算子适用）。
- 路线优先级：recommended_fastpath（诊断规则）。
- 明确不适用 / 未验证场景：`--seed`（base_seed）与用例 `batch_seed` 同时存在时的固定行为（该分支语义为跨用例同 shape 同值比较，本 trace 未验证）；aclnn/e2e 模式的输入生成路径未验证。

## 判别方法

根因机制（源码级）：TTK 的 kernel 模式输入生成只有在 CLI `--seed`（base_seed）与用例 batch_seed **同时存在**时才固定随机流，否则每个输入张量都从全局 numpy 随机状态采样——每次独立运行的实例都不同。生成源码（`ttk/core_modules/framework_api/input_generation.py`，verbatim）：

```python
def generate_np_storages(testcase, switches):
    """Generate contiguous numpy storage arrays for all tensors.

    Input tensors get random data; pure output tensors get fixed values.
    Stores result in testcase.np_storages.
    """
    np_storages = []
    distribution = switches.input_distribution
    pure_output_indexes = set(testcase.pure_output_indexes)
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = resolve_custom_numpy_dtypes(testcase.flat_tensor_dtypes)
    ranges = testcase.flat_input_data_ranges or ()
    base_seed = getattr(switches, "random_seed", None)
    batch_seed = getattr(testcase, "batch_seed", None)
    for idx, view_shape in enumerate(flat_shapes):
        if view_shape is None:
            np_storages.append(None)
            continue

        dtype = flat_dtypes[idx]
        s_shape = testcase.flat_storage_shape(idx)
        # Storage shape may be None even if view_shape is not (e.g. fp8 tensors
        # without explicit storage shape). Fall back to view_shape.
        if s_shape is None:
            s_shape = view_shape
        data_range = ranges[idx] if idx < len(ranges) else (None, None)

        if idx not in pure_output_indexes:
            if base_seed and batch_seed is not None:
                # batch consistency compare different case support same shape tensor has same value
                np.random.seed(base_seed + idx)
            rd = RandomData(dtype, s_shape, data_range)
            np_storages.append(rd.generate(distribution))
        else:
            from ttk.utilities.data import fixed_np_array

            init_val = 0 if testcase.api_name in ("torch.ones", "tf.ones") else 1
            np_storages.append(fixed_np_array(dtype, s_shape, init_value=init_val))
    testcase.np_storages = np_storages
```

uniform 分布的实际采样（`ttk/utilities/data.py` 的 `RandomData._random` → `_gen_uniform_data`，verbatim 核心）：

```python
    @staticmethod
    def _gen_uniform_data(low, high, dtype, shape):
        """Generate uniform samples into a pre-allocated typed buffer chunk by chunk.

        numpy.random.uniform consumes the RandomState stream element by element
        (low + (high-low) * next_double), so the chunked output is bitwise identical
        to a single full-size uniform(...).astype(dtype) call.
        """
        elem_count = int(numpy.prod(shape)) if numpy.ndim(shape) else 1
        if elem_count <= 2 * CHUNK_ELEMS:
            return numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
```

为什么偏偏是归约输出的 mare 在实例间波动：bias 类输出 = 沿 batch 维对每列求和，随机输入下个别列的列和天然接近 0（相消列）。相对误差 `rel_err = |a − g| / (|g| + 1e-7)` 的分母由该列和的大小决定——列和 ≈ 0 时，两个同样正确的 fp32 求和序（golden 与 kernel）之差被极小分母放大，mare 可达 1e-3 量级；换一个随机实例，最险的那一列换了位置和深浅，mare 就完全不同。这就是"同用例两跑结果翻转"的数值机制，与 kernel 实现是否稳定无关（同输入下 kernel 输出仍可位级一致）。

判读操作：
1. 对比两份 result CSV 的同用例行，确认差异集中在归约类输出（bias/沿归约维的输出）且 mere 均达标、只有 mare 波动；
2. 回查跑 TTK 的命令行是否带 `--seed`——不带即本形态；
3. 结论按"输入实例不同导致的相消敏感波动"归档，不要启动 kernel 修复，也不要把两次结果不一致当成回归。

## 设计决策

本卡为诊断规则、无被放弃的实现设计，无证据。与 TTK 设计相关的取舍记录：seed 固定分支要求 base_seed 与 batch_seed 双条件，说明 TTK 有意把"默认随机"与"批量一致性比较"区分开——默认不固定种子是框架现状，判读时只能适应（多实例采样或补 seed），不能假设它默认可复现。

## 预防

- 需要结果可复现/可对账时（debug 单例、两套结果对照），给 TTK 命令显式加 `--seed`；或先用一次带 dump 的运行把输入固化。
- 对含相消敏感归约输出的用例集，判读口径按"多实例分布"而非单次 mare 定论；在验收文档里预先写明这类输出的实例依赖性，避免每轮执行都重新解释一遍波动。

## 验证信号

来源说明：以下信号来自同一 AscendC 算子开发分支白盒验证中 release 精度套件与 OOM 插桩套件的两次独立全量执行（TTK kernel 模式、上板 Ascend950PR_9579、CANN 9.0.0-beta.2），以及对 TTK 输入生成源码的核对。

- 同用例跨套件结果翻转实测：同一大 B fp32 用例（B=96、H=4160、范围 (-1,1)）grad_hidden_bias 在 release 套件 mare=3.0712e-3 判 FAIL，在 OOM 套件同用例五路全 PASS；另有三例（含 B=2043 用例）在 release 套件全 PASS、在 OOM 套件 bias 输出 mare=1.34e-3~3.74e-2 判 FAIL（mere 1e-6~3.9e-6 均达标）（来源平台：950；证据等级：tool_observed）。
- seed 机制源码事实：`generate_np_storages` 仅在 `base_seed and batch_seed is not None` 时 `np.random.seed(base_seed + idx)`，本次两套执行均未传 `--seed`（命令行 verbatim 见事件流）（来源平台：950；证据等级：tool_observed）。
- 实例依赖的量化佐证：同 shape/range/dtype 独立采样 10 个实例计算归约输出噪声，grad_hidden_bias 噪声 min=2.731e-4 / med=1.268e-3 / max=3.436e-2——跨实例相差两个数量级，与跨套件翻转幅度一致（来源平台：主机 CPU golden 仿真计算，非上板；证据等级：tool_observed）。

## 失效边界

- 本卡解释的是"浮点相消敏感输出的实例间波动"；若同输入位级重跑出现差异，那是另一类问题（确定性违约），本卡不覆盖。
- OOM 插桩套件与 release 套件除实例不同外还存在插桩二进制 codegen 差异，两因素在本 trace 未拆分；若两套差异远超实例采样波动范围，应怀疑插桩影响另查。
- 相消不敏感的输出（逐元素路径、无求和拓扑差的列）不受本机制影响，两跑应一致。
- `--seed` + batch_seed 双条件分支的行为（跨用例同 shape 同值）未验证，不能据此推断加了 seed 就完全确定。

## 坏实践

- 把"同用例两套结果不同"直接归因为 kernel 非确定或插桩破坏精度，未查 seed 机制就下结论。
- 为消除波动而在评测脚本里放宽容差或剔除波动用例——掩盖的是输入随机性，不是 kernel 缺陷（测试侧放松，反模式）。
- 用单次运行的 mare 值作为归约输出的精度结论对外报告，不标注实例依赖性。

## 误诊

- 误诊为"kernel 跑两次结果不一样 = 违反确定性"：确定性的定义是同输入位级一致；两次运行的输入本来就不同（无 seed），不构成违约证据。
- 误诊为"OOM 插桩改变了 kernel 数值行为"：在本 trace 的观测里，翻转幅度与纯实例采样的波动范围一致，插桩因素无法单独解释方向性翻转（同用例双向翻转均出现）。

## 未决事项

- `--seed` 与用例 `batch_seed` 双条件语义的实测验证（加了 seed 后同用例重跑是否位级复现）未做。
- OOM 插桩 codegen 对数值的实际影响（与实例差异的拆分实验）未做，需要时可用带 seed 的成对运行对照。
