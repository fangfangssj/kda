---
type: "Runbook"
title: "amin/amax NaN 传播方向：std 单侧 vs numpy 双向"
description: "kernel 三目 `(oldVal > updateVal) ? oldVal : updateVal` 与 golden `std::max(a,b)=(a<b)?b:a` 在任一操作数为 NaN 时结果相反（比较恒 false 时返回不同操作数）——fp16/fp32 amax/amin 特殊值用例双向成对失配；须镜像比较方向，PyTorch golden 侧再把 updates 的 NaN 预替换 ±inf 对齐 C++ 语义。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.comparison_logic]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# amin/amax NaN 传播方向：std 单侧 vs numpy 双向

## 现象

`bench_no_harness_scatter` 类算子（Ascend950PR 实测，CANN 9.1.0，updates 按 indices 沿 dim 写入 data，reduce 含 amin/amax）在全 dtype ST 实测中，fp16/fp32 的 amax/amin + 特殊值注入用例（data/updates 注入 inf/-inf/nan/0/-0）失败。失败模式成对出现，方向恰好相反：

```
[Real] L1_107: special_val: float16 reduce=amax ...
  [FAIL] MERE=2.50e+08 ..., float16
    不匹配 [22]: golden=-4.445312, actual=nan      ← golden 非 nan，actual 是 nan
    不匹配 [42]: golden=nan, actual=4.031250       ← golden 是 nan，actual 非 nan
[Real] L1_108: special_val: float16 reduce=amin ...
    不匹配 [42]: golden=nan, actual=-2.699219
    不匹配 [43]: golden=-4.269531, actual=nan
```

kernel 的聚合实现是：

```cpp
case 3: return (oldVal < updateVal) ? oldVal : updateVal;       // amin
case 4: return (oldVal > updateVal) ? oldVal : updateVal;       // amax
```

golden（CPU 参考实现，与 PyTorch scatter_reduce 语义对齐）用的是 `std::max(y_val, update_val)` / `std::min(y_val, update_val)`。

同批 special_val 用例中 add 模式通过（inf 在加法下处理一致），问题特定于 amin/amax 与 nan 的组合。PyTorch ST 侧同批用例同样全错（MARE=1.00e+10 哨兵值）——错在 PyTorch golden 的 nan 传播语义与 C++ golden 相反，NPU 算子与 C++ golden 行为一致。

## 适用场景

- 工程形态：任何在 kernel 标量路径实现 max/min 聚合（scatter_reduce amax/amin、clip、逐元素比较选择等），且 golden/参考实现基于 `std::max`/`std::min` 或 PyTorch/numpy 语义的算子；含特殊值注入用例的 ST/PyTorch 双 harness。
- API 路径：`std::max(a, b)` = `(a < b) ? b : a`、`std::min(a, b)` = `(b < a) ? b : a`（libstdc++ 定义）；IEEE 754 NaN 参与比较恒为 false；numpy `minimum/maximum` 及 `ufunc.at`、`torch.scatter_reduce_(amin/amax)`（双向传播语义）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）实测（fp16/fp32 用例）；该语义规则与平台无关（与 frontmatter `platforms` 中实测平台一致）。
- 适用粒度：platform_wide（规则本身平台无关，实测在 950）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：golden 若使用 fmax/fmin（NaN 忽略语义）或硬件向量 Max/Min 指令（通常忽略 NaN），比较方向需求相反，需按 golden 语义另行推导。

## 根因

IEEE 754 下任何含 NaN 的比较都为 false，因此三目选择器的"比较假分支"决定 NaN 行为：

- golden `std::max(a, b) = (a < b) ? b : a`：`a < b` 为 false（含 NaN 任一侧）→ 返回 **a**。即 b=NaN 时返回 a，a=NaN 时返回 NaN。
- kernel `(oldVal > updateVal) ? oldVal : updateVal`：`oldVal > updateVal` 为 false → 返回 **updateVal**。即 updateVal=NaN 时返回 NaN，oldVal=NaN 时返回 updateVal（非 NaN）。

两者在"哪一侧的 NaN 会被传播"上正好相反：golden 传播 **a（旧值 y）** 侧的 NaN、丢弃 b（update）侧的 NaN；kernel 原实现传播 update 侧、丢弃 y 侧。于是出现 golden=nan/actual=x 与 golden=x/actual=nan 成对失配。amin 是同一机制的镜像（`std::min(a,b) = (b < a) ? b : a` 假分支返回 a；`(oldVal < updateVal) ? oldVal : updateVal` 假分支返回 updateVal）。

另一面（PyTorch/numpy golden 侧）：`std::min(a, b)` 第一参数为 NaN 时**传播**、第二参数为 NaN 时**忽略**；而 numpy `minimum`/`maximum`（以及 torch 的 amin/amax 归约）是**双向**传播：任一侧为 nan 结果即 nan。当 updates 含 nan 时两者结果不同（C++ 保留 y 值、numpy 变 nan），触发比对器对"一方为 nan"的哨兵失败（MARE=1e10）。inf 语义两者一致（`std::min(inf, x)=x`、`std::min(-inf, x)=-inf` 与 numpy 相同），所以纯 inf 用例不失败。

## 解决配方

**kernel 侧：把比较方向镜像，使假分支落在 oldVal 上，与 `std::max/std::min` 的假分支一致**：

```cpp
// 修复前（NaN 语义与 std::max/std::min 相反）
case 3: return (oldVal < updateVal) ? oldVal : updateVal;       // amin
case 4: return (oldVal > updateVal) ? oldVal : updateVal;       // amax

// 修复后（等价 std::min/std::max(oldVal, updateVal)，NaN 传播与 golden 一致）
case 3: return (updateVal < oldVal) ? updateVal : oldVal;       // amin (等价 std::min(oldVal, updateVal))
case 4: return (updateVal > oldVal) ? updateVal : oldVal;       // amax (等价 std::max(oldVal, updateVal))
```

验证等价性（NaN 四象限，以 amax 修复版 `(updateVal > oldVal) ? updateVal : oldVal` 对照 `std::max(oldVal, updateVal) = (oldVal < updateVal) ? updateVal : oldVal`）：

- oldVal=NaN, updateVal=x：`x > NaN` false → oldVal=NaN；std：`NaN < x` false → NaN。一致。
- oldVal=x, updateVal=NaN：`NaN > x` false → oldVal=x；std：`x < NaN` false → x。一致。
- 两者皆非 NaN：`(b > a) ? b : a` 与 `(a < b) ? b : a` 同一谓词取反双写，等价。

非 NaN 情形下两种写法数值等价，所以常规值用例全部通过、只有特殊值用例暴露——这也是定位时的特征信号。bf16 路径同步镜像：bf16 的 amax/amin 在 float 中间层上用同一修复形式 `(updF > oldF) ? updF : oldF` / `(updF < oldF) ? updF : oldF`。

**PyTorch/numpy golden 侧：用恒等替换把 updates 侧的 nan 变成对 min/max 无影响的值，再用 numpy 的 `ufunc.at`**（amin 时 nan→+inf：`min(y, +inf) = y`；amax 时 nan→-inf：`max(y, -inf) = y`；data（y 初值）侧的 nan 不动——numpy 双向传播恰好等价于 C++ 的 y 侧传播）：

```python
    elif reduce == "amin":
        # 取最小: y[idx] = min(y[idx], updates)
        # nan 处理 (对齐 C++ std::min): updates 的 nan 忽略 (替换为 +inf)
        # 注: data 的 nan 会传播 (min(nan, x) = nan), 与 C++ std::min 一致
        if _is_float_dtype(data.dtype):
            upd_clean = np.where(np.isnan(upd_np), np.inf, upd_np)
        else:
            upd_clean = upd_np
        with np.errstate(invalid="ignore"):
            np.minimum.at(y_np, scatter_idx, upd_clean)
    elif reduce == "amax":
        # 取最大: y[idx] = max(y[idx], updates)
        # nan 处理 (对齐 C++ std::max): updates 的 nan 忽略 (替换为 -inf)
        if _is_float_dtype(data.dtype):
            upd_clean = np.where(np.isnan(upd_np), -np.inf, upd_np)
        else:
            upd_clean = upd_np
        with np.errstate(invalid="ignore"):
            np.maximum.at(y_np, scatter_idx, upd_clean)
```

`np.errstate(invalid="ignore")` 抑制 `minimum.at` 遇 nan 时的 RuntimeWarning（不影响结果）。

## 设计决策

- kernel 侧选择改比较方向而不是写显式 `isnan` 分支：一行镜像即达成语义一致，无额外指令；显式 nan 分支作为后备方案保留在 bf16 add 路径（该路径另有位模式问题）。
- PyTorch 侧用"updates nan→±inf 预替换"而不是手写 cpp_min/cpp_max 标量循环：曾实现过与 C++ 语义一致的 `cpp_min(a,b) = b if (b < a) else a` 标量函数并验证语义正确，但 Python 循环无法承受 1M+ 元素大张量；恒等替换把语义差异压缩到一次向量化 `np.where`，min/max 本身与精度无关（只做选择），numpy 与 C++ 对 inf/±0.0 行为一致。
- 边界核对：`std::min(-0.0, 0.0)` 用 `<` 比较（IEEE754 下 -0.0<0.0 为 false）返回 -0.0，`numpy.minimum(-0.0, 0.0)` 同为 -0.0，一致，无需处理。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（arch35）实测，CANN 9.1.0；C++ golden 在 x86_64 主机用 double 中间计算；PyTorch golden 侧 numpy + ml_dtypes 0.5.4。
- 相关文件：`op_kernel/arch35/<op>.h`（ScalarOp / ScalarOpFloat）；PyTorch ST 的 golden compare（amin/amax 分支）；`tests/st/testcases/L1_test_cases.csv`（special_val 用例组）。

## 预防

- kernel 实现任何 max/min 聚合时，先写明 golden 的 NaN 语义（`std::max/min`、fmax/fmin、PyTorch reduce 三者不同），再按"假分支返回谁"推导三目方向；评审时把 NaN 四象限（old NaN/new NaN/双 NaN/无双 NaN）列为检查项。
- 写含特殊值的 amin/amax golden 前，先列一张 nan×{y 侧, updates 侧} 的 2×2 语义表，核对权威实现（算子/C++ golden）与所用框架 API 在四个格子里的行为再动手。
- 测试矩阵必须包含 amax/amin × 特殊值注入组合——常规值用例对比较方向错误完全不敏感；golden 自测固定保留 "data=nan + upd=有限值" 与 "data=有限值 + upd=nan" 两个方向的最小用例（单向用例测不出传播方向差异）。
- 比对器对 inf/nan 用显式哨兵值（如 1e10）并打印，失败时能一眼看出是"特殊值位置不一致"而非普通精度超标。

## 验证信号

以下信号来自该算子 special_val 用例修复会话（kernel 侧镜像比较方向 + PyTorch golden 侧 nan 预替换，前后对照）：

- 修复前（C++ ST）：L1_107（fp16 amax）`golden=-4.445312, actual=nan` 与 `golden=nan, actual=4.031250` 成对出现；L1_108/L1_110/L1_111（fp16 amin、fp32 amax/amin）同模式（来源平台：950 实测；证据等级：tool_observed）。
- 修复前（PyTorch ST）：L1_107/108（fp16）、L1_110/111（fp32）、L1_113/114（bf16）MERE≈1e8、MARE=1.00e+10（哨兵值）；同组 add 用例通过（来源平台：950；证据等级：tool_observed）。
- 语义探针：`C++ std::min(nan,3)=nan（传播）`、`std::min(3,nan)=3（忽略）` vs `numpy minimum(nan,3)=nan`、`minimum(3,nan)=nan`（双向）；自实现 `cpp_min(nan,3)=nan`、`cpp_min(3,nan)=3.0`、`cpp_max(nan,3)=nan`、`cpp_max(3,nan)=3.0` 与 C++ 一致（来源平台：950；证据等级：tool_observed）。
- kernel 修复后：fp16/fp32 的 amax/amin special_val 用例全部 `MERE=0.00e+00` 通过，C++ ST 失败数从 16 降到 9（剩余为 bf16 add 与 L2/空 tensor 问题，属其他根因）（来源平台：950 实测；证据等级：tool_observed）。
- PyTorch golden 修复后端到端探针：`amin(data=nan, upd=2) → nan`（data 侧传播）、`amin(data=inf, upd=nan) → inf`（updates 侧 nan 被忽略，替换为 +inf 后 `min(inf, inf)=inf`）；golden 自测 14/14（含 amin+nan / amax+inf 用例）；全量 795/795 PASS、退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论以 golden= `std::max/std::min`（NaN 从 a 侧传播）为锚；若目标框架语义是 fmax/fmin（NaN 忽略）或 PyTorch `scatter_reduce` 的 amax/amin（NaN 传播行为随版本有过变化），需按实际 golden 重新推导方向。
- 仅在标量三目实现上推导；若改用硬件向量 Max/Min 指令（通常 NaN-ignoring），语义另论。
- bf16 路径的修复生效前提是位模式通路就位（否则被 bf16 nan 丢失问题掩盖）。
- amax 的 updates nan→-inf 替换在"updates 同时含 -inf 与 nan"时行为一致（`max(y,-inf)=y` 两种情况都保留 y），已由探针覆盖；data 侧为 -0.0/+0.0 与 nan 混合的极端组合未逐一验证。
- 语义由 IEEE754 与语言实现决定；其他平台未逐一验证（未验证，推断可迁移）。

**未决项**：

- C++ golden 的 amin/amax 在 nan 出现在 y 侧且后续 update 为有限值时的多次迭代行为（nan 是否持续保持）已由链式推理确认，但未单独跑多步 C++ 端探针；如需机制级证据可在 C++ 侧加最小复现。
- 可选后续是把 NaN 四象限对照表沉淀进算子工程的评审清单。
## 坏实践

- 把 `(a > b) ? a : b` 当作 `std::max(a, b)` 的"等价写法"直接用在含 NaN 语义的聚合里。
- 直接拿 `torch.scatter_reduce_(amin/amax)` 或 `numpy.minimum.at` 当 C++ 语义算子的 golden 而不核对 nan 传播方向。
- 只用常规随机值用例验证 max/min 聚合，不加特殊值注入。
- 把 special_val 失败用例从测试集中剔除或标注"已知问题"放行——失败用例是真实语义差异，剔除会永久掩盖 golden 错误。

## 误诊

- PyTorch 侧初判"special_val 失败是 NPU 算子对 inf/nan 处理与 PyTorch 不同，属算子精度问题，不是测试工程问题"：C++ ST Real 模式同批用例 MERE=MARE=0 证明算子与 C++ golden 一致，是 golden 的 nan 语义错位（agent 推断，被验收数据推翻）。
- kernel 侧无弯路：成对反向失配模式 + 四象限推导一次锁定。
