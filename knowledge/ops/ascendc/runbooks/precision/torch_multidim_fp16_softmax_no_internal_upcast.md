---
type: "Runbook"
title: "fp16 1-ULP失配：torch多维softmax不升fp32"
description: "torch.nn.functional.softmax 对多维 fp16 张量不内部升 fp32 累加（与 1D 切片行为分叉），golden 与按 spec accumulator_dtype=float32 累加的 kernel 在 fp16 subnormal 舍入中点处差 1 ULP，被 stat_rel_err 分母 |g|+1e-7 放大成 mare 超阈孤例；修复=golden 插件输入上转 fp32 计算后经 torch 下转回原 dtype。"
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# fp16 1-ULP失配：torch多维softmax不升fp32

## 现象

在 Ascend950PR（CANN 9.2.0）上对自研 softmax 基准算子（kernel 按 spec `accumulator_dtype=float32` 计算：fp16 输入 → fp32 max/exp/sum/div → fp16 输出）跑 TTK aclnn 模式黑盒全量，535 条正向用例中恰 1 条 fp16 用例失败（黑盒 60 失败中唯一非 bf16 条目，bf16 批量失败另卡）：

- 用例：fp16，shape `(320196, 3, 1, 1, 1, 6, 2)`，`dim=0`（归约轴长 320196），输入范围 `(100, 1000)`。
- 指标：`mere=8.15e-7`（达标，阈 2^-10=9.77e-4）、`mare=0.046`（超阈 4.7×，阈 10×2^-10=9.77e-3）；其余 150 条 fp16 全过，次高 mare 仅 9.7e-4——本例是 47× 于次高值的孤例。
- dump 输入/输出逐元素分析：11,527,056 个元素中**仅 140 个**超 mare 阈，且全部落在**同一个归约切片**、golden 值完全相同：

```
Top 10 worst elements:
  golden=3.39746475e-06  actual=3.33786011e-06  abs_diff=5.96046448e-08  rel_err=0.017042  n_ulp=1.00
  （140 个超阈元素全部同值同差）
Elements with rel_err >= 10*2^-10 (mare threshold): 140 / 11527056
Golden value stats for mare-failing elements:
  min=median=max golden: 3.39746475e-06   golden subnormal (< 2^-14): 140
  fp16 ULP at golden: 5.96046448e-08
```

即：golden=3.39746475e-06（fp16 subnormal，bits `0x0039`）、kernel=3.33786e-06（`0x0038`），恰差 **1 个 fp16 subnormal ULP（2^-24=5.96e-8）**。这 140 个元素输入 x=998（切片 max=1000，exp(-2)=0.13533528），是切片内同值输入的自然聚集。同一 kernel 的白盒 TTK 与 C++ aclnn ST（独立双精度累加 oracle，181 例）全部通过；kernel 未做任何改动。

## 适用场景

- 工程形态：TTK aclnn/kernel 模式精度评测中 fp16/bf16 输出的孤例 mare 超阈分诊；golden 用 `torch.nn.functional.softmax`（CPU）且 kernel 按 spec 更高精度累加的算子。
- API 路径：`torch.nn.functional.softmax`（torch 2.7.1+cpu）；TTK `stat_rel_err`（FLOOR=1e-7）；golden_plugin `_wrap`/`__spec__` 机制；numpy/torch 离线 dump 复算。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579），CANN 9.2.0，torch 2.7.1+cpu。
- 适用粒度：platform_wide——所有「fp16/bf16 输入 + CPU torch golden + kernel fp32 累加」的算子评测（softmax 类因输出常落 subnormal 区天然高发）。
- 路线优先级：recommended_fastpath（dump 复算确认 1-ULP + golden 交叉对照后修 golden 计算精度，不动 kernel 判据）。
- 明确不适用 / 未验证场景：kernel 在 fp16 域累加的算子（应对齐 golden 原样，不该上转）；spec 明确规定 native dtype 直算的算子——对齐目标相反，须先核对权威 golden 语义；输出不落 subnormal 邻域的 shape；golden 非 torch 的评测链。

## 判别方法

**第 1 步（孤例确认）**：全 dtype 分布看失败是否集中在个别用例；本例 fp16 内部 mare 排序 0.046 vs 次高 9.7e-4，孤例特征明显。

**第 2 步（dump 逐元素分析）**：单用例 `--dump-on-fail` 后 numpy 复算 mere/mare，取超阈元素统计 golden 值分布、abs_diff 与 fp16 ULP 之比。判据：超阈元素占比极小（140/11.5M）、全部 n_ulp=1.00、golden 全部落在 subnormal 区（< 2^-14）。

**第 3 步（舍入中点机制推导）**：该切片 fp32 真值 = exp(-2)/S = 0.13533528/40189.0234 = **3.3674687e-06**，恰在 fp16 `0x0038`（56×2^-24=3.3379e-6）与 `0x0039`（57×2^-24=3.3975e-6）的舍入中点（56.5×2^-24=3.3677e-6）附近：

- CPU `np.float16(3.3674687e-06)` → `0x0038` = **kernel 输出**（kernel fp32 结果正确舍入）；
- golden 文件值 `0x0039` ⇒ golden 的 fp32 值在中点另一侧（其 S 比 kernel 小 ~0.5/40189≈1.2e-5 相对）。
- mare = 5.96e-8 / (3.397e-6 + 1e-7) = **0.017**——1 个 subnormal ULP 被 stat_rel_err 分母 `|g|+1e-7` 放大成 1.7% 相对误差（若按 DESIGN 的 `max(|g|, ε=2^-14)` 分母则为 9.25e-4，可达标）。

**第 4 步（golden 实现对照实验，定位真因）**：同一段输入数据分别计算：

```
Pure fp16（fp16 累加）:  S=40192.000000, y[x=998] = 3.3974647522e-06   ← = golden 文件值
Fp32 compute:           S=40189.023438, y[x=998] = 3.3378601074e-06   ← = kernel 输出
torch.nn.functional.softmax(x16 多维整张量, dim=0): y[x=998] = 3.397e-06 (0x0039)   ← golden 生成路径
torch.nn.functional.softmax(x16 单切片 1D, dim=0):  y[x=998] = 3.338e-6 (0x0038)   ← = fp32 路径！
Full vs slice match: False
```

**torch 对多维 fp16 张量的 softmax 不做内部 fp32 升精度（与 1D 切片行为分叉）**——golden 生成走的是整张量路径，等效 fp16 累加（S=40192），与按 spec fp32 累加的 kernel 差 1 ULP。DESIGN 文档「torch CPU 内部对 fp16/bf16 走 fp32 中间累加」的假设对多维输入不成立（torch 2.7.1+cpu 实测）。

**判定准则**：1-ULP subnormal + 中点机制 + golden 多维/1D 行为分叉三者成立 ⇒ 根因在 golden 计算精度失配，修 golden 口径而不是 kernel 或容差。

## 根因

**单句根因：`torch.nn.functional.softmax` 对多维 fp16 张量不内部升 fp32 累加（等效 fp16 域求和），golden 因此与按 spec `accumulator_dtype=float32` 计算的 kernel 在 fp16 subnormal 舍入中点处产生 1-ULP 差异，被 stat_rel_err 分母 `|g|+1e-7` 放大成 mare 超阈。**

机制链：golden 走 fp16 输入 → fp16 累加（exp/求和/除法都在 fp16 精度）→ fp16 输出；kernel 走 fp16 输入 → fp32 累加 → fp32→fp16 Cast 舍回。softmax 大输入（range(100,1000)）经 max-shift 后 exp 值极小、输出大量落在 fp16 subnormal 区（ULP 步进 2^-24 量级），两条舍入链在次正规舍入边界处的方向可差 1 ULP；相对误差 `1-ULP/(|g|+1e-7)` 在 |g|≈3.4e-6 时达 0.017，远超 2^-10 阈值 → 误报。这是 oracle 未实现 spec 累加契约，不是 kernel 缺陷。

## 解决配方

**修复前**（blackbox `golden_plugin.py` 的 `_wrap`：按输入原 dtype 直通 golden——golden 收到 fp16 整张量，走 torch 多维路径=fp16 累加）：

```python
def _wrap(*positions, **kwargs):
    # positions 为 GetWorkspaceSize 全位置参数（含标量属性位与输出位）；剔除输出位，只把真实输入喂外部 golden
    inputs = [p for i, p in enumerate(positions) if i not in _OUTPUT_IDX]
    np_ins = [_to_np(t) for t in inputs]
    ...
    out = _golden_fn(*np_ins, **call_kwargs)
    ...
```

**修复后**（输入上转 fp32 → golden 以 fp32 计算 → 输出经 torch 下转回原 dtype，保证舍入与 kernel 的 fp32→b16 Cast 同构；注意 dtype 判断**必须先查 `bfloat16` 再查 `float16`**——`"bfloat16"` 字符串含子串 `"float16"`，顺序颠倒会把 bf16 误判成 fp16）：

```python
def _downcast_np(arr, orig_dt):
    """Downcast fp32 numpy array back to original dtype (fp16/bf16) via torch
    to ensure correct rounding (matches kernel's fp32→b16 Cast)."""
    if orig_dt is None or not hasattr(arr, 'dtype') or 'float32' not in str(arr.dtype):
        return arr
    import torch
    t = torch.from_numpy(_np.ascontiguousarray(arr.astype(_np.float32)))
    if orig_dt == 'float16':
        t = t.to(torch.float16)
    else:  # bfloat16
        t = t.to(torch.bfloat16)
    return _to_np(t)

def _wrap(*positions, **kwargs):
    inputs = [p for i, p in enumerate(positions) if i not in _OUTPUT_IDX]
    np_ins = [_to_np(t) for t in inputs]
    # DESIGN §4.3 accumulator_dtype=float32: upcast fp16/bf16 → fp32 before
    # golden computation.  torch.nn.functional.softmax on multi-dim fp16/bf16
    # tensors doesn't internally upcast to fp32 (unlike 1D), causing 1-ULP
    # mismatches with the kernel which computes in fp32 per spec.
    orig_dt = None
    upcast_ins = []
    for arr in np_ins:
        if hasattr(arr, 'dtype'):
            ds = str(arr.dtype)
            if 'bfloat16' in ds:          # ← 必须先判 bfloat16！
                orig_dt = 'bfloat16'
                upcast_ins.append(arr.astype(_np.float32))
            elif 'float16' in ds:
                orig_dt = 'float16'
                upcast_ins.append(arr.astype(_np.float32))
            else:
                upcast_ins.append(arr)
        else:
            upcast_ins.append(arr)
    ...
    out = _golden_fn(*upcast_ins, **call_kwargs)
    ...  # 输出经 _downcast_np(out, orig_dt) 下转回原 dtype
```

要点：**上转与下转必须配对**——只上转输入而不把输出舍回原 dtype，比较口径仍不一致；下转用 torch 的 fp32→fp16/bf16 Cast（与 kernel 的 Cast 舍入行为一致），不用 numpy `astype` 直接截断。

性质界定：这是修正 golden 的**计算精度**使其符合 spec 的 `accumulator_dtype=float32` 契约（数学公式 softmax 不变、判据阈值不变），不是改参考实现或放松容差——kernel 本来就按 fp32 计算，修复让 oracle 口径与 spec/kernel 对齐。

## 设计决策

- **修 golden 侧对齐 spec，而非改 kernel 对齐 torch**：spec 的 `accumulator_dtype=float32` 是精度契约的权威——kernel 已按 spec 正确实现（C++ 双精度 oracle ST 181/181 全过佐证）；torch 多维 fp16 不升精度是参考实现的口径缺口，由包装层补齐。反过来把 kernel 降级到 fp16 累加去凑 torch，才是真正的精度缺陷。
- **下转放 `_downcast_np` 单独函数**：多输出（tuple/list）时逐个下转；显式走 torch 保证 `fp32→b16` 舍入与 kernel Cast 一致。
- **记录残余边界**：上转消除的是"累加精度差"，不是"subnormal 舍入固有差"——fp16 subnormal 区的残余 1-ULP 舍入差是 fp16 固有精度极限（TTK FLOOR=1e-7 相对设计 ε=2^-14 会放大该差异），算子包侧无法也不应再修。

## 预防

- 为 fp16/bf16 算子写 TTK torch golden 时，先查 spec 的 accumulator 契约：spec 规定 fp32 累加的，golden 包装层一律上转 fp32 计算、下转回 native dtype，不要信 torch 会自己升精度（多维就不会）。
- 黑盒单例 fp16 失败而白盒/C++ ST 全过时，先 dump 该用例输入输出做 numpy 双口径（fp16 直算 vs fp32 累加后舍回）对照，再决定改哪侧。
- 只跑单例就收工要不得——TTK 输入无种子，孤例失败是数据相关的，修复后须全量 + 多次单例复跑。

## 验证信号

以下信号来自该算子黑盒失败修复会话及其独立验收（黑盒全量复跑 + 验收器独立双门禁复核 + git 溯源确认未改 golden.py 与 kernel + 构建回归），均为实测：

- 失败孤例：该 fp16 用例 mere=8.15e-7 / mare=0.046（阈 9.77e-3），151 条 fp16 中唯一失败，次高 mare=9.74e-4（来源平台：950；证据等级：tool_observed）
- dump 逐元素：140/11,527,056 超阈，全部 golden=3.39746475e-06（0x0039）、actual=3.33786e-06（0x0038）、abs_diff=5.96e-8=1 fp16 subnormal ULP、n_ulp=1.00，同一切片同输入值 x=998（来源平台：950；证据等级：tool_observed）
- golden 路径对照：纯 fp16 累加 y=3.397e-6（=golden 文件值）；fp32 计算 y=3.338e-6（=kernel）；torch 多维 softmax=3.397e-6、同数据 1D 切片 softmax=3.338e-6（CPU 离线实验；来源平台：950；证据等级：tool_observed）
- 修复后单例：mere=0.0 / mare=0.0（golden 与 kernel 逐位一致）；三次独立运行 mere 4.7e-8~1.6e-5、mare 2.4e-3~3.9e-3 全 PASS（TTK 不设种子、输入逐次不同）（来源平台：950；证据等级：tool_observed）
- 最终黑盒全量：正向 total=535 PASS=535 FAIL=0（含该 fp16 用例转 PASS）；负向 6/6 被拒绝保持；验收器独立复核 FORWARD 535/535 EXIT=0、REVERSE NEGATIVE-GATE PASS EXIT=0（来源平台：950；证据等级：tool_observed）
- 无回归：git 确认 golden.py 与 op_kernel/op_host 未被触碰；`./build.sh` 退出码 0；C++ aclnn ST 181/181 PASS（来源平台：950；证据等级：tool_observed）

## 失效边界

- 上转修复的前提是 kernel 按 fp32 累加（spec `accumulator_dtype=float32`）；若 kernel 在 b16 域累加，应反向对齐 golden 原样，否则会把 kernel 的低精度暴露出来。
- golden 多维不升 fp32 的行为来自 torch 2.7.1+cpu 实测；其它 torch 版本/后端（CUDA、其它 release）行为未验证，套用前先跑第 4 步对照实验（未验证，推断可迁移）。
- TTK 无随机种子（输入逐次不同），修复后仍存在「随机输入恰好把 fp32 结果压在 fp16 舍入中点另一侧」的残余风险——本开发会话一次全量 + 三次单例通过，未证明对所有输入稳定。
- stat_rel_err 分母 `|g|+1e-7` 与 DESIGN ε=2^-14 的差异仍在（从算子包不可修）；本修复通过消除差值本身绕开，未修指标。
- bf16 golden 上转路径（同一 `_wrap` 覆盖 bf16）无独立前后对照用例（本批 bf16 失败全部由容差套用解释，另卡）。

**未决项**：

- 残余数据依赖风险未定量：对 `(100,1000)` 范围 fp16 输入，元素落在 fp16 舍入中点另一侧的概率未建模（可构造固定种子探针统计）。
- TTK `stat_rel_err` 的 FLOOR=1e-7 vs DESIGN ε=2^-14 分母差异属工具缺陷，从算子包不可修，未提上游。
- torch 多维/1D softmax 精度行为分叉的版本边界（哪些版本/后端升 fp32）未系统验证。
## 坏实践

- 迷信文档假设「torch 内部走 fp32 中间累加」——同数据 1D vs 多维对照实验一步戳破；golden 行为必须实测。
- 用 `isclose` 等非注册 standard 改判据绕过——TTK spec 的 `standard` 字段只接受注册值，设错会令全部用例 `INPUT_GEN_FAILURE`（本开发会话实测 535/535 全灭，已回退）；且换更宽 metric 属测试侧放松红线。
- golden 下转 dtype 判断先查 `float16` 再查 `bfloat16`——顺序错误会把 bf16 golden 下转成 fp16，随后 `np.promote_types(bfloat16, float16)` 抛 `DTypePromotionError`、全量 COMPARE_FAILURE（本开发会话实测踩坑后修复）。
- 把 kernel 降级到 native dtype 累加去凑 torch golden 的低精度行为——违反 spec accumulator 契约，引入真精度缺陷。
- 只上转输入、忘记把输出舍回原 dtype——比较口径仍不对齐，且易在多输出算子上漏转。

## 误诊

- 曾怀疑 kernel 的 exp/div 硬件精度、ReduceSum 累加顺序、-inf padding——dump 逐元素分析（140 个元素全部恰好 1 ULP）直接排除多 ULP 级缺陷假设。
- 曾据 golden 反推 S（golden 的 S≈40188.5 以下 vs kernel 40189.02）怀疑 kernel 求和偏差 1.2e-5——方向反了：kernel 的 S 与 fp64 参考更接近，是 golden（等效 fp16 累加）的 S=40192 偏得更多。
- 该 fp16 失败最初与 59 条 bf16 失败混在同一批黑盒失败报告里，容易一并归因"kernel fp16/bf16 精度缺陷"；实际是两个独立根因（bf16=容差套用错误，fp16=golden 累加口径），修复位置都在 golden 包装层但互不相干，须分开归因分开修。
