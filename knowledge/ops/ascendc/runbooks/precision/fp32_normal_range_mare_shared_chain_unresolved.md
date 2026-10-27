---
type: "Runbook"
title: fp32 常规值域 mare 跨分支同构未解
description: 多分支算子共享计算链时，TTK 白盒 fp32 常规值域用例成簇 mare 超阈（4e-3~0.47）、fp16 全过、bf16 仅极值失配——逐 dtype×类别分类 + 跨分支失败画像对照 + tilingKey 核验三步探针可把问题归因收敛到两分支共享的计算链、排除布局寻址；机制级根因未定论，先分类后归因勿急改 kernel。复现增量：branch-0（BSND）236 条同指纹实例 + golden 通路核验/同 shape 跨 dtype 对照探针 + bf16 mare=0.0 证明力误诊警示。
tags: [task.debugging, topic.precision, task.testing]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# fp32 常规值域 mare 跨分支同构未解

## 现象

RoPE 算子 Branch-2（half-BNSD-cos2d，与 branch-0 仅 `kIsBnsd` 坐标反算不同）kernel 实现 ST 全绿（12/12）后，用 TTK kernel 模式在 Ascend950PR_9579 全量跑 238 条白盒用例（golden 为独立 torch 实现，经 numpy↔torch 桥接插件喂入）：

```
total=238 pass=175 fail=62 skip=0   GATE: FAIL
（另有 1 条 PROFILE_CRASH：case00231 (8,128,4096,128) fp32）
```

逐 dtype 拆开是清晰的三段画像：

| dtype | total | pass | fail | 失败构成 |
|-------|-------|------|------|----------|
| float16  | 72 | **72** | 0  | — 全过 |
| bfloat16 | 72 | 60 | 12 | 全为极值 ±3.3895e38/±inf/nan 的 golden↔kernel NaN/Inf 不一致 |
| float32  | 94 | 43 | 51 | 30 条 normal-range mare 超阈 + 12 极值 + 8 expect_error（算子正确拒绝非法输入，TTK 记 FAIL）+ 1 PROFILE_CRASH |

fp32 的 30 条 normal-range 失败覆盖 ±0.1~±200 的正常取值范围（阈值 1.22e-4）：case00087 mare=0.4728/0.3654（range ±5~10）、case00074 mare=0.0164（±0.1~0.2）、case00081 mare=0.0041（±0.1）——量级与 branch-0（BSND 布局）白盒阶段的 "4e-3~4.77" 同构。

## 适用场景

- 工程形态：TTK kernel 模式白盒精度执行与失败分类（多分支算子，多个分支共享一条计算链模板）。
- API 路径：`ttk kernel -i <csv> --pc=1 --plugin <golden_plugin>`；`parse_result.py`（total/pass/fail 门禁）；result.csv 的 precision_status/dyn_precision/precision_metrics/dyn_tiling_data 列。
- 验证过的平台与 CANN 版本：Ascend950PR_9579，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：same_operator——本算子分支族内的精度失败判读；"共享计算链 → 失败画像跨布局复现"的推理模式可推广到其他多分支模板化 kernel（operator_family，未验证）。
- 路线优先级：safety_fallback（判读规则：先分类归因、不急于改 kernel；本卡不提供修复）。fallback 声明：本卡只给失败分诊与归因收敛方法，不含任何已验证的 kernel 修复；在根因定论前不得据本卡断言 kernel 有缺陷或无缺陷。
- 明确不适用/未验证场景：单分支算子（无跨分支对照可用）；fp32 mare 的机制级根因（舍入顺序/累加误差 vs 真缺陷）本卡未定论。

## 判别方法（三步探针）

**第一步：逐 dtype × 失败类别分类**。把 63 条非通过按（dtype × 类别）计数——类别按 `dyn_tiling_key`/`dyn_precision`（error-reject）、`input_data_ranges`（极值 vs 正常范围）、`precision_status`（PROFILE_CRASH）划分：

```
=== Non-PASS classification (n=63) ===
  expect_error_compile_reject: total=1  by_dtype={'float32': 1}
  expect_error_tiling_reject:  total=7  by_dtype={'float32': 7}
  extreme_maxrange: total=12  by_dtype={'float32': 6, 'bfloat16': 6}
  normal_range_mare: total=30 by_dtype={'float32': 30}
  other: total=12            by_dtype={'float32': 6, 'bfloat16': 6}   # inf/nan 字符串 parse_err
  profile_crash: total=1      by_dtype={'float32': 1}
```

分类探针（verbatim 取自执行脚本）：

```python
def fclass(r):
    dp=r["dyn_precision"]; tk=r["dyn_tiling_key"]; ps=r["precision_status"]
    rng=r["input_data_ranges"]
    if "OPTILING_FAILURE" in tk or "INVALID_TILING" in dp:
        return "expect_error_tiling_reject"
    if "DYN_COMPILE_FAILURE" in dp or "DYN_COMPILE_FAILURE" in tk:
        return "expect_error_compile_reject"
    if ps=="PROFILE_CRASH":
        return "profile_crash"
    # range-based
    try:
        tup=eval(rng); lo,hi=tup[0]
        if (isinstance(lo,str) and 'nan' in lo.lower()) or (isinstance(hi,str) and 'nan' in hi.lower()): return "extreme_nan"
        if (isinstance(lo,str) and 'inf' in lo.lower()) or (isinstance(hi,str) and 'inf' in hi.lower()): return "extreme_inf"
        lo_f,hi_f=float(lo),float(hi); mx=max(abs(lo_f),abs(hi_f))
        if mx>=1e38: return "extreme_maxrange"
        return "normal_range_mare"
    except Exception:
        return "other"

nonpass=[r for r in rows if r["precision_status"]!="PASS"]
cls=collections.defaultdict(lambda: collections.Counter())
for r in nonpass:
    cls[fclass(r)][dtype_of(r["input_dtypes"])]+=1
```

mare 提取（normal-range 失败样本，从 result.csv 的 precision_metrics 列递归找 'mare' 键）：

```python
def extract_mare(metrics_str):
    try:
        d=ast.literal_eval(metrics_str)
    except Exception:
        return None
    def walk(o):
        out=[]
        if isinstance(o,dict):
            for k,v in o.items():
                if isinstance(v,(int,float)) and any(t in str(k).lower() for t in ("mare","rel_err","max_rel","stat_rel")):
                    out.append((k,v))
                else: out.extend(walk(v))
        elif isinstance(o,list):
            for x in o: out.extend(walk(x))
        return out
    return walk(d)
```

**第二步：跨分支失败画像对照**。同一算子三个分支的画像并排：

- branch-0（BSND，共享计算链）：fp16 72/72 全过、bf16 极值-only、fp32 normal-mare + 极值 + crash + error-reject——**与 branch-2 同构**；
- branch-1（interleaved，独立计算链 + 当时存在 D=88 对齐 bug）：fp16 也大量失败——**不同构**。
- branch-0 与 branch-2 的 kernel 除 `kIsBnsd` 坐标反算（s=row%S vs s=(row/N)%S，只影响 cos/sin 加载与 GM 定位）外逐位相同：rotate_half 分解、Mul/`asc_vf_call<MulAddVF>` 链、升精度 Cast、CopyOut 全部共享。fp32 失败画像在两个布局下同构 ⇒ 排除 BNSD/BSND 寻址差异，归因收敛到**共享 fp32 计算链**（或 golden 侧 fp32 口径）——该归因是排除法假设，机制级未验证（见误诊）。

**第三步：分支调度核验**（防"测错分支"假象）：确认 tiling 数据末字段 tilingKey=2——229 条成功编译用例全部携带 2，证明失败确实出自 branch-2 路径而非误入 branch-0（判读方法见同批《TTK dyn_tiling_key 恒 0：分支调度看 tiling 数据末字段》卡）。

## 设计决策

- 判读顺序选"先分类后归因"而不是"见 FAIL 就修 kernel"：63 条非通过里 20 条（8 error-reject + 12 极值 parse/NaN 口径）不是精度缺陷、1 条是 crash，真正的可疑面只有 fp32 normal-range 30 条——先分类能把调试面收窄 2/3。
- 归因依据选"代码结构对照 + 画像同构"双证据（两者都可重建），不单凭画像相似下结论。
- 本任务契约"只跑不改 kernel"（whitebox-tester），故止步于诊断；修复留给 developer 侧。
- 当前方案路线档位：safety_fallback（诊断规则本身；对 fp32 mare 的最终处置未定）。

## 系统信息

- 硬件/软件：Ascend950PR_9579，CANN 9.2.0；TTK（ops-test-kit 安装根 `$TTK`）；golden 插件 `tests/whitebox/ttk/golden_plugin.py`（numpy↔torch 桥接，数学在独立 golden.py）。
- 产物：`tests/whitebox/ttk/result.csv`（238 行 + 35 列含 precision_metrics/dyn_tiling_data）、`per_case_precision.csv`、`whitebox_failures.md`（63 条逐例详报）。
- 被测：Branch0Kernel<DTYPE_QUERY, true>（branch-2 BNSD 路径），vendor 源安装于 `opp/vendors/BenchHarnessApplyRotaryPosEmb`。

## 验证信号

以下信号来自 Branch-2 白盒 TTK 全量执行会话（只跑不改 kernel；分类/提取脚本为会话内探针）：

- 门禁：`total=238 pass=175 fail=62 skip=0`，另有 PROFILE_CRASH 1 条（parse_result 计 fail=63）；逐 dtype fp16 72/72、bf16 60/72、fp32 43/94（来源平台：950；证据等级：tool_observed）。
- fp32 normal-range mare 样本：case00087 mare=0.4728022515773773/0.36542072892189026（range (-5,10)）、case00074 mare=0.016391277313232422（(-0.1,0.2)）、case00081 mare=0.0040978193283081055（(-0.1,0.1)）、case00076 mare=0.02068212255835533（(-1,2)）——阈值 1.22e-4 全部超阈（来源平台：950；证据等级：tool_observed）。
- 失败分类计数：normal_range_mare 30（全 fp32）、extreme_maxrange 12（fp32×6+bf16×6）、inf/nan parse_err 12（fp32×6+bf16×6）、error-reject 8（算子正确拒绝 rank2/rank0/D-odd/cos 不匹配）、profile_crash 1（来源平台：950；证据等级：tool_observed）。
- 跨分支画像对照事实：branch-0 失败画像与 branch-2 同构、branch-1 不同构，且 branch-0/2 kernel 仅 kIsBnsd 坐标反算差异（会话内结果文件与源码读取）（来源平台：950；证据等级：tool_observed）；由此推出的"归因共享 fp32 计算链"是排除法假设，机制级未验证（见误诊）。

## 失效边界

- 归因"共享计算链"是排除法结论（排除了布局寻址），不是定位结论——fp32 链内具体哪一环（Mul/MulAddVF 舍入顺序、golden torch fp32 口径差、UB 加载路径）未验证。
- 画像同构判读依赖"两分支 kernel 确实只差坐标反算"这一代码结构前提；若分支实现已分叉，同构推理失效。
- bf16 极值 12 条与 fp32 极值 12 条属"双方同溢出下的比较语义"（golden 自身在 fp32 内溢出→inf/NaN），非 kernel 缺陷证据，但本切片只做了分类未复核 golden（branch-0 阶段曾逐例核过）。
- PROFILE_CRASH（约 2GB×6 张量）疑 OOM/超时，未诊断。
- 该判读规则依赖 fp16 全过这一对照：若失败连 fp16 也挂，应回到切分/路由/tiling 排查，本卡规则不适用。

## 坏实践

- 把 63 条 FAIL 笼统当"branch-2 精度回归"去修 kernel——其中 20 条是工具口径（error-reject + 极值 parse）、1 条 crash，真正的可疑面只有 fp32 normal-range 30 条。
- 不核验分支调度就下"某分支精度差"的结论——`dyn_tiling_key=0` 列会误导（host 单入口 SEL 限制），必须看 tiling 数据末字段。
- 在机制未定论前把"共享计算链归因"当结论引用——它只是收窄了排查面。

## 误诊

- 归因"共享 fp32 计算链"当前仅为强假设（本切片 whitebox-tester 契约只跑不改，无 kernel 侧实验）；机制级根因（如 MulAddVF 与 torch fp32 逐元素舍入差异、或近零相消相对误差放大）未定论，下结论前不得引用本卡为定论。

## 遗留问题（优先项）

- **复现并追 fp32 mare 根因到机制层**（优先项）：最小 fp32 用例（如 (3,7,19,32) 小 shape）上逐元素 diff kernel 与 golden 的 fp32 输出，定位 Mul/MulAddVF 链中的舍入差异点或确认近零相消的相对误差放大假设。
- case00231（(8,128,4096,128) fp32）PROFILE_CRASH 原因（疑 OOM/超时）未诊断。
- bf16/fp32 极值 24 条的"双方同溢出比较语义"本切片未复核 golden（branch-0 阶段已核，可引用其结论补齐）。

## 后续增量（2026-09-17，并行策展批：branch-0（BSND）同指纹实例与 golden 通路核验探针）

同一 RoPE 算子 branch-0（half-BSND-cos2d）白盒阶段的 236 条先行观测与本卡 branch-2 画像同指纹，并补两个本卡未做的 golden 通路核验探针与一条误诊警示：

**1. branch-0 实例**：`total=236 pass=172 fail=64 skip=0`——fp16 72/72 全过；bf16 60/72（12 条失败全部是 ±3.39e38/±inf 极值的 NaN/Inf_mismatch，正常范围全过且 case00001（range (-50,200)，shape (8,57,431,64)）mare=0.0 精确匹配）；fp32 40/92——31 条正常数值范围（(-50,200)、(-0.1,0.2)、(-5,10)、(-1,2) 等）的 mare 超阈（4.1e-3~4.77，阈值 1.22e-4）。该 kernel 此前的 12 条 ST 用例（小 shape、逐元素 `diff <= atol + rtol*|expected|` 判据）三 dtype 全绿——失配只在 TTK 大 shape + mare/mere 指标下出现。

**2. golden 通路核验（本卡 branch-2 切片未做、branch-0 切片补齐）**：
- 用与 TTK 相同的真实输入直接调用外部 golden（绕过 kernel 与插件），确认插件桥接无误（bf16 (8,57,431,64) `direct golden match q: True k: True`）；
- 对失败用例的输入范围跑 golden 自身 sanity（case00074 range (-0.1,0.2) golden has_nan=False、max=0.0779——golden 正常）；
- 同 shape 同 range 跨 dtype对照：bf16 case00001 PASS mare=0.0 vs fp32 case00073 FAIL mare=8.74e-2。

**3. 误诊警示（bf16 mare=0.0 的证明力）**：分析中曾以"bf16 mare=0.0 精确匹配 ⇒ kernel fp32 计算与 golden 完全一致 ⇒ fp32 失配必为 kernel dtype 分支 bug"推进推理——该推理有漏洞：bf16 的 mare=0.0 只证明差异小于 bf16 量化步长（~0.4%），不能排除 fp32 级微小差异（正是"近零相消放大"假设的基础）。两种假设（kernel fp32 路径真缺陷 vs 指标近零相消放大）在 branch-0 切片同样未定论，区分探针（失败用例 kernel/golden 逐元素 diff、看最大相对误差点的 |golden| 是否≈0）仍未执行。

### 复现增量验证信号

- branch-0 全量：total=236 pass=172 fail=64 skip=0；fails by dtype {'float32': 52, 'bfloat16': 12}（fp16=0）；float32|mare_exceeded|normal 31 条（4.10e-03~4.77e+00）（来源平台：950；证据等级：tool_observed）。
- bf16 case00001 (8,57,431,64) range(-50,200) PASS mare=0.0 vs fp32 case00073 同 shape 同 range FAIL mare=8.74e-2；case00074 golden has_nan=False max=0.0779；direct golden match q:True k:True（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- branch-0 与 branch-2 的失败画像同构进一步支持"共享 fp32 计算链"归因，但机制级根因（MulAddVF 融合舍入序 vs 近零相消放大）仍属两假设未定论——复现并定论根因仍是优先项。
- 极值范围 NaN/Inf mismatch 另有独立判别（双方同溢出下的比较语义），不并入本判别。
