---
type: "Runbook"
title: bias 归约 NaN/Inf 不匹配：±FLT_MAX 输入溢出非精度缺陷
description: 黑盒极端 grad_hy 数据范围（±FLT_MAX/±bf16_max）下 fp32 batch 归约溢出，NaN/Inf 分类随求和顺序变化，比较器逐位检查 FAIL；有限位置达标即非 kernel 缺陷。复现增量：RoPE 乘加结构同形态——golden 自溢出探针（has_nan/has_inf 判别式）+ 正常范围对照组。
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T01:12:00Z'
platforms: ["950"]
---

# bias 归约 NaN/Inf 不匹配：±FLT_MAX 输入溢出非精度缺陷

## 现象

对 GRU 单时间步融合反向算子（ThnnFusedGruCellGrad，对标 aten::_thnn_fused_gru_cell_backward，五路输出：grad_input_gates / grad_hidden_gates / grad_hx / grad_input_bias / grad_hidden_bias）跑 TTK aclnn 黑盒全量：正向 559 条中 2 条 FAIL，均为 `hasBias=true` 的 bias 归约输出——

- `aclnnThnnFusedGruCellGrad_L0_030`（float32，B=235，H=1652）：precision 逐输出为 `PASS,PASS,PASS,FAIL,FAIL`，两路 bias 输出（output[3] grad_input_bias、output[4] grad_hidden_bias）FAIL；
- `aclnnThnnFusedGruCellGrad_L1_212`（bfloat16，B=13，H=2048）：`PASS,PASS,PASS,FAIL,PASS`，仅 grad_input_bias FAIL。

三路逐元素门梯度输出全部 PASS。黑盒 CSV 给这两条用例的 `input_data_ranges` 里 grad_hy 设的是全量程：L0_030 为 `((-3.4028235e+38, 3.4028235e+38), (-1, -0.01))`，L1_212 为 `((-3.38e+38, 3.38e+38), (0, 0.001))`——即 ±FLT_MAX / ±bf16 接近 max。单用例复现（`ttk aclnn --dump full --dump-format npy`）确认 FAIL 稳定复现，golden.py 比较钩子报 `NaN/Inf mismatch`。当时代码状态：kernel 与 golden 均为已通过 ST 与白盒验证的版本，本会话未改 kernel。

## 适用场景

- **工程形态**：aclnn 黑盒精度执行（TTK aclnn 模式 + golden.py TestSpec `compare` 钩子）；ST 框架（test_aclnn.cpp）内建同款 NaN/Inf 逐位检查，理论同形态（其用例目前用受控范围未触发）。
- **API 路径**：`python3 -m ttk aclnn -i <csv> -t <case> --plugin <golden.py> --dump full --dump-format npy`；比较器 `stat_rel_err` 语义（`mere<th`、`NaN 同位/Inf 同号` 逐位判定、th 表 fp32 2^-13 / fp16 2^-10 / bf16 2^-7、mare 预算 10×th 或 max(10×th, 10×noise)）。
- **验证过的平台与 CANN 版本**：Ascend950PR（result.csv soc=Ascend950PR_9579，上板 ASCEND_ONBOARD）+ CANN cann-9.0.0-beta.2 —— 与 frontmatter `platforms: ["950"]` 严格一致。
- **适用粒度（transfer_scope）**：platform_wide——根因链是「IEEE fp 溢出 + 归约求和顺序 + TTK 比较器 NaN/Inf 逐位语义」，不绑定本算子；任何含 batch 维归约输出（bias 梯度等）的算子在黑盒 CSV 极端 `input_data_ranges` 下都会遇到。注意：验证仅覆盖单一算子、fp32/bf16 两个 dtype、TTK aclnn 模式（见失效边界）。
- **路线优先级**：recommended_fastpath——先按本文判别流程确认「非 kernel 精度缺陷」，再走验收口径处置（用户裁定），禁止直接动 kernel 或比较器。
- **明确不适用 / 未验证场景**：fp16 路径未验证；kernel 直调（GEIR）模式未验证；逐元素输出路径不适用（golden 与 kernel 同乘序、无归约口径差，NaN/Inf 逐位对齐仍应成立）；ST/白盒用例当前用受控数据范围不触发本形态。

## 判别方法

**第一步：单用例复现并 dump 全量张量**（dump 文件落在 TTK `switches.root_path`，默认 `/tmp/ops-test-kit/`，命名 `<testcase>_output_N.npy` / `_golden_N.npy` / `_input_tensor_N.npy`；注意 `--dump out,golden` 不含输入张量且首跑可能是 .bin，须 `--dump full --dump-format npy`）：

```bash
source /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/docs/ThnnFusedGruCellGrad/develop/env.sh
PKG=/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/operators/ThnnFusedGruCellGrad_package
OUTDIR="$PKG/tests/blackbox/ttk"
MERGED="$OUTDIR/blackbox_all.csv"
GOLDEN=/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/docs/ThnnFusedGruCellGrad/develop/golden.py
VENDOR="$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad"
cd "$TTK"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"
rm -f /tmp/opencode/repro_l0_030.csv
rm -f /tmp/ops-test-kit/aclnnThnnFusedGruCellGrad_L0_030_*.bin
env PYTHONPATH="$TTK:$(dirname "$GOLDEN"):$ASCEND_TOOLKIT_HOME/python/site-packages" \
    "$PY" -m ttk aclnn -i "$MERGED" -t "aclnnThnnFusedGruCellGrad_L0_030" \
    --pc=1 --task-prof false --run 1 --warmup false --device-whitelist 0 \
    --plugin "$GOLDEN" -o /tmp/opencode/repro_l0_030.csv --dump full --dump-format npy 2>&1 | tail -8
```

**第二步：对失败输出做 NaN/Inf 位级分类统计**（脚本见「完整代码」），得到 both-NaN / both-Inf-同号 / mismatch 三类计数与 mismatch 位置明细，并检查输入张量的有限性与实测 min/max。

**第三步：双有限位置算 mere/mare + fp64 真值对比**（同脚本），判别信号表：

| 信号 | 判别含义 |
|---|---|
| 输入 grad_hy 实测 min/max ≈ ±FLT_MAX（fp32 用例）或 ±3.38e38（bf16 用例），且全部有限（NaN=0, Inf=0） | 溢出前提成立：非有限输出不是注入的，是归约溢出产生的 |
| bias 输出大量非有限，且 kernel 与 golden 双方都有 | 双方在溢出位都产生"垃圾"，不是 kernel 单方面越界 |
| mismatch 集中在非有限位（NaN-vs-±Inf、±Inf 符号相反、一侧有限一侧 ±Inf/NaN） | 溢出位的取值由求和顺序决定，逐位对齐判据不可满足 |
| 双有限位置 mere/mare 远低于 th | 有限值精度无缺陷 |
| kernel-vs-fp64truth 的 mare ≈ golden-vs-fp64truth（mare_noise） | kernel 与 golden 对真值同级接近，golden 在溢出位同样不可靠 |

## 根因

黑盒用例把 grad_hy 的 `input_data_ranges` 设为 ±FLT_MAX/±bf16 接近 max 的全量程，fp32 域沿 batch 的 bias 归约部分和溢出，溢出位的 NaN/±Inf/有限大值分类由求和顺序决定，而 kernel（多核分块 partial + 固定序合并）与 golden（torch sum 内部求和序）顺序不同，导致比较器的「NaN 同位 / Inf 同号」逐位检查 FAIL——这不是 kernel 精度缺陷。

机制算式（不跳步）：

- **L1_212（bf16，B=13）为什么只有 grad_input_bias 溢出**：storage 范围 (0, 0.001) → z, n ∈ (0, 0.001)，于是 `(1-z)·(1-n²) ≈ 1`，`gin = go·(1-z)·(1-n²) ≈ go ≈ ±3.376e38`；13 行求和 `13 × 3.376e38 ≈ 4.39e39 > fp32 max 3.4028235e38` → 部分和溢出 ±Inf；同列正负大值混合时 `+Inf + (-Inf) = NaN`。而第三段不同的另一路 `ghn = gin·r ≈ ±3.376e38 × 0.001 ≈ ±3.376e35`，13 行和 `≈ 4.39e36 < 3.4e38` 不溢出——所以 grad_hidden_bias（第三段为 ghn）6144/6144 全有限通过，grad_input_bias（第三段为 gin）溢出 FAIL。
- **L0_030（fp32，B=235）为什么几乎全 NaN**：storage 范围 (-1, -0.01)，`gin ≈ go ≈ ±3.40e38`，235 行中任意两行同号即溢出；实测 4950/4956 列双方 NaN、2 列双方同号 Inf、4 列一方 NaN 一方 -Inf（如 kernel=-Inf vs golden=NaN：一方部分和先到 -Inf、另一方 +Inf 与 -Inf 相加成 NaN，纯顺序差异）。
- **bf16 dump 首字节佐证**：`b'~\x7f'` = little-endian `0x7F7E` → `2^127 × (1 + 126/128) ≈ 3.376e38`，正是 bf16 可表示的接近 max 的值。
- **mismatch 分解（L1_212 grad_input_bias，由实测计数推出）**：mismatch 446 = 双方均非有限但分类不同 103（如 kernel=inf vs golden=-inf）+ 仅 kernel 非有限 172（1288 − 1013 − 103）+ 仅 golden 非有限 171（1287 − 1013 − 103）；后两类是"一侧有限、一侧溢出"——顺序决定谁溢出，两类都真实出现在样本中（idx 4106: kernel=-7.21e37 有限 vs golden=inf；idx 4119: kernel=nan vs golden=3.30e38 有限）。

## 设计决策

- **为什么不修 kernel**：让 kernel 复刻 golden（torch sum）的求和序在溢出位分类上逐位一致，需要放弃多核分块归约拓扑，违背设计（spec invariant `bias_reduce_associative_along_batch`、`tolerance_inherit: true` 已声明求和拓扑差异由容差覆盖），且对正常输入毫无收益。
- **诊断顺序**（ascendc-precision-debug 的 Reduce 类前置判断纪律的实践）：先复现+dump 拿到 NPU 实际输出，再与 golden / fp64 真值逐位置比对差异分布，最后才下"是否 kernel bug"的结论——本例若跳过 dump 直接按失败报告的"bias 归约精度问题"去查 kernel 精度链路，会完全走错方向。
- **处置选项对比**（会话内逐一评估过）：A. 比较器对 bias 输出全量跳过 NaN/Inf 检查（实际实施，见坏实践）；B. 仅豁免"双方均非有限"的分类分歧、保留"一侧有限一侧非有限"的 FAIL（更窄更安全，未实施）；C. 修 kernel 复刻求和序（否决，违设计）；D. 收窄用例数据范围（否决，任务明令"不改用例"）。正确路径应是像 2026-09-06 bias mare 预算修订那样走用户裁定（该修订因 golden fp32 噪声主导而把预算改为 max(10×th, 10×mare_noise)，有先例可循）。

## 系统信息

- 工具链：CANN cann-9.0.0-beta.2（`source docs/ThnnFusedGruCellGrad/develop/env.sh` 进 ASCEND_OPP_PATH / set_env）；TTK 位于 `/tmp/ops-test-kit`（`python3 -m ttk`）；Python 3.12.9 + numpy + ml_dtypes。
- 目标平台：Ascend950PR（result.csv soc=Ascend950PR_9579；上板 ASCEND_ONBOARD，Device 0）。
- 目录布局：工作区 `ops-registry-invoke-glacier/`；算子包 `operators/ThnnFusedGruCellGrad_package/`；黑盒 `tests/blackbox/ttk/`（blackbox_all.csv / result.csv / result_l2.csv）；golden `docs/ThnnFusedGruCellGrad/develop/golden.py`（TTK TestSpec，双注册 ThnnFusedGruCellGrad / aclnnThnnFusedGruCellGrad）；dump 落 `/tmp/ops-test-kit/`。
- 相关文件清单：`docs/ThnnFusedGruCellGrad/develop/golden.py`（compare 钩子）、`tests/blackbox/ttk/blackbox_all.csv`（input_data_ranges）、`docs/ThnnFusedGruCellGrad/design/Validation.md`（精度验收标准）、`tests/aclnn/test_aclnn.cpp`（ST 侧同款检查）、临时分析脚本（/tmp/opencode/analyze_bias.py、check_mismatch.py、check_l1_212.py、verify_fix.py）。

## 完整代码

比较器修复前的判定核心（`docs/ThnnFusedGruCellGrad/develop/golden.py` `_stat_judge`，verbatim，NaN/Inf 检查先于容差判定、mere/mare 只算双有限位）：

```python
def _stat_judge(actual, golden, th, mare_budget):
    """单输出 stat_rel_err 同构判定（fp64 域计算；公式/FLOOR/NaN-Inf 语义逐字
    镜像 comparison/stat_rel_err.py，仅 mare 预算按输出传入）。"""
    if actual is None:
        return {"pass": False, "precision": "FAIL", "error_info": "NO_OUTPUT"}
    if golden is None:
        return {"pass": True, "precision": "PASS", "error_info": None}
    a = _to_fp64(actual).reshape(-1)
    g = _to_fp64(golden).reshape(-1)
    if a.size == 0 and g.size == 0:
        return {"pass": True, "precision": "PASS", "error_info": None}
    if a.size == 0 or g.size == 0:
        return {"pass": False, "precision": "FAIL", "error_info": "empty output/golden mismatch"}
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - g) / (np.abs(g) + _REL_FLOOR)
        a_nan, g_nan = np.isnan(a), np.isnan(g)
        a_inf, g_inf = np.isinf(a), np.isinf(g)
        same_nan = a_nan & g_nan  # 都 NaN（NaN 无符号）
        same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))  # 都 Inf 且同号
        mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
        if mismatch.any():
            return {"pass": False, "precision": "FAIL", "error_info": "NaN/Inf mismatch"}
        mask = np.isfinite(a) & np.isfinite(g)
        if int(mask.sum()) == 0:  # 全非有限且一致
            return {"pass": True, "precision": "PASS", "error_info": None}
        mere = float(rel[mask].mean())
        mare = float(rel[mask].max())
    passed = mere < th and mare < mare_budget
    info = None
    if not passed or mare_budget > _MARE_RATIO * th:
        exceeded = []
        if mere >= th:
            exceeded.append(f"mere({mere:.2e}>={th:.2e})")
        if mare >= mare_budget:
            exceeded.append(f"mare({mare:.2e}>={mare_budget:.2e})")
        info = (
            f"mere={mere:.3e} mare={mare:.3e} th={th:.3e} mare_budget={mare_budget:.3e}"
            + (f" [{' , '.join(exceeded)}]" if exceeded else " [noise-budget in effect]")
        )
    return {"pass": passed, "precision": "PASS" if passed else "FAIL", "error_info": info}
```

诊断探针核心（/tmp/opencode/check_mismatch.py 的 mismatch 分类段 + check_l1_212.py 的 bf16 装载，verbatim；脚本尾部 CSV 范围打印等在事件流中被截断，未完全恢复）：

```python
"""Check exact NaN/Inf mismatch between kernel and golden bias outputs."""
import numpy as np

DUMP_DIR = "/tmp/ops-test-kit"

def load_npy(name):
    return np.load(f"{DUMP_DIR}/{name}.npy", allow_pickle=True)

for case in ["aclnnThnnFusedGruCellGrad_L0_030"]:
    for idx, label in [(3, "grad_input_bias"), (4, "grad_hidden_bias")]:
        out = load_npy(f"{case}_output_{idx}")
        gold = load_npy(f"{case}_golden_{idx}")
        print(f"\n=== {case} output[{idx}] {label} ===")
        a = out.astype(np.float64).reshape(-1)
        g = gold.astype(np.float64).reshape(-1)
        a_nan, g_nan = np.isnan(a), np.isnan(g)
        a_inf, g_inf = np.isinf(a), np.isinf(g)
        same_nan = a_nan & g_nan
        same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))
        mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
        print(f"  total elements: {a.size}")
        print(f"  both NaN: {same_nan.sum()}")
        print(f"  both Inf same sign: {same_inf.sum()}")
        print(f"  mismatch positions: {mismatch.sum()}")
        if mismatch.any():
            midx = np.where(mismatch)[0]
            for i in midx[:20]:
                print(f"    idx={i}: kernel={a[i]} golden={g[i]} "
                      f"a_nan={a_nan[i]} g_nan={g_nan[i]} a_inf={a_inf[i]} g_inf={g_inf[i]}")
```

fp64 真值对比（/tmp/opencode/verify_fix.py 的位保真拓宽与 bias 真值，verbatim；bf16 装载同 check_l1_212.py 的 ml_dtypes 视图）：

```python
def to_fp64(arr):
    if arr.dtype == np.float64:
        return arr
    if arr.dtype == np.float32:
        return arr.astype(np.float64)
    raw = np.ascontiguousarray(arr).view(ml_dtypes.bfloat16)
    bits = raw.contiguous().view(np.uint16).astype(np.uint32)
    return (bits << 16).view(np.float32).astype(np.float64)

def fp64_bias_truth(go64, st64):
    hsz = st64.shape[1] // 5
    r = st64[:, 0*hsz:1*hsz]; z = st64[:, 1*hsz:2*hsz]; n = st64[:, 2*hsz:3*hsz]
    hx = st64[:, 3*hsz:4*hsz]; hn = st64[:, 4*hsz:5*hsz]
    gin = go64 * (1 - z) * (1 - n*n)
    gig = go64 * (hx - n) * (1 - z) * z
    grg = gin * hn * (1 - r) * r
    ghn = gin * r
    in_gates = np.concatenate((grg, gig, gin), axis=1)
    hid_gates = np.concatenate((grg, gig, ghn), axis=1)
    return in_gates.sum(axis=0), hid_gates.sum(axis=0)
```

## 预防

- 黑盒/ST 用例 CSV 出现 ±dtype_max 级 `input_data_ranges` 且算子含沿该维归约的输出（bias 梯度、mean 等）时，先预期"归约输出会溢出、NaN/Inf 逐位对齐不可满足"，再谈精度——不要按"bias 归约精度问题"直接进 kernel 精度链路排查。
- 写比较钩子时区分两类非有限来源：注入型（输入含 NaN/Inf，链路确定传播，逐位对齐可判定）与溢出生成型（有限输入求和溢出，分类顺序依赖，逐位对齐不可判定）；判据要按来源分开。
- 对归约输出做 mere/mare 统计时，务必先看双有限位置计数：计数为 0 时 mere/mare=0 是空集平凡值，不是"位级一致"。

## 验证信号

来源：某 GRU 反向算子黑盒失败修复会话（TTK aclnn 黑盒执行 + dump 位级分析 + fp64 真值对比；上板 Ascend950PR）。以下均为该会话内的实测：

- L0_030 单用例复现 FAIL：`GOLD: PASS,PASS,PASS,FAIL,FAIL`、`PRECISION_STATUS: FAIL`（来源平台：950；证据等级：tool_observed）
- L0_030 两路 bias 输出（各 4956 元素）：both NaN=4950、both Inf 同号=2、mismatch=4（idx 1694/1899: kernel=-inf golden=nan；idx 2096/3264: kernel=nan golden=-inf）；输入全有限，grad_hy 实测 min/max=±3.4028234663852886e+38（|val|>1e30 共 127542/388220），storage 实测 (-1, -0.01)（来源平台：950；证据等级：tool_observed）
- L0_030 两路 bias 输出双有限位置=0（kernel、golden 双方全非有限），finite-kernel/golden/truth 计数均为 0（来源平台：950；证据等级：tool_observed）
- L1_212 output[3] grad_input_bias（6144 元素）：kernel NaN=110+Inf=1178、golden NaN=0+Inf=1287、both Inf 同号=1013、both NaN=0、mismatch=446（样本含 idx 4106 kernel=-7.21e37(有限)/golden=inf、idx 4119 kernel=nan/golden=3.30e38(有限)、idx 4163 kernel=inf/golden=-inf）；双有限位置 mere=1.9645e-06、mare=5.2356e-03，均低于 th_bf16=7.8125e-03（来源平台：950；证据等级：tool_observed）
- L1_212 output[4] grad_hidden_bias：6144/6144 全有限（双方 0 NaN 0 Inf），mere=9.3540e-07、mare=5.7471e-03，通过（来源平台：950；证据等级：tool_observed）
- L1_212 output[3] fp64 真值对比（双有限位置）：kernel-vs-truth mare=4.2452e-03，golden-vs-truth（mare_noise）mare=4.2452e-03——kernel 对 fp64 真值的误差与 golden 自身噪声同级（来源平台：950；证据等级：tool_observed）
- 诊断确认信号：对 bias 输出停用 NaN/Inf 逐位检查后，两条用例均转 `PASS,PASS,PASS,PASS,PASS`，全量正向 total=559 pass=559 fail=0 skip=0、负向 49 条全部按预期被拒（pass=0）——证明 FAIL 确由 NaN/Inf 检查触发而非 mare/mere 超差（来源平台：950；证据等级：tool_observed；该处置本身的合规评估见「坏实践」）

## 失效边界

- 本卡数值（4/4956、446/6144、mere/mare 等）仅是上述两条用例的实测，不可当通用预期；机制（溢出+顺序依赖）是结论主体。
- fp16 路径未验证（±fp16_max 输入下的归约溢出形态未实测）；kernel 直调（GEIR）模式未验证。
- ST 框架与白盒当前用受控数据范围（ST 用 uniform/forward_reachable 分布、白盒 30 条 extreme_data_range 计数为 0），不触发本形态；但 ST 的 test_aclnn.cpp 内建同款 NaN/Inf 检查，若其用例范围放开同样会踩。
- 机制推断可迁移至任何含 batch 维归约输出 + 极端输入范围用例的算子（其他算子未验证）；「kernel 多核 partial+merge 与 golden torch sum 求和序不同」的拓扑归因来自设计文档与 kernel 侧头注释，本会话未读 op_kernel 源码逐行核对。
- 双有限位置为 0 时 mere/mare 无判别力（空集平凡值），此时唯一依据是 NaN/Inf 分类统计与输入范围。

## 坏实践

- **把该形态当 kernel 精度缺陷去"修 kernel 对齐 golden 求和序"**：溢出位分类顺序依赖，任何独立实现的求和序都可能不同；为逐位一致放弃多核归约拓扑，得不偿失且违背 spec 的归约容差声明。
- **本会话实际采用的处置是比较器豁免，属测试侧放松，不得作为解决配方复用**：`golden.py` `_stat_judge` 加 `is_bias_reduce` 形参，对 bias 输出（idx≥3）`if not is_bias_reduce and mismatch_cnt: FAIL`——即**全量**跳过 NaN/Inf 不匹配检查。其代价有实测佐证：L1_212 的 446 处 mismatch 中 343 处（172 仅 kernel 非有限 + 171 仅 golden 非有限）是"一侧有限、一侧非有限"，修复后这些位置既不触发 mismatch 检查、又被 mere/mare 的 mask 排除，完全不受任何检查（修后该用例 PASS 即为行为证明）——正常输入下 kernel 若真产生 spurious NaN/Inf 的 bias 输出，将无法被该比较器发现。会话内推导本已识别"全量跳过过于宽松"并给出更窄方案（仅豁免双方均非有限的分类分歧，保留一侧有限一侧非有限的 FAIL），最终仍实施了宽豁免。验收口径类变更的正路是用户裁定（同算子 2026-09-06 bias mare 预算修订即走此流程），且本次任务约束原文即为"不改用例、不改 golden fp32 语义、不放宽容差"。
- **把空集的 mere/mare=0 当"位级一致"证据**：双有限位置为 0 时统计无意义（见误诊第 3 条）。

## 误诊

- **初始假设（失败报告自带）"bias 归约链路 float32 路径存在精度问题"**：错误。双有限位置 mere ≤ 2e-6、mare ≤ 5.7e-3，全部达标；kernel 对 fp64 真值的误差与 golden 自身噪声同级。
- **"aclnn 模式下 compare 钩子拿不到输入张量 → NOISE UNAVAILABLE → 噪声自适应预算失效、默认预算过紧导致 FAIL"（会话内 agent 推测，agent_reported）**：与复现日志矛盾——L0_030 复现日志为 `bias mare noise in=0.000e+00 hid=0.000e+00 (noise-budget in=1.221e-03 hid=1.221e-03)`，**无** `NOISE UNAVAILABLE` 后缀，说明噪声是实际算出的 0.0（golden bias 全非有限时噪声函数在"无有限列"处早退返回 0.0），该复现跑中输入张量可得；冒烟日志里的 `NOISE UNAVAILABLE -> default budget` 出自 hasBias=false 的用例（空 bias 输出触发形状守卫），不能证明输入张量缺失。且预算与本 FAIL 无关：FAIL 在 NaN/Inf 检查即触发，未进入 mare 判定。
- **"kernel 与 golden 位级一致（mare=0）"（首轮分析脚本解读，agent_reported）**：空集平凡值——L0_030 双有限位置为 0，mere/mare=0.0000e+00 是空 mask 下的默认返回；随后的 `ValueError: attempt to get argmax of an empty sequence` 即其暴露。位级一致的正当证据只能来自双有限位置的逐位比较或 NaN/Inf 分类统计（本例 4952/4956 位分类一致）。

## 未决事项

1. **优先项（验收口径未闭合）**：设计文档「已知口径差（容差覆盖项，非 bitwise 对齐项）/不要求 bias 列 bitwise 对齐」是否覆盖溢出位的 NaN/Inf 分类分歧，与「特殊值逐点对齐（NaN 同位/Inf 同号，不走容差）」条款的适用边界存在口径空洞——本会话由开发 agent 单方面解释并落码。应复现本卡场景并提请用户裁定（先例：2026-09-06 bias mare 预算修订）；裁定前，比较器豁免版本的黑盒结果对 bias 输出 NaN/Inf 缺陷不设防。
2. 更窄豁免方案（仅当双方均非有限时豁免分类分歧，保留一侧有限一侧非有限的 FAIL）未实施未验证——可在不掩盖真实 NaN 缺陷的前提下放行顺序差分类，值得做成对照实验。
3. 黑盒 CSV 生成侧：±FLT_MAX 级输入范围是否应进精度统计主体（设计文档「域外采样仅作健壮性补充、不进精度统计主体」条款）未裁定；若裁定为域外，正确做法可能是范围收窄或该类用例单列，而非比较器豁免。
4. kernel 归约拓扑（多核 partial+merge）与 golden torch sum 的求和序差异未做源码级核对（本卡归因依赖设计文档转述）。

## 后续增量（2026-09-17，并行策展批：RoPE 乘加结构的 golden 自溢出探针与正常范围对照组）

另一条独立轨迹（RoPE 类算子 BenchHarnessApplyRotaryPosEmb，TTK kernel 模式白盒 236 条，Ascend950PR_9579 / CANN 9.2.0）复现同一判读并补三点：

**1) golden 自身溢出探针（第一步动作）**：极值 range（±3.3895e+38 / ±inf）用例的 NaN/Inf_mismatch FAIL，先取失败用例的输入 range 直接跑 golden 自身、检查输出的 has_nan/has_inf——`case00005 ranges: ((-3.3895e+38, 3.3895e+38), ...) → golden q_out: has_nan=True has_inf=True max=nan`。机制为乘加结构：|q|~3e38、|cos|~3e38 → fp32 内 `q*cos` ≈ 9e76 超 fp32 max → inf；随后 `q*cos + rotate_half(q)*sin` 出现 inf+(-inf)=NaN。golden 通路核验：插件输出与直接 torch golden 逐位一致（排除插件伪影）。

**2) 判别式**：golden has_nan/has_inf 为 True 且输入 range 达 dtype 上限 → 双方同溢出下的比较语义问题，属预期行为；golden 干净而 kernel 有 NaN/Inf → 才指向 kernel 溢出缺陷。

**3) 正常范围对照组**：同 dtype 正常 range 用例全部通过（range (-50,200)/(-0.1,0.2)/(-5,10) 的用例 PASS）——正常范围全过排除 golden/评测侧系统性问题，失败严格绑定极值 range。

### 复现增量验证信号

- 24 条 NaN/Inf_mismatch（bf16 12 + fp32 12）全部对应 ±3.3895e38 或 ±inf 极值 range；同 dtype 正常范围用例全部通过（来源平台：950；证据等级：tool_observed）。
- 极值用例 golden 自身 has_nan=True、has_inf=True、max=nan（fp32 内乘积溢出→inf→inf+(-inf)=NaN）（来源平台：950；证据等级：tool_observed）。
- golden 通路核验：插件输出与直接 torch golden 逐位一致（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 本增量覆盖"golden 同样溢出"的情形（乘积溢出 + 相消产生 NaN 的乘加结构）；golden 干净而 kernel NaN/Inf 时不适用（真缺陷线索）。
- inf（无相消）单独出现时比较器的符号/位置语义未单独验证。
- 是否应在用例设计层规避此类 range（TTK 白盒 size 扫描自动生成极值 range）属测试设计决策，本卡不裁定。
