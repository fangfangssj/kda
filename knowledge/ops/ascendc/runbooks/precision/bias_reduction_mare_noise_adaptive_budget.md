---
type: "Runbook"
title: ST bias mare 预算噪声自适应 max(10×th,10×noise)
description: 归约主导 bias 输出 mare 预算由固定 10×th 改为 max(10×th,10×golden fp64 噪声)：期望值生成时逐用例现算噪声落盘 meta.json，C++ 判读逐输出预算并报告，重跑 20/20。
tags: [task.testing, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# ST bias mare 预算噪声自适应 max(10×th,10×noise)


## 现象

开发 ThnnFusedGruCellGrad 算子（对标 torch aten `_thnn_fused_gru_cell_backward` 的 GRU Cell 反向算子，aclnn 算子仓流程）的 Branch-0 Kernel ST：20 条用例中 4 条大 B fp32 用例（P7 B=32,H=2048；B0-a B=64,H=2176；B0-c B=96,H=4160；B0-e B=256,H=2176）两路 bias 输出（grad_input_bias / grad_hidden_bias）mare 超固定预算 10×th=1e-4，mere 全过。本次按修订口径重跑实测 kernel-vs-golden mare：P7 8.085e-4（两路同值）、B0-a 2.250e-4 / 1.215e-4、B0-c 7.654e-4 / 7.774e-4、B0-e 5.112e-3（两路同值）；逐元素三路输出（grad_input_gates / grad_hidden_gates / grad_hx）在全部打印用例上 mare=0.000e+00（bitwise 对齐 golden，含 4 条大 B 用例）。

当时代码状态：判读实现 `tests/aclnn/test_aclnn.cpp` 的 `CompareTensor` 内部硬编码 `r.mare < kMareRatio * threshold`（kMareRatio=10.0 局部常量），预算不可按输出区分；期望值生成器 `gen_st_oracle.py` 的 meta.json 不含噪声字段。任务约束：按用户裁定的 Validation.md「精度验收标准」修订口径更新期望值生成/判读链路并重跑全部 20 条——**不改 kernel、不改期望值位流、不改用例 shape/dtype/输入**；mere 预算与逐元素三路口径不变。

## 适用场景

- 工程形态：aclnn 算子仓 ST 数值验收（ops-registry-invoke 流程，TTK `stat_rel_err` 口径）；精度验收标准的口径修订落地与全量重跑。
- API 路径：`tests/aclnn/gen_st_oracle.py`（期望值+噪声生成）→ `build/st_data/<case>/meta.json`（噪声随期望值落盘）→ `tests/aclnn/test_aclnn.cpp`（C++ 判读装载与逐输出预算）+ `st_cases_branch0.json`（compare_semantics 同步）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN cann-9.0.0-beta.2（构建、上板 ST 20/20、TilingUT 60/60 均在该环境实测）。
- 适用粒度（transfer_scope 最窄档）：operator_family——归约主导输出的精度验收方法学；"生成时现算 → 落盘 → 逐输出预算 → max() 退化"管道可迁移到任何 golden 自身 fp32 噪声可测的算子，fp64 真值链须替换为目标算子语义。
- 路线优先级：recommended_fastpath——**前提是具备证据链**（kernel 无缺陷独立证明 + golden 噪声主导证明 + 口径经用户/规范裁定）；无证据链时禁止采用（见「坏实践」）。
- 明确不适用/未验证场景：无独立 golden/oracle 的场景（噪声无从定义）；fp16/bf16 大 B 用例的噪声形态（本批半精度用例均为小 B，未验证）；极端相消列噪声超出 mere+逐元素判据兜底范围的形态。

## 根因

固定 mare 预算 10×th 对归约主导的 bias 输出数学上不可达——golden 自身 fp32 归约噪声主导超差，机制链：

- golden 期望值链 = 输入升 fp32 → 五链 fp32 → 沿 batch fp32 累加（torch sum 内部序）→ 输出前单次收窄；kernel = 多核分块 partial + 固定序合并（设计强制 ReduceSum 拓扑）。求和拓扑差异本应由容差覆盖。
- 大 B 用例相消列上 golden 自身 vs fp64 真值的噪声 mare（threeway 口径逐输出真值实测）：P7 最差列 1684，|fp64 真值|=5.518e-05、|golden−truth|=1.786e-08 → rel = 1.786e-8 / (5.518e-5 + 1e-7) = 3.2316e-4；B0-a 8.863e-4/7.578e-5、B0-c 6.055e-4、B0-e 1.135e-3。10×噪声 = 7.578e-4~1.135e-2，全部远超原预算 1e-4——即使 kernel 做 fp64 精确计算再舍回 fp32，旧阈也过不了（修订记录记载四用例 mare 仍为 1.9e-4~5.0e-3）。
- mere 全过（4.4e-7~1.7e-6 < 1e-5）+ 逐元素三路 bitwise 对齐（本会话实测 mare=0）+ 仿真器与 kernel 输出 8/8 bitwise 一致（前会话证据记载）⇒ 超差来自 fp32 求和序噪声而非 kernel 缺陷。

## 解决配方

**before——固定预算（CompareTensor 内部硬编码，判读不可按输出区分）**：

```cpp
static OutCompare CompareTensor(const std::vector<uint8_t>& actual,
                                const std::vector<uint8_t>& expected,
                                StDtype dt, double threshold, bool exactZero)
{
    OutCompare r;
    const size_t es = (dt == StDtype::FP32) ? 4 : 2;
    if (actual.size() != expected.size() || actual.size() % es != 0) {
        r.pass = false;
        r.nanInfMismatch = -1;  // 尺寸不一致按框架错误处理（调用方已前置校验，不期望到达）
        return r;
    }
    const size_t n = actual.size() / es;
    if (n == 0) {
        r.empty = true;  // (0,) 空张量：比对短路通过（TTK _check_empty 同语义）
        return r;
    }
    if (exactZero) {
        for (size_t i = 0; i < n; ++i) {
            const double a = ElemToDouble(actual.data() + i * es, dt);
            if (!(a == 0.0)) {  // NaN/Inf/非零均不满足 equals 0.0
                ++r.nonZeroActual;
            }
            const double g = ElemToDouble(expected.data() + i * es, dt);
            if (!(g == 0.0)) {
                ++r.nonZeroOracle;
            }
        }
        r.pass = (r.nonZeroActual == 0) && (r.nonZeroOracle == 0);
        return r;
    }
    // stat_rel_err（TTK）：rel_err=|a-g|/(|g|+1e-7)；mere<th 且 mare<10*th；
    // NaN 两侧同 NaN、Inf 两侧同号，否则 nan_inf_mismatch；一致的非有限值不计入 mere/mare。
    const double kFloor = 1e-7;
    const double kMareRatio = 10.0;
    double sumRel = 0.0;
    double maxRel = -1.0;
    long long finiteCnt = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = ElemToDouble(actual.data() + i * es, dt);
        const double g = ElemToDouble(expected.data() + i * es, dt);
        const bool aNan = std::isnan(a);
        const bool gNan = std::isnan(g);
        const bool aInf = std::isinf(a);
        const bool gInf = std::isinf(g);
        const bool sameNan = aNan && gNan;
        const bool sameInf = aInf && gInf && ((a > 0.0) == (g > 0.0));
        if ((aNan || gNan || aInf || gInf) && !sameNan && !sameInf) {
            ++r.nanInfMismatch;
            continue;
        }
        if (aNan || gNan || aInf || gInf) {
            continue;  // 一致的非有限值：排除出 mere/mare
        }
        const double rel = std::fabs(a - g) / (std::fabs(g) + kFloor);
        sumRel += rel;
        ++finiteCnt;
        if (rel > maxRel) {
            maxRel = rel;
        }
    }
    r.mere = (finiteCnt > 0) ? (sumRel / static_cast<double>(finiteCnt)) : 0.0;
    r.mare = (finiteCnt > 0) ? maxRel : 0.0;
    r.pass = (r.nanInfMismatch == 0) &&
             (finiteCnt == 0 || (r.mere < threshold && r.mare < kMareRatio * threshold));
    return r;
}
```

**after——三段管道（gen 现算 → meta.json 落盘 → C++ 逐输出预算）**：

1. 生成侧（gen_st_oracle.py，期望值生成时逐用例现算噪声并随期望值落盘，代码见「完整代码」第 1 段）：`fp64_bias_truth` 逐输出构建 fp64 真值，`mare_noise` 按判读同款 rel 公式取 mare，写入 `meta.json` 的 `bias_mare_noise`。
2. C++ 判读侧（test_aclnn.cpp）：
   - `kMareRatio` 移至文件作用域并注明按输出预算；
   - `RunStCase` 在 shape 契约检查后装载 `meta.json` 的 `bias_mare_noise`（缺失/非 NUM 一律框架错误——防旧生成器产物静默退化为固定预算，代码见「完整代码」第 2 段）；
   - 预算应用（「完整代码」第 3 段）：逐元素三路 `10×th`；两路 bias `max(10×th, 10×noise)`；mere 全输出 `<th` 不变；
   - 逐输出报告 mere/mare/mare_budget 与预算来源标注（「完整代码」第 4 段）。
3. manifest 同步（st_cases_branch0.json `compare_semantics` 改为）：

```text
"stat_rel_err（同 TTK）：rel_err=|a-g|/(|g|+1e-7)；NaN 需两侧同 NaN、Inf 需两侧同号，否则 nan_inf_mismatch 判 FAIL；有限元素 mere<th 判 PASS；mare 预算按输出——逐元素三路（grad_input_gates/grad_hidden_gates/grad_hx）mare<10*th，两路 bias 输出（grad_input_bias/grad_hidden_bias）mare<max(10*th, 10*bias_mare_noise)，bias_mare_noise=golden 期望值 vs fp64 真值（同语义五链+batch 归约 fp64 域全链计算，threeway.py 口径）的 mare，由 gen_st_oracle.py 期望值生成时逐用例现算落盘 meta.json（Validation.md「精度验收标准」2026-09-06 用户裁定修订）；五路输出分别比对均须过阈。exact_zero：五路输出逐元素 == 0.0（无容差，Validation.md S1 equals 0.0）。确定性：同输入连跑 2 次五路输出位级一致。"
```

## 设计决策

- **噪声在期望值生成时现算、不写死**：噪声随 shape / 分布 / 相消形态逐用例不同（本批实测 0 ~ 3.9e-3），写死即失真；噪声随期望值一同落盘 meta.json 供 ST 判定与审计引用。
- **max() 形式而非人为阈值切换**：`mare_noise ≤ th` 时预算自动退化为原 10×th——实测其余 16 条用例中 15 条预算保持原值（1.000e-04 / 1.000e-02 / 4.000e-02，noise≤th 形态，含 exact_zero 的 S1）；P5（fp16 B=1）噪声 3.942e-3 > fp16 th=1e-3，预算抬至 3.942e-2，但该用例 kernel 输出与 golden bitwise 一致（mare=0），抬升不改变判定——体现预算由 golden 驱动而非 kernel 差异驱动。
- **fp64 真值用 numpy 全 fp64 链**（输入位流升 fp64 → 五链 fp64 → 逐输出拼接 → fp64 归约）：与证据 threeway.py 口径逐字一致并经复算逐值对齐；弃用 torch fp64 通道（torch/numpy 求和序差异 ~1e-16 量级虽可忽略，但为与证据口径一致选 numpy）。
- **C++ 侧只读不写**：meta.json 缺 `bias_mare_noise` 即报框架错误，杜绝旧生成器产物被静默判为固定预算——审计性优先于向后兼容（run.sh 每次全量重生成，正常流程不会触发）。
- **噪声计算排除 golden 非有限列**（NaN/±Inf 由 ST nan_inf_mismatch 口径管辖；fp16/bf16 收窄溢出成 Inf 的列同理——golden=Inf vs truth=有限属非有限形态，不进 rel 统计）；truth 在 golden 有限列上非有限时直接 raise（保证生成可信性）。
- **修订范围最小化**：mere 预算（<th 全输出）、逐元素三路、特殊值逐点对齐、精确零断言均不动；仅两路 bias 的 mare 预算改自适应。这是用户裁定的需求级口径修订，预算公式只依赖 golden 噪声，与 kernel 输出无关。
- 当前方案属哪档路线：recommended_fastpath（满足前置证据链时）。

## 系统信息

- 环境：Ascend950PR NPU（npu-smi 25.7.rc1.6，vector_core_cnt=64、ub 253952），CANN cann-9.0.0-beta.2；torch 2.7.1+cpu、numpy 2.5.1、CMake + GNU 13.3.0。
- 目录布局：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/`（gen_st_oracle.py、st_cases_branch0.json、test_aclnn.cpp、run.sh、build/st_data/）。
- 相关文件清单（与本次改动一一对应，基线为干净 worktree、改动未提交）：`tests/aclnn/gen_st_oracle.py`（+88 行：噪声现算与落盘）、`tests/aclnn/test_aclnn.cpp`（+103/-24 行：判读链路）、`tests/aclnn/st_cases_branch0.json`（compare_semantics）、`tests/aclnn/run.sh`（Step 3 注释一行）。

## 完整代码

第 1 段——gen_st_oracle.py 生成侧（verbatim，终版 diff 内容）：

```python
def to_np_fp64(t: torch.Tensor) -> np.ndarray:
    """torch 张量（fp32/fp16/bf16）→ numpy fp64（位保真拓宽；bf16 经 uint16 位桥）。"""
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        bits = t.view(torch.uint16).numpy().astype(np.uint32)
        return (bits << 16).view(np.float32).astype(np.float64)
    return t.numpy().astype(np.float64)


def fp64_bias_truth(go64: np.ndarray, st64: np.ndarray, H: int):
    """fp64 真值：同一语义公式（五条梯度链乘序 + 两路 bias 沿 batch 归约）fp64 域全链计算。

    乘序与 golden.py / spec formula 逐字一致（aten CUDA kernel 左结合尾序）；
    归约 = np.concatenate 后 sum(axis=0)（与证据 threeway.py 口径一致）。
    返回 (grad_input_bias 真值, grad_hidden_bias 真值)，均为 (3H,) fp64。
    """
    r = st64[:, 0 * H:1 * H]
    z = st64[:, 1 * H:2 * H]
    n = st64[:, 2 * H:3 * H]
    hx = st64[:, 3 * H:4 * H]
    hn = st64[:, 4 * H:5 * H]
    gin = go64 * (1 - z) * (1 - n * n)
    gig = go64 * (hx - n) * (1 - z) * z
    grg = gin * hn * (1 - r) * r
    ghn = gin * r
    in_gates = np.concatenate((grg, gig, gin), axis=1)
    hid_gates = np.concatenate((grg, gig, ghn), axis=1)
    return in_gates.sum(axis=0), hid_gates.sum(axis=0)


def mare_noise(golden_bias: np.ndarray, truth: np.ndarray, cid: str, name: str) -> float:
    """golden 期望值（收窄后）vs fp64 真值的 mare（Validation.md 2026-09-06 修订口径）。

    rel_err = |golden − truth| / (|truth| + 1e-7)（分母取 fp64 真值，与 ST 比对同款
    公式、与证据 threeway.py 口径一致）；golden 非有限列（NaN/±Inf：IEEE 传播 /
    半精度收窄溢出，S2/S3/S4/S6 形态）由 ST nan_inf_mismatch 口径管辖、不计入噪声
    统计（对应列在 kernel-vs-golden mare 中同样被排除）。
    """
    if golden_bias.size == 0:
        return 0.0
    finite = np.isfinite(golden_bias)
    if not finite.any():
        return 0.0
    t = truth[finite]
    if not np.isfinite(t).all():
        raise ValueError(f"{cid}: {name} fp64 真值在 golden 有限列上非有限——噪声计算异常")
    g = golden_bias[finite]
    rel = np.abs(g - t) / (np.abs(t) + 1e-7)
    return float(rel.max())
```

main 循环接线（期望值现算后、落盘前）：

```python
        # golden fp64 噪声（Validation.md 2026-09-06 修订）：期望值生成时逐用例现算、
        # 随期望值一同落盘（meta.json bias_mare_noise）——ST 判读对两路 bias 输出取
        # mare 预算 = max(10×th, 10×噪声)；不写死数值。
        if bool(case["has_bias"]):
            t_in64, t_hid64 = fp64_bias_truth(to_np_fp64(grad_hy), to_np_fp64(storage), H)
            noise_in = mare_noise(to_np_fp64(outs[3]), t_in64, cid, "grad_input_bias")
            noise_hid = mare_noise(to_np_fp64(outs[4]), t_hid64, cid, "grad_hidden_bias")
        else:
            noise_in = 0.0
            noise_hid = 0.0
```

meta.json 扩展字段：

```python
            # 两路 bias 输出的 golden fp64 噪声 mare（期望值生成时现算；
            # Validation.md「精度验收标准」2026-09-06 修订：bias mare 预算 =
            # max(10×th, 10×mare_noise)，mare_noise = |golden−fp64真值|/(|fp64真值|+1e-7)
            # 的 mare，golden 非有限列不计入；fp64 真值 = 同语义五链 + batch 归约
            # fp64 域全链计算，口径与证据 threeway.py 一致）
            "bias_mare_noise": {
                "grad_input_bias": noise_in,
                "grad_hidden_bias": noise_hid,
            },
```

第 2 段——test_aclnn.cpp 噪声装载（RunStCase 内，shape 契约检查之后；verbatim）：

```cpp
    // 2.5) golden fp64 噪声装载（Validation.md「精度验收标准」2026-09-06 修订）：
    //      两路 bias 输出 mare 预算 = max(10*th, 10*bias_mare_noise)；噪声 = golden
    //      期望值 vs fp64 真值（同语义公式 fp64 域全链计算，threeway.py 口径）的 mare，
    //      由 gen_st_oracle.py 期望值生成时逐用例现算、随期望值落盘 meta.json——
    //      此处仅读取执行，不在测试侧写死数值。
    double biasNoise[2] = {0.0, 0.0};  // [0]=grad_input_bias [1]=grad_hidden_bias
    {
        std::vector<uint8_t> metaRaw;
        if (!ReadFileBytes(dir + "/meta.json", metaRaw)) {
            res.frameworkError = true;
            res.message = "meta.json 缺失（bias 噪声随期望值落盘，须先运行 gen_st_oracle.py）";
            return false;
        }
        stjson::Value metaRoot;
        stjson::Parser metaParser(std::string(metaRaw.begin(), metaRaw.end()));
        if (!metaParser.Parse(metaRoot) || metaRoot.type != stjson::Value::OBJ) {
            res.frameworkError = true;
            res.message = "meta.json 解析失败: " + dir + "/meta.json";
            return false;
        }
        const stjson::Value* noiseObj = metaRoot.find("bias_mare_noise");
        const stjson::Value* nIn = (noiseObj != nullptr) ? noiseObj->find("grad_input_bias") : nullptr;
        const stjson::Value* nHid = (noiseObj != nullptr) ? noiseObj->find("grad_hidden_bias") : nullptr;
        if (noiseObj == nullptr || noiseObj->type != stjson::Value::OBJ ||
            nIn == nullptr || nIn->type != stjson::Value::NUM ||
            nHid == nullptr || nHid->type != stjson::Value::NUM) {
            res.frameworkError = true;
            res.message = "meta.json 缺少 bias_mare_noise（旧版生成器产物，请重跑 gen_st_oracle.py）";
            return false;
        }
        biasNoise[0] = nIn->number;
        biasNoise[1] = nHid->number;
    }
```

第 3 段——CompareTensor after（签名增 `double mareBudget`，pass 条件 `r.mare < mareBudget`；verbatim，exactZero 分支与 before 相同略）：

```cpp
static OutCompare CompareTensor(const std::vector<uint8_t>& actual,
                                const std::vector<uint8_t>& expected,
                                StDtype dt, double threshold, double mareBudget, bool exactZero)
{
    OutCompare r;
    const size_t es = (dt == StDtype::FP32) ? 4 : 2;
    if (actual.size() != expected.size() || actual.size() % es != 0) {
        r.pass = false;
        r.nanInfMismatch = -1;  // 尺寸不一致按框架错误处理（调用方已前置校验，不期望到达）
        return r;
    }
    const size_t n = actual.size() / es;
    if (n == 0) {
        r.empty = true;  // (0,) 空张量：比对短路通过（TTK _check_empty 同语义）
        return r;
    }
    if (exactZero) {
        // ……（与 before 完全相同：逐元素 == 0.0 精确断言，略）……
    }
    // stat_rel_err（TTK）：rel_err=|a-g|/(|g|+1e-7)；mere<th；mare 按输出预算判
    // （Validation.md 2026-09-06 修订：逐元素三路 10*th，两路 bias 输出
    //  max(10*th, 10*golden fp64 噪声 mare)——噪声随期望值落盘 meta.json）；
    // NaN 两侧同 NaN、Inf 两侧同号，否则 nan_inf_mismatch；
    // 一致的非有限值不计入 mere/mare。
    const double kFloor = 1e-7;
    double sumRel = 0.0;
    double maxRel = -1.0;
    long long finiteCnt = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = ElemToDouble(actual.data() + i * es, dt);
        const double g = ElemToDouble(expected.data() + i * es, dt);
        const bool aNan = std::isnan(a);
        const bool gNan = std::isnan(g);
        const bool aInf = std::isinf(a);
        const bool gInf = std::isinf(g);
        const bool sameNan = aNan && gNan;
        const bool sameInf = aInf && gInf && ((a > 0.0) == (g > 0.0));
        if ((aNan || gNan || aInf || gInf) && !sameNan && !sameInf) {
            ++r.nanInfMismatch;
            continue;
        }
        if (aNan || gNan || aInf || gInf) {
            continue;  // 一致的非有限值：排除出 mere/mare
        }
        const double rel = std::fabs(a - g) / (std::fabs(g) + kFloor);
        sumRel += rel;
        ++finiteCnt;
        if (rel > maxRel) {
            maxRel = rel;
        }
    }
    r.mere = (finiteCnt > 0) ? (sumRel / static_cast<double>(finiteCnt)) : 0.0;
    r.mare = (finiteCnt > 0) ? maxRel : 0.0;
    r.pass = (r.nanInfMismatch == 0) &&
             (finiteCnt == 0 || (r.mere < threshold && r.mare < mareBudget));
    return r;
}
```

预算应用与调用点（RunStCase 精度比对节；verbatim）：

```cpp
        // 6) 精度比对（run1 vs 独立 oracle 期望；五路分别判，均须过阈）。
        //    mare 预算按输出（Validation.md 2026-09-06 修订）：逐元素三路 10*th；
        //    两路 bias 输出 max(10*th, 10*bias_mare_noise)；mere 全输出 <th 不变。
        const double mareBudgetElem = kMareRatio * c.threshold;
        const double mareBudgetBias[2] = {
            std::max(mareBudgetElem, kMareRatio * biasNoise[0]),
            std::max(mareBudgetElem, kMareRatio * biasNoise[1]),
        };
        double mareBudget[5];
        OutCompare cmp[5];
        for (int k = 0; k < 5; ++k) {
            mareBudget[k] = (k >= 3) ? mareBudgetBias[k - 3] : mareBudgetElem;
            cmp[k] = CompareTensor(runOut[1][k], expected[k], c.dtype, c.threshold, mareBudget[k], c.exactZero);
        }
        res.precisionOk = true;
        for (int k = 0; k < 5; ++k) {
            if (!cmp[k].pass) {
                res.precisionOk = false;
            }
        }
```

第 4 段——逐输出报告（可复核；verbatim）：

```cpp
        // 8) 用例报告（逐输出 mere/mare 与所应用 mare 预算，可复核）
        printf("[ST] %s %s B=%lld H=%lld bias=%d src=%s\n", c.caseId.c_str(), DtypeName(c.dtype),
               static_cast<long long>(c.B), static_cast<long long>(c.H), c.hasBias ? 1 : 0,
               c.source.c_str());
        for (int k = 0; k < 5; ++k) {
            // 预算来源标注：bias 输出 max(10*th, 10*noise)（含噪声值）；逐元素输出 10*th
            char budgetSrc[96];
            if (k >= 3) {
                snprintf(budgetSrc, sizeof(budgetSrc), "max(10*th,10*noise=%.3e)", biasNoise[k - 3]);
            } else {
                snprintf(budgetSrc, sizeof(budgetSrc), "10*th");
            }
            if (c.exactZero) {
                printf("     %-18s %s (exact_zero: nonZeroActual=%lld nonZeroOracle=%lld)\n",
                       kStOutputNames[k], cmp[k].pass ? "PASS" : "FAIL",
                       cmp[k].nonZeroActual, cmp[k].nonZeroOracle);
            } else if (cmp[k].nanInfMismatch < 0) {
                printf("     %-18s FAIL (byte-size mismatch)\n", kStOutputNames[k]);
            } else {
                printf("     %-18s %s mere=%.3e mare=%.3e mare_budget=%.3e [%s] "
                       "th=%.1e nan_inf_mismatch=%lld%s\n",
                       kStOutputNames[k], cmp[k].pass ? "PASS" : "FAIL",
                       cmp[k].mere, cmp[k].mare, mareBudget[k], budgetSrc,
                       c.threshold, cmp[k].nanInfMismatch, cmp[k].empty ? " [empty]" : "");
            }
        }
        printf("     determinism %s | verdict %s\n", res.determinismOk ? "OK(bitwise,2runs)" : "MISMATCH",
               (res.precisionOk && res.determinismOk) ? "PASS" : "FAIL");
```

## 预防

- 精度口径修订走"证据链（kernel 无缺陷证明 + golden 噪声主导证明）→ 用户裁定 → 规范文本修订 → 判读链路同步 → 全量重跑 + 回归"的顺序，跳任何一步即投机。
- 归约主导输出的 mare 类判据在**设计期**就评估 golden 自身 fp32 噪声（期望值生成时现算），避免验收期才发现预算数学上不可达。
- 判读链路消费的任何阈值类字段要求生成侧显式落盘（缺失即框架错误），不设静默默认值。
- 验收输出逐用例打印 mere/mare 与所应用预算（可复核），预算来源（10×th 还是 10×noise）显式标注。

## 验证信号

以下信号来自 ThnnFusedGruCellGrad Branch-0 ST 判读口径更新开发会话：期望值生成器与 C++ 判读链路按修订口径改造后，全量重跑 20 条 ST 用例与回归项，全部为该会话内工具执行实测。

1. 期望值位流不变：新旧 st_data 4 用例 × 4 文件（grad_hy.bin / storage.bin / exp_grad_input_bias.bin / exp_grad_hidden_bias.bin）逐位 SAME（来源平台：950；证据等级：tool_observed）。
2. 噪声现算与 threeway 口径独立复算一致：4 用例 input/hidden 共 8 值全部吻合（如 P7 3.231617e-4/3.231617e-4、B0-e 1.135488e-3/1.135488e-3）（来源平台：950；证据等级：tool_observed）。
3. 构建通过：`( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package )` 退出 0（SUCCESS）（来源平台：950；证据等级：tool_observed）。
4. ST 全量重跑 20/20 PASSED、run.sh 退出 0：4 条原超差用例 bias mare 全部落入新预算——P7 8.085e-4 < 3.232e-3（noise=3.232e-4）、B0-a 2.250e-4/1.215e-4 < 8.863e-3/7.578e-4、B0-c 7.654e-4/7.774e-4 < 6.055e-3、B0-e 5.112e-3 < 1.135e-2（noise=1.135e-3）；mere 全部 <1e-5；逐输出 mere/mare/预算逐用例打印（来源平台：950；证据等级：tool_observed）。
5. 预算退化性质实测：其余 16 条用例中 15 条 mare_budget 保持原值（1.000e-04 / 1.000e-02 / 4.000e-02，noise≤th 形态）；P5（fp16）预算抬至 3.942e-2（noise=3.942e-3）但其 kernel mare=0.000e+00（bitwise）（来源平台：950；证据等级：tool_observed）。
6. 确定性：20 用例同输入连跑 2 次五路输出位级一致（determinism OK(bitwise,2runs)）（来源平台：950；证据等级：tool_observed）。
7. 回归全绿：TilingUT 60/60（退出 0）；四项 ops-impl-review 脚本（scalar_mainchain / vf_reachability / workspace_consistency / non3510_scaffold）退出码全 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 前置条件（不满足即禁用本配方）：①kernel 无缺陷的独立证据（bitwise 仿真对齐，或逐元素路 bitwise——本会话实测逐元素三路 mare=0）；②golden 噪声主导证明（fp64-exact 也无法过旧阈）；③口径修订经用户/规范裁定并落规范文本。三者缺一即属"跑不过就抬预算"。
- 噪声必须独立于 kernel 输出计算（golden 期望值 vs fp64 真值）；任何"按 kernel 差异调噪声/预算"的变体禁止。
- 未验证场景：fp16/bf16 大 B 用例的噪声形态（本批半精度用例均为小 B）；极端相消列噪声超出 mere+逐元素判据兜底范围的形态（Validation.md 记载该形态下 mare 统计无判别力，由 mere<th 与逐元素 max_relative 判据承担）。
- 跨算子迁移：五链公式（gin/gig/grg/ghn）为 GRU Cell 反向特有——迁移时替换为目标算子语义的 fp64 真值链；"生成时现算 → meta 落盘 → 逐输出预算 → max() 退化"管道与判读报告格式通用。其他平台（950DT 等）未验证，推断可迁移（判读链路为平台无关 C++/Python）。

## 坏实践

- 无证据链抬预算：未证明 kernel 无缺陷、未证明噪声主导，直接把预算调到当前 mare 之上过测试——banned（测试侧放松）。
- 按 kernel 输出差异调噪声/预算——预算必须 golden-driven（本配方预算公式不看 kernel 输出）。
- 改期望值位流 / 参考实现语义 / 用例 shape/dtype/输入来"对齐"（本配方实测期望位流逐位不变）。
- 顺手放松 mere 或逐元素判据（修订范围仅两路 bias 的 mare；mere<th 全输出不变）。
- 直接采信证据 JSON 噪声数值不复算——本会话实测证据 hidden 噪声失真 3~6 个数量级，采信即得 ~3942.8 的无约束预算（见同会话[bias 噪声证据失真卡](bias_evidence_json_cross_output_fp64_truth_reuse.md)）。

## 误诊

- "只有 kernel bitwise 复刻 torch casc16 求和序才能过旧阈"的路线（前会话证据文件记载，本会话未复现）：已被裁定禁止——需偏离设计强制的 ReduceSum+二分树拓扑且数值上劣于现 kernel（RMS 1.4 倍）；本会话按修订口径实施，未采用该路线（来源：前会话证据文件记载，agent_reported/hearsay，仅作背景）。
- "超差源于 kernel 归约实现缺陷"的初始怀疑（前会话）：被仿真器 8/8 bitwise 一致否决；本会话实测逐元素三路输出 mare=0.000e+00 佐证 kernel 逐链精确（来源：本会话 tool_observed + 前会话证据记载）。

## 未决事项

- spec.yaml `numerical_tolerance` 是否同步登记本次修订留待后续裁定（Validation.md 一致性登记记载：被修订的 mare<10×th 属 TTK stat_rel_err 框架统计惯例而非 spec 条目，逐元素判据仍满足）。
- B0-e kernel mare（5.112e-3）为噪声（1.135e-3）的 4.5 倍：证据记载为单列极值统计形态（"统计运气"），本会话未逐列下钻复算——若后续实测 mare 逼近预算（1.135e-2），应先做逐列归因再考虑口径。
- 证据 JSON（branch0_bias_mare_evidence.json）中 grad_hidden_bias 噪声数值失真未修正（见同会话[bias 噪声证据失真卡](bias_evidence_json_cross_output_fp64_truth_reuse.md)）。
