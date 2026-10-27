---
type: "Runbook"
title: aclnn ST 自研 TTK stat_rel_err 比较器
description: TTK stat_rel_err 语义（FLOOR=1e-7、mere 与 mare 双阈值、NaN 同位/Inf 同号）在 C++ aclnn ST 中的逐字复刻实现与红基线实测。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# aclnn ST 自研 TTK stat_rel_err 比较器

## 现象

在为自定义算子（GRU Cell 反向，对标 torch `aten::_thnn_fused_gru_cell_backward`）编写 aclnn ST（系统测试）时，需要在 C++ 测试二进制里把设备端读回的输出位流与独立参考 oracle（python golden，落盘为 .bin 位流）逐张量比对，且比对口径必须与评测工具链 TTK（ops-test-kit）的 `stat_rel_err` 标准一致——否则 ST 结果与 TTK 白盒/黑盒跑分口径互相矛盾（TTK 判过的 ST 判挂，或反过来）。

当时的工作区状态：算子包 `operators/ThnnFusedGruCellGrad_package/tests/aclnn/` 已有一条最小调用通路 smoke 用例；kernel 处于接口桩阶段（只注册 tiling + barrier，不写输出）；本任务（测试先行）要求新增 manifest 驱动的数值 ST 用例，走 `aclnnThnnFusedGruCellGradGetWorkspaceSize → aclnnThnnFusedGruCellGrad` 两段式调用链。skill 文档里引用的现成测试框架头 `aclnn_st_core.h` 在整个工作区不存在（glob/find 均无果），比较器必须自己写。

TTK 的 `stat_rel_err` 语义埋在 `/tmp/ops-test-kit/ttk/core_modules/comparison/stat_rel_err.py` 源码里，与直觉差异很大：朴素写 `|a−g| ≤ atol + rtol·|g|` 的逐点容差在 NaN/Inf 处理、mare 上界、除零保护三处都会偏离 TTK 口径。

## 适用场景

- 工程形态：aclnn ST（ops-registry-invoke 算子仓 `tests/aclnn/` 下的 C++ 测试二进制）；亦适用于任何需要 host 侧 C++ 复刻 TTK 比对口径的场景。
- API 路径：TTK（ops-test-kit）`stat_rel_err` comparison 标准；aclnn 两段式调用链读回的输出缓冲。
- 验证过的平台与 CANN 版本：Ascend950PR（npu-smi 25.7.rc1.6）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致；比较器为 host 侧纯 CPU 代码，其他平台未实测，推断可迁移见失效边界）。
- 适用粒度（transfer_scope 最窄档）：`operator_family`——任意自定义算子的 aclnn ST 比较器（比对语义与算子无关，输入是两张位流）。
- 路线优先级：`recommended_fastpath`（自研 ST 无法直接调 TTK python 框架时的标准做法）。
- 明确不适用/未验证场景：直接跑 TTK 框架的场合（用 TTK 本体，不要复刻）；INT8/UINT8 等量化 dtype（`ElemToDouble` 只覆盖 fp32/fp16/bf16）；绿基线（kernel 已实现）下 mere/mare 过阈判 PASS 的路径本开发会话 未实测（本开发会话 为红基线，全部走 FAIL 分支）；TTK 其他 comparison 标准（cosine/binary/requant 等）不适用本卡语义。

## 判别方法（TTK stat_rel_err 精确语义）

从 TTK 源码（`ttk/core_modules/comparison/stat_rel_err.py`，本开发会话 内逐字读取）提炼，写 C++ 前先对齐以下六条：

1. **rel_err 定义带 FLOOR**：`rel_err = |a−g| / (|g| + 1e-7)`。FLOOR=1e-7 是算法常量（源码注释明言"不是配置"），起 near-zero 保护作用：golden=0 时 `rel_err = |a| / 1e-7 = |a| × 1e7`，例如 actual=1e-12 → rel_err=1e-5，即 golden 为 0 处一个极小的非零偏差也会被放大到阈值量级。
2. **NaN/Inf 位置匹配优先于数值**：两侧同 NaN → 匹配（NaN 无符号）；两侧同 Inf 且同号 → 匹配；其余任何 NaN/Inf 组合（含一侧 NaN 一侧有限、同 Inf 异号）→ `nan_inf_mismatch`，**整张量直接 FAIL**，mere/mare 不参与判定。
3. **一致的非有限值不计入 mere/mare**：匹配成功的 NaN/Inf 元素被排除出统计；若全部元素一致非有限（n_finite==0）→ PASS。
4. **双阈值判据**：有限元素上 `mere = mean(rel_err)`、`mare = max(rel_err)`，PASS 当且仅当 `mere < th` **且** `mare < MARE_RATIO × th`（MARE_RATIO=10）。注意 mare 上界是 10×th：mean 过阈或单点过 10×th 任一触发即 FAIL。
5. **空数组短路 PASS**：0 元素张量（如 has_bias=false 时的 (0,) bias 输出）在 compare() 入口 `_check_empty` 短路通过。
6. **阈值来源要分清**：TTK 默认按 dtype 给 `fp32 2^-13 ≈ 1.22e-4`、`fp16 2^-10 ≈ 9.77e-4`、`bf16 2^-7 ≈ 7.81e-3`；但项目精度验收表（Validation.md，rtol==atol）可覆盖为 `fp32 1e-5`、`fp16 1e-3`、`bf16 4e-3`——stat_rel_err 单阈值取 rtol。两者对 fp32 差一个数量级（1.22e-4 vs 1e-5），搞混会直接放水或误杀。

## 设计决策

- **为何逐字读 TTK 源码而不是按文档/记忆写**：FLOOR、MARE_RATIO、NaN/Inf 规则、空短路这四点在文档层面几乎从不会被完整叙述；比对器与 TTK 口径不一致的后果是 ST 与 TTK 互相矛盾，且这种不一致只有在数值贴边时才暴露，极难排查。
- **误判分支的 reporting 设计**：TTK 在首个 mismatch 即 return FAIL；本实现选择继续扫描并计数 `nanInfMismatch`（判据仍要求 ==0），为了在日志里给出"错配多少个、在哪路输出"，便于定位 NaN 传播路径——判定语义与 TTK 等价，仅日志更细。
- **exact_zero 模式（附加）**：全零期望用例（如 grad_hy 全零 → 五路输出全 0）按 spec 要求"equals 0.0 精确断言无容差"：每个元素断言 `a == 0.0`（数值相等）。**不用位级相等**的原因：golden 链 `0 × 负数` 会产生 −0.0，kernel 侧同样算式可能给 +0.0，IEEE 754 下 `−0.0 == +0.0` 为真而位级比较会误杀；同时自检 oracle 侧 `nonZeroOracle` 应为 0（oracle 自身声称全零的健全性检查）。
- **被放弃的方案**：逐点 `|a−g| ≤ atol + rtol·|g|`（无 mare 上界、NaN 行为未定义、golden=0 处语义不同）；把比较挪回 python（ST 二进制必须独立可执行，且拒绝在测试进程里再起 python）。
- 当前方案属 `recommended_fastpath` 路线。

## 系统信息

- TTK 源（语义事实源）：`/tmp/ops-test-kit/ttk/core_modules/comparison/stat_rel_err.py`、`ttk/core_modules/comparison/resolve.py`（阈值解析）。
- 实现落点：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（函数 `CompareTensor` / `ElemToDouble`，C++17）。
- 构建：`tests/aclnn/CMakeLists.txt`，链接 `ascendcl`/`nnopbase`/`cust_opapi`（vendor 安装包）；宿主 gcc，x86_64 小端。
- 运行环境：Ascend950PR + CANN 9.0.0-beta.2；`run.sh` 编排（构建包 → 构建 test → 生成 oracle 数据 → 运行）。
- 相关文件清单（与事件流实际读写一一对应）：`test_aclnn.cpp`（写）、`st_cases_branch0.json`（manifest，读）、`build/st_data/<case>/*.bin`（gen_st_oracle.py 生成、C++ 读）、`stat_rel_err.py`/`resolve.py`（只读参照）。

## 完整代码

`CompareTensor`（含 exact_zero 分支）与配套 `ElemToDouble`，verbatim 取自本开发会话 的 `test_aclnn.cpp` 写入事件（fp16/bf16 位转 float 的 `Fp16BitsToFloat`/`Bf16BitsToFloat` 见同文件，另有专卡覆盖，此处为保证自包含一并内联）：

```cpp
struct OutCompare {
    bool pass = true;
    bool empty = false;
    double mere = 0.0;
    double mare = 0.0;
    long long nanInfMismatch = 0;
    long long nonZeroActual = 0;   // exact_zero：actual 非零元素数
    long long nonZeroOracle = 0;   // exact_zero：oracle 非零元素数（自检，应为 0）
};

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

```cpp
static double ElemToDouble(const uint8_t* p, StDtype dt)
{
    switch (dt) {
        case StDtype::FP32: {
            float f = 0.0f;
            std::memcpy(&f, p, sizeof(f));
            return static_cast<double>(f);
        }
        case StDtype::FP16: {
            uint16_t h = 0;
            std::memcpy(&h, p, sizeof(h));
            return static_cast<double>(Fp16BitsToFloat(h));
        }
        case StDtype::BF16: {
            uint16_t b = 0;
            std::memcpy(&b, p, sizeof(b));
            return static_cast<double>(Bf16BitsToFloat(b));
        }
    }
    return 0.0;
}
```

（`StDtype` 为 `enum class StDtype { FP32, FP16, BF16 };`；`Fp16BitsToFloat`/`Bf16BitsToFloat` 为位级转换函数，见[fp16/bf16 位流转 float 零依赖转换卡](cxx_fp16_bf16_bitstream_to_float_conversion.md)，此处不重复。）

阈值表（manifest 侧逐用例声明，来源于项目精度验收表）：

```json
"tolerance_by_dtype": {
  "float32": { "standard": "stat_rel_err", "threshold": 1e-5 },
  "float16": { "standard": "stat_rel_err", "threshold": 1e-3 },
  "bfloat16": { "standard": "stat_rel_err", "threshold": 4e-3 }
}
```

## 预防

- 写 aclnn ST 比较器前，先 `cat` 评测工具链的 comparison 实现源码，把 FLOOR/比率常量/NaN-Inf 规则/空短路四点抄进注释，再动手写 C++。
- 阈值一律从项目精度验收表读入 manifest，不要用 TTK 默认值（两者 fp32 差一个数量级：默认 2^-13≈1.22e-4 vs 项目表 1e-5）。
- 全零期望用例单独走 exact 断言模式，并保留 oracle 侧全零自检（`nonZeroOracle`），防止用例构造错误（golden 没全零）被静默放过。

## 验证信号

来源说明：以下信号来自一次 GRU Cell 反向算子 Branch-0 aclnn ST 的开发会话（测试先行、kernel 接口桩阶段），比较器随整套 ST 首次编译运行即产出如下红基线实测。

- 编译链接通过，20 条 manifest 用例全部执行、逐张量产出 mere/mare/nan_inf_mismatch 指标（来源平台：950；证据等级：tool_observed）。
- NaN/Inf 匹配逻辑与 golden IEEE 传播逐位对齐：S2（grad_hy 单元素 NaN）实测 `nan_inf_mismatch=3/3/1/3/3`（两路门梯度各 3、grad_hx 1、两路 bias 各 3），与独立 numpy 复算的 golden NaN 位置计数完全一致；S3（storage r 平面单 NaN）实测 `1/2/0/1/2`；S4（单 +Inf）实测 `3/3/1/3/3`（来源平台：950；证据等级：tool_observed）。
- fp16 溢出 Inf 检出：S6（fp16 全 60000 输入）golden bias 输出含 6 个 Inf 位置，实测 `nan_inf_mismatch=6` 被逐位检出（来源平台：950；证据等级：tool_observed）。
- exact_zero 模式实测：S1（期望全零）`grad_input_gates FAIL (exact_zero: nonZeroActual=96 nonZeroOracle=0)`——poison 保留的非零输出被精确断言拦下，oracle 全零自检通过（来源平台：950；证据等级：tool_observed）。
- 空张量短路实测：E04（has_bias=false）两路 (0,) bias 输出无 FAIL 行（0 元素短路通过），仅三路非空输出参与比对（来源平台：950；证据等级：tool_observed）。
- 红基线总体：`Branch-0 ST summary: PASSED=0 FAILED=20 ERROR=0 TOTAL=20`，全部 20 例 `determinism OK(bitwise,2runs)`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅覆盖 fp32/fp16/bf16 三 dtype；INT/量化 dtype 需先扩展 `ElemToDouble`。
- FLOOR=1e-7、MARE_RATIO=10 是所读 TTK 版本（ops-test-kit 当前 checkout）的源内常量；TTK 升级后须复核。
- 绿基线（kernel 实现后）mere/mare 过阈判 PASS 的路径在本开发会话 未执行（红基线全走 FAIL 分支）；采用前建议先用一条已知能过的 case 冒烟。
- `finiteCnt==0`（全一致非有限）判 PASS 沿袭 TTK；若项目要求"全 NaN 也算失败"需显式收紧。
- 其他 NPU 平台未实测；比较器为 host 侧纯 CPU 代码，未验证，推断可迁移。
- 浮点累加用 double（TTK numpy 用 promote 到 float32 以上），极端大数组下 mere 与 TTK 的浮点求和顺序差异理论上可达 ulp 级——判据在阈值边界贴边时可能分歧。

## 坏实践

- 朴素逐点 `|a−g| ≤ atol + rtol·|g|`：无 mare 上界（单点大偏差被 mean 稀释）、NaN 参与比较行为未定义、golden=0 处与 TTK 的 FLOOR 语义不同——三处都偏离 TTK 口径。
- 把 TTK 默认阈值当项目验收阈值（fp32 差一个数量级，直接放水 10 倍）。
- 对全零期望用例继续走容差模式：红基线阶段若输出缓冲被清零会巧合通过（见[输出 poison 防 stub 巧合命中卡](st_red_baseline_poison_prefill_exact_zero.md)），绿基线阶段小偏差也会被 atol 吞掉，违背"equals 0.0 精确断言"的 spec 要求。

## 误诊

- 常见错误假设："stat_rel_err 就是 max relative error，单阈值 mare<th"。实际是双阈值 `mere<th 且 mare<10×th`，mean 与 max 同时受限。
- "NaN 两侧都有 NaN 就算过"：只有**同位置**双侧 NaN 才匹配；actual 全 NaN 而 golden 局部 NaN 仍 FAIL（多余 NaN 位置计入 mismatch）。
- "空 (0,) 输出会挂比较"：TTK 与本实现均短路 PASS，这是契约行为不是漏洞。

## 增量：PyTorch 侧同口径移植（独立复现，逐元素 exp 类算子，Ascend950PR + CANN 9.1.0）

同一套判定语义可原样移植到 PyTorch ST 工程的 `compare.py`（`tests/st/torch/`，torch_adapter.so 接入 NPU）：通过标准 `MERE < threshold 且 MARE < 10 × threshold`；阈值按 dtype 分档 `fp16 2^-10 / bf16 2^-7 / fp32、fp64 2^-13`（dtype 缺省回退 2^-13）；nan/±inf 不进统计、精确匹配（同 NaN、同号 Inf）否则直接 FAIL；空 tensor 优先短路 PASS；相对误差分母加 eps 防 golden=0 除零；失败时打印最大相对误差的全局下标与两侧数值便于定位。配套两点设计：

- **golden 的精度策略对齐 kernel**：fp16/bf16 输入先升 fp32 计算完整链、最后转回原 dtype——与 kernel 内部"Cast 到 fp32 计算（CAST_NONE）、结果 CAST_RINT 回原 dtype"的路径一致，避免参考实现自身引入与 kernel 不同的舍入次序造成假失败（C++ 侧的位级做法见[ST golden 多 dtype 扩展卡](st_golden_multi_dtype_bits_rne.md)）。
- 坏实践同本卡：把 inf/nan 混进相对误差统计（`(a-g)/(|g|+eps)` 对 inf/nan 得 nan，MERE 变 NaN 后比较结果不可预期，特殊值失败被吞掉）；golden 直接用输入 dtype 全程计算（fp16 中间舍入与 kernel 的 fp32 中间路径不一致）；只报一个 max_diff 不带阈值口径。

增量验证信号：

- golden 自测 19 用例 PASS（含 `exp(inf)=inf`、`exp(-inf)=0`、`exp(nan)=nan` 的 IEEE 754 传播验证）（来源平台：950（CPU golden 于 NPU 测试环境内执行）；证据等级：tool_observed）。
- NPU 36/36 用例通过：fp16/bf16 用例 MERE/MARE 均为 0.00e+00（threshold 9.77e-04 / 7.81e-03），fp32 用例 MERE 在 1e-8 量级（threshold 1.22e-04）、MARE 在 1e-7 量级（10×threshold 内），特殊值输入用例（0/+0/-0/nan/inf/-inf）与空 tensor 用例均 PASS（来源平台：950；证据等级：tool_observed）。
- 独立验收会话经 `run.sh --torch` 复跑 36/36 通过（来源平台：950；证据等级：tool_observed）。


