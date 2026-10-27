---
type: "Runbook"
title: ST 红基线输出 poison 防 stub 巧合命中
description: 接口桩 kernel 不写输出时，输出设备缓冲预填有限哨兵（fp32 3e30/fp16 30000/bf16 截断 3e30），防全零期望用例在红基线意外通过。复现增量：0xFF 全 NaN 哨兵在 golden 全有限用例集上的红/绿/四分支闭环 + special 标签 NaN 感知比对化解 0xFF 与 NaN 期望的碰撞面。复现增量：aclrtMalloc 新分配常表现为全零——不预填也会假绿（S6 实测拆除）；-999 与 0xCD 两个哨兵变体；绿基线无扰升级为实测。复现增量：0xFF 字节哨兵对 IEEE 浮点是 NaN 位型，NaN 期望用例上 no-op stub 假通过（实测）；NaN 期望集合必须用有限位型（0x41）。复现增量：aclrtMalloc 新分配（HUGE_FIRST 首页）常表现为全零——不显式清零也会假绿；-999/0xCD 哨兵变体均实测有效，绿基线（真实 kernel）517 用例级验证 poison 不干扰正常写回比对。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# ST 红基线输出 poison 防 stub 巧合命中

## 现象

在算子开发的"测试先行"阶段（ST 用例先于 kernel 数值实现落地），被测 kernel 是接口桩：入口只做 `AscendC::InitSocState()` + `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` + `PipeBarrier<PIPE_ALL>()`，不写任何输出。此时 aclnn 两段式调用链（GetWorkspaceSize → aclnn → Sync）**全程返回 OK**，输出设备缓冲保持调用前的内容。

如果测试框架按惯常做法把输出缓冲清零（`aclrtMemset(..., 0, ...)`）再跑数值比对，会遇到一个隐蔽的坑：**期望值恰好全零的用例会"巧合通过"**。本场景的具体算子是 GRU Cell 反向（grad_hy 全零输入 → 五路输出全零的 spec 用例）：golden 链上 `gin = go·(1−z)·(1−n²)` 等五条梯度在 go=0 时全部为 0，若 actual 也被清成 0，则 mere=0 < th，比对通过——而此时 kernel 明明什么都没算。红基线验收口径"本分支数值用例期望通过数 = 0"被击穿，测试先行阶段失去了"证明比对器真的能抓住错误输出"的自证能力（即"stub 不得巧合命中 oracle"原则）。

## 适用场景

- 工程形态：aclnn ST（ops-registry-invoke 算子仓 `tests/aclnn/`），测试先行/红基线阶段（kernel 为接口桩或部分未实现）。
- API 路径：`aclrtMemcpy`（H2D poison 上载）+ aclnn 两段式调用链；输出张量由测试方构造描述符并分配设备内存。
- 验证过的平台与 CANN 版本：Ascend950PR（npu-smi 25.7.rc1.6）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`operator_family`——任何"输出为调用方分配的设备缓冲、kernel 可能不写"的自定义算子 ST 均适用。
- 路线优先级：`recommended_fastpath`（红基线诚实性的必要手段，无替代方案时直接采用）。
- 明确不适用/未验证场景：kernel 已实现后的绿基线（poison 会被全量覆写，无害但非必要——本开发会话 未验证绿基线）；输出确需保留上轮内容的增量语义算子；kernel 只写部分输出的场景（poison 仍能暴露未写区域，但"部分通过"的解读需结合用例设计）；多 stream 并发写同一输出缓冲的场景。

## 根因

接口桩 kernel 不写输出，且 **aclnn 框架层也不会清零或触碰输出缓冲**——输出设备内存保持测试方上送的原值。于是"实际输出"完全由测试方的输出缓冲初始化决定：

- 初始化为 0（memset 清零）+ 期望全零用例（如 go=0 → 五路输出全 0）→ `rel_err = |0−0|/(0+1e-7) = 0` → mere=0 < th → **PASS（假通过）**。
- 红基线的意义在于证明"比对器能抓住错误输出"；一条巧合通过就破坏了这层自证。

机制算式（为什么选有限大哨兵而不是 0 或 NaN）：

- 哨兵 s 与 golden g 的 rel_err = |s−g|/(|g|+1e-7)。取 s=3.0e30（fp32）、g∈O(0.05–2)（随机数据域）→ rel_err ≈ 3.0e30/|g| ≈ 1.5e30–6e31，mere/mare 远超任何阈值 → 稳定 FAIL。
- 取 fp16 哨兵 30000（< 65504 最大有限值，且远离 U(0,1)/U(−2,2) 数据域）→ rel_err ≈ 30000/|g|，同样稳定 FAIL。
- **不用 NaN 做哨兵**：NaN 位置匹配语义（两侧同 NaN → 匹配）下，NaN 哨兵会与特殊值用例 golden 中的 NaN 位置"同位匹配"（如单 NaN 注入用例的对应列）。虽然整体仍会 FAIL（其余位置错配），但有限哨兵把这最后一类巧合也彻底排除——特殊值用例的 NaN 位置检出必须来自 kernel 真实写入的 NaN，而不是缓冲残留。

## 解决配方

每次调用 aclnn 前，把五路输出缓冲（host 侧填充 + H2D memcpy）覆盖为按 dtype 的**有限哨兵**：

1. fp32 → `3.0e30f`；fp16 → fp16 位型(30000.0)；bf16 → 截断位型(3.0e30)。
2. 每轮调用（含确定性双跑的每一轮）都重新 poison——stub 下两轮输出同为哨兵，位级一致，不影响确定性检查；绿基线下 kernel 覆写，poison 无副作用。
3. (0,) 空输出（0 字节）跳过 poison，也跳过比对（空短路）。
4. 确定性双跑各自 fresh poison 后比对两轮输出位级一致。

为什么不用 `aclrtMemset`：字节 pattern 无法跨 dtype 表达任意位型——memset 0x7F 在 fp32 是 0x7F7F7F7F（≈3.4e38，有限），在 fp16 却是 0x7F7F（指数全 1、尾数非 0 → NaN）；host 填充 + memcpy 是唯一 dtype 精确的方式。

## 设计决策

- **哨兵值选择**：必须(a)有限、(b)远离用例数据值域与 golden 值域、(c)在该 dtype 可精确表示。fp32/bf16 用 3.0e30（bf16 与 fp32 共享指数域，3e30 可表示）；fp16 上限 65504，取 30000。
- **NaN 哨根被否决**：与"NaN 同位匹配"语义存在残余巧合面（见根因节推导），有限哨兵零巧合。
- **poison 放在每轮调用内而非套件开头**：保证确定性双跑各自从已知态出发；也让"kernel 到底写没写"在两轮日志里可读。
- **方向转换器只需截断舍入**：哨兵只须"非零有限"，float→fp16/bf16 的 poison 方向转换用截断即可，无须 round-to-nearest（数值语义转换见 [fp16/bf16 位流转 float 零依赖转换卡](cxx_fp16_bf16_bitstream_to_float_conversion.md)）。
- 当前方案属 `recommended_fastpath` 路线。

## 系统信息

- 实现落点：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（函数 `FillPoison` / `FloatToFp16Bits` / `FloatToBf16Bits`，及 `RunStCase` 中的 poison 调用点）。
- 运行环境：Ascend950PR + CANN 9.0.0-beta.2；构建链 tests/aclnn/CMakeLists.txt（ascendcl/nnopbase/cust_opapi）。
- 相关文件：`test_aclnn.cpp`（写）、`st_cases_branch0.json`（用例清单，S1 为 exact_zero 全零期望用例）、`gen_st_oracle.py`（golden 期望位流）。

## 完整代码

verbatim 取自本开发会话 的 `test_aclnn.cpp` 写入事件：

```cpp
// 输出 poison 填充：接口桩阶段 kernel 不写输出，poison 保证红基线不被
// 「device 缓冲恰为零/恰为期望」巧合命中（stub 不得巧合命中 oracle）。
static void FillPoison(std::vector<uint8_t>& buf, StDtype dt)
{
    uint16_t pat16 = 0;
    float pat32 = 0.0f;
    if (dt == StDtype::FP32) {
        pat32 = 3.0e30f;
    } else if (dt == StDtype::FP16) {
        pat16 = FloatToFp16Bits(30000.0f);
    } else {
        pat16 = FloatToBf16Bits(3.0e30f);
    }
    const size_t es = (dt == StDtype::FP32) ? 4 : 2;
    for (size_t i = 0; i + es <= buf.size(); i += es) {
        if (dt == StDtype::FP32) {
            std::memcpy(buf.data() + i, &pat32, sizeof(pat32));
        } else {
            std::memcpy(buf.data() + i, &pat16, sizeof(pat16));
        }
    }
}
```

```cpp
// float → fp16 / bf16 位流（仅用于输出 poison 填充；截断舍入即可，只须非零有限）
static uint16_t FloatToFp16Bits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    const uint32_t expF = (x >> 23) & 0xFFU;
    const uint32_t man = x & 0x7FFFFFU;
    if (expF == 0xFFU) {
        return static_cast<uint16_t>(sign | 0x7BFFU);  // Inf/NaN → 最大有限（poison 保持有限）
    }
    const int32_t exp = static_cast<int32_t>(expF) - 127 + 15;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7BFFU);
    }
    if (exp <= 0) {
        return static_cast<uint16_t>(sign | 0x0001U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

static uint16_t FloatToBf16Bits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t bits = x >> 16;
    if (bits == 0x7F80U || bits == 0xFF80U) {
        bits = 0x7F7FU;  // Inf/NaN → 有限大值
    }
    return static_cast<uint16_t>(bits);
}
```

调用点（`RunStCase` 内，每轮执行前）：

```cpp
// 5) 同输入连跑 2 次（确定性必检项）：每次 poison 输出 → 完整两段式 aclnn → 读回
std::vector<uint8_t> runOut[2][5];
for (int run = 0; run < 2 && ok; ++run) {
    // 5.1 poison 输出（红基线反巧合：接口桩不写输出时保持非期望值）
    for (int k = 0; k < 5; ++k) {
        if (outBytes[k] == 0) {
            continue;
        }
        std::vector<uint8_t> poison(outBytes[k]);
        FillPoison(poison, c.dtype);
        if (aclrtMemcpy(devOut[k], outBytes[k], poison.data(), outBytes[k],
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            res.frameworkError = true;
            res.message = "aclrtMemcpy poison 失败";
            ok = false;
            break;
        }
    }
    ...  // 描述符构造 → GetWorkspaceSize → aclnn → Sync → 读回
}
```

## 预防

- 测试先行阶段写 ST 时，输出缓冲初始化策略要过一遍"期望值反推"：枚举 manifest 里期望全零/常量的用例，确认清零初始化不会与之巧合相等；有疑义直接上 poison。
- poison 值选好后做一次量级核对：`|s|/max(|g|)` 应超出阈值至少 10 个数量级（本例 3e30/2 ≈ 1.5e30 vs 阈 1e-5）。
- 特殊值用例（NaN/Inf 注入）存在时，优先有限哨兵而非 NaN 哨兵。

## 验证信号

来源说明：以下信号来自 GRU Cell 反向算子 Branch-0 aclnn ST（测试先行、kernel 接口桩）的红基线实测，同一套 20 条 manifest 用例、Ascend950PR 上单次运行。

- S1（grad_hy 全零 → 期望五路全 0，exact_zero 断言）实测 `grad_input_gates FAIL (exact_zero: nonZeroActual=96 nonZeroOracle=0)`、`grad_hidden_gates FAIL (exact_zero: nonZeroActual=96 nonZeroOracle=0)`——输出保持 poison 非零值被精确断言拦下，oracle 侧全零自检通过；全零期望用例**未**巧合通过（来源平台：950；证据等级：tool_observed）。
- 红基线总量：`Branch-0 ST summary: PASSED=0 FAILED=20 ERROR=0 TOTAL=20`——20/20 全 FAIL，无一条数值用例意外通过（来源平台：950；证据等级：tool_observed）。
- FAIL 指标量级与 poison 保留一致（stub 未写输出的间接证据）：E01 fp32 `mere=8.254e+32 mare=5.964e+34`，与 `3.0e30/|g|`（g∈O(0.05–2)）的推导量级吻合；E02 fp16 `mare=2.280e+08` 与 `30000/|g|` 吻合（来源平台：950；证据等级：tool_observed）。
- 同输入连跑 2 次全部 20 例 `determinism OK(bitwise,2runs)`——poison 不干扰确定性检查（stub 下两轮同为哨兵位型）（来源平台：950；证据等级：tool_observed）。
- E04（has_bias=false）两路 (0,) 空 bias 输出跳过 poison 且不产生误报（空短路）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 哨兵值落入用例数据/golden 值域时会产生局部元素匹配——选值时须避开（本例数据域 U(−2,2)/U(0,1)，golden 值域同量级，3e30/30000 无交集）。
- 绿基线（kernel 实现后）poison 被全量覆写的行为未在本开发会话 验证（红基线 trace）；采用后建议绿基线首跑核对 poison 无残留。
- 若 kernel 合法地只写部分输出（如尾部块清零逻辑缺失），poison 会暴露为"未写区域保持哨兵"——这正确反映了缺陷，但需要结合逐元素 diff 定位。
- fp16 哨兵 30000 与"近最大值"类极值用例（如全 60000 输入的 golden 输出可达 ~6e4）同量级时，mare 仍远超阈值（rel_err≈30000/60000≈0.5 ≫ 1e-3），判 FAIL 不受影响，但"个别元素恰好等于哨兵"的理论可能仍在——极值用例值域贴近 30000 时改选其他哨兵。

## 坏实践

- 输出缓冲 `aclrtMemset` 清零后直接跑红基线：全零期望用例（go=0 类）巧合通过，红基线失真。
- 用 NaN 做哨兵：与"同位 NaN 匹配"语义存在残余巧合面（特殊值用例的 golden NaN 位置会被缓冲残留命中）。
- 只在套件开头 poison 一次：确定性双跑的第二轮起点状态不受控，位级一致性结论失效。

## 误诊

- "stub kernel 会把输出清零/框架会帮忙初始化"——不会：aclnn 只把设备指针传给 kernel，桩不写就是保持原值；这是本卡整个前提的反面误判。
- "红基线反正都会挂，输出初始化随便"——正是全零期望用例这条漏网之鱼让"随便"变成假通过。

## 后续增量（2026-09-17，并行策展批：aclrtMalloc 零页观测、-999/0xCD 哨兵变体与绿基线无扰实证）

softplus_v2_grad 轨迹（Ascend950 + CANN 9.1.0）从红基线到绿基线补齐原卡缺失的三个证据点：

**1. aclrtMalloc 零页观测（把"框架清零"误判的另一半补全）**：原卡前提是"测试方清零导致巧合"；实测发现**不显式清零也会踩同一个坑**——`aclrtMalloc` 返回的 device 内存不保证是"脏"的：新分配（尤其 `ACL_MEM_MALLOC_HUGE_FIRST` 首页）常表现为全零。于是"kernel 未写输出"与"kernel 写了全零"在输出缓冲上不可区分；当用例期望恰好全零（S6：grad_output 全零 → dx = dy·σ(z) ≡ 0）时，未写回的零缓冲直接通过 `|a-e| <= atol` 比对——红基线被污染成"部分绿"（`total=15 PASSED=2 FAILED=13`，S6 是唯一解释不了的 PASS）。

**2. -999 有限哨兵变体（红基线对照实测）**：

```cpp
// 用哨兵值预填输出 buffer：kernel 桩不写回时输出保持哨兵 → 不匹配期望（TDD 红），
// 避免「device 内存恰好为 0 偶合全零期望（如 S6）」的假绿。真实 kernel 会覆写为正确值。
if (c.dtype == "float32") {
    std::vector<float> sent(n, -999.0f);
    CHECK_ACL(aclrtMemcpy(devOut, bytes, sent.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
} else {
    aclFloat16 sentH = aclFloatToFloat16(-999.0f);
    std::vector<aclFloat16> sent(n, sentH);
    CHECK_ACL(aclrtMemcpy(devOut, bytes, sent.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
```

红基线对照：加哨兵后同套用例 `total=15 PASSED=1 FAILED=14 ERROR=0`（仅空 tensor 用例合理通过），S6 显示 `expected=0 actual=-999` 类失配——假绿被拆除。

**3. 0xCD 字节哨兵变体（黑盒框架、dtype 无关）**：`std::vector<uint8_t> sent(bytesOut, 0xCD)` 整体 H2D 预填——0xCD 与浮点模式（0xCD68... 量级异常值）及整数模式都不像合法输出，且对"最常见的期望全零"用例保持保护（0x00 哨兵会失去该保护）；一次 `aclrtMemcpy` 的成本相对用例本身可忽略；空输出（nOut==0）跳过（专设 PASS 分支）。

**4. 绿基线无扰实证（补原卡失效边界第 2 条）**：原卡标注"绿基线 poison 被全量覆写的行为未验证"。本轨迹两次闭环：① 红转绿后（哨兵保留）两次完整运行 `total=15 PASSED=15 FAILED=0 ERROR=0`、exit=0，输出两次逐位一致；② 黑盒全量 517 用例（含 111 条 >100M 元素大 shape 与全零梯度/特殊值用例）在该预填下真实比对 517/517 PASSED——poison 不干扰真实写回路径的正常比对，绿基线无副作用从"推断"升级为"实测"。

**5. 验证信号**：

- 修复前（kernel 桩、无哨兵）：`total=15 PASSED=2 FAILED=13`，S6（全零期望）`[PASS] 8 elements match golden`——kernel 未写回却通过（来源平台：950；证据等级：tool_observed）。
- 修复后（kernel 桩、-999 哨兵）：`total=15 PASSED=1 FAILED=14 ERROR=0`，仅空 tensor N3 通过，S6 显示 expected=0 actual=-999 类失配（来源平台：950；证据等级：tool_observed）。
- 真实 kernel 落地（哨兵保留）：两次运行 `total=15 PASSED=15 FAILED=0 ERROR=0`、输出两次逐位一致（来源平台：950；证据等级：tool_observed）。
- 0xCD 变体绿基线：黑盒全量 `total=517 PASSED=517 FAILED=0`（含大 shape 与全零梯度用例），比对为真实逐元素 golden 比对；哨兵拦截路径（桩未写回 → 0xCD mismatch → FAIL）在全量通过前提下未被触发，有效性为构造性论证 + 代码注释佐证（来源平台：950；证据等级：tool_observed）。

**6. 边界与坏实践补充**：device 缓冲"恰好为零"是观测行为（本环境 HUGE_FIRST 分配），不是契约——依赖"非零脏数据"与依赖"清零"同样错误，预填才是确定性的；若算子输出值域可能等于哨兵值（-999 附近合法数据 / 0xCD 重复模式），需换哨兵或改用"写覆盖位图"方案（未尝试）；红基线里每个 PASS 都要能解释（空 tensor 类），解释不了的就是假绿。





## 后续增量（2026-09-17，并行策展批：不预填也假绿实测、-999/0xCD 哨兵变体、绿基线无扰实测）

softplus_v2_grad 算子（Ascend950，CANN 9.1.0）的两条 ST 轨迹补齐本卡三个证据点：

**1. 失败侧新形态：不预填也会假绿（aclrtMalloc 新分配常为全零）**。本卡原前提是"输出设备缓冲保持测试方上送的原值"；实测 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)` 的新分配（尤其首页）**常表现为全零**——即使测试方不做任何清零/预填，"kernel 未写输出"与"kernel 写了全零"在缓冲上同样不可区分。红基线实测：kernel 接口桩、输出缓冲未预填时 `total=15 PASSED=2 FAILED=13`，其中全零期望用例（grad_output 全零 → 期望 dx 全零）显示 `[PASS] 8 elements match golden`——未写回的零缓冲与全零期望巧合吻合。加哨兵后 `total=15 PASSED=1 FAILED=14`（仅空 tensor 用例合理通过，全零期望用例失配 `expected=0 actual=-999`）——假绿被拆除。"device 缓冲恰好为零"是观测行为（非契约）：依赖"非零脏数据"与依赖"清零"同样错误，显式预填才是确定性的。

**2. 两个哨兵变体**：

- **-999 有限哨兵**（红基线 ST，fp32/fp16 双 dtype）：fp32 直接填 float 向量，fp16 用 `aclFloatToFloat16(-999.0f)`（`acl_base_rt.h`）构造原生半精度位型再整块上传；选 -999 而非 NaN——NaN 会与"期望为 NaN 的用例"的 isnan 分支语义相混（与复现增量的 0xFF 教训同源）。
- **0xCD 字节哨兵**（黑盒框架，dtype 无关）：`std::vector<uint8_t> sent(bytesOut, 0xCD)` 整块 H2D——不依赖 dtype 解码，全零期望（0x00 哨兵会假绿）与常见浮点/整数模式都不会撞上；空输出（bytesOut==0）跳过。

```cpp
    // 输出（按 output_shape/dtype）
    size_t allocOut = (bytesOut == 0) ? 1 : bytesOut;
    CHECK_ACL(aclrtMalloc(&devOut, allocOut, ACL_MEM_MALLOC_HUGE_FIRST));
    if (bytesOut > 0) {
        // 哨兵预填，防 kernel 桩/异常不写回时假绿
        std::vector<uint8_t> sent(bytesOut, 0xCD);
        CHECK_ACL(aclrtMemcpy(devOut, bytesOut, sent.data(), bytesOut, ACL_MEMCPY_HOST_TO_DEVICE));
    }
```

**3. 绿基线无扰从"未验证"升级为实测**（补齐原卡失效边界第 2 条）：真实 kernel 落地后哨兵保留，红基线 ST 两次完整运行 `total=15 PASSED=15 FAILED=0 ERROR=0`、exit=0 且输出两次逐位一致；黑盒全量 517/517 PASSED（含 111 条 >100M 元素大 shape 与全零梯度/特殊值用例）——哨兵预填不干扰真实写回路径的正常比对。

### 复现增量验证信号

- 未预填红基线：`total=15 PASSED=2 FAILED=13`，全零期望用例 `[PASS] 8 elements match golden`（kernel 桩未写回）（来源平台：950；证据等级：tool_observed）。
- 加 -999 哨兵后：`total=15 PASSED=1 FAILED=14 ERROR=0`，全零期望用例失配 `expected=0 actual=-999`，RUN_SH_EXIT=1（来源平台：950；证据等级：tool_observed）。
- 真实 kernel + 哨兵保留：两次 `total=15 PASSED=15 FAILED=0 ERROR=0`、exit=0、输出两次逐位一致（来源平台：950；证据等级：tool_observed）。
- 黑盒 0xCD 哨兵全量 517/517 PASSED、0 FAIL（含大 shape 与全零梯度用例），逐元素 golden 比对逐用例打印（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 哨兵拦截路径（桩未写回 → 0xCD mismatch → FAIL）在黑盒全量通过的前提下未被触发，有效性为构造性论证（红基线轨迹的 -999 拦截已实测）。
- 若算子输出值域可能覆盖哨兵值（-999 附近、0xCDCD 解码值），需换哨兵；0xCD 解码对不同 dtype 呈异常量级浮点/大整数，实践碰撞概率可忽略。

## 后续增量（2026-09-17，并行策展批：0xFF 字节哨兵在 NaN 期望用例上的实测假通过与 0x41 选型）

euclidean_norm 算子 group 分支（Ascend950PR，CANN 9.1.0）的路由探针实测把本卡"不用 NaN 做哨兵"的设计推导升级为**观测到的假通过**，并给出字节填充场景的选型规则：

**1. 0xFF 哨兵的实测假阴性**。探针用"输出 buffer 预填 0xFF 哨兵 + 与独立 CPU golden 比对"判定 shape 路由（输出匹配 golden ⇒ 已实现分支；输出保持哨兵 ⇒ stub 分支）。8 个适配用例中的 NaN 注入用例（输入含 NaN → 期望 kernel 输出 NaN）被误判 `IMPLEMENTED(normal)`——而同 shape 的非 NaN 姊妹用例判 `STUB key4`，且该 group 分支当时根本没有实现。机制：**0xFF 字节填充对 IEEE 754 浮点（fp32/fp16/bf16）是 NaN 位型**；比较器的"golden NaN ⇒ 期望 device NaN ⇒ 匹配"路径无法区分"kernel 计算出的 NaN"与"kernel 从未写入、哨兵残留的 NaN"——no-op stub 借哨兵位型假通过。`sed` 把哨兵换成 0x41 后同一用例立即正确判 `STUB key4`。

**2. 选型规则（NaN 期望用例集合必须有限位型哨兵）**：哨兵位型必须在**所有被测 dtype**上都是有限、非零、非特殊值的数——0x41 满足（fp32 0x41414141=12.078431、int32=1094795585、bf16 0x4141=12.0625、fp16≈2.627），"未写输出"在数值比较上必然失败，NaN 期望只能由真实写入的 NaN 满足。无 NaN 期望的集合仍可用 0xFF（其 NaN 哨兵对"未写输出"区分力最强）；本场景空 tensor ST（golden 恒 0）用 0xFF、group ST（含 NaN 用例）用 0x41。选值理由与各 dtype 解码表固化在测试头注释。

**3. 判别手法**：同 shape 的非 NaN 姊妹用例对照——同一 shape 两个判据矛盾（一判 STUB 一判 IMPLEMENTED）时，先怀疑测试脚手架（哨兵/比较器）再怀疑被测对象。

### 复现增量验证信号

- 0xFF 哨兵：NaN 期望用例（实际路由 stub key=4）被误判 `IMPLEMENTED(normal)`，同 shape 非 NaN 姊妹用例判 `STUB key4`——矛盾即假阴性证据（来源平台：950；证据等级：tool_observed）。
- 0x41 哨兵：同一用例正确判 `STUB key4(group tail-A)`（来源平台：950；证据等级：tool_observed）。
- 正式 ST（0x41 哨兵）端到端复核：kernel 接线前 0/8 全失败（该用例失败值 dev=12.078431 即哨兵解码，证明未写输出被正确暴露）、接线后 8/8 PASSED（NaN 传递通过）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- 本增量只解决"NaN 期望 × 哨兵位型"的假阴性；不覆盖比较器本身的 NaN 语义（本例比较器对 golden NaN 的处理是正确的，错在哨兵位型与其正交冲突）。
- 0x41 解码值表基于 IEEE 754 的 fp32/fp16/bf16 与二补码 int32；出现其他 dtype（如 fp64/int16）需重算解码表。
- 坏实践增补：含 NaN 期望的用例集合用 0xFF 字节哨兵——假通过恰好发生在"最需要检测 stub"的测试先行阶段；哨兵值不留解码表记录（失败输出里的哨兵解码值会被误当数值 bug）。

## 后续增量（2026-09-17，并行策展批：0xFF 全 NaN 哨兵在 golden 全有限用例集上的红/绿闭环与 special 标签 NaN 感知比对）

RoPE 算子（BenchHarnessApplyRotaryPosEmb，Ascend950PR + CANN 9.2.0）的 aclnn ST（C++ 直调两段式、manifest 驱动）再补一个 0xFF 变体实例，给出与本卡先前增量的"0x41 选型"并列的第三条路线：

**1. 0xFF 字节毒化 + golden 全有限值用例集（可用的充分条件）**：调被测接口前把输出 device buffer 用 0xFF 填满（`std::vector<uint8_t> poison(qoN * elem, 0xFF)` 经 `aclrtMemcpy` H2D）——0xFF 模式在 fp32/fp16/bf16 的 IEEE 754 表示里都是 quiet NaN。桩 kernel（入口 else 分支仅 `PipeBarrier`、不写输出）跑完后读回全 NaN，与 golden 的有限期望值逐点必不匹配（含期望为 0 的全零输入用例）→ 所有 numeric 用例确定性红。实测红基线：kernel 未实现时 12 用例中 11 个 numeric 全 FAILED（`a6_fp32 FAILED: queryOut mismatch at idx=0 actual=-nan expected=0.389963`；`c2_all_zero expected=0` 也 mismatch——**期望 0 不假绿**）；`returns_empty` 空张量用例（0 元素无比对）按结构判定单独通过。转绿后正确输出覆盖毒化值，12/12 全绿；同一毒化框架对四组分支 manifest 各 12/12 回归通过。

**2. 与本卡先前增量的 0xFF 假阴性教训的兼容路线（special 标签 NaN 感知比对）**：先前增量实测 0xFF 会在"期望本身为 NaN"的用例上假通过（哨兵 NaN 与 kernel 真实写入的 NaN 不可分）。本实例的框架用 **special 标签驱动的 NaN 感知比较**化解：期望含 NaN 的用例（如 `c1_query_nan`）由 manifest 的 `special` 字段（produces_nan/matches_oracle）选择 NaN 语义分支，比对器按标签区分"期望 NaN ⇒ actual 须 isnan"与有限值逐点 rtol/atol——0xFF 与期望 NaN 的碰撞面由此消除，无需换哨兵值。无标签机制的框架仍应按先前增量结论选有限位型哨兵（0x41/-999/0xCD）。

**3. `actual=-nan` 报错签名的可读性**：桩不写输出时读回值即毒化位型，`actual=-nan` 一眼可辨来源——同时排除"碰巧相等"与"看不出来源"两种红基线失真；把 `actual=-nan` 误读为"输入数据被 NaN 污染"是常见误诊方向，先确认测试框架是否毒化（本框架 NaN 期望用例红基线报有限期望值，说明 mismatch 索引落在期望有限值元素上，NaN 来自毒化而非输入）。

**4. 配套独立性设计**：期望值由独立 torch golden 离线生成 `.bin`（golden 与被测 kernel 无共享路径，引用扫描 `PASS: golden is independent of kernel (no aclnn/kernel references)`；极小 shape 手算对照 `PASS: golden interleaved-BNSD matches manual computation`）；输入 .bin 即上 device 的同源数据。

### 复现增量验证信号

- 红基线（kernel 未实现、入口落桩）：12 用例 `numeric 0/11 PASSED, returns_empty 1/1 OK (total 1/12)`；11 个 numeric 用例全部 `actual=-nan`（a6_fp32 idx=0 actual=-nan expected=0.389963；c2_all_zero expected=0 也 mismatch，证明 0 期望不假绿）（来源平台：950；证据等级：tool_observed）。
- golden 独立性：`PASS: golden interleaved-BNSD matches manual computation` + `PASS: golden is independent of kernel (no aclnn/kernel references)`（来源平台：950；证据等级：tool_observed）。
- 转绿（kernel 实现后）：12/12（`numeric 11/11 PASSED, returns_empty 1/1 OK`），含 fp32/fp16/bf16、B 广播、最小 D、大 shape、非对齐 D、NaN/Inf/大值特殊值用例（来源平台：950；证据等级：tool_observed）。
- 跨分支回归：四分支 manifest 各 12/12、`run.sh` 退出码 0——同一毒化框架对四组分支均判别有效（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 0xFF→NaN 论证只对 IEEE 754 浮点（fp32/fp16/bf16）成立；整型或特殊编码 dtype 须另选毒化模式，未验证。
- 桩 kernel 若写了部分输出（半实现），actual 是"毒化+真实"混合，签名弱化——桩必须全不写（仅 PipeBarrier）。
- 毒化只证明"kernel 写没写输出"，不证明写得对——数值正确性仍靠 golden 逐点比对；两机制互补。
- 期望本身为 NaN 的用例在本框架靠 special 标签 + NaN 感知比较区分；无标签机制的框架不得用 0xFF（见 复现增量）。
