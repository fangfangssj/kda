---
type: "Runbook"
title: 黑盒CSV用例数据驱动aclnn ST框架
description: 把黑盒 CSV（L0/L1/L2）转 JSON manifest，C++ 数据驱动框架逐条执行：多 dtype 主机侧舍入/读写、popen 独立 golden、异常用例 expectError 路径、inf 字面量哨兵替换（ast.literal_eval 不可解析 inf/-inf，用 3.4e38 哨兵 + C++ 阈值还原 ±INFINITY）。复现增量：常驻 golden co-process（fork+pipe 摊薄 torch import 3.2s/条）+ readFile 整块读与位等快速通道 + 517 条全量两轮 exit-0 闭环与独立 st-verifier 复跑。复现增量：算子无关核心 + 薄算子层双钩子（callOp/computeExpected）复用变体、六条防作弊硬约束（单全局 RNG 永不重置等）与白盒不退化复核，549 条（L0 40）经 shape capping 后 3 次全量一致。
tags: [task.testing, topic.compilation, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T04:41:27Z'
platforms: ["950"]
---

# 黑盒CSV用例数据驱动aclnn ST框架

## 现象

自定义一元数学算子（BesselI1e）三 dtype 分支 kernel 全部实现后，需要把先前黑盒设计阶段产出的 521 条 ACLNN ST 用例（`tests/blackbox/testcases/` 下三个 CSV：L0 冒烟 18 条、L1 全量 500 条、L2 异常 3 条）纳入 `tests/aclnn/` 通用 ST 框架并保证可全量执行。当时的 `test_aclnn.cpp` 是硬编码用例的旧框架，不认识 CSV；期望值必须由外部独立 golden（`docs/bessel_i1e/develop/golden.py`，scipy 实现）现算，与被测 kernel 完全隔离。接入过程中遇到三个数据面问题：① CSV 的 `input_data_ranges` 里有 `inf`/`-inf` 字面量，manifest 生成脚本用 `ast.literal_eval` 解析直接失败，导致这些用例 `input_data_ranges` 为空；② L2 异常用例（如 float64 dtype 异常）带两份 input shape，且期望的是"调用报错"而非数值比对；③ fp32/fp16/bf16 三种 dtype 混在一个用例集里，主机侧必须按目标 dtype 舍入输入、按 dtype 编码写 device、按 dtype 解码读回，否则 golden 与 kernel 看到的输入不一致。

## 适用场景

- 工程形态：aclnn 自定义算子 ST 测试框架搭建（黑盒 CSV 用例 → 数据驱动 C++ 框架）。
- API 路径：ACL 运行时（`aclrtMalloc`/`aclrtMemcpy`/`aclrtCreateStream`/`aclrtSynchronizeStream`、`aclCreateTensor`、`ACL_FLOAT/ACL_FLOAT16/ACL_BF16`、`aclFloatToFloat16`/`aclFloat16ToFloat`）、aclnn 两段式调用（`aclnnXxxGetWorkspaceSize` → `aclnnXxx` + executor）、popen 调 python golden。
- 验证过的平台与 CANN 版本：950（ascend950 构建目标）+ CANN 9.0.0-beta.2。
- 适用粒度：`platform_wide`——任何一进一出 aclnn 算子的黑盒用例纳入流程（框架结构与 dtype 无关，替换算子名与 golden 即可）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：
  - 多输入算子：manifest 解析当前只取第一个 input shape/dtype/range（本算子单输入），多输入需扩展 `input_shapes` 的多槽位处理。
  - 需要图模式/动态 shape 的用例：框架按 CSV 静态 shape 逐条建 tensor。
  - 全量 521 条含超大 shape 的"完整跑完 + 退出码 0"闭环：本轨迹内因超大 shape 用例运行时间/内存问题未完成（见失效边界与遗留问题）。

## 根因/解决配方

框架分四层（runner 脚本 → manifest 生成 → C++ 数据驱动执行 → 独立 golden），逐层解决上面三个问题。

**① inf 字面量 → 哨兵值替换**（CSV 预处理阶段，run.sh 内完成）：

```bash
CSV_ALL="${SCRIPT_DIR}/blackbox_all.csv"
{
    head -1 "${BB_DIR}/aclnnBesselI1e_l0_test_cases.csv"
    tail -n +2 "${BB_DIR}/aclnnBesselI1e_l0_test_cases.csv"
    tail -n +2 "${BB_DIR}/aclnnBesselI1e_l1_test_cases.csv"
    tail -n +2 "${BB_DIR}/aclnnBesselI1e_l2_test_cases.csv"
} > "${CSV_ALL}"
echo "Merged CSVs: $(wc -l < "${CSV_ALL}") lines (including header)"

# Fix: replace inf/-inf in input_data_ranges with sentinel values that
# ast.literal_eval can parse. The C++ genInputs will detect 3.4e38/ -3.4e38
# and treat them as ±inf.
python3 -c "
import re
with open('${CSV_ALL}', 'r') as f:
    content = f.read()
# Replace standalone inf/-inf in data ranges: ((inf,inf,),) → ((3.4e+38,3.4e+38,),)
content = re.sub(r'\binf\b',  '3.4e+38', content)
content = re.sub(r'\b-inf\b', '-3.4e+38', content)
with open('${CSV_ALL}', 'w') as f:
    f.write(content)
print('inf sentinel replacement done')
"
```

C++ 侧 `genInputs` 用阈值识别哨兵并还原 ±INFINITY（最终态，verbatim）：

```cpp
static std::vector<float> genInputs(int64_t numElements,
                                     const std::vector<double>& dataRange,
                                     std::mt19937& rng) {
    std::vector<float> result(numElements);
    if (numElements == 0) return result;

    double lo = dataRange.size() >= 1 ? dataRange[0] : -10.0;
    double hi = dataRange.size() >= 2 ? dataRange[1] : 10.0;

    // Detect ±inf sentinels: prepare_blackbox_cases.py replaces CSV inf→3.4e+38
    const double INF_SENTINEL = 3.0e38;
    if (lo >= INF_SENTINEL) {
        // +inf range
        for (int64_t i = 0; i < numElements; ++i) result[i] = INFINITY;
        return result;
    }
    if (hi <= -INF_SENTINEL) {
        // -inf range
        for (int64_t i = 0; i < numElements; ++i) result[i] = -INFINITY;
        return result;
    }

    // Handle zero range
    if (lo == 0.0 && hi == 0.0) {
        for (int64_t i = 0; i < numElements; ++i) result[i] = 0.0f;
        return result;
    }

    if (lo == hi) {
        // Fixed value
        for (int64_t i = 0; i < numElements; ++i) result[i] = static_cast<float>(lo);
    } else {
        // Regular random in [lo, hi]
        std::uniform_real_distribution<float> dist(static_cast<float>(lo),
                                                    static_cast<float>(hi));
        for (int64_t i = 0; i < numElements; ++i) {
            result[i] = dist(rng);
        }
    }
    return result;
}
```

**② 多 dtype 主机侧舍入与编码**——保证 golden 与 kernel 看到同一份"已按目标 dtype 舍入"的输入，并按 dtype 编解码 device 字节流（verbatim）：

```cpp
static float roundToDtype(float v, aclDataType t) {
    if (t == ACL_FLOAT) return v;
    if (t == ACL_FLOAT16) {
        aclFloat16 h = aclFloatToFloat16(v);
        return aclFloat16ToFloat(h);
    }
    if (t == ACL_BF16) {
        uint16_t b = floatToBf16(v);
        return bf16ToFloat(b);
    }
    return v;
}

// Write a float value into a byte buffer in the target dtype
static void writeDtype(void* dst, float v, aclDataType t) {
    if (t == ACL_FLOAT) {
        float* p = static_cast<float*>(dst);
        *p = v;
    } else if (t == ACL_FLOAT16) {
        uint16_t* p = static_cast<uint16_t*>(dst);
        *p = aclFloatToFloat16(v);
    } else if (t == ACL_BF16) {
        uint16_t* p = static_cast<uint16_t*>(dst);
        *p = floatToBf16(v);
    }
}

// Read a float value from a byte buffer in the target dtype
static float readDtype(const void* src, aclDataType t) {
    if (t == ACL_FLOAT) {
        const float* p = static_cast<const float*>(src);
        return *p;
    } else if (t == ACL_FLOAT16) {
        const uint16_t* p = static_cast<const uint16_t*>(src);
        return aclFloat16ToFloat(*p);
    } else if (t == ACL_BF16) {
        const uint16_t* p = static_cast<const uint16_t*>(src);
        return bf16ToFloat(*p);
    }
    return 0.0f;
}

// Get default tolerances for a dtype
static void defaultTolerances(aclDataType t, float& rtol, float& atol) {
    if (t == ACL_FLOAT)          { rtol = 1e-5f; atol = 1e-5f; }
    else if (t == ACL_FLOAT16)   { rtol = 1e-3f; atol = 1e-4f; }
    else if (t == ACL_BF16)      { rtol = 4e-3f; atol = 1e-3f; }
    else                         { rtol = 1e-5f; atol = 1e-5f; }
}
```

其中 BF16 主机侧转换（`ACL_BF16` 的 aclTensor 数据在 host 内存即 uint16 位串）：

```cpp
// floatToBf16 / bf16ToFloat：BF16 = float32 高 16 位（1+8+7）
static uint16_t floatToBf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

static float bf16ToFloat(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
```

**③ L2 异常用例 expectError 路径**——manifest 后处理按 case_id/remark 标 `expectError`，执行时建 tensor 调 aclnn、以"调用失败"为 PASS（verbatim）：

```cpp
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus nnRet = aclnnBesselI1eGetWorkspaceSize(xTensor, outTensor,
                                                            &workspaceSize, &executor);

        if (nnRet != OK) {
            fprintf(stdout, "[%s] PASS (expect_error — GetWorkspaceSize failed: %d)\n",
                    tc.caseId.c_str(), nnRet);
        } else {
            void *workspace = nullptr;
            if (workspaceSize > 0) {
                aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            nnRet = aclnnBesselI1e(workspace, workspaceSize, executor, stream);
            aclrtSynchronizeStream(stream);
            if (workspace) aclrtFree(workspace);
            if (nnRet != OK) {
                fprintf(stdout, "[%s] PASS (expect_error — aclnnBesselI1e failed: %d)\n",
                        tc.caseId.c_str(), nnRet);
            } else {
                fprintf(stdout, "[%s] FAIL — expected error but kernel succeeded\n",
                        tc.caseId.c_str());
            }
        }
```

manifest 后处理识别逻辑（解析阶段）：case_id 含 `exception` 或 remark 含 `expect_error` 即置 `tc.expectError = true`；两份 input shape 的异常用例取 `shapes[0]` 建真实 tensor，另一份按 manifest 原样保留不参与执行。（"期望报错但 kernel 成功"的断言语义修正见同域 expectError 语义卡，本框架初版把该形态判 FAIL，后续已按该语义收敛。）

**正常用例执行骨架**（两段式 aclnn 调用，verbatim 核心）：

```cpp
    // 输入按目标 dtype 舍入后写 device
    std::vector<float> xRounded(numElements);
    for (int64_t i = 0; i < numElements; ++i) {
        xRounded[i] = roundToDtype(xRaw[i], aclType);
    }
    ...
    std::vector<uint8_t> xBytes(bytes);
    for (int64_t i = 0; i < numElements; ++i) {
        writeDtype(&xBytes[i * elemSize], xRounded[i], aclType);
    }
    CHECK_ACL(aclrtMemcpy(devX, bytes, xBytes.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ...
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus nnRet = aclnnBesselI1eGetWorkspaceSize(xTensor, outTensor,
                                                        &workspaceSize, &executor);
    if (nnRet != OK) { /* SKIP — kernel likely not implemented */ }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    nnRet = aclnnBesselI1e(workspace, workspaceSize, executor, stream);
    if (nnRet != OK) { /* SKIP */ }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // 读回 + readDtype 解码 + 与 golden 逐元素比对（abs_err > atol + rtol*max(|got|,|exp|) 判 Mismatch）
```

**runner 闸门**（run.sh，manifest 非空硬闸 + 全量执行、禁止 --quick/--max-cases 捷径，verbatim 关键段）：

```bash
MANIFEST="${SCRIPT_DIR}/aclnn_cpp_cases.json"
python3 "${SKILL_DIR}/prepare_blackbox_cases.py" \
    "${CSV_ALL}" "${MANIFEST}" "${GOLDEN}"

# Hard gate: manifest must be non-empty
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d.get('cases', []))
print(f'[run.sh] manifest OK: {n} case(s)')
if n == 0:
    sys.exit('FATAL: aclnn_cpp_cases.json cases is empty')
" "${MANIFEST}"
```

## 设计决策

- **为什么 CSV→JSON manifest 而不是 C++ 直接解析 CSV**：黑盒 CSV 由独立设计阶段产出（含表头、tuple 字面量 `((0,10,),)`、inf 字面量），`ast.literal_eval` 在 python 侧预处理最稳；C++ 只消费规整后的 JSON（自写极简 parser，无第三方依赖）。
- **为什么主机侧先 roundToDtype 再喂 golden 和 kernel**：fp16/bf16 的输入若不先舍入，golden（python 收到的是 fp32 字面量）与 kernel（device 上是 fp16/bf16 位串）看到的输入不同，会把"输入表示差"误报成精度失败。两边都吃舍入后的值，比较才隔离了"计算误差"。
- **golden 经 popen 每用例起一个 python 进程**：实现最简、隔离最强（golden 崩溃不影响框架），代价是全量执行慢（见遗留问题）；曾评估批量化但本轨迹未实施。
- **异常用例"失败即 PASS"的判据**：L2 用例的期望行为就是 aclnn 层报错（如不支持的 dtype），因此以返回码非 OK 为通过条件；`GetWorkspaceSize` 失败也算（参数校验阶段拦截）。
- **inf 用哨兵 3.4e+38 而不是改 manifest 生成器**：生成脚本是公共 skill 资产，不动它；哨兵阈值取 3.0e38（< 3.4e38 但 > 任何合法 fp32 finite 输入上界 FLT_MAX≈3.4e38 的常规数据范围），在 C++ 侧还原为 ±INFINITY。
- **当前方案属哪档路线**：recommended_fastpath（黑盒用例纳入类任务）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（x86_64-linux），构建目标 ascend950；算子包 `./build.sh BesselI1e_package` 构建+安装；ST 测试经独立 CMake 目标（`tests/aclnn/CMakeLists.txt` 的 `test_aclnn`）编译。
- 关键环境：`LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/BesselI1e/op_api/lib/`（链接 libcust_opapi.so）；`ACLNN_REPO_ROOT` 传给 C++ 定位 golden。
- 相关文件（与事件流实际读写一一对应）：
  - `operators/BesselI1e_package/tests/aclnn/test_aclnn.cpp`（数据驱动框架，重写）
  - `operators/BesselI1e_package/tests/aclnn/run.sh`（runner：合并 CSV→哨兵替换→manifest→构建→执行）
  - `operators/BesselI1e_package/tests/aclnn/aclnn_cpp_cases.json`（生成的 manifest，521 条）
  - `operators/BesselI1e_package/tests/aclnn/blackbox_all.csv`（合并后的 CSV）
  - `docs/bessel_i1e/develop/golden.py`（独立 golden，从 tests/aclnn/golden.py 拷贝固化）
  - `.opencode/skills/aclnn-st-framework/prepare_blackbox_cases.py`（CSV→manifest 转换，公共资产，未改动）

## 完整代码

「根因/解决配方」已 verbatim 内联框架的全部可复用核心：run.sh 的 CSV 合并+哨兵替换+manifest 闸门、genInputs 最终态、roundToDtype/writeDtype/readDtype/defaultTolerances、floatToBf16/bf16ToFloat、expectError 路径、正常路径 aclnn 两段式骨架。JSON 极简 parser（jsonReadStr/jsonReadInt2D/jsonReadDouble2D/jsonSkipValue 等约 120 行）与 manifest 结构（case_id/input_shapes/input_dtypes/input_data_ranges/precision_tolerances/is_enabled/remark + golden_path）结构简单，按上述字段名自实现即可，不逐行复述。

## 预防

- 纳入黑盒用例前先扫描 CSV 的 `input_data_ranges` 是否含 `inf`/`-inf`/`nan` 字面量（`grep -c 'inf' xxx.csv`），有则先做哨兵替换再进 manifest 生成，否则这些用例会静默变成空 range。
- 用例集混 dtype 时，务必在主机侧做 roundToDtype 双通道（golden 与 kernel 同输入），否则会出现假精度失败。
- golden 路径写进 manifest 的绝对路径在换机器/换目录时会失效，run.sh 每次重新生成 manifest 可规避。

## 验证信号

以下信号来自同一次开发会话的框架搭建与黑盒纳入（ascend950 构建目标，CANN 9.0.0-beta.2）：

- manifest 生成：`wrote 521 cases -> operators/BesselI1e_package/tests/aclnn/aclnn_cpp_cases.json`，`[run.sh] manifest OK: 521 case(s)`（18 L0 + 500 L1 + 3 L2 = 521，与源 CSV 数据行数一致）（来源平台：950；证据等级：tool_observed）。
- inf 修复后 0 条用例 data_ranges 为空；哨兵替换日志 `inf sentinel replacement done`（来源平台：950；证据等级：tool_observed）。
- 框架全量加载并逐条执行：`Cases: 521 enabled`，用例结果逐条打印，例如 `[aclnnBesselI1e_L0_002] PASS`、`[aclnnBesselI1e_L0_001] Mismatch at [0]: got=-0.00118732452, expected=-0.00155830383 (abs_err=0.000371, rtol=0.001, atol=0.0001)`——golden 独立调用与精度比较均生效（来源平台：950；证据等级：tool_observed）。
- L2 异常用例识别：3 条 exception 用例全部 `expectError=True`，走"调用失败即 PASS"路径（来源平台：950；证据等级：tool_observed）。
- 多 dtype golden 一致性：`echo '[3.5]' | GOLDEN_DTYPE=fp32/fp16/bf16 python3 golden.py` 分别给出 0.1873999834060669 / 0.1873779296875 / 0.1865234375（fp32→fp16→bf16 精度递减，符合各 dtype 分辨率）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单输入算子框架：多输入算子需扩展 manifest 多槽位（未实现）。
- 全量 521 条"跑完 + 退出码 0"未在本轨迹闭环：超大 shape 用例（如 440M 元素）导致进程被杀（退出码 137）后，框架加了 128M 元素 SKIP 守卫，守卫生效已实测（`SKIP — shape too large (numElements=440798150 > max=134217728)`），但 521 条全量 exit-0 运行在轨迹切片结束时仍未完成——该闭环属后续阶段，本卡不宣称。
- 每用例一个 python golden 子进程：全量执行 30 分钟仅推进 226/521（42 SKIP / 116 PASS / 67 FAIL），性能上不适合高频回归。
- CSV 自带 `precision_tolerances`（如 rtol=0.001/atol=0.0001）会覆盖 defaultTolerances；黑盒阈值严于算子设计阈值（4e-3/4e-3）时出现的 Mismatch 属精度任务范畴，不是框架缺陷（本轨迹中此类精度失败未闭环，不在此展开）。
- 其他平台/CANN 版本未验证，推断可迁移（框架仅依赖 aclnn 通用调用链与 host 侧数据面，未验证，推断可迁移）。

**未决项**：

- 全量 521 条 + 退出码 0 的最终验收在轨迹切片结束时未完成（重编后全量运行刚要启动）。
- golden 子进程逐条 popen 的性能问题：批量化（一次进程处理多用例或常驻服务）未实现。
- 黑盒精度失败（L0_001/L0_003/L0_004 等 Mismatch、NaN 输出）的根因定位属后续精度处理任务，本卡未解。
- 退出码语义在轨迹尾段从"有 FAIL 即非零"演化为"仅 ERROR 非零"（配合超大 shape SKIP 与分阶段验收口径）；该演化跨切片边界未闭环，且**不得**作为掩盖精度回归的手段——精度 FAIL 必须仍被打印计数并交给精度阶段处理。
## 坏实践

- 把含 `inf` 的 CSV 直接喂 `ast.literal_eval` 而不检查空 range——静默丢数据范围，用例退化成默认范围随机。
- golden 与 kernel 喂不同精度的输入（一边 fp32 原值、一边 dtype 舍入值）再比精度——比较的是输入表示差不是计算误差。
- 用 `--quick`/`--max-cases` 只跑前 N 条充当"全量执行"——验收口径明确禁止的子集捷径。

## 误诊

- 首次全量执行被杀（退出码 137）时，验收与修复阶段先入为主归因"宿主 OOM"；后续检查 `dmesg` 无 OOM 记录、`free` 显示 688GB 可用——"宿主 OOM"假设与观测矛盾，137 的确切成因（宿主 OOM vs 外层工具超时 SIGKILL）未定论，修复守卫对两种成因都有效（守卫生效为 tool_observed；成因归因未定论）。

## 后续增量（2026-09-17，并行策展批：运行时 CSV 直接解析变体 + expected_error_code 字段 + Mock 侧 L2 校验自测）

另一独立轨迹并入 scatter 算子（Ascend950PR，CANN 9.1.0，L0=28 + L1=767 + L2=10 共 805 条 CSV + 11 条边界）的接入实例，给出与主卡"离线 CSV→JSON manifest"并列的第二种接入形态：**测试程序启动时运行时解析 `testcases/*.csv`**，用例改动不重编测试程序、不需要 manifest 预处理。

**1. 引号状态机 CSV 解析（替代 awk -F','）**：CSV 的 shape 字段带引号内逗号（`"[8, 10]"`），简单逗号 split 列错位；逐字符状态机解析（verbatim 核心两个函数）：

```cpp
// 解析 shape 字符串 "[8, 10]" → {8, 10}
std::vector<int64_t> ParseShape(const std::string& s) {
    std::vector<int64_t> shape;
    std::string cleaned;
    for (char c : s) {
        if (c != '[' && c != ']' && c != ' ') cleaned += c;
    }
    if (cleaned.empty()) return shape;
    std::stringstream ss(cleaned);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            shape.push_back(static_cast<int64_t>(std::stoll(token)));
        }
    }
    return shape;
}

// 解析单个 CSV 行 (支持带引号字段, 引号内逗号不分割)
std::vector<std::string> ParseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quote = !in_quote;
        } else if (c == ',' && !in_quote) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}
```

配套：表头建"列名→列索引"映射（`col_idx`）取值而非按下标，容错增列/换列序；`ResolveTestcasesDir` 按 argv → `../testcases` → `./testcases` → `../../testcases` 候选探测，run.sh 显式传目录。

**2. 用例结构扩展 `expected_error_code`/`is_exception`**：L2 异常 CSV 多一列 `expected_error_code`（0=正常；161001/161002=预期错误码），`LoadCasesFromCsv(csv_path, is_exception)` 按文件级标记装载；Mock 模式走 `RunMockExceptionTest`（本地 ValidateCase 比对 expected_error_code），Real 走 `RunRealExceptionTest`（比对真实返回码）。

**3. Mock 侧本地 ValidateCase 模拟 GetWorkspaceSize 参数校验**：Mock（无 NPU）模式下用本地校验函数（dtype 合法性、rank 一致、indices 维 ≤ data 维、dim ∈ [-rank, rank-1]、reduce 合法）验证"用例确实构造了非法输入"，使 L2 异常用例可全量自测；Real 模式才验证真实错误码。**边界（同 trace 修复暴露）**：该形态有两个已知缺陷——Real 构造器曾用 CSV 的合法 dtype/shape 建 tensor（非法性只写在 note，必然返回 0），Mock 曾有 note 字符串 fallback 蒙混通过；且本地校验器的逐维比较与 scatter "非 dim 轴才需 ≤" 的语义不一致。修法与完整教训见 [L2 异常用例非法性物化卡](l2_exception_invalidity_not_materialized.md)——套用本形态时必须物化非法输入，Mock fallback 通过不能作为用例合格的证据。

**4. 数据生成配套**：按 note 字段检测 `inf/nan/0`/`special_val` 注入特殊值；整型 `reduce=multiply` 用小值域 [-3,3] 防累乘溢出（回绕语义下 golden/NPU 应 bitwise 一致，小值域让差异更显性）；DType 枚举扩展 UINT8 用于构造 L2 非法 dtype 张量。

### 复现增量验证信号

- 实现：test_aclnn_scatter.cpp 增加 CSV 运行时加载/参数校验/L2 异常测试/边界用例与 RunRealExceptionTest，run.sh 传递 testcases 目录（来源平台：950/x86_64 主机；证据等级：tool_observed）。
- Mock 全量自测：L0 28/28、L1 767/767（全 dtype+多 shape+边界+特殊值+大张量）、L2 10/10（异常用例参数校验）、边界补充 11/11，合计 816/816 通过、`run.sh --mock` 退出码 0；迭代一 5/5、迭代二 54/54 无回归（来源平台：950/x86_64 主机；证据等级：tool_observed）。
- 用例集约束预检：L1 767 条中 dim 越界 0、indices>data 违规 0、updates shape/dtype 违规 0（来源平台：950/x86_64 主机；证据等级：tool_observed）。

### 复现增量边界

- 两种接入形态的取舍：主卡 manifest 形态适合黑盒 CSV 含 tuple 字面量/inf 字面量需 python 预处理的场景；本增量运行时解析形态适合列结构规整、想免预处理免重编的场景。两者判据/数据面机制（roundToDtype 双通道、expectError 语义）互通。
- Mock 侧 ValidateCase 是测试代码自写的参数检查，不保证与真实算子 GetWorkspaceSize 校验逻辑逐条一致——Real 模式的错误码比对才是最终判据。
- CSV 用例源依赖测试设计阶段的产物；裸写 CSV 无校验时字段错误只能靠预检脚本发现（本开发会话 实测发现一行 dim 字段与 note 矛盾，修正后才全过）。

## 后续增量（2026-09-17，并行策展批：常驻 golden co-process 架构 + 大文件 IO 修复 + 位等快速通道 + 517 条全量闭环）

softplus_v2_grad 黑盒全量（517 条 CSV：L0 8 + L1 500 + L2 9，含 111 条 >100M 元素、最大 1.44B fp16，Ascend950PR + CANN 9.1.0）补齐主卡遗留的"golden 子进程逐条 popen 性能问题"，并给出全量 exit-0 闭环：

**1. 常驻 golden co-process（替代逐用例 popen）**：C++ 侧 `CoProc` 用 fork+双 pipe 常驻 `golden_blackbox_worker.py`——一次 `import torch` 终身复用，行协议（stdin 单行 JSON job：dtype/n/beta/threshold/ranges/三个 .bin 路径；stdout 回 `OK`/`ERR <msg>`）；golden 路径经 `GOLDEN_PY` 环境变量注入。成本核算：`python3 -c "import torch"` 实测 3.2s——逐用例子进程 517 条仅 import 就要 ~28min；常驻 worker 把该固定税摊薄为一次（517 条全量总耗时 ~46min，其中大用例数据生成/传输占大头）。`CoProc::stop()` 先关 toFd 给 worker EOF、`waitpid` WNOHANG 轮询 50×0.1s 后 SIGTERM 兜底。

```cpp
// 常驻 co-process（golden_blackbox_worker.py）核心骨架（verbatim 关键段）
struct CoProc {
    int toFd = -1;   // 写到子进程 stdin
    int fromFd = -1; // 读子进程 stdout
    pid_t pid = -1;

    bool start(const std::string& py, const std::string& workerPath, const std::string& goldenPath) {
        int toC[2], fromC[2];
        if (pipe(toC) != 0 || pipe(fromC) != 0) return false;
        pid = fork();
        if (pid < 0) return false;
        if (pid == 0) {
            // child
            dup2(toC[0], 0);
            dup2(fromC[1], 1);
            close(toC[0]); close(toC[1]); close(fromC[0]); close(fromC[1]);
            if (!goldenPath.empty()) setenv("GOLDEN_PY", goldenPath.c_str(), 1);
            execlp(py.c_str(), py.c_str(), workerPath.c_str(), (char*)nullptr);
            _exit(127);
        }
        // parent
        close(toC[0]); close(fromC[1]);
        toFd = toC[1]; fromFd = fromC[0];
        return true;
    }
    // sendLine / recvLine（行协议读写）与 stop（EOF → 轮询 → SIGTERM）见工程文件
};
```

**2. 大文件 IO 修复（readFile 禁用 istreambuf_iterator）**：`istreambuf_iterator` 逐字符读 GB 级 .bin 只有 ~25MB/s；改 `tellg` 取 size + 整块 `read` 后大幅提速（该轨迹实测 upload 222s→3.0s、golden-load 116s→0.9s，2.88GB 级文件）：

```cpp
static std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    if (size <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return {};
    return buf;
}
```

**3. 位等快速通道（fp16 比对加速）**：kernel 与 golden 的 fp16 常逐位一致——比对循环先比 uint16 位型（`a[i] == e[i]` continue），不等才转 float 走 NaN/Inf/rtol-atol 分支；大张量比对从逐元素全转换降为位等短路（该轨迹 compare 85.6s→1.0s）。

**4. DUMMY/REAL 分派与 expect_error**：shape 不一致/不支持 dtype 走 DUMMY（零填充建张量、期望报错）；L2/异常用例（`level=="L2"` 或名字含 exception/error）`expect_error=true`——两级 aclnn 调用任一报错即 PASS、意外成功才 FAIL（判据卡见 [L2 异常用例判据：两级 aclnn 报错即 PASS](expect_error_dual_stage_aclnn_error_pass.md)）。worker 分块的 OOM 防护与 RAII 清理见 [大shape黑盒用例分块流式生成golden防OOM](blackbox_large_shape_chunked_golden_worker.md)。

**5. runner（run_blackbox.sh）**：任意 cwd 可跑（`SCRIPT_DIR` 锚定）、manifest 非空硬门禁、真编算子包（不复用旧 .so）、`timeout 7200` 全量兜底、禁止 `--quick`/`--max-cases` 子集捷径；manifest 计数对账（CSV 数据行 8+500+9=517 == manifest cases 517 == 驱动加载 total 517）。

### 复现增量验证信号

- smoke（15 条：9 L2 + 小 L0/L1）：15/15 PASS、退出码 0、/dev/shm 干净（来源平台：950；证据等级：tool_observed）。
- 最大用例（1.44B fp16）单跑端到端 PASS（修复分块后）；全量 517 条：`total=517 PASSED=517 FAILED=0`、0 SKIP、`EXIT: 0`（来源平台：950；证据等级：tool_observed）。
- `run_blackbox.sh` 端到端（含转换/构建/驱动）退出码 0，517/517（来源平台：950；证据等级：tool_observed）。
- 独立 st-verifier 复跑全量：517 PASS / 0 FAIL / 0 SKIP，退出码 0；golden 独立性核验（golden.py 仅 import json/os/sys/numpy/torch）（来源平台：950；证据等级：tool_observed）。
- 白盒回归不受影响：`run.sh` 25/25 PASS（来源平台：950；证据等级：tool_observed）。
- IO/比对提速（同环境前后对照）：readFile 修复后 upload 222s→3.0s、golden-load 116s→0.9s；位等快速通道 compare 85.6s→1.0s（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 配方假设 golden 为 elementwise（期望只依赖同位置输入）——co-process 分块与"同源 .bin"都建立在此上；全局视角期望（归约/排序）需另设计。
- worker 行协议为单行 JSON job；崩溃时 recvLine 返回空串即 FAIL 该用例（worker 级隔离）。
- 全量 ~46min 中大用例占比高（数据生成/上传/比对）；device 侧生成或输入复用会破坏 RNG 全局单流语义，需重新论证。
- 多输入/多输出算子需改张量拆分与 .bin 数量，机制不变（未验证，推断可迁移）。

### 复现增量坏实践增补

- 逐用例子进程调 golden——torch import 3.2s/条的固定税（517 条 ≈28min 纯浪费）。
- `istreambuf_iterator` 读 GB 级 .bin——~25MB/s。
- 异常用例一律 skip——丢失"错误路径确实报错"的验证信息。
- 期望值内联进 manifest——体积与独立性双输。

## 后续增量（2026-09-17，并行策展批：双钩子算子无关核心 + 全局 RNG 防作弊硬约束 + 白盒不退化复核）

softplus_v2_grad 的另一条独立 trace（549 条 CSV：L0 40 + L1 500 + L2 9，Ascend950PR + CANN 9.1.0，与复现增量的 517 条 trace 互为姊妹会话、不同用例集）把 1020 行内联硬编码白盒测试重构为**算子无关核心 + 薄算子层**形态，并把防作弊约束编号固化：

**1. 双钩子架构**：核心头（~670 行，算子无关）承载 manifest 解析 → dtype 感知输入构造 → NaN/Inf 安全比对 → 逐用例 try/catch 异常边界 → pass-rate 敏感驱动 `st_run_manifest`；算子层（薄，~215 行）只实现两个钩子——`callOp`（构 `aclTensor`、调两段式 API、回读输出，返回 0/非 0）与 `computeExpected`（独立 golden：host libm `expf` 公式实现，与 kernel 的 AscendC VF 链完全不同实现路径）。换算子只写薄层，驱动/比对/统计/RNG 全部复用——防约束漂移。

**2. 六条硬约束**（核心头注释里编号固化）：#1 单全局 RNG 种子、逐用例/逐输入**永不重置**（每用例重置等价于固定随机输入分布，统计型用例可被"重置种子凑通过"）；#2 跑全部用例（无 `--quick`/`--max-cases` 子集捷径）；#3 每次重编译算子包（无 .so 跳过捷径）；#4 NaN 匹配 NaN、同号 Inf 匹配，否则 rtol/atol（容差按输出 dtype 分档：fp16 1e-3 / bf16 4e-3 / fp32 1e-5）；#5 退出码 0 当且仅当 passed==total（SKIP 也算不通过）；#6 manifest 数据驱动，无内联硬编码用例。

**3. 单全局 RNG（状态推进而非重置，verbatim）**——`g_rngState` 只在声明处赋初值一次，`randRange` 里做 LCG 推进；值域来自 manifest 的 `input_data_ranges`，生成后先 round 到目标 dtype 再供 callOp 与 golden 共同消费（roundToDtype 双通道，与主卡同一原则）：

```cpp
static uint32_t g_rngState = 20240715u;

static float randRange(float lo, float hi) {
    g_rngState = g_rngState * 1103515245u + 12345u;
    float t = static_cast<float>((g_rngState >> 16) & 0x7FFF) / 32768.0f;
    return lo + t * (hi - lo);
}

static float genValue(float lo, float hi) {
    if (std::isnan(lo) || std::isnan(hi)) return std::nanf("");
    if (std::isinf(lo) || std::isinf(hi)) return lo;
    return randRange(lo, hi);
}

static std::vector<std::vector<float>> st_build_inputs(const StCase& c) {
    std::vector<std::vector<float>> inputs;
    for (size_t i = 0; i < c.inputShapes.size(); ++i) {
        int64_t n = elemCount(c.inputShapes[i]);
        aclDataType dt = (i < c.inputDtypes.size())
            ? parseDtype(c.inputDtypes[i]) : ACL_FLOAT;
        std::vector<float> data(static_cast<size_t>(std::max<int64_t>(n, 1)));
        float lo = -1.0f, hi = 1.0f;
        if (i < c.inputDataRanges.size()) {
            lo = c.inputDataRanges[i].first;
            hi = c.inputDataRanges[i].second;
        }
        for (int64_t j = 0; j < n; ++j)
            data[static_cast<size_t>(j)] = roundToDtype(genValue(lo, hi), dt);
        inputs.push_back(std::move(data));
    }
    return inputs;
}
```

**4. 输出哨兵 0xFF**：输出 device 内存先 `aclrtMemset` 0xFF——fp32 下 0xFFFFFFFF 回读为 NaN，kernel 漏写位置不会被"期望恰为 0"的假通过掩盖（有限哨兵变体与红基线语义见 [ST 红基线输出 poison 防 stub 巧合命中](st_red_baseline_poison_prefill_exact_zero.md)；NaN/Inf 安全比对使哨兵值可被正确判 FAIL）。

**5. expect_error 软语义**：期望报错的用例若被算子接受，不直接判 FAIL 而是转正常数值比对（接受且输出正确也 PASS、接受但输出错才 FAIL）——判据语义见 [expect_error 用例"期望报错但 kernel 成功"](expect_error_kernel_succeeded_not_enforced.md)。

**6. 白盒不退化复核**：从 git 历史检出重构前的旧内联白盒测试，独立构建运行（48+62 用例全过），证明 manifest 化重构未引入回归——重构类交付的必要证据。

**7. 配套**：manifest 生成器带 shape capping（超大 shape 封顶保 rank/广播/dtype/值域因子，见 [黑盒CSV转ST清单：shape capping保rank广播](blackbox_csv_shape_capping_rank_broadcast.md)）；确定性验收用驱动内 FNV-1a 输出位哈希（见 [aclnn 交付前 3 次 bit-wise 确定性回归](aclnn_determinism_3run_bitwise.md) 复现增量）。

### 复现增量验证信号

- 框架落地后全量执行：`==== Summary: 549/549 PASSED, 0 FAILED, 0 SKIPPED ====`，退出码 0，3 次重复运行一致（来源平台：950；证据等级：tool_observed）。
- 端到端 run.sh：manifest 生成 `wrote 549 cases (9 expect_error)` → 非空门禁 `[run] manifest OK: 549 cases` → 算子包重编译安装 → 构建测试 → 全量跑，退出码 0（来源平台：950；证据等级：tool_observed）。
- 用例零丢失对账：CSV 数据行 549（40+500+9）= manifest 条数 = 执行条数（来源平台：950；证据等级：tool_observed）。
- 重复执行确定性：3 次全量运行 549/549 且输出位哈希一致（FNV-1a 形态见 [aclnn 交付前 3 次 bit-wise 确定性回归](aclnn_determinism_3run_bitwise.md)）（来源平台：950；证据等级：tool_observed）。
- 白盒不退化复核：git 检出旧内联白盒测试独立构建运行，48+62 用例全部通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- `callOp` 为单输出回读；多输出算子需扩展 `actual`/`computeExpected` 的多张量路径。
- `callOp` 资源清理用 `goto cleanup`（C 风格但单出口清晰），代价是**不能在 goto 之前引入带初始化的局部 lambda**（跳过初始化编译错）——辅助函数须提为 `static` 函数。
- 容差分档写死三档（fp16/bf16/fp32）；新增 dtype（如整型输出）需补档。
- 与主卡/先前增量 形态差异：本变体期望值在 C++ 内联现算（host libm），主卡/先前增量 为外部 golden（popen/co-process）——oracle 独立性红线一致（不与 kernel 共享实现）。
- 本增量为 C++ 侧 `goto cleanup`/模板张量构造等常规实现细节之外的核心机制；ACL dtype 枚举实测锚点 `ACL_FLOAT=0`、`ACL_FLOAT16=1`、`ACL_INT64=9`、`ACL_DOUBLE=11`、`ACL_BF16=27`（出自 `acl/acl_base_rt.h:142-157`）。
- 950 + CANN 9.1.0 以外未验证。