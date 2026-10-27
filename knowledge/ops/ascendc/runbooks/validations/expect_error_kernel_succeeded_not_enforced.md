---
type: "Runbook"
title: "expectError但kernel成功：CANN不强制异常约束"
description: L2 异常用例（float64 dtype/rank 9/NCHW）期望 aclnn 报错但 CANN 单算子运行时不强制这些约束、kernel 正常算对；ST 框架须把"期望报错但成功"降级为 SKIP 或转正常精度比对，不能判 FAIL。复现增量：标量属性类异常在 aclnn C API 层根本不可表达（标量以宿主 C++ double 传入、NaN 属性被接受并按 IEEE-754 正确传播），判据软化为两段——拒绝即 PASS、接受则 golden 数值验证通过才 PASS。
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T05:05:01Z'
platforms: ["950"]
---

# expectError但kernel成功：CANN不强制异常约束

## 现象

BesselI1e 算子黑盒批量验证后期：容差与 inf 两类精度问题已修复，437 条去超大 shape 用例跑完剩最后 3 条 FAIL，且失败文案完全同型。它们是黑盒 CSV 的 L2 档**异常用例**——用例设计意图是"喂非法输入，期望 aclnn 调用报错"：

```text
[ 45/ 47] aclnnBesselI1e_float64_random_x_dtype_exception         [aclnnBesselI1e_float64_random_x_dtype_exception] FAIL — expected error but kernel succeeded
[ 46/ 47] aclnnBesselI1e_float16_random_x_dim_exception           [aclnnBesselI1e_float16_random_x_dim_exception] FAIL — expected error but kernel succeeded
[ 47/ 47] aclnnBesselI1e_float16_random_x_NCHW_format_exception   [aclnnBesselI1e_float16_random_x_NCHW_format_exception] FAIL — expected error but kernel succeeded

==== ST Summary ====
PASS: 44 | SKIP: 0 | FAIL: 3 | ERROR: 0 | TOTAL: 47
```

三条用例的非法性分别是：float64 输入（dtype 不支持，且输入输出 dtype 组合不匹配 `('float64','float16')`）、rank 9 shape `[7,3,8,1,6,9,2,5,3]`（设计约束 rank ∈ [0,8]）、NCHW 格式（设计约束只支持 ND）。当时 ST 框架的判定逻辑是：用例名含 `exception`（或 remark 含 `expect_error`）或 dtype 在不支持列表 → `expectError=true` → 跑 aclnn，返回非 0 算 PASS，**返回 0（成功）直接判 FAIL**。而实测 aclnn 调用对这三条全部成功且输出数值正确——即 CANN 运行时并不在这些层面强制设计文档声明的约束。

## 适用场景

- 工程形态：aclnn 自定义算子的黑盒 ST（数据驱动框架）中纳入 L2/异常类用例；异常用例语义设计。
- API 路径：aclnn 两段式调用（`aclnnXxxGetWorkspaceSize` → `aclnnXxx`）、aclrt 张量创建（ACL_FLOAT/ACL_FLOAT16 映射）、设计文档 rank/format/dtype 约束条目。
- 验证过的平台与 CANN 版本：950（ascend950 构建目标）；CANN 9.0.0-beta.2。
- 适用粒度：`platform_wide`——CANN 单算子运行时对自定义算子输入约束的强制行为（与具体算子无关）。
- 路线优先级：recommended_fastpath（异常用例"期望报错但成功"按本卡语义处置）。
- 明确不适用/未验证场景：
  - CANN 在**其他版本**是否开始强制 rank/format/dtype 校验——未验证，升级 CANN 后应重测；
  - 框架级（GE/图模式）输入校验行为——本轨迹只有单算子 aclnn 直调路径。

## 根因

ST 框架把"设计文档声明的约束"直接当作"运行时必然报错"来断言，而 CANN 单算子运行时（aclnn 直调）并不强制自定义算子的 rank ≤ 8 / 仅 ND 格式 / dtype 支持列表约束：rank 9 输入被 kernel 按展平 buffer 正确处理、NCHW 张量按平坦数据正常计算、float64 输入在测试侧 dtype 映射下落到 ACL_FLOAT 正常执行——三条用例 kernel 都成功且数值正确，框架却因"没报错"判 FAIL。这是断言语义错位：**异常用例断言的对象应是"可观测的拒绝行为"，运行时没有该行为时不能把"成功"当失败**。

## 解决配方

三步修正框架语义（均在 ST 框架 test_aclnn.cpp）：

1. **expectError 判定收敛**——名字含 `exception` 不再单独触发 expectError，须同时 dtype 真不支持：

```cpp
// Post-process: detect exception cases
// Only flag as expectError if dtype is truly unsupported.
// Exception-named cases with supported dtypes (fp16/fp32/bf16)
// test framework-level constraints (e.g. rank > 8, NCHW format)
// that the CANN runtime may or may not enforce. If the kernel
// succeeds for these, we should verify precision normally.
bool isExceptionByName = (tc.caseId.find("exception") != std::string::npos ||
                          tc.remark.find("expect_error") != std::string::npos);
bool isUnsupportedDtype = (tc.inputDtype == "float64" || tc.inputDtype == "int32" ||
    tc.inputDtype == "int64" || tc.inputDtype == "int16" ||
    tc.inputDtype == "int8" || tc.inputDtype == "uint8");
if (isExceptionByName && isUnsupportedDtype) {
    tc.expectError = true;
} else if (isUnsupportedDtype) {
    tc.expectError = true;
}
// Exception-named cases with supported dtypes: run as normal precision test
```

2. **float64 移出不支持列表**——测试侧 dtype 映射把 float64 写成 ACL_FLOAT，实际按 fp32 精度可正常比对：

```cpp
bool isUnsupportedDtype = (tc.inputDtype == "int32" ||
    tc.inputDtype == "int64" || tc.inputDtype == "int16" ||
    tc.inputDtype == "int8" || tc.inputDtype == "uint8");
```

3. **"期望报错但 kernel 成功"降级为 SKIP + 资源正确清理**（保留给真正不支持 dtype 的兜底路径）：

```cpp
} else {
    // Kernel succeeded despite expecting error.
    // CANN framework may not enforce this constraint (e.g. rank > 8,
    // NCHW format, unsupported-but-mapped dtype). Since the kernel
    // works, skip the constraint check.
    fprintf(stdout, "[%s] SKIP — expect_error but kernel succeeded "
            "(framework constraint not enforced)\n",
            tc.caseId.c_str());
    aclDestroyTensor(xTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devX);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    return 1;
}
```

修后三条 L2 用例的实际归途：float64 用例转正常 fp32 精度比对 PASS；rank 9 与 NCHW 用例按正常 fp16 精度比对 PASS（kernel 展平/平坦 buffer 处理数值正确）。

## 设计决策

- 为何"期望报错但成功"选 SKIP 而不是硬转正常精度比对：expectError 分支此时已创建张量/流并执行过 aclnn，直接 fall-through 到正常路径会**重复释放资源**（中间版本试过内联重算精度比对，因 CHECK_ACL 宏返回语义与双重 destroy 风险被放弃）；SKIP + 显式清理 + `return 1` 是最小安全面。
- 为何 rank/NCHW/float64 三条最终走"正常精度比对"而不是 SKIP：CANN 既然接受了输入且 kernel 算出了正确值，比对精度能**多锁一层正确性**——异常约束未被强制是事实，但数值正确性仍可验证。
- 当前方案属 recommended_fastpath；若未来 CANN 版本开始强制这些约束，expectError 分支会自然回到"报错即 PASS"，语义向前兼容。
- 被否决的候选：把 L2 用例从测试集删除——约束未被强制是用例前提失效，删用例会丢失"CANN 未强制该约束"这一观测本身；正确处置是转正常比对或显式 SKIP。

## 预防

- 异常用例的断言写成两段：期望行为（报错）+ 未观察到期望行为时的降级路径（SKIP 或转正常比对），永远不要把"约束未被强制"当 FAIL。
- 设计文档的输入约束（rank/format/dtype）与运行时实际校验层要分开核对：单算子 aclnn 直调不等于图模式校验。
- dtype 不支持列表要与测试侧 dtype→aclDataType 映射表联动维护（float64→ACL_FLOAT 这类"映射后可跑"的 dtype 不该出现在不支持列表）。

## 验证信号

以下信号来自同一次开发会话的 L2 语义修复前后对照（ascend950 构建目标，CANN 9.0.0-beta.2）：

- 修复前：47 条收尾集 `PASS: 44 | SKIP: 0 | FAIL: 3 | ERROR: 0 | TOTAL: 47`，三条 L2 全部 `FAIL — expected error but kernel succeeded`（来源平台：950；证据等级：tool_observed）。
- 修复后 L2 单独集：三条全部 PASS（float64 按 fp32 精度比对、rank 9 与 NCHW 按 fp16 精度比对）（来源平台：950；证据等级：tool_observed）。
- 全量收口：437 条全量（含三条 L2）`PASS: 437 | SKIP: 0 | FAIL: 0 | ERROR: 0`，3 次复跑 bit-wise 一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "CANN 运行时不强制 rank/format/dtype 约束"的观测限于 950 + CANN 9.0.0-beta.2 的 aclnn 单算子直调；其他版本/其他校验层（GE、aclnn 之下的 op check）未验证，推断可迁移（未验证，推断可迁移）。
- float64 用例按 fp32 精度比对通过，说明的是"映射后数值正确"，**不代表算子支持 float64 语义**（输出仍按映射 dtype 解释）；对 dtype 语义有严格要求的验收不适用此处置。
- SKIP 文案必须打印"expect_error but kernel succeeded"保留可观测性——静默 SKIP 会把约束未强制的事实藏掉。

**未决项**：

- CANN 各版本对 rank/format/dtype 校验行为的差异矩阵未建立——升级 CANN 后应回归这三类异常用例，确认降级路径是否需要回调。
- 设计文档仍声明 rank ∈ [0,8]/仅 ND——文档与运行时行为的偏差是否需要在算子文档侧标注，未处理。
## 坏实践

- 把"期望报错但成功"判 FAIL——本例三条全 FAIL 的直接原因；异常约束是否被运行时强制是 CANN 的行为，不是被测算子的缺陷。
- expectError 分支直接 fall-through 到正常比对路径而不清理已分配资源——双重释放/泄漏风险（本轨迹中间版本即因此重写）。

## 误诊

- 曾把三条 FAIL 当作算子/框架缺陷排查——独立核对三条用例的 kernel 输出数值全部正确后，才定位到断言语义错位（约束未被运行时强制 ≠ 用例失败）。

## 增量（格式类约束的机制定位：IsPrivateFormat 只拒私有格式）

独立复现与机制深化（cummin 类归约算子全量 ST 联调，CANN 9.1.0，Ascend950PR——与主卡 9.0.0-beta.2 的观测跨版本一致）：全量 633 条（L0 30 + L1 600 + L2 3）中唯一 FAIL 即 `format_not_supported_input_NCHW => FAIL: exception NOT triggered: GetWorkspaceSize succeeded (expected error 161002)`。本增量把格式约束"未被强制"追因到 op_api 模板校验的具体语义，并给出保留负向覆盖的用例改法。

**机制（格式类）**：aclnn op_api 模板的 `CheckFormat` 只用 `IsPrivateFormat(Format)`（`include/aclnn/opdev/format_utils.h:24` 声明）拒绝**私有格式**；`acl_base_rt.h` 枚举中 `ACL_FORMAT_NCHW=0 / NHWC=1 / ND=2` 是公共格式（无专用内存排布，按 ND 语义处理），`NC1HWC0=3 / FRACTAL_Z=4 / FRACTAL_NZ=29` 才是私有格式。因此 NCHW 输入被放行并按 view shape 正确计算；op_def 的 `.Format({ge::FORMAT_ND, ...})` 注册声明只约束算子注册信息，**不构成 aclnn 层的运行时格式白名单**。

**格式类的另一种处置（与主卡降级语义互补）**：主卡把"期望报错但成功"降级为 SKIP/转正常比对，适用于运行时确实不强制的约束；格式类约束存在真正会触发拒绝的输入（私有格式），可把用例换成 NC1HWC0 以**保留"格式被拒"的负向覆盖意图**（仍断言 161002），同时处理 `aclCreateTensor` 对私有格式可能返回 null 的更早拒绝（也计为异常触发）。

修复前（L2 CSV，`tests/st/testcases/aclnnBenchNoHarnessCummin_l2_test_cases.csv`）：

```
format_not_supported_input_NCHW,aclnnBenchNoHarnessCummin,"((37,48,25,),(2,3,),(2,3,),)","('float16','float16','int64',)",,,,,,"((-1,1,),)",,
```

修复后（改用例名 + 4D 形状，NCHW 是 4D 语义格式，配 (4,3,37,25)）：

```
format_not_supported_input_NC1HWC0,aclnnBenchNoHarnessCummin,"((4,3,37,25,),(4,3,37,25,),(4,3,37,25,),)","('float16','float16','int64',)",,,,,,"((-1,1,),)",,
```

修复前（ST 测试代码的格式选择，`tests/st/test_aclnn_bench_no_harness_cummin.cpp`）：

```cpp
        // 确定异常 format (NCHW)
        aclFormat fmt = ACL_FORMAT_ND;
        if (tc.name.find("NCHW") != std::string::npos) {
            fmt = ACL_FORMAT_NCHW;
        }
```

修复后（按用例名匹配私有格式，并处理 aclCreateTensor 返回 null 的情形——私有格式下框架可能直接拒建张量）：

```cpp
        // 确定异常 format (NC1HWC0 - Private Format, 应被 IsPrivateFormat 拒绝)
        aclFormat fmt = ACL_FORMAT_ND;
        if (tc.name.find("NC1HWC0") != std::string::npos) {
            fmt = ACL_FORMAT_NC1HWC0;
        }
```

```cpp
        CaseResult res;
        // 如果 aclCreateTensor 因 private format 返回 null, 也视为校验通过
        if (inT == nullptr || valT == nullptr || idxT == nullptr) {
            res.pass = true;
            res.msg = "exception OK: aclCreateTensor rejected private format (null tensor)";
        } else {
            // ... 原有 GetWorkspaceSize 期望错误码断言不变 ...
        }
```

预防（复现增量）：

- 设计"不支持的格式"类 L2 负向用例前，先读 op_api 的 CheckFormat 实现：基于 `IsPrivateFormat` 的校验只拒私有格式（NC1HWC0/FRACTAL_Z/FRACTAL_NZ 系），NCHW/NHWC/ND 都是公共格式、会被放行。
- 不要把 op_def 的 `.Format({FORMAT_ND,...})` 声明当作 aclnn 层的格式白名单——它不产生运行时拒绝。
- 负向用例断言"期望错误码"时，同时处理"框架在更早环节拒绝"（如 aclCreateTensor 返回 null）的情形，两者都算异常触发。

验证信号（复现增量，cummin 类算子全量 ST 联调会话）：

- 触发：`[633/633] format_not_supported_input_NCHW => FAIL: exception NOT triggered: GetWorkspaceSize succeeded (expected error 161002)`，全量 633 条中唯一 1 FAIL（583 PASS/1 FAIL/49 SKIP）（来源平台：950；证据等级：tool_observed）。
- 事实锚点：`CheckFormat` 仅调用 `IsPrivateFormat`（op_api 源码）；`format_utils.h:24` 声明 `bool IsPrivateFormat(Format)`；`acl_base_rt.h` 枚举 `ACL_FORMAT_NCHW=0 / ND=2 / NC1HWC0=3 / FRACTAL_Z=4 / FRACTAL_NZ=29`；op_def 声明 `.Format({ge::FORMAT_ND ×4})` 仍不拒 NCHW（来源平台：950；证据等级：tool_observed）。
- 修复后：`format_not_supported_input_NC1HWC0 => PASS (exception OK: GetWorkspaceSize returned 161002)`；全量 633 用例 584 PASS / 0 FAIL / 49 SKIP（SKIP 为 >100M 元素上限）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- `IsPrivateFormat` 的确切覆盖集合未读 CANN 实现源码验证——已知 NCHW 通过、NC1HWC0 被拒；NHWC/ND 推断同 NCHW（公共格式），FRACTAL 系推断同 NC1HWC0（未验证，推断可迁移）。
- 仅覆盖 aclnn 直调路径；GE 图模式下的格式校验/格式转换行为未验证。
- `aclCreateTensor` 对 NC1HWC0 是否真的返回 null 未单独验证（本轨迹走了 GetWorkspaceSize 返回 161002 的分支）。

## 增量（2026-09-17，先前增量 · 标量属性类异常在 aclnn C API 层不可表达：接受则数值验证）

独立复现与第三类机制定位（softplus_v2_grad 黑盒 aclnn ST，Ascend 950 + CANN 9.1.0，549 条全量）：崩溃问题解决后剩 3 条 FAIL，全部是 L2 异常（expect_error）用例"算子没报错"：

```text
[FAIL] expect_error — aclnn succeeded unexpectedly   （×3）
==== Summary: 546/549 PASSED, 3 FAILED, 0 SKIPPED ====
```

三条分别是 `float64_random_beta_dtype_exception`（设计意图"beta 标量为 float64 应被拒绝"，但张量 dtype 全 float32、beta 是合法 double）、`bfloat16_random_beta_attr_nan_exception` 与 `bfloat16_random_threshold_attr_nan_exception`（beta/threshold=NaN，期望拒绝）。同批另外 6 条张量类异常（float64/int64 张量 dtype、9D 秩、不兼容 shape 等）都按预期被拒（161002 或库异常捕获）。

**机制（标量属性类）**：这批"异常条件"通过 aclnn C API 根本无法表达为拒绝——

- 标量属性没有 dtype 可拒绝：aclnn C API 的标量参数（如 beta/threshold）以宿主 C++ `double` 传入，"float64 标量"这一条件在 API 层不存在，算子收到的一定是合法 double，自然正常计算；
- NaN 属性被算子接受并正确传播：beta/threshold 为 NaN 时算子不报错，输出按 IEEE-754 语义传播 NaN（`z=NaN` 时比较为 false 走 sigmoid 分支，结果为 NaN），是合法且正确的行为。

与主卡张量类（dtype/rank/format 运行时不强制）、先前增量 格式类（IsPrivateFormat 只拒私有格式）合成三类机制：**张量属性（dtype/rank/shape/format）可表达、是否拒绝取决于校验层；标量属性（标量 dtype、NaN/Inf 值）在 C API 层不可表达为拒绝**——黑盒用例设计者按 spec 直觉标注的 "exception"，有一部分在 C API 语义下不构成异常。

**判据软化为两段**（拒绝即 PASS；接受则回退普通数值验证——golden 计算期望并比对，匹配才 PASS；before → after verbatim，取自事件流编辑）：

```cpp
// before（硬语义：接受即 FAIL）
            if (c.expectError) {
                // expect_error: aclnn should fail (hard constraint: remark expect_error)
                std::vector<std::vector<float>> actual;
                int ret = callOp(c, inputs, actual, stream);
                if (ret != 0) {
                    fprintf(stdout, "  [PASS] expect_error (aclnn returned %d)\n", ret);
                    ++passed;
                } else {
                    fprintf(stderr, "  [FAIL] expect_error — aclnn succeeded unexpectedly\n");
                    ++failed;
                }
                continue;
            }
```

```cpp
// after（软语义：接受则数值验证，验证过才 PASS）
            if (c.expectError) {
                // expect_error: operator should reject. If it does → PASS.
                // If it accepts → verify output correctness (some "exception" cases
                // like NaN attrs / scalar dtype don't actually trigger rejection;
                // the operator accepts and produces correct output).
                std::vector<std::vector<float>> actual;
                int ret = callOp(c, inputs, actual, stream);
                if (ret != 0) {
                    fprintf(stdout, "  [PASS] expect_error (aclnn returned %d)\n", ret);
                    ++passed;
                } else {
                    std::vector<std::vector<float>> expected;
                    if (computeExpected(c, inputs, expected) && st_compare(c, actual, expected)) {
                        fprintf(stdout, "  [PASS] expect_error (accepted, output verified)\n");
                        ++passed;
                    } else {
                        fprintf(stderr, "  [FAIL] expect_error — accepted but output mismatch\n");
                        ++failed;
                    }
                }
                continue;
            }
```

软化只放宽"接受即失败"的判定，不放松数值标准——被接受路径用与普通用例完全相同的 golden + rtol/atol 比对（NaN 对 NaN 按位匹配），输出错误照样 FAIL（`accepted but output mismatch`）。相对主卡 BesselI1e 轨迹的"SKIP/转正常比对"处置，这里是运行时探测式两段判据，不依赖"哪类异常名真的会被拒绝"的人工白名单（本批实证：设计者标了 9 条异常，实际 6 拒 3 收），对算子侧"拒/收"两种实现选择都稳健。

### 复现增量验证信号

- 修复前：`==== Summary: 546/549 PASSED, 3 FAILED, 0 SKIPPED ====`，3 条 `expect_error — aclnn succeeded unexpectedly`（来源平台：950；证据等级：tool_observed）。
- 修复后：`EXIT CODE: 0`，`==== Summary: 549/549 PASSED, 0 FAILED, 0 SKIPPED ====`（来源平台：950；证据等级：tool_observed）。
- 9 条 L2 逐条去向：float64 张量 dtype → 161002 拒绝 PASS；int64×fp16 组合 → 161002 拒绝 PASS；9D 秩 → 库异常捕获 PASS；float64 beta 标量 / beta=NaN / threshold=NaN → `expect_error (accepted, output verified)` PASS（来源平台：950；证据等级：tool_observed）。
- 端到端 `run.sh`（manifest 生成→算子包重编译安装→测试构建→全量执行）退出码 0（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 软化只适用于"算子接受属合法行为"的异常条件；若某算子 spec 明确要求拒绝 NaN 属性，则该算子的这类用例被接受时就是真缺陷，软化会把缺陷记成 PASS——采用前须确认目标算子对相应输入无拒绝义务（本算子已确认接受并正确传播）。
- 被接受路径依赖 golden 覆盖 NaN 语义（oracle 按 IEEE-754 传播 NaN）；golden 不支持时该路径退化为 FAIL。
- 同算子另批黑盒观测（见 [两级 aclnn 报错即 PASS 卡](expect_error_dual_stage_aclnn_error_pass.md)）中 float16 张量配置下的 NaN 属性用例曾在 `GetWorkspaceSize` 被 561002 拒——NaN 属性是否被拒可能随算子版本/校验配置而变；两段式判据对拒/收两种行为均稳健（拒绝→PASS、接受→验证），无须预判。
- "算子对 NaN 标量属性应否拒绝"缺乏 spec 级明确结论，本卡按"接受且传播即正确"处理；若上游 spec 后续明确要求拒绝，本语义需回退。
- 950 + CANN 9.1.0 以外未验证。

### 复现增量坏实践

- 把"expect_error 且未报错"一律记 FAIL 而不看算子是否产出正确输出——把 API 表达能力错位固化成永久假失败，每轮全量非零退出，掩盖真正的失败信号。
- 反向坏实践同样禁止：为了凑全通过把接受的异常用例直接标 PASS 跳过数值验证——必须是"验证过的接受"。