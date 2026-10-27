---
type: "Runbook"
title: OP_CHECK 断言条件写反致 161002 拒合法输入
description: "aclnn op_api 参数校验宏 OP_CHECK 断言条件为真，把 `!=` 写成断言条件会让所有合法 dtype/shape 输入被拒并报 161002（ACLNN_ERR_PARAM_INVALID）；三处条件反转改 `==` 后恢复，判别靠 plog 校验行号 + strings 报错文案归属。"
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# OP_CHECK 断言条件写反致 161002 拒合法输入

## 现象

在给自定义算子（Add→RMSNorm→DynamicQuant 类结构，输入 x1/x2/gamma，输出 y(int8)/xOut/scaleOut）做首轮 Real 模式 ST 联调时，编译安装算子包后运行 C++ 原生 ST 测试：CPU Golden 自测 7 项全部通过，但 10 条 L0 用例**全部**在第一阶段就失败，报错完全一致：

```
[FAIL] GetWorkspaceSize 失败: 161002 (=161002)
```

用例覆盖 fp16/bf16、M=1~4096、N=128~16384，输入 dtype 和 shape 均按算子定义构造（y 是 INT8、scaleOut 是 FLOAT、x1 与 x2 shape 完全一致），却仍被参数校验拒绝。打开 plog（`ASCEND_GLOBAL_LOG_LEVEL` 提级后）定位到具体触发点：

```
[ERROR] OP ... [aclnn_..._dynamic_quant.cpp:108][NNOP][CheckShape] ... errno[161002]
  OpName:[aclnn..._0] x1 and x2 shape must be identical.
[ERROR] OP ... [aclnn_..._dynamic_quant.cpp:123][NNOP][CheckParams] ... errno[161002] ...
```

同时 strings 检查已安装的 `libcust_opapi.so`，其中的校验文案包括 `y dtype must be INT8, got %d.`、`scaleOut dtype must be FLOAT, got %d.`——即报错来自本算子自己的 op_api 检查代码，而不是环境或框架。

当时的 op_api 代码是首轮生成的版本，`CheckParams/CheckShape` 里用 `OP_CHECK` 做参数校验。

## 适用场景

- 工程形态：aclnn 自定义算子（op_api 层 `aclnnXxxGetWorkspaceSize` 的参数校验），Real 模式 ST 首跑联调。
- API 路径：`OP_CHECK(cond, OP_LOGE(ACLNN_ERR_PARAM_INVALID, ...), return false)` 校验宏；`CheckParams`/`CheckShape` 辅助函数。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——OP_CHECK 宏语义与平台无关，任何版本 aclnn op_api 校验代码均适用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`OP_CHECK_IF(cond, ...)` 是"cond 为真即报错并 return"的反向语义，与 `OP_CHECK` 不同，不要套用同一改法（本会话核对过 op_host 侧 `OP_CHECK_IF(storageShape.GetDimNum() != 2, ...)` 是正确写法）。

## 根因

`OP_CHECK(cond, log, ret)` 的语义是**断言 cond 为真**：cond 为 false 时打印 log 并执行 ret。首轮生成的校验代码把"期望相等"写成了 `!=`，于是**合法输入反而触发失败分支**，报 161002（ACLNN_ERR_PARAM_INVALID）：

- `y->GetDataType() != DataType::DT_INT8` —— y 恰好是 INT8 时条件为假 → 触发报错；
- `scaleOut->GetDataType() != DataType::DT_FLOAT` —— 同理；
- `x1->GetViewShape() != x2->GetViewShape()` —— shape 一致时反而报 "must be identical"。

三处是同一根因（断言条件方向写反），一处修复模式。

## 解决配方

修复前（op_api，`!=` 写进断言，合法输入被拒）：

```cpp
    // y 固定 int8
    OP_CHECK(y->GetDataType() != DataType::DT_INT8,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must be INT8, got %d.",
                     static_cast<int>(y->GetDataType())),
             return false);
    // xOut dtype 与 x1 一致
    OP_CHECK_DTYPE_NOT_MATCH(xOut, x1->GetDataType(), return false);
    // scaleOut 固定 fp32
    OP_CHECK(scaleOut->GetDataType() != DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scaleOut dtype must be FLOAT, got %d.",
                     static_cast<int>(scaleOut->GetDataType())),
             return false);
```

```cpp
    // x1 与 x2 shape 必须完全一致
    OP_CHECK(x1->GetViewShape() != x2->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 shape must be identical."),
             return false);
```

修复后（断言"期望条件"，`==`）：

```cpp
    // y 固定 int8
    OP_CHECK(y->GetDataType() == DataType::DT_INT8,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must be INT8, got %d.",
                     static_cast<int>(y->GetDataType())),
             return false);
    // xOut dtype 与 x1 一致
    OP_CHECK_DTYPE_NOT_MATCH(xOut, x1->GetDataType(), return false);
    // scaleOut 固定 fp32
    OP_CHECK(scaleOut->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scaleOut dtype must be FLOAT, got %d.",
                     static_cast<int>(scaleOut->GetDataType())),
             return false);
```

```cpp
    // x1 与 x2 shape 必须完全一致
    OP_CHECK(x1->GetViewShape() == x2->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 shape must be identical."),
             return false);
```

修复后重新 `bash build.sh --soc=ascend950` 并 `bash custom_opp_ubuntu_x86_64.run --quiet --force` 安装、重跑 ST。

判别 / 定位方法（本会话走通）：

1. 161002 在 `aclnnXxxGetWorkspaceSize` 阶段统一报出且**所有用例同报**（与输入 shape/dtype 无关）→ 优先怀疑 op_api 校验逻辑本身，而非数据构造。
2. 提升日志级别跑一次并 grep plog 中 `errno[161002]` 的函数名与行号（CheckShape/CheckParams），直接定位到具体校验点。
3. `strings libcust_opapi.so | grep "must be"` 确认报错文案来自本算子代码（排除环境/旧库）。

## 设计决策

- 修复方向选择"把断言条件改对"而不是"删掉校验"：参数校验是 aclnn 接口契约的一部分，删除会让非法输入穿透到 kernel 期才爆，定位成本更高。
- 同一轮顺手 grep 了 `OP_CHECK.*!=` 全部命中并逐条核对语义，确认只剩 `OP_CHECK_IF`（反向语义，正确）——同类 bug 一次清干净，避免"修一处跑一次再爆下一处"。
- 当前方案路线档位：recommended_fastpath（正确性必需）。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510/arch35），CANN 9.1.0，npu-smi 25.7.rc1。
- 相关文件：`op_api/aclnn_<op>.cpp`（CheckParams/CheckShape 修复点）、`tests/st/`（ST 工程，plog 定位与验证）。
- 复现命令：`cd tests/st && bash run.sh --real`（修复前 10/10 报 161002）。

## 预防

- 写 `OP_CHECK` 时先读一遍宏定义再落笔：它断言**条件为真**，校验"必须等于"就写 `==`。
- ST 全用例在 GetWorkspaceSize 阶段同报 161002 且 Golden 自测通过时，直接查 op_api 校验条件方向，不要怀疑数据构造。
- Code review 对生成的 op_api 校验代码重点看断言方向（`!=`/`==`）与日志文案是否自洽（日志说 "must be identical" 而条件是 `!=`，本身就是矛盾信号）。

## 验证信号

以下信号来自该算子首轮 Real 模式 ST 联调（Ascend950PR，CANN 9.1.0）：修复前后各跑一轮全量 L0 用例对比。

- 修复前：10/10 L0 用例 `GetWorkspaceSize 失败: 161002`；plog 显示 `errno[161002] x1 and x2 shape must be identical.` 与 dtype 校验报错（来源平台：950；证据等级：tool_observed）。
- 修复后（dtype + shape 三处 `!=`→`==`，重编译安装）：GetWorkspaceSize 全部通过，ST 进入 kernel 执行阶段（后续剩余失败均为其他独立根因，见对应卡片）（来源平台：950；证据等级：tool_observed）。
- 同轮 FullLoad 路径用例最终 8/8 通过（M=1~4096、N=128~8192，fp16/bf16，y 绝对误差≤1、xOut MERE≤7.5e-05、scaleOut MERE≤1e-06 量级）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"断言方向写反"这一根因；161002 还可由 genuinely 非法参数触发（真传错 dtype/shape 时是正确拦截）。
- `OP_CHECK_DTYPE_NOT_MATCH(xOut, x1->GetDataType(), return false)` 这类封装宏语义随 CANN 版本可能不同，未在本会话逐宏核对（未验证）。
- 其他 CANN 版本的 OP_CHECK 语义未验证，推断可迁移（宏契约层面的行为）。

**未决项**：

- 无（该根因已闭环到机制层：宏语义 + 条件方向）。
## 坏实践

- 把 `OP_CHECK(cond, ...)` 当成 "cond 为真就报错" 来写（那是 `OP_CHECK_IF` 的语义）。
- 看到 161002 先怀疑测试数据构造或环境，而不先 grep plog 拿到具体失败的校验行号。

## 误诊

- 首次看到 161002 时曾怀疑"安装的 libcust_opapi.so 是旧版本未含修复"：核对库 mtime（10:21）晚于源码修改（10:20）、strings 中含最新报错文案后排除——问题就在当前代码里。

## 增量（同根因在 L0/L2 成对出现并派生 561103）

独立复现 R2（MoE FinalizeRouting 算子 ST 上板联调，CANN 9.1.0，Ascend950PR）：同一极性错误的新形态与增量判别法。

**新形态 1——报错日志实参与断言自相矛盾**：`OP_CHECK(!IsDtypeSupported(epDtype), ...)` 让 dtype 支持时反而报错，plog 打出 `expanded_permuted_rows dtype not supported: dtype=1. Supported: FLOAT16, FLOAT, BFLOAT16.`（dtype=1 就是 DT_FLOAT16，在支持列表里却被判不支持）；`OP_CHECK(dimNum < 2, ...)` 让 dim num 是 2 时报 `dim num must be >= 2, got 2`。"合法值被判非法"（日志 got X 且 X 满足断言）是本错误的指纹。

**新形态 2——L0 InferShape 层同型错误派生 561103**：修完 aclnn 层后，L0 API 的 InferShape 文件以同型错误再次暴露（`dim num must be >= 2, got 2`），且 InferShape 失败导致 opResult 为空指针，下游派生 `561103: check opResult != nullptr failed`——本根因在 L2 与 L0 两个文件**成对出现**，只修第一处，第二处会以 561103 形式再次暴露。

**诊断探针增量**：在报错日志里追加判定变量打印（如 `IsSupported=%d`）再复跑，观测到 `IsSupported=1` 时错误仍触发，即可把怀疑从"判定函数写错"转向"宏极性写反"——比直接读宏定义先缩小范围。

**修复（全部改为正向守卫条件，两个文件共 13 处，before/after 节选）**：

```cpp
    // before：错误条件直接进 OP_CHECK（合法输入被拒）
    OP_CHECK(!IsDtypeSupported(epDtype),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "expanded_permuted_rows dtype not supported: dtype=%d. Supported: FLOAT16, FLOAT, BFLOAT16.",
                     static_cast<int>(epDtype)),
             return false);

    OP_CHECK(epViewShape.GetDimNum() < 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "expanded_permuted_rows dim num must be >= 2, got %zu.",
                     epViewShape.GetDimNum()),
             return false);

    static bool CheckDropPadMode(int64_t dropPadMode)
    {
        OP_CHECK(dropPadMode < 0 || dropPadMode > 3,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "drop_pad_mode must be in [0, 3], got %ld.", dropPadMode),
                 return false);
        return true;
    }
```

```cpp
    // after：全部改为期望成立的守卫条件
    OP_CHECK(IsDtypeSupported(epDtype),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "expanded_permuted_rows dtype not supported: dtype=%d. Supported: FLOAT16, FLOAT, BFLOAT16.",
                     static_cast<int>(epDtype)),
             return false);

    OP_CHECK(epViewShape.GetDimNum() >= 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "expanded_permuted_rows dim num must be >= 2, got %zu.",
                     epViewShape.GetDimNum()),
             return false);

    static bool CheckDropPadMode(int64_t dropPadMode)
    {
        OP_CHECK(dropPadMode >= 0 && dropPadMode <= 3,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "drop_pad_mode must be in [0, 3], got %ld.", dropPadMode),
                 return false);
        return true;
    }
```

规则一句话：`OP_CHECK` 传"期望成立的条件"（守卫式），`OP_CHECK_IF` 传"错误条件"（触发式）；op_host 的 InferShape/Tiling 里用 `OP_CHECK_IF` 写错误条件是正确用法，两套宏不能混用直觉。

验证信号（复现增量，MoE FinalizeRouting ST 上板联调会话）：

- 修复前：4 个 L0 用例全部 `GetWorkspaceSize 失败: 161002`，plog 报 `dtype=1 not supported. Supported: FLOAT16...`（来源平台：950；证据等级：tool_observed）。
- 修复后：dtype 校验通过，首个 fp16 全可选用例 PASS（MERE=2.47e-08 / 524288 elems）；InferShape 层再修后 4 个 L0 用例全部 PASS（MERE 2.13e-08~4.13e-08），Golden 自测 9/9、UT 8/8（来源平台：950；证据等级：tool_observed）。
- 判别探针：日志追加 `IsSupported=%d` 打印复跑，观测 `IsSupported=1` 时错误仍触发，锁定宏极性（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 只修一处报错文件就收工——本根因在 L2 与 L0 两个文件成对出现，修完第一处后第二处以 561103 形式再次暴露；修复时整仓 grep `OP_CHECK` 逐条核对极性。
- Code review 检查项：`OP_CHECK` 第一参里出现 `!`、`<`、`>`、`!=`、`||` 组合的"错误条件"即可疑。

失效边界（复现增量）：

- `OP_CHECK_IF` 的正确用法（错误条件）在本会话 op_host 侧验证；其他专用宏（`OP_CHECK_DTYPE_NOT_SUPPORT` 等 `if (错误条件)` 模式）语义各自独立，勿由本卡推广。

## 增量（第三条独立 trace：RoPE 双 layout 算子，自相矛盾日志指纹与 CheckAttrs 形态）

独立复现 R3（RoPE 类双输出算子 apply_rotary_pos_emb，query/key + cos/sin，BSND/BNSD 两种 layout、half/interleaved 两种 rotaryMode，CANN 9.1.0，Ascend950PR）：编译、打包、安装全部正常，ST 15/15 用例全部在 GetWorkspaceSize 阶段返回 161002，CPU golden 自测 7/7 通过。op_api 侧 `CheckShape`/`CheckAttrs` 的所有 `OP_CHECK` 都把**错误条件**（而非期望条件）写进了第一个参数。

**判别指纹增量——报错原文与输入自相矛盾（无需额外埋点）**：plog 同一段落同时打印 `query: aclTensor(view_shape = [8, 191, 1, 4], ...)` 与 `query must be 4D, got 4 dims`——4D 输入触发"必须是 4D"的报错，即条件在输入满足要求时仍触发，可直接锁定"条件写反"而不是数据问题。这是 R2 增量"日志 got X 且 X 满足断言"指纹的维度计数形态，且不需要在测试里追加调试打印。

**新形态——CheckAttrs 属性区间条件反转**：除 shape 校验外，标量属性的区间检查同样会整体反转（`&&` 连接的"都不满足"错误条件 → `||` 连接的"任一满足"期望条件），before/after 逐字取自该会话修复：

```cpp
    // before：区间"错误条件"直接进 OP_CHECK（合法值全被拒）
    static bool CheckAttrs(int64_t layout, int64_t rotaryMode)
    {
        OP_CHECK(layout != 0 && layout != 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout must be 0 or 1, got %ld", layout),
                 return false);
        OP_CHECK(rotaryMode != 0 && rotaryMode != 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotaryMode must be 0 or 1, got %ld", rotaryMode),
                 return false);
        return true;
    }
```

```cpp
    // after：改为"期望为真"的断言条件（&& → ||，!= → ==）
    static bool CheckAttrs(int64_t layout, int64_t rotaryMode)
    {
        // 注意: OP_CHECK(cond, ...) 是断言风格，当 cond 为假时报错，故 cond 应为期望条件
        OP_CHECK(layout == 0 || layout == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout must be 0 or 1, got %ld", layout),
                 return false);
        OP_CHECK(rotaryMode == 0 || rotaryMode == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotaryMode must be 0 or 1, got %ld", rotaryMode),
                 return false);
        return true;
    }
```

CheckShape 侧同型反转的修法与主卡一致（`GetDimNum() != 4` → `== 4`、`cosDimNum != 2 && != 3` → `== 2 || == 3`、`D % 2 != 0` → `D % 2 == 0`、shape 相等性 `!=` → `==` 等，逐条翻转）。

**审计方式增量**：本开发会话 审计时，`CheckDtypeValid` 中本来就按断言语义正确书写的 `OP_CHECK(IsDtypeSupported(...), ...)` 未被改动——同类修复必须按"条件方向"逐条判定，不能对全文件 `!=` 一刀切批量替换，否则会把本来就正确的条件一并改坏。

验证信号（复现增量，RoPE 双输出算子首次 ST 真机联调会话）：

- 修复前：15/15 用例 `GetWorkspaceSize 失败: 161002`，CPU golden 自测 7/7 通过（来源平台：950；证据等级：tool_observed）。
- 修复前 plog：`[CheckShape] ... errno[161002] ... query must be 4D, got 4 dims`，同段日志打印 `query: aclTensor(view_shape = [8, 191, 1, 4], ...)`（来源平台：950；证据等级：tool_observed）。
- 修复后（重新编译 + `--force` 重装 + ST 复跑）：161002 全部消失，用例进入 kernel 执行阶段（后续失败转为 507035/精度比对失败，属 kernel 侧独立问题）（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 全文件无差别替换 `!=`/`==` 修这类问题——会把本来就正确的 `OP_CHECK(IsDtypeSupported(...))` 一并改坏；应逐条判定条件方向。

失效边界（复现增量）：

- `OP_CHECK_WRONG_DIMENSION`、`OP_CHECK_SHAPE_NOT_EQUAL` 等专用宏自带"不满足即报错"语义，直接按字面用即可，不存在本卡的反逻辑问题；本增量证据来自 CANN 9.1.0 头文件，跨版本使用前建议按主卡 grep 探针重新确认。

## 增量（五条独立 trace：极性陷阱形态谱系、UT 放松掩盖与错误码速查）

五条独立复现（R4–R8，Ascend950PR，CANN 9.1.0 与 9.2.0 均有命中），补齐极性陷阱的形态谱系与两条流程教训：

**R4（MLaProlog 类算子，CANN 9.2.0）——dtype 白名单极性 + 同批多检查反转**：`OP_CHECK(!IsDtypeSupported(token_x->GetDataType()), ...)` 让白名单含 DT_BF16（ACL_BF16=27）的合法 bf16 输入反报 `AclNN_Parameter_Error(EZ1001): token_x dtype not supported: 27`；同文件 `CheckNHeads`（合法 2 的幂触发 "must be power of 2"）与 `CheckEpsilon`（界内 eps 触发区间报错）同样反转——dtype 检查只是第一个被踩到的雷，只修第一处会逐轮暴露。修复后 UT 12/12 通过。判别要点：报错里的 dtype 数值（27）对照 `acl_base_rt.h` 枚举即可确认被拒的是合法值。

**R5（GRU 类算子，CANN 9.2.0）——极性反转的双重后果**：`x dim must be 3, got 3` 自相矛盾日志（合法 dimNum==3 时 `!= 3` 为假 → 触发报错），且校验对真正非法的输入反而全部放行，进程随后段错误——极性反转不只是"误拒合法"，还有"放行非法"的另一半；修复后 5 个 L0 用例进入 kernel 执行。

**R6（GQA 类算子，CANN 9.1.0）——format/shape/因果约束三类翻转点**：`CheckFormat` 的 `IsPrivateFormat(fmtQ) || ... || IsPrivateFormat(fmtY)`（全非私有格式=合法时条件为假 → 报 "Private format not supported"，输入为 ND）、`CheckShape` 的 `GetDimNum() != 4`（"query dim must be 4, got 4"）、`is_causal` 的 `S > S_kv`（合法约束 S<=S_kv 反被打回）。设计教训：统一改写为**正向不变式**（`== 4`、`!IsPrivateFormat(...) && ...`、`S <= S_kv`）而不是在错误条件外包 `!`——宏命名不总提示极性，正向不变式最不易错。误诊素材：曾怀疑 `IsPrivateFormat(ND)` 返回 true 或符号解析失败——`op::IsPrivateFormat` 定义在 libnnopbase.so（nm -D 实证），ND 输入本身没问题。

**R7（RoiAlign 类算子，CANN 9.1.0）——plog 挖掘方法与 UT 假阴性**：`ASCEND_GLOBAL_LOG_LEVEL=0/1` 都不会把 `OP_LOGE` 打到 stdout，真正的报错在 `~/ascend/log/debug/plog/plog-<pid>_*.log`（`x must be 4D, got dimNum=4`）；错误码速查（`op_errno.h` 实测）：`ACLNN_ERR_PARAM_NULLPTR=161001`、`ACLNN_ERR_PARAM_INVALID=161002`、`ACLNN_ERR_INNER_CREATE_EXECUTOR=561101`、`ACLNN_ERR_INNER_NULLPTR=561103`。遗留教训：UT 在条件反转状态下仍 35/35 通过（非法输入可能被后续检查兜住）——条件反转对 UT 负向用例不一定可见，CI 应加一条合法输入冒烟用例在 ST 前暴露此类问题。

**R8（MoeGatingTopKSoftmax 类算子，CANN 9.1.0）——同函数混用两种极性 + UT 放松掩盖**：同一 `CheckShape` 里前三个 `OP_CHECK`（dimNum/E/k 范围）写法正确、后 6 处（输出 shape `!=`、finished `!=`、`IsEmpty`）全部反写——"plog 里 y 形状与期望一致却报 y shape mismatch" 即铁证；混用是这类 bug 漏检的主因。流程教训：op_api UT 的 ValidParams 用例当时已捕获"合法参数返回 161002"，却被误判为"UT 环境无 NPU 的预期失败"而把 `EXPECT_NE(ret, 161002)` 放松成 `SUCCEED()`——合法参数返回参数校验错误永远值得怀疑，不应因"无 NPU 环境"合理化放松。

验证信号（复现增量，五条独立 ST 联调会话）：

- R4：修复前 6/6 用例 `GetWorkspaceSize 失败: 161002` + `EZ1001: token_x dtype not supported: 27`（27 与 ACL_BF16 枚举核对一致）；修复后 UT 12/12 通过、ST 不再出现 161002（来源平台：950；证据等级：tool_observed）。
- R5：修复前 `[CheckShape] errno[161002] ... x dim must be 3, got 3.` 后进程段错误；宏定义核对 `op_log.h:227` 展开为 `if (unlikely(!(cond)))`；修复后校验通过、golden 自测通过、5 个 L0 用例进入 kernel 执行（来源平台：950；证据等级：tool_observed）。
- R6：修复前 6/6 `GetWorkspaceSize` 返回 161002（CheckFormat "Private format not supported" → 修后第二轮 CheckShape "query dim must be 4, got 4"）；三处条件全部改为正确状态并重建重装后 GetWorkspaceSize 全部通过（来源平台：950；证据等级：tool_observed）。
- R7：修复前 8/8 用例 `[FAIL] GetWorkspaceSize 失败: ret=161002`、plog `[CheckShapeValid] x must be 4D, got dimNum=4`；修复后（重建+重装+重跑 ST Real）GetWorkspaceSize 全部通过、UT 回归 35/35（来源平台：950；证据等级：tool_observed）。
- R8：修复前 ST 45/45 全部 `GetWorkspaceSize 失败: 161002`、plog `[CheckShape] y shape mismatch` 而同帧日志 y view_shape 与期望一致；修复后重建重装重跑 161002 全部消失、用例进入 kernel 输出比对阶段（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 同一函数内混用两种极性写法——review 时对每个 `!=`/`IsEmpty` 形态的条件逐个质疑极性；"must be X, got X" 式自相矛盾日志即指纹。
- 把 UT 里"合法参数返回 161002"的失败用 `SUCCEED()` 或放宽容差合理化——属测试侧放松，会把极性 bug 拖到 ST 才暴露。
- 只修第一个被踩到的检查（dtype）不全文排查同类反转（n_heads/epsilon/format/causal 同批皆反）。
- 只看 stdout 判定"没有报错信息"——`OP_LOGE` 走 plog（`~/ascend/log/debug/plog/`），不在 stdout/stderr。

失效边界（复现增量）：

- 五条 trace 覆盖 CANN 9.1.0 与 9.2.0 的 `op_log.h` 宏形态，语义一致（均为 `if (unlikely(!(cond)))` 触发式）；其他版本升级时建议重读一次宏展开。
- UT 在反转状态下仍全绿（R7）的机制（非法输入被后续检查兜住或断言宽松）未验证到机制层；"CI 加合法输入冒烟"为建议项，未落地。

## 增量（五条独立 trace 复现：dtype/shape/区间/标量与 L0 成对再暴露）

并行策展批五条独立 trace（分组矩阵乘 grouped_matmul、融合 dynamic_quant、Mish 类激活、foreach 范数、GroupNorm，均为 Ascend950PR、CANN 9.1.0、ST/UT 实测）再次复现同一根因，补齐新形态与判别增量：

**形态增量 1——出参侧 GetOutputCount 全军覆没**（分组矩阵乘）：ST 全部用例在 `aclnnBenchNoHarnessGroupedMatmulGetOutputCount` 阶段报 161002，plog 打出 `weight must be 3D, got dimNum=3`（dimNum=3 就是 3D，报错值即合法值）；同文件 14 处 `OP_CHECK` 全部反向（`!= 3`、`< 0 || > 3`、`!= E`、`K != wK`、`(*group_list)[i] < (*group_list)[i-1]` 等），全量翻转后 161002 消失、推进到下一独立错误。修复时全文件一次性修完，不能只修报错那条——其余 13 处同样反着，只修一处会留下"非法输入被放行"的更隐蔽问题。

**形态增量 2——DFX 入参打印对照 + 枚举值探针**（融合 dynamic_quant，同根因诊断卡）：`[CheckDtypeValid] errno[161002] y dtype must be INT8, got 2.` 而同一运行 DFX 打印 `y: aclTensor(... data_type = DT_INT8 ...)`——"got 2"与"must be INT8"自相矛盾（ACL_INT8 = DT_INT8 = 2）。用独立小程序编译期打印枚举值确认 `ACL_FLOAT = 0` 且 `ge::DT_FLOAT = 0`（相等），排除"两套 DataType 枚举值不一致"假设后锁定宏极性（注意 `ACL_DT_UNDEFINED = -1` 与 `ge::DT_UNDEFINED = 28` 确实不同，但 DT_FLOAT 恰好同为 0）。

**形态增量 3——UT 双向用例暴露（放行/拦截颠倒）**（Mish 类激活）：op_api UT 同时覆盖"应拒绝"与"应放行"两个方向时本 bug 现形——ShapeMismatch 用例（x={2,3}, y={2,4}）未得 161002 反被放行到执行路径；EmptyTensor 用例（shape={0}）本应 ACLNN_SUCCESS 却被 161002 拦截。条件翻转后双向均恢复。教训：op_api 校验 UT 必须双向覆盖，单向用例测不出条件反转。

**形态增量 4——L0 与 L2 成对出现、修完一层下一层再暴露**（foreach 范数）：L2 `CheckScalar` 报 `scalar dtype 0 != DT_FLOAT`（DFX 刚打印 `aclScalar(data_type = DT_FLOAT, value = 2.000000)`）修完后，L0 op 的 `OP_CHECK(x == nullptr, ...)` 让非空输入报 "x is null"——同一批约 15 处在 L2/L0 两层成对写反，修复须整仓 grep `OP_CHECK` 逐条核对极性（与本卡 复现增量的 InferShape/L0 成对形态互证）。修后须重编译重装算子包再跑，否则运行时仍加载旧库。

**形态增量 5——区间检查的维度数形态**（GroupNorm）：`x dimNum=2 not in [2,5]`（dimNum=2 合法）；同函数 gamma/beta 一维、C 整除 numGroups、shape 相等等本来满足的约束全部报错——"所有合法输入被无差别拒绝"是本根因的系统性指纹。

修复配方与主卡一致（错误条件 → 期望条件），补充可复用修法：`OP_CHECK_DTYPE_NOT_MATCH(y, DataType::DT_INT8, return false);` 这类自带"不匹配即报错"语义的专用宏可直接替代手写 dtype 等值断言，少踩一次极性坑。

验证信号（复现增量，五条 trace 的 ST/UT 实测）：

- 分组矩阵乘：修复前全部用例 `[FAIL] GetOutputCount 失败: 161002`，plog `weight must be 3D, got dimNum=3`；14 处翻转重编重装后 161002 全部消失（来源平台：950；证据等级：tool_observed）。
- 融合 dynamic_quant：plog `y dtype must be INT8, got 2` + DFX `data_type = DT_INT8` 同运行对照；`acl_base_rt.h` 枚举核对 ACL_INT8=2、ACL_FLOAT=0 与 ge::DataType 一致（来源平台：950；证据等级：tool_observed）。
- Mish 激活：修复前 ShapeMismatch 用例被放行、EmptyTensor 用例被 161002 拦截；`!=`→`==` 后 op_api UT 9/9、全量 UT 21/21、ST 上板 12/12（来源平台：950；证据等级：tool_observed）。
- foreach 范数：修复前首用例段错误、plog `scalar dtype 0 != DT_FLOAT` 与 L0 `x is null`（输入非空）先后暴露；L2/L0 约 15 处翻转后推进到执行阶段（来源平台：950；证据等级：tool_observed）。
- GroupNorm：修复前 41/41 用例 161002、`x dimNum=2 not in [2,5]`；区间条件翻转后 dimNum 类报错消失、推进到 epsilon 校验（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 只修第一处报错就收工——本根因在同一文件/同一批文件成片出现（14 处、15 处），报错的那条只是最先被合法输入撞上的。
- 修完不重编译重装算子包直接复跑——运行时仍加载旧库，误判"修复无效"。

失效边界（复现增量）：

- `OP_CHECK_NULL(param, ret)`、`OP_CHECK_DTYPE_NOT_MATCH`、`OP_CHECK_DTYPE_NOT_SUPPORT` 等专用宏自带"坏值即报错"语义且用法常见正确，勿随本卡一起翻转（五条 trace 均确认）。
- 与 op_host 层 `OP_CHECK_IF`（错误条件语义，正确用法）的跨层极性差异见同域 [op_host OP_CHECK_IF 宏名卡](op_host_op_check_if_macro_name.md)。

## 后续增量（2026-09-17，并卡合并：宏定义原文、sigmoid 首例与 gather 四处同源反转）

#### 并入自 op_check_cond_inversion_161002（宏定义原文、CHECK_COND 等价性与 sigmoid 首个闭环）

**宏定义原文（`include/aclnn/opdev/op_log.h:227`，CANN 9.1.0）**——`OP_CHECK` 是断言语义，条件为 false 时才执行报错和返回：

```c
#define OP_CHECK(cond, log_func, return_expr) \
    do {                                      \
        if (unlikely(!(cond))) {              \
            log_func;                         \
            return_expr;                      \
        }                                     \
    } while (false)
```

同族 `CHECK_COND` 同样是 `!(cond)` 触发：`CHECK_COND(x == y, ACLNN_ERR_PARAM_INVALID, "...")` 与 `OP_CHECK(x == y, ...)` 等价，修复时可保留项目原有 `OP_CHECK + OP_LOGE + return false` 风格，改动最小。

**sigmoid 首例（BenchNoHarnessSigmoid，Ascend950PR + CANN 9.1.0）**：UT 25/25 通过、CPU Golden 自测 18/18 全过后，NPU Real 模式 20/20 全部用例 `GetWorkspaceSize 失败: 161002`；plog（`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1` 落文件后 grep）定位 `[aclnn_bench_no_harness_sigmoid.cpp:88][CheckShape] errno[161002] Shape mismatch: x_shape vs y_shape. Broadcasting not supported.`。调试打印逐维对比 x/y 的 `GetViewShape()`（x_dim=2, y_dim=2, x[0]=1, y[0]=1, x[1]=5, y[1]=5）证明两侧完全相同，排除真实 shape 不一致，锁定宏语义反转；`!=` 改 `==` 后 20/20 全过（MERE/MARE 全部远低于 fp32 阈值 1.22e-04，如 1,738,100 元素用例 MERE=0.00e+00）。

判别方法补充（并入）：

- 在 ST 里打印 `aclCreateTensor` 前后的 shape 与 x/y 的 view shape 逐维值，确认报"mismatch"的张量实际 shape 一致（比读宏定义先缩小范围）。
- 逐个核对同文件其它 `OP_CHECK` 用法：`OP_CHECK(IsDtypeSupported(...), ...)`、`OP_CHECK(!(IsPrivateFormat(x) || IsPrivateFormat(y)), ...)`、L0 层 `OP_CHECK(npuArch == NpuArch::DAV_3510, ...)` 这类"期望条件"写法是正确的，只有把 `!=` 当断言条件的 `CheckShape` 反了——同文件混用时逐条判定，不能一刀切批量替换。

误诊补充（并入）：

- 起初怀疑 ST 框架建 tensor 时 format/strides 设置不当导致 shape 检查失败，以及 L0 层 `IsAiCoreSupport`（`npuArch == DAV_3510`）平台检查不通过；打印 x/y 实际 view shape 一致后两者均被排除。

#### 并入自 op_check_cond_inversion_161002（并行策展批：gather 算子四处同源反转 + 红旗口诀）

另一独立轨迹并入 gather 算子（Ascend950PR/DAV_3510，CANN 9.1.0，另一 trace）的同根因复现：`CheckShape` 里 **4 个条件全部**按"错误条件"书写（x/index 维度数不等、dim 越界、y 与 index shape 不等、index 各维超 x 对应维），UT 17/17 通过但 ST Real 10/10 全部 `[FAIL] GetWorkspaceSize 失败: 161002`。修复（4 处全部反转为合法状态断言，verbatim 对照）：

```cpp
    // x 与 index 维度数必须相同
    OP_CHECK(x->GetViewShape().GetDimNum() == index->GetViewShape().GetDimNum(),   // 修复前为 !=
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "x and index must have the same number of dimensions: x_ndim=%zu, index_ndim=%zu",
                     x->GetViewShape().GetDimNum(), index->GetViewShape().GetDimNum()),
             return false);

    // dim 范围检查
    int64_t ndim = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    OP_CHECK(dim >= 0 && dim < ndim,                                              // 修复前为 dim < 0 || dim >= ndim
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "dim out of range: dim=%ld, ndim=%ld", dim, ndim),
             return false);

    // y shape 必须与 index shape 一致
    OP_CHECK(y->GetViewShape() == index->GetViewShape(),                          // 修复前为 !=
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape must match index shape"),
             return false);

    // 除 dim 维外，index 各维 size 不大于 x 对应维 size
    for (size_t i = 0; i < x->GetViewShape().GetDimNum(); i++) {
        if (static_cast<int64_t>(i) == dim) { continue; }
        OP_CHECK(index->GetViewShape().GetDim(i) <= x->GetViewShape().GetDim(i),  // 修复前为 >
                  OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                          "index shape[%zu]=%ld exceeds x shape[%zu]=%ld (dim != %ld)",
                          i, index->GetViewShape().GetDim(i), i, x->GetViewShape().GetDim(i), dim),
                  return false);
    }
```

**新判别信号（"合法值出现在报错里"）**：plog 报 `x and index must have the same number of dimensions: x_ndim=3, index_ndim=3`——两个维度数**相等**却触发"必须相同"的报错，连 1D 用例（1,1）也失败，即所有合法输入都在触发报错、校验条件方向写反。

**红旗口诀（自查，并入）**：`!=`/`||`/`>` 出现在 OP_CHECK 条件里时几乎必然写反，应为 `==`/`&&`/`<=`；一次修复必须整文件核对全部同源条件（本实例 4 处同源，只修一处会留下下一轮 161002）。

验证信号（另一独立轨迹并入，均为 tool_observed）：

- 修复前：ST Real 10/10 全部 `[FAIL] GetWorkspaceSize 失败: 161002`；plog `CheckShape` 在 `x_ndim=3, index_ndim=3`（合法）上报 161002（来源平台：950；证据等级：tool_observed）。
- 修复后（4 条件反转 + 重编重装）：161002 全部消失，错误推进为 507035（独立根因，另卡）；最终迭代一 ST 10/10、迭代二 ST 25/25 全部 bitwise 通过（来源平台：950；证据等级：tool_observed）。
- sigmoid 首例信号：修复前 20/20 用例 161002、plog `[CheckShape] errno[161002] Shape mismatch`；x/y view shape 逐维打印完全一致（如 shape=1,5 两维均相同）；修复后重编 + `--force` 重装 + ST Real 重跑 20/20 全过（MERE=0.00e+00 量级）（来源平台：950；证据等级：tool_observed）。