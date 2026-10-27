---
type: "Runbook"
title: "expect_error 用例被判 FAIL：host 拒收非缺陷"
description: "TTK kernel 模式无预期失败语义：expect_error 负向用例被 host tiling 正确拒收（OPTILING_FAILURE）仍记 precision FAIL；判框架限制登记豁免，勿改 kernel/用例凑绿。R2（aclnn 黑盒模式）：TTK aclnn 黑盒的 *_exception 负向用例被 GetWorkspaceSize 正确拒绝（ACLNN_ERR_PARAM_INVALID）仍记 GOLD: NO_OUTPUT / PRECISION_STATUS: FAIL 或元素数失配——同一框架缺口在黑盒路径的形态；黑盒 CSV 无 expected_error 列，按用例名后缀分桶并逐例归档拒绝证据。R3（白盒 min-shape 用例）：正确拒绝条数多于 expect_error 标记数时，多出条目可能是生成侧 min-shape special 路径未套用 dim_even 约束、默认 [1,1,1,1] 对偶数维算子非法且未标 expect_error——生成侧显式设 min_shapes（如 [[1,1,1,2]]）为一劳永逸解。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# expect_error 用例被判 FAIL：host 拒收非缺陷

## 现象

在对一个对标 torch `aten::_thnn_fused_gru_cell_backward` 的 GRU Cell 反向融合算子（双输入 `grad_hy`/`storage`，AscendC 算子仓开发流程，vendor 部署）执行 Branch-1（group kernel 分支，系统测试已 11/11 通过）的白盒验证时，用 TTK kernel 模式对本分支成品 kernel 全量跑白盒用例表（30 条用例；超大 shape 旁表 26 条按约定排除不执行）。release 精度套件跑完 30 条里 29 条精度 PASS，唯一失败的是一条 CSV `remark=expect_error;group=` 的负向探针用例：它由白盒用例生成侧对双输入算子**自动注入**（incompat-broadcast 探针），故意把 `storage` 输入给成 `(2, 4)`、`grad_hy` 给成 `(2, 3)`——不满足算子契约要求的 `storage.shape == (B, 5H)`（`H = grad_hy.shape[1] = 3`，合法 storage 应为 `(2, 15)`；`4 ≠ 15` 即违约）——用于探测算子对非法广播输入是否正确拒收。

TTK 执行日志（full_run.log）中该用例的原始证据链：

```
2026-09-07 12:12:08,071 [ERROR] [2499225 case00038 OnBinOpTiling] [dynamic_compilation.py:258]: OPTILING_FAILURE
RuntimeError: OPTILING_FAILURE:
2026-09-07 12:12:08,194 [ERROR] [2499225 case00038 OnGenGolden] [output_generation.py:296]: Golden generation failure: shape_mismatch: grad_hy (2, 3) vs storage (2, 4)（须满足 storage.shape == (B, 5H)）
ValueError: shape_mismatch: grad_hy (2, 3) vs storage (2, 4)（须满足 storage.shape == (B, 5H)）
Tiling Data Parsed Dict: ('OPTILING_FAILURE', False)
BIN_PERF: INVALID_TILING           BIN_GOLD: INVALID_TILING           BIN_MEM: UNKNOWN
```

result.csv 中该行 `precision_status=FAIL`、`bin_precision=INVALID_TILING`。随后单独执行的动态 OOM 插桩套件（`op_debug_config=oom`）里，同一用例 `dyn_oob_result=UNKNOWN`、`memory_oob_status=PASS`，结果解析器判为 MISSING_OOB_EVIDENCE（无越界证据），门禁两套均 FAIL——但此时 kernel 本身没有任何精度或内存缺陷。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发中的白盒评测——TTK kernel 模式（`ttk kernel -i <用例CSV> --plugin <golden.py> -o <result.csv>`）对本分支成品 kernel 全量执行。
- API 路径：TTK `kernel` 子命令；用例 CSV 的 `remark` 列（`expect_error` 标记）；host 侧 `CheckTiling`/tiling 校验（`OnBinOpTiling` 阶段）；golden TestSpec（`OnGenGolden` 阶段）；仓库内 `tests/whitebox/ttk/known_blockers.md` 框架级阻断登记机制。
- 验证过的平台与 CANN 版本：Ascend950PR（TTK Device Platform: Ascend950PR_9579）+ CANN 9.0.0-beta.2（TTK 检出于 `/tmp/ops-test-kit`）。与 frontmatter `platforms`（950）一致。
- 适用粒度（transfer_scope）：`operator_family`——判读规则绑定「TTK 不识别 expect_error 标记」这一框架行为与「host 校验层拒收非法输入」的算子设计约定，对任何带负向用例、host 侧做 shape 校验的自定义算子成立，不绑定本算子的具体公式。
- 路线优先级：`recommended_fastpath`——负向用例判读走"对照登记豁免"路线，不改 kernel、不改用例、不改 golden。
- 明确不适用 / 未验证场景：**不带** `expect_error` 标记的用例出现 `OPTILING_FAILURE`/`INVALID_TILING`（那是真 tiling 缺陷或必选属性缺失，须另行定位）；aclnn/ST 等非 TTK kernel 模式执行路径；其他 NPU 平台与其他 CANN/TTK 版本未验证（推断可迁移，见失效边界）。

## 判别方法

对 TTK kernel 模式结果中的 `precision_status=FAIL` 用例，按序核查：

1. 查用例 CSV 的 `remark` 列是否含 `expect_error`——这是负向用例（预期算子拒收）的语义标记；
2. `grep` 执行日志确认失败发生在 **tiling 阶段**（`OnBinOpTiling ... OPTILING_FAILURE`）而非 kernel 执行阶段，且 `result.csv` 该行 `bin/dyn_precision=INVALID_TILING`；
3. 确认 golden 侧**同步**显式报 shape 校验错（`OnGenGolden ... Golden generation failure: shape_mismatch: ...`）——host 拒收与 golden 拒收两路独立信号互为旁证，说明这是"输入非法"而非"实现算错"；
4. 若仓库存在 `known_blockers.md`，对照已登记的失败模式（本例为登记条目 KB-1：同用例、同 shape、同 OPTILING_FAILURE + shape_mismatch 失败模式），一致则在汇报中注明"与已登记根因一致"并按登记豁免放行；
5. OOM 插桩套件中该用例表现为 `dyn_oob_result=UNKNOWN` + `memory_oob_status=PASS`（tiling 拒收 → kernel 未执行 → 无可插桩对象），这是同一根因在内存套件的推论表现，不是内存安全问题。
6. R2 扩展（TTK aclnn 黑盒模式，`*_exception` 用例）：对黑盒 result.csv 中 `NO_OUTPUT`/元素数不匹配且用例名含 `exception` 的 FAIL——取该用例在 full_run.log 的上下文（`grep -n "<case_name>" full_run.log` 后读对应行段），确认日志中 aclnn 报错为 `ACLNN_ERR_PARAM_INVALID`（来自 `GetWorkspaceSize` 参数校验路径，如 `RuntimeError: Acl API Call aclnn<Op>GetWorkspaceSize() Failed: ACLNN_ERR_PARAM_INVALID`）且紧随 `GOLD: NO_OUTPUT / PRECISION_STATUS: FAIL`；对照设计约束矩阵（dtype 白名单、rank 上限、axes 1-D 与界内等）确认输入确属非法 → 算子拒绝是正确行为，FAIL 判定是框架缺乏 expected-error 语义所致。反向排除：若日志无 ACLNN_ERR_PARAM_INVALID 而是段错误/超时/内核崩溃，则是真缺陷，回 kernel 排查。注意黑盒 CSV（本版本 12 列）无 expected_error/remark 列，负向语义只能靠用例名后缀识别。

命中 1+2+3 即可判定：该 FAIL 是框架级假阴性（TTK 无"预期失败"语义），不是 kernel 缺陷。

## 根因

TTK kernel 模式**不识别用例 CSV 的 `expect_error` 标记**（评测框架没有"预期失败"语义，不区分"算子算错"与"算子按设计拒收非法输入"），而这类负向用例的设计意图恰恰是让 host tiling 在校验层拒绝非法 shape——`OPTILING_FAILURE` 拒收本身就是被测语义的正确行为；golden 侧 TestSpec 对同一非法输入按 shape 契约显式 `raise ValueError`，两路独立拒收共同证明"输入非法"而非"实现算错"。机制链：`storage.shape[1]` 必须 `== 5 × grad_hy.shape[1]`（行内 [r,z,n,hx,hn] 五平面布局），探针输入 `4 ≠ 5×3=15` → host tiling 校验拒绝（`OPTILING_FAILURE`）→ kernel 不执行 → TTK 无 golden 可比（golden 同步报 `shape_mismatch`）→ 结果行记 `INVALID_TILING` → 门禁 `precision_status=FAIL`；OOM 套件因 kernel 未执行 → 无插桩对象 → `dyn_oob_result=UNKNOWN` → MISSING_OOB_EVIDENCE。

## 设计决策

- **golden 对非法输入显式报错，而非静默截断**：TestSpec 的 `golden()` 在计算前显式校验 shape 契约并 `raise ValueError`（注释原话："非法输入让 golden 显式报错，而非靠 torch 切片静默截断 / 广播出错值"）。若 golden 静默截断/广播，非法输入会产生一个"看似合法"的期望值，负向用例的判读信号就消失了。备选方案（依赖 torch 切片自然报错）被放弃：报错信息不可控且语义含糊。
- **负向用例在 host 校验层拒绝、不进任何 kernel 分支**：这是该算子分支路由设计的既定约定（设计文档与 TilingUT 负向用例断言、黑盒 L2 非法输入用例共同旁证；前序登记文件 known_blockers.md 引述）。因此"改 host 掾收非法输入让用例变绿"会直接违反设计，被明确禁止。
- **处置走登记豁免而非修复**：本卡路线为 `recommended_fastpath` 的判读路线——在 known_blockers.md 登记失败模式与证据后按逃逸条款放行（仅精度/框架类失败适用；**内存越界不得作为已知阻断放行**），并要求 TestSpec 侧用 `oob_exemptions` 声明豁免语义（`expect_error → tiling_rejection_no_kernel_execution`），使豁免可审计而非口头放行。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`，`ASCEND_OPP_PATH`/`ASCEND_TOOLKIT_HOME` 由 `set_env.sh` 提供）；TTK 检出于 `/tmp/ops-test-kit`（`PY=python3`；执行时 `PYTHONPATH=$TTK:<golden目录>:$ASCEND_TOOLKIT_HOME/python/site-packages`，`LD_LIBRARY_PATH` 前置 `vendors/<OpType>/op_api/lib`，CWD 须为 TTK 根）。
- 目标平台：Ascend950PR（npu-smi 25.7.rc1.6 实测在位；TTK 日志 Device Platform: Ascend950PR_9579）。
- 目录布局：算子包 `operators/<Op>_package/`（白盒用例 `tests/whitebox/branch-1/ttk_design_doc_cases.csv`、超大旁表 `..._oversized.csv`、结果 `tests/whitebox/ttk/result.csv`/`result_oom.csv`、登记 `tests/whitebox/ttk/known_blockers.md`）；文档侧 `docs/<Op>/develop/`（`env.sh`、`golden.py`、`LOG.md`）。
- 相关文件：`docs/<Op>/develop/golden.py`（TestSpec：shape 校验 + `oob_exemptions` 声明）、`tests/whitebox/branch-1/ttk_design_doc_cases.csv`（remark 列）、`tests/whitebox/ttk/known_blockers.md`（框架级阻断登记）、TTK 执行日志 `full_run.log`/`oom_debug.log`。

## 完整代码

golden TestSpec 中与该判读直接相关的实现（`docs/<Op>/develop/golden.py`，`golden()` 函数 verbatim——shape 契约校验位于计算前）：

```python
    def golden(grad_hy, storage, has_bias=False, *pure_outputs, **kwargs):
        """五路梯度融合反向，返回 [grad_input_gates, grad_hidden_gates, grad_hx,
        grad_input_bias, grad_hidden_bias]（顺序与 spec outputs 一致）。

        **kwargs 接收框架元信息（tensor_dtypes / shapes / soc_version 等）。
        """
        go32, out_dtype, torch_out = _to_fp32_torch(grad_hy)
        st32, _, _ = _to_fp32_torch(storage)

        # 形状合法性（spec shape_constraints；非法输入让 golden 显式报错，
        # 而非靠 torch 切片静默截断 / 广播出错值）
        if go32.dim() != 2 or st32.dim() != 2:
            raise ValueError(
                f"ThnnFusedGruCellGrad 仅支持 2D 输入，got rank "
                f"{go32.dim()}/{st32.dim()}"
            )
        if st32.shape[0] != go32.shape[0] or st32.shape[1] != 5 * go32.shape[1]:
            raise ValueError(
                f"shape_mismatch: grad_hy {tuple(go32.shape)} vs storage "
                f"{tuple(st32.shape)}（须满足 storage.shape == (B, 5H)）"
            )
        has_bias = _as_bool(has_bias)

        hsz = st32.shape[1] // 5
        go = go32[:, 0:hsz]
        r = st32[:, 0 * hsz : 1 * hsz]
        z = st32[:, 1 * hsz : 2 * hsz]
        n = st32[:, 2 * hsz : 3 * hsz]
        hx = st32[:, 3 * hsz : 4 * hsz]
        hn = st32[:, 4 * hsz : 5 * hsz]

        # 五链乘序逐字照抄 spec formula / CUDA kernel（左结合，尾序 ·(1-z)·z / ·(1-r)·r）
        gin = go * (1 - z) * (1 - n * n)
        gig = go * (hx - n) * (1 - z) * z
        grg = gin * hn * (1 - r) * r
        ghn = gin * r
        grad_hx32 = go * z

        input_gates32 = torch.cat((grg, gig, gin), dim=1)
        hidden_gates32 = torch.cat((grg, gig, ghn), dim=1)

        if has_bias:
            input_bias32 = input_gates32.sum(dim=0)
            hidden_bias32 = hidden_gates32.sum(dim=0)
        else:
            input_bias32 = torch.zeros(0, dtype=torch.float32)
            hidden_bias32 = torch.zeros(0, dtype=torch.float32)

        return [
            _cast_back(input_gates32, out_dtype, torch_out),
            _cast_back(hidden_gates32, out_dtype, torch_out),
            _cast_back(grad_hx32, out_dtype, torch_out),
            _cast_back(input_bias32, out_dtype, torch_out),
            _cast_back(hidden_bias32, out_dtype, torch_out),
        ]
```

负向用例的 OOB 豁免声明（同文件 TestSpec 类属性，使豁免语义可审计）：

```python
    # OOB 豁免声明（Validation.md 负向用例处置）：
    # CSV remark=expect_error 的负向用例（非法 shape，如 case00038 storage!=(B,5H)）
    # 在 host tiling 阶段即被拒收（OPTILING_FAILURE），kernel 未执行 → 无可插桩对象，
    # dyn_oob_result=UNKNOWN 属预期（非内存安全问题）。此类用例的 OOB 要求标 N/A
    # （tiling 拒绝 = 正确拒收，不要求 kernel 级 OOB 证据；memory_oob_status=PASS 仍成立）。
    oob_exemptions = {
        "expect_error": "tiling_rejection_no_kernel_execution",
    }
```

## 性能影响

无性能故事（本卡为评测判读知识，不涉及两条实现路线的取舍）。

## 预防

- 用例生成侧：对双输入算子自动注入的 incompat-broadcast 探针**必须**带 `remark=expect_error` 标记，否则 TTK 结果无法与真缺陷区分。
- golden 编写侧：TestSpec 必须在计算前对 shape 契约显式校验并报错（如上 `golden()`），不要依赖 torch 切片静默截断——显式报错是负向用例判读的 golden 侧证据。
- 执行侧：建立 `known_blockers.md` 登记-对照机制——框架级根因登记失败模式与证据后，后续分支命中同模式即可短路判读；汇报注明"与已登记根因一致"。
- 判读红线：豁免仅适用精度/框架类失败；**内存越界不得作为已知阻断放行**。
- R2 扩展（黑盒门禁统计）：把 `*_exception` 用例单独分桶——`NO_OUTPUT` + 用例名含 exception + 日志含 ACLNN_ERR_PARAM_INVALID 的组合不计入 kernel 缺陷率，避免负向用例污染"全量通过率"结论；豁免须逐例归档 ACLNN_ERR_PARAM_INVALID 日志证据，不留证据直接标"已知问题"放行属测试侧放松。需要把负向用例计入自动断言时，走自写等价 aclnn 两段式验证程序路线（见 [TTK无负向支持：aclnn两段式负向验证卡](../validations/aclnn_negative_two_stage_verify_padded_out.md)）。

## 验证信号

来源说明：以上判读在一次 AscendC 算子仓白盒执行会话中被完整实测——对该 GRU Cell 反向融合算子的 Branch-1 成品 kernel，用 TTK kernel 模式跑完 release 精度与动态 OOM 插桩两套全量（各 30 条用例，与用例表逐条对账无缺漏），下列信号全部取自本次会话的命令输出与结果文件：

- release 精度套件 30 条：29 PASS / 1 FAIL / 0 SKIP，唯一 FAIL 即 expect_error 探针用例，`bin_precision=INVALID_TILING`、tiling 阶段 `OPTILING_FAILURE`，golden 侧同步报 `shape_mismatch: grad_hy (2, 3) vs storage (2, 4)（须满足 storage.shape == (B, 5H)）`（来源平台：Ascend950PR；证据等级：tool_observed）。
- 动态 OOM 插桩套件 30 条：29 条 OOB PASS 且 `dyn_oob_result` 证据全为 OK/PASS（无内存越界），0 条 OOB FAIL，唯一 invalid 为同一探针用例（`dyn_oob_result=UNKNOWN`、`memory_oob_status=PASS`，tiling 拒收 → kernel 未执行 → 无可插桩对象）（来源平台：Ascend950PR；证据等级：tool_observed）。
- 两套结果文件用例集合与输入用例表精确一致（各 30 条、集合相等、无重复），确认失败信号非子集执行或行序混淆所致（来源平台：Ascend950PR；证据等级：tool_observed）。
- 除该探针外全部 29 条可执行用例（含素数 shape、大 B、bias 有/无等组合）精度逐输出 PASS，佐证 host 校验层拒收未影响任何合法输入路径（来源平台：Ascend950PR；证据等级：tool_observed）。

- 独立复现 R2（euclidean_norm 黑盒全量 636 例中的 6 个 L2 `*_exception` 用例，TTK aclnn 黑盒模式，Ascend950PR + CANN 9.1.0）：`float16_ND_ND_ND_exception_shape_dim_error`（axes 传 3-D shape）日志显示 `aclnnEuclideanNormGetWorkspaceSize() Failed: ACLNN_ERR_PARAM_INVALID` + `GOLD: NO_OUTPUT` + `PRECISION_STATUS: FAIL`——算子正确拒绝了非法输入（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：6 例清单与非法性核对——float64/int64 x dtype 未注册（OpDef 仅 {fp16,bf16,fp32,int32}）、rank=9 > 上限 8、NCHW axes 越界 ×2、axes 3-D；dtype 类拒绝记 NO_OUTPUT，golden 与 kernel 输出元素数不一致的记 `6 vs 1`、`6 vs 24500`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：修复后权威全量重跑 636 例中该 6 例仍 FAIL（`NO_OUTPUT` / `6 vs 1` / `6 vs 24500`），与"框架缺口、非 kernel 缺陷"的归档结论一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判读前提是用例确实带 `expect_error` 语义且算子契约要求 host 校验层拒收：若**合法 shape** 用例出现 `OPTILING_FAILURE`/`INVALID_TILING`，本卡不适用，须按真 tiling 缺陷（或 CSV 缺必选属性显式值）排查。
- 本卡实测仅在 Ascend950PR + CANN 9.0.0-beta.2 + 本次 TTK 检出上完成；其他 NPU 平台、其他 CANN/TTK 版本未验证，推断可迁移（"TTK 不识别 expect_error"为框架侧行为，与本算子/平台无关的推断，未逐一实测）。
- R2 扩展：上文"不适用 aclnn/ST 等非 TTK kernel 模式执行路径"的限定针对 kernel 模式判读法（OPTILING_FAILURE/INVALID_TILING 签名）；TTK aclnn 黑盒模式的对偶形态（ACLNN_ERR_PARAM_INVALID + NO_OUTPUT/元素数失配）已由 R2 实测覆盖（CANN 9.1.0）。R2 判别只覆盖"非法输入被 GetWorkspaceSize 参数校验拒绝"的形态；非法输入走到执行阶段才失败（段错误/设备错误）不适用，须按真缺陷排查。`6 vs 1`/`6 vs 24500` 型失配的 golden 侧语义（golden 对非法 rank/format 输入仍产出不同 shape 的结果）未逐例深挖——只要输入非法成立，框架侧归因即成立。TTK 后续版本若增加 expected-error 字段（本版本黑盒 CSV 12 列中无此列），本卡判读规则需更新。
- `dyn_oob_result=UNKNOWN` 的成因不止一种：本卡的 UNKNOWN 特指"tiling 拒收 → kernel 未执行"；release 套件中 UNKNOWN 的另一成因是二进制未做 OOM 插桩（须跑动态插桩套件取得证据），两种成因不可混同判读。

## 坏实践

- **改 host tiling 放行非法 shape** 让负向用例"变绿"——直接违反"负向输入在 host 校验层拒绝"的设计要求，等于让算子接受非法广播输入。
- **删减/改写负向用例或改 golden 期望**凑绿——负向探针是设计要求的验收语义，删用例属测试侧放松。
- 把该 FAIL 记成 kernel 精度缺陷并进修复循环——浪费修复轮次且会把正确行为改错。
- 把 `dyn_oob_result=UNKNOWN` 当 PASS 上报——UNKNOWN 是"无证据"，不是"无越界"；OOM 证据必须由动态插桩套件的 OK/PASS 证据支撑。

## 误诊

- 常见误诊方向：把 `OPTILING_FAILURE` 直接当 tiling 实现缺陷排查（真缺陷与正确拒收在结果列上同为 `INVALID_TILING`，必须回看日志阶段与 golden 侧同步报错才能区分）；把 golden 报 `shape_mismatch` 当 golden 自身的 bug 去"修 golden"。
- 本会话中该用例的判读未被误诊（正确对照登记豁免）；同会话另有一次无关的试跑失败曾被归因为"transient 管道问题"，实际是命令中路径变量笔误，与本卡不同根因，不构成对本卡结论的影响。

## 未决事项

- TTK 框架后续版本是否会原生支持 `expect_error`/预期失败语义未知（本登记机制是临时绕行）；若框架支持，本卡判读路线应退役。
- "TTK kernel 模式全树无 expect_error 处理"的断言来自前序登记文件（本会话未重新 grep 验证 TTK 源树），若升级 TTK 版本须复核。


## 后续增量（2026-09-17，并行策展批：R3——未标注非法用例的新成因：min-shape special 路径不套用约束 sidecar）

R3（RoPE 类算子白盒跑批失败分类，Ascend950PR，CANN 9.2.0）：本卡适用场景中"**不带** expect_error 标记的用例出现 INVALID_TILING（那是真 tiling 缺陷，须另行定位）"这一分支有了新的成因实例——多出的"正确拒绝"条目可能来自**生成侧的非法用例**，不是 kernel 缺陷。

**对账判据（本形态的判别入口）**：跑批失败分类中"正确拒绝"类（dyn_precision 含 INVALID_TILING/DYN_COMPILE_FAILURE）条数 **>** 用例集 `expect_error` 标记条数 ⇒ 存在未标注的非法用例；逐例回 `design_doc_cases.json` 查 shape 即定位。本例：9 条正确拒绝 vs 8 条 expect_error 标记，多出的第 9 条是 min-shape 用例（query/key `[1,1,1,1]`、cos/sin `[1,1,0]`、`expect_error=False`）——末维 D=1 是奇数，违反算子"D 为偶数且 D==2*Dh"的 spec 强约束，host CheckShape 正确拒绝、TTK gate 记 FAIL。

**根因（会话内分析，未读引擎该路径源码逐行核验；依据是结构用例末维均被取偶、special 用例保持 1 的观测对比）**：min-shape 用例由引擎 special 路径按默认 `min_shapes=[[1,1,1,1]]` 生成，该路径不应用约束 sidecar（whitebox_constraints.yaml）的 dim_even 规则——结构扫描生成的 shape 末维会被取偶，special 路径不会；shape 合法性校验（validate-shapes）只查 rank_range 与 shape_rule，拦不住奇偶性违规。

**处置（生成侧优先，一劳永逸）**：design_facts 显式设 `min_shapes=[[1,1,1,2]]`（末维满足约束的最小偶数）——最小合法形本身值得真跑（D=2 时 Dh=1，覆盖单对尾块路径），不要把 min-shape 用例标 expect_error 了事（丢覆盖）。跑批侧对账为兜底：固定做"正确拒绝条数 vs expect_error 标记数"对账，超出部分逐例查 shape 合法性。生成侧预防探针（对 special 用例打印 shape 与 expect_error）：

```python
import json
d = json.load(open('operators/<Op>_package/tests/whitebox/<branch>/design_doc_cases.json'))
for c in d['cases']:
    sc = c.get('_scenario', '')
    if 'min' in sc or 'empty' in sc or 'extreme' in sc or 'scalar' in sc:
        q = c['tensors']['inputs']['query']['shape']
        print(sc, '| query=', q, '| err=', c.get('expect_error', False))
```

R3 验证信号：

- 跑批失败分类：正确拒绝类 9 条（INVALID_TILING/DYN_COMPILE_FAILURE，全 fp32）vs 用例集 expect_error 标记 8 条；min-shape 用例（query [1,1,1,1]，D=1 奇数，expect_error=False）在 FAIL 清单内——多出的第 9 条按排除法对账定位（来源平台：950；证据等级：tool_observed）。
- 用例集核对：该分支 min-shape 为 [1,1,1,1] 未标 expect_error；interleaved 姊妹分支生成时显式设 `min_shapes=[[1,1,1,2]]`，其 min-shape 用例合法（来源平台：950；证据等级：tool_observed）。

R3 边界与坏实践：

- 显式 min_shapes 只救 min-shape 用例；其他 special 路径（extreme/empty）若也携带约束违规维度，同样思路逐类检查（special 路径硬编码 params={} 不注入 attrs 的相邻缺口见 [白盒 special 用例 attrs 空致误路由卡](../validations/whitebox_special_cases_empty_attrs_misdispatch.md)）。默认最小形对 rank 类等其他约束是否也绕过未逐一验证。
- 坏实践：看到 INVALID_TILING/DYN_COMPILE_FAILURE 的 FAIL 就当 kernel 缺陷去修 kernel（先对账用例是否本就非法）；生成侧偷懒不设 min_shapes、跑批侧把多出来的正确拒绝 FAIL 手工划掉（掩盖用例集缺陷）。
