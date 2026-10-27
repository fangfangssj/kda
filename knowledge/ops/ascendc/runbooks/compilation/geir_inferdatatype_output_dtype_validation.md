---
type: "Runbook"
title: GEIR负向SUPPRESSED：InferDataType缺输出dtype校验
description: L2 负向用例输出 dtype 与输入不一致时 GEIR 图编译照常通过致 SUPPRESSED 放行；InferDataType 补 GetOutputDataType 显式设置校验，不一致返回 GRAPH_FAILED。
tags: [task.testing, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GEIR负向SUPPRESSED：InferDataType缺输出dtype校验


## 现象

在 ACLNN 算子仓开发流程（ops-registry-invoke）中给 ThnnFusedGruCellGrad 做 GEIR 图模式验收，负向套件以 `--golden-mode Disable` 运行（`ttk geir -i …/negative.csv --golden-mode Disable --plugin …/tests/geir/geir_spec.py`）。49 条 L2 负向用例中 42 条按预期 EXEC_FAILURE（图编译/执行阶段被拒收），但 7 条的判定是 `precision=SUPPRESSED status=PASS`：

```
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_shape_023: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_dimval_029: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_dimval_033: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_dimval_038: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_dimval_039: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_dimval_041: precision=SUPPRESSED status=PASS
non-reject: aclnnThnnFusedGruCellGrad_L2_exc_constraint_046: precision=SUPPRESSED status=PASS
precision distribution: {'EXEC_FAILURE': 42, 'SUPPRESSED': 7}
```

这 7 条用例的 shape 全部合法（storage 第二维 = 5×H）、输入 dtype 合法（如 `('float32','float32')`），所以 GE 图编译、执行都成功了；验收脚本要求负向"全部真实拒绝"（EXEC_FAILURE + `ERROR:` 前缀），7 条 SUPPRESSED 被归类为 pass，reject 检查失败。任务目标"L2 负向全部在 GEIR 执行阶段真实拒绝"无法达成。

## 适用场景

- **工程形态**：GEIR 图模式负向验收（TTK `geir` 模式 + `--golden-mode Disable`，V2 自定义算子）；ops-registry-invoke 算子仓流程。
- **API 路径**：`gert::InferDataTypeContext`（`GetInputDataType` / `GetOutputDataType` / `SetOutputDataType`，头文件 `include/exe_graph/runtime/infer_datatype_context.h`）；`IMPL_OP(<Op>).InferShape(…).InferDataType(…)` V2 注册。
- **验证过的平台与 CANN 版本**：950（ascend950）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- **适用粒度**：`operator_family`——任何多输出 V2 自定义算子的 InferDataType，其负向用例通过 CSV/图构造方显式声明输出 dtype。
- **路线优先级**：`recommended_fastpath`——在 InferDataType 内补输出 dtype 一致性校验（一处宿主侧代码，无需动 kernel/tiling/测试）。
- **明确不适用/未验证场景**：单输出且 dtype 恒定的算子意义有限；V1（OpDef SetInferShape）注册路径未验证；图构造方未显式设置输出 dtype（DT_UNDEFINED）的场景不走该校验（by design，见设计决策）。

## 根因

（单句）这 7 条 L2 用例在 CSV 里声明了与输入不一致的**混合输出 dtype**（如 float32 输入配 `('float16','float32','bfloat16','float32','float32')` 输出），而算子 `op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp` 的 `InferDataType` 只校验输入 dtype 合法域与双输入一致，随后无条件 `SetOutputDataType(i, gradHyDtype)` 覆盖了图构造方声明的输出 dtype，使本应在 aclnn 侧触发 `ACL_ERROR_DTYPES_NOT_MATCH` 类错误的负向用例在 GEIR 图编译阶段通过。

**机制链（为何表现为 SUPPRESSED 而非 FAIL）**：TTK 负向套件 `--golden-mode Disable` → 框架把 golden_arrays 置为字符串 `"SUPPRESSED"`（不生成 golden）→ 比较框架见 `isinstance(golden, str)` 即记 `precision="SUPPRESSED"`、`pass=_filter_fake_fail("SUPPRESSED")=True` → 结果 CSV 的 precision_status 列为 "PASS" → 验收 `_classify` 按 precision_status 归 "pass" → reject 检查（要求全部 `precision=="EXEC_FAILURE"` 且 `precision_status.startswith("ERROR:")`）失败。即：**图没被拒收 + golden 被禁用 = 静默放行**。

**证据（CSV 原文，输出 dtype 列与输入 dtype 列不一致）**：

```
L2_exc_shape_023:       输入 ('bfloat16','bfloat16') → 输出 ('float32','bfloat16','bfloat16','bfloat16','float16')
                        remark=输出shape与期望不匹配; expect ACL_ERROR_SHAPE_NOT_MATCH
L2_exc_dimval_029:      输入 ('float32','float32')   → 输出 ('float16','float32','bfloat16','float32','float32')
                        remark=维度上溢; expect ACL_ERROR_INVALID_DIMENSION
L2_exc_dimval_038:      输入 ('float32','float32')   → 输出 ('bfloat16','float16','float16','float32','bfloat16')
                        remark=维度下溢; expect ACL_ERROR_INVALID_DIMENSION
L2_exc_constraint_046:  输入 ('float32','float32')   → 输出 ('float16','float32','bfloat16','bfloat16','float16')
                        remark=EX-S5; 约束违反; expect ACL_ERROR_DTYPES_NOT_MATCH
（dimval_033/039/041 同模式）
```

## 解决配方

**before**（`op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp` 的 `InferDataType`，修改前完整函数）：

```cpp
graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType gradHyDtype = context->GetInputDataType(0);
    const ge::DataType storageDtype = context->GetInputDataType(1);

    // ── dtype 合法域校验：仅 {DT_FLOAT, DT_FLOAT16, DT_BF16}（proto.md §2）──
    const bool gradHyOk =
        (gradHyDtype == ge::DT_FLOAT) || (gradHyDtype == ge::DT_FLOAT16) || (gradHyDtype == ge::DT_BF16);
    const bool storageOk =
        (storageDtype == ge::DT_FLOAT) || (storageDtype == ge::DT_FLOAT16) || (storageDtype == ge::DT_BF16);
    if (!gradHyOk || !storageOk) {
        return GRAPH_FAILED;
    }

    // ── 双输入 dtype 一致校验（proto.md §2：三组合内逐 IO 同 dtype）──
    if (gradHyDtype != storageDtype) {
        return GRAPH_FAILED;
    }

    // ── 五输出 dtype 随 grad_hy ──
    for (size_t i = 0; i < 5; ++i) {
        if (context->SetOutputDataType(i, gradHyDtype) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}
```

**after**（在双输入一致校验与 SetOutputDataType 之间插入输出 dtype 一致性校验）：

```cpp
    // ── 双输入 dtype 一致校验（proto.md §2：三组合内逐 IO 同 dtype）──
    if (gradHyDtype != storageDtype) {
        return GRAPH_FAILED;
    }

    // ── 五输出 dtype 与输入一致校验（proto.md §2：三组合内逐 IO 同 dtype）──
    // 图构造方（CSV / aclnn caller）可能为输出设置与输入不一致的 dtype。
    // 若输出 dtype 已显式设置（非 DT_UNDEFINED）且与 grad_hy 不一致，拒绝
    // （对应 aclnn 侧 ACL_ERROR_DTYPES_NOT_MATCH 语义，使 L2 负向用例在
    // GEIR InferDataType 阶段被正确拒收，而非 SUPPRESSED 通过）。
    for (size_t i = 0; i < 5; ++i) {
        const ge::DataType outDtype = context->GetOutputDataType(i);
        if (outDtype != ge::DT_UNDEFINED && outDtype != gradHyDtype) {
            return GRAPH_FAILED;
        }
    }

    // ── 五输出 dtype 随 grad_hy ──
    for (size_t i = 0; i < 5; ++i) {
        if (context->SetOutputDataType(i, gradHyDtype) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}
```

**重建与重装**（改的是算子包源码，必须重建 + 重新 staging，否则 GE 仍加载旧 so）：

```bash
source docs/ThnnFusedGruCellGrad/develop/env.sh && cd operators/ThnnFusedGruCellGrad_package \
  && rm -rf build && cmake -S . -B build >&2 2>&1 && cmake --build build --target binary package -j 2>&1 | tail -5

STAGING=$(mktemp -d /tmp/ttk_geir_ThnnFusedGruCellGrad.XXXXXX) \
  && operators/ThnnFusedGruCellGrad_package/build/custom_opp_ubuntu_x86_64.run --install-path="$STAGING" --quiet 2>&1 | tail -3 \
  && VENDOR=$(find "$STAGING/vendors" -mindepth 1 -maxdepth 1 -type d | head -1) \
  && echo "$VENDOR" > operators/ThnnFusedGruCellGrad_package/tests/geir/runtime/staging.path
```

**负向套件重跑**（`--golden-mode Disable`，同 TTK 入口；详见"验证信号"）。

## 设计决策

- **为何放在 InferDataType 而非 InferShape**：dtype 校验语义归属 InferDataType，且 `GetOutputDataType` 只在 `InferDataTypeContext` 暴露（CANN 头文件 `exe_graph/runtime/infer_datatype_context.h:82`）；InferShapeContext 拿不到输出 dtype。
- **为何加 `DT_UNDEFINED` 守卫**：图构造方未显式设置输出 dtype 时不拒绝——避免误伤"不声明输出 dtype、由算子推导"的正常调用路径；正向用例输出 dtype 与输入一致（proto 三组合内逐 IO 同 dtype），不受影响（终验 44/44 × 4 套不回归证实）。
- **放弃的候选**：① 改 golden 函数让这 7 例抛错——golden-mode Disable 下 golden 函数根本不被调用，不可行；② 改验收脚本把 SUPPRESSED 当拒收——脚本在 skills/ 下不可改，且会弱化"真实拒绝"语义；③ 接受为已知限制——违背任务"L2 负向全部真实拒绝"的验收目标。
- **当前方案属哪档路线**：`recommended_fastpath`（一处宿主侧校验，等价对齐 aclnn 侧 `ACL_ERROR_DTYPES_NOT_MATCH` 拒收语义）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（gcc 13.3.0，cmake + CPack 打包 `custom_opp_ubuntu_x86_64.run`）；TTK `/tmp/ops-test-kit`；GEIR 由 `skills/geir-route-tester/scripts/run_ttk_geir.sh` / `verify_ttk_geir.py` 驱动。
- 目标平台：ascend950。
- 相关文件：
  - `operators/ThnnFusedGruCellGrad_package/op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp`（本次修改：InferDataType 增输出 dtype 校验；`IMPL_OP(ThnnFusedGruCellGrad).InferShape(ge::InferShape).InferDataType(ge::InferDataType)` V2 注册不变）
  - `operators/ThnnFusedGruCellGrad_package/tests/blackbox/testcases/aclnnThnnFusedGruCellGrad_l2_exception.csv`（负向用例源，只读）
  - `operators/ThnnFusedGruCellGrad_package/tests/geir/negative.csv`（投影后负向用例，只读）
  - `/tmp/ops-test-kit/ttk/core_modules/comparison/comparison.py`、`/tmp/ops-test-kit/ttk/core_modules/npu/op/output_generation.py`（SUPPRESSED 机制参照，只读）
  - `docs/ThnnFusedGruCellGrad/develop/GEIR_VERIFY.md` / `LOG.md`（验收记录）

## 完整代码

见"解决配方"——before/after 为 `InferDataType` 函数级完整代码块（含重建/重装/重跑命令）。无其他改动文件（kernel、tiling、OpAPI、用例 CSV、golden 均未动）。

## 预防

- V2 多输出算子的 `InferDataType` 不要只做"输入校验 + 无条件 SetOutputDataType"——**图构造方声明的输出描述（dtype）也要校验**：显式设置且与推导结果不一致 → `GRAPH_FAILED`。
- 负向验收不能只看 `status=PASS/FAIL`：`precision=SUPPRESSED`（golden 禁用/缺失）+ 执行成功 = 静默放行；验收标准应要求 `precision==EXEC_FAILURE` 且 `precision_status` 以 `ERROR:` 开头。
- 改算子包源码（op_graph/op_host）后必须重建 + 重新 staging 再跑 GEIR，GE 按暂存目录加载 so。

## 验证信号

本卡验证来自同一次 GEIR 图模式修复会话（ThnnFusedGruCellGrad，ascend950 + CANN 9.0.0-beta.2）：修复前负向套件 42 EXEC_FAILURE + 7 SUPPRESSED，重建重装后重跑负向与全部六套。

- 修复后 negative-source-static 重跑：49/49 全部 `precision=EXEC_FAILURE`，`precision_status` 前缀 `ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error! … Call InferShapeAndType for node:op1(ThnnFusedGru…`（来源平台：950；证据等级：tool_observed）。
- 验收脚本确认：`verify_ttk_geir.py check-result --expect reject` → `PASS geir-result expect=reject {'total': 49, 'pass': 0, 'fail': 49, 'skip': 0}`（来源平台：950；证据等级：tool_observed）。
- 正向不回归：4 个正向套件 `expect=pass` 全 PASS（各 44/44，static 两套含 2 条已登记 GEIR 已知阻断）（来源平台：950；证据等级：tool_observed）。
- 终态六套验证：正向 4×44 PASS + 负向 2×49 EXEC_FAILURE，check-staging/check-frozen/projection 全 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `DT_UNDEFINED` 守卫依赖"未显式设置"语义：若某框架路径在设置输出 dtype 之前调用 InferDataType，会读到 DT_UNDEFINED 而漏校验（本 trace 未观察到该路径，属理论边界）。
- 仅在"图构造方显式声明输出 dtype"的通路（TTK CSV output_dtypes → 生成 C++ 的 `update_output_desc_*`）上验证；其他图构造方式未验证。
- 拒收错误在 GE 侧统一表现为 EZ9999 Inner Error + `Call InferShapeAndType` 失败，错误码粒度不区分 dtype/shape 原因——需要区分时靠用例构造。
- 950 + CANN 9.0.0-beta.2 实测；其他平台未验证（InferDataTypeContext 属 CANN 通用接口，推断可迁移）。

## 坏实践

- `InferDataType` 只校验输入、无条件覆盖输出 dtype——负向用例（输出 dtype 不匹配）全部静默通过，且因 golden Disable 连 FAIL 都不报（SUPPRESSED+PASS）。
- 只看结果 CSV 的 `status` 列不看 `precision` 列——SUPPRESSED 混在 PASS 里不易察觉；应按 precision 值分布统计（`{'EXEC_FAILURE': 42, 'SUPPRESSED': 7}`）。

## 误诊

- 初始怀疑"7 例是 GE 引擎层面的已知限制、只能文档化接受"：框架源码级排查（comparison.py 的 `isinstance(golden, str)` 分支 + output_generation.py 的 golden-mode Disable 行为）确认 SUPPRESSED 只是"无 golden 可比"，真正的漏洞在算子侧 InferDataType 未拒收——修复后 49/49 EXEC_FAILURE 证明可修，无需登记限制。

## 未决事项

- 7 例中 L2_exc_shape_023 的 remark 期望 `ACL_ERROR_SHAPE_NOT_MATCH`（shape 语义），本次经输出 dtype 不一致路径拒收——shape 维度是否需要在 GEIR 侧独立校验（对应 remark 的 aclnn 语义）未深挖；当前 dtype 校验已足够使其被拒收。
