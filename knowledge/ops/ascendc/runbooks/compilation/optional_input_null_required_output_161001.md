---
type: "Runbook"
title: "可选输入空指针合法，必选输出不可漏建 161001"
description: 可选输入（perTokenScales）传 nullptr 合法，但其对应的输出若是 REQUIRED（permutePerTokenScales），测试侧不创建该输出 tensor 会让 aclnnXxxGetWorkspaceSize 返回 161001（ACLNN_ERR_PARAM_NULLPTR）；输入可选性与输出必选性是两个独立契约，输出创建只看 def 的 ParamType。
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# 可选输入空指针合法，必选输出不可漏建 161001

## 现象

BenchNoHarnessMoeReRouting（MoE token 重排算子）的 Real 模式 ST 中，所有「无scales」用例在进入 kernel 前就被拒绝：

```
[Real] L0_002 - 基础fp16+int32+无scales (dtype=float16/int32, scales=0, A=64, H=32, N=4, E=4)
[FAIL] GetWorkspaceSize 失败: 161001 (caseId=L0_002)
[Real] L0_007 - 中等shape+无scales ...
[FAIL] GetWorkspaceSize 失败: 161001 (caseId=L0_007)
```

错误码查证（头文件）：`include/aclnn/opdev/op_errno.h:15: #define ACLNN_ERR_PARAM_NULLPTR 161001`。有scales用例的 GetWorkspaceSize 全部正常。

## 适用场景

- 工程形态：aclnn 直连测试程序（两段式调用），覆盖「可选输入 + 必选输出」组合的算子。
- API 路径：`aclnnXxxGetWorkspaceSize`、op_api 生成的 `CheckNotNull`/`CheckParams`、`aclCreateTensor`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（op_api 参数校验层语义，与芯片无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：输出本身也是 OPTIONAL 的算子（def 中 ParamType(OPTIONAL) 输出，本轨迹未涉及）；只传 nullptr 输入、不检查输出契约的调用方。

## 根因

`per_token_scales` 是 OPTIONAL **输入**（不提供时传 nullptr 合法），但 `permute_per_token_scales` 是 REQUIRED **输出**——op_api 层的 `CheckNotNull` 会校验全部必选输出的非空。测试代码沿「无scales ⇒ scales 相关的输入输出都不创建」的直觉走，漏建了必选输出 tensor，`GetWorkspaceSize` 第一阶段校验即返回 161001。输入可选性与输出必选性是两个独立契约，不能联动推导。

## 解决配方

修复前（无scales场景跳过创建 scales 输出，verbatim）：

```cpp
// tests/st/test_aclnn_bench_no_harness_moe_re_routing.cpp
        // 3. 创建输出 aclTensor
        if (CreateAclOutputTensor(tokensBytesSize, tokensShape, tDtype,
                                  &permuteTokensDev, &permuteTokensTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 permuteTokens tensor 失败"); break;
        }
        if (tc.hasScales) {                                   // ← 无scales时漏建必选输出
            if (CreateAclOutputTensor(tc.A * sizeof(float), scalesShape, ACL_FLOAT,
                                      &permuteScalesDev, &permuteScalesTensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 permutePerTokenScales tensor 失败"); break;
            }
        }
```

修复后（REQUIRED 输出一律创建）：

```cpp
// tests/st/test_aclnn_bench_no_harness_moe_re_routing.cpp
        // 3. 创建输出 aclTensor
        if (CreateAclOutputTensor(tokensBytesSize, tokensShape, tDtype,
                                  &permuteTokensDev, &permuteTokensTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 permuteTokens tensor 失败"); break;
        }
        // permutePerTokenScales is REQUIRED output - always create it
        if (CreateAclOutputTensor(tc.A * sizeof(float), scalesShape, ACL_FLOAT,
                                  &permuteScalesDev, &permuteScalesTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 permutePerTokenScales tensor 失败"); break;
        }
```

定位方法（错误码 → 语义 → 契约核对的三步，可复用）：

```bash
# 1. 错误码语义
grep -rn "161001" $ASCEND_HOME_PATH/include/aclnn/opdev/op_errno.h
# op_errno.h:15:#define ACLNN_ERR_PARAM_NULLPTR 161001
# 2. 读 op_api 生成代码的 CheckNotNull：逐参数看哪些被校验（本例 perTokenScales 输入被注释豁免，
#    但 permutePerTokenScales 输出在必选校验清单里）
# 3. 回 def 核对每个 IO 的 ParamType：输入 OPTIONAL ≠ 输出 OPTIONAL
```

## 预防

- 写 aclnn 直连测试时，输出 tensor 的创建只看 def 的 `ParamType`，不看本次用例是否提供对应输入。
- 161001 出现时先跑上面三步定位法，不要先怀疑可选输入的 nullptr 传递方式。

## 验证信号

来源说明：BenchNoHarnessMoeReRouting 迭代一联调验证会话（Ascend950PR，CANN 9.1.0），修复前后同套 ST 对照。

- 修复前：`[FAIL] GetWorkspaceSize 失败: 161001 (caseId=L0_002/L0_007)`（仅无scales用例）（来源平台：950；证据等级：tool_observed）。
- 错误码锚点：`op_errno.h:15:#define ACLNN_ERR_PARAM_NULLPTR 161001`（来源平台：950；证据等级：tool_observed）。
- 修复后：L0_002 进入 kernel 执行（GetWorkspaceSize ret=0、aclnn ret=0），161001 消失（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 纯 op_api 参数校验层结论；若算子 def 把输出也声明为 OPTIONAL，行为不同（未验证）。
- 其他 CANN 版本的 CheckNotNull 生成形态未验证（推断可迁移）。

**未决项**：

- 无。
## 坏实践

- 「可选输入不传 ⇒ 对应输出也不给」——两个契约独立，漏建必选输出必吃 161001。

## 误诊

- 初判怀疑是可选输入 `perTokenScales` 的 nullptr 被框架拒绝——逐行读生成代码后排除：`CheckNotNull` 明确豁免该输入（注释 `// perTokenScales can be nullptr (optional)`），真实被拒的是漏建的必选输出。
