---
type: "Runbook"
title: ARA 向量化 padding 三方案无效须标量回退
description: 对 A0<64 的向量化 Max/Compare/Select ARA 路径，-INFINITY SetValue 填充、Cast 后 float 缓冲填充、Duplicate 预填充+对齐 castCount 三种 padding 修补全部无效（91/101 不变，无 padding 用例也失败），证明污染源在向量化路径自身逻辑层；正确出路是整体标量回退（101/101）。
tags: [task.debugging, topic.precision, topic.alignment]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# ARA 向量化 padding 三方案无效须标量回退

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上修自定义 argmax 算子 ARA 路径（dim 非最内轴）的 10 条 A0<64 边界用例失败时，先后按"把 padding 区域填成极小值，让它永远赢不了 Max"的思路连续尝试了三种向量化 padding 修补方案，每一种都完成代码修改、重编译、重装包、全量 ST Real 复测，**通过率全部停在 91/101，同样的 10 条用例以同样的 mismatch 率失败**：

- 方案一（搬入后填输入缓冲）：`DataCopyPad` 搬入 `curA0Len` 个有效元素后，用 `SetValue` 把 `[curA0Len, a0Align)` 填成 `-INFINITY`（float 类）或 `INT32_MIN`（int32 类），再做 Cast/Copy 进 maxLocal。结果：较此前 89/101 恢复了少数 ARA 用例（升至 91/101），但目标 10 条边界用例未恢复。
- 方案二（Cast 后填 float 缓冲）：把填充点从输入缓冲挪到 Cast 之后的 float 缓冲（`maxLocal`/`rowData` 上直接 `SetValue` -INFINITY）。结果：91/101 纹丝不动。
- 方案三（Duplicate 预填充 + 对齐 count Cast 覆盖有效区）：先 `Duplicate<float>(maxLocal, -INFINITY, a0Align)` 整批预填充，再只对有效长度按 32B 对齐的 `castCount` 做 `Cast` 覆盖前段。结果：91/101 纹丝不动。

三种方案代码形态彼此不同、填充时机覆盖"搬入后/Cast 后/预填充"三个位置，都无法让 A0<64 用例恢复——说明该向量化框架的问题不在"padding 值填没填对"，最终以整体标量回退收场（标量回退后同 10 条用例全部 bitwise-equal，101/101）。

## 适用场景

- 工程形态：aclnn 自定义算子 AscendC kernel，非最内轴归约（ARA）向量化 Max/Compare/Select 实现。
- API 路径：`DataCopyPad` + `Cast`/`DataCopy` + `Duplicate` + `SetValue` 组合的 padding 处理；`Max/Compare/Select`（count 须 256B 对齐）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——同平台"对齐宽度不足需 padding"的向量化比较/选择类算子。
- 路线优先级：banned（本卡路线 = "继续向量化 padding 修补"，对该问题已被客观证明无效并放弃；正确路线是标量回退）。
- 明确不适用/未验证场景：A0 恒 ≥64（无 padding）的向量化实现不适用；填充方案在"纯 Max 求值（不带索引跟踪）"场景是否有效未单独验证（本场景是 Max+索引跟踪组合）。

## 根因/判别

三种 padding 修补无效的判别结论：A0 < a0Align 时该向量化 Max/Compare/Select 框架的索引污染不只来自"padding 值不够小"——填 -INFINITY 后有效位置的索引更新仍然错误，且 A0=512（完全无 padding）的用例同样失败，证明污染源在向量化路径自身的逻辑层（多核切分边界或 Select 掩码跨轮消费），padding 填充救不了。

## 尝试与观测（可重建）

方案一（搬入后填输入缓冲，float/int32 两分支）：

```cpp
            LocalTensor<D_T_X> inData = inQueueX.template DeQue<D_T_X>();

            // Fill padding elements [curA0Len, a0Align) with minimum value
            // to prevent garbage from affecting Max/Compare/Select
            if (curA0Len < a0Align) {
                if constexpr (IsFloatDtype<D_T_X>::value) {
                    float minVal = -NumericLimits<float>::Infinity();
                    for (uint32_t i = curA0Len; i < a0Align; i++) {
                        inData.SetValue(i, static_cast<D_T_X>(minVal));
                    }
                } else {
                    for (uint32_t i = curA0Len; i < a0Align; i++) {
                        inData.SetValue(i, static_cast<D_T_X>(-2147483648LL));
                    }
                }
                PipeBarrier<PIPE_V>();
            }
```

方案二（Cast 后填 float 缓冲）：

```cpp
            if constexpr (NeedsCastToFloat<D_T_X>::value) {
                Cast<float, D_T_X>(maxLocal, inData, RoundMode::CAST_NONE, a0Align);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(maxLocal, inData, a0Align);
                PipeBarrier<PIPE_V>();
            }

            // Fill padding in float maxLocal with -INFINITY (after Cast)
            if (curA0Len < a0Align) {
                for (uint32_t i = curA0Len; i < a0Align; i++) {
                    maxLocal.SetValue(i, -NumericLimits<float>::Infinity());
                }
                PipeBarrier<PIPE_V>();
            }
```

方案三（Duplicate 预填充 + 对齐 castCount 覆盖有效区）：

```cpp
            // Pre-fill maxLocal with -INFINITY, then Cast/Copy only valid elements.
            // Padding stays -INFINITY, preventing garbage from affecting Max/Compare/Select.
            Duplicate<float>(maxLocal, -NumericLimits<float>::Infinity(), a0Align);
            PipeBarrier<PIPE_V>();

            if constexpr (NeedsCastToFloat<D_T_X>::value) {
                // Cast curA0Len elements (Cast count must be 32B-aligned for src)
                uint32_t castCount = (curA0Len + (32 / sizeof(D_T_X) - 1)) & ~(32 / sizeof(D_T_X) - 1);
                if (castCount > a0Align) castCount = a0Align;
                Cast<float, D_T_X>(maxLocal, inData, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
            } else {
                // fp32: DataCopy aligned count
                uint32_t copyCount = (curA0Len + 7) & ~7u;  // align to 8 (32B for float)
                if (copyCount > a0Align) copyCount = a0Align;
                DataCopy(maxLocal, inData, copyCount);
                PipeBarrier<PIPE_V>();
            }
```

逐方案观测（每方案后重编译、`--force` 重装、ST Real 全量复测）：

- 方案一后：ST 91/101（10 条 A0<64/边界用例仍失败，mismatch 率不变）；
- 方案二后：ST 91/101（同 10 条，无变化）；
- 方案三后：ST 91/101（同 10 条，无变化）。

## 失效边界

- 本卡证明无效的范围：A0 < 64 的向量化 Max+Compare(EQ)+Select(VSEL_TENSOR_SCALAR_MODE) 索引跟踪场景（fp16/fp32/bf16/int32，A0∈{4,16,32,42,50}），以及 A0=512 无 padding 仍失败的事实。
- 不能外推为"所有 padding 填充都没用"：若问题确实只出在 padding 值（例如纯 Max 求值、无 Select 掩码/索引跟踪），-INFINITY 填充仍是标准做法——本卡的判别要点是"填充后失败不变 + 无 padding 用例也错"这一组合证据。
- 正确出路：该场景最终由标量 argmax 回退修复（放弃向量化 Max/Compare/Select，改为标量逐元素比较跟踪索引），ST 恢复 101/101。

**未决项**：

- 三种方案为何无效的机制层解释（填充是否被后续 `Duplicate(idxLocal,0,a0Align)` 或 Select 的整批宽度覆盖回垃圾/初值）未逐一实验；若未来必须向量化，需先追 A0=512 用例的根因（复现并追根因为优先项）。
## 坏实践

- 在没有"无 padding 用例也失败"交叉证据前，连续多轮重复同一思路（换填充位置/换填充方式）——三轮修补消耗了整段联调时间，收益为零。
- 修补方案不逐轮记录观测（哪条用例恢复/没变），导致无法及时判断思路整体无效。

## 验证信号

以下信号来自该算子联调会话的三轮 padding 修补循环，每轮均以 ST Real 全量复测收口（均为工具实测）：

- 方案一（输入缓冲 SetValue 填充）后：ST Real 91/101，10 条 ARA 边界用例仍失败（来源平台：950；证据等级：tool_observed）。
- 方案二（Cast 后 float 缓冲填充）后：ST Real 91/101，同 10 条失败、mismatch 率不变（来源平台：950；证据等级：tool_observed）。
- 方案三（Duplicate 预填充+对齐 castCount）后：ST Real 91/101，同 10 条失败、mismatch 率不变（来源平台：950；证据等级：tool_observed）。
- 对照：放弃 padding 思路、整体标量回退后同 10 条用例全部 bitwise-equal，101/101 通过（来源平台：950；证据等级：tool_observed）。
