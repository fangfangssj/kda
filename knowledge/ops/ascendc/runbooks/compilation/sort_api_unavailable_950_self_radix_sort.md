---
type: "Runbook"
title: 950 无 Sort API：unique 改自实现排序
description: CANN 9.1.0 asc-devkit 的 Sort/Sort32/MrgSort/MrgSort4/RpSort16 Context API 产品支持矩阵仅列 A3/A2/Kirin X90/Kirin 9030、不含 Ascend950，且 Sort 族仅支持 half/float dtype；需要排序语义（如 unique 去重）的算子须自实现排序，本轨迹采用 uint16 可排序键编码 + 4-bit LSD radix sort（GM ping-pong）。
tags: [task.design, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-17T04:41:27Z'
updated_at: '2026-09-17T04:41:27Z'
platforms: ["950"]
---

# 950 无 Sort API：unique 改自实现排序

## 现象

为 BenchNoHarnessUnique 算子（对标 torch.unique 的去重，7 种 dtype + 可选 return_inverse）做方案设计时，算法路径需要"先排序后去重"。最初的 API 选型是 Context 级排序 API（Sort/MrgSort 家族，按 score 排序并携带 index）。逐一查阅本机 CANN 9.1.0 asc-devkit 的 API 文档（`asc-devkit/docs/api/context/` 下的 Sort.md、Sort32.md、MrgSort.md、MrgSort-37.md、MrgSort4.md、RpSort16.md、GetSortLen.md、GetSortTmpSize.md、GetSortOffset.md），每一篇的"产品支持情况"表都是：

```
| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Kirin X90 | √ |
| Kirin 9030 | √ |
```

均未列出 Ascend950 系列（开发目标平台是 Ascend950PR / arch35 / DAV_3510）；RpSort16 连 A3/A2 都标 ×。另外两个限制：Sort 系列 API 仅支持 half/float 两种 dtype（无法覆盖本算子 uint8/int8/fp16/bf16/int32/fp32/int64 七种）；其输出是 (score, index) 定长结构排布，与去重所需的"值 + 位置"复用不直接匹配。结论：在 950 上 Sort 家族不可用，排序必须自实现。

## 适用场景

- 工程形态：AscendC 算子方案设计——任何需要排序/去重/按序紧凑语义的算子（unique、sort-based top-k、去重计数等）在 Ascend950 上的 API 选型。
- API 路径：`asc-devkit/docs/api/context/` 的 Sort/Sort32/MrgSort/MrgSort-37/MrgSort4/RpSort16/GetSortLen/GetSortTmpSize/GetSortOffset 文档（选型核对面）；替代实现走基础向量 API（DataCopyPad/Cast/Compare/Select/And/Xor/ShiftRight/Duplicate/ReduceSum/ReinterpretCast）与标量回退。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510）、CANN 9.1.0（与 frontmatter `platforms` 一致；证据为本机 devkit 文档支持矩阵 + 替代方案骨架编译通过）。
- 适用粒度：platform_wide——950 上所有需要排序语义的算子设计。
- 路线优先级：本轨迹采纳自实现 radix sort（recommended_fastpath，在 Sort API 补齐 950 支持前）；若未来版本补齐，Sort API 路线需重新评估（dtype 覆盖仍受限）。
- 明确不适用 / 未验证场景：A3/A2/Kirin 平台（文档明确支持 Sort 族，直接用官方 API）；"未列出 = 不支持"的判定仅基于文档支持矩阵，未实际编译 Sort 调用在 950 上的报错形态（未验证）。

## 根因

CANN 9.1.0 的 Sort/Sort32/MrgSort/MrgSort4/RpSort16 Context API 支持矩阵只覆盖 A3/A2/Kirin X90/Kirin 9030 平台，不含 Ascend950（DAV_3510），且仅支持 half/float dtype，排序类算子在 950 上无现成高层 API 可用。

## 解决配方（本轨迹采纳的替代方案：4-bit LSD radix sort 自实现）

设计要点（已在骨架迭代编译验证）：

1. **uint16 可排序键编码**（全部 7 种 dtype 归一为 uint16 word 序列，使无符号升序 = 原值升序）：

| 输入 dtype | 位宽 | 键 word 数 | 编码变换 |
|-----------|------|-----------|---------|
| uint8 | 8 | 1 | `key = (uint16)val`（零扩展） |
| int8 | 8 | 1 | `key = (uint16)val ^ 0x0080`（翻转符号位后零扩展） |
| fp16 | 16 | 1 | `raw = ReinterpretCast<uint16>(val)`; `if (raw & 0x8000) key = ~raw; else key = raw \| 0x8000` |
| bf16 | 16 | 1 | 同 fp16（均为 16-bit，符号位在 bit15） |
| int32 | 32 | 2 | 高 word: `hi = (uint16)(val >> 16) ^ 0x8000`; 低 word: `lo = (uint16)val` |
| fp32 | 32 | 2 | `raw = ReinterpretCast<uint32>(val)`; 高 word: `if (raw & 0x80000000) hi = ~(uint16)(raw>>16); else hi = (uint16)(raw>>16) \| 0x8000`; 低 word: `lo = (uint16)raw` |
| int64 | 64 | 4 | 最高 word: `w3 = (uint16)(val >> 48) ^ 0x8000`; 其余: `w2..w0` 直接截位 |

   编码特性：负数映射为小值；`-inf → 0x0000`（最小）、`+inf → 0xFFFF`（最大）、`-0.0 → 0x7FFF` 与 `+0.0 → 0x8000` 相邻（去重时按 IEEE754 语义合并 ±0、保留 +0 符号）。

2. **为什么按 uint16 word 而不是 uint32 处理**：And/Or/Xor 文档标注仅支持 int16_t/uint16_t dtype（不支持 int32/uint32），32/64-bit 键必须拆 word 做位运算。

3. **排序趟数**：每 word 4 趟（16 bits / 4-bit digit = 16 桶）；1 word dtype 4 趟、int32/fp32 8 趟、int64 16 趟。LSD 从低位到高位保证稳定性。

4. **单趟流程**（骨架已实现为标量版，向量化为后续迭代）：
   - 加载当前 word 的 key 到 UB；
   - 提取 4-bit digit（`digit = (key >> digitShift) & 0x000F`）；
   - 16 桶直方图 → 前缀和得全局桶偏移；
   - 按桶收集元素到 compactBuf、`DataCopyPad` 连续写出（per-bucket 连续写出避免逐元素随机 scatter）；
   - GM ping-pong：偶数趟 keysGM → sortedGM，奇数趟反向，趟数偶数时结果落 keysGM。

5. **去重与 inverse**：排序后相邻比较（`decoded[i] != decoded[i-1]`）生成紧凑标记，前缀和得 compactId；唯一值写出 y；`inverse[sortedIndices[i]] = compactId[i]` 散射生成逆索引；浮点 ±0 在编码层已相邻、按语义合并。

骨架 kernel（三阶段：EncodeKeys / RadixSort / DedupAndInverse；当前骨架仅实现 int32 + 无 inverse + 单核）在 ascend950 编译通过，生成算子包；向量化与多核化留待后续迭代。

## 设计决策

- 为什么选 4-bit digit（16 桶）而不是 8-bit（256 桶）：桶数 16 时直方图/前缀和/桶内收集都是小常数循环，UB 占用可控（骨架 UB 系数按 3 buffer × keySize 预算）；8-bit 桶数翻 16 倍，散列循环开销在标量骨架期不划算。趟数与桶数的权衡（4-bit × 8 趟 vs 8-bit × 4 趟）在向量化后再评估。
- 为什么自实现而不是"借用其他高层 API 组合"（如 CumSum+Compare+Select 模拟）：设计阶段排查过 Concat/CumSum/Extract/Select 等 Context API，均不提供排序语义；Compare/Select/ReduceSum 可作 radix 单趟的向量化积木。
- 被放弃的路线：Sort/MrgSort 家族（平台不支持）；RpSort16（连 A3/A2 都不支持）。
- 当前方案路线档位：recommended_fastpath（950 上排序语义的可行解）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1）、CANN 9.1.0、x86_64 宿主机编译。
- 相关文件：`docs/DESIGN.md` §3.1（radix 替代方案与键编码表）、`issues/issue_20260721_sort_api_unavailable_1.md`（Sort API 不可用记录）、`op_kernel/arch35/bench_no_harness_unique.h`（三阶段骨架实现）。

## 验证信号

以下信号来自该算子的设计核验与骨架编译：先逐一读取 Sort 族文档确认支持矩阵，再以 radix 骨架编译验证替代方案可落地。

- Sort/Sort32/MrgSort/MrgSort-37/MrgSort4/RpSort16/GetSortLen/GetSortTmpSize/GetSortOffset 九篇文档的"产品支持情况"表均只列 Atlas A3/A2/Kirin X90/Kirin 9030，无 Ascend950；RpSort16 对 A3/A2 标 ×（来源平台：950 开发环境，CANN 9.1.0 asc-devkit 文档；证据等级：tool_observed）。
- And/Or/Xor 文档 dtype 表标注仅支持 int16_t/uint16_t（A2/A3/Kirin 各行一致），据此键编码拆 uint16 word（来源平台：950；证据等级：tool_observed）。
- 替代方案骨架（int32 + 8 趟 4-bit radix + 去重）在 ascend950 编译通过，生成 kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 算子包（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "Sort 族在 950 不可用"的判定基于 CANN 9.1.0 asc-devkit 文档支持矩阵（文档未列出 950），未实际编写 Sort 调用在 950 编译验证报错形态；未来 CANN 版本若补齐 950 支持则本卡结论过时，届时优先官方 API（但 half/float dtype 限制仍需评估）。
- 非 950 平台未在本轨迹验证（A3/A2/Kirin 按文档直接使用官方 Sort API）。
- radix 方案的向量化实现（向量 API 正确签名）与多核化（SyncAll 正确用法）未验证；标量骨架的性能不代表最终性能。
- 浮点 ±0/±inf 语义依赖键编码的 IEEE754 全序性质；带 NaN 输入的排序语义（NaN 与自身不相等）未设计、未验证。
- int64（16 趟）与 1 word dtype（4 趟）路径未在骨架中实现（骨架只做 int32）。

**未决项**：

- radix 向量化（Duplicate+张量-张量 Xor/And、完整 6 参 ShiftRight、CMPMODE Compare、void ReduceSum）与多核化的编译/性能验证——未验证的接续点。
- 上板正确性（L0 ST 全量、±0/±inf/NaN 边界）待联调验证。
## 坏实践

- 在 950 上不做支持矩阵核查、按 A2/A3 经验直接调用 Sort/MrgSort 家族——设计返工。
- 只看 API 名字（"有 Sort 就用"）不看 dtype 覆盖——即使平台支持，half/float 限制也覆盖不了整数去重算子。

## 误诊

- 无（设计期一次核验定位，未走弯路）。
