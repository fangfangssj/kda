---
type: "Runbook"
title: 二进制数少于TilingKey组合数不等于缺实例
description: ASCENDC_TPL 多模板算子编译产物按输入 dtype 生成二进制，同一二进制内含多个 TilingKey 实例；3 个 .o 覆盖 6 组合是正常现象，判别看 binary_info_config.json。
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# 二进制数少于TilingKey组合数不等于缺实例

## 现象

自定义 GRU 算子（Ascend950PR，CANN 9.2.0）用 ASCENDC_TPL 声明了双模板参数：`ASCENDC_TPL_UINT_DECL(DTYPE_ID, 2, ..., FP32, FP16, BF16)` × `ASCENDC_TPL_UINT_DECL(HAS_BIAS, 1, ..., 0, 1)`，即 3 dtype × 2 bias = 6 个 TilingKey 组合；`ASCENDC_TPL_SEL` 也按 3 个 ARGS_SEL（每 dtype 一个、内含 HAS_BIAS 0/1）声明。补齐 kernel 模板分支后编译通过，但检查产物：

```text
ls build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_gru/*.o | wc -l
3
```

只有 3 个 `.o`（`BenchNoHarnessGru_48c9e4f6...o`、`..._5b052ad1...o`、`..._b4ce59a3...o`），少于声明的 6 个组合。开发过程中两次修改 `ASCENDC_TPL_SEL` 写法（先合并为一个 ARGS_SEL，再拆回每个 dtype 一个）、清空 build 目录和 `~/atc_data/kernel_cache` 后全量重编，`.o` 数量始终是 3。当时怀疑"bf16（d=2）没被编译进产物"，把后续 fp16/bf16 测试失败也归因于此，绕了数轮弯路。

## 适用场景

- 工程形态：AscendC 算子仓开发（ASCENDC_TPL 模板算子），编译产物核对 / TilingKey 覆盖检查。
- API 路径：`op_kernel/arch35/*_tiling_key.h` 的 `ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_UINT_DECL / ASCENDC_TPL_SEL`；构建产物 `build/op_kernel/ascendc_kernels/binary/config/ascend950/binary_info_config.json` 与 `binary/ascend950/<op>/<Op>_<hash>.json`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.2.0 自定义算子编译产物结构通用的判别规则。
- 路线优先级：recommended_fastpath（判别规则，无副作用）。
- 明确不适用 / 未验证场景：
  - 真正缺实例的情形（例如 ASCENDC_TPL_SEL 漏列某个取值、tiling 手算 TilingKey 与声明 bit 布局不一致）——那类问题表现为运行时选错/找不到 kernel 或某组合行为异常，不是"数 .o 文件"能确诊的；本卡只提供"数量少 ≠ 缺实例"的判别与核实方法。
  - Cube/Vector 混合编译（coreType 多于一种）时产物数量规律未验证。

## 判别方法

触发条件：编译成功但 `.o` 数量 < `ASCENDC_TPL_SEL` 声明的模板组合数，且想判断是否缺实例。

可执行探针（本例实测命令与输出）：

```bash
# 1) 数产物
ls build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_gru/*.o | wc -l
# → 3

# 2) 看每个二进制覆盖哪些键、对应哪个 .o
cat build/op_kernel/ascendc_kernels/binary/config/ascend950/binary_info_config.json
```

关键输出（节选，`binaryList` 每项一个二进制）：

```json
{
  "BenchNoHarnessGru": {
    "dynamicRankSupport": true,
    "simplifiedKeyMode": 0,
    "binaryList": [
      {
        "coreType": 2,
        "simplifiedKey": [
          "BenchNoHarnessGru/d=0,p=1/0,2/0,2/0,2/0,2/0,2",
          "BenchNoHarnessGru/d=0,p=0/0,2/0,2/0,2/0,2/0,2",
          "BenchNoHarnessGru/d=1,p=1/0,2/0,2/0,2/0,2/0,2",
          "BenchNoHarnessGru/d=1,p=0/0,2/0,2/0,2/0,2/0,2"
        ],
        "binPath": "ascend950/bench_no_harness_gru/BenchNoHarnessGru_48c9e4f6fbd3f1ddb67942c52090be0a.o"
      },
      {
        "coreType": 2,
        "simplifiedKey": [
          "BenchNoHarnessGru/d=0,p=1/1,2/1,2/1,2/1,2/1,2",
          "BenchNoHarnessGru/d=0,p=0/1,2/1,2/1,2/1/1,2",
          "BenchNoHarnessGru/d=1,p=1/1,2/1,2/1,2/1,2/1,2",
          "BenchNoHarnessGru/d=1,p=0/1,2/1,2/1,2/1,2/1,2"
        ],
        "binPath": "ascend950/bench_no_harness_gru/BenchNoHarnessGru_5b052ad16297b5f640468eb8f449c9db.o"
      },
      {
        "coreType": 2,
        "simplifiedKey": [
          "BenchNoHarnessGru/d=0,p=1/27,2/27,2/27,2/27,2/27,2",
          "BenchNoHarnessGru/d=0,p=0/27,2/27,2/27,2/27,2/27,2",
          "BenchNoHarnessGru/d=1,p=1/27,2/27,2/27,2/27,2/27,2",
          "BenchNoHarnessGru/d=1,p=0/27,2/27,2/27,2/27,2/27,2"
        ],
        "binPath": "ascend950/bench_no_harness_gru/BenchNoHarnessGru_b4ce59a3882e8dcd725cb0ac77b76872.o"
      }
    ]
  }
}
```

解读：每个二进制的 `simplifiedKey` 斜杠后是逐输入的 `dtypeId,format` 列表——三个二进制分别对应全部输入 dtypeId=0（fp32）、1（fp16）、27（bf16）；每个二进制同时挂着 `d=0/1 × p=0/1` 共 4 条键。也就是说**二进制按输入 dtype 选择，bias/no-bias 等其余模板维度不拆分二进制**，多个 TilingKey 实例编译进同一个 `.o`，运行时由 TilingKey 在二进制内选实例。

最终行为验证（比数文件更有说服力）：跑覆盖全部 6 组合的 ST 用例（fp32/fp16/bf16 × bias/no-bias，含 L1-015/L1-016 no-bias、L1-027 fp16 no-bias 多层双向等）全部通过——3 个二进制确实覆盖了 6 个组合。

## 设计决策

- 判别顺序选"先读 binary_info_config.json 再谈重编"：改 `ASCENDC_TPL_SEL` 写法 + 清缓存重编一轮要数分钟且大概率不改变产物结构；读一个 JSON 是秒级、且直接给出"哪个二进制覆盖哪些键"的证据。
- 把"行为验证（全组合用例通过）"作为最终判据：产物数量的规律属于构建系统内部行为，文档化程度低；用例通过才是用户可依赖的结论。

## 预防

- 在算子仓的联调 checklist 里，"TilingKey 覆盖核对"应以 binary_info_config.json 的键列表 + 全组合用例通过为准，不以 `.o` 文件计数为准。
- 若确实怀疑某组合缺实例，最小复现是只跑该组合的用例并观察是否命中"找不到 kernel/行为错"类报错，而不是反复改 TPL 声明。

## 验证信号

来源说明：以下为该 GRU 算子迭代三编译与联调过程中的实际命令输出（构建产物 JSON 均为 tool 读取原文）。

- `.o` 数量为 3（多次重编、清缓存后不变），少于 6 个声明组合（来源平台：950；证据等级：tool_observed）。
- binary_info_config.json 三个二进制分别绑定全部输入 dtypeId=0/1/27，每个含 4 条 `d×p` 键（来源平台：950；证据等级：tool_observed）。
- 6 组合全量 ST 通过：L0 5/5 + L1 28/28 = 33/33（含 fp16/bf16 × bias/no-bias 全部组合），UT 51/51（来源平台：950；证据等级：tool_observed）。

## 失效边界

- simplifiedKey 中 `d=` / `p=` 维度的确切语义（本例实测不随 dtype/bias 拆分二进制）未在官方文档中核实，不同算子/代次可能不同；判别时请以"逐输入 dtypeId 列表 + 实际用例通过"为准，不要照抄 d/p 含义。
- 本卡不解决"TilingKey 手算值与 ASCENDC_TPL bit 布局不一致"类问题（那会选错实例），该问题在本算子的历史 issue（tilingkey encoding mismatch）中另行记录，属另一根因。

**未决项**：

- 构建系统"何时拆分二进制、何时合并实例"的官方规则未找到文档依据；建议后续在固定版本 CANN 上用最小样例（单模板参数 × 多取值）实验归纳。
## 坏实践

- 见到 `.o` 数量 < 模板组合数就反复改 `ASCENDC_TPL_SEL` 写法并全量重编——本例实测两轮重编产物数量不变，纯属时间浪费。
- 把低精度用例失败先归因到"编译缺实例"，跳过对失败输出（MERE/MARE/不匹配比例）的机理分析。

## 误诊

- 主会话曾根据 `binary/config/ascend950/bench_no_harness_gru.json` 的 binList 只有 4 条 simplifiedKey（d=0/1 × p=0/1）推断"bf16 (d=2) 没编译、fp16 可能映射到 fp32 二进制"——混淆了该文件（按输入 dtype 聚合的静态描述）与 binary_info_config.json（逐二进制键映射）；实际 fp16/bf16 失败根因是 kernel 内跨流水线同步缺失（见同批 V→MTE2 卡）。
