---
type: "Runbook"
title: TTK 取值域仅整张量语义，单元素NaN/Inf注入用例壳不物化
description: TTK CSV input_data_ranges 只有整张量 (lo,hi) 值域，单元素特殊值用例只落 shape/dtype 壳；精确注入须走 manual_input_binaries 或 C++ ST。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["agnostic"]
---
# TTK 取值域仅整张量语义，单元素NaN/Inf注入用例壳不物化


## 现象

为对标 torch GRU Cell 反向算子（aten 逐字序五链梯度）的 AscendC 多分支算子验收 Branch-0（base 分支）白盒用例（TTK kernel 模式）时，设计文档 Validation.md 特殊值矩阵明确要求三条「单元素」注入用例：grad_hy 单元素 NaN（对应 hidden 列五路输出全 NaN、位置逐位对齐断言 produces_nan）、storage 单元素 NaN（NaN 沿触及该平面元素的链路按 IEEE 传播）、grad_hy 单元素 +Inf（门梯度链按 IEEE 传播 ±Inf/NaN，如 (hx−n)=0 处 Inf·0=NaN，不饱和不截断）。

生成产物 design_doc_cases.json 里有三条对应 case（_note 分别标 `S2-gradhy-single-nan` / `S3-storage-single-nan` / `S4-gradhy-single-posinf`），但逐条打印其 `inputs` 构造后发现：这三条的输入与标准随机用例完全相同——`grad_hy` 为 `value_range [-1,1]`、`storage` 为 `value_range [-1,1]`，没有任何单元素注入信息；导出的 TTK CSV 中这三行的 `manual_input_binaries` / `manual_golden_binaries` 列也全为空。即执行时这三条用例就是普通均匀随机输入，与正向标准用例（E01，同 shape/dtype/attr）无任何区别——「单元素 NaN/Inf 传播」测试点名义有 case、实际零覆盖；而结构覆盖度审计（按 valclass 签名统计）仍然通过。

生成侧在 design_doc_cases.json 的 `gaps` 首条如实登记了该限制（「input_data_ranges 为整张量值域，无法表达『单元素特殊值 + 其余有限』……由下游支持精确 tensor 注入的框架（aclnn ST 工程 / TTK manual_input_binaries）补充」），验收据此判定为「如实登记的框架限制」而非隐瞒缩档。

## 适用场景

- 工程形态：白盒/黑盒用例生成与验收（TTK kernel / aclnn 模式 CSV）、覆盖度审计中对特殊值测试点的解读。
- API 路径：TTK 用例 CSV 的 `input_data_ranges` / `manual_input_binaries` / `manual_golden_binaries` 列；TTK 输入生成链（`core_modules/npu/op/input_generation.py` → `RandomData`）。
- 验证过的平台与 CANN 版本：宿主机 TTK 仓源码与用例产物级检视，未上板；输入生成在宿主机完成、结论按定义与 NPU 型号无关（frontmatter `platforms: [agnostic]`）。CANN 版本未涉及。
- 适用粒度（transfer_scope）：platform_wide（TTK 用例格式语义，跨算子）。
- 路线优先级：recommended_fastpath——单元素/精确特殊值注入首选路由到 C++ aclnn ST 工程（host 侧输入构造有完整表达力）；CSV 内替代通道为 manual_input_binaries 手工 bin。
- 明确不适用 / 未验证场景：整张量可表达的取值形态（[0,0] 全零、常数、[lo,hi] 随机）不受本卡影响；manual_input_binaries 手工 bin 路径在本场景未实际生成并跑通（未验证）；e2e 模式（torch API 用例）取值通道同族限制但未逐一验证；TTK 版本差异未查证。

## 根因

TTK 用例 CSV 的取值通道只有整张量 (lo, hi) 值域语义（`input_data_ranges` 每输入一个 (min,max) 对），没有逐元素/单点注入表达；「单元素特殊值 + 其余有限」只能经 `manual_input_binaries` 手工输入 bin 或 C++ ST 工程代码内构造承载。

## 判别方法

（以下 probe 为该验收会话实际执行的命令，可复用）

1. 逐条对照 case 的场景标注（_note/purpose）与实际 inputs 构造：

```bash
python3 -c "
import json
d = json.load(open('operators/ThnnFusedGruCellGrad_package/tests/whitebox/branch-0/design_doc_cases.json'))
for cid in ('case00045','case00046','case00047','case00048','case00050'):
    c = [x for x in d['cases'] if x['id']==cid][0]
    print('====', cid, c['_note'])
    print('inputs:', json.dumps(c.get('inputs'), indent=1, ensure_ascii=False))
    print('expected:', json.dumps(c.get('expected'), ensure_ascii=False)[:500])
    print()
"
```

原始观测（verbatim，节选）——`S2-gradhy-single-nan` 的 inputs 与普通随机用例无异：

```
==== case00046 S2-gradhy-single-nan
inputs: {
 "grad_hy": {
  "shape": [
   4,
   8
  ],
  "dtype": "float32",
  "value_range": [
   -1,
   1
  ]
 },
 "storage": {
  "shape": [
   4,
   40
  ],
  "dtype": "float32",
  "value_range": [
   -1,
   1
  ]
 }
}
```

对照组（整张量可表达的形态真实落地）：S1（grad_hy 全零）case00045 的 grad_hy `value_range [0,0]`；S5（storage 全零）case00049 的 storage `value_range [0,0]`；S6（fp16 溢出边界）case00050 的 `value_range [60000,60000]`。

2. 核对 CSV 手工注入列是否承载：TTK kernel 模式 CSV 表头 26 列含 `manual_input_binaries`、`manual_golden_binaries`；S2/S3/S4 三行两列均为空（行尾 `,True,group=,,0,,,`）。

3. 确认 TTK 输入生成链的取值来源：`/workspace/ops-test-kit/ttk/core_modules/npu/op/input_generation.py` 中存在手工输入文件读取通路（日志 `"Reading manual input file: {fp} ..."`）；无手工文件时按 CSV 的 `input_data_ranges` 逐输入构造 `RandomData(dtype, shape, data_range)` 均匀随机生成——即取值只来自「整张量值域均匀采样」或「手工 bin」两条路，没有第三条。

## 解决配方

单元素/精确特殊值注入不要指望 `input_data_ranges`：

1. **首选（recommended_fastpath）**：路由到 C++ aclnn ST 工程——host 侧输入构造有完整表达力，可代码内对任意位置写 NaN/Inf 并做位置逐位对齐断言（本例中该工程已有 REF_SAMPLE 模式保留：原样例的 host 输入构造 + CPU golden 逐元素比对能力待恢复/扩展）；TTK 层保留 shape/dtype/attr 落位与可物化取值形态的用例。
2. **CSV 内替代**：生成手工输入 bin 文件并填 `manual_input_binaries` 列（TTK 官方列，输入生成链有读取通路），golden 侧配 `manual_golden_binaries` 或插件执行期现算。
3. **无法物化时**：在产物 `gaps` / known_limitations 如实登记「未物化 + 下游承接载体」，不得以 case 壳冒充覆盖；审计/验收侧把「特殊值标注 + 普通 range + 空 manual 列」判为未物化壳，单独跟踪。

## 设计决策

- 特殊值覆盖按「可物化性」分层承载：整张量可表达形态（全零/常数大值/值域随机）留在 TTK range 用例；单元素/位置断言类交 C++ ST——不采用「删除 S2/S3/S4 case」（shape/dtype/attr 落位信息仍有价值），也不采用「假装覆盖」（壳 + 不登记）。
- 本卡路线属 recommended_fastpath（分层承载 + 如实登记）。

## 系统信息

- TTK（ops-test-kit）仓：宿主机 `/workspace/ops-test-kit`（环境中另有多份 checkout：`/workspace/ttk-dev/ttk` 等，检视取 `/workspace/ops-test-kit`）；关键文件 `ttk/utilities/data.py`、`ttk/core_modules/npu/op/input_generation.py`。
- 白盒统一入口：`ascendc-whitebox-design` skill `references/whitebox_designer.py`。
- 用例产物：`operators/ThnnFusedGruCellGrad_package/tests/whitebox/branch-0/{design_doc_cases.json, ttk_design_doc_cases.csv, ttk_design_doc_cases_oversized.csv, WORKFLOW_PROVENANCE.json}`。
- 下游注入载体：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（当前 1 条接口桩冒烟，REF_SAMPLE 端到端数值测试整段注释保留待恢复）。
- 黑盒同族产物：`tests/blackbox/testcases/aclnnThnnFusedGruCellGrad_l{0,1,2}_*.csv`（同为 input_data_ranges 格式）。

## 完整代码

判别方法中的 probe 脚本即完整可执行诊断。补充黑盒侧同族确认命令（该验收会话实际执行）：

```bash
head -3 operators/ThnnFusedGruCellGrad_package/tests/blackbox/testcases/aclnnThnnFusedGruCellGrad_l0_functional.csv | cut -c1-600; echo ====; grep -c "^" operators/ThnnFusedGruCellGrad_package/tests/blackbox/testcases/*.csv
```

观测：黑盒 CSV 表头同为 `input_data_ranges`（(min,max) 对）格式，无逐元素注入通道（l0 45 行 / l1 516 行 / l2 50 行）。

## 预防

- 用例生成后、执行前，逐条核对「场景标注（note/purpose/sig 的 valclass）vs inputs 实际构造 vs CSV manual 列」：特殊值标注 + 普通 range + 空 manual 列 = 未物化壳。
- 特殊值测试点在 TTK 层只承认可物化形态；单元素/位置断言显式分流到有表达力的载体并登记承接。

## 验证信号

本卡证据来自一次白盒用例验收会话（对 Branch-0 白盒用例生成产物做逐测试点核对与独立审计时的源码/产物级检视，未上板）。

- case00046/47/48（_note 标 single-nan/single-posinf）的 inputs 构造与标准随机用例完全相同（value_range [-1,1]、无任何注入信息）（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）
- 对照组真实落地：case00045（S1 全零）grad_hy value_range [0,0]、case00050（S6）value_range [60000,60000]（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）
- TTK kernel 模式 CSV 表头 26 列含 manual_input_binaries / manual_golden_binaries，S2/S3/S4 行均为空（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）
- TTK input_generation.py 存在手工输入文件读取通路（"Reading manual input file"），无手工文件时按 data_range 经 RandomData 均匀随机生成（来源平台：agnostic（宿主机 TTK 仓源码检视）；证据等级：tool_observed）
- 生成侧 gaps 首条登记「input_data_ranges 为整张量值域，无法表达『单元素特殊值 + 其余有限』」并指明下游承接载体（来源平台：agnostic（宿主机产物检视）；证据等级：tool_observed）

## 失效边界

- 整张量可表达的取值形态（[0,0] 全零、常数、[lo,hi] 随机）不受本卡影响，可正常经 range 物化。
- manual_input_binaries 手工 bin 路径未实际生成并跑通（含 golden 侧如何取 bin 内值现算比对）——标注为未验证。
- e2e 模式（torch API 用例）取值通道同族限制，未逐一验证；TTK 版本差异未查证（本结论基于当前 /workspace/ops-test-kit 源码）。
- 卡内 case 编号/shape 为现象复现示例，非可搬运清单。

## 坏实践

- 只看覆盖度审计的 valclass 签名通过就认定特殊值已覆盖——签名只证明「case 存在」，不证明「测试意图可达」。
- 生成「note 标特殊值、inputs 是普通随机」的壳用例充数且不登记 gap。

## 误诊

- 「TTK 支持 (nan,nan) range，所以 NaN 场景已覆盖」——整张量 range 即便物化也是全 NaN，验证强度远弱于单元素（全 NaN 输入→全 NaN 输出掩盖传播路径正确性；单元素才验证 NaN 不污染无关位置且精确到达指定列）；且 (nan,nan) 是否物化还受采样器数字化行为影响（见同域另一卡 [TTK 采样将 nan→0、inf→±max](ttk_random_data_digitizes_inf_nan_ranges.md)）。该判断为审计推导（agent 推导），非实测。


