---
type: Guide
title: Counter Mask 的作用域与尾部计数
description: 区分 Normal、Counter 与 Compare/Select 重载的实际覆盖，保留 A2/A3 与 950 的文档边界。
tags:
- task.implementation
- topic.implementation.masking
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P67_vector_counter_mode.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P67_vector_counter_mode.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P67.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_count.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_norm.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/vector_compute/vector_counter_mode.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/SIMD_compute/mask.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/compare_and_select/Compare.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/compare_and_select/Select.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-10-17T19:15:04Z'
platforms: []
---

# Counter Mask 的作用域与尾部计数

Normal模式用repeat次数和每repeat的mask描述有效计算，Counter模式直接描述总元素数，由支持该模式的Vector重载推断迭代和尾部。来源希望省去主尾块的Scalar分支；对于half的256字节repeat，15000元素为117个满repeat加24个尾元素，而float满repeat为64元素，不能混用宽度。

显式SetMaskCount和SetVectorMask须与isSetMask=false等实际允许外部mask控制的重载配合；若API默认自行设置mask，外部写入可能被覆盖。Counter的计数、mask参数形状、repeat占位值及stride依所用API规定，不能把所有调用的repeat统一改成1，或假设所有接口均支持2^32-1元素。Counter不会扩充UB容量，也不自动把超大输入分配到多个tile。

模式选择和mask值是两份状态。ResetMask恢复默认mask值不等于SetMaskNorm恢复Normal模式；手动Counter区域结束后，按后续代码合同恢复模式和值。可以在有明确状态约定的helper间使用，不要求物理上必须同一函数，但异常分支、提前返回、其他helper修改与后续普通Vector均需覆盖。每条操作使用的有效count必须明确，不能默认前一条调用后的状态仍适合当前dtype/长度。

来源P67把一组归约API统一列为不支持Counter，P68又在Counter中调用WholeReduceSum；这些相互冲突的列表不能充当版本支持矩阵，应逐个重载核对。Cast链允许有显式转换的混合类型，但count语义、源目的容量及精度要相应处理。

Normal尾mask也有边界：`(1<<tail)-1`可能发生整型宽度或移位溢出，half尾部超过64位时不能只生成一个64位bitmask。连续mask和逐bitmask不能混为一谈。静态长度可能已被编译器折叠，小任务模式切换成本也可能抵消节省；来源约20条Scalar及0–10%性能变化未在本批复现。验证0长度、整repeat、跨64/128边界、多helper混用和Counter后首个Normal计算。

## 固定文档中的模式和每次迭代范围

以下增补依据 `asc-devkit@639b2b5c1031188107d23d785d52b935f1b4f13b` 的 Mask、Compare、Select 文档，属于该快照的接口规则整理；没有新增设备试验，也不认证历史 Conv1D、DequantSwigluQuant 或 Compare/Select 修复成功。消费时仍须核对实际 SDK、产品和解析到的重载。

Mask 文档把 Normal 模式的连续 mask 定义为单次 repeat 内参与计算的元素数：16 位操作数为 `[0,128]`，32 位为 `[0,64]`，64 位为 `[0,32]`。该文档说明，超过单次 repeat 上限仍按一次 repeat 的最大元素数执行，不能用 `mask=128` 表示一个 FP32 repeat 归约 128 项。Counter 模式的 mask 则表示整个矢量计算的元素数。这些模式定义不能绕过某个接口自己的保留参数、能力或覆盖约束；legacy `WholeReduceSum` 是否采用此模式、是否支持 Counter，仍按目标版本的实际声明确认，不能拿新版 `ReduceRepeat` 的名字代替核验。

## Compare 的平台与重载分别核对

下表只比较固定 Compare 文档列出的 Atlas A2/A3 训练、推理系列和 Ascend 950PR/950DT；“A2/A3”不包含名称相近但另列的 Atlas 200I/500 A2 推理产品。Compare 的 `isSetMask` 在该文档中是保留参数，应保持默认；不能把其他矢量接口的 `isSetMask=false` 用法直接套到 Compare。

| Compare 接口或参数 | A2/A3 训练、推理系列 | Ascend 950PR/950DT |
|---|---|---|
| tensor 前 n 个数据的 `count` 重载 | `count*sizeof(T)` 要求 256B 对齐；未对齐部分不参与计算，仅完整对齐块有效 | 此页没有列出相同的 256B count 限制；不能因此推断任意 count、dtype 或地址组合都合法 |
| 高维切分重载的 `mask` / `mask[]` | 保留参数，设置无效；不能用短 mask 宣称尾部已被过滤 | 设置有效；仍按该重载及 Mask 文档解释 |
| 整个 tensor 参与计算的运算符重载 | `src0`、`src1` 按该页要求 256B 对齐 | `src0`、`src1` 按该页要求 256B 对齐 |
| `count==0` 或 `repeatTime==0` | 文档定义为 NOP，不写目的操作数 | 经 VF 调用 Reg 矢量 API 实现兼容，不保证是 NOP |

地址对齐和计算覆盖是两份条件：Compare 的参数表还要求 `src0/src1/dst` 的 LocalTensor 起始地址 32B 对齐；整个 tensor 运算符的 256B 源地址要求不能替换为这个一般参数要求。dtype 和比较模式也要单独核对，例如该页在 A2/A3 对 `int32_t` 只列 `CMPMODE::EQ` 支持，不能从 half/float 支持六种模式推导整数 NE 可用。

仅作长度核算：64 个 FP32 为 256B，65 个为 260B；128 个 half 为 256B，129 个为 258B。因此在上表的 A2/A3 count 重载下，不能认为 `count=65` 或 `129` 已生成全部有效比较位。若选择扩大计算范围，源、目的和 mask 的真实容量及填充值必须覆盖扩大后的范围，下游仍只消费逻辑有效数据；不能仅改 count 或把尾部 mask 当成已经生效。这里没有执行这些边界用例。

## Select 的选择掩码与计算掩码

Select 文档区分选择用的 `selMask` 与高维切分计算用的 `mask`：前者决定取 `src0` 还是 `src1`；后者在支持的重载中再次过滤写入，无效 lane 保留原 `dst`。二者的位数、容量和作用不能混用。

| `SELMODE` | 单次 repeat 使用的选择位 | 多 repeat 的消费方式 |
|---|---|---|
| `VSEL_CMPMASK_SPR`，模式 0 | `256/sizeof(T)` 位有效 | 多轮复用同一组有效选择位 |
| `VSEL_TENSOR_SCALAR_MODE`，模式 1 | 每轮 `256/sizeof(T)` 位 | 连续消耗 `selMask` 的不同部分 |
| `VSEL_TENSOR_TENSOR_MODE`，模式 2 | 每轮 `256/sizeof(T)` 位 | 连续消耗 `selMask` 的不同部分 |

模式 0 不能仅因传入较大的 count 就解释为自动消费一份多 repeat mask。模式 1/2 也必须实际分配并生成每轮会读取的选择位，计算有效 count 不会自动初始化整份 mask。该页对 950 的 `count==0` 或 `repeatTime==0` 同样不保证 NOP；不能把 Compare 在 A2/A3 的零参数规则当作所有 Select 重载的规则。

本文补的是平台、重载和覆盖的文档合同。[arch35 Select mask 复用案例](../../runbooks/precision/arch35_select_vsel_cmpmask_spr_mask_reuse_count_over_64.md)另有其目标设备和版本证据；二者不能相互替代。审查时分别检查单 repeat 边界、跨 repeat、短尾、零长度、连续调用和 Counter 后的首个 Normal 操作，结果与 mask 原始位模式应一起保留。
