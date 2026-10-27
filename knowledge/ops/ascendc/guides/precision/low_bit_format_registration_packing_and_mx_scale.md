---
type: Guide
title: 低比特格式注册、打包与 MX Scale
description: 保存FP4/FP8/INT4实现约束，纠正存储收益和枚举互通。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_04_fp8_int4_conversion.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# 低比特格式注册、打包与 MX Scale

OpDef输入/输出dtype表通常描述按位置对应的合法组合，不能将两个列表自动作笛卡尔积。来源AddRMSDynamicQuant列FP16/BF16配INT8/E4M3FN/E5M2/HIFLOAT8，AscendQuant另列FP32及INT4，MX列E2M1/E1M2/FP8与E8M0 scale；这些是各算子/平台版本范围，省略号和if constexpr片段不证明完整8种实例均已实现。Host验证、tiling dispatch、Kernel实例及实际编译目标须一致。

INT4/FP4相对INT8元素载荷减半，FP8与INT8均8bit；source“FP4比INT8少75%、FP8少50%”错误。相对FP16才分别少75%和50%。总存储还含scale、padding及对齐，压缩比不自动等于总带宽收益或端到端加速。

两INT4共享一字节时，长度/地址除2仅对相应单位与偶数边界成立；typed int4b_t的下标/sizeof包装规则须按API确认，不能所有指针都再右移一次。奇数尾需ceil字节数、nibble顺序和另一半字节保留，或者像dynamicQuantUpdateScatter源合同一样拒绝最后维奇数。两个核写同一字节不同nibble仍可能冲突。

源DataCopyExtParams把rowLen*sizeof(int4b_t)先cast uint16再右移，较大值可能在除2前已截断，应在宽域算真实byte长度并检查该字段/API允许范围。DataCopy与DataCopyPad长度单位、rowStride、dstOffset和打包产物的物理布局一致，避免只减blockLen却未修寻址。

FP8/FP4是浮点编码，不能用static_cast<uint8_t>(scaledVal)替代。MicroAPI CastTrait的RegLayout/SAT/ZEROING/rounding分别影响lane布局、范围、无效lane和舍入；HIFLOAT8 round/hybrid与E4M3FN/E5M2仅rint是该实现Host允许子集，不代表所有FP8格式标准强制统一枚举。dynamicBlockQuant用RINT1/ROUND4/HYBRID7，dynamicMX用ROUND0/FLOOR1/CEIL2/TRUNC3/RINT4/HYBRID5，两套值不得混传；Tiling key中rounding位与Kernel模板同表。

来源CastTrait lambda只给RMode1或4的return，其他实例缺分支不能称完整支持hybrid。饱和边界、NaN/Inf、subnormal和round中点应按确切格式验证，不能以金融/训练/推理用途标签替代定义。

GroupedMX源每32元素共用一个E8M0 scale，先取最大指数，减目标最大指数得到共享scale指数，再构造倒数指数用于FP32乘法。偏置/移位必须匹配BF16或FP32位格式，unsigned减法下溢及零/极值需单独处理；不能把截取指数公式或省略的特殊分支称完整精确标准实现。每块1B scale仍有块尾和布局开销。

FP16/BF16→FP32、乘逆scale、Interleave、Cast到低比特、Pack链中，Interleave/Pack需要证明输出lane顺序与逻辑元素一致，特别是奇数尾与mask。源__NPU_ARCH__3101临时写FLOAT_OVERFLOW_MODE_CTRL仅适用该目标；读旧值后必须按生命周期恢复，局部示例未展示恢复不代表可以污染调用者状态。特殊NaN码与寄存器设置不能统一宣称CPU/GPU结果相同。
