---
type: Guide
title: Sage 量化与 K 平滑补偿合同
description: 保存INT8 QK与FP16 PV路径，修正精确性和类型转换推断。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 量化与 K 平滑补偿合同

来源v1为HND输入Q[B,H,Sq,D]、K/V[B,Hkv,Sk,D]，FP16输出；每128有效K行求mu_j，令Ks=K-mu，sK=127/max(amax|Ks|,amax|mu|)，Khat=round(Ks*sK)转INT8；Q块用sQ=127/amax|Q|和Qhat=round(Q*sQ)。K尾块均值的分母是nValid而非128，padding不应污染均值/scale；树归约破坏原K后需从保存的half重建再减mu。

重建score为S_int*sm/(sQ*sK_j)+comp_j，S_int=Qhat*Khat^T，comp_j=(Qhat*mu_j)*sm/sQ。sK只在主项中，补偿里已消去；若写Q而非Qhat，则应使用Q的实际近似值而不是额外再除sQ。source recipe一处dot(Q,mu)*(sm/sQ)与其余定义不一致，不能同时采用。mu是FP32，Qhat*mu的dot不是整数dot，不能据结果小于2M就称FP32精确或与另一增广GEMM逐位相同。

对有限有效score按online max/sum更新，P在[0,1]后降FP16，由FP16 Cube PV产生FP32未归一化输出，最终除sum再转FP16。量化与P降精度都有误差，不是原始全精度attention的代数完全等价。all-zero块的amax0需约定scale和量化零；来源helper把除数0换1是一种实现选择，NaN/Inf与饱和/舍入仍需显式合同。

来源两级Cast K/Q经FP32→FP16→INT8，RINT时点及饱和必须与目标支持一致，不能凭函数名断言无双重舍入。bf16位模式不能当half读取；先数值转half可修正位解释，但会引入新的舍入和FP16范围限制，不等于完整保留BF16接口值域。生成BF16用FP32高16位截断也不同于最近偶数转换，生成器需明确实际舍入。

quantScheme=1的FP8 MX PV在来源未完成。MmadFAIPVMx可能要求L1已是指定FP8类型及P/V scale，不能把half缓存交给DataCopy期待隐式量化；FP8格式、scale粒度、布局及转换路径需由实际组件确认。LSE、varlen、NHD和D96是来源v1未补齐的能力，22个历史case通过不能覆盖这些缺口。
