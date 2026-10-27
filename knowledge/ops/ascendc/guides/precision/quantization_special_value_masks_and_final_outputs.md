---
type: Guide
title: 量化特殊值掩码与最终输出
description: 纠正ZEROING、NaN修正位置和Scale倒数的泛化。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_05_softmax_stability.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# 量化特殊值掩码与最终输出

动态量化invScale=127/amax、输出scale=1/invScale在实数域等于amax/127，浮点上两次倒数可能改变舍入。amax0使invScale为Inf，0*Inf仍可能NaN；应按合同定义全零块scale和q。过小/过大amax、Inf/NaN同样需保护，不以乘法替代除法名义承诺数值更稳。

minScale是scale下限，invScale相应上限；它能限制倒数过大，却不能解决异常大值压缩正常小值的问题，增大scale反而可能让更多小值量化为0。必须区分原始amax与已除目标上限的scale单位；来源两者混称时不能直接用同阈值。MicroAPI MaskMergeMode::ZEROING仅描述被mask关闭的lane，不能推出有效lane除数0自动得0；Compare与Select的输入、掩码方向及执行顺序才决定最终除零策略。

FakeQuant源先Compare(x,x,EQ)识别NaN，将中间量Select成0，之后又减zeroPoint再乘scale；因此NaN最终为-zeroPoint*scale，只有zeroPoint0等条件下才是输出0。若合同要求实数0，可在最终输出选择0或将中间量替为对应zeroPoint。必须同时定义cacheMask及NaN传播，不可把“抑制NaN便于训练”当通用语义。

MX位处理须区分exponent全1且fraction0的Inf与fraction非0的NaN；exponent0还包括subnormal，不能一律叫零。来源0x7f81/0x7f810000只是其指定NaN位模式，不能保证所有CPU/GPU或所有FP4/FP8编码行为一致。FP4指数clip与scale特殊码的边界由确切输出格式决定，不能从一段未给完整常量的Compare/Select证明符合所有IEEE行为。

FP32转所谓FP19片段遇exponent0只留sign，实际会把subnormal冲成有符号零；保留NaN高尾数时若所有保留位为0，可能误变Inf，需确保NaN编码仍非零并按合同处理payload。普通round进位还需指数联动，不能仅增加mantissa。MAX_INT9在该片段充当指数上界的意义必须核对，不据名字推断位宽。

BatchNorm默认epsilon1e-5、LayerNormV3 1e-5/V1 1e-7是来源版本值；用户提供0、负数、过小值仍需按合同接受/拒绝，epsilon并不必然防所有除零。RMS的rms==0取factor1是一种输出约定，只检查精确0不能避免很小非零倒数溢出或负根NaN；判断通常每统计row一次而非每元素一次。

ClippedSwiGLU在exp之前clip可限制指数输入，但范围为|gluAlpha|*gluLimit；limit7不能在任意alpha下保证不溢出。前置clip(x)与后置clip(输出)不是等价变换，且大负x时exp(-x)是上溢风险，原文称下溢方向相反。只按真实算子clip位置执行。

DeepNorm把float参数位拷贝到uint32 tiling字段，再按位读回，可保持位模式；普通合法float字段本身也可保持同样位精度，不能声称整数容器天然更精确。长度、端序、字段ABI、memcpy状态和合法bit_cast/memcpy读取方式需配套，reinterpret指针别名不是所有C++模式下通用保证。
