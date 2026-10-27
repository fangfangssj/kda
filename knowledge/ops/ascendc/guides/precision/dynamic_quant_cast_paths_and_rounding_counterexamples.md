---
type: Guide
title: DynamicQuant 转换路径与舍入反例
description: 保留 A2/CANN8.5 来源边界，修正双重舍入、最大值与结构对齐误读。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_dynamic_quant_kb.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_quantization_patterns.md
  role: supporting
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# DynamicQuant 转换路径与舍入反例

来源A2/CANN8.5.0例定义每行scale=max(max(abs(x))/127,1e-10)，q=clamp(round(x/scale),-128,127)，scale FP32且shape为去最后一维。epsilon下界、负/零scale处理、zero-point及量化轴属于合同，不能称所有npu_dynamic_quant均严格用此公式，内部是否使用未公开quant单元也未由本批证据建立。

来源Cast能力表声称FP32→int32/int64/int16支持RINT/FLOOR/CEIL/ROUND/TRUNC，FP32→half支持NONE/ODD，FP32→BF16及BF16→int32列五种舍入，half→int16只RINT、half→int8列FLOOR/CEIL/ROUND/TRUNC/NONE；FP32→int8、int32→int8/int16在该表缺失。另一量化文件又给half→int8用RINT，二者冲突，必须按目标版本真实重载核验，不能跨平台发布“永不支持”。

RINT为nearest-even，FLOOR/CEIL为向负/正无穷，TRUNC向零，ROUND在所述整型转换中为nearest-away；NONE不能一概叫截断或所有方向等于RINT。ODD的round-to-odd通常是在丢弃位非零时令保留尾位为1，不等于“最近邻奇数”；精确可表示值不应强制变奇。异常值、subnormal/flush和NaN payload按真实指令定义。

FP32→half→int8与FP32先RINT到int16并不数值等价。half在100附近ULP=0.0625，会先把100.51舍入为100.5，再nearest-even成100；FP32直接nearest-even成101。若先在FP32域完成整数舍入并clamp至[-128,127]，这些整数在int16/half中可精确表示，后续可用受支持链保持值；不能把“最接近硬件”当未测依据。int16输出再to(int8)会增加中间空间和设备转换，8192×16384为256MiB INT16；Python调用.to不表示CPU执行，且超范围窄化不自动饱和。

Div与乘以舍入后的倒数可在半整数边界产生不同结果，差异不限固定1–2ULP。FP16/BF16有限值数学上可无损嵌入FP32，但flush/异常表示需核对。有限值的Abs+Max不会因低精度比较丢失1ULP，FP32的优势主要在后续除法/算术；不能用来源该错误理由强制所有Max先升精度。

每行超UB可先分chunk求全行max，再用同一scale量化各chunk；如果跨核分行内N，必须额外合并并发布统计。reshape为[M,N]需保留stride，contiguous可能实际复制；rank0/N0、元素乘积和空scale输出需定义。usedCoreNum=min(AIVCount,ceil(M/blockM))与GetSubBlockNum一起使用时要明确launch block是AIC组还是AIV，避免多乘/少乘子核；blockM不能整分时需尾行归属。

来源DynamicQuantTiling含6个int32成员共24字节，pad0注释“32B对齐”不成立；需要真实sizeof/alignof和序列化容量，不能靠字段名认为已对齐。FP16/BF16分发、INT8或INT16临时输出与FP32 scale需匹配Host/Device ABI。纯Vector算子不必使用MIX_AIC_1_2，混合模式的参与者/核数按真实执行模型。

小scale写回可用32B对齐UB配DataCopyPad写4有效字节；填满8个float是安全初始化方式之一，不表示必须写8个到GM。GlobalTensor::SetValue不是一律不能用，需按Scalar cache、刷新和设备完成合同，不能断言Host永远读不到。Cast从buffer起始处对齐不保证任意offset后仍对齐；TQue depth0引用式Alloc/DeQue与depth≥1返回值形式是来源重载差异。

来源验证器给INT8/INT16最大绝对差≤1、INT32/INT64/bool torch.equal，FP16 MERE/MARE 9.77e-4/9.77e-3、BF16 7.81e-3/7.81e-2、FP32 1.22e-4/1.22e-3。保留为该验证器配置，不替代用户量化合同；MERE/MARE公式未提供，不应猜测。INT8/16差值先升INT32避免窄类型溢出；空tensor的max和NaN/Inf浮点处理另需定义。
