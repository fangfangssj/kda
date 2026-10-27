---
type: Guide
title: 模板状态、队列与寄存器 Lane 合同
description: 保存CRTP和混合类型状态布局，纠正同位宽视图与自动转换推断。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_01_mixed_precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_02_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: supporting
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 模板状态、队列与寄存器 Lane 合同

存储类型、计算类型、索引类型和step类型可以独立参数化，实例化合法组合后可消除对应运行时分支；Host选择仍有成本，最终是否只保留某TilingKey分支取决于构建机制。模板数量增加编译时间/二进制体积，不自动解决I-cache或获得最优性能；FP16与BF16相对FP32都是每元素存储减半，来源FP16只省25%没有相同基数依据。

ApplyAdamWV2Fp/B16/MixType分别组织同构FP32、低精度存储FP32计算、FP32 var/m/v配低精度grad等状态。T和U独立输入/输出队列便于按dtype预算，但不能自动消除对齐/Cast需求；max_exp_avg_sq实际类型须查注册与算法状态，不能按变量名max_grad_norm随意归入grad队列。每个队列槽、状态offset与合法别名均单独计byte。

Embedding MT/CT分离、cacheQue FP32累加、低精度放后半再Cast至前区是实现候选。MTE2→V事件须在真实DMA完成后再转换，Cast完成前不能释放/复用源queue；最后输出cast必须等全部相同索引贡献完成。宽精度不保证冲突安全，也不能推出Vector FP32吞吐永远优于half。

Foreach CRTP结构Base→Unary→具体Triangle/OneScalarBinary/Quaternary可复用列表遍历、分块、队列与调用adapter。bufferNum、paramsCount、needCopyOut属于具体实例合同，needCopyOut=false仍需合法原地/隐式输出发布；DeQue→Compute→Free与Alloc→EnQue必须按真实读写完成关系。函数指针非类型模板参数使op可在编译期绑定，但不保证所有编译器必内联或没有代码体积成本。

DynamicBlockQuant来源xCalType默认half、BF16特化float；DynamicMX另一模板half→float、BF16→BF16用于特定指数处理，两者正相反也可各自成立。不能泛化所有BF16运算保持原格式或都必须升精，更不能仅因std::conditional就声称MicroAPI自动Cast。xCalType默认half对新增FP32等类型危险，应限制支持域或补特化。

纯Gather/Transpose可能把BF16按half载体搬运，以保留16位bit而不做浮点运算；这不等于把BF16数值解释为FP16再计算。LastDim tiling中INT8 storage需求改按2B、INT32 index按8B可能是内部staging格式，不能连GM真实elementSize一起改；来源INT6_DSIZE应核对真实常量，且示例注册表未列INT8不能据helper分支宣称已可调用。

Pooling gradient的T1为grad、T2存储argmax、T3内部索引。半精度DataCopy→UnPack→Cast到FP32和LayerNorm DIST_UNPACK_B16/DIST_PACK_B32链需按lane布局/有效mask配套；UnPack不是数值升精。INT64加载RegTraitNumTwo后把reg[0]视成INT32不自动提取每个INT64低32位：必须核对加载分布、lane顺序、高位与范围，简单reinterpret可能交错高低字或只覆盖部分索引。

RMSGrad Cast2FloatIf<T>跳过float路径时，castLocal必须已经指向/持有float输入，不能因没有Cast就读取未初始化临时区。FakeQuant FP16路径同时分float/half/int32临时，FP32路径仍含int32，正文“只一buffer”不精确；不同dtype须按同时存活缓冲合计预算。
