---
type: Guide
title: 广播 Mask 极性、步长与暂存复用
description: 纠正BA/AB方向及Select极性，保留Softmax布局。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_12_broadcast_mask.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/data_filter/select.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/data_filter/get_select_max_min_tmp_size.md
  role: constraint
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-11T05:55:38Z'
platforms: []
---

# 广播 Mask 极性、步长与暂存复用

AdaGrad DAG Vec::Duplicate<U>把lr标量广播，再Cast<T,U>后Mul，是统一向量图的候选；源文字称Duplicate始终FP32，与U可能half/BF16矛盾。Muls本身已经是向量乘标量，不能无测量判低效；显式广播另耗UB/指令，不能假设总被访存隐藏。Scalar读取位置与类型、生命周期、lr广播合同均须一致。

MaskedScatter两类折叠以实际公式定义：BA令xInner=maskElems、xOuter=xElems/maskElems，maskIdx=i%xInner，即每outer重复一份inner mask；AB令xOuter=maskElems、xInner=xElems/maskElems，maskIdx=row，即每outer mask广播到该inner所有元素。来源文字“前/后维广播”容易与字母倒置，应以逐维shape合法性与以上寻址核验，不从总元素整除推定广播成立。maskElems0、xElems0、交错多轴广播和不可折叠模式须另处理。

来源P12前后对SelectWithBytesMask极性矛盾。已读取的仓内Select接口卡说明mask值0取src0，非0取src1，非按bit；因此示例(src, MASK_VALUE, mask)是非0置MASK_VALUE，不能在梯度段反解释为true保留src。此为仓内固定版本材料约束，当前SDK具体旧名/重载还需对应；如果业务true表示可见，须转换极性或换源次序。

布尔选择置0与乘0不同，Inf*0/NaN*0仍可能NaN；有限数乘1/0本身不必存在额外精度损失。Softmax有限-10000 sentinel与任意加性float mask不是通用等价，极端有效score、scale及全mask仍须定义；mask1B相比4B只节省mask载荷75%，不代表整个算子存储/耗时同幅下降。OPTIONAL mask缺失应走无mask分支，不访问nullptr。

Softmax batch/channel广播用bit0/bit1并将相应stride置0，前提是该mask轴为1或与目标一致；仅batch!=maskBatch不能接受任意不同长度。MaskOffset的batch/channel/lineOffset是坐标，而GetOffset中的realBatch/realChannel/realLine实际要是相应物理stride，不能误传轴长度。NextChannel取模channelNum须非零，wrap推进batch并清line；同batch内也可能跨多个channel，不能只判断起止batch相同便当单channel。

来源Select shapeInfo有firstAxis、srcLastAxis、maskLastAxis，mask物理宽可大于source；有效行数、padLineNum和真实width需分别传。仓内接口卡还说明isReuseMask=true且尾轴不同可能修改mask，后续重复使用时应禁改或重新加载；不能把mask视为只读却给可修改重载。

SoftMax Host用实际shape与FP32字节宽查询scratch并填tiling，Kernel shape含真实lines、物理padLineNum与有效width。主块tiling能否用于尾行按接口参数化约束，不能默认所有尾数都可复用。Select与Softmax临时区若顺序不重叠，可分配max(两者实际需求)共享；只查询Softmax大小不证明足够Select，另核对GetSelectMaxMinTmpSize及isReuseMask参数。共享区不能同时存放后续仍需的mask数据。

API优化、高阶与手写的数值正确性分别核对，不能因高阶名字宣称精度必优或固定10–20%加速；本批未直接读取归档raw文档或在设备调用该接口。
