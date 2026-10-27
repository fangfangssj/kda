---
type: Guide
title: LocalTensor 视图与 IoU 分量寿命
description: 区分无搬运切片与非连续重排，保存示例中被覆盖的坐标与真正输出。
tags:
- task.implementation
- topic.compute.indexing
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P97_zero_copy_subtensor.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/data_structures/local_tensor/operator.md
  role: constraint
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# LocalTensor 视图与 IoU 分量寿命

LocalTensor下标以元素偏移返回共享原存储的新Tensor，创建视图本身不搬运。它适合已在UB中的连续子段，不能把任意非连续Gather或AoS→SoA重排变成零拷贝。后续Vector仍实际读取源并写目的；父Tensor释放或槽被覆写后旧视图不能继续使用。

有效操作范围需满足offset与count不越原分配及合法数据，尾部padding和API实际访问也须计入。来源GetLen/SubTensor<0,len>是未经本批核对的版本写法；不能把运行时len直接当编译期模板参数，或只依赖句柄推定可访问长度。重叠视图不是一律非法，但写读别名要符合接口和最后消费者约束。

批量IoU需分别计算xLeft=max(curX1,candX1)、xRight=min(curX2,candX2)、yTop=max(curY1,candY1)、yBottom=min(curY2,candY2)，再形成相应宽高、交集和并集。源计算完width后把w0与candX1用于yTop，已经丢失curY1和candY1，不能保留为正确模板。Max/Min/Mul的普通Tensor重载也不因实参名字像标量就支持广播。

输出面积规则必须与原算子一致，包括是否有坐标+1、无效/负面积、union=0、阈值大于或大于等于及NaN。来源最终结果在w4却从iouLocal读取，没有建立别名关系，需明确真正生产和消费的地址。Vector结果改由GetValue逐项比较前仍有V→S完成要求。

比较接口是否支持当前count、mask和布局应按重载判断，不能统一规定count*sizeof(float)必须256B整除；可用合法tail策略或Scalar候选，BATCH<=64也不保证其开销可忽略。批量大小受新增中间区与已常驻父Tensor共同约束，避免只按视图数估算空间。

验证非零bStart、尾批、坐标分量互不相同、parent复用与不同面积边界，先逐分量对照标量公式，再衡量省去Gather/Pad的实际流量与完整时间。来源2–5倍等未在本批复现。
