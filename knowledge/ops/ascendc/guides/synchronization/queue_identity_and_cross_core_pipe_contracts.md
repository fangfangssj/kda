---
type: Guide
title: 队列身份与跨核 Pipe 合同
description: 修正同步指南中的示意释放、固定 Pipe 和事件 ID 泛化。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_dynamic_quant_kb.md
  role: supporting
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# 队列身份与跨核 Pipe 合同

PipeBarrier处理所选管线内顺序，HardEvent表达同核跨pipe依赖，CrossCore表达相应核组的事件，TQue负责其生产消费生命周期。四者不能按“都等待”相互替换。PIPE_ALL不能无证据当全设备GM可见性fence，PIPE_S是否支持应查目标API，来源“显式调用必硬件错误”未绑定所有版本。

HardEvent源_目的如MTE2_V、MTE2_MTE1、V_MTE3、MTE3_MTE2，应按实际搬运/计算方向配对同类型同ID，并在真实生产操作后Set、消费前Wait。FetchEventID可避免与框架分配冲突，但静态保留且完整管理的固定ID并非语言禁止；事件池上限、释放和复用仍需核对。来源MTE2 barrier例中间插入compute，下一次覆盖还需等待V消费，MTE2自身barrier不能解决这一依赖。

来源总览Alloc的是inLocal，却FreeTensor(fp32Local)；除非fp32Local确为同一合法分配句柄且API允许该视图，这可能释放错误对象。队列槽位的分配者、En/DeQue句柄、最后消费者和Free必须一致，不能因LocalTensor类型相同就互换。调用形式如引用式Alloc/DeQue或返回值式需对应真实depth/重载，VECIN/VECOUT并非只能depth0，A1/B1/A2/B2/CO1也不是永远depth1。

来源mode2的CV1:2模型为AIC一次Set唤醒对应两个AIV，而两个AIV各Set后AIC才能完成Wait。少一个AIV、AIC重复Set或分支提前return均可能破坏token计数；该表不覆盖950 mode4等其他硬件模式。其“误用1”代码注释称AIV Set，实际写Wait，不能据注释计算完整协议。

AIC Fixpipe写GM后发布ready需对应FIX完成，AIV MTE2读取后release槽可对应MTE2完成；如果AIV生产结果供AIC读，ready则应约束MTE3写出而非固定MTE2。AIC消费者可能是MTE2或其他pipe，不是所有AIC Wait都必须FIX。泛型WorkspaceQueue应表达真正生产/消费pipe及slot身份，不能只按核型写死。

CrossCoreWaitFlag可用哪些模板实参/默认pipe由真实重载决定；漏写pipe不是所有版本必死锁，显式写了错误pipe也不正确。Set/Wait的模式与事件参与者需对应，生产pipe和消费pipe本来可不同。空任务与尾块即使不做算术也可能仍需完成协议；L1→L0和L0C→GM示例同样需真实数据就绪、累加及释放依赖。
