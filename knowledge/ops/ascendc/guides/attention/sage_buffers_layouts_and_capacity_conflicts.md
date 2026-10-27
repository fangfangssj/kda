---
type: Guide
title: Sage 缓冲布局与容量冲突检查
description: 保存GM和片上地址合同，揭示源布局表中的重叠与扩容缺口。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-rules.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 缓冲布局与容量冲突检查

量化Qhat为BSND[B*sqPad,H*D] INT8，Khat为[B*kvBlocks*128,Hkv*D] INT8；V为[B*skPad,Hkv*D] FP16，O写回HND。Q加载GQA下GM行间跨其他head，来源srcStride=(H-1)*D对INT8可同时按元素/字节得到同数，换dtype需重新检查API单位。K索引同时含batch、KV head和逻辑块，不能只换head数量。

SQ/SK以[blocks,8]FP32复制，索引必须block*8；mu为[B*Hkv*kvBlocks,D*8]FP32，来源读取每块首D值、写8份向量，不能擅自解释成每个元素交错复制8次。复制是该实现的物理ABI，不是量化数学要求。po[totalWork,128,128]FP32保留D有效列，pml[totalWork,512]FP32含sub0在0的m64/l64和sub1在256的m64/l64及padding。

来源dav-3510示例资源为L0A/B各64KiB、L0C256KiB、L1512KiB、每AIV UB248KiB，核数例28AIC/56AIV但需实际查询；这些值未在本批设备核验。FAIQK静态Shape读(M,N,K)，FAIPV读(M,K,N)，需按真实版本确认。QK INT8 <128,128,256>每A/B槽32KiB，与PV HALF <128,128,128>每槽32KiB匹配；运行D128不等于静态K也必须128。共用从0起的地址与事件时，槽覆盖范围和生命周期必须兼容，不同槽距并非所有独立设计都禁止。

来源L0C QK/PV各两份64KiB共256KiB，已占满示例容量；cursor顺序分配不自动防溢出。L1四份P各32KiB共128KiB，加Q/K各两份32KiB及V两份32KiB，总320KiB，另计实际组件其他区。

UB表S两份共64KiB，O两份64KiB，oacc32KiB，O16 16KiB，P两份32KiB，累计208KiB。QF从208KiB开始却列INT8 8KiB+half8KiB+FP32 16KiB=32KiB，跨度到240KiB；mu又从232KiB开始，状态233KiB、scratch236KiB，按表同时存活会重叠。来源声称合计约240KiB需由实际chunk分配/合法别名证明，不能直接照表当无冲突布局。

加深S/O从2到4仅这两者就额外128KiB，若P也加深再加32KiB为160KiB；原240KiB即使删32KiB QF也不能放进248KiB。KV tile加到256虽然可将QK单份L0C128KiB+PV128KiB装入256KiB，却也扩大S/P和量化块合同，不能只核算L0C就断言可实现。Fixpipe DEQF16省流量候选还会改变int32 score的数值范围与scale路径。

D64时来源PV向UB写紧密[64,64]，RS不能固定行距128；po GM则仍可能以128为物理行距。Cast与输出长度按有效SH*D，合法1D连续写可避免某2D配置错误，但不是所有128B blockLen搬运都非法。
