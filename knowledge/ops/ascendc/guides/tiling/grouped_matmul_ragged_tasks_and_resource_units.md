---
type: Guide
title: GMM 不规则分组任务与资源单位
description: 保留分组、量化、缓存和调度模型，修正任务数及预算公式。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/matmul/ascendc-api-gmm-tiling.md
  role: primary
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# GMM 不规则分组任务与资源单位

来源groupType=0分M、1分K、-1不固定以及groupListType=0/2等属于其算子/版本语义，不是通用GMM枚举。各组Mi/Ki/Ni需有对应描述；Ni不同而输出仍称单一[M,N]是不完整合同，需说明变长拼接/stride。累计groupList应非递减、边界合法且最终与输入分段一致，允许空组与否按接口。GMMArray的128项和mList[0]=-1哨兵是来源结构设计，不是实际shape，更不是所有版本组数上限。

来源字段表达groupNum/coreNum/groupType、activeType/hasBias/quantParam、singleN及singleX/Y/Weight、groupListType、m/k/n数组、ubBaseK/N/CalSize/RestBytes、vBaseM、workspaceSize、withOffset、isOutputDisableL2Cache、isPreTiling、quantGroupNum。迁移时应保存单位、每组/全局语义和消费方式，不能只按同名字段推断ABI。

真实块任务数为Σ_i ceil(Mi/baseM)*ceil(Ni/baseN)，可用组前缀任务数反解(group,mTile,nTile)并跳过空组。maxM/maxN网格乘G会包括不存在的块，若保留此规则网格必须屏蔽这些任务。分组共用一个Matmul对象不是普遍限制；usedCore=min(aicNum,mDim*nDim)忽略组数只在相应顺序组调度下成立，组间独立并行需计所有有效任务。

对角遍历与蛇形不是同一映射。来源同时要求mDim*nDim=C及min(mDim,nDim)≥C，C>1时不能共同成立；另一处末尾nDim≥cores也与前面条件不一致。应以完整覆盖、唯一性及真实复用验证候选。singleN选择中curTask/alignUp(curTask,cores)若为整数运算，非整除时为0，不可能表达0.95利用率；需浮点/有理比较并处理curTask=0。增大N块可能减少任务而降低并行。

baseN=128/256/512等和A8W8调整范围是候选，不是跨版本固定值。baseK可从L0B预算除以buffer份数、baseN及B字节数再按允许粒度向下对齐；baseM同时受L0A/(baseK*A字节)、L0C/(baseN*acc字节)约束，需计各自buffer与保留区。int4不能用整数sizeof=0.5，需位宽/打包步长。任一向下对齐结果为0都不能继续除法。

L1先保留256KiB的例子要求实际可用量足够，否则无符号减法下溢；A/B dtype不同时不能对称平分元素容量，量化scale/offset和其他常驻区也占容量。stepK须正，depth=2*stepK、dbL0C=1等编码需按对应库解释。UB总字节除以块数仍是字节，不能直接叫元素数；ubIoBlkNum所占输入输出区不能遗漏。把ubCalSize/ubBaseK再AlignUp为ubBaseN可能超过原预算，需回代验证。

量化workspace=4*baseM*baseN*coreNum*sizeof(int32)隐含四份同时存活累加块；必须与真实流水份数对应。反量化workspace只累计ΣKi*Ni*weightBytes不能覆盖同时存在的x、scale或其他staging，生命周期决定是否可复用。静态Tiling例128/256/128仍需要合法shape和尾部处理；FullLoadK/N/A仅在相应L1容量、格式和复用条件成立，K=2048/7168、N=7168/4096、G=4等特化不能推广。

AIV 1:1/2:1以及token=128、GELU等选择是来源启发，K≤1024或K≥2048与1024<K≤2048在2048重叠，不能直接生成唯一分支。硬件执行比例需真实内核类型与同步支持，改一个字段不足以改变硬件模式。输出跳过L2需按真实工作集、后继复用和缓存模式测量。
