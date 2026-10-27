---
type: Guide
title: Sage 游标 Split-KV 与归并边界
description: 保存任务首步、完整状态槽与因果位置合同。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 游标 Split-KV 与归并边界

work g对应task=g//SK和shard=g%SK，jStart=floor(jEndF*shard/SK)、jEnd=floor(jEndF*(shard+1)/SK)。区间连续且大小差≤1，但jEndF<SK会有空片；来源“无空片”与后文空片紧凑化矛盾。Host cap splitK到全局kvBlocks仍不能消除causal早期query只见少量块的空片。totalTasks0也不能先算ceil(aicNum/totalTasks)。

Cursor Next先保存jCur再j++，返回后使用jCur；isFirst为jCur==jStart，不能统一jCur==0。首步状态重新初始化与全局step槽轮转不同：step s读(s-1)%N的同一work旧状态，写s%N，新work首步不可读取前一query的残留。Q借用L1仅在同一work真实Q不变且最后消费者结束后可轮转。

每步mNew=max(mOld,max(S))，alpha=exp(mOld-mNew)，l=alpha*l+sum(exp(S-mNew))，oacc=alpha*oacc+P*V；max、sum、alpha保留到对应延迟RS消费者。分区写未归一化oacc及m/l，归并mStar=max(mk)、wk=exp(mk-mStar)、lStar=sum(wk*lk)、O=sum(wk*ok)/lStar。MulDstAdd不能凭名字当acc+=a*b；按目标API确认其乘的是dst还是两个source。

pml紧凑化需要保存两AIV所有有效m/l；来源整slot512float含sub1偏移256，copy128只搬sub0会丢半行。可以copy完整512或分别搬两段有效值，后者须保持正确同步/stride。k==nValid时可免copy但仍要计数，不能continue跳过计数。若po没同步紧凑化，必须保留有效原shard编号，不能拿移动后的m/l配错po。

右下causal候选key≤query+(sk-sq)，属于该接口而非普遍默认。块上界由最后有效query行和kvBlocks夹紧，负上界、Sq>Sk及空可见行要处理，不能依C++负整数截断自动得到数学floor。阈值lim-1与列索引比较方向不能反；padding后的零K会产生score0，必须显式mask使其不进入分母。

分区全空、整行全mask及非有限score不能无条件计算-inf-(-inf)；输出与LSE按接口特殊值策略。GQA head映射需H可被Hkv整除且正确映到各batch的KV头，不能只凭8:1等已测比例宣称通用支持。
