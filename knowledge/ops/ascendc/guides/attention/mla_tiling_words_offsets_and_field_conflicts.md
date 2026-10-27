---
type: Guide
title: MLA Tiling Word 偏移与字段冲突
description: 保存400加17布局，纠正Float数组和绝对地址混淆。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-paged.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md
  role: supporting
created_at: '2026-09-11T05:34:07Z'
updated_at: '2026-09-11T05:34:07Z'
platforms: []
---

# MLA Tiling Word 偏移与字段冲突

MLA来源使用32位word表，通用分配400+17*batch个uint32，TP1按token重组时需其实际条目数。不是把所有字段数值转为FP32；整数长度/偏移按uint32读，tor仅将float位模式写同一word再用float视图读。gTilingFp64变量实际GlobalTensor<float>，名字不代表double。

recipe头区为0batch、1H、2Dc、3numBlocks、4blockSize、5maxBlocks、6tor、7KVheads、8headSize、9paraSize、10headSplitSize、11headSplitNum、12mask、13Dr、14maxKV、15split、16maxSplit、17maxQ、18totalTokens、19formerTasks、20tailTasks、21processNum。旧paged说明把processNum放19，与recipe21冲突，必须由真实Host writer和Device reader同一版本决定，不能混抄。

每batch段起点header[8]+header[9]*b，来源默认400+17*b：0qLen、1kvLen、2排序前batch号、3pageSize、4/5Q偏移高低字、6/7QRope、8/9mask、11/12l、13/14oFd、15KV每核段长、16该batch实际split数。第10槽未定义不能擅自赋义；40位以上偏移重建需先uint64(high)<<32再或low。

来源多处称这些数是GM绝对地址，但Host公式计算的是相对各张量的元素offset，后续GlobalTensor[offset]也再次加基址；应明确是元素偏移还是绝对byte地址，不能两者兼用。Q和O同shape可复用相同元素offset，却仍是不同base；不意味着把Q指针当O指针或允许原地别名。

l偏移累计H*qLen*globalMaxSplit，oFd基础offset累计H*qLen*Dc，使用时再乘globalMaxSplit；已乘split的量不得再乘。FP16输入、FP32partial及word表各自元素宽度不同。per-batch排序后必须保留原Q/QRope/O/页表对应关系，不能把sorted索引直接当原batch。

CUTASK前缀在word25起，声明最多130项则batch+1≤130，还要不与其他header字段重叠。前缀终值与processNum、实际每batch split总数一致，0代表另一解码路径时不得同时写非空split表却让Device走蛇形。分配前检查溢出和完整表长度。

tilingKey来源通用0/1、AMLA4/5、TP1候选7/8，H128分支及numTokens%blockDim≤10、batch≤40是特定策略；早文写%20是假设20核，不能搬到其他blockDim。Host须检查GetMLATilingParam状态，不能先用未填tiling尺寸分配工作区。
