---
type: Guide
title: Blaze Pure Reduce 的 partial 状态与模板缺口
description: 从 Softmax 改写中保留归约单位元、跨核合并、UB 暂存与零长度边界。
tags:
- task.implementation
- topic.compute.reduction
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/reduce-adaptation-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-development.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# Blaze Pure Reduce 的 partial 状态与模板缺口

Matmul已完成全部K的logits按完整N行归约时，可先对各N分区求partial，再按同一逻辑行合并。max/min/sum分别以-inf/+inf/0作有限浮点模型的单位元；输出为[R,1]。整数dtype需要适用单位元，NaN、tie、空行及数值顺序须另定。浮点sum的分组顺序改变可能改变舍入，不能把实数结合律当逐位相等证明。

纯max/min/sum不需要Softmax的expWorkspace、mHistory和指数和重缩放，只需每状态分区每行的partial，FP32逻辑容量C*R*4字节另加对齐。mean需同时保留计数或正确权重，argmax需值/索引及tie规则，variance等也要相应可合并统计；不能泛称所有Reduce只换一个函数名即可。

V1寄存器逐lane累积完整向量后，尾部用有效mask合并，再做lane归约。curN必须先判断为正；来源vfN=ceil(curN/VL)、mainLoop=vfN-1在curN=0时下溢，uint16 rows/vfN也需在窄化前验证。尾部inactive lane保留单位元，负值max不能用0填充。Reg加载/归约/首元素store的签名与地址空间依实际版本。

来源简化UB表只列mmData和resultT，但伪代码又读取preResultAddr_与历史GM partial：仍需为历史值分配合法UB或明确等价驻留，并完成GM→UB、MTE2→V和前次partial写回到本次读取的依赖。只删Softmax缓冲而留下未分配preResult指针不是完整实现。resultT写GM后才能被下一阶段或下一次更新使用。

V2对自己负责的行读取所有producer partial并合并，UB容量受每批行数*C影响；必要时分批并保留完整行/核域。Reg Store通常写相应UB视图再由MTE3搬GM，来源伪代码直接StoreAlign(outputReduceAddr...)不能当合法GM store接口。删除rescale阶段后同步也应按保留的真实操作重新核验，不因与Softmax相似就一律沿用旧wait。

验证未分到N tile的分区、全部负值max/正值min、N=0合同、向量宽度两侧、多核和重复更新，并比对最终归约值而非只检查partial。来源torch.reduce是泛指，实际Golden需选择sum/amax/amin等对应算子及输出shape；不是一个可直接调用的通用函数。本批未编译伪代码。
