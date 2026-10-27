---
type: Guide
title: Sage 分层取证与证据边界
description: 保留拟合和中间量探针，避免余弦相似度与症状唯一归因。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-rules.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 分层取证与证据边界

可分原始输入独立FP32 golden、表达量化/补偿/online/split语义的emulator、设备实现三层比较。但cosine0.9999不证明设计普遍正确，kernel与emu差也未必都是kernel bug，emu和dump仍可能错。若emu直接使用Kernel A产出的Qhat/Khat/mu/scale，它只能局部检查Kernel B，必须另用原始输入独立验证量化产物。

余弦不检测纯正倍缩放：O和2O的cosine为1；还需目标合同要求的绝对/相对误差、特殊值、shape和必要逐阶段统计。来源22case、cos0.999592–0.999909、128²约31us/1k²约73us/4k causal约505us/200k约719ms为历史声称，未在本批重跑；不同文段还给不同区间，不能合并为新的实测结论。只抽样max-planes防golden OOM时应明示未检查平面范围。

比较Qhat*Khat整数S、comp逐行、m/l/alpha每step、P*V及最终merge可缩小故障位置。dScale拟合轮流代入各块scale，merge最小二乘拟合po列组合，可发现候选错序/错系数；矩阵秩不足、多个假设相同或误差可拟合时不能唯一归因。cols0–63、rows32–63或随块数恶化是线索，不是某个L0/子核/scale bug的证明。

来源512KiB qdbg将S保留区、P(+128KiB)、O(+144/+336KiB)、状态400–416KiB、comp416–424KiB、AIC进度250KiB、AIV452–460KiB、merge464–496KiB用于不同探针。多个范围可能复用或重叠，P还有双sub竞写；只能在明确相位、slot、step、写者及范围内解读，不能当无冲突通用dump布局。长序列step快照必须限制容量或轮转并存step身份。

相位alive/smDone/rsWait/rsDone和迭代号可定位最后进展，AIC/AIV分别记录；Scalar GM写及小DataCopyPad不落盘在来源为特定环境观测，“32B必失败/≥1KiB必可靠/AIV标量store全失败”没有普遍依据。扩大拷贝可能改变时序，仍需合法长度、双方同步、Host回读完成；不能把8倍复制当API全局要求。

srun返回码、输出存在性与mtime能排除许多旧文件假通过，但秒级mtime、并发同目录和旧binary仍可漏判。失败/timeout产物可留作取证，不能当成功运行精度证据；每次运行以唯一身份、制品和完整输出对应更可靠。npu-smi Health正常不证明所有执行状态无异常，“挂两次换卡”是来源运行策略，不迁移为知识规则。

进程总时长需拆setDevice、内核、同步、reset/finalize。来源1.8s/4.6s初始化销毁与约4ms内核只说明那次测量，持久进程能摊薄而非保证全消失。header未重编、JSON true被atoll读0、批量replace未命中或括号破坏也可导致旧制品/错误配置，应以真实编译和输入解析证据区别设备故障。
