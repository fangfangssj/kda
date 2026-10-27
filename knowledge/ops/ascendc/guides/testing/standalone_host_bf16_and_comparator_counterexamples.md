---
type: Guide
title: 独立 Host 的 BF16 编码与比较器反例
description: 保留最近偶数舍入方法，识别 NaN 漏检及异常资源窗口。
tags:
- task.testing
- topic.precision
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-to-direct-invoke/references/host-driver-template.md
  role: primary
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# 独立 Host 的 BF16 编码与比较器反例

来源FloatToBf16Bits通过memcpy获取float32位模式，加0x7FFF+高16位最低位再右移16，对有限IEEE binary32常见值实现向最近偶数的BF16舍入；反向把uint16左移16后memcpy回float。此法依赖真实32位IEEE表示，不能把主机struct名字当自动ABI证明。NaN必须单独保留为NaN：如0x7F800001按该公式会成为0x7F80（Inf），负NaN加法也可能回绕；特殊值和payload合同不可省略。

SampleBFloat16仅保存uint16编码，SampleHalf使用来源SDK half，Host/Device类型与launch实例应分别核对。golden统一float再cast只适合接口允许的路径，FP64、精确整数、特殊归约累加或量化不应自动降为FP32。示例float1e-4、half/BF16 5e-2绝对容差只是占位，需来自算子验收合同。

CompareBuffer先查长度，再以fabs(actual-expected)>atol计错；NaN使这个比较为false，std::max(0,NaN)还可保留0，因此新增NaN可能以0 mismatch通过，Inf-Inf也产生NaN，需显式分类合法/非法非有限值。相同元素数不同shape未被检测，±0符号、整数低位及相对误差也不在本比较器合同内。空向量0 mismatch只能说明没有元素可比。

ACL init/setdevice/createstream逐步取得资源，任一步失败都应释放已成功阶段；来源main在Init失败直接返回，没有统一回收。malloc示例位于try之前，后续分配失败可泄漏前面的buffer；成功清理中CheckAcl若抛出又会跳过后续free。应以资源所有权保证部分构造/异常路径，且未完成设备访问前不可释放。

来源RunOneCase核心计算、malloc、H2D、launch、D2H与比较全部注释，剩余代码会尝试free null；这个骨架不能声称最小可运行算子或精度通过。补齐调用时需检查真实返回、stream完成、workspace大小和输出新鲜性，再让比较失败影响进程退出。确定性输入有利复现，但不能代替边界、不同dtype及非有限值覆盖。
