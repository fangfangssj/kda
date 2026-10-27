---
type: Guide
title: Blaze Launcher 的空输入与物理输入证据
description: 补齐逻辑输入到设备字节、零工作量、输出清单及实际执行产物的一致性。
tags:
- task.testing
- topic.memory
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/launcher/launcher-development.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-cmake-build-guide.md
  role: supporting
created_at: '2026-09-11T02:57:41Z'
updated_at: '2026-09-11T02:57:41Z'
platforms: []
---

# Blaze Launcher 的空输入与物理输入证据

Launcher的逻辑参数、物理buffer、Tiling/Params、GM实参、workspace与最终消费者必须描述同一对象。转置/预打包/paired view后，不仅改变地址，还要同步Pattern、shape、stride/lda、storage span、gate offset及Copy消费者。输出final/partial、optional/null和broadcast范围也需明确，不能仅按Kernel名称补出参数顺序。

合法零工作量需先验证shape/dtype、group endpoint、文件字节数与输出合同，再选择不启动设备的空输出路径或已证明支持零维的实现。空Tensor与R>0的全零值行不同；不启动Kernel不能成为跳过metadata校验的理由。K=0时可能仍有非空输出及bias/identity语义，不能一概返回空。

物理转换的验证可用独立于Launcher转换代码的实现或可手算模式，比对最终H2D的shape、长度和实际字节；只让生成器与Launcher共享同一错误公式可能同时自洽。比较完整重绑定与“字节正确但消费者stride仍旧”的反例能暴露合同断裂。逻辑值、最终交付字节及Golden需绑定同一case，量化参考应基于真正交付的编码值。

输入/输出路径不应随调用者cwd或环境脚本变化。source脚本可能改变当前shell位置参数，业务argv需保持并核验；不同阶段实际收到的shape/group列表必须相同。多输出各自记录shape/dtype/文件身份，不能用一个成功输出代表全部结果，也不能用历史同名文件补齐。

Kernel启动返回码与独立验证结果是不同证据；超时、测试未执行、verifier失败和缺失原始输出不能汇总为PASS。跳过构建时必须能确定实际运行binary对应的源码、依赖及配置，多个build目录尤其需明确。符号lookup成功只证明名字可解析，不证明typed启动ABI、POD布局和设备指令路径。

资源释放覆盖正常及异常路径，最终输出观测需要对应stream完成；输入允许别名时按语义证明安全，不能照搬来源一律“不重叠”限制排除合法in-place。来源建议拒绝非有限输入也仅在算子合同要求时适用。本批提取知识约束，报告字段、阶段回退、固定目录和执行清单继续留在Skill。
