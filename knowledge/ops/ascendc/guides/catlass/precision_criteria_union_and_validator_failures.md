---
type: Guide
title: CATLASS 精度条件的并集与验证器漏判
description: 保存MERE MARE和误差比例的区别及模板反例。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/precision-verification.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/verify_result_template.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/rules.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# CATLASS 精度条件的并集与验证器漏判

MERE为平均相对误差，MARE为最大相对误差；来源使用abs(golden)+1e-7作分母。逐元素absDiff≤atol+rtol*abs(golden)再允许一定失败比例，是另一判据。两者不等价，取OR只会扩大接受集合；不能以“更鲁棒”之名自动放宽算子已定义的验收标准。

模板阈值FP16=2^-10、BF16=2^-7、FP32/FP64=2^-13、FP8 E4M3=2^-3、E5M2=2^-2和0.1%误差比例仅是该脚本配置，本批未核验为官方所有算子标准。未知dtype静默回退FP16会掩盖配置错误，应拒绝未知配置或明确经确认的映射。FP8 E4M3在代码映射float8_e4m3fn，具体格式变体也属于二进制合同。

允许0.1% outlier且不另设maxAbs上限时，1000个元素中1个任意大误差仍可能通过比例条件；模板虽然打印maxAbs却未据此拒绝。MERE很小也不能单独排除稀疏结构错误。接近0时相对误差可大，但绝对误差1e-5或1e-4不是所有值域都可称正常量化误差；需真实容差和dtype ULP。

模板仅提前检查output的NaN/Inf，未对golden检查。golden含NaN时absDiff>tol比较为false，Inf的Inf>Inf也为false，比例条件可能把非有限reference当0个错误而通过。必须先定义并执行双方特殊值策略，再做有限域比较；不能依赖最终OR来拒绝未定义算术。

函数在验证shape相同前flatten/运算，直接调用可能发生NumPy广播；分母按actual长度又可能与比较后元素数不一致。CLI reshape能发现一部分文件数量问题，不修复函数合同。空数组max会异常或比例除0，需显式空结果语义。

所有dtype先转float32会丢掉FP64的低位差异和大整数差异，例如1+2^-40与1、2^24+1与2^24；有限float64大数转换还可能变Inf，而预检查发生在转换之前。比较dtype应保留待检差异，避免整数减法溢出，并检查转换后域。binary dtype、shape、端序和制品对应关系须明确，不能只因旧文件可reshape就认定来自本次实现。

来源把精度失败优先解释成验证器过严缺乏依据。应同时检验验证器、独立golden及候选实现；修复判定缺陷与降低既定标准不同。脚本测试通过仅说明其当前接受逻辑，不证明设备实现正确。
