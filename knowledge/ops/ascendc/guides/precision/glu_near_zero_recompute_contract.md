---
type: Guide
title: GLU 近零重算与局部近似的数值合同
description: 保留 FP16 输入逐 K 重算及 Swish 近零公式的严格适用域和同步变化。
tags:
- task.debugging
- topic.precision
- topic.compute.elementwise.math_function
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/glu-precision-diagnosis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-development.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# GLU 近零重算与局部近似的数值合同

GLU输出activation(act)*gate的误差既可能来自act也可能来自gate。接近零的正确结果会放大相对误差，不能仅由Full失败或一个max-relative异常推断根因；应保留实际raw C、两分支、绝对/相对误差、参考量级与K累加顺序。实际输入域与用于定位的有限probe范围不同，不能以probe通过缩窄算子支持域。

来源近零重算策略限于连续、非转置FP16 A/B，FP32累加及SwiGLU输出，参考按K递增逐项FP32累加，且已隔离布局/同步与Vector公式。普通MMAD和顺序FP32累加的加法顺序可能不同，低绝对误差仍可能违反某个严格近零相对标准；这不证明MMAD一般错误，也不能把参考累加顺序视为任意Matmul的默认合同。

refineAbsThreshold是独立的非负有限参数，0禁用。来源AIV在C就绪后判断act或gate是否命中，再从原始FP16输入按参考K顺序重算该位置两路并覆盖UB。需要正确的group/tile/分支地址、所有原始输入寿命与UB ownership；来源选择Kernel持有这些状态，Epilogue只消费局部值。开启策略的ND/DN限制来自实际特化，关闭后某些转置路径可用，不代表NZ/ZN也已支持。

首消费者变为Scalar时，需要覆盖到PIPE_S的C-ready依赖；Scalar写UB后还要S→V，使后续Vector读取新值。不能只保留原Vector等待而忽略新增Scalar访问。逐点原始GM读取增加工作量，需记录命中率、K、两路重算和任务时间，不能宣称不增加第二个Kernel就没有性能成本。

Swish(x)=x/(1+exp(-x))的近零式x/2+x²/4是截断近似，不是恒等式；其首个被省略项为-x⁴/48。nearZeroSwishThreshold与MMAD重算门限是两个独立条件，必须在V-known隔离出公式误差后按实际验收选择，不能为掩盖地址错任意扩大。gate幅值还会放大激活近似误差。

验证门限前/等于/后、act近零、gate近零、两者近零、正负输入、特殊值、不同K顺序及重复执行；保存C-direct、C-through-L0C2UB、V-known与Full各自证据。来源具体历史策略没有在本批设备复测，不默认开启、不改变Golden或容差。
