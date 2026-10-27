---
type: Guide
title: MoE 样例 Golden 与 FP16 造数边界
description: MoE 样例 Golden 与 FP16 造数边界的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/scripts/verify_dispatch.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/test/test_moe_dispatch.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/run_multi_data_test.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/README.md
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例 Golden 与 FP16 造数边界

golden读取实际dump的FP16输入和expertIds，构造route事件，按localExpert、srcRank、linearIdx排序，再同时精确比较expandX、expandIdx、expertTokenNums、epRecvCounts的完整容量。纯dispatch无算术时精确值/顺序比较合适；np.array_equal对NaN即使同位置也不默认相等，不能把NaN拷贝判定与数学误差混用。零尾要求依赖Host初始化，不能反推kernel清尾。

验证器topK由每rank expertIds文件长度//BS推断，却最后只保留一个topK，未检查所有rank相同；BS0会除零，非整倍数通常reshape报错，需显式错误信息。expert_id只拒绝dst_rank>=rankSize，没有拒绝负值，Python负索引会路由到末rank；需0<=id<EP*localE。容量BS*EP*min(localE,K)还依赖同token不重复expert，与Host当前K1一致但不是任意dump的保证。

Host FloatToFP16直接截mantissa，不做标准round-to-nearest-even，exponent>=31统一输出Inf从而丢NaN身份；低指数靠右移生成subnormal。当前生成rank*100+token*10+col的小整数在可表示域内可精确，放大形状后会暴露截断/溢出。golden以实际编码输入为准能验证搬运，却不能验证该Host编码器是否符合期望浮点转换；二者须独立测试。

缺失文件/错长度有明确检查，但默认三组均匀route不覆盖非法输入、极端热点、每source不同count、零接收rank、大EP、window复用。核对四输出相互约束：sum(epRecvCounts)=expertTokenNums=有效expand行数，triple回指实际输入与路由且每有效route一次。通过脚本只能证明所运行用例；本次仅源码审视，没有设备运行证据。
