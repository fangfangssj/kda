---
type: Guide
title: 白盒采样与覆盖证据的边界
description: 保存元组覆盖和采样复现，分清清单一致性与实际命中。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-d/04-step4-output.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-d/05-step5-gen-cases.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/04-test-design-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/design-verifier/00-execution-order.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/00-execution-order.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/00-execution-order.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒采样与覆盖证据的边界

声明覆盖可按(group,dtype,path,key)元组对账，每个声明可达路径与 key 都应有候选用例；ID、整数 key、已知 group 和 primary dtype 的结构校验有助于发现漏项。但把 path/key 填入 meta 不会让输入自动走该路，需用最终 shape/attrs/const_inputs 重新求谓词，并以实际 Tiling/Kernel 观察确认运行覆盖。多个报告都由同一错误模型生成也不是独立证据。

固定随机 seed 提供复现，不保证边界、模数条件或维度交互覆盖。每 entry 约100条、count5、min(10,100//a)等是预算选择，a=0需专门处理；有限域要求唯一值时不能超出状态数无限采样。模数/对齐条件可由合法离散集合构造，不能指望5个任意整数自动对齐。

用最短候选池长度截断所有 pool 会丢较长池的值；重复轮转相同 seed 或同索引配对也未必增加 pairwise 覆盖。group/path 同名变量通过字典合并时可能互相覆盖，应保留来源并验证一致性。全字段去重包含 path/key 元数据时，输入相同但标记不同仍算两条，统计应分开逻辑声明与独立执行输入。

JSON 可解析、脚本正常结束、输出文件齐全或 unresolved=0都不保证输入模型完整。旧 verifier/LLM 报告在源码、模型或用例改变后可能失效，证据需绑定内容版本；不能为保留人工内容而复用过期结论。自动生成的表格可以确定性重建，但推导公式及分组仍须技术校验。

测试覆盖还包含异常输入、特殊数值、输出别名、动态/可选组合及跨核资源边界；以路径为主的 shape 采样不是全部正确性测试。阶段先后、部分文档 marker、脚本分工及回退重跑规则属于 Skill 实施流程，抽取的是上述可复用覆盖原则。
