---
type: Guide
title: 算子 Python 包装的形状、路径与进程参数
description: 提取除数合法性、整除维度、环境根与列表传参，纠正仅 exists 就算路径安全的示例。
tags:
- task.implementation
- topic.implementation.operator_registration
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/python-secure.md
  role: primary
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# 算子 Python 包装的形状、路径与进程参数

Python 包装与 Meta 的 shape 推导必须产生合法整数维度。来源 q.shape[2]/ori_kv.shape[2] 除数可能为零，且 / 返回浮点；若语义要求分组整除，应先确认分母为正且分子可整除，再使用 //。只把 / 换成 // 会静默向下截断非法组合；合法零工作量按接口另行处理。短路 and 可保护后续取模，但需要实际控制流确实先验证除数。

环境安装根和外部文件路径要验证非空、规范化结果与实际用途。exists 只证明当前有路径，不证明它是目录、所需文件可读、包含正确工具链或落在某个允许根。若产品要求路径限制，还应考虑 ..、符号链接和检查后使用的变化；不应因示例只用 exists 就宣称已处理路径遍历。工具链路径、输出目录和用户数据路径各有自己的合法合同。

subprocess 以 argv 列表和 shell=False 调用可避免 shell 对输入中的空格、分号、$() 等作二次解释。每个参数仍须符合目标程序语法，外部值可能被目标程序解释为选项，必要时使用其支持的 --；列表本身不证明程序选择和路径合法。记录实际退出码与超时/缺产物，不能忽略编译或测试子进程失败。

异常应足以定位 shape、dtype 和属性合同，但若实际输入含口令/密钥，不应原样拼进错误消息。普通维度不是默认敏感数据。版权头、年份、审查路由与禁止 shell=True 的统一检查政策继续留在 Skill，本文仅保留机制和适用条件。
