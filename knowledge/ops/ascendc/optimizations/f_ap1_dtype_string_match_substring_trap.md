---
type: Optimization
title: "dtype string-match substring trap"
description: "cpp // \"bfloat16\".find(\"float16\") == 1 → matches! if (dtype.find(\"float16\") != npos) { ... } // BUG else if (dtype.find(\"bfloat16\") != npos) { ... } // never executes Fix: the bfloat16 check must come"
tags: [task.debugging, topic.precision]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-ap1-dtype-string-match-substring-trap.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# dtype string-match substring trap

> 原 OKF v1 卡号：**F-AP1**。

## 反模式

```cpp
// "bfloat16".find("float16") == 1 → matches!
if (dtype.find("float16") != npos) { ... }      // BUG
else if (dtype.find("bfloat16") != npos) { ... } // never executes
```

**Fix**: the `bfloat16` check must come before `float16`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-AP1，convert_patterns_to_okf.py）。confidence 未升格。 -->
