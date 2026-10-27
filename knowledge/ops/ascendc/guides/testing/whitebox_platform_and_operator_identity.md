---
type: Guide
title: 白盒分析的平台与算子身份
description: 分开保存硬件身份、注册名、调用入口和可验证证据。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/S1-input-collection.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/workflow.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒分析的平台与算子身份

chip_model、SoC 产品名、arch 编译目标与运行核型是不同层次。npu-smi 的设备信息可作当前环境证据，ASCEND_SOC_VERSION 等环境变量只说明配置，不能单独证明实际设备。HBM 可见容量受 SKU、虚拟化和保留空间影响，不能仅凭112GB/其他数值唯一判定 PCIe 或整机型号。核数应说明 AIC/AIV、逻辑核或可用 launch 资源；UB 容量注明 byte、KB 或 KiB，不能混用。

算子规范名可来自 OP_ADD 注册的 op_type 及消费者实际 camel_to_snake 规则，目录名仅是源码位置。例如 lamb_next_m_v 目录与 lamb_next_mv 调用名可以不同；缩写和数字处理不能简单以全小写替代。op_host/op_kernel 两目录存在也不足以证明目标平台有实现，还需有效注册、构建选择与设备入口。

源码条件、用例生成清单、静态校验报告、编译日志、运行结果和数值比较分别回答不同问题。其证据要绑定同一算子、源码版本、设备、接口与用例，不可因流程文档最后写 completed 就推导全路径可运行或精度合格。来源的交互、阶段读取和文件调度约定留在 Skill。
