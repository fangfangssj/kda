---
type: Guide
title: Kirin Ascend C Simulator 与 mobile-station 发行包范围
description: 保存 Kirin 端侧开发材料中 Simulator 与 mobile-station CANN 的配套关系，区分服务器环境检查与目标模拟器可用性。
tags:
- task.debugging
- topic.compilation
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-env-check/references/kirin_platform_guide.md
  role: primary
created_at: '2026-09-11T01:20:00Z'
updated_at: '2026-09-11T01:20:00Z'
platforms: []
---

# Kirin Ascend C Simulator 与 mobile-station 发行包范围

固定版本的 Kirin 开发材料将 KirinX90、Kirin9030 等列为端侧 AI 处理器，并以 Simulator 作为其所述主要开发方式。该材料指出，Kirin 的模拟器需要相应的 mobile-station CANN 发行包，所用 CANN 与服务器开发发行包有差异。

因此，服务器版 CANN 的编译器、头文件或通用环境检查通过，不能单独证明 Kirin Simulator 已安装或能够运行目标程序。需要核对目标产品、实际发行包及其中提供的模拟器支持，分别记录编译和模拟运行结果。模拟器存在也不证明所有设备特性、性能计时或服务器 API 均可直接复用。

这是一条来源明确但版本范围尚不完整的环境知识：原材料未给出 mobile-station 的具体版本、完整产品支持表或当前产品到 NPU 架构的映射。本卡保持 draft 和未定位平台范围，不从“Kirin”前缀推定所有型号具有相同支持，也不据此限制未来或其他已核实的开发方式。源文指向的 Harmony 推理工程和安装指南是进一步核验入口，未作为本卡已经阅读或验证的上游证据。
