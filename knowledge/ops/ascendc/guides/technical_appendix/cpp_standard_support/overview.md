---
type: Guide
title: 技术附录概述
description: Host侧与clang15一致，支持完整的C/C++标准。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/technical_appendix/cpp_standard_support/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 概述<a name="ZH-CN_TOPIC_0000002509891888"></a>

Host侧与clang15一致，支持完整的C/C++标准。

Device侧，默认支持C++11标准，支持指定C++14、C++17、C++20。由于硬件限制，部分C++运行时能力无法支持。

