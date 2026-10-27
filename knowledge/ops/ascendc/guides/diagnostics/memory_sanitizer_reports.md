---
type: Guide
title: Ascend C 内存检测报告的判读与覆盖边界
description: 根据 msSanitizer 报告中的地址空间、访问跨度、核号和调用栈定位问题，区分非法访问、多核重叠与 Host 分配生命周期。
tags:
- task.debugging
- topic.memory
- topic.implementation.debugging
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-crash-debug/references/memcheck/mssanitizer_guide.md
  role: primary
created_at: '2026-09-10T12:38:25Z'
updated_at: '2026-09-10T12:38:25Z'
platforms: []
---

# Ascend C 内存检测报告的判读与覆盖边界

msSanitizer 的 memcheck 可以提供内存异常的具体访问证据。判读时应共同记录异常类别、读写方向、字节跨度、首地址、GM/UB/L1/L0 地址空间、Device/Block、PC 和可用调用栈，而不是只保留最后一行 ERROR。

| 报告类别 | 应核对的对象 |
|---|---|
| Illegal read/write | 访问区间是否落在有效分配及合法片上容量内 |
| Multi-core overwrite / 对应重叠告警 | 不同核的 GM 区间是否重叠、是否有写者、交接是否具有正确同步 |
| Misaligned access | 实际地址与该方向、dtype、重载的对齐要求 |
| Illegal free | 地址的分配来源、是否重复释放或并非原始分配起点 |
| Leak / unused allocation | 分配与释放生命周期，以及实际数据是否使用了预期缓冲 |

同一条访问的 size、地址和上下文应一起分析。对齐要求不能统一写成 32 字节；多核重叠报告也应结合合法共享协议判断，不能单凭出现多个访问者就认定业务错误。

## 启用与观测条件

来源以 MindStudio 80.RC1 为版本范围，给出的入口为 `mssanitizer --tool=memcheck <application>`；泄漏与未使用内存检测分别通过 `--leak-check=yes`、`--check-unused-memory=yes` 显式启用。命令和产品支持应以安装版本为准。

含调试信息的构建有助于得到源文件调用栈。框架通过 GM 内存池分配 tensor 时，工具未必自动获知每个子分配的真实边界，来源要求按工具接口上报分配/释放；否则“未发现问题”不能视为每个 tensor 访问均合法。

## 覆盖与验证

来源版本不支持对 ATB 算子仓进行该内存检测；当前知识库的功能调试说明还区分 SIMD 与 SIMT 工具支持。不能将 memcheck 的覆盖范围推广到所有编程模式、版本或通信路径。

对同一失败输入核对报告与实际分配、搬运参数和核间分工；修复后在相同检测选项下复跑，并确认原异常位置消失。无报告只表示这次受支持且已启用的检测未发现异常，不证明数值精度或所有时序均正确。参见 [NPU 域功能调试](../programming_guide/debug_and_tuning/functional_debug/npu_board_debug.md)。
