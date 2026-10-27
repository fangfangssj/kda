---
type: Guide
title: asys 健康状态与硬件检测结果的不同含义
description: 区分 Healthy、Warning、Pass、Warn、Fail 和 HBM 新增 ECC 计数，保留设备选择及终端显示截断边界。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/info-and-health.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/diagnose-and-config.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/collect-and-launch.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- asys health
- health_result.txt
- Pass(n)
- HBM ECC
---

# asys 健康状态与硬件检测结果的不同含义

asys 的健康状态、检测是否成功和业务能否运行是不同问题。本卡面向 Ascend C 排障，保存固定 reference 的输出语义，不将简化状态表升级为业务准入规则；实际版本/型号可能改变字段，本卡未执行健康或硬件检测。

## info 与 health 的设备范围

`asys info -r=status|software|hardware` 的 -r 必填；status 给指定 Device，-d 只在 status 有效，缺省 Device 0。software 是 Host OS/内核/CANN 包等软件信息；hardware 包含 Host CPU/内存/硬盘及 Device NPU/AI CPU/AI Core/AI Vector 数量。不能给 software/hardware 加 -d 后就假定所有输出仅属该 Device。

status 可含 Chip Name、Power(W)、Temperature(C)、health；CPU 的 Count/Usage(%)/Frequency(MHZ)；AI Core 的 Count/Usage(%)/Frequency(MHZ)/Voltage(MV)；HBM Total/Used(MB)、Bandwidth Usage(%)、Frequency(MHZ)。括号保留来源显示单位，MV 等大小写不应自行转化为实测物理量；示例中的核数、温度、电压和容量不是产品常数。

`asys health [-d=ID]` 未指定时查看全部 Device；指定后异常时终端只显示前 5 组故障。采集/复跑产物 health_result.txt 保留全部故障码与信息，不能因为终端只有 5 项就认定故障总数为 5。Group of N Device 表示输出分组，Overall Health 为整体状态，ErrorCode Num 与具体故障行需一起阅读。

| 故障级别 | 来源 health 标签 |
|---|---|
| 正常或提示 | Healthy |
| 次要 | Warning |
| 重要 | Alarm |
| 紧急 | Critical |
| 未知 | Unknown |

Healthy 也可能对应提示项，不意味着完全无告警。Warning 不是所有业务都可用的证明，Unknown 也不等于已确定硬件损坏；应结合完整故障清单、对应版本错误码手册和真实算子执行验证。设备身份进一步见[设备与编译架构](../environments/device_identity_and_architecture_selection.md)。

## diagnose 的结果按模式解释

| 模式 | Pass | Warn | Fail |
|---|---|---|---|
| stress_detect | 检测成功 | 检测失败，进一步区分下发/用例、精度或硬件问题 | 来源未给此分支 |
| hbm_detect | 检测成功；可能附新增 ECC 数 | 检测失败 | 来源未给此分支 |
| cpu_detect | 检测成功 | 检测过程中任务调度问题 | 检出硬件故障 |
| aicore_stl_detect | 检测成功 | 调度问题，可能来自软件或硬件 | 检出硬件故障 |
| component | AI Vector 组件检测成功 | 来源未给此分支 | 组件检测异常，结合 debug_info.txt |

不能用一个全局规则把所有 Warn 归为“可用”或“硬件坏”。stress_detect 的 `[ERROR] AML` 相关错误码首位：1 为用例/下发类，2 为精度，3 为硬件；HBM 对应 1 为用例/下发、4 为硬件。该前缀含义限定在这些检测，不适用于全部 CANN 错误码。

HBM 的 `Pass(9)` 或聚合 `(0,9,0,0)` 表示检测后新增 ECC 计数，Pass 并不表示计数为零。来源将检测与激发风险地址、后续隔离关联；不能由 Pass(n) 单独证明已完成隔离或后续业务必然正常。需要保留设备顺序、检测前后状态与完整日志。

diagnose 未指定 -d 时显示所有 Device，只有一张时显示该设备；状态一致可压成 `Pass - All`/`Warn - All` 等，不一致按设备列出。压力检测位于 Performance 分组，HBM/CPU/STL 位于 Hardware，Vector 组件位于 Component。分组名并不改变每个检测的返回语义，也不能用字符串里出现 Pass 就忽略其他 Device 的 Warn/Fail。

diagnose 不传 --output 时仅终端展示、不落盘，与 collect 默认创建输出目录不同。重放/归档时应明确保存原始结果及与之关联的日志，不能用示例数字补空结果。检测的超时、依赖和副作用见[诊断执行边界](asys_diagnostic_side_effects_and_platform_scope.md)。
