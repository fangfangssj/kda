---
type: Guide
title: msnpureport 的设备调试配置、核屏蔽与恢复边界
description: 区分 Device 和工具日志，解释 singlecommit、核屏蔽、自动复位及 icache 检查对算子排障和性能的影响。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/config-set-aicore.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/config-query.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/config-set-log.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/tool-log-config.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/product-support.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/troubleshooting.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- msnpureport config
- singlecommit
- accelerator_recover
- Coremask
- icachecheck
---

# msnpureport 的设备调试配置、核屏蔽与恢复边界

本卡面向 Ascend C 算子排障，保存固定 msnpureport reference 描述的工具合同。该资料未固定 driver/msnpureport 发行号；以下默认值、产品范围和恢复要求需要与实际 `msnpureport version`、子命令 help、设备及运行模式核对。本卡未执行设备配置命令。

## 配置身份与查询字段

`msnpureport config --get [--device ID]` 查询日志和 AI Core 等全量配置；旧式 `msnpureport -r` 只查 Device slog 的全局/模块/Event 日志级别。查询、设置未指定设备时默认逻辑 ID 0；[导出命令](msnpureport_export_scope_and_log_identity.md)未指定设备时默认全部，二者不能互换。`help`、`config --help`、`report --help`、`version` 是查询入口，来源说明帮助/版本查询无权限限制。

| 输出字段 | 应解释为 |
|---|---|
| `Icache check Range` | icache 校验窗口，来源单位 KB |
| `Accelerator Recover` | TaskSchedule 是否自动复位加速器 |
| `Aic Coremask` / `Aiv Coremask` | 核屏蔽状态 bitmap；不是直接的核数整数 |
| `Aic Singlecommit` | AI Core 内部多指令串行/并行模式 |
| `Global Level` / `Module [xxx]` | Device 系统类日志的全局/模块级别 |
| `Event Level` | Device Event 日志开关 |

`-d/--device` 指逻辑 ID，不能直接填入物理编号。来源以可见物理 ID `0,1,4,5` 对应逻辑 ID `0,1,2,3` 为例；排序映射和 SMP 分组须用本机信息确认，不能硬编码每 4 个一组。在 SMP 共用 OS 场景，`--log -g ... -d ID` 同时作用于 OS 共用进程模块和指定 Device 小核模块。来源 config 参数表还混入“导出该 OS 日志”的文字，那是导出语义，不能据此声称 config 操作会导出文件或改动组内所有独立小核。

## 改变执行行为的诊断配置

命令形态为 `msnpureport config --set --device ID <选项>`。来源要求设置类命令由 root 执行，并建议在 AI 应用退出后设置；运行中修改可能使应用异常。容器中还需当前版本支持 `--docker`，并能找到 driver/tools 中的工具；命令存在不证明该功能在当前部署形态可用。

| 选项 | 来源定义的效果 | 恢复与解释边界 |
|---|---|---|
| `--accelerator_recover 0` | 关闭 TaskSchedule 自动复位，便于保留更详细寄存器现场，但影响性能 | 来源要求定位完成后重启运行环境恢复 AI Core 业务；不能仅将值改回 1 就宣称业务已恢复。默认 1 为自动复位 |
| `--singlecommit 1` | AI Core 内部多指令串行化 | 设为 0 或重启后关闭；此状态下的性能不代表正常并行执行性能 |
| `--aic_switch 0 --coreid ...` | 屏蔽指定 AI Core；分离架构指 AIC | `--aic_switch 1 --coreid 0xFFFF` 或重启恢复来源初始状态 |
| `--aiv_switch 0 --coreid ...` | 屏蔽指定 Vector Core | `--aiv_switch 1 --coreid 0xFFFF` 或重启恢复；不能与 AIC 掩码混用 |
| `--icachecheck K` | 比较出错 PC 前后各 K KB 范围的 icache 与 GM | 来源范围 `[0,131072]`，配置前记录原值，定位后恢复；不能把 K 当作总窗口宽度 |

屏蔽命令一次最多列 4 个 core ID，以逗号分隔；任一个无效都会报错。`0xFFFF` 是来源的恢复特殊值，不能当作普通 core ID 或实际核数。对于涉及核间同步的算子，执行时需要一次获得所需核数；屏蔽后可用核数不足可能使算子失败。对照 `Aic/Aiv Coremask` 和本次 Host plog 中的 `core_num`，避免把人为减少核数导致的失败误判为算子原有死锁。来源建议单算子调用场景用于定位故障核，不能从一次屏蔽后通过推导被屏蔽核必然损坏。

A2/A3 的寄存器定位材料可以与 `--accelerator_recover 0`、复现及 `report --type 2` 配合；保留现场和恢复执行环境是不同动作。芯片复位会改变现场，不能成为所有错误的无条件首步。

来源支持表中，AIC 屏蔽不支持 Atlas 200I/500 A2；AIV 屏蔽还不支持 Atlas 训练系列；icache 检查不支持 Atlas 训练系列。A2/A3 与 950 的分离架构解释不应扩展到所有 Atlas 产品。工具整体的 EP/虚拟化边界见[导出范围](msnpureport_export_scope_and_log_identity.md)。

## Device 日志与工具自身日志

`config --set --log` 中，`-g/--global` 设置全局级别，`-m/--module 模块:级别` 设置模块级别；模块名称从本次 `config --get` 的 `Module [xxx]` 读取。级别为 debug、info、warning、error、null，null 表示不输出。`-e/--event enable|disable` 独立控制 Event，来源默认 enable。SLOG、CCE 是示例模块，不是固定完整列表。

`msnpureport report --print 0|1` 与 `report --log_level debug|info|warning|error` 只控制工具自身维测日志。来源默认 print=0、log_level=info；print=0 的 Host 日志位置为 aarch64 的 `/var/log/messages`、x86_64 的 `/var/log/syslog`，实际还应核对系统日志服务。print=1 为终端输出；容器中无法读取对应 syslog 时可用它观察工具过程，不代表 Device 日志已采齐。这些选项不改变 Device 的全局/模块级别。

调成 debug 会增加日志量并可能加快老化，应记录和恢复原级别，保留本次配置、设备映射与采集时间。确认配置生效需读回字段，并验证恢复后的业务行为；无输出或工具自身没有报错都不能代替这些证据。
