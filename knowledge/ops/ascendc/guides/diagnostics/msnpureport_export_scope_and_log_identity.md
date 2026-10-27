---
type: Guide
title: msnpureport 的日志导出范围、设备身份与材料定位
description: 解释单次和连续导出的范围、SMP 和容器限制、目录中的设备号及缺失日志的观测边界。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/export-one-time.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/export-permanent.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/output-layout.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/product-support.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/troubleshooting.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- msnpureport report
- hisi_logs
- fault_event
---

# msnpureport 的日志导出范围、设备身份与材料定位

Ascend C 错误分析依赖与故障对应的 Device、Host、黑匣子和调用栈材料。固定 reference 的 msnpureport 提供这些导出能力；以下行为限定在该资料，实际 driver/工具版本需核实，不能从一次命令成功推断材料完整。工具来自 `{Driver安装目录}/driver/tools/msnpureport`，具体安装前缀不固定。

## 命令、设备与范围

单次导出采用 `msnpureport report [options]`，来源还保留旧式 `msnpureport [options]`，但设备选择 `-d/--device` 仅支持 report 形式。连续导出采用 `msnpureport report --permanent [options]`。两者的 `-d` 都是逻辑 ID，未指定时导出全部 Device；SMP 共用一个 OS 时，指定组内一个 ID 会覆盖该 OS 下的 Device 日志。不要把配置查询的缺省 Device 0 套到导出，也不要假定指定一个 ID 就只得到单个设备材料。

| 单次范围 | 在基础导出上增加的内容 |
|---|---|
| 不加范围参数 | Control CPU 与其他小核系统日志、进程 coredump 调用栈、黑匣子日志、事件调度、Host 驱动内核与 Device OS 日志 |
| `-a/--all` | 额外包含黑匣子设备事件信息 |
| `-f/--force` | 再包含黑匣子存储空间历史维测信息 |

按类型选择的 `-t/--type` 与上述三档不是同一参数：

| type | 来源中的材料 |
|---|---|
| 0 | 系统日志、调用栈、黑匣子及设备事件、事件调度、Host 驱动与 Device OS 日志 |
| 1 | 系统日志、事件调度、Device OS 及其他模块日志 |
| 2 | 黑匣子、设备事件、历史维测；A2/A3 另可导出当前硬件寄存器 |
| 3 | Device 进程 coredump 调用栈 |
| 4 | vmcore |
| 5 | DVPP、TEE、driver proc 等其他模块，具体范围依产品 |
| 6 | Unified Bus 维测信息 |
| 7 / 8 / 9 | AO 区日志 / AO 计数 / 串口录音 |

来源说明 vmcore 在 Device OS 心跳丢失时伴随黑匣子生成，明确列出 A2/A3/950 支持、Atlas 推理系列与 200I/500 A2 不生成；Atlas 训练系列在产品表中未列清，保持未知。不能因未找到 vmcore 就排除其他故障。type 6–9 在资料中仅列 950 支持，不能泛化到其他产品。

## 部署、并发与输出限制

资料限定工具为 Ascend EP 形态，不支持昇腾虚拟化实例。导出要求 root，输出位置应只允许适当用户访问；普通用户的 Device 权限还取决于 driver 安装方式（来源提到 `--install-for-all`），这不取消导出的 root 要求。

普通容器单次只支持 Host 驱动内核日志，连续导出不支持；来源给特权容器相应例外。容器调用需配套 `--docker` 和 driver/tools 搜索路径，但增加标志不能突破能力限制。自动导出脚本即使在特权容器也不受支持，见[保留与轮转](msnpureport_retention_and_auto_export.md)。

单次可多进程，来源建议不超过 4 个；type 2、4 不支持多进程并发。同路径同时间导出会产生时间戳目录冲突。连续模式同一 Device 不支持并发，在 SMP 下约束扩大到同一 OS；例如两个 Device 共用 OS，分别指定二者也不是独立采集任务。并发数建议不等于可靠性保证。

单次默认在执行目录创建时间戳目录；连续模式可用 `-o/--output` 指定**已存在的相对路径**，不要混入旧单次导出的时间戳目录。持续采集以 Ctrl+C 或 SIGTERM 结束，异常时仍可能失败。路径不可访问、目录属性限制或资源不足应作为执行失败保留，不把旧产物当作新采集结果。不要为方便而自动解除加锁目录属性。

## 日志目录提供的证据

| 材料/位置 | 观测对象与限制 |
|---|---|
| `hisi_logs/device-id/history.log` | 异常场景及历史目录身份；时间戳子目录的 `bin/vmcore` 只在相应故障条件下生成 |
| `hisi_logs/device_info.txt` | 导出设备信息 |
| `slog/dev-os-id/debug/device-id/` | 非 Control CPU 小核系统日志，如低功耗、TS、ISP |
| `slog/dev-os-id/{debug,run}/device-os/` | OS 系统进程调试/运行日志 |
| `slog/dev-os-id/{debug,run,security}/device-app-pid/` | Device 应用进程残留日志；来源说回传 Host 失败时才生成，缺此目录不代表应用没有日志 |
| `slog/dev-os-id/run/event/`、`security/device-os/`、`slogd/` | Event、安全、日志工具自身的不同观测面 |
| `slog/host/host_kernel.log` | 单次导出的 Host 驱动内核日志 |
| `stackcore/dev-os-id/` | 进程调用栈，产品可能使用 stackcore 或 coretrace 文件名，需匹配解析器与符号 |
| `message`、`system_info` | Device OS、事件调度及 HAL/proc/TEE 等信息 |
| `ub_info/dev-os-id/{ubctl,mami}/` | Unified Bus 信息，早期 `hisi_logs/.../ub` 是到该域的软链接；不是算子 Unified Buffer Dump |
| `ao_info/dev-os-id/{ao_self,ao_cnt,ao_uart}/` | AO 日志、计数和串口材料 |

连续导出布局没有单次的 `slog/host/`、顶层 `message`、`ub_info/`、`ao_info/`；故障触发的 OS 日志可在 `system_info/device-id/fault_event_*/message`。布局差异不能当作同一命令随机漏文件。来源输出表在 EVENT 进程归属的措辞上并非完全一致，具体归属应由实际日志头和发行版说明确定。

目录中的 Device ID：950 为逻辑 ID，A2/A3、Atlas 训练/推理系列及 200I/500 A2 为物理 ID；命令行 ID 始终按来源的逻辑 ID 解释。`dev-os-id`、device-id 与应用 pid 也表示不同对象。关联材料时记录命令设备号、映射、OS 分组、pid、时间戳和本次运行，不能直接按同一个数字拼目录。

黑匣子导出慢或卡住时，来源提出芯片复位后重试；复位会改变现场和业务状态，应保留原始失败与已得材料，再按实际诊断需要处理。长任务日志会老化，异常设备也可能无法导出；持续采集能提高异常前材料的保留机会，不能保证所有日志永不丢失。调用栈/Unified Bus 文件的解析入口见 [asys 解析合同](asys_analysis_inputs_symbols_and_error_scope.md)。验证结果需要核对本次退出状态、实际类型、文件时间和设备身份，不能只看目录是否存在。
