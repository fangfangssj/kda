---
type: Guide
title: msnpureport 连续采集的日志保留、容量与脚本轮转
description: 保存各类日志数量和大小参数、覆盖边界及自动导出脚本的路径、清理和半容量轮转合同。
tags:
- task.debugging
- topic.architecture
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/export-permanent.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/auto-export-script.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msnpureport-toolkit/references/troubleshooting.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- msnpureport_auto_export.sh
- logClearFlag
- sys_log_num
- stackcore_num
---

# msnpureport 连续采集的日志保留、容量与脚本轮转

持续导出用于减少长任务老化或异常后无法采集造成的证据损失。固定 reference 给出原生 `report --permanent` 和 `msnpureport_auto_export.sh` 两条路径，它们的参数、容器支持和路径合同不同。资料未固定工具发行号，本卡未运行脚本，以下行为应与实际安装实现核对。

## 原生连续模式的保留窗口

数量超过配置时，新日志覆盖最早日志；单文件超过大小时转到新文件。数量与大小不能互换，也不能简单把所有字段相乘当作全量磁盘上限。单位保留来源的 M/G 写法，不自行解释为 MiB/GiB。

| 参数 | 对象 | 范围 | 默认 |
|---|---|---|---|
| `--sys_log_num` | 系统进程 debug/run 目录各自的最大文件数 | 1–1000 | 10 |
| `--sys_log_size` | 上述单文件大小，M | 1–100 | 10 |
| `--event_log_num` / `--event_log_size` | EVENT 文件数 / 单文件大小 M | 1–1000 / 1–100 | 10 / 10 |
| `--sec_log_num` / `--sec_log_size` | 安全日志文件数 / 单文件大小 M | 1–1000 / 1–100 | 10 / 10 |
| `--fw_log_num` / `--fw_log_size` | 非 Control CPU 系统日志文件数 / 大小 M | 1–1000 / 1–100 | 10 / 10 |
| `--stackcore_num` | 调用栈文件数 | 1–100 | 50 |
| `--bbox_num` | 黑匣子日志文件夹数 | 1–10000 | 1000 |
| `--fault_event_num` | 故障触发的 Proc/OS 文件夹数 | 1–100 | 10 |
| `--slogd_log_num` | 日志工具自身维测文件数 | 1–100 | 2 |
| `--app_dir_num` | Device 未来得及回传 Host 的剩余应用日志数量 | 1–100 | 8 |

`--sys_log_num` 分别约束 debug 与 run，不是两者合计；`--stackcore_num` 分别约束 `stackcore/dev-os-id/` 与其 `udf/` 下文件，也不能当作全设备合计。文件名可因产品变为 coretrace。`--app_dir_num` 名称含 dir，但原文描述为剩余应用日志文件数量；真实计数对象应查对应版本，不能仅由名字自行改成目录数。

`-o/--output` 只接收已存在的相对路径；不传则使用执行目录。持续模式一个 Device/同一 SMP OS 不能并发导出；异常场景仍可能失败。材料过期应按最早/最晚时间判断，增大 file_num 或 size 只改变保留容量，不证明整个任务时间段都已覆盖。

## 自动导出脚本的不同合同

脚本位于实际 driver/tools，来源参数表为：

| 参数 | 来源要求 |
|---|---|
| `timeInterval` | 大于 0 的整数，单位秒 |
| `logAbsolutePathCapacity` | 大于等于 2 的整数，单位 G |
| `logAbsolutePath` | 绝对路径；不存在时脚本创建 |
| `logClearFlag` | 1 表示启动清理目标路径日志，0 不清理，默认 1 |

来源调用格式及例子只显示前三个参数，表中却有第四项。未核对实际脚本前不能推定该版本接受第四个位置参数；尤其不能在已有故障材料的目录直接照抄默认调用。脚本要求 root、不支持容器（包含特权容器），不应映射进容器；重复并发会重复采集所有 Device 并浪费资源。终止使用 Ctrl+C 或 SIGTERM；加锁目录属性会影响执行，不把自动解除保护纳入采集逻辑。

脚本在目标目录使用 `msnpureport_log_new` 和 `msnpureport_log_old`：

- Device 的 `history.log` 和 `message.log` 会在同名文件内老化，因此导出后追加到 `history_new.log` / `message_new.log` 并去重；去重细节以脚本实现为准，不能用任意“相同行删除”算法替代。
- 其他时间戳命名日志按拷贝覆盖方式保留，每次快照不是一份完全独立的新增日志全集。
- 当 new 中日志容量**超过**配置容量的一半，清空 old，再将 new 全部移动至 old。配置 10G 时阈值为 5G；恰等于 5G 不满足来源写明的“超过”条件。

由此可知 old 也会被后续轮转删除，连续导出不等于永久归档。来源称实际占用约为配置容量，但它不是瞬时严格上界：一次采集量、写入粒度与文件系统开销可能带来超量，仍需独立监测空间。原生模式按各类文件上限覆盖，脚本按 new/old 容量轮转，不应合并成同一保留公式。

判断材料完整性需记录采集起止、故障时刻、当前最早日志、轮转参数和启动清理选择；对照故障附近相邻日志是否跨越被覆盖区间。仅看到 new/old 两个目录或采集进程仍运行都不足以证明证据未丢失。设备/OS 身份及目录差异见 [导出范围](msnpureport_export_scope_and_log_identity.md)。
