---
type: Runbook
title: "registry-invoke（非直调）MC2 自定义算子的 runtime env 契约：ASCEND_CUSTOM_OPP_PATH 绝对路径 + 真实 HCCL 通信组/comm 名"
description: "经注册表 aclnn 工程化路径（安装的 op package，非 ACLRT_LAUNCH 直调）派发、绑定 HCCL 通信域的 MC2 自定义算子有两条硬性 runtime env 要求：ASCEND_CUSTOM_OPP_PATH（下划线形式，非 OPPPATH）必须指向安装的 vendors/<name>/ 目录且必须是绝对路径（相对路径在 real_path() 内因 python CWD 与导出 CWD 不同而失败）；派发时算子框架要求有效 HCCL 通信组——即使单进程也要先 init_process_group('hccl')，group 属性用 get_hccl_comm_name(rank) 取真实名，绝不能写字面量 'hccl_world_group'（未注册 → null handle）。"
tags:
- task.testing
- topic.parallelism
- topic.data_movement
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- a2
- a3
- "950"
aliases:
- OL-303
---

# registry-invoke（非直调）MC2 自定义算子的 runtime env 契约：ASCEND_CUSTOM_OPP_PATH 绝对路径 + 真实 HCCL 通信组/comm 名

Registry-invoked MC2 custom-op runtime env contract (aclnn dispatch through the installed op package — NOT the kernel-direct ACLRT_LAUNCH path; applies to ops whose dispatch binds an HCCL comm domain) — `ASCEND_CUSTOM_OPP_PATH` must be ABSOLUTE (underscore OPP_PATH, not OPPPATH); HCCL dispatch needs a real comm group (`init_process_group` even single-process; group attr = real `get_hccl_comm_name`, NOT the literal "hccl_world_group").

**Principle**: Two runtime-env requirements bite when dispatching a registry-invoked MC2 custom op of this class, both failing with cryptic errors if violated:
1. `ASCEND_CUSTOM_OPP_PATH` (note: `OPP_PATH` with underscore, NOT `OPPPATH`) must point at the installed `vendors/<name>/` dir AND must be ABSOLUTE. A relative path fails inside `real_path()` because the python CWD differs from the export CWD — always set it absolute.
2. The MC2 op framework REQUIRES a valid HCCL comm group at dispatch (`NnopbaseSetHcomGroup` → `HcomGetCommHandleByGroup`). Even a single-process run must `dist.init_process_group('hccl', rank=0, world_size=1)` first, and the group attr passed to the op must be the REAL comm name — fetched via `dist.group.WORLD._get_backend(torch.device('npu')).get_hccl_comm_name(rank)` — NOT the literal string `"hccl_world_group"` (not a registered group → null handle).
