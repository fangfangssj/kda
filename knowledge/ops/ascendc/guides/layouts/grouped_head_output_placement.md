---
type: Guide
title: 分组 Head 输出到 NBSD 的块放置
description: 给出分组 head、batch、query 起点的完整目标地址及普通搬运参数单位。
tags:
- task.implementation
- topic.data_movement.layout_transform
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P59_output_transpose_fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P59.yaml
  role: supporting
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# 分组 Head 输出到 NBSD 的块放置

对于目的连续布局[N,B,S,D]，组内head坐标h=n2Idx*G+gStart+j、batch b、query起点s0、列d的元素偏移为 `((h*B+b)*S+s0)*D+d`。来源起始公式包含n2Idx、gStart和b，未包含s0；它只在当前tile从query0开始、或该偏移已在外层基址加入时完整，不能将其原样用于所有S1块。

若当前UB按g个连续[S_tile,D]块组织，每次写出一个块，目的相邻head块起点相距B*S*D元素。普通DataCopyParams的blockLen可取S_tile*D*sizeof(T)/32，dstStride取(B*S-S_tile)*D*sizeof(T)/32，srcStride=0描述连续源块。所有除以32之前必须确认可整除、非负及字段范围，不能截断余数后静默少写。源UB若有额外padding，需按真实物理布局调整，而非固定0。

gCount只覆盖当前有效head范围，S_tile只覆盖当前query尾长。普通参数不满足非对齐或字段上限时，可拆批或使用支持的扩展重载并重新换算GM/UB单位；不能只更换结构体类型。连续写入大块可能更快，但此处应比较省去中间布局写读与独立Transpose后总成本，stride本身并不保证固定性能次序。

BNSD、BSND、BSH、TND/NTD需要各自展开坐标关系：BSH先从H恢复head与D，变长T使用真实累计token边界。仅列出这些名字不能证明同一个DataCopy公式全支持。验证非零s0、非零batch/head组、末组gCount、query双尾和源坐标编码，确认无遗漏或邻接输出覆盖。
