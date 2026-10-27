---
type: API
title: 排序与合并概述
description: Region Proposal（简称RP）是目标检测算法中用于定位潜在目标区域的候选框。Region Proposal采用特定的数据结构进行存储和管理，以支持高效的排序和筛选操作。在业务场景下，为了从大量Region
  Proposals中筛选出最有可能包含目标的区域，需要根据每个RP的置信度score进行排序。
tags:
- task.lookup
- topic.compute.ordering
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/sort_and_merge_ISASI/overview.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 概述

Region Proposal（简称RP）是目标检测算法中用于定位潜在目标区域的候选框。Region Proposal采用特定的数据结构进行存储和管理，以支持高效的排序和筛选操作。在业务场景下，为了从大量Region Proposals中筛选出最有可能包含目标的区域，需要根据每个RP的置信度score进行排序。


以Atlas A3系列产品和Atlas A2系列产品为例：

Region Proposal支持两种数据类型：half/float。

下图展示了n个Region Proposal的结构示意图，每组\[x1， x2， y1，y2，score，label\]代表一个完整的Region Proposal。其中，x1、x2、y1、y2表示候选框的坐标，score为置信度，label为目标的类别标签。

**图1**  Region Proposal结构示意图

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/region_proposal.png)<a id="图1-Region Proposal结构示意图"></a>

排序组合接口用于实现（score, index）对的排序操作，其中index用于索引完整的RP数据，数据结构请参考[Sort32排序示意图](sort32.md#图1-Sort32排序示意图)。其中，[Sort32接口](sort32.md)实现一次迭代内对32个（score, index）对的降序排列。[MrgSort接口](mrg_sort.md)将多个（最多4个）已经排好序的子队列，归并成一个全局有序的队列。

