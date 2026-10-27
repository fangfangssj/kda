---
type: API
title: GetUserWorkspace
description: 获取用户使用的workspace指针。Kernel直调开发方式下，如果未开启HAVE_WORKSPACE编译选项，框架不会自动设置系统workspace。如果使用了Matmul
  Kernel侧接口等需要系统workspace的高阶API，kernel侧需要通过SetSysWorkSpace设置系统workspace，此时用户workspace需要通过该接口获取。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/workspace/GetUserWorkspace.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
---

# GetUserWorkspace<a name="ZH-CN_TOPIC_0000001666932721"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取用户使用的workspace指针。核函数（Kernel）直调开发方式下，如果未开启HAVE\_WORKSPACE编译选项，框架不会自动设置系统workspace。如果使用了[Matmul核函数（Kernel）侧接口](../../../adv_api/cube_compute/matmul_kernel/index.md)等需要系统workspace的高阶API，核函数（Kernel）侧需要通过[SetSysWorkSpace](set_sys_work_space.md)设置系统workspace，此时用户workspace需要通过该接口获取。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  接口参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1185064715302"><a name="p1185064715302"></a><a name="p1185064715302"></a>传入workspace的指针，包括系统workspace和用户使用的workspace。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

用户使用workspace指针。

- [Im2colBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/im2col_backward_950.md)
- [CircularPad2d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [BlockSparseAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention.md)
- [Rasterizer A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/rasterizer.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [CircularPad3d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_3d.md)

## 配套 API 与常见组合

- [SyncAll](../../sync_control/inter_core_sync/sync_all.md)——跨核协作中 GetUserWorkspace 提供各核写中间结果的共享工作区基址、SyncAll 在阶段间保证全核可见（rasterizer 两阶段实证，P(a|b)=0.70）
- [CrossCoreSetFlag](../../sync_control/inter_core_sync/cross_core_set_flag_isasi.md)——AIC/AIV 混合核以 user workspace 为跨核共享数据通道，CrossCoreSetFlag 对其读写定序（add_lora 实证，P(a|b)=0.87）
- [CrossCoreWaitFlag](../../sync_control/inter_core_sync/cross_core_wait_flag_isasi.md)——跨核流水的数据通道+同步通道配对：AIV/AIC 经 user workspace 交换中间结果（P(a|b)=0.87）
