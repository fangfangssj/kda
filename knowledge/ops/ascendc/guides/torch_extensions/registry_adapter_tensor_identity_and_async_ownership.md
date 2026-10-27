---
type: Guide
title: 注册 PyTorch 适配器的 Tensor 身份与异步所有权
description: 补全 Meta 与真实分发一致性、ACL 描述符、回调捕获和异常释放。
tags:
- task.implementation
- topic.memory
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/torch/torch_adapter.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/st-test-guide.md
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# 注册 PyTorch 适配器的 Tensor 身份与异步所有权

来源适配器将Torch Tensor→ACL描述符→GetWorkspaceSize→OpCommand回调运行，并分别注册Meta和PrivateUse1。Meta检查同shape/dtype并empty_like，但真实NPU函数没有相同检查；Meta不会必然先于每次真实调用，不能依赖它代替入口验证。来源用x1的shape/dtype给x2也创建描述符，若x2不同可错误解释数据甚至越界，需核对两输入device、dtype、shape及支持域。

连续化后可用实际shape生成连续stride、offset0的描述符，不能对未连续视图直接这样伪装。Byte/Char/Int/Long/Half/Float/Double/Bool/BFloat16映射到ACL enum只表示描述符转换能力，不代表当前Add支持所有dtype；ACL_DT_UNDEFINED应明确拒绝。输出仍需正确device和当前stream，缺DeviceGuard或错误当前device可能使多卡调用不一致。

workspace由Byte Tensor分配可接入框架allocator，但uint64 workspace_size转int64须有界。源码回调只捕获ws、workspace_ptr、acl_stream，未捕获x1_contig/x2_contig/z/workspace_tensor的拥有者；若回调延迟或设备访问未完成，局部Tensor析构可能过早释放/复用。需按实际OpCommand及allocator协议保活并记录stream使用，不能仅凭指针被捕获认为内存有效。

ACL描述符、executor、原始Tensor存储和workspace是不同资源。OpWorkspace析构销毁descriptor却只把executor置空，不等于按所有版本释放executor；运行接口可能消耗它，也可能有独立协议。回调提交成功不等于设备完成，描述符是否可在调用后释放需实际API保证。参考[符号与生命周期](aclnn_symbol_resolution_and_lifetime.md)。

来源TORCH_CHECK(false)之后的delete ws不可达，创建部分ACL对象或第一段失败时会漏清理；应采用异常安全的所有权管理。OpCommand若拒绝/抛出，尚未执行回调也需释放。重新typedef aclnnStatus不能通过自定义宏证明与SDK原声明兼容。

stream(false)和RunOpApiV2属于具体TorchNPU版本语义，不能作为所有版本万能异步封装。结果.cpu()提供后续观测也不能补回已经失效的输入寿命；用例冷却或反复clone只改变复现概率。验证多设备、不同当前stream、非连续输入、异步连续提交、异常和空输入，保持Meta与NPU输出合同一致。
