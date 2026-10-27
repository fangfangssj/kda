---
type: Guide
title: 错误层级与非对齐搬运证据
description: 保存 ACLNN 诊断入口，并纠正以 507035 推断统一 32B 搬运长度的做法。
tags:
- task.debugging
- topic.data_movement
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-runtime-debug/references/debug_workflow.md
  role: primary
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-09-11T02:40:45Z'
platforms: []
---

# 错误层级与非对齐搬运证据

来源区分ACLNN调用层与异步设备层。GetWorkspaceSize等调用返回非零时，结合实际返回码和aclGetRecentErrMsg获取上下文，消息为空时需容错；调用成功后仍需检查后续stream完成结果。错误可能晚于其发生操作上报，不能只定位到最后一个同步调用。

| 来源代码 | 诊断入口 |
|---|---|
| 161001 / 161002 | 非法空指针与参数域：dtype、shape、属性、format及输入类型推导关系 |
| 361001 | API内部Runtime调用异常，需要更底层错误信息 |
| 561002 | Tiling异常：输入域、分支、算术/数组与实际key产出 |
| 561003 | Kernel查找异常：编译变体、SEL/key、原型约束、vendor和SOC |
| 561107 / 561112 | OPP环境路径未检测到 / 算子二进制库未加载 |
| 507035 / 507046 | Vector Core Exception / Stream Sync Timeout，不能混同 |

这些是固定参考的代码映射，不是“某个代码只可能一种根因”。找不到Kernel不等于只缺op_api的.so，还可能缺具体设备变体或配置。Host UT可隔离参数/Tiling函数，但通过只覆盖所测输入、mock和链接产物，不证明全部Tiling正确；失败也需区分测试装配与实现问题。

507035不能按来源“对齐高置信、越界低置信”直接排序根因。DataCopyPad的非对齐扩展就是为有效字节数非整32B等场景服务，不能把blockLen一律向上取整后从GM多读/多写。DataCopyExtParams的blockLen按字节填写，来源将{1,1,innerDim,0}中的字段位置混作有效长度且再按元素换算，不能照抄。应按具体方向/重载核对有效长度、行数、stride与UB padding。

创建LocalTensor偏移视图本身不意味着每个offset都需32B；其实际消费者可要求特定对齐。若确需对齐行距，定义独立物理pitch，并同步更新GM/UB地址、分配容量及padding；不能只改逻辑chunk尺寸后沿用所有其他公式。32B对应FP16/BF16的16元素、FP32的8元素是单位换算，不是每个接口统一的count约束。

UB越界、scratch不足、错误SOC/执行模式、地址/同步和设备异常都需原始日志及数据流佐证。减小tileRows同时改变容量、循环和时序，异常消失不能确认仅是chunk对齐。空间查询API需匹配实际Reduce等接口，不能认定所有临时区都由同一个函数查询。验证原失败输入、非对齐tail、其他dtype、完整key映射与真实安装产物，本批未上板重放。
