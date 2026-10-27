---
type: Guide
title: DataCopyPad 的版本、重载与尾部真实范围
description: 结合同一来源的失败及后续成功记录限定 DataCopyPad 可用性，避免用填充 workaround 掩盖越界或改变布局。
tags:
- task.debugging
- topic.data_movement.data_copy
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: primary
created_at: '2026-09-14T04:24:48Z'
updated_at: '2026-09-30T01:53:56Z'
platforms: []
---

# DataCopyPad 的版本、重载与尾部真实范围

DataCopyPad 的可用性应按 SDK 发行、SoC、方向、dtype、参数重载和实际访问范围判断。固定来源 EC-23 先记录 CANN 9.0.0/Ascend950PR_9589 的 UB→GM 507035 失败，随后又追加 V351 上双向 FP32、FP16/BF16 以及单个 int64 输出的成功案例。因此不能把该旧标题概括为“Ascend950 的 UB→GM DataCopyPad 不支持”。这些是来源报告，原始环境和全部探针未在本轮重跑。

同一错误码不足以区分不支持重载、非法对齐、源/目的容量不足、错误 stride、架构不匹配或异步依赖。先定位失败指令与完整子错误、实际解析的签名、单位和地址，再缩小形状、方向与类型做比较。EC-53 的 MTE 地址越界记录与 EC-27 的目标配置记录已说明 507035 不具有唯一根因；某平台一次未报错也不证明其允许越界。

DataCopy 的对齐 count、DataCopyPad 的有效 blockLen、UB 源 padding 和 GM 行距是不同量。不能将所有接口统一按 32 字节扩大逻辑长度。对末行或小输出，如果改用对齐 DataCopy，需证明源和目的真实分配覆盖整个物理搬运区间，并且没有写入其他行或其它核拥有的区域；SetGlobalBuffer 声明长度不会扩大分配。

Host 预填充可构成一种有条件的布局方案：输入、输出真实物理行距都按要求分配并传给 Kernel，再按输出合同返回有效逻辑区域。只在整个 Tensor 末尾增加空间，不会保护中间各行的越界写。narrow 返回的视图还可能非连续；若接口要求连续结果，需保留该合同并计入额外复制成本。

来源 EC-22/23 同时含有“向上取整 count 即可”“后续写能自我修复”的建议，这些不能替代逐区域所有权与实际地址验证。多核重叠写、源尾部读取、count 小于对齐块以及行间 padding 均需单独处理；单核通过不唯一指向该类竞态。

成功测试的范围也应保留：一个 8 字节 int64 UB→GM 写回成功，不证明所有打包类型、stride 和多块组合可用；若只有上游旧 guard 而没有目标 SDK 证据，不应据此禁用已存在的 API，也不应盲目删除所有 guard。按实际安装声明和边界测试选择可用路径，比较完整布局转换与内核成本。

## 配套 API 与常见组合

- [DataCopyPad](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md)
- [DataCopyPad](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_ub_to_gm.md)
- [GmUbDataMove](../../apis/simd_api/basic_api/data_move_guide/gm_ub_data_move.md)
