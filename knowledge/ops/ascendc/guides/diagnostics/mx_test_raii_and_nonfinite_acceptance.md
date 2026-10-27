---
type: Guide
title: MX 测试的资源析构与非有限值验收
description: 保留较完整 I/O 和 RAII 设计，同时定位异步失败与两份验证器的差异。
tags:
- task.testing
- topic.implementation.debugging
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/acl_utils.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/io_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/verify_result.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze_torch.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/run.sh
  role: supporting
created_at: '2026-09-11T02:53:42Z'
updated_at: '2026-09-11T02:53:42Z'
platforms: []
---

# MX 测试的资源析构与非有限值验收

来源以状态位记录ACL初始化、设卡和stream创建，以unique_ptr管理每次成功申请的Host/Device内存，失败时抛异常。会话对象先建立、buffer后建立，正常析构可先释放buffer再结束会话。正常路径同stream依次H2D、Kernel、D2H并同步后写文件，这个顺序是正确结果观测的条件。

RAII只管理Host控制流上的释放，不能自动证明异常路径中已排队的异步操作结束。若后续提交或同步失败，清理需依据实际Runtime状态/所有权协议处理在途访问，不能简单把“有析构器”当全部生命周期正确。析构忽略返回值也不能作为释放成功证据。

ReadExactFile要求实际字节数等于预期，读取还检查seek与短读；WriteFile循环处理短写/EINTR。这些比只检查文件存在更强，但预期大小本身仍须由正确dtype、逻辑shape、物理布局与不溢出的算术得到。按/proc/self/exe定位输入输出使默认路径独立于cwd，fallback仍需绑定实际路径，防止生成脚本与程序读不同副本。

文件验证器将uint16按BF16位视图恢复并reshape(M,N)，拒绝非有限输入/输出/差值；每点相对误差不得超过0.1，同时绝对误差>0.001的元素比例不得超过0.001。它并不是文档容差表所描述的标准rtol/atol加性条件。golden=0且差值非零被视为无穷相对误差，也是该脚本的明确选择。

Torch测试用clamp_min(1e-8)作相对误差分母，且未并入非有限mask：NaN的大小比较返回False，可能使point/error计数均为0，即使max_diff为NaN仍显示通过。两份测试标准因此不一致，不能相互替代；应先显式分类NaN/Inf、核对shape和预期元素数，再按真实验收统计。见[误差指标与输入表示](../precision/error_metrics_and_represented_test_inputs.md)。

摘要的均值、RMSE、左上角样本不能替代全矩阵比较和最差坐标；随机正数样例也不证明符号、抵消和边界Scale语义。运行脚本删除旧输出后要求新文件存在可避免部分陈旧结果，但跳过构建和未固定依赖仍需产物身份检查。本批未运行设备测试，未继承来源性能或精度通过的宣称。
