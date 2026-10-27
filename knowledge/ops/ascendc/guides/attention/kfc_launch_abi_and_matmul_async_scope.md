---
type: Guide
title: KFC Launch ABI 与 Matmul 异步范围
description: 保留系统区和用户区分离，修正参数数目和硬编码地址窗口。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate_all.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/wait_iterate_all.md
  role: constraint
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: supporting
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-14T04:24:48Z'
platforms: []
---

# KFC Launch ABI 与 Matmul 异步范围

KFC混合入口需要按实际编译/runtime约定注册Matmul对象、系统workspace和tiling；AIC只执行注册分支、AIV调度是该入口模型，不是所有Matmul都按此写。多个框架Matmul对象可合法注册，不应混同底层FAQK实例共享资源问题。

来源workspace属性位于第8参、地址观测差4GiB、系统保留16MiB、消息区清64KiB及launch中间槽nullptr为特定CANN9.1/dav2201案例。它们不能推广为地址恒等式或任何入口固定参数位置；尤其原文称14参却只列9个名字，不能凭标题构造ABI。Host和Device真实签名、属性及指针应逐字段匹配。

将用户workspace作为独立普通GM_ADDR可避免从特殊系统参数错误派生基址，但仍需真实系统区容量与Host allocation边界。不能把GetSysWorkSpacePtr()与入参差值直接手工加减4GiB，也不能仅因一次测试通过就只清固定64KiB。workspace初始化若为每launch必需应计入完整执行成本。

MatmulApiTiling可推导类型、形状、bias及buffer设置，GetTiling状态必须检查；之后手改shareL1/L0C/depth可能破坏计算器结果，不能据“手填TCubeTiling挂死”宣称所有明确合法手工tiling都禁止。类型级transB与运行时SetTensorB的支持组合按实际实现验证，source类型级转置错值仅约束其配置。

IterateAll<false>为异步，消费结果前使用匹配WaitIterateAll及End；已有API分别有GM和Local输出重载，不能因GM版本正确就直接替UB输出。单次输出容量至少singleCoreM*singleCoreN。Iterate/GetTensorC也有合法同步模型，来源慢4倍或数值错的尝试不能形成“唯一又快又对的调用形态”。

同一对象先提交c+1再等待c需确认API允许并发请求和可区分结果；双GM槽不自动建立请求队列。AIV worker数应从真实核型/资源与task规划推导，固定min(40,tasks)不是跨平台值，闲worker也要遵循注册/同步参与协议。

## 编译分支与系统 Workspace 初始化的版本条件

插件错误汇总 EC-57 记录，某 CANN 9.0.0 DYNAMIC_MODE 构建只设置全局 ASCENDC_MATMUL_AICORE，未使 AIC/AIV 分别选择 KFC 的服务端/客户端分支；来源用分 pass 的 SPLIT_CORE_CUBE、SPLIT_CORE_VEC 修正。其后更正又称 CANN 9.1.0 可从 __DAV_CUBE__/__DAV_VEC__ 派生这两个宏。因此可复用要求是核对真实编译命令、预处理分支与注册协议，不是所有 SDK 必须手动加同一组宏。不能把两类宏同时全局定义来代替分 pass 配置，也不能把任何 507015 都归因于这些宏。

EC-58 的构建中 matmul_intf.h 由显式 include 或相应 pass 的 interface library 传播；kernel_operator.h 的存在不单独证明 MatmulImpl/REGIST_MATMUL_OBJ 已可见。检查实际头文件和 include 链，不能将别的高级 API 的传递包含方式统一套用。

EC-68 在 CANN 9.1.T500 的独立 ACLRT_LAUNCH 案例里记录了先设置 SetSysWorkspaceForce(workspaceGM)、再初始化 TPipe/Matmul 及派生用户区的修复。这是该入口依赖的系统基址合同；其 16MiB 偏移不应固化到所有版本。Host 分配必须覆盖真实系统保留区及全部用户区，单纯把普通入参命名为 workspace 不证明运行时使用了这个基址。上述版本记录来自固定说明，本轮未验证对应 SDK 分支。
