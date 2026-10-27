---
type: Guide
title: Blaze MX 组件 Params 与直调启动合同
description: 保留组件组合、实际 dtype 与转置分发，并纠正仅凭目录或宏推导功能支持。
tags:
- task.implementation
- topic.implementation.tiling
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_kernel/matmul_blaze_kernel.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_kernel/matmul_blaze_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/run.sh
  role: supporting
created_at: '2026-09-11T02:53:42Z'
updated_at: '2026-09-11T02:53:42Z'
platforms: []
---

# Blaze MX 组件 Params 与直调启动合同

来源由Host计算Tiling并选择transA/transB四个编译实例；Device以ProblemShape、DispatchPolicy、BlockMmad、BlockScheduler、GemmUniversal及其Params组装计算。Block管理搬运/MMAD及片上流水，Scheduler管理任务坐标与核分配，具体遍历算法依赖实际组件版本，不能将所有Scheduler概括为蛇形。

当前Kernel固定A/B=fp8_e4m3fn_t、输出bfloat16_t、两Scale为fp8_e8m0_t，bias指针为nullptr，batch=1。转置通过DNExtLayoutPtn/NDExtLayoutPtn选择物理解释，四个启动分支均采用NONE_FULL_LOAD_MODE。目录中存在FP4生成器、BiasType或QBMM类名，不代表该Kernel已支持FP4、bias或任意batch。

组件::Params应从选定类型推导，并逐字段核对顺序、单位与意义。TilingData与多个Params之间是显式字段映射关系，不要求二者拥有完全相同内存布局。聚合初始化错位有时会因类型不匹配而编译失败，也可能在字段同类型时静默编译并产生错误值；不能把excess elements诊断当唯一表现。

来源TilingData按8字节pack/align共享，Host桥按值传入结构；声明、定义、调用参数顺序及Host/设备工具链布局必须一致。grid取usedCoreNum，不能为零或超出实际硬件支持。融合__mix__只选择执行模式，不会自动提供Epilogue输入、workspace、同步或最终归并语义；A常驻、Grouped和CV需要匹配组件及独立合同。

Blaze与blaze_custom是否可组合取决于接口和状态/资源协议，目录名本身不是类型证明。不能无条件混用，也不能把来源“一律不得混用”的流程规则提升为永恒技术限制。官方依赖变化还会改变Params字段，因此应固定实际blaze与tensor_api版本并记录兼容编译器。run.sh按目录是否存在判断依赖并浅克隆活动分支，目录存在不证明版本一致，跳过构建也不证明旧产物对应当前依赖。

CMake把Kernel cpp标记ASC、普通注册单元保留C++，目标arch为dav-3510；目前Python/Torch发现无条件发生，所谓“可选扩展”不等于当前CMake可无Torch完成配置。验证类型链实例、四种transpose、非零tile和真实目标产物；本批仅静态分析，未取得外部依赖或完成编译。
