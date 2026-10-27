---
type: Guide
title: 设备身份、芯片型号与编译架构的区别
description: 分开核对设备映射、SoC 与 NPU 架构，并沿构建和依赖链确认实际选择的实现。
tags:
- task.debugging
- topic.architecture
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-env-check/references/npu_commands.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-env-check/references/asys_commands.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/npu-arch-adapt.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/platform_info/platform_ascend_c/get_soc_version.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/platform_info/platform_ascend_c/get_cur_npu_arch.md
  role: constraint
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/npu-arch/SKILL.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/npu-arch/references/npu-arch-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/npu-arch/references/npu-hardware-params.md
  role: supporting
created_at: '2026-09-10T13:16:07Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 设备身份、芯片型号与编译架构的区别

Device ID、Chip ID、逻辑设备号、SoC 型号和 NPU 架构号表示不同对象。多卡或容器环境中，不能假设用户可用设备必然是 0，也不能直接把设备枚举表第一列当作所有运行时接口的逻辑编号。

npu-smi 和 asys 可提供设备映射、状态和软硬件信息，但命令选项、字段和支持范围受版本及产品限制。自动化消费者应核对实际工具输出结构；人读表格的列宽和换行不适合作为固定解析契约。健康状态、设备可见和当前进程能够执行算子也需分别验证，不能将单个 Warning/Healthy 标签直接转换为所有场景的可用性结论。

芯片型号的简写可能不足以区分产品。[GetSocVersion](../../apis/utils_api/platform_info/platform_ascend_c/get_soc_version.md)还说明，部分 A2/A3 产品为了二进制复用返回相同枚举；精确产品识别与架构识别应使用各自的接口或安装配置。[GetCurNpuArch](../../apis/utils_api/platform_info/platform_ascend_c/get_cur_npu_arch.md)提供产品到架构的映射入口。

构建系统中的芯片选择、代码中的架构宏和头文件 dispatcher 必须一致。某文件包含目标 API，不证明当前构建会编入该文件；应沿实际 SoC 配置和预处理分支确认实现。现有 [架构变体查证案例](../../runbooks/compilation/arch_variant_lookup_chain_soc_ini_npuarch_dispatcher.md)展示了误读其他架构变体的风险，案例中的具体路径需按安装版本复核。

为同架构产品复用实现时，还要检查产品相关资源、能力开关和编译配置。架构相同不能证明全部产品配置等价。适配范围也包括被依赖的算子和共享构建目标，否则顶层目标虽然声明支持，间接依赖仍可能缺失对应产物。

验证应把实际设备型号、架构选择、构建宏、注册配置及最终二进制串联起来；新增芯片名或目录本身不是适配完成的证据。

## 架构别名与查询失败

来源 npu-arch 资料将 __DAV_C310__ 描述为历史工具链别名；宏后缀不能直接转换成产品系列或 NPU 架构数值，应核对当前 SDK 的定义、dispatcher 和实际编译分支。查询接口返回保留值、未识别值或失败时，应保留未知状态并继续核验安装版本和设备上下文，不能按枚举数字猜硬件能力；DAV_RESV 的具体行为还需固定 SDK 实现支持。

同产品系列的 SKU、部署形态或 vNPU 可见资源也可能不同。INI 的容量/核数是该安装配置声明，不是设备实测；白皮书规格、编译期上限、运行时可见值和实测值分别标识。新增能力判断需确认实际消费者使用哪个值，不能从平台表一行推导所有实例均可用的资源。
