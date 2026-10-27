---
type: Guide
title: 注册转直调的 Tiling 保真与 launch ABI
description: 提取依赖闭包、双 pass 可见性和参数布局，限定机械替换规则。
tags:
- task.implementation
- topic.compilation
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-to-direct-invoke/references/custom-op-to-kernel-launch-checklist.md
  role: primary
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# 注册转直调的 Tiling 保真与 launch ABI

注册转直调需要从框架上下文中提取shape/dtype/属性、平台查询、workspace和blockDim，同时保持Tiling的对齐分支、ceil/floor、clamp、溢出检查及整数乘除顺序。看似等价的数学式在截断整数算术下可不同；动态平台查询不能直接换成随意constexpr。PlatformAscendCManager和gert::TilingContext的替换需在相同SoC/资源含义上证明，不是文本一换就等价。

依赖闭包以实际使用符号的定义及其直接依赖为单位；中转头与真实namespace可能不同。SDK API头、traits、平台常量和算法helper应各自保留必要定义，include路径清零不证明编译闭包完整，DTYPE_*及编译宏仍可能来自外部。header只放声明也不代表所需template实例和链接符号已提供。

把宏TilingData变为plain struct需保留字段宽度、顺序、数组、嵌套及所有被使用字段，还要验证size/alignment/offsetof和序列化布局；来源“不需要pack”不能覆盖有不同对齐的Host/Device ABI。按值传struct与传GM tiling指针是不同launch合同，均受目标编译器支持和参数容量约束，不能只改入口签名。

原TilingKey的dtype/模式/variant选择需在Host dispatch或相应入口保留完整覆盖及非法值处理；拆为多个显式入口是一种实现选择，非所有直调都必须删除extern C、TILING_KEY_IS。KERNEL_TASK_TYPE_DEFAULT和kernel qualifier需与实际Cube/Vector/MIX执行一致；只消警告不等于workspace可删除或永远为0。

-xasc Host/Device双pass中，Host不可见的MicroAPI/Reg实现可由匹配版本的__NPU_HOST__等宏隔离，但共享POD类型和launch函数签名需两侧可见；函数body条件编译与整个声明被删不同。BF16 Host编码容器和设备bfloat16_t要区分类型名、mangling及内存位宽，可用经验证的显式入口减少实例化歧义，不能断言所有BF16都只在Device可用。

Host设备指针和入口GM_ADDR需满足真实ABI与地址空间。以GM_ADDR贯穿是一种工程做法，“不能经void*”不是普遍C++/ACL规则；通过void**写入不同指针对象须考虑类型安全，必要时先接收void*再按公共接口转换。输入校验可先做静态shape/attr关系，涉及数值的合法性也不能仅因会同步就省略，须由接口合同决定如何实现。

本地化依赖、独立编译、成功launch及数值等价是四层证据；原算法逐行保留有助定位变化，却不证明原实现正确或当前支持域完整。目录命名、全量裁剪授权和交付清单属于原Skill约定。
