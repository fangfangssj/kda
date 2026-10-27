---
type: Guide
title: Blaze 头文件、宏与三种链接上下文
description: 保留同源模板依赖、宏值、头搜索及普通 Host/ASC Host/Device 的独立验证。
tags:
- task.debugging
- topic.compilation
- topic.programming_model
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-cmake-build-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/troubleshooting/compile-troubleshooting.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/launcher/launcher-development.md
  role: supporting
created_at: '2026-09-11T02:57:41Z'
updated_at: '2026-09-11T02:57:41Z'
platforms: []
---

# Blaze 头文件、宏与三种链接上下文

同工程可能同时包含普通CXX Host、ASC联合编译的Host阶段及Device阶段。设备宏只描述当前编译上下文；将Host声明包在设备宏里会让普通CXX不可见，把Host-only实现暴露到Device又可能导致设备缺符号。文件后缀、CMake缓存或某一侧编译通过，均不能证明三种上下文已采用相同API。

-DNAME=0仍使#ifdef NAME成立，#if NAME才按值判断。CPU debug、Host selector和Device selector需按真实预处理结果核验。普通inline/constexpr不自动产生Device函数实体，运行参数helper仍要有正确的设备可见定义；模板返回类型推导也需在调用处拥有所需定义。不能用dummy库或伪声明掩盖真实未解析符号。

Blaze/tensor_api的模板、内部相对include和公共头应来自兼容版本。只平铺同名头可能破坏目录拓扑；混入工具链或另一副本同名声明可能改变constexpr、宏或模板特化。按实际编译命令和include trace查清来源，target级include/compile/link配置能缩小影响范围。-iquote仅针对相应引号include搜索，不解决尖括号、ABI或版本不兼容。

未解析符号要区分定义未编入target、参数/linkage不一致、archive中存在却未参与链接，以及静态库的传递依赖/顺序缺失。头可见不证明库已链接，库含符号不证明当前执行文件加载它。CANN安装根需非空且来自一致工具链，环境变量与package导出值只是发现路径，不能容许空根退化为/include或混入另一套库。

Host、Tiling、Device helper的命名空间可避免CeilDiv/Align等同名冲突；翻译单元级using namespace会重新扩散查找范围。在namespace内include外部头会改变声明归属并可能破坏类型/ABI，不是可靠隔离方式。Kernel入口linkage与作用域应保持真实工具链绑定。

依赖输入/Golden的Host测试必须有对应生成与完整性结果，零测试收集不算通过。平台查询为空时先判空；可选诊断可以记录未执行，合法Tiling必需资源缺失则不能伪造零或默认值继续。数据缺失/截断属于测试装配或I/O失败，不应混称Kernel失败。本批未编译这些模板，目录和工作流门禁继续留于Skill。
