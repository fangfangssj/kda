---
type: Guide
title: Host ELF 构建选项与加载边界
description: 解释 PIC、PIE、RELRO、栈保护、符号裁剪与搜索路径，限定于真实 Host 制品。
tags:
- task.implementation
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/compile-secure.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-api.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# Host ELF 构建选项与加载边界

以下涉及 Linux Host ELF 编译和加载，适用于相应 Tiling 库或 Host 可执行文件；不要直接套到 NPU Device target。判定依据是实际 target、工具链和 ABI，而不是来源声称“所有 NPU 二进制非 ELF、Device 没有栈”。这些断言并不能由文件用途推出。

- -fPIC 为共享库生成位置无关代码；PIE 可执行程序通常同时使用编译 -fPIE 与链接 -pie。实际 ASLR 还依系统配置，单个编译选项不保证地址每次随机。-fpic/-fPIC 的差异与目标架构和 GOT 限制有关，不只是大小不同。
- -fstack-protector-strong/all 对相应函数添加栈 canary 检查，不能覆盖所有内存错误。Linux 内核栈保护配置是另一 target 的构建设置，不能当成 NPU Kernel 选项。
- -Wl,-z,relro 使相应重定位区域只读，结合 -Wl,-z,now 提前绑定可形成更完整的 GOT 保护；立即绑定将符号解析成本前移，也可能提前暴露未解析依赖，不是修复循环依赖的万能选项。
- -Wl,-z,noexecstack 声明不需要可执行栈；依赖执行栈 trampoline 的扩展要单独兼容性分析。普通 inline 函数不因此失效，来源“内嵌函数会出错”过宽。
- strip/-s 可减小非运行必需符号和调试数据，但动态符号与重定位仍可能参与加载；不能宣称链接后所有符号表都无用。发布符号裁剪需保留排障/回溯对应信息，未完成链接的 .o/archive 不能盲目按最终制品裁剪。
- RPATH 与 RUNPATH 的搜索优先级、传递性不同，new-dtags 会影响实际动态段，不能统称都高于 LD_LIBRARY_PATH。路径可写性、$ORIGIN 展开、依赖链和实际加载结果共同决定可控性；来源禁止 rpath 是工程策略，不是加载器正确性的通用条件。

dlopen/dlclose 与 thread_local 的危险取决于是否仍有调用、回调、TLS 析构或借用指针，以及加载器真实卸载行为。不能断言一次 dlclose 必然立刻卸载所有相关代码；需要保持库与对象寿命覆盖使用阶段，具体平台需验证。

API 日落、头拆分和库更名要绑定目标版本的声明、迁移说明及删除期限。来源列出的 GroupedMatmul 版本替换、GetVersion 替换、op_proto→op_graph、libopapi 分库只是其历史迁移线索，本批未核验最新清单，不将示例期限发布为当前事实。替换时还需核对参数、语义、link target 和真正加载的符号，不能仅换名字。
