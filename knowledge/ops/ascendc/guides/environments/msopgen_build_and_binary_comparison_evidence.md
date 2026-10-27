---
type: Guide
title: msOpGen 构建与逐字节比较的证据范围
description: 抽取生成器、实际加载包及输出身份的验证条件，保留脚本漏检。
tags:
- task.implementation
- topic.compilation
- topic.implementation.operator_registration
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/build-verify.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/examples.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/rules.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/verification-checklist.md
  role: supporting
created_at: '2026-09-11T04:17:28Z'
updated_at: '2026-09-11T04:17:28Z'
platforms: []
---

# msOpGen 构建与逐字节比较的证据范围

来源msopgen gen以JSON原型、ai_core目标和显式-lan cpp选择Ascend C工程，显式语言参数可减少默认模板差异；是否默认py应按实际版本，出现其他目录也可能是复用旧工程。工具可执行、CMake生成、Host编译、Device编译、package生成、安装、运行和比较是不同证据。MSOPGEN_BIN经过symlink/Python入口时不能机械向上三级猜SDK根，要核对实际compiler/include/lib与版本。

拆分Host cpp后构建目标必须收集所有新文件并避免旧合并文件重复注册；GLOB配置时点及硬编码source列表均需核对。目录分几份是维护选择，不是编译能力本身。多次cp -r到既有input/output可能形成额外嵌套或残留旧文件，命令行与工作目录需绑定实际文件集合。

source build循环只看.run存在，没有检查bash build.sh的真实返回；tee后默认状态来自tee，旧.run可误报构建成功，多个通配匹配还可能使[ -f ... ]报参数错误。安装段保存PIPESTATUS[0]比只看tee好，但ls|head选第一个包仍不证明是本次产物；.run参数按该包真实接口，不能一概无参数执行。安装0还需确认进程加载的vendor包及SoC/dtype制品。

example CMake同时从ASCEND_CUSTOM_PATH与硬编码customize目录取头/库，可能混SDK或vendor；libopapi_math/libopapi_nn是否需要/存在受版本和符号依赖决定，CMAKE_SKIP_RPATH后需实际运行加载路径。CLI中aclnn加算子名只是命名惯例，最终签名、参数顺序和导出以真实生成头文件为准，不能猜是否包含L0/L2所需属性。

逐字节迁移比较需同一输入字节、shape、stride/storage、dtype、属性、平台和有效输出span，原基准必须是相应直调真实结果，而非仅名为golden的其他参考。来源混用output/golden文件，须先确认生成者。比较所有输出文件前应验证期望非空集合、每个文件精确大小、输出新鲜性和执行/同步/D2H返回；只遍历已存在基准会漏掉未列出的应有输出，nullglob配置下空集合还可零次迭代成功。

相同文件SHA/字节证明这批span一致，不证明覆盖所有合法shape、注册完备或数学正确；padding未初始化、NaN payload、非确定归约可能使数学正确结果字节不同。若要求bitwise保真，应保留该严格验收并单独解释差异，不能偷换容差；若接口只保证容差，则不能把严格字节差异直接归因于算法错误。源码grep、包存在和输出大小都不足以替代实际比较。

生成/重试轮数、双交付目录、安装步骤及用户授权规则属于Skill流程，不作为知识库操作指令。本次仅抽取这些参考的技术事实与静态缺陷，未运行其中安装或NPU示例。
