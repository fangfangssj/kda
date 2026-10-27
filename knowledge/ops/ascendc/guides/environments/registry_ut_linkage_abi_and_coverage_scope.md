---
type: Guide
title: 注册 UT 的链接 ABI 与覆盖率范围
description: 保留动态注册库保留、依赖身份及覆盖率统计的边界。
tags:
- task.testing
- topic.implementation.tiling
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/CMakeLists.txt
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/cmake/BuildGoogleTest.cmake
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_api/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_host/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/run.sh
  role: supporting
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# 注册 UT 的链接 ABI 与覆盖率范围

来源将Host定义/推导/Tiling构建为独立.so并AddSoToRegistry加载，以whole-archive保留静态注册对象、no-as-needed保留需要的动态依赖；这些选项需要有界使用并核对重复符号和库顺序。UT executable没有对Host .so显式add_dependencies，默认all可构建二者，单独构建UT目标却不保证.so已生成。RPATH $ORIGIN与实际部署目录须对应。

_GLIBCXX_USE_CXX11_ABI=0必须与实际CANN库、GoogleTest及跨边界STL类型匹配，源码统一使用0是此配置选择，不意味着所有系统GTest都使用1，也不能以“直接链接”宣称没有任何ABI问题。C++17 inline静态成员支持属于语言要求；aarch64 include硬编码需要按Host ISA调整。

SOC_VERSION以严格Ascend950选择arch35，否则走arch22；两套header目录同时在include路径时还需核对真实命中。Shell未传SOC_VERSION，默认值和CMake缓存可能决定测试平台，命令外层--soc不自动进入子构建。--ophost/--opapi只控制运行，顶层仍add_subdirectory并make全部，未运行目标不能计入执行覆盖。

GoogleTest版本1.14.0、JSON3.11.3和镜像URL固定名字不足以固定下载内容；TLS_VERIFY OFF与缺URL_HASH没有校验制品身份，既有tarball也可能残缺。ExternalProject导入target依赖需保证实际库/头先生成；$(MAKE)写法还受生成器约束。名为intf_llt_pub_asan_cxx17的接口只有coverage相关选项且未见链接使用，不能凭名称声称开启ASan。

ENABLE_GCOV需在被统计源及其链接制品生效，lcov排除common/test等后分母应明确。执行未涵盖某目标、跳过测试或缺lcov时没有该覆盖率证据；脚本总体0不生成凭空报告。set-e使失败后的手写$?分支可能不可达，不能按打印模板判断实际控制流。清理、下载和测试调度仍留Skill。
