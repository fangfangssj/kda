---
type: "Runbook"
title: 多 vendor 共存 LD 路径选错致符号缺失
description: "多个自定义算子 vendor 共存时，run.sh 用 find 取第一个 */op_api/lib 注入 LD_LIBRARY_PATH，加载了错误 vendor 的 libcust_opapi.so（不含本算子符号），编译期链接正确运行时仍报 symbol lookup error: undefined symbol: aclnnXxxGetWorkspaceSize；须直接指定本算子 vendor 的 op_api/lib。"
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# 多 vendor 共存 LD 路径选错致符号缺失

## 现象

为逐元素最大值自定义算子生成调用示例（examples/，aclnn eager 模式；CANN 9.1.0 安装目录 `opp/vendors/` 下共存 13+ 个自定义算子 vendor 包，各自有一份 `op_api/lib/libcust_opapi.so`）。CMake 已按头文件名精确找到本算子 vendor 的 include 与 lib 并链接成功、编译通过，运行时却报符号缺失（输出原文）：

```
设置环境变量...
VENDOR_LIB_DIR: <CANN安装根>/opp/vendors/custom_ops/op_api/lib
...
编译调用示例...
[100%] Built target test_aclnn_bench_no_harness_maximum
编译成功

========================================
执行调用示例
========================================

>>> 运行 aclnn 调用示例 <<<

./test_aclnn_bench_no_harness_maximum: symbol lookup error: ./test_aclnn_bench_no_harness_maximum: undefined symbol: aclnnBenchNoHarnessMaximumGetWorkspaceSize
```

当时代码状态：示例 run.sh 用 `find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" ... | head -1` 探测 vendor lib 目录注入 LD_LIBRARY_PATH，且 CMake 配置带 `-DCMAKE_SKIP_RPATH=TRUE`（rpath 被禁用，运行期完全依赖 LD_LIBRARY_PATH）；find 返回的是字母序靠前的 `custom_ops` vendor（其 libcust_opapi.so 不含本算子符号）。

## 适用场景

- 工程形态：算子仓的调用示例/用户面脚本，在多自定义算子 vendor 共存的 CANN 环境运行。
- API 路径：动态加载 `libcust_opapi.so`（每 vendor 一份）；LD_LIBRARY_PATH 注入逻辑。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）+ CANN 9.1.0。
- 适用粒度：workflow 级——多 vendor 环境的示例/测试运行脚本。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：单 vendor 环境（find 不会选错）；保留 rpath（`CMAKE_SKIP_RPATH=FALSE`）时运行期不依赖 LD_LIBRARY_PATH，本卡问题不出现；一次进程调用多个自定义算子需同时加载多个 vendor lib 的场景未验证。

## 根因

`find ... -path "*/op_api/lib" | head -1` 在多 vendor 目录里选中了字母序靠前的其它 vendor（custom_ops）而非本算子 vendor，LD_LIBRARY_PATH 加载了错误 vendor 的 `libcust_opapi.so`（不含本算子 ACLNN 符号），叠加 `CMAKE_SKIP_RPATH=TRUE` 使链接期正确的库在运行期被绕过。

## 解决配方

修复（before/after verbatim 取自本轨迹编辑事件）：

```bash
# before：find 取第一个匹配的 op_api/lib（多 vendor 下选错）
VENDOR_LIB_DIR=$(find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" -name "lib" -type d 2>/dev/null | grep -v "^${ASCEND_HOME_PATH}/opp/vendors/vendors" | head -1)
if [ -n "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
fi
```

```bash
# after：直接指定本算子 vendor 路径，find 仅作存在性兜底
VENDOR_LIB_DIR="${ASCEND_HOME_PATH}/opp/vendors/bench_no_harness_maximum_custom/op_api/lib"
if [ ! -d "${VENDOR_LIB_DIR}" ]; then
    VENDOR_LIB_DIR=$(find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/bench_no_harness_maximum_custom/op_api/lib" -type d 2>/dev/null | head -1)
fi
if [ -n "${VENDOR_LIB_DIR}" ] && [ -d "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
fi
```

## 设计决策

- 直接拼 vendor 名路径、find 只兜底路径存在性：每算子示例的 vendor 名在包安装时即确定，无需运行期模糊探测。
- 备选"恢复 rpath（去掉 CMAKE_SKIP_RPATH）"被放弃：示例脚本保持与环境变量注入一致的行为更可控，且 rpath 会绑死安装绝对路径。
- 当前方案属 recommended_fastpath。

## 预防

- 示例/测试脚本模板：LD_LIBRARY_PATH 一律写 `<ASCEND_HOME_PATH>/opp/vendors/<本算子_vendor>/op_api/lib`，禁止跨 vendor 的通配 find。
- 代码评审清单：脚本里出现 `find ... op_api/lib | head -1` 即打回（多 vendor 环境必错）。

## 验证信号

问题与验证来源：调用示例的运行验证（修复后清理重建再跑）。

- 修复前：编译成功但运行 `symbol lookup error: undefined symbol: aclnnBenchNoHarnessMaximumGetWorkspaceSize`，且日志显示 `VENDOR_LIB_DIR: .../vendors/custom_ops/op_api/lib`（选错 vendor 的直接证据）（来源平台：950；证据等级：tool_observed）。
- 修复后：`VENDOR_LIB_DIR: .../vendors/bench_no_harness_maximum_custom/op_api/lib`，示例运行输出 `max result[0]=8.000000（max(1,8)）` 等结果正确，"执行结果: PASS"；`run.sh --clean` 清理后重建复跑仍 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- vendor 目录名与算子包名不一致的安装布局需按实际 `<vendor>` 调整（本卡给出的是"直接指定 + 存在性兜底"的模式，不是可移植常量）。
- 若多个 vendor 的 lib 都需同时加载（一次进程调用多个自定义算子），单目录注入不够，本卡未验证该场景。

**未决项**：

- 无。
## 坏实践

- 用通配 find 探测 vendor lib 目录（多 vendor 必选错）。
- 编译期 find 头文件按名字精确匹配、运行期却按目录通配（两侧口径不一致，正是本缺陷形态）。

## 误诊

- 无（日志中 VENDOR_LIB_DIR 指向 custom_ops 与 undefined symbol 直接对应）。
