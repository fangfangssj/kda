---
type: "Runbook"
title: "aclnn 直调符号望文生义三处编译失败"
description: "C++ ST 直调 aclnn 两段式接口时按直觉拼符号名会三处连错：executor 销毁实名 aclDestroyAclOpExecutor（aclnn/acl_meta.h）、bf16 枚举实名 ACL_BF16（=27，acl_base_rt.h）、ACLNN_ERR_* 错误码宏在 aclnn/opdev/op_errno.h 且 vendor 安装头链不引入（名字写对也拿不到，可本地 constexpr 对齐数值）。"
tags: [task.testing, topic.implementation.api_layer.c_api, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# aclnn 直调符号望文生义三处编译失败

## 现象

为融合算子（Add + RMSNorm + 动态量化类，一进一出 elementwise）编写 C++ ST 测试（Real 模式：直调 aclnn 两段式接口 `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`，NPU 输出与 CPU golden 比对）。测试工程 CMake 把 include 指向 vendor 安装目录（`$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include`）。首次 `bash tests/st/run.sh --real` 编译测试程序时报一批符号错误（同一轮输出）：

- `ACLNN_ERR_INNER_NULLPTR` 未声明——该错误码宏（值 561103）定义在 CANN 主 include 的 `aclnn/opdev/op_errno.h`，vendor 安装头链只引 `aclnn/acl_meta.h`，不带出来；
- `ACL_BFLOAT16` 未声明——acl 基础枚举里 bf16 的实名是 `ACL_BF16`（=27，`acl/acl_base_rt.h:157`），没有 `ACL_BFLOAT16` 这个拼写；
- `aclDestroyExecutor` 未声明——executor 的销毁函数实名 `aclDestroyAclOpExecutor`（`aclnn/acl_meta.h`），与 `aclDestroyTensor/aclDestroyScalar` 等同族声明在一起。

## 适用场景

- 工程形态：手写 C++ ST/集成测试直调 aclnn 两段式接口（不经过 TTK 等 harness）。
- API 路径：`aclnn/acl_meta.h`（aclnnStatus、aclDestroyTensor/aclDestroyAclOpExecutor 等）、`acl/acl_base_rt.h`（ACL_* 数据类型枚举）、`aclnn/opdev/op_errno.h`（ACLNN_ERR_* 错误码）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510），CANN 9.1.0。
- 适用粒度：platform_wide——aclnn C API 符号面问题，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：经 TTK/harness 的评测不受影响（不自己链接这些符号）；其它 CANN 版本的宏布局未逐一核对。

## 根因

测试代码按"望文生义"的名字使用 aclnn C API 符号：错误码宏、bf16 枚举、executor 销毁函数三处名字与真实定义不符；其中错误码宏还叠加一层可用性问题——vendor 安装头只 include `aclnn/acl_meta.h`，不含 `opdev/op_errno.h`，即使名字写对也拿不到宏（需从 CANN 主 include 路径取，或本地定义常量）。

## 解决配方

**修复 1：错误码宏改本地常量**（op_errno.h 未随安装头链暴露，测试内自定义避免依赖）：

```cpp
// 本地错误码（op_errno.h 未通过安装的头文件传递，使用本地定义避免依赖）
constexpr aclnnStatus ERR_INNER_NULLPTR = static_cast<aclnnStatus>(561103);
```

调用点 `return ACLNN_ERR_INNER_NULLPTR;` 相应替换为 `return ERR_INNER_NULLPTR;`。若需要全量错误码语义，也可直接 include CANN 主 include 路径（`${CANN_ROOT}/include`）下的 `aclnn/opdev/op_errno.h`。

**修复 2：bf16 枚举实名**：

```cpp
// 修复前
aclDataType inDtype = (tc.dtype == Dtype::FP16) ? ACL_FLOAT16 : ACL_BFLOAT16;
// 修复后
aclDataType inDtype = (tc.dtype == Dtype::FP16) ? ACL_FLOAT16 : ACL_BF16;
```

**修复 3：executor 销毁函数实名**（全部出现处替换）：

```cpp
// 修复前
aclDestroyExecutor(executor);
// 修复后
aclDestroyAclOpExecutor(executor);
```

写码前的符号核对命令：

```bash
grep -n "aclDestroy" $ASCEND_HOME_PATH/include/aclnn/acl_meta.h   # executor 销毁为 aclDestroyAclOpExecutor
grep -n "ACL_BF" $ASCEND_HOME_PATH/include/acl/acl_base_rt.h      # ACL_BF16 = 27
grep -rn "ACLNN_ERR_INNER_NULLPTR" $ASCEND_HOME_PATH/include/aclnn/opdev/op_errno.h
```

## 设计决策

- 错误码选"本地 constexpr 常量（值对齐 op_errno.h 的 561103）"而不是 `#include "aclnn/opdev/op_errno.h"`：测试的失败路径只需非零返回值区分，避免把 CANN 内部头拉进 vendor 头的编译单元。
- bf16 枚举值经 `acl/acl_base_rt.h` grep 确认（ACL_BF16 = 27），与 `ge::DT_BF16 = 27` 数值一致，正好和 D2D 的 dtype 映射对上。

## 系统信息

- 硬件/软件：Ascend950PR（ascend950 / DAV_3510），CANN 9.1.0；测试工程 include 指向 vendor 安装目录。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（三处符号修复点）、`tests/st/CMakeLists.txt`（include 与链接配置）。

## 预防

- 写 aclnn 直调测试前先 grep `aclnn/acl_meta.h` 拿销毁/查询函数实名，grep `acl/acl_base_rt.h` 拿 ACL_* 枚举实名，别按直觉拼写。
- ACLNN_ERR_* 宏一律从 `aclnn/opdev/op_errno.h` 取（或本地定义并注明数值来源）；不要假设 vendor 安装头链会带出错误码头。

## 验证信号

以下信号来自该 ST 工程的真实编译过程：

- 修复前：Real 模式编译报 ACLNN_ERR_INNER_NULLPTR / ACL_BFLOAT16 / aclDestroyExecutor 等符号未声明（来源平台：950；证据等级：tool_observed）。
- grep 确认符号真实位置：`acl_meta.h` 含 `aclDestroyTensor/aclDestroyScalar/...`（executor 销毁为 aclDestroyAclOpExecutor）、`acl_base_rt.h:157: ACL_BF16 = 27`、ACLNN_ERR_INNER_NULLPTR=561103 在 opdev/op_errno.h（来源平台：950；证据等级：tool_observed）。
- 三处替换后 Real 模式测试编译成功并进入执行（随后暴露运行期 161001，属另一问题）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 错误码数值 561103 对应 CANN 9.1.0 的 op_errno.h；跨版本使用本地常量前应重新核对数值。
- 本卡只覆盖"符号名/宏来源"层；安装头与源码头签名不一致导致的运行期问题是另一类问题。
- 未验证，推断可迁移：CANN 9.x 各版本的 acl_meta.h / acl_base_rt.h 符号面。

**未决项**：

- 无。
## 坏实践

- 按直觉拼 aclnn 符号名（ACL_BFLOAT16、aclDestroyExecutor 都是看似合理但不存在的名字）。
- 在 vendor 安装头的编译单元里假设 ACLNN_ERR_* 宏可用。

## 误诊

- 无。
