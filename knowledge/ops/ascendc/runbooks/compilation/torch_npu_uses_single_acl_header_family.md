---
type: "Runbook"
title: "torch_npu 只保留一族 ACL 头文件"
description: "direct-launch .asc 同时引入 CANN toolkit acl_rt.h 与 torch_npu bundled ACL 时，同一 aclmdl 函数出现不兼容声明；改用 torch_npu include root 下的 third_party ACL 后解除冲突。"
tags: ["task.debugging", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# torch_npu 只保留一族 ACL 头文件

## 现象

在 A3 direct-launch 的 `gather.asc` 中同时包含 `kernel_operator.h`、CANN toolkit 的 `acl/acl_rt.h`、torch 与 `NPUStream.h` 后，`cmake --build` 在任何 kernel 实例化之前停止：

```text
torch_npu/include/third_party/acl/inc/acl/acl_mdl.h:1652:30:
error: conflicting types for 'aclmdlRIDebugJsonPrint'
aclError aclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char* path);

CANN/include/acl/acl_rt.h:4730:30: note: previous declaration is here
aclError aclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char *path, uint32_t flags);
```

同一翻译单元从 torch_npu 依赖链加载 bundled ACL，又显式加载 toolkit ACL；两套版本对同一符号给出不同参数表。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`aclmdlRIDebugJsonPrint`、`acl_rt.h`、`acl_mdl.h`、`torch_npu` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖相同 torch_npu、ACL 版本和 include/link 组合。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

torch_npu 已经通过自身 include tree 固定一套 bundled ACL，翻译单元再显式包含 toolkit `acl/acl_rt.h` 会混合两个 ABI 版本，并对 `aclmdlRIDebugJsonPrint` 产生不兼容声明。

## 解决配方

使用 CMake 已加入的 torch_npu include root，统一包含 `third_party/acl/inc/acl/acl_rt.h`；不要混用 toolkit ACL 头，也不要手写 extern 声明。

### before

```cpp
#include "kernel_operator.h"
#include "acl/acl_rt.h"

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

using namespace AscendC;
```

```cpp
static uint32_t QueryAivCores(int32_t deviceId) {
    int64_t aiv = 0;
    aclError rc = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId),
                                    ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
    if (rc != 0 || aiv <= 0) {
        // Fallback: a3 verified 40 AIV. Range-checked kernel tolerates fewer/more.
        return 40;
    }
    return static_cast<uint32_t>(aiv);
}
```

### after

```cpp
#include "kernel_operator.h"

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
// Use the ACL headers bundled inside torch_npu (NOT CANN's acl/acl_rt.h — it conflicts with the
// bundled acl_mdl.h). The torch_npu include root is configured in CMakeLists. This provides
// aclrtGetDevice, aclrtGetDeviceInfo, ACL_DEV_ATTR_VECTOR_CORE_NUM, aclError, ACL_SUCCESS.
#include "third_party/acl/inc/acl/acl_rt.h"

using namespace AscendC;
```

```cpp
static uint32_t QueryAivCores() {
    int32_t deviceId = 0;
    aclError rc1 = aclrtGetDevice(&deviceId);
    TORCH_CHECK(rc1 == ACL_SUCCESS,
                "gather: aclrtGetDevice failed rc=", static_cast<int64_t>(rc1));
    int64_t aiv = 0;
    aclError rc2 = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId),
                                      ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
    TORCH_CHECK(rc2 == ACL_SUCCESS && aiv > 0,
                "gather: aclrtGetDeviceInfo(dev=", static_cast<int64_t>(deviceId),
                ", VECTOR_CORE_NUM) failed rc=", static_cast<int64_t>(rc2),
                " aiv=", aiv);
    return static_cast<uint32_t>(aiv);
}
```

## 设计决策

1. 统一以 torch_npu bundled ACL 为 host adapter 的声明来源，因为本翻译单元已经依赖 torch_npu headers。
2. 不使用手写 `extern "C"` 猜签名；中间方案未进入成功构建，属于 `banned`。
3. after 同时把核数查询改为 fail closed；这是独立安全修复，不是解除头冲突的根因。

## 系统信息

- Toolchain：CANN 9.2.0、bisheng、torch_npu；平台：a3/dav-2201；
- CMake 已把 torch_npu include root 加入 target include path；
- 同轮还有 `GetDevice` 与局部变量重定义等独立错误，未合并为本卡根因。

## 完整代码

解决配方已内联完整 include consumer：翻译单元 preamble 与完整 `QueryAivCores` before/after。

## 预防

- 先确认框架依赖自带的 ACL include root，再选择 runtime 头；同一翻译单元只保留一个版本族。
- 编译器同时打印两个绝对 header 路径时，逐个核对其版本来源与函数签名。
- 不以 forward declaration 绕过 ABI 冲突。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：混用头文件后出现 aclmdlRIDebugJsonPrint 声明冲突及 GetDevice 缺失。（来源平台：a3；证据等级: tool_observed）。
- 修复后：统一头文件族后 build exit=0，目标对象与共享库完成构建。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 最终成功构建同时修复了其他 host 错误，因此本卡不把整个 build success 归因于单一 include 变化；头冲突本身由编译器双路径诊断直接证明。
- 非 torch_npu 工程可能应使用 toolkit ACL；不要机械复制 bundled 路径。
- 其他版本可能已统一签名，仍应以目标环境实际 include tree 为准。

## 坏实践

- 同时包含 toolkit ACL 与框架 bundled ACL。
- 为绕过冲突手写函数原型或修改第三方头。
- 只删除报错 API 调用，却保留混合 ABI 头。

## 误诊

该错误在头文件解析阶段发生，与 kernel dtype、tiling、UB 或 runtime 设备状态无关。
