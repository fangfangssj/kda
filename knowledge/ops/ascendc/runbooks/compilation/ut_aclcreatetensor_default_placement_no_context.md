---
type: "Runbook"
title: "op_api UT 默认 device placement 无上下文误拒"
description: op_api UT 的输入 tensor 用 aclCreateTensor 创建（默认 placement=kOnDeviceHbm）但 data 是 host 指针，且 UT main 未初始化 ACL runtime 时，被测的 aclrtMemcpy(DEVICE_TO_HOST) 失败返回 161002 误拒合法输入；修复为 UT main 补 aclInit/aclrtSetDevice，且产线代码禁止"aclrtMemcpy 失败回退 memcpy"。
tags: [task.testing, topic.memory, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# op_api UT 默认 device placement 无上下文误拒

## 现象

为一个 MoE token 重排自定义算子的 op_api 层新增数据内容校验（读 `expertTokenNumPerRank` 数据校验 Sum==A、元素>0，device 数据经 `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` 拷到 host 缓冲，见同批《op_api 补数据内容校验》卡）后，代码在 ST Real 模式（NPU 上、数据真在 device）按预期工作，但跑 op_api UT 时，4 个**正常路径**用例全部失败——这些用例的 num 数据是合法的（16 个元素全 64，sum=1024=A），却被判参数无效：

```text
[ RUN      ] AclnnBenchNoHarnessMoeReRoutingTest.ValidParamsFp16Int32NoScales
.../test_aclnn_bench_no_harness_moe_re_routing.cpp:528: Failure
Expected: (ret) != (161002), actual: 161002 vs 161002
[  FAILED  ] AclnnBenchNoHarnessMoeReRoutingTest.ValidParamsFp16Int32NoScales (0 ms)
（ValidParamsBf16Int32WithScales / ValidParamsInt8Int64NoScales / ValidParamsFp16Int64WithScales 同样失败）
```

中间曾用"aclrtMemcpy 失败回退直接 memcpy"让 UT 全绿；代码检视（SEC-4.1 外部输入合法性校验 / SEC-3.5 指针判空条款）移除该回退后，同 4 个用例立刻回归失败——客观证明此前"通过"依赖回退（UT 里 data 实际是 host 内存，回退碰巧拷得动）。探针实测补齐根因：`[DEBUG] num placement = 0`，对照 `include/exe_graph/runtime/tensor_data.h` 枚举 `kOnDeviceHbm = 0 / kOnHost = 1 / kFollowing = 2 / kOnDeviceP2p = 3`——`aclCreateTensor` 创建的 tensor **默认 placement 是 device 侧**，即便 data 指针是 host 内存；且 `aclTensor` 无公开 `SetPlacement`（rg 全 CANN include 确认只有 `gert::Tensor`/`gert::TensorV2` 有），UT main 又没有 `aclInit`/`aclrtSetDevice`——被测的 `aclrtMemcpy(DEVICE_TO_HOST)` 因无 NPU 上下文而失败。

## 适用场景

- 工程形态：自定义算子 op_api 单元测试（gtest，CMake 直编 op_api 源码、链 libacl_rt.so），被测路径含 `aclrtMemcpy` 或设备语义 API。
- API 路径：`aclCreateTensor` / `aclTensor::GetPlacement()` / `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` / `aclInit` / `aclrtSetDevice`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有 op_api UT，只要被测代码路径含 aclrtMemcpy 且输入 tensor 由 aclCreateTensor 创建。
- 路线优先级：recommended_fastpath（UT main 初始化 NPU 上下文）；memcpy 回退路线 banned。
- 明确不适用 / 未验证场景：op_host tiling UT 用 gert::Tensor，可显式 `SetPlacement(kOnHost)`，不受此坑影响；无 NPU 设备的 CI 环境不能用本方案（mock 路线未验证）。

## 根因

op_api UT 的输入 tensor 用 `aclCreateTensor` 创建（默认 placement=`kOnDeviceHbm`）但 data 是 host 指针，且 UT main 未初始化 ACL runtime——被测的 `aclrtMemcpy(DEVICE_TO_HOST)` 因无 NPU 上下文而失败。此前该失败一直被回退分支掩盖，4 个用例"假通过"。两层结论：

1. placement 元数据在 UT 这种 host 数据构造下并不反映真实数据位置，不能单独作为读取方式的依据；
2. aclrtMemcpy 的失败原因无法在 op_api 侧区分——"无 NPU 上下文/数据实际在 host"（回退碰巧可用）与"内存映射失效/非法 device 指针"（回退段错误）返回的是同样的失败码，一律回退 host memcpy 等于用"碰巧可读"赌指针可达性，同时掩盖调用环境缺陷。

## 解决配方

第一步（banned 路线移除）：`CheckDataValid` 的 aclrtMemcpy 失败分支从"回退 memcpy"改为"报错返回"（verbatim）：

```cpp
// 修复前（含风险回退，掩盖环境缺陷 + 真实 device 指针下段错误）：
    if (isOnDevice) {
        // Device 数据：通过 aclrtMemcpy 拷到 Host
        // 若 aclrtMemcpy 失败（如 UT 环境无 NPU 上下文），回退到直接 memcpy
        auto ret = aclrtMemcpy(hostBuf.data(), totalBytes, tensorData, totalBytes,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            // aclrtMemcpy 失败，回退到直接 memcpy（适用于 Host 数据或无 NPU 上下文的 UT 环境）
            std::memcpy(hostBuf.data(), tensorData, totalBytes);
        }
    } else {
        // Host 数据：直接拷贝
        std::memcpy(hostBuf.data(), tensorData, totalBytes);
    }

// 修复后（失败即报错返回）：
    if (isOnDevice) {
        // Device 数据：通过 aclrtMemcpy 拷到 Host
        // aclrtMemcpy 失败时返回错误码，禁止回退 memcpy（Device 指针直接 memcpy 会导致段错误，SEC-4.1/SEC-3.5）
        auto ret = aclrtMemcpy(hostBuf.data(), totalBytes, tensorData, totalBytes,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclrtMemcpy failed for device data, ret=%d.", static_cast<int>(ret));
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        // Host 数据：直接拷贝
        std::memcpy(hostBuf.data(), tensorData, totalBytes);
    }
```

第二步（正解）：在 `tests/ut/op_api/test_op_api_main.cpp` 初始化 NPU 上下文（verbatim，取自修复写入事件）。

修复前：

```cpp
#include <iostream>
#include <gtest/gtest.h>

using namespace std;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

修复后：

```cpp
#include <iostream>
#include <gtest/gtest.h>
#include "acl/acl.h"

using namespace std;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    // 初始化 ACL runtime，使 aclrtMemcpy 等 API 可用（CheckDataValid 读取 device 数据需要）
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        std::cerr << "[WARN] aclInit failed, ret=" << ret << std::endl;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[WARN] aclrtSetDevice(0) failed, ret=" << ret << std::endl;
    }
    int runRet = RUN_ALL_TESTS();
    aclrtResetDevice(0);
    aclFinalize();
    return runRet;
}
```

CMake 侧无需改动：op_api UT 的 CMakeLists 已链接 `libacl_rt.so` 且 include 路径含 `acl/acl.h`。关键事实：**有 NPU 上下文时，`aclrtMemcpy(DEVICE_TO_HOST, src=host指针)` 也能成功**——ACL 内部对 host 地址容错（按 host 拷贝处理），所以 UT 不必把数据真的搬到 device，最小改动即可让被测路径真正执行。

诊断链（三步，可复用）：

1. 确认错误码语义：`rg -n "161002|ACLNN_ERR_PARAM_INVALID" $ASCEND_HOME_PATH/include/aclnn/`——锁定"读数据"而非"算子执行"。
2. 探针实测 placement：用例里临时打印 `static_cast<int>(num.Get()->GetPlacement())`，输出 0 即 kOnDeviceHbm（用后移除）。
3. 机制链闭环：placement=kOnDeviceHbm → 走 aclrtMemcpy 分支 → 无 NPU 上下文 → 失败 → 161002。

## 设计决策

- 为什么选"UT main 初始化 NPU 上下文"而不是"把 tensor 标成 Host placement"：`aclTensor` 无公开 `SetPlacement`（rg 确认），UT 侧做不到；`gert::Tensor` 才有。
- 为什么不把输入改成真实 device 内存（aclrtMalloc + H2D 拷贝）：改动面大（TestTensorFactory 增加设备路径、用例逐个切换、TearDown 释放），且实测有上下文时 ACL 对 host 指针容错，最小改动即可。
- 放弃的候选：mock aclrtMemcpy（链接 mock 库或 LD_PRELOAD，实现复杂）；在 CheckDataValid 里按 aclrtMemcpy 错误码区分"无上下文可回退 / 真实错误不可回退"（无上下文时 data 也可能是 device 指针，不安全，且会在产线代码里留测试后门）。
- 错误码选择 `ACLNN_ERR_PARAM_INVALID`（161002）：读取失败即视为无法确认参数合法性，与该函数其他校验失败的返回一致。
- 当前方案路线档位：recommended_fastpath（UT 上下文）；产线 memcpy 回退 banned。

## 预防

- 写 op_api UT 时，只要被测路径含 `aclrtMemcpy`/设备语义 API，main 里先 `aclInit` + `aclrtSetDevice`（容忍 `ACL_ERROR_REPEAT_INITIALIZE`），结束后 `aclrtResetDevice` + `aclFinalize`。
- 记住 `aclCreateTensor` 的默认 placement 是 `kOnDeviceHbm`（枚举值 0）——"data 是 host 内存"不代表"placement 是 host"。
- 依赖"期望错误码"的负向用例要警惕"因 aclrtMemcpy 失败而意外通过"——无上下文时它们会在数据校验前就返回同一个错误码，用例失效但不报错。
- 评审时把"注释里写着'适用于 UT 环境'的产线回退分支"当作红旗——测试环境适配应发生在测试代码里。

## 验证信号

来源说明：该算子 op_api 数据校验开发的 UT 调试会话 + 检视问题修复会话（Ascend950PR，CANN 9.1.0）：先全量复现 4 例误拒，加回退让 UT 通过，检视移除回退后回归复现，探针定位 placement，最终 UT main 补上下文复验。

- 失败态：4 个 ValidParams 用例（合法 num 数据 sum=1024=A）全部 `actual: 161002` 失败，单跑复现；OP_LOGE 错误文本不出现在 stdout（来源平台：950；证据等级：tool_observed）。
- 回退掩盖证明：移除回退后同 4 个用例立刻回归 161002，而回退存在时同批用例通过——单变量对照证明此前通过依赖回退（"修复前通过"依据任务验收前提"UT 无回归" + 回退代码 verbatim + 修复后失败的受控对比推断）（来源平台：950；证据等级：tool_observed〔修复后失败 + 修复前代码；前置通过为受控对比推断〕）。
- 探针观测：`[DEBUG] num placement = 0`（kOnDeviceHbm），`aclTensor` 无公开 SetPlacement（rg 全 CANN include 确认）（来源平台：950；证据等级：tool_observed）。
- 修复后单用例：`ValidParamsFp16Int32NoScales` 从失败转为 `[ OK ]`（含真实 aclrtMemcpy 执行）（来源平台：950；证据等级：tool_observed）。
- 修复后全量：op_api UT 30/30 通过、op_host UT 90/90 不受影响；最终整体验证 ST L0+L1+L2 77/77 + UT 120/120（来源平台：950；证据等级：tool_observed）。
- 语义恢复：SumNotEqualsAFailure（numData 全 1，sum=16≠A）/ ElementNotPositiveFailure（numData 全 0）真正读到数据后才返回 161002，不再是"aclrtMemcpy 失败的意外通过"（来源平台：950；证据等级：tool_observed〔修复后 ValidParams 首用例含真实 aclrtMemcpy 耗时、负向用例与正向用例行为分化〕）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.1.0 实测；"aclrtMemcpy 对 host 指针容错"是 9.1.0 观测到的行为，其他版本未验证（推断可迁移）。
- 无 NPU 设备的环境（纯 CPU CI）不能用本方案，需 mock aclrtMemcpy 或分层测试（未尝试）。
- `ACL_ERROR_REPEAT_INITIALIZE` 容错只覆盖重复初始化；其他 aclInit 失败模式按 WARN 放行后用例仍可能失败。
- "device 指针 memcpy 段错误"的风险结论来自代码路径分析与检视推导，本轨迹未实测复现段错误（无"真实 device 指针 + aclrtMemcpy 失败"的组合用例）；"回退掩盖失败"则有回归对照的实测支撑。

**未决项**：

- 无 NPU 的 CI 环境如何跑此类 op_api UT（mock aclrtMemcpy 或测试分层）未尝试。
- `aclrtMemcpy` 对 host 指针容错的确切判定机制（地址区间识别还是 fallback 拷贝）未深挖，仅观测到行为。
## 坏实践

- op_api UT 无 NPU 上下文，靠被测代码里的回退分支"碰巧"通过——回退一移除立刻回归。
- 负向用例（期望错误码）在无上下文环境下"全绿"就认为测试有效——可能全部走的是 aclrtMemcpy 失败捷径。
- 在产线 op_api 代码里为 UT 留 memcpy 回退后门；把"拷贝失败"与"参数非法"共用同一个返回码（第一版实现的教训）。
- 在 op_api 校验里无条件信任 `GetPlacement()`：UT/桩环境构造的张量元数据未必反映真实数据位置。

## 误诊

- 最初假设一：4 个失败是"L0 算子 API 在 UT 环境未注册导致"的预存在失败——被错误码核对否定（161002 恰是 CheckDataValid 的返回值，且失败用例全是会读 num 数据的 ValidParams）。
- 最初假设二：`aclCreateTensor` 默认 placement 可能是 kOnHost，失败另有原因——被探针实测否定（placement=0）。
- 排查时一度怀疑 UT 用例数据没传对或 OP_LOGE 输出丢失导致看错失败分支；单用例过滤复现 + 代码走查锁定在 aclrtMemcpy 失败路径（该路径由回退修复前后的对照反证，返回码未直接观测）。
