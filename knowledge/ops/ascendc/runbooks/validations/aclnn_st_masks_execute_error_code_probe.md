---
type: "Runbook"
title: aclnn ST 掩盖 execute 错误码：直连探针
description: aclnn ST 用例集体失败但详情只有 golden 对比/全 0 信息、无 GetWorkspaceSize=/aclnn= 错误码时，自建 ST 框架在 s1==0 时仍跑 golden 对比并覆盖 ret.detail，execute 阶段错误码被掩盖；最小直连探针逐返回码打印可还原真实错误码（如 361001）。
tags: [task.debugging, topic.implementation.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T00:14:22Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# aclnn ST 掩盖 execute 错误码：直连探针

## 现象

在 Ascend 950（CANN 9.1.0）上跑自定义算子的 aclnn ST 套件（自建 `test_aclnn.cpp`，两段式调用 `aclnnXxxGetWorkspaceSize` + `aclnnXxx`，测试先把输出 device buffer `aclrtMemset` 清零再对比独立 golden）：新分支的用例集体失败，但失败详情**全部是对比类信息**（`invariant sum_to_one: sum=0.000000`、`isnan mismatch at 0`），**没有任何 `GetWorkspaceSize=` 或 `aclnn=` 错误码**。表面看像"kernel 跑了但算错"，实际是 kernel 根本没执行——输出仍是测试预清零的 buffer。这种"全 0 + 无错误码"的形态下，ST 框架本身把真实返回码吞掉了，需要绕过它直连取回。

## 适用场景

- 工程形态：aclnn 两段式接口的自定义算子 ST 联调（自建测试可执行文件 + run.sh），故障形态为"输出全 0/全默认值 + 详情无错误码"。
- API 路径：`aclnnXxxGetWorkspaceSize` / `aclnnXxx` / `aclrtSynchronizeStream` / `aclrtMemcpy`；ST 框架的 `RunRet ret{false,""}` 详情拼接逻辑。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`workflow_only` 之上的最小技术粒度：两段式 aclnn 调用链的错误码观测方法，平台无关成分高；归 `workflow` 层 debug 配方。
- 路线优先级：`recommended_fastpath`（出现该形态先跑探针，再决定查数值还是查执行链）。
- 明确不适用/未验证场景：ST 框架本身已把 s2 错误码透传到详情的工程（无需探针）；AICPU/图模式执行未验证。

## 判别方法

**触发条件**：ST 用例批量失败，详情只含 golden 对比/invariant 字样、无 `GetWorkspaceSize=`、无 `aclnn=`；且失败用例输出为全 0（测试有 memset 清零的前提）。

**为什么 ST 看不到错误码**（框架源码级根因，`runOneCase` 结构）：

```cpp
    aclnnStatus s1 = aclnnXxxGetWorkspaceSize(x, t.dim, y, &wsSize, &exec);
    RunRet ret{false, ""};
    if (s1 != 0) {
        ret.detail = std::string("GetWorkspaceSize=") + std::to_string(s1);
    } else {
        // ...
        aclnnStatus s2 = aclnnXxx(ws, wsSize, exec, stream);
        if (s2 != 0) {
            ret.detail = std::string("aclnn=") + std::to_string(s2);   // ← 写入了错误码
        } else {
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(outBytes.data(), bytes, devY, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    // ...
    if (s1 == 0) {                       // ← 只看 s1：s2 失败时依然进入
        // ... golden 对比 ...
        ret.pass = cr.pass;
        ret.detail = cr.detail;          // ← 覆盖掉 "aclnn=..."，露出 "sum_to_one: sum=0.000000"
    }
```

`s1==0` 而 `s2!=0` 时，golden 对比在零输出上照跑并**覆盖** `ret.detail`——错误码被 `sum=0.000000` 顶掉。两段式 aclnn 的 execute 阶段错误（kernel binary 选不到、执行异常等）正落在这个盲区。

**探针**：最小直连程序（单用例、逐返回码打印、非零元素计数）。以下为实测用探针全文（S3 fp16 `[4,2048,50272]`，按需替换 shape/dtype/算子名）：

```cpp
// Minimal test: run S3_fp16 and check aclrtSynchronizeStream return
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_bench_harness_softmax.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    // S3 [4, 2048, 50272] fp16
    int64_t shape[] = {4, 2048, 50272};
    int64_t N = 4 * 2048 * 50272;
    size_t bytes = N * 2; // fp16

    void *devX, *devY;
    aclrtMalloc(&devX, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(devY, bytes, 0, bytes);

    // Fill x with some data
    std::vector<uint8_t> xhost(bytes, 0);
    for (int64_t i = 0; i < N; i++) {
        // simple fp16 values: 0.5, 1.0, -0.5, 0.0 repeating
        uint16_t vals[] = {0x3800, 0x3C00, 0xB800, 0x0000}; // 0.5, 1.0, -0.5, 0.0
        uint16_t v = vals[i % 4];
        memcpy(&xhost[i*2], &v, 2);
    }
    aclrtMemcpy(devX, bytes, xhost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t strides[] = {2048LL*50272, 50272, 1};
    int64_t storageDims[] = {4, 2048, 50272};
    aclTensor* x = aclCreateTensor(shape, 3, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, storageDims, 3, devX);
    aclTensor* y = aclCreateTensor(shape, 3, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, storageDims, 3, devY);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus s1 = aclnnBenchHarnessSoftmaxGetWorkspaceSize(x, -1, y, &wsSize, &exec);
    printf("GetWorkspaceSize: %d, wsSize=%lu\n", s1, wsSize);

    if (s1 == 0) {
        void* ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtStream stream;
        aclrtCreateStream(&stream);
        aclnnStatus s2 = aclnnBenchHarnessSoftmax(ws, wsSize, exec, stream);
        printf("aclnn: %d\n", s2);
        aclError se = aclrtSynchronizeStream(stream);
        printf("SynchronizeStream: %d\n", se);
        if (ws) aclrtFree(ws);
        aclrtDestroyStream(stream);
    }

    // Read back a few values
    std::vector<uint8_t> yhost(bytes, 0);
    aclrtMemcpy(yhost.data(), bytes, devY, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    int nonzero = 0;
    for (int64_t i = 0; i < N; i++) {
        uint16_t v;
        memcpy(&v, &yhost[i*2], 2);
        if (v != 0) nonzero++;
    }
    printf("nonzero output elements: %d / %ld\n", nonzero, N);

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(devX);
    aclrtFree(devY);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

编译执行（头文件/库路径按本机 CANN 安装与算子 vendor 目录代入）：

```bash
g++ -std=c++17 \
  -I$ASCEND_HOME/x86_64-linux/include -I$ASCEND_HOME/include \
  -I$ASCEND_HOME/opp/vendors/<OpName>/op_api/include \
  /tmp/test_chunked_debug.cpp -o /tmp/test_chunked_debug \
  -L$ASCEND_HOME/lib64 -L$ASCEND_HOME/opp/vendors/<OpName>/op_api/lib \
  -lascendcl -lnnopbase -lcust_opapi
LD_LIBRARY_PATH=$ASCEND_HOME/opp/vendors/<OpName>/op_api/lib:$ASCEND_HOME/lib64:$LD_LIBRARY_PATH \
  /tmp/test_chunked_debug
```

（实测首次编译会因缺 `<cstring>` 报 `memcpy was not declared`，探针已补 include。）

**原始观测**（本例实测输出）：

```
GetWorkspaceSize: 0, wsSize=0
aclnn: 361001
SynchronizeStream: 0
nonzero output elements: 0 / 411828224
```

**解读规则**：

- `GetWorkspaceSize=0` + `aclnn!=0` → tiling/host 侧通过、execute 阶段失败：查 kernel binary 选择（tilingKey 匹配、安装产物 `.o` 是否存在/被覆盖）、kernel 执行异常。本例 361001 = `ACLNN_ERR_RUNTIME_ERROR`（`include/aclnn/opdev/op_errno.h` 定义），配合安装目录每 dtype 只有一个 `.o` 且 host 设了 `SetTilingKey(1)`，定位为 tilingKey 选不到单一 binary。
- `aclnn=0` 但 `nonzero==0` → kernel 执行返回成功但没写 y：查 kernel 内早退分支/条件分发。
- `SynchronizeStream!=0` → 异步执行错误在同步点暴露（本例为 0，错误在 enqueue 阶段即返回）。
- 错误码语义可再回 CANN 头文件 grep：`grep -rn "361001" $ASCEND_HOME/include`。

## 设计决策

- 为何独立探针而不是修 ST 框架：探针 5 分钟可得真实返回码，先定方向；ST 框架的 `ret.detail` 覆盖逻辑属测试脚手架缺陷，修复它要动共享测试代码（超出当时任务边界），且修好后也不如探针直达（探针还能打印 sync 返回码与非零计数这两个 ST 不暴露的信号）。
- 放弃的候选：改 ST 框架把 `s2!=0` 短路（`if (s2 == 0)` 包住 golden 对比）——正确但属另一处变更；本卡只记录"如何看见错误码"，框架修复列为预防项。
- 固定输入模式（0.5/1.0/-0.5/0.0 循环）而非随机：非零输出计数才可比（随机输入下"全 0"判定同样有效，但固定模式顺带排除数据依赖）。
- 当前方案属 `recommended_fastpath`：出现"全 0 + 无错误码"时的第一步；数值类失败（部分元素错）不需要它。

## 预防

- 自建 aclnn ST 框架时，`s1==0` 的 golden 对比分支必须再判 `s2==0`（`if (s1 == 0 && s2 == 0)`），不要让对比结果覆盖 execute 错误码。
- 把本探针保存为算子工程的可复用调试脚手架（`tests/aclnn/debug_repro.cpp` 之类），新分支联调"全 0"时直接换 shape 跑。
- 两段式 aclnn 的所有返回码（s1/s2/sync）在任何测试输出里都应显式打印，不依赖"失败详情"隐式携带。

## 验证信号

问题与验证来源：BenchHarnessSoftmax Branch-1 首次联调"全 0 无错误码"故障的直连诊断。逐条实测信号：

- ST 侧：Branch-1 0/9 全失败、详情均为 `sum_to_one: sum=0.000000` / `isnan mismatch at 0`，无任何错误码（来源平台：950；证据等级：tool_observed）。
- 探针侧：同一 S3 fp16 用例 `GetWorkspaceSize: 0, wsSize=0`、`aclnn: 361001`、`SynchronizeStream: 0`、`nonzero output elements: 0 / 411828224`（来源平台：950；证据等级：tool_observed）。
- 错误码定义：`include/aclnn/opdev/op_errno.h:17 #define ACLNN_ERR_RUNTIME_ERROR 361001`（来源平台：950；证据等级：tool_observed）。
- 探针结论被后续修复闭环证实：`SetTilingKey` 改为恒 0 后同 binary 立即可执行，Branch-1 由 0/9 转为可执行（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 探针只能还原返回码与输出形态，不能定位 kernel 内部逻辑错误；数值类失败（部分 mismatch）请走精度对比路径。
- 大 shape（本例 fp16 824MB device 内存）探针同样要吃满内存，device 余量不足时 malloc 会失败——探针未检查 `aclrtMalloc` 返回值，复用时按需加判。
- `nonzero==0` 的判读依赖"测试/探针先 memset 清零输出"这一前提；未清零时全 0 判定不可靠（可能是残留旧值）。
- ST 框架的覆盖行为是自建 `test_aclnn.cpp` 的实现细节；官方 TTP/UTT 框架的错误透传行为未验证，不适用本卡结论。

**未决项**：

- ST 框架 `ret.detail` 覆盖缺陷在原工程未修复（当时按"最小变更"原则绕过），复用该 ST 模板的工程会继续踩；修 `if (s1 == 0 && s2 == 0)` 列为复用时的第一改动。
- `aclrtMalloc` 大块分配失败路径未在探针中覆盖（未尝试，无证据）。
## 坏实践

- 见"全 0"就钻进 kernel 数值代码逐行重审（UB 预算、reduce 参数、对齐逐项排查），不先验证 kernel 是否执行——本例走查消耗数轮，直连探针 5 分钟出方向。
- 信任"失败详情"是完整错误信息——自建框架的详情拼接逻辑可能吞码，任何关键返回码都应显式打印。

## 误诊

- 曾据 ST 输出反推"s1==0 且 s2==0（否则详情会有错误码），所以 kernel 跑了但算错"——前提（ST 会透传 s2 错误码）不成立，详情被 golden 对比覆盖；这正是本卡要消除的误判路径。
