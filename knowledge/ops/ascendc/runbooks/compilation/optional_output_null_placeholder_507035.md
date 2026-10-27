---
type: "Runbook"
title: "可选输出null时占位tensor致507035"
description: OPTIONAL 输出运行时传 null，框架 gen_placeholder 生成地址无效的占位 tensor，辅助 op 访问即 MTE 越界崩 507035；L2 层自建 dummy tensor 修复。
tags: [task.debugging, topic.implementation.api_layer, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 可选输出null时占位tensor致507035

## 现象

BenchNoHarnessUnique 自定义算子（Ascend950PR，CANN 9.1.0，`bash build.sh --soc=ascend950 -j8` 构建、`bash build/custom_opp_ubuntu_x86_64.run --quiet --force` 安装）的 ST 测试中，op 定义把第二个输出 `inverse` 声明为 `ParamType(OPTIONAL)`；当用例 `return_inverse=false` 时，C++ 测试按语义把 `inverse` 传 nullptr。结果这些用例**全部**在流同步处崩溃：

```
[numel=1 单元素 return_inverse=false 用例] dtype=int32, return_inverse=0, shape=[1], dedup=all_same, sort=random
  [FAIL] sync stream: 507035
...（所有 return_inverse=false 用例同样失败）
```

507035 = `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（`include/acl/error_codes/rt_error_codes.h`）。kernel 已用 TQue 模式重写、`return_inverse=true`（inverse 非 null）的用例全部 PASS——唯独可选输出为 null 的用例崩。plog 里错误细节为：

```
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4.
[ERROR] ... PrintErrorInfoForDavinciTask:[DFX_INFO]AI Core kernel execution failed ...
fault kernel_name=BenchNoHarnessUnique_5b5227062859d6097351dcb83f899f41_3
GetArgsInfo:tilingKey = 0, print 1 Times totalLen=(13*8), argsSize=104, schemMode=0, blockDim=1
```

且 kernel 入参里 `inverse` 并不是 null，而是一个框架给的占位地址（如 `0x120000019000`）——但该地址背后的设备内存并未真正分配，任何 MTE 访问都会越界。

## 适用场景

- 工程形态：自定义算子 aclnn 开发（registry-invoke 流程），算子带 OPTIONAL 输出且运行时可能传 null。
- API 路径：op 定义 `ParamType(OPTIONAL)` 输出 + 框架 `optionalOutputMode: "gen_placeholder"`（kernel config JSON）+ L2 `aclnnXxxGetWorkspaceSize` / L0 `ADD_TO_LAUNCHER_LIST_AICORE`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：platform_wide——所有在 CANN 9.x aclnn 框架下声明可选输出的自定义算子。
- 路线优先级：recommended_fastpath（L2 层兜底，对上层调用方透明）。
- 明确不适用 / 未验证场景：
  - 可选**输入**为 null 的占位行为未验证（本卡只验证了可选输出）；
  - 框架若提供官方开关（如关闭 gen_placeholder / 用 executor 管理 dummy 生命周期）属更优解，本轨迹未找到（见遗留问题）；
  - 其他 CANN 版本的 optionalOutputMode 行为未验证。

## 根因

可选输出运行时为 null 时，aclnn 框架按 `optionalOutputMode: "gen_placeholder"`（可在安装后的 kernel config JSON 里看到）为该输出生成**占位 tensor**：它有一个形似合法的设备地址，但内存并未实际分配。框架在算子 kernel 前后会插入 MemSet / MemSetV2 / NonFiniteCheck 等辅助 op，这些辅助 op 会访问该占位地址，触发 MTE 指令 DDR 越界（plog errcode 95），表现为 `aclrtSynchronizeStream` 返回 507035。

崩溃发生在框架侧，与 kernel 代码无关——本轨迹用排除法证明：kernel 里加 `if (hasInverse_ == 0) return;` 直接返回（什么都不做）仍然崩；L0 层做条件 `OP_OUTPUT`（null 时不把 inverse 传给 launcher）也仍然崩；只有让 L2 层在 inverse 为 null 时**传入一个内存真实有效的 tensor**，崩溃才消失。

## 解决配方

在 L2 API（`aclnnBenchNoHarnessUniqueGetWorkspaceSize`）里，调用 L0 之前，若 `inverse == nullptr` 则用 `aclrtMalloc` + `aclCreateTensor` 构造一个真实的 dummy tensor 并传下去，使框架不再生成占位 tensor、MemSet 有合法内存可写。

修复前（L2 直接把调用方传入的 null 透传给 L0）：

```cpp
    // 调用 L0 API
    const aclTensor* opResult = l0op::BenchNoHarnessUnique(
        xContiguous, returnInverse, y, inverse, uniqueExecutor.get());
```

修复后（L2 为 null 的可选输出兜底分配有效内存；新增 `#include "acl/acl_rt.h"`）：

```cpp
    // 空 Tensor 检查
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 当 inverse 为 null 时，创建一个有效的 dummy tensor
    // 框架的 optionalOutputMode=gen_placeholder 会为 null 的可选输出生成占位 tensor，
    // 但占位 tensor 的设备内存无效，导致框架的 MemSet/NonFiniteCheck 访问越界。
    // 因此在 L2 层创建一个有效内存的 dummy tensor。
    aclTensor* inverseTensor = inverse;
    if (inverseTensor == nullptr) {
        int64_t numel = x->GetViewShape().GetShapeSize();
        void* invDev = nullptr;
        auto mallocRet = aclrtMalloc(&invDev, static_cast<size_t>(numel * sizeof(int64_t)),
                                      ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(mallocRet == ACL_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        std::vector<int64_t> invShape = {numel};
        std::vector<int64_t> invStrides = {1};
        inverseTensor = aclCreateTensor(invShape.data(), invShape.size(), ACL_INT64,
                                        invStrides.data(), 0, ACL_FORMAT_ND,
                                        invShape.data(), invShape.size(), invDev);
        CHECK_RET(inverseTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
```

并把 L0 调用改传 `inverseTensor`：

```cpp
    // 调用 L0 API
    const aclTensor* opResult = l0op::BenchNoHarnessUnique(
        xContiguous, returnInverse, y, inverseTensor, uniqueExecutor.get());
```

重新构建、`--force` 安装、重跑 ST。

## 判别方法

1. 症状特征：带 OPTIONAL 输出的算子，可选输出为 null 的用例统一 `sync stream: 507035`，非 null 用例正常。
2. 开 plog 看错误细节（定位流程见同批 `plog_aivec_exception_diagnosis_507035`）：`grep -iE "(errcode|errorStr|out of bounds|tilingKey|fault kernel_name)" ~/ascend/log/debug/plog/plog-<pid>_*.log` → `errcode:(95) ... DDR address of the MTE instruction is out of range`，fault kernel 名为自家 kernel 的变体后缀（如 `_3`）。
3. 看框架给 kernel 的 args：plog `GetArgsInfo ... args(0 to 12)` 中可选输出参数位不是 0 而是一个占位地址（本例 `0x120000019000`）。
4. 确认配置：`cat $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/config/<soc>/<op>.json` → `optionalOutputMode: "gen_placeholder"`。
5. 终局排除法：kernel 里对"可选输出未启用"分支直接 return（空 kernel）——若仍崩，则崩溃在框架辅助 op，kernel 侧任何防护都无效，必须从 L2 层给有效 tensor。

## 设计决策

- 为什么修在 L2 而不是 kernel/L0：排除法证明框架辅助 op（MemSet/NonFiniteCheck）在 kernel 之外访问占位地址，kernel 内不写该输出也无法避免；L0 层条件 `OP_OUTPUT`（null 时不传该输出给 launcher）同样无效——只要 L0 收到的还是 null，框架照样生成占位 tensor。只有 L2 在进 L0 前把 tensor 变成"非 null 且内存有效"，占位机制才不触发。
- 为什么 dummy 尺寸取 `numel * sizeof(int64_t)`：inverse 语义是每个输入元素一个 int64 索引，按上界 numel 分配即可保证任何写入都落在有效内存内。
- 已知代价：两段式 L2（GetWorkspaceSize + 执行段）里分配的 dummy 设备内存在本实现中未释放（生命周期无法挂到第二段），存在泄漏；对测试/小规模场景可接受，量产路径需改进（见遗留问题）。
- 当前方案路线档位：recommended_fastpath（correctness 优先；内存泄漏代价见上）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0，自定义算子包安装于 `opp/vendors/bench_no_harness_unique_custom`。
- 相关文件：
  - `op_api/aclnn_bench_no_harness_unique.cpp`——L2 API（本卡修复点：dummy inverse tensor）。
  - `op_api/bench_no_harness_unique.cpp`——L0 API（曾试条件 OP_OUTPUT，最终回退）。
  - `op_host/bench_no_harness_unique_def.cpp`——op 定义（inverse 声明 `ParamType(OPTIONAL)`）。
  - kernel config：`opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/bench_no_harness_unique.json`（optionalOutputMode 证据）。
  - 算子仓 issue 记录（占位内存问题归档）。

## 预防

- 设计时先想清楚 OPTIONAL 输出的运行时形态：只要存在"调用方传 null"的路径，L2 层就要为它准备有效 tensor（或确认框架版本的占位行为安全后再依赖）。
- 联调带可选输出的算子时，把"null / 非 null"两种用例都放进冒烟集，不要只测非 null 路径。

## 验证信号

来源说明：以下信号来自该算子迭代一的联调修复：kernel 以 TQue 模式重写后先观察到 HAS_INVERSE=0 与 =1 的分野，经 plog/配置/args 三路证据定位到占位输出，最后在 L2 层加 dummy tensor 并全量回归。

- 修复前分界：ST 中 `return_inverse=true`（inverse 非 null）用例全部 PASS，`return_inverse=false`（inverse 为 null）用例全部 `[FAIL] sync stream: 507035`（来源平台：950；证据等级：tool_observed）。
- plog 证据：`inverse: nullptr` 传入 L2；kernel args 中该参数为占位地址 `0x120000019000`；`errcode:(95) The DDR address of the MTE instruction is out of range`、fault kernel `BenchNoHarnessUnique_..._3`、tilingKey=0；（手动读 tiling 的版本曾报 `errcode:(271) The address for scalar to access the internal buffer is out of bounds`）（来源平台：950；证据等级：tool_observed）。
- 配置证据：kernel config JSON 含 `"optionalOutputMode": "gen_placeholder"`（来源平台：950；证据等级：tool_observed）。
- 排除法证据：kernel 内 `if (hasInverse_ == 0) return;`（空 kernel）仍 507035；L0 条件 `OP_OUTPUT` 仍 507035——崩溃在框架侧（来源平台：950；证据等级：tool_observed）。
- 修复后：全部 int32 用例 PASS（`return_inverse=true/false` 双形态），最终 ST 32/34（int32 32/32=100%；float16 0/2 为迭代一不支持、预期失败），UT 13/13（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.1.0 + Ascend950PR 验证；其他版本 optionalOutputMode 的占位行为（是否同样无效地址）未验证。
- dummy tensor 走 `ACL_MEM_MALLOC_HUGE_FIRST`，超大 numel 时按 `numel*8` 上界分配可能吃紧设备内存（本轨迹用例 ≤ 数千元素，未验证大 shape）。
- 本卡结论是"占位地址无效"这一行为层面的根因；框架为何生成无效地址（保留区映射细节）未追到机制层。

**未决项**：

- 优先项：追根因到机制层——复现并定位 gen_placeholder 占位 tensor 的地址来源与 MemSet 访问路径（本卡止步于"行为层面：占位地址无效 + 框架辅助 op 访问"的排除法结论）。
- 优先项：解释旧 kernel 中 `GET_TILING_DATA_WITH_STRUCT` 与崩溃的二分相关性为何成立（tiling 结构 host/kernel 两侧定义是否曾不一致未查证）。
- dummy tensor 设备内存泄漏：两段式 L2 无法在执行段释放；候选方案（executor 生命周期管理、框架侧开关）未尝试。
- 大 numel 下 `numel*8` 上界分配的内存压力未验证。
## 坏实践

- 在 kernel 里对"可选输出未启用"做分支就以为安全——框架辅助 op 不经过你的分支。
- L0 层做条件 `OP_OUTPUT` 试图绕过 null 可选输出——框架仍会为 null 输出生成占位 tensor，无效。

## 误诊

排查过程中先后被证伪的假设（全部有重建-复测证据）：
- "`GET_TILING_DATA_WITH_STRUCT` 宏在 arch35 上导致 MTE 异常"：更早的调试轮里用空 kernel 二分得到"有该宏即崩、去掉即不崩"的相关性，并据此改写成逐字段手动读 tiling；但最终 L2 dummy-tensor 修复后，同一 `GET_TILING_DATA_WITH_STRUCT` 宏保留在 kernel 里全部用例通过——二分结论被框架侧崩溃混淆，宏本身不是根因（宏与崩溃的相关机制未追到底，见遗留问题）。
- "`DataCopyPad` 参数结构用错（`DataCopyParams` 应改 `DataCopyExtParams`）"：替换后 507035 原样复现；最终 TQue 版 kernel 用 `DataCopyParams` 也正常。
- "UB 超额（UB_COEFF/预留不足）"：调大 UB_RESERVED、动态 UB_COEFF 后仍崩。
- "kernel 未初始化 invQueue_ 等缓冲导致"：改为无条件初始化全部缓冲后仍崩。
