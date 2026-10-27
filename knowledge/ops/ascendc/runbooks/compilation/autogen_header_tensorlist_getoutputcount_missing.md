---
type: "Runbook"
title: autogen 头缺 TensorList/GetOutputCount 声明
description: custom_opp 安装包自动生成的 aclnn 头基于 REG_OP 单 OUTPUT 声明，缺手写实现的 GetOutputCount/TensorList 接口；libcust_opapi.so 实际导出全量符号。ST/PyTorch 适配层应优先 include 源码 op_api 头（SOURCE_OP_API_DIR 前置 + 回退安装头）。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# autogen 头缺 TensorList/GetOutputCount 声明

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义 MoE 分组矩阵乘算子（变长输出：split_item=0/1 时输出 E 个 tensor，2/3 时输出 1 个）开发 C++ ST 与 PyTorch 适配层，调用方按算子源码 `op_api/aclnn_*.h` 的人工声明调用三个接口（`GetOutputCount` → 预分配 → `GetWorkspaceSize(y_list)` → `Execute`），编译统一失败。C++ ST 报"函数未声明/参数类型不匹配"；PyTorch 适配层 `libtorch_adapter.so` 首次编译报两条错误：

```text
torch_adapter.cpp:234:26: error: 'aclnnBenchNoHarnessGroupedMatmulGetOutputCount' was not declared in this scope; did you mean 'aclnnBenchNoHarnessGroupedMatmulGetWorkspaceSize'?
torch_adapter.cpp:257:9: error: cannot convert 'aclTensorList*' to 'const aclTensor*'
In file included from torch_adapter.cpp:45:
/.../cann-9.1.0/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include/aclnn_bench_no_harness_grouped_matmul.h:35:22: note: initializing argument 7 of 'aclnnStatus aclnnBenchNoHarnessGroupedMatmulGetWorkspaceSize(..., const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)'
```

即编译器拿到的是**安装包里的头文件**（`opp/vendors/<vendor>/op_api/include/`，文件头注释 `this file was generated automaticlly donot change it`），该头：

1. 完全没有声明 `aclnnBenchNoHarnessGroupedMatmulGetOutputCount`；
2. `aclnnBenchNoHarnessGroupedMatmulGetWorkspaceSize` 的输出参数是 `const aclTensor *out`（单 tensor），不是 `aclTensorList* y`。

而算子源码目录 `op_api/aclnn_*.h` 里人工声明的完整 API 是三个接口、输出为 `aclTensorList *y`——即"装出来的头"与"写出来的头"不一致。

## 适用场景

- 工程形态：自定义 aclnn 算子（op_graph 定义 REG_OP + op_api 手写 L2/L0 实现 + 构建系统自动生成安装头），ST/上层调用方/PyTorch 适配层直连 aclnn 接口。
- API 路径：`op_graph/<op>_proto.h` 的 `REG_OP(...).OUTPUT(y, ...)`；构建系统生成的 `vendors/<op>_custom/op_api/include/aclnn_<op>.h`；`nm -D` 查 `libcust_opapi.so` 导出符号。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有用 REG_OP 单 OUTPUT 声明、却在 op_api 手写 TensorList/多输出接口的 registry-invoke 类自定义算子。
- 路线优先级：recommended_fastpath（调用侧优先用源码头；若要安装头正确，需改 REG_OP 为动态输出声明，本轨迹未验证）。
- 明确不适用 / 未验证场景：算子输出本来就是单 tensor（REG_OP 与手写接口一致）时无此问题；官方内置算子（无源码 op_api 可指向）；调用方机器上没有算子源码 checkout 时只能用安装头，此时无法调用 autogen 头未声明的接口；`REG_OP` 改用 `DYNAMIC_OUTPUT` 后生成头是否变为 TensorList 形态未在本轨迹验证（仅推断）。

## 判别方法

1. 编译报"函数未声明/参数类型不匹配"时，先分清测试 include 的是哪个头：`find`/`#include` 路径里 vendors 安装头优先就会拿到生成版。
2. 用 `nm -D` 核对安装库里实际导出的接口——若 `.so` 导出了生成头里没有的函数（本例 `aclnnBenchNoHarnessGroupedMatmulGetOutputCount` 为 T 导出符号），说明库是按 op_api 源码编译的、只有头是生成的：

```bash
nm -D .../vendors/<op>_custom/op_api/lib/libcust_opapi.so | grep aclnnBench
# 000000000000845c T aclnnBenchNoHarnessGroupedMatmulGetOutputCount
# 00000000000087e2 T aclnnBenchNoHarnessGroupedMatmulGetWorkspaceSize
# 000000000000914b T aclnnBenchNoHarnessGroupedMatmul
```

3. 对照 `op_graph/<op>_proto.h`：`REG_OP(...).OUTPUT(y, TensorType{...})` 是单 tensor 输出声明，生成器据此产出 `const aclTensor* out` 形态的头；而 `op_api/aclnn_<op>.h`（手写）声明 `aclTensorList* y`。

## 根因

构建系统的 aclnn 头生成器以 `op_graph` 中 `REG_OP` 的 `OUTPUT(y, ...)`（单 tensor）为事实源生成安装头；op_api 源码里手写的 L2 接口（3 函数 + `aclTensorList*` 多输出）仍被正常编译进 `libcust_opapi.so`。两者事实源不同导致"库与头不匹配"：库是手写接口，头是 REG_OP 推导接口——缺的只是声明。

## 解决配方

修复（测试/适配工程 CMake：优先使用算子源码 `op_api/` 目录的手写头，回退到安装头）：

```cmake
    # 优先使用算子源码 op_api 目录中的的人工声明头文件（含 GetOutputCount + TensorList 接口），
    # 而非安装包中根据 REG_OP 自动生成的头文件（仅单 Tensor 输出）。
    set(OP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)
    if(EXISTS ${OP_SOURCE_DIR}/aclnn_bench_no_harness_grouped_matmul.h)
        set(CUSTOM_OP_INCLUDE_DIR ${OP_SOURCE_DIR})
        message(STATUS "Using source op_api include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        find_path(CUSTOM_OP_INCLUDE_DIR
            NAMES aclnn_bench_no_harness_grouped_matmul.h
            PATHS
                ${CANN_ROOT}/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
                /usr/local/Ascend/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
                $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
        )
        if(CUSTOM_OP_INCLUDE_DIR)
            message(STATUS "Using installed custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "Custom operator include directory not found."
                  " Please install the operator first, or use Mock mode: cmake -DUSE_MOCK=ON ..")
        endif()
    endif()
```

修复前（只找安装头）：

```cmake
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_bench_no_harness_grouped_matmul.h
        PATHS
            ${CANN_ROOT}/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
            /usr/local/Ascend/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_grouped_matmul_custom/op_api/include
    )

    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found."
              " Please install the operator first, or use Mock mode: cmake -DUSE_MOCK=ON ..")
    endif()
```

链接库仍用安装包的 `libcust_opapi.so`（其导出的就是手写接口），只有头文件换源码版。

适配此配方时的两个已踩坑点：

1. `OP_SOURCE_DIR` 相对路径要数清 CMakeLists 所在目录到 `op_api/` 的层级——测试工程嵌套在 `tests/st/torch/` 下时是 `../../../op_api`（写 `../../op_api` 会因路径不存在而静默回退到安装头，报错重现）。
2. `CUSTOM_OP_VENDOR_NAME`（或等价 vendor 名变量）不能只在 `else()` 分支里设置——后面 `find_library(... cust_opapi PATHS .../vendors/${CUSTOM_OP_VENDOR_NAME}/...)` 在走源码头分支时也要用变量展开查找动态库，变量为空会报 `cust_opapi library not found.` 且 `Configuring incomplete`。

## 设计决策

- 为什么改调用侧 include 路径而不是改 REG_OP 为 `DYNAMIC_OUTPUT` 重新生成：改 REG_OP 牵动算子注册/图模式行为，影响面大且当轮只验证 ST 直连路径；include 源码头是最小改动，且 `.so` 本来就是按手写接口编译的（nm 已证）。
- 为什么回退保留 find_path：保证仓库外（只装了包、没源码）的使用者仍可编译。
- 放弃的候选：手抄一份"修正头"进 ST 目录——会与 op_api 源头漂移，未采用。
- 当前方案路线档位：recommended_fastpath（调用侧适配；根治须 REG_OP 与 op_api 接口对齐，超出本卡证据）。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；头文件对比对象：`$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include/aclnn_<op>.h`（生成）vs `operators/<op>/op_api/aclnn_<op>.h`（手写）。
- 相关文件：`op_graph/bench_no_harness_grouped_matmul_proto.h`（REG_OP）、`op_api/aclnn_bench_no_harness_grouped_matmul.h/.cpp`（手写 3 函数接口）、`tests/st/CMakeLists.txt` 与 `tests/st/torch/CMakeLists.txt`（本次修复）、`torch_adapter.cpp`（PyTorch 适配层）。

## 预防

- 手写 op_api L2 接口与 REG_OP 输出声明不一致（如 TensorList 多输出）时，所有直连 aclnn 的测试/调用工程统一优先 include 源码 `op_api/` 头，并在 CMake 里打印实际使用的 include 目录。
- 编译报"接口不存在"先 `nm -D` 看库导出，再决定改哪一侧；不要按生成头改测试逻辑（会丢多输出能力）。

## 验证信号

以下信号来自该算子首轮 ST/PyTorch 适配层联调会话（Ascend950PR，CANN 9.1.0）：

- 修复前：ST 编译失败——测试调用的 `aclnnBenchNoHarnessGroupedMatmulGetOutputCount` 在安装头中不存在，`GetWorkspaceSize` 第 7 参类型 `const aclTensor*` 与测试传入的 `aclTensorList*` 不匹配（来源平台：950；证据等级：tool_observed）。
- 事实核对：安装头仅 2 个函数、单 tensor 输出（`/* calution: this file was generated automaticlly donot change it. */`）；源码头声明 3 函数 + `aclTensorList* y`；REG_OP 为 `.OUTPUT(y, ...)` 单输出；`libcust_opapi.so` 导出 3 个 aclnn 函数（T 符号）（来源平台：950；证据等级：tool_observed）。
- 中间失败：`OP_SOURCE_DIR` 路径过浅时 CMake 打出 `Using installed custom operator include dir`（静默回退安装头）；路径修对但 vendor 名只在 else 分支设置时报 `cust_opapi library not found.`、`Configuring incomplete`（来源平台：950；证据等级：tool_observed）。
- 修复后：ST 重新编译通过，测试推进到 NPU 运行期（`[FAIL] GetOutputCount 失败: 161002`，即接口已正确链接、进入参数校验逻辑，属另一问题）；PyTorch 适配层 `cmake + make` 成功产出 `libtorch_adapter.so`（`[100%] Built target torch_adapter`）（来源平台：950；证据等级：tool_observed）。
- 同仓 C++ ST 工程（Real NPU 模式，同款源码头策略）编译并全量运行通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡解决"头与库不一致"的编译期问题；生成头在图模式（REG_OP 走 GE）下是否可用未验证。
- 调用方没有算子源码 checkout 时无法套用本配方，只能受限于 autogen 头声明的接口子集。
- 若 `nm -D` 显示符号确实缺失，则是库打包问题，本配方无效。
- 若后续把 REG_OP 改为 `DYNAMIC_OUTPUT` 使生成头与手写接口对齐，本卡的 include 回退逻辑可保留但不再是必需（未验证）。
- 多输出 TensorList 接口在运行期的正确性（split_item=0/1 多输出精度）是另一个独立问题，本卡不覆盖。
- autogen 头的生成规则（能否配置为生成辅助接口声明）未验证；跨 CANN 版本行为未验证。

**未决项**：

- autogen 安装头为何不生成 `GetOutputCount`/`aclTensorList` 声明（打包器对辅助接口的生成规则）未追到机制层——复现并追根因是优先接续项。
- 生成器对 `DYNAMIC_OUTPUT` 的支持与生成头形态未验证；若要根治"安装头即正确接口"，需在该算子上做 REG_OP 动态输出改造并回归。
## 坏实践

- 拿安装包自动生成的头当唯一接口事实源去改测试代码（删 GetOutputCount、把 TensorList 改单 tensor）——会静默丢掉多输出能力，且与 `.so` 实际导出不符。
- 在 vendors 安装目录里手改生成头补声明——下次安装即被覆盖，且污染 CANN 安装目录。
- 把 `OP_SOURCE_DIR` 写错层级后不看 CMake 的 `Using ... include dir` 状态行，误以为已切换源码头。

## 误诊

- 初看两条编译错误容易误判为"torch_adapter 调用姿势写错"或"算子根本没实现 GetOutputCount"；`nm -D` 符号表一步即可区分"声明缺失"与"实现缺失"。
- 无（本问题一次定位到头文件来源差异）。
