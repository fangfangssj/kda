---
type: "Runbook"
title: 安装头 scalar 类型与源码不符致堆破坏 161002
description: 安装的 autogen 头把 scalar 声明为 const aclTensor*，测试按源码头传 aclScalar*，类型混淆堆污染致后续用例 tensor_ 为 nil 报 dtype 43；统一为 aclTensor* 1 元素张量。
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.implementation.operator_registration, topic.memory]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# 安装头 scalar 类型与源码不符致堆破坏 161002

## 现象

在 Ascend950PR（CANN 9.1.0）上联调 BenchNoHarnessForeachNorm 的 ST（10 个用例顺序执行于同一进程）。修复 workspace 与收尾链问题后，第一条用例（[1024] fp32）通过，但第二条用例（[32,32] fp32）在 `aclnnBenchNoHarnessForeachNormGetWorkspaceSize` 即失败（原文）：

```text
[ERROR] OP(...): ... [CheckTensorList] ... errno[161002] OpName:[aclnnBenchNoHarnessForeachNorm_1] dtype 43 not supported
[ERROR] OP(...): ... [CheckParams] ... errno[161002] ... CheckTensorList failed
[ERROR] OP(...): ... errno[561000] ... check ret == ACLNN_SUCCESS failed.
  [FAIL] GetWorkspaceSize 失败: 161002
```

dtype 43 不是任何合法 dtype（正常应为 0=FP32）；plog 同期打印 `the aclTensor(0x…)'s tensor_ address is (nil). please check the memory usage in your code.`——新创建的输入张量元数据已损坏。第三条用例起进程 `std::bad_alloc` / 段错误级联，ST 通过率 1/10。把第二条用例换到第一个执行则通过、原第一条变第二个则失败——**首个 ACLNN 调用污染进程堆，殃及后续所有用例**。

## 适用场景

- 工程形态：自定义算子 aclnn 二段式 API（GetWorkspaceSize + 执行）开发与 ST 直连测试，同一进程顺序跑多用例。
- API 路径：`aclnnBenchNoHarnessForeachNormGetWorkspaceSize`、`aclCreateScalar`/`aclCreateTensor`/`aclCreateTensorList`、算子包安装头（`opp/vendors/<vendor>/op_api/include/`）与源码 `op_api/*.h` 的关系、`l0op::Contiguous`/`ConvertToTensor`/`AllocTensorList`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有"标量参数 + autogen 安装头"模式的自定义 aclnn 算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：官方内置算子（头文件由 CANN 统一发布，无 autogen/源码双头问题）；单用例进程（无跨用例污染窗口，但类型混淆本身仍属未定义行为）。

## 根因

算子包构建时自动生成的安装头（`build/autogen/aclnn_bench_no_harness_foreach_norm.h`，随包装进 `opp/vendors/.../op_api/include/`）把 scalar 参数生成为 **`const aclTensor *scalar`**，而手写源码头 `op_api/aclnn_bench_no_harness_foreach_norm.h` 声明为 **`const aclScalar *scalar`**。ST 按源码头编译，用 `aclCreateScalar` 造 `aclScalar*` 传入；实际链接到的库按 autogen 路径把该指针当 `aclTensor*` 使用（`NnopbaseAddInput` 等），两个不透明类型的内部布局完全不同——**按 aclTensor 布局解引用 aclScalar 对象，读写越界破坏进程堆**，首个用例执行后框架内部状态（后续用例 aclTensor 的 `tensor_` 指针等）被污染，级联出 dtype 43 / `tensor_ is nil` / double free / std::bad_alloc。

两个头文件的关键差异（verbatim）：

安装的 autogen 头（`this file was generated automaticlly donot change it.`）：

```c
__attribute__((visibility("default")))
aclnnStatus aclnnBenchNoHarnessForeachNormGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *scalar,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

源码头（手写）：

```cpp
ACLNN_API aclnnStatus aclnnBenchNoHarnessForeachNormGetWorkspaceSize(
    const aclTensorList *x,
    const aclScalar *scalar,
    aclTensorList *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

判别探针：写一个最小程序直接 include **安装头**并传 `aclCreateScalar` 产物——编译器立刻报出真实签名不匹配（而 ST 因 include 了源码头而"编译通过"，掩盖了冲突）：

```text
/tmp/quick_test.cpp:56:70: error: cannot convert 'aclScalar*' to 'const aclTensor*'
.../opp/vendors/.../op_api/include/aclnn_bench_no_harness_foreach_norm.h:26:22: note:
  initializing argument 2 of 'aclnnStatus aclnnBenchNoHarnessForeachNormGetWorkspaceSize(
      const aclTensorList*, const aclTensor*, const aclTensorList*, uint64_t*, aclOpExecutor**)'
```

`nm -D`/构建产物核对确认源码 cpp 同时编译进 `libcust_opapi.so`——以库的实际导出签名为准，而不是以任何一个头文件的声明为准。

## 解决配方

以安装的 autogen 头为准统一三处签名（源码头、源码实现、测试侧传参）。

1. 源码头改为 `const aclTensor *scalar`（verbatim）：

```cpp
ACLNN_API aclnnStatus aclnnBenchNoHarnessForeachNormGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *scalar,
    aclTensorList *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

2. 源码实现同步（`CheckNotNull`/`CheckParams`/`CheckScalar` 的 scalar 形参类型；`CheckScalar` 改查张量 dtype，verbatim 节选）：

```cpp
static bool CheckScalar(const aclTensor* scalar)
{
    OP_CHECK(scalar != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scalar tensor is null"),
             return false);
    OP_CHECK(scalar->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "scalar dtype %d != DT_FLOAT",
                     static_cast<int>(scalar->GetDataType())),
             return false);

    return true;
}
```

同时移除 L2 内的 `l0op::Contiguous` 逐张量连续化与 `uniqueExecutor->AllocTensorList` 包装，直接把入参 TensorList 传给 L0 op（校验阶段已确认输入连续），仅保留必要的 scalar 转换：

```cpp
    auto ret = CheckParams(x, scalar, y, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将 scalar 转换为 1元素张量（L0 API 需要 aclTensor* 类型）
    const aclTensor* scalarTensor = uniqueExecutor->ConvertToTensor(scalar, DataType::DT_FLOAT);
    CHECK_RET(scalarTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 直接使用原始 TensorList（输入已在 CheckParams 中确认连续）
    // 跳过 Contiguous + AllocTensorList，避免 executor 管理额外 TensorList 资源
    // 调用 L0 算子
    const aclTensorList* opResult = l0op::BenchNoHarnessForeachNorm(
        x, scalarTensor, y, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

3. ST 侧 scalar 改用设备端 1 元素张量（verbatim）：

```cpp
    // 2. 创建 scalar (1元素张量，autogen API 要求 aclTensor* 类型)
    float scalar_val = tc.scalar;
    void* scalar_dev = nullptr;
    aclrtMalloc(&scalar_dev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(scalar_dev, sizeof(float), &scalar_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    // 使用 1D shape={1} 而非 0D，避免框架对 0 维张量的潜在问题
    int64_t scalar_shape[] = {1};
    int64_t scalar_strides[] = {1};
    aclTensor* scalar_acl = aclCreateTensor(scalar_shape, 1, ACL_FLOAT, scalar_strides,
                                             0, ACL_FORMAT_ND, scalar_shape, 1, scalar_dev);
```

清理路径相应把 `aclDestroyScalar` 换成 `aclDestroyTensor(scalar_acl)` + `aclrtFree(scalar_dev)`。

## 设计决策

- 为什么向 autogen 头对齐而不是反过来让 autogen 生成 `aclScalar*`：autogen 头由构建系统从算子定义生成且随包安装，测试与外部调用方实际 include 的是它；源码侧改成一致成本最低。（若能在算子定义/spec 层控制生成签名则更佳，本会话未验证该路径。）
- scalar 用 shape={1} 的 1D 张量而非 0 维：0 维张量（`aclCreateTensor(nullptr, 0, …)`）在本会话曾配合类型修复后仍出现后续段错误，保守取 1D {1}。
- 移除 Contiguous/AllocTensorList：该路径曾被怀疑为泄漏/double free 来源（实验证明移除后 dtype 43 依旧，真因是类型混淆），保留移除是为减少 executor 托管资源面。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；算子包 `bench_no_harness_foreach_norm_custom` 安装于 `opp/vendors/`，`load_priority` 含该 vendor。
- 相关文件：`op_api/aclnn_bench_no_harness_foreach_norm.h/.cpp`（源码双层）、`build/autogen/aclnn_bench_no_harness_foreach_norm.h`→安装头、`tests/st/test_aclnn_bench_no_harness_foreach_norm.cpp`、`tests/st/CMakeLists.txt`（include 顺序）。
- 判别命令：

```bash
# 看安装头真实签名
sed -n '1,40p' ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_api/include/aclnn_<op>.h
# 用最小程序 include 安装头编译，暴露签名不匹配
g++ -std=c++17 -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/x86_64-linux/include" \
    -I"${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_api/include" /tmp/quick_test.cpp -o /tmp/quick_test
# 看库导出符号
nm -D ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_api/lib/libcust_opapi.so | grep <OpName>
```

## 预防

- 自定义 aclnn 算子联调前，先 `sed`/编译探针核对**安装头**与源码头的每个参数类型是否一致，尤其 scalar（aclScalar vs aclTensor）与 TensorList；include 顺序掩盖不了运行时布局差异。
- 测试工程不要靠"把源码 op_api 目录放 include 首位"来绕过安装头——那只是让编译过，运行时按库的真实签名走。
- 同进程顺序跑多用例时，首用例后出现"新创建对象元数据损坏"（tensor_ nil / 字段变垃圾值），优先怀疑前一次调用的类型混淆/越界写破坏了堆。

## 验证信号

以下信号来自该算子联调与内存污染排查会话：级联复现 → 探针定位 → 三处统一修复 → 顺序交换实验：

- 修复前：第一条用例通过后第二条报 `errno[161002] dtype 43 not supported`，plog 伴 `tensor_ address is (nil)`；第三条用例起 `std::bad_alloc`；ST 1/10（来源平台：950；证据等级：tool_observed）。
- 探针：include 安装头的最小程序编译报 `cannot convert 'aclScalar*' to 'const aclTensor*'`，确认安装头签名为 aclTensor*（来源平台：950；证据等级：tool_observed）。
- 顺序交换实验：第二条用例换首个执行 `[PASS] MERE=0.00e+00`，原第一条变第二个即失败——跨用例堆污染来自首次调用，与具体 shape 无关（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 类型修复后、进程隔离引入前，同进程第二条用例仍失败（executor 析构路径的堆残留，属另一独立根因，另行沉淀）——本卡不覆盖该层。
- 0 维张量作 scalar 曾配合本修复出现后续段错误，本卡最终采用的是 1D shape={1} 形态；0 维形态不可用/可用未单独定论。
- 仅 CANN 9.1.0 实测；其他版本 autogen 头生成规则未验证（未验证，推断可迁移——"以安装头为准核对签名"的方法与版本无关）。
- 三处签名统一修复后的全量 ST 通过明细在后续记录段（任务返回口径），未逐条复核日志，本卡证据止于探针定位与顺序交换实验。

**未决项**：

- 优先项：autogen 头为何把 scalar 生成为 `aclTensor*`（算子定义的哪一层决定）、能否在构建配置层控制——未验证。
- `ConvertToTensor` 托管资源在"不显式销毁 executor"路径下的最终归宿依赖进程隔离兜底，进程内干净释放方案未找到。
- 0 维 scalar 张量的可用性未单独定论。
- 三处签名统一后的全量 ST 通过明细待后续记录复核。
## 坏实践

- ST 编译期用源码头覆盖安装头（include 顺序），对签名分歧"编译过就是对的"。
- 用 `aclCreateScalar` 产物直接传给声明为 `aclTensor*` 的接口（不透明指针让编译器闭嘴，运行时按错误布局解引用）。
- 把首用例后的级联失败当成"各用例各自独立的 bug"逐个修——它是同一份堆损坏的不同投影。

## 误诊

- 曾假设 kernel 的 DataCopyPad 写 workspace 越界污染内存：逐层核对 `GetUserWorkspace`（16MB 已对齐）、`CopyUbufToGmAlignV2` 的 4 字节写语义，未发现越界——真因在宿主侧类型混淆，不在 kernel。
- 曾假设 executor 未释放导致泄漏，加 `aclDestroyAclOpExecutor` 修复：触发 double free（属另一独立根因）；移除 `Contiguous`/`AllocTensorList` 后 dtype 43 依旧，均排除为根因。
