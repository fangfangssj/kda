---
type: "Runbook"
title: aclnn 冒烟测试以 autogen 生成头为签名源
description: 多输出+属性算子的 aclnn 两段式签名以 build/autogen 生成头为事实源（输入→属性→输出 Out→ws→executor）；签名内容逐项由 OpDef 注册决定（attr 名 snake→camel、Input 单张量 vs TensorList），手写 aclnn 与 op_def 无联动会与安装头错配；附完整冒烟测试。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# aclnn 冒烟测试以 autogen 生成头为签名源

## 现象

自定义算子包（2 输入 + 1 个 bool 属性 + 5 输出）构建安装通过后，要写 `tests/aclnn` 冒烟测试验证最小调用通路。两段式 aclnn API 的参数顺序与命名**不能凭直觉拼**：属性放哪、输出参数叫什么、workspace 参数怎么排，猜错就是编译错或参数错位。实测做法是：**先构建，再读 `build/autogen/aclnn_<snake算子名>.h` 生成头**，按生成头逐参照抄。生成头给出的真实签名：

```c
__attribute__((visibility("default")))
aclnnStatus aclnnThnnFusedGruCellGradGetWorkspaceSize(
    const aclTensor *gradHy,
    const aclTensor *storage,
    bool hasBias,
    const aclTensor *gradInputGatesOut,
    const aclTensor *gradHiddenGatesOut,
    const aclTensor *gradHxOut,
    const aclTensor *gradInputBiasOut,
    const aclTensor *gradHiddenBiasOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnThnnFusedGruCellGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 适用场景

- 工程形态：算子仓（ops-registry-invoke 流程）`tests/aclnn/` 的**接口桩阶段验收**——最小调用通路冒烟（`aclnn<Op>GetWorkspaceSize` → `aclnn<Op>` → `aclrtSynchronizeStream` 全链返回 OK 即 PASS），任务验收口径明确该阶段**不做数值精度校验**（数值用例属 kernel 实现阶段，不是本卡用"放松测试"换取通过——桩阶段 kernel 本就无算法可校）。
- API 路径：`build/autogen/aclnn_<op>.{h,cpp}`（op_build 从 OpDef 生成的 API 桩）；`acl/acl.h`（aclInit/aclrtSetDevice/aclrtMalloc/aclrtMemset/aclCreateTensor/aclrtSynchronizeStream）；`tests/aclnn/CMakeLists.txt`（include/link 指向 vendors 安装树）。
- 验证过的平台与 CANN 版本：950（ascend950/arch35）+ CANN 9.0.0-beta.2；签名规则与 OpDef 注册写法另在 CANN 9.1.0、9.2.0 实测。
- 适用粒度：operator_family——所有走 aclnn 两段式调用的自定义算子接口验收。
- 路线优先级：recommended_fastpath（构建 → 读生成头 → 按生成头写冒烟；dtype 覆盖 = OpDef DataType 列表逐组合）。
- 明确不适用 / 未验证：数值精度验收（本卡不含数值比对）；zero-size 张量（如 has_bias=false 的空 bias 输出）未测；非 ND 格式未测；属性为 List/必选属性的签名位置未测。

## 判别方法

签名映射规律（本例实测归纳，codegen 按 OpDef 的 INPUT→ATTR→OUTPUT 声明序展开）：

```
OpDef Input("grad_hy")        → const aclTensor *gradHy          （snake_case → camelCase）
OpDef Attr("has_bias")        → bool hasBias                     （位于全部输入之后、全部输出之前）
OpDef Output("grad_input_gates") → const aclTensor *gradInputGatesOut（输出名追加 Out 后缀）
尾参固定                       → uint64_t *workspaceSize, aclOpExecutor **executor
第二函数 aclnn<Op>            → (void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

判别动作：构建后 `cat build/autogen/aclnn_<op>.h`，以生成头为唯一签名事实源；不确定时再看生成 `.cpp` 的属性透传（bool 属性经 `NnopbaseAddAttrWithDtype(&hasBias, sizeof(bool), 0, kNnopbaseBool)` 直传 executor，无额外校验）。

**OpDef（OP_ADD 体系）属性注册写法（CANN 9.1.0 实测）**：布尔属性用构建器方法链：

```cpp
this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
```

**两套注册体系辨析（防混）**：`operator_reg.h` 的 REG_OP 体系用 `.ATTR(name, Bool, default)` 链式宏（`ATTR(x, Type, ...)` 展开 `ATTR_Bool(x, __VA_ARGS__)`，`ATTR_Bool` 等在 132-145 行定义为空、由 `ATTR` 统一展开 `Op##Type get_attr_##x()`）；自定义算子包的 OpDef 体系用构建器方法 `.Attr(name).AttrType(OPTIONAL).Bool(default)`。二者不是同一套 API，别混写。

**dtype 组合矩阵（CANN 9.1.0 实测）**：多个输入/输出的 `.DataType({...})` 按**列**对齐成 N 组组合，每组生成一个 kernel `.o`（改写前 2 个 → 改写后 8 个，实测安装目录 `.o` 数量与组合数一致）：

```cpp
this->Input("x").ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
               ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32, ge::DT_INT32})
    ...
this->Input("axes").ParamType(REQUIRED)
    .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
               ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
    ...
```

**op_def 的 Input() 注册形态决定 autogen 签名（CANN 9.2.0 实测）**：autogen 头的签名完全由 op_def 的输入注册决定——`Input("weight_ih")` 生成 `const aclTensor *weightIh`；本版本 `register/op_def.h` 无 DynamicInput 一类方法，单张量注册即单张量签名。手写 aclnn 的头/实现与 op_def 无联动，写什么导什么（.so 导出对应手写版符号）。两条生成链各说各话时，安装到 vendors 的头（autogen 版）与库里符号（手写版）不匹配：按安装头编译的消费方要么编译失败（类型不兼容），要么链接到签名不符的符号埋下运行期崩溃。

三信号交叉核对（5 分钟定位签名漂移，CANN 9.2.0 实测）：

```bash
# 1) 看安装头（autogen 产物）的形参类型
grep -n "weightIh\|TensorList" $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include/aclnn_<op>.h

# 2) 看 .so 实际导出符号（手写实现的签名），必要时对比 demangle
nm -D $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib/libcust_opapi.so | grep GetWorkspaceSize

# 3) 回看 op_def 的输入注册形态
grep -n "Input(" <op_host>/<op>_def.cpp
```

安装头与 .so 符号、op_def 注册三者不一致即命中；提交前跑一次三信号核对，任何一处签名漂移直接挡下。

## 解决配方

测试骨架与流程：

1. `CHECK_ACL`/`CHECK_ACLNN` 宏包错检查（失败打印码值+位置并返回 -1）；
2. 每个 dtype 一例：按算子 Shape 契约构造 dims/strides（连续 ND 布局），`aclrtMalloc` + `aclrtMemset` 清零（桩阶段不构造数据，清零保运行确定性），`aclCreateTensor` 建描述符；
3. Phase 1：`aclnn<Op>GetWorkspaceSize(输入…, 属性, 输出…, &workspaceSize, &executor)`；`workspaceSize > 0` 才 malloc workspace；
4. Phase 2：`aclrtCreateStream` → `aclnn<Op>(workspace, workspaceSize, executor, stream)` → `aclrtSynchronizeStream`；
5. 全链 OK 打印 `[smoke] <case> PASSED`；释放全部 tensor/device 内存并销毁 stream；
6. `main`：`aclInit` → `aclrtSetDevice(0)` → 按 proto dtype 组合表逐行跑例 → `aclrtResetDevice` + `aclrtFinalize`。

dtype 覆盖规则：测试用例数 = OpDef `DataType({...})` 列表长度（本例 fp32/fp16/bf16 各一例，各 IO 同 dtype）。元素字节数：fp32=4，fp16/bf16=2（bf16 无 host 数据类型，按 `ACL_BF16` 枚举 + 按字节分配即可）。

配套链接顺序（`tests/aclnn/CMakeLists.txt`，vendor 安装目录置于 CANN lib64 **之前**，保证链接到本工程 run 包安装的 `libcust_opapi.so`）：

```cmake
# before（样例原序：CANN lib64 在前）
link_directories(
    ${CANN_HOME}/lib64
    ${CANN_HOME}/opp/vendors/<OpType>/op_api/lib  # libcust_opapi.so
)
## after（vendor 优先加载）
link_directories(
    ${CANN_HOME}/opp/vendors/<OpType>/op_api/lib  # libcust_opapi.so (优先加载)
    ${CANN_HOME}/lib64
)
```

`run.sh` 运行期前置：`export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/<OpType>/op_api/lib/:${LD_LIBRARY_PATH}`。

**手写 aclnn 与 op_def 签名不一致时的两条落地路径（CANN 9.2.0 实测）**：

路径 A（联调期，恢复头/库一致）：每次 `--force` 安装后，用手写头覆盖安装头，并验证覆盖生效：

```bash
cp op_api/aclnn_bench_no_harness_gru.h \
   $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include/aclnn_bench_no_harness_gru.h
# 验证：
grep -n "TensorList" $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include/aclnn_bench_no_harness_gru.h
#   const aclTensorList  *weightIh,
#   const aclTensorList  *weightHh,
#   const aclTensorList  *biasIh,
```

注意：曾尝试在 op_host/CMakeLists.txt 里给 `cust_opapi` 加 `add_custom_command(POST_BUILD ...)` 自动拷贝头到 autogen 路径——`cust_opapi` 是 OBJECT 库，CMake 不支持对 OBJECT 库用 POST_BUILD，构建报错后回退为手工覆盖。

路径 B（测试适配，面向 autogen 接口）：若当期实现只支持单层单向，ST 直接按 autogen 的单张量签名传参（不再为 weight/bias 造 TensorList），测试侧从 `aclCreateTensorList` 改为逐个 `aclCreateTensor` 传入。

正式修复方向：改 op_def，把 weight/bias 注册成与手写 aclnn 一致的形态（若目标版本 op_def 支持动态输入），或把手写 aclnn 降到与 op_def 一致的单张量签名，使三处（op_def/autogen 头/.so 符号）同构。两条过渡路径都属 safety_fallback：A 每次重装都要重覆盖（易漏）、只在本机 vendors 生效不可作为发布手段；B 隐含接口收缩，多层双向场景必须回到正式修复。

## 设计决策

- **签名事实源 = 生成头**：生成头由 op_build 从 OpDef 机械展开，属性位置/输出 `Out` 后缀等约定随版本演进，读当次产物比背文档可靠。
- **先独立预构建测试**（`tests/aclnn` 下单独 `rm -rf build && mkdir build && cd build && cmake .. && make`，已安装的 vendor 包即可支撑链接），通过后再跑官方 `./run.sh`（内部重跑全量 build.sh + 测试）——测试自身的编译错不必烧一次全量重建。
- **桩阶段零值输入**：不构造数值数据，`aclrtMemset` 清零保证运行确定性；冒烟只证明调用通路（含 tiling 执行与 kernel 启动），不证明数值正确——这两件事在验收口径里是分开的。
- **dtype 覆盖对齐 OpDef 枚举**：3 例 = DataType 列表长度 = 产出的 kernel `.o` 数，三方对账。
- **联调期选手工覆盖（A）而不是改 op_def**：op_def 变更会牵动 infershape/tiling/生成链路，联调窗口内风险大；覆盖头是零侵入的可逆操作。
- **迭代一最终走 B**：接口以 op_def 实际注册为准（单张量），测试跟着真实接口走比强行维持设计稿的 TensorList 更诚实，后续迭代支持多层时再统一升级三处。
- 当前方案属 recommended_fastpath（接口桩阶段验收的官方口径路径）。

## 系统信息

- CANN 9.0.0-beta.2 / ascend950（arch35）；`ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp`。
- 目录布局：`operators/<Pkg>_package/tests/aclnn/{CMakeLists.txt, run.sh, test_aclnn.cpp}`；include 路径 `${CANN_HOME}/opp/vendors/<OpType>/op_api/include`（安装后的生成头）；链接 `ascendcl, nnopbase, cust_opapi`。
- 相关文件：`op_host/<op>_def.cpp`（OpDef，决定生成签名）、`build/autogen/aclnn_<op>.h`（签名事实源）、`$ASCEND_OPP_PATH/vendors/<OpType>/op_api/{include,lib}`（安装树）。
- 另两次会话（CANN 9.1.0 / 9.2.0，Ascend950）的相关文件：`op_host/euclidean_norm_def.cpp`（OpDef 布尔属性注册与 8 组 dtype 矩阵）、`build/autogen/aclnn_euclidean_norm.h`（生成签名核对）、`op_host/bench_no_harness_gru_def.cpp` + `op_api/aclnn_bench_no_harness_gru.{h,cpp}`（单张量注册 × 手写 TensorList 错配）、`tests/st/test_aclnn_bench_no_harness_gru.cpp`（报错消费方）。

## 完整代码

`tests/aclnn/test_aclnn.cpp` 终版全文（verbatim 取自当次终版写入；2 输入/1 bool 属性/5 输出算子，fp32/fp16/bf16 三例）：

```cpp
/**
 * =============================================================================
 * <Pkg>_package/tests/aclnn/test_aclnn.cpp
 * =============================================================================
 * Role: ACLNN 接口冒烟测试（接口桩阶段）。
 *       只验证最小调用通路：aclnn<Op>GetWorkspaceSize →
 *       aclnn<Op> → aclrtSynchronizeStream 两段式调用返回 OK
 *       即 PASS，不做数值精度校验（数值用例由后续 kernel 实现阶段验收）。
 *       dtype 覆盖 = 算子原型 dtype 组合表的三行：fp32 / fp16 / bf16。
 * =============================================================================
 */

#include <cstdint>
#include <cstdio>

#include "acl/acl.h"
// 本算子 vendor 包的公共 ACLNN API 头（build 后安装于
// $ASCEND_OPP_PATH/vendors/<OpType>/op_api/include/）。
#include "aclnn_thnn_fused_gru_cell_grad.h"

#define CHECK_ACL(x)                                                          \
    do {                                                                      \
        aclError __ret = (x);                                                 \
        if (__ret != ACL_SUCCESS) {                                           \
            fprintf(stderr, "ACL error %d at %s:%d: %s\n",                    \
                    __ret, __FILE__, __LINE__, #x);                           \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define CHECK_ACLNN(x)                                                        \
    do {                                                                      \
        aclnnStatus __ret = (x);                                              \
        if (__ret != OK) {                                                    \
            fprintf(stderr, "ACLNN error %d at %s:%d: %s\n",                  \
                    __ret, __FILE__, __LINE__, #x);                           \
            return -1;                                                        \
        }                                                                     \
    } while (0)

namespace {

// 按 dtype 取元素字节数：fp32=4；fp16/bf16=2（无 aclBFloat16 类型，bf16 仅按
// ACL_BF16 枚举 + 按字节分配即可）。
size_t ElemSize(aclDataType dtype)
{
    return (dtype == ACL_FLOAT) ? 4 : 2;
}

// 最小调用通路冒烟用例：两段式 aclnn 调用全链返回 OK 即 PASS。
int RunSmokeCase(const char* caseName, aclDataType dtype)
{
    constexpr int64_t kB = 4;
    constexpr int64_t kH = 8;

    const int64_t gradHyDims[2] = {kB, kH};
    const int64_t gradHyStrides[2] = {kH, 1};
    const int64_t storageDims[2] = {kB, 5 * kH};
    const int64_t storageStrides[2] = {5 * kH, 1};
    const int64_t gatesDims[2] = {kB, 3 * kH};
    const int64_t gatesStrides[2] = {3 * kH, 1};
    const int64_t gradHxDims[2] = {kB, kH};
    const int64_t gradHxStrides[2] = {kH, 1};
    const int64_t biasDims[1] = {3 * kH};
    const int64_t biasStrides[1] = {1};

    const size_t elem = ElemSize(dtype);
    const size_t gradHyBytes = static_cast<size_t>(kB * kH) * elem;
    const size_t storageBytes = static_cast<size_t>(kB * 5 * kH) * elem;
    const size_t gatesBytes = static_cast<size_t>(kB * 3 * kH) * elem;
    const size_t gradHxBytes = static_cast<size_t>(kB * kH) * elem;
    const size_t biasBytes = static_cast<size_t>(3 * kH) * elem;

    // --- 设备内存分配（桩阶段不构造/校验数据，清零保持运行确定性） ---
    void* devGradHy = nullptr;
    void* devStorage = nullptr;
    void* devGradInputGates = nullptr;
    void* devGradHiddenGates = nullptr;
    void* devGradHx = nullptr;
    void* devGradInputBias = nullptr;
    void* devGradHiddenBias = nullptr;
    CHECK_ACL(aclrtMalloc(&devGradHy, gradHyBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devStorage, storageBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradInputGates, gatesBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHiddenGates, gatesBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHx, gradHxBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradInputBias, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHiddenBias, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemset(devGradHy, gradHyBytes, 0, gradHyBytes));
    CHECK_ACL(aclrtMemset(devStorage, storageBytes, 0, storageBytes));
    CHECK_ACL(aclrtMemset(devGradInputGates, gatesBytes, 0, gatesBytes));
    CHECK_ACL(aclrtMemset(devGradHiddenGates, gatesBytes, 0, gatesBytes));
    CHECK_ACL(aclrtMemset(devGradHx, gradHxBytes, 0, gradHxBytes));
    CHECK_ACL(aclrtMemset(devGradInputBias, biasBytes, 0, biasBytes));
    CHECK_ACL(aclrtMemset(devGradHiddenBias, biasBytes, 0, biasBytes));

    // --- 创建 aclTensor 描述符（2D/1D 连续 ND 布局） ---
    aclTensor* gradHy = aclCreateTensor(gradHyDims, 2, dtype, gradHyStrides, 0,
                                        ACL_FORMAT_ND, gradHyDims, 2, devGradHy);
    aclTensor* storage = aclCreateTensor(storageDims, 2, dtype, storageStrides, 0,
                                         ACL_FORMAT_ND, storageDims, 2, devStorage);
    aclTensor* gradInputGates = aclCreateTensor(gatesDims, 2, dtype, gatesStrides, 0,
                                                ACL_FORMAT_ND, gatesDims, 2, devGradInputGates);
    aclTensor* gradHiddenGates = aclCreateTensor(gatesDims, 2, dtype, gatesStrides, 0,
                                                 ACL_FORMAT_ND, gatesDims, 2, devGradHiddenGates);
    aclTensor* gradHx = aclCreateTensor(gradHxDims, 2, dtype, gradHxStrides, 0,
                                        ACL_FORMAT_ND, gradHxDims, 2, devGradHx);
    aclTensor* gradInputBias = aclCreateTensor(biasDims, 1, dtype, biasStrides, 0,
                                               ACL_FORMAT_ND, biasDims, 1, devGradInputBias);
    aclTensor* gradHiddenBias = aclCreateTensor(biasDims, 1, dtype, biasStrides, 0,
                                                ACL_FORMAT_ND, biasDims, 1, devGradHiddenBias);

    // --- Phase 1: 查询 workspace 大小并获取 executor ---
    // 签名照抄生成头：输入(gradHy, storage) → 属性(bool hasBias) →
    // 输出×5 → workspaceSize → executor。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnThnnFusedGruCellGradGetWorkspaceSize(
        gradHy, storage, true, gradInputGates, gradHiddenGates, gradHx,
        gradInputBias, gradHiddenBias, &workspaceSize, &executor));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // --- Phase 2: 下发算子并等待完成 ---
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACLNN(aclnnThnnFusedGruCellGrad(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    fprintf(stdout, "[smoke] %s PASSED\n", caseName);

    // --- 清理 ---
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclDestroyTensor(gradHy);
    aclDestroyTensor(storage);
    aclDestroyTensor(gradInputGates);
    aclDestroyTensor(gradHiddenGates);
    aclDestroyTensor(gradHx);
    aclDestroyTensor(gradInputBias);
    aclDestroyTensor(gradHiddenBias);
    aclrtFree(devGradHy);
    aclrtFree(devStorage);
    aclrtFree(devGradInputGates);
    aclrtFree(devGradHiddenGates);
    aclrtFree(devGradHx);
    aclrtFree(devGradInputBias);
    aclrtFree(devGradHiddenBias);
    return 0;
}

} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    int failed = 0;
    // dtype 覆盖 = 算子原型 dtype 组合表的三行（fp32 / fp16 / bf16）。
    if (RunSmokeCase("thnn_fused_gru_cell_grad fp32", ACL_FLOAT) != 0) {
        failed++;
    }
    if (RunSmokeCase("thnn_fused_gru_cell_grad fp16", ACL_FLOAT16) != 0) {
        failed++;
    }
    if (RunSmokeCase("thnn_fused_gru_cell_grad bf16", ACL_BF16) != 0) {
        failed++;
    }

    fprintf(stdout, "\n==== ThnnFusedGruCellGrad aclnn smoke summary: %d/3 cases PASSED ====\n",
            3 - failed);

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclrtFinalize());
    return (failed == 0) ? 0 : -1;
}
```

## 预防

- 写 aclnn 测试前固定先 `cat build/autogen/aclnn_<op>.h`，参数顺序/命名逐字照抄。
- 测试文件写完先做廉价自检（异常词 grep、花括号平衡、`int main` 存在）再编译——长文件一次写入可能损坏。
- 测试用例数与 OpDef `DataType({...})` 列表长度、kernel `.o` 数三方对账。
- 评审时把"op_def 输入注册形态"与"手写 aclnn 签名"放在同一 diff 里看；提交前跑一次三信号核对（安装头 vs .so 符号 vs op_def）。
- 不要给 OBJECT 库目标挂 POST_BUILD（CMake 不支持）；用 add_custom_target 依赖关系或装后脚本。
- 不读 autogen 头、直接按 OpDef 里的 snake_case 属性名写 aclnn 调用——生成形参名是 camelCase。

## 验证信号

以下信号来自同一次"样例重建为接口桩"会话：按生成头签名写冒烟测试后，独立预构建与官方 run.sh 两条路径均通过。

- 独立预构建：`TEST_BUILD_EXIT=0`，`fp32/fp16/bf16` 3/3 PASSED，`TEST_RUN_EXIT=0`（来源平台：950；证据等级：tool_observed）。
- 官方 `tests/aclnn/run.sh`：`RUN_EXIT=0`、`FAILED` 计数 0、3/3 PASSED（来源平台：950；证据等级：tool_observed）。
- `nm -D libcust_opapi.so` 同时导出 `aclnnThnnFusedGruCellGrad` 与 `aclnnThnnFusedGruCellGradGetWorkspaceSize`（来源平台：950；证据等级：tool_observed）。
- `ldd` 测试可执行文件将 `libcust_opapi.so` 解析到本工程 vendor 安装目录；`objdump -T` 存在对 `GetWorkspaceSize` 的 UND 引用（来源平台：950；证据等级：tool_observed）。

以下信号来自另两次会话，补充 OpDef 注册写法、dtype 矩阵与手写错配判别（Ascend950）：

- 改写前基线：`opp/vendors/EuclideanNorm/op_impl/ai_core/tbe/kernel/ascend950/euclidean_norm/` 下 2 个 `.o`（旧接口 fp16/fp32 两组合）；改写后构建 exit 0、无 `do not registe tiling struct` 告警；安装目录 8 个 `.o`（= dtype 组合数）；`libcust_opapi.so` 符号 `aclnnEuclideanNorm` 与 `aclnnEuclideanNormGetWorkspaceSize` 均存在（来源平台：950；证据等级：tool_observed）。
- autogen 头确认签名：`(x, axes, bool keepDims, out, &workspaceSize, &executor)`——属性名 snake→camel（keep_dims→keepDims）、bool 型、输入后输出前（来源平台：950；证据等级：tool_observed）。
- 冒烟 `run.sh`：8/8 dtype 组合 PASSED、EXIT=0（桩 kernel，验证调用链路非数值）（来源平台：950；证据等级：tool_observed）。
- 手写错配场景：GRU 类算子 ST 编译报 `error: cannot convert 'aclTensorList*' to 'const aclTensor*'`；`nm -D libcust_opapi.so` 导出 `aclnnBenchNoHarnessGruGetWorkspaceSize`（手写 TensorList 实现的符号），与安装头（autogen 单张量）不一致（来源平台：950；证据等级：tool_observed）。
- 路径 A 覆盖安装头后 ST 编译通过、进入 NPU 执行；路径 B 适配后（重建重装），ST 重新编译通过并执行到用例输出（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖**调用通路**冒烟，不含数值校验——不能以此证明 kernel 数值正确（数值用例属 kernel 实现阶段验收）。
- zero-size 张量（has_bias=false 的空 bias）、非 ND 格式、`workspaceSize == 0` 时仍传非空 workspace 等边界未测。
- 签名映射规律（属性位置、`Out` 后缀）由本例单个 OPTIONAL Bool 属性归纳；ListAttr/必选属性/多属性形态未验证。
- snake→camelCase 转换规则以单属性（keep_dims→keepDims）为证；多词/下划线连缀属性（如 eps_len→epsLen）与可选属性的 Optional 后缀行为未验证。
- 生成头路径与内容随 CANN 版本演进——以当次构建产物为准（推断规律可迁移，未验证）。
- 路径 A 的装后覆盖只在本机 vendors 生效，换机/重装即失效；不可作为发布手段。路径 B 隐含"当期不支持多层"的接口收缩。
- op_def 侧动态输入注册修法未验证（本版本 `register/op_def.h` 未找到 DynamicInput API）。
- 走 aclnn_exclude 路径的算子另有一类"安装头为生成版、与源码签名不一致"的错配（见 [aclnn_exclude 头文件自动生成覆盖源码致签名不符卡](aclnn_exclude.md)），与本卡的"op_def 注册形态决定签名"是不同生成链机制。

## 坏实践

- 凭记忆或别家算子的例子拼 aclnn 签名（属性位置/输出 `Out` 后缀最易错位）——应读当次生成头。
- 长测试文件一次写完不回读自检——当次会话首写即在文件中段混入垃圾片段且尾部缺失，靠回读检查（异常词 grep、花括号平衡）才发现后整文件重写。
- 把"冒烟 3/3 PASSED"当数值正确性证据——通路与数值是两个验收维度。
- 只看手写头就断言 aclnn 接口形态（消费方编译用的是安装头）。
- 在 OBJECT 库上用 POST_BUILD 拷贝文件（CMake 直接报错）。
- 安装头被覆盖后不 grep 验证就继续编译。
- 在 OpDef 体系里找 `.ATTR(...)` 链式写法（那是 REG_OP 体系的），或反之在 REG_OP 里用 `.Attr().Bool()`。

## 误诊

- 曾担心生成 stub 里有 dtype/format/attr 的隐藏校验导致调用失败——展开生成 `.cpp` 确认是标准 Nnopbase 通路：张量描述透传 + bool 属性 `NnopbaseAddAttrWithDtype` 直传，无额外校验（生成码 grep 佐证）。
- 手写错配场景初期以为是 include 路径指错（把源码头和安装头混用）；核对 ST CMakeLists 的 include 顺序后确认指向安装头，问题在签名不一致而非路径。

## 未决事项

- 可接续：kernel 实现阶段的数值用例（host golden 构造 + 逐元素比对 + 多 shape 矩阵）与 `has_bias=false` 空 bias 分支用例。
- 统一 op_def/autogen/手写 aclnn 三处签名的正式修复（TensorList 或单张量二选一），去掉装后覆盖步骤——优先项。
- 多词属性名 camelCase 规则、可选属性 Optional 后缀行为未实测。

## 增量（单输出统一改名 out + OPTIONAL int 属性物化 + 属性不进 kernel 入口）

另一独立轨迹并入"1 输入 + OPTIONAL int 属性 + 1 输出"形态的实测（softmax 类算子，CANN 9.1.0 编译 / 9.2.0 运行），补三条签名规则：

**1. 单输出统一改名 `out`**：OpDef `Output("y")` 在 autogen 头中生成 `const aclTensor *out`——**单输出不按名字追加后缀，统一命名为 `out`**（与多输出的"camelCase 名 + Out 后缀"规则并存：单输出 y→out，多输出 grad_input_gates→gradInputGatesOut）。手写调用照抄 OpDef 输出名（`y`）会编译失败或参数错位。

**2. OPTIONAL int 属性物化为显式参数**：`this->Attr("dim").AttrType(OPTIONAL).Int(-1L)` 生成 `int64_t dim` 显式参数，插在输入与输出之间——**带默认值的 OPTIONAL 属性也不省略参数**，测试调用必须传值，默认值语义在框架层生效。

**3. 属性不进 kernel 入口参数表**：kernel 入口签名与属性无关，`GM_ADDR` 参数表 = 各输入 + 各输出 + `GM_ADDR workspace, GM_ADDR tiling`；桩阶段保留框架 4 行样板（`AscendC::InitSocState();` / `REGISTER_TILING_DEFAULT(<Op>TilingData);` / `GET_TILING_DATA(tilingData, tiling);` / `AscendC::PipeBarrier<PIPE_ALL>();`）。

生成头 verbatim（单输出 + int 属性形态，注意输出参数名 `out` 与属性参数位置）：

```c
aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnn<Op>(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

配套 kernel 入口桩（属性 `dim` 不出现在参数表，verbatim）：

```cpp
extern "C" __global__ __aicore__ void <op>(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(<Op>TilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

流程顺序固化：改 OpDef 后**先构建**（autogen 头是构建产物，不重建就是旧契约）→ 读 `build/autogen/aclnn_<op>.h`（或安装后的 `vendors/<Op>/op_api/include/`）→ 按生成头原型写测试调用，不凭 proto/设计文档手推。

### 复现增量验证信号

- 构建：`./build.sh` 退出码 0；生成 3 个 kernel 二进制（fp16/fp32/bf16 三组 dtype 各一），安装到 `$ASCEND_OPP_PATH/vendors/<Op>/`（来源平台：950；证据等级：tool_observed）。
- autogen 契约：`build/autogen/aclnn_<op>.h` 中 `GetWorkspaceSize` 原型为 `(const aclTensor *x, int64_t dim, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`——输出参数名为 `out` 而非 OpDef 中的 `y`，OPTIONAL 属性物化为显式 `int64_t dim` 参数插在输入与输出之间（来源平台：950；证据等级：tool_observed）。
- tiling 注册检查：构建日志无 `do not registe tiling struct`；`nm -D libcust_opapi.so` 输出 `T aclnn<Op>` 与 `T aclnn<Op>GetWorkspaceSize`（来源平台：950；证据等级：tool_observed）。
- 冒烟：按 autogen 签名编写的测试 3/3 dtype PASSED（两段式调用返回 OK；桩阶段无数值校验），`run.sh` 退出码 0（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 单输出改名 `out` 与 OPTIONAL int 属性物化在 CANN 9.1.0（编译）/9.2.0（运行）实测；REQUIRED 属性、ListAttr、动态 shape 的签名形态仍未验证。
- autogen 头是构建产物：改 OpDef 后不重新构建则头文件过期，照旧头写测试即使通过也不代表新接口正确。

## 增量（运行时段错误面目与测试侧 flatten 适配闭环）

两条独立 trace 把"单张量注册 × TensorList 错配"的两个落地面目补全（均为 GRU/LSTM 类权重参数，Ascend950PR + CANN 9.2.0）：

**面目一（测试侧，编译期）**：op_proto 把 5 个权重参数（weightIh/weightHh/biasIh/biasHh/weightHr）注册为单 `INPUT`（非 `DYNAMIC_INPUT`），生成头对权重收 `const aclTensor*` 且 kernel 侧按 1D flat 拼接布局 + offset 访问（`GetWeightIhOffset = Σ_{l<layer} numDirs*gateSize*inDim(l) + dir*gateSize*inDim`，`GetWeightHhOffset = (layer*numDirs+dir)*gateSize*H`，`GetBiasOffset = (layer*numDirs+dir)*gateSize`）；测试脚手架按 PyTorch per-layer 列表语义构造 `aclTensorList*` 并错排标量参数，与生成 API 契约不匹配：

```text
test_aclnn_<op>.cpp:781:16: error: cannot convert 'aclTensorList*' to 'const aclTensor*'
  781 |         wihRes.tensorList,
      |               ~~~~~~~^~~~~~~~~~
      |                      |
      |                      aclTensorList*
note: initializing argument 2 of 'aclnnStatus aclnn<Op>GetWorkspaceSize(const aclTensor*, ...)'
acl_meta.h:30:16: note: class type 'aclTensorList' is incomplete
```

修复（路径 B 的完整落地）：加 `flattenWeights` 辅助把 `vector<vector<float>>` 展平为单个 `vector<float>`（拼接顺序即 kernel offset 语义的 `layer*numDirections + dir` 逐段连续）；权重用 `CreateAclTensor(flat, {totalNumel}, ...)` 创建单个 `aclTensor*`（1D shape 总元素数——生成侧 CheckShape 只校验总元素数）；参数顺序严格对齐生成头；可选权重按条件传 `nullptr`；清理路径从 `DestroyTensorList` 换成逐个 `aclDestroyTensor`。修复后 Real 模式 L0 4/4 PASS（y MERE=8.74e-07）、Mock 回归 4/4、独立验收复跑 4/4。

**面目二（手写 L2 API 侧，运行期段错误）**：OpDef 将 weight_ih/weight_hh/bias_ih/bias_hh 声明为单 Tensor（REQUIRED），按 OpDef 生成的 aclnn 头与 ST 测试都传 `const aclTensor*`；而手写的 L2 API 把这四个参数声明为 `const aclTensorList*` 并在参数校验里调用 `weightIh->Size()`——`aclTensor*` 被当作 `aclTensorList*` 解引用，两者内存布局完全不同，直接段错误。崩溃点定位用测试侧 fflush 分段打印（无 core dump 时最快收敛）：

```text
  [DEBUG] Creating output tensors...
  [DEBUG] Calling GetWorkspaceSize...
（之后进程直接段错误，无错误码、无 core dump）
```

修复：以生成头文件为准（用户实际调用的契约）重写 L2 API 为 `aclTensor*` 直传；注意 op_api 目录下的**本地** `aclnn_*.h` 必须与 .cpp 同步改，只改 .cpp 会报 `conflicting declaration`（本地头仍声明 `aclTensorList*`）。修复后重建安装，同批用例 GetWorkspaceSize 全部正常返回（推进到流同步阶段）。

判别补充：host 侧崩溃无错误码时，先用 fflush 分段打印定位崩溃函数，再对照"安装头 vs L2 API 源码形参类型 vs op_def 输入注册"三处签名。

坏实践（独立 trace 增补）：

- 手写 L2 API 时凭"GRU/LSTM 权重按层/方向是列表"的语义直觉直接用 `aclTensorList*`，而不核对 OpDef 与生成头文件的实际契约。
- 编译错只改类型不改参数顺序（标量参数错位会让错误从编译期变成静默错误行为）。
- 只改实现 .cpp 不改同目录的声明 .h，再被 conflicting declaration 打回一轮。

验证信号（独立 trace 增补，两条 trace 的 ST 联调，Ascend950PR / CANN 9.2.0）：

- 面目一修复前：`bash run.sh --real` 编译失败 `error: cannot convert 'aclTensorList*' to 'const aclTensor*'`（来源平台：950；证据等级：tool_observed）。
- 面目一修复后：Real 模式 L0 4/4 PASS（L0-001 y MERE=8.74e-07、MARE=1.32e-04），Mock 回归 4/4（MERE=0），测试工程师独立验收（清理 build 重跑 Real）再次 4/4（来源平台：950；证据等级：tool_observed）。
- 面目二修复前：5 个 L0 用例均在 "[DEBUG] Calling GetWorkspaceSize..." 后进程段错误（无错误码、无 core dump）（来源平台：950；证据等级：tool_observed）。
- 面目二中间步骤：仅改 .cpp 时构建报 conflicting declaration；同步改本地 .h 后构建成功（来源平台：950；证据等级：tool_observed）。
- 面目二修复后：同批 5 个用例 GetWorkspaceSize 全部正常返回，测试推进到 `aclrtSynchronizeStream` 阶段（报 507015 kernel 执行错误，属下一独立问题）（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 面目二的段错误直接机制（`aclTensorList::Size()` 读到非法内存）由类型布局差异推断，未做反汇编级验证。
- 两面目均在 950 + CANN 9.2.0 + REQUIRED 单 Tensor OpDef 场景实测；多层/双向的 flat 拼接上板验证与 projSize>0 布局未实测（后续迭代范围；多层 flat 拼接的补零坑另见 rnn weight_ih flat 拼接卡）。

