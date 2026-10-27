---
type: "Runbook"
title: PyTorch 接入 TensorList+scalar 算子 ST 工程
description: "torch_adapter.cpp 完整配方：TORCH_LIBRARY 注册 Tensor[] 签名（Meta+PrivateUse1）、c10::ScalarType→aclDataType 映射、逐元素 aclCreateTensor + aclCreateTensorList 包装、double 标量直传、OpWorkspace RAII（列表所有权）、RunOpApiV2 入队执行、golden/compare 向量化比对。复现增量：scalar 走 1 元素 FP32 aclTensor 的头文件契约变体（ForeachNorm，74/74）+ CANN 9.1.0 堆污染下的直调两段式/全泄漏/每用例子进程隔离形态 + 非连续 stride 保留与 FP64 golden。"
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# PyTorch 接入 TensorList+scalar 算子 ST 工程

## 现象

一个已通过 C++ ST 验证的自定义 aclnn 算子（`y_i = x1_i + (x2_i / x3_i) * scalar`，三个 TensorList 输入 + double 标量 + TensorList 输出，目标 Ascend950PR / CANN 9.1.0），需要做 PyTorch 接入的 L0+L1 批量精度验收（65 用例，含 64M 元素大 shape、FP16/FP32/BF16、特殊值）。官方模板只有单 Tensor + 标量的 add 示例，TensorList（`Tensor[]`）签名、多输入列表的 acl 资源组织、以及 Python 长驻进程下的资源清理都没有现成参考。本卡是成功实现的完整工程配方（validated_pattern）：torch_adapter.cpp + golden.py + compare.py + test.py + CMakeLists.txt 一次落地，65/65 NPU 通过、exit 0 无泄漏报错。期间踩掉两个坑：aclTensorList 所有权（析构双重销毁，另行沉淀）与 Python 循环比对超时（64M 元素用例 5 分钟超时，向量化后秒级）。

## 适用场景

- 工程形态：aclnn 两段式自定义算子的 PyTorch 接入测试（`python3 test.py --lib ./build/libtorch_adapter.so`），torch 2.7.1 + torch_npu 环境。
- API 路径：`TORCH_LIBRARY_FRAGMENT/TORCH_LIBRARY_IMPL`（Meta + PrivateUse1）、`aclCreateTensor/aclCreateTensorList/aclDestroyTensorList`、`c10_npu::getCurrentNPUStream().stream(false)`、`at_npu::native::OpCommand::RunOpApiV2`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post5 / Python 3.13（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——适用于任何 TensorList+标量的 aclnn 算子 PyTorch 适配；单 Tensor 算子可直接套官方模板更简单。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：反向传播/autograd（仅前向）；CPU/Mock 模式（需真 NPU）；其他 torch/torch_npu 版本的 OpCommand 接口兼容性未验证。

## 解决配方

适配层四要素（相对官方单 Tensor 模板的增量点）：

1. **注册签名用 `Tensor[]`**：schema `forward(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[]`；Meta 实现做 list 长度/逐元素 shape/dtype 一致性校验并 `at::empty_like` 分配输出；PrivateUse1 实现挂 NPU 前向。
2. **TensorList 资源组织**：逐元素 `CreateAclTensor`（裸指针 + shape + strides + storageDims=shape + deviceAddr），再 `aclCreateTensorList(const aclTensor* const*, uint64_t)` 包装；标量按 API 签名直接传 `double`（本算子 L2 形参是 double，不需要 aclCreateScalar）。
3. **两段式执行**：GetWorkspaceSize 在 host 侧立即调用；workspace 用 `torch::empty(kByte)` 分配（生命周期归 PyTorch tensor 管，不用 aclrtMalloc/aclrtFree）；真正执行经 `OpCommand::RunOpApiV2` 以 lambda 入 NPU 队列，lambda 内执行后 `delete ws` 释放 acl 资源。
4. **比对必须向量化**：MERE/MARE/near-zero/nan-inf 全部用 torch 张量运算，禁止 Python 逐元素循环（64M 元素用例循环比对 >5 分钟超时，向量化后整套 65 用例秒级）。

## 设计决策

- workspace 交给 `torch::empty` 管理：避免 host 侧手动 aclrtFree 与异步执行的时序问题，lambda 捕获的 `workspace_tensor` 保证引用存活到执行完成。
- 输出张量由 Meta 的 `empty_like` 创建、PrivateUse1 复用 `forward_meta` 同款分配：保持 torch 算子协议（Meta 先行），非连续输入在适配层 `.contiguous()` 归一（算子侧非连续支持另有边界，见非连续 Tensor 相关失败路线卡）。
- `getCurrentNPUStream().stream(false)` 取流不同步：执行序由 `RunOpApiV2` 入队保证。
- 析构所有权采用"列表优先、tensor 兜底"（`aclDestroyTensorList` 会销毁内含 tensor；列表未建成时才逐个销毁）——这是首版 double-free 后的修正，细节另行沉淀。

## 系统信息

- 工具链：CANN 9.1.0（`ASCEND_HOME_PATH` 指向 cann-9.1.0 安装根）；torch 2.7.1+cpu、torch_npu 2.7.1.post5、Python 3.13；自定义算子包安装于 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/{include,lib}`。
- 目标平台：Ascend950PR（arch35 / DAV_3510）。
- 目录布局：`tests/st/torch/{CMakeLists.txt, torch_adapter.cpp, golden.py, compare.py, test.py, build/libtorch_adapter.so}`；`build/`、`__pycache__/` 入 .gitignore。
- 相关文件：CMakeLists（find_package torch/torch_npu + CANN include/lib + vendor 头与 libcust_opapi）、golden.py（CPU golden：量化到目标 dtype 再升精度计算）、compare.py（向量化 MERE/MARE + 近零绝对误差兜底 + nan/inf 位匹配）、test.py（65 用例表 + 调度 + 汇总）。

## 完整代码

torch_adapter.cpp 核心实现（verbatim 取自终版，含 double-free 修正后的析构；版权头与文件级注释见源事件，此处节选功能段完整函数）：

```cpp
namespace {

// PyTorch c10::ScalarType -> CANN aclDataType 映射
aclDataType ScalarTypeToAclDType(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Byte:    return ACL_UINT8;
        case c10::ScalarType::Char:    return ACL_INT8;
        case c10::ScalarType::Int:     return ACL_INT32;
        case c10::ScalarType::Long:    return ACL_INT64;
        case c10::ScalarType::Half:    return ACL_FLOAT16;
        case c10::ScalarType::Float:   return ACL_FLOAT;
        case c10::ScalarType::Double:  return ACL_DOUBLE;
        case c10::ScalarType::Bool:    return ACL_BOOL;
        case c10::ScalarType::BFloat16:return ACL_BF16;
        default: return ACL_DT_UNDEFINED;
    }
}

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclTensor* CreateAclTensor(const void* data_ptr,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    aclTensor* tensor = aclCreateTensor(
        shape.data(),
        shape.size(),
        dtype,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        const_cast<void*>(data_ptr)
    );
    return tensor;
}

// Workspace 封装 (持有 TensorList 中所有 aclTensor + aclTensorList)
struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<aclTensor*> x1_tensors;
    std::vector<aclTensor*> x2_tensors;
    std::vector<aclTensor*> x3_tensors;
    std::vector<aclTensor*> y_tensors;
    aclTensorList* x1_list = nullptr;
    aclTensorList* x2_list = nullptr;
    aclTensorList* x3_list = nullptr;
    aclTensorList* y_list = nullptr;

    ~OpWorkspace() {
        // aclDestroyTensorList 会销毁其包含的 aclTensor，因此：
        // - 列表已创建时: 仅销毁列表 (列表负责销毁内部 tensor)
        // - 列表未创建时 (创建失败): 逐个销毁 tensor
        if (x1_list) { aclDestroyTensorList(x1_list); x1_list = nullptr; }
        else { for (auto* t : x1_tensors) { if (t) aclDestroyTensor(t); } }
        if (x2_list) { aclDestroyTensorList(x2_list); x2_list = nullptr; }
        else { for (auto* t : x2_tensors) { if (t) aclDestroyTensor(t); } }
        if (x3_list) { aclDestroyTensorList(x3_list); x3_list = nullptr; }
        else { for (auto* t : x3_tensors) { if (t) aclDestroyTensor(t); } }
        if (y_list)  { aclDestroyTensorList(y_list);  y_list = nullptr; }
        else { for (auto* t : y_tensors)  { if (t) aclDestroyTensor(t); } }
        executor = nullptr;
    }
};

// 第一阶段: 创建 aclTensor/aclTensorList + GetWorkspaceSize
OpWorkspace* OpGetWorkspace(
    const std::vector<torch::Tensor>& x1_contig,
    const std::vector<torch::Tensor>& x2_contig,
    const std::vector<torch::Tensor>& x3_contig,
    std::vector<torch::Tensor>& y,
    aclDataType dtype,
    double scalar) {

    auto ws = new OpWorkspace();
    size_t list_len = x1_contig.size();

    for (size_t i = 0; i < list_len; ++i) {
        auto shape = x1_contig[i].sizes().vec();
        ws->x1_tensors.push_back(CreateAclTensor(x1_contig[i].data_ptr(), shape, dtype));
        ws->x2_tensors.push_back(CreateAclTensor(x2_contig[i].data_ptr(), shape, dtype));
        ws->x3_tensors.push_back(CreateAclTensor(x3_contig[i].data_ptr(), shape, dtype));
        ws->y_tensors.push_back(CreateAclTensor(y[i].data_ptr(), shape, dtype));
    }

    ws->x1_list = aclCreateTensorList(
        const_cast<const aclTensor* const*>(ws->x1_tensors.data()),
        static_cast<uint64_t>(list_len));
    ws->x2_list = aclCreateTensorList(
        const_cast<const aclTensor* const*>(ws->x2_tensors.data()),
        static_cast<uint64_t>(list_len));
    ws->x3_list = aclCreateTensorList(
        const_cast<const aclTensor* const*>(ws->x3_tensors.data()),
        static_cast<uint64_t>(list_len));
    ws->y_list = aclCreateTensorList(
        const_cast<const aclTensor* const*>(ws->y_tensors.data()),
        static_cast<uint64_t>(list_len));

    if (!ws->x1_list || !ws->x2_list || !ws->x3_list || !ws->y_list) {
        TORCH_CHECK(false, "OpGetWorkspace: aclCreateTensorList failed, "
                    "x1=", (ws->x1_list ? "ok" : "null"),
                    ", x2=", (ws->x2_list ? "ok" : "null"),
                    ", x3=", (ws->x3_list ? "ok" : "null"),
                    ", y=", (ws->y_list ? "ok" : "null"));
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnBenchNoHarnessForeachAddcdivScalarGetWorkspaceSize(
        ws->x1_list, ws->x2_list, ws->x3_list, scalar,
        ws->y_list, &ws->workspace_size, &ws->executor);

    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: GetWorkspaceSize failed, "
                    "aclnnStatus=", status);
        delete ws;
        return nullptr;
    }

    return ws;
}

// 第二阶段: 执行算子
aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream) {
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnBenchNoHarnessForeachAddcdivScalar(
        workspace_ptr, ws->workspace_size, ws->executor, stream);
}

} // anonymous namespace

// NPU 实现: stream(false) + OpCommand + lambda 入 queue
static std::vector<at::Tensor> forward_npu(
    const std::vector<at::Tensor>& x1,
    const std::vector<at::Tensor>& x2,
    const std::vector<at::Tensor>& x3,
    double scalar) {

    TORCH_CHECK(!x1.empty(), "forward: x1 TensorList must not be empty");
    size_t list_len = x1.size();

    // 输入转连续 (支持非连续 Tensor 输入)
    std::vector<torch::Tensor> x1_contig, x2_contig, x3_contig;
    x1_contig.reserve(list_len);
    x2_contig.reserve(list_len);
    x3_contig.reserve(list_len);
    for (size_t i = 0; i < list_len; ++i) {
        x1_contig.push_back(x1[i].contiguous());
        x2_contig.push_back(x2[i].contiguous());
        x3_contig.push_back(x3[i].contiguous());
    }

    // 分配输出 Tensor (与输入同 shape/dtype/device, 连续布局)
    std::vector<torch::Tensor> y;
    y.reserve(list_len);
    for (size_t i = 0; i < list_len; ++i) {
        y.push_back(torch::empty_like(x1_contig[i]));
    }

    auto dtype = ScalarTypeToAclDType(x1_contig[0].scalar_type());

    OpWorkspace* ws = OpGetWorkspace(x1_contig, x2_contig, x3_contig, y, dtype, scalar);
    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    // 分配 workspace (由 PyTorch tensor 管理生命周期)
    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor = torch::empty(
            {static_cast<int64_t>(ws->workspace_size)},
            torch::dtype(torch::kByte).device(x1_contig[0].device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2(
        "ascendc_bench_no_harness_foreach_addcdiv_scalar", acl_call);

    return y;
}

TORCH_LIBRARY_FRAGMENT(bench_no_harness_foreach_addcdiv_scalar, m) {
    m.def("forward(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(bench_no_harness_foreach_addcdiv_scalar, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(bench_no_harness_foreach_addcdiv_scalar, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
```

运行方式：

```bash
cd tests/st/torch
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
cd ..
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/lib:${LD_LIBRARY_PATH}
python3 test.py --lib ./build/libtorch_adapter.so
```

## 预防

- 适配层写完先跑 golden 自测（纯 CPU 路径）再上 NPU，把 golden 错误与接入错误分离。
- 大 shape 用例（≥千万元素）的比对逻辑先做向量化设计评审，不要事后超时再改。

## 验证信号

问题与验证来源：一次 PyTorch ST 适配层开发与最终精度验收会话的构建、自测与全量运行记录。

- golden CPU 自测 7 项全部 PASS（基础复合/零标量/单位分母/负标量/除零/inf 传播/多张量）（来源平台：950；证据等级：tool_observed）。
- `libtorch_adapter.so` 编译链接成功（torch/torch_npu/CANN/vendor 依赖全解析）（来源平台：950；证据等级：tool_observed）。
- 全量 65 用例 NPU 运行：通过 65 / 失败 0 / 跳过 0，exit 0，无 double-free/泄漏报错（含 64M 元素用例）（来源平台：950；证据等级：tool_observed）。
- 逐 dtype 精度汇总（237 次张量级比对）：float32 最差 MARE 6.85e-04、float16 9.63e-04、bfloat16 0.00e+00，均在阈值内（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `OpCommand::RunOpApiV2` 为 torch_npu 内部接口，版本升级可能变动（本轨迹 2.7.1.post5 验证）。
- 适配层在 `.contiguous()` 处归一非连续输入——算子本体的非连续能力边界不受此影响（见非连续 Tensor 相关失败路线卡）。
- 仅前向；反向需另注册 autograd。

**未决项**：

- 未在适配层覆盖非连续输入直传（`contiguous()` 归一后不触发算子非连续路径）。
- torch_npu 升级后 `RunOpApiV2` 兼容性需回归。
## 坏实践

- Python 逐元素循环做 MERE/MARE 比对——64M 元素用例 5 分钟超时；比对必须 torch 向量化。
- workspace 用 aclrtMalloc 手动管理——与异步执行时序耦合，交给 `torch::empty` 更稳。
- 析构两套 acl 句柄全销毁——长驻 Python 进程必报 double-free（aclDestroyTensorList 已销毁内含 tensor，另行沉淀）。

## 误诊

- 无（接入一次成功；两个坑——double-free 与比对超时——均现场定位修复）。

## 后续增量（2026-09-17，并行策展批：scalar 走 1 元素 aclTensor 的变体 + 直调/全泄漏/子进程隔离形态）

ForeachNorm 类算子（BenchNoHarnessForeachNorm：输入 TensorList、对每个张量独立计算 p 范数、输出 0 维标量张量列表，Ascend950PR + CANN 9.1.0）补齐与主卡 double 标量直传并列的另一种头文件契约形态，并记录 CANN 9.1.0 堆污染下的生命周期替代策略：

**1. scalar 走 1 元素 FP32 aclTensor（头文件契约变体）**：该算子生成的 `aclnn<Op>.h` 里 scalar 阶数是 `const aclTensor*`（1 元素 FP32 张量）而非 `double`——需求文档写的 `aclScalar*` 被生成头证伪（文档转述与生成代码不一致时，永远以生成代码为准）。适配层用 `torch::full({1}, scalar, kFloat32, device)` 持有设备内存、`CreateAclTensorFromTorch` 包装，生命周期归 torch tensor。

**2. 0 维标量输出 aclTensor**：`aclCreateTensor(nullptr, 0, dtype, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, ptr)`——viewDims/strides/storage 全空。

**3. 非连续输入不 contiguous 化**：保留原 strides/storage_offset，storage_dims 用 storage 元素总数的一维描述，让算子自己处理 stride 语义（stride=2 用例实测 PASS）；与主卡"适配层 `.contiguous()` 归一"是两种可选策略，按被测算子的非连续支持能力选择。

**4. 直调两段式 + 全泄漏（CANN 9.1.0 堆污染规避）**：不走 `OpCommand::RunOpApiV2`（其内部 `CommonOpExecutorRun` 析构 executor 是一次堆操作，该环境上 ACLNN 执行后堆已污染，任何 `aclDestroy*` 都可能 double free——机制与复现见 [ACLNN CommonOpExecutorRun 析构致 double free](../compilation/common_op_executor_run_double_free.md)）；改为主线程直调 `GetWorkspaceSize` + `aclnn<Op>` + `aclrtSynchronizeStream` 后返回，**不调用任何 aclDestroy***（acl 资源全部泄漏由 OS 回收），`torch::Tensor` 照常 RAII（流同步后析构安全）。这是测试态规避，生产接入需待 CANN 修复后重评。

**5. 每用例独立子进程**：同进程多次调用后 scalar 类参数失效/输出垃圾（BATCH_SIZE=1 是唯一被全量验证安全的批次，12/15 条批均实测失败——判别与独立复现方法见 [ST 连续运行失败而独立 repro 全对的二分](../precision/st_continuous_run_fail_independent_repro_pass.md)）；`test.py` 全量模式每用例 `--case <id> --skip-golden` 自跑自比、父进程汇总，单用例/`--no-fork` 模式留给对照实验。子进程隔离的固定启动成本（torch import 为主）使全量耗时显著高于进程内连续执行——该成对代价是为规避缺陷支付的（来源平台：950；参照路线：进程内连续执行，仅对照用）。

**6. golden FP64 累加**：48M 元素 FP32 顺序累加自身漂移会把 NPU 树形归约误判失败，golden 中间累加统一 FP64（机制见 [大归约维 fp32 golden 顺序累加致假精度失败](../precision/fp32_golden.md)）；golden 内置 22 项自测（正/负/分数阶、±inf、全零、strided、foreach 多张量）。

### 复现增量验证信号

- 编译与加载：`libtorch_adapter.so` 构建成功，`torch.ops.load_library` 加载成功，NPU 设备 Ascend950PR_9579 可见（来源平台：950；证据等级：tool_observed）。
- golden 自测 22/22 通过（正/负/分数阶、±inf、全零、strided、foreach 多张量）（来源平台：950；证据等级：tool_observed）。
- 单用例端到端：L0_001 MERE=2.16e-08 PASS；dtype/scalar/非连续覆盖抽查（L1_002 FP16、L1_003 BF16、L1_008 p=+inf、L1_010 p=-1、L1_063 stride=2）全部 PASS（来源平台：950；证据等级：tool_observed）。
- 全量回归：74/74 通过，FP32 用例（48 条）最大 MERE=6.43e-07、FP16（13 条）/BF16（13 条）0 误差，确定性用例 3 次运行结果一致（来源平台：950；证据等级：tool_observed）。
- 48M 大张量（L1_061）golden 修 FP64 后 MERE=4.07e-08 PASS（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 全泄漏与子进程隔离是 CANN 9.1.0 的测试态规避；长驻服务不适用，需等 CANN 修复后回退全泄漏、评估进程内批量执行（子进程池/按 dtype 分组复用方案未尝试，BATCH_SIZE>1 的失败数据见连续运行判别卡）。
- scalar-as-aclTensor 是该算子生成头文件的契约；aclScalar 形态或其他算子的头文件须按其真实声明调整（本配方未覆盖）。
- 0 维输出、TensorList 进出是该算子形态；普通 Tensor 进出的算子可简化列表包装步骤。
- 适配层对 FP64 输入（ACL_DOUBLE）只做了映射未实测精度；仅 Ascend950PR + CANN 9.1.0 + torch 2.7.1/torch_npu 2.7.1.post5 实测（骨架部分与平台弱耦合，未验证，推断可迁移）。

### 复现增量坏实践增补

- 照搬官方 add_example 的 OpCommand + 析构销毁模板到 CANN 9.1.0 的 aclnn 算子——退出 double free（三种局部修复均失败的记录见 double free 卡）。
- 全量用例在单进程里循环跑——scalar 类用例集体出错（同进程多例失效判别卡）。
- golden 直接用输入 dtype 的 `torch.norm`——48M 用例 golden 自身失真（fp32 golden 卡）。