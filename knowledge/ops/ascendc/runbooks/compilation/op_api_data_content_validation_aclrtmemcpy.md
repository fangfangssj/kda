---
type: "Runbook"
title: "op_api 补数据内容校验：读 device 数据验 Sum=A"
description: "aclnn 接口文档承诺的数据内容类约束（Sum(expertTokenNumPerRank)==A、元素>0）与 H 范围约束未实现时，非法输入不被 161002 拒绝；GetWorkspaceSize 阶段可用 aclTensor::GetData + GetPlacement 分支 + aclrtMemcpy(DEVICE_TO_HOST) 读数据补齐内容校验，\"需要读数据\"不是跳过校验的理由。"
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# op_api 补数据内容校验：读 device 数据验 Sum=A

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个 MoE 重排类自定义算子跑 L2 异常用例（NPU Real 模式）。按 aclnn 接口文档，以下非法输入应被 `aclnnXxxGetWorkspaceSize` 以 `ACLNN_ERR_PARAM_INVALID`（161002）拒绝：

- Sum 约束：`expertTokenNumPerRank`（int32/int64 的 (N,E) 矩阵）所有元素之和必须等于 `tokens` 的第 0 维 A；
- 元素约束：所有元素 > 0（N*E>A 时无法全为正，必违反）；
- H 范围约束：`0 < H < 16384`。

实测两条用例失败，报错原文：

```text
[Real-Exception] L2_016 - 异常_N*E=16大于A=4元素无法全大于0 (dtype=float16/int32, scales=1, A=4, H=32, N=4, E=4, ...)
  [FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）
[Real-Exception] L2_018 - 异常_H=16384违反H小于16384约束 (dtype=float16/int32, scales=1, A=64, H=16384, N=4, E=4, ...)
  [FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）
```

当时的 `op_api/aclnn_<op>.cpp` 的 `CheckParams` 只做了空指针、dtype、维数、属性值（expertTokenNumType==1 / idxType==0）和空 tensor 五类检查——接口文档里的数据内容约束（Sum=A / 元素>0）和 H 范围约束一行都没有。已存在的 3 个对应 UT 用例此前被标成 `DISABLED_` 并注释"校验缺失，参考 issue"。

## 适用场景

- 工程形态：aclnn 自定义算子的 op_api 参数校验补齐（接口文档与实现不一致时以文档为准补实现）。
- API 路径：`aclnnXxxGetWorkspaceSize` → `CheckParams` → `CheckShapeValid`/`CheckDataValid`；`aclTensor` 的 `GetViewShape()/Numel()/GetDataType()/GetData()/GetPlacement()`；`aclrtMemcpy`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"合法性与输入张量数据内容相关"（如计数矩阵行和、元素非负）的 aclnn 算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：数据内容校验需要在大张量上做（本卡输入仅 N*E≤16384 个 int 元素，拷贝开销可忽略；大张量数据校验的性能代价未评估）；ValueDepend 等 host 侧取值的替代机制未在本卡验证。

## 根因

`CheckParams` 的校验清单只覆盖了"纯元数据"维度（指针/dtype/维数/属性/空），两类约束没有被实现：H 范围（其实也是纯 shape 元数据，属于实现遗漏）与 Sum=A/元素>0（需要读张量数据内容，容易被默认"op_api 阶段读不到 device 数据"而搁置）。实际上 `aclTensor::GetData()` 拿到数据指针、按 `GetPlacement()` 判定位置后，device 数据用 `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` 拷到 host 缓冲即可校验——GetWorkspaceSize 阶段完全可以做数据内容校验。

## 解决配方

修复分三处（op_api 层）。

1) `CheckShapeValid` 补 H 范围检查（before → after）：

```cpp
// before：只查维数
    // tokens: 2D (A, H)
    const auto& tokensShape = tokens->GetViewShape();
    OP_CHECK(tokensShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "tokens must be 2D, got %zu dims.", tokensShape.GetDimNum()),
             return false);

// after：插入 H 范围检查
    // tokens: 2D (A, H)
    const auto& tokensShape = tokens->GetViewShape();
    OP_CHECK(tokensShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "tokens must be 2D, got %zu dims.", tokensShape.GetDimNum()),
             return false);

    // H range: 0 < H < 16384 (token 长度约束)
    OP_CHECK(tokensShape.GetDim(1) > 0 && tokensShape.GetDim(1) < 16384,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "tokens dim 1 (H) must satisfy 0 < H < 16384, got H=%ld.",
                     tokensShape.GetDim(1)),
             return false);
```

2) 新增 `CheckDataValid`（读数据内容校验 Sum==A 与元素>0，int32/int64 双 dtype；device 数据走 aclrtMemcpy）：

```cpp
// =============================================================================
// CheckDataValid: 读取 expertTokenNumPerRank 数据，校验 Sum==A 和所有元素>0
// 说明：
//   - 根据 tensor 的 placement 判断数据在 Host 还是 Device
//   - Host 数据直接读取，Device 数据通过 aclrtMemcpy 拷贝到 Host 后读取
//   - 支持 INT32 和 INT64 两种 dtype
// =============================================================================
static aclnnStatus CheckDataValid(const aclTensor* tokens, const aclTensor* expertTokenNumPerRank)
{
    const int64_t A = tokens->GetViewShape().GetDim(0);
    const int64_t numElements = expertTokenNumPerRank->Numel();
    const op::DataType dtype = expertTokenNumPerRank->GetDataType();

    void* tensorData = expertTokenNumPerRank->GetData();
    OP_CHECK(tensorData != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expertTokenNumPerRank data pointer is nullptr."),
             return ACLNN_ERR_PARAM_INVALID);

    // 计算数据总字节数
    size_t elemSize = (dtype == op::DataType::DT_INT32) ? sizeof(int32_t) : sizeof(int64_t);
    size_t totalBytes = static_cast<size_t>(numElements) * elemSize;

    // 根据 placement 判断数据位置，读取到 host 缓冲区
    std::vector<uint8_t> hostBuf(totalBytes);
    auto placement = expertTokenNumPerRank->GetPlacement();
    bool isOnDevice = (placement == op::TensorPlacement::kOnDeviceHbm ||
                       placement == op::TensorPlacement::kOnDeviceP2p);

    if (isOnDevice) {
        // Device 数据：通过 aclrtMemcpy 拷贝到 Host
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

    // 校验：所有元素 > 0 且 Sum == A
    int64_t sum = 0;
    if (dtype == op::DataType::DT_INT32) {
        const auto* data = reinterpret_cast<const int32_t*>(hostBuf.data());
        for (int64_t i = 0; i < numElements; i++) {
            OP_CHECK(data[i] > 0,
                     OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                             "expertTokenNumPerRank element[%ld]=%d, must be > 0.",
                             i, data[i]),
                     return ACLNN_ERR_PARAM_INVALID);
            sum += data[i];
        }
    } else { // DT_INT64
        const auto* data = reinterpret_cast<const int64_t*>(hostBuf.data());
        for (int64_t i = 0; i < numElements; i++) {
            OP_CHECK(data[i] > 0,
                     OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                             "expertTokenNumPerRank element[%ld]=%ld, must be > 0.",
                             i, data[i]),
                     return ACLNN_ERR_PARAM_INVALID);
            sum += data[i];
        }
    }

    OP_CHECK(sum == A,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Sum(expertTokenNumPerRank)=%ld, must equal tokens dim 0 (A)=%ld.",
                     sum, A),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
```

**重要边界（同轨迹后续证伪）**：上段中"aclrtMemcpy 失败回退直接 memcpy"分支后来被代码检视移除——该回退会掩盖调用环境缺陷且对真实 device 指针有段错误风险；aclrtMemcpy 失败应直接 `OP_LOGE + return ACLNN_ERR_PARAM_INVALID`，配套要求依赖该路径的 UT 在 main 里初始化 NPU 上下文。回退移除与 UT 上下文配套的完整闭环见《op_api UT 默认 device placement 且无上下文致误拒》卡（同批 staged）。

3) `CheckParams` 末尾接入（before → after）：

```cpp
// before
    // Check not empty
    OP_CHECK(!tokens->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tokens is empty tensor."), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(!expertTokenNumPerRank->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expertTokenNumPerRank is empty tensor."), return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// after
    // Check not empty
    OP_CHECK(!tokens->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tokens is empty tensor."), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(!expertTokenNumPerRank->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expertTokenNumPerRank is empty tensor."), return ACLNN_ERR_PARAM_INVALID);

    // Check data: Sum(expertTokenNumPerRank) == A and all elements > 0
    auto dataRet = CheckDataValid(tokens, expertTokenNumPerRank);
    CHECK_RET(dataRet == ACLNN_SUCCESS, dataRet);

    return ACLNN_SUCCESS;
}
```

配套 UT：把 3 个 `DISABLED_` 用例启用（H 超限 / Sum≠A / 元素≤0 各一，断言 `EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID)`）；同时给正常路径用例提供满足约束的 num 数据（16 个元素全 64，sum=1024=A）。

## 设计决策

- 选 GetWorkspaceSize 内读数据校验，而非 tiling 阶段或 device 侧校验：异常输入在 host 侧即拒绝（接口文档语义），不占用 NPU 执行、不会带崩设备；读取量只有 N*E 个整型元素。
- 备选（放弃）：ValueDepend 机制让 host 拿到输入值——该算子此前有 dtype conflict 的前车之鉴且改动面大，未采用。
- 校验缺失期间用 `DISABLED_` UT 用例固化预期（断言写好），补齐后启用——避免"文档承诺、实现无声缺失"长期并存。
- 当前方案属 recommended_fastpath。

## 预防

- 新算子落 op_api 时，把接口文档的约束清单逐条映射到 `CheckParams` 的检查项，"需要读数据"不是跳过校验的理由。
- 正常路径 UT 用例必须提供满足数据约束的输入——用零初始化数据（sum=0≠A）的"合法参数"用例，一旦补上数据校验就会误挂。

## 验证信号

来源说明：该算子 op_api 校验补齐的联调会话（Ascend950PR，CANN 9.1.0），修复后重新编译并安装算子包复测。

- 修复前：L2_016/L2_018 `[FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）`（来源平台：950；证据等级：tool_observed）。
- 修复后 op_api UT 30/30 通过，含启用的 `HExceedsLimitFailure`、`SumNotEqualsAFailure`、`ElementNotPositiveFailure` 三个用例（来源平台：950；证据等级：tool_observed）。
- 修复后 ST 全量（异常用例先行的编排，见同批顺序敏感卡）：77/77，8 个异常用例全部 `[PASS] GetWorkspaceSize 返回错误码 161002（预期 161002），算子已拒绝无效输入`，L2_016/L2_018 在列（来源平台：950；证据等级：tool_observed）。
- 验收复跑全量 77/77 通过、无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 数据内容校验每次调用都同步拷贝 N*E 个元素到 host：本算子规模（≤16384 个 int）开销可忽略；若换成 MB 级输入张量需重新评估（未验证）。
- 上板真实 device 数据路径已由 ST 77/77 验证；UT 环境（host 指针构造 tensor + 无 NPU 上下文）下 aclrtMemcpy 失败行为的边界见同批 UT placement 卡，回退分支已被证伪移除。
- 其他平台未验证（推断可迁移）。

**未决项**：

- 无（根因与修复均已闭环；aclrtMemcpy 回退分支的边界见同批 UT placement 卡）。
## 坏实践

- 接口文档写了约束、实现只查元数据，还把对应 UT 用例长期 `DISABLED_`——校验缺失被测试侧"让路"而非被修复。
- 在产线 op_api 校验里保留"aclrtMemcpy 失败回退 memcpy"的测试后门分支——掩盖环境缺陷且对真实 device 指针有段错误风险（同轨迹检视已移除）。

## 误诊

- 无（本卡根因明确；关联的"UT 环境误拒"是修复过程中独立发现的第二根因，另卡记录）。
