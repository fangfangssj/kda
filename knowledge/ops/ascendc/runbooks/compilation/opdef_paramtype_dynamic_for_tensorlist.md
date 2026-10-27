---
type: "Runbook"
title: TensorList OpDef 须 ParamType(DYNAMIC)
description: OpDef 用单张量 ParamType(REQUIRED) 声明 TensorList 输入时 autogen aclnn 签名变成 const aclTensor*，ST 传 aclTensorList* 编译失败；改 DYNAMIC 后自动生成 aclTensorList*。
tags: [task.implementation, topic.compilation, topic.implementation.operator_registration, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# TensorList OpDef 须 ParamType(DYNAMIC)

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个 foreach 类算子（三个 TensorList 输入 x1/x2/x3、scalar 属性、TensorList 输出 y）做首轮 ST 联调验证时，ST 测试程序（C++ 直连 aclnn，包含安装目录下的 autogen 头文件 `aclnn_<op>.h`）编译报错：

```text
error: 'ACL_BFLOAT16' was not declared in this scope; did you mean 'ACL_FLOAT16'?
error: cannot convert 'aclTensorList*' to 'const aclTensor*'
  550 |         x1List, x2List, x3List, scalar, yList, &workspaceSize, &executor);
      |         ^~~~~~~
note: initializing argument 1 of 'aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensor*, const aclTensor*, const aclTensor*, double,
    const aclTensor*, uint64_t*, aclOpExecutor**)'
```

即安装目录 `opp/vendors/<op>_custom/op_api/include/` 下自动生成的 aclnn 接口签名是**单张量** `const aclTensor*`，而 ST 按设计意图传 `aclTensorList*`。当时手写 op_api L0 层（源码树 `op_api/<op>.h/.cpp`）已经是 `const aclTensorList*` 形参（此前一轮编译修复改过），但算子原型声明（`op_host/<op>_def.cpp`）里 x1/x2/x3 用 `this->Input("x1").ParamType(REQUIRED)`、y 用 `ParamType(REQUIRED)` 声明——autogen 只看 OpDef 声明，不看手写 op_api，于是生成单张量签名，两层互相矛盾。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），OpDef/autogen aclnn 两段式工程；ST 为 C++ 直连 aclnn 测试。
- API 路径：`op_host/<op>_def.cpp` 的 `this->Input(...).ParamType(...)` 声明链 → `build/autogen/aclnn_<op>.h` 自动生成签名 → `op_api/` 手写 L0/L2 层。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有需要 TensorList（动态数量张量）输入/输出的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：单张量算子（用 REQUIRED 即可）；属性值传递路径见下文解决配方第 2 部分（aclScalar* 形参是另一处不一致，本卡一并给出对齐写法）。

## 根因

OpDef 声明的 `ParamType` 决定 autogen aclnn 签名的张量类型：`REQUIRED` 生成 `const aclTensor*`，`DYNAMIC` 生成 `const aclTensorList*`。设计为 TensorList 的算子若漏写 `ParamType(DYNAMIC)`，autogen 与手写 op_api 层签名分裂，ST 一包含安装头文件即暴露。参考工程（同机已验证的 foreach_norm 算子）的 def 与其安装头对照确认了这一映射：`ParamType(DYNAMIC)` 的输入在安装头里生成 `const aclTensorList *x`。

## 解决配方

修复前（`op_host/<op>_def.cpp`，autogen 生成单张量签名）：

```cpp
        // TensorList 输入 x1：被加数
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输入 x2：分子
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输入 x3：分母
        this->Input("x3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输出 y
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
```

修复后（x1/x2/x3 输入与 y 输出全部改 `ParamType(DYNAMIC)`）：

```cpp
        // TensorList 输入 x1：被加数 (DYNAMIC_INPUT)
        this->Input("x1")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输入 x2：分子 (DYNAMIC_INPUT)
        this->Input("x2")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输入 x3：分母 (DYNAMIC_INPUT)
        this->Input("x3")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // TensorList 输出 y (DYNAMIC_OUTPUT)
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
```

同一闭环里还需把手写层的标量形参与"属性按值传递"语义对齐（autogen 已生成 `double scalar`，手写层却收 `const aclScalar*`）：

L0 层（`op_api/<op>.h` 声明与 `op_api/<op>.cpp` 实现）：

```cpp
const aclTensorList* <Op>(
    const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
    float scalar, aclTensorList* y, aclOpExecutor* executor)
{
    L0_DFX(<Op>, x1, x2, x3, scalar, y);

    OP_CHECK(IsAiCoreSupport(x1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(<Op>,
        OP_INPUT(x1, x2, x3), OP_ATTR(scalar), OP_OUTPUT(y));
```

（修复前 L0 形参是 `const aclScalar* scalar`，内部先 `OP_CHECK(scalar != nullptr, ...)` 再 `float scalarValue = scalar->ToFloat();`、`OP_ATTR(scalarValue)`。）

L2 层（`op_api/aclnn_<op>.h/.cpp`）对齐为 `double scalar` 按值：

```cpp
ACLNN_API aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensorList *x3,
    double scalar,
    aclTensorList *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
extern "C" aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
    double scalar, aclTensorList* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(<Op>, DFX_IN(x1, x2, x3), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, x3, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调用 L0（框架自动遍历 TensorList，AutoContiguous 处理非连续 Tensor）
    auto opResult = l0op::<Op>(
        x1, x2, x3, static_cast<float>(scalar), y, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

修复后重编译（`bash build.sh --soc=ascend950`）并重装 run 包，autogen 头变为：

```cpp
aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensorList *x3,
    double scalar,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

ST 端同步删掉 `aclCreateScalar/aclDestroyScalar`，直接传 double。

## 设计决策

- 为什么修 def.cpp 而不是让 ST 适配单张量签名：算子语义（foreach 类）决定必须是 TensorList；单张量签名无法表达 listLen>1。
- 属性 vs 输入张量：scalar 是 `RequiredAttr`（tiling 侧 `attrs->GetAttrPointer<float>(0)` 读取），autogen 生成值形参 `double scalar`；手写 L0/L2 层也应收值而不是 `aclScalar*`，否则 ST 还要额外构造/销毁 aclScalar。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0（x86_64-linux 布局，aarch64-linux/include/lib64 为符号链接）。
- 相关文件：`op_host/<op>_def.cpp`（声明）、`build/autogen/aclnn_<op>.h`（自动生成，重编译后更新）、`op_api/<op>.h/.cpp`（L0）、`op_api/aclnn_<op>.h/.cpp`（L2）、`tests/st/test_aclnn_<op>.cpp`（ST 调用方）。
- 验证命令：`cd operators/<op> && bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8`，然后 `bash build/custom_opp_ubuntu_x86_64.run --quiet` 重装，再编译运行 ST。

## 预防

- 写 TensorList 算子 OpDef 时，Input/Output 一律 `ParamType(DYNAMIC)`；提交前 diff 一眼检查 autogen 头（`build/autogen/aclnn_<op>.h`）的形参类型是否与设计一致。
- 属性参数在 L0/L2 层用值形参（float/double），不要用 `aclScalar*`。

## 验证信号

以下信号来自该算子首轮 ST 联调修复会话：ST 编译失败后修改 def.cpp 与 op_api 手写层，重编译重装并检查 autogen 头。

- 修复前：ST 编译报 `cannot convert 'aclTensorList*' to 'const aclTensor*'`，安装头签名为 `const aclTensor*`×4 + `double scalar`（来源平台：950；证据等级：tool_observed）。
- 修复后重编译：`build.sh --soc=ascend950` 成功产出 run 包（来源平台：950；证据等级：tool_observed）。
- 修复后 autogen 头：`const aclTensorList *x1/x2/x3/out` + `double scalar`（`build/autogen/aclnn_<op>.h` 实测）（来源平台：950；证据等级：tool_observed）。
- ST 编译推进到链接与运行阶段（后续上板失败为 kernel 侧 TensorList 寻址问题，属另一独立根因：DYNAMIC 输入的 GM_ADDR 是描述符地址非数据地址）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.1.0 实测；ParamType(DYNAMIC)→aclTensorList* 的映射在其他 CANN 版本未验证（推断可迁移，未验证）。
- DYNAMIC 声明只解决签名层；kernel 侧收到的是 TensorList 描述符地址而非数据地址（须解引用取各张量地址），tiling 侧须用 GetDynamicInputShape——签名正确不等于运行正确。

**未决项**：

- `ParamType(DYNAMIC)`→`aclTensorList*` 映射在 CANN 头文件中的正式文档出处未查证（本卡结论来自同机参考算子 foreach_norm 的 def 与安装头对照 + 修复实测）。
## 坏实践

- 只改手写 op_api 层的类型而不改 def.cpp——autogen 头仍生成旧签名，ST 包含安装头即失败。
- ST 端为了编译过而把 aclTensorList 强转/拆成 aclTensor 适配错误签名——掩盖声明层错误。

## 误诊

- 首轮排查曾以为是 ST 自身把 aclScalar 传错（先修了 `ACL_BFLOAT16`→`ACL_BF16` 的 dtype 枚举名），对照 autogen 头后才定位到 OpDef 声明层。
