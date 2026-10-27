---
type: "Runbook"
title: "aclIntArray 无 At()，元素访问用下标"
description: "op_api 层读 aclIntArray 元素报 has no member named 'At'：opdev 头里 aclIntArray 经 ACL_ARRAY 宏继承 aclArray<int64_t>，只提供 (*arr)[i]/Size()/GetData()；公开 C 头（acl_meta.h）则是不透明类型，仅有 aclCreateIntArray/aclGetIntArraySize。grep 'aclIntArray' 查不到类定义是宏陷阱。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T05:27:53Z"
platforms: ["950"]
---

# aclIntArray 无 At()，元素访问用下标

## 现象

两个独立算子工程（均为 ascend950，CANN 9.1.0）的 op_api 层读 `aclIntArray*` 属性（group_list / filterSize 等 ListInt 属性）时按 STL 直觉写了 `At()`，编译报同款错误：

```text
op_api/aclnn_<op>.cpp:150:26: error: 'const aclIntArray' {aka 'const class aclIntArray'} has no member named 'At'
op_api/aclnn_<op>.cpp:152:34: error: ... has no member named 'At'
op_api/aclnn_<op>.cpp:157:30: error: ... has no member named 'At'
（另一工程同款，L0 层 outShape.SetDim(i, filterSize->At(i)) 等多处）
```

排查时的关键陷阱：在 CANN 安装目录里 grep `"aclIntArray"` 翻 `opdev/common_types.h` 只命中 aclTensor 构造重载与 ToString，**看不到类定义**——因为 `aclIntArray` 是由 `ACL_ARRAY(Int, int64_t)` 宏展开生成的（宏体里只有 `acl##Type##Array` 拼接，不含字面 `aclIntArray`）。一度据此误判"op_api 层没有元素访问 API，只能规避"；实际上 `common_types.h` 第 102 行起的 `aclArray<T>` 模板就是基类，元素访问一直可用（下标），见根因。

分层事实（两条轨迹 + 头文件直读共同确认）：

- **公开 C 头** `aclnn/acl_meta.h`：`typedef struct aclIntArray aclIntArray;`（不透明），公开 API 只有 `aclCreateIntArray`/`aclDestroyIntArray`/`aclGetIntArraySize`——没有取元素入口。
- **opdev 头** `aclnn/opdev/common_types.h`：`ACL_ARRAY(Int, int64_t)` 宏生成 `class aclIntArray : public aclArray<int64_t>`，公开成员 `operator[]`/`Size()`/`GetData()`/`ToString()`——op_api 层代码（会 include opdev 头链，报错信息 `aka 'const class aclIntArray'` 即证明类定义可见）可直接用。

## 适用场景

- 工程形态：registry-invoke 自定义算子 op_api 层（L0 `l0op::<Op>` / L2 `aclnn<Op>`）读取 aclnn 调用方传入的 `aclIntArray*` 属性值（如 group_list / filter_size / axes 等列表属性的元素级校验或 shape 构造）。
- API 路径：`include/aclnn/opdev/common_types.h` 的 `aclArray<T>` / `aclIntArray` / `aclBoolArray` / `aclFloatArray` 家族；`include/aclnn/acl_meta.h` 的 C API（`aclCreateIntArray`/`aclGetIntArraySize`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（两个独立工程交叉编译实测；与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 9.1.0 aclnn 公共类型契约）。
- 路线优先级：recommended_fastpath（op_api 层直接用 `(*arr)[i]` + `Size()`）；safety_fallback：只允许依赖公开 C API 面（不 include opdev 头）的代码用"tensor shape 承载语义 + `aclGetIntArraySize` 长度校验"规避（见解决配方末尾，须声明契约代价）。
- 明确不适用 / 未验证场景：host tiling 侧读属性用 `gert::RuntimeAttrs`（GetListInt 等）与 `GetData()` 原始指针，是另一套 API；aclFloatArray/aclBoolArray 推断同构（ACL_ARRAY 同宏生成），未逐一验证；其他 CANN 版本的成员清单未验证。

## 根因

`aclIntArray` 的基类 `aclArray<T>`（`opdev/common_types.h`）只重载了 `operator[]`（以及 `Size()`/`GetData()`/`ToString()`），从未提供 `At()` 成员；把 STL 风格（`std::vector::at` / 某些框架的 `At`）记忆套到 aclnn 数组类型上即产生同名编译错。类定义本体（102-124 行，verbatim）：

```cpp
template <typename T>
class aclArray : public op::Object {
public:
    ~aclArray() override;

    T& operator[](uint64_t i);

    const T& operator[](uint64_t i) const;

    uint64_t Size() const;

    const T* GetData() const;

    ge::AscendString ToString() const;

protected:
    aclArray(const T* value, uint64_t size);
    ...
};
```

`aclIntArray` 即 `ACL_ARRAY(Int, int64_t)` 的宏展开——这解释了为什么 grep `"aclIntArray"` 在 common_types.h 里找不到类定义（排查陷阱），而报错信息里 `aka 'const class aclIntArray'` 又证明类可见。

## 判别方法

触发信号即 `error: 'const aclIntArray' {aka 'const class aclIntArray'} has no member named 'At'`。确认可用面的探针（本轨迹实际执行）：

```bash
# 1) 基类与成员（注意搜 aclArray 而不是 aclIntArray——后者是宏生成名）
grep -n "class aclArray\|ACL_ARRAY" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/common_types.h
# 2) 公开 C API 面（只有创建/销毁/取长度）
grep -n "IntArray" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/acl_meta.h
# 27:typedef struct aclIntArray aclIntArray;
# 68:ACL_FUNC_VISIBILITY aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
# 91:ACL_FUNC_VISIBILITY aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size);
```

## 解决配方

**recommended_fastpath（op_api 层，opdev 头可见时）**：元素访问改下标，长度维持 `Size()`（sed 批量替换）：

```bash
cd operators/<op>/op_api && \
  sed -i 's/group_list->At(\([^)]*\))/(*group_list)[\1]/g' aclnn_<op>.cpp
```

修复前（同一段校验代码，verbatim）：

```cpp
    // group_list 末值 == M
    OP_CHECK(group_list->At(E - 1) != M,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "group_list last value %ld != M %ld",
                     group_list->At(E - 1), M),
             return false);

    // group_list 单调非递减
    for (int64_t i = 1; i < E; i++) {
        OP_CHECK(group_list->At(i) < group_list->At(i - 1),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "group_list must be non-decreasing, but [%ld]=%ld < [%ld]=%ld",
                         i, group_list->At(i), i - 1, group_list->At(i - 1)),
                 return false);
    }
```

修复后：

```cpp
    // group_list 末值 == M
    OP_CHECK((*group_list)[E - 1] != M,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "group_list last value %ld != M %ld",
                     (*group_list)[E - 1], M),
             return false);

    // group_list 单调非递减
    for (int64_t i = 1; i < E; i++) {
        OP_CHECK((*group_list)[i] < (*group_list)[i - 1],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "group_list must be non-decreasing, but [%ld]=%ld < [%ld]=%ld",
                         i, (*group_list)[i], i - 1, (*group_list)[i - 1]),
                 return false);
    }
```

长度校验维持 `group_list->Size()`（该成员存在，编译原本就通过）。`operator[]` 无边界检查，先 `Size()` 校验长度再按下标访问；其形参是 `uint64_t`，负数下标会被隐式转换成巨大无符号数，务必保证索引非负。

**safety_fallback（只依赖公开 C API 面的代码）**：不 include opdev 头、只用 `acl_meta.h` 时，aclIntArray 是不透明类型，无元素访问入口。另一工程的规避方案：把"从属性取值"改为"从输出 tensor 取 shape + 仅用 `aclGetIntArraySize` 校验属性长度"——L0 侧输出 shape/dtype 从 `y->GetViewShape()`/`y->GetDataType()` 读取后经 `executor->AllocTensor` 构造输出；L2 侧值校验经 y.shape 间接完成：

```cpp
    uint64_t filterSizeSize = 0;
    aclGetIntArraySize(filterSize, &filterSizeSize);
    if (filterSizeSize != 5) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "filterSize must have 5 elements, got %lu.", filterSizeSize);
        return false;
    }
    // 通过 y.shape 校验 filterSize 一致性（y.shape[0]=C_out, y.shape[1]=C_in/groups）
    if (y->GetViewShape().GetDim(0) != cOut) { ... return false; }
    if (y->GetViewShape().GetDim(1) * groups != cIn) { ... return false; }
```

fallback 声明：该路线的代价是属性值不再直接参与校验，正确性转移给"调用方必须正确设置 y.shape"这一契约，须在文档/issue 中显式声明；op_api 层代码本就能用下标路线，优先 recommended_fastpath，fallback 只服务"必须保持公开 API 依赖面"的场景。配套动作：把该限制与规避方案落成 issue 文档，避免后续迭代误删规避代码。

## 设计决策

- recommended_fastpath 选 `(*arr)[i]` + `Size()`：与 opdev 头的类成员直接一致，改动最小（一处 sed），且两条轨迹中一条已实测编译闭环。
- 放弃 `reinterpret_cast` 按假设布局读元素：内存布局非公开契约，编译能过但运行时可能崩，版本升级即坏（另一轨迹明确考虑后放弃；布局误解致垃圾值的实测案例见 [561103 aclIntArray 布局误解卡](561103_aclintarray_layout_misread.md)——该卡的 layout 手解路线只应作兜底，下标路线可用时优先下标）。
- fallback 选 y.shape 承载语义：y 的 view shape 本来就是调用方按 filter_size 设置的，语义等价且只用公开 API。

## 预防

- 写 aclnn 参数校验前先看 `include/aclnn/opdev/common_types.h` 的 `aclArray<T>` 成员表；可用面就三个：`(*arr)[i]`、`arr->Size()`、`arr->GetData()`。
- 在 CANN 头里查类型定义时，宏生成的类名（ACL_ARRAY 家族）grep 字面名会漏——改搜基类名或宏名。
- 不校验 `Size()` 直接按下标遍历。

## 验证信号

来源说明：两个独立算子工程的 op_api 编译闭环（ascend950，CANN 9.1.0，x86_64 交叉编译）。

- 工程一修复前：`error: 'const aclIntArray' ... has no member named 'At'`（aclnn_<op>.cpp 150/152/157/160 行共 9 处）（来源平台：950；证据等级：tool_observed）。
- 工程一修复后：下标访问替换后 op_api 编译单元通过，后续报错全部转移至 op_kernel 侧（其他根因）；全量回归 `build.sh --soc=ascend950 -j8` 生成 custom_opp_ubuntu_x86_64.run 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。
- 工程二（fallback 路线）修复前：L0 层 `outShape.SetDim(i, filterSize->At(i))` 报同款 `has no member named 'At'`（来源平台：950 交叉编译；证据等级：tool_observed）。
- 工程二（fallback 路线）：头文件检索覆盖 acl_meta.h/common_types.h/op_arg_def.h/aicpu_task.h 及 lib64 符号表；L0/L2 按规避方案重写后 cust_opapi 编译通过，`has no member named 'At'` 消失，最终 build.sh 完整编译通过并产出算子包（来源平台：950 交叉编译；证据等级：tool_observed）。
- 类定义与公开 C API 面均为头文件直接读取（common_types.h 的 aclArray 成员表、acl_meta.h 的 27/68/91 行）（来源平台：950 交叉编译环境；证据等级：tool_observed）。

## 失效边界

- 成员清单基于 CANN 9.1.0 的 common_types.h / acl_meta.h；后续版本若补 `At()` 或公开元素访问 API，两种写法都可用（未验证，推断可迁移）。
- fallback 的"y.shape 承载 filter_size"契约依赖调用方正确设置 y 的 view shape；未设置/设置错误的运行时行为未验证（未上板）。
- aclFloatArray/aclBoolArray 推断同构（同一 ACL_ARRAY 宏），未逐一验证。
- 两工程均为交叉编译产出算子包，未上板运行。

**未决项**：

- 查证 CANN 是否有官方 aclIntArray C++ 元素访问 API 的文档化承诺或布局契约——供后续版本复核 fallback 是否可退役。
## 坏实践

- 按 STL `at()` 记忆写 `arr->At(i)`。
- 用 `reinterpret_cast<const int64_t* const*>(arr)[0]` 之类按假设布局读 aclIntArray 元素——布局非公开契约（见布局误解卡的下场：读到 vtable/指针垃圾值触发 561103）。
- 在 CANN 头里 grep 宏生成类名的字面串来判定"API 不存在"——先确认类是不是宏展开生成的。
- 在 op_api 层为绕开该限制而自造与 ACLNN 数据布局耦合的容器类型。

## 误诊

- 工程二初判"opdev 层应该有 aclIntArray 的 C++ 包装类（类似 aclTensor）"，翻遍 common_types.h 只找到构造 aclTensor 的重载与 ToString，据此得出"无元素访问 API、只能规避"的结论——该结论被工程一实测的下标路线与本卡对 ACL_ARRAY 宏的直读修正：类定义存在，只是 grep `"aclIntArray"` 搜不到宏生成的名字。aclScalar 倒是有 ToInt64() 等取值方法，容易让人误以为数组也有。

## 后续增量（2026-09-17，并卡合并：resize_bilinear 实例与 aclFloatArray 同型覆盖）

#### 并入自 aclintarray_no_at_use_operator_bracket（resize_bilinear 实例、float 元素日志告警与 aclFloatArray 覆盖）

双线性插值算子（BenchNoHarnessResizeBilinear，ascend950，CANN 9.1.0）的同类复现与补充：

- cust_opapi 目标一次报 4 处同类错误（`aclIntArray` 与 `aclFloatArray` 各 2 处）：`hOut = outputSize->At(0);`、`wOut = outputSize->At(1);`、`float sh = scaleFactor->At(0);`、`float sw = scaleFactor->At(1);`——**aclFloatArray（`aclArray<float>`）同型触发**，L2 接口 `CheckAttrMutex` 的正负性校验（`outputSize->At(0) > 0` 等）同轮报错。
- 修复同法（`arr->At(i)` → `(*arr)[i]`）；**float 元素进 `%f` 日志时补 `static_cast<double>`** 消除可变参数告警：

```cpp
    OP_CHECK((*scaleFactor)[0] > 0.0f && (*scaleFactor)[1] > 0.0f,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale_factor elements must be positive: [%f, %f].",
                     static_cast<double>((*scaleFactor)[0]), static_cast<double>((*scaleFactor)[1])),
             return false);
```

- 设计决策补充（并入）：选 `(*arr)[i]` 而不是 `arr->GetData()[i]`——两者等价，`operator[]` 更直观且与 `Size()` 配套构成完整访问面；`GetData()` 留给需要批量指针传递的场景。
- 预防速查（并入）：元素 `(*arr)[i]`、条数 `arr->Size()`、指针 `arr->GetData()`、判空 `arr != nullptr && arr->Size() > 0`；LLM 生成或照抄其它容器习惯的代码时，重点排查 `At(`、`.Get(` 这类 ACLNN 数组上不存在的方法名。

验证信号（并入，resize_bilinear，ascend950 + CANN 9.1.0）：

- 修复后同一构建不再出现任何 `no member named 'At'` 错误（构建日志只剩 `OP_CHECK_NULL` 宏错误，属另一独立问题）；该算子最终 `bash build.sh --soc=ascend950 -j8` 全量成功，3 个 dtype 内核二进制生成并打包（来源平台：950；证据等级：tool_observed）。