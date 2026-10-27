---
type: "Runbook"
title: aclnn String 属性 char* 传 const 编译失败
description: "CANN 自动生成的 aclnn 头文件把可选 String 属性形参声明为非 const 的 char*，adapter 用返回 const char* 的辅助函数传参直接编译失败；须把字符串存入 std::string 成员并以 data() 传可变指针。复现增量：同族的 aclCreateTensor shape/strides/storageShape 参数同为非 const int64_t*，const 容器 data() 直传同样编译失败，包装函数边界做局部可变副本。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-14T12:12:55Z'
updated_at: '2026-09-17T03:16:34Z'
platforms: ["950"]
---

# aclnn String 属性 char* 传 const 编译失败

## 现象

在 Ascend950PR（CANN 9.2.0）上为一个 scatter/scatter-reduce 自定义算子（原型 `scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y`）搭建 Torch 测试路由：run.sh 用 `torch.utils.cpp_extension.load_inline` 现场编译一个 TORCH_LIBRARY adapter，把 `torch.ops.<ns>.forward` 绑到本地安装 vendor 包的 aclnn C API 上。该算子有一个可选 String 属性 reduce，读生成头文件确认签名为：

```c
aclnnStatus aclnnXxxGetWorkspaceSize(
    const aclTensor *data, const aclTensor *indices, const aclTensor *updates,
    int64_t dim,
    char *reduceOptional,          /* 注意：非 const 的 char* */
    const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

首次执行 run.sh 时 adapter 编译失败，报错原文：

```text
.../build/torch_adapter/main.cpp:127:26: error: invalid conversion from 'const char*' to 'char*' [-fpermissive]
  127 |         dim, invocation->reduceCstr, invocation->outputAcl,
      |              ~~~~~~~~~~~~^~~~~~~~~~~
In file included from ...:
.../install/vendors/<VENDOR>/op_api/include/aclnn_<op>.h:32:11: note: initializing argument 5 of
'aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, int64_t, char*, const aclTensor*, uint64_t*, aclOpExecutor**)'
   32 |     char *reduceOptional,
```

当时 adapter 的属性转换辅助函数声明返回 `const char *`（`ReduceToCstr`），成员 `reduceCstr` 也是 `const char *`。

## 适用场景

- 工程形态：aclnn 自定义算子的 Torch 路由/测试脚手架——用 load_inline 或任何 C++ 工程直接调用 `aclnnXxxGetWorkspaceSize`/`aclnnXxx` C API，且算子带 String 类属性。
- API 路径：CANN 算子包构建时 autogen 产出的 `aclnn_<op>.h`（部署于 `<vendor_root>/op_api/include/`）；String 属性形参（可选属性名带 Optional 后缀）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何在该平台 CANN 版本下写 aclnn adapter、算子含 String 属性的场景，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：纯 aclnn C 调用（C 里没有 const 传 char* 的问题，C 允许）；其他 CANN 版本的 autogen 头文件约定未验证；Bool/Int 等非 String 属性不涉及。

## 根因

C++ 禁止把 `const char*` 隐式转换为 `char*`（连 `-fpermissive` 都不放行）。而 CANN 自动生成的 aclnn 头文件对 String 类属性形参统一使用**非 const** 的 `char*`（与张量形参的 `const aclTensor*` 形成对比）。辅助函数 `ReduceToCstr` 的返回类型写成 `const char *` 后，`storage.data()` 在该返回语境下按 const 处理，传出值再传入 `char*` 形参即触发硬编译错误。

## 解决配方

修复前（run.sh 内嵌 adapter 的属性转换辅助函数与成员声明，函数级完整）：

```cpp
// reduce attr mapping:
//   reduce is None  -> nullptr (optional attr, framework default "update" = overwrite)
//   reduce == "update" -> nullptr (same overwrite semantics)
//   reduce == add/multiply/amin/amax -> the C string
// The kernel tiling treats nullptr and "update" identically (mode 0).
const char *ReduceToCstr(const c10::optional<std::string> &reduce, std::string &storage)
{
    if (!reduce.has_value()) {
        return nullptr;
    }
    const std::string &r = *reduce;
    if (r.empty() || r == "update") {
        return nullptr;
    }
    storage = r;
    return storage.data();
}
```

```cpp
    c10::optional<std::string> reduce;
    std::string reduceStorage;
    const char *reduceCstr;
```

修复后（返回类型与成员改为 `char*`，字符串本体存放在 `Invocation` 的 `std::string` 成员里，C++17 起 `data()` 对非 const string 返回可变指针）：

```cpp
// reduce attr mapping:
//   reduce is None  -> nullptr (optional attr, framework default "update" = overwrite)
//   reduce == "update" -> nullptr (same overwrite semantics)
//   reduce == add/multiply/amin/amax -> the C string (mutable, aclnn wants char*)
// The kernel tiling treats nullptr and "update" identically (mode 0).
char *ReduceToCstr(const c10::optional<std::string> &reduce, std::string &storage)
{
    if (!reduce.has_value()) {
        return nullptr;
    }
    const std::string &r = *reduce;
    if (r.empty() || r == "update") {
        return nullptr;
    }
    storage = r;
    return storage.data();  // C++17: non-const .data() on non-const std::string
}
```

```cpp
    c10::optional<std::string> reduce;
    std::string reduceStorage;
    char *reduceCstr;
```

要点：`storage` 是 `Invocation`（shared_ptr 持有）的非 const 成员，生命周期覆盖 `GetWorkspaceSize` 与后续 launch，无悬垂；`reduce=None` 或 `"update"` 传 `nullptr`（本例算子 tiling 中 `MapReduceMode(nullptr)` 与 `"update"` 同为 mode 0，语义等价——换算子须先 grep 各自 tiling 源码确认 nullptr 处理后再折叠）。

## 设计决策

- 选"std::string 成员 + data()"而不是字符串字面量直传：字面量到 `char*` 的转换在 C++11 起已弃用（`-Wwrite-strings` 告警），且 `c10::optional<std::string>` 归一化后必须有一个可写缓冲。
- None/"update" 折叠为 nullptr：先 grep 算子 tiling 源码确认 `if (reduce == nullptr) return 0;` 与 `"update"` 分支等价（都映射覆盖模式），才敢把两种取值合并成 nullptr 传参。
- 当前方案路线档位：recommended_fastpath（两行类型修正，零运行时代价）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；torch 2.x + torch_npu（miniconda python 3.13）；adapter 经 `torch.utils.cpp_extension.load_inline`（ninja）编译。
- 相关文件：`<op>_package/tests/torch/run.sh`（内嵌 adapter C++ 源，本修复位置）；`<install>/vendors/<VENDOR>/op_api/include/aclnn_<op>.h`（生成头文件，char* 声明证据）。

## 预防

- 写 aclnn adapter 前先打开生成头文件逐参数核对 const 性：张量 `const aclTensor*`、String 属性 `char*`、整型属性值传递。
- 凡 String 属性，统一模式："std::string 成员存字符串 + `data()` 传参"，可选值缺省传 nullptr 前先确认算子 tiling 对 nullptr 的处理。
- 同一生成头的相邻契约：可选参数的 `Optional` 后缀只加在指针型形参（含 `char*`）上，见 [aclnn生成头Optional后缀只加指针型参数](../validations/aclnn_header_optional_suffix_pointer_params_only.md)；多输出算子 adapter 须按生成头逐参对齐（输入+属性+全部输出+ws+exec），见 [多输出算子 aclnn 适配器签名须逐参对齐生成头](aclnn_multi_output_adapter_signature_autogen_header.md)。

## 验证信号

以下信号来自该算子 Torch 路由搭建会话：属性转换函数改为 char* 后重跑 run.sh 的实测结果。

- 修复前：load_inline ninja 编译失败，`error: invalid conversion from 'const char*' to 'char*' [-fpermissive]`，指向 aclnn 头文件 `char *reduceOptional` 形参（来源平台：950；证据等级：tool_observed）。
- 修复后：run.sh 完整跑通——adapter 编译成功、`direct_call_output: [10.0, 2.0, 30.0, 4.0]`（data=[1,2,3,4]、indices=[0,2]、updates=[10,30]、reduce=None 覆盖后 y[0]=10、y[2]=30，语义正确）、并打印 torch_route_ready 的 adapter/vendor/golden 路径（来源平台：950；证据等级：tool_observed）。
- 修复后：golden 对照一致 `npu: [10.0, 2.0, 30.0, 4.0] golden: [10.0, 2.0, 30.0, 4.0]`；TTK e2e eager 冒烟单用例 PASS（1/1）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.2.0 autogen 头文件实测；若未来版本把 String 属性改为 `const char*`，本修复无副作用（char* 实参同样匹配 const char* 形参）。
- 若属性字符串需要穿透到异步 launch lambda 之后使用，必须像本例一样把 `std::string` 存进随 lambda 捕获的对象（Invocation），局部 std::string 的 data() 会悬垂。
- 机制是 C++ const 规则层面的，推断可迁移至任何手写 aclnn 调用方的 AscendC 工程（复现增量已在 aclCreateTensor 上验证第二个实例，见文末同族增量；其余 API 未逐一验证）。

**未决项**：

- 无（根因已追到 C++ const 规则与 aclnn 头文件生成约定层面并闭环验证）。
## 坏实践

- 辅助函数返回 `const char*` 再传 `char*` 形参——必然编译失败。
- 把属性字符串做成局部 `std::string` 传 `data()` 后在异步 launch 里继续用——生命周期悬垂（本例通过 Invocation 成员规避）。

## 误诊

- 无（编译器报错与头文件 note 直接指向形参类型不匹配，一步定位）。

## 同族增量（aclCreateTensor 的 shape/strides/storageShape 同为非 const）

另一独立轨迹并入第二个已验证 API：PyTorch 适配层（torch_adapter.cpp，把 `torch::Tensor` 的裸指针包装成 `aclTensor` 后走 aclnn 两段式调用）的辅助函数 `CreateAclTensor(const void* data_ptr, const std::vector<int64_t>& shape, aclDataType dtype)` 内部直接 `shape.data()` 传给 `aclCreateTensor`，Ascend950PR + CANN 9.1.0（gcc 13.3.0）CMake 构建该文件时报错原文：

```text
torch_adapter.cpp:82:40: error: invalid conversion from 'const long int*' to 'int64_t*' {aka 'long int*'} [-fpermissive]
   82 |     int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
```

`aclCreateTensor` 的 shape/strides/storageShape 参数类型均是非 const 的 `int64_t*`（API 只读语义未体现在签名上，与 aclnn String 属性的 `char*` 同一约定；base ACL C API，非仅 autogen aclnn 头）。从 `const std::vector<int64_t>&` 取 `data()` 得到 `const int64_t*`，直接传参构成 const 丢失的非法转换。

修复（包装函数边界做一次局部可变副本，完整函数）：

```cpp
aclTensor* CreateAclTensor(const void* data_ptr,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    // aclCreateTensor 参数为非 const int64_t*，使用局部可变副本
    std::vector<int64_t> shape_mut(shape);
    std::vector<int64_t> strides_mut(strides);
    aclTensor* tensor = aclCreateTensor(
        shape_mut.data(),
        static_cast<int>(shape.size()),
        dtype,
        strides_mut.data(),
        0,
        ACL_FORMAT_ND,
        shape_mut.data(),
        static_cast<int>(shape.size()),
        const_cast<void*>(data_ptr)
    );
    return tensor;
}
```

设计要点：拷贝发生在 host 侧、shape/strides 元素量级为 rank（个位数），成本可忽略；`const_cast<int64_t*>(shape.data())` 能编过但把未声明的可变性契约埋进代码，不推荐。data 指针保留 `const_cast<void*>(data_ptr)`：`aclCreateTensor` 末参本就是 `void*` 且数据指针来自 PyTorch tensor 管理，是该 API 的常规用法。若调用方本来就持有非 const 容器，直接传 `data()` 即可，无需副本。

### 复现增量验证信号

来源说明：该算子 PyTorch ST 工程开发会话（Ascend950PR + CANN 9.1.0）修复前后的构建与全量 ST。

- 修复前：`invalid conversion from 'const long int*' to 'int64_t*'`（-fpermissive 下 Error 1 终止构建）（来源平台：950；证据等级：tool_observed）
- 修复后：同一目标编译通过 `[100%] Linking CXX shared library libtorch_adapter.so`（来源平台：950；证据等级：tool_observed）
- 后续该适配层全量 ST：159 条用例 157 通过 + 2 已知限制、失败 0（含 CreateAclTensor 包装路径的全 dtype 覆盖）（来源平台：950；证据等级：tool_observed）

### 复现增量失效边界

- aclCreateTensor 签名按 CANN 9.1.0 实测；与主文 9.2.0 的 aclnn String 形参同约定，跨版本以实际头文件为准（若后续版本改为 `const int64_t*`，可直接传 `data()`，本修复无副作用）。
- 局部副本要求调用期间 tensor 元数据在 `aclCreateTensor` 返回后不再被引用（本用法满足：ACL 侧拷贝元数据）。