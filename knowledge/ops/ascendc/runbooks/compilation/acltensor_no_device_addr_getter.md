---
type: "Runbook"
title: ACL 无 aclGetTensorAddr，D2H 直传 Malloc 指针
description: CANN 9.1.0 头文件无 aclGetTensorAddr（编译报 was not declared in this scope）；aclTensor 的设备地址无法反取，须在 aclrtMalloc 处保存原始指针并随 tensor 一起传给 D2H 拷贝。
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# ACL 无 aclGetTensorAddr，D2H 直传 Malloc 指针

## 现象

为逐元素最大值自定义算子开发 C++ ST 测试的 Real 模式（链接已安装算子包的 `libcust_opapi.so`，NPU Ascend950PR）时，重构后的辅助函数 `RunAclnnMaximum` 编译失败。该函数此前被改为"只拿 aclTensor*、在 D2H 拷贝处从输出 tensor 反取设备地址"，编译器对每个模板实例化都报同一错误（输出原文）：

```
/workspace/.../tests/st/test_aclnn_bench_no_harness_maximum.cpp:1376:65: error: 'aclGetTensorAddr' was not declared in this scope; did you mean 'aclSetTensorAddr'?
 1376 |     ret = aclrtMemcpy(outData.data(), outBytes, aclGetTensorAddr(outTensor), outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
      |                                                 ~~~~~~~~~~~~~~~~^~~~~~~~~~~
      |                                                 aclSetTensorAddr
```

随后在整个 CANN 9.1.0 安装目录 `include/`（含 acl_base.h、acl_rt.h、acl_op_kernel.h）grep `aclGetTensorAddr|aclGetTensorStorage|GetTensorAddr`，无任何命中——ACL 公共 API 面没有"从 aclTensor 取 device 地址"的接口。

## 适用场景

- 工程形态：算子仓开发的 C++ ST/示例，走 ACLNN 两段式（GetWorkspaceSize + 执行）并需要把 NPU 输出拷回 host 比对。
- API 路径：`aclrtMalloc`/`aclrtMemcpy`/`aclCreateTensor`/`aclSetTensorAddr`（ACL C API）；不存在的 `aclGetTensorAddr`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 上所有需要 aclTensor 设备地址的 host 侧代码。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：未来 CANN 版本若新增取址 API 不在本卡范围；`aclSetTensorAddr`（设置方向）存在但语义不同，不能反取。

## 根因/解决配方

**根因（单句）**：ACL 公共头文件没有"从 aclTensor 反取 device 地址"的 API（只有设置方向的 aclSetTensorAddr），重构时臆造了 `aclGetTensorAddr` 这个接口名。

修复：`RunAclnnMaximum` 增加 `void* outDev` 参数，D2H 拷贝直接使用创建输出 tensor 时 `aclrtMalloc` 返回的原始指针（该指针本就是 `aclCreateTensor` 的存储地址），两个调用点同步传参。

修复前（失败行，取自编译错误原文，因果完整）：

```cpp
    // D2H 拷贝试图从 aclTensor 反取设备地址 —— ACL 无此 API
    ret = aclrtMemcpy(outData.data(), outBytes, aclGetTensorAddr(outTensor), outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
```

修复后（关键因果片段 verbatim；函数其余部分为 workspace 分配/两段式执行/流同步，与修复无涉）：

```cpp
// 通用执行辅助：创建 tensor、调用算子、释放资源
// 返回 true 表示算子执行成功且结果已拷贝到 outData
template<typename T>
bool RunAclnnMaximum(aclTensor* x1Tensor, aclTensor* x2Tensor, aclTensor* outTensor,
                     const std::vector<int64_t>& outShape, void* outDev,
                     std::vector<T>& outData, aclrtStream stream) {
    ...
    size_t outBytes = static_cast<size_t>(outSize) * sizeof(T);
    ret = aclrtMemcpy(outData.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ...
}
```

调用点（两处，verbatim）：

```cpp
        bool ok = RunAclnnMaximum<T>(x1Tensor, x2Tensor, outTensor, outShape, outDev, outData, stream);

        aclDestroyTensor(x1Tensor); aclDestroyTensor(x2Tensor); aclDestroyTensor(outTensor);
        aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
```

```cpp
    bool ok = RunAclnnMaximum<T>(x1Tensor, x2Tensor, outTensor, outShape, outDev, outData, stream);

    aclDestroyTensor(x1Tensor); aclDestroyTensor(x2Tensor); aclDestroyTensor(outTensor);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
```

> 说明：重构函数体的完整写入事件在轨迹中被截断，此处内联的是签名、D2H 拷贝行与全部调用点等因果完整片段；其余未变部分（GetWorkspaceSize/执行/同步）不属于本修复单元。

## 设计决策

- 选择"直传 `aclrtMalloc` 原始指针"而不是恢复旧的"调用方自行拷贝"结构：`outDev` 即 `aclCreateTensor` 的存储地址，直传零成本且保持 `RunAclnnMaximum` 的封装。
- 备选"遍历 strides 重算地址"被放弃：aclTensor 不暴露存储指针，任何重算都拿不到基址。
- 当前方案属 recommended_fastpath。

## 预防

- 写 ACL host 侧代码时，把"aclrtMalloc 返回的指针"与"aclTensor 句柄"作为一对一起传递，不要期望事后从句柄反查地址。
- CR 清单：出现 `aclGet*Addr`/`aclGetTensorStorage` 类"直觉命名"时，先在 `$ASCEND_HOME_PATH/include` grep 确认 API 存在。

## 验证信号

问题与验证来源：该算子 C++ ST 测试工程的 Real 模式编译与运行（同一测试源文件，修复后重编译）。

- 修复前：Real 模式编译失败，每个模板实例化报 `error: 'aclGetTensorAddr' was not declared in this scope; did you mean 'aclSetTensorAddr'?`（来源平台：950；证据等级：tool_observed）。
- API 面核验：`grep -rn "aclGetTensorAddr\|aclGetTensorStorage\|GetTensorAddr" $ASCEND_HOME_PATH/include/` 无命中（来源平台：950；证据等级：tool_observed）。
- 修复后：`g++ -std=c++17 -fsyntax-only -Wall`（含 CANN include 与算子 op_api 头）退出码 0、无警告输出；随后 Real 模式 CMake 全量编译通过并执行（首轮 70/72，失败项为空张量分配的独立问题；修复空张量后 72/72）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定 CANN 9.1.0 的公共头文件 API 面；更高版本若提供取址 API，本卡"直传指针"仍是兼容做法。
- 未验证 torch_npu/GE 等框架内部取址路径（它们走内部接口，不在公共 API 面）。

**未决项**：

- 无。
## 坏实践

- 凭命名直觉调用未 grep 确认的 ACL API（`aclSetTensorAddr` 的存在容易诱导臆造 `aclGetTensorAddr`）。
- 为绕开该限制去解析 aclTensor 内部结构体布局取指针（未定义行为，未尝试，列为禁用方向）。

## 误诊

- 无（编译器报错与 grep 核验直接锁定 API 缺失）。
