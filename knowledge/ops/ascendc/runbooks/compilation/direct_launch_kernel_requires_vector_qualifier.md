---
type: "Runbook"
title: "SIMT 核类型推导失败 — 补 __vector__"
description: "SIMT VF 启动 shell 只标 __aicore__ 时 ld.lld 无法推导模板核函数类型；补 __vector__ 后恢复链接。"
tags: ["task.debugging", "topic.programming_model.execution_model.simt", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SIMT 核类型推导失败 — 补 __vector__

## 现象

在 Ascend950 Flash direct-launch 工程中编写一个 `.asc` 单文件算子时，Device 侧用
`__simt_vf__` 定义 SIMT VF，再由 `__global__` shell 调用 `asc_vf_call`。代码能进入
Bisheng 编译流程，但在生成共享库的链接阶段，18 个模板实例化全部先出现：

```text
WARNING: kernel type of __global__ func: void gather_kernel<...>(...) is not marked.
auto type derivate may be failed.
```

随后 `ld.lld` 对各实例化重复报错：

```text
ld.lld: error: Error: the type of kernel function void gather_kernel<...>(...)
auto derivate failed, please manually add the function type attribute
bisheng: error: ld.lld command failed with exit code 1
```

当时代码中的 shell 只有 `__global__ __aicore__`，函数体仅负责调用
`asc_vf_call<gather_simt_vf<...>>`，没有声明 Vector Core 函数类型属性。

## 适用场景

- 工程形态：Ascend950 Flash / kernel direct-launch 的 `.asc` 单文件构建；
- API 路径：`__global__` Device shell 调用 `asc_vf_call` 启动 `__simt_vf__`；
- 验证过的平台与版本：`950`，CANN 9.0.1（Bisheng/`ld.lld`）；与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于同类“Vector Core shell 启动 SIMT VF”的算子，不依赖 gather 计算语义；
- 路线优先级：`recommended_fastpath`；这里表示通过链接所需的推荐属性组合，不宣称性能收益；
- 明确不适用或未验证：已经显式声明其他合法 kernel type 的入口不适用；A2、A3、其他 CANN 版本以及 Cube/混合核入口未验证。

## 根因

SIMT VF 启动 shell 只写 `__global__ __aicore__` 而缺少 `__vector__` 函数类型属性，使 `ld.lld` 无法为模板实例化确定 AIV kernel type，因而链接失败。

## 解决配方

为调用 `asc_vf_call` 的 `__global__` shell 补上 `__vector__`，保留原有 `__aicore__`
执行空间限定和完整函数体。

### before

代码证据：revision=`trace pre-fix worktree snapshot`（未提交）；
path=`operators/gather/gather.asc`；symbol=`gather_kernel<T, I>`；lines=87–95。
该函数完整展示了缺少 kernel type 属性的出错形态。

```cpp
template <typename T, typename I>
__global__ __aicore__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR ctx)
{
    asc_vf_call<gather_simt_vf<T, I>>(
        dim3{SIMT_THREADS_PER_BLOCK, 1, 1},
        (__gm__ const T *)x, (__gm__ const I *)index, (__gm__ T *)y,
        (__gm__ const uint32_t *)ctx);
}
```

### after

代码证据：revision=`trace post-fix worktree snapshot`（随后提交为轨迹记录的短 SHA
`f06a72d`；40 位 SHA 未保留）；path=`operators/gather/gather.asc`；
symbol=`gather_kernel<T, I>`；lines=87–95。完整函数由 before 的完整写入与精确 Edit
确定性恢复；除新增 `__vector__` 外逐字不变。

```cpp
template <typename T, typename I>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR ctx)
{
    asc_vf_call<gather_simt_vf<T, I>>(
        dim3{SIMT_THREADS_PER_BLOCK, 1, 1},
        (__gm__ const T *)x, (__gm__ const I *)index, (__gm__ T *)y,
        (__gm__ const uint32_t *)ctx);
}
```

## 设计决策

- 选择给现有 `__aicore__` shell 叠加 `__vector__`，因为链接器缺的是 AIV 函数类型属性，而 `asc_vf_call` 调用关系和函数体本身无需改变；
- 轨迹曾考虑 `__qinqi__` 或把签名直接换成 `__global__ __vector__`，但没有在本工程实测这些备选，最终采用了同工具链参考工程已有的 `__aicore__ __vector__` 组合；
- 当前方案属于 `recommended_fastpath`：它是本轨迹中唯一完成“失败链接 → 修改 → 成功链接并上板测试”的路线，不据此推导性能优劣。

## 系统信息

- toolchain：CANN 9.0.1，Bisheng 编译器与 `ld.lld` 链接器；
- 目标平台：Ascend950 / dav-3510；
- 工程形态：Flash direct-launch，单个 `.asc` 翻译单元构建共享库；
- 相关文件：`operators/gather/gather.asc`；
- 相关符号：`gather_kernel<T, I>`、`gather_simt_vf<T, I>`、`asc_vf_call`。

## 预防

- 新写 SIMT VF 启动 shell 时，代码评审同时检查执行空间限定符与 kernel type 属性，不要只看到 `__aicore__` 就认为链接器能自动推导；
- 模板入口若对全部 dtype 组合重复出现同一 `auto derivate failed`，优先检查入口签名的函数类型属性，而不是逐个排查模板实参；
- 修复后必须重新经历 ASC 链接并运行 NPU 测试；只通过 Host 编译或静态检查不足以验证该问题。

## 验证信号

来源说明：本知识来自一次 Ascend950 Flash 算子开发中的首次远程链接失败、精确签名修复，以及修复后对同一 `.asc` 翻译单元的远程重新构建和 NPU pytest。

- 修复前，`ld.lld` 对全部模板实例化报告 `auto derivate failed`，Bisheng 以链接退出码 1 失败（来源平台：950；证据等级：tool_observed）；
- 补 `__vector__` 后，构建输出出现 `[100%] Linking ASC shared library` 与 `[100%] Built target`（来源平台：950；证据等级：tool_observed）；
- 同一修复后产物在 NPU 上收集 66 项测试并全部通过（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_scatter。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R4：工具报告 [100%] Built target op_gather；NPU/pytest 报告 66 passed。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R5：工具报告 [100%] Built target op_scatter。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 本结论只在 Ascend950 / CANN 9.0.1 的 SIMT VF 启动 shell 上实测；A2、A3 与其他 CANN 版本未验证，推断可迁移但必须重新构建确认；
- 已经使用工具链认可的其他 kernel type 属性、纯 Host 函数、普通 `__simt_vf__` callee 或不经 `asc_vf_call` 的入口，不应机械套用本配方；
- 本卡只解决链接器类型推导，不证明 kernel 的索引逻辑、精度或性能正确。

## 坏实践

- 只写 `__global__ __aicore__`，期待链接器从模板函数体自动推导 AIV kernel type；该写法已在本轨迹对全部实例化稳定触发链接失败；
- 看到多个 dtype 实例化同时失败就逐一修改模板分发；它们共享同一入口签名，逐实例排查会掩盖共同根因。