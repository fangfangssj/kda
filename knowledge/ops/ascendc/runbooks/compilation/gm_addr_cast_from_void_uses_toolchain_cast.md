---
type: "Runbook"
title: "GM_ADDR host侧cast被拒或静默剥离属性"
description: "dav-3510 host pass 拒绝 void* 到 __gm__ 地址空间指针的 static_cast（casts away qualifiers）与 reinterpret_cast（not allowed）；C 风格转换可编过但静默剥离 cce_global 属性只剩 warning。直调 launcher 推荐 GM_ADDR 直接作参数类型（C ABI 8 字节指针槽），双 TU 头文件用条件宏别名。"
tags: ["task.debugging", "topic.implementation", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# GM_ADDR host侧cast被拒或静默剥离属性

## 现象

**R1（首次复现，CANN 9.0.1）**：在为 Ascend950 `dav-3510` 构建单文件 direct-launch Ascend C 工程时，开发者已经把 device kernel、tiling 与 PyTorch host 入口写进 `gather.asc`，随后通过远程 helper 触发 CMake/ASC 编译。编译停在 host 函数 `op_gather::gather` 将 `at::Tensor::data_ptr()` 交给 kernel launch 的位置，四个输入都报同一错误：

```text
gather.asc:317:18: error: static_cast from 'void *' to '__gm__ uint8_t *' (aka '__gm__ unsigned char *') casts away qualifiers
auto x_ptr = static_cast<GM_ADDR>(xP.data_ptr());
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

编译器同时展开 `GM_ADDR` 为 `__gm__ uint8_t*`，并指出 `__gm__` 对应 `cce_global` 属性；因此故障发生在 host 指针到 Ascend C GM 地址空间指针的适配层，尚未进入 kernel 运行。

**R2（第二次独立复现，CANN 9.2.0，split 构建 host TU）**：为 `<<<>>>` kernel 直调库写 host 侧 C ABI launcher 时，把外部传入的 `void*` 设备指针转成 `GM_ADDR` 传给 kernel。`reinterpret_cast` 形态直接编译失败：

```cpp
GM_ADDR xg = reinterpret_cast<GM_ADDR>(x);
```

```text
abi.cpp:13:18: error: reinterpret_cast from 'void *' to '__gm__ uint8_t *'
      (aka '__gm__ unsigned char *') is not allowed
    GM_ADDR xg = reinterpret_cast<GM_ADDR>(x);
                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
2 errors generated.
```

而 C 风格强转 `GM_ADDR xg = (GM_ADDR)x;` 能编译，但每处只报 warning：

```text
warning: 'cce_global' attribute ignored when parsing type [-Wignored-attributes]
    GM_ADDR xg = (GM_ADDR)x;
note: expanded from macro 'GM_ADDR'
#define GM_ADDR __gm__ uint8_t*
note: expanded from macro '__gm__'
#define __gm__ __attribute__((cce_global))
```

即 host pass 把 `__gm__`（`cce_global`）属性整个丢掉，转换结果退化为普通 `uint8_t*`。

## 适用场景

- 工程形态：Ascend C direct-launch 工程的 host 入口（单文件 `.asc` + CMake/ASC 工具链，或 split 构建直调库的 host TU / 对外 C ABI launcher 头文件，ctypes、g++ 侧都要链接的接口）；
- API 路径：`at::Tensor::data_ptr()` → `GM_ADDR` → 模板 kernel launch；`GM_ADDR` / `__gm__` / `__attribute__((cce_global))` 地址空间限定；`<<<>>>` launch 参数；
- 验证过的平台与 CANN 版本：Ascend950 / `dav-3510`，R1 为 CANN 9.0.1 单文件 ASC 编译链、R2 为 CANN 9.2.0（ccec/bisheng 15.0.5）split 构建 host pass，与 frontmatter `platforms: [950]` 一致；
- 适用粒度：`platform_wide`，结论约束同一 CANN/Bisheng 工具链里的 host→GM 指针转换，不依赖具体算子数学语义；
- 路线优先级：`recommended_fastpath`——R2 验证的最优形态是 **GM_ADDR 直接作为 host 函数的参数类型**（C ABI 8 字节指针槽与外部 `void*`/`uint64_t` 对接，零显式转换、属性不丢失）；双 TU 共享头文件用 `__CCE__` 条件宏别名。C 风格转换 `(GM_ADDR)x` 是 R1 验证可通过的参考形态，但会静默剥离 `__gm__` 属性（R2 实测，见坏实践）；`static_cast` 与 `reinterpret_cast` 均被编译器拒绝，`banned`。
- 明确不适用 / 未验证场景：普通 CPU C++ 编译单元、没有 `__gm__` 地址空间限定的指针、A2/A3 工具链，以及上述之外的其他 CANN 版本未验证；device 侧（`__NPU_DEVICE__`）代码内的指针转换语义不同，本卡结论不适用（R2 已标注）。

## 根因

在 dav-3510 的 ASC/ccec 编译单元里，`GM_ADDR` 展开为带 `__gm__`/`cce_global` 地址空间限定的 `uint8_t*`。host pass 对该限定指针的转换约束是：

1. **C++ 命名转换被拒**：`static_cast` 报 `casts away qualifiers`（R1），`reinterpret_cast` 报 `is not allowed`（R2）——host 语义下不允许用命名转换把普通 `void*` "升级"为带地址空间属性的指针；
2. **C 风格转换放行但丢属性**：C 风格转换在 host 语义下合法，但 `cce_global` 属性被静默剥离（仅 `-Wignored-attributes` warning），结果退化为普通 `uint8_t*`（R2 实测）。

## 解决配方

### 路线 A（R2，recommended）：GM_ADDR 直接作参数类型 + 条件宏别名

GM_ADDR 直接作为 host 函数的参数类型，靠 C ABI 的 8 字节指针槽与外部 `void*`/`uint64_t` 对接，函数体内不出现任何显式转换（探针 verbatim）：

```cpp
#include "kernel_operator.h"
using namespace AscendC;

__global__ __aicore__ void abi_probe(GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    (void)x; (void)y; (void)pipe;
}

extern "C" void abi_launch_gmaddr(GM_ADDR x, GM_ADDR y, uint32_t blockDim, void* stream)
{
    abi_probe<<<blockDim, nullptr, stream>>>(x, y);
}
```

g++/ctypes 侧按普通指针槽理解即可（实测 `extern "C" void abi_from_u64(uint64_t x, uint64_t y, uint32_t blockDim, void* stream)` 的 g++ TU 链接调用通过）。

对外发布头文件用条件宏，同一份头在 ccec TU 与 g++ TU 下都成立（真实工程 verbatim）：

```cpp
// ccec（-xasc）翻译单元内 GM_ADDR 来自 kernel 头；g++/ctypes 侧按 void* 理解。
// 两者在 C ABI 层都是 8 字节指针槽，符号一致。
#ifdef __CCE__
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
#define FRNSM_GM_ADDR GM_ADDR
#else
#define FRNSM_GM_ADDR void*
#endif
```

### 路线 B（R1，参考形态）：host 入口内 C 风格转换

R1 场景（PyTorch 扩展 host 入口，`data_ptr()` 处）把 `static_cast<GM_ADDR>(...)` 改成工具链参考形态 `(GM_ADDR)...`：

### before

代码来源说明：修复前完整 `file_write` 经同轨迹中先发生的 UB 分母 exact edit 重建；revision 短哈希 `af67593`，仓库相对路径 `operators/gather/gather.asc`，symbol `op_gather::gather`，行段 274–350。40 位 revision 与精确 blob URL 未在 trace 中恢复，正式发布前需补齐。

```cpp
    auto x_ptr = static_cast<GM_ADDR>(xP.data_ptr());
    auto idx_ptr = static_cast<GM_ADDR>(idxP.data_ptr());
    auto y_ptr = static_cast<GM_ADDR>(yP.data_ptr());
    auto meta_ptr = static_cast<GM_ADDR>(metaNpu.data_ptr());
    auto tile = static_cast<uint32_t>(t.tileElems);
    auto nl = nLead;
    auto pa = static_cast<int64_t>(t.pathA ? 1 : 0);

    const auto dtype = x.scalar_type();
    if (dtype == at::kHalf) {
        gather_kernel<half><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kFloat) {
        gather_kernel<float><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kBFloat16) {
        gather_kernel<bfloat16_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kChar) {  // int8
        gather_kernel<int8_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kInt) {
        gather_kernel<int32_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kLong) {
        gather_kernel<int64_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else {
        TORCH_CHECK(false, "op_gather::gather unsupported dtype: ", dtype);
    }
```

### after

代码来源说明：同一完整函数经 exact edit 重建，并在短哈希 `d71e48f` 固化后再次远程编译；仓库相对路径 `operators/gather/gather.asc`，symbol `op_gather::gather`，行段 274–350。40 位 revision 与精确 blob URL 未在 trace 中恢复，正式发布前需补齐。

```cpp
    auto x_ptr = (GM_ADDR)xP.data_ptr();
    auto idx_ptr = (GM_ADDR)idxP.data_ptr();
    auto y_ptr = (GM_ADDR)yP.data_ptr();
    auto meta_ptr = (GM_ADDR)metaNpu.data_ptr();
    auto tile = static_cast<uint32_t>(t.tileElems);
    auto nl = nLead;
    auto pa = static_cast<int64_t>(t.pathA ? 1 : 0);

    const auto dtype = x.scalar_type();
    if (dtype == at::kHalf) {
        gather_kernel<half><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kFloat) {
        gather_kernel<float><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kBFloat16) {
        gather_kernel<bfloat16_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kChar) {  // int8
        gather_kernel<int8_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kInt) {
        gather_kernel<int32_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else if (dtype == at::kLong) {
        gather_kernel<int64_t><<<t.numBlocks, nullptr, aclStream>>>(
            x_ptr, idx_ptr, y_ptr, meta_ptr, S, L, numRows, t.rowsPerBlock, tile, nl, pa);
    } else {
        TORCH_CHECK(false, "op_gather::gather unsupported dtype: ", dtype);
    }
```

## 设计决策

- 选择 `(GM_ADDR)tensor.data_ptr()`（R1）：同一 dav-3510 编译命令已证明该写法能通过 `__gm__` 地址空间指针检查，且与工程参照中的 launch 形态一致；R1 当时未尝试 `reinterpret_cast`；
- R2 补齐 `reinterpret_cast` 判定：报 `not allowed`，同样不可用，不能按标准 C++ 直觉推断为等价替代；
- R2 升级路线为 GM_ADDR 直接作参数类型：三个候选（reinterpret_cast 编译拒绝、C 风格强转属性被剥离有静默语义风险、GM_ADDR 直接传参 0 错误 0 告警）中唯一既可行又保属性的形态；
- 之所以不让 host 侧统一用 `uint64_t`：kernel 侧 launch 仍需 `GM_ADDR` 语义，条件宏让同一头文件在两侧各自取正确类型，避免在 launcher 里出现任何显式转换；
- C 风格 cast 只应局限在工具链要求的 `data_ptr()`→`GM_ADDR` 边界（R1 形态可用但属性被剥离，能不改就不改），不应扩散到其他指针转换。

## 系统信息

- R1：目标 Ascend950PR_957b / `dav-3510`；CANN 9.0.1，Bisheng/ASC 编译链；单文件 `gather.asc` + CMake direct-launch PyTorch 扩展；相关文件 `operators/gather/gather.asc`，symbol `op_gather::gather`，产物 `libop_gather.so`。
- R2：Ascend950PR，CANN 9.2.0（ccec/bisheng 15.0.5）；split 构建直调库（设备 TU `dav-c310-vec` + host TU `-xasc --cce-host-only`）的 host 侧 C ABI 探针 `abi.cpp`/`abi3.cpp` 与真实工程 launch.h（`FRNSM_GM_ADDR` 条件宏形态）。

## 预防

- 在 direct-launch host 入口审查所有 `data_ptr()`/`void*`：目标参数若是 `GM_ADDR`，不要机械套用普通 C++ 的 `static_cast`/`reinterpret_cast`；
- 新写直调 launcher 头文件时一律 GM_ADDR（或条件宏别名）直接作参数类型，杜绝在函数体内做 `void*`→`GM_ADDR` 转换；
- 编译报错同时出现 `void*`、`__gm__ uint8_t*` 与 `cce_global` 时，优先检查地址空间限定转换，而不是修改 kernel 模板参数；
- 出现 `'cce_global' attribute ignored when parsing type` warning 时不要忽略——它说明 C 风格转换已把地址空间属性剥离（R2）；
- 在提交前用固化代码状态重新触发一次完整 ASC 编译，避免修复只停留在未提交工作树。

## 验证信号

来源为两次独立轨迹的构建实测（R1：Ascend950 direct-launch 算子开发的首次完整实现构建、cast 精确修复和固化提交后复验；R2：同日另一注册转直调会话的 host TU cast 探针与真实工程头文件）。

R1（CANN 9.0.1，单文件 ASC 链）：

- 修复工作树后，同一远程 dav-3510 构建完成 `Linking ASC shared library libop_gather.so` 与 `[100%] Built target op_gather`（来源平台: 950 / dav-3510；证据等级: tool_observed）；
- cast 修复固化到提交后再次远程构建，输出 `4 warnings generated`、`[100%] Linking ASC shared library libop_gather.so`、`[100%] Built target op_gather`（来源平台: 950 / dav-3510；证据等级: tool_observed）。

R2（CANN 9.2.0，split 构建 host pass，第二次独立复现）：

- `reinterpret_cast<GM_ADDR>(void*)` 报 2 错误、无产物（来源平台：950；证据等级：tool_observed）。
- C 风格 `(GM_ADDR)x` 编译通过但每处报 `'cce_global' attribute ignored when parsing type` 告警（共 4 处）（来源平台：950；证据等级：tool_observed）。
- GM_ADDR 直接作参数形态编译 0 错误，`abi_launch_gmaddr` 符号存在；g++ TU 以 uint64 ABI 调用链接通过（来源平台：950；证据等级：tool_observed）。
- 真实直调工程 launch.h 采用 `FRNSM_GM_ADDR` 条件宏形态，后续构建全程无该告警（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本结论在 CANN 9.0.1（单文件 ASC 链）与 9.2.0（split 构建 host pass）的 `dav-3510` 实测；其他 CANN/Bisheng 版本可能改变地址空间转换规则；
- A2、A3 未验证，不能从这两条轨迹推断相同 cast 行为；
- 若 API 直接返回带正确地址空间限定的类型，或 launch 参数不使用 `GM_ADDR`，本修复不适用；
- R2 的 `__CCE__` 宏判定依赖 ccec 预置宏（`-E -dM` 实测 ccec 下定义 `__CCE__ 1`，g++ 下不定义），换编译器需重新确认；
- device 侧（`__NPU_DEVICE__`）代码内的指针转换语义不同，本卡不适用。

## 坏实践

- 看到普通 C++ 项目偏好 `static_cast`，就不检查 Ascend C 的 `__gm__` 地址空间限定，机械替换所有 C 风格 cast；
- 用 C 风格强转"绕过"命名 cast 报错并当作最终修复——能编过但 `__gm__` 属性被静默剥离，属性相关检查/寻址语义可能悄悄变化，只有 warning 提示（R2 实测；本卡 R1 曾以该形态作为修复，R2 证据表明它是带静默隐患的参考形态而非最优解）；
- 只看编译输出末尾的 warning 而忽略前面的 `error:`，把未生成目标库误判为构建成功；
- 修复后只在未提交工作树构建一次，不检查 Git 中固化的代码仍是否保留失败写法。

## 误诊

- 该错误发生在 host 入口的指针类型适配，不是 device kernel 的计算、同步、UB 或精度问题；修改 kernel body 不会消除编译器对 `static_cast<GM_ADDR>`/`reinterpret_cast<GM_ADDR>` 的拒绝。
