---
type: "Runbook"
title: "单TU编译NdDma报target feature需split构建"
description: "含 NdDma（5D DataCopy）或 __DAV_VEC__ 依赖的 kernel 用 ccec -xasc 单 TU 直调编译报 nd_dma_dci does not support the given target feature（-xasc device pass 按 dav-3510 MIX 整核编译，该内建仅 dav-c310-vec 向量子目标可用）；改用框架纯设备模式编设备 TU + -xasc --cce-host-only --cce-aicore-bin 编 host TU 的 split 构建。"
tags: ["task.implementation", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# 单TU编译NdDma报target feature需split构建

## 现象

把一个使用 NdDma（5 维 DataCopy）的注册算子（FusedRmsNormSigmoidMul，arch35）转成 `<<<>>>` kernel 直调工程时，用 ccec 单 TU 模式编译 host+device 一体的 kernel cpp：

```bash
ccec -xasc --npu-arch=dav-3510 -O2 -std=c++17 \
    -I$ASC/asc/include -I$ASC/ascendc/include -I$ASC/include \
    -c fused_rms_norm_sigmoid_mul_kernel.cpp
```

编译报错（`--asc-aicore-lang` 模式报错相同）：

```
asc/impl/basic_api/dav_3510/core_mng/roc/../../kernel_operator_data_copy_impl.h:1549:5:
error: function type 'void (void) noexcept' of 'nd_dma_dci' does not support the given target feature
1 error generated.
```

用一个最小 kernel（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); TPipe pipe; NdDmaDci();`）即可独立复现。围绕该错误尝试过的变体全部同样失败：`-xasc -mcpu=dav-c310-vec`、`-xasc --cce-aicore-arch=dav-c310-vec`（含与 `--cce-aicore-only` 组合）、`--cce-aiv`、`--offload-device-only`、`-D__DAV_C310__`、换 CANN 9.1.0 的 ccec；把子核名直接给 `--npu-arch`（`-xasc --npu-arch=dav-c310-vec`）则被拒绝：`ccec: error: Unsupported NPU architecture or soc: NPU arch is not supported!!!`。

## 适用场景

- **工程形态**：注册算子工程转 `<<<>>>` kernel 直调（kernel 直调/benchmark 工程），kernel 含 NdDma / `DataCopy<..., 5>` 或依赖 `__DAV_VEC__` 的 Reg SIMD、`asc_vf_call` 路径。
- **API 路径**：ccec（bisheng）编译器；`-xasc` / `--asc-aicore-lang` / `-xcce` 语言模式；`--npu-arch` 与 `--cce-aicore-arch` 架构选择。
- **验证过的平台与 CANN 版本**：Ascend950PR（arch35 / dav-3510，AIV 子目标 dav-c310-vec），CANN 9.2.0（ccec=bisheng 15.0.5）。与 frontmatter `platforms` 一致。
- **适用粒度**：`platform_wide`（950 系直调构建；其他 SoC 的子目标名未验证）。
- **路线优先级**：`recommended_fastpath`（该场景下 split 构建是唯一已验证可行的编译路线）。
- **明确不适用 / 未验证场景**：不使用 NdDma 且不依赖 `__DAV_VEC__` 的简单 kernel 单 TU `-xasc` 可直接编译（本轨迹的探针 kernel 即如此），无需 split；非 950 平台未验证。

## 根因/解决配方

**根因（单句）**：`-xasc`（及 `--asc-aicore-lang`）的 device pass 以 dav-3510 **MIX 整核**目标编译，而 `nd_dma_dci` 内建函数仅在 **dav-c310-vec 向量子目标**下可用，MIX / cube 目标都不支持该 target feature。

**解决配方 = split 构建**（设备 TU 与 host TU 分开编译再经 `--cce-aicore-bin` 结合）：

```bash
ASC=$CANN_HOME/x86_64-linux
# 框架设备编译 include 集（与 CANN op 编译器 compile_op.py 生成的完全一致，共 22 项）
FW_INCLUDES="-I$ASC/asc/impl/adv_api -I$ASC/asc/impl/basic_api -I$ASC/asc/impl/c_api \
 -I$ASC/asc/impl/basic_api/reg_compute -I$ASC/asc/impl/simt_api -I$ASC/asc/impl/utils \
 -I$ASC/asc -I$ASC/asc/include -I$ASC/asc/include/adv_api -I$ASC/asc/include/basic_api \
 -I$ASC/asc/include/aicpu_api -I$ASC/asc/include/c_api -I$ASC/asc/include/basic_api/reg_compute \
 -I$ASC/asc/include/simt_api -I$ASC/asc/include/utils -I$ASC/include -I$ASC/include/ascendc \
 -I$ASC/ascendc/act -I$ASC/tikcpp -I$ASC/tikcpp/tikcfw -I$ASC/tikcpp/tikcfw/impl \
 -I$ASC/tikcpp/tikcfw/interface"

# [1] 设备 TU：框架纯设备模式（与原注册工程 kernel 编译同构）
ccec -c -O3 -xcce device.cpp \
    --cce-aicore-arch=dav-c310-vec -D__DAV_C310__ --cce-aicore-only -std=c++17 \
    $FW_INCLUDES -o device.o

# [2] host TU：eager launch 胶水，嵌入设备二进制
ccec -xasc --cce-host-only --cce-aicore-bin=device.o \
    --npu-arch=dav-3510 -fPIC -O2 -std=c++17 \
    -I$ASC/asc/include -I$ASC/ascendc/include -I$ASC/include \
    -c host.cpp -o host.o

# [3] 链接 .so（依赖按报错补齐，见失效边界）
ccec -shared -fPIC host.o -L$ASC/lib64 -lascendcl -lnnopbase -lopapi -lplatform \
    $ASC/lib64/libtiling_api.a -o libkernel.so
```

三个必要细节（缺一即失败，均为实测）：

1. **`-std=c++17` 必须显式给**：否则报 `error: statement not allowed in constexpr function`（kernel_event.h:341）与 `'auto' not allowed in template parameter until C++17`（kernel_utils_macros.h:343），容易误判为架构问题。
2. **include 集必须用框架全集**：只给 `-I$ASC/asc/include` 会先后报 `fatal error: 'kernel_tpipe.h' file not found` → `fatal error: 'impl/utils/sys_macros.h' file not found`。该 include 集的权威来源是 CANN 安装内 `python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py`（约 1289–1310 行）生成的编译选项，不要逐个试错补 -I。
3. **`--cce-aicore-arch` 必须是 `dav-c310-vec`**：`dav-c310-cube` 同样报 target feature 错误；AIV kernel 用 vec 子目标。

host TU 编译产物的符号形态（用于确认胶水已生成）：`nm` 可见 kernel mangled 名（T）、`__cce_rtKernelEagerLaunch`（U）、`__cce_rtLinkedDevBinaryEagerRegister`（U）、`rtRegisterFuncSymbol`（U）；链接成 .so 后出现 `BishengDevBinaryEagerRegister`（T）。

## 设计决策

- **为什么放弃找 `-xasc` 的开关**：`--cce-aicore-arch=dav-c310-vec` 与 `-xasc` 组合、`-mcpu`、`--cce-aiv`、`--offload-device-only`、`-D__DAV_C310__`、CANN 9.1.0 ccec 全部实测无效；`-xasc` 的 `--npu-arch` 只接受 SoC 级值（dav-3510），子核值直接 `Unsupported NPU architecture or soc`。结论：单 TU eager 模式没有 vec 子目标通道。
- **为什么设备侧沿用框架编译**：与原注册工程 kernel 编译完全同构（同 arch、同 `-O3`、同 `__DAV_C310__`），保证设备二进制行为与注册路径一致；host 侧只补 eager launch 胶水，改动面最小（忠实转换原则）。
- **当前方案属哪档路线**：`recommended_fastpath`（该场景唯一已验证路线；备选的 objcopy/二进制 patch 方向在 TPipe 507015 排查中已被证伪，见坏实践）。

## 系统信息

- 工具链：CANN 9.2.0（`ASCEND_TOOLKIT_HOME` 指向的安装），`bin/ccec` 即 bisheng（15.0.5，md5 `1a8d991e151accf31ea31a562600108b`）。
- 目标平台：Ascend950PR（arch35 / dav-3510；AIV 子目标 dav-c310-vec）。
- 目录布局（本轨迹实际产物）：`direct_baseline/build.sh`（split 构建入口）、`csrc/op_kernel/fused_rms_norm_sigmoid_mul_device.cpp`（设备 TU，仅 include 共享 entry 头）、`csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp`（host TU）、`csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h`（注册工程 kernel.h 忠实拷贝 + `#if !defined(__NPU_HOST__)` 两行 guard）、`lib/`（device.o / host.o / libfrnsm.so / kernel_symbols.json）。

## 验证信号

（以下信号来自一次 FusedRmsNormSigmoidMul 注册工程转直调的开发会话，均在 Ascend950PR / CANN 9.2.0 工具链上实测）

- 单 TU `-xasc --npu-arch=dav-3510` 编译含 NdDma 的 kernel 报 `nd_dma_dci ... does not support the given target feature`（来源平台：950；证据等级：tool_observed）。
- `--asc-aicore-lang`、`-mcpu=dav-c310-vec`、`-xasc --cce-aicore-arch=dav-c310-vec`、`--cce-aiv`、`--offload-device-only`、`-D__DAV_C310__`、CANN 9.1.0 ccec 等变体全部复现同一错误（来源平台：950；证据等级：tool_observed）。
- `-xasc --npu-arch=dav-c310-vec` 报 `Unsupported NPU architecture or soc`（来源平台：950；证据等级：tool_observed）。
- 框架纯设备模式 `dav-c310-vec` + `-std=c++17` + 框架 include 集编译通过（`ndma_fw4.o` 生成）（来源平台：950；证据等级：tool_observed）。
- 同模式下 `dav-c310-cube` 编译失败（2 errors）（来源平台：950；证据等级：tool_observed）。
- host TU `--cce-host-only --cce-aicore-bin=<设备.o>` 编译 0 错误，链接出的 .so 导出 `BishengDevBinaryEagerRegister`/kernel mangled 符号（来源平台：950；证据等级：tool_observed）。
- 真实算子 split 构建 5 步全部完成并产出 libfrnsm.so（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该配方只解决**编译**：含 TPipe 的 kernel 在 eager 直调下运行期同步会 507015，根因与修复（链接后 AIVEC magic 补丁）见 [tpipe_eager_launch_507015_aivec_magic_patch](tpipe_eager_launch_507015_aivec_magic_patch.md)。
- `dav-c310-vec` 是 950 系（arch35）的 AIV 子目标名；其他 SoC 的子目标名与可用性未验证（未验证，推断可按同样方法从 `compile_op.py` 反推）。
- 链接行依赖按用途取舍：可执行文件用 aclnn 时缺 `-lnnopbase` 报 `libnnopbase.so: error adding symbols: DSO missing from command line`；host 侧 tiling 用 `PlatformAscendCManager` 时缺库会 dlopen 报 `undefined symbol: _ZN16platform_ascendc22PlatformAscendCManager15platformInitMtxE`（该符号实际定义在**静态** `libtiling_api.a`，`-lplatform` 单独不够）。

**未决项**：

- `-xasc` 单 TU 模式是否存在官方的 vec 子目标开关（`ccec --help` 未见）未证实。
## 坏实践

- 只给 `-I$ASC/asc/include` 就尝试 `-xcce`：连续踩 `kernel_tpipe.h not found` → `impl/utils/sys_macros.h not found`，逐个补 -I 不如直接抄框架全集。
- 漏 `-std=c++17`：constexpr / auto 模板参数报错与目标特性错误混在一起，容易误判为架构选择问题。

## 预防

- 转直调任务先确认 kernel 是否使用 NdDma / `__DAV_VEC__` 路径（grep `DataCopy` 维度、`NdDma`、`asc_vf_call`）；命中则直接按 split 构建搭工程（完整转换配方见 [registered_to_direct_launch_split_recipe](registered_to_direct_launch_split_recipe.md)），不要从单 TU `-xasc` 起步。
