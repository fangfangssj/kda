---
type: "Runbook"
title: "l0op::Contiguous未定义符号致安装校验失败"
description: "自定义算子 libcust_opapi.so 引用 l0op::Contiguous（实现只在 CANN libopapi.so 导出，官方 CMake 不为 cust_opapi 链接它），.run 安装校验/独立 dlopen 报 undefined symbol。三条处置路线：算子已 AutoContiguous 时删除显式调用；确需调用时给最终 cust_opapi 目标补链 libopapi.so（安装校验无 --force 通过）；或按校验器隔离加载的已知行为判良性后 --force 安装 + 运行期把 CANN lib64 纳入 LD_LIBRARY_PATH。"
tags: [task.debugging, topic.compilation, topic.data_movement]
status: stable
sources: []
created_at: "2026-09-14T12:18:28Z"
updated_at: '2026-09-16T10:43:46Z'
platforms: ["950"]
---

# l0op::Contiguous未定义符号致安装校验失败

## 现象

Cummin 自定义算子包编译成功后，执行 `./build/custom_opp_ubuntu_x86_64.run` 安装，安装在"validate shared libraries before installation"阶段被拒绝：

```
[ops_custom] [ERROR] Failed to load libcust_opapi.so wish env ASCEND_HOME_PATH(...),error message is libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor.
[ops_custom] [ERROR] Shared library validation failed. To skip this validation and continue installation, run this package again with --force.
```

报错符号 demangle 后是 `l0op::Contiguous(aclTensor const*, aclOpExecutor*)`。当时 `op_api/aclnn_cummin.cpp` 的 GetWorkspaceSize 里在调用算子前做了显式连续化：

```cpp
#include "aclnn_kernels/contiguous.h"
...
auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

auto result = l0op::Cummin(xContiguous, dim, values, indices, uniqueExecutor.get());
```

随后用 `find $ASCEND_HOME_PATH -name "*.so" | xargs nm -D | grep Contiguous | grep l0op` 探针确认：`_ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`（T，已定义）由系统侧 `libopapi.so`、`libopapi_math.so` 等库提供，头文件声明在 `include/aclnn_kernels/contiguous.h`。

R2（CANN 9.1.0，独立会话，--extract 校验）：自研算子包（op_api 含手写 L0 实现）全量构建后对产物做 `--extract` 内容核对，安装前校验打出同样两条 ERROR；解包后的文件本身齐全（libcust_opapi.so、kernel .o、liboptiling.so 都在）。对官方参考 add_example 包做同样的 `--extract` 校验，输出与自研包**逐字相同**——证明这是校验器的已知行为而非本工程缺陷。

R3（CANN 9.2.0，独立会话）：`bash build.sh --soc=ascend950` 编译产出 `.run` 后执行 `--quiet` 安装，同样在 "validate shared libraries before installation" 阶段报 `undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor` 直接退出。该算子的 L2 层对非连续输入做 `l0op::Contiguous(x, uniqueExecutor.get())`、输出用 `l0op::ViewCopy(opResult, y, ...)`，两个函数只声明在 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn_kernels/contiguous.h`（namespace l0op），实现不在算子仓里。

R4（CANN 9.1.0，独立会话，torch 侧 dlopen）：同一符号问题在两个场景爆雷。场景一：重装 vendor 包时 `.run` 安装器自校验拒绝安装（构建循环里每次重装都会遇到，两次复现）。场景二：跑 PyTorch ST 测试时用 `python3 test.py --lib <vendor>/op_api/lib/libcust_opapi.so`，`torch.ops.load_library` 内部 `ctypes.CDLL` dlopen 抛 OSError：

```
OSError: <CANN安装根>/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor
```

（当时 LD_LIBRARY_PATH 只含 vendor 的 op_api/lib，未含 CANN lib64。）另有第三坑：`--lib` 传裸 `libcust_opapi.so` 时即使加载成功也会报 `'_OpNamespace' ... object has no attribute 'forward'`——torch.ops 命名空间由桥接库 `libtorch_adapter.so`（内含 TORCH_LIBRARY 注册并链接 cust_opapi）注册，`--lib` 应指向它。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），op_api 层 aclnn 两段式接口（GetWorkspaceSize + 执行）使用 `l0op::` 辅助函数；vendor 包安装（.run 校验）与 PyTorch/ctypes 等直接 dlopen 算子适配库的测试接入。
- API 路径：`aclnn_kernels/contiguous.h` 的 `l0op::Contiguous`；算子定义 `OpDef` 的 `AutoContiguous()`（声明位置与语义见 [ACLNN 561000 非连续张量需 AutoContiguous 卡](aclnn_561000_autocontiguous.md)）；`x86_64-linux/lib64/libopapi.so`（符号导出方）；`x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake` 的 `npu_op_package_add`（打包链接方）；custom_opp run 包安装器的共享库独立加载校验；`LD_LIBRARY_PATH`/`LD_PRELOAD`。
- 验证过的平台与 CANN 版本：Ascend 950 环境编译 + CANN 9.1.0（安装校验、--extract、torch dlopen）与 9.2.0（安装校验 + 链接修复）。
- 适用粒度：`workflow_only`（安装校验与链接边界，属工程/构建层；随 build track 落 compilation）。
- 路线优先级：三路线按场景选择——①算子已 AutoContiguous：删除冗余调用（recommended_fastpath）；②确需 l0op 调用：给最终 cust_opapi 目标补链 libopapi.so（recommended_fastpath，安装校验无 --force 通过）；③保持模板链接配置：判良性后 `--force` 安装 + 运行期把 CANN lib64 纳入 LD_LIBRARY_PATH（safety_fallback）。
- 明确不适用/未验证场景：符号不在任何 CANN 已导出库中的真链接缺陷（须修打包链接，本卡判别方法同样适用）；NPU 侧 kernel 行为无关；不经 dlopen、由 GE/框架托管加载的场景（框架自带解析链）。

## 根因

`l0op::Contiguous`/`l0op::ViewCopy` 只在 CANN 系统库（libopapi.so 家族）中定义，自定义算子的 `libcust_opapi.so` 引用它但安装器在校验阶段以独立方式加载该 .so（不保证 libopapi.so 在其解析路径上），未定义符号直接判失败。而该算子的 `OpDef` 已经声明 `AutoContiguous()`，框架会在算子执行链路内自动处理输入连续性，显式调用完全冗余。

R3 机制补全（CANN 9.2.0，本会话逐项核实）：

1. `nm -D $ASCEND_HOME_PATH/x86_64-linux/lib64/libopapi.so | grep -E "Contiguous|ViewCopy"` 可见 `T Contiguous`、`T ViewCopy` 等导出符号——实现确实在 libopapi.so。
2. `ldd` 安装目录下的 `libcust_opapi.so` 依赖清单里没有 `libopapi.so`，且 `nm -D` 显示 `_ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`、`_ZN4l0op8ViewCopyEPK9aclTensorS2_P13aclOpExecutor` 为 `U`（undefined）。
3. 读 `func.cmake` 1353 行附近：`target_link_libraries(${target_package_name}_ascendc_cust_opapi PRIVATE intf_pub $<IF:...cust_intf_pub...> ${_ascendc_aclnn_target})`——官方链接清单不含 opapi，CANN cmake 也没有定义名为 opapi 的 target。
4. 安装器的共享库校验会 try-load `libcust_opapi.so`，未定义符号 → 校验失败拒绝安装。

R2/R4 机制（校验环境与加载环境）：安装器的共享库校验在一个没有把 `lib64`（libopapi.so 所在目录）纳入加载路径的环境里做 dlopen 式检查，l0op 符号因此"未定义"；已成功安装运行的其他自定义算子的 `libcust_opapi.so` 里，`l0op::Contiguous`/`l0op::ViewCopy` 同样是 `U`（未定义）——运行时由 `LD_LIBRARY_PATH` 中的 `libopapi.so` 解析。R4 全库扫描显示该符号由 `libopapi.so` / `libopapi_cv.so` / `libopapi_math.so` / `libopapi_nn.so` / `libopapi_oam.so` / `libopapi_transformer.so`（`lib64` 与 `x86_64-linux/lib64` 双目录）共同提供。

## 根因/解决配方

### 路线一（原卡）：删除冗余调用

`OpDef` 已声明 `AutoContiguous()` 时，显式连续化与框架自动处理重复，删除 `#include "aclnn_kernels/contiguous.h"` 与显式调用，把原张量直接传给算子：

修复前：

```cpp
#include "aclnn_cummin.h"
#include "cummin.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
...
    if (HasEmptyTensor(x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::Cummin(xContiguous, dim, values, indices, uniqueExecutor.get());
```

修复后：

```cpp
#include "aclnn_cummin.h"
#include "cummin.h"
#include "aclnn_kernels/common/op_error_check.h"
...
    if (HasEmptyTensor(x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto result = l0op::Cummin(x, dim, values, indices, uniqueExecutor.get());
```

### 路线二（R3，CANN 9.2.0）：给最终 cust_opapi 目标补链 libopapi.so

在 `npu_op_package_add` 之后对最终共享库目标追加 libopapi.so 链接。注意 `npu_op_library` 生成的 `cust_opapi` 是 OBJECT 库，OBJECT 库的 target_link_libraries 不会把动态库依赖传播给最终 so，必须链到 `${package_name}_ascendc_cust_opapi`：

```cmake
# =============================================================================
# Add libraries to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)

# 链接 CANN 标准库 libopapi.so：aclnn_kernels/contiguous.h 中声明的
# l0op::Contiguous / l0op::ViewCopy 由 libopapi.so 提供；官方 npu_op_package_add
# 不链它，缺失时 libcust_opapi.so 存在未定义符号，安装器 symbol validation 失败
if(TARGET ${package_name}_ascendc_cust_opapi)
    find_library(OPAPI_LIB opapi
        PATHS $ENV{ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64
        NO_DEFAULT_PATH
    )
    if(OPAPI_LIB)
        target_link_libraries(${package_name}_ascendc_cust_opapi PRIVATE ${OPAPI_LIB})
        message(STATUS "Linked libopapi.so for l0op::Contiguous/ViewCopy: ${OPAPI_LIB}")
    else()
        message(WARNING "libopapi.so not found, l0op::Contiguous/ViewCopy symbols will be unresolved")
    endif()
endif()
```

提交前自查：`find build -name libcust_opapi.so | head -1 | xargs ldd | grep opapi`，能 grep 到才算链上。

### 路线三（R2/R4/R5）：判良性 + --force + 运行期补加载路径

适用前提：算子确需 l0op 调用（Contiguous/ViewCopy 是 aclnn 两段式接口处理非连续 Tensor 的标准结构，与 add_example 官方模板一致），且探针确认符号由 CANN 运行时提供。

```bash
# 安装：按报错提示跳过校验
bash build/custom_opp_ubuntu_x86_64.run --quiet --force

# 运行期：两段库路径（vendor op_api/lib 优先，其次 CANN lib64）
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH}
```

torch 测试侧：`--lib` 指向桥接库 `libtorch_adapter.so`（内含 TORCH_LIBRARY 注册并链接 cust_opapi），不要传裸 `libcust_opapi.so`（不注册 torch.ops，报 `no attribute 'forward'`）。不要用 `LD_PRELOAD=libopapi.so` 常驻解决——会破坏 GE 图模式 TBE 编译初始化（见 [LD_PRELOAD libopapi 致 GE TBE 初始化失败卡](../runtime/ld_preload_libopapi_e40021_tbe_init.md)）。

安装后核验登记：`cat $ASCEND_OPP_PATH/vendors/config.ini`（load_priority 含本 vendor）。

## 判别方法（探针汇总，均可直接执行）

```bash
# 1. 在 CANN 运行时库中找符号提供者（nm -D 而不是 grep 文件名）
for lib in $ASCEND_HOME_PATH/lib64/libnnopbase.so \
           $ASCEND_HOME_PATH/lib64/libopapi.so \
           $ASCEND_HOME_PATH/lib64/libascendcl.so; do
    if [ -f "$lib" ]; then
        nm -D "$lib" 2>/dev/null | grep -c "l0op.*Contiguous" && echo "  ^found in $lib"
    fi
done

# 2. 对照参考包：对官方 add_example 产物做同样的 --extract 校验，输出逐字相同即校验器已知行为

# 3. 对照已成功安装运行的算子：其 libcust_opapi.so 对同一符号同样为 U 引用
nm -D $ASCEND_HOME_PATH/opp/vendors/<other_op>_custom/op_api/lib/libcust_opapi.so | grep Contiguous

# 4. 确认谁引用了 l0op（工作区源码）
grep -rln "l0op::Contiguous" operators/*/op_api/
```

判读：探针指向"提供者在 CANN 运行时库"即可判良性/可修复；若全树找不到符号定义，则是真链接缺陷，不适用路线三。

## 设计决策

- 路线一为什么选删除而不是补链接：显式连续化与 OpDef 的 AutoContiguous 重复；补链接需要让 libcust_opapi.so 声明对 libopapi.so 的运行时依赖，而安装器校验阶段的加载上下文不受控，删除冗余调用是最小且确定的修复。
- 路线三为什么停在"判定良性"而不改链接配置（R2 会话）：符号提供者明确在 CANN `libopapi.so`，且 add_example（官方参考实现，链接配置一致）同样报错——给 `libcust_opapi.so` 强行静态补符号既无必要也无先例；保持与模板一致的链接配置更安全。对照把"我的代码有问题"和"校验器行为"区分开，是本判别规则的关键一步。
- 路线三的 --force 纪律（R5 会话）：只对"符号确实由 CANN 运行时提供"的误报场景使用；用探针确认符号来源 + 安装后真机运行验证闭环。盲目 --force 会把真链接缺陷装进环境。op_api 构建由 npu_op_package 宏自动托管（无手写 CMakeLists 可改）时尤其如此。
- R4 为什么不用 LD_PRELOAD 常驻解决：会破坏 GE 图模式 TBE 编译初始化（成对负结果见 LD_PRELOAD 卡）；推荐形态是 LD_LIBRARY_PATH + 桥接库，一次运行内同时需要 torch 测试与 geir 时尤其注意先 `unset LD_PRELOAD`。
- 路线二为什么链到 `${package_name}_ascendc_cust_opapi` 而不是 `cust_opapi`：OBJECT 库的链接不传播到最终共享库；最终 `libcust_opapi.so` 由 `npu_op_package_add` 创建的 SHARED 目标组装，链接必须加在它上面。为什么用 `find_library` 而不是写死路径：libopapi.so 在 `$ASCEND_HOME_PATH/lib64` 与 `x86_64-linux/lib64` 均有分布，find_library + NO_DEFAULT_PATH 保证取 CANN 安装内版本且找不到时显式 WARNING。

## 系统信息

- 工具链：CANN 9.1.0（安装校验、--extract、torch dlopen 会话）；CANN 9.2.0（链接修复会话，`func.cmake` 1339-1357 行链接逻辑）；安装器 custom_opp run 包。
- 相关文件：`op_api/aclnn_*.cpp`（l0op:: 调用点）、`op_host/CMakeLists.txt`（npu_op_package_add，路线二修复位置）、CANN `include/aclnn_kernels/contiguous.h`（声明）、`lib64/libopapi.so`（符号定义）、`opp/vendors/config.ini`（load_priority 登记）。

## 验证信号

验证来自同一开发会话：删除调用后重新 `bash build.sh --soc=ascend950 -j8` 并安装，`Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created` 且安装流程不再报 undefined symbol，后续 `--install-path=... --force` 安装 SUCCESS、算子可正常调用（来源平台：950；证据等级：tool_observed）。

R2（CANN 9.1.0，独立会话）：

- 自研包 --extract 校验：`[ERROR] Failed to load libcust_opapi.so ... undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`，同时解包产物齐全（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 符号提供者：`nm -D` 命中 `l0op.*Contiguous` 15 处于 `libopapi.so`，`libnnopbase.so`/`libascendcl.so` 均为 0（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 参考包对照：add_example 包 --extract 校验输出与自研包逐字相同（同样的 undefined symbol 与 `--force` 提示）（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。

R3（CANN 9.2.0，独立会话，链接修复）：

- 修复前：`custom_opp_ubuntu_x86_64.run --quiet` 安装即报 `undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`，校验失败（来源平台：950；证据等级：tool_observed）。
- 机制锚点：`nm -D libopapi.so` 见 `T Contiguous/T ViewCopy` 导出；`ldd libcust_opapi.so` 无 libopapi.so、`nm -D` 见两个 `U _ZN4l0op…` 符号；func.cmake 1353 行链接清单无 opapi（来源平台：950；证据等级：tool_observed）。
- 修复后：编译日志输出 `-- Linked libopapi.so for l0op::Contiguous/ViewCopy`；打包产物 `ldd` 显示 `libopapi.so => …/lib64/libopapi.so`（`nm -D` 的 `U` 残留属动态链接正常解析态）（来源平台：950；证据等级：tool_observed）。
- 修复后：重装 `.run` 无 `--force` 即通过 "validate shared libraries before installation" 并 SUCCESS，随后 ST 测试可在 NPU 上执行（来源平台：950；证据等级：tool_observed）。

R4（CANN 9.1.0，独立会话，torch 侧）：

- 安装失败复现：`.run` 安装两次报 "Failed to load libcust_opapi.so ... undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor" + "Shared library validation failed"（来源平台：950；证据等级：tool_observed）。
- 符号定位探针：`nm -D` 全库扫描命中 libopapi.so / _cv / _math / _nn / _oam / _transformer（lib64 与 x86_64-linux/lib64 双目录）（来源平台：950；证据等级：tool_observed）。
- dlopen 失败复现：LD_LIBRARY_PATH 仅含 vendor op_api/lib 时 `torch.ops.load_library(libcust_opapi.so)` 抛 OSError 同一符号（来源平台：950；证据等级：tool_observed）。
- 处置后：LD_LIBRARY_PATH 纳入 CANN lib64 后加载 `libtorch_adapter.so`，PyTorch ST 795/795 PASS、通过率 100%（来源平台：950；证据等级：tool_observed）。
- `--force` 安装后回归：UT 17/17、ST Real 876/876、geir 双模式 PASS（来源平台：950；证据等级：tool_observed）。

R5（CANN 9.1.0，独立会话，--force + 真机闭环）：

- 修复前：`./custom_opp_ubuntu_x86_64.run --quiet` 安装失败，报同一 undefined symbol（来源平台：950；证据等级：tool_observed）。
- 处置：`--quiet --force` 安装成功（`skip shared library validation` → `SUCCESS`），`vendors/config.ini` 的 load_priority 首位登记本 vendor（来源平台：950；证据等级：tool_observed）。
- 运行时验证（关键闭环）：安装后 ST Real 模式在 Ascend950PR 真机运行，3 条用例全 PASS，MERE=0.00e+00、MARE=0.00e+00——aclnn 调用链中 l0op:: 符号在运行时全部正常解析，证明安装期报错是校验环境缺依赖的误报（来源平台：950；证据等级：tool_observed）。
- 工作区对照：其余已成功安装算子的 op_api 均不使用 l0op::Contiguous（grep 仅本算子命中），无既有先例可参照（来源平台：950；证据等级：tool_observed）。

并行策展批（CANN 9.1.0，Ascend950PR，五个独立 trace，op_api 均引用 `l0op::Contiguous`/`l0op::ViewCopy`）：

- 五个独立开发会话安装同类算子包时报同一原文 `Failed to load libcust_opapi.so ... undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`，其中两个会话各复现两次（来源平台：950；证据等级：tool_observed）。
- `nm -D` 全库检索：符号 `T` 定义只在 `libopapi.so`（0xb04a90）与 `libopapi_math.so`（0x833500）命中，`libnnopbase.so`/`libascendcl.so` 无，`libopapi_oam/nn/cv/transformer.so` 为 `U` 引用方（来源平台：950；证据等级：tool_observed）。
- 参照系对照：同机已安装且正常运行过的算子（sigmoid 类），其 `libcust_opapi.so` 用 `nm -D` 看到同样的 `U` 符号、`ldd` 同样无 libopapi 依赖——"U 引用 + 提供方在运行时环境"是该工程模板的正常形态而非缺陷（来源平台：950；证据等级：tool_observed）。
- `--force` 安装 SUCCESS 后运行期符号解析正常（多组上板证据）：ST Real 全量通过（12/12、16/16 与 29/29）、GE IR 图模式 Session run success 且精度 OK、PyTorch 适配 36/36 通过（来源平台：950；证据等级：tool_observed）。
- 根治链接方案验证：在 `op_host/CMakeLists.txt` 对 `<pkg>_ascendc_cust_opapi` 目标 `target_link_libraries(... PRIVATE opapi)` 后，`link.txt` 链接命令出现 `-lopapi`，安装通过 "upgrade op api" 共享库校验、无需 --force（来源平台：950；证据等级：tool_observed）。

并行策展批（CANN 9.1.0，Ascend950PR，三个独立 trace：GroupNorm/Conv2d/AdaptiveAvgPool3d 类算子，op_api 以 `l0op::Contiguous`/`l0op::ViewCopy` 组装两段式或含 L0 API）：

- 触发：不带参数安装必在 "validate shared libraries" 阶段 Failed，三条轨迹逐字复现同一报错（含同一工程两次独立安装会话）（来源平台：950；证据等级：tool_observed）。
- 判别链：`nm -D libcust_opapi.so` 中 `_ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor` 为 `U`（另见 `U _ZN4l0op8ViewCopy...`）；`nm -D libopapi.so` 同符号为 `T`（另有 `l0op::OptimizeView/CanOptimizeView` 等同级定义）；`ldd` 测试二进制同时解析 libascendcl/libopapi/libnnopbase/libcust_opapi 四个库——孤立加载伪影判定成立（来源平台：950；证据等级：tool_observed）。
- 处置：`--force` 安装 SUCCESS（完整走完 framework/op_proto/op_impl/op_api/uninstall/version.info 升级流程；`[WARNING] --force is enabled, skip shared library validation.`）（来源平台：950；证据等级：tool_observed）。
- 运行期反证：--force 安装后 ST 实际跑到参数校验与 kernel 执行阶段（aclnn 返回码来自算子自身校验逻辑，无任何 "symbol lookup error"）；GE IR 图模式示例 `Session run ir compute graph success` 且输出正确、aclnn eager 示例输出正确；按提示设置 LD_LIBRARY_PATH 后 ST Real（NPU 直调 aclnn）L0 fp32 37/37 通过、后续全量 L1 逐 dtype 全部通过（来源平台：950；证据等级：tool_observed）。
误诊补充：一条轨迹曾把该校验失败与 `aclnn GetWorkspaceSize 返回 1` 因果关联（返回 1 的真实根因是算子自身校验函数返回类型问题），修复并 --force 重装后返回码行为立即改变，符号问题与返回码无因果；另一轨迹曾按报错怀疑 CMake 链接配置，排查后确认库本身无误。

## 失效边界

- 路线一仅适用于"OpDef 已 AutoContiguous（或输入本就连续）"的场景；确需显式连续化的算子不适用。
- 安装器校验行为可能随 CANN 版本变化（是否预加载 opapi 库），未验证，推断可迁移；若后续 CANN 版本把校验环境补全，路线三的误报判别自然失效（无害）。
- 路线二结论以 CANN 9.2.0 的 func.cmake 链接清单为准；后续版本若官方补上 opapi 链接则该路线失效（未验证）。只覆盖 `l0op::` 前缀的 aclnn_kernels 辅助符号；op_api 引用其他 CANN 动态库符号需按同样思路逐个核对链接清单。
- 路线三 `--force` 跳过的是整包共享库校验，不是只放行 l0op 符号：若包内还有其他真实链接缺陷，会一并放行——使用前必须完成符号来源判别，使用后必须真机运行验证。
- `--force` 安装的另一使用场景：安装器校验因本符号拒绝时，`--force` 是构建循环的固定一步（R4 会话工作流）。
- `l0op::Contiguous` 在运行期语义上另有已闭合的 561103 根因（同名算子签名冲突时内置版对特定参数返回 nullptr），见 [561103 同名算子签名冲突卡](561103.md)——与本卡的安装期链接边界是不同问题域。
- R4 "CANN lib64 加入 LD_LIBRARY_PATH"与"LD_PRELOAD=libopapi.so"在同一次成功运行中同时启用，未做单变量隔离——推荐组合（LD_LIBRARY_PATH + 桥接库）已整体验证。

**未决项**：

- 安装器校验环境具体的 LD_LIBRARY_PATH 组成未进一步核查（不影响处置结论）。
- R4/R5 优先项：单变量隔离验证"仅 LD_LIBRARY_PATH 含 CANN lib64（不 preload）"能否独立支撑 dlopen libcust_opapi.so；`libcust_opapi.so` 对 `libopapi.so` 是否有 DT_NEEDED 显式依赖（`readelf -d` 未查），影响"为什么必须环境变量补位"的机制解释。
- R5：用 `nm -D $ASCEND_HOME_PATH/lib64/libopapi.so | grep _ZN4l0op` 直接核对 l0op 符号的宿主库与符号版本（该会话未执行，机制归因由 R2/R3/R4 的 nm 探针补齐）。
- func.cmake 为什么不给 ACLNN 类型算子默认链接 opapi（是否所有用 l0op:: 的官方算子都自带了这段补链 CMake）未核实，属工具链侧机制问题。
## 坏实践

- 用 `--force` 跳过安装校验来"解决"undefined symbol 而不做符号来源判别——校验失败说明 .so 的符号解析链有问题，真链接缺陷强装后风险后移。
- op_api 层照抄示例代码里的 `l0op::Contiguous` 而不核对自身 OpDef 是否已声明 AutoContiguous——两者职责重叠。
- 见报错就改 CMakeLists 给 libcust_opapi 补链接或删掉 l0op 调用而不先做探针（R2 会话：官方参考包同样报错，说明是校验环境而非工程缺陷）。
- `--lib` 直接指向 `libcust_opapi.so`（不注册 torch.ops，报 `no attribute 'forward'` 后误以为算子没装好）。
- 为图省事全局 `export LD_PRELOAD=libopapi.so` 再去跑图模式示例——会引入 TBE 初始化失败（见 LD_PRELOAD 卡）。
- 运行 ST/UT 前的 `LD_LIBRARY_PATH` 只含 vendor `op_api/lib` 一段（安装器完成提示里只给了前者，不够；须同时含 CANN lib64）。

## 误诊

- 无（nm -D 探针直接定位到符号提供方，未走弯路）。
- R2 初判"是我的链接配置漏了库"：`npu_op_library(cust_opapi ACLNN ...)` 的链接目标与模板完全一致，且参考包同样报错，排除。
- R5 初见报错时曾按"编译期链接缺陷"思路排查（找 op_api 的 CMakeLists 想补链接 libopapi.so）；随后确认 op_api 构建由 npu_op_package 自动托管、且安装器错误信息明确是"load .so 校验"阶段（非编译阶段），才转向"校验环境依赖缺失"的正确归因。
- R3 曾把 ST 全用例返回 161002 归因于该未定义符号：补链重装后 161002 依旧，最终 plog 证实 161002 是 `ACLNN_ERR_PARAM_INVALID`，由 op_api CheckAttrs 参数传递 bug 引起。补链本身仍是必要的（否则安装都过不去），但不是 161002 的因。

## 后续增量（2026-09-16，并行策展批：五个 9.1.0 trace 复核与根 CMakeLists 时序坑）

并行策展批五个独立 trace（CANN 9.1.0 / Ascend950PR）对上文三条路线做了交叉复核，补充三点：

**1. NEEDED 依赖机制（部分回答 R4/R5 遗留的 DT_NEEDED 问题）**：安装脚本的 "validate shared libraries before installation" 步骤会**独立 dlopen** 待装的 `libcust_opapi.so` 并要求全部符号当场可解析；`l0op::Contiguous` 的定义在 CANN `lib64` 的 `libopapi.so`/`libopapi_math.so` 中，且它**不是** `libcust_opapi.so` 的 NEEDED 依赖（`ldd` 无 libopapi 条目；链接时允许未定义、运行时延迟绑定）。独立 dlopen 场景下没有其他库先加载，全局符号表里没有该定义，于是校验误报"undefined symbol"。真实运行时进程（aclnn 测试程序）里 `libopapi.so` 已在 `LD_LIBRARY_PATH` 的 CANN `lib64` 中被加载，符号经全局符号表正常解析——四个 trace 的 `--force` 安装后运行期全量验证（ST 12/12、16/16+29/29、GE IR、PyTorch 36/36）佐证。

**2. 根 CMakeLists 时序坑（路线二的落点补充）**：在**根 CMakeLists.txt** 的 `npu_op_package(...)` 之后写 `if(TARGET ...) target_link_libraries(...)` 不会生效——`npu_op_package` 宏在根文件展开时 `cust_opapi` 共享库目标尚未创建，`if(TARGET)` 为假**静默跳过**，重配后 `link.txt` 仍无 `-lopapi`。正确位置是 `op_host/CMakeLists.txt` 的 `npu_op_package_add(...)` 之后（与上文路线二一致）；验证方式：重新构建后查看 `build/op_host/CMakeFiles/<package_name>_ascendc_cust_opapi.dir/link.txt`，链接命令末尾应出现 `-lopapi`，再安装应通过 "validate shared libraries" 校验（实测 `upgrade op api` 通过、无需 --force）。交叉编译（x86_64 主机编 aarch64 目标）时若手写库路径，须链目标架构目录（aarch64-linux/lib64）下的库，不能只看到 x86_64 目录的 libopapi.so 就链它。

**3. `--force` 的前提纪律与上文路线三一致**：只对"nm 确认符号由 CANN 公共库提供"的误报场景使用；采用前对 libcust_opapi.so 做一次 `nm -D`/`ldd -r` 自查，采用后立即真机运行验证。op_host 直连 libruntime 内部符号（rtSetDevice 等）导致的拒装不属于本误报场景，`--force` 会掩盖真缺陷（另见 [op_host 直连 libruntime 内部符号致安装拒装](op_host_extern_c_libruntime_undefined_symbol.md)）。

## 后续增量（2026-09-16，并行策展批：外部 C++ ST 直调的链接四件套）

两个独立会话（CANN 9.2.0，Ascend950PR：GRU 类手写 aclnn 算子 + 稀疏 attention 算子的外部 C++ ST 工程）补充**消费方（测试/示例程序）侧**的链接契约——上文路线覆盖安装与运行期路径，本增量补"进程内直调 aclnn 两段式接口的测试程序怎么链"：

直调自定义 aclnn 算子的测试程序需要三类符号：① 算子自己的 `aclnn<Op>` 符号——在**安装后的 vendor 包** `opp/vendors/<vendor>/op_api/lib/libcust_opapi.so` 里（注意 vendor 库的真实名字就是 `libcust_opapi.so`，不存在 `libaclnn_<op>.so`，`-laclnn_<opname>` 会报 cannot find）；② CANN aclnn 框架符号（`aclDestroyAclOpExecutor`、`aclCreateTensor` 等）在 `libnnopbase.so`，不显式链接时报 `undefined reference to symbol 'aclDestroyAclOpExecutor'` + `libnnopbase.so: error adding symbols: DSO missing from command line`；③ `l0op::Contiguous` 等辅助符号在 `libopapi.so`。只链 CANN 公共库 `ascendcl opapi` 必然缺符号/缺运行时路径。

CMake Real 模式段落（修复后，verbatim）：

```cmake
if(USE_MOCK)
    target_compile_definitions(test_aclnn_<op> PRIVATE USE_MOCK)
    message(STATUS ">>> Building in MOCK mode (CPU golden only, no NPU)")
else()
    # Real 模式：CANN include + 算子 op_api 头文件
    target_include_directories(test_aclnn_<op> PRIVATE
        ${CANN_ARCH_DIR}/include
        ${OP_ROOT}/op_api
    )
    # 算子 vendor 库路径（libcust_opapi.so 含 aclnn 算子符号）
    set(VENDOR_NAME "<op>_custom")
    set(VENDOR_LIB_DIR "${CANN_HOME}/opp/vendors/${VENDOR_NAME}/op_api/lib")
    target_link_directories(test_aclnn_<op> PRIVATE
        ${CANN_ARCH_DIR}/lib64
        ${CANN_HOME}/lib64
        ${VENDOR_LIB_DIR}
    )
    target_link_libraries(test_aclnn_<op> PRIVATE
        cust_opapi ascendcl opapi nnopbase
    )
endif()
```

g++ 直链形态（诊断程序）：

```bash
g++ -std=c++17 -o /tmp/test_debug /tmp/test_debug.cpp \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include \
    -L$ASCEND_HOME_PATH/lib64 \
    -L$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib \
    -lascendcl -lcust_opapi -lnnopbase \
    -Wl,-rpath,$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib
```

run.sh 侧：安装用 `bash <run包> --force`（跳过共享库校验），运行前 `export LD_LIBRARY_PATH=<vendor>/op_api/lib:$LD_LIBRARY_PATH`（与路线三的两段路径一致）。vendor 目录名 = `<算子名>_custom` 由打包命名决定，换算子要同步改 `VENDOR_NAME`；更稳健做法是 glob `opp/vendors/*/op_api/lib`。

验证信号（来源平台：950；证据等级：tool_observed）：

- 链接修复后 `[100%] Built target test_aclnn_<op>`；ST 诊断程序在补链 `-lnnopbase` 与 libopapi 后链接通过、进入执行。
- `--force` 安装后 vendor 目录出现 `libcust_opapi.so`、`libcust_opsproto_rt2.0.so`、`liboptiling.so`、`libcust_opmaster_rt2.0.so` 及 `op_api/include/aclnn_*.h`。
- Real 模式 NPU 实测：GetWorkspaceSize status=0、execute status=0，两个小用例 PASS=2 FAIL=0（MERE=1.78e-04/0）。
- 误诊补充：初见安装校验失败曾怀疑算子 op_api 构建缺库（真链接缺陷）；经 `nm -D libopapi.so` 确认 `l0op::Contiguous` 已定义、且运行期 LD_LIBRARY_PATH 含 libopapi 所在目录，判定为安装器校验环境限制——`--force` 后 Real 模式全链路跑通佐证（与上文 R2/R5 的"孤立加载伪影"结论一致）。


## 后续增量（2026-09-16，并行策展批：六个 9.1.0 trace 复现与 cust_opapi 直链变体）

**1. 路线二的 9.1.0 模板变体（链接目标名随工程结构而异）**：grid_sampler_3d 类工程（CANN 9.1.0 / x86_64 宿主机）在 `op_host/CMakeLists.txt` 对 `cust_opapi` 目标直接补链即通过校验——与上文 `${package_name}_ascendc_cust_opapi` 落点是同一思路的两种模板形态；采用时以构建后 `link.txt` 出现 `libopapi` 为准核对链接目标选对。before/after（verbatim）：

```cmake
# 修复前（参考示例模板没有给 cust_opapi 链接 libopapi.so）
target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

# 修复后
target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

# 链接 ACLNN 内核库（提供 l0op::Contiguous, l0op::ViewCopy 等）
target_link_libraries(cust_opapi PRIVATE
    ${ARCH_PATH}/lib64/libopapi.so
)
```

（`${ARCH_PATH}` 即 `$ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux`。）

**2. 六个独立 9.1.0 trace（resize_bilinear 类、gcd、gather、grid_sampler_3d、argmax、AdamW 类）的复现与运行兜底**：

- 默认安装（`--quiet` 或直接执行）稳定报 `Failed to load libcust_opapi.so ... undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor` 并 `Shared library validation failed`，重新打包重装每次复现（来源平台：950；证据等级：tool_observed）。
- nm/readelf 探针：`l0op::Contiguous` 在 `libopapi.so` 为 T 定义（该 9.1.0 安装为 0xb38d60，`libopapi_math.so` 亦有 T），包内 `libcust_opapi.so` 为 U 引用且 `readelf -d` NEEDED 列表只有 `libnnopbase.so`/`libstdc++.so.6`/`libgcc_s.so.1`/`libc.so.6`，`libnnopbase.so` 无 libopapi 传递依赖（来源平台：950；证据等级：tool_observed）。
- `--quiet --force` 安装 SUCCESS 后运行期符号解析正常（多组上板兜底）：GE IR 与 ACLNN 示例 PASS、ST 全量 23/23；另一工程 ST 基础级 24/24、扩展级 503/503；另一工程 UT 27/27、ST 全量真实下发 NPU 执行正常；PyTorch 接入测试全量 92/92 真机通过（含调用 Contiguous 路径的 aclnn 入口）（来源平台：950；证据等级：tool_observed）。
- 上文变体 1 的链接修复实测：`op_host/CMakeLists.txt` 补链 `${ARCH_PATH}/lib64/libopapi.so` 后重新构建，`./build/custom_opp_ubuntu_x86_64.run` 不带 `--force` 直接输出 `upgrade op api ... SUCCESS`，安装校验通过（来源平台：950；证据等级：tool_observed）。
- 运行期加载顺序旁证：对链接了 `libascendcl` 的测试程序 `ldd` 可见 `libopapi.so` 先于 `libcust_opapi.so` 加载，符号可正常解析（来源平台：950；证据等级：tool_observed）。

**3. 误诊补充**：历史安装均带 `--force` 会掩盖校验错误，让人误以为"本次构建引入了新问题"——对比两次安装行为时先核对命令行参数是否一致；"符号来自其他已安装算子包，装上那个包就能过"——提供者实为 CANN 自身 libopapi.so，与多算子包共存无关（由运行时单包环境实测通过反证）。