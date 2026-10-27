---
type: "Runbook"
title: "aclnn_exclude 头文件自动生成覆盖源码致签名不符"
description: "自定义算子的 vendors 安装头由 op_def 自动生成，与源码 op_api 实现签名可能分叉（多出属性参数、把 aclScalar* 标量展开为原生 double、或属性类型 String→Int 改动只改 _def.cpp 不改源码头），导致调用方编译失败；迁就安装头改参数会把编译错变成 NPU 运行期 ABI 错位段错误。通过对比源码声明、实现、安装头、CMake 命中路径与 nm 符号定位；调用方 include 顺序把源码 op_api 排首位；属性类型改动必须全链同步（def/proto/源码头/L2/L0/tiling/UT/ST）。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-18T05:31:13Z"
updated_at: "2026-09-17T04:15:00Z"
platforms: ["950"]
---

# aclnn_exclude 头文件自动生成覆盖源码致签名不符

## 现象

ReduceStdV2Update 算子 ST 测试编译报 `aclnnReduceStdV2UpdateGetWorkspaceSize` 签名不符：测试代码按 8 参数调用，但编译器报参数个数不匹配。核查发现源码 `op_api/aclnn_reduce_std_v2_update.h` 是手写 8 参数版本（与 `op_api/aclnn_reduce_std_v2_update.cpp` 实现一致），而安装到 vendors 下的同名头文件是 10 参数版本（多了 `ifStd` 和 `correction`）。ST CMakeLists 的 `find_path` 命中 vendors 下的错误版本，导致按 10 参数签名编译与 8 参数调用不匹配。

R2（CANN 9.1.0，独立会话，标量参数变体）：为已安装自定义算子 `bench_no_harness_add_rms_norm_dynamic_quant` 开发 PyTorch 适配层 `torch_adapter.cpp` 时连续遇到两个现象。其一，适配层按源码风格创建 `aclScalar* epsilon_acl` 传入 `aclnnBenchNoHarnessAddRmsNormDynamicQuantGetWorkspaceSize(...)`，编译直接失败：

```
torch_adapter.cpp:182:52: error: cannot convert 'aclScalar*' to 'double'
  182 |         ws->x1_acl, ws->x2_acl, ws->gamma_acl, ws->epsilon_acl,
In file included from torch_adapter.cpp:46:
<CANN安装根>/opp/vendors/<op>_custom/op_api/include/aclnn_bench_no_harness_add_rms_norm_dynamic_quant.h:32:12: note:
   initializing argument 4 of 'aclnnStatus aclnnBenchNoHarnessAddRmsNormDynamicQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, const aclTensor*, const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**)'
```

其二，按安装头文件改成 `double epsilon` 后编译通过、CPU golden 自测 8/8 PASS，但一上 NPU 跑第一个测试用例就崩溃：

```
timeout: the monitored command dumped core
/bin/bash: line 1: ... Segmentation fault      timeout 300 python3 -u test.py --lib ./build/libtorch_adapter.so --level L0
```

当时代码状态：适配层 include 的是安装目录里的自动生成头文件（CMakeLists 的 include 顺序只有指向 `opp/vendors/<op>_custom/op_api/include` 的安装目录），链接的是该 vendor 包 `op_api/lib/libcust_opapi.so`。同一套 C++ ST 测试用 Real 模式跑同一算子却完全正常——差异只在 ST 工程的 include 顺序里多了源码 `op_api/` 目录且排在最前。

## 适用场景

- **工程形态**：算子构建系统（build.sh + CPack 打包）+ ST 测试 CMake 编译；以及自定义算子出包安装后的任何调用方开发——PyTorch 适配层、独立 aclnn 示例、任何 `#include "aclnn_<op>.h"` 的 host 侧程序。
- **API 路径**：`op_host/*_def.cpp` 的 `OP_ATTR` 声明 → 构建系统自动生成 aclnn 头文件 → CPack 打包到 vendors → ST/调用方 CMake `find_path`/include 顺序引用 vendors 头文件。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510 / arch35），CANN 9.2.0（属性参数变体）与 CANN 9.1.0（标量类型变体）。
- **适用粒度**：`platform_wide`（CANN 自定义算子 aclnn 头自动生成机制跨平台通用）。
- **路线优先级**：属性参数变体为 `diagnostic_only`（定位生成物错配）；标量类型变体为 `recommended_fastpath`（源码 op_api 目录排 include 首位 + 按源码签名传参是正确性必需）。
- **明确不适用或未验证场景**：A3/A2 未验证；源码与安装头一致（无签名分叉）的算子；GE 图模式调用路径（未涉及）。

## 判别方法

1. **触发信号**：ST 编译报 `aclnnXxxGetWorkspaceSize` 签名不符——测试代码按 8 参数调用，编译器报参数个数不匹配；或 `cannot convert 'aclScalar*' to 'double'`（及反向）。属系统性签名错配，非单 case 问题。
2. **可执行探针**：`diff` 源码 `operators/<op>/op_api/aclnn_*.h` 与 vendors 安装版 `${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/include/aclnn_*.h` 的 `GetWorkspaceSize` 签名；核查 `op_host/*_def.cpp` 的 `OP_ATTR` 声明数量；核查 ST CMakeLists `find_path` 的搜索路径顺序。
3. **原始观测**：源码头文件为 8 参数（verbatim 取自 trace reasoning 对源码 `op_api/aclnn_reduce_std_v2_update.h` 的引用）：
   ```c
   aclnnStatus aclnnReduceStdV2UpdateGetWorkspaceSize(
       const aclTensor* self,
       const aclTensor* mean,
       const aclIntArray* dim,
       bool unbiased,
       bool keepdim,
       aclTensor* output,
       uint64_t* workspaceSize,
       aclOpExecutor** executor);
   ```
   vendors 安装版为 10 参数（自动生成，verbatim 取自 trace reasoning 对 vendors 头文件的引用）：
   ```c
   aclnnStatus aclnnReduceStdV2UpdateGetWorkspaceSize(
       const aclTensor *x,
       const aclTensor *mean,
       const aclIntArray *dim,
       bool ifStd,           // 多了 ifStd
       bool unbiased,
       bool keepdim,
       int64_t correction,   // 多了 correction
       const aclTensor *out,
       uint64_t *workspaceSize,
       aclOpExecutor **executor);
   ```
   `op_host/reduce_std_v2_update_def.cpp` 声明 5 个 `OP_ATTR`（dim/if_std/unbiased/keepdim/correction）。ST CMakeLists `find_path` 查找 `aclnn_reduce_std_v2_update.h` 命中 vendors 下错误版本。
4. **解释**：本例中源码声明与实现采用 8 参数接口，vendors 安装头采用由 op_def 属性展开出的 10 参数接口，而 ST 实际命中后者。由此可确认编译失败来自接口生成物与实现契约不一致；应以源码声明、实现和实际导出符号的联合核对为准，不能仅凭任一份头文件推断通用生成机制。

R2 探针（标量类型变体，本会话走通的链路，可复用）：

1. 编译期看到 `cannot convert 'aclScalar*' to 'double'`（或反向）时，先 `cat $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include/aclnn_<op>.h`，再对照源码 `operators/<op>/op_api/aclnn_<op>.h` 与 `aclnn_<op>.cpp` 的实际参数类型——两边签名分叉即是本坑。安装头的文件头注释为 "this file was generated automaticlly donot change it"，include 的是 `aclnn/acl_meta.h` 而非源码头的 `aclnn/aclnn_base.h`。
2. 已按安装头（原生类型）编译通过的程序上 NPU 段错误时，用 `nm -C $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so | grep GetWorkspaceSize` 确认导出符号来自源码实现，再读源码 `.cpp` 的形参类型定 ABI。
3. 对照一个能正常运行的调用方工程（如 C++ ST）的 CMakeLists include 顺序，看它把哪个 `op_api` 目录排在前面。

## 根因/解决配方（R2 标量类型变体）

安装包里的自动生成 aclnn 头文件与实际编译进 `libcust_opapi.so` 的源码实现签名不一致：源码 `op_api/aclnn_<op>.cpp` 第 4 参为 `const aclScalar* epsilon`（库由源码编译，`nm -C libcust_opapi.so` 可见导出符号），而安装头文件声明为 `double epsilon`。

机制链（按 x86_64 System V ABI 展开）：

1. 调用方按安装头编译：`double epsilon` 作为第 4 参走 **XMM0** 浮点寄存器，后续输出参数依次走 RCX/R8/R9。
2. 库实现按源码编译：`const aclScalar* epsilon` 作为指针走 **RCX**，后续指针参数整体后移一位、最后一个参数落到栈上。
3. 两边寄存器约定完全错位：库里把 RCX 里的输出指针当 `epsilon` 解引用、把垃圾值当输出指针写——首次下发即段错误；编译期只报一个"类型不匹配"，一旦迁就安装头改成原生类型，错误就从编译期转移到运行期崩溃。
4. C++ ST 工程能正常跑，是因为其 CMakeLists 把源码 `op_api/` 目录列在 include 首位，编译器解析到的是与库一致的 `const aclScalar*` 头文件。

修复（include 顺序 + 按源码签名传参）：

```cmake
# 修复前：include 顺序只有安装目录，安装版自动生成头文件（double epsilon）被解析
target_include_directories(torch_adapter PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}          # 安装版自动生成头文件在这里被解析
)

# 修复后：源码 op_api 目录排 include 首位（与库一致的头文件优先解析）
target_include_directories(torch_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api   # 源码头文件（const aclScalar*）优先解析
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)
```

```cpp
// 修复态调用（与源码/库签名一致的 aclScalar* 传参，RAII 管理）
struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x1_acl = nullptr;
    aclTensor* x2_acl = nullptr;
    aclTensor* gamma_acl = nullptr;
    aclScalar* epsilon_acl = nullptr;
    aclTensor* y_acl = nullptr;
    aclTensor* xOut_acl = nullptr;
    aclTensor* scaleOut_acl = nullptr;

    ~OpWorkspace() {
        if (x1_acl) { aclDestroyTensor(x1_acl); x1_acl = nullptr; }
        if (x2_acl) { aclDestroyTensor(x2_acl); x2_acl = nullptr; }
        if (gamma_acl) { aclDestroyTensor(gamma_acl); gamma_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        if (xOut_acl) { aclDestroyTensor(xOut_acl); xOut_acl = nullptr; }
        if (scaleOut_acl) { aclDestroyTensor(scaleOut_acl); scaleOut_acl = nullptr; }
        if (epsilon_acl) { aclDestroyScalar(epsilon_acl); epsilon_acl = nullptr; }
        executor = nullptr;  // 由 CommonOpExecutorRun 内部析构
    }
};

// 调用处：epsilon 用 aclCreateScalar 创建后按 aclScalar* 传递，与源码签名一致
ws->epsilon_acl = aclCreateScalar(&epsilon, ACL_FLOAT);
aclnnStatus status = aclnnBenchNoHarnessAddRmsNormDynamicQuantGetWorkspaceSize(
    ws->x1_acl, ws->x2_acl, ws->gamma_acl, ws->epsilon_acl,
    ws->y_acl, ws->xOut_acl, ws->scaleOut_acl,
    &ws->workspace_size, &ws->executor);
```

## 设计决策

- R2 选择「源码 op_api 目录排 include 首位」而不是给函数写前向声明：与库一致的头文件只有源码侧一份，前向声明会在参数再变化时再次失配；include 优先级方案正是同一算子 C++ ST 工程已在用且实测可跑的做法。
- R2 选择恢复 `aclCreateScalar(&epsilon, ACL_FLOAT)` + `aclDestroyScalar` 的 RAII 管理（OpWorkspace 析构里统一释放），而不是裸传栈上 Scalar：与 aclTensor 生命周期管理对称，避免 executor 复用期间悬垂。
- 「换原生类型签名迁就安装头」已被实测证伪（编译通过但 NPU 段错误）。

## 系统信息

- 工具链：CANN 9.2.0（属性参数变体）/ CANN 9.1.0（标量类型变体，Ascend950PR 单卡，torch 2.7.1+cpu + torch_npu 2.7.1.post4）；安装器 custom_opp run 包。
- 相关文件：`op_api/aclnn_<op>.cpp`（实现）、`operators/<op>/op_api/aclnn_<op>.h`（源码头）、`$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include/aclnn_<op>.h`（安装头）、`$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so`（库）、调用方 CMakeLists（include 顺序）。

## 预防

- 自定义算子走 aclnn_exclude 路径时，在打包验收中同时核对源码声明、实现签名、vendors 安装头和最终导出符号，确保四者一致。
- 联调阶段遇到"aclnn 接口签名不符"且测试代码与源码实现一致时，优先怀疑 vendors 头文件是自动生成版本，diff 源码与 vendors 即可定位。
- 为已安装自定义算子写任何 aclnn 调用方时，CMakeLists 一律把源码 `op_api/` 目录放在安装 vendor include 之前；不要相信安装包自动生成的 aclnn 头文件签名。
- 遇到「编译报 aclScalar*/原生类型不匹配」，禁止直接迁就安装头改参数类型——先核对 `.so` 的真实实现签名（源码 .cpp / nm）。
- 标量参数一律 `aclCreateScalar` 创建、用完 `aclDestroyScalar`，与 aclTensor 同等对待。

## 验证信号

- 源码 `op_api/aclnn_reduce_std_v2_update.h` 的 `aclnnReduceStdV2UpdateGetWorkspaceSize` 为 8 参数（无 ifStd/correction）（来源平台：950；证据等级：tool_observed）。
- vendors 安装版 `aclnn_reduce_std_v2_update.h` 为 10 参数（多 ifStd、correction），与源码不一致（来源平台：950；证据等级：tool_observed）。
- `op_host/reduce_std_v2_update_def.cpp` 声明 5 个 `OP_ATTR`（dim/if_std/unbiased/keepdim/correction），是自动生成 10 参数头文件的属性来源（来源平台：950；证据等级：tool_observed）。
- ST CMakeLists `find_path` 查找 `aclnn_reduce_std_v2_update.h` 命中 vendors 下的错误版本（来源平台：950；证据等级：tool_observed）。

R2（CANN 9.1.0，独立会话，标量类型变体）：

- 修复前：适配层编译失败 `error: cannot convert 'aclScalar*' to 'double'`（安装头文件第 4 参声明为 `double epsilon`）（来源平台：950；证据等级：tool_observed）。
- 修复前（误诊态）：改按 double 传参后编译通过、CPU golden 自测 8/8 PASS，但 NPU 首个用例 `Segmentation fault ... dumped core`（来源平台：950；证据等级：tool_observed）。
- 定位探针：`nm -C libcust_opapi.so` 显示 `aclnnBenchNoHarnessAddRmsNormDynamicQuantGetWorkspaceSize` 由源码实现导出；源码 `aclnn_<op>.cpp` 使用 `const aclScalar* epsilon`，与安装头 `double epsilon` 分叉（来源平台：950；证据等级：tool_observed）。
- 修复后：重建适配层，NPU 上基础层用例 11/11 PASS（fp16/bf16、单 token、ColSplit N=16384 等用例 y(int8) max_err≤1、xOut/scaleOut MERE/MARE 达标）（来源平台：950；证据等级：tool_observed）。
- 修复后：进阶层 60/62 PASS，仅 2 条 2.1B 元素用例因 >200M 内存守卫主动跳过，失败 0；最终精度验收复跑 73 条通过 71、跳过 2、失败 0（fp16 MERE≈7.5e-05 / MARE≈9.77e-04，bf16 MERE≈6.1e-04 / MARE≈7.81e-03，y(int8) max_err≤1，scaleOut MARE≈2e-07）（来源平台：950；证据等级：tool_observed）。
- 同根因二次验证：examples 工程出现同一编译错，同样修复后示例在 NPU 运行输出正确（y=127、xOut=1.000000、scaleOut=7.874012e-03）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- Ascend950/CANN 9.2.0 与 9.1.0 实测；A3/A2 未验证，推断可迁移（头文件生成机制跨平台一致）。
- 属性参数变体诊断的是 aclnn_exclude 路径特有的头文件生成错配；非 aclnn_exclude 路径（L2 接口暴露全部属性）的算子，自动生成版本与源码一致，不触发该变体。
- R2 标量类型变体绑定「源码用 aclScalar*、安装头展开为原生类型」这类签名分叉；若源码本身就用原生类型声明标量参数，则安装头与源码一致，无此问题。
- 安装头文件的展开规则（哪些 CANN 版本、哪些参数类型会展开为原生类型）未验证；CANN 版本升级后应重新对照两套头文件（未验证，机制推断可迁移）。
- R2 的 ABI 错位分析基于 x86_64 System V 调用约定；arm64 环境寄存器映射不同，但「签名分叉 → 调用约定错位 → 运行期崩溃」的机制相同（未验证）。
- 段错误发生在首次调用 GetWorkspaceSize 的路径上；若错配参数恰好不参与首批解引用，也可能表现为垃圾输出而非崩溃——不能以"没崩"推断签名一致。

**未决项**：

- 安装包自动生成头文件的生成器在什么条件下把 `const aclScalar*` 展开为 `double`（以及是否会展开其他 aclScalar 类型参数）未追到机制层，建议在下一个自定义算子出包时对比源码头/安装头系统性验证。
## 坏实践

- 手工覆盖 vendors 头文件后把一次编译通过当成生成链已修复；重新打包仍可能产生不一致头文件。
- R2：拿安装包自动生成头文件的签名当事实，编译报错就直接把调用方参数类型改成安装头声明的类型。
- R2：只链接安装库、只 include 安装头，而不对照源码 `op_api/` 的头文件与实现。
- R2：在 PyTorch 适配层里为图省事把标量参数改成本地 double/float 直传，绕过 `aclCreateScalar`。

## 误诊

- R2 第一次修复方向反了：看到编译错后把适配层改成 `double epsilon` 直传（迁就安装头文件），编译通过误以为已解决——实际把编译期错误转移成 NPU 运行期段错误；该误诊态在事件流中有完整 before/after，已作为本卡解决配方的负例保留（编译与崩溃输出均为工具实测）。
- R2 段错误出现后曾怀疑 `OpCommand::RunOpApiV2` 兼容性 / stream 获取 / aclTensor 创建问题；逐一排除后由 `nm -C` 符号核对与源码 .cpp 阅读锁定签名分叉。

## 复现增量（R3）：属性类型改动（String→Int）只改 _def.cpp 触发同型签名分叉——双头文件编译路径机制

另一独立轨迹并入独立 trace（scatter 类算子，Ascend950PR + CANN 9.1.0）的第三条触发路径：为规避 string 属性传递问题，把算子定义的 reduce 属性从 String 改为 Int（`op_host/*_def.cpp` 的 `Attr("reduce").Int()`、`op_graph/*_proto.h` 同步 `.ATTR(reduce, Int, 0)`），L0 API 改 `int64_t reduceMode`。构建、安装都成功后 ST 编译失败：

```
tests/st/test_aclnn_<op>.cpp:1267:17: error: invalid conversion from 'char*' to 'int64_t' {aka 'long int'} [-fpermissive]
  1267 |         tc.dim, reduce_ptr, y_tensor,
      |                 ^~~~~~~~~~
In file included from ...:41:
<CANN 安装根>/opp/vendors/<op>_custom/op_api/include/aclnn_<op>.h:32:13: note:
  initializing argument 5 of 'aclnnStatus aclnn<Op>GetWorkspaceSize(..., int64_t, ...)'
```

三头对照（判定探针）：

```
=== Installed header ===  32:    int64_t reduce,
=== Source header ===     49:    const char *reduce,
=== Build autogen ===    32:    int64_t reduce,
```

**机制（R3 新增）**：构建系统用源码 `op_api/` 头编译 L2 实现（`build/op_host/CMakeFiles/cust_optapi.dir/flags.make` 的 CXX_INCLUDES 含 `-I.../op_host/../op_api`），但安装给用户/ST 的是按 `_def.cpp` 属性类型自动生成的 `build/autogen/aclnn_*.h`——两处签名没有单一事实源，属性类型改动只改 `_def.cpp` 会造成"实现按旧签名编译、安装头按新签名声明"的 ABI 错位（本例被编译期类型检查拦下；若两签名恰好同型不同义则更隐蔽）。即签名分叉不限于 aclnn_exclude 路径——正常路径下"半改" `_def.cpp` 同样触发。

**全链同步清单（R3 修复，本例 String→Int 逐文件）**：`op_host/*_def.cpp`（`Attr("reduce").Int()`）→ `op_graph/*_proto.h`（`.ATTR(reduce, Int, 0)`）→ 源码 L2 头 `op_api/aclnn_<op>.h`（`int64_t reduce`）→ L2 实现 `op_api/aclnn_<op>.cpp`（整数枚举校验）→ L0 头/实现 `op_api/<op>.h/.cpp`（`int64_t reduceMode`）→ tiling `op_host/arch35/*_tiling.cpp`（`attrs->GetInt(1)` 替代 `GetStr(1)`）→ UT faker 属性（`AnyValue::CreateFrom<int64_t>`）→ ST 调用点（字符串→枚举转换）。改完后 `bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 干净构建 + `.run --force` 重装 + ST 重跑。

**R3 新增验证信号**（scatter 算子 reduce 属性 String→Int 端到端改造，Ascend950PR + CANN 9.1.0）：

- 失败信号：ST 编译 `error: invalid conversion from 'char*' to 'int64_t'`，note 指向安装头 `aclnn_<op>.h:32:13: initializing argument 5`（来源平台：950 编译目标；证据等级：tool_observed）。
- 机制证据：安装头/autogen 头均为 `int64_t reduce`、源码头为 `const char *reduce`；`flags.make` 的 CXX_INCLUDES 含 `-I.../op_host/../op_api`（实现按源码头编译）（来源平台：950 编译目标；证据等级：tool_observed）。
- 全链同步后：`--make_clean` 干净构建成功，UT 8 tests PASSED，ST 编译通过进入执行（来源平台：950；证据等级：tool_observed）。

**R3 扩展边界**：

- 本卡原"非 aclnn_exclude 路径不触发"的边界按 R3 修正为：非 exclude 路径下源码头与 autogen 头默认一致，但属性类型改动不同步源码头时同样分叉——"错位方向与直觉相反：旧的是源码头"（撞到 ST 编译错时第一反应"安装头是旧包残留"，核对后发现安装头反而是新构建产物）。
- string 属性在 runtime 链路是否可用未在该轨迹隔离验证（相关实验被 executor 缓存污染），不要引用本卡作为"string 属性不可用"的证据。
- autogen 头生成器对属性→参数映射的完整规则（可选属性、list 属性等）未系统核对。
- 坏实践增补：只改 `_def.cpp` 属性类型就提交构建；用手改 `build/autogen/` 头或安装目录头的方式"救急"（每次重建被覆盖，且掩盖真实不一致）。