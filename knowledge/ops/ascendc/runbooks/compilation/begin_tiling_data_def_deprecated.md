---
type: "Runbook"
title: TilingData 注册：BEGIN_TILING_DATA_DEF 已废弃
description: 自定义算子 TilingData 注册机制选型：BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 字段级宏族在 9.1.0 内置算子语料中零使用且被内置算子规范明令禁用（adv_api 头的旧宏风格属 CANN 自身路径不可照抄）；标准做法是平凡 POD struct + kernel 入口 REGISTER_TILING_DEFAULT + GET_TILING_DATA，注册经 .ascendc_tiling ELF 段由编译管线读出结构体大小，不依赖字段描述宏。
tags: [task.implementation, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-17T04:50:06Z'
updated_at: '2026-09-17T04:50:06Z'
platforms: ["950"]
---

# TilingData 注册：BEGIN_TILING_DATA_DEF 已废弃

## 现象

两条独立轨迹在为自定义算子落地 host→device 共享的 TilingData 结构体时，都遇到了"两套注册机制并存、口径互相冲突"的选型问题（CANN 9.1.0）：

1. `include/register/tilingdata_base.h` 提供的 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` + `REGISTER_TILING_DATA_CLASS`——字段级序列化框架（`TilingDef`/`FieldInfo` 类，带 `SaveToBuffer`/`GetFieldInfo` 等反射式接口），且该头文件注释里的示例就是这种写法；
2. kernel 侧宏 `REGISTER_TILING_DEFAULT`（定义在 `asc/impl/basic_api/utils/kernel_utils_macros.h:394`，展开为 `ENABLE_FEATURE_FOR_TILING(default, tiling_struct)`），配 `GET_TILING_DATA` 取数。

冲突点有三处：

- 轨迹一（euclidean_norm，`op_kernel/arch35/euclidean_norm_tiling_struct.h`）：任务描述要求"`BEGIN_TILING_DATA_DEF`/`REGISTER_TILING_DEFAULT` 注册"，但内置算子参考实现（`opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/acosh/arch35/acosh_tiling_data.h`）文件头明确写着"✅ 必须使用标准 C++ struct 定义 TilingData；❌ 禁止使用废弃的 `BEGIN_TILING_DATA_DEF` 宏"，而同一安装目录的 adv_api 头文件（`logsoftmax_tilingdata.h` 等）却仍是 `BEGIN_TILING_DATA_DEF`/`TILING_DATA_FIELD_DEF` 旧宏风格——两种写法并存，必须判定哪种能真正通过自定义算子包的编译注册链路。
- 轨迹二（softplus_v2_grad，13 字段单结构体两分支共用）：对内置算子做全量检索，`grep -rl "REGISTER_TILING_DATA_CLASS" ~/Ascend/cann-9.1.0/opp/built-in/ | wc -l` = **0**；使用 `REGISTER_TILING_DEFAULT` 的内置 ascendc 算子文件包括 `avg_pool/avg_pool_apt.cpp`、`avg_pool_v2`、`concat_offset`、`conv3d_*_v2`、`repeat_interleave_grad` 等，配套 tiling 结构全部是普通 class/struct（POD、public 成员、无任何宏包装）。
- 判据是构建硬门槛：`./build.sh <pkg>` 退出码 0，且日志中不出现 `do not registe tiling struct` 告警。

两条轨迹均判定机制 2（plain POD struct + `REGISTER_TILING_DEFAULT`）是通过编译注册链路的现代路径，落地后构建通过、无注册告警、真机冒烟通过。

## 适用场景

- 工程形态：算子仓自定义算子开发（ops-registry / regbase 工作流，`op_kernel/arch35/` 结构体 + `op_kernel/<op>_apt.cpp` 或 `op_kernel/<op>_kernel.cpp` 注册）；op_host 侧 tiling 写入与之配套。
- API 路径：`REGISTER_TILING_DEFAULT(<TilingData>)`、`GET_TILING_DATA(var, tiling)`、（多 tilingKey 时）`REGISTER_TILING_FOR_TILINGKEY` + `GET_TILING_DATA_WITH_STRUCT` + `TILING_KEY_IS`；`kernel_utils_macros.h` 的宏展开与 `asc_op_compile_base/asc_op_compiler/compile_op.py` 的段发现机制；host 侧 `context->GetTilingData<T>()`。
- 验证过的平台与 CANN 版本：Ascend950（arch35，`AICore().AddConfig("ascend950")`）+ CANN 9.1.0（两条独立轨迹，与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本编译管线下任何自定义算子的 TilingData 注册选型，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：adv_api（高阶 API）内部的 tilingdata 头仍用旧宏族——那是 CANN 自身代码路径，不受自定义算子包规范约束，也不要照抄进自定义算子包；需要在 host 侧做字段反射/按名读写的工具链场景（那正是字段级框架的存在动机）未在本轨迹涉及；`REGISTER_TILING_FOR_TILINGKEY`（多 tilingKey→多结构体）只做源码级参照未实测；其他 CANN 版本未验证。

## 根因（选型依据）

两套机制的分工：`tilingdata_base.h` 的 `BEGIN_TILING_DATA_DEF` 族是 **host 侧字段描述框架**（供工具按字段名序列化/校验），而 kernel 二进制编译期真正消费的是 `REGISTER_TILING_DEFAULT` 注册的 plain struct 内存布局。注册机制链（各环节均有轨迹证据）：

1. `REGISTER_TILING_DEFAULT(tiling_struct)` 展开为 `ENABLE_FEATURE_FOR_TILING(default, tiling_struct)`（`kernel_utils_macros.h:394`），在设备侧编译单元生成 `.ascendc_tiling.<struct>` ELF 段（`TILING_STRUCT_SECTION_INIT` 放一个 `static const uint64_t` = `sizeof(val)`）。
2. 编译管线 `compile_op.py:819 _get_tiling_struct_size` 用 `llvm-objdump -s -j .ascendc_tiling.<struct>` 读该段内容的前 8 字节解出结构体大小（`max_tiling_size`），随后 `llvm-objcopy --remove-section=.ascendc_tiling.*` 移除该段。
3. 该机制只要求"平凡可复制的普通 C++ struct"——`GET_TILING_DATA` 按 `sizeof` 反序列化（`convert_from_bytes<T>`），不依赖字段描述宏；host 侧 `GetTilingData<T>()` 同样只认类型。
4. 内置 ascendc 算子全部走 plain struct + `REGISTER_TILING_DEFAULT`（语料库检索 `REGISTER_TILING_DATA_CLASS` 计数 0），说明工具链对该路径的支持是一等公民；`REGISTER_TILING_DATA_CLASS` 在内置算子里零使用，选它反而偏离已验证路径。

## 解决配方

### 1. tiling 头：普通 POD struct（before/after 函数级完整，verbatim）

修复前（桩，仅占位字段）：

```cpp
#ifndef EUCLIDEAN_NORM_TILING_STRUCT_H
#define EUCLIDEAN_NORM_TILING_STRUCT_H

#include <cstdint>

struct EuclideanNormTilingData {
    uint32_t totalLength;  // stub placeholder field
};

#endif // EUCLIDEAN_NORM_TILING_STRUCT_H
```

修复后（终版，plain struct，25 字段，含定长数组与默认成员初始化器）：

```cpp
#ifndef EUCLIDEAN_NORM_TILING_STRUCT_H
#define EUCLIDEAN_NORM_TILING_STRUCT_H

#include <cstdint>

// 合轴后轴数上限（DESIGN §5.3.1）。axisShape/axisStride 定长 9；未用位置
// axisShape 填 1、axisStride 填 0。axisType[i] 由 i%2 决定（偶→A，奇→R），不入 TilingData。
constexpr int32_t MAX_PATTERN_RANK = 9;

// EuclideanNorm has no euclidean_norm-specific TilingData fields (sqrt is
// in-register, no coefficient; contrast mean which needs invRTotal). Layout is
// the shared ReduceNormalGenericTilingData, plus the group-template-only field
// `rGroupCnt` (DESIGN §5.3).
struct EuclideanNormTilingData {
    // ─── pattern 描述（去 1 / 合轴 / 补 leading A / 补 R 增广之后）───
    int32_t axisNum = 0;                                       // 2~MAX_PATTERN_RANK
    int64_t axisShape[MAX_PATTERN_RANK] = {0};      // 合轴后每根轴 size（element），未用位置填 1
    int64_t axisStride[MAX_PATTERN_RANK] = {0};     // 每根轴 GM stride（element）

    // ─── 多核切分（fused aLoop）───
    int64_t aLoopCntTotal = 0;                                 // ∏(outer A) × aSplitChunkCnt
    int64_t aSplitChunkCnt = 0;                                // CeilDiv(axisShape[aSplit], aUbFactor)
    int64_t aBigCoreLoopCnt = 0;                               // 大核 aLoop 数
    int64_t aSmallCoreLoopCnt = 0;                             // 小核 aLoop 数；==0 ↔ aLoopCntTotal < coreNum
    int32_t aBigCoreCnt = 0;                                   // 大核数
    int32_t usedCoreNum = 0;                                   // 实际用核数（aBigCoreCnt + smallCoreCnt）

    // ─── UB 切分 ───
    int32_t aSplitAxisIdx = 0;                                 // 切在最内的 A 轴在 pattern 中的 idx
    int32_t rSplitAxisIdx = 0;                                 // 切在最内的 R 轴在 pattern 中的 idx
    int64_t aUbFactor = 0;                                     // valid A 步距（GM 步距、partial chunk 判定）
    int64_t aUbFactorAlign = 0;                                // padded A 步距（UB padded 步距）
    int64_t rUbFactor = 0;                                     // valid R count（GM dense 步距、RLenOfChunk）
    int64_t rUbFactorAlign = 0;                                // padded R count（ReduceSum srcShape[1]、totalElems）
    int64_t innerAProd = 0;                                    // valid inner A 积（输出元素数、CopyOut blockCount）
    int64_t innerAProdAlign = 0;                               // padded inner A 积（UB 预算、stride、ReduceSum 外层维）
    int64_t innerRProd = 0;                                    // valid inner R 积
    int64_t innerRProdAlign = 0;                               // padded inner R 积（ReduceSum 内层维、UB padded 步距）

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0;                                 // ∏(outer R) 总外层 R loop 数

    // ─── UB buffer 字节数 ───
    int64_t preReduceUbSize = 0;                               // pre-elewise buf 字节
    int64_t postReduceUbSize = 0;                              // post-reduce buf 字节
    int64_t tmpBufUbSize = 0;                                  // fp32 tmp buf 字节
    int64_t cacheBufUbSize = 0;                                // 固定 16 KB（kCacheBufBytes = 16*1024），tiling 写入

    // ─── group 模板专属 ───
    int64_t rGroupCnt = 0;                                     // Phase 1 分组数 = Phase 2 workspace R 维
};

#endif // EUCLIDEAN_NORM_TILING_STRUCT_H
```

> 跨算子复用的是**结构机制**（平凡 POD + 定宽标量 + 定长数组 + 默认初始化器 + REGISTER_TILING_DEFAULT），不是字段清单本身——字段是各算子设计文档的产物。

### 2. kernel 入口：注册 + 取数（softplus_v2_grad 桩期完整形态，verbatim）

```cpp
#include "kernel_operator.h"
#include "softplus_v2_grad_tiling.h"

// Stub: interface skeleton only, no computation. The four lines below are
// framework-required boilerplate (NOT algorithm logic); removing them causes
// 361001 at run time.
extern "C" __global__ __aicore__ void softplus_v2_grad(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(SoftplusV2GradTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

（注：注释中"删除四行样板会在运行时触发 361001"来自模板注释，轨迹内未实测复现该错误码；361001 的已验证形态见同域 tilingKey 逐 key 注册卡。）

euclidean_norm 侧的既有用法同构（`op_kernel/euclidean_norm_apt.cpp`，只读核对）：

```cpp
REGISTER_TILING_DEFAULT(EuclideanNormTilingData);
// ... kernel 入口内：
GET_TILING_DATA(tilingData, tiling);
```

### 3. 选型证据检索命令（可复用的判别探针，verbatim）

```bash
# 宏定义位置
grep -rn "REGISTER_TILING_DEFAULT" ~/Ascend/cann-9.1.0/ 2>/dev/null | head -10
# → x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h:394:
#   #define REGISTER_TILING_DEFAULT(tiling_struct)  ENABLE_FEATURE_FOR_TILING(default, tiling_struct)

# 内置算子里两种机制的使用面
grep -rln "REGISTER_TILING_DATA_CLASS\|BEGIN_TILING_DATA_DEF" ~/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops/ 2>/dev/null | head
grep -rl "REGISTER_TILING_DATA_CLASS" ~/Ascend/cann-9.1.0/opp/built-in/ 2>/dev/null | wc -l   # → 0
grep -rln "REGISTER_TILING_DEFAULT" ~/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/ 2>/dev/null | head -8
```

### 4. 构建门槛与验收命令（可直接复用）

```bash
cd operators && ./build.sh <pkg> 2>&1 | tee /tmp/build.log
grep -qi 'do not registe tiling struct' /tmp/build.log && echo FAIL || echo PASS
```

多 tilingKey 时的内置参照（`avg_pool_apt.cpp` verbatim 节选，供分支化 kernel 参考）：

```cpp
extern "C" __global__ __aicore__ void avg_pool(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AvgPoolTilingData);
    if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300002", AvgPoolNCHWSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNCHWSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNCHWSmallPadKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_OUT_DIV_NCHW)) {
        ...
```

## 设计决策

- 为什么选 plain struct 而不是宏族：三重证据互证——内置算子参考（acosh）明令禁用旧宏；语料库 `REGISTER_TILING_DATA_CLASS` 计数 0、内置算子（avg_pool 等）全用 `REGISTER_TILING_DEFAULT` + 普通 class/struct；注册机制（`.ascendc_tiling` ELF 段大小发现）不依赖字段宏。宏族还有两条已验证的失败路径（宿主侧命名空间编译错、共享 tiling.h 时 DT 宏冲突），见同域两张宏范式修复卡。
- 定长数组 `axisShape[9]`/`axisStride[9]` + 默认初始化 `= {0}`：`GET_TILING_DATA` 按 `sizeof` 整体反序列化，定长数组随结构体拷贝，无需字段宏描述。
- 不用位域/打包属性/非平凡成员（如 `std::string`）：保持平凡可复制，按字节反序列化才安全。
- 字段顺序/类型严格对齐设计表：tiling 结构体按声明顺序线性布局，字段偏移即 host 写入与 kernel 读取的契约；分支私有差异只改字段**取值**而不加字段，`tilingKey` 做分发、dtype 不进 TilingData（由 kernel 模板参数选择）。
- 当前方案路线：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950（arch35）、CANN 9.1.0（`/home/<user>/Ascend/cann-9.1.0`）、GCC 13.3.0；构建 `cd operators && ./build.sh <package>`（安装到 `$ASCEND_OPP_PATH/vendors/<OpType>/`）。
- 轨迹一相关文件：`operators/euclidean_norm_package/op_kernel/arch35/euclidean_norm_tiling_struct.h`（结构体，本卡主体）、`op_kernel/euclidean_norm_apt.cpp`（REGISTER_TILING_DEFAULT + GET_TILING_DATA，只读核对）、`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（host 侧 GetTilingData，只读核对）。
- 轨迹二相关文件：`operators/softplus_v2_grad_package/op_kernel/softplus_v2_grad_tiling.h`（13 字段实现）、`op_kernel/softplus_v2_grad_kernel.cpp`（注册点）、`op_host/softplus_v2_grad_host.cpp`（tiling 头的 host 侧消费）。
- 参考锚点：`opp/built-in/.../ascendc/acosh/arch35/acosh_tiling_data.h`（禁用旧宏注释）；`x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h:394`；`python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py:819`；内置 `avg_pool_apt.cpp`/`avg_pool_struct.h`；`include/register/tilingdata_base.h`（宏族定义与示例）。

## 验证信号

以下信号来自两条独立轨迹的 TilingData 结构体落地会话（写入终版后构建+冒烟/验收）。

轨迹一（euclidean_norm，25 字段）：

- 终版写入后 `./build.sh` exit 0（SUCCESS/EXIT=0）（来源平台：950；证据等级：tool_observed）。
- 注册门槛：无 `do not registe tiling struct` 告警；`grep EuclideanNormTilingData` PASS；无 `<...>` 占位残留；`MAX_PATTERN_RANK == 9`；axisShape/axisStride 定长 9；四对 valid/padded 双字段齐（来源平台：950；证据等级：tool_observed）。
- 字段审计：设计定义的 25 个字段全部就位，0 缺失（来源平台：950；证据等级：tool_observed）。
- `tests/aclnn run.sh` 冒烟 8/8 dtype 组合 PASSED、EXIT=0（桩 kernel，验证编译注册与 aclnn 启动链路，非数值语义）（来源平台：950；证据等级：tool_observed）。
- 独立验收复跑：build exit 0 + run.sh 通过（来源平台：950；证据等级：tool_observed）。

轨迹二（softplus_v2_grad，13 字段）：

- 检索证据：`REGISTER_TILING_DATA_CLASS` 在 `opp/built-in/` 下使用计数 0；`REGISTER_TILING_DEFAULT` 宏定义于 `kernel_utils_macros.h:394`；内置 ascendc 算子（avg_pool、avg_pool_v2、concat_offset、conv3d_*_v2、repeat_interleave_grad 等）均使用之，配套结构体均为普通 class/struct（来源平台：950；证据等级：tool_observed）。
- 13 字段结构体 + kernel 注册落地后 `./build.sh softplus_v2_grad_package` 退出码 0，`[ascend950] Generating SoftplusV2Grad_<hash> ... Done`（来源平台：950；证据等级：tool_observed）。
- 构建日志检查 `! grep -qi 'do not registe tiling struct'` 通过（无 tiling 注册告警），13 个字段名逐一 grep 全 OK，字段类型与设计表逐行一致（来源平台：950；证据等级：tool_observed）。
- `run.sh` 真机冒烟 4/4 PASSED（两级 aclnn 调用返回 OK）；kernel `.o` 2 个（= dtype 组合数）；`nm -D libcust_opapi.so` 符号齐全（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.1.0 的语料与 `compile_op.py` 段发现机制；CANN 升级后需重验（未验证，机制推断随版本延续）。
- `REGISTER_TILING_DATA_CLASS` 计数=0 的证据限定 CANN 9.1.0 的这一份安装；未验证，推断可迁移到相近版本。
- 冒烟通过 ≠ tiling 数值正确——轨迹一的 TilingFunc 当时仍是桩（`SetBlockDim(1)`、`workspace[0]=0`），数值验证属后续 tiling 实现任务。
- struct 含非平凡成员（如 `std::string`）会破坏按字节反序列化——应避免（机制推断，未验证）。
- adv_api 头文件的旧宏风格属 CANN 自身代码路径，不要照抄进自定义算子包。
- 多 tilingKey 的 `REGISTER_TILING_FOR_TILINGKEY` 模式只做了源码级参照（avg_pool），两条轨迹均未落地多 kernel 分支，未实测。

**未决项**：

- `REGISTER_TILING_FOR_TILINGKEY`（多 tilingKey→多结构体）场景未在两条轨迹使用，未验证。
- `do not registe tiling struct` 告警的确切触发条件未追（何时会打、打了意味着哪类注册缺失）。
- host 侧写 TilingData（`RawTilingData`/`SetTilingData` 路径）的写入侧验证属后续阶段。
## 坏实践

- 看到 `tilingdata_base.h` 里成体系的字段级框架就默认"更正规"而改用 `BEGIN_TILING_DATA_DEF`——内置算子零使用的路径缺乏工具链级验证，且已有宿主侧命名空间与 kernel 侧 DT 宏冲突两条失败路径。
- 照抄 adv_api 头文件的 `BEGIN_TILING_DATA_DEF`/`TILING_DATA_FIELD_DEF` 风格到自定义算子包——9.1.0 内置规范已禁用。
- 只写 struct、不在设备侧编译单元放 `REGISTER_TILING_DEFAULT`——编译管线读不到 `.ascendc_tiling` 段（对应 `do not registe tiling struct` 告警路径）。
- tiling 结构体字段顺序与设计/host 写入顺序不一致——布局即契约，错位后 host 写的值 kernel 读到别的字段，属于静默数据错误。

## 误诊

- 无（两条轨迹均通过参考锚点 + 语料检索 + 基线构建一次判定；轨迹二初期曾按"版本太老/头文件没包全"方向找 `GetAttrs` 类接口，属另一知识点的弯路，与本卡根因无关）。
