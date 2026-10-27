---
type: "Runbook"
title: TPL 双 bool 模板入口与 sub-kernel 实例化验证
description: 入口改为 template <bool,bool> 后 ASCENDC_TPL_SEL 才生效，3 dtype × 3 行 = 9 sub-kernel；用 inspect_build 与 mocker 字符串逐项验证实例化。
tags: [task.implementation, topic.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TPL 双 bool 模板入口与 sub-kernel 实例化验证


## 现象

把一个 AscendC 算子（GRU Cell 反向）的入口从"占位 stub"改造为设计要求的多分支模板分发时，需要回答三个问题：TPL 声明什么时候生效、构建产物里应该出现几个 sub-kernel、怎么验证。改造前后的客观差异（构建产物实测）：

- **stub 状态**（入口为普通函数 + `REGISTER_TILING_DEFAULT`）：codegen 生成的 kernel.cpp 里 `#define TEMPLATE_PARAMS -1` / `TEMPLATE_PARAMS_LEN 0`，kernelList 每 dtype 只有 `_0` 一个 sub-kernel——struct.h 里的 `ASCENDC_TPL_SEL` 三行声明**完全没有生效**。
- **模板入口状态**（入口为 `template <bool isGroup, bool isEmptyTensor>` + `REGISTER_NONE_TILING`）：每 dtype 的 .o 里出现 `_0`/`_1`/`_2` 三个 sub-kernel（对应 base/empty/group 三行 SEL），3 dtype × 3 = 9 个 sub-kernel。

另一个易混点：构建产物里的 `ThnnFusedGruCellGrad-...-0.sh / -1.sh / -2.sh` 三个编译脚本对应的是 **3 个 dtype 编译**（外层枚举，由 ops-info.json 的 dtype combos 驱动 `-DDTYPE_X=float/half/bfloat16_t` 宏），不是 tiling key——不能拿 .sh 的编号去对 tilingKey。

## 适用场景

- 工程形态：AscendC 算子仓开发（ops-registry-invoke 流程），op_kernel 入口（`*_apt.cpp`）改造与构建产物验证；多分支（按 TilingKey bool 字段分派 base/group/empty）算子。
- API 路径：`template_argument.h` 的 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 宏族、`REGISTER_NONE_TILING`、`GET_TILING_DATA_WITH_STRUCT`、`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；构建侧 `build.sh` + `asc_opc` + `inspect_build.py`。
- 验证过的平台与 CANN 版本：Ascend 950（AIV/arch35）+ CANN 9.0.0-beta.2。与 frontmatter `platforms` 严格一致。
- 适用粒度：`operator_family`（用 TPL 模板参数 + TilingKey 分派的算子族；机制是 CANN 构建系统通用契约）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：整型/类型模板参数（本例为双 bool，`ASCENDC_TPL_BOOL=3` 类型码）未在同一 trace 验证，但"sub-kernel 数 = SEL 行数"的验证方法推断可迁移（未验证）；dtype 维度不需要（也不允许）做成模板参数（见设计决策④）。

## 设计决策

1. **为何模板入口而非多个入口函数**：TilingKey 的双 bool 字段（isGroup/isEmptyTensor）与模板参数一一对应、顺序一致，框架按 key 路由到对应 sub-kernel 后缀（SEL 行序即实例化枚举序：base(0,0)→`_0`、empty(0,1)→`_2`、group(1,0)→`_1`——后缀是 key 编码值，key=3 设计空洞不实例化）。
2. **为何 `REGISTER_NONE_TILING` + 各分支手动 `GET_TILING_DATA_WITH_STRUCT`**：三个分支的 TilingData 结构不同（base/group 共用一个 struct、empty 独立一个），统一注册单一 struct 无法表达；手动读取让各分支取各自的 tiling struct。`GET_TILING_DATA_WITH_STRUCT` 还会在 .o 里落下 `__AUX__STRUCT_ascendc_struct_mocker_<变量名>` 字符串，可当每 fatbin .o 的 tiling 布线验证锚点。
3. **为何 group/empty 留桩**：分支开发是增量的（本会话只实装 base），桩保持"该分支不产出"的占位语义；本分支 ST 用例只路由到 key=0，桩的存在不产生回归（此前所有分支都是空跑，现在 base 有真实现、其余仍空跑）。
4. **为何禁止给入口加 `typename D_T` 模板参数**：dtype 由框架按首输入名生成 `DTYPE_GRAD_HY` 宏（`-DDTYPE_X=float/half/bfloat16_t`）在外层枚举编译期实例化；入口再加 dtype 模板参数会与外层枚举机制冲突。
5. **为何去掉 `extern "C"`**：codegen 用 `-Dthnn_fused_gru_cell_grad=thnn_fused_gru_cell_grad_<n>_tilingkey` 重命名符号并以 `thnn_fused_gru_cell_grad<TEMPLATE_PARAMS>(...)` 调用；模板 + extern "C" 组合无法被该包装层调用。
6. **`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`**：显式声明 AIV 任务类型（本算子纯向量核）；该宏在 kernel_utils_macros.h，经 kernel_operator.h 包含链可达。
7. **eventID 一律 `GetTPipePtr()->FetchEventID(...)` 获取、不硬编码 0/1/2**：每事件类型是独立 8 slot 池，硬编码会在多事件场景碰撞。

## 完整代码

入口全文（`op_kernel/thnn_fused_gru_cell_grad_apt.cpp`，verbatim 取自终版写入；base/group/empty 三个模板头与共享 VF 链头由其 include）：

```cpp
/**
 * =============================================================================
 * ThnnFusedGruCellGrad_package/op_kernel/thnn_fused_gru_cell_grad_apt.cpp
 * =============================================================================
 * Role: Ascend C kernel 入口 for ThnnFusedGruCellGrad on arch35。
 *       函数名 = snake_case(OpType)，与运行期 kernel 二进制目录名一致；
 *       参数 = 输入(声明序) → 输出(声明序) → workspace → tiling，
 *       与 docs/ThnnFusedGruCellGrad/develop/proto.md §3 的 kernel 函数签名一致。
 *
 *       入口约束（docs/ThnnFusedGruCellGrad/design/Kernel.md §1.2）：
 *         - 模板参数与 TilingKey 双 bool 字段（isGroup / isEmptyTensor）一一对应、
 *           顺序一致（ASCENDC_TPL_SEL 三行实例化 base=_0 / empty=_2 / group=_1
 *           sub-kernel，key=3 设计空洞不实例化）；禁止加 typename D_T——dtype 由
 *           DTYPE_GRAD_HY（框架按首输入名自动生成）编译期实例化。
 *         - REGISTER_NONE_TILING 统一注册；各分支分别 GET_TILING_DATA_WITH_STRUCT
 *           获取对应 TilingData（base/group 共用 ThnnFusedGruCellGradTilingData，
 *           empty 独立 ThnnFusedGruCellGradEmptyTilingData，TilingData.md §1）。
 *         - AscendC::TPipe pipe 统一在入口申请、以指针传入各模板。
 *
 *       三路分发（Kernel.md §1.2）：
 *         isEmptyTensor → tilingKey=2：empty 模板（Task 28.3.4 占位）
 *         isGroup       → tilingKey=1：group 模板（Task 28.2.4 占位）
 *         否则          → tilingKey=0：base 模板（Branch-0，DESIGN-BRANCH-0.md
 *                          §3–§5：A(H) 条带分核 + 核内 R 二分缓存树 + 双路 bias）
 * =============================================================================
 */

#include "kernel_operator.h"
#include "arch35/thnn_fused_gru_cell_grad_tiling_struct.h"  // TilingData 双 struct（TilingData.md §1）
#include "arch35/thnn_fused_gru_cell_grad_struct.h"          // TPL 双 bool 声明（TilingKey.md §1）
#include "arch35/thnn_fused_gru_cell_grad_kernel.h"          // 跨分支共享 VF 链（Kernel.md §9）
#include "arch35/thnn_fused_gru_cell_grad_base.h"            // base 模板（Branch-0，Task 28.1.4）
#include "arch35/thnn_fused_gru_cell_grad_group.h"           // group 模板（Branch-1，Task 28.2.4）
#include "arch35/thnn_fused_gru_cell_grad_empty.h"           // empty 模板（Branch-2，Task 28.3.4）

// [REF_SAMPLE] 原样例工程入口在此实例化 KernelThnnFusedGruCellGrad<DTYPE_X>
// 并执行 op.Init(x, y, z, tilingData.totalLength, tilingData.blockLength,
// tilingData.tileLength) + op.Process() 双缓冲流水（样例计算类整段以 [REF_SAMPLE]
// 注释保留于 arch35/thnn_fused_gru_cell_grad_kernel.h 尾部）。

template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void thnn_fused_gru_cell_grad(
    GM_ADDR gradHy,             // 输入 (B, H)：上游梯度 ∂L/∂hy
    GM_ADDR storage,            // 输入 (B, 5H)：前向 workspace，行内 [r, z, n, hx, hn] 五平面
    GM_ADDR gradInputGates,     // 输出 (B, 3H)：行内 [grg, gig, gin]
    GM_ADDR gradHiddenGates,    // 输出 (B, 3H)：行内 [grg, gig, ghn]
    GM_ADDR gradHx,             // 输出 (B, H)：grad_hy ⊙ z（直连项）
    GM_ADDR gradInputBias,      // 输出 (3H,)；has_bias=false 时 (0,) 空张量不写
    GM_ADDR gradHiddenBias,     // 输出 (3H,)；has_bias=false 时 (0,) 空张量不写
    GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING：tiling data 由各分支 GET_TILING_DATA_WITH_STRUCT 手动读取
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // AscendC::TPipe：入口统一申请，以指针传入各模板（范式约定，Kernel.md §1）
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        // tilingKey=2：B==0 或 H==0（EMPTY_A/EMPTY_R 共用独立 EmptyTilingData）
        GET_TILING_DATA_WITH_STRUCT(ThnnFusedGruCellGradEmptyTilingData, tilingData, tiling);
        ThnnFusedGruCellGradEmptyKernel<DTYPE_GRAD_HY> op;
        op.Init(gradHy, storage, gradInputGates, gradHiddenGates, gradHx,
                gradInputBias, gradHiddenBias, &tilingData, &pipe);
        op.Process();
    } else if constexpr (isGroup) {
        // tilingKey=1：Phase 1 2D 分核 → SyncAll → Phase 2 RA（读共用 TilingData）
        GET_TILING_DATA_WITH_STRUCT(ThnnFusedGruCellGradTilingData, tilingData, tiling);
        ThnnFusedGruCellGradGroupKernel<DTYPE_GRAD_HY> op;
        op.InitGroup(gradHy, storage, gradInputGates, gradHiddenGates, gradHx,
                     gradInputBias, gradHiddenBias, workspace, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        // tilingKey=0：A(H) 条带分核 + 核内二分缓存树（DESIGN-BRANCH-0.md §5）
        GET_TILING_DATA_WITH_STRUCT(ThnnFusedGruCellGradTilingData, tilingData, tiling);
        ThnnFusedGruCellGradBaseKernel<DTYPE_GRAD_HY> op;
        op.Init(gradHy, storage, gradInputGates, gradHiddenGates, gradHx,
                gradInputBias, gradHiddenBias, &tilingData, &pipe);
        op.Process();
    }
    (void)workspace;    // base：用户 workspace 无（§8，bias 归约核内闭环）；group：InitGroup 消费
}
```

实例化验证命令（构建后执行）：

```bash
# 1) 构建
( set -o pipefail; cd operators && ./build.sh <package> ) ; echo "EXIT=$?"
## 2) 产物检查：dtype 组合数（外层）、每 .o 的 sub-kernel 数（= ASCENDC_TPL_SEL 行数）
python3 skills/ascendc-build-system/scripts/inspect_build.py operators/<package>/build
## 3) tiling 布线锚点：每个 fatbin/sub-kernel .o 都应含 mocker 字符串
for o in operators/<package>/build/<Op>_<soc>/kernel_*/kernel_meta_*/kernel_meta/*.o; do
  c=$(strings "$o" | grep -c "__AUX__STRUCT_ascendc_struct_mocker_"); echo "$(basename $o): mocker=$c"
done
```

## 系统信息

- 工具链：CANN cann-9.0.0-beta.2（bisheng/asc_opc/ld.lld，`asc/include/utils/tiling/template_argument.h` 的 ASCENDC_TPL_* 宏族）；目标 ascend950 / arch35 / AIV（coreType VectorCore）。
- 目录布局：`op_kernel/thnn_fused_gru_cell_grad_apt.cpp`（入口）；`op_kernel/arch35/{thnn_fused_gru_cell_grad_struct.h（TPL 双 bool 声明）, thnn_fused_gru_cell_grad_tiling_struct.h（TilingData 双 struct）, thnn_fused_gru_cell_grad_kernel.h（跨分支共享 VF 链）, thnn_fused_gru_cell_grad_base.h / _group.h / _empty.h（三模板）}`；构建产物 `build/<Op>_ascend950/`（codegen 工作目录含 `_param.json ×3`、`-0/-1/-2.sh ×3`、kernel_meta 每 dtype 一套）。
- 相关文件清单（与实际读/写一一对应）：入口重写（apt.cpp）；新建 base.h / group.h / empty.h；扩写 kernel.h；op_host tiling 侧仅命名对齐。

## 验证信号

- 模板入口构建 EXIT=0、无 error（来源平台：950；证据等级：tool_observed）。
- inspect_build.py 实测：dtype combos 3（float32/float16/bfloat16 → `-DDTYPE_X=float/half/bfloat16_t`）；.o 3 个、每 .o `sub-kernels: 3 ← 应等于 ASCENDC_TPL_SEL 行数`，列出 `_0`/`_2`/`_1`（base/empty/group）；合计 Total sub-kernels: 9（外层 × 内层）；Pipeline COMPLETE 3/3（来源平台：950；证据等级：tool_observed）。
- mocker 锚点实测：3 dtype 的 fatbin .o 与 `_0`/`_1`/`_2` sub-kernel .o 全部 mocker=1，`_meta_info.o` mocker=0（元数据对象，非 kernel 二进制）（来源平台：950；证据等级：tool_observed）。
- 最终回归构建：EXIT=0、error 计数 0、非 meta 的 kernel .o 共 12（3 dtype ×（fatbin + 3 sub-kernel））（来源平台：950；证据等级：tool_observed）。
- 运行期路由旁证：ST 20 条用例全部按 tilingKey=0 走 base 分支产出真值（修复寻址 bug 后 16/20 过、门梯度/ghx 全对），TilingUT 60/60 断言用例到分支的路由（来源平台：950；证据等级：tool_observed）。
- 反面对照（stub 状态）：codegen 产物 `TEMPLATE_PARAMS -1` / `TEMPLATE_PARAMS_LEN 0`、kernelList 每 dtype 仅 `_0`——TPL_SEL 未生效的客观特征（来源平台：950；证据等级：tool_observed）。

## 性能影响

无性能故事（本卡为构建/实例化机制，无 profiling 数据）。

## 预防

- 改造入口前先看构建产物里的 `TEMPLATE_PARAMS` 宏值与 kernelList：`-1`/`LEN 0` 说明 SEL 未生效，别急着怀疑 struct.h 声明。
- 实例化计数写成固定验收项：期望 sub-kernel 总数 = dtype 组合数 × SEL 行数；每个 sub-kernel .o 的 mocker 字符串 ≥1（tiling 布线证据）。
- 入口模板参数只放 TilingKey 对应的 bool/整型维度，dtype 交给框架宏。

## 失效边界

- ASCENDC_TPL_* 宏族行为验证于 CANN 9.0.0-beta.2 的 `template_argument.h`（含 `ASCENDC_TPL_PRE` @@ 标记版供 codegen 预处理、`ASCENDC_TPL_KERNEL` 设备编译版的分层结构）；其它版本未验证。
- 双 bool（`ASCENDC_TPL_BOOL=3` 类型码）已实测；整型/类型参数的 SEL 语法相同但未在本 trace 出现（未验证，推断可迁移）。
- sub-kernel 后缀 = TilingKey 编码值这一映射依赖 SEL 行的取值组合；新增分支或改 key 编码需同步核对 kernelList 与 binary_info_config.json 的 simplifiedKeys。

## 坏实践

- 看到 3 个 `-0/-1/-2.sh` 就当作"3 个 tiling key 各编一次"——它们是 3 个 dtype 的编译脚本。
- stub 状态下 kernelList 只有 `_0` 就以为 TPL 已生效、直接去跑分支用例。

## 误诊

- 无。

## 未决事项

- group/empty 桩为空跑（不写输出）；两个分支实装后需复验各自 sub-kernel 的运行期路由与 mocker 锚点（本 trace 内未实装）。
