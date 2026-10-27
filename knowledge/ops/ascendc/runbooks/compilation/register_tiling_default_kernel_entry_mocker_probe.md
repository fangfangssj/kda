---
type: "Runbook"
title: TilingData 注册核验：strings .o 查 mocker 符号
description: REGISTER_TILING_DEFAULT 的注册以 __AUX__STRUCT_ascendc_struct_mocker_ 符号串物化进 kernel binary；源级 + 二进制级两级探针确认 TilingData 注册未因重构/桩化丢失。
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingData 注册核验：strings .o 查 mocker 符号


## 现象

在按 proto.md 把一个 ACLNN 自定义算子工程（AscendC，ops-registry-invoke 流程，目标 ascend950）改造为接口桩后的复验中，验收口径要求「TilingData 结构体注册与 OP_ADD 注册语句保留，不因改原型而丢失」。第一反应是在 op_host 下找注册宏：

```
grep -R "REGISTER|TILING_DATA|BEGIN_TILING_DATA_DEF" <pkg>/op_host
No files found
```

op_host 全目录没有任何 TILING 注册宏——如果带着「注册宏应该在 op_host（BEGIN_TILING_DATA_DEF / IMPL_OP_OPTILING 一类）」的预期，会误判为「注册在改造中丢了」。实际上该流程的 TilingData 注册不在 op_host，而在 **kernel 入口**里：`op_kernel/<op>_apt.cpp` 内 `REGISTER_TILING_DEFAULT(<OpType>TilingData)` + `GET_TILING_DATA_WITH_STRUCT(...)`；结构体本身是 `op_kernel/arch35/<op>_tiling_struct.h` 里的平凡 POD struct（host tiling 与 device kernel 共同包含）。与同仓样例工程（add_custom、scale_custom）grep 比对确认是同款形态。源级确认后，再对编译出的 kernel binary 逐枚做 strings 计数，才能证明注册真正进入了产物。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：aclnn / ops-registry-invoke 算子仓（op_build autogen 流程）；重构/桩化/搬运 kernel 入口后需要核验 TilingData 注册仍然生效的场景。
- API 路径：`op_kernel/<op>_apt.cpp` 内 `REGISTER_TILING_DEFAULT(<Struct>)` / `GET_TILING_DATA_WITH_STRUCT(<Struct>, var, tilingAddr)`；`op_host/arch35/<op>_tiling_arch35.cpp` 的 TilingFunc 经 OpDef `AICore().SetTiling(optiling::TilingFunc)` 绑定（不走 IMPL_OP_OPTILING）。
- 验证过的平台与 CANN 版本：ascend950（注册平台名 950）+ CANN cann-9.0.0-beta.2。
- 适用粒度（transfer_scope）：platform_wide——op_build + REGISTER_TILING_DEFAULT 流程内跨算子一致，与算子语义无关。
- 路线优先级：recommended_fastpath（两级探针一次做完，成本两条命令）。
- 明确不适用 / 未验证场景：走 IMPL_OP_OPTILING + BEGIN_TILING_DATA_DEF 的旧式 op_host 侧注册工程（注册位置与探针都不同，未验证）；GEIR tiling 单测流程；未验证场景——其他 CANN 版本的符号命名。

## 判别方法

两级探针（都过才算注册保留）：

1. **源级**——op_kernel 下存在非注释的注册与读取调用：

```bash
grep -n 'GET_TILING_DATA_WITH_STRUCT' <pkg>/op_kernel/<op>_apt.cpp | grep -v '^\s*[0-9]*:\s*//'
# 预期命中类似： 33:    GET_TILING_DATA_WITH_STRUCT(<OpType>TilingData, tilingData, gruCellGradTiling);
```

2. **二进制级**——对每枚交付 kernel binary 统计 mocker 符号串（每 dtype 组合一枚，逐枚查；build 树与安装目录各查一遍更稳）：

```bash
for f in <pkg>/build/ascendc_kernels/binary/<soc>/<op>/*.o; do
  c=$(strings "$f" | grep -c '__AUX__STRUCT_ascendc_struct_mocker_')
  echo "$(basename $f): mocker_count=$c"
done
```

判读：每枚计数 ≥ 1 → `REGISTER_TILING_DEFAULT` 的注册已进入编译产物；计数 = 0 → 注册丢失（宏被注释/删除，或所在编译单元未被包含进构建）。op_host 下 grep 不到 TILING 注册宏是**该流程的正常形态**，不构成注册丢失的证据。

## 设计决策

- 源级 grep 只能证明「源码里写了」，不能证明「编译产物里有」——接口桩/重构类改动恰恰容易在搬运、注释、替换入口时丢注册，所以补二进制级 strings 探针；两级都过才下「注册保留」的结论。此为本卡采用的核验设计，属 recommended_fastpath。
- 注册形态对齐样例工程而非引入 BEGIN_TILING_DATA_DEF：比对显示 add_custom（`REGISTER_TILING_DEFAULT(AddCustomTilingData)` 于 kernel 入口）与 scale_custom（`GET_TILING_DATA_WITH_STRUCT(TilingData4/TilingData8, td, tiling)` 多 tiling 分支）均为此形态，POD struct 放 op_kernel/arch35 共享头（须 trivially copyable，仅定宽标量字段，因结构会被序列化进 tiling 缓冲下发设备）——有样例比对证据支撑，非臆选。

## 系统信息

- 构建环境：x86_64 Ubuntu，GCC 13.3.0，CMake + op_build；目标 ascend950（arch35）；CANN cann-9.0.0-beta.2。
- 相关文件（与本次事件流实际读取一一对应）：
  - `op_kernel/thnn_fused_gru_cell_grad_apt.cpp`（注册点：kernel 入口内 REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT）
  - `op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`（POD struct，host↔device 共享）
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（TilingFunc 桩，经 OpDef SetTiling 绑定）
  - `build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/*.o` 与 `$ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/*.o`（被检二进制）
  - 样例参照：`operators/add_custom_package/op_kernel/add_custom_apt.cpp`、`operators/add_custom_package/op_kernel/arch35/add_custom_tiling_struct.h`、`operators/scale_custom_package/op_kernel/scale_custom_apt.cpp`

## 完整代码

被核验的注册形态（接口桩 kernel 入口完整函数，verbatim 摘自事件流读到的 `op_kernel/thnn_fused_gru_cell_grad_apt.cpp`）：

```cpp
extern "C" __global__ __aicore__ void thnn_fused_gru_cell_grad(
    GM_ADDR grad_hy, GM_ADDR storage,
    GM_ADDR grad_input_gates, GM_ADDR grad_hidden_gates, GM_ADDR grad_hx,
    GM_ADDR grad_input_bias, GM_ADDR grad_hidden_bias,
    GM_ADDR gruCellGradWorkspace, GM_ADDR gruCellGradTiling)
{
    AscendC::InitSocState();

    // TilingData 必须注册（REGISTER_TILING_DEFAULT），并以
    // GET_TILING_DATA_WITH_STRUCT 反序列化读取。
    REGISTER_TILING_DEFAULT(ThnnFusedGruCellGradTilingData);

    GET_TILING_DATA_WITH_STRUCT(ThnnFusedGruCellGradTilingData, tilingData, gruCellGradTiling);
    (void)tilingData;

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

被核验的结构体（`op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`，verbatim）：

```cpp
struct ThnnFusedGruCellGradTilingData {
    uint32_t totalLength;
};
```

样例工程同款注册（`operators/add_custom_package/op_kernel/add_custom_apt.cpp`，grep 输出 verbatim）：

```
Line 36:     REGISTER_TILING_DEFAULT(AddCustomTilingData);
```

```
operators/scale_custom_package/op_kernel/scale_custom_apt.cpp:
  Line 117:         GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
  Line 124:         GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
```

探针命令见 ## 判别方法（完整可执行）。

## 预防

- 重构/桩化 kernel 入口时，`REGISTER_TILING_DEFAULT(<Struct>)` 与 `GET_TILING_DATA_WITH_STRUCT(<Struct>, var, tilingAddr)` 必须随入口一起保留；验证时不要只 grep op_host（该流程注册不在 op_host）。
- 交付前对每枚 kernel binary 跑一次 mocker 计数（记住每 dtype 组合一枚，逐枚查，不要只查一枚就推广）。
- TilingData 结构体保持平凡 POD（仅定宽标量、无指针/虚函数）——它会被序列化进 tiling 缓冲下发设备。

## 验证信号

以下信号来自该接口桩工程的复验：亲自执行 op_host/kernel 目录 grep、样例工程比对 grep、以及对 build 树与安装目录全部 6 枚 kernel binary 的 strings 计数取得。

- op_host 全目录 grep `REGISTER|TILING_DATA|BEGIN_TILING_DATA_DEF` → `No files found`（来源平台：950；证据等级：tool_observed）
- `op_kernel/<op>_apt.cpp` 非注释 `GET_TILING_DATA_WITH_STRUCT` 命中（line 33），同文件含 `REGISTER_TILING_DEFAULT(ThnnFusedGruCellGradTilingData)`（line 31）（来源平台：950；证据等级：tool_observed）
- build 树 3 枚交付 .o 各 `__AUX__STRUCT_ascendc_struct_mocker_` 计数 = 1；安装目录同 3 枚亦各 = 1（合计 6 枚次均 ≥ 1）（来源平台：950；证据等级：tool_observed）：

```
ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o: mocker_count=1
ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.o: mocker_count=1
ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.o: mocker_count=1
--- installed ---
ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o: mocker_count=1
ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.o: mocker_count=1
ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.o: mocker_count=1
```

- 样例工程比对：`add_custom_apt.cpp:36 REGISTER_TILING_DEFAULT(AddCustomTilingData)`、`scale_custom_apt.cpp:117/124 GET_TILING_DATA_WITH_STRUCT(TilingData4/TilingData8, td, tiling)`、add_custom tiling struct 亦为三 uint32_t 字段 plain struct（来源平台：950；证据等级：tool_observed）

## 失效边界

- mocker 计数 ≥ 1 只证明注册符号进入产物，**不证明 tiling 字段语义正确**（字段内容、分核算法是否正确需 tiling 单测/运行验证）。
- 观察限于 cann-9.0.0-beta.2 + ascend950 的 op_build 产物；其他 CANN 版本该符号前缀是否同名未验证——未验证，推断可迁移（迁移前先在一枚已知注册正常的 .o 上复验探针）。
- BEGIN_TILING_DATA_DEF / IMPL_OP_OPTILING 类 op_host 侧注册不适用本探针（本卡流程不使用该形态，探针对其行为未验证）。

## 坏实践

- 只看源码里有 `REGISTER_TILING_DEFAULT` 就判注册生效，不做产物级核验——搬运/注释/编译单元遗漏不会被发现。
- 在 op_host 下找 TILING 注册宏、找不到就判「注册丢失」——该流程注册点在 kernel 入口（本例实测 op_host grep 为空而注册实际存在且产物计数=1）。
- 只对一枚 .o 计数就下结论——多 dtype 组合时每枚都需查（本例 3 枚逐枚查）。

## 误诊

（本次未形成结论性误诊：op_host grep 落空后转向 kernel 侧核实，未错误判定注册丢失。记录一个近似误诊路径供警觉——「op_host 无注册宏 ⇒ 注册丢失」是本现象下最自然的错误假设，须用 kernel 侧 grep + 二进制计数排除。）

## 未决事项

- **优先项：复现并追根因到机制层**——`REGISTER_TILING_DEFAULT` 宏如何展开/经 op_build 生成 `__AUX__STRUCT_ascendc_struct_mocker_` 符号串的确切代码路径未追（本卡只建立了「注册宏 ⇒ 产物内符号串」的实测对应关系）；如需在非 op_build 流程复用该探针，需先核实该版本工具链的符号生成逻辑。
- mocker 计数是否恒为 1（多 struct/多注册时会否 >1）未观察——本例全部为 1，仅以 ≥ 1 为判据。
