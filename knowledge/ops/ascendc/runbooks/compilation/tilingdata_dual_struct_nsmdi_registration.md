---
type: "Runbook"
title: TilingData 结构体落地——类内初始化器与注册兼容
description: 多分支 TilingData 双 struct（base/group 共用 + empty 独立）带类内默认初始化器落地：设备侧 GET_TILING_DATA_WITH_STRUCT 为裸字节反序列化不依赖构造语义，REGISTER_TILING_DEFAULT 只取类型名求 sizeof，构建与 aclnn 冒烟实测通过。
tags: [task.implementation, topic.implementation.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingData 结构体落地——类内初始化器与注册兼容


## 现象

在 AscendC 算子仓开发流程（ops-registry-invoke，aclnn 接口）里，给 GRU Cell 反向算子（对标 torch 的 GRU Cell backward，ascend950/arch35 平台）落地公共 TilingData 汇总结构体时：设计文档（docs/ThnnFusedGruCellGrad/design/TilingData.md §1）已把 tiling 方案定为三分支两份布局——base（tilingKey=0）/group（tilingKey=1）两个模板**共用**一个 TilingData 结构体（23 个字段：pattern 描述、多核切分、UB 切分、buffer 字节数、group 专属 `rGroupCnt`、算子自定义 `hiddenSize`/`isBias`），empty（tilingKey=2）分支**独立**一个 9 字段结构体；且设计文档的代码块里所有字段都带**类内默认初始化器**（如 `int32_t axisNum = 0;`、`int64_t axisShape[MAX_PATTERN_RANK] = {0};`）。而工程当时的 `op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h` 只是一个单字段桩（`uint32_t totalLength;`），host 侧 TilingFunc 桩还在写这个 `totalLength` 字段。

落地时要回答三个问题：
1. 参考工程（仓内 scale_custom）的 tiling struct 都**不带**类内初始化器，设备编译器（ccec）能不能编译带 NSDMI（non-static data member initializer）的结构体？注册/反序列化宏会不会因初始化器出问题？
2. 两个结构体怎么"编入工程构建"——都要在 kernel 入口注册吗？
3. 改了结构体（`totalLength` 字段删除）后，host 桩里对该字段的写入怎么处理？

最终方案：结构体头文件整文件重写（双 struct + 两个 constexpr 常量，字段与设计文档逐字一致）；kernel 入口的 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` **一行不动**；host 桩删掉字段写入、只保留 tiling 缓冲判空 + `SetBlockDim(1)` + `workspace[0]=0`。构建退出码 0，内核二进制元数据 opParaSize 从 12 涨到 208，aclnn 冒烟 1/1 通过。

## 适用场景（必填 — 检索前的第一道滤网）

- **工程形态**：ops-registry-invoke 算子仓开发（aclnn 接口），tiling 结构体落地阶段——kernel 入口与注册已存在（桩）、TilingFunc 为桩、切分算法尚未实现。
- **API 路径**：`op_kernel/arch35/*_tiling_struct.h`（host/device 共享头）；`REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY` / `TILING_KEY_IS`（kernel_operator.h → kernel_utils_macros.h）；`GET_TILING_DATA_WITH_STRUCT`（编译器 codegen 注入）；host 侧 `gert::TilingContext::GetTilingData<T>()`（exe_graph/runtime/tiling_context.h）；OpDef `AICore().SetTiling(optiling::TilingFunc)` 绑定。
- **验证过的平台与 CANN 版本**：ascend950（arch35）+ CANN cann-9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度（transfer_scope）**：`operator_family`——reduction 范式多分支算子（共用 struct + 分支独立 struct 的两份布局）落地 TilingData 的场景；NSDMI 兼容与注册机制本身是工具链行为。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用 / 未验证场景**：per-tilingKey 注册（`REGISTER_TILING_FOR_TILINGKEY`）的运行时路径未实测（见 ## 失效边界）；TilingFunc 切分算法、kernel 分支计算逻辑不在本卡；非 arch35 平台未验证（未验证，推断可迁移——机制来自 CANN 头文件/codegen，与具体 arch 无关但未逐一实测）。

## 解决配方

按顺序执行（完整代码见 ## 完整代码）：

1. **结构体头文件**：双 struct + constexpr 常量按设计文档代码块**逐字**落地（字段名、类型、数组维、默认初始化器、单位注释全部保留）。约束：仅定宽标量（`int32_t`/`int64_t`/定长数组），保持 trivially copyable（无指针/虚函数/STL 成员）——该结构会被序列化进 tiling 缓冲下发设备。
2. **kernel 入口不动**：`REGISTER_TILING_DEFAULT(最大 struct)` + `GET_TILING_DATA_WITH_STRUCT(同 struct, var, tilingAddr)` 保持原样——注册宏只取类型名求 sizeof，struct 变大不影响调用点；反序列化按 sizeof 裸字节拷贝。
3. **host 桩同步**：删除对已不存在字段的写入；**不填写任何 TilingData 字段**（后续 TilingUT 红基线需要，见 ## 设计决策 4）；保留 `GetTilingData` 指针判空（顺带探测容量）+ `SetBlockDim(1)` + `workspace[0]=0`。
4. **字段一致性机械校验**：正则逐字段抽取设计文档 §1 与实现头文件对比，要求 EXACT MATCH（不靠目测，脚本见 ## 完整代码）。
5. **构建后验证**：跑四探针（构建日志无注册警告 / 每 .o mocker 串 ≥1 / kernel json opParaSize = 8+sizeof / `.ascendc_tiling` section 含 sizeof）+ aclnn 冒烟回归。

**机制链——为什么 NSDMI 结构体是安全的（三条路径都不依赖构造语义）**：

- **设备侧读取**：`GET_TILING_DATA_WITH_STRUCT(T, var, tilingAddr)` 展开调用 `InitTilingData<T>`（宏定义不在普通头文件，由 codegen 注入；构建产物 `kernel_meta_*.h` 生成头可见；下为节选，尾部 4/2/1 字节分支同构）：
  ```cpp
  template <class T>
  __inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
  {
      constexpr uint64_t all_bytes = sizeof(T);
      // CPU_DEBUG / __DAV_C100__ 分支：16 字节块 + 8/4/2/1 尾部裸拷贝
      constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
      uint32_t i = 0;
      if (judge_bytes > 0) {
          for (; i < judge_bytes; i += 16) {
              (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i));
              (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i + 8));
          }
      }
      if (all_bytes & 0x00000008) { /* 8 字节尾部裸拷贝 */ }
      if (all_bytes & 0x00000004) { /* 4 字节尾部裸拷贝 */ }
      // ... 2/1 字节同理
  ```
  NPU 分支走 `copy_gm_to_ubuf(_align)` + `copy_data_align64`——同样是裸字节搬运。即使宏先默认构造了带 NSDMI 的局部变量（全 0），随后也被字节流整体覆盖，编译与运行均无碍。
- **注册**：`REGISTER_TILING_DEFAULT(tiling_struct)` → `ENABLE_FEATURE_FOR_TILING(default, tiling_struct)` → `TILING_STRUCT_SECTION_INIT(__COUNTER__, tiling_struct)`（CANN 头文件 kernel_utils_macros.h 第 363-392 行，节选；中段 `__CHECK_FEATURE_AT_PRECOMPILE` 双态展开见[TilingData 编入构建四探针卡](tilingdata_compiled_into_build_four_probes.md)的完整摘录）：
  ```c
  #define TILING_STRUCT_SECTION_INIT_BASE(counter, val)                                                                \
      static const uint64_t __ascendc_tiling_struct_##counter __attribute__((used, section(".ascendc_tiling."#val))) = \
      sizeof(val)
  #define TILING_STRUCT_SECTION_INIT(counter, val)       TILING_STRUCT_SECTION_INIT_BASE(counter, val)
  /* …第 368-376 行：__CHECK_FEATURE_AT_PRECOMPILE 预编译/正常双态展开（节选省略）… */
  #define REGISTER_TILING_DEFAULT(tiling_struct)  ENABLE_FEATURE_FOR_TILING(default, tiling_struct)
  #define REGISTER_TILING_FOR_TILINGKEY(expression, tiling_struct)  ENABLE_FEATURE_FOR_TILING(expression, tiling_struct)
  ```
  只取类型名求 `sizeof(val)` 放进 `.ascendc_tiling.<StructName>` section——NSDMI 不参与。
- **host 侧**：`GetTilingData<T>()` 返回框架 tiling 缓冲指针（`GetCapacity() < sizeof(T)` 时返回空指针），host 写字段是裸指针写，同样不构造。

**sizeof 算术**（结构体自然对齐，用于预判 opParaSize 与容量）：主 struct `sizeof = 200`（int32 axisNum 4 + pad 4 + int64[3]×2 = 48 + int64×4 = 32 + int32×4 = 16 + int64×8 = 64 + int64×4 = 32 + int32 isBias 4 + 尾部 pad 4 = 200）；empty struct 按同法推算 `sizeof = 72`。实测与 objdump section 内容（0xC8=200）一致。

## 设计决策

1. **三分支并入方式**（设计文档约定）：base/group 共用一个 struct（group 专属信息由尾部字段 `rGroupCnt` 承载，base 不读），empty 独立 struct——"不设第三份分支私有结构体"，算子差异由尾部自定义字段 `hiddenSize`/`isBias` 覆盖全部分支。依据是 reduction 范式的两份布局约定；本卡实测的是这个约定的落地路径。
2. **只注册最大 struct 为 default**：default 注册的 200 字节 struct 是两个 struct 中较大者；按 opParaSize = 8 + sizeof(注册 struct) 的观测规律，框架 tiling 缓冲按 200(+8) 分配，较小的 72 字节 empty struct 后续经 `GetTilingData` 容量检查（72 ≤ 200）也能通过——**此推断未实测**（见 ## 失效边界）。实测部分：200 字节主 struct 的 `GetTilingData` 在 opParaSize=208 下非空（aclnn 冒烟通过；桩在空指针时返回 GRAPH_FAILED 会直接失败）。
3. **per-tilingKey 注册推迟**：empty struct 的 `REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", ...)` 注册属分支 kernel 任务范围，本轮不实施（最小改动原则）。CANN 内置算子 avg_pool 的范式（本轮从工具链源码观测；节选首个分支，其余约 10 个分支同构）：
  ```cpp
  REGISTER_TILING_DEFAULT(AvgPoolTilingData);
  if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_NCHW)) {
      REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300002", AvgPoolNCHWSmallKernelTilingData);
      GET_TILING_DATA_WITH_STRUCT(AvgPoolNCHWSmallKernelTilingData, tilingData, tiling);
      AvgPool::AvgPoolNCHWSmallPadKernel<DTYPE_X> op(&pipeBase, &tilingData);
      op.Init(x, y);
      op.Process();
  } else if (TILING_KEY_IS(...)) { /* 每个分支：注册对应 struct + 反序列化 + 实例化 */ }
  ```
4. **桩不填字段（含被否决的备选）**：备选 A"全字段填 INT32_MIN 哨兵 / 按 0x80 memset"被否决——过度设计；备选 B"零填全部字段"被否决——可能与空 tensor oracle 的全零期望巧合命中。最终**完全不写**：各 UT case 都断言 tilingKey，桩从不 `SetTilingKey` ⇒ `GetTilingKey()=0`，group/empty case 的 tilingKey 断言（期望 1/2）必失败；base case 的 oracle 字段非全零必失败——红基线 0 pass。（此推理为开发时分析，UT 未在本轮运行，agent_reported。）
5. **include guard 风格**：设计文档代码块用 `#pragma once`（参考工程同风格），替换原文件的 `#ifndef` 守卫——按"设计文档代码块即规格"处理。

## 系统信息

- **toolchain**：CANN cann-9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`；ccec 设备编译器；构建命令 `( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package )`，≥600s 超时；build.sh = 清空 build 目录 → cmake → `make -j binary package` → 执行 .run 安装）。
- **目标平台**：ascend950（arch35；内核二进制与 json 落 `build/ascendc_kernels/binary/ascend950/<snake_case 算子名>/`）。
- **目录布局**：`operators/ThnnFusedGruCellGrad_package/{op_kernel/arch35/, op_host/arch35/, op_kernel/, tests/aclnn/}`。
- **相关文件**（与实际读写一一对应）：
  - `op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`（本轮整文件重写）
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`（本轮修改桩）
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.h`（本轮注释更新）
  - `op_kernel/thnn_fused_gru_cell_grad_apt.cpp`（只读保持：注册 + 反序列化所在）
  - `docs/ThnnFusedGruCellGrad/design/TilingData.md`（设计规格，只读）
  - `<CANN>/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`（REGISTER_TILING 宏定义，只读）
  - `build/ThnnFusedGruCellGrad_ascend950/kernel_0/kernel_meta_*.h`（codegen 生成的 InitTilingData，只读）
  - `<CANN>/x86_64-linux/include/exe_graph/runtime/tiling_context.h`（GetTilingData 容量检查，只读）

## 完整代码

**落地后：`op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`（整文件，verbatim）**：

```c++
/**
 * =============================================================================
 * ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h
 * =============================================================================
 * Role: ThnnFusedGruCellGrad 的 host ↔ device 共享 TilingData 结构。
 *       被 host tiling（op_host/arch35）与 device kernel（op_kernel）同时包含。
 *       字段与 docs/ThnnFusedGruCellGrad/design/TilingData.md §1 逐字段一致
 *       （不增不删）：base（tilingKey=0）/ group（tilingKey=1）共用
 *       ThnnFusedGruCellGradTilingData（group 专属信息由 rGroupCnt 承载，
 *       base 不读）；empty（tilingKey=2）专用 ThnnFusedGruCellGradEmptyTilingData。
 *       必须保持 trivially copyable（仅定宽标量字段，无指针/虚函数）——
 *       该结构会被序列化进 tiling 缓冲下发设备（GET_TILING_DATA_WITH_STRUCT
 *       按字节拷贝反序列化）。字段单位：切分/块计数字段为元素数，
 *       usedCoreNum/aBigCoreCnt 为核数，preBufSize/postBufSize/cacheBufUbSize
 *       为字节数。
 * =============================================================================
 */

// GRU 单步融合反向 TilingData — 按 reduction 范式 reduction-template-overview.md §4.1
#pragma once
#include <cstdint>

// 合轴后 pattern 恒 [1(A), B(R), H(A)]：A 起头、A 结尾（tail-A），rseg=1 → 2*1+1 = 3
constexpr int32_t MAX_PATTERN_RANK = 3;
// 两路 bias × [r,z,n] 三段 = 6 个 H 宽块：in_r|in_z|in_n|hid_r|hid_z|hid_n（s-major，
// base 核内六段累加布局与 group workspace 行布局共用；r/z 段两路数值相同）
constexpr int32_t BIAS_SEG_CNT = 6;

// ─── base（tilingKey=0）/ group（tilingKey=1）模板共用 ───
struct ThnnFusedGruCellGradTilingData {
    // ─── pattern 描述（axisType 不需要：i 偶→A，i 奇→R）───
    int32_t axisNum = 0;                        // 恒 3（[1(A), B(R), H(A)]）
    int64_t axisShape[MAX_PATTERN_RANK] = {0};  // 合轴后每根轴 size = [1, B, H]
    int64_t axisStride[MAX_PATTERN_RANK] = {0}; // 主输入 storage（行宽 5H）按元素的 GM stride = [5*B*H, 5*H, 1]

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 usedCoreNum 均匀分核 + 大小核均衡）───
    int64_t aLoopCntTotal = 0;                  // ∏(outer A) × aSplitChunkCnt = CeilDiv(H, aUbFactor)（前导 1 轴不贡献）
    int64_t aSplitChunkCnt = 0;                 // CeilDiv(axisShape[aSplitIdx]=H, aUbFactor)
    int64_t aBigCoreLoopCnt = 0;                // 大核处理的块数（= aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0)）
    int64_t aSmallCoreLoopCnt = 0;              // 小核处理的块数（= aLoopCntTotal / usedCoreNum, floor）
    int32_t aBigCoreCnt = 0;                    // 大核个数（= aLoopCntTotal % usedCoreNum）
    int32_t usedCoreNum = 0;                    // 实际使用核数（base: min(coreNum, aLoopCntTotal)；group: numBlocks）

    // ─── UB 切分 ───
    int32_t aSplitIdx = 0;                      // UB 内被切分的 A 轴下标（恒 2：H）
    int32_t rSplitIdx = 0;                      // UB 内被切分的 R 轴下标（恒 1：B）
    int64_t aUbFactor = 0;                      // valid：A(H) 维单 chunk 元素数（逐元素主链 tile 宽）
    int64_t rUbFactor = 0;                      // valid：R(B) 维单 chunk 元素数（逐元素主链 tile 高）
    int64_t rUbFactorAlign = 0;                 // padded：R 维 UB 行 stride（本算子恒 = rUbFactor，见 §3 约束表）
    int64_t innerAProdAlign = 0;                // padded：含最内 burst-tail A 的 CeilAlign
    int64_t innerRProdAlign = 0;                // padded：含最内 burst-tail R 的 CeilAlign

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0;                  // ∏(outer R) × CeilDiv(axisShape[rSplitIdx]=B, rUbFactor) = CeilDiv(B, rUbFactor)

    // ─── UB buffer 字节数 ───
    int64_t preBufSize = 0;                     // pre 阶段单份 buffer 大小（含 R 维，maxDtypeSize = max(sizeof(D_T), sizeof(fp32))，blocksize 对齐）
    int64_t postBufSize = 0;                    // post 阶段单份 buffer 大小（不含 R 维；六段 bias 归约结果暂存）
    int64_t cacheBufUbSize = 0;                 // 固定 16 * 1024（base 核内 R 二分缓存树专用 buffer）

    // ─── group 模板 ───
    int64_t rGroupCnt = 0;                      // Phase 1 分组数 = Phase 2 workspace R 维大小（base 不读）

    // ─── 算子自定义字段（范式允许按需追加，类比 mean 的 invRTotal）───
    int64_t hiddenSize = 0;                     // H：storage 行宽 5H / 门梯度行宽 3H / grad_hy 与 grad_hx 行宽 H 的基；六段 bias 段偏移 = seg * H
    int32_t isBias = 0;                         // attr has_bias：1=执行双路 bias 归约（group 另含 partial 写 + Phase 2），0=仅逐元素主链
};

// ─── empty（tilingKey=2）模板专用，EMPTY_A / EMPTY_R 共用 ───
struct ThnnFusedGruCellGradEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum = 0;                    // EMPTY_A（H==0，或 B==0 且 isBias==0）：0（全核早退，SetBlockDim(1)）；EMPTY_R（B==0 且 isBias==1）：零填充切分核数
    int64_t aTotal = 0;                         // ∏(所有 A 轴) = H（EMPTY_A 不读）
    int64_t aUbFactor = 0;                      // 单 chunk a 元素数（4 约束取 min：4KB 下界 / 优先多核 / UB 上限 / aTotal 兜底）
    int32_t aBigCoreCnt = 0;                    // 大核个数
    int64_t aBigCoreLoopCnt = 0;                // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt = 0;              // 每小核 chunk 数

    // ─── UB buffer ───
    int64_t postBufSize = 0;                    // 填充阶段单份 buffer 字节数

    // ─── 算子自定义字段 ───
    int64_t hiddenSize = 0;                     // H：EMPTY_R 零填充元素数 = BIAS_SEG_CNT * H（仅 isBias==1 时）
    int32_t isBias = 0;                         // 1=B==0 时零填充两路 bias（6H 个 0，空 batch 求和的 fp 数值语义）；0=bias 输出为 (0,) 空张量，无 GM 写
};
```

> 字段名/类型/数组维/默认初始化器为该算子设计文档的规格（落地时逐字保留）；**可复用的是结构机制**——定宽标量 + NSDMI + trivially copyable + 最大者注册为 default。

**落地前：`op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`（整文件，verbatim）**：

```c++
/**
 * =============================================================================
 * ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h
 * =============================================================================
 * Role: ThnnFusedGruCellGrad 的 host ↔ device 共享 TilingData 结构（桩）。
 *       被 host tiling（op_host/arch35）与 device kernel（op_kernel）同时包含。
 *
 * 桩说明：仅一个占位字段；真实字段（pattern/hiddenSize/isBias 等）由
 * tiling-developer 按 design/TilingData.md 落地。必须保持 trivially copyable
 * （仅定宽标量字段，无指针/虚函数）——该结构会被序列化进 tiling 缓冲下发设备。
 * =============================================================================
 */

#ifndef THNN_FUSED_GRU_CELL_GRAD_TILING_STRUCT_H
#define THNN_FUSED_GRU_CELL_GRAD_TILING_STRUCT_H

#include <cstdint>

struct ThnnFusedGruCellGradTilingData {
    uint32_t totalLength;
};

#endif // THNN_FUSED_GRU_CELL_GRAD_TILING_STRUCT_H
```

**落地后：host 桩 TilingFunc（`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp` 的函数与前置注释，verbatim）**：

```c++
// 桩：TilingData 结构体已按 design/TilingData.md 落地（base/group 共用
// ThnnFusedGruCellGradTilingData + empty 专用 ThnnFusedGruCellGradEmptyTilingData），
// 但切分公式尚未实现（后续 TilingUT/Tiling 任务）：桩不填写任何 TilingData
// 字段（供 TilingUT 红基线 = 0 pass），仅保证 tiling 通路可用（1 核 / 无 workspace）。
// ⚠️ 芯片平台信息（ubSize/coreNum 等）必须经 context->GetPlatformInfo() 获取，
//    不能依赖 GetCompileInfo<T>()：aclnn 通路不调用 TilingParse，恒返回 null。
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 仅探测 tiling 缓冲可用（容量按 REGISTER_TILING_DEFAULT 注册的
    // ThnnFusedGruCellGradTilingData 大小分配）；桩阶段不写字段。
    ThnnFusedGruCellGradTilingData* tiling = context->GetTilingData<ThnnFusedGruCellGradTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(1);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

**落地前：host 桩 TilingFunc（同文件函数体，verbatim）**：

```c++
// 桩：1 核、无 workspace、仅写 totalLength 占位字段。
// ⚠️ 芯片平台信息（ubSize/coreNum 等）必须经 context->GetPlatformInfo() 获取，
//    不能依赖 GetCompileInfo<T>()：aclnn 通路不调用 TilingParse，恒返回 null。
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ThnnFusedGruCellGradTilingData* tiling = context->GetTilingData<ThnnFusedGruCellGradTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = 0;
    const auto* gradHyStorageShape = context->GetInputShape(0);
    if (gradHyStorageShape != nullptr) {
        totalLength = static_cast<uint32_t>(gradHyStorageShape->GetOriginShape().GetShapeSize());
    }
    tiling->totalLength = totalLength;

    context->SetBlockDim(1);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

**kernel 入口注册段（`op_kernel/thnn_fused_gru_cell_grad_apt.cpp`，本轮未改动，verbatim）**：

```c++
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

**字段一致性机械校验脚本（bash 内嵌 python，verbatim）**：

```bash
python3 - <<'EOF'
import re

def extract_fields(path, struct_name):
    text = open(path, encoding='utf-8').read()
    # find struct block
    m = re.search(r'struct\s+' + struct_name + r'\s*\{(.*?)\n\};', text, re.S)
    if not m:
        return None, f"STRUCT {struct_name} NOT FOUND in {path}"
    body = m.group(1)
    fields = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('───') or line.startswith('──'):
            continue
        # strip trailing comment
        code = line.split('//')[0].strip()
        if not code:
            continue
        fields.append(code)
    return fields, None

design = 'docs/ThnnFusedGruCellGrad/design/TilingData.md'
impl = 'operators/ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h'

ok = True
for sname in ['ThnnFusedGruCellGradTilingData', 'ThnnFusedGruCellGradEmptyTilingData']:
    d, e1 = extract_fields(design, sname)
    i, e2 = extract_fields(impl, sname)
    if e1 or e2:
        print(f"ERROR: {e1 or e2}"); ok = False; continue
    if d == i:
        print(f"== {sname}: {len(d)} fields, EXACT MATCH")
    else:
        ok = False
        print(f"== {sname}: MISMATCH (design {len(d)} vs impl {len(i)})")
        for n,(a,b) in enumerate(zip(d,i)):
            if a != b:
                print(f"  line {n}: design='{a}' impl='{b}'")
        if len(d) != len(i):
            print("  design-only:", set(d)-set(i))
            print("  impl-only:", set(i)-set(d))

# also constants
dtext = open(design, encoding='utf-8').read()
itext = open(impl, encoding='utf-8').read()
for c in ['constexpr int32_t MAX_PATTERN_RANK = 3;', 'constexpr int32_t BIAS_SEG_CNT = 6;']:
    print(('OK  ' if c in dtext else 'MISS-IN-DESIGN? ') + ('OK  ' if c in itext else 'MISS-IN-IMPL ') + c)

print("RESULT:", "PASS" if ok else "FAIL")
EOF
```

## 性能影响

无性能数据（结构体落地阶段，未跑性能评测；无两条路线的相对对比）。

## 预防

- TilingData 结构体只允许定宽标量与定长数组（trivially copyable），禁指针/虚函数/STL 成员——结构体会被序列化进 tiling 缓冲下发设备。
- 增删/重命名字段前先 grep 全包引用（本轮先 grep `totalLength` 确认只有 host 桩在写它，kernel 侧只在 REF_SAMPLE 注释里出现）。
- 字段落地后跑机械 diff 对设计文档（脚本见 ## 完整代码），不靠目测。
- 构建后跑四探针（见同批[TilingData 编入构建四探针卡](tilingdata_compiled_into_build_four_probes.md)）+ aclnn 冒烟，不只看退出码。

## 验证信号

来源说明：本验证信号取自本轮 tiling 结构体落地开发会话——双 struct 写入后的构建、产物元数据与 aclnn 回归实测。

- 构建 `( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package )` 退出码 0（cmake + make binary package + .run 安装成功）（来源平台：950；证据等级：tool_observed）
- 字段机械 diff：主 struct 23 字段、empty struct 9 字段均与设计文档 §1 EXACT MATCH，`MAX_PATTERN_RANK=3`/`BIAS_SEG_CNT=6` 双方在位，结果 PASS（来源平台：950；证据等级：tool_observed）
- 内核二进制元数据：3 个 dtype 变体 kernel json 的 opParaSize 全部 = 208 = 8 + 200（sizeof 新 struct，替换前 4 字节桩时为 12）（来源平台：950；证据等级：tool_observed）
- `.ascendc_tiling.ThnnFusedGruCellGradTilingData` section（16 字节）内容首 8 字节小端 0xC8 = 200 = sizeof（来源平台：950；证据等级：tool_observed）
- 构建日志 "do not registe tiling struct" 警告计数 0；3 个 .o 的 `__AUX__STRUCT_ascendc_struct_mocker_` 计数各 1（来源平台：950；证据等级：tool_observed）
- aclnn 冒烟回归：EXIT=0，`[smoke] thnn_fused_gru_cell_grad fp32 PASSED`，`1/1 cases OK`，FAILED 计数 0——空指针会 GRAPH_FAILED，冒烟通过反证 tiling 缓冲按新 sizeof 分配（来源平台：950；证据等级：tool_observed）
- 改动范围：git diff 仅 3 个 tiling 文件（tiling_struct.h / tiling_arch35.cpp / tiling_arch35.h），def.cpp 与 infershape.cpp 未动（来源平台：950；证据等级：tool_observed）

## 失效边界

- **per-tilingKey 注册路径未实测**：本轮只 default 注册；empty struct 经 `SetTilingKey(2)` 分发 + `GetTilingData<EmptyTilingData>` 的运行行为未验证。"小 struct 在大 default struct 缓冲下容量检查通过"是从 tiling_context.h 容量检查代码 + opParaSize 观测推断，未实测 empty 分支。
- NSDMI 兼容性验证于 arch35/ascend950 + cann-9.0.0-beta.2；InitTilingData 的其他 arch 分支（`__DAV_C100__`、CPU_DEBUG、3510/5102 的 dc_preload 路径）同为裸字节拷贝（源码可见），但未逐一实测（未验证，推断可迁移）。
- 本卡不覆盖：TilingFunc 切分算法（未实现）、kernel 分支计算逻辑、TilingUT 红基线的实际通过数（未运行）。
- `REGISTER_TILING_FOR_TILINGKEY` 非预编译展开中 expression 被丢弃的机制（运行时如何按 tilingKey 选 struct）未追到——见 ## 未决事项。

## 坏实践

本闭环无失败轮次，无客观坏实践记录。被否决的备选（哨兵填充/零填充全部字段）见 ## 设计决策 4——它们不是错误做法，而是被判定为不必要或有一点点巧合风险的过度设计。

## 误诊

无（本闭环未发生错误归因）。

## 未决事项

1. `REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == <k>", Struct)` 的非预编译展开丢弃 expression、section 名只含 struct 名——运行时如何把 tilingKey 映射到具体 struct 未追到机制层；分支 kernel 任务落地 per-tilingKey 注册时需优先验证（含 tilingKey=2 时框架按哪个 struct 分配缓冲）。
2. opParaSize 的 "+8" 头部确切构成未从 CANN 源码确认（两点经验式：4→12、200→208）。
3. TilingUT 红基线 = 0 pass 的推理（桩不填字段 + 不 SetTilingKey ⇒ 全 case 失败）待 UT 阶段实测确认。
4. 多 struct 注册场景下 opParaSize 取值（max 还是 per-key）未验证。
