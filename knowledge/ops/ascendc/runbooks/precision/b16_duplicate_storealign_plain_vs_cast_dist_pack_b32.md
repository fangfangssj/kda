---
type: "Runbook"
title: b16常量零填充StoreAlign无需DIST_PACK_B32
description: "AscendC Reg 级 VF 中 Reg::Duplicate 产出的 b16 寄存器用不带 StoreDist 的 StoreAlign 落 UB 即位级正确（常量 0 实测）；DIST_PACK_B32 是 Cast fp32→b16 输出（b16 驻留 fp32 lane）路径的打包需求，两者不可混用直觉。R2（CANN 9.2.0）反向实验闭合：Cast b32→b16 后省略 DIST_PACK_B32 的典型症状为 fp16/bf16 output[0] 正确、elem 1 起全错（fp32 全对）；修复两条路——VF 外 UB 级 Cast(CAST_RINT) 或 StoreAlign<D_T, DIST_PACK_B32>。"
tags: [task.implementation, topic.compute.elementwise.type_conversion, topic.precision, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: "2026-09-09T11:49:41Z"
updated_at: "2026-09-17T03:25:37Z"
platforms: ["950"]
---

# b16常量零填充StoreAlign无需DIST_PACK_B32

## 现象

在为 GRU Cell 反向算子的空张量分支（空 batch 时两路 bias 输出零填充）编写跨分支共享的寄存器级 VF（`__simd_vf__` + `asc_vf_call`）时，代码库里 AscendC Reg 级 store 存在两种写法并存：

- 主链 VF（如 GhxVfImpl）在 `Cast<D_T, float, kCastTraitF32ToB16>` 把 fp32 计算结果转成 b16 后，store 必须带打包分布：`AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(ghxTile + off + c, bReg, mask);`（fp32 分支则用不带 StoreDist 的 plain `StoreAlign`）。
- 而新写的常量填充 VF（DuplicateEmptyROutputVfImpl，按设计文档与 reduction 范式模板）对包括 fp16/bf16 在内的所有 T 都用**不带 StoreDist 的 plain** `AscendC::Reg::StoreAlign(dst + off, vReg, mask);`。

实现时的真实疑虑：`Reg::Duplicate(RegTensor<half>, value)` 产出的寄存器到底是"原生 b16 打包寄存器"还是"b16 值驻留在 fp32 lane 的低位"——若是后者，plain StoreAlign 会把高位垃圾/交错数据写进 UB，fp16/bf16 的零填充输出将位级错误，必须像 Cast 输出那样加 `DIST_PACK_B32`。该疑虑有实际依据：同一代码库里两种写法并存，盲目模仿任一侧都可能错。

本卡给出判别规则与实测边界：**值的来源决定 store 形态**——`Reg::Duplicate` 直接以 `RegTensor<T>` 声明类型产出原生 T 域寄存器（b16 时 128 个元素打包占满 256B 寄存器），plain StoreAlign 按原生 T 连续落 UB；`Cast` 从 fp32 域转出的 b16 结果驻留 fp32 lane（每 32bit lane 存 1 个 b16），落 UB 前必须 DIST_PACK_B32 打包。

## 适用场景

- **工程形态**：AscendC kernel 的寄存器级 VF 编程（`__simd_vf__` 实现体 + `asc_vf_call` 调用）；涉及 fp16/bf16 常量填充、初始化、清零路径，或 fp32→b16 缩位输出路径。
- **API 路径**：`AscendC::Reg::RegTensor<T>` / `AscendC::Reg::Duplicate` / `AscendC::Reg::StoreAlign`（含 `StoreDist::DIST_PACK_B32` 模板参）/ `AscendC::Reg::Cast<D_T, float, kCastTraitF32ToB16>` / `AscendC::Reg::UpdateMask<T>`；头文件 `ascendc/include/basic_api/impl/common_types.h` 等 Reg API。
- **验证过的平台与 CANN 版本**：Ascend 950 系列（arch35，AIV 向量寄存器 256B）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- **适用粒度（transfer_scope）**：`platform_wide`——950 AIV Reg API 的寄存器/lane 语义，跨算子适用。
- **路线优先级**：`recommended_fastpath`（常量填充按值来源选 plain StoreAlign；Cast 缩位输出保持 DIST_PACK_B32——两条各自正确的推荐写法，不是互为 fallback）。
- **明确不适用 / 未验证场景**：非零常量值的 b16 Duplicate+plain StoreAlign lane 语义未实测（本卡结论严格限定常量 0 填充路径，见失效边界）；b16 多 rep（count > VL=128）循环未实测；`LoadAlign` 侧 `DIST_UNPACK_B16` 与 `DIST_PACK_B32` 的对称性为代码库既有模式（文件读取佐证），未做单独最小实验；非 950 平台寄存器宽度未验证。

## 判别方法

写 b16 数据落 UB 前，先问一个问题：**b16 值当前在哪种寄存器表示里？**

1. **原生 T 域寄存器**（`RegTensor<T>` 声明后由 `Reg::Duplicate(vReg, value)` 直接产出）：寄存器按 `VL = kVlBytes / sizeof(T)` 个 T 元素组织——fp32=64、b16=128 个元素/256B 寄存器，元素连续打包。→ **plain `StoreAlign(ptr, reg, mask)`**（不带 StoreDist），按原生 T 连续写 UB。
2. **fp32 lane 驻留**（`Cast<D_T, float, kCastTraitF32ToB16>(bReg, tReg, mask)` 从 fp32 计算结果转出）：b16 值每 32bit lane 存 1 个。→ **必须 `StoreAlign<D_T, StoreDist::DIST_PACK_B32>`** 打包后落 UB。

判别探针（当时实际执行的验证路径）：读 CANN 头文件确认枚举/常量（`kVlBytes = 256U` 注释"ascend950 AIV 向量寄存器宽度（字节）"），按设计注释 `VL = 256 / sizeof(T)`（fp32=64，b16=128）推断寄存器模型——128 个 b16 元素 × 2B = 256B 恰好占满寄存器，即原生打包表示；再用 ST 的 poison 初化 + 位级比对实测（见验证信号）。

## 机制推导

- **寄存器模型**：950 AIV 向量寄存器宽 256B（`kVlBytes = 256U`）。`VL = kVlBytes / sizeof(T)` → fp32 每寄存器 64 元素、fp16/bf16 每寄存器 128 元素。b16 的 128 元素 × 2B = 256B 整寄存器——`RegTensor<T>` 按声明类型 T 组织元素，`Reg::Duplicate` 在该原生布局上广播常量。
- **Cast 路径为何不同**：`Cast<D_T, float, kCastTraitF32ToB16>` 的输入是 fp32 寄存器（64 lane），转出的 b16 结果驻留在 fp32 lane（每 32bit 存 1 个 b16，64 元素/寄存器）——落 UB 成连续 b16 流需要 2:1 打包，即 `DIST_PACK_B32`。加载侧对称：`LoadAlign<D_T, DIST_UNPACK_B16>` 把 UB 的 b16 解包进 fp32 lane 供 fp32 域计算（代码库 CastUp 扩位 VF 模式）。
- **实测判据的算式**：ST 对输出做 poison 初化后与 golden 位级比对。若 plain StoreAlign 在 b16 域写出交错/高位垃圾，poison 位不会被全零覆盖，mere/mare > 0；实测 fp16 与 bf16 的 EMPTY_R 用例 mere=0.000e+00、mare=0.000e+00、nan_inf_mismatch=0 且 2 次运行位级一致 → 写入字节全零且全覆盖。多 rep 循环（`repU16 = CeilDiv(count, VL)` 次、每次 `UpdateMask<T>(remaining)` 收敛掩码、偏移 `i × VL` 元素）在 fp32 域经 count=1024（16 rep）+ 尾块 952 实测位级正确。
- **为何"0 通过"仍能排除关键故障模式**：若 Duplicate 产出的寄存器高位含垃圾（b16 值仅占 fp32 lane 低 16 位），plain StoreAlign 会把垃圾写进交替位置，poison 比对立刻失败——实测全零排除了这一当时最担心的故障模式；但 0 的字节模式对"打包 vs 解包"两种正确语义不敏感（两种都写全零字节），故非零常量的一般性结论未由此实证（见失效边界）。

## 设计决策

- 按设计文档与 reduction 范式模板用 plain `StoreAlign`（`DuplicateEmptyROutputVfImpl` 逐字取自 Kernel.md §9.8），而不是盲目模仿同文件 GhxVfImpl 的 `DIST_PACK_B32`——依据是值的来源不同（Duplicate 直接产 b16 vs Cast 从 fp32 转出），而非"加 PACK 更保险"。
- 决策依据链：设计注释 `VL = 256/sizeof(T)`（b16=128 元素/寄存器）→ 原生打包寄存器推断 → ST 位级实测背书；实现前曾长期自我辩论是否要"保险起见"加 DIST_PACK_B32，最终以设计为准并通过实测验证该选择正确。
- `UpdateMask<T>(remaining)` 引用语义自动消耗 remaining，配合 `repU16` 外层循环实现尾块精确掩码——与主链 VF（fp32 口径 kRepF32）同一套模式，仅 VL 按 sizeof(T) 换算。

## 系统信息

- **toolchain/环境**：CANN 9.0.0-beta.2（ASCEND_HOME_PATH 实测）；Ascend 950 系列 arch35；AIV 向量寄存器 256B。
- **相关文件**：`op_kernel/arch35/thnn_fused_gru_cell_grad_kernel.h`（共享 VF 链：§9.8 新增 DuplicateEmptyROutputVf；既有 GhxVfImpl/CastUp 等提供 Cast+PACK 与 LoadAlign+UNPACK 对照模式）；`op_kernel/arch35/thnn_fused_gru_cell_grad_empty.h`（消费该 VF 的空张量分支 kernel）；ST 驱动 `tests/aclnn/test_aclnn.cpp` + `run.sh`（poison 初化 + 位级比对 + 2 次确定性检查）。
- **验证命令**：`cd operators && ./build.sh <pkg>` 编译安装 → `cd tests/aclnn && ./run.sh` 跑 ST（含 fp16/bf16 B=0 用例与 fp32 切分边界用例）。

## 完整代码

常量填充 VF（`op_kernel/arch35/thnn_fused_gru_cell_grad_kernel.h` §9.8 追加，verbatim；`kVlBytes = 256U` 为该文件既有常量）——**plain StoreAlign 路径**：

```c++
// kEmptyROutputValue：sum 归约的空 batch 退化值（空和 = 0）
constexpr float kEmptyROutputValue = 0.0f;

// DuplicateEmptyROutputVf（逐字取自 Kernel.md §9.8 / reduction-empty-kernel-template.md §5.1）
template <typename T>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(
    __ubuf__ T* dst, T value, uint32_t count, uint16_t repU16)
{
    constexpr uint32_t VL = kVlBytes / sizeof(T);   // 寄存器 VL（元素数）：fp32=64，b16=128
    AscendC::Reg::RegTensor<T> vReg;
    AscendC::Reg::Duplicate(vReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repU16; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::StoreAlign(dst + off, vReg, mask);
    }
}

// 调用侧：value = 0（kEmptyROutputValue, sum）；count = aUbFactor（单 chunk 元素数）
template <typename T>
__aicore__ inline void DuplicateEmptyROutputVf(__ubuf__ T* dst, T value, uint32_t count)
{
    constexpr uint32_t VL = kVlBytes / sizeof(T);
    const uint16_t repU16 = static_cast<uint16_t>((count + VL - 1) / VL);
    asc_vf_call<DuplicateEmptyROutputVfImpl<T>>(dst, value, count, repU16);
}
```

对照路径——同文件既有 GhxVfImpl 的缩位分支（verbatim 摘录）：**Cast 产出的 b16 必须 DIST_PACK_B32**：

```c++
template <typename D_T>
__simd_vf__ inline void GhxVfImpl(
    __ubuf__ float* goSlot, __ubuf__ float* zSlot, __ubuf__ D_T* ghxTile,
    uint32_t aUnit, uint16_t rChunkU16, uint16_t colRepU16)
{
    AscendC::Reg::RegTensor<float> goReg, zReg, tReg;
    AscendC::Reg::RegTensor<D_T> bReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t row = 0; row < rChunkU16; ++row) {
        const int32_t off = static_cast<int32_t>(row) * static_cast<int32_t>(aUnit);
        uint32_t remaining = aUnit;
        for (uint16_t rep = 0; rep < colRepU16; ++rep) {
            const int32_t c = static_cast<int32_t>(rep) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(goReg, goSlot + off + c);
            AscendC::Reg::LoadAlign(zReg,  zSlot  + off + c);
            AscendC::Reg::Mul(tReg, goReg, zReg, mask);            // ghx = go·z（fp32 域）
            if constexpr (std::is_same_v<D_T, float>) {
                AscendC::Reg::StoreAlign(ghxTile + off + c, tReg, mask);   // fp32：无缩位
            } else {
                AscendC::Reg::Cast<D_T, float, kCastTraitF32ToB16>(bReg, tReg, mask);
                AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    ghxTile + off + c, bReg, mask);
            }
        }
    }
}
```

## 预防

- 写 b16 落 UB 的 store 前，先判定值的来源：`Reg::Duplicate`/原生 T 域 → plain `StoreAlign`；`Cast`（fp32→b16）→ `StoreAlign<D_T, DIST_PACK_B32>`。
- 不要因"同文件别处用了 DIST_PACK_B32"就盲目照抄到 Duplicate 路径，也不要反向把 Cast 输出改成 plain store——两种形态各自绑定各自的寄存器表示。
- 每个 `__simd_vf__` 必须有 `asc_vf_call` 调用点（仓库 vf_reachability 门禁）。
- b16 新路径上线时优先补一条 poison 初化 + 位级比对的 ST 用例（0 填充能挡住高位垃圾/覆盖不全两类故障）。

## 验证信号

以下为该 VF 落地当次开发会话的实测结果（ST 驱动对输出 poison 初化、与独立 golden 位级比对、同输入 2 次确定性检查）：

- fp16 常量 0 填充位级正确：B=0,H=8,has_bias=true 用例 mere=0.000e+00、mare=0.000e+00、nan_inf_mismatch=0、determinism OK(bitwise,2runs)（来源平台：950；证据等级：tool_observed）。
- bf16 常量 0 填充位级正确：B=0,H=8,has_bias=true 用例 mere=0.000e+00、mare=0.000e+00、nan_inf_mismatch=0、determinism OK(bitwise,2runs)（来源平台：950；证据等级：tool_observed）。
- 多 rep + 尾块掩码循环位级正确（fp32 域）：H=2048（count=1024，repU16=16，无尾块）与 H=3000（尾 chunk valid=952）用例 mere=0.000e+00、mare=0.000e+00（来源平台：950；证据等级：tool_observed）。
- 对照路径（Cast + DIST_PACK_B32 的主链 VF）同批回归通过：Branch-0 ST 20/20、Branch-1 11/11（来源平台：950；证据等级：tool_observed）。
- vf_reachability 门禁：12 个 `__simd_vf__` 定义全部被 `asc_vf_call` 引用、13 处调用点，退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **值域边界（重要）**：实测常量恒为 0。0 的字节模式（全零）对"原生打包"与"fp32 lane 全零"两种正确语义不敏感，故"非零常量下 plain StoreAlign 的 lane 语义"**未实测**——非零 b16 常量填充（如填充 1.0）复用本模式前应先补位级用例。
- **规模边界**：b16 实测仅覆盖单 rep + 掩码（count=8 ≤ VL=128）；多 rep（count > 128）循环仅在 fp32 域实测（count=1024）。b16 多 rep 路径未实测。
- **API 边界**：结论限定 `Reg::Duplicate` 产出的寄存器；其他产 b16 的 Reg API（Select/Gather 等）的 lane 语义未验证。`LoadAlign<D_T, DIST_UNPACK_B16>` 与 `DIST_PACK_B32` 的对称性为代码库既有模式（文件读取佐证），未单独实验。"Cast 输出省略 DIST_PACK_B32 会位错"是既有已验证路径的隐含语义，本轨迹未做反向（省略）实验。
- **平台边界**：寄存器宽度按 950 AIV 256B（kVlBytes）推导；其他平台宽度不同时 VL 公式需重推——**未验证，推断可迁移**（Reg API 语义为 AscendC 通用机制）。

## 坏实践

- 盲目双向模仿：见同文件 Cast 路径有 `DIST_PACK_B32` 就给 Duplicate 路径也加（无依据的"保险"），或反过来认为 Cast 输出也能 plain store——两种都是把"值的来源决定 store 形态"误当成"store 形态二选一"。
- 只跑非 poison 的比对（例如输出 buffer 恰好初值为 0）就宣称 b16 store 位级正确——覆盖不全/高位垃圾两类故障会被初值掩盖。

## 误诊

- 实现前的疑虑"plain StoreAlign 对 b16 必须配 DIST_PACK_B32 否则写出交错数据"本身是被实测排除的误诊假设（对常量 0 填充路径）：fp16/bf16 用例位级全零通过。该假设的合理内核（Cast 路径确实需要 PACK）来自对照路径，两者不可混为一谈。
- "fp32 主链 VF 用 kRepF32=64 掩码口径，b16 也应按 64 掩码"——错：掩码/VL 口径按 `sizeof(T)` 换算（b16=128），本卡 VF 的 `UpdateMask<T>` 即按 T 口径。

## 未决事项

- 优先项：补一条**非零常量 + b16 多 rep（count > 128）**的位级用例，把"Duplicate→原生 b16 寄存器"的机制解释从推断升级为实测。
- `DIST_PACK_B32` 加在 Duplicate 路径上的行为（仍正确/报错/位错）未测——若要给"盲目照抄"定性为硬错误，需该反向实验。

## 后续增量（2026-09-17，并行策展批：R2——Cast 输出省略 DIST_PACK_B32 的反向实验闭合与两条修复配方）

R2（softmax 类算子分支 kernel ST 精度调试，Ascend950/arch35/dav-3510，CANN 9.2.0）：本卡失效边界中悬置的"Cast 输出省略 DIST_PACK_B32 会位错"由本轨迹实测闭合。

**症状签名（判别用）**：fp32 用例全部 PASS，fp16/bf16 用例 `PRECISION_FAIL (first fail @ elem 1)`——每个输出行第 0 个元素正确、第 1 个元素起全错；受影响为 fast path 小 shape 的全部 fp16/bf16 用例。kernel 收尾链为 VF 内 `Reg::Div` → `Reg::Cast<Tdst, Tsrc, kCastTraitB32ToB16>` → `Reg::StoreAlign(dstAddr, dstReg, aReg, mask)`（未带 StoreDist 模板参数，默认 dense 分布）。

**修复配方 A——把收窄 cast 移出 VF，用 UB 级 API（一轮验证）**：VF 内只做 fp32 Div（同型实例化），VF 外 `AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_RINT, cnt)` 完成布局正确的写回；fp16/bf16 失败点从 elem 1 移到 elem 3（残留为独立精度缺口），fp16_overflow 用例转 PASS。

**修复配方 B——留在 VF 内但显式打包分布（SDK SimpleSoftMax 的 StoreIfNeedCast 同款，二轮重写验证）**：

```cpp
template <typename D_T>
__simd_callee__ inline void StoreIfNeedCast(__ubuf__ D_T* dstUb,
                                             AscendC::Reg::RegTensor<float>& srcReg,
                                             AscendC::Reg::MaskReg& preg)
{
    if constexpr (sizeof(D_T) == 2) {
        AscendC::Reg::RegTensor<D_T> tmpReg;
        AscendC::Reg::Cast<D_T, float, kCastTraitF32ToB16>(tmpReg, srcReg, preg);
        AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstUb, tmpReg, preg);
    } else {
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(
            reinterpret_cast<__ubuf__ float*>(dstUb), srcReg, preg);
    }
}
```

配方 B 后同型 elem-1 损坏不再出现；选择依据：A 把布局责任交给 UB 级 Cast（一步到位、易审计，多一次缓冲往返），B 保持 VF 融合但必须正确携带 StoreDist 且 CastTrait 在 VF 外定义为全局 constexpr。

R2 验证信号（同一次 ST 精度调试，每轮改码后全量重跑）：

- 修复前：fp16/bf16 fast-path 用例 `PRECISION_FAIL (first fail @ elem 1)`（fp32 全 PASS）（来源平台：950；证据等级：tool_observed）。
- 配方 A 后：通过数 13/46 → 26/46，失败点从 elem 1 移到 elem 3，fp16_overflow 转 PASS（来源平台：950；证据等级：tool_observed）。
- 配方 B 后：elem-1 模式消失，最终 29/46 precision PASSED、46 aclnn-OK、0 ACLNN_FAIL（残留失败为行内中段独立精度缺口，非本根因）（来源平台：950；证据等级：tool_observed）。

R2 边界与坏实践补充：

- `DIST_PACK_B32` 只适用于 b32 寄存器收窄 b16 的存储；反向（b16 源 LoadAlign 的对应 LoadDist）未测。配方 A 的 UB 级 `AscendC::Cast` 无 SatMode 参数（默认行为），输出值域超出 b16 的场景需另行评估。
- 坏实践：`Reg::Cast` b32→b16 后直接 `StoreAlign`（不带 StoreDist）；看到 elem 1 起错先怀疑 CopyOut 的 DataCopyPad 参数或 tiling 行距——本例行距/搬运参数正确，错在 VF 内存储分布。
