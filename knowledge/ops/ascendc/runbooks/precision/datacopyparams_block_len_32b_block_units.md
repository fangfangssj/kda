---
type: "Runbook"
title: DataCopy blockLen 以 32B 块计非字节
description: DataCopyParams.blockLen 以 32 字节块为单位而非字节，直接填 count*sizeof(T) 会使 GM↔UB 搬运长度按 32 倍失真；改为 ceil(byteCount/32) 并把 InitBuffer 尺寸 32B 对齐。
tags: [task.debugging, topic.data_movement, topic.alignment]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# DataCopy blockLen 以 32B 块计非字节

## 现象

在 Ascend950（CANN 9.0.0-beta.2）上为 bessel_i1e Branch-0（FP32）kernel 编写 GM↔UB 搬运：`CopyIn`/`CopyOut` 用 `AscendC::DataCopyParams` 构造 DMA 参数，`blockLen` 一栏直接填了元素数×元素大小（字节数）。白盒 ST 12/14 精度失败（该宏观故障的主因是同文件内的 Chebyshev 系数错误）；在修复会话按 CANN 头文件核查搬运语义时发现：`DataCopyParams.blockLen` 的单位是 **32 字节块（ONE_BLK_SIZE）而不是字节**。直接传字节数意味着实际搬运 `blockLen×32` 字节——例如 1024 个 fp32 元素填 `blockLen=4096`，DMA 会搬运 4096 个 32B 块 = 128KB，从 GM 读入并写穿 UB 中后续相邻 buffer（scratch TBuf 等），属于被系数故障现象掩盖的独立搬运缺陷。修复会话将其与系数一并改正，并用"恒等搬运探针 + 多块大 shape 全量回归"确认修正后的搬运语义正确。

## 适用场景

- 工程形态：AscendC kernel 直调开发（`DataCopy(LocalTensor, GlobalTensor, DataCopyParams)` 形态的 GM↔UB 搬运）。
- API 路径：`AscendC::DataCopyParams{blockCount, blockLen, srcGap, dstGap}` + `pipe.InitBuffer`。
- 验证过的平台与 CANN 版本：Ascend950（AIV/VectorCore），CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有用 `DataCopyParams` 整块搬运且按"元素数×dtype 大小"思考长度的 AscendC kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`DataCopyPad` 搭配 `DataCopyParams` 的调用形态——同日另一轨迹（arch35/CANN 9.1.0）实测该组合下 blockLen 按字节解释（见失效边界）；`DataCopyExtParams`（blockLen 为字节数的另一族接口）；非 32B 块对齐约束的架构（其他 arch 的块大小常量不同，须按头文件核对）。

## 根因

`AscendC::DataCopyParams.blockLen` 按 32 字节块（`ONE_BLK_SIZE=32`，见 `asc/include/basic_api/kernel_struct_data_copy.h` 与 `kernel_utils_constants.h`）计数，而代码把字节数 `count * sizeof(T)` 直接塞进 `blockLen`，导致每次搬运的实际长度是预期的 32 倍，读写越过预期跨度。

## 解决配方

修复前（`op_kernel/bessel_i1e_kernel.cpp`，`blockLen` 直接传字节数；`Init` 中 buffer 尺寸也未做 32B 对齐）：

```cpp
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        auto xLocal = queIn.template AllocTensor<T>();
        AscendC::DataCopyParams cp{
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(count * sizeof(T)), 0, 0
        };
        AscendC::DataCopy(xLocal, xGm[offset], cp);
        queIn.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto yLocal = queOut.template DeQue<T>();
        AscendC::DataCopyParams cp{
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(count * sizeof(T)), 0, 0
        };
        AscendC::DataCopy(yGm[offset], yLocal, cp);
        queOut.FreeTensor(yLocal);
    }
```

```cpp
        uint32_t bufBytes = static_cast<uint32_t>(this->ubFormer * sizeof(T));
        pipe.InitBuffer(queIn, 1, bufBytes);
```

修复后（`blockLen` 改为字节数向上取整到 32B 块；`Init` 中 buffer 尺寸同样 32B 对齐；终版文件读回核对）：

```cpp
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        auto xLocal = queIn.template AllocTensor<T>();
        // blockLen is in units of ONE_BLK_SIZE (32 bytes), NOT bytes
        // Use ceiling division to handle non-aligned counts
        uint32_t byteCount = static_cast<uint32_t>(count * sizeof(T));
        uint16_t blockLen = static_cast<uint16_t>((byteCount + 31) / 32);
        AscendC::DataCopyParams cp{static_cast<uint16_t>(1), blockLen, 0, 0};
        AscendC::DataCopy(xLocal, xGm[offset], cp);
        queIn.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto yLocal = queOut.template DeQue<T>();
        // blockLen is in units of ONE_BLK_SIZE (32 bytes), NOT bytes
        uint32_t byteCount = static_cast<uint32_t>(count * sizeof(T));
        uint16_t blockLen = static_cast<uint16_t>((byteCount + 31) / 32);
        AscendC::DataCopyParams cp{static_cast<uint16_t>(1), blockLen, 0, 0};
        AscendC::DataCopy(yGm[offset], yLocal, cp);
        queOut.FreeTensor(yLocal);
    }
```

```cpp
        // Align buffer size to ONE_BLK_SIZE=32 for DataCopy DMA alignment
        constexpr uint32_t ONE_BLK_SIZE = 32;
        uint32_t rawBytes = static_cast<uint32_t>(this->ubFormer * sizeof(T));
        uint32_t bufBytes = (rawBytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        pipe.InitBuffer(queIn, 1, bufBytes);
        pipe.InitBuffer(queOut, 1, bufBytes);
```

## 设计决策

- **为什么用向上取整（ceil）而不是向下**：不足一个 32B 块的尾数也必须搬完；向下取整会丢尾块数据。
- **为什么 `InitBuffer` 尺寸也做 32B 对齐**：DMA 按 32B 块搬运，UB buffer 尺寸含非整块尾数时对齐后可保证块级读写不出界。
- **语义依据**：`kernel_struct_data_copy.h` 中 `blockLen` 为 `uint16_t`（块数上限 65535），配合 `kernel_utils_constants.h` 的 `ONE_BLK_SIZE=32`、`DEFAULT_DATA_COPY_NBURST=1`；单位结论另由修复后行为反证支撑（见验证信号）——若 `blockLen` 是字节语义，`ceil(bytes/32)` 只会搬运 1/32 的数据，多块大 shape 用例必然失败，而全量通过。
- **当前方案路线档位**：recommended_fastpath。

## 预防

- 写 `DataCopyParams` 时先在头文件确认所用接口族 `blockLen` 的单位（`DataCopy` 走 `DataCopyParams` 时为 32B 块；`DataCopyExtParams` 为字节；`DataCopyPad` 搭配 `DataCopyParams` 时按字节），不要凭"长度=字节数"的直觉填参。
- 对 `blockLen` 做上限核算：`uint16_t` 块数上限 65535（×32B ≈ 2MB/次调用），超限要分块搬运。
- 搬运正确性用恒等搬运探针一次性验证：Compute 旁路为 `y=x`，上板输出与输入一致即数据通路（含 `blockLen`）正确。

## 验证信号

以下信号来自 bessel_i1e Branch-0 kernel 的一次修复会话（`blockLen` 修正与系数修正同会话完成，搬运语义由下述信号单独锚定）：

- 修正后恒等搬运探针（Compute 旁路 `y=x`）：`got 0.327972412`，与该用例输入完全一致（该输入下期望输出 i1e(0.328)≈0.1197，旁路探针只验证搬运）（来源平台：950；证据等级：tool_observed）。
- 修正后全量白盒 ST：14/14 PASS，含 `01_f32_normal_128x256`（32768 元素、多 UB 块搬运）与 `0-D_f32_single_element`（单元素 4 字节，blockLen=1 块）——若 `blockLen` 为字节语义，`ceil(bytes/32)` 只搬 1/32 数据，多块用例必失败（来源平台：950；证据等级：tool_observed）。
- 独立验收复跑：14/14 PASS 且 `EXIT_CODE=0`（来源平台：950；证据等级：tool_observed）。
- 头文件常量：`ONE_BLK_SIZE=32`、`DEFAULT_DATA_COPY_NBURST=1`、`blockLen` 为 `uint16_t`（来源平台：宿主机 CANN 头文件阅读；证据等级：tool_observed）。

## 失效边界

- 向上取整使尾块实际搬运补齐到 32B，最多多写 31 字节；本开发会话的 GM 张量分配可容纳（全部用例通过），对紧凑分配/内存越界敏感的场景应改用 `DataCopyPad`（未在本开发会话验证）。
- 本卡证据采集时 tiling 侧 `ubFormer` 被限制 ≤1024 元素（单次搬运 ≤4096B、blockLen ≤128 块）；更大的 `blockLen`（如 `ubFormer=24576` 对应 3072 块）仍在 `uint16_t` 范围内，但该量级因随后另一根因（UB 超额 507035）未单独上板验证。
- `ONE_BLK_SIZE=32` 为本 CANN 版本/架构常量；其他架构或后续版本的块大小须按头文件重新核对。
- 单位结论绑定 `DataCopy` + `DataCopyParams` 组合；同日另一独立轨迹（arch35/CANN 9.1.0）实测 `DataCopyPad` 搭配同一 `DataCopyParams` 结构时 blockLen 按**字节**解释（2M~4M 元素 tile bit-exact 通过）——两族接口对同一结构体的单位语义不同，混用族接口时必须分别核对。

**未决项**：

- 字节语义的反事实实验（故意传字节数跑恒等探针观察越界冲刷表现）未在 trace 中执行；32 倍失真的具体越界表现（是否必然 507035 或静默污染）未观测。
## 坏实践

- `DataCopyParams.blockLen` 直接填 `count * sizeof(T)`（把块数当字节数，搬运长度放大 32 倍，冲刷 UB 相邻 buffer）。
- 只改 `CopyIn` 不改 `CopyOut`（两个方向同一语义）。
- `InitBuffer` 尺寸保留非 32B 对齐的尾数。

## 误诊

- 修复初期曾怀疑"构建未生效、上板跑的是旧二进制"，为此比对了装机 `.o` 的 sha256 与 kernel json 内嵌哈希（一致）并注入 +100 常数探针（got 100）——排除了部署问题，才把注意力放回代码语义；`blockLen` 单位错误正是在这之后的头文件核查中发现。

## 增量：两族结构体字段宽度/单位全景与"统一 Ext 组"配方（独立 trace，RoPE 类算子调试，arch35 + CANN 9.1.0 头文件实读）

单位结论的机制源头与结构体全景（`kernel_struct_data_copy.h` / `kernel_utils_base.h` / dav_3510 impl 头文件直读）：

- `DataCopyParams`（`kernel_struct_data_copy.h:48`）：`blockCount/blockLen/srcStride/dstStride` **全部 `uint16_t`**；`GetGMLen` 的 `uint16_t burstLenUnit = 32; if (isMovAlignIntri) { burstLenUnit = 1; }`——非 Pad 搬运路径 `blockLen` 单位是 **32B 块**，stride 同。
- `DataCopyExtParams`（同文件 :188，arch 3510/5102 分支）：`blockCount` uint16、`blockLen` **uint32（字节）**、`srcStride/dstStride` **int64（字节）**、多一个 `rsv`。
- `DataCopyPad` 内部 `CopyGmToUbufAlignV2` 的 `unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN`——**Pad 路径 blockLen 按字节解释**；即同一 `DataCopyParams` 结构体，`DataCopy`（非 Pad）按块、`DataCopyPad` 按字节，两族接口对同一结构体的单位语义不同。
- `DataCopyPad` 有两组互斥重载，copy/pad 参数必须**成组配对**：旧组 `(DataCopyParams, DataCopyPadParams)`、新组 `(DataCopyExtParams, DataCopyPadExtParams<T>)`；花括号 `{false, 0, 0, 0}` 可隐式转换为 `DataCopyPadParams`，混搭"旧组 copy + 显式类型 `DataCopyPadExtParams<D_T_X>` 变量"落在两组之间编译报 no matching function（该配对约束的编译失败形态与 ld.lld 掩盖症状另见[DataCopyPad 第4参类名 CTAD 失配卡](../compilation/datacopypad_4th_param_ctad_bare_braces.md)）。

推荐路线：**统一使用新组（字节语义）**，消除 32 倍换算这一整类错误（verbatim）：

```cpp
    // 构建 DataCopyExtParams（连续搬运 elementCount 个元素，blockLen 以字节为单位）
    __aicore__ inline DataCopyExtParams MakeLoadParams(uint32_t elementCount) const
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = elementCount * sizeof(INPUT_T);
        params.srcStride = 0;
        params.dstStride = 0;
        return params;
    }
```

调用点成对使用 `DataCopyExtParams` + `DataCopyPadExtParams<T>`（GM→UB）与三参数版（UB→GM）：

```cpp
    auto copyParams = MakeLoadParams(halfD_);
    DataCopyPadExtParams<INPUT_T> padParams{false, 0, 0, 0};
    DataCopyPad(cosInLocal, cosGm_[cosOffset], copyParams, padParams);      // GM→UB
    ...
    DataCopyPad(gmOut[rowOffset], yOutDeq, MakeStoreParams(D_));            // UB→GM（3 参数）
```

若坚持旧组（块语义），`blockLen` 必须填 `CeilDivision(bytes, 32)` 的块数——但新组字节语义与"元素数 × sizeof(T)"的心智模型一致，推荐统一新组。该开发会话的机制判别属 API 语义核对（507035 主因是独立的 Gather 越界读与向量对齐，仅修单位不足以消除当时全部 507035）。

增量验证信号：

- `kernel_struct_data_copy.h:48` `DataCopyParams.blockLen` 为 `uint16_t`；`:188-215` `DataCopyExtParams.blockLen` 为 `uint32_t`、arch3510 分支 stride 为 `int64_t`（来源平台：950；证据等级：tool_observed）。
- `kernel_utils_base.h` `GetGMLen` 双重载：`burstLenUnit = 32; if (isMovAlignIntri) { burstLenUnit = 1; }`——按路径切换单位（来源平台：950；证据等级：tool_observed）。
- `DataCopyPad`（dav_3510 impl）存在两组重载：`(LocalTensor, GlobalTensor, DataCopyParams, DataCopyPadParams)` 与 `(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams<T>)`（来源平台：950；证据等级：tool_observed）。
- 该开发会话后续所有 kernel 版本均沿用统一 Ext 组写法（来源平台：950；证据等级：tool_observed）。
