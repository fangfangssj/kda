---
type: "Runbook"
title: "sub-kernel 后缀 _N 是 tilingKey 位编码非序号"
description: "编译产出的 .o 中 sub-kernel 后缀 _N 是 FastEncodeTilingKeyDirect 的位编码 tilingKey（T|E<<1|R<<2），非连续序号；_3 空位是设计性 gap，[0,4,2,1,5] 是 TPL_SEL 展开顺序副产物。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["dump_cce"]
---

# sub-kernel 后缀 _N 是 tilingKey 位编码非序号

## 现象

用 `inspect_kernel_binaries.sh` 检查 CANN 9.1.0 / Ascend950 上 EuclideanNorm 算子编译产出的 `.o`（路径 `opp/vendors/EuclideanNorm/op_impl/ai_core/tbe/kernel/ascend950/euclidean_norm/`），每个 `.o`（bf16/fp16/fp32 各一）含 5 个 sub-kernel，后缀依次为 `_0 _4 _2 _1 _5`，`_3` 缺失。用 `nm -S` 查符号表，5 个 `T` 类全局符号 `<hash>_<N>_mix_aiv` 中确实没有 `_3`。`readelf -x .ascend.meta.<name>` 各 per-kernel meta section 偏移 0x74 自报索引也与符号名后缀一致（0,4,2,1,5），确认 `_3` 不是读取遗漏而是真的不存在。源码 `euclidean_norm_tiling_struct.h:38-41` 的注释把 normal 写成 tilingKey 0+1、group 写成 3+4，与二进制实际后缀不符。

## 适用场景

- **工程形态**: 构建 / 调试（编译产出的 `.o` sub-kernel 符号结构与 host 侧 tilingKey 编码的交叉核验）
- **API 路径**: `GET_TPL_TILING_KEY(...)` 宏 → `AscendC::FastEncodeTilingKeyDirect(g_tilingDeclareParams, {...})`（`tiling/template_argument.h:291-325`）；`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_BOOL_SEL` / `ASCENDC_TPL_SEL`（`euclidean_norm_tiling_struct.h` / `euclidean_norm_apt.cpp`）；`SetTilingKey(...)`（`euclidean_norm_def.cpp`）
- **验证过的平台与 CANN 版本**: Ascend950 / CANN 9.1.0
- **适用粒度**: `operator_family`（任意使用 `ASCENDC_TPL_SEL` 多模板变体 + `GET_TPL_TILING_KEY` 位编码的算子均适用）
- **路线优先级**: `recommended_fastpath`（理解位编码后可从二进制反推 host tilingKey 设置逻辑）
- **不适用或未验证场景**: 不使用 `ASCENDC_TPL_SEL`（单模板无变体）的算子不产生多 sub-kernel，本判别不适用；Ascend950 以外平台未验证（机制推断一致）。

## 判别方法

### 触发

编译产出的 `.o` 中 sub-kernel 后缀 `_N` 不连续（出现 gap，如 `_0 _4 _2 _1 _5` 缺 `_3`），或后缀顺序非数值升序。

### probe（可执行诊断命令）

对任意 AscendC 算子编译产出的 `.o`：

```bash
# 1. 列出 sub-kernel 入口符号与大小（T 类全局符号）
nm -S <OpType>_<hash>.o | grep ' T '

# 2. 读 per-kernel meta section 的自报索引（偏移 0x74 的 u32）
for s in $(readelf -S <OpType>_<hash>.o | grep '.ascend.meta' | awk '{print $2}'); do
  echo "=== $s ==="
  readelf -x "$s" <OpType>_<hash>.o | head -5
done

# 3. 查 host 侧 tilingKey 编码实现
grep -n 'FastEncodeTilingKeyDirect' \
  <CANN>/x86_64-linux/asc/include/tiling/template_argument.h

# 4. 查算子的 TPL 声明顺序与 host SetTilingKey 调用
grep -n 'ASCENDC_TPL_ARGS_DECL\|ASCENDC_TPL_BOOL_SEL\|ASCENDC_TPL_SEL\|GET_TPL_TILING_KEY\|SetTilingKey' \
  <pkg>/op_kernel/*_tiling_struct.h <pkg>/op_host/*_def.cpp <pkg>/op_kernel/*_apt.cpp
```

### 原始观测

以 EuclideanNorm（bf16, hash 2bb5…）为例，`nm -S` 输出 5 个 `T` 符号：

| 符号 | .text 偏移 | 函数大小 | 索引 |
|---|---|---|---|
| `…_4_mix_aiv` | 0x00000 | 0x3908 | 4 |
| `…_1_mix_aiv` | 0x04c00 | 0x4b54 | 1 |
| `…_0_mix_aiv` | 0x0af00 | 0x3e0c | 0 |
| `…_2_mix_aiv` | 0x10200 | 0x075c | 2 |
| `…_5_mix_aiv` | 0x10d00 | 0x4684 | 5 |

`_3` 不存在。`.ascend.meta.<name>` 各 section 偏移 0x74 的 u32 与符号名后缀逐字节一致（0,4,2,1,5）。

host 侧 `ASCENDC_TPL_ARGS_DECL` 声明顺序为 `(templateType, isEmptyTensor, isTailR)`，三个均为 `ASCENDC_TPL_BOOL`（`bitWidth=ASCENDC_TPL_1_BW=1`）。

### 解释（interpretation，根因分析）

`FastEncodeTilingKeyDirect`（`template_argument.h:291-321`）按 `ASCENDC_TPL_ARGS_DECL` 声明顺序逐参数位移拼装 tilingKey。对 EuclideanNorm 的三参数遍历：

| 循环 i | param.name | param.bitWidth | totalBits(执行前) | line 313 执行 |
|---|---|---|---|---|
| 0 | templateType | 1 | 0 | `tilingKey \|= T << 0` |
| 1 | isEmptyTensor | 1 | 1 | `tilingKey \|= E << 1` |
| 2 | isTailR | 1 | 2 | `tilingKey \|= R << 2` |

故 `tilingKey = T | (E<<1) | (R<<2)`。后缀 `_N` 就是这个编码值：

| 后缀 | (T,E,R) | tilingKey | 语义 | host 设置点 |
|---|---|---|---|---|
| `_0` | (0,0,0) | 0 | Normal Tail-A | def.cpp isGroup=0,isTailR=0 |
| `_4` | (0,0,1) | 4 | Normal Tail-R | def.cpp isGroup=0,isTailR=1 |
| `_2` | (0,1,0) | 2 | EMPTY | def.cpp 显式 (0,1,0) |
| `_1` | (1,0,0) | 1 | Group Tail-A | def.cpp isGroup=1,isTailR=0 |
| `_5` | (1,0,1) | 5 | Group Tail-R | def.cpp isGroup=1,isTailR=1 |
| `_3` | (0,1,1) | 3 | **不存在** | — |

`_3 = (0,1,1)` 即 `isEmptyTensor=1 且 isTailR=1`。EMPTY 分支里 `isTailR` 无意义，host 在 `FillEmptyTilingData` 固定传 `isTailR=0`，故 key=3 永不产生，`ASCENDC_TPL_SEL` 也不实例化它。**设计性空位，非编译失败、非死代码消除。**

`[0,4,2,1,5]` 顺序是 `ASCENDC_TPL_SEL` 三个 `ARGS_SEL` 块按声明顺序展开、块内 `BOOL_SEL(v0, v1)` 按 v0→v1 展开后各实例的 tilingKey 序列（块1: T=0,E=0 → key 0,4；块2: T=0,E=1 → key 2；块3: T=1,E=0 → key 1,5）。runtime 不靠这个顺序调度——host `SetTilingKey(K)`，runtime 用 K 直接选 `…_K_mix_aiv` 符号；表顺序只是 codegen 的实例化遍历顺序，无调度语义。

**BOOL 与 UINT 的编码区别**：`ASCENDC_TPL_BOOL` 直接用值进入位域（`encodeVal = *argIter`）；`ASCENDC_TPL_UINT` 取值在 DECL 枚举列表中的**下标**（`index = iter - cbegin()`），不是值本身。故对 UINT 参数，后缀 `_N` 中的 N 是下标而非枚举值。

## 系统信息

- **工具链**: CANN 9.1.0, clang 15.0.5 (clang-5c68a1cb1231), LLD 15.0.5
- **目标平台**: Ascend950
- **相关文件**:
  - `<CANN>/x86_64-linux/asc/include/tiling/template_argument.h:291-325` — `FastEncodeTilingKeyDirect` + `GET_TPL_TILING_KEY` 宏
  - `<pkg>/op_kernel/arch35/euclidean_norm_tiling_struct.h:30-41` — `ASCENDC_TPL_ARGS_DECL` 声明顺序 + 注释（注释编号有误）
  - `<pkg>/op_host/euclidean_norm_def.cpp:245,420` — `SetTilingKey(GET_TPL_TILING_KEY(...))` 调用点
  - `<pkg>/op_kernel/euclidean_norm_apt.cpp` — `ASCENDC_TPL_SEL` / `ASCENDC_TPL_BOOL_SEL` 块
  - 编译产出 `.o`：`ascend950/euclidean_norm/EuclideanNorm_<hash>.o`

## 设计决策

- **BOOL 取值、UINT 取下标**：`FastEncodeTilingKeyDirect` 对 `ASCENDC_TPL_BOOL` 直接用原值，对 `ASCENDC_TPL_UINT` 用值在枚举列表中的下标。设计意图是让 BOOL 占 1 bit 紧凑拼装，UINT 用下标避免大枚举值占用过多位宽。
- **设计性空位**：当某参数组合在业务上不可能出现（如 EMPTY+TailR），host 不产生对应 tilingKey，`ASCENDC_TPL_SEL` 不实例化，二进制留 gap。这不是死代码消除，是 host 侧逻辑保证的。
- **三种顺序互不相同**：数值顺序（0,1,2,3,4,5）、`.text` 物理布局（地址升序：4,1,0,2,5）、分发表/`.ubuf`/脚本输出（0,4,2,1,5）。编译器把 `_4` 放在 `.text` 最前面（地址 0），但调度表把 `_0` 排第一。NPU kernel 不靠 ELF entry 起跳，靠 host `SetTilingKey(K)` 按调度表逐个调用 sub-kernel。

## 坏实践

- 看到 `_3` 缺失就怀疑编译失败或死代码消除：实际是 host 侧逻辑保证的设计性空位，应查 `ASCENDC_TPL_SEL` 块与 host `SetTilingKey` 调用点。
- 把 `[0,4,2,1,5]` 当成 runtime 调度顺序：实际是 codegen `ASCENDC_TPL_SEL` 展开顺序的副产物，runtime 用 tilingKey 直接选符号。
- 信任源码注释的 tilingKey 编号：`euclidean_norm_tiling_struct.h:38-41` 注释按"连续递增"心智模型写成 0,1,2,3,4，但位编码在 `_3` 处留空位，group 落在 1 和 5。**注释错误，二进制是权威事实。**

## 验证信号

- `nm -S` 输出 5 个 `T` 符号 `<hash>_<N>_mix_aiv`，N ∈ {0,1,2,4,5}，`_3` 不存在（来源平台：Ascend950；证据等级: tool_observed）
- `.ascend.meta.<name>` 偏移 0x74 的 u32 与符号名后缀逐字节一致（0,4,2,1,5）（来源平台：Ascend950；证据等级: tool_observed）
- `FastEncodeTilingKeyDirect` 位编码 `T|(E<<1)|(R<<2)` 与 5 个后缀 + `_3` 空位三重交叉验证闭合（来源平台：Ascend950；证据等级: artifact_verified）

## 失效边界

- Ascend950 以外平台（a2/a3）：`FastEncodeTilingKeyDirect` 与 `ASCENDC_TPL_SEL` 机制是 CANN 框架级，推断一致，但未验证，推断可迁移。
- CANN 大版本升级时 `template_argument.h` 的编码逻辑可能变化，需重新核对 line 291-321。
- 不使用 `ASCENDC_TPL_SEL` 多模板变体的算子（单模板）不产生多 sub-kernel 与 tilingKey gap，本判别不适用。