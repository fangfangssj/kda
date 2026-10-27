---
type: "Runbook"
title: "kernel.h 须 include struct.h 才能识别 TPL"
description: "给算子加 RANK 模板参数（ASCENDC_TPL_ARGS_DECL）后，若 *_kernel.h 未 include *_struct.h，asc_opc 在 apt.cpp 预处理 include 树里看不到 TPL 宏，退化成非模板调用，kernel 编译失败被 wrapper dummy-output 掩盖。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# kernel.h 须 include struct.h 才能识别 TPL

## 现象

在 AscendC regbase 算子 `MaskedFill`（ascend950 / CANN 9.1.0）的「TilingData 汇总结构体与 TilingKey 模板」阶段，开发者按 DESIGN §5.1 给算子加 RANK 模板参数：在 `op_kernel/arch35/masked_fill_struct.h` 用 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 声明 RANK∈{4,8} 两个 selection（RANK=4→tilingKey=0，RANK=8→tilingKey=1），并把 `op_kernel/arch35/masked_fill_apt.cpp` 的 `masked_fill` 改成 `template<int RANK> __global__ __aicore__ void masked_fill(...)`，host stub 改用 `ctx->SetTilingKey(GET_TPL_TILING_KEY(MASKEDFILL_RANK_4))`。kernel 主体仍为桩（DESIGN 要求只动结构体与 TPL 声明，kernel 维持桩）。

执行 `cd operators && ./build.sh masked_fill_package`：**退出码 0**。但部署后核对发现 kernel 二进制异常：

- build 日志出现 `[asc_opc wrapper] Real asc_opc failed (exit=1), generating minimal output... Created dummy output`；
- 链接阶段 `ld.lld: error: cannot open .../MaskedFill_XXX_0.o: No such file or directory`（COMPILE 阶段未产出 `_0.o`）；
- 预期 6 dtype × 2 RANK = 12 个 kernel binary，实际只生成 6 个（无 tilingKey 分档）；
- 生成 kernel.cpp 头部为 `#define TEMPLATE_PARAMS -1` / `#define TEMPLATE_PARAMS_LEN 0`，调用分支走 `#else masked_fill(...)`（非模板），与 `template<int RANK> masked_fill(...)` 定义不匹配 → bisheng `-c` 编译失败 → 无 `.o`。

即：**build 退 0 是假的**——asc_opc wrapper 在 real asc_opc 失败时生成 dummy `.o`/`.json`，掩盖了 kernel 编译失败。

## 适用场景

- **工程形态**：regbase 算子仓开发（ops-registry-invoke-glacier），host + kernel 联合 build（`./build.sh <pkg>` → cmake → op_build → asc_opc → FATBIN）。
- **API 路径**：`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_UINT_DECL` / `ASCENDC_TPL_SEL` / `GET_TPL_TILING_KEY`（声明于 `ascendc/host_api/tiling/template_argument.h`）；kernel 入口 `*_apt.cpp` 经 `*_kernel.h` 传递性 include TPL 文件。
- **验证过的平台与 CANN 版本**：Ascend950、CANN 9.1.0（路径 `/home/developer/Ascend/cann-9.1.0`，kernel 输出 `ascend950/`）。
- **适用粒度**：`operator_family`——任何采用 ASCENDC_TPL（RANK/dtype 等模板参数）多 binary 特化的 regbase 算子，首次给 kernel 加 TPL 时都会踩。
- **路线优先级**：`recommended_fastpath`——写 TPL 时同步让 `*_kernel.h` include `*_struct.h`，一次到位。
- **不适用或未验证场景**：非 TPL 算子（单一 kernel、无模板参数）不涉及；Ascend310/910B 等其他 SoC 的 asc_opc 行为未在本 trace 验证（推断一致，标"未验证，推断可迁移"于 `## 失效边界`）。

## 根因

asc_opc（Stage 4 编译器）**通过预处理 `*_apt.cpp` 的 include 树来发现 TPL 宏**（`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL`），而非独立扫描 `op_kernel/` 目录下的 `*_struct.h`。ascendc-build-system 的 build-pipeline 文档对此有明文记载（见 `## 验证信号`）：

- Stage 4：`asc_opc` → `检测 TPL: ASCENDC_TPL_ARGS_DECL → RANK∈{4,8}` → `模板特化: RANK=4 → sub-kernel _0, RANK=8 → sub-kernel _1`；
- TPL 声明文件约定为 `op_kernel/*_struct.h`。

参考算子 `xdivy_custom`、`scale_custom` 的 canonical 写法是 `*_kernel.h` 直接 `#include *_struct.h`，而 `*_apt.cpp` 只 include `*_kernel.h`——于是 TPL 宏经 `apt.cpp → kernel.h → struct.h` 传递性进入 apt.cpp 的预处理结果，asc_opc 即可识别。

本算子的 `masked_fill_kernel.h`（桩阶段骨架）只 include 了 `kernel_operator.h` 与 `masked_fill_tiling_struct.h`（TilingData 结构体），**漏 include `masked_fill_struct.h`（TPL 声明文件）**。于是 asc_opc 预处理 apt.cpp 时看不到 `ASCENDC_TPL_ARGS_DECL` → 判定 `TEMPLATE_PARAMS_LEN 0` → 走非模板分支调用 `masked_fill(...)` → 与 `template<int RANK>` 定义无法匹配 → bisheng 编译失败不产出 `_0.o` → FATBIN 链接报 `cannot open ..._0.o` → wrapper 兜底生成 dummy 输出 → build 退出 0。

根因句：**`*_kernel.h` 未 `#include *_struct.h`，导致 asc_opc 在 apt.cpp 的预处理 include 树里看不到 `ASCENDC_TPL_ARGS_DECL`，退化成非模板调用，kernel 编译失败被 wrapper dummy-output 掩盖。**

## 解决配方

修复点唯一且最小：在 `op_kernel/arch35/masked_fill_kernel.h` 的 include 区加一行 `#include "masked_fill_struct.h"`，使 TPL 声明经 `apt.cpp → kernel.h → struct.h` 进入 apt.cpp 的预处理树。kernel 主体仍为桩，仅加一个 include，不违反"kernel 维持桩阶段"约束。

before（修复前，`masked_fill_kernel.h` include 区，verbatim 取自 trace edit oldString）：

```c++
#include "kernel_operator.h"
#include "masked_fill_tiling_struct.h"

// Skeleton: no kernel class implemented yet.
```

after（修复后，verbatim 取自 trace edit newString）：

```c++
#include "kernel_operator.h"
#include "masked_fill_tiling_struct.h"
#include "masked_fill_struct.h"

// Skeleton: no kernel class implemented yet.
```

TPL 声明文件 `masked_fill_struct.h`（verbatim，trace 内 file_read 完整恢复）——它就是 asc_opc 要在 apt.cpp include 树里看到的 TPL 宏来源：

```c++
#ifndef MASKED_FILL_STRUCT_H
#define MASKED_FILL_STRUCT_H

// template_argument.h: provides ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_UINT_DECL /
// ASCENDC_TPL_SEL / GET_TPL_TILING_KEY / etc.
#include "ascendc/host_api/tiling/template_argument.h"

// The two allowed RANK values for the MaskedFill kernel TPL (DESIGN §5.1).
#define MASKEDFILL_RANK_4 4
#define MASKEDFILL_RANK_8 8

// Declares a TPL parameter named RANK for the MaskedFill operator.
ASCENDC_TPL_ARGS_DECL(MaskedFill,
    ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST,
        MASKEDFILL_RANK_4, MASKEDFILL_RANK_8)
);

// Enumerates the selections to emit at compile time. Each selection produces a
// separate kernel binary in the .run package:
//   first selection  -> RANK = MASKEDFILL_RANK_4 (tilingKey = 0)
//   second selection -> RANK = MASKEDFILL_RANK_8 (tilingKey = 1)
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, MASKEDFILL_RANK_4)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, MASKEDFILL_RANK_8))
);

#endif // MASKED_FILL_STRUCT_H
```

修复后 host `ctx->SetTilingKey(GET_TPL_TILING_KEY(MASKEDFILL_RANK_4))` 经 `g_tilingDeclareParams`（非 ASCENDC_DEBUG 路径由 `ASCENDC_TPL_ARGS_DECL` 定义、`FastEncodeTilingKeyDirect` 编码）正确产出 tilingKey=0/1，runtime 据 tilingKey 路由到对应 sub-kernel binary。

## 判别方法

当 `./build.sh <pkg>` 退出 0 但怀疑 kernel 实际未编译时，按顺序执行（命令 verbatim 取自 trace）：

1. 查 build 日志是否出现 dummy-output 兜底：

   ```bash
   grep -iE "Created dummy output|Real asc_opc failed|generating minimal output" /tmp/<OP>_build.log
   ```

   命中即 asc_opc wrapper 掩盖了 real asc_opc 失败。

2. 查生成 kernel.cpp 的 TPL 槽（`build/.../kernel_meta/..._<hash>_kernel.cpp` 头部）：

   ```bash
   head -60 <build>/<Op>_<soc>/kernel_0/kernel_meta_*/kernel_meta/<Op>_<hash>_<n>_kernel.cpp
   ```

   若 `#define TEMPLATE_PARAMS -1` / `#define TEMPLATE_PARAMS_LEN 0` → asc_opc 未识别 TPL。

3. 链接阶段报 `.o` 缺失：

   ```bash
   grep -iE "ld.lld: error: cannot open.*\.o" /tmp/<OP>_build.log
   ```

4. build 后用确定性脚本核对真实 binary（不依赖 build 退出码）：

   ```bash
   python3 .opencode/skills/ascendc-build-system/scripts/inspect_build.py operators/<pkg>/build
   bash   .opencode/skills/ascendc-build-system/scripts/inspect_kernel_binaries.sh $ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel
   ```

   `inspect_build.py` 报 `❌ INCOMPLETE` 或 `.o` 数 < 外层 dtype 数即失败；`inspect_kernel_binaries.sh` 报每个 `.o` 的 sub-kernel 数与 opParaSize，dummy binary 的 opParaSize 为 0。

5. 定位 TPL 可见性：确认 `*_apt.cpp` 的传递性 include 树里能看到 `ASCENDC_TPL_ARGS_DECL`——即 `*_kernel.h` 是否 `#include *_struct.h`。

## 验证信号

问题与验证来源：本 trace 的 MaskedFill 加 RANK TPL 后 build 退 0 但 kernel 异常，经上述判别定位到 kernel.h 漏 include struct.h，加一行后重建。

- TPL 声明文件 `masked_fill_struct.h` 完整含 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 两个 selection（来源平台：Ascend950；证据等级: artifact_verified，trace file_read 完整恢复 61 行）
- ascendc-build-system build-pipeline 文档明载 asc_opc Stage 4「检测 TPL: ASCENDC_TPL_ARGS_DECL → RANK∈{4,8}」「模板特化: RANK=4 → sub-kernel _0，RANK=8 → sub-kernel _1」，并提供 `inspect_build.py`/`inspect_kernel_binaries.sh` 核对脚本（来源平台：Ascend950；证据等级: artifact_verified）
- 修复 edit（`masked_fill_kernel.h` 加 `#include "masked_fill_struct.h"`）为 trace 内 verbatim tool_call（来源平台：Ascend950；证据等级: artifact_verified）

## 预防

- 写 TPL 算子时，把"`*_kernel.h` 必须 `#include *_struct.h`"作为 TPL 声明 checklist 的硬性条目，与 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 同步落地；不要因为"kernel 仍是桩"就推迟这个 include——asc_opc 在桩阶段就要识别 TPL 才能产出 12 个 binary。
- 验收脚本不要只看 `./build.sh` 退出码；把 `inspect_build.py`（构建期）与 `inspect_kernel_binaries.sh`（部署期）列为强制门，检查 `.o` 数 = dtype 数 × selection 数、sub-kernel 数、opParaSize 非零。

## 坏实践

- 看到 build 退出 0 就认为 kernel 编译成功——asc_opc wrapper 的 dummy-output 兜底会让真实失败隐形。
- 把 `ASCENDC_TPL_ARGS_DECL` 只放进 `*_struct.h` 却不令 `*_kernel.h` 引入它，指望 asc_opc 自己扫目录发现 TPL——asc_opc 只看 apt.cpp 的预处理 include 树。

## 误诊

排查初期曾怀疑：① struct.h 未被 build 系统扫描（实际 source struct.h 含 TPL，但生成的 `dynamic/op_kernel/.../struct.h` 副本里 `ASCENDC_TPL_ARGS_DECL` 出现 0 次——后查实那是 build 系统因未识别 TPL 而生成的退化副本，非源文件问题）；② CMake 缺 TPL 选项（对比 xdivy CMakeLists 一致，排除）；③ `ASCENDC_TPL_SEL` 仅在 `ASCENDC_DEBUG` 下定义导致 `g_tilingSelectParams` 缺失（实际非 debug 路径 `GET_TPL_TILING_KEY` 用 `FastEncodeTilingKeyDirect` 只依赖 `g_tilingDeclareParams`，且 TPL 发现是文本解析宏而非预处理符号定义，与 ASCENDC_DEBUG 无关）。最终定位到 include 树缺 struct.h。

## 失效边界

- Ascend950 / CANN 9.1.0 实测；Ascend310、Ascend910B 等其他 SoC 的 asc_opc 行为未在本 trace 验证，**未验证，推断可迁移**（asc_opc 的 TPL 发现机制是工具链通用的文本解析 + apt.cpp include 树预处理，与具体 SoC 无关）。
- 非 TPL 算子（单一 kernel、无模板参数）不涉及本坑。
- CANN 大版本若改变 TPL 发现机制（例如改为独立扫描 `*_struct.h`），本卡前提变化——届时须重新核验。