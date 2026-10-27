---
type: "Runbook"
title: ASCENDC_TPL 多参数 TilingKey 声明与全量实例化
description: 用 ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL 在 tiling_key.h 声明并实例化多维 TilingKey 模板参数，host 侧 ASCENDC_TPL_SEL_PARAM 编码下发，从 binary json kernelList 反查实际实例化 key 验证。复现增量：模板机制的激活条件——声明头必须被 kernel 类头（.h）include，仅入口 .cpp include 时构建绿但 kernelList 仍单实例（机制未激活）；激活判据是 kernelList 实例数而非 .o 文件数或 simplifiedKey 字段。
tags: [task.implementation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# ASCENDC_TPL 多参数 TilingKey 声明与全量实例化

## 现象

一个 LSTM 类自定义算子（纯 AIV 路线）迭代一的 TilingKey 只有 2 个模板参数（`DTYPE_ID` 2bit + `HAS_BIAS` 1bit），`IS_BIDIR`/`HAS_PROJ`/`IS_BATCH_FIRST` 三个分支维度只是 TilingData 里的运行时标志。迭代二要求把 5 个维度全部转为编译期模板参数（fp32 全 16 种组合实例化），迭代三进一步要求全 dtype 48 种组合，同时保持迭代一的 key 值不变（fp32+bias=4、fp32+nobias=0、fp16+bias=5、bf16+bias=6）。当时代码状态：`op_kernel/arch35/<op>_tiling_key.h` 已用 ASCENDC_TPL 框架声明 2 参数，`op_host/arch35/<op>_tiling.cpp` 手动编码：

```cpp
    // 8. 设置 TilingKey
    // ASCENDC_TPL 编码: DTYPE_ID (bits 0-1) | HAS_BIAS (bit 2)
    uint32_t dtypeId = 0; // fp32=0, fp16=1, bf16=2
    if (dataType == ge::DT_FLOAT16) dtypeId = 1;
    else if (dataType == ge::DT_BF16) dtypeId = 2;
    uint32_t tilingKey = dtypeId + (hasBias ? 4U : 0U);
    context->SetTilingKey(tilingKey);
```

## 适用场景

- 工程形态：算子仓自定义算子开发（AscendC TPL 模板分发，多分支维度转编译期模板参数）。
- API 路径：`tiling/template_argument.h`（位于 `$ASCEND_HOME_PATH/x86_64-linux/asc/include/utils/tiling/`，配合 `-I${ASCEND_LINUX_DIR}/asc/include` 时 `#include "tiling/template_argument.h"` 可用，`asc/include/tiling/` 目录内是指向 `../adv_api`、`../utils` 等真实位置的符号链接）；宏族 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_UINT_DECL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_ARGS_SEL` / `ASCENDC_TPL_UINT_SEL` / `ASCENDC_TPL_SEL_PARAM`；`context->SetTilingKey`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510 / arch35）+ CANN cann-9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（该 CANN 版本下任何需要多维 TilingKey 分发的自定义算子）。
- 路线优先级：recommended_fastpath（宏框架统一声明/实例化/编码三处，规模可扩且可自动验证）。
- 明确不适用/未验证场景：不使用 TPL 框架、纯手动 `SetTilingKey` 的工程（思路可迁移但宏不可直接用）；其他 CANN 版本的宏集合与位宽规则未验证。单块多值 SEL 的坑见 [ASCENDC_TPL_DATATYPE_SEL 须每块单 dtype 卡](ascendc_tpl_datatype_sel_single_dtype_per_block.md)。

## 解决配方

三处协同 + 一处验证：

1. **声明与实例化**（`op_kernel/arch35/<op>_tiling_key.h`）：`ASCENDC_TPL_ARGS_DECL` 声明参数名/位宽/取值表（key 编码 = 各参数按声明顺序从低位拼接），`ASCENDC_TPL_SEL` 列出要实例化的取值组合——实例化范围就是编译产物范围，按 dtype 分组、每组一个 `ASCENDC_TPL_ARGS_SEL`。
2. **host 侧编码下发**（`op_host/arch35/<op>_tiling.cpp`）：`ASCENDC_TPL_SEL_PARAM(context, …)` 按声明顺序传各参数当前值，替换手动拼 key。
3. **kernel 入口模板**（`op_kernel/<op>_arch35.cpp`）：入口函数模板参数与声明一一对应，`if constexpr` 按 `DTYPE_ID` 分派元素类型。
4. **产物验证**：python 解析 `build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json` 的 `kernelList`，从 `kernelName` 尾缀提取实际实例化的 key 集合并按位解码——这是"实例化清单真的生效"的唯一客观证据（UT 断言 + ST 分支覆盖可作补充）。

## 完整代码

修复前（2 参数声明 + 手动编码，见现象中 tiling.cpp 段；tiling_key.h 修复前声明）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessLstm,
    ASCENDC_TPL_UINT_DECL(DTYPE_ID, 2, ASCENDC_TPL_UI_LIST,
                          BENCH_LSTM_DTYPE_FP32, BENCH_LSTM_DTYPE_FP16, BENCH_LSTM_DTYPE_BF16),
    ASCENDC_TPL_UINT_DECL(HAS_BIAS, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_FP32),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_FP16),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_BF16),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);
```

修复后 tiling.cpp 的 TilingKey 设置段（edit 后终版，verbatim）：

```cpp
    // 8. 设置 TilingKey (使用 ASCENDC_TPL_SEL_PARAM)
    // 编码: DTYPE_ID(2bit) | HAS_BIAS(1bit) | IS_BIDIR(1bit) | HAS_PROJ(1bit) | IS_BATCH_FIRST(1bit)
    // 向后兼容: 迭代一 TilingKey 值不变 (fp32+bias=4, fp32+nobias=0, fp16+bias=5, bf16+bias=6)
    uint32_t dtypeId = 0; // fp32=0, fp16=1, bf16=2
    if (dataType == ge::DT_FLOAT16) dtypeId = 1;
    else if (dataType == ge::DT_BF16) dtypeId = 2;
    ASCENDC_TPL_SEL_PARAM(context, dtypeId,
                          hasBias ? 1 : 0,
                          bidirectional ? 1 : 0,
                          hasProj ? 1 : 0,
                          batchFirst ? 1 : 0);
```

修复后 tiling_key.h 终版（48 种全实例化，verbatim，迭代三状态；迭代二状态为 fp16/bf16 组各只保留 `HAS_BIAS` 取 0,1、其余参数固定 0 的 2 种兼容组合）：

```cpp
#ifndef __BENCH_NO_HARNESS_LSTM_TILING_KEY_H__
#define __BENCH_NO_HARNESS_LSTM_TILING_KEY_H__
#include "tiling/template_argument.h"

// TilingKey 模板参数定义 — 迭代二：5 参数完整定义
// DTYPE_ID(2bit) | HAS_BIAS(1bit) | IS_BIDIR(1bit) | HAS_PROJ(1bit) | IS_BATCH_FIRST(1bit)
//
// 向后兼容（迭代一 TilingKey 值不变）:
//   fp32+bias+单向+无投影+非batchFirst = 0|4|0|0|0 = 4  (与迭代一一致)
//   fp32+nobias = 0|0|0|0|0 = 0  (与迭代一一致)
//   fp16+bias   = 1|4|0|0|0 = 5  (与迭代一一致)
//   bf16+bias   = 2|4|0|0|0 = 6  (与迭代一一致)
#define BENCH_LSTM_DTYPE_FP32 0
#define BENCH_LSTM_DTYPE_FP16 1
#define BENCH_LSTM_DTYPE_BF16 2

ASCENDC_TPL_ARGS_DECL(BenchNoHarnessLstm,
    ASCENDC_TPL_UINT_DECL(DTYPE_ID, 2, ASCENDC_TPL_UI_LIST,
                          BENCH_LSTM_DTYPE_FP32, BENCH_LSTM_DTYPE_FP16, BENCH_LSTM_DTYPE_BF16),
    ASCENDC_TPL_UINT_DECL(HAS_BIAS, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(IS_BIDIR, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(HAS_PROJ, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(IS_BATCH_FIRST, 1, ASCENDC_TPL_UI_LIST, 0, 1));

// 迭代三实例化范围: 48 种 TilingKey 全模板实例化
//   fp32: 全 16 种组合 (HAS_BIAS × IS_BIDIR × HAS_PROJ × IS_BATCH_FIRST)
//   fp16: 全 16 种组合 (迭代三扩展, 内部 Cast fp32 计算)
//   bf16: 全 16 种组合 (迭代三扩展, 内部 Cast fp32 计算)
//   总计: 3 × 16 = 48 种, 覆盖全 dtype × 全分支
ASCENDC_TPL_SEL(
    // fp32: 全分支组合 (16 种)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_FP32),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BIDIR, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_PROJ, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BATCH_FIRST, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // fp16: 全分支组合 (16 种, 迭代三扩展)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_FP16),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BIDIR, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_PROJ, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BATCH_FIRST, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // bf16: 全分支组合 (16 种, 迭代三扩展)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE_ID, ASCENDC_TPL_UI_LIST, BENCH_LSTM_DTYPE_BF16),
        ASCENDC_TPL_UINT_SEL(HAS_BIAS, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BIDIR, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_PROJ, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IS_BATCH_FIRST, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);
#endif
```

kernel 入口（`op_kernel/<op>_arch35.cpp` 终版，verbatim；tiling.cpp 侧还需在 include 列表追加 `#include "../../op_kernel/arch35/<op>_tiling_key.h"`）：

```cpp
#include "arch35/bench_no_harness_lstm.h"
#include "arch35/bench_no_harness_lstm_tiling_key.h"

// ============================================================================
// 纯 AIV 模式 Kernel 入口 — BenchNoHarnessLstm (arch35)
// ----------------------------------------------------------------------------
// 完全不使用 Matmul API, 不需要 REGIST_MATMUL_OBJ
// 矩阵乘法通过 Mul + ReduceSum 在 Vector 核上实现
// coreType = AIV (非 MIX)
//
// 迭代二: 5 模板参数 (DTYPE_ID, HAS_BIAS, IS_BIDIR, HAS_PROJ, IS_BATCH_FIRST)
//   - IS_BIDIR / IS_BATCH_FIRST: 运行时已支持, 模板参数用于 TilingKey 分发
//   - HAS_PROJ: 编译期分支, 新增投影 VectorMatmul 代码路径
//
// 核函数参数顺序: 输入 → 输出 → workspace → tiling (固定)
// weightHr: 投影权重 (projSize>0 时有效)
// ============================================================================
template <uint32_t DTYPE_ID, uint32_t HAS_BIAS, uint32_t IS_BIDIR, uint32_t HAS_PROJ, uint32_t IS_BATCH_FIRST>
__global__ __aicore__ void bench_no_harness_lstm(
    GM_ADDR x, GM_ADDR weightIh, GM_ADDR weightHh,
    GM_ADDR biasIh, GM_ADDR biasHh, GM_ADDR weightHr,
    GM_ADDR h0, GM_ADDR c0,
    GM_ADDR y, GM_ADDR hn, GM_ADDR cn,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessLstmTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessLstmTilingData, tilingData, tiling);

    if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_FP32) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<float, HAS_BIAS, IS_BIDIR, HAS_PROJ, IS_BATCH_FIRST> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    } else if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_FP16) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<half, HAS_BIAS, IS_BIDIR, HAS_PROJ, IS_BATCH_FIRST> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    } else if constexpr (DTYPE_ID == BENCH_LSTM_DTYPE_BF16) {
        NsBenchNoHarnessLstm::BenchNoHarnessLstm<bfloat16_t, HAS_BIAS, IS_BIDIR, HAS_PROJ, IS_BATCH_FIRST> op;
        op.Init(x, weightIh, weightHh, biasIh, biasHh, weightHr, h0, c0,
                y, hn, cn, workspace, &tilingData);
        op.Process();
    }
}
```

产物验证脚本（bash 内联 python，verbatim；key 解码规则随参数声明走）：

```bash
python3 -c "
import json, glob
all_keys = []
for f in sorted(glob.glob('build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json')):
    with open(f) as fh:
        d = json.load(fh)
    name = f.split('/')[-1].replace('.json','')
    klist = d.get('kernelList', [])
    keys = []
    for k in klist:
        kn = k.get('kernelName','')
        # extract trailing number after last _
        parts = kn.rsplit('_', 1)
        if len(parts) == 2:
            try:
                keys.append(int(parts[1]))
            except:
                pass
    keys.sort()
    print(f'{name}: {len(keys)} kernels, keys={keys}')
    all_keys.extend(keys)

all_keys.sort()
print(f'\nTotal: {len(all_keys)} kernel instantiations')
print(f'Keys: {all_keys}')

# Decode each key
print('\n=== TilingKey decode ===')
for k in all_keys:
    dtype_id = k & 0x3
    has_bias = (k >> 2) & 0x1
    is_bidir = (k >> 3) & 0x1
    has_proj = (k >> 4) & 0x1
    is_batch_first = (k >> 5) & 0x1
    dt = ['fp32','fp16','bf16'][dtype_id] if dtype_id < 3 else f'unk({dtype_id})'
    print(f'  key={k:2d}: {dt:5s} bias={has_bias} bidir={is_bidir} proj={has_proj} batchFirst={is_batch_first}')
"
```

## 设计决策

- **为何用 ASCENDC_TPL 框架而非继续手动位编码**：宏框架把"参数声明（位宽/取值表）—实例化清单—host 编码"三处统一在同一份声明上，key 编码由声明自动推导；手动编码每加一个维度都要同步改 tiling.cpp 拼接算式与 kernel 模板参数列表，两轮迭代扩展（2→5 参数、20→48 实例化）只改 tiling_key.h 的声明与清单各一处。
- **向后兼容优先、分两步放量**：迭代二先只放开 fp32 全 16 组合（fp16/bf16 保持迭代一的 2 种兼容组合），迭代三再放开到 48；每步都以 binary key 反查 + UT 分支断言 + ST Real 收口，避免一次改到位后 48 分支同时引入错误无处定位。
- **验证以产物为准，不以编译通过为准**：编译成功只证明"能编"，kernelList 反查才证明"该实例化的都实例化了、没多没少"；UT 参数化断言（如 fp32+bias+bidir→8、全开→60）与 ST 分支矩阵分析（13/16 由 ST 直接覆盖、其余 3 分支由 UT 补全到 100%）构成双保险。
- **统计口径陷阱**：binary 的 `.json` 文件按编译单元分组、不按 dtype 分组（迭代二每个 json 都含全部 20 个 key，首次统计出现"fp32 TilingKey 数: 20（期望 16）"就是混入了 fp16/bf16 兼容 key）——按 dtype 统计必须先解析 key 编码再分组，不能按文件分组。
- **被放弃的方案**：`IS_BIDIR`/`IS_BATCH_FIRST` 继续留在 TilingData 做运行时标志（迭代一状态）——分支无法编译期特化，UT 也无法按 key 断言分发正确性。
- **已知警告（容忍未修复）**：`ASCENDC_TPL_SEL_PARAM` 对 `batchFirst ? 1 : 0` 产生 narrowing conversion 警告（int→宏期望的无符号类型，模板参数处）；CANN 头文件 `kernel_type.h:101` 的 `FORMAT_CN macro redefined` 警告（编译命令行 `-DFORMAT_CN=FORMAT_ND` 与头文件定义冲突）。两者均为警告级，不影响产物。

## 系统信息

- toolchain：CANN cann-9.2.0、gcc 13.3.0、cmake 3.20.5；目标平台 Ascend950PR（ascend950 / DAV_3510 / arch35）。
- 关键路径：`op_kernel/arch35/<op>_tiling_key.h`（声明 + 实例化清单）、`op_host/arch35/<op>_tiling.cpp`（`ASCENDC_TPL_SEL_PARAM` 编码；include 追加 tiling_key.h）、`op_kernel/<op>_arch35.cpp`（5 参数模板入口 + `if constexpr` dtype 分派）、`op_kernel/arch35/<op>.h`（op 类模板增加对应布尔模板参数）。
- TPL 头文件：`$ASCEND_HOME_PATH/x86_64-linux/asc/include/utils/tiling/template_argument.h`（`asc/include/tiling/` 内为符号链接目录；op_host CMake 的 `${ASCEND_LINUX_DIR}/asc/include` include 路径已满足解析）。
- 产物：`build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json`（kernelList 即实例化清单）、3 个 kernel `.o`、`custom_opp_ubuntu_x86_64.run`。
- 验证命令：`bash build.sh --soc=ascend950 -j8`（全量实例化编译耗时上升，轨迹内按 15 分钟超时执行）。

## 验证信号

段首来源说明：本卡信号来自该算子迭代二/迭代三开发会话的编译产物与三级测试。

- 迭代二：`build.sh --soc=ascend950 -j8` 编译通过（仅 FORMAT_CN/narrowing 警告），binary json 解码 20 keys = fp32 16 种 [0,4,8,…,60] + fp16 {1,5} + bf16 {2,6}（来源平台：950；证据等级：tool_observed）
- 迭代二 UT 32/32 通过（含 6 条新 TilingKey 分支断言：fp32+bias+bidir→8、fp32+bias+proj→20、fp32+bias+batchFirst→36、全开→60 等）；ST Mock 30/30 + 1 跳过（来源平台：950；证据等级：tool_observed）
- 迭代三：binary json `kernelList` 条目数 48，fp32[0,4,…,60] / fp16[1,5,…,61] / bf16[2,6,…,62] 各 16 种；UT 148/148；ST Real（NPU）43/43 + 1 跳过，TilingKey 16 分支 ST+UT 联合覆盖 100%（来源平台：950；证据等级：tool_observed）

## 失效边界

- key 位宽受 TilingKey 总位宽限制：本例 2+1+1+1+1=6bit（最大 key=62）；维度更多时需确认平台 key 位宽上限。
- 实例化组合数 = 编译时间与 `.o` 体积：48 种全实例化后编译明显变慢（轨迹内需按 15 分钟超时执行）；按需放量（先主力 dtype 全分支、其余 dtype 兼容组合）是可回退的中间态。
- 该配方假设 host 侧能从输入属性推导出全部模板参数取值（dtype/bias/bidir/proj/batchFirst 均为运行时可知）；需要依赖 shape 数值的分支维度不适用此模式（应留在 TilingData）。
- 其他 CANN 版本的宏集合/位宽规则未验证（推断可迁移）。

**未决项**：

- `ASCENDC_TPL_SEL_PARAM` 的 narrowing conversion 警告未消除（可在传参前显式转换为宏期望类型，未尝试）。
## 坏实践

- 只看"build 成功"就认为实例化清单生效——清单写错（如某维度漏列取值）不会编译报错，只会静默少编分支，运行时命中未实例化 key 才暴露。
- 按 binary `.json` 文件分组统计各 dtype 实例数——文件按编译单元分组，混装多 dtype 的 key。

## 增量（激活条件：声明头必须被 kernel 类头 include）

一条独立 trace（BenchHarnessSoftmax，Ascend950PR + CANN 9.1.0，单 binary 运行时 if 分发升级为 ASCENDC_TPL_SEL 模板分发）补齐激活条件的失败形态："构建绿、机制死"。

**现象**：新建声明 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 的 `*_tiling_key.h` 并**只从 kernel 入口 `.cpp` include** 它——构建成功、无告警，但检查 vendor 部署产物发现每个 dtype的 `.o` 里 `kernelList` 仍只有 1 个内核实例（`…_0`）——模板机制没有激活，等于白改。把 include 挪进 kernel 类头文件（`<op>.h`，被 kernel 源包含的类定义头）后重建，`kernelList` 变为 2 个实例（每个 tilingKey 一个）、sha256 变化。

**激活条件（单句）**：构建系统通过扫描 kernel 类头文件（`<op>.h` 一类被 kernel 源包含的类定义头）发现 `ASCENDC_TPL_*` 声明来激活模板机制，仅在 `.cpp` 里 include 声明头不会生效（依据：tiling 模板编程参考文档"kernel.h 必须 include struct.h（缺这行，构建系统走 #define 包装器路径，模板机制不激活）"+ 本轨迹前后对比实测；扫描的确切实现未读到工具链源码层）。

**激活前后的 include 对比（verbatim）**：

```cpp
// ── before（模板机制不激活：仅入口 .cpp include）──
#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "bench_harness_softmax_tiling.h"

// ── after（kernelList 1→2 实例）──
#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "bench_harness_softmax_tiling.h"
#include "bench_harness_softmax_tiling_key.h" // ASCENDC_TPL_*（构建系统据此激活模板机制）
```

**验证方法（判断是否激活的金标准）**：

```bash
# 每个 dtype 的 .o 数不变（实例打包在同一 .o 内），看 kernelList 实例数：
cat $ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/<op_dir>/<OpType>_<hash>.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['kernelList'], d['sha256'])"
# 激活前：kernelList=[{"kernelName": "..._0"}]（单实例）
# 激活后：kernelList=[{"kernelName": "..._0"}, {"kernelName": "..._1"}]（每 tilingKey 一个实例），sha256 变化
```

判据必须是 `kernelList` 实例数，**不是** `.o` 文件数（实例本就打包在同一 .o 内），也**不是** `simplifiedKey` 的 `d=/p=` 字段（本轨迹实证其语义与 tilingKey 模板注册无对应关系，曾被误读为"未激活"）。

验证信号（复现增量，BenchHarnessSoftmax 空 tensor 分支模板化会话）：

- 改造前基线：每 dtype binary `kernelList` 仅 1 个内核实例（`…_0`），`sha256=0c0100f5...`（来源平台：950；证据等级：tool_observed）。
- 仅 `.cpp` include 声明头：构建 exit=0，`.o` 仍 3 个，`kernelList` 无新实例（机制未激活）（来源平台：950；证据等级：tool_observed）。
- kernel 类头补 include 后重建：`kernelList` 2 个实例（`…_0`/`…_1`），`sha256=5ccc8e1f...`（来源平台：950；证据等级：tool_observed）。
- 功能回归：`run.sh` 三分支全 PASS（empty 用例走 tilingKey=1 模板早退，非空走 tilingKey=0）；UT 114/114 无退化；清理重建复验一致（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 激活条件结论绑定本 CANN 版本（9.1.0）的构建系统扫描行为；其他版本扫描路径可能变化（未验证，推断机制相似）。
- 本例单 bool（2 组合）；多 bool 笛卡尔积组合的 SEL 未在该轨迹落地。
- `GET_TPL_TILING_KEY(1)` 的具体整数值未展开验证（黑盒使用宏返回值）；依赖宏保证与 `ASCENDC_TPL_SEL` 声明顺序一致。

坏实践（复现增量）：

- 只在 kernel `.cpp` include `_tiling_key.h` 就交付——构建绿、机制死，`kernelList` 不变，tilingKey 分发实际未生效（若此时 host 设了非 0 tilingKey，运行时将找不到 binary 报 361001）。
- 用 `.o` 文件数或 `simplifiedKey` 的 `d=/p=` 字段判断模板机制是否激活——两者都不反映 tilingKey 实例数。