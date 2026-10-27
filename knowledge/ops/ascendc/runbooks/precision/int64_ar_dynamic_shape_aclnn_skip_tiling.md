---
type: "Runbook"
title: int64+AR 动态 shape aclnn 直调跳过自定义 tiling
description: 动态 shape 自定义算子 int64 输入 + AR 模式（dim=最内轴）+ R≤8 + 多输出时，aclnn 直调路径跳过自定义 tiling 与 kernel，ST mismatch 且实际值恒为固定小整数；含 tiling 落盘日志/常量输出探针/plog kernel 名三步定位法。
tags: [task.debugging, topic.precision, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# int64+AR 动态 shape aclnn 直调跳过自定义 tiling

## 现象

在开发一个自定义 argmax 算子（AscendC，目标 ascend950，注册了 5 种输入 dtype × AR/ARA 两种模式共 10 个 TilingKey 分支，动态 shape 支持 `DynamicShapeSupportFlag(true)`）的迭代三 ST 测试（C++ Real 模式，NPU 执行 + CPU golden 比对）时，159 条用例中有 2 条 int64 用例确定性失败，且无论怎么改 kernel 代码结果都不变：

- `[8,8]` shape、`dim=1`（dim 是最内轴，AR 模式，R=8，totalOutput=8）：`不匹配 7/8 (mismatch-rate=0.875000)`，`期望=7, 实际=6`；
- `[8,8,8]` shape、`dim=2`（AR 模式，R=8，totalOutput=64）：`不匹配 46/64 (mismatch-rate=0.718750)`，`期望=7, 实际=6`。

失败的值恒定为 6，每次运行完全相同。而同为 int64 的其他 case 全部通过：`[8] dim=0`（单输出 AR）、`[100,50] dim=1`（大 R 的 AR）、`[8,8] dim=0`（ARA 模式）、`[4,8] dim=1`（非连续输入的 AR）都正确。算子包已用 `bash build.sh --soc=ascend950` 重新编译并用 `custom_opp_ubuntu_x86_64.run --force` 重装，kernel cache 已清除，问题依旧。

## 适用场景

- 工程形态：算子仓自定义算子（OpDef/`_def.cpp` 注册）+ aclnn 两段式直调 ST 测试；kernel 为 AscendC 模板多 TilingKey 实现。
- API 路径：`asc_opc_tool`（opc 编译工具）生成的 kernel 配置 JSON（`int64Mode` 字段）、gert TilingContext 自定义 tiling 函数、aclnn 单算子执行路径。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR，SocVersion ASCEND950，DAV_3510/arch35）+ CANN 9.1.0。
- 适用粒度：`operator_family`（所有"动态 shape + int64 输入 + AR（最内轴归约）"的自定义算子在 aclnn 直调路径都可能命中）。
- 路线优先级：`safety_fallback`（无算子侧修复；规避路线见"解决配方"末尾，且需成对评估代价）。
- 明确不适用/未验证场景：GE 图模式（未验证框架在图模式下是否同样跳过 tiling）；静态 shape 算子（OPC 工具对非动态算子会按输入输出 dtype 计算 int64_mode，行为不同）；其他 CANN 版本未验证，推断新版本可能修复（未验证，推断可迁移）。

## 判别方法

三步探针把"精度错"区分为"kernel 算错"还是"kernel 根本没被调用"：

**第一步：tiling 函数加落盘日志，确认 tiling 是否被调用。** 在自定义 tiling 函数开头插入 `fopen/fprintf`（写 `/tmp/tiling_debug.log`），重建重装后跑 ST，再逐 case 核对 shape：

```cpp
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // Debug: log tiling invocation
    {
        FILE* dbg = std::fopen("/tmp/tiling_debug.log", "a");
        if (dbg) {
            std::fprintf(dbg, "TilingFunc called: dtype=%d, rank=%ld", static_cast<int>(dataType), static_cast<long>(rank));
            for (int64_t i = 0; i < rank; i++) {
                std::fprintf(dbg, ", dim[%ld]=%ld", static_cast<long>(i), static_cast<long>(inputShape.GetDim(static_cast<size_t>(i))));
            }
            std::fprintf(dbg, "\n");
            std::fclose(dbg);
        }
    }
```

在 tiling 末尾（`ASCENDC_TPL_SEL_PARAM(context, dTypeX, mode)` 之后）再补一条模式与关键字段的日志：

```cpp
    // Debug: log TilingKey and mode
    {
        FILE* dbg = std::fopen("/tmp/tiling_debug.log", "a");
        if (dbg) {
            std::fprintf(dbg, "  -> mode=%u, blkFactor=%u, realCoreNum=%u, ubFactorR=%u, bufNum=%u\n",
                         mode, blkFactor, realCoreNum, tiling->ubFactorR, tiling->bufNum);
            std::fclose(dbg);
        }
    }
```

**第二步：kernel 输出常量，确认 kernel 是否被执行。** 把怀疑路径的 kernel 输出写死为一个常量（如 42），重建重装后重跑：

```cpp
        outLocal.SetValue(0, static_cast<int64_t>(42));  // DEBUG: constant output
        PipeBarrier<PIPE_V>();
```

**第三步：分析 plog 中实际下发的 kernel 名。** 打开 `ASCEND_GLOBAL_LOG_LEVEL=1` 跑一次，然后在 `~/ascend/log/debug/plog/` 最新日志里统计 `StreamLaunchKernelV2` 行的 kernel 名后缀：

```bash
rg "StreamLaunchKernelV2.*BenchNoHarnessArgMax" ~/ascend/log/debug/plog/<最新>.log \
  | rg -o "BenchNoHarnessArgMax_[a-f0-9]+_[0-9]+|coreDim=[0-9]+" | paste - - | sort | uniq -c | sort -rn
```

kernel 名后缀即 TilingKey（本例 `_9`=INT64+AR、`_265`=INT64+ARA、`_0`=FLOAT+AR……）。

## 根因

本切片证据支持的结论：**框架在 aclnn 直调路径下，对"int64 输入 + AR 模式（dim=最内轴）+ R≤8 + totalOutput>1"的 shape 类跳过自定义 tiling 函数，因而自定义 kernel 二进制从未被下发，输出由框架默认实现产生（恒为 6）。** 机制层旁证：OPC 编译工具对动态 shape 算子强制 `int64_mode=False`（`asc_opc_tool/op_compilation.py` 的 `__set_int64_mode`：`int64_mode = False; if not op_info[OpcOptions.IS_DYNAMIC]: ...`，动态算子 `IS_DYNAMIC=True` 恒走 False 分支），kernel JSON 中 `int64Mode` 恒为 `false`。框架内部为何恰好按"R≤8 且 totalOutput>1"选择默认实现，未追到源码级机制（见遗留问题）。

## 解决配方

无算子侧修复。同一切片内验证过的规避/排查路线结论：

1. **不要把时间花在算子侧开 int64Mode 上**：注册 `SetOpSupportInfo` 回调（aclnn 直调不触发）、手改安装目录 kernel JSON 的 `int64Mode`（无效且被重装覆盖）都无效，详见同仓 negative_result 卡 [手改 kernel JSON int64Mode 无效且被重装覆盖](../compilation/int64mode_json_patch_ineffective.md)。
2. **不要在 aclnn 层用 `l0op::Reshape` 变通**（把 AR 拍平成 ARA 触发 `GetWorkspaceSize` 561103 失败），详见 [l0op::Reshape 变通 int64 AR 触发 561103 回退](../compilation/l0op_reshape_int64_ar_workaround_561103.md)。
3. 可行的工程处置：把这 2 条 case 作为已知框架限制记录（issue 文档 + 测试脚本 `known_limitation` 标注，不计入失败统计但保留失败事实），并推动升级 CANN 版本或在图模式使用该算子；先用上面三步探针确认"未调用"归属，避免在 kernel 精度逻辑上空耗。

## 设计决策

- 探针顺序先 tiling 后 kernel：tiling 日志能同时回答"是否调用"和"mode/切分参数是什么"，一次重建获得两个维度的信息；常量输出探针只回答"kernel 是否执行"，放在第二步用于交叉确认。
- 常量选 42：与任何合法 argmax 索引（0..R-1，R=8 时最大 7）和错误值 6 都不冲突，输出无歧义。
- plog 统计放在最后：plog 文件大、噪声多（含 Transpose 等框架插入的 kernel），只在前两步定位到"调用层"问题后用于核对实际选中的二进制。

## 系统信息

- 芯片：Ascend950PR（DAV_3510 / arch35，SocVersion ASCEND950）；CANN：cann-9.1.0。
- 相关文件：`op_host/arch35/*_tiling.cpp`（探针 1 插入点）、`op_kernel/arch35/*.h`（探针 2 插入点，Int64-AR 路径 `ProcessInt64AR` 的输出写回处）、`op_host/*_def.cpp`（OpDef 注册）、安装目录 `opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/{config, ascend950}` 下的 kernel JSON、`~/ascend/log/debug/plog/`（探针 3）。
- 构建入口：`bash build.sh --soc=ascend950 -j8` + `bash build/custom_opp_ubuntu_x86_64.run --force`；ST 入口：`tests/st/run.sh --real`。

## 验证信号

以下信号来自该 argmax 算子迭代三联调会话（NPU 实测，Ascend950PR + CANN 9.1.0），探针代码均按上文插入后重建重装取得：

- ST 失败现象：`[8,8] dim=1` 不匹配 7/8（mismatch-rate=0.875）、`[8,8,8] dim=2` 不匹配 46/64（mismatch-rate=0.71875），实际值恒为 6（来源平台：950；证据等级：tool_observed）。
- 探针 1 观测：tiling 日志中出现 `dtype=9, rank=2, dim[0]=8, dim[1]=8 -> mode=1`（这是 dim=0 的 ARA case）、`dtype=9, rank=1, dim[0]=8 -> mode=0`（[8] dim=0）、`dtype=9, rank=2, dim[0]=4, dim[1]=8 -> mode=0`（非连续 [4,8] dim=1），但**不存在** `[8,8] dim=1`（mode=0, R=8, totalOutput=8）与 `[8,8,8] dim=2` 的 AR 调用记录；111 次总调用中 int64（dtype=9）正常覆盖其余 case（来源平台：950；证据等级：tool_observed）。
- 探针 2 观测：Int64-AR kernel 改输出常量 42 后，`[100,50] dim=1`、`[8] dim=0` 等 case 输出变为 42（kernel 被调用），而失败 case 输出仍为 6（kernel 未被调用，走框架默认实现）（来源平台：950；证据等级：tool_observed）。
- 探针 3 观测：plog `StreamLaunchKernelV2` 统计出 10 个 kernel 名后缀与 10 个 TilingKey 一一对应（`_9` INT64+AR 共 25 次下发、`_265` INT64+ARA 19 次等），失败 case 所需的 coreDim=8 的 Int64-AR 下发记录缺失，只有 `_265`/`_256` 带 coreDim=8 的记录（ARA/FLOAT+ARA 路径）（来源平台：950；证据等级：tool_observed）。
- OPC 工具源码证据：`op_compilation.py` 的 `__set_int64_mode` 对动态 shape 算子（`IS_DYNAMIC=True`）把 `int64_mode` 置 False，安装后 kernel JSON 5 个 binList 条目全部 `"int64Mode": false`（来源平台：950；证据等级：tool_observed）。
- 最终回归：接受 2 条已知限制后 ST 157/159（98.74%），全量非限制 case 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 触发条件是"int64 + AR（dim=最内轴）+ R≤8 + totalOutput>1"的交集：单输出（totalOutput=1）、大 R、ARA 模式、非连续输入（框架先插 Transpose/Contiguous 改变路径）均不触发。
- 本卡结论绑定 aclnn 单算子直调路径 + CANN 9.1.0 + 动态 shape 算子；GE 图模式未验证（未验证，推断行为不同）；其他 CANN 版本未验证，推断可迁移。
- 探针 2 的常量输出会破坏所有走过该路径的用例（本切片中一度使通过率降到 83.02%），只能作为临时诊断手段，收尾必须移除。
- 若失败值不是恒定小整数、或 tiling 日志显示 tiling 已被调用，则不属于本卡场景，应回到 kernel 精度逻辑排查。

**未决项**：

1. （优先项）复现并追根因到机制层：定位 CANN 运行时在 aclnn 路径下按 shape 类选择默认实现的代码路径，确认"R≤8 + totalOutput>1"的判定来源，以及在更新 CANN 版本上是否已修复。
2. 验证 GE 图模式下同类 shape 是否触发相同跳过（图模式 tiling 日志 + kernel 下发统计）。
3. `SetOpSupportInfo` 在图模式是否被调用（本切片仅推断"图模式有效"，无日志证据）。
## 坏实践

- 只看 ST mismatch 就开始改 kernel 精度逻辑：本切片中 kernel 反复重构对这 2 条 case 零影响，因为 kernel 根本没执行。
- 依据"int64 的 tiling 调用次数为 0"这类未经验证的结论定向排查：实测 int64 大多数 case 的 tiling 都被正常调用，跳过只发生在特定 shape 类。

## 误诊

- 早期假设"int64 dtype 的自定义 tiling 调用次数为 0"——被 tiling 落盘日志证伪（大多数 int64 case 正常调用）（来源平台：950；证据等级：tool_observed，作为误诊记录）。
- 假设"失败由 `dynamicCompileStatic` 引起"——关闭该 flag 重建后失败不变（来源平台：950；证据等级：tool_observed，作为误诊记录）。
- 假设"kernel JSON 的 `int64Mode` 字段控制 aclnn 路径是否调自定义 tiling"——字段是 OPC 编译期产物，运行时手改无效（见同仓 JSON patch negative_result 卡）。

## 后续增量（2026-09-16，并行策展批：10 种修复尝试清单、supportSuperKernel 切换实验与 SetOpSupportInfo API 途径）

同算子迭代三联调的另一切片补充修复尝试全景与 API 途径（与本卡同 trace、同 2 条失败用例，证据互补）：

**1. 已尝试的修复（10 种，均无效）**：① MTE3→V `SetFlag/WaitFlag<HardEvent::MTE3_V>` 流水线同步（6 路径 18 处）——错值逐位不变（排除竞争）；② TBuf + PipeBarrier 替代 TQue 输入——不变；③ `GlobalTensor::GetValue` 绕过 UB/DataCopyPad 直读 GM——不变；④ `volatile` 防编译器优化——不变；⑤ kernel JSON 与 binary_info_config 手工改 `int64Mode=true`——框架不读取（见同仓 JSON patch 负结果卡）；⑥ ops-info `dynamicCompileStatic=false`——无效；⑦ `precision_reduce=false`——无效；⑧ kernel JSON `supportSuperKernel=0`——**其余 int64 AR 用例改走普通 kernel（输出常量 42 探针生效，证明超级内核切换对它们生效），但 [8,8] 仍输出 6**（来源平台：950；证据等级：tool_observed）；⑨ 输入缓冲加大到 256 字节——无效；⑩ 清 `~/atc_data/kernel_cache`——无 argmax 条目，排除编译缓存。

**2. `int64Mode` 的构建期来源与 API 途径**：kernel JSON 的 `int64Mode` 来自构建期 `support_info`（`ascendc_ops_config.py` 的 `add_dict_key(bin_info, 'int64Mode', support_info.get('int64Mode'))`），运行时改安装目录 JSON 无效；算子侧可经 `op_def.h` 的 `OpAICoreDef::SetOpSupportInfo(OP_CHECK_FUNC)` 注册 `get_op_support_info` 回调、返回 `{"int64Mode": true}`：

```cpp
static ge::graphStatus GetOpSupportInfo(const ge::Operator &op, ge::AscendString &result) {
    result = "{\"int64Mode\": true}";
    return ge::GRAPH_SUCCESS;
}
// 注册：this->AICore().SetOpSupportInfo(GetOpSupportInfo);
```

该修复的落地与验证不在本卡证据范围内（后续会话的任务结果摘要显示该回调在图模式生效、aclnn 直调模式不被调用，属转述线索，证据等级 agent_reported）。

**3. 验证信号（补充）**：

- 失败确定性：2 条失败用例错值跨运行与跨 kernel 修改逐位不变；修复会话入场时发现向量化源码处于无法编译的中间态（`maxLocal` 被引用 6 处但未声明），而联调日志显示"编译成功"且有 91/101 运行结果——该轮跑的是旧安装包（增量构建吞编译错误），先重建"源码↔binary 一致"基线再定位（来源平台：950；证据等级：tool_observed）。
- `supportSuperKernel=0` 后其余 int64 AR 用例改走普通 kernel（输出 42），[8,8] 仍输出 6（来源平台：950；证据等级：tool_observed）。
- 10 种尝试后 [8,8] 错值不变（来源平台：950；证据等级：tool_observed）。

**4. 边界补充**："R=8 + 多输出"是本轨迹观测到的触发条件边界；R 阈值的精确取值（≤8 还是更细）未做扫描实验（若后续复现可补 R=4/8/16/32 的最小触发实验）。