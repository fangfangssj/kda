---
type: "Runbook"
title: "REGIST_MATMUL_OBJ arch35 MIX AIC L1 崩溃"
description: Matmul 高阶 API REGIST_MATMUL_OBJ 在 arch35 (DAV_3510) MIX 模式 AIC 上构造即崩（l1 error 0xdffd0017fefe）；tiling/容量/配置排查矩阵全部无效，规避路线为纯 Vector contraction。
tags: [task.debugging, topic.implementation.api_layer.high_level_api, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: '2026-09-17T03:21:08Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# REGIST_MATMUL_OBJ arch35 MIX AIC L1 崩溃

## 现象

在 conv3d 反向滤波器自定义算子（im2col + Matmul 方案，ascend950 / Ascend950PR，arch35 MIX 模式 taskRation 1:2，CANN 9.1.0）完成 workspace 参数传递修复（tiling 已能读到实值 M=128 N=1728 K=4096、workspace 指针有效）之后，ST Real 5/5 用例仍然流同步失败 `ret=507015`。查 `~/ascend/log/debug/plog/` 最新日志（run 级 plog 无错误详情），得到 AIC 侧硬件异常：

```
[ERROR] ... PrintTaskErrorMsg:Task run failed, stream_id=61, pos=0, task_sn=2, sqe_type=0(aic), errType=0x1(task exception), sqSwStatus=0.
[ERROR] ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), ... there is an aicore error exception, core id is 0, error code = 0, dump info: pc start: 0x120041000000, current: 0x120041000ef0, ..., vec error info: 0, cube error info: 0, l1 error info: 0xdffd0017fefe, aic error mask: 0x395856, para base: 0x120000200000, ...
The extend info: errcode:(0) errorStr: timeout or trap error. subErrType: 0x4.
```

在 kernel 入口的 `REGIST_MATMUL_OBJ` 前后插入 `AscendC::PRINTF` 二分：`DBG before REGIST_MATMUL_OBJ` 打印出现、`DBG after REGIST_MATMUL_OBJ` 未出现——崩溃点就是 `REGIST_MATMUL_OBJ` 构造本身，尚未进入任何 Matmul 计算。小 shape 用例（1x1x1 卷积核，N=64）不崩但输出错（MERE=1.00e+00 / MARE=8.98e+00），大 shape 用例全部崩。

## 适用场景

- 工程形态：算子仓自定义算子（build.sh `--soc=ascend950`，aclnn 调用，ST Real 上板）。
- API 路径：Matmul 高阶 API `REGIST_MATMUL_OBJ`（= `REGIST_CUBE_OBJ`，见 CANN `asc/include/adv_api/matmul/matmul_intf.h`）、`matmul::MatmulType<TPosL1::TSCM, CubeFormat::NZ, ...>`、Host 侧 `MultiCoreMatmulTiling`（`SetDim`/`SetShape`/`SetSingleShape`/`SetBufferSpace`/`SetMatmulConfigParams`）。
- 验证过的平台与 CANN 版本：Ascend950（arch35 / DAV_3510，MIX 模式）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——arch35 MIX 模式下使用 Matmul 高阶 API 且带 TSCM/NZ 布局与 ASCENDC_TPL 模板机制的自定义算子。
- 路线优先级：banned（该配置下直接使用 REGIST_MATMUL_OBJ）；规避路线（纯 Vector contraction）为 safety_fallback，其相对代价见"性能影响"。
- 明确不适用 / 未验证场景：CANN 内置算子 swin_attention_score_quant、conv3d_backprop_filter_v2 arch35 在同平台使用 REGIST_MATMUL_OBJ 正常（C 类型 VECCALC/NZ 或 GM/ND+CFG_ENVECND2NZ、TILING_KEY_IS 模板机制），说明不是"arch35 一律不能用了"；其他 arch / CANN 版本未验证。

## 根因

`REGIST_MATMUL_OBJ` 在 arch35 (DAV_3510) MIX 模式下于 AIC 上构造即触发 L1 error（l1 error info: 0xdffd0017fefe），与 tiling 参数值、L0/L1 buffer 容量、Matmul 配置项均无关——机制层根因未追到（诚实声明）。

排查矩阵（每轮均 build exit 0 后上板复测，错误形态不变）：

| 尝试 | 实测结果 |
|---|---|
| 跳过 MatmulCompute（只保留 REGIST_MATMUL_OBJ 构造） | 仍 ret=507015 → 崩在构造点，与计算无关 |
| singleCoreK 4096 → 1024（aL1/bL1 各 32KB） | 仍崩，DBG tiling 确认参数已生效 |
| baseK → 256（L0A = 2×16×256×2 = 16KB << 64KB） | 仍崩 → 排除 L0A/L0B 容量溢出 |
| `SetDim(1)` 单核 | 仍失败（DBG 输出消失，行为不同但仍崩） |
| 移除 `SetMatmulConfigParams`/CFG_ENVECND2NZ（默认配置） | 仍崩 → 排除该配置项 |
| 移除 `SetLocalWorkspace` | 仍崩 |
| 核对 workspace 指针 | sysWs=0x12004c600000 非空、user=workspace 形参 → 排除系统 workspace 无效 |

排除项之外与 swin/conv3d_bp_v2（同平台正常）的剩余差异：本算子 C 类型为 `GM/ND/float`；swin 为 `VECCALC/NZ`（qkBmm）或 `GM/ND`+CFG_ENVECND2NZ（pvBmm）；本算子用 ASCENDC_TPL 模板机制，swin 用 TILING_KEY_IS + if-else；swin 有 `SetMaskNorm()` 调用而本算子无。`REGIST_CUBE_OBJ` 内部在 AIC 上创建 `KfcServer` 并 `server.Init(workspace)`、在 AIV 上创建 `KfcCommClient(workspace, ...)`（CANN `asc/impl/adv_api/detail/kfc/kfc_register_obj.h`），并未调用 `SetSysWorkspaceForce`——KFC 服务模型在 MIX 任务比下的初始化路径是剩余疑点。

## 解决配方

本卡为负结果：未找到使 REGIST_MATMUL_OBJ 在该配置下工作的修复。已采取的规避决策与路线：

1. **改用纯 Vector contraction（safety_fallback，已实施）**：删除全部 Matmul 相关代码（REGIST_MATMUL_OBJ、Matmul 类型、`CopyND2NZOnTheFly`、`SetLocalWorkspace`、L1 TQue、Host 侧 MultiCoreMatmulTiling），用 AIV 上的 `DataCopyPad` + `Cast` + `Mul` + `ReduceSum` 实现 C[M,N] = A[M,K]×B[K,N]（按 M 维多核切分、K 分块累加、fp32 累加器）。该实现 UT 10/10 通过；ST 验证被独立的用户 workspace 指针问题阻塞（见同轨迹的纯 AIV 写 workspace 507035 卡）。
2. 未走通/未尝试的替代方向（给后续接续）：对比 swin 完整 Host tiling（SetMatmulConfigParams 参数、TCubeTiling 字段值）；`ASCENDC_CUBE_ONLY` 纯 Cube 模式（AIV 阶段需另行处理）；回退 ASCENDC_TPL 为 TILING_KEY_IS + if-else 再试 REGIST_MATMUL_OBJ；手写 LoadData + MMAD + Fixpipe 低阶路径；咨询 CANN 框架专家确认 arch35 MIX 下 TSCM/NZ Matmul 的正确用法。

诊断方法（可复用）：在 `REGIST_MATMUL_OBJ` 前后插 `AscendC::PRINTF` 定位崩溃点；用如下探针打印 tiling 与指针（`(uint64_t)` 转换，__gm__ 指针不能 cast 到 void*）：

```cpp
    NsConv3dBpFilter::Conv3dBpFilterKernel<D_T_X, GROUPS_MODE> op;
    // 诊断：输出 tiling 关键值，验证动态模式下 tiling 数据是否正确读取
    if (GetBlockIdx() == 0) {
        AscendC::PRINTF("DBG tiling: M=%ld N=%ld K=%ld aL1=%d bL1=%d cubeSharedUb=%d sM=%d sN=%d sK=%d bM=%d bN=%d bK=%d usedCore=%d transLen=%d\n",
            (int64_t)tilingData.matmulM, (int64_t)tilingData.matmulN, (int64_t)tilingData.matmulK,
            tilingData.aL1BufSize, tilingData.bL1BufSize, tilingData.cubeSharedUbSize,
            tilingData.cubeTiling.singleCoreM, tilingData.cubeTiling.singleCoreN, tilingData.cubeTiling.singleCoreK,
            tilingData.cubeTiling.baseM, tilingData.cubeTiling.baseN, tilingData.cubeTiling.baseK,
            tilingData.cubeTiling.usedCoreNum, tilingData.cubeTiling.transLength);
    }
    // REGIST_MATMUL_OBJ 在 Init 之前（参考 swin:47-48，用栈上 op.pipe_）
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
    op.Init(x, grad, y, user, &tilingData);
    op.Process();
```

```cpp
    // REGIST_MATMUL_OBJ 在 Init 之前（参考 swin:47-48，用栈上 op.pipe_）
    if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF("DBG before REGIST_MATMUL_OBJ\n"); }
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
    if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF("DBG after REGIST_MATMUL_OBJ, before Init\n"); }
    op.Init(x, grad, y, user, &tilingData);
    if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF("DBG after Init, before Process\n"); }
    op.Process();
    if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF("DBG after Process\n"); }
```

## 设计决策

- 选择 Vector contraction 规避而非继续调 Matmul：REGIST_MATMUL_OBJ 崩溃经 6+ 轮排查矩阵无效，属构造层失败而非参数层可修；CPU Mock 5/5 已证明算法逻辑正确，生成阶段与 Cast 阶段在 AIV 上正常，替换 contraction 阶段风险最低。
- 性能取舍显式声明：性能暂不优化（后续迭代再回到 Matmul 优化），优先保证功能正确——Vector 路线相对 Matmul 路线预期更慢（无 Cube 算力），属 safety_fallback。
- 未选择 ASCENDC_CUBE_ONLY / 手写 MMAD：改动面更大，且当时 workspace 侧还有独立问题待闭合。

## 系统信息

- 工具链：CANN 9.1.0，build.sh `--soc=ascend950`；debug plog 路径 `~/ascend/log/debug/plog/`（run 级 plog 无该错误详情）。
- 相关文件：`op_kernel/arch35/*.h`（Matmul 类型与 KERNEL_TASK_TYPE）、`op_host/arch35/*_tiling.cpp`（MultiCoreMatmulTiling/TCubeTiling）、`op_kernel/*_arch35.cpp`（REGIST_MATMUL_OBJ 调用点）。
- 参考对照：CANN 内置 `swin_attention_score_quant.cpp`、`conv3d_backprop_filter_v2` arch35 实现。

## 验证信号

问题与排查来自同一次上板联调；下列均为修复 workspace 参数后、Matmul 崩溃排查期间的实测：

- debug plog AIC 异常：`aicore error exception, core id is 0, ... l1 error info: 0xdffd0017fefe, ... errorStr: timeout or trap error. subErrType: 0x4`（来源平台：950；证据等级：tool_observed）。
- PRINTF 二分：`[AIC Block 0/28] DBG tiling: M=128 N=1728 K=4096 ...` 与 `DBG before REGIST_MATMUL_OBJ` 出现、`DBG after REGIST_MATMUL_OBJ` 未出现（来源平台：950；证据等级：tool_observed）。
- 排查矩阵各轮：build exit 0 + ST `ret=507015` 不变（skip MatmulCompute / singleCoreK=1024 / baseK=256 / SetDim(1) / 去 CFG_ENVECND2NZ / 去 SetLocalWorkspace，逐轮上板复测）（来源平台：950；证据等级：tool_observed）。
- Vector contraction 规避实现 UT 10/10 通过（来源平台：950 主机侧；证据等级：tool_observed）；ST Real 结果受独立 workspace 问题阻塞，不能作为本路线的功能验证。

## 性能影响

- 规避路线（纯 AIV Vector contraction）相对 Matmul 路线（AIC Cube）为 safety_fallback：放弃 Cube 算力，矩阵乘在 AIV 上用 Mul+ReduceSum 标量级吞吐完成，预期显著慢于 Matmul 路线（参照路线：REGIST_MATMUL_OBJ + TSCM/NZ；来源平台：950）。本开发会话 内两条路线均未获得可比的上板性能数据（Matmul 路线无法运行、Vector 路线被 workspace 问题阻塞），故不给出具体倍数——采用前需自行实测成对对比。

## 失效边界

- 结论限定：arch35 (DAV_3510) + MIX 模式（taskRation 1:2）+ TSCM/NZ + GM/ND float C 类型 + ASCENDC_TPL 模板机制的组合；不能外推为"arch35 上 REGIST_MATMUL_OBJ 一律不可用"（CANN 内置算子同平台正常）。
- 排查矩阵未覆盖：swin 完整 tiling 参数逐字段对齐、TILING_KEY_IS 模板机制替换、ASCENDC_CUBE_ONLY 模式、CANN 版本差异。
- 小 shape 用例（N=64）不崩但输出错的现象未解释（该用例 AIC DBG 无输出），未闭环。
- 其他平台（未验证，推断不可迁移——需按平台实测）。

**未决项**：

- **优先项：复现并追 REGIST_MATMUL_OBJ 崩溃到机制层**——l1 error info 0xdffd0017fefe 的微架构含义、KFC 服务模型在 MIX 任务比下的初始化路径；建议按"与 swin 差异"清单（C 类型 / 模板机制 / SetMaskNorm / Host tiling 参数）逐项最小化复现。
- TILING_KEY_IS 替换 ASCENDC_TPL 后 REGIST_MATMUL_OBJ 是否恢复，未尝试。
- 小 shape 用例不崩但输出错的现象未解释。
## 坏实践

- 在 REGIST_MATMUL_OBJ 构造层崩溃时继续调 tiling 参数（singleCoreK/baseK/SetDim…）——本开发会话 6+ 轮证明该方向全部无效，构造失败不是参数失败。
- 用"跳过 MatmulCompute 还崩"推断计算逻辑有错——崩点在构造，与计算无关。

## 误诊

- 早期（workspace 修复前）该 L1 error 曾被"AIC 读到空数据、未触发"掩盖，表现为输出全 0——先修 workspace 才暴露本问题；两层问题先后出现，容易把新错误误判为修复引入的回归。
- swin 参考的 Host tiling 无法直接对比：swin 的 tiling 在 Python（tbe 框架 .so）里而非 C++，曾因此中断对比（过程事实，非结论）。
