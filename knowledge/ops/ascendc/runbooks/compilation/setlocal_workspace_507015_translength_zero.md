---
type: "Runbook"
title: SetLocalWorkspace 507015：transLength 恒 0
description: L1 常驻 + CFG_ENVECND2NZ 下 SetLocalWorkspace 触发流同步失败 ret=507015（AIV error）；根因是 MultiCoreMatmulTiling 不计算 TCubeTiling.transLength（恒 0）致 ND2NZ workspace 空/不足，host tiling 手动补 transLength = baseM*baseN*sizeof(float) 后异常消除。
tags: [task.debugging, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# SetLocalWorkspace 507015：transLength 恒 0

## 现象

arch35（Ascend950PR / DAV_3510，MIX 模式 AIC+AIV，taskRation 1:2）三维卷积反传算子按修订设计实现 L1 常驻 Matmul（A/B 为 `MatmulType<TPosition::TSCM, CubeFormat::NZ, ...>`，`Matmul<..., CFG_ENVECND2NZ>`，`SetLocalWorkspace(cubeSharedUb)` 提供 ND2NZ 转换 workspace）。排除陈旧二进制问题、清理重建真实上板后，ST Real 5 个 L0 用例全部：

```
[FAIL] 流同步失败: ret=507015
```

（507015 = AIV error，aivmap 非零。）当时 kernel 状态：MatmulCompute 带 `if ASCEND_IS_AIV { return; }` 守卫（仅 AIC 执行），SetLocalWorkspace 在循环前调用，CopyND2NZOnTheFly 完成 GM(ND)→L1(NZ) 搬运。

## 适用场景

- 工程形态：算子仓开发 op_kernel，Matmul 高阶 API + `CFG_ENVECND2NZ`（`GetNormalConfig(false, false, true)`）+ A/B 常驻 TSCM/NZ，需要 `mm.SetLocalWorkspace(LocalTensor<uint8_t>)`。
- API 路径：`Matmul::SetLocalWorkspace`；Host 侧 `matmul_tiling::MultiCoreMatmulTiling::GetTiling` 产出的 `TCubeTiling.transLength`；`TBuf` cubeShared（`pipe.InitBuffer(cubeSharedTBuf, tilingData->cubeSharedUbSize)`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，MIX）+ CANN 9.1.0。
- 适用粒度：operator_family（所有使用 enVecND2NZ/SetLocalWorkspace 的 Matmul 算子）。
- 路线优先级：recommended_fastpath（enVecND2NZ 路线的必要配套修复）。
- 明确不适用 / 未验证场景：不启用 CFG_ENVECND2NZ/SetLocalWorkspace 的默认模板路径；A/B 在 GM 的路径（该路径另有 MIX 服务模型问题，另见 950 Matmul 选型卡）。

## 判别方法

二分定位法（每轮只改一个开关、清理重建、跑 ST Real 观察 ret）——本轨迹的完整判定链：

| 试验 | 结果 |
|---|---|
| 跳过整个 MatmulCompute（AIC 直接返回） | 无 507015（输出全 0） |
| 只保留 CopyND2NZOnTheFly，跳过 SetTensorA/B/IterateAll | 仍 507015 |
| 只保留 SetLocalWorkspace + Alloc/FreeTensor，跳过搬运与计算 | 仍 507015 |
| 跳过 SetLocalWorkspace，其余保留 | 无 507015 |

→ 触发点锁定 **SetLocalWorkspace**。

配套探针：Host tiling 的 fprintf 显示 `GetTiling ... transLen=0`（所有用例恒 0）；读 CANN 头文件 `asc/include/adv_api/matmul/constant_tiling.h`（159-164 行）确认 transLength 的正规计算公式：

```cpp
// C matrix ND2NZ
int32_t c1Length = 0;
if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::pos == TPosition::GM) {
    c1Length = baseM * baseN * GetBitSize<typename C_TYPE::T>() / ONE_BYTE_BIT_SIZE;
}
tiling.transLength = MaxValue<int32_t>(c1Length, biasLength);
```

而 `matmul_client.h` 中 ND2NZ 转换直接以 `cubeTiling.GetTransLength()` 作为 local workspace 尺寸使用（`localWorkspace_[cubeTiling.GetTransLength()]`）。

## 根因

`MultiCoreMatmulTiling::GetTiling` 产出的 `TCubeTiling.transLength` 恒为 0（constant_tiling.h 中 transLength 的计算路径未被 MultiCoreMatmulTiling 调用）。`transLength=0` → `cubeSharedUbSize` 只剩兜底值 → `mm_.SetLocalWorkspace(cubeSharedUb)` 给 Matmul ND2NZ 引擎设置了空/严重不足的 workspace → AIV 侧 ND2NZ 转换越界，触发 AIV 异常（流同步失败 ret=507015）。SetLocalWorkspace 本身语义没错，错在喂给它的 workspace 尺寸来源（transLength）未被引擎填充。

## 解决配方

Host tiling 在 GetTiling 之后手动补 transLength 并以其为准设置 cubeSharedUbSize。

before：

```cpp
    // cubeSharedUbSize：Matmul ND2NZ 引擎要求的 workspace（mm.SetLocalWorkspace 使用）
    // transLength 由 Matmul Tiling 引擎计算（启用 enVecND2NZ 时），单位为元素数
    int32_t transLen = cubeTiling.transLength;
    if (transLen <= 0) {
        transLen = 8192;  // 兜底默认值
    }
    int32_t cubeSharedUbSize = transLen * static_cast<int32_t>(FLOAT_SIZE);
```

after：

```cpp
    // cubeSharedUbSize：Matmul ND2NZ 引擎要求的 workspace（mm.SetLocalWorkspace 使用）
    // transLength 由 Matmul Tiling 引擎计算（启用 enVecND2NZ 时），
    // 但 MultiCoreMatmulTiling 不计算此字段（恒为 0），需手动设置。
    // 公式（参考 constant_tiling.h:159-164）：transLength = baseM * baseN * sizeof(float)（字节数）
    // 用于 C 矩阵 ND2NZ 转换的 UB workspace
    if (cubeTiling.transLength <= 0) {
        int32_t bM = cubeTiling.baseM > 0 ? cubeTiling.baseM : 16;
        int32_t bN = cubeTiling.baseN > 0 ? cubeTiling.baseN : 16;
        cubeTiling.transLength = bM * bN * static_cast<int32_t>(FLOAT_SIZE);
    }
    // transLength 已经是字节数，直接用作 cubeSharedUbSize
    int32_t cubeSharedUbSize = cubeTiling.transLength;
    // 限制不超过 UB 容量（248KB，留余量给其他 buffer，限制 80KB）
    constexpr int32_t CUBE_SHARED_UB_LIMIT = 80 * 1024;
    if (cubeSharedUbSize > CUBE_SHARED_UB_LIMIT) {
        cubeSharedUbSize = CUBE_SHARED_UB_LIMIT;
    }
    // vecSharedUbSize：ND→NZ 中转 UB（预留，按 32KB 固定）
    int32_t vecSharedUbSize = 32 * 1024;
```

注意配套条件（缺一不可，均为本轨迹最终生效组合）：Kernel 侧 `SetLocalWorkspace` 调用保留且 `CopyND2NZOnTheFly` 仅 AIC 执行（AIV 不做 GM→L1 搬运）；AIV 需执行到 SetLocalWorkspace（参与 MatmulClient 的 ND2NZ 协作）或按最终布局由 AIC 独占搬运——具体守卫布局见 arch35 MIX Matmul L1 常驻接入模式卡。

## 设计决策

- 修复位置选 Host tiling（补 transLength）而非 Kernel 侧硬编码 workspace 大小：transLength 语义上属于 cubeTiling（kernel 侧 `REGIST_MATMUL_OBJ` 会把 `&tilingData.cubeTiling` 传给 Matmul 对象），在源头补齐可同时服务 kernel 侧读取。
- `CUBE_SHARED_UB_LIMIT = 80KB`：UB 总量 248KB，还要容纳 grad/im2col 暂存与 Cast 双 buffer，取保守上限；该值为工程余量选择，非引擎约束。
- 80KB 截断与 `transLength` 真值的取舍：当 baseM*baseN*4 超过 80KB（如 baseM=baseN=128 的极大块）时截断可能仍不足——本轨迹用例未触及，列为失效边界。
- 被放弃的方案：不启用 enVecND2NZ（CFG_NORM + TSCM+NZ）——AIV error 消除但 Matmul 不产出（IterateAll 静默无结果），路线不可行。

## 预防

- 使用 enVecND2NZ + SetLocalWorkspace 时，UT/host 探针打印 `cubeTiling.transLength`，为 0 即视为未配置，必须在 host tiling 显式补齐。
- 把"SetLocalWorkspace 的 workspace 尺寸来源"列入 enVecND2NZ 路径的检查清单。

## 验证信号

本验证来自该算子 ST Real 的二分调试与最终回归（Ascend950PR 实机）。

- 修复前：ST Real 5/5 `流同步失败: ret=507015`；探针显示所有用例 `transLen=0`（来源平台：950；证据等级：tool_observed）。
- 二分链：跳过 MatmulCompute→无错；仅 CopyND2NZOnTheFly→507015；仅 SetLocalWorkspace+Alloc/Free→507015；跳过 SetLocalWorkspace→无错（来源平台：950；证据等级：tool_observed）。
- 修复后：CFG_ENVECND2NZ + SetLocalWorkspace + transLength 补齐的组合下，ST Real 不再出现 ret=507015（用例转为 MERE=1.00 输出全 0 的另一独立问题，见同轨迹 DynamicCompileStaticFlag 传参错位卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- transLength 手动公式按"C 为 GM/ND"情形推导（constant_tiling.h 条件分支）；C 输出到 UB（VECCALC/NZ）时 c1Length 计算不同，本卡公式未验证该情形。
- baseM/baseN 兜底 16 仅适用 half 级别的小块；`cubeSharedUbSize` 80KB 截断对更大 fractal 组合可能不足。
- 507015 的触发面很广（任何 AIV 异常都报此码）：本卡只覆盖"SetLocalWorkspace + transLength=0"这一成因；其他成因（如 workspace 指针无效）需另行判别。

**未决项**：

- `MultiCoreMatmulTiling::GetTiling` 为何不调用 constant_tiling 的 transLength 计算（引擎实现细节），未追根因；如需引擎原生支持，须向工具链侧确认。
- AIC/AIV 在 SetLocalWorkspace/ND2NZ 协作中的精确分工（AIV 是否必须执行到 SetLocalWorkspace）本轨迹只以守卫布局试验收敛，未获得机制层证据。
## 坏实践

- 给 SetLocalWorkspace 喂固定兜底值（如 8192 元素 × 4B）而不核对引擎真实需求——本轨迹 before 版本即如此，直接触发 507015。
- 见到 507015 就改守卫/删 SetLocalWorkspace 绕过（可消除报错但 ND2NZ 路径随之失效，输出全 0）。

## 误诊

- 中途假设"SetLocalWorkspace 在 CFG_ENVECND2NZ 模板下触发 AIV 通知、AIV 无对应处理逻辑导致异常"——方向错误：AIV 参与正是该模板的设计（MatmulClient ND2NZ 协作）；真实原因是 workspace 尺寸为 0 导致越界。
- 中途怀疑 `SetMatmulConfigParams` 的 enVecND2NZ 未生效（因 transLen=0）——被证伪：配置已生效，只是 transLength 字段不被 MultiCoreMatmulTiling 填充。
