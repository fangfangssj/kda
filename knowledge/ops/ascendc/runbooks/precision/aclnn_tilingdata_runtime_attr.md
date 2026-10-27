---
type: "Runbook"
title: "aclnn TilingData 缓存致 runtime attr 陈旧"
description: "aclnn L2 executor 对可 repeat 算子按 (算子, shape, dtype) 缓存整块 TilingData；若算子把每次调用都不同的 runtime scalar attr（如 beta/threshold/scalar）写进 TilingData，则同 (op,shape,dtype) 的第二批调用会跳过 tiling 函数、复用第一批的陈旧 attr，kernel 据此取错分支。判别=单跑通过 + 批跑失败 + 同 shape 不同 attr + tiling 函数日志两批只出现一次（或期望有限/实际 ±inf 的撞形签名）；规避=互异 shape 或同 shape 单 scalar，治本=AbandonCache(true)。复现增量：属性不改变输入 shape/dtype 的算子（如 isCausal 布尔属性）同样命中，且把属性编入 TilingKey / reset device / re-init 均无效，已验证的整 suite 规避是逐用例分进程。 复现增量：IntArray/slice 属性形态（begin/end/strides/masks）同样命中——PyTorch ST 同进程连续执行中同 shape/dtype 仅切片参数不同的用例输出沿用先前 begin=0；六步探针组（参数打印/-1 预填/直连两段式/换 shape 触发面）逐一排除适配层与内存复用，CANN 9.1.0。"
tags: ["task.debugging", "topic.precision", "topic.tiling", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-17T17:21:52Z"
updated_at: "2026-09-17T02:32:55Z"
platforms: ["950"]
aliases: ["repeat_cache"]
---

# aclnn TilingData 缓存致 runtime attr 陈旧

## 现象

在 Ascend950PR（RegBase / arch35、CANN cann-9.0.0-beta.2、build `--soc=ascend950`）上调试 `softplus_v2_grad` 算子精度。失败集中在 **scalar shape `()` 的 L1 case**：例如 L1_037（beta=-127.75, threshold=-0.0065, self∈(2,10), gradOutput∈(-1,1)）。正确结果应为 bx = beta*self = -127.75*(2..10) ∈ (-1277.5, -255.5)，全部 ≪ threshold(-0.0065) → sigmoid 分支 → grad_sig ≈ 0 → golden=0；但批跑时 actual ≈ gradOutput，即 kernel 取了**线性分支**（gradInput = gradOutput）。

关键差异：**L1_037 单跑通过；紧接 L1_001（同为 scalar shape `()`、但 beta=0.0023）后跑则失败**。把 L1_037 排在 L1_001 之后，actual 正好等于"用 beta=0.0023 算"的结果：bx = 0.0023*(2..10) ∈ (0.0046, 0.023) > threshold(-0.0065) → 线性分支 → gradInput = gradOutput。即第二批 case 复用了第一批的 beta/threshold。

## 适用场景

- **工程形态**：aclnn L2 接口 + RegBase 算子仓（op_host/op_api/op_kernel），kernel 经 aclnn executor 走。
- **API 路径**：`aclnnSoftplusV2GradGetWorkspaceSize` → `uniqueExecutor` → `ReleaseTo(executor)`；tiling 函数把 `beta`/`threshold` 写进 `TilingData`。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册平台 `950`），CANN cann-9.0.0-beta.2，arch 目录 arch35，build `--soc=ascend950`。
- **适用粒度**：`operator_family`——所有把 runtime scalar attr 写进 `TilingData`、且 aclnn L2 接口对同 (op,shape,dtype) 复用 executor 的算子都会踩。
- **路线优先级**：`recommended_fastpath`（在 op_api 的 `ReleaseTo` 前调 `AbandonCache(true)` 禁用 repeat/tiling cache，是最小改动且经本开发会话实测有效；根治方向是把 attr 移出 TilingData 改走 per-call workspace）。
- **明确不适用/未验证场景**：TilingData 内本就不含 runtime attr 的算子不适用；不走 aclnn L2 executor 的 kernel 直调场景不适用；a2/a3 未验证（推断可迁移，见 `## 失效边界`）。

## 根因

aclnn L2 executor 对"可 repeat"的算子以 **(算子, shape, dtype)** 为 key 缓存整块 `TilingData`。本算子把 `beta`/`threshold` 这两个**每次调用都不同**的 runtime scalar 写进了 `TilingData`。当第二批调用 shape/dtype 与第一批相同（scalar shape `()` + 同 dtype）时，框架**跳过 tiling 函数**、直接复用第一批的 `TilingData`——里面的 beta/threshold 是第一批的陈旧值，kernel 据此取错分支。shape 越是"撞 key"（scalar、或同 shape 反复跑）越容易触发；多 case 批跑必发，单跑看不出来。

机制链：`aclnnSoftplusV2GradGetWorkspaceSize` 内 `CREATE_EXECUTOR` → L0 op 经 `ADD_TO_LAUNCHER_LIST_AICORE(... OP_ATTR(beta, threshold))` 把 attr 喂给 launcher → tiling 函数把 beta/threshold 写进 `TilingData` → `CommonOpExecutorRun` 时若 executor 可 repeat 则按 (op,shape,dtype) 复用整块 `TilingData`（含陈旧 attr）→ kernel 读到陈旧 beta/threshold。

## 判别方法

触发（tool_observed）：批跑时同 (op,shape,dtype)、不同 runtime attr 的 scalar case 取错分支——单跑该 case 通过，紧接另一个同 shape 不同 attr 的 case 后跑则失败，且 actual 数值正好等于"用前一批的 attr 算"的结果。

探针：① 对怀疑的 scalar case 单跑，记录通过/失败；② 紧接一个同 scalar shape `()` 但不同 beta/threshold 的 case 后再跑同一 case，记录 actual；③ 在 tiling 函数入口加日志探针（如 `GetShapeAttrsInfo` 调用点打印一行），跑两批同 (op,shape,dtype) 调用，数日志出现次数。

观测（tool_observed）：单跑通过 + 接同 shape 不同 attr 后跑失败，actual 对应陈旧 attr 取值；tiling 函数日志在两批同 (op,shape,dtype) 调用中只出现一次——第二批根本没进 tiling 函数，TilingData（含 beta_scalar/threshold_scalar）被框架直接复用。

解释（interpretation）：aclnn L2 executor 的 repeat/tiling cache 以 (op,shape,dtype) 为 key 缓存整块 TilingData；runtime attr 写进 TilingData 后，同 key 的第二批调用复用陈旧 attr，kernel 据陈旧 beta/threshold 取错分支。在 op_api 的 `ReleaseTo(executor)` 之前调 `uniqueExecutor->AbandonCache(true)` 禁用 repeat/tiling cache、强制 tiling 函数每次调用都执行，可消除该症状（本开发会话实测：加探针后 tiling 函数从"两批只跑一次"变为"每批都跑"，beta/threshold 不再陈旧）。

### 增量（同 shape 异 scalar 的撞形签名与测试侧规避）

独立复现（TensorList 算子 `y_i = x1_i + (x2_i/x3_i)*scalar`，TilingData 含 `float scalar = 0.0f` 字段，ST 为覆盖特殊值用**相同 shape [1024]** 设计 5 个 scalar 用例 inf/nan/0/-1024/1024）：首个 scalar=inf PASS，其余 4 个同 shape 用例全 FAIL 且形态高度特征化——期望有限值/nan、实际全 ±inf 交替、符号随 x2/x3（即 kernel 实际执行 `x1 + (x2/x3)*inf`，首个用例的 scalar）。

判别法（区分本问题与真实计算错误）：

1. 看失败形态：期望有限值、实际全 ±inf（或实际值恰好等于"按另一个 scalar 计算"的结果）——不是精度漂移，是输入语义被替换。
2. 对照实验：把失败用例 shape 改成进程内未出现过的值重跑——通过即为缓存撞形，仍错则为算子真实缺陷。
3. 检查 TilingData：若属性字段进 TilingData 且失败用例与先前用例同 shape，即可判本问题。

测试侧规避（ST 用例设计规则，verbatim before/after）：**同一 shape 只绑定一个 scalar 值；要覆盖多个 scalar 特殊值时，为每个用例选互异 shape**（元素数可以只差几个，不影响覆盖语义）：

```cpp
// 修复前：5 个用例全部 shape [1024] → 后 4 个被首个 scalar=inf 的缓存 tiling 污染
cases.push_back({"FP32 scalar=nan [1024] listLen=1 (nan传播)",
     {{1024}}, std::numeric_limits<float>::quiet_NaN(), ...});
cases.push_back({"FP32 scalar=0 [1024] listLen=1 (零标量, y=x1)",
     {{1024}}, 0.0f, ...});
```

```cpp
// 修复后：互异 shape，注释固化规则
// 注意: 使用不同 shape 避免 tiling cache 复用旧 scalar
cases.push_back({"FP32 scalar=nan [1000] listLen=1 (nan传播)",
     {{1000}}, std::numeric_limits<float>::quiet_NaN(), ...});
cases.push_back({"FP32 scalar=0 [960] listLen=1 (零标量, y=x1)",
     {{960}}, 0.0f, ...});
cases.push_back({"FP32 scalar=-1024 [928] listLen=1 (边界值)",
     {{928}}, -1024.0f, ...});
cases.push_back({"FP32 scalar=1024 [896] listLen=1 (边界值)",
     {{896}}, 1024.0f, ...});
// x3 含零用例同理: [1024]→[864], [256,256]→[255,255]
```

## 验证信号

- 单跑 L1_037（scalar shape `()`、beta=-127.75）通过；紧接 L1_001（同 shape、beta=0.0023）后跑则失败、actual 对应陈旧 beta=0.0023 的取值（来源平台：950；证据等级：tool_observed）
- tiling 函数 `GetShapeAttrsInfo` 日志：修复前两批同 (op,shape,dtype) 调用只出现一次、修复（AbandonCache(true)）后每批都出现（来源平台：950；证据等级：tool_observed）
- AbandonCache(true) 后 scalar shape 失败用例数大幅下降、PyTorch ST 总通过率显著上升（来源平台：950；证据等级：tool_observed）

增量信号（TensorList scalar 撞形，Ascend950PR + CANN 9.1.0，真实 NPU ST 五轮对照）：

- 首轮：scalar=inf [1024] PASS，同 shape 的 scalar=nan/0/-1024/1024 四例全 FAIL，实际值全 ±inf、符号随 x2/x3（来源平台：950；证据等级：tool_observed）。
- 互异 shape 修复后第二轮起：5 个特殊 scalar 用例与 2 个 x3 含零用例全部 PASS（`MERE=0.00e+00, MARE=0.00e+00`，nan 传播用例按位一致）（来源平台：950；证据等级：tool_observed）。
- 第二次独立复现：65 例中唯一失败为 `FP32 非连续 1D [1024] scalar=0.5`（与 scalar=inf [1024] 撞形）全元素 ±inf；shape 改 [970] 后 65/65 = 100%（来源平台：950；证据等级：tool_observed）。
- TilingData 实读：结构含 `float scalar = 0.0f` 字段（tiling_data.h）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 平台：本开发会话只在 Ascend950PR（`950`）实测。a2/a3 上同 CANN aclnn L2 executor cache 机制预期一致，但**未验证，推断可迁移**；如要落到 a2/a3 须重测。
- 算子族：本卡根因（aclnn executor 按 (op,shape,dtype) 缓存整块 TilingData）对一切"把 runtime attr 写进 TilingData 且经 aclnn L2 接口 repeat 调用"的算子都成立，不止 softplus_v2_grad。
- 边界：TilingData 内不含 runtime attr 的算子不受影响；不走 aclnn L2 executor 的 kernel 直调场景不适用。
- （增量）scalar 撞形轨迹的结论为行为级（同 shape 撞形 → 旧 scalar 生效；互异 shape → 正确），tiling 缓存键的精确组成与失效时机未在机制层验证；不排除缓存键还含 dtype/TilingKey 等维度（同 dtype 内已复现，跨 dtype 未测）。"互异 shape"规避只对"进程内连续调用"的 ST 场景有效；若框架缓存跨进程或长期驻留（未验证），需另行确认。

### 增量（R4：isCausal 布尔属性撞缓存——无效手段清单与逐用例分进程规避）

独立复现 R4（稀疏 top-K 注意力算子，属性含 isCausal，causal 与非 causal 用例共用全部输入 shape/dtype，Ascend950PR + CANN 9.2.0，ST Real 模式）：

- 污染复现：causal 用例单独运行 `MERE=1.82e-04` 通过；放进完整套件、紧跟同 shape 非 causal 用例之后运行 `MERE=1.81`（因果掩码完全未生效的量级）失败。**诊断指纹**：期间多轮修改 kernel 代码（含一次因果掩码重构）后全量结果与修改前**逐位相同**——先做"改代码→结果逐位不变"检查可避免在 kernel 侧浪费排错轮次（安装与 kernel 更新已用 .o md5 与调试乘法双重排除）。
- 探针（与主卡探针③同构）：tiling 函数入口加 `fprintf(stderr, "[TILING] isCausal=%d ...")`，同 shape 的第一个（非 causal）调用打印日志，紧随其后的 causal 用例**不再打印任何日志**——缓存命中发生在 tiling 函数执行之前，复用缓存 TilingData（含非 causal 的 TilingKey 选择）按错误变体执行。
- **无效手段清单（均实测无效，勿再走）**：`aclrtResetDevice + aclrtSetDevice`；`aclFinalize + aclInit`；增大 tiling workspace 尺寸；把 IS_CAUSAL 移入/移出 TilingKey（模板参数与运行时字段两种配置下污染行为完全相同——缓存命中时 tiling 函数不执行，`ASCENDC_TPL_SEL_PARAM` 无从参与缓存 key；该结论与 staged《TilingKey 扩展绕不过 aclnn 执行器缓存》卡互证）。
- **已验证的整 suite 规避**：ST 主程序支持单用例过滤（`--case-id`），run.sh Real 模式逐用例分进程（Mock 模式无 NPU 可单进程）——分进程全量 24 用例 `PASS=24 FAIL=0`，causal 用例 `MERE=1.76e-04 MARE=5.03e-04 smallVal=0`。代价：每用例多一次进程启动 + ACL 初始化 + 上板开销，大批量用例评测总时长显著增加。
- 坏实践（R4）：把 isCausal 编码进 inputLayout 属性值（`layoutVal |= 2`）试图制造缓存区分——被 L0 参数校验以 `GetWorkspaceSize failed: 161002` 拒绝，属污染接口语义的 hack。

验证信号（复现增量，稀疏 top-K 注意力 ST 联调会话）：

- 全量套件中 causal 用例 `MERE=1.81 MARE=6.90e+04`，同用例单独运行 `MERE=1.82e-04`（来源平台：950；证据等级：tool_observed）。
- 探针观测：同 shape 第二个（causal）调用无任何 `[TILING]` 日志输出，tiling 函数未被调用（来源平台：950；证据等级：tool_observed）。
- 无效尝试（均实测无效）：aclrtResetDevice+aclrtSetDevice；aclFinalize+aclInit；workspace 尺寸调整；IS_CAUSAL 移出/移入 TilingKey（来源平台：950；证据等级：tool_observed）。
- 修复后：分进程全量 24 用例 PASS=24 FAIL=0，causal 用例 MERE=1.76e-04 MARE=5.03e-04 smallVal=0（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- R4 证据为 CANN 9.2.0；"属性不影响输入 shape/dtype 的算子必踩本坑"的推断对生产图执行同进程属性切换场景未验证。
- 缓存的确切存储位置未定位（`~/atc_data/kernel_cache` 与 `$ASCEND_HOME_PATH` 下均未见本算子条目），官方清除/禁用手段未找到；AbandonCache(true) 路线（主卡）在本 R4 trace 未尝试。
- 遗留（R4）：定位 ACL runtime tiling 缓存的存储位置与官方清除/禁用手段（列为优先项，给框架侧反馈）；缓存 key 精确包含哪些字段（format? attr?）仅有 shape/dtype 的排除性结论。

## 坏实践

- 看到 scalar case "单跑通过"就当环境/种子问题放行——掩盖了 cache 陈旧真因。
- 把 scalar 失败归因为"极端 beta 值导致溢出"——但 beta=-127.75、self∈(2,10) 算出 bx∈(-1277.5,-255.5) 远未溢出，且单跑通过，与"溢出"假设矛盾；真因是 cache 陈旧，非数值溢出。
- 没卸载旧 vendor 包就开测——曾误判为"两个 SoftplusV2Grad vendor 冲突"，卸载旧 vendor 后结果不变，排除该干扰项后才定位到 tiling cache
- （增量）用同一 shape 批量跑不同 scalar 特殊值，看到全 ±inf 就往 kernel 除零/溢出方向排查——方向错误，浪费定位时间。
- （增量）把失败用例的期望值"修正"为 ±inf 以求通过——掩盖框架缓存行为，属于测试侧放松。
- 自有算子把"同 shape 不同 attrs"用例的 input shape 改成全局唯一来"修绿"——测试侧放松，掩盖真实缺陷（生产同 shape 不同 attrs 调用仍会出错）；AbandonCache 修复后应保持原始 shape 全量通过才算闭环。黑盒无法改算子时的互异 shape 规避（见上文撞形增量）与该禁令的适用前提不同：前者是外部测试的用例设计规则，后者是算子所有者的修复验收标准。
- 把 `aclDestroyAclOpExecutor` 当"刷新 executor/tiling 缓存"的手段——实测既不刷新缓存又直接 double free（见 [CommonOpExecutorRun 加销毁循环 double free 卡](../compilation/common_op_executor_run_double_free.md)）。

## 增量（四条独立 trace：AbandonCache 落地代码、调用计数探针族与 fork 兜底细节）

另一独立轨迹并入四条独立 trace 的复现与修复（strided_slice / conv3d 反传 / scatter / grouped_matmul，Ascend950PR，CANN 9.1.0，属性分别为 masks 类 aclIntArray、dilation/pad、reduce 枚举、transpose_weight 布尔——**属性不进 TilingData 也一样命中**，只要 tiling/执行依赖属性）：

**1. AbandonCache 落地代码（此前本卡只有文字描述）**。在 op_api 第一段接口中、`CREATE_EXECUTOR()` 成功后插入（strided_slice 轨迹 verbatim before/after）：

```cpp
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 禁用 executor cache 复用：ACLNN executor cache 的 key 基于 (op_type, shape)，
    // 未将 attrs 纳入 cache key。本算子的 tiling 依赖 attrs，相同 shape 不同 attrs
    // 复用 cache 会导致错误的 tiling。
    // AbandonCache(true) 禁止当前 executor 复用历史缓存且不写入缓存。
    uniqueExecutor->AbandonCache(true);
```

scatter 轨迹的等价落点：L0 调用之后、`uniqueExecutor.ReleaseTo(executor)` 之前（两个位置均实测有效）；声明在 `aclnn/opdev/op_executor.h`（`make_op_executor.h` 已传递包含），`aclOpExecutor` 为公开结构体可直接调用。修复后四条 trace 全部转绿（65/65、5/5、范围用例 5/5 + UT 8/8、UT 12/12）。

**2. 调用计数探针族（把"缓存命中"变成可数事实）**：

- tiling 函数计数：TilingFunc 末尾加 `OP_LOGE(context, "TILING_DEBUG: ...")`，`ASCEND_SLOG_PRINT_TO_STDOUT=1` 运行后 `grep -c "TILING_DEBUG"`——行数 < 用例数且缺的恰是同 shape 用例即命中（conv3d 轨迹：5 用例仅 4 次；strided_slice 轨迹：23 用例批次仅 17 次）。
- L2 入口计数（更强，能发现"整层被跳过"）：`GetWorkspaceSize` 入口插 `printf("[DIAG-L2] ...")`/`fprintf(stderr,...)`，行数 < 用例数 ⇒ 未打点的用例没进 L2（scatter 轨迹：32 用例仅 4 行）。**注意用 printf 不用 OP_LOGI——L2 层的 OP_LOGI 不落 plog**（L0/tiling 的会落）；plog 佐证看算子实例编号跳号（`aclnn<Op>_0/_1/_2/_21`）。
- 顺序反转实验：把失败用例调到首位即通过、原首位反转为失败（"首跑者胜"）——是缓存/复用行为而非数据竞争或算子逻辑错的判别信号（conv3d 轨迹实测，MERE 随顺序反转翻转）。

**3. 无效手段补充（均实测无效，勿再走）**：环境变量禁用缓存——`ACL_OP_COMPILER_CACHE_MODE`、`ACL_OP_SELECT_CACHE_MODE` 等均针对算子编译缓存而非 executor cache，无公共开关；kernel/tiling 侧改造——`DynamicCompileStaticFlag` 切换、TilingKey 增加 dilation 编码、整包重构为 TILING_KEY_IS 运行时分发均无效（缓存命中时 tiling 函数根本不执行，kernel 侧任何改动不参与缓存 key；TilingKey 扩展路线另见 [TilingKey 扩展绕不过 aclnn 执行器缓存卡](../compilation/tilingkey_expansion_cannot_bypass_aclnn_cache.md)）；`aclrtResetDevice` 不清进程级 executor cache（与复现增量互证）。

**4. UT 盲区**：UT 的 TilingContext faker 按名直接构造 RuntimeAttrs（`contextFaker.Attr(attrName, ...)`），不走 executor/launcher 序列化，也不触发 executor 缓存——**UT 全绿不能证明 runtime 属性链路正确**，属性传递必须在 ST Real 验证。

**5. fork 兜底的工程细节（黑盒 ST 无法改 op_api 时）**：逐用例 fork 子进程完整走 `aclInit → aclrtSetDevice → aclrtCreateStream → RunRealTest → std::_Exit(退出码)`，父进程 `waitpid` 收结果。要点：**父进程不得初始化 ACL/持有 stream**（同轨迹早期在父进程已初始化 ACL 后 fork，子进程设备上下文失效、`aclrtMalloc` 失败——见 [同进程连跑 Segfault：勿销毁框架已缓存 tensor 卡](../compilation/aclnn_cached_tensor_pointer_destroy_segfault.md)）；子进程不做 `aclFinalize/aclrtResetDevice` 直接 `_Exit`；子进程输出需父进程 fork 前 `fflush(stdout)` + 子进程诊断走无缓冲 stderr + `_Exit` 前 `fflush`（stdout 接管道时全缓冲，`std::_Exit` 不走 atexit 冲刷）。结果回传用退出码（0=过/1=败/2=子进程 ACL 初始化失败）。grouped_matmul 轨迹：fork 改造后基础 37 条无 segfault、全量 687/687 无跨用例污染复发，"某维 128→129 躲缓存"的 shape 规避回退为 128 后仍通过。

### 复现增量验证信号

- conv3d 反传（CANN 9.1.0）：连跑时 dilation=2 用例（与 dilation=1 用例同 shape）MERE=7.38e+00、actual[0]=15.367188 与前一用例 expected[0] 逐位相同；单跑/调序到首位 MERE=0.00e+00，调序后失败者反转；tiling 日志 5 用例仅 4 次；AbandonCache(true) 落地重装后原顺序 5/5 全 PASS（MERE=0.00e+00），UT 12/12（来源平台：950；证据等级：tool_observed）。
- strided_slice（PyTorch ST，CANN 9.1.0）：修复前 62/65 通过、3 失败全为与先前用例同 shape 不同 attrs 的用例，Golden shape 65/65 一致；OP_LOGI 打点 23 用例批次仅 17 次 tiling；AbandonCache(true) 重编重装后 65/65 且连续两次稳定，C++ ST 回归 34/34（来源平台：950；证据等级：tool_observed）。
- scatter（CANN 9.1.0）：reduce=add 用例与 reduce=None 用例同 shape/dtype，kernel reduceMode_ 恒 0；L2 入口 stderr 打点 32 用例仅 4 行（全 reduce=0），plog 实例编号跳号；AbandonCache(true) + 干净重建后该用例 PASS（MERE=0.00e+00），穿刺 4/4、范围用例 5/5、UT 复验 8/8（来源平台：950；证据等级：tool_observed）。
- grouped_matmul（CANN 9.1.0）：同进程连跑同 shape 不同 transpose_weight 用例精度失败/段错误；改逐用例 fork 隔离（父进程不初始化 ACL）后基础 37 条无 segfault、迭代验收 154/154、191/191、全量 687/687 无跨用例污染复发；shape 规避（某维 128→129）回退后仍通过（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 四条轨迹均为 CANN 9.1.0；与本卡先前增量 的 9.2.0 证据合并后，结论跨 9.1.0/9.2.0 成立，但缓存 key 是否含 format 等其他维度仍未源码级确认。
- L2 入口计数与 plog 路由事实（L2 OP_LOGI 不进 plog）为 CANN 9.1.0 实测；其他版本需复核。
- fork 兜底不修复污染本身，只把失败限制在单用例；每用例一次进程创建 + ACL 初始化的代价未做逐用例计时对比（相对倍数未验证）。

## 后续增量（2026-09-17，并行策展批：IntArray 切片属性形态 + 六步排除探针组）

StridedSlice 类自定义算子（aclnn 两段式，Ascend950PR / CANN 9.1.0 + torch 2.7.1 + torch_npu）的 PyTorch ST 轨迹把命中面从标量属性扩展到 **IntArray 切片属性**，并补齐一套可复用的排除探针组：

**1. 触发形态**：PyTorch ST 同进程连续执行 65 条用例，3 条失败形态完全一致——**输出从 begin=0 开始，而非调用参数指定的 begin**（如 x 为 shape=(64,) fp16、begin=[32]、end=[64]、end_mask=1，语义 x[32:64]，期望 [32,33,…,63]，实际输出 [0,1,…,31]；负索引 begin=[-32]→32 与 2D begin=[-4,-4]→[4,4] 同形态）。其余 62 条通过；三条失败用例各自单条运行（`--case`）全部通过；C++ 原生 ST 34/34 通过——但其同名用例输入 shape 用的是 (80,) 而非 (64,)（历史差异，天然规避缓存命中）。

**2. 六步排除探针组（全部同进程实测、可复现）**：

1. **执行模式隔离**：单条运行通过；任意"1 条前序 + 失败用例"的两两组合通过；按测试顺序连跑 5 条（其中第 4 条 begin_mask=1 生效 begin=0、输出 shape 与失败用例相同）后第 5 条稳定失败，连续 5 次无一例外——小序列复现不稳定本身是缓存类问题的特征。
2. **参数边界探针**：适配层 acl_call 内 fprintf 打印实际传入 GetWorkspaceSize 的参数：`begin=[32,] end=[64,] strides=[1,] bm=0 em=1` 正确——排除适配层传参。
3. **写入探针**：输出 tensor 用 `torch::full(y_shape, -1.0)` 预填充后，算子仍写出 [0,1,2,3,4]（非 -1）——排除"输出内存未写/复用旧值"，算子确实按 begin=0 的语义写入了。
4. **通道对照**：绕过 torch_npu 的 RunOpApiV2，改为与 C++ ST 完全相同的直连两段式（GetWorkspaceSize → Execute → `aclrtSynchronizeStream` → 销毁参数）后，5 用例序列仍然失败——排除适配层调度机制。
5. **同系列排除**：`RunOpApiV2(..., true)`（sync=true）、把 GetWorkspaceSize 移入 acl_call、每用例前 `torch.npu.synchronize()` + `torch.npu.empty_cache()`，均无效——排除异步执行/参数提前销毁/caching allocator。
6. **触发面探针**：同一前序序列下，同参数但 shape=(80,) 的用例通过，shape=(64,) 失败——触发条件与"输入输出 shape/dtype 完全相同、仅 begin/end/strides/mask 等 IntArray/scalar 属性不同"强相关，坐实"缓存键含 shape 不含 attrs"。

**3. 归因与边界**：连续执行中框架按（op_type、shape/dtype）命中并复用先前调用的执行器/tiling 状态，缓存键未纳入切片属性，同 shape 异 begin 的后续调用沿用了先前 begin_mask=1 生效的 begin=0。该归因是行为实验推断（op_cache.h 的 CheckCacheable/AbandonCache、aclSetAclOpExecutorRepeatable 等头文件线索），本轮未做源码级证实；修复闭环未在本轮完成（专项修复任务进行中）。**临时规避（测试工程侧）**：避免相邻用例"同 shape 仅 attrs 不同"（C++ ST 把该用例 shape 设为 (80,) 即此类规避）；对失败用例单独进程/单条运行复核，防误判为算子数学错误。

### 复现增量验证信号

- 全量 65 条连续执行：62/65，3 条失败均为 Actual 从 0 开始（最大差异位置 [0] Golden=32/36，Actual=0，MERE≈7e-1）（来源平台：950；证据等级：tool_observed）
- 3 条失败用例各自单条运行：全部 PASS；5 用例最小序列连跑 5 次：第 5 条全部 FAIL（输出 [0,1,2] 而非 [32,33,34]）；任意 2 用例组合通过（来源平台：950；证据等级：tool_observed）
- 参数打印 begin=[32] 正确 + y 预填 -1 后算子仍写 [0,1,2,3,4]（来源平台：950；证据等级：tool_observed）
- 直连两段式 + 同步后销毁：仍失败；sync=true / GetWorkspaceSize 入 acl_call / synchronize+empty_cache 均无效（来源平台：950；证据等级：tool_observed）
- shape=(80,) 通过 vs shape=(64,) 失败（attrs 相同、前序相同）（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- 本增量为诊断规则（排查路径）；根治修复（AbandonCache / aclSetAclOpExecutorRepeatable 等）未在本轮闭环，不得当作已修复结论引用。
- 外部强同步、手动销毁 executor 分别导致段错误与 double free（见同域负结果卡），未排除适配层前不要尝试。
- 同 shape 异 attrs 中"仅 strides 不同""仅 mask 不同"等变体的触发面未逐一验证。
