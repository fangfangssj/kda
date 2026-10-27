---
type: "Runbook"
title: "arch35 非末维 S 写 V 读 UB 不一致未解"
description: "arch35(DAV_3510) TopK 非末维（dim!=last）用 GM GetValue+UB SetValue 逐元素搬入时，inner=256 规模下 S pipe 大量写 UB 后 V pipe（TopK）读到不一致数据，约 50%-98% 输出错值且无任何报错；Mock 全过 Real 失败、换 4 种搬运方案失败率不变是判别要点；根因未追到机制层，规避方向是避开 S pipe 写 UB。"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# arch35 非末维 S 写 V 读 UB 不一致未解

## 现象

在 Ascend 950PR（NpuArch=DAV_3510，arch35，CANN 9.1.0）上开发自定义算子 bench_no_harness_top_k（TopK）时，非末维（dim != ndim-1）用例的处理路径是：输入在 GM 中非连续（沿排序维步长 srcDimStride 跳跃），kernel 用循环把每行 inner 个元素逐元素搬入 UB，再用 AscendC TopK API 排序。`bash build.sh --soc=ascend950 -j8` 编译安装后 `bash tests/st/run.sh`（Real 模式）出现 4 个 fp16 非末维用例失败：

- 2D [256,1024] dim=0 largest=true：`y 比对 MERE=0.571, mismatch=10097/10240`（另一次运行 5280/10240，逐次波动）；
- 2D [256,1024] dim=0 largest=false：`MERE=0.594, mismatch=10103/10240`；
- 3D [16,256,1024] dim=1 largest=true：`MERE=0.0426, mismatch=158346/163840`（约 98%）；
- 3D [16,256,1024] dim=1 largest=false：`MERE=0.0428, mismatch=158306/163840`。

没有任何编译错误、运行时报错或流同步失败——是静默的数据不一致。同批次末维用例（inner 同为 256）与非末维 inner=16 用例全部通过。

## 适用场景

- 工程形态：自定义算子 kernel 直调（AscendC）+ aclnn 自定义算子包，ST Real 模式上板验证（Mock 模式不触发）。
- API 路径：`GlobalTensor::GetValue/SetValue`（`__inout_pipe__(S)` 标量 GM 访问）、`LocalTensor::SetValue`、`DataCopyPad`（blockCount+srcStride / 32B 对齐 / blockLen=sizeof(T) 各形态）、`Gather`、`PipeBarrier<PIPE_ALL>`、TopK API（V pipe）。
- 验证过的平台与 CANN 版本：Ascend 950PR（arch35/DAV_3510）+ 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——arch35 上任何"S pipe（SetValue/GetValue 标量路径）批量写 UB 后 V pipe 立即读同一 UB 区域"的 kernel。
- 路线优先级：safety_fallback（本卡是诊断/规避规则，不是确认修复；当前已验证的安全回退是"非末维场景跳过该路径/划出支持范围"）。fallback 声明：该回退会放弃非末维用例的数值验证覆盖，仅作根因未定位前的止损手段。
- 明确不适用 / 未验证场景：末维路径（一次 DataCopyPad 搬入）不受影响；inner 很小（本开发会话实测 16）时逐元素写入未见失败；其他 NpuArch 未验证。

## 判别方法

### 触发

arch35 上非末维（dim!=last）fp16 用例 ST Real 模式 y 比对 mismatch 约 50%-98% 且逐次运行波动、无任何报错；同 dtype 末维用例通过。失败与 inner 规模相关：本开发会话对照矩阵——

| 场景 | inner | 结果 |
|---|---|---|
| 末维 DataCopyPad 一次搬入 | 256 | PASS |
| 非末维 SetValue 逐元素搬入 | 16 | PASS |
| 非末维 SetValue 逐元素搬入 | 256 | FAIL ~51%-98% |

### 探针

1. **Mock/Real 差分**：同一份 kernel 跑 `bash run.sh --mock`（CPU Golden 自校验）与 `bash run.sh`（Real）。本 case Mock 55/55 全通过（含全部非末维用例）、Real 4 个非末维失败——证明算法逻辑与 golden 一致，失败是硬件侧执行差异。
2. **搬运方案替换**（判别"问题在搬运引擎还是在同步"）——依次把非末维搬入改成以下形态并上板，观察失败率是否变化：
   - GM GetValue + UB SetValue 逐元素（基线形态）：

     ```cpp
             } else {
                 // 非末维：使用 GM GetValue + UB SetValue 逐元素搬入（arch35 DataCopyPad 对小 blockLen 有对齐问题）
                 // SetValue 签名: SetValue(uint32_t index, T value)
                 int64_t baseOffset = rowIdx * srcRowStride_;
                 for (int64_t t = 0; t < inner_; t++) {
                     srcIn.SetValue(static_cast<uint32_t>(t),
                                    gmX.GetValue(baseOffset + t * srcDimStride_));
                 }
                 PipeBarrier<PIPE_ALL>();
             }
     ```

   - DataCopyPad 32B 对齐整块搬入 + Gather 提取（全 V pipe，绕开 S pipe 写 UB）：

     ```cpp
             } else {
                 // 非末维：DataCopyPad 32B 对齐搬入 + Gather 提取（全 V pipe，避免 S pipe 同步问题）
                 LocalTensor<T> nonLastDimData = nonLastDimBuf.Get<T>();
                 constexpr int32_t elemsPerBlock = 32 / sizeof(T);
                 DataCopyExtParams copyParams;
                 copyParams.blockCount = static_cast<uint16_t>(inner_);
                 copyParams.blockLen = 32;
                 copyParams.srcStride = (srcDimStride_ - elemsPerBlock) * static_cast<int64_t>(sizeof(T));
                 copyParams.dstStride = 0;
                 DataCopyPad<T>(nonLastDimData, gmX[rowIdx * srcRowStride_], copyParams,
                                {false, 0, 0, static_cast<T>(0)});
                 PipeBarrier<PIPE_ALL>();
                 // 构造字节偏移表 [0, 32, 64, ..., (inner_-1)*32]（每个 block 32 字节，第 1 个元素的偏移）
                 LocalTensor<int32_t> offsetBuffI32 = srcIndexBuf.Get<int32_t>();
                 Arange<int32_t>(offsetBuffI32, 0, 1, static_cast<int32_t>(inner_));
                 PipeBarrier<PIPE_V>();
                 Muls<int32_t>(offsetBuffI32, offsetBuffI32, static_cast<int32_t>(32),
                               static_cast<int32_t>(inner_));
                 PipeBarrier<PIPE_V>();
                 // Gather 提取每个 block 的第 1 个元素到 srcIn
                 LocalTensor<uint32_t> offsetBuffU32 = srcIndexBuf.Get<uint32_t>();
                 Gather<T>(srcIn, nonLastDimData, offsetBuffU32, 0, static_cast<uint32_t>(inner_));
                 PipeBarrier<PIPE_V>();
             }
     ```

   - DataCopyPad blockCount=1 逐元素循环搬入 + Gather；
   - DataCopyPad 整行搬入 UB + UB GetValue/SetValue 重排。

### 观测

本开发会话中 4 种替换方案上板后失败率全部维持在 ~55%（3D 用例 ~98%）不变——问题与具体搬运引擎/地址形态无关；结合 Mock 全过，指向"写入路径与 V pipe 读取之间的可见性/同步"而非数据搬错地址。TopK 之后读到的部分元素是旧值/垃圾，导致选出的 top-k 集合部分错误（mismatch 逐次波动、无越界 idx 报错）。

### 解释

推测（未到机制层）：arch35 的 `PipeBarrier<PIPE_ALL>()` 在 S pipe 大量 `SetValue` 写 UB（inner=256 次）后，不能完全保证后续 V pipe（TopK）读到全部写入；inner=16 时写入量小、时序上恰巧一致，inner=256 时同步窗口被放大。此假设与全部观测吻合（规模相关、方案无关、Mock 通过），但微架构机制未复现验证。

## 设计决策

- 排查顺序选择：先换搬运方案（排除地址/对齐因素）再做 Mock/Real 差分（排除算法因素）——两者都是低成本可上板验证的判别实验，先于任何"加同步指令试试"的盲改。
- 修复方向评估（本开发会话内未完成验证，供后续接续）：① 用 TQue 替代 TBuf（SetFlag/WaitFlag 精确同步）；② GM workspace 做 transpose 后按末维处理（本开发会话实测该路线被 workspace 大小问题阻断）；③ DataCopyPad 一次搬入整行 + Copy（V pipe）提取；④ TransDataTo5HD 做 UB 内转置。当前已验证安全的 fallback 是非末维场景不启用该逐元素路径。
- 本卡以 diagnostic_rule 形态归档而非 repair_recipe：没有同轨迹修后通过信号，不宣称修复。

## 系统信息

- Toolchain/SOC：Ascend 950PR（NpuArch=DAV_3510，arch35），CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + 安装 `.run` 包；`bash tests/st/run.sh`（Real）/ `bash tests/st/run.sh --mock`（Mock）。
- 相关文件：`op_kernel/arch35/bench_no_harness_top_k.h`（CopyIn/CopyInBlock/CopyOut 非末维路径、nonLastDimBuf）、`op_host/arch35/bench_no_harness_top_k_tiling.cpp`（UB 规划）、`tests/st/test_aclnn_bench_no_harness_top_k.cpp`（用例与 skip 逻辑）。

## 预防

- 在 arch35 上写非末维/stride 访问 kernel 时，避免"S pipe 批量 SetValue 写 UB → PipeBarrier<PIPE_ALL> → V pipe 读"的形态，优先 DataCopyPad 成块搬运（末维形态）或 TQue 事件同步。
- ST 必须同时跑 Real 和 Mock：Mock 全过 + Real 部分错且无报错时，第一时间怀疑硬件同步/可见性类问题，而不是继续改算法。
- 遇到 mismatch 逐次波动（同用例两次运行 5280 vs 10097）且无报错，把它当作"读到未定义/旧数据"的信号——确定性算法错误不会波动。

## 验证信号

问题与验证来自一次 bench_no_harness_top_k 迭代二联调修复会话：Cast 位宽问题修复完成后，剩余 4 个非末维 fp16 用例持续失败，随后进行了多轮搬运方案替换与 Mock/Real 对照。逐条实测信号：

- Cast 修复后基线：ST Real 51 通过 / 4 失败，失败全部为非末维（含一条非末维 inner=16 用例 PASS 作对照）（来源平台：950；证据等级：tool_observed）。
- DataCopyPad blockCount+srcStride、Gather 提取、blockCount=1 循环、整行搬入+UB GetValue 四种方案替换后：失败用例与失败率不变（mismatch ~55% 维持）（来源平台：950；证据等级：tool_observed）。
- 验收阶段 Real vs Mock：Real 总计 61（跳过 6）通过 51 / 失败 4（均为 dim!=last）；Mock 总计 61（跳过 6）通过 55 / 失败 0——Mock 含非末维用例全过（来源平台：950；证据等级：tool_observed）。
- 失败用例量化（验收报告）：MERE 0.0426~0.594、mismatch 10097/10240 ~ 158346/163840（约 51%-98%）（来源平台：950；证据等级：tool_observed）。
- 最终交付态：非末维用例标记为迭代三范围跳过后 ST 50/50 通过、退出码 0——确认其余路径无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本判别规则在 arch35/DAV_3510 + 9.1.0 上实测；其他 NpuArch 未验证（不推断可迁移）。
- inner 的失败阈值未标定：只知道 16 通过、256 失败，中间值（如 64/128）未测，不要把 256 当作确定性边界。
- "S pipe→V pipe 同步"是行为学归因（推测），不是机制结论；不排除其他共性因素（如某种地址/缓存形态），后续 kernel 不要仅凭本卡断言微架构行为。
- Mock 模式完全不能复现本问题（CPU 模拟下同步语义正确）。

**未决项**：

- **优先项：复现并追根因到机制层**——构造最小 kernel（S pipe 写 256 个 UB 元素 + PipeBarrier + V pipe 读）在 arch35 上复现，确认是 PipeBarrier 语义边界、S/V pipe 交互还是缓存一致性问题。
- TQue/SetFlag/WaitFlag 精确同步替代 TBuf 的方案未验证。
- inner 失败规模阈值（16~256 之间）未标定。
- GM workspace transpose 路线在本开发会话被 workspace 大小问题阻断（另一独立问题），修复后可作为绕开 UB 同步的候选路线重新评估。
## 坏实践

- 在非末维路径上用 `DataCopyPad` 逐元素形态（blockLen=sizeof(T)，fp16 即 2 字节）——arch35 MTE2 对小 blockLen 有对齐要求，会直接 507035 流同步失败（本开发会话前序已踩，已从该形态迁移走）。
- 见到非末维 mismatch 就反复换偏移计算公式——本开发会话中偏移公式修正只修复了偏移本身的错误，inner=256 的失败依旧，说明盲目调公式无法收敛。
- 只跑 Mock 或只跑 Real 就下结论——两者差分才是本类问题的判别锚点。

## 误诊

- 误诊 1（未证实）："非末维失败是 DataCopyPad 的 32 字节对齐问题"——按 32B 对齐方案重写后失败率不变，且 GM GetValue（不经过 MTE2）同样失败，对齐假设无法解释方案无关性。
- 误诊 2（被差分排除）："算法对非末维的 outer/inner/stride 展开有错"——Mock 模式同一套逻辑 55/55 全过，算法展开正确。
