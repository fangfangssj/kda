---
type: "Runbook"
title: MIX kernel 507015：AIC L1+AIVec 双侧异常诊断
description: MIX（Cube+Vector）kernel 上板流同步失败 507015，设备日志同时出现 AICore 271（l1 error 非零）与 AIVec（vec error 非零）；用 kernel 二进制清单+host tiling 打点+分侧 dump 字段定位，对照 CANN 内置 GRU 锁定四个结构差异（直接成员 TCubeTiling/入口 REGIST_MATMUL_OBJ/阶段间 SyncAll/Reset 后 InitBuffer）。
tags: [task.debugging, topic.architecture, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# MIX kernel 507015：AIC L1+AIVec 双侧异常诊断

## 现象

GRU 类自定义算子（MIX 模式：Cube 两次 Matmul + Vector sigmoid/tanh 逐元素，arch35/Ascend950PR（DAV_3510，AIC=28 核/AIV=56 核），CANN 9.2.0）迭代一汇合验证：算子包编译通过、`aclnnBenchNoHarnessGruGetWorkspaceSize` 调用通过、golden 自测通过，但 5 个 L0 用例（fp32、无 bias、单层单向、S=20/B=8/inputSize=128/hiddenSize=64）在 NPU 上执行 kernel 时全部失败，`aclrtSynchronizeStream` 返回 507015。

设备错误日志（st_log.txt）显示**两类异常同时发生**：

```
[ERROR] RUNTIME(...): ProcessDavidStarsCoreErrorInfo: there is an aicore error exception,
core id is 0, error code = 271, dump info:
  pc start: 0x120041000000, current: 0x120041009980,
  sc error info: 0xffffffffffff,
  su error info: 0xeff7ff371f642668, 0x650ff4ff800f7eb,
  mte error info: 0x716fd5f80002fffa,
  vec error info: 0,
  cube error info: 0,
  l1 error info: 0x5e5a00179fda,
  aic error mask: 0x395856, ...
```

- AICore 侧（多核，core id 0/1/19/20…）：error code 271，`l1 error info` 非零、`vec error info` = 0、`cube error info` = 0 → Cube/Matmul 路径 L1 相关异常，pc 偏移 0x9980。
- AIVec 侧（core id 0/1/2/3/38/39/40…）：`vec error info` 非零、`l1 error info` = 0 → Vector 路径异常，pc 偏移约 0x1b1c。

kernel 二进制清单（`binary/ascend950/.../BenchNoHarnessGru_*.json`）确认部署形态：`coreType: "MIX"`、`taskRation: "1:2"`、`intercoreSync: 1`、kernelList 单条目。

当时 kernel 的结构（崩溃形态）：入口是非模板函数（ASCENDC_TPL 在 MIX 模式编译失败的临时回退，固定 `<float, 0>`，该编译问题另见 MIX 模式 TPL DATATYPE_DECL 卡）；TilingData 用 int32 raw buffer 承载两个 `TCubeTiling`，kernel 内 `reinterpret_cast` 取用（容量须实测 sizeof，另见宏生成 tiling 结构容量卡）；Matmul 对象在算子类 `Init` 内部构造，未在入口做 `REGIST_MATMUL_OBJ`；Process 里 Cube 与 Vector 阶段之间无 `SyncAll()`；InitBuffer 前无 `pipe.Reset()`。

## 适用场景

- 工程形态：算子仓自定义算子 kernel 上板联调（MIX 模式单 kernel：Cube Matmul + Vector element-wise 混排）。
- API 路径：`Matmul` 高阶 API + `REGIST_MATMUL_OBJ`、`GET_TILING_DATA_WITH_STRUCT`、`SyncAll`、`TPipe::Reset/InitBuffer`、`GetSysWorkSpacePtr`、`aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend950PR（AIC 28/AIV 56），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有 Cube+Vector 混排的 MIX kernel（GRU/LSTM/attention 逐时间步类尤甚）。
- 路线优先级：safety_fallback（四步诊断手法是 fastpath；"对照内置实现整体重构"的修法在本会话仅验证到编译通过，上板结论未闭合，不得当作已验证修复引用——已验证的替代路线是完全去 Cube 化的纯 AIV 重写，见相关卡）。
- 明确不适用 / 未验证场景：纯 Vector 或纯 Cube kernel 的 507015（单侧错误形态不同）；错误码 271 在非 MIX 场景的其他含义未验证；MIX 报文核型判别（AIC vs AIV 故障归属）以 device log 的 aicmap/aivmap 位图为准，本卡聚焦 MIX 双侧同时异常的结构性诊断。

## 判别方法

四步证据链（全部可在联调机上完成）：

1) 确认部署形态：读 kernel 二进制 json（coreType/taskRation/intercoreSync），确认 MIX 与 AIC:AIV 配比。

2) host tiling 打点（重建算子包前在 tiling.cpp 尾部加 OP_LOGI，输出 M/N/K/blockDim/workspace，运行期从日志回收）：

```cpp
context->SetBlockDim(usedCores);

// DEBUG: 打印 tiling 关键参数
OP_LOGI(context, "[DEBUG] blockDim=%ld, batchPerCore=%ld, usedCores=%ld, totalTasks=%ld, coreNum=%ld",
    blockDim, batchPerCore, usedCores, totalParallelTasks, coreNum);
OP_LOGI(context, "[DEBUG] Gi Matmul: M=%ld, N=%ld, K=%ld", batchPerCore, NUM_GATES * hiddenSize, inputDim);
OP_LOGI(context, "[DEBUG] Gh Matmul: M=%ld, N=%ld, K=%ld", batchPerCore, NUM_GATES * hiddenSize, hiddenSize);
OP_LOGI(context, "[DEBUG] workspace: totalUserWs=%ld, sysWs=%u, totalWs=%ld, perCoreWs=%ld",
    totalUserWs, sysWorkspaceSize, totalWsSize, perCoreWsSize);
```

本例实测：blockDim=8、batchPerCore=1、Gi M=1/N=192/K=128、Gh M=1/N=192/K=64、totalUserWs=131072、sysWs=16777216、totalWs=16908288——参数量级正常，排除 tiling 数值荒谬。

3) 读设备错误 dump 的分侧字段：`l1 error info`（Cube/L1）与 `vec error info`（Vector）哪一个非零、错误落在哪些 core id、pc 偏移多少。双侧同时非零 → 指向结构性/初始化问题而非单条指令越界。

4) 对照官方同形态实现：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/gru/`（gru.cpp / gru_tiling_data.h / gru_tiling_key.h / gru_fp32.h）是 CANN 自带的 MIX 模式 GRU，逐项 diff 自己的 TilingData/TilingKey/入口/同步结构。

## 根因

本会话未追到机制层单一归因（见遗留问题），但结构对照锁定四个高嫌疑差异（相对 CANN 内置 GRU）：

1. **TilingData 用 raw int32 buffer + `reinterpret_cast` 承载 `TCubeTiling`**（官方为直接结构体成员，`GET_TILING_DATA_WITH_STRUCT` 整体反序列化）——两侧类型/布局一旦不一致，Matmul 拿到的 tiling 即坏，Cube 侧 L1 分配错误与 271 吻合。
2. **Matmul 对象未在 kernel 入口 `REGIST_MATMUL_OBJ`**（官方模式：入口处 `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling)`）——注册时机/方式不对会直接影响系统工作空间与 L1 绑定；该宏在 AIC 分支内嵌 `return`，放进 `Init()` 只退出 Init 会破坏 AIC/AIV 分工（机制见 REGIST_MATMUL_OBJ 入口调用卡）。
3. **Cube 与 Vector 阶段之间无 `SyncAll()`**（官方 GRU 在 gi 预计算后、每个时间步内都有 SyncAll；AIV 读到未就绪的 workspace 数据可解释 vec error）。
4. **InitBuffer 前无 `pipe.Reset()`、调用了 `End()`**（官方模式 Reset 后 InitBuffer，且不调 End()）。

双侧（AIC L1 + AIVec）同时异常、崩溃对 blockDim/M/bias 等参数扰动不敏感，符合"初始化/结构层公共原因"而非单点越界。

## 解决配方

按 CANN 内置 GRU 模式整体重构（本会话落地并编译通过；上板结果未闭合）：

1) TilingData 改 `TCubeTiling` 直接成员（kernel 侧）：

```cpp
struct BenchNoHarnessGruTilingData {
    int64_t seqLen = 0;         // S: 序列长度
    int64_t batchSize = 0;      // B: batch 大小
    int64_t inputSize = 0;      // 输入特征维度
    int64_t hiddenSize = 0;     // H: 隐藏状态维度
    int64_t numLayers = 0;      // 循环层数
    int64_t numDirections = 0;  // 1 或 2
    int64_t blockDim = 0;       // 使用的核数
    int64_t batchPerCore = 0;   // 每核处理的 batch 数
    int64_t batchTail = 0;      // 尾核处理的 batch 数
    int64_t batchStart = 0;     // 当前核 batch 起始索引
    int64_t hTileSize = 0;      // H 维度单次处理元素数
    int64_t bTileSize = 0;      // B 维度单次处理元素数
    int64_t hasBias = 0;
    int64_t batchFirst = 0;
    int64_t bidirectional = 0;
    // Matmul Tiling 参数 (直接使用 TCubeTiling 结构体, 避免 host/kernel 类型不一致)
    TCubeTiling mmGiTiling;     // gi 预计算 Matmul tiling
    TCubeTiling mmGhTiling;     // gh 逐步计算 Matmul tiling
    uint32_t sigmoidTmpSize = 0;
    uint32_t tanhTmpSize = 0;
    int64_t giAllOffset = 0;
    int64_t ghOffset = 0;
    int64_t hPrevOffset = 0;
    int64_t perCoreWsSize = 0;
};
```

host 侧对应改直接成员拷贝 + 布局防线：

```cpp
// 拷贝 Matmul Tiling 到结构体成员 (TCubeTiling 直接成员, 无需 raw buffer)
// optiling::TCubeTiling (BEGIN_TILING_DATA_DEF) 与 kernel 侧 TCubeTiling 内存布局一致
static_assert(sizeof(optiling::TCubeTiling) == sizeof(TCubeTiling),
              "optiling::TCubeTiling and TCubeTiling size mismatch");
memcpy_s(&(tiling->mmGiTiling), sizeof(TCubeTiling), &mmGiTiling, sizeof(optiling::TCubeTiling));
memcpy_s(&(tiling->mmGhTiling), sizeof(TCubeTiling), &mmGhTiling, sizeof(optiling::TCubeTiling));
```

2) 入口模板函数 + `REGIST_MATMUL_OBJ`（pipe_/Matmul 对象改 public）：

```cpp
template <uint32_t DTYPE_ID, uint32_t HAS_BIAS>
__global__ __aicore__ void bench_no_harness_gru(
    GM_ADDR x, GM_ADDR weight_ih, GM_ADDR weight_hh,
    GM_ADDR bias_ih, GM_ADDR bias_hh, GM_ADDR h0,
    GM_ADDR y, GM_ADDR hn,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGruTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGruTilingData, tilingData, tiling);

    const TCubeTiling* __restrict mmGiTilingPtr = &(tilingData.mmGiTiling);
    const TCubeTiling* __restrict mmGhTilingPtr = &(tilingData.mmGhTiling);

    if constexpr (DTYPE_ID == BENCH_GRU_DTYPE_FP32) {
        NsBenchNoHarnessGru::BenchNoHarnessGru<float, HAS_BIAS> op;
        REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(),
                          op.mmGi_, mmGiTilingPtr,
                          op.mmGh_, mmGhTilingPtr);
        op.Init(x, weight_ih, weight_hh, bias_ih, bias_hh, h0,
                y, hn, workspace, &tilingData);
        op.Process();
    }
}
```

3) Process 补 Cube/Vector 阶段间 `SyncAll()`，空闲核补齐同步次数：

```cpp
::Process()
{
    // MIX 模式: AIC(Cube/Matmul) 和 AIV(Vector) 并行执行
    // SyncAll() 确保 Cube 和 Vector 阶段正确同步

    // 1. 预计算 gi_all (AIC: Matmul)
    if (actualBatch_ > 0) {
        PreComputeGi();
    }

    // 同步: 等待所有核完成 gi 预计算
    SyncAll();

    // 2. 时间步循环 (AIC: Gh Matmul + AIV: Vector 计算)
    if (actualBatch_ > 0) {
        TimeStepLoop();
    } else {
        // 空闲核也需要参与 SyncAll (TimeStepLoop 内部有 2*seqLen 次 SyncAll)
        for (int64_t t = 0; t < seqLen_; t++) {
            SyncAll();
            SyncAll();
        }
    }

    // 3. 存储 hn (AIV: Vector)
    if (actualBatch_ > 0) {
        StoreHn();
    }
}
```

4) Init 内 InitBuffer 前 `pipe_.Reset()`，全链路不再调用 `End()`（对齐官方 GRU）。

## 设计决策

- 修复选"整体对齐官方实现"而不是逐项二分验证：四个嫌疑差异互相耦合（SyncAll 依赖正确的心智模型、REGIST_MATMUL_OBJ 依赖入口结构），逐项验证成本高；官方实现是已验证可运行的同形态参照。
- 空闲核补 `2*seqLen` 次 SyncAll：MIX 模式 SyncAll 是全核栅栏，任何核少参与一次都会挂起其他核，因此 actualBatch_==0 的核必须按 TimeStepLoop 的同步次数空转对齐。
- 保留 host tiling 打点直至稳定：MIX 崩溃排查里 M/N/K/workspace 是第一手证据。

## 预防

- MIX 模式 kernel 从第一版就按官方同形态算子（`ops_nn/ascendc/gru` 等）的结构写：TilingData 直接成员、入口 REGIST_MATMUL_OBJ、阶段间 SyncAll、Reset 后 InitBuffer。
- raw buffer + reinterpret_cast 传递宏生成 tiling 结构是高危模式，static_assert 大小一致只是最低防线。
- 联调环境保留 `ASCEND_GLOBAL_LOG_LEVEL` 调试与 st_log 全量落盘习惯，507015 类错误的第一手证据全在 dump 字段里。

## 验证信号

问题与验证来自同一次联调+调试会话（Ascend950PR，CANN 9.2.0）：

- 触发：5/5 用例 `aclrtSynchronizeStream` 返回 507015（来源平台：950；证据等级：tool_observed）。
- 设备 dump：AICore error code=271、l1 error info 非零（如 0x5e5a00179fda）；AIVec 侧 vec error info 非零（如 0x41005224003111b0）；错误核覆盖 AIC 与 AIV 两组（来源平台：950；证据等级：tool_observed）。
- 排除性实验（均未消除崩溃）：blockDim 8→4（M=2）、`GetCoreNumAic()`、`EnableBias(false)`、`SetBufferSpace` 调整（来源平台：950；证据等级：tool_observed）。
- tiling 打点实测参数正常（blockDim=8、Gi M=1/N=192/K=128、Gh M=1/N=192/K=64、sysWs=16MB）（来源平台：950；证据等级：tool_observed）。
- 结构对照结论（CANN GRU 的四项差异）来自官方源码阅读（来源平台：950；证据等级：tool_observed）。
- 重构后全量编译通过并出包（kernel binary 重新生成）（来源平台：950；证据等级：tool_observed）。
- 上板：L0-001 开始执行（日志出现用例头），最终结果未在本会话记录内闭合（来源平台：950；证据等级：tool_observed；边界限制）。

## 失效边界

- 本卡是诊断卡：重构修法**只验证到编译通过 + 测试启动**，507015 是否消除、精度是否达标均未闭合，不得引用为"已修复"。同 trace 后续在 MIX 通道内的同步变体尝试（2×/3×SyncAll、LSTM 式 VECCALC+PipeBarrier）全部失败，最终以纯 AIV 重写收尾（见 MIX SyncAll 死锁卡与纯 AIV 重写路线卡）。
- 四个嫌疑差异未做单一归因（未逐项消融），若重构后仍崩溃，须回到分侧 dump 逐项验证。
- `2*seqLen` 空转同步数与 TimeStepLoop 内部实现强耦合，改循环结构时必须同步改空转核的补齐次数。
- 未验证，推断可迁移：A2/310P 等其他芯片的 MIX 栅栏语义推定一致，但 taskRation 与核数不同，空闲核判定逻辑需按平台重推。

**未决项**：

- 优先项：上板闭环验证重构后 kernel 是否消除 507015 并通过 L0 精度对比（本会话测试启动后记录中断；同 trace 后续 MIX 变体仍全部失败并转向纯 AIV）。
- 优先项：若仍崩溃，对四个结构差异逐项消融，把根因追到机制层（哪一项直接触发 l1/vec error）。
- `GetSysWorkSpacePtr()` 在 MIX 双侧（AIC/AIV）的语义与 16MB 系统工作空间布局未单独验证。
## 坏实践

- raw int32 buffer + `reinterpret_cast` 在 kernel 侧还原 `TCubeTiling`。
- Cube 与 Vector 混排却不在阶段边界 `SyncAll()`，或空闲核不参与全核栅栏。
- Matmul 对象藏在类私有成员里自行构造，不走入口 `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), ...)`。
- 见到 507015 先调 blockDim/M 值碰运气（本会话四轮参数扰动全部无效，浪费联调窗口）。

## 误诊

- 误诊一：怀疑 M=1 低于 Matmul 对齐下限 → 改 M=2 仍崩，排除。
- 误诊二：怀疑 blockDim 按 AIV 核数误分发 → 改 `GetCoreNumAic()` 仍崩，排除。
- 误诊三：怀疑 bias 未显式关闭 → `EnableBias(false)` 仍崩，排除。
- （以上均为合理的首排除象，但均非根因；真实方向在结构层。）
