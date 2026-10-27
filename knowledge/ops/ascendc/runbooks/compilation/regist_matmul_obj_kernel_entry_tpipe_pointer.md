---
type: "Runbook"
title: REGIST_MATMUL_OBJ 须入口调用且首参传 &pipe
description: REGIST_MATMUL_OBJ 是 REGIST_CUBE_OBJ 别名：首参须传 TPipe* 指针（传 TPipe 值报 InitObj no matching conversion），且宏在 AIC 核分支内嵌 return——放进 op.Init() 只退出 Init，AIC 继续跑 Process 破坏 KFC 分工致 StreamSynchronize 永不返回；必须放在 kernel 入口函数。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.high_level_api, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# REGIST_MATMUL_OBJ 须入口调用且首参传 &pipe

## 现象

在 Ascend950PR（arch35 / DAV_3510，CANN 9.1.0）上为 MoE 分组矩阵乘算子（高阶 Matmul API 路线）开发 kernel，先后踩到同一注册宏 `REGIST_MATMUL_OBJ` 的两个用法契约：

**契约一（编译期）**：在 kernel 类 `Init` 里注册高阶 Matmul 对象时，按"传对象"直觉写了 `REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm, &tilingData->mmTiling)`（首参传 TPipe 值）。kernel 编译报错（宏展开链完整可见）：

```text
op_kernel/arch35/bench_no_harness_grouped_matmul.h:99:5:
error: no matching member function for call to 'InitObj'
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm, &tilingData->mmTiling);
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.../lib/matmul/matmul_intf.h:37:27: note: expanded from macro 'REGIST_MATMUL_OBJ'
#define REGIST_MATMUL_OBJ REGIST_CUBE_OBJ
.../kfc/kfc_register_obj.h:326:12: note: expanded from macro 'REGIST_CUBE_OBJ'
    server.InitObj(tpipe, __VA_ARGS__);
    ~~~~~~~^~~~~~~
.../kfc/kernel_kfc.h:245:28: note: candidate function template not viable:
no known conversion from 'AscendC::TPipe' to 'AscendC::TPipe *' for 1st argument;
take the address of the argument with &
    __aicore__ inline void InitObj(TPipe* tpipe, T& a, Args&&... b)
```

同时 TBE 编译框架级联报 `ld.lld: error: cannot open .../kernel_meta_..._mix_aic_1.o: No such file or directory`（.o 没生成是因为源码编译失败，ld.lld 报错是级联噪音，不要去追链接问题）。

**契约二（运行期）**：把首参改成 `&pipe` 后编译安装通过，单用例 ST 运行 `timeout` 超时（exit 124）。plog 显示算子二进制被正确选中并启动：

```
SelectBin: Available bin for op BenchNoHarnessGroupedMatmul is .../BenchNoHarnessGroupedMatmul_015562fd711bbc3ebfb95dc276219c1d.o. Key is 0/1/|float16/ND/float16/ND/float16/ND/
get task type, task ration: 1:2
non-single-bin multi-kernel 4, 2
LaunchKernel: numBlocks: 4, scheduleMode: 0, blockDimOffset: 0
```

随后 `StreamSynchronize: stream_id=61.` 之后再也没有返回日志——kernel 在 NPU 上不结束。当时 kernel 结构是"标准"的类封装：入口函数 `GET_TILING_DATA_WITH_STRUCT` 后 `op.Init(...); op.Process();`，`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData->mmTiling)` 写在 `Init()` 成员函数里。

## 适用场景

- 工程形态：AscendC 高阶 Matmul API（`Matmul<...>` + `REGIST_MATMUL_OBJ`）kernel，Ascend950 类 AIC/AIV 协同（taskRation 1:2）平台。
- API 路径：`adv_api/matmul/matmul_intf.h`（`#define REGIST_MATMUL_OBJ REGIST_CUBE_OBJ`）→ `asc/impl/adv_api/detail/kfc/kfc_register_obj.h`（宏展开，AIC 分支含 `while(server.isRun()){server.Run(...);}; server.Quit(); return;`）；`KernelKfc::InitObj(TPipe* tpipe, T& a, Args&&... b)`；kernel 入口 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——宏行为由 CANN 头文件决定，适用于该架构上所有使用 matmul 对象注册的写法。
- 路线优先级：recommended_fastpath（正确性必需，无替代路线）。
- 明确不适用 / 未验证场景：纯 AIV kernel（无 matmul/KFC 服务）不涉及；`asEnableMixDualMaster` 配置下宏的 AIC 分支不 return 的情形未验证（本卡针对默认 `CFG_NORM`）；其他 CANN 版本的宏展开需重查（推断可迁移）。

## 根因

`REGIST_MATMUL_OBJ` 即 `REGIST_CUBE_OBJ`，其展开（CANN 9.1.0 `kfc_register_obj.h:194-229`）在 **AIC 核分支内含 `return;`**：

```cpp
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                       \
    CountMatmulObj(tpipe, __VA_ARGS__);                                                              \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;   \
    constexpr int8_t enableHardPollKfc =                                                             \
        AscendC::ENABLE_HARD_POOL ? AscendC::ENABLE_HARD_POOL : ASCubeObjConfig::enableABShareValue; \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;              \
    AscendC::TQue<AscendC::QuePosition::CO1, 1, &AscendC::gCO1Config> qCO1;                          \
    if ASCEND_IS_AIC {                                                                               \
        if constexpr (!asEnableMixDualMaster) {                                                      \
            AscendC::ClearWorkspace(reinterpret_cast<__gm__ uint8_t*>(workspace));                   \
        }                                                                                            \
        AscendC::gCO1Que = &qCO1;                                                                    \
        AscendC::KfcServer<enableHardPollKfc> server;                                                \
        server.Init(workspace);                                                                      \
        server.InitObj(tpipe, __VA_ARGS__);                                                          \
        if constexpr (!asEnableMixDualMaster) {                                                      \
            while (server.isRun()) {                                                                 \
                server.Run(__VA_ARGS__);                                                             \
            };                                                                                       \
            server.Quit();                                                                           \
            return;                                                                                  \
        }                                                                                            \
    }                                                                                                \
    AscendC::KfcCommClient __kfcClient__(                                                            \
        workspace, AscendC::GetSubBlockIdx(), enableHardPollKfc, asEnableMixDualMaster);             \
    if ASCEND_IS_AIV {                                                                               \
        ...                                                                                          \
    }
```

执行模型：AIC 核只应运行 KFC 服务循环（响应 AIV 发来的 matmul 硬件请求），AIV 核运行用户代码（Init/Process/End，经 KfcCommClient 下发请求）。宏里的 `return` 退出的是**调用宏的函数**——设计意图是让它在 kernel 入口函数里直接退出整个 kernel（AIC 分支）。原代码把宏放在 `op.Init()` 内：

- AIC 核：`Init()` 被宏内 `return` 提前退出 → 入口函数继续调 `op.Process()` → AIC 核执行了本应由 AIV 执行的用户循环（SetTail/SetTensorA/IterateAll），KFC 服务循环无人运行；
- AIV 核：`Process()` 里的 matmul 请求发往不存在的服务/跨核同步永等；
- 宏观表现：kernel 永不结束，`aclrtSynchronizeStream` 挂起，进程 timeout。

首参类型方面：宏内部调用 `server.InitObj(tpipe, __VA_ARGS__)`，而 `InitObj` 的第一个形参类型是 `TPipe*`（指针）；传 `pipe`（TPipe 值）无法隐式转换成指针，报 no matching member function。

## 解决配方

把 `REGIST_MATMUL_OBJ` 移到 **kernel 入口函数**（`GET_TILING_DATA_WITH_STRUCT` 之后、`op.Init()/op.Process()` 之前），首参传 `&pipe`（TPipe 指针），并把 `mm`/`pipe` 改为 public 供入口函数访问。

修复前（宏在 Init 内 + 首参传值）：

```cpp
// arch35/bench_no_harness_grouped_matmul.h —— Init 内注册（错误位置 + 错误实参）
template <typename D_T_X, int HAS_BIAS, int TRANSPOSE_WEIGHT>
__aicore__ inline void BenchNoHarnessGroupedMatmul<D_T_X, HAS_BIAS, TRANSPOSE_WEIGHT>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling,
    const BenchNoHarnessGroupedMatmulTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    // 设置 GM buffer
    xGm.SetGlobalBuffer((__gm__ D_T_X*)x);
    wGm.SetGlobalBuffer((__gm__ D_T_X*)weight);
    yGm.SetGlobalBuffer((__gm__ D_T_X*)y);
    if constexpr (HAS_BIAS) {
        biasGm.SetGlobalBuffer((__gm__ float*)bias);
    }

    // 注册 Matmul 对象（须在 InitBuffer 之前调用）
    // 使用系统 workspace 供 Matmul API 内部使用
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm, &tilingData->mmTiling);
}

// bench_no_harness_grouped_matmul_arch35.cpp —— 入口直接 Init/Process
template <typename D_T_X, int HAS_BIAS, int TRANSPOSE_WEIGHT>
__global__ __aicore__ void bench_no_harness_grouped_matmul(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGroupedMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGroupedMatmulTilingData, tilingData, tiling);
    NsBenchNoHarnessGroupedMatmul::BenchNoHarnessGroupedMatmul<D_T_X, HAS_BIAS, TRANSPOSE_WEIGHT> op;
    op.Init(x, weight, bias, y, workspace, tiling, &tilingData);
    op.Process();
}
```

修复后（宏移到入口函数、首参取地址、`mm`/`pipe` 公开）：

```cpp
// bench_no_harness_grouped_matmul_arch35.cpp —— 入口函数注册
template <typename D_T_X, int HAS_BIAS, int TRANSPOSE_WEIGHT>
__global__ __aicore__ void bench_no_harness_grouped_matmul(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGroupedMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGroupedMatmulTilingData, tilingData, tiling);

    // REGIST_MATMUL_OBJ 必须在 kernel 入口函数中调用（不能在 Init 中）：
    // 该宏在 AIC 核上包含 return 语句，会退出整个 kernel 函数，
    // 确保 AIC 核只运行 KFC server 循环，AIV 核运行用户代码 (Init/Process)。
    NsBenchNoHarnessGroupedMatmul::BenchNoHarnessGroupedMatmul<D_T_X, HAS_BIAS, TRANSPOSE_WEIGHT> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.mmTiling);

    op.Init(x, weight, bias, y, workspace, tiling, &tilingData);
    op.Process();
}

// arch35/bench_no_harness_grouped_matmul.h —— 类成员公开 + Init 去掉注册
public:
    // pipe 和 mm 需要公开访问，以便在 kernel 入口函数中调用 REGIST_MATMUL_OBJ
    // （该宏在 AIC 核上包含 return 语句，必须在 kernel 入口函数中调用）
    Matmul<AType, BType, CType, BiasType, CFG_NORM> mm;
    TPipe pipe;
    ...
    // Init 中原 REGIST_MATMUL_OBJ 处保留注释：
    // 注意：REGIST_MATMUL_OBJ 必须在 kernel 入口函数中调用（不在 Init 中），
    // 因为该宏在 AIC 核上包含 return 语句，会退出整个 kernel 函数。
```

判别 / 定位链路（本会话走通，可复用）：

1. 编译期：报 `InitObj` + `no known conversion from 'AscendC::TPipe' to 'AscendC::TPipe *'` + 级联 `ld.lld cannot open *_mix_aic_*.o` → 直接到 kernel 源码找 REGIST_* 宏的实参类型，忽略链接器报错。
2. 运行期：timeout 后取 plog，确认 kernel 已被 SelectBin 选中并 LaunchKernel（排除二进制缺失/下发失败）。
3. 看 `task ration: 1:2`、`non-single-bin multi-kernel` 确认 AIC/AIV 协同任务。
4. `StreamSynchronize` 之后无任何完成日志 → kernel 不结束，查 kernel 结构中与 AIC/AIV 分工相关的构造（REGIST_MATMUL_OBJ 位置）。
5. 读 `$ASCEND_HOME_PATH/x86_64-linux/asc/impl/adv_api/detail/kfc/kfc_register_obj.h` 宏展开，确认 AIC 分支内嵌 `return;`。
6. 对照内置算子（如 `opp/built-in/.../detect_flash_attention_score.cpp`）的用法：`REGIST_MATMUL_OBJ` 都在 kernel 入口/等价顶层作用域调用。

## 设计决策

- 为什么"移宏"而不是"让 Init 内联到入口"：宏的 `return` 语义要求它处于入口函数作用域；把整个 Init 展开进入口会破坏类封装且模板实例化更脆弱。
- 为什么 `mm`/`pipe` 设为 public：入口函数需要传 `&op.pipe`、`op.mm` 给宏；C++ kernel 侧没有更轻的访问手段（友元亦可，public 更直观）。
- 注册放在 InitBuffer 之前、GM buffer 绑定之后：REGIST_MATMUL_OBJ 只建立 mm 与 pipe/workspace/tiling 的绑定，不搬数据；放在最前可保证后续任何 Matmul 调用（SetTail/SetTensorA/IterateAll）都已绑定。
- workspace 用 `GetSysWorkSpacePtr()`（系统 workspace，host 侧 TilingFunc 已按 `GetLibApiWorkSpaceSize()` 声明大小），不自管用户 workspace。
- 参考样本：CANN 内置 ascendc 算子在入口顶层（或等价的宏展开点）调用 `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, ...)`，佐证入口作用域是预期用法。

## 预防

- 记住签名：`REGIST_MATMUL_OBJ(TPipe* pipe, workspace, matmulObj, tiling)`——第一个参数是**指针**，其余按对象/指针原样传。
- 写高阶 Matmul API kernel 时，把 `REGIST_MATMUL_OBJ` 固定放在 kernel 入口函数（GET_TILING_DATA 之后、op.Init/Process 之前），并加注释说明宏内含 return。
- 封装 kernel 类时把 `mm`/`pipe` 的可访问性提前设计好，避免后期为入口注册改可见性。
- 遇到"kernel 启动成功但 StreamSynchronize 永不返回 + taskRation 1:2"，第一怀疑 AIC/AIV 分工类结构错误（宏位置、flag 配对），而不是 tiling 数值。

## 验证信号

以下信号来自该算子 A1-Main 骨架的编译闭环与首轮上板联调（Ascend950PR，CANN 9.1.0）：

- 契约一修复前：`error: no matching member function for call to 'InitObj'` + `no known conversion from 'AscendC::TPipe' to 'AscendC::TPipe *' for 1st argument; take the address of the argument with &` + `ld.lld: error: cannot open ..._mix_aic_1.o`（来源平台：950；证据等级：tool_observed）。
- 契约一修复后：`bash build.sh --soc=ascend950 -j1` 的 grep error 无输出；`-j8` 全量构建生成 `custom_opp_ubuntu_x86_64.run`（694K）、3 个 kernel .o（覆盖 12 TilingKey，coreType MIX / taskRation 1:2 / intercoreSync 1）（来源平台：950；证据等级：tool_observed）。
- 契约二修复前：单用例 ST 运行 exit 124（timeout）；plog 显示 SelectBin 命中、`task ration: 1:2`、`non-single-bin multi-kernel 4, 2`、LaunchKernel numBlocks=4 后 `StreamSynchronize: stream_id=61.` 挂起无返回（来源平台：950；证据等级：tool_observed）。
- 机制事实：`kfc_register_obj.h:194-229` 宏展开在 AIC 分支含 `while (server.isRun()) { server.Run(...); }; server.Quit(); return;`（来源平台：950；证据等级：tool_observed——安装头文件逐字读取）。
- 契约二修复后：plog 出现 `StreamSynchronize: stream synchronize exit, stream_id=61, result=0x0`——同步成功，kernel 正常结束，测试推进到输出精度比对阶段（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对默认 `CFG_NORM`（`asEnableMixDualMaster=false`）配置；若 cube obj 配置启用 mix dual master，AIC 分支不 return，行为另论（未验证）。
- `PrintTilingData` 全 0 与挂起的因果未定论：宏位置修复后挂起消失但未复查该打印；它可能打印的是 kernel 启动参数而非 TilingData 本体（未验证）。若在别的场景见到全 0 打印，勿直接套用本卡结论。
- 宏签名与展开基于 CANN 9.1.0（`kfc_register_obj.h` / `kernel_kfc.h`）；版本变更需重查（未验证，推断可迁移）。
- 注册后 Matmul 对象的运行时行为（多组 IterateAll）在本会话未上板验证。

**未决项**：

- 复现并解释 `PrintTilingData[3]/[4]` 全 0 打印的本体（TilingData 还是 rts 参数）——作为独立观察优先复核，避免误导后续挂起类排障。
## 坏实践

- `REGIST_MATMUL_OBJ(pipe, ...)` 传值。
- 把 `REGIST_MATMUL_OBJ` 封装进任意成员函数/子函数——宏内 `return` 只退出那一层函数，AIC/AIV 分工即被破坏。
- 看到 ld.lld cannot open .o 就去查磁盘/路径/权限——先看上一条编译 error。
- 见"kernel 挂起"就先怀疑 tiling 数值全 0/死循环边界，而不检查 matmul 注册的调用位置。

## 误诊

- 曾沿"TilingData 全 0 → kernel 读到全 0 死循环"方向排查（对照 tiling 实现、TilingData 结构、GET_TILING_DATA_WITH_STRUCT 宏、add_example 同款写法），未找到 Tiling 侧问题；真正根因在 REGIST_MATMUL_OBJ 的调用位置。TilingData 全 0 打印与挂起的关系至今未定论。
