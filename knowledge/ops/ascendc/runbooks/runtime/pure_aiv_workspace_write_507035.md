---
type: "Runbook"
title: "纯 AIV 写 workspace 507035 errcode 95"
description: "ascend950/arch35 上无 REGIST_MATMUL_OBJ 的纯 AIV kernel（taskRation=0:1）里，GM→UB 读 workspace 不崩、UB→GM 写 workspace 必崩 ret=507035（plog errcode 95 MTE DDR 越界），写 y 输出正常；GetUserWorkspace 忽略入参走 KFC 地址，偏移/+16MB/SetSysWorkspaceForce/恢复 REGIST_MATMUL_OBJ(GM/ND) 等方案全部实测失败。"
tags: [task.debugging, topic.memory, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 纯 AIV 写 workspace 507035 errcode 95

## 现象

在把 conv3d 反向滤波器算子（ascend950 / Ascend950PR，arch35，CANN 9.1.0）的矩阵乘阶段改写为纯 AIV Vector contraction（删除 `REGIST_MATMUL_OBJ` 与全部 Matmul 代码，kernel 二进制 taskRation="0:1"）之后：UT 10/10、ST Mock（CPU golden）5/5 全部通过，但 ST Real 5/5 用例流同步失败 `ret=507035`。查 `~/ascend/log/debug/plog/` 最新日志得到 AIV 侧硬件异常：

```
[ERROR] ... PrintTaskErrorMsg:Task run failed, stream_id=61, pos=0, task_sn=2, sqe_type=1(aiv), errType=0x1(task exception), sqSwStatus=0.
[ERROR] ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), ... there is an aivec error exception,
  core id is 0, error code = 95, ... mte error info: 0x3012000020cbf, ...
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4.
[ERROR] ... rtStreamSynchronize:ErrCode=507035, desc=[vector core exception]
```

规律性现象（读写不对称）：GM→UB **读** workspace 不崩（读回内容全零），UB→GM **写** workspace 必崩（DataCopyPad 与 DataCopy 均崩）；同参数写输出张量 `gmY_[0]` 始终正常。workspace 需求大的用例（约 16MB 级）崩，workspace 需求小的用例（约 1.5~2MB 级）不崩但输出错（MERE=1.00）——提示写入落在了"部分有效"的错误区域。此时 kernel 入口为标准动态模式 5 参签名 + `SetSysWorkspace(workspace)` + `GetUserWorkspace(workspace)`。

## 适用场景

- 工程形态：算子仓自定义算子（build.sh `--soc=ascend950`，aclnn 调用，ST Real 上板），kernel 为纯 AIV（无 REGIST_MATMUL_OBJ / 无 Cube 对象，taskRation="0:1"）且需要用户 workspace。
- API 路径：`AscendC::GetUserWorkspace` / `SetSysWorkspace` / `SetSysWorkspaceForce` / `GetSysWorkSpacePtr`（CANN `asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h`、`asc/include/basic_api/kernel_operator_swap_mem_intf.h`、`asc/impl/basic_api/utils/kernel_utils_constants.h`）；`REGIST_MATMUL_OBJ`/`REGIST_CUBE_OBJ`（kfc_register_obj.h）；`DataCopyPad`/`DataCopy`（UB↔GM）。
- 验证过的平台与 CANN 版本：Ascend950 / Ascend950PR（arch35 / DAV_3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——arch35 上不经 REGIST_MATMUL_OBJ 而直接使用用户 workspace 的 kernel。
- 路线优先级：safety_fallback（本卡为诊断规则：给出定位手段与已验证的机制事实；修复未在本轮闭合，已知规避是"不使用框架 workspace"，见失效边界）。
- 明确不适用 / 未验证场景：带 REGIST_MATMUL_OBJ 且真正调用 Matmul compute 的 MIX kernel（其 workspace 使用是另一问题域，本轨迹未测试）；CPU Mock / UT 环境（tiling faker 固定 CORE_NUM=32、真机 GetCoreNumAic()=28，UT 通过不能佐证上板行为）；其他 arch 的 RESERVED_WORKSPACE 取值不同（910/310P 为 2MB）。

## 判别方法

触发信号：纯 AIV kernel 上板 `ret=507035` + debug plog `errcode:(95) The DDR address of the MTE instruction is out of range`（sqe_type=1 aiv）。按下列阶段隔离链定位（每步 rebuild + install + 单跑 ST，全部上板实测）：

1. 注释掉新写的 contraction 计算（保留生成阶段）→ 仍崩 → 与新计算逻辑无关。
2. 再注释核间同步 `SyncAll` → 仍崩 → 与同步无关。
3. Process 全空（只保留 Init）→ 不崩（5/5 输出错但无流同步失败）→ Init/指针建立本身安全，崩在某个 GM 访问。
4. Process 里只写输出张量 `DataCopyPad(gmY_[0], ubTemp, dbgParams)` → 不崩 → 输出指针有效。
5. Process 里只写用户 workspace `DataCopyPad(gmWsGrad_[0], ubTemp, dbgParams)` → 5/5 崩 → **用户 workspace 指针无效**。
6. 读 workspace→写 y 与 读 grad→写 workspace 的 A/B 对照：只保留前者 5/5 完成；加上后者 5/5 ret=507035（读写不对称）。
7. 换对齐版 `DataCopy`（32B 对齐）写 workspace → 仍崩 → 与 DataCopyPad/DataCopy 的 API 选择无关。
8. 对照（带 REGIST_MATMUL_OBJ 的旧 kernel 同平台实测）：`DBG ptrs: workspace=0x12004d600000 user=0x12004d600000 sysWs=0x12004c600000`——即 user = workspace 形参本身、系统 workspace（KFC）= workspace − 16MB。删除 REGIST_MATMUL_OBJ 后同样入口读到的 user 指针写入即越界。

探针代码（第 4/5/6 步，verbatim）：

```cpp
    // DEBUG: 测试 GM 指针有效性
    LocalTensor<D_T_X> ubTemp = ubTempBuf_.Get<D_T_X>();
    Duplicate<D_T_X>(ubTemp, static_cast<D_T_X>(3.0f), 16);
    DataCopyExtParams dbgParams{1, 32, 0, 0, 0};
    DataCopyPad(gmY_[0], ubTemp, dbgParams);  // 测试 gmY_

    Duplicate<D_T_X>(ubTemp, static_cast<D_T_X>(5.0f), 16);
    DataCopyPad(gmWsGrad_[0], ubTemp, dbgParams);  // 测试 gmWsGrad_（workspace）

    // 读回 wsGrad 写到 y（确认 workspace 可读写）
    DataCopyPad(ubTemp, gmWsGrad_[0], dbgParams, DataCopyPadExtParams<D_T_X>{false, 0, 0, 0});
    DataCopyPad(gmY_[0], ubTemp, dbgParams);
```

## 根因

已验证到 CANN 头文件层的机制事实（cann-9.1.0 源码，逐字核对）：

```cpp
// asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
{
    if (g_sysWorkspaceReserved == nullptr) {
        g_sysWorkspaceReserved = workspace;
    }
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if (WORKSPACE_PARAM_OFFSET == 0xffffffff)
#if defined(__NPU_DEVICE__)
    __set_kfc_workspace_addr(workspace);
#else
    g_sysWorkspaceReserved = workspace;
#endif
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
    (void)(workspace);                       // 入参被忽略！
    return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;
}
```

```cpp
// asc/include/basic_api/kernel_operator_swap_mem_intf.h
#ifndef WORKSPACE_PARAM_OFFSET
#define WORKSPACE_PARAM_OFFSET 0xffffffff     // 默认值
#endif
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
{
    // kernel launch
#if defined(__NPU_DEVICE__) && defined(__NPU_ARCH__)
    if constexpr (__NPU_ARCH__ ==  2201 || __NPU_ARCH__ == 3510) {
        return __get_kfc_workspace_addr();    // arch35 走 KFC 地址，不读 g_sysWorkspaceReserved
    } else {
        return g_sysWorkspaceReserved;
    }
#else
    ...
#endif
}
```

```cpp
// asc/impl/basic_api/utils/kernel_utils_constants.h
// workspace system reserve 16MB
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;   // arch35 = 16MB，不是 2MB
```

机制链：`GetUserWorkspace` 忽略入参，返回 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`；arch35 上 `GetSysWorkSpacePtr()` 走 `__get_kfc_workspace_addr()`（`SetSysWorkspace` 设置的 `g_sysWorkspaceReserved` 在该路径上被忽略）；KFC workspace 地址在带 REGIST_MATMUL_OBJ 的 kernel 里由 `REGIST_CUBE_OBJ` 内部的 `KfcServer::Init(workspace)` / `KfcCommClient(workspace, ...)` 设置，而纯 AIV kernel 没有这条路径——`GetUserWorkspace` 依赖的基址来源不确定，返回无效指针，AIV 的 MTE 写入越界报 error 95。

**修复未在本轮调试内闭合**——已试方案（全部实测失败）：

1. `SetSysWorkspaceForce(workspace)`：全部用例崩（预期它设置 KFC 地址，行为相反，原因未解释）。
2. `user = workspace` 直连：全部崩。
3. `workspace + 1MB / +2MB / +16MB` 与 tiling workspace 显式 +16MB：大 workspace 用例仍崩、小用例不崩但输出错（ST 侧打印确认缓冲已含 +16MB、目标地址在 `aclrtMalloc` 缓冲范围内仍崩——限制来自内存区域属性/注册机制而非越界）。
4. 换 `ACL_MEM_MALLOC_NORMAL_ONLY` 分配标志：均崩。
5. 恢复 `REGIST_MATMUL_OBJ(GM/ND)`（不调任何 Matmul compute，只为把 taskRation 变为 "1:2"）+ `matmul_tiling::MultiCoreMatmulTiling` + `optiling::TCubeTiling::SaveToBuffer` 序列化 cubeTiling（data size 200 与 kernel 侧结构体一致）+ 16MB RESERVED_WORKSPACE + `SetSysWorkspace`/`GetUserWorkspace`：ST Real 仍 0/5 ret=507035——**"taskRation=1:2 即可写"的假说被证伪**（至少不充分）。期间错误形态一度变化：cubeTiling 数据异常/全零的阶段出现过 ret=507015（aivec "timeout or trap error"、MTE error=0）与"不崩但输出全零（MERE=1.00）"，但只要 cubeTiling 经 SaveToBuffer 正确序列化，写 workspace 就回到 507035。
6. 手写 `ManualSyncAll`（GM 标志位 + DataCopyPad 轮询）：仍 507035。

被上述负结果排除的假说清单：cubeTiling 数据全零/损坏、workspace 大小不足、SetSysWorkspace 缺失、REGIST_MATMUL_OBJ 未初始化（零 cubeTiling 不崩但全零输出）、DataCopyPad 参数结构（`DataCopyParams` → `DataCopyExtParams`）、V→MTE3 / MTE2→V 事件同步缺失、aclrtMalloc 分配标志。

## 设计决策

- 隔离策略选"由外到内逐阶段注释 + 单 GM 指针写测试 + 读写 A/B"而非逐行 debug：MTE 越界在 plog 里只有 PC 与错误码，逐阶段二分最快收敛到"哪类指针无效"；空 Process 定位阶段（Init vs 搬运）、读写 A/B 定位方向（读 vs 写）、换 API 排除形态问题——三轮即可把"workspace 写不可用"钉死，避免在地址算术上兜圈（本开发会话 在定位到"写不可用"前消耗了多轮构建）。
- 打印指针用 `(uint64_t)` + `%lx`：__gm__ 指针不能 cast 到 void*（CCE 限制，另卡）。
- 偏移方案（workspace + N）先后按 1MB→2MB→16MB 推进：16MB 来自头文件 RESERVED_WORKSPACE 的 arch35 取值与带 Matmul 时 sysWs = workspace − 16MB 的对照，而非拍脑袋。
- REGIST_MATMUL_OBJ 恢复方案的动机来自历史对照："GM/ND + REGIST_MATMUL_OBJ + 生成阶段写 workspace"曾在旧版本工作——该推理链的前提（旧版本工作）依赖更早的 trace 片段，本 shard 内未能复现"工作"状态；四层加固后仍失败，该路线在本平台判 banned（不暗示它永不适用，但在没有进一步框架侧证据前，不应再把"恢复 REGIST_MATMUL_OBJ(GM/ND) 即可使 workspace 可写"当作修复手段重复投入）。
- 参考实现定位方法：`grep -rln REGIST_MATMUL_OBJ $ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/` 找到 `cumsum_cube`（GM/ND + GetUserWorkspace + SetSysWorkspace 的官方形态）作为模板——其 tiling/host 侧写法可吸收，但其在真实 MIX 任务里"workspace 可写"的收益未迁移成功。

## 预防

- ascend950 上写需要中间缓冲的纯 Vector 算子前，先用本卡"判别方法"第 6 步的读写 A/B 快速验证 workspace 可写性，再决定算法是否依赖 workspace。
- 不要假设 `GetUserWorkspace(workspace)` 返回"workspace 参数 + 偏移"——arch35 实现忽略入参，语义依赖全局 KFC 地址。
- 在纯 AIV kernel 上沿用带 Matmul kernel 的 workspace 假设（16MB 保留区由 KFC 机制声明、框架自动分配）——该假设在无 REGIST_MATMUL_OBJ 时不成立（本轮实测）。
- 出现整批 ret=507035 而 Mock/UT 全绿时，先查 plog errcode（95=MTE DDR 越界）再查计算逻辑；只看 run 级 plog 会漏掉错误详情（本错误详情只在 debug plog 中）。

## 验证信号

来源说明：以下信号来自同一次纯 AIV 重写后的上板联调与系统排查会话——多轮探针构建 + ST Real 运行 + plog 读取，每轮"清 build → build.sh → 安装 .run → UT → ST Real → 读 plog"。

- 触发：ST Real 5/5 `ret=507035`；debug plog `errcode:(95) errorStr: The DDR address of the MTE instruction is out of range`（sqe_type=1 aiv，core id 0/47 起）（来源平台：950；证据等级：tool_observed）。
- 空 Process 不崩：5/5 用例 MERE=1.00e+00 但无流同步失败（来源平台：950；证据等级：tool_observed）。
- 读写不对称：仅"读 workspace→写 y"5/5 完成（MERE=1.00）；加"写 workspace"后 5/5 ret=507035；`DataCopy`（32B 对齐）写 workspace 仍 ret=507035（来源平台：950；证据等级：tool_observed）。
- 只写 gmY_[0] 不崩（输出错值）、只写 gmWsGrad_[0] 5/5 崩（来源平台：950；证据等级：tool_observed）。
- 带 REGIST_MATMUL_OBJ 对照：`DBG ptrs: workspace=0x12004d600000 user=0x12004d600000 sysWs=0x12004c600000 tiling=0x120000200030`（sysWs = workspace − 16MB）（来源平台：950；证据等级：tool_observed）。
- ST 打印：L0-01 workspaceSize=32866816（0x1f58200，含 +16MB）、workspace=0x12004c600000；L0-02 workspaceSize=18383360；L0-03 workspaceSize=24601088（含 +16MB 后仍崩）（来源平台：950；证据等级：tool_observed）。
- 偏移方案实测：`user=workspace+2MB / +16MB / tiling+16MB` 后大 workspace 用例仍 507035、小 workspace 用例不崩但 MERE=1.0（来源平台：950；证据等级：tool_observed）。
- REGIST_MATMUL_OBJ 恢复尝试 A/B/C/D 各轮 ST Real 均 0/5：`[FAIL] 流同步失败: ret=507035` ×5；plog errcode 95、args 打印 workspace=0x12004c600000、blockDim=28；中间形态 cubeTiling 异常阶段 ret=507015（timeout or trap error，MTE error=0）、cubeTiling 全零阶段不崩但 MERE=1.00；每轮编译成功、UT 10/10、ST Mock 5/5（MERE=0.00）——失败仅限 NPU workspace 访问（来源平台：950；证据等级：tool_observed）。
- CANN 头文件机制事实（GetUserWorkspace 忽略入参 / arch35 GetSysWorkSpacePtr 走 KFC 地址 / RESERVED_WORKSPACE=16MB / WORKSPACE_PARAM_OFFSET 默认 0xffffffff）：逐字读取 CANN 9.1.0 安装目录源码（来源平台：950 主机侧；证据等级：tool_observed）。
- 对照旁证：ST Mock（CPU golden）5/5 通过 MERE=0.00、UT 10/10 通过（计算逻辑正确性旁证）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论闭合到"用户 workspace 指针无效 + CANN 头文件层机制"，修复未闭合：`user = workspace + 16MB` 直连 + tiling 显式含 16MB 开销的最终验证在本轮调试记录结束时仍未完成——不得当作已验证配方使用。
- 大/小 workspace 用例差异化表现（大崩小不崩但输出错）未被任何已试方案消除，其精确内存布局解释未确定；`GetUserWorkspace` 小 workspace 不崩的现象依赖运行时内存布局（F+16MB 之后恰有足够可写 GM），属非确定性行为，不可作为可靠方案。
- UT faker（CORE_NUM=32）与真机（GetCoreNumAic()=28，blockDim=28）不一致：UT 10/10 不能佐证上板行为。
- RESERVED_WORKSPACE 取值按 arch 区分（910/310P 为 2MB，arch35 为 16MB），跨平台迁移需按目标 arch 重查 `kernel_utils_constants.h`（其他平台未验证）。
- "读可写不可"的机制解释（内存区域属性/注册）是推断，未在框架源码层定论；未尝试路径（非结论）：`enableMixDualMaster=true`（需 ibshare MatmulConfig）、手工 `KfcCommClient` 只注册不建 Matmul、框架侧 workspace 注册 API。

**未决项**：

- **优先项：复现并追根因到机制层**——阅读 `CommonOpExecutorRun`/runtime 为 "0:1" 任务的 workspace 分配与 MTE 注册路径，确认"写不可用"的确切机制（内存区域属性？未注册？）；确认 ascend950 纯 AIV kernel 的正确 workspace 使用方式（是否存在显式注册 API）；解释 `SetSysWorkspaceForce` 全崩的原因与大/小 workspace 差异表现。
- 未实施候选的验证：`enableMixDualMaster=true`+ibshare 配置、手工 `KfcCommClient`、cumsum_cube 在真实 MIX 任务下的 workspace 写行为对比。
- `__get_kfc_workspace_addr()` 的实际值未直接观测（仅通过行为差异推断 ≠ workspace 参数）。
- 本开发会话 最终转向 DirectContraction（无 workspace 直算）路线——其精度闭环不在本卡范围。
## 坏实践

- 在地址算术上反复尝试（+16MB、-16MB、GetUserWorkspace、SetSysWorkspaceForce 组合）而不先做读写 A/B 探针。
- 把 `GetUserWorkspace` 返回地址当用户 workspace 用（arch35 忽略入参），小规模侥幸不崩、大规模必崩且数据错误。
- 把"零 cubeTiling 不崩"当成"workspace 已可写"的证据——不崩只说明服务循环没被触发，输出全零（MERE=1.00）恰恰证明数据通路没工作。
- 把"历史某版本能写 workspace"的记忆当作当前版本的可复现事实，直接派生修复方案而不先写最小复现。

## 误诊

- 最初怀疑新写的 VectorContraction（Duplicate 重复次数 / PipeBarrier / ReduceSum tmpBuffer / Cast 模板参数）——注释掉 contraction 后仍崩，排除。
- 怀疑核间 SyncAll——注释后仍崩，排除。
- 怀疑 blockDim/核数切分（UT faker 32 核 vs 真机 28 核）——空 Process 不崩，排除为主因（核数差异仍是不一致点）。
- （agent_reported 假设）"workspace+16MB 超出分配缓冲"——ST 打印证明缓冲含 +16MB 且目标地址在范围内，排除；"HUGE_FIRST 分配的内存对 AIV MTE 不可见"——读同一地址成功、x/grad/y 同标志分配可写，排除；"HBM vs DDR 区域差异"——同地址在有 REGIST_MATMUL_OBJ 的旧 kernel 可写，与区域说矛盾，未采纳。
- host 侧 double free 曾被当作独立内存 bug 优先排查，实际是 507035 失败后错误清理路径的伴生现象。
