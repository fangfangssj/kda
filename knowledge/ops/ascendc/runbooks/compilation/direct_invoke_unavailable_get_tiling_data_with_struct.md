---
type: "Runbook"
title: "直调不可用 GET_TILING_DATA_WITH_STRUCT 宏"
description: "`<<<>>>` 直调工程无法使用自定义算子编译期生成的 GET_TILING_DATA_WITH_STRUCT/REGISTER_TILING_DEFAULT 宏，需逐字段从 __gm__ 拷贝 tiling 到 local；其它 void*→GM_ADDR、std::string、goto 限制为次要附带项，见对应卡。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-17T18:09:12Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 直调不可用 GET_TILING_DATA_WITH_STRUCT 宏

## 现象

在 Ascend950PR（dav-3510，56 核）上为 `movalign_bw_read`（纯读带宽测试 kernel，GM→UB，MTE2）搭建 `<<<>>>` 直调工程。原算子工程目录 `validatebw/movalign_bw_read/` 顶层 `CMakeLists.txt` 用 `add_all_modules_sources(... COMPUTE_UNIT ascend950 TILING_DIR arch35)` 走自定义算子打包路径；kernel 入口 `op_kernel/movalign_bw_read_apt.cpp` 用 `GET_TILING_DATA_WITH_STRUCT(MovalignBwTilingData, tilingData, tiling)` + `REGISTER_TILING_DEFAULT(MovalignBwTilingData)` 解包 tiling。新建 `direct_invoke/movalign_bw_read_direct.asc`（kernel 入口 + host main，复用 `op_kernel/arch35/movalign_bw_read_regbase.h` 的 `Gm2UBBwRead` 类与 `bw_die_scheduler.h` 的 `ShouldExecuteOnDie`），用 bisheng/ASC 工具链以 `--npu-arch=dav-3510` 编译时主报错：

- `GET_TILING_DATA_WITH_STRUCT` 宏未定义——直调编译路径没有自定义算子打包系统生成的宏。

附带的次要报错（本卡不展开，见对应卡与下方简述）：`void*` 不能隐式转 `GM_ADDR`（host 侧 cast，见已发布卡 `gm-addr-cast-from-void`）；`std::string` 在 ASC 编译器不可用（host main 改 C 风格即可）；`goto cleanup` 跳过变量初始化（改 early-return 即可）。

## 适用场景

- **工程形态**：`<<<>>>` 直调（ops-direct-invoke-flash 核函数构建），即把 kernel 与 host main 写进同一 `.asc`，用 CMake ASC 工具链编为可执行/`.so`，host 侧 `kernel<<<blockDim, l2ctrl, stream>>>(args...)` 直接启动。
- **API 路径**：`GET_TILING_DATA_WITH_STRUCT` / `REGISTER_TILING_DEFAULT`（自定义算子编译期宏）vs 直调路径的 `__gm__` tiling 指针手动解包；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；`GM_ADDR` kernel 形参；`ShouldExecuteOnDie` 双 die 调度。
- **验证过的平台与 CANN 版本**：Ascend950PR（dav-3510，56 核），CANN 9.1.0。
- **适用粒度**：`operator_family`（所有从自定义算子工程迁移到 `<<<>>>` 直调的 AIV kernel 通用）。
- **路线优先级**：`recommended_fastpath`——直调路径采用逐字段 `__gm__`→local 拷贝是标准做法。
- **不适用或未验证场景**：自定义算子打包路径（`add_all_modules_sources`）下原宏可用，不需要此替换；a2/a3 等其他架构未在本 trace 验证。

## 判别方法

- **触发**：直调 `.asc` 编译报 `GET_TILING_DATA_WITH_STRUCT undeclared` / `use of undeclared identifier`，且原 kernel 来自用 `add_all_modules_sources` 打包的自定义算子工程。
- **可执行探针**：在 CANN 安装树搜宏定义——`grep -rn "GET_TILING_DATA_WITH_STRUCT" ${ASCEND_HOME_PATH}/python/site-packages/tbe/tikcpp/`；若只在 `get_op_tiling.py` 等 host 侧 tiling 生成代码中出现、而无 ASC 编译期内置定义，即可判定为编译时由打包系统注入、直调不可用。
- **原始观测**：本 trace 中 `grep` 命中宏只在 host 侧 tiling 生成路径，ASC 编译器对直调 `.asc` 不注入该宏；改为逐字段拷贝后编译通过（仅余 `__gm__` cast 在 host 侧被忽略的无害 warning）。
- **解释**：`GET_TILING_DATA_WITH_STRUCT` 由自定义算子打包系统在编译前生成（按 `movalign_bw_read_struct.h` 的 `MovalignBwTilingData` 结构生成 `__gm__` 指针解引用 + 字段拷贝的展开），直调路径不经过打包系统，故宏不存在；直调 `.asc` 须把 `tiling` 形参显式 `cast` 为 `const __gm__ MovalignBwTilingData*`，再逐字段读入 local 变量后传 `&tilingData` 给 `Init`。

## 解决配方

直调 `.asc` 的 kernel 调用点与 tiling 解包改为：

- 把 `tiling`（`GM_ADDR`/`void*`）显式 `cast` 为 `const __gm__ MovalignBwTilingData*`；
- 逐字段从 `__gm__` 读入 local `MovalignBwTilingData`（或确认 ASC 编译器允许整体拷贝时用 `MovalignBwTilingData tilingData = *tilingGM;`），再传 `&tilingData` 给 `op.Init`。

附带的次要修复（仅简述，不展开）：`<<<>>>` 调用处的 host 侧 `void*` 需显式 `cast` 为 `GM_ADDR`（如 `(GM_ADDR)xDevice`），见已发布卡 `gm-addr-cast-from-void`；host main 命令行参数解析改用 C 风格 `char*`/`sscanf`，避免 `std::string`；cleanup 错误处理避免 `goto` 越过变量初始化（提前声明或改 early-return）。

对照——自定义算子 kernel 入口（`op_kernel/movalign_bw_read_apt.cpp`，本 trace file-read 事件 verbatim 恢复）使用的是打包系统宏：

```cpp
extern "C" __global__ __aicore__ void movalign_bw_read(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MovalignBwTilingData);
    GET_TILING_DATA_WITH_STRUCT(MovalignBwTilingData, tilingData, tiling);

    int32_t enableCoreNum = tilingData.enableCoreNum;
    int32_t totalCores    = tilingData.totalCores;

    // 双 die 对称调度：只让 enableCoreNum 个核执行
    uint32_t blockIdx = GetBlockIdx();
    if (!BwTest::ShouldExecuteOnDie(blockIdx, enableCoreNum, totalCores)) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(5000UL)) {
        BwRead::Gm2UBBwRead<DTYPE_X> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
```

直调路径下 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` / `TILING_KEY_IS` 不可用，须用上述手动 `__gm__` 拷贝替代；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 在直调路径仍可用且应保留。

## 系统信息

- **工具链**：CANN 9.1.0，bisheng/ASC 编译器，CMake ASC 工具链（`ascendc_kernel_cmake/asc_modules`）。
- **目标平台**：Ascend950PR（`--npu-arch=dav-3510`，56 核 AIV）。
- **相关文件**（与事件流实际读写一一对应）：
  - `op_kernel/movalign_bw_read_apt.cpp`（自定义算子 kernel 入口，file-read 恢复）
  - `CMakeLists.txt`（顶层，`add_all_modules_sources`，file-read 恢复）
  - `op_kernel/movalign_bw_read_struct.h`（`MovalignBwTilingData` 结构）
  - `op_kernel/arch35/movalign_bw_read_regbase.h`（`Gm2UBBwRead` 类）
  - `op_kernel/bw_die_scheduler.h`（`ShouldExecuteOnDie`/`GetLogicalCoreIdx`）
  - `direct_invoke/movalign_bw_read_direct.asc`（直调 kernel+host main，新建）
  - `direct_invoke/CMakeLists.txt`（直调构建，`dav-3510`）

## 验证信号

- 逐字段 `__gm__` 拷贝 tiling（替代 `GET_TILING_DATA_WITH_STRUCT`）后，`direct_invoke/movalign_bw_read_direct.asc` 以 `--npu-arch=dav-3510` 编译通过（仅 `__gm__` 在 host 侧被忽略的无害 warning），可执行文件在 Ascend950PR NPU 上 1/4/8/28/56 核多场景全部跑通（来源平台：950；证据等级: tool_observed）

## 失效边界

- a2/a3（dav-2201 等）架构未在本 trace 验证；直调路径的 `__gm__`→local 拷贝机制跨架构应通用，但 `DataCopy`/`DataCopyPad` 签名在 dav-2201 与 dav-3510 不同，换架构可能需调整 kernel 内搬运代码（未验证，推断可迁移）。
- 自定义算子打包路径下原宏仍可用，本卡所述替换仅适用于 `<<<>>>` 直调。

## 坏实践

- 在直调 `.asc` 里直接 `#include` 自定义算子打包系统生成的宏头，期望宏自动存在——直调不经过打包系统，宏不存在。
- 用 `std::string`/`std::vector` 等 STL 写 host main——ASC 编译器不接受，改用 C 风格。
- `goto cleanup` 越过后续 `aclrtEvent` 等变量声明——ASC 编译器严格拒绝 jump-over-initialization。