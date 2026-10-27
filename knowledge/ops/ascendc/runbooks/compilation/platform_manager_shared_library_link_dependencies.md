---
type: "Runbook"
title: "直调共享库加载未定义平台符号—补齐CANN链接依赖"
description: "Ascend C 直调共享库可链接却在加载时缺 PlatformAscendCManager 符号；为目标补齐 CANN 平台运行依赖集与库目录。"
tags: ["task.debugging", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# 直调共享库加载未定义平台符号—补齐CANN链接依赖

## 现象

在 CANN 9.2.0 的单文件 Ascend C kernel 直调工程中，`rms_norm.asc` 已经能够编译、链接并产出 `libop_rms_norm.so`。该 host 入口调用了 `platform_ascendc::PlatformAscendCManager::GetInstance()` 来取得 AIV 核数，但 `CMakeLists.txt` 的目标只链接了 `torch_npu` 和 `Python3::Python`。

随后运行 NPU 测试时，失败不是发生在 kernel 启动或数值比较阶段，而是发生在 pytest 收集阶段的动态库加载：

```text
E   OSError: .../build/libop_rms_norm.so: undefined symbol: _ZN16platform_ascendc22PlatformAscendCManager15platformInitMtxE
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
---EXIT:2---
```

这类现象容易被链接阶段的成功误导：ELF 共享库已经生成并不等于其所有运行期符号都已由目标依赖解析。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`PlatformAscendCManager`、`platformInitMtxE`、`libplatform.so`、`tiling_api` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖相同直调共享库、Platform Manager 调用和安装布局。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

`rms_norm.asc` 的 host 侧代码引用了 CANN 平台管理器，但目标的链接依赖仍停留在仅满足 PyTorch 扩展注册的 `torch_npu + Python3::Python`。共享库链接器允许该符号暂时未解析，因此 `cmake --build` 返回成功；直到 `ctypes.CDLL` 经 `torch.ops.load_library` 实际装载 `.so`，动态加载器才报告 `PlatformAscendCManager::platformInitMtxE` 未定义。

根因链为：

1. host 入口调用 `PlatformAscendCManager`；
2. 目标没有声明 CANN platform/runtime 依赖和 CANN `lib64` 搜索目录；
3. `.so` 仍被生成，但保留未解析的平台符号；
4. Python 动态加载该 `.so` 时才失败，测试甚至无法完成收集。

## 解决配方

修复边界是 `CMakeLists.txt` 中该共享库目标的完整链接库与链接目录配置。下面的 before/after 均为开发轨迹中实际写入的配置段，未改写符号名或依赖顺序。

### before

配置状态：未提交的失败工作树；相对路径 `operators/rms_norm/CMakeLists.txt`；配置段 `target_link_libraries(op_rms_norm)` 与 `target_link_directories(op_rms_norm)`；该段支撑“仅链接 PyTorch 依赖导致加载期缺平台符号”的结论。

```cmake
target_link_libraries(op_rms_norm PRIVATE
    torch_npu
    Python3::Python
)

target_link_directories(op_rms_norm PRIVATE
    ${TORCH_NPU_LIBRARIES}
)
```

### after

配置状态：同一次开发会话中的修复后工作树；相对路径 `operators/rms_norm/CMakeLists.txt`；同一完整配置段；该段支撑“补入 CANN 直调依赖集及库目录后可以重链并动态加载”的结论。

```cmake
target_link_libraries(op_rms_norm PRIVATE
    torch_npu
    Python3::Python
    ascendcl
    platform
    register
    tiling_api
    runtime
)

target_link_directories(op_rms_norm PRIVATE
    ${TORCH_NPU_LIBRARIES}
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
```

实际应用时还要在重新配置前正确导出 `ASCEND_HOME_PATH`，因为 after 配置通过该变量定位 CANN 的 `aarch64-linux/lib64`。

## 设计决策

- 选择完整的 direct-launch 依赖集，是因为工程同时包含平台查询、直调启动和 tiling/runtime 入口；轨迹先确认了 `libascendcl.so`、`libplatform.so`、`libregister.so`、`libtiling_api.a` 与 `libruntime.so` 在同一 CANN `lib64` 目录中，再一次性更新目标依赖。
- 没有采用“只看 `cmake --build` 是否返回 0”的判定方式，因为这不能发现共享库的延迟符号解析问题；修复后额外执行独立 `load_library` 探针。
- 没有验证逐个删除 `ascendcl/register/tiling_api/runtime` 后的最小集合，因此不能把五个库都宣称为解决该单一符号的最小必要集合。
- 当前方案属于 `recommended_fastpath`：它是本次直调工程已验证可加载的配置，不包含性能收益声明。

## 系统信息

- toolchain：CANN 9.2.0，ASC 编译器位于 CANN 安装树，CMake 架构 `dav-2201`。
- 目标平台：Atlas A3；设备探测输出 `SocVersion: Ascend910_9362`、`NpuArch: dav-2201`。
- 工程文件：`operators/rms_norm/CMakeLists.txt`、`operators/rms_norm/rms_norm.asc`、`operators/rms_norm/test_rms_norm.py`。
- 构建产物：`build/libop_rms_norm.so`；加载入口为 `torch.ops.load_library`。

## 预防

- 只要 host 代码新增 `platform_ascendc`、ACL、tiling 或 runtime API，就同步核对目标的显式链接库和 CANN 库搜索目录。
- 构建共享库后必须增加一个独立动态加载探针；不要把“`.so` 已生成”当成“运行时依赖已闭合”。
- 对 `undefined symbol` 先把 mangled name 解读到所属 CANN 组件，再核对链接目标；不要直接进入 kernel 精度调试。
- CMake 修改后要重新执行 configure，再执行 build，避免只用旧生成系统重链造成错误归因。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：加载扩展时报 undefined symbol: PlatformAscendCManager::platformInitMtx，pytest collection exit=2。（来源平台：a3；证据等级: tool_observed）。
- 修复后：共享库重新加载输出 loaded ok，并返回 True。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 该验证只覆盖 CANN 9.2.0 的 aarch64 host 安装布局；其他 host 架构或 CANN 版本的库目录和库拆分可能不同。
- 其他平台注册平台未验证；相同 direct-launch 依赖机制仍须重新确认安装布局和所需组件，不能加入本卡 `platforms`。
- 本修复只关闭共享库的动态加载问题，不证明 kernel 的数值正确性、同步正确性或性能。原会话的算子正确性最终没有闭环。
- 如果目标根本不调用平台管理、ACL、tiling 或 runtime API，完整依赖集可能过宽，应以实际未定义符号和目标引用为准收缩。

## 坏实践

- 只因为链接器成功产出 `.so` 就开始排查 kernel 计算逻辑；加载期未定义符号说明测试尚未进入 kernel。
- 看到 `undefined symbol` 后只补 `link_directories` 而不把提供符号的库加入 `target_link_libraries`；目录只解决查找位置，不能声明依赖。
- 反过来只添加库名、不提供与当前 CANN 安装匹配的搜索目录；这会把问题从加载期变成配置或链接期找不到库。

## 误诊

动态加载错误与后续出现的 NaN、部分 shape 错误是不同闭环。前者在 Python 装载共享库时发生；后者只有库成功加载并启动 kernel 后才可能出现，不能合成同一个“精度问题”。