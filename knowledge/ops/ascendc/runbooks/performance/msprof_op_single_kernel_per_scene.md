---
type: "Runbook"
title: msprof op 单次仅采单 kernel，按场景分跑
description: 自定义 aclnn 算子的单算子性能采集完整配方：perf demo（多 TilingKey 场景 + 完整两段式循环）+ msprof op 按场景分跑 + perf_summary 归档；一次 msprof op 运行只产出单个 kernel 的数据是多 TilingKey 算子采集的关键约束。
tags: [task.testing, topic.implementation.debugging, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# msprof op 单次仅采单 kernel，按场景分跑

## 现象（任务背景）

在 Ascend950PR 上为 BenchNoHarnessStridedSlice（自定义 aclnn 算子，3 个 TilingKey 场景：CONTIGUOUS/STRIDED_POS/STRIDED_NEG）做性能验收：需要产出每个场景的 msprof 8-CSV 数据、汇总 summary 与验收报告。过程中踩了两个流程坑并形成稳定做法：(1) `msprof op` 一次运行只产出**单个** kernel 的 OpBasicInfo/CSV 数据（demo 跑 `all` 场景 100 循环，OPPROF 目录里只有 1 个 kernel 的数据），多 TilingKey 算子必须**按场景分别运行** demo 再各自采集；(2) demo 初版负步长场景参数语义错误（begin/end/mask 组合触发 561103 报错），修正后三场景采集全部成功并归档为 round_001。

## 适用场景

- 工程形态：自定义 aclnn 算子的上板性能采集与验收（ops-registry-invoke 流程性能验收段）。
- API 路径：aclnn 两段式接口（GetWorkspaceSize → aclnn，循环内完整走两段）；`msprof op --warm-up=<n> --output=<dir> <binary> <args>`；perf_summary.py 归档脚本。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950，56 VectorCore），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only（采集流程配方本身；demo 骨架可复用于任意自定义 aclnn 算子，场景参数按算子替换）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：整图（GE）性能采集未验证；CPU/NPU 混合流程未覆盖；warm-up 与循环次数对统计稳定性的影响未做敏感性分析。

## 设计决策

- **按场景分跑 msprof**：一次 `msprof op` 只出单 kernel 数据（OpBasicInfo.csv 单条；多 TilingKey 算子一个 demo 进程会跑多个 kernel，只有其中一个被采集）。落地方案：demo 支持命令行选场景（`./perf_demo 1|2|3|all [loop]`），采集时每场景单独起一次 msprof，输出目录分开（msprof_scene1/2/3）。
- **循环内完整两段式**：性能循环里每次都重新 `GetWorkspaceSize` + `aclnn`（新建 executor），不做"一次探针 + 循环只调第二段"的捷径——保证测量的是真实调用路径（代码注释明确记录该取舍）。
- **tensor/IntArray 循环外创建**：输入输出 tensor 与 aclIntArray 在循环外分配复用，避免内存分配噪声进入计时；workspace 循环外分配一次。
- **数据量要够**：场景 shape 选择保证搬运量在 MB 级（初版 [1024] 只有 2KB，头开销占比 >90%，测量无意义；放大到 1M 元素 2MB）。
- 放弃的候选：单进程 `all` 一次采集——被"单 kernel 采集"约束否决；修改 perf_summary.py 使其聚合多 OPPROF——不改工具，分目录归档更简单。

## 系统信息

- 平台与工具链：Ascend950PR（56 VectorCore）、CANN 9.1.0、msprof op 采集（`--warm-up=10`）。
- 采集物路径约定：`docs/perf/round_001/scene*/`（每场景一个目录）。
- 相关文件：`perf_demo.cpp`（采集 demo）、`CMakeLists.txt`（构建）、`perf_summary.py`（结果汇总）。
- 环境要求：x86_64 宿主可运行 msprof；NPU 侧单算子执行。

## 完整代码

perf_demo.cpp（终版 = 初版 file_write + 负步长场景修正；`//` 注释均为原始代码自带）：

```cpp
// ============================================================================
// perf_demo.cpp
// BenchNoHarnessStridedSlice 性能测试 demo
//
// 用途: msprof op 性能采集，覆盖 3 个 TilingKey 场景
//   场景1 (TilingKey 1 CONTIGUOUS):  [1024,1024] fp16 strides=[1,1]  连续切片
//   场景2 (TilingKey 2 STRIDED_POS): [1024,1024] fp16 strides=[2,2]  间隔切片
//   场景3 (TilingKey 3 STRIDED_NEG): [1024]      fp16 strides=[-1]   负步长切片
//
// 用法:
//   ./perf_demo <scene> [loop_count]
//     scene: 1 | 2 | 3 | all (默认 all)
//     loop_count: 每个场景循环执行次数 (默认 100)
//
// 采集:
//   msprof op --warm-up=10 --output=./msprof_output ./perf_demo all 100
// ============================================================================

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <cstdlib>

#include "acl/acl.h"
#include "aclnn_bench_no_harness_strided_slice.h"

// fp16 类型 (x86_64 使用标准 _Float16)
using float16_t = _Float16;

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 辅助: 计算shape元素数
// ============================================================================
static int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;  // 0 维标量
    int64_t size = 1;
    for (auto d : shape) size *= d;
    return size;
}

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides[shape.size() - 1] = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// ============================================================================
// 创建 aclTensor (设备内存, 预填充数据)
// ============================================================================
static int CreateAclTensorFp16(const std::vector<int64_t>& shape,
                               void** deviceAddr, aclTensor** tensor) {
    size_t elemCount = static_cast<size_t>(GetShapeSize(shape));
    size_t bytes = elemCount * sizeof(float16_t);
    if (bytes == 0) bytes = 1;

    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclrtMalloc 失败: %d", ret);
        return ret;
    }

    // 填充输入数据 (x[i] = i mod 65536, fp16 范围内)
    std::vector<float16_t> hostData(elemCount);
    for (size_t i = 0; i < elemCount; i++) {
        hostData[i] = static_cast<float16_t>(i % 65536);
    }
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclrtMemcpy 失败: %d", ret);
        aclrtFree(*deviceAddr);
        return ret;
    }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

// ============================================================================
// 创建输出 aclTensor (设备内存, 零初始化)
// ============================================================================
static int CreateOutputAclTensorFp16(const std::vector<int64_t>& shape,
                                     void** deviceAddr, aclTensor** tensor) {
    size_t elemCount = static_cast<size_t>(GetShapeSize(shape));
    size_t bytes = elemCount * sizeof(float16_t);
    if (bytes == 0) bytes = 1;

    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclrtMalloc(output) 失败: %d", ret);
        return ret;
    }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

// ============================================================================
// 性能测试场景定义
// ============================================================================
struct PerfScene {
    std::string name;
    std::vector<int64_t> x_shape;
    std::vector<int64_t> y_shape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    int64_t begin_mask;
    int64_t end_mask;
    int64_t ellipsis_mask;
    int64_t shrink_axis_mask;
    int64_t new_axis_mask;
    int64_t expected_tiling_key;
    std::string desc;
};

// 场景1: CONTIGUOUS [1024,1024] fp16 strides=[1,1]
//   输出=[1024,1024], 最内层 srcStride=1 → TilingKey 1
//   搬运量: 1024*1024*2 = 2 MB (读), 2 MB (写)
static PerfScene MakeScene1() {
    PerfScene s;
    s.name = "scene1_CONTIGUOUS";
    s.x_shape = {1024, 1024};
    s.y_shape = {1024, 1024};
    s.begin = {0, 0};
    s.end = {1024, 1024};
    s.strides = {1, 1};
    s.begin_mask = 0;
    s.end_mask = 0;
    s.ellipsis_mask = 0;
    s.shrink_axis_mask = 0;
    s.new_axis_mask = 0;
    s.expected_tiling_key = 1;
    s.desc = "[1024,1024] fp16 strides=[1,1] 连续切片 (TilingKey 1)";
    return s;
}

// 场景2: STRIDED_POS [1024,1024] fp16 strides=[2,2]
//   输出=[512,512], 最内层 srcStride=2 → TilingKey 2
//   搬运量: 读 1024*1024*2=2MB(间隔读), 写 512*512*2=0.5MB
static PerfScene MakeScene2() {
    PerfScene s;
    s.name = "scene2_STRIDED_POS";
    s.x_shape = {1024, 1024};
    s.y_shape = {512, 512};
    s.begin = {0, 0};
    s.end = {1024, 1024};
    s.strides = {2, 2};
    s.begin_mask = 0;
    s.end_mask = 0;
    s.ellipsis_mask = 0;
    s.shrink_axis_mask = 0;
    s.new_axis_mask = 0;
    s.expected_tiling_key = 2;
    s.desc = "[1024,1024] fp16 strides=[2,2] 间隔切片 (TilingKey 2)";
    return s;
}

// 场景3: STRIDED_NEG [1048576] fp16 strides=[-1], begin_mask=1, end_mask=1
//   begin_mask=1(负步长→begin=dimSize-1), end_mask=1(负步长→end=-1, 含索引0)
//   输出=[1048576], 最内层 srcStride=-1 → TilingKey 3
//   等价于 x[::-1]（完整逆序），搬运量: 读 2MB, 写 2MB
//   注: 任务原文为 [1024] fp16，但 2KB 数据量过小（头开销占比 >90%），
//   改用 1M 元素 (2MB) 以获得稳定的带宽利用率测量。
static PerfScene MakeScene3() {
    PerfScene s;
    s.name = "scene3_STRIDED_NEG";
    s.x_shape = {1048576};  // 1M elements = 2MB fp16
    s.y_shape = {1048576};
    s.begin = {0};
    s.end = {0};
    s.strides = {-1};
    s.begin_mask = 1;  // 负步长: begin = dimSize-1
    s.end_mask = 1;    // 负步长: end = -1 (包含索引 0)
    s.ellipsis_mask = 0;
    s.shrink_axis_mask = 0;
    s.new_axis_mask = 0;
    s.expected_tiling_key = 3;
    s.desc = "[1048576] fp16 strides=[-1] bm=1,em=1 负步长逆序 (TilingKey 3, x[::-1])";
    return s;
}

// ============================================================================
// 运行单个场景的性能测试
// ============================================================================
static bool RunScene(const PerfScene& scene, aclrtStream stream, int loopCount) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("场景: %s", scene.name.c_str());
    LOG_PRINT("描述: %s", scene.desc.c_str());
    LOG_PRINT("循环次数: %d", loopCount);
    LOG_PRINT("========================================");

    // 1. 创建输入/输出 tensor (在循环外, 避免内存分配影响)
    void* x_dev = nullptr;
    void* y_dev = nullptr;
    aclTensor* x_tensor = nullptr;
    aclTensor* y_tensor = nullptr;

    if (CreateAclTensorFp16(scene.x_shape, &x_dev, &x_tensor) != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] 创建 x tensor 失败");
        return false;
    }
    if (CreateOutputAclTensorFp16(scene.y_shape, &y_dev, &y_tensor) != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] 创建 y tensor 失败");
        aclDestroyTensor(x_tensor);
        aclrtFree(x_dev);
        return false;
    }

    // 2. 创建 aclIntArray (在循环外, 复用)
    aclIntArray* begin_arr = aclCreateIntArray(scene.begin.data(), scene.begin.size());
    aclIntArray* end_arr = aclCreateIntArray(scene.end.data(), scene.end.size());
    aclIntArray* strides_arr = aclCreateIntArray(scene.strides.data(), scene.strides.size());

    // 3. 预先获取 workspace (GetWorkspaceSize 在循环外调用一次)
    //    注意: executor 每次调用需要重新创建(ACLNN 两段式约束)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnBenchNoHarnessStridedSliceGetWorkspaceSize(
        x_tensor, begin_arr, end_arr, strides_arr,
        scene.begin_mask, scene.end_mask, scene.ellipsis_mask,
        scene.shrink_axis_mask, scene.new_axis_mask,
        y_tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] GetWorkspaceSize 失败: %d", ret);
        aclDestroyIntArray(begin_arr); aclDestroyIntArray(end_arr); aclDestroyIntArray(strides_arr);
        aclDestroyTensor(x_tensor); aclDestroyTensor(y_tensor);
        aclrtFree(x_dev); aclrtFree(y_dev);
        return false;
    }
    // 释放这个探针 executor (后续循环中重新创建)
    // 注意: 标准做法是每次调用都 GetWorkspaceSize + aclnn，此处为了性能测试的公平性，
    //       每次循环都走完整两段式流程

    LOG_PRINT("workspaceSize = %lu bytes", workspaceSize);

    // 4. 分配 workspace (在循环外复用)
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] workspace 分配失败: %d", ret);
            aclDestroyIntArray(begin_arr); aclDestroyIntArray(end_arr); aclDestroyIntArray(strides_arr);
            aclDestroyTensor(x_tensor); aclDestroyTensor(y_tensor);
            aclrtFree(x_dev); aclrtFree(y_dev);
            return false;
        }
    }

    // 5. 性能循环: 完整两段式调用
    LOG_PRINT("开始性能循环...");
    for (int i = 0; i < loopCount; i++) {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        ret = aclnnBenchNoHarnessStridedSliceGetWorkspaceSize(
            x_tensor, begin_arr, end_arr, strides_arr,
            scene.begin_mask, scene.end_mask, scene.ellipsis_mask,
            scene.shrink_axis_mask, scene.new_axis_mask,
            y_tensor, &ws, &exec);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] 循环 %d: GetWorkspaceSize 失败: %d", i, ret);
            break;
        }

        ret = aclnnBenchNoHarnessStridedSlice(workspace, ws, exec, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] 循环 %d: aclnn 调用失败: %d", i, ret);
            break;
        }

        aclrtSynchronizeStream(stream);
    }
    LOG_PRINT("性能循环完成 (%d 次)", loopCount);

    // 6. 清理
    if (workspace) aclrtFree(workspace);
    aclDestroyIntArray(begin_arr);
    aclDestroyIntArray(end_arr);
    aclDestroyIntArray(strides_arr);
    aclDestroyTensor(x_tensor);
    aclDestroyTensor(y_tensor);
    aclrtFree(x_dev);
    aclrtFree(y_dev);

    LOG_PRINT("[OK] 场景 %s 完成", scene.name.c_str());
    return true;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[]) {
    std::string sceneArg = "all";
    int loopCount = 100;
    if (argc >= 2) sceneArg = argv[1];
    if (argc >= 3) loopCount = std::atoi(argv[2]);
    if (loopCount <= 0) loopCount = 100;

    LOG_PRINT("========================================");
    LOG_PRINT("BenchNoHarnessStridedSlice 性能测试 demo");
    LOG_PRINT("========================================");
    LOG_PRINT("场景参数: %s, 循环次数: %d", sceneArg.c_str(), loopCount);

    // 1. ACL 初始化
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        LOG_PRINT("[FAIL] aclInit 失败: %d", ret);
        return 1;
    }

    // 2. 打开设备
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] aclrtSetDevice 失败: %d", ret);
        aclFinalize();
        return 1;
    }

    // 3. 创建 stream
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] aclrtCreateStream 失败: %d", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    LOG_PRINT("ACL 初始化完成, deviceId=%d", deviceId);

    // 4. 运行选定场景
    bool allOk = true;
    if (sceneArg == "1" || sceneArg == "all") {
        if (!RunScene(MakeScene1(), stream, loopCount)) allOk = false;
    }
    if (sceneArg == "2" || sceneArg == "all") {
        if (!RunScene(MakeScene2(), stream, loopCount)) allOk = false;
    }
    if (sceneArg == "3" || sceneArg == "all") {
        if (!RunScene(MakeScene3(), stream, loopCount)) allOk = false;
    }

    // 5. 清理
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n========================================");
    LOG_PRINT("性能测试 %s", allOk ? "完成 (全部场景成功)" : "完成 (部分场景失败)");
    LOG_PRINT("========================================");
    return allOk ? 0 : 1;
}
```

（MakeScene3 为修正后终版：初版 begin=[1023]/end=[0]/mask 全 0 会触发 561103；其余函数与初版 file_write 逐字一致。）

CMakeLists.txt（verbatim）：

```cmake
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for BenchNoHarnessStridedSlice perf demo
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(perf_bench_no_harness_strided_slice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 路径
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH not set! source set_env.sh first.")
endif()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(OP_VENDOR_NAME bench_no_harness_strided_slice_custom)

message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# 头文件路径
set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)

# 查找自定义算子头文件
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_bench_no_harness_strided_slice.h
    PATHS
        ${CANN_ROOT}/opp/vendors/${OP_VENDOR_NAME}/op_api/include
        /usr/local/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/include
)
if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "Custom operator include dir not found. Install operator first.")
endif()
message(STATUS "Found custom op include: ${CUSTOM_OP_INCLUDE_DIR}")

# 查找自定义算子库
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
        /usr/local/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
)
if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "cust_opapi library not found. Install operator first.")
endif()
message(STATUS "Found custom op lib: ${CUSTOM_OP_LIBRARY}")

# 编译目标
add_executable(perf_demo perf_demo.cpp)
target_include_directories(perf_demo PRIVATE
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)
target_link_libraries(perf_demo
    ${CANN_ROOT}/lib64/libascendcl.so
    ${CANN_ROOT}/lib64/libopapi.so
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CUSTOM_OP_LIBRARY}
)
```

采集命令序列（verbatim，路径按 `<OP_DIR>`/`<CANN>` 替换）：

```bash
# 0. 环境
source <CANN>/set_env.sh
export LD_LIBRARY_PATH=<CANN>/opp/vendors/<vendor>/op_api/lib:${LD_LIBRARY_PATH}

# 1. 编译 demo
cd <OP_DIR>/tests/perf && mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 2. 冒烟（all 场景先跑通，3 次循环）
./perf_demo all 3

# 3. 按场景分别采集（关键：一次 msprof op 只出单 kernel 数据）
msprof op --warm-up=10 --output=./msprof_scene1 ./perf_demo 1 100
msprof op --warm-up=10 --output=./msprof_scene2 ./perf_demo 2 100
msprof op --warm-up=10 --output=./msprof_scene3 ./perf_demo 3 100

# 4. 归档（每场景 8 个 CSV + summary.txt 落到 docs/perf/round_001/scene*/）
SCENE1_DIR=$(ls -td ./msprof_scene1/OPPROF_* | head -1)
python3 <ops-profiling>/scripts/perf_summary.py "$SCENE1_DIR" <OP_DIR> --round-name=round_001/scene1_contiguous
# scene2/scene3 同理（--round-name=round_001/scene2_strided_pos / scene3_strided_neg）
```

## 验证信号

来源说明：性能验收 round_001 全流程实测（同机同日，三场景各自单独采集）。

- demo 冒烟：三场景 `[OK] ... 完成`，循环 3 次无失败（来源平台：950；证据等级：tool_observed）
- 分场景采集：三场景各产出 OPPROF 目录与 8 个 CSV；OpBasicInfo 分别记录 `_1`/`_257`/`_513` 三个 TilingKey kernel（BlockDim 56，满频 1650/1650）（来源平台：950；证据等级：tool_observed）
- "一次 msprof 只出单 kernel"约束观测：`msprof op ... ./perf_demo all 100` 的 OPPROF 目录 OpBasicInfo.csv 仅一条记录（56 核数据对应单 kernel），遂改为分场景采集（来源平台：950；证据等级：tool_observed）
- 归档：`docs/perf/round_001/` 下 28 个文件（3 场景 × 8 CSV + 3 场景 summary + 总 summary），perf_summary.py 输出 `Archived 8 CSV files to: ...`（来源平台：950；证据等级：tool_observed）

## 失效边界

- "一次 msprof op 只采集单 kernel"是 CANN 9.1.0 msprof 的观测行为（未从文档核实），多 kernel 需求下以分场景分跑为准；若新版 msprof 支持多 kernel 采集，本配方第 3 步可简化。
- demo 骨架绑定 fp16（`_Float16` + `ACL_FLOAT16`）；其他 dtype 需替换 tensor 创建与填充逻辑。
- 归档依赖 ops-profiling 技能的 perf_summary.py 与其 round 目录约定；脱离该工具时自行汇总 CSV。
- 推断可迁移（未验证）：其他 CANN 版本的 msprof op 参数（--warm-up/--output）与 CSV 结构预计兼容；未实测。

**未决项**：

- warm-up=10 与 100 循环的统计稳定性未做重复采集方差分析；正式验收可加多轮取中位数。
## 坏实践

- 一个进程跑完所有场景只调一次 msprof——只会拿到其中一个 kernel 的数据，其余场景白跑。
- 性能循环里复用同一个 executor 只调第二段接口——测的不是真实调用路径，计时失真（demo 注释中明确禁止的捷径）。
- 用 KB 级小 shape 测带宽——头开销占比 >90%，数据无意义。

## 误诊

- 无（本配方为成功路径沉淀；过程中的负步长 561103 失误属场景参数语义问题，修正后消除）。
