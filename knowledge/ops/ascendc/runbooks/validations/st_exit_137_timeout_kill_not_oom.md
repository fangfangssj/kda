---
type: "Runbook"
title: ST 进程退出码 137：区分 host OOM 与 kernel 未完成
description: aclnn ST 大 shape 用例进程被 SIGKILL（退出码 137）且日志截断时先别按 OOM 排查；排除 host 内存后用阶段 stderr 插桩定位阻塞点——打印停在 aclnn ret=0 之后、aclrtSynchronizeStream 之前，说明 host 下发完成、kernel 未执行完，是被执行工具超时杀掉；修复 kernel 并行度后 sync ret=0 出现，判别链正反闭环。
tags: [task.debugging, topic.implementation.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T04:06:11Z'
updated_at: '2026-09-17T04:06:11Z'
platforms: ["950"]
---

# ST 进程退出码 137：区分 host OOM 与 kernel 未完成

## 现象

在 Ascend950PR（CANN 9.1.0）上跑自定义算子 BenchNoHarnessConv2d 的 ST Real 全量测试（C++ 程序顺序执行 aclnn 两段式接口：`aclnnBenchNoHarnessConv2dGetWorkspaceSize` → `aclnnBenchNoHarnessConv2d` → `aclrtSynchronizeStream`）：小程序用例正常，运行到大规模用例时测试进程被 SIGKILL 杀死，退出码 137，日志截断在该用例中段，没有任何错误输出。同样的现象在全量运行、单用例复跑、nohup 后台、`stdbuf -oL` 行缓冲等多种方式下都复现。首轮排查按 OOM 方向走了很远（cgroup 限额、/proc/meminfo、OpenMP 线程栈、容器总 RSS、strace），全部无法证实内存不足；真实原因是 kernel 在 NPU 上执行时间过长，进程阻塞在 `aclrtSynchronizeStream` 上，被外层执行工具的超时机制 SIGKILL。

## 适用场景

- 工程形态：aclnn 两段式接口 ST 测试（C++ 测试程序 + NPU 执行），大 shape / 大计算量用例；执行命令外层有超时机制（shell 工具、CI runner）。
- API 路径：`aclnnXxxGetWorkspaceSize` / `aclnnXxx` / `aclrtSynchronizeStream` 的两段式语义（调用返回 0 只代表下发成功，流同步返回才是 kernel 执行完成点）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该判别流程与平台无关地适用于任何 aclnn ST 长时间无输出被杀的场景；本会话在 950 上验证。
- 路线优先级：recommended_fastpath（鉴别流程，先判别再定方向）。
- 明确不适用 / 未验证场景：程序段错误（139/segfault）、编译失败、NPU 侧报错码（507035 等）走各自错误码路径；无超时机制的环境下慢 kernel 表现为"永远挂着"而非 137。

## 判别方法

判定规则（先看退出码，再看插桩落点）：

1. 退出码 137 = 128 + 9（SIGKILL）。SIGKILL 只有两个常见来源：cgroup OOM killer 或外部工具超时。先查 `dmesg` 有无 OOM kill 记录。
2. 排除 host OOM：`/proc/meminfo`（MemTotal/MemAvailable）、cgroup `memory.limit_in_bytes` 与 usage、容器内全部进程总 RSS（`ps aux` 求和）、用 Python 试分配大块内存验证限额真实生效。注意 cgroup v1 文件可能因权限读不到内容，不要把"读不到"当成"有限额"。
3. 阶段插桩定位阻塞点：测试程序每个执行阶段后 `fprintf(stderr, ...) + fflush(stderr)`（stderr 直写，重定向到文件时必须 fflush，否则缓冲会伪造"最后一条打印"），覆盖：数据生成 → 量化/转二进制 → 创建 aclTensor → GetWorkspaceSize → aclrtMalloc → aclnn 调用 → aclrtSynchronizeStream。
4. 读日志最后一条打印：
   - 停在 `aclnnXxx ret=0` 之后、无 `aclrtSynchronizeStream ret=` 打印 ⇒ host 下发完成、kernel 未执行完（过慢或 hang）⇒ 查 kernel 执行时间（切分并行度、算法路径），不是内存问题；
   - 停在 GetWorkspaceSize / aclnn 调用之前 ⇒ host 侧问题（数据生成、tensor 创建、内存分配）。
5. `npu-smi` 确认 NPU Health=OK、无残留进程占用。

探针代码（本会话实际插入的插桩，逐阶段）：

```cpp
void GenerateData(ConvData& data, const TestCase& tc, uint32_t case_seed) {
    RNG rng(case_seed);

    fprintf(stderr, "[DEBUG] GenerateData step1: alloc raw (x=%zu f=%zu b=%zu)\n",
            (size_t)tc.x_elems, (size_t)tc.filter_elems, (size_t)tc.bias_elems);
    fflush(stderr);

    // 1. 生成原始 float 随机数据 (标准正态分布)
    std::vector<float> x_raw(tc.x_elems);
    std::vector<float> filter_raw(tc.filter_elems);
    std::vector<float> bias_raw(tc.bias_elems);

    fprintf(stderr, "[DEBUG] GenerateData step2: FillNormal\n");
    fflush(stderr);

    rng.FillNormal(x_raw);
    rng.FillNormal(filter_raw);
    rng.FillNormal(bias_raw);

    fprintf(stderr, "[DEBUG] GenerateData step3: QuantizeToDType\n");
    fflush(stderr);

    // 2. 量化到目标 dtype 再转回 float (模拟设备端数据精度)
    QuantizeToDType(x_raw, data.x, tc.dtype);
    QuantizeToDType(filter_raw, data.filter, tc.dtype);
    QuantizeToDType(bias_raw, data.bias, tc.dtype);

    fprintf(stderr, "[DEBUG] GenerateData step4: ComputeConv2dGolden (y=%zu)\n", (size_t)tc.y_elems);
    fflush(stderr);

    // 3. CPU golden 计算 (使用量化后的数据, fp32 累加)
    ComputeConv2dGolden(data.x, data.filter, data.bias, data.golden, tc);

    fprintf(stderr, "[DEBUG] GenerateData done\n");
    fflush(stderr);
}
```

```cpp
    // 5. 分配 workspace 并执行
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[DEBUG] RunRealTest: aclrtMalloc workspace ret=%d\n", (int)ret);
        fflush(stderr);
        if (ret != ACL_SUCCESS) {
```

```cpp
    ret = aclrtSynchronizeStream(stream);
    fprintf(stderr, "[DEBUG] RunRealTest: aclrtSynchronizeStream ret=%d\n", (int)ret);
    fflush(stderr);
    if (ret != ACL_SUCCESS) {
```

host 内存排查探针（本会话实际执行）：`free -g` / 读 `/proc/meminfo`；`cat /proc/self/cgroup` + 读对应 cgroup memory 文件；`ps aux` 求容器总 RSS；`python3 -c` 循环分配 100MB×50 验证限额。

## 原始观测

- 修复定位阶段（带插桩、300 秒超时运行 L0-032 单用例）：工具报 `shell tool terminated command after exceeding timeout 300000 ms`；日志完整走完 `[DEBUG] GenerateData step1..done`、`[DEBUG] RunRealTest: GenerateData done, step FloatToBinary`、`FloatToBinary done, step CreateAclTensor`、`CreateAclTensor done, step GetWorkspaceSize`、`GetWorkspaceSize ret=0, workspaceSize=512`、`aclrtMalloc workspace ret=0`、`aclnnBenchNoHarnessConv2d ret=0`，然后终止——**没有** `aclrtSynchronizeStream ret=` 打印。
- `/proc/meminfo`：MemTotal 754GB、MemAvailable 696GB；cgroup v1 limit 32GB 但容器内 58 个进程总 RSS 仅 4.8GB；Python 顺序分配到 5GB 无压力；dmesg 无 OOM kill 记录；npu-smi Health=OK。

## 解释

退出码 137 本身不区分 OOM 与超时。本会话的判别链：host 内存证据全部正常（步骤 2）→ 插桩落点在 `aclnn ret=0` 之后、sync 之前（步骤 4）→ host 侧两段式调用已成功下发，进程阻塞等待 kernel 执行完成 → 结论是 kernel 执行过慢（本次根因为 tiling 多核切分退化导致并行度坍缩），进程被外层工具超时 SIGKILL。修复 kernel 并行度后，同一插桩程序出现 `aclrtSynchronizeStream ret=0` 打印、用例 PASS——判别链路正反两面闭环。

## 验证信号

以下信号来自该算子联调验证会话（大 shape 用例反复被杀的定位过程）。

- 多次运行退出码均为 137 且日志截断在大用例处；带插桩 + 300 秒超时复跑：日志终止于 `[DEBUG] RunRealTest: aclnnBenchNoHarnessConv2d ret=0`，无 sync 返回打印（来源平台：950；证据等级：tool_observed）。
- host 内存排查：MemTotal 754GB / MemAvailable 696GB；容器内全部进程总 RSS 4.8GB；Python 试分配 5GB 成功；dmesg 无 OOM 记录（来源平台：950；证据等级：tool_observed）。
- 修复 kernel 并行度后同一插桩程序：`aclrtSynchronizeStream ret=0` 打印出现，L0-032 PASS（MERE=1.76e-04，16,777,216 元素）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- stderr 重定向到文件时若不 `fflush`，缓冲可能让"最后一条打印"失真——本例所有插桩均带 `fflush(stderr)`。
- 本判别只把问题定位到"sync 阻塞 = kernel 未完成"，不区分"kernel 慢"与"kernel 死锁"；进一步区分需 plog 时间戳或 profiling（未验证）。
- 执行工具无超时或超时极长时，慢 kernel 表现为进程长期挂起而非 137。
- cgroup v1 文件可能因权限读不到内容——"读不到限额"不等于"存在 32GB 限额"，需用分配实验验证。
- 平台边界：判别链在 Ascend950PR + CANN 9.1.0 实测；其他平台与 CANN 版本未验证，判别方法本身与平台无关（未验证，推断可迁移）。

**未决项**：

- "kernel 慢"与"kernel 死锁"的区分方法（plog 时间戳对比、profiling 采集）未在本会话验证，列为下一测量点。
## 坏实践

- 看到 137 直接按 OOM 排查，在 cgroup/meminfo/线程栈上耗数小时（本会话真实发生）；先做插桩定位成本远低于此。
- 子集用例通过就当作全量没问题（子集往往不含真正的大用例）。
- 用 nohup/后台方式绕超时——进程照样会被杀，且日志为空更难定位。

## 误诊

- OOM 理论（cgroup 32GB 限额 + 大用例内存）：限额存在但用量远未触及，dmesg 无记录，最终排除。
- 曾怀疑特定用例（对称填充 L0-016）触发、NPU 残留 python3 进程资源冲突：分别被子集复跑通过、npu-smi 健康检查排除。
- 曾以"日志文件为空"推断程序启动即被杀：实际是重定向缓冲未刷新 + 程序在后续阶段才阻塞；插桩 + fflush 后才看清真实落点。
