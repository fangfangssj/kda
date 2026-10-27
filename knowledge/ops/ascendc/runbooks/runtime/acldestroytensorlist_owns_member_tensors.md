---
type: "Runbook"
title: "aclDestroyTensorList 拥有成员勿重复销毁"
description: "aclDestroyTensorList 析构时已释放其成员 aclTensor 的元数据（列表拥有内含 tensor），清理循环再逐个 aclDestroyTensor 构成二次释放——C++ ST 表现为下一用例创建张量时段错误（延迟爆炸），PyTorch 长驻进程表现为退出 free(): double free detected in tcache 2；列表存在只销毁列表 + aclrtFree 设备内存。 复现增量：第三种暴露形态——独立示例进程清理段挂起（timeout EXIT=124）而非即崩，叠加 stdout 全缓冲造成\"零输出假挂起\"；排查先 setbuf/stderr 排除缓冲假象，再用同源码手工编译对照锁定。"
tags: [task.debugging, topic.implementation.memory_management, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:30:52Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# aclDestroyTensorList 拥有成员勿重复销毁

## 现象

同一缺陷在两种宿主形态下先后暴露（同一次联调会话、同一算子：三个 aclTensorList 输入 + double 标量 + aclTensorList 输出的 TensorList 型 aclnn 自定义算子，Ascend950PR / CANN 9.1.0）：

**形态一（C++ ST，延迟爆炸）**：executor 的 double free 修掉之后，第 1 个用例完整通过（含清理），第 2 个用例刚开头就段错误：

```text
[Real] 测试 - FP32 基础shape [1024,1024] listLen=1 scalar=1.0
  [PASS] MERE=1.76e-09, ...
  cleanup: done
[Real] 测试 - FP32 1D shape [4096] listLen=1 scalar=0.5
  creating tensors[0]...
Segmentation fault (core dumped)
```

在创建张量的打印与清理步骤间逐步加打印定位：崩溃点不在清理里（"cleanup: done" 已打印），而在**下一个用例**的 `GenerateTensorData`/`CreateAclTensor` 处——上一个用例的堆破坏延迟爆炸。当时的清理代码是：先 `aclDestroyTensorList(x1List)`，再循环对每个成员 `aclDestroyTensor(x1Tensors[t]); aclrtFree(x1Devs[t]);`。

**形态二（PyTorch 适配层，收尾即崩）**：适配层用 RAII 结构 `OpWorkspace` 管理 acl 资源，首版析构函数先逐个 `aclDestroyTensor` 再 `aclDestroyTensorList`。首个 NPU 用例计算结果完全正确（失败: 0、Golden 自测: PASS），但进程收尾时报：

```
free(): double free detected in tcache 2
```

值得注意的是：同工程的 C++ ST 测试在成功路径只调 `aclDestroyTensorList` + `aclrtFree`（不逐个销毁 tensor）不崩，而错误回退路径两者都调且**未崩**——短生命周期的独立进程掩盖了双重释放，长驻的 Python 进程则必然暴露。

## 适用场景

- 工程形态：C++ ST / PyTorch（torch_npu）适配层等长驻宿主程序，用 `aclCreateTensorList(const aclTensor* const*, uint64_t)` 从成员张量构造 TensorList 后的清理。
- API 路径：`aclCreateTensorList` / `aclDestroyTensorList` / `aclDestroyTensor` / `aclrtFree`（ACL C 接口所有权契约）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0。
- 适用粒度：platform_wide——ACL 资源所有权语义，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不经 `aclCreateTensorList` 包装、独立创建/销毁的 `aclTensor`（那时 `aclDestroyTensor` 是必须的）；其他 CANN 版本的所有权语义未验证（未验证，升级需重验）。

## 根因

`aclDestroyTensorList` 销毁列表时会一并销毁其包含的 `aclTensor` 元数据（列表拥有内含 tensor 的所有权）。析构/清理里先逐个 `aclDestroyTensor` 再 `aclDestroyTensorList`（或先销毁列表再逐个销毁成员），同一批 tensor 被释放两次。该堆破坏不一定在释放当场崩溃（free 检测可能未触发），而是在下一个 malloc/构建张量时以段错误形式出现——崩溃点与肇事点隔了一个用例，极难凭直觉定位；长驻进程（Python 宿主）则更早以 glibc abort 暴露。

## 解决配方

修复前（C++ ST 清理：销毁 TensorList 后又销毁成员）：

```cpp
    // 清理资源
    if (workspace) aclrtFree(workspace);
    // executor 由 CommonOpExecutorRun 内部析构，不显式调用 aclDestroyAclOpExecutor (CANN 9.1.0 上会导致 double free)
    aclDestroyTensorList(x1List);
    aclDestroyTensorList(x2List);
    aclDestroyTensorList(x3List);
    aclDestroyTensorList(yList);
    for (size_t t = 0; t < listLen; ++t) {
        aclDestroyTensor(x1Tensors[t]); aclrtFree(x1Devs[t]);     // ← double free
        aclDestroyTensor(x2Tensors[t]); aclrtFree(x2Devs[t]);     // ← double free
        aclDestroyTensor(x3Tensors[t]); aclrtFree(x3Devs[t]);     // ← double free
        aclDestroyTensor(yTensors[t]);  aclrtFree(yDevs[t]);      // ← double free
    }
```

修复后（C++ ST：TensorList 已释放成员元数据，只清理设备内存）：

```cpp
    // 清理资源
    if (workspace) aclrtFree(workspace);
    // executor 由 CommonOpExecutorRun 内部析构，不显式调用 aclDestroyAclOpExecutor (CANN 9.1.0 上会导致 double free)
    aclDestroyTensorList(x1List);
    aclDestroyTensorList(x2List);
    aclDestroyTensorList(x3List);
    aclDestroyTensorList(yList);
    for (size_t t = 0; t < listLen; ++t) {
        // 注意: aclDestroyTensorList 已释放 tensor 元数据, 不再显式 aclDestroyTensor
        // 仅释放 device 内存
        aclrtFree(x1Devs[t]);
        aclrtFree(x2Devs[t]);
        aclrtFree(x3Devs[t]);
        aclrtFree(yDevs[t]);
    }
```

PyTorch 适配层的等价修复（`OpWorkspace` 析构：列表存在只销毁列表；列表未创建（创建失败路径）才逐个销毁 tensor）：

```cpp
    ~OpWorkspace() {
        // aclDestroyTensorList 会销毁其包含的 aclTensor，因此：
        // - 列表已创建时: 仅销毁列表 (列表负责销毁内部 tensor)
        // - 列表未创建时 (创建失败): 逐个销毁 tensor
        if (x1_list) { aclDestroyTensorList(x1_list); x1_list = nullptr; }
        else { for (auto* t : x1_tensors) { if (t) aclDestroyTensor(t); } }
        if (x2_list) { aclDestroyTensorList(x2_list); x2_list = nullptr; }
        else { for (auto* t : x2_tensors) { if (t) aclDestroyTensor(t); } }
        if (x3_list) { aclDestroyTensorList(x3_list); x3_list = nullptr; }
        else { for (auto* t : x3_tensors) { if (t) aclDestroyTensor(t); } }
        if (y_list)  { aclDestroyTensorList(y_list);  y_list = nullptr; }
        else { for (auto* t : y_tensors)  { if (t) aclDestroyTensor(t); } }
        executor = nullptr;
    }
```

判定要点：`aclCreateTensorList` 成功后所有权转移给列表，调用方只保留"列表句柄 + 设备内存（`aclrtFree` 另行管理）"；`aclDestroyTensor` 只用于从未进入列表的 tensor（列表创建失败的回退清理）。

定位手段（延迟崩溃 → 用打印二分清理与创建序列）：在清理各步骤与下一用例创建张量处逐步打点（`cleanup: aclDestroyTensorList x1List` / `cleanup: tensor[%zu]` / `cleanup: done` / `creating tensors[%zu]...`），崩溃落在创建处即回看上一用例清理。

## 设计决策

- 选"列表优先、tensor 兜底"而非固定只销毁列表：`aclCreateTensorList` 可能部分失败（部分列表建成、其余失败），此时未入列的 tensor 必须逐个销毁避免泄漏；分支化后泄漏与崩溃都不发生。
- 设备内存（`aclrtMalloc` 的地址）仍归用户释放，本卡只豁免成员张量的**元数据**销毁。
- 长驻进程（Python/torch 宿主）里任何 acl 资源双重释放都会以 glibc abort 暴露——把它当作清理代码的正确性测试器。

## 预防

- 用 `aclCreateTensorList` 包装成员后，清理**只** `aclDestroyTensorList` + `aclrtFree` 设备内存；`aclDestroyTensor` 仅用于从未入列表的独立张量。
- "下一用例初始化时段错误、上一用例全过"是延迟堆破坏指纹：优先回看上一用例清理里对容器成员的重复销毁。
- 封装 acl 资源时先明确所有权：哪些句柄被容器接管，RAII 析构只销毁"自己拥有"的句柄；不要把错误回退路径的"全量销毁"复制到成功路径。

## 验证信号

来源说明：以下信号来自该算子首轮上板联调会话（C++ ST 打印二分定位 + PyTorch 适配层修复前后复跑）。

- 修复前（C++ ST）：用例 1 PASS 且 `cleanup: done` 打印后，用例 2 在 `creating tensors[0]...` 后 `Segmentation fault (core dumped)`（来源平台：950；证据等级：tool_observed）。
- 修复后（C++ ST，仅移除成员级 aclDestroyTensor，其余不动）：5 用例中 4 个通过、进程全程无崩溃（剩余 1 个失败为 tiling 缓存问题，精度形态完全不同，另卡覆盖）（来源平台：950；证据等级：tool_observed）。
- 最终（C++ ST）：配合 tiling 缓存规避后同 ST 5/5 通过、进程正常退出（最终验收 UT 14/14 + ST 5/5）（来源平台：950；证据等级：tool_observed）。
- 修复前（PyTorch 适配层）：首用例 PASS（失败: 0、Golden 自测: PASS）后进程收尾报 `free(): double free detected in tcache 2`（来源平台：950；证据等级：tool_observed）。
- 修复后（PyTorch 适配层）：单用例干净退出（无 glibc 报错）；全量 65 用例 NPU 通过 65 / 失败 0 / 跳过 0，进程 exit 0、无 double-free（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 所有权结论由"修复后 double-free/段错误消失"实验闭合，`aclDestroyTensorList` 内部实现未读源码；若未来版本改为不接管 tensor（纯容器），本配方需反向调整（届时逐个销毁 + 只释放列表容器）。未验证，升级需重验。
- 设备内存（`aclrtMalloc`/`aclrtFree`）与 tensor 描述符是两套资源，本卡只涉及描述符所有权；`aclDestroyTensorList` 是否释放成员张量关联的设备内存未单独验证（本例设备内存由用户 aclrtFree，行为正常）。
- 同期另一轨迹（executor 堆污染背景，见 CommonOpExecutorRun double free 卡）中"不销毁 TensorList 也崩"的观测与本卡不冲突——那属于 aclnn 执行后进程堆已污染的场景，任何释放都可能崩；本卡结论适用于堆未污染时的所有权清理。

**未决项**：

- `aclDestroyTensorList` 的所有权语义建议以官方头文件注释/文档复核（本轨迹以行为实验闭合，未读实现）。
## 坏实践

- 成功路径同时调 `aclDestroyTensorList` 和逐个 `aclDestroyTensor`——在独立短命进程里"没崩"不代表正确（C++ ST 的错误回退路径正是这样活下来的）。
- 清理代码按"创建的都要销毁"直觉全量 Destroy——容器吸收成员所有权后即为双重释放。
- 崩在创建处就怀疑输入数据/shape 生成逻辑，不回看上一用例清理。
- 把错误路径的防御性清理（两套都销毁）当作通用模板复制进 RAII 析构。

## 误诊

- 曾假设 `aclCreateTensorList` 的 listLen 形参类型（size_t vs uint64_t）或 ACL 库内部问题导致 double free，均未命中；打印二分后才锁定成员级重复销毁。

## 后续增量（2026-09-17，并行策展批：示例进程清理段挂起形态与 stdout 缓冲假象）

同一缺陷的第三种暴露形态（同算子家族的 aclnn 调用示例 `examples/test_aclnn_*.cpp`，Ascend950PR + CANN 9.1.0）：示例计算结果全部正确，但进程在**打印完结果后的资源清理阶段挂起**——`timeout 120` 运行得 `EXIT=124`，且重定向到文件时 `wc -l` 报 0 行，形成"main 之前就挂死、一行输出都没有"的假象。真相是两层叠加：

1. 清理代码先 `aclDestroyTensorList` 再循环 `aclDestroyTensor`（本卡根因的双重释放），本次不表现为即崩/tcache abort 而表现为清理段挂死（挂起的具体内部接口未定位——本机 gdb 无 ptrace 权限、strace 未安装，只闭合了所有权层因果）；
2. stdout 重定向到文件时全缓冲，进程被 timeout 杀掉前缓冲未刷出——"零输出"是缓冲假象，不代表挂起点在输出之前。

修复同本卡配方（删除 TensorList 之后的逐 tensor `aclDestroyTensor`，保留 `aclrtFree` 释放设备内存），并保留 `setbuf(stdout, NULL)`：示例常被管道/重定向/timeout 包裹运行，不设 setbuf 会在挂起/超时场景丢失全部已打印输出、严重误导排查方向。

排查"timeout 杀进程 + 零输出"类挂起的顺序：先加 `setbuf(stdout, NULL)` 或改 `fprintf(stderr, ...)` 排除缓冲假象 → 再用"同环境正常样本对照 + g++ 手工编译同源码"二分锁定源码（本例：ST 二进制正常 / 只做 aclInit 的最小程序正常 / g++ 手工编译示例同源码仍挂 → 锁定示例自身；曾先后怀疑并否定 rpath、CMake 链接方式、静态初始化等环境假设）。

### 复现增量验证信号

来源说明：该算子 aclnn 示例联调会话的挂起定位与修复复跑。

- 修复前：`timeout 120` 运行 `EXIT=124`、输出文件 0 行；加 stderr 调试后计算结果正确（`Tensor[0][0] = 3.500000`、`Tensor[1][0] = 18.500000`，与手算期望 3.5/18.5 一致）但 `EXIT=124`（挂起在清理段）（来源平台：950；证据等级：tool_observed）。
- 修复后：g++ 手工编译与 CMake 重编运行均 `EXIT=0`、结果正确，`DEBUG: destroying tensor lists → … → aclFinalize → done` 全链走通；`bash run.sh` 端到端（重编+运行+判定）`执行结果: PASS`（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 双重释放的三种实测形态汇总：C++ ST 下一用例创建时段错误（延迟爆炸）、PyTorch 长驻进程收尾 tcache abort、独立示例进程清理段挂起（本增量）——同一根因，暴露形态随宿主形态与堆状态变化。
- 挂起发生在清理阶段的具体 ACL 接口内部（aclDestroyTensor 自身还是 aclFinalize 的资源回收死锁）未用调试器定位。
- 坏实践：排查挂起先怀疑环境（rpath/LD_LIBRARY_PATH/工作目录）而不先做"同源码手工编译"对照；在重定向/管道场景依赖默认 stdout 缓冲判断执行进度。