---
type: "Runbook"
title: errcode 264 workspace 标量 GM 访问崩溃
description: Ascend950(arch35) kernel 内对 GM workspace 的标量访问（GlobalTensor SetValue/ReinterpretCast/原始 __gm__ 指针写）触发 aiv task exception errcode 264 'The address for scalar to access GM is invalid'，而同 kernel 的 x/y 标量 GM 访问与 DataCopyPad 正常；6 种规避全部失败后回退单块（不使用 workspace）方案恢复 16/16 通过。
tags: [task.debugging, topic.memory, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# errcode 264 workspace 标量 GM 访问崩溃

## 现象

在开发去重算子（对标 `torch.unique`，ascend950 / arch35，CANN 9.1.0）迭代三的多块归并路径时，kernel 需要把分块排序的中间结果累积到 GM workspace。代码在 `Init()` 里用 `wsKeys_.SetGlobalBuffer((__gm__ uint16_t*)((__gm__ char*)workspace + wsKeysOffset_))` 绑定 workspace 区域，`Process()` 里做标量写。重建（`bash build.sh --soc=ascend950 -j8`）、重装（`bash build/custom_opp_ubuntu_x86_64.run --force --quiet`）后跑 ST Real 模式 `--scope=iter1`（16 个小 shape 用例），全部失败：

```
  K=1(golden), y_ok=0, inv_ok=1 FAIL
  ...（16/16 FAIL）
总计: 16
通过: 0
失败: 16
```

用日志环境变量把 NPU 错误打到 stdout：

```
ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn_bench_no_harness_unique --scope=iter1 2>&1 | rg "errorStr|Task run failed"
```

得到 aiv 任务异常与错误串：

```
[ERROR] RUNTIME(517905,test_aclnn_bench_no_harness_unique):... [task_recycle_cqrpt_base.cc:68]... PrintTaskErrorMsg:Task run failed, stream_id=61, pos=0, task_sn=2, sqe_type=1(aiv), errType=0x1(task exception), sqSwStatus=0.
The extend info: errcode:(264) errorStr: The address for scalar to access GM is invalid subErrType: 0x4.
```

注意该会话早前已确认的对照事实：同一 kernel 内 `inputGMX_.GetValue(0)`（输入 x 标量读）与 `outputGMY_.SetValue(0, val)`（输出 y 标量写）工作正常，`DataCopyPad` 批量搬运也正常——**只有 workspace 上的标量访问崩溃**。迭代二曾以 507035（向量核异常）形态复现过 workspace 标量写问题并移除该路径；本切片是迭代三在更大 workspace（多块归并的 wsKeys_/wsSortedKeys_ 区域）上的延续调试与最终收敛。

## 适用场景

- 工程形态：自定义算子 kernel 开发（op_kernel），需要在 kernel 内对 GM workspace 做小尺寸标量读写的场景（中间计数、K 回写、跨阶段状态）。
- API 路径：`GlobalTensor::SetGlobalBuffer(workspace + offset)` + `SetValue`/`GetValue`；`ReinterpretCast<T>()`；原始 `__gm__ T*` 指针写；tiling 侧 `GetWorkspaceSizes` 声明。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35 / DAV_3510），CANN 9.1.0；仅 950 有实测证据（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（950 平台上任何想对 GM workspace 做标量访问的 kernel）。
- 路线优先级：safety_fallback——"kernel 标量访问 GM workspace" 在 950 上按当前证据不可用，推荐回退为"结果全部驻留 UB / 直接写输出 GM"的单块方案；这是保守规避而非根因修复。fallback 声明：单块截断方案对 `totalNum > ubFactor` 的输入结果错误（非慢），是功能边界而非性能折中；多块方案待 workspace 访问机制澄清后重启。
- 明确不适用 / 未验证场景：a2/a3 等其他 SoC 未实测（未验证，推断可迁移——需上板复现）；经 `DataCopyPad` 批量搬运 workspace 的路径不受本卡结论约束（同轨迹实测该通路可用）；x/y 输入输出 GM 的标量访问不受影响。

## 判别方法

**已证事实（单句归因）**：kernel 内经 `SetGlobalBuffer(workspace + offset)` 绑定后的 `SetValue` 标量写（以及绕过 GlobalTensor 的原始 `__gm__ int32_t*` 指针写）触发 errcode 264 "The address for scalar to access GM is invalid" 的 aiv task exception，而同一 kernel 对 x/y GM 的标量访问正常——问题特定于 workspace 指针的标量访问通路。

**判别步骤**（区别于"代码写错地址"或"NPU 损坏"）：

1. 挂起/失败时先取真实 NPU 错误（不依赖测试侧 sync 返回值）：
   ```bash
   ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_aclnn_bench_no_harness_unique --scope=iter1 2>&1 | rg "errorStr|Task run failed"
   ```
   命中 `errcode:(264) ... The address for scalar to access GM is invalid` 即本卡现象。
2. 做最小对照：只保留 `inputGMX_.GetValue(0)` + `outputGMY_.SetValue(0, val)`（x/y 通路）→ 若通过，再单独加一行 workspace 标量写 → 若复现崩溃，即锁定 workspace 通路。

**六种已尝试且全部无效的规避**（不要重复尝试）：

1. workspace buffer 增大到 32 字节最小粒度——未解决；
2. 用 `(__gm__ char*)workspace + offset` 替代 `__gm__ uint8_t*` 基指针——未解决；
3. 跳过 offset 0、从 offset 32 开始访问——未解决；
4. `ReinterpretCast<int32_t>` 替代 `uint16_t` 的 `SetValue`——未解决；
5. 添加 V→MTE2 / MTE3→MTE2 流水线同步——未解决；
6. 绕过 GlobalTensor，直接 `__gm__ int32_t* wsRaw = (__gm__ int32_t*)((__gm__ char*)workspace_ + wsKeysOffset_); wsRaw[0] = 42;`——同样崩溃。

代表性的失败尝试代码（verbatim，替换进 `Process()` 开头后 16/16 崩溃）：

```cpp
    if (totalNum_ <= 0) {
        return;
    }

    // TEMP: minimal copy + wsKeys_ write test
    D_T_X val = inputGMX_.GetValue(0);
    wsKeys_.SetValue(0, 42);  // Test wsKeys_ write
    outputGMY_.SetValue(0, val);
    auto wsK = wsSortedKeys_.template ReinterpretCast<int32_t>();
    wsK.SetValue(0, 1);
    return;
```

原始 `__gm__` 指针版本（同样崩溃）：

```cpp
    // TEMP: test raw __gm__ pointer workspace access
    D_T_X val = inputGMX_.GetValue(0);
    // Use raw pointer instead of GlobalTensor
    __gm__ int32_t* wsRaw = (__gm__ int32_t*)((__gm__ char*)workspace_ + wsKeysOffset_);
    wsRaw[0] = 42;  // Write via raw pointer
    outputGMY_.SetValue(0, val);
    auto wsK = wsSortedKeys_.template ReinterpretCast<int32_t>();
    wsK.SetValue(0, 1);
    return;
```

**当前规避方案（safety_fallback，实测有效）**：回退为单块处理——所有元素（截断到 `ubFactor`）一次搬入 UB，排序/去重/输出全部在 UB 与输出 GM y 上完成，完全不触碰 workspace。修复后 `Process()` 开头（verbatim）：

```cpp
    if (totalNum_ <= 0) {
        return;
    }

    // 单块处理：所有元素放入 UB（ubFactor 内）
    // 对于 totalNum > ubFactor 的大 shape，当前仅处理前 ubFactor 个元素
    // TODO: 后续迭代添加多块归并支持
    int64_t n = (totalNum_ <= ubFactor_) ? totalNum_ : ubFactor_;

    auto inputLocal = inputQueue_.template AllocTensor<D_T_X>();
    DataCopyParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = static_cast<uint32_t>(n * DTYPE_SIZE);
    inParams.srcStride = 0;
    inParams.dstStride = 0;
    DataCopyPad(inputLocal, inputGMX_[0], inParams, {false, 0, 0, 0});
    inputQueue_.EnQue(inputLocal);
    inputLocal = inputQueue_.template DeQue<D_T_X>();
```

该方案上线后 iter1 16/16 通过、全 7 dtype × 2 inverse = 14 TilingKey 编译通过（7 个 .o 文件 = 14 变体），大 shape 成为已知截断限制（见失效边界）。

## 设计决策

- **为何回退而不是继续追 workspace**：多块归并强依赖 workspace 中间结果传递（Phase 1→2→3），在 6 种规避全部失败、且 x/y 通路与 DataCopyPad 通路均正常的情况下，判定该问题短期内无法在算子侧绕过；选择保交付（功能正确性优先），把大 shape 支持延后并立项 issue 跟踪。
- **被放弃的设计**：一次"多块处理：分块排序+去重，流式归并到 GM y"的 Process 重写（含块循环与标量 GM 归并）已动笔，因过于复杂且仍需 workspace/标量 GM 通路而回退到单块简化版。

## 预防

- 在 kernel 里需要"跨阶段小状态"时，优先考虑放输出 GM 的保留位或 UB 内完成，避免依赖 GM workspace 标量访问（950 当前证据下不可用）。
- 若必须用 workspace：先用最小 kernel（一行 `SetValue`）单独验证该通路再铺开实现，避免与业务逻辑混在一起排查。
- ST 失败先跑 `ASCEND_SLOG_PRINT_TO_STDOUT=1 ... | rg "errorStr|Task run failed"` 拿真实错误串再定位（264/507035 等 aiv 异常不会出现在测试侧的普通输出里）。

## 验证信号

来源说明：该去重算子迭代三联调的开发会话（ascend950，CANN 9.1.0），下述信号均为开发过程中实测。

- workspace 标量写版本 ST iter1 16/16 FAIL，`ASCEND_SLOG_PRINT_TO_STDOUT=1` 捕获 `errcode:(264) errorStr: The address for scalar to access GM is invalid subErrType: 0x4`（aiv task exception，pos=0/1 两个任务均失败）（来源平台：950；证据等级：tool_observed）
- 原始 `__gm__` 指针写版本同样 16/16 FAIL，同一错误串（来源平台：950；证据等级：tool_observed）
- 回退单块（不使用 workspace）后 ST iter1 16/16 PASS（K=1 用例 `y_ok=1, inv_ok=1 PASS`），编译 7 个 .o = 14 TilingKey 变体（来源平台：950；证据等级：tool_observed）
- 对照事实：同 kernel 内 `inputGMX_.GetValue(0)`/`outputGMY_.SetValue(0, val)`（x/y 标量通路）正常，`wsSortedKeys_.ReinterpretCast<int32_t>().SetValue(0, 1)` 曾在个别 offset 上通过而 `wsKeys_` 同地址失败——workspace 通路行为不稳定/不可依赖（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本卡规避方案（单块处理）只对 `totalNum ≤ ubFactor` 的输入正确；超过 ubFactor 的大 shape 被截断为前 ubFactor 个元素，输出错误（同轨迹 ST 中 10 个大 shape 用例因此失败，已立项 issue）。ubFactor 由 tiling 按 dtype 组给出（fp16 类 10464 / int32 类 7840 / int64 类 5216）。
- 其他 SoC（a2/a3 等）未实测；"未验证，推断可迁移"——需上板复现 errcode 264 才能确认同样的不可用性。
- `DataCopyPad` 批量搬 workspace 的通路实测可用，本卡不约束该路径。
- 若后续 CANN 版本修复 workspace 标量访问，本卡规避即过期。

**未决项**：

- **优先项：复现并追根因到机制层**——确认 GM_ADDR/框架 workspace 传参在 aiv 标量访问下为何无效（是否需要专用 workspace 访问 API、SetGlobalBuffer 多实例行为、或 264/95 错误码的平台语义），这是重启多块归并的前置条件。
- 大 shape（> ubFactor）多块归并支持：当前被本卡问题阻塞，需在 workspace 通路澄清后重写（流式归并或 GM y 直接归并的备选设计已讨论未实现）。
- `wsSortedKeys_` 偶发可写 / `wsKeys_` 同地址失败的矛盾现象未解释。
## 坏实践

- 在没有最小验证的情况下，把 workspace 标量访问嵌进多阶段业务逻辑（Phase 1→2→3）里——失败后无法快速二分出是业务 bug 还是通路 bug，本轨迹为此消耗了多轮重建-重装-重跑循环。
- 用测试侧"没有 errorStr 输出"推断 kernel 没崩——必须显式开 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 取证。

## 误诊

- 最初怀疑"workspace 为 null / GM_ADDR 比较失效"并加了 `if (workspace_ == 0)` 分支探针——该探针版本同样 16/16 崩溃，说明不是 null 判定问题而是访问本身（来源平台：950；证据等级：tool_observed）
- 一度认为"`static_cast<D_T_X>(-1)` 常量写入与 `GetValue` 读值有差异导致崩溃"——换成 `inputGMX_.GetValue(0)` 读值后仍崩，排除该假设（来源平台：950；证据等级：tool_observed）
- 曾把"`wsSortedKeys_` 某次通过、`wsKeys_` 同地址失败"解释为 GlobalTensor 实例差异——未获机制层证实，只能作为"行为不稳定"的观测线索保留。
