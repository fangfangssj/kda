---
type: "Runbook"
title: "aclnn 带宽计时用事件法剔除 host 开销"
description: aclnn 两段式接口的端到端 per-call 计时把 GetWorkspaceSize/描述符创建/workspace 分配/同步的 host 开销摊进内核时间，带宽被系统性低估（本例约 6%），足以翻转"≥80% 峰值"类阈值判定；应预构建 executor 并用 aclrtEvent 只计 device 段，msprof aiv_time 交叉印证。
tags: [task.testing, topic.performance.analysis, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:30:52Z'
updated_at: '2026-09-17T02:30:52Z'
platforms: ["950"]
---

# aclnn 带宽计时用事件法剔除 host 开销

## 现象

为验收 softplus_v2_grad 的带宽指标（L1 [4096,4096] fp32，2R+1W 共 201326592 字节/次），先后写了两版基准：

- 第一版端到端计时：每个迭代里做 `aclCreateTensor`×3 → `aclnnSoftplusV2GradGetWorkspaceSize` → `aclrtMalloc`(workspace) → `aclnnSoftplusV2Grad` → `aclrtSynchronizeStream` → 释放，用 `std::chrono` 计整批时间。
- 第二版 kernel-only 计时：先离线预构建 N=100 个 `aclOpExecutor`（host 侧描述符/workspace 全部提前做好），循环内只 `aclnnSoftplusV2Grad(...)` 提交 launch，用 `aclrtRecordEvent`/`aclrtEventElapsedTime` 只计 device 上执行的时间。

两个口径的成对对比（同设备同 shape 同输入输出，N=100）：端到端口径的每调用耗时比事件法高约 6%，换算带宽被低估约 6%（端到端相对 kernel-only 约 0.94x）。对"≥80% 峰值"这类阈值判定，6% 的系统性低估足以翻转结论。期间还把 host 侧工作挪出计时窗口后确认过：设备侧耗时真实（host 开销不是 device 耗时的主因，但确实贡献了口径差）。

## 适用场景

- 工程形态：aclnn 直调基准/验收测试（C++ host 驱动），尤其做带宽、加速比这类对毫秒敏感的指标。
- API 路径：ACL C API `aclrtCreateEvent/aclrtRecordEvent/aclrtEventElapsedTime/aclrtSynchronizeEvent`；aclnn 两段式 `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`；交叉印证用 msprof `--op` 的 `op_summary.csv`（aiv_time）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR，56 AIV）+ CANN 9.1.0。
- 适用粒度：platform_wide（计时方法论，与算子无关）。
- 路线优先级：recommended_fastpath（阈值判定一律用 kernel-only 事件法口径，端到端口径并列作对照）。
- 明确不适用 / 未验证场景：单次调用延迟（冷启动、首次编译/加载）评测——那本来就要含 host 开销，不适用本卡的"剔除"口径；CPU 侧 prof（clock()/gettimeofday）不受此影响但测不到 device 真实时长。

## 根因

aclnn 接口是两段式：`GetWorkspaceSize` 阶段在 host 上做 shape/dtype 校验、tiling 计算、workspace 尺寸推导并产出 `aclOpExecutor`；`aclnn<Op>` 阶段才把执行器提交到 stream。若计时窗口覆盖"每调用完整流程"（创建张量描述符 → 查 workspace → 分配 → 提交 → 同步），则 host 侧串行开销与 device 执行时间叠加，`带宽 = 字节量 / (device 时间 + host 摊销)` 被低估。`aclrtEvent` 记录在 stream 上，只随 device 侧执行推进，天然剔除 host 段。

## 解决配方

修正后的测量骨架（kernel-only 事件计时 + 端到端对照，两个口径同程序同输入输出、数字可直接对比；CHECK_ACLNN 以 autogen 头的 `constexpr aclnnStatus OK = 0` 判定）：

```cpp
    aclrtEvent start, stop;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&stop));

    // pre-create executors for N iterations (host setup amortized but each launch fresh)
    const int N = 100;
    struct Exec { aclOpExecutor* ex; uint64_t ws; void* wsp; };
    std::vector<Exec> execs(N);
    for (int i=0;i<N;i++) {
        execs[i].ws=0; execs[i].ex=nullptr; execs[i].wsp=nullptr;
        aclTensor* go = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dGradOut);
        aclTensor* sf = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dSelf);
        aclTensor* ot = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dOut);
        CHECK_ACLNN(aclnnSoftplusV2GradGetWorkspaceSize(go,sf,1.0f,20.0f,ot,&execs[i].ws,&execs[i].ex));
        if (execs[i].ws>0) CHECK_ACL(aclrtMalloc(&execs[i].wsp, execs[i].ws, ACL_MEM_MALLOC_HUGE_FIRST));
        aclDestroyTensor(go); aclDestroyTensor(sf); aclDestroyTensor(ot);
    }

    // warmup
    for (int w=0; w<10; w++) {
        CHECK_ACLNN(aclnnSoftplusV2Grad(execs[0].wsp, execs[0].ws, execs[0].ex, stream));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // kernel-only timing: record event before first launch, launch N, record after, sync.
    CHECK_ACL(aclrtRecordEvent(start, stream));
    for (int i=0;i<N;i++) {
        CHECK_ACLNN(aclnnSoftplusV2Grad(execs[i].wsp, execs[i].ws, execs[i].ex, stream));
    }
    CHECK_ACL(aclrtRecordEvent(stop, stream));
    CHECK_ACL(aclrtSynchronizeEvent(stop));
    float ms = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&ms, start, stop));
    double sec = ms / 1000.0;
    double kernelBw = bytesPerCall * N / sec;

    // end-to-end (host setup + launch + sync) amortized over a single stream loop — 对照口径
    auto t0 = std::chrono::steady_clock::now();
    for (int i=0;i<N;i++) {
        uint64_t ws=0; aclOpExecutor* ex=nullptr;
        aclTensor* go = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dGradOut);
        aclTensor* sf = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dSelf);
        aclTensor* ot = aclCreateTensor(shape,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,shape,2,dOut);
        CHECK_ACLNN(aclnnSoftplusV2GradGetWorkspaceSize(go,sf,1.0f,20.0f,ot,&ws,&ex));
        void* wsp=nullptr; if(ws>0) CHECK_ACL(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACLNN(aclnnSoftplusV2Grad(wsp,ws,ex,stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if(wsp) aclrtFree(wsp);
        aclDestroyTensor(go); aclDestroyTensor(sf); aclDestroyTensor(ot);
    }
    auto t1 = std::chrono::steady_clock::now();
```

要点：① 每次迭代用独立 executor（避免复用单个执行器引入额外串行化，也贴近真实调用形态）；② warmup 排除首次加载抖动；③ 两个口径同程序同输入输出，数字可直接对比；④ 需要绝对印证时用 msprof `--op` 看 `aiv_time`。

## 设计决策

- 选 aclrtEvent 而非只在 host 计时后减去"预估 host 开销"：事件时间戳由 device 侧推进，无需建模 host 段，口径干净。
- 保留端到端口径作对照而非弃用：端到端数字对"用户视角每调用成本"仍有意义（真实部署每调用都要付 host 段），报告时两个口径并列、注明分母字节量。
- msprof 作第三方印证：`op_summary.csv` 的 `aiv_time` 换算带宽与事件法结果相符（差 <1%），排除测量法系统性偏差。

## 预防

- 写 aclnn 基准时默认两个口径都打：kernel-only（事件法）与 end-to-end（含 host），并在报告里标注口径；阈值判定一律用 kernel-only。
- 阈值判定前先算一遍"host 开销占比"量级，若与判定裕量同量级，必须换事件法。

## 验证信号

来源说明：以下信号来自一次 softplus_v2_grad 带宽验收的计时口径修正（同设备同 shape，N=100 的成对对比）。

- 成对口径对比：端到端（host setup + sync）相对 kernel-only（aclrtEvent，预构建 executor）每调用耗时高约 6%，即带宽被低估约 6%（端到端约 0.94x）（来源平台：950；证据等级：tool_observed）。
- msprof 交叉印证：Block Num=56、`op_summary` 的 aiv_time 换算带宽与事件法一致（差 <1%）（来源平台：950；证据等级：tool_observed）。
- 更早一轮把 host 侧工作挪出计时窗后 device-only 耗时与事件法一致，确认设备真实耗时（来源平台：950；证据等级：tool_observed）。

## 性能影响

- 同算子内成对相对对比（参照路线=kernel-only 事件法口径）：端到端口径带宽被 host 摊销压低约 6%（来源平台：950）。无绝对通用预期收益，具体占比随 host 段工作量与 kernel 时长变化。

## 失效边界

- 事件法测的是"批提交的 device 总时长/N"：若 kernel 极短（微秒级）而 launch 间隙大于执行时长，流上可能出现空泡，事件窗口包含空泡时带宽仍被低估——必要时加大 N 并确认提交端不成为瓶颈。
- 每 iteration 独立 executor 会占内存（N×workspace）；workspace 大时用小 N 或复用（复用会轻微改变串行行为，需注明）。
- 未验证，推断可迁移：其他 CANN 版本 aclrtEvent 语义一致，推断可迁移但未实测。

**未决项**：

- 无（根因为计时窗口构成，机制明确；host 段各组成部分的耗时分解未做，对结论非必需）。
## 坏实践

- 把 `std::chrono` 计的端到端时间直接当"内核时间"报带宽（本例会低估 6%）。
- 阈值判定（如 ≥80%）用端到端口径、而优化对比用 kernel-only 口径——口径混用会让结论不可比。

## 误诊

无（口径差异在对照实验中直接量化）。
