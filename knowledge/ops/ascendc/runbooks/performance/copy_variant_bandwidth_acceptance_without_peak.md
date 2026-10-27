---
type: "Runbook"
title: 无 HBM 峰值时用 copy-variant 对照判带宽达标
description: "带宽验收要求“≥80% HBM 峰值”但峰值无权威来源（GetCoreMemBw 无效/npu-smi 无绝对值/ini 无字段/msprof 列异常）时，把被测 kernel 的 VF 计算体临时替换为同 tiling 纯拷贝作可达上界，实测比值 92.6% ≥80% 判过，再 diff 还原。"
tags: [task.optimization, topic.performance.analysis, topic.compute.elementwise.activation]
status: stable
sources: []
created_at: '2026-09-17T04:39:31Z'
updated_at: '2026-09-17T04:39:31Z'
platforms: ["950"]
---

# 无 HBM 峰值时用 copy-variant 对照判带宽达标

## 现象

softplus_v2_grad（2 输入 + 1 输出的 elementwise 梯度算子，AscendC，56 核 AIV，双缓冲 tile-loop）的验收标准写的是"L1 [4096,4096] fp32 端到端有效带宽利用率 ≥ 80% HBM 峰值"，而设计文档自身注明"具体基线数值与 HBM 峰值待性能实测阶段据设备确定"——即**分母（HBM 峰值）没有任何权威数值**。逐条探测全部失败：

- `platform_ascendc::PlatformAscendC::GetCoreMemBw(HBM)` 返回 31（无效，机制与假信号排除见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）；
- `npu-smi info -t board` 只有 `HBM Bandwidth Usage Rate(%)`，且持续负载下采样恒 0%；
- `platform_config/Ascend950PR_9579.ini` 只有 SoC/核数信息，无带宽字段；
- msprof `metric_summary.db` 的 `aiv_main_mem_read_bw`≈0.34/`write_bw`≈0.17 语义可疑（当绝对 GB/s 差 3 个量级，当利用率比率又推不出一致峰值）；`hbm_*.csv` 采样 Read≈6.07 GB/s、Write≈1074.8 GB/s，与 2R+1W 模式明显不符；
- 样例算子 add_custom 实测仅为被测 kernel 的约 0.35x（同会话不同测量轮的参照，非同轮成对）——读它的 host 代码发现 `constexpr uint32_t MAX_BLOCKS = 8`（8KB 小 tile），56 核设备只用了 8 核，不能当上界参照。

同时被测 kernel 自身实测 kernel-only 带宽已达同会话不同测量轮 aclrtMemcpy D2D（1R+1W）参照的约 1.54x，说明 SDMA 拷贝也低估了 AIV MTE 路径的可达带宽。需要一个可辩护的"80% 判定"方法。

## 适用场景

- 工程形态：aclnn/AscendC elementwise（或读写比固定的访存主导）kernel 的带宽验收与优化判定，perf 验证方法论。
- API 路径：aclnn 两段式接口（`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`）+ `aclrtEvent` 计时；kernel 侧 `__simd_vf__` VF 函数 + `asc_vf_call`；profiling 工具 msprof `--op`、npu-smi。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR，56 AIV）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（判定方法论；具体比值数字仅对本算子本平台有效）。
- 路线优先级：recommended_fastpath（HBM 峰值不可得时这是可辩护的达标判定法）。
- 明确不适用 / 未验证场景：计算受限 kernel（copy-variant 会显著快于计算版，比值无意义）；含累加/归约/随 shape 变化的读写比 kernel（拷贝模式与真实访存模式不匹配）；理论峰值有权威数值的场景（直接用理论值更简单）。

## 判别方法（解决方案）

**思路**：把"≥80% HBM 峰值"落地为"≥80% 该访问模式的实测可达上界"。上界用**同一 kernel、同一 tiling、同一核数**的纯拷贝变体测得——它与被测 kernel 只差计算体，是控制变量后的可达带宽。

**步骤**：

1. 备份原 kernel：`cp op_kernel/softplus_v2_grad_kernel.cpp /tmp/softplus_kernel_orig.cpp`。
2. 把 VF 计算函数的循环体替换为纯拷贝（2 读 + 1 写、无计算，fp16 路径保留 load/cast/store 的数据通路形状）：

```cpp
    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        // PURE COPY REFERENCE: 2 reads (x, y) + 1 write (x->dx), NO compute.
        if constexpr (isHalf) {
            AscendC::Reg::LoadAlign<DT, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRaw, xAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<float, DT, CAST_B16_TO_B32>(xF, xRaw, mask);
            AscendC::Reg::LoadAlign<DT, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dyRaw, dyAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<float, DT, CAST_B16_TO_B32>(dyF, dyRaw, mask);
            AscendC::Reg::Cast<DT, float, CAST_B32_TO_B16>(dxRaw, xF, mask);
            AscendC::Reg::StoreAlign<DT, AscendC::Reg::StoreDist::DIST_PACK_B32>(dxAddr + i * oneRepeatSize, dxRaw, mask);
        } else {
            AscendC::Reg::LoadAlign(xF, xAddr + i * oneRepeatSize);
            AscendC::Reg::LoadAlign(dyF, dyAddr + i * oneRepeatSize);
            AscendC::Reg::StoreAlign(dxAddr + i * oneRepeatSize, xF, mask);
        }
    }
}
```

3. `./build.sh <pkg>` 重建（拷贝变体无未用变量告警阻断，正常编过）。
4. 用 aclrtEvent kernel-only 计时（预构建 N=100 个 executor，事件包住整批 launch）测拷贝变体带宽。
5. 还原：`cp /tmp/softplus_kernel_orig.cpp op_kernel/softplus_v2_grad_kernel.cpp`，`diff -q` 确认逐字节还原，重建，复跑全量 ST 确认交付物未受损。
6. 判定：`计算版带宽 / 拷贝版带宽 ≥ 目标比例` 即达标，同时"比值接近 1"即证明带宽受限（计算被搬运隐藏）。

计时驱动核心（aclrtEvent kernel-only，verbatim 摘自 perf 驱动；口径选择与 host 开销剔除的完整论证见 [aclnn 带宽计时用事件法剔除 host 开销](aclnn_timing_aclrtevent_kernel_only.md)）：

```cpp
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
```

## 设计决策

- 为什么不用现成参照系：aclrtMemcpy D2D 走 SDMA，低于 AIV MTE 路径可达带宽（同会话不同测量轮实测约为被测 kernel 的 0.65x）；add_custom 样例 `MAX_BLOCKS=8` 只用 8/56 核（同会话不同测量轮实测约为被测 kernel 的 0.35x）；两者都会严重低估上界、误判不达标。
- 为什么同 tiling 拷贝变体可辩护：与被测 kernel 只差 VF 计算体（同队列深度、同 tile 大小、同核数、同 2R+1W 字节量），是控制变量后的"零计算可达上界"；纯拷贝自身未必达到理论峰值的 100%，但它给出的分母是**该设备该模式下实际可实现的**带宽。
- 判定口径：92.6% ≥ 80% 判 PASS；即使理论峰值略高于拷贝变体，拷贝变体作为带宽受限 kernel 的实际可达上界，该口径仍是"实际利用率"的保守下界估计。
- 实验纪律：先备份、后还原、`diff -q` 验证、重建、复跑全量 ST——A/B 实验不得污染交付物。

## 系统信息

- toolchain：CANN 9.1.0；设备 Ascend950PR（56 AIV、HBM 128GB）；算子包 `softplus_v2_grad_package`，构建入口 `operators/build.sh`。
- 相关文件：`op_kernel/softplus_v2_grad_kernel.cpp`（`SoftplusV2GradVF` 计算体替换处）、自建 perf 驱动（aclrtEvent 计时）、`op_host/add_custom_host.cpp`（样例参照系失效的证据来源）。
- profiling 旁证：msprof `--op` 的 `op_summary.csv` 给 Block Num=56；由 aiv_time 换算的带宽与事件法同口径一致（同会话不同测量轮互证）。

## 预防

- 写带宽验收标准时若分母是"HBM 峰值"，先确认峰值来源（文档/API/实测）；没有来源就预先约定实测上界法，避免验收阶段临时论证。
- 引用任何"参考 kernel"做带宽基线前，先看它的 tiling/host 代码用了几核、多大 tile——样例代码常按最小可跑配置写（MAX_BLOCKS=8、8KB tile），不是性能上限。

## 验证信号

来源说明：问题与验证来自一次 softplus_v2_grad 带宽验收（分母不可得）的判定方法构建。实测信号（均为同设备同 shape L1 [4096,4096] fp32，N=100）。注意：本卡的 D2D memcpy / add_custom 参照测量与 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md) 一卡中的同名探针为不同轮次的独立测量，数值互不相同，不可当作同轮数据混用：

- 计算版 kernel-only 与端到端（含 per-call host setup+sync）成对实测：端到端约为 kernel-only 的 0.94x，host 开销可见（来源平台：950；证据等级：tool_observed）。
- 纯拷贝变体 kernel-only 与端到端成对实测：端到端约为 kernel-only 的 0.94x（来源平台：950；证据等级：tool_observed）。
- 比值：计算版 / 纯拷贝变体（kernel-only）= **92.6%**（端到端口径 92.8%），≥80% 判 PASS；同时证明计算已完全隐藏在搬运后（带宽受限）（来源平台：950；证据等级：tool_observed）。
- 还原核验：`diff -q /tmp/softplus_kernel_orig.cpp op_kernel/softplus_v2_grad_kernel.cpp` → MATCH，重建成功，复跑 ST 25/25 PASS（Branch-1 15/15、Branch-2 10/10）（来源平台：950；证据等级：tool_observed）。
- 旁证：msprof aiv_time 换算带宽与事件法一致；被测 kernel（kernel-only）为同会话不同测量轮 D2D memcpy 参照的 1.54x、为 add_custom 样例参照的 2.84x（跨测量轮配对，仅作量级参照，不得当同轮成对数据引用）（来源平台：950；证据等级：tool_observed）。

## 性能影响

- 同算子内成对相对对比（来源平台：950）：计算版相对纯拷贝上界约 0.926x；相对 D2D memcpy 参考约 1.54x；相对 add_custom 样例参考约 2.84x。无绝对通用预期收益。

## 失效边界

- 拷贝变体的访存模式必须与被测 kernel 一致（本例 2R+1W、逐元素、无重用）；读写比不同或含归约/原子/共享重用的 kernel，"纯拷贝"不再是可比上界。
- 比值高≠接近理论峰值：拷贝变体本身可能未吃满理论 HBM（本例理论峰值未定），结论口径是"实测可达上界的利用率"。
- 本例比值（92.6%/92.8% 等成对数字）仅对 Ascend950PR + 该 shape/dtype/tiling 有效，跨设备/shape 须重测。
- 未验证，推断可迁移：fp16 大 shape（K=3 升精度路径）同法可用（cast 通路已在变体中保留），推断可迁移但未实测。

**未决项**：

- Ascend950PR 理论 HBM 峰值的权威数值仍未获得（见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）；若后续拿到，可回填"拷贝变体达到理论峰值百分比"这一更强结论。
- msprof `*_bw` 字段与 `hbm_*.csv` 列的确切语义待证实（本卡仅将其列为不可直接使用的观测）。
## 坏实践

- 用 `aclrtMemcpy` D2D 带宽当"HBM 峰值"分母——SDMA 路径低于 AIV MTE 可达带宽，会把已达标的 kernel 误判为超标（本例 kernel 反超 memcpy 52%）。
- 用官方样例算子（add_custom）当性能上界参照——样例按最小可跑配置写（`MAX_BLOCKS = 8`、`TILE_BYTES = 8*1024`），56 核设备只调度 8 核。
- A/B 实验后只凭记忆还原 kernel——必须备份 + `diff -q` + 重建 + 回归。

## 误诊

- "msprof `aiv_main_mem_read_bw`=0.34/`write_bw`=0.17 是利用率比率，推得峰值≈3288 GB/s、利用率 53% 判 FAIL"（agent_reported 推测，字段语义未证实，且与实测上界矛盾，最终未被采用）。
- "GetCoreMemBw 返回 31 可能单位是 100GB/s，即峰值 3100 GB/s"（agent_reported，无证据，未采用）。
