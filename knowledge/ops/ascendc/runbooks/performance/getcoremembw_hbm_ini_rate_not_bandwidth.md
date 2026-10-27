---
type: "Runbook"
title: 950PR HBM 峰值判定：GetCoreMemBw=31 非带宽
description: "Ascend950PR 无公开 HBM 峰值数字，PlatformAscendC 的 GetCoreMemBw(CoreMemType::HBM) 返回的是 platform_config ini 的相对权重 31（ddr_rate）而非 Bps；排除 memcpy/参考算子/msprof 假信号后，用同 tiling copy-only 探针（实测 >1.0x 于 1.64 TB/s 候选峰值）排除该假设、按 npu-smi 3200MT/s×8×1024-bit 推断峰值 ~3.28 TB/s。"
tags: [task.debugging, topic.memory, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T04:39:31Z'
updated_at: '2026-09-17T04:39:31Z'
platforms: ["950"]
---

# 950PR HBM 峰值判定：GetCoreMemBw=31 非带宽

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个逐元素自定义算子做带宽验收，验收标准是"L1 `[4096,4096]` fp32 端到端有效带宽利用率 ≥ 80% HBM 峰值"。实测 kernel 有效带宽（aclrtEvent 计时，20 iter）需要与峰值比对才能判定，而问题正在于：整条工具链里找不到权威的 HBM 峰值数字——

- `platform_ascendc::PlatformAscendC::GetCoreMemBw(CoreMemType::HBM, bw)` 返回 **31**，显然不是 Bps 量级的带宽；同一个平台对象上 `GetCoreNumAiv()`=56、`GetCoreMemSize(CoreMemType::UB)`=253952（248KB，符合 950PR 规格）均正常，唯独 HBM 档带宽查询无效；
- `aclrtGetDeviceCapability` 枚举里没有任何内存带宽属性；
- `npu-smi info -t board / -t memory` 只给 HBM 容量 131072 MB、时钟 3200 MHz、厂商 ID，没有总线宽度/峰值；板上 `HBM Bandwidth Usage Rate(%)` 采样在持续负载下恒为 0%；
- msprof 系统级 HBM 采样 CSV 的数值（Read 843 MB/s / Write 156634 MB/s）量纲混乱，无法直接当峰值。

峰值取 1.64 TB/s 还是 3.28 TB/s，直接决定利用率是 99.5%（PASS）还是 49.7%（FAIL），判定必须落在客观证据上。

## 适用场景

- 工程形态：aclnn 自定义算子性能验收 / 任何需要"HBM 峰值"做分母的带宽利用率判定。
- API 路径：`platform_ascendc`（PlatformAscendCManager / GetCoreMemBw / GetCoreMemSize）、`npu-smi`、`msprof --sys-hardware-mem`、`aclrtMemcpy`、自建 aclnn 事件计时 harness。
- 验证过的平台与 CANN 版本：Ascend950PR（A310-50-C00MM304A1，NPU Name 9579），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（Ascend950PR 平台级判别方法；其他平台方法可迁移、数值须重测）。
- 路线优先级：recommended_fastpath（判定带宽型算子是否达标的标准流程）。
- 明确不适用 / 未验证场景：其他 SoC（ddr_rate/ini 字段、stack 配置不同）；需要理论精确峰值（而非工程判定）的场合——本方法给出的是与全部观测一致的推断值。

## 判别方法

按"先排除假信号、再用 copy-only 探针定界、最后理论值对账"三步：

**第一步：识别并排除假峰值信号。**

1. `GetCoreMemBw(HBM)` 返回 31 —— 与 `Ascend950PR_9579.ini` 的 `[AICoreMemoryRates] ddr_rate=31` 完全一致（同 ini 中 `l2_rate=100` 为参照、`[VectorCoreMemoryRates] ddr_rate=16`），它是 tiling 成本模型的**相对权重**，不是 Bps。旁证：`GetCoreMemSize` 逐项查询输出 `L2 size=134217728 bw=100`、`HBM size=137438953472 bw=31`、其余层 bw=0——"bw" 列就是 ini 的 rate 字段直读。`CoreMemType::HBM` 枚举值（=6）已对照 `platform_ascendc.h` 核对，排除传错档位；同对象 `GetCoreNumAiv`/`GetCoreMemSize(UB)` 数值正常且有旁证，排除 PlatformInfo 未初始化/取空。
2. `PlatformAscendCManager` 用显式 soc version 字符串（`"ascend950"`、`"Ascend950PR"`）初始化会 FAILED；只有无参 `GetInstance()` 成功（返回 SocVersion=4、aiv=56、ub=253952）。
3. `aclrtMemcpy` D2D（192 MB，1 读 + 1 写）：实测明显低于被测 kernel（成对约 0.77x）——DMA 引擎路径不是 AI 核可达峰值的代理，只能当 HBM 的下界。
4. 同仓模板算子 AddCustom（2 读 + 1 写、一条 `AscendC::Add`）参考实测仅为被测 kernel 的约 0.37x——它的多核 tiling 没吃满核，未优化参考算子不能当峰值。
5. `msprof --sys-hardware-mem=on --sys-hardware-mem-freq=100` 产出的 `hbm_*.csv`（`Device_id,Metric,Read(MB/s),Write(MB/s)`，Average 行 Read=843.202 / Write=156634.449）：100 Hz 采样对仅百微秒量级的 kernel 太粗，且读/写量纲明显不自洽（1:186），不可用。
6. msprof op_summary 的 `aiv_main_mem_read_bw=0.316` / `aiv_main_mem_write_bw=0.157`（读:写 ≈ 2:1，与 2R+1W 流量比一致）：按利用率比例解释可反推峰值 ~3.6 TB/s，量级与最终推断一致，但确切口径未定论，只作旁证。

**第二步：copy-only 探针定界（本方法的判别核心）。**

把被测 kernel 的 VF 计算链整体替换成单条 `Adds`（保留完全相同的 CopyIn/CopyOut/tiling/多核路径），重编译后跑同一 harness：

```cpp
        // TEMP COPY-ONLY: dxF = dyF (bandwidth probe, revert after measurement)
        AscendC::Reg::Adds(dxF, dyF, 0.0f, mask);
```

实测 copy-only 探针带宽约为推断峰值的 ~52%（best，深度 1 队列；峰值口径见第三步）。两个结论：
- 完整计算链与 copy-only 探针只差 ~5%（copy-only 仅快 ~5%）→ kernel 是带宽/流水受限而非计算受限；
- copy-only 探针实测 **>1.0x 于 1.6384 TB/s 候选峰值**（"4×32GB stack、4096-bit 总线"假设）→ 该候选峰值被实测排除；copy-only 本身成为该访问模式经 AI 核的可达下界。

**第三步：理论值对账。**

HBM3 每堆栈 1024-bit（128 B）接口；npu-smi 给 3200 MT/s、128 GB 容量：
- 8×16 GB 堆栈：总线 1024 B → 1024 × 3200e6 = **3.2768 TB/s**；
- 4×32 GB 堆栈：总线 512 B → 1.6384 TB/s（已被第二步排除）。

取 3.28 TB/s 为峰值：kernel 利用率 ≈ 49.7% → 验收 FAIL；DMA 约为峰值的 38%（DMA 合理）；AddCustom 自身未优化（远低于拷贝探针）。全部观测与该值自洽。

## 完整代码

平台信息查询程序（`query_bw.cpp`，verbatim；展示 `GetCoreMemBw` 的取值方式与"返回 31"的复现入口）：

```cpp
// Query HBM peak bandwidth + AIV core count via platform_ascendc API.
#include <cstdint>
#include <cstdio>
#include "tiling/platform/platform_ascendc.h"

int main() {
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        fprintf(stderr, "FATAL: PlatformAscendCManager init failed\n");
        return 1;
    }
    uint32_t aiv = plat->GetCoreNumAiv();
    uint64_t hbmBw = 0;
    plat->GetCoreMemBw(platform_ascendc::CoreMemType::HBM, hbmBw);
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto soc = plat->GetSocVersion();
    fprintf(stdout, "SocVersion=%d aiv_cores=%u hbm_bw_Bps=%llu ub_size=%llu\n",
            static_cast<int>(soc), aiv,
            static_cast<unsigned long long>(hbmBw),
            static_cast<unsigned long long>(ubSize));
    return 0;
}
```

独立编译命令（仅链 `libtiling_api.a` 会报 `undefined reference to 'CheckLogLevel'`（libunified_dlog.so DSO missing），必须补全依赖，verbatim）：

```bash
CANN=/home/<user>/Ascend/cann-9.1.0; g++ -std=c++17 -O2 \
  -I${CANN}/x86_64-linux/asc/include \
  -I${CANN}/x86_64-linux/include \
  -I${CANN}/include \
  /tmp/perf_query/query_bw.cpp \
  ${CANN}/x86_64-linux/lib64/libtiling_api.a \
  -L${CANN}/x86_64-linux/lib64 -lplatform -lge_common -lgraph -lexe_graph -lregister -lunified_dlog -lc_sec -ldl -lpthread \
  -Wl,-rpath,${CANN}/x86_64-linux/lib64 \
  -o /tmp/perf_query/query_bw && /tmp/perf_query/query_bw
# 输出：SocVersion=4 aiv_cores=56 hbm_bw_Bps=31 ub_size=253952
```

真实算子 tiling 上下文核验（op_host 的 TilingFunc 平台查询块临时插桩，带 null/退化守卫，构建需文件头补 `#include <cstdio>`，跑完 ST 后还原；证明 tiling 真实路径拿到的同样是 31）：

```cpp
    // —— 平台查询：availableCoreNum / ubSize（带 null/退化守卫）——
    int32_t availableCoreNum = 1;
    int64_t ubSize = 196608;  // DAV_3510 级默认 UB 预算（运行时实测覆盖）
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC plat(platformInfo);
        uint32_t aiv = plat.GetCoreNumAiv();
        if (aiv > 0) {
            availableCoreNum = static_cast<int32_t>(aiv);
        }
        uint64_t ub = 0;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        if (ub > 0) {
            ubSize = static_cast<int64_t>(ub);
        }
        // TEMP DIAG: log HBM peak bandwidth + UB + coreNum (remove after fix)
        uint64_t hbmBw = 0;
        plat.GetCoreMemBw(platform_ascendc::CoreMemType::HBM, hbmBw);
        fprintf(stderr, "[DIAG] aiv=%u ubSize=%ld hbmBw=%lu B/s (%.1f GB/s)\n",
                aiv, (long)ubSize, (unsigned long)hbmBw, hbmBw / 1e9);
    }
```

`aclrtMemcpy` D2D 峰值代理测量核心（事件计时，warmup 5 + 20 iter；D2D 拷贝搬 2×buffer 字节）：

```cpp
    const int ITERS = 20;
    std::vector<double> times;
    for (int i=0;i<ITERS;i++) {
        aclrtEvent start, stop;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&stop));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_ACL(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeEvent(stop));
        float ms=0;
        CHECK_ACL(aclrtEventElapsedTime(&ms, start, stop));
        times.push_back((double)ms);
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(stop));
    }
    std::sort(times.begin(), times.end());
    double best=times[0], median=times[ITERS/2];
    double totalBytes = 2.0 * bytes;   // D2D copy moves 2*bytes (1 read + 1 write)
    double bwBest = totalBytes / (best*1e-3) / 1e9;
```

## 设计决策

- 为什么用 copy-only 探针而不是 DMA/参考算子当峰值：DMA 引擎与 AI 核访存路径不同（实测 DMA 约为被测 kernel 的 0.77x，反向证明）；参考算子的 tiling 质量不可控（AddCustom 约为被测 kernel 的 0.37x）。copy-only 保留被测 kernel 的全部搬运路径（同 tiling、同多核、同队列），替换的只有计算，因此它给出"该访问模式下该 kernel 结构的可达上界"，且能一步排除低于它的候选峰值。
- 为什么探针用 `Adds(dxF, dyF, 0.0f, mask)` 而不是删掉计算：寄存器 API 没有裸搬移原语，`dst = src + 0` 是等价单op拷贝，改动最小且不触碰 load/store 结构。
- 为什么最终采信 3.28 TB/s 而非 3.6 TB/s（op_summary 反推）：3.2768 TB/s 有封闭的理论算式（8×1024-bit×3200 MT/s）且与 npu-smi 时钟/容量对账；op_summary 的 bw 口径未定论只作旁证。判定带宽型算子验收时，宁可采信有算式、多源一致的值并在报告里写明推断性质。
- 探测路线取舍：优先尝试平台 API（零成本）→ 返回无效后放弃 API 路线，改走实测法；曾计划扩展为 L1/L2/UB/HBM 四路查询定位单位语义，因插桩编辑事故未执行，且即使拿到也解决不了 HBM 档的无效问题，遂不再投入。
- 当前方案路线档位：recommended_fastpath（判定流程本身）；结论数值仅适用 950PR。

## 系统信息

- 硬件/软件：Ascend950PR（npu-smi：Product A310-50-C00MM304A1、NPU Name 9579、HBM 131072 MB @3200 MHz、HBM Manufacturer ID 0x5A、Aicore Count 28、Aicore Freq 1650 MHz；平台查询 aiv=56、ub=253952、L2=128 MiB、HBM=137438953472 B），CANN 9.1.0，平台头 `x86_64-linux/asc/include/tiling/platform/platform_ascendc.h`。
- 关键配置文件：`x86_64-linux/data/platform_config/Ascend950PR_9579.ini`（`ddr_rate=31`、`l2_rate=100`、`ub_to_l2_rate=128`、`vector_core_cnt=56`、`ai_core_cnt=28`、`io_die_num=2`）。
- 工具：`npu-smi info -t board/-t memory/-t common`、`msprof --aic-metrics=Memory --sys-hardware-mem=on --sys-hardware-mem-freq=100`（产物 `mindstudio_profiler_output/hbm_*.csv`、`op_summary_*.csv`、sqlite `metric_summary.db`）。
- 自建 harness：aclrtEvent 计时（注意本版本 `acl/acl.h` 只暴露单参 `aclrtCreateEvent(&e)` / `aclrtSynchronizeEvent(e)`；带 `ACL_EVENT_TIME_LINE` 标志或 timeout 的重载不在该头里，误用会编译报 too many arguments）。

## 预防

- 需要 HBM 峰值做利用率分母时，不要把 `GetCoreMemBw` 当唯一来源；先做一次 30 秒的量级探针验证返回值，再决定是否信任。
- tiling 里做平台自适应时，`GetCoreNumAiv`/`GetCoreMemSize` 可用（本例数值正常且有旁证），但同样建议带退化守卫（默认值兜底），本卡完整代码中 tiling 插桩的 `if (aiv > 0)`/`if (ub > 0)` 写法即为此。

## 验证信号

来源说明：以下信号来自该算子带宽验收会话的多探针实测（同一设备同一夜），按判别链条列出：

- `GetCoreMemBw(HBM)` 独立程序与 op_host 真实 tiling 上下文均返回 31（SocVersion=4、aiv=56、ub=253952）；逐层 `GetCoreMemSize/GetCoreMemBw` 输出 `L2 bw=100`、`HBM size=137438953472 bw=31`、其余层 bw=0（来源平台：950；证据等级：tool_observed）。
- op_host tiling DIAG 输出 `[DIAG] aiv=56 ubSize=253952 hbmBw=31 B/s (0.0 GB/s)`；`CoreMemType::HBM` 枚举值（=6）已对照 `platform_ascendc.h` 核对排除传错档位；`GetCoreNumAiv`=56 与 msprof op_summary 的 Block Num=56 一致、`GetCoreMemSize(UB)`=253952 符合 248KB 规格（来源平台：950；证据等级：tool_observed）。
- `Ascend950PR_9579.ini` 含 `[AICoreMemoryRates] ddr_rate=31 / l2_rate=100`、`[VectorCoreMemoryRates] ddr_rate=16`，与 API 返回值对上（来源平台：950；证据等级：tool_observed）。
- 显式 soc version 字符串初始化 PlatformAscendCManager：`[ascend950] init FAILED`、`[Ascend950PR] init FAILED`；无参 `GetInstance()` 成功（来源平台：950；证据等级：tool_observed）。
- `aclrtMemcpy` D2D 192 MB：best/median 均明显低于被测 kernel（成对约 0.76~0.77x，约为推断 3.28 TB/s 峰值的 38%）（来源平台：950；证据等级：tool_observed）。
- AddCustom 参考算子同 shape：best/median 均仅为被测 kernel 的约 0.37x（未优化参照，不能当峰值）（来源平台：950；证据等级：tool_observed）。
- msprof 系统级 hbm CSV（100 Hz）：Average Read=843.202 MB/s、Write=156634.449 MB/s，逐采样行同量级；对百微秒量级 kernel 不可用（来源平台：950；证据等级：tool_observed）。
- msprof op_summary（被测 kernel 行）：`aiv_main_mem_read_bw=0.316`、`aiv_main_mem_write_bw=0.157`、`aiv_ub_read_bw=12.741 GB/s`、Task Duration 为百微秒量级、Block Num 56（来源平台：950；证据等级：tool_observed）。
- copy-only 探针 kernel 与完整计算链同 harness 成对对比：copy-only 仅快 ~5%（best/median 口径一致），且 copy-only 探针 >1.0x 于 1.64 TB/s 候选峰值（排除该候选）（来源平台：950；证据等级：tool_observed）。
- 独立实测 kernel-only 带宽（aclrtEvent）比 `GetCoreMemBw` 返回值 31 高约 7 个数量级（10^7 倍量级），返回值不可能是任何合理单位的 HBM 峰值（来源平台：950；证据等级：tool_observed）。
- npu-smi：HBM Capacity 131072 MB、HBM Clock Speed 3200 MHz、Aicore Count 28；`HBM Bandwidth Usage Rate(%)` 在持续负载下采样恒 0%，platform_config ini 无带宽字段（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 峰值结论 ~3.28 TB/s 只对 Ascend950PR（9579 配置）成立；同系列不同内存配置（如 950z）或其他 SoC 须重走第三步对账，方法可迁移、数值不可。
- copy-only 探针给的是"该 kernel 搬运结构的可达带宽"，深度 1 队列下它本身就被流水串行压低（约为推断峰值的 ~52%）——它用于**排除**低于它的候选峰值与证明计算不占瓶颈，不能当作峰值本身；要逼近真峰值须叠加双缓冲（见 [TQue 深度 1 带宽减半：fp32 双缓冲修复](tque_depth1_bandwidth_double_buffer.md)）。
- `GetCoreMemBw` 的返回语义（ini rate 直读）在 CANN 9.1.0 验证；后续版本若语义变化须重验（未验证，推断可迁移）。
- msprof op_summary 的 aiv_*_bw 口径未定论，本卡只将其作旁证，不单独用于判定。

**未决项**：

- Ascend950PR HBM 峰值 ~3.28 TB/s 仍是多源推断值，未见官方数字——优先项：找到权威规格（总线宽度/堆栈数）或用双缓冲后逼近峰值的实测封顶它。
- msprof `aiv_main_mem_read_bw/write_bw` 的确切口径（比例 vs 绝对值、统计窗口）未定论，按比例反推峰值 ~3.6 TB/s 与 3.28 TB/s 有 ~10% 偏差。
- `GetCoreMemBw` 与 ini rate 字段的对应关系为多源对账实证（31/ddr_rate、100/l2_rate 逐项吻合），未读到 CANN 实现源码逐行确认——源码级定位仍为可选项。
- 双缓冲修复后的 copy-only 上界会抬到多少（即该平台 AI 核 2R+1W 模式的实际可达峰值）未测。
## 坏实践

- 拿 `GetCoreMemBw(HBM)` 的返回值当 Bps 参与利用率计算（会把实测带宽算成百分之几千的荒谬利用率，或误导排查方向）。
- 拿 `aclrtMemcpy` D2D 或未优化的模板算子当 HBM 峰值代理——两者都低于被测 kernel，算出的"利用率"会超过 100%（本例按这类参照分母即得 ~144%），应立即意识到参照系错误。
- 直接采信 msprof 系统级 HBM 采样 CSV 的 Read/Write 数值而不核对采样窗口与量纲。
- 在报告里只写"利用率 50%"而不写峰值怎么来的——峰值是推断值时必须写明推断链（本卡第三步），否则结论不可复核。

## 误诊

- 被测算子的开发者自评时直接假设一个无依据的低峰值分母（远低于本卡推断的 3.28 TB/s），按该分母得出 ≈128% 利用率并通过自检——无依据的峰值假设让"利用率 > 100%"这一明显矛盾信号被忽略（该自评为 agent 转述，证据等级：agent_reported；推翻它的探针实测均为 tool_observed）。
- 排查早期把 `hbm_bw_Bps=31` 打印字段的变量名当事实，一度困惑"31 是不是 GB/s 单位"；对照 ini 后确认是相对权重（推断，已被 ini 对账证实）。
