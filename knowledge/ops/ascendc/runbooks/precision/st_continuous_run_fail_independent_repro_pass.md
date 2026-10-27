---
type: "Runbook"
title: ST 连续运行失败而独立 repro 全对的二分
description: ST 全量运行中个别 FP32 极端参数用例失败（大 lr / beta=0 / eps=0，失败元素差≈lr），把 Sqrt/Div 换高精度模式逐位无效；用同 RNG、同参数、同 shape 的独立 aclnn 程序复现则精度全对——kernel 单调用正确，失败只在 ST 框架连续运行上下文出现，转查框架侧；框架侧根因未闭合。复现增量：跨算子复现"scalar 参数被无视、actual 恒似默认分支/元素总数/垃圾值"签名与批次二分（12/批 33 败、15/批 8 败、1/批 74/74），每用例子进程隔离（Python subprocess / C++ fork）为 CANN 9.1.0 上唯一被全量验证的执行方式；fork+execl 每用例全新进程分钟级、全量回归 banned。复现增量（CANN 9.2.0）：同 shape 不同属性（布局/causal/dtype）相邻用例的进程内漂移实测（MERE 2.53e-01 vs 子进程 3.52e-07，四个数量级）与前序 NaN/设备错误用例污染两种形态；cooldown 100ms 实测无效；cgroup 32GB exit 137 判别；逐用例隔离编排（串行 bash + 并行 ProcessPoolExecutor）实测全量恢复。复现增量：RoPE 类算子 600 条 C++/PyTorch 双 harness 复现——逆序执行失败集翻转钉死位置依赖、大幅错值（MERE 10+）形态、V↔S HardEvent 同步假设证伪；前/后 200 条分段仍 ~25% 失败再证分批不充分。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-17T02:14:14Z'
platforms: ["950"]
---

# ST 连续运行失败而独立 repro 全对的二分

## 现象

在 Ascend950PR（CANN 9.1.0）上，自研 aclnn 算子（AdamW 类）完成多 TilingKey 策略整合（FP32/FP16/BF16 × 单/双缓冲共 6 档）后跑 ST Real 全量 92 用例：FP16/BF16 全部通过，但 5 个 FP32 极端参数用例失败，失败样例：

```
[Real] <case> - dtype=float32, shape=(100, 100), size=10000
  [FAIL] MERE=8.46e-01 (threshold=5.00e-03), MARE=1.01e+00 (threshold=5.00e-02)
  不匹配 [0]: 期望=-0.92427733, 实际=0.07587364, rel_err=1.00e+00
```

5 个失败用例的共性是极端标量参数：beta2=0（MARE=1.95）、lr=0.1（MARE=0.295）、lr=1.0（MARE=1.99）、lr=1+beta=0+eps=0（MARE=1.01）、lr=1+weight_decay=1+maximize（MARE=1.87）。失败元素的实际值与期望值之差约为 lr（上例差≈1.0），像是"更新项整体丢失/多算一步"而非舍入误差。当时代码状态：kernel FP32 路径使用 AscendC::Muls/Axpy/Sqrt/Adds/Div/Add 组合，tiling 侧预计算 invBiasCorr1/2。

## 适用场景

- 工程形态：aclnn 自定义算子 ST Real 全量回归（一个进程顺序跑几十至上百个用例）+ 独立 aclnn 诊断程序对照。
- API 路径：aclnn 两段式接口（GetWorkspaceSize + 执行）、`aclrtSynchronizeStream`、host 侧独立 repro（g++ 直连 aclnn，链接 `libascendcl -lcust_opapi`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only（判别方法论：ST 框架失败 vs kernel 失败的二分；不宣称 kernel/框架任一侧的具体缺陷机制）。
- 路线优先级：recommended_fastpath（出现"全量跑挂、单独跑对"形态时的第一定位步骤）。
- 明确不适用/未验证场景：独立 repro 也复现失败时（kernel 侧问题）本判别不适用；本卡的"框架侧归因"为待验证假设，最终根因在本次会话内未闭合（见失效边界与遗留问题）。

## 判别方法

"ST 全量个别用例失败"先做独立 repro 二分，再谈改 kernel：

1. **写独立 aclnn 诊断程序**（与 ST 完全同参数）：小规模已知值先确认 kernel 数值正确——8 元素、手写 var/grad/m/v 已知值、double 精度手算 golden、直接调 `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`，逐元素打印 golden vs NPU。编译命令（可复用）：

```bash
g++ -o debug_precision debug_precision.cpp \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/opp/vendors/<vendor>_custom/op_api/include \
    -L$ASCEND_HOME_PATH/lib64 -lascendcl -lopapi \
    -L$ASCEND_HOME_PATH/opp/vendors/<vendor>_custom/op_api/lib -lcust_opapi
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/opp/vendors/<vendor>_custom/op_api/lib:${LD_LIBRARY_PATH}
./debug_precision
```

2. **升级 repro 到与 ST 完全同构**：把 ST 的 RNG 类与 golden 函数原样拷进 repro（同 seed、同 shape、同参数），消除"数据不同"的辩护空间。本例 v2 程序骨架（节选自实测程序，SimpleRNG 与 golden 公式与 ST 逐字一致）：

```cpp
// 与 ST 测试完全相同的 RNG
class SimpleRNG {
    uint64_t state_;
public:
    explicit SimpleRNG(uint64_t seed) : state_(seed) {}
    float NextFloat(float lo, float hi) {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t x = static_cast<uint32_t>(state_ >> 33);
        float t = static_cast<float>(x) / static_cast<float>(0x80000000u);
        return lo + t * (hi - lo);
    }
};

void GenerateTestData(uint64_t seed, size_t size,
                      std::vector<float>& var,
                      std::vector<float>& grad,
                      std::vector<float>& m,
                      std::vector<float>& v) {
    var.resize(size); grad.resize(size); m.resize(size); v.resize(size);
    SimpleRNG rng(seed);
    for (size_t i = 0; i < size; ++i) {
        var[i]  = rng.NextFloat(-1.0f, 1.0f);
        grad[i] = rng.NextFloat(-0.5f, 0.5f);
        m[i]    = rng.NextFloat(0.0f, 0.1f);
        v[i]    = rng.NextFloat(0.0001f, 0.01f);
    }
}
```

main 中按失败用例取参（本例：seed=68、size=10000、shape={100,100}、lr=1.0、beta1=0、beta2=0、weight_decay=0.01、epsilon=0、step=1、maximize=false），先打印前几个元素的手算 golden 与输入数据，再走 aclInit → 建张量 → 两段式调用 → D2H → 与 double golden 逐元素比对并统计 MERE/MARE。

3. **读数规则**：
   - repro 全对（本例 MERE=2.8013e-08、MARE=8.6524e-08、失败 0/10000）⇒ kernel 对该参数组合正确 ⇒ 失败来自 ST 框架连续运行上下文（用例间状态、executor/tensor 生命周期、输入组包差异），转查框架侧；
   - repro 复现失败 ⇒ kernel 侧问题，按数值路径继续二分。

## 根因

本卡为诊断规则卡：实测钉死的事实是"kernel + aclnn 单调用对该参数组合完全正确，失败只在 ST 框架单进程连续运行 90+ 用例的场景出现"；框架侧具体根因（当期排查方向为 `aclOpExecutor` 未显式释放导致的状态累积/串扰）在本次会话内未闭合，不作定论。

## 设计决策

- 为什么先写独立 repro 而不是继续改 kernel 精度配置：Sqrt/Div 高精度模式替换（见误诊）已实测零效果，说明"继续在 kernel 数值上用力"是在没有证据的方向上消耗；独立 repro 一次实验就能把问题空间砍半。
- 为什么 repro v2 要逐字拷贝 ST 的 RNG 与 golden：数据分布或 golden 公式的任何差异都会让"全对"失去说服力。
- 当前方案路线档位：recommended_fastpath（定位路径），对"框架侧根因"本身不定档。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；独立 repro 用系统 g++ 直连 aclnn（见判别方法中的完整编译命令）。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（ST 框架：SimpleRNG/GenerateTestData/ComputeGolden/RunTest）、`tests/st/testcases/`（用例参数 CSV）、独立诊断程序 `debug_precision.cpp` / `debug_precision_v2.cpp`（会话内新建于 ST 目录）。

## 预防

- ST 框架出现"个别用例失败"时，把独立 repro（同 RNG、同参数、同 shape）作为标准第一步骤，避免在 kernel 侧盲试精度配置。
- ST 框架设计上显式管理每个用例的 executor/tensor/workspace 生命周期，不留隐式跨用例状态。

## 验证信号

以下信号来自该算子联调验证会话（均为工具实测）：

- 触发：ST Real 92 用例中 5 个 FP32 极端参数用例失败（MARE 0.295~2.0），FP16/BF16 全过；失败样例期望=-0.92427733、实际=0.07587364（差≈lr=1.0）（来源平台：950；证据等级：tool_observed）。
- 排除实验：kernel 的 Sqrt/Div 由默认 INTRINSIC 换 `PRECISION_0ULP_FTZ_FALSE`（SqrtConfig/DivConfig 模板参数）重编重装后，5 个失败用例的 MERE/MARE 与换前逐位相同（来源平台：950；证据等级：tool_observed）。
- 独立 repro v1（8 元素已知值、同参数）：NPU vs golden 逐元素 rel_err 1.09e-08~5.60e-08（来源平台：950；证据等级：tool_observed）。
- 独立 repro v2（同 RNG seed=68、10000 元素、同参数）：MERE=2.8013e-08、MARE=8.6524e-08、失败元素 0/10000（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只证明"kernel 单调用正确 + 失败与 ST 连续运行上下文绑定"这一实测事实；"aclOpExecutor 生命周期/状态泄漏"是会话末尾的排查方向而非结论——最终根因未闭合，使用本卡时不得把该假设当定论引用。
- 独立 repro 结论只覆盖被复现的参数组合与 shape；其他失败形态需各自 repro。

**未决项**：

- 【优先】复现并追到机制层根因：ST 单进程连续运行下 5 个 FP32 用例失败、独立 repro 全对——会话末尾正沿 `aclOpExecutor` 未显式释放/状态累积方向排查（aclnn 框架未见显式 destroy 接口），后续会话接续；接续者应以"ST 框架最小化复现（两用例序列）+ executor 生命周期对照"为切入点。
- 候选机制核对清单：该算子带 per-call 标量属性（lr/beta 等）且失败元素差≈lr，与"同 shape 不同属性值时 TilingData/L2 缓存复用陈旧属性"的已知机制形态吻合——排查时应先按 [标量属性入 TilingData 致同 shape 跨调用陈旧卡](tilingdata_shape.md) 与 [aclnn TilingData 缓存致 runtime attr 陈旧卡](aclnn_tilingdata_runtime_attr.md) 的判别方法（tiling 函数日志两批只出现一次 / 子进程隔离对照）排除该家族，再继续 executor 生命周期方向。
- 5 个失败用例是否共享同一框架侧触发条件（极端参数是否只是让误差显形）未验证。
## 坏实践

- 见"全量跑挂"就一头扎进 kernel 数值精度调整（换高精度 Sqrt/Div、改累加顺序），不做独立 repro 二分——本例该路线实测零收益。
- repro 与 ST 的 RNG/golden/参数有任何一处不一致还拿结果下结论。

## 误诊

- 假设"arch35 上 Sqrt/Div 的 INTRINSIC 模式精度不足"：查阅 `kernel_reg_compute_utils.h` 确认 arch35 的 Sqrt/Div 支持 `SqrtConfig`/`DivConfig` 模板参数（`SqrtAlgo::PRECISION_0ULP_FTZ_FALSE`、`DivAlgo::PRECISION_0ULP_FTZ_FALSE` 等；Sqrt 在 3510 上使用保留 UB，`--cce-disable-asc-reserved-ubuf` 时禁用），实施替换后 5 个失败用例误差逐位不变——假设被实测排除。该 API 机制事实（arch35 高精度模式的存在与用法）本身可复用，但它不是本例根因。
- 顺带排除了"ST 用例乱序执行/前序大 shape 用例影响"等框架侧猜想——这些在会话内同样未定论，归入遗留问题。

## 后续增量（2026-09-17，并行策展批：scalar 参数被无视签名、每用例子进程隔离配方与隔离代价边界）

独立轨迹（BenchNoHarnessForeachNorm，TensorList 逐张量 p 范数，Ascend950PR + CANN 9.1.0，74 条 L0+L1）把"连续运行失败、独立复现全对"的判别推到**可交付的执行方式**，并新增一个更强的污染签名：

**1. 新签名（scalar 类参数被无视）**：同一 Python 进程顺序执行全部用例时，scalar=2.0 的用例基本通过，**scalar≠2.0 的用例全部失败且 actual 恒在 30~38 附近**——恰是 1024 个 N(0,1) 元素的 L2 范数量级（sqrt(1024)=32）：L1_005（p=1，期望 777.68）actual=30.64、L1_008（p=+inf，期望 3.38）actual=32.84、L1_009（p=-inf，期望 0.0014）actual=32.03、L1_004（p=0 非零计数，期望 512）actual=23.02——像 NPU 无视 scalar 参数、一律按 p=2 计算。部分用例返回**张量元素总数**（L1_050 actual=[1048576,262144,65536,16384]）、近零或 1e18 级垃圾值（--no-fork L0 套件 actual=4.58e18，rel_err=4.47e+15）。该"结果退化为默认分支"的形态与本卡遗留问题中"同 shape 不同属性值时 TilingData/L2 缓存复用陈旧属性"的候选机制家族吻合（scalar 被无视 = 属性未生效），排查时应先按该家族判别方法排除。

**2. 独立进程对照与批次二分（tool_observed）**：同批失败用例逐条放到独立进程重跑（`--case L1_004` 等）**全部正确**；批次加大同样不安全——每批 12 条子进程 33/74 失败、15 条一批 8 条失败（逐条重跑全对）；**BATCH_SIZE=1（每用例一进程）后 74/74 稳定通过**（修 golden 后含确定性 3/3、全局最大 MERE=6.43e-07）。即污染在批次内同样发生，只有进程级隔离是干净的。

**3. 每用例子进程隔离配方**（Python 侧等价于 C++ ST 的 fork 隔离；父进程只做调度与结果汇总，子进程带 `--case <id> --skip-golden` 自跑自比，verbatim 核心段）：

```python
        # 全量模式: 每个用例在独立子进程中执行
        # （CANN 9.1.0 已知问题: ACLNN 执行后堆被污染，acl 资源泄漏累积导致
        #   NPU 状态损坏。子进程隔离确保每个用例获得干净的 ACL 状态。）
        test_script = os.path.realpath(__file__)
        for case_idx, tc in enumerate(test_cases):
            cmd = [sys.executable, test_script,
                   "--lib", lib_path, "--case", tc.id, "--skip-golden"]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                output = proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                ...  # 计 FAIL 并 continue
            # 从输出提取 "结果: PASS/FAIL" 或 "确定性验证结果: PASS/FAIL" 汇总
```

要点：`BATCH_SIZE=1` 是唯一被全量验证安全的批次；子进程判 FAIL 且输出为空时**先重跑一次再定性**（偶发子进程崩溃：一次全量中 L1_064 子进程输出为空判 FAIL，单独重跑与后续全量重跑均 PASS）。

**4. 隔离代价边界（fork+execl 全新进程路线 banned）**：C++ 侧为彻底规避 fork+aclInit 累积（约 35 个用例后 ACL 驱动拒绝分配）改用每用例 **fork+execl** 全新进程（子进程 `execl` 自身 + `--case <ID>` 单用例模式，`readlink("/proc/self/exe")` 取路径），单用例需完整动态库加载 + `aclInit/aclFinalize`——**900 秒超时仅完成 L0_001 一个用例**，74 用例全量回归不可用（banned）；单用例诊断场景（失败用例隔离复跑）实测有效且结果正确。Python 侧每用例子进程固定开销约 25s（torch import 占大头）对比进程内连续执行单条约秒级（来源平台：950）——隔离以一个固定启动常数为代价换取结果可信，74 条全量一轮约半小时量级，须后台 + 轮询执行。

**5. 验证信号**：

- 同进程全量：scalar≠2.0 用例全部失败（L1_005 golden=777.68 vs actual=30.64、L1_009 rel_err=2.26e+04、L1_050 actual=各张量元素总数），33+ 条失败；--no-fork L0 套件 actual=4.58e18 级垃圾值（来源平台：950；证据等级：tool_observed）。
- 同批失败用例独立进程逐条重跑全对（L1_004 actual=512=golden，L1_005~L1_011 全 PASS）（来源平台：950；证据等级：tool_observed）。
- 批次二分：12 条/批 33/74 失败、15 条/批 8 条失败；BATCH_SIZE=1 后 74/74、确定性 3/3、全局最大 MERE=6.43e-07（来源平台：950；证据等级：tool_observed）。
- fork+execl 代价：`timeout 900` 运行全量套件仅完成 L0_001（PASS 后无输出）被超时杀；对照的单用例新进程隔离探针结果正确（actual=golden=808, PASS）（来源平台：950；证据等级：tool_observed）。
- 偶发子进程崩溃：一次全量 L1_064 子进程空输出判 FAIL，重跑与后续全量均 PASS（来源平台：950；证据等级：tool_observed）。

**6. 坏实践补充**：在同一进程里循环调用 aclnn 算子并直接采信后续用例的结果——第一批调用之后的结果不可信，会把工具链污染误判成算子精度缺陷；失败后先改算子/适配层代码再重跑全量（应先做独立进程对照）；子进程空输出直接判 FAIL 入库（先重跑一次排除偶发崩溃）；用更大的超时阈值硬扛每用例全新进程方案（总耗时随用例数线性膨胀，不是工程解）。

**7. 遗留问题更新**：本增量把"框架侧根因未闭合"的执行面收口（进程隔离为 CANN 9.1.0 上唯一被全量验证的执行方式），机制层根因（进程内第 N 次 aclnn 调用后状态如何损坏、为何 scalar 类参数最先失效）仍未定位——候选方向仍按本卡遗留问题的 TilingData/L2 缓存陈旧属性家族优先排除，再用最小双用例程序 + 逐次结果比对定位首个污染点；进程池/分 dtype 复用子进程的折中方案未尝试。



## 增量（2026-09-17，并行策展批：两类污染形态实测、cooldown 无效反证、cgroup 137 判别与隔离编排，CANN 9.2.0）

两条 9.2.0 轨迹（稀疏 FlashAttention PyTorch ST 183 用例、MLA 前处理 PyTorch ST 27 用例）补齐污染形态与处置：

**1. 同 shape 不同属性相邻用例的漂移形态（稀疏 FA 轨迹）**：单进程模式（`--no-subprocess`）中，某 prefill 用例（S1=1024、BNSD、causal=True、bf16）FAIL `MERE=2.53e-01 MARE=4.22e+00 smallVal=908836/930265`，最大差异位置 Golden=0.032959/Actual=0.008301（数值系统性偏小，非 NaN）；**同一用例**在逐用例独立子进程模式稳定 PASS（`MERE=3.52e-07 MARE=7.81e-03`）。其前一用例是同 S1=1024、BSND、non-causal、fp16——shape 规模相同而布局/掩码/dtype 不同，正是进程内缓存撞键的高危组合。语义属性（is_causal）此前已恢复进 TilingKey，单进程串跑仍复现漂移——进程内还有其他维度的状态参与（布局/dtype，未定位，机制层待追）。

**2. 前序 NaN/设备错误用例的污染形态（MLA 轨迹）**：全零输入用例单独执行四输出全 PASS（MERE=0.00e+00），同进程批量执行（前序用例产生 NaN 输出与 507035 设备错误）中总结果 FAIL；**用例间加 100ms cooldown 后批量仍 FAIL**——污染不是简单的时序抖动，不要在 cooldown 方向耗时。kernel 修复后的健康版本上批量与单跑一致性恢复——污染只在"坏 kernel 时代"出现，但判读必须以单用例独立执行为准。

**3. cgroup 内存上限判别（exit 137）**：整批跑到第 76~80 个用例被 SIGKILL（`Killed ... EXIT_CODE=137`）时，`free -h` 显示系统内存充足（695GB available）、`dmesg` 无 OOM——必须再查 `/sys/fs/cgroup/memory.max`（本例 34359738368=32GB）：容器 cgroup 限额内长跑进程内存累积被 cgroup OOM killer 杀掉。逐用例进程隔离同时解决"被杀"与"算错"两类问题。

**4. 隔离编排（两套等价实现）**：串行版（bash 外部编排，逐用例 `timeout 300 python3 test.py --case-id <id>` 独立进程 + 结果解析汇总）与并行版（Python `ProcessPoolExecutor`，每用例仍是一个独立 subprocess，worker 数与单用例超时独立可调）。要点：隔离单位是 **OS 进程**，不是线程也不是同进程内清理；并行版失败用例必须串行复核（CPU 争用会放大 CPU golden 耗时导致单用例超时误判失败）。实测：逐用例隔离跑全量 183 用例，此前单进程模式会漂移的用例全部 PASS（首轮 178/183，5 个失败均为 CPU golden 内存问题而非 NPU 结果错误）。

### 复现增量验证信号

- 子进程模式全量跑到第 76 个用例均 PASS（含漂移用例 MERE=3.52e-07），之后进程 exit 137 被 SIGKILL（来源平台：950；证据等级：tool_observed）。
- 单进程模式同批次中漂移用例 FAIL MERE=2.53e-01/MARE=4.22e+00/smallVal=908836/930265；同批相邻用例仍 PASS（污染按用例组合选择性发生，非整体崩溃）（来源平台：950；证据等级：tool_observed）。
- 环境探针：`/sys/fs/cgroup/memory.max=34359738368`（32GB）、free -h 695GB available、dmesg 无 OOM——exit 137 由 cgroup 限制引起，与精度漂移是两个独立问题（来源平台：950；证据等级：tool_observed）。
- 隔离编排验证：逐用例独立进程全量 183 用例，此前漂移用例全部 PASS（首轮 178/183，5 个失败均为 CPU golden 内存问题，另卡闭合）（来源平台：950；证据等级：tool_observed）。
- MLA 轨迹：全零用例单独执行四输出全 PASS（MERE=0.00e+00，百万元素级）；同进程批量（前序含 NaN/507035 用例）总结果 FAIL；加 100ms cooldown 批量仍 FAIL；kernel 修复后终版批量与单跑一致（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 进程内污染的确切缓存层次与缓存键（GE 编译缓存？ACL tiling cache？键是否含布局/dtype/causal）未做机制级验证；is_causal 入 TilingKey 的先前修复不能消除本现象，残余维度未定位（复现并追根因为优先项）。
- 隔离不解决单用例自身的 CPU golden 内存/耗时问题（大 shape 用例的 golden OOM 与超时是独立根因）。
- 其他 CANN 版本/芯片的进程级缓存行为未验证；并行编排 worker 数需按宿主 CPU 内存调整（8 worker 时高内存 golden 用例会被 CPU 争用拖到超时，本轨迹降到 4 worker + 失败串行复核）。
- 逐用例隔离的墙钟成本：每用例重新 import torch 约 5~7s 固定开销，183 用例为主要成本；fork 预热能否避开污染未验证（spawn 语义下）。



## 增量（2026-09-17，并行策展批：逆序翻转定位实验、大幅错值形态与 V↔S 同步假设证伪，CANN 9.1.0）

RoPE 类自定义算子（aclnn 两段式 + AscendC VectorCore kernel）L1 全量 ST 的独立复现（600 条 CSV 用例，C++ ST 与 PyTorch adapter 两条接入路径），把污染的定位实验与形态再推进一步：

**1. 逆序执行失败集翻转——位置依赖的决定性证据**：正向 600 条 458/600（76.3%：half 297/300=99.0%、interleaved 161/300=53.7%），dtype/layout 分布均匀、D=2/D=2048 边界全过——"失败集中在某一 mode"是最大误导；把用例 CSV 逆序后跑全量，失败集随之翻转：half 降至 ~55%、interleaved 升至 ~99%。**先执行的大多通过、后执行的大多失败，与 mode/dtype/layout 无关**。一条逆序实验即可证伪"某 mode 的 kernel 分支缺陷"假设，避免一轮无效的 kernel 修改-重编-回归循环——建议纳入本卡判别方法的标准步骤。

**2. 大幅错值形态（非精度漂移）**：失败用例 MERE=6.45e+00~1.26e+01、MARE=1.58e+04~3.25e+06（阈值 1.22e-04/1.22e-03），数值完全错误；PyTorch 路径批量 50 条中首个 L1 用例即确定性失败（最大误差位置 [393828] golden=-1.778767/actual=1.560751，多次运行同位置同数值）。

**3. 排除实验补齐**：8 条失败 shape 隔离执行 100% 通过 ×3 次（MERE=0.00e+00）；失败用例前驱+自身成对连跑通过（排除相邻用例污染）；L1-only 600 条（去 L0 前缀）同分布 458/142；前 200 条单独 150/200、后 200 条单独 156/200（失败率相近——非简单累积计数阈值）；用例间 500ms 冷却（CASE_COOLDOWN_MS=500）仍失败（与本卡 9.2.0 增量实测的 100ms 冷却无效结论互证）；分进程执行 L0 15/15 + L1 前 35 条 35/35 全过。

**4. V↔S（Scalar/Vector 管线）竞态假设证伪（含完整同步代码）**：kernel 用 GetValue/SetValue 做 interleaved 布局重排、与 Vector 管线 Cast/Mul 并行执行，一度怀疑 Scalar 读到 Vector 未写完的数据；加双向 HardEvent 显式同步（每个 Scalar↔Vector 交替处共 4 个同步点）后重编（3 个 kernel 二进制中 2 个 .o 大小变化，证实重编生效）再跑 600 条，结果与修复前完全相同（458/142，interleaved 53.7%）——V↔S 管线竞态不是原因，改动已回退。同步函数（可作同类假设的对照实现）：

```cpp
    // V→S 同步：确保 Vector 管线操作完成后，Scalar 管线才读取数据
    __aicore__ inline void SyncV2S()
    {
        event_t evtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evtID);
        WaitFlag<HardEvent::V_S>(evtID);
    }

    // S→V 同步：确保 Scalar 管线操作完成后，Vector 管线才读取数据
    __aicore__ inline void SyncS2V()
    {
        event_t evtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtID);
        WaitFlag<HardEvent::S_V>(evtID);
    }
```

插入原则：V 写→S 读处插 SyncV2S，S 写→V 读处插 SyncS2V。本例同时说明：该 kernel 的 GetValue/SetValue 与 Vector 操作在此场景未表现出需要手动 V↔S 同步的竞争。

**5. 跨 harness 复现与触发位置差异**：C++ ST harness（约 300 条位置阈值）与 PyTorch adapter 路径（批量 50 条中首个 L1 用例即失败）均复现；两条 harness 触发位置不同的机制差异未解释。

**6. 边界再收紧：分批不充分**：本轨迹前/后 200 条分段执行仍各 ~25% 失败，与本卡先前增量的 12-15 条批次失败互证——分批本身不充分，进程级隔离仍是唯一被全量验证的执行方式；"≤200 条/进程 + 批间 aclrtResetDevice"的分批设想未被验证（aclrtResetDevice 批间重置有效性未实测；另一独立轨迹另证逐用例 aclrtResetDevice+aclrtSetDevice 无效）。

### 复现增量验证信号

- L0 15/15 通过（MERE=0.00e+00），L1 批量 600 条 458/600：half 297/300（99.0%）、interleaved 161/300（53.7%），两次独立全量运行分布一致（来源平台：950；证据等级：tool_observed）。
- 8 条失败 shape 隔离执行 100% 通过、重复 3 次（MERE=0.00e+00）；成对连跑通过；L1-only 600 条同分布 458/142（来源平台：950；证据等级：tool_observed）。
- 逆序 600 条：half 降至 ~55%、interleaved 升至 ~99%（失败集随执行顺序翻转；总体统计值自完整运行输出提取）（来源平台：950；证据等级：tool_observed）。
- 前 200 条单独执行 150/200（interleaved 51/100）、后 200 条单独执行 156/200（interleaved 56/100）（来源平台：950；证据等级：tool_observed）。
- PyTorch 路径批量 50：首个 L1 用例 MERE=1.26e+01/MARE=3.25e+06（位置 [393828]，golden=-1.778767/actual=1.560751，多次运行一致）；同用例隔离执行 PASS；用例间 500ms 冷却仍失败（来源平台：950；证据等级：tool_observed）。
- 分进程执行：L0 15/15 + L1 前 35 条 35/35 全过（来源平台：950；证据等级：tool_observed）。
- V↔S HardEvent 同步修复前后 600 条批量结果完全相同（458/142，interleaved 53.7%）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- "先跑的过、后跑的挂"的 ~300 条位置阈值只在 C++ harness 观测到；PyTorch 路径表现为批量 50 条中首个 L1 用例即失败——两条 harness 触发位置差异的机制未解释（复现并追根因为优先项，同本卡遗留问题）。
- 判别式依赖"隔离通过"前提：若 kernel 本身也有真实缺陷，需先修到隔离全过再套用本判别。
- 本增量结论限定 CANN 9.1.0 + Ascend950PR 实测环境；其他版本/芯片未验证，推断可迁移。
- 坏实践补充：看到失败率集中在某一 mode/dtype 就顺着该分支查 kernel 代码——先做一次逆序执行实验（成本一轮全量）即可证伪"mode 相关"假设；用 sleep/cooldown 或重试掩盖批量失败继续跑全量，把运行时问题漏进验收结论。