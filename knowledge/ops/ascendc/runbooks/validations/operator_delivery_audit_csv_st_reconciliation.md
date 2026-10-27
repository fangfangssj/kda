---
type: "Runbook"
title: 算子交付审计：验证真 AscendC kernel 与 ST 实测覆盖
description: 审计算子仓交付是否为手写 AscendC kernel（而非 torch/已有算子包装）、计算是否全在 kernel、CSV 用例数与 ST 实跑数是否对账、白盒用例是否存在的完整命令配方；本例发现 CSV 520 条中 C++ ST 实际只执行 115 条（104 builtin + 11 条缩放 shape 的 L0），L1/L2 CSV 未被加载。
tags: [task.testing, topic.implementation.operator_registration, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# 算子交付审计：验证真 AscendC kernel 与 ST 实测覆盖

## 现象

BenchNoHarnessSwiGlu 算子（Ascend950PR/arch35，CANN 9.1.0，算子仓开发流程）全流程开发完成（UT 46/46、C++ ST 115/115、PyTorch ST 70/70、性能与精度验收报告齐备）后，用户下发交付审计请求，明确"禁止修改已有代码"，要求核查四件事并落盘 `MANUAL_CHECK.md`：① 是否确实开发了 AscendC 算子，而不是用 torch 接口走已有算子实现；② 是否所有计算逻辑都在 kernel 内完成，而不是在 host 上实现；③ `tests/st` 下 CSV 共有多少条黑盒用例，手动运行 C++ ST 真实 NPU 用例，总结执行了多少条、通过多少条、执行的用例是否就是 CSV 中的用例；④ 是否有 `tests/whitebox` 目录，如有共多少条白盒用例并手动运行。

审计按"读源码验证真伪 → 数 CSV → 数内置用例 → 实跑 ST 计数对账 → 查白盒目录"执行，全部检查只用只读命令（find/grep/wc/ls）加一次真实 NPU ST 运行。结论：确为手写 AscendC kernel 且计算全在 kernel；但**用例对账存在显著差距**——`tests/st/testcases` 三个 CSV 共 520 条（L0=12、L1=500、L2=8），C++ ST Real NPU 实际执行 115 条 = 104 条内置硬编码用例（92 个 `addCase(` + 12 个 `addNonContCase(`）+ 11 条 L0 CSV 用例（经 `ScaleDownShape` 缩小 shape，非原始值；L0 CSV 首条用例因 dim=-4 对 rank-2 张量无效被 `tc.valid` 过滤），L1/L2 CSV 完全未被测试程序加载（源码中引用计数为 0）；`tests/whitebox` 目录不存在。115 条实跑全部通过。

## 适用场景

- 工程形态：算子仓开发流程（op_host/op_kernel/op_api/op_graph + build.sh + tests/{ut,st}）交付后的验收审计 / 防作弊核查；用户或上游要求核实"LLM/Agent 生成的算子交付是真实现"时。
- API 路径：`ADD_TO_LAUNCHER_LIST_AICORE`（自定义 kernel 注册调度入口）、`bash build.sh --soc=<soc>`、`bash build/custom_opp_ubuntu_x86_64.run`（vendors 安装）、`tests/st/run.sh --real`（真实 NPU ST）、`tests/ut/run.sh`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——审计流程与命令可跨算子复用，但具体用例计数 grep 模式（`addCase(`/`addNonContCase(`/`[Real]` 前缀/`ScaleDownShape`）依赖目标工程 ST 脚手架的命名约定，换工程需先读脚手架源码适配。
- 路线优先级：recommended_fastpath（交付审计的标准做法）。
- 明确不适用 / 未验证场景：不适用于 ops-direct-invoke 直调工程（无 vendors 安装环节，环境变量与运行方式不同）；PyTorch ST（torch 适配层）的用例计数未在本卡覆盖（本例只对账了 C++ ST）；本卡不判定 kernel 实现本身的正确性（只验真伪与覆盖对账）。检查 ①②（实现真伪、计算归属）的深度判读标准另见同域"自主 AscendC kernel 静态核查配方"与"host/kernel 计算归属审计"卡；本卡的独特增量是 ③④ 的用例覆盖对账。

## 判别方法

四项检查各自的判定标准：

1. **真 AscendC kernel（防 torch/已有算子包装）**：`grep -r "torch|torch_npu|<基线算子 aclnn 名>"` 限定 `op_kernel/`、`op_host/`、`op_api/`、`op_graph/` 四个实现目录，命中即疑（本例 torch/aclnnSwiGlu 仅出现在 `tests/perf/perf_test.cpp` 基线对比与 `tests/st/torch/` 适配层，属正常）；正向证据为 kernel 侧出现 `__global__ __aicore__`、AscendC 原生 API（DataCopyPad/Cast/Muls/Exp/Adds/Div/Mul、GlobalTensor/LocalTensor/TQue/TBuf/TPipe），且 L0 API 经 `ADD_TO_LAUNCHER_LIST_AICORE(<Op>, ...)` 注册调度。
2. **计算全在 kernel**：逐个 host 侧文件核对职责——`*_def.cpp`（算子定义）、`*_infershape.cpp`（shape 推导）、`*_tiling.cpp`（切分计算）、L2 API（参数校验/Contiguous 处理）、L0 API（调度）——均不得出现算子的数学运算；计算链（本例 Cast→Muls→Exp→Adds→Div→Mul）应完整出现在 kernel `Compute()` 中。
3. **CSV 用例数与实跑数对账**：CSV 条数用 `wc -l` 减头行；内置用例数用 `rg -c 'addCase\('` + `rg -c 'addNonContCase\('`（按脚手架实际注册函数名调整）；实跑数用 ST 输出的 `[Real]` 行前缀 grep 计数并与测试报告总数互验；再 grep ST 源码确认加载了哪几个 CSV 文件、是否对 shape 做缩小（本例 `ScaleDownShape` 把 dim 维换成 8、其余维换成 2）。若 CSV 条数 ≫ 实跑数，必须如实披露差距而不是只报通过率。
4. **白盒用例**：`ls tests/whitebox` 判存在性；不存在时如实记录"无白盒用例"，不得跳过该检查项。

## 完整代码

审计命令序列（verbatim 取自本会话执行的命令；`OP_DIR` 为算子目录、`ASCEND_HOME_PATH` 为 CANN 安装根、`<soc>` 为目标芯片，按目标工程替换）：

```bash
# ① 源码清单（排除构建产物）
find <OP_DIR> -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.csv" -o -name "*.py" -o -name "*.sh" \) \
  | grep -v '/build' | grep -v '/build-mock' | grep -v '/build-real' | grep -v '__pycache__' | sort

# ② 防 torch / 已有算子包装（只查实现目录；命中 tests/ 下属正常）
grep -r --include='*.{cpp,h,py}' -E 'torch|npu_swiglu|aclnnSwiGlu|torch_npu' <OP_DIR>
# 正向证据：读 op_kernel/*.cpp 与 op_kernel/arch35/*.h 确认 __global__ __aicore__ + AscendC API；
# 读 op_api/*.cpp 确认 ADD_TO_LAUNCHER_LIST_AICORE(<Op>, ...) 调度。

# ③ CSV 用例计数（减去表头）
wc -l <OP_DIR>/tests/st/testcases/*_l0_test_cases.csv
tail -n +2 <OP_DIR>/tests/st/testcases/*_l0_test_cases.csv | wc -l   # L0 数据行
tail -n +2 <OP_DIR>/tests/st/testcases/*_l1_test_cases.csv | wc -l   # L1 数据行
tail -n +2 <OP_DIR>/tests/st/testcases/*_l2_test_cases.csv | wc -l   # L2 数据行
# 确认 ST 程序加载了哪些 CSV（本例仅 L0 被加载，返回 0 即未引用）
rg -c 'l1_test_cases\|l2_test_cases' <OP_DIR>/tests/st/test_<op>.cpp || echo "0 (L1/L2 CSV not loaded)"

# ④ 内置用例计数（按脚手架注册函数名调整）
cd <OP_DIR>/tests/st && rg -c 'addCase\(' test_<op>.cpp; rg -c 'addNonContCase\(' test_<op>.cpp

# ⑤ 白盒目录存在性
ls -la <OP_DIR>/tests/whitebox 2>/dev/null && echo "whitebox EXISTS" || echo "whitebox directory does NOT exist"

# ⑥ 实跑 C++ ST（真实 NPU）：先装包，再带 vendors lib 路径运行
source $ASCEND_HOME_PATH/set_env.sh
export ASCEND_HOME_PATH=<CANN_ROOT>
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib:$LD_LIBRARY_PATH
cd <OP_DIR>/tests/st && bash run.sh --real 2>&1 | tail -40        # 看测试报告（总计/通过/失败/跳过）

# ⑦ 实跑用例数对账（与测试报告总数互验）
cd <OP_DIR>/tests/st && bash run.sh --real 2>&1 | grep -E '^\[Real\]' | wc -l                    # 实跑总数
bash run.sh --real 2>&1 | grep -E '^\[Real\] <Op>_L0_' | wc -l                                     # CSV 来源条数
# 注意：整套 ST 可能跑数分钟，计数命令每次都会整跑一遍，勿在同一命令里串联多个整跑（本会话曾因此 300s 超时）
```

## 设计决策

- 为何用"grep 实现目录 + 正向读 kernel/API 源码"双通道验真伪：反向 grep 只能证明"没调用 torch/基线算子"，还需正向确认 `__global__ __aicore__`、AscendC API 与 `ADD_TO_LAUNCHER_LIST_AICORE` 调度链存在，两方向合起来才能下"真 AscendC kernel"结论。
- 为何"数 CSV + 数内置 + 数实跑"三方对账而不是只看测试报告的通过率：通过率 100% 无法回答"跑的是不是 CSV 里的用例"；本例对账后才发现 L1/L2 CSV（508 条）完全未加载、L0 的 11 条被缩小 shape——这类覆盖差距只有计数对账能暴露，直接关系反投机取巧（防止"宣称覆盖、实际没跑"）。
- 为何实跑要 `--force` 安装包并显式设置 vendors `op_api/lib` 的 `LD_LIBRARY_PATH`：本会话安装 custom_opp 包时共享库校验失败（`libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`），`--force` 跳过校验后运行时在正确库路径下符号可解析、ST 正常；该符号问题的根因与处置路线见同域 l0op::Contiguous 未定义符号卡。审计目标是覆盖对账，不在安装工具链问题上分叉。
- 为何白盒目录不存在也要如实记录：审计请求的第四项是存在性检查，"没有"本身就是审计结论的一部分（交付缺少白盒用例是事实披露，不是可跳过项）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510，arch35），CANN 9.1.0，cmake 3.20.5，g++ 13.3.0。
- 审计对象目录结构：`op_host/`（def/infershape/arch35 tiling）、`op_kernel/`（arch35 入口 + 类头）、`op_api/`（L0/L2 aclnn）、`op_graph/`（proto）、`tests/ut/`（run.sh）、`tests/st/`（run.sh --real、test_<op>.cpp、testcases/*.csv、torch/）、`tests/perf/`、`build/custom_opp_ubuntu_x86_64.run`。
- 相关文件：`tests/st/test_aclnn_bench_no_harness_swi_glu.cpp`（用例注册与 CSV 加载逻辑所在）、`tests/st/testcases/aclnnBenchNoHarnessSwiGlu_{l0,l1,l2}_test_cases.csv`、`op_api/bench_no_harness_swi_glu.cpp`（ADD_TO_LAUNCHER_LIST_AICORE）、`MANUAL_CHECK.md`（审计结论落盘）。

## 验证信号

以下信号来自该算子交付后的手动审计会话：全部检查命令实际执行并回读输出，ST 在真实 NPU 上完整跑通一轮。

- 真伪核查：`op_kernel/`、`op_host/`、`op_api/`、`op_graph/` 中 grep torch/torch_npu/aclnnSwiGlu/npu_swiglu 无实现侧命中（torch 仅在 `tests/st/torch/` 适配层，aclnnSwiGlu 仅在 `tests/perf/perf_test.cpp` 作基线对比）；kernel 侧含 `__global__ __aicore__` 与 DataCopyPad/Cast/Muls/Exp/Adds/Div/Mul 原生 API，L0 API 经 `ADD_TO_LAUNCHER_LIST_AICORE` 调度（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。
- host 无计算：`*_def.cpp`/`*_infershape.cpp`/`*_tiling.cpp`/L2/L0 API 逐文件核对均为定义、shape 推导、切分与调度，SwiGLU 计算链完整位于 kernel `Compute()`（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。
- 用例对账：CSV 计数 L0=12、L1=500、L2=8（共 520）；内置注册 `addCase(`=92、`addNonContCase(`=12（合计 104）；ST 实跑 `[Real]` 行数=115，其中 CSV L0 前缀 11 行；测试报告"总计 115 / 通过 115 / 失败 0 / 跳过 0"；ST 源码对 l1/l2_test_cases 引用计数 0（来源平台：950；证据等级：tool_observed）。
- ST 实跑通过：115/115 在真实 NPU 上通过（MERE/MARE 全部低于阈值，如 fp32 大 shape MERE=1.21e-09，阈值 1.22e-04）（来源平台：950；证据等级：tool_observed）。
- 白盒：`tests/whitebox` 目录不存在，无白盒用例（来源平台：950 开发环境宿主侧；证据等级：tool_observed）。

## 失效边界

- 计数 grep 模式（`addCase(`/`addNonContCase(`/`[Real]` 前缀/`ScaleDownShape`）绑定本类 ST 脚手架的命名约定；其他工程的脚手架注册函数、输出前缀、CSV 加载方式不同，须先读脚手架源码适配（方法本身可迁移，模式不可照抄）。
- CSV 用例被缩小 shape（`ScaleDownShape`）意味着 CSV 原始大 shape 的覆盖并未真正执行；本卡只披露该事实，未验证原始 shape 下的行为（PyTorch ST 另有 70 条真实 shape 用例，不在本卡对账范围）。
- L0 首条用例被过滤归因于 dim=-4 对 rank-2 无效，系由 CSV 内容与差值推断（与脚手架 `tc.valid` 检查逻辑一致），未直接打印过滤日志。
- `--force` 跳过安装校验仅服务于审计推进，不构成"包符号完整性无问题"的结论；其他环境照抄可能有风险。
- 其他平台/形态未验证（审计方法为 host 侧只读命令 + 一次 ST 实跑，推断可迁移）。

**未决项**：

- L1/L2 CSV（500+8 条）未被 C++ ST 程序加载、L0 用例 shape 被缩小——交付方是否需要补全加载与原始 shape 执行，属交付决策，本卡只提供发现与对账方法。
- PyTorch ST 的用例数与 C++ ST 的关系（是否存在同样"宣称 vs 实跑"差距）未按同一方法对账。
## 坏实践

- 只引用测试报告的"通过率 100%"作为覆盖证据，不核对"宣称的用例数 vs 实际执行的用例数"——本例 520 条 CSV 中 508 条未被 C++ ST 加载，报告数字完全掩盖了这一差距。
- 用 torch 适配层或基线算子（如 aclnnSwiGlu）冒充自定义算子实现——grep 实现目录可直接识破。
- 把计算逻辑放到 host 侧 tiling/infershape 里凑"kernel 已实现"的表象——逐文件职责核对可识破。
- 在一条命令里串联多次整套 ST 运行做计数（每次整跑数分钟，容易超时）；应一次运行保存输出再多次 grep。

## 误诊

- 无（本审计各检查项均有命令输出直接支撑；安装校验失败的归因已在同域 l0op::Contiguous 未定义符号卡中定位，此处不再重复假设）。
