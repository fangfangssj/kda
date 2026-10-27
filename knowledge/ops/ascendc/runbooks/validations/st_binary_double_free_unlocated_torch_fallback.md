---
type: "Runbook"
title: ST 二进制 double free 未定位改 torch 路线
description: 多 vendor 共享 CANN 环境下 C++ ST 测试二进制直连自定义 aclnn 库启动即 double free（tcache 2）；最小复现与同构 GRU 均正常，9 类尝试未消除，根因未定位（无 gdb/ASAN）；改用 torch_adapter 共享库 + Python 路线完成验证。
tags: [task.debugging, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# ST 二进制 double free 未定位改 torch 路线

## 现象

在 Ascend950PR + CANN 9.2.0 的多用户共享环境（`$ASCEND_HOME_PATH` 指向 cann-9.2.0，`opp/vendors/` 下装有 6 个自定义算子 vendor）联调自定义算子 ST 真机测试。算子包已重建重装、UT 12/12 通过、参数校验错误已修复，ST 测试二进制（`test_aclnn_mla_prolog`，链接 `libascendcl.so` + `libopapi.so` + `libnnopbase.so` + vendor 的 `libcust_opapi.so`）构建成功，但一执行就崩：

```
========================================
执行测试
========================================
CSV 用例: ../testcases/L0_test_cases.csv
free(): double free detected in tcache 2
.../tests/st/run.sh: line 225: 1619762 Aborted                 (core dumped) ./test_aclnn_mla_prolog "${CASE_FILE}"
```

用 `-O0 -g` 重编并在 `main()` 首行加 `printf` 后，能打印 `[DEBUG] main() started`，随后同样 `free(): double free detected in tcache 2`——崩溃发生在 main 启动后、任何测试逻辑执行前；另一次 `LD_DEBUG=libs` 运行显示崩溃出现在库 fini（卸载）阶段。直接执行二进制同样立即崩溃、无任何输出。崩溃首次出现在"重命名 + 参数校验修复后的新 libcust_opapi.so 重链"之后；但同一库在最小程序中加载正常，无法归因于库本身。

## 适用场景

- 工程形态：自定义算子（custom opp）ST 真机测试，C++ 单二进制直连 aclnn 库路线。
- API 路径：链接 `libascendcl.so`/`libopapi.so`/`libnnopbase.so`/`<vendor>/op_api/lib/libcust_opapi.so` 的 C++ 测试工程；`opp/vendors/config.ini` 的 `load_priority` 机制（`libnnopbase.so` 按其逐 vendor dlopen）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（多用户共享安装）。
- 适用粒度：C++ ST 路线的受阻判定与替代路线选择（本卡为负结果，不宣称 kernel 行为）。
- 路线优先级：C++ 单二进制直连路线在本环境标记为受阻（不可作为验证手段）；替代路线 **safety_fallback**：`torch_adapter.cpp` 编译为共享库 + Python（torch_npu）调用同一组 aclnn 两段式接口完成 NPU 精度验证。fallback 声明：放弃 C++ 单二进制直连，接受"Python 进程 + 适配器共享库"的额外工程形态与 Python 侧开销，换取绕开宿主进程启动期的崩溃。
- 明确不适用/未验证场景：非共享环境/其他机器未必复现；GRU 算子的同构 C++ ST 二进制在本环境运行正常，说明路线并非普适性损坏。

## 尝试与观测（负结果，全部可重建）

按时间序的完整尝试矩阵（均为实际执行过的命令/改动）：

1. **链接顺序调整**（`target_link_libraries` 重排）→ 仍崩。
2. **从链接行去掉 `libopapi.so`**（`aclCreateTensor` 实际定义在 `libnnopbase.so`，`nm -D` 全库扫描证实）→ 仍崩。
3. **改 dlopen/dlsym 动态加载**（测试不再直接链接 `libcust_opapi.so`，改 `dlopen("<vendor>/op_api/lib/libcust_opapi.so", RTLD_NOW)` + 函数指针调用）→ 仍崩。
4. **ABI 与工具链变量**：去掉 `-D_GLIBCXX_USE_CXX11_ABI=0`、换静态 libstdc++、`-O0`、`-no-pie`、单步 g++ 编译 → 均仍崩。
5. **`MALLOC_CHECK_=0`** → 仍报 double free。
6. **`GLIBC_TUNABLES=glibc.malloc.tcache_count=0`**（关 tcache）→ 变为 `Segmentation fault (core dumped)`，EXIT_CODE=139——证明存在**真实内存破坏**，不是 tcache 误报。
7. **库解析排查**：`LD_DEBUG=libs ./test ... 2>&1 | grep cust_opapi` 发现某次运行动态链接器把 `libcust_opapi.so` 解析到**外来用户的 vendor 路径**（`calling init:`/`calling fini:` 均落在外来路径）；把目标 vendor lib 目录前置到 `LD_LIBRARY_PATH` 后确认解析到本算子 vendor → **仍崩**（该外来路径是并行发现的环境隐患，不是本崩溃根因，判定加载源的探针见同批 LD_DEBUG 卡）。
8. **vendor 隔离**：把 `opp/vendors/config.ini` 的 `load_priority` 缩减到仅本算子 vendor → 仍崩（排除多 vendor 连锁）。
9. **全量重建重装**：`rm -rf build/ && bash build.sh --soc=ascend950` + 重装 `.run` 包 → 仍崩。

对照观测（证明"库组合本身无罪"）：

- 链接同组库（`libcust_opapi.so` + `libnnopbase.so` + `libascendcl.so`）的最小 Hello 程序 → 正常。
- 复刻测试文件全部 include（`acl/acl.h`、`aclnn/acl_meta.h`、`cpu_golden.h`）+ dlopen 代码 + 静态变量的 min_test 系列（逐步加料）→ 全部正常（含 `aclInit`/`aclrtSetDevice` 走通）。
- **同环境 GRU 算子的 ST 二进制**（`test_aclnn_bench_no_harness_gru`，同样链接结构、同样 CANN）→ 运行正常，用例执行、golden 输出正常。
- 工具可用性：gdb 因 ptrace 权限受限不可用；`libasan.so.5` 缺失无法 ASAN——这是根因未定位的直接原因。

## 根因

未定位（诚实声明）。已排除：测试代码静态初始化器（`readelf -S` 查 `.init_array`/`.fini_array` 为空）、`libcust_opapi.so` 静态构造（init_array 与 GRU 库同为 3 项、结构一致）、libopapi 直接参与（去掉仍崩）、ABI/优化/PIE、多 vendor config.ini 连锁、外来 vendor 路径抢占、构建产物损坏（重建重装仍崩）。崩溃介于 main 启动早期与库 fini 之间、随构建形态漂移；tcache 关闭后转为段错误证明内存破坏真实存在。在无 gdb/ASAN 的环境下无法进一步收敛到重复释放点。

## 系统信息

- 芯片 Ascend950PR（arch35）；CANN 9.2.0；GCC 13.3.0；CMake 3.20.5；多用户共享安装（`opp/vendors/config.ini`: `load_priority=mla_prolog_custom,bench_no_harness_lstm_custom,custom_ops,bench_no_harness_gru_custom,BenchHarnessApplyRotaryPosEmb,BenchHarnessApplyAdamW`）。
- 相关文件：`tests/st/CMakeLists.txt`（链接行与 dlopen 化改动）、`tests/st/test_aclnn_mla_prolog.cpp`（直连/dlopen 两版调用）、`tests/st/run.sh`（LD_LIBRARY_PATH 注入）、`<vendor>/op_api/lib/libcust_opapi.so`、`opp/vendors/config.ini`、临时最小复现程序 min_test 系列（已清理）。

## 预防

- 共享环境下跑自定义算子 ST 前，先用 `LD_DEBUG=libs` 确认 `libcust_opapi.so` 实际解析路径（多 vendor 同 SONAME，`LD_LIBRARY_PATH` 顺序决定加载谁）。
- 测试工程尽量复用同仓已验证可跑的算子的 ST 骨架（链接结构、run.sh 环境注入），少自创链接形态。

## 验证信号

来源说明：以下为该次联调排查过程的实测信号（崩溃与对照观测）；**本卡为负结果卡，无"修复成功"信号**。

- 崩溃现象：`free(): double free detected in tcache 2` + `Aborted (core dumped)`（run.sh 与直跑均复现）；`-O0` 下 `[DEBUG] main() started` 后崩（来源平台：950；证据等级：tool_observed）。
- `GLIBC_TUNABLES=glibc.malloc.tcache_count=0` 下转为 `Segmentation fault (core dumped)`，EXIT_CODE=139（来源平台：950；证据等级：tool_observed）。
- 对照：链接同组库的最小程序正常；min_test 系列正常；GRU 同构 ST 二进制正常（来源平台：950；证据等级：tool_observed）。
- LD_DEBUG 观测到外来 vendor 路径的 `libcust_opapi.so` 被解析加载；目标路径前置后解析正确但崩溃不变（来源平台：950；证据等级：tool_observed）。
- 替代路线决策：改派 PyTorch ST（torch_adapter 共享库 + Python/torch_npu 调用同一组 aclnn 接口）完成 NPU 精度验证（决策与派发为实际发生的动作；其 NPU 验证结果在后续联调中闭合，不在本卡证据范围内）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限于该共享环境与该测试工程形态；最小复现不崩、GRU 同构不崩说明触发条件未识别，**不可**据本卡断言"C++ 直连 aclnn 路线普遍不可用"。
- 外来 vendor 路径解析（LD_LIBRARY_PATH 污染）在本例中不是崩溃根因；但它是独立成立的共享环境隐患，纠正加载路径的正确性不受本卡负结果影响。
- 未验证，推断可迁移：具备 gdb/ASAN/valgrind 的环境可复现并定位。

**未决项**：

- **优先项：复现并追根因到机制层**——在具备 gdb（或放开 ptrace）/ASAN/valgrind 的环境复现本崩溃，定位真实重复释放点；对比 GRU 与本测试二进制的符号表、初始化段与链接差异；排查 driver/libascend_hal 层的 fini 顺序。
- PyTorch ST 替代路线的精度结论与该路线的稳定性需在后续联调记录中闭合。
## 坏实践

- 在无调试器环境下反复盲试链接形态变体（本例 9 类尝试均无效），消耗大量联调时间——应更早切换替代验证路线。
- 把 `MALLOC_CHECK_=0` 当"绕过"手段（掩盖不了 tcache 检测，且内存破坏仍在）。

## 误诊

- "符号冲突未修干净"：361001 已消失、错误链路已进入自定义实现，排除。
- "外来 vendor 库抢占导致"：LD_DEBUG 发现外来路径后一度认为找到根因；纠正 `LD_LIBRARY_PATH` 后崩溃依旧，证伪（该发现本身仍有效，见失效边界）。
- "静态初始化器问题"：`readelf` 查 init_array 为空、min_test 全过，排除。
- "环境兼容性问题（非测试代码逻辑）"：最终判断——最小复现与 GRU 均正常，指向该测试二进制与环境的特定组合，但未证实（假设，机制层未验证）。
