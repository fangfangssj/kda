---
type: "Runbook"
title: 多 vendor 同名 TilingFunc 符号冲突 561002
description: "多个 vendor 的 liboptiling.so 都导出同名 optiling::TilingFunc 外部符号，load_priority 靠前的先加载后，后加载算子的 IMPL_OP_OPTILING 注册把 TilingFunc 解析到别人的定义，tiling 调进别的算子源码报 OPTILING_FAILURE/INVALID_TILING/561002；治本修复是 TilingFunc 加 static（内部链接），config.ini 提前只是会被并发安装重置的临时手段。"
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# 多 vendor 同名 TilingFunc 符号冲突 561002

## 现象

一台共享的 Ascend 950 开发机（Ascend950PR_9579，CANN 9.2.0）上，同一 CANN 安装（`$ASCEND_OPP_PATH/vendors/`）下并存多个自定义算子 vendor（AdamW、Softmax、AddCustom、RotaryPosEmb 等）。AdamW 算子（`BenchHarnessApplyAdamW`，full-load tilingKey=0 / split-load tilingKey=1 双分支）此前白盒 32 例跑出 29 PASS / 3 FAIL；一次全量重跑突然变成 2 PASS / 30 FAIL，`result.csv` 里 `dyn_precision=INVALID_TILING`、`perf_status=FAIL`，TTK 报 `RuntimeError: OPTILING_FAILURE`。单独跑一个用例却能 PASS——全量与单例结果不一致。

同时算子包自带的 aclnn C++ 测试（`tests/aclnn` 的 `run.sh`）**所有用例**统一失败，plog（`ASCEND_GLOBAL_LOG_LEVEL=1`）中的关键错误：

```
[ERROR] OP(...):...[indv_executor.cpp:879][NNOP][NnopbaseExecutorDoTiling][...] errno[1228972098] OpName:[BenchHarnessApplyAdamW_0] Failed to execute operator BenchHarnessApplyAdamW_0. Reason: Failed to execute tiling function.
```

即对外报 ACLNN 561002（tiling 函数执行失败）。而算子包构建本身 `EXIT=0`，kernel 二进制正常生成，且最近的代码提交只改了 kernel 侧文件、完全没动 host/tiling 代码；TTK 侧与属性无关——连 `attributes={}`（全默认）的用例也失败。

开 plog 抓到决定性一幕：**给 AdamW 算子做 tiling 时，执行的是 Softmax 工程里的 tiling 函数**：

```text
[ERROR] OP(...):[...] [.../BenchHarnessSoftmax_package/op_host/arch35/bench_harness_softmax_tiling_arch35.cpp:203][OPS_BASE][TilingFunc][...] OpName:[BenchHarnessSoftmax] dim=4568451461268312687 out of range [-rank=-2, rank=2)
[ERROR] GE(...):[...][op_tiling_py.cc:1954]... ErrorNo: 4294967295(failed)Op BenchHarnessApplyAdamW tiling failed
```

即 Softmax 的 `TilingFunc` 拿着 AdamW 的输入（shape/rank 完全对不上）去算，报 dim out of range，AdamW 的 tiling 失败（`dim=4568451461268312687` 这种垃圾值就是 Softmax 的 TilingFunc 按 Softmax 的语义解析 AdamW 的输入得到的）。

## 适用场景

- 工程形态：vendor 包部署的自定义算子（op_host tiling 经 `npu_op_library(cust_optiling TILING)` 编入 liboptiling.so）+ aclnn C++ 测试 + TTK kernel 模式白盒执行；多算子共用一个 CANN 安装/一台共享开发机，或测试机上有并发安装其他 vendor 包的任务。
- API 路径：`IMPL_OP_OPTILING(<OpType>)` 注册宏 + `gert::OpImplRegisterV2::Tiling(TilingFunc)`；`$ASCEND_OPP_PATH/vendors/<Vendor>/op_impl/ai_core/tbe/op_tiling/liboptiling.so`；`opp/vendors/config.ini` 的 `load_priority` 决定 liboptiling.so 加载顺序。
- 验证过的平台与 CANN 版本：Ascend950PR_9579（注册平台 950）+ CANN 9.2.0。其他平台未验证。
- 适用粒度：platform_wide——同机多 vendor 部署下任何导出同名 `optiling::TilingFunc` 的算子对。
- 路线优先级：recommended_fastpath（static 修复一步到位且不受加载顺序影响）。
- 明确不适用 / 未验证场景：单 vendor 独占安装不会触发；`para_check` 的 E80003 参数类型报错（编译期 Python 异常，另一根因）；tiling 代码自身返回 GRAPH_FAILED 的真实 tiling 缺陷；软链接/静态合入等其他链接形态未验证。

## 根因

**根因一句话：多个 vendor 的 liboptiling.so 都以外部链接导出同名符号 `optiling::TilingFunc(gert::TilingContext*)`（mangled: `_ZN8optiling10TilingFuncEPN4gert13TilingContextE`），动态链接器把后加载 .so 里 `IMPL_OP_OPTILING(...).Tiling(TilingFunc)` 中的 `TilingFunc` 符号解析到先加载 .so 的定义，导致后加载算子注册到了别人的 tiling 函数。**

机制链：

1. 每个算子的 arch35 tiling 胶水都按 CANN 模板写 `namespace optiling { ge::graphStatus TilingFunc(gert::TilingContext* context) {...} }`——外部链接，符号名相同；
2. `IMPL_OP_OPTILING(BenchHarnessApplyAdamW)` 展开为静态初始化对象，构造时调 `.Tiling(TilingFunc)` 注册**函数指针**；
3. 运行时进程按 `config.ini` 的 `load_priority` 依次 dlopen 各 vendor 的 liboptiling.so；Softmax 在前时，AdamW 的 .so 里对 `TilingFunc` 符号的引用被解析到 Softmax 的定义（ELF 符号插拔/先到先得）；
4. AdamW 注册表里存的是 Softmax 的函数指针 → AdamW 的 tiling 请求调进 Softmax 代码 → shape 语义不符 → dim out of range → `OPTILING_FAILURE`，TTK 侧表现为 `INVALID_TILING` 批量失败，aclnn 侧表现为 561002 `Failed to execute tiling function`。

判别证据（`nm -D` 两个 .so 的输出，两者都是 `T` 全局导出）：

```
=== AdamW TilingFunc symbol ===
0000000000020c19 T _ZN8optiling10TilingFuncEPN4gert13TilingContextE
=== Softmax TilingFunc symbol ===
000000000002214c T _ZN8optiling10TilingFuncEPN4gert13TilingContextE
```

最直接的指纹在报错日志里：**报错的是另一个算子的 tiling 源文件，而被编译/执行的是当前算子**（见现象段的 plog 摘录）。

前置 workaround（改 `config.ini` 把自家 vendor 提到 `load_priority` 最前）不可靠：CANN 的 vendor `install.sh` 安装时会把**正在安装的 vendor 前插**到 `load_priority`（`sed -i "s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g"`），实测两次把 AdamW 提前后均被并发的 Softmax 任务安装改回 `BenchHarnessSoftmax,...` 在前。

## 解决配方

治本修复 = 把自家 TilingFunc 改为**内部链接**（`static`），声明与定义都要改：

`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.h`（before → after，verbatim）：

```cpp
// before
ge::graphStatus TilingFunc(gert::TilingContext* context);

// after
// static (internal linkage): prevents the exported `optiling::TilingFunc`
// symbol from colliding with the same-named symbol exported by other vendors
// (e.g. BenchHarnessSoftmax) whose liboptiling.so is loaded into the same
// process. The framework invokes TilingFunc via the function pointer
// registered by IMPL_OP_OPTILING, not via dlsym, so internal linkage is safe.
static ge::graphStatus TilingFunc(gert::TilingContext* context);
```

`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.cpp`（before → after，verbatim）：

```cpp
// before
ge::graphStatus TilingFunc(gert::TilingContext* context)

// after
static ge::graphStatus TilingFunc(gert::TilingContext* context)  // static (internal linkage) — see header
```

要点：

- 头文件是该 .cpp 的唯一包含者时，声明与定义都加 `static` 保持一致（C++ 中首个声明为 static 则定义继承内部链接，但显式写出更清晰）；
- **安全性依据**：框架经 `OpImplRegisterV2::Tiling(...)` 注册的函数指针调用，不做 dlsym 按名查找，内部链接不影响注册与回调；
- 重编后用 `nm -D` 验证：自家 liboptiling.so 中 `TilingFunc` 动态符号数必须为 0，且 `OpImplRegisterV2` 相关注册符号仍在。

验证命令（verbatim）：

```bash
# 重编安装（clean build + .run 安装）
bash operators/build.sh BenchHarnessApplyAdamW_package
# 符号验证：应为 0
nm -D $ASCEND_OPP_PATH/vendors/BenchHarnessApplyAdamW/op_impl/ai_core/tbe/op_tiling/liboptiling.so | grep -c "TilingFunc"
# 对拍：冲突方（Softmax）仍导出同名符号——不修改他人工程也能自愈
nm -CD $ASCEND_OPP_PATH/vendors/BenchHarnessSoftmax/op_impl/ai_core/tbe/op_tiling/liboptiling.so | grep "TilingFunc"
```

临时 workaround（仅当无法改源码时；会被并发安装重置，见根因）：把 `config.ini` 的 `load_priority` 中当前被测算子的 vendor 挪到第一位。操作要点：先备份 `cp $CONFIG $CONFIG.bak.<tag>`；改完 `cat` 确认内容真的变了再跑测试（若编辑工具报成功但 `cat` 仍见旧内容，直接用 `printf 'load_priority=...\n' > $CONFIG` 重写）；每次运行前重新 `cat` 检查一次；测完恢复并删除备份。诊断命令：

```bash
nm -D $ASCEND_OPP_PATH/vendors/<VendorA>/op_impl/ai_core/tbe/op_tiling/liboptiling.so | grep TilingFunc
nm -D $ASCEND_OPP_PATH/vendors/<VendorB>/op_impl/ai_core/tbe/op_tiling/liboptiling.so | grep TilingFunc
cat $ASCEND_OPP_PATH/vendors/config.ini
```

两个 .so 输出同一 mangled 符号 + `load_priority` 里别的 vendor 在前，即确诊。

## 设计决策

- static vs 改名（`TilingFunc` → `<OpType>TilingFunc`）：改名同样消除冲突且更显式，但要同时动头文件声明、定义、注册点三处且改公共模板约定；static 只加一个关键字，对模板侵入最小。两者都成立，本轨迹选 static。
- static vs 持续维护 config.ini 顺序：config.ini 是**环境共享状态**，任何并发 vendor 安装都会重写（install.sh 前插逻辑是 CANN 脚本固有行为），把它当作修复载体等于把正确性押在无人协调的共享文件上；static 把不变量收进自己的 .so。定位过程中评估过 5 条路线（改 static / 改唯一函数名 / 改 load_priority 顺序 / 删改其他 vendor 的 .so / LD_PRELOAD 隔离加载）：前两条要改 kernel 源码或模板约定，第四条影响别的算子，LD_PRELOAD 对 TTK 的 tiling 经 Python `op_tiling.py` 路径调用无效（实测）；只有 config.ini 是纯部署配置改动、可逆、立即生效，故作临时手段，治本仍选 static。
- 为什么只修自家不改 Softmax：只修改与当前任务直接相关的文件；且 static 自愈不依赖对方是否导出同名符号。

## 系统信息

- 工具链：CANN 9.2.0（`npu_op_library(cust_optiling TILING)` 产出 liboptiling.so），TTK kernel 模式（动态编译）与 aclnn C++ 测试触发 tiling。
- 相关文件：`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.{h,cpp}`（修复点）、`$ASCEND_OPP_PATH/vendors/config.ini`（load_order 载体，非修复依赖）、CANN `x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/scripts/install.sh`（load_priority 前插行为）、CANN `include/register/op_impl_registry.h`（`IMPL_OP_OPTILING` 展开为 `OpImplRegisterV2` 函数指针注册）。

## 预防

- 算子工程模板（arch35 tiling 胶水）默认给 `TilingFunc` 加 `static`，从源头避免多 vendor 共存踩雷；存量工程自查 `nm -D liboptiling.so | grep TilingFunc` 非零即中招。
- 共享开发机上的验收/复测若见 `INVALID_TILING`/561002 + plog 出现**别的算子工程路径**的 TilingFunc 报错，先查符号冲突，不要先怀疑自己的 tiling 逻辑。
- 多 vendor 共存机器上跑算子验证前，可 `cat $ASCEND_OPP_PATH/vendors/config.ini` 了解当前加载顺序；但不要依赖手工改 `load_priority` 做修复——它会被下一次任何 vendor 安装重写。
- 排查时优先 grep plog/TTK 输出里**别的算子名**的 tiling 源文件路径——这是符号被接管的直接指纹。

## 验证信号

问题与验证来自同一共享 CANN 安装上的 AdamW 白盒 TTK 全量执行、aclnn C++ 测试与符号表实测（Ascend950PR_9579，CANN 9.2.0），均为工具输出。

- 修复前：全量 32 例 2 PASS / 30 FAIL，`dyn_precision=INVALID_TILING` 30 例；plog 显示 AdamW 的 tiling 调进 `bench_harness_softmax_tiling_arch35.cpp:203` 的 TilingFunc 并报 `OpName:[BenchHarnessSoftmax] dim out of range`、`Op BenchHarnessApplyAdamW tiling failed`；aclnn 侧所有用例 561002 `Failed to execute tiling function`（errno 1228972098）（来源平台：950；证据等级：tool_observed）。
- config.ini workaround 失效证据：把 `load_priority` 改为 AdamW 在前后，下次检查又回到 `BenchHarnessSoftmax,BenchHarnessApplyAdamW,...`（并发任务重置，两次复现）； workaround 生效期间此前 `INVALID_TILING` 的用例一度变为 `STATUS: PASS`，全量 32 例恢复 19 PASS（剩余为真实精度失败，无任何 tiling/编译失败）（来源平台：950；证据等级：tool_observed）。
- 修复后符号表：AdamW liboptiling.so `nm -D | grep -c TilingFunc` = 0，Softmax 的仍导出 `T optiling::TilingFunc(gert::TilingContext*)`（预期，不修他人文件）；AdamW 的 `OpImplRegisterV2::TilingParse<...BenchHarnessApplyAdamW>` 等注册符号仍在（来源平台：950；证据等级：tool_observed）。
- 修复后功能回归：白盒全量恢复 29 PASS / 3 FAIL（无 INVALID_TILING，且此时 config.ini 仍是 Softmax 在前——证明 static 不依赖加载顺序）；TilingUT 37/37 PASS；aclnn ST 41/41 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- static 只保护"自家符号不被别人抢注/不去抢注别人"；若两个算子**都需要**被外部按名调用 TilingFunc（无此已知场景），static 不适用。
- 冲突的对称性：Softmax 自己若也被更早加载的第三方抢注，Softmax 仍会挂——本修复只保证 AdamW 自愈，不解决环境里其他算子间的同类冲突。
- 同名导出符号冲突的更广家族（如 `optiling::TilingParse`、infershape 同名函数）未逐一排查；推断同理可用内部链接消除（未验证，推断可迁移）。
- CANN 后续版本若把 tiling 注册改为按名 dlsym，static 方案需重新评估（本版本注册走函数指针，安全）。
- **config.ini 会被环境进程改写**：观测到文件 mtime 落在 TTK 运行期间、内容被改回 Softmax 优先——写入者未定位。因此 workaround 修复不持久。
- 本卡只覆盖"tiling 被别的算子函数接管"这一根因；同一次会话里叠加出现的 para_check E80003（attr 类型）是另一独立根因，需分别修。

**未决项**：

- 【优先】定位改写 `config.ini` 的确切进程/机制（当时已 grep TTK `_env.py`、`asc_op_compile_base/op_tiling.py`、`set_env.sh` 均未命中写者），并找出持久化的加载顺序管理方式（如 vendor 安装时自动 append 并固定顺序）。
- 环境里其余 vendor（AddCustom、RotaryPosEmb 等）的 liboptiling.so 是否也导出同名 TilingFunc 未逐一审计（`nm -D` 扫一遍 vendors/ 即可，属低优先接续项）。
- 评估构建侧根治方案（每个 vendor 的 TilingFunc 唯一符号化或局部绑定链接选项）的可行性；CANN 侧是否计划在 install.sh/加载器层面治理同名符号无公开信息，未追。
## 坏实践

- 手工改 `$ASCEND_OPP_PATH/vendors/config.ini` 的 `load_priority` 当"修复"：环境共享状态，任何并发 vendor 安装都会重写；实测两次被改回，浪费整轮全量执行。
- 见到 `INVALID_TILING`/561002 就重装 CANN/重编自己的 tiling 逻辑：符号冲突时自家代码完全正确，重编不改变符号解析结果。
- 用 `LD_PRELOAD` 自家 liboptiling.so 强改符号解析：TTK 的 tiling 经 Python `op_tiling.py` 路径调用，preload 不覆盖该路径（实测无效）。
- 测完后不恢复 `config.ini`、不删备份，会把机器的加载顺序永久改掉，坑后续会话。
- 只看 aclnn 的 561002 就断定是 kernel 回归、开始 checkout 旧版 kernel 做 bisect——本例实际用旧 kernel 也能跑过部分用例，白白消耗一轮排查（见误诊）。

## 误诊

- 曾以为 30 例 INVALID_TILING 是自己 tiling 代码回归（改动过 TilingData 结构）——单例 PASS、全量 FAIL 的不一致性 + plog 里的 Softmax 工程路径才是符号冲突的指纹（该判断已被 plog 与 nm 实测证实）。
- 曾怀疑 `rm -rf $TTK/kernel_meta/*`（清动态编译缓存）引发了问题——清缓存只是伴随操作，冲突与缓存无关（该怀疑被后续复现排除）。
- 早期一次同现象事件中，在"只改了 kernel、host 未动"的前提下 checkout 上一版 kernel 重测（full-load 21/21 PASS、split-load 16/20 PASS），恢复新版后对算子包与测试目录做全清重建又 41/41 PASS，于是归因为"陈旧内核缓存/瞬时设备状态"（归因未验证机制层）。后继才定位到 vendor 同名符号冲突；"全清重建后恢复"与本根因的因果关联是推断（可能是重装顺带改变了加载顺序），未直接复现验证。
