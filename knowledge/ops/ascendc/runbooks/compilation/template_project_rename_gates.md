---
type: "Runbook"
title: 复制模板工程改名的四形态映射与五项门禁
description: 自定义算子冷启动配方：cp 模板工程后按 PascalCase/snake/大写三套标识符 + kernel 入口 snake_case(OpType) + aclnn API 名全量改名，再用 5 项门禁（build/警告/残留/so 符号/.o 数）验收。
tags: [task.implementation, topic.implementation.operator_registration, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-17T05:31:07Z'
platforms: ["950"]
---

# 复制模板工程改名的四形态映射与五项门禁

## 现象

在从零搭建一个新自定义算子（softplus_v2_grad）的 ACLNN 工程时，常规做法是把现成的 AddCustom 示例工程整包复制过来再改名。复制本身一条 `cp -r` 就完成，但模板里的算子名以**四种大小写形态**散布在文件名、CMake、host/kernel 源码、autogen 头文件引用、测试脚本里；漏改任何一处，轻则 `grep` 残留、重则 build 找不到源文件或 test 链接失败。本轨迹完整走通了一次复制-改名-验收：复制 `operators/add_custom_package` → `operators/softplus_v2_grad_package`，改名后 build 退出 0、`test_aclnn` 28/28 PASSED、`grep -riE 'add_?custom'` 残留为 0、`libcust_opapi.so` 导出 `aclnnSoftplusV2Grad`/`aclnnSoftplusV2GradGetWorkspaceSize` 两个符号、kernel `.o` 恰好 2 个（= proto 里声明的 dtype 组合数）。

## 适用场景

- 工程形态：自定义算子工程冷启动（operators/<pkg>/，`npu_op_package` + `npu_op_code_gen` + `npu_op_library` + `npu_op_kernel_sources` CMake 骨架 + `tests/aclnn` 冒烟测试）。
- API 路径：`OP_ADD(<OpType>)` 注册、autogen `aclnn_<snake>.h`、两级 aclnn 调用 `aclnn<OpType>GetWorkspaceSize` / `aclnn<OpType>`、`nm -D libcust_opapi.so` 符号检查。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.1.0。
- 适用粒度：`workflow_only` 中的工程搭建配方，但产物规则（命名派生关系）是 `platform_wide` 事实。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不从模板复制、用 `msopgen` 直接生成工程的路径（本轨迹未用 msopgen 生成，仅检查了其存在性）；已有多算子共享一个 package 的拆分场景。

## 解决配方

### 1. 复制与文件改名

```bash
cp -r operators/add_custom_package operators/softplus_v2_grad_package
# Rename source files
mv operators/softplus_v2_grad_package/op_host/add_custom_host.cpp operators/softplus_v2_grad_package/op_host/softplus_v2_grad_host.cpp
mv operators/softplus_v2_grad_package/op_kernel/add_custom_kernel.cpp operators/softplus_v2_grad_package/op_kernel/softplus_v2_grad_kernel.cpp
mv operators/softplus_v2_grad_package/op_kernel/add_custom_tiling.h operators/softplus_v2_grad_package/op_kernel/softplus_v2_grad_tiling.h
```

### 2. 标识符映射（四种形态 + 两个派生规则）

| 旧（AddCustom） | 新（SoftplusV2Grad） | 出现位置 |
|---|---|---|
| `AddCustom` | `SoftplusV2Grad` | OpDef 类名、`OP_ADD(...)`、CMake `set(package_name ...)`、`npu_op_kernel_sources(OP_TYPE ...)`、tests CMake 的 vendors 路径 |
| `add_custom` | `softplus_v2_grad` | 文件名、`#include "../op_kernel/<snake>_tiling.h"`、autogen `aclnn_<snake>.h`、kernel 目录名 |
| `ADD_CUSTOM` | `SOFTPLUS_V2_GRAD` | include guard 等大写场景 |
| `addCustom` | `softplusV2Grad` | 小驼峰场景（如出现） |
| `add`（测试函数名） | `softplus_v2_grad` | test_aclnn.cpp |

两个派生规则（必须遵守，否则符号对不上）：

- **kernel 入口函数名 = snake_case(OpType)**：`SoftplusV2Grad` → `extern "C" __global__ __aicore__ void softplus_v2_grad(...)`。host 侧 autogen 按这个名字查找 kernel。
- **aclnn API 名 = "aclnn" + OpType（PascalCase 原样拼接）**：`aclnnSoftplusV2Grad` / `aclnnSoftplusV2GradGetWorkspaceSize`，头文件 `aclnn_softplus_v2_grad.h`（snake）。

### 3. 必改文件清单（与事件流实际读写一一对应）

- `CMakeLists.txt`：`package_name`、`host_ops_srcs` glob 的 `op_host/<snake>_host.cpp`、`npu_op_kernel_sources` 的 `OP_TYPE <Pascal>` / `KERNEL_FILE <snake>_kernel.cpp`；
- `op_host/<snake>_host.cpp`：类名、`OP_ADD`、tiling 头 include 路径；
- `op_kernel/<snake>_kernel.cpp` + `<snake>_tiling.h`：kernel 入口函数名、tiling 结构体名、include guard；
- `tests/aclnn/CMakeLists.txt`：`project(test_aclnn_<snake>)` 与 vendors 头文件/库搜索路径里的 `<Pascal>`；
- `tests/aclnn/run.sh`：包名参数；
- `tests/aclnn/test_aclnn.cpp`：`#include "aclnn_<snake>.h"`、两级 aclnn 调用名、测试函数名。

### 4. 验收门禁（五项，全部退出码 0 才算过）

```bash
# ① 抓日志编译（pipefail 下取真实退出码）
( set -o pipefail; cd operators && ./build.sh softplus_v2_grad_package 2>&1 | tee /tmp/<snake>_build.log ); echo "exit=${PIPESTATUS[0]:-$?}"
# ② 日志无 tiling 注册告警（告警存在说明 tiling 结构体没被 kernel 正确注册）
! grep -qi 'do not registe tiling struct' /tmp/<snake>_build.log
# ③ 无旧算子名残留（1=clean）
grep -riE 'add_?custom' operators/<snake>_package/ ; echo "exit=$?"
# ④ so 符号（两级 aclnn API 都要导出）
nm -D $ASCEND_OPP_PATH/vendors/<Pascal>/op_api/lib/libcust_opapi.so | grep -iE 'aclnn<Pascal>'
# ⑤ kernel .o 数量 = proto 声明的 dtype 组合数
ls $ASCEND_OPP_PATH/vendors/<Pascal>/op_impl/ai_core/tbe/kernel/ascend950/<snake>/*.o
```

再跑 `tests/aclnn/run.sh` 真机冒烟（模板算法 x+y 未删时 14 shapes × {fp16,fp32} = 28/28 PASSED）。

## 设计决策

- 为什么整包复制而不是用 `msopgen` 重新生成：模板工程已带好 CMake/CPack/run.sh 骨架与已验证的构建链路，复制 + 全量改名把冷启动风险收缩为"机械替换 + 门禁验证"；本环境虽有 msopgen，但模板路线一步到位。
- 改名策略是**逐文件手工 edit** 而非全局 sed：文件数量少（6 个文本文件），手工 edit 可逐处核对四种大小写形态，避免误伤；代价是多花几轮编辑。
- 验收门禁里把"`.o` 数量 = dtype 组合数"独立成项：dtype 组合在 OpDef `DataType({...})` 里声明（本例 fp16/fp32 两种），每种组合生成一个 kernel 二进制（文件名 `<Pascal>_<hash>.o`），数量对不上说明 OpDef dtype 声明与预期不符。
- 当前路线属 recommended_fastpath。

## 系统信息

- CANN 9.1.0（`ASCEND_HOME_PATH`/`ASCEND_OPP_PATH` 指向 `~/Ascend/cann-9.1.0`），GNU 13.3.0，`ASCEND_COMPUTE_UNIT ascend950`。
- 安装产物布局（改名正确性的旁证）：`$ASCEND_OPP_PATH/vendors/SoftplusV2Grad/op_api/include/aclnn_softplus_v2_grad.h`、`op_api/lib/libcust_opapi.so`、`op_impl/ai_core/tbe/kernel/ascend950/softplus_v2_grad/SoftplusV2Grad_<hash>.o`、`op_impl/ai_core/tbe/SoftplusV2Grad_impl/dynamic/op_kernel/softplus_v2_grad_kernel.cpp`、`op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so`。
- 运行测试前需 `export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/<Pascal>/op_api/lib/:$LD_LIBRARY_PATH`（安装器输出里给出的要求）。

## 完整代码

见"解决配方"四小节——复制/改名命令、映射表、必改文件清单与五项门禁命令均为 verbatim（占位符 `<snake>`/`<Pascal>` 为本卡对源算子名的参数化，替换规则即映射表）。

## 预防

- 复制后先做门禁 ③（残留 grep）再 build——残留 grep 是最快的漏改指示器。
- 改名时对照映射表逐形态过一遍，特别注意 tests/aclnn/CMakeLists.txt 里 vendors 路径用的是 PascalCase，而 include 的头文件是 snake_case。

## 验证信号

本配方在 softplus_v2_grad 工程冷启动时完整执行并验证。实测信号：

- `./build.sh softplus_v2_grad_package` 退出码 0，`[ascend950] Generating SoftplusV2Grad_<hash> ... Done`，CPack 打包 + 安装 SUCCESS（来源平台：950；证据等级：tool_observed）。
- 构建日志无 `do not registe tiling struct` 告警（来源平台：950；证据等级：tool_observed）。
- `grep -riE 'add_?custom' operators/softplus_v2_grad_package/` 无匹配（exit=1，clean）（来源平台：950；证据等级：tool_observed）。
- `nm -D .../libcust_opapi.so`：`00000000000018b6 T aclnnSoftplusV2Grad`、`00000000000013b9 T aclnnSoftplusV2GradGetWorkspaceSize`（来源平台：950；证据等级：tool_observed）。
- kernel `.o` 数量 2，路径 `.../kernel/ascend950/softplus_v2_grad/SoftplusV2Grad_<hash>.o`，与 proto 的 2 个 dtype 组合一致（来源平台：950；证据等级：tool_observed）。
- `run.sh` 真机冒烟：`==== Shape generalization summary: 28/28 cases PASSED (14 shapes x {fp16, fp32}) ====`，exit=0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 模板算法（x+y）在本阶段**刻意保留**——此配方的验收信号（28/28 数值通过）依赖模板算法还在；删除算法换成接口桩后，冒烟语义变为"两级 aclnn 调用返回 OK"（无数值比对），门禁 ①②④⑤ 仍然适用。
- "`.o` 数 = dtype 组合数"的对应关系以 OpDef `DataType({...})` 声明为准；若后续引入 per-dtype tilingKey 多 Kernel，`.o` 数会按 (dtype × tilingKey) 组合增长，不能简单画等号。
- 未验证，推断可迁移：其他架构家族（替换 `ascend950` 目录段）与 CANN 版本。
- `do not registe tiling struct` 告警的触发条件（何种错误会打这条告警）未在本轨迹复现，本卡只把它当作门禁用的阴性信号。

**未决项**：

- `do not registe tiling struct` 告警的机制层成因未追（何类 tiling 注册缺失会触发），当前仅作门禁阴性信号使用。

#### 并入自 template_package_derive_new_operator_cold_start（BenchHarnessSoftmax 形态：sed 批量替换与 11 文件结构）

同配方的另一独立落地（BenchHarnessSoftmax，softmax 类单输入单输出 fp16/fp32/bf16，ascend950）：模板包含 `op_graph/` 与 `arch35/` 子目录时共 11 个源文件需 mv，替换策略改用**大小写敏感 sed 批量替换**（与本卡"逐文件手工 edit"互补：文件多时 sed 更稳；替换精确、验收从严——检查仍用大小写不敏感 grep 兜底）：

```bash
# 重命名全部源文件（11 个）后，四种大小写批量替换（排除 build/）
find . -type f -not -path './build/*' \( -name '*.cpp' -o -name '*.h' -o -name '*.cc' \
  -o -name '*.txt' -o -name '*.sh' -o -name '*.md' \) -exec sed -i \
  -e 's/ADD_CUSTOM/BENCH_HARNESS_SOFTMAX/g' \
  -e 's/AddCustom/BenchHarnessSoftmax/g' \
  -e 's/add_custom/bench_harness_softmax/g' \
  -e 's/addCustom/benchHarnessSoftmax/g' {} + && echo "sed done"
```

- **11 文件清单**（含 op_graph 的模板结构）：`op_host/{def,infershape}.cpp`、`op_host/arch35/tiling_arch35.{cpp,h}`、`op_graph/{graph_infer.cpp,proto.cc,proto.h}`、`op_kernel/apt.cpp`、`op_kernel/arch35/{kernel.h,struct.h,tiling_struct.h}`；目录布局不同时 mv/sed 目标需相应调整。
- **两阶段脚手架**：脚手架阶段刻意保留模板的加法语义——先把工程链路（注册/编译/安装/符号导出/运行）变成可验证事实，接口与算法语义留给后续任务替换；语义替换后重跑同一套冒烟对比。
- **`.o` 路径陷阱**：安装目录段是 snake_case（`.../ascend950/<snake_case>/`）而 `.o` 文件名前缀是大驼峰（`<PascalCase>_<hash>.o`）——按大驼峰目录名去找会扑空；dtype 组合声明改成三组合后 `.o` 数相应为 3。
- **混合版本环境事实**：该工作区 CMake 的 `CANN_HOME` 硬编码指向 cann-9.1.0（编译用），而 `ASCEND_OPP_PATH` 指向 cann-9.2.0（安装与运行）——构建与冒烟均通过（ABI 兼容），但这是工作区特定状态，不是可推广结论。
- **坏实践补充**：只构建不冒烟——so 未导出、vendors 未更新、`LD_LIBRARY_PATH` 未设等链接期/安装期问题不会在构建退出码中暴露。

并入验证信号（BenchHarnessSoftmax 冷启动会话，与本卡同一套门禁）：

- `./build.sh BenchHarnessSoftmax_package` 退出码 0，日志含 `Compile op info cfg successfully`、CPack 打包并安装 SUCCESS；`do not registe tiling struct` 计数 0（来源平台：950；证据等级：tool_observed）。
- `nm -D libcust_opapi.so` 导出 `T aclnnBenchHarnessSoftmax` 与 `T aclnnBenchHarnessSoftmaxGetWorkspaceSize`（来源平台：950；证据等级：tool_observed）。
- `ascend950/bench_harness_softmax/` 下 `.o` 共 2 个，等于 OpDef 声明的 fp16+fp32 两组 dtype 组合（来源平台：950；证据等级：tool_observed）。
- 冒烟 `28/28 cases PASSED (14 shapes x {fp16, fp32})`，退出码 0，每例与 CPU 参考比对通过（含 1 元素与百万级元素边界规模）；`grep -riE 'add_?custom'` 整包退出码 1（无匹配）（来源平台：950；证据等级：tool_observed）。

并入边界：

- 命令假定模板包目录约定（op_host/op_kernel/op_graph/tests 与外层 build.sh）；不同布局需调整 mv/sed 目标与 find 过滤条件。
- 平台/版本边界：ascend950 + CANN 9.1.0（编译）/9.2.0（安装与运行）组合实测；其他版本未验证，推断可迁移。
## 坏实践

- 只替换 `add_custom` 一种形态就开 build——`AddCustom`（PascalCase，CMake `package_name`/`OP_TYPE`/vendors 路径）与 `ADD_CUSTOM`（guard）必然残留，build 或链接阶段失败。
- 用宽松的 `grep -i add` 检查残留——门禁 ③ 的 `add_?custom` 模式同时覆盖 `add_custom` 与 `addcustom` 两种粘连形态。

## 误诊

- 编辑期间编辑器 LSP 对每个 host/kernel 文件报大量 "kernel_operator.h / op_def_registry.h not found" 类诊断——这是编辑器没有 CANN include 路径的固定噪音，不是改名引入的问题；以 `build.sh` 实际编译为准（本轨迹每次 build 均通过）。
