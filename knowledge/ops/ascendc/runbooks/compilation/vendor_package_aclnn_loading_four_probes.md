---
type: "Runbook"
title: vendor 包 aclnn 装载链路验证探针清单
description: build.sh 安装后用 nm/ldd/objdump/find 四探针验证 aclnn 装载链路：符号对导出、链接指向安装目录、入口引用计数、.o 数==dtype 组合数；kernel 二进制在 snake_case 目录，按 PascalCase 找 .o 必失败。
tags: [task.debugging, topic.implementation.operator_registration, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# vendor 包 aclnn 装载链路验证探针清单

## 现象

在 ops-registry-invoke 算子仓流程中，自定义算子包（本例 ThnnFusedGruCellGrad，Ascend 950 目标）经 `operators/build.sh <package>` 构建并安装为 vendor 包（`$ASCEND_OPP_PATH/vendors/<算子名>/`），aclnn 集成测试可执行文件也已链接生成。此时需要确认两件事：**测试程序运行时装载的确实是本包刚安装的 aclnn 实现（而不是环境里残留的同名旧库），且包内 dtype 注册齐全**。直接跑冒烟能证明调用链通，但无法区分链接到的是新库还是旧库（旧库同名符号同样能跑通），也看不出是否有 dtype 组合漏注册/漏编译。本轨迹在该算子包上执行了四组探针，全部得到预期观测。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发——vendor 包安装后的 aclnn 装载链路验收/排障（动态链接排查、安装完整性核对）；不适用于 kernel 直调流程。
- **API 路径**：vendor 包 aclnn 动态库 `$ASCEND_OPP_PATH/vendors/<Op>/op_api/lib/libcust_opapi.so`（两段式符号对 `aclnn<Op>` / `aclnn<Op>GetWorkspaceSize`）；探针为 binutils（`nm -D`/`ldd`/`objdump -T`）+ `find`。
- **验证过的平台与 CANN 版本**：Ascend 950 系列（平台注册值 `950`）+ CANN 9.0.0-beta.2（`ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp`，与 frontmatter `platforms` 严格一致）；布局约定与扩展门禁另在 CANN 9.1.0（构建头）+ 9.2.0（运行时/安装）双安装环境实测。
- **适用粒度**：`operator_family`——vendor 包目录布局（`vendors/<Op>/op_api/{include,lib}`）与 aclnn 符号命名约定对全部自定义算子包一致；`<Op>` 按算子名替换。
- **路线优先级**：`recommended_fastpath`（vendor 包安装后/冒烟结果可疑时的标准探针组；非 fallback）。
- **明确不适用/未验证场景**：tiling 动态库（`cust_optiling` 目标）的符号/装载验证（本探针组未覆盖）；GEIR/图模式（`op_proto`/`cust_op_proto`）装载检查；其他 CANN 版本的安装目录布局。

## 判别方法

**触发条件**：vendor 包构建安装完成后、冒烟运行前后，需要证明"测试链路绑定的是本包产物且 dtype 注册齐全"；或冒烟结果可疑（行为与预期不符、怀疑链接到旧库）时。

**探针 1 — 符号对导出**（库侧）：

```bash
nm -D $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so | grep -E 'aclnnThnnFusedGruCellGrad'
```

预期：**两个 `T`（导出）符号成对出现**——`aclnn<Op>` 与 `aclnn<Op>GetWorkspaceSize`。aclnn 公共 API 固定成对导出（执行入口 + workspace 查询入口）；只出现一个或全缺 → 安装不完整/构建异常。

**探针 2 — 动态链接指向**（可执行文件侧，防误链旧库）：

```bash
ldd operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/test_aclnn | grep 'libcust_opapi.so'
```

预期：`libcust_opapi.so` 解析到 **本包安装目录** `.../opp/vendors/<Op>/op_api/lib/libcust_opapi.so`（run.sh 已把该目录放进 `LD_LIBRARY_PATH`）。指向别处（系统目录/旧安装路径）→ 测的是旧库，须修 `LD_LIBRARY_PATH` 或重装后再验。

**探针 3 — 入口引用计数**（可执行文件确实经 aclnn 入口调用）：

```bash
objdump -T operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/test_aclnn | grep -c 'aclnnThnnFusedGruCellGradGetWorkspaceSize'
```

预期：`>= 1`（测试二进制存在对 `aclnn<Op>GetWorkspaceSize` 的未定义符号引用）。

**探针 4 — dtype 注册齐全**（安装目录 kernel 目标计数）：

```bash
find $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad -name '*.o' 2>/dev/null | wc -l
```

预期：**== 原型文档 dtype 组合表行数**。机制：AscendC 按 dtype 组合逐个实例化 kernel 目标，本包 proto §2 表为 3 行（float32/float16/bfloat16，全 IO 同 dtype），故安装目录 `.o` 计数应为 `3 == 3`；少一个即有 dtype 组合未注册/未编译进包。

**布局约定（探针 4 的路径前提，CANN 9.x vendors 安装实测）**：vendor 安装根目录用 PascalCase OpType，kernel 二进制子目录用 snake_case **kernel 入口符号名**——完整路径模式：

```text
$ASCEND_OPP_PATH/vendors/<OpType>/                          ← PascalCase OpType
  op_api/lib/libcust_opapi.so                               ← 导出 PascalCase 符号 aclnn<OpType> 与 aclnn<OpType>GetWorkspaceSize
  op_api/include/aclnn_<snake_case>.h                       ← 头文件名 snake_case
  op_impl/ai_core/tbe/kernel/<soc>/<snake_case>/<OpType>_<hash>.o   ← 二进制在 snake_case 目录
  op_impl/ai_core/tbe/kernel/config/<soc>/binary_info_config.json   ← binPath 同为 snake_case 相对路径
```

按 PascalCase 路径找 `.o` 会直接扑空（`ls .../ascend950/<PascalCase>/*.o` → No such file or directory，exit 2）；验收路径改用 snake_case（kernel 入口符号名 = snake_case(OpType)）即可命中。写涉及 vendors 安装路径的脚本/验收标准前，先 `ls $ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/<soc>/` 确认实际目录名，或直接用 binary_info_config.json 的 `binPath` 字段取相对路径。

**扩展门禁（与四探针配套，可复用命令集）**：

```bash
# ① 构建（退出码 0）
( set -o pipefail; cd operators && ./build.sh <Pkg>_package 2>&1 | tee /tmp/<Op>_build.log ); \
  test ${PIPESTATUS[0]:-$?} -eq 0
# ② tiling 注册检查：REGISTER_TILING_DEFAULT 缺失时 build 仍 exit 0 但打此警告（运行期 361001 风险）
test -f /tmp/<Op>_build.log && ! grep -qi 'do not registe tiling struct' /tmp/<Op>_build.log
# ③ ACLNN 冒烟（需设备）：build → cmake/make test_aclnn → 执行，两段式 aclnn 调用
cd <Pkg>_package/tests/aclnn && ./run.sh        # 期望 N/N PASSED（N=dtype 组合数）
# ④⑤ 即探针 1 与探针 4（注意 ⑤ 的路径是 snake_case 目录）
```

门禁②存在的原因：tiling 注册缺失是"build 过、运行挂"的静默失败模式——构建退出码 0 完全不能证明 tiling 已注册，必须 grep 日志。门禁④⑤分离的原因：④验证 host 侧 aclnn API 面（libcust_opapi.so 导出），⑤验证 device 侧二进制面（按 dtype 组合数计数）；两者失败模式独立（缺 tiling 注册时⑤能暴露、链接配置错时④能暴露）。

**解读**：四探针齐过 = 测试程序装载的确实是本包 aclnn 实现且 dtype 注册齐全；任一不符按对应探针定位（缺符号→重装/查构建日志；链错→查 `LD_LIBRARY_PATH`；计数不符→查 OpDef dtype 注册与构建产物）。

## 设计决策

- **为何用 ldd 而非只看运行结果**：运行通过不能排除链接旧库——旧库同名符号同样能跑通；`ldd` 直接给出动态链接器的实际解析路径，是"测的是哪个库"的一手证据。
- **为何查符号对而非单符号**：aclnn 两段式 API 固定成对导出；只导出执行入口或缺 `GetWorkspaceSize` 都说明安装/构建异常，单看一个符号会漏判。
- **为何用 `.o` 计数对 dtype 组合数**：按 dtype 组合实例化 kernel 是 AscendC 构建机制，计数是最快的完整性判据（3 组合 → 3 个 `.o`），无需逐个反查内容。
- **未采用的备选**：直接重装重跑（不定位装载因果，链错时复跑仍可能链错）、逐文件比对安装目录（慢且无判据）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp`）、GNU binutils（nm/objdump/ldd）；构建：CMake + GNU 13.3.0，`build.sh` 产出 `cust_optiling`/`cust_opapi`/`cust_op_proto`/`ascendc_kernels` 目标并安装到 vendors 目录。
- 目标平台：ascend950（构建产物 `ai_core/ascend950/aic-ascend950-ops-info.json`）。
- 相关路径：vendor 安装根 `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/`（含 `op_api/include`、`op_api/lib`）；测试可执行文件 `operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/test_aclnn`；一键脚本 `tests/aclnn/run.sh`（自动设置 `LD_LIBRARY_PATH` 指向安装库目录）。

## 完整代码

四探针命令（按本算子包原样执行；迁移时把算子名/包路径替换为目标算子）与实测输出：

```bash
# 探针 1：符号对导出（预期两个 T 符号）
nm -D $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so | grep -E 'aclnnThnnFusedGruCellGrad'
## 实测输出：
## 0000000000003039 T aclnnThnnFusedGruCellGrad
## 0000000000002809 T aclnnThnnFusedGruCellGradGetWorkspaceSize

## 探针 2：动态链接指向（预期解析到本包安装目录）
ldd operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/test_aclnn | grep 'libcust_opapi.so'
## 实测输出：
##   libcust_opapi.so => /home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so (0x00007f695bdb1000)

## 探针 3：入口引用计数（预期 >= 1）
objdump -T operators/ThnnFusedGruCellGrad_package/tests/aclnn/build/test_aclnn | grep -c 'aclnnThnnFusedGruCellGradGetWorkspaceSize'
## 实测输出：
## 1

## 探针 4：安装目录 kernel .o 计数（预期 == proto dtype 组合表行数，本包 3）
find $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad -name '*.o' 2>/dev/null | wc -l
## 实测输出：
## 3
## （同批辅助测量：算子包目录树 .o=29、其中 build 目录 .o=28——含包内构建中间产物，不作判据）
```

## 预防

- vendor 包每次 `build.sh` 重装后，跑冒烟前先过探针 1/2——尤其多算子并行开发、环境里存在多个同名 `libcust_opapi.so` 时。
- 冒烟结果与预期不符时，**先 ldd 再重装**：确认当前实际链接目标，再决定修复动作。
- 验收 dtype 覆盖时用探针 4 的计数判据，不要靠肉眼翻安装目录。
- 写涉及 vendors 安装路径的脚本/验收标准时，先确认实际目录名再写死路径（布局约定见判别方法）；dtype 组合数 N 按 OpDef 的 dtype 定义推算，不要硬编码。

## 验证信号

以下信号来自 ThnnFusedGruCellGrad 算子包接口桩阶段的开发会话：测试用例收敛修复后，对刚安装的 vendor 包与新生成的测试可执行文件执行上述四探针，随后又重装一次（验收口径构建命令）并复跑端到端：

- `nm -D` 输出两个 T 符号：`aclnnThnnFusedGruCellGrad`、`aclnnThnnFusedGruCellGradGetWorkspaceSize`（来源平台：950；证据等级：tool_observed）
- `ldd` 将 `libcust_opapi.so` 解析到 `/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_api/lib/`（来源平台：950；证据等级：tool_observed）
- `objdump -T` 对 `aclnnThnnFusedGruCellGradGetWorkspaceSize` 引用计数 == 1（≥1 判据通过）（来源平台：950；证据等级：tool_observed）
- 安装目录 `.o` 计数 == 3 == proto dtype 组合表行数（float32/float16/bfloat16）（来源平台：950；证据等级：tool_observed）

以下信号来自另一次 950 目标自定义算子（BenchHarnessApplyAdamW，OpDef 桩阶段）验收会话（CANN 9.1.0 构建头 + 9.2.0 运行时双安装环境），补充布局约定与扩展门禁：

- 门禁①②：`build.sh` 退出码 0，生成 3 个 kernel（`[ascend950] Generating <OpType>_<hash> ... Done` ×3）；构建日志无 `do not registe tiling struct`（GATE2 PASS）（来源平台：950；证据等级：tool_observed）
- 门禁③：`run.sh` 冒烟 3/3 PASSED（fp32/fp16/bf16 两段式 aclnn 调用 OK，需设备）（来源平台：950；证据等级：tool_observed）
- 门禁④：`nm -D` 命中 `aclnnBenchHarnessApplyAdamW` 与 `aclnnBenchHarnessApplyAdamWGetWorkspaceSize` 两个符号（来源平台：950；证据等级：tool_observed）
- 门禁⑤：PascalCase 路径 `ls` 失败（exit 2）；snake_case 路径命中 3 个 .o（count=3）；AddCustom 参考模板同为 snake_case 目录；binary_info_config.json 的 binPath 为 `ascend950/bench_harness_apply_adam_w/...`（来源平台：950；证据等级：tool_observed）

## 失效边界

- 探针组只覆盖 **op_api 装载链路与 kernel `.o` 计数**；tiling 库（`cust_optiling`）符号/装载与 GEIR 图模式（`cust_op_proto`）未覆盖（未验证，如需可按同法对相应库文件执行 nm/ldd）。
- `.o` 计数 == dtype 组合行数的对应关系在本包实测（每个 dtype 组合一个 kernel 目标）；若包结构不同（多 arch 目录/共享目标），须按实际构建产物调整计数口径。
- `nm` 输出中的地址（如 `0x3039`）是本次安装的瞬时链接布局值，**只看符号存在性与 T 类型，地址不可作判据**。
- `ldd` 反映的是执行探针时的环境（`LD_LIBRARY_PATH`）；在改过环境后须重跑，不能沿用旧输出。
- 布局证据来自 CANN 9.x（9.0.0-beta.2/9.1.0/9.2.0）vendors 安装；其他大版本安装布局变化未验证。双 CANN 安装（构建工具 9.1.0 + 运行时 9.2.0）是环境事实，其版本交互影响未单独验证；单一安装环境下路径推导相同。
- 冒烟门禁③依赖设备可用；无设备时只能执行①②④⑤。

## 坏实践

- **只跑冒烟不查 ldd**：链接到残留旧库时冒烟照样通过，"通过"结论对刚构建的包失真。
- **把 nm 输出地址当对错判据**：地址随安装变化，无判别力。
- **结果可疑就盲目重装重跑**：不先定位装载因果（探针 1/2），链错根因仍会复现。
- 按验收标准文本里的 PascalCase 写法去找 kernel `.o`——以实装布局（snake_case 目录）为准，AddCustom 模板对照证实为通用约定而非个例。

## 误诊

- "冒烟通过 = 链接的就是新装的库"：误。同名符号的旧库同样能跑通；`ldd` 的解析路径才是直接证据（本轨迹中以探针 2 显式核对）。

## 未决事项

- tiling 动态库与 GEIR 图模式装载验证未纳入本探针组（未验证，可作扩展）。
- "包目录树 .o=29 / build 目录 .o=28" 的精确构成（哪些属测试构建中间产物）未逐项核对，不影响本卡判据（判据只用安装目录计数）。
