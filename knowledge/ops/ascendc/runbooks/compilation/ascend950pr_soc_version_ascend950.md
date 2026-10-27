---
type: "Runbook"
title: Ascend950PR 的 soc_version 确认为 ascend950
description: "npu-smi 板卡信息给 Product Name/NPU Name/Chip Name 三套命名；经已装 vendor 算子包 kernel 目录确认 Ascend950PR → soc_version=ascend950（arch35）。R2（检视视角）：设计文档产品名 Ascend950PR 与代码芯片配置名 ascend950 表面不一致是命名口径差异非缺陷；CMake/build.sh/AddConfig 三处一致用 ascend950，该命名构建的算子包在 Ascend950PR 上板全量精度验证通过。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# Ascend950PR 的 soc_version 确认为 ascend950

## 现象

在 Ascend950PR 机器上首次构建自定义算子时需要填对两处 soc 名：`cmake -DASCEND_COMPUTE_UNIT=<soc>` 与 OpDef 的 `this->AICore().AddConfig("<soc>", ...)`。但设备侧命名对不齐：`npu-smi info` 顶部显示 `Ascend950PR`；`npu-smi info -t board -i 0` 给出 `Product Name: A310-50-C00MM304A1`、`NPU Name: 9579`、`Chip Name: Ascend950PR`——三套命名没有一个直接等于构建工具要的 soc_version 字符串，且 CANN API 文档的产品支持表里也没列这款芯片，直接照抄设备名存在填错风险。

## 适用场景

- 工程形态：算子仓开发（新机器/新芯片上首次配置自定义算子构建；OpDef AddConfig 与 CMake ASCEND_COMPUTE_UNIT 取值确认）。
- API 路径：`npu-smi info` / `npu-smi info -t board -i 0`；`$ASCEND_HOME_PATH/opp/vendors/<pkg>/op_impl/ai_core/tbe/kernel/<soc>/` 目录结构。
- 验证过的平台与 CANN 版本：Ascend950PR（本卡结论即该芯片的映射）+ CANN 9.1.0；构建配置级实测（与 frontmatter `platforms` 一致）。
- 适用粒度：exact_task 级映射结论 + workflow 级确认方法（方法本身可推广到其他芯片，具体映射值不可）。
- 路线优先级：recommended_fastpath（有已装包可对照时）；无已装包时退回查 CANN 支持列表（未验证路径）。
- 明确不适用/未验证场景：其他芯片（Ascend910B/310P 等）的映射值不能由本卡外推；`NPU Name 9579` 与 tikicpulib 目录里 `Ascend950PR_9599` 的命名差异未深究（不影响 soc_version 取值）。

## 判别方法（确认链）

1. **触发**：构建自定义算子需要 soc_version，设备名（Ascend950PR / A310-50-C00MM304A1 / 9579）没有一个是文档里的直接可用值。
2. **探针一**：`npu-smi info -t board -i 0` 拿三套命名的权威值——本机实测输出：

```text
NPU ID                         : 0
Product Name                   : A310-50-C00MM304A1
NPU Name                       : 9579
Chip Name                      : Ascend950PR
Chip Version                   : V100
```

3. **探针二**：查本机已安装的同类自定义算子 vendor 包，看真实构建产物落在哪个 soc 目录、源码用什么 arch 后缀（可执行命令）：

```bash
grep -r "AddConfig\|soc_version\|SOC" $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_*_custom/ 2>/dev/null | head
ls $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_exp_custom/op_impl/ai_core/tbe/kernel/
grep -r "src_file_dict\|__ALLSOC__" $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_exp_custom/op_impl/ai_core/tbe/bench_no_harness_exp_custom_impl/dynamic/*.py | head
```

4. **原始观测**：已装包 kernel 目录为 `.../tbe/kernel/ascend950/bench_no_harness_exp/`（存在 `BenchNoHarnessExp_<hash>.json` 等多个二进制清单），且其动态编译脚本声明 `src_file_dict = {'__ALLSOC__': './bench_no_harness_exp_arch35.cpp'}`——即 soc 目录名 `ascend950`、源文件 arch 后缀 `arch35`（DAV_3510）。
5. **解释**：Chip Name `Ascend950PR` 对应 soc_version 字符串 `ascend950`（小写）、架构 arch35/DAV_3510；`NPU Name 9579` 是板级命名，不必与 tikicpulib 的型号目录名（如 Ascend950PR_9599）对齐。构建取值用 `ascend950`。

R2 确认链（检视场景，TopK 类算子交付前代码检视）：设计文档（DESIGN.md）写目标芯片 `ascend950pr`、算子注册代码写 `ascend950`，表面不一致疑似命名错误时，用 rg 全量检索构建配置核对三处口径（在算子目录执行）：

```bash
rg -n "ascend950" CMakeLists.txt op_host/ op_kernel/ --no-heading
rg -n "ascend950pr|ascend950" build.sh CMakeLists.txt --no-heading
```

原始观测（本轨迹）：`CMakeLists.txt:21 set(ARCH35_COMPUTE_UNITS ascend950)`、`CMakeLists.txt:23 注释"默认编译 ascend950（Ascend950PR）"`、`op_host/<op>_def.cpp:67 this->AICore().AddConfig("ascend950", aicoreConfig950)`、`build.sh:14 SUPPORT_COMPUTE_UNITS=("ascend950")`——三处全部 `ascend950` 且 `ascend950pr` 零命中；op_kernel/op_host 的 CMakeLists 对 `ASCEND_COMPUTE_UNIT` 设白名单，非 `ascend950` 直接 `FATAL_ERROR "Unsupported ASCEND_COMPUTE_UNIT"` 终止配置。结论：产品名（Ascend950PR，arch35/DAV_3510 为其架构名）与芯片配置名/SoC 名（ascend950，无 PR 后缀）是两个口径，检视发现的"不一致"是口径差异而非缺陷。

## 验证信号

按上述结论实际构建该算子（CANN 9.1.0）：

- `cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` 配置通过（opbuild 生成 cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels 目标）（来源平台：950；证据等级：tool_observed）。
- 全量构建产出 `build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_cross_entropy_loss/` 下 9 个 kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 包，clean build 复验通过（来源平台：950；证据等级：tool_observed）。
- OpDef 侧 `this->AICore().AddConfig("ascend950", aicoreConfig950)` 与该取值一致编译通过（来源平台：950；证据等级：tool_observed）。

R2 增量信号（TopK 类算子交付检视 + 上板验证，Ascend950PR + CANN cann-9.1.0）：

- rg 检索确认 `set(ARCH35_COMPUTE_UNITS ascend950)`、`SUPPORT_COMPUTE_UNITS=("ascend950")`、`AddConfig("ascend950", aicoreConfig950)` 三处一致，`ascend950pr` 在构建配置中零命中；CMakeLists 注释明确"默认编译 ascend950（Ascend950PR）"（来源平台：950 开发环境；证据等级：tool_observed）。
- 以 `ascend950` 命名构建并安装的算子包，在 Ascend950PR NPU 上 PyTorch 接入全量精度测试：总计 736 用例、通过 577、失败 0、跳过 159（dim≠last 已知限制）、通过率 100%（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 方法依赖机器上已装有同 soc 的 vendor 算子包；全新机器无已装包时需改查 CANN 官方支持列表（该路径本卡未验证）。
- 映射值 `Ascend950PR → ascend950` 只在 CANN 9.1.0 验证；命名规范（soc 目录全小写、arch 数字后缀）是观察归纳，不保证其他芯片同构。
- `NPU Name`/`Product Name` 与 soc_version 之间没有推导关系，只能作旁证。
- R2：把产品名当芯片配置名写进构建配置（如 `ascend950pr`）会被白名单校验 `FATAL_ERROR "Unsupported ASCEND_COMPUTE_UNIT"` 拦截（本仓 op_kernel/op_host CMakeLists 实测行为）；工程未设白名单时会在更晚的编译/安装阶段失败。
- R2：本卡不覆盖 `_def.cpp` 中 AICore 之外的配置维度（如 AIUP/Cube 配置命名），未验证；该命名与 SoC 探测返回值（npu-smi / ACL 探测的 soc version 字符串）的精确对应关系未验证。

**未决项**：

- 无已装 vendor 包时的通用确认路径（CANN 支持列表/文档检索）未在本轨迹验证。
## 误诊

- 想 tikicpulib 目录名 `Ascend950PR_9599` 与 `NPU Name 9579` 必须匹配——两者是不同命名空间（板级型号 vs 编译库目录），对不上不影响 soc_version 取值（该判断经构建通过佐证）。
- R2：检视时见到 `_def.cpp` 注册 `ascend950` 与 DESIGN.md 目标芯片 `Ascend950PR` 不一致就直接当缺陷改代码——应先全局检索三处配置 + 核对注释/上板证据，确认是口径差异还是真实笔误（本轨迹经 rg 取证与注释核对后确认为口径差异，该假设已被推翻，记录于此防止重复误判）。

## 后续增量（2026-09-17，并卡合并：构建系统 SOC 名单与 opdesc_parser 映射机制）

#### 并入自 npu_smi_chip_name_vs_build_soc（ASC_VALID_SOC_LIST 全集、opdesc_parser 映射表与 CMake 固定写法）

- **构建系统合法 SOC 名单（`fwk_modules/config.cmake` 的 `ASC_VALID_SOC_LIST`，可直接 grep）**：`ascend310b ascend310p ascend610 ascend910 ascend910b ascend910_93 ascend950 bs9sx1a bs9sx2a ascend610lite mc61am21a mc62 ascend910_96 kirinx90 kirin9030 mc32dm11a ascend350`——名单收录的是**计算单元名**，不含 `ascend950pr` 等变体名。
- **变体→计算单元映射（`fwk_modules/util/opdesc_parser.py`，机制链直读）**：

```text
"ascend950pr_9599": "ascend950",
"ascend950pr_958a": "ascend950",
"ascend950pr_9589": "ascend950",
"ascend950pr_958b": "ascend950",
"ascend950pr_9579": "ascend950",
...（全部 ascend950pr_* → ascend950）
```

  即：npu-smi 的 Chip Name（Ascend950PR）→ 芯片型号（ascend950pr_9579，NPU Name 9579）→ 构建计算单元名（ascend950，对应 DAV_3510 / arch35 目录）；`ascend950pr_*` 等具体型号只是 `host_config.cmake` 里 `ascend950_list` 的子变体。
- **CMake 固定写法（顶层 CMakeLists.txt）**：

```cmake
# 架构配置
# arch35: ascend950 (DAV_3510)
set(ARCH35_COMPUTE_UNITS ascend950)

# 默认编译 ascend950
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH35_COMPUTE_UNITS})
endif()
```

  编译命令 `bash build.sh --soc=ascend950 -j8`，kernel 源码与 tiling 放在 `arch35/` 子目录（op_kernel/arch35、op_host/arch35）。
- **设计取舍（并入）**：不用完整型号 ascend950pr_9579（不在合法名单）；不用 DAV 架构名（构建系统不按此校验）。
- 坏实践补充（并入）：把 npu-smi 的 Chip Name（Ascend950PR）或任务描述里的 "ascend950pr" 直接当 `--soc` 参数；在没有探测映射表的情况下猜测 SOC 名并反复试错编译。
- 失效边界补充（并入）：结论基于 CANN 9.1.0 构建系统；版本升级后 `ASC_VALID_SOC_LIST` 与映射表可能变化，需重新探测（未验证，推断可迁移）；其他芯片系列的变体命名映射未逐一核对。

验证信号（并入，Ascend950PR 目标算子工程首次搭建，CANN 9.1.0）：

- npu-smi 探测：Chip Name=Ascend950PR、NPU Name=9579；构建系统探测：`ASC_VALID_SOC_LIST` 含 `ascend950`，`opdesc_parser.py` 中 `"ascend950pr_9579": "ascend950"`（来源平台：950；证据等级：tool_observed）。
- `bash build.sh --soc=ascend950 -j8` 构建成功：`[INFO] Build completed successfully!`，kernel 二进制生成于 `build/op_kernel/ascendc_kernels/binary/ascend950/<op>/`，算子包 `custom_opp_ubuntu_x86_64.run` 生成（来源平台：950；证据等级：tool_observed）。