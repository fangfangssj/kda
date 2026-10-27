---
type: "Runbook"
title: "TilingData 改布局须重编 kernel 并清 kernel_meta"
description: TilingData 结构体加字段后只重编 host tiling 库不重编 device kernel，kernel 按旧偏移读字段（maximize 错位成相邻字段尾字节）致与改动无关的用例新失败；重编 kernel 后若不清 TTK 动态编译缓存（kernel_meta）仍会全量失败，且出现"单例 PASS/全量 FAIL"分裂。修复链 = 重编 kernel 目标 + 清 TTK kernel_meta。
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T02:32:55Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# TilingData 改布局须重编 kernel 并清 kernel_meta

## 现象

在 Ascend 950 上给 AdamW 类算子做白盒精度修复：向 `op_kernel/arch35/<op>_tiling_struct.h` 的 TilingData 结构体**新增两个 float 字段**（插在 `beta2Pow` 与 `maximize` 之间），host tiling 库（liboptiling.so）随之重编安装。之后重跑白盒 32 例：

- 先是 28 PASS / 4 FAIL——原本稳定通过的 **bf16 用例（101^3 元素）新失败**（precision 99.9999523%，1 个元素超差），而它和本次改动在数学上毫无交集；
- 按"kernel 没重编"处理，触发 kernel 重编目标后再跑，结果反而 **2 PASS / 30 FAIL**，几乎全量 INVALID/精度崩；
- 同一份安装，单独跑一个用例 PASS、全量跑 30 FAIL，结果随执行方式漂移。

最终确认两个叠加的构建态问题：kernel .o 未随结构体重编（布局错位）+ TTK 动态编译缓存（kernel_meta）陈旧。

## 适用场景

- 工程形态：CANN 自定义算子包（CMakeLists：`npu_op_kernel_sources` + `npu_op_kernel_library`），host tiling 与 device kernel 分别产出（liboptiling.so / `<OpType>_*.o`），测试用 TTK kernel 模式（动态编译，缓存于 `$TTK/kernel_meta`）。
- API 路径：`TilingData` 结构头被 host 与 kernel 双侧包含；`npu_op_library(cust_optiling TILING)`（host）与 `make` 目标 `ascendc_kernels_ascendc_bin_ascend950_gen_ops_config`（kernel）。
- 验证过的平台与 CANN 版本：Ascend950PR（注册平台 950）+ CANN 9.2.0。
- 适用粒度：platform_wide——任何"TilingData/共享结构体改动 + 增量构建 + TTK 动态编译"组合。
- 路线优先级：recommended_fastpath（结构体变更后固定走全量重编 + 清缓存）。
- 明确不适用 / 未验证场景：只改 host 侧纯函数实现（不动结构体）时 kernel 无需重编；cst/bin（预编译二进制）模式不经 kernel_meta，缓存问题不适用。

## 根因

**根因一句话：TilingData 是 host tiling 库与 device kernel 的共享二进制契约，增量构建只重编了 host 侧，kernel .o 仍按旧字段偏移解释 tiling buffer（`maximize` 读到相邻字段的字节），且 TTK 动态编译按 kernel_meta 缓存复用旧产物，两个陈旧构建态叠加造成布局错位的静默错误结果。**

机制链：

1. 结构体在 `beta2Pow` 后插入两个 float → `maximize`（uint8）及其后字段的偏移整体后移 8 字节；
2. host 库重编后按**新布局**写 tiling buffer；kernel .o 未重编，仍按**旧布局**读——`maximize` 位置读到的字节实际是新字段的尾字节，符号翻转标志错乱，bf16 用例等出现"与本次数学改动无关"的新失败；
3. 触发 kernel 重编后，TTK kernel 模式的动态编译产物缓存在 `$TTK/kernel_meta/`：单例执行触发了该例的重新编译（新布局）所以 PASS；全量执行命中旧缓存条目（旧布局 kernel）所以 30 FAIL——同一安装两种结果；
4. 清空 `kernel_meta` 强制全量重新编译后，host/kernel/缓存三者布局一致，恢复 29 PASS / 3 FAIL 基线。

构建层"kernel 头改动不触发重编"的通用机制与更多变体见 [kernel头改动后增量编译沿用旧二进制卡](incremental_build_stale_kernel_binary.md)；本卡的增量是 **TilingData 布局契约**这个触发面与 **TTK kernel_meta 运行期缓存**这一层。

## 解决配方

TilingData（或任何 host/kernel 共享结构体）变更后的固定三步（verbatim 命令）：

```bash
# 1) 重编 device kernel（gen_ops_config 目标真正驱动 kernel .o 生成）
source docs/<Op>/develop/env.sh
cd operators/<Pkg>_package/build
cmake --build . --target ascendc_kernels_ascendc_bin_ascend950_gen_ops_config

# 2) 安装到 vendor（或整包 build.sh：rm -rf build && make binary package && 安装）
bash operators/build.sh <Pkg>_package

# 3) 清 TTK 动态编译缓存，强制按新 tiling 结构重新编译
rm -rf "$TTK/kernel_meta"/*
```

注意：只跑 `cmake --build . --target install` 不够——本轨迹中该目标在 kernel 二进制目录缺失时报 `file INSTALL cannot find ".../ascendc_kernels/binary/ascend950"`，必须先跑 gen_ops_config 目标；`make`/`make install` 的 "Built target" 输出不代表 kernel .o 已按新结构体重编。

## 设计决策

- 为什么 host 与 kernel 会失同步：CANN 包构建里 host tiling（普通 C++ 目标，头文件依赖被 CMake 跟踪）与 device kernel（ascendc 编译链，结构体变更是否触发重编取决于其依赖声明）是两条独立链路；本次 host 侧秒级重编、kernel 侧未动，是最容易踩的窗口。
- 为什么选择清空整个 kernel_meta 而不是精确失效：kernel_meta 条目按编译输入键控，手工判定哪些条目受结构体影响不可靠；清空代价是下一次全量重编（分钟级），换确定性。
- 结构体字段放置的防御性选择：新字段追加到结构体**尾部**（本例是插在中间）可让旧 kernel 读旧前缀字段不错位——但这只是降低爆炸半径，不是正确性保证（长度不匹配仍错），不作为推荐做法。

## 系统信息

- 工具链：CANN 9.2.0 + CMake（`npu_op_kernel_library`），TTK kernel 模式动态编译（缓存 `$TTK/kernel_meta/`）。
- 相关文件：`op_kernel/arch35/<op>_tiling_struct.h`（结构体，host/kernel 双侧包含）、`op_host/<op>_tiling_host.cpp`（写入侧）、`op_kernel/arch35/<op>_kernel.h`（读取侧）、`operators/build.sh`（整包 clean build + 安装）。

## 预防

- 把"改共享结构体"当作破坏性变更对待：提交信息/任务描述显式标注"TilingData 布局变更，需重编 kernel + 清 kernel_meta"，评审时检查是否附带两条命令。
- CI/回归脚本在检测到 `*_tiling_struct.h` 变更时自动 `rm -rf kernel_meta` 并走 gen_ops_config 全量重编。
- 白盒结果出现"与改动无关联用例的新失败"或"单例 PASS/全量 FAIL"分裂时，先怀疑构建态（结构体布局/缓存），再怀疑数学逻辑。

## 验证信号

来源说明：以下信号来自该算子白盒 TTK 全量执行的构建态实测（Ascend950PR，CANN 9.2.0），均为本轨迹内工具输出。

- 结构体加字段 + 仅重编 host 库后：白盒 28 PASS / 4 FAIL，原通过的 bf16 用例新失败（precision 99.9999523%，与数学改动无关的信号）（来源平台：950；证据等级：tool_observed）。
- 触发 kernel 重编（gen_ops_config 目标，日志 `Generating <OpType>_*.o ... Done` ×3）后未清缓存：全量 2 PASS / 30 FAIL（来源平台：950；证据等级：tool_observed）。
- 同一构建态下单例执行 PASS、全量执行 30 FAIL 的分裂现象（缓存命中差异）（来源平台：950；证据等级：tool_observed）。
- `rm -rf "$TTK/kernel_meta"/*` 后全量重跑：恢复 29 PASS / 3 FAIL（与改动前基线一致，3 FAIL 为已定性的属性截断问题）（来源平台：950；证据等级：tool_observed）。
- TilingUT（同源编译 host 纯函数）37/37 PASS——host 侧公式本身正确，佐证问题在构建态而非计算逻辑（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡覆盖"结构体变更 → 布局错位 → 静默错误结果"的构建传播链；3 FAIL 残余是属性精度问题（另卡），勿混同。
- `kernel_meta` 缓存语义基于 TTK（ops-test-kit）当前实现；其他评测框架的动态编译缓存（如 GE 侧 `atc_data/kernel_cache`）有类似陈旧风险但键控方式不同，未验证。
- gen_ops_config 目标名含 SoC（ascend950），换芯片需对应替换（Ascend910B 系列为对应 compute unit 名）。
- 未验证，推断可迁移：其他 CANN 版本同构的 host/kernel 双链路构建。

**未决项**：

- CANN 构建系统对 `npu_op_kernel_library` 的头文件依赖跟踪为何未覆盖 `op_kernel/arch35/*_tiling_struct.h`（该头被 kernel 源包含）未深挖到构建脚本层；若上游能修依赖声明，可从根上免除手工三步（优先项：定位 ascendc 编译链的 DEPENDS 声明）。
- kernel_meta 缓存键是否可加入 tiling 结构体哈希以自动失效，属 TTK 侧改进，未验证。
## 坏实践

- 改完共享结构体只 `make`/`make install` 就跑全量验证：host/kernel 失同步直接产出静默错误结果，且失败模式（无关用例新失败）极具误导性。
- 见到全量崩就回滚代码：本例代码正确，回滚会掩盖真实的构建链问题。
- 依赖单例试跑判断构建健康：单例会触发新鲜编译恰好绕过缓存，掩盖全量才会暴露的陈旧缓存。

## 误诊

- 第一时间把 bf16 用例新失败归因于同批的数学顺序改动并回退了它——实际是结构体布局错位；顺序改动好坏另论，但归因错了（该归因当时无对照实验，被后续"重编+清缓存后同样改动下 bf16 用例恢复"证伪）。
- 曾以为 30/32 崩盘是多 vendor TilingFunc 符号冲突复发——plog 里没有他人工程路径，且重装/查 config.ini 无效，最终由清 kernel_meta 解决。
