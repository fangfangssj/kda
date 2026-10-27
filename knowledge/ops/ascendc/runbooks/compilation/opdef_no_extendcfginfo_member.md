---
type: "Runbook"
title: "OpDef 顶层无 ExtendCfgInfo—kernel 文件按命名约定映射"
description: "自定义算子 def.cpp 在算子类上调用 this->ExtendCfgInfo(\"opFile.value\", ...) 编译报 has no member named 'ExtendCfgInfo'；该 API 挂在 OpAICPUDef/OpHostCPUDef/OpAICoreConfig 配置层级，kernel 文件映射由 <op_snake>_arch{N}.cpp 命名约定（CMake 按 ASCEND_COMPUTE_UNIT 组织）自动完成，应删除该调用。"
tags: [task.implementation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# OpDef 顶层无 ExtendCfgInfo—kernel 文件按命名约定映射

## 现象

两个独立算子工程（TopK 类算子与双线性插值算子，均为 Ascend950PR / arch35，CANN 9.1.0）在 `op_host/<op>_def.cpp` 的 OpDef 构造函数里，为了让构建系统找到 arch35 架构的 kernel 入口文件，各写了一句显式映射：

```cpp
this->ExtendCfgInfo("opFile.value", "bench_no_harness_top_k_arch35");
this->ExtendCfgInfo("opFile.value", "bench_no_harness_resize_bilinear_arch35");
```

首次执行 `bash build.sh --soc=ascend950 -j8` 时，该行报编译错误（TopK 实例原文）：

```
error: 'class ops::BenchNoHarnessTopK' has no member named 'ExtendCfgInfo'
   80 |         this->ExtendCfgInfo("opFile.value", "bench_no_harness_top_k_arch35");
       |               ^~~~~~~~~~~~~
```

（双线性插值实例同款：`error: 'class ops::BenchNoHarnessResizeBilinear' has no member named 'ExtendCfgInfo'`，def.cpp:54:15。）

当时算子类定义形如 `class <Op> : public OpDef`，在构造函数里链式配置 Input/Output/Attr 与 `AICore().AddConfig("ascend950", aicoreConfig950)` 后尝试调用 ExtendCfgInfo。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子仓开发，多架构目录组织（`op_kernel/<op_snake>_arch{N}.cpp` 入口 + `op_kernel/arch{N}/` 头文件）。
- API 路径：`ops::OpDef` 派生类的构造注册链；CANN `register/asc/op_def.h` 中 `ExtendCfgInfo` 的真实宿主类；`op_kernel/CMakeLists.txt` 按 `ASCEND_COMPUTE_UNIT` 选择源文件。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35）+ CANN 9.1.0（两个独立工程复现，与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（所有自定义算子 def 注册均适用）。
- 路线优先级：recommended_fastpath（命名约定是构建系统的既定机制，不是 workaround）。
- 明确不适用/未验证场景：其他 CANN 版本中若 OpDef 提供 opFile 配置入口，以该版本头文件为准（未验证）；`ExtendCfgInfo` 在 `OpAICoreConfig`/`OpAICPUDef` 配置对象上的正向用法（如确需 `opFile.value` 语义的 AICPU/HostCPU 算子场景）本卡未验证（本两条轨迹选择移除而非迁移调用）。

## 根因

`ExtendCfgInfo` 不定义在 `OpDef`（及其算子派生类）上，而在 `OpAICPUDef`、`OpHostCPUDef`、`OpAICoreConfig` 上（op_def.h 中多处声明）；且自定义算子的 kernel 文件映射本就由两条机制自动完成——算子类型 PascalCase 转 snake_case 后加 `_arch{N}` 后缀对应 `op_kernel/<op_snake>_arch{N}.cpp`，`op_kernel/CMakeLists.txt` 按 `ASCEND_COMPUTE_UNIT` 分支组织源文件——无需也不应在算子类上用 `ExtendCfgInfo("opFile.value", ...)` 显式指定。

核验注册头文件中 ExtendCfgInfo 的宿主类：

```bash
grep -n "ExtendCfgInfo" $ASCEND_HOME_PATH/aarch64-linux/include/register/asc/op_def.h
```

本例输出（三处宿主均非 OpDef）：

```
293:  OpAICPUDef &ExtendCfgInfo(const char *key, const char *value);
326:  OpHostCPUDef &ExtendCfgInfo(const char *key, const char *value);
410:  OpAICoreConfig &ExtendCfgInfo(const char *key, const char *value);
```

再核验官方模板无此用法（结果为空即模板不使用它）：

```bash
grep -rn "ExtendCfgInfo" <官方模板>/references/add_example/op_host/
```

## 解决配方

**before**（`op_host/<op>_def.cpp` 构造函数末尾，TopK 实例）：

```cpp
        // 仅注册 arch35（Ascend950PR/Ascend950DT，DAV_3510）
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);

        // Kernel 文件名映射（arch35 入口）
        this->ExtendCfgInfo("opFile.value", "bench_no_harness_top_k_arch35");
    }
```

**after**（删除 ExtendCfgInfo 调用，以注释记录命名约定映射规则）：

```cpp
        // 仅注册 arch35（Ascend950PR/Ascend950DT，DAV_3510）
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);

        // Kernel 文件名映射通过命名约定自动匹配：
        // 算子类型 BenchNoHarnessTopK → snake_case bench_no_harness_top_k → _arch35 后缀
        // 对应 op_kernel/bench_no_harness_top_k_arch35.cpp
    }
```

双线性插值实例的修复同式（删除该行，保留 AICore 配置），arch 内核文件映射改由 `op_kernel/CMakeLists.txt` 承担（同轨迹的实际写法）：

```cmake
# op_kernel CMakeLists.txt

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    npu_op_kernel_sources(ascendc_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_no_harness_resize_bilinear_arch35.cpp
    )
endif()
```

## 设计决策

- 选择"删除显式映射、依赖命名约定"而非"想办法在正确的对象上调用 ExtendCfgInfo"：官方 add_example 模板不使用该调用，命名约定是既定机制；显式配置反而引入与约定不一致的风险。
- 双线性插值实例选择直接移除而不是把调用迁移到 `aicoreConfig950` 上：目标是让 arch35 内核源参与编译，CMake 的 `ASCEND_COMPUTE_UNIT` 分支已经完成这件事；在配置对象上再挂 `opFile.value` 属于重复且语义未验证。最小改动路径。
- 多架构扩展约定：新增架构时新增 `op_host/arch{N}/`、`op_kernel/arch{N}/` 目录和 `op_kernel/<op_snake>_arch{N}.cpp` 入口文件，并在 `_def.cpp` 中添加对应 `AddConfig`。
- 注意区分：kernel 入口**文件名**由命名约定映射（本卡）；kernel 入口**函数名**必须等于算子 snake_case 名（见「kernel entry not implement」一卡，独立根因）。

## 系统信息

- 工具链：`bash build.sh --soc=ascend950 -j8`，CANN 9.1.0。
- 相关文件：`op_host/<op>_def.cpp`（修改）、`aarch64-linux/include/register/asc/op_def.h`（只读核验）、官方模板 `add_example/op_host/`（只读对照）、`op_kernel/CMakeLists.txt`（arch 文件映射）。

## 预防

- 写 def.cpp 时直接按命名约定放置 kernel 入口文件，不要显式配置 opFile。
- 在 def.cpp 里调用 OpDef 成员前，先 grep `include/register/op_def*.h` 确认方法属于哪个类层级；`ExtendCfgInfo`/`ExtendCfg` 家族挂在 AICore/AICPU 配置对象上，不在 OpDef 顶层。
- arch 专属内核文件映射优先在 `op_kernel/CMakeLists.txt` 按 `ASCEND_COMPUTE_UNIT` 分支组织，不依赖 def.cpp 扩展配置。
- 拿不准某个链式方法挂在哪个对象上时，先 grep 注册头文件确认宿主类再使用。

## 验证信号

来源说明：两个独立算子工程（TopK 类 / 双线性插值类）的 def.cpp 首次构建修复闭环，均为 Ascend950PR + CANN 9.1.0。

- TopK 实例修复前：`error: 'class ops::BenchNoHarnessTopK' has no member named 'ExtendCfgInfo'`（def.cpp:80）（来源平台：950 编译链；证据等级：tool_observed）。
- TopK 实例修复后：同一构建不再出现 `ExtendCfgInfo` 成员错误，编译推进到后续阶段；最终构建成功后，kernel 二进制按命名约定从 `op_kernel/bench_no_harness_top_k_arch35.cpp` 正确生成：`build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_top_k/BenchNoHarnessTopK_*.o`（来源平台：950 编译链；证据等级：tool_observed）。
- 双线性插值实例修复前：`error: 'class ops::BenchNoHarnessResizeBilinear' has no member named 'ExtendCfgInfo'`（def.cpp:54:15）（来源平台：950；证据等级：tool_observed）。
- 双线性插值实例修复后：同一构建命令继续执行，`bench_no_harness_resize_bilinear_def.cpp` 编译通过（构建日志前进到下一处独立错误，即 def.cpp 已不再报错）；该算子最终 `bash build.sh --soc=ascend950 -j8` 全量成功，3 个 dtype（fp16/fp32/bf16）内核二进制生成并打包为 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论验证于 cann-9.1.0 的注册头文件与构建机制；其他版本若映射机制变化需重新核验（未验证，推断 CANN 后续版本保持该约定）。
- 若 kernel 入口文件名不满足 `<op_snake>_arch{N}.cpp` 约定（自定义命名），本卡不覆盖该场景。
- `OpAICoreConfig::ExtendCfgInfo` 的正向用法（如确需 `opFile.value` 语义的场景）未验证，推断可迁移但未实测。

**未决项**：

- 如后续确实需要 `opFile.value` 扩展配置语义（例如 AICPU/HostCPU 算子），应在 `OpAICoreConfig`/`OpAICPUDef` 对象上验证 `ExtendCfgInfo` 的调用方式——优先项：写一个最小 def 复现并确认编译。
## 坏实践

- 在 `OpDef` 派生类上调用 `ExtendCfgInfo`（编译直接失败）。
- 认为必须显式配置 kernel 文件映射，从而绕路去 `OpAICoreConfig` 等对象上找配置入口。
- 凭记忆或类比对（"OpDef 应该有个扩展配置口"）直接在顶层 OpDef 上调用未核实的方法。

## 误诊

- 无（两条轨迹均以头文件 grep 宿主类一次定位）。
