---
type: "Runbook"
title: AddConfig 产品变体名致 SoC 未配置构建失败
description: '自定义算子 OpDef 的 AddConfig 照抄 spec 里的产品变体名（Ascend950PR/Ascend950DT）或其小写直译（ascend950pr/ascend950dt）时 opbuild 收集到的 SoC 配置为空，CMake 报 "The soc version of op ... is not configured"；改成与 ASCEND_COMPUTE_UNIT 一致的小写基础型号名 ascend950 后构建、安装、上板冒烟全部通过。复现增量：CANN 9.1.0 双独立 trace 复验，并给出 soc 串全集枚举探针（全集 ascend310p/ascend910_55/ascend910_93/ascend910b/ascend950，无 950pr/950dt）与 platform_config INI 文件名假阳性陷阱。'
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# AddConfig 产品变体名致 SoC 未配置构建失败

## 现象

在开发逐元素数学算子 bessel_i1e（一阶第一类修正贝塞尔函数的指数缩放版，fp32/fp16/bf16 三种 dtype）时，先从 AddCustom 模板工程复制改名得到 `operators/BesselI1e_package/`，再按原型定义文档（proto）把 op_host 重写为接口桩：OpDef 声明 1 输入 `x`、1 输出 `y`，`DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})`。原型文档里的平台配置是照着需求 spec 的平台列表抄的：

```cpp
this->AICore()
    .SetTiling(optiling::TilingFunc)
    .AddConfig("Ascend950PR")
    .AddConfig("Ascend950DT");
```

当时工程其余部分（tiling 桩、InferShape/InferDataType、kernel 桩、CMakeLists）都已就绪，CMakeLists 里计算单元是 `set(ASCEND_COMPUTE_UNIT ascend950)`。在 `operators/` 目录下执行 `./build.sh BesselI1e_package`，算子包本体编译完成，但 CMake 配置阶段（npu_op_code_gen → opbuild）直接失败，报错原文：

```
opbuild ops info:
opbuild ops result: 1
CMake Error at <CANN安装根>/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:657 (message):
  opbuild ops error: The soc version of op BesselI1e is not configured, at
  least one soc version must be configured.  Check whether the soc version is
  added to the op proto type definition in the host implementation(Opdef,
  through AddConfig).

Call Stack (most recent call first):
  CMakeLists.txt:17 (npu_op_code_gen)
```

注意 `opbuild ops result: 1` 说明 op proto 本身被解析到了（1 个算子），卡的是 SoC 配置收集。

## 适用场景

- 工程形态：aclnn 自定义算子工程（AscendC kernel + op_host 的 OpDef/TilingFunc，CMake `custom_op_prj`，`build.sh` 走 `npu_op_code_gen`/opbuild 工具链）。
- API 路径：op_host 侧 `OpDef::AICore().SetTiling(...).AddConfig(<soc>)`；工程 CMakeLists 的 `set(ASCEND_COMPUTE_UNIT <soc>)`。
- 验证过的平台与 CANN 版本：Ascend950（构建目标 `ascend950`，修复后上板冒烟实测通过），CANN 9.0.0-beta.2。
- 适用粒度：platform_wide——该 CANN 版本下任何自定义算子 OpDef 的 SoC 注册写法，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本是否接受 `Ascend950PR`/`Ascend950DT` 字符串未验证；非 950 系列 SoC 的合法 AddConfig 取值未验证（机制推断用对应小写基础型号名）；`ASCEND_COMPUTE_UNIT` 与 AddConfig 值不同 SoC 时的行为未验证。

## 根因

OpDef 的 AddConfig 传入了产品变体名（`Ascend950PR`/`Ascend950DT`），opbuild 收集算子 SoC 配置时得到空集合，于是触发 "at least one soc version must be configured"；该 CANN 版本下 AddConfig 需要的是与 `ASCEND_COMPUTE_UNIT` 一致的小写基础型号名 `ascend950`。

机制链（全部为本会话实测证据）：

1. 需求 spec 列出的支持平台是产品名（Ascend950PR / Ascend950DT），原型文档生成时原样带进了 OpDef 代码片段——产品命名体系与工具链 SoC 命名体系不是一回事。
2. 工程实际构建的计算单元是 `ASCEND_COMPUTE_UNIT=ascend950`（全小写、无变体后缀）。
3. opbuild 解析 op proto 成功（`opbuild ops result: 1`）后收集 AddConfig 声明的 SoC 集合；`Ascend950PR`/`Ascend950DT` 不被认作有效 soc version，集合为空 → `func.cmake:657` 抛 CMake Error。
4. 唯一改动：两行变体名 AddConfig 换成一行 `.AddConfig("ascend950")`，其余代码零改动，同一命令重建即通过全链路（opbuild 生成 3 个 dtype 的 kernel binary → CPack 打包 → 安装 SUCCESS）→ 差异唯一归因于 SoC 名写法。

（"字符串不被识别" 与 "必须等于 ASCEND_COMPUTE_UNIT" 两种内部机制在本会话内不可区分，见 ## 遗留问题。）

## 解决配方

修复前（op_host/bessel_i1e_host.cpp，按原型文档照抄产品变体名）：

```cpp
namespace ops {
class BesselI1e : public OpDef {
public:
    explicit BesselI1e(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("Ascend950PR")
            .AddConfig("Ascend950DT");
    }
};
OP_ADD(BesselI1e);
} // namespace ops
```

修复后（同文件，仅 SoC 名改为与 CMakeLists 计算单元一致的小写基础型号）：

```cpp
namespace ops {
class BesselI1e : public OpDef {
public:
    explicit BesselI1e(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend950");
    }
};
OP_ADD(BesselI1e);
} // namespace ops
```

判别 / 定位方法（本会话走通的链路，可复用）：

1. 看报错位置：`func.cmake:657` 的 message 明说 "Check whether the soc version is added to the op proto type definition in the host implementation(Opdef, through AddConfig)"——直接指向 host 实现里的 AddConfig，不要去查 CMakeLists 的 npu_op_code_gen 调用。
2. 对照工程 CMakeLists 的 `set(ASCEND_COMPUTE_UNIT ...)` 取值（本例 `ascend950`），AddConfig 与之同值同大小写。
3. 改完重建同一命令验证（构建 + 打包 + 安装 + 上板冒烟一条龙）。

## 设计决策

- 为什么不保留两个变体名分别注册：实测该版本 opbuild 对 `Ascend950PR`/`Ascend950DT` 一个都不认（构建直接失败），基础型号名是该版本下唯一可行写法；一个通用名即覆盖整个系列。
- 为什么选 `ascend950` 而不是尝试其它大小写组合：与 CMakeLists 的 `ASCEND_COMPUTE_UNIT` 值逐字符一致，且模板工程（AddCustom）原本就用该值成功构建过——有前例可循，避免试错。
- 契约冲突的处理：原型文档（proto）此时仍是 Ascend950PR/DT 写法，与可构建代码不一致；会话中以"工具链可构建"为准先改代码，文档回写留作后续（见 ## 遗留问题）。
- 当前方案路线档位：recommended_fastpath（正确性必需，无替代路线）。

## 系统信息

- 硬件/软件：Ascend950 构建目标（kernel .o 生成于 `ascend950` 目录），CANN 9.0.0-beta.2，GCC 13.3.0。
- 相关文件：`operators/BesselI1e_package/op_host/bessel_i1e_host.cpp`（OpDef/AddConfig 所在）、`operators/BesselI1e_package/CMakeLists.txt`（`ASCEND_COMPUTE_UNIT ascend950`）、算子原型文档（AddConfig 错误值的来源）。
- 复现 / 验证命令：

```bash
# 构建（失败复现与修复验证同一命令）
cd operators && ./build.sh BesselI1e_package
# 端到端冒烟（重建 + 重装 + 上板运行）
cd operators/BesselI1e_package/tests/aclnn && ./run.sh
```

## 预防

- 写自定义算子 OpDef 时，AddConfig 一律用与工程 CMakeLists `ASCEND_COMPUTE_UNIT` 一致的小写基础型号名（如 `ascend950`），不要照抄需求 spec / 产品文档里的产品变体名。
- 从 spec 或需求文档自动生成 OpDef 代码时，"支持平台"字段必须经过"产品名 → SoC 基础型号名"映射，不能直传；生成后应与 CMakeLists 的计算单元做一次一致性检查。

## 验证信号

以下信号来自 bessel_i1e 算子工程的 OpDef 接口桩改造会话：首次构建在 CMake 配置阶段报 SoC 未配置，单点修改 AddConfig 后重建并跑通端到端冒烟。

- 修复前：`./build.sh BesselI1e_package` 失败，`opbuild ops result: 1` 后报 `opbuild ops error: The soc version of op BesselI1e is not configured, at least one soc version must be configured.`（CMake Error at func.cmake:657，Call Stack CMakeLists.txt:17 npu_op_code_gen）（来源平台：950；证据等级：tool_observed）。
- 修复后：同一命令全链路通过——`Opbuild generating sources - done`、3 个 dtype 的 kernel binary 生成（`[ascend950] Generating BesselI1e_<hash> ... Done` ×3）、CPack 打包 `custom_opp_ubuntu_x86_64.run` 并安装 SUCCESS（来源平台：950；证据等级：tool_observed）。
- 修复后：上板冒烟测试 3/3 PASSED（fp16/fp32/bf16，n=128，零填充输入过桩 kernel，`==== Smoke test summary: 3/3 cases PASSED ====`）（来源平台：950；证据等级：tool_observed）。
- 修复后：`nm -D $ASCEND_OPP_PATH/vendors/BesselI1e/op_api/lib/libcust_opapi.so` 含 `aclnnBesselI1e` 与 `aclnnBesselI1eGetWorkspaceSize` 两个导出符号（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.0.0-beta.2 的 opbuild 行为；升级 CANN 后 `Ascend950PR`/`Ascend950DT` 变体名是否被接受须重验（未验证）。
- 非 Ascend950 系列 SoC 的合法 AddConfig 取值未验证（机制推断：用对应小写基础型号名，未验证，推断可迁移）。
- `ASCEND_COMPUTE_UNIT` 与 AddConfig 指向不同 SoC（如 CMake 写 A、AddConfig 写 B）时是报错还是静默生成其它 SoC 的产物，未验证。

**未决项**：

- opbuild 对 AddConfig 字符串的精确匹配规则（"变体名字符串非法"还是"必须等于 ASCEND_COMPUTE_UNIT"）未追到工具链源码层——优先项：在 `func.cmake`/opbuild 侧复现并追根因到机制层（可用 `Ascend950PR` 与 `ascend950pr` 等变体做控制变量实验）。（复现增量部分推进：已枚举出 CANN 9.1.0 安装内 AddConfig 实参全集——见下文增量，`ascend950pr`/`ascend950dt` 确不在其中；精确匹配规则仍未读到源码。）
- 原型文档中的 AddConfig 示例仍是 `Ascend950PR`/`Ascend950DT`，与实际可构建写法不一致，未回写；后续有人照抄会再踩同一个坑。
## 坏实践

- 把需求 spec / 产品文档里的平台名（`Ascend950PR`、`Ascend950DT` 这类产品变体名）原样抄进 AddConfig。
- 遇到该报错去改 CMakeLists 的 `npu_op_code_gen` 调用、或干脆删掉 AddConfig——报错文本已指明修改位置是 host 实现里的 AddConfig。

## 误诊

- 曾在"原型文档是唯一契约源、不该偏离"上犹豫：但构建环境不认就是硬约束，契约文档与工具链能力冲突时以可构建为准，并把偏差记回文档（本会话文档回写未完成，见遗留问题）。
- 曾怀疑大小写敏感性问题方向正确但未停在猜测：直接用与 `ASCEND_COMPUTE_UNIT` 逐字符一致的值一次性修复，避免多轮试错。

## 增量（CANN 9.1.0 双独立复验 + soc 串全集枚举探针）

两个独立 trace 在 CANN 9.1.0 + Ascend 950 系列上复验同一结论并补充判定工具。

**复验一（SoftplusV2Grad 原型提取，诊断链）**：spec.yaml `supported_chips: [Ascend950PR, Ascend950DT]` 与工具包 soc 注册串集合不一致。判定探针（在本机 CANN 工具包上执行，输出为实测原文）：

```bash
# 探针 1：工具包头文件里的 ascend soc 串全集（不含 950pr/950dt）
grep -rhoE "ascend[0-9][a-z0-9_]*" /home/<user>/Ascend/cann-9.1.0/x86_64-linux/include/ | sort -u
# → ascend035 / ascend910 / ascend910_93 / ascend910_96 / ascend950

# 探针 2（陷阱演示）：全盘 grep 芯片名会命中 platform_config INI 文件名，
# 不能当作 soc 串存在性证据
grep -rhoiE "950pr|950dt|950_pr|950_dt" /home/<user>/Ascend/cann-9.1.0/ | sort -u
# → 950dt / 950DT / 950pr / 950PR（全部来自 Ascend950PR_*.ini / Ascend950DT_*.ini 文件名）
ls /home/<user>/Ascend/cann-9.1.0/x86_64-linux/data/platform_config/ | grep -i "950"
# → Ascend950DT_*.ini 等 25 个 + Ascend950PR_*.ini 等 9 个

# 探针 3（判定性）：枚举整个安装下所有 AddConfig 实参
grep -rhoE "AddConfig\(\"ascend[0-9a-z_]*\"" /home/<user>/Ascend/ | sort -u
# → AddConfig("ascend310p" / AddConfig("ascend910_55" / AddConfig("ascend910_93"
#   / AddConfig("ascend910b" / AddConfig("ascend950" —— 无 ascend950pr / ascend950dt
```

判别链：探针 3 是判定性证据——工具包里**所有** built-in 算子对 950 家族只用 `"ascend950"` 一个串（built-in `mat_mul_v3_def.cpp:137` 为 `this->AICore().AddConfig("ascend950", aicConfig);`）；探针 2 的命中（platform_config INI）只证明存在 PR/DT 两个 SKU 的平台配置文件，不证明存在同名 soc 注册串——两者是两套命名。三层命名各司其职：**spec/需求层**写产品型号（`Ascend950PR`/`Ascend950DT`），**platform_config 层**写带数字后缀的具体 SoC（`Ascend950PR_9579`），**OpDef/构建层**写小写架构家族（`ascend950`）；家族内所有型号共用一个 `AddConfig` 值。

据此落地的 OpDef（可与 `SetTiling` 链式书写）：

```cpp
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend950");
```

**复验二（同族正向验证）**：采用 `"ascend950"` 后 `./build.sh` 退出码 0，9 个 dtype 组合的 kernel `.o` 生成并安装到 `$ASCEND_OPP_PATH/vendors/SoftplusV2Grad/op_impl/ai_core/tbe/kernel/ascend950/softplus_v2_grad/`，aclnn 两段式冒烟 9/9 dtype 组合 PASSED；另一算子工程同配置下 `run.sh` 真机执行 test_aclnn 28/28 PASSED（14 shapes × {fp16,fp32}）。

增量验证信号（独立 trace，均为 tool_observed）：

- 探针 1 输出：CANN 9.1.0 include 下 ascend soc 串全集 = ascend035/ascend910/ascend910_93/ascend910_96/ascend950，无 950pr/950dt（来源平台：950；证据等级：tool_observed）。
- 探针 2 输出：全安装 grep `950pr|950dt` 命中仅为 platform_config INI 文件名（Ascend950PR_* 9 个、Ascend950DT_* 25 个）（来源平台：950；证据等级：tool_observed）。
- 探针 3 输出：全安装 AddConfig 实参枚举 = ascend310p/ascend910_55/ascend910_93/ascend910b/ascend950；built-in mat_mul_v3 def 第 137 行 `this->AICore().AddConfig("ascend950", aicConfig);`（来源平台：950；证据等级：tool_observed）。
- 构建验证：带 `AddConfig("ascend950")` 的 op_host 经 `./build.sh` 退出码 0，9 个 dtype 组合的 kernel `.o` 生成并安装到 `op_impl/ai_core/tbe/kernel/ascend950/softplus_v2_grad/`；aclnn 两段式调用 9/9 dtype 组合 PASSED（来源平台：950；证据等级：tool_observed）。
- 同族工程：`run.sh` 真机执行 test_aclnn 28/28 PASSED（14 shapes × {fp16,fp32}），后续 13 字段 TilingData 版本再次构建 + 冒烟 4/4 PASSED（来源平台：950；证据等级：tool_observed）。

增量坏实践：

- `AddConfig("ascend950pr")` / `AddConfig("ascend950dt")`——工具包中不存在这两个注册串，属于芯片名小写直译。
- 用全盘 grep `950pr|950dt` 的命中（platform_config INI 文件名）当作"存在对应 soc 串"的证据——开发中途一度据此得出"soc 串应为 ascend950pr/ascend950dt"的错误结论，随后被探针 3 的全量枚举推翻。
- 拿 platform_config 的 `SoC_version=Ascend950PR_9579` 当 `AddConfig` 实参——那是平台侧具体型号标识，不是 OpDef 注册用的家族名。

增量失效边界：

- soc 注册串集合按 CANN 9.1.0 实测；其他 CANN 版本的集合可能变化（新版本可能注册更细粒度的 soc 串），迁移前须用探针 3 重新枚举（未验证，推断可迁移）。
- 950DT 实机未验证（本地设备具体 SKU 未确认，仅能由 ascend950 内核实跑通过推断为同族）。
- `AddConfig("ascend950", aicConfig)`（带 per-AICore 配置结构，含 DynamicCompileStaticFlag 等标志）的高级形态未在本工作流采用和验证。

## 后续增量（2026-09-17，并卡合并：系列级注册的正向配方、BUILD_SOC_VERSION 变体分流与六门验收）

#### 并入自 ascend950_addconfig_series_level_compute_unit（正向冷启动配方、变体条件编译与六门机械验收）

- **变体差异由 `BUILD_SOC_VERSION` 宏承担**：注册与构建统一用系列级 `ascend950` 后，PR/DT 等变体级条件编译走 `-D BUILD_SOC_VERSION=Ascend950` 编译定义（CMakeLists 同时 `set(ASCEND_COMPUTE_UNIT ascend950)`，opbuild 传 `--compute_unit=${ASCEND_COMPUTE_UNIT}`）；`platform_config/` 下 PR/DT 是同一 950 系列的两个变体 .ini，家族内所有型号共用一个 AddConfig 值。
- **核心映射（取自已验证工程，与上文错误路径互为正反例）**：`op_host/<op>_def.cpp` 构造器注册 `this->AICore().AddConfig("ascend950", aicoreConfig)`（系列级计算单元名，不是 spec `supported_chips` 里的变体名）；按 `Input()/Output().DataType({...})` 列出的 N 个 dtype 自动生成 N 个 `<OpType>_<hash>.o` kernel 二进制，安装到 `vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/<kernel入口名snake_case>/` 下。
- **六门机械验收（全部可脚本化，退出码判定）**：

```bash
# ① 构建退出码 0（抓日志）
( set -o pipefail; cd operators && ./build.sh <package_dir> 2>&1 | tee /tmp/<op>_build.log )
# ② 构建日志无 tiling 结构注册缺失
! grep -qi 'do not registe tiling struct' /tmp/<op>_build.log
# ③ aclnn 冒烟全 dtype 通过
cd operators/<package_dir>/tests/aclnn && ./run.sh
# ④ opapi so 导出两符号
nm -D ${ASCEND_OPP_PATH}/vendors/<OpType>/op_api/lib/libcust_opapi.so | grep -E 'aclnn<OpType>$|aclnn<OpType>GetWorkspaceSize'
# ⑤ kernel .o 数量 == dtype 组合数
ls ${ASCEND_OPP_PATH}/vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/<snake入口名>/*.o | wc -l
# ⑥ 无旧模板名残留
! grep -rni 'add_custom' operators/<package_dir>/
```

- 失效边界补充（并入）：直接用变体名（`"Ascend950PR"`/`"Ascend950DT"`）注册 AddConfig 未验证（与上文错误路径一致，该版本不认）；混合 dtype 组合（输入输出不同 dtype）的 kernel 变体生成规则未验证；geir/tiling 子测试仅随包构建、未运行；构建头文件取 CANN 9.1.0、运行时与 vendors 安装目标指向 CANN 9.2.0 的同包混用实测可行。

验证信号（并入，BenchHarnessApplyAdamW 类算子冷启动，四输入 var/grad/m/v → 输出 y，fp32/fp16/bf16 三组 dtype）：

- 系列级注册（`AddConfig("ascend950")` + `ASCEND_COMPUTE_UNIT ascend950` + `BUILD_SOC_VERSION=Ascend950`）构建一次通过；按 `Input().DataType` 列出的 3 个 dtype 自动生成 3 个 kernel 二进制，安装到 `op_impl/ai_core/tbe/kernel/ascend950/<snake入口名>/`（来源平台：950；证据等级：tool_observed）。
- 两阶段 aclnn 冒烟 3/3 通过（`tests/aclnn/run.sh`，构建头 9.1.0 + 运行时/安装 9.2.0 混用）（来源平台：950；证据等级：tool_observed）。