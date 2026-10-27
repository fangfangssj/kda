---
type: "Runbook"
title: DynamicCompileStaticFlag 传参错位致输出全 0
description: op_def 开启 DynamicCompileStaticFlag(true) 后框架按 (输入,输出,tiling) 绑定 kernel 实参：5 参签名错位——tiling 形参读到垃圾值、workspace 形参落只读区（写即 507015）、或第 5 参 NULL，AIC IterateAll 静默不写、输出全 0 且 plog 无错；修复须 flag=false + workspace 前置标准签名 + 3 参 GET_TILING_DATA_WITH_STRUCT 联动。
tags: [task.debugging, topic.implementation.operator_registration, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# DynamicCompileStaticFlag 传参错位致输出全 0

## 现象

在开发一个 conv3d 反向滤波器自定义算子时（im2col + Matmul 实现方案，目标芯片 ascend950 / Ascend950PR，arch35 MIX 模式，CANN 9.1.0），CPU 侧全部正常（UT 10/10、ST Mock 5/5 通过），但 ST Real 上板测试输出全 0：

```
[Real] L0-01 - 基础shape K=3 stride=1 pad=1 dilation=1 groups=1 核心路径
  x=[N=2,C_in=64,D=8,H=16,W=16] grad=[N=2,C_out=128,D_out=8,H_out=16,W_out=16]
  [DEBUG] workspaceSize=17138176 bytes
  [FAIL] MERE=1.00e+00 (>= 9.77e-04), MARE=1.00e+00 (>= 9.77e-03)
  不匹配 [0]: 期望=15.367188, 实际=0.000000, rel_err=1.00e+00
```

三个关键并存事实：同一二进制 ST Mock（CPU golden）5/5 全过（MERE=0.00e+00）；plog grep error/exception/aivmap 无任何错误（kernel 正常跑完）；`npu-smi` 显示设备正常。这是一个**静默失败**：计算逻辑正确（Mock 证明）、执行无异常（plog 证明）、但 NPU 上 Matmul 结果没落到输出。

算子的 op_info.json 中 `dynamicCompileStatic: {"flag": "true"}`（即 `_def.cpp` 里注册了 `DynamicCompileStaticFlag(true)`），kernel 二进制元数据显示 `coreType=MIX`、`taskRation=1:2`、`parameters=[null,null,null,null]`（4 项，修复前后均为该形态）、`opParaSize=496`。在 kernel 内把 tiling 字段写到输出观测：`M_=1/0、K_=0`（期望 M=128、K=4096），且不同形参映射组合下读到的垃圾值不同——说明 tiling 是从运行时形参读到的错误数据，而不是没执行；向第 5 形参（workspace 位）写入数据则直接触发 AIV error（ret=507015，只读区域）。AIV 阶段向 workspace 写 9.0 再读回为 0，生成阶段数据全部落空，最终输出全 0。

早期同会话在旧签名 `(x, grad, y, workspace, tiling)`（照 add_example 模板，workspace 在前）下的形态：写第 4 参（workspace 位）即 507015、位图 `aivmap[5]:00000001`；同构最小 kernel 改写输出 `gmY_[0]` 则正常——即"写输出没事，写 workspace 必炸"。

## 适用场景

- 工程形态：算子仓开发（CMake + build.sh `--soc=ascend950` 构建，aclnn L2 API + `l0op::Contiguous/ViewCopy` + `CommonOpExecutorRun` 调用，ST Real 上板验证），op_def 开启 `DynamicCompileStaticFlag(true)`，kernel 需要 workspace。
- API 路径：`OpAICoreConfig::DynamicCompileStaticFlag`（op_host `_def.cpp` 注册）、kernel 入口签名、`GET_TILING_DATA_WITH_STRUCT` / `GET_TILING_DATA` / `REGISTER_TILING_DEFAULT`、`AscendC::SetSysWorkspace` / `GetUserWorkspace`、`AscendC::PRINTF` / GM SetValue/GetValue 探针。
- 验证过的平台与 CANN 版本：Ascend950（ascend950 / Ascend950PR，arch35 / DAV_3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何使用 `DynamicCompileStaticFlag(true)`（或 ASCENDC_TPL 模板机制）且 kernel 需要 workspace 的自定义算子。
- 路线优先级：recommended_fastpath（标准动态模式是 CANN 内置算子 swin_attention_score_quant / conv3d_backprop_filter_v2 arch35 的通行模式）。
- 明确不适用 / 未验证场景：CPU Mock / UT 环境（tiling faker 不复现此问题）；无 workspace 需求的算子；不走 l0op executor 的直调路径；其他 CANN 版本与 arch 的传参布局未验证（推断可参考，未实测）。

## 判别方法

前提工具坑：该环境 `AscendC::printf` 在 stdout 与 plog 均不可见（多轮试验无输出），探针一律改用 **GM SetValue/GetValue + ST "不匹配" 输出读数** 或 **host 侧 0x1234 预填探针**。

**探针阶梯一（GM SetValue/GetValue，每步只验证一个环节，全部上板实测）**：

1. **验证 AIV 输出通路**：Cast 阶段（AIV）写常量 1.0 到 `y[0..31]`，跳过正常 Cast。观测：`不匹配 [0]: ... 实际=1.000000` → AIV 在执行、gmY_ 可写、输出链路正常。
2. **验证 workspace 可读**：AIV 读 `wsGrad[0]/wsIm2col[0]/wsC[0]`（GetValue）转写 y。观测：三者读数全部等于上一轮探针写入的 2.0 → workspace 读数与预期不符。
3. **验证 workspace 写读**：AIV 写 5.0 到 `wsGrad[0]` 再读回输出。观测：读回 0 → 写读不一致，workspace 指针可疑。
4. **验证 AIC 是否写结果**：SyncAll 前 AIV 预写 3.0 到 `wsC[0]`，跑完整流程。观测：wsC[0] 仍为 3.0 未被覆盖 → **AIC 的 IterateAll 根本没有写入 wsC**。
5. **对照**：跳过整个 MatmulCompute → 无异常（其余阶段健康）；plog 全程无 error。

判读：AIV 通路正常 + workspace 写读异常 + AIC 不写 wsC + 无任何报错 + Mock 全过 → 指向 kernel 参数层（workspace 指针无效），而非 kernel 内部逻辑。

**探针阶梯二（host 侧 0x1234 预填，判别第 N 参是否绑定用户 workspace）**：

Host 侧（测试代码，执行段前预填）：

```cpp
    // 第二段：执行
    // [BISECT] pre-fill workspace with known pattern 0x1234 (half bits) to test kernel workspace binding
    if (workspace != nullptr) {
        uint16_t pat[8] = {0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234};
        aclrtMemcpy(workspace, 16, pat, 16, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    ret = aclnnBenchNoHarnessConv3dBackpropFilter(workspace, workspaceSize, executor, stream);
```

kernel 侧（最小读回：把待验参数指向的前几个 half 拷到输出 y，`blockIdx_ == 0` 限流）：

```cpp
    // [BISECT] minimal single DataCopyPad to test API usage on arch35 AIV
    if (blockIdx_ == 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(wOutBytes), 0, 0, 0};
        DataCopyPadExtParams<D_T_X> padParams{false, 0, 0, 0};
        // [BISECT] READ from gmWsGrad_ (workspace) -> UB, then WRITE UB -> gmY_
        DataCopyPad(ubGrad, gmWsGrad_[0], copyParams, padParams);
        DataCopyPad(gmY_[0], ubGrad, copyParams);
    }
    return;
```

Host 侧回读判读：

```cpp
    // [BISECT] debug: check if kernel read workspace pattern into y
    {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(output_data.data());
        LOG_PRINT("  [DBG-PAT] y[0..3]=0x%04x 0x%04x 0x%04x 0x%04x (expect 0x1234 if kernel workspace=user workspace)",
                  p[0], p[1], p[2], p[3]);
    }
```

实测：`y[0..3]=0x0000 0x0000 0x0000 0x0000` → 待验形参不是用户 workspace。

## 根因

`DynamicCompileStaticFlag(true)` 静态编译模式下框架传入的 kernel 实参与 5 参签名错位：tiling 被编译期烘焙进二进制（`GET_TILING_DATA_WITH_STRUCT` 生成的宏以 `const uint8_t __ascendc_arr_##tiling_data[{tiling_size}] = {...}` 常量数组反序列化，忽略运行时 tiling 实参），workspace 形参要么落在只读区域（写入触发 AIV error 507015），要么未被传入——用户 workspace 不可写、生成阶段全部落空，ST 输出全 0。

烘焙证据（`get_op_tiling.py` 生成的宏，CANN 9.1.0，会话内读取安装文件）：

```python
        class_body += "#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                \\\n"
        class_body += "    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                                  \\\n"
        class_body += \
            f"    const uint8_t __ascendc_arr_##tiling_data[{tiling_size}] = {{{tiling_arr_data_str}}};    \\\n"
        class_body += \
            "    const tiling_struct tiling_data = "
        class_body += \
            f"convert_from_bytes<tiling_struct>(__ascendc_arr_##tiling_data); \n\n"
```

机制链（均为上板可观测证据）：

1. flag=true + 签名 `(x,grad,y,tiling,workspace)`：tiling 字段读到 `M_=1/0、K_=0`；写第 5 参 → ret=507015（只读）。
2. flag=true + 交换签名 `(x,grad,y,workspace,tiling)`：写第 4 参（workspace 位）→ ret=507015（只读），说明该模式下第 4 参也不是可写 workspace。
3. 单独把 flag 改为 false、但保留旧签名和 `SetSysWorkspace(tiling)` hack：仍然全 0——flag 与签名必须一起改。
4. flag=false + 标准签名 `(x,grad,y,workspace,tiling)` + `SetSysWorkspace(workspace)` + 3 参 `GET_TILING_DATA_WITH_STRUCT`：tiling 读到实值（M=128 N=1728 K=4096），workspace 形参为有效指针。

## 解决配方

两处必须联动修改。

**第 1 处：`op_host/*_def.cpp` 注册配置**，把静态模式切为标准动态模式：

```cpp
        // 修复前
        // 仅注册 ascend950 配置
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
```

```cpp
        // 修复后
        // 仅注册 ascend950 配置
        // DynamicCompileStaticFlag(false): 运行时从 tiling 参数读取 TilingData（标准动态模式），
        // 框架传递 5 个 kernel 参数 (x, grad, y, workspace, tiling)，workspace 可正常访问。
        // 参考 swin_attention_score_quant / conv3d_backprop_filter_v2 arch35（均未开启 static 烘焙）。
        // ⚠️ DynamicCompileStaticFlag(true) 会将 tiling 编译期烘焙，框架只传 4 参数，
        // workspace 指针无法正确获取，导致生成阶段写入失败、Matmul 读到空数据（issue 根因）。
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
```

**第 2 处：kernel 入口重写为标准 5 参签名**（workspace 在 tiling 之前，与 swin / conv3d_backprop_filter_v2 arch35 一致）：

```cpp
// 修复前（DynamicCompileStaticFlag(true) 模式下的错位签名 + 把第 4 参当 workspace 的 hack）：
// DynamicCompileStaticFlag(true): 框架传 4 参数 (x, grad, y, tiling)
// tiling 参数实际是整个 workspace buffer 的起始指针（含 system ws + tiling data + user ws）
template <typename D_T_X, int GROUPS_MODE>
__global__ __aicore__ void bench_no_harness_conv3d_backprop_filter(
    GM_ADDR x, GM_ADDR grad, GM_ADDR y, GM_ADDR tiling, GM_ADDR workspace)
{
    // tiling（4th 参数）是 workspace buffer 起始指针
    // SetSysWorkspace 设置系统 workspace（buffer 开头）
    AscendC::SetSysWorkspace(tiling);
    // GetUserWorkspace 获取用户 workspace 指针（system workspace 之后）
    GM_ADDR user = AscendC::GetUserWorkspace(tiling);

    REGISTER_TILING_DEFAULT(BenchNoHarnessConv3dBackpropFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);

    NsConv3dBpFilter::Conv3dBpFilterKernel<D_T_X, GROUPS_MODE> op;
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
    op.Init(x, grad, y, user, &tilingData);
    op.Process();
}
```

```cpp
// 修复后（标准动态模式）：
// 标准动态模式：框架传 5 参数 (x, grad, y, workspace, tiling)
// workspace 在 tiling 之前（参考 swin:22-36, conv3d_backprop_filter_v2_arch35.h:32-33）
template <typename D_T_X, int GROUPS_MODE>
__global__ __aicore__ void bench_no_harness_conv3d_backprop_filter(
    GM_ADDR x, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // SetSysWorkspace 用 workspace 参数（非 tiling），设置系统 workspace
    AscendC::SetSysWorkspace(workspace);

    // GET_TILING_DATA_WITH_STRUCT：运行时从 tiling 参数读取 TilingData（标准动态模式，非编译期烘焙）。
    // ⚠️ ASCENDC_TPL 模板机制下 2 参数 GET_TILING_DATA 为 no-op（ascendc_tiling_no_register=True），
    //    必须使用 3 参数 GET_TILING_DATA_WITH_STRUCT + REGISTER_TILING_DEFAULT（参考 add_example）。
    //    DynamicCompileStaticFlag(false) 保证 tiling 运行时从 tiling 参数读取（非编译期烘焙）。
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv3dBackpropFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);

    // GetUserWorkspace 用 workspace 参数获取用户 workspace 指针（系统 workspace 之后）
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    NsConv3dBpFilter::Conv3dBpFilterKernel<D_T_X, GROUPS_MODE> op;
    // REGIST_MATMUL_OBJ 在 Init 之前（参考 swin:47-48，用栈上 op.pipe_）
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
    op.Init(x, grad, y, user, &tilingData);
    op.Process();
}
```

**宏形态要点（重要）**：在 ASCENDC_TPL 模板机制（`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL`）下，2 参形式 `GET_TILING_DATA(tilingData, tiling)` 会被生成为 no-op，编译报 `use of undeclared identifier 'tilingData'`（随后链接报 `ld.lld: error: cannot open ....o`）。必须使用 3 参 `GET_TILING_DATA_WITH_STRUCT(Struct, tilingData, tiling)` + `REGISTER_TILING_DEFAULT(Struct)`——这是 add_example 模板的标准组合；`DynamicCompileStaticFlag(false)` 负责让该宏在运行时从 tiling 参数读取，而非编译期烘焙。清理 build 目录后重编译（build 缓存不清理时 binary 元数据可能不变）。

## 设计决策

- 为何保留 `GET_TILING_DATA_WITH_STRUCT` 而非任务原定的 2 参 `GET_TILING_DATA`：2 参形式在本工程 ASCENDC_TPL 模板机制下实测为 no-op（编译错误证明），而 CANN 内置 conv3d_backprop_filter_v2 arch35 同样使用 3 参 WITH_STRUCT + REGISTER_TILING_DEFAULT 且工作正常——宏形态与模板机制绑定，不是随便二选一。
- 为何 flag 与签名必须一起改：单独改 flag（旧签名 + `SetSysWorkspace(tiling)` hack）实测仍全 0；单独交换签名（flag 仍 true）实测写第 4 参即 507015。两者是同一个传参约定的两端。
- 参考实现选择 swin_attention_score_quant.cpp 与 conv3d_backprop_filter_v2 arch35：两者均为 CANN 内置、在 arch35 上运行的 workspace 用户（签名 workspace 前置 + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`）。
- 为什么 tilingData 正确也不能证明参数顺序正确：静态模式下 tiling 来自编译期烘焙（宏忽略 tiling 实参），tiling/M/N/K 全对只是烘焙的功劳——签名错位只会在"用 workspace 的算子"上暴露。
- 探针选择 0x1234 半精度位模式 + 只搬 16 字节：足够小（不触碰对齐/越界），足够特异（与清零内存区分）。
- 为什么不直接信 add_example 模板签名：官方模板顺序在本框架组合下与实测绑定不符，且模板算子不用 workspace，错位对它不可见。

## 预防

- 在 op_host `_def.cpp` 里开启 `DynamicCompileStaticFlag(true)` 前，先确认 kernel 是否需要 workspace；需要 workspace 的算子直接走标准动态模式（flag=false）+ workspace 前置 5 参签名。
- kernel 入口对照 CANN 内置参考（swin_attention_score_quant.cpp、conv3d_backprop_filter_v2 arch35）核对三件事：签名顺序（workspace 在 tiling 前）、`SetSysWorkspace(workspace)` 用 workspace 形参、宏形态（ASCENDC_TPL 下必须 3 参 WITH_STRUCT + REGISTER_TILING_DEFAULT）。
- 上板出现"CPU 全过、NPU 全 0"时，第一步先在 kernel 里 PRINTF tiling 关键字段（M/N/K）与指针值，再谈算法层排查。
- 对照模板抄签名时，凡模板自身不使用的参数（workspace）都视为未验证接口，须自行探针（host 预填魔数读回）。

## 验证信号

问题与验证来自同一次 conv3d 反向滤波器算子的上板联调（含探针阶梯与修复复验，Ascend950PR，CANN 9.1.0）：

- 触发：ST Real 5/5 `MERE=1.00e+00`、`实际=0.000000`；ST Mock 5/5 `MERE=0.00e+00`（来源平台：950；证据等级：tool_observed）。
- plog 静默：`grep -i "error|fail|exception|aivmap|aicore|assert"` 无命中（来源平台：950；证据等级：tool_observed）。
- 探针阶梯一：AIV 写 1.0 → ST 输出 `实际=1.000000`；workspace 读数恒为上轮写入值；写 5.0 读回 0；AIV 预写 wsC[0]=3.0 流程结束后仍 3.0 未被 AIC IterateAll 覆盖（来源平台：950；证据等级：tool_observed）。
- 探针阶梯二（flag=true 旧签名下）：最小写 workspace kernel → ret=507015、`aivmap[5]:00000001`；同构 kernel 改写 gmY_ → 正常；0x1234 探针 `y[0..3]=0x0000...`（非 0x1234）→ 待验形参 ≠ 用户 workspace（来源平台：950；证据等级：tool_observed）。
- 机制链 2（flag=true + 交换签名）：写第 4 参（workspace 位）→ ret=507015（只读）（来源平台：950；证据等级：tool_observed）。
- 机制链 3（单独改 flag=false、保留旧签名 + SetSysWorkspace(tiling) hack）：仍全 0（来源平台：950；证据等级：tool_observed）。
- 修复后 kernel 内 `AscendC::PRINTF` 打印 tiling 实值：`[AIC Block 0/28] DBG tiling: M=128 N=1728 K=4096 aL1=131072 bL1=131072 cubeSharedUb=1024 sM=16 sN=16 sK=4096 bM=16 bN=16 bK=1024 usedCore=28 transLen=1024`（来源平台：950；证据等级：tool_observed）。
- 修复后指针打印：`DBG ptrs: workspace=0x12004d600000 user=0x12004d600000 sysWs=0x12004c600000 tiling=0x120000200030`——workspace 形参非空、tiling 形参独立有效（来源平台：950；证据等级：tool_observed）。
- 修复后 UT 10/10 通过（含 fp16/bf16 各 TilingKey 的 tiling 用例，无回归）（来源平台：950 主机侧；证据等级：tool_observed）。
- 修复后 ST Real 错误形态从"静默全 0"变为 5/5 `ret=507015` 流同步失败（AIC aicore 异常）——原"全 0"现象消失，新错误为 REGIST_MATMUL_OBJ 场景的独立根因（见 arch35 MIX 相关卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `DynamicCompileStaticFlag(true)` 下框架传参布局（"4 个 GM 参数 + opPara"）是行为归纳模型，未从框架源码逆向验证；其他 CANN 版本 / arch 未验证，推断可迁移。
- 早期"签名交换（flag 仍 true）后最小写探针通过"与"host plog 的 AIC_INFO `args(3)=0x12004c600000` 与 host workspace 指针同值"两组观测存在未调和的矛盾（第 4 参只读身份未定位）；终态修复（flag=false + 标准签名）不依赖该中间结论。
- kernel 二进制 `parameters=[null,null,null,null]`（4 项）在修复前后都存在，属正常元数据形态（tiling 走 opParaSize 通道），不能单独作为"参数缺失"的证据。
- host 侧 `sizeof(TilingData)=488` 与 kernel 二进制 `opParaSize=496` 的 8 字节差异不是决定因素（修复后 opParaSize 仍 496 而 tiling 已正确读取），其来源未验证。
- 修复后 ST Real 仍失败（ret=507015，AIC 侧）——该独立根因不在本卡范围。

**未决项**：

- `DynamicCompileStaticFlag(true)` 下框架传参的精确机制（4 GM 参数 + opPara 的布局与只读属性）未从框架源码验证，本卡模型来自行为归纳——复现并追根因（读 op_executor/CommonOpExecutorRun 的实参装配代码）是优先接续项。
- host 488 vs opParaSize 496 的 8 字节差异来源未追。
- 修复后 ST Real 仍失败（ret=507015，AIC 侧独立根因）——其微架构层根因未定位。
- `AscendC::printf` 在该环境不可见的机制（未追因）。
## 坏实践

- 把第 4 形参当 workspace 起点用：`AscendC::SetSysWorkspace(tiling); GM_ADDR user = AscendC::GetUserWorkspace(tiling);`——静态模式下 tiling 形参收到的不是 workspace，生成阶段全部写空（本卡 before 状态）。
- 在 tiling 里盲目给 workspace 加 1MB / 2MB 预留试图"绕过"系统 workspace——对本问题无效（实测 workspace 从 16MB 涨到 17MB 仍全 0），且会连带 UT 期望值失效。
- 只改 flag 不改签名（或反之），然后凭"没报错"判断已修复——两个方向单独改实测都无效。
- 用"workspaceSize 申报正确、aclrtMalloc 成功"推断 kernel 侧一定能写。
- 输出全 0 时先怀疑计算逻辑（Mock 已排除却仍反复检查 kernel 数学）；在参数层根因未解决前堆叠无关修复（参数顺序/预留/SetOrgShape/blockIdx 映射——全部无效尝试）。
- 用 `AscendC::printf` 做该环境的 kernel 调试（实测 stdout/plog 均不可见，纯耗时）。

## 误诊

- "AIC 的 blockIdx 编号问题（MIX 下 AIC/AIV 编号不配对）"——试验 `% coreNum` 映射无效，且探针 4 直接证明 AIV 预写值未被任何核覆盖。
- "CopyND2NZOnTheFly 在 AIC 上没搬数据"——被跳段对照排除（跳过搬运仍不改变全 0 形态的根因层）。
- "REGIST_MATMUL_OBJ 位置不对"——移位后无变化（该调整本身正确、属 swin 模式要求，但非本问题根因）。
- host 488 vs opParaSize 496 的 8 字节差异曾被当作根因（怀疑 host/kernel 结构体对齐不一致），实际修复后该差异仍在而 tiling 已正确。
- `SetSysWorkspace(tiling) + GetUserWorkspace(tiling)`（把 tiling 当 workspace）曾因"不再触发 AIV error"被误认为接近修复——不报错只是因为不再向只读区写入，输出仍全 0。
- `AscendC::printf` 不可见曾浪费多轮探针轮次——该环境应改用 GM 探针。
