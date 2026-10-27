---
type: "Runbook"
title: 自主 AscendC kernel 静态核查配方
description: 算子交付验收时用"反向排除 grep（逐条核对命中行，警惕算子名子串假阳性）+ 正向原语确认 + 编译产物 .o 校验 + 调度链/host 纯度审查"静态核查是否为自主 AscendC kernel（而非 torch 封装/转调已有算子）以及计算逻辑是否全在 kernel 内；两条独立轨迹的验收核查均一次通过。R3 增量：补动态执行核验——重建 ST 二进制经 LD_LIBRARY_PATH 直连 vendors 包真机跑用例，执行用例参数与 CSV 行逐条对照（防程序内置假用例）、CSV 计数去重、白盒用例存在性检查。
tags: [task.testing, topic.architecture, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# 自主 AscendC kernel 静态核查配方

## 现象

两个独立轨迹的算子交付验收都提出了同样的人工核查要求（明确禁止修改已有代码、仅读取分析）：

1. 轨迹一（gather 类算子，PyTorch torch.gather 语义，Ascend950PR/arch35，CANN 9.1.0）：算子已走完四阶段全流程并合并回 master——UT 62/62 通过、`bash build.sh --soc=ascend950 -j8` 全量构建成功、精度验收全量 Bitwise Match、代码检视问题已修复。用户在交付验收前要求核查：是否确实开发了 AscendC 算子（而不是用 torch 接口走已有算子实现）？是否所有计算逻辑都在 kernel 内完成（而不是在 host 上实现）？
2. 轨迹二（grid_sampler_3d 算子，ascend950/arch35，工程与编译产物齐备）：同样的"是否手写 AscendC kernel 而非 torch 包装"检查。该轨迹的第一个动作直接 grep torch 相关关键词，在 op_kernel/op_host/op_api 三个目录分别得到 9、10、7 处"命中"——只看命中数很容易误判成"存在 torch 依赖"，逐条看命中行才发现全部是文件名和注释里的算子名子串（算子名 bench_no_harness_grid_sampler_3d 本身包含 "grid_sample"）。

交付文档（LOG、设计文档、检视报告）都不能直接回答"自主实现"与"kernel 内计算"这两个属性——需要一套只读、可复现、有明确判读标准的静态检查步骤。两次核查结论均为通过，检查过程与判读标准沉淀如下。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke 类流程）的交付验收 / 人工核查 / 静态代码审计；核查本身不执行算子、不修改代码。
- API 路径：AscendC kernel（`__global__ __aicore__` 入口、REGISTER_TILING_DEFAULT / GET_TILING_DATA(_WITH_STRUCT)、DataCopyPad / GetValue / SetGlobalBuffer / GlobalTensor / LocalTensor / TBuf 原语）、op_host TilingFunc / InferShape / IMPL_OP_OPTILING、op_api L2（aclnn 两段式）与 L0（ADD_TO_LAUNCHER_LIST_AICORE）、PyTorch 适配层（torch library 注册 + aclnn 调用）；编译产物 `build/op_kernel/ascendc_kernels/binary/<soc>/<op>/<OpType>_<hash>.o`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（被核查算子在同一开发轨迹内以 `--soc=ascend950` 完整构建并通过 UT，核查本身为源码级静态检查）。
- 适用粒度：workflow_only——检查配方作用于"算子仓交付验收"这一工作流，与具体算子语义无关。
- 路线优先级：recommended_fastpath（交付验收的快速核查通道；不能替代 UT/ST 精度与性能实测）。
- 明确不适用 / 未验证场景：不验证数值正确性、精度或性能（静态检查无法覆盖，需 ST/评测实测）；不覆盖运行时动态行为（如框架把调用路由到其他实现、运行时 fallback）；字符串拼接后 dlopen 动态加载 torch 符号的形态 grep 探针会漏；aclnn L2 内调用 `l0op::Contiguous` / `l0op::ViewCopy` 等通用格式处理原语属合法操作，不视为"用已有算子实现语义"；未在多分支 L0 调度（按 shape 分流到不同实现）的算子上验证过。

## 判别方法

四步静态核查，全部只读，逐步给出判读标准。

**第 1 步：反向排除——grep 已有实现/封装痕迹（必须逐条核对命中行）**

对 `op_kernel/`、`op_host/`、`op_api/` 三个目录分别执行（模式按目标算子语义替换已有算子名；应同时覆盖 torch C++ 命名空间（`at::`/`aten`）、Python 侧名称与该算子对应的 torch 函数名）：

```
grep -R "torch\.gather|torch_npu|aclop|ACLLNN_GATHER|aclnnGather" op_kernel/
grep -R "torch\.gather|aclop|ACLLNN_GATHER|aclnnGather" op_host/
grep -R "torch\.gather|aclop|ACLLNN_GATHER|aclnnGather" op_api/
```

判读标准：期望零命中。`op_kernel/` 命中 torch_npu/aclop 说明 kernel 侧混入框架调用；`op_api/` 命中 ACLNN_GATHER/aclnnGather 说明 L2 实现在转调已有 gather 算子（封装而非自主实现）。`op_host/` 的算子定义文件注释里出现"语义: PyTorch torch.gather"属于语义说明文字，不算调用。

**警惕算子名子串假阳性**：若算子名本身含 torch API 名子串（如 grid_sampler_3d ⊃ grid_sample），grep `torch|grid_sample|aten|at::|pytorch` 会在三个目录各命中若干处（实测 9/10/7），逐条打开命中行核对——全部为 `* \file ..._grid_sampler_3d_*.cpp` 注释行与 `#include "..._grid_sampler_3d.h"` 文件名行，没有任何一处是 torch API 调用。**只看命中数就下"存在 torch 依赖"的结论必误判。**

**第 2 步：正向确认——kernel 入口形态与 AscendC 原语**

```
grep -R "DataCopyPad|DataCopy|GetValue|SetGlobalBuffer|GlobalTensor|LocalTensor|__aicore__|__global__|TPipe|TQue|Cast|Duplicate" op_kernel/
```

判读标准：期望大量真实命中（轨迹一核查 47 处；轨迹二核查 67 处，覆盖 `__global__ __aicore__` 入口、kernel 类全部 `__aicore__ inline` 方法、DataCopyPad GM↔UB 搬运、SetGlobalBuffer、GetValue、SetFlag/WaitFlag 管道同步等）。并打开 kernel 入口文件确认标准 AscendC 骨架（以下为被核查算子的入口，verbatim）：

```cpp
template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void bench_no_harness_gather(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGatherTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGatherTilingData, tilingData, tiling);
    NsBenchNoHarnessGather::BenchNoHarnessGather<D_T_X, BUFFER_MODE> op;
    op.Init(x, index, y, &tilingData);
    op.Process();
}
```

`__global__ __aicore__` 入口 + REGISTER_TILING_DEFAULT/GET_TILING_DATA(_WITH_STRUCT) 取 tiling + kernel 类 Init/Process 是自主 AscendC kernel 的标准编程模式；kernel 类内应看到原语的**实质使用**（而不是只 include 头文件）。

**第 3 步：编译产物校验（排除"只有源码、没有 kernel 二进制"的 stub 形态）**

```
find build -name '*.o' -path '*/binary/*' 2>/dev/null | head -10
```

判读标准：`build/op_kernel/ascendc_kernels/binary/ascend950/<规范化算子名>/` 下存在 `BenchNoHarness*_<hash>.o` 二进制（轨迹二核查返回 2 个，对应 TilingKey 模板实际实例化的 dtype × 模式组合）。只查源码不查编译产物时，"源码里写了 kernel 但从未编译出二进制"的 stub 形态会漏判；产物路径形态为 `binary/<soc>/<op>/<OpType>_<hash>.o`。

**第 4 步：调度链审查 + host 纯度检查——L0 注册、torch 适配层与计算位置**

- L0 API（`op_api/{op}.cpp`）应通过 `ADD_TO_LAUNCHER_LIST_AICORE(算子名, ...)` 把算子挂到 kernel 调度器（调度的是自主开发的 kernel，而非已有算子）；
- PyTorch 适配层（`tests/st/torch/torch_adapter.cpp`）应调用本算子的 aclnn 两段式接口（`aclnn{Op}GetWorkspaceSize` + `aclnn{Op}`）并通过 torch library 注册自定义 op，而不是直接调 `torch.gather` / `at::gather`。torch_adapter 与 golden.py 属测试侧（适配层调 ACLNN 两段式接口、参考实现做精度比对），不算算子本体的 torch 依赖；
- host 纯度（计算是否全在 kernel）：`op_host` TilingFunc 只做平台信息获取（UB 大小/核数）、多核切分（blockFactor）、UB 切分（ubFactor）、TilingData 填充、BlockDim/TilingKey 设置——全部是调度参数，不触碰输入数据值；host 预计算 stride 传给 kernel 属参数传递，实际地址计算在 kernel 侧完成；InferShape 只做形状/dtype 推导（输出 shape = index shape）；op_api L2 只做参数校验 + Contiguous 处理 + L0 调度 + ViewCopy。tiling 中的步长预计算（hwStride=H*W 等）属合法切分参数、不算数据计算（host 侧逐文件按"是否触碰输入数据值"白名单核对）。
- 辅助判据：workspace 大小为 0（`WS_SYS_SIZE = 0`）说明无 host 侧中间数据暂存，全部数据通路是 kernel 内 GM↔UB 直接完成。

**判别结论规则**：四步全部满足——①torch/已有算子模式无 API 调用命中（命中全为文件名/注释子串，逐条核对后定性）＋②AscendC API 特征大量真实命中＋③`binary/` 下存在 .o 二进制＋④调度链（L0 注册 + torch 适配层）指向自主 kernel 且 host 侧无数据计算——即可判定为"真实自主 AscendC 实现 + 计算逻辑全在 kernel"；任一步不过 → 定位到具体文件交回开发修复后复核。

## 设计决策

- 为什么用"反向排除 + 正向确认"双通道而不是只看其一：只做反向 grep，空的算子目录或纯头文件壳也能零命中，不能证明有实现；只做正向原语命中，拷来的样例代码也能命中，不能排除同时转调已有算子。因此以"无已有算子/torch 调用（排除）∧ 有 AscendC kernel 骨架与原语实质使用（确认）∧ 编译产物存在（落地）∧ 调度链与 host 纯度闭环（归属）"四重证据下结论。
- 为什么一步 grep 不够、必须配 .o 产物校验：防止"源码里写了 kernel 但从未编译出二进制"的 stub 形态骗过纯源码检查。
- 命中数不能直接下结论：算子名含 torch API 名子串时第一步必然出现文件名假阳性，必须逐条核对命中行内容后再定性。
- "kernel 内计算"单独作为第二问独立核查（host 纯度 + workspace=0 判据），不与第一问混判——一个算子可能自主实现却在 host 上做数据计算，两者是独立缺陷。
- 该配方定位为交付验收 fastpath：成本是几次 grep + find + 少量文件阅读，不上板；深度验证（精度/性能/运行时行为）仍走 ST/评测通道。
- 核查时禁止顺手"修复"发现的问题——交付验收核查应只读；发现问题走独立的修复-回归通道（两次核查全程未修改任何代码）。

## 增量 R3：动态执行核验（第 5 步，RoPE 类算子只读交付审计，独立 trace，Ascend950PR + CANN 9.1.0）

静态四步之后补"用例真实执行且与 CSV 一致"的动态核验，闭合本卡遗留的"运行时执行路径核验未做"项。全程只读（不改仓内代码与环境），审计命令序列（逐字取自审计执行记录；算子目录以实际 `<op>` 替换，`$ASCEND_HOME_PATH` 指向 CANN 安装根）：

**5a) CSV 用例计数（区分两套格式并去重）**：`tests/st/testcases/` 同时存在简化格式（`L0/L1/L2_test_cases.csv`）与 TTK 格式同名 CSV，内容一一对应；不去重会把 639 条报成 1278 条。逐文件统计数据行数（排除 header）后按内容去重合计。

**5b) ST 二进制重建 + 真机执行（LD_LIBRARY_PATH 直连 vendors 包）**：

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/:${LD_LIBRARY_PATH} && \
cd operators/<op>/tests/st && rm -rf build-real && mkdir -p build-real && cd build-real && \
cmake .. -DUSE_MOCK=OFF 2>&1 | tail -5 && make -j8 2>&1 | tail -5
```

```bash
# L0 全量
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/:${LD_LIBRARY_PATH} && \
cd operators/<op>/tests/st/build-real && \
./test_aclnn_<op> --level L0 2>&1 | \
  grep -E '\[Real\]|\[PASS\]|\[FAIL\]|总计|通过:|失败:|通过率|加载测试'

# L1 子集（分批隔离，规避同进程连续大批量 aclnn 调用的 CANN 运行时资源耗尽）
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/:${LD_LIBRARY_PATH} && \
cd operators/<op>/tests/st/build-real && \
./test_aclnn_<op> --case ../testcases/L1_subset_50.csv 2>&1 | \
  grep -E '\[PASS\]|\[FAIL\]|总计|通过:|失败:|通过率|加载测试'
```

**5c) 执行用例 ↔ CSV 参数逐条对照（防"程序内置假用例"的关键一步）**：把程序输出的每条 `[Real]` 用例参数（B/S/N/D/cos/layout/mode/dtype）与 CSV 行逐一比对，并用 `head`/`tail + cut -d, -f1` 校验子集 CSV 的 case_id 与全量 CSV 前缀一致——通过率 100% 可能来自程序内置假用例或宽松比较器，只有"执行用例参数 == CSV 行参数"才能证明 CSV 用例被真实执行：

```bash
./test_aclnn_<op> --level L0 2>&1 | grep '\[Real\]' | head -3 && \
echo "---" && head -3 operators/<op>/tests/st/testcases/L0_test_cases.csv

# 子集 case_id 与全量 CSV 前缀一致性
tail -n +2 operators/<op>/tests/st/testcases/L1_subset_50.csv | cut -d, -f1 | head -5 && \
echo "=== L1_test_cases case IDs (first 5) ===" && \
tail -n +2 operators/<op>/tests/st/testcases/L1_test_cases.csv | cut -d, -f1 | head -5
```

**5d) 白盒用例存在性**：glob `tests/whitebox/**` 并 `ls tests/` 核对实际目录（本例不存在，白盒 0 条，如实记录）。另一独立轨迹的 CSV↔ST 对账反例（CSV 520 条中 C++ ST 实际只执行 115 条、L1/L2 CSV 未被加载）见[算子交付审计：验证真 AscendC kernel 与 ST 实测覆盖卡](operator_delivery_audit_csv_st_reconciliation.md)——对账步骤相同，结论形态相反（本例对得上，该例对不上即交付缺陷）。

R3 实测边界（审计中发现并如实记录，不擅自修复）：

- L2 异常用例（多值 dtype 字段，如 `float32/float16/float32/...`）加载即崩溃：`terminate called after throwing an instance of 'std::invalid_argument' what(): stoll`——简化格式 CSV 解析器（简单行分割 + stoll 数值解析）不支持多值 dtype；24 条 L2 未执行，异常路径校验只能依赖 UT（交付方声明，本次未逐一核验）。
- L1 全量 600 条未单进程完整执行：同进程连续大批量 aclnn 调用存在 CANN 运行时资源耗尽问题（开发过程记录于算子仓 issues，非 kernel 缺陷），审计采用 ≤50 条/进程分批规避；其他环境是否复现未验证。
- 审计报告引用的"L2 校验已由 UT 覆盖（38 条）"为交付方文档转述，未逐一执行核验——审计结论只采信实测部分。
- 白盒用例目录不存在是否符合交付标准需对照流程要求确认（本审计仅如实记录）。

R3 验证信号：

- L0 全量真实 NPU：15/15 通过，全部 MERE=0.00e+00、MARE=0.00e+00（threshold 按 dtype：fp32 1.22e-04 / fp16 9.77e-04 / bf16 7.81e-03；最大单用例 12,829,600 elems）（来源平台：950；证据等级：tool_observed）。
- L1 子集（全量 CSV 前 50 条，独立进程）：50/50 通过，全部 MERE=0.00e+00（来源平台：950；证据等级：tool_observed）。
- 执行用例与 CSV 一致：程序输出 `[Real] L0_001 (B=8,S=191,N=1,D=4,cos=2D,layout=0,mode=0,dtype=float32)` 与 CSV 行 `L0_001,8,191,1,4,2D,0,half,float32,...` 逐字段一致；L1_subset_50 的 case_id（L1_001~L1_050）与全量 L1 CSV 前 50 条完全一致（来源平台：950；证据等级：tool_observed）。
- CSV 计数：L0=15、L1=600、L2=24（去重合计 639）；TTK 格式同名 CSV 一一对应不另计（来源平台：950；证据等级：tool_observed）。
- L2 异常用例加载即崩溃：`std::invalid_argument: stoll`（24 条未执行）；`tests/whitebox` 不存在，白盒 0 条（来源平台：950；证据等级：tool_observed）。

R3 坏实践增补：只看"通过率 100%"就采信交付——必须做执行用例参数 ↔ CSV 行的逐条对照；用例计数不去重多套同名 CSV（简化 + TTK），把 639 条报成 1278 条；把 host 侧 tiling 的切分参数计算（bsPerCore/bsTail）误判为"计算逻辑在 host"（判据是算子数学所在层级）。

## 完整代码

核查用完整命令序列（在算子仓根执行，全部只读；模式中的已有算子名按目标语义替换）：

```bash
# 第 1 步：反向排除（逐条核对命中行，警惕算子名子串假阳性）
grep -R "torch\.gather|torch_npu|aclop|ACLLNN_GATHER|aclnnGather" op_kernel/
grep -R "torch\.gather|aclop|ACLLNN_GATHER|aclnnGather" op_host/
grep -R "torch\.gather|aclop|ACLLNN_GATHER|aclnnGather" op_api/

# 第 2 步：正向原语确认
grep -R "DataCopyPad|DataCopy|GetValue|SetGlobalBuffer|GlobalTensor|LocalTensor|__aicore__|__global__" op_kernel/

# 第 3 步：kernel 二进制产物
find build -name '*.o' -path '*/binary/*' 2>/dev/null | head -10

# 第 4 步：文件审查（只读）
#   op_kernel/*_arch35.cpp        —— __global__ __aicore__ 入口 + REGISTER_TILING_DEFAULT
#   op_api/{op}.cpp               —— ADD_TO_LAUNCHER_LIST_AICORE 注册
#   op_api/aclnn_{op}.cpp         —— L2 仅校验/Contiguous/调度/ViewCopy
#   op_host/arch35/*_tiling.cpp   —— 仅 tiling 参数（核对 WS_SYS_SIZE 是否为 0）
#   tests/st/torch/torch_adapter.cpp —— aclnn 两段式调用而非 torch.gather
```

## 验证信号

以下信号来自两条独立轨迹的交付验收核查（全部为对算子仓源码与编译产物的只读 grep/find 与文件审查，核查结论落盘为仓库根 MANUAL_CHECK.md 供复核）。

轨迹一（gather 算子）：

- 反向 grep（`torch\.gather|torch_npu|aclop|ACLLNN_GATHER|aclnnGather`）：`op_kernel/` 0 命中、`op_api/` 0 命中、`op_host/` 仅算子定义文件注释 1 处（语义说明）（来源平台：950；证据等级：tool_observed）。
- 正向 grep（`DataCopyPad|DataCopy|GetValue|SetGlobalBuffer|GlobalTensor|LocalTensor|__aicore__|__global__`）：`op_kernel/` 命中 47 处，含 kernel 入口 `__global__ __aicore__`、DataCopyPad GM↔UB 搬运、SetGlobalBuffer、GetValue、SetFlag/WaitFlag 管道同步等（来源平台：950；证据等级：tool_observed）。
- 被核查算子在同一开发轨迹内 `bash build.sh --soc=ascend950 -j8` 完整构建成功且修复后 UT 62/62 通过（来源平台：950；证据等级：tool_observed）。
- PyTorch 适配层源码审查确认调用 `aclnn{Op}GetWorkspaceSize` + `aclnn{Op}` 两段式接口（来源平台：950；证据等级：tool_observed）。
- host 纯度：TilingFunc/InferShape/L2 逐文件审查无数据计算逻辑，`WS_SYS_SIZE = 0`（来源平台：950；证据等级：tool_observed）。

轨迹二（grid_sampler_3d 算子）：

- grep torch 模式（`torch|grid_sample|aten|at::|pytorch`）在 op_kernel / op_host / op_api 分别命中 9 / 10 / 7 处，逐条核对全部为文件名/注释中的算子名子串（bench_no_harness_grid_sampler_3d 含 "grid_sample"），无任何 torch API 调用（来源平台：950；证据等级：tool_observed）。
- grep AscendC API 模式（`__aicore__|DataCopyPad|GlobalTensor|LocalTensor|TPipe|TQue|Cast|Duplicate`）在 op_kernel 命中 67 处，含 `__global__ __aicore__` 入口与 kernel 类全部计算方法（来源平台：950；证据等级：tool_observed）。
- find 产物校验返回 2 个 kernel 二进制 `build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_grid_sampler3d/BenchNoHarnessGridSampler3d_*.o`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 静态检查只证明"源码形态是自主 AscendC 实现 + 编译产物存在"，不证明运行时行为：若框架在运行时把调用路由到其他实现（未观测到此类行为，未验证），本配方无法发现，需上板 profiling/精度对比验证。
- "计算逻辑在 kernel 内"的判定依赖人工阅读 host 侧代码：若把数据计算伪装成"参数预计算"（例如在 TilingFunc 里遍历输入数据值），grep 无法判定，需审查者按"是否触碰输入数据值"逐段判别。
- 混合实现（部分 shape 走自主 kernel、部分转调已有算子）需要检查 L0/L2 的全部分支；核查过的算子 L0 仅一条 ADD_TO_LAUNCHER_LIST_AICORE 路径，多分支场景未验证。
- grep 模式需按目标算子语义调整：核查非 gather 语义算子时，应把 ACLNN_GATHER/aclnnGather 换成对应语义的已有算子名（如 scatter 语义换成 ACLNN_SCATTER/aclnnScatter 等），否则反向排除会漏检。
- .o 数量与 TilingKey 模板划分的对应关系未逐符号核验（推断为实际实例化的组合，未验证）；其他 CANN 版本 / SOC 的产物布局未验证，推断可迁移。

**未决项**：

- 运行时执行路径核验未做：接续时优先补上板 profiling（确认 AICore 侧实际执行该 kernel、无框架 fallback 到 CPU/其他算子）或精度对比信号。
- 其他 CANN 版本 / 其他 SOC 的 build 产物布局（`binary/<soc>/` 路径形态）未验证，推断可迁移。
## 坏实践

- 只看交付文档 / 开发日志声称"已开发 AscendC 算子"就验收——文档声明不构成实现证据。
- 只看 grep 命中数就下"存在/不存在 torch 依赖"的结论——算子名含 torch API 名子串时（本例 grid_sampler_3d ⊃ grid_sample）文件名/注释假阳性必然出现。
- 只查源码不查编译产物——"源码有 kernel 但从未编译出 .o"的 stub 形态会漏判。
- 只确认"存在 `__aicore__` 函数"就下结论——必须同时反向排除已有算子转调与 torch 封装，并审查调度链闭环。
- 核查时顺手"修复"发现的问题——交付验收核查应只读。

## 误诊

- （轨迹二 agent 侧真实发生过的险情）grep `torch|grid_sample|...` 命中 9~10 处时，若直接按命中数判"有 torch 依赖"即误诊；实际全部命中为算子名子串的文件名/注释行（排除过程为 agent_reported，最终以逐条核对收敛，核对结果为 tool_observed）。
