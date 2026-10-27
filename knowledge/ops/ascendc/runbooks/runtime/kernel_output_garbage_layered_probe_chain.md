---
type: "Runbook"
title: kernel 垃圾值分层探针：常量写入→回读→向量拷贝→参数回显
description: AscendC kernel 输出含垃圾值时的分层定位法：常量写入探针验证分支/写通路，tiling 参数回显验证传参，GetValue 标量回读与 Adds 向量拷贝逐层判定上游 buffer 是否被污染；UB→UB 拷贝不能用 DataCopyPad。本方法在 k==E 垃圾值排查中把根因从"TopK API bug"修正为"SoftMax 输出 buffer 污染"。R5：标量输出错值时补"中间值直写"步，两步二分把故障隔离到输出通路/计算段/被跳过段。
tags: [task.debugging, topic.implementation.debugging, topic.memory]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# kernel 垃圾值分层探针：常量写入→回读→向量拷贝→参数回显

## 现象

（方法适用场景还原）在 Ascend950PR（arch35，CANN 9.1.0）上调一个 SoftMax→TopK 门控算子：k==E（TopK 选取全部元素）的部分用例输出在尾部位置出现 14992.0/12752.0 量级的垃圾值，MERE 达 5.53e+07。这类"部分位置错、部分位置对、错值像未初始化数据"的精度问题，归因候选很多——分支没执行？写通路坏了？TopK API 有 bug？SoftMax 输出错？上游 DataCopy 错？——盲试修复（本例五种 workaround 全无效）代价极高。本卡沉淀当时走通的**分层探针链**：不改算法语义，只在可疑位置插入确定性探针，一次一轮把问题压到具体层。

## 适用场景

- 工程形态：AscendC kernel 精度联调（输出部分位置垃圾值/错值，需要定位污染层）。
- API 路径：`Duplicate`/`SetValue`（常量写入）、`GetValue`（标量回读）、`Adds`（向量拷贝探针）、`DataCopy`（32B 对齐 UB→UB）、`PipeBarrier<PIPE_ALL/PIPE_V>`（探针间隔步）；`ASCEND_SLOG_PRINT_TO_STDOUT=1`（plog）；单用例 CSV 传参复现。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——AscendC kernel 通用排障方法。
- 路线优先级：recommended_fastpath（垃圾值类问题的第一反应动作）。
- 明确不适用 / 未验证场景：崩溃类（507035/段错误）问题应先走 plog 错误码路线；探针法要求输出张量能被测试程序数值化打印比对。

## 判别方法（探针链，自上而下）

**前置：单用例复现。** 从用例 CSV 抽出一条失败用例写成独立 CSV，直接传给测试可执行文件，缩短每轮验证周期，并开 plog：

```bash
mkdir -p /tmp/debug_cases
echo "case_id,desc,x_shape,x_dtype,finished,k,y_dtype,threshold" > /tmp/debug_cases/fail_case.csv
echo "fail_case,失败用例描述（照抄原用例参数）,\"[1024,64]\",fp16,none,64,fp16,0.0009765625" >> /tmp/debug_cases/fail_case.csv
cd tests/st/build-real && ASCEND_TOOLKIT_HOME=<toolkit> \
LD_LIBRARY_PATH=<toolkit>/lib64:<vendor>/op_api/lib:$LD_LIBRARY_PATH \
ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test_<op> /tmp/debug_cases/fail_case.csv
```

**第 1 层：常量写入探针（验证"分支执行 + 写通路"）。** 在可疑分支开头写确定性常量到输出：

```cpp
if (isKEqE_ == 1) {
    // DEBUG 探针: 常量写入验证分支执行
    PipeBarrier<PIPE_ALL>();
    Duplicate<COMPUTE_T>(yCompute, static_cast<COMPUTE_T>(1.0),
                         static_cast<uint32_t>(curBatch * kPad_));
    PipeBarrier<PIPE_V>();
}
```

输出如期变常量（本例全 1.0，MERE 从 5.53e+07 跳到 1.37e+04 的"全错但错得整齐"形态）→ 分支在执行、写通路完好；输出纹丝不动 → 先去查构建/安装/分支条件（md5sum + 探针双确认 kernel 修改生效性）。

**第 2 层：tiling 参数回显（验证传参）。** 把 tiling 下发的关键字段当数据写到输出头部：

```cpp
yCompute.SetValue(0, static_cast<COMPUTE_T>(isKEqE_));
yCompute.SetValue(1, static_cast<COMPUTE_T>(k_));
yCompute.SetValue(2, static_cast<COMPUTE_T>(E_));
yCompute.SetValue(3, static_cast<COMPUTE_T>(topkInner_));
```

回读到期望值（本例 1/64/64/64）→ tiling 传参与分支条件都对；回读错 → 查 TilingData 结构布局（新增字段会改变 opParaSize 与对齐）。

**第 3 层：标量回读探针（验证"读通路 + 逐元素数据"）。** 把上游 buffer 的元素搬 to 输出：

```cpp
for (int64_t i = 0; i < 8; i++) {
    yCompute.SetValue(static_cast<int32_t>(i), softmaxDst.GetValue(static_cast<int32_t>(i)));
}
```

本例结果：位置 0~2 读到 0.000321/0.021835/0.102112（错但量级正常），位置 3 直接读到 **12752.0** ——垃圾值在进入本算子逻辑之前就已经在 softmaxDst 里。

**第 4 层：向量整块拷贝探针（排除标量读的个体差异）。** 用向量算术整块拷上游 buffer，绕开一切标量与后续 API：

```cpp
// UB → UB 整块拷贝: Adds + 0（不能用 DataCopyPad，仅支持 GM↔UB，编译报 no matching function）
Adds<COMPUTE_T>(yCompute, softmaxDst, static_cast<COMPUTE_T>(0),
                static_cast<uint32_t>(curBatch * topkInner_));
```

本例结果：MERE=1.05e+06，仍是垃圾 → **污染在 SoftMax 输出 buffer 本身**，与 TopK、与标量读都无关，归因方向从"TopK API bug"修正为"SoftMax 输出 buffer 污染"。

**收尾**：探针全部移除、重建重装、回归验证后交付。

### 增量 R5：标量输出错值的"中间值直写"二分（ForeachNorm 轨迹，独立 trace）

适用变体：单/少元素标量输出的 kernel 数值错误（非垃圾值形态），且多轮"合理猜想式"修复后输出值一字不变——"值不变"本身说明改动很可能不在故障路径上。两步旁路，每步只改一处、每步都有可读回的观测值：

**第 0 步（先看盲修信号）**：若多轮修复后输出值完全不变，优先怀疑"改动不在故障路径/输出从未收到计算值"，停止叠加盲修，进入旁路。

**第 1 步（常数旁路，验证输出通路）**：在 kernel 输出位置直接写死一个特征常数，走最小已知原语链（`SetValue` → `S_MTE3` 同步 → `DataCopyPad`），绕开全部计算（verbatim）：

```cpp
        // DEBUG: 完全跳过计算，直接写 42.0 到输出
        {
            LocalTensor<D_T_X> outLocal = inQueue_.template AllocTensor<D_T_X>();
            outLocal.SetValue(0, D_T_X(42.0f));

            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

            __gm__ D_T_X* outGM = GetTensorAddr(static_cast<uint16_t>(k), yPtr_);
            GlobalTensor<D_T_X> outTensorGM;
            outTensorGM.SetGlobalBuffer(outGM, 1);

            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(sizeof(D_T_X)), 0, 0, 0};
            DataCopyPad(outTensorGM[0], outLocal, outCopyParams);

            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

            inQueue_.FreeTensor(outLocal);
            continue;
        }
```

读回 `actual == 42.000000` ⇒ 寻址、S→MTE3 同步、DataCopyPad、ST 读回全链可用；`actual != 42` ⇒ 故障在输出通路，转查寻址/同步/搬出。常数选 42.0 这类与任何可能计算结果（0、输入值、和、平方和）都无碰撞风险的特征值；不要用输入数据本身（如 x[0]）做旁路常数——与计算结果可能碰撞，无法判定来源。

**第 2 步（中间值直写，验证计算链）**：保留真实计算，跳过可疑收尾段（该例为 Sqrt/Cast），把中间量（合并和 `mergedResult`）经同一已验证通路直写：

```cpp
        // DEBUG: 跳过 Sqrt/Cast，直接写 mergedResult 到输出
        {
            LocalTensor<D_T_X> outLocal = inQueue_.template AllocTensor<D_T_X>();
            outLocal.SetValue(0, D_T_X(mergedResult));
            // …同上 S_MTE3 同步 + DataCopyPad…
        }
```

该例读回 `actual = 1024.528320`，而 golden=32.008255 且 `32.008255² ≈ 1024.528` ⇒ 输入搬运、平方、归约、跨核合并全部正确；故障隔离在 Sqrt/Cast 收尾段（该收尾段的确切修复见[单标量输出向量链丢值卡](../precision/scalar_output_v_mte3_loss_readback.md)）。

**第 3 步（结论规则）**：
- 第 1 步通过 + 第 2 步中间值正确 ⇒ 故障在"被跳过的收尾段"；
- 第 1 步通过 + 第 2 步中间值错误 ⇒ 故障在计算/合并段；
- 第 1 步不通过 ⇒ 故障在输出通路（寻址/同步/搬出），先修输出再做第 2 步。

第 2 步复用第 1 步验证过的原语链，把"写 GM"固定为已知好通道，差值就只归因于被跳过的段——这是二分法的"控制变量"。不在 kernel 里 printf：AscendC kernel 侧没有现成日志通道时，标量读回（ST 的 D2H 比对）就是最便宜的"kernel 打印"。

## 设计决策

- 为什么每层探针都配 `PipeBarrier`：探针写入与被测数据通路跨流水（V/MTE），不加屏障可能读到乱序旧值，把探针本身变成噪声源。
- 为什么第 4 层用 `Adds +0` 而不是 `DataCopyPad`：实测 `DataCopyPad` 没有 UB→UB 重载（编译错 `no matching function for call to 'DataCopyPad'`）；`DataCopy` 可用但要求 32B 对齐，`Adds +0` 对任意元素数最稳。
- 为什么强调"一轮一探针"：多探针叠加会让失配输出无法归层；本链的每一层都只回答一个二元问题。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510/arch35），CANN 9.1.0。
- 相关文件：`op_kernel/arch35/<op>.h`（探针插入点）、`tests/st/build-real/test_<op>`（接受 CSV 路径参数）、`tests/st/testcases/*.csv`（单用例抽取源）。
- 每轮循环：改探针 → `bash build.sh --soc=ascend950` → 卸载旧包（vendor `scripts/uninstall.sh`）→ 安装 run 包 → 单用例 CSV 复跑。

## 预防

- 精度问题排查先画数据流（GM→UB→SoftMax→TopK→Cast→GM），对每个中间 buffer 预设"直拷到输出"的探针位。
- 探针代码用显眼注释标记（DEBUG），交付前 `grep` 清零。

## 验证信号

以下信号来自该算子 k==E 垃圾值的排查会话，每层探针均经重建重装后单用例验证。

- 第 1 层：常量 Duplicate/SetValue 探针使输出变为常量（全 1.0；另一轮 1.0/2.0/3.0 序列），MERE 形态改变（5.53e+07 → 1.37e+04 / 1.04e+00）（来源平台：950；证据等级：tool_observed）。
- 第 2 层：回显 isKEqE_=1、k_=64、E_=64、topkInner_=64，全部符合期望（来源平台：950；证据等级：tool_observed）。
- 第 3 层：GetValue 回读 softmaxDst 位置 3 = 12752.000000（期望 0.123693，rel_err=1.03e+05）（来源平台：950；证据等级：tool_observed）。
- 第 4 层：Adds+0 整块拷贝 softmaxDst→yCompute，MERE=1.05e+06 / MARE=1.74e+09，垃圾仍在（来源平台：950；证据等级：tool_observed）。
- API 事实：`DataCopyPad` 无 UB→UB 重载，编译报 `no matching function for call to 'DataCopyPad'`（来源平台：950；证据等级：tool_observed）。

R5 增量信号（ForeachNorm 标量输出错值会话，独立 trace，三步全部实际执行）：

- 触发：L0_001 `golden=32.008255, actual=0.092471`，三轮盲修（单核/跳 Cast/同步重排）actual 不变——"值不变"即改动不在故障路径的信号（来源平台：950；证据等级：tool_observed）。
- 探针 1（常数直写 42.0）：`actual=42.000000`——输出通路正确（来源平台：950；证据等级：tool_observed）。
- 探针 2（跳过 Sqrt/Cast 直写 mergedResult）：`actual=1024.528320`（= golden²，1024 元素平方和）——计算/合并正确，故障锁定 Sqrt/Cast 收尾段（来源平台：950；证据等级：tool_observed）。
- 后续按该结论修复收尾链后 L0_001 `MERE=0.00e+00` 通过（闭环见[单标量输出向量链丢值卡](../precision/scalar_output_v_mte3_loss_readback.md)）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 探针结论依赖"输出张量被数值化比对"——测试只报 PASS/FAIL 时需临时加打印。
- 第 4 层的 Adds 拷贝探针要求源/目标同为 COMPUTE_T 布局；跨 dtype（fp32→fp16）需换 Cast 探针。
- 本方法定位到"哪一层被污染"为止；污染的写入路径（哪条指令写坏 buffer）仍需结合 plog/最小 repro 继续追。
- R5 增量：只能隔离"值错"类故障；崩溃、挂死、error code 类故障不适用（转 crash/plog 路线）。观测依赖 ST 能读回输出张量；输出在 workspace 中间态（不落 GM）时需要临时落 GM。每轮 build+安装+上板有分钟级成本，两步旁路是成本下界。单核调试模式下验证；多核时序类故障可能被单核旁路掩盖。

**未决项**：

- 【优先】复现并追根因：SoftMax 输出 buffer 被污染的写入路径（哪条指令/哪个阶段写坏）未定位——本卡只把污染层定位到 SoftMax 输出 buffer；后续应做脱离算子的最小独立 repro（单 SoftMax + 直拷输出）下钻。
- 未尝试用 core dump / 单指令级调试（如 coredump+gdb）继续下钻；对框架高阶 API（SoftMax）内部实现，用户态探针链到此为止。
## 坏实践

- 不做探针直接连环改算法（换 API、加初始化、改 tiling）——本例五种 workaround 连试全部无效，等于没有信息增益地烧了五轮构建。
- 多个探针一次插入，失配输出无法归层。
- 用 `DataCopyPad` 做 UB→UB 拷贝探针（编译不过）。

- R5 增补：在无任何旁路观测的情况下连续盲改同步/数据通路（该会话前三轮零信息增益）；用输入数据本身（如 x[0]）做旁路常数——与计算结果可能碰撞，无法判定来源。

## 误诊

- 本方法在本例中直接推翻了初始假设"arch35 TopK API 在 k==inner 时产生垃圾值"——第 4 层探针显示不经过 TopK 直接拷 SoftMax 输出也是垃圾。若不做分层探针，该误诊会被五种无效 workaround 不断"印证"。
