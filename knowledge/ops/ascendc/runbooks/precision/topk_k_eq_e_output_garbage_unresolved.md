---
type: "Runbook"
title: "k==E 输出垃圾值：SoftMax buffer 污染未解"
description: "arch35 fp16 k==E（TopK 取全部元素）用例输出尾部出现 14992.0 级有限垃圾值；tmp 零初始化、扩大 topkInner、Sort API、标量选择排序、isKEqE tiling 标志五类 kernel 内 workaround 全部无效；分层探针证明污染在 SoftMax 输出 buffer 本身且结果数据依赖；按已知 API 限制归档，k==E kernel 内修复路线 banned。"
tags: [task.debugging, topic.precision, topic.compute.ordering, topic.mixture_of_experts]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# k==E 输出垃圾值：SoftMax buffer 污染未解

## 现象

在 Ascend950PR（DAV_3510/arch35，CANN 9.1.0）上验证一个 MoE 门控算子（SoftMax → TopK → Cast，TopK 用 `AscendC::TopK` adv_api，SoftMax 用 `SoftMax<COMPUTE_T, true>`，fp16/bf16 输入先 Cast 到 fp32 计算）。三个 **fp16 且 k==E**（TopK 取走整行全部专家）的用例稳定失败：

- `[1024,64]`，k=64=E：MERE=5.53e+07，MARE=2.68e+08；
- `[1024,128]`，k=128=E：MERE=7.45e+07；
- `[1024,1024]`，k=1024=E：MERE=1.27e+08（修复 UB 崩溃前它还伴随 507035 流同步失败）。

失败形态非常特征化：每行输出的**前段基本正确**（rel_err ~1e-03，与正常用例同量级），而**尾部若干位置**出现 14992.0、12752.0、16272.0 这类量级的垃圾值——它们不是 NaN/Inf，也不是越界索引，像未初始化/被覆盖的 float 数据。同 E、k<E 的用例全部正常；fp32 无 k==E 用例（未覆盖）。

## 适用场景

- 工程形态：AscendC kernel 内 SoftMax→TopK 串联（fp16/bf16 输入、fp32 中间计算、TopK 高阶 API）的精度联调。
- API 路径：`AscendC::TopK`（TOPK_NORMAL 模式，3510 MergeSort 实现：Sort32 分组排序 + MergeSort 合并 + GatherMask 提取）、`AscendC::Sort`、`SoftMax`、`GetTopKMaxMinTmpSize`/`TopKTilingFunc`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0。
- 适用粒度：same_operator~operator_family——限 arch35 + CANN 9.1.0 的 TopK/SoftMax 组合，且表现为数据依赖。
- 路线优先级：banned（对"k==E 靠 kernel 内 workaround 修复"这一路线）——五类尝试均失败；现实处置是按已知 API 限制归档或规避 k==E（规避路线为 safety_fallback，代价是牺牲 k==E 场景的 kernel 加速，相对 kernel TopK 路线的成对比较未测）。
- 明确不适用 / 未验证场景：k<E 的 TopK 用法不受影响（全部通过）；fp32/bf16 的 k==E 未验证；其他 CANN 版本的 TopK 实现未验证。

## 尝试与失败（negative_result 主体）

以下五类 workaround 逐个落地（每类都经过 build.sh 重建 + 卸载重装 + 单用例复跑验证，且经 md5/常量探针确认新二进制在跑、分支在执行），**全部无效，垃圾值逐次完全相同**：

1. **TopK 前零初始化临时区**——把 `sharedTmpBuf_`（SoftMax/TopK 共享 tmp）和输出 `yBuf_` 先 `Duplicate` 清零再进 TopK，防止 Sort/GatherMask 读到未覆盖位置的残留：

```cpp
    // 零初始化 yBuf
    Duplicate<COMPUTE_T>(yCompute, static_cast<COMPUTE_T>(0),
                         static_cast<uint32_t>(curBatch * kPad_));
    PipeBarrier<PIPE_V>();
    // 零初始化 sharedTmpBuf_
    {
        uint32_t tmpElements = sharedTmpSize_ / sizeof(COMPUTE_T);
        if (tmpElements > 0) {
            Duplicate<COMPUTE_T>(sharedTmp.ReinterpretCast<COMPUTE_T>(),
                                 static_cast<COMPUTE_T>(0), tmpElements);
            PipeBarrier<PIPE_V>();
        }
    }
```

2. **扩大 topkInner 远离 k==inner 退化点**——k==E 时把 EAlign 加 pad（`EAlign = EAlign + TOPK_INNER_ALIGN`）乃至 `EAlign = CeilAlign(2 * E, alignElemE)`（k=inner/2），并把 padding 区显式填 `-1e30`（先 `Duplicate` 后改标量 `SetValue` 双试）。附带发现：kernel 内部头文件的 `TOPK_INNER_ALIGN` 常量在 op_host tiling 代码不可用（编译错 not declared，需本地定义且先定义后使用）。

3. **Sort API 替代 TopK**——取反（`Muls -1`，-inf padding 变 +inf 沉底）→ 逐行 `Sort<COMPUTE_T, int32_t>` 带索引升序 → 前 k 个再取反恢复。期间踩两个编译坑：Sort 重载参数须用左值/匹配签名（`no matching function for call to 'Sort'`）；UB→UB 整块拷贝探针不能用 `DataCopyPad`（同样 `no matching function`），要用 `DataCopy`（32B 对齐）或向量算术（`Adds +0`/`Cast`）。

4. **标量选择排序**（最终保留在代码里的形态）——完全绕开 TopK，逐行 `GetValue` 扫最大值、`SetValue` 写输出、已选位置置 -inf：

```cpp
    if (isKEqE_ == 1) {
        // 修复: k==E 时 TopK API 产生垃圾值, 使用标量选择排序替代
        // 注意: arch35 上 k==E 场景 SoftMax 输出 buffer 可能存在数据污染,
        // 标量排序从 softmaxDst 读取, 尽最大努力获取正确值
        PipeBarrier<PIPE_ALL>();
        uint32_t infBits = 0xFF800000U;
        COMPUTE_T negInf = *reinterpret_cast<COMPUTE_T*>(&infBits);

        for (int64_t row = 0; row < curBatch; row++) {
            for (int64_t j = 0; j < k_; j++) {
                int32_t baseOff = static_cast<int32_t>(row * topkInner_);
                COMPUTE_T maxVal = softmaxDst.GetValue(baseOff);
                int32_t maxIdx = 0;
                for (int64_t i = 1; i < topkInner_; i++) {
                    COMPUTE_T val = softmaxDst.GetValue(baseOff + static_cast<int32_t>(i));
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = static_cast<int32_t>(i);
                    }
                }
                yCompute.SetValue(static_cast<int32_t>(row * kPad_ + j), maxVal);
                ...
                softmaxDst.SetValue(baseOff + maxIdx, negInf);
            }
        }
        PipeBarrier<PIPE_V>();
    } else {
        // 正常 TopK 路径 (k < E)
        ...
    }
```

5. **tiling 增加 `isKEqE` 运行时标志**——把 `if (k_ == E_)` 的判断换成 tiling 下发的 `int32_t isKEqE` 字段（`tiling->isKEqE = (k == E) ? 1 : 0;` kernel `isKEqE_ = tilingData->isKEqE;`），排除"分支被编译器优化掉"的怀疑（配合常量探针证明分支本来就在执行，此项不是根因修复而是排查手段）。期间还试过 `volatile` 比较加 `goto` 跳转的写法，被 C++ 规则打回（goto 不能跨越变量初始化）。

**关键探针结论**（定位污染源；分层探针方法见同域[垃圾值分层探针链卡](../runtime/kernel_output_garbage_layered_probe_chain.md)）：

- 常量写入探针：输出如期变 1.0/2.0/3.0 → 写通路、分支、tiling 参数（isKEqE=1, k=64, E=64, topkInner=64）全对；
- `GetValue` 回读 softmaxDst：位置 3 直接读到 12752.0；
- `Adds` 向量整块拷贝 softmaxDst→yCompute（不经过 TopK、不经过标量读）：MERE=1.05e+06 仍是垃圾。

即：**不进 TopK、直接把 SoftMax 的输出 buffer 拷到结果也是垃圾——污染在 SoftMax 输出 buffer（softmaxDst）本身**，这与最初"TopK Sort+GatherMask 在 k==inner 时有 bug"的假设矛盾。

**判别信号（归纳）**：

- k==E 且仅尾部位置出垃圾值（量级 1e+4 的有限值），k<E 同 E 正常；
- 常量探针正常 + 上游 buffer 直拷仍垃圾 → 污染在上游 buffer；
- **数据依赖**：同一 kernel、同一 shape/阈值，换一套随机输入（PyTorch 侧生成）后 `[1024,64]` k=64=E 用例 MERE=3.24e-04 通过——不同数据触发与否不同，不能只在单一固定种子下下结论（`[1024,1024]` k=1024=E 仍失败）。

## 设计决策

- 为什么最终保留"标量选择排序 + isKEqE tiling 标志"形态交付：它把 k==E 从 TopK 路径隔离出来（k<E 主路径不受影响），虽然对触发数据仍读到污染值，但至少不再放大问题；性能上 O(k*E) 标量操作只在 k==E 边界场景触发，可接受。
- 为什么按"已知 API 限制"归档而不是继续修：五类 workaround + 分层探针已把问题压到"SoftMax 输出 buffer 在 k==E 场景被污染"这一无法在用户态继续拆解的层面（SoftMax 是框架高阶 API，内部实现对用户是编译库/头文件实现），继续投入应转向向 CANN 侧反馈。
- 当前方案路线档位：banned（kernel 内 workaround 路线）；规避路线（限制 k<E 或接口层特判 k==E）为 safety_fallback——fallback 声明：接受该边界意味着 k==E 场景不进入精度验收通过项，需在验收报告中显式登记。

## 预防

- MoE 门控类算子若业务允许，设计期就限制 k<E（TopK 语义上 k==E 等价于全排序，可用 Sort 或跳过 TopK 表达）。
- 评审时把"k==E / k==inner"列入 TopK 用例边界清单，提前在接口文档标注。
- 遇到本形态失败先做分层探针（常量写入 → 标量回读 → 向量拷贝）再定归因方向，不要先入为主认定是某个高阶 API 的 bug。

## 验证信号

以下信号来自该算子 k==E 三用例的排查会话（Ascend950PR，CANN 9.1.0；每步均重建重装后单用例/全量复跑）：

- 修复 UB 预算后基线：40/45，k==E 三例仍 MERE=5.53e+07/7.45e+07/1.27e+08，垃圾值 14992.0/12752.0/16272.0 位于输出尾部（来源平台：950；证据等级：tool_observed）
- workaround 1~5 落地后：三例 MERE 与失配位置逐次完全相同，无任何改善（来源平台：950；证据等级：tool_observed）
- 常量探针：输出变为 1.0/2.0/3.0（MERE 由 5.53e+07 → 1.04e+00 量级）；tiling 回显 isKEqE_=1、k_=64、E_=64、topkInner_=64（来源平台：950；证据等级：tool_observed）
- `GetValue` 回读 softmaxDst：位置 3 = 12752.0；`Adds` 向量拷贝 softmaxDst→yCompute：MERE=1.05e+06（污染在上游 buffer）（来源平台：950；证据等级：tool_observed）
- 数据依赖对照：PyTorch ST（torch 随机数据）同一 kernel 下 `[1024,64]` k=64=E 用例 MERE=3.24e-04 PASS；`[1024,1024]` k=1024=E 仍失败（来源平台：950；证据等级：tool_observed）
- 事实锚点：TopK 3510 实现使用 `__ASC_USE_RESERVED_UBUF__`，其参数约束（k<=n<=inner、inner%32==0）的 ASCENDC_ASSERT 在真机为空实现（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本卡结论限于 arch35 + CANN 9.1.0 + fp16 输入 + fp32 中间计算的 SoftMax→TopK 组合；fp32/bf16 的 k==E、其他 TopK 模式（TOPK_RADIX 等）未验证。
- "SoftMax 输出 buffer 污染"是探针归纳的现象级结论，污染写入路径（哪条指令/哪个布局交错）未定位；不排除 k==E 特定 tiling（topkInner==E 时 buffer 布局与 k<E 不同）诱发 SoftMax 内部越界的机制。
- 数据依赖意味着固定种子下"通过"也不能证明已修复（反之亦然）。
- 其他 NPU 世代未验证，推断可迁移（判别方法通用，触发面按平台复测）。

**未决项**：

- **优先项：复现并追到机制层根因**——k==E 时 softmaxDst 被污染的确切写入路径（SoftMax 内部实现/布局/tiling 交互），以及为何 k<E 同 E 不触发；建议用最小独立 repro（脱离本算子，仅 SoftMax 写 buffer + 直拷回读）向 CANN 侧反馈。
- k==E 数据依赖的触发条件（哪类输入分布触发）未刻画。
- 规避路线（接口层 k==E 特判/限制 k<E）的相对性能代价未测。
## 坏实践

- 只在单一随机种子下断言 k==E 修复成功/失败。
- 把五类 workaround 叠加一次性提交，无法区分哪类有效（本轨迹逐类验证才得出"全无效"的负结论）。
- 在未做"上游 buffer 直拷"探针前，直接把锅扣给 TopK API。

## 误诊

- 初始假设"arch35 TopK Sort+GatherMask 在 k==inner 时产生垃圾值"：Adds 直拷探针显示不经过 TopK 也是垃圾，该假设至少不完整——SoftMax 输出 buffer 本身已污染（issue 归档文本与探针结论存在此张力，采用探针结论）。
- 曾怀疑"分支被编译器优化掉/二进制没更新"：md5 对比与常量探针证明分支始终在执行。
- 曾怀疑 padding 区（E..topkInner）残留垃圾影响标量排序：把 padding 显式填 -1e30（Duplicate 与标量 SetValue 双试）后输出不变，排除。
