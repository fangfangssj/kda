---
type: "Runbook"
title: "DataCopyPad 第4参类名 CTAD 失配"
description: "GM↔UB 四参 DataCopyPad 的第 3/4 参必须参数组配对：DataCopyParams 配 DataCopyPadParams、DataCopyExtParams 配 DataCopyPadExtParams<T>。第 4 参显式写 DataCopyPadExtParams{false,0,0,0} 会被 CTAD 固化为 <int>、或声明为具名 DataCopyPadExtParams<T> 变量搭配旧组 copy 参数，均与重载形参不可转换，编译报 no matching function；改写裸 braced-init-list {false, 0, 0, 0} 让编译器按形参类型构造即可通过。上游常见误导性症状是 ld.lld cannot open .o 链接错。"
tags: [task.implementation, topic.compilation, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# DataCopyPad 第4参类名 CTAD 失配

## 现象

在为 BenchNoHarnessUnique 算子（radix sort 去重骨架，目标 SoC ascend950）编写 kernel 的 GM↔UB 数据搬运时，搬运调用按"带扩展参数"的四参形式书写，第 4 参显式写了类名：

```cpp
DataCopyPad(inputLocal, inputGMX_[gmOffset + chunkOffset], copyParams,
            DataCopyPadExtParams{false, 0, 0, 0});
```

执行 `bash build.sh --soc=ascend950 -j8` 后 kernel 编译（TBE/ccec 阶段）在同一文件的多个调用点（GM→UB 读、UB→GM 写、排序 ping-pong 读写等共 5 处）全部报错，报错原文：

```
op_kernel/arch35/bench_no_harness_unique.h:216:17: error: no matching function for call to 'DataCopyPad'
                DataCopyPad(inputLocal, keysGM_[chunkOffset], copyParams,
                ^~~~~~~~~~~
<CANN_HOME>/x86_64-linux/asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h:1289:45:
note: candidate function template not viable: no known conversion from
'AscendC::DataCopyPadExtParams<int>' to 'const AscendC::DataCopyPadParams' for 4th argument
```

其余候选也全部 not viable：`requires 3 arguments, but 4 were provided`（三参重载收到四参）、`could not match 'LocalTensor' against 'GlobalTensor'`（方向不符）等。该错误在 kernel 的两轮版本中重复出现，是当时编译失败清单里占比最高的一类。

同型第二形态（SwiGLU 轨迹，独立 trace）：BenchNoHarnessSwiGlu 的 TilingKey=2 非连续拷贝路径 `CopyNonContiguous()` 中，第 3 参声明了 `DataCopyParams copyParams`，第 4 参单独声明具名变量 `DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};`（模板实参正确），调用 `DataCopyPad(copyLocal, inputGM[srcOffset], copyParams, padParams)`——同样报 `error: no matching function for call to 'DataCopyPad'`（3 个 kernel 变体同报）。且该轨迹上游先看到的是误导性链接错误 `ld.lld: error: cannot open .../kernel_2/kernel_meta_.../..._283.o: No such file or directory`，用 `bash build.sh ... 2>&1 | rg -i "error|fail"` 重跑抓全量日志才看到真正的编译错误。同文件旧代码 `DataCopyPad(x0Local[localOffset], inputGM[x0GmOffset], copyParams, {false, 0, 0, 0});`（内联初始化列表）一直编译正常——差异只在 pad 参数是具名 Ext 变量还是内联列表。

## 适用场景

- 工程形态：AscendC kernel 开发（op_kernel 侧向量编程），使用带 pad 扩展参数的四参 `DataCopyPad` 做 GM↔UB 不对齐搬运。
- API 路径：`AscendC::DataCopyPad(const LocalTensor<T>&, const GlobalTensor<T>&, params, extParams)` 及反向重载（`kernel_operator_data_copy_intf_impl.h` / `kernel_operator_data_copy_intf.h`）；第 4 参 pad 扩展参数（`{false, 0, 0, 0}` = 不清零、左右 pad 0）；`kernel_struct_data_copy.h` 的参数结构体。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）、CANN 9.1.0、ccec kernel 编译（两条独立轨迹，与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本上所有使用四参 DataCopyPad 的 kernel，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：三参（无扩展参数）DataCopyPad 重载；UB→UB 形式；其他 CANN 版本的模板/推导行为（未验证，机制推断可迁移）。

## 根因

DataCopyPad 的 GM→UB 方向有两组互斥重载，第 3/4 参数必须**成对**出现：

```cpp
// 旧参数组
DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
            const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams);
// 新参数组（3510/5102 有 PaddingMode 模板参）
DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
            const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams);
```

`DataCopyParams` 与 `DataCopyExtParams` 是两个无关结构体（无隐式转换），`DataCopyPadParams` 与 `DataCopyPadExtParams<T>` 亦然。

CTAD 形态（Unique 轨迹）：第 4 参显式写出类模板名 `DataCopyPadExtParams{false, 0, 0, 0}` 时，类模板实参推导（CTAD）把该对象固化为 `AscendC::DataCopyPadExtParams<int>`，与第 3 参 `DataCopyParams` 所在旧参数组期望的第 4 形参 `DataCopyPadParams` 不可转换，所有候选 not viable。

具名变量形态（SwiGLU 轨迹）：具名 `DataCopyPadExtParams<T>` 变量把重载决议锁定到新参数组，而第 3 参 `DataCopyParams` 又只匹配旧参数组，两组均失败 → `no matching function`。旧代码传内联 `{false, 0, 0, 0}` 时，初始化列表可同时适配 `DataCopyPadParams`（paddingValue 为 uint64_t）与 `DataCopyPadExtParams<T>`（paddingValue 为 T），由第 3 参 `DataCopyParams` 一致选中旧参数组，故能编译——这正是"裸 brace 可用、显式 Ext 类型不可用"的机制。

两组结构体的字段差异（头文件确认，3510 分支）：

```cpp
struct DataCopyParams {          // blockLen 为 uint16_t
    __aicore__ DataCopyParams(const uint16_t count, const uint16_t len,
        const uint16_t srcStrideIn, const uint16_t dstStrideIn) ...
};
struct DataCopyExtParams {       // 3510/5102: blockLen uint32_t, stride int64_t
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const int64_t srcStrideIn,
        const int64_t dstStrideIn, const uint32_t rsvIn)
#else
    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const uint32_t srcStrideIn,
    const uint32_t dstStrideIn, const uint32_t rsvIn)
#endif
```

## 解决配方

CTAD 形态修复（Unique 轨迹）：对 kernel 头文件做全局文本替换（5 处调用点一次修完）：

```bash
cd operators/bench_no_harness_unique && \
sed -i 's/DataCopyPadExtParams{false, 0, 0, 0}/{false, 0, 0, 0}/g' \
    op_kernel/arch35/bench_no_harness_unique.h
```

修复前（kernel 内 GM→UB 读，四参带 pad 扩展）：

```cpp
        // GM → UB: 加载 int32 输入数据
        auto inputLocal = inputKeysBuf_.template Get<int32_t>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = chunkSize * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGMX_[gmOffset + chunkOffset], copyParams,
                    DataCopyPadExtParams{false, 0, 0, 0});
```

修复后（第 4 参改裸 braced-init-list，其余不动）：

```cpp
        // GM → UB: 加载 int32 输入数据
        auto inputLocal = inputKeysBuf_.template Get<int32_t>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = chunkSize * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGMX_[gmOffset + chunkOffset], copyParams,
                    {false, 0, 0, 0});
```

排序 ping-pong 读（keysGM_/sortedGM_ 交替）与 UB→GM 写同理：

```cpp
            if (passIdx % 2 == 0) {
                DataCopyPad(inputLocal, keysGM_[chunkOffset], copyParams,
                            {false, 0, 0, 0});
            } else {
                DataCopyPad(inputLocal, sortedGM_[chunkOffset], copyParams,
                            {false, 0, 0, 0});
            }
```

具名变量形态修复（SwiGLU 轨迹，`CopyNonContiguous()` 行拷贝循环）：删掉具名 Ext 变量、改内联初始化，并给 `inputGM[]` 下标加 `uint64_t` 显式转换。

修复前（verbatim）：

```cpp
    // DataCopyPad 参数
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(copyRowSize_ * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    // Step 4: 逐行拷贝 (GM → UB → GM)
    for (uint64_t row = coreRowStart; row < coreRowEnd; row++) {
        // 分解行索引为多维索引, 计算源偏移
        uint64_t remaining = row;
        int64_t srcOffset = inputOffset_;
        for (uint32_t d = 0; d < outerRank; d++) {
            uint64_t idx = remaining / static_cast<uint64_t>(outerStrideProduct[d]);
            remaining = remaining % static_cast<uint64_t>(outerStrideProduct[d]);
            srcOffset += static_cast<int64_t>(idx) * inputStrides_[d];
        }

        uint64_t dstOffset = row * copyRowSize_;

        // GM(非连续) → UB → GM(workspace, 连续)
        LocalTensor<T> copyLocal = copyQueue.template AllocTensor<T>();
        DataCopyPad(copyLocal, inputGM[srcOffset], copyParams, padParams);
        copyQueue.EnQue(copyLocal);
        LocalTensor<T> copyData = copyQueue.template DeQue<T>();
        DataCopyPad(workspaceGM[dstOffset], copyData, copyParams);
        copyQueue.FreeTensor(copyData);
    }
```

修复后（verbatim）：

```cpp
    // DataCopyPad 参数
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(copyRowSize_ * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    // Step 4: 逐行拷贝 (GM → UB → GM)
    for (uint64_t row = coreRowStart; row < coreRowEnd; row++) {
        // 分解行索引为多维索引, 计算源偏移
        uint64_t remaining = row;
        int64_t srcOffset = inputOffset_;
        for (uint32_t d = 0; d < outerRank; d++) {
            uint64_t idx = remaining / static_cast<uint64_t>(outerStrideProduct[d]);
            remaining = remaining % static_cast<uint64_t>(outerStrideProduct[d]);
            srcOffset += static_cast<int64_t>(idx) * inputStrides_[d];
        }

        uint64_t dstOffset = row * copyRowSize_;

        // GM(非连续) → UB → GM(workspace, 连续)
        LocalTensor<T> copyLocal = copyQueue.template AllocTensor<T>();
        DataCopyPad(copyLocal, inputGM[static_cast<uint64_t>(srcOffset)], copyParams, {false, 0, 0, 0});
        copyQueue.EnQue(copyLocal);
        LocalTensor<T> copyData = copyQueue.template DeQue<T>();
        DataCopyPad(workspaceGM[dstOffset], copyData, copyParams);
        copyQueue.FreeTensor(copyData);
    }
```

判别方法：看到 `no matching function for call to 'DataCopyPad'` 且 note 里出现 `no known conversion from 'AscendC::DataCopyPadExtParams<...>'`（或第 3 参 `DataCopyParams` 与第 4 参 Ext 类型同时出现），直接检查第 3/4 参是否跨参数组混搭——显式类名（CTAD）与具名 Ext 变量两种写法都触发。

## 设计决策

- 为什么选裸 braced-init-list 而不是显式写全模板实参（如 `DataCopyPadExtParams<int32_t>{...}`）：裸 `{false, 0, 0, 0}` 的类型由重载形参决定，跟随 dst/src 的 `T` 自动一致；显式写死元素类型在任何一处 `T` 变化（模板化 kernel 换 dtype）时都会再次失配。本算子 kernel 是 `template <typename D_T_X, int HAS_INVERSE>` 模板，裸 brace 是唯一随模板参数自动适配的写法。
- 为什么选内联 `{false, 0, 0, 0}` 而非把第 3 参换成 `DataCopyExtParams`（SwiGLU 轨迹）：与文件内既有 8 处 DataCopyPad 调用保持同一参数组（旧组），改动面最小；且该算子行字节数（copyRowSize_×sizeof(T)）实测小于 65535，`DataCopyParams` 的 uint16_t blockLen 够用。
- 已知边界并接受（SwiGLU 轨迹）：`DataCopyParams.blockLen` 是 uint16_t，单行字节数 > 65535（如 fp32 行长 > 16384 元素）会截断，届时必须整组换 `DataCopyExtParams`（3510 上其 blockLen 为 uint32_t、stride 为 int64_t）；本算子约束 dim 折半后内层连续段不超该规模，未做预防性切换。
- `srcOffset` 加 `static_cast<uint64_t>`：消除 int64_t 下标到 GlobalTensor::operator[] 的符号转换歧义，属顺手加固。
- 该写法在修复后随整个 kernel 一起编译通过，成为工程内 DataCopyPad 的固定风格。

## 系统信息

- 硬件/软件：Ascend950PR（arch35）、CANN 9.1.0（`x86_64-linux/asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h` 为报错指向的实现头；`kernel_struct_data_copy.h` 为参数结构体）、ccec kernel 编译。
- 相关文件：`op_kernel/arch35/bench_no_harness_unique.h`（5 处调用点）；`op_kernel/arch35/bench_no_harness_swi_glu.h`（`CopyNonContiguous()` 行拷贝循环）。

## 预防

- DataCopyPad 第 3/4 参永远成对写：`(DataCopyParams, DataCopyPadParams)` 或 `(DataCopyExtParams, DataCopyPadExtParams<T>)`；不要给 pad 参数单独声明具名 Ext 类型变量、或显式写省略模板实参的类名，再搭配旧组 copy 参数。
- 需要具名变量时，类型跟着第 3 参走：`DataCopyParams copyParams` 配 `DataCopyPadParams padParams`。
- 给四参 DataCopyPad 的第 4 参写裸 braced-init-list（或与形参一致的显式模板实参），不写省略模板实参的类名。
- 单块字节数可能超 65535 时直接用 Ext 组（blockLen uint32_t）。

## 验证信号

以下信号来自该算子 radix sort 骨架的两轮编译（Unique 轨迹，来源说明：同会话构建输出）：

- 修复前：kernel 编译在 5 个调用点报 `error: no matching function for call to 'DataCopyPad'`，note 原文 `no known conversion from 'AscendC::DataCopyPadExtParams<int>' to 'const AscendC::DataCopyPadParams' for 4th argument`（来源平台：950；证据等级：tool_observed）。
- 替换后：`grep -n "DataCopyPadExtParams" op_kernel/arch35/bench_no_harness_unique.h` 无任何命中，确认全部替换（来源平台：950；证据等级：tool_observed）。
- 重编后：构建成功，生成 kernel 二进制与算子包 `custom_opp_ubuntu_x86_64.run`，`[INFO] Build completed successfully!`（来源平台：950；证据等级：tool_observed）。

以下信号来自 SwiGLU 轨迹 kernel 侧非连续拷贝路径实现会话（独立 trace）的编译错误与修复、全量重编与二进制 JSON 核验：

- 报错原文 `bench_no_harness_swi_glu.h:508:9: error: no matching function for call to 'DataCopyPad'`（3 个 kernel_0/1/2 变体同报）；上游先观测到误导性 `ld.lld: error: cannot open ..._283.o: No such file or directory` 链接错（来源平台：950；证据等级：tool_observed）。
- 头文件核对确认两组重载签名与 DataCopyParams/DataCopyExtParams 构造函数字段类型（uint16_t blockLen vs 3510 上 uint32_t blockLen/int64_t stride）（来源平台：950；证据等级：tool_observed）。
- 修复后 `--make_clean` 全量重编：`[INFO] Build completed successfully!`，binary JSON 确认 6 个 kernel 变体（TilingKey 0/1/27 + 256/257/283）全部生成（来源平台：950；证据等级：tool_observed）。
- UT 46/46 通过，无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论基于 CANN 9.1.0 arch35 的重载集合；其他版本若重载形参类型或 CTAD 行为变化（例如提供 deduction guide），显式类名写法可能重新可用，以头文件为准（未验证）。
- 本卡验证于 `__NPU_ARCH__ == 3510`；非 3510/5102 平台 `DataCopyExtParams` 的 stride 字段为 uint32_t，混搭结论相同但位宽细节不同（未验证，推断可迁移）。
- 只覆盖"第 4 参扩展参数"的写法/参数组配对问题；DataCopyPad 的 blockLen 按**字节**计（`chunkSize * sizeof(T)`）、stride 语义等数值约束不在本卡范围。
- SwiGLU 轨迹的修复服务于 TilingKey=2 非连续路径的**编译**闭环；该路径后续在联调中被证明无法在 CANN 9.1.0 触发并回退（见[noncontig_tilingkey_dead_path_three_layers](noncontig_tilingkey_dead_path_three_layers.md)），不影响本卡"参数结构配对"这一 API 层结论。

**未决项**：

- 无（两轨迹均修复闭环，编译与头文件双重确认）。
## 坏实践

- 给四参 DataCopyPad 的第 4 参显式写省略模板实参的 `DataCopyPadExtParams{...}` 类名——CTAD 固化成 `<int>` 后与形参不可转换。
- 给 pad 参数单独声明具名 `DataCopyPadExtParams<T>` 变量、搭配 `DataCopyParams` 第 3 参——两组重载均失配。
- 看到一串 not viable 候选就怀疑自己搬运方向或参数个数全错、把代码改成三参形式丢掉 pad 语义——先看 note 里的 `DataCopyPadExtParams<...>` 字样即可定位。
- 把 FATBIN 链接期的 `ld.lld: error: cannot open ....o: No such file or directory` 当根因——它是编译失败的**下游症状**（.o 未生成 → 链接找不到），必须回滚日志找第一条 `error:`；用 `2>&1 | rg -i "error|fail"` 过滤重跑是可行做法。

## 误诊

- 无（Unique 轨迹 note 原文直接给出 `DataCopyPadExtParams<int>` 与形参类型不可转换，一次定位）。
- SwiGLU 轨迹首轮被 ld.lld 链接错误带偏（误以为链接/构建产物问题），抓全量编译日志后一次定位到参数混搭。
