---
type: "Runbook"
title: "arch35 跨位宽 Cast 按较大位宽读源致错值"
description: "arch35(DAV_3510) 上 Cast<dst,src> 位宽不同时 count 按较大类型字节数读源 buffer（count*sizeof(larger)），元素错位 + 越界读垃圾：bf16→float 使 TopK 输出 MERE=nan/MARE=inf（垃圾大值被选入 top-k）、int8/uint8 错值且 uint8 反向 255 被 clip 成 127；修复为 int8/uint8 走 int16 同位宽两步 Cast、bf16 用 SetValue 位重排完全绕开 Cast API。复现增量：largest 方向失败规律指纹 + printf 不可见时的短路差分/ReduceMax 探针定位法。复现增量：float→int64 方向（argmax 索引落盘）——输出为两个 float 位模式打包的 int64（16 进制拆高低 32 位指纹）、count=1 档位串位/越界小整数，标量 SetValue/GetValue 循环替代后 ST Real 101/101。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T03:16:41Z'
platforms: ["950"]
---

# arch35 跨位宽 Cast 按较大位宽读源致错值

## 现象

在 Ascend 950PR（NpuArch=DAV_3510，arch35，CANN 9.1.0）上开发自定义 TopK 算子（kernel 直调 + aclnn 自定义算子包，`bash build.sh --soc=ascend950 -j8` 编译安装后 Real 模式 ST 上板）。该算子对非原生排序 dtype 走 Cast 路径：bf16 输入先 `Cast<float, bfloat16_t>` 转 fp32 排序、int8/uint8 输入先 `Cast<half, int8_t>` 转 fp16 排序，TopK 后再反向 Cast 回原 dtype。

联调时 ST 报 15 个用例失败（总计 61，通过 35 / 失败 15 / 跳过 11），与 Cast 相关的失败呈两类特征：

- bf16 用例：`[FAIL] y 比对: MERE=nan (>= 7.81e-03), MARE=inf (>= 7.81e-02), mismatch=1021/2560`；largest=false 用例 `MARE=6.32e+04, mismatch=4/2560`——输出混进 NaN/Inf 量级的值，排序输入被垃圾数据污染后，垃圾中的大值被 TopK 选为最大值。
- int8/uint8 用例：输出错值；uint8 反向 Cast 还把 255 clip 成 127（目标按有符号 int8 处理）。

fp16/fp32 原生 dtype 用例全部通过，编译无任何报错——问题只出现在 `sizeof(dst_T) != sizeof(src_T)` 的 Cast 上。

## 适用场景

- 工程形态：自定义算子 kernel（AscendC basic_api vector Cast）+ aclnn 算子包，ST Real 模式上板验证。
- API 路径：`AscendC::Cast<dst_T, src_T>(dst, src, roundMode, count)`（basic_api vector 类型转换）；涉及 `LocalTensor::GetValue/SetValue`、`Duplicate`、`PipeBarrier`。
- 验证过的平台与 CANN 版本：Ascend 950PR（arch35/DAV_3510）+ CANN 9.1.0（与 frontmatter 一致）。
- 适用粒度：operator_family——任何在 arch35 上使用跨位宽 Cast 做 dtype 前置/后置转换的算子（TopK/排序/argmax 类尤其直接，垃圾大值会直接改变选出的 top 集合）。
- 路线优先级：recommended_fastpath（int16 同位宽两步 Cast + bf16 位重排是实测通过的正路）。
- 明确不适用 / 未验证场景：其他 NpuArch（A2/A3/310P 等）未验证，不宣称同 bug 存在；`Cast<half, bfloat16_t>` 同位宽路线在本轨迹实测仍错（见误诊），不要当作安全选项；同位宽组合（如 int32↔float、int16↔half）实测不受影响。

## 根因

arch35（DAV_3510）上 `Cast<dst_T, src_T>` 当 `sizeof(dst_T) != sizeof(src_T)` 时，`count` 参数按较大类型的字节数计算实际读取字节数（`count * sizeof(larger_type)`）而非源类型字节数（`count * sizeof(smaller_type)`），并把源数据当作较大类型数组处理，导致元素错位 + 从源 buffer 越界读取垃圾数据：

- `Cast<float, bfloat16_t>`：读 `count*4` 字节而源仅有 `count*2` 字节有效——2 个连续 bf16 被当作 1 个 float；
- `Cast<half, int8_t>`：读 `count*2` 字节而源仅有 `count*1` 字节有效；
- 反向 `Cast<uint8_t, half>`：目标按有符号 int8 处理，255 被 clip 为 127。

垃圾数据进入排序输入后，垃圾中的大值被选为最大值（largest=true 输出完全错误）或恰好不影响最小值（largest=false 部分通过/边界失败），与观测到的 MERE=nan/MARE=inf、mismatch 逐 case 波动的特征吻合。

## 解决配方

核心策略：**在 arch35 上避免一切跨位宽 Cast**。同 dtype 家族用"同位宽 Cast + 手动位操作"替代。

**1) int8/uint8 路径：int16 中转（同位宽 2 字节 Cast 无 bug）**。前向（CopyIn 的 IS_CAST 分支）修复前 → 修复后（verbatim）：

```cpp
        // 修复前（跨位宽直 Cast）
        PipeBarrier<PIPE_ALL>();
        Cast<SortT, T>(srcValueSort, srcValueOrig, RoundMode::CAST_NONE,
                       static_cast<uint32_t>(innerPad_));
        PipeBarrier<PIPE_V>();

        // 修复后（零初始化 int16 + SetValue 符号/零扩展 + 同位宽 Cast<half, int16_t>）
        PipeBarrier<PIPE_ALL>();
        // 1. 零初始化 srcValueSort（int16 视角），确保 padding 区域为 0
        LocalTensor<int16_t> srcValueSortI16 = srcValueSortBuf.Get<int16_t>();
        Duplicate<int16_t>(srcValueSortI16, static_cast<int16_t>(0),
                           static_cast<int32_t>(innerPad_));
        PipeBarrier<PIPE_V>();
        // 2. 逐元素符号扩展（int8）/零扩展（uint8）到 int16
        for (int64_t i = 0; i < inner_; i++) {
            T val = srcValueOrig.GetValue(static_cast<uint32_t>(i));
            if constexpr (std::is_same_v<T, int8_t>) {
                srcValueSortI16.SetValue(static_cast<uint32_t>(i),
                                         static_cast<int16_t>(val));
            } else {
                srcValueSortI16.SetValue(static_cast<uint32_t>(i),
                                         static_cast<int16_t>(static_cast<uint16_t>(val)));
            }
        }
        PipeBarrier<PIPE_ALL>();
        // 3. Cast<half, int16_t>（同位宽 2 字节，无 bug）
        Cast<half, int16_t>(srcValueSort, srcValueSortI16, RoundMode::CAST_NONE,
                            static_cast<uint32_t>(innerPad_));
        PipeBarrier<PIPE_V>();
```

反向（CastBack，排序 dtype → 原 dtype）修复后（verbatim）：

```cpp
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // half → int8/uint8：不直接 Cast（位宽不同有 bug）
            // 改用两步：Cast<int16_t, half>（同位宽）+ SetValue 取低字节
            LocalTensor<int16_t> dstValueI16 = dstValueOrigBuf.Get<int16_t>();
            Cast<int16_t, half>(dstValueI16, srcSort, RoundMode::CAST_ROUND,
                                static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            for (int64_t i = 0; i < count; i++) {
                int16_t val = dstValueI16.GetValue(static_cast<uint32_t>(i));
                dstOrig.SetValue(static_cast<uint32_t>(i), static_cast<T>(val));
            }
            PipeBarrier<PIPE_V>();
        }
```

配套 buffer 分配（Init，small/large 两条路径同改）——int8/uint8 的 `dstValueOrigBuf` 必须从 `kPad_ * sizeof(T)`（1 字节）扩到 `kPad_ * sizeof(int16_t)`（2 字节），否则 `Cast<int16_t, half>` 写 int16 越界：

```cpp
        if constexpr (IS_CAST) {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                pipe.InitBuffer(dstValueOrigBuf, kPad_ * sizeof(int16_t));
            } else {
                pipe.InitBuffer(dstValueOrigBuf, kPad_ * sizeof(T));
            }
        }
```

**2) bf16 路径：完全绕开 Cast API 的位重排**。bf16 的位定义就是 float 的高 16 位，转换是纯位操作：`float_bits = (uint32_t)bf16_bits << 16`，反向 `bf16_bits = (uint16_t)(float_bits >> 16)`。用 `GetValue/SetValue` 逐元素 union 位重排，前后向都不过 Cast API：

```cpp
        // 前向 bf16 → float（CopyIn 内，verbatim）
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            for (int64_t i = 0; i < inner_; i++) {
                bfloat16_t val = srcValueOrig.GetValue(static_cast<uint32_t>(i));
                union { uint16_t u16; bfloat16_t b; } bf16Cvt;
                bf16Cvt.b = val;
                union { uint32_t u32; float f; } floatCvt;
                floatCvt.u32 = static_cast<uint32_t>(bf16Cvt.u16) << 16;
                srcValueSort.SetValue(static_cast<uint32_t>(i), floatCvt.f);
            }
            PipeBarrier<PIPE_ALL>();
        }

        // 反向 float → bf16（CastBack 内，verbatim）
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            for (int64_t i = 0; i < count; i++) {
                float val = srcSort.GetValue(static_cast<uint32_t>(i));
                union { uint32_t u32; float f; } floatCvt;
                floatCvt.f = val;
                uint16_t bf16Bits = static_cast<uint16_t>(floatCvt.u32 >> 16);
                union { uint16_t u16; bfloat16_t b; } bf16Cvt;
                bf16Cvt.u16 = bf16Bits;
                dstOrig.SetValue(static_cast<uint32_t>(i), bf16Cvt.b);
            }
            PipeBarrier<PIPE_ALL>();
        }
```

排序 dtype 保持 bf16→float（而非 half）：bf16 指数域 8 位与 float 相同、half 只有 5 位，bf16 数据超出 half 表示范围会变 inf；位重排是全指数域无损往返。tiling 侧 `GetSortDTypeSize` 的 bf16 档随排序 dtype 同步取 4（float），保证 kPad/UB 规划与 kernel 侧一致——kernel 与 tiling 的 dtype 尺寸表必须成对修改。

## 设计决策

- 选"同位宽 Cast + 手动位操作"而不是继续调 Cast 参数：`DataCopyPad<half> raw copy + ReinterpretCast`（改变元素偏移读错地址）与两步 Cast 中 int8→int16 的同类位宽问题已在归档阶段排除；同位宽 `Cast<half, int16_t>`/`Cast<int16_t, half>` 是唯一实测通过的 Cast 用法。
- int8/uint8 与 bf16 合并处置：同一根因（跨位宽按较大位宽读取）、同一策略（避免跨位宽 Cast），只是位操作细节不同。
- 逐元素 SetValue 是标量路径，性能劣于 vector Cast；本卡以正确性优先（该算子每行 inner 有上限，可接受）。对热点路径应先测同位宽 Cast 是否可用再决定。

## 预防

- 在 arch35 上对 `sizeof(dst) != sizeof(src)` 的类型对直接 `Cast<dst_T, src_T>` 前，先用最小用例验证（bf16→float、int8→half、half→uint8、int32→int64 均中招；同位宽组合实测安全）。
- 排序/选值类算子看到 MERE=nan/MARE=inf 先查 dtype 转换路径的内存语义，而不是查测试数据或容差。
- Mock 模式（CPU golden 自校验）不能暴露本 bug——它是硬件 Cast 行为差异，Mock 下 Cast 语义正确。

## 验证信号

来源说明：问题与验证来自一次 TopK 算子联调修复会话（Ascend950PR，CANN 9.1.0）：先修 int8/uint8（int16 中转两步 Cast），再修 bf16（位重排），每步重建算子包后跑完整 ST Real 回归。

- 修复前 ST Real：总计 61，通过 35 / 失败 15 / 跳过 11；bf16 用例 `MERE=nan, MARE=inf, mismatch=1021/2560`，largest=false 用例 `MARE=6.32e+04`（来源平台：950；证据等级：tool_observed）。
- int8/uint8 两步 Cast + buffer 扩容落地后：ST 44 通过 / 11 失败（int8/uint8 相关 4 例转好）；int8/uint8 边界用例（-128/127、0/255）全部元素精确一致（atol=0 比对），idx gather 校验通过（来源平台：950；证据等级：tool_observed）。
- bf16 位重排落地后：ST 51 通过 / 4 失败，bf16 全部 7 个用例（含 NaN/Inf 用例）PASS，`MERE=0.00e+00, MARE=0.00e+00`；剩余 4 个失败全部为非末维 dim 用例（另一独立问题）（来源平台：950；证据等级：tool_observed）。
- 终态回归：迭代范围内 ST 50/50 + UT 53/53（op_host 39 + op_api 14）无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复在 arch35/DAV_3510 + CANN 9.1.0 上实测；其他 NpuArch 未验证（不推断移植性，若需使用先按预防节探针复核）。
- bf16 位重排依赖"bf16 = float 高 16 位"的 IEEE 截断关系，不适用任何非该关系的自定义格式。
- int8/uint8 两步 Cast 的 `RoundMode::CAST_ROUND` 只在 half→int16 一步；若输出值超出 int8/uint8 表示范围会截断取低字节（TopK 场景输出必然在原 dtype 值域内，不受影响）。
- 逐元素 SetValue 的性能代价未量化（相对 vector Cast 的成对对比未测）；inner 极大的热点路径应先测同位宽 Cast 可用性（未验证项）。
- `Cast<half, bfloat16_t>` 同位宽失败的确切原因未归因（见误诊 1）。

**未决项**：

- 未在机制层确认 arch35 Cast 跨位宽读取语义的硬件/微码原因（观测归因到"按较大类型字节数读取"，无指令级证据）——复现最小用例并追到指令行为是优先项。
- `Cast<half, bfloat16_t>` 同位宽失败的确切原因未归因，需单独最小复现判定是 API 缺陷还是使用问题。
- 逐元素 SetValue 位重排的性能代价未量化；若成为热点，可探索 vector 指令实现的位重排替代。
## 坏实践

- 在 arch35 上对 `sizeof(dst) != sizeof(src)` 的类型对直接 `Cast<dst_T, src_T>`。
- 只改 kernel 的 dtype 路径不改 tiling 的 `GetSortDTypeSize`（或反之）——kPad/UB 规划与 kernel 视角错位。
- int8/uint8 反向 Cast 后仍按 1 字节分配 `dstValueOrigBuf`——`Cast<int16_t, half>` 需要 2 字节/元素的中间空间。
- 发现 MERE=nan/MARE=inf 后先去查测试数据或容差，而不是查 dtype 转换路径的内存语义。

## 误诊

- 误诊 1（未证实）："同位宽 `Cast<half, bfloat16_t>` 应该没有位宽 bug，把排序 dtype 从 float 改成 half 即可"——实测改完后 bf16 7 用例仍全部失败（MERE=nan/MARE=inf），该同位宽浮点格式互转路线在本环境下不可靠，最终放弃。
- 误诊 2（被证伪）："bf16 失败是 `srcValueOrigBuf` 的 padding 区域未初始化，Cast 把垃圾 bf16 转成 Inf/NaN"——给 padding 逐元素零初始化后重新编译上板，结果与修复前逐位相同，假设被证伪；真实解是完全绕开 Cast API。
- 误诊 3（before 代码里就有）："Cast 的 count 按目标元素数传即可"——arch35 跨位宽时实际读取字节数是 `count * sizeof(larger_type)`，源 buffer 不足即越界。

## 增量（独立轨迹的短路差分定位法与失败规律指纹）

另一条独立轨迹（TopK 类算子 ST Real，Ascend950PR/arch35，CANN 9.1.0）复现同一缺陷并给出"printf 不可见环境"的定位手段与失败规律指纹，补充三点：

**1) 失败规律指纹**：bf16/int8/uint8 的 `largest=false`（取最小 k）用例大部分通过、`largest=true`（取最大 k）全部失败——垃圾大值被 TopK 选为最大值（全错）而恰好不干扰最小值选择；bf16 输入 [-1,1] 输出却出现 ~1e+38 量级值；uint8 边界用例期望 255 实际 127、int8 期望 127 实际 84。

**2) 短路差分探针（printf 不可见时的替代）**：该环境 `AscendC::printf` 在 ST Real 下不可见（`ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_SLOG_PRINT_TO_STDOUT=1` 时 stdout 与 plog 均无输出），改用两步短路差分：

```cpp
    // 探针①（短路 TopK，验证 Cast 头部正确性）：跳过 TopK，把 Cast 后的
    // srcValueSort[0..k) 直接 CastBack 写到输出 y
    if constexpr (IS_CAST) {
        LocalTensor<T> dstValueOrig = dstValueOrigBuf.Get<T>();
        CastBack(dstValueOrig, srcValueSort /* 前 k 个 */, k_);
        ... return;
    }
```

判读：y 恰为输入前 k 个原值（int8 实测 121/116/93/-18/83，均在输入域内）→ Cast 在 [0..k) 正确，问题在被 TopK 消费的数据。

```cpp
    // 探针②（ReduceMax 全范围扫描，验证 Cast 尾区）：对 srcValueSort[0..innerPad)
    // 做 ReduceMax 后 CastBack 写 y[0]
    if constexpr (IS_CAST) {
        LocalTensor<SortT> maxVal = dstValueSortBuf.Get<SortT>();
        ReduceMax<SortT>(maxVal, srcValue, tmpLocal, static_cast<int32_t>(innerPad_), false);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> dstValueOrig = dstValueOrigBuf.Get<T>();
        CastBack(dstValueOrig, maxVal, 1);
        ... return;
    }
```

判读：int8 全范围 max 应为 127、实测 24；bf16 输入 [-1,1] 实测 max=1.17e+38 → Cast 在 [k..innerPad) 区段产生垃圾大值。结合探针①（头部正确）：垃圾集中在尾区，与 largest 方向失败规律吻合。诊断分支不得留在交付 kernel（会让 Cast 用例按设计失败）。

**3) 无效替代路线存档（勿重复）**：① SetValue/GetValue 逐元素标量循环替代 Cast 值转换——通过数 36→32（回归）；② `DataCopyPad<half>` 原始字节搬运 + `ReinterpretCast`——元素偏移按 half 计读错地址；③ 两步 Cast（int8→int16→half 用 Cast 做扩展步）——扩展步本身跨位宽同样中招（与本卡主修复"SetValue 手动扩展 + 零初始化 + 同位宽 Cast"的差别正在于此）。另：idx 的 `Cast<int64_t, int32_t>` 改标量循环会大面积破坏 idx 正确性（通过数 34→24），须保持 Cast 并配合扩 buffer + 零初始化。

验证信号（复现增量，独立轨迹 11+ 轮 ST Real 实测与两次 kernel 内差分探针）：

- 触发：uint8 期望 255 实际 127、int8 期望 127 实际 84/59/0、bf16 largest=true MARE=1.17e+38、大 inner Cast 路径 largest=true 40/40 全错；同 dtype largest=false 多数通过（来源平台：950；证据等级：tool_observed）。
- 探针①观测：短路 TopK 后 int8 y 为输入前 k 个原值（Cast [0..k) 正确）（来源平台：950；证据等级：tool_observed）。
- 探针②观测：ReduceMax 扫 [0..innerPad) 得 int8 max=24（应 127）、bf16 max=1.17e+38（输入 [-1,1]）→ 尾区垃圾（来源平台：950；证据等级：tool_observed）。
- printf 不可见佐证：环境变量齐全时 grep DBG=0、plog 无相关行（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 该轨迹会话终态 ST 35/50、剩余 15 个失败均为本问题（修复未在该会话落地，修复路线以主卡为准）；"largest=false 通过"不是正确性证明，边界/统计用例仍可能失败（观测到 bf16 largest=false 边缘失败 MERE 7.90e-03 略超阈 7.81e-03）。
- 探针短路分支移除后通过数 30→36 含此因素（留在 kernel 会让 Cast 用例全部失败）。

## 增量（float→int64 方向：位打包指纹与标量替代闭环）

自定义 argmax 算子（Ascend950PR/arch35，CANN 9.1.0，ST Real 模式 NPU 执行 + CPU golden 逐位比对）补上 **dst 更宽的 4B→8B 方向**（float→int64 索引落盘）与两个 count 档位的失败指纹，并闭合标量替代：

**1) 失败指纹（两类）**：

- AR 路径（逐行输出 1 个 int64 索引，`Cast<int64_t, float>(outLocal, idxLocal, RoundMode::CAST_ROUND, 1)` count=1）：输出越界小整数 + "前一个期望值串位"——`shape=[100,50] dim=1` 用例 `期望=22, 实际=1561`（合法范围 0~49），同输入 keepdim=true/false 表现不一致；ST 首轮仅 22/101 通过。
- ARA 路径（批量输出，count=a0Align ≥64、256B 对齐）：输出形如 `4570842907728013644` 的巨大 int64——按 16 进制拆开（=0x3F7B83C4BF82FD14），高 32 位 0x3F7B83C4 是有效 float 位模式（≈0.983），低 32 位 0xBF82FD14 也是有效 float（≈-1.022）。**判别式：巨大 int64 的高低 32 位均为有效 float 位模式 = Cast 做的是位打包/搬运而非数值转换**，与本卡"把源数据当作较大类型数组处理"机制一致（4B float 源被按 8B int64 视角搬运，每个输出打包两个相邻 float）；count=1 小 count 档位另叠加块处理读写污染（串位/越界小整数）。

**2) 修复（标量替代，两路同构）**：AR 路径 `outLocal.SetValue(0, static_cast<int64_t>(globalIdx))`（不再经过 float/Cast）；ARA 路径 GetValue/SetValue 循环替代向量 Cast：

```cpp
        // Cast float index → int64 via scalar loop (Cast<int64_t,float> vector
        // API produces incorrect results on arch35 — float bits packed as int64)
        LocalTensor<int64_t> outLocal = outBufAra.Get<int64_t>();
        for (uint32_t i = 0; i < curA0Len; i++) {
            float idxVal = idxLocal.GetValue(i);
            outLocal.SetValue(i, static_cast<int64_t>(idxVal));
        }
        PipeBarrier<PIPE_V>();
```

索引输出量级远小于数据量级，标量代价可忽略；顺带把 `DataCopyExtParams.rsv = 0` 显式补上（`kernel_struct_data_copy.h` 里该结构默认构造为空、成员不清零，只设 blockCount/blockLen/stride 漏 rsv 会留未初始化字段）。

**3) 判别方法补充**：看到输出"巨大但像两个数拼起来"的 int64，第一反应做 16 进制拆位检查；二分替换——把唯一可疑的 `Cast<int64_t, float>` 换成标量 SetValue、其他不动，重编重装复测。

验证信号（复现增量，argmax 迭代二联调，均 tool_observed/950）：

- 修复前：ST Real 首轮 22/101；AR 用例 `期望=22, 实际=1561`（越界）等 mismatch≈98/100；ARA 输出 4570842907728013644（=0x3F7B83C4BF82FD14，高低 32 位均为有效 float 位模式）。
- AR 改 SetValue 后（同批含 int32 树归约修复）：ST 22/101→75/101，AR 路径 73/73 全部 bitwise-equal。
- ARA 改标量循环后：ST 升至 89/101；最终（含后续 ARA 标量回退）ST Real 101/101（100%）通过、UT 69/69 无回归。

失效边界（复现增量）：

- 仅实测 float→int64 方向与 count=1、对齐大 count（≥64）两档；int64→float 方向、int32→int64 等其他组合本增量未测（int32→int64 已在主卡预防节列为中招，来源为 先前增量 轨迹）。
- 标量替代假设索引值在 float 精确整数范围（<2^24）内（行号计数天然满足）；单行索引超 2^24 时 float 中间量本身不安全，应改 int/int64 中间量。
- CANN 后续版本是否修复该接口未验证；若升级后要用回向量化路径，需先用单测复验位模式。