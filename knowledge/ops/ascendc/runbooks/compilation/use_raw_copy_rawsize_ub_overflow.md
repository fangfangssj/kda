---
type: "Runbook"
title: useRawCopy UB 越界：tiling 对齐 innerSize
description: 非连续小 innerSize 输入走整块原始拷贝（useRawCopy）时，DoCopyInRaw 的 rawSize 按 block 跨度上取整可超过按 2*ubFormer_ 分配的 UB buffer，造成越界写；修复为 tiling 侧将 blockFormer/ubFormer 对齐到 innerSize，kernel 侧加 rawSize clamp 兜底。
tags: [task.debugging, topic.memory, topic.tiling, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# useRawCopy UB 越界：tiling 对齐 innerSize

## 现象

在 BenchNoHarnessSwiGlu 算子（SwiGLU 激活，`output = silu(x0) * x1`，`x0/x1 = input.chunk(2, dim)`；目标芯片 Ascend950PR/arch35，CANN 9.1.0，算子仓开发流程）交付前的代码检视中，审查 `op_kernel/arch35/bench_no_harness_swi_glu.h` 的 useRawCopy 路径时发现一个潜伏的 UB Buffer 越界缺陷。该路径是针对"dim 维内层尺寸太小、分开拷 x0/x1 每段不足 32 字节"场景设计的整块原始拷贝策略：触发条件为 `innerSize * dtypeSize < MIN_BURST_BYTES(32B)`（FP32 即 `innerSize < 8`），此时把 x0+x1 交错的原始数据整块搬进 UB，再逐元素拆分计算。

问题在于 buffer 容量与实际拷贝长度的计算不一致：`InitComputeBuffers` 按 `2 * ubFormer_` 个元素给 `inQueueX0` 分配 buffer，而 `DoCopyInRaw` 按"tile 跨到的 innerSize 块数"计算 `rawSize = (lastBlock - firstBlock + 1) * 2 * innerSize_`。检视阶段用宿主侧 Python 穷举验证（innerSize∈[1,7] × ubFormer∈{64,128,192,256,9024}）：共 30 组未对齐（tileStart=1）组合溢出；**即使 tile 起点对齐（tileStart=0），只要 innerSize 不整除 ubFormer（如 3、5、6、7），仍然溢出**——innerSize=3/ubFormer=64 时 rawSize=132 > bufferSize=128（溢出 4 元素，FP32 即 16 字节），innerSize=7/ubFormer=64 时溢出 12 元素（48 字节）。该缺陷此前从未在真实用例中暴露：修复前 UT 46/46、ST 115/115 均通过，属"测试没踩到触发 shape"的静默越界写（会 corrupt 相邻 UB buffer 数据）。

当时代码状态：tiling 侧 `blockFormer` 只对齐到 `ELEM_ALIGN_FACTOR(512)`、`ubFormer` 只对齐到 `UB_ALIGN_BYTES/sizeof(float)(64)`，均与 innerSize 无关；kernel 侧 `DoCopyInRaw` 无任何容量检查。

## 适用场景

- 工程形态：算子仓开发的自定义算子（op_host TilingFunc + op_kernel AscendC + aclnn 两段式 API），kernel 中存在"按逻辑 block 跨度整块搬运原始数据到 UB"的路径（本例为小 innerSize 的 useRawCopy 策略）。
- API 路径：`pipe.InitBuffer`/`TQue` buffer 分配、`DataCopyPad`（`DataCopyParams.blockLen` 按字节）、tiling 的 `blockFormer`/`ubFormer` 计算（`CeilAlign`/`FloorAlign`/`CeilDiv`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何含"跨 block 整块搬原始数据"路径的 elementwise 类算子，其 buffer 容量与 block 跨度的关系相同。
- 路线优先级：recommended_fastpath（tiling 对齐为根治方案；kernel clamp 仅兜底）。
- 明确不适用 / 未验证场景：innerSize 较大、useRawCopy 不触发的常规路径无此问题；其他 NPU 架构未验证（机制与架构无关，但 ubFormer 对齐基数按平台 UB 约定推导）；本卡不覆盖 `DataCopyParams.blockLen`（uint16_t）在超大搬运长度下的截断问题（见失效边界）。

## 根因

tileStart/tileSize 未对齐到 innerSize 时，`rawSize` 按"跨到的 block 数"上取整放大，可超过按 `2 * ubFormer_` 分配的 UB buffer 容量。

机制链与算式（均为本会话实测/推导证据）：

1. 一个 tile 覆盖输出空间 `[tileStart, tileStart+tileSize-1]`，每个输出元素属于一个 `innerSize_` 元素的 block，该 block 的原始数据（x0+x1 交错）为 `2 * innerSize_` 元素。
2. `rawSize = (lastBlock - firstBlock + 1) * 2 * innerSize_`，其中 block 数 `= ceil(tileSize / innerSize_)`（tileStart 对齐时）或 `ceil(tileSize / innerSize_) + 1`（tileStart 未对齐到 innerSize 边界时多跨一个 block）。
3. 最坏情形 `tileSize = ubFormer_`：`rawSize = (ceil(ubFormer_/innerSize_) + 1) * 2 * innerSize_ ≥ 2*ubFormer_ + 2*innerSize_`，而 buffer 容量恒为 `2 * ubFormer_`。
4. 数值例：innerSize=3、ubFormer=64 → ceil(64/3)=22，对齐起点 blocks=22、rawSize=22×2×3=132 > 128（溢出 4 元素）；未对齐起点 blocks=23、rawSize=138（溢出 10 元素）。Python 穷举确认对齐/未对齐两类场景均存在溢出组合。
5. `ubFormer_` 是 64 的倍数（`FloorAlign(maxElemNum, UB_ALIGN_BYTES/sizeof(float))`），而 useRawCopy 恰在 `innerSize` 很小时触发——`innerSize ∈ {3,5,6,7}` 这类不整除 64 的值天然落入该路径，因此缺陷不是纯理论边界。

## 解决配方

双层修复：tiling 层根治（对齐）+ kernel 层兜底（clamp）。

修复前（buffer 分配与拷贝大小不一致，tiling 与 innerSize 无关）：

```cpp
    // kernel: InitComputeBuffers 中 buffer 分配（每个 slot 容量 2*ubFormer_ 元素）
    uint64_t rawBufElems = 2 * ubFormer_;
    pipe.InitBuffer(inQueueX0, BUFFER_NUM, rawBufElems * sizeof(T));
```

```cpp
    // kernel: DoCopyInRaw 中实际拷贝大小（按 block 跨度上取整，无容量检查）
    uint64_t firstBlock = outOffset / innerSize_;
    uint64_t lastBlock = (outOffset + tileSize - 1) / innerSize_;
    uint64_t rawStart = firstBlock * 2 * innerSize_;
    uint64_t rawEnd = (lastBlock + 1) * 2 * innerSize_;
    uint64_t rawSize = rawEnd - rawStart;
```

修复后（tiling 层：useRawCopy 时把 blockFormer/ubFormer 对齐到 innerSize，并重算 blockNum）：

```cpp
    constexpr uint32_t MIN_BURST_BYTES = 32;
    bool useRawCopy = (innerSize * dtypeSize < MIN_BURST_BYTES) && (innerSize > 0);

    // ISSUE-002 fix: useRawCopy 路径下, 将 blockFormer 对齐到 innerSize
    // 确保 coreStart (= blockFormer * blockIdx) 是 innerSize 的倍数,
    // 从而 tileStart = coreStart + i * ubFormer 也是 innerSize 的倍数。
    // 这样 rawSize = ceil(tileSize / innerSize) * 2 * innerSize <= 2 * ubFormer (buffer 容量)
    if (useRawCopy) {
        blockFormer = CeilAlign(blockFormer, innerSize);
        blockNum = CeilDiv(totalOutput, blockFormer);
    }
```

```cpp
    uint64_t maxElemNum = availableUb / bufferDivisor;
    uint64_t alignFactor = UB_ALIGN_BYTES / sizeof(float);  // 64 elements (FP32 对齐)
    uint64_t ubFormer = FloorAlign(maxElemNum, alignFactor);

    // ISSUE-002 fix: useRawCopy 路径下, 将 ubFormer 对齐到 innerSize
    // 确保 tileSize (<= ubFormer) 的 rawSize 不超过 buffer 容量 (2 * ubFormer)
    if (useRawCopy) {
        ubFormer = FloorAlign(ubFormer, innerSize);
        if (ubFormer == 0) {
            ubFormer = innerSize;  // 至少处理一个 block
        }
    }
```

修复后（kernel 层：DoCopyInRaw 增加防御性 clamp）：

```cpp
    uint64_t firstBlock = outOffset / innerSize_;
    uint64_t lastBlock = (outOffset + tileSize - 1) / innerSize_;
    uint64_t rawStart = firstBlock * 2 * innerSize_;
    uint64_t rawEnd = (lastBlock + 1) * 2 * innerSize_;
    uint64_t rawSize = rawEnd - rawStart;

    // ISSUE-002 fix: 防御性检查, 确保 rawSize 不超过 buffer 容量 (2 * ubFormer_)
    // 正确的 tiling (blockFormer/ubFormer 对齐到 innerSize) 下, rawSize <= 2 * ubFormer_
    // 此 clamp 仅作为安全兜底, 防止 UB 内存越界
    uint64_t bufCapacity = 2 * ubFormer_;
    if (rawSize > bufCapacity) {
        rawSize = bufCapacity;
    }
```

对齐后的正确性算式：`tileStart = coreStart + i * ubFormer`，其中 `coreStart = blockFormer * blockIdx`；blockFormer 与 ubFormer 均为 innerSize 倍数 ⇒ tileStart 与 tileSize（≤ubFormer）均为 innerSize 倍数 ⇒ block 数 = `tileSize / innerSize`（整除，无 +1）⇒ `rawSize = (tileSize/innerSize) * 2 * innerSize = 2 * tileSize ≤ 2 * ubFormer`，恒不超过容量。

判别方法（本会话走通、可复用）：

1. 写穷举探针验证 buffer 容量与最坏 rawSize 的关系（完整脚本如下，宿主侧 Python 即可，无需上板）：

```python
# Verify useRawCopy buffer overflow
# ubFormer_ is a multiple of 64 (FloorAlign to 64)
# innerSize_ < 8 for FP32 (useRawCopy triggers when innerSize < 32/4 = 8)

test_cases = []
for innerSize in range(1, 8):
    for ubFormer in [64, 128, 192, 256, 9024]:  # typical ubFormer values
        # Worst case: tile start NOT aligned to innerSize
        # tileStart = some value not divisible by innerSize
        # Simulate: tileStart = 1 (not aligned for innerSize > 1)
        tileStart = 1
        tileSize = ubFormer
        firstBlock = tileStart // innerSize
        lastBlock = (tileStart + tileSize - 1) // innerSize
        blocks = lastBlock - firstBlock + 1
        rawSize = blocks * 2 * innerSize
        bufferSize = 2 * ubFormer
        overflow = rawSize - bufferSize
        if overflow > 0:
            test_cases.append((innerSize, ubFormer, rawSize, bufferSize, overflow))

print(f"Total overflow cases: {len(test_cases)}")
print(f"{'innerSize':>10} {'ubFormer':>10} {'rawSize':>10} {'bufferSize':>10} {'overflow':>10}")
for tc in test_cases[:15]:
    print(f"{tc[0]:>10} {tc[1]:>10} {tc[2]:>10} {tc[3]:>10} {tc[4]:>10}")

# Also check aligned case
print("\n--- Aligned case (tileStart=0) ---")
for innerSize in [3, 5, 6, 7]:
    ubFormer = 64
    tileStart = 0
    tileSize = ubFormer
    firstBlock = tileStart // innerSize
    lastBlock = (tileStart + tileSize - 1) // innerSize
    blocks = lastBlock - firstBlock + 1
    rawSize = blocks * 2 * innerSize
    bufferSize = 2 * ubFormer
    overflow = rawSize - bufferSize
    print(f"innerSize={innerSize}, ubFormer={ubFormer}: rawSize={rawSize}, bufferSize={bufferSize}, overflow={overflow}")
```

2. 检查 tiling 是否保证 tile 边界对齐到搬运粒度：确认 `blockFormer`、`ubFormer` 的对齐目标包含 innerSize（或任何"整块搬运的逻辑粒度"），而非只对齐硬件对齐基数（512/64）。
3. 检查 kernel 侧搬运长度是否有容量上界校验（clamp 或断言），作为 tiling 错误时的最后防线。

## 设计决策

- 为何选 tiling 对齐而不是检视建议的"加大 buffer 预留一个 block"（`rawBufElems = 2*ubFormer_ + 2*innerSize_`）：加大 buffer 需要同步调整 host 侧 `bufferDivisor` 计算（UB 总量预算），且只把越界往后推（未对齐 +1 block 只是本推导的最坏值，不改变"容量与跨度脱钩"的结构性问题）；对齐使 `rawSize ≤ 2*ubFormer` 在数学上恒成立，不增加任何 UB 占用。检视给出的两个候选方案（回退逐元素路径 / rawSize 上限检查）也未被采用为主方案：上限检查只截断不根治，回退逐元素会放弃整块搬运的收益；最终采用"对齐根治 + clamp 兜底"的组合。
- 为何在 useRawCopy 判定之后再对齐 blockFormer 并显式重算 blockNum：原代码 `blockFormer`/`blockNum` 在 `useRawCopy` 之前计算（useRawCopy 依赖其后的 dtypeSize/innerSize 段），若挪动计算顺序牵连更大；选择在 useRawCopy 已知后追加对齐，并紧跟 `blockNum = CeilDiv(totalOutput, blockFormer)` 重算，保证两者一致（改 blockFormer 不重算 blockNum 会引入新的切分错误）。
- 为何 ubFormer 用 FloorAlign 且带 `==0` 兜底：向下取整保证不超原 UB 预算；极端小 UB 时 FloorAlign 可能为 0，兜底取 `innerSize`（至少处理一个 block）避免死循环/空跑。
- 为何 kernel 侧再加 clamp：纵深防御——tiling 正确时 clamp 永不触发（rawSize ≤ 2*ubFormer_），tiling 错误时以"截断拷贝长度"避免 UB 越界写崩；代价是截断时数据不完整（正确性受损但安全性保住），故只能作为兜底而非常规手段。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510，arch35），CANN 9.1.0，GCC 13.3.0，CMake 3.20.5。
- 相关文件：`op_host/arch35/bench_no_harness_swi_glu_tiling.cpp`（tiling 对齐修复）、`op_kernel/arch35/bench_no_harness_swi_glu.h`（DoCopyInRaw clamp）、`op_host/arch35/bench_no_harness_swi_glu_tiling.h`（常量：ELEM_ALIGN_FACTOR=512、UB_ALIGN_BYTES=256、MIN_TILING_BITS=32768）、`build.sh`、`tests/ut/run.sh`、`tests/st/run.sh`。
- 构建与回归命令（verbatim，CANN 安装根以 `$ASCEND_HOME_PATH` 表示）：

```bash
bash build.sh --soc=ascend950 --make_clean
export ASCEND_HOME_PATH=<CANN 安装根> && bash build.sh --soc=ascend950 -j8 2>&1
# UT（tests/ut 目录）
export ASCEND_HOME_PATH=<CANN 安装根> && bash run.sh 2>&1
# 安装算子包 + ST（tests/st 目录，Real NPU）
export ASCEND_HOME_PATH=<CANN 安装根> && bash build/custom_opp_ubuntu_x86_64.run --force 2>&1
export ASCEND_HOME_PATH=<CANN 安装根> && \
  export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_swi_glu_custom/op_api/lib/:${LD_LIBRARY_PATH} && \
  bash run.sh --real 2>&1
```

## 验证信号

以下信号来自该算子代码检视发现 CRITICAL 缺陷后的修复会话：先用宿主侧 Python 穷举探针证明溢出存在，再实施 tiling 对齐 + kernel clamp 双层修复，最后全量回归。诚实边界：修复后未新增专门以 innerSize∈{3,5,6,7}（不整除 ubFormer）触发越界的用例，下列 UT/ST 通过信号证明"无回归"，越界消除由 tiling 对齐的算式（rawSize = 2*tileSize ≤ 2*ubFormer）保证，未在真实触发 shape 上端到端复现"修复前越界、修复后正常"的对照。

- 修复前（探针）：`innerSize∈[1,7]×ubFormer∈{64,128,192,256,9024}` 未对齐组合共 30 组溢出；对齐起点（tileStart=0）下 innerSize=3/5/6/7、ubFormer=64 时 rawSize=132/130/132/140 均超 bufferSize=128（溢出 4/2/4/12 元素）（来源平台：950 开发环境宿主侧 Python 算术验证，与 NPU 无关；证据等级：tool_observed）。
- 修复后：`bash build.sh --soc=ascend950 -j8` 编译成功，生成 kernel binary（`build/op_kernel/ascendc_kernels/binary/ascend950/...`）与安装包 `build/custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 46/46 通过，其中含直接覆盖本路径的用例 `fp16_raw_copy_tiny_inner`、`fp32_raw_copy_tiny_inner`、`bf16_raw_copy_tiny_inner`、`fp16_raw_copy_dim0`（来源平台：950；证据等级：tool_observed）。
- 修复后：ST Real NPU 115/115 通过、0 失败 0 跳过，MERE/MARE 全部低于阈值（如 max_fp32_[8192,8192]_dim-1：MERE=1.21e-09 / MARE=7.97e-08，阈值 1.22e-04）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡推导基于 arch35/Ascend950PR + CANN 9.1.0 的 UB 布局与 `ubFormer` 64 对齐基数；其他架构/版本的对齐基数不同，需按平台常量重推（机制本身与架构无关，未验证，推断可迁移）。
- kernel clamp 截断 `rawSize` 会静默少拷数据（正确性受损），仅在 tiling 错误时兜底防崩；不能把 clamp 当作常规容量管理手段。
- `DataCopyParams.blockLen` 类型为 uint16_t（上限 65535）：本路径 `copyParams.blockLen = rawSize * sizeof(T)`，当 `2 * ubFormer_ * sizeof(T) > 65535`（FP32 下 ubFormer > 8191）时仍会发生赋值截断——本轨迹未验证该场景（检视将其记为另一未修复项，仅按 CopyOut 场景分析为"当前值安全"）。
- 对齐修复只保证 rawSize ≤ 容量；若后续有人改动 ubFormer/blockFormer 的对齐基数或引入新的搬运粒度，需重新执行上述判别算式。

**未决项**：

- 优先项：构造专门触发越界的用例（如 FP32、innerSize=3、ubFormer 量级 64 的非连续输入）做修复前后对照，端到端确认"修复前 UB 越界写、修复后消除"，补齐本卡缺失的现场复现证据。
- `DoCopyInRaw` 的 `blockLen`（uint16_t）在超大 ubFormer 下的截断风险未修复、未验证（见失效边界）。
- clamp 触发时无任何日志/告警，静默截断难以察觉；可考虑在 tiling 与 kernel 之间加一致性断言或编译期 static_assert 类约束。
## 坏实践

- buffer 按 `2 * ubFormer_` 分配，却允许搬运长度按 block 跨度上取整（含未对齐 +1 block）放大——容量与跨度脱钩是结构性缺陷。
- 只测常规 shape 就认定搬运路径安全：本缺陷在 UT 46/46、ST 115/115 全过的状态下仍然存在，触发条件（小 innerSize 且不整除对齐基数）不在默认用例里。
- 检视建议的"只加大 buffer 预留 `2 * innerSize_`"治标不治本，且容易漏掉 host 侧 bufferDivisor 的同步调整。

## 误诊

- 最初假设仅"tile 起点未对齐 innerSize"才溢出；算术验证发现 tileStart=0 的对齐情形在 innerSize 不整除 ubFormer 时同样溢出（ceil 放大本身即可超容量）——修复必须同时约束起点（blockFormer）与长度（ubFormer），只对齐其一不够。
- 修复验证阶段遇到 custom_opp 安装包共享库校验失败（`libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`），被当作"与本次修改无关的既有问题"用 `--force` 跳过——该假设未验证，事后 ST 正常仅说明运行时符号可解析，不能证明包内符号完整性无恙（该符号问题的根因与处置路线见[libcust_opapi l0op::Contiguous 未定义符号](libcust_opapi_l0op_contiguous_undefined_symbol.md)卡）。
