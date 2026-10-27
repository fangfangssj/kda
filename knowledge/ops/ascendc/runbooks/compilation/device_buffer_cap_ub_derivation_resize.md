---
type: "Runbook"
title: "设备侧 N*E 缓冲上限须按 UB 推导扩容"
description: kernel 常量表里 device 侧 N*E 缓冲上限按早期保守场景写死（如 4096），tiling 会把 N*E 超限的合法大规模输入判为 GRAPH_FAILED；该上限是"缓冲放得进 UB"的自估保护值而非硬件约束，扩容前必须按最差 shape 推导 UB 占用，并同步迁移 UT 边界用例。
tags: [task.debugging, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# 设备侧 N*E 缓冲上限须按 UB 推导扩容

## 现象

在 Ascend950PR（CANN 9.1.0，arch35）上为 MoE 重排类自定义算子补齐 L2 边界用例时，用例表里出现了 N=128、E=64（N*E=8192）和 N=64、E=128（N*E=8192）这样的大规模 rank×expert 组合。审查 `op_kernel/arch35/*_tiling_data.h` 时发现 device 侧缓冲上限常量按早期保守场景写死为 4096，tiling 中的检查是：

```cpp
uint32_t ne = N * E;
OP_CHECK_IF(ne > BENCH_MOE_DEVICE_MAX_NE,
            OP_LOGE(context, "N*E=%u exceeds device UB limit %u", ne, BENCH_MOE_DEVICE_MAX_NE),
            return ge::GRAPH_FAILED);
```

也就是说这批 L2 用例上板会直接被 tiling 拒绝（GRAPH_FAILED）。当时的 UT 也固化了旧边界：既有用例 `test_failed_ne_exceeds_limit` 断言 N*E=65*65=4225 > 4096 时 tiling 返回 GRAPH_FAILED，且该 UT 在当轮全量回归中是通过的（op_host UT 74/74）——即"拒绝 N*E>4096"是当时实际生效的行为。

## 适用场景

- 工程形态：算子仓开发（op_kernel/op_host + UT + ST），aclnn 自定义算子的 tiling 约束维护。
- API 路径：op_host TilingFunc 中的 shape 约束检查、op_kernel 侧 tiling data 常量表、UT tiling 边界用例。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有把"随输入维数线性增长的 device 侧缓冲（如 N*E 矩阵/偏移表）放进 UB"的算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：上限本身来自硬件/驱动约束（而非自估 UB 容量）的场景；其他 arch 的 UB 总容量未验证（本卡数值按 248KB 推导）。

## 根因

device 侧 N*E 缓冲上限常量是按早期"实际场景 N×E ≤ 4096"的保守估算写死的，没有随大规模用例重新推导 UB 容量；该上限并不是硬件约束，而是"matrix+srcOffset 等设备缓冲放得进 UB"的自估保护值。

机制账（int64 matrix + int32 srcOffset，32 字节对齐）：

- matrixBuf = CeilAlign(N*E × 8, 32)，srcOffsetBuf = CeilAlign(N*E × 4, 32)，expertNumBuf = CeilAlign(E × 8, 32)；
- 可用 UB = 248KB − 8KB 保留 ≈ 240KB；
- 剩余空间按 per-token 成本（CeilAlign(H×dtype,32) + idx + scales）折算 tileTokens（封顶 4095）；
- 最差场景 N=128、E=128、H=16383、fp16：matrix 128KB + srcOffset 64KB + expertNum 1KB + tile 缓冲 + 保留 8KB ≈ 233KB < 248KB，故 N*E=16384 可行；N*E=8192 场景约 185KB，余量更大。

## 解决配方

第一步，用可复现脚本按最差 shape 推导 UB 占用（本会话实际执行的验证脚本，输出 `OK/OVERFLOW`）：

```python
UB_TOTAL = 248 * 1024  # 253952
RESERVED = 8 * 1024     # 8192

def ceil_align(x, align):
    return ((x + align - 1) // align) * align

scenarios = [
    ('L2_006 N=128,E=64,H=32,fp16', 128*64, 32, 2),
    ('L2_007 N=64,E=128,H=32,fp16', 64*128, 32, 2),
    ('N=128,E=128,H=32,fp16', 128*128, 32, 2),
    ('N=128,E=128,H=16383,fp16(worst)', 128*128, 16383, 2),
    ('N=128,E=128,H=16383,int8(worst)', 128*128, 16383, 1),
    ('L2_001 A=1,H=1,N=1,E=1', 1, 1, 2),
    ('L2_005 A=64,H=16383,N=4,E=4', 16, 16383, 2),
]

for name, ne, H, token_size in scenarios:
    matrix_buf = ceil_align(ne * 8, 32)
    src_offset_buf = ceil_align(ne * 4, 32)
    E = 128 if ne >= 128 else (ne // 1 if ne == 1 else 4)
    expert_num_buf = ceil_align(E * 8, 32)
    device_buf_total = matrix_buf + src_offset_buf + expert_num_buf
    available_ub = UB_TOTAL - RESERVED
    if available_ub <= device_buf_total:
        print(f'{name}: FAIL (deviceBuf {device_buf_total} > available {available_ub})')
        continue
    usable_ub = available_ub - device_buf_total
    h_aligned = ceil_align(H * token_size, 32)
    per_token_cost = h_aligned + 32 + 32  # idx + scales (CeilAlign(4,32)=32)
    tile_tokens = max(1, usable_ub // per_token_cost)
    if tile_tokens > 4095:
        tile_tokens = 4095
    # kernel actual usage
    token_size_actual = ceil_align(tile_tokens * H * token_size, 32)
    idx_size = ceil_align(tile_tokens * 4, 32)
    scales_size = ceil_align(tile_tokens * 4, 32)
    tile_buf_total = token_size_actual + idx_size + scales_size
    total_ub = RESERVED + device_buf_total + tile_buf_total
    status = 'OK' if total_ub <= UB_TOTAL else 'OVERFLOW!'
    print(f'{name}: ne={ne}, tileTokens={tile_tokens}, totalUB={total_ub}/{UB_TOTAL} ({total_ub/1024:.0f}KB/{UB_TOTAL/1024:.0f}KB) {status}')
```

实测输出（最差 233KB < 248KB，全部 OK）：

```text
L2_006 N=128,E=64,H=32,fp16: ne=8192, tileTokens=1144, totalUB=189888/253952 (185KB/248KB) OK
N=128,E=128,H=32,fp16: ne=16384, tileTokens=376, totalUB=232896/253952 (227KB/248KB) OK
N=128,E=128,H=16383,fp16(worst): ne=16384, tileTokens=1, totalUB=238656/253952 (233KB/248KB) OK
N=128,E=128,H=16383,int8(worst): ne=16384, tileTokens=2, totalUB=238656/253952 (233KB/248KB) OK
```

第二步，修改常量（before/after，`op_kernel/arch35/*_tiling_data.h`）：

```cpp
// before
constexpr uint32_t BENCH_MOE_MAX_N = 128;
constexpr uint32_t BENCH_MOE_MAX_E = 128;
constexpr uint32_t BENCH_MOE_MAX_NE = BENCH_MOE_MAX_N * BENCH_MOE_MAX_E; // 16384
// Device 侧 UB 能容纳的 matrix + srcOffset 上限（int64 matrix + int32 srcOffset）
// 实际场景 N×E ≤ 4096（matrix 32KB + srcOffset 16KB = 48KB），可安全放入 UB
constexpr uint32_t BENCH_MOE_DEVICE_MAX_NE = 4096;

// after
constexpr uint32_t BENCH_MOE_MAX_N = 128;
constexpr uint32_t BENCH_MOE_MAX_E = 128;
constexpr uint32_t BENCH_MOE_MAX_NE = BENCH_MOE_MAX_N * BENCH_MOE_MAX_E; // 16384
// Device 侧 UB 能容纳的 matrix + srcOffset 上限（int64 matrix + int32 srcOffset）
// UB 容量验证（248KB）: N*E=16384 最差场景（H=16383, fp16, HAS_SCALES）
//   matrixBuf=128KB + srcOffsetBuf=64KB + expertNumBuf=1KB + tileBuf=32KB + reserved=8KB = 233KB < 248KB
// 覆盖 L2 边界场景: N=128,E=64 / N=64,E=128 (N*E=8192) 及 N=128,E=128 (N*E=16384)
constexpr uint32_t BENCH_MOE_DEVICE_MAX_NE = 16384;
```

第三步，同步 UT 边界用例（`tests/ut/op_host/test_*_tiling.cpp`）——超限用例改到新边界之上，并在新边界两侧补 pass/fail 用例：

```cpp
// 超限用例：N*E = 200*100 = 20000 > 新上限 16384 -> GRAPH_FAILED
TEST_F(MoeReRoutingTilingTest, test_failed_ne_exceeds_limit)
{
    // N*E = 200*100 = 20000 > BENCH_MOE_DEVICE_MAX_NE (16384) -> GRAPH_FAILED
    // Need A >= N*E for valid input
    int64_t N = 200;
    int64_t E = 100;
    int64_t A = N * E;
    vector<int32_t> numData(N * E, 1);

// 新增边界用例（节选，N*E=16384 恰好等于新上限应通过）：
// N*E = 16384 (exactly at new limit) -> should pass
TEST_F(MoeReRoutingTilingTest, test_ne_boundary_exactly_16384_pass)
{
    int64_t N = 128;
    int64_t E = 128;
    int64_t A = N * E;  // A = 16384
    vector<int32_t> numData(N * E, 1);

    TilingInfo info;
    ASSERT_TRUE(RunTilingInt32({A, 32}, {N, E}, ge::DT_FLOAT16, numData, false, info));
    auto* tiling = reinterpret_cast<const BenchNoHarnessMoeReRoutingTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->N, static_cast<uint32_t>(N));
    EXPECT_EQ(tiling->E, static_cast<uint32_t>(E));
    // N*E=16384 reserves large device UB (matrix 128KB + srcOffset 64KB),
    // but tileTokens should still be > 0 (verified: 376 for H=32,fp16).
    EXPECT_GT(tiling->tileTokens, 0U);
    EXPECT_LE(tiling->tileTokens, 4095U);
}

// N*E = 16385 (just over new limit) -> should fail
TEST_F(MoeReRoutingTilingTest, test_ne_boundary_16385_fail)
{
    // N*E = 16385 > BENCH_MOE_DEVICE_MAX_NE (16384) -> GRAPH_FAILED
    int64_t N = 16385;
    int64_t E = 1;
    int64_t A = N * E;
    vector<int32_t> numData(N * E, 1);
```

（另有 `test_ne_boundary_exactly_4096_pass` 保留为界内用例。）

## 设计决策

- 为什么先跑 UB 推导再改常量：该上限不是硬件约束，而是"device 侧缓冲放得进 UB"的自估保护值；不推导就扩会把越界风险带进 kernel。
- 放弃的候选：不改常量、把大 N*E 场景在 tiling 里走 GM 侧缓冲路径——改动面大且本代场景 UB 装得下，未采用。
- 当前方案属 recommended_fastpath：一处常量 + UT 边界同步，风险由推导脚本与上板回归兜底。

## 预防

- 自估上限常量须在注释里保留推导账（最差 shape 的逐 buffer 字节数），换 buffer 布局或换 arch 时按公式重跑脚本，不能照抄常量。
- 只改常量不动 UT——旧边界用例（4225>4096 拒绝）会误挂，或被顺手删除而非迁移到新边界，丢失回归保护。

## 验证信号

来源说明：该算子 L2 边界用例补齐与上板回归的开发会话（Ascend950PR，CANN 9.1.0），修改 tiling 常量并同步 UT 后的实测。

- 修改前一轮全量 op_host UT 74/74 通过，其中既有 `test_failed_ne_exceeds_limit`（N*E=4225>4096 → GRAPH_FAILED）生效，证明旧上限确实拒绝 N*E>4096（来源平台：950；证据等级：tool_observed）。
- 扩容后 NE 相关 UT 7/7 通过，含 `test_ne_boundary_exactly_16384_pass`（通过）与 `test_ne_boundary_16385_fail`（拒绝）（来源平台：950；证据等级：tool_observed）。
- op_host UT 全量 75/75 通过（来源平台：950；证据等级：tool_observed）。
- NPU Real 模式 ST 回归：L0 12/12、L1 45/45 无回归；L2 单独运行时 N*E=8192 的 L2_006/L2_007 边界用例精度比对全部通过（fp16 MERE=0.00e+00, MARE=0.00e+00）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡数值（248KB UB、4096→16384、233KB 最差占用）按本算子缓冲布局与 arch35 推导；换算子/换 arch 必须重跑推导脚本，不能照抄常量（其他平台未验证，推导方法推断可迁移）。
- 上限提升只覆盖"UB 装得下"的范围；更大数据量应改走 GM 侧缓冲设计（本会话未实现，未验证）。
- int64 输入时 matrixBuf 按 int64 展开，UB 消耗翻倍，N×E 上限相应减半（按公式推导，未单独实测）。

**未决项**：

- 无（根因与边界均已闭合；GM 侧大缓冲路径为独立设计议题，见失效边界）。
## 坏实践

- 只改常量不跑最差 shape 的 UB 推导——上限从自估值变成拍脑袋值。
- 只改常量不动 UT 边界用例，或把旧边界用例直接删除而非迁移到新边界。

## 误诊

- 并发的测试侧子任务按记忆中的旧值 4096 预判 L2_006/L2_007 必被 tiling 拒绝；实际另一子任务已把常量改为 16384，上板全部通过。并发修改同一工作区时，结论必须以当时 grep 源码的实测值为准（该误判未造成代码改动，属判断偏差）。
