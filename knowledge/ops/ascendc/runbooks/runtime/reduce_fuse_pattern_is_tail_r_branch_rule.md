---
type: "Runbook"
title: 合轴 fuse R 无尾 A → tail-R key=1（非 key=0）
description: reduce 算子合轴四步后 A 起头严格交替，isTailR=（axisNum 为偶数）；多轴 reduce 合轴成单段 R 且无尾随 A（如 [2,3,4] axes=[1,2]）是 AR pattern、key=1，不是 key=0；补 R 增广（PadRIfPureA）只在完全没有 R 轴时触发，负轴索引须先归一化。
tags: [task.debugging, topic.tiling, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# 合轴 fuse R 无尾 A → tail-R key=1（非 key=0）

## 现象

EuclideanNorm（L2 范数 reduce）算子按"合轴 pattern → tilingKey"路由到不同 kernel 分支（tail-A=0 / tail-R=1 / empty=2 / group=4,5）。Branch-0（tail-A, key=0）验收时 ST 一批用例全 0，开发分支设计文档的用例表把 `[2,3,4] axes=[1,2]`、`[1024] axes=[]`、`[8] axes=[0]` 都标成 key=0（tail-A）。但按合轴逻辑手推：`[2,3,4] axes=[1,2]` 中 dim1/dim2 都是 reduce 轴，合轴（fuse）成一根 R=12，前面还有 A=2，没有尾随 A——pattern 是 `A R`（axisNum=2，偶数），应判 tail-R（key=1）。设计文档的用例表本身把分支标错了。需要一条不依赖被测代码的判支方法来仲裁。

## 适用场景

- 工程形态：多分支 reduce 算子开发（合轴预处理 → tilingKey 编码 → 分支 kernel），及为这类算子编写/核对 ST 用例的分支归属。
- API 路径：host 侧合轴四步预处理（DropSizeOneAxes → FuseAxis → PadLeadingOneA → PadRIfPureA）+ `context->SetTilingKey((isGroup<<2)|(isEmpty<<1)|isTailR)`；仲裁探针：`$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/<op>/arch35/*_tiling_key.h`（内置同名算子的模板参数声明）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0。
- 适用粒度：operator_family——合轴判支规则适用于采用"A 起头严格交替 pattern + isTailR 编码"的 reduce 算子族。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不含合轴预处理、直接按原始 axes 切分的 reduce 实现；group/empty 分支的编码细节（本卡只裁 tail-A vs tail-R）。

## 判别方法

判支规则（合轴四步执行后）：

1. **负轴归一化先行**：`axes=[((a % rank) + rank) % rank for a in axes]`（再去重、排序；空 axes + rank>0 → 全轴 reduce）。负轴不归一化就做 membership 判定，会把 pattern 算错（见验证信号）。
2. 四步预处理后 pattern 必形如 `A R A R …`（A 起头、严格交替、每轴 size≥2，首部补 1 除外）；校验 `axisNum ∈ [2,9]`、`isReduceAxis[i] == (i%2==1)`。
3. **`isTailR = (axisNum % 2 == 0)`**：偶数轴（尾轴是 R）→ tail-R；奇数轴（尾轴是 A）→ tail-A。
4. **关键陷阱：多轴合轴产出的"单段 R + 无尾 A"是 tail-R**。`[2,3,4] axes=[1,2]` → `[(2,A),(12,R)]` → axisNum=2 → key=1，不是 key=0。
5. **补 R 增广（PadRIfPureA）只在"完全没有 R 轴"的纯 A 退化时触发**：A=1（∏shape==1）末尾补 R=1 变 AR（tail-R）；A>1 前置 `[A=1,R=1]` 变 ARA（tail-A），禁末尾补 R=1。已经有任何 R 轴时该步绝不触发。

仲裁探针（两路独立证据）：

- 查内置同名算子的 tiling_key 头文件（工具链权威声明）：

```bash
sed -n '1,60p' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/euclidean_norm/arch35/euclidean_norm_tiling_key.h
```

实测输出（节选，直接判死争议）：`isTailR == 1 → 偶数轴 pattern，UB 内是 AR（尾 R）；isTailR == 0 → 奇数轴 pattern，UB 内是 RA（尾 A）`。

- 在 ST 里内置独立合轴 oracle（不调用被测算子），按真实 tilingKey 路由每个用例（完整函数）：

```cpp
// Independent 合轴 oracle: mirrors DESIGN §5.4.4 four-step preprocessing
// (DropSizeOneAxes → FuseAxis → PadLeadingOneA → PadRIfPureA) and §5.2.3
// SetTilingKey to predict the tilingKey a case MUST route to.
//
// Used to classify each ST case by the branch it actually belongs to:
//   key=0  → Normal Tail-A (Branch-0, this suite's scope)
//   key=1  → Normal Tail-R (Branch-1, Task 27/28 — not implemented here)
//
// This is an independent oracle (does not call the op); it lets the suite
// honestly separate Branch-0 (key=0) precision cases from Branch-1 (key=1)
// cases that the test author originally mis-filed under Branch-0 (per the
// erroneous DESIGN-BRANCH-0 §9.1 table, which conflicts with both the
// authoritative reduce-pattern-preprocess.md spec and the built-in
// euclidean_norm tiling_key.h — both confirm [2,3,4]/axes=[1,2], [1024]/axes=[],
// [8]/axes=[0] are AR patterns → axisNum=2 → isTailR=true → key=1).
static int computeExpectedTilingKey(const Case& tc)
{
    // normalized axes (negative wrap, dedup, sort) — empty → full reduce
    int64_t normAxes[kMaxAxes];
    int nAxes;
    if (tc.nAxes == 0 && tc.inRank > 0) {
        nAxes = tc.inRank;
        for (int d = 0; d < tc.inRank; ++d) normAxes[d] = d;
    } else {
        nAxes = normalizeAxes(tc.axesRaw, tc.nAxes, tc.inRank, normAxes);
    }

    // seed (size, isReduce) per dim
    struct Pair { int64_t size; bool isR; };
    std::vector<Pair> axes;
    axes.reserve(tc.inRank);
    for (int d = 0; d < tc.inRank; ++d) {
        bool isR = false;
        for (int i = 0; i < nAxes; ++i) if (normAxes[i] == d) { isR = true; break; }
        axes.push_back({tc.inShape[d], isR});
    }

    // Step 1: drop size==1 (keep one A=1 placeholder if everything drops)
    std::vector<Pair> kept;
    for (const auto& a : axes) if (a.size != 1) kept.push_back(a);
    if (kept.empty()) kept.push_back({1, false});

    // Step 2: fuse consecutive same-type
    std::vector<Pair> fused;
    for (const auto& a : kept) {
        if (!fused.empty() && fused.back().isR == a.isR) fused.back().size *= a.size;
        else fused.push_back(a);
    }

    // Step 3: pad leading A=1 if first is R
    if (!fused.empty() && fused.front().isR) fused.insert(fused.begin(), {1, false});

    // Step 4: pad R only if pure-A (no R axis at all)
    bool anyR = false;
    for (const auto& a : fused) if (a.isR) { anyR = true; break; }
    if (!anyR) {
        if (fused.size() == 1 && fused.front().size == 1) {
            fused.push_back({1, true});                       // A=1 → AR (tail-R)
        } else {
            fused.insert(fused.begin(), {1, true});           // prepend R=1
            fused.insert(fused.begin(), {1, false});          // prepend A=1 → ARA (tail-A)
        }
    }

    int axisNum = (int)fused.size();
    bool isTailR = (axisNum > 0) && (axisNum % 2 == 0);
    // empty short-circuit not triggered by these cases (no zero-size dims)
    return isTailR ? 1 : 0;
}
```

## 系统信息

- 平台/工具链：Ascend950PR（arch35），CANN 9.1.0，npu-smi 25.7.rc1。
- 相关文件：`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（被仲裁的合轴/SetTilingKey 实现）、`tests/aclnn/test_aclnn.cpp`（oracle 路由）、`$ASCEND_OPP_PATH/built-in/.../euclidean_norm/arch35/euclidean_norm_tiling_key.h`（内置仲裁证据）、reduce pattern 预处理规范文档（skill 范式库 `references/paradigms/reduction/references/reduce-pattern-preprocess.md`）。

## 设计决策

- 仲裁证据选了"内置同名算子 tiling_key 头文件 + 范式规范文档"两个独立来源，而不是被测实现自证——被测实现的合轴正是争议对象。
- oracle 以完整四步预处理镜像（而非只查 axes 个数）实现，顺带覆盖负轴归一化、size-1 轴丢弃、纯 A 退化三条边界，测试作者手写用例表时可整体复用。

## 验证信号

判支过程与结果（Ascend950PR 实测）：

- 内置 tiling_key.h 头文件声明原文：`isTailR == 1 → 偶数轴 pattern，UB 内是 AR（尾 R）；isTailR == 0 → 奇数轴 pattern，UB 内是 RA（尾 A）`；模板选择为 normal×2（isTailR 0/1）+ empty×1 + group×2 共 5 组合（来源平台：950；证据等级：tool_observed）。
- 范式规范文档原文：步骤 4"补 R 增广"仅"若上一步完成后仍无 R 轴（纯 A 场景）"触发（来源平台：950；证据等级：tool_observed）。
- ST 按真实 key 路由后：`[2,3,4] axes=[1,2]`、`[1024] axes=[]`、`[8] axes=[0]`、`[17] axes=[-1]` 等被判 key=1（DEFERRED 到 tail-R 分支），Branch-0 (key=0) 精度 5/5 PASSED、run.sh exit 0（来源平台：950；证据等级：tool_observed）。
- tail-R 分支 kernel 落地后，上述被判 key=1 的 aclnn 用例 18/18 PASSED（maxRel≤8.467e-08）——反向证明判支正确（来源平台：950；证据等级：tool_observed）。
- 负轴归一化陷阱：手算 oracle 脚本未归一化 `axes=[-1]`（rank=2）时误判 ar_basic_expand "NOT tail-R (axisNum=3)" 断言失败；按 `((a%rank)+rank)%rank` 归一化后 axisNum=2（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 规则以"合轴后 A 起头严格交替 + isTailR=(axisNum 偶数)"的编码方案为前提；若另一算子采用不同的 tilingKey 编码（如按 reduce 轴个数分段），判式需按其 tiling_key.h 重推。
- axisNum 上界（本算子 [2,9]）是算子设计约束，不是平台约束；超界 pattern 的行为未验证。
- group/empty 短路（空 tensor、分组 reduce）在 key 编码里另有位（isGroup<<2 | isEmpty<<1），本卡只裁 normal 分支的 tail-A/tail-R。

**未决项**：

- 设计文档用例表的 key 标注错误已在测试侧以 oracle 路由纠偏，文档本身的修正未在本切片完成。
## 坏实践

- 照抄设计文档用例表的分支标注而不自查合轴结果——本例中用例表本身标错 8 个用例的 key，若按表验收会把 tail-R 用例当 Branch-0 失败去修 Branch-0 kernel（方向完全反掉）。
- 负轴索引不归一化直接做 isReduce membership 判定（`-1` 永远不等于任何 `d∈[0,rank)`），pattern 少标一根 R 轴。
