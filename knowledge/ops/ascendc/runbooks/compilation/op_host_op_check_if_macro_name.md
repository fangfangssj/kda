---
type: "Runbook"
title: op_host 层 OP_CHECK 未定义须用 OP_CHECK_IF
description: op_host（InferShape/Tiling）源码使用不存在的 OP_CHECK 宏导致编译失败并连锁 OP_LOGE 处 expected primary-expression before 'do'；op_common/log/log.h 只提供 OP_CHECK_IF 与 OP_CHECK_NULL_WITH_CONTEXT，aclnn/opdev 层才有 OP_CHECK。
tags: [task.debugging, topic.compilation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# op_host 层 OP_CHECK 未定义须用 OP_CHECK_IF

## 现象

在为 MoE 分组矩阵乘算子（Ascend950PR / arch35 / CANN 9.1.0，registry-invoke 手写 AscendC 工程）搭建主骨架后首次执行 `bash build.sh --soc=ascend950 -j8`，op_host 侧编译报错。报错原文（节选）：

```text
op_host/bench_no_harness_grouped_matmul_infershape.cpp:53:5:
error: ‘OP_CHECK’ was not declared in this scope; did you mean ‘OP_CHECK_IF’?
    53 |     OP_CHECK(xShape->GetDimNum() != 2,
        |     ^~~~~~~~
        |     OP_CHECK_IF

op_common/log/log.h:1078:5: error: expected primary-expression before ‘do’
 1078 |     do {                                                \
        |     ^~
note: in expansion of macro ‘OP_LOGE’
```

当时 infershape.cpp 和 tiling.cpp 里的形状/属性校验都是照 aclnn 层的写法用三段式 `OP_CHECK(cond, OP_LOGE(...), return ge::GRAPH_FAILED);` 写的。`grep` CANN 安装头文件确认 `op_common/log/log.h` 只有两个相关宏：

```text
1083:#define OP_CHECK_IF(condition, log, return_expr) \
1090:#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)
```

## 适用场景

- 工程形态：registry-invoke 自定义算子的 op_host 层（InferShape / TilingFunc）C++ 源码。
- API 路径：`op_common/log/log.h` 的校验日志宏（OP_CHECK_IF / OP_CHECK_NULL_WITH_CONTEXT）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN op_host 公共宏在 ascend950 工具链上行为一致。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：op_api（aclnn 适配层）不受影响——本轨迹同一份编译里 op_api 源文件的 `OP_CHECK(...)` 正常编译通过（aclnn/opdev 宏体系自带 OP_CHECK，其 OP_LOGE 首参是错误码而非 context）；其他 CANN 版本的宏清单未验证。

## 根因

op_host 层引用的 `op_common/log/log.h` 中不存在名为 `OP_CHECK` 的宏，只有 `OP_CHECK_IF(condition, log, return_expr)`；`OP_CHECK` 是 aclnn/opdev 层的宏名。跨层照抄宏名后，预处理器把 `OP_CHECK(...)` 当函数调用展开失败，连带 `OP_LOGE` 展开处的 `do {...}` 产生 "expected primary-expression before 'do'" 的连锁报错。

注意两层宏的**条件极性也相反**：`OP_CHECK_IF` 传"错误条件"（cond 为真即报错并 return），`OP_CHECK` 传"合法条件"（cond 为假才报错）——op_host 里把 `!=`/越界条件交给 `OP_CHECK_IF` 是正确用法，不要按 OP_CHECK 的语义再去取反。

## 解决配方

修复前（op_host infershape.cpp，照 aclnn 层写法）：

```cpp
    // x: [M, K]
    OP_CHECK(xShape->GetDimNum() != 2,
             OP_LOGE(context, "x must be 2D [M, K], got dimNum=%zu", xShape->GetDimNum()),
             return ge::GRAPH_FAILED);
    int64_t M = xShape->GetDim(0);
    int64_t K = xShape->GetDim(1);

    // weight: [E, K, N] 或 [E, N, K]
    OP_CHECK(wShape->GetDimNum() != 3,
             OP_LOGE(context, "weight must be 3D, got dimNum=%zu", wShape->GetDimNum()),
             return ge::GRAPH_FAILED);
```

修复后（改为 op_host 层的 OP_CHECK_IF，三段式形态相同、条件极性不变）：

```cpp
    // x: [M, K]
    OP_CHECK_IF(xShape->GetDimNum() != 2,
                OP_LOGE(context, "x must be 2D [M, K], got dimNum=%zu", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    int64_t M = xShape->GetDim(0);
    int64_t K = xShape->GetDim(1);

    // weight: [E, K, N] 或 [E, N, K]
    OP_CHECK_IF(wShape->GetDimNum() != 3,
                OP_LOGE(context, "weight must be 3D, got dimNum=%zu", wShape->GetDimNum()),
                return ge::GRAPH_FAILED);
```

tiling.cpp 中同类用法批量替换：

```bash
cd operators/<op>/op_host/arch35 && \
  sed -i 's/OP_CHECK(/OP_CHECK_IF(/g' <op>_tiling.cpp && \
  grep -c "OP_CHECK_IF(" <op>_tiling.cpp   # 12 处全部替换
```

## 设计决策

- 修复选择"宏名替换"而不是改写为手写 if：OP_CHECK_IF 与 OP_CHECK 三段式形态一致（cond / 日志 / 返回表达式），sed 批量替换即可，且 op_host 侧 OP_CHECK_IF 传错误条件的写法本来就是该层正确语义。
- 不把 aclnn/opdev 的头文件引进 op_host：两层日志宏体系不同（OP_LOGE 首参一个是 context、一个是错误码），跨层 include 会引入更多隐性差异。

## 预防

- 写 op_host 层校验前先 `grep -n "define OP_CHECK" $ASCEND_HOME_PATH/.../op_common/log/log.h` 确认宏名；不同层（op_host / op_api / kernel）的校验与日志宏体系不同，不要跨层复制。
- op_host 层空指针校验用 `OP_CHECK_NULL_WITH_CONTEXT(context, ptr)`；条件校验用 `OP_CHECK_IF(cond, OP_LOGE(context, ...), return ge::GRAPH_FAILED);`。
- 看到 `expected primary-expression before 'do'` 指向 log 头文件宏体时，先怀疑同文件里有未定义的宏名被当函数调用解析，而不是去改 OP_LOGE 写法。

## 验证信号

以下信号来自该算子主骨架的编译闭环：宏替换后重新 `bash build.sh --soc=ascend950`，OP_CHECK 相关报错消失，编译错误转向下一根因（include 路径），最终全量编译通过。

- 修复前：`error: ‘OP_CHECK’ was not declared in this scope; did you mean ‘OP_CHECK_IF’?` + `error: expected primary-expression before ‘do’`（来源平台：950；证据等级：tool_observed）。
- 修复后：`grep -c "OP_CHECK_IF(" tiling.cpp` 返回 12，后续编译输出中 OP_CHECK 报错不再出现（来源平台：950；证据等级：tool_observed）。
- 全量回归：op_host 宏修复连同后续 include/命名空间等修复落地后，`build.sh --soc=ascend950 -j8` 生成 `custom_opp_ubuntu_x86_64.run`（694K）与 3 个 kernel .o（覆盖 12 TilingKey）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.1.0 的 op_common/log/log.h 宏清单下验证；不同 CANN 版本宏名可能调整（未验证，推断可迁移）。
- aclnn/opdev 层的 OP_CHECK 用法（首参为错误码的 OP_LOGE）不受本卡影响，不要误改。
- op_host 侧 `OP_CHECK_IF` 传错误条件、aclnn 侧 `OP_CHECK` 传合法条件的极性差异，其语义反转陷阱见同域 [OP_CHECK 断言条件写反卡](161002_op_check_condition_inverted.md)。

**未决项**：

- aclnn/opdev 层 OP_CHECK 的定义位置未在轨迹中直接 grep 确认（仅由"同轮 op_api 文件编译通过"反推），需要时可到 `include/aclnn/opdev/` 下定位。
## 坏实践

- 从 aclnn 样例或直觉把 `OP_CHECK` 复制进 op_host 源码——两层宏体系不同名且极性相反。
- 只看 "expected primary-expression before 'do'" 这类连锁报错去改 OP_LOGE 写法——真正根因是宏名不存在。

## 误诊

无。编译器的 `did you mean ‘OP_CHECK_IF’?` 提示直接指向了正确宏名，本次未被误导。

## 增量（memset_s 返回值检查的显式 if 形态与错误路径运行验证）

独立复现（MlaProlog 类算子 op_host tiling/infershape，CANN 9.2.0 安装目录、version.info 9.1.0 基线，Ascend950PR 构建链）：同一根因的第三个报错形态与两类修复形态补齐。

**报错形态增量——逗号表达式内的 return**：除主卡的"宏未声明"与 log.h:1078 的 `expected primary-expression before 'do'` 外，旧写法 `OP_CHECK(cond, OP_LOGE(context, ...), return ge::GRAPH_FAILED)` 还会在逗号表达式内的 return 处报 `expected primary-expression before 'return'`（触发点 tiling.cpp:140）——三种形态同源于"宏不存在 + OP_LOGE 是 do-while 语句块"，修复后一并消失。

**修复形态增量——带函数返回值的检查改显式 if**：`memset_s`/`memcpy_s` 返回值检查不能塞进任何宏的参数表——`OP_LOGE(context, ...)` 作参数传给 `OP_CHECK_IF` 可以（其按语句展开），但 `return` 不能与它一起进手写逗号表达式。verbatim（修复前 → 修复后）：

```cpp
    // 修复前（旧逗号表达式风格 + memset_s 检查）
    MlaPrologTilingData* tiling = context->GetTilingData<MlaPrologTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK(memset_s(tiling, sizeof(MlaPrologTilingData), 0, sizeof(MlaPrologTilingData)) != EOK,
             OP_LOGE(context, "memset_s tiling data failed"), return ge::GRAPH_FAILED);

    // 修复后（memset_s 用显式 if）
    MlaPrologTilingData* tiling = context->GetTilingData<MlaPrologTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(MlaPrologTilingData), 0, sizeof(MlaPrologTilingData)) != EOK) {
        OP_LOGE(context, "memset_s tiling data failed");
        return ge::GRAPH_FAILED;
    }
```

普通条件检查则与主卡同法（`OP_CHECK(` → `OP_CHECK_IF(`，极性不变）：

```cpp
    // 修复前
    OP_CHECK(*coreNum <= 0, OP_LOGE(context, "coreNum is %ld", *coreNum), return ge::GRAPH_FAILED);
    // 修复后
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "coreNum is %ld", *coreNum), return ge::GRAPH_FAILED);
```

**运行验证增量（主卡仅编译级）**：修复后 UT `MlaPrologInfershape` 6/6 通过，其中 `infershape_invalid_2d_input`（2D 输入被拒绝）与 `infershape_invalid_nheads_zero`（n_heads=0 被拒绝）证明 `OP_CHECK_IF` 错误路径在运行期行为正确，不只是编译通过。

坏实践（复现增量）：

- 手写 `OP_CHECK` 宏定义来兼容旧样例——掩盖版本差异，且 `OP_LOGE` 的 do-while 展开仍会让逗号表达式编译失败。
- 把 `OP_LOGE(context, ...)` 与 `return` 一起写进手写逗号表达式传给任何宏。

验证信号（复现增量，MlaProlog op_host 编译修复会话，修复前后连续两轮完整构建对照）：

- 修复前一轮构建：`'OP_CHECK' was not declared in this scope; did you mean 'OP_CHECK_IF'?`、`expected primary-expression before 'return'`（tiling.cpp:140）、`expected primary-expression before 'do'`（log.h:1078 展开）三类错误同现（来源平台：950 构建链；证据等级：tool_observed）。
- 修复后重建：`cust_optiling`、`cust_op_proto` 目标编译通过，op_host 侧无残留错误（后续错误只剩 op_api 的 include 路径问题，属另一根因）（来源平台：950 构建链；证据等级：tool_observed）。
- infershape 守卫逻辑运行验证：UT `MlaPrologInfershape` 6/6 通过，含 `infershape_invalid_2d_input`（2D 输入被拒绝）与 `infershape_invalid_nheads_zero`（n_heads=0 被拒绝）证明 OP_CHECK_IF 错误路径行为正确（来源平台：x86 host UT（非 NPU）；证据等级：tool_observed）。

失效边界（复现增量）：

- 本增量证据来自 CANN 9.2.0 安装目录（version.info 9.1.0 基线）的 `op_common/log/log.h`，与主卡 9.1.0 证据互证宏形态跨小版本稳定；更大版本变化仍需按主卡 grep 探针重确认。