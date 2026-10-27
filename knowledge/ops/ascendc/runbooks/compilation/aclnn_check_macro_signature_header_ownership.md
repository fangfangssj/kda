---
type: "Runbook"
title: "aclnn 检查宏形态：OP_CHECK 三参与判空宏归属"
description: aclnn op_api 检查宏分属不同头文件族且签名不同：OP_CHECK 是三参宏（cond, OP_LOGE(...), return）；判空宏 OP_CHECK_NULL 只定义在 aclnn_kernels/common/op_error_check.h，自定义工程/L0 编译单元不可见；opdev 的 OP_CHECK_NOTNULL 只接受单参。按层选宏、判空手写 if+OP_LOGE+return。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# aclnn 检查宏形态：OP_CHECK 三参与判空宏归属

## 现象

四条独立轨迹、同一宏族的四种编译失败形态（Ascend950PR，CANN 9.1.0 与 9.2.0 均有命中）：

1. **L2 两参 OP_CHECK**（aclnn L2 层，CANN 9.2.0）：参数校验函数写了 `OP_CHECK(cond, return false)` 两参数断言形式，编译报 `error: 'OP_CHECK' was not declared in this scope`，报错点在宏调用行；同 include 集合的模板参照工程可编译，差异只在实参个数。
2. **L0 判空宏不可见 + 单参不匹配**（CANN 9.2.0）：只 include opdev 头的 op_api 编译单元里判空照两参宏习惯写 `OP_CHECK_NULL(x, ACLNN_ERR_PARAM_NULLPTR)`，先报 `'OP_CHECK_NULL' was not declared in this scope; did you mean 'OP_CHECK_NOTNULL'?`；按编译器建议改成 `OP_CHECK_NOTNULL(x, ACLNN_ERR_PARAM_NULLPTR)` 后又报 8 处 `error: macro "OP_CHECK_NOTNULL" passed 2 arguments, but takes just 1`——同一文件连犯两版宏签名。
3. **自定义工程无内部头**（CANN 9.1.0）：自定义算子 op_api L0 层照内部 aclnn kernel 仓写法用 `OP_CHECK_NULL(x, return nullptr)`，报 `'OP_CHECK_NULL' was not declared` + `expected primary-expression before 'return'`——该宏来自内部仓公共头，不属于 CANN 安装目录对外发布的 op_api 头集合。
4. **宏头文件归属 + 分层职责**（CANN 9.1.0）：L0 API 文件只含 opdev 系列头却在入口写 `OP_CHECK_NULL(...)`，L2 API 文件包含 `aclnn_kernels/common/op_error_check.h` 却把 `OP_CHECK` 写成两参——两类"宏未声明"同轮出现。

## 适用场景

- 工程形态：自定义算子仓 op_api 层（L0 `op_api/<op>.cpp` 与 L2 `op_api/aclnn_<op>.cpp`）参数检查代码编写；从内部仓/旧样例抄写校验写法时。
- API 路径：`OP_CHECK` 定义于 `aclnn/opdev/op_log.h`（三参 `(cond, log_func, return_expr)`）；`OP_CHECK_NULL(param, retExpr)` 及 OP_CHECK_DTYPE_* 一族定义于 `aclnn_kernels/common/op_error_check.h`；opdev 头集合里的 `OP_CHECK_NOTNULL` 为单参宏；`OP_LOGE(errCode, fmt, ...)`（opdev 族，首参为错误码）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.1.0 与 9.2.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：platform_wide——宏定义与分层约定由 CANN 头文件/模板决定，与算子无关。
- 路线优先级：recommended_fastpath（按层选宏 + 三段式 OP_CHECK + 手写判空兜底）。
- 明确不适用 / 未验证场景：op_host 侧 `op_common/log/log.h` 宏族（OP_LOGE 首参为 context、OP_CHECK 已移除只剩 OP_CHECK_IF）是另一套形态，不适用本卡；include 了 `aclnn_kernels/common/op_error_check.h` 的编译单元可能有两参判空宏可用（本批未验证其存在与签名）；其他 CANN 版本宏集合可能变化，用前 grep。

## 根因

aclnn 检查宏分属两个头文件族且签名不同，宏不可重载、实参个数或可见性不匹配即在预处理/编译期失败：

- `OP_CHECK` 定义在 `aclnn/opdev/op_log.h`，是**三参数**宏 `(cond, log_func, return_expr)`——中参必须是日志动作（`OP_LOGE(<错误码>, <格式串>, ...)`），两参简写不合宏定义；需要"条件不满足仅返回"时用同文件的 `CHECK_COND(cond, ret, fmt, ...)`。
- `OP_CHECK_NULL`（及 OP_CHECK_DTYPE_NOT_SUPPORT / OP_CHECK_BROADCAST 等一族）只定义在 `aclnn_kernels/common/op_error_check.h`；该头属内部仓公共头，自定义算子工程与只含 opdev 头的 L0 编译单元不默认可见——宏未定义即展开失败，连带 "expected primary-expression" 语法错。
- opdev 头集合里可见的 `OP_CHECK_NOTNULL` 只接受 1 个参数（判空 + 默认失败动作），不接收错误码。
- 按 aclnn 分层约定（add_example 模板同款）：必选参数的 null 检查在 **L2 API** 完成（L2 已包含 op_error_check.h，`CheckNotNull` 用 `OP_CHECK_NULL` 一族宏统一检查），L0 API 假设必选参数非空、只做 InferShape/IsAiCoreSupport 等检查——L0 里写 `OP_CHECK_NULL` 本身就放错了层。

## 解决配方

**形态一：L2 的 OP_CHECK 一律三段式**（两参 → 三参，verbatim）：

```cpp
    // 修复前：两参写法，宏不匹配定义
    OP_CHECK(query->GetViewShape().GetDimNum() == 4, return false);

    // 修复后：三参数形式（错误码 + 日志 + 返回值一次到位）
    OP_CHECK(query->GetViewShape().GetDimNum() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query dim must be 4, actual=%zu.",
                     query->GetViewShape().GetDimNum()),
             return false);
```

**形态二：L0 判空改手写 if + OP_LOGE(errCode) + return**（verbatim）：

```cpp
    // 修复前（两版错误写法都真实出现过）：
    OP_CHECK_NULL(x, ACLNN_ERR_PARAM_NULLPTR);          // 宏不可见
    OP_CHECK_NOTNULL(x, ACLNN_ERR_PARAM_NULLPTR);       // 单参宏传 2 参

    // 修复后（合并判空 + opdev 族 OP_LOGE + 显式 return）：
    if (x == nullptr || y == nullptr || expertIdx == nullptr || rowIdx == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "input/output tensor is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
```

**形态三：L0 移除判空、按分层交给 L2**（L0 入口修复前后，verbatim）：

```cpp
    // 修复前：null 检查放错层且宏头文件缺失
    OP_CHECK_NULL(expandedPermutedRows, return nullptr);
    OP_CHECK_NULL(expandedSrcToDstRow, return nullptr);
    OP_CHECK_NULL(executor, return nullptr);

    Shape outShape;
    OP_CHECK(BenchNoHarnessMoeFinalizeRoutingInferShape(...),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);
```

```cpp
    // 修复后：null 检查在 L2 API 完成，L0 API 假设必选参数非空
    Shape outShape;
    OP_CHECK(BenchNoHarnessMoeFinalizeRoutingInferShape(...),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, expandedPermutedRows->GetDataType());

    OP_CHECK(IsAiCoreSupport(expandedPermutedRows),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);
```

判别方法：遇 `'XXX' was not declared in this scope` 且 XXX 疑似宏时，先查实参个数与宏可见性，再怀疑 include；用宏前 `grep -rn "#define <MACRO>" $ASCEND_HOME_PATH/<arch>-linux/include/` 确认定义位置、参数个数与首参语义。

## 设计决策

- L0 选择移除判空而不是补 include 内部仓头：自定义算子包只应依赖 CANN 安装目录公共头，引入内部仓头破坏可移植性；且模板分层约定判空归 L2（L0 补头做防御式检查编译也能过，属团队风格取舍）。
- 多指针合并为一条 if（`||` 链）：同一失败路径（PARAM_NULLPTR + return）折叠成一条，绕开宏签名差异并减少重复日志。
- 不自造兼容宏：改调用侧写法对齐宏语义即可，自造宏会让后续维护者再踩一次签名歧义。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写 aclnn L0/L2 前对齐模板包含集：L2 = `aclnn_kernels/contiguous.h` + `aclnn_kernels/common/op_error_check.h` + opdev 系列；L0 = opdev 系列（不含 op_error_check.h）。
- 复制其它工程/内部仓的校验写法时，连同其 include 列表一起核对。
- 跟着编译器 `did you mean` 建议盲改宏名而不核对签名，会再错一轮（OP_CHECK_NULL→OP_CHECK_NOTNULL 即实例）。

## 验证信号

来源说明：四条独立轨迹的 op_api 编译失败—修复闭环（CANN 9.1.0 / 9.2.0，Ascend950PR 构建链），各轨迹修复后均以完整构建推进到下一编译阶段验证。

- 轨迹一（L2 两参）：修复前三参错误清单中含 `'OP_CHECK' was not declared in this scope`；改为三段式后 cust_opapi 该类错误清零、后续全量构建 `Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- 轨迹二（判空两连错）：`'OP_CHECK_NULL' was not declared` 与 `macro "OP_CHECK_NOTNULL" passed 2 arguments, but takes just 1`（8 处）均为完整编译输出；改手写 if + OP_LOGE 后该类错误全部消失、op_api 编译通过（来源平台：950；证据等级：tool_observed）。
- 轨迹三（自定义工程）：修复前 `'OP_CHECK_NULL' was not declared in this scope; did you mean 'OP_CHECK_NOTNULL'?`（4 处）+ `expected primary-expression before 'return'`；手写判空后 cust_opapi 编译通过、构建推进到 kernel 编译阶段（来源平台：950；证据等级：tool_observed）。
- 轨迹四（分层）：宏定义锚点 `op_error_check.h:103 #define OP_CHECK_NULL(param, retExpr)`、`op_log.h:227 #define OP_CHECK(cond, log_func, return_expr)`；L0 移除判空 + L2 改三参后 op_api 库编译通过、全量构建退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 宏所属头文件与行号基于 CANN 9.1.0/9.2.0 安装布局；版本升级后行号/归属可能变化，用 grep 重新定位（未验证，机制推断可迁移）。
- "L0 不做空检查"是 aclnn 模板约定而非编译器强制；选择让 L0 也做防御式判空的工程，编译同样可过。
- `aclnn_kernels/common/op_error_check.h` 是否随 CANN 安装目录对外发布、其两参判空宏的可用性未在全部版本逐一验证；include 该头的编译单元（L2）内 `OP_CHECK_NULL` 可用（轨迹四的 L2 实证）。

**未决项**：

- 无（四条轨迹的宏形态坑均闭环到编译通过；op_host 侧宏族形态另卡记录）。
## 坏实践

- 凭记忆把内部仓宏（OP_CHECK_NULL 等）直接写进自定义算子工程或 L0 文件——编译期才爆，报错行号在宏调用处，不易第一眼看出是"宏不存在"。
- 把 `OP_CHECK` 当两参断言用（缺 log_func 中参），或为图省事退化成裸 `if (...) return false;` 丢失错误码日志（验收阶段返回码校验 UT 会暴露）。
- 逐个修宏调用点而不成批统一三段式——同类调用点会接连报错，每错一轮跑一次完整构建。

## 误诊

- 轨迹一初见报错曾怀疑 include 缺失（op_error_check.h 未被包含）；模板参照工程同 include 集合可编译，对照后确认差异只在实参个数。
- 轨迹二跟着 `did you mean` 从 OP_CHECK_NULL 改到 OP_CHECK_NOTNULL，未先看宏定义，又错一轮。
