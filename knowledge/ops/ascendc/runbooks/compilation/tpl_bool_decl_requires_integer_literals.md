---
type: "Runbook"
title: TPL_BOOL_DECL 取值须整型 0/1 非 false/true
description: ASCENDC_TPL_BOOL_DECL/BOOL_SEL 的取值列表必须显式写整型字面量 0/1：写 false/true 时 template_tiling.py 解析不出取值集合，报 "values of ASCENDC_TPL_BOOL_DECL <name> is empty!"；完全不传值列表时 {__VA_ARGS__} 展开为空 vector 同样报错；多条目还会先因签名退化为 "ASCENDC_TPL_SELECT has duplicated definitions!"。
tags: [task.debugging, topic.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# TPL_BOOL_DECL 取值须整型 0/1 非 false/true

## 现象

两个独立算子（BenchNoHarnessRoiAlign / BenchNoHarnessMoeGatingTopKSoftmax，均为 Ascend950PR，CANN 9.1.0，registry-invoke 工程）给 TilingKey 增加布尔模板参数时，以三种错误形态踩中同一约束：

**形态一（不传值列表）**。`op_kernel/arch35/<op>_tiling_key.h` 按"布尔模板参数"直觉写 `ASCENDC_TPL_BOOL_DECL(HAS_FINISHED)`（下方 `ASCENDC_TPL_SEL` 里用 `ASCENDC_TPL_BOOL_SEL(HAS_FINISHED, 0)` / `(HAS_FINISHED, 1)` 列了两个取值），构建推进到 kernel 编译阶段时 Python 侧工具链直接抛异常：

```text
[ERROR] TBE(...): [op_compilation.py:311][__single_op_compile]
  Exception occured, values of ASCENDC_TPL_BOOL_DECL HAS_FINISHED is empty!
[ERROR] TBE(...): [opc_common.py:227][_log_error] Traceback (most recent call last):
  File ".../asc_op_compile_base/asc_op_compiler/template_tiling.py", line 397, in extract_template_tiling_info
```

且 `build/op_kernel/ascendc_kernels/binary/ascend950/<op>/BenchNoHarnessMoeGatingTopKSoftmax_<hash>.json not generated!`。

**形态二（写 false/true）**。写成 `ASCENDC_TPL_BOOL_DECL(D_ALIGNED, false, true)`、SEL 里 `ASCENDC_TPL_BOOL_SEL(D_ALIGNED, false)` 组合后，报错变为：

```text
[ERROR] TBE(124295,python3):... [template_tiling.py:397][extract_template_tiling_info] ...
    Exception occured, values of ASCENDC_TPL_BOOL_DECL D_ALIGNED is empty!
```

**形态三（4 条目混排先报重复）**。最初写成 4 条目 `ASCENDC_TPL_SEL`（dtype 相同、布尔值组合不同）时，构建直接失败：

```
[ERROR] TBE(123778,python3):... [template_tiling.py:739][decode_tiling]
    raise RuntimeError("ASCENDC_TPL_SELECT has duplicated definitions!")
RuntimeError: ASCENDC_TPL_SELECT has duplicated definitions!
```

把 DECL/SEL 里的 `false/true` 全部换成 `0/1`（并给 DECL 显式补值列表）后构建成功、正常生成 kernel。

## 适用场景

- 工程形态：AscendC 自定义算子的多 TilingKey/多模板实例开发（`tiling_key.h` 声明 + host tiling 侧 `ASCENDC_TPL_SEL_PARAM` 选择）。
- API 路径：`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_BOOL_DECL` / `ASCENDC_TPL_BOOL_SEL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_ARGS_SEL`（`<arch>-linux/asc/include/tiling/template_argument.h` 宏定义；Python 侧消费方 `asc_op_compile_base/asc_op_compiler/template_tiling.py`）。
- 验证过的平台与 CANN 版本：Ascend950PR（soc=ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有使用 ASCENDC_TPL_BOOL_DECL 的算子工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`ASCENDC_TPL_UINT_DECL` 等其他类型声明宏从 add_example 样例看同样要求显式值，未实测；其他 CANN 版本未验证。

## 根因

布尔模板参数的 DECL/SEL 取值必须写成**整型字面量 `0`/`1`**，且必须显式给出：

- `ASCENDC_TPL_BOOL_DECL(x, ...)` 展开为 `ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "DECL"}`——不传剩余参数时 `{__VA_ARGS__}` 是**空 vector**；kernel 编译期 `template_tiling.py` 的 `extract_template_tiling_info` 遍历 DECL 结构取值列表，发现 values 为空即抛 `values of ASCENDC_TPL_BOOL_DECL <name> is empty!`。SEL 侧写了取值不弥补 DECL 侧的空列表。
- 写 `false`/`true` 时模板参数解析器解析不出取值集合，同样报 "is empty!"。
- 4 条目混排版本因为布尔值解析为空，多个条目退化成相同签名，先触发 `ASCENDC_TPL_SELECT has duplicated definitions!`（`decode_tiling` 的去重检查）。

## 解决配方

给 `ASCENDC_TPL_BOOL_DECL` 显式补整型值列表 `(0, 1)`，SEL 侧同步用 `0`/`1`。

修复前（形态二，false/true 写法）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessRoiAlign,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(D_ALIGNED, false, true),
    ASCENDC_TPL_BOOL_DECL(D_SR_MODE, false, true)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, false),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, false)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, false),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, true)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, true),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, false)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, true),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, true)
    )
);
```

修复前（形态一，不带值列表）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessMoeGatingTopKSoftmax,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16,
                              ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(HAS_FINISHED)
);
```

修复后（整型 0/1，两例同法）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessMoeGatingTopKSoftmax,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16,
                              ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(HAS_FINISHED, 0, 1)
);
```

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessRoiAlign,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(D_ALIGNED, 0, 1),
    ASCENDC_TPL_BOOL_DECL(D_SR_MODE, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, 0),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, 0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, 0),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, 1),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, 0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(D_ALIGNED, 1),
        ASCENDC_TPL_BOOL_SEL(D_SR_MODE, 1)
    )
);
```

（`ASCENDC_TPL_SEL` 里 `ASCENDC_TPL_BOOL_SEL(HAS_FINISHED, 0)` / `(HAS_FINISHED, 1)` 取值部分保持不变。）

## 设计决策

- 直接补 `(0, 1)` 而不是改成 UINT_DECL：BOOL 语义与 kernel 模板参数 `template <typename D_T_X, bool HAS_FINISHED>` 对齐，改 UINT 会牵动 SEL 与 kernel 签名。
- 判别顺序：报 `values of ASCENDC_TPL_<TYPE>_DECL <参数名> is empty!` → 直接回 tiling_key.h 给对应 DECL 补整型值，不必怀疑 SEL 或 host 传参；报 `ASCENDC_TPL_SELECT has duplicated definitions!` → 先怀疑取值解析失败导致条目退化，再怀疑条目真重复。
- 形态二所在的"布尔属性进 TilingKey"方案最终被整体回退（dtype-only TilingKey）：0/1 修复后构建成功，但 4 个 SEL 条目只生成了 1 个 kernel 二进制，多实例目标未达成，且逐用例进程方案（见同批缓存卡）已解决根本问题。本卡只保留编译层面的可复用结论。

## 预防

- 写 `ASCENDC_TPL_BOOL_DECL` 时模板固定带值列表 `(0, 1)`；参照同模板 add_example 的 `ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)`——这一族 DECL 宏都要求显式枚举取值。
- TilingKey 模板参数的取值一律用整型/枚举字面量（BOOL 用 0/1），不用 C++ 布尔关键字——它由 Python 侧解析，取值语法是整型。
- 报 "duplicated definitions" 时先拆成标准 ARGS_DECL/BOOL_DECL 结构暴露真正的 "empty values" 报错再定位。

## 验证信号

来源说明：两个独立算子的 tiling_key.h 修复前后连续构建实测（每次改动全量重建）。

- 形态一修复前：kernel 编译阶段 `Exception occured, values of ASCENDC_TPL_BOOL_DECL HAS_FINISHED is empty!`（template_tiling.py:397 抛出），kernel json not generated（来源平台：950；证据等级：tool_observed）
- 形态一修复后：模板参数提取通过，构建推进到 kernel 源码编译并报出 `op_kernel/arch35/<op>.h` 里的 `std::max` 错误（另一根因），本异常消失（来源平台：950；证据等级：tool_observed）
- 形态二修复前：构建失败 `values of ASCENDC_TPL_BOOL_DECL D_ALIGNED is empty!`（此前 4 条目混排版本为 `ASCENDC_TPL_SELECT has duplicated definitions!`）（来源平台：950；证据等级：tool_observed）
- 形态二修复后：`bash build.sh --soc=ascend950 -j8` 构建成功，kernel 包正常生成（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅验证 CANN 9.1.0 的 `template_tiling.py` 解析行为；其他版本是否接受 false/true 或默认填充布尔值列表未验证（推断同样要求 0/1，未验证）。
- DECL 值列表与 SEL 取值需一致（DECL 列了 SEL 未列的值不会被路由）；本例两边都是 {0,1}，一致性未做独立验证。
- 本卡不涉及"多条目 SEL 为何只生成 1 个二进制"——该现象未解决即被回退，机制未知。

**未决项**：

- "4 条目 SEL 只生成 1 个 kernel 二进制"的机制未查（可能同 dtype 条目被合并）；若后续确需属性级多实例，要重新验证该路径。
## 坏实践

- 认为"BOOL 只有 0/1，编译器应该知道"而省略值列表——template_tiling.py 不做默认填充，空列表直接异常终止。
- 把 TilingKey 宏当 C++ 宏对待、随手写 `true/false` 字面量。

## 误诊

- 第一次看到 "duplicated definitions" 时以为是条目逻辑重复，重排条目无效；拆成标准 ARGS_DECL/BOOL_DECL 结构后才暴露真正的 "empty values" 报错，定位到 false/true 写法。
