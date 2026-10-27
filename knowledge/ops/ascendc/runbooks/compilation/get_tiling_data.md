---
type: "Runbook"
title: "GET_TILING_DATA 首参传类型名致变量遮蔽类型"
description: "GET_TILING_DATA(<类型名>, tiling) 把类型名作宏首参，宏展开声明同名变量遮蔽类型，致 InitTilingData<HistogramTilingData> 模板实参 T 无法匹配 InitTilingData(const __gm__ uint8_t*, T*) 签名，报 candidate template ignored: invalid explicitly-specified argument for template parameter 'T'；三参形式 GET_TILING_DATA(T, data, tiling) 同样报错（'does not refer to a value' + 'use of undeclared identifier'）。GET_TILING_DATA 是两参宏 (tiling_data, tiling)，带类型名的三参形式是 GET_TILING_DATA_WITH_STRUCT，两者都需先 REGISTER_TILING_DEFAULT；判别依据为编译错 expanded from macro 行 &tiling_data 取首参地址，证明首参为变量名。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T02:21:59Z"
platforms: ["950", "a2"]
---

# GET_TILING_DATA 首参传类型名致变量遮蔽类型

## 现象

**形态一（历史证据：增量前库卡会话，Ascend 910B，CANN 8.5.0-beta.1）**：在 Ascend 910B 上开发 Histogram 自定义算子，上一处 tiling 头冲突编译错修复后，kernel 入口改用 `GET_TILING_DATA(HistogramTilingData, tiling)` 从 GM 拷贝 tiling 数据。再次 `make` 构建，kernel 代码生成阶段报（verbatim 编译错，取自会话内 `make` 输出）：

```text
histogram.cpp:153:7: error: no matching function for call to 'InitTilingData'
      GET_TILING_DATA(HistogramTilingData, tiling);
      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.../kernel_meta_Histogram_..._tiling_data.h:169:5: note: expanded from macro 'GET_TILING_DATA'
    InitTilingData<HistogramTilingData>(tiling_arg, &tiling_data);
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.../kernel_meta_Histogram_..._tiling_data.h:30:59: note: candidate template ignored: invalid explicitly-specified argument for template parameter 'T'
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
                                                          ^
```

随后连锁 `ld.lld: error: cannot open .../kernel_meta_Histogram_..._2.o` 与 `RuntimeError: Kernel Compilation Error: OpType Histogram`。

**形态二（复现增量Ascend950，CANN 9.1.0）**：独立复现——为排查 tiling 获取方式，把原本可用的 `GET_TILING_DATA_WITH_STRUCT` 写法改成"看起来同构"的三参 `GET_TILING_DATA`（`bash build.sh --soc=ascend950 -j8`）：

```cpp
    AscendC::TPipe tPipe;
    GET_TILING_DATA(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);
    (void)workspace;
```

编译报错原文：

```
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:11:21: error: 'BenchNoHarnessConv3dBackpropFilterTilingData' does not refer to a value
op_kernel/bench_no_harness_conv_3d_backprop_filter_arch35.cpp:11:67: error: use of undeclared identifier 'tiling'
```

**根因（三参形态）**：`GET_TILING_DATA` 是**两参**宏（形参依次为 tiling_data 目标变量名、tiling 入参指针名），宏体按位置把第一个实参拼接进 `uint8_t __ascendc_arr_##tiling_data[...]` 等声明——把 tiling struct **类型名**放在第一参，会被展开成 `__ascendc_arr_BenchNoHarnessConv3dBackpropFilterTilingData`（"类型名不是值"，即 "does not refer to a value"），第三个实参溢出宏形参个数产生 "use of undeclared identifier 'tiling'"。带类型名的形式是另一个宏 `GET_TILING_DATA_WITH_STRUCT`。

## 适用场景

- **工程形态**：自定义算子工程（ops-registry-invoke），kernel 侧用 `GET_TILING_DATA` 宏从 GM 拷贝 tiling 数据到本地变量（CANN 8.5.0-beta.1 自动生成 `kernel_meta_*/..._tiling_data.h` 提供 `GET_TILING_DATA`/`InitTilingData` 宏与 `HistogramTilingData` 类）。
- **API 路径**：`GET_TILING_DATA(var_name, tiling_arg)` → 展开为 `HistogramTilingData var_name; InitTilingData<HistogramTilingData>(tiling_arg, &var_name);`（首参为变量名，非类型名）。
- **验证平台与版本**：主证据 Ascend950 + CANN 9.1.0（与 frontmatter `platforms: ["950"]` 一致）；历史证据 Ascend 910B（AIV only，属 a2 系列）+ CANN 8.5.0-beta.1（增量前库卡会话，当时库卡未定位注册平台）。
- **适用粒度**：`platform_wide`——任何用 `GET_TILING_DATA` 宏的 kernel 均可能误传类型名作首参，不依赖具体算子公式。
- **路线优先级**：recommended_fastpath（两参/三参宏按 REGISTER_TILING_DEFAULT + 正确形参表使用）。
- **不适用或未验证场景**：若宏签名变更（如显式分类型/变量名两参）则本卡描述的遮蔽不再适用，迁移前先读生成的 `tiling_data.h` 确认宏展开（原"CANN 9.x 宏签名是否相同未验证"一项已由 950 复现增量实测闭合，见「失效边界」）。

## 判别方法

**触发**：两种形态的触发经过与 verbatim 编译错见「现象」——共同入口是宏实参表用错（形态一把类型名当宏首参，形态二给两参宏套三参）。

**探针**（可执行检查）：

1. 读编译错 `expanded from macro` 指向的自动生成头 `..._tiling_data.h` 第 169 行（`GET_TILING_DATA` 宏展开模板），确认展开形如 `InitTilingData<HistogramTilingData>(tiling_arg, &tiling_data);`——`&tiling_data` 取的是宏首参的地址，证明首参被用作**变量名**而非类型。
2. 对照 built-in kernel（如 `histogram_v2`）的 `GET_TILING_DATA` 调用，确认正解首参为小写变量名（如 `tilingData`），与类型名（大写首字母）区分。
3. 在报错 kernel 行把宏首参从类型名 `HistogramTilingData` 改为小写变量名 `tilingData`，重跑 `make`，观察 `no matching function for call to 'InitTilingData'` 与 `candidate template ignored` 是否消失。

**观测**（tool_observed）：

- 编译错 `expanded from macro` 行为 `InitTilingData<HistogramTilingData>(tiling_arg, &tiling_data);`——`&tiling_data` 取宏首参地址，首参=变量名。
- `InitTilingData` 模板签名为 `void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)`，模板参数 `T` 须匹配第二参指针类型；显式实参 `InitTilingData<HistogramTilingData>` 被编译器拒绝：`candidate template ignored: invalid explicitly-specified argument for template parameter 'T'`。

**判读**（根因解读）：`GET_TILING_DATA` 宏首参是变量名。首参传类型名 `HistogramTilingData` 时，宏展开声明变量 `HistogramTilingData HistogramTilingData;`——**变量名遮蔽类型名**；于是 `InitTilingData<HistogramTilingData>` 中的 `HistogramTilingData` 解析为变量而非类型，模板参数 `T` 无法匹配 `InitTilingData(const __gm__ uint8_t*, T*)` 的签名，编译器报 `candidate template ignored: invalid explicitly-specified argument for template parameter 'T'`。关键判据二联：宏展开行 `&<首参>` 取地址证明首参为变量名；显式模板实参与变量名同名时被解析为变量致模板实参无效。

## 解决配方

修复（两参形式 + REGISTER_TILING_DEFAULT）：

```cpp
    AscendC::TPipe tPipe;
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv3dBackpropFilterTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
```

等价的带类型三参形式（本算子最终采用的写法，同样编译通过）：

```cpp
    AscendC::TPipe tPipe;
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv3dBackpropFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);
    (void)workspace;
```

配套细节：两参 `GET_TILING_DATA` 展开产物与 `GET_TILING_DATA_WITH_STRUCT` 的**取值形态不同**（一个得到指针、一个得到结构体值），向 kernel 类 `Init(..., const T* tilingData, ...)` 传参时前者直接传 `tilingData`、后者传 `&tilingData`——混用两种宏而不调整取地址会连锁编译错。

预防要点：

- 记住宏签名：`GET_TILING_DATA(data, ptr)` 两参；`GET_TILING_DATA_WITH_STRUCT(T, data, ptr)` 三参；都用 `REGISTER_TILING_DEFAULT(T)` 注册。
- 看到 "does not refer to a value" 指向宏首参时，先核对宏的形参个数而不是猜类型注册问题。

## 验证信号

主证据来源：BenchNoHarnessConv3dBackpropFilter 算子联调会话（Ascend950，CANN 9.1.0，kernel 入口写法的受控切换实验）。

- 三参 `GET_TILING_DATA(T, tilingData, tiling)` 构建失败，报 `'BenchNoHarnessConv3dBackpropFilterTilingData' does not refer to a value` 与 `use of undeclared identifier 'tiling'`（来源平台：950；证据等级：tool_observed）。
- 改回 `REGISTER_TILING_DEFAULT(T)` + 两参 `GET_TILING_DATA(tilingData, tiling)` 后同环境重新构建 `Build completed successfully!` 并完成算子包安装（来源平台：950；证据等级：tool_observed）。

历史证据（增量前库卡会话，非本卡增量；Ascend 910B，属 a2 系列，CANN 8.5.0-beta.1，当时库卡未定位注册平台）：

- 按判读把宏首参从类型名 `HistogramTilingData` 改为小写变量名 `tilingData` 后，下一次 judge 运行返回 `ret=561002`（aclnn 运行时错，aclnnHistogramGetWorkspaceSize 层）而**不再是 kernel 编译错**——`no matching function for call to 'InitTilingData'` 与 `candidate template ignored` 消失，kernel 已成功编译生成二进制（来源平台：a2；证据等级：tool_observed）

## 失效边界

- CANN 9.1.0 的自动生成宏实测与 8.5.0-beta.1 同为两参形式（主卡适用场景中"CANN 9.x 未验证"一项由此闭合）；其他版本宏签名若有变化需以当版 `generate_tiling_code.py` 生成体为准（未验证，推断可迁移）。
- "两参/三参产物取值形态不同（指针 vs 值）"由本轨迹连锁编译错与传参修正推证（`op.Init(..., tilingData, ...)` 与 `op.Init(..., &tilingData, ...)` 两次切换），未逐行展开宏体验证。
- 历史证据（910B/8.5.0-beta.1）仅覆盖两参类型名遮蔽形态；三参误用形态的证据来自 Ascend950/9.1.0。

**未决项**：

- 无（根因与修复为受控对照验证）。
## 坏实践

- 凭记忆给 `GET_TILING_DATA` 套三参（类型名打头）——两个 GET 宏形参列表不同，报错信息完全不会提示"宏用错了"。
