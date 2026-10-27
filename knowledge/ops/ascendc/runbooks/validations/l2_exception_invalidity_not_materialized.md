---
type: "Runbook"
title: "L2 异常用例非法性须物化：Mock 过 Real 返 0"
description: L2 参数校验用例的非法性若只写在 CSV 的 note 字段，Real 模式用合法 dtype/shape 构造 tensor 调真算子必然返回 0 而非 161002，Mock 再靠 note 字符串 fallback 蒙混通过——两条执行器互相掩护。修复是按用例意图注入真实非法属性（dtype/reduce/shape），并修正 scatter dim 轴形状约束的期望语义。
tags: [task.testing, topic.compute.indexing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# L2 异常用例非法性须物化：Mock 过 Real 返 0

## 现象

`bench_no_harness_scatter` 算子（Ascend950PR 实测，CANN 9.1.0）的 L2 异常/参数校验用例组（CSV 定义 10 条，预期 `aclnnBenchNoHarnessScatterGetWorkspaceSize` 返回 161002/161001）出现一个方向性分歧：

- Mock 模式（CPU golden，不真调算子）10/10 全过；
- Real 模式（真调 GetWorkspaceSize）6 条失败，全部是：

```
[FAIL] GetWorkspaceSize 返回 0 (expect 161002)
```

失败的是 L2_001（indices dtype 不支持 uint8）、L2_002（updates dtype ≠ data dtype）、L2_003（data dtype 不支持 uint8）、L2_004（reduce 字符串非法）、L2_006（indices.shape ≠ updates.shape）、L2_007（indices 维度大小 > data 维度大小）。

读测试脚手架发现两个脚手架层缺陷：

1. `RunRealExceptionTest` 用 CSV 里的 dtype/shape 直接构造 tensor——而 CSV 为了让用例"可构造"，这些字段填的是**合法值**（如 L2_001 的 indices_dtype 列是 int32），非法性只写在 note 文本里（"indices dtype unsupported (uint8)"）。于是 Real 模式构造出来的是一套完全合法的输入，算子当然返回 0；
2. Mock 模式的 `RunMockExceptionTest` 走 `ValidateCase` 检查 CSV 字段（合法→通过），失败后还有一个按 note 字符串匹配 "dtype"/"shape" 就判 PASS 的 fallback——两段逻辑互相掩护，谁都没真正构造非法输入。

另外 L2_007 的用例定义本身违反 scatter 语义：CSV 写 data=[8,10]、indices=[12,6]、dim=0，期望"indices.shape[0]=12 > data.shape[0]=8"报 161002——但 scatter 里 **indices 沿 dim 轴的形状不受 data 限制**（只要求索引值 ∈ [0, data.shape[dim])），需要 ≤ data 的是**非 dim 轴**。Mock 的 `ValidateCase` 恰好也写了"逐维比较所有轴"的错误检查（含 dim 轴），所以 Mock 又"通过"了。

## 适用场景

- 工程形态：aclnn 算子的 L2/异常/参数校验用例开发——Mock（CPU golden）与 Real（真调 aclnnGetWorkspaceSize）双模式脚手架。
- API 路径：`aclnn<Op>GetWorkspaceSize` 的参数校验错误码（161002=ACLNN_ERR_PARAM_INVALID、161001 空指针类）；scatter 类算子的 shape 约束语义（非 dim 轴 ≤ data、dim 轴不限、索引值域 [0, data.shape[dim])）。
- 验证过的平台与 CANN 版本：Ascend950PR 实测，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——L2/异常用例设计与 Mock/Real 双模式脚手架的通用陷阱；scatter/gather/scatter_reduce 族的形状约束语义。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非索引类算子的 shape 校验语义（各自按算子定义）；框架级报错路径（报错发生在 aclnn 层之前的情况）。

## 根因

L2 异常用例的"非法性"停留在 CSV 的 note 字段（人类可读描述），两个执行器都没有把它物化成真实输入：Real 用 CSV 的合法 dtype/shape 构造 tensor 调真算子（必然返回 0），Mock 用同一批字段做静态校验再加 note 字符串 fallback（必然"识别为构造正确"）。同时 L2_007 的期望本身把 scatter 的 dim 轴形状约束写反了。

## 解决配方

三处修复（这是收紧测试强度而非放松——构造真实非法输入，让校验路径真正被走到）：

1. 重写 `RunRealExceptionTest`：默认构造合法输入，按 case 注入对应非法属性（verbatim 节选）：

```cpp
// 其余 L2: 根据用例意图构造非法输入, 调用 GetWorkspaceSize
// 默认构造合法输入, 然后按 case_id 注入对应的非法属性
DType data_dt = tc.data_dtype;
DType idx_dt  = tc.indices_dtype;
DType upd_dt  = tc.data_dtype;               // updates dtype 默认 = data
DType y_dt    = tc.data_dtype;               // y dtype 默认 = data
auto idx_shape  = tc.indices_shape;           // indices shape
auto upd_shape  = tc.indices_shape;           // updates shape 默认 = indices shape
int64_t reduce_mode = ParseReduceMode(tc.reduce);

// L2_001: indices dtype unsupported (uint8)
if (tc.case_id == "L2_001") {
    idx_dt = DType::UINT8;
}
// L2_002: updates dtype != data dtype (data=float32, updates=float16)
if (tc.case_id == "L2_002") {
    upd_dt = DType::FLOAT16;
}
// L2_003: data dtype unsupported (uint8) → data/updates/y 全部 uint8
if (tc.case_id == "L2_003") {
    data_dt = DType::UINT8;
    upd_dt  = DType::UINT8;
    y_dt    = DType::UINT8;
}
// L2_004: reduce string invalid → 传入越界整数 (合法范围 0~4)
if (tc.case_id == "L2_004") {
    reduce_mode = 5;
}
// L2_006: indices.shape != updates.shape (updates 用不同 shape)
if (tc.case_id == "L2_006") {
    upd_shape = {5, 7};
}
// L2_008: y dtype != data dtype
if (tc.note.find("y dtype") != std::string::npos) {
    y_dt = DType::UINT8;
}

make_tensor(data_dt, tc.data_shape, &d_dev, &d_t);
make_tensor(idx_dt, idx_shape, &i_dev, &i_t);
make_tensor(upd_dt, upd_shape, &u_dev, &u_t);
make_tensor(y_dt, tc.data_shape, &y_dev, &y_t);
// 之后照常调 aclnnBenchNoHarnessScatterGetWorkspaceSize 并比对返回码
```

2. 修正 L2_007 的 CSV，把越界放到**非 dim 轴**（dim=0，data=[8,10]，indices=[5,12] 使 indices[1]=12 > data[1]=10）：

```text
修复前: L2_007,L2,"[8, 10]",float32,"[12, 6]",int32,"[12, 6]",float32,0,None,N/A,indices dim size > data dim size (12>8),ACLNN_ERR_PARAM_INVALID,161002,indices dim size must <= data dim size
修复后: L2_007,L2,"[8, 10]",float32,"[5, 12]",int32,"[5, 12]",float32,0,None,N/A,indices non-dim size > data (12>10),ACLNN_ERR_PARAM_INVALID,161002,indices non-dim size must <= data size
```

3. Mock 的 `ValidateCase` 逐维比较保持与算子一致的非 dim 轴语义（本会话通过 CSV 改形使两侧一致，未改 ValidateCase 的逐维逻辑——如后续算子校验精确化，需同步把 dim 轴从比较中排除）。

设计语义备忘（scatter 族）：indices.shape[dim] 可任意（决定沿 dim 展开的更新次数）；indices.shape[非 dim 轴] ≤ data.shape[对应轴]（非 dim 坐标要落在 data 内）；索引**值**域 [0, data.shape[dim])。算子侧实际校验可能比 torch 语义更严（dim 轴也查）——先逐维核对用例形状与算子实际校验分支，见同域 [scatter 复现用例形状撞校验卡](aclnn_561002_scatter_non_dim_shape_violation.md)。

## 设计决策

- 选"按 case_id 注入非法属性"而不是把 CSV 的 dtype/shape 列直接改成非法值：CSV 列还要供 Mock/golden 数据生成使用（非法 dtype 无法生成 golden 数据），注入式保持 CSV 单一事实源、Real 构造器负责物化非法性。
- L2_007 修 CSV 而不是在测试代码里改形：用例语义（哪一维越界）应体现在用例数据里，避免测试代码里堆积 per-case 特判（先在代码里改形验证、确认后落回 CSV，两步都做了）。
- 当前方案路线档位：recommended_fastpath。

## 预防

- L2 用例评审清单：每条用例必须能回答"非法性物化在哪一环"——note 只是说明，构造器必须真的造出非法输入；Mock 的 fallback 通过不能作为用例合格的证据。
- scatter 族校验用例矩阵显式区分 dim 轴/非 dim 轴/索引值域三类约束，各配正反用例。
- "Mock 全过 + Real 全挂"且 Real 返回码都是 0，是脚手架未构造非法输入的特征信号。

## 验证信号

以下信号来自 L2 用例组修复会话：修复前 Real 6 条返回 0，注入真实非法属性并修正 L2_007 语义后全组转绿。

- 修复前：L2_001/002/003/004/006/007 六条 `GetWorkspaceSize 返回 0 (expect 161002)`，Mock 同组 10/10 通过（note fallback）（来源平台：950；证据等级：tool_observed）。
- 修复后：L2 组 10/10 通过（含改形后的 L2_007 非 dim 轴越界用例）；全量 ST Real 875/875、Mock 875/875（Mock 无回归）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 注入式构造器绑定本算子的 case_id 命名；换用例编号方案需同步维护映射。
- "indices.shape[dim] 不受限"是 scatter 语义（与 PyTorch 对齐）；若某实现/框架额外限制 dim 轴形状，需按其 spec 重写正反用例。
- Mock 的 `ValidateCase` 仍是逐维比较（含 dim 轴）——当前靠 CSV 用例避开 dim 轴差异，校验器本身未精确化。

**未决项**：

- `ValidateCase` 的逐维比较应改为显式跳过 dim 轴，与算子校验语义完全对齐（当前靠用例数据规避）。
- reduce 越界值（5）的注入是硬编码，若算子后续扩展 reduce 枚举需同步。
## 坏实践

- 用 note 字符串 fallback 让 Mock"通过"异常用例（掩盖构造器缺陷）。
- 把 CSV 的 dtype/shape 列改成非法值来"修"Real 失败（会破坏 golden 生成）。
- 在 dim 轴上构造 indices.shape[dim] > data.shape[dim] 并期望报错（这是合法输入）。

## 误诊

- 最初 6 条失败一度被并入"算子参数校验逻辑缺失"的总因里排查算子侧 `CheckShape`；读脚手架构造逻辑后确认算子侧校验本身工作（其余 4 条 L2 用例一直通过），问题全在测试构造层。
