---
type: "Runbook"
title: "SIMT VF 栈数组动态索引致输出全零"
description: "__simt_vf__ 计算函数内声明栈分配固定大小数组（如 int32_t rowPtrStart[32]）并用运行期变量下标动态索引（rowPtrStart[localR]）时，SIMT 编译器生成错误代码——数组读写未落在预期栈位置，Pass1 marker 与 Pass2 填充全部失效，Convert kernel 启动却静默产出全零（bsrColIndC/bsrValC 全零，bsrRowPtr 正确）。判别=二分定位含/不含局部数组版本的参数化测试。"
tags: ["task.debugging", "topic.programming_model.execution_model.simt", "topic.implementation.api_layer.simt_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T09:12:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SIMT VF 栈数组动态索引致输出全零

## 现象

在 Ascend950（arch35/DAV_3510，CANN 9.0.0-beta.2）上为 ops-sparse 仓 PR #91 的 `csr2gebsr` 算子（CSR→GEBSR 格式转换，SIMT）做性能优化时，为减少 Pass1 与 `FillBlockValues` 对 `csrRowPtrA` 的重复 GM 读，在 `__simt_vf__` Convert 计算函数 `Csr2gebsrConvertSimtCompute` 内声明固定大小局部数组 `int32_t rowPtrStart[32]` / `int32_t rowPtrEnd[32]`，在循环里把 `csrRowPtrA[rowStart..rowEnd]` 缓存到局部数组，Pass1 再用 `rowPtrStart[localR]`（`localR = r - rowStart`，随循环变化）做运行期变量下标动态索引读取。编译通过（无 error，LSP 对 AscendC 专有类型误报），运行 `csr2gebsr_test` 全部参数化用例 FAIL：`bsrRowPtr` 正确（Nnz kernel 未改），但 `bsrColIndC` 与 `bsrValC` 全零——Convert kernel 启动却未写入任何值。单测 `L0_4x4_b2x2_idxMix`（m=4,n=4,rowBlockDim=2,colBlockDim=2,baseA=0,baseC=1）输出 bsrColInd=(0,0,0,0)、Golden=(1,2,1,2)。

## 适用场景

- **工程形态**：SIMT kernel 直调（ops-sparse 仓 arch35 csr2gebsr），`__simt_vf__`/`__simt_callee__` 计算函数。
- **API 路径**：SIMT VF 函数声明规范（`__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline`），参数指针类型仅支持 `__gm__ T*`/`__ubuf__ T*`，标量支持 int32_t 等；规范未承诺函数体内栈数组的动态索引语义。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510/arch35），CANN 9.0.0-beta.2。仅 950 有实测。
- **适用粒度**：`platform_wide`——所有使用 SIMT `__simt_vf__` 计算函数并尝试用栈分配数组做运行期变量下标动态索引的 AscendC 算子。
- **路线优先级**：`recommended_fastpath`——在 VF 函数内避免栈分配数组的动态索引，改用标量或直读 `__gm__` 指针。
- **不适用或未验证场景**：传统 AscendC Vector（非 SIMT）编程模型的 UB/栈行为不同，不适用本结论；a2/a3 未验证，推断可迁移（同 SIMT 编译器），标注"未验证，推断可迁移"。

## 根因

`__simt_vf__`（SIMT VF 计算函数）内声明栈分配的固定大小数组（如 `int32_t rowPtrStart[32]`）并在循环里用运行期变量下标动态索引（`rowPtrStart[localR]`，`localR = r - rowStart` 随循环变化），SIMT 编译器生成错误代码——数组写入/读取未真正落到预期栈位置，导致 Pass1 标记 marker 与后续 Pass2 填充全部失效，Convert kernel 静默产出全零。VF 函数声明规范（vf-declaration.md）只承诺参数支持 `__gm__ T*`/`__ubuf__ T*` 指针与标量类型，未承诺函数体内栈数组的动态索引语义，故此用法超出 VF 函数承诺范围。

## 判别方法

**触发条件**（tool_observed 失败症状）：SIMT kernel 编译通过、运行无 errno，但 Convert 类 kernel 输出全零（`bsrColIndC`/`bsrValC` 全零）而其他未改 kernel 的输出（`bsrRowPtr`）正确；单测输出数组为 `(0,...,0)` 而 Golden 非零。即"kernel 启动却静默不写入"的全零模式。

**可执行探针**：二分定位——对含局部数组动态索引的版本与移除局部数组改直读 `__gm__ csrRowPtrA` 的版本，分别跑 `csr2gebsr_test --gtest_filter=*L0_4x4_b2x2_idxMix*`；逐步叠加无关修复（Fix1 双指针 / Fix4 转型 / Fix2 min-max）观察是否仍 PASSED，仅加回局部数组缓存（Fix3）观察是否 FAIL；最后 `git checkout` 原码确认基线 PASSED。判别核心是"加回局部数组即 FAIL、移除即 PASSED"的单点归因，而非逐行代码推理。

**原始观测**（Ascend950，CANN 9.0.0-beta.2，csr2gebsr_test）：
- 含局部数组版本：全部参数化用例 FAIL，`bsrColIndC`/`bsrValC` 全零、`bsrRowPtr` 正确（Nnz kernel 未改）。
- 二分：Fix1(双指针)+Fix4(转型) 单测 PASSED；+Fix2(min/max) PASSED；+Fix3(局部数组缓存) FAIL；`git checkout` 原码 PASSED——局部数组动态索引为唯一元凶。
- 移除局部数组、保留 `rowBlockDim > 32` 防御检查、Pass1 直读 `__gm__ csrRowPtrA`：64/64 PASSED（37 参数化 + 26 异常 + 1 perf）。

**机制解释**（根因分析，属 interpretation）：SIMT VF 函数内栈分配数组 + 运行期变量下标动态索引触发编译器错误代码生成，数组读写未落在预期栈位置 → Pass1 marker 与 Pass2 填充失效 → kernel 启动但静默不写入 → 输出保持初始全零缓冲。判别核心是"加回局部数组即 FAIL、移除即 PASSED"的实验归因，与 VF 声明规范"只承诺参数指针/标量、不承诺函数体内栈数组动态索引"的边界一致。

## 预防

- 在 `__simt_vf__`/`__simt_callee__` 计算函数内禁止用栈分配数组做运行期变量下标的动态索引；如需缓存多值，用标量集合或 `__ubuf__`/`__gm__` 缓冲。
- 性能优化引入局部数组后，必须跑全量参数化测试（不只跑 perf sweep），因本 bug 编译通过、perf 可能仍跑通但功能静默全零。

## 验证信号

- 含局部数组（rowPtrStart[32]/rowPtrEnd[32] 动态索引）版本：csr2gebsr_test 全部参数化用例 FAIL，bsrColIndC/bsrValC 全零、bsrRowPtr 正确（来源平台：950；证据等级：tool_observed）
- 二分定位：Fix1+Fix4 单测 PASSED；+Fix2 PASSED；+Fix3(局部数组缓存) FAIL；git checkout 原码 PASSED——局部数组动态索引为唯一元凶（来源平台：950；证据等级：tool_observed）
- 移除局部数组、Pass1 直读 __gm__ csrRowPtrA 后：64/64 PASSED（37 参数化 + 26 异常 + 1 perf）（来源平台：950；证据等级：tool_observed）
- before（含局部数组 buggy 版）git diff 与 after（移除局部数组版）文件读逐字可核（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 传统 AscendC Vector（非 SIMT）编程模型的 UB/栈行为不同，本结论（栈数组动态索引失效）不适用。
- a2/a3 未实测，推断可迁移（同 SIMT 编译器，限制应一致），但未验证。
- 本判别只针对"运行期变量下标动态索引"的栈数组；编译期常量下标（如 `rowPtrStart[2]`）未在本 trace 中单独验证是否同样触发。

## 坏实践

- 见"编译通过 + perf sweep 跑通"即放行优化，不跑功能参数化测试——本 bug 正是 perf 可跑通、功能静默全零。
- 在 `__simt_vf__` 函数体内用栈数组缓存 GM 读以"减少重复访存"——这超出 VF 函数声明规范承诺的参数语义，触发编译器错误代码生成。
- 逐行代码推理定位全零根因，而不做二分加/减局部数组的单点归因实验——全零无 errno、无越界，逐行推理易误判。

## 误诊

- 无明显误诊路径：二分定位较早排除了 Fix1(双指针)/Fix2(min/max)/Fix4(转型) 的责任（三者单独叠加均 PASSED），并经 `git checkout` 原码 PASSED 反证，直接锁定 Fix3(局部数组缓存) 为唯一元凶。