---
type: "Runbook"
title: "507035崩溃：负模运算产生UB负偏移"
description: "环形缓冲行号用 (L-SL+n)%SL 计算，L<SL 时 C++ 负模产生负索引，标量 GetValue 以负偏移访问 UB 触发 507035；改为恒非负的 (L+n)%SL 即修复。"
tags: [task.debugging, topic.compute.indexing, topic.memory]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# 507035崩溃：负模运算产生UB负偏移

## 现象

在为自定义 ACLNN 算子包（BenchHarness 前缀、Ascend C 纯 Vector kernel，`op_kernel/arch35/` 目录）做 Normal 分支（tilingKey=0）kernel 上板联调时，ST 黑盒测试全部崩溃：在 `tests/aclnn` 下编译出的 `./test_aclnn` 在**最小调用通路用例**上即返回

```
ACL err 507035 at .../operators/engram_gate_fusion_package/tests/aclnn/test_aclnn.cpp:435
```

（在 `aclrtSynchronizeStream` 处返回 507035，AICore 执行失败），23 个用例 0 通过，连最小通路也过不了。此前的第一轮修复（含补齐 InferShape 的 output[1] 形状）没有消除崩溃。当时代码状态：kernel 主链为 `__simd_vf__` + `asc_vf_call` 的 7 步前向 + lookback conv，主输出 out0 逐 l 用 `DataCopyPad` 写 GM，崩溃点不明——报错只给错误码，不给 kernel 内部位置。

## 适用场景

- 工程形态：aclnn 算子包 ST 黑盒联调（`operators/<pkg>/tests/aclnn/run.sh` → `./test_aclnn`），kernel 侧含标量 `LocalTensor.GetValue/SetValue` 循环。
- API 路径：AscendC 标量 UB 访问（`LocalTensor::GetValue/SetValue`）、`AscendC::PRINTF`（定位用）、ACL 错误码 507035（AICore kernel error）。
- 验证过的平台与 CANN 版本：950 + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——"C++ 负模 → 负索引 → UB 负偏移标量访问"机制不依赖具体算子，任何在 kernel 侧用取模做环形/回绕索引的写法都适用。
- 路线优先级：recommended_fastpath（把索引表达式改成恒非负形式，零额外开销）。
- 明确不适用 / 未验证场景：崩溃若定位在 DataCopyPad/MTE 搬运通路或 host 侧（tiling 分发、TilingData 序列化），根因不同；本卡的 PRINTF 二分定位方法仍可复用，但结论不可直接套用。

## 判别方法

kernel 侧无源码级调试器，用 `AscendC::PRINTF` 二分定位崩溃点（宏用法与 ASCENDC_DUMP 开关见 [kernel printf 二分定位卡](kernel_printf_ascendc_macro_ascendc_dump_bisect.md)）：在 kernel 入口 Init（打印全部 tiling 参数与输出 GM 指针）、每个处理单元入口、out0 写前/写后、conv_state_out gather 开始、csout 写开始等关键点插 PRINTF，重编译重跑。本例观测：

```
[AIV Block 0/2] K Init: B=1 L=4 HC=2 D=8 K=4 dil=3 SL=9 dAlign=8 hasCS=0 out0=0x12000001e000 out1=0x12000001f000
[AIV Block 0/2] K before out0 write: l=0
[AIV Block 0/2] K after out0 write: l=0
...（l=0..3 全部完成）
[AIV Block 0/2] K csout gather start: b=0 hc=0 SL=9 D=8
（再无 "csout write start"，随后 507035）
```

要点：两个输出 GM 指针均非空有效（排除"输出指针无效"假设）；out0 的 DataCopyPad 全部完成；日志停在 gather 循环开始 → 崩溃在**标量 SetValue/GetValue 循环**内，而非 MTE3 写出。

同错误码 507035 的三种不同根因（同码互斥判别，先分诊再修复）：

- [507035 workspace 保留区卡](mte_ddr_507035_workspace_reserved_region.md)——根因是 tiling 声明的 workspace 未含平台保留区（950 为 16MB），kernel 对用户工作区的 MTE 写落在分配范围外（plog 错误码 95 MTE DDR 越界）；区分条件：kernel 使用 workspace 且 plog 报 MTE DDR 越界→查保留区声明，本卡不涉及 workspace、崩溃点在标量 gather 循环。
- [507035 多余核越界分发卡](507035_unguarded_extra_cores_oob_dispatch.md)——根因是硬件全核启动而任务数小于核数、多余核以越界任务号访问 GM；区分条件：printf 见 Block 号远超任务总数且崩溃集中在大 D+大 L→查分发守卫，本卡崩溃与短序列（L<SL）强相关。

其他相邻卡片与区分条件：

- [arch35 64 位 div/mod 位宽卡](../compilation/arch35_64_div_mod_bitwidth_32.md)——同为取模/除法家族但形态不同：那卡是 64 位 div/mod 的后端编译约束（`Div of bitwidth greater than 32 not supported` 编译期报错），本卡是运行期 C++ 负模语义（取模本身合法，负值索引才是坑）。
- [arch35 GetValue/SetValue 标量 bypass 卡](../compilation/arch35_getvalue_setvalue_bypass.md)——同涉标量 `LocalTensor::GetValue/SetValue`：那卡是 RegBase 向量 kernel 用标量 bypass 的精度劣化（banned 路线），本卡是负偏移越界直接崩溃，现象与边界不同。

## 根因

conv_state_out 收集循环用 `sr=(L_-stateLen_+n)%stateLen_` 计算环形缓冲行号。C++ 的 `%` 对负被除数返回**负值**：当 `L_=4 < stateLen_=9` 时，`(4-9+n)%9 = (n-5)%9`，在 `n<5` 时得到 -5..-1；随后的 `lb.GetValue(sr*dAlignElems_+d)` 以负偏移访问 UB。kernel 侧标量 UB 访问没有边界保护，负偏移直接触发 AICore 异常，向上表现为 `aclrtSynchronizeStream` 返回 507035。数学上 `(L-SL+n) mod SL ≡ (L+n) mod SL`（因为 `SL ≡ 0 (mod SL)`），所以 `(L_+n)%stateLen_` 与原式**同值但被除数恒非负**（`L_≥0, n≥0`），一行即可修复。

## 解决配方

修复前（`op_kernel/arch35/*_kernel.h`，conv_state_out 收集循环）：

```cpp
        // Gather: csOut[d*SL+n] = bf16(lookback[(L-SL+n)%SL, d])
        for(int64_t d=0;d<D_;d++) {
            for(int64_t n=0;n<stateLen_;n++) {
                int64_t sr=(L_-stateLen_+n)%stateLen_;
                float v=lb.GetValue(sr*dAlignElems_+d);
                uint16_t bits=fp32_to_bf16_rint_scalar(v); T bv; __builtin_memcpy(&bv,&bits,sizeof(bits));
                cs.SetValue(d*stateLen_+n, bv);
            }
        }
```

修复后（索引恒非负）：

```cpp
        // Gather: csOut[d*SL+n] = bf16(lookback[(L+n)%SL, d])
        for(int64_t d=0;d<D_;d++) {
            for(int64_t n=0;n<stateLen_;n++) {
                int64_t sr=(L_+n)%stateLen_;
                float v=lb.GetValue(sr*dAlignElems_+d);
                uint16_t bits=fp32_to_bf16_rint_scalar(v); T bv; __builtin_memcpy(&bv,&bits,sizeof(bits));
                cs.SetValue(d*stateLen_+n, bv);
            }
        }
```

改动仅一行索引表达式，不动数据布局。修完执行 `bash operators/build.sh <pkg>` 重编译，再在 `tests/aclnn` 下 `export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/<Vendor>/op_api/lib/:${LD_LIBRARY_PATH} && ./test_aclnn` 复跑。

## 设计决策

- 选"改索引为恒非负"而不是"加 `(sr+SL)%SL` 二次归一"：`(L+n)%SL` 与原式模 SL 同余且一次取模即非负，无额外算术；二次归一在 kernel 标量循环里多一次取模开销。
- 不给 `GetValue` 加越界保护分支：标量热循环内加分支会拖慢 gather，且此类索引错误应在源头消除而非运行时兜底。
- 当前方案属 recommended_fastpath：零成本、数学等价、不改变任何 buffer 语义。

## 预防

- 环形/回绕索引一律写成被除数恒非负的形式（`(i+n)%N` 而非 `(i-N+n)%N`）。
- code review 专项检查 `%` 左侧可能出现负值的表达式（`L-SL+n`、`i-k` 这类差值形式是高危信号）。
- 上板出现裸 507035 时，先用 PRINTF 二分收敛到 kernel 内具体循环，再谈修复；不要先改 host 侧（InferShape/tiling）假设。

## 验证信号

以下为该 Normal 分支 kernel 崩溃修复会话中的上板实测（构建链 `operators/build.sh` + `tests/aclnn` 下 `./test_aclnn`）：

- 修复前：首用例即 `ACL err 507035`，23 用例 0 通过；PRINTF 轨迹终止在 `csout gather start: b=0 hc=0 SL=9 D=8`，未到达写出阶段（来源平台：950；证据等级：tool_observed）。
- 索引修复后重编译重跑（同轮）：`Test minimal_call_path: OK`，kernel 完整执行（PRINTF 相继出现 `csout write done`、`ProcessOneBhc done`），数值用例从"崩溃"变为可比较的 bitwise mismatch（来源平台：950；证据等级：tool_observed）。
- 移除全部 PRINTF 后的终态复跑：最小通路仍 OK，无 507035 复现（来源平台：950；证据等级：tool_observed）。

同算子更早排障轮次（第二输出写回 507035，当时根因未追到）的补充观测：

- 对 output[1] 的 MTE3 `DataCopyPad` 写入换五种形状（按元素 strided 2B / 按通道连续 SL*2B / gather 成 D*SL 整块 / 32B 对齐块+尾块拆分 / 换已知可用源 buffer）逐一重编上板，均复现同一 `ACL err 507035`；同参数写 output[0] 主输出不触发（来源平台：950；证据等级：tool_observed）。
- 将该写回段整体短路（`if (cond && false)`）后无 ACL err，`Test minimal_call_path: OK`，全零输入用例主输出 0/64 elems differ（数值正确）（来源平台：950；证据等级：tool_observed）。
- kernel Init 内 PRINTF 观测两个输出 GM 指针均非空且连续相差 4096B，排除"输出指针无效"（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论锚定 950 + CANN 9.2.0 的标量 `LocalTensor` 访问行为；其他芯片代次上"负 UB 偏移"是否同样表现为 507035 未验证，推断机制一致（未验证，推断可迁移）。
- 仅覆盖"负模负索引"这一根因；同样表现为 507035 的 UB 越界（tile 尾块越界）、tilingKey 不匹配、TilingData 结构不一致、workspace 保留区缺失（见同码判别）等根因不在本卡范围。
- `L ≥ SL` 的用例负模不产生负值，故该 bug 只在短序列（L<SL，如 decode/短 prefill）用例上崩溃——长序列用例通过不代表此索引正确。

**未决项**：

- 无（根因已闭合到机制层：负模 → 负偏移 → AICore 异常）。
## 坏实践

- 见"`## 误诊`"：先改 host 侧 InferShape 而不做 kernel 内定位，浪费一轮编译验证。
- 用 `(sr+stateLen_)%stateLen_` 式"事后归一"掩盖源头负值——多一次取模且仍依赖读者意识到负值存在。
- 机制不明时在多种 DataCopyPad 写入形状之间反复试错（单元素 strided / 按通道连续 / 整块 / 32B 对齐拆分 / 换源 buffer 共五轮，每轮完整重编+上板）——该写回段内嵌的正是负模 gather 循环，五种写形状共享同一 gather，"换写形状"是无效搜索维度；正确动作是先用短路 A/B 隔离崩溃域，再在域内用 PRINTF 二分定位（见判别方法）。

## 误诊

- 误诊一：把 507035 归因于"InferShape 未设置 output[1]（conv_state_out）形状 → 框架分配 0 字节/传无效 GM 指针"。按此假设补全 InferShape（output[1]=[B, CD, SL]）并重编译后，507035 依旧；PRINTF 随后证实两个输出 GM 指针均非空、崩溃点在 gather 循环。InferShape 缺形状是真实的卫生缺陷（修复保留），但**不是本例崩溃根因**——该假设的因果在本轮未被隔离验证（仅 agent 陈述，无单独复跑对照）。
- 误诊二（同算子更早排障轮次）：曾把第二输出写回处的 507035 归因为"对第二输出 GM 的 MTE3 DataCopyPad 写入动作本身"，并怀疑 aclnn 框架对 output[1] 的地址传递/映射异常或 MTE3 对 multi-output 的支持限制——A/B 短路隔离只证明了崩溃域在该写回段内，"写动作本身触发"与两个框架侧假设在该轮均未验证。事后看该写回段内嵌的 gather 循环正是本卡同款负模索引（`sr=(L_-stateLen_+n)%stateLen_` 后 `lb.GetValue(sr*...)`），本卡的索引修复保留写回路径后 kernel 完整执行、507035 消失，与那批"五种写法全崩"的观测相容；但不要把"第二输出 MTE3 写入触发 507035"当作已验证结论引用。
- 教训：修复后必须复跑验证因果，"修了一个真缺陷"不等于"修掉了这个现象"。
