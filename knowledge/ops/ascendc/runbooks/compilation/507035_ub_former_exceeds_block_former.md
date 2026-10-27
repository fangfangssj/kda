---
type: "Runbook"
title: "507035：ubFormer 超 blockFormer 尾块越界"
description: "tiling 接线后所有非空用例在 aclrtSynchronizeStream 报 507035：Process 主循环按 ubFormer 计数拷贝而每核 GM 块只有 blockFormer 元素（ubFormer=12288 > blockFormer=512），独立尾块偏移 loopNum*ubFormer 亦越界；尾块须并入最后一次主循环迭代。"
tags: [task.debugging, topic.memory, topic.data_movement.memory_access, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035：ubFormer 超 blockFormer 尾块越界

## 现象

bessel_i1e 算子 Branch-0（FP32）kernel 在把 op_host TilingFunc 正确接线（真实 dim0/coreNum/ubFormer 写入 TilingData）之后，编译通过，但白盒 ST 的所有非空用例在第一个同步点崩溃：

```
[ 1/14] 0-A_f32_main_block_aligned          ACL error 507035 at .../test_aclnn.cpp:351: aclrtSynchronizeStream(stream)
[ 2/14] 0-B_f32_main_plus_tail              ACL error 507035 ...
...（除两个空张量用例 PASS 外全部 507035）
```

在运行时错误码头文件里确认：

```
.../x86_64-linux/include/acl/error_codes/rt_error_codes.h:#define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      507035 // vector core exception
```

即 kernel 执行期间向量核异常。plog/device 日志无细化信息。用 tiling 推演脚本复算 0-A（dim0=512）的参数：`dim0=512 coreNum=1 blockFormer=512 blockNum=1 ubFormer=12288 ubLoopF=1 ubTailF=512`——**ubFormer(12288) > blockFormer(512)**。

## 适用场景

- 工程形态：算子仓 kernel 开发 + 白盒 ST（run.sh），多级切分（blockFormer 每核 GM 块 × ubFormer UB tile）的 element-wise kernel。
- API 路径：kernel 侧 `Process/CopyIn/Compute/CopyOut` 分层 + `AscendC::DataCopyParams`（blockLen 为 uint16）；tiling 字段 ubLoopOfFormerBlock/ubTailOfFormerBlock/ubLoopOfTailBlock/ubTailOfTailBlock。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何"UB tile 尺寸与每核 GM 块尺寸解耦"的切分实现。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：大 shape 用例修复后残余的 507035（ubFormer 过大/UB 超额）是另一根因，见库内"tile 超 UB"系列卡。

## 根因

Process 主循环固定按 `ubFormer` 个元素做 CopyIn/Compute/CopyOut，并把尾块作为**独立段**放在主循环之后（偏移 `loopNum * ubFormer`）。当 `ubFormer > blockFormer`（小 shape + 大 UB tile 时 tiling 公式给出 ubLoop=1、ubTail=blockFormer）：

- 主循环 i=0 就按 ubFormer=12288 个元素拷贝，而该核 `xGm.SetGlobalBuffer(base, blockFormer=512)` 只有 512 个合法元素 → DataCopy 越界读 GM；
- 独立尾块的偏移 `loopNum * ubFormer` 本身也越过块尾。

越界的 GM 访问触发向量核异常，宿主侧表现为 aclrtSynchronizeStream 返回 507035。tiling 公式与 kernel 消费端各自演进、两端对"ubFormer>blockFormer"的语义没有统一契约，是这一缺陷的工程根源。

## 解决配方

修复前（独立尾块 + 主循环固定 ubFormer）：

```cpp
    __aicore__ inline void Process()
    {
        if (this->isEmpty) return;
        for (int64_t i = 0; i < this->loopNum; ++i) {
            CopyIn(i * this->ubFormer, this->ubFormer);
            Compute(this->ubFormer);
            CopyOut(i * this->ubFormer, this->ubFormer);
        }
        if (this->tailNum > 0) {
            CopyIn(this->loopNum * this->ubFormer, this->tailNum);
            Compute(this->tailNum);
            CopyOut(this->loopNum * this->ubFormer, this->tailNum);
        }
    }
```

修复后（尾块并入最后一次主循环迭代）：

```cpp
    __aicore__ inline void Process()
    {
        if (this->isEmpty) return;
        for (int64_t i = 0; i < this->loopNum; ++i) {
            // Last iteration: use tailNum if non-zero, else ubFormer
            int64_t cnt = (this->tailNum > 0 && i == this->loopNum - 1)
                          ? this->tailNum : this->ubFormer;
            CopyIn(i * this->ubFormer, cnt);
            Compute(cnt);
            CopyOut(i * this->ubFormer, cnt);
        }
    }
```

该形态同时覆盖四种情形：tailNum==0（全主块）；tailNum>0（末迭代收尾）；ubFormer>blockFormer（loopNum=1、tailNum=blockFormer，首迭代即收尾）；ubFormer<=blockFormer（常规）。

判别方法：507035 且 plog 无细化信息时，用独立脚本复算 tiling 参数（dim0/coreNum/blockFormer/blockNum/ubFormer/ubLoop/ubTail），核对 ubFormer 与 blockFormer 的大小关系与尾块偏移算式。

## 设计决策

- 为什么并入主循环而不是给尾块换偏移公式：独立尾块的偏移 `loopNum*ubFormer` 在 ubFormer>blockFormer 时必然越界，而 `(loopNum-1)*ubFormer + tailNum <= blockFormer` 恒成立，并入末迭代后偏移与计数天然自洽。
- 为什么不在 tiling 侧强 cap `ubFormer ≤ blockFormer`（本会话也试过）：cap 只能消除越界维度之一，尾部对齐/末块语义仍需 kernel 侧统一收口；两者可并用。
- 当前方案路线档位：recommended_fastpath。

## 预防

- tiling 公式与 kernel 消费端建立统一契约：host 只管算 ubFormer，kernel 只管按 ubFormer 拷贝时，两端必须对"ubFormer>blockFormer"的语义（首迭代即收尾）有一致理解。
- 把尾块当独立段处理（额外一次起止偏移计算），而不是并入主循环末迭代。
- 507035 无专属 plog 细化信息时，先用 tiling 推演脚本复算参数做算术级排除，再怀疑多核 launch/变体派发。

## 验证信号

来源说明：以下信号来自接线 tiling 后的修复循环（重编译安装 + 白盒 ST 复跑）。

- 修复前：所有非空用例 `ACL error 507035 at ... aclrtSynchronizeStream(stream)`（错误码确认 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`）（来源平台：950；证据等级：tool_observed）。
- 修复后（尾块并入末迭代）：构建 SUCCESS；恒等直通验证下 10/14 用例完成并正确回读输入（0-A got 0.327972412 = 输入值、0-B got 1.9536314、0-D got -7.04158783 等），不再 507035；仅大 dim0 用例（0-C/0-E）仍崩——独立根因（ubFormer 过大），另见对应卡（来源平台：950；证据等级：tool_observed）。
- tiling 推演（独立脚本）：`dim0=512 coreNum=1 blockFormer=512 blockNum=1 ubFormer=12288 ubLoopF=1 ubTailF=512`，`UB bytes queIn: 49152 queOut: 49152`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该修复消除了"ubFormer 与 blockFormer 失配 + 独立尾块"造成的越界；`ubFormer` 自身过大引发的 507035（大 dim0 用例）不受此修复影响（未验证，属另一根因）。
- `DataCopyParams.blockLen` 为 uint16（count*elemBytes ≤ 65535）的上限是另一条约束，本卡未展开。
- 未在其他切分形态（多输出/带 stride）上验证（未验证，推断可迁移）。

**未决项**：

- 507035 无专属 plog 细化信息，定位只能靠排除法；建议后续在干净工程上做"越界读 GM 的最小复现"以固化判别路径（未尝试）。
## 坏实践

- tiling 公式与 kernel 消费端各自演进：host 只管算 ubFormer，kernel 只管按 ubFormer 拷贝，两端对"ubFormer>blockFormer"的语义没有统一契约。
- 507035 出现后先查多核 launch（SetBlockDim）与 kernel 变体选择（simplifiedKey/tilingKey 派发）而不复算 tiling 算术。

## 误诊

- 507035 出现后先查 plog/device 日志（无细化信息），一度怀疑多核 launch（SetBlockDim(blockNum)）与 kernel 变体选择（simplifiedKey/tilingKey 派发）——用 tiling 推演脚本复算 0-A 参数后才锁定越界算术。
