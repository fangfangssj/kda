---
type: Runbook
title: 短向量链先固定数值与存储合同，再消除重复工作
description: 用普通 Sigmoid 的原位短链和常驻 ones 展示可审阅的中间存储与重复初始化模式；不把实数恒等式、源码 pass 数或整算子成绩当作单机制收益。
tags:
- task.optimization
- topic.precision
- topic.memory
status: draft
sources: []
created_at: '2026-10-17T18:12:04Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 短向量链先固定数值与存储合同，再消除重复工作

## 适用场景

短逐元素链的中间值无需保留，且目标 API 重载允许相应 src/dst 别名。候选包括跨 tile 不变的常量初始化、已死亡中间量的原位链，以及保持参考求值顺序的常量处理；本卡不提供通用“系数为 0/1 即可删除”的规则。

## 普通 Sigmoid 的实际单元

该实现的静态分流为 `NEED_CAST = is_same_v<dataType,bfloat16_t>`、`USE_DIV = is_same_v<dataType,float>`。初始化时，bf16 分支分配一个 fp32 中间 buffer；fp32 分支分配 ones buffer。`FillOnes` 在 Process 的 tile 循环前调用一次。这里只记录源码选择，不据此保证任意输入都满足数值门限。

```cpp
    __aicore__ inline void FillOnes()
    {
        if constexpr (USE_DIV) {
            if ASCEND_IS_AIV {
                AscendC::LocalTensor<dataType> ones = onesBuf_.Get<dataType>();
                AscendC::Duplicate(ones, static_cast<dataType>(1.0),
                                   static_cast<uint32_t>(tiling_.blockElems));
            }
        }
    }
```

```cpp
    template <typename T, bool DIV>
    __aicore__ inline void SigmoidChain(
        AscendC::LocalTensor<T> &dst, AscendC::LocalTensor<T> &buf,
        AscendC::LocalTensor<T> &ones, uint32_t count)
    {
        AscendC::Muls(buf, buf, static_cast<T>(-1.0), count);
        AscendC::Exp(buf, buf, count);
        AscendC::Adds(buf, buf, static_cast<T>(1.0), count);
        if constexpr (DIV) {
            AscendC::Div(dst, ones, buf, count);
        } else {
            AscendC::Reciprocal(dst, buf, count);
        }
    }
    __aicore__ inline void Compute(int32_t count)
    {
        yOutLocal_ = yOutQueue_.AllocTensor<dataType>();
        xInLocal_ = xInQueue_.DeQue<dataType>();
        if constexpr (NEED_CAST) {
            AscendC::LocalTensor<float> xLocal = xCastBuf_.Get<float>();
            AscendC::LocalTensor<float> ones;
            AscendC::Cast(xLocal, xInLocal_, AscendC::RoundMode::CAST_NONE, count);
            SigmoidChain<float, false>(xLocal, xLocal, ones, static_cast<uint32_t>(count));
            AscendC::Cast(yOutLocal_, xLocal, AscendC::RoundMode::CAST_ROUND, count);
        } else if constexpr (USE_DIV) {
            AscendC::LocalTensor<dataType> ones = onesBuf_.Get<dataType>();
            SigmoidChain<dataType, true>(yOutLocal_, xInLocal_, ones, static_cast<uint32_t>(count));
        } else {
            AscendC::LocalTensor<dataType> ones;
            SigmoidChain<dataType, false>(yOutLocal_, xInLocal_, ones, static_cast<uint32_t>(count));
        }
        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
    }
```

`SigmoidChain` 明确执行负号缩放、Exp、加一，再选择 Div 或 Reciprocal。bf16 先转 fp32，链在同一个 fp32 中间区执行，最后转回；fp32 使用常驻 ones 作分子。原始类的队列负责输入消费与输出发布，不能将这几个方法脱离 Init、分配长度和队列生命周期当作完整算子。

## 精度与存活期复核

ones 常量只有在对应 buffer 未被覆盖、且本次有效长度不超过已初始化长度时才能复用；变长 tail 仅消费有效 count。中间区可原位使用，必须由具体重载及最后消费者的完成性支持，不能仅凭 C++ 变量名消失判断。其他任务的 Muls(1)、Adds(0)、Axpy 或常量融合，仍要分别核对 NaN/Inf、有符号零、相消与转回舍入；实数等价不自动满足这些合同。

## 验证信号

来源为普通 Level1 Sigmoid 导出实现 `csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h`，原文件 SHA-256 为 `8e6f726e49e8552fcd264faea7d85c7d19b321e00f0130f66ce83418fbd9445f`。[公开源码片段与单条历史记录](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_pr136_sigmoid_evidence_20261019.zip)的 ZIP SHA-256 为 `6f0f43fca6f7476e7206eb9c4f8cfd0f91fe1495188b9788e5fd30e2f48fbd76`。公开附件中的具体材料提供核验入口，PR 自述本身不作为验证依据。

包内 `source_excerpt.txt` 提供 Init、FillOnes、Process、SigmoidChain、Compute 的完整机制方法与原行号；只去除注释，不是独立可编译算子。`source_binding.json` 给出原文件、摘录、来源包、原评测报告和选中记录各自的哈希，说明原导出同一提交的关联方法及其限制；摘录 SHA 与原完整文件 SHA 不同。

`historical_record.json` 的 `/benchmark` 为 `official-tasks 1.1.0`、`910c/A3`，`/result` 为 `compile_pass=true`、`accuracy_passed_cases=passed_cases=total_cases=80`。这是单条历史导出记录的脱敏投影，支持原整 artifact 的80/80历史观察；不证明每条dtype分支、有效长度分区或本模式独立得到验证，也不是实际加载二进制的身份认证。CANN 版本未独立绑定，本轮未编译或运行。`platforms: []` 保留未定位的模式适用范围。包内 `verify_public.py` 仅核对文件哈希和脱敏字段，不重放历史设备评测。

可从 Init/Process/FillOnes/Compute 核对 buffer 分流、常量初始化频率和具体链条。现有资料没有本轮目标 lowering 产物或该子机制 before/after，所以不宣称比高阶 Sigmoid 少某个硬件 pass、节省固定 UB 字节或带来固定速度提升。原源码中的估计误差位数、微秒数字也不作为本卡证据。

## 失效边界与采用检查

更改 dtype、后续数学链或参考比较器时重新建立[数值路由合同](../../guides/precision/dsl_api_routing_and_numerical_scope.md)。需要保留原中间结果、别名受限、常量随 tile 变化或输入范围超出已验证域时，保留独立缓冲和精确路径。采用时冻结参考，覆盖特殊值、近零、饱和、舍入边界和尾部；数值先通过，再比较完整调用及资源成本。
