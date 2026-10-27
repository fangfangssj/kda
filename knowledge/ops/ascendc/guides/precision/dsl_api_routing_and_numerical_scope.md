---
type: Guide
title: DSL API 路由的数值与重载边界
description: 保留归约、单目、双目及Gather映射，修正标量替换和Rsqrt泛化。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/ascend_api/tl_asc_routing.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/error_correction/error_correction_examples.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/common_pitfalls.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/multi_dtype_guide.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: []
---

# DSL API 路由的数值与重载边界

路由把reduce_sum/max/min映射归约，把vexp/vsqrt/vrsqrt/vabs/vreciprocal/vrelu/vtanh/vasin/vsin/vlog/vlog2/vlog10/verf映射单输入算子，vadd/vsub/vmul/vdiv/vmax/vmin映射双输入，乘/加/最小/最大标量与clamp映射tensor-scalar，gather映射索引取数。这是语义族导航，实际AscendC名字、基础/高阶/寄存器家族、scratch、dtype、精度和平台支持须分别匹配，不能所有都套(dst,src,count)。

归约count表示有效输入域，workspace及src/dst别名按具体重载约束；来源把三者都可共享任意同址说成节省地址空间过宽。ReduceSum缺workspace的报错可从真实签名修正，但版本也可能有其他重载，不能据样例一概排除。Gather的offset数值单位依API，ReinterpretCast<uint32_t>只保留位模式，不能把元素序号自动转字节偏移或把float数值变整数索引。

减标量可在合法加法重载中使用负数，除标量可用乘倒数作为候选，但二者浮点舍入、0/Inf/NaN、整数溢出或有符号范围不总等价。1.0f/n避免整数除法，仍涉及n到浮点的转换；把static_cast<float>(unsigned)删掉不保证设备不再执行受限转换，需目标编译和数值范围检查。禁止显式float cast的源建议不能推广到全部Ascend C语言模式。

Scalar sqrt/log与Tensor Sqrt/Log不是相同签名，未声明名字可因头文件、命名空间、编译模式或根本不支持。count1 Vector加GetValue是可选实现，需合法地址、有效count和V→Scalar完成；另一个修复例又把AscendC::Sqrt(scalar)换成sqrt，说明需按实际编译环境，不能统一“所有标量数学都用Vector”。count≥8并非所有FP32运算的有效元素数下限，存储对齐与计算mask分开。

来源dav_c220给BF16→FP32 CAST_NONE、FP32→BF16 CAST_RINT、FP16→FP32 CAST_NONE、FP32→FP16 CAST_NONE等路径；输出RINT表达最近偶数，确切支持还需SDK实现。CAST_NONE错误路径可能静态断言、运行断言或未写输出，不能统一断言总是静默garbage且所有精度测试必失败。同型float Cast也不应无意义调用；源Add注释列ROUND/FLOOR/CEIL/TRUNC并非本轮已验证支持矩阵。

Rsqrt所有dtype固定约10bit、所有使用者必须NR、一次NR必达FP32是来源过度泛化。y1=y0*(3-x*y0*y0)/2在正有限x且初值适当时改进近似，实际精度受初值、舍入、overflow、特殊值和容差影响。标准Softmax不计算1/sqrt(variance+eps)，不能因归一化名字列入该Rsqrt模板。Direct基本API也有误差，近似可在明确误差合同下合法，不是任何近似都禁止。

多tile归约需要正确初始化和累加，数值次序不同于一次全长归约；输入依赖必须保持数学语义，但可证明不影响结果的输入可被优化消除，不是每个输入都必须物理加载。源码全零/复制等占位不能替代实际公式，GetValue循环替换Vector的10–100倍收益未附测量，不作通用性能断言。

## 普通 Sigmoid 实现补例：dtype 分流不是普适误差保证

普通 Sigmoid 导出实现提供了一个具体候选：`float` 输出分支使用 `Div` 和持久化 ones，`bfloat16_t` 分支转 fp32 后走 `Reciprocal`，其余半精度分支保留本型链。其可审阅的核心为：

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
```

调用点为 `SigmoidChain<float, false>(xLocal,xLocal,ones,count)`（bf16 桥）、`SigmoidChain<dataType,true>(yOutLocal_,xInLocal_,ones,count)`（fp32）或 `SigmoidChain<dataType,false>(yOutLocal_,xInLocal_,ones,count)`（半精度）。bf16 返回时使用源码中的 `CAST_ROUND`；这只是该实现的调用，不构成所有平台的 Cast 支持矩阵。常量与缓冲生命周期见[短向量链模式](../../runbooks/performance/semantic_vector_pass_reduction.md)。

来源为普通 Level1 Sigmoid 的 `csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h`，原文件 SHA-256 `8e6f726e49e8552fcd264faea7d85c7d19b321e00f0130f66ce83418fbd9445f`。[本节公开证据包](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_pr136_sigmoid_evidence_20261019.zip)的 ZIP SHA-256 为 `6f0f43fca6f7476e7206eb9c4f8cfd0f91fe1495188b9788e5fd30e2f48fbd76`；本节实现证据见上述公开附件，上列固定 DSL 来源支持前文路由讨论，不能替代本节的实现证据。

包内 `source_excerpt.txt` 的 SigmoidChain/Compute 片段保留原方法及行号；`source_binding.json` 记录原文件、摘录和原导出提交/评测记录的哈希关联。`historical_record.json` 的 `/benchmark` 为 `official-tasks 1.1.0`、`910c/A3`，`/result` 中 `accuracy_passed_cases=passed_cases=total_cases=80`。这是一条历史记录的脱敏投影，只支持原整 artifact 的80/80观察，不独立证明每条模板分支得到覆盖，也不认证实际加载二进制。本轮未编译或运行，CANN 版本未独立绑定；既有 `draft` 与空平台范围不变。包内验证器只核对文件完整性与脱敏字段。

不能把一次整 artifact 通过改写为“低精度舍入总会吸收近似误差”。验收域同时包含输入范围、特殊值、后续运算、最终 dtype 和比较器；从一次 Reciprocal 选择也不能推出一次 Rsqrt 精化就必达 fp32。改变其中任一项时，保留精确候选并重新做分区数值验证。源码注释中的误差位数和速度数字没有独立原始记录支持，本节不采用。
