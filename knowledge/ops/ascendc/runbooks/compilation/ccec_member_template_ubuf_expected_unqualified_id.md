---
type: "Runbook"
title: "__ubuf__模板成员辅助函数编译失败改宏"
description: "arch35 kernel里把带__ubuf__指针形参的分块辅助函数写成类模板成员（类外定义）触发ccec报expected unqualified-id与undeclared identifier；放弃模板成员改用预处理宏展开asc_vf_call分块即可编译并通过。"
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# __ubuf__模板成员辅助函数编译失败改宏

## 现象

为给 kernel 里所有 `asc_vf_call<VF>` 调用点做分块改造（规避 `__simd_vf__` 多 rep 截断），先按常规 C++ 写法在 kernel 类里声明了一组模板成员辅助函数（形参带 `__ubuf__` 地址空间限定指针），并在类外提供定义：

```cpp
// 类内声明（节选）
// Chunked VF call helpers (split into MAX_VF_REP-sized chunks to avoid __simd_vf__ loop truncation)
template<typename VF> __aicore__ inline void callVF1(__ubuf__ float* dst, __ubuf__ float* src);
template<typename VF> __aicore__ inline void callCastIn(__ubuf__ float* dst, __ubuf__ bfloat16_t* src);
template<typename VF> __aicore__ inline void callReduceP(__ubuf__ float* work, __ubuf__ float* src);
// ... 共 9 个，覆盖 0/1/2 src、标量、cast、partial-reduce 等调用形态

// 类外定义（节选）
template <typename T, int isEmpty>
template<typename VF>
__aicore__ inline void BenchHarnessEngramGateFusionKernel<T, isEmpty>::callVF0(__ubuf__ float* dst) { ... }
```

执行 `cd operators && bash build.sh <pkg>` 编译失败，报错原文：

```text
RuntimeError: Kernel Compilation Error: OpType BenchHarnessEngramGateFusion Kernel File .../op_kernel/bench_harness_..._apt.cpp!
<..._kernel.h>:353:5: error: expected unqualified-id
<..._kernel.h>:362:5: error: expected unqualified-id
<..._kernel.h>:366:5: error: expected unqualified-id
<..._kernel.h>:374:5: error: expected unqualified-id
<..._kernel.h>:388:32: error: use of undeclared identifier 'tmpBuf_'
<..._kernel.h>:389:31: error: use of undeclared identifier 'lookbackBuf_'
```

错误行号全部落在类外模板成员定义处；`undeclared identifier 'tmpBuf_'` 是定义体脱离类作用域后的连锁报错。

## 适用场景

- 工程形态：aclnn 算子仓 kernel 编译（operators/<pkg>/op_kernel/arch35，bash build.sh 自定义算子包构建，ccec kernel 编译器）。
- API 路径：`__ubuf__` 地址空间限定指针、`__aicore__ inline` 成员函数、`asc_vf_call<VF>`、kernel 类模板（`template <typename T, int isEmpty>`）。
- 验证过的平台与 CANN 版本：950（arch35）+ CANN 9.2.0。
- 适用粒度：operator_family——所有需要给 VF 调用做封装/分块的 arch35 kernel。
- 路线优先级：recommended_fastpath（宏实现分块调用）。
- 明确不适用/未验证场景：普通（非 `__ubuf__` 形参、非 kernel 类）模板成员不受此影响；CANN 后续版本若修复 ccec 支持可回退模板写法。

## 根因

单句归因：ccec（Ascend C kernel 编译器）对"kernel 类模板的类外成员模板定义 + `__ubuf__` 限定指针形参"组合支持不完整，类外定义的 `template <typename T, int isEmpty> template<typename VF> ... ::callVF0(...)` 语法解析失败（expected unqualified-id）。

相邻卡片与区分条件：

- [simd_vf 互调 helper 卡](simd_vf_helpers_use_simd_callee.md)——函数级标注问题：`__simd_vf__` 互调须把被调 helper 标 `__simd_callee__`；本卡是类结构的成员定义形态问题，与函数标注无关。
- [CastTrait 文件域声明卡](reg_casttrait_define_inside_simd_vf.md)——同为"ccec 对 C++ 声明作用域/形态支持不全"家族，那卡是常量须从文件域移入 `__simd_vf__` 函数域；本卡是成员函数须从类外模板定义改为宏展开，方向相反。
- [__simd_vf__ if constexpr 致 hoisted 报错卡](simd_vf_if_constexpr_lowering_banned_exit_70.md)——VF 体内部代码形态的后端限制；本卡在类外定义处即解析失败（expected unqualified-id），错误位置先于代码生成阶段。
- [dim3 未声明须引入 SIMT 公共头卡](asc_vf_call_launch_config_requires_simt_header.md)——`asc_vf_call` 调用配置的前置坑（common_functions.h 暴露 dim3 与重载）；做 VF 分块封装前先确认该头已引入。

## 解决配方

放弃模板成员辅助函数，改用预处理宏在调用点直接展开分块循环（宏体 verbatim）：

```cpp
// Chunked VF call macros — split large D into MAX_VF_REP-sized chunks to avoid
// __simd_vf__ loop truncation beyond 8 reps (observed on Ascend 950 / DAV 3510).
#define CHUNKED_CALL1(VF, dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CALL0(VF, dst) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CALL2(VF, dst, a, b) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (a)+_co, (b)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CALL1S(VF, dst, sc, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, sc, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CALL1SW(VF, dst, src, sc, w) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, sc, (w)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CAST_IN(VF, dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CAST_OUT(VF, dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_REDUCE_P(VF, work, src) do { \
    uint32_t _so=0; uint16_t _rr=(uint16_t)dAlignElems_; uint16_t _wi=0; \
    while(_rr>0) { uint16_t _ck=_rr>MAX_VF_ELEMS?MAX_VF_ELEMS:_rr; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((work)+_wi, (src)+_so, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _so+=_ck; _rr-=_ck; _wi+=_cr; } } while(0)
#define CHUNKED_REDUCE_DOT_P(VF, work, a, b) do { \
    uint32_t _so=0; uint16_t _rr=(uint16_t)dAlignElems_; uint16_t _wi=0; \
    while(_rr>0) { uint16_t _ck=_rr>MAX_VF_ELEMS?MAX_VF_ELEMS:_rr; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((work)+_wi, (a)+_so, (b)+_so, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _so+=_ck; _rr-=_ck; _wi+=_cr; } } while(0)
```

同时删除类内的模板成员声明与类外定义；调用点用 sed 批量把 `callXxx<VF>(` 替换为 `CHUNKED_XXX(VF, ` 形态：

```bash
cd operators/<pkg>/op_kernel/arch35 && sed -i \
  -e 's/callCastIn<\(CastBf16ToF32RowVF\)>(/CHUNKED_CAST_IN(\1, /g' \
  -e 's/callCastOut<\(CastF32ToBf16RowVF\)>(/CHUNKED_CAST_OUT(\1, /g' \
  -e 's/callReduceP<\(ReduceSumSqPartialsVF\)>(/CHUNKED_REDUCE_P(\1, /g' \
  -e 's/callReduceDotP<\(ReduceDotPartialsVF\)>(/CHUNKED_REDUCE_DOT_P(\1, /g' \
  -e 's/callVF1sw<\(RmsNormRowVF\)>(/CHUNKED_CALL1SW(\1, /g' \
  -e 's/callVF1s<\(BcastMulRowVF\)>(/CHUNKED_CALL1S(\1, /g' \
  -e 's/callVF0<\(ZeroFillRowVF\)>(/CHUNKED_CALL0(\1, /g' \
  -e 's/callVF1<\(SiLURowVF\)>(/CHUNKED_CALL1(\1, /g' \
  -e 's/callVF2<\(AddRowVF\)>(/CHUNKED_CALL2(\1, /g' \
  bench_harness_<op>_kernel.h
```

注意：替换/删除大批调用点后先 `grep "error:"` 确认无残留旧符号（如漏删的旧模板定义、丢失的函数签名），再跑完整构建。

## 设计决策

- 宏 vs 模板成员：模板成员类型安全、可调试，但 ccec 编译不过（本卡现象）；宏无类型检查但与 `__ubuf__` 指针、`asc_vf_call<VF>` 模板实参组合可靠。取舍为可靠编译优先。
- 宏内每块后 `PipeBarrier<PIPE_ALL>`：保守同步，避免块间读改写竞争；更细粒度同步未验证。

## 预防

- arch35 kernel 里需要"按模式封装 VF 调用"时，直接写宏，不要先写类模板成员再踩编译器支持坑。
- 大批量机械替换调用点后，用 `grep -c` 对账新旧符号数量，防止残留半改状态（本次曾出现漏删旧定义与丢失函数签名两轮返工）。

## 验证信号

本次验证来自该融合算子 kernel 分块改造过程中的编译修复。

- 模板成员版：`bash build.sh` 报 `error: expected unqualified-id`（4 处）+ `use of undeclared identifier 'tmpBuf_'/'lookbackBuf_'`（来源平台：950；证据等级：tool_observed）。
- 宏版：`bash build.sh` 编译成功（`grep "error:"` 无输出），随后 run.sh 14/23 用例通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡限定 ccec + arch35 + CANN 9.2.0；后续 CANN 版本可能修复类模板成员 + `__ubuf__` 支持，届时可回退模板写法。
- 宏不做类型检查：调用点参数个数/类型错误只能在 `asc_vf_call` 实例化时暴露，替换时需逐形态核对。

**未决项**：

- 无（该编译错误已闭环；分块语义本身的验证见"__simd_vf__多rep截断"卡）。
## 坏实践

- 半改状态直接构建：先删声明再改调用点、中间态丢函数签名（`ProcessOneBhc` 签名丢失）又引入一轮 `expected unqualified-id`——机械替换要一次到位并对账。
