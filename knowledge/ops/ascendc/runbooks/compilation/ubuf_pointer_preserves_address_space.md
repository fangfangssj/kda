---
type: "Runbook"
title: "__ubuf__ 指针传普通形参导致编译失败"
description: "Ascend C 地址空间是类型的一部分；__ubuf__ 指针不能隐式绑定到普通 T* 形参，未使用参数应删除或按真实地址空间声明。"
tags: ["task.debugging", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# __ubuf__ 指针传普通形参导致编译失败

## 现象

在 Ascend950 / dav-3510 kernel 直调工程中，把输出 UB 的 `GetPhyAddr()` 转为 `__ubuf__ T*` 与 `__ubuf__ int64_t*`，然后调用 helper 函数 `StoreOut(off, cnt, vp, ip)`。远端 bisheng build 在四种 dtype 实例化上重复报：

```text
error: cannot initialize a parameter of type 'float *' with an lvalue of type '__ubuf__ float *'
error: cannot initialize a parameter of type 'int *' with an lvalue of type '__ubuf__ int *'
error: cannot initialize a parameter of type 'half *' with an lvalue of type '__ubuf__ half *'
error: cannot initialize a parameter of type '__bf16 *' with an lvalue of type '__ubuf__ __bf16 *'
```

当时代码中 `StoreOut` 的 `vp`/`ip` 形参声明为普通 `T*`/`int64_t*`，而调用点传入 UB 地址空间指针；这两个参数在函数体内并未使用。

## 适用场景

- 工程形态：Ascend C kernel 直调构建；
- API 路径：`LocalTensor::GetPhyAddr()`、`__ubuf__` 地址空间限定、device helper 函数形参；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`platform_wide`，属于 bisheng 的 Ascend C 地址空间类型约束，与算子公式无关；
- 路线优先级：`recommended_fastpath`，删除未使用参数既保持类型正确也减少接口噪音；
- 明确不适用：host C++ 普通内存指针；确实需要传 UB 地址的 helper 应显式保留匹配的地址空间限定，不能照抄“删除参数”。

## 根因

bisheng 把 `__ubuf__` 地址空间编码进指针类型，`__ubuf__ T*` 不能隐式转换为普通 `T*` 形参。

## 解决配方

先检查 helper 是否实际使用这些指针：若未使用，删除形参与所有调用实参；若确实使用，应把形参声明为与实参一致的 UB 地址空间类型。本轨迹属于前者。以下为事件流真实写入的完整函数；同一原子编辑还修正了 `PIPE_ALL` 的独立名称查找错误，该根因不并入本卡。

### before

path: `operators/cummin/cummin.asc`；symbols: `KernelCummin<T>::StoreOut`、`KernelCummin<T>::ProcessPathA`；修复前工作区完整函数。

```cpp
    __aicore__ inline void StoreOut(int64_t off, uint32_t cnt, T *vp, int64_t *ip)
    {
        AscendC::DataCopyExtParams cpv{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyExtParams cpi{1, cnt * (uint32_t)sizeof(int64_t), 0, 0, 0};
        AscendC::PipeBarrier<AscendC::PIPE_ALL>();        // S(scalar) writes done -> MTE3
        AscendC::DataCopyPad(valuesGm_[off], valOutBuf_.Get<T>(), cpv);
        AscendC::DataCopyPad(indicesGm_[off], idxOutBuf_.Get<int64_t>(), cpi);
        AscendC::PipeBarrier<AscendC::PIPE_ALL>();        // MTE3(store) done -> reuse buffers
    }

    // ----- Path A: contiguous line (dim is last axis) -----
    __aicore__ inline void ProcessPathA()
    {
        CT accVal;
        int64_t accIdx = 0;
        for (int64_t li = lineStart_; li < lineEnd_; ++li) {
            int64_t base = li * L_;          // contiguous line start (element offset)
            bool inited = false;
            int64_t remaining = L_;
            int64_t off = 0;
            while (remaining > 0) {
                uint32_t cnt = static_cast<uint32_t>(remaining < MAX_TILE ? remaining : MAX_TILE);
                AscendC::DataCopyExtParams cp{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(stepBuf_.Get<T>(), xGm_[base + off], cp, pad);
                AscendC::PipeBarrier<AscendC::PIPE_ALL>();   // MTE2(load) done -> S(scalar)

                __ubuf__ T *sp = (__ubuf__ T *)stepBuf_.Get<T>().GetPhyAddr();
                __ubuf__ T *vp = (__ubuf__ T *)valOutBuf_.Get<T>().GetPhyAddr();
                __ubuf__ int64_t *ip = (__ubuf__ int64_t *)idxOutBuf_.Get<int64_t>().GetPhyAddr();
                for (uint32_t i = 0; i < cnt; ++i) {
                    T raw = sp[i];
                    CT b = ToCompute<T>(raw);
                    if (!inited) {
                        accVal = b; accIdx = off + i; inited = true;
                    } else if (CumminTake<CT>(accVal, b)) {
                        accVal = b; accIdx = off + i;
                    }
                    vp[i] = FromCompute<T>(accVal);
                    ip[i] = accIdx;
                }
                StoreOut(base + off, cnt, vp, ip);
                off += cnt;
                remaining -= cnt;
            }
        }
    }
```

### after

path: `operators/cummin/cummin.asc`；symbols: `KernelCummin<T>::StoreOut`、`KernelCummin<T>::ProcessPathA`；修复后构建成功的完整函数。

```cpp
    __aicore__ inline void StoreOut(int64_t off, uint32_t cnt)
    {
        AscendC::DataCopyExtParams cpv{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyExtParams cpi{1, cnt * (uint32_t)sizeof(int64_t), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_ALL>();                 // S(scalar) writes done -> MTE3
        AscendC::DataCopyPad(valuesGm_[off], valOutBuf_.Get<T>(), cpv);
        AscendC::DataCopyPad(indicesGm_[off], idxOutBuf_.Get<int64_t>(), cpi);
        AscendC::PipeBarrier<PIPE_ALL>();                 // MTE3(store) done -> reuse buffers
    }

    // ----- Path A: contiguous line (dim is last axis) -----
    __aicore__ inline void ProcessPathA()
    {
        CT accVal;
        int64_t accIdx = 0;
        for (int64_t li = lineStart_; li < lineEnd_; ++li) {
            int64_t base = li * L_;          // contiguous line start (element offset)
            bool inited = false;
            int64_t remaining = L_;
            int64_t off = 0;
            while (remaining > 0) {
                uint32_t cnt = static_cast<uint32_t>(remaining < MAX_TILE ? remaining : MAX_TILE);
                AscendC::DataCopyExtParams cp{1, cnt * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(stepBuf_.Get<T>(), xGm_[base + off], cp, pad);
                AscendC::PipeBarrier<PIPE_ALL>();   // MTE2(load) done -> S(scalar)

                __ubuf__ T *sp = (__ubuf__ T *)stepBuf_.Get<T>().GetPhyAddr();
                __ubuf__ T *vp = (__ubuf__ T *)valOutBuf_.Get<T>().GetPhyAddr();
                __ubuf__ int64_t *ip = (__ubuf__ int64_t *)idxOutBuf_.Get<int64_t>().GetPhyAddr();
                for (uint32_t i = 0; i < cnt; ++i) {
                    T raw = sp[i];
                    CT b = ToCompute<T>(raw);
                    if (!inited) {
                        accVal = b; accIdx = off + i; inited = true;
                    } else if (CumminTake<CT>(accVal, b)) {
                        accVal = b; accIdx = off + i;
                    }
                    vp[i] = FromCompute<T>(accVal);
                    ip[i] = accIdx;
                }
                StoreOut(base + off, cnt);
                off += cnt;
                remaining -= cnt;
            }
        }
    }
```

## 设计决策

- 选择删除参数：`StoreOut` 只通过成员 `valOutBuf_`/`idxOutBuf_` 访问输出 UB，`vp`/`ip` 从未参与函数体语义；
- 放弃强行 cast 成普通指针：这会抹掉地址空间信息，既不符合编译器类型系统，也可能产生错误寻址；
- 未采用的候选：把 helper 形参声明为 `__ubuf__ T*`/`__ubuf__ int64_t*`；只有函数确实消费指针时才合理，本例会留下无用接口；
- 当前方案属于 `recommended_fastpath`：删除死参数无运行时成本，也让 API 更难误用。

## 系统信息

- 目标：Ascend950PR_957b / dav-3510；
- 工具链：CANN 9.0.1，bisheng；
- 修改文件：`operators/cummin/cummin.asc`；
- 受影响实例化：float、int32、half、bfloat16_t。

## 预防

- 把 `__gm__`、`__ubuf__` 等地址空间限定视为类型的一部分；
- device helper 设计时先确认参数是否真实使用，死参数直接删除；
- 必须传指针时，让形参与 `GetPhyAddr()` 转换后的地址空间完全匹配；
- 对多 dtype 模板，优先看第一条 address-space 诊断，不要把后续重复实例化错误当成多个根因。

## 验证信号

本知识来自一次 Ascend C kernel helper 重构后的真实 bisheng 模板实例化失败与同轮修复复验。

- 编译器对 float、int32、half、bfloat16_t 四种实例化均报告普通指针形参与 `__ubuf__` 实参不兼容（950，证据等级: tool_observed）；
- 删除两个未使用形参与三处调用实参后，构建输出 `[100%] Built target op_cummin`（950，证据等级: tool_observed）；
- 后续干净重建再次输出 `[100%] Built target op_cummin`（950，证据等级: tool_observed）。

## 失效边界

- 若 helper 真正需要从传入地址读写 UB，不能删除参数，应保留正确的 `__ubuf__` 形参；
- 本卡不讨论 `__gm__`、L1/L0 等其他地址空间之间的转换规则；
- A2、A3 与其他 CANN 版本未验证，推断同一 bisheng 地址空间类型规则可迁移，但属于“未验证，推断可迁移”。

## 坏实践

- 为消除编译错误，把 `__ubuf__ T*` 强转为普通 `T*`；
- 给所有 helper 复制普通 C++ 指针签名，忽略 device 地址空间；
- 把四种 dtype 的重复报错拆成四个问题，而不先检查共同模板形参。

## 误诊

该错误不是某个 dtype 缺少特化；四种实例化同时失败正说明共同的 helper 签名违反了地址空间类型约束。