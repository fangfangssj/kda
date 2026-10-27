---
type: Runbook
title: int8 连续输出的四字节打包、对齐门控与标量回退
description: 从普通 Gather 实现提炼四字节输出打包的完整门控，区分读端宽度和写端合并，并保留非对齐及短尾的逐元素写回。
tags:
- task.implementation
- topic.data_movement
- topic.alignment
status: draft
sources: []
created_at: '2026-10-17T18:12:04Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# int8 连续输出的四字节打包、对齐门控与标量回退

## 适用场景

输出值为一字节位模式、写入位置连续且每段由单个写者拥有；输入仍按真实元素宽度读取。这里仅记录 int8 路线，不把源注释提到的 fp16/bf16 扩成已实现模式。

## 门控与核心实现

源实现同时检查段起点 `E0` 为 4 的倍数、输出基址按 4 字节对齐。每个 16 项主块将四个一字节值提升为 `uint32_t`，再按 `v0 | v1<<8 | v2<<16 | v3<<24` 组成一字。组内不是把四次窄数据读改为一次宽读；地址可以完全不连续。

下列为原 INNER 宏中完整的“对齐 / 非对齐 / 标量尾部”实现单元，删除了未经本轮独立测量的注释。参数合同为 `out[E0+k] = x[idx[E0+k]*XMS+B+k]`，调用者需保证两侧范围及乘加地址合法。

```cpp
#define GATHER_SEG16_INNER(T, IndexT, xGm, idxGm, outGm, E0, LEN, XMS, B) \
do { \
    int64_t _k = 0; \
    if (sizeof(T) == 1) { \
         \
        const __gm__ uint8_t *_xb = (const __gm__ uint8_t *)(const __gm__ uint8_t *)(xGm); \
        __gm__ uint8_t *_ob = (__gm__ uint8_t *)(__gm__ uint8_t *)(outGm); \
        if ((((E0) & 3) == 0) && ((((uintptr_t)(outGm)) & 3) == 0)) { \
            for (; _k + 15 < (LEN); _k += 16) { \
                int64_t _e = (E0) + _k; \
                int64_t _b = (B) + _k; \
                int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b+0,   o1=(int64_t)idxGm[_e+1]*(XMS)+_b+1,   o2=(int64_t)idxGm[_e+2]*(XMS)+_b+2,   o3=(int64_t)idxGm[_e+3]*(XMS)+_b+3, \
                        o4=(int64_t)idxGm[_e+4]*(XMS)+_b+4,   o5=(int64_t)idxGm[_e+5]*(XMS)+_b+5,   o6=(int64_t)idxGm[_e+6]*(XMS)+_b+6,   o7=(int64_t)idxGm[_e+7]*(XMS)+_b+7, \
                        o8=(int64_t)idxGm[_e+8]*(XMS)+_b+8,   o9=(int64_t)idxGm[_e+9]*(XMS)+_b+9,   o10=(int64_t)idxGm[_e+10]*(XMS)+_b+10, o11=(int64_t)idxGm[_e+11]*(XMS)+_b+11, \
                        o12=(int64_t)idxGm[_e+12]*(XMS)+_b+12, o13=(int64_t)idxGm[_e+13]*(XMS)+_b+13, o14=(int64_t)idxGm[_e+14]*(XMS)+_b+14, o15=(int64_t)idxGm[_e+15]*(XMS)+_b+15; \
                uint32_t v0=_xb[o0], v1=_xb[o1], v2=_xb[o2], v3=_xb[o3], \
                         v4=_xb[o4], v5=_xb[o5], v6=_xb[o6], v7=_xb[o7], \
                         v8=_xb[o8], v9=_xb[o9], v10=_xb[o10], v11=_xb[o11], \
                         v12=_xb[o12], v13=_xb[o13], v14=_xb[o14], v15=_xb[o15]; \
                __gm__ uint32_t *_ow = (__gm__ uint32_t *)(_ob + _e); \
                _ow[0] = v0 | (v1<<8) | (v2<<16) | (v3<<24); \
                _ow[1] = v4 | (v5<<8) | (v6<<16) | (v7<<24); \
                _ow[2] = v8 | (v9<<8) | (v10<<16) | (v11<<24); \
                _ow[3] = v12 | (v13<<8) | (v14<<16) | (v15<<24); \
            } \
        } else { \
            for (; _k + 15 < (LEN); _k += 16) { \
                int64_t _e = (E0) + _k; \
                int64_t _b = (B) + _k; \
                int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b+0,   o1=(int64_t)idxGm[_e+1]*(XMS)+_b+1,   o2=(int64_t)idxGm[_e+2]*(XMS)+_b+2,   o3=(int64_t)idxGm[_e+3]*(XMS)+_b+3, \
                        o4=(int64_t)idxGm[_e+4]*(XMS)+_b+4,   o5=(int64_t)idxGm[_e+5]*(XMS)+_b+5,   o6=(int64_t)idxGm[_e+6]*(XMS)+_b+6,   o7=(int64_t)idxGm[_e+7]*(XMS)+_b+7, \
                        o8=(int64_t)idxGm[_e+8]*(XMS)+_b+8,   o9=(int64_t)idxGm[_e+9]*(XMS)+_b+9,   o10=(int64_t)idxGm[_e+10]*(XMS)+_b+10, o11=(int64_t)idxGm[_e+11]*(XMS)+_b+11, \
                        o12=(int64_t)idxGm[_e+12]*(XMS)+_b+12, o13=(int64_t)idxGm[_e+13]*(XMS)+_b+13, o14=(int64_t)idxGm[_e+14]*(XMS)+_b+14, o15=(int64_t)idxGm[_e+15]*(XMS)+_b+15; \
                T v0=xGm[o0], v1=xGm[o1], v2=xGm[o2], v3=xGm[o3], \
                  v4=xGm[o4], v5=xGm[o5], v6=xGm[o6], v7=xGm[o7], \
                  v8=xGm[o8], v9=xGm[o9], v10=xGm[o10], v11=xGm[o11], \
                  v12=xGm[o12], v13=xGm[o13], v14=xGm[o14], v15=xGm[o15]; \
                outGm[_e+0]=v0;  outGm[_e+1]=v1;  outGm[_e+2]=v2;  outGm[_e+3]=v3; \
                outGm[_e+4]=v4;  outGm[_e+5]=v5;  outGm[_e+6]=v6;  outGm[_e+7]=v7; \
                outGm[_e+8]=v8;  outGm[_e+9]=v9;  outGm[_e+10]=v10; outGm[_e+11]=v11; \
                outGm[_e+12]=v12; outGm[_e+13]=v13; outGm[_e+14]=v14; outGm[_e+15]=v15; \
            } \
        } \
    } else { \
        for (; _k + 15 < (LEN); _k += 16) { \
            int64_t _e = (E0) + _k; \
            int64_t _b = (B) + _k; \
            int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b+0,   o1=(int64_t)idxGm[_e+1]*(XMS)+_b+1,   o2=(int64_t)idxGm[_e+2]*(XMS)+_b+2,   o3=(int64_t)idxGm[_e+3]*(XMS)+_b+3, \
                    o4=(int64_t)idxGm[_e+4]*(XMS)+_b+4,   o5=(int64_t)idxGm[_e+5]*(XMS)+_b+5,   o6=(int64_t)idxGm[_e+6]*(XMS)+_b+6,   o7=(int64_t)idxGm[_e+7]*(XMS)+_b+7, \
                    o8=(int64_t)idxGm[_e+8]*(XMS)+_b+8,   o9=(int64_t)idxGm[_e+9]*(XMS)+_b+9,   o10=(int64_t)idxGm[_e+10]*(XMS)+_b+10, o11=(int64_t)idxGm[_e+11]*(XMS)+_b+11, \
                    o12=(int64_t)idxGm[_e+12]*(XMS)+_b+12, o13=(int64_t)idxGm[_e+13]*(XMS)+_b+13, o14=(int64_t)idxGm[_e+14]*(XMS)+_b+14, o15=(int64_t)idxGm[_e+15]*(XMS)+_b+15; \
            T v0=xGm[o0], v1=xGm[o1], v2=xGm[o2], v3=xGm[o3], \
              v4=xGm[o4], v5=xGm[o5], v6=xGm[o6], v7=xGm[o7], \
              v8=xGm[o8], v9=xGm[o9], v10=xGm[o10], v11=xGm[o11], \
              v12=xGm[o12], v13=xGm[o13], v14=xGm[o14], v15=xGm[o15]; \
            outGm[_e+0]=v0;  outGm[_e+1]=v1;  outGm[_e+2]=v2;  outGm[_e+3]=v3; \
            outGm[_e+4]=v4;  outGm[_e+5]=v5;  outGm[_e+6]=v6;  outGm[_e+7]=v7; \
            outGm[_e+8]=v8;  outGm[_e+9]=v9;  outGm[_e+10]=v10; outGm[_e+11]=v11; \
            outGm[_e+12]=v12; outGm[_e+13]=v13; outGm[_e+14]=v14; outGm[_e+15]=v15; \
        } \
    } \
    for (; _k < (LEN); _k++) { \
        int64_t _e = (E0) + _k; \
        outGm[_e] = xGm[(int64_t)idxGm[_e]*(XMS) + (B) + _k]; \
    } \
} while(0)
```

## 边界与存储语义

4B 条件约束的是当前输出字地址；不能只核基址而忽略分段偏移。主块长度必须足以容纳全部 16 项，剩余不足 16 的元素仍逐项写回。打包不得跨张量尾部，也不得覆盖相邻核拥有的字节。输入/输出重叠、非连续输出、其他数据位宽或目标字节顺序不匹配时，保留窄写回并重新证明合同。

该模式按位保存 int8 数据，不做数值舍入、饱和或量化。推广到半精度需要另行核对位模式、对齐、别名与异常值，而不是直接替换移位常数。

## 验证信号

来源为普通 Gather 导出实现 `csrc/ops/gather/op_kernel/gather_kernel.cpp`，文件 SHA-256 为 `9407bba207759be15226694f5d6ce612d3163391ce291cac57c6015d6c936577`。本卡记录可直接审阅的源码结构。公开材料见[普通算子源码与单条评测证据](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/ordinary_adamw_gather_evidence_20261020.zip)（ZIP SHA-256 `44c0ccfe24752744aef8854f9a7b7dce191e069539342b41e0b0c4979aac054e`）；包内根目录为 `ordinary_optimization/`。公开附件提供证据入口，技术证据是包内下列文件。

`gather_source_excerpt.txt` 保留 `GATHER_SEG16_INNER`（原 L85–155）、`GATHER_SEG16_FLAT`（L161–227）、`GATHER_IDX16`（L240–252）和 `GATHER_SEG16_INNER_PF`（L266–351）的完整代码单位，只删除非许可注释，并非可独立编译工程；原行号、完整源文件／原范围／公开节录哈希见 `gather_source_binding.json` 的 `/original_source`、`/published_excerpt/ranges`。单条脱敏历史记录为 `gather_historical_record.json`：`/benchmark` 标识 official-tasks 1.1.0、A3（910c），`/result` 为 compile_pass=true、80/80，`/artifact_binding` 绑定原 report、选中原记录、原源包和源码 SHA-256。恢复 receipt 与该原记录的同提交关联经过核对，未重新打开原包。

上述历史数值仅属于原导出完整 artifact，不能据此认定本卡分支各自得到覆盖，也不认证实际加载二进制、CANN 版本、单机制因果或性能收益。本轮没有重新编译或运行；`platforms: []` 保持为空。

可从源宏确认双重 4B gate、四字节拼接、逐元素非对齐分支和标量尾部。源码没有提供本轮独立的窄写回/宽写回配对计时，注释中的槽位数或性能百分比不作为本卡事实。采用时逐字节比较输出，覆盖起点模 4 的全部余数、零长度和短尾，并单独检查多核分段边界。

相关的[独立 load 展开](../performance/independent_index_load_mlp.md)解释地址组，不替代这里的写端对齐合同。
