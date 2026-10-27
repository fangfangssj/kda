---
type: Runbook
title: 索引 Gather 的独立 load 展开与标量尾部
description: 把索引、数据和输出分组展开，显式保留地址独立性与短尾回退；展开宽度是候选参数，不能从源码断言硬件重叠或提速。
tags:
- task.optimization
- topic.data_movement
- topic.memory
status: draft
sources: []
created_at: '2026-10-17T18:12:04Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 索引 Gather 的独立 load 展开与标量尾部

## 适用场景

逐元素 Gather 的输出段中，每个输出可独立计算源地址，索引数组与输出连续，数据源可随机访问。调用者已确定合法索引、源 stride 和每核互斥输出段；不处理带副作用的 load，也不把前一个 load 的结果当作下一索引。

## 模式与地址合同

以最后一轴的 FLAT 变体为例，段起点为 `E0`、长度为 `LEN`，对每个 `k` 写入 `out[E0+k] = x[idx[E0+k]*XMS+B]`。先生成 16 个地址，再读取 16 个数据值，最后逐项存储，使彼此独立的工作在源码中可见。INNER 变体还含 `B+k`，两种地址公式不可互换。

下面是原 `GATHER_SEG16_FLAT` 的完整宏，删除了性能注释，保留可执行语句及 int8 写回分支。其打包条件单独见[窄输出打包](../validations/aligned_subword_output_packing.md)，不把该分支的效果归给 load 展开。

```cpp
#define GATHER_SEG16_FLAT(T, IndexT, xGm, idxGm, outGm, E0, LEN, XMS, B) \
do { \
    int64_t _k = 0; \
    if (sizeof(T) == 1) { \
         \
        const __gm__ uint8_t *_xb = (const __gm__ uint8_t *)(const __gm__ uint8_t *)(xGm); \
        __gm__ uint8_t *_ob = (__gm__ uint8_t *)(__gm__ uint8_t *)(outGm); \
        if ((((E0) & 3) == 0) && ((((uintptr_t)(outGm)) & 3) == 0)) { \
            for (; _k + 15 < (LEN); _k += 16) { \
                int64_t _e = (E0) + _k; \
                int64_t _b = (B); \
                int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b,  o1=(int64_t)idxGm[_e+1]*(XMS)+_b,  o2=(int64_t)idxGm[_e+2]*(XMS)+_b,  o3=(int64_t)idxGm[_e+3]*(XMS)+_b, \
                        o4=(int64_t)idxGm[_e+4]*(XMS)+_b,  o5=(int64_t)idxGm[_e+5]*(XMS)+_b,  o6=(int64_t)idxGm[_e+6]*(XMS)+_b,  o7=(int64_t)idxGm[_e+7]*(XMS)+_b, \
                        o8=(int64_t)idxGm[_e+8]*(XMS)+_b,  o9=(int64_t)idxGm[_e+9]*(XMS)+_b,  o10=(int64_t)idxGm[_e+10]*(XMS)+_b, o11=(int64_t)idxGm[_e+11]*(XMS)+_b, \
                        o12=(int64_t)idxGm[_e+12]*(XMS)+_b, o13=(int64_t)idxGm[_e+13]*(XMS)+_b, o14=(int64_t)idxGm[_e+14]*(XMS)+_b, o15=(int64_t)idxGm[_e+15]*(XMS)+_b; \
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
                int64_t _b = (B); \
                int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b,  o1=(int64_t)idxGm[_e+1]*(XMS)+_b,  o2=(int64_t)idxGm[_e+2]*(XMS)+_b,  o3=(int64_t)idxGm[_e+3]*(XMS)+_b, \
                        o4=(int64_t)idxGm[_e+4]*(XMS)+_b,  o5=(int64_t)idxGm[_e+5]*(XMS)+_b,  o6=(int64_t)idxGm[_e+6]*(XMS)+_b,  o7=(int64_t)idxGm[_e+7]*(XMS)+_b, \
                        o8=(int64_t)idxGm[_e+8]*(XMS)+_b,  o9=(int64_t)idxGm[_e+9]*(XMS)+_b,  o10=(int64_t)idxGm[_e+10]*(XMS)+_b, o11=(int64_t)idxGm[_e+11]*(XMS)+_b, \
                        o12=(int64_t)idxGm[_e+12]*(XMS)+_b, o13=(int64_t)idxGm[_e+13]*(XMS)+_b, o14=(int64_t)idxGm[_e+14]*(XMS)+_b, o15=(int64_t)idxGm[_e+15]*(XMS)+_b; \
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
            int64_t _b = (B); \
            int64_t o0=(int64_t)idxGm[_e+0]*(XMS)+_b,  o1=(int64_t)idxGm[_e+1]*(XMS)+_b,  o2=(int64_t)idxGm[_e+2]*(XMS)+_b,  o3=(int64_t)idxGm[_e+3]*(XMS)+_b, \
                    o4=(int64_t)idxGm[_e+4]*(XMS)+_b,  o5=(int64_t)idxGm[_e+5]*(XMS)+_b,  o6=(int64_t)idxGm[_e+6]*(XMS)+_b,  o7=(int64_t)idxGm[_e+7]*(XMS)+_b, \
                    o8=(int64_t)idxGm[_e+8]*(XMS)+_b,  o9=(int64_t)idxGm[_e+9]*(XMS)+_b,  o10=(int64_t)idxGm[_e+10]*(XMS)+_b, o11=(int64_t)idxGm[_e+11]*(XMS)+_b, \
                    o12=(int64_t)idxGm[_e+12]*(XMS)+_b, o13=(int64_t)idxGm[_e+13]*(XMS)+_b, o14=(int64_t)idxGm[_e+14]*(XMS)+_b, o15=(int64_t)idxGm[_e+15]*(XMS)+_b; \
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
        outGm[_e] = xGm[(int64_t)idxGm[_e]*(XMS) + (B)]; \
    } \
} while(0)
```

调用前须同时保证：`E0 >= 0`、`LEN >= 0`、`[E0,E0+LEN)` 位于索引和输出范围内；`idx*XMS+B` 不发生整数溢出且落在输入合法域；输入与输出重叠不会破坏尚未读取的数据。该宏没有补做索引裁剪或负索引归一化。16 项主块不足时，标量循环按同一公式完成余项；不能为了凑齐一组而越界读取。

## 验证信号

来源为普通 Gather 导出实现 `csrc/ops/gather/op_kernel/gather_kernel.cpp`，文件 SHA-256 为 `9407bba207759be15226694f5d6ce612d3163391ce291cac57c6015d6c936577`。本卡记录可直接审阅的源码结构。公开材料见[普通算子源码与单条评测证据](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/ordinary_adamw_gather_evidence_20261020.zip)（ZIP SHA-256 `44c0ccfe24752744aef8854f9a7b7dce191e069539342b41e0b0c4979aac054e`）；包内根目录为 `ordinary_optimization/`。公开附件提供证据入口，技术证据是包内下列文件。

`gather_source_excerpt.txt` 保留 `GATHER_SEG16_INNER`（原 L85–155）、`GATHER_SEG16_FLAT`（L161–227）、`GATHER_IDX16`（L240–252）和 `GATHER_SEG16_INNER_PF`（L266–351）的完整代码单位，只删除非许可注释，并非可独立编译工程；原行号、完整源文件／原范围／公开节录哈希见 `gather_source_binding.json` 的 `/original_source`、`/published_excerpt/ranges`。单条脱敏历史记录为 `gather_historical_record.json`：`/benchmark` 标识 official-tasks 1.1.0、A3（910c），`/result` 为 compile_pass=true、80/80，`/artifact_binding` 绑定原 report、选中原记录、原源包和源码 SHA-256。恢复 receipt 与该原记录的同提交关联经过核对，未重新打开原包。

上述历史数值仅属于原导出完整 artifact，不能据此认定本卡分支各自得到覆盖，也不认证实际加载二进制、CANN 版本、单机制因果或性能收益。本轮没有重新编译或运行；`platforms: []` 保持为空。

原宏的地址组、值组、输出组和标量尾部可以逐项对照；这是源码观察。编译器是否按期望保留并提前发射 load、寄存器占用是否抵销收益，需要目标编译产物与实际时间线；本卡不据静态展开声称 16 路并发或固定收益。

## 失效边界与采用检查

展开宽度增加可能提高寄存器压力；索引连续不保证数据连续。地址有跨项依赖、非连续索引视图、输出冲突或不同索引语义时不能直接套用。采用时应对照标量基线检查索引边界、重复索引、短段以及分段起点，正确性通过后再比较同条件完整调用。若需要下一组索引提前加载，另读[有界索引预取](bounded_index_prefetch_consumption.md)，不能仅添加一个未消费的读表达式。
