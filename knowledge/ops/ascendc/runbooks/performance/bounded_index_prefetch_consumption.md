---
type: Runbook
title: 索引预取同时保留消费关系与下一批边界
description: 预加载的索引值应进入真实地址计算；首批、下一批和标量尾部各有边界，源码可证消费关系，缓存命中与性能需另证。
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

# 索引预取同时保留消费关系与下一批边界

## 适用场景

普通 Gather 中，索引以连续 16 项为一批，数据读取独立于下一批索引加载。这里只讨论真实有效索引的提前读取，不利用越界地址或丢弃的 volatile 读充当预取。

## 首批、当前批与下一批

`GATHER_SEG16_INNER_PF` 按输出类型和写回对齐分为三条路径：`sizeof(T)==1` 时，只有 `(E0 & 3)==0` 且 `outGm` 基址按 4 字节对齐，才选择 `uint32_t` 打包写回，否则使用逐元素写回；`sizeof(T)!=1`（如 float/half）使用普通 T 写回。三条路径都在 `LEN > 15` 时预读首批的第 0、15 项，以及 64 位索引时的第 8 项；上述对齐 gate 选择写回方式，不禁止非对齐 int8 路径的索引预取。`GATHER_IDX16` 将这些值分别赋给当前批的 `c0/c15/c8`，参与本批实际数据地址；只有 `_k+31 < LEN` 才加载下一批边界索引。循环后不足一批的部分使用原始索引逐项计算，不执行索引预取宏。

下面保留两个完整宏以包含初始化、值消费、循环和尾部，删除了源注释中的性能及缓存断言。`GATHER_IDX16` 的声明需要在调用者块作用域内可见，不能随意套一层 `do-while` 改变 `c0..c15` 的作用域。

```cpp
#define GATHER_IDX16(IndexT, idxGm, _e, _k, LEN) \
    IndexT c0 = _pf0, c15 = _pf15; \
    IndexT c1 = idxGm[(_e)+1],  c2 = idxGm[(_e)+2],  c3 = idxGm[(_e)+3], \
           c4 = idxGm[(_e)+4],  c5 = idxGm[(_e)+5],  c6 = idxGm[(_e)+6], \
           c7 = idxGm[(_e)+7]; \
    IndexT c8; \
    if (sizeof(IndexT) == 8) { c8 = _pfM; } else { c8 = idxGm[(_e)+8]; } \
    IndexT c9 = idxGm[(_e)+9],  c10 = idxGm[(_e)+10], c11 = idxGm[(_e)+11], \
           c12 = idxGm[(_e)+12], c13 = idxGm[(_e)+13], c14 = idxGm[(_e)+14]; \
    if ((_k) + 31 < (LEN)) { \
        _pf0 = idxGm[(_e)+16]; _pf15 = idxGm[(_e)+31]; \
        if (sizeof(IndexT) == 8) { _pfM = idxGm[(_e)+24]; } \
    }
```

```cpp
#define GATHER_SEG16_INNER_PF(T, IndexT, xGm, idxGm, outGm, E0, LEN, XMS, B) \
do { \
    int64_t _k = 0; \
    if (sizeof(T) == 1) { \
         \
        const __gm__ uint8_t *_xb = (const __gm__ uint8_t *)(const __gm__ uint8_t *)(xGm); \
        __gm__ uint8_t *_ob = (__gm__ uint8_t *)(__gm__ uint8_t *)(outGm); \
        if ((((E0) & 3) == 0) && ((((uintptr_t)(outGm)) & 3) == 0)) { \
            if (15 < (LEN)) { \
                IndexT _pf0 = idxGm[(E0)+0], _pf15 = idxGm[(E0)+15]; \
                IndexT _pfM = 0; if (sizeof(IndexT) == 8) { _pfM = idxGm[(E0)+8]; } \
                for (; _k + 15 < (LEN); _k += 16) { \
                    int64_t _e = (E0) + _k; \
                    int64_t _b = (B) + _k; \
                    GATHER_IDX16(IndexT, idxGm, _e, _k, LEN) \
                    int64_t o0=(int64_t)c0*(XMS)+_b+0,   o1=(int64_t)c1*(XMS)+_b+1,   o2=(int64_t)c2*(XMS)+_b+2,   o3=(int64_t)c3*(XMS)+_b+3, \
                            o4=(int64_t)c4*(XMS)+_b+4,   o5=(int64_t)c5*(XMS)+_b+5,   o6=(int64_t)c6*(XMS)+_b+6,   o7=(int64_t)c7*(XMS)+_b+7, \
                            o8=(int64_t)c8*(XMS)+_b+8,   o9=(int64_t)c9*(XMS)+_b+9,   o10=(int64_t)c10*(XMS)+_b+10, o11=(int64_t)c11*(XMS)+_b+11, \
                            o12=(int64_t)c12*(XMS)+_b+12, o13=(int64_t)c13*(XMS)+_b+13, o14=(int64_t)c14*(XMS)+_b+14, o15=(int64_t)c15*(XMS)+_b+15; \
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
            } \
        } else { \
            if (15 < (LEN)) { \
                IndexT _pf0 = idxGm[(E0)+0], _pf15 = idxGm[(E0)+15]; \
                IndexT _pfM = 0; if (sizeof(IndexT) == 8) { _pfM = idxGm[(E0)+8]; } \
                for (; _k + 15 < (LEN); _k += 16) { \
                    int64_t _e = (E0) + _k; \
                    int64_t _b = (B) + _k; \
                    GATHER_IDX16(IndexT, idxGm, _e, _k, LEN) \
                    int64_t o0=(int64_t)c0*(XMS)+_b+0,   o1=(int64_t)c1*(XMS)+_b+1,   o2=(int64_t)c2*(XMS)+_b+2,   o3=(int64_t)c3*(XMS)+_b+3, \
                            o4=(int64_t)c4*(XMS)+_b+4,   o5=(int64_t)c5*(XMS)+_b+5,   o6=(int64_t)c6*(XMS)+_b+6,   o7=(int64_t)c7*(XMS)+_b+7, \
                            o8=(int64_t)c8*(XMS)+_b+8,   o9=(int64_t)c9*(XMS)+_b+9,   o10=(int64_t)c10*(XMS)+_b+10, o11=(int64_t)c11*(XMS)+_b+11, \
                            o12=(int64_t)c12*(XMS)+_b+12, o13=(int64_t)c13*(XMS)+_b+13, o14=(int64_t)c14*(XMS)+_b+14, o15=(int64_t)c15*(XMS)+_b+15; \
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
        } \
    } else { \
        if (15 < (LEN)) { \
            IndexT _pf0 = idxGm[(E0)+0], _pf15 = idxGm[(E0)+15]; \
            IndexT _pfM = 0; if (sizeof(IndexT) == 8) { _pfM = idxGm[(E0)+8]; } \
            for (; _k + 15 < (LEN); _k += 16) { \
                int64_t _e = (E0) + _k; \
                int64_t _b = (B) + _k; \
                GATHER_IDX16(IndexT, idxGm, _e, _k, LEN) \
                int64_t o0=(int64_t)c0*(XMS)+_b+0,   o1=(int64_t)c1*(XMS)+_b+1,   o2=(int64_t)c2*(XMS)+_b+2,   o3=(int64_t)c3*(XMS)+_b+3, \
                        o4=(int64_t)c4*(XMS)+_b+4,   o5=(int64_t)c5*(XMS)+_b+5,   o6=(int64_t)c6*(XMS)+_b+6,   o7=(int64_t)c7*(XMS)+_b+7, \
                        o8=(int64_t)c8*(XMS)+_b+8,   o9=(int64_t)c9*(XMS)+_b+9,   o10=(int64_t)c10*(XMS)+_b+10, o11=(int64_t)c11*(XMS)+_b+11, \
                        o12=(int64_t)c12*(XMS)+_b+12, o13=(int64_t)c13*(XMS)+_b+13, o14=(int64_t)c14*(XMS)+_b+14, o15=(int64_t)c15*(XMS)+_b+15; \
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
    } \
    for (; _k < (LEN); _k++) { \
        int64_t _e = (E0) + _k; \
        outGm[_e] = xGm[(int64_t)idxGm[_e]*(XMS) + (B) + _k]; \
    } \
} while(0)
```

## 可检查的不变量

首批预读的长度条件是 `LEN > 15`；int8 的 4 字节对齐 gate 只选择打包或逐元素写回，两条 int8 路径和 `sizeof(T)!=1` 路径都有相同的首批初始化及 `GATHER_IDX16` 调用。第 `k` 批中的 `_e=E0+k`，`_pf0/_pf15/_pfM` 对应当前批的真实值；下一批 gate 同时涵盖 `e+16/e+31/e+24`。`LEN=16..31` 时只执行一批，不读取第二批；`LEN<16` 及完整批之后不足 16 项的余数进入公共标量尾部，直接从 `idxGm[_e]` 取索引，不执行首批或下一批预取。调用者仍要保证 `E0+LEN`、数据地址乘加和索引域合法；这些 gate 不等于索引值范围检查。

## 验证信号

来源为普通 Gather 导出实现 `csrc/ops/gather/op_kernel/gather_kernel.cpp`，文件 SHA-256 为 `9407bba207759be15226694f5d6ce612d3163391ce291cac57c6015d6c936577`。本卡记录可直接审阅的源码结构。公开材料见[普通算子源码与单条评测证据](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/ordinary_adamw_gather_evidence_20261020.zip)（ZIP SHA-256 `44c0ccfe24752744aef8854f9a7b7dce191e069539342b41e0b0c4979aac054e`）；包内根目录为 `ordinary_optimization/`。公开附件提供证据入口，技术证据是包内下列文件。

`gather_source_excerpt.txt` 保留 `GATHER_SEG16_INNER`（原 L85–155）、`GATHER_SEG16_FLAT`（L161–227）、`GATHER_IDX16`（L240–252）和 `GATHER_SEG16_INNER_PF`（L266–351）的完整代码单位，只删除非许可注释，并非可独立编译工程；原行号、完整源文件／原范围／公开节录哈希见 `gather_source_binding.json` 的 `/original_source`、`/published_excerpt/ranges`。单条脱敏历史记录为 `gather_historical_record.json`：`/benchmark` 标识 official-tasks 1.1.0、A3（910c），`/result` 为 compile_pass=true、80/80，`/artifact_binding` 绑定原 report、选中原记录、原源包和源码 SHA-256。恢复 receipt 与该原记录的同提交关联经过核对，未重新打开原包。

上述历史数值仅属于原导出完整 artifact，不能据此认定本卡分支各自得到覆盖，也不认证实际加载二进制、CANN 版本、单机制因果或性能收益。本轮没有重新编译或运行；`platforms: []` 保持为空。

以上源码能证明提前读取值有实际消费者，并保留首批和下一批的边界。是否产生更早的机器 load、是否命中缓存或隐藏延迟，尚未通过本轮编译产物和设备时间线核验；不使用源注释中的缓存行数或提速数字。

## 失效边界与采用检查

提前读会延长寄存器存活期，也可能与编译器既有调度重叠。索引会在计算期间被其他写者修改时，提前读取可能改变语义；本模式要求本次调用中的只读索引。采用时检查短段、段尾、32/64 位索引以及分段起点；性能比较要包含完整调用和额外寄存器代价。预取的存在不能代替[写回对齐](../validations/aligned_subword_output_packing.md)或[地址独立性](independent_index_load_mlp.md)合同。
