# 基础数学直调算子

本目录保存基础数学与选择类 Direct Invoke 算子知识。

- [ArgMax 950](arg_max_950.md)：说明 950 最大值索引归约、布局专用与跨核路线。
- [Cummin 950](cummin_950.md)：说明 950 累积最小值的 AIV、SIMT 与多 Kernel 扫描路线。
- [Exp 950](exp_950.md)：说明 950 指数计算的 dtype 和向量流水特化路线。
- [Gcd 950](gcd_950.md)：说明 950 最大公约数的 SIMT 与广播 fallback 路线。
- [MaskedScale 950](masked_scale_950.md)：待复核草稿；整理掩码缩放路线，FP16 补充实现来源尚未闭合。
- [Maximum 950](maximum_950.md)：说明 950 等形和多类广播 Maximum 路线。
- [TopK 950](top_k_950.md)：说明 950 TopK 的 radix、merge、SIMT、NDDMA 与 workspace 路线。
