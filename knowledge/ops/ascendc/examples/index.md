# 代码示例

可参考的 Ascend C kernel 代码范式、生成模板与实测样例。

## MIX cube+vector 模板与范式

- [Cube-Vector Fusion (MIX) — port_a3 cube-class canonical pattern](cube_vector_fusion.md)：A5/Ascend950PR cube 类算子的原生 MIX 范式（P-P102）：手写 Mmad cube + vector 收尾 + WorkspaceQueue 跨核同步。
- [FlashAttention-class (FA) Template Domain — A5 reproducible-framework KB](fa_class_template.md)：仅限 arch35/A5 的 FA 类可复现框架（P-P103）：约 90% 可复用骨架、功能块清单、host tiling 逻辑与反向 stitch 配方。
- [FlashAttention-class MIX Template Domain (a3 / arch22)](fa_class_a3_mix_template.md)：a3/arch22 专用 FA 类 MIX_AIC_1_2 生成模板（P-P116）：MIX 派发不加任务宏、AIC↔AIV 双 flag 握手、库原语优先。

## Cube 原语与分块递推

- [a3 MIX many-small-matmul cube primitives](a3_mix_small_matmul_cube.md)：a3 上多个 ≤128 小矩阵乘的两条 cube 决策（P-P119/P-P120）：手写单 block cube 原语替代 IterateAll、Mmad cmatrixInitVal 把 (I+P) 折进 L0C。
- [Chunked Gated-Delta-Rule / Gated-Linear-Attention Forward](gated_delta_rule_a3_recurrence.md)：a3 上分块门控线性注意力前向生成范式（P-P117）：每头顺序分块递推、kernel AIV 内逐块 decay-fold、MIX 执行。
- [Chunked Gated-Delta-Rule / Gated-Linear-Attention BACKWARD](gated_delta_rule_bwd_a3_recurrence.md)：a3 上分块 gated-delta-rule 反向三趟结构（P-P118）：并行 PASS A、逆向递推 PASS B、逐块 PASS C 产出六个梯度。

## 多核与跨核握手实测

- [RESULT — a3 multi-core FA via library/KFC path](fa_a3_multicore_result.md)：a3 多核 FA 走 MatmulImpl IterateAll + 每对一次 MODE2 握手的结果：精度 20/20 且与单核逐位一致，性能为 vendor 的 0.186 倍（20 核）。
- [a3 c220 GM-comm-slot LOOPED bidirectional MIX probe — RUN_LOG](fa_a3_multicore_gmslot_run_log.md)：GM 通信槽式双向 MIX 循环握手实测日志（负面结果）：共享广播 flag + GM 分槽仍未逃出 LOOPED 双向死锁。

## GMM 量化融合

- [GMM SwiGLU Quant A8W8 — COOKBOOK](gmm_swiglu_quant_a8w8_class_template.md)：A5/V351 上 GMM+SwiGLU+A8W8 融合算子 cookbook（P-P104）：独立 pybind11 模式须手写 Mmad、flagId 8/9/10 为 V351 保留。
- [GMM SwiGLU Quant A8W8 — Host Tiling Advisory Cookbook](gmm_swiglu_quant_a8w8_host_tiling_template.md)：上条的 host tiling 配套清单（P-P104-HOST）：TilingData 字节契约、ExtractConfig/ComputeTiling 检查项与 pybind11 注册。

## SIMT 与哈希表

- [AscendC SIMT Kernel 编程模式参考](simt_patterns.md)：源自 HierarchicalKV-ascend 已完成 kernel 的 SIMT 编程规范：__simt_vf__ 签名模板、GM_ADDR 转换、线程索引、原子操作与 dispatcher 模板。
- [HKV 哈希表算子专属 Patterns](hkv_patterns.md)：HierarchicalKV 哈希表类算子专属范式（lock/evict/digest/bucket）：两阶段提交锁分离、vcmpeq4 digest 向量化查找、纯查找路径免原子操作。
