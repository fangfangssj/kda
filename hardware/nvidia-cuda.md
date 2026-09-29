# Hardware Profile: `NVIDIA · CUDA C++`

KDA 原生支持的组合：CUDA C++ kernel + KernelWiki 知识 + NCU 归因。

## 1. 目标硬件与工具链

- 器件：NVIDIA GPU，Ampere（A100）起步，重点 Hopper（H100/H200）与 Blackwell（B200/B300）。
- 基线：CUDA Toolkit ≥ 12.x，nvcc target 与器件匹配（sm_80 / sm_90 / sm_100）。
- 环境清单：`nvcc --version`、`nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv`，记录进 `docs/draft.md`。

## 2. 实现语言与 API

- CUDA C++（`.cu/.cuh`），宿主为 PyTorch extension 或 cuda-python driver API。
- CUTLASS / CUB 仅在任务契约允许时使用。
- ptxas flags、`-arch`、宏定义的任何变更都是候选的一部分，必须记录在候选描述里。

## 3. 正确性验证约定

- 参考实现：PyTorch eager 或权威库（cuBLAS / cuDNN / FlashInfer）。
- 判定：`torch.allclose(out, ref, rtol=..., atol=...)`，dtype 默认值见 `nvidia-triton.md` 第 3 节；随机种子固定并记录。
- 边界用例：非对齐 leading dimension、K 不整除 tile、M=1、超大 shape（检查 int64 索引溢出）。
- 内存安全：`compute-sanitizer --tool memcheck`（需要时加 `racecheck`）纳入验证命令，晋升候选至少完整跑一次。

## 4. 性能测量约定

- 计时：CUDA event，≥ 100 次取 median；若可锁频（`nvidia-smi -lgc`）则锁频并记录时钟。
- 基线集合：cuBLAS / FlashInfer / 被替换的现有实现。
- `benchmark.csv`：`candidate,algo_variant,flags,shape,dtype,median_us,baseline_us,speedup,clk,date`。

## 5. Profiling 与归因

- 内核级：`ncu --set full --export profile/<candidate> <cmd>`，报告用 `ncu-report-skill` 解读；重点看 SM Busy、Memory Throughput%、warp stall 原因、achieved occupancy、寄存器与 shared memory 用量。
- 端到端：`nsys profile` 确认瓶颈确实在该 kernel（而不是 launch / H2D / 同步开销）。
- SASS 检查：`cuobjdump -sass` 验证是否生成了目标指令（cp.async / TMA / MMA 变体），以及 bank conflict 相关指令密度。

## 6. 技能映射

| 需求 | 技能 | 位置 |
|---|---|---|
| tiling / pipelining / swizzle / 占用率等内核优化知识 | KernelWiki | `skills/KernelWiki`（git 子模块） |
| NCU 报告解读 | ncu-report-skill | `skills/ncu-report-skill`（git 子模块） |

## 7. 自动调优纪律

- 算法变体（split-K、persistent kernel、double buffering、不同 tiling 层级）与配置参数（block/tile 尺寸、num_stages）**分成两组候选**，逐组推进。
- 每个 flag / 宏 / 模板参数变更记入候选描述；禁止无记录的"顺手改"。

## 8. 常见坑

- stream 未同步导致验证"偶然通过"；验证代码必须 `cudaDeviceSynchronize` 后再比较。
- shared memory 超限只在特定 tile 组合出现——配置扫描要带编译期（`static_assert`）或启动期断言。
- Hopper 与 Blackwell 的最优策略不同（TMA、num_stages 行为差异），跨代结论必须重测。
