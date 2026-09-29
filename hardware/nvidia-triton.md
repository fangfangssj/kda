# Hardware Profile: `NVIDIA · Triton`

Triton kernel（OpenAI Triton，NVIDIA 后端）。配套 starter prompt：`prompts/triton-flow.md`。

## 1. 目标硬件与工具链

- 器件：NVIDIA GPU，Ampere 起步，重点 Hopper / Blackwell。
- 基线：PyTorch ≥ 2.4，Triton ≥ 3.x（或 PyTorch 自带版本）。
- 环境清单：`python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count)"`，记录进 `docs/draft.md`。

## 2. 实现语言与 API

- Triton（`triton.language as tl`）+ Python 宿主 + PyTorch tensor；`@triton.jit` / `@triton.autotune` / `@triton.heuristics`。
- 契约未特批时禁止：手写 PTX 内联、引入外部编译产物、用 CUDA 源码替代 Triton 实现。

## 3. 正确性验证约定

- 参考实现：PyTorch eager / 权威算子（SDPA、cuBLAS 路径）。
- 判定：`torch.allclose(out, ref, rtol=..., atol=...)`，dtype 相关默认（可在契约中收紧）：
  - fp32：`rtol=1e-5, atol=1e-6`
  - fp16：`rtol=1e-3, atol=1e-3`
  - bf16：`rtol=1.6e-2, atol=1e-2`
- 形状覆盖：非 2 幂维、奇数维（触发 mask 分支）、M=1 / K=1 边界、大 shape。
- 验证命令模板：`python validate.py --dtype bf16 --shapes small,mid,odd`（由任务工作区实现）。

## 4. 性能测量约定

- 计时统一用 `triton.testing.do_bench`（自带 L2 清理），报告 median，并记录 warmup/rep 配置。
- 基线集合：torch eager、（如适用）`torch.compile` / SDPA / cuBLAS。
- `benchmark.csv`：`candidate,algo_variant,config,shape,dtype,median_us,baseline_us,speedup,date`。
- JIT 编译墙钟时间**单独记录**，不计入 do_bench 结果。

## 5. Profiling 与归因

- 内核级：`ncu --set full --export profile/<candidate> <cmd>`，报告用 `ncu-report-skill` 解读。
- 中间表示：设 `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump/`（或直接读 Triton cache），检查 `.ttir/.ttgir/.ptx`——向量化宽度、shared memory 用量、是否出现意外序列化。
- 端到端：`nsys profile` 确认瓶颈归属。

## 6. 技能映射

| 需求 | 技能 | 位置 |
|---|---|---|
| tiling / coalescing / occupancy / pipelining 概念 | KernelWiki（CUDA 语境，概念迁移见下） | `skills/KernelWiki` |
| NCU 报告解读 | ncu-report-skill | `skills/ncu-report-skill` |

KernelWiki 概念 → Triton 手柄对应：

| CUDA 概念 | Triton 中的对应做法 |
|---|---|
| memory coalescing | 连续 `tl.load` + 合理 BLOCK 划分 |
| bank conflict | 调整 index 计算方式（等价于隐式 padding） |
| shared memory swizzle | 修改指针排布 / `tl.reshape`+`tl.trans` 组合 |
| cp.async 软件流水 | `num_stages` 参数 |
| warp 级原语 | `tl.reduce` / `tl.scan`（能力受限，注意） |

## 7. 自动调优纪律

1. **先算法变体，后配置扫描**：split-K、persistent、不同 tiling 层级是"算法候选"；`BLOCK_SIZE/num_warps/num_stages` 是"配置候选"。候选必须标注属于哪组。
2. `@triton.autotune` 的 `key` 必须覆盖影响最优配置的 shape 语义（如 M/N/K 分桶），防止跨 shape 污染缓存的最优配置。
3. 配置扫描必须报告搜索空间大小与每个配置的耗时，防止"用搜索时间掩盖无改进"。
4. 注意 constexpr 特化爆炸：每个新 (shape 桶， config) 组合都触发重编译。

## 8. 常见坑

- mask 边界差一（`offs < n` 与 `offs <= n`）。
- `do_bench` 的 L2 清理行为与真实 workload 不符时，结论会偏向利好小 kernel。
- bf16 累加：确认 accumulator 是 fp32（`acc.to(tl.float32)` 再累加）。
- Hopper → Blackwell 的 `num_stages` / TMA 行为差异：历史最优 config 不可直接沿用，必须重测。
