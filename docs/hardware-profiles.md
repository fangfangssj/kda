# 硬件档案（Hardware Profiles）

KDA 的最小循环（契约 → 草稿 → 计划 → 候选 → 验证 → 测量 → 证据 → 晋升）本身与硬件无关。真正绑定硬件的只有三件事：**用什么语言写 kernel、怎么验证正确性、怎么测量与归因性能**。硬件档案就是把这三件事参数化：**一份 profile = 一个硬件组合 + 一套语言 / 验证 / 测量 / 技能的约定**。

## 规则

1. 每个任务契约必须声明 `Hardware profile` 字段，指向任务工作区内可读取的 profile；推荐统一使用工作区根目录的 `hardware-profile.md`。
2. 若任务工作区独立于本仓库，启动 agent 前先将本仓库 `hardware/<profile>.md` 复制为任务工作区的 `hardware-profile.md`。自定义 profile 也应放在任务工作区内。
3. agent 在写 `docs/draft.md` 之前必须先读完工作区中的 profile，并以其中的命令约定、精度默认值、纪律条款为默认值；任务契约可以覆盖 profile（任务契约 > profile）。
4. 任务可以在自己的工作区维护私有 profile 覆盖内置档案，不必回传本仓库。
5. profile 只写**机制**（工具、命令、纪律、技能映射），不写具体任务的私有阈值或数据集。

例如，进入独立任务工作区后，将路径替换为本地 KDA 仓库位置：

```bash
cp /path/to/kda/hardware/ascend-ascendc.md /path/to/task-workspace/hardware-profile.md
```

## 内置档案

| 档案 | 硬件 | 实现语言 | 正确性验证 | 测量与归因 | 知识技能 |
|---|---|---|---|---|---|
| `nvidia-cuda.md` | NVIDIA A100/H100/B200… | CUDA C++ | torch 参考 + compute-sanitizer | NCU + nsys | KernelWiki, ncu-report-skill |
| `nvidia-triton.md` | 同上 | OpenAI Triton | torch 参考 + allclose | do_bench + NCU + TTGIR/PTX | KernelWiki（概念迁移）， ncu-report-skill |
| `ascend-ascendc.md` | Atlas A2/A3（910B/910C） | Ascend C | torch_npu 参考 + 精度标准 + ST/UT | msprof（ops-profiling） | npu-arch, ascendc-* |
| `ascend-triton.md` | 同上 | triton_ascend | torch_npu 参考 + triton-op-verifier | ops-profiling compare | triton-*, npu-arch |

配套 starter prompt：CUDA 用 `prompts/basic-flow.md`，Triton（两端）用 `prompts/triton-flow.md`，AscendC 用 `prompts/ascendc-flow.md`。

## 新增一个硬件（扩展指南）

1. 复制 `hardware/TEMPLATE.md` 为 `hardware/<vendor>-<device>.md`。
2. 逐节填写：工具链与版本基线、正确性验证约定（参考实现 + 精度默认值 + 边界形状）、性能测量约定（计时工具 + 统计口径 + csv 行格式）、profiling 采集与报告解读、技能映射、自动调优纪律、常见坑。
3. 若该硬件有现成的知识技能（例如 AMD ROCm 的 rocprof 解读技能、寒武纪 MLU 知识库），把技能目录放进 `skills/`（昇腾系放 `skills/ascend/`）并在上表登记。
4. 若需要专用 starter prompt，在 `prompts/` 新增并在 `prompts/README.md` 登记；Triton 系硬件可直接复用 `prompts/triton-flow.md`。

示例：AMD MI300 上跑 Triton → 复制模板为 `amd-triton.md` → 语言填 triton(ROCm 后端) → 验证填 torch ROCm 参考 + allclose → 测量填 do_bench(ROCm) + rocprof 归因 → 技能放自建的 rocprof-report-skill → 在上表登记。机制不变，只换档案。
