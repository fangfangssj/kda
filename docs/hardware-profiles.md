# 昇腾硬件档案

本仓库专注 Ascend NPU 开发。硬件档案记录目标昇腾芯片与 NpuArch、CANN 工具链、Ascend C 实现约定、正确性验证、性能测量和技能映射，供不同昇腾型号与开发任务复用。

## 规则

1. 每个任务契约必须声明 `Ascend hardware profile`，推荐将档案复制到任务工作区根目录并命名为 `hardware-profile.md`。
2. 启动 agent 前，将 `hardware/ascend-ascendc.md` 复制到任务工作区；同时在任务契约中注明 SoC、NpuArch、CANN 与 torch_npu 版本。
3. agent 在写 `docs/draft.md` 之前必须先读完 profile，并按其中的命令约定、精度标准和验证流程执行；任务契约可以进一步收紧要求。
4. 需要覆盖其他昇腾芯片代际或工具链差异时，可在任务工作区维护档案副本；可复用的代际约定再回填到本仓库。
5. profile 只写可复用的昇腾开发机制，不写具体任务的私有阈值或数据集。

例如，进入独立任务工作区后，将路径替换为本地 KDA 仓库位置：

```bash
cp /path/to/kda/hardware/ascend-ascendc.md /path/to/task-workspace/hardware-profile.md
```

## 内置档案

| 档案 | 硬件 | 实现语言 | 正确性验证 | 测量与归因 | 知识技能 |
|---|---|---|---|---|---|
| `ascend-ascendc.md` | Atlas A2/A3（910B/910C） | Ascend C | torch_npu 参考 + 精度标准 + ST/UT | msprof（ops-profiling） | npu-arch, ascendc-* |

配套 starter prompt：Ascend C kernel 任务使用 `prompts/ascendc-flow.md`；其他 Ascend 开发任务可用 `prompts/basic-flow.md`。

## 维护昇腾芯片档案

1. 需要单独记录新的 Ascend SoC、NpuArch 代际或 CANN 工具链差异时，复制 `hardware/TEMPLATE.md` 并使用 `ascend-<family>.md` 命名。
2. 填写芯片与工具链基线、Ascend C API/tiling 约定、正确性验证、上板测量与 profiling 方法、技能映射和常见坑。
3. 更新本文件的档案表，并确认对应技能存在于 `skills/ascend/`。
4. 若该型号需要不同的任务流程，再新增 Ascend 专用 starter prompt 并在 `prompts/README.md` 登记。

示例：新增一个 Atlas 芯片代际档案 → 复制模板为 `ascend-atlas-<family>.md` → 填写 SoC/NpuArch、CANN 版本、验证命令、msprof 采集方法与代际特有注意事项 → 在上表登记。
