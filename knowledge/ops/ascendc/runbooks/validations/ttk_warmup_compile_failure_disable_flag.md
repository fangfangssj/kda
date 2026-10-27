---
type: "Runbook"
title: "TTK warmup 编译失败须 --warmup false（0 无效）"
description: "TTK kernel 模式启动即抛 RuntimeError: Compile warmup kernel failed（kernel_meta 有 warmup.cce/json 无 warmup.o，编译错误被吞）；关闭开关只认字符串 false，--warmup 0 无效；跳过 warmup 仅用于推进定位，精度结论可信度须按 banned 卡交叉验证。增量（aclnn 复跑形态）：`--no-warmup` 不是合法 flag（argparse 只认 `--warmup=false`），报错文案与 kernel_meta 实际状态不符，勿按文案补文件。"
tags: ["task.testing", "topic.compilation", "topic.implementation.debugging"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T03:26:59Z"
platforms: ["950"]
---

# TTK warmup 编译失败须 --warmup false（0 无效）

## 现象

在 Ascend950PR（CANN 9.2.0）上用 TTK kernel 模式单用例复现精度失败：

```bash
$PY -m ttk kernel -i <cases.csv> -t <失败用例名> --plugin golden_plugin.py \
    --no-prof --warmup 0
```

启动即抛：

```
RuntimeError: Compile warmup kernel failed. kernel or json file does not exist in <kernel_meta>
```

调用栈：`instance_refactor.setup_profile_object → _compile_help_kernels → Opc().compile_warmup_kernel`。`kernel_meta` 里只有 `warmup.cce`/`warmup.json`，没有 `warmup.o`——warmup 辅助核的编译错误被吞掉，只剩"产物文件不存在"的 RuntimeError。本意是传 `--warmup 0` 关掉 warmup，**但没有生效**（仍走了 warmup 编译）。

查看 `warmup.cce` 内嵌的编译命令，目标 arch 是旧的：

```
// ccec -c -O2 .../kernel_meta/warmup.cce --cce-aicore-arch=dav-c310-cube \
//   --cce-aicore-only -o ./kernel_meta/warmup.o ...
```

而本机是 Ascend950PR（dav-3510 系）——warmup 模板按 dav-c310-cube 编译，疑为静默失败原因（未直接验证）。

TTK CLI 源码（`ttk/cli/device.py`）揭示了 `--warmup 0` 失效的原因：

```python
parser.add_argument("--warmup", default=True,
                    type=lambda x: x.lower() != "false",
                    help="Warmup before profiling (default: true)")
```

`--warmup 0` → `"0".lower() != "false"` → `True`（仍开启）；只有字符串 `false` 才能关闭。

## 适用场景

- 工程形态：TTK（ops-test-kit）kernel/aclnn 模式单用例复现与评测。
- API 路径：`ttk kernel`/`ttk aclnn` CLI（`--warmup` 参数、`ttk/cli/device.py` argparse）；TTK warmup 辅助核（`kernel_meta/warmup.cce`、`tbe_interface.compile_warmup_kernel`）。
- 验证过的平台与 CANN 版本：Ascend950PR（本机 SOC 串 Ascend950PR_9579），CANN 9.2.0；TTK 为仓库检出版本（未 pip 安装，`PYTHONPATH=$TTK` 运行）。
- 适用粒度：workflow_only（TTK 工具链使用方法，与算子无关）。
- 路线优先级：safety_fallback——`--warmup false` 只用于"跳过编译失败的 warmup、推进到用例自身编译"的定位场景。fallback 声明：warmup 关闭后跑批的精度结论不可信，同批成对实测表明 `--warmup false` 全量跑批会把真实 PASS 用例变成 `RTS_BINARY_FAILURE` 假失败（见 [TTK --warmup false 批量假失败卡](ttk_warmup_false_rts_binary_failure.md)，banned）；需要精度结论时保持默认 warmup 并按该卡用子集 `--pc 1` 重跑，性能评测更不可用此规避。
- 明确不适用 / 未验证场景：需要 warmup 的性能评测流程；其他 TTK 版本（argparse lambda 与 arch 固定值均为当时版本行为）。

## 判别方法

两件事叠加，分别判别：

1. **warmup 编译失败**：报错形态固定为 `RuntimeError: Compile warmup kernel failed. kernel or json file does not exist in <kernel_meta>`，且 `kernel_meta` 有 `warmup.cce`/`warmup.json` 无 `warmup.o`；可 `cat kernel_meta/warmup.cce` 查看尾部内嵌 ccec 命令的 `--cce-aicore-arch=` 值，与本机 SOC 对照（本例 dav-c310-cube vs Ascend950PR/dav-3510）。
2. **关闭参数不生效**：`--warmup` 的 argparse 类型是 `lambda x: x.lower() != "false"`——任何非 `"false"` 字符串（含 `0`、`no`、`off`）都等于开启；**必须传 `--warmup false`**。

## 解决配方

```bash
export PYTHONPATH=$TTK:$PYTHONPATH      # TTK 检出版本未安装时
$PY -m ttk kernel -i <cases.csv> -t <case> --plugin golden_plugin.py \
    --no-prof --warmup false            # 注意：--warmup 0 无效
```

另外：旧命令 `ttk op ...` 在当前版本已不存在（`invalid choice: 'op' (choose from 'kernel', 'aclnn', 'e2e', 'info', 'list')`），等价子命令是 `ttk kernel`。

## 验证信号

问题与验证来自同一次 TTK kernel 模式单用例复现会话（Ascend950PR，CANN 9.2.0）：

- `--warmup 0`：仍抛 `Compile warmup kernel failed. kernel or json file does not exist`（来源平台：950；证据等级：tool_observed）。
- `--warmup false`：不再触发 warmup 编译，流程推进到用例自身的 DYN 编译阶段（随后暴露的是另一个独立问题：部署 vendor 缺 op_host/tiling_data.h，另卡描述）（来源平台：950；证据等级：tool_observed）。
- `warmup.cce` 内嵌 `--cce-aicore-arch=dav-c310-cube` 与 `device.py` 的 argparse lambda 均为源码/产物文件直接读取（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `--warmup false` 只是跳过 warmup，不修复 warmup 编译本身；warmup 关闭下的精度/通过率结论可信度未在本轨迹单独验证，同批 banned 卡的成对实测表明不可信，须交叉验证后再下结论。
- arch 不匹配→编译失败是推断（文件内容 + 现象吻合），未用 dav-3510 参数手工重编译 warmup.cce 验证。
- TTK 版本差异：`ttk op` 移除、argparse lambda、warmup 模板 arch 均为当时版本行为，升级后需重核（未验证，推断可迁移的是排查方法而非具体取值）。

**未决项**：

- **优先项**：复现并追根因——用本机正确 arch 参数手工编译 `warmup.cce` 验证 arch 假设；或查 TTK warmup 模板的 arch 选择逻辑为何固定 dav-c310-cube（Ascend950PR 支持缺失）。
- warmup 编译的原始 stderr 被吞（编译失败只留"文件不存在"），可在 TTK 侧提高日志级别复现拿原始报错。
## 坏实践

- 依赖 `--warmup 0` 的常规布尔语义关闭 warmup——argparse 自定义 type 不按布尔解析。
- warmup 编译失败被当成 kernel_meta 状态问题去清理产物——`compile_warmup_kernel` 每次会删旧 `warmup.o` 重编，清理无济于事；问题在 warmup 模板本身的编译配置。
- 在 `--warmup false` 下直接采信跑批通过率作精度结论（见 banned 卡）。

## 误诊

- 曾以为 `--warmup 0` 已关闭 warmup（数值 0 的常规布尔语义）——实际仍开启并卡在 warmup 编译失败上。
- warmup 编译失败最初被当成 kernel_meta 状态问题去清理产物，方向错位。

## 增量（aclnn 复跑形态与 --no-warmup 误试）

另一独立 trace（euclidean_norm 黑盒失败用例修复会话，Ascend950PR + CANN 9.1.0）在 **ttk aclnn 模式**复现同一 warmup 编译失败，并补两条边界：

**触发**：为复现 5 个失败用例，从全量 CSV 抽出子集后运行 `python3 -m ttk aclnn -i repro.csv --pc=1 --device-whitelist 0 --plugin golden_plugin.py -o repro_result.csv --dump-on-fail`，TTK 在用例解析完成后、执行前抛 `RuntimeError: Compile warmup kernel failed. kernel or json file does not exist in <TTK 根>/kernel_meta`（栈至 `instance_refactor.py:83 setup_profile_object → _compile_help_kernels → tbe_interface.py:379 compile_warmup_kernel`）。而同一环境此前的全量黑盒正常完成；检查 `kernel_meta/`：`warmup.cce`(798B) 与 `warmup.json`(400B) 都在、无编译产物（.o）——`compile_warmup_kernel` 每次启动先删旧产物再重编，本次重编失败。

**误试（CLI 不识别）**：直觉的 `--no-warmup` 被拒：

```bash
... --no-warmup
# → ttk: error: unrecognized arguments: --no-warmup
```

正确形式是 `--warmup=false`（与主卡 `--warmup false` 等价；`ttk/cli/device.py` 的 argparse `type=lambda x: x.lower() != "false"` 只接受字符串 `false`，`bridge.py` 消费 `args.warmup`）：

```bash
timeout 300 python3 -m ttk aclnn -i repro.csv --pc=1 --device-whitelist 0 \
  --plugin golden_plugin.py -o repro_result.csv --dump-on-fail --dump-format npy --warmup=false
# → 5 个用例正常执行（GOLD/PRECISION_STATUS 逐例输出），repro_result.csv 生成
```

**报错文案误导**：文案说 "kernel or json file does not exist"，实际 `.cce/.json` 在、缺的是编译产物——不要按文案字面往 kernel_meta 手工塞文件（编译产物须由 TBE 生成，手工文件版本不匹配会造成更隐蔽的运行期问题）。复跑黑盒子集（失败用例 repro）默认带 `--warmup=false`，把 warmup 留给计时场景。

坏实践（独立 trace 增补）：

- 按报错文案字面往 kernel_meta 手工补 `.o/.json` 文件。
- 因 warmup 崩溃放弃子集复现、直接跑全量——全量耗时长且难以附加 dump/调试参数。

验证信号（独立 trace 增补，euclidean_norm 黑盒失败修复会话，Ascend950PR / CANN 9.1.0）：

- 崩溃现场：`RuntimeError: Compile warmup kernel failed. kernel or json file does not exist`（栈至 `tbe_interface.py:379 compile_warmup_kernel`）；kernel_meta 有 warmup.cce/warmup.json、无编译产物（来源平台：950；证据等级：tool_observed）。
- `--no-warmup` 被拒：`ttk: error: unrecognized arguments: --no-warmup`（来源平台：950；证据等级：tool_observed）。
- `--warmup=false` 生效：同命令加参后 5 用例逐例输出 `GOLD/PRECISION_STATUS` 并产出 result CSV（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 本增量场景是 aclnn 模式的**精度复现**复跑（warmup 只服务计时稳定）；性能对比场景仍须保留 warmup 并解决编译环境（主卡 fallback 声明与 banned 卡约束不变）。
- warmup 重编失败的确切成因（TBE 编译环境在复跑 shell 的差异）未追根，仅绕过（与主卡遗留问题同源）。