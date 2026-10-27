---
type: "Runbook"
title: 手改 kernel JSON int64Mode 无效且被重装覆盖
description: 安装目录下 kernel config JSON 与单 kernel JSON 的 int64Mode 改为 true 并清 cache 后 aclnn 直调行为不变；重装算子包还会把 JSON 覆盖回 false——该字段是 OPC 编译期产物，运行时 patch 安装产物不是可持久的修复（banned）。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# 手改 kernel JSON int64Mode 无效且被重装覆盖

## 现象

在排查自定义 argmax 算子（bench_no_harness_arg_max）int64 用例失败（框架对 int64 AR 小 shape 跳过自定义 tiling 的已知限制）时，尝试直接修改安装目录里的 kernel 元数据：用 Python 脚本把 `opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/bench_no_harness_arg_max.json` 中全部 5 个 `binList` 条目的 `int64Mode` 改为 `true`，又把 `kernel/ascend950/bench_no_harness_arg_max/` 下每个单 kernel JSON 里的 `supportInfo.int64Mode` 也改为 `true`，随后删除 `~/atc_data/kernel_cache/` 重跑 ST——失败用例依旧输出 `期望=7, 实际=6`，行为没有任何变化。更隐蔽的是：只要再执行一次 `bash build/custom_opp_ubuntu_x86_64.run --force` 重装，这些手改会被安装包覆盖回 `"int64Mode": false`，导致"改了没验证到"的假象。

## 适用场景

- 工程形态：算子仓自定义算子的安装产物（custom_opp .run 包安装到 `opp/vendors/<vendor>/`）调试；aclnn 直调执行。
- API 路径：kernel 配置 JSON（`config/ascend950/<op>.json` 的 `binList[].int64Mode`）、单 kernel JSON 的 `supportInfo.int64Mode` 字段、`~/atc_data/kernel_cache/`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（OPC 产物生成与安装行为，与算子无关）。
- 路线优先级：banned（该路线已被证伪：运行时手改安装产物不改变 aclnn 选核行为，且不可持久）。
- 明确不适用/未验证场景：atc 图编译离线场景是否读取这些字段未验证；结论限于 CANN 9.1.0 aclnn 直调。

## 根因

`int64Mode` 是 OPC 编译工具在打包时写入 kernel JSON 的编译期字段（对动态 shape 算子由 `__set_int64_mode` 强制 False），安装后的 .run 重装会以包内产物覆盖安装目录；而 aclnn 直调路径在运行时并不因为这些 JSON 字段被手改而改变 tiling/选核行为。两重机制叠加：**改了不生效，生效窗口也会被重装抹掉**。

## 解决配方

**尝试过的 patch 脚本形态**（记录为反面参照；对 config JSON 与单 kernel JSON 分别处理）：

```python
import json, os

base = '<ASCEND_HOME_PATH>/opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel'

# 1. Kernel config JSON - set int64Mode=true in binList
config_path = os.path.join(base, 'config/ascend950/bench_no_harness_arg_max.json')
with open(config_path) as f:
    data = json.load(f)
for entry in data.get('binList', []):
    entry['int64Mode'] = True
with open(config_path, 'w') as f:
    json.dump(data, f, indent=2)

# 2. Individual kernel JSONs - supportInfo.int64Mode
kernel_dir = os.path.join(base, 'ascend950/bench_no_harness_arg_max')
for fname in os.listdir(kernel_dir):
    if fname.endswith('.json'):
        fpath = os.path.join(kernel_dir, fname)
        with open(fpath) as f:
            kdata = json.load(f)
        if 'supportInfo' in kdata:
            kdata['supportInfo']['int64Mode'] = True
        with open(fpath, 'w') as f:
            json.dump(kdata, f, indent=2)
```

配套动作（同样无效，但属于该尝试的完整闭环）：

```bash
# Clear kernel cache
rm -rf ~/atc_data/kernel_cache/ 2>/dev/null
rm -rf /tmp/atc_data/kernel_cache/ 2>/dev/null
```

正确方向：字段值由编译期决定，若要改应在算子定义/编译选项层解决（本会话同时证明算子侧 `SetOpSupportInfo` 设置该字段同样无效），或升级 CANN；运行时 patch 安装产物不是可持久的修复。

## 设计决策

- 同时 patch config JSON 与单 kernel JSON：单 kernel JSON 的字段藏在 `supportInfo` 对象内（顶层 `grep int64Mode` 命中的其实是它），两处都改才能排除"改错文件"的假阴性——事后证明两处都改也不生效。
- patch 后立即用 `grep int64Mode` 回读确认写入成功、并清 kernel cache 排除缓存解释，把"没写进去/缓存"两个干扰变量排除后，才把无效归因到"框架不读"。

## 系统信息

- 芯片 Ascend950PR + CANN 9.1.0；安装根 `<ASCEND_HOME_PATH>/opp/vendors/<vendor>_custom/`。
- 相关文件：`op_impl/ai_core/tbe/kernel/config/ascend950/bench_no_harness_arg_max.json`（binList 5 条）、`op_impl/ai_core/tbe/kernel/ascend950/bench_no_harness_arg_max/*.json`（单 kernel，`supportInfo.int64Mode`）、`~/atc_data/kernel_cache/`。
- 构建安装：`bash build.sh --soc=ascend950 -j8` → `bash build/custom_opp_ubuntu_x86_64.run --force`（重装即覆盖）。

## 预防

- 不要把运行时 patch 安装目录 JSON 当修复提交或写进交付脚本：不可持久（重装覆盖）、不生效（框架不读）。
- 若要做"框架是否读字段"的对照实验，本方法仍可作为实验手段（本会话正是用它证伪了字段的作用），但必须 patch 后立即回读 + 清缓存 + 不重装的前提下复测。

## 验证信号

以下信号来自该 argmax 算子 int64 排查会话（Ascend950PR + CANN 9.1.0），均为工具实测：

- patch 脚本回读：`Config: 5 entries set to int64Mode=true`、`Individual kernels: 5 files set to int64Mode=true`（写入成功）（来源平台：950；证据等级：tool_observed）。
- 清 cache 后重跑 ST：失败用例仍 `期望=7, 实际=6`，通过率不变（来源平台：950；证据等级：tool_observed）。
- 一次 `--force` 重装后 `grep int64Mode` 回读全部回到 `"int64Mode": false`（覆盖复现）（来源平台：950；证据等级：tool_observed）。
- 全 CANN 安装目录检索：内置算子（含官方 ArgMax 的 int64 条目）也全部 `int64Mode: false`，无一 `true` 案例（旁证该字段在 9.1.0 上不是常规开启项）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限于 aclnn 直调 + CANN 9.1.0 + 动态 shape 自定义算子；atc 离线图编译、其他版本未验证（推断可迁移）。
- 若目标是"临时做实验观察框架是否读字段"，本方法仍可作为实验手段（本会话正是用它证伪了字段的作用），只是不能作为交付修复。

**未决项**：

1. （优先项）确认 aclnn 运行时实际读取选核元数据的完整来源（内存中的 ops-info/二进制元数据 vs 磁盘 JSON），从机制层解释"为什么不读"。
2. atc 图编译路径是否消费手改后的 JSON 字段（未验证）。
## 坏实践

- 把运行时 patch 安装目录 JSON 当修复提交或写进交付脚本：不可持久（重装覆盖）、不生效（框架不读），还会污染后续排查的环境基线。
- patch 后忘记回读与清缓存就下结论"字段没用"：必须先排除"没写对文件/缓存命中原值"两种解释。

## 误诊

- 最初以为此前一次 patch 实验失败是因为"修改被重装覆盖了所以没验证到"——后续在不重装的前提下 patch 并立即重跑，失败依旧，证明即使改动持续存在也不生效（来源平台：950；证据等级：tool_observed，作为误诊排除记录）。
