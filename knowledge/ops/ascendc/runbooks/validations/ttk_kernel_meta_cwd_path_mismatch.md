---
type: "Runbook"
title: "动态编译产物落 CWD，TTK 查根 kernel_meta"
description: "TTK 白盒动态编译报 build artifacts not found：TBE 编译产物写入 CWD 相对 ./kernel_meta（测试目录），TTK 在 <TTK安装根>/kernel_meta 查找，两路径错位；从 TTK 根运行或在测试目录建 symlink 修复。"
tags: ["task.debugging", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# 动态编译产物落 CWD，TTK 查根 kernel_meta

## 现象

为自定义算子包跑 TTK 白盒（`python -m ttk kernel -i whitebox_all.csv --plugin golden_plugin.py ...`），工作目录在测试目录（operators/<Package>/tests/whitebox/branch-0/ttk/）。清掉 .ttk 失败缓存后，单用例仍失败：

```
[ERROR] [<用例名> OnDynCompilation] [dynamic_compilation.py:182]: Dyn operator dyn_op_<用例名> build artifacts not found
```

给 TTK 的 dynamic_compilation.py 临时加调试打印后确认：TTK 在 kernel_meta=<TTK根>/kernel_meta 里找 dyn_op_<用例名>.json，目录里只有 warmup.cce / warmup.json。设置 `export ASCENDC_BUILD_LOG_DIR=/tmp/ascendc_build_log` 重跑，编译日志（ascend950/0_dyn.log）显示 bisheng 编译命令全部正常执行、链接完成（`[LD] ld.lld -m aicorelinux -Ttext=0 ./kernel_meta/dyn_op_<用例名>.o ...`）；全盘 `find` 搜索发现产物 dyn_op_<用例名>.json / .o / _cube.i / _vec.i 等全部落在**运行目录下的 ./kernel_meta/**（TBE 以 CWD 相对路径写产物），而不是 TTK 查找的安装根 kernel_meta。把产物拷到 TTK kernel_meta 后该用例立即 PASS；改为从 TTK 根目录运行（输入/插件用绝对路径）全量 74/74 PASS。之后再从测试目录裸跑又复现失败，最终在测试目录建 `kernel_meta -> <TTK根>/kernel_meta` 符号链接解决。

## 适用场景

- 工程形态：TTK（ops-test-kit）白盒 kernel 模式动态编译（`python -m ttk kernel -i ... --plugin ...`）。
- API 路径：ttk/core_modules/npu/op/compilation/dynamic_compilation.py（读侧）；TBE compile_op / bisheng（写侧）。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.2.0。
- 适用粒度：platform_wide（TTK 工具行为，与具体算子无关）。
- 路线优先级：recommended_fastpath（symlink 一次到位；从 TTK 根运行也可，但改变工作目录会影响相对路径输入与结果文件落位）。
- 明确不适用 / 未验证场景：TTK 修正 root_path/kernel_meta 派生逻辑后需重验；静态编译产物路径未验证。

## 判别方法

1. 触发：TTK 白盒动态编译报 `Dyn operator <name> build artifacts not found`（dynamic_compilation.py:182，最终置 DYN_OPERATOR_BUILD_LOST），且**清了 .ttk 缓存后仍逐字复现**。
2. 探针（可执行）：

```bash
# a) 确认 TTK kernel_meta 里确实没有产物（默认只有 warmup.*）
ls <TTK根>/kernel_meta/
# b) 打开 TBE 编译日志捕获（bisheng 命令与产物相对路径都在里面）
export ASCENDC_BUILD_LOG_DIR=/tmp/ascendc_build_log
mkdir -p /tmp/ascendc_build_log
# 重跑失败用例后查看：tail -40 /tmp/ascendc_build_log/ascend950/0_dyn.log
# c) 全盘找产物实际落点
find <仓库根> -name "<dyn_op名>.*" 2>/dev/null
```

3. 原始观测：产物出现在**运行目录**的 ./kernel_meta/（如 operators/<Package>/tests/whitebox/branch-0/ttk/kernel_meta/dyn_op_*.json），TTK 查找路径为 <TTK根>/kernel_meta（ttk/utilities/classes.py: `self.kernel_meta = os.path.join(self.root_path, "kernel_meta")`，root_path 为 TTK 安装根的三级上级目录）。
4. 解释：TBE 编译侧以 CWD 相对 `./kernel_meta/` 写产物，TTK 侧按安装根绝对路径读产物，两边不一致 → FileNotFoundError → BUILD_LOST。kernel_meta 是隐藏 switch，无 CLI/配置入口。

## 根因

单一根因：TBE 动态编译产物写入 CWD 相对 ./kernel_meta/，TTK 在 <TTK安装根>/kernel_meta/ 查找，两路径错位。

## 解决配方

两种等价修复（任选其一，均实测通过）。

方式 A——从 TTK 根目录运行（输入与插件改绝对路径）：

```bash
cd <TTK根>
python -m ttk kernel \
  -i /abs/path/to/whitebox_all.csv \
  --plugin /abs/path/to/golden_plugin.py \
  -o /abs/path/to/result.csv \
  --single-log --no-prof --warmup false
```

方式 B——在测试目录建符号链接，使 CWD 的 ./kernel_meta 即 TTK kernel_meta：

```bash
WB_DIR=/abs/path/to/operators/<Package>/tests/whitebox/branch-0/ttk
rm -rf $WB_DIR/kernel_meta
ln -s <TTK根>/kernel_meta $WB_DIR/kernel_meta
```

注意：symlink 会作为未跟踪文件出现在 git status（仓库 .gitignore 只忽略 build/）；交付前需清理或在忽略规则中登记。

## 设计决策

- 备选 1：每次编译后拷贝产物到 TTK kernel_meta——否决：74 个用例逐个拷贝不可维护，仅用于单用例验证根因。
- 备选 2：改 TTK 源码把 kernel_meta 指到 CWD——否决：kernel_meta 为隐藏 switch 无 CLI 入口，且不应 patch 测试工具本体。
- 采用：优先方式 A（不动文件系统）；验收方习惯在测试目录跑（result.csv 落位），故同时保留方式 B symlink。
- 当前方案属 recommended_fastpath。

## 验证信号

问题与验证来源：一次白盒修复会话中定位并修复 kernel_meta 错位（Ascend 950 + CANN 9.2.0，TTK kernel 模式 74 用例）：

- 产物实际落点：find 命中测试目录 ttk/kernel_meta/ 下 dyn_op_*.{json,o,_cube.i,_vec.i,...}，而 TTK kernel_meta 仅有 warmup.cce/warmup.json（来源平台：950；证据等级：tool_observed）。
- 单用例对照：拷贝产物到 TTK kernel_meta 后该用例 STATUS: PASS / PRECISION_STATUS: PASS（来源平台：950；证据等级：tool_observed）。
- 方式 A（TTK 根运行）：Total 74 / PASS=74 / FAIL=0（来源平台：950；证据等级：tool_observed）。
- 反证：修复后从测试目录裸跑（无 symlink）仍 STATUS: FAIL（来源平台：950；证据等级：tool_observed）。
- 方式 B（symlink）：从测试目录运行 PASS=74 FAIL=0，result.csv 落测试目录（来源平台：950；证据等级：tool_observed）。

## 失效边界

- TTK 版本更新后 root_path/kernel_meta 派生或 TBE 写路径行为可能变化，需按判别方法重验。
- 多目录并发跑 TTK 时 symlink 方案共享同一 kernel_meta，可能互相覆盖 dyn_op 产物（未验证）。
- 其他平台未验证（推断可迁移）。

**未决项**：

- TTK 侧 root_path 与 TBE 侧 CWD 相对路径不一致属工具缺陷，未向上游反馈/修复。
- git 未忽略 kernel_meta symlink（.gitignore 只忽略 build/），交付仓库会带未跟踪文件。
## 坏实践

- 在测试目录裸跑 TTK 白盒且不看 kernel_meta 落点，把 BUILD_LOST 当成 kernel/精度问题去修。
- 用拷贝产物代替路径修复（每次编译后都要人肉拷）。

## 误诊

- 先后怀疑过：TBE 编译器被缓存/返回假成功、op_host 源文件未安装导致 tiling 编不了、"kernel_operator.h 找不到"（手工直跑 ccec 缺 -I 的红鲱鱼）——均非本根因；靠 ASCENDC_BUILD_LOG_DIR 日志 + find 产物落点证伪。
