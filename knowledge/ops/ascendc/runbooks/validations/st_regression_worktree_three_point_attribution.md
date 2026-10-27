---
type: "Runbook"
title: ST 回归归因 worktree 三历史点复跑
description: 验收发现既有分支 ST 从全绿退化时，用 git worktree 在"kernel 提交 / tiling 提交 / 测试框架提交"三个历史点 clean build + 上板复跑，把回归归因到测试侧改动并排除 kernel/tiling/golden；worktree 构建会覆盖共享 OPP 安装，验后必须回主仓重建恢复。
tags: [task.testing, task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# ST 回归归因 worktree 三历史点复跑

## 现象

在 Ascend950 上验收 RoPE 类算子新分支（interleaved 布局）的 kernel 实现时：新分支自身 ST 12/12 全绿，但回归核对发现**既有 half 分支的 ST 从验收基线 12/12（数值 11/11 + 空张量 1/1）退化为 5/11**，失败集固定为 B=2 广播、大 shape fp16、D=6 非对齐、query NaN、query inf、fp16 大值六类用例，通过的是三 dtype 基线、最小 D、全零与空张量等结构用例。失败模式是真实数值错（如广播用例 actual=0.418682 expected=0.470306），不是调用报错。实现该 kernel 的开发者声称"退化是前序 tiling 改动引入的既有回归，非本任务引入"——需要客观归因：退化到底是 kernel、tiling、golden oracle 还是测试框架改动引入的？

## 适用场景

- 工程形态：CANN 自定义算子多分支迭代开发（tiling / kernel / ST 测试框架分任务改动、逐提交落库），验收时发现既有分支回归。
- API 路径：git worktree / git log 按文件归因；`./build.sh <package>`（clean 构建并安装到 `$ASCEND_OPP_PATH/vendors/<vendor>/`）；aclnn ST 的 `run.sh`（重建 + 重新生成用例数据 + 跑 manifest）。
- 验证过的平台与 CANN 版本：Ascend950（Ascend950PR），CANN 9.2.0。
- 适用粒度：workflow 级回归归因流程，结论层绑定"构建产物安装到共享 OPP 路径"这一 CANN 工程特性。
- 路线优先级：recommended_fastpath（出现"谁引入的回归"争议时的标准动作）。
- 明确不适用 / 未验证场景：工作区有未提交改动（worktree 检出的是提交历史，未提交状态需先提交或 stash 后才能复跑）；跨算子仓的依赖变更不在本流程覆盖内。

## 根因/解决配方（判别方法）

归因原理：把"最近三个语义提交点"（kernel 实现点 / tiling 扩展点 / 测试框架改造点）分别在独立 worktree 里 clean build + 上板复跑同一 ST，比较结果与当前失败集是否一致；同时用按文件的提交历史锁定"哪个提交之后哪个文件再没改过"，形成交叉证据。

可执行探针（在本轨迹的实际执行，路径按需替换）：

```bash
# 0) 先固定当前状态证据：主仓 HEAD、工作区干净、当前失败集（略）
git -C <repo> status --short && git -C <repo> log --oneline -8

# 1) 按文件归因：kernel / golden / 用例生成器 / 测试框架 各自最后被哪个提交改过
git -C <repo> log --oneline -- operators/<pkg>/op_kernel/<op>_kernel.cpp
git -C <repo> log --oneline -- docs/<op>/develop/golden.py
git -C <repo> log --oneline -- operators/<pkg>/tests/aclnn/gen_<branch0>_cases.py
git -C <repo> log --oneline -- operators/<pkg>/tests/aclnn/test_aclnn.cpp

# 2) 三分点 worktree 复跑（每个点：clean build → 重新生成用例 → 跑 ST）
git -C <repo> worktree add /tmp/<verify_dir> <kernel_impl_commit>
cd /tmp/<verify_dir>/operators && source ../../docs/<op>/develop/env.sh
bash build.sh <package> && cd <package>/tests/aclnn && bash run.sh  # 记录结果与退出码
git -C <repo> worktree remove /tmp/<verify_dir>
#    对 <tiling_commit>、<test_framework_commit> 重复同样步骤

# 3) 【关键】恢复共享安装：worktree 的构建会把旧 kernel 安装进共享 OPP，覆盖当前版本
cd <主仓>/operators && bash build.sh <package>   # 重建当前 HEAD 的 kernel 并重装
```

判读规则：设当前失败集为 F。若 kernel 实现点全绿、tiling 扩展点全绿、测试框架改造点出现与 F 完全一致的失败集，且回归窗口内 kernel/golden/用例生成器均无提交——则回归由**测试侧改动**引入（kernel 数值本身没退化，是测试的评价链路变了），验收判 FAILED 并把修复派回测试侧；反之在哪个点变红，回归就引入于该点之后。

本轨迹的实测结果：kernel 实现点 12/12 全绿（RUN_EXIT=0）；tiling 扩展点 12/12 全绿（RUN_EXIT=0）；测试框架改造点 5/11、失败集与当前逐用例一致——归因成立：回归由"为支持新分支 manifest 而改造 test_aclnn.cpp/run.sh/新增第二套用例生成器"的测试侧提交引入；kernel（实现点之后仅新分支改动）、golden.py（更早提交后未改）、原分支用例生成器（未改）排除。开发者"前序 tiling 改动引入"的说法被 worktree 证据证伪。

## 设计决策

- 用 worktree 而不是 checkout 切换：主仓工作区保持当前验收状态不动，多历史点可并行开目录复跑，互不污染。
- 三分点选择"kernel / tiling / 测试框架"而不是逐提交 bisect：提交粒度是按任务的，三个语义点已能把四类嫌疑（kernel/tiling/golden/测试）两两切开，成本远低于逐提交上板。
- 每个点都做 clean build + 用例重新生成：该工程的 run.sh 每次重新生成 .bin 数据（.gitignore 排除），必须排除"用例数据陈旧"这一假信号。
- 复跑后强制回主仓重建：构建即安装——worktree 里构建旧提交会把旧 kernel 装进共享 `$ASCEND_OPP_PATH/vendors/<vendor>/`，直接覆盖当前正在验收的版本；不恢复的话后续所有测试跑的都是旧 kernel（本轨迹已实际执行恢复，RESTORE_EXIT=0）。

## 预防

- 多分支共用一套 ST 测试框架时，改测试框架（哪怕只是输出标签、manifest 支持这类"无害"改动）必须连同**全部既有分支**一起回归跑，再提交。
- 验收口径里把"前序已过用例回归不退化"写成硬标准（对比历史基线数值），并在验收环境里保存基线结果摘要（12/12 vs 5/11 这类），避免"开发者说是既有问题"这类口头归因通过验收。
- 团队共享一台机器/一个 CANN 环境时，任何历史点复跑后都要重建当前版本——把"恢复安装"写进归因流程的固定收尾步骤。

## 验证信号

以下信号来自该算子新分支 kernel 的验收会话：对三个历史提交点分别 clean build + ST 复跑，与当前失败集对照。

- 当前状态：既有分支 ST numeric 5/11（失败集固定六类）+ 空张量 1/1，新分支 12/12（来源平台：950；证据等级：tool_observed）。
- 历史点复跑：kernel 实现点 12/12 全绿（退出码 0）；tiling 扩展点 12/12 全绿（退出码 0）；测试框架改造点 numeric 5/11、失败集与当前逐用例一致（来源平台：950；证据等级：tool_observed）。
- 文件历史交叉证据：kernel 源在实现点之后仅被新分支任务修改；golden.py 与原分支用例生成器在回归窗口内零提交；test_aclnn.cpp/run.sh 被测试框架改造提交修改（来源平台：950；证据等级：tool_observed）。
- 收尾恢复：回主仓重建当前 kernel 并重装成功（RESTORE_EXIT=0），worktree 全部清理，主仓工作区 clean（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只能归因到"提交点"粒度；测试侧改动里**具体哪一行**导致既有分支数值错，本流程不回答（该机制定位落在后续开发段，见遗留问题）。
- 依赖"用例数据生成确定性"：若用例生成器带未固定随机种子，历史点重新生成的数据与当年不同，结果对比会被数据噪声污染（本轨迹用例生成确定性成立，失败值逐位复现）。
- 共享 OPP 安装的覆盖问题意味着该流程**不能并行**跑多个历史点构建（互相覆盖安装），只能串行逐点执行。
- 未验证场景：算子包安装在 `$ASCEND_HOME_PATH/vendors/` 与 `$ASCEND_OPP_PATH/vendors/` 双位置并存的多版本冲突形态（那是另一类问题）。

**未决项**：

- **优先项：定位测试侧改动的确切机制**——本流程只完成"归因到提交点"；两个分支的用例生成器共用 `cases/<case_id>/` 目录名（同类用例同名）导致后生成的数据覆盖先生成的数据这一机制假设，未验证，修复动作已派发但闭合于后续开发段。
- 历史点复跑串行执行的耗时成本未度量；若需频繁归因，可评估"安装目录按提交隔离"的改进方案（未尝试）。
## 坏实践

- 采信开发者口头归因（"既有回归、非我引入"）而不做历史点复跑——本例该说法被三个 worktree 点证伪。
- 在主仓直接 checkout 历史提交复跑：丢失当前验收状态，且容易忘记切回、忘记重建。
- worktree 复跑完不恢复共享 OPP 安装就继续验收——后续所有测试都在跑旧 kernel，得出混乱结论。

## 误诊

- 开发者判断"既有分支 5/11 是前序 tiling（新增分发分支）改动引入的既有回归、与本任务无关"——worktree 复跑证伪：tiling 扩展提交点上既有分支仍 12/12 全绿；该判断为口头叙述，被工具观测推翻，已按证据改判为测试侧改动引入。
