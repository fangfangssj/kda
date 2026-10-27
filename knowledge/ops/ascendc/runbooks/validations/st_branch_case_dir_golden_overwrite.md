---
type: "Runbook"
title: "多分支 ST 共用 case 目录 golden 被覆盖"
description: "两个分支的 ST 用例生成脚本把同名 case_id 的 .bin 写进同一个 cases/<id>/ 目录，run.sh 顺序执行使后生成分支的 golden 覆盖前一分支，前分支 kernel 正确却报数值失配；按分支子目录隔离 .bin 修复。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# 多分支 ST 共用 case 目录 golden 被覆盖

## 现象

在 Ascend950PR（CANN 9.2.0）上开发多分支自定义算子（双模式 half/interleaved × 双布局 BSND/BNSD，共 8 个 tilingKey 分支；此处以 RoPE 类算子为例）。已完成的 half 模式分支（tilingKey=0，BSND + half + cos.ndim=2）ST 原本 12/12 全绿；在实现完 interleaved 分支（tilingKey=1，BSND + interleaved）并跑通本分支 ST 12/12 后验收发现：**前一分支的 ST 数值用例从 11/11 退化为 5/11**。失败的恰好是两分支**共用 case_id** 的用例（b1_b_broadcast、b4_large_fp16、b5_d6_unaligned、c1_query_nan、c3_query_inf 等），报 output mismatch，例如 NaN 注入用例报 `actual=0.38775 expected=nan at idx=1`（NaN 落位因 rotaryMode 不同而不同）；而 a1/a2/a3 等**非共用** case_id 用例全部通过。

诊断探针（判别关键）：清空用例数据后**只**重生成前一分支、不让后一分支覆盖，再跑该分支 manifest——

```bash
source docs/<OP>/develop/env.sh && cd operators/<OP>_package/tests/aclnn \
  && rm -rf cases && python3 gen_branch0_cases.py >/dev/null 2>&1 \
  && cd build && export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/<OP>/op_api/lib/:${LD_LIBRARY_PATH} \
  && ./test_aclnn ../cases/branch0_cases.manifest 2>&1 | tail -16
```

输出 `==== branch0_cases ST summary: numeric 11/11 PASSED, returns_empty 1/1 OK (total 12/12) ====`——前一分支恢复全绿，证明 kernel 本身正确、问题出在测试数据。

## 适用场景

- 工程形态：aclnn 自定义算子**多分支（多 tilingKey / 多 attr 模式）ST 测试**——golden .bin + manifest 驱动的 C++ 比对器，多个 `gen_*_cases.py` 由 `run.sh` 顺序执行。
- API 路径：`tests/aclnn/gen_branch{0,1}_cases.py`（torch 独立 oracle 离线产 6 个 .bin：q/k/cos/sin/exp_qout/exp_kout）、`run.sh`（顺序跑两个 gen + 两个 manifest）、`test_aclnn`（dataDir = manifest 所在目录，按 manifest 相对路径读 .bin）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——任何"多个用例生成脚本 + 共享输出目录 + 顺序执行"的 ST 组织方式（跨算子可迁移）。
- 路线优先级：recommended_fastpath（分支子目录隔离是最小、对称、不触碰 kernel/比对器的修复）。
- 明确不适用 / 未验证场景：单分支 ST（无共享目录问题）；kernel 侧真实精度缺陷——本卡现象的判别特征是"共用 case_id 的用例失配 + 非共用用例全过 + 失配值恰好等于另一模式的 golden"。

## 根因

两个分支的用例生成脚本把**同名 case_id** 的 6 个 .bin 写到**同一个** `cases/<case_id>/` 目录；`run.sh` 先跑 gen_branch0 再跑 gen_branch1，后者用 interleaved 模式的 golden 覆盖了前者先写的 half 模式 golden，于是前一分支 ST 拿正确的 half kernel 输出去对比 interleaved golden。

机制链：两个 gen 脚本对共用 case_id（b1/b2/b3/b4/b5/c1-c4）用**相同 seed 与 shape** 生成输入（输入 .bin 内容相同），仅 `rotaryMode` 不同 → golden 不同；被覆盖后"输入仍正确、期望错误"→ 比对必失配。三个例外精确解释了 5/11 的失败面：D=2 时 half 与 interleaved 数学结果相同（只有一对元素，两模式一致），b2_d2_min 仍通过；全零输入两种模式输出同为零，c2_all_zero 仍通过；b3 空 tensor（S=0）结构用例无比对，仍 OK。即失败集 = 共用 case_id ∩ 模式敏感。

## 解决配方

修复前（`gen_branch0_cases.py` 的 `save_case`，`gen_branch1_cases.py` 同构、仅 manifest 名不同）：

```python
CASES_DIR = os.path.join(_HERE, "cases")
MANIFEST = os.path.join(CASES_DIR, "branch0_cases.manifest")

def save_case(case_id, dtype_name, q, k, cos, sin, layout, rotary_mode, special):
    """调独立 oracle 算期望，落 6 个 .bin；返回 manifest 相对路径列表。"""
    exp_q, exp_k = apply_rotary_pos_emb(q, k, cos, sin, layout, rotary_mode)
    d = os.path.join(CASES_DIR, case_id)
    os.makedirs(d, exist_ok=True)
    rel = []
    for name, t in [("q", q), ("k", k), ("cos", cos), ("sin", sin),
                     ("exp_qout", exp_q), ("exp_kout", exp_k)]:
        with open(os.path.join(d, name + ".bin"), "wb") as f:
            f.write(to_raw(t, dtype_name))
        rel.append("{}/{}.bin".format(case_id, name))
    return rel
```

修复后（加分支子目录常量，`gen_branch1_cases.py` 对称改 `BRANCH = "branch1"`）：

```python
CASES_DIR = os.path.join(_HERE, "cases")
MANIFEST = os.path.join(CASES_DIR, "branch0_cases.manifest")
# 分支数据子目录：与 gen_branch1_cases.py 隔离 .bin 落盘路径，避免两分支共用 case_id
# （b1/b2/b3/b4/b5/c1/c2/c3/c4）时 gen_branch1 覆盖 gen_branch0 的 half-mode golden。
BRANCH = "branch0"

def save_case(case_id, dtype_name, q, k, cos, sin, layout, rotary_mode, special):
    """调独立 oracle 算期望，落 6 个 .bin；返回 manifest 相对路径列表。"""
    exp_q, exp_k = apply_rotary_pos_emb(q, k, cos, sin, layout, rotary_mode)
    d = os.path.join(CASES_DIR, BRANCH, case_id)
    os.makedirs(d, exist_ok=True)
    rel = []
    for name, t in [("q", q), ("k", k), ("cos", cos), ("sin", sin),
                     ("exp_qout", exp_q), ("exp_kout", exp_k)]:
        with open(os.path.join(d, name + ".bin"), "wb") as f:
            f.write(to_raw(t, dtype_name))
        rel.append("{}/{}/{}.bin".format(BRANCH, case_id, name))
    return rel
```

兼容性要点：`test_aclnn` 的 dataDir 取 manifest 所在目录（`cases/`），manifest 条目相对路径改为 `branch0/<case_id>/q.bin` 后 C++ 比对器**无需改动**；`run.sh` 顺序执行两个 gen 的逻辑也无需改动。`cases/` 整体在 `tests/aclnn/.gitignore` 中（由 gen 脚本重建、不提交），故该修复纯属运行期数据布局，不涉及仓库数据迁移。

## 设计决策

- 选"双方都加分支子目录"（对称、每分支 manifest 自包含），放弃"只给后实现的分支加前缀"（不对称）与"重命名后实现分支的 case_id"（改变用例 ID 语义、更侵入）。
- 修复放在 gen 脚本（落盘侧）而不是 test_aclnn/run.sh（读取/编排侧）：覆盖发生在落盘时刻，读取侧无法区分数据属于哪个分支。

## 验证信号

以下信号来自该算子 interleaved 分支实现完成后的跨分支回归修复会话（修复 = 两个 gen 脚本按分支子目录隔离 .bin，kernel/host/比对器零改动）：

- 修复后 `./run.sh` 全量重建执行：branch0 `numeric 11/11 PASSED, returns_empty 1/1 OK (total 12/12)`、branch1 `numeric 11/11 PASSED, returns_empty 1/1 OK (total 12/12)`，`run.sh exit=0`（来源平台：950/Ascend950PR；证据等级：tool_observed）。
- UT 全套 57/57 通过（公共 33 + branch-0 12 + branch-1 12 TilingUT），前序已过用例无回归（来源平台：950；证据等级：tool_observed）。
- `git diff --stat` 仅两个 gen 脚本被改（`2 files changed, 10 insertions(+), 4 deletions(-)`），kernel/host/test_aclnn.cpp/run.sh 未动（来源平台：不涉及 NPU；证据等级：tool_observed）。
- 判别探针本身：只重生成 branch-0 数据（无覆盖）时该分支即恢复 12/12——覆盖机制的直接证据（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复只解决"顺序执行覆盖"；若两个分支对同一 case_id 需要不同 seed/shape 的输入，仍须各自独立维护（本例两分支共用 seed 属刻意设计）。
- D=2（单对元素）等模式不敏感 shape 无法暴露覆盖问题（b2 通过是**假阴性**），回归发现依赖模式敏感用例的存在。
- 3 个以上分支共用同一 cases/ 顶层目录的布局未验证（推断同样适用，未验证）。

**未决项**：

- 无（覆盖链已闭合验证；`cases/` 为 gitignored 再生数据，无仓库侧残留）。
## 坏实践

- 多分支 ST 直接复用同一 `cases/` 顶层目录，且各分支 case_id 命名空间重叠。
- 看到他分支精度回归先怀疑 kernel：本例验收一度把回归归因到 kernel / ST 框架改动，而 kernel 无任何改动；应先做"单分支重生成对照"再下结论。

## 误诊

- 无（判别探针一次定位：共用 case_id 失配 + 非共用全过 + 单分支重生成即恢复）。
