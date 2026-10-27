---
type: "Runbook"
title: "黑盒全量验收：manifest过滤+3×bit-wise确定性"
description: 黑盒 ST 验收闭环配方：生成侧把超大 shape 用例挡在 manifest 外（不产生 SKIP 计数）、固定种子下 3 次全量运行 bit-wise 比对、白盒分支回归与退出码语义，一次性通过验收。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T00:49:13Z'
platforms: ["950"]
---

# 黑盒全量验收：manifest过滤+3×bit-wise确定性

## 现象

BesselI1e 算子黑盒批量验证首次交付被验收打回。当时状态：521 条黑盒用例已全部纳入数据驱动 ST 框架，精度三类根因（默认容差、inf→NaN、L2 异常语义）已修复，全量跑出 `PASS: 437 | SKIP: 84 | FAIL: 0`——84 条 SKIP 全是框架运行时的 128M 元素 OOM 守卫（`SKIP — shape too large (numElements=… > max=134217728)`）。验收结论 **FAILED**，三项不通过：

1. `[SKIP]` 不为 0：验收标准明确要求 `[SKIP]` / `[FAIL]` 均为 0；84 条 SKIP 被定性为"生成侧 max_shape_elems 未对齐框架 skip 阈值——黑盒生成器产出了超出 ST 框架上限的 shape，生成与执行对不上"；
2. 确定性验证缺失：要求 3 次全量运行 bit-wise 一致，LOG 只有单次执行记录；
3. 白盒回归证据缺失：LOG 声称白盒分支回归 PASS 但无可核对的执行日志。

修复要求（验收方给出的方向）：超大 shape 用例在 manifest 阶段过滤（不加入执行列表，不产生 SKIP 计数）、3 次全量运行比对 bit-wise 一致、全部白盒分支回归无误退化。目标 `SKIP=0, FAIL=0, 3 次 bit-wise 一致，全部白盒 PASS`。

## 适用场景

- 工程形态：aclnn 自定义算子的黑盒批量验证与验收（数据驱动 ST：CSV → manifest → C++ 框架 → 上板全量执行）。
- API 路径：manifest 生成脚本（`prepare_blackbox_cases.py`）、ST runner（`run.sh` / `test_aclnn`）、白盒分支 runner（`run_branch1.sh` / `run_branch2.sh`）。
- 验证过的平台与 CANN 版本：950（ascend950 构建目标，npu-smi 25.7.rc1）；CANN 9.0.0-beta.2。
- 适用粒度：`workflow_only`——黑盒验收工作流配方（对任何算子的黑盒批量验收适用）。
- 路线优先级：recommended_fastpath（验收口径要求 SKIP=0 + 确定性 + 回归时的标准做法）。
- 明确不适用/未验证场景：
  - 验收口径要求**全量 CSV 覆盖**（含超大 shape）时不适用 manifest 过滤——本配方以"过滤已记录、超大 shape 精度未验证"为边界前提；
  - 输出含非确定性来源（如 kernel 内原子累加、时间戳）的算子——3×bit-wise 可能天然不一致，须先消除非确定性再套用。

## 解决配方（三件套）

**① 生成侧过滤（治 SKIP 计数）**：超大 shape 用例不进 manifest，从源头消除 SKIP 计数；过滤数量打印留痕。`prepare_blackbox_cases.py`（verbatim）：

```python
MAX_ELEMENTS = 128 * 1024 * 1024  # 128M — 超大 shape 会 OOM，不纳入 manifest


def _num_elements(case):
    """计算用例的总元素数（取第一个 input shape 各维乘积）。"""
    shapes = case.get("input_shapes", [])
    if not shapes:
        return 0
    prod = 1
    for d in shapes[0]:
        prod *= d
    return prod


def prepare(csv_path, out_path, golden_path=None):
    """黑盒 TTK CSV → manifest。返回 case 数。"""
    cases = []
    op_name = ""
    filtered_count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = _blackbox_row_to_case(row)
            if case["op_name"] and not op_name:
                op_name = case["op_name"]
            # 过滤超大 shape：numElements > 128M → 不加入 manifest，避免 OOM
            if _num_elements(case) > MAX_ELEMENTS:
                filtered_count += 1
                continue
            cases.append(case)
    manifest = {
        "schema_version": 1,
        "source_csv": str(csv_path),
        "op_name": op_name,
        "golden_path": str(golden_path) if golden_path else None,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if filtered_count > 0:
        print(f"[prepare_blackbox_cases] filtered {filtered_count} cases (numElements > {MAX_ELEMENTS})")
    return len(cases)
```

**② 3×bit-wise 确定性比对（治确定性证据）**：`run.sh` 把单次执行改成三次全量运行 + 两两 diff，退出码语义收紧为"全 PASS 且 bit-wise 一致才 0"（verbatim，Step 4 段）：

```bash
# ============================================================================
# 4. Deterministic verification: run 3x, compare stdout bit-wise
# ============================================================================
echo ""
echo "--- Step 4: Deterministic verification (3x runs) ---"

# Capture 3 full runs with separate stdout/stderr
RUN_LOG_DIR="/tmp/bessel_i1e_st_runs_$$"
rm -rf "${RUN_LOG_DIR}" && mkdir -p "${RUN_LOG_DIR}"

RC_ARRAY=()
for run_n in 1 2 3; do
    echo "=== Run ${run_n}/3 ==="
    set +e
    ./test_aclnn "${MANIFEST}" > "${RUN_LOG_DIR}/run_${run_n}.stdout" 2> "${RUN_LOG_DIR}/run_${run_n}.stderr"
    RC_ARRAY[${run_n}]=$?
    set -e
    echo "  exit code: ${RC_ARRAY[${run_n}]}"
done

# ---------------------------------------------------------------------------
# Compare 3 stdout files for bit-wise identity
# ---------------------------------------------------------------------------
DIFF12=$(diff -q "${RUN_LOG_DIR}/run_1.stdout" "${RUN_LOG_DIR}/run_2.stdout" 2>/dev/null || true)
DIFF13=$(diff -q "${RUN_LOG_DIR}/run_1.stdout" "${RUN_LOG_DIR}/run_3.stdout" 2>/dev/null || true)
DIFF23=$(diff -q "${RUN_LOG_DIR}/run_2.stdout" "${RUN_LOG_DIR}/run_3.stdout" 2>/dev/null || true)

BITWISE_OK=true
[ -n "${DIFF12}" ] && BITWISE_OK=false
[ -n "${DIFF13}" ] && BITWISE_OK=false
[ -n "${DIFF23}" ] && BITWISE_OK=false

# ---------------------------------------------------------------------------
# Display run 1 output as canonical result
# ---------------------------------------------------------------------------
echo ""
echo "--- Canonical result (run 1) ---"
cat "${RUN_LOG_DIR}/run_1.stdout"
if [ -s "${RUN_LOG_DIR}/run_1.stderr" ]; then
    echo ""
    echo "--- Run 1 stderr ---"
    cat "${RUN_LOG_DIR}/run_1.stderr" >&2
fi

# ---------------------------------------------------------------------------
# Deterministic verdict
# ---------------------------------------------------------------------------
TEST_RC=${RC_ARRAY[1]}
echo ""
if [ "${BITWISE_OK}" = true ]; then
    echo "=== DETERMINISTIC VERIFICATION: BIT-WISE IDENTICAL (3/3 runs match) ==="
else
    echo "=== DETERMINISTIC VERIFICATION: MISMATCH DETECTED ==="
    if [ -n "${DIFF12}" ]; then echo "  [DIFF] run1 ≠ run2"; fi
    if [ -n "${DIFF13}" ]; then echo "  [DIFF] run1 ≠ run3"; fi
    if [ -n "${DIFF23}" ]; then echo "  [DIFF] run2 ≠ run3"; fi
    # Show first few diff lines for diagnosis
    if [ -n "${DIFF12}" ]; then
        echo "  --- run1 vs run2 (first 10 lines) ---"
        diff "${RUN_LOG_DIR}/run_1.stdout" "${RUN_LOG_DIR}/run_2.stdout" 2>/dev/null | head -10
    fi
fi

echo "=== Test finished: exit=${TEST_RC}, bitwise_ok=${BITWISE_OK} ==="

if [ ${TEST_RC} -eq 0 ]; then
    if [ "${BITWISE_OK}" = true ]; then
        echo "ALL cases PASSED and bit-wise consistent"
    else
        echo "ALL cases PASSED but NOT bit-wise consistent"
    fi
else
    echo "Cases with issues (exit ${TEST_RC})"
fi

# Exit 0 only if all runs pass AND bit-wise consistent
if [ ${TEST_RC} -eq 0 ] && [ "${BITWISE_OK}" = true ]; then
    exit 0
else
    exit 1
fi
```

**③ 白盒分支回归 + 验收留痕**：黑盒全量之外跑 `./run_branch1.sh`（FP16 白盒 11 条）与 `./run_branch2.sh`（BF16 白盒 17 条）确认无误退化，并把过滤数量、三次运行摘要、白盒结果逐项写入开发 LOG 供验收方核对。

## 设计决策

- 为何在 manifest 生成侧过滤而不是运行时 SKIP：验收口径是 `[SKIP]=0`，运行时守卫必然产生 SKIP 计数（首轮交付即因此被打回）；生成侧过滤让"超大 shape 不进入执行列表"成为显式、可留痕的范围决策（打印 `filtered 84 cases`），运行时守卫保留为兜底安全网。
- 为何 3×bit-wise 用 stdout 比对 + md5 复核：框架逐用例打印 PASS/FAIL 行与汇总，stdout 逐字节一致等价于三次结果完全一致；md5sum 三份文件同值是验收方可独立复核的最小证据（本例 `45e6de9d378a7b9b94f45abc0e09211a` ×3）。
- 确定性的机制前提：测试二进制全局 RNG 固定种子 `std::mt19937 rng(42)`（源码注释 "Global RNG — seed once only (hard constraint #1)"）+ manifest 顺序固定 → 每次生成相同输入、相同输出；若 RNG 逐用例重播种或 manifest 顺序变化，bit-wise 一致性前提被破坏。
- 放弃的候选：优化内存使超大 shape 可执行（分块 golden/流式比对）——实现成本高且 golden（scipy 单进程）对亿级元素同样不可行，本轮按验收方给定方向过滤并明示边界。
- 当前方案属 recommended_fastpath（在"过滤已授权"的验收口径内）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`，ascend-toolkit；LD_LIBRARY_PATH 需加 `opp/vendors/BesselI1e/op_api/lib/`）；npu-smi 25.7.rc1；GNU 13.3.0；构建入口 `operators/build.sh <pkg>`。
- 目标平台：ascend950（自定义算子 vendor 包安装至 `opp/vendors/BesselI1e/`）。
- 目录布局：`operators/BesselI1e_package/tests/aclnn/`（run.sh、test_aclnn.cpp、aclnn_cpp_cases.json、blackbox_all.csv、build/）；`skills/aclnn-st-framework/prepare_blackbox_cases.py`（manifest 生成）；`docs/bessel_i1e/develop/`（golden.py、LOG.md、STATE.md、DESIGN.md）。
- 相关文件清单（与本配方改动一一对应）：`prepare_blackbox_cases.py`（过滤）、`run.sh`（3×比对与退出码）、`test_aclnn.cpp`（RNG 固定种子、运行时 128M 守卫兜底）、`docs/bessel_i1e/develop/LOG.md`（验收留痕）。

## 预防

- 黑盒生成器的 `max_shape_elems` 与 ST 框架的可执行上限必须对齐（生成侧先算 numElements 再产出用例，或生成后跑一次过滤统计）。
- 验收标准（SKIP=0 / 3×确定性 / 白盒回归）在开发前就写进 runner：把"验收证据生产"内建到 `run.sh`，而不是交付后手工补跑。
- 全量上板前先估算时长：本例 437 条 × 约 6~7 秒/条，单轮全量天然超过工具单命令超时窗口（10~30 分钟），必须后台执行 + 轮询监控，前台全量运行必被超时杀掉且不留失败痕迹。

## 验证信号

以下信号来自同一次开发会话的验收闭环（ascend950 构建目标，CANN 9.0.0-beta.2）：

- 首轮交付（运行时 SKIP 方案）：全量 521 条 `PASS: 437 | SKIP: 84 | FAIL: 0`，验收 FAILED（SKIP≠0、无 3×确定性、无白盒回归日志）（来源平台：950；证据等级：tool_observed）。
- 过滤生效：manifest 生成日志 `[prepare_blackbox_cases] filtered 84 cases (numElements > 134217728)` + `wrote 437 cases`；manifest 复核 437 条（fp16 154 / bf16 137 / fp32 145 / float64 1，最大 129,895,048 元素 < 上限）（来源平台：950；证据等级：tool_observed）。
- 3 次全量运行：每次 `PASS: 437 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 437`、退出码 0；`DETERMINISTIC VERIFICATION: BIT-WISE IDENTICAL (3/3 runs match)`；三次 stdout md5 均为 `45e6de9d378a7b9b94f45abc0e09211a`、stderr 全空（来源平台：950；证据等级：tool_observed）。
- 白盒回归：Branch-1（FP16）11/11 PASS、Branch-2（BF16）17/17 PASS，无误退化（来源平台：950；证据等级：tool_observed）。
- 重新验收 PASS：验收方按 5 个维度实质核查（manifest 437 非空、纳入 437==437、golden 为 scipy.special.i1e 外部独立、3 次执行 MD5 一致且无 --quick/--max-cases、白盒回归 + DESIGN §4.3 阈值达标）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **84 条 >128M 元素用例（最大 1.8B 元素）的精度从未验证**——`SKIP=0` 以生成侧过滤为前提，过滤是验收方授权的范围决策而非精度结论；超大 shape 用例若纳入验收口径，需分块/流式 golden 方案（未实现）。
- "bit-wise 一致"是 stdout 层（逐用例 PASS 行 + 汇总）等价，未做输出张量二进制 dump 对比；stdout 一致依赖框架打印完整覆盖每个用例结果。
- 确定性前提是固定种子 + 固定顺序：换种子、改 manifest 顺序、或 kernel/框架引入非确定输出都会打破 3×一致。
- 进程被杀根因（超大 shape 执行时 137/静默死亡，宿主 754GB 内存、无 OOM dmesg）未定论——128M 上限是经验阈值，不是推导出的安全边界。
- 推断可迁移：其他 ascend 平台与算子的黑盒验收流程结构相同，未逐一验证（未验证，推断可迁移）。

**未决项**：

- 超大 shape 用例的可执行化（分块 golden / 流式比对 / 设备侧 golden）未尝试——若验收口径升级为全量 CSV 覆盖，这是第一接续点。
- 超大 shape 执行时进程被杀的根因（宿主 OOM vs SIGKILL）仍未定位——128M 阈值的推导化需要该根因。
- 3×bit-wise 的 stdout 等价性可升级为输出张量 dump 对比（更强证据），未实现。
## 坏实践

- 用运行时 SKIP 守卫应付"SKIP=0"的验收口径——SKIP 计数本身就是不通过项；范围决策要么执行、要么在生成侧显式过滤并留痕。
- 交付时只在 LOG 里声称"全量通过、白盒回归 PASS"而不留执行日志——本例验收方按"无可核对日志"判不通过；证据要能被独立复核（md5、diff、逐用例输出）。
- 前台跑全量上板验证赌工具超时——437 条级别必须后台 + 轮询；被超时杀掉的运行不留任何失败痕迹（stderr 空）。

## 误诊

- 首轮修复会话曾试图直接前台跑 3×全量（10~30 分钟超时窗口），多次被工具超时杀掉后一度怀疑进程挂死——实际是单轮全量天然需要远超工具超时窗口的时间，改后台 + 轮询后正常完成（来源平台：950；证据等级：tool_observed）。
