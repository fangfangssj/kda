---
type: "Runbook"
title: "OOM 插桩是编译期：release 套件 OOB=UNKNOWN 属预期"
description: op_debug_config=oom 为编译期插桩，不能附在 -b release 上冒充已检测；OOB 证据须 -d true 动态编译套件单独产出，UNKNOWN 不得当 PASS，且要与 tiling 拒收型 UNKNOWN 区分。
tags: [task.testing, topic.memory]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# OOM 插桩是编译期：release 套件 OOB=UNKNOWN 属预期


## 现象

在对一个 AscendC 自定义算子（vendor 部署，GRU Cell 反向融合算子）执行 Branch-1 白盒验证时，先用 TTK kernel 模式的 release 精度命令（`-d false -b release`，复用已编译二进制）单例试跑一条用例：结果行里 `memory_oob_status=PASS`，但 `dyn_oob_result=UNKNOWN`、`bin_oob_result=UNKNOWN`——内存列看似通过却没有任何动态越界检测证据。若此时直接下结论"内存无越界"，就是把"没检测"当成了"没越界"。

随后按白盒执行流程对同一用例表（30 条）单独执行动态 OOM 插桩套件（`-d true -c false -b false --compile-opts op_debug_config=oom`，关闭 const 与 release 二进制、只留动态编译插桩结果）：29 条可执行用例全部拿到实打实的动态 OOB 证据（`memory_oob_status=PASS` 且 `dyn_oob_result` 六项证据全为 `OK`），0 条越界；唯一无证据的一条是负向探针用例（tiling 拒收 → kernel 未执行 → 无可插桩对象，`dyn_oob_result=UNKNOWN` 另有成因）。也就是说：release 套件里的 `UNKNOWN` 是"二进制未插桩"的必然产物，与算子是否越界无关。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发中的白盒评测——TTK kernel 模式对 vendor 部署的成品 kernel 执行"release 精度套件 + 动态 OOM 插桩套件"双套全量。
- API 路径：TTK `kernel` 子命令的编译/执行开关组合（`-d`/`-c`/`-b`、`--compile-opts op_debug_config=oom`）；结果 CSV 的 `memory_oob_status` / `dyn_oob_result` / `bin_oob_result` 列；`parse_result.py --memory-oob` 内存门禁。
- 验证过的平台与 CANN 版本：Ascend950PR（TTK Device Platform: Ascend950PR_9579）+ CANN 9.0.0-beta.2（TTK 检出于 `/tmp/ops-test-kit`）。与 frontmatter `platforms`（950）一致。
- 适用粒度（transfer_scope）：`platform_wide`——双套件命令与判读规则完全算子无关（不含任何算子常量），对 TTK kernel 模式下所有 vendor 自定义算子成立；仅在本平台实测（见失效边界）。
- 路线优先级：`recommended_fastpath`——内存安全结论必须以动态插桩套件证据为准，这是标准路径（无备选安全路线）。
- 明确不适用 / 未验证场景：aclnn 模式（子命令与 golden 形态不同）；`op_debug_config` 的其他取值（如 overflow 检测）未验证；其他 NPU 平台与 CANN/TTK 版本未验证。

## 判别方法

拿到两套结果后按列判读：

1. **精度结论只看 release 套件** `result.csv` 的 `precision_status`；OOM 套件的精度列不作为精度结论（其编译方式不同）。
2. **内存结论只看 OOM 套件** `result_oom.csv`：门禁要求每条 `memory_oob_status=PASS` **且** `dyn_oob_result` 的本轮运行证据全为 `OK`/`PASS`；缺列、空值、`UNKNOWN`、其他未知值、`FAIL`、缺行或多行均不通过。
3. release 套件中 `dyn_oob_result=UNKNOWN` 属预期（release 二进制无插桩），既不是越界也不是无越界的证据——不要用它下内存结论。
4. OOM 套件中若个别用例 `dyn_oob_result=UNKNOWN`，先回执行日志区分成因：tiling 拒收（`OPTILING_FAILURE`/`INVALID_TILING`，kernel 未执行 → 无可插桩对象）与插桩/执行失败是两种不同的 UNKNOWN，前者按负向用例豁免机制处置，后者必须重跑排查。

## 根因

`op_debug_config=oom` 的越界检测（TTK/CANN 的 out-of-bounds memory 检测，不是耗尽 HBM 制造 Out-of-Memory）是**编译期插桩**：检测代码在 kernel 动态编译时注入。release 套件用 `-b release` 复用未插桩的已编译二进制、且 `-d false` 关闭动态 shape 编译，因此运行期不可能产生动态 OOB 数据，结果列只能是 `UNKNOWN`。所以内存越界证据必须对同一用例表单独跑动态编译套件（`-d true -c false -b false --compile-opts op_debug_config=oom`）来取；`-c false -b false` 确保结果里只保留动态插桩一路，避免 const/release 路的 `UNKNOWN` 干扰判读。

## 设计决策

- **为什么 OOM 不能附在 `-b release` 上**：插桩在编译期生效，把 `--compile-opts op_debug_config=oom` 加到 release 命令上不会让已编译二进制带上检测，只会造成"已检测"的假象——因此内存套件与精度套件必须分开跑，两套结果各司其职。
- **为什么 `-c false -b false` 只留动态一路**：OOB 判读只认动态插桩证据；保留 const/release 路会引入 `UNKNOWN` 列值，与真失败混杂。
- **为什么精度以 release 套件为准**：release 路复用正式编译产物，精度结论与交付形态一致；OOM 套件的动态编译产物不同，其精度列不具交付代表性。
- **内存门禁是硬门禁**：内存越界是安全问题，不适用精度失败的重试耗尽逃逸条款；唯一合法的"无证据"豁免是负向用例的 tiling 拒收（kernel 未执行、无可插桩对象），且须经 TestSpec `oob_exemptions` 声明可审计。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`ASCEND_OPP_PATH`/`ASCEND_TOOLKIT_HOME` 由 `set_env.sh` 提供；环境经 `docs/<Op>/develop/env.sh` 导出 `PY=python3`、`TTK=/tmp/ops-test-kit`）；TTK 检出于 `/tmp/ops-test-kit`（CWD 须为 TTK 根，`msprof` 写入相对此目录且须当前用户可写）。
- 目标平台：Ascend950PR（npu-smi 实测在位；TTK 日志 Device Platform: Ascend950PR_9579）。
- 目录布局：算子包 `operators/<Op>_package/`（用例表 `tests/whitebox/branch-<n>/ttk_design_doc_cases.csv`、共享输出目录 `tests/whitebox/ttk/`）；golden `docs/<Op>/develop/golden.py`；vendor 部署于 `$ASCEND_OPP_PATH/vendors/<OpType>/`（`op_impl/ai_core/tbe/<OpType>_impl/dynamic/`）。
- 相关文件：`result.csv`（release 精度）、`result_oom.csv`（动态 OOM）、`full_run.log`/`oom_debug.log`（两套执行日志）、`known_blockers.md`（负向用例豁免登记）。

## 完整代码

双套件完整命令（变量定义 + 两套调用，`verbatim` 取自实测会话；`$INPUT_DIST_ARG` 由 spec.yaml 机械推导，冒烟与全量必须带同一份）：

```bash
source <绝对路径>/docs/<Op>/develop/env.sh       # → $PY / $TTK / ASCEND_*
PKG=<operator_package_dir 绝对路径>
OPTYPE=<OpType>                                   # 驼峰，如 ThnnFusedGruCellGrad
SOC=Ascend950PR                                   # SocVersion
GOLDEN=<docs/<Op>/develop/golden.py 绝对路径>
OUTDIR="$PKG/tests/whitebox/ttk"
MERGED="$OUTDIR/whitebox_all.csv"                 # 本分支用例表副本（按分支重新拷贝）
VENDOR="$ASCEND_OPP_PATH/vendors/$OPTYPE"
cd "$TTK"                                         # CWD 必须 == TTK 根
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$LD_LIBRARY_PATH"   # 走 vendor 源，不设 ASCEND_CUSTOM_OPP_PATH

# ── 套件一：release 精度（-d false -b release，复用已编译二进制）──
rm -f -- "$OUTDIR/result.csv"
env PYTHONPATH="$TTK:$(dirname "$GOLDEN"):$ASCEND_TOOLKIT_HOME/python/site-packages" \
    TTK_FULL_SOC_VERSION="$SOC" \
    "$PY" -m ttk kernel -i "$MERGED" -d false -b release --pc=1 --task-prof false \
    --run 1 --warmup false $INPUT_DIST_ARG --plugin "$GOLDEN" -o "$OUTDIR/result.csv"

## ── 套件二：动态 OOM 插桩（-d true -c false -b false + op_debug_config=oom）──
TBE="$VENDOR/op_impl/ai_core/tbe"
OOM_RESULT="$OUTDIR/result_oom.csv"
OOM_LOG="$OUTDIR/oom_debug.log"
rm -f -- "$OOM_RESULT"
OOM_TTK_STATUS=0
env PYTHONPATH="$TTK:$(dirname "$GOLDEN"):$ASCEND_TOOLKIT_HOME/python/site-packages:$TBE" \
    TTK_FULL_SOC_VERSION="$SOC" \
    "$PY" -m ttk kernel -i "$MERGED" -d true -c false -b false \
    --compile-opts op_debug_config=oom --pc=1 --task-prof false \
    --run 1 --warmup false $INPUT_DIST_ARG --plugin "$GOLDEN" \
    -o "$OOM_RESULT" >"$OOM_LOG" 2>&1 || OOM_TTK_STATUS=$?
printf 'TTK-OOM-EXIT status=%s\n' "$OOM_TTK_STATUS" >>"$OOM_LOG"
```

OOM 证据逐条核查（内存门禁：每条 `memory_oob_status=PASS` 且 `dyn_oob_result` 证据全 `OK`/`PASS`）：

```python
import csv
OUTDIR = "<tests/whitebox/ttk 绝对路径>"
with open(OUTDIR + "/result_oom.csv", newline="") as f:
    rows = list(csv.DictReader(f))
ok = 0; flagged = []
for r in rows:
    name = r.get("testcase_name")
    mos = (r.get("memory_oob_status") or "").strip()
    doob = (r.get("dyn_oob_result") or "").strip()
    parts = [p.strip() for p in doob.split(",") if p.strip()]
    if mos == "PASS" and parts and all(p in ("OK", "PASS") for p in parts):
        ok += 1
    else:
        flagged.append((name, mos, doob))   # 逐条回日志区分成因：tiling 拒收 vs 插桩/执行失败
print(f"OOB PASS with full evidence: {ok}/{len(rows)}")
for n, m, d in flagged:
    print(f"  {n}: mos={m} doob={d}")
```

## 性能影响

无性能故事（本卡为评测执行与判读配方；两套件为正确性/安全验证用途，耗时仅作执行成本参考：release 精度全量 30 条约 46s、OOM 动态插桩全量 30 条约 79s，同一平台单次实测，不构成通用预期）。

## 预防

- 白盒/精度验收一律按"双套件"设计任务：release 精度套件与动态 OOM 插桩套件是两个独立命令，不能合并也不能互相替代。
- 结果判读脚本把 `dyn_oob_result=UNKNOWN` 显式判为 invalid（而非 PASS），缺证据即 FAILED，逼着执行者回日志查成因。
- 负向用例（tiling 拒收型无 kernel 执行）的 OOB 豁免必须走 TestSpec `oob_exemptions` 声明与 known_blockers 登记，不允许口头放行。

## 验证信号

来源说明：以下信号全部来自一次 AscendC 算子白盒执行会话的实测输出——同一用例表（30 条）先跑 release 精度套件、再跑动态 OOM 插桩套件，两套结果逐条对账：

- release 精度套件（`-d false -b release`）单例试跑结果：`memory_oob_status=PASS` 但 `dyn_oob_result=UNKNOWN`、`bin_oob_result=UNKNOWN`——确认 release 二进制不产生动态 OOB 证据（来源平台：Ascend950PR；证据等级：tool_observed）。
- 动态 OOM 插桩套件（`-d true -c false -b false --compile-opts op_debug_config=oom`）全量 30 条：29 条可执行用例 `memory_oob_status=PASS` 且 `dyn_oob_result` 六项证据全为 `OK`（无越界），0 条 OOB FAIL；确认 OOB 证据须由动态插桩套件产出（来源平台：Ascend950PR；证据等级：tool_observed）。
- 同套件中唯一 `dyn_oob_result=UNKNOWN` 的用例为负向探针（tiling 阶段 `OPTILING_FAILURE`、`dyn_precision=INVALID_TILING`、kernel 未执行），与 release 套件的"未插桩型 UNKNOWN"成因不同，可按日志区分（来源平台：Ascend950PR；证据等级：tool_observed）。
- 两套结果用例集合与输入用例表精确一致（各 30 条、集合相等、无重复），确认证据覆盖全量无缺漏（来源平台：Ascend950PR；证据等级：tool_observed）。

## 失效边界

- `dyn_oob_result=UNKNOWN` 有两种成因（release 未插桩 vs tiling 拒收后 kernel 未执行），本配方靠"套件来源 + 执行日志"区分；若不区分成因直接豁免，会把插桩/执行失败误放行。
- 实测仅覆盖 Ascend950PR + CANN 9.0.0-beta.2 + 本次 TTK 检出、单个自定义算子（30 条用例）；其他平台/版本/算子未验证，推断可迁移（命令与判读规则完全算子无关、平台无关的推断，未逐一实测）。
- `op_debug_config=oom` 只检测访存越界，不覆盖耗尽主机/HBM 的资源类问题；超大 shape 用例按约定隔离在 `*_oversized.csv` 旁表不执行，避免把 HBM 耗尽与访存越界混为一谈。

## 坏实践

- 把 `--compile-opts op_debug_config=oom` 加到 `-b release` 命令上就宣称"已做内存检测"——插桩是编译期的，release 二进制不带检测，结果必然 `UNKNOWN`。
- 把 release 套件里的 `dyn_oob_result=UNKNOWN` 当 PASS 上报——`UNKNOWN` 是"无证据"，不是"无越界"。
- 只跑 OOM 套件、用它的精度列下精度结论——精度必须以 release 套件为准。
- 把 OOM（out-of-bounds memory 检测）理解为"耗尽 HBM 制造 Out-of-Memory"去构造资源压力用例——检测对象完全不同。

## 误诊

- 本会话未发生内存判读误诊（release `UNKNOWN` 被正确识别为预期、OOM 证据由动态套件补齐）；记录两个易错方向供对照：见"坏实践"前两条。
- 本会话中另有一次试跑命令失败曾被归因为"transient 管道问题"，实际是命令里路径变量笔误，与插桩机制无关（不同根因，不影响本卡结论）。


