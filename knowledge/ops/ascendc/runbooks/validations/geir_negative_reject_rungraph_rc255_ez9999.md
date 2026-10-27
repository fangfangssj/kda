---
type: "Runbook"
title: GEIR 负向拒绝签名 — RunGraph rc=255 EZ9999
description: "GEIR 图模式负向非法输入在图执行阶段以 PRECISION_STATUS: ERROR（RunGraph failed -1, EZ9999 Inner Error）真实拒绝且 Kernel 未执行；勿当框架故障，勿把 PASS/SKIP/INVALID/精度 FAIL 当拒绝。"
tags: [task.testing, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GEIR 负向拒绝签名 — RunGraph rc=255 EZ9999


## 现象

在验收一个自定义算子（对标 torch GRU Cell 反向、按算子仓流程开发）GEIR 图模式实现时，验收方对两套负向执行产物（源码静态/源码动态，各 49 条非法输入用例——非法 dtype/shape/属性组合等）做"真实拒绝"核验。验收口径要求：两种通路全部在 GEIR 执行阶段真实拒绝，不接受 PASS/SKIP/INVALID/普通精度 FAIL，且 reject node 未生成目标 Kernel。日志中每条负向用例的判定行原文（逐字）：

```
PRECISION_STATUS: ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!
```

结果 CSV 中对应 precision=EXEC_FAILURE、precision_status 以 `ERROR:` 开头。表面看 `EZ9999: Inner Error!` 像图引擎内部故障/环境问题，第一反应容易登记成框架阻断或去排查环境；但同一环境、同一轮的四套正向用例 44/44 全 PASS 且 Kernel bin 正常生成下发——说明这不是全局性框架故障，而是负向用例被图执行阶段拒绝的预期形态。当时单套日志 5.3 GB（含二进制字节，grep 需 `-a`），日志设备行 `| NPU:0 Ascend950PR_9579 |`。

## 适用场景（必填 — 检索前的第一道滤网）

- **工程形态**：算子仓（ops-registry-invoke）流程 GEIR 图模式的 L2 负向（非法输入）验收与结果判读。
- **API 路径**：TTK GEIR runner（`run_ttk_geir.sh` 六套执行，负向套件以 `--golden-mode Disable` 运行）→ `verify_ttk_geir.py check-result --expect reject`（结果 CSV 判定）→ GE `RunGraph` 执行日志。
- **验证过的平台与 CANN 版本**：Ascend 950（Ascend950PR_9579，NpuArch 3510）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度（transfer_scope 五档取最窄）**：`platform_wide`——该平台/版本上自定义算子 GEIR 负向验收的判读。
- **路线优先级**：`recommended_fastpath`（官方 check-result + 窄口径日志分布 + Kernel 证据对照的三重核验）。
- **明确不适用 / 未验证场景**：aclnn 直调模式的负向报错形态（本会话未观测，两模式形态不可互相套用）；binary 通路负向未跑（该验收口径明确负向仅跑源码静态/动态两套）；若负向出现部分拒绝部分 PASS，须逐 case 归因，不能整体放行。

## 判别方法

1. **官方判定（结果 CSV 口径）**——reject 合同为 precision==`EXEC_FAILURE` 且 precision_status 以 `ERROR:` 开头、pass=0、skip=0、拒绝数==total：

```bash
python3 skills/geir-route-tester/scripts/verify_ttk_geir.py check-result \
  --cases "$GEIR/negative.csv" --result "$GEIR/results/negative-source-static.csv" \
  --expect reject --mode source --shape static
```

   实测输出：`PASS geir-result expect=reject {'total': 49, 'pass': 0, 'fail': 49, 'skip': 0}`。**注意语义反转**：expect=reject 时 `fail=49` 才是全部拒绝的通过态，`pass`/`skip` 必须为 0——SKIP（含 soc_not_support/disabled）与 INVALID 都不算真实拒绝。
2. **日志侧窄口径分布核对**（GB 级日志勿用跨词宽正则）：

```bash
for f in negative-source-static negative-source-dynamic; do
  grep -o "PRECISION_STATUS: [A-Za-z_]*" "$GEIR/logs/$f.log" | sort | uniq -c
done
```

   实测两套均只有 `49 PRECISION_STATUS: ERROR`（无任何 PASS/SKIP/INVALID 行）。
3. **Kernel 未执行的对照核验**：负向拒绝发生在 RunGraph 阶段（图执行失败），Kernel 不应执行。**注意探针的判别力**：`grep -ac "AddSoToRegistry.*op_kernel.*<OpType>"` 在负向日志为 0 匹配，但同一模式在**正向**日志也是 0 匹配（kernel so 注册不产生该日志形态）——该探针不具判别力，不能单独作证。有判别力的对照是正向日志中真实存在的 Kernel 证据形态：

```bash
# 正向日志（对照组，应有多条）
grep -a "SetKernelBin\|LoadTBEKernelBinToOpDesc" "$GEIR/logs/positive-source-static.log" | head -5
## 负向日志（应无匹配——本会话未跑此对照，属推荐补测项，见遗留问题）
grep -ac "SetKernelBin\|LoadTBEKernelBinToOpDesc" "$GEIR/logs/negative-source-static.log"
```
4. **同环境正向对照**：同轮四套正向 check-result 全 PASS（各 44/44、fail=0、skip=0）且 Kernel bin 生成下发非空（394320 字节）——证明拒绝不是环境/框架全局故障。

## 根因

GEIR 图模式下，负向用例的非法输入在图执行阶段被拒后，GE 以 RunGraph 失败统一上浮：rc=255、`RunGraph failed: -1`、`EZ9999: Inner Error!`——具体的算子侧校验错误信息不随该错误行透出。这是"设计要求的拒绝行为"在图执行日志里的预期形态（验收口径所称"在 GEIR 执行阶段真实拒绝"），不是框架故障；同环境正向 44/44 跑通排除全局故障假设。更深层机制（host 侧校验异常如何逐层映射为 RunGraph rc=255 + EZ9999）本会话未逐层验证，见遗留问题。

## 设计决策

- **为何以 EXEC_FAILURE + `ERROR:` 前缀 + skip=0 为唯一合格拒绝形态**：PASS=没拒绝；SKIP/INVALID（校验器 INFRA_PRECISION 集合：GOLDEN_FAILURE/INVALID/UNSUPPORTED/SOC_NOT_SUPPORT）= 基础设施口径逃逸，无法区分"用例非法被拒"与"环境没跑"；普通精度 FAIL=进了 Kernel 算错，都不是"在 GEIR 执行阶段真实拒绝"。
- **放弃的候选**：把负向也跑 binary 通路——该验收口径明确负向仅源码静态/动态两套（拒绝语义在 host 校验层，binary 编译产物不改变判定入口），放弃。
- **当前方案属哪档路线**：`recommended_fastpath`（三重核验：官方 CSV 判定 + 日志窄口径分布 + 正向对照）。

## 系统信息

- 产物布局：`operators/<Op>_package/tests/geir/{logs/*.log（4.6~5.3 GB/套，含二进制字节）, results/*.csv, negative.csv（冻结负向用例表）, projection.json, baseline.json}`。
- 校验器：`verify_ttk_geir.py check-result`（reject 合同实现于 check_result 的 expect=="reject" 分支：`precision == "EXEC_FAILURE" and precision_status.startswith("ERROR:")`，且 `stats["pass"] or stats["skip"]` 或拒绝数不足即 ValueError）。
- runner：`run_ttk_geir.sh`——负向套件以 `--golden-mode Disable` 运行（预期拒绝不做 golden 对比）；每套先 `rm -f` 旧结果 CSV 再执行（防旧成功结果掩盖 TTK crash）。
- 平台：Ascend 950（Ascend950PR_9579，NpuArch 3510）、CANN 9.0.0-beta.2。

## 完整代码

可执行核验命令序列（bash；`GEIR=operators/<Op>_package/tests/geir`，日志 GB 级须 `grep -a`）：

```bash
## 1) 官方拒绝判定（两套：static / dynamic；语义反转：fail=total 才是通过）
python3 skills/geir-route-tester/scripts/verify_ttk_geir.py check-result \
  --cases "$GEIR/negative.csv" --result "$GEIR/results/negative-source-static.csv" \
  --expect reject --mode source --shape static
python3 skills/geir-route-tester/scripts/verify_ttk_geir.py check-result \
  --cases "$GEIR/negative.csv" --result "$GEIR/results/negative-source-dynamic.csv" \
  --expect reject --mode source --shape dynamic

## 2) 日志侧窄口径分布（应仅 ERROR；勿用跨词宽正则）
for f in negative-source-static negative-source-dynamic; do
  grep -o "PRECISION_STATUS: [A-Za-z_]*" "$GEIR/logs/$f.log" | sort | uniq -c
done

## 3) 拒绝行原文抽查
grep -a "PRECISION_STATUS: ERROR" "$GEIR/logs/negative-source-static.log" | head -5

## 4) Kernel 证据对照（正向应有多条；负向应 0——有判别力的探针）
grep -ac "SetKernelBin\|LoadTBEKernelBinToOpDesc" "$GEIR/logs/positive-source-static.log"
grep -ac "SetKernelBin\|LoadTBEKernelBinToOpDesc" "$GEIR/logs/negative-source-static.log"
```

实测关键输出（同轮 Ascend 950 + CANN 9.0.0-beta.2，逐字）：

```
PASS geir-result expect=reject {'total': 49, 'pass': 0, 'fail': 49, 'skip': 0}
     49 PRECISION_STATUS: ERROR          # 两套负向日志的 uniq -c 输出
PRECISION_STATUS: ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!
```

## 预防

- 负向验收前先固定"合格拒绝形态"判定口径（EXEC_FAILURE + `ERROR:` 前缀、pass=0、skip=0），避免把 INVALID/SKIP/普通精度 FAIL 混入"拒绝"计数。
- 校验脚本对 expect=reject 的输出做语义反转提示（fail=total 是通过态），防止读数误判。
- GB 级日志统计一律用 `grep -o` 提取-去重的窄口径，勿用跨词宽正则。

## 验证信号

来源说明：以下信号来自一次 GEIR 图模式实现的独立验收会话——验收方对两套负向执行产物（各 49 条非法输入用例）复跑官方校验器并对 5.3 GB 原始日志做窄口径 grep 核对；全部为同一轮 Ascend 950 + CANN 9.0.0-beta.2 实测产物。

- check-result 两套 expect=reject 均 PASS：`{'total': 49, 'pass': 0, 'fail': 49, 'skip': 0}`（negative-source-static / negative-source-dynamic）（来源平台：950；证据等级：tool_observed）。
- 日志窄口径分布：两套均为 49×`PRECISION_STATUS: ERROR`，0 PASS/SKIP/INVALID（来源平台：950；证据等级：tool_observed）。
- 拒绝行原文含 `GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!`（来源平台：950；证据等级：tool_observed）。
- 同轮正向对照：四套 check-result 全 PASS（各 44/44、fail=0、skip=0），正向日志 Kernel bin 生成并加载 394320 字节（`SetKernelBin` → `LoadTBEKernelBinToOpDesc`）——同环境可正常执行，排除框架全局故障（来源平台：950；证据等级：tool_observed）。
- （负向日志的 Kernel 证据对照 grep：`AddSoToRegistry.*op_kernel.*<OpType>` 0 匹配——但该模式正向日志同为 0 匹配，不具判别力，仅作记录不作证据；来源平台：950；证据等级：tool_observed。）

## 失效边界

- 仅实测源码静态/动态两套、Ascend 950 + CANN 9.0.0-beta.2；binary 通路负向未跑（验收口径不要求），aclnn 直调模式负向形态未观测。
- 本卡证明"拒绝发生在图执行阶段（RunGraph 失败、Kernel 不执行）"；"拒绝发生在 Kernel 编译之前（如 InferShape/InferDataType 阶段）"是验收记录的归因，本会话未独立复证。
- 若个别负向用例拒绝形态不同（如部分 PASS、部分 INVALID），须逐 case 归因，本卡签名不适用。
- 负向套件以 `--golden-mode Disable` 运行：该签名是执行层拒绝，不含 golden 对比层信息。

## 坏实践

- 见 `EZ9999: Inner Error!` 即判框架/环境故障并登记阻断放行——同环境正向 44/44 跑通 + 全部用例统一 RunGraph 失败形态证明这是校验拒绝的预期表现。
- 用宽正则（如 `PRECISION_STATUS:.*PASS|precision.*PASS|GEIR.*PASS`）在 GB 级日志里数状态——实测误配 1862 条无关行，得出"负向日志里有大量 PASS"的假象。
- 把 SKIP/INVALID 计入"已拒绝"（reject 合同要求 skip=0）。
- 用不具判别力的探针（正/负向日志同为 0 匹配的模式）作"无 Kernel 下发"的唯一证据。

## 误诊

- 登记（转述，文档记载、本会话未复现）：同项目 TTK kernel 模式下负向用例的拒绝形态不同——非法 shape 在 tiling 阶段以 OPTILING_FAILURE/shape_mismatch 拒绝，且框架不识别 expect_error 标记会把预期失败判成 FAIL（known blockers 台账 KB-1 条目记载）。kernel 模式与 GEIR 图模式的拒绝签名不可互相套用。
- 本会话实测中的误诊险情：宽正则在负向日志命中 1862 条含 "PASS" 字样的无关行，一度像是"负向里有大量 PASS"——窄口径提取-去重后真实分布为 49×ERROR。


