---
type: "Runbook"
title: CANN 内置无 GRU Cell 融合反向算子（9.0.0-beta.2）
description: CANN 9.0.0-beta.2 内置注册表检索无 ThnnFusedGruCellGrad；vendors/ 下同名包属前代开发安装而非内置权威，据此裁定须自定义开发。
tags: [task.lookup, task.migration, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# CANN 内置无 GRU Cell 融合反向算子（9.0.0-beta.2）


## 现象

在为对标 PyTorch `aten::_thnn_fused_gru_cell_backward` 的 GRU 单时间步融合反向算子（IR 名 ThnnFusedGruCellGrad：输入上游梯度 grad_hy (B,H) 与前向保存的 storage (B,5H)，融合输出 5 路梯度）启动 AscendC 算子仓开发前，需要先判定 CANN 是否已有内置同名/同义算子——这决定走"直接调用"还是"自定义开发"，也决定原型文档里是否必须写「内置同名 proto 裁定」块。在开发机上执行内置注册表检索：

```
echo "ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
# → ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp

grep -rln "ThnnFusedGruCellGrad" $ASCEND_OPP_PATH/built-in/op_proto/inc/
echo "grep-exit:$?"
## → grep-exit:1（无任何命中，两次独立复测一致）
```

内置原型头文件目录里没有任何 `REG_OP(ThnnFusedGruCellGrad)`（全量 `ThnnFusedGruCell*` 检索也无命中）。但 `ls $ASCEND_OPP_PATH/vendors/` 能看到一个同名目录 `ThnnFusedGruCellGrad/`（内含 op_proto/op_impl/op_api/framework/scripts/version.info），其中 `op_proto/inc/op_proto.h` 里**确实有** `REG_OP(ThnnFusedGruCellGrad)`——很容易被误判成"内置已有此算子"。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（spec/原型提取阶段的自定义开发决策）；任何"torch 有、CANN 覆盖情况未知"的算子移植前检索同样适用。
- API 路径：`$ASCEND_OPP_PATH/built-in/op_proto/inc/`（内置 GEIR 原型注册表，grep 检索）；`$ASCEND_OPP_PATH/vendors/<算子名>/`（本机 vendor 安装区，与内置注册表是两个不同分区）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR/Ascend950DT 工具链）+ CANN 9.0.0-beta.2；验证为 host 侧安装包 grep/artifact 核验（未上板）。
- 适用粒度（transfer_scope）：operator_family——RNN cell 融合反向族（GRU/LSTM 单步融合反向算子）。
- 路线优先级：recommended_fastpath——移植前先 grep 内置注册表并区分 vendors 安装区，再决定自定义开发。
- 明确不适用 / 未验证场景：整序列动态 RNN 分步件（DynamicGRU/DynamicGRUV2Grad/GRUV2HiddenGradCell 等，输入为门张量 + 4D 布局、自带 gate_order/t_state 属性）与 THNN workspace 口径不同，不在本卡判定范围；CANN 9.0.0-beta.2 之外的版本未验证（内置覆盖可能随版本变化）。

## 判别方法

**根因（单句）**：内置注册表（`built-in/op_proto/inc/`）与本机 vendor 安装区（`vendors/`）是两个语义不同的分区——前者是官方内置交付，后者是"自己/第三方装上去的包"；检索时若不区分路径段，会把前代开发的安装基线误当成官方内置，或反之。

**probe（可执行，照抄）**：

```bash
## 1. 定位内置注册表（以 $ASCEND_OPP_PATH 实际指向为准，勿信文档/任务指令里的版本号路径）
echo "ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
find /home/<user>/Ascend -maxdepth 4 -name "op_proto" -type d

## 2. 检索内置同名 REG_OP（退出码必须取 grep 自身，勿接在管道 head 之后）
grep -rln "ThnnFusedGruCellGrad" $ASCEND_OPP_PATH/built-in/op_proto/inc/
echo "grep-exit:$?"

## 3. 检查 vendor 安装区（区分"内置"与"前代/第三方安装"）
ls $ASCEND_OPP_PATH/vendors/
ls $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/
cat $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h
```

**原始观测**（本会话实测）：
- `grep -rln ThnnFusedGruCellGrad $ASCEND_OPP_PATH/built-in/op_proto/inc/` → 空输出，`grep-exit:1`（两次独立执行一致）。
- `$ASCEND_OPP_PATH/vendors/` 列表含 `ThnnFusedGruCellGrad`（op_proto/op_impl/op_api/framework 齐备），其 REG_OP 逐字可读：`.INPUT(grad_hy, ...)` `.INPUT(storage, ...)`，输出 `grad_hx` 在首，`.ATTR(has_bias, Bool, false)`。
- `$ASCEND_OPP_PATH/built-in/op_impl/` 分区为 `ai_core/aicpu/dpu`（内置实现分区佐证，与 vendors 分离）。

**解释（interpretation）**：
- 内置注册表无同名算子 → 无「内置同名 proto 裁定」义务；接口签名以 PyTorch aten 接口为事实源锚定、以本工程 spec.yaml 为权威，走自定义开发。
- vendors/ 下的同名包是前一次开发的成品安装（待精进的前代基线），不构成内置冲突，只可作接口骨架参考；新开发须独立重做设计/实现/测试（其输出序等设计选择不可照抄，见 [proto 输出序裁定卡](proto_output_order_follows_spec_aten_order.md)）。
- 需求文档记载 CANN 对该算子交付链空白（GEIR/信息库/aclnn/Shape 推导/Tiling-Kernel/融合 PASS，另加 op-plugin、torchair 适配）且同域仅有整序列分步件与 LSTM 孪生——该多层断言属需求期记载，本会话仅对 built-in op_proto 层与 vendor 安装区做了复测。

## 设计决策

- 检索目标选 `built-in/op_proto/inc/` 而非全盘 find：内置 GEIR 原型的权威注册位置就是该头文件目录；`vendors/` 是本机安装区，二者必须分开下结论。
- 退出码取 grep 自身（`; echo "grep-exit:$?"` 紧跟 grep），不接 `| head`——管道后 `$?` 是 head 的退出码，会出现"无输出却 exit 0"的假阳性（本会话实际出现并复测修正，详见坏实践）。
- 结论落点：无内置 → 原型文档不写裁定块，按本工程 spec.yaml 生成；vendor IR 仅作参考。

## 系统信息

- toolchain：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`，x86_64-linux；`/home/<user>/Ascend/cann` 软链与 `ascend-toolkit/latest` 均指向该目录；`ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp`）。
- 目标平台：Ascend950PR / Ascend950DT（spec `supported_chips`；AddConfig("ascend950")）。
- 相关文件（本会话实际读/检索）：
  - `docs/ThnnFusedGruCellGrad/spec.yaml`、`docs/ThnnFusedGruCellGrad/REQUIREMENTS.md`（需求/规格事实源）
  - `$ASCEND_OPP_PATH/built-in/op_proto/inc/`（内置注册表，grep 检索）
  - `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h`（前代安装基线 IR，逐字参考读）
  - `docs/ThnnFusedGruCellGrad/develop/proto.md`（产出）

## 完整代码

判别脚本（可直接复跑）：

```bash
## A. 内置覆盖判定（退出码取 grep 自身）
grep -rln "ThnnFusedGruCellGrad" "$ASCEND_OPP_PATH/built-in/op_proto/inc/"
echo "grep-exit:$?"          # 1 = 内置无同名 REG_OP（这本身就是判定信号，不是失败）
ls "$ASCEND_OPP_PATH/built-in/op_impl"   # ai_core/aicpu/dpu——内置实现分区

## B. vendor 安装区区分（同名包 ≠ 内置）
ls "$ASCEND_OPP_PATH/vendors/"
cat "$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h"  # 前代基线 IR 逐字核对
```

内置判定后在产出验收里的记录（原文）：

```
5.PASS no built-in REG_OP (grep exit 1, adjudication block not required)
```

## 预防

- 移植任何"torch 有"的算子前先跑上表 probe A；`grep-exit:1` 不是命令失败，它就是"无内置"的判定信号。
- 检索前先 `echo $ASCEND_OPP_PATH` 确认实际 CANN 安装（本机多版本并存时，文档/任务指令里的版本号路径可能与环境不符——以环境变量实测为准）。
- 同名包先看路径段（`built-in/` 还是 `vendors/`）再下结论。

## 验证信号

本卡信号来自一次 GRU Cell 融合反向算子（ThnnFusedGruCellGrad）原型提取会话中的内置注册表检索与安装包核对（host 侧命令实测）：

- `grep -rln "ThnnFusedGruCellGrad" $ASCEND_OPP_PATH/built-in/op_proto/inc/` 两次独立执行均无命中、`grep-exit:1`（来源平台：950——CANN 9.0.0-beta.2 安装包 host 侧检索；证据等级：tool_observed）
- `$ASCEND_OPP_PATH/vendors/` 列表含 `ThnnFusedGruCellGrad` 目录（op_proto/op_impl/op_api/framework 齐备），其 REG_OP IR 逐字可读且第二输入实名 storage（来源平台：950——CANN 9.0.0-beta.2 安装包 vendor 区 artifact 核验；证据等级：tool_observed）
- 产出原型文档验收第 5 项 `no built-in REG_OP (grep exit 1, adjudication block not required)` PASS（来源平台：950——CANN 9.0.0-beta.2 工具链 host 侧；证据等级：tool_observed）

## 失效边界

- 结论绑定 CANN 9.0.0-beta.2 的内置注册表内容；其他 CANN 版本未验证（未验证，推断需在新版本重跑同一 probe 判定，不可直接沿用"无内置"结论）。
- `built-in/op_proto/inc/` 的覆盖判定是包级（跨 soc 共享头文件）；soc 专属信息库（`aic-<soc>-ops-info-*.json`）层面的逐 soc 覆盖本会话未复测（需求文档记载 ascend950 信息库 thnn/gru 条目为 0，属文档断言）。
- 只判定"同名/同义融合 cell 反向"这一算子族；整序列 RNN 分步件、LSTM 孪生（已有内置交付）不在本卡结论内。

## 坏实践

- 把 `grep ... | head; echo $?` 的 0 当"有内置"证据——管道退出码陷阱，本会话实测出现"无输出但 exit 0"，去掉 head 复测才得到真值 1。
- 把 `$ASCEND_OPP_PATH/vendors/<同名包>` 当成官方内置、照抄其 IR 当权威——它是前代/第三方安装基线，其设计选择（如输出序）可能与新规格冲突。
- 文档/任务指令写什么版本路径就信什么——本会话任务指令提及 9.1.0 路径，实测环境是 9.0.0-beta.2。

## 误诊

- 无。

## 未决事项

- 未追查：CANN 后续版本（如 9.1.0）是否补齐 GRU cell 融合反向的内置交付（接入新版本工具链后重跑 probe A 即可判定）。
- 未复测：需求文档记载的交付链空白中，信息库/aclnn/Shape 推导/Tiling-Kernel 各层（本会话只复测了 built-in op_proto 层与 vendor 安装区）。
