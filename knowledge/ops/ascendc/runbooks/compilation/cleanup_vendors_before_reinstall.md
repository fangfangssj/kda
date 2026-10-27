---
type: "Runbook"
title: 重装算子包前清理 OPP vendors 旧安装残留
description: 上一轮安装的 vendors/<OpType> 旧产物会污染重装后的 .o 计数与安装树核验；重装前先删该目录再构建。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 重装算子包前清理 OPP vendors 旧安装残留


## 现象

在算子仓里重建一个**曾经安装过**的自定义算子包。场景：某算子的完整实现工作流被重置，从接口桩骨架重新开始——复制样例工程、按算子原型改写 OpDef/tiling/kernel 桩之后，准备跑 `./build.sh <Pkg>_package` 构建安装。构建前检查安装环境时发现 `$ASCEND_OPP_PATH/vendors/<OpType>/` 已经存在，是**上一轮工作流留下的完整实现安装产物**：

- `op_impl/ai_core/tbe/<OpType>_impl/dynamic/op_kernel/arch35/` 下有上一轮的分支 kernel 头（`*_base.h`、`*_group.h`、`*_empty.h`）；
- `op_impl/ai_core/tbe/kernel/ascend950/<snake_case算子名>/` 下有 3 个旧内容哈希的 `.o`/`.json`（与本轮骨架将产出的新哈希完全不同）；
- 旧 `op_api/lib/libcust_opapi.so`、旧 `op_api/include/aclnn_*.h`、旧 tiling 库 `op_tiling/.../libcust_opmaster_rt2.0.so` 等全套。

本轮要安装的是接口桩骨架（arch35 下只有 `kernel.h/struct.h/tiling_struct.h` 三个头、3 个新哈希 `.o`）。安装日志显示安装器对每个子目录执行的是 "create …/vendors/<OpType>/<subdir>" + "copy new ops …" 的**合并升级**语义——新包里不存在的旧文件没有证据会被删除。若不清理直接安装，"已安装 `.o` 数 == OpDef `DataType({...})` 列表长度"的核验、安装树新旧文件核对、`ldd` 是否指向本轮产物等检查都会被旧产物干扰（按合并语义推演：旧 3 个 `.o` + 新 3 个 `.o` = 计数 6 ≠ 3，验收直接失真）。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发——`operators/<Pkg>_package` 经 `build.sh`（cmake + `make binary package` + CPack 生成 `.run` 并自动执行安装）落到 `$ASCEND_OPP_PATH/vendors/<OpType>/`。
- API 路径：`$ASCEND_OPP_PATH/vendors/<OpType>/{framework,op_proto,op_impl,op_api,scripts}` 安装布局；旧安装自带 `scripts/uninstall.sh`。
- 验证过的平台与 CANN 版本：950（ascend950，arch35 模板）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——一切安装到 OPP vendors 的自定义算子包"重建/重装"场景。
- 路线优先级：recommended_fastpath（重装前清理旧安装：一次 `rm -rf`，零成本，消除整类核验污染）。
- 明确不适用 / 未验证：首次安装（vendors 下无旧目录）无需清理；多算子共用同一 vendor 目录（package_name 撞名）未验证；其他 CANN 版本安装器语义未验证（推断可迁移，见失效边界）。

## 判别方法

构建安装前先探测旧安装是否存在、是否属于"上一轮/别的工作流"的产物：

```bash
# 列出旧安装全部文件（重点看 impl 侧 kernel 分支头与 .o 哈希）
find $ASCEND_OPP_PATH/vendors/<OpType> -type f | sort
du -sh $ASCEND_OPP_PATH/vendors/<OpType>
ls $ASCEND_OPP_PATH/vendors/   # 同名算子目录是否早于本次工作流存在
```

判别信号（任一命中即应清理）：

- `op_impl/ai_core/tbe/<OpType>_impl/dynamic/op_kernel/arch35/` 里出现当前工程没有的分支头（如 `*_base.h`/`*_group.h`/`*_empty.h`）；
- `kernel/ascend950/<snake>/` 下 `.o` 文件名的内容哈希与本次构建产物不一致；
- vendor 目录里的 `aclnn_*.h`/`libcust_opapi.so` 生成时间早于本轮构建。

## 解决配方

重装前删除旧安装目录（安装侧等价于源码侧的 `rm -rf build`），再构建安装：

```bash
rm -rf $ASCEND_OPP_PATH/vendors/<OpType> && echo "stale vendor removed"
cd <算子仓>/operators && ./build.sh <Pkg>_package > /tmp/<OpType>_build.log 2>&1; st=$?; echo "BUILD_EXIT=$st"; tail -25 /tmp/<OpType>_build.log
```

安装完成后核对安装树只含本轮产物：`.o` 计数 == OpDef `DataType({...})` 列表长度；impl/dynamic 的 arch35 目录下无上一轮分支头残留。

## 设计决策

- **为何 `rm -rf` 而不是指望安装器覆盖**：安装日志对每个子目录只打印 "create + copy new ops"（合并/升级语义），没有任何"先清空目录"的证据；直接删除是唯一可确定干净状态的手段。
- **备选**：执行旧安装自带的 `scripts/uninstall.sh`——语义上应等价，本例未对比实测；选择 `rm -rf` 因其更直接、不依赖脚本行为。
- **时机**：放在 `build.sh` **之前**（build.sh 末尾就会执行 `.run` 安装）；装完再清会误删刚装好的产物。
- 当前方案属 recommended_fastpath：一次清理换取全套核验的确定性。

## 系统信息

- CANN：9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`，`ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp`），cmake 3.20.5。
- 目标平台：ascend950（arch35 模板；`.o` 安装于 `op_impl/ai_core/tbe/kernel/ascend950/<snake>/`）。
- 相关目录：`operators/<Pkg>_package/`（源工程）、`build/custom_opp_ubuntu_x86_64.run`（CPack 产物）、`$ASCEND_OPP_PATH/vendors/<OpType>/`（安装目标）、`build.sh`（清理 build/ → cmake → make → 安装）。

## 完整代码

探测 + 清理 + 构建安装 + 安装树核对的完整命令（实测通过形态）：

```bash
## 1) 探测旧安装（实测列出上一轮完整实现的 base/group/empty 头与 3 个旧哈希 .o）
find $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad -type f | head -50
du -sh $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad

## 2) 清理 + 构建安装（长超时执行、日志落盘看尾部）
rm -rf $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad && echo "stale vendor removed"; \
cd /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/operators && \
./build.sh ThnnFusedGruCellGrad_package > /tmp/ThnnFusedGruCellGrad_build.log 2>&1; st=$?; \
echo "BUILD_EXIT=$st"; tail -25 /tmp/ThnnFusedGruCellGrad_build.log

## 3) 安装树核对（本轮产物唯一性）
find $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel -name "*.o" | wc -l
##   期望 == OpDef DataType({...}) 列表长度（本例 3：fp32/fp16/bf16）
ls $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/ThnnFusedGruCellGrad_impl/dynamic/op_kernel/arch35/
##   期望仅剩本轮工程的头文件，无上一轮分支头
```

## 预防

- 每次重装自定义算子包前，把 `ls $ASCEND_OPP_PATH/vendors/ | grep <OpType>` 当作构建前置检查项。
- 工作流重置/重试（骨架重来）时，把清理旧 vendor 安装写进第一步。
- 凡以 `.o` 计数 / 安装树内容 / `ldd` 路径做验收的门，先确认安装目录是本轮独占的。

## 验证信号

以下信号来自一次"样例工程重建为接口桩骨架"的开发会话：构建前实测探测到旧安装，删除后构建安装并跑完产物/符号/冒烟全套核验。

- 旧安装实测存在：`find` 列出 `thnn_fused_gru_cell_grad_base.h`/`_group.h`/`_empty.h`、3 个旧哈希 `.o`（`*_024ea6c4…`/`*_1e043a7d…`/`*_5580799…`）及旧 `libcust_opapi.so`（来源平台：950；证据等级：tool_observed）。
- 清理后构建一次通过：`BUILD_EXIT=0`，`.run` 生成并安装 SUCCESS，安装日志逐子目录 create + copy new（来源平台：950；证据等级：tool_observed）。
- 安装树唯一性：impl/dynamic 的 arch35/ 下仅剩骨架三头文件（kernel.h/struct.h/tiling_struct.h），无 base/group/empty 残留（来源平台：950；证据等级：tool_observed）。
- 已安装 `.o` 计数 == 3 == OpDef DataType 组合数（fp32/fp16/bf16），且 3 个 `.o` 的 mocker 串计数均为 1（来源平台：950；证据等级：tool_observed）。
- aclnn 冒烟 3/3 PASSED，`ldd` 解析到本轮 vendor 安装的 `libcust_opapi.so`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 安装器对**已存在** vendor 目录的合并/覆盖语义未实测（本例先删后装）；"不清理会怎样"的失败形态是机制推演，不是本 trace 的实测信号。
- 多算子共享同一 `vendors/<OpType>` 目录的场景未验证。
- 其他 CANN 版本的安装器行为未验证——"重装前清理"作为通用实践推断可迁移（未验证，推断可迁移）。

## 坏实践

- 不清理直接重装，然后按 `.o` 计数/安装树内容/`ldd` 路径做验收——旧产物会让计数失真或混入上一轮文件（本卡以先删后装规避，失败形态未实测）。
- 装完再 `rm -rf` vendor 目录——会把本轮刚装好的产物一并删掉。

## 误诊

- 开发中曾假设"新 `.run` 包安装会整体替换 vendor 目录"——安装日志的 create + copy new 措辞表明是合并式升级，该假设不成立（agent 分析，未实测对照，见失效边界）。

## 未决事项

- 优先项：实测安装器对已存在 vendor 目录的行为（覆盖 vs 合并）——确认不清理时旧 `.o`/分支头是否实际留存、`.o` 计数是否翻倍，把本卡的"合并污染"从推演升级为实测结论。
- `scripts/uninstall.sh` 与 `rm -rf` 的等价性未对比实测。
