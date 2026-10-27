---
type: "Runbook"
title: 561103 — vendor 未注册 load_priority 不被加载
description: CANN 按 opp/vendors/config.ini 的 load_priority 加载自定义算子 vendor；手动 cp 或 .run --install-path 安装不更新该列表，vendor 不被加载，aclnn GetWorkspaceSize 报 561103；标准安装（不带 --install-path）或手工补登后恢复。
tags: [task.debugging, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-16T10:29:17Z'
platforms: ["950"]
---

# 561103 — vendor 未注册 load_priority 不被加载

## 现象

自定义 ACLNN 算子（TopK 类，Ascend950PR / arch35，CANN 9.1.0）的 `custom_opp_ubuntu_x86_64.run` 包已能通过共享库验证，先用 `--install-path=$ASCEND_HOME_PATH` 安装（SUCCESS），随后跑 C++ ST Real 模式（两段式 aclnn 接口，用例为 fp16 2D 末维 topk）：CPU Golden 自测全过，但两个真实 NPU 用例均在第一阶段失败：

```text
[Real] 基础2D fp16 末维 topk(k=10) 最大值
  [FAIL] GetWorkspaceSize 失败: 561103
  结果: FAIL
```

期间还有两个易踩的环境事实：`--install-path=$ASCEND_HOME_PATH` 实际把包落到了 `<CANN>/vendors/`（而不是 `<CANN>/opp/vendors/`），改用 `--install-path=$ASCEND_HOME_PATH/opp` 后文件位置才对，但 561103 依旧。

同一根因的两种变体在另外两个独立会话复现：

- GQA 会话（手动 cp 安装）：`.run` 安装器反复挂起后改用"卸载旧版 + 手动 cp vendor 目录到 opp/vendors/"——文件树与包内完全一致，但 6/6 用例全部 `GetWorkspaceSize 失败: 561103`（此前同一产物用 `.run --force` 安装时可正常调用）。
- LSTM 会话（标准安装 + 手工补登）：安装后 `config.ini` 的 `load_priority` 不含本算子 vendor，ST Real 的 CMake 找不到 vendor include；把 vendor 名追加进 `load_priority` 并设置 `LD_LIBRARY_PATH` 后 ST Real L0 4/4 PASS（NPU，y MERE=8.74e-07）。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子仓的 run 包安装 + C++ ST（aclnn 两段式接口，GetWorkspaceSize/Execute）验证；安装器不可用/挂起时手动铺 `opp/vendors/<vendor>` 目录的部署调试。
- API 路径：`custom_opp_*.run` 的 `install.sh`（targetdir/config.ini 分支逻辑）、`opp/vendors/config.ini` 的 `load_priority`、`ASCEND_CUSTOM_OPP_PATH`（set_env.bash）、kernel 动态加载（查 `op_impl/ai_core/tbe/kernel/config/<soc>/<op>.json` 时按已注册 vendor 逐目录查找）；aclnn GetWorkspaceSize 返回码 561103（`op_errno.h` 中 `ACLNN_ERR_INNER_NULLPTR`）。
- 验证过的平台与 CANN 版本：950（Ascend950PR 实机 ST）+ CANN 9.1.0 / 9.2.0。
- 适用粒度：platform_wide——所有以 run 包分发或手动部署的自定义算子安装。
- 路线优先级：recommended_fastpath（不带 --install-path、由 ASCEND_OPP_PATH 决定目标并自动注册 config.ini；手动部署则必须补登 load_priority）。
- 明确不适用/未验证场景：config.ini 已含 vendor 仍 561103 的情形（需查 kernel json 与 SOC 匹配等，另见 [561103 同名算子签名冲突卡](561103.md)等其他 561103 根因）；非 run 包（直接拷贝 vendors 目录）以外的注册方式未逐一验证；多 vendor 同名算子的优先级覆盖语义未验证。

## 根因

单句归因：CANN 运行期按 `opp/vendors/config.ini` 的 `load_priority` 列表逐 vendor 查找算子实现；vendor 目录文件齐全但名字不在列表中即被整体跳过，L0 API 的 `ADD_TO_LAUNCHER_LIST_AICORE` 失败返回 nullptr，aclnn 层以 561103（ACLNN_ERR_INNER_NULLPTR）暴露。三种安装方式只有标准安装会写注册表：

机制链（日志 + 包内脚本逐环验证）：

1. 错误码定性：`grep 561103` 于 `include/aclnn/opdev/op_errno.h` → `#define ACLNN_ERR_INNER_NULLPTR 561103`；op_api 侧 CheckNotNull 失败返回的是 161002 族参数错，561103 的实际来源是 L0 API 的 `ADD_TO_LAUNCHER_LIST_AICORE` 失败返回 nullptr。
2. 开日志复跑（`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1`）：`[GetCustomImplPath] ASCEND_CUSTOM_OPP_PATH is null.`，且 kernel json（如 `bench_no_harness_top_k.json`）在全部已注册 vendor 目录逐个查找失败（`is invalid or does not exist, continue next priority`）——查找列表里根本没有本算子的 vendor 目录。
3. 注册表核对：`cat $ASCEND_HOME_PATH/opp/vendors/config.ini` 的 `load_priority=<...>` 不含本算子 vendor。
4. 安装器机制（读包内 `build/scripts/install.sh`）：指定 `--install-path` 时走 `targetdir=${INSTALL_PATH}` 分支，只写 set_env.bash、不更新 `<targetdir>/vendors/config.ini` 的 `load_priority`；不指定时走 `targetdir=${ASCEND_OPP_PATH}` 分支，用 `sed` 把 vendor 名登记进 `load_priority`。手动 cp 只复制文件、天然不更新注册表。
5. 即：`--install-path` 分支 = "便携安装"语义（要求 source set_env.bash 提供 `ASCEND_CUSTOM_OPP_PATH` 才能被加载）；手动 cp = 未注册。两条加载通路（config.ini 注册、ASCEND_CUSTOM_OPP_PATH 环境变量）都没接上时，vendor 文件再齐全也不被加载。

## 解决配方

**标准安装（让安装器自动注册 config.ini）**：不带 `--install-path`，让 install.sh 用环境变量 `ASCEND_OPP_PATH`（= `$ASCEND_HOME_PATH/opp`）作 targetdir，走 config.ini 更新分支：

```bash
# 确认目标根
echo $ASCEND_OPP_PATH    # 应指向 <CANN>/opp
cd <op>/build && bash custom_opp_ubuntu_x86_64.run --quiet
# 安装尾部提示为 export LD_LIBRARY_PATH=...（而非 source set_env.bash）即走了 config.ini 分支

# 验证注册与 kernel 配置
grep -o "<vendor_name>" $ASCEND_HOME_PATH/opp/vendors/config.ini
ls $ASCEND_HOME_PATH/opp/vendors/<vendor_name>/op_impl/ai_core/tbe/kernel/config/<soc>/
```

Before（561103，`--install-path` 安装）：

```text
load_priority=bench_no_harness_moe_re_routing_custom,custom_ops,...,cummin_custom,BenchSoftmax
（不含 bench_no_harness_top_k_custom；日志：ASCEND_CUSTOM_OPP_PATH is null + 各 vendor 下 <op>.json 查找失败）
```

After（标准安装后自动改写）：

```text
load_priority=bench_no_harness_top_k_custom,bench_no_harness_moe_re_routing_custom,custom_ops,...,BenchSoftmax
（首部追加本算子 vendor；kernel 目录下 <op>.json 与 binary_info_config.json 存在）
```

**手动部署时必须补登 load_priority**（`.run` 安装器挂起/不可用时的替代路径，幂等）：

```bash
CONFIG=$ASCEND_OPP_PATH/opp/vendors/config.ini
VENDOR=<vendor_name>
if ! grep -q "$VENDOR" "$CONFIG"; then
    sed -i "s/$/,$VENDOR/" "$CONFIG"
    echo "Added to config.ini"
fi
```

**运行环境**：按安装器提示 `export LD_LIBRARY_PATH=<CANN>/opp/vendors/<vendor>/op_api/lib/:${LD_LIBRARY_PATH}`（或 source vendor `bin/set_env.bash`）。

判别安装走了哪个分支的快捷信号：安装尾部 `using requirements` 提示 `source .../bin/set_env.bash` → set_env 分支（未注册）；提示 `export LD_LIBRARY_PATH=...` → config.ini 分支（已注册）。

## 设计决策

- 为什么选"不带 --install-path 重装"而不是手工改 config.ini 或 source set_env.bash：手工改注册表绕过安装器、重复安装时语义不明；source set_env.bash 依赖每个运行环境记得 source，CI/多人环境下易漏；由安装器走标准分支注册最稳。两条替代通路机制上均成立（手动 cp 会话以 sed 补登闭环、GQA 会话验证），本轨迹未作为标准修复验证。
- `--install-path=$ASCEND_HOME_PATH` 与 `=$ASCEND_HOME_PATH/opp` 的落点差异（vendors/ vs opp/vendors/）说明 `--install-path` 语义是"自定义安装根"，目录拼接不含 opp——用该参数必须自行对齐目录结构，且仍不注册 config.ini。
- 手动部署会话用 `grep -q` 守卫 + `sed` 追加，保证幂等（重复执行不会重复登记）；同时把包内 `scripts/` 目录一并补齐（卸载脚本等安装器产物）。
- 没有选择反复重试挂起的 `.run` 安装器：其挂起原因（交互/环境）未定位，调试主线优先，手动部署 + 注册表补登是确定可行路径。

## 判别方法

- 触发信号：全量用例统一死在 `GetWorkspaceSize` 阶段且返回 561103（与 kernel 内部错误无关的错误位置）；此前用过 `--install-path` 安装或手动铺过 vendor 目录。
- 探针（按序）：

```bash
# 1) 确认错误码含义
grep -rn "561103" $ASCEND_HOME_PATH/include/aclnn/opdev/op_errno.h

# 2) 开日志看 kernel 查找路径（关键行）
ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./<st_bin> 2>&1 | \
  grep -E "ASCEND_CUSTOM_OPP_PATH|ParseDynamicKernelConfig|is invalid or does not exist"

# 3) 核对注册表
cat $ASCEND_HOME_PATH/opp/vendors/config.ini

# 4) 手动部署时对账文件树（证明文件本身没铺错）
diff -r <构建产物>/packages/vendors/<vendor> $ASCEND_OPP_PATH/opp/vendors/<vendor>
```

- 判读：日志显示 `ASCEND_CUSTOM_OPP_PATH is null.` 且 `<op>.json` 在各 vendor 目录逐一查找失败 + config.ini 无本算子 vendor → 即本卡根因；文件树 diff 一致可排除"铺错文件"。

## 验证信号

段首来源说明：以下信号来自三个独立算子会话（TopK/Unique 会话的 --install-path 安装与标准重装对照、GQA 会话的手动 cp 部署与补登对照、LSTM 会话的三步部署验收），平台 950，CANN 9.1.0 / 9.2.0。

- TopK 会话：修复前 ST 两用例均 `[FAIL] GetWorkspaceSize 失败: 561103`；标准方式重装后 `config.ini` 含本算子 vendor（时间戳更新），kernel 目录下 `bench_no_harness_top_k.json`、`binary_info_config.json`、kernel `.o/.json` 存在；同一 ST 再跑 561103 消失，aclnn 两段式调用成功进入精度比对阶段（来源平台：950 实机 ST；证据等级：tool_observed）。
- GQA 会话（手动 cp 变体）：手动 cp 安装后 6/6 用例 `GetWorkspaceSize 失败: 561103`（此前 `.run --force` 安装同一产物时可调用）；`cat opp/vendors/config.ini` 显示 load_priority 长列表不含本 vendor，SRC/DST 文件树 diff 一致；vendor 名追加进 load_priority 后重跑 ST，561103 全部消失、错误转入下游 507035（算子已可加载执行）（来源平台：950；证据等级：tool_observed）。
- LSTM 会话（三步部署验收）：安装输出逐项 `upgrade framework/op_proto/op_impl/op_api` 且末尾 `SUCCESS`，`opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so` 出现；`config.ini` 追加 vendor 名后，ST Real 的 CMake 配置日志显示 `Found custom operator include dir: .../opp/vendors/<op>_custom/op_api/include` 并链接成功；ST Real L0 4/4 PASS（NPU 上真实执行，y MERE=8.74e-07 远低于 mere<0.05 阈值），UT 26/26 与独立清理重建复验 Real 4/4 复现同一部署下的可重复性（来源平台：950；证据等级：tool_observed）。
- 安装器脚本证据：包内 `build/scripts/install.sh` 的 `if [ -n "${INSTALL_PATH}" ]` 分支只写 set_env.bash，else 分支才 `sed` 更新 config.ini（artifact 读取事实）（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- 本卡只覆盖"vendor 未注册 load_priority → 561103"这一根因；若 config.ini 已含 vendor 仍 561103，需查 kernel json 与 SOC 是否匹配（binary_info_config.json 的 soc 段）等，不属本卡（561103 还有多张独立根因卡：[同名算子签名冲突](561103.md)、[CheckBroadcastShape 过严](561103_checkbroadcastshape.md)、[isGroup 哨兵路由失配](561103_isgroup_1_tilingkey.md)）。
- `--force`（跳过共享库校验）与注册行为正交：校验通过 ≠ 已注册，安装 SUCCESS 也不代表被运行期加载——注册与否要看 config.ini 与安装尾部提示分支。
- 手工改 config.ini、source set_env.bash（ASCEND_CUSTOM_OPP_PATH 通路）机制上均成立，前者在 GQA 会话作为修复验证、后者未作为修复验证。
- install.sh 的分支逻辑随 CANN 版本可能调整（本轨迹 9.1.0/9.2.0）；升级后应以包内 scripts/install.sh 实际代码为准。
- 卸载错误位置的安装时未检查权限（uninstall.sh 报 Permission denied 残留目录）。
- `cp -r $SRC $DST` 当 `$DST` 已存在时会造成 `<DST>/<vendor>/...` 目录嵌套——铺完必须 `find` 核对目标层级（GQA 会话实际踩到并修正）。

**未决项**：

- `config.ini` 的 `load_priority` 顺序语义（多 vendor 同名算子冲突时的优先级/覆盖规则）未查证。
- `.run` 安装器挂起的原因（疑似等待交互输入或文件锁）未定位；定位前手动部署 + 本卡补登步骤是替代路径。
- 可向上游反馈：`--install-path` 分支不更新 config.ini 且安装输出仍报 SUCCESS，建议在提示中显式说明需 source set_env.bash 才能被运行期加载。
## 坏实践

- `--install-path=$ASCEND_HOME_PATH` 直接装：包落到 `<CANN>/vendors/`（错误位置），且不注册。
- 手动铺完 vendor 目录直接跑测试，把 561103 当代码回归排查（浪费整轮构建）。
- 只看安装输出 `SUCCESS` 就认为可运行——注册与否要看 config.ini 与安装尾部提示分支。
- 全量用例统一死在 GetWorkspaceSize 且 561103 时，先怀疑部署/注册（config.ini、文件树），不要先怀疑代码。
- 曾以为"vendor 目录存在 + 文件齐全"即可被加载——CANN 只按 config.ini 的 load_priority 逐目录查找，未注册目录即使文件齐全也被跳过。

## 误诊

- 初见 561103 按"参数校验失败"思路查 op_api 的 CheckNotNull/dtype 校验——错误码 561103 属 ACLNN_ERR_INNER_NULLPTR（内部空指针），与 161002 族参数错不同源；顺着 L0 API 的 ADD_TO_LAUNCHER_LIST_AICORE 才走到 kernel 加载层。
- GQA 会话曾怀疑手动安装遗漏 `scripts/` 目录或 version.info 导致 561103：文件树对账证明结构一致，补登 load_priority 后即恢复，排除。
