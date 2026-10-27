---
type: "Runbook"
title: 共享只读 CANN 上 .run 安装权限失败重定向
description: 共享 CANN 安装的 opp/vendors 为 root 属主，自定义算子 build.sh 编译打包成功但 .run 自安装 chmod/mkdir 全被 Permission 拒绝致退出 1；export ASCEND_CUSTOM_OPP_PATH 重定向到可写目录即可，且目录必须先 mkdir -p 预创建（install.sh 自动建目录分支传空路径会报 dir path is empty）。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-16T10:24:39Z'
platforms: ["950"]
---

# 共享只读 CANN 上 .run 安装权限失败重定向

## 现象

在共享 CANN 安装（多用户共用、`opp/vendors` 由 root 属主、模式 drwxrwxr-x，当前用户既非 root 也不在 root 组）上开发 Ascend950 自定义算子：`source set_env.sh` 后执行 `cd operators && ./build.sh <Op>_package`，编译与打包全部成功（`.run` 包已生成、kernel `.o`/`.json` 齐全），但最后的自安装步骤（build.sh 直接执行 `"$RUNFILE"`）失败退出 1。失败输出原文（节选，同一模式重复数十行，含他人安装的算子目录）：

```
chmod: changing permissions of '<CANN安装根>/opp/vendors/SoftplusV2Grad/op_impl/ai_core/tbe/kernel/ascend950/softplus_v2_grad': Operation not permitted
chmod: changing permissions of '<CANN安装根>/opp/vendors/SoftplusV2Grad/op_api/lib/libcust_opapi.so': Operation not permitted
[ops_custom] [ERROR] chmod <CANN安装根>/opp/vendors 750 failed.
[ops_custom] [INFO] create <CANN安装根>/opp/vendors/BenchHarnessApplyRotaryPosEmb.
mkdir: cannot create directory '<CANN安装根>/opp/vendors/BenchHarnessApplyRotaryPosEmb': Permission denied
[ops_custom] [ERROR] create dir <CANN安装根>/opp/vendors/BenchHarnessApplyRotaryPosEmb failed.
EXIT=1
```

换用 `export ASCEND_CUSTOM_OPP_PATH=/tmp/bh_opp` 重定向安装目标后，安装仍失败，但报错变成两条更迷惑的（当时目录尚未创建）：

```
[ops_custom] [ERROR] dir path is empty
[ops_custom] [ERROR] /tmp/bh_opp no exist
=== build.sh exit=1 ===
```

由于 build.sh `set -e`，安装失败使整体退出码为 1，即使编译产物完好。

## 适用场景

- 工程形态：CANN 自定义算子包构建（`cmake + make binary package` 产出 makeself 格式 `custom_opp_ubuntu_x86_64.run`，build.sh 末尾自动执行 `.run` 安装）。
- API 路径：`.run` 内嵌 `install.sh` 的目标目录判定逻辑；环境变量 `ASCEND_CUSTOM_OPP_PATH`；安装器参数 `--install-path=<绝对路径>`。
- 验证过的平台与 CANN 版本：Ascend950（算子包内容为 ascend950 kernel），CANN 9.1.0（共享只读安装）与 9.2.0（可写安装）。
- 适用粒度：platform_wide（目标目录判定是安装器脚本行为，与算子无关；机制为宿主工具链行为，跨 NPU 平台未验证，推断可迁移）。
- 路线优先级：recommended_fastpath（无 sudo 或不便改共享目录属主时的首选；另一条路线是改用自己可写的 CANN 安装）。
- 明确不适用 / 未验证场景：对 vendors 有写权限的自装 CANN 不存在此问题；`sudo` 改属主/权限也可解，但在共享环境会波及其他用户，本卡不推荐；`ASCEND_CUSTOM_OPP_PATH` 含 `:`（路径列表）时安装器直接报错，只接受单路径；`--install-for-all` 等参数未验证。

## 根因

`.run` 内嵌 `install.sh` 的目标目录（`targetdir`）优先级为：`--install-path` 参数 > 环境变量 `ASCEND_CUSTOM_OPP_PATH` > 默认 `$ASCEND_OPP_PATH`；共享安装下默认目标 `opp/vendors` 为 root 属主且模式 775，普通用户既不能在其下 `mkdir` 子目录、也不能对已有内容 `chmod`，安装器在 `apply_chmod`/`create_dir` 两步相继失败并使 `.run` 退出非零，`set -e` 的 build.sh 随之退出 1。

install.sh 目标判定关键逻辑（从 `.run --noexec` 解包后读出，verbatim 节选）：

```
351:if [ -n "${INSTALL_PATH}" ]; then
362:    targetdir=${INSTALL_PATH}
363:elif [ -n "${ASCEND_CUSTOM_OPP_PATH}" ]; then
364:    if [[ "${ASCEND_CUSTOM_OPP_PATH}" == *:* ]]; then
365:        log "[ERROR] environment variable ASCEND_CUSTOM_OPP_PATH=... is set and \
370:    if [ ! -d ${ASCEND_CUSTOM_OPP_PATH} ]; then
371:        create_dir ...   # 自动建目录分支
373:    targetdir=${ASCEND_CUSTOM_OPP_PATH}
```

`ASCEND_CUSTOM_OPP_PATH` 可以把安装目标重定向到可写目录，但本会话解包读到的 9.1.0/9.2.0 `.run` 包里，该自动建目录的 `create_dir` 调用传错了变量（形如 `create_dir "${INSASCEND_CUSTOM_OPP_PATHTALL_PATH}"`，未定义变量展开为空串），所以重定向目录必须**事先存在**，否则 `create_dir ""` 报 `dir path is empty`、随后目标目录检查报 `<path> no exist`。

## 解决配方

修复前（直接构建，安装目标为只读 opp/vendors）：

```bash
source <只读CANN>/set_env.sh   # 编译环境仍可用只读 CANN
cd operators && ./build.sh <Op>_package
# → 编译/打包成功，安装阶段 chmod ... Operation not permitted，exit=1
```

错误尝试（只设变量、不建目录）：

```bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/bh_opp
rm -rf /tmp/bh_opp
cd operators && ./build.sh <Op>_package
# → [ops_custom][ERROR] dir path is empty
#   [ops_custom][ERROR] /tmp/bh_opp no exist
#   build.sh exit=1
```

修复后（重定向 + 预创建目录；verbatim 为实际执行的命令形态）：

```bash
source <只读CANN>/set_env.sh   # 编译环境仍可用只读 CANN
export ASCEND_CUSTOM_OPP_PATH=/home/<user>/custom_opp_install   # 单个绝对路径，指向用户可写目录
mkdir -p "$ASCEND_CUSTOM_OPP_PATH"                              # 关键：install.sh 不会替你建这个目录
cd operators && ./build.sh <Op>_package
# → [ops_custom] ... create .../custom_opp_install/vendors/<Op>/{op_proto|op_impl|op_api}.
#   copy new ... files ... SUCCESS ... build.sh exit=0
```

要点：

- `ASCEND_CUSTOM_OPP_PATH` 必须是单个绝对路径（含 `:` 会被安装器拒绝），指向用户可写目录；编译阶段不受该变量影响，仍按 `set_env.sh` 的 CANN 头文件/工具链编译。
- 安装成功后安装器提示运行期需要把算子运行库加入库路径：`export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/vendors/<OpType>/op_api/lib/:${LD_LIBRARY_PATH}`——重定向安装的算子不在默认 `opp/vendors` 下，aclnn 调用方必须补这条，否则链接/加载不到 `libcust_opapi.so`（提示文案见验证信号，本轨迹未实测运行期加载）。

判别 / 定位方法：

1. 报错形态区分两种根因：逐文件 `chmod ... Operation not permitted` = 安装目标只读；`dir path is empty` + `<path> no exist` = `ASCEND_CUSTOM_OPP_PATH` 目录未预创建。
2. 确认安装目标：`echo $ASCEND_CUSTOM_OPP_PATH`（空 = 装到 `$ASCEND_OPP_PATH/opp/vendors`）。
3. `.run` 包能力自检：`bash build/custom_opp_ubuntu_x86_64.run --help` 可见 `--install-path`、`--install-for-all` 等选项；`--info` 看内嵌信息。`--install-path` 需绝对路径。
4. 替代路线：若环境里存在自己可写的 CANN 安装，直接 `source` 那份的 `set_env.sh` 再构建，安装进它的 opp/vendors 即可，无需重定向。

## 设计决策

- 选环境变量重定向而不是 `--install-path` 参数：build.sh 以无参方式调用 `"$RUNFILE"`，改脚本侵入大；`ASCEND_CUSTOM_OPP_PATH` 在 build.sh 之外 export 即生效，零侵入（`.run` 安装脚本原生支持该变量，`grep -rn "ASCEND_CUSTOM_OPP_PATH" $ASCEND_HOME_PATH/` 可见 `opp/vendors/*/bin/set_env.bash` 里就在用它做叠加）。
- 不用 sudo 改共享 vendors 属主：共享目录上提权会波及其他用户的安装（且安装器每次都会 `chmod -R` 收权 750），重定向到用户目录完全不触碰共享状态。
- 曾评估"vendors 改 1777 sticky"方案并放弃：安装器结尾 `apply_chmod vendors 750` 会把开放权限收回去，方案不稳定。
- 为什么必须 `mkdir -p`：实测只设变量不建目录必失败（`dir path is empty`/`no exist`）；install.sh 的 `create_dir` 路径参数是变量名 typo 的未定义变量，指不上。该 typo 只在 9.1.0/9.2.0 的 .run 包里观察到，其他版本未验证——但"预创建目录"在任何版本都无害。
- 放弃的候选：解包 `.run` 手工拷贝（绕过安装逻辑，不可复现 uninstall）。
- 当前方案路线档位：recommended_fastpath（只读 CANN 环境下的标准做法；本会话内被两个独立子会话先后走通并复现）。

## 系统信息

- 环境：共享服务器，只读 CANN 安装（属主他人）+ 可写 CANN 安装各一份；目标设备 Ascend950PR（kernel 产物 `op_impl/ai_core/tbe/kernel/ascend950/`）。
- 相关文件：`operators/build.sh`（入口，`make -j binary package` 后自动执行 `build/custom_opp_ubuntu_x86_64.run`）；`build/custom_opp_ubuntu_x86_64.run`（makeself 自解压安装包，内嵌 install.sh）；安装产物 `$ASCEND_CUSTOM_OPP_PATH/vendors/<OpName>/{op_api,op_impl,op_proto,scripts,bin,version.info}`。

## 预防

- 多用户共享 CANN 环境下，构建自定义算子前先 `ls -ld $ASCEND_OPP_PATH/vendors` 确认写权限；无写权限则在 source `set_env.sh` 之后、跑 build.sh 之前 `export ASCEND_CUSTOM_OPP_PATH=<自建可写目录>` 并 `mkdir -p`。
- 把 `export ASCEND_CUSTOM_OPP_PATH` 与配套 `LD_LIBRARY_PATH` 追加写进工程的环境脚本（如 env.sh），避免每次手工重设。

## 验证信号

问题与修复来源：共享只读 CANN 上的自定义算子构建会话，同一 build.sh 仅改变量前后对照；后续分支开发子会话独立复现一次。

- 只读目标安装失败：`chmod ... Operation not permitted` 数十条 + `mkdir: cannot create directory ... Permission denied`，build.sh exit=1；同目录他人算子（SoftplusV2Grad）同样无法 chmod，证实属环境性只读而非本工程问题（来源平台：950；证据等级：tool_observed）。
- 只设 `ASCEND_CUSTOM_OPP_PATH` 未建目录：`[ops_custom][ERROR] dir path is empty` + `[ops_custom][ERROR] <path> no exist`，exit=1（来源平台：950；证据等级：tool_observed）。
- `mkdir -p` 预创建后同一构建：安装 `SUCCESS`、`build.sh exit=0`，输出 `create .../custom_opp_install/vendors/<Op>/...` 与 `copy new ... files`（来源平台：950；证据等级：tool_observed）。
- 安装器输出运行期要求：`execute the command [ export LD_LIBRARY_PATH=.../custom_opp_install/vendors/<Op>/op_api/lib/:${LD_LIBRARY_PATH} ] to set the environment path`（来源平台：950；证据等级：tool_observed）。
- 预创建后 UT 33/33 通过、无 tiling 注册告警；独立复现：后续分支开发会话先同样 chmod 失败，`export ASCEND_CUSTOM_OPP_PATH=...; mkdir -p ...` 后 `EXIT_CODE=0`（来源平台：950；证据等级：tool_observed）。
- `.run` 为 makeself 自解包归档（`--info` 显示 Target directory，内含 `install.sh/uninstall.sh/upgrade.sh/packages`），install.sh 目标判定逻辑见根因段（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `ASCEND_CUSTOM_OPP_PATH` 机制属宿主侧安装器行为，与 NPU 型号无关，但本卡实测仅在 CANN 9.1.0/9.2.0 的 custom_opp 安装器上完成（其他平台未验证，推断可迁移）。
- 重定向后算子不在默认 `opp/vendors`，后续 aclnn 直连测试、TTK 黑盒执行是否需要在运行环境额外 `export ASCEND_CUSTOM_OPP_PATH` 或仅 `LD_LIBRARY_PATH` 即可——本轨迹只拿到安装器提示，运行期加载路径未实测。
- 若 build.sh 由其他脚本封装且不透传环境变量，本方案失效，需改用 `--install-path` 参数或在封装层注入。
- install.sh 变量名 typo 属 9.1.0/9.2.0 观察到的事实；未来版本修掉 typo 后"必须预创建"可能不再是必要条件（预创建仍无害）。
- 若 `ASCEND_CUSTOM_OPP_PATH` 与 `$ASCEND_OPP_PATH` 指向同一棵树，重定向不改变只读问题。

**未决项**：

- 重定向安装后的运行期加载链路（`LD_LIBRARY_PATH` 一条是否充分、`aclnn` 头文件查找路径是否也要指向新目录）未在本轨迹实测，接续验证点：在 `ASCEND_CUSTOM_OPP_PATH` 安装下跑一次 aclnn 直连冒烟。
- install.sh typo 在 CANN 官方版本间的存在范围未追（只查了本机 9.1.0/9.2.0 的 .run 包）。
- `--install-path` 与 `ASCEND_CUSTOM_OPP_PATH` 并存时的优先级未实测。
## 坏实践

- 在共享 `opp/vendors` 上 `sudo chmod -R 777` 或改属主：安装器随后仍会 `chmod 750` 收权，且破坏其他用户安装的算子权限。
- 看到 `chmod: Operation not permitted` 就误判为"自己的算子包打包坏了"而反复重编——先看属主与模式，编译产物与安装失败是两层问题。
- 只 `export ASCEND_CUSTOM_OPP_PATH` 不建目录就重试：必然得到 `dir path is empty`/`no exist`，与权限问题混淆，浪费排障轮次（本会话实际多花了一轮）。

## 误诊

- 曾把安装失败与编译失败混为一谈准备回退代码：build 日志 grep 证明无任何编译错误、`.run` 已生成，失败仅在 install 步——分清"编译产物完好"与"部署被权限挡住"后再选修复层。
- 初见 `dir path is empty` 容易以为是环境变量拼写或 set_env.sh 没生效；实际是 install.sh 内部 create_dir 的参数变量为空——变量本身设置是对的（`targetdir=${ASCEND_CUSTOM_OPP_PATH}` 正常），只是建目录分支坏了（此为对 .run 解包 install.sh 的人工判读，两条 ERROR 输出为实测）。
