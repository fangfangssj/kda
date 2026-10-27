---
type: "Runbook"
title: .run 重装只读 scripts 目录 rm 权限拒绝
description: CANN 自定义算子 .run 安装包会把 vendors/<VENDOR>/scripts（含 uninstall.sh）建成无属主写位的只读目录；重装脚本 rm -rf 旧 install 目录时报 Permission denied，须先 chmod -R u+w 再删除重装。
tags: [task.implementation, topic.implementation.operator_registration, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T12:12:55Z'
updated_at: '2026-09-14T12:12:55Z'
platforms: ["950"]
---

# .run 重装只读 scripts 目录 rm 权限拒绝

## 现象

在 Ascend950PR（CANN 9.2.0）上迭代开发自定义算子的 Torch 测试路由：run.sh 每轮都重新构建 custom opp `.run` 包（CPack 产出 `custom_opp_ubuntu_x86_64.run`），先 `rm -rf install/` 清掉旧安装再执行 `bash <pkg>.run --quiet --nox11 --install-path=install/` 重装。首轮安装一切正常；第二轮起在安装步骤报错（`.run` 包本身构建成功）：

```text
rm: cannot remove '.../tests/torch/install/vendors/BenchHarnessScatter/scripts/uninstall.sh': Permission denied
```

用 `ls -la` 检查 install 目录，发现 `.run` 安装器（ops_custom）在上一轮安装时创建的文件/目录带只读 mode 位：

```text
dr-xr-x---. 2 user user   34 Sep  4 20:44 .
-r-xr-x---. 1 user user 2492 Sep  4 20:44 uninstall.sh
```

`vendors/<VENDOR>/scripts/` 目录是 `dr-xr-x---`（属主无写位），`uninstall.sh` 是 `-r-xr-x---`。当前用户就是属主，但 `rm` 仍无法删除。

## 适用场景

- 工程形态：CANN 自定义算子（custom opp）`.run` 包的重复安装/卸载脚本——任何"rm -rf 旧安装目录再重装 .run"的迭代开发流程（算子仓开发、Torch/TF/GEIR 测试路由的 run.sh 均是此形态）。
- API 路径：CPack 产出的 `custom_opp_ubuntu_x86_64.run` 自解压安装器（ops_custom）；安装布局 `<install>/vendors/<VENDOR>/{scripts,op_api,op_impl,op_proto,bin}`。
- 验证过的平台与 CANN 版本：Ascend950PR 宿主机（x86_64 Ubuntu），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该安装器行为与算子类型无关，凡重装即触发。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：install 目录属主非当前用户（chmod -R u+w 无效，需 sudo）；`--install-path` 指向 CANN 系统安装目录（通常 root 属主）的场景未验证。

## 根因

CANN custom opp `.run` 安装器安装时把 `vendors/<VENDOR>/scripts/`（含 `uninstall.sh`）以去除属主写位的 mode（`r-xr-x---`）创建。POSIX 语义下删除文件需要**父目录的写权限**：scripts 目录本身无属主写位，所以即使 `uninstall.sh` 属主是当前用户，`rm -rf install/` 也无法删除该目录下的条目，递归删除在 scripts 目录处中断，重装流程失败。

## 解决配方

修复前（run.sh 安装段）：

```bash
# ---- 2. Install it under this script's directory -----------------------
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
bash "${RUN_PACKAGE}" --quiet --nox11 --install-path="${INSTALL_DIR}"
```

修复后（rm 前先恢复属主写位）：

```bash
# ---- 2. Install it under this script's directory -----------------------
# The .run installer marks some files read-only; make them writable before rm.
chmod -R u+w "${INSTALL_DIR}" 2>/dev/null || true
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
bash "${RUN_PACKAGE}" --quiet --nox11 --install-path="${INSTALL_DIR}"
```

`chmod -R u+w` 只加属主写位不动其他位；`2>/dev/null || true` 容忍首次安装（目录尚不存在）时的报错，保持 `set -euo pipefail` 下脚本可用。

## 设计决策

- 选"chmod 后 rm"而不是复用安装器的 uninstall.sh：uninstall.sh 在被删除的目录里且本身只读，先 chmod 再调用它反而路径更绕；直接恢复写位后整目录删除最短路径。
- `|| true` 而不是先 `test -d`：一行覆盖"目录不存在/部分文件已可写/其他用户文件"三种情况，幂等。
- 当前方案路线档位：recommended_fastpath（一行前置命令，零副作用）。

## 系统信息

- 硬件/软件：x86_64 Ubuntu 宿主机 + Ascend950PR 环境，CANN 9.2.0；安装器输出带 `[ops_custom]` 前缀。
- 相关文件：`<op>_package/tests/torch/run.sh`（安装段，本修复位置）；`install/vendors/<VENDOR>/scripts/uninstall.sh`（只读位来源）；`build/custom_opp_ubuntu_x86_64.run`（CPack 产物）。

## 预防

- 任何含 `rm -rf <install_dir>` + `.run` 重装的脚本，把 `chmod -R u+w "${INSTALL_DIR}" 2>/dev/null || true` 作为固定前置行。
- 看到属主是自己的文件却 `rm: Permission denied`，先 `ls -ld` 查父目录写位，不要怀疑权限系统或用户身份。
- 相邻场景区分：清理 OPP 全局 `vendors/<OpType>` 旧安装是另一类重装坑（旧产物按合并语义残留、污染 `.o` 计数等核验），见 [重装算子包前清理 OPP vendors 旧安装残留](cleanup_vendors_before_reinstall.md)——彼卡清理 `$ASCEND_OPP_PATH` 下全局安装，本卡处理脚本本地 `install/` 目录的只读位。

## 验证信号

以下信号来自该算子 Torch 路由搭建会话：chmod 前后两轮 run.sh 的实测输出。

- 修复前：第二轮 run.sh 在 `rm -rf install` 处失败，报 `rm: cannot remove '.../install/vendors/BenchHarnessScatter/scripts/uninstall.sh': Permission denied`（来源平台：950；证据等级：tool_observed）。
- 观测：`ls -la install/vendors/<VENDOR>/scripts/` 显示目录 `dr-xr-x---`、`uninstall.sh -r-xr-x---`，均由安装器创建（来源平台：950；证据等级：tool_observed）。
- 修复后：重跑 run.sh 安装段输出 `[ops_custom] ... upgrade uninstall ... copy new uninstall.sh files ... SUCCESS`，随后 adapter 编译、直调 `direct_call_output: [10.0, 2.0, 30.0, 4.0]` 全链路通过（来源平台：950；证据等级：tool_observed）。
- 修复后：同一脚本在其后两轮重建（adapter 迭代修复期间）中未再出现权限错误，重装稳定（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只读位由当前版本 ops_custom 安装器设置；若未来安装器不再去写位，chmod 前置行幂等无害。
- `chmod -R u+w` 仅恢复属主写位：对 root/他人属主的 install 目录无效（本场景 install 目录由当前用户创建，属主一致）。
- 机制是 POSIX 目录权限语义层面的，推断可迁移至任何使用该安装器的 Linux 环境（未验证，推断可迁移）。

**未决项**：

- 安装器为何特意去除 scripts 目录写位（上游设计意图，推测为防误改卸载脚本）未核实；不影响本修复的适用性。
## 坏实践

- 装完 `.run` 包后直接 `rm -rf` 重装——第二轮必挂。
- 用 `sudo rm -rf` 绕过——掩盖安装器只读位事实，且把属主关系搞乱后更难清理。

## 误诊

- 无（Permission denied + `ls -la` 的 mode 位一眼定位；错误发生在 rm 而非安装器，方向明确）。
