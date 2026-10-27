---
type: "Runbook"
title: .run 安装 chmod 噪音与建目录权限失败判别
description: CANN 自定义算子包 build.sh 末段大量 chmod/sed "Operation not permitted" 多为 .run 安装器对其他 vendor 目录的非致命噪音；编译成败以分离的 cmake/make binary package 退出码与本包目录 create dir 失败为准。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# .run 安装 chmod 噪音与建目录权限失败判别

## 现象

在 x86_64 构建服务器上编译自定义算子包 BenchHarnessSoftmax（目标 ascend950，CANN 自定义算子工程），执行一键脚本：

```bash
cd operators && ./build.sh BenchHarnessSoftmax_package
```

脚本流程是 `rm -rf build && mkdir build && cd build && cmake .. && make -j binary package`，然后执行生成的 `custom_opp_ubuntu_x86_64.run` 自安装包。首次运行时 build.sh **非零退出**，末尾输出铺满了几十行同类错误（原文摘录）：

```
chmod: changing permissions of '/home/<user>/Ascend/cann-9.1.0/opp/vendors/custom_nn/op_impl/ai_core/tbe/kernel/ascend950/softshrink/...': Operation not permitted
chmod: changing permissions of '/home/<user>/Ascend/cann-9.1.0/opp/vendors/custom_nn/op_api/lib/libcust_opapi.so': Operation not permitted
[ops_custom] [2026-07-25 18:50:55] [ERROR] chmod /home/<user>/Ascend/cann-9.1.0/opp/vendors 750 failed.
[ops_custom] [2026-07-25 18:50:55] [INFO] create /home/<user>/Ascend/cann-9.1.0/opp/vendors/BenchHarnessSoftmax.
mkdir: cannot create directory '/home/<user>/Ascend/cann-9.1.0/opp/vendors/BenchHarnessSoftmax': Permission denied
[ops_custom] [2026-07-25 18:50:55] [ERROR] create dir /home/<user>/Ascend/cann-9.1.0/opp/vendors/BenchHarnessSoftmax failed.
```

当时代码状态：算子包源码（op_host/op_kernel/tests）本身没有改动过，怀疑"编译失败"。实际上 cmake 与 make 阶段已经全部成功，失败只发生在最后的 `.run` 安装步骤——`opp/vendors/` 目录属主为 `root:root`（`drwxrwxr-x root:root`），当前普通用户（非 root 组）对 other 位只有 `r-x`，无法写入。

## 适用场景

- 工程形态：CANN 自定义算子包（Ascend C vendor package）构建与 `.run` 安装；`build.sh` + `make binary package` + ops_custom 安装器链路。
- API 路径：`$ASCEND_HOME_PATH/tools/op_project_templates/ascendc/customize/cmake/func.cmake` 生成的 `binary`/`package` 目标；ops_custom 自安装 `.run`。
- 验证过的平台与 CANN 版本：x86_64 构建环境，CANN cann-9.1.0（set_env.sh 环境）；算子包目标平台 ascend950（arch35）。NPU 上板不涉及本卡结论。
- 适用粒度：platform_wide（同一台共享构建服务器上的所有自定义算子包构建）。
- 路线优先级：recommended_fastpath（判断编译成败时先分离退出码，不要直接追 chmod 噪音）。
- 明确不适用 / 未验证场景：`cmake`/`make` 阶段本身的编译错误（那才是真编译失败）；`pip`/驱动等其他安装器的权限错误；其他用户属主布局未验证。

## 判别方法

关键区分**致命错误**与**非致命噪音**：

1. **先分离编译与安装两个阶段**，单独跑编译并打印退出码（本例二者均 0，证明编译成功）：

```bash
cd <pkg>/build 2>/dev/null || { mkdir -p <pkg>/build && cd <pkg>/build; }
cmake .. > /tmp/cmake.log 2>&1; echo "cmake exit: $?"
make binary package > /tmp/make.log 2>&1; echo "make exit: $?"
```

`make` 末尾出现 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created.` 即 `.run` 打包成功。

2. **在 build.sh 全量输出里按错误行分类**：

- **致命（本包目录创建失败）**：`mkdir: cannot create directory '.../opp/vendors/<YourOp>': Permission denied` + `[ops_custom] [ERROR] create dir ... failed.` → 你的包没有装进去，build.sh 非零退出。
- **非致命噪音**：对**其他 vendor 目录**（custom_nn 等）每个文件的 `chmod: ... Operation not permitted`、`chmod .../opp/vendors/config.ini 750 failed.`、`sed: couldn't open temporary file .../opp/vendors/sedXXXX: Permission denied` → `.run` 安装器扫描整个 vendors 目录时的附带动作失败，不影响本包安装；安装器随后仍打印 `SUCCESS` 且 build.sh 退出码为 0。

3. **本包目录属主修复**（有免密 sudo 的环境）：

```bash
sudo -n mkdir -p /home/<user>/Ascend/cann-9.1.0/opp/vendors/BenchHarnessSoftmax
sudo -n chown <user>:<user> /home/<user>/Ascend/cann-9.1.0/opp/vendors/BenchHarnessSoftmax
```

修复后重跑 build.sh，即使 chmod 噪音仍存在，最终 `SUCCESS` + 退出码 0。

## 根因

`opp/vendors/` 由 root 安装 CANN 时创建（root:root，other 无写位），普通用户既无法给其他厂商目录改权限（chmod 噪音来源），也无法创建自己包的目录（致命失败来源）；而 build.sh 的 `set -e` 只在 `.run` 安装器以非零退出时才中断——chmod 失败不改变安装器退出码，只有本包 `create dir failed` 会传导为失败。

## 验证信号

来源：BenchHarnessSoftmax 算子包首次接入 gtest TilingUT 时的构建会话（build.sh / cmake / make 命令输出与修复前后对照）。

- 分离运行 `cmake ..` 退出码 0、`make binary package` 退出码 0，且 `custom_opp_ubuntu_x86_64.run successfully created`（来源平台：950 工具链构建环境（x86_64, CANN 9.1.0）；证据等级：tool_observed）。
- 修复前 build.sh 因 `create dir .../opp/vendors/BenchHarnessSoftmax failed.` 非零退出；`sudo mkdir + chown` 后 build.sh 退出码 0（来源平台：950 工具链构建环境（x86_64, CANN 9.1.0）；证据等级：tool_observed）。
- 修复后同一输出里仍出现 `sed: couldn't open temporary file .../opp/vendors/sedI8SZay: Permission denied` 与 `chmod .../opp/vendors/config.ini ... Operation not permitted`，但 `.run` 最终打印 `SUCCESS`、build.sh 退出码 0——证明这些行是噪音（来源平台：950 工具链构建环境（x86_64, CANN 9.1.0）；证据等级：tool_observed）。
- 后续每次全量 build.sh 门禁（编译 + UT 编入 + 安装）均退出码 0，chmod 噪音持续存在（来源平台：950 工具链构建环境（x86_64）；证据等级：tool_observed）。

## 失效边界

- 本卡只判定"安装阶段权限噪音 vs 编译失败"；`cmake`/`make` 阶段的真实编译错误（模板实例化、头文件缺失等）不在判别范围内。
- `sudo -n` 修复要求当前用户在 sudo 组且免密；无 sudo 权限的共享环境只能请管理员预建 vendor 目录。
- 观察于 CANN 9.1.0/9.2.0 的 ops_custom 安装器行为；其他 CANN 版本安装器对 chmod 失败的容忍策略未验证（未验证，推断行为一致）。

**未决项**：

- ops_custom 安装器为何对其他 vendor 目录逐文件 chmod、以及 config.ini 修改失败是否影响后续多 vendor 场景（如 vendor 加载顺序），未追根因到安装器实现层。
## 坏实践

- 看到 build 输出末尾几十行 `chmod: Operation not permitted` 就判定"编译失败"并去改源码或重装 CANN——本例中编译与打包全部成功。
- 把 chmod 噪音当作阻塞问题去逐条修复其他厂商目录权限——它们与本包安装无关。

## 误诊

- 会话初期曾把整个 build.sh 非零退出归因为"编译失败"，准备检查编译配置；分离 cmake/make 退出码后确认编译成功，真正的失败点是 `.run` 安装步骤的本包目录创建（对其他 vendor 目录的 chmod 噪音从头到尾都存在且无害）。

## 预防

- 判定构建成败时固定使用"分离退出码"探针：`cmake ..; echo $?` / `make binary package; echo $?`，再对全量日志只 grep 本包目录的 `create dir ... failed`。
- 共享构建服务器上首次部署某算子包前，先确认 `$ASCEND_HOME_PATH/opp/vendors/<YourOp>` 目录存在且当前用户可写。
