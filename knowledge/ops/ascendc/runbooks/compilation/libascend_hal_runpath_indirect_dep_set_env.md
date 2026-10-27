---
type: "Runbook"
title: UT 缺 libascend_hal.so：source set_env.sh
description: op_api UT 二进制报 libascend_hal.so 找不到：该库在 NPU 驱动目录（/usr/local/Ascend/driver/lib64/driver）且为间接依赖，RUNPATH 不传递；source CANN set_env.sh 即解。
tags: [task.testing, topic.architecture, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:21:08Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# UT 缺 libascend_hal.so：source set_env.sh

## 现象

自定义算子 BenchNoHarnessMaximum 扩展 UT（新增 6 dtype + 广播用例）后，op_host 的 32 个 UT 用例全部通过；接着运行 op_api UT 二进制（`./build/op_api/bench_no_harness_maximum_op_api_ut`，构建于 CANN 9.1.0 环境，实机 Ascend950PR），无论 `LD_LIBRARY_PATH` 指向 CANN lib64 与否都报同一个错：

```
./build/op_api/bench_no_harness_maximum_op_api_ut: error while loading shared libraries: libascend_hal.so: cannot open shared object file: No such file or directory
```

当时已确认：CANN 安装的 lib64 里没有这个库；`find` 只在 `x86_64-linux/devlib/device/` 和 `x86_64-linux/lib64/device/lib64/` 找到一份 `-r--r-----.` 权限的 `libascend_hal.so`。先后试过：把它拷到二进制旁（RUNPATH 含 `$ORIGIN`）、`chmod 755`、`LD_PRELOAD`（报 `ERROR: ld.so: object ... cannot be preloaded (cannot open shared object file): ignored`）——全部无效。

## 适用场景

- 工程形态：自定义算子 UT 开发（op_api 层 gtest 二进制，链接 libcust_opapi/libopapi/libnnopbase/libplatform 等）。
- API 路径：CANN 环境脚本 `$ASCEND_HOME_PATH/set_env.sh`；动态链接 RUNPATH/LD_LIBRARY_PATH 解析规则；NPU 驱动库目录 `/usr/local/Ascend/driver/lib64`。
- 验证过的平台与 CANN 版本：Ascend950PR 宿主机 + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——凡在安装了 NPU 驱动的宿主机上直接运行链接 CANN 栈的二进制都适用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：容器内无驱动环境（库物理缺失，source 也解不了）；MSDebug/模拟器环境的库路径（本轨迹未涉及）。

## 判别方法

完整诊断链（本会话走通，可复用）：

1. **触发**：直接运行 UT 二进制报 `libascend_hal.so: cannot open shared object file`，且 `LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:...` 依旧。
2. **探针一（排除 CANN 目录）**：`find $ASCEND_HOME_PATH -name "libascend_hal.so"` → 只在 `x86_64-linux/devlib/device/`、`x86_64-linux/lib64/device/lib64/` 有，主 lib64 没有——说明它不是随 CANN lib64 发布的普通库。
3. **探针二（排除 RUNPATH 方案）**：`readelf -d <binary> | grep RUNPATH` → `[$ORIGIN:<本机 CANN 安装根>/lib64]`。把库拷到 `$ORIGIN` 并 `chmod 755` 后仍然找不到——因为该库是 libplatform 等库的**间接**依赖，而 RUNPATH 只对直接依赖生效、不传递（这是与 LD_LIBRARY_PATH 的关键差异）；`LD_PRELOAD` 该拷贝件也报 cannot be preloaded。
4. **探针三（找真正的库）**：`find /usr/local/Ascend/driver/lib64 -name "libascend_hal*"` → `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`——它属于 NPU 驱动层，不在 CANN 安装树。
5. **探针四（看官方环境脚本怎么配）**：`source $ASCEND_HOME_PATH/set_env.sh && echo $LD_LIBRARY_PATH` → 追加了 `/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver` 等驱动目录——官方脚本就是把驱动库目录拼进 LD_LIBRARY_PATH 的。
6. **解释/处置**：运行前 source set_env.sh（或把 source 写进 UT 的 run.sh），让 LD_LIBRARY_PATH 覆盖驱动目录。

## 根因

libascend_hal.so 是 NPU 驱动库，物理位置在 `/usr/local/Ascend/driver/lib64/driver/`，不在 CANN 的 lib64 中；它又是 CANN 运行库（libplatform 等）的间接依赖。二进制 RUNPATH 只解析直接依赖，无法传递给间接依赖；只有全局 LD_LIBRARY_PATH 能覆盖。CANN 官方 set_env.sh 正是通过向 LD_LIBRARY_PATH 追加驱动 lib64 目录解决这一布局。

## 解决配方

修复前（run.sh 只设置算子包与 CANN lib64 相关路径，直接执行二进制）：

```bash
if [ -z "$ASCEND_HOME_PATH" ]; then
    ...
fi
```

修复后（在 UT run.sh 中先 source CANN 环境脚本再执行，确保驱动库目录进入 LD_LIBRARY_PATH）：

```bash
if [ -z "$ASCEND_HOME_PATH" ]; then
    source ~/Ascend/cann-9.1.0/set_env.sh   # 按本机 CANN 安装根调整
fi
```

交互执行时的等价命令：

```bash
cd operators/<op>/tests/ut && source "$ASCEND_HOME_PATH/set_env.sh" && ./build/op_api/<op>_op_api_ut
```

## 预防

- 任何直接运行链接 CANN 栈的二进制（op_api UT、自建 aclnn repro 等）前，先 `source $ASCEND_HOME_PATH/set_env.sh`，不要手工拼 LD_LIBRARY_PATH（漏驱动目录）。
- 遇到 `libascend_hal.so`/`libdrvdav*` 等驱动层库缺失，直接查 `/usr/local/Ascend/driver/lib64`；别在 CANN 安装树里找。
- 记住 RUNPATH 不解析间接依赖：`$ORIGIN` 旁拷库对二阶依赖无效。

## 验证信号

以下信号来自 BenchNoHarnessMaximum 算子 UT 扩展会话：op_api UT 因库缺失无法启动，经诊断链定位后 source 环境脚本重跑验证。

- 触发：`./build/op_api/bench_no_harness_maximum_op_api_ut` 报 `error while loading shared libraries: libascend_hal.so: cannot open shared object file`，设置 `LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:...` 后依旧（来源平台：950；证据等级：tool_observed）。
- 探针：`readelf -d` 显示 RUNPATH 含 `$ORIGIN` 与 CANN 安装根 lib64；库实际位于 `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`；`LD_PRELOAD` 二进制旁拷贝件报 cannot be preloaded（来源平台：950；证据等级：tool_observed）。
- 处置后：source CANN set_env.sh 后运行 `./build/op_api/bench_no_harness_maximum_op_api_ut` → 23 个 op_api 用例全部 OK（参数校验 10 + 正常调用 13，含 6 dtype 与广播正常调用）（来源平台：950；证据等级：tool_observed）。
- 同会话随后将该 source 步骤并入 UT run.sh，整包 `bash run.sh` 干净重建运行 55/55（op_host 32 + op_api 23）通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 路径 `/usr/local/Ascend/driver/lib64[/driver]` 为本机标准驱动安装布局；定制安装路径的机器须以 set_env.sh 展开结果为准。
- set_env.sh 会同时改写多个环境变量（LD_LIBRARY_PATH/ASCEND_HOME_PATH/PATH 等）；只需最小环境时可用其展开的驱动目录手动拼接 LD_LIBRARY_PATH（本轨迹未单独验证最小集）。
- 本卡只解决"库存在但搜不到"；驱动未安装（目录整体缺失）属环境安装问题，另议。

**未决项**：

- set_env.sh 中驱动目录的完整最小集合（common/driver 子目录哪些必需）未逐项验证。
## 坏实践

- 把 devlib 下的 libascend_hal.so 拷到二进制旁并指望 `$ORIGIN` 生效——RUNPATH 不传递给间接依赖，实测无效。
- 对 `-r--r-----.` 权限的库拷贝件做 `chmod 755` 后继续走 RUNPATH 路线——权限不是根因。
- 只把 CANN lib64 加进 LD_LIBRARY_PATH 就以为环境配好了——驱动目录才是缺失项。

## 误诊

- 曾怀疑库文件权限（`-r--r-----`）导致加载失败：chmod 后依旧，排除——根因是搜索路径不含驱动目录。
- 曾尝试 LD_PRELOAD 拷贝件：ld.so 直接拒绝加载（cannot be preloaded），说明该拷贝路径本身不可用，也非权限单一问题。
