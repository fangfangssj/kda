---
type: "Runbook"
title: "TTK Opc init failed：PYTHONPATH 被覆盖"
description: "harness 用 env PYTHONPATH=... 完全覆盖（而非追加）set_env.sh 导出的 tbe impl 路径，TTK aclnn 启动即报 Opc init failed（find_spec(\"impl\") 为 None）；改为前置追加保留原值后内置算子上板 PASS。"
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# TTK Opc init failed：PYTHONPATH 被覆盖

## 现象

在为 euclidean_norm 自定义 aclnn 算子准备 TTK（ops-test-kit）黑盒评测环境时，环境预检脚本 `check_env.sh` 用 CANN 包内置 `aclnnAdd` 用例上设备验证整条工具链。脚本内部以下面方式拉起 ttk：

```bash
( cd "$TTK" && env PYTHONPATH="$TTK:$ASCEND_TOOLKIT_HOME/python/site-packages" \
    "$PY" -m ttk aclnn -i "$HERE/references/aclnn_add.csv" --pc=1 -o "$OUT" ) >/dev/null 2>&1
```

结果 result.csv 未产出，预检直接 FAIL。屏蔽重定向手动复现（同样只给 `$TTK` 和 `python/site-packages` 两个路径）得到崩溃：

```
[CRITICAL] [tbe_interface.py:244]: Opc init failed: 'NoneType' object has no attribute 'submodule_search_locations'
Traceback (most recent call last):
  ...
  File "<TTK 检出目录>/ttk/core_modules/operator/tbe_interface.py", line 240, in __init__
    self.impl_path = list(importlib_util.find_spec("impl").submodule_search_locations)[0]
AttributeError: 'NoneType' object has no attribute 'submodule_search_locations'
```

同一开发会话后续跑黑盒 smoke 时再次复现：命令里 `env PYTHONPATH="$TTK:$(dirname "$PLUGIN"):$ASCEND_TOOLKIT_HOME/python/site-packages"` 以同样方式覆盖了已 source 的环境，报同一个 `submodule_search_locations` 崩溃。报错文案完全不提 PYTHONPATH 或 impl，从表面极难定位。

## 适用场景

- 工程形态：TTK（ops-test-kit）aclnn 黑盒评测 / 环境预检 / 任何用 `env PYTHONPATH=...` 或 `export PYTHONPATH=...` 包装 `python -m ttk` 的 harness 脚本。
- API 路径：`python -m ttk aclnn`；崩溃点 `ttk/core_modules/operator/tbe_interface.py` 的 `Opc.__init__`（`find_spec("impl").submodule_search_locations`）。
- 验证过的平台与 CANN 版本：Ascend950PR（soc=Ascend950PR_9597）+ CANN 9.1.0（set_env.sh 导出环境）。
- 适用粒度：platform_wide（CANN 工具链环境行为，与具体算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：未 source CANN `set_env.sh` 的干净 shell（无原 PYTHONPATH 可保留，需先 source）；非 ttk 的 Python 工具不受本卡约束。

## 根因

harness 用自定义 PYTHONPATH **完全覆盖**（而非追加）`set_env.sh` 已导出的 PYTHONPATH，丢掉了 `opp/built-in/op_impl/ai_core/tbe` 路径，导致 `importlib.find_spec("impl")` 返回 None。

机制链：

1. CANN `set_env.sh` 导出的 PYTHONPATH 含两段：`$ASCEND_TOOLKIT_HOME/python/site-packages` **和** `$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe`；
2. `impl` 包位于 tbe 路径下的 `impl/`（tbe 动态算子实现包）；
3. TTK 的 `Opc.__init__`（tbe_interface.py:240）执行 `find_spec("impl").submodule_search_locations`，不含 None 保护；
4. 包装脚本写死 `PYTHONPATH="$TTK:$ASCEND_TOOLKIT_HOME/python/site-packages"` 时 tbe 段被丢弃 → `find_spec("impl")` → None → `'NoneType' object has no attribute 'submodule_search_locations'`。

## 解决配方

before（`check_env.sh` 原句——覆盖式 PYTHONPATH）：

```bash
( cd "$TTK" && env PYTHONPATH="$TTK:$ASCEND_TOOLKIT_HOME/python/site-packages" \
    "$PY" -m ttk aclnn -i "$HERE/references/aclnn_add.csv" --pc=1 -o "$OUT" ) >/dev/null 2>&1
```

after（保留已导出 PYTHONPATH，仅前置 `$TTK`；调用方未 source set_env.sh 时回退默认值）：

```bash
# 保留 set_env.sh 已导出的 PYTHONPATH（含 tbe 路径，`import impl` 必需），仅前置 $TTK；
# 若调用方未 source set_env.sh，退回 $ASCEND_TOOLKIT_HOME/python/site-packages。
( cd "$TTK" && env PYTHONPATH="$TTK:${PYTHONPATH:-$ASCEND_TOOLKIT_HOME/python/site-packages}" \
    "$PY" -m ttk aclnn -i "$HERE/references/aclnn_add.csv" --pc=1 -o "$OUT" ) >/dev/null 2>&1
```

黑盒 smoke 命令同样改为前置追加（第二处同根因实例）：

```bash
# FIX: prepend to existing $PYTHONPATH (preserve env.sh's tbe impl path), not replace
export PYTHONPATH="$TTK:$(dirname "$PLUGIN"):$PYTHONPATH"
env PYTHONPATH="$TTK:$(dirname "$PLUGIN"):$PYTHONPATH" \
    "$PY" -m ttk aclnn -i "$MERGED" --ti=1 --pc=1 --device-whitelist 0 --plugin "$PLUGIN" -o "$OUTDIR/result_smoke.csv"
```

判别方法（不靠报错文案猜，两条探针命令即可定位）：

```bash
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
echo "$PYTHONPATH" | tr ':' '\n'   # 必须含 .../opp/built-in/op_impl/ai_core/tbe 段
python3 -c "import importlib.util as u; print(u.find_spec('impl'))"   # None 即缺 tbe 路径
```

## 设计决策

- 修脚本而非每次调用手动 export：环境预检脚本会被多个后续任务反复调用，修复必须落在脚本内一次性生效。
- 回退分支 `${PYTHONPATH:-...}`：允许未 source set_env.sh 的干净 shell 仍能跑（但此时只有 site-packages，`import impl` 仍需 source，已在注释中写明前置条件）。
- 判别探针独立于 ttk：`find_spec('impl')` 不依赖 ttk 代码路径，可在任何环境快速自检。

## 系统信息

- CANN：cann-9.1.0 安装目录（set_env.sh 导出 ASCEND_HOME_PATH / ASCEND_OPP_PATH / PYTHONPATH）。
- TTK checkout：仓检出版本（`python -m ttk aclnn` 方式运行，需 `$TTK` 在 PYTHONPATH）。
- 设备：Ascend950PR_9597。
- 相关文件：`check_env.sh`（环境预检脚本，修复落点）；黑盒 smoke 命令（同模式第二处修复）；工程 `env.sh`（预检产物，source 后取得完整环境）。

## 验证信号

本卡信号来自 euclidean_norm 算子黑盒评测的环境预检与 smoke 两处实测：

- 修复前：手动 `env PYTHONPATH="$TTK:$ASCEND_TOOLKIT_HOME/python/site-packages" python3 -m ttk aclnn ...` → `Opc init failed: 'NoneType' object has no attribute 'submodule_search_locations'`，无 result.csv 产出（来源平台：950；证据等级：tool_observed）。
- 修复后：`PY=python3 TTK=... ENV_FILE=... bash check_env.sh` → `[env-check] PASS: 工具链就绪（内置 aclnnAdd precision PASS）`，退出码 0，env.sh 生成且可 source（来源平台：950；证据等级：tool_observed）。
- 第二处实例：黑盒 smoke 命令改为前置追加后不再出现 impl 崩溃，错误推进到下一层（op_api_info 解析层），证明本根因在该实例同样闭合（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖"覆盖式 PYTHONPATH 导致 impl 不可见"这一根因；TTK 后续的头文件发现、符号加载、用例属性绑定是独立根因，另有卡片。
- `${PYTHONPATH:-...}` 回退分支在未 source set_env.sh 时仍缺 tbe 路径（回退值不含 tbe 段），此时必须先 source set_env.sh。
- 其他 CANN 版本 set_env.sh 的 PYTHONPATH 组成未验证，机制同源推断可迁移（impl 包随 CANN opp 路径分发，未验证，推断可迁移）。

**未决项**：

- 无（根因与修复均已机制级闭环并有两处实例验证）。
## 坏实践

- 在包装脚本里写死 `PYTHONPATH="$A:$B"` 覆盖调用方环境——对任何依赖 site-packages 之外路径的工具链都是隐患。
- 看到 `'NoneType' object has no attribute ...` 就怀疑工具自身 bug 去改 TTK 源码；本例是纯环境问题。

## 误诊

- 首次崩溃时先怀疑 ttk 安装不完整/需要重装——被探针推翻：`echo $PYTHONPATH` 对比 set_env.sh 前后差异即可确认是路径丢失，重装无意义（该假设为会话内推断，被 tool_observed 探针修正）。
