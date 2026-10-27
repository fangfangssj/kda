---
type: "Runbook"
title: "ttk aclnn warmup 依赖 tbe→scipy 双路径"
description: "ttk aclnn 模式启动会编译 warmup kernel，需 import CANN 的 tbe 框架；tbe 依赖 scipy 且 PYTHONPATH 须同时含 ttk 仓与 $ASCEND_TOOLKIT_HOME/python/site-packages。缺 scipy 时先报 Import tbe failed 再报 AttributeError '_tbe'，易被误读为 ttk 版本问题。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# ttk aclnn warmup 依赖 tbe→scipy 双路径

## 现象

为 Ascend950 上的自定义算子准备 TTK 黑盒精度测试环境：克隆 ops-test-kit 仓、在自备 conda/venv Python 环境（Python 3.13）补装 ttk 直接依赖（decorator/psutil/sympy/pandas/attrs/six）后，`import ttk, torch, numpy, decorator` 已全部通过。随后按 TTK 环境自检流程跑内置算子最小用例：

```
PYTHONPATH="<ttk_repo>:$ASCEND_TOOLKIT_HOME/python/site-packages" \
  <venv_python> -m ttk aclnn -i <...>/aclnn_add.csv -o "$OUT"
```

TTK 启动正常（`Mode: ASCEND_ONBOARD`、`Device Platform: Ascend950PR_9579`），读取用例后在编译 warmup kernel 阶段崩溃，报错原文：

```
[tbe_interface.py:115]: Import `tbe` failed. Maybe it has been removed: No module named 'scipy'
Traceback (most recent call last):
  File ".../ops-test-kit/ttk/cli/aclnn.py", line 25, in _handle_aclnn
  File ".../ops-test-kit/ttk/cli/bridge.py", line 216, in run_with_switches
  File ".../ops-test-kit/ttk/core_modules/infra/instance_base.py", line 115, in profile
    self.setup_profile_object()
  File ".../ops-test-kit/ttk/core_modules/npu/instance_refactor.py", line 161, in _compile_help_kernels
    Opc().compile_warmup_kernel()
  File ".../ops-test-kit/ttk/core_modules/operator/tbe_interface.py", line 333, in compile_warmup_kernel
    full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
  File ".../ops-test-kit/ttk/core_modules/operator/tbe_interface.py", line 126, in get_soc_spec
    return self._tbe.common.platform.platform_info.get_soc_spec(key)
AttributeError: 'TbeOpc' object has no attribute '_tbe'
```

首行日志 `Import tbe failed. Maybe it has been removed` 的措辞极易把人引向"ttk 与 CANN 版本不匹配/tbe 被移除"的方向；真实原因只是执行 Python 环境缺 `scipy`。

## 适用场景

- 工程形态：TTK（ops-test-kit）aclnn 黑盒精度/性能测试的执行环境搭建，尤其在自备 conda/venv Python 而非 CANN 自带 Python 的机器上。
- API 路径：`python -m ttk aclnn`；CANN `python/site-packages` 下的 `tbe` 框架（`tbe/tvm/relay/analysis/sparse_dense.py` → `import scipy.sparse`）；环境变量 `PYTHONPATH`（ttk 仓 + `$ASCEND_TOOLKIT_HOME/python/site-packages`）。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579，上板实测），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（tbe 导入链与平台无关，任何用 ttk aclnn 模式的环境同构）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：TTK 的非 aclnn 模式（如纯 kernel 直调模式）是否同样触发 warmup kernel 编译未验证；CANN 自带 Python 环境通常已含 scipy，不适用本卡。

## 根因

ttk 的 aclnn 模式在 profile 启动时会编译 warmup kernel（`instance_refactor.py::_compile_help_kernels` → `tbe_interface.py::compile_warmup_kernel`），需要 import CANN 自带的 `tbe` 框架（位于 `$ASCEND_TOOLKIT_HOME/python/site-packages`）；而 `tbe` 的模块链 `tbe/tvm/relay/analysis/sparse_dense.py` 顶层 `import scipy.sparse`——自备 Python 环境没装 scipy 时 tbe 导入失败，`TbeOpc.__init__` 里的 `self._tbe` 随之未赋值，后续 `get_soc_spec` 抛出的 `AttributeError: 'TbeOpc' object has no attribute '_tbe'` 只是导入失败的下游症状，不是 ttk/tbe 接口不兼容。

## 解决配方

两步修复（本轨迹实际执行，verbatim）：

1. 补装 scipy。pypi.org 直连可能超时（`ReadTimeoutError ... (read timeout=15)` / `No matching distribution found`），换国内镜像：

```
<venv_python> -m pip install --timeout 300 -i https://mirrors.aliyun.com/pypi/simple/ scipy
```

2. 保持 PYTHONPATH 双路径并在 ttk 仓目录下执行（warmup kernel 产物写入 CWD 的 `./kernel_meta`，须与 `$TTK/kernel_meta` 一致）：

```
cd <ttk_repo> && \
OUT="$(mktemp -d)/envcheck_result.csv" && \
env PYTHONPATH="<ttk_repo>:$ASCEND_TOOLKIT_HOME/python/site-packages" \
  <venv_python> -m ttk aclnn \
  -i <...>/ttk-env-check/references/aclnn_add.csv -o "$OUT"
```

依赖自检探针（修复前后各跑一次，定位缺哪个模块）：

```
PYTHONPATH="<ttk_repo>:$ASCEND_TOOLKIT_HOME/python/site-packages" \
  <venv_python> -c "import tbe; print('tbe OK')"
```

修复前该探针的输出直接给出缺失位置：

```
File "$ASCEND_TOOLKIT_HOME/python/site-packages/tbe/tvm/relay/analysis/sparse_dense.py", line 25, in <module>
    import scipy.sparse as sp
ModuleNotFoundError: No module named 'scipy'
```

## 设计决策

- 用 `import tbe` 单行探针而不是反复跑完整 ttk 用例：tbe 导入链独立于 ttk 主流程，探针秒级定位缺失模块，避免每次上板重跑。
- PYTHONPATH 必须同时含 ttk 仓与 `$ASCEND_TOOLKIT_HOME/python/site-packages` 两条：前者提供 `ttk` 包，后者提供 `tbe`——只含其一分别报 `No module named ttk` / `Import tbe failed`。
- 执行目录固定 `cd $TTK`：tbe 把 warmup kernel 写到 CWD 的 `./kernel_meta`，在别处执行会散落产物。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 新环境跑 ttk 前按清单自检：`import ttk, torch, numpy, decorator`（ttk 直接依赖）+ `import tbe`（CANN 框架 + scipy 隐式依赖）+ `PYTHONPATH` 双路径 + `cd $TTK`。
- 把上述环境（PY、TTK、PYTHONPATH）固化进工程 env.sh，一次 source 全量生效。

## 验证信号

以下信号来自该算子工程的 TTK 环境预检会话（缺 scipy 失败 → 镜像安装 + 双路径 PYTHONPATH 后重跑同一内置用例，Ascend950PR_9579 上板）：

- 修复前：`ttk aclnn` 报 `Import tbe failed. Maybe it has been removed: No module named 'scipy'`，随后 `AttributeError: 'TbeOpc' object has no attribute '_tbe'`；独立探针 `import tbe` 报 `tbe/tvm/relay/analysis/sparse_dense.py:25 import scipy.sparse → ModuleNotFoundError`；pip 走 pypi.org 超时（`Read timed out`，15s）且 `No matching distribution found`（来源平台：950；证据等级：tool_observed）。
- 修复后：`pip install --timeout 300 -i https://mirrors.aliyun.com/pypi/simple/ scipy` 安装成功；同一命令重跑，内置 aclnnAdd 上板执行输出 `GOLD: PASS`、`PRECISION_STATUS: PASS`，result.csv 落盘且 `mere: 0.0, mare: 0.0, pass: True`（来源平台：950；证据等级：tool_observed）。
- 环境自检脚本整链通过：`check_env.sh` 输出 `[env-check] PASS: 工具链就绪（内置 aclnnAdd precision PASS）`，退出码 0，环境写入工程 env.sh（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡实测 CANN 9.2.0 的 tbe 模块链；其他 CANN 版本 tbe 的 scipy 依赖深度可能不同（未验证，机制同构：探针 `import tbe` 会直接暴露缺失模块名）。
- pypi 镜像可用性随网络环境变化；aliyun 镜像在本轨迹可用，其他镜像（tsinghua 等）未测。
- 本卡不覆盖 ttk 其余隐式依赖（torch 版本匹配、CLANG 路径等），仅覆盖 tbe→scipy 这一条已实测链路。

**未决项**：

- 无（本闭环归因与修复均已实测验证；ttk 其他模式的依赖差异见失效边界）。
## 坏实践

- 见到 `Maybe it has been removed` 就去换 ttk 版本或换 CANN——先跑 `import tbe` 探针看真实缺失模块。
- 只把 ttk 仓放进 PYTHONPATH 而漏掉 `$ASCEND_TOOLKIT_HOME/python/site-packages`，或反之。
- 在任意 CWD 直接 `python -m ttk`，让 warmup 产物散落在当前目录的 kernel_meta。

## 误诊

- 首次环境自检失败时输出里混有 csv 相对路径的 FileNotFoundError，一度怀疑用例文件问题；改绝对路径复跑后仍稳定复现 tbe 导入失败，才把归因收敛到 scipy/PYTHONPATH——路径问题与依赖问题是两个独立噪声源，应分别排除。
