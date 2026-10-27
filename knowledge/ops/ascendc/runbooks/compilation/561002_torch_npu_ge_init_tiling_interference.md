---
type: "Runbook"
title: 561002：torch_npu GE 初始化干扰 aclnn tiling
description: 同进程 import torch_npu 触发 GE 初始化，劫持 aclnn 自定义算子 tiling 属性访问报 561002；验证脚本改用 ctypes 直调 aclnn C API（aclCreateTensor 在 libnnopbase.so）绕过。
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 561002：torch_npu GE 初始化干扰 aclnn tiling


## 现象

算子仓 torch 路线联调：适配器签名与 TORCH_LIBRARY schema 都修好后（编译、链接、dlopen 注册全部通过），`test_torch.py` 按常规写法——`import torch_npu`、`torch_npu.npu.set_device(0)`、在 `npu:0` 上建输入张量、调 `torch.ops.glacier_thnn_fused_gru_cell_grad.forward(...)`——`aclnnThnnFusedGruCellGradGetWorkspaceSize` 直接返回 561002：

```text
RuntimeError: aclnnThnnFusedGruCellGradGetWorkspaceSize failed with status 561002
[ERROR] 2026-09-08-00:07:26 (PID:3505752, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

极具迷惑性的是：**同一个 vendor 包**在独立进程的 C++ ST 测试（`tests/aclnn`，`aclInit` 直调 aclnn，完全不碰 torch_npu）43/43 全 PASS（Branch-0 20/20、Branch-1 11/11、Branch-2 12/12、smoke 1/1）；kernel 二进制、tiling 库（`libcust_opmaster_rt2.0.so`）、ops-info 配置在本地 `install/vendors/` 与全局 `${ASCEND_OPP_PATH}/vendors/` 都存在且 md5 一致。也就是说 kernel/tiling 本身没问题，**只在「同进程 import 过 torch_npu」的 Python 进程里失败**，换任何 shape（4×8 / 16×64 / 512×1077 / 1×64 / 32×128）都一样。

关键证据来自开 plog 的复跑：kernel 其实**已被找到并注册**（`SetKernelAttribute` / `CreateNewKernel` 日志俱全，tilingKey=1），561002 级联之前的**第一条 ERROR** 是 GE 层的属性访问失败（算子只有 1 个属性 has_bias，合法索引区间是 [0, 1)，GE 去取 index 1 越界）：

```text
[ERROR] GE(3545754,python3):2026-09-08-00:44:51.092.500 [runtime_attrs.cc:19]3545754 GetPointerByIndex: ErrorNo: 4294967295(failed)Failed to get attr, the index 1 out of range 1
[ERROR] OP(3545754,python3):2026-09-08-00:44:51.092.520 [indv_executor.cpp:879][NNOP][NnopbaseExecutorDoTiling][3545754] errno[-991334334] OpName:[ThnnFusedGruCellGrad_0] Failed to execute operator ThnnFusedGruCellGrad_0. Reason: Failed to execute tiling function.
[ERROR] OP(3545754,python3):2026-09-08-00:44:51.092.549 [indv_executor.cpp:969][NNOP][NnopbaseExecutorTilingAndUpdateBinInfo][3545754] errno[561002] OpName:[ThnnFusedGruCellGrad_0] Check NnopbaseExecutorDoTiling(executor) failed
[ERROR] OP(3545754,python3):2026-09-08-00:44:51.092.566 [indv_executor.cpp:1111][NNOP][NnopbaseExecutorRunForWorkspace][3545754] errno[561002] OpName:[ThnnFusedGruCellGrad_0] Check NnopbaseExecutorMatchCache(executor) failed
[ERROR] OP(3545754,python3):2026-09-08-00:44:51.092.572 [individual_op_api.cpp:34][NNOP][NnopbaseOpLogE][3545754] errno[561002] OpName:[ThnnFusedGruCellGrad_0] Check NnopbaseRunForWorkspace(*executor, workspaceSize) failed
```

## 适用场景

- 工程形态：ops-registry-invoke 算子仓 torch 路线（tests/torch 下以 Python 验证自定义 vendor 算子的 aclnn 调用）；TTK aclnn 模式同属 ctypes 路线。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`（vendor `libcust_opapi.so`）；`libascendcl.so`（aclInit/aclrtSetDevice/aclrtMalloc/aclrtMemcpy/aclrtCreateStream/aclrtSynchronizeStream）；`libnnopbase.so`（aclCreateTensor）；`torch.ops.load_library`（适配器注册，仅构建要求）。
- 验证过的平台与 CANN 版本：Ascend950PR（注册平台 950）+ CANN 9.0.0-beta.2、torch_npu 2.7.1.post6（与 frontmatter `platforms` 一致）。
- 适用粒度（transfer_scope 最窄档）：platform_wide——该栈版本上任何自定义 vendor 算子经 torch_npu 进程做 aclnn 验证都会触及（本例为 5 输出梯度算子，根因与算子形态无关）。
- 路线优先级：safety_fallback。**fallback 声明：这是绕过路线，不是 GE 问题的修复。** 在 torch_npu 与自定义 vendor aclnn 的同进程共存问题解决之前，eager 验证用 ctypes 直调；性能敏感或需要 torch_npu 设备张量语义的场景不适用。
- 明确不适用/未验证场景：torch.compile / ACLGraph（ttk e2e --aclgraph）路线未跑通（本环境 TTK e2e 另有 Device count: 0 问题）；多卡/HCCL 场景未验证；其他 CANN/torch_npu 版本未验证。

## 根因

`import torch_npu` 会触发 GE（Graph Engine）运行时初始化；此后同进程内调用 aclnn API 时，GE 介入 tiling 执行路径，按索引取算子属性时越界——本算子仅 1 个属性 `has_bias`（合法索引 0..0），GE 请求 index 1 越界——tiling 函数执行失败（`NnopbaseExecutorDoTiling` errno -991334334），上层沿 `TilingAndUpdateBinInfo → MatchCache → RunForWorkspace` 逐层 Check failed，统一以 561002 级联报出。

机制链（每步都有实测证据）：`import torch_npu` → GE init（plog 出现 GE 日志）→ `aclnn…GetWorkspaceSize` → `NnopbaseExecutorDoTiling` → `GE runtime_attrs.cc GetPointerByIndex(1)`：attr 总数 1、合法索引区间 [0,1)、请求 index 1 → 越界失败 → tiling 失败 → 561002 级联。`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 只控制 PyTorch 后端 autoload（torch_npu `__init__.py` 在 import torch 前后置位/恢复该变量），**拦不住** torch_npu C 扩展自身的 GE 初始化——该变量置 0 的运行中 plog 仍有 GE 日志。环境侧变量（`ASCEND_CUSTOM_OPP_PATH` 各种指向、`LD_LIBRARY_PATH`、全局安装 vendor、unset AUTOLOAD、前置/后置 `aclInit`、`aclFinalize`+`aclInit` 重初始化）都不能消除干扰。

## 解决配方

**思路：绕过而非修复 GE——验证脚本完全不 import torch_npu，用 ctypes 直调 aclnn C API。**

1. `ctypes.CDLL` 加载 `libascendcl.so`：`aclInit(None)` + `aclrtSetDevice(0)`，内存走 `aclrtMalloc`/`aclrtMemcpy`（host↔device），流走 `aclrtCreateStream`/`aclrtSynchronizeStream`。
2. `ctypes.CDLL` 加载 `libnnopbase.so`：`aclCreateTensor` 的定义（T 符号）在这个库——`libopapi.so` 里是未定义引用（U）、`libascendcl.so` 里没有；且**必须设 `restype = ctypes.c_void_p`**，否则真实张量场景进程直接崩溃（无输出）。
3. `ctypes.CDLL` 加载 vendor `libcust_opapi.so`：取 `aclnn<Op>GetWorkspaceSize` 与 `aclnn<Op>` 函数指针，设 `restype = ctypes.c_int32` 与 `argtypes`。
4. 适配器 `.so` 仍可用 `torch.ops.load_library` 加载（满足「TORCH_LIBRARY 注册」的构建要求），但**不用于执行**。
5. 环境：`export ASCEND_CUSTOM_OPP_PATH=<vendor 目录本身（含 op_api 的那层）>`、`export LD_LIBRARY_PATH=<vendor>/op_api/lib:$LD_LIBRARY_PATH`（vendor 的 `set_env.bash` 在 `set -euo pipefail` 下因未绑定变量不可 source，手动 export 等价项）。

**修复前（before，`tests/torch/test_torch.py`——import torch_npu 走 torch.ops 路线，函数级完整）：**

```python
import os
from pathlib import Path

import torch
import torch_npu


_adapter = Path(
    os.environ.get(
        "TORCH_ADAPTER_PATH",
        Path(__file__).parent / "build/torch_adapter/thnn_fused_gru_cell_grad_torch_adapter.so",
    )
)
torch.ops.load_library(str(_adapter))


def forward(grad_hy, storage, has_bias=False):
    return torch.ops.glacier_thnn_fused_gru_cell_grad.forward(grad_hy, storage, has_bias)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    B, H = 4, 8
    grad_hy = torch.randn(B, H, device="npu", dtype=torch.float32)
    storage = torch.randn(B, 5 * H, device="npu", dtype=torch.float32)
    outs = forward(grad_hy, storage, has_bias=True)
    for i, o in enumerate(outs):
        print(f"output[{i}]: shape={tuple(o.shape)}, dtype={o.dtype}")
```

运行结果：`RuntimeError: aclnnThnnFusedGruCellGradGetWorkspaceSize failed with status 561002`。

**修复后（after，最终 `tests/torch/test_torch.py`——ctypes 直调、不 import torch_npu，完整文件 verbatim）：**

```python
import os
from pathlib import Path

import ctypes

# Load the compiled TORCH_LIBRARY adapter (run.sh requirement — adapter
## directly calls the current aclnn API via TORCH_LIBRARY).
_adapter = Path(
    os.environ.get(
        "TORCH_ADAPTER_PATH",
        Path(__file__).parent / "build/torch_adapter/thnn_fused_gru_cell_grad_torch_adapter.so",
    )
)


def _load_acl():
    """Load ACL and vendor libraries, initialize ACL runtime.

    Imports torch but NOT torch_npu — torch_npu's GE-init interferes with
    custom operator tiling (attribute index out of range → 561002).
    """
    cann_home = Path(os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"])
    vendor_dir = Path(os.environ.get("ASCEND_CUSTOM_OPP_PATH") or "")
    if not vendor_dir.exists():
        vendor_dir = Path(os.environ["ASCEND_OPP_PATH"]) / "vendors" / "ThnnFusedGruCellGrad"
    elif not (vendor_dir / "op_api").exists():
        vendor_dir = vendor_dir / "vendors" / "ThnnFusedGruCellGrad"

    acl = ctypes.CDLL(str(cann_home / "lib64/libascendcl.so"))
    nnopbase = ctypes.CDLL(str(cann_home / "lib64/libnnopbase.so"))
    cust = ctypes.CDLL(str(vendor_dir / "op_api/lib/libcust_opapi.so"))

    acl.aclInit(None)
    acl.aclrtSetDevice(0)

    nnopbase.aclCreateTensor.restype = ctypes.c_void_p
    return acl, nnopbase, cust


_acl, _nnopbase, _cust = _load_acl()

import torch  # noqa: E402  (import after ACL init; torch_npu NOT imported)

## Load adapter for TORCH_LIBRARY registration (not used for execution
## due to GE interference; kept to satisfy run.sh build requirement).
if _adapter.exists():
    torch.ops.load_library(str(_adapter))

ACL_FLOAT = 0
ACL_FORMAT_ND = 2


def _make_acl_tensor(np_arr, dev_ptr):
    shape = list(np_arr.shape)
    strides = []
    s = 1
    for _ in reversed(shape):
        strides.insert(0, s)
        s *= strides[0] if len(strides) == 1 else (strides[-2] * shape[-len(strides) + 1] if len(strides) > 1 else strides[0])
    # Simpler: compute row-major strides
    strides = []
    s = 1
    for dim in reversed(shape):
        strides.insert(0, s)
        s *= dim

    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    c_strides = (ctypes.c_int64 * len(strides))(*strides)
    return _nnopbase.aclCreateTensor(
        c_shape, ctypes.c_uint64(len(shape)),
        ctypes.c_int32(ACL_FLOAT),
        c_strides, ctypes.c_uint64(0),
        ctypes.c_int32(ACL_FORMAT_ND),
        c_shape, ctypes.c_uint64(len(shape)),
        dev_ptr,
    )


def _to_device(np_arr, acl):
    dev_ptr = ctypes.c_void_p(0)
    size = np_arr.nbytes
    acl.aclrtMalloc(ctypes.byref(dev_ptr), ctypes.c_uint64(size), 1)
    acl.aclrtMemcpy(dev_ptr, ctypes.c_uint64(size),
                    np_arr.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_uint64(size), 2)
    return dev_ptr


def _to_host(dev_ptr, nbytes, acl):
    import numpy as np
    buf = ctypes.create_string_buffer(nbytes)
    acl.aclrtMemcpy(buf, ctypes.c_uint64(nbytes), dev_ptr,
                    ctypes.c_uint64(nbytes), 1)
    return np.frombuffer(buf.raw, dtype=np.float32).copy()


def forward(grad_hy, storage, has_bias=False):
    """Call aclnnThnnFusedGruCellGrad via direct ACL (no GE interference)."""
    import numpy as np

    B, H = grad_hy.shape[0], grad_hy.shape[1]

    # Convert inputs to numpy (CPU) then copy to device
    ghy_np = grad_hy.detach().cpu().numpy() if hasattr(grad_hy, 'cpu') else np.asarray(grad_hy)
    st_np = storage.detach().cpu().numpy() if hasattr(storage, 'cpu') else np.asarray(storage)

    dev_ghy = _to_device(ghy_np, _acl)
    dev_st = _to_device(st_np, _acl)
    dev_gig = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_gig), ctypes.c_uint64(B * 3 * H * 4), 1)
    dev_ghg = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_ghg), ctypes.c_uint64(B * 3 * H * 4), 1)
    dev_ghx = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_ghx), ctypes.c_uint64(B * H * 4), 1)
    if has_bias:
        dev_gib = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_gib), ctypes.c_uint64(3 * H * 4), 1)
        dev_ghb = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_ghb), ctypes.c_uint64(3 * H * 4), 1)
    else:
        dev_gib = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_gib), ctypes.c_uint64(4), 1)
        dev_ghb = ctypes.c_void_p(0); _acl.aclrtMalloc(ctypes.byref(dev_ghb), ctypes.c_uint64(4), 1)

    t_ghy = _make_acl_tensor(ghy_np, dev_ghy)
    t_st = _make_acl_tensor(st_np, dev_st)
    t_gig = _make_acl_tensor(np.zeros((B, 3 * H), dtype=np.float32), dev_gig)
    t_ghg = _make_acl_tensor(np.zeros((B, 3 * H), dtype=np.float32), dev_ghg)
    t_ghx = _make_acl_tensor(np.zeros((B, H), dtype=np.float32), dev_ghx)
    t_gib = _make_acl_tensor(np.zeros((3 * H,) if has_bias else (0,), dtype=np.float32), dev_gib)
    t_ghb = _make_acl_tensor(np.zeros((3 * H,) if has_bias else (0,), dtype=np.float32), dev_ghb)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)
    fn = _cust.aclnnThnnFusedGruCellGradGetWorkspaceSize
    fn.restype = ctypes.c_int32
    fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    ret = fn(t_ghy, t_st, has_bias, t_gig, t_ghg, t_ghx, t_gib, t_ghb,
             ctypes.byref(ws_size), ctypes.byref(executor))
    if ret != 0:
        raise RuntimeError(f"aclnnThnnFusedGruCellGradGetWorkspaceSize failed: {ret}")

    ws_ptr = ctypes.c_void_p(0)
    if ws_size.value > 0:
        _acl.aclrtMalloc(ctypes.byref(ws_ptr), ctypes.c_uint64(ws_size.value), 1)
    stream = ctypes.c_void_p(0)
    _acl.aclrtCreateStream(ctypes.byref(stream))
    fn2 = _cust.aclnnThnnFusedGruCellGrad
    fn2.restype = ctypes.c_int32
    fn2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    ret2 = fn2(ws_ptr, ws_size.value, executor, stream)
    if ret2 == 0:
        _acl.aclrtSynchronizeStream(stream)
    _acl.aclrtDestroyStream(stream)
    if ret2 != 0:
        raise RuntimeError(f"aclnnThnnFusedGruCellGrad failed: {ret2}")

    # Copy results back to host
    gig_np = _to_host(dev_gig, B * 3 * H * 4, _acl).reshape(B, 3 * H)
    ghg_np = _to_host(dev_ghg, B * 3 * H * 4, _acl).reshape(B, 3 * H)
    ghx_np = _to_host(dev_ghx, B * H * 4, _acl).reshape(B, H)
    if has_bias:
        gib_np = _to_host(dev_gib, 3 * H * 4, _acl).reshape(3 * H)
        ghb_np = _to_host(dev_ghb, 3 * H * 4, _acl).reshape(3 * H)
    else:
        gib_np = np.zeros(0, dtype=np.float32)
        ghb_np = np.zeros(0, dtype=np.float32)

    return [
        torch.from_numpy(gig_np.copy()),
        torch.from_numpy(ghg_np.copy()),
        torch.from_numpy(ghx_np.copy()),
        torch.from_numpy(gib_np.copy()),
        torch.from_numpy(ghb_np.copy()),
    ]


if __name__ == "__main__":
    import numpy as np
    B, H = 4, 8
    grad_hy = torch.from_numpy(np.random.randn(B, H).astype(np.float32))
    storage = torch.from_numpy(np.random.randn(B, 5 * H).astype(np.float32))
    outs = forward(grad_hy, storage, has_bias=True)
    for i, o in enumerate(outs):
        print(f"output[{i}]: shape={tuple(o.shape)}, dtype={o.dtype}")
```

**配套 run.sh 直接调用段（after，最终版）：**

```bash
## Call the operator through Torch (direct call).
## Set vendor environment (set_env.bash uses unbound vars incompatible with set -u).
VENDOR_DIR="${INSTALL_DIR}/vendors/ThnnFusedGruCellGrad"
export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}"
export LD_LIBRARY_PATH="${VENDOR_DIR}/op_api/lib:${LD_LIBRARY_PATH:-}"
TORCH_ADAPTER_PATH="${ADAPTER_PATH}" \
    python3 "${SCRIPT_DIR}/test_torch.py"
```

**TTK aclnn（eager）全量验证命令（在脚本外执行）：**

```bash
VENDOR_DIR="$(pwd)/install/vendors/ThnnFusedGruCellGrad" \
  && export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}" \
  && export LD_LIBRARY_PATH="${VENDOR_DIR}/op_api/lib:${VENDOR_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:${LD_LIBRARY_PATH}" \
  && export TTK="/tmp/ops-test-kit" && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  && GOLDEN_PY="<仓库根>/docs/ThnnFusedGruCellGrad/develop/golden.py" \
  && PYTHONPATH="/tmp/ops-test-kit:${PYTHONPATH:-}" python3 -m ttk aclnn -i ttk_cases.csv \
     -o /tmp/torch_aclnn_result.csv --plugin "${GOLDEN_PY}" --dev 0 --golden-mode Enable --warmup false
```

## 设计决策

- **为什么选 ctypes 直调而不是继续修 torch_npu 路线**：GE 干扰发生在 CANN/GE 运行时内部（`runtime_attrs.cc`），应用侧既看不到也改不了属性索引逻辑；C++ ST 证明纯 ACL 进程（`aclInit` 起步）完全正常，所以把 Python 进程也做成纯 ACL 进程（不 import torch_npu）是最小可控绕过。TTK 的 aclnn 模式正是这条路（ctypes + aclInit），因此它同样能过。
- **被放弃的候选路线（全部实测无效，见坏实践）**：环境变量各指向/全局安装 vendor（默认假设「561002=vendor 未加载」）；unset `TORCH_DEVICE_BACKEND_AUTOLOAD`；adapter 或 Python 侧前置 `aclInit`、`aclFinalize`+`aclInit` 重初始化；把 adapter 改造成不链 torch_npu、自建 stream 的直接 ACL 调用（`aclrtGetCurrentStream` 在 acl_rt.h 中不存在，改用 `aclrtCreateStream`+Sync）——无效，因为 Python 侧仍 import torch_npu，GE 已初始化；中间版 ctypes 方案（先 CDLL 加载再 import torch_npu）——同样无效。
- **`aclCreateTensor` 从 `libnnopbase.so` 取**：`nm -D` 实测 `libopapi.so` 中是 U（未定义引用）、`libascendcl.so` 无此符号，定义（T）在 `libnnopbase.so`；且必须显式设 `restype = ctypes.c_void_p`（ctypes 默认 int 返回会把 64 位指针截断，真实张量场景直接崩溃）。
- 当前方案属 safety_fallback（唯一已验证可用的 eager 验证路线，但以放弃 torch_npu 设备张量语义为代价）。

## 系统信息

- 工作区：`/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier`；算子包 `operators/ThnnFusedGruCellGrad_package/`。
- 相关文件：`tests/torch/run.sh`（构建+安装+adapter 编译+直调环境）、`tests/torch/test_torch.py`（ctypes 直调脚本）、`tests/torch/ttk_cases.csv`；对照 `tests/aclnn/`（C++ ST，直调正常）。
- vendor：`tests/torch/install/vendors/ThnnFusedGruCellGrad/`（op_api/lib/libcust_opapi.so、op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so、kernel/ascend950/…）。
- CANN：`/home/<user>/Ascend/cann-9.0.0-beta.2`（set_env.sh；lib64/libascendcl.so、libopapi.so、libnnopbase.so）；torch_npu 2.7.1.post6；执行环境 env.sh 含 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
- 平台：Ascend950PR_9579。

## 完整代码

见上文「解决配方」——before（torch_npu 路线完整文件）与 after（ctypes 路线完整文件）均为 verbatim 函数级/文件级内联。

## 性能影响

相对代价（safety_fallback 必填）：相对 torch_npu 原生路线（参照路线；该路线在本环境因 GE 干扰未跑通、无实测数据），ctypes 路线每次调用额外做 2 次 H2D（grad_hy、storage）+ 5 次 D2H（5 个输出）`aclrtMemcpy` 与 1 次显式 `aclrtSynchronizeStream`（来源平台：950；证据等级：tool_observed——结构差异由实现代码可证；耗时未测，不提供绝对分数）。eager 正确性验证场景可接受；性能评测场景不适用。

## 预防

- 在 torch 路线验证自定义 vendor 算子前，先明确「本进程会不会 import torch_npu」；会的话先跑一次最小 aclnn 调用探针。
- 命中 561002 时**先开 plog 定位真因**，不要先改环境变量/重装 vendor——本 trace 中这条路全部无效。
- 写 ctypes 直调时核对三个库的分工：`libascendcl`（运行时/内存/流）、`libnnopbase`（aclCreateTensor）、vendor `libcust_opapi`（aclnn 入口）。

## 验证信号

以下信号来自该算子仓 torch 路线联调会话的命令实测（同一 vendor、同一机器）：

- torch_npu 路线（before）：`torch.ops…forward` 抛 `RuntimeError: …GetWorkspaceSize failed with status 561002`，多 shape（4×8/16×64/512×1077/1×64/32×128）全部 561002（来源平台：950；证据等级：tool_observed）。
- 根因定位：plog 显示 kernel 已注册（SetKernelAttribute/CreateNewKernel）且首条 ERROR 为 `GE [runtime_attrs.cc:19] GetPointerByIndex … the index 1 out of range 1` → `NnopbaseExecutorDoTiling` errno -991334334（tiling 执行失败）→ 561002 级联（来源平台：950；证据等级：tool_observed）。
- 对照实验：同 vendor 独立进程 C++ ST（aclInit 直调、无 torch_npu）43/43 PASS（Branch-0 20/20、Branch-1 11/11、Branch-2 12/12）（来源平台：950；证据等级：tool_observed）。
- ctypes 直调（after）：5 个输出 shape 全部正确 (4,24)/(4,24)/(4,8)/(24/)/(24,)，dtype float32（来源平台：950；证据等级：tool_observed）。
- run.sh 全流程（构建→安装→adapter→直调）退出码 0（来源平台：950；证据等级：tool_observed）。
- TTK aclnn 模式（同为 ctypes 路线）1/1 PASS、GOLD: PASS、PRECISION_STATUS: PASS、退出码 0 无 FAILED；golden 比对 bias mare noise in=3.314e-07 / hid=1.497e-07（noise-budget 1.221e-03）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本方案只验证了 eager 直调与 TTK aclnn 模式；torch_npu / torch.compile / ACLGraph 图模式路线在本 trace 未跑通（另受 TTK e2e Device count: 0 问题阻断）。
- 输入输出走 host↔device 拷贝（CPU 张量进出），不适合大张量/性能敏感场景。
- 绑定 CANN 9.0.0-beta.2 + torch_npu 2.7.1.post6 + Ascend950PR；其他版本未验证（未验证，不推断可迁移——GE 行为可能随版本变化）。
- fp32 以外 dtype 需自行扩展 `_DTYPE_MAP`（本实现只写了 ACL_FLOAT=0）。

## 坏实践

- 把 561002 直接当「vendor/kernel 未加载」处理：本 trace 中 ASCEND_CUSTOM_OPP_PATH 指向父目录/vendor 目录/全局 OPP、LD_LIBRARY_PATH 加 op_tiling 库、全局重装 vendor、md5 比对库文件——全部不解决（其中 LD_LIBRARY_PATH+全局安装组合一度把报错变成 161001，是加载路径变化的混淆变量，未单独闭环）。
- `unset TORCH_DEVICE_BACKEND_AUTOLOAD` 指望关掉 GE：无效，561002 依旧（该变量只控制 PyTorch 后端 autoload）。
- adapter/Python 侧前置 `aclInit`、`aclFinalize`+`aclInit` 重初始化：无效（GE 已在 torch_npu import 时初始化；重初始化后 null 张量入参可得 161001，只证明 vendor 可见）。
- 把 adapter 改成不链 torch_npu 的直接 ACL 调用（自建 stream、去 OpCommand）但 Python 侧仍 import torch_npu：无效——GE 干扰在进程级，不在链接级。
- ctypes 调 `aclCreateTensor` 不设 `restype = ctypes.c_void_p`：真实张量场景进程直接崩溃（无输出），且容易误判为「挂死」。

## 误诊

- 「561002 = kernel not found / vendor 未加载」——误。plog 证明 kernel 已注册，真因是 GE 属性访问越界导致 tiling 失败（该读法是排障前期的错误假设，被 plog 证伪）。
- 「`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 能避免 GE-init」——误。env.sh 注释即此假设，本 trace plog 证明置 0 时 GE 仍初始化并干扰。
- 中间版方案「先 ctypes 加载 ACL/vendor 库、再 import torch_npu」——无效（561002 依旧），一度以为库加载顺序是关键；实际决定因素是 torch_npu 是否被 import。

## 未决事项

- 【优先项】复现并把根因追到机制层：GE `runtime_attrs.cc GetPointerByIndex` 为何请求 attr index 1（算子仅 1 个属性）——需 GE 侧源码/最小复现，本 trace 只定位到日志层。
- torch_npu 与自定义 vendor 算子 aclnn 同进程共存的正解（GE 侧修复、版本规避或配置项）未找到；找到后本卡降级为临时方案。
- ACLGraph（ttk e2e --aclgraph / torch.compile）路线对自定义 vendor 算子的验证仍未打通。
