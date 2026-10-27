---
type: "Runbook"
title: is_available=True 仍 507033：NPU 可用性须实测探测
description: torch.npu.is_available() 返回 True 不代表设备可用（SetDevice 报 507033 Failed to start the device）；测试框架须实际建 tensor + 同步探测，并用 507033/107001、/dev/devmm_svm 缺失、npu-smi -8010 等信号判别"可见但不可启动"的环境故障，SKIP 而非逐条 FAIL。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# is_available=True 仍 507033：NPU 可用性须实测探测

## 现象

自定义算子的 PyTorch ST 测试工程首次全量运行（687 条用例，Real 模式）：CPU golden 自测 7/7 通过后，测试框架打印 `NPU 可用: True / NPU 设备数: 1`（基于 `torch.npu.is_available()`），随即第一条用例 L0-001 在 NPU 执行时抛出：

```text
[FAIL] NPU 执行异常: LazySetDevice:../torch_npu/csrc/core/npu/NPUFunctions.cpp:342 NPU function error: err, error code is 507033
[ERROR] ... ERR00100 PTA call acl api failed
[Error]: Failed to start the device.
[PID: ...] Invalid_Argument(EE1001): ... rtGetDevMsg execution failed, the context is a null pointer.
TraceBack (most recent call last):
        TsdOpen failed. devId=0, tdt error=1.[FUNC:PrintfTsdError][FILE:runtime.cc][LINE:2135]
        Check param failed, dev can not be NULL![FUNC:DeviceRetain][FILE:runtime.cc][LINE:3162]
        rtSetDevice execution failed, reason=device retain error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:69]
```

同时 `npu-smi info` 返回 `Failed to execute npu-smi info. ret is -8010.`。此时若信 `is_available()==True` 继续跑，687 条会逐条失败，且极易误判为算子/适配层代码缺陷。后续精度验收与检视修复两个阶段在同机上再次复现同一状态（`aclrtSetDevice(0)` 恒 507033）。

## 适用场景

- 工程形态：ST/UT 测试框架、PyTorch 适配层、任何需要在跑批前判 NPU 可用性的脚本。
- API 路径：`torch.npu.is_available()/set_device()`、`aclInit/aclrtSetDevice/aclrtGetDeviceCount`（libascendcl.so）、`npu-smi`、`/dev/davinci*`、`/dev/devmm_svm`、`ASCEND_RT_VISIBLE_DEVICES`。
- 验证过的平台与 CANN 版本：Ascend950PR（容器环境，宿主 8×d806），CANN 9.1.0，torch 2.7.1+cpu + torch_npu。
- 适用粒度：platform_wide。
- 路线优先级：recommended_fastpath（探测代码模式）。
- 明确不适用 / 未验证场景：SetDevice 成功而算子执行失败的排查（非本卡范围）；驱动重装/设备 reset 等恢复手段未验证（reset 需 root，本环境无权限）。

## 判别方法

**第一层（必须做）：实测探测，不信 is_available。** 用 tensor 创建 + 同步代替 `is_available()`：

```python
# python3 -c 形式探针
import torch
import torch_npu
print('torch.npu.is_available():', torch.npu.is_available())
print('device_count:', torch.npu.device_count())
try:
    torch.npu.set_device(0)
    t = torch.tensor([1.0], device='npu')
    print('cpu:', t.cpu())
    print('NPU OK')
except Exception as e:
    print('NPU ERROR:', type(e).__name__, str(e)[:500])
```

本例输出：`is_available(): True / device_count: 1 / NPU ERROR: ... error code is 507033`——可见性与可用性分离。

**第二层：ctypes 直调 ACL，绕开 torch_npu 归因。**

```python
python3 -c "
import ctypes, os
lib = ctypes.CDLL(os.path.join(os.environ['ASCEND_HOME_PATH'], 'lib64/libascendcl.so'))
lib.aclInit.restype = ctypes.c_int32
ret = lib.aclInit(None)
print('aclInit ret=', ret)
lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
lib.aclrtSetDevice.restype = ctypes.c_int32
for d in [0, 4]:
    r = lib.aclrtSetDevice(d)
    print(f'aclrtSetDevice({d}) ret=', r)
"
```

**第三层：C 探针 + 设备节点检查。**

```cpp
// g++ test.cpp -I${ASCEND_HOME_PATH}/include -I${ASCEND_HOME_PATH}/x86_64-linux/include \
//   -L${ASCEND_HOME_PATH}/x86_64-linux/lib64 -lascendcl -o test && LD_LIBRARY_PATH=... ./test
#include "acl/acl.h"
#include <cstdio>
int main() {
    aclInit(nullptr);
    uint32_t count = 0;
    aclrtGetDeviceCount(&count);
    printf("DeviceCount: %u\n", count);
    for (uint32_t i = 0; i < count + 4; i++) {
        auto r = aclrtSetDevice(i);
        printf("aclrtSetDevice(%u): %d\n", i, (int)r);
        if (r == 0) aclrtResetDevice(i);
    }
    aclFinalize();
    return 0;
}
```

```bash
ls -la /dev/davinci* /dev/devmm_svm 2>&1
npu-smi info -t common -i 0 2>&1
```

**观测结果矩阵（本例实测）**：

| 探针 | 输出 | 含义 |
|---|---|---|
| `aclInit` | ret=0 | ACL 库加载正常 |
| `aclrtGetDeviceCount` | 1 | 运行时可见 1 个设备 |
| `aclrtSetDevice(0)` | 507033 | 设备存在但启动失败 |
| `aclrtSetDevice(1..7)` | 107001 | 设备号无效 |
| `ls /dev/davinci*` | 仅 `davinci4` + `davinci_manager`，`/dev/devmm_svm` 不存在 | 容器设备节点缺失（devmm_svm 是内存管理必需节点） |
| `npu-smi info` | ret -8010 / 挂起；`-t common -i 0` 指标全 0 | 驱动通信异常 |
| `ASCEND_RT_VISIBLE_DEVICES=4` | `is_available: False`，`aclInit 107001 Invalid device ID` | 逻辑设备号与物理节点号不映射，不能靠该环境变量选中 davinci4 |

**解释**：`is_available()` 只反映驱动可见性，不保证设备可启动。`aclInit=0 + SetDevice(0)=507033 + 其他设备号 107001 + /dev/devmm_svm 缺失` 组合指向容器/驱动层环境故障（设备节点不全、驱动态异常），不是算子代码或适配层问题——应 SKIP NPU 用例并回退 CPU golden/UT/Mock 验证，而不是继续跑批。

## 解决配方

测试框架探测逻辑修复。修复前（test.py 原始探测，被 `is_available()` 欺骗）：

```python
    # 检查 NPU 可用性
    npu_available = torch.npu.is_available()
    print(f"NPU 可用: {npu_available}")
    if npu_available:
        print(f"NPU 设备数: {torch.npu.device_count()}")
```

修复后（实际建 tensor + 同步触发）：

```python
    # 检查 NPU 可用性（不仅检查 is_available，还尝试实际创建 tensor）
    npu_available = False
    if torch.npu.is_available():
        try:
            torch.npu.set_device(0)
            _test_tensor = torch.tensor([1.0], device="npu")
            _test_tensor.cpu()  # 触发同步
            npu_available = True
        except Exception as e:
            print(f"[WARN] NPU is_available=True 但设备不可用: {e}")
            npu_available = False
    print(f"NPU 可用: {npu_available}")
    if npu_available:
        print(f"NPU 设备数: {torch.npu.device_count()}")
```

配套退出码语义（NPU 不可用时用例 SKIP 而非 FAIL，golden 自测通过即 0）：

```python
    if skipped_count > 0 and not npu_available:
        print(f"  (NPU 不可用，{skipped_count} 条用例跳过 NPU 执行)")
    # 退出码: golden 自测通过且无失败用例即视为成功
    # (NPU 不可用时所有用例跳过，不算失败)
    return 0 if (failed_count == 0 and golden_passed) else 1
```

## 设计决策

- 探测放在框架入口而非每条用例内：一次探测、全局 SKIP，避免 687 条逐条超时失败浪费时间并污染结果。
- 用 `tensor.cpu()` 触发同步：仅 `set_device` 成功也未必代表后续执行可用，创建 + 拷回覆盖 LazySetDevice 与执行两层。
- C++ ST Real 模式同机同样 507033 后，回退链为"C++ ST 既有真机产物（raw log/验收报告）+ UT 65/65 + ST Mock 37/37"——Mock 模式（CPU golden 对照）可继续验证测试脚手架本身。
- 当前方案属 recommended_fastpath（探测与 SKIP 语义是标准做法）。

## 验证信号

来源说明：以下信号来自 PyTorch ST 工程开发、精度验收与检视修复三个阶段在同一开发机上的探测与回退记录。

- 修复前：`NPU 可用: True` 后 L0-001 执行报 507033，测试报告"总计 1 / 失败 1"（来源平台：950；证据等级：tool_observed）。
- python/ctypes/C 三层探针输出矩阵如上（`aclInit=0`、`SetDevice(0)=507033`、`SetDevice(4)=107001`、`DeviceCount=1`）（来源平台：950；证据等级：tool_observed）。
- `/dev` 节点检查：仅 `davinci4`/`davinci_manager`，`/dev/devmm_svm` 不存在；lspci 见 8×d806（来源平台：950；证据等级：tool_observed）。
- `npu-smi info -t common -i 0` 指标全 0，`set -t reset` 提示 root-only，`npu-smi info` 挂起被超时终止（来源平台：950；证据等级：tool_observed）。
- 修复后：687 条用例正确 SKIP、golden 自测 7/7、`EXIT_CODE=0`（来源平台：950；证据等级：tool_observed）。
- 回退验证：C++ ST Mock 37/37、UT 65/65 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡判别的是"设备可见但不可启动"的环境故障信号组合；若 `SetDevice` 本身返回 0 而算子执行出错，属算子/适配层问题，不适用。
- 507033 亦可能由设备被占/驱动态异常等其他原因引起——本例观测到 devmm_svm 缺失，但未在机制层归因（见遗留问题）。
- 恢复手段（驱动重装、root reset、容器重建）全部未验证；本卡只提供判别与回退，不提供修复。
- 未验证，推断可迁移：其他 CANN 版本/芯片上错误码语义（507033/107001）应一致，但未实测。

**未决项**：

- 设备进入"可见但 507033"状态的确切驱动层原因（容器 /dev/devmm_svm 缺失是否充分必要）未追到机制层——复现并追根因是优先接续项。
- 恢复手段（root 权限 reset、驱动重载、容器重建）均未验证。
## 坏实践

- 跑批前只查 `torch.npu.is_available()` 就断定 NPU 可用：可见性≠可用性，本例直接导致首条用例误报 FAIL。
- 试图用 `ASCEND_RT_VISIBLE_DEVICES=4` 选中物理节点 `/dev/davinci4`：实测 `is_available: False` + `aclInit 107001 Invalid device ID`，逻辑设备号不映射物理节点号。
- 设备不可用时把 687 条全部计为 FAIL 继续跑：既浪费时间又制造"全红"假象，掩盖真实回退路径。

## 误诊

- 初判怀疑 torch 为 CPU-only 版本（2.7.1+cpu）与 torch_npu 不兼容导致：ctypes 直调 libascendcl.so 同样 507033，排除 torch 层因素，锁定环境。
- 初判怀疑设备被其他进程占用：`ps`/进程检查无可疑占用者，且 `npu-smi -8010` 与 devmm_svm 缺失指向容器设备节点问题。
