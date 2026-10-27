---
type: "Runbook"
title: 显式负指数链未降低 Sigmoid 调用成本
description: 独立构建和等值输入测量表明，当前绑定下展开负指数基础链未形成高阶显式 scratch Sigmoid 的完整调用成本优势。
tags: [task.optimization, topic.performance.analysis, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-10-18T12:25:19Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# 显式负指数链未降低 Sigmoid 调用成本


## 现象

在 Ascend910B4 上比较一个有界一维 Sigmoid kernel 的两种实现时，开发者为 half、float 各自独立编译高阶 `AscendC::Sigmoid(yo,xi,scratch,n)` 与 `Muls → Exp → Adds → Duplicate → Div` 负指数链，每个模块只有一个 device kernel 和一个 `activation(x)` 入口。两条路线使用相同 dtype 的 VECIN、VECOUT 与 VECCALC 声明容量，同一个已冻结输入经同一 wrapper 进行运算。wrapper 每次检查输入、分配输出，并在当前 stream 启动一个 AIV block。

比较的目的是真实检验“把高阶调用展开即可减少成本”这条路线。实际执行没有构建或运行错误，A/B 输出在记录的数值控制上逐bit相同；但 FP32 的显式链完整调用成本更高，half 只有微小或方向不确定的差异。由此应停止把这次展开作为已有提速证据，保留高阶路线作为本比较的参照。

## 适用场景

- 工程形态：AscendC kernel 直调，经 PyTorch NPU pybind wrapper 的完整调用比较。
- API 路径：带显式 `LocalTensor<uint8_t>` scratch 的 `Sigmoid`、基础向量负指数链、`TPipe/TQue/TBuf`；公开 scratch 入口被实际安装头展开为 counter mask 路线。
- 实测平台：a2 / Ascend910B4；构建 compiler 与捕获头位于 CANN 9.1.0，Python 3.11.14、torch 2.9.0+cpu、torch_npu 2.9.0。运行时实际加载的 CANN 共享库未捕获，其版本仍 unknown。
- 适用粒度：`exact_task`，仅本文给出的独立 probe、输入资格、构建与 wrapper 绑定。
- 路线优先级：`banned` 仅指在本文 `exact_task` 绑定下，为降低完整调用成本而采用这次显式替换；该尝试没有建立成本优势。此边界不适用于以其他数值合同或数据流为目标的实现，高阶路线保留为本比较的回退参照。
- 未验证：其他 scratch 容量、数据布局、多核/融合业务算子、其他编译器或平台、不同计时范围、复用输出分配、正指数或 Reciprocal 重写。

## 判别方法

首先核对独立模块的实际加载 hash 与各自构建产物，核对每个源码 hash；相同函数名或同一个 runtime mode switch 不能替代这项绑定。本文实际采用四个不同 module name 和四次单入口构建。

其次只给 A/B 全部 finite 且逐bit相同的冻结 uniform 输入做成本排序。边界/非有限值、normal、wide normal 都保留作数值控制，不混入排名。CPU `torch.sigmoid(x.double()).to(dtype)` 是独立诊断参考；A/B 相同不能推出它们都满足这个参考的精度合同。

每个 ABBA 块的两次高阶观测取均值 `A=(A1+A2)/2`，两次显式观测取均值 `B=(B1+B2)/2`，该块成本比为 `B/A`；对完整块集合取中位数并自举其中位比区间。比大于 1 表示显式比参照高阶更慢。只有完整协议且 Event 与 wall 的区间方向一致时，才作该绑定下的成本方向判断。

## 根因

这次显式展开没有消除已安装高阶 scratch 实现中的负指数计算链，因而“展开高阶名即可省掉额外计算”的提速依据不成立；观测到的具体成本差异尚未归因到最终机器执行机制。

安装头中 `splitSize = sharedTmpBuffer.GetSize() / sizeof(T)`，同 dtype 分配的 scratch 元素容量一致；`loopCount = calCount / splitSize` 与 `calcTail = calCount % splitSize` 控制完整块和尾部。高阶实现使用 counter mask 与显式 `PipeBarrier<PIPE_V>()`；尝试路线使用 calCount API 与编译器 auto-sync。两条路线均表达 `1 / (1 + exp(-x))`，但这只说明源码计算链关系，不能证明相同 lowering、指令数、同步代价或 occupancy。

## 设计决策

使用独立模块避免在同一 kernel 内 mode 分支；相同 dtype 的三类 UB 声明容量一致，避免把显式 scratch 与自动 stack scratch 混为控制变量。源码声明容量仅是控制条件，未测量硬件资源占用。

选择负指数链保留高阶源码中的计算次序，不改成正指数或 Reciprocal。wrapper 每次真实计算输出；输入传输与 CPU 参考在计时外。数值资格决定哪些输入能用于成本排序，不修改参考值、容差或测试集合来制造成功。该路线当前只作为失败的提速尝试保存，没有后续成功修复。

## 系统信息

使用一轮四个独立构建，SoC target 为 Ascend910B4。legacy helper 的 CMake 配置固定 Debug，实际 verbose device flags 为 `-O3`、auto-sync、`dav-c220-vec`，不能把它称为 Release。运行选择逻辑 device 0，记录 `ASCEND_RT_VISIBLE_DEVICES=2` 和同次映射；wrapper 没有独立检查输入 device index 等于当前 stream，故不声称任意多设备安全。

下面完整保留四个 kernel、四个 wrapper、build driver/helper、运行 driver、相关已安装 Sigmoid 头，以及计时依赖完整类。它们是此次尝试的实际源码快照，代码中的固定容量和测试参数只服务于这次机制复查，不是业务算子 shape 清单或通用推荐。`source_manifest.json` 与 `PREDECLARED.md` 是 driver 所需的完整性记录元数据，内容不参与 device 计算。配套审阅证据保留 PREDECLARED 原件副本；公开 source_manifest 仅把 helper_source 改为相对路径，其余键和值保持一致，文件整体 hash 因此不同。新建复查时应记录该元数据的新 hash，不能将正文代码单独视为已经重新运行验证的复现包。此处不构造示意替代源码，也不把计时 driver 当成新 kernel 知识。

## 完整代码

### `explicit_float/kernel/probe.cpp`

```cpp
#include "kernel_operator.h"
#include "activation/sigmoid.h"

extern "C" __global__ __aicore__ void c014_explicit_float(GM_ADDR x,GM_ADDR y,uint32_t n) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using T=float;
    constexpr uint32_t capacity=4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp;
    pipe.InitBuffer(qi,1,capacity*sizeof(T));
    pipe.InitBuffer(qo,1,capacity*sizeof(T));
    pipe.InitBuffer(tmp,capacity*sizeof(T));
    AscendC::GlobalTensor<T> gx,gy;
    gx.SetGlobalBuffer((__gm__ T*)x,n);gy.SetGlobalBuffer((__gm__ T*)y,n);
    auto xi=qi.AllocTensor<T>();AscendC::DataCopy(xi,gx,n);qi.EnQue(xi);
    xi=qi.DeQue<T>();auto yo=qo.AllocTensor<T>();auto scratch=tmp.Get<uint8_t>();
    auto ones=scratch.ReinterpretCast<T>();
    AscendC::Muls(yo,xi,(T)-1,n);
    AscendC::Exp(yo,yo,n);
    AscendC::Adds(yo,yo,(T)1,n);
    AscendC::Duplicate(ones,(T)1,n);
    AscendC::Div(yo,ones,yo,n);
    qo.EnQue(yo);qi.FreeTensor(xi);yo=qo.DeQue<T>();
    AscendC::DataCopy(gy,yo,n);qo.FreeTensor(yo);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c014_explicit_float_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t n) {
    c014_explicit_float<<<bd,nullptr,s>>>(x,y,n);
}
#endif
```

### `explicit_float/kernel/pybind11.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void c014_explicit_float_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
static at::Tensor activation(const at::Tensor& x) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1,"requires NPU input");
    TORCH_CHECK(x.scalar_type()==at::kFloat,"dtype fixed by independent build");
    TORCH_CHECK(x.dim()==1&&x.is_contiguous(),"requires contiguous 1D input");
    TORCH_CHECK(x.numel()>=16&&x.numel()<=4096&&x.numel()%16==0,"n must be a multiple of 16 in [16,4096]");
    auto y=at::empty_like(x);auto s=c10_npu::getCurrentNPUStream().stream(false);
    c014_explicit_float_do(1,s,ptr(x),ptr(y),static_cast<uint32_t>(x.numel()));
    return y;
}
PYBIND11_MODULE(_c014_explicit_float,m){m.def("activation",&activation);}
```

### `explicit_half/kernel/probe.cpp`

```cpp
#include "kernel_operator.h"
#include "activation/sigmoid.h"

extern "C" __global__ __aicore__ void c014_explicit_half(GM_ADDR x,GM_ADDR y,uint32_t n) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using T=half;
    constexpr uint32_t capacity=4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp;
    pipe.InitBuffer(qi,1,capacity*sizeof(T));
    pipe.InitBuffer(qo,1,capacity*sizeof(T));
    pipe.InitBuffer(tmp,capacity*sizeof(T));
    AscendC::GlobalTensor<T> gx,gy;
    gx.SetGlobalBuffer((__gm__ T*)x,n);gy.SetGlobalBuffer((__gm__ T*)y,n);
    auto xi=qi.AllocTensor<T>();AscendC::DataCopy(xi,gx,n);qi.EnQue(xi);
    xi=qi.DeQue<T>();auto yo=qo.AllocTensor<T>();auto scratch=tmp.Get<uint8_t>();
    auto ones=scratch.ReinterpretCast<T>();
    AscendC::Muls(yo,xi,(T)-1,n);
    AscendC::Exp(yo,yo,n);
    AscendC::Adds(yo,yo,(T)1,n);
    AscendC::Duplicate(ones,(T)1,n);
    AscendC::Div(yo,ones,yo,n);
    qo.EnQue(yo);qi.FreeTensor(xi);yo=qo.DeQue<T>();
    AscendC::DataCopy(gy,yo,n);qo.FreeTensor(yo);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c014_explicit_half_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t n) {
    c014_explicit_half<<<bd,nullptr,s>>>(x,y,n);
}
#endif
```

### `explicit_half/kernel/pybind11.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void c014_explicit_half_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
static at::Tensor activation(const at::Tensor& x) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1,"requires NPU input");
    TORCH_CHECK(x.scalar_type()==at::kHalf,"dtype fixed by independent build");
    TORCH_CHECK(x.dim()==1&&x.is_contiguous(),"requires contiguous 1D input");
    TORCH_CHECK(x.numel()>=16&&x.numel()<=4096&&x.numel()%16==0,"n must be a multiple of 16 in [16,4096]");
    auto y=at::empty_like(x);auto s=c10_npu::getCurrentNPUStream().stream(false);
    c014_explicit_half_do(1,s,ptr(x),ptr(y),static_cast<uint32_t>(x.numel()));
    return y;
}
PYBIND11_MODULE(_c014_explicit_half,m){m.def("activation",&activation);}
```

### `high_float/kernel/probe.cpp`

```cpp
#include "kernel_operator.h"
#include "activation/sigmoid.h"

extern "C" __global__ __aicore__ void c014_high_float(GM_ADDR x,GM_ADDR y,uint32_t n) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using T=float;
    constexpr uint32_t capacity=4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp;
    pipe.InitBuffer(qi,1,capacity*sizeof(T));
    pipe.InitBuffer(qo,1,capacity*sizeof(T));
    pipe.InitBuffer(tmp,capacity*sizeof(T));
    AscendC::GlobalTensor<T> gx,gy;
    gx.SetGlobalBuffer((__gm__ T*)x,n);gy.SetGlobalBuffer((__gm__ T*)y,n);
    auto xi=qi.AllocTensor<T>();AscendC::DataCopy(xi,gx,n);qi.EnQue(xi);
    xi=qi.DeQue<T>();auto yo=qo.AllocTensor<T>();auto scratch=tmp.Get<uint8_t>();
    AscendC::Sigmoid(yo,xi,scratch,n);
    qo.EnQue(yo);qi.FreeTensor(xi);yo=qo.DeQue<T>();
    AscendC::DataCopy(gy,yo,n);qo.FreeTensor(yo);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c014_high_float_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t n) {
    c014_high_float<<<bd,nullptr,s>>>(x,y,n);
}
#endif
```

### `high_float/kernel/pybind11.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void c014_high_float_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
static at::Tensor activation(const at::Tensor& x) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1,"requires NPU input");
    TORCH_CHECK(x.scalar_type()==at::kFloat,"dtype fixed by independent build");
    TORCH_CHECK(x.dim()==1&&x.is_contiguous(),"requires contiguous 1D input");
    TORCH_CHECK(x.numel()>=16&&x.numel()<=4096&&x.numel()%16==0,"n must be a multiple of 16 in [16,4096]");
    auto y=at::empty_like(x);auto s=c10_npu::getCurrentNPUStream().stream(false);
    c014_high_float_do(1,s,ptr(x),ptr(y),static_cast<uint32_t>(x.numel()));
    return y;
}
PYBIND11_MODULE(_c014_high_float,m){m.def("activation",&activation);}
```

### `high_half/kernel/probe.cpp`

```cpp
#include "kernel_operator.h"
#include "activation/sigmoid.h"

extern "C" __global__ __aicore__ void c014_high_half(GM_ADDR x,GM_ADDR y,uint32_t n) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using T=half;
    constexpr uint32_t capacity=4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp;
    pipe.InitBuffer(qi,1,capacity*sizeof(T));
    pipe.InitBuffer(qo,1,capacity*sizeof(T));
    pipe.InitBuffer(tmp,capacity*sizeof(T));
    AscendC::GlobalTensor<T> gx,gy;
    gx.SetGlobalBuffer((__gm__ T*)x,n);gy.SetGlobalBuffer((__gm__ T*)y,n);
    auto xi=qi.AllocTensor<T>();AscendC::DataCopy(xi,gx,n);qi.EnQue(xi);
    xi=qi.DeQue<T>();auto yo=qo.AllocTensor<T>();auto scratch=tmp.Get<uint8_t>();
    AscendC::Sigmoid(yo,xi,scratch,n);
    qo.EnQue(yo);qi.FreeTensor(xi);yo=qo.DeQue<T>();
    AscendC::DataCopy(gy,yo,n);qo.FreeTensor(yo);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c014_high_half_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t n) {
    c014_high_half<<<bd,nullptr,s>>>(x,y,n);
}
#endif
```

### `high_half/kernel/pybind11.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void c014_high_half_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
static at::Tensor activation(const at::Tensor& x) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1,"requires NPU input");
    TORCH_CHECK(x.scalar_type()==at::kHalf,"dtype fixed by independent build");
    TORCH_CHECK(x.dim()==1&&x.is_contiguous(),"requires contiguous 1D input");
    TORCH_CHECK(x.numel()>=16&&x.numel()<=4096&&x.numel()%16==0,"n must be a multiple of 16 in [16,4096]");
    auto y=at::empty_like(x);auto s=c10_npu::getCurrentNPUStream().stream(false);
    c014_high_half_do(1,s,ptr(x),ptr(y),static_cast<uint32_t>(x.numel()));
    return y;
}
PYBIND11_MODULE(_c014_high_half,m){m.def("activation",&activation);}
```

### `build_all.py`

```python
#!/usr/bin/env python3
"""Single build round for the four predeclared standalone variants."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("high_half", "explicit_half", "high_float", "explicit_float")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"argv": argv, "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"argv": argv, "exception": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soc-version", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.resolve().relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    if any((ROOT / variant / "kernel/build").exists() for variant in VARIANTS):
        raise RuntimeError("Existing build directory: refuse an undeclared rebuild or clean")
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    install = Path(os.environ["ASCEND_INSTALL_PATH"]).resolve()
    if cann != install:
        raise RuntimeError("ASCEND_HOME_PATH and ASCEND_INSTALL_PATH disagree")
    ccec = cann / "bin/ccec"
    manifest = {"schema_version": "c014.independent_build_round.v1", "origin": "new_predeclared_bounded_local_probe",
                "python": sys.version, "platform": platform.platform(), "soc_version": args.soc_version,
                "actual_cann_path": str(cann), "compiler_path": str(ccec), "compiler_realpath": str(ccec.resolve()),
                "compiler_sha256": sha(ccec), "compiler_version": observe([str(ccec), "--version"]),
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("ASCEND", "PATH", "LD_LIBRARY_PATH"))},
                "helper_sha256": sha(ROOT / "build_ascendc.py"), "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
                "build_environment_overrides": {"VERBOSE": "1"},
                "headers": [], "variants": [], "started_unix": time.time()}
    for relative in ("aarch64-linux/asc/include/adv_api/activation/sigmoid.h",
                     "aarch64-linux/asc/include/adv_api/sigmoid/sigmoid.h",
                     "aarch64-linux/asc/impl/adv_api/detail/activation/sigmoid/sigmoid_common_impl.h",
                     "aarch64-linux/asc/impl/adv_api/detail/activation/sigmoid/sigmoid_impl.h"):
        source = cann / relative
        dest = out / "headers" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest["headers"].append({"installed_path": str(source), "realpath": str(source.resolve()), "captured": str(dest.relative_to(ROOT)), "sha256": sha(source)})
    (out / "prebuild_environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for variant in VARIANTS:
        argv = [sys.executable, str(ROOT / "build_ascendc.py"), str(ROOT / variant), "--soc-version", args.soc_version]
        log = out / f"{variant}.log"
        before = time.monotonic()
        build_env = os.environ.copy()
        build_env["VERBOSE"] = "1"
        with log.open("w") as f:
            child = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=build_env)
        build_dir = ROOT / variant / "kernel/build"
        artifacts = []
        if build_dir.is_dir():
            for file in sorted(build_dir.rglob("*")):
                if not file.is_file():
                    continue
                if file.suffix in (".so", ".o", ".a") or file.name in ("compile_commands.json", "flags.make", "build.make", "CMakeCache.txt", "Labels.json") or "auto_gen/kernels" in str(file):
                    artifacts.append({"path": str(file.relative_to(ROOT)), "bytes": file.stat().st_size, "sha256": sha(file)})
        manifest["variants"].append({"variant": variant, "argv": argv, "rc": child.returncode,
                                     "elapsed_seconds": time.monotonic() - before, "log": str(log.relative_to(ROOT)),
                                     "log_sha256": sha(log), "artifacts": artifacts})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if child.returncode:
            raise RuntimeError(f"{variant} build failed; inspect preserved log; no automatic repair")
    manifest["finished_unix"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "variants": [{"variant": r["variant"], "rc": r["rc"], "elapsed_seconds": r["elapsed_seconds"]} for r in manifest["variants"]]}))


if __name__ == "__main__":
    main()
```

### `build_ascendc.py`

```python
#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _detect_ascend_path() -> Path:
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[-1]


def _find_kernel_sources(kernel_dir: Path) -> list[Path]:
    sources = sorted(
        path for path in kernel_dir.glob("*.cpp")
        if path.name != "pybind11.cpp"
    )
    if not sources:
        raise FileNotFoundError(f"No kernel .cpp sources found in {kernel_dir}")
    return sources


def _extract_pybind_module_name(pybind_path: Path) -> str:
    content = pybind_path.read_text(encoding="utf-8")
    match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    if not match:
        raise ValueError(f"Unable to detect PYBIND11_MODULE name from {pybind_path}")
    return match.group(1)


def _pybind11_includes() -> str:
    """返回 `-I<python> -I<pybind11>` 形式的 include flags。

    优先级:
      1. torch 自带的 pybind11（`<torch>/include/pybind11/`）—— 与 libtorch 编译时
         使用的 pybind11 ABI 对齐，推荐用于 torch extension。
      2. pip 安装的 pybind11（`python3 -m pybind11 --includes`）—— fallback。

    返回的字符串用空格分隔，便于后续 CMake `string(REPLACE " " ";" ...)` 转为 list。
    """
    import sysconfig
    py_include = sysconfig.get_path("include")

    # 1) torch 自带 pybind11
    try:
        import torch  # type: ignore
        torch_include = Path(torch.__file__).resolve().parent / "include"
        if (torch_include / "pybind11" / "pybind11.h").is_file():
            return f"-I{py_include} -I{torch_include}"
    except ImportError:
        pass

    # 2) fallback: pip 装的 pybind11 module
    try:
        result = subprocess.run(
            ["python3", "-m", "pybind11", "--includes"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True,
        )
        flags = result.stdout.strip()
        if flags:
            return flags
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Cannot locate pybind11 headers. Tried torch-bundled "
        "(<torch>/include/pybind11/) and `python3 -m pybind11 --includes`. "
        "Install pybind11 via `pip install pybind11`, or ensure torch is importable."
    )


def _format_cmake_list(items: list[str], indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _generate_cmakelists(
    kernel_dir: Path,
    build_dir: Path,
    module_name: str,
    sources: list[Path],
    ascend_path: Path,
    pybind11_includes: str,
) -> str:
    include_dirs = [kernel_dir]
    catlass_include = kernel_dir / "catlass" / "include"
    if catlass_include.is_dir():
        include_dirs.append(catlass_include)

    task_catlass_include = kernel_dir.parent / "catlass" / "include"
    if task_catlass_include.is_dir() and task_catlass_include not in include_dirs:
        include_dirs.append(task_catlass_include)

    source_lines = [str(path) for path in sources]
    include_lines = [str(path) for path in include_dirs]

    return f"""cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)

ascendc_include_directories(kernels PRIVATE
{_format_cmake_list(include_lines)}
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
)

add_library(pybind11_lib SHARED "{kernel_dir / 'pybind11.cpp'}")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "{kernel_dir}"
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
# PYBIND11 includes are precomputed by utils/build_ascendc.py at file-generation time.
# Priority: torch-bundled <torch>/include/pybind11/ (ABI-aligned with libtorch), then pip `python3 -m pybind11 --includes`.
set(PYBIND11_INC "{pybind11_includes}")
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)
"""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[build_ascendc] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build(task: str, soc_version: str, build_type: str, clean: bool) -> Path:
    task_dir = _resolve_task_dir(task)
    kernel_dir = task_dir / "kernel"
    pybind_path = kernel_dir / "pybind11.cpp"
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")
    if not pybind_path.is_file():
        raise FileNotFoundError(f"Missing pybind11 entry: {pybind_path}")

    sources = _find_kernel_sources(kernel_dir)
    module_name = _extract_pybind_module_name(pybind_path)
    build_dir = kernel_dir / "build"
    cmake_dir = build_dir / "_autogen_cmake"
    ascend_path = _detect_ascend_path()
    pybind11_includes = _pybind11_includes()
    print(f"[build_ascendc] pybind11 includes: {pybind11_includes}")

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = cmake_dir / "CMakeLists.txt"
    cmakelists_path.write_text(
        _generate_cmakelists(
            kernel_dir=kernel_dir,
            build_dir=build_dir,
            module_name=module_name,
            sources=sources,
            ascend_path=ascend_path,
            pybind11_includes=pybind11_includes,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_path)
    cmake_configure = [
        "cmake",
        "-S",
        str(cmake_dir),
        "-B",
        str(build_dir),
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    cmake_build = ["cmake", "--build", str(build_dir), "-j2"]

    _run(cmake_configure, cwd=task_dir, env=env)
    _run(cmake_build, cwd=task_dir, env=env)
    return build_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AscendC kernels for a task without task-local run.sh")
    parser.add_argument("task", help="Task directory name or path")
    parser.add_argument("-v", "--soc-version", default="Ascend910B2", help="Ascend SoC version")
    parser.add_argument("--build-type", default="Debug", help="CMake build type")
    parser.add_argument("--clean", action="store_true", help="Remove kernel/build before configuring")
    args = parser.parse_args()

    build_dir = build(
        task=args.task,
        soc_version=args.soc_version,
        build_type=args.build_type,
        clean=args.clean,
    )
    print(f"[build_ascendc] Build completed: {build_dir}")


if __name__ == "__main__":
    main()
```

### `run_npu.py`

```python
#!/usr/bin/env python3
"""C014 predeclared local A/B; never an official evaluator."""
import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
SIZES = (256, 2048, 4096)
BLOCKS = 24
CALLS = 64
WARMUP = 64
TIMER_WARMUP = 16
TIME_LIMIT_SECONDS = 120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def bits(t):
    size = t.element_size() * 8
    view_dtype = torch.int16 if size == 16 else torch.int32
    return [int(v) & ((1 << size) - 1) for v in t.contiguous().view(view_dtype).tolist()]


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else "inf" if v > 0 else "-inf"
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, tuple):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    return v


def observe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"argv": argv, "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"argv": argv, "exception": repr(e)}


def event_binding(out):
    cls = torch.npu.Event
    record = {"qualified_name": f"{cls.__module__}.{cls.__qualname__}"}
    for name, obj in (("class", cls), ("init", cls.__init__), ("record", cls.record), ("elapsed_time", cls.elapsed_time)):
        item = {}
        try:
            item["signature"] = str(inspect.signature(obj))
        except Exception as e:
            item["signature_exception"] = repr(e)
        try:
            path = Path(inspect.getfile(obj)).resolve()
            item.update({"source_path": str(path), "source_sha256": sha(path)})
            (out / "installed_event_source.py").write_bytes(path.read_bytes())
        except Exception as e:
            item["source_exception"] = repr(e)
        record[name] = item
    return record


def make_inputs(dtype, n):
    values = [-math.inf, -1000., -104., -100., -90., -88.75, -88.5,
              -20., -17., -16., -15., -12., -11.1, -11., -8., -1.,
              -0., 0., 1., 8., 11., 11.1, 12., 15., 16., 17., 20.,
              88.5, 88.75, 90., 100., 104., 1000., math.inf, math.nan]
    for center in (-88.75, -17., -11.1, -1., 0., 1., 11.1, 17., 88.75):
        c = torch.tensor(center, dtype=dtype)
        values.extend([torch.nextafter(c, torch.tensor(-math.inf, dtype=dtype)).item(),
                       c.item(),
                       torch.nextafter(c, torch.tensor(math.inf, dtype=dtype)).item()])
    boundary = torch.tensor((values * ((n + len(values) - 1) // len(values)))[:n], dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(61018014 + n)
    uniform = (torch.rand(n, generator=generator, dtype=torch.float32) * 16 - 8).to(dtype)
    normal = torch.randn(n, generator=generator, dtype=torch.float32).to(dtype)
    wide = (torch.randn(n, generator=generator, dtype=torch.float32) * 100).to(dtype)
    return {"boundary_nonfinite": boundary, "uniform_finite": uniform,
            "normal_finite": normal, "wide_normal_finite": wide}


def reference_metrics(y, ref):
    same = (y == ref) | (torch.isnan(y) & torch.isnan(ref))
    valid_abs = torch.isfinite(y) & torch.isfinite(ref)
    valid_rel = valid_abs & (ref != 0)
    delta = (y.double() - ref.double()).abs()
    return {"total": y.numel(), "same_value_or_nan": int(same.sum()),
            "finite_abs_denominator": int(valid_abs.sum()),
            "max_abs_error": float(delta[valid_abs].max()) if valid_abs.any() else None,
            "nonzero_finite_rel_denominator": int(valid_rel.sum()),
            "max_rel_error": float((delta[valid_rel] / ref.double()[valid_rel].abs()).max()) if valid_rel.any() else None,
            "same_nan_class": int((torch.isnan(y) & torch.isnan(ref)).sum()),
            "same_posinf_class": int((torch.isposinf(y) & torch.isposinf(ref)).sum()),
            "same_neginf_class": int((torch.isneginf(y) & torch.isneginf(ref)).sum())}


def equivalence(a, b):
    ab, bb = bits(a), bits(b)
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    finite = torch.isfinite(a) & torch.isfinite(b)
    return {"total_denominator": a.numel(), "exact_value_or_nan_count": int(same.sum()),
            "both_finite_denominator": int(finite.sum()), "either_nonfinite_denominator": int((~finite).sum()),
            "both_nan_denominator": int((torch.isnan(a) & torch.isnan(b)).sum()),
            "bitwise_equal_count": sum(x == y for x, y in zip(ab, bb)),
            "signed_zero_bit_mismatch_count": sum(x != y and float(av) == 0 and float(bv) == 0 for x, y, av, bv in zip(ab, bb, a, b)),
            "value_or_nan_mismatch_indices": torch.where(~same)[0].tolist(),
            "bit_mismatch_indices": [i for i, (x, y) in enumerate(zip(ab, bb)) if x != y]}


def measure(fn, x, events, calls):
    start, end = events
    torch.npu.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    last = None
    for _ in range(calls):
        last = fn(x)
    end.record()
    torch.npu.synchronize()
    wall_ns = time.perf_counter_ns() - wall_start
    event_ms = float(start.elapsed_time(end))
    # Keep the last output alive until after synchronization and event query.
    retained_output_numel = None if last is None else last.numel()
    return {"calls": calls, "event_ms": event_ms, "synchronized_wall_ns": wall_ns,
            "valid": math.isfinite(event_ms) and event_ms >= 0 and wall_ns >= 0,
            "event_us_per_call": event_ms * 1000 / calls if calls else None,
            "synchronized_wall_us_per_call": wall_ns / 1000 / calls if calls else None,
            "retained_output_numel": retained_output_numel}


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo) if lo != hi else ordered[lo]


def summarize(rows):
    if any(not row.get("valid", False) for row in rows):
        return {"interpretation": "invalid_timing_observation_no_cost_claim", "invalid_observation_count": sum(not row.get("valid", False) for row in rows)}
    result = {}
    for metric in ("event_us_per_call", "synchronized_wall_us_per_call"):
        pairs = []
        for offset in range(0, len(rows) - 3, 4):
            block = rows[offset:offset + 4]
            if [x["implementation"] for x in block] != ["high", "explicit", "explicit", "high"]:
                raise RuntimeError("ABBA order corrupted")
            a = (block[0][metric] + block[3][metric]) / 2
            b = (block[1][metric] + block[2][metric]) / 2
            pairs.append({"block": block[0]["block"], "high_us": a, "explicit_us": b,
                          "explicit_over_high": b / a if a > 0 else None, "explicit_minus_high_us": b - a})
        if not pairs or any(p["explicit_over_high"] is None for p in pairs):
            result[metric] = {"paired_blocks": len(pairs), "status": "insufficient_valid_timing"}
            continue
        rng = random.Random(61018014)
        ratios = [p["explicit_over_high"] for p in pairs]
        medians = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
        ci = [percentile(medians, .025), percentile(medians, .975)]
        result[metric] = {"paired_blocks": len(pairs), "pairs": pairs,
                          "median_high_us": statistics.median([p["high_us"] for p in pairs]),
                          "median_explicit_us": statistics.median([p["explicit_us"] for p in pairs]),
                          "median_explicit_over_high": statistics.median(ratios),
                          "median_explicit_minus_high_us": statistics.median([p["explicit_minus_high_us"] for p in pairs]),
                          "ratio_median_bootstrap_95pct_ci": ci,
                          "explicit_faster_blocks": sum(r < 1 for r in ratios),
                          "explicit_slower_blocks": sum(r > 1 for r in ratios),
                          "equal_blocks": sum(r == 1 for r in ratios),
                          "direction": "explicit_faster" if ci[1] < 1 else "high_faster" if ci[0] > 1 else "inconclusive_parity_in_interval"}
    directions = [x.get("direction") for x in result.values()]
    result["interpretation"] = "incomplete_protocol_no_cost_claim" if len(rows) != BLOCKS * 4 else "bounded_complete_call_direction_agrees" if len(set(directions)) == 1 and directions[0] in ("explicit_faster", "high_faster") else "no_established_cost_advantage"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(args.device)
    run_start = time.monotonic()
    modules = {}
    loaded = []
    for dtype in ("half", "float"):
        for impl in ("high", "explicit"):
            variant = f"{impl}_{dtype}"
            sys.path.insert(0, str(ROOT / variant / "kernel/build"))
            mod = importlib.import_module(f"_c014_{variant}")
            modules[variant] = mod
            loaded.append({"variant": variant, "module_file": str(Path(mod.__file__).resolve()), "module_sha256": sha(mod.__file__)})
    result = {"schema_version": "c014.bounded_independent_cost_ab.v1", "origin": "new_independent_bounded_local_probe_not_formal_evaluation",
              "logical_device": args.device, "device_name": torch.npu.get_device_name(args.device),
              "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "environment": {k: v for k, v in os.environ.items() if k.startswith(("ASCEND", "LD_LIBRARY_PATH", "PATH"))},
              "start_npu_smi": observe(["npu-smi", "info"]), "loaded_modules": loaded,
              "physical_device_assigned_by_root": 2,
              "device_nodes": sorted(str(p) for p in Path("/dev").glob("davinci*")),
              "start_npu_mapping": observe(["npu-smi", "info", "-m"]),
              "start_assigned_physical_frequency": observe(["npu-smi", "info", "-t", "freq", "-i", "2"]),
              "event_implementation": event_binding(out),
              "source_files": {str(x.relative_to(ROOT)): sha(x) for x in sorted(ROOT.glob("*/kernel/*.cpp"))},
              "runner_sha256": sha(__file__), "predeclared_sha256": sha(ROOT / "PREDECLARED.md"),
              "protocol": {"sizes": SIZES, "blocks": BLOCKS, "calls_per_observation": CALLS, "warmup_calls_per_variant": WARMUP,
                           "timer_warmup_calls": TIMER_WARMUP, "numerical_calls": 48, "maximum_wrapper_calls_total": 37760,
                           "time_limit_seconds": TIME_LIMIT_SECONDS, "order": ["high", "explicit", "explicit", "high"],
                           "event_interval_scope": "stream interval around complete calls; may include launch starvation", "wall_scope": "event recording + complete calls + final synchronize"},
              "reference": "CPU torch.sigmoid(input.double()).to(dtype), diagnostic exact-value-or-NaN contract retained", "numerical": [], "performance": []}
    (out / "run_binding.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    archive = {}
    perf_inputs = []
    for dtype_name, dtype in (("half", torch.float16), ("float", torch.float32)):
        for n in SIZES:
            for name, x in make_inputs(dtype, n).items():
                key = f"{dtype_name}_n{n}_{name}"
                xd = x.to(f"npu:{args.device}")
                high = modules[f"high_{dtype_name}"].activation(xd)
                explicit = modules[f"explicit_{dtype_name}"].activation(xd)
                torch.npu.synchronize()
                high, explicit = high.cpu(), explicit.cpu()
                ref = torch.sigmoid(x.double()).to(dtype)
                eq = equivalence(high, explicit)
                eligible = name == "uniform_finite" and eq["bitwise_equal_count"] == n and eq["both_finite_denominator"] == n
                row = {"case": key, "dtype": dtype_name, "n": n, "distribution": name, "input_sha256": tensor_sha(x),
                       "high_sha256": tensor_sha(high), "explicit_sha256": tensor_sha(explicit), "reference_sha256": tensor_sha(ref),
                       "input_bits": bits(x), "high_bits": bits(high), "explicit_bits": bits(explicit), "reference_bits": bits(ref),
                       "equivalence": eq, "high_vs_reference": reference_metrics(high, ref), "explicit_vs_reference": reference_metrics(explicit, ref),
                       "eligible_for_performance": eligible}
                result["numerical"].append(row)
                with (out / "numerical_rows.jsonl").open("a") as f:
                    f.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
                for suffix, t in (("input", x), ("high", high), ("explicit", explicit), ("reference", ref)):
                    archive[f"{key}_{suffix}"] = t
                if name == "uniform_finite":
                    perf_inputs.append((key, dtype_name, xd, eligible))
    torch.save(archive, out / "inputs_outputs.pt")
    result["tensor_archive_sha256"] = sha(out / "inputs_outputs.pt")
    events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
    timer_key, timer_dtype, timer_x, _ = perf_inputs[-1]
    for _ in range(TIMER_WARMUP):
        timer_last = modules[f"high_{timer_dtype}"].activation(timer_x)
    torch.npu.synchronize()
    timer_empty = measure(modules[f"high_{timer_dtype}"].activation, timer_x, events, 0)
    timer_calls = measure(modules[f"high_{timer_dtype}"].activation, timer_x, events, CALLS)
    timer_valid = (timer_empty["event_ms"] >= 0 and timer_empty["synchronized_wall_ns"] >= 0 and
                   timer_calls["event_ms"] > timer_empty["event_ms"] and timer_calls["synchronized_wall_ns"] > 0)
    result["timer_control"] = {"case": timer_key, "empty": timer_empty, "repeated": timer_calls, "valid": timer_valid,
                               "limitation": "sanity control only; event interval is not isolated kernel timing"}
    for key, dtype_name, xd, eligible in perf_inputs:
        row = {"case": key, "eligible": eligible, "observations": [], "complete": False}
        result["performance"].append(row)
        if not eligible or not timer_valid:
            row["status"] = "numerical_contract_excluded" if not eligible else "timer_control_failed"
            continue
        functions = {impl: modules[f"{impl}_{dtype_name}"].activation for impl in ("high", "explicit")}
        for impl in ("high", "explicit"):
            for _ in range(WARMUP):
                warm_last = functions[impl](xd)
        torch.npu.synchronize()
        for block in range(BLOCKS):
            if time.monotonic() - run_start > TIME_LIMIT_SECONDS:
                row["status"] = "predeclared_time_limit"
                break
            for position, impl in enumerate(("high", "explicit", "explicit", "high")):
                observation = measure(functions[impl], xd, events, CALLS)
                observation.update({"block": block, "position": position, "implementation": impl})
                row["observations"].append(observation)
                with (out / "timing_observations.jsonl").open("a") as f:
                    f.write(json.dumps({"case": key, **observation}, ensure_ascii=False, allow_nan=False) + "\n")
        row["complete"] = len(row["observations"]) == BLOCKS * 4
        row["status"] = "completed" if row["complete"] else row.get("status", "incomplete")
        row["summary"] = summarize(row["observations"])
    torch.npu.synchronize()
    result["end_npu_smi"] = observe(["npu-smi", "info"])
    result["end_npu_mapping"] = observe(["npu-smi", "info", "-m"])
    result["end_assigned_physical_frequency"] = observe(["npu-smi", "info", "-t", "freq", "-i", "2"])
    result["elapsed_wall_seconds"] = time.monotonic() - run_start
    result["claims_excluded"] = ["equal lowering", "opcode or instruction count", "measured UB occupancy", "full operator benefit", "official score or hidden-case benefit"]
    (out / "result.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "numerical_cases": len(result["numerical"]),
                      "performance": [{"case": r["case"], "status": r["status"], "summary": {k: v for k, v in r.get("summary", {}).items() if k == "interpretation"}} for r in result["performance"]],
                      "elapsed_wall_seconds": result["elapsed_wall_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

### `build_round_001/headers/aarch64-linux/asc/impl/adv_api/detail/activation/sigmoid/sigmoid_common_impl.h`

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_common_impl.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/activation/sigmoid/sigmoid_common_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/activation/sigmoid.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_COMMON_IMPL_H__
#endif

#ifndef IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H
#define IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H

#include "../../../../../include/basic_api/kernel_basic_intf.h"
#include "../../../../../include/basic_api/kernel_tensor.h"
#include "../../../../basic_api/kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/activation/sigmoid/sigmoid_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

// V100 does not support counter mode. Thus different implementation is needed.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "sigmoid_impl.h"
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 1001
#include "sigmoid_v100_impl.h"
#elif defined(__NPU_ARCH__) && \
    (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "sigmoid_3510_impl.h"
#endif

namespace AscendC {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 1001)
template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t calCount)
{
    CHECK_FUNC_HIGHLEVEL_API(Sigmoid, (T, isReuseSource), (dstTensor, srcTensor, sharedTmpBuffer, calCount));
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    ASSERT(
        ((TPosition)dstTensor.GetPosition() == TPosition::VECIN ||
         (TPosition)dstTensor.GetPosition() == TPosition::VECOUT ||
         (TPosition)dstTensor.GetPosition() == TPosition::VECCALC));
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(
            KERNEL_ERROR,
            "calCount is %u, which should not "
            "larger than srcTensor size %u",
            calCount, srcTensor.GetSize());
    });

    uint32_t splitSize = sharedTmpBuffer.GetSize() / sizeof(T);
    ASCENDC_ASSERT((splitSize != 0), { KERNEL_LOG(KERNEL_ERROR, "splitSize should not be 0!"); });

    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
    SigmoidCompute<T, isReuseSource>(dstTensor, srcTensor, tmpBuffer, splitSize, loopCount, calcTail);
}
#endif

template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> stackBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SigmoidImpl<T, isReuseSource>(dstTensor, srcTensor, stackBuffer, calCount);
}
} // namespace AscendC
#endif // IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_COMMON_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_COMMON_IMPL_H__
#endif
```

### `build_round_001/headers/aarch64-linux/asc/impl/adv_api/detail/activation/sigmoid/sigmoid_impl.h`

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sigmoid_impl.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/activation/sigmoid/sigmoid_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/activation/sigmoid.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_IMPL_H__
#endif

#ifndef IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H
#define IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H
#include "../../../../../include/basic_api/kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
template <typename T>
__aicore__ inline void SigmoidIntrinsicsImpl(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& stackBuffer)
{
    struct UnaryRepeatParams repeatParams;
    struct BinaryRepeatParams binaryRepeatParams;
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dst, src, static_cast<T>(-1.0), MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(dst, dst, MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(dst, dst, static_cast<T>(1), MASK_PLACEHOLDER, 1, repeatParams);
    Duplicate<T, false>(
        stackBuffer, static_cast<T>(1.0), MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Div<T, false>(dst, stackBuffer, dst, MASK_PLACEHOLDER, 1, binaryRepeatParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidCompute(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& sharedTmpBuffer,
    const uint32_t splitSize, const uint32_t loopCount, const uint32_t calcTail)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
    for (uint32_t i = 0; i < loopCount; ++i) {
        SigmoidIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize], sharedTmpBuffer);
    }
    if (calcTail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        SigmoidIntrinsicsImpl(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize], sharedTmpBuffer);
    }

    SetMaskNorm();
    ResetMask();
}
} //  namespace AscendC
#endif // IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_IMPL_H__
#endif
```

### `build_round_001/headers/aarch64-linux/asc/include/adv_api/activation/sigmoid.h`

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid.h
 * \brief Sigmoid is an activation function,
 * Mathematical formulas: Sigmoid(x) = 1 / (1 + exp(-x))
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_H__
#endif

#ifndef LIB_SIGMOID_SIGMOID_H
#define LIB_SIGMOID_SIGMOID_H

#include "kernel_tensor.h"
#if defined(__NPU_ARCH__) &&                                                                         \
    (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || \
     __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "../../../impl/adv_api/detail/activation/sigmoid/sigmoid_common_impl.h"
#endif
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup Sigmoid
 * \brief compute Sigmoid elementwisely
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             sigmoid_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sigmoid(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t calCount)
{
#if defined(__NPU_ARCH__) &&                                                                         \
    (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || \
     __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
    SigmoidImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
#endif
}

/*!
 * \ingroup Sigmoid
 * \brief compute Sigmoid elementwisely
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sigmoid(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
#if defined(__NPU_ARCH__) &&                                                                         \
    (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || \
     __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
    SigmoidImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
#endif
}

/*!
 * \ingroup Sigmoid
 * \brief compute Sigmoid elementwisely
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             sigmoid_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sigmoid(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Sigmoid<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/*!
 * \ingroup Sigmoid
 * \brief compute Sigmoid elementwisely
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Sigmoid<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

#pragma end_pipe
} // namespace AscendC
#endif // LIB_SIGMOID_SIGMOID_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_SIGMOID_H__
#endif
```

### 已安装计时依赖的完整 Event 类

此类依赖同文件原有的 `ctypes`、`torch_npu`、`torch_npu._C` imports；底层C扩展实现未捕获，未据此声明内核专属计时。

```python
class Event(torch_npu._C._NPUEventBase):
    r"""Wrapper around a NPU event.

    NPU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize NPU
    streams.

    The underlying NPU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(cls, enable_timing=enable_timing, blocking=blocking,
                                         interprocess=interprocess, graph_external=False)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_npu.npu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_npu.npu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def recorded_time(self):
        r"""Returns the record time stamp after the event was recorded.
        """
        return super(Event, self).recorded_time()

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``npuEventSynchronize()``: see
            `NPU Event documentation`_ for more info.
        """
        super(Event, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.npu_event)

    def __repr__(self):
        if self.npu_event:
            return '<torch_npu.npu.Event {0:#x}>'.format(self._as_parameter_.value)
        else:
            return '<torch_npu.npu.Event uninitialized>'

```


## 性能影响

来源平台均为 a2 / Ascend910B4，参照路线均为同 dtype 的高阶显式 scratch Sigmoid。FP32 三个合格 case 的显式/高阶完整调用成本中位比：Event 为约 1.027–1.044，wall 为约 1.027–1.044，即显式相对参照约慢 2.7%–4.4%。这些是同一次有界 run 的成对成本，不是 isolated kernel latency，也不是全算子或跨平台预期收益。

half 三个 case 中，两个未形成两计时一致的显著方向；另一个显式相对高阶约慢 0.6%，两计时的自举区间均略高于 1。不能把微小差异概括成 half 一定相同或一定更快。

## 预防

在替换高阶 API 前，先查看本次构建实际使用的头与 overload；把 scratch 形态、同 dtype 声明资源、输入与 module 绑定作为控制条件。对未出现成本优势的路线保留负结果，在业务算子中另行验证更改数据流、融合或输出分配是否产生收益，不能从基础 API 名字直接推导收益。

## 验证信号

信号来自新建的独立 Sigmoid 成本比较：原始结果保存每元素 input/high/explicit/reference bits 和逐次 ABBA Event/wall 观测，本文数值从这些原始字段复算；导出的 outcome 摘要只作交叉核对。

- 四个独立单入口构建完成，同次加载的四个 module hash 分别对应构建产物；源码 hash 与 runtime 记录逐个匹配。（来源平台：a2 / Ascend910B4；证据等级：artifact_verified）
- 24 行数值控制共 51200/51200 个 A/B bit相同，其中双方 finite 的分母为 50994；没有 A/B signed-zero bit差异。对 CPU fp64→dtype 诊断参考，双方 finite 值只有 37508/50994 exact 相等，另有 206/206 NaN 类别匹配，合计 37714/51200 exact-value-or-NaN；此处不要求参考 NaN payload bit相同。故只证明记录输入上的 A/B 等价，未宣称参考精度满过。（来源平台：a2 / Ascend910B4；证据等级：artifact_verified）
- 6 个具备资格的 finite uniform case 各完成 24 个 ABBA 块；每个 A/B 观测执行 64 个真实 wrapper 调用，完整总预算为 37760 次调用。（来源平台：a2 / Ascend910B4；证据等级：artifact_verified）
- FP32 显式相对参照高阶，Event 中位成本比约 1.027–1.044，同步 wall 约 1.027–1.044；三个 case 的两计时自举区间都高于 1。half 没有显式更快的共同结论，仅一个 case 两计时一致约慢 0.6%。（来源平台：a2 / Ascend910B4；证据等级：artifact_verified）
- installed Event 记录到 stream 并返回两 marker 的 elapsed time；本 driver 的 Event 区间覆盖完整 wrapper 流，可能包含 launch starvation，同步 wall 同时包含 event recording 与 final synchronization。频率查询前后均被工具以 rc215 拒绝，频率 unknown。（来源平台：a2 / Ascend910B4；证据等级：artifact_verified）

## 失效边界

**未决项，优先项：复现并追根因到机制层。** 若要解释 FP32 的具体成本差异，需要独立复现、取得 ISA/最终 lowering 与同步/launch 分解；现有源码和成本观测不能确定是哪个机制导致差异。

本次只是单一来源、一次有界设备 run，ABBA 块不增加跨轨迹复现次数。未获得实时频率，未捕获运行时加载 CANN 共享库；环境变化可能改变方向。Event 不是 isolated kernel 计时，wall 包含 allocation、checks、dispatch、launch 与同步。没有测量 resource occupancy、没有完整业务算子收益、没有正式评测认证，也没有恢复或认证旧优化成功结论。

其他 Ascend 平台、其他 overload/容量/编译路径均未验证；若推断可迁移，仍须重新建立数值资格与同范围成对成本。对更大或不同布局、多设备调用、融合内核、输出复用及其他表达式均无证据，不套用本负结果。

## 坏实践

把 A/B 逐bit一致称为相对 CPU 参考“精度全部通过”；把相同源码算术链称为同 lowering；把相同声明 UB 容量称为同 occupancy；把 Event stream 区间称为内核专属执行时间；只报单个绝对分数或挑出有利 ABBA 块；把该 probe 的负结果推广成禁用高阶或显式 Sigmoid 的普遍规则。

## 误诊

“展开高阶名字便可省掉高阶开销”只是设计假设；已安装实现包含同一负指数算术链，实际比较没有建立该显式尝试的提速优势。源码路径仍有 mask/barrier/auto-sync 差异，因此不能继续误诊成“二者必然生成完全相同代码”。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/c014/public_result.json#/numerical`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/public_result.json#/performance`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/public_result.json#/loaded_modules`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/public_result.json#/compiler_and_headers`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/public_result.json#/public_selection`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/sources/PREDECLARED.md`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/sources/source_manifest.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/sources/build_all.py`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/sources/run_npu.py`；`cannbot_non_level4_split_review_20261019/evidence_public/c014/README.md`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
