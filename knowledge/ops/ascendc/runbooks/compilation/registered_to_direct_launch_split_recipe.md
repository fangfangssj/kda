---
type: "Runbook"
title: "注册工程转直调split构建与eager启动配方"
description: "Ascend950PR 上注册算子工程转 <<<>>> kernel 直调的完整配方：kernel.h 忠实拷贝加 __NPU_HOST__ guard，设备 TU 用框架纯设备模式 dav-c310-vec 编译、host TU 用 -xasc --cce-host-only --cce-aicore-bin 嵌入设备二进制，链接 .so 后按 (blockDim, dynUbuf, stream, args...) 调 eager stub；最小 kernel 上板写读验证通过。"
tags: ["task.migration", "task.implementation", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# 注册工程转直调split构建与eager启动配方

## 现象（任务形态）

需要把一个注册算子工程（op_graph/op_host/op_kernel，经 vendor 注册运行）忠实转换为 `<<<>>>` kernel 直调工程，用于绕开注册框架做性能对比采集。转换对象 FusedRmsNormSigmoidMul（arch35，含 NdDma、TPipe、workspace、tiling），目标平台 Ascend950PR / CANN 9.2.0。单 TU `ccec -xasc` 编译因 `nd_dma_dci` target feature 报错不可行（见 [nd_dma_dci_single_tu_target_feature_split_build](nd_dma_dci_single_tu_target_feature_split_build.md)），因此整套工程按 **split 构建**组织，并用最小 kernel 先把"编译—链接—上板启动—写读验证"机制跑通，再接入真实 kernel。

## 适用场景

- **工程形态**：注册算子工程 → `<<<>>>` kernel 直调（性能调优 baseline、benchmark、复现实验）；不适用于仍在注册接口上交付的场景。
- **API 路径**：ccec（`-xcce --cce-aicore-arch=dav-c310-vec` 设备模式、`-xasc --cce-host-only --cce-aicore-bin` host 模式、`ccec -shared`）；ACL runtime（`aclInit/aclrtSetDevice/aclrtCreateStream/aclrtMalloc/aclrtSynchronizeStream/aclrtMemcpy`）；`PlatformAscendCManager`（host 侧 tiling）。
- **验证过的平台与 CANN 版本**：Ascend950PR（arch35/dav-3510），CANN 9.2.0。与 frontmatter `platforms` 一致。
- **适用粒度**：`platform_wide`（950 系；其他 SoC 需替换子目标名，未验证）。
- **路线优先级**：`recommended_fastpath`（含 NdDma/`__DAV_VEC__` 的 kernel 直调构建唯一已验证路线；无这些依赖的简单 kernel 可用单 TU `-xasc`）。
- **明确不适用 / 未验证场景**：torch_npu 在本机不支持 Ascend950PR（`Unsupported soc version`），torch 系 warmup/参考进程不可用；含 TPipe 的 kernel 需叠加 AIVEC magic 补丁才能上板（见失效边界）。

## 系统信息

- 工具链：CANN 9.2.0（`ASCEND_TOOLKIT_HOME`），`bin/ccec`（= bisheng 15.0.5），`tools/bisheng_compiler/bin/llvm-nm/llvm-objdump/llvm-objcopy`。
- 设备：Ascend950PR（NPU 0），驱动 25.7.rc1.6。
- 目录布局（本配方实际产物）：

```
direct_baseline/
├── build.sh                                    # split 构建入口（5 步）
├── csrc/op_kernel/
│   ├── fused_rms_norm_sigmoid_mul_device.cpp   # 设备 TU（仅 include 共享 entry 头）
│   ├── fused_rms_norm_sigmoid_mul_kernel.cpp   # host TU（tiling/ACL/launch stub）
│   ├── fused_rms_norm_sigmoid_mul_launch.h     # 对外 C ABI（FRNSM_GM_ADDR 条件类型）
│   ├── fused_rms_norm_sigmoid_mul_local_deps.h # #include "kernel_operator.h" 垫片
│   └── kernel/arch35/                          # 注册工程忠实拷贝：
│       ├── fused_rms_norm_sigmoid_mul_kernel.h       #   + 两行 __NPU_HOST__ guard（唯一改动）
│       ├── fused_rms_norm_sigmoid_mul_tiling_struct.h #   零修改
│       └── fused_rms_norm_sigmoid_mul_entry.h        # 共享入口（两 TU 分别编译）
└── lib/  # device.o / host.o / libfrnsm.so / chain_reference / kernel_symbols.json
```

- 相关文件清单与事件流实际读写一一对应：注册工程 `op_kernel/arch35/*.h`、`op_kernel/fused_rms_norm_sigmoid_mul_apt.cpp`、`op_host/arch35/*_tiling_arch35.cpp`（忠实转换来源）；探针 `kfc-test/*.cpp`（机制验证）。

## 完整代码

### 1. 最小验证 kernel + split 构建 + 上板启动（机制验证单元，verbatim）

```cpp
// kfc_min.cpp —— 无 TPipe 的最小直调 kernel（写常量 42）
#include "kernel_operator.h"
using namespace AscendC;
__global__ __aicore__ void kfc_min(GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> o;
    o.SetGlobalBuffer((__gm__ uint64_t*)out);
    o.SetValue(0, 42);
}
```

```bash
ASC=$CANN_HOME/x86_64-linux
# 框架设备编译 include 集（与 CANN op 编译器 compile_op.py 生成一致）
INC="-I$ASC/asc/impl/adv_api -I$ASC/asc/impl/basic_api -I$ASC/asc/impl/c_api \
 -I$ASC/asc/impl/basic_api/reg_compute -I$ASC/asc/impl/simt_api -I$ASC/asc/impl/utils \
 -I$ASC/asc -I$ASC/asc/include -I$ASC/asc/include/adv_api -I$ASC/asc/include/basic_api \
 -I$ASC/asc/include/aicpu_api -I$ASC/asc/include/c_api -I$ASC/asc/include/basic_api/reg_compute \
 -I$ASC/asc/include/simt_api -I$ASC/asc/include/utils -I$ASC/include -I$ASC/include/ascendc \
 -I$ASC/ascendc/act -I$ASC/tikcpp -I$ASC/tikcpp/tikcfw -I$ASC/tikcpp/tikcfw/impl \
 -I$ASC/tikcpp/tikcfw/interface"
# [设备 TU]
ccec -c -O3 -xcce kfc_min.cpp --cce-aicore-arch=dav-c310-vec -D__DAV_C310__ \
     --cce-aicore-only -std=c++17 $INC -o kfc_min_dev.o
# [host TU]
ccec -xasc --cce-host-only --cce-aicore-bin=kfc_min_dev.o --npu-arch=dav-3510 -fPIC \
     -I$ASC/asc/include -I$ASC/ascendc/include -I$ASC/include -c kfc_min.cpp -o kfc_min_host.o
# [链接 .so]
ccec -shared -fPIC kfc_min_host.o -L$ASC/lib64 -lascendcl -o libkfc_min.so
```

```cpp
// test_min_main.cpp —— g++ 上板启动器（eager stub 的 C ABI：blockDim, shm, stream, args...）
#include <cstdio>
#include <cstdint>
#include "acl/acl_rt.h"
extern "C" void _Z7kfc_minPh(unsigned blockDim, unsigned shm, void* stream, void* out);
#define CHECK(x) do { auto e=(x); if(e){printf("ERR %s=%d\n",#x,(int)e);return 1;} } while(0)
int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream s; CHECK(aclrtCreateStream(&s));
    uint64_t* out; CHECK(aclrtMalloc((void**)&out, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    _Z7kfc_minPh(1, 0, s, out);
    printf("launched\n");
    CHECK(aclrtSynchronizeStream(s));
    uint64_t h = 0;
    CHECK(aclrtMemcpy(&h, 8, out, 8, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("GetUserWorkspace = 0x%lx\n", h);   // 打印标签沿用模板；实测输出 0x2a = 42
    return 0;
}
```

```bash
g++ -O2 test_min_main.cpp -I$ASC/include -L$ASC/lib64 -L. -lkfc_min -lascendcl \
    -Wl,-rpath,$ASC/lib64 -Wl,-rpath,. -o test_min && ./test_min
# launched / ... = 0x2a
```

### 2. 真实工程 build.sh（终版，由一次完整 write 与两次链接修复 edit 组装；[3/5] 步骤的 AIVEC magic 修正见伴生卡）

```bash
#!/bin/bash
# direct_baseline/build.sh —— 注册工程 → `<<<>>>` 直调工程的编译入口
# Split 构建：
#   [1] 设备 TU：ccec 框架纯设备模式（--cce-aicore-arch=dav-c310-vec
#       -D__DAV_C310__ --cce-aicore-only，与原注册工程 kernel 编译一致）
#   [2] host TU：ccec -xasc --cce-host-only --cce-aicore-bin=<设备.o>
#   [3] 链接 libfrnsm.so（ccec -shared）+ AIVEC 加载 magic 修正（见伴生卡）
#   [4] g++ 编译非融合链参考 chain_reference（aclnn level2 内置算子）
#   [5] 提取目标 kernel 精确 mangled 符号 → lib/kernel_symbols.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOC_VERSION="ascend950"
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*) SOC_VERSION="${1#*=}"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "${SOC_VERSION}" != "ascend950" ]]; then
    echo "[ERROR] 本直调工程仅支持 ascend950（arch35/dav-3510），当前: ${SOC_VERSION}"
    exit 1
fi

CANN_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}"
if [[ -z "${CANN_HOME}" ]]; then
    echo "[ERROR] ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH 未设置（继承当前 CANN 环境）"
    exit 1
fi
CCEC="${CANN_HOME}/bin/ccec"
ASC="${CANN_HOME}/x86_64-linux"
BISHENG_NPU_ARCH="dav-3510"   # ascend950 → arch35（与原工程一致）
FW_AICORE_ARCH="dav-c310-vec" # 框架设备编译的 AIV 子目标（NdDmaDci/Reg SIMD 需要）

BUILD_DIR="${SCRIPT_DIR}/lib"
mkdir -p "${BUILD_DIR}"

DEVICE_SRC="${SCRIPT_DIR}/csrc/op_kernel/fused_rms_norm_sigmoid_mul_device.cpp"
DEVICE_OBJ="${BUILD_DIR}/fused_rms_norm_sigmoid_mul_device.o"
HOST_SRC="${SCRIPT_DIR}/csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp"
HOST_OBJ="${BUILD_DIR}/fused_rms_norm_sigmoid_mul_host.o"
KERNEL_SO="${BUILD_DIR}/libfrnsm.so"

CHAIN_SRC="${SCRIPT_DIR}/csrc/chain_reference/chain_reference.cpp"
CHAIN_BIN="${BUILD_DIR}/chain_reference"

# 框架设备编译 include 集（与 CANN op 编译器 compile_op.py 生成的完全一致）
FW_INCLUDES=(
    -I"${ASC}/asc/impl/adv_api"
    -I"${ASC}/asc/impl/basic_api"
    -I"${ASC}/asc/impl/c_api"
    -I"${ASC}/asc/impl/basic_api/reg_compute"
    -I"${ASC}/asc/impl/simt_api"
    -I"${ASC}/asc/impl/utils"
    -I"${ASC}/asc"
    -I"${ASC}/asc/include"
    -I"${ASC}/asc/include/adv_api"
    -I"${ASC}/asc/include/basic_api"
    -I"${ASC}/asc/include/aicpu_api"
    -I"${ASC}/asc/include/c_api"
    -I"${ASC}/asc/include/basic_api/reg_compute"
    -I"${ASC}/asc/include/simt_api"
    -I"${ASC}/asc/include/utils"
    -I"${ASC}/include"
    -I"${ASC}/include/ascendc"
    -I"${ASC}/ascendc/act"
    -I"${ASC}/tikcpp"
    -I"${ASC}/tikcpp/tikcfw"
    -I"${ASC}/tikcpp/tikcfw/impl"
    -I"${ASC}/tikcpp/tikcfw/interface"
)

HOST_INCLUDES=(
    -I"${SCRIPT_DIR}/csrc/op_kernel"
    -I"${SCRIPT_DIR}/csrc/op_kernel/kernel"
    -I"${ASC}/asc/include"
    -I"${ASC}/ascendc/include"
    -I"${ASC}/include"
)

echo "[1/5] ccec 框架纯设备模式编译设备 TU（dav-c310-vec）..."
"${CCEC}" -c -O3 -xcce "${DEVICE_SRC}" \
    --cce-aicore-arch="${FW_AICORE_ARCH}" -D__DAV_C310__ --cce-aicore-only -std=c++17 \
    -I"${SCRIPT_DIR}/csrc/op_kernel" \
    "${FW_INCLUDES[@]}" \
    -o "${DEVICE_OBJ}"

echo "[2/5] ccec -xasc --cce-host-only 编译 host TU（嵌入设备二进制）..."
"${CCEC}" -xasc --cce-host-only --cce-aicore-bin="${DEVICE_OBJ}" \
    --npu-arch="${BISHENG_NPU_ARCH}" -fPIC -O2 -std=c++17 \
    "${HOST_INCLUDES[@]}" \
    -c "${HOST_SRC}" -o "${HOST_OBJ}"

echo "[3/5] 链接 libfrnsm.so + AIVEC 加载 magic 修正 ..."
# ccec 链接后必须做 AIVEC magic 常量替换（含 TPipe 的 kernel 不补则同步 507015），
# 补丁脚本逐字见 tpipe_eager_launch_507015_aivec_magic_patch 卡。
"${CCEC}" -shared -fPIC "${HOST_OBJ}" \
    -L"${ASC}/lib64" -lascendcl -lnnopbase -lopapi -lplatform "${ASC}/lib64/libtiling_api.a" -o "${KERNEL_SO}"
python3 - "${KERNEL_SO}" <<'PY'
# （AIVEC magic 替换脚本，verbatim 见伴生卡：检索 LE 字节 45 42 55 43 → 46 52 41 41，
#  count==0 即构建失败）
PY

echo "[4/5] g++ 编译非融合链参考 chain_reference ..."
g++ -O2 -std=c++17 \
    -I"${ASC}/include" \
    "${CHAIN_SRC}" \
    -L"${ASC}/lib64" -lascendcl -lnnopbase -lopapi \
    -Wl,-rpath,"${ASC}/lib64" \
    -o "${CHAIN_BIN}"

echo "[5/5] 提取目标 kernel 精确 mangled 符号 → kernel_symbols.json ..."
NM="${CANN_HOME}/tools/bisheng_compiler/bin/llvm-nm"
[[ -x "${NM}" ]] || NM=nm
python3 - "${NM}" "${KERNEL_SO}" "${BUILD_DIR}/kernel_symbols.json" <<'PY'
import json, subprocess, sys
nm, so, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = subprocess.run([nm, "--defined-only", so],
                     capture_output=True, text=True, check=True).stdout
mangled = {}
for line in raw.splitlines():
    parts = line.split()
    if len(parts) == 3 and parts[1] in ("T", "W"):
        mangled[parts[2]] = parts[1]
targets = [m for m in mangled if m.startswith("_Z") and "fused_rms_norm_sigmoid_mul" in m]
if len(targets) != 1:
    print(f"[ERROR] expected exactly 1 fused kernel symbol, got {targets}", file=sys.stderr)
    sys.exit(1)
data = {
    "target_kernel_sequence": [targets[0]],
    "source_symbol": targets[0],
    "note": ("direct <<<>>> launch kernel appears in op_summary under its "
             "Itanium mangled symbol; extracted from libfrnsm.so at build time"),
}
with open(out, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"    target kernel symbol: {targets[0]}")
PY

echo "=== build complete ==="
ls -la "${BUILD_DIR}"
```

### 3. 忠实转换的入口骨架（两 TU 共享，节选自 entry 头注释与实现）

```cpp
// 设备 TU（ccec -xcce --cce-aicore-arch=dav-c310-vec 框架纯设备模式）：
//   __NPU_HOST__ 未定义 → 完整 kernel 体编译进设备二进制
// host TU（ccec -xasc --cce-host-only --cce-aicore-bin=<设备.o>）：
//   __NPU_HOST__ 已定义 → 体被跳过，只生成 launch stub / 注册胶水
// 与原入口的对应关系（仅框架胶水差异，算法零修改）：
//   GET_TILING_DATA_WITH_STRUCT(...)  → by-value 入参 tilingData
//   TILING_KEY_IS(1)/ASCENDC_TPL 模板分发 → tilingKey==1 ⇔ rGroupCnt>=2 运行时等价分发
//   REGISTER_NONE_TILING             → 删除（框架注册胶水，直调模式不存在）
// 直调 launch 环境差异（非算法改动）：框架 launch 会在 kernel 入口前设置 KFC
// workspace 寄存器；`<<<>>>` eager launch 不做（Ascend950PR 实测寄存器读数为
// 0，见 eager_direct_launch_getuserworkspace_constant 卡）。入口第一行调用
// AscendC::SetSysWorkspaceForce(workspace) 补齐，workspace 由 host 按
// 16MiB RESERVED_WORKSPACE(dav_3510) + workspaceSize 分配。

__global__ __aicore__ void fused_rms_norm_sigmoid_mul(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR other, GM_ADDR y, GM_ADDR workspace,
    FusedRMSNormSigmoidMulTilingData tilingData)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tilingData.axisShape[0] <= 0 || tilingData.rTotal <= 0) return;
    AscendC::SetSysWorkspaceForce(workspace);
    AscendC::TPipe pipe;
    if (tilingData.rGroupCnt >= 2) {
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        FusedRMSNormSigmoidMul::GroupKernel kernel;
        kernel.Init(x, gamma, other, y, userWorkspace, &tilingData, &pipe);
        kernel.Process();
    }
    /* ... base 路径与框架分发同构，此处节选 ... */
#endif
}
```

（host TU 侧 launcher 的最终形态见 [direct_launch_config_static_cast_parse_error](direct_launch_config_static_cast_parse_error.md) 卡 after 代码；kernel.h 的唯一改动为文件首尾各加 `#if !defined(__NPU_HOST__)` / `#endif` 两行 guard。）

## 设计决策

1. **为什么 split 而不是单 TU**：单 TU `-xasc` 的 device pass 按 dav-3510 MIX 整核编译，本算子 CopyLogical/CopyOther 用到的 NdDmaDci（5D ND-DMA）仅在 dav-c310-vec 子目标可用，MIX/cube 均报 target feature 错；`asc_vf_call`/Reg SIMD 路径也依赖 `__DAV_VEC__`。设备侧因此沿用与注册工程完全相同的框架设备编译（同 arch、同 -O3、同 `__DAV_C310__`），host 侧只补 eager 胶水。
2. **Base vs Group 路径**：Group 路径（kernel 内 `GetUserWorkspace`）受 eager launch 不设 KFC workspace 寄存器限制（实测读数恒 0x1000000，见 [eager_direct_launch_getuserworkspace_constant](eager_direct_launch_getuserworkspace_constant.md)）；Base 路径（workspace 显式 GM_ADDR 入参）完全可用 → 主路径 Base，Group 配 `SetSysWorkspaceForce` + host `16MiB+workspaceSize` 分配缓解。
3. **kernel.h 忠实拷贝原则**：注册工程 kernel.h 整文件搬运，唯一允许的包裹性改动是 `#if !defined(__NPU_HOST__)` 两行 guard（diff 仅 4 行），算法零修改；tiling 头零修改；host tiling 的 `gert::TilingContext` 接口换成 `PlatformAscendCManager` + 本地参数，AICoreSpec 字符串换成 arch35 硬件契约 fallback 常量。
4. **warmup 策略**：torch_npu 不支持 Ascend950PR（`Unsupported soc version: Ascend950PR 9579`），torch 系 warmup 不可用 → host TU 内直接用 raw aclnn（`aclnnArgMax` int64 (1024,1024) / `aclnnMaxV2` fp16 scalar，走 GetWorkspaceSize→executor→执行三段式）做 eager warmup。
5. **msprof 对账**：直调 kernel 在 op_summary 里以 Itanium mangled 名出现（探针实测 `_Z10kfc_probe2Ph`，Task Type AI_VECTOR_CORE，Name 列为 None）→ 构建期用 llvm-nm 从 .so 提取目标 kernel 唯一 mangled 符号（`_Z26fused_rms_norm_sigmoid_mulPhS_S_S_S_32FusedRMSNormSigmoidMulTilingData`）写入 kernel_symbols.json，采集侧按符号匹配。
6. **链接依赖按报错补齐并固化进 build.sh**：`-lnnopbase`（否则 `libnnopbase.so: error adding symbols: DSO missing from command line`）；`-lplatform` + 静态 `libtiling_api.a`（否则 dlopen 报 `undefined symbol: PlatformAscendCManager15platformInitMtxE`，该符号定义在 libtiling_api.a）。
7. **manifest 由 Python 侧统一写入**（sequence 名集中管理），C++ 二进制不写 manifest，避免格式漂移。

## 验证信号

（来自同一次注册转直调会话，Ascend950PR / CANN 9.2.0 实测）

- 最小 kernel `kfc_min` 经 split 构建 + eager stub 启动：`launched` 后设备写回值 42（0x2a）被正确读出——机制端到端成立（来源平台：950；证据等级：tool_observed）。
- 真实工程 build.sh 5 步全部完成：设备 TU、host TU、libfrnsm.so、chain_reference、kernel_symbols.json（`target kernel symbol: _Z26fused_rms_norm_sigmoid_mulPhS_S_S_S_32FusedRMSNormSigmoidMulTilingData`）（来源平台：950；证据等级：tool_observed）。
- host TU 产物含 eager 胶水符号（`__cce_rtKernelEagerLaunch`/`__cce_rtLinkedDevBinaryEagerRegister`/`BishengDevBinaryEagerRegister`）（来源平台：950；证据等级：tool_observed）。
- 链接修复前后对照：缺 `-lnnopbase`/tiling 库的报错原文如上，补齐后 dlopen 与构建通过（来源平台：950；证据等级：tool_observed）。
- **未通过项（如实记录）**：真实 FusedRmsNormSigmoidMul 直调 kernel smoke 在 `stream sync failed (device fault?)` 失败——TPipe 相关 507015（来源平台：950；证据等级：tool_observed）。

## 性能影响

本片只完成直调基线工程建设，未采集直调性能数据；融合 kernel 相对非融合链的性能对比与优化在后续阶段进行（无成对相对数据，不虚构）。

## 失效边界

- 含 TPipe 的 kernel：split 构建可编译、可链接，但 eager 启动必须叠加 AIVEC magic 后链接补丁才能同步通过——根因与补丁脚本见 [tpipe_eager_launch_507015_aivec_magic_patch](tpipe_eager_launch_507015_aivec_magic_patch.md)（该问题已在本轨迹后续阶段闭环：补丁后 base/group 两路径 smoke 与 18/18 功能验收全部通过）。
- 平台绑定：`dav-c310-vec`/`dav-3510`/16MiB 保留区均为 950 系（arch35）口径；其他 SoC 需换子目标名与保留区档位（未验证，推断可迁移）。
- torch_npu 依赖的参考实现/warmup 在 950PR 不可用；需要 torch 参考的用例要走 numpy golden 或 raw aclnn 链。
- `SetSysWorkspaceForce` 缓解未独立验证，Group 路径语义恢复与否待确认。

**未决项**：

- 直调 kernel 的性能采集（msprof 按符号对账）与相对基线的优化闭环——本轨迹后续阶段已完成（见 performance 区合并 DMA / VF 收拢 / host tiling 重平衡三卡）。
- Group 路径 `SetSysWorkspaceForce` 缓解的独立验证。
## 坏实践

- 从单 TU `-xasc` 起步搭含 NdDma 的直调工程（必踩 target feature 错，浪费排查轮次）。
- host 侧 launcher 内联 `static_cast` 进 `<<<>>>` 配置（解析错位，见 [direct_launch_config_static_cast_parse_error](direct_launch_config_static_cast_parse_error.md)）。
- 用 `void*`→`GM_ADDR` 的 reinterpret_cast/C 风格强转（见 [gm_addr_cast_from_void_uses_toolchain_cast](gm_addr_cast_from_void_uses_toolchain_cast.md)）。

## 预防

- 转直调先跑最小 kernel（本卡第 1 节）把机制验证通，再接入真实 kernel——真实 kernel 失败时才有干净的二分基线。
