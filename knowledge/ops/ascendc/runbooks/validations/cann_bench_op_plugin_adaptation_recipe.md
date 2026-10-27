---
type: "Runbook"
title: cann-bench op_plugin算子适配配方
description: 用 bench 框架的 ACLNN_CMD 宏 + TORCH_LIBRARY(cann_bench) 把自定义 aclnn 算子做成提交包：adapt 脚本生成骨架、op_plugin 一行两段式调用、__init__.py 换完整签名，构建 wheel+run 后 torch.ops schema 注册成功、10 种子精度全过、bench 评测 19/20。
tags: [task.implementation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# cann-bench op_plugin算子适配配方

## 现象（任务场景）

自定义算子（depthwise conv，Ascend950PR，CANN 9.2.0）已完成 registry-invoke 全流程（UT 57/57、ST 53/53、PyTorch ST 43/43），需按 cann-bench 提交包格式适配，使 `run_evaluation.sh --source-dir` 能扫描编译安装并评测。适配动作：执行 `bench-adaptation` 的 `adapt_to_bench.sh` 生成提交包骨架（复制 cann-bench 的 cmake/scripts 构建基础设施 + 拷贝 op_kernel/op_host/op_api 源码 + 生成 `cann_bench/__init__.py` 占位）；随后手工补两块内容——`csrc/ops/<op>/op_plugin/<op>_plugin.cpp`（torch 算子插件）与 `cann_bench/__init__.py` 的真实函数签名；最后 `bash build.sh` 出 wheel + run 双包，pip 安装后 `torch.ops.cann_bench.<op>` 可调用、按社区标准自验精度、再跑 bench 评测。

## 适用场景

- 工程形态：cann-bench 算子提交包适配（`run_evaluation.sh --source-dir` 可接受的格式）；任何要把 `operators/<op_snake>/` 产出转成 bench 提交包的任务。
- API 路径：`csrc/ops/_common/aclnn_common.h` 的 `ACLNN_CMD(aclnn_api, ...)` 宏（自动追加 workspaceSize/executor 地址并完成两段式调用）；`TORCH_LIBRARY_FRAGMENT(cann_bench, m)` + `Meta`/`PrivateUse1` 实现；`adapt_to_bench.sh --operator-dir --op-pascal --op-snake --bench-root --output-dir`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0，torch 2.7.1+cpu + torch_npu 2.7.1.post4（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow 骨架跨算子通用，schema/meta 部分按各算子 proto.yaml 套用。
- 路线优先级：recommended_fastpath（bench 框架已封装好调用与构建，不要手写两段式）。
- 明确不适用 / 未验证场景：非 bench 场景的 torch 接入（用 torch 扩展直连配方，见 [aclnn 算子 PyTorch ST 接入工程五件套配方](../validations/aclnn_pytorch_st_torch_adapter_project_recipe.md)）；算子包未安装到 vendors 的环境（ACLNN_CMD 运行期要找到 libcust_opapi.so）。

## 设计决策

- **执行体用框架的 `ACLNN_CMD` 宏而不是手写 aclnn 两段式**：宏内部自动 ① 调 `aclnn<Op>GetWorkspaceSize(...所有参数..., &workspace_size, &executor)` ② 调 `aclnn<Op>(workspace, workspace_size, executor, stream)`，一行完成；与手写适配层相比代码量 1:10，且与 bench 评测器同一调用路径，减少适配层自身引入偏差的可能。
- **Meta impl 手算输出 shape**（`torch::empty({N, C, hout, wout})`）而不是让 NPU impl 兼任——保持 torch 侧 shape 推导可用（无 NPU 时也能构图）。
- **schema 字符串逐字对齐 proto.yaml**：`depthwise_conv_2d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor y`——bench 用例按 proto 传参，schema 不一致会直接调用失败。
- **`__init__.py` 占位签名必须替换**：模板注释明确给出 Python→C++ 类型映射（Tensor→const torch::Tensor&、int→int64_t、float→double、bool→bool），按 proto.yaml 属性逐项替换占位单参数函数。
- 备选：改 bench 框架让它直接吃 operators/ 目录——不可行也不该做（提交包格式是评测方契约）。

## 系统信息

- 工具链：CANN 9.2.0；cann-bench（V0.4.0，bench 仓含 `examples/aclnn_launch_example` 基础设施与 `tasks/level3/<op>/proto.yaml`）；torch 2.7.1+cpu + torch_npu 2.7.1.post4；pip。
- 目标平台：Ascend950PR_957b（`torch.npu.get_device_name(0)` 读数，build.sh 据此自动选 ascend950）。
- 目录布局（提交包根）：`build.sh`、`CMakeLists.txt`、`setup.py`、`cann_bench/__init__.py`、`csrc/ops/_common/`（框架公共）、`csrc/ops/<op_snake>/{CMakeLists.txt(自动生成), op_kernel/, op_host/, op_api/, op_plugin/<op>_plugin.cpp(手写)}`、`dist/`（wheel + run 产物）。
- 相关文件清单：`adapt_to_bench.sh`（骨架生成，执行成功）、`op_plugin/bench_no_harness_depthwise_conv_2d_plugin.cpp`（手写，本卡完整代码）、`cann_bench/__init__.py`（签名替换，本卡完整代码）、自动生成的 `csrc/ops/<op>/CMakeLists.txt`（HOST_SRCS/API_SRCS/KERNEL_FILE 自动检出正确）、`build.sh`（SoC 自检 → build_run.sh 出 run 包 → build_wheel.sh 出 wheel）。

## 完整代码

op_plugin/bench_no_harness_depthwise_conv_2d_plugin.cpp（完整，verbatim 取自该会话写入；许可头从略，交付时保留原文）：

```cpp
// op_plugin for BenchNoHarnessDepthwiseConv2d
//
// proto.yaml schema:
//   depthwise_conv_2d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize,
//     int[] stride, int[] padding, int[] dilation, int groups) -> Tensor y
//
// aclnn API:
//   aclnnBenchNoHarnessDepthwiseConv2dGetWorkspaceSize(x, weight, bias,
//     kernelSize, stride, padding, dilation, groups, y, workspaceSize, executor)
//   ACLNN_CMD auto-appends workspaceSize_addr and executor_addr.

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../../_common/aclnn_common.h"

namespace cann_bench {

static torch::Tensor depthwise_conv_2d_meta(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    c10::IntArrayRef kernelSize, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups) {
    auto N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t kh = kernelSize[0], kw = kernelSize[1];
    int64_t sh = stride[0], sw = stride[1];
    int64_t ph = padding[0], pw = padding[1];
    int64_t dh = dilation[0], dw = dilation[1];
    int64_t hout = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t wout = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    return torch::empty({N, C, hout, wout}, x.options());
}

static torch::Tensor depthwise_conv_2d_npu(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    c10::IntArrayRef kernelSize, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups) {
    auto x_c = x.contiguous();
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();
    auto N = x_c.size(0), C = x_c.size(1), H = x_c.size(2), W = x_c.size(3);
    int64_t kh = kernelSize[0], kw = kernelSize[1];
    int64_t sh = stride[0], sw = stride[1];
    int64_t ph = padding[0], pw = padding[1];
    int64_t dh = dilation[0], dw = dilation[1];
    int64_t hout = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t wout = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    auto y = torch::empty({N, C, hout, wout}, x_c.options());
    ACLNN_CMD(aclnnBenchNoHarnessDepthwiseConv2d, x_c, weight_c, bias_c,
              kernelSize, stride, padding, dilation, groups, y);
    return y;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("depthwise_conv_2d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize, "
          "int[] stride, int[] padding, int[] dilation, int groups) -> Tensor y");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("depthwise_conv_2d", depthwise_conv_2d_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("depthwise_conv_2d", depthwise_conv_2d_npu);
}

}  // namespace cann_bench
```

cann_bench/__init__.py（替换占位后的完整签名，verbatim）：

```python
def depthwise_conv_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                      kernelSize: list, stride: list, padding: list,
                      dilation: list, groups: int) -> torch.Tensor:
    """depthwise_conv_2d operator."""
    return torch.ops.cann_bench.depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding, dilation, groups)
```

（占位模板原样为单参数 `def bench_no_harness_depthwise_conv_2d(x: torch.Tensor) -> torch.Tensor:`，替换前调用侧无法传参。）

## 预防

适配前检查：`operators/<op>/` 四阶段已完成且算子包已安装（`libcust_opapi.so` 存在、`aclnn<Op>GetWorkspaceSize` 符号可寻址——可用 `nm`/ctypes 预检）；proto.yaml 与 aclnn 头文件签名逐参一致；op_plugin 的 schema 字符串与 proto.yaml 逐字一致。

## 验证信号

来源说明：验证来自该算子 bench 适配 6 步流程（同一会话完成：适配→插件→签名→构建→安装→精度自验），全部为 950 真机实测。

- 适配脚本：`adapt_to_bench.sh` 执行成功，自动生成的 `csrc/ops/<op>/CMakeLists.txt` 正确检出 HOST_SRCS（arch35/tiling.cpp + def.cpp + infershape.cpp）、API_SRCS（aclnn + bench 适配层）与 KERNEL_FILE（arch35.cpp）（来源平台：950；证据等级：tool_observed）。
- 构建：`bash build.sh` 产出 `dist/cann_bench-1.0.0-cp38-abi3-linux_x86_64.whl` 与 `dist/cann_bench_20260718_linux_x86_64.run`（来源平台：950；证据等级：tool_observed）。
- 安装与注册：pip 安装后 `import cann_bench` 成功，`torch.ops.cann_bench.depthwise_conv_2d._schemas` 打印完整 schema——与 proto.yaml 逐字一致，schema 注册成功（来源平台：950；证据等级：tool_observed）。
- 精度自验（MERE/MARE 社区标准，fp16 MobileNet shape）：`MERE=0.000176 < 0.000977 PASS`、`MARE=0.000488 < 0.009766 PASS`；多种子 seed 0~9 `10/10 passed`，MARE 范围 [0.000486, 0.003407]（来源平台：950；证据等级：tool_observed）。
- bench 评测：`run_evaluation.sh --source-dir <pkg> --operator DepthwiseConv2D --device-id 0 --no-perf` 20 用例 19 通过；唯一失败 case5 属 fp64-golden 精度画像问题（诊断见 [fp32算子bit-exact仍败于fp64 golden比对](../precision/fp64_golden_fp32_cancellation_dtype_limit.md)），非适配层缺陷——适配层本身在全量用例上正确调通（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 依赖 cann-bench 仓库与 `examples/aclnn_launch_example` 基础设施存在；bench 版本升级后 cmake/scripts 布局变化需重新适配（未跨版本验证）。
- ACLNN_CMD 只封装两段式调用；算子若需要 pre/post 处理（cast、reshape）仍要在 npu impl 里自行处理（本例仅 contiguous()）。
- 提交包目录为临时产物路径（适配输出目录），未做 git 提交/推送。

**未决项**：

- bench 评测 case5 的算子侧精度提升闭环未完成（跨会话接续，见 fp64-golden 诊断卡遗留问题）。
- run 包（`.run`）的独立安装路径未实测（本会话只走了 wheel + 已安装的算子包）。
## 坏实践

- 保留 `__init__.py` 的单参数占位签名不替换——`torch.ops` 已注册但 Python 包装层无法传参，调用侧报参数不匹配。
- op_plugin 手写 GetWorkspaceSize/执行两段式重复造轮子——框架宏已封装，手写只增加引入偏差的面。

## 误诊

- 无（适配一次通过；精度自验首版判据误用绝对误差的问题单独成卡，见 [绝对max_diff误比相对阈值致fp16假FAIL](../precision/absolute_diff_misused_as_relative_threshold.md)）。
