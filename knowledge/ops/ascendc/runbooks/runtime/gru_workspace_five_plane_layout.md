---
type: "Runbook"
title: GRU workspace 五平面布局：平面 4 为 hn+隐藏偏置
description: PyTorch v2.7.1 RNN.cu 一手核验：融合 GRU 前向把 5 个中间平面按行内 5H 写入 workspace（rg/ig/ng/hx/hn+b2n），平面 4 是加过隐藏偏置的 hidden 门预激活——反向 grg 链消费它而非 ng，实现 storage 语义时照此契约。
tags: [task.design, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GRU workspace 五平面布局：平面 4 为 hn+隐藏偏置


## 现象

在为 ThnnFusedGruCellGrad 设计 CANN 侧 `storage` 输入（torch 侧叫 `workspace`，因与 aclnn 保留名冲突改名）的语义时，必须回答一个具体问题：这个前向传给反向的缓存张量**到底存了什么、什么布局**——反向算子的实现完全建立在这份契约上，存错/读错一个平面就是系统性梯度错误。

答案从 PyTorch v2.7.1 官方源码（GitHub raw `aten/src/ATen/native/cuda/RNN.cu`，tag v2.7.1 逐字抓取）核验得到，并与本机 torch 2.7.1 头文件的 schema 相互印证：

- 前向 host 侧：`GRU_WORKSPACE_MULTIPLIER = 5`，`workspace = at::empty({hx.size(0), hx.size(1) * 5})`——workspace 形状是 `(B, 5H)`。
- 前向 device kernel `gru_cell_forward` 把 5 个平面按行内偏移 `offset = (linearIndex/hsz)*5*hsz + linearIndex%hsz`、`offset + k*hsz (k=0..4)` 写入：**[rg, ig, ng, hx, hn+b2n]**。注意平面 4 写的不是 ng（tanh 后的新门输出）也不是裸 hn（hidden 侧门预激活），而是 `F2H(H2F(hn) + H2F(b2n))`——**加过隐藏侧偏置的 hidden 门预激活**。
- 反向 device kernel `gru_cell_backward` 读同一布局的 5 个平面，直接计算五条梯度链；其中 `grg = gin*H2F(hn)*(1-H2F(rg))*H2F(rg)` 里的 `hn` 就是平面 4 的值（含偏置）。
- 反向 host 侧 `checkGRUBackwardSizes` 强校验 workspace 形状 `{grad_hy.size(0), grad_hy.size(1) * 5}`；`hidden_size = workspace.size(1) / 5` 由 workspace 反推；bias 梯度仅在 `has_bias` 时对 gate 梯度做 `sum(0)`（否则两个 bias 梯度返回未定义张量）。

autograd 侧挂接（官方 `derivatives.yaml` v2.7.1 逐字核验）：`_thnn_fused_gru_cell_backward(grad, result1, input_bias.defined())`——反向的第二个入参就是前向的第二个输出（workspace），`has_bias` 由 `input_bias.defined()` 推导，不是用户显式传的。

## 适用场景

- **工程形态**：算子仓开发（aclnn/AscendC）需求/规格期确定 `storage`/`workspace` 类缓存输入的语义契约；实现期核对平面布局与梯度链。
- **API 路径**：aten `_thnn_fused_gru_cell(_backward)`、PyTorch CUDA 参考实现 `RNN.cu`、CANN 侧同名算子的 storage 输入设计。
- **验证过的平台与 CANN 版本**：参考实现语义取自 PyTorch v2.7.1 官方源码（GitHub raw 抓取逐字核验）+ 本机 torch 2.7.1 头文件（在 950PR 本机执行核验；与 frontmatter `platforms` 一致）。**布局是框架级契约，与芯片平台无关；NPU 实现须按此契约对齐**。
- **适用粒度（transfer_scope）**：`exact_task`→`operator_family` 之间取窄：本布局契约对 `_thnn_fused_gru_cell` 族精确成立，方法（"缓存张量契约从参考实现 device kernel 逐字核验"）可推广。
- **路线优先级**：`recommended_fastpath`（实现前读参考 kernel，不凭公式想象布局）。
- **明确不适用/未验证场景**：其他 torch 版本的布局未核验（v2.7.1 以外的版本须重抓源码）；LSTM fused cell 的 workspace 是 4 平面（`checkLSTMBackwardSizes` 里 `exp_size[0]*exp_size[1]*4`），不要把 GRU 的 5 平面类比过去。

## 判别方法

三源交叉核验，任何单一来源都不够：

1. **本地 schema**（接口形态与 dispatch 域）：
   ```bash
   cat .../torch/include/ATen/ops/_thnn_fused_gru_cell_backward.h          # schema 注释 + 5 返回值
   cat .../torch/include/ATen/ops/_thnn_fused_gru_cell_backward_native.h   # 仅 _cuda 后缀 → CUDA-only
   ```
2. **官方源码 device kernel**（布局与梯度链真值）：抓 `https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/aten/src/ATen/native/cuda/RNN.cu`，读 `kernel::gru_cell_forward` 的 `//SAVE FOR BACKWARDS` 段与 `kernel::gru_cell_backward` 的读平面段。
3. **autograd 挂接**（谁传 workspace、has_bias 从哪来）：抓 `tools/autograd/derivatives.yaml` v2.7.1，grep `_thnn_fused_gru_cell` 挂接公式。

## 根因

融合 GRU cell 的反向需要 5 个前向中间量（rg、ig、ng、hx、hn+b2n）才能不重算 sigmoid/tanh 直接出梯度；torch 参考实现选择**前向全量缓存**（5H 行内拼接）而非反向重算，且把平面 4 存成 `hn+b2n`（加过隐藏偏置）——这样反向的 grg 链免一次加法，has_bias 分支也统一。这份布局是前向/反向两个算子之间的**私有契约**：反向按 `offset+k*hsz, k=0..4` 读取，任何一侧改动布局（例如存 ng、漏掉偏置、平面顺序变化）另一侧即静默产出系统性错误梯度。

具体后果算式（平面 4 存错时的偏差结构）：

- 若误存 `ng`（tanh 后）而非 `hn+b2n`：`grg = gin·hn·(1-rg)·rg` 变成 `gin·ng·(1-rg)·rg`——ng 与 hn 量级不同（ng∈(-1,1)，hn 无界），且 gin 自身依赖 ng 的位置完全错位，全 case 错。
- 若误存裸 `hn`（漏 b2n）：has_bias=true 时 `grg` 系统性缺失 `gin·b2n·(1-rg)·rg` 项——偏差随偏置幅值线性增长，无偏置时才碰巧相等。

## 设计决策

- **为什么 5 平面行内拼接而不是 5 个独立张量**：一次 CopyIn 整行进片上缓存，平面切片退化为偏移寻址（零拷贝）；`(B,5H)` 单张量也让接口保持"一个 storage 输入"。
- **为什么平面 4 存 `hn+b2n` 而非裸 hn**：反向 grg 需要的正是加偏置后的预激活；提前合并让反向免一次加法，并让 has_bias=true/false 在该平面上语义统一（false 时 b2n=0，`F2H(0.0)` 常量路径）。
- **备选与放弃**：反向重算中间量（不缓存）被 torch 参考实现放弃——需要重跑 sigmoid/tanh 与两次投影，融合收益归零；本会话将其登记为"存储换计算"的既定取舍。
- **当前方案档位**：`recommended_fastpath`（照参考实现契约实现，不自创布局）。

## 系统信息

- 参考真值源：PyTorch v2.7.1 `aten/src/ATen/native/cuda/RNN.cu`（GitHub raw 逐字抓取）、`tools/autograd/derivatives.yaml` v2.7.1、本机 torch 2.7.1+cpu 头文件（`/opt/buildtools/python-3.12.9/lib/python3.12/site-packages/torch/include/ATen/ops/`）。
- 核验环境：Ascend 950PR + CANN 9.0.0-beta.2 本机（核验命令执行处）；布局契约本身为框架级。

## 完整代码

前向 device kernel（逐字，v2.7.1 RNN.cu，截取 `//SAVE FOR BACKWARDS` 相关完整段）：

```cpp
__global__ void gru_cell_forward(
            TensorInfo<scalar_t, index_type> Input,
            TensorInfo<scalar_t, index_type> Hidden,
            TensorInfo<scalar_t, index_type> Bias1,
            TensorInfo<scalar_t, index_type> Bias2,
            TensorInfo<scalar_t, index_type> _hx,
            TensorInfo<scalar_t, index_type> _hy,
            TensorInfo<scalar_t, index_type> storage,
            index_type hsz,
            index_type totalElements) {
  bool has_bias = Bias1.data != nullptr;
  for (index_type linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
      index_type offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

      scalar_t ir = DEVICE_LINEAR_GET(Input, offset+0*hsz);
      scalar_t ii = DEVICE_LINEAR_GET(Input, offset+1*hsz);
      scalar_t in = DEVICE_LINEAR_GET(Input, offset+2*hsz);
      scalar_t hr = DEVICE_LINEAR_GET(Hidden,offset+0*hsz);
      scalar_t hi = DEVICE_LINEAR_GET(Hidden,offset+1*hsz);
      scalar_t hn = DEVICE_LINEAR_GET(Hidden,  offset+2*hsz);

      scalar_t hx = DEVICE_LINEAR_GET(_hx, linearIndex);
      scalar_t* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);

      scalar_t b1r, b1i, b1n, b2r, b2i, b2n;

      if (has_bias) {
        b1r = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+0*hsz);
        b1i = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+1*hsz);
        b1n = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+2*hsz);

        b2r = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+0*hsz);
        b2i = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+1*hsz);
        b2n = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+2*hsz);
      } else {
#ifndef THC_REAL_IS_HALF
        b1r = 0.0; b1i = 0.0; b1n = 0.0;
        b2r = 0.0; b2i = 0.0; b2n = 0.0;
#else
        b1r = F2H(0.0); b1i = F2H(0.0); b1n = F2H(0.0);
        b2r = F2H(0.0); b2i = F2H(0.0); b2n = F2H(0.0);
#endif
      }

      offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

      accscalar_t rg, ig, ng;

      rg = sigmoid(H2F(ir) + H2F(hr) + H2F(b1r) + H2F(b2r));
      ig = sigmoid(H2F(ii) + H2F(hi) + H2F(b1i) + H2F(b2i));

      ng = H2F(in) + H2F(b1n) + rg*( H2F(hn)+H2F(b2n) );
      ng = ::tanh(ng);
      *hy = F2H( ng + ig * ( H2F(hx)-ng ) );

      //SAVE FOR BACKWARDS
      DEVICE_LINEAR_GET(storage, offset+0*hsz) = F2H(rg);
      DEVICE_LINEAR_GET(storage, offset+1*hsz) = F2H(ig);
      DEVICE_LINEAR_GET(storage, offset+2*hsz) = F2H(ng);
      DEVICE_LINEAR_GET(storage, offset+3*hsz) = hx;
      DEVICE_LINEAR_GET(storage, offset+4*hsz) = F2H(H2F(hn) + H2F(b2n));
    }
}
```

反向 device kernel（逐字）：

```cpp
__global__ void gru_cell_backward(
             TensorInfo<scalar_t, index_type> gradInInput,
             TensorInfo<scalar_t, index_type> gradInHidden,
             TensorInfo<scalar_t, index_type> gradOutput,
             TensorInfo<scalar_t, index_type> gradInputHx,
             TensorInfo<scalar_t, index_type> storage,
             index_type hsz,
             index_type totalElements) {
  for (index_type linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    index_type offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

    scalar_t rg = DEVICE_LINEAR_GET(storage, offset+0*hsz);
    scalar_t ig = DEVICE_LINEAR_GET(storage, offset+1*hsz);
    scalar_t ng = DEVICE_LINEAR_GET(storage, offset+2*hsz);
    scalar_t hx = DEVICE_LINEAR_GET(storage, offset+3*hsz);
    scalar_t hn = DEVICE_LINEAR_GET(storage, offset+4*hsz);

    scalar_t go = DEVICE_LINEAR_GET(gradOutput, linearIndex);

    offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

    accscalar_t gig = H2F(go)*( H2F(hx)-H2F(ng) )*( 1-H2F(ig) )*H2F(ig);
    accscalar_t ghx = H2F(go)*H2F(ig);
    accscalar_t gin = H2F(go)*( 1-H2F(ig) )*( 1-H2F(ng)*H2F(ng) );
    accscalar_t ghn = gin * H2F(rg);
    accscalar_t grg = gin *H2F(hn)*( 1-H2F(rg) )*H2F(rg);

    DEVICE_LINEAR_GET(gradInInput, offset+0*hsz) = F2H(grg);
    DEVICE_LINEAR_GET(gradInInput, offset+1*hsz) = F2H(gig);
    DEVICE_LINEAR_GET(gradInInput, offset+2*hsz) = F2H(gin);

    DEVICE_LINEAR_GET(gradInHidden, offset+0*hsz) = F2H(grg);
    DEVICE_LINEAR_GET(gradInHidden, offset+1*hsz) = F2H(gig);
    DEVICE_LINEAR_GET(gradInHidden, offset+2*hsz) = F2H(ghn);
    DEVICE_LINEAR_GET(gradInputHx, linearIndex) = F2H(ghx);
  }
}
```

反向 host 侧关键段（逐字）：

```cpp
static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

void checkGRUBackwardSizes(const TensorArg& grad_hy, const TensorArg& workspace) {
  CheckedFrom c = "fused_gru_cell_backward";
  checkDim(c, grad_hy, 2);
  checkSize(c, workspace, {grad_hy->size(0), grad_hy->size(1) * GRU_WORKSPACE_MULTIPLIER});
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _thnn_fused_gru_cell_backward_cuda(
      const Tensor& grad_hy, const Tensor& workspace, bool has_bias) {
  checkGRUBackwardSizes({grad_hy, "grad_hy", 1}, {workspace, "workspace", 2});

  int64_t hidden_size = workspace.size(1) / GRU_WORKSPACE_MULTIPLIER;
  auto grad_input_gates = at::empty({workspace.size(0), hidden_size * 3}, workspace.options());
  auto grad_hidden_gates = at::empty({workspace.size(0), hidden_size * 3}, workspace.options());
  auto grad_hx = at::empty_like(grad_hy, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad_hy.scalar_type(),
    "_thnn_fused_gru_cell_cuda_backward",
    [&] {
      if (canUse32BitIndexMath(workspace)) { // See Note [64-bit index math check elision]
        gru_backward_impl<scalar_t, int32_t>(grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx);
      } else {
        gru_backward_impl<scalar_t, int64_t>(grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx);
      }
  });

  at::Tensor grad_input_bias, grad_hidden_bias;
  if (has_bias) {
    grad_input_bias = grad_input_gates.sum(0, /*keepdim=*/false);
    grad_hidden_bias = grad_hidden_gates.sum(0, /*keepdim=*/false);
  }

  return std::make_tuple(
    std::move(grad_input_gates),
    std::move(grad_hidden_gates),
    std::move(grad_hx),
    std::move(grad_input_bias),
    std::move(grad_hidden_bias)
  );
}
```

autograd 挂接（derivatives.yaml v2.7.1，逐字）：

```yaml
- name: _thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
  input_gates, hidden_gates, hx, input_bias, hidden_bias: "grad.defined() ? (GradMode::is_enabled() ? _thnn_differentiable_gru_cell_backward(grad, input_gates, hidden_gates, hx, input_bias, hidden_bias) : _thnn_fused_gru_cell_backward(grad, result1, input_bias.defined())) : std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>()"
```

## 预防

- 实现"前向缓存/反向消费"类融合算子时，把缓存张量契约（平面数、每平面语义、偏置合并与否）写进 spec 并附参考 kernel 逐字摘录，实现与测试都以摘录为准，不凭记忆。
- 反向实现前先核对：平面 4 语义（hn+b2n）、平面 3 是原始 hx（不是 hy）、门平面顺序 [rg, ig, ng] 与 gate 梯度输出三段 [grg, gig, gin]/[grg, gig, ghn] 的对应关系。
- has_bias=false 时 bias 梯度输出的表达（aten 返回未定义张量；aclnn 层需要实体张量，如 shape (0,) 空张量或零填充）是独立的接口决策，不要和布局混为一谈。

## 验证信号

以下均为本次开发会话内的实测核验：

- PyTorch v2.7.1 RNN.cu 从 GitHub raw 逐字抓取，`GRU_WORKSPACE_MULTIPLIER=5`、前向 5 平面写入（平面 4 = `F2H(H2F(hn)+H2F(b2n))`）、反向 5 链读平面、`checkGRUBackwardSizes` 强校验 `{B, H*5}` 全部与抓取原文一致（来源平台：950（本机抓取核验；契约本身为 PyTorch v2.7.1 框架级事实）；证据等级：tool_observed）。
- 本机 torch 2.7.1 头文件 schema `(Tensor grad_hy, Tensor workspace, bool has_bias) -> 5 tensors`、native 仅 CUDA 后缀（来源平台：950（本机 torch 2.7.1 头文件）；证据等级：tool_observed）。
- derivatives.yaml v2.7.1 挂接公式逐字核验（workspace=result1、has_bias=input_bias.defined()）（来源平台：950（本机抓取核验）；证据等级：tool_observed）。
- NPU 实测前向 `_thnn_fused_gru_cell` 返回 workspace=None（torch_npu 分解路径不产出该张量，佐证"workspace 是融合路径私有契约"）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 布局契约绑定 PyTorch v2.7.1；其他版本须重抓源码核验（未验证其他版本）。
- CUDA kernel 的 `H2F/F2H`（half↔float 提升与收窄）是 CUDA 侧精度链；NPU 实现的收窄 RoundMode 须另行对齐 aten 的 round-to-nearest-even 语义（CAST_RINT），参见 f32→f16 收窄 RoundMode 一卡。
- LSTM fused cell 的 workspace 是 4 平面布局，不可类比。

## 坏实践

- 拿数学公式（rg/ig/ng/hx/hn 符号）反推缓存布局而不读参考 kernel 的写入代码——最容易在"平面 4 是 hn 还是 hn+b2n 还是 ng"上错。
- 前向/反向分头实现、各自理解布局，不做契约对账。

## 误诊

- 无（本会话未发生布局误读；旧工作区研究摘录经官方源码重抓后逐字一致）。


