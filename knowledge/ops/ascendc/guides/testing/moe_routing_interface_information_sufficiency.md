---
type: Guide
title: 用 MoE 路由反例证明内核接口的信息充分性
description: 以固定原版 MoE 路由合同和无相等键的 CPU 反例，证明遗漏专家数据的扩展接口不能唯一决定重排输出。
tags:
- task.testing
- topic.precision
- topic.programming_model
status: draft
sources:
- resource: https://github.com/Ascend/op-plugin/blob/5ac12689b6a61ec9db3fc260eab17bbc5447567e/test/test_custom_ops/test_npu_moe_init_routing.py
  role: primary
- resource: https://github.com/Ascend/op-plugin/blob/5ac12689b6a61ec9db3fc260eab17bbc5447567e/op_plugin/ops/opapi/MoeInitRoutingKernelNpuOpApi.cpp
created_at: '2026-10-18T16:25:28Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: []
---

# 用 MoE 路由反例证明内核接口的信息充分性

在实现路由排序前，先固定调用边界：某张量存在于外层 `forward`，不等于其数据已经传入扩展。若扩展只能看到输入的投影 `V(u)`，而正式输出为 `C(u)`，必须满足同一 `V` 对应唯一 `C`。找出合法的 `u₁,u₂` 使 `V(u₁)=V(u₂)` 且 `C(u₁)≠C(u₂)`，即可证明这个接口无法完整实现该合同；增加分核或修正搬运索引无法补回被丢弃的信息。

## 固定本次采用的合同

本页的新证据起点是 `Ascend/op-plugin@5ac12689b6a61ec9db3fc260eab17bbc5447567e` 的原版 `npu_moe_init_routing`，不是 V2/V3。它也不证明任何历史环境安装了该提交。固定测试中的 `cpu_op_exec` 定义下列 oracle，固定 C++ 适配层显式接收 `x,row_idx,expert_idx,active_num` 并调用 `aclnnMoeInitRouting`。

令 `N=x.shape[0]`、`K=expert_idx.shape[1]`、`L=N*K`，`e=flatten(expert_idx)`、`r=flatten(row_idx)`：

1. `p=argsort(e)`，`d=r[p]`；测试对相等 key 使用稳定排序。
2. `expanded_row_idx[d[i]]=i`，`expanded_expert_idx=sort(e)`。
3. `A=min(active_num,N)*K`，`expanded_x[i,:]=x[d[i]%N,:]`，`0≤i<A`。

这里的 `row_idx` 沿用固定测试的排列构造：`arange(N*K).reshape(K,N).T`。它在 `[0,L)` 上是一一映射；不能替换为任意重复 row id 后仍假定逆映射有相同含义。C++ 层分配的三个输出形状分别是 `[min(active_num,N)*K,H]`、`[L]`、`[L]`，并检查输入 rank、dtype、对应形状与非负 `active_num`。这些是本次固定源码的边界，不是跨版本兼容声明。

## 不可区分输入对

选取固定测试矩阵已有的 `N=10,K=2,H=200,float32`，两次均令 `active_num=10`。第 `t` 行的 200 个特征都取 `(t+1)/16`，为可精确表示的二进制分数。`row_idx[t,j]=t+10*j`，dtype 为 int32。两组 expert 输入分别为：

```text
expert_a[t,j] = 2*t+j
expert_b[t,j] = 19-(2*t+j)
```

两组 key 都是 `0..19` 的排列，没有相等 key；本反例不依赖稳定排序。`active_num=N` 保留全部 20 个展开行，也不依赖截断位置或零值特殊规则。

考察如下候选扩展边界：

```python
run(x_padded, row_idx.contiguous().view(20).to(int32), H, XStride, YStride)
```

其中 `x_padded` 仅由 `x`、dtype 与 32B 行对齐生成。本例 float32 的 `H=200` 已经满足行字节数 800 对齐，故 `XStride=YStride=200`。两次扩展可见的全部输入值、dtype、shape、stride、offset、可见 buffer 的别名关系以及输出形状都相同。`expert_idx` 与可见输入不别名；它的值没有沿声明接口进入扩展。

固定 oracle 的实际 CPU 结果为：

| 输出 | 输入 a | 输入 b |
|---|---|---|
| `expanded_x[:,0]` | `1/16,1/16,2/16,2/16,...,10/16,10/16` | `10/16,10/16,9/16,9/16,...,1/16,1/16` |
| `expanded_row_idx` | `[0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19]` | `[19,17,15,13,11,9,7,5,3,1,18,16,14,12,10,8,6,4,2,0]` |
| `expanded_expert_idx` | `[0,1,...,19]` | `[0,1,...,19]` |
| 三个输出形状 | `[20,200] / [20] / [20]` | 相同 |

其余 199 列与第 0 列相同，因此上表也完整指定了两个 `expanded_x`。即使扩展能观察输出形状或排序后的专家集合，也不能区分这对输入。

## 可直接检查的 CPU 模型与证明

以下代码只重述上述无截断、无相等 key 的固定子域。它不调用框架算子：

```python
x = [[i / 16] * 200 for i in range(1, 11)]
r = [t + 10 * j for t in range(10) for j in range(2)]
ea = list(range(20))
eb = ea[::-1]

def oracle(e):
    p = sorted(range(20), key=e.__getitem__)
    d = [r[i] for i in p]
    inverse = [0] * 20
    for i, source in enumerate(d):
        inverse[source] = i
    return [x[source % 10] for source in d], inverse, sorted(e)

def visible_projection(e):
    # e is deliberately absent from the declared extension arguments.
    return x, r, 200, 200, 200, (20, 200)

assert visible_projection(ea) == visible_projection(eb)
assert oracle(ea)[0][0][0] == 1 / 16
assert oracle(eb)[0][0][0] == 10 / 16
assert oracle(ea)[1] != oracle(eb)[1]
assert oracle(ea)[2] == oracle(eb)[2]
```

若存在只依赖声明输入的实现 `F`，则投影相等推出 `F(V(u₁))=F(V(u₂))`。但正确性要求它分别等于两组不同的 `expanded_x`，形成矛盾。证明限定实现只能读取声明输入、合法配置与已声明状态；指针地址或未声明外部状态不具有 expert 语义。在受控 CPU 重放中可以复用相同 backing buffers，使地址本身也相等。通过调用帧、邻接内存或参考算子另取隐藏输入，已经改变了这个信息边界，不能作为原接口充分的反证。

因此在此子域上，接口至少需要显式传入能区分路由的 `expert_idx` 或等价的已计算路由排列。若被保护的边界无法改动，应报告与目标合同不相容；不能把这种限制归因成单纯的 kernel 索引错误。

## 验证范围与复用边界

本次公开证据实际执行了固定官方测试文件中 AST 提取的原始 `cpu_op_exec`，保留完整输入、输出与源码 SHA-256。公开包中的候选接口投影是显式建模，不声称执行过原 PyTorch 包装层或历史 `forward`；内部历史源码 AST 核对不在该公开包内，也不作为本页正式输出合同的来源。公开 CPU oracle、相同可见投影和数学矛盾共同支持上述限定接口结论。

本页没有运行 NPU、编译 kernel、测量性能或验证历史修复。未知历史 `torch_npu` 版本仍需另行绑定；固定的新提交不能认证旧 wrapper 的同版本输出，也不继承任何调用参考算子的旧“通过”记录。迁移到别的 API 时，须重新固定排序方向、row 编码、输出映射和 active 范围；本例 `%N` 公式不能替换为 V3 的其他 row 编码。

接口层次与关联字段的一般建模见[白盒接口模型的层次与关联字段](whitebox_interface_contract_layers.md)。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/c033/claims.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/full_outputs.json#/projections`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/full_outputs.json#/observed_oracle_outputs`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/full_outputs.json#/checks`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/interface_witness.py`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/test_npu_moe_init_routing.py`；`cannbot_non_level4_split_review_20261019/evidence_public/c033/MoeInitRoutingKernelNpuOpApi.cpp`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
