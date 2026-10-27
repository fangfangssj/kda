---
type: Guide
title: 先界定未定义占位输出的比较合同
description: 通过非零 shape 占位输出与扩展调用前返回的独立源码，区分内容无定义、数值失败和真实内核执行。
tags:
- task.testing
- topic.precision
- topic.programming_model
status: draft
sources:
- resource: https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/_torch_docs.py
  role: constraint
- resource: https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/docs/source/deterministic.md
  role: constraint
created_at: '2026-10-19T05:14:57Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: []
---

# 先界定未定义占位输出的比较合同

比较失败前先确认返回值的合同：某个分支返回了 Tensor，不表示该 Tensor 的元素具有规定值；函数名中的 `empty` 也不表示零元素。对 Ascend C 扩展 wrapper，需要同时检查 reference 返回的含义、wrapper 实际路径和比较器所比较的对象。不能仅凭数值不等就定位到设备计算，也不能反过来用占位分支排除其他内核缺陷。

## 参考与 wrapper 的独立证据

本页只采用一个普通 KvRmsNormRopeCache 任务自己的 reference、wrapper 和两份原始验证日志。原 reference 文件 SHA-256 为 `cc3c4a08a789d9481238cfd10423883c769309eba4c43423da0218fc789ea2e8`，wrapper 为 `53dc420890712eb20b099d8c4ab9fc0b633dc1845bf70c3b50ffd0db3a231491`。原件中的 `is_output_kv=False` 分支为：

```python
# Reference: k_rope and c_kv have already been computed.
if is_output_kv:
    return k_rope, c_kv
else:
    return torch.empty_like(k_rope), torch.empty_like(c_kv)
```

wrapper 在调用 `_ext.run_kv_rmsnorm_rope_cache(...)` 前先执行：

```python
D = kv.shape[-1]
Dv = D // 2
if not is_output_kv:
    k_rope_empty = torch.empty(cos.shape, dtype=cos.dtype, device=kv.device)
    c_kv_empty = torch.empty(kv.shape[:-1] + (Dv,), dtype=kv.dtype, device=kv.device)
    return k_rope_empty, c_kv_empty
```

日志中该分支的输入形状是 `kv=(1,1,16,128)`、`cos=(1,1,16,64)`、dtype 为 float16。两个返回值形状均为 `(1,1,16,64)`，**每个有 1024 个元素**，并非零元素输出。它们与参考分支有相同 shape，并不能推出两次独立分配的内容相等。

在这个 wrapper 的单次 `forward` 中，false 分支的返回先于唯一扩展运行调用，因此这条分支上的内容比较失败，不能直接归因为该扩展调用的数值计算错误。这不表示模块没有提前 import、注册或构建，也没有证明 true 分支或其他输入的设备实现正确。

## `empty` 的边界

固定 PyTorch `d38164a545b4a4e4e0cf73ce67173f70574890b6` 文档将 `empty` 与 `empty_like` 的默认内容描述为未初始化；有分配空间不等于有数学上指定的结果，也没有“必为零”的保证。

同时要保留一个明确例外：`torch.use_deterministic_algorithms(True)` 与 `torch.utils.deterministic.fill_uninitialized_memory=True` 同时生效时，固定文档说明浮点和复数填 NaN、整数填最大值。NaN 的比较仍取决于比较规则；确定性填充不能自动产生这个算子本应计算的结果。本文没有绑定历史运行的 PyTorch 版本或上述设置，不能把当前固定文档当作旧环境填充值的证明。

## 失败记录能说明什么

两份原始日志各记录前四个 case 匹配，第五个 false 分支失败，即 **4/5**；第一返回值分别有 1019/1024 和 1021/1024 个元素不等。两份文件 SHA-256 分别为 `cba6d99db2ff1ef3afe476e0d5cd52c2b94bb7860c86c98a0f422bee82506e58`、`8b5cfe561cf21265a77258322e18a3ba4e67758b76a32a8aa6f8f162e1bc2204`。它们是归档观察，不是本次新增复现，也不足以推断第二返回值的比较结果或同一个二进制被独立复现两次。

本次新做的标准库 CPU 探针只读取允许片段的 AST，以受限解释器重放 shape 计算、分支、抽象分配和返回，不 import 旧工程或调用 PyTorch／NPU。历史 case 元数据和另一组原创 shape 均返回两个非零元素占位输出，且 false 分支没有进入扩展调用。另构造内容全 0 与全 1 的两个抽象分配见证：二者 metadata 相同、内容不同，证明仅靠 metadata 无法要求相等。这些数值是明确标注的逻辑见证，不是从历史 allocator 读到的值。true 分支仅检查没有命中这条 early return，没有执行其余计算。

## 处理顺序与停止条件

1. 固定 reference、wrapper、比较器版本，逐分支写清返回数量、shape、dtype、device、别名／状态更新以及内容是否有定义。
2. 将实际失败 case 绑定到分支；用源码调用顺序或独立执行证据确认是否到达扩展运行调用。区分模块导入、扩展启动和计算完成。
3. 若输出内容没有定义，记录合同与比较器之间的冲突，并交由合同所有者确认应比较哪些性质。只有明确授权和合同依据才能调整比较器或 wrapper。
4. 对内容有定义的分支继续做正常精度验证；内容未定义的分支也仍需检查合同规定的 shape、dtype、device、状态及安全性。不要把所有可选输出都统一改为只比 shape，也不要用更宽 tolerance 掩盖内容无定义。

本页保持 draft、平台未确认。可审阅附件 `split_contracts` 提供逐行原件哈希绑定、完整输入／输出、`verify_contracts.py` 和 `MANIFEST.json`；脚本只认证上述静态合同与数学见证，不认证历史修复、设备执行或整算子正确性。接口信息充分性的另一类静态反例见[MoE 路由接口的信息充分性](moe_routing_interface_information_sufficiency.md)。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/source_snippets.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/inputs.json#/placeholder_cases`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/outputs.json#/placeholder_ast_cases`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/outputs.json#/historical_extract_checks`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/verify_contracts.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
