---
type: Runbook
title: ARM64 Torch bmm 的 FMA 归约顺序失配判别
description: 固定 Torch2.10 ARM64 原dtype CPU bmm 合同时，诊断FP32顺序FMA与FP16四路partial的舍入路径；原完整50
  CPU41→50通过、zero全拒绝，native两腿12/50、性能未测、实际内层CPU微核未知。
tags:
- task.debugging
- topic.precision
- topic.compute.reduction
status: draft
sources: []
created_at: '2026-10-09T14:16:28Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases:
- bmm sequential FMA four partial
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# ARM64 Torch bmm 与 FP64 更近仍失配：FMA 归约顺序判别

## 触发与适用范围

原验收合同固定为 CPU `torch.bmm(A, B)`、输入原 dtype、冻结输入及全输出比较时，补偿点积更接近诊断 FP64 数学结果，仍可能不符合合同。一个已归档的 Ascend950/DAV_3510 精度修复中，前版完整50为41通过9失败：剩余7个 FP32、2个 FP16；BF16 30条已经通过。新版本仅改变非BF16的融合乘加和归约顺序，完整50达到CPU合同50通过。

本卡给出诊断与处置步骤，不提供完整算子源码或可继承的二进制成绩。范围绑定实际 CPU/工具链、冻结输入和同一新DSO；其他 Torch版本、CPU架构、backend、shape、dtype或tiling须重新判别。状态为 draft，实际内层CPU microkernel仍 UNKNOWN。

| 层次 | 实际绑定与证据边界 |
|---|---|
| CPU参考 | ARM64 Torch2.10.0+cpu，commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`，SVE256，OpenMP192线程 |
| CPU库 | oneDNN3.7.1，commit `8d263e693366ef8db40acc569cc7d8edf644556d`；安装的ACL库SHA256 `db7012d7cd455078a260d75cc3fd47bbde72086c235b4ddf824f313e9d860f45`，治理绑定ACL v52.6.0/commit `007264fa740de5723ebddef16b7bb3657692c088` |
| Device构建 | Ascend950/DAV_3510，实际 CANN9.1.0 inner B243，bisheng15.0.5（2026-07-30），实际驱动声明并接受 `--npu-arch=dav-3510` |
| 入口与输出 | 原 `torch.bmm` 模型；输入/输出原 FP32、FP16、BF16；设备float carrier由原输入无损提升而来，末尾转换回原dtype，没有参考结果回填或库bmm回退 |
| 未测项 | 性能P/Q、MFU均 UNMEASURED；不能由运行完成推出效率、speedup或性能路线推荐 |

## 先固定真值与复現输入

1. 记录原模型、用例集合、比较器和原输入manifest的hash；复算使用原输入的同一字节、shape、dtype、layout，不用相同分布的新随机数据代替。
2. 明确合同究竟要求数学高精度结果，还是固定CPU实现的输出。本例后者已冻结；FP64、标量FMA和不同累加方式只作诊断，不替换验收真值。先计算原CPU结果并核对与保存的CPU输出是否逐字节相同；本次两个代表用例已核对相同。
3. 保留CPU、native和candidate三方输出，分别执行candidate-vs-CPU、native-vs-CPU、candidate-vs-native。另执行全零candidate负控。native失配保持失败，不由CPU成功覆盖。
4. 每次新DSO绑定其source/build/library hash并重跑完整分母；worst坐标或修复子集通过只是诊断。

原冻结比较器按输出dtype要求shape/dtype相同，NaN/Inf位置及Inf符号一致；有限元素满足 `abs(a-g) <= atol + rtol*abs(g)` 的比例至少0.99，且全输出最大绝对差不超过cap：

| dtype | rtol | atol | max_abs cap |
|---|---|---|---|
| FP32 | 2^-10 | 2^-16 | 0.01 |
| FP16 | 2^-9 | 2^-9 | 0.1 |
| BF16 | 2^-6 | 2^-6 | 1.0 |

这个表只复述本次原合同，不授权修改其他项目标准。有关CPU真值、完整分母和既定阈值的一般规则见[精度标准](../../guides/precision/precision_standard_v2_1.md)；其优先FP64的标准不能覆盖已另行冻结为原dtype CPU bmm的合同。

## 绑定实际backend，再检验算术假设

1. 从实际Torch build/version与安装库绑定固定提交，先读 [LinearAlgebra.cpp 的 bmm 调度](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/LinearAlgebra.cpp#L1604)：`bmm_out_cpu` 调用 `bmm_out_or_baddbmm_`，后者按条件选择MKLDNN、small naive、BLAS或按batch拆分的addmm。只沿实际适用分支继续读相关backend，以及适用时的 [CPUBlas.cpp](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/CPUBlas.cpp) 和 [BlasKernel.cpp](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/BlasKernel.cpp)；不假定每个bmm都经过同一链，也不将其他版本源码当当前实现。
2. 对原输入采集CPU backend日志。本次FP32代表用例 `DNNL_VERBOSE` 观察到 `gemm:acl`；FP16代表用例没有ACL matmul事件。没有事件不证明实际调用了哪一个fallback函数。
3. 固定ACL [FP32选择源码](https://github.com/ARM-software/ComputeLibrary/blob/007264fa740de5723ebddef16b7bb3657692c088/src/core/NEON/kernels/arm_gemm/gemm_fp32.cpp)及相关SVE FMLA片段仅为假设线索：未定位实际选择的内层执行symbol/object，不能由选择表或文件名断言microkernel。
4. 对失败worst坐标取原A行/B列，逐次显式FP32舍入，分别比较非融合乘加、顺序FP32 `fmaf`、四路FP32 `fmaf`、旧补偿点积；FP16再做原末尾half转换。诊断使用的CPU `libm.fmaf` 和设备SIMT FMA是两条实现，不能以一个标量相等替代整版设备证据。

### 算术判別与观测

- **FP32顺序全K FMA**：初值s=0，依原K顺序作 `s <- FMA32(A[k], B[k], s)`。7个保留worst坐标全部与原CPU标量相同；非融合或四路FP32没有覆盖全部7个。
- **FP16四路partial**：4个FP32 partial初值0；每个完整4项组分别依序融合到s0..s3。末尾不足4项全部继续融合到s0；最终按 `((s0+s1)+s2)+s3` 左折叠，随后只做一次原half转换。匹配固定PyTorch相关half reduction的源码形态，并在2个worst坐标与原CPU输出相同；顺序FMA和旧Neumaier结果失配。
- **更接近FP64不保证更接近合同**：旧补偿候选在这些坐标更接近诊断FP64点积，却仍不等于原CPU输出。这是复现固定算术路径的判别，不是“补偿算法普遍无效”或数学精度天花板。

| FP16原坐标 | 原CPU / 旧candidate | 顺序FP32 FMA → half | 四路FP32 FMA → half |
|---|---|---|---|
| (0,14,47) | 5460 / 5456 | 5457.952637 → 5456 | 5458.009277 → 5460 |
| (0,10,2) | -2534 / -2532 | -2532.983398 → -2532 | -2533.007568 → -2534 |

[跨CANN代归约树卡](cross_generation_reduction_tree_prefer_more_accurate.md)的证据是CANN目标端golden、候选自身误差更大、简单树匹配多次失败后改用更准的树满足其合同。本例冻结原dtype CPU bmm，旧补偿候选虽更近FP64却失败，而有版本绑定的FMA顺序探针及完整新DSO通过；两者条件不同，均不能扩展为所有归约必须匹配树或必须采用更准树。

两点在正常half值域的舍入边界附近，不能借用[次正规1ULP判别](fp16_subnormal_1ulp_mare_amplification_diagnosis.md)放行，也不采用[softmax上转golden修复](torch_multidim_fp16_softmax_no_internal_upcast.md)改变本例原CPU。9个worst探针提供可证伪假设，完整50新DSO实测才提供最终精度证据。

## 实施与同步边界

- 只改变被证伪的非BF16算术分支。FP32全K保持一个累加器；FP16完整4项组和剩余项的分配、最终左折叠必须保持。切K、重置partial、改变合并树或跨线程归约会改变舍入路径，采用后须重新完整验证。
- 本次设备路径为SIMT AIV独立输出点积：每个输出只由一个线程写，输入只读，partial线程私有；数据/类型/layout完整且输出点全覆盖。carrier提升、kernel和原dtype输出转换走同一NPU stream，验收同步后读回。没有跨线程partial共享，也未以未初始化输出或裁尾获得通过。
- 若改为UB/Vector/Cube或共享workspace，必须重新证实搬入完成、计算到写回、buffer重用及跨线程partial可见性的依赖；本次SIMT结果不证明另一条流水的同步正确。有关Cast→标量的特定跨pipe规则见[Cast链卡](fp16_cast_up_fp32_compute_cast_down_chain.md)，不将那条实现直接套到SIMT。
- 匹配CANN工具链中，SIMT `simt_api/math_functions.h` 的global inline `fmaf`提供 `__fma` 路径；前次 `__builtin_fmaf` 编译成未解析external `fmaf`，其失败保留。切换为该实际API声明后新DSO构建成功。编译成功不证明数值/融合语义，应继续原合同实测；其他编译器/头文件/target需重新核对。
- 本次保留已通过的BF16分支，未重新论证其所有shape/backend适用性；本卡的非BF16修复不能独自生成或保证BF16成绩。[BF16多matmul链卡](bf16_matmul.md)处理多级workspace量化，职责也不同。

## 整版证据与版本身份

同一冻结50、原whole-tensor比较器、新DSO实测，consumer exit0；完整记录包括错误与未运行数量，全部没有错误或未运行：

| 对照 | PASS | FAIL | ERROR | NOT_RUN |
|---|---|---|---|---|
| 前版candidate vs 原CPU | 41 | 9 | 0 | 0 |
| 新版candidate vs 原CPU | 50 | 0 | 0 | 0 |
| native vs 原CPU | 12 | 38 | 0 | 0 |
| 新版candidate vs native | 12 | 38 | 0 | 0 |
| zero vs 原CPU | 0 | 50 | 0 | 0 |

FP32 12/12、FP16 8/8、保留BF16 30/30。7个FP32和2个FP16旧失败全部转通过，先前通过无回归；新版本全50最大绝对差为0。独立非作者产品证据审查核验全部50存储字节、dtype/shape/offset/stride与原CPU一致，并核验全150份CPU/native/candidate原输出的归档hash。此审查是结果证据，不能替代本卡官方committer内容认可或新消费者复现。

| 冻结对象 | SHA256 |
|---|---|
| 原模型 | `0cb4e8b5ada312f68bc97a9f14fb95cc3647763ab231266799e80b7a6ea97464` |
| 原cases | `0d7f2c65bb2eca3896b942ccc9f2a8386cd7933ccf262b776ad180eaba9c4333` |
| 原比较器 | `bb0a2515bb7590ee6bf1fe08dcafc5913c0ba5b5bf28a02ea7f4d080a9273df2` |
| 原输入manifest | `e0ee72712e711d0242c8a615ec7258b7ce7c11df0bdb26c62b7fa3b44b256c71` |
| 新kernel源码 | `93367f898a8e4e10e94b3a5c6312e5f608885c3539cba45ff8d6817a939022e6` |
| 新wrapper源码 | `7e055fa2d52db5045df6e208ba8fb9c2f23442a178e747b29b42b57cafbabbf1` |
| 新build源码 | `9dd3f7972708363b19db34b5a5476392877b4b9572bc28e36c2388aa065d1df1` |
| 实际loaded DSO | `28b8f708cf47483f8ed1d030bafaf6cbb06fcb6f46e07b3650919d88b5c2573e` |

公开固定提交的完整50逐例结果（内部材料，不公开）、前后完整50变化（内部材料，不公开）、产品非作者审查（内部材料，不公开）和终态绑定（内部材料，不公开）支持这一精度-only结论。来源可回到精确版本，未向知识仓导入实现源码、DSO、原输入、会话轨迹或内部设备标识。

## 停止条件与未覆盖范围

- 原CPU50通过后保存该整体版本；不为追native把已满足CPU合同的算法悄改。Native两腿12/50与零50拒绝均是完整负证据，不能隐藏。
- P/Q和MFU尚未测；SIMT串行K与4路partial不是性能建议，运行时限内完成不证明吞吐。既有[AIV大batch超时卡](../performance/aiv_vec_matmul_large_batch_timeout_507035.md)保留规模风险，不能把本次50完成推广为任意规模可行。
- 实際内层CPU microkernel UNKNOWN。源码链、backend日志与scalar witness共同支持本次修复，未证明每次CPU派发或全部shape都选相同kernel。
- 原合同固定50的成功不证明全部输入、非有限输入、其他K分割或不同库版本的等价性；新DSO必须重新验收。
- 本次为WHITE证据链，旧KB-assisted BLACK结果41/50保持原样。后续新冷知识消费者须是未曝光人员，独立读取受治理知识、写source、构建自己的DSO并跑原完整50/zero/native；不能把作者更名BLACK或继承旧DSO成绩。此后续冷复现在本卡形成时 NOT_DONE，不据本次结果声称因果KB优势或非辅助发现。

## 适用场景

- **工程形态**：固定 Torch2.10 ARM64 原dtype CPU bmm 合同时。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
