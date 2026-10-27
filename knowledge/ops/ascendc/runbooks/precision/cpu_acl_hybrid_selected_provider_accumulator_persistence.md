---
type: Runbook
title: CPU ACL hybrid 实际选择与跨 K 块累加状态判别
description: 固定原 CPU FP32 合同时，透明观察实际 factory/config 并检查 balanced K 分块后的累加器延续；保留完整修复与混合 dtype 转移失败的边界。
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: draft
sources: []
created_at: '2026-10-10T02:33:34Z'
updated_at: "2026-10-15T00:00:00Z"
platforms: ['950']
aliases: [ACL hybrid persistent FP32 accumulator, balanced K reload, gemm acl inner_k_block, 跨 K 块舍入失配, 顺序 FMA partial join 差异]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CPU ACL hybrid 实际选择与跨 K 块累加状态判别

## 触发与结论

原验收固定为 CPU 模型、原 dtype、原输入及 whole-output 比较时，一个 shape 的数值方案不一定适用于另一 shape。同一 Torch 版本、FP32 内部计算或 `gemm:acl` 日志都不能独自确定实际 microkernel。先绑定原 factory 返回的对象与 config，再读适用的执行路径和跨 K 块累加状态。

本卡处理一个窄问题：ACL hybrid 在后续 K 块把既有 FP32 输出重新装入 accumulator，继续同一顺序的 FMLA 链。把每块独立清零、最后相加，改变了舍入路径。输入搬运分块可以不同于 provider 的 K 分块；是否保留同一 FP32 状态才是待验证条件。

这是 WHITE 源码与实测证据，不是未曝光 BLACK_BOX 复现。新冷知识消费者的完整产品复现及迭代节省尚未测量。已有 [ARM64 Torch bmm FMA 判别](torch_arm64_bmm_fma_reduction_order.md)覆盖一般原 CPU 合同、标量探针和全分母验收；它保留的实际内层选择 UNKNOWN 不由本卡跨 shape 消除。

## 版本与证据范围

| 层次 | 已绑定事实 | 限制 |
|---|---|---|
| 原 CPU | 保留 backend 日志含 oneDNN `3.7.1`，git `8d263e693366ef8db40acc569cc7d8edf644556d` | 本卡所列原件不足以绑定该 factory episode 的 Torch git/build，记 UNKNOWN；同一 provider 不代表同一 microkernel |
| ACL 库身份记录 | 保留的镜像提取元数据记库 SHA256 `db7012d7cd455078a260d75cc3fd47bbde72086c235b4ddf824f313e9d860f45`、17,850,377字节，见后文固定来源 | 这是历史提取元数据；该库与 factory episode 的直接 loaded-maps 绑定、embedded 版本、公开 `007264fa740de5723ebddef16b7bb3657692c088` 源码映射均 UNKNOWN |
| 目标修复 | 保留结果的 Ascend950 AscendC 路径 | 这里不以未绑定的 CANN inner B243 或公开 v9.1.0 声称 exact build；实际 SDK 源码映射 UNKNOWN |
| 原合同 | 冻结模型、输入、dtype、全输出 grader；HF32 disabled；CPU 与 native 独立分账 | FP64/标量 FMA 仅诊断，不替换 oracle、裁尾或放宽阈值 |

## 诊断步骤

1. **绑定实际 provider。** 保存原模型、cases、输入 manifest、Torch build、已加载库的 SHA 与版本。读取实际 CPU backend 日志。`gemm:acl` 提供 provider 线索，但不证明具体 selected strategy。
2. **透明观察原选择。** 用匹配版本的原 factory ABI，原参数调用原函数；对原返回对象读取公共 `get_config()`。不强制 kernel filter，不替换 reference。记录 M/N/K、method、filter、inner K、outer N、fixed-format、fast-mode、accumulate 和 weight format。将原输出 hash 与未插桩运行比较，确认观测未改变结果。可选日志 getter 的符号解析失败属于观测 setup error；修正只删除该失败日志，不能改变 factory 参数。
3. **完整读适用链。** 读 factory 条件、参数/虚函数 ABI、strategy 选择、hybrid constructor、K/N blocking、fixed-format 输入执行、accumulate/activation/writeback 与失败路径。再读真实适用的输出高度、K 主循环和 remainder。选择表、文件名或可用 kernel 数不是运行选择证据。
4. **区分 K block 与状态边界。** 下述为固定公开 `007264fa740de5723ebddef16b7bb3657692c088` 的源码机制；由于实际库到源码的桥接 UNKNOWN，它是待验证的适用假设。此公开版本 `compute_k_block` 有约 512 的 target，并按总 K 与策略约束计算均衡分块。执行 `k0` 循环的后续 pass 传 `!first_pass || original_accumulate`，从 FP32 输出延续累加。检查该分支在实际 selected kernel 的 accumulator load/store 和全部 remainder 上是否成立。
5. **建立可证伪的算术探针。** 对原失败坐标的同字节 A 行/B 列，比较顺序 FP32 `std::fma`、独立 partial 再 join、已用方案以及诊断高精度结果。保留最终原 dtype 转换。匹配若干坐标只支持一个新假设；不能给未消费的整版 PASS。
6. **只做有依据的改动。** 若状态延续得到证据，保持一个 FP32 accumulator 跨输入 reload，避免清零后 join。输入搬运 chunk 大小不是 provider K block 的等价身份。新实现仍须满足搬入完成、Vector/Scalar 与下一搬入、输出写回的依赖；定向依赖方法见 [MTE2 与 Scalar](scalar_s_pipe_requires_explicit_mte_dependencies.md)及 [Vector/Scalar UB 交接](vector_scalar_ub_handoff_requires_explicit_completion.md)，不重复发明全局 barrier 方案。
7. **验收完整新版本。** 绑定新 source/wrapper/DSO 与实际 loaded maps，按原全分母、whole-output grader、原 dtype 和控制执行。CPU、native 与零控制分别保留；非有限结构、未写尾部和 ERROR/NOT_RUN 不能隐藏。精度合格后，性能仍须同版独立测量。

## 两个实际观测与失效边界

### Dense FP32：支持状态延续修复

原 dense `[4,4096,1024] × [4,1024,512]` 用例使前版完整50成为49通过1失败，最大差 `0.044921875`。两次透明 CPU 观测的完整输出 hash 相同，原输出最大差0；对象 config 为 method5、`sve_ffhybrid_fp32_mla_6x4VL`、innerK512、outerN32、fixed1、fast0、weightformat `0x100800`。第二份原始观测记录 getter 返回枚举0。固定公开 `007264fa740de5723ebddef16b7bb3657692c088` 的 `CPPTypes.h` 以 GENERIC 为第一项，所以**若实际库采用该枚举定义**，0对应 GENERIC；实际库到该源码的桥接未保留，真实 model/generic 分支仍 UNKNOWN。generic 高度路径只作为公开源码的条件性算术解释，不宣布实际执行了它。

1,084 个保留坐标的顺序 FP32 FMA 与原 CPU 逐位相等，支持将独立 chunk partial 改为连续 accumulator。新 DSO 的原完整50 CPU比较50通过、零控制50拒绝。保留的完整50比较支持原 grader 通过；本卡不额外声称新 candidate 的 dense whole-tensor 字节相等。独立 native leg 为32通过18失败，不作为 CPU真值或新增两边必须同时全过的门。

同版完整50 fair whole-forward Q 为 `0.09381108266287147`，未达1.1；精度成功不能推出性能成功。生成 task P 与 whole-forward Q 分开，不借其他版本或冷 Event。前版逐例失败和后版通过只能比较各自绑定的候选结果；缺少前版算法判别原件，不把该失败归因于 OpenBLAS、352分块或某种 partial-join 实现。

### 混合 dtype 全链：转移只修复一部分

另一个全链的 FP32 projection 观察到 M2/M4、K=N3840，相同 method/filter，但 innerK480、outerN32：512 target 经 `3840 / 8` 均衡为480。可选 CPUInfo getter 初次未解析；删除该日志后原 factory/参数/reference hash 保持，CPU model 仍 UNKNOWN。源码中读到的 generic/A64FX 高度2/4共同顺序链支持算术假设，不据文件名宣布真实 model 分支。

438 个选坐标顺序 FMA 相等，支持仅改变末尾 projection 的 continuous-FMA 路径。新实现用352元素输入 reload 并保持 FP32 accumulator，352不是原 ACL 的480。固定 R10/R11 逐例原件与本地元数据重算表支持：完整49原 CPU比较从11通过38失败到21通过28失败；全部49输入及原 CPU hash不变、10失败转通过、0回退。这是两个绑定版本的结果比较，不证明状态延续是唯一差异或唯一原因。选例的whole tensor相等仍不能覆盖其他28失败，包括保留的非有限结构失配。native-vs-CPU独立为10通过39失败。零控制为47拒绝、2合法接受，沿原合同保留；没有另加全控制拒绝门。Q/P未跑，原完整精度未合格。

因此，实际 provider/config 与状态延续是有用 discriminator，不能保证混合 dtype 全链成功。剩余误差需要新的中间数据流定位；不由这次失败宣布数值上限。

## 应拒绝的迁移

- 不把512、480、352写成通用 K 常数，也不把相同 `gemm:acl`、Torch版本或 dtype 当相同 kernel 的证据。
- 不把“顺序FMA”与32/64链交错FMA树、四路partial或独立chunk join视作相同数值算法。
- 不因更接近 FP64 就替换已冻结 CPU合同；不因 native 输出含忽略尾部而裁尾、poison oracle或拟合allocator残值。
- 不把选坐标/选例通过、编译通过、插桩无变化或归档存在升级为完整资格。
- 新 shape、dtype、CPU model、布局、provider/build、K尾部或输出高度都须重核选择与原全输出。完整50成功和完整49失败分别限定各自版本，不能拼接成绩。
- 来源与作者观测不证明新冷知识消费者已采用本卡，也不证明迭代次数、成功率或耗时改善。

## 来源核对与后续消费者

ACL `gemm_hybrid_indirect.hpp` SHA256 `348fc27733b090f4ba0c27e586e4998cc50e8ed27c5c3b25ba165169ad97e51b`；strategy header `2905fffd07b1a407e7e7af12d60ba648da219bf317012335759273cb69bdbdba`；generic kernel `3cf92c1b340417a1fccbcaa9e486f678cd2573d9bf62c195b3aa6d5aa6e268a7`。公开 ACL 实现与团队私有实测证据分列于 sources；后者不能匿名读取，不是公开复现。卡片不导入 candidate 答案代码、DSO 或输入，也不授权使用原运行资源。

## 实测证据访问与认证复核

公开 ARM ACL 源码支持所读版本的算术/分块机制；factory 的真实选择、所选坐标和完整产品结果来自**暂时不能匿名访问的团队 Wiki**。匿名 raw URL 的404不能当成公开复现成功；有权限的维护者可在已认证的该仓 clone 中核对，不在卡内或 Git 历史记录凭据。

原两份整版证据绑定提交 `38ea146494ff63ab273b53de8cdf6e97c2ed8308`：

| 文件 | Git blob | SHA256 |
|---|---|---|
| `projects/torch-to-a5-mfu-56/evidence/32_R58_QUALIFIED_CURRENT_20261009/SELECTION.json` | `cefe8ad1179e2a5b7cc8f8e44eba1a6c39d29401` | `fc88231de596613beafa3009e39b60adf998e045516e2ab91453170cffecf492` |
| `agents/codex02/level4_performance_20261008T2340Z/rotate0004/precision417/CONTINUE_0136/R11/NPU_W01/ORIGINAL49_TERMINAL.json` | `4fe8f6cd36e54ec0aa9b823d3594333b7013e186` | `a9a6cda07efe0bd3b301a17b1128a16199b8bb653558146275d96528593c27c8` |

透明 factory 参数、恢复限制、选坐标与终态的保留记录，另绑定提交 `fc8acb1b2a11d55f4c6837a4fab26071fd68de55`：

| 文件（同一固定提交） | Git blob | SHA256 |
|---|---|---|
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/CPU13_FACTORY_EXECUTION.json` | `91eee2b0da150a5250d452bd9e0e2a321d7cd91c` | `de9b78dff22e4233e30c6b02b9ecc6c52ef237b2fbdb2aea77df8ba982263315` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/CPU13_FACTORY_RESULT.json` | `d0b1a35c408502936d43be168ffc1c83969e306d` | `0e5abffb03cb211e3741ed0b3e4b837409eb651016077ec9d7577bdd3b13d54e` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/CPU_PROJECTION_ROWS.json` | `dc6f14443c3fc2f8aa071aa5bb4b7801c857e2e6` | `4955b2e5056c4698aac6945a47f81380f0f498321924f646b2d5f12b4010dd11` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/CPU_TERMINAL.json` | `e26713e88cf51de423ecd64308d020e710b9edd6` | `4431fb95dd10745b855a2254f6b9c8dc3375a40f7d3a29b2e7e0222d408aeb89` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/FACTORY_ACTUAL_TRACE.txt` | `0072d52bb730ccf12cfe3c488ab0a778692a4afa` | `82135921069846d9614ca8c6ed741e7e8490d3934884b95855ac6e284b161151` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/INSTRUMENT_RECOVERY.json` | `279f7987e423163bd901a4d0e69717f7841c29a1` | `b962afd357a800cd106ef08782061d15b5ced1e99c96f27aca934e2ebfb744c9` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/ORIGINAL49_TERMINAL.json` | `4fe8f6cd36e54ec0aa9b823d3594333b7013e186` | `a9a6cda07efe0bd3b301a17b1128a16199b8bb653558146275d96528593c27c8` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/PRIVATE_WIKI_ORIGINAL49_TERMINAL.json` | `4fe8f6cd36e54ec0aa9b823d3594333b7013e186` | `a9a6cda07efe0bd3b301a17b1128a16199b8bb653558146275d96528593c27c8` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/PRIVATE_WIKI_SELECTION.json` | `cefe8ad1179e2a5b7cc8f8e44eba1a6c39d29401` | `fc88231de596613beafa3009e39b60adf998e045516e2ab91453170cffecf492` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/R58_CPU_QUALIFICATION_PROJECTION.json` | `2a3a7056c82a995a38e2ae2952348c5bbdd39796` | `3a39810f5d6866f3d2a5661c26b7bc06de68e34fdc84fcbbac16fdd1aae7bcc2` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/acl_source_evidence/SOURCE_MANIFEST.json` | `9398432a8666abd0db753ca0be9a8f5040f3b954` | `b21c6e2344afc653dd6512a9c959ca59e235c4ac323f669a5f37715c5abc3c8d` |

复核方法：在获授权的 Wiki clone 中 `git fetch origin main`，用 `git show <40位提交>:<表中路径>` 取原始字节，`git rev-parse <提交>:<路径>` 核同表中 blob，并以 `sha256sum` 核字节。两张表的提交不可换成移动的 main。SOURCE_MANIFEST.json 逐项列出 SHA256/大小；历史元数据检查不等于重跑 PT/DSO 或公开张量复现。复核者须按需要回到完整数值原件，不用本卡替换 oracle。

Dense 两次透明 CPU 观测及1,084坐标探针另绑定提交 `9978b1ced3b9de0fdd54753aa21a7c964a291893`。这是已有历史 CPU stdout、reader 和observer源的保留字节；原始容器时间2026-09-29T17:40/17:45Z照录，不把2026-10-10出版时间当测量时间。两次原CPU SHA均为 `be1b8a339e3a2c801ea3b38e63ec64dd3e74f8c142e83006837db46716150153`。坐标探针只覆盖所选1,084点，不是新candidate完整资格；observer/reader源哈希与stdout中的记录逐项一致。

| 文件（同一固定提交） | Git blob | SHA256 |
|---|---|---|
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/CPU13_FACTORY_LAUNCH.stdout` | `edc37272b0ff1b216705cab7d00490f17fc03fc6` | `b5637dfc9706c0e9803061fd8859325adfb38395a7f31b4798c3d646be184049` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/CPU13_REPLAY_LAUNCH.stdout` | `761c0d3e056bdfcfdb35c987d3e400dd4ec66867` | `adc05db9104a930a0eac23a9815baebc315faad56ffe1116be8d71182c452591` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/CPU13_FACTORY_READER.py` | `bd33c9497bfb9a478af885b6eeb0e4cc497c618d` | `5805f5a3c3a418eeb2cc538902b00f4132229f03dfcf8ead8d1233cc11d4b0d7` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/CPU13_FACTORY_REPLAY_READER.py` | `346541c81c6dd708580163f976ea39edead064d8` | `ae7b80647de5cf9f93c17ec83996a0c36503a95147bbcbb1663ed1f0857d52d6` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/ACL_FACTORY_OBSERVER.cpp` | `5e1d13559ceb164f0a6d4e582629a3baeeffdd83` | `6571ac705809f0408777f2116a9dba893e7874e2b8fdc648e595e6f17d7ffa63` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/ACL_FACTORY_OBSERVER_v2.cpp` | `748dcea88c2dcc9267c3aaa715832961aa604100` | `918c9c575fd3f531dc9f40fba2c2529d73f480e8e911f901c5381666b4a62cb6` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/README.md` | `c1cdaf7a42e04f96d22edadf6e2ea68cf00fb6ec` | `60b28e9e01cd2c163b3c12e06874eccc955b8302a456bf692e3605f713343840` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_CPU13_REPLAY_SOURCE_20261010T0425Z/SOURCE_MANIFEST.json` | `451a8a2336f2ac8d060fcd78b4c40a3f17400e02` | `9ebb1d7eefd3b3242cccc5c4764ec86ed2003fb6a3c1c0205a7691b7819f0e63` |

公开 `arm_compute/core/CPP/CPPTypes.h` 同提交 SHA256 `1af2b04da42ca52211d47750b95e7656f7c976b246404762d67be012f6f3b652`，其中GENERIC为未赋显式值的CPUModel第一项；getter枚举0的解释仅限定这个公开版本。实际getter结果来自上述私有观测，但实际库与公开版本的桥接 UNKNOWN，所以实际 CPU model 不由该枚举推定；访问和上游替换条件不变。

A64FX 公开同提交实现 SHA256 `65c58ca5b39d619edfb8ffab1ca405f7c2aafc70f932529b7b064d8993cf1865`，仅为已读高度2/4累加延续的源码依据，UNKNOWN 的 CPU model 仍 UNKNOWN。


### 库身份及前后版比较补充原件

以下已有本地记录绑定固定 Wiki 提交 `bfc24891e9ec28f4e03c85f023592abfa133ba33`，与上述相同认证访问限制。`ACTUAL_CPU_ACL_LIBRARY_BINDING.json` 是旧镜像提取的库大小/哈希元数据，并未保留 embedded 版本字符串或 factory 进程的 loaded-maps 桥接，不据它推定公开源码版本。

Dense 前版 R57 的50行原件按 `independent_CPU.status` 重算49通过1失败；case13输入 SHA `e205c8078e99e67a32f87e22a56c2e894a6f1456653173b0ac0079c65a55975e`、CPU SHA `be1b8a339e3a2c801ea3b38e63ec64dd3e74f8c142e83006837db46716150153`、DSO SHA `0490eb1b1d81740de1654d80e630298ac134c0c6b05b25ea7be4903e868c43ea`，原 grader 的 max_abs `0.044921875` 超过 cap `0.01`。这些原件不认证前版内部算法。R58完整资格继续由前述 R58 projection 绑定，不能借 R57 的 native 结果。

混合链 R10 原件绑定49行前版 candidate/CPU/input/DSO；R11原件仍固定在前述 `fc8acb1b2a11d55f4c6837a4fab26071fd68de55`。`R10_R11_STATUS_TRANSITIONS.json` 用这些已有结果做 CPU-only 元数据比较，逐项保留相同49个输入和CPU输出哈希、10个FAIL→PASS、0个PASS→FAIL；它不是新张量/DSO执行，也不单独证明算法因果关系。

| 文件（同一固定提交） | Git blob | SHA256 |
|---|---|---|
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/ACTUAL_CPU_ACL_LIBRARY_BINDING.json` | `1244d2bacd1d1ea4d91c0534254df050abc79713` | `2c73a35094f78e5d31e3481f7f65644c72c746a81658f9a6831d252e0eea4a23` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/R10_ORIGINAL49_TERMINAL.json` | `d700b0d7d60b9ecc2d474bca8d763a24b972ed3c` | `483511403c8d1ae7a544e00eb33b9f9eec52e6554255077c9553b0299143b5d8` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/R10_R11_STATUS_TRANSITIONS.json` | `9b545d4ea91360935ce537be82ada3e6577c58f0` | `871dccd663d088508fb9d37002d651de152c1e0e81a6f3fe87392d08c2b79658` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/R57_CPU_QUALIFICATION_PROJECTION.json` | `165e348690c56e0c2a1cfe3c22f140e704ae2615` | `69ef0f68efd5dd6166e0245547aaf54710e898c655d333029e556a0e4f7303fe` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/R57_precision.jsonl` | `b06268aede68df936336c07d3715e0c619230b1b` | `4dfc8eda660dadcbd183790c90a70f484bdec5acef61210c38cdc165d6300f7f` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/README.md` | `9f54bb8a31a8dc9d2cf60173acfcfbd07f5aae6b` | `15318d7fca102881460144d40005096ea277698bb704dffda018b55912a6e4d8` |
| `projects/torch-to-a5-mfu-56/evidence/ACL_SOURCE_BINDING_COMPLETION_20261010T0518Z/SOURCE_MANIFEST.json` | `9fafb8aea7e5d715cb03782a4e7485c944566954` | `d35b6696466d3fd848fbd05981371d1321826327a479b8f1617e803edec869fb` |

**上游/公开晋升前，必须替换这些私有实测来源为获授权的公共证据，或改写无法公开复核的实测范围。** 本次只修复团队来源声明；不补造原始日志，不追加设备运行，不改变历史 P/F/Q。

新未曝光消费者应从受治理知识 Query→全文Read→记录采用/拒绝开始，自建 source/DSO，再执行其原完整合同。当前该 cold 产品实验 NOT_MEASURED；同一曝光作者的召回/全文读取只验证知识可达性。本卡内容已于 2026-10-10 经 fresh 非作者审查 ACCEPT（P2=1 非阻断），status 已晋级 stable。

## 适用场景

- **工程形态**：固定原 CPU FP32 合同时，透明观察实际 factory/config 并检查 balanced K 分块后的累加。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
