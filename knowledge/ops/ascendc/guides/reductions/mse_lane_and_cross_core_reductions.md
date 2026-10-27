---
type: Guide
title: MSE 的 Lane 累加与跨核归约
description: 保存两种归约树，补齐短尾、标量写回、工作区和全核到达条件。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/sync_all.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.md
  role: constraint
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md
  role: constraint
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-10-17T19:15:04Z'
platforms: []
---

# MSE 的 Lane 累加与跨核归约

mean MSE为sum((pred-target)^2)/N，前提是输入配对/广播与N符合接口。来源一核方案每个tile按lane累计acc[j]+=sq_tile[j]，最后对tileLength个lane归约；多核方案每tile先归约为scalar，核内再累加partial，最后核0汇总全部核。两种求和树在实数等价，浮点顺序与误差不同，不能用一方结果强制宣称另一方错误。

一核方案acc先初始化0，对最后短tile只更新有效cnt个lane，其余lane保留此前贡献；若补齐sq为0可保持整长Add，但不能把acc尾lane清0而丢掉之前满tile贡献。源ceil(nElements/1024)配整块DataCopy会过读；多核pivot+ceil(2048)也同样需要每核myElems和有效cnt。老init floor(elemsPerCore/tileSize)同时可能丢核间余数和核内尾部。

输出仅一个数学结果，来源InferShape复制pred形状是接口错误，需与实际标量/单元素合同一致；outputGm长度1不能靠更大分配假装形状正确。普通DataCopy(...,1)的4B写回及每核workspace[pid]写回不普遍合法，使用满足短尾与地址要求的路径并确认相邻核写入互不破坏。只有已写的nUsed个partial可被最终读取，pad区不能参与sum。

多核阶段需要每个实际参与核写出partial完成后，全核同步，再由核0读。SyncAll不是任意环境中无条件合法：核型、参与者、所有核到达、事件资源和实际重载支持须匹配。硬同步与软件同步不同，软件接口还需GM/UB标记区及初始化，不能把nUsed个partial空间兼作未分隔标记。不能在闲核提前return后让其他核继续等待其到达。

来源CopyOut1从Scalar SetValue直接发DMA且立即Free，未通过出队交接；需明确Scalar→MTE3可见性、DMA完成与buffer复用的合同。ReduceSum后GetValue需要V结果已就绪。核内queue交接与核间发布是不同依赖，SyncAll不能修复发布之前的错误数值或越界。

Host申请userWorkspace=nUsed*sizeof(float)+sysWorkspace并不自动决定Kernel的user pointer，按框架取实际用户区基址。workspaceInQueue需覆盖nUsed有效值及搬运物理补齐，归约scratch按nUsed和重载要求，不因tileSize2048就必定足够所有核数。源一核6个tileLength FP32区加输出至少需24*tileLength有效数据字节和输出对齐，多核7个tileSize区加workspace输入为28*tileSize+4*nUsed，均需额外资源与对齐检查。

N0时均值按接口定义可为NaN或其他行为，不能除以0后默认正常；sum/none reduction也不是这两篇mean算法。减法/平方可能溢出，FP32求和并非所有N和数值范围都足够，验证时保留算法精度合同和独立reference。

## 单结果写回的有效长度与同步条件

以下增补依据 `asc-devkit@639b2b5c1031188107d23d785d52b935f1b4f13b` 的 Scalar 访问、UB→GM DataCopyPad 和 SetFlag/WaitFlag 文档，仅补齐接口约束。没有新增 NPU 测试，也不继承历史量化 scale 写回、Sum kernel 注册或标量 DMA 的成功结论。

一个 FP32 数学结果是 4B，UB 源起始地址、源槽位容量、GM 有效输出范围与搬运描述符分别核算。固定 UB→GM DataCopyPad 文档支持 VECIN/VECOUT→GM，UB 源起点要求 32B 对齐；`DataCopyExtParams.blockLen` 单位为字节，且为 `sizeof(T)` 的整数倍。可为一个 FP32 结果安排满足目标重载物理要求的对齐 UB 槽位，单块搬运使用 `blockCount=1`、`blockLen=sizeof(float)`、零 stride 和 `rsv=0`。这表示 4B 有效 GM 写入，不表示必须写 8 个 float，也不能据此用普通 `DataCopy(...,1)` 替代非对齐搬运。内部分块、dummy 与源容量仍按目标版本检查；不能把逻辑位置和起点对齐一概推广到任意 LocalTensor 视图。

输出位置也要保持契约：单元素输出、每核 partial workspace 与归约 API 的目的 tensor 是不同角色。直接把归约目的地设在合法 VECOUT tensor，可避免一次额外 UB 拷贝的设计候选，但须核对该归约重载的输出布局、dtype、别名和工作区要求；不能仅从“最终只要一个标量”推出这种别名总是允许。GM workspace 的每核槽位仍须有独立所有权和合法物理搬运范围，不可为对齐而扩大写入并覆盖相邻槽位。

### 自动同步与三段依赖

官方 Scalar 文档说明，`GetValue` / `SetValue` 属于 PIPE_S：算子工程开启自动同步时，不需要为这些访问另加手动同步事件；关闭自动同步时，开发者必须补齐相应依赖。先确认实际构建模式，不能把“手动加事件”或“编译器会处理”作为所有工程的统一假设。

| 依赖 | 要证明的事实 | 不能代替它的动作 |
|---|---|---|
| Vector 归约结果→Scalar `GetValue` | Scalar 读取前，Vector 已产生实际读取的值；手动模式按目标可用机制建立 V→S 依赖 | 先读取再等待，或只在外层 helper 返回后加 barrier |
| Scalar `SetValue`→MTE3 读取该 UB 槽位 | 手动模式下，固定 Scalar 文档示例在搬运前使用匹配的 `S_MTE3` SetFlag/WaitFlag；自动模式按工程合同处理 | `PIPE_V` 的文本存在、反向事件，或搬运之后才补等待 |
| MTE3 消费完 UB→下一个写者复用槽位 | 搬出已完成对该源槽位的读取；通过目标可用的同步机制或正确的 Queue 生命周期交接 | 只有前一段 S→MTE3 就绪依赖，或发起搬运后立即覆盖源数据 |

SetFlag/WaitFlag 文档规定事件按“源流水_目的流水”命名，模板事件和 eventID 必须完全匹配；在 TPipe-TQue 范式中按其资源管理接口取得 ID。SetFlag 在源流水前序读写完成后设置标志，但不阻塞源流水的下一条指令。因此，搬出前建立 S→MTE3 可见性，不等于搬出后 Scalar 已可再次 `SetValue`，也不等于 Host 已可读取 GM 输出。最后一次设备搬出到 Host 消费还需要实际执行框架的 stream/设备完成合同。

SetFlag/WaitFlag 属于 ISASI 接口，文档不保证跨硬件版本兼容；此处不规定所有平台共用一个手动事件序列或事件 ID。Queue 的 EnQue/DeQue 也只在真实匹配的 producer/consumer 生命周期内提供对应交接，不能用一个无关 Queue 或跨核 `SyncAll` 替代槽位的读写依赖。

审查单结果路径时，依次标出结果生产者、Scalar 读取点、Scalar 写入点、实际搬运重载、GM 目的区和下一次 UB 复用点。短尾/相邻 partial、多次调用、最后一次搬出、自动同步开启和关闭应分别按所用工程验证；本文没有把这份检查清单记为已执行探针。
