---
type: Runbook
title: Host Tiling UT 基础设施故障诊断
description: 面向 ACLNN 等注册式算子包，按编译、链接、动态加载、Registry、Context 和平台 mock 分层诊断 Host Tiling 单元测试故障，区分测试设施失败与算子断言失败。
tags: [task.debugging, task.testing, topic.compilation, topic.implementation.tiling, topic.implementation.operator_registration, topic.implementation.debugging]
status: stable
sources:
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/workflow/op-develop.yaml
    role: constraint
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/add_custom_package/tests/tiling/test_tiling_gtest.cpp
    role: example
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/scale_custom_package/tests/tiling/test_tiling_gtest.cpp
    role: example
  - resource: https://gitcode.com/opencann/skills-dev/blob/09989172eea4a7f83ede82fab7e49ba1a47b0083/plugins-official/ops-registry-invoke-glacier/operators/scale_custom_package/tests/tiling/CMakeLists.txt
    role: example
created_at: "2026-09-10T10:08:59Z"
updated_at: "2026-09-11T02:50:20Z"
platforms: [agnostic]
aliases:
  - ACLNN Host Tiling UT 故障诊断
  - 注册式算子包 Tiling 测试故障
  - Tiling gtest infrastructure failure
  - Host Tiling UT all red
  - Tiling Registry failure
  - TilingContext mock failure
  - GetCoreNumAiv GetCoreMemSize mock failure
  - Host Tiling SDK symbol diagnosis
---

# Host Tiling UT 基础设施故障诊断

## 适用场景

本卡主要适用于 ACLNN 等注册式算子包的 Host Tiling 层测试。Registry、ContextBuilder 和平台 mock 的诊断依赖对应设施；独立直调工程可复用编译、链接、动态加载与断言分层方法，并按实际 host 切分入口定位调用故障。

本 Runbook 用于 Host Tiling UT 尚未进入可信 oracle 判断时的工程故障，包括：

- 测试或 op_host 构建失败；
- `.so` 存在但加载失败；
- Registry 中找不到算子或 TilingFunc；
- ContextBuilder 编译通过但调用崩溃；
- 平台核数、UB 等 mock 值没有被 TilingFunc 读取；
- 红阶段全部失败，但失败来自基础设施而非 oracle；
- 为寻找接口而反复扫描 SDK 二进制，仍无法闭合调用链。

如果 TilingFunc 已经稳定执行，只有预期字段和 oracle 不一致，应回到算子设计和 [Host Tiling 单元测试开发](../../guides/operator_practice/host_tiling_ut.md)，而不是继续处理基础设施。

首次接入测试设施时，按指南的“最小调用链与覆盖扩展”建立可编译和调用的探针。已有测试设施则优先使用能够复现故障的最小用例；首个编译、链接或加载错误决定后续排查范围。

## 先确定失败层级

一次只回答“失败发生在哪一层”：

| 层级 | 最早可信信号 | 常见现象 |
|---|---|---|
| 1. 源码编译 | 编译器已处理目标测试 TU | 头文件不存在、类型/成员不存在、模板或命名空间不匹配 |
| 2. 链接 | 链接器已接收全部目标对象 | undefined reference、找不到库、目标链接到错误架构对象 |
| 3. 动态加载 | 加载器尝试打开打印出的绝对 `.so` | undefined symbol、依赖库缺失、加载了旧包 |
| 4. Registry | `.so` 已成功加载并执行注册 | op type 不存在、TilingFunc 为空、默认 Registry 不是测试安装的实例 |
| 5. Context 构造 | 函数指针存在，builder 开始组装输入 | IO slot、attr、buffer、接口族或生命周期不匹配 |
| 6. Platform mock | TilingFunc 能执行但平台相关输出异常 | 核数/UB 为零、总是默认值、修改 mock 后输出不变 |
| 7. Oracle/红基线 | TilingFunc 被真实调用且输出可读 | 失败集合不符合 stub 预期、意外通过或断言不完整 |

后层现象不能用来跳过前层证据。例如测试进程退出不证明是 Platform mock 问题；必须先证明 `.so`、Registry 和函数调用已闭合。

## 最小诊断证据

每次运行至少保存：

```text
测试可执行文件路径
隔离安装根
准备加载的 op_host .so 绝对路径
动态加载成功/失败
查询使用的准确 op type
Registry 与 TilingFunc 是否非空
首个最小 case 是否进入 TilingFunc
进程退出码和未截断的首个根错误
```

只保存测试框架末尾的“若干 case failed”会丢掉首个根错误。若设备或进程状态随后发生级联，诊断只采用第一个非级联失败。

## 层级 1：编译错误

### 头文件存在，但类型或成员不存在

优先判断是否混用了接口族或不同版本样例：

1. 从当前测试 TU 的 include 确定 builder 的命名空间。
2. 在当前环境的公开头文件中定位完整类声明。
3. 核对样例使用的类、方法和 `Build` 返回类型是否来自同一声明。
4. 删除为了迎合另一个接口族而添加的 adapter，先让最小 builder 用法编译。

不要通过盲试相似成员名推进。`ContextBuilder`、`OpTilingContextBuilder` 等名称相近，但内部上下文类型和槽位不一定兼容。

### 类型完整性或析构报错

检查是否只 include 了前置声明，以及持有者是否需要完整类型定义。智能指针析构、模板实例化和 `sizeof` 通常要求完整类型。优先补充声明该类型的公开头文件，不要通过泄漏对象或删除析构逻辑绕过。

### 构建命令没有编译本轮文件

检查 verbose build 中目标测试源路径和时间戳。若构建目录缓存了旧源、旧安装前缀或另一个算子包，应清理本测试目录可再生的 build，再重新 configure；共享安装使用独立前缀隔离。

## 层级 2：链接错误

### 找不到库

区分两类问题：

- `cannot find -lX`：搜索目录或组件未安装；
- 已找到库但 `undefined reference`：声明存在，提供实现的目标依赖未闭合或链接顺序/可见性有误。

先从当前头文件所属组件、已有 CMake target 和相邻公开样例推导库依赖。记录完整链接命令，再对缺失的准确符号做一次定向查询。不要一开始把 SDK 下所有库加入链接列表。

对于采用 `gert::OpTilingContextBuilder` 与 `OpImplSpaceRegistryV2` 的工程，可按下表核对基础依赖，再决定是否需要符号工具：

| 缺失能力或符号族 | 优先核对的公开组件 |
|---|---|
| `OpImplSpaceRegistryV2`、`OppSoDesc` | `libopp_registry.so` |
| gert/ge Context、Tensor、Shape、属性元数据 | `libmetadef.so` |
| `fe::PlatFormInfos` | `libplatform.so` |
| 测试可执行文件直接使用 `PlatformAscendC` | `libtiling_api.a` |
| CANN dlog 输出 | `libunified_dlog.so` |
| `memcpy_s` 等安全 C 接口 | `libc_sec.so` |

同时检查 `_GLIBCXX_USE_CXX11_ABI=0`、隔离 op_host 目录与 CANN `lib64` 的 RPATH/`LD_LIBRARY_PATH`。这张表用于形成第一组候选，不表示每个目标都要无条件链接全部组件。

### 链接成功不代表可加载

共享库可能允许未解析符号保留到加载期。因此构建成功后仍要执行独立加载探针，并将加载失败归到下一层。平台管理相关缺符号可参考 [直调共享库加载未定义平台符号—补齐CANN链接依赖](platform_manager_shared_library_link_dependencies.md)。

## 层级 3：动态加载错误

### 先证明加载的是目标文件

打印规范化绝对路径，并检查它位于本轮隔离安装树。常见假成功包括：

- `LD_LIBRARY_PATH` 优先命中了共享安装目录；
- build 使用新代码，run.sh 却指向旧 dev-package；
- `.so` 名称相同，但 vendor 或算子包不同；
- 重新构建后没有重新安装。

必要时记录文件摘要或修改时间，但路径仍是最基本证据。

### undefined symbol

按以下顺序处理：

1. 保存准确的 demangled/mangled 符号和加载目标。
2. 根据声明头文件或现有 API 卡判断所属组件。
3. 查看目标的完整链接命令和运行期依赖。
4. 仅对候选提供库执行定向符号查询。
5. 补齐依赖后重新 configure、build、install 和 load。

如果尚未确定加载的是本轮 `.so`，不要进入符号归属分析。

## 层级 4：Registry 错误

`.so` 加载成功而 TilingFunc 为空时，逐项检查：

- 注册宏使用的 op type 与测试查询字符串是否完全一致；
- 当前进程查询的默认 Registry 是否就是测试创建并安装的 owner；
- Registry owner 在查询和函数调用期间是否仍存活；
- 动态加载是否发生在查询之前；
- 注册代码是否实际包含在被加载的 op_host `.so`；
- 本轮加载路径是否指向正确 vendor 和算子包。

验证算子包注册与加载时，测试应通过 Registry 调用 TilingFunc。直接调用源文件内部函数只能验证函数自身逻辑，不能证明算子包注册与加载正确。

## 层级 5：Context 构造与生命周期

### 编译成功、调用时崩溃

首先怀疑 Context 引用的底层对象已经失效。列出 builder 接收的每个指针，并标记 owner：

| 对象 | owner | 有效期必须覆盖 |
|---|---|---|
| input/output shape 与 TensorDesc | case storage | builder 构造到 TilingFunc 返回 |
| attrs 与属性容器 | case storage | 同上 |
| platform storage | case storage/fixture | 同上 |
| TilingData buffer | case storage | TilingFunc 写入与断言读取 |
| workspace 容器 | case storage | TilingFunc 写入与断言读取 |
| Registry owner | fixture/suite | `.so` 注册、查找与全部调用 |

临时 vector、局部 attr、函数内 platform 对象和仅返回裸 context 指针都是重点检查项。

### 输入存在但 slot 错位

核对 OpDef 和 Interface 中的 IO 数量、顺序、可选输入实例数、dtype 与 StorageShape。不要用“空 TensorDesc”代替一个不存在的 optional input，除非当前 builder 契约明确这样表达。

### TilingData 或 workspace 无法读取

确认缓冲区容量、对齐和 builder 的对应输出 slot。对 TilingData 先用零或哨兵值初始化，调用后检查真正发生写入；但不能把“哨兵发生变化”替代逐字段 oracle。

## 层级 6：Platform mock 未生效

典型信号是：TilingFunc 可以返回，但核数、UB 相关输出始终为零、默认值，或修改 mock 值后输出不变。

使用两组只改变一个平台资源的最小 case：

```text
Case P1: core = C1, UB = U
Case P2: core = C2, UB = U       // 只改变 core

Case M1: core = C, UB = U1
Case M2: core = C, UB = U2       // 只改变 UB
```

选择会跨越设计阈值的值，使输出应发生可预测变化。如果仍无变化，沿下面四点检查：

1. **producer**：资源值是否真的写入平台 owner；
2. **storage**：序列化或内部存储是否仍有效；
3. **injection**：是否从当前 builder 接口族的正确入口注入；
4. **consumer**：TilingFunc 是否通过与注入数据匹配的平台读取对象访问。

不要仅检查 `platformInfo != nullptr`。一个非空但布局、接口族或生命周期错误的指针仍可能返回默认资源，甚至导致未定义行为。

API 定义可从 [PlatformInfo](../../apis/utils_api/tiling_tuning/context_builder/platform_info.md) 与 [PlatformAscendC](../../apis/utils_api/platform_info/platform_ascend_c/index.md) 查询。若 API 卡给出 `source_headers`，优先读取当前环境对应声明，不跨版本猜内部字段。

对于采用 `fe::PlatFormInfos` 和 `gert::OpTilingContextBuilder` 的工程，来源中的公开样例提供以下生产者—消费者映射；使用时核对当前 SDK 接口族与消费 API：

| 消费 API | `SetPlatformRes` label | 关键 key |
|---|---|---|
| `GetCoreNumAiv()` | `SoCInfo` | `vector_core_cnt` |
| `GetCoreMemSize(UB)` | `AICoreSpec` | `ub_size` |
| 当前核心类型选择 | `SoCInfo` + `SetCoreNumByCoreType` | `core_type_list` + `AICore` |
| SoC/架构查询 | `version` | `Short_SoC_version`、`NpuArch` |

完整初始化片段和生命周期约束见 [Host Tiling 单元测试开发](../../guides/operator_practice/host_tiling_ut.md)。具体资源数值来自当前任务的平台输入；不要复制样例的芯片常量。

若 TilingFunc 还读取 block size、cache line、vector register size 或系统 workspace，先判断它们在当前设计中是“case 可控资源”还是“平台固定量”。前者需要 producer—注入—消费者差分闭环；后者由设计值或公开 API 形成 oracle，不应通过枚举 `.so/.a` 重新发现。接口签名有疑问时只读取声明该准确 API 的头文件。

## 层级 7：测试结果不可信

对已有实现进行回归时，先检查失败是否发生在加载、Context 或平台初始化阶段；调用链成立后，字段断言才用于判断实现与规格是否一致。以下红基线检查适用于测试先行开发：stub 的已实现行为必须已知，才能判断预期通过和失败集合。

### 所有 case 都失败

逐个检查第一个失败是否满足：

- `.so` 已加载；
- TilingFunc 已找到并进入；
- Context 和平台数据可读；
- 失败来自 expected 与 stub 输出不同。

只要其中任一项不成立，“全红”就是基础设施失败，不是 TDD 红基线。

### 除豁免外仍有 case 通过

该 case 的断言可能只检查返回码、默认零值或 stub 也能满足的字段。回到设计规则，为它加入至少一个未实现状态无法偶然满足的字段或 route 断言。不要加入与算子语义无关的强制失败。

### 失败数正确，但原因错误

测试框架只统计失败数不够。保存每个 TEST 的首个断言位置，并抽查不同类别：普通 route、阈值边界、平台资源变化、异常出口。确认它们分别落在预期 oracle，而不是共享 Fixture 的同一个错误。

## 有界 SDK 查询策略

当知识卡不足以确定接口时，只围绕当前缺口查询，避免把任务变成 SDK 逆向：

| 已知信号 | 下一步允许扩大到的范围 |
|---|---|
| 准确类或 API 名 | 对应 API 卡及其 `source_headers` |
| 准确 include | 该头文件、直接 include 和声明所属组件 |
| 准确 undefined symbol | 当前目标链接命令和少量候选提供库 |
| 准确 op type 未注册 | 当前 op_host 注册源、加载路径和 Registry |
| 平台值未生效 | 当前 builder 声明、平台读取 API 和公开同族样例 |

### 符号排查的边界与停止条件

以下行为容易扩大排查范围。只有它们能够回答一个尚未被公开声明、构建命令和依赖关系解释的具体问题时，才具有诊断价值：

- 循环扫描多个 `.so`/`.a`；
- 对静态库执行 `ar x`；
- 对 SDK 对象文件执行 `objdump`；
- 用 `strings` 猜平台资源键或日志函数；
- 逐库试链但没有保存第一个 undefined symbol。

如果问题属于 Registry、`PlatFormInfos`、`GetCoreNumAiv`、`GetCoreMemSize(UB)`、基础链接集合或 stdout 日志捕获，可先核对本 Runbook 与同接口族公开样例给出的调用和依赖关系，再用最小编译或运行用例验证尚不确定的部分。

停止宽搜的条件：

- 已经找到当前版本的公开声明；
- 当前工程采用的接口族已经确定；
- 下一个问题可以由最小编译或运行探针回答；
- 新一轮搜索只返回同一批符号而没有新增契约信息。

只有在确定的 ABI、链接或动态加载问题无法由头文件、构建命令、公开样例和依赖树解释时，才对一个明确候选库使用 `nm` 或 `readelf`。每次探针应回答一个问题，并记录它排除了什么；禁止遍历整个 SDK、解包静态库或反汇编对象后再凭出现次数猜接口。

## 修复后的验证顺序

修复任何基础设施问题后，从最小层级重新验证：

1. 单个测试 TU 编译；
2. 测试目标链接；
3. 目标 `.so` 独立加载；
4. Registry 查找准确 op type；
5. 一个最小 Context 调用；
6. 一组平台资源差分探针；
7. 一个 route 的完整 oracle；
8. 完整测试集；采用测试先行开发时，核对红基线。

不要在第 2 层修复后直接运行数十个 case。分层验证能明确这次修改关闭了哪个信号，并避免多个错误相互遮蔽。

## 修复验收检查表（待执行）

本卡依据固定版本公开样例整理诊断方法；以下是执行者应采集的预期信号，不是本卡已经完成的实验记录。实际排障后，应保存对应环境、原始日志和观察结果，再判断基础设施是否闭环；这些信号不单独证明算子 oracle 正确：

- 构建日志显示目标测试源和目标 op_host 均来自当前工作区；
- 动态加载日志打印隔离安装树中的绝对 `.so` 路径；
- Registry 查找到准确 op type 的非空 TilingFunc；
- 最小合法 Context 调用可控返回；
- 对读取平台资源的实现或测试探针，改变核数或 UB mock 后，设计相关输出按预期变化；
- 回归结果能够区分设施故障与规格断言失败；采用红阶段时，失败定位到 oracle 断言，通过集合符合 stub 已满足的契约。

## 失效边界

- 本 Runbook 不提供特定 CANN 版本的内部 platform 二进制布局。
- 某一版本所需的链接库集合不能未经核对复制到另一版本。
- Registry、Context 和 Platform 均闭合后，字段值仍错误属于算子设计、实现或 oracle 问题，不应继续扩大基础设施排查。
- 设备侧 kernel 错误、精度错误和性能问题不属于 Host Tiling UT 基础设施范围。
