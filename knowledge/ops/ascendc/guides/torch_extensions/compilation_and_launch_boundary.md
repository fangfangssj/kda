---
type: Guide
title: PyTorch Ascend C 扩展的编译单元与启动 ABI
description: 分离设备编译单元、普通 C++ 注册代码和共享 Tiling 布局，核对编译成功到算子可调用之间的各层契约。
tags:
- task.implementation
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ascendc-op-extension/references/direct-invoke/troubleshooting.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ascendc-op-extension/references/direct-invoke/operation_checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: supporting
created_at: '2026-09-10T12:45:59Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: []
---

# PyTorch Ascend C 扩展的编译单元与启动 ABI

设备实现包含 Ascend C 关键字或启动语法时，应由相应设备编译链处理；普通 C++ 的 PyTorch 注册单元不能靠包含整个设备源文件取得实现。跨编译链接口通常由不含设备专用语法的声明头连接。

Host 与 Kernel 共用的 Tiling 数据需要保持字段类型、顺序和布局一致。若启动包装采用 `extern "C"`，调用侧声明和定义的参数数量、顺序、指针及整数宽度也必须一致；消除 C++ 名字修饰并不消除 ABI 不匹配。编译器、C++ ABI 选项和运行库链接需要与当前框架及 CANN 安装相容。

编译产物存在、动态库成功加载、dispatcher 完成注册、NPU 实现被调用，是不同层次的证据。出现未定义符号时核对真正加载的共享库及依赖；出现 `torch.ops` 属性缺失时核对扩展加载和注册命名空间；只有特定设备调用失败时，核对相应 dispatch 实现。

Meta 实现描述输出的 shape、dtype 等元信息。Meta 可用不保证图后端已有对应转换规则，eager 正确也不保证图编译路径使用了相同输入、属性和输出契约。测试应分别覆盖扩展加载、eager 调用、Meta 推导及实际支持的图后端。

来源中的 CMake 布局、安装路径和构建命令是模板实例，不是所有扩展必须采用的目录或流程。

## Tensor 绑定的动态符号归属

插件 EC-55 区分两个常被混淆的符号：其 Tensor 绑定中 type_caster<at::Tensor>::cast 由 libtorch_python 提供，AutogradMeta 的 vtable 则来自 PyTorch 核心 libtorch_cpu，通常经 libtorch 依赖可达。来源工程仅链接 torch_npu 时扩展可生成，却在导入时报缺符号；增加所需 torch/torch_python 依赖后修复。这支持先检查符号定义与实际依赖，不能由任意 torch 符号报错直接下 ABI 不匹配结论。

核对当前安装实际提供的库与生成链接命令、DT_NEEDED 及加载路径；ldd -r 独立运行可能报告由 Python 解释器提供的 C API 符号，最终导入需在匹配的全新解释器中观察。已经导入旧扩展的进程可能缓存制品，重建文件存在不保证测试消费新库。依赖与符号解析正确后，再核对 torch/torch_npu/扩展的 ABI 组合。

缺符号问题先核对实际依赖与符号解析。[EC-63](../../runbooks/compilation/ec_63_std_string_in_pybind11_crashes_with_basic_string_n.md) 记录了 V220 ARM64 / CANN 9.0.0 的 pybind11 字符串参数调用出现 `basic_string null`、SIGSEGV 139，并给出改用 `const char*` 或 `py::str` 的规避建议。采用前，应核对目标宿主 ELF 架构与实际加载制品、编译器 / 运行库和 C++ ABI 配置、调用栈及字符串参数寿命，判断当前故障是否匹配原记录。这里沿用其原观测和规避建议，未重放该历史实验；设备架构或编译器名称本身不能替代目标宿主的 ABI 证据，参数类型替换也不能代替缺符号原因排查。


## 构建成功但注册信号异常时，先区分数值错误与执行链路

当整批输出明显失配，而同次日志出现 `RegisterAscendBinary mix ret 107000`，先核对设备二进制注册、启动包装和实际消费的产物，再研究舍入或容差。以下是两个普通算子任务的历史观测，支持这个分诊顺序；它们不证明错误码只有一个根因。

| 独立任务 | 修改前的同次原始日志 | 终版原始日志及静态源码 |
|---|---|---|
| Add | 构建 `rc=0`；编译器告警 `Multiple kernel functions are detected. It is recommended to define only one kernel function per file.`；随后注册返回 `107000`，10 条比较均失败。 | 精简验证 10/10、全量验证 50/50；终版快照有 8 个源码文件，3 个 dtype 设备入口分别位于不同 `.cpp`，由普通 C++ 绑定选择对应 launcher。 |
| GroupNormSwish | 同样先构建 `rc=0`，再出现多 kernel 告警、注册返回 `107000`，10 条比较均失败。 | 精简验证 10/10、全量验证 50/50；终版同样有 8 个源码文件和分开的 dtype 入口。终版还包含同步、UB/tiling、数据搬运和数值路径的处理，没有单因素对照，不能把通过结果单独归因于拆分文件。 |

两组日志的构建命令均选择 `Ascend910B3` 和名为 `cann-8.5.0` 的安装目录。这只是历史构建配置，不足以认证实际物理 NPU 型号、加载的运行库或本轮重新执行结果。告警中的“建议”也不能推广成所有 CANN 版本和构建方式的“一文件只能一个 kernel”全局限制。

### 将观测、推断和验证分开

1. 保留同次构建退出码、首发注册信号、最终比较结果和源码快照身份。静态检查终版设备入口、launcher 声明、dtype 路由及构建清单；看到生成 `.so` 不能直接跳到“设备计算已经发生”。
2. Add 本轮历史策略明确跳过设备插桩。已有档案没有实际的 null-handle、输出哨兵或设备 `printf` 探针，因此“handle 为空”“kernel 未执行”“输出必为零”都不能作为直接观测。注册返回与大范围失配只是继续检查链路的信号。
3. 重现类似问题时，记录新解释器实际加载的扩展路径及哈希、依赖、注册返回和受控执行信号，再与固定输入的正式比较关联；新的实验是新来源，不能回填旧档案的加载身份。保留比较失败，不用放宽容差掩盖启动链路疑点。
4. 终版通过证明所记录版本在当时验证集上的结果；本节采用的证据未包含完整修改前源码、加载二进制绑定和单因素对照，不认证“只拆分文件即可修复”或完整执行根因。

### 审阅证据与去重

随本次审阅材料提供 `split_registration/`：16 份终版源码及两份原快照 manifest 保持原字节；`observations.derived.json` 是六份 stdout 的明确派生逐行摘录，保留原文件 SHA-256、行号、构建／注册信号和全部比较行，去除了私有路径及输入明细；`probe_policy.derived.json` 保留 Add 跳过探针的原字段。`verify_public.py` 只以 Python 标准库核对这些文件的身份、入口结构和观测统计，不编译、不加载历史二进制、不执行 NPU。

旧 Add 摘要与本次 Add 档案共享 SHA-256 为 `40ab475cf556e131591446ee15e7b9110e0742508185afc9d28b1b783d5b4cba` 的同一份取证文件，不能计为独立复现。其他未找回准确原件的旧摘要仅作主题线索，不继承本节的验证结果。原源码中的历史自证注释不扩大这里声明的范围；本卡继续保持草稿和未定位平台状态。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/observations.derived.json#/observations`；`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/probe_policy.derived.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/source_manifests/add.original.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/source_manifests/group_norm_swish.original.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/provenance.derived.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_registration/MANIFEST.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
