---
type: Guide
title: msaicerr 的 Dump dtype 解析与 AI Core Error 分析边界
description: 保存 msaicerr 对 dtype 覆盖、故障材料、路径约束、首次错误和环境样例的行为，区分解析成功与数值正确。
tags:
- task.debugging
- topic.precision
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/dump-dtype-conversion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/functions-and-restrictions.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/dump-parsing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/aicore-error-analysis.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/environment-check.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/msaicerr-toolkit/references/environment-preparation.md
  role: supporting
created_at: '2026-09-14T07:15:54Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
aliases:
- msaicerr
- dest_dtype
- bfloat16ext
---

# msaicerr 的 Dump dtype 解析与 AI Core Error 分析边界

本卡用于 Ascend C 算子排障，保存固定 `msaicerr-toolkit` reference 描述的工具合同。来源没有钉住 msaicerr 发行号，平台范围尚未核实，因此以下命令和不支持项是该资料的适用线索，不声称所有 CANN 版本都相同。分析前确认实际工具版本、安装路径、help 和采集环境；不要把旧版不支持项推成算子本身不受硬件支持。

## 原始 Dump 与指定 dtype 的转换

来源区分两种输入：`-d/--data` 传入包含文件名的原始 Dump，可解析出输入、输出或 workspace 的 `.bin/.npy`；对已解析的 `.bin` 额外传 `-dtype/--dest_dtype`，则按指定类型输出 `.npy`。

```bash
python3 msaicerr.py -d /path/to/exception_dump -out /path/to/parsed
python3 msaicerr.py -d /path/to/input.bin -out /path/to/converted -dtype int8
```

关键风险是：来源明确说指定 dtype 与原始记录不一致时只给 Warning，仍按用户指定类型解析。这不是先按原 dtype 读取再数值 cast 的等价保证。生成 `.npy` 或输出“Success convert”只证明该解析路径完成，不能证明数值、shape 或有效字节范围正确。真实 dtype 应来自算子接口、编译信息与对应输入/输出记录，不能由多个可成功的解析结果或输出文件名反推。

资料列出的 dtype 为 float32、float16、float64、int8/16/32/64、uint8/16/32/64、bool、bfloat16。输出名带所选 dtype，例如 `.input.0.int8.npy`；该后缀表达解析选择，不是独立类型证据。若日志含 `Can not read with dtype bfloat16`，来源给出 `bfloat16ext` 依赖线索，需按当前工具/Python 环境核实，不能以改成 float16 解析来绕过格式支持。

Dump 输入、输出、workspace 应关联同一任务、算子和阶段，核对编码、元素数、有效 span 与文件长度；workspace 不能仅按输出 shape 解读。通用字节完整性和部分失败边界见 [二进制测试 I/O](binary_test_io_and_partial_failure.md)。

## AI Core Error 分析的输入和覆盖

`-p/--report_path` 接收故障信息目录，资料要求同时具备 Dump、异常算子编译 `.o/.json` 和 CANN Host 日志，示例组织在 `dfx/data-dump`、`dfx/log/host/cann`。它分析材料内按日志时间第一次出现的 AI Core Error；一个 `info.txt` 不能证明全部错误均已解释，也不能将后续连锁错误反推成首个根因。

```bash
python3 msaicerr.py -p /path/to/fault_bundle -out /path/to/analysis -dev 0
```

执行目录不能位于 `-p` 目录或其子目录，`-out` 也不能等于或位于 `-p` 之下；来源将违反这些关系与解析卡住/失败关联。这是工具输入/输出约束，不能当作可删除的纯流程说明。未给出输出目录时，`-p` 分析默认写当前工作目录，而 `-d` 解析默认在 Dump 所在目录，两种默认值不同。路径非法、不可写或创建失败会使分析退出，不能用残留旧报告判断本次成功。

`-p` 会运行内置算子样例；独立 `-e/--env` 也可运行该样例，`-dev/--device_id` 默认为 0。样例通过只覆盖相应设备和样例执行链，不能证明业务算子、其他设备或完整环境组合正确。失败时应保留样例、构建和运行日志，不能仅凭失败便唯一归因驱动/固件或直接排除业务问题。`debug_info.txt` 是工具过程证据，`info.txt` 是故障分析入口，两者职责不同。

## 版本化能力边界

来源要求 Toolkit/CANN 运行环境、Python 3.7.5+，并将工具描述为在故障产生的本地运行环境分析，不支持 Ascend RC 形态。环境变量应由与实际安装配套的 `set_env.sh` 建立；示例 `/usr/local/Ascend/cann` 不是所有安装的固定路径。内置样例确实需要运行环境，不应从能够离线读取二进制推断完整 `-p` 分析可任意跨环境使用；也不能把这一分析限制自动扩大为所有 Dump 解码在物理上都必须上板。

该资料列出的 AI Core Error 分析不支持项为 MatmulAllReduce 类、MatmulAllReduceAddRmsNorm、MatmulAllReduceInplaceAddRmsNorm、AllGatherMatmul、MatmulReduceScatter、GroupedMatmulAllReduce、MemSet、NonMaxSuppressionBucketize。使用实际版本前核实这些边界；不支持只收窄工具的分析能力，仍需保留原始错误、源码/编译制品与运行证据，不能将空报告当成算子无问题。

验证解析结果需用真实接口 dtype、原始元数据、字节长度及已知输入交叉核对；验证分析报告需核对首次错误时间、制品版本、设备和本次日志身份。本卡未运行 msaicerr 或安装第三方依赖，只保存固定资料能够支持的行为及其判读限制。
