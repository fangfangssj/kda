---
type: "Runbook"
title: "507035 向量核异常的 plog errcode 定位法"
description: "aclrtSynchronizeStream 报 507035 时到运行用户 ~/ascend/log/debug/plog 查 errcode 与 core_id；errcode 82=MTE2 写 UB 越界、263 叠加=标量/GM 地址越界，AIC_INFO 段含 tiling 展开与故障 kernel 实例名。"
tags: [task.debugging, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 507035 向量核异常的 plog errcode 定位法

## 现象

aclnn ST 用例在 `aclrtSynchronizeStream(stream)` 处返回 `ACL error 507035`，测试侧只打印这一行错误码（`test_aclnn.cpp` 的 CHECK_ACL 宏输出），完全看不到设备侧具体原因。本例发生在 softplus_v2_grad RANK_4 kernel 首次联调（9 条 `[1,1024]⊗[256,1024]` 广播用例全挂），后续多次复用该定位法区分了不同类别的设备侧越界（tile 超 UB 的 host 切分轴坐标系错位、1D 拷贝循环的 GM 源地址越界）。

## 适用场景

- 工程形态：aclnn 直连 ST / 任何触发 kernel 设备侧异常的运行（编译能过、运行时向量核异常）。
- API 路径：ACL runtime 错误码（`rt_error_codes.h`）+ plog 设备错误日志（`~/ascend/log/{debug,run}/plog/`，运行用户家目录）。
- 验证过的平台与 CANN 版本：ascend950（dav-3510）+ cann-9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（与具体算子无关的设备侧错误定位方法）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：编译期错误（不走 plog）；host 侧纯软件错误（看 stderr/coredump 更直接）。

## 判别方法

**第 0 步——确认错误码语义**（本例）：

```bash
grep -rn "507035" <CANN安装根>/include/acl/error_codes/rt_error_codes.h
# → #define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      507035 // vector core exception
```

**第 1 步——定位最新 plog**：

```bash
ls -t ~/ascend/log/debug/plog | head    # 另有 ~/ascend/log/run/plog
```

plog 文件名形如 `plog-<pid>_<timestamp>.log`，按时间取最新（每次复现对应新文件）。

**第 2 步——提取 errcode 与故障核**：

```bash
grep -m5 "extend info: errcode" <最新plog>          # errcode 语义串
grep -m5 "AddExceptionRegInfo\|fault kernel_name" <最新plog>   # 故障 core_id / kernel 实例
grep -m3 "AIC_INFO" <最新plog>                       # tiling 数据展开 + blockDim
```

**观测到的 errcode 语义**（plog 原文摘录）：

```text
The extend info: errcode:(82, 341) errorStr: The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds.The address for VEC to access UB is out of bounds. subErrType: 0x4.
...
there is an aivec error exception, core id is 42, error code = 82, 341, dump info: pc start: 0x..., mte error info: 0x28900, vec error info: 0x...
Vector Core kernel execution failed, retCode=0x31.
GetArgsInfo:[AIC_INFO] args(0 to 19) after execute:0x12000019e000, ... tilingKey = 0, print 3 Times totalLen=(51*8), argsSize=408, schemMode=0, blockDim=18
fault kernel_name=SoftplusV2Grad_42334230bb5c946471377d5f5e580514_0
```

```text
The extend info: errcode:(263, 82) errorStr: The address for scalar to use is unaligned or out of bounds The GM address exceeds 48 bits, or the on-chip buffer address exceeds the size of the buffer.The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds.
```

（另有单独 `errcode:(82)`：仅 "The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds"，无 VEC 访问 UB 子码。）

**解读**：

- **82 = MTE2（GM→UB 搬运）写 UB 越界** → 优先核算 CopyIn 的 tile 字节数 vs `per_buf_bytes`（本轨迹中该信号最终指向 host 切分轴坐标系错位，而非 NDDMA stride 约束，见 [split_axis_padandsqueeze_507035](../compilation/split_axis_padandsqueeze_507035.md)）。
- **(82, 341) 叠加** "VEC to access UB out of bounds"：向量单元访问 UB 也越界（越界 tile 往往同时触发两段）。
- **263 叠加** = 标量/GM 地址越界或超 48bit——多为**源地址**算错（本轨迹中出现在"以 1D 拷贝循环替代 NDDMA 随路广播"的版本：把含广播维的内层当连续块读，越过 tensor 实际数据）。
- **AIC_INFO args 段**是 tiling 数据逐 int64 展开（`argsSize=408` = 51×8），带 `blockDim`、`tilingKey`；`fault kernel_name` 是带 hash 的 dtype 组合实例名，可用于核对是哪个 binary 出错。
- `AddExceptionRegInfo` 的 core_id 列表与 `blockDim` 对比，可判断是否有 BlockDim 之外的核被执行（另卡）。

## 设计决策

- 先查 errcode 再动代码：本轨迹中两条不同根因（tile 超 UB / 1D 拷贝源地址越界）都靠 errcode 语义先行分诊，避免了在错误假设下盲改。

## 验证信号

本定位法在该算子开发会话中的实测效果：

- 507035 = `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（rt_error_codes.h 原文 grep 命中）（来源平台：950；证据等级：tool_observed）
- errcode (82,341)/(82)/(263,82) 三种语义串均在 plog 原文观测到，且各自对应后续定位出的不同缺陷类别（来源平台：950；证据等级：tool_observed）
- AIC_INFO `blockDim=18` + core_id=42-44 对比 → 发现 BlockDim 外核执行问题（来源平台：950；证据等级：tool_observed）

## 失效边界

- plog 目录 `~/ascend/log/` 按运行用户家目录分布，其他部署可能不同；未验证其他 CANN 版本的 errcode 编码稳定性。
- errcode 语义串来自运行时翻译，个别子码（341）的精确硬件含义未查到官方对照，解读以 errorStr 文本为准。

**未决项**：

- 子码 341 的精确语义（仅从 errorStr 推断为 VEC 访问 UB 越界）未查到官方文档对照，可接续查证。
## 坏实践

只看测试侧 `ACL error 507035` 一行就开始改 kernel——507035 是"向量核异常"总称，不区分 UB 越界/GM 越界/非法指令；不查 plog 就无法分诊。

## 误诊

无（本卡为定位方法；注意 plog 中 errcode 的"81/82 等编码"易被误读为十进制错误码列表，实为 `(主码, 子码)` 二元组）。
