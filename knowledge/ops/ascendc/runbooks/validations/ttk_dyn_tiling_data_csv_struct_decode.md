---
type: "Runbook"
title: "TTK tiling 元组按 C 结构体布局解码"
description: "从 TTK result.csv 的 dyn_tiling_data 元组人工解码 tiling 字段会因 int64 对齐 padding 与截断拷贝出错（axisStride 误读为 1），须取完整 CSV 字段并按 C struct 布局 unpack，再做消费字节与语义双自洽校验。复现增量：dyn_tiling_key 列判读——host 单入口（无 per-key SEL）算子上该列恒为注册口径 0（cosmetic），不代表 kernel 实际分支；判分支生效须按 TilingData 声明位置解析 dyn_tiling_data 内的 tilingKey 字段（如末 int32），并以分支专属属性×精度作交叉证据。"
tags: [task.debugging, topic.tiling, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# TTK tiling 元组按 C 结构体布局解码

## 现象

排查一个 fp16 softmax 白盒失败用例时，需要解读 TTK result.csv 里 `dyn_tiling_data` 字段记录的 tiling 结构体元组（形如 `(np.int32(2), np.int32(0), np.int32(2293200), ...)`，共 56 个 int32，对应 224 字节）。第一次解码把 grep 预览输出的单行内容手工粘贴成字符串再正则提取，得到 `axisStride=(1, 1, 0)`——对连续 AR 布局（A 轴 2,293,200 行 × R 轴 11 元素）来说 axisStride[0] 应等于 R=11，于是怀疑 host 侧 `ComputeContiguousStrides` 算错、kernel 在读错位的数据，调查差点被带偏去改 tiling。随后改用 `csv.DictReader` 从完整 CSV 字段重新提取原始元组、并按结构体 C 布局 unpack，得到正确的 `axisStride=(11, 1, 0)`——tiling 无误，调查转回 golden 口径差异。

## 适用场景

- 工程形态：TTK 白盒/aclnn 评测产物的离线解读（result.csv 的 `dyn_tiling_data` / `bin_tiling_data` 字段）；AscendC 自定义算子 tiling 排查。
- API 路径：TTK result.csv tiling 字段；算子 `op_host/tiling_data.h` 的 tiling 结构体；Python `struct.unpack_from`。
- 验证过的平台与 CANN 版本：Ascend950PR_9579，CANN 9.2.0，TTK v3.0.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——方法对任何"CSV 记录的 tiling 二进制元组"通用；具体字段集以各算子 tiling_data.h 为准。
- 路线优先级：recommended_fastpath（正确解码是 tiling 排查的第一步，成本低）。
- 明确不适用 / 未验证场景：非本算子的 tiling struct（字段集不同，须以各自 tiling 头文件布局为准）；`npu` 侧 dump 的二进制 tiling（非 CSV 元组形态）。

## 判别方法

tiling 元组是 C 结构体的 int32 小端序列化，**字段间有对齐 padding**：`axisNum`（int32）后有 4 字节 padding 把后续 int64 数组对齐到 8 字节边界；每个 int64 字段占两个 int32（低位在前、高位 0 交错出现）；`aBigCoreCnt/usedCoreNum/aSplitAxisIdx/rSplitAxisIdx` 为连续 int32。本例布局（224 字节 = 56×int32，与 tiling_data.h 一致）：

```
axisNum(int32) + pad(int32)
axisShape[3](int64)  axisStride[3](int64)
aLoopCntTotal, aSplitChunkCnt, aBigCoreLoopCnt, aSmallCoreLoopCnt (int64×4)
aBigCoreCnt, usedCoreNum, aSplitAxisIdx, rSplitAxisIdx (int32×4)
aUbFactor, aUbFactorAlign, rUbFactor, rUbFactorAlign (int64×4)
innerAProd, innerAProdAlign, innerRProd, innerRProdAlign (int64×4)
rLoopCntTotal (int64)
preReduceUbSize, expBufUbSize, postReduceUbSize, tmpBufUbSize, cacheBufUbSize, rGroupCnt (int64×6)
```

正确解码脚本（从 result.csv 完整字段提取，verbatim）：

```python
import csv, re, struct

with open(".../result.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("testcase_name", "").strip() == "case00076":
            td = row.get("dyn_tiling_data", "")
            vals = [int(x) for x in re.findall(r'np\.int32\((-?\d+)\)', td)]
            raw_bytes = struct.pack(f'<{len(vals)}i', *vals)

            off = 0
            axisNum = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            pad = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            axisShape = struct.unpack_from('<3q', raw_bytes, off); off += 24
            axisStride = struct.unpack_from('<3q', raw_bytes, off); off += 24
            aLoopCntTotal = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            aSplitChunkCnt = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            aBigCoreLoopCnt = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            aSmallCoreLoopCnt = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            aBigCoreCnt = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            usedCoreNum = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            aSplitAxisIdx = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            rSplitAxisIdx = struct.unpack_from('<i', raw_bytes, off)[0]; off += 4
            aUbFactor = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            aUbFactorAlign = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            rUbFactor = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            rUbFactorAlign = struct.unpack_from('<q', raw_bytes, off)[0]; off += 8
            # …后续 int64 字段按 tiling_data.h 逐个 unpack…
            print(axisNum, axisShape, axisStride, ...)
```

解码后做两个自洽校验再下结论：① 消费字节数 = 总字节数（本例 224/224）；② 字段语义（axisShape[0]=outerA、axisShape[1]=R、连续布局下 axisStride[0]=R、rLoopCntTotal 与命中的 kernel 路径一致）。

## 设计决策

- 用 `csv.DictReader` 取完整 `dyn_tiling_data` 字段而不是 grep/复制粘贴：CSV 单元格内含逗号与长元组，终端预览会截断、手工转录会丢字符（本例把 `11` 抄成 `1` 直接改写了 axisStride[0] 的值）。
- 用 `struct.unpack_from` 按 C 布局显式走 offset 而不是肉眼数 int32：int32/int64 混排 + 对齐 padding 下肉眼极易错位；padding 的存在还可由 `vals[1]==0`（axisNum 后的填充位）快速印证。
- 解码结论必须与 kernel 实测行为交叉验证（本例 C++ ST 全过、输出 mere=4.09e-8，若 stride 真错不可能只剩 1 个用例失败）——信号冲突时优先怀疑解码而不是 tiling。

## 系统信息

- 环境：Ascend950PR_9579，CANN 9.2.0，TTK v3.0.0；Python 3.13（仅解码用 struct/csv 标准库）。
- 相关文件：TTK `result.csv`（`dyn_tiling_data`/`bin_tiling_data`/`dyn_tiling_key` 列）；算子 `op_host/tiling_data.h`（结构体字段集与类型的唯一事实源）；`op_host/arch35/*_tiling_arch35.cpp`（`ComputeContiguousStrides` 被调出处，用于语义校验）。

## 预防

- 从 TTK 产物解读 tiling 一律走"完整 CSV 字段 + C 布局 unpack + 消费字节/语义双校验"，禁止从 grep/终端预览截断行手工转录。

## 验证信号

本卡来源说明：一次 fp16 白盒失败用例排查中，对同一 result.csv 的 tiling 元组做了两次解码并对照 kernel 行为。逐条实测信号：

- 误解码：从 grep 单行输出手工提取的 56 个 int32 解出 `axisStride=(1, 1, 0)`（正确值应为 11），触发"stride 计算 bug"怀疑（来源平台：950；证据等级：tool_observed）。
- 正确解码：`csv.DictReader` 完整字段 + struct 布局解出 `axisStride=(11, 1, 0)`，同时 axisShape=(2293200, 11, 1)、aUbFactor=912/aUbFactorAlign=912、rUbFactor=11/rUbFactorAlign=16、rLoopCntTotal=1、usedCoreNum=56，与 fast path 路由和 C++ ST 通过事实自洽；消费 224/224 字节（来源平台：950；证据等级：tool_observed）。
- 交叉验证：同 shape 的 C++ ST 复现 500 种子零失败——若 stride 真错，输出不可能仅差 1-ULP 量级，确认 tiling 无误（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 布局字段集来自本算子的 tiling_data.h；其它算子字段不同（长度、int32/int64 混排、数组维度），必须以各自 tiling 头文件为准——方法（完整字段 + C 布局 unpack + 消费字节校验）通用，字段表不可照抄。
- 仅验证 CSV 元组形态（`np.int32(...)` 列表）；其它序列化形态（纯二进制 dump）未验证。
- 若算子 tiling struct 使用 `#pragma pack`/显式 packed，则无对齐 padding，须按 packed 布局调整 offset（本例为默认对齐）。
- 其他 TTK 版本的字段序列化格式未验证（推断可迁移）。

**未决项**：

- 无（方法卡；本例误解码已当场纠正，未影响最终结论）。
## 坏实践

- 从 grep/终端预览的截断行复制 tiling_data 字符串解码：长单元格被截断或转录丢字符，数值错位后结论完全反转（本例 11→1）。
- 不做消费字节/语义自洽校验就宣布"tiling 算错了"：缺少交叉验证的解码结论会把调查带偏（本例差点去改正确的 `ComputeContiguousStrides`）。
- 肉眼按 int32 逐个数 offset 解 int32/int64 混排结构：对齐 padding 极易数错。

## 误诊

- 首解 axisStride=(1,1,0) 后的推断"若 stride[0]=1 则 kernel 读错数据、可 kernel 大部分输出正确，矛盾"——矛盾本身即是解码有误的信号，正确做法是先复核解码而不是构造解释。

## 增量（dyn_tiling_key 列恒 0 的判读：分支调度看 tiling 数据字段）

同一 result.csv 的另一判读坑（三个分片独立撞到，RoPE 类多分支单入口算子，Ascend950PR_9579 + CANN 9.2.0）：**`dyn_tiling_key` 列在"host 单入口（单一 TilingFunc / 未声明 per-key ASCENDC_TPL）+ kernel 内按 TilingData.tilingKey 字段分发"的自定义算子上，对全部正常用例恒显示 `0 (0x0000000000000000)`**，TTK OnProfilingPrint 的 `Tiling Key: 0` 同样——按字段面值判读会得出"分支从未命中、白盒全测错对象"的假 critical 结论，而实际分发正常。

**机制**：host 的 `SetTilingKey` 对齐唯一的 `opc _0` 入口（无 per-key SEL 可选），注册 key 恒 0；真正的分支选择由 host TilingFunc 把 tilingKey 作为普通字段写进 TilingData POD，kernel `GET_TILING_DATA` 后读 `td->tilingKey` 分发。`dyn_tiling_key` 是 host 注册口径（结果汇报层的取值限制，产生层级未在 TTK/SEL 源码定位），`dyn_tiling_data` 元组中按结构体声明位置的字段才是 kernel 实际消费的分支键——与本卡主文"按 C 结构体布局解码"同一方法，只是要解的是 tilingKey 字段而非 stride。

**探针**（对 result.csv 全量统计 tilingKey 字段分布，本例 tilingKey 声明在 18 字段 POD 末位）：

```python
import csv, re
from collections import Counter
rows = list(csv.DictReader(open(".../tests/whitebox/ttk/result.csv")))
kc = Counter()
for r in rows:
    td = r.get("dyn_tiling_data", "")
    if "np.int32" not in td:
        kc["<none>"] += 1; continue
    vals = re.findall(r"np\.int32\((-?\d+)\)", td)
    kc[vals[-1] if vals else "?"] += 1   # 末字段=tilingKey（按算子 tiling 头的字段序定位）
print("=== tilingKey field distribution ===")
print(dict(kc))
```

**判读规则**：末字段（或按声明序定位的字段）分布与预期分支键一致 → 分支调度真实生效，`dyn_tiling_key=0` 是注册口径噪音；字段值 ≠ 预期键（或全 0）→ 才是真实的分发问题。交叉证据（按证据强度排序）：① 该分支语义独有属性（如 rotaryMode=interleaved）的用例数值 PASS——不同分支语义对同输入产生不同输出，误分发必逐点失配；② kernel 入口 dispatch 代码核对；③ 宿主侧 TilingUT 独立断言 ChooseTilingKey 输出。

**误读实例**：首次解析把元组**首元素**当 tilingKey，得到 `key=0:2 / 1:7 / 2:2 / 3:108 / 7:72 / 8:37` 的"分布"一度怀疑分发混乱——对照单例 POD（`[8,57,431,64,...]` 首元素 8 是 batch B）后纠正为末字段解析；统计时还需先过滤编译失败行（该列为 `DYN_COMPILE_FAILURE`/`OPTILING_FAILURE` 等错误字符串）。

验证信号（复现增量，同算子三个分支的白盒 TTK 执行会话）：

- branch-2：`dyn_tiling_key` 分布 `{'0': 229, 'OPTILING_FAILURE': 7, 'DYN_COMPILE_FAILURE': 1, 'PROFILE_CRASH': 1}`；`dyn_tiling_data` 末字段分布 `{'2': 229, '?': 9}`（9 条 = 8 error-reject + 1 crash，无 tiling 数据）；同用例 OnProfilingPrint `Tiling Key: 0` 与 `Tiling Data Parsed Tuple` 末位 `np.int32(2)` 并存（来源平台：950；证据等级：tool_observed）。
- branch-4：228/238 例末字段 tilingKey=4、10 例无 tiling 数据；case00001 的 18 字段 POD `[8,57,431,64,32,196536,3584,3000,96,32,64,8,57,32,0,64,2,4]`（首元素 8=B、末元素 4=tilingKey）（来源平台：950；证据等级：tool_observed）。
- branch-5：228 例非 error 用例 `dyn_tiling_key` 全 0，同批 136 例数值 PASS 且全部带 interleaved 专属属性（half 分支误接管必失配）→ kernel 内 tilingKey==5 分发生效（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- "末字段=tilingKey"依赖算子自己的 TilingData 声明序；换算子须先对照其 tiling 头定位字段下标（与本卡主文"字段集以各自 tiling_data.h 为准"同理）。
- 仅覆盖 host 单入口 + kernel 内分发的形态；已声明 per-key SEL/多入口的算子 `dyn_tiling_key` 可能携带真实 key（未验证，届时两口径应一致，不一致反而是问题）。
- 无 tiling 数据的用例（error-reject、crash）无法用此探针判定；`dyn_tiling_data` 为空/None 须先过滤。
- 字段恒 0 的产生机制（TTK 结果汇报层 vs SEL 查询层哪一层丢失 tilingKey）未定位到源码层。

坏实践（复现增量）：

- 把 `dyn_tiling_key=0` 直接读作"分支 0 被调度/分发失效"，据此作废整轮白盒结果或去改 host 注册。
- 把 `dyn_tiling_data` 元组首元素当 tilingKey 统计——首元素是第一个 tiling 字段（如 B），会得到一组看似有意义的错误分布。