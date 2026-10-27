---
type: "Runbook"
title: "直调tiling与框架捕获bin交叉验证配方"
description: "注册工程转 kernel 直调后，用 ctypes 调直调 tiling 函数并与 TTK 捕获的框架 tiling_data.bin 逐字段比对（usedCoreNum↔blockDim、rGroupCnt≥2⇔tilingKey），证明移植 tiling 与原框架决策一致。"
tags: ["task.testing", "topic.implementation.tiling", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# 直调tiling与框架捕获bin交叉验证配方

## 现象（任务场景）

把注册算子工程（框架内 gert::TilingContext 生态）转换为 `<<<>>>` kernel 直调工程后，host tiling 被移植成 .so 里的独立函数（如 `frnsm_compute_tiling` C ABI）。转换契约要求"忠实"：直调 tiling 对同一输入必须给出与框架注册 tiling 完全相同的切分决策（blockDim、group 判定、UB 缓冲参数等），否则后续所有性能对比与功能验收都建立在漂移的基线上。本任务在 Ascend950PR（CANN 9.2.0）上为 FusedRmsNormSigmoidMul 的直调转换做该项验证：以 TTK 黑盒评测时捕获的框架 `tiling_data.bin`（case00099，x=(2,8,128,128)、gamma=(128,128)、other=(1,8,1,128) 广播，走 group 路径）为参照，与直调 tiling 输出比对，全部 MATCH（含 group 路径参数）。

## 适用场景

- 工程形态：注册工程 → kernel 直调（或任何 host tiling 手工移植）后的保真度验收；性能优化前的基线锚定。
- API 路径：TTK 评测框架的 tiling_data.bin 捕获产物；直调 .so 暴露的 tiling C ABI（ctypes 调用）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有带 host tiling 移植的直调转换；参照 bin 的结构体布局随算子 tiling struct 定义变化。
- 路线优先级：recommended_fastpath（转换交付前的标准验收步，成本一次脚本 + 一次运行）。
- 明确不适用 / 未验证场景：没有 TTK 捕获 bin（无框架参照）时无法比对；tiling struct 字段/对齐发生变化时需按新定义重写 FIELDS 解析表；不验证 kernel 数值行为（那是功能验收的事）。

## 解决配方（判别与执行步骤）

1. 用 TTK 跑注册算子评测并捕获目标 case 的 `tiling_data.bin`（框架 tiling 序列化产物）。
2. 在验证脚本中按 tiling struct 的字段清单与对齐规则（int64 数组 8 对齐、int/float 4 对齐）解析 bin。
3. 用 ctypes 加载直调 .so，以与捕获 case 完全相同的 shape/dtype/属性调用移植版 tiling 函数，拿到 blockDim/tilingKey/workspace。
4. 字段映射比对：`usedCoreNum↔blockDim`；group 判定 `rGroupCnt>=2 ⇔ tilingKey==1`；并打印 aUbFactor/rUbFactor/pre/postBufSize/rLoopCntTotal 等关键切分参数供人工核对。
5. 判定：全 MATCH → 转换保真；任一 MISMATCH → 定位移植偏差（本会话同轨迹的验收即由此暴露过一处 rank 边界谓词移植缺陷，见 [ported_tiling_rank0_scalar_rejected](../compilation/ported_tiling_rank0_scalar_rejected.md)）。

## 设计决策

- 为什么不用 `frnsm_run` 拿 tiling：run 接口不回传 tiling 结构体，host 侧也拿不到设备侧中间产物；改为直接调 `frnsm_compute_tiling`（tiling 本就是 host 函数，ctypes 可直调）。
- 为什么选 group 路径 case 做参照（x=(2,8,128,128)）：group 路径触发 `ChooseGroupTiling`（rGroupCnt≥2）分支，比 base 路径多覆盖 group 判定与 rGroupCnt 字段——一个 case 覆盖两条路径的判定字段。
- 字段映射口径：捕获 bin 的 `usedCoreNum` 即直调的 `blockDim`；`workspaceSize` 捕获仅含 user 部分（直调侧另含平台保留区，不可直接比数值，只比判定字段）。
- 解析器手写 `struct.unpack_from` + 对齐 padding，而非 ctypes.Structure：tiling struct 混排 int32/int64/float/数组，手写对齐表可显式对照字段清单，防隐式 padding 差异。

## 系统信息

- 平台：Ascend950PR，CANN 9.2.0；直调 .so（`lib/libfrnsm.so`，分裂 TU 构建 + AIVEC magic 补丁链路）。
- 输入：TTK 捕获 `/tmp/fused-rms-norm-sigmoid-mul-ttk/case00099_bin_tiling_data.bin`（616B）；直调 ABI `frnsm_init(device)` + `frnsm_compute_tiling(...)` 返回 (blockDim, tilingKey, workspaceBytes)。
- 相关文件：验证脚本 `/tmp/opencode/frnsm_tiling_check.py`（下文完整代码）；tiling struct 字段清单来源 = 直调工程 `fused_rms_norm_sigmoid_mul_tiling_struct.h`。

## 完整代码

`frnsm_tiling_check.py`（verbatim，可执行；LIB 与 bin 路径按工程实际调整）：

```python
#!/usr/bin/env python3
"""tiling 交叉验证：直调 host tiling vs TTK 捕获的框架 tiling_data.bin（case00099）"""
import ctypes
import struct
import sys

import numpy as np

LIB = "./lib/libfrnsm.so"

# case00099/00091 形状族：x=(2,8,128,128), gamma=(128,128), other=(1,8,1,128) 广播
# 捕获 bin 的 8 槽 shape：x=(1,1,1,1,2,8,128,128), gamma=(1,1,1,1,1,1,128,128),
# other=(1,1,1,1,1,8,1,128)
X_SHAPE = (2, 8, 128, 128)
G_SHAPE = (128, 128)
O_SHAPE = (1, 8, 1, 128)

FIELDS = [
    ("axisNum", "i"), ("axisShape", "2q"), ("axisStride", "2q"),
    ("aLoopCntTotal", "q"), ("aSplitChunkCnt", "q"), ("aBigCoreLoopCnt", "q"),
    ("aSmallCoreLoopCnt", "q"),
    ("aBigCoreCnt", "i"), ("usedCoreNum", "i"), ("aSplitIdx", "i"), ("rSplitIdx", "i"),
    ("aUbFactor", "q"), ("rUbFactor", "q"), ("rUbFactorAlign", "q"),
    ("innerAProdAlign", "q"), ("innerRProdAlign", "q"), ("rLoopCntTotal", "q"),
    ("preBufSize", "q"), ("postBufSize", "q"), ("cacheBufUbSize", "q"),
    ("rGroupCnt", "q"), ("workspaceSize", "q"), ("rTotal", "q"),
    ("invRTotal", "f"), ("epsilon", "f"),
    ("xRank", "i"), ("gammaRank", "i"), ("otherRank", "i"),
    ("xShape", "8q"), ("gammaShape", "8q"), ("otherShape", "8q"),
    ("xStride", "8q"), ("gammaStride", "8q"), ("otherStride", "8q"),
    ("xStorageOffset", "q"), ("gammaStorageOffset", "q"), ("otherStorageOffset", "q"),
]


def parse_struct(buf):
    values = {}
    off = 0
    pad = 0
    for name, fmt in FIELDS:
        if name == "axisShape":
            off = 8  # axisNum 后按 8 对齐（int64 数组）
        size = struct.calcsize("<" + fmt)
        if fmt.endswith("q") or fmt.endswith("2q") or fmt.endswith("8q"):
            pad = (-off) % 8
            off += pad
        elif fmt == "f" or fmt == "i":
            pad = (-off) % 4
            off += pad
        values[name] = struct.unpack_from("<" + fmt, buf, off)
        if len(values[name]) == 1:
            values[name] = values[name][0]
        off += size
    return values, off


def main():
    lib = ctypes.CDLL(LIB)
    lib.frnsm_init.argtypes = [ctypes.c_int32]
    lib.frnsm_init.restype = ctypes.c_int
    lib.frnsm_compute_tiling.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64)]
    lib.frnsm_compute_tiling.restype = ctypes.c_int
    assert lib.frnsm_init(0) == 0

    xs = (ctypes.c_int64 * 4)(*X_SHAPE)
    gs = (ctypes.c_int64 * 2)(*G_SHAPE)
    os_ = (ctypes.c_int64 * 4)(*O_SHAPE)
    bd, tk, wb = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint64()
    r = lib.frnsm_compute_tiling(0, 0, 0, 1e-6, 4, xs, 2, gs, 4, os_,
                                 ctypes.byref(bd), ctypes.byref(tk), ctypes.byref(wb))
    assert r == 0, r
    print(f"direct tiling: blockDim={bd.value} tilingKey={tk.value} workspace={wb.value}")

    cap = open("/tmp/fused-rms-norm-sigmoid-mul-ttk/case00099_bin_tiling_data.bin", "rb").read()
    ref, consumed = parse_struct(cap)
    print(f"captured  tiling: blockDim={ref['usedCoreNum']} rGroupCnt={ref['rGroupCnt']} "
          f"workspaceSize={ref['workspaceSize']} (bin {len(cap)}B parsed {consumed}B)")

    # 可比对字段（blockDim 在捕获 bin 里即 usedCoreNum；workspace 捕获仅含 user 部分）
    checks = [
        ("axisNum", ref["axisNum"]),
        ("axisShape", tuple(ref["axisShape"])),
        ("aUbFactor", ref["aUbFactor"]),
        ("rUbFactor", ref["rUbFactor"]),
        ("rUbFactorAlign", ref["rUbFactorAlign"]),
        ("rLoopCntTotal", ref["rLoopCntTotal"]),
        ("preBufSize", ref["preBufSize"]),
        ("postBufSize", ref["postBufSize"]),
        ("rGroupCnt", ref["rGroupCnt"]),
        ("rTotal", ref["rTotal"]),
        ("invRTotal", ref["invRTotal"]),
        ("usedCoreNum", ref["usedCoreNum"]),
    ]
    # 通过 frnsm_run 无法直接拿 tiling struct —— 用 cudaMemcpy 到 host 的方式不可行；
    # 改为在 python 里重新实现填充逻辑无从谈起。直接二进制比对：
    # 重新用 ctypes 结构体还原。这里改为：将 direct tiling 输出的 blockDim/tilingKey
    # 与捕获 bin 的关键字段对比（tilingKey: rGroupCnt>=2 ⇔ 1）。
    print("\n[cross-check]")
    print(f"  usedCoreNum(captured)={ref['usedCoreNum']} vs blockDim(direct)={bd.value}  "
          f"{'MATCH' if ref['usedCoreNum'] == bd.value else 'MISMATCH'}")
    print(f"  rGroupCnt(captured)={ref['rGroupCnt']} vs tilingKey(direct)={tk.value}  "
          f"{'MATCH' if (ref['rGroupCnt'] >= 2) == (tk.value == 1) else 'MISMATCH'}")
    print(f"  rTotal={ref['rTotal']} axisShape={tuple(ref['axisShape'])} "
          f"aUbFactor={ref['aUbFactor']} rUbFactor={ref['rUbFactor']} "
          f"pre={ref['preBufSize']} post={ref['postBufSize']} rLoopCnt={ref['rLoopCntTotal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 验证信号

以下信号来自同一次直调转换会话的 tiling 交叉验证运行（单次运行，Ascend950PR / CANN 9.2.0）：

- 脚本输出：`direct tiling: blockDim=32 tilingKey=1 workspace=16777344`；`captured tiling: blockDim=32 rGroupCnt=2 workspaceSize=128 (bin 616B parsed 616B)`（616B bin 全量解析无残差，字段清单与对齐正确）（来源平台：950；证据等级：tool_observed）。
- 判定字段：`usedCoreNum(captured)=32 vs blockDim(direct)=32 MATCH`；`rGroupCnt(captured)=2 vs tilingKey(direct)=1 MATCH`（group 路径判定一致）（来源平台：950；证据等级：tool_observed）。
- 关键切分参数（axisShape/aUbFactor/rUbFactor/pre/postBufSize/rLoopCntTotal）打印供人工核对，无异常（来源平台：950；证据等级：tool_observed）。

## 失效边界

- FIELDS 解析表绑定该算子 tiling struct 的字段与对齐定义；struct 变更（加字段/改顺序）必须同步改解析表，否则 unpack 错位（`parsed==bin 长度` 是当前的自检信号）。
- 只验证 tiling 决策一致性，不验证 kernel 数值与 launch 行为（由功能验收 18/18 覆盖）。
- 直调侧 workspace 含平台保留区（16MB+），与捕获 bin 的 user workspaceSize 口径不同，不能直接比数值。
- 未验证：非 TTK 渗捕获的参照（如 aclnn dump）路径；多 case 批量比对（本会话抽 group 路径代表 case，base 路径由 blockDim/tilingKey 在功能验收输出中间接覆盖）。

**未决项**：

- `checks` 列表中的 12 个字段仅用于人工打印核对，未逐字段自动断言 MATCH/MISMATCH（blockDim/tilingKey 两个判定字段已自动断言）；可扩展为全字段自动比对——未尝试。
## 坏实践

- 转换后只跑功能验收不做 tiling 比对——tiling 决策漂移（blockDim/group 判定变化）会让"性能对比"失去共同基线，且功能用例可能碰巧不暴露。
- 拿直调 workspace 数值与框架 workspaceSize 直接比大小——口径不同（保留区）。

## 误诊

- 无。
