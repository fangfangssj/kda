---
type: "Runbook"
title: "host tiling改动先Python仿真对账bd/tk"
description: "修改 AscendC host tiling 逻辑（au 求解、UB 方程、门控）后，先在 Python 中逐字复刻 C++ 求解逻辑，把冻结验收集逐 case 的 blockDim/tilingKey 与基线 manifest 对账，再构建上板——构建前即可捕获门控/路由回归与自证输入错误；构建后再用 ctypes 直调真库 tiling 入口复核。"
tags: ["task.testing", "topic.tiling"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# host tiling改动先Python仿真对账bd/tk

## 现象

在 Ascend950PR（CANN 9.2.0）上优化融合算子 FusedRmsNormSigmoidMul 的 host tiling：候选方案要改 au 求解（新增 `SolveFastAu`/`SolveBalancedAu` 均衡扫描）、UB 合法性方程和快路门控（`arContigFast`），而验收契约要求"除目标类 case 外，18 条冻结 L0 用例的 blockDim/tilingKey 不得漂移"（group 路径触发域、广播/tiny case 的路由都必须与基线逐字节一致）。如果直接改 C++ → 构建（单次 split 构建数分钟）→ 上板跑验收才发现门控写错或某 case 路由漂移，迭代成本高；而且 tiling 是纯整数求解逻辑，完全可以在构建前用 Python 精确复刻验证。实施时在 `/tmp/opencode/sim_tiling.py` 逐字复刻求解逻辑并跑 18 条 case 对账，第一次就抓到了自己仿真输入写错的问题（目标 case 的 gamma shape 误写为 `(6,128)` 导致 A=128/R=768，断言失败；修正为 `(128,)` 后 A=768/R=128 才是真实语义），证明了该探针的有效性——它连"测试输入错误"都能拦截。

## 适用场景

- 工程形态：AscendC 算子 host tiling 修改（直调 `ComputeTilingImpl` 或注册工程 `TilingFuncPublic`）的构建前验证；任何"整数求解 + 冻结用例路由一致性"的 tiling 契约场景。
- API 路径：Python 复刻 `PubCeil/PubUp/PubDown`、`ChooseGroupTiling`、au/UB 方程与门控；构建后用 ctypes 直调 `.so` 的 tiling 入口（直调工程的 `frnsm_compute_tiling`）做真库身份复核。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（方法对整数求解类 tiling 通用；case 表与 tiling 逻辑按目标算子替换）。
- 路线优先级：recommended_fastpath（构建前的低成本门禁）。
- 明确不适用 / 未验证场景：不适用于含浮点收敛/迭代求解的 tiling（逐字复刻难度大，未验证）；不能替代上板功能验收（只验证 tiling 决策，不验证 kernel 执行）；注册工程无直调 `.so` tiling 入口时第二步探针形态需换用 TTK/单测入口（未验证）。

## 判别方法

**探针 1（构建前 Python 仿真对账）**——把 C++ tiling 逻辑逐字翻译成 Python（对齐宏、group 决策、au 求解、门控），对冻结验收集逐 case 计算 `(blockDim, tilingKey, fast)` 并与基线 manifest 的期望值对账。完整可执行脚本（以本算子为例，case 表含 18 条 L0 的期望 bd/tk）：

```python
import math

def PubCeil(a,b): return a//b + (1 if a%b else 0)
def PubUp(a,b): return PubCeil(a,b)*b
def PubDown(a,b): return a//b*b

def P2CapacityLegal(pre,post,groups,aTotal):
    if pre<=0 or post<=0 or groups<=0 or aTotal<=0: return False
    rawP2 = (pre//4)//groups
    if rawP2<8: return False
    aUbFactorP2 = min(PubDown(rawP2,8), post//4, aTotal)
    return aUbFactorP2>0 and PubUp(aUbFactorP2,8)<=rawP2 and aUbFactorP2<=post//4

def WorkspaceBytesLegal(groups,aTotal):
    if groups<=0 or aTotal<=0: return False
    return True

def ChooseGroupTiling(aOuter,rOuter,coreNum,pre,post,aTotal):
    if aOuter<=0 or aOuter>coreNum//2 or rOuter<2: return None
    totalOuter=aOuter*rOuter
    perCore=PubCeil(totalOuter,coreNum)
    blocks0=PubCeil(totalOuter,perCore)
    blocksUp=PubUp(blocks0,aOuter)
    blocks = blocksUp if blocksUp<=coreNum else PubDown(blocks0,aOuter)
    if blocks<=0 or blocks>coreNum or blocks%aOuter!=0: return None
    groups=blocks//aOuter
    if groups<2 or groups>rOuter or not P2CapacityLegal(pre,post,groups,aTotal): return None
    return blocks,groups

def SolveFastAu(A,ra,coreNum,ubSize,auCap):
    elemBytes=2; minTileBytes=2048
    if A<1 or ra<1 or coreNum<1 or ubSize<1 or auCap<1: return 0
    gammaBytes=PubUp(ra*elemBytes,32)
    tb=lambda au: PubUp(au*ra*elemBytes,32)
    rb=lambda au: PubUp(au*4,32)
    upper2=upper1=0
    for au in range(1,auCap+1):
        if 5*tb(au)+gammaBytes+rb(au)<=ubSize: upper2=au
        if 3*tb(au)+gammaBytes+rb(au)<=ubSize: upper1=au
    upper = upper2 if upper2>=1 else upper1
    if upper<1: return 0
    cands=[]
    p=1
    while p<=upper:
        cands.append(p); p<<=1
    for m in range(16,upper+1,16):
        if m not in cands: cands.append(m)
    if upper not in cands: cands.append(upper)
    hasMinTile = any(c<=A and tb(c)>=minTileBytes for c in cands)
    bestAu=bestCrit=bestExact=bestUsed=0
    for au in cands:
        if au>A: continue
        if hasMinTile and tb(au)<minTileBytes: continue
        tiles=PubCeil(A,au); used=min(tiles,coreNum); bigLoop=PubCeil(tiles,used)
        crit=bigLoop*au; exact=1 if A%au==0 else 0
        better = (bestAu==0 or crit<bestCrit or
            (crit==bestCrit and (exact>bestExact or
             (exact==bestExact and (au>bestAu or (au==bestAu and used>bestUsed))))))
        if better:
            bestAu,bestCrit,bestExact,bestUsed=au,crit,exact,used
    return bestAu

def tiling(xShape,gammaShape,otherShape, coreNum=56, ubSize=253952, blockSize=32):
    # 左补 1 的 8 槽
    def slots(s):
        out=[1]*8
        for i,v in enumerate(s): out[8-len(s)+i]=v
        return out
    xs=slots(xShape); gs=slots(gammaShape); os_=slots(otherShape) if otherShape else [1]*8
    gammaRank=len(gammaShape)
    gslot=8-gammaRank
    A=1
    for i in range(gslot): A*=xs[i]
    R=1
    for i in range(gslot,8):
        if xs[i]!=gs[i]: return None
        R*=gs[i]
    blockElems=blockSize//2
    tileLimit=16384//4
    ru=min(R,tileLimit); ra=PubUp(ru,blockElems)
    au=max(1,tileLimit//ra)
    post=PubUp(au*4,blockSize); pre=PubUp(au*ra*4,blockSize)
    if 3*pre+4*post+16384>ubSize: return None
    ao=PubCeil(A,au); ro=PubCeil(R,ru); usedC=min(ao,coreNum); groups=1; ws=0
    g=ChooseGroupTiling(ao,ro,coreNum,pre,post,A)
    group=False
    if g: blocks,groups=g; usedC=blocks; group=True
    fast=False
    if not group:
        noBroadcast=all(os_[i]==xs[i] for i in range(8))
        singleChunk=(ro==1)
        if noBroadcast and singleChunk:
            auF=SolveFastAu(A,ra,coreNum,ubSize,au)
            if auF>0:
                fast=True; au=auF; ao=PubCeil(A,au); usedC=min(ao,coreNum)
                pre=PubUp(au*ra*2,blockSize); post=PubUp(ra*2,blockSize)
    return dict(A=A,R=R,au=au,ao=ao,usedC=usedC,blockDim=usedC,tilingKey=1 if group else 0,fast=fast,ra=ra,pre=pre,post=post)

cases = [
 ("L0_003",(5,2,9371233),(2,9371233),(1,2,9371233),55,1),
 ("L0_007",(2,2,2,2,2,2,411629),(2,2,2,2,2,2,411629),(),56,1),
 ("L0_008",(6,6,8,6,8,8,8,9),(6,6,8,6,8,8,8,9),(6,6,8,6,8,8,8,9),56,1),
 ("L0_009",(2,2,2,2,2,2,2,299256),(299256,),(1,2,1,2,1,2,1,299256),56,0),
 ("L0_010",(100,63,2,1,1,1,2,1352),(100,63,2,1,1,1,2,1352),(1,63,2,1,1,1,2,1352),56,1),
 ("L0_011",(2501,6610),(6610,),(),56,0),
 ("L0_012",(4,8,9,4,5,9,4,7),(4,7),(4,8,9,4,5,9,4,7),56,0),
 ("L0_015",(1,4096),(4096,),(1,4096),1,0),
 ("L0_016",(32,4096),(4096,),(32,4096),32,0),
 ("L0_017",(2,128,4096),(4096,),(1,128,4096),56,0),
 ("L0_018",(2,8,128,128),(128,128),(1,8,1,128),32,1),
 ("L0_019",(3,17),(17,),(1,17),1,0),
 ("L0_021",(2,128,4096),(4096,),(4096,),56,0),
 ("L0_022",(1,),(1,),(1,),1,0),
 ("L0_024",(1,1,1,1,1),(1,),(),1,0),
 ("L0_026",(1,1,1,1,1,1,1,1),(1,),(1,1,1,1,1,1,1,1),1,0),
 ("L0_028",(643,7,9027),(643,7,9027),(),56,1),
 ("L0_033",(11,4,7,10,8,12,11),(11,),(1,4,1,10,1,12,1),56,0),
]
print(f"{'case':10s} {'A':>8s} {'R':>9s} {'au':>5s} {'bd':>4s} {'key':>3s} fast  base_bd base_key")
for name,x,g,o,bd,key in cases:
    r=tiling(x,g,o)
    mark = "OK " if (r and r['blockDim']==bd and r['tilingKey']==key) else "DIFF"
    print(f"{name:10s} {r['A']:8d} {r['R']:9d} {r['au']:5d} {r['blockDim']:4d} {r['tilingKey']:3d} {str(r['fast']):5s} {bd:7d} {key:7d} {mark}")

# target case（注意 gamma shape 语义：gamma=(128,) → A=768, R=128）
r=tiling((128,6,128),(128,),(128,6,128))
print("\ntarget fused_128x6x128:", r)
assert r['au']==16 and r['usedC']==48 and r['fast'], "target case must hit au=16/48cores/fast"
```

判读：任何 `DIFF` 行 = 门控/路由回归（必须修）；目标 case 断言失败 = 要么求解逻辑错、要么仿真输入写错（先核 shape 语义再怀疑逻辑）。

**探针 2（构建后真库 tiling 身份复核）**——ctypes 直调候选 `.so` 的 tiling 入口，确认真实 C++ 实现与仿真一致（不跑 kernel，秒级）：

```python
import ctypes as C
lib = C.CDLL("<候选目录>/lib/libfrnsm.so")
lib.frnsm_compute_tiling.restype = C.c_int
lib.frnsm_compute_tiling.argtypes = [C.c_int32, C.c_int32, C.c_int32, C.c_double,
    C.c_int32, C.POINTER(C.c_int64), C.c_int32, C.POINTER(C.c_int64), C.c_int32, C.POINTER(C.c_int64),
    C.POINTER(C.c_uint32), C.POINTER(C.c_uint32), C.POINTER(C.c_uint64)]
assert lib.frnsm_init(0) == 0
def arr(xs): return (C.c_int64*len(xs))(*xs)
def tiling(xs, gs, os_):
    bd = C.c_uint32(); tk = C.c_uint32(); ws = C.c_uint64()
    x=arr(xs); g=arr(gs); o=arr(os_)
    st = lib.frnsm_compute_tiling(0,0,0, 1e-6, len(xs), x, len(gs), g, len(os_), o, C.byref(bd), C.byref(tk), C.byref(ws))
    return st, bd.value, tk.value
st, bd, tk = tiling([128,6,128],[128],[128,6,128])
print(f"target fused_128x6x128: status={st} blockDim={bd} tilingKey={tk}")
assert st==0 and bd==48 and tk==0
# …其余 case 同法与基线期望 (bd, tk) 对账 …
```

## 设计决策

- 为什么逐字复刻而不是抽象近似：对账目标是"逐 case bd/tk 一致"，任何近似（简化 group 决策、跳过 P2CapacityLegal）都会让 DIFF 不可信；复刻成本一次性，且 Python 版可作为求解逻辑的活文档。
- 为什么 case 期望值取自基线验收 manifest 而不是手抄：manifest 是冻结口径（含 SHA），对账才有效力；本会话从预处理 manifest 提取 18 条 (bd, tk)。
- 为什么构建后还要探针 2：仿真复刻可能漏掉 C++ 侧整数提升/类型截断等实现细节；真库 tiling 入口直调（带完整 argtypes/restype，避免 ctypes 默认 32 位返回截断）把"C++ 实现"与"仿真"闭环。
- 当前方案路线档位：recommended_fastpath（构建前门禁 + 构建后身份复核两段式）。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0（coreNum=56、UB=253952B），CANN 9.2.0；Python 3（无第三方依赖）。
- 相关文件：`/tmp/opencode/sim_tiling.py`（探针 1，临时脚本）；候选目录 `lib/libfrnsm.so`（探针 2 目标）；冻结期望来源：预处理验收 manifest（18 条 L0 的 blockDim/tilingKey）。

## 验证信号

以下信号来自该融合算子三个优化候选的实施会话（Ascend950PR / CANN 9.2.0）：每个候选改 tiling 前先仿真对账、构建后真库复核，再走唯一验收入口。

- 仿真对账抓到自证输入错误：目标 case 断言 `AssertionError: target case must hit au=16/48cores/fast`——原因是仿真输入 gamma shape 误写为 `(6,128)`（A=128/R=768）；修正为 `(128,)` 后 18/18 case 对账 OK、目标 au=16/48 核/fast=True（来源平台：950；证据等级：tool_observed）。
- 仿真对账通过后上板复核一致：合并 DMA 快路候选 18 条 L0 仿真 bd/tk 与基线 manifest 全部一致（含 L0_019/024 广播 case 正确回退一般路径）；仅 host tiling 重平衡候选仿真预测"17/18 一致 + bound_019 填充提升（bd 1→3）"，构建后真库身份检查 17/18 逐字节一致、目标 case blockDim=48/tilingKey=0，与预测吻合（来源平台：950；证据等级：tool_observed）。
- 真库复核脚本（fresh-process、ctypes 全 argtypes）：目标 case status=0/bd=48/tk=0，边界 case（32,4096)/(1,)/(3,17 广播)/(7d)/(2501,6610 标量 other)/(5,2,9371233 group) 全部与期望一致（来源平台：950；证据等级：tool_observed）。
- 后续 18 条 L0 上板验收 strict 18/18 PASS，bd/tk 逐 case 与基线 manifest 一致——两段探针的预测全部兑现（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仿真只能验证 tiling 决策（bd/tk/au/fast），不能验证 kernel 执行与数值——功能验收仍必须上板走唯一入口。
- C++ 侧若有浮点参与求解（本算子 eps 只透传不参与决策），Python 复刻需注意浮点语义差异；含迭代收敛求解的 tiling 不建议此法（未验证）。
- 探针 2 依赖直调工程导出 tiling 入口符号；注册工程形态不同，需换入口（未验证）。
- case 期望值必须来自冻结 manifest：手抄或凭记忆写期望会引入假通过/假失败。

**未决项**：

- 探针 1 与 C++ 实现的同步目前靠人工逐字翻译，tiling 逻辑再演进时可考虑从 C++ 半自动抽取求解常量（未尝试）。
## 坏实践

- 改完 tiling 直接构建上板跑全量验收——构建与上板成本远高于一次 Python 对账，且门控错误往往只影响个别 case，全量验收日志里不易第一眼定位。
- 仿真输入 shape 手写不复核语义——本会话目标 case 断言失败的根因正是 gamma shape 写错（`(6,128)` vs `(128,)`），"探针失败先怀疑自己输入"是正确顺序。
- ctypes 调 `.so` 不设 argtypes/restype——64 位指针/返回值被默认 32 位截断，探针本身崩溃或误报（同会话另有脚本因此段错误）。
