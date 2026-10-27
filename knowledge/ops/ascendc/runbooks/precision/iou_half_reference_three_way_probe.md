---
type: Runbook
title: "IoU 用三方探针对齐半精度面积合同"
description: "在固定版本当前后端上，对照逐步半精度、逐步单精度和无面积偏移参考，避免从输出落点猜内部精度。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: "2026-10-18T08:29:56Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# IoU 用三方探针对齐半精度面积合同

## 现象

对当前 torch_npu.npu_iou 参考开展固定输入实验时，首先把 box 布局写成转置的方阵而得到全零。这一试探没有认证任何内部算法。随后依据输入合同改用不相等的非方阵 `[N,4]`/`[M,4]`，保留第一次试探并另建新结果；再比较逐步半精度、逐步单精度、无面积偏移三条独立 CPU 参考。

## 适用场景

工程形态是参考后端合同探针；API 路径为当前 `torch_npu.npu_iou` 的 IoU/IoF；实际平台为 a2/Ascend910B4、CANN 9.1.0、torch_npu 2.9.0。适用粒度 `same_operator`，路线优先级 `recommended_fastpath` 指先固定输入布局并做三方参考诊断。旧 CANN、其他入口、编译选路与任意输入域未认证。

## 系统信息

本次独立框架探针记录 Ascend910B4（a2）、CANN 9.1.0 版本标记与 PyTorch/torch_npu 2.9.0。实际入口为 `run_iou_corrected_layout.py` 对 `torch_npu.npu_iou` 的调用；公开结果保留 `script_sha256`、`reference_code_sha256`、四组完整记录及保存输入/输出bits。本卡没有单独构建或加载自定义IoU微核，不以同批其他探针的CMake或模块hash作为此框架调用的身份；这是单次独立实验，不继承历史任务成功。

## 判别方法

固定实际坐标与输出 dtype，分别比较逐步半精度的坐标/面积路径、逐步单精度后最终回退和去偏移路线。面积有偏移，交集不加该偏移；保留缩放和 union 顺序。NaN 按类别单列。下列 `cast`、`div` 和 `iou_compute` 是实际 CPU 模块中逐字提取的完整独立函数及原导入行，只保留本卡所需机制。CPU 建模阶段本身不执行 NPU；随后调用脚本才产生下述设备结果，两阶段不互相替代。实际调用脚本保持全文如下：

### cpu_mechanisms.py 的完整独立依赖和 IoU 函数

```python
import hashlib,json,math,platform,random,struct,sys

def cast(x,kind):
    try:return struct.unpack('<'+kind,struct.pack('<'+kind,x))[0]
    except OverflowError:return math.copysign(math.inf,x)

def div(a,b):
    if b==0:return math.nan if a==0 else math.copysign(math.inf,a*b)
    return a/b

def iou_compute(box,gt,kind,offset=.01,scale=.2,order='sub_add'):
    q=lambda v:cast(v,kind)
    b=list(map(q,box));g=list(map(q,gt));eps=q(offset);s=q(scale)
    def area(v):return q(q(q(q(v[2]-v[0])+eps)*s)*q(q(q(v[3]-v[1])+eps)*s))
    ba=area(b);ga=area(g)
    w=q(max(q(q(min(b[2],g[2])-max(b[0],g[0]))*s),0.))
    h=q(max(q(q(min(b[3],g[3])-max(b[1],g[1]))*s),0.))
    inter=q(w*h)
    den=q(q(ba-inter)+ga) if order=='sub_add' else q(q(ba+ga)-inter)
    return q(div(inter,den)),{'bbox_area':ba,'gt_area':ga,'inter':inter,'den':den}
```

### run_iou_corrected_layout.py

```python
import hashlib,json,math,sys,subprocess
from pathlib import Path
import torch,torch_npu
from cpu_mechanisms import iou_compute,cast,div
ROOT=Path(__file__).resolve().parent
out=ROOT/'npu_iou_corrected_layout';out.mkdir(exist_ok=False);torch.npu.set_device(0)
# The first probe used a [4,4] transposed tensor; that layout was ambiguous.
# This new run deliberately uses N!=M and N,M!=4 with [N,4] box layout.
boxes=[[0.,0.,1.,1.],[0.,0.,.001,.001],[0.,0.,10.,10.],[.1,.2,.9,.8],[1.,1.,2.,2.]]
gts=[[0.,0.,1.,1.],[0.,0.,.002,.002],[.1,.2,.9,.8]]
records=[];ts={}
for dtype in [torch.float16,torch.float32]:
 b=torch.tensor(boxes,dtype=dtype);g=torch.tensor(gts,dtype=dtype);ts[f'b_{dtype}']=b;ts[f'g_{dtype}']=g
 for mode in [0,1]:
  y=torch_npu.npu_iou(b.to('npu:0'),g.to('npu:0'),mode=mode);torch.npu.synchronize();y=y.cpu();ts[f'output_{dtype}_{mode}']=y
  refs={}
  for label,kind,offset in [('half_steps_offset','e',.01),('fp32_steps_offset','f',.01),('half_steps_nooffset','e',0.)]:
   ref=[]
   for gg in g.tolist():
    row=[]
    for bb in b.tolist():
     val,mid=iou_compute(bb,gg,kind,offset=offset)
     if mode==1:val=cast(div(mid['inter'],mid['gt_area']),kind)
     row.append(val)
    ref.append(row)
   r=torch.tensor(ref,dtype=dtype);ts[f'ref_{label}_{dtype}_{mode}']=r
   eq=(y==r)|(torch.isnan(y)&torch.isnan(r))
   refs[label]={'same_value_or_nan':int(eq.sum()),'total':y.numel(),'reference_output':r.tolist(),'max_abs_diff_finite':float((y.float()-r.float()).abs()[torch.isfinite(y)&torch.isfinite(r)].max())}
  records.append({'dtype':str(dtype),'mode':mode,'input_shapes':[list(b.shape),list(g.shape)],'output_shape':list(y.shape),'output':y.tolist(),'references':refs,'half_grid_count':int(((y==y.half().to(dtype))|torch.isnan(y)).sum())})
torch.save(ts,out/'inputs_outputs.pt')
clean=lambda v:'nan' if isinstance(v,float) and math.isnan(v) else 'inf' if isinstance(v,float) and v==math.inf else '-inf' if isinstance(v,float) and v==-math.inf else [clean(z) for z in v] if isinstance(v,list) else {k:clean(z) for k,z in v.items()} if isinstance(v,dict) else v
r={'schema_version':'expanded_local.group_a.iou_layout.v1','run_kind':'new_controlled_reference_comparison_corrected_input_layout','device':torch.npu.get_device_name(0),'torch':torch.__version__,'torch_npu':torch_npu.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_code_sha256':hashlib.sha256((ROOT/'cpu_mechanisms.py').read_bytes()).hexdigest(),'legacy_iou_sha256':hashlib.sha256((ROOT/'npu_run_001/legacy_iou_1.py').read_bytes()).hexdigest(),'records':records,'limitations':['Current CANN9.1 backend only; not original historical run','Matching these paths does not prove an unobserved backend code path selected the recovered Python file','FP16-grid outputs alone do not establish internal compute dtype','No old comparator or oracle was changed']}
(out/'result.json').write_text(json.dumps(clean(r),indent=2,allow_nan=False)+'\n');print(json.dumps(clean(r),indent=2,allow_nan=False))
```

## 根因

当前观测能区分参考数值合同，但没有追到实际选中的后端内部实现。仅凭输出落在 fp16 网格上不能证明内部全链 fp16：fp32 计算末尾 cast 到 fp16 再转回也有相同落点。受测三方参考的逐元素一致性比网格统计更强，但仍不能代替设备执行路径绑定。

## 验证信号

来源为本次更正布局后的独立参考后端调用及保存的三方参考。

- 输入/输出为 half/fp32 的 IoU/IoF 共四组，逐步半精度并带面积偏移参考各 15/15 相等。（来源平台：a2；证据等级：artifact_verified）
- fp32 输出的两组，逐步 fp32 参考各只有 5/15 相等；去面积偏移参考为 7/15 和 6/15。该信号拒绝把这几条受测路径当作相同数值合同。（来源平台：a2；证据等级：artifact_verified）

## 设计决策

用非方阵输入暴露布局错误，独立保留新旧实验；不通过改旧比较器、容差或期望值制造通过。恢复的当前 legacy iou.py 仅用于限定静态源码内容，不凭同名文件声称它就是本次实际执行路径。

## 坏实践

从“输出大部分为半精度精确值”直接认证内部全链 fp16，或根据同名 legacy 文件猜选路。

## 性能影响

未测性能。

## 失效边界

- 优先项：复现并追根因到机制层，绑定当前实际后端入口/编译选路，再讨论内部 dtype。
- 本次只验证给定独立新参考的局部数值合同；不能认证历史 legacy 版本、常量输入竞争结论或所有退化 box 行为。

## 预防

参考取证先记录输入布局、坐标实际值、版本与源码 hash，再使用有区分力的三方对照。

## 误诊

首次转置方阵全零是本次试探的输入布局问题；它没有证明后端计算故障或旧 kernel 缺陷。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/iou_corrected_layout.json#/records`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/iou_corrected_layout_bits.json`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/run_iou_corrected_layout.py`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/cpu_mechanisms.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
