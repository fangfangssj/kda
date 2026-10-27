---
type: Runbook
title: FP16输入不保证NMS采用FP16的IoU阈值路径
description: 真实npu_nms_v4两框阈值反例排除逐阶段half及最终half模拟，仅约束受测平台与版本。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-10-18T08:21:20Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# FP16输入不保证NMS采用FP16的IoU阈值路径

## 现象

旧NMS候选把half框转为FP32、计算FP32 IoU、再把最终IoU转half后与FP32阈值比较。新固定两框直接调用真实`torch_npu.npu_nms_v4`，没有加载旧扩展，发现该路线在一个相邻阈值上与参考的有效选择数量不同。

冻结坐标全部可用half精确表示：`[[0,0,0.96435546875,1.849609375],[0.369140625,0.0958251953125,2.494140625,2.744140625]]`；scores=`[0.9,0.8]`，max_output_size=2，score_threshold=-1，无分数平局。FP32逐阶段模拟IoU=0.1639391928911209；逐阶段half模拟=0.163818359375；FP32后仅最终half=0.1639404296875。

| FP32阈值 | 全FP32预测valid count | 逐阶段half预测 | 仅最终half预测 | 实际half输入valid count | 实际float输入valid count |
|---|---|---|---|---|---|
|0.16387876868247986|1|2|1|1|1|
|0.16393980383872986|2|2|1|2|2|
|0.1639404296875|2|2|2|2|2|

因此第一个点反驳逐阶段half模拟，第二个点反驳仅最终half模拟。参考结果与全FP32预测在这对输入上相容，不能从选择数量恢复内部完整IoU DAG。边界阈值也不能用`>=`/`>`相同的泛化假设替代，必须保留参考契约。

完整可执行机制载体为`probe_group_b.py`中的`f16`、`f32`、`iou`、`find_nms`与完整`nms`函数；原始bits与输出在`npu_observations.json#/observations/C044_actual_npu_nms_v4`。公开审材料必须携带这些实际函数和冻结坐标/输出，而非仅给旧私有路径。历史任务通过摘要、docstring的Python伪实现及被旧扩展monkeypatch后的reference均不作为本结论证据。此处未重建旧大N输入，也没有声称原case已修复。

## 适用场景

- 工程形态：Ascend NPU算子精度校准的小规模框架探针。
- API路径：npu_nms_v4；禁止把half输入推断成通用half IoU模拟。
- 验证平台与版本：a2/Ascend910B4，CANN9.1.0，torch/torch_npu2.9.0。
- 适用粒度：`exact_task`。
- 路线优先级：`banned`。
- 不适用或未验证场景：结果与全FP32预测相容，不证明内部完整DAG；未复现旧大N输入，不声称旧case修复。

## 判别方法

以下为实际执行公共模块 `probe_group_b.py` 的全文，包含导入、闭包上下文、输入生成、接口调用和原始输出记录；依赖该模块的 fresh 进程脚本另列如下。所有代码保持执行版本逐字一致。

```python
#!/usr/bin/env python3
"""Small, deterministic calibration only. No benchmark, model fitting or submission.
CPU mode has no dependencies. NPU mode requires the operator's installed torch_npu.
Every tensor is copied back and archived as raw hex, values and sha256.
"""
import argparse, hashlib, json, math, os, platform, struct, subprocess, sys, time
from pathlib import Path

def f32(x):
    try:return struct.unpack('<f',struct.pack('<f',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def f16(x):
    try:return struct.unpack('<e',struct.pack('<e',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def frombits(b):return struct.unpack('<f',struct.pack('<I',b))[0]
def bits(x):return '%08x'%struct.unpack('<I',struct.pack('<f',x))[0]
def rne11(x):
    u=int(bits(x),16)
    if (u&0x7f800000)==0x7f800000:return x
    return frombits((u+0x7ff+((u>>12)&1))&0xfffff000)
def bf16(x):
    u=int(bits(x),16)
    return frombits((u+0x7fff+((u>>16)&1))&0xffff0000)
def veltkamp(x):
    c=f32(f32(x)*4097.0);return f32(c-f32(c-x))
def seq(xs):
    r=0.0
    for x in xs:r=f32(r+x)
    return r

def iou(a,b,rounder):
    w=rounder(max(0.,rounder(min(a[2],b[2])-max(a[0],b[0]))))
    h=rounder(max(0.,rounder(min(a[3],b[3])-max(a[1],b[1]))))
    inter=rounder(w*h)
    ar=[rounder(rounder(x[2]-x[0])*rounder(x[3]-x[1])) for x in [a,b]]
    union=rounder(rounder(ar[0]+ar[1])-inter)
    return {'width':w,'height':h,'inter':inter,'areas':ar,'union':union,'iou':rounder(inter/union)}

def find_nms():
    # All coordinates are exactly representable half inputs before simulating stages.
    import random
    rng=random.Random(20261018)
    for _ in range(20000):
        a=[0.,0.,f16(rng.uniform(.5,3)),f16(rng.uniform(.5,3))]
        b=[f16(rng.uniform(0,.4)),f16(rng.uniform(0,.4)),f16(rng.uniform(.5,3)),f16(rng.uniform(.5,3))]
        v32=iou(a,b,f32);v16=iou(a,b,f16);last=f16(v32['iou'])
        vals=sorted(set([v32['iou'],v16['iou'],last]))
        if len(vals)==3 and vals[0]>.05:
            thresholds=[f32((x+y)/2) for x,y in zip(vals,vals[1:])]
            if len(set(thresholds))==2:return {'boxes':[a,b],'fp32_stages':v32,'fp16_stages':v16,'fp32_final16':last,'thresholds':thresholds}
    raise RuntimeError('No separating pair')

def cpu():
    vals=[frombits(x) for x in [0x3f800800,0x3f801800,0x00000001,0x000116c2,0x79800000,0x7b4097ce]]
    rows=[{'input':x,'input_fp32_bits':bits(x),'rne11':rne11(x),'rne11_bits':bits(rne11(x)),'veltkamp':veltkamp(x),'veltkamp_bits':bits(veltkamp(x))} for x in vals]
    xs=[frombits(0x42ab7fff),2.**-19,2.**-19]
    left=seq(xs);right=f32(xs[0]+f32(xs[1]+xs[2]))
    bias=[frombits(0x42ab8001),frombits(0xc2ab8001)]
    return {'C056_veltkamp_counterexamples':rows,'C059_midpoint':{'inputs':xs,'input_fp32_bits':[bits(x) for x in xs],'left':left,'left_bits':bits(left),'right':right,'right_bits':bits(right),'left_bf16':bf16(left),'right_bf16':bf16(right),'exact_sum':sum(xs)},'C059_fixed_bias_counterexamples':[{'input':x,'unbiased_bf16':bf16(x),'subtract_1e_4_bf16':bf16(f32(x-1e-4))} for x in bias],'C044_separating_pair':find_nms(),'claim_scope':'CPU arithmetic simulations only; no AscendC or reference-internal DAG inferred.'}

def record(t):
    t=t.detach().cpu().contiguous()
    raw=bytes(t.reshape(-1).view(torch.uint8).tolist())
    return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'raw_hex':raw.hex(),'sha256':hashlib.sha256(raw).hexdigest(),'values':t.reshape(-1).tolist()}

def read_setting(obj,name):
    try:return getattr(obj,name)
    except Exception as e:return {'unavailable':str(e)}
def npu_probe(device):
    global torch
    import torch
    import torch_npu
    torch.npu.set_device(device)
    env={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_npu':getattr(torch_npu,'__version__','unknown'),'device':device,'device_name':torch.npu.get_device_name(device),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'precision_environment':{k:v for k,v in os.environ.items() if k.startswith(('ASCEND','ACL_OP','NPU_')) and not any(s in k for s in ['TOKEN','SECRET','PASSWORD'])},'settings':{}}
    for nm,obj in [('torch.npu.conv',getattr(torch.npu,'conv',None)),('torch.npu.matmul',getattr(torch.npu,'matmul',None)),('torch.backends.cuda.matmul',torch.backends.cuda.matmul)]:
        env['settings'][nm]={x:read_setting(obj,x) for x in ['allow_hf32','allow_tf32']}
    for command in [['npu-smi','info']]:
        try:r=subprocess.run(command,capture_output=True,text=True,timeout=10);env[' '.join(command)]={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except Exception as e:env[' '.join(command)]={'error':str(e)}
    out={'environment':env,'cpu_simulations':cpu(),'observations':{},'limitations':['Framework outputs do not establish AscendC Compare/Rsqrt semantics or internal reduction/multiply DAG.','No candidate AIV-vs-Cube performance claim; no full benchmark.']}
    # Each group isolated so an unavailable API never becomes an invented observation.
    def run(key,fn):
        try:out['observations'][key]=fn();torch.npu.synchronize()
        except Exception as e:out['observations'][key]={'error':type(e).__name__+': '+str(e)}
    def nms():
        p=out['cpu_simulations']['C044_separating_pair'];rows=[]
        for dtype in [torch.float16,torch.float32]:
            boxes=torch.tensor(p['boxes'],dtype=dtype,device=device);scores=torch.tensor([.9,.8],dtype=dtype,device=device)
            for threshold in p['thresholds']+[p['fp32_final16']]:
                th=torch.tensor(threshold,dtype=torch.float32,device=device);st=torch.tensor(-1.,dtype=torch.float32,device=device)
                r=torch_npu.npu_nms_v4(boxes,scores,2,th,st,pad_to_max_output_size=True)
                rows.append({'boxes':record(boxes),'scores':record(scores),'threshold':record(th),'result':[record(x) for x in r],'candidate_suppression_gt':{'fp32':p['fp32_stages']['iou']>threshold,'step_fp16':p['fp16_stages']['iou']>threshold,'final_fp16':p['fp32_final16']>threshold}})
        return rows
    run('C044_actual_npu_nms_v4',nms)
    def conv_and_matmul():
        rows=[]
        # One active channel prevents reduction-order explanations of product differences.
        vals=[frombits(x) for x in [0x3f800801,0x3f801801,0x3f802801,0xbf800801,0x3f7ff801,0x3f800001,0x3f801fff,0x3f802001]]
        x=torch.zeros((1,16,8),dtype=torch.float32);x[0,0,:]=torch.tensor(vals)
        w=torch.zeros((16,16,1),dtype=torch.float32);w[:,0,0]=frombits(0x3f801801)
        A=torch.tensor(vals,dtype=torch.float32).reshape(8,1).repeat(1,16);A[:,1:]=0
        B=torch.zeros((16,16),dtype=torch.float32);B[:,0]=frombits(0x3f801801)
        for allow in [None,False,True]:
            changed=[]
            try:
                if allow is not None:
                    for obj in [getattr(torch.npu,'conv',None),getattr(torch.npu,'matmul',None)]:
                        if obj is not None and hasattr(obj,'allow_hf32'):
                            old=obj.allow_hf32;obj.allow_hf32=allow;changed.append((obj,old))
                xd=x.to(device);wd=w.to(device);ad=A.to(device);bd=B.to(device)
                conv=torch.nn.functional.conv1d(xd,wd);mm=torch.matmul(ad,bd.T)
                rows.append({'allow_hf32_requested':allow,'switches_changed':len(changed),'x':record(x),'weight':record(w),'conv1d':record(conv),'A':record(A),'B':record(B),'matmul_ABt':record(mm),'sim_full_f32_product':[f32(v*frombits(0x3f801801)) for v in vals],'sim_rne11_product':[f32(rne11(v)*rne11(frombits(0x3f801801))) for v in vals]})
            finally:
                for obj,old in changed:obj.allow_hf32=old
        return rows
    run('C056_C078_single_active_channel',conv_and_matmul)
    def sums():
        rows=[]
        for xs in [[frombits(0x42ab7fff),2.**-19,2.**-19],[1e8,1.,-1e8]]:
            t=torch.tensor(xs,dtype=torch.float32,device=device)
            outputs={'torch_sum':t.sum(),'left':(t[0]+t[1])+t[2],'right':t[0]+(t[1]+t[2])}
            rows.append({'input':record(t),'outputs':{k:record(v) for k,v in outputs.items()},'bf16_outputs':{k:record(v.to(torch.bfloat16)) for k,v in outputs.items()}})
        return rows
    run('C059_sum_midpoint',sums)
    def rsqrts():
        t=torch.tensor([frombits(x) for x in [0x3f800000,0x40400000,0x3727c5ac,0x2edbe6ff,0x00800000,0x00000001,0x7f7fffff,0x00000000,0x80000000,0x7f800000,0xbf800000,0x7fc12345]],dtype=torch.float32,device=device)
        y=torch.rsqrt(t);z=torch.reciprocal(torch.sqrt(t))
        return {'fixed_var_plus_eps':record(t),'framework_rsqrt':record(y),'framework_sqrt_reciprocal':record(z),'fp64_math_reference':[1/math.sqrt(x) if x>0 and math.isfinite(x) else ('positive_inf' if x==0 else (0. if x==float('inf') else 'nan')) for x in t.cpu().tolist()]}
    run('C066_framework_only_rsqrt',rsqrts)
    env['loaded_ascend_libraries'] = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(s in line.lower() for s in ['ascend','torch_npu','libacl'])))
    env['ascend_version_files'] = {str(p): p.read_text(errors='replace') for p in [Path('/usr/local/Ascend/ascend-toolkit/latest/version.cfg'),Path('/usr/local/Ascend/cann-9.1.0/version.cfg'),Path('/usr/local/Ascend/cann-9.1.0/version.info')] if p.is_file()}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu','npu'],required=True);ap.add_argument('--device',default='npu:1');ap.add_argument('--output',required=True);args=ap.parse_args()
    p=Path(args.output)
    if p.exists():raise SystemExit('Refuse to overwrite '+str(p))
    out=cpu() if args.mode=='cpu' else npu_probe(args.device)
    out['producer']={'command':sys.argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=True)+'\n');print(json.dumps({'output':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'groups':list(out.get('observations',out))},ensure_ascii=False))
if __name__=='__main__':main()
```

实际执行命令（容器已source实际CANN9.1.0环境，local device0）：

```sh
python3 probe_group_b.py --mode npu --device npu:0 --output npu_observations.json
```

## 设计决策

选择冻结bits和小输入以区分精度路径，而不是从张量dtype或历史全任务通过反推内部DAG。保留未支持的解释，不把新来源称为旧会话恢复。

## 系统信息

本次是固定输入的小规模框架参考校准，使用Ascend910B4（a2）、公开结果所记CANN9.1.0版本、torch/torch_npu2.9.0。源码载体为`probe_group_b.py`和`fresh_precision_probe.py`，原始结果记录输入/输出dtype、shape、stride、raw hex及SHA256。没有运行整benchmark或官网提交，也未加载旧NMS扩展。

## 验证信号

以下来自本次实际小探针的原始机器结果，而非旧摘要。
- 同一对框在三个阈值下，half与float输入的npu_nms_v4有效数量均为1/2/2；逐阶段half与仅最终half路径分别在第一个及第二个阈值预测不同。（来源平台：a2；证据等级：tool_observed）

## 失效边界

结果与全FP32预测相容，不证明内部完整DAG；未复现旧大N输入，不声称旧case修复。其他平台未验证，推断可迁移须重新校准。

## 坏实践

将一次数据类型声明或历史通过总结当作未观测内部数值路径的证明；为对齐少数边界案例采用无条件补偿而不留反例。

## 性能影响

未测性能，不作性能排序或收益声称。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/reference.json#/observations/C044_actual_npu_nms_v4`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/probe_group_b.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
