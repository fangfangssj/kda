---
type: Runbook
title: "Sigmoid 正指数重排在饱和区产生 NaN"
description: "同输入对比高阶 Sigmoid、负指数 Div 链与正指数商，区分代数等价和设备有限精度等价。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: "2026-10-18T08:29:56Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# Sigmoid 正指数重排在饱和区产生 NaN

## 现象

在 AscendC 直调微核 `probe.cpp` 中，同一输入分别执行高阶 `AscendC::Sigmoid`、`Muls→Exp→Adds→Duplicate→Div`、`Muls→Exp→Adds→Reciprocal` 和 `Exp(x)/(1+Exp(x))`。在有限正半精度输入 `11.1015625` 上，高阶与负指数 Div 链输出 `1`，正指数商输出 `NaN`，因此不能只凭代数等价替换公式。

## 适用场景

工程形态为 kernel 直调；API 路径为上述 AscendC Vector 激活/指数/除法接口；验证平台为 a2/Ascend910B4、CANN 9.1.0。适用粒度 `operator_family`，路线优先级 `recommended_fastpath` 指诊断优先使用同输入单公式对照。其他芯片、版本、bf16 和融合算子的完整语义未验证。

## 系统信息

本次独立探针使用 Ascend910B4（a2）、CANN 9.1.0、PyTorch/torch_npu 2.9.0 和 Python 3.11。编译包路径为 `/usr/local/Ascend/cann-9.1.0`，以项目现有 `build_ascendc.py` 的隔离拷贝生成 CMake 并限并发构建。实际模块路径、源码/模块 SHA-256、输入输出快照和原始日志保留在审核材料；这是单次独立实验，不继承历史任务成功。

## 判别方法

固定输入、队列、搬运、输出 dtype 和比较器，只按 `mode` 切公式。分别比较普通区、饱和区和非有限值；用 CPU fp64 Sigmoid 后回退 dtype 作独立参考。NaN 类别和值相等分开处理，不能把所有 NaN 当正常输出通过。以下完整探针文件和 Host 绑定实际用于本次编译：

### activation_int64/kernel/probe.cpp

```cpp
#include "kernel_operator.h"
#include "activation/sigmoid.h"

template<typename T>
__aicore__ inline void Activation(GM_ADDR x, GM_ADDR y, uint32_t mode) {
    constexpr uint32_t n=256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesBuf;
    pipe.InitBuffer(qi,1,n*sizeof(T));pipe.InitBuffer(qo,1,n*sizeof(T));
    pipe.InitBuffer(onesBuf,n*sizeof(T));
    AscendC::GlobalTensor<T> gx,gy;gx.SetGlobalBuffer((__gm__ T*)x,n);gy.SetGlobalBuffer((__gm__ T*)y,n);
    auto xi=qi.AllocTensor<T>();AscendC::DataCopy(xi,gx,n);qi.EnQue(xi);
    xi=qi.DeQue<T>();auto yo=qo.AllocTensor<T>();auto ones=onesBuf.Get<T>();
    if(mode==0) {AscendC::Sigmoid(yo,xi,n);}
    else if(mode==3) {
        AscendC::Exp(xi,xi,n);AscendC::Adds(ones,xi,(T)1,n);AscendC::Div(yo,xi,ones,n);
    } else {
        AscendC::Muls(xi,xi,(T)-1,n);AscendC::Exp(xi,xi,n);AscendC::Adds(xi,xi,(T)1,n);
        if(mode==1) {AscendC::Duplicate(ones,(T)1,n);AscendC::Div(yo,ones,xi,n);}
        else {AscendC::Reciprocal(yo,xi,n);}
    }
    qo.EnQue(yo);qi.FreeTensor(xi);yo=qo.DeQue<T>();AscendC::DataCopy(gy,yo,n);qo.FreeTensor(yo);
}
extern "C" __global__ __aicore__ void activation_float(GM_ADDR x,GM_ADDR y,uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);Activation<float>(x,y,mode);
}
extern "C" __global__ __aicore__ void activation_half(GM_ADDR x,GM_ADDR y,uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);Activation<half>(x,y,mode);
}

// Independent all-int32 scalar comparison after bit-preserving UB reinterpret.
// This does not claim vector Gather or vector int64 comparison is supported.
extern "C" __global__ __aicore__ void compare64(GM_ADDR a,GM_ADDR b,GM_ADDR out,GM_ADDR planes,uint32_t broadcast) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t n=256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,1> qa,qb;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> qo,qp;
    pipe.InitBuffer(qa,1,n*8);pipe.InitBuffer(qb,1,n*8);pipe.InitBuffer(qo,1,n*8);pipe.InitBuffer(qp,1,n*4*4);
    AscendC::GlobalTensor<int64_t> ga,gb,gout;AscendC::GlobalTensor<int32_t> gp;
    ga.SetGlobalBuffer((__gm__ int64_t*)a,n);gb.SetGlobalBuffer((__gm__ int64_t*)b,n);gout.SetGlobalBuffer((__gm__ int64_t*)out,n);gp.SetGlobalBuffer((__gm__ int32_t*)planes,n*4);
    auto ai=qa.AllocTensor<int64_t>();auto bi=qb.AllocTensor<int64_t>();
    AscendC::DataCopy(ai,ga,n);AscendC::DataCopy(bi,gb,n);qa.EnQue(ai);qb.EnQue(bi);
    ai=qa.DeQue<int64_t>();bi=qb.DeQue<int64_t>();
    auto av=ai.ReinterpretCast<int32_t>();auto bv=bi.ReinterpretCast<int32_t>();
    auto oi=qo.AllocTensor<int64_t>();auto ov=oi.ReinterpretCast<int32_t>();auto pv=qp.AllocTensor<int32_t>();
    for(uint32_t i=0;i<n;i++) {
        uint32_t j=broadcast?0:i;
        int32_t al=av.GetValue(2*i),ah=av.GetValue(2*i+1),bl=bv.GetValue(2*j),bh=bv.GetValue(2*j+1);
        const bool aGreater=(ah>bh)||((ah==bh)&&((uint32_t)al>(uint32_t)bl));
        ov.SetValue(2*i,aGreater?al:bl);ov.SetValue(2*i+1,aGreater?ah:bh);
        pv.SetValue(4*i,al);pv.SetValue(4*i+1,ah);pv.SetValue(4*i+2,bl);pv.SetValue(4*i+3,bh);
    }
    qo.EnQue(oi);qp.EnQue(pv);qa.FreeTensor(ai);qb.FreeTensor(bi);
    oi=qo.DeQue<int64_t>();pv=qp.DeQue<int32_t>();AscendC::DataCopy(gout,oi,n);AscendC::DataCopy(gp,pv,n*4);qo.FreeTensor(oi);qp.FreeTensor(pv);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void activation_float_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t mode) {activation_float<<<bd,nullptr,s>>>(x,y,mode);}
extern "C" void activation_half_do(uint32_t bd,void* s,uint8_t* x,uint8_t* y,uint32_t mode) {activation_half<<<bd,nullptr,s>>>(x,y,mode);}
extern "C" void compare64_do(uint32_t bd,void* s,uint8_t* a,uint8_t* b,uint8_t* y,uint8_t* p,uint32_t broadcast) {compare64<<<bd,nullptr,s>>>(a,b,y,p,broadcast);}
#endif
```

### activation_int64/kernel/pybind11.cpp

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void activation_float_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
extern "C" void activation_half_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t);
extern "C" void compare64_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
at::Tensor activation(const at::Tensor& x,uint32_t mode) {
    TORCH_CHECK(x.numel()==256&&x.is_contiguous(),"requires contiguous 256 input");
    TORCH_CHECK(mode<=3,"invalid mode");
    auto y=at::empty_like(x);auto s=c10_npu::getCurrentNPUStream().stream(false);
    if(x.scalar_type()==at::kFloat)activation_float_do(1,s,ptr(x),ptr(y),mode);
    else {TORCH_CHECK(x.scalar_type()==at::kHalf,"dtype must float/half");activation_half_do(1,s,ptr(x),ptr(y),mode);}
    return y;
}
pybind11::tuple maximum64(const at::Tensor& a,const at::Tensor& b,bool broadcast) {
    TORCH_CHECK(a.scalar_type()==at::kLong&&b.scalar_type()==at::kLong,"int64 required");
    TORCH_CHECK(a.numel()==256&&b.numel()==256&&a.is_contiguous()&&b.is_contiguous(),"256 contiguous inputs required; broadcast mode reads b[0]");
    auto y=at::empty_like(a);auto p=at::empty({256,4},a.options().dtype(at::kInt));auto s=c10_npu::getCurrentNPUStream().stream(false);
    compare64_do(1,s,ptr(a),ptr(b),ptr(y),ptr(p),broadcast);return pybind11::make_tuple(y,p);
}
PYBIND11_MODULE(_expanded_a_probe,m){m.def("activation",&activation);m.def("maximum64",&maximum64);}
```

### run_npu.py

```python
#!/usr/bin/env python3
"""Bounded newly authored evidence; run only on the root-assigned logical device."""
import argparse,hashlib,importlib.util,json,math,os,platform,random,subprocess,sys,time
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'activation_int64/kernel/build'))
import _expanded_a_probe as ext
from cpu_mechanisms import iou_compute,parts

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,float):return 'nan' if math.isnan(v) else 'inf' if v==math.inf else '-inf' if v==-math.inf else v
 if isinstance(v,list):return list(map(clean,v))
 if isinstance(v,dict):return {k:clean(w) for k,w in v.items()}
 return v

def h(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def observe(cmd):
 try:
  p=subprocess.run(cmd,text=True,capture_output=True,timeout=20);return {'argv':cmd,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except Exception as e:return {'argv':cmd,'exception':repr(e)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 torch.npu.set_device(args.device)
 result={'schema_version':'expanded_local.group_a.npu.v1','run_kind':'new_independent_probe_not_historical_recovery','device_logical':args.device,'device_name':torch.npu.get_device_name(args.device),'torch':torch.__version__,'torch_npu':torch_npu.__version__,'python':sys.version,'platform':platform.platform(),'env':{k:v for k,v in os.environ.items() if k.startswith(('ASCEND','LD_LIBRARY_PATH','PATH'))},'npu_smi':observe(['npu-smi','info']),'module_path':ext.__file__,'module_sha256':sha(ext.__file__),'script_sha256':sha(__file__),'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in sorted((ROOT/'activation_int64/kernel').glob('*')) if x.is_file()},'timing_claims':False}
 # Fixed CPU-generated inputs and CPU references, copied to assigned NPU.
 xs=[-math.inf,-1000.,-100.,-90.,-20.,-12.,-11.1,-11.,-5.5,-1.,-0.,0.,1.,5.5,11.,11.1,12.,20.,90.,100.,1000.,math.inf,math.nan]
 xs=(xs*((256+len(xs)-1)//len(xs)))[:256]
 activation=[];tensors={}
 for dtype in [torch.float16,torch.float32]:
  x=torch.tensor(xs,dtype=dtype);tensors[f'activation_input_{dtype}']=x
  outputs=[]
  for mode in range(4):
   y=ext.activation(x.to(f'npu:{args.device}'),mode);torch.npu.synchronize();yc=y.cpu();outputs.append(yc);tensors[f'activation_{dtype}_mode{mode}']=yc
  ref=torch.sigmoid(x.to(torch.float64)).to(dtype);tensors[f'activation_reference_{dtype}']=ref
  same=lambda a,b:((a==b)|(torch.isnan(a)&torch.isnan(b)))
  activation.append({'dtype':str(dtype),'input_sha256':h(x),'reference':'CPU torch.sigmoid fp64 then cast; includes NaN/Inf','input':x.tolist(),'outputs':[y.tolist() for y in outputs],'reference_output':ref.tolist(),'same_class_or_value_vs_reference':[int(same(y,ref).sum()) for y in outputs],'mode0_vs_mode123_equal_counts':[int(same(outputs[0],y).sum()) for y in outputs[1:]],'n':256,'mode_names':['AscendC::Sigmoid','Muls/Exp/Adds/Duplicate/Div','Muls/Exp/Adds/Reciprocal','Exp/Adds/Div']})
 result['C002']=activation
 boundary=[-(1<<63),-(1<<63)+1,-(1<<32)-1,-(1<<32),-(1<<31)-1,-(1<<31),-1,0,1,(1<<31)-1,1<<31,(1<<32)-1,1<<32,(1<<32)+1,(1<<63)-2,(1<<63)-1]
 a=torch.tensor([u for u in boundary for _ in boundary],dtype=torch.int64);b=torch.tensor(boundary*len(boundary),dtype=torch.int64)
 rng=random.Random(61018);pairs=[(a,b),(torch.tensor([rng.randrange(-(1<<63),1<<63) for _ in range(256)],dtype=torch.int64),torch.tensor([rng.randrange(-(1<<63),1<<63) for _ in range(256)],dtype=torch.int64))]
 comparison=[]
 for case,(a,b) in enumerate(pairs):
  for broadcast in [False,True]:
   y,planes=ext.maximum64(a.to(f'npu:{args.device}'),b.to(f'npu:{args.device}'),broadcast);torch.npu.synchronize();y=y.cpu();planes=planes.cpu()
   ref=torch.maximum(a,b[0] if broadcast else b)
   expected=torch.tensor([[parts(int(aa))[1]-(1<<32) if parts(int(aa))[1]>>31 else parts(int(aa))[1],parts(int(aa))[0],parts(int(bb))[1]-(1<<32) if parts(int(bb))[1]>>31 else parts(int(bb))[1],parts(int(bb))[0]] for aa,bb in zip(a,b[0].expand_as(a) if broadcast else b)],dtype=torch.int32)
   prefix=f'int64_case{case}_broadcast{broadcast}';tensors[prefix+'_a']=a;tensors[prefix+'_b']=b;tensors[prefix+'_out']=y;tensors[prefix+'_planes']=planes;tensors[prefix+'_ref']=ref
   comparison.append({'case':case,'broadcast_b0':broadcast,'n':256,'maximum_exact_match':int((y==ref).sum()),'bitplane_exact_match_entries':int((planes==expected).sum()),'bitplane_entries':1024,'a_sha256':h(a),'b_sha256':h(b),'out_sha256':h(y),'limitation':'UB ReinterpretCast<int32_t> plus scalar int32/uint32 comparison; no vector Gather or saturating Cast was tested'})
 result['C024']=comparison
 # New independent controlled target-library observation, no rewritten old oracle.
 iou_rows=[]
 boxes=[[0.,0.,1.,1.],[0.,0.,.001,.001],[0.,0.,10.,10.],[0.,0.,0.,0.]]
 b=torch.tensor(boxes,dtype=torch.float32).t().contiguous();g=b.clone()
 for dtype in [torch.float16,torch.float32]:
  bc=b.to(dtype);gc=g.to(dtype);tensors[f'iou_bboxes_{dtype}']=bc;tensors[f'iou_gtboxes_{dtype}']=gc
  for mode in [0,1]:
   try:
    y=torch_npu.npu_iou(bc.to(f'npu:{args.device}'),gc.to(f'npu:{args.device}'),mode=mode);torch.npu.synchronize();yc=y.cpu();tensors[f'iou_{dtype}_{mode}']=yc
    iou_rows.append({'dtype':str(dtype),'mode':mode,'input_layout':'[4,N] and [4,M]','output_shape':list(yc.shape),'output':yc.tolist(),'output_sha256':h(yc),'half_grid_elements':int((yc==yc.half().to(yc.dtype)).sum()),'total':yc.numel(),'interpretation':'Actual current torch_npu backend observation, no claim about old backend internal dtype'})
   except Exception as e:iou_rows.append({'dtype':str(dtype),'mode':mode,'phase':'target_reference_call','exception':repr(e)})
 result['C034']=iou_rows
 legacy=[]
 for base in [Path('/usr/local/Ascend/cann-9.1.0'),Path('/usr/local/Ascend/ascend-toolkit/latest')]:
  for path in base.glob('**/iou.py'):
   if 'ops_legacy' in str(path) or 'dynamic' in str(path):
    legacy.append({'path':str(path),'realpath':str(path.resolve()),'sha256':sha(path)})
    (out/('legacy_iou_'+str(len(legacy))+'.py')).write_bytes(path.read_bytes())
 result['legacy_iou_sources']=legacy
 torch.save(tensors,out/'inputs_outputs.pt');result['tensor_archive_sha256']=sha(out/'inputs_outputs.pt')
 (out/'result.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(out),'device':result['device_name'],'C002':[r['mode0_vs_mode123_equal_counts'] for r in activation],'C024':comparison,'C034':iou_rows,'legacy':legacy},ensure_ascii=False,default=str))
if __name__=='__main__':main()
```

## 根因

正指数结果在饱和区成为 `Inf`，分子及加一后的分母均为 `Inf`，商成为 `Inf/Inf`；负指数路径在相同正输入处的指数接近零，仍能给出有限输出。该解释由有限输入的 CPU 舍入模型和设备三方输出共同支持；不能据此推导所有 CANN Exp 的实现或误差界。

## 验证信号

来源为本次独立激活微核的实际加载模块及保存的输入输出。

- 两种 dtype 的高阶 Sigmoid 与负指数 Div 链均为 256/256 同值或同 NaN 类；这只是数值观测，不代表 lowering 相同。（来源平台：a2；证据等级：artifact_verified）
- 半精度有限输入 `11.1015625`：高阶输出 `1`，正指数商输出 `NaN`；fp32 输入 `90` 也有同类区分。（来源平台：a2；证据等级：artifact_verified）

### cpu_mechanisms.py 的实际导入依赖

上述 runner 导入的模块在本页只采用下列完整依赖单元。代码按原文件段落逐字摘录；这些函数定义、导入和常量足以满足该 runner 的模块依赖，未包含其他独立实验的测试入口。

```python
#!/usr/bin/env python3
"""New independent CPU models. Never validates a CANN device API or historical run."""
import hashlib,json,math,platform,random,struct,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def cast(x,kind):
    try:return struct.unpack('<'+kind,struct.pack('<'+kind,x))[0]
    except OverflowError:return math.copysign(math.inf,x)
def div(a,b):
    if b==0:return math.nan if a==0 else math.copysign(math.inf,a*b)
    return a/b
MASK=(1<<32)-1
def parts(x):
    raw=struct.pack('<q',x)
    lo,hiu=struct.unpack('<II',raw)
    hi=hiu-(1<<32) if hiu>>31 else hiu
    return hi,lo
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

## 设计决策

保留同一搬运和队列骨架，以公式 mode 为干预量；不使用整融合算子通过记录隔离激活因果。Reciprocal 路作为相邻对照，其近似误差已有独立问题域，不把该实测扩张为通用误差界。

## 坏实践

只改写代数式便声称所有合法 dtype/输入等价，或用最终 cast 掩盖中间溢出。

## 性能影响

未测量性能，不推断任何模式更快。

## 失效边界

本实验覆盖少量代表输入，不能认证完整 Sigmoid 误差合同。高阶接口与 Div 链的编译操作数是否一致属于独立 compiler 证据，未由本卡认证。融合算子缓冲、量化 scale 等历史问题未复验；没有修复成功声明。

## 预防

公式重排先枚举中间量的范围和非有限值，再做相同输入的单变量设备对照。

## 误诊

旧融合算子摘要同时改变多条路径，不能用于本卡公式替换的因果证明。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/activation_int64.json#/C002`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/activation_int64_bits.json`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/activation_int64/kernel/probe.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/activation_int64/kernel/pybind11.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/run_npu.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
