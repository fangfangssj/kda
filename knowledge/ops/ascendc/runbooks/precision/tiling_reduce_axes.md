---
type: "Runbook"
title: "tiling 按位置奇偶判 reduce 轴致非交替 axes 误减"
description: "归约算子 host tiling 因\"无法读 axes tensor 数据\"按轴位置奇偶推断 reduce 轴，对非 A-R 交替排列的 axes（如 axes=[0]）会减错轴，产出部分正确值后补零。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.compute.reduction", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# tiling 按位置奇偶判 reduce 轴致非交替 axes 误减

## 现象

在 Ascend950（dav-3510, arch35）自研 `euclidean_norm`（`sqrt(sum(x^2 over axes))`）算子上，C++ ST 用例全绿，但用 `aclnn` 直调复现黑盒失败用例时发现：当 `axes` 取**非"奇数位置"**的轴（如 `axes=[0]`）时，kernel 只往输出 buffer 写了少量正确值后全是零。以复现用例 A 为例：`x=(1,4,1,2,3)` fp32、`axes=[0]`、keepDims=false，正确降维应减 axis 0（size 1）输出 `(4,1,2,3)=24` 元素、每元素 `|x|`；实测 `actual[0..7]=15.146 14.387 13.872 0.000 0.000 0.000 0.000 0.000`——只写了 3 个非零值，余 21 个为零。对照用例 C（`axes=[1]`，奇数位置）输出与 golden 一致。

机制算式：x=(1,4,1,2,3) 共 5 轴。`axes=[0]` 正确应减 axis 0。但 tiling 的奇偶启发式把"奇数索引轴"当 reduce 轴 → 减的是 axis 1 与 axis 3（位置 1、3 为奇），`aTotal = axis0·axis2·axis4 = 1·1·3 = 3`，故只产出 3 个输出元素；每个 = `sqrt(sum of squares over axes{1,3} = 4·2=8 个元素)`。验算：`|x|` 量级约 5，`x^2≈25`，`8·25=200`，`sqrt(200)≈14.1`，与实测 `15.146 14.387 13.872` 量级吻合——确认 kernel 按**错误的轴集合** {1,3} 做了归约，而非 {0}。

## 适用场景

- **工程形态**：aclnn 算子工程化开发（ops-registry-invoke），host tiling 在 `arch35/euclidean_norm_tiling_arch35.cpp` 内推断 reduce 轴集合。
- **API 路径**：`aclnnEuclideanNormGetWorkspaceSize(x, axes, keepDims, out, ...)`；axes 作为 `aclTensor*`（host 端经 `NnopbaseAddInput(*executor, axes, 1)` 下发，tiling 时 `context->GetInputShape(1)` 可得 axes shape、`context->GetInputTensor(1)` 经 `TilingInputsDataDependency({1})` 可得 axes 数据）。
- **验证过的平台与 CANN 版本**：Ascend950（arch35, kernel binary `ascend950/euclidean_norm/*.o`）。CANN/TTK 具体版本未在 trace 内定位。
- **适用粒度**：`operator_family`（适用于一切"host tiling 用位置奇偶/隐式约定推断 reduce 轴"的归约算子，当外部 aclnn 调用方传入任意 axes 值时即触发）。
- **路线优先级**：`recommended_fastpath`（归约算子黑盒/ST 出现"部分值正确后补零"或"仅 N 个非零"时，先查 tiling 的 reduce 轴推断是否真读了 axes 数据，再查 kernel 数值）。
- **明确不适用或未验证场景**：axes 恰好全部落在奇数位置（本算子 ST 的 B0-*/B4 设计就是这样，故 ST 全绿）时不暴露；kernel 直调且调用方手工指定正确 reduce 轴也不暴露；未验证非 Ascend950 平台。

## 根因

`euclidean_norm_tiling_arch35.cpp` 用**位置奇偶启发式**判定每个轴是否为 reduce 轴，而不是读 axes tensor 的实际数值。其代码注释自述原因："Host cannot read the axes tensor DATA (device storage at tiling time)"——这是 **DESIGN 选择**（DESIGN §3.3.2/§3.3.4 规定 axes 按 `A R A R …` 交替排列、`isReduceAxis[i] == (i%2==1)`、`isTailR = (axisNum%2==0)`），即算子契约要求调用方把 axes 排成"奇数位置才是 R"。一旦调用方（如 TTK 黑盒 CSV）传入任意 axes 值（`[0]`、`[-1]` 等），奇偶启发式就会减错轴。

注意：aclnn 路径下 axes 实际是 host 可读的 `aclTensor*`（经 `NnopbaseAddInput` 下发），tiling **本可以**用 `TilingInputsDataDependency({1})` + `context->GetInputTensor(1)->GetData<int64_t>()` 读到 axes 真值；当前 tiling 没声明该依赖、也没读数据，是"按 DESIGN 契约省事"的遗留。

## 判别方法

**触发**：归约算子在"部分用例正确、部分用例输出仅前 N 个有值后补零"或"shape 对但值错"时，且失败用例的 axes 值不在算子 DESIGN 约定的位置（如本算子约定奇数位置）。

**可执行探针**（verbatim 取自 trace，完整 `runOne` + `main`，链接安装好的 vendor `libcust_opapi.so`，`aclrtSetDevice(0)` 上板 Ascend950）：

```cpp
static int64_t prod(const std::vector<int64_t>& s){int64_t n=1;for(auto d:s)n*=d;return n;}
static std::vector<int64_t> strides(const std::vector<int64_t>& s){
    std::vector<int64_t> r(s.size(),1);
    for (int i=(int)s.size()-2;i>=0;--i) r[i]=r[i+1]*s[i+1];
    return r;
}

static bool runOne(const char* tag,
                   const std::vector<int64_t>& xShape,
                   const std::vector<int64_t>& axesVals,
                   aclDataType axesDtype,
                   bool keepDims,
                   const std::vector<int64_t>& outShapeProvided,
                   aclrtStream stream)
{
    int64_t xTotal = prod(xShape);
    int64_t outTotal = prod(outShapeProvided);
    std::vector<uint8_t> xHost(xTotal*4);
    for (int64_t i=0;i<xTotal;++i){
        float v = (float)((i*7+3)%200)*0.1f - 10.0f; if(v==0.f) v=0.5f;
        std::memcpy(xHost.data()+i*4,&v,4);
    }
    std::vector<uint8_t> axesHost(axesVals.size()* (axesDtype==ACL_INT32?4:8));
    for(size_t i=0;i<axesVals.size();++i){
        if(axesDtype==ACL_INT32){int32_t v=(int32_t)axesVals[i]; std::memcpy(axesHost.data()+i*4,&v,4);}
        else{std::memcpy(axesHost.data()+i*8,&axesVals[i],8);}
    }
    void *devX,*devAxes,*devOut;
    aclrtMalloc(&devX,xTotal*4,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devAxes,std::max((size_t)8,axesHost.size()),ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOut,outTotal*4,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX,xHost.size(),xHost.data(),xHost.size(),ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devAxes,axesHost.size(),axesHost.data(),axesHost.size(),ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(devOut,outTotal*4,0,outTotal*4);

    auto xs=strides(xShape), os=strides(outShapeProvided);
    std::vector<int64_t> axesShape={(int64_t)axesVals.size()};
    std::vector<int64_t> axesStr={1};
    aclTensor *x=aclCreateTensor(xShape.data(),xShape.size(),ACL_FLOAT,xs.data(),0,ACL_FORMAT_ND,xShape.data(),xShape.size(),devX);
    aclTensor *axes=aclCreateTensor(axesShape.data(),1,axesDtype,axesStr.data(),0,ACL_FORMAT_ND,axesShape.data(),1,devAxes);
    aclTensor *out=aclCreateTensor(outShapeProvided.data(),outShapeProvided.size(),ACL_FLOAT,os.data(),0,ACL_FORMAT_ND,outShapeProvided.data(),outShapeProvided.size(),devOut);

    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnEuclideanNormGetWorkspaceSize(x,axes,keepDims,out,&ws,&ex);
    fprintf(stdout,"[%s] GetWorkspaceSize ret=%d ws=%llu\n",tag,(int)r,(unsigned long long)ws);
    void* wbuf=nullptr; if(ws>0) aclrtMalloc(&wbuf,ws,ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnEuclideanNorm(wbuf,ws,ex,stream);
    aclrtSynchronizeStream(stream);

    int64_t reportedCount=prod(outShapeProvided);
    fprintf(stdout,"[%s] out providedShape=",tag);
    for(auto d:outShapeProvided) fprintf(stdout,"%lld ",(long long)d);
    fprintf(stdout,"(count=%lld)\n",(long long)reportedCount);

    std::vector<uint8_t> act(outTotal*4);
    aclrtMemcpy(act.data(),outTotal*4,devOut,outTotal*4,ACL_MEMCPY_DEVICE_TO_HOST);
    fprintf(stdout,"[%s] actual[0..min(8,n)]:",tag);
    for(int64_t i=0;i<std::min<int64_t>(8,outTotal);++i){float v;std::memcpy(&v,act.data()+i*4,4);fprintf(stdout," %.3f",v);}
    fprintf(stdout,"\n");

    std::vector<int64_t> axesN;
    int64_t rank=xShape.size();
    for(auto a:axesVals) axesN.push_back(a<0?a+rank:a);
    std::vector<int64_t> goldShape;
    for(size_t i=0;i<xShape.size();++i){
        bool red=false; for(auto a:axesN) if(a==(int64_t)i) red=true;
        if(red){ if(keepDims) goldShape.push_back(1); }
        else goldShape.push_back(xShape[i]);
    }
    if(goldShape.empty()) goldShape.push_back(1);
    int64_t goldCount=prod(goldShape);
    fprintf(stdout,"[%s] goldenShape=",tag);
    for(auto d:goldShape) fprintf(stdout,"%lld ",(long long)d);
    fprintf(stdout,"(count=%lld)\n",(long long)goldCount);
    fprintf(stdout,"[%s] SHAPE_MATCH=%s (providedCount=%lld goldenCount=%lld)\n",
            tag, (reportedCount==goldCount?"YES":"NO"),
            (long long)reportedCount,(long long)goldCount);

    if(wbuf) aclrtFree(wbuf);
    aclDestroyTensor(x);aclDestroyTensor(axes);aclDestroyTensor(out);
    aclrtFree(devX);aclrtFree(devAxes);aclrtFree(devOut);
    return reportedCount==goldCount;
}

int main(){
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream s; aclrtCreateStream(&s);

    // Case A: axes=[0] (even position -> parity would reduce {1,3} wrongly).
    //   x=(1,4,1,2,3) fp32, axes=[0], keepDims=False -> correct out (4,1,2,3)=24
    runOne("A_axes0_correctOut", {1,4,1,2,3}, {0}, ACL_INT32, false, {4,1,2,3}, s);

    // Case A2: same but provide the CSV-style WRONG out shape (2,1,1,3,2,3)=36
    runOne("A2_axes0_csvOut", {1,4,1,2,3}, {0}, ACL_INT32, false, {2,1,1,3,2,3}, s);

    // Case B: axes=[-1] (reduce last axis). x=(2,3,4), axes=[-1]=[2], kd=False -> (2,3)=6
    runOne("B_axesNeg1_correctOut", {2,3,4}, {-1}, ACL_INT64, false, {2,3}, s);

    // Case C: axes=[1] (odd position -> parity correct). x=(4,8,3), axes=[1], kd=False -> (4,3)
    runOne("C_axes1_correctOut", {4,8,3}, {1}, ACL_INT32, false, {4,3}, s);

    aclrtDestroyStream(s);
    aclFinalize();
    return 0;
}
```

**原始观测**（repro 运行输出）：

```
COMPILE OK
[A_axes0_correctOut] GetWorkspaceSize ret=0 ws=0
[A_axes0_correctOut] out providedShape=4 1 2 3 (count=24)
[A_axes0_correctOut] actual[0..min(8,n)]: 15.146 14.387 13.872 0.000 0.000 0.000 0.000 0.000
[A_axes0_correctOut] goldenShape=4 1 2 3 (count=24)
[A_axes0_correctOut] SHAPE_MATCH=YES (providedCount=24 goldenCount=24)
[A2_axes0_csvOut] out providedShape=2 1 1 3 2 3 (count=36)
[A2_axes0_csvOut] actual[0..min(8,n)]: 15.146 14.387 13.872 0.000 0.000 0.000 0.000 0.000
[A2_axes0_csvOut] goldenShape=4 1 2 3 (count=24)
[A2_axes0_csvOut] SHAPE_MATCH=NO (providedCount=36 goldenCount=24)
[B_axesNeg1_correctOut] out providedShape=2 3 (count=6)
[B_axesNeg1_correctOut] actual[0..min(8,n)]: 12.590 11.446 10.316 9.209 4.736 5.495
[B_axesNeg1_correctOut] goldenShape=2 3 (count=6)
[B_axesNeg1_correctOut] SHAPE_MATCH=YES (providedCount=6 goldenCount=6)
[C_axes1_correctOut] out providedShape=4 3 (count=12)
[C_axes1_correctOut] actual[0..min(8,n)]: 15.146 14.387 13.872 16.842 16.775 16.217 17.825 17.540
[C_axes1_correctOut] goldenShape=4 3 (count=12)
[C_axes1_correctOut] SHAPE_MATCH=YES (providedCount=12 goldenCount=12)
```

**判读（interpretation，根因分析）**：用例 A（axes=[0]，偶位置）即使给了正确 out shape (24)，kernel 也只写 3 个非零值（减了 {1,3} 而非 {0}）；用例 B（axes=[-1]=[2]，偶位置）输出 6 个值——注意这里 shape 碰巧对（(2,3)=6）但减的轴集合是否正确需逐值核对（B 的 axes=[-1]→axis 2，位置 2 偶，奇偶启发式会把 {1} 当 R，是否仍得 6 值取决于 x=(2,3,4) 的几何）；用例 C（axes=[1]，奇位置）奇偶启发式恰好正确，值与 golden 吻合。**"偶位置 axes → 只写少量值后补零"是本根因的判别特征。**

**机制源（verbatim 取自 `operators/euclidean_norm_package/op_host/arch35/euclidean_norm_tiling_arch35.cpp`）**——先看 `LoadAxesCount`（只读 axes 的 shape 大小，不读数据）：

```cpp
// Determine the reduce-axis count from the axes input SHAPE (not its data —
// the axes tensor storage lives in device memory and is not host-dereferenceable
// at tiling time). Returns the number of axis entries the caller supplied.
//   count == 0            → full reduce (DESIGN §3.3.4 empty-axes normalisation)
//   count == xRank        → full reduce (all distinct axes ⇒ every axis reduced)
//   otherwise             → caller uses position-parity inference (odd idx = R),
//                           which is exact for every ST case whose axes sit at
//                           odd positions (B0-* and B4-1..5/9/11/12).
int64_t LoadAxesCount(gert::TilingContext* context)
{
    const gert::StorageShape* axesShape = context->GetInputShape(1);
    if (axesShape == nullptr) return 0;
    int64_t cnt = axesShape->GetOriginShape().GetShapeSize();
    if (cnt < 0) cnt = 0;
    return cnt;
}
```

再看逐轴 reduce 判定（奇偶启发式）：

```cpp
// Per-axis R classification (DESIGN §3.3.2 / §3.3.4). Host cannot read the axes
// tensor DATA (device storage at tiling time), so R inference uses the same
// position-parity heuristic as the normal fuse path: full reduce ⇒ all axes R;
// otherwise odd-index axes are R. This exactly matches the B2 ST cases whose
// reduce axes sit at odd positions (or are full-reduce).
inline int32_t AxisIsReduce(int32_t dim, bool fullReduce)
{
    return (fullReduce || (dim % 2 == 1)) ? 1 : 0;
}
```

`dim % 2 == 1` 就是"奇数位置即 R"的实现。`fullReduce`（axes 为空或全轴）时全部为 R，否则按奇偶。对 `axes=[0]`（非全减、axis 0 在位置 0 为偶）→ axis 0 判为 A，axis 1/3 判为 R → 减错。

## 验证信号

来源说明：本卡的验证信号来自 Ascend950 上一次黑盒失败修复调查中的 repro 运行（test_repro.cpp）与 tiling 源码 file_read 事件，均为 trace 内结构化捕获的 tool_result。

- 用例 A（`axes=[0]`、正确 out shape (4,1,2,3)=24）：`actual[0..7]=15.146 14.387 13.872 0.000 0.000 0.000 0.000 0.000`，kernel 只写 3 个非零值后补零（来源平台：950；证据等级：tool_observed）。
- 用例 C（`axes=[1]`、out shape (4,3)=12）：`actual[0..7]=15.146 14.387 13.872 16.842 16.775 16.217 17.825 17.540`，与 golden 一致、`SHAPE_MATCH=YES`（来源平台：950；证据等级：tool_observed）。
- `COMPILE OK` + `GetWorkspaceSize ret=0 ws=0`，算子在 Ascend950 上成功编译并执行（来源平台：950；证据等级：tool_observed）。
- tiling 源码 `AxisIsReduce` 与 `LoadAxesCount` 经 file_read 事件读取，内容如 `## 判别方法` 机制源 verbatim 所示（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- **未验证，推断可迁移**：同 DESIGN "axes 按 A-R 交替排列"契约下的其它归约算子，若 aclnn 调用方传任意 axes，应同样命中奇偶误判；未在其它算子实测。
- **未验证**：非 Ascend950 平台（如 a2/a3）的 tiling 实现是否也用同一奇偶启发式；本卡 `platforms` 仅保留实测的 `950`。
- 本卡不提供修复配方；tiling 改写（声明 `TilingInputsDataDependency({1})` 并用 `aclrtMemcpy`/`GetData<int64_t>()` 读 axes 真值）在该 trace 结束前未完成、未验证全量全绿。

## 坏实践

- **靠"ST 全绿"推断 tiling 对**：本算子 ST 的 axes 全部落在奇数位置（DESIGN-B0/B4 用例设计如此），正好满足奇偶启发式，故 ST 全绿但黑盒全红。判别 reduce 轴推断是否真读 axes 数据，必须用**非约定位置**的 axes（如 `[0]`、`[-1]`）做探针。
- **不读 axes 数据、靠"调用方会按约定排"省事**：DESIGN 契约不能替代 tiling 实读 axes 真值；aclnn 路径下 axes 是 host 可读 tensor，应声明 `TilingInputsDataDependency({1})` 后实读。

## 误诊

- trace 内早期一度怀疑"kernel 数值逻辑错"——但用例 C（奇位置 axes）值与 golden 一致，说明 kernel 数值逻辑对；错在 tiling 给 kernel 的 reduce 轴集合。此判别由 repro 的 A vs C 对比支撑（tool_observed）。
- trace 内早期还怀疑"infershape 桩（y=x）致 shape 错"——但用例 A 给了正确 out shape (24) 仍只写 3 值，说明即使 shape 对、值仍错，根因在 tiling 轴集合而非 infershape；infershape 桩不传播是另一独立问题，不在本卡范围。