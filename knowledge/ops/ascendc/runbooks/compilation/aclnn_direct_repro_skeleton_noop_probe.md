---
type: "Runbook"
title: "aclnn 直连复现骨架 + dtype 全编码往返探针"
description: "为 vendor 自定义算子写单文件 C++ 直连 aclnn 复现程序（g++ 直接链 op_api，绕过工程 CMake），配合 no-op reduce + 全编码枚举探针设计，把 kernel 内 dtype 转换从算子语义中剥离成可枚举验证的对象。"
tags: [task.debugging, task.testing, topic.implementation.api_layer.c_api, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# aclnn 直连复现骨架 + dtype 全编码往返探针

## 现象

在 Ascend950PR（CANN 9.2.0）上做 BenchHarnessScatter（vendor 自定义算子）黑盒失败用例修复时，需要一个能快速迭代的复现手段：TTK 全量回归一轮约 17~34 分钟、单用例 dump 也要分钟级，而且 TTK 用随机数据无法控制编码角落。会话中沉淀出一套"单文件 C++ 直连 aclnn"骨架：不进工程 CMakeLists、g++ 一条命令编译、直接链 vendor 算子的 `op_api` 动态库，几十秒完成"改用例→编译→上板→比对"一轮；再配合"no-op reduce（amax + 全 -Inf updates）+ 全编码枚举 data"的探针设计，把 kernel 内的 fp16 位级转换变成可全量枚举验证的对象，一举定位了次正规移位缺陷（见 [FloatToHalf 次正规阈值卡](../precision/floattohalf_subnormal_threshold_exp102.md)）。

## 适用场景

- 工程形态：AscendC 自定义算子开发/黑盒修复中的复现与诊断（kernel 直调类工程同样适用思路）。
- API 路径：aclnn 两段式调用（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）、`aclCreateTensor`（带 view strides 与 storage shape）、vendor `op_api/lib` 动态链接。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何 vendor 自定义算子的复现程序骨架与探针设计方法。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：需要多输入流/动态 shape 的场景未覆盖；`aclCreateTensor` 的 stride 语义按本骨架的"view stride + storage shape"用法验证，其他用法未验证。

## 设计决策

- **为什么绕过工程 CMake 直连 g++**：工程构建（`bash operators/build.sh`）要重编 kernel（分钟级）且产出的测试工程挂 CMake；复现程序只调 host 侧 aclnn API，不需要 kernel 参与编译，单文件 + 一条 g++ 命令把迭代周期压到几十秒。
- **为什么手工算 view strides / storage shape**：`aclCreateTensor(dims, rank, dtype, strides, 0, format, storageDims, rank, ptr)` 需要同时给 view strides（按 C-order 连续张量算）与 storage shape；骨架里用一个倒序循环统一生成（见完整代码），避免 8 维手写。
- **字符串属性是 `char*` 非 `const char*`**：生成的 aclnn API 里 reduce 这类字符串属性参数是可变指针，传字面量会有告警且部分编译器拒绝；用 `char reduce[]="amax";` 局部数组。
- **两段式调用顺序**：先 `GetWorkspaceSize`（此时 host 侧 TilingFunc 会跑完所有校验——非法形状在这里就被拒，见 [ACLNN 561002 scatter 形状校验卡](../validations/aclnn_561002_scatter_non_dim_shape_violation.md)），再按返回的 wsSize 分配 workspace 并调执行函数。
- **no-op reduce 探针**：amax + 全 -Inf updates 数学上是恒等映射，输出=输入当且仅当 kernel 的读→转换→写链路恒等——把"dtype 转换正确性"从"scatter 语义正确性"中剥离；同理可用 update + 全零索引构造纯拷贝探针。
- **全编码枚举优于随机值**：fp16 次正规区间只有 2048 个编码（含符号），全枚举零死角且失败输出直接给出出错编码；会话中受控随机数据恰好没命中 0x0001，若停在随机复现就会误判 kernel 正确。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1.6），CANN 9.2.0（`ASCEND_HOME_PATH` 指向安装目录），vendor 算子包已 `bash operators/build.sh` 安装到 `opp/vendors/<Vendor>`。
- 目录布局：复现程序放 `operators/<pkg>/tests/aclnn/`（与 `aclnn_<op>.h` 同源），产物放 `build_repro/`（用后删除）。
- 相关文件：`repro_failures.cpp`（多用例复现：多块拷贝/重复索引/拷贝+update/万级重复）、`repro_fp16.cpp`（受控数据 fp16 模式）、`repro_subnormal.cpp`（全编码次正规往返探针，见上卡）、`repro_adjacent.cpp`（单元素写邻接腐蚀对照）。
- 编译与运行（verbatim 取自会话中实际执行的单条命令；换算子时替换 vendors 目录名、头文件名与源文件名即可）：

```bash
g++ -std=c++17 -O2 -I$ASCEND_HOME_PATH/x86_64-linux/include -I$ASCEND_HOME_PATH/include -I$ASCEND_HOME_PATH/opp/vendors/BenchHarnessScatter/op_api/include repro_subnormal.cpp -o build_repro/repro_subnormal -L$ASCEND_HOME_PATH/lib64 -L$ASCEND_HOME_PATH/opp/vendors/BenchHarnessScatter/op_api/lib -lascendcl -lnnopbase -lcust_opapi 2>&1 | grep error | head; export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/BenchHarnessScatter/op_api/lib/:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH; ./build_repro/repro_subnormal 2>&1
```

## 完整代码

`repro_fp16.cpp`（verbatim 取自会话中的写入事件——单文件直连骨架 + 受控 fp16 数据 + no-op amax 探针；把算子 API 名换成目标算子即可复用）：

```cpp
// tests/aclnn/repro_fp16.cpp — fp16-specific repro for the fp16 amax failure pattern
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_bench_harness_scatter.h"

#define CHECK_ACL(x) do { aclError _r=(x); if(_r!=ACL_SUCCESS){fprintf(stderr,"ACL err %d %s:%d\n",_r,__FILE__,__LINE__);return -1;} } while(0)
#define CHECK_ACLNN(x) do { aclnnStatus _r=(x); if(_r!=0){fprintf(stderr,"ACLNN err %d %s:%d\n",_r,__FILE__,__LINE__);return -1;} } while(0)

// fp16 helpers
static uint16_t float_to_fp16(float f) {
    uint32_t u; memcpy(&u,&f,4);
    uint32_t sign=(u>>31)&1, exp=(u>>23)&0xFF, mant=u&0x7FFFFF;
    uint16_t h;
    if(exp==0xFF) h=(uint16_t)((sign<<15)|0x7C00|(mant?((mant>>13)|0x200):0));
    else if(exp>=143) h=(uint16_t)((sign<<15)|0x7C00);
    else if(exp>=113){ uint32_t he=exp-127+15,tm=mant>>13,dr=mant&0x1FFF;
        if(dr>0x1000||(dr==0x1000&&(tm&1)))tm++;
        if(tm>0x3FF){tm=0;he++;}
        h=(he>=31)?(uint16_t)((sign<<15)|0x7C00):(uint16_t)((sign<<15)|(he<<10)|tm);
    } else h=(uint16_t)(sign<<15);
    return h;
}
static float fp16_to_float(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(uint32_t)(h&0x7C00)>>10, mant=h&0x03FF, f;
    if(exp==0){ if(mant){ while(!(mant&0x400)){mant<<=1;exp--;} f=sign|((exp+112)<<23)|((mant&0x3FF)<<13);} else f=sign; }
    else if(exp==31) f=sign|0x7F800000|(mant<<13);
    else f=sign|((exp+112)<<23)|(mant<<13);
    float r; memcpy(&r,&f,4); return r;
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    // Test: fp16 data, multi-chunk copy (no scatter), then amax with -Inf (no-op scatter)
    // Shape (7,6,7,8,2,5,8,3) = 564480 elements, same as the failing fp16 amax case
    const int64_t N=564480;
    int64_t dDims[8]={7,6,7,8,2,5,8,3};
    int64_t iDims[8]={3,3,3,4,1,2,4,1}; // 864 indices
    int64_t uDims[8]={3,3,3,4,1,2,4,1};

    // Data: small fp16 values in [-0.01, 0.01]
    std::vector<uint16_t> data(N);
    for(int64_t i=0;i<N;i++){
        float v = ((i*7)%2000 - 1000) * 0.00001f; // -0.01 to 0.01
        data[i]=float_to_fp16(v);
    }
    // Indices: all 0, Updates: all -Inf
    std::vector<int64_t> idx(864, 0);
    std::vector<uint16_t> upd(864, 0xFC00); // -Inf in fp16

    // Expected = data (amax with -Inf is no-op)
    std::vector<uint16_t> expected = data;

    int64_t dStr[8], dSto[8], iStr[8], iSto[8], uStr[8], uSto[8], oStr[8], oSto[8], oDims[8];
    dStr[7]=1;iStr[7]=1;uStr[7]=1;oStr[7]=1;
    for(int k=6;k>=0;k--){dStr[k]=dStr[k+1]*dDims[k+1];iStr[k]=iStr[k+1]*iDims[k+1];uStr[k]=uStr[k+1]*uDims[k+1];oStr[k]=oStr[k+1]*dDims[k+1];}
    for(int k=0;k<8;k++){dSto[k]=dDims[k];iSto[k]=iDims[k];uSto[k]=uDims[k];oDims[k]=dDims[k];oSto[k]=dSto[k];}

    size_t dB=N*2, iB=864*8, uB=864*2, oB=N*2;
    std::vector<uint16_t> out(N, 0);

    void *dD,*dI,*dU,*dO;
    CHECK_ACL(aclrtMalloc(&dD,dB,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dI,iB,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dU,uB,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dO,oB,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dD,dB,data.data(),dB,ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dI,iB,idx.data(),iB,ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dU,uB,upd.data(),uB,ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *tData=aclCreateTensor(dDims,8,ACL_FLOAT16,dStr,0,ACL_FORMAT_ND,dSto,8,dD);
    aclTensor *tIdx=aclCreateTensor(iDims,8,ACL_INT64,iStr,0,ACL_FORMAT_ND,iSto,8,dI);
    aclTensor *tUpd=aclCreateTensor(uDims,8,ACL_FLOAT16,uStr,0,ACL_FORMAT_ND,uSto,8,dU);
    aclTensor *tOut=aclCreateTensor(oDims,8,ACL_FLOAT16,oStr,0,ACL_FORMAT_ND,oSto,8,dO);

    int64_t dim=0; char reduce[]="amax";
    uint64_t wsSize=0; aclOpExecutor*exec=nullptr;
    CHECK_ACLNN(aclnnBenchHarnessScatterGetWorkspaceSize(tData,tIdx,tUpd,dim,reduce,tOut,&wsSize,&exec));
    aclrtStream stream; CHECK_ACL(aclrtCreateStream(&stream));
    void*ws=nullptr; if(wsSize>0) CHECK_ACL(aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLNN(aclnnBenchHarnessScatter(ws,wsSize,exec,stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(out.data(),oB,dO,oB,ACL_MEMCPY_DEVICE_TO_HOST));

    int64_t mism=0;
    for(int64_t i=0;i<N;i++){
        if(out[i]!=expected[i]){
            if(mism<5) fprintf(stderr,"MISMATCH[%ld]: got=%g exp=%g\n",i,fp16_to_float(out[i]),fp16_to_float(expected[i]));
            mism++;
        }
    }
    fprintf(stdout,"fp16 amax -Inf test: %s  mismatch=%ld/%ld\n",mism==0?"PASSED":"FAILED",mism,N);

    if(ws)aclrtFree(ws);aclDestroyTensor(tData);aclDestroyTensor(tIdx);aclDestroyTensor(tUpd);aclDestroyTensor(tOut);
    aclrtFree(dD);aclrtFree(dI);aclrtFree(dU);aclrtFree(dO);aclrtDestroyStream(stream);
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
```

（本文件自带的 `float_to_fp16` 只覆盖 normal 区间——它正是"受控数据没命中次正规编码"的原因之一；全编码版本把 data 换成 0x0001..0x03FF/0x8001..0x83FF 枚举，见 [FloatToHalf 次正规阈值卡](../precision/floattohalf_subnormal_threshold_exp102.md)的探针代码。）

## 验证信号

以下信号来自该算子的一次黑盒失败用例修复会话：直连骨架全程支撑复现-修复-回归闭环。

- 骨架编译运行：单文件 g++ 编译 exit=0，上板执行正常返回比对结果（多块拷贝、重复索引、拷贝+update、万级重复、邻接腐蚀、全编码次正规六类探针均按此骨架构建）（来源平台：950；证据等级：tool_observed）。
- no-op 探针有效性：受控数据 fp16 amax -Inf 564,480 元素 0 失配（未命中缺陷编码）；全编码次正规探针同骨架下精确复现 0x0001→0x2000 单点失配，修复后 0/2050（来源平台：950；证据等级：tool_observed）。
- 效率事实：TTK 全量一轮约 17~34 分钟；直连骨架"改-编-跑"一轮约几十秒（编译+上板执行）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 骨架的 `float_to_fp16` 辅助函数只实现 normal 区间（次正规一律归 0），做全编码验证时不能依赖它构造期望值——期望值应直接用原始编码（位级恒等）。
- 直连程序只覆盖 host 侧 API 调用路径；kernel 是否被真正执行取决于算子实现（本算子无 bypass），对有 fallback/静态 shape 分支的算子需先确认走的是目标 kernel。
- vendor 库路径（`opp/vendors/<Vendor>/op_api/lib`）要求算子已 build 安装；`LD_LIBRARY_PATH` 缺该路径会在加载期报符号缺失。

**未决项**：

- 无（骨架在本会话内多探针复用并全部有效）。
## 坏实践

- 复现只写随机数据用例——编码角落（次正规、边界值）永远命中不了；用 no-op reduce + 全枚举把转换变成可判定问题。
- 把复现程序挂进工程 CMake——每次改用例都要过完整构建，迭代周期膨胀两个数量级。
