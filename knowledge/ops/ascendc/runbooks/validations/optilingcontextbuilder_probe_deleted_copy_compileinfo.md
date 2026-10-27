---
type: "Runbook"
title: TilingContext 独立探针配方（builder 双坑）
description: "用 gert::OpTilingContextBuilder 构造 standalone TilingContext 探针；Tensor 不可拷贝须传指针，Build 前必须喂 CompileInfo 否则得到空 context；附完整探针代码与编译链接命令。"
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingContext 独立探针配方（builder 双坑）


## 现象

在为 ThnnFusedGruCellGrad（GRU 单步融合反向算子，ascend950/arch35，ops-registry-invoke 算子仓流程）实现 host 侧公共 TilingFunc 之前，需要先验证 `gert::TilingContext` 的一批 API（SetBlockDim / SetTilingKey / SetScheduleMode / GetWorkspaceSizes / GetTilingData / GetAttrs / GetPlatformInfo）在 gtest UT 所用的 `gert::OpTilingContextBuilder` 构建的上下文上是否可用——这决定 tiling 实现的哪些写法能被 UT 覆盖。走完整 TilingUT 链路（构建 run 包 → 安装 → 编 gtest → 执行）一次要数分钟且变量多，于是写了一个 standalone 探针 `probe.cpp` 直接链 CANN 的 `libtiling_api` / `libmetadef` 等库快速迭代，先后踩了两个坑：

1. **坑 1（编译失败）**：`std::vector<gert::Tensor> ins = {in1, in2};` 直接编译失败，报错原文：
   ```
   probe.cpp:34:46: error: use of deleted function 'gert::Tensor::Tensor(const gert::Tensor&)'
      34 |     std::vector<gert::Tensor> ins = {in1, in2};
   ```
   根因是 `gert::TensorData` 的拷贝构造 `= delete`，`gert::Tensor` 的拷贝构造因此被隐式删除——初始化列表会拷贝元素，编译即失败。
2. **坑 2（静默空 context）**：修掉拷贝后程序运行输出 `PROBE: context is null`——`builder.Build()` 返回的 holder 上 `GetContext()` 为 nullptr，**没有任何报错信息**指示原因。
3. **修复**：builder 链式调用中补一行 `.CompileInfo(&compileInfo)`（`compileInfo` 是任意空 struct 对象，取地址即可），context 立即构建成功，全部 API 探测通过。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（host tiling 函数实现 + TilingUT）；也适用于任何需要在 UT 外快速验证 TilingContext API 行为的场合
- API 路径：`gert::OpTilingContextBuilder`（`base/context_builder/op_tiling_context_builder.h`）、`gert::TilingContext`（`exe_graph/runtime/tiling_context.h`）、`gert::TilingData::CreateCap` / `gert::ContinuousVector::Create`、`fe::PlatFormInfos`（`platform/platform_infos_def.h`）、`platform_ascendc::PlatformAscendC`（`tiling/platform/platform_ascendc.h`）
- 验证过的平台与 CANN 版本：Ascend 950（host 侧探针，`fe::PlatFormInfos` mock：Short_SoC_version=Ascend950 / NpuArch=3510）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）
- 适用粒度：`operator_family`（任何 host tiling 函数的 API 行为验证）
- 路线优先级：`recommended_fastpath`（写 tiling 前先探针验证 API 面，比跑完整 UT 链路快一个量级）
- 明确不适用/未验证：真机 GE 运行时构造的 TilingContext 行为（探针是 host 侧 mock，未验证，推断一致）；其他 CANN 版本的 builder 行为未验证

## 判别方法

- **坑 1 判别**：编译报 `use of deleted function 'gert::Tensor::Tensor(const gert::Tensor&)'` → 不要用初始化列表/值语义收集 `gert::Tensor`；用具名局部对象 + `std::vector<gert::Tensor*>` 传指针，且对象生命周期必须覆盖 `Build()` 与后续 context 使用（探针末尾 `(void)outs;` 显式保活）。
- **坑 2 判别**：`Build()` 后 `GetContext()==nullptr` 且无报错 → 检查 builder 链是否漏 `.CompileInfo(...)`；该槽位要求非空指针，空 struct 的地址即可满足。
- **平台参数判别**：`PlatformAscendC` 直接构造在 `ctx->GetPlatformInfo()` 返回的 `fe::PlatFormInfos*` 上（先用 `SetPlatformRes` + `SetCoreNumByCoreType` 填 mock），即可查询核数/UB 等平台参数——与 TilingUT 的 `InitPlatformInfo` 完全同款。

## 设计决策

- **为何用 standalone 探针而不是直接跑 UT**：完整 TilingUT 链路一次数分钟且混入构建/安装/gtest 编译多个变量；探针秒级编译运行、单变量隔离，适合逐一钉死 API 行为后再写实现（本次实现 33 条 UT 一次转绿，探针预先消除了 API 面的不确定性是关键前提之一）。
- **为何传 Tensor 指针**：`gert::TensorData` 拷贝构造 delete，值语义不可行；builder 的 `InputTensors`/`OutputTensors` 本来就收指针数组。
- **为何 DummyCompileInfo 可以是空 struct**：builder 只要求该槽位非空指针，本探针场景不消费其内容。
- **为何平台 mock 照抄 UT 的 InitPlatformInfo**：保证探针结论可直接迁移到 UT 环境（同一 mock 填法），探针说可用的 API 在 UT 里就可写。

## 系统信息

- CANN 9.0.0-beta.2，安装于 `/home/<user>/Ascend/cann-9.0.0-beta.2`（`cann` 符号链接指向）
- 头文件 include 路径：`-I${CANN}/x86_64-linux/include -I${CANN}/x86_64-linux/pkg_inc -I${CANN}/x86_64-linux/pkg_inc/base -I${CANN}/x86_64-linux/asc/include`
- 链接库：`-L${CANN}/lib64 -ltiling_api -lopp_registry -lmetadef -lplatform -lc_sec -ldl -lpthread -lunified_dlog`
- 运行环境：`LD_LIBRARY_PATH=${CANN}/lib64 ./probe`
- 编译器标志：`g++ -std=c++17 -fno-access-control -D_GLIBCXX_USE_CXX11_ABI=0`（与算子仓 host 侧编译一致使用旧 ABI）
- 相关文件：探针 `/tmp/opencode/probe_ctx/probe.cpp`、`probe2.cpp`（仅存在于开发过程，作为本卡证据恢复）

## 完整代码

最终版探针（由初次写入 + 三次修复编辑组装恢复，已标注组装来源；输出为实测原样）：

```cpp
// Probe: does OpTilingContextBuilder-built TilingContext support SetScheduleMode?
#include <cstdio>
#include <cstring>
#include <vector>
#include "graph/ascend_string.h"
#include "graph/types.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "platform/platform_infos_def.h"

static gert::StorageShape MakeShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape s;
    for (int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

int main()
{
    gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    // 坑1修复：具名对象 + 指针数组（初始化列表拷贝 Tensor 编译失败）
    gert::Tensor in1(MakeShape({4, 8}), fmt, gert::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor in2(MakeShape({4, 40}), fmt, gert::kOnHost, ge::DT_FLOAT, nullptr);
    std::vector<gert::Tensor *> inPtrs = {&in1, &in2};
    std::vector<gert::Tensor> outs;
    for (int i = 0; i < 5; ++i) {
        outs.emplace_back(MakeShape({0}), fmt, gert::kOnHost, ge::DT_FLOAT, nullptr);
    }
    std::vector<gert::Tensor *> outPtrs;
    for (auto &t : outs) {
        outPtrs.push_back(&t);
    }
    // keep tensors alive
    (void)outs;

    auto tilingDataBuf = gert::TilingData::CreateCap(8192);
    auto wsBuf = gert::ContinuousVector::Create<size_t>(16);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::OpTilingContextBuilder builder;
    struct DummyCompileInfo {} compileInfo;   // 坑2修复：CompileInfo 槽位必填（空 struct 即可）
    builder.OpType(ge::AscendString("ThnnFusedGruCellGrad"))
        .OpName(ge::AscendString("ThnnFusedGruCellGrad"))
        .IONum(2, 5)
        .InputTensors(inPtrs)
        .OutputTensors(outPtrs)
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .TilingData(reinterpret_cast<gert::TilingData *>(tilingDataBuf.get()))
        .Workspace(reinterpret_cast<gert::ContinuousVector *>(wsBuf.get()));
    builder.AppendAttr(true);

    auto holder = builder.Build();
    gert::TilingContext *ctx = holder.GetContext();
    if (ctx == nullptr) {
        printf("PROBE: context is null\n");
        return 1;
    }
    printf("PROBE: GetBlockDim()=%u\n", ctx->GetBlockDim());
    auto r1 = ctx->SetBlockDim(3);
    printf("PROBE: SetBlockDim(3) ret=%d GetBlockDim()=%u\n", (int)r1, ctx->GetBlockDim());
    auto r2 = ctx->SetTilingKey(7);
    printf("PROBE: SetTilingKey(7) ret=%d GetTilingKey()=%lu\n", (int)r2, (unsigned long)ctx->GetTilingKey());
    auto r3 = ctx->SetScheduleMode(1);
    printf("PROBE: SetScheduleMode(1) ret=%d GetScheduleMode()=%u\n", (int)r3, ctx->GetScheduleMode());
    size_t *ws = ctx->GetWorkspaceSizes(1);
    printf("PROBE: GetWorkspaceSizes(1)=%p\n", (void *)ws);
    if (ws != nullptr) {
        ws[0] = 12345;
        printf("PROBE: ws[0]=%zu\n", ws[0]);
    }
    auto *td = ctx->GetTilingData<gert::TilingData>();
    printf("PROBE: GetTilingData<gert::TilingData>=%p\n", (void *)td);
    // attrs
    const auto *attrs = ctx->GetAttrs();
    printf("PROBE: GetAttrs()=%p\n", (const void *)attrs);
    if (attrs != nullptr) {
        const bool *b = attrs->GetAttrPointer<bool>(0);
        printf("PROBE: attr0(bool)=%p val=%d\n", (const void *)b, b ? (int)*b : -1);
    }
    // platform info
    auto *pi = ctx->GetPlatformInfo();
    printf("PROBE: GetPlatformInfo()=%p\n", (void *)pi);
    return 0;
}
```

编译并运行（实测命令原样）：

```bash
CANN=/home/<user>/Ascend/cann-9.0.0-beta.2; g++ -std=c++17 -fno-access-control -D_GLIBCXX_USE_CXX11_ABI=0 \
  -o probe probe.cpp \
  -I${CANN}/x86_64-linux/include -I${CANN}/x86_64-linux/pkg_inc -I${CANN}/x86_64-linux/pkg_inc/base \
  -I${CANN}/x86_64-linux/asc/include \
  -L${CANN}/lib64 -ltiling_api -lopp_registry -lmetadef -lplatform -lc_sec -ldl -lpthread -lunified_dlog \
  && LD_LIBRARY_PATH=${CANN}/lib64 ./probe
```

平台 mock 查询段（probe2 在同一 builder 流程后追加，与 TilingUT 的 InitPlatformInfo 同款填法）：

```cpp
    // mirror UT InitPlatformInfo
    fe::PlatFormInfos *pi = ctx->GetPlatformInfo();
    std::map<std::string, std::string> socInfos;
    socInfos["ai_core_cnt"] = "48";
    socInfos["cube_core_cnt"] = "24";
    socInfos["vector_core_cnt"] = "48";
    socInfos["core_type_list"] = "AICore";
    socInfos["l2_size"] = "33554432";
    pi->SetPlatformRes("SoCInfo", socInfos);
    std::map<std::string, std::string> aicoreSpec;
    aicoreSpec["ub_size"] = "262144";
    aicoreSpec["l0_a_size"] = "65536";
    aicoreSpec["l0_b_size"] = "65536";
    aicoreSpec["l0_c_size"] = "131072";
    aicoreSpec["l1_size"] = "524288";
    pi->SetPlatformRes("AICoreSpec", aicoreSpec);
    pi->SetCoreNumByCoreType("AICore");
    std::map<std::string, std::string> socVersion;
    socVersion["Short_SoC_version"] = "Ascend950";
    socVersion["NpuArch"] = "3510";
    pi->SetPlatformRes("version", socVersion);

    platform_ascendc::PlatformAscendC plat(pi);
    printf("PROBE2: coreNumAiv=%u\n", plat.GetCoreNumAiv());
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    printf("PROBE2: ubSize=%lu\n", (unsigned long)ub);
    printf("PROBE2: libApiWorkSpaceSize=%lu\n", (unsigned long)plat.GetLibApiWorkSpaceSize());
    // storage shape read path
    auto *shp0 = ctx->GetInputShape(0);
    auto g = shp0->GetStorageShape();   // g 是 gert::Shape：用 GetShapeSize()，没有 GetStorageShape()
    printf("PROBE2: in0 rank=%zu dims=[%ld,%ld] size=%ld\n", g.GetDimNum(), (long)g.GetDim(0), (long)g.GetDim(1),
           (long)g.GetShapeSize());
    auto *shp1 = ctx->GetInputShape(1);
    auto g1 = shp1->GetStorageShape();
    printf("PROBE2: in1 rank=%zu dims=[%ld,%ld]\n", g1.GetDimNum(), (long)g1.GetDim(0), (long)g1.GetDim(1));
    printf("PROBE2: in0 dtype=%d in1 dtype=%d\n", (int)ctx->GetInputTensor(0)->GetDataType(),
           (int)ctx->GetInputTensor(1)->GetDataType());
```

（probe2 需额外 `#include "tiling/platform/platform_ascendc.h"` 与 `<map>`/`<string>`。）

## 性能影响

无性能故事（本卡为 host 侧 API 验证配方，不涉及 kernel 执行）。

## 预防

- 写 host tiling 实现前，把要用到的 TilingContext API 列成清单，先用本探针逐项钉死行为（尤其 SetScheduleMode 这类依赖 output 槽分配的 API），再写实现。
- 收集 `gert::Tensor` 一律用指针；对象生命周期覆盖 `Build()` 之后的所有使用。
- `OpTilingContextBuilder` 链式调用最后自查一遍：`OpType / OpName / IONum / InputTensors / OutputTensors / CompileInfo / PlatformInfo / TilingData / Workspace` 九段齐全再 `Build()`。

## 验证信号

本卡信号来自 ThnnFusedGruCellGrad 公共 TilingFunc 实现会话中的独立 C++ 探针（OpTilingContextBuilder + PlatformAscendC mock），全部为本会话内实测输出：

- 坑 1 现象：`error: use of deleted function 'gert::Tensor::Tensor(const gert::Tensor&)'`（来源平台：950，host 探针；证据等级：tool_observed）
- 坑 2 现象：`PROBE: context is null`（缺 CompileInfo 时 Build() 静默返回空 context）（来源平台：950，host 探针；证据等级：tool_observed）
- 修复后 API 面实测输出原样：`GetBlockDim()=1598335808`（未初始化值）→ `SetBlockDim(3) ret=0 GetBlockDim()=3`、`SetTilingKey(7) ret=0 GetTilingKey()=7`、`SetScheduleMode(1) ret=0 GetScheduleMode()=1`、`GetWorkspaceSizes(1)` 非空可写、`GetTilingData<gert::TilingData>` 非空、`attr0(bool) val=1`（AppendAttr(true) 读回）、`GetPlatformInfo()` 非空（来源平台：950，host 探针 mock；证据等级：tool_observed）
- 平台 mock 查询实测：`coreNumAiv=48`、`ubSize=262144`、`libApiWorkSpaceSize=16777216`、`in0 rank=2 dims=[4,8] size=32`、`in1 rank=2 dims=[4,40]`、`in0 dtype=0 in1 dtype=0`（来源平台：950，host 探针 mock（Short_SoC_version=Ascend950）；证据等级：tool_observed）

## 失效边界

- 探针结论限于 builder 构建的 context；真机 GE 运行时构造的 TilingContext 行为未验证（推断一致，未验证）。
- `GetLibApiWorkSpaceSize()=16777216` 是 mock 未设置该项时走到的默认值路径；真机值未观测。
- CANN 9.0.0-beta.2 验证；其他版本 builder 是否同样强制 CompileInfo、Tensor 是否同样禁拷贝未验证。
- 探针只构造了 2 输入/5 输出/1 个 bool attr 的 context；其他 IONum/attr 类型组合未逐一探测（机制上同链路，推断可用）。

## 坏实践

- `std::vector<gert::Tensor> ins = {in1, in2};`——初始化列表拷贝，`gert::TensorData` 拷贝构造 delete，编译失败。
- builder 链漏 `.CompileInfo(...)`——`Build()` 静默返回空 context，探针只看到 nullptr，无任何错误输出，极易误判为 builder/API 本身不可用。
- 在 `gert::Shape`（`GetStorageShape()` 的返回值）上再调 `GetStorageShape()`——Shape 没有这个方法（那是 `StorageShape` 的成员），Shape 上是 `GetShapeSize()`；本会话探针实测编译报 `'struct gert::Shape' has no member named 'GetStorageShape'`。

## 误诊

无。


