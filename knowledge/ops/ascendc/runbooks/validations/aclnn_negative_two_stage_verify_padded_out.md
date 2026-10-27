---
type: "Runbook"
title: "TTK无负向支持：aclnn两段式负向验证"
description: "TTK aclnn 模式不解析 expected-error 用例（源码核对无 remark/expect 逻辑）；负向（异常输入）验收需自写等价 aclnn 程序逐 case 断言返回码——第一段 GetWorkspaceSize 拦截判定，第二段对未拦截 case 在执行阶段复验，out 存储按 numel(x) 垫底防越界、view shape 保持用例原值。"
tags: ["task.testing", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# TTK无负向支持：aclnn两段式负向验证

## 现象

在 Ascend950PR（CANN 9.2.0）上把性能优化后的融合算子 FusedRmsNormSigmoidMul 集成回注册工程并验收：功能验收除了 18 条正向 L0（TTK aclnn 模式可直接跑），还有 17 条 L2 负向用例（`aclnnFusedRMSNormSigmoidMul_l2_exception.csv`，remark 列注明预期 `ACL_ERROR_*` 错误名，属 aclnn 接口层校验语义——dtype 域 7 条、format 域 2 条、输出 shape 2 条、空 tensor 2 条、多重约束 4 条）。先查 TTK 源码确认其对负向用例的支持：`ttk/core_modules/aclnn/acl_interface.py` 与 `testcase_manager/testcase_aclnn.py` 中无 remark/expected-error 解析——TTK 不支持 expected-error 用例，负向 case 无法用 TTK 断言"应被拒绝"。需要等价验证路径。验证程序开发过程中还遇到两个真实问题：① C++ 嵌套聚合初始化编译错（`could not convert '{{"case_name", {{{shape}, ACL_BOOL, ...}}}'`）；② 首条 case 正确返回 161002 后第二条段错误（exit=139），gdb 定位到 stride 计算里 `rank=0` 时 `rank-1` 无符号下溢。

## 适用场景

- 工程形态：注册算子工程（vendor 包已装入 CANN）的 aclnn 接口层负向验收；任何"预期返回错误码"的接口语义验证。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`（vendor 包 `op_api/include` 头文件 + `-lcust_opapi`）、`aclCreateTensor`（dims/strides/storageOffset/format/view 形参）、`aclrtMalloc`/`aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend950PR NPU 0，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（aclnn 接口层验证方法，与具体算子无关；case 表按目标算子冻结 CSV 翻译）。
- 路线优先级：recommended_fastpath（TTK 无该能力时的唯一等价路径）。
- 明确不适用 / 未验证场景：TTK 未来版本若支持 expected-error 解析则直接用 TTK；直调工程（无 aclnn 层）不适用；正向用例验收仍走 TTK/golden 主链路，本卡只覆盖负向语义。
- 负向用例本身的"真实违约 + remark 锁错误码"设计约定见 [l2_negative_true_violation_lock_error_code](l2_negative_true_violation_lock_error_code.md)——本卡是其在"TTK 无法执行断言"环境下的执行侧对偶。

## 判别方法

**判定 TTK 是否支持负向用例**：读 TTK 源码 `core_modules/aclnn/acl_interface.py`（错误码表 `161002: "ACLNN_ERR_PARAM_INVALID"` 等）与 `core_modules/testcase_manager/testcase_aclnn.py`（fail_reason 赋值路径），确认无 remark/expected 列的解析与断言逻辑——支持则用 TTK，不支持走本卡等价程序。

**两段式验证设计**：

1. 第一段（GetWorkspaceSize 阶段）：逐 case 构造 x/gamma/other/out 四个 aclTensor（shape/dtype/format/epsilon 与冻结 CSV 逐字段一致），调 `aclnn<Op>GetWorkspaceSize`，`ret!=0` 判 `REJECTED`（对照 remark 预期错误名），`ret==0` 判 `UNEXPECTED_SUCCESS`（进入第二段）。
2. 第二段（执行阶段）：对第一段未拦截的 case，按 workspaceSize 分配 workspace 后调 `aclnn<Op>` 执行 + `aclrtSynchronizeStream`，记录 execRet；执行仍不报错判 `NOT_ENFORCED`（接口层未强制该约束的确定结论）。**out 存储按 numel(x) 垫底**：负向 case 若未被拦截，kernel 会按 x 的逻辑元素数写回 out，按用例（错误的）out shape 分配会越界破坏设备内存——垫底（`minElems=numel(x)`）保证执行阶段安全，view shape 仍传用例原值（语义不受影响）。

**关键实现（verbatim，最终形态）**——垫底构造与无符号安全的 stride 计算（rank=0 时 `rank-1` 下溢是真实踩过的段错误点）：

```cpp
int64_t Numel(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

size_t DtypeBytes(aclDataType t)
{
    switch (t) {
        case ACL_BOOL: return 1;
        case ACL_FLOAT: return 4;
        case ACL_COMPLEX128: return 16;
        case ACL_BF16: return 2;
        default: return 2;
    }
}

aclTensor* MakeTensor(const TensorSpec& spec, std::vector<void*>& devPtrs)
{
    return MakeTensorPadded(spec, devPtrs, 0);
}

aclTensor* MakeTensorPadded(const TensorSpec& spec, std::vector<void*>& devPtrs, int64_t minElems)
{
    const size_t rank = spec.shape.size();
    int64_t numel = Numel(spec.shape);
    if (numel < 0) numel = 0;
    int64_t allocElems = numel > minElems ? numel : minElems;
    size_t bytes = static_cast<size_t>(allocElems) * DtypeBytes(spec.dtype);
    if (bytes == 0) bytes = 32;
    void* dev = nullptr;
    if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
    devPtrs.push_back(dev);
    std::vector<int64_t> strides(rank, 1);
    int64_t acc = 1;
    for (size_t i = rank; i-- > 0;) {
        strides[i] = acc;
        acc *= spec.shape[i];
    }
    static int64_t kEmpty[1] = {1};
    const int64_t* dims = rank > 0 ? spec.shape.data() : kEmpty;
    const int64_t* str = rank > 0 ? strides.data() : kEmpty;
    return aclCreateTensor(dims, rank, spec.dtype, str, 0, spec.format, dims, rank, dev);
}
```

两段式主循环（do/while(false) + devPtrs 侧表管理释放；`TensorSpec{shape, dtype, format}` 与 `Case{name, x, g, o, y, epsilon, expected}` 结构从冻结 CSV 逐条翻译，case 表此处从略——每条即一个字面量条目，形态见下）：

```cpp
    for (const auto& c : Cases()) {
        ++total;
        std::vector<void*> devPtrs;
        aclTensor* x = nullptr;
        aclTensor* g = nullptr;
        aclTensor* o = nullptr;
        aclTensor* y = nullptr;
        aclnnStatus ret = 0;
        do {
            // out 存储按 numel(x) 垫底：负向 case 若 GetWorkspaceSize 未拦截，
            // 执行阶段 kernel 按 x 的逻辑元素数写回，垫底防止越界破坏（view
            // shape 仍为用例原值，语义不受影响）。
            const int64_t xElems = Numel(c.x.shape);
            x = MakeTensor(c.x, devPtrs);
            g = MakeTensor(c.g, devPtrs);
            o = MakeTensor(c.o, devPtrs);
            y = MakeTensorPadded(c.y, devPtrs, xElems);
            if (x == nullptr || g == nullptr || o == nullptr || y == nullptr) {
                std::fprintf(stderr, "L2RESULT|case=%s|ret=ALLOC_FAILED|expected=%s|verdict=ERROR\n",
                             c.name, c.expected);
                break;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            aclnnStatus ret = aclnnFusedRMSNormSigmoidMulGetWorkspaceSize(
                x, g, o, c.epsilon, y, &workspaceSize, &executor);
            if (ret == 0) {
                // 两段式第二段：验证执行阶段是否拦截（workspace 按需分配）。
                void* workspace = nullptr;
                if (workspaceSize > 0 &&
                    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    std::fprintf(stderr, "L2RESULT|case=%s|ret=WS_ALLOC_FAILED|expected=%s|verdict=ERROR\n",
                                 c.name, c.expected);
                    break;
                }
                aclrtStream stream = nullptr;
                // …创建 stream → aclnnFusedRMSNormSigmoidMul(x,g,o,eps,y,workspace,wsSize,executor,stream)
                //   → aclrtSynchronizeStream → 记 execRet，verdict=NOT_ENFORCED …
            }
            // …打印 L2RESULT 行（ret / execRet / expected / verdict）…
        } while (false);
        if (x != nullptr) aclDestroyTensor(x);
        // …g/o/y 同法销毁；for (auto p : devPtrs) aclrtFree(p); …
    }
```

case 表条目形态（从冻结 CSV 逐字段翻译，dtype/format 域错误示例；shape 数据按目标算子 CSV 替换）：

```cpp
struct TensorSpec {
    std::vector<int64_t> shape;
    aclDataType dtype;
    aclFormat format;
};

struct Case {
    const char* name;
    TensorSpec x;
    TensorSpec g;
    TensorSpec o;
    TensorSpec y;
    double epsilon;
    const char* expected;  // CSV remark 预期错误名
};

const std::vector<Case>& Cases()
{
    static const std::vector<Case> kCases = {
        {"aclnnFusedRMSNormSigmoidMul_L2_exc_dtype_001",
         {{{10523739}, ACL_BOOL, ACL_FORMAT_ND}}, {{{10523739}, ACL_BF16, ACL_FORMAT_ND}},
         {{{1}, ACL_BF16, ACL_FORMAT_ND}}, {{{10523739}, ACL_BF16, ACL_FORMAT_ND}},
         0.0, "ACL_ERROR_DTYPES_NOT_MATCH"},
        // …其余 16 条同法，逐字段与冻结 CSV 一致…
    };
    return kCases;
}
```

编译命令（直连 vendor 包 + CANN 运行时，`-O0 -g` 便于段错误定位；路径按实际 CANN 安装与 vendor 部署替换）：

```bash
CANN=<CANN 安装路径>; VENDOR=$CANN/opp/vendors/FusedRMSNormSigmoidMul; \
g++ -std=c++17 -O0 -g -fPIC l2_verify.cpp \
  -I$VENDOR/op_api/include -I$CANN/x86_64-linux/include -I$CANN/include \
  -L$VENDOR/op_api/lib -lcust_opapi -L$CANN/lib64 -lascendcl -lnnopbase -lopapi \
  -Wl,-rpath,$VENDOR/op_api/lib -Wl,-rpath,$CANN/lib64 -o l2_verify && \
./l2_verify > ../logs/l2-verify.stdout.log 2> ../logs/l2-verify.log
```

## 设计决策

- 为什么两段式：GetWorkspaceSize 阶段与执行阶段的校验职责不同——本会话实测 15 条在 GetWorkspaceSize 即被拒，2 条 shape 负向两阶段都放行；只跑第一段会把"未拦截"误报为"意外成功"，两段式才能给出 `NOT_ENFORCED`（接口层未强制约束）的确定结论。
- 为什么 out 垫底而不是改用例 shape：改 shape 就不是负向用例了；垫底只影响物理分配，aclTensor 的 view（dims/strides）保持用例原值，语义不变而执行安全。
- 为什么错误码以运行时返回码为准而不是仅对照枚举名：remark 里的预期是错误"名"，实际返回是数值码（如 161002=ACLNN_ERR_PARAM_INVALID 域、561002=空 tensor/参数域），程序记录数值 + 名双字段便于比对。
- 为什么 `-O0 -g`：验证程序自身也可能有 bug（本会话就段错误过一次），带符号的调试构建让 gdb 一跳即中。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0（宿主机上有他人常驻 test_aclnn 进程属正常，勿清理），CANN 9.2.0；vendor 包已 `opp/vendors/FusedRMSNormSigmoidMul/` 安装。
- 相关文件：`docs/perf-integrate/l2_verify/l2_verify.cpp`（验证程序）、`perf-opt-out/functional-full/aclnnFusedRMSNormSigmoidMul_l2_exception.csv`（冻结用例，SHA 1f09c8ca…）、`docs/perf-integrate/logs/l2-verify.log`（结果日志）；TTK 源码检出目录（负向支持判定依据）。

## 验证信号

以下信号来自该融合算子集成回注册工程后的 L2 负向验收会话（Ascend950PR / CANN 9.2.0）：程序在 17 条冻结负向用例上逐条执行并断言返回码。

- TTK 能力判定：`ttk/core_modules/aclnn/acl_interface.py` 与 `testcase_manager/testcase_aclnn.py` 源码核对无 remark/expected-error 解析（来源平台：950；证据等级：tool_observed）。
- 程序自证（踩坑记录）：聚合嵌套初始化编译错（改显式构造后通过）；`rank-1` 无符号下溢段错误（`for (size_t i = rank - 1; i-- > 0;)` 在 rank=0 时 i 下溢为 SIZE_MAX），改为 `for (size_t i = rank; i-- > 0;)` 正向累乘后消失（来源平台：950；证据等级：tool_observed）。
- 最终结论：17 条中 15 条 `REJECTED` 且返回码与 remark 预期域一致——dtype/format/constraint 域 161002（如 `L2_exc_dtype_001|ret=161002|expected=ACL_ERROR_DTYPES_NOT_MATCH|verdict=REJECTED`）、空 tensor/参数域 561002（`L2_empty_019|ret=561002`、`L2_exc_constraint_030/032/033|ret=561002`）（来源平台：950；证据等级：tool_observed）。
- 2 条 shape 负向（`L2_exc_shape_013/014`）两段式结果 `ret=0|execRet=0|verdict=NOT_ENFORCED`——GetWorkspaceSize 与执行阶段均未拦截；经 aclnn 层集成前后逐字节相同（SHA 对照）证据确认属集成前即存在的接口层校验缺口，非本次集成引入（来源平台：950；证据等级：tool_observed）。
- 执行后设备状态正常（常驻进程无恙），垫底策略未发生越界破坏（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡验证的是"接口层拒绝语义"；`NOT_ENFORCED` 的根因在 autogen stub 的校验覆盖（本会话未追查、未修复），下个算子可能同样存在——负向结论须逐算子实测。
- 垫底按 `numel(x) * DtypeBytes(y.dtype)` 估算写回规模：若 kernel 写回规模与 x numel 无关（如按 out shape 放大的算子），垫底口径需按目标算子重新推导。
- 验证程序自身使用 `aclInit(nullptr) + aclrtSetDevice(0)` 初始化模式；与常驻进程共存时无需显式 create_context（宿主机常规形态）。
- TTK 版本升级后应重查负向支持能力，避免维护重复的等价程序（未验证 TTK 后续版本）。

**未决项**：

- `NOT_ENFORCED` 的 autogen stub 根因（为何输出 shape 不参与 GetWorkspaceSize 校验）未追到机制层——若下个算子需要"接口层强制 shape 校验"，须先复现并定位 stub 生成规则（优先项）。
- 161002/561002 的错误码→语义映射来自 TTK 源码表与运行时观测，CANN 官方错误码文档对照未做。
## 坏实践

- 负向 case 的 out 按用例（错误）shape 分配且不做垫底——一旦接口未拦截，执行阶段越界写破坏设备内存，后果是随机段错误/数据破坏而非干净断言（本会话以垫底规避）。
- 只跑 GetWorkspaceSize 就下"全部拒绝"结论——会把执行阶段才拦截（或两阶段都不拦截）的 case 误判。
- stride 计算用 `for (size_t i = rank - 1; i-- > 0;)` 反向递推——rank=0 输入（标量/空 shape case）必然无符号下溢；用 `for (size_t i = rank; i-- > 0;)` 正向累乘。
- 嵌套 TensorSpec 聚合初始化一口气写完——混合类型字面量易触发 `could not convert` 编译错；改显式构造（`TensorSpec{...}` 逐层）。
