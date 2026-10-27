---
type: "Runbook"
title: SetMatmulConfigParams 枚举需命名空间前缀
description: "Host tiling 调用 SetMatmulConfigParams 时 ScheduleType/MatrixTraverse 按文档示例无限定书写会编译报 'X' has not been declared；两者实际是 matmul_tiling 命名空间内的 enum class，跨命名空间调用须逐个加前缀。R2：SetAType/SetBType/SetCType/SetBiasType 的形参类型 TPosition/CubeFormat/DataType 同型——须 matmul_tiling:: 限定或函数内 using namespace（注意与 ge::DataType 短名二义）。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-17T04:05:46Z'
platforms: ["950"]
---

# SetMatmulConfigParams 枚举需命名空间前缀

## 现象

在为 arch35（Ascend950PR / DAV_3510，MIX 模式）上的三维卷积反传 filter 梯度算子实现 L1 常驻 Matmul 方案时，Host 侧 tiling 函数（`op_host/arch35/*_tiling.cpp`，`namespace optiling` 内）按官方 `SetMatmulConfigParams` 文档的示例写法调用了：

```cpp
tiling.SetMatmulConfigParams(0, false, ScheduleType::INNER_PRODUCT, MatrixTraverse::NOSET, true);
```

执行 `bash build.sh --soc=ascend950 -j8` 编译 op_host 时报错（同一行两处）：

```
op_host/arch35/<op>_tiling.cpp:184:44:
  error: 'ScheduleType' has not been declared
op_host/arch35/<op>_tiling.cpp:184:73:
  error: 'MatrixTraverse' has not been declared
```

当时代码状态：同文件中其他 Matmul tiling 调用已经全部使用 `matmul_tiling::MultiCoreMatmulTiling`、`matmul_tiling::TPosition::TSCM`、`matmul_tiling::CubeFormat::NZ` 等限定名，唯独这两个枚举照抄了文档的无限定写法。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的 op_host TilingFunc，使用 `MultiCoreMatmulTiling` / `MatmulApiTiling` 配置 `SetMatmulConfigParams`（启用 enVecND2NZ 等高级配置）。
- API 路径：`matmul_tiling::MatmulApiTilingBase::SetMatmulConfigParams`；枚举 `ScheduleType` / `MatrixTraverse` 定义于 `asc/include/adv_api/matmul/matmul_tiling_base.h` 的 `namespace matmul_tiling`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0（x86_64-linux 安装，交叉编译验证）。
- 适用粒度：platform_wide——所有使用 matmul tiling 配置接口的 Host 代码。
- 路线优先级：recommended_fastpath（一次性编译修复，无权衡）。
- 明确不适用 / 未验证场景：Kernel 侧（`__aicore__` 代码）不涉及本枚举；`MatmulConfig`/`GetNormalConfig`（Kernel 侧模板参数）是另一套类型，不受此影响。

## 根因

`ScheduleType`、`MatrixTraverse` 是 `enum class`，定义在 `matmul_tiling` 命名空间内（`matmul_tiling_base.h` 第 23 行 `namespace matmul_tiling {`，第 150/177 行分别为两个枚举）；官方 `SetMatmulConfigParams` 文档的调用示例使用无限定名（其示例上下文隐含 `using namespace matmul_tiling` 或处于该命名空间内），直接照抄到 `namespace optiling` 的 Host tiling 代码中即编译失败。

## 解决配方

修复（before/after，同一行）：

before：

```cpp
    // 启用 enVecND2NZ（配合 A/B 在 TSCM + NZ 格式）
    tiling.SetMatmulConfigParams(0, false, ScheduleType::INNER_PRODUCT, MatrixTraverse::NOSET, true);
```

after：

```cpp
    // 启用 enVecND2NZ（配合 A/B 在 TSCM + NZ 格式）
    tiling.SetMatmulConfigParams(0, false, matmul_tiling::ScheduleType::INNER_PRODUCT,
                                 matmul_tiling::MatrixTraverse::NOSET, true);
```

## 设计决策

- 备选方案：在文件头 `using namespace matmul_tiling;`——被放弃，因为 Host tiling 文件同时存在 `optiling`、`AscendC::tiling` 等多个命名空间，全量 using 易引入歧义（同文件已对 `matmul_tiling::TPosition` 等逐个限定）。
- 采用逐调用限定前缀，与同文件其余 `matmul_tiling::` 用法保持一致。
- 本修复属 recommended_fastpath：无性能/语义权衡。

## 系统信息

- 工具链：CANN 9.1.0（x86_64-linux），目标 ascend950，GCC 13.3.0。
- 相关文件：`op_host/arch35/*_tiling.cpp`（报错行）、`$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/matmul/matmul_tiling_base.h`（枚举定义）。

## 预防

- 照抄 AscendC 文档示例到 Host tiling 代码时，检查示例中的类型是否带命名空间；`matrix/bmm_tiling.h` 间接引入的类型多数位于 `matmul_tiling`。
- 编译报 `‘X’ has not been declared` 时，先用 `grep -rn "enum class X" $ASCEND_HOME_PATH/x86_64-linux/asc/include/` 定位真实命名空间再改。

## 验证信号

本验证来自该算子 L1 常驻 Matmul 方案落地时的编译循环（修改 op_host tiling 后重建）。

- 修复前 op_host 编译报 `error: 'ScheduleType' has not been declared` / `error: 'MatrixTraverse' has not been declared`（来源平台：950 工具链；证据等级：tool_observed）。
- 修复后 `bash build.sh --soc=ascend950 -j8` 完整通过并生成安装包 `custom_opp_ubuntu_x86_64.run`（CPack 打包成功）（来源平台：950 工具链；证据等级：tool_observed）。

## 失效边界

- 仅验证于 CANN 9.1.0 的头文件布局；若后续版本将枚举移出 `matmul_tiling` 命名空间或文档示例修正，本卡失效。
- Kernel 侧模板参数（`MatmulConfig`、`GetNormalConfig`）不走此路径。

**未决项**：

- 无。`matmul_tiling_base.h` 中 `SetMatmulConfigParams` 声明本身使用无限定名（其位于命名空间内部），文档示例因此不带前缀——文档与跨命名空间调用者之间的表述差异属文档问题，未在本轨迹反馈。
## 坏实践

- 直接复制文档/参考代码片段而不核对命名空间上下文。
- 编译错误后盲目改用 `using namespace` 全量引入，掩盖其他符号冲突。

## 误诊

无（本错误信息自解释，编译器提示直接指向缺失声明）。

## 增量（SetAType 族形参类型同型）

op_host 用 `MatmulApiTiling` 生成 tiling 时（MoE 分组矩阵乘算子，Ascend950PR / arch35 / CANN 9.1.0），按直觉裸写类型名：

```cpp
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, mmDtype, false);
    DataType mmDtype = DataType::DT_FLOAT16;
```

`bash build.sh --soc=ascend950 -j1` 报错原文（同轮还有 lambda auto 报错，此处只列本根因）：

```text
error: ‘DataType' was not declared in this scope
error: ‘mmDtype' was not declared in this scope; did you mean ...
error: ‘DataType' is not a class, namespace, or enumeration
error: ‘TPosition' has not been declared
error: ‘CubeFormat' has not been declared
```

根因与主卡同源：`MatmulApiTiling` 继承自 `MatmulApiTilingBase`，二者及其形参类型 `TPosition`、`CubeFormat`、`DataType` 全部定义在 `namespace matmul_tiling`（`asc/include/tiling/matrix/matmul_tiling_base.h`）中；`SetAType(TPosition::GM, ...)` 的实参推导发生在调用点所在的命名空间上下文，不会自动进入被调类的命名空间找枚举。

修复（函数内 `using namespace matmul_tiling;` + 关键类型仍全限定）。before（GenerateMatmulTiling，类型裸用）：

```cpp
static ge::graphStatus GenerateMatmulTiling(platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             int64_t maxMi, int64_t N, int64_t K,
                                             ge::DataType dtype, bool hasBias,
                                             optiling::TCubeTiling& mmTiling)
{
    matmul_tiling::MatmulApiTiling tilingApi(ascendcPlatform);

    // 数据类型映射
    DataType mmDtype = DataType::DT_FLOAT16;
    if (dtype == ge::DT_FLOAT) {
        mmDtype = DataType::DT_FLOAT;
    } else if (dtype == ge::DT_BF16) {
        mmDtype = DataType::DT_BF16;
    }

    // 设置 A/B/C/Bias 类型
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, mmDtype, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, mmDtype, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, mmDtype);
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    ...
}
```

after（函数内 `using namespace matmul_tiling;` + 全限定类型名）：

```cpp
static ge::graphStatus GenerateMatmulTiling(gert::TilingContext* context,
                                             platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             int64_t maxMi, int64_t N, int64_t K,
                                             ge::DataType dtype, bool hasBias,
                                             optiling::TCubeTiling& mmTiling)
{
    using namespace matmul_tiling;

    matmul_tiling::MatmulApiTiling tilingApi(ascendcPlatform);

    // 数据类型映射
    matmul_tiling::DataType mmDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (dtype == ge::DT_FLOAT) {
        mmDtype = matmul_tiling::DataType::DT_FLOAT;
    } else if (dtype == ge::DT_BF16) {
        mmDtype = matmul_tiling::DataType::DT_BF16;
    }

    // 设置 A/B/C/Bias 类型
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    ...
}
```

（同轮顺手修正：可选输入 bias 的存在性检测从 `context->GetOptionalInputShape(0)` 改为 `context->GetInputShape(2)`——按实际输入索引取 shape，缺省时返回 nullptr；此改动无独立报错信号。）

写法取舍（与主卡"逐调用限定"互补）：函数内 `using namespace matmul_tiling;` + 关键赋值处保留 `matmul_tiling::` 前缀消歧——纯 using 会与宿主 `ge::DataType`（TilingFunc 里到处用）产生 `DataType` 短名二义风险；host 侧 `ge::DataType` 与 matmul_tiling 的 `DataType` 是两套枚举，须逐分支显式映射，不做隐式转换假设。不要在 tiling.cpp 顶部全局 `using namespace matmul_tiling;`（短名与 `ge::DataType` 在整个编译单元冲突）。

### 增量验证信号

- 修复前：`error: 'TPosition' has not been declared`、`error: 'CubeFormat' has not been declared`、`error: 'DataType' was not declared in this scope`（来源平台：950；证据等级：tool_observed）。
- 修复后：上述报错串不再出现，host 编译推进到 `PlatformAscendC` 构造阶段（下一根因）；连同后续修复全量回归 `build.sh --soc=ascend950 -j8` 生成 `custom_opp_ubuntu_x86_64.run` 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。

### 增量失效边界

- 函数级 using namespace 的二义性规避写法在引入更多同名类型（如同时用 std::DataType 类库）时需再评估。
- kernel 侧 `MatmulType<TPosition::GM, ...>` 是 AscendC 命名空间体系（kernel 头同时用 `using namespace AscendC;` 时不报此错），不要把 host 侧的 matmul_tiling 限定搬到 kernel。