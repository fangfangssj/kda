---
type: "Runbook"
title: opbuild dtype 列表不等长报 dtype size 0
description: '各输入/输出的 DataType/Format 列表是按索引 zip 成 dtype 组合的，长度不等时 opbuild 报误导性 "The dtype size of input[0] of op X is 0."；须把所有列表展开成等长、按索引对齐的组合（固定 dtype 输入/输出重复填充，含 optional 输出），嵌套 vector 则直接编译失败。所有端口 DataType 已等长、仅 Format/UnknownShapeFormat 落后同样触发（R6）；ascend910b 上同套约束复现（R6 复验）。R9：给已有算子扩 dtype 时可选输出落在单条目同样触发；报错行被 shell 管道截断（"is" 后丢失）时勿因文本不完整而误判。R9：可选 bool 输入/int32 输出的单条目形态复现，且 autogen ops-info.ini 的 inputN 段缺 dtype/format 行可直接作为注册信息解析为空的探针。'
tags: [task.implementation, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950", "a2"]
---

# opbuild dtype 列表不等长报 dtype size 0

## 现象

三类独立工程在首次 cmake configure / `bash build.sh --soc=ascend950` 的 "opbuild ops" 阶段撞到同一报错。以 MoE token 重排算子为例（多输入多输出，tokens 为 fp16/bf16/int8、expert_token_num_per_rank 为 int32/int64）：

```
CMake Error at $ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:783 (message):
  opbuild ops error: The dtype size of input[0] of op
  BenchNoHarnessMoeReRouting is 0.
Call Stack (most recent call first):
  op_host/CMakeLists.txt:48 (npu_op_code_gen)
```

当时 `op_host/*_def.cpp` 中各输入/输出的 `DataType({...})` 列表长度互不相等：tokens 3 项、计数输入 2 项、可选 scales 输入 1 项、各输出 3/1/1/2 项。已先把 bf16 枚举名修成 `ge::DT_BF16`（含 proto.h 残留的 `DT_BFLOAT16`）后重编，错误依旧——排除枚举名因素。func.cmake:783 只是包装 opbuild（`$ASCEND_HOME_PATH/tools/opbuild/op_build`）的退出信息，真正原因在 opbuild 对 def 的组合生成。

另一工程为补齐该问题时曾把 dtype 映射写成嵌套初始化列表，编译期直接报另一类错：

```
error: no matching function for call to 'OpParamDef::DataType(std::vector<std::vector<ge::DataType> >)'
   180 |   OpParamDef &DataType(std::vector<ge::DataType> types);
```

## 适用场景

- 工程形态：ops-registry-invoke / registry-invoke 自定义算子开发，op_host 侧 def.cpp（OpDef）编写；多输入多输出、各输入 dtype 集合不同（如主数据 3 dtype × 计数张量 int32/int64 × 固定 int32 索引）的算子首次构建。
- API 路径：`register/op_def_registry.h` 的 `this->Input(...)/Output(...)` 链式 `.DataType({...}).Format({...}).UnknownShapeFormat({...})`；经 npu_op_code_gen → opbuild 生成 ops info（每个 dtype 组合对应一个 kernel binary）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950/arch35），CANN 9.1.0（三个独立工程复现）。
- 适用粒度：platform_wide——CANN 9.x opbuild 的 op def 组合生成机制，与具体算子无关。
- 路线优先级：recommended_fastpath（等长是硬约束，无替代路线）。
- 明确不适用 / 未验证场景：所有输入 dtype 列表本就等长的算子（单输入天然等长）不会触发；通过属性传 dtype 而非多 dtype 输入的算子；opbuild 内部为何以 "input[0] ... is 0" 措辞报错未读源码核实（见失效边界）。

## 根因

op def 中各输入/输出的 `DataType({...})` 列表是**按索引 zip 成 dtype 组合**的：第 i 项列表元素们构成第 i 种组合（tokens[i] 配 num[i] 配 scales[i]…），不是各输入独立取笛卡尔积；dtype 列表长度 = opbuild 生成的 kernel binary 数。当列表长度不等（3 项的 tokens 对 2 项的 num、1 项的 scales）时，组合表无法按位置对齐生成，opbuild 落到 "dtype size is 0" 的兜底报错——错误文本指向 input[0]，实际成因是某个输入（如 int32 索引输入）的列表长度与主输入不一致，具有误导性。另：`OpParamDef::DataType` 的签名只接受一维 `std::vector<ge::DataType>`（每个 IO 一个平铺列表），嵌套 `{{...},{...}}` 无法匹配任何重载，直接编译失败。修复方式是把所有输入/输出列表展开成**等长且按索引对齐**的组合枚举；固定 dtype 输入重复填充自己的 dtype 到等长。

注：组合 zip 的内部实现未直接读源码验证，因果由三个独立工程的实验闭合（等长展开后该错误消失、configure 通过；不等长即报）。

## 解决配方

修复前（长度不等的列表 + 嵌套 vector 两种错误形态）：

```cpp
        // 嵌套 vector 形态（编译直接失败：OpParamDef::DataType 只有一维重载）
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT},
                       {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT}})
            ...

        // 平铺但不等长形态（opbuild 报 dtype size 0）
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})   // 3 项
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})                 // 2 项，与主输入不等长
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(REQUIRED, DependScope::TILING)
            .AutoContiguous();

        this->Input("per_token_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})                               // 1 项
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
```

修复后（全部展开为 6 组索引对齐组合——3 token dtype × 2 num dtype；注释即组合表；固定 dtype 输入重复填充）：

```cpp
        // === Inputs ===
        // tokens: fp16/bf16/int8, expert_token_num_per_rank: int32/int64
        // DataType lists must have matching length across inputs/outputs
        // Combinations: (fp16,int32), (bf16,int32), (int8,int32), (fp16,int64), (bf16,int64), (int8,int64)
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                        ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(REQUIRED, DependScope::TILING)
            .AutoContiguous();

        this->Input("per_token_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

各输出同理等长展开（permute_tokens 6 项随 tokens、permute_token_idx 6 项重复 DT_INT32、expert_token_num 6 项随计数输入），Format / UnknownShapeFormat 列表长度同步补齐（三个列表长度一致）。

固定 dtype 输入的轻量等长形态（另一工程验证：主输入 3 dtype、int32 索引输入只列 1 个 dtype 报错，补齐为 3 个 DT_INT32 后 configure 立即通过）：

```cpp
        this->Input("expanded_src_to_dst_row")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

dtype 之间的强约束关系（如 x1=FP16 时 xOut 必须=FP16）不在 `_def.cpp` 表达，改由 L2 API 的 `CheckDtypeValid` 检查（`OP_CHECK_DTYPE_NOT_MATCH(xOut, x1->GetDataType(), ...)`）；注册层只声明各 IO 支持的 dtype 组合数。

设计决策：

- 保持全组合显式枚举而不是删 dtype 缩小支持面——等长展开是保留语义的最小改动；另一工程的替代路线是"收窄"（全部 input/output 收窄为长度 1，主路径只留 fp16、索引留 int32），两者都满足等长约束，选择取决于 kernel 模板当前支持的 dtype 范围。
- 同仓既有算子（如 arg_max 类）把全部 5 个 dtype 同时列在 input 和 output 上保持等长——与"定长输入陪跑重复"是同一机制的不同取法。
- y/scaleOut 这类固定 dtype 输出用重复 dtype 占位而不是缩短列表：opbuild 按位置对齐映射，等长是硬约束。

另两例独立复现（R4/R5，TensorList p-范数算子与动态量化算子，同为 Ascend950PR + CANN 9.1.0）：

- R4（TensorList p-范数算子，DYNAMIC_INPUT 三 dtype 扩展）：x 声明 `DataType({F, F16, BF16})` 3 项但 `Format({FORMAT_ND})` 仅 1 项——同错复现；**只把 scalar 输入的 DataType 补到 3 元素（Format 仍单元素）后同错依旧**——证明 DataType/Format/UnknownShapeFormat 三个列表必须同时等长，单补一列无效；三字段全部按 3 组合等长后 opbuild 通过、构建推进到 kernel 编译阶段。
- R5（动态量化算子，fp16+bf16 双 dtype）：输出 y 为 `DataType({DT_INT8})`、scale 为 `DataType({DT_FLOAT})` 各 1 项对输入 2 项——同错复现；输出侧 dtype 也按输入组合数各写两份（`{INT8, INT8}`、`{FLOAT, FLOAT}`）后 cmake 配置通过（`Opbuild generating sources - done`），产出 2 个 kernel 二进制与 .run 包。输出物理 dtype 不随输入变化时用重复值占位，而不是留 1 项。

R4/R5 增补的机制要点：

- opbuild 按输入 dtype 组合数对齐校验**输出**列表；输出 dtype 不随输入变化时重复填充占位（R5 实测闭合）。
- 同批实验确认 `DataTypeList({F, F16, BF16})` + `FormatList({ND})` 形态可过 opbuild，但 `DataTypeList` 在多个输入/输出之间做**笛卡尔积**（3×1×3=9 组合），会引入组合爆炸与模板不匹配的新问题——需要"输出跟随输入 dtype"的场景不要用 DataTypeList，回到等长 `DataType` 列表。
- op_graph `REG_OP` 的 TensorType 列表无此约束（同批同时修改 op_graph 未触发该错误），约束只作用于 op_def/opbuild 路径。

R6 增量（BenchNoHarnessMaximum，Ascend950PR + CANN 9.1.0，ascend910b 复验）：**所有端口 DataType 列表本已等长（x1/x2/y 各 6 种 dtype）、仅 `Format({ge::FORMAT_ND})` 与 `UnknownShapeFormat({ge::FORMAT_ND})` 各只 1 项**（单 dtype 模板写法原样保留）——同错复现 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMaximum is 0.`（op_host/CMakeLists.txt:54 npu_op_code_gen）。把三个端口的 Format/UnknownShapeFormat 均补齐 6 项 FORMAT_ND 后 opbuild 通过、报错推进到 kernel 阶段，最终 ascend950 与 ascend910b 各产出 6 个 kernel 二进制。即：跨端口 DataType 等长**不能豁免**端口内 Format 与 DataType 的等长约束；从单 dtype 算子模板复制 def.cpp 后只扩 `.DataType({...})` 不动 `.Format({...})` 是该形态的典型成因。

## 判别方法

- 触发：`opbuild ops error: The dtype size of input[N] of op X is 0.`（func.cmake:783 包装，`-- Configuring incomplete, errors occurred!`）。
- 探针：逐个数所有 input/output 的 `DataType(...)`/`Format(...)`/`UnknownShapeFormat(...)` 三个列表的项数，找不一致者——不要只看报错指向的那个输入（错误指向 input[0] 是误导，真正不等长的可能是索引输入，也可能是某个输出）。
- 手工复现 opbuild 时**先清掉上次产物**再跑，否则残留 `.ini` 会造成 EXIT=0 假成功（R7 实测踩过，见坏实践）：

```bash
cd build/autogen && rm -f *.ini *.txt opbuild.log
OPS_PRODUCT_NAME=ascend950 $ASCEND_HOME_PATH/tools/opbuild/op_build libascend_all_ops.so . "--compute_unit=ascend950"; echo "EXIT=$?"
```

- 嵌套花括号 `{{...},{...}}` 直接不要写——`OpParamDef::DataType` 只有一维重载，报 `no matching function ... std::vector<std::vector<ge::DataType>`。
- 修复自查：写组合注释（`(fp16,int32), (bf16,int32), ...`）核对索引对齐。
- 归因实验（R8）：对疑似端口做最小化对照——去掉该端口+Attr → 配置通过；加回单条目 → 复现失败；条目数补到与组合数相等 → 通过——即可锁定"该端口列表长度"这一唯一变量。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0；报错来源 `$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:765`（执行 `${ASCEND_CANN_PACKAGE_PATH}/tools/opbuild/op_build`）与 `:783`（FATAL_ERROR 包装）。
- 相关文件：各工程 `op_host/*_def.cpp`（dtype 列表声明）。

## 预防

- 写多输入多输出 op def 时，先定"主 dtype 数量 N"，再逐个 input/output 检查三个列表（DataType/Format/UnknownShapeFormat）长度都为 N；int32 索引、bool 掩码等固定 dtype 输入用重复填充。
- 见到 "The dtype size of input[i] of op X is 0" 优先怀疑 def 的列表长度不等，而不是 dtype 本身（fp16/bf16/int8 尺寸都是已知的非 0 值），更不是报错指向的那个输入。

## 验证信号

以下信号来自三个独立工程的首次构建失败-修复闭环（均为 Ascend950PR + CANN 9.1.0 实测）。

- MoE 重排算子：`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 两次稳定复现 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMoeReRouting is 0.`（其中一次已先修好 proto.h 的枚举名，错误不变，排除枚举名因素）；def.cpp 展开为 6 组等长组合重编后 "dtype size 0" 消失、报错切换为 ValueDepend 相关的下一类组合错误，证明本根因已消除（来源平台：950；证据等级：tool_observed）。
- MoE finalize 算子：主输入 3 dtype + 两个 int32 索引输入各 1 dtype → configure 报 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMoeFinalizeRouting is 0.`、`Configuring incomplete`；仅把两个 int32 输入补齐为 `{ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}`（其余不动）→ `-- Opbuild generating sources - done`、`-- Configuring done / -- Generating done`（来源平台：950；证据等级：tool_observed）。
- 融合量化算子：嵌套 vector 首轮报 `OpParamDef::DataType(std::vector<std::vector<ge::DataType>>)` 无匹配重载；改平铺但不等长（y 只列 `{INT8}` 1 项 vs x1 `{FP16, BF16}` 2 项）又报 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessAddRmsNormDynamicQuant is 0.`；等长终版（全部 IO 2 项，y/scaleOut 重复自己的 dtype）写入后同一构建流程推进通过 opbuild/def 编译，最终全工程编译成功（来源平台：950；证据等级：tool_observed）。

R4/R5 增量信号（独立 trace，均为 Ascend950PR + CANN 9.1.0 实测）：

- R4（TensorList p-范数算子）：初始不齐列表报 `The dtype size of input[0] of op BenchNoHarnessForeachNorm is 0.`；仅把 scalar 的 DataType 补到 3 元素（Format 仍单元素）同错依旧；三个字段全部 3 组合等长后 opbuild 阶段通过，构建推进到 kernel 编译阶段；与 TilingKey 模板修复共同作用后全量构建成功，产出 3 个 dtype 的 kernel binary 并安装（来源平台：950；证据等级：tool_observed）。
- R5（动态量化算子）：fp16+bf16 双 dtype 注册报 `The dtype size of input[0] of op BenchNoHarnessDynamicQuant is 0.`（func.cmake:783，配置不完整）；Format/UnknownShapeFormat 补齐、输出 dtype 重复占位后同命令 cmake 配置通过（`Opbuild generating sources - done`，target 名正常列出），后续完整构建产出 2 个 kernel 二进制与 custom_opp .run 包；生成产物 `aic-ascend950-ops-info.json` 可见 `input0 dtype: "float16,bfloat16"`、`output0 dtype: "int8,int8"` 的成对结果（来源平台：950；证据等级：tool_observed）。

R6 增量信号（独立 trace，Ascend950PR + CANN 9.1.0 实测、ascend910b 复验）：

- 修复前：`bash build.sh --soc=ascend950 -j8` 在 CMake 配置阶段报 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMaximum is 0.`（DataType 6 项/Format 1 项形态）（来源平台：950；证据等级：tool_observed）。
- 修复后（仅补齐 Format/UnknownShapeFormat 至 6 项）：opbuild 阶段通过，报错推进到 kernel 编译阶段的独立问题，最终 ascend950 构建成功产出 6 个 kernel 二进制（来源平台：950；证据等级：tool_observed）。
- 同套 def.cpp 随后在 ascend910b 构建同样成功，产出 6 个 kernel 二进制——等长约束跨架构成立（来源平台：a2；证据等级：tool_observed）。

R7/R8 增量信号（本批两条独立 trace，Ascend950PR，CANN 9.1.0 与 9.2.0 实测）：

- R7（SparseFlashAttention 算子，CANN 9.2.0，固定 dtype 输入 sparseIndices 形态）：主输入 query/key/value/y 为 fp16/bf16 双 dtype，sparseIndices 固定 int32 只列 1 条目 → `bash build.sh --soc=ascend950` 连续多轮（含补 DataType 后、含 shell 侧 export OPS_PRODUCT_NAME 后）在 opbuild 处报同一错误；手工复现 `OPS_PRODUCT_NAME=ascend950 op_build <abs>.so <abs> "--compute_unit=ascend950"` 稳定 EXIT=1 并打印同一错误；**仅补 DataType 至 2 条目后仍失败**，Format/UnknownShapeFormat 补成 `{FORMAT_ND, FORMAT_ND}` 后 `-- Opbuild generating sources - done`；最终全量 clean build 成功（`[INFO] Build completed successfully!`，8 个 TilingKey 变体二进制生成）（来源平台：950；证据等级：tool_observed）。
- R8（TopK 算子，CANN 9.1.0，固定 dtype 输出 idx 形态）：输入 x 扩为 5 dtype（fp16/fp32/bf16/int8/uint8）、输出 idx 仍单条目 DT_INT64 → cmake 配置报 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessTopK is 0`（func.cmake:783）；最小化对照实验：精简到 x/y 各 2 dtype + idx 单条目 → 仍报；去掉 idx+Attr → 通过；加回单条目 idx → 复现失败；idx 改 2×DT_INT64 → 通过；最终 5 dtype + idx 5×DT_INT64 的完整工程 cmake 配置通过、`bash build.sh --soc=ascend950 -j8` 编译通过、UT 40/40（op_host 27 + op_api 13）通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "按索引 zip 组合 dtype"是对 opbuild 行为的经验性结论（等长即过、不等长即报，三个工程实验闭合），opbuild 内部判定为何落在 input[0] 的措辞上未核实；CANN 版本升级后建议重验（未验证，机制推断可迁移）。
- 组合数按输入 dtype 笛卡尔积展开时列表长度随输入数指数增长（本例 3×2=6）；更多 dtype 维度时显式枚举变得冗长但机制不变。
- 补齐等长后 kernel 侧模板实例化必须同步覆盖（否则进入下一类编译失败，如同工程的 TilingKey SEL 实例化问题）。
- 本卡只覆盖 dtype 列表长度约束；`.Follow()` 与等长显式列表混用时组合表的优先级未验证。

**未决项**：

- 【优先】读 opbuild（npu_op_code_gen）组合生成源码，把 "按索引 zip" 机制从因果实验上升为源码级结论，并确认 "dtype size" 报错文本的确切计算路径与指向逻辑。
- 未验证：`.Follow()` 与等长显式列表混用时组合表的优先级。
## 坏实践

- 各输入按自己的 dtype 集合独立写列表（3 项、2 项、1 项混用）——opbuild 不是按独立集合解析的。
- 用嵌套 vector 表达 dtype 组合映射（编译直接失败）；映射关系应下沉到 L2 校验层（CheckDtypeValid）。
- 只改报错指向的输入、不逐个核对全部 input/output 列表长度——换了报错措辞但过不了 configure；Format/UnknownShapeFormat 与 DataType 列表长度不同步补齐。
- （增量 R6）从单 dtype 算子模板复制 def.cpp 后只扩 `.DataType({...})` 不动 `.Format({...})`——所有端口 DataType 等长也不能豁免端口内三列表等长。
- （增量 R7）在带残留 `*.ini` 的输出目录里手工跑 opbuild 验证"修好了"——残留产物会让 EXIT=0 假成功，清理后同一命令同样 EXIT=1；手工复现必须先 `rm -f *.ini *.txt opbuild.log`。
- （增量 R8）反复怀疑并改动 CMakeLists include 路径、对比 `.so` 的注册符号、检查符号链接——这些方向与该错误无关（R8 轨迹实测排除过一轮），先数列表条目数。
- 看到 "dtype size is 0" 去查 tiling 计算或 kernel 代码——报错发生在 opbuild 代码生成阶段（op_host/CMakeLists.txt 的 npu_op_code_gen），与下游实现无关。

## 误诊

- 曾先后怀疑：optional 输入只列 1 种 dtype 导致、proto.h 残留 DT_BFLOAT16 导致、opbuild 对 DT_BF16 的尺寸表缺失导致——逐一排除（修 proto.h 后错误不变）后，才由等长展开实验闭合到列表长度不等这一根因。
- 曾先怀疑报错指向的 input[0]（主数据输入）自身 dtype/Format 数量问题：逐一数过其 3 个列表均正确，排除——错误措辞指向 input[0] 是误导。
- 曾怀疑 OPTIONAL_INPUT 的注册顺序 / proto.h 与 def.cpp 的 input 顺序差异：与 IR 索引无关，排除。

## 后续增量（2026-09-17，并行策展批：R9——MoeGatingTopKSoftmax 形态与 autogen ini 缺行探针）

R9（CANN 9.1.0，独立 trace，Softmax+TopK 融合门控算子 BenchNoHarnessMoeGatingTopKSoftmax）：输入 `x` 支持三种 dtype（fp16/fp32/bf16），可选输入 `finished` 只有 bool 一种，输出 `expert_idx`/`row_idx` 只有 int32 一种——直觉上"单 dtype 的端口写一条就够了"。`bash build.sh --soc=ascend950` 在 opbuild 阶段报同款 `opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMoeGatingTopKSoftmax is 0.`（func.cmake:783）。

**新判别探针（autogen ops-info.ini 缺行）**：cat `build/autogen/aic-ascend950-ops-info.ini`，本算子的 `input0` 段没有 dtype/format 行：

```text
[BenchNoHarnessMoeGatingTopKSoftmax]
input0.name=x
input0.unknownshape_format=ND,ND,ND
input0.shape=all
input0.paramType=required
input1.name=finished
input1.unknownshape_format=ND
...
```

而对照可正常构建的 add_example（其所有端口都是 2 条 dtype）的 ini，正常形态应有 `input0.dtype=float32,int32` 与 `input0.format=ND,ND` 这样的行——`inputN` 段缺 dtype/format 行即为注册信息解析为空，回 def.cpp 查条目数。

**修复（单 dtype 端口重复填充，verbatim before/after 节选）**：

```cpp
// before（端口条目数不一致：x/y 为 3，finished/expert_idx/row_idx 为 1）
        this->Input("finished")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
```

```cpp
// after（finished 扩成 DT_BOOL×3，expert_idx/row_idx 扩成 DT_INT32×3，与 x/y 的 3 条对齐）
        this->Input("finished")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

设计取舍：选择"重复单 dtype 枚举对齐条目数"而不是"把 x 砍成单 dtype"——算子规格要求 x 支持 fp16/fp32/bf16 三种输入；重复枚举在语义上与"该端口在所有 dtype 组合下均为同一类型"等价。可选输入和 int32 索引输出最容易漏（它们往往真的只有一种 dtype）。

R9 验证信号：

- 修复前：`opbuild ops error: The dtype size of input[0] of op BenchNoHarnessMoeGatingTopKSoftmax is 0.`，autogen ini 的 input0 段无 dtype/format 行（来源平台：950；证据等级：tool_observed）。
- 修复后：opbuild 通过（`aic-ascend950-ops-info.ini` 正常生成且含 dtype 行），host 侧 `cust_optiling`/`cust_opapi`/`cust_op_proto` 编译继续推进，构建错误推进到 op_api 源码的宏问题（另一根因）（来源平台：950；证据等级：tool_observed）。

R9 边界补充：`dynamicFormat.flag=true` 的场景（本算子为 false）未验证动态格式下的行为；报错指向 input[0]（第一条端口）但不指向真正"短"的端口，仍按全端口条目数清点。


## 后续增量（2026-09-17，并行策展批：R9——可选输出单条目形态与报错截断误判）

R9（去重类算子 dtype 扩展，Ascend950PR/arch35，CANN 9.1.0，独立 trace）：给已有算子扩展多 dtype 时，把输入 `x` 和输出 `y` 的 dtype 从单一 int32 扩成 `{DT_INT32, DT_FLOAT16}` 双条目，而**可选输出** `inverse`（逆索引，语义上恒为 int64）保持单条目 `{DT_INT64}` 不动——同错复现 `opbuild ops error: The dtype size of input[0] of op <Op> is`（func.cmake:783，CMake 配置阶段终止）。修复为把 `inverse` 补齐成等长双条目（dtype 重复同一个即可，Format/UnknownShapeFormat 同步补）：

```cpp
// 修复前（可选输出单条目，触发）：
    this->Output("inverse").ParamType(OPTIONAL)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
// 修复后（重复同一 dtype 补齐等长；语义不变，ops-info 层只要求列表等长，
// 多余组合不产生额外 kernel 变体——kernel 侧 TilingKey 由 ASCENDC_TPL_DATATYPE_SEL 独立控制）：
    this->Output("inverse").ParamType(OPTIONAL)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
```

两点判读补充：

- 本形态的错误信息在管道/日志截断下只剩 `The dtype size of input[0] of op <Op> is`（"is" 后丢失）——文本不完整不代表信息缺失，仍按本卡"数全部 IO 三个列表条目数"判别；构建命令不要用 `grep -i "error"` 过滤 opbuild 输出，保留完整报错行。
- 修复后核对面：`binary_info_config.json` 的 simplifiedKey 应覆盖 N(dtype) × 有无可选输出 类型的组合（本例 4 组合：int32/fp16 × inverse/no-inverse），kernel 二进制数 = dtype 组合数。

R9 验证信号：

- 修复前：`bash build.sh --soc=ascend950 -j8` 于 opbuild 阶段报 `The dtype size of input[0] of op <Op> is`（CMake Error at func.cmake:783，配置终止）（来源平台：950；证据等级：tool_observed）。
- 修复后：`[INFO] Build completed successfully!`，产出 2 个 kernel 二进制（int32/fp16）、4 个 simplifiedKey 组合（来源平台：950；证据等级：tool_observed）。

R9 边界：op_graph `REG_OP`（proto）侧的"同一 input 各 dtype 字节数一致"是另一条规则（见 [ACLNN 561102 单 dtype 原型注册卡](aclnn_561102_dtype_opbuild.md)），与本卡"条目数等长"不同根因，勿混。

## 后续增量（2026-09-17，并卡合并：gather 18 组合形态与 CrossEntropyLoss 9 组合形态）

#### 并入自 opbuild_dtype_size_zero_datatype_lists（gather 算子 18 组合形态、Format 对照证伪与二进制定位探针）

独立 trace（torch.gather 语义算子：数据源 x 支持 6 种 dtype、索引 index 支持 3 种、输出 y 跟随 x，合法组合 6×3=18 种；Ascend950PR + CANN 9.1.0）：

- 各端口独立写不等长列表（x/y 各 6 项、index 3 项）→ 同款 `opbuild ops error: The dtype size of input[0] of op <Op> is 0.`（func.cmake:783，尚未进入任何 .cpp 编译）。
- **对照实验（判别增量）**：仅把 Format/UnknownShapeFormat 扩到与**各自端口**的 DataType 等长（6/3/6 项）→ 报错逐字不变——证明"端口内 Format 跟随本端口 DataType 长度"不能替代"跨端口全部列表等长"；与上文 R6 的"端口内三列表等长"约束叠加成立：全部端口 × 三个列表的长度都须等于同一组合数 N，单补任何一列无效。
- 修复为 18 项索引对齐展开后，恰好生成 18 个 kernel 二进制变体（= 6 x dtype × 3 index dtype 合法组合，无 `y.dtype != x.dtype` 非法组合）——**kernel 变体数 = 等长列表长度**，可作修复后对账口径；组合数上百项时是否有更紧凑写法未验证、未尝试。
- 工具链探针补充：opbuild 是 stripped ELF 二进制（`<CANN>/tools/opbuild/op_build`），报错串可用 `grep -a "dtype size" <CANN>/tools/opbuild/op_build` 在二进制内定位。
- 误诊补充（并入）：曾按模板 add_example 中"Format 与 DataType 等长"的现象推测 Format 列表长度是根因——扩长后报错逐字不变证伪；应先数跨端口 DataType 条目数，而不是先动 Format。

验证信号（并入，均为 tool_observed）：

- 修复前：`bash build.sh --soc=ascend950 -j8` 于 CMake 配置阶段失败，`opbuild ops result: 1`，报错原文 `opbuild ops error: The dtype size of input[0] of op <Op> is 0.`（来源平台：950；证据等级：tool_observed）。
- 中间对照（假设证伪）：Format/UnknownShapeFormat 扩为与各自 DataType 等长（6/3/6 项）后重建，报错逐字相同（来源平台：950；证据等级：tool_observed）。
- 修复后终态：opbuild 通过（cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels 四 target 正常进入编译），kernel 二进制 18 个 `.o`（= 6 x dtype × 3 index dtype 合法组合），自定义算子包 `custom_opp_ubuntu_x86_64.run`（474542 字节）打包成功（来源平台：950；证据等级：tool_observed）。

#### 并入自 opdef_datatype_list_length_mismatch（CrossEntropyLoss 9 组合形态与模板 zip 旁证）

独立 trace（交叉熵损失类算子 BenchNoHarnessCrossEntropyLoss，`cmake -DASCEND_COMPUTE_UNIT=ascend950 ..`；Ascend950PR + CANN 9.1.0）：

- 三个端口列表长度 3/5/3 互不相等（input 3 dtype、target 5 dtype、loss 3 dtype，把各端口当"该端口支持的 dtype 集合"独立罗列）→ 同款报错。
- 修复为 9 组索引对齐展开（合法组合 = 3 种输入 dtype × 3 种标签形态：FP32/FP16/BF16 × {INT32 硬标签, INT64 硬标签, 与输入同 dtype 软标签}，输出 dtype 恒等于输入 dtype），`target` 列表按索引写成 `{INT32, INT64, FLOAT, INT32, INT64, FLOAT16, INT32, INT64, BF16}`。
- 模板旁证：add_example 参考模板中三个端口都是等长的 2 元素列表，zip 出 `FLOAT+FLOAT→FLOAT`、`INT32+INT32→INT32` 两组——模板本身就是"等长 zip"写法的官方示例，可作对照锚点。
- 软标签组合（索引 2/5/8）在该阶段只编译、未实现运行路径，上板行为未验证（并入边界）。

验证信号（并入）：

- 修复后：`rm -rf build && cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` opbuild 阶段通过，输出 `Opbuild generating sources - done` 并列出 cust_optiling/cust_opapi/cust_op_proto/ascendc_kernels 目标；最终全量构建产出 9 个按 TilingKey 区分的 kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 自定义算子包，clean build 从头复验通过（来源平台：950；证据等级：tool_observed）。