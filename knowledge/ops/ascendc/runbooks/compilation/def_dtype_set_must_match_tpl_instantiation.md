---
type: "Runbook"
title: def dtype 集合与 TPL 实例化须双向同步
description: 自定义算子 def.cpp 的 DataType 集合与 ASCENDC_TPL 模板实例化集合必须双向一致：声明了未实例化的 dtype 在 opbuild 阶段报 "The dtype size of input[0] is 0" 构建失败；只扩 tiling_key.h 不扩 def 则该 dtype 静默无二进制（构建成功但缺 .o 与 binList 条目）。def 是二进制生成的事实源。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# def dtype 集合与 TPL 实例化须双向同步

## 现象

TopK 类自定义算子（Ascend950PR / arch35，CANN 9.1.0，TilingKey 模板编程）分迭代扩展 dtype（5 种 → 7 种）时，两个方向各踩一次"集合不一致"：

**方向一（def 声明 ⊃ 模板实例化）→ 构建显式失败**。迭代一 op_host 的算子定义文件按接口规格声明了全部 7 种 dtype（int8/uint8/int32/int64/fp16/fp32/bf16），而 op_kernel 侧 TilingKey 模板参数（`ASCENDC_TPL_DATATYPE_DECL` 声明的 D_T_X）只有 fp16 实例化。执行 `bash build.sh --soc=ascend950 -j8`，构建在 `npu_op_code_gen` 的 opbuild 阶段直接失败：

```
CMake Error at .../cann-9.1.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:783 (message):
  opbuild ops error: The dtype size of input[0] of op BenchNoHarnessTopK is
  0.
Call Stack (most recent call first):
  op_host/CMakeLists.txt:51 (npu_op_code_gen)
```

错误信息很短且不指向具体源码行。

**方向二（模板声明 ⊃ def 注册）→ 静默缺核**。迭代三已把 `op_kernel/arch35/<op>_tiling_key.h` 的 `ASCENDC_TPL_DATATYPE_DECL` 扩到 7 种 dtype、`ASCENDC_TPL_SEL` 扩到 14 组（7 dtype × 2 inner mode），并改完 op_host tiling，但 def.cpp 仍处于迭代二状态（5 dtype）。构建**成功**（`[INFO] Build completed successfully!`），无 error 无告警，但检查产物：`.o count = 5`（应为 7，每 dtype 一个二进制），binList 只有 5 条（int8/float16/bfloat16/float32/uint8），int32/int64 完全缺失。`--make_clean` 全量重建仍是 5 个，排除增量缓存。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子仓开发，采用 TilingKey 模板编程（`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_DATATYPE_DECL`）+ 分迭代扩展 dtype 的策略。
- API 路径：`ops::OpParamDef::DataType({...})` 注册集合（`op_host/<op>_def.cpp`，及 op_graph proto 的 TensorType 集合）vs `op_kernel/arch{N}/<op>_tiling_key.h` 的模板 dtype 声明与 `ASCENDC_TPL_SEL` 选择逻辑；构建产物 `build/op_kernel/ascendc_kernels/binary/config/<soc>/<op>.json` 的 binList。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有用 TilingKey 模板做多 dtype 分发的自定义算子。
- 路线优先级：recommended_fastpath（两侧同步是工具链的硬约束，无替代路径）。
- 明确不适用 / 未验证场景：不用模板分发、单 dtype 算子不涉及；其他 CANN 版本未验证。

## 根因

两个方向的机制：

1. opbuild 为 def 里声明的每个 dtype 计算输入的 dtype size 时，依赖 kernel 侧 TilingKey 模板为该 dtype 生成的实例化信息；"def 声明了、模板未实例化"的 dtype 其 dtype size 计算结果为 0，opbuild 将其作为错误拦截。因此 def 的 `DataType({...})` 集合必须与 TilingKey 模板实例化集合保持一致，分迭代扩展时两侧必须同步收敛（声明多少就实例化多少）。
2. opbuild 生成哪些 kernel 二进制由 `op_def`（`<op>_def.cpp`）注册的 dtype 列表推导（生成 ops-info/binList），**不是**由 `tiling_key.h` 的 `ASCENDC_TPL_DATATYPE_DECL/ASCENDC_TPL_SEL` 推导。def.cpp 未扩时，opc 只为 def 注册的 dtype 生成二进制；构建不报错、不告警，"静默缺核"。
3. 另外 opbuild 要求所有输入/输出的 dtype/Format/UnknownShapeFormat 列表长度一致——输出为固定 dtype 时（如 idx 固定 `ge::DT_INT64`）需重复 N 次与输入列表等长。

## 判别方法

1. 方向一报错特征：`opbuild ops error: The dtype size of input[N] of op <OpName> is 0.`，出现在 npu_op_code_gen/opbuild 阶段而非具体编译行。
2. 方向二判别：构建"成功"后固定执行产物核对——`.o` 计数 + binList dtype 清单（见下述验证命令）；缺条目即 def 未扩。
3. 对账两份清单：
   - def 侧：`grep -A3 'DataType({' op_host/<op>_def.cpp`（以及 op_graph proto 的 TensorType 集合）；
   - kernel 侧：`grep "ASCENDC_TPL_DATATYPE_DECL" op_kernel/arch{N}/<op>_tiling_key.h`；
   - 两个集合的差集里的 dtype 即问题候选（方向一差集在 def 侧多出、方向二差集在 tiling_key 侧多出）。

## 解决配方

**方向一修复**（迭代一 def 只声明 fp16，与 TilingKey 模板实例化集合严格一致；后续迭代扩展时 def 与 TilingKey 同步加）：

```cpp
        // 输入 x：迭代一仅声明 fp16（与 TilingKey 模板参数一致，后续迭代扩展 7 种 dtype）
        // dtype 列表需与 tiling_key.h 的 ASCENDC_TPL_DATATYPE_DECL 保持一致
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 y：topk 值，dtype/format 跟随 x
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
```

（输出 idx 固定 `ge::DT_INT64`，不随迭代扩展，无需改动。）

**方向二修复**（def.cpp 补齐 int32/int64；Input/y 列表扩到 7 项、Format 与 UnknownShapeFormat 同步扩到 7 项、idx 的 DT_INT64 重复 7 次）：

```cpp
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8,
                       ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8,
                       ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

构建后核对二进制数量与 binList dtype（固定动作）：

```bash
cd operators/<op> && bash build.sh --make_clean 2>&1 | tail -3
bash build.sh --soc=ascend950 -j8 2>&1 | rg -i "generating|error|fail" | head -20
ls build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o | wc -l
python3 -c "
import json
d = json.load(open('build/op_kernel/ascendc_kernels/binary/config/ascend950/<op>.json'))
for i, b in enumerate(d.get('binList', [])):
    print(f'  [{i}] dtype={b[\"inputs\"][0][\"dtype\"]}')
print(f'Total: {len(d.get(\"binList\", []))} entries')"
```

## 设计决策

- 放弃"接口先行"（def 先声明全量 dtype 保持接口完整、kernel 后补）的策略：当前工具链下 def 声明集合会被 opbuild 逐 dtype 校验，声明未实例化的 dtype 直接挡住构建，拦截逻辑根本到不了运行时。替代策略是"实现多少声明多少"：接口文档/需求侧仍可描述目标全量 dtype 集合，但 def/proto/tiling_key 三者的机器可执行集合按迭代同步收敛；未支持 dtype 的运行时拦截交给 aclnn 层 `CheckDtypeValid`（op_api 侧参数校验），而不是 def 声明。
- 每个 dtype 只有一个 `.o` 而不是每组 SEL 一个：opbuild 按 dtype 归并模板实例，同一 dtype 的 small/large inner 等 TilingKey 组合共享一个二进制，通过 binList 条目的多个 `simplifiedKey` 映射到同一 bin 文件；核对产物时按 dtype 计数而不是按 SEL 条目计数。
- "binList 按 def 推导、与 ASCENDC_TPL_SEL 无关"的机制基于控制变量观测（tiling_key.h 已扩而 def 未扩 → 缺核；def 扩后 → 有核），未读 opbuild 源码证实推导链路。

## 预防

- 采用 TilingKey 模板分发时，把"def DataType 集合 == proto TensorType 集合 == ASCENDC_TPL_DATATYPE_DECL 实例化集合"写成检查项，每次扩展 dtype 时三处同步改。
- 新增对外 dtype 时把"`op_def` 注册表"列为第一检查项：构建"成功"不代表新 dtype 有二进制，构建后固定执行 `.o 计数 + binList dtype 清单`核对。
- dtype 列表、Format 列表、UnknownShapeFormat 列表三者必须等长同步扩展，输出为固定 dtype 时用重复项补齐。
- opbuild 阶段的短错误（`dtype size ... is 0`）优先怀疑"声明了但未实例化"的集合不一致。

## 验证信号

来源说明：该算子迭代一（方向一）与迭代三（方向二）int32/int64 扩展的构建环节实测，每次改动全量重建。

- 方向一修复前：`opbuild ops error: The dtype size of input[0] of op BenchNoHarnessTopK is 0.`（func.cmake:783，npu_op_code_gen 阶段）（来源平台：950；证据等级：tool_observed）
- 方向一修复后：def 收敛为只声明 fp16，重新构建不再出现 dtype size 0 错误，最终构建成功（来源平台：950；证据等级：tool_observed）
- 方向二修复前：`bash build.sh --make_clean` 全量重建后仍只有 5 个 `.o`、binList 5 条（int32/int64 缺失）；构建日志仅 4-5 条 `Generating BenchNoHarnessTopK_*`，无 error/fail；`rg "DataType" op_host/bench_no_harness_top_k_def.cpp` 显示 Input/y 仍为 5 项（来源平台：950；证据等级：tool_observed）
- 方向二修复后：`--make_clean` 全量重建产出 7 个 `.o`，新增 `Generating BenchNoHarnessMoeTopK_f0f0c0793b19d6e06c99145b06757c60 ...`；binList 7 条含 int32/int64；安装包重装后 `$ASCEND_OPP_PATH/opp/vendors/.../config/ascend950/` 内核配置含 int32/int64（各 4 个 simplifiedKey）（来源平台：950；证据等级：tool_observed）

## 失效边界

- "声明集合必须与实例化集合一致"与"binList 按 def 推导"均验证于 cann-9.1.0 的 opbuild 行为；其他版本行为未验证（推断可迁移）。
- 本卡只闭合"构建产物生成"。def.cpp 补齐后上板仍可能报 561103（L0 API dtype 白名单未扩，另卡处理）；ST 数据超 fp32 表示域仍会精度失配。
- 同一 dtype 多 TilingKey 归并单二进制的归并规则（simplifiedKey 的 `d=/p=` 字段完整语义）未逐字段核对。
- opbuild 由 def 推导实例集的确切源码级机制（哪一步读 ops-info、是否可配置）未追溯。

**未决项**：

- 无（该根因已双向闭环到 7 dtype 全量二进制生成；后续迭代扩展时的三处同步是既定策略）。
## 坏实践

- "接口先行"：def 声明全量 dtype 而 kernel 只实现一部分，指望 tiling 阶段报错拦截——构建直接失败在 opbuild。
- 只扩展 `ASCENDC_TPL_DATATYPE_DECL/ASCENDC_TPL_SEL` 和 tiling/kernel 代码就认为 dtype 已接入——def.cpp 才是二进制生成的开关，漏改时构建静默成功。
- 用增量构建判断模板实例是否生成：本例第一次误判为"增量编译缓存"，`--make_clean` 全量重建后仍是 5 个，才排除缓存假说。
- 看到短错误不查集合对账，转而去改 CMake 或重装环境。

## 误诊

- 方向二首诊怀疑增量编译缓存/构建系统未检测到 tiling_key.h 变化，先做了 `--make_clean` 全量重建（无效）；随后怀疑 `C_DT_INT32/C_DT_INT64` 常量未定义（若未定义编译会报错，与"构建成功"矛盾，自行排除）。
