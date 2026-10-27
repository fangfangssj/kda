---
type: "Runbook"
title: "bf16 枚举实名 DT_BF16：全称 DT_BFLOAT16 不存在"
description: "CANN 各层 bf16 枚举实名是缩写：ge/op 层 DT_BF16、C 层 C_DT_BF16（=27）、AscendCL 层 ACL_BF16（=27）；全称 DT_BFLOAT16/C_DT_BFLOAT16/ACL_BFLOAT16 均不存在，写错按层分别报 not a member/undeclared；matmul_tiling::DataType 另有 DT_BFLOAT16=33 是不同值，与 ge 层对齐必须用 DT_BF16；编译器 did-you-mean 建议的 DT_FLOAT16 是语义陷阱。增量（8 个独立 trace）：dtype 声明分散 def/tiling 白名单/TilingKey 模板/L0/L2 op_api/proto 六层批量失败、报错逐轮暴露须一次清零；另有 no matching function（OpParamDef::DataType 重载决议失败）与 std::set 初始化转换失败两个掩盖性报错形态。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# bf16 枚举实名 DT_BF16：全称 DT_BFLOAT16 不存在

## 现象

为自定义算子声明 bf16 支持时，按"bfloat16 全称"直觉拼写枚举名，在四个独立工程的首次编译中分别触发同一族错误：

1. **op_host 算子定义**（`op_host/*_def.cpp`，`.DataType({ge::DT_FLOAT16, ge::DT_BFLOAT16})`）：

```text
op_host/bench_no_harness_conv_3d_backprop_filter_def.cpp:29:44:
error: ‘DT_BFLOAT16’ is not a member of ‘ge’; did you mean ‘DT_FLOAT16’?
   29 |             .DataType({ge::DT_FLOAT16, ge::DT_BFLOAT16})
```

2. **op_api 的 L0/L2**（`DataType::DT_BFLOAT16` 写进 `AICORE_DTYPE_SUPPORT_LIST`）：

```text
op_api/aclnn_bench_no_harness_moe_finalize_routing.cpp:41:57:
  error: 'DT_BFLOAT16' is not a member of 'op::DataType' {aka 'ge::DataType'}
   41 |     DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BFLOAT16
error: could not convert '{ge::DT_FLOAT16, ge::DT_FLOAT, <expression error>}'
  to 'const std::initializer_list<ge::DataType>'
```

（花括号初始化列表里混入不存在的枚举名，还会连锁报 `initializer_list` 转换失败，位置在后续行，别误判为列表写法问题。）

3. **op_kernel TilingKey 模板**（`ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_BFLOAT16, ...)`）：

```text
error: 'C_DT_BFLOAT16' was not declared in this scope; did you mean 'C_DT_FLOAT16'?
连锁: no matching function for call to 'ParamStruct::ParamStruct'
      no matching function for call to 'std::vector<ParamStruct>::vector'
```

（拼错的名字进入 `ASCENDC_TPL_DATATYPE_DECL` 变参后，宏展开成含未声明标识符的花括号初始化，连锁报 ParamStruct/vector 构造不匹配——后两条错误的位置在 `template_argument.h`，具有误导性。）

4. **op_host tiling 按 dtype 字符串直觉写比较**（`if (dtypeX == ge::DT_BFLOAT16)`）：

```text
op_host/arch35/bench_no_harness_conv_3d_backprop_filter_tiling.cpp:144:23:
error: 'DT_BFLOAT16' is not a member of 'ge'; did you mean 'DT_FLOAT16'?
```

5. **AscendCL C 接口**（测试代码创建 bf16 张量写 `ACL_BFLOAT16`）：

```text
test_aclnn_bench_no_harness_moe_finalize_routing.cpp:1120:25:
error: 'ACL_BFLOAT16' was not declared in this scope
```

（Mock 模式不包含 ACL 头时编译通过，切 Real 模式才暴露。）

## 适用场景

- 工程形态：自定义算子全栈（op_host def/tiling、op_api L0/L2、op_graph proto、op_kernel tiling_key 模板声明、C++ ST 测试）任何需要声明或比较 bf16 dtype 的位置。
- API 路径：`ge::DataType`（`graph/types.h`）、C 层 `C_DT_*`（`graph/c_types.h`）、`op::DataType`（aclnn/opdev）、`aclDataType`（`acl/acl_base_rt.h`）、`ASCENDC_TPL_DATATYPE_DECL/SEL` 的 C_DT_* 常量、`matmul_tiling::DataType`（`asc/include/tiling/matrix/matmul_tiling_base.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR / ascend950（DAV_3510、arch35），CANN 9.1.0；四个独立工程的编译闭环。
- 适用粒度：platform_wide——枚举拼写是 CANN 头文件级契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本的枚举表（若后续版本引入全称别名需重验，未验证）；`matmul_tiling::DataType::DT_BFLOAT16`(=33) 的确切用途未查证（未使用）；dvpp 像点格式的 `HI_PIXEL_FORMAT_*_BF16` 命名体系勿混淆。

## 根因

bf16 的合法枚举名在 CANN 各层都是缩写：`ge::DT_BF16`（`graph/types.h` 中 `DT_BF16 = ::C_DT_BF16` 一一映射自 `::C_DT_*` 全局 C 枚举）、C 层 `C_DT_BF16 = 27`、aclnn opdev 层 `op::DataType::DT_BF16`（与 ge 同名同值）、AscendCL 层 `ACL_BF16 = 27`——不存在任何一层的全称 `*BFLOAT16` 拼写（`matmul_tiling::DataType` 是唯一例外，见下）。数值上 `ACL_FLOAT=0、ACL_FLOAT16=1、ACL_INT32=3、ACL_BF16=27` 与 `ge::DT_*`/`C_DT_*` 同值，plog 里打印的 `dtype=1` 即 DT_FLOAT16/ACL_FLOAT16，排障时可互对照。

两个陷阱：

- **`matmul_tiling::DataType` 同时存在 `DT_BF16 = 27` 与 `DT_BFLOAT16 = 33` 两个不同值**。与 `ge::DataType` 对应时必须用 `DT_BF16`——见到 `matmul_tiling::DataType::DT_BFLOAT16` 能编译不代表 ge 层也能用。
- **编译器 nearest-name 建议是语义陷阱**：`did you mean 'DT_FLOAT16'` 给出的是另一个真实存在的枚举，照抄会把 bf16 判定静默变成永假比较（bf16 输入走 fp16 分支），留下语义 bug 而非编译错。

## 解决配方

头文件证据（工具检索原文）：

```text
$ASCEND_HOME_PATH/x86_64-linux/include/graph/c_types.h:18:  C_DT_FLOAT16 = 1,  // fp16 type
$ASCEND_HOME_PATH/x86_64-linux/include/graph/c_types.h:44:  C_DT_BF16 = 27,   // bf16 type
$ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h:83:    DT_FLOAT16 = ::C_DT_FLOAT16,
$ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h:108:   DT_BF16 = ::C_DT_BF16,
$ASCEND_HOME_PATH/include/acl/acl_base_rt.h:157:            ACL_BF16 = 27,
```

```text
--- matmul_tiling DataType--- （asc/include/tiling/matrix/matmul_tiling_base.h）
35: enum class DataType : int32_t {
37:    DT_FLOAT16 = 1,         // fp16 type
62:    DT_BF16 = 27,           // bf16 type
68:    DT_BFLOAT16 = 33,       // bf16 type（与 ge 层不同值，勿混用）
```

按层修正（修复前 → 修复后）：

```cpp
// op_host/*_def.cpp 与 op_api L0/L2（ge:: 与 op:: 层）
.DataType({ge::DT_FLOAT16, ge::DT_BFLOAT16})    // before
.DataType({ge::DT_FLOAT16, ge::DT_BF16})        // after

// op_graph/*_proto.h
.INPUT(x, TensorType({DT_FLOAT16, DT_BFLOAT16}))   // before
.INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))       // after

// op_kernel/arch35/*_tiling_key.h（C_DT_ 常量层）
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_BFLOAT16, ASCENDC_TPL_INPUT(0)),   // before
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),       // after

// op_host/arch35/*_tiling.cpp（ge:: 与 matmul_tiling:: 双层）
} else if (dt == ge::DT_BFLOAT16) {                          // before
    return matmul_tiling::DataType::DT_BFLOAT16;
} else if (dt == ge::DT_BF16) {                              // after
    return matmul_tiling::DataType::DT_BF16;

// AscendCL C 接口（ST 测试创建 bf16 张量）
aclCreateTensor(..., ACL_BFLOAT16, ...);   // before（不存在）
aclCreateTensor(..., ACL_BF16, ...);       // after
```

批量修正命令（可直接复用）：

```bash
cd <算子目录> && \
sed -i 's/ge::DT_BFLOAT16/ge::DT_BF16/g' op_host/*_def.cpp op_api/*.cpp && \
sed -i 's/DataType::DT_BFLOAT16/DataType::DT_BF16/g' op_api/*.cpp && \
sed -i 's/C_DT_BFLOAT16/C_DT_BF16/g' op_kernel/arch35/*_tiling_key.h && \
sed -i 's/ge::DT_BFLOAT16/ge::DT_BF16/g; s/matmul_tiling::DataType::DT_BFLOAT16/matmul_tiling::DataType::DT_BF16/g' op_host/arch35/*_tiling.cpp && \
sed -i 's/DT_BFLOAT16/DT_BF16/g' op_graph/*_proto.h && \
sed -i 's/ACL_BFLOAT16/ACL_BF16/g' tests/st/*.cpp
```

核对命令（不确定枚举名时直接搜头文件，比查文档快）：

```bash
rg -n "DT_BF" $ASCEND_HOME_PATH/x86_64-linux/include/graph/types.h
rg -n "ACL_FLOAT16|ACL_FLOAT|ACL_BF16|ACL_BF" $ASCEND_HOME_PATH/include/acl/acl_base_rt.h
# 交叉验证：grep 同仓既有算子 op_api 的既有用法
grep -rn "DT_BF16\|DT_BFLOAT16" <参考算子>/op_api/
```

### 增量（独立 trace：六层批量失败形态、掩盖性报错与全层清零纪律）

8 个独立 trace 的复现补充（均为 Ascend950PR / CANN 9.1.0）：

1. **dtype 声明跨五~六层，报错逐轮暴露**：OpDef（`op_host/*_def.cpp`）、tiling 白名单（`std::set<ge::DataType>` / dtype switch）、TilingKey 模板（`op_kernel/arch35/*_tiling_key.h`）、L0/L2 op_api 白名单、op_graph proto.h 至少五~六层各自独立编译——修完 host 层后下一轮在 op_api 层报同一错误（有轨迹三轮构建 + UT 阶段共三次复现）。处置纪律：一次 `grep -rn "BFLOAT16"` 全层清零，不要逐报错位置挤牙膏。
2. **`no matching function for call to 'ops::OpParamDef::DataType(<brace-enclosed initializer list>)'` 掩盖形态**：初始化列表里混入不存在的 `ge::DT_BFLOAT16` 后整条列表无法转换成 `std::vector<ge::DataType>`，编译器把失败点显示为 `DataType(...)` 的重载决议错误（note 指向 `OpParamDef& DataType(std::vector<ge::DataType>)`），真因（枚举名拼错）被错误文本掩盖——首见该报错先怀疑列表里有拼错的枚举成员，再怀疑语法。
3. **`std::set<ge::DataType>` 初始化连带失败变体**：`const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BFLOAT16};` 报 `could not convert '{...}' to 'const std::set<ge::DataType>'`——同为"非法枚举名污染整条初始化列表"。
4. **proto dtype 字符串不可直译**：proto/IR 里的 dtype 字符串是 `bfloat16`，按字面直译成 `DataType::DT_BFLOAT16` 是错误猜测（`float32` 直译成 `DT_FLOAT32` 同错，实名 `DT_FLOAT`）；常见引入路径是 op_def 用 `ge::DT_BF16` 能编过、op_api 另写 `DataType::DT_BFLOAT16` 两套写法不自洽。
5. **"踩过一次仍未免疫"**：同一工程首次踩坑修复后，后续迭代重新扩展 dtype 声明时又按 `DT_FLOAT16` 类推写出 `DT_BFLOAT16`（FP16 带 "FLOAT"、BF16 不带，命名不对称使类推几乎必然出错）——防靠流程（提交前 `grep -rn "BFLOAT16"` 必须为 0，可固化为提交检查项），不靠记忆。
6. **修复选全局改名而非 using/宏别名**：`C_DT_*` 出现在 `ASCENDC_TPL_DATATYPE_SEL`/`ASCENDC_TPL_DATATYPE_DECL` 宏展开里，别名无法覆盖宏上下文，直接改拼法是唯一干净解。
7. **ACL 层 grep 要点**：BF16 相关键在 `acl_base_rt.h`（`ACL_BF16 = 27`），只查 `acl_base.h` 无命中，易误判为"ACL 不支持 bf16 枚举"。

sed 全层一次替换的可用形态（覆盖 op_api/op_host/op_graph）：

```bash
cd operators/<op_pkg> && \
find op_api op_host op_graph -name "*.cpp" -o -name "*.h" | xargs sed -i 's/DT_BFLOAT16/DT_BF16/g' && \
grep -rn "DT_BFLOAT16" op_api/ op_host/ op_graph/   # 应为空
```

## 设计决策

- `matmul_tiling::DataType` 同时有 `DT_BF16=27` 与 `DT_BFLOAT16=33`：选 `DT_BF16`，因为 Host 侧 `ge::DataType` 的 bf16 值是 27（`DT_BF16 = ::C_DT_BF16`），tiling 层传入的 dtype 枚举值必须与 ge 层对齐，33 是该命名空间内另一个独立条目。
- 日志文案里的 "BFLOAT16"（如 `"Supported: FLOAT16, BFLOAT16."`）是字符串，不受影响，无需改动。
- 修复一次 grep 全仓再统一改，而不是只改报错的第一处——op_host/op_api/op_graph/tiling_key 各层是独立编译单元，会轮流报同样的错。

## 系统信息

- 工具链：CANN 9.1.0（x86_64-linux），GCC 13.3.0；目标 ascend950（arch35/DAV_3510）。
- 相关文件：`op_host/*_def.cpp`、`op_host/arch35/*_tiling.cpp`、`op_api/*.cpp`、`op_graph/*_proto.h`、`op_kernel/arch35/*_tiling_key.h`、`tests/st/*.cpp`。
- 事实源头文件：`include/graph/types.h`、`include/graph/c_types.h`、`include/acl/acl_base_rt.h`、`asc/include/tiling/matrix/matmul_tiling_base.h`、`aclnn/opdev`（op::DataType 与 ge 同名同值）。

## 预防

- 声明 bf16 支持前先查枚举表：`grep -n "DT_BF" .../graph/types.h`；从其他层复制枚举名时先确认目标层命名空间。
- 牢记三处一致：ge 层 `DT_BF16`、C 层 `C_DT_BF16`、tiling_key 模板层 `C_DT_BF16`；AscendCL 层 `ACL_BF16`。
- 遇到 "did you mean" 提示时，先确认建议项的语义是否与意图一致（DT_FLOAT16 ≠ bf16 的任何别名），再决定是否采纳。
- Mock/Real 双模式测试工程里，ACL 枚举只在 Real 分支可见——Mock 编译通过不代表 Real 可编译，bf16 相关代码至少跑一次 Real 编译。

## 验证信号

以下信号来自四个独立工程的 bf16 支持接入编译闭环（Ascend950PR / ascend950 目标，CANN 9.1.0）：

- Conv3D 反向算子四层修正：`ge::DT_BFLOAT16`/`C_DT_BFLOAT16`/`DataType::DT_BFLOAT16` 报错原文如现象节；全部 sed 修正后 `grep -rn "BFLOAT" op_host/ op_kernel/ op_api/ op_graph/` 仅剩日志文案命中、代码零残留；`cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` configure 通过，最终 build.sh 完整编译通过并产出 ascend950 二进制与算子包（来源平台：950 交叉编译；证据等级：tool_observed）。
- MoE FinalizeRouting op_api：修复前 `error: 'DT_BFLOAT16' is not a member of 'op::DataType'` + initializer_list 转换失败；参考算子 op_api grep 命中 `DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16`；修复后 op_api 编译通过、最终全量构建退出码 0（来源平台：950；证据等级：tool_observed）。
- MoE FinalizeRouting ST 测试：Real 模式编译报 `'ACL_BFLOAT16' was not declared`；`acl_base_rt.h` 实测 `ACL_FLOAT=0、ACL_FLOAT16=1、ACL_INT32=3、ACL_BF16=27`（与 ge::DataType 数值一致）；改为 ACL_BF16 后 Real 模式编译通过，bf16 用例上板 PASS（MERE=0.00e+00）（来源平台：950；证据等级：tool_observed）。
- op_host tiling 比较：`ge::DT_BFLOAT16` 构建失败（报错含 `did you mean 'DT_FLOAT16'?`）；改为 `ge::DT_BF16` 后同环境构建 `Build completed successfully!`（来源平台：950；证据等级：tool_observed）。
- TilingKey 模板声明：`C_DT_BFLOAT16` 报 undeclared 并连锁 ParamStruct/`std::vector<ParamStruct>` 构造失败；`graph/c_types.h:17-18/44` 确认 `C_DT_FLOAT=0、C_DT_FLOAT16=1、C_DT_BF16=27`；改名后 cmake configure 通过（来源平台：950 交叉编译；证据等级：tool_observed）。

增量信号（独立 trace，8 个独立 trace 的构建/测试闭环，均为 Ascend950PR + CANN 9.1.0）：

- foreach 标量除法类算子（六层全写错）：首轮 `def.cpp` 行 31/39/47/55、`tiling.cpp:85`、`proto.h` 报 `‘DT_BFLOAT16’ is not a member of ‘ge’`，`tiling_key.h:30` 报 `‘C_DT_BFLOAT16’ was not declared`；第三轮构建在 op_api L0 白名单复现同一错误；六层替换后 `--soc=ascend950` 构建成功，3 个 kernel .o × 2 缓冲模式 = 6 TilingKey 组合，binary_info_config 含 fp16(1)/fp32(0)/bf16(27) 三 dtype（来源平台：950；证据等级：tool_observed）。
- 同一算子全 dtype 迭代会话（同工程第二次独立踩坑）：六层替换后构建成功、UT 21/21、真实 NPU ST 65/65、PyTorch NPU 批量验收 65/65（来源平台：950；证据等级：tool_observed）。
- 归约索引类算子（ge/C 宏/op::DataType 三处拼法同轮失败）：6 处改名后宿主侧三目标（cust_optiling/cust_opapi/cust_op_proto）全部编译通过，后续完整构建生成 5 个 ascend950 kernel 二进制与 custom_opp 安装包（来源平台：950；证据等级：tool_observed）。
- Mish 类激活算子（sed 全局替换路线）：`sed 's/DT_BFLOAT16/DT_BF16/g'` 后 `grep -rn "DT_BFLOAT16"` 为空，下一轮构建该类错误消失（剩余为两个独立根因），最终 build exit 0（来源平台：950；证据等级：tool_observed）。
- TensorList p-范数算子（五类文件同步改名）：首编 `op_host/arch35/*_tiling.cpp:53` 报错 + `make Error 1` + `RuntimeError: compile ascend all ops failed`；五层改名后错误推进到下一独立根因；最终 `Build completed successfully!` 并生成 kernel 二进制（来源平台：950 交叉编译；证据等级：tool_observed）。
- Conv2d 基准算子（no matching function 掩盖形态）：`.DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BFLOAT16})` 报 `no matching function for call to 'ops::OpParamDef::DataType(<brace-enclosed initializer list>)'`；全链替换后该报错不再出现、编译推进到下一独立错误，`grep` 确认 `types.h` 实名 `DT_BF16 = ::C_DT_BF16`（来源平台：950；证据等级：tool_observed）。
- Elementwise 指数算子（proto 字符串直译路径）：op_api 白名单 `DataType::DT_BFLOAT16` 报 `'DT_BFLOAT16' is not a member of 'op::DataType' {aka 'ge::DataType'}`（伴随 `gert::Shape` 无 `ToString` 的独立小错）；修复后 op_host/op_api 编译通过，最终 3 个 dtype 变体全部 Done、custom_opp run 包生成（来源平台：950；证据等级：tool_observed）。
- 交叉熵损失算子 ST/UT（ACL 层）：ST `ACL_BFLOAT16` 报 was not declared、UT GE 层 `DT_BFLOAT16` 同族出现；改名后 ST/UT 编译通过该错误点（UT 14/14）、ST 进入 NPU Real 运行（来源平台：950；证据等级：tool_observed）。
- GroupNorm 类算子 ST（ACL 层 grep 要点）：`grep` 确认 `include/acl/acl_base_rt.h:157: ACL_BF16 = 27`（`acl_base.h` 无命中）；改名清理重建后 `[100%] Built target`、测试运行进入用例执行（来源平台：950；证据等级：tool_observed）。
- foreach 范数算子 ST（ACL_BFLOAT16 → ACL_BF16）：改名后 ST 全量编译通过并在 NPU Real 模式执行 BF16 用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 枚举拼写与数值（C_DT_BF16=27）绑定 CANN 9.1.0 头文件；版本升级后需重新核对 c_types.h/acl_base_rt.h（未验证，推断可迁移）。
- `matmul_tiling::DataType::DT_BFLOAT16`(=33) 的语义未验证；若某接口确实要求 33，本卡"对齐 ge 层一律用 DT_BF16"不适用。
- 本卡只覆盖枚举命名层；bf16 数据在 host 侧的位级转换、bf16 kernel 计算路径不在此范围。

**未决项**：

- `matmul_tiling::DataType::DT_BFLOAT16`(=33) 的确切用途未查证；如后续踩到可补独立记录。
## 坏实践

- 全局搜索替换式地把某一层的枚举拼写复制到另一层（或从设计文档照抄常量名），不做分层确认。
- 采纳编译器 `did you mean 'DT_FLOAT16'` 的建议——会把 bf16 判定静默变成永假比较。
- 只改一处命名空间就重编——各层独立编译单元会轮流报同样的错；两侧只裁一处（如只改 tiling_key.h 不改 def.cpp）会让失败推迟到运行期且形态迷惑。
- （独立 trace 补充）只修编译器当前报错的那一个文件就重构建——dtype 声明跨五~六层，漏层会在下轮构建复现；依赖"上次踩过坑的记忆"而不做提交前 grep 清零——同一工程二次踩坑已有实测。

## 误诊

- 初看 `cannot convert '<brace-enclosed initializer list>' to 'std::vector<ge::DataType>'` 曾被误判为 `.DataType({...})` 初始化列表写法问题；实际是列表里混入不存在的枚举名导致推导失败，枚举名修正后该错误同步消失。
- 从 op_graph proto.h 的 `TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16})` 拼写类推到 op_api——proto 头在本工程中未被编译验证（不参与构建目标），不能作为 op_api 侧依据；op_api 的权威拼写以 `ge::DataType` 定义与同仓既有 op_api 用法为准。
- （独立 trace 补充）`no matching function for call to 'ops::OpParamDef::DataType(...)'` 首见时误判为"三元素初始化列表不能转 std::vector / DataType 与 Format 元素数不匹配"之类的语法问题——实为列表含不存在的枚举成员导致整体转换失败；该误判为开发过程中的真实第一反应，后经查 types.h 排除。

## 后续增量（2026-09-17，并卡合并：aicpu 例外、跨层对照表、ACL 层细节与替换纪律）

#### 并入自 bf16_enum_dt_bf16_not_dt_bfloat16（aicpu 例外、dtype 跨层对照表与 sed 替换纪律）

- **aicpu 例外（不适用本卡替换方向）**：`aicpu/cpu_types.h:49` 中合法名恰是全拼 `DT_BFLOAT16 = 27`——aicpu（CPU kernel）上下文不能反换成 DT_BF16；`DT_BFLOAT16` 只在 aicpu 与 `matmul_tiling::DataType`(=33) 两套枚举中合法且值不同。tiling 的 `SetAType/SetBType` 统一用 `matmul_tiling::DataType::DT_BF16` 编译通过（实测）。
- **常用 dtype 跨层对照（写码前对照，aclDataType 属另一套头文件 acl_base_rt.h）**：

| 语义 | ge/op::DataType | aclDataType | C_DT_*（tiling_key 模板） | C++ 类型（kernel 实例化） |
|---|---|---|---|---|
| fp32 | `DT_FLOAT` | `ACL_FLOAT` | `C_DT_FLOAT`（=0） | `float` |
| fp16 | `DT_FLOAT16` | `ACL_FLOAT16` | `C_DT_FLOAT16`（=1） | `half` |
| bf16 | `DT_BF16` | `ACL_BF16` | `C_DT_BF16`（=27） | `bfloat16_t` |
| int8/int32/int64/uint8 | `DT_INT8/DT_INT32/DT_INT64/DT_UINT8` | 同名 `ACL_*` | 同名 `C_DT_*` | `int8_t/int32_t/int64_t/uint8_t` |

- **sed 批量替换纪律**：先替换长名再替换短名——`DT_BFLOAT16` 是 `C_DT_BFLOAT16` 的子串，替换顺序不当会把已正确的 `C_DT_BF16` 再改坏；先 `grep -rn "DT_BFLOAT16"` 盘点残留再统一 sed，替换后 grep 应为空。
- **漏改 proto.h 的下游形态**：proto.h 残留 `DT_BFLOAT16` 会在下一阶段以更难定位的 opbuild 运行期错误（`The dtype size of input[0] ... is 0`，见 [opbuild dtype 列表等长卡](opbuild_dtype_lists_must_be_equal_length.md) 的同批观测）复现；文档（DESIGN.md 等）同步替换，留着会误导后续迭代照抄错名。
- 同族陷阱：op_api 白名单按类型宽度直觉写 `DataType::DT_FLOAT32`（fp32 实名 `DT_FLOAT`）报 `'DT_FLOAT32' is not a member of 'op::DataType'`。
- 误诊补充（并入）：模板 tiling 宏场景曾怀疑 `ASCENDC_TPL_BOOL_DECL` 参数个数/宏用法错误——对照 template_argument.h 宏定义确认用法无误后，才把怀疑转向 DATATYPE 宏实参里的 `C_DT_BFLOAT16`，经 c_types.h 反查定案。

验证信号（并入，9 个独立开发轨迹，Ascend950PR，CANN 9.1.0 与 9.2.0）：

- 修复前：GRU 类算子首编 `'DT_BFLOAT16' is not a member of 'ge'`、`'C_DT_BFLOAT16' was not declared` 成组出现（def.cpp/proto.h/tiling_key.h/op_api 四处），伴随 brace-enclosed initializer list 转 `std::vector<ge::DataType>` 失败；改成显式 `std::vector<ge::DataType>{...}` 后仍失败（来源平台：950 编译链（CANN 9.2.0）；证据等级：tool_observed）。
- 修复前：模板 tiling 宏（ASCENDC_TPL_ARGS_DECL）展开处报 `std::vector<ParamStruct>` 构造候选不匹配（candidate expects 2/1/0 arguments, 3 provided），对照宏定义确认 BOOL 宏用法无误后定案为 `C_DT_BFLOAT16` 未声明（来源平台：950 编译链（CANN 9.1.0）；证据等级：tool_observed）。
- 头文件探针：`graph/types.h: DT_BF16 = ::C_DT_BF16`、`graph/c_types.h:44: C_DT_BF16 = 27`、`aicpu/cpu_types.h: DT_BFLOAT16 = 27`、`matmul_tiling_base.h: DT_BF16=27 与 DT_BFLOAT16=33 并存`（多轨迹 grep 一致）（来源平台：CANN 9.1.0/9.2.0 x86_64-linux 头文件；证据等级：tool_observed）。
- 修复后（稀疏注意力算子，CANN 9.2.0）：4 文件 13 处 sed 全局替换 + `--make_clean` 全量重编，DT_BFLOAT16 组错误 0 条，构建推进到下一阶段 opbuild（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（StridedSlice 类算子，CANN 9.1.0）：4 文件 11 处替换后 DT_BFLOAT16 系列报错全部消失，Host 侧（def/infershape/tiling）编译通过（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（TopK 类算子，CANN 9.1.0）：def.cpp/proto.h 同步替换后 npu_op_code_gen 阶段通过，错误推进到下一独立问题（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（Transpose 算子 op_api 白名单，CANN 9.1.0）：全量编译 `[100%] Built target binary`，CPack 生成 custom_opp .run 包（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（MlaProlog 算子 op_api，CANN 9.2.0 目录）：`[87%] Built target cust_opapi`，最终全量构建成功并打包（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（去重算子迭代三，CANN 9.1.0）：`DT_FLOAT32 → DT_FLOAT` 与 ACL 侧枚举修复后 `Build completed successfully!`，产出 7 个 kernel 二进制对象（7 dtype × 2 inverse = 14 TilingKey 后缀 `_0/_256,...,_27/_283`）（来源平台：950；证据等级：tool_observed）。
- 修复后（GQA 算子，CANN 9.1.0）：tiling_key.h/host/api/graph 三处批量替换后，后续三轮重建中 vector<ParamStruct> / DT_BFLOAT16 类错误不再出现（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后（MoE 重排算子，CANN 9.1.0）：五处替换（def/tiling_key/tiling switch/op_api/proto）完成后编译推进到下一阶段，最终 6 个 TilingKey 的 kernel 二进制与算子包编译成功（来源平台：950 编译链；证据等级：tool_observed）。

#### 并入自 acl_bf16_enum_not_acl_bfloat16（ACL 层非 scoped 枚举、Mock 掩盖机制与枚举取值）

- **aclDataType 是普通 enum（非 enum class）**：`ACL_BF16` 全局作用域可见，写 `aclDataType::ACL_BF16` 作用域限定在部分编译器配置下同样报 not-a-member——直接用裸名。
- **枚举取值节选（CANN 9.1.0/9.2.0 实读）**：`ACL_FLOAT=0, ACL_FLOAT16=1, …, ACL_BF16=27, ACL_INT4=29, ACL_FLOAT8_E5M2=35, …} aclDataType;`。
- **Mock/Real 双模式放大机制（并入细化）**：dtype 映射函数常位于 `#ifndef USE_MOCK_ACLNN` 段内（只有 Real 分支编译真实 ACL 头），且 mock 头可能自造 `ACL_BFLOAT16` 拼写——Mock 全绿完全掩盖仅 Real 可见的编译错误。设计决策：以真实头为准修测试而不是给 mock 头加别名（mock 的职责是模拟真实接口，枚举名分叉本身就是 mock 失真；dtype 这类小集合可让 mock 直接 `#include` 真实基础头）；用精确枚举名而不是魔法数 27（27 是版本相关布局）。
- 一处映射常有多处引用（dtype 映射、元素大小分支 `dataType == ACL_BFLOAT16`、组合判断 `ACL_FLOAT16 || ACL_BFLOAT16`），全局 grep 后一次替换干净。
- 修复形态（ST dtype 映射，verbatim）：`case TestDtype::BFLOAT16: return ACL_BFLOAT16;` → `case TestDtype::BFLOAT16: return ACL_BF16;`；host 侧示例 `aclDataType::ACL_BFLOAT16` → 裸名 `ACL_BF16`。

验证信号（并入，10 个独立开发轨迹，Ascend950PR，CANN 9.1.0 与 9.2.0）：

- 修复前：Real 模式编译报 `'ACL_BFLOAT16' was not declared in this scope; did you mean 'ACL_FLOAT16'?`（映射函数 691/938/74/76 行等多处）；同一代码 Mock 模式编译通过且 9/9 golden 自测通过（来源平台：950 开发环境（CANN 9.1.0）；证据等级：tool_observed）。
- 头文件事实：`acl_base_rt.h:157: ACL_BF16 = 27`，枚举内无任何 `ACL_BFLOAT16` 记号（多轨迹 grep 一致；CANN 9.1.0 与 9.2.0 头均实测）（来源平台：CANN x86_64-linux include 树；证据等级：tool_observed）。
- 修复后：ST Real 模式编译链接成功，92 条用例（L0=12 + L1=80，含 FP16/BF16）进入 NPU 执行（来源平台：950（CANN 9.1.0）；证据等级：tool_observed）。
- 修复后：host 侧 aclnn 示例改 `ACL_BF16` 后编译通过并产出可执行文件（来源平台：950 开发环境（CANN 9.2.0）；证据等级：tool_observed）。
- 修复后：多算子 ST（MoeReRouting/GRU/ArgMax/Mish/Sigmoid/MoeGatingTopKSoftmax/GQA/DepthwiseConv2d 等）Real 模式 `[100%] Built target test_aclnn_<op>`，bf16 用例随后进入真机执行（后续精度问题为独立根因）（来源平台：950（CANN 9.1.0/9.2.0）；证据等级：tool_observed）。
- 交叉佐证：bf16 三套名字 host 侧 `ACL_BF16`、kernel 侧 `C_DT_BF16`、ge 侧 `ge::DT_BF16` 均为 27，没有一处是全拼 `BFLOAT16`（来源平台：CANN 头文件实读；证据等级：tool_observed）。

坏实践补充（并入）：

- 凭 torch 生态的 `bfloat16` 命名惯性或 `DT_FLOAT16` 全拼规律臆造 `DT_BFLOAT16`/`DT_FLOAT32`/`ACL_BFLOAT16`；报错后凭直觉猜枚举名（如顺手写 `ACL_BFLOAT`）而不 grep 真实头。
- mock 头自造与真实头不一致的枚举拼写，靠"Mock 过了"当作编译已验证。
- sed 替换先短名后长名——子串误伤把正确的 `C_DT_BF16` 改坏。

失效边界补充（并入）：

- aicpu 上下文（`aicpu/cpu_types.h`）合法名是 `DT_BFLOAT16 = 27`，不适用本卡替换方向。
- 本卡并入信号覆盖 CANN 9.1.0 与 9.2.0；`DT_BF16 = 27` 等数值不要硬编码进业务代码，仅作检索锚点。