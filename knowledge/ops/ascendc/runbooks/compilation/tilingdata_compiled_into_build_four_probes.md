---
type: "Runbook"
title: TilingData 编入构建四探针：opParaSize=8+sizeof
description: 改动 TilingData 结构体后用四个构建产物探针确认真正编入：构建日志无注册警告、每 .o mocker 串 ≥1、kernel json opParaSize=8+sizeof(struct)、.ascendc_tiling section 内 sizeof 元数据。
tags: [task.testing, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TilingData 编入构建四探针：opParaSize=8+sizeof


## 现象

在 AscendC 算子仓工程里把 TilingData 结构体从 4 字节单字段桩（`uint32_t totalLength;`）换成 200 字节完整布局（23 字段共用 struct + 9 字段 empty struct，带类内初始化器）后重新构建：构建退出码 0 只能证明类型合法、编译通过，**不能**证明新结构体的 sizeof 真的进了内核二进制元数据、框架 tiling 缓冲按新大小分配。本轮在构建产物上跑了四个探针自检：构建日志无 tiling 注册告警、每个 .o 的 `__AUX__STRUCT_ascendc_struct_mocker_` 计数 ≥1、kernel json 的 opParaSize、objdump 的 `.ascendc_tiling` section 内容。实测观察到：替换前（4 字节 struct）kernel json `opParaSize=12`；替换后（200 字节 struct）3 个 dtype 变体 json 全部 `opParaSize=208`——两个数据点吻合经验式 **opParaSize = 8 + sizeof(REGISTER_TILING_DEFAULT 注册的结构体)**。

## 适用场景（必填 — 检索前的第一道滤网）

- **工程形态**：算子仓构建验证（ascendc_kernels binary 构建之后、aclnn 联调之前的产物自检）；适用于任何改动了 TilingData 结构体或注册方式的轮次。
- **API 路径**：`operators/build.sh`（cmake + `make -j binary package` + .run 安装）；构建产物 `build/ascendc_kernels/binary/<soc>/<snake_case 算子名>/*.json` 与 `*.o`；`strings` / `objdump` / `grep`。
- **验证过的平台与 CANN 版本**：ascend950 + CANN cann-9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度（transfer_scope）**：`platform_wide`——凡走 ascendc_kernels binary 构建的算子仓工程（机制在 CANN 注册宏与构建系统，与具体算子无关）；本卡验证于单算子单平台。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用 / 未验证场景**：多 struct（`REGISTER_TILING_FOR_TILINGKEY`）场景的 opParaSize 语义未验证；TTK/其他构建体系未验证；告警串与 mocker 串字面量以 cann-9.0.0-beta.2 日志/产物为准，其他版本需复核。

## 判别方法

**触发条件**：改了 TilingData 结构体（字段增删/换布局）或注册方式之后，需要确认结构体真正"编入工程构建"。

**探针（构建后执行；`<op_dir>` = `build/ascendc_kernels/binary/<soc>/<snake_case 算子名>`）**：

```bash
# 探针 1：构建日志无 tiling 注册告警（注意 CANN 告警原文就是 "registe"，不要"纠正"拼写）
grep -ci "do not registe tiling struct" /tmp/<build>.log        # 期望 0

## 探针 2：每个 .o 的 tiling 反序列化调用点（mocker 串）计数 ≥ 1
for f in <op_dir>/*.o; do
    printf "%s: " "$(basename "$f")"
    strings "$f" | grep -c "__AUX__STRUCT_ascendc_struct_mocker_"
done                                                              # 期望每个 .o ≥ 1

## 探针 3：kernel json 的 tiling 缓冲大小
grep -o '"opParaSize": [0-9]*' <op_dir>/*.json                    # 期望 8 + sizeof(注册的 struct)

## 探针 4：注册 struct 的 sizeof 元数据 section
objdump -h <op_dir>/<某>.o | grep -i ascendc_tiling               # 期望 .ascendc_tiling.<StructName>，大小 0x10
objdump -s -j .ascendc_tiling.<StructName> <op_dir>/<某>.o        # 期望首 8 字节（小端）= sizeof(Struct)
```

**机制（探针为什么有效）**：

- 探针 4 的来源——`REGISTER_TILING_DEFAULT(t)` 的宏展开（CANN 头文件 `x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h` 第 363-392 行，verbatim；中间第 378-390 行为与本卡无关的 ENABLE_DETERMINISTIC / KERNEL_TASK_TYPE 宏，此处未摘）：
  ```c
  #define TILING_STRUCT_SECTION_INIT_BASE(counter, val)                                                                \
      static const uint64_t __ascendc_tiling_struct_##counter __attribute__((used, section(".ascendc_tiling."#val))) = \
      sizeof(val)
  #define TILING_STRUCT_SECTION_INIT(counter, val)       TILING_STRUCT_SECTION_INIT_BASE(counter, val)

  #ifdef __CHECK_FEATURE_AT_PRECOMPILE
  #define ENABLE_FEATURE_FOR_COMPILE(f, val) auto __enable_feature_for_compile_##f = val
  #define ENABLE_FEATURE_FOR_TILING(expression, val) auto __enable_custom_tiling val = expression
  #define REGISTER_NONE_TILING auto __enable_no_register_custom_tiling ascendc_trigger_tiling_struct = default
  #else
  #define ENABLE_FEATURE_FOR_COMPILE(f, val)
  #define ENABLE_FEATURE_FOR_TILING(expression, val) TILING_STRUCT_SECTION_INIT(__COUNTER__, val)
  #define REGISTER_NONE_TILING
  #endif

  #define REGISTER_TILING_DEFAULT(tiling_struct)  ENABLE_FEATURE_FOR_TILING(default, tiling_struct)
  #define REGISTER_TILING_FOR_TILINGKEY(expression, tiling_struct)  ENABLE_FEATURE_FOR_TILING(expression, tiling_struct)
  ```
  即在 section `.ascendc_tiling.<StructName>` 放一个 `static const uint64_t` = `sizeof(Struct)`——objdump 直接读它。
- 探针 3 的公式——**opParaSize = 8 + sizeof(注册 struct)**：本轮两个实测数据点：4 字节桩 struct → opParaSize=12；200 字节 struct → opParaSize=208（3 个 dtype 变体 json 一致）。"+8" 头部的确切字段构成未从源码确认（推测与 tilingKey/头部字段相关，见 ## 未决事项）。
- 探针 2 的 mocker 串——来自 `GET_TILING_DATA_WITH_STRUCT` 的 codegen 展开：宏定义不在普通头文件（grep 整个 CANN 安装仅命中 `libaihac_codegen.so` 与 pkg_inc 内置算子源）；kernel_meta 生成头内可见 `InitTilingData<T>` 模板（裸字节反序列化）。本轮 1 个调用点 → 每 .o 计数 1；多调用点（多分支 kernel）场景的计数关系未验证。
- 探针 1 的告警——构建系统在 kernel 未注册 tiling struct 时打 `do not registe tiling struct`（原文拼写如此）；grep 必须按原文，做拼写放缩会漏匹配。

**因果链（为什么必须看 opParaSize 而不只看退出码）**：host 侧 `gert::TilingContext::GetTilingData<T>()` 在 tiling 缓冲容量不足时返回空指针（`exe_graph/runtime/tiling_context.h`，grep 原文）：

```text
390:  auto GetTilingData() -> T* {
391:    auto tiling_data = GetRawTilingData();
395:    if (tiling_data->GetCapacity() < sizeof(T)) {
```

若 struct 长大而 opParaSize 没跟着涨，`GetTilingData` 返回空指针，TilingFunc 返回 GRAPH_FAILED，aclnn 用例必挂；反之，aclnn 冒烟通过（桩在空指针时返回 GRAPH_FAILED）可反证缓冲按新 sizeof 分配。即：**探针 3 + 冒烟通过 = 结构体尺寸贯穿 host 缓冲分配的完整证据**。

## 设计决策

- 四探针各覆盖一层装配路径：告警（注册存在性，构建系统层）、mocker 串（设备侧反序列化调用点，codegen 层）、opParaSize（框架 tiling 缓冲尺寸，元数据层）、section（sizeof 元数据，目标文件层）——单看退出码 0 只覆盖"编译通过"这一层。
- 告警串按 CANN 日志原文 grep（含 "registe" 拼写），不做模式放缩。
- opParaSize 公式按两点经验式记录（4→12、200→208），不外推到未验证场景（多 struct 注册）。
- 观测时机：构建日志重定向到 /tmp 后 grep；探针 4 的 `objdump -s` 只看尾部 3 行（section 内容行）。

## 系统信息

- **toolchain**：CANN cann-9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`）；构建命令 `( set -o pipefail; cd operators && ./build.sh <package_name> )`（≥600s 超时，日志重定向）。
- **目标平台**：ascend950（产物目录 `build/ascendc_kernels/binary/ascend950/<snake_case 算子名>/`）。
- **相关文件/产物**（与实际操作一一对应）：
  - `/tmp/<构建日志>.log`（探针 1 输入）
  - `build/ascendc_kernels/binary/ascend950/<op>/*.o`（探针 2/4 输入）
  - `build/ascendc_kernels/binary/ascend950/<op>/*.json`（探针 3 输入，每个 dtype 变体一份）
  - `<CANN>/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`（宏定义，机制佐证）
  - `build/<Op>_ascend950/kernel_0/kernel_meta_*.h`（codegen 生成的 InitTilingData，机制佐证）
  - `<CANN>/x86_64-linux/include/exe_graph/runtime/tiling_context.h`（容量检查，机制佐证）

## 完整代码

本轮实际执行的探针命令（verbatim，前一条为单行命令原文；末行为随后单独执行的 section 内容查看）：

```bash
grep -ci "do not registe tiling struct" /tmp/thnn_tilingdata_build.log; echo "--- mocker count per .o ---"; for f in operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/*.o; do printf "%s: " "$(basename "$f")"; strings "$f" | grep -c "__AUX__STRUCT_ascendc_struct_mocker_"; done; echo "--- opParaSize / tiling section ---"; grep -o '"opParaSize": [0-9]*' operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/*.json; objdump -h operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o | grep -i ascendc_tiling
objdump -s -j .ascendc_tiling.ThnnFusedGruCellGradTilingData operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o | tail -3
```

对应的原始观测输出（verbatim 摘录）：

```text
0                                              # 探针 1：告警计数
--- mocker count per .o ---
ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o: 1
ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.o: 1
ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.o: 1
--- opParaSize / tiling section ---
...ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.json:"opParaSize": 208
...ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.json:"opParaSize": 208
...ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.json:"opParaSize": 208
  2 .ascendc_tiling.ThnnFusedGruCellGradTilingData 00000010  0000000000000170  0000000000000170  00000220  2**3

Contents of section .ascendc_tiling.ThnnFusedGruCellGradTilingData:
 0170 c8000000 00000000 c8000000 00000000  ................
```

（`c8000000 00000000` 小端即 0xC8 = 200 = sizeof(新 struct)；替换前同一 section 已存在，json 为 `"opParaSize": 12`。）

## 性能影响

无（构建产物元数据自检，无运行性能数据）。

## 预防

- 改 TilingData 结构体/注册方式后的标准自检清单 = 四探针 + aclnn 冒烟；先算好 opParaSize 期望值（8 + sizeof(注册 struct)）再对。
- 新增 kernel 入口（apt.cpp）时，`GET_TILING_DATA_WITH_STRUCT` 与 `REGISTER_TILING_DEFAULT` 必须成对出现——探针 2 计数 0 或探针 1 告警非 0 即装配断裂。

## 验证信号

来源说明：本验证信号取自本轮 tiling 结构体落地开发会话——结构体替换前后对同一工程构建产物的实测对比。

- 替换前基线：4 字节桩 struct 的 kernel json `"opParaSize": 12`（来源平台：950；证据等级：tool_observed）
- 替换后：3 个 dtype 变体 kernel json 的 opParaSize 全部 = 208 = 8 + 200（来源平台：950；证据等级：tool_observed）
- 构建日志 "do not registe tiling struct" 计数 0（来源平台：950；证据等级：tool_observed）
- 3 个 .o 的 `__AUX__STRUCT_ascendc_struct_mocker_` 计数各 1（来源平台：950；证据等级：tool_observed）
- `.ascendc_tiling.ThnnFusedGruCellGradTilingData` section（16 字节，0x10）内容首 8 字节小端 0xC8 = 200 = sizeof（来源平台：950；证据等级：tool_observed）
- aclnn 冒烟 1/1 PASSED、FAILED 计数 0（空指针会 GRAPH_FAILED → 冒烟通过反证 tiling 缓冲按新 sizeof 分配）（来源平台：950；证据等级：tool_observed）

## 失效边界

- opParaSize = 8 + sizeof 为两点经验式（4→12、200→208）；未在多 struct 注册（`REGISTER_TILING_FOR_TILINGKEY`）场景验证 opParaSize 取 max 还是 per-key。
- mocker 计数与 `GET_TILING_DATA_WITH_STRUCT` 调用点数的对应关系验证于单调用点；多分支 kernel（每分支一个调用点）未验证。
- 告警串 "do not registe tiling struct" / mocker 串 / section 命名均以 cann-9.0.0-beta.2 为准；版本升级需复核字面量。
- 其他平台/SoC 未实测（未验证，推断可迁移：机制来自 CANN 宏与构建系统，与 arch 无关）。
- section 观测到 16 字节内容为 sizeof 出现两次（`c8000000…c8000000…`）——宏源码只定义一个 uint64；第二个 8 字节的产生机制未溯源，不影响"首 8 字节 = sizeof"的判读。

## 坏实践

只看构建退出码 0 就认为 TilingData 已编入构建——退出码只证明编译成功，不证明 sizeof 进入二进制元数据、tiling 缓冲按新大小分配（本轮 opParaSize 12→208 的变化退出码本身完全不会体现；若元数据未更新，失败要到 aclnn 联调才会暴露）。

## 误诊

无（本闭环未发生错误归因）。

## 未决事项

1. opParaSize "+8" 头部的确切构成——需从 CANN 源码/文档溯源（优先项）。
2. `REGISTER_TILING_FOR_TILINGKEY` 多 struct 场景下 opParaSize 与 section 的语义（expression 在非预编译展开中被丢弃，运行时如何按 tilingKey 选 struct）——需在分支 kernel 落地 per-tilingKey 注册时验证。
3. mocker 串与 `GET_TILING_DATA_WITH_STRUCT` 调用点在多分支 kernel 下的计数关系（是否每调用点 +1）待验证。
