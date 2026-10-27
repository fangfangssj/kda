---
type: "Runbook"
title: ExtendCfgInfo 无点键致打包 ValueError
description: OpAICoreConfig::ExtendCfgInfo 的键必须形如 key.subkey；无点键（如 "jitCompile"）生成的 INI 行会让 CANN 打包脚本 parse_ini_to_json.py 在 key1.split(".") 处抛 ValueError 打断整个算子包构建。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# ExtendCfgInfo 无点键致打包 ValueError

## 现象

在 Ascend950PR（CANN 9.2.0）上给自定义算子 BenchHarnessScatter 的 `op_host/bench_harness_scatter_def.cpp` 增加 op 配置项：想在 OpDef 的 AICore 配置里加 `jitCompile` 字段，写法是直接把无点的键名传给 `ExtendCfgInfo`：

```cpp
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile", "enable");
```

重新执行算子包构建（cmake + gmake 的 ascendc_kernels 打包流水线），构建在 `Generating ascendc_kernels/tbe/op_info_cfg/ai_core/ascend950/aic-ascend950-ops-info.json` 步骤崩溃，完整报错原文：

```
Traceback (most recent call last):
  File "/workspace/…/Ascend/cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py", line 471, in <module>
    if not parse_ini_to_json(parse_ini_list, OUTPUT_FILE_PATH):
           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/…/Ascend/cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py", line 441, in parse_ini_to_json
    tbe_ops_info = parse_ini_files(ini_file_paths)
  File "/workspace/…/Ascend/cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py", line 152, in parse_ini_files
    parse_ini_to_obj(ini_file, tbe_ops_info)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/…/Ascend/cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py", line 196, in parse_ini_to_obj
    key1_0, key1_1 = key1.split(".")
    ^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 1)
```

随后 `gmake[1]: *** [CMakeFiles/Makefile2:542: CMakeFiles/binary.dir/rule] Error 2`，binary 目标失败、staging 产物（*.run）缺失。当时代码状态：OpDef 其余部分正常（DynamicCompileStaticFlag 等专用链式方法都生成 `key.flag=value` 形式的 INI 行），问题只出在新增的无点键。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：AscendC 自定义算子包构建（CPack/custom_opp 打包，ascendc_kernels_ops_info_gen 生成 ops-info.json）。
- API 路径：`OpAICoreConfig::ExtendCfgInfo(const char *key, const char *value)`（op_def_registry.h）→ `build/autogen/aic-<soc>-ops-info.ini` → CANN `parse_ini_to_json.py`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）；该脚本属 CANN 安装目录，同版本内与算子无关。
- 适用粒度：`platform_wide`——CANN 9.2.0 打包流水线上所有用 ExtendCfgInfo 加自定义配置项的算子。
- 路线优先级：`recommended_fastpath`（正确写法即下述 after）。
- 明确不适用 / 未验证场景：通过 ExtendCfgInfo 设置的键即使打进 ops-info，框架是否读取、值是否合法是另一层问题（见同批 jitCompile 负结果卡——`jitCompile.value`/`jitCompile.flag` 打包成功但均未达到目的）；其他 CANN 版本的解析器行为未验证。

## 根因

CANN 打包脚本 `parse_ini_to_json.py` 的 `parse_ini_to_obj` 对 INI 每一行按 `=` 切出键后，无条件执行 `key1_0, key1_1 = key1.split(".")`，要求键名**恰好含一个点**、拆成两段；`ExtendCfgInfo("jitCompile", "enable")` 生成的 INI 行 `jitCompile=enable` 键名无点，`split` 只返回 1 个元素，解包失败抛 `ValueError: not enough values to unpack (expected 2, got 1)`，整个 ops-info 生成步骤中断。

解析器源码（`cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py`，verbatim）：

```python
            elif "=" in line:
                key1 = line[: line.index("=")]
                key2 = line[line.index("=") + 1 :]
                key1_0, key1_1 = key1.split(".")
                if key1_0 not in op_dict:
                    op_dict[key1_0] = {}
                if key1_1 in op_dict.get(key1_0):
                    raise RuntimeError(
                        "Op:" + op_type + " " + key1_0 + " " + key1_1 + " is repeated!"
                    )
                dic_key = op_dict.get(key1_0)
                dic_key[key1_1] = key2
```

即：INI 键 `a.b=v` 被组装成 JSON 的 `{"a": {"b": "v"}}` 两级结构——这也解释了既有配置为何都是 `opFile.value`、`dynamicCompileStatic.flag`、`precision_reduce.flag` 这类点键（分别来自 `ExtendCfgInfo("opFile.value", ...)` 与专用链式方法）。

## 解决配方

**before**（无点键，打包必崩）：

```cpp
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile", "enable");
```

对应生成的 INI（`build/autogen/aic-ascend950-ops-info.ini` 尾部，`jitCompile=enable` 无点，verbatim）：

```
dynamicCompileStatic.flag=true
dynamicFormat.flag=false
dynamicRankSupport.flag=true
dynamicShapeSupport.flag=true
needCheckSupport.flag=false
precision_reduce.flag=false
opFile.value=bench_harness_scatter_apt
jitCompile=enable
prebuildPattern.value=Opaque
coreType.value=AiCore
opInterface.value=bench_harness_scatter
```

**after**（点键，形如 `key.subkey`；子键按目标读取方命名——框架读 `jitCompile` 时会再查 `jitCompile.flag`，故子键用 flag）：

```cpp
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile.value", "enable");
```

重新构建后 `aic-ascend950-ops-info.json` 生成成功（键入 JSON 为 `{"jitCompile": {"value": "enable"}}`），打包流水线走通。

## 设计决策

- 子键名怎么选：解析器只强制"恰好一个点"，子键语义由读取方决定。对 `jitCompile`，框架日志先查顶层 `jitCompile`、再查 `jitCompile.flag`（`GetStrFromOpContent:Operation <Op> not found in jitCompile.flag of OpContent!`），所以面向框架读取应选 `.flag` 子键（`.value` 打包成功但框架不读）。本会话先用了 `.value`（可打包）再改 `.flag`（可打包且被框架读到，进入值校验）。
- 为什么不直接改生成的 INI/JSON：`build/autogen/` 是构建产物，每次构建重新生成，必须从 `ExtendCfgInfo` 源头改。
- 当前方案路线档位：recommended_fastpath（点键是 ExtendCfgInfo 的唯一可用形态）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0（构建工具链 `cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py`）。
- 相关文件：
  - `op_host/bench_harness_scatter_def.cpp`——ExtendCfgInfo 调用点（修复处）。
  - `build/autogen/aic-ascend950-ops-info.ini`——中间产物，验证键形态。
  - `cann-9.2.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/parse_ini_to_json.py`——解析器（只读，定位根因）。

## 预防

- 使用 `ExtendCfgInfo` 前先看同算子既有键的形态：全部是 `key.subkey`；新增键保持同形态。
- 加键后本地先跑一次完整打包，确认 `Compile op info cfg successfully` 再继续后续验证（打包失败会连带 staging/安装步骤全部空转）。

## 验证信号

来源说明：以下信号来自该算子一次 GEIR 验收会话：为启用预编译二进制匹配尝试给 OpDef 加 jitCompile 配置，触发打包失败后定位到解析器键格式约束。

- before（`ExtendCfgInfo("jitCompile", "enable")`）：ops-info 生成步骤抛 `ValueError: not enough values to unpack (expected 2, got 1)`，`gmake binary` Error 2，无 .run 产物（来源平台：950；证据等级：tool_observed）。
- 生成的 INI 含无点键行 `jitCompile=enable`，其余行均为 `key.subkey=value` 形态（来源平台：950；证据等级：tool_observed）。
- after（改为 `jitCompile.value` / `jitCompile.flag` 点键）：构建输出 `Compile op info cfg successfully` + `[85%] Built target ascendc_kernels_ops_info_gen_ascend950`，打包与安装恢复正常（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只解决"打包崩溃"：点键写入 ops-info 后，框架是否读取该键、接受该值是独立问题——`jitCompile.value` 打包成功但框架不读，`jitCompile.flag` 被读到但值 `enable`/`0` 均被判 invalid（见同批 jitCompile 负结果卡）。
- "恰好一个点"还隐含：多于一个点的键（`a.b.c`）同样会解包失败（`too many values`），未实测但由解析器代码可证。
- 仅在 CANN 9.2.0 的 parse_ini_to_json.py 验证；其他版本解析器行为未验证，推断可迁移。

**未决项**：

- 无（该根因已闭环；ExtendCfgInfo 之上"框架读不读/值合不合法"的问题在同批 jitCompile 负结果卡中记录）。
## 坏实践

- 把无点键传给 `ExtendCfgInfo` 然后只看 C++ 编译结果——编译期不检查键格式，崩溃发生在打包脚本的 Python 侧，且报错栈不在算子源码里，容易被误判为环境问题。

## 误诊

无（本闭环一次定位成功：报错栈直接指向 `key1.split(".")`）。
