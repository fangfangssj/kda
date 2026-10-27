---
type: "Runbook"
title: OpDef 改动不生效：ops-info 与 autogen py 不刷新
description: 改 OpDef attr 类型后 make install 报 Up-To-Date 不重编；ops-info.json 仅在 cmake configure 阶段再生，vendor 里的 autogen Python 装饰器不在构建产物链、构建不刷新——源码改了但部署态不变，须强制重建并多层核对部署产物。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# OpDef 改动不生效：ops-info 与 autogen py 不刷新

## 现象

在 `BenchHarnessApplyAdamW`（AscendC 自定义算子包）的修复会话中，把 OpDef 里的 `step` 属性从 `.Float(1.0f)` 改成 `.Int(1)`（`op_host/bench_harness_apply_adam_w_def.cpp`），随后在 `operators/<Pkg>/build` 下执行 `make install` 与 `cmake --build . --target install`，输出全是：

```
-- Up-to-date: .../build/packages/vendors/BenchHarnessApplyAdamW/op_impl/ai_core/tbe//kernel/...
```

没有任何重编迹象。重新跑测试，报错与改前一模一样（仍然期望 float）：

```
RuntimeError: ({'errCode': 'E80003', ..., 'param_name': 'step', 'param_type': "<class 'float'>", 'actual_type': 'int'}, ...)
```

进一步检查发现两层"旧值残留"：构建产物 `build/packages/.../aic-ascend950-ops-info.json` 里 `attr_step` 仍是 `"type": "float"`（源码已是 Int、autogen `custom_opc_options.ini` 已写 `attr_step.type=int`）；而真正决定 TTK 动态编译类型检查行为的 vendor 安装文件 `.../BenchHarnessApplyAdamW_impl/dynamic/bench_harness_apply_adam_w.py` 的 `@para_check.check_op_params(...)` 装饰器仍是 `OPTION_ATTR_FLOAT`。

## 适用场景

- 工程形态：AscendC 自定义算子包的 CMake 构建（`npu_op_package` 宏体系）+ vendor 部署到 `$ASCEND_OPP_PATH/vendors/`；修改 OpDef（attr 类型/默认值等接口事实）后验证行为不变的场景。
- API 路径：`CMakeLists.txt` 的 `execute_process`（configure 阶段编译 `libascend_all_ops.so` 并跑 `op_build` 生成 ops-info）；`build/autogen/`、`build/ascendc_kernels/tbe/op_info_cfg/`、`build/packages/vendors/`、`$ASCEND_OPP_PATH/vendors/<V>/` 四层产物链。
- 验证过的平台与 CANN 版本：Ascend950PR_9579（注册平台 950）+ CANN 9.2.0。
- 适用粒度：platform_wide（构建系统行为，与算子无关）。
- 路线优先级：recommended_fastpath（改 OpDef 后按本配方强制再生并核对三层产物）。
- 明确不适用 / 未验证场景：只改 kernel `.cpp/.h` 不动 OpDef 的构建（正常增量编译即可）；其他构建系统版本未验证。

## 根因

部署产物有**两个不会随常规构建刷新的事实源**：

1. `ops-info.json`（含 attr 类型/默认值）在 **cmake configure 阶段**由 `execute_process`（编译 `libascend_all_ops.so` → 跑 `op_build`）生成；`make install` 只打包已有产物，对 OpDef 源码变化无依赖跟踪，报 Up-To-Date 直接跳过，`build/packages` 里的 ops-info 停留旧值。
2. vendor 安装目录的 autogen Python 实现（`<Vendor>_impl/dynamic/<op>.py`，含 `@para_check.check_op_params` 类型装饰器）**根本不在构建产物链里**——`build/packages` 全树搜不到任何 `.py`，它只在 `$ASCEND_OPP_PATH/vendors/` 里存在（早期某次流程装入），当前构建既不生成也不更新它；而 TTK 动态编译的类型检查恰恰读它。

因此"源码改了 + make install 成功"不等于"部署态变了"。

## 解决配方

以 attr 类型改动为例的完整生效链：

1. **触发重新 configure**（让 execute_process 重跑 op_build）：进 `build/` 删除对应缓存对象（如 `rm -f CMakeFiles/cust_optiling_obj.dir/op_host/*.o` 及相关 depend 文件）后 `cmake --build . --target install`；必要时 `rm -rf packages _CPack_Packages CMakeFiles CMakeCache.txt` 全清重配（全清后注意 kernel 二进制目录结构会变，安装脚本若找 `binary/ascend950/` 而实际产物在 `binary/dynamic/op_kernel/`，需按 CMake install 规则核对）。
2. **核对中间产物**：`build/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend950/aic-ascend950-ops-info.json` 中 `attr_step.type` 应为新值（本例 `int`）；若 `build/packages/.../ops-info.json` 仍旧值，说明打包用了缓存，继续清。
3. **同步 vendor**：`cp -rf build/packages/vendors/<V>/* $ASCEND_OPP_PATH/vendors/<V>/`，再验证安装位 ops-info 的 `attr_step.type`。
4. **手工同步 vendor autogen Python**（构建不会再生）：改 `$ASCEND_OPP_PATH/vendors/<V>/op_impl/ai_core/tbe/<V>_impl/dynamic/<op_snake>.py` 装饰器中对应参数位（attr 在 `_build_args` 形参表里的下标 +6 前移量按文件实际为准，本例 step 是第 11 参）：

   before：
   ```python
   @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
   ```

   after（仅 step 一位 FLOAT→INT）：
   ```python
   @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
   ```

5. 重建 tiling 库并同步（若改了 arch35 胶水）：`rm -f CMakeFiles/cust_optiling_obj.dir/op_host/arch35/*.o` → `cmake --build . --target install` → 拷贝 `liboptiling.so` 到 vendor。

验证生效的三层核对命令：

```bash
python3 -c "import json;d=json.load(open('$ASCEND_OPP_PATH/vendors/<V>/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'));print('step type:', d['<V>']['attr_step']['type'])"
grep -n "OPTION_ATTR" $ASCEND_OPP_PATH/vendors/<V>/op_impl/ai_core/tbe/<V>_impl/dynamic/<op_snake>.py | head -3
```

## 设计决策

- 曾尝试"轻量路径"：`touch` OpDef 源文件再 `make install`——无效（无依赖跟踪）；删除单个 `.o` 强制重编——能重编 tiling/def 对象，但 packages 里 ops-info 仍取缓存；最终采用"删对象重编 + 必要时全清 + 手工拷贝 vendor + 手改 autogen py"的组合。全清重建曾因 kernel 二进制目录结构变化（`binary/ascend950/` vs `binary/dynamic/op_kernel/`）导致 install 找不到目录，故日常优先用"删对象 + 重编 + 手动拷贝"而非全清。
- autogen py 手改是当前构建链的无奈之举（该文件无生成入口），长期应让 `op_build` 一并产出并随 install 部署——当次会话未实现，列为遗留。

## 系统信息

- SOC：Ascend950PR_9579；CANN 9.2.0；构建产物链：`build/autogen/`（.ini）→ `build/ascendc_kernels/tbe/op_info_cfg/`（ops-info）→ `build/packages/vendors/`（打包）→ `$ASCEND_OPP_PATH/vendors/<V>/`（部署）。
- 相关文件：`operators/<Pkg>/CMakeLists.txt`（execute_process/op_build/INSTALL 规则）；`op_host/bench_harness_apply_adam_w_def.cpp`；`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.cpp`；vendor `$ASCEND_OPP_PATH/vendors/BenchHarnessApplyAdamW/`。

## 预防

- 任何 OpDef 级改动（attr 类型、默认值、输入输出签名）后，不要相信 `make install` 的 Up-To-Date；按上面第 5 步的三层核对命令逐一验证部署态。
- 在验证脚本里固化"部署态断言"：跑测试前先断言 vendor ops-info 的 attr 类型与源码期望一致，失败即中止，避免拿旧部署态测试新代码得出错误结论。

## 验证信号

问题与修复来自同一次 BenchHarnessApplyAdamW 修复会话（Ascend950PR_9579，CANN 9.2.0，TTK kernel 模式）：

- 修复前：两次 `make install` 均 Up-To-Date；`build/packages/.../aic-ascend950-ops-info.json` 的 `step type: float`，且改后单测仍报 E80003（期望 float）（来源平台：950；证据等级：tool_observed）。
- 强制重编后：`build/ascendc_kernels/tbe/op_info_cfg/.../ops-info.json` 的 `step type: int`（来源平台：950；证据等级：tool_observed）。
- 拷贝 vendor 后：安装位 ops-info `step type: int`（来源平台：950；证据等级：tool_observed）。
- 手改 vendor autogen py 装饰器后：单测不再报 E80003，报错推进到 tiling 层（`INVALID_TILING`，当时叠加的 vendor 符号冲突问题），证明类型部署链已全部生效（来源平台：950；证据等级：tool_observed）。
- 最终（叠加修复 load_priority 后）：执行状态 `STATUS: PASS`、`DYN_PERF` 从 `INVALID_TILING` 变为有效测量值（执行链生效）；同一输出块另有 `DYN_GOLD: FAIL`、`PRECISION_STATUS: FAIL`——精度比对未通过，属另一未闭环事项，本卡仅主张执行链生效与性能测量有效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡描述的是 CANN 9.2.0 时代该算子包 CMake 模板的行为（execute_process 在 configure 期生成 ops-info、autogen py 不在产物链）；若后续版本构建模板改为 build 期生成或补齐 py 部署，本配方步骤 1/4 即失效（未验证，推断）。
- 全清重建（`rm -rf CMakeFiles` 等）会触发 kernel 二进制目录结构变化，install 规则可能失配——全清是最后手段。
- 本卡只管"改动是否真正部署生效"；改动本身是否正确（类型语义）见配套的 E80003 类型对齐问题。

**未决项**：

- 【优先】让 autogen Python impl 进入构建产物链并随 install 自动部署（当前只能手工同步，极易漏改导致部署态与源码态长期漂移）。
- 全清重建后 kernel 二进制目录从 `binary/ascend950/` 变为 `binary/dynamic/op_kernel/` 的原因（CMake install 规则与 kernel 编译输出的目录约定不一致）未深究。
## 坏实践

- 见"源码已改、make install 绿了"就认定修复无效或有效——两都不成立，必须核对部署态。
- 只盯 ops-info.json 排查（本例它一度显示 NOT FOUND/float 的排查弯路）——para_check 的真正事实源是 autogen py 装饰器。

## 误诊

- 无（本根因链各环节均有工具输出直接佐证）。
