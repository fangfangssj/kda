---
type: "Runbook"
title: "旧算子包遮蔽工程模板致新 TilingKey 不识别"
description: "自定义算子扩展 ASCENDC_TPL_ARGS_DECL/SEL 模板后新增 dtype 组合编译报 no matching function for call to '..._0_tilingkey' 并伴随 ld.lld cannot open ..._mix_aiv_0.o——CANN 9.1.0 编译系统 get_kernel_source() 优先使用 opp/vendors 下已安装旧算子包的 tiling_key 头，需卸载旧包并分步构建（all → 复制包文件到 opp/vendors → binary → package → --force 安装）。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# 旧算子包遮蔽工程模板致新 TilingKey 不识别

## 现象

在开发 scatter 类自定义算子（ascendc-registry-invoke 算子仓流程，目标芯片 Ascend950PR / DAV_3510 / arch35，`--soc=ascend950`，CANN 9.1.0，x86_64 构建主机）时，把 TilingKey 模板定义从 1 个组合（fp16 data + int32 indices）扩展为 10 个组合（`ASCENDC_TPL_DATATYPE_DECL(D_T_DATA, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, C_DT_INT32, C_DT_INT64, ASCENDC_TPL_INPUT(0))` × `D_T_IDX` 两种），工程源码与 `*_def.cpp` 的 OpDef dtype 列表都已更新，然后执行 `bash build.sh --soc=ascend950 -j8`。

编译出现两个层次的现象：

1. 10 个 kernel 编译任务只有 6 个输出 `Done`（fp16/fp32/int32 data 的组合），bf16 与 int64 data 的 4 个组合全部失败；
2. 失败 kernel 的报错先是链接期假象：

```
ld.lld: error: cannot open build/op_kernel/BenchNoHarnessScatter_ascend950/kernel_4/kernel_meta_.../kernel_meta/BenchNoHarnessScatter_c3173cb0de516efe6277858a4447a037_mix_aiv_0.o: No such file or directory
```

单独运行该 kernel 的编译脚本（`build/op_kernel/BenchNoHarnessScatter_ascend950/bin_param/BenchNoHarnessScatter-bench_no_harness_scatter-4.sh`，内部调用 `asc_opc --main_func=bench_no_harness_scatter --soc_version=Ascend950PR_9599`）才暴露真实错误：

```
kernel.cpp:27:5: error: no matching function for call to 'bench_no_harness_scatter_0_tilingkey'
    bench_no_harness_scatter(data_in__, indices_in__, updates_in__, y_out_, usrWorkspace, tiling);
    ^~~~~~~~~~~~~~~~~~~~~~~~
<command line>:22:34: note: expanded from here
#define bench_no_harness_scatter bench_no_harness_scatter_0_tilingkey
$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/op_impl/ai_core/tbe/bench_no_harness_scatter_custom_impl/dynamic/./bench_no_harness_scatter_arch35.cpp:25:28: note: candidate template ignored: couldn't infer template argument 'D_T_DATA'
```

注意 candidate 的路径是 `$ASCEND_HOME_PATH/opp/vendors/...`（已安装算子包），不是当前工程目录——这是定位根因的第一条线索。

## 适用场景

- 工程形态：ascendc-registry-invoke 算子仓开发的自定义算子，修改了 `op_kernel/arch35/*_tiling_key.h`（`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL`）或 kernel 源码后重新 `bash build.sh --soc=<soc>`。
- API 路径：CANN 9.1.0 `asc_opc` 编译工具链；`get_kernel_source()` 源文件搜索顺序（`PYF_PATH/../ascendc/` → `BUILD_KERNEL_SRC` 环境变量 → `$ASCEND_HOME_PATH/opp/vendors/<pkg>/op_impl/.../dynamic/`）；`ascendc_compile_gen_code.py` 的 `gen_template_tiling_params`、`kernel_info_infer.py` 的 `tpl_tilingkey_native_extract`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，编译目标），CANN 9.1.0，x86_64 构建主机。
- 适用粒度：platform_wide——同一 CANN 安装上任何"已安装旧版算子包 + 工程内更新模板定义"的自定义算子工程。
- 路线优先级：recommended_fastpath（改模板前先卸载旧包是该场景下编译成功的必要条件）。
- 明确不适用 / 未验证场景：从未安装过该算子包的全新环境；CANN 8.x 及更早版本的 `get_kernel_source` 行为（未验证）；host 侧（op_host/op_api）单独编译不受此影响。

## 根因

CANN 9.1.0 编译系统在编译 kernel binary 时，`BenchNoHarnessScatter.py` 里的 `get_kernel_source()` 按"PYF_PATH/../ascendc/ → BUILD_KERNEL_SRC → opp/vendors"的优先级搜索 kernel 源文件，`opp/vendors` 下已安装的旧版算子包排在工程源码（`BUILD_KERNEL_SRC`）之前，于是预处理的是旧包里只有 1 个模板的 `bench_no_harness_scatter_tiling_key.h`。

证据链（对编译系统 Python 脚本临时插桩后取得，插桩已恢复）：

- `gen_template_tiling_params` 收到的 `template_tiling_info` 是空字典 `{}`，导致生成的 wrapper 里 `TEMPLATE_PARAMS` 保持 `-1`，没有任何 `#if TILING_KEY_VAR == ...` 分支；
- `kernel_info_infer.py` 中 `decode_tiling()` 只解出 1 个 TilingKey：`decode(before)={769:['float16','int32']}`；而 `op_info.inputs` 的 dtype 是正确的（`bfloat16/int32/bfloat16`），`tpl_tilingkey_native_extract` 按 dtype 过滤后 `tk_list=[]`——唯一已知的 769 不匹配 bf16，全被滤掉；
- 预处理产物 `declare_param_str` 只含 `{C_DT_FLOAT16, (100000000 +(0))}`、`select_param_str` 只有 1 个选择项——即旧包的 1 模板定义；
- `cat $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/.../arch35/bench_no_harness_scatter_tiling_key.h` 确认安装包内 `ASCENDC_TPL_ARGS_SEL` 只有 1 个条目。

即：新模板组合（如 bf16+int32 的 TilingKey=795，按 D_T_IDX/D_T_DATA 各 8bit 编码拼接）从未进入编译系统的模板映射，wrapper 函数 `bench_no_harness_scatter_0_tilingkey` 无实例可调，cce 编译失败不产 `.o`，ld.lld 随后报 cannot open `.o`（链接期报错是次生现象）。

## 解决配方

修复前状态（问题现场）：`opp/vendors/bench_no_harness_scatter_custom/` 内是旧版安装（1 模板 TilingKey），工程内 `op_kernel/arch35/bench_no_harness_scatter_tiling_key.h` 已是 10 模板。

第一步：卸载旧版算子包（卸载脚本随包安装）：

```bash
bash $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom/scripts/uninstall.sh
# 输出: [ops_custom] ... Successfully deleted bench_no_harness_scatter_custom.
```

卸载后立刻重编会得到新错误 `ModuleNotFoundError: No module named 'impl.dynamic'`（`asc_opc` 的 `get_built_in_op_operator` 找不到 `BenchNoHarnessScatter.py` 等 host 侧 Python 模块）——这说明 kernel 编译还依赖 `opp/vendors` 里已安装的 host 侧产物，不能只卸载不补。

第二步：分步构建，先把新版 host 侧产物补进 `opp/vendors`，再编 kernel：

```bash
# 1) 编译 host 侧（cust_optiling / cust_opapi / cust_op_proto 等）
cd build && cmake -DASCEND_COMPUTE_UNIT=ascend950 .. && cmake --build . --target all -j8

# 2) 从 CPack 暂存目录把算子包结构复制进 opp/vendors（缺 BenchNoHarnessScatter.py 会报 impl.dynamic）
PKG=build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/bench_no_harness_scatter_custom
VENDOR=$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_scatter_custom
mkdir -p "$VENDOR" && cp -r "$PKG"/* "$VENDOR/"
cp build/op_kernel/BenchNoHarnessScatter_ascend950/dynamic/BenchNoHarnessScatter.py \
   "$VENDOR/op_impl/ai_core/tbe/bench_no_harness_scatter_custom_impl/dynamic/"
cp build/_CPack_Packages/.../op_impl/ai_core/tbe/op_tiling/liboptiling.so \
   "$VENDOR/op_impl/ai_core/tbe/op_tiling/"
cp build/autogen/aic-ascend950-ops-info.ini "$VENDOR/op_impl/ai_core/tbe/config/ascend950/"
cp -r build/scripts "$VENDOR/"

# 3) 编译 kernel binary（此时错误应从 _mix_aiv_0.o 变为 _mix_aiv_795.o，说明 TilingKey 已被正确识别）
cmake --build . --target binary -j8

# 4) 打包并安装（安装校验若报 libcust_opapi.so undefined symbol l0op::Contiguous，用 --force 跳过该校验）
cmake --build . --target package -j8
bash build/custom_opp_ubuntu_x86_64.run --force
```

判断修复生效的关键信号：失败对象从 `..._mix_aiv_0.o` 变成 `..._mix_aiv_795.o`（0 是"无模板参数映射"的特征值，795 是 bf16+int32 的真实 TilingKey）。

若 `package` 目标报 `file INSTALL cannot find .../ascendc_kernels/binary/ascend950`，这是"package 依赖 kernel binary、kernel binary 依赖已安装 host 产物"的循环依赖，按上面 2→3→4 的顺序打破。

## 设计决策

- 为什么选择"卸载旧包 + 分步构建"而不是只改工程：`get_kernel_source()` 的搜索顺序由 CANN 安装目录内的脚本决定，工程侧无法覆盖；唯一可控变量是 `opp/vendors` 的内容。
- 备选方案：每次构建前 `--make_clean`——更慢，且 clean 后若 `opp/vendors` 为空会直接撞上 `No module named 'impl.dynamic'`，仍需分步；未采用。
- 长期建议：在 `build.sh` 里增加"编译前卸载旧版算子包"步骤，或至少校验 `opp/vendors` 内 `*_tiling_key.h` 的 `ASCENDC_TPL_ARGS_SEL` 条目数与工程一致。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：x86_64 构建主机（g++ 13.3.0，cmake 3.20.5），CANN 9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根，`ASCEND_OPP_PATH` 指向其 opp），目标 `--soc=ascend950`（Ascend950PR / DAV_3510 / arch35）。
- 相关文件：`op_kernel/arch35/bench_no_harness_scatter_tiling_key.h`（扩展对象）、`build/op_kernel/BenchNoHarnessScatter_ascend950/bin_param/BenchNoHarnessScatter-bench_no_harness_scatter-*.sh`（单 kernel 编译脚本）、CANN 侧 `python/site-packages/asc_op_compile_base/asc_op_compiler/{kernel_info_infer.py,ascendc_compile_gen_code.py,template_tiling.py,ascendc_compile_utils.py}`（插桩点，已恢复）、`opp/vendors/bench_no_harness_scatter_custom/`（旧版安装）。

## 预防

- 修改 TilingKey 模板定义或 kernel 源码后，构建前先检查 `$ASCEND_OPP_PATH/vendors/<pkg>/.../dynamic/arch35/*_tiling_key.h` 是否与工程同版（`grep -c ASCENDC_TPL_ARGS_SEL` 对比条目数）。
- 把"卸载旧包 → all → 补 vendors → binary → package → 安装"固化进 `build.sh`，避免每次手工分步。
- `ld.lld cannot open *.o` 出现在自定义算子 kernel 编译时，先怀疑 cce 编译失败（用单 kernel 编译脚本抓真实错误），不要按链接问题排查。

## 验证信号

以下信号来自该算子模板扩展与重建的同一开发会话：先在旧包在位时复现模板识别失败，再卸载旧包、分步构建确认 10 个 TilingKey 全部编译并安装。

- 修复前：bf16/int64 的 4 个组合 `Done` 缺失，单 kernel 编译报 `no matching function for call to 'bench_no_harness_scatter_0_tilingkey'`，candidate 指向 `opp/vendors/.../bench_no_harness_scatter_arch35.cpp:25:28`（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 插桩观测：`template_tiling_info={}`（空）、`decode(before)={769:['float16','int32']}`、`after native_extract: tk_list=[]`、`declare_param_str` 只含 `C_DT_FLOAT16` 1 个 dtype（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 旧包内容证据：`opp/vendors` 内 `bench_no_harness_scatter_tiling_key.h` 的 `ASCENDC_TPL_ARGS_SEL` 仅 1 条（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 卸载旧包后：失败对象变为 `..._mix_aiv_795.o`（TilingKey 795 被正确生成），且 vendors 为空时出现 `No module named 'impl.dynamic'`（来源平台：950 编译目标/x86_64 主机；证据等级：tool_observed）。
- 分步构建后：10 个 `BenchNoHarnessScatter_<hash> Done` 全部出现，`build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_scatter/` 下 10 个 `.o`；安装后 `opp/vendors` 内 10 个 `.o` + `ASCENDC_TPL_ARGS_SEL` 10 条；ST 回归迭代一 5/5、迭代二 54/54（来源平台：950 实测；证据等级：tool_observed）。

## 失效边界

- 行为仅对 CANN 9.1.0 的 `asc_opc`/`get_kernel_source` 搜索顺序实测；其他 CANN 版本搜索顺序可能不同（未验证，机制推断可迁移：以 `opp/vendors` 实际内容为准）。
- 本卡只覆盖"旧安装包遮蔽工程模板定义"这一根因；同一会话里 bf16 kernel 后续还遇到 arch35 后端不支持 bf16 标量算术（`i64 = setcc`/`bf16 = fmul`/`not support bf16 type cast`）的独立问题，见对应卡。
- 安装阶段 `libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor` 的校验报错在本会话中一律用 `--force` 跳过；该符号问题的根因未追（未验证跳过是否在所有场景安全）。

**未决项**：

- `build.sh` 未固化"卸载旧包 + 分步构建"流程（已写入算子工程 issue，建议向工程维护方同步）。
- `--force` 跳过 `l0op::Contiguous` 符号校验的安全性未评估。
## 坏实践

- 改了 `*_tiling_key.h` 后直接 `bash build.sh`，不检查 `opp/vendors` 里装的是哪个版本。
- 看到 `ld.lld: cannot open *.o` 就去查链接器/磁盘/权限，而不先用 `bin_param/<Op>-<op>-N.sh` 单独跑该 kernel 抓 cce 真实报错。
- 卸载旧包后不补 host 侧产物直接编 kernel，撞上 `impl.dynamic` 后又把旧包原样装回去（会回到模板遮蔽状态）。

## 误诊

- 最初按"bf16/int64 的运算符在 arch35 上不可用"解释 4 个组合的编译失败——方向部分正确（bf16 确实另有后端限制），但这轮失败的直接原因是 wrapper 函数不存在（模板映射为空），与运算符支持无关；`_mix_aiv_0.o` 里的 `0`（而非 795）是区分两个根因的关键信号。
