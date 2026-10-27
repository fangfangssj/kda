---
type: "Runbook"
title: kernel binary .o 定位：build/packages 不存在
description: CPack 日志里的 ./packages/ 路径是 .run 自解包 staging 内部布局；常驻交付 kernel binary 在 build/ascendc_kernels/binary/<soc>/<op>/，安装后在 $ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/<soc>/<小写snake算子名>/<OpType>_<hash>.o——kernel 子目录用小写 snake_case 而非 OpType，按 OpType 拼路径必落空，勿误判安装失败。增量（R3）：验收脚本按两级命名独立断言（目录=核函数名 snake_case、文件=OpType 前缀），file 探针核 ELF（unknown arch 0x1029）。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-09T11:49:41Z"
updated_at: "2026-09-17T04:17:42Z"
platforms: ["950"]
---

# kernel binary .o 定位：build/packages 不存在

## 现象

在对一个按 proto.md 改造为原型一致接口桩的 ACLNN 自定义算子工程（AscendC，ops-registry-invoke 算子仓流程，目标 ascend950）做编译验收时，`cd operators && ./build.sh <OpType>_package` 退出码 0，CPack 打包日志里清楚列出了交付产物的内部路径，例如：

```
Adding files to archive named "custom_opp_ubuntu_x86_64.run"...
./packages/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o
./packages/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so
```

随后要对 kernel binary 逐枚执行 strings 检查，按日志里的路径拼接 `build/packages/...` 去找 .o，直接落空（同一命令里两条 ls 各报一次）：

```
ls build/packages/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/*.o
ls: cannot access 'build/packages/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/*.o': No such file or directory
count=0
```

改用 `find <package>/build -name '*.o'` 全量枚举，返回 28 枚 .o，分布在四类位置——只有其中 3 枚才是最终交付的 kernel binary，其余是 host 侧 CMake 对象、kernel_meta 中间产物和 CPack staging 副本，混在一起很容易拿错。其中交付 kernel binary 的命中（输出 verbatim，路径中工作区前缀以 … 缩写）：

```
…/operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.o
…/operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o
…/operators/ThnnFusedGruCellGrad_package/build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.o
```

安装目录查询（`find $ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad -name '*.o'`，输出 verbatim，用户名已脱敏）：

```
/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_35a6f98eb48c771cd0e8edb80bf279f6.o
/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_bf6eb06d4164f1cf9f51a4b1391902a1.o
/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_f44a71c3e89dd2fadd786cc5ca9ee63a.o
```

R2（CANN 9.0.0-beta.2 / 9.1.0，两个独立会话，OpType 拼路径落空）：bessel_i1e 算子包（OpType 为 `BesselI1e`）构建、安装、上板冒烟 3/3 全部通过后，按验收清单检查 kernel 二进制是否落盘，脚本把 OpType 直接当子目录名：

```bash
ls $ASCEND_OPP_PATH/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/BesselI1e/*.o
# ls: cannot access '.../kernel/ascend950/BesselI1e/*.o': No such file or directory，EXIT_CODE=2
```

而算子明明构建、安装、运行都成功了——实际布局在**小写 snake_case** 子目录：`.../kernel/ascend950/bessel_i1e/BesselI1e_<32位hash>.o` ×3，tiling 配置位于 `op_impl/ai_core/tbe/config/ascend950/bessel_i1e.json`（CPack 清单同样显示 `./packages/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/bessel_i1e/BesselI1e_<hash>.o`）。另一会话的 vendor 部署核验（BenchHarnessSoftmax/BenchHarnessGcd 两算子）同样模式：按 `<OpType>` 目录 ls 报 No such file or directory，find 命中 `.../kernel/ascend950/bench_harness_softmax/BenchHarnessSoftmax_<hash>.o` ×3 与 `.../bench_harness_gcd/BenchHarnessGcd_<hash>.o` ×3。

## 适用场景

- 工程形态：aclnn / ops-registry-invoke 算子仓开发与验收（CMake + op_build + CPack External 自定义算子包流程）；对 kernel binary 做 strings/nm/objdump/数量核对等产物级检查的场景；vendor 部署后的安装核验、TTK kernel 模式排障前的二进制存在性检查。
- API 路径：`operators/build.sh <OpType>_package` → CMake 目标 `ascendc_kernels`/`binary` → CPack External 产出 `custom_opp_ubuntu_x86_64.run` → run 包安装到 `$ASCEND_OPP_PATH/vendors/<OpType>/`。
- 验证过的平台与 CANN 版本：ascend950（注册平台名 950）+ CANN cann-9.0.0-beta.2；R2 另实测 CANN 9.1.0（vendor 部署，BenchHarnessSoftmax/Gcd）与 CANN 9.0.0-beta.2（BesselI1e）。
- 适用粒度（transfer_scope）：platform_wide——凡走 op_build + CPack 的自定义算子仓产物布局一致，与具体算子无关。
- 路线优先级：recommended_fastpath（直接按本文三处 canonical 路径定位，或一次 find 后按路径模式归类；核验时用 find 枚举而非按 OpType 硬拼路径）。
- 明确不适用 / 未验证场景：GEIR/TTK kernel 模式的动态编译产物（在 `$TTK/kernel_meta/dyn_op_case*.o`，另一套路径）；非 vendor 安装（`ASCEND_CUSTOM_OPP_PATH` 自定义路径）布局未验证；CANN 内置算子（非 vendors 自定义包）的产物路径不适用；其他 CANN 版本 / 其他 SoC 的产物布局差异未验证。

## 判别方法

三层定位（按需取用）：

1. **常驻交付 kernel binary**（做完 build 后一直存在，做产物级检查用这批）：
   `<package>/build/ascendc_kernels/binary/<soc>/<op>/<OpType>_<32位hash>.o`，每个 dtype 组合恰好一枚（本例 fp32/fp16/bf16 三组合 → 3 枚）。
2. **CPack staging**（对应 CPack 日志里 `./packages/...` 相对路径的真实落点，位于自解包目录内）：
   `<package>/build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/...`。
3. **run 包安装目录**（执行过 run 包安装后）：
   `$ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/<soc>/<op>/`。

安装目录命名规则（R2 两会话实证）：**vendor 目录用 OpType 原名（大驼峰，如 `BesselI1e`），kernel 二进制子目录用小写 snake_case 形态（如 `bessel_i1e`），.o 文件名是 `<OpType>_<32位hex>.o`**；tiling 配置位于 `op_impl/ai_core/tbe/config/<soc>/<snake>.json`。一个算子可有多个 `.o`（多 tiling/binary 变体，每变体一个 hash 文件）。检查脚本把 OpType 用在 kernel 子目录段就会大小写不符而落空——`ls .../kernel/<soc>/<OpType>/*.o` 报 No such file or directory 不代表安装失败。

用 `find <package>/build -name '*.o'` 全量枚举时，按下表归类判别，避免拿错：

| 类别 | 路径模式 | 本例数量 | 是否交付物 |
|---|---|---|---|
| 交付 kernel binary | `build/ascendc_kernels/binary/<soc>/<op>/<OpType>_<hash>.o` | 3 | 是 |
| CPack staging 副本 | `build/_CPack_Packages/Linux/External/*.run/packages/vendors/...` | 3 | 是（同上 3 枚的打包副本） |
| host 侧 CMake 对象 | `build/CMakeFiles/cust_optiling_obj\|cust_proto_obj\|cust_opapi_obj/...` | 10 | 否 |
| kernel_meta 中间产物 | `build/<OpType>_<soc>/kernel_N/kernel_meta_*/kernel_meta/*.o`（含 `*_infer_channel_dav-c310-cube_0.o`、`*_vec_0.o`、`*_0.o`） | 12 | 否 |

安装目录单独查：`find $ASCEND_OPP_PATH/vendors/<OpType> -name '*.o'`（不要按 `<OpType>` 子目录名硬拼）。

## 设计决策

- 诊断路线选择：先按 CPack 日志相对路径猜 `build/packages`（实测失败），再改用 `find` 全量枚举后按路径模式归类——一次 find 即给出全部落点，无需逐个读 CMake install/CPack 配置推导。备选（读 CMakeLists 的 install 规则推导产物路径）未采用，成本高于一次 find。当前方案属 recommended_fastpath。
- 交付物判别以路径模式（`ascendc_kernels/binary/<soc>/<op>/`）而非文件数量为准：`.o` 总数随 host 目标数、dtype 组合数、kernel_meta 结构变化，只有路径模式稳定。
- R2：核验脚本不直接写死完整路径的原因——hash 后缀每次构建会变，只能用通配；目录段里唯一容易写错的就是大小写形态，判别首选 `find` 枚举而非 `ls` 固定路径。

## 系统信息

- 构建机：x86_64 Ubuntu，GCC 13.3.0，CMake + op_build + CPack External。
- 目标 SoC：ascend950（arch35）；CANN cann-9.0.0-beta.2（`$ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp`）；R2 会话另为 CANN 9.0.0-beta.2（BesselI1e 包）与 CANN 9.1.0（vendor 部署核验）。
- 目录布局：`operators/<OpType>_package/{op_host,op_kernel,op_graph,tests}`；`build.sh` 在 `operators/` 下以包名为参数调用。
- 相关文件（与本次事件流实际读写/枚举一一对应）：
  - `build/ascendc_kernels/binary/ascend950/thnn_fused_gru_cell_grad/ThnnFusedGruCellGrad_<hash>.o` ×3（交付 kernel binary）
  - `build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/...`（staging）
  - `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_impl/ai_core/tbe/kernel/ascend950/thnn_fused_gru_cell_grad/` ×3（安装后）
  - `build/<OpType>_ascend950/kernel_{0,1,2}/kernel_meta_*/kernel_meta/`（中间产物，勿当交付物）
  - R2：`$ASCEND_OPP_PATH/vendors/BesselI1e/op_impl/ai_core/tbe/{kernel,config}/ascend950/bessel_i1e/`、`.../vendors/BenchHarnessSoftmax/.../bench_harness_softmax/`、`.../vendors/BenchHarnessGcd/.../bench_harness_gcd/`

## 完整代码

定位/枚举探针（可执行；`<pkg>` = 算子包根目录，`<OpType>`/`<op>` = PascalCase/snake_case 算子名）：

```bash
# 0) 反例一：按 CPack 日志相对路径拼接（预期失败，该目录常驻不存在）
ls <pkg>/build/packages/vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/<op>/*.o

# 0b) 反例二：安装树按 OpType 当 kernel 子目录（预期失败，实际子目录是小写 snake_case）
ls $ASCEND_OPP_PATH/vendors/<OpType>/op_impl/ai_core/tbe/kernel/ascend950/<OpType>/*.o

## 1) 常驻交付 kernel binary（每 dtype 组合一枚，hash 后缀）
ls <pkg>/build/ascendc_kernels/binary/ascend950/<op>/

## 2) CPack staging（CPack 日志 ./packages/... 的真实落点）
find <pkg>/build/_CPack_Packages -name '*.o'

## 3) run 包安装目录（勿按 OpType 硬拼子目录，直接枚举）
find $ASCEND_OPP_PATH/vendors/<OpType> -name '*.o'
#    确需精确路径时按 <soc>/<小写snake算子名>/<OpType>_<hash>.o 通配拼接

## 4) 全量枚举 + 人工按路径模式归类（交付物/主机对象/kernel_meta 中间产物）
find <pkg>/build -name '*.o' | sort
```

## 预防

- 不要按 CPack 日志的 `./packages/...` 相对路径去 `build/packages/` 下找产物——那是 `.run` 自解包内部布局，实体在 `_CPack_Packages/Linux/External/*.run/packages/` staging 内。
- 对 kernel binary 做 strings/nm/objdump/计数检查前，先 `find build -name '*.o'` 枚举并按 `ascendc_kernels/binary/<soc>/<op>/` 路径模式筛出交付物，避免把 kernel_meta 的 `*_infer_channel_dav-c310-cube_0.o` 或 host 侧 CMake .o 计入交付物口径。
- 验收口径若要求「.o 数量 == dtype 组合数」，计数对象应是第 1 类交付物（或第 3 类安装物），不是 find 的全部结果。
- 写自定义算子验收/巡检脚本时，kernel binary 检查路径用 `<soc>/<OpType 的 snake_case>/<OpType>_<hash>.o` 通配，或干脆用 `find ... -name '*.o'` 枚举后按 `<OpType>_` 前缀断言；不要把 OpType 字符串直接拼进路径中段。
- 从模板生成验收标准时，"kernel binary 存在"这类检查不要把 OpType 字符串直接拼进路径中段。

## 验证信号

以下信号来自一次自定义算子接口桩工程的编译验收：亲自执行 build.sh、按日志路径 ls、find 全量枚举与安装目录查询取得。

- `./build.sh <OpType>_package` 退出码 0（BUILD_EXIT=0），CPack 日志列出 `./packages/vendors/.../<OpType>_<hash>.o` 等包内文件（来源平台：950；证据等级：tool_observed）
- `ls build/packages/vendors/.../*.o` → `No such file or directory`，count=0（来源平台：950；证据等级：tool_observed）
- `find <pkg>/build -name '*.o'` 返回 28 枚：交付 kernel binary 3 枚位于 `build/ascendc_kernels/binary/ascend950/<op>/`；staging 副本 3 枚位于 `build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/...`；host CMake 对象 10 枚；kernel_meta 中间产物 12 枚（来源平台：950；证据等级：tool_observed）
- `find $ASCEND_OPP_PATH/vendors/<OpType> -name '*.o'` 返回与交付物相同的 3 枚已安装 .o（来源平台：950；证据等级：tool_observed）
- 交付 .o 数量 3 == proto dtype 组合数 3（fp32/fp16/bf16，构建日志 `[ascend950] Generating <OpType>_<hash>` 恰 3 次）（来源平台：950；证据等级：tool_observed）

R2（两个独立会话）：

- 触发：`ls $ASCEND_OPP_PATH/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/BesselI1e/*.o` 报 `No such file or directory`，EXIT_CODE=2（来源平台：950；证据等级：tool_observed）。
- 探针观测：`find $ASCEND_OPP_PATH/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ -name "*.o"` 列出 3 个 .o，全部位于 `ascend950/bessel_i1e/` 子目录、文件名 `BesselI1e_<hash>.o`；`ls .../ascend950/bessel_i1e/*.o` 退出码 0（来源平台：950；证据等级：tool_observed）。
- 打包侧佐证：构建日志的 CPack 清单含 `./packages/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/bessel_i1e/BesselI1e_<hash>.o` 与 `./packages/vendors/BesselI1e/op_impl/ai_core/tbe/config/ascend950/bessel_i1e.json`（来源平台：950；证据等级：tool_observed）。
- 跨算子实证：同环境 BenchHarnessSoftmax 与 BenchHarnessGcd 同模式——按 `<OpType>` 目录 ls 报 No such file or directory；find 命中 `.../kernel/ascend950/bench_harness_softmax/BenchHarnessSoftmax_<hash>.o` ×3 与 `.../bench_harness_gcd/BenchHarnessGcd_<hash>.o` ×3（各 3 个 .o 变体）（来源平台：950；证据等级：tool_observed）。
- 二进制确认存在后，同环境 TTK kernel 模式白盒重跑 19/19 PASS（kernel 加载执行正常，佐证二进制有效）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 观察来自单次构建（CANN cann-9.0.0-beta.2、ascend950、CPack External 流程）；其他 CANN 版本或其他打包方式（直接 make install、非 CPack）的布局可能不同——未验证，推断可迁移。
- `kernel_N` 中间目录数量随 dtype 组合数变化（本例 3 组合 → kernel_0/1/2）；`<OpType>_<hash>` 的 hash 由 op_build 按源内容生成，跨算子/跨内容不同，不可当稳定文件名引用。
- `build/packages` 是否会在某些时序（如 CPack 中途、未清理的旧构建）下以实体目录存在，未验证；本文只证实「本次 build 完成后该路径不存在、交付物在 ascendc_kernels/binary 下」。
- R2：子目录名的确切来源（OpType 的 snake_case 还是 kernel 入口函数名）在两会话中不可区分——两者同形；若某算子 kernel 函数名与 OpType snake_case 不同名，须以 find 实测为准（未验证）。
- R2：`<hash>` 后缀按构建内容变化，任何写死完整文件名的检查都会失效；多 `.o` 与 tiling 变体的确切对应关系未验证。

## 坏实践

- 按 CPack 日志相对路径 `./packages/...` 拼出 `build/packages/...` 直接找产物 → `No such file or directory`（本例实测踩坑）。
- `find build -name '*.o'` 后不归类，把 kernel_meta 的 `*_infer_channel_dav-c310-cube_0.o`/`*_0.o` 或 host 侧 CMake .o 当成交付 kernel binary 去做 strings/符号检查 → 计数与结论口径错误。
- 按 OpType 目录名硬拼安装树 kernel 子目录路径，落空后直接断言"安装失败/需要重装"。
- 检查失败就下"产物没装上"的结论而不枚举实际目录——产物齐全，只是目录名形态与直觉不符。

## 误诊

- 本次验收中曾把一次交错执行的两条 `ls`（op_host+op_kernel 一条、op_graph 一条）输出误读为「op_kernel 下含有 legacy_v1_infer_example.cpp / proto.cc 等文件」，后续 `head` 该路径报不存在后重新 `ls -la` 才纠正（这些文件实际在 op_graph）——多目录并列 ls 时输出按目录分段交错，读结果须按目录头归属逐段核对，勿凭印象归目录。
- R2 列为负例警示：若先入为主认为"安装失败"去重装，会浪费一轮构建安装而不改变结果（本例定位一步到位：ls 失败后立即 find 枚举拿到真实路径）。

## 未决事项

- `build/packages` 路径在何种流程/时序下会以实体目录出现未验证（如需绝对排除误用，可补一次 CPack 中途态观察）。
- 其他 CANN 版本 / 其他 SoC（如 a2/a3）下同名产物路径是否一致未验证。
- R2：`<OpType>_<hash>.o` 的 hash 生成规则与多变体选择机制未追（不影响核验用途）；kernel 二进制子目录命名的确切来源（OpType snake_case vs kernel 入口函数名）未区分——优先项：用一个 kernel 函数名与 OpType snake_case 不同名的算子复现验证。

## 后续增量（2026-09-17，并行策展批：R3——验收脚本两级命名独立断言 + file ELF 探针，CANN 9.2.0）

R3（rotary pos emb 类算子产物验收，Ascend950，CANN 9.2.0）：验收脚本按 OpType PascalCase 拼 kernel 目录并统计 `.o` 数量，`ls "$KDIR"/*.o`（KDIR 末段为 `BenchHarnessApplyRotaryPosEmb`）报 `No such file or directory`、计数 N=0、验收判 FAIL——而 build.sh 退出 0、安装日志正常。`ls -d $ASCEND_OPP_PATH/vendors/*/op_impl/ai_core/tbe/kernel/ascend950/*/` 唯一命中 `bench_harness_apply_rotary_pos_emb/`；目录内 3 个 `.o` 齐全，文件名才是 `<OpType>_<hash>`。

两级命名各自独立断言（R3 定形的验收脚本口径）：

- **目录段**：用核函数名 snake_case（从 op_kernel 源文件或 proto 的核函数签名 `extern "C" __global__ __aicore__ void <核函数名>(...)` 取；常规算子核函数名即 OpType 的 snake_case，与 R2 的"小写 snake_case 子目录"同一条规则）。
- **文件段**：`.o` 计数用 `<OpType>_*.o` 通配——两级规则各自独立判定，不要把 OpType 同时用在两段。
- 产物校验探针：`for f in "$KDIR"/*.o; do file "$f"; done`——kernel `.o` 为 `ELF 64-bit LSB executable, *unknown arch 0x1029*, statically linked, not stripped`（arch 0x1029 = Ascend950 kernel 目标），可区分于 host 侧对象；同目录内 `.json`（kernel 元数据）与 `.o` 同名并存。
- 定位用 `ls -d .../tbe/kernel/<soc>/*/` 摸清目录命名再拼路径，优于全树 `find $ASCEND_OPP_PATH -name '<OpType>*.o'`（慢且会命中 `_CPack_Packages` 中间产物，见主卡归类表）。

### 复现增量验证信号（R3）

- 按 OpType 拼目录：`ls .../ascend950/BenchHarnessApplyRotaryPosEmb/*.o` → `No such file or directory`，N=0，验收判 FAIL（来源平台：950；证据等级：tool_observed）。
- 按核函数名拼目录：`ls -d .../ascend950/*/` 唯一命中 `bench_harness_apply_rotary_pos_emb/`；目录内 3 个 `BenchHarnessApplyRotaryPosEmb_<hash>.o` 经 `file` 校验均为 `ELF 64-bit LSB executable, *unknown arch 0x1029*, statically linked, not stripped`（来源平台：950；证据等级：tool_observed）。
- 同目录 `.json`（kernel 元数据）与 `.o` 同名并存，与 build 侧 `build/ascendc_kernels/binary/ascend950/bench_harness_apply_rotary_pos_emb/` 的中间产物布局一致（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 一个算子多个核函数（一个 OpType 多个 entry）时目录数量与命名未验证；其他 soc 目录结构未逐一验证，机制同构但以实际 `ls` 为准。
- R3 会话的目录名与核函数名一致，但该算子核函数名恰为 OpType 的 snake_case——与 R2 相同，"目录名=核函数名"与"目录名=OpType snake_case"在常规算子上同形不可区分，主卡未决事项的区分实验仍待做。
