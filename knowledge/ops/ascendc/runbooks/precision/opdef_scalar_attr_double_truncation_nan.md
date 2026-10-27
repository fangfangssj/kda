---
type: "Runbook"
title: "OpDef 属性截断 double β 致 NaN 失配"
description: "CANN OpDef 标量属性仅支持 .Float，autogen aclnn 桩用 static_cast<float> 把 double β2 截断，(1-β) 在 β→1 时被近抵消输入放大令 v_new 跨零翻转，NaN 位置与 golden 失配；修复为构建期 int64 位重释传 double + tiling 恢复 + host 预算 + 左结合序。复现增量：白盒 --compare close 口径两轮修复后稳定 29/32（3 例 fp32 近零穿越元素超 atol/ptol、失败集随种子漂移）+ tiling_data 解码实证（biasCorr2 字段 raw 解码一次判定 host 收到 fp32 截断 β2）+ CANN 9.2.0 头文件面无任何 double 属性通道的逐点证据链。"
tags: ["task.debugging", "topic.precision", "topic.optimizer", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# OpDef 属性截断 double β 致 NaN 失配

## 现象

在 Ascend950PR（CANN 9.2.0）上开发一个对齐 `torch.optim.AdamW` 语义的 Stateful 优化器单步算子（fp32/fp16/bf16，full-load/split-load 双分支 kernel，β1/β2/weightDecay 等超参数以 OpDef 标量属性从 aclnn 调用方传入，黑盒 golden 为 torch 参考实现）时，黑盒正向 606 例中 43 例失败：38 个 fp32 用例报 NaN/Inf mismatch（golden 与 kernel 一侧 NaN、一侧有限值），5 个 fp16 用例 mare 超阈。当时的 kernel 已通过全部 ST（41 例）与 TilingUT（37 例），说明问题出在数据面精度而不是结构错误。失败用例的共同特征：二阶矩输入 v 的取值范围是**负值脏数据**（如 v∈[-0.001, 0]），即 v_new = β2·v + (1-β2)·grad² 有相当比例元素落在零附近的穿越带（实测模拟约 12% 元素 v_hat<0）。设计约定（对齐 torch.optim.AdamW）是 v_hat<0 时 sqrt(负数)=NaN、输出对应元素为 NaN、不 clamp——所以 kernel 与 golden 必须在**哪些元素是 NaN** 这一点上逐位一致才算对。修复会话沿"自建 aclnn 诊断 repro 构造临界输入 → 定位属性通道"的路线收敛到 β2 的传递精度，随后在构建期给 autogen 桩打补丁、在 tiling/host/kernel 三层配套修正，黑盒全量转绿。

## 适用场景

- 工程形态：aclnn 自定义算子（OpDef 声明 + autogen 生成 op_api 桩 + tiling/host/kernel 分层），黑盒精度评测（TTK，golden 为 torch/numpy 参考实现）。
- API 路径：OpDef 标量属性（`.Float()`）→ autogen `aclnn_<op>.cpp` 中的属性序列化（`NnopbaseAddAttrWithDtype`）→ `TilingData` → kernel 消费；host `GetFloat/GetInt` 读属性；kernel 内 `1.0f - beta2_` 或 host 预算 `(1-β)` 经 TilingData 下发。
- 验证过的平台与 CANN 版本：Ascend950PR_9579（arch35）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何在 950 上以 OpDef 标量属性传递近 1/近 0 双精度超参数（β、动量、epsilon 等）并要求与 double 域 golden 对齐的算子；机制是属性通道的类型截断 + 近抵消放大，非算子语义。
- 路线优先级：recommended_fastpath（属性通道位重释 + host 预算，是让 double 语义穿过 .Float-only 属性通道的正解）。
- 明确不适用 / 未验证场景：调用方本来就用 float 传参（无截断）；int/string 等非浮点属性；v 全为正（无零穿越带）时不会触发本现象；fp16/bf16 输入（升 fp32 计算后出口回写）的 NaN 传播行为未单独验证；weightDecay 等"非近 1"参数的截断敏感度未单独评估（本例保持 float 传递）。

## 根因

CANN OpDef 的标量属性只支持 `.Float()`（float32）；autogen 生成的 aclnn 桩用 `static_cast<float>(beta2)` 把调用方传入的 double `β2=0.9999` 截断为 float `0.99990004`，kernel 侧由此得到 `(1-β2) = 1.0f - 0.99990004f ≈ 9.996e-5`，而 golden 在 double 域计算 `(1.0 - 0.9999) = 1e-4` 后再舍入。叠加第二个算术差异：kernel 若用 fp32 减法 `1.0f-(float)β2`（β→1 时灾消减：结果量级 1e-4、绝对误差 1.66e-8）或与 golden 不同的乘法结合序 `(1-β2)·(grad·grad)`（golden 左结合 `((1-β2)·grad)·grad`），同量级误差会再次引入。

机制链与算式：

- 绝对差 `|1e-4 - 9.996e-5| ≈ 4.3e-8`，相对差 `4.3e-4`（0.043%）——对普通输入无感，但对 `(1-β)` 本身参与运算的路径是系统性偏差。
- AdamW 的二阶矩更新 `v_new = β2·v + (1-β2)·g²`。当 β2→1、`v` 取负的小量（如 `v ∈ [-0.01, -0.001]`）且 `(1-β2)·g²` 与 `β2·v` 接近抵消时，`v_new` 是两个大数之差得到的近零小量；kernel 与 golden 的 `v_new` 之差为 `(omb2_g - omb2_k)·g² = 4.3e-8·g²`。
- 当 `|v_new| < 4.3e-8·g²` 时，两侧 `v_new` 跨零异号：一侧 `sqrt(v_new)` 得 NaN、另一侧得有限值 → 黑盒按 NaN/Inf mismatch 判 FAIL（该类别不经 mere/mare 统计，直接判失败）。这正是 38 个 fp32 用例集中爆发的形态；随机输入下穿越带元素占比小（~0.06%~0.08%），表现为"99.9% 精度、个别 NaN/Inf mismatch"的批量失败。
- 另 5 个 fp16 用例是近零输出上的相对误差放大（mare 0.0105~0.0856 超出 TTK 社区默认阈 2^-10，但在设计文档预先规定的 rtol=0.1 契约内），与本根因无关，须分别归因、不得混判。

## 解决配方

端到端修复共六步（①②在本卡给出 verbatim，③④⑥的原始编辑在同轨迹更早片段，本卡以修复日志与构建产物验证为据）：

1. **构建期补丁改 autogen 桩**（`operators/<pkg>/CMakeLists.txt`）：在 autogen 生成后、编译前，用 `file(READ)` + 两条 `string(REPLACE)` + `file(WRITE)` 把 β1/β2 的序列化替换为 int64 位重释，绕开 `.Float` 截断；
2. **arch35 tiling glue**：用 `GetInt` 读出 int64 属性再 `memcpy` 回 double；
3. **host tiling**：用恢复出的 double 预计算 `(1-β1)/(1-β2)/biasCorr` 写入 TilingData；
4. **kernel**：消费 host 预算的 `omb1/omb2`（不再用 `1.0f - β_f` 重算），且 `v_new` 采用与 golden 一致的左结合乘法序 `((omb2*grad)*grad)`（近抵消区间内浮点乘法不可结合，结合序差异本身就能改变符号）；
5. golden 插件注册设计阈值容差（fp16 mare 用例属容差契约问题，另行判读）；
6. 触发 kernel 重编译的配套动作（kernel 二进制缓存按 .cpp 内容 hash，仅改头文件不触发重编译）。

修复前（autogen 桩生成代码，`static_cast<float>` 截断 double 属性——CMake 补丁的匹配串，即补丁生效前的真实文件内容）：

```cpp
    float tmp1 = static_cast<float>(beta1);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1), sizeof(float), 1, kNnopbaseFloat));
    float tmp2 = static_cast<float>(beta2);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp2), sizeof(float), 2, kNnopbaseFloat));
```

修复后（构建产物中验证到的位重释形式；`weightDecay` 保持 float，因其值域 ~0.01 不在 β→1 灾难敏感区）：

```cpp
    int64_t tmp1_i = 0; memcpy(&tmp1_i, &beta1, sizeof(double));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1_i), sizeof(int64_t), 1, kNnopbaseInt));
    int64_t tmp2_i = 0; memcpy(&tmp2_i, &beta2, sizeof(double));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp2_i), sizeof(int64_t), 2, kNnopbaseInt));
    float tmp3 = static_cast<float>(weightDecay);
```

承载上述替换的 CMake 补丁块（终版 verbatim；`file(WRITE ... "${_autogen_src}")` 写回行随后，输入参数必须写成 `"${_autogen_src}"`，裸 token 会把源文件写坏）：

```cmake
file(READ ${AUTO_GEN_DIR}/aclnn_bench_harness_apply_adam_w.cpp _autogen_src)
string(REPLACE
    "    float tmp1 = static_cast<float>(beta1);\n    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1), sizeof(float), 1, kNnopbaseFloat));"
    "    int64_t tmp1_i = 0; memcpy(&tmp1_i, &beta1, sizeof(double));\n    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1_i), sizeof(int64_t), 1, kNnopbaseInt));"
    _autogen_src "${_autogen_src}")
string(REPLACE
    "    float tmp2 = static_cast<float>(beta2);\n    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp2), sizeof(float), 2, kNnopbaseFloat));"
    "    int64_t tmp2_i = 0; memcpy(&tmp2_i, &beta2, sizeof(double));\n    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp2_i), sizeof(int64_t), 2, kNnopbaseInt));"
    _autogen_src "${_autogen_src}")
```

### 判别探针（边界扫描）

写一个直连 aclnn 的 C++ 复现程序（borderline 探针），构造 512 个元素：grad 全取 1.0，v 在零穿越点附近线性扫描（穿越点 v* ≈ -(1-β2)/β2·grad²，β2=0.9999、grad=1 时 v*≈-0.0001，扫描带 [-0.00011, -0.00009]）；同时在本地位实现两套 golden（golden 语义：omb=(float)(1.0-beta2_double)、乘法左结合 `((omb·grad)·grad)`；kernel 语义：omb=1.0f-(float)beta2、`(omb·(grad·grad))`），与真实 kernel 输出三方对拍。核心对拍逻辑：

```cpp
    // golden (matching golden.py exactly)
    float b2_f = (float)beta2;
    float omb_g = (float)(1.0 - beta2);  // golden: double sub then cast
    float omb_k = 1.0f - (float)beta2;   // kernel: fp32 sub
    ...
    for (size_t i = 0; i < n; ++i) {
        // golden (torch order): ((1-beta2)*grad)*grad
        float v_new_g = b2_f * hV[i] + (omb_g * hGrad[i]) * hGrad[i];
        ...
        // kernel order: (1-beta2)*(grad*grad), kernel omb
        float v_new_k = b2_f * hV[i] + omb_k * (hGrad[i] * hGrad[i]);
        ...
        bool gNaN = std::isnan(golden[i]), aNaN = std::isnan(actual[i]);
        bool match = (gNaN == aNaN);
        if (!match) mismatch++;
    }
    printf("\nTotal mismatches: %d / %zu\n", mismatch, n);
```

原始观测（β2=0.9999、step=100、lr=0.000337、wd=1.0、eps=0.1、maximize=true）：

```text
omb_kernel=1.0001659393e-04  omb_golden=9.9999997474e-05  diff=-1.6596459318e-08
biasCorr2=9.9506611004e-03
  i    v          v_new_golden   v_new_kernel   v_hat_golden  v_hat_kernel  golden     actual     match?
  ...
255  -1.00019570e-04  -9.567884e-09  7.028575e-09  -9.6153e-07  +7.0634e-07        -nan -4.8508e+00  MISMATCH

Total mismatches: 1 / 512
```

配套判别流程：先用随机输入直连复现确认 Sqrt 的 NaN 行为本身没错（排除"AscendC Sqrt 不产 NaN"假设——Golden NaN=248 / Actual NaN=248，nanMismatch=0），再用边界扫描把 mismatch 锁定到具体元素与具体算术路径；host 预算 (1-β) + 乘法顺序对齐修复实施重编后边界扫描仍 1/512 mismatch（元素 255 不变）——单靠 kernel/host 侧对齐无法闭合，需上游 double β2，即指向属性通道截断。

## 设计决策

- 为什么选"int64 位重释穿过属性通道"而不是其他方案：OpDef 属性类型白名单里没有 double；把 β 拆成两个 float（β 与 1-β 分别传）会改变接口语义且仍受 float 精度限制；把超参数硬编码进 TilingData 则属性来自调用方的事实被破坏。位重释只在"序列化容器"层面借用 int64 的 8 字节，语义在 tiling glue 用 `memcpy` 恢复，是改动面最小且不扩散截断的通道。
- 为什么在 CMake 构建期打补丁而不是手改 autogen 文件：autogen 目录每次构建重新生成，手改会被覆盖；补丁固化在 CMakeLists.txt 中幂等重放。
- 为什么 kernel 改为消费 host 预算值并固定左结合序：golden 在 double 域算 `(1-β)` 且乘法按源码顺序左结合；kernel 若用 `1.0f - β_f` 重算或改结合序，即使 β 传对了也会在近抵消区间复现符号翻转。
- 探针选择边界扫描而非随机大样本：随机输入撞上穿越带的元素数不可控且随种子漂移；构造 v 扫描带 + grad=1 使 v_new 单调可控，能稳定复现并把 mismatch 锁定到具体元素；在本地位同时实现两套 golden 把"golden 语义差"与"kernel 实现差"分离。
- `weightDecay` 保持 float 传递：其典型值 ~0.01 的 float 截断误差不进入近抵消敏感路径（本例未单独评估，属有意取舍）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（Ascend950PR_9579），CANN 9.2.0；npu-smi 25.7.rc1；TTK(ops-test-kit) 3.0.0 黑盒评测（ASCEND_ONBOARD）+ 直连 aclnn C++ 复现（g++ -std=c++17 链接 libascendcl/libnnopbase/libcust_opapi）。
- 目录布局：`operators/BenchHarnessApplyAdamW_package/` 下 `CMakeLists.txt`（补丁）、`build/autogen/aclnn_bench_harness_apply_adam_w.cpp`（被补丁的生成桩）、`op_host/bench_harness_apply_adam_w_tiling_host.cpp`、`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.cpp`、`op_kernel/arch35/bench_harness_apply_adam_w_tiling_struct.h`、`op_kernel/arch35/bench_harness_apply_adam_w_kernel.h`、`op_kernel/bench_harness_apply_adam_w_apt.cpp`、`tests/blackbox/ttk/golden_plugin.py`、`tests/aclnn/`（ST 41 例）、`tests/tiling/`（TilingUT 37 例）。
- 构建入口：`bash operators/build.sh BenchHarnessApplyAdamW_package`。

## 预防

- 在 OpDef 里声明 double 语义的超参数前，先确认属性通道的实际类型（autogen 桩里 grep `static_cast<float>`）：当前 OpDef 标量属性仅支持 `.Float`，double 会被静默截断。
- 凡是"参数 p 近 1、算法里出现 (1-p)"的算子（Adam/AdamW/LAMB 等优化器、EMA、bias correction），把 `(1-p)` 的相对误差当成一级精度风险评估：`|Δ(1-p)|/|1-p| = |Δp|/|1-p|`，β=0.9999 时放大 1e4 倍。
- 含 `(1-β)` 系数的 kernel：`(1-β)` 一律 host 侧用 `(float)(1.0 - beta_double)` 预算再经 TilingData 下发，禁止 kernel 内 `1.0f - beta_f`（fp32 减法在 β→1 时灾消减）；乘法结合序与 golden（torch 左结合）对齐。
- golden 在 double 域计算时，host/kernel 侧任何 float 中转都要对齐舍入路径（先算后舍 vs 先舍后算），近抵消输入上还要对齐乘法结合序。
- 精度用例设计：负值 v 脏数据用例若配 NaN 逐点对齐判据，须意识到穿越带元素是"符号由舍入误差决定"的混沌区；用例生成时可避开 |v_new| < 10·|omb 误差| 的带（或接受该带内失败并单独归类）。

## 验证信号

以下信号来自该算子黑盒失败修复会话及随后的独立验收复算（修复前后黑盒全量回归、自建临界探针、ST/TilingUT 回归；均为上板/工具实测）：

- 修复前：黑盒正向 606 例中 38 个 fp32 用例 NaN/Inf mismatch（失败用例 v 输入范围均为负值带）、5 个 fp16 用例 mare 超阈，共 43 例 FAIL（来源平台：950；证据等级：tool_observed）。
- 随机输入直连复现（2048 元素，v∈[-0.001,0]）：Golden NaN=248 / Actual NaN=248，nanMismatch=0、finiteMismatch=0——排除"AscendC Sqrt 不产 NaN"假设（来源平台：950；证据等级：tool_observed）。
- 边界扫描探针（修复前形态）：omb_kernel=1.0001659393e-04 vs omb_golden=9.9999997474e-05（diff=-1.6596459318e-08）；元素 255 处 v_new_golden=-9.567884e-09（→golden=-nan）vs v_new_kernel=+7.028575e-09（→actual=-4.8508e+00），Total mismatches 1/512；host 预算 (1-β) + 乘法顺序对齐修复实施重编后边界扫描仍 1/512 mismatch——证明需上游 double β2（来源平台：950；证据等级：tool_observed）。
- 修复后：临界抵消构造上 kernel 与 golden `Total mismatches: 0 / 512`（来源平台：950；证据等级：tool_observed）。
- 修复后：NaN 模式 A1（lr=0）Golden/Actual NaN=248/248、Finite=1800/1800，nanMismatch=0、infMismatch=0、finiteMismatch=0；模式 A2（lr=0, var=3.4e38, v∈[-0.01,-0.001]）NaN=1419/1419，A2 nanMismatch=0（来源平台：950；证据等级：tool_observed）。
- 回归：ST 41/41 PASSED（Branch-0 21/21 + Branch-1 20/20）、TilingUT 37 tests PASSED（来源平台：950；证据等级：tool_observed）。
- 黑盒全量：正向 L0/L1 606/606 PASS（0 FAIL、0 SKIP，覆盖此前全部 43 个失败用例）；负向 L2 5 例全部 FAIL/NO_OUTPUT（非法输入全拒绝，负向门禁 PASS）；独立验收会话复算 result.csv/result_l2.csv 结果一致（来源平台：950；证据等级：tool_observed）。
- 构建产物：`-- Opbuild: patched beta1/beta2 to pass double via int64 bit-cast` + 自解压 run 包创建 SUCCESS；构建后 autogen 桩 grep 确认 int64 位重释行存在（来源平台：构建宿主（目标 950 算子包）；证据等级：tool_observed）。

## 失效边界

- 本卡机制针对"OpDef 属性通道类型截断 + 近抵消放大"；若调用方与 OpDef 本来就是 float，无此问题。
- 位重释要求 tiling glue 与 host 侧配套恢复（GetInt+memcpy、double 预算）；只改 autogen 桩而不配套，属性按 int64 语义被消费会得到完全错误的 β。
- 灾难敏感只在 `v_new` 近抵消输入上显现；普通分布输入下 4.3e-4 的相对差可能被常规容差掩盖，"测试过了"不等于通道无截断。
- 触发需要 β 靠近 1（omb 相对误差大）且输入落在穿越带（|v_new| 与 omb 误差同量级）；β2=0.9 等远离 1 的取值 omb 误差极小，不触发。
- `weightDecay` 等非近 1 参数的 float 截断影响未验证（本例有意保留 float）。
- tiling glue/host/kernel 侧配套代码与".apt.cpp 触发重编译"的原始编辑事件位于同轨迹更早片段，本卡以修复日志记载与 autogen 桩/CMake 补丁 verbatim 为据（部分恢复）。
- 机制层面（IEEE-754 截断 + 近抵消放大 + 属性通道类型）不依赖具体算子；其他 CANN 版本的 OpDef 属性类型支持范围未验证，升级版本时须重新确认 autogen 桩行为（未验证，推断可迁移）。
- isclose 判据 equal_nan=True 时"双方都 NaN"通过；stat_rel_err 判据 NaN 逐点对齐——两种判据下 NaN/有限值翻转都失败，迁移判据时需确认。

**未决项**：

- `weightDecay` 保持 float 传递的截断敏感度未单独评估（未尝试）。
- 自建 borderline/debug_repro 探针程序源码未沉淀（会话末已删除临时目录），如需复现需按「判别探针」一节的构造思路重写。
- 升级 CANN 版本后 OpDef 属性类型（是否支持 double）与 autogen 桩模板是否变化，未验证。
- 穿越带宽度与 omb 误差的定量关系（|v_new| < |Δomb|·grad² 即进入混沌区）可进一步给出解析式与用例生成避让规则。
## 坏实践

- 在 OpDef 里按 double 语义声明超参数并默认 autogen 桩会正确传递（当前仅支持 `.Float`，静默截断）。
- kernel 内用 `1.0f - β_f` 重算 `(1-β)` 而不消费 host 预算值；或在与 golden 不同的乘法结合序下比较近抵消结果。
- 只修 fp32 NaN 侧而不追 5 个 fp16 mare 用例的阈值契约来源——两类失败必须分别归因。
- 用 clamp/fmax 把负 v_hat 拉到 0 让 kernel 输出有限值：与设计语义（对齐 torch，不 clamp、v<0 → NaN）冲突，会把 NaN 位置失配变成数值失配，且更难判别。
- 只看 NaN 总数是否匹配就判定通过：NaN 总数一致但个别元素 NaN/有限值互换时，总数对拍不出来的逐位失配才是本现象本体。

## 误诊

- "mare 超阈 = kernel 精度缺陷"：38 个 fp32 NaN/Inf mismatch 是真实的属性通道截断缺陷（本卡），而 5 个 fp16 mare 超的是 TTK 社区默认阈、在设计文档 rtol=0.1 契约内；混为一谈会导致要么误改 kernel、要么误放容差。
- "补丁没生效/构建缓存问题"：构建期曾出现 autogen 源被写成单行 `_autogen_src` 的编译失败（`'_autogen_src' does not name a type`），表象像"补丁未生效"，实际是 CMake 补丁自身的变量解引用错误，与属性截断根因无关。
- 初始假设"Ascend `Reg::Sqrt` 对负输入不产生 NaN"——被随机复现否定（NaN 数量与位置在不穿越时完全一致）。
- 初始假设"(1-β2) 精度差异影响太小不会翻转符号"（均匀随机 v 模拟显示 sign mismatch 0/10000）——该模拟穿越带命中概率低；边界扫描探针才暴露稳定翻转。

## 后续增量（2026-09-17，并行策展批：白盒 --compare close 口径基线 + tiling_data 解码实证 + 头文件面证据链）

同轨迹（BenchHarnessApplyAdamW，Ascend950PR_9579 / CANN 9.2.0）的白盒 TTK 侧（32 例，`--compare close`，CSV rtol=0.05、atol=1e-8、ptol=1e-7）补齐本卡修复前的失败基线与两个判别探针：

**1. 白盒失败基线（修复前形态）**：两轮修复后稳定 29 PASS / 3 FAIL，失败用例全是 float32、近零穿越元素超 `exceeded atol/rtol`：

- 极端超参例（β2=0.9999、step=1，1456×1440）：precision 99.92%；
- 负值 v 输入例（默认 β2=0.999，1000003 元素）：precision 99.94%；
- 大 shape 例（β2=0.999、step=10，2048×2048）：precision 99.99995%（仅 1~2 个元素失败，ptol=1e-7 意味着百万级 shape 下连 1 个失败元素都不允许）。

失败集还在两次全量重跑间漂移（β2=0.9999 的例与大 shape 的例互换），说明是随机输入落在近零穿越带上的概率事件，不是确定性计算错误。偏差校正分母失配比值 1.000166（β2=0.9999）被 `1/(1-β2^step)` 放大最高 10000×；β2=0.999 时分母失配仅 0.0013%，多数用例可通过。

**2. tiling_data 解码实证（判别 host 实际收到什么精度）**：把 TTK result.csv 里失败用例的 `dyn_tiling_data` 原始 int32 序列按 TilingData 布局解码，与 double/fp32 两种场景预测值对拍，一次定性：

```text
  beta2      raw=  1065351538  -> float=0.9998999834060669
  biasCorr2  raw=   953270272  -> float=0.00010001659393310547

IF host got DOUBLE beta2=0.9999: biasCorr2 would be 9.999999747378752e-05
IF host got FLOAT  beta2=0.9998999834060669: biasCorr2 would be 0.00010001659393310547
ACTUAL tiling_data biasCorr2          = 0.00010001659393310547

=> host received FLOAT (truncated) beta2
```

证明截断发生在 host 之前（autogen 属性传递），host 内再用 double 计算已于事无补。曾按"aclnn 传入的 attr 本就是 double"的错误前提派发过一轮修复（TilingData 增加 biasCorr1/biasCorr2 字段、host 用 `(float)(1.0 - pow(beta_d, step))` 预算），TilingUT 37/37 通过但白盒仍 29/32 原样——修复无效的直接证据。

**3. CANN 9.2.0 头文件面证据链（无任何 double 属性通道）**：

- `include/graph/attr_value.h`：`using FLOAT = float`；
- `include/acl/acl_op.h`：`aclopSetAttrFloat(attr, name, float)`，形参就是 float；
- 不存在 `aclopSetAttrDouble` / `GetDouble`；`NnopbaseAttrDtype` 枚举无 Double；
- 全算子包 grep `GetDouble|SetAttrDouble|aclopSetAttrDouble` 零命中。

**4. torch 标量语义实测**：fp32 tensor 除以 Python double 标量时，torch 先把标量 cast 为 fp32 再除（`v_new / f32(1.0-beta2)` 与 `v_new / (1.0-beta2)` 结果逐位相等）——golden 分母即 fp32(double 减法结果)，与 kernel 侧拿到 fp32 截断 β2 后的分母相差 ~1 ULP 级。

**5. 边界补充**：`GetAttrPointer<double>(index)` 按 8 字节解引用 4 字节 float 存储会读到相邻属性内存（垃圾值）；OpDef `OpAttrDef` 只有 Bool/Float/Int/String/List* 系列，不存在 `.Double()`。本卡主文的 int64 位重释修复即针对该通道限制；红线（不改 golden/CSV/容差）内无代码侧修复的结论以本增量证据为准——若 golden 按算子实际能拿到的精度（fp32 β2）对齐或 CSV absolute_precision 调到 DESIGN 的 atol=1e-7，失败可消除，但属"改参考/改容差"，必须由用例与 golden 的 owner 决策。

### 复现增量验证信号

- 白盒全量（`--compare close`，rtol=0.05/atol=1e-8/ptol=1e-7）两轮修复后 29/32 PASS、3 FAIL，失败例 precision 99.92%~99.99995%，`exceeded atol/rtol`（来源平台：950；证据等级：tool_observed）
- 失败用例 `dyn_tiling_data` 解码：biasCorr2=0.00010001659393310547 = fp32 截断场景预测值（double 场景应为 9.999999747378752e-05），证明 host 收到 fp32 β2（来源平台：950；证据等级：tool_observed）
- CANN 9.2.0 头文件面：`ge::AttrValue::FLOAT = float`、`aclopSetAttrFloat(…, float)`、无 `aclopSetAttrDouble`/`GetDouble`、`NnopbaseAttrDtype` 枚举无 Double；全算子包 grep 零命中（来源平台：950；证据等级：tool_observed）
- torch 标量语义实测：fp32 tensor 除以 Python double 标量时 torch 先把标量 cast 为 fp32 再除（两种写法结果逐位相等）（来源平台：950；证据等级：tool_observed）
- host double 预算修复实施后：TilingUT 37/37 PASS，白盒仍 29/32、3 FAIL 原样——修复无效的直接证据（来源平台：950；证据等级：tool_observed）