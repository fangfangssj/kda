---
type: "Runbook"
title: "bf16 极值输入 ε=0 NaN/Inf 语义分歧未解"
description: "白盒数据范围 (-inf,inf) 的 bf16 AdamW 用例（ε=0、β1=0、非对齐尾块）约 12.5% 元素 kernel/golden NaN/Inf 单向失配，多方修复不消；根因未追到 VF 指令层，已验证的处置只有用例侧数据范围改有限值。复现增量：跨算子（RoPE）复现同簇 NaN/Inf mismatch 并给出分诊规则——±dtype_max/±inf 值域套到声明值域 [-1,1] 的辅助输入（cos/sin）上时成簇失配（mere/mare=None），fp16 全过/fp32+bf16 失败的 dtype 签名 + input_data_ranges 对照声明域 + 真实值域 ST 交叉验证可判非 kernel 主路径缺陷。"
tags: [task.debugging, topic.precision, topic.optimizer]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T00:14:22Z'
platforms: ["950"]
---

# bf16 极值输入 ε=0 NaN/Inf 语义分歧未解

## 现象

在 Ascend950PR（CANN 9.2.0）上修 BenchHarnessApplyAdamW 白盒失败时，有一个用例在所有修复落地后仍然失败：bf16、shape (4607,)（= 4608−1 非对齐尾块）、属性 `lr=0.0001, beta1=0.0, beta2=0.5, weight_decay=0.0, epsilon=0.0, step=1.0, maximize=False`、四个输入的 `input_data_range` 均为 `(-inf, inf)`。kernel 正常执行（perf=PASS，编译、运行、显存检查都过），但精度比对失败：

- TTK 默认 stat_rel_err 标准报 `mere=None, mare=None, reason: 'NaN/Inf mismatch'`（一侧 NaN/Inf、另一侧不是，或符号不同）。
- 换 `--compare close`（np.isclose，equal_nan=True）后报 `precision=87.43%`，固定随机种子 42 复跑报 `87.54%`，`reason: 'exceeded atol/rtol'`——即约 12.5%（~580/4607）元素失配，且比例跨种子、跨比对标准稳定。

该失败在以下修复全部落地后依旧存在：OpDef step 类型对齐、β^t 预计算改 double pow、golden 插件补默认属性。验收以两次尝试上限放行，用例留作已知失败。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发 + TTK kernel 白盒评测（含随机输入生成）。
- API 路径：AscendC `__simd_vf__`/`asc_vf_call` 向量计算链（fp32 寄存器域，含 Muls/Add/Mul/Div/Sub/Sqrt/Cast）对照 torch golden；TTK `RandomData` 输入生成（`(-inf, inf)` 范围数值化为 dtype 极值）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——一切输入可能含极值（±dtype max）且计算链含平方/开方/除法的 bf16 逐元素算子（优化器类最典型：v_new=β2·v+(1−β2)·grad² 在 grad=±3.39e38 时平方溢出 fp32 为 Inf）。
- 路线优先级：safety_fallback——已验证的处置只有用例侧规避（数据范围改有限值）；无任何已验证的 kernel 侧修复，不得把本卡当作修复配方使用。fallback 声明：该规避会放弃对极值/非正规输入语义的覆盖（本轨迹实测该覆盖会暴露 kernel 与 golden 的真实语义分歧，规避即放弃检验该边界），仅作未定位前的止损手段。
- 明确不适用 / 未验证场景：受控数据范围的用例——同算子 ST 侧 `epsilon_0_div0`（fp32、(-1,1) 范围、grad=0/v=0/m=1 人工注入 0/0）与 `v_inf_fp32`（v=+Inf 注入）均通过；fp32 dtype 的 (-inf,inf) 用例未单独验证。

## 根因

未追到机制层（如实承认）。已锚定的客观边界与可疑机制链如下：

- 输入侧：TTK `RandomData` 对 `(-inf, inf)` 范围按 `_digitize_inf_nan` 数值化为 dtype 极值——bf16 即 ±3.39e38 量级的有限值（非 Inf 本身）。
- 计算侧可疑链（推理，未做逐元素 dump 对照验证）：`grad²` 在 fp32 域溢出为 +Inf → `v_new`、`v̂`、`sqrt(v̂)` 变为 Inf → `update = m̂/(sqrt(v̂)+ε)` 出现 finite/Inf、0/0 等边界运算；`weight_decay=0` 时 `0×var`、`ε=0` 时除零无稳定裕量。若 AscendC VF 指令（Div/Sqrt/Muls）在 0×Inf、0/0、finite/Inf 等边界上与 torch 的 IEEE 754 语义不一致（一侧 NaN、另一侧 0 或 Inf），即产生单向 NaN/Inf 失配。哪些具体输入组合、哪条指令分歧——未定位。

## 解决配方

无已验证的 kernel 侧修复。已验证的规避（用例侧，属 safety_fallback）：把白盒 CSV 该类用例的 `input_data_range` 从 `(-inf, inf)` 改为有限范围（如 (-10,10)），使中间量不进入 Inf/NaN 边界——同批受控范围用例全部通过即为对照证据。kernel 侧若要真正对齐，须先完成「遗留问题」的逐元素归因，再决定是在 VF 链内做边界预处理（如 ε 下限、饱和钳位——会偏离 golden 语义，不可轻动）还是接受语义分歧并从用例侧排除该输入域。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.2.0；TTK kernel 白盒评测。
- 相关文件：算子包 `BenchHarnessApplyAdamW` 的白盒 CSV 用例与 golden 插件（`tests/blackbox/` 下）。

## 预防

- 对优化器类算子的白盒用例随手写 `(-inf, inf)` 数据范围——TTK 会数值化为 dtype 极值，平方/开方链随即进入溢出边界，把「语义分歧检验」混进「常规精度回归」，阻塞全绿门禁。
- 在未做逐元素归因前，为让该类用例通过而动 golden 语义（改 NaN 为 0 之类）——oracle 语义是验收锚点，不可为过门禁而改。

## 验证信号

以下信号来自该算子白盒修复与验收会话（该用例在两轮修复、多种比对配置下反复复跑）：

- stat_rel_err：`mere=None, mare=None, reason='NaN/Inf mismatch'`，perf=PASS（kernel 正常执行非崩溃）（来源平台：950；证据等级：tool_observed）。
- isclose（`--compare close`）：precision=87.43%；固定种子 42 复跑 precision=87.54%，reason='exceeded atol/rtol'——约 12.5% 元素失配，跨种子比例稳定（来源平台：950；证据等级：tool_observed）。
- 对照组：同算子 ST 侧受控输入的 `epsilon_0_div0`（fp32、(-1,1)、人工注入 0/0 组合）与 `v_inf_fp32`（v=+Inf）用例均通过；白盒其余 33 例（含其它 bf16 用例）在全修复后通过（来源平台：950；证据等级：tool_observed）。
- 修复不消性：OpDef step 类型对齐、β^t double pow、golden 插件默认值补齐全部落地后该用例仍以相同形态失败（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 触发组合（本轨迹实测）：bf16 + `input_data_range=(-inf, inf)` + `epsilon=0` + `beta1=0` + 非对齐尾块 shape；四个条件各自单独出现时（受控范围、ε>0、其它 dtype）未见同类失配。
- 比对标准的 equal_nan 语义只让「双侧同为 NaN」匹配，单向 NaN/Inf 仍判失配——换比对标准不能消除本现象（87.43% 与 'NaN/Inf mismatch' 是同一失配在两种标准下的呈现）。
- 仅在 Ascend950PR + CANN 9.2.0 实测；其它架构 VF 指令的边界语义未验证（未验证，不推断可迁移）。

**未决项**：

- 优先项：复现并追根因到机制层——用 `--dump-on-fail` 落 kernel/golden 双侧 npy，逐元素筛出失配位置，回溯其输入组合（是否 0×Inf、0/0、Inf−Inf），再用最小复现 kernel 对照 AscendC Div/Sqrt/Muls 与 IEEE 754 的边界语义，定位分歧指令。
- fp32 dtype 的 (-inf,inf) 用例是否同样失配未测（本轨迹失配用例仅 bf16）。
- 若确认语义分歧，应评估「算子语义声明极值域为未定义」还是「kernel 对齐 IEEE」——这是设计决策，须回到算子 spec 层。
## 坏实践

- 见上「预防」两条（随手写 (-inf,inf) 范围；为过门禁改 golden NaN 语义）。

## 误诊

- 「疑似根因：bf16 数值敏感边界（ε=0 + β1=0 + 非对齐尾块）」的验收结论只圈定了属性组合，未验证机制——非对齐尾块与失配无因果证据（12.5% 元素均匀分布，非尾块集中），真实边界在极值输入触发的 Inf/NaN 语义（该假设为推理线索，尚未做逐元素 dump 验证）。
- 曾考虑在 golden 插件里把 NaN 归一为 0 使比对通过——该改法会使「kernel 双侧同 NaN」的合法匹配反被判失配，且属改 oracle 语义，未采纳。

## 增量（RoPE 跨算子复现与大值域失败分诊规则）

另一算子（RoPE 类 BenchHarnessApplyRotaryPosEmb tilingKey=6 分支，Ascend950PR_9579 + CANN 9.2.0，fp16/bf16 升 fp32 的 VF MulAdd 融合链对照 torch 逐 op golden）以更大簇复现同一现象并补齐分诊规则：TTK kernel 模式白盒全量 238 例跑出 `total=238 pass=174 fail=64`，其中 **55 条 PRECISION_MISMATCH 的 precision_metrics 均为 `{'standard': 'stat_rel_err', 'mere': None, 'mare': None, 'reason': 'NaN/Inf mismatch'}`**——与本卡 AdamW 实例同签名（比较器发现两侧 NaN/Inf 分歧，连相对误差都无法计算）。

**新锚点 1——值域超出辅助输入的声明域**：抽查失配用例的 `input_data_ranges`：case00005（bf16）四个输入（query/key/cos/sin）全部 `(-3.3895e+38, 3.3895e+38)`（bf16 最大值）、case00006 四输入全部 `(-inf, inf)`——白盒结构用例的无界/asym-large valclass 把 ±dtype_max 同样加到了 cos/sin 上，而该算子 spec 声明的 cos/sin 值域是 **[-1,1]**（预计算位置编码）；对照同批极值用例 C1-C4 的 cos/sin 保持 `(-1,1)` 且全部通过。即"用例值域超算子声明域"不只是"数据范围随手写 (-inf,inf)"，还包括生成器对**全部输入统一套 valclass**把声明域 [-1,1] 的辅助张量也推到极值——判读时先查 `input_data_ranges` 与各输入声明值域的对照，再判 kernel 缺陷。

**新锚点 2——dtype 签名**：失配分布 fp32 43 + bf16 12 + **fp16 0**（全过）。机制（推断，未逐元素验证）：fp16 最大 65504，其任何乘积 ≈4.3e9 远小于 fp32 上限，升 fp32 计算链不可能溢出；fp32/bf16 的 ±dtype_max 输入使乘积 ~1e77 必然溢出——失败集中在"会溢出的 dtype"与值域溢出强相关，同 shape 的 fp16 全过已排除切分/路由/tiling 因素。若失败连 fp16 也挂，则回到切分/路由/tiling 排查，本分诊不适用。

**候选机制（与主卡互补，同样未逐元素验证）**：golden（torch 逐 op 舍入）`y = round(round(x·cos) + round(rotate_half(x)·sin))` vs kernel（VF MulAdd 融合单次舍入）`y = round(x·cos + rotate_half(x)·sin)`——当 `|x·cos|` 超 fp32 上限而和因符号相消落回有限范围时，golden 的 `(+inf)+(-inf)=NaN` 与融合乘加的有限值在 NaN/Inf 上分歧。

**失败构成剔除规则（把"改 kernel"放到最后）**：

1. dtype 分裂（fp16 全过）→ 查 `input_data_ranges`，不先怀疑切分/路由/tiling；
2. 值域含 ±dtype_max/±inf 且作用于声明域外的辅助输入 → 用例值域超算子声明域，归数值边界问题而非主路径缺陷；
3. 错误注入用例（非法 layout/rotaryMode、shape 不匹配、rank0）报 `INVALID_TILING`/`OPTILING_FAILURE` → tiling 按设计拒绝，非缺陷，从精度失败计数剔除；
4. `PROFILE_CRASH` 且错误串 `exited unexpectedly with code -9`（SIGKILL，本例为 536M 元素 fp32 边界用例）→ 宿主资源问题，单独归档；
5. 交叉验证：同 kernel 真实值域 ST 全绿（含单点 NaN/inf/大值特殊用例，cos/sin 保持 [-1,1]）+ UT 全绿 → 进一步排除主路径缺陷。

**预防（生成侧）**：白盒用例生成后、执行前先审 CSV 的 `input_data_ranges`：对声明值域为 [-1,1] 的位置编码类输入，把无界/asym-large/dtype_boundary valclass 的范围收窄到声明域（或在约束 sidecar 里逐输入限定值域），避免"用例值域超算子域"的失败混进 kernel 缺陷清单。

验证信号（复现增量，RoPE tilingKey=6 分支 TTK kernel 白盒全量执行会话）：

- 全量执行：`total=238 pass=174 fail=64 skip=0`；失败构成 PRECISION_MISMATCH 55（fp32 43 + bf16 12 + fp16 0）、INVALID_TILING 7、DYN_COMPILE_FAILURE 1、PROFILE_CRASH 1（来源平台：950；证据等级：tool_observed）。
- 失配用例 metrics（抽查 case00005/00006）：`standard=stat_rel_err、mere=None、mare=None、reason='NaN/Inf mismatch'`（来源平台：950；证据等级：tool_observed）。
- 值域锚点：case00005 四输入均 `(-3.3895e+38, 3.3895e+38)`、case00006 四输入均 `(-inf, inf)`（cos/sin 亦被赋予超域值）；对照极值用例 C1-C4 的 cos/sin 保持 `(-1,1)` 且全部通过（来源平台：950；证据等级：tool_observed）。
- 交叉验证：同 kernel 真实值域 ST branch0-6 全部 12/12（含单点 NaN/inf/fp16 大值特殊用例）、UT 131/131、RUN_EXIT=0（来源平台：950；证据等级：tool_observed）。
- 跨分支系统性：上一分支（interleaved-BSND-cos3d）同流程白盒全量跑计数 `FAIL:100 / PASS:136 / PROFILE_CRASH:1`（来源平台：950；证据等级：tool_observed）。
- 候选机制（MulAdd 单次舍入 vs torch 逐 op 舍入在溢出区的分歧）为推断，无逐元素实测信号（见主卡遗留问题的 dump 方法）。

失效边界（复现增量）：

- 分诊规则依赖"fp16 全过"对照与"golden 为 torch 逐 op 实现 + 比较器为 stat_rel_err"两个前提；golden 或比较器换实现时失败形态可能变化（未验证）。
- 值域超域用例的最终归属（收窄生成器值域 vs 比较器对超域用例的口径）未裁决；在裁决前该类失败按"非 kernel 主路径缺陷"分诊，不得据此放行真实精度问题。
- PROFILE_CRASH 的 -9 崩溃阈值随宿主内存而变，不可迁移。