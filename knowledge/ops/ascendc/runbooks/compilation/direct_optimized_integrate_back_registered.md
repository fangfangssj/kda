---
type: "Runbook"
title: "直调优化实现忠实集成回注册工程配方"
description: "把直调工程性能优化后的实现忠实移植回注册算子工程：kernel.h 整文件移植（去直调 guard）、tiling struct 尾部追加、host tiling 从入参形态翻译回 gert::TilingContext 形态并补显式 abiContig 连续性守卫、K_MAX_SHAPE_DIM 落到 apt.cpp；部署后逐产物 SHA 核验（含 liboptiling）。"
tags: ["task.migration", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# 直调优化实现忠实集成回注册工程配方

## 现象

在 Ascend950PR（CANN 9.2.0）上完成融合算子 FusedRmsNormSigmoidMul 的直调性能优化（终选实现相对原始融合基线 9.16x，见 [vf_lane_accumulate_row_loop_consolidation](../performance/vf_lane_accumulate_row_loop_consolidation.md)）后，需要把最终实现集成回原始注册工程（`FusedRmsNormSigmoidMul_package/`，至今零改动，`op_host/arch35/` 下有 `.orig` 备份文件）并经注册 aclnn 接口验收。两边的工程形态差异是移植的主要风险：直调实现的 host tiling 是 `ComputeTilingImpl` 入参形态（x/gamma/other 的 shape/stride 直接来自 launch ABI），注册工程是 `gert::TilingContext` 形态；直调实现的设备 TU 入口是自定义 entry/launch，注册工程是 apt.cpp；直调实现里有 `__NPU_HOST__` 直调专属 guard。集成时先系统性 diff 两边 kernel 头列出全部差异桶，再按桶移植、构建、部署、验收。

## 适用场景

- 工程形态：注册算子工程（CMake + vendor 包 + .run 安装到 CANN `opp/vendors/`）接收直调工程的优化实现；也适用于任何"直调原型 → 注册工程"的代码回迁。
- API 路径：`gert::TilingContext` 形态的 `TilingFuncPublic`、`platform_ascendc::PlatformAscendC`、注册工程 CMake（`-DCANN_HOME/-DASC_DIR/-DCMAKE_PREFIX_PATH`）、vendor `.run` 安装、TTK aclnn 模式验收。
- 验证过的平台与 CANN 版本：Ascend950PR NPU 0，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（移植方法跨"直调原型→注册工程"回迁通用；映射清单按目标算子的改动面生成）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：直调实现依赖的专属文件（entry/launch/device TU、run_cases.py 等）**不迁移**；注册工程其余文件（proto/def/infershape/CMakeLists）不动；多 CANN 版本共存的 shell 环境需显式钉路径（见失效边界）。

## 解决配方

**1. 差异分桶与映射清单（INTEGRATION_PLAN）**——先系统性 diff 直调终版与注册工程现状，按桶列映射：

| 注册工程文件 | 来源（直调终版） | 移植方式 |
|---|---|---|
| `op_kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h` | `final_optimized/csrc/op_kernel/kernel/arch35/` 同名文件 | 整文件移植，仅去掉直调专属 `__NPU_HOST__` guard（算法/结构逐行保留） |
| `op_kernel/arch35/fused_rms_norm_sigmoid_mul_tiling_struct.h` | 同上 | 尾部追加 `arContigFast` 字段（host/device 共享头，两 TU 同步重编译） |
| `op_kernel/fused_rms_norm_sigmoid_mul_apt.cpp` | —（直调专属宏的等价落点） | 在本 TU 首个 `kernel_operator.h` include 前定义 `K_MAX_SHAPE_DIM 0` |
| `op_host/arch35/fused_rms_norm_sigmoid_mul_tiling_arch35.cpp` | `final_optimized/csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp` | 形态翻译：`SolveFastAu` 逐行照搬；死常量清理照搬；快路门控从 `ComputeTilingImpl` 入参形态翻译回 `gert::TilingContext` 形态；group 路径 `ChooseGroupTiling` 公式保持原始形态 |

**2. host tiling 的形态翻译（核心风险点）**——直调 ABI 下"x/y 恒连续（无描述符输入：行主序 stride + offset 0）"是隐含性质；注册 ABI 下输入可带描述符，必须在门控里**显式等价表达**（stride 与行主序参照相等且 storageOffset==0），否则快路会在非连续输入上误触发产生错误数据。翻译后的门控（verbatim）：

```cpp
  // ---- 连续快路门控与 au 重解（直调 ComputeTilingImpl 等价翻译）----
  // 条件：base 路径成立(group==false) && other 无广播维(所有 otherShape[i]
  // ==xShape[i]) && R 单 chunk(R<=ru) && x/gamma/other 连续(offset 0 行主序
  // —— 注册 ABI 下直调"无描述符输入恒连续"性质的显式等价表达，描述符恰为
  // 行主序时同样成立)。group 决策已用原始 au/pre/post 完成，触发域与基线
  // 一致（仅 base 路径内的 au 重解）。
  bool noBroadcast=true;
  for(int i=0;i<8;++i){
    if(t->otherShape[i]!=t->xShape[i]){noBroadcast=false;break;}
  }
  auto contig=[&](const int64_t* cur,const int64_t* rm,int64_t off){return off==0&&std::equal(cur,cur+8,rm);};
  bool abiContig=contig(t->xStride,xStrideRm,t->xStorageOffset)&&contig(t->gammaStride,gammaStrideRm,t->gammaStorageOffset)&&contig(t->otherStride,otherStrideRm,t->otherStorageOffset);
  int64_t auFast=0;
  if(!group&&noBroadcast&&R<=ru&&abiContig){
    auFast=SolveFastAu(A,ra,static_cast<int64_t>(coreNum),static_cast<int64_t>(ubSize));
  }
  const bool arContigFast=auFast>0;
  if(arContigFast){
    // pre/post 保持原公式形态（仅 au 数值变化）；快路 kernel 不消费
    // preBufSize（InitBuffer 按快路字节清单分配）。
    au=auFast;
    post=PubUp(au*4,blockSize);
    pre=PubUp(au*ra*4,blockSize);
    ao=PubCeil(A,au);
    used=std::min<int64_t>(ao,coreNum);
  }
```

其中行主序参照在 `stride()` 后保存（`xStrideRm` 等三维数组），供 `std::equal(cur, cur+8, rm)` 逐槽比较。`SolveFastAu` 与死常量清理从直调实现逐行照搬。

**3. 构建前身份基线**——改码前对注册工程做 before 快照（源仓副本核对：除 workflow 添加的 golden.py 外与源仓一致），改码后生成 diff 存档（`docs/perf-integrate/diff/{before,after}/*.diff`）；before 快照的正确性用直调基线交叉验证（直调基线的 kernel.h = 注册改前 kernel.h + host guard）。

**4. 构建与部署**——cmake configure 显式钉住目标 CANN（规避 shell 残留的其他版本前缀）：

```bash
cd <REG_DIR> && rm -rf build && mkdir -p build docs/perf-integrate/logs && cd build && \
cmake .. -DCANN_HOME=<目标 CANN 安装路径> \
  -DASC_DIR=<目标 CANN 安装路径>/lib64/cmake \
  -DCMAKE_PREFIX_PATH=<目标 CANN 安装路径>/lib64/cmake && \
make -j8 binary package
```

部署：清理旧 vendor 部署后执行包内 `.run` 安装；**部署核验逐产物 SHA**——不止 kernel `.o` 与 `libcust_opapi.so`，还要定位并核验 `op_impl/ai_core/tbe/op_tiling/liboptiling.so`（tiling 代码已改，其 SHA 必须不同于旧部署；本会话第一次核验时路径找错，重新定位到 `op_tiling/` 下才确认新构建产物已部署）。

**5. 验收**——注册接口（vendor 包）上执行：正向 18 条 L0 用 TTK aclnn 模式（`--plugin golden.py`，从 TTK 目录执行）；负向 17 条 L2 因 TTK 无 expected-error 支持，走等价 aclnn 两段式验证程序（见 [aclnn_negative_two_stage_verify_padded_out](../validations/aclnn_negative_two_stage_verify_padded_out.md)）。

## 设计决策

- 为什么 kernel.h 整文件移植而不是逐段摘取：终版 kernel 头经过全集验收（239/239），整文件移植 + 只去 guard 使"注册版 kernel 逻辑 = 直调终版"成为可 SHA 证明的事实（集成后 kernel.h 与终版差异仅为 guard）；逐段摘取容易漏上下文（局部化变量、事件 ID 声明）。
- 为什么快路门控要补 `abiContig` 而不是照抄直调门控：直调 ABI 的"输入恒连续"在注册 ABI 不成立（描述符输入）；照抄会在非连续输入上走合并 DMA 快路产生错误数据。显式守卫是直调隐含性质的等价翻译，描述符恰为行主序+offset0 时快路仍命中。
- 为什么 K_MAX_SHAPE_DIM 落在 apt.cpp：注册工程设备 TU 的首个 kernel_operator.h include 点在 apt.cpp（直调工程在 local_deps.h）；宏定义必须先于本 TU 首个 include（见 [k_max_shape_dim_zero_before_first_include](k_max_shape_dim_zero_before_first_include.md)）。
- 为什么 group 路径公式保持原始形态：集成契约要求触发域零漂移；group 决策在 au 重解之前用原始 au/pre/post 完成。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR NPU 0，CANN 9.2.0；TTK 源码目录执行（Rev 2474d06）；vendor 部署于 `$ASCEND_OPP_PATH/vendors/FusedRMSNormSigmoidMul/`（关键产物：`op_api/lib/libcust_opapi.so`、`op_impl/ai_core/tbe/op_tiling/liboptiling.so`、`op_impl/ai_core/tbe/kernel/ascend950/<op>/<hash>.o`）。
- 相关文件：`docs/perf-integrate/INTEGRATION_PLAN.md`、`VERIFY_REPORT.md`、`diff/{before,after}/*.diff`、`logs/`（cmake-configure.log、make-package.log、install-vendor.log、ttk-aclnn-l0-full.log、l2-verify.log）；修改面 = 4 个文件，其余零改动。

## 验证信号

以下信号来自该融合算子 Stage 4 集成会话：4 文件按桶移植 → 构建部署 → 注册接口验收，全流程产物落盘 `docs/perf-integrate/`。

- 身份核验：集成后 `op_kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h` 与直调终版同名文件 SHA 对照差异仅为直调 guard；before 快照经直调基线交叉验证一致（来源平台：950；证据等级：tool_observed）。
- 构建/部署：cmake configure + `make -j8 binary package` exit=0（含新 tiling 的 host so 编译通过）；`.run` 安装后逐产物 SHA 核验——kernel `.o`、`libcust_opapi.so` 与新构建一致，`op_tiling/liboptiling.so` SHA 不同于旧部署（来源平台：950；证据等级：tool_observed）。
- 正向验收：TTK aclnn 模式 18/18 L0 全部 PASS（GOLD: PASS / PRECISION_STATUS: PASS，18/18，PassRate 100%）（来源平台：950；证据等级：tool_observed）。
- 负向验收：17 条 L2 中 15 条以预期错误码域拒绝、2 条 shape 负向 NOT_ENFORCED 且经 aclnn 层前后逐字节相同证据确认属集成前既有缺口（来源平台：950；证据等级：tool_observed）。
- 性能语义保持：集成为忠实移植（无算法改动），直调终版相对原始基线 9.16x 的结论随身份核验传递；集成阶段按契约未重复采集性能（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡假设直调终版已通过全集验收（集成是搬运不是修复）；若直调实现本身有未覆盖场景，集成不会发现。
- `abiContig` 守卫按"8 槽 stride 逐槽相等 + storageOffset==0"判定：stride 数组槽位数固定 8，rank>8 的输入未验证。
- 多 CANN 共存环境必须显式钉路径：本会话 shell 残留其他版本前缀，不钉住会 configure 到错误版本（安装目录内 compiler version.info 自报的版本串与安装版本号不一致属组件版本串，勿据此误判环境）。
- TTK aclnn 验收依赖从 TTK 目录执行与 golden 插件路径；换环境需重配（未验证其他布局）。

**未决项**：

- 2 条 shape 负向的接口层校验缺口（NOT_ENFORCED）为集成前既有状态，本卡只记录证据未修复（根因在 autogen stub 生成规则，未追查）。
- 注册 ABI 下描述符输入（非行主序）命中 `abiContig=false` 而走一般路径的性能损耗未量化（该算子实际输入恒行主序，未测）。
## 坏实践

- 部署后只核验 kernel `.o`/`libcust_opapi.so` 不核验 `liboptiling.so`——tiling 改动是否真正上线无从判断（本会话第一次核验就找错了路径；liboptiling 在 `op_impl/ai_core/tbe/op_tiling/` 下）。
- 直调门控照抄进注册工程——隐含连续性前提在注册 ABI 不成立，必须显式守卫。
- 逐段摘取 kernel 改动而不是整文件移植——漏上下文风险高且不可 SHA 证明等价。
- 在残留其他 CANN 版本前缀的 shell 里直接 cmake——configure 结果不可控。
