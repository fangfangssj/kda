---
type: "Runbook"
title: aclnn ST 空张量用例接入：0 字节分配与拷贝守卫
description: 为空张量分支（B==0 或 H==0）扩展 aclnn ST 桩——manifest 放行 0 维、输入按 32B 占位分配、0 字节 memcpy 跳过；0 维 aclCreateTensor 描述符与空输出比对短路均实测可用。复现增量：自写 oracle 归约循环以非归约维乘积为边界时空乘积=1 致越界段错误（EXIT=139）的失败侧实证与 N==0 早退修复。 复现增量：Python 比较器侧同族守卫——对空 tensor 做 relative_errors.max() 直接 RuntimeError（numel()==0 需显式 dim），_compare_float/_compare_integer 入口须先 numel()==0 短路；Mock 模式 output=golden 拷贝会掩盖 Real 模式的空 vector 差异，同文件内 float/int 两条比对路径的边界行为必须一致。复现增量：跳过分配+全链路执行变体——size==0 跳过 aclrtMalloc/aclrtMemcpy、nullptr 地址照常 aclCreateTensor，空 Tensor 用例仍真实走 GetWorkspaceSize+aclnn 两段式（验证 host 侧 IsEmpty 快速返回），D2H 拷贝加 outBytes>0 守卫；aclrtFree(nullptr) 返回 ACL_SUCCESS 可无条件清理；ST Real 75/75。 复现增量：空 axes（全 reduce，nAxes=0）形态实测 aclrtMalloc(0) 报 ACL error 100000（补 未查证的确切错误码）；空 axes 的 CPU golden 须展开为全归约（否则 golden=1.0 假错）；第三独立轨迹复现 CreateAclTensor 无守卫 aclrtMalloc(0)——分配与拷贝守卫须成对（memcpy 有守卫而 malloc 没有）。复现增量：第四独立轨迹复现（scatter 算子空 tensor 用例 Real 失败/Mock 通过，全量 875/875 规模）＋ max(size,1) 一字节占位变体；空张量（shape 含 0）与空指针（预期错误码）是两类用例，占位而非 nullptr 以保住用例语义。
tags: [task.testing, topic.implementation.memory_management, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T04:50:06Z'
platforms: ["950"]
---

# aclnn ST 空张量用例接入：0 字节分配与拷贝守卫

## 现象

在为 ThnnFusedGruCellGrad（对标 torch GRU Cell 反向的 NPU 算子，ops-registry-invoke 算子仓开发流程）开发空张量分支的系统测试时，遇到的问题：该算子的空张量分支进入条件是 `B == 0 或 H == 0`（grad_hy 形状 (B, H)），host TilingFunc 有空张量短路（HandleEmptyTensor 路由 tilingKey=2），但已有的 aclnn ST 测试桩是按正维度用例搭的，空张量用例根本进不了测试：

- ST 桩的 manifest 装载函数 LoadManifest 里写着 `if (c.B < 1 || c.H < 1) { err = "... B/H 须为正（空张量属 Branch-2）"; return false; }`——B==0 或 H==0 的用例在装载阶段就被拒绝；
- 输入侧设备内存按 `aclrtMalloc(&devGradHy, numGradHy * es, ...)` 原样分配——空张量用例下 `numGradHy = B * H = 0`、`numStorage = B * 5 * H = 0`，变成对 0 字节的 `aclrtMalloc`；
- 输入上载 `aclrtMemcpy(devGradHy, 0, inGradHy.data(), 0, ACL_MEMCPY_HOST_TO_DEVICE)`——空 `std::vector` 的 `data()` 可能为 nullptr，0 字节拷贝没有必要冒险。

与之对照，输出侧早在正维度无 bias 用例（bias 输出为 `(0,)`）时代就有了空张量口径：空输出按 32B 占位分配、poison 与读回对 0 字节输出跳过、比对对 0 元素短路通过。本卡记录把这套口径补齐到输入侧与 manifest 校验、使空张量用例全链路跑通真实 aclnn 调用链的完整配方。

## 适用场景（必填）

- **工程形态**：aclnn（ops-registry-invoke 算子仓开发流程；ST 走 `aclnnXxxGetWorkspaceSize → aclnnXxx` 两段式调用链，自研 manifest 驱动 C++ ST 桩）。
- **API 路径**：`aclrtMalloc` / `aclrtMemcpy` / `aclCreateTensor`（含 0 维描述符）/ aclnn 两段式调用链；ST 桩为 manifest(JSON) 驱动的自研 C++ 程序。
- **验证过的平台与 CANN 版本**：Ascend 950PR（arch35，运行时 coreNum=64、ub=253952 字节）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度**：`operator_family`——空张量退化的 aclnn ST 桩适配模式对带空输入合法退化的算子族同构。
- **路线优先级**：`recommended_fastpath`（接入空张量用例的最短正确路径；未见更优替代）。
- **明确不适用 / 未验证场景**：Kernel 直调（TTK）用例未在本卡验证；`aclrtMalloc` 以 0 字节单独调用是否必然报错未做 A/B 实测（32B 占位是防御性接入，见设计决策）；非 950PR 平台 / 其他 CANN 版本上 0 维 `aclCreateTensor` 的接受性未验证。

## 根因

ST 桩的输入链路与 manifest 校验假定 B、H 恒正——校验用 `< 1` 拒绝 0 维、输入侧对 0 字节 `aclrtMalloc` / `aclrtMemcpy` 未做占位与守卫，空张量用例无法进入测试。

## 解决配方

四处适配（before/after 完整代码见「完整代码」，均为事件流内逐字恢复）：

1. **manifest 校验放行 0 维**：`c.B < 1 || c.H < 1` 改为 `c.B < 0 || c.H < 0`（错误信息同步改为"须非负"）——B==0 / H==0 是算子接口合法退化，不是非法输入。
2. **输入 32B 占位分配**：`aclrtMalloc(&devGradHy, std::max(inGradHyBytes, static_cast<size_t>(32)), ...)`——与输出侧既有占位口径同款；占位不改变描述符 shape（仍是 `(0, 8)` / `(4, 0)` 这类含 0 维形状）。
3. **0 字节 memcpy 跳过**：输入上载改为 `(inGradHyBytes > 0 && aclrtMemcpy(...))` 短路守卫——两输入都为 0 字节时整条上载被跳过、`ok` 保持 true；非空输入行为不变。
4. **多套件 manifest 与退出码口径**：`main()` 增加 Branch-2 的 manifest/数据目录实参（argv[5..6]），退出码对红基线分支只要求 `stTotal2 > 0`（用例已编入）而不要求全过——测试先行阶段 kernel 未实现时回归基线不被破坏。

oracle 侧（gen_st_oracle.py / golden.py）**零改动**：`torch.empty(0, H)` / `torch.empty(B, 0)` 合成、`uniform_`、`inject`（numel==0 恒不注入）、空 batch 的 `sum(dim=0)`、空张量 `tobytes()`（0 字节文件）全部天然兼容，实测 12 个空张量用例的期望值生成全部正确。

## 设计决策

- **为何 32B 占位而不是照传 0 字节**：`aclrtMalloc` 传 0 的行为是实现定义（可能返回 null 或报错）；输出侧在正维度无 bias 用例上已验证过 32B 占位口径，输入侧直接沿用同一口径，占位字节不参与任何计算与比对。本开发会话 未做"不占位"的 A/B 对照——守卫版全链路实测通过（诚实标注：占位的**必要性**是静态推理，**充分性**是实测）。
- **为何 memcpy 用短路守卫而不是依赖 0 字节拷贝的良定义性**：空 `std::vector::data()` 可返回 nullptr，`aclrtMemcpy(dst, 0, nullptr, 0, ...)` 无必要冒险；守卫后 0 字节用例的 `ok` 语义保持"上载成功"。
- **退出码为何不要求红基线分支全过**：测试先行流程下新分支 kernel 为桩、数值用例预期失败；退出码锚定"旧分支零回归 + 新分支用例已编入"（`stTotal2 > 0`），避免每次测试先行阶段 CI 假红。
- **被放弃的方案**：在 kernel 桩里临时写零让用例全过——放弃，会污染红基线信号（见同域[全空输出用例不构成红基线卡](empty_output_cases_cannot_form_red_baseline.md)）；为空张量单独写一套精简 ST 桩——放弃，与既有 manifest 驱动机制重复且丢失同链路可比性。

## 系统信息

- 工具链/环境：Ascend 950PR（arch35，coreNum=64、ub=253952）、CANN 9.0.0-beta.2、GCC 13.3.0、cmake 构建测试二进制。
- 调用链：`aclnnThnnFusedGruCellGradGetWorkspaceSize → aclnnThnnFusedGruCellGrad` 两段式；vendor 包经 build.sh 安装至 `${ASCEND_OPP_PATH}/vendors/ThnnFusedGruCellGrad/`。
- 相关文件（与事件流实际读/写一一对应，路径前缀 算子仓仓库根（ops-registry-invoke 工程，以下为仓库内相对路径））：
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/st_cases_branch2.json`（新建：12 条空张量用例 manifest）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（改造：LoadManifest 校验、RunStCase 输入分配/上载、main() 第三套件与退出码）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/run.sh`（改造：MANIFEST_B2、第三个 oracle 数据生成调用、test_aclnn 传参 argv[5..6]）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/gen_st_oracle.py`（读，零改动）
  - `docs/ThnnFusedGruCellGrad/develop/golden.py`（读，零改动——独立参考 oracle）
  - `operators/ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_empty.h`（读——empty 桩，Process 全核早退）
  - `operators/ThnnFusedGruCellGrad_package/op_host/arch35/`（读——host tiling 的空张量短路 HandleEmptyTensor：EMPTY_A 时 SetBlockDim(1) 严禁 0、usedCoreNum=0；EMPTY_R 时 4 约束切分）
  - `docs/ThnnFusedGruCellGrad/design/branches/DESIGN-BRANCH-2.md`、`docs/ThnnFusedGruCellGrad/design/Validation.md`（读——分支定义与用例矩阵）

## 完整代码

以下代码逐字取自事件流（改造前状态来自测试桩文件的完整读取；改造前后差异来自编辑事件的完整 old/new 串）。

### 1. LoadManifest 维度校验（before → after）

```cpp
// before（正维度口径，空张量用例被拒绝）：
        if (c.B < 1 || c.H < 1) {
            err = "manifest case " + c.caseId + " B/H 须为正（空张量属 Branch-2）";
            return false;
        }

// after（放行 B==0 / H==0）：
        if (c.B < 0 || c.H < 0) {
            err = "manifest case " + c.caseId + " B/H 须非负（B==0 或 H==0 属 Branch-2 empty）";
            return false;
        }
```

### 2. RunStCase 设备内存分配段（before → after）

```cpp
// before（输入按元素数原样分配，空用例即 0 字节 aclrtMalloc）：
    // 3) 设备内存（空输出 (0,) 按 32B 占位分配，描述符 shape 仍为 (0,)）
    void* devGradHy = nullptr;
    void* devStorage = nullptr;
    void* devOut[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    const size_t outBytes[5] = {
        static_cast<size_t>(expectNum[0]) * es, static_cast<size_t>(expectNum[1]) * es,
        static_cast<size_t>(expectNum[2]) * es, static_cast<size_t>(expectNum[3]) * es,
        static_cast<size_t>(expectNum[4]) * es};
    if (aclrtMalloc(&devGradHy, static_cast<size_t>(numGradHy) * es, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&devStorage, static_cast<size_t>(numStorage) * es, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {

// after（输入同输出口径：32B 占位分配；字节量提为具名常量供上载守卫复用）：
    // 3) 设备内存（空输出 (0,) 按 32B 占位分配，描述符 shape 仍为 (0,)；
    //    空输入（B==0 或 H==0）同理按 32B 占位分配，0 字节 memcpy 跳过）
    void* devGradHy = nullptr;
    void* devStorage = nullptr;
    void* devOut[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    const size_t outBytes[5] = {
        static_cast<size_t>(expectNum[0]) * es, static_cast<size_t>(expectNum[1]) * es,
        static_cast<size_t>(expectNum[2]) * es, static_cast<size_t>(expectNum[3]) * es,
        static_cast<size_t>(expectNum[4]) * es};
    const size_t inGradHyBytes = static_cast<size_t>(numGradHy) * es;
    const size_t inStorageBytes = static_cast<size_t>(numStorage) * es;
    if (aclrtMalloc(&devGradHy, std::max(inGradHyBytes, static_cast<size_t>(32)), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&devStorage, std::max(inStorageBytes, static_cast<size_t>(32)), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
```

### 3. RunStCase 输入上载段（before → after）

```cpp
// before（无 0 字节守卫；空 vector 的 data() 可为 nullptr）：
        // 4) 输入上载（device 与 golden 同一份位流——两端同源）
        if (aclrtMemcpy(devGradHy, static_cast<size_t>(numGradHy) * es, inGradHy.data(),
                        static_cast<size_t>(numGradHy) * es, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(devStorage, static_cast<size_t>(numStorage) * es, inStorage.data(),
                        static_cast<size_t>(numStorage) * es, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {

// after（短路守卫：0 字节输入整条跳过，ok 语义不变）：
        // 4) 输入上载（device 与 golden 同一份位流——两端同源）；
        //    空输入（0 字节）跳过 memcpy
        if ((inGradHyBytes > 0 && aclrtMemcpy(devGradHy, inGradHyBytes, inGradHy.data(),
                        inGradHyBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) ||
            (inStorageBytes > 0 && aclrtMemcpy(devStorage, inStorageBytes, inStorage.data(),
                        inStorageBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)) {
```

### 4. 输出侧既有空张量口径（本卡未改，空张量通过链的其余环节，逐字）

```cpp
// poison 阶段：0 字节输出跳过（红基线反巧合机制对空输出天然让位）
            // 5.1 poison 输出（红基线反巧合：接口桩不写输出时保持非期望值）
            for (int k = 0; k < 5; ++k) {
                if (outBytes[k] == 0) {
                    continue;
                }
                std::vector<uint8_t> poison(outBytes[k]);
                FillPoison(poison, c.dtype);
                if (aclrtMemcpy(devOut[k], outBytes[k], poison.data(), outBytes[k],
                                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                    res.frameworkError = true;
                    res.message = "aclrtMemcpy poison 失败";
                    ok = false;
                    break;
                }
            }

// 读回阶段：0 字节输出跳过，本地缓冲 resize(0)
// 5.5 读回五路输出
            if (ok) {
                for (int k = 0; k < 5; ++k) {
                    runOut[run][k].resize(outBytes[k]);
                    if (outBytes[k] > 0 &&
                        aclrtMemcpy(runOut[run][k].data(), outBytes[k], devOut[k], outBytes[k],
                                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        res.frameworkError = true;
                        res.message = std::string("aclrtMemcpy 读回失败: ") + kStOutputNames[k];
                        ok = false;
                        break;
                    }
                }
            }

// 比对：0 元素短路通过（TTK _check_empty 同语义），结果置 empty 标记
    const size_t n = actual.size() / es;
    if (n == 0) {
        r.empty = true;  // (0,) 空张量：比对短路通过（TTK _check_empty 同语义）
        return r;
    }
```

### 5. main() 第三套件与退出码（after，逐字）

```cpp
    const char* manifestPath2 = (argc > 5) ? argv[5] : "../st_cases_branch2.json";
    const char* dataDir2 = (argc > 6) ? argv[6] : "../st_data_branch2";
    int stTotal2 = 0;
    int stPassed2 = 0;
    int stFailed2 = 0;
    int stErrors2 = 0;
    if (!RunStSuite("Branch-2", manifestPath2, dataDir2, stTotal2, stPassed2, stFailed2, stErrors2)) {
        stErrors2++;  // manifest/流级失败按 ERROR 计（exit 非 0）
        stTotal2 = (stTotal2 > 0) ? stTotal2 : 0;
    }
    // ……（summary 打印略）
    // 退出码：smoke + Branch-0/1 ST 全过 = 0；Branch-2 红基线阶段（本分支数值
    // 用例 EMPTY_R 部分失败、EMPTY_A 空张量短路通过属桩即最终语义，Branch-0/1
    // 与最小通路用例保持通过）退出码 0 属预期。
    return (smokeFailed == 0 && stTotal0 > 0 && stFailed0 == 0 && stErrors0 == 0 &&
            stTotal1 > 0 && stFailed1 == 0 && stErrors1 == 0 &&
            stTotal2 > 0) ? 0 : -1;
```

### 6. run.sh 数据生成与传参（after，逐字）

```bash
MANIFEST_B2="${SCRIPT_DIR}/st_cases_branch2.json"
# ……（每套件一次全量重生成，禁用「已存在则跳过」缓存；TORCH_DEVICE_BACKEND_AUTOLOAD=0 关 torch_npu autoload）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 "${SCRIPT_DIR}/gen_st_oracle.py" \
    --manifest "${MANIFEST_B2}" --out "${SCRIPT_DIR}/build/st_data_branch2" --golden "${GOLDEN}"

export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/ThnnFusedGruCellGrad/op_api/lib/:${LD_LIBRARY_PATH}
./test_aclnn "${MANIFEST_B0}" "${SCRIPT_DIR}/build/st_data" \
             "${MANIFEST_B1}" "${SCRIPT_DIR}/build/st_data_branch1" \
             "${MANIFEST_B2}" "${SCRIPT_DIR}/build/st_data_branch2"
```

## 预防

为支持空张量分支的算子搭 aclnn ST 桩时，接入空张量用例前逐项检查：

1. manifest 装载校验是否用 `< 1` 拒绝了 0 维（应 `< 0` 放行合法退化）；
2. 输入侧 `aclrtMalloc` 是否对 0 字节做了占位（对齐输出侧 `(bytes > 0) ? bytes : 32` 口径）；
3. 输入上载 `aclrtMemcpy` 是否有 `bytes > 0` 短路守卫；
4. 期望值落盘/读取对 0 字节文件是否兼容（`fseek/ftell` 得 sz=0 → `resize(0)` 直接返回 true）；
5. poison/读回/比对三处对 0 字节输出是否已跳过/短路；
6. 退出码是否会把"红基线分支的预期失败"放大成全局假红。

## 验证信号

来源：ThnnFusedGruCellGrad 空张量分支（Branch-2，tilingKey=2）aclnn ST 测试先行开发会话——空张量用例接入后的全量实机运行（vendor 包安装 + cmake 构建 + oracle 数据生成 + ST 全量执行，真实设备）。

- 测试二进制编译通过（cmake/make `Built target test_aclnn`）（来源平台：950；证据等级：tool_observed）。
- 独立 oracle 对 12 个空张量用例的期望值生成全部正确：B=0 用例输出 shape 形如 `[[0, 24], [0, 24], [0, 8], [24], [24]]`（有 bias，bias 为 24 元素）或 `[[0, 24], [0, 24], [0, 8], [0], [0]]`（无 bias）；H=0 用例 `[[4, 0], [4, 0], [4, 0], [0], [0]]`；空张量数据文件实测 0 字节、EMPTY_R 的两路 bias 期望文件各 96 字节且内容全零（来源平台：950；证据等级：tool_observed）。
- 12 个空张量用例全部经真实 aclnn 两段式调用链执行、零 frameworkError——即含 0 维的 `aclCreateTensor` 描述符被 `GetWorkspaceSize` 接受、空张量路由 tilingKey=2 成功、32B 占位分配与 0 字节 memcpy 跳过全程无错（来源平台：950；证据等级：tool_observed）。
- 全量回归不退化：smoke 1/1、Branch-0 20/20、Branch-1 11/11 全 PASS；run.sh 退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本配方验证于 Ascend 950PR / CANN 9.0.0-beta.2 / arch35；其他平台（950DT 等核数不同型号、a2/a3/310p）与 CANN 版本上 0 维 `aclCreateTensor` 与空张量 tiling 路由的行为未验证，推断可迁移（空张量合法退化是 aclnn 层通用语义）。
- `aclrtMalloc` 0 字节占位的**必要性**未做 A/B 实测（无失败侧证据）；若某环境实测 `aclrtMalloc(0)` 良定义，可去掉占位但无收益。
- 空张量**输出侧**需要 kernel 写入的场景（如空 batch 有 bias 的零填充）不在本卡范围——那是"空 batch 归约"卡的主题；本卡只覆盖"调用链跑通 + 空输出比对短路"。
- 非连续布局/广播输入不在支持域（本算子 spec ND/contiguous required），本卡未涉及。

## 坏实践

- 直接把空张量用例塞进按正维度搭的 ST 桩而不做输入侧守卫——0 字节 `aclrtMalloc` / `aclrtMemcpy` 行为实现定义，可能在某些 CANN 版本上报框架错误，把语义正确的用例误判为 ERROR。
- 用 manifest 校验（`< 1`）替算子接口做合法性裁决——B==0 / H==0 是算子接口层的合法退化（host TilingFunc 有专门短路路径），ST 桩无权拒绝。

## 误诊

- 会话早期曾怀疑"aclnn 框架层可能会拒绝空张量、所有用例会变成 frameworkError"——实测证伪：`GetWorkspaceSize` 接受含 0 维的描述符并正常路由空张量分支，12 用例零 frameworkError（该假设当时标注为待实测推断，实测后修正）。

## 后续增量（2026-09-16，并行策展批：aclrtMalloc 0 字节失败侧实证、64B 占位变体）

逐元素最大值算子（y=max(x1,x2)，声明支持空 Tensor）迭代三联调（Ascend950PR/arch35 + CANN 9.1.0，Real 模式 NPU 执行 + CPU golden 比对）补上原卡缺失的**失败侧证据**，并给出更简脚手架形态的同类守卫：

**1. 失败侧实证（更新原卡失效边界第 2 条）**：原卡标注"`aclrtMalloc` 0 字节占位的必要性未做 A/B 实测（无失败侧证据）"。本次实测：`CreateAclTensor` 用 `byteSize = GetShapeSize(shape) * sizeof(T)` 直接调 `aclrtMalloc`，空张量 byteSize=0 时 `aclrtMalloc(deviceAddr, 0, ACL_MEM_MALLOC_HUGE_FIRST)` 返回非 ACL_SUCCESS，用例报 `创建 x1 tensor 失败` 判 FAIL——72 用例中仅 2 条空 Tensor 用例（L1-057 双空 float32、L1-058 x2空+广播 fp16）失败。**同用例 Mock 模式（CPU golden，无设备分配环节）通过、UT empty_tensor tiling 用例通过**——失败点在测试脚手架的设备分配环节，与算子的空张量语义无关。必要性在 CANN 9.1.0 上已从"静态推理"升级为"实测失败"。

**2. 守卫变体（64B 最小对齐占位 + 跳过 0 字节 memcpy + 错误码日志）**：

```cpp
// after：空张量分配 64B 最小对齐 buffer，跳过 0 字节 memcpy（尾部 aclCreateTensor 不变）
    size_t byteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    // aclrtMalloc 不支持 0 字节分配，空张量（含 0 维度）分配最小对齐 buffer（64B）。
    // 算子对空张量不实际读写数据，仅需有效 device 地址。
    size_t allocSize = (byteSize == 0) ? 64 : byteSize;
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtMalloc 失败, ret=%d, byteSize=%zu", static_cast<int>(ret), byteSize);
        return ret;
    }

    if (byteSize > 0) {
        ret = aclrtMemcpy(*deviceAddr, allocSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(*deviceAddr);
            return ret;
        }
    }
```

设计要点：64B 与设备分配对齐粒度一致（原卡用 32B 亦可，两者均为实测有效取值，非官方最小值）；`aclrtMemcpy` 的 destMax 用 `allocSize`、count 用 `byteSize`（容量 vs 拷贝量语义分离）；失败分支打印错误码与 byteSize，下次分配失败可一眼看到是否 0 字节。

**3. 验证信号（来源：该算子迭代三 ST Real 全量运行，修复后同一测试二进制重编译重跑并保存完整日志）**：

- 修复前：Real 首轮 70/72，L1-057/L1-058 报「创建 x1 tensor 失败」FAIL；同用例 Mock 模式通过（来源平台：950；证据等级：tool_observed）。
- 修复后：Real 复跑 `L1-057 → [PASS] float32 空张量（0 元素）`、`L1-058 → [PASS] fp16 空张量（0 元素）`，全量 `总计: 72 通过: 72 失败: 0`，`ST EXIT_CODE=0`，日志汇总 72 个 PASS、0 FAIL（来源平台：950；证据等级：tool_observed）。
- 后续测试工程师验收轮复跑 Real 72/72 可复现（来源平台：950；证据等级：tool_observed）。

**4. 边界补充**：`aclrtMalloc` 对 0 字节请求的确切错误码与官方最小分配约束未查证（对复现非必需，64B 方案已稳定）；把空 Tensor 用例从 Real 测试集剔除/skip 以"通过"属测试侧放松（掩盖脚手架缺陷且丢失空张量上板验证，本轨迹未采用）。

## 后续增量（2026-09-17，并行策展批：比对器 null 检查须放行 0 元素；跳过分配+早返回变体）

两条独立轨迹补齐空张量用例接入的另外两个守卫点：

**1. 比对器 null 检查在 size==0 之后短路（逐元素最大值算子轨迹，同 复现增量算子的下一轮修复）**：分配侧修好后，比对器仍把"空张量（size==0）"与"非法空指针"混为一谈——空 `std::vector` 的 `data()` 合法地返回 nullptr，而 null 检查先于 size==0 判定执行，0 元素比对被误判为非法输入（`[ERROR] null pointer input`，当轮 70/72）。机制链：`GetShapeSize([0,256])=0` → `GenTestDataByPattern` 返回空 vector → 空向量 `data()==nullptr`（C++ 标准允许）→ null 检查命中 → 误判 FAIL，而 size==0 根本无需解引用。

```cpp
// before：null 检查先于 size==0，空 vector 的 data()==nullptr 被误判
bool CompareResults(DType dt, const double* golden, const double* actual, size_t size) {
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }
    if (GetDTypeInfo(dt).isFloat) {
        return CompareResultsFloat(dt, golden, actual, size);
    }
    return CompareResultsInt(dt, golden, actual, size);
}

// after：size==0 优先短路（空 tensor 视为通过，空 vector 的 data() 可能为 nullptr，但 size==0 时无需访问）
bool CompareResults(DType dt, const double* golden, const double* actual, size_t size) {
    // 空 tensor 优先判定（空 vector 的 data() 可能为 nullptr，但 size==0 时无需访问）
    if (size == 0) {
        DTypeInfo info = GetDTypeInfo(dt);
        LOG_PRINT("  [PASS] %s 空张量（0 元素）", info.name);
        return true;
    }
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }
    if (GetDTypeInfo(dt).isFloat) {
        return CompareResultsFloat(dt, golden, actual, size);
    }
    return CompareResultsInt(dt, golden, actual, size);
}
```

要点：短路放在比对器函数最前而不是调用方跳过比对（空张量语义属比对器职责，调用方逐处判空会漏）；保留 null 检查在 size>0 路径——非空张量的 nullptr 仍是真非法输入，不能因修空张量而放松非法检测。该比对器随后在 Real 模式（NPU 执行 + CPU golden 比对）全量 72/72 中复用，空张量用例在 NPU 上通过。

**2. 跳过分配 + host 侧早返回变体（GCD 广播算子轨迹，整数 int16/int32/int64）**：与 32B/64B 占位路线相对，另一条已验证守卫是**完全不分配**——`CreateAclTensor` 在 size==0 时跳过 `aclrtMalloc/aclrtMemcpy`、deviceAddr 置 nullptr，仍用原始 shape 创建 `aclTensor` 描述（保留 0 维信息）；清理函数按指针判空不调 `aclrtFree`。`RunCaseReal` 计算 ny 后，ny==0 直接记 PASS 返回——由广播规则（任一输入 0 元素 ⇒ 输出 0 元素），该早返回同时覆盖全部空输入组合：

```cpp
// CreateAclTensor：0-size 跳过分配，deviceAddr 兜底 nullptr（verbatim 关键段）
    size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    *deviceAddr = nullptr;
    // 空 tensor：aclrtMalloc(0) 会失败，跳过分配，仅创建 tensor 描述
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret; }
    }
    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
                              *deviceAddr == nullptr ? reinterpret_cast<void*>(0) : *deviceAddr);

// RunCaseReal：ny==0 早返回（插在创建任何张量之前）
    // 空 tensor：跳过 NPU 执行，直接判定 PASS（空 tensor 的 GCD 结果也是空）
    if (ny == 0) {
        LOG_PRINT("  [PASS] 空 tensor (ny=0, 跳过 NPU 执行)");
        return true;
    }
```

**两条路线的取舍**：占位分配路线（原卡 32B / 先前增量 64B）让空张量用例走完整 aclnn 调用链（0 维描述符被 GetWorkspaceSize 接受、空张量 tiling 路由被真实执行）；跳过分配+早返回路线完全不触碰 NPU 执行路径，更简但**kernel 的 totalLength=0/空分支未被真实执行**——需要覆盖 kernel 侧空张量行为时应采用占位路线，只关心"调用链不挂、用例不误报"时跳过路线即可。

**3. 验证信号**：

- 比对器短路修复（最大值算子轨迹）：修复后 Mock 全量 `总计: 72 通过: 72 失败: 0`，其中 `[Mock] L1-057: 空Tensor float32 双空 → [PASS] float32 空张量（0 元素）`、`L1-058 → PASS`；同轮自测"测试16: 空张量比对 (float32/int64)"、"测试24: 空 Tensor 广播"均 PASS；同一比对器随后在 Real 模式全量 72/72 中复用（来源平台：950；证据等级：tool_observed）。
- 跳过分配+早返回（GCD 算子轨迹）：L0 回归 24/24 通过，其中 3 条空 tensor 用例输出 `[PASS] 空 tensor (ny=0, 跳过 NPU 执行)`；L1 回归 503/503 通过，含空 tensor 用例（来源平台：950；证据等级：tool_observed）。空张量用例首次失败的原始报错文本未保留在该轨迹记录中（发生于会话前段，失败事实按修复叙述记录，不作为实测信号；失败侧实测证据以 复现增量为准）。

**4. 坏实践补充**：在调用方为空张量用例单独跳过比对（掩盖算子侧空张量行为是否被验证）；把 null 检查删除以"放过"空张量（放开真正的非法输入检测，属测试侧放松）。




## 后续增量（2026-09-17，并行策展批：1 字节保底变体 + oracle 空输入早退 + CMP_EMPTY 比对档位，CANN 9.2.0）

softmax 类算子 BenchHarnessSoftmax 的空 tensor 分支（任一维 size=0，tilingKey=2 Empty）aclnn ST（8 个 shape × fp16/fp32/bf16 共 24 条用例）补齐第三种守卫组合，并给出与原卡/上节不同的两个新守卫面：

**1. 保底字节数的第三取值（1 字节）**：空 tensor 的 `bytes==0` 时 `allocBytes = bytes ? bytes : 1`——保底分配 1 字节使设备指针非空，tensor descriptor 仍携带真实（空）shape，保底字节永不被读写。与原卡 32B、上节 64B 并列，三者均为实测有效取值（非官方最小值），语义等价。

```cpp
    // ── device memory ──
    // Allocate at least 1 byte so device pointers are non-null even for empty
    // tensors (numel == 0, branch-2).  The tensor descriptor still carries the
    // real (empty) shape, so this backing byte is never read or written.
    size_t allocBytes = bytes ? bytes : 1;
    void* devX = nullptr;
    void* devOut = nullptr;
    CHECK_ACL(aclrtMalloc(&devX, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST));
```

**2. oracle 空输入早退（新守卫）**：参考实现在进入任何循环/rank 判定前先对空输入早退，`y` 保持空 vector——镜像 `torch.nn.functional.softmax` 对空输入返回同 shape 空 tensor 的行为，同时防住 EMPTY_R（reduceSize==0、outer>0）形态下归约循环的越界访问（早退为预防性守卫，本开发会话 无越界报错样本）：

```cpp
    int rank = static_cast<int>(shape.size());
    y.assign(x.size(), 0.0f);

    // Empty tensor (any axis size == 0): softmax returns a same-shape empty
    // tensor, mirroring torch.nn.functional.softmax on empty input.  This guard
    // also prevents OOB access when reduceSize == 0 but outer > 0 (the
    // EMPTY_R sub-case of branch-2, where the reduce axis itself is 0).
    if (x.empty()) {
        return;
    }
```

**3. CMP_EMPTY 显式比对档位（新守卫）**：为"0 元素比较集"定义显式枚举档位，与"逐位精确"区分语义；空用例的 pass = aclnn 框架 OK + 输出 numel==0（由 shape descriptor 保证），元素循环因 n==0 不执行：

```cpp
enum CmpMode {
    CMP_RELATIVE,      // normal: rtol/atol per dtype
    CMP_STRICT_EXACT,  // scalar/reduce=1/all-zero: y == g exactly (atol=0)
    CMP_NAN_AWARE,     // NaN/+inf/-inf slices: isnan(y)==isnan(g), non-NaN relative
    CMP_MIXED_INF,     // -inf+finite: -inf pos strict 0, finite relative
    CMP_EMPTY          // empty tensor: 0-element set, no comparison
};
```

**4. 用例矩阵设计**：同 shape 不同 dim 区分 EMPTY_A/EMPTY_R（`[0,4] dim=-1` 是 EMPTY_A，`[0,4] dim=0` 是 EMPTY_R），`[0,0]` 覆盖多 0 维优先级，高 rank 带 0 维覆盖合轴前判定；每 shape × 3 dtypes。

### 复现增量验证信号

- 测试可编译：tests/aclnn 目录 `cmake + make` 退出码 0；算子包 `./build.sh` 退出码 0（来源平台：host 构建；证据等级：tool_observed）。
- 首次全量 `run.sh`：Branch-2（Empty，tilingKey=2）24/24 precision PASSED、24 aclnn-OK、0 aclnn-FAIL（5 EMPTY_A shape + 3 EMPTY_R shape × fp16/fp32/bf16，逐条 `[PASS]` 输出）；Branch-0 46/46、Branch-1 37/37 回归不退化；EXIT=0（来源平台：950；证据等级：tool_observed）。
- oracle 独立性：静态核对 ref_softmax 不引用被测 aclnn 接口/ kernel 产物（无 import/调用命中）（来源平台：host 静态检查；证据等级：tool_observed）。

### 复现增量边界

- 0 字节 `aclrtMalloc` 的确切失败形态本开发会话 未实测（1 字节为预防性守卫）；oracle 越界触发路径亦未实测（早退为预防性守卫）——两守卫的必要性证据见原卡与上节（0 字节分配失败侧实证）。
- 多输出、带 workspace、inplace 语义的算子：输出/中间 buffer 的保底分配须逐个补，本增量只示范输入+单输出。
- 仅 950 + CANN 9.2.0 实测；其他平台未验证，推断可迁移。

## 后续增量（2026-09-17，并行策展批：oracle 归约循环越界的失败侧实证——空乘积 1 机制）

另一条 softmax 基准算子轨迹（opbuild 自定义算子包，Ascend 950 / CANN 9.1.0）补上 复现增量边界中缺失的 **oracle 越界触发路径实测**：为空 tensor 分支新增 13 个 `Kind::Empty` 用例（`[0,4]`、`[0]`、`[0,0,0]`、`[4,0,8] dim=1`、`[0,2097152]` × fp16/fp32/bf16）后，设备侧已做 `allocBytes=max(bytes,1)` 防护，首次跑 `run.sh` 仍在 host 侧测试进程内段错误：

```
./run.sh: line 32: 1032775 Segmentation fault      (core dumped) ./test_aclnn
EXIT_CODE=139
```

**机制（空乘积陷阱）**：CPU golden oracle 把归约循环边界取为非归约维乘积（outerCount × innerSize），而非元素总数 N。以崩溃用例 `[0]` dim=-1 推导：rank=1、normDim=0、R=0、N=0，而 outerCount 与 innerSize 都是**空乘积 = 1**——循环体执行 1 次并读 `x[0]`，x 是空 vector，越界段错误。对照组解释失败面：`[0,4]` dim=-1 时 outerCount=shape[0]=0，循环一次都不执行（B3 三例因此 PASS，纯前导 0 维被 outerCount=0 掩盖）；`[4,0,8]` dim=1 时 outerCount=4、innerSize=8、R=0，循环执行 32 次读 `x[0]`——同样会崩（排在 `[0]` 之后未及执行）。**判别规律：只有"归约轴之前/之后还存在非 0 维"的空 shape 才触发。**

**修复**（与复现增量的 `if (x.empty()) return;` 同型，早退放 `y.assign` 之后、归约循环之前——空输入的期望输出就是空 tensor，`y.assign(0, 0.0f)` 已产生空结果，早退只须绕过循环；用 N==0 而非 R==0 作入口判据，可同时覆盖"0 不在归约轴"的空 tensor）：

```cpp
    int64_t N = static_cast<int64_t>(x.size());
    y.assign(N, 0.0f);

    // 空 tensor（任一维==0 → N==0）：返回空 tensor（returns_empty）。
    // 必须在归约循环之前返回——当 R=0 但 outerCount/innerSize > 0 时（如 [4,0,8] dim=1），
    // 循环会执行并访问 x[base]，但 x 为空 → 越界崩溃。
    if (N == 0) return;
```

**定位技巧**：段错误后用行缓冲直跑二进制（`stdbuf -oL -eL ./test_aclnn`）拿到崩溃前的最后一条 PASS，把崩溃定位到具体用例，再区分 host 进程崩溃（oracle/harness）与 device 侧异常（kernel）。

### 复现增量验证信号

- 崩溃复现：`run.sh` → Segmentation fault EXIT_CODE=139；行缓冲直跑：Branch-0 32/32 + Branch-1 17/17 + Branch-2 前 3 例（`[0,4]` 三 dtype）PASS 后于 `E0D1_fp16 [0]` 崩溃（来源平台：950；证据等级：tool_observed）。
- 修复后：`run.sh` EXIT_CODE=0，Branch-2 13/13 全部 `PASS (returns_empty ok)`（含 `[0]`×3、`[0,0,0]`×3、`[4,0,8]`×3、`[0,2097152]`）（来源平台：950；证据等级：tool_observed）。
- 回归：Branch-0 32/32 + Branch-1 17/17 不退化（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 结论绑定"自写 CPU oracle 以非归约维乘积做循环边界"的实现形态；若 oracle 用连续扁平下标 + N 边界（`for i < N`）则空输入天然不越界。
- `aclrtMalloc(0)` 防护是防御性的（该轨迹崩溃源是 oracle，malloc(0) 崩溃证据见 复现增量）。
- 初判怀疑方向（aclrtMalloc 1 字节分配与含 0 维 aclCreateTensor 的组合、或 aclnn/TilingFunc 对 rank-1 空 shape 的处理）均被行缓冲输出排除——设备侧调用均已返回，崩溃在 host 侧 oracle 循环。

## 后续增量（2026-09-17，并行策展批：storageDims=nullptr 致 161001 陷阱 + L2 侧 totalElems==0 早退 + 1 元素保底变体，TensorList 场景）

另一独立轨迹并入 TensorList 算子（Ascend950PR，CANN 9.1.0，Real 模式 65 用例含 0 元素边界：shape `[0]`、`[2,0,3]`、FP16 `[0]`）的实例，补上 0 元素链路的第四个守卫点：**aclCreateTensor 的 storageDims 形参**。

**1. storageDims=nullptr 陷阱（新守卫面，区别于分配侧/比对器侧）**：测试侧给 0 元素张量兜底分配最小内存后（本实例为最小分配 1 个元素，与 32B/64B/1B 并列的又一实测有效取值），第二轮失败形态变为 L2 接口参数错（161001 = ACLNN_ERR_PARAM_INVALID）——`aclCreateTensor(dims, dimNum, dataType, strides, offset, format, storageDims, storageDimNum, deviceAddr)` 的 `storageDims` 在 `logicalSize == 0` 时被按 `logicalSize > 0` 分支传了 `nullptr`，构造出的张量描述符不完整，GetWorkspaceSize 路径按参数无效拒绝。**storageDims 必须始终给出（0 元素也应传 shape）**：

```cpp
    // 修复前：0 元素时 storageDims 传 nullptr → GetWorkspaceSize 161001
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND,
                              (logicalSize > 0) ? shape.data() : nullptr,
                              shape.size(), *deviceAddr);

    // 修复后：始终传递 shape 作为 storageDims (含 0 元素)
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND,
                              shape.data(),  // 始终传递 shape 作为 storageDims (含 0 元素)
                              shape.size(), *deviceAddr);
```

配套的最小分配（本实例取 1 个元素，与本卡先前增量 的 64B、先前增量 的 1 字节语义等价）：

```cpp
    // 空 Tensor (0 元素): 分配最小 1 个元素, 避免 aclrtMalloc(0) 失败
    size_t allocSize = static_cast<size_t>(physicalSize > 0 ? physicalSize : 1) * elemSize;
```

**2. 算子 L2 侧 totalElems==0 早退（op 侧守卫，与本卡先前增量 的测试侧 ny==0 早返回相对）**：全空 TensorList（所有张量元素总数为 0）时 `GetWorkspaceSize` 直接返回、不启动 Kernel——这是写在**算子 op_api**（而非测试脚手架）里的空输入分支，与本卡先前增量 测试侧早返回的取舍一致：早退时不触碰 launcher 对 0 元素 tensor 的行为（未验证），空输入没有可计算内容：

```cpp
    auto ret = CheckParams(x1, x2, x3, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // ========================================================================
    // 空 Tensor 快速返回 (0 元素, DESIGN.md §5.3 边界处理)
    // 当所有张量元素总数为 0 时, 无需启动 Kernel, 直接返回
    // ========================================================================
    uint64_t listLen = x1->Size();
    int64_t totalElems = 0;
    for (uint64_t i = 0; i < listLen; i++) {
        totalElems += (*x1)[i]->GetViewShape().GetShapeSize();
    }
    if (totalElems == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
```

早退以 x1 的元素总数为准（本算子约束 x1/x2/x3 同构）；"部分张量 0 元素、部分非 0"的混合场景按正常路径走，由 tiling/kernel 的既有 0 元素处理覆盖。含 0 元素语义的算子，L2 GetWorkspaceSize 一开始就写全空早退，不要等 161001 再补。

### 复现增量验证信号

- 第一轮（直接创建 0 元素 aclTensor，无最小分配）：3 条 0 元素用例（FP32 [0]、FP32 [2,0,3]、FP16 [0]）全部 `创建 x1[0] tensor 失败`（来源平台：950；证据等级：tool_observed）。
- 第二轮（min-alloc 后，storageDims 仍传 nullptr）：`GetWorkspaceSize 失败: 161001`（来源平台：950；证据等级：tool_observed）。
- 修复 storageDims（恒传 shape）后：161001 消失，第四轮 64/65、第五轮 65/65 = 100%（含全部 0 元素边界用例）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- storageDims 结论绑定 `aclCreateTensor` 的 C 接口契约（CANN 9.1.0 实测）；161001 在本实例表现为"描述符不完整被参数校验拒绝"，与复现增量的"分配侧 161001 失败侧实证"是相邻但不同的失败点（分配失败 vs 描述符被拒）。
- 测试侧最小分配取值（1 元素）与 32B/64B/1B 均为实测有效、非官方最小值；非连续张量的 physicalSize 计算需按 strides 展开（本实例 `nonContiguous && logicalSize > 0` 分支）。
- L2 早退绑定"全空 TensorList"语义；逐 tensor 部分为空的算子需按各自 spec 设计分支。

## 后续增量（2026-09-17，并行策展批：Python 比对器空 tensor 短路 + Mock 掩盖 Real 的边界）

StridedSlice 类算子（x_shape=[0] 空输出边界用例，Ascend950PR / CANN 9.1.0）的两套 ST 工程补齐空张量比对链路的 Python 侧守卫与两个判读边界：

**1. Python 比对器的空 tensor 崩溃（与复现增量的 C++ 侧 null 检查同族）**：PyTorch ST 的空切片用例在比对器处直接崩溃（而非误判）：

```text
File ".../tests/st/torch/compare.py", line 62, in _compare_float
    mare = relative_errors.max().item()
RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0.
```

修复：`_compare_float` 入口先短路（`_compare_integer` 在 `torch.equal` 前同样加）：

```python
def _compare_float(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """浮点精度比对，按 MERE/MARE 阈值判定"""
    golden_f = golden.flatten().float()
    actual_f = actual.flatten().float()

    # 空 tensor 视为匹配
    if golden_f.numel() == 0:
        return True

    # MERE/MARE: |actual - golden| / (|golden| + 1e-7)
    relative_errors = (golden_f - actual_f).abs() / (golden_f.abs() + EPS)
    mere = relative_errors.mean().item()
    mare = relative_errors.max().item()
```

PyTorch 对 `numel()==0` 的 `max()` 要求显式 dim——Python 侧的空短路与 C++ 侧（复现增量的 size==0 先于 null 检查）是同一条铁律的两个语言面。

**2. C++ 侧 float/int 路径一致性（缺陷信号）**：同文件里整数路径 `CompareResultsInt` 先判 `size==0` 再访问指针，浮点路径 `CompareResultsFloat` 却先判空指针——两条路径行为不一致本身就是缺陷信号。C++ Real 模式下空 `std::vector<T> npu_output(GetShapeSize(y_shape))` 的 `.data()` 返回 nullptr，null 检查先命中即误报 `[ERROR] null pointer input`；修复=把 `size==0` 短路提到判空之前（与整数路径一致，语义"空比较恒真"），before/after 形态与复现增量同构，此处补记触发场景：算子本身返回 `ACL_SUCCESS`、`CreateAclTensor` 对 size==0 已有 1 字节兜底、Mock 模式同用例通过——失败只在 Real 模式的比对器。

**3. Mock 掩蔽 Real 的判读边界**：本例 Mock 模式同一空用例通过（Mock 的 `output = golden` 拷贝掩盖了 Real 的空 vector 差异）——Mock 通过不能替代 Real 验证，空 tensor 用例必须两种模式都跑。

### 复现增量验证信号

- C++ 修复前 Real ST：空 tensor 用例 FAIL（`[ERROR] null pointer input`），34 用例 33/34；Mock 模式同用例通过（来源平台：950；证据等级：tool_observed）
- C++ 修复后重编译重跑：该用例输出 `[PASS] 空 tensor（0 元素），无需比对`，34/34 = 100%（来源平台：950；证据等级：tool_observed）
- Python 修复前：空切片用例 `RuntimeError: max(): ... numel() == 0`，65 用例全量运行中断（来源平台：950；证据等级：tool_observed）
- Python 修复后：65 用例全量跑完，空切片用例 PASS（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- 只处理"输出为空属合法结果"的场景；算子对空输入本应返回错误码而未返回，是算子侧问题另论。
- "golden 非空而 actual 为空"的 shape 不一致场景应由 shape 校验先拦截（未验证）。

## 后续增量（2026-09-17，并行策展批：跳过分配+全链路执行变体与 aclrtFree(nullptr) 语义）

bench_no_harness_sigmoid（Ascend950PR + CANN 9.1.0，ST Real 75 用例含 3 条空 Tensor）补上第三条已验证路线与两个配套事实：

**1) 失败复现（与复现增量同款、另一独立轨迹）**：`CreateAclTensor` 对 `hostData.size()==0` 无条件调 `aclrtMalloc(deviceAddr, 0, ACL_MEM_MALLOC_HUGE_FIRST)` 返回非 ACL_SUCCESS，3 个空 Tensor 用例（fp32/fp16/bf16 各一，`shape_size=0`）全部报 `[FAIL] 创建 x tensor 失败`，非空用例全过——失败与 dtype 无关，问题在脚手架而非 kernel（host 侧 `GetWorkspaceSize` 已有 `IsEmpty()` 快速返回、kernel 侧 early-return、Mock 模式已通过）。

**2) 跳过分配 + 全链路执行变体（与复现增量的"跳过+早返回"差异在**仍真实走 aclnn 调用链**）**：size==0 分支跳过 `aclrtMalloc/aclrtMemcpy`、`*deviceAddr = nullptr` 照常 `aclCreateTensor`（保留 0 维 shape 信息），空 Tensor 用例仍完整执行 `GetWorkspaceSize` + `aclnn` 两段式——**正是为了验证 host 侧 `IsEmpty()` 快速返回路径**（该用例的测试目的）；D2H 拷贝按 `outBytes > 0` 守卫：

```cpp
int CreateAclTensor(const std::vector<uint8_t>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor) {
    size_t size = hostData.size();

    // 空 Tensor 特殊处理: 跳过设备内存分配 (aclrtMalloc(0) 会失败)
    // Kernel 对空 Tensor 已有 early return 保护, 不会访问设备内存
    if (size == 0) {
        *deviceAddr = nullptr;
        auto strides = ComputeStrides(shape);
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                                   0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return ACL_SUCCESS;
    }

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        return ret;
    }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
```

```cpp
    // 拷贝输出到 Host（outBytes==0 时跳过 D2H，空 vector/空地址不走 aclrtMemcpy）
    size_t outBytes = static_cast<size_t>(shapeSize) * elemSize;
    std::vector<uint8_t> npuOutputBytes(outBytes);
    if (outBytes > 0) {
        ret = aclrtMemcpy(npuOutputBytes.data(), outBytes, yDevAddr, outBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) { /* 原失败清理路径不变 */ }
    }
```

**3) aclrtFree(nullptr) 语义**：按 ACL 文档返回 `ACL_SUCCESS`——清理路径 `aclrtFree(xDevAddr)` 对空 Tensor 无条件调用是安全的，无需按指针判空；`aclDestroyTensor` 对空 Tensor 描述符正常释放。

**4) 路线取舍（三条已验证路线的定位）**：占位分配（原卡 32B / 先前增量 64B）与跳过+早返回之外，本变体"跳过分配 + 仍走全链路"适用于**必须真实执行 host 侧空 Tensor 快速返回/路由**的用例；算子 host 侧没有 `IsEmpty()` 快速返回（真的会给 kernel 派发任务）时，nullptr 设备地址会被 kernel 侧解引用——移植前先确认双侧 early return。

### 复现增量验证信号

- 修复前：ST Real 75 用例中 3 个空 Tensor 用例报 `创建 x tensor 失败`（fp32/fp16/bf16 各一）（来源平台：950；证据等级：tool_observed）
- 修复后：ST Real 75/75 全过，空 Tensor 用例输出 `[PASS] 空 Tensor (0 元素)`（来源平台：950；证据等级：tool_observed）
- 回归：同轮 UT 52/52 不受影响；后续验收轮 ST Real 复跑仍 75/75（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- `aclrtMalloc(0)` 必失败的行为在 CANN 9.1.0 实测；其它版本若允许 0 字节分配，本修复仍然兼容（size==0 分支只是更早返回）。
- 跳过分配的前提是算子 host/kernel 双侧对空 Tensor early return（本算子已确认）；否则 nullptr 会在 kernel 侧解引用。
- 空 Tensor 用例只在 Mock 模式验证暴露不了本问题（Mock 不过 aclrt 路径）——与复现增量的"Mock 掩蔽 Real"边界同族，空 Tensor 用例必须 Real 模式跑。
- 误诊提示：同一批空 Tensor 用例在更早 Mock 调试轮的失败（`CompareResults` 判空指针检查先于 size==0）是另一独立根因（复现增量已记录），与本增量的 `aclrtMalloc(0)` 不可互相替代。

## 后续增量（2026-09-17，并行策展批：空 axes 形态 + aclrtMalloc(0) 确切错误码 100000 + 空 axes golden 全归约展开 + 分配/拷贝守卫须成对）

两条独立轨迹（L2 范数 reduce 算子 ST，Ascend950 + CANN 9.1.0，16 例矩阵含全 reduce 用例；Mish 类激活算子 ST Real，Ascend950PR + CANN 9.1.0，12 例含 shape=(0) 空 tensor 用例）补齐四点：

**1. aclrtMalloc(0) 的确切错误码（补 复现增量留空项）**：先前增量 边界曾标注"`aclrtMalloc` 对 0 字节请求的确切错误码未查证"。reduce 轨迹实测：全 reduce 用例（x=[1024]、axes=[]、keepdims=false）把 axes 建为 0 元素张量后，`nAxes=0 → axesBytesTotal=0`，`aclrtMalloc` 报 **ACL error 100000**（参数非法），错误行原文 `ACL error 100000 at .../tests/aclnn/test_aclnn.cpp:522: aclrtMalloc(&devAxes, axesBytesTotal, ACL_MEM_MALLOC_HUGE_FIRST)`——CHECK_ACL 打印错误并中止该用例，其余 15 例照常执行、该用例结果行直接缺失（用例在分配阶段中止，不是全量崩溃）。

修复前（按元素数直接算字节数传给 aclrtMalloc）：

```cpp
    size_t axesBytesTotal = (size_t)tc.nAxes * elemSizeOf(tc.axesDt);   // nAxes=0 → 0
    ...
    CHECK_ACL(aclrtMalloc(&devAxes, axesBytesTotal, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, tc.axesRaw, axesBytesTotal,
                          ACL_MEMCPY_HOST_TO_DEVICE));
```

修复后（本会话最终形态，含空张量创建；分配钳位 32B 与原卡口径同款，0 字节跳过 memcpy，空 axes 按 shape [0]、ndim=1 创建 aclTensor）：

```cpp
    size_t inBytesTotal  = (size_t)nIn  * elemSizeOf(tc.xDt);
    size_t axesBytesTotal = (size_t)tc.nAxes * elemSizeOf(tc.axesDt);
    size_t outBytesTotal = (size_t)nOut * elemSizeOf(tc.xDt);
    // aclrtMalloc rejects size 0; when axes is empty (full reduce, nAxes==0)
    // still allocate a minimal buffer so the (empty) axes tensor has a valid addr.
    size_t axesAllocBytes = std::max(axesBytesTotal, (size_t)32);

    // 2. Allocate device memory + copy input
    void *devX = nullptr, *devAxes = nullptr, *devOut = nullptr;
    CHECK_ACL(aclrtMalloc(&devX,    inBytesTotal,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devAxes, axesAllocBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut,  outBytesTotal,  ACL_MEM_MALLOC_HUGE_FIRST));
    // Zero output so stub (no-write) gives deterministic zeros
    CHECK_ACL(aclrtMemset(devOut, outBytesTotal, 0, outBytesTotal));
    CHECK_ACL(aclrtMemcpy(devX, inBytesTotal, inBytes.data(), inBytesTotal,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    if (axesBytesTotal > 0) {
        CHECK_ACL(aclrtMemcpy(devAxes, axesBytesTotal, tc.axesRaw,
                              axesBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // 3. Create tensors
    aclTensor* x    = createDenseTensor(tc.inShape, tc.inRank, tc.xDt, devX);
    // axes tensor: shape [nAxes] for non-empty, or [0] for full-reduce empty axes
    int64_t axesShape1[1] = { (int64_t)tc.nAxes };
    aclTensor* axes = createDenseTensor(axesShape1, 1, tc.axesDt, devAxes);
```

**2. 空 axes（值依赖 0 元素输入）是又一触发形态**：与前述 B==0/H==0 输入、TensorList 0 元素、`[0]` 形状并列——归约算子的 axes 张量本身可以是合法的 0 元素输入（nAxes=0 表示全归约）。分配守卫口径不变（钳位 + 0 字节拷贝跳过），空 axes 按 shape `[0]`、ndim=1 创建描述符。

**3. 空 axes 的 golden 语义（新误诊面，区别于 先前增量 的 oracle 越界）**：分配修复后全 reduce 用例 golden 显示 1.000000（期望约 18932）——不是分配问题复发，而是自建 CPU golden 没把空 axes 展开为全归约（按设计语义"0 元素 axes = 对全部维度归约"展开后 golden=sqrt(Σi², i=1..1024)=18932.458984）。修一层 bug 后要用数值合理性复核下一层，别把新现象归回旧根因；测试侧 golden 必须同步实现空 axes 的全归约展开。

**4. 分配/拷贝守卫须成对（Mish 轨迹，第三独立复现）**：CreateAclTensor 对 numel=0（shape=(0)）直接 `aclrtMalloc(dev, 0)` 返回非 SUCCESS，报 `[FAIL] 创建 x tensor 失败`（与 并行轨迹 同款失败形态、又一独立轨迹）。判别要点：该脚手架 `aclrtMemcpy` 已有 `size > 0` 保护、唯独 `aclrtMalloc` 没有——**分配与拷贝是两条独立路径，守卫必须成对补**。占位取 `sizeof(T)`（1 元素，与本卡先前增量 的 1 元素变体同款，与 32B/64B/1B 并列均为实测有效取值）：

```cpp
template<typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor) {
    size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);

    // 空 tensor(size==0)时 aclrtMalloc(0) 会失败，分配占位内存避免该问题；
    // 占位内存不会被实际读写（aclrtMemcpy 有 size>0 保护，aclCreateTensor 用 numel=0 的 shape）。
    size_t allocSize = (size > 0) ? size : sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }
    }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
```

取舍：占位方案不动各用例的清理逻辑（`aclrtFree` 照常可调），改动最小；跳过分配方案需同步改所有 free 点（独立 trace 已证 `aclrtFree(nullptr)` 返回 ACL_SUCCESS，两条路线均可行）。

### 复现增量验证信号

- 修复前（reduce 轨迹）：全 reduce 用例结果行缺失，日志出现 `ACL error 100000 at test_aclnn.cpp:522: aclrtMalloc(&devAxes, axesBytesTotal, ACL_MEM_MALLOC_HUGE_FIRST)`，其余 15 例照常执行（来源平台：950；证据等级：tool_observed）。
- 修复后（reduce 轨迹）：16 例全部执行、无 ACL error，全 reduce 用例结果行正常出现（该行 FAILED: exceeds rtol/atol 是 kernel stub 阶段的预期基线，分配问题已消除）（来源平台：950；证据等级：tool_observed）。
- 修复前（Mish 轨迹）：ST Real 12 条中 L0_024（shape=(0)）报 `[FAIL] 创建 x tensor 失败`，其余 11 条通过（来源平台：950；证据等级：tool_observed）。
- 修复后（Mish 轨迹）：ST Real 12/12 通过（含空 tensor 用例），CPU golden 自测 20/20；算子 op_api 层对空 tensor 返回 ACLNN_SUCCESS、workspaceSize=0、输出 numel=0 与 golden（空）比对直接通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 错误码 100000 绑定 CANN 9.1.0 实测（两条轨迹一致）；不同 ACL 版本对 size=0 的容忍度可能不同。
- 空 axes 的语义（0 元素 axes 张量 = 对全部维度归约）是算子级约定，测试侧 golden 必须同步实现该展开（见增量第 3 点）。
- 钳位取值 32B / sizeof(T) 均为实测有效（与 64B/1B/1 元素并列），非官方最小值；更小钳位值未测。

## 后续增量（2026-09-17，并行策展批：第四独立轨迹复现 + max(size,1) 变体 + 空张量/空指针用例语义区分）

scatter 类算子（bench_no_harness_scatter，Ascend950PR + CANN 9.1.0）的全 dtype ST 矩阵新增空 tensor 边界用例（CSV 外硬编码补充：`I3_BD_EMPTY_1D`：data shape `[0]`、indices `[0]`、dim=0、reduce=None；`I3_BD_EMPTY_2D`：data `[0,10]`、indices `[0,5]`）——第四条独立轨迹复现同一根因：Mock（CPU golden）模式两条通过（空张量按 0 元素直接判 PASS），Real 模式（真实调 aclnn 接口）两条都报 `[FAIL] 创建 data tensor 失败`。失败点在脚手架 `CreateAclTensor`：`size = numel * dtypeSize = 0` 直接传 `aclrtMalloc(dev, 0, ACL_MEM_MALLOC_HUGE_FIRST)` 不返回成功，函数在第一步就失败（与 并行轨迹/先前增量 同款形态、又一独立轨迹）。

**1. 修复（max(size,1) 一字节占位变体 + 条件 memcpy，与本卡先前增量 的 1 字节 / 先前增量 的 sizeof(T) 同族）**：

```cpp
// ===== 修复后 =====
size_t num_elems = static_cast<size_t>(GetShapeSize(shape));
size_t size = num_elems * DTypeSize(dtype);
// 空 tensor 时分配至少 1 字节 (aclrtMalloc(0) 可能失败)
size_t alloc_size = std::max<size_t>(size, 1);

auto ret = aclrtMalloc(deviceAddr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
if (ret != ACL_SUCCESS) return ret;

if (size > 0) {
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }
}
```

要点：`aclCreateTensor` 仍用真实 shape/strides（shape 含 0 维），设备地址只需有效；`aclrtMemcpy` 仅在 `size > 0` 时执行（0 长度拷贝顺带规避，且 `hostData.data()` 对空 vector 可能为 nullptr）。

**2. 占位而非 nullptr 的用例语义理由（新增设计决策）**：空 tensor（shape 含 0 维、元素数 0）与空指针（tensor/入参为 nullptr）是**两类用例**——本测试矩阵里另有一条"空指针异常"用例预期错误码 161001。若为空张量走"跳过分配传 nullptr"路线（并行轨迹 变体），会把"空张量合法退化"与"空指针非法输入"两类用例的设备地址形态混同，破坏各自断言语义；占位路线（本增量及原卡 32B）让两类用例在脚手架层天然可区分。选择跳过路线的既有前提（算子 host/kernel 双侧空张量早退）不变，此处补充第三个判据：测试矩阵中是否同时存在空指针类预期错误码用例。

### 复现增量验证信号

- 修复前：Real 模式 2 条空 tensor 用例（`I3_BD_EMPTY_1D/2D`）报 `创建 data tensor 失败`，Mock 模式同用例通过（来源平台：950；证据等级：tool_observed）。
- 修复后：边界补充用例 11/11（含空 tensor 1D/2D）通过；全量 ST Real 875/875、Mock 875/875（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- `aclrtMalloc(0)` 的确切返回码本轨迹未单独记录（以"失败"现象处理；错误码 100000 的实测见 复现增量，同为 950/CANN 9.1.0）。
- 仅覆盖 host 侧脚手架；若算子 kernel 侧没有 numel==0 早退，空张量仍可能在 kernel 侧出问题（本算子已处理）。
