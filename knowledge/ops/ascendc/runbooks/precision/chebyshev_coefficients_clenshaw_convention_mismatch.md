---
type: "Runbook"
title: Chebyshev 系数与 Clenshaw 约定不符输出小 4 个量级
description: 分段 Chebyshev 逼近 kernel 的系数数组与 Clenshaw 递推的消耗约定不匹配，上板输出比期望小 4~5 个数量级；用对递推本身最小二乘拟合的系数替换并离线验证后修复。
tags: [task.debugging, topic.precision, topic.activation]
status: stable
sources: []
created_at: '2026-09-17T04:37:43Z'
updated_at: '2026-09-17T04:37:43Z'
platforms: ["950"]
---

# Chebyshev 系数与 Clenshaw 约定不符输出小 4 个量级

## 现象

在 Ascend950（CANN 9.0.0-beta.2）上为自定义算子 bessel_i1e 的 Branch-0（FP32，tilingKey=0）做 kernel 联调。kernel 采用两段 Chebyshev 逼近：|x|≤8 段用 17 个系数 A（自变量 y=x/2−2，i1e(x)=x·chbevl(y,A)），|x|>8 段用 7 个系数 B（自变量 y=32/x−2，i1e(x)=chbevl(y,B)/√x），chbevl 用 Clenshaw 递推（b0=C[0]; 每轮 t=y·b0−b1+C[k]; 结果=(b0−b2)/2）在 AscendC 向量 API 上实现。

运行白盒 ST（`cd operators/BesselI1e_package/tests/aclnn && ./run.sh`，oracle 为 scipy.special.i1e，rtol=1e-5）时，14 条用例中 12 条精度失败，输出普遍比期望小 4~5 个数量级，但符号和分段结构都正确：

- `0-G_f32_scalar`：`Mismatch at [0]: got 1.08514323e-05, expected 0.187399983 (abs_err=0.187, rel_tol=1e-05)`
- `0-K_f32_max_arg`（走 B 段）：`got 2.9047435e-13, expected 0.01261093`
- `0-L_f32_negatives_full`：`got -5.11603389e-07, expected -0.045298446`（负号正确，量级错）
- `0-A_f32_main_block_aligned`：`got 1.58482235e-05, expected 0.134896502`

当时 kernel 的流水与数据通路已经过一轮排查：Compute 旁路为恒等搬运（y=x）时上板返回值与输入完全一致，说明 DataCopy/TilingData/多核切分都正常；故障集中在 Chbevl 求值结果过小。进一步的链路探针（见下文判别方法）显示：把 ChbevlA 硬编码为返回常数 0.05 时，输出正好变成 0.328×0.05=0.0164——调用链完全正确，问题在 Chbevl 内部的数值。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 直调开发 + aclnn 白盒 ST（scipy golden 对拍）。
- API 路径：用 AscendC 向量 API（`Mul`/`Sub`/`Adds`/`Muls`/`Duplicate`）实现的 Clenshaw 递推；系数常量数组放在 `op_kernel/*.h`（并镜像到 `op_host/*_tiling_host.h`）。
- 验证过的平台与 CANN 版本：Ascend950（AIV/VectorCore，kernel json `core_type: AIV`），CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有把 Cephes/scipy 式分段 Chebyshev 多项式逼近移植进 AscendC kernel 的特殊函数（i0e/i1e/iv/erf/gamma 族），只要 kernel 用 Clenshaw 递推求值。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：直接逐项求和（非 Clenshaw）的实现；系数由代码生成器与求值端同源生成的实现；FP16/bf16 上溯 FP32 计算路径（本卡只实测了 FP32 分支，Branch-1/2 kernel 当时未实现）。

## 根因

kernel 内 `CHEBYSHEV_A[17]`/`CHEBYSHEV_B[7]` 系数数组与 Clenshaw 递推的系数消耗约定不匹配，多项式退化为量级 1e-6 的近常数。

机制链（用离线复现逐项验证过）：

1. 该 Clenshaw 递推（与 Cephes `chbevl` 同构）的求值语义是 `chbevl(y, C) = Σ_{i=0}^{N-1} C[i]·T_{N-1-i}(y/2)`，其中**最后一个数组元素充当减半的常数项**（T_0 项），即数组是按"最高次系数在前、常数项在末尾且存 c0/2"的约定消耗的。
2. kernel 里的旧数组是一组 2.78e-18 … 6.50e-6 递增的微小值（与 `tests/aclnn/golden.py` 里 scipy 缺失时的回退路径数组**逐字节相同**，而该回退在 golden.py 中被注释明确标注 `Crude fallback; NOT for real validation`）。在这组值里，最大值 6.4959e-6 恰好落在常数位，其余系数都 ≤3.4e-7。
3. 于是整段多项式 ≈ 常数 3.2e-6：离线复现 x=3.5（y=−0.25）得 `chbevl=3.100409e-06`，而目标 `i1e(3.5)/3.5=5.354285e-02`，差 4 个数量级；`a·chbevl=1.085143e-05` 与上板 `got 1.08514323e-05` 完全一致——离线模型逐位复现了上板故障。
4. 双精度对照：FP32 与 FP64 离线计算同得 1.085143e-05，排除"kernel 精度模式/中间类型"问题，锁定系数本身；B 段同样过小（x=3.5 时 y_gt=7.14，`chbevl=4.478896e-12`），两段系数都错。

一句话根因：**把与求值递推约定不匹配（且本身就不是该函数该分段）的系数数组抄进了 kernel，使 Chebyshev 逼近退化为 1e-6 量级常数。**

## 判别方法

按以下顺序隔离（每步命令均在本会话真实执行）：

1. 数据通路探针：Compute 旁路恒等搬运 → got 与输入一致；`|x|*0.5` → got 为输入一半；确认 DMA/tiling 正常。
2. 调用链探针：把 ChbevlA 硬编码 `Duplicate(r, T(0.05f), cnt)` → got=0.328×0.05，确认调用链正常、Chbevl 内部错。
3. golden 对拍：`echo '[3.5]' | python3 golden.py` → `[0.1873999834060669]`。
4. 离线隔离到 chbevl 一级（python 复刻 kernel 递推）：

```python
# x=3.5: y_le = 3.5*0.5-2 = -0.25
# y=-0.25, chbevl=3.100409e-06, a*chbevl=1.085143e-05   ← 代码系数
# y_gt=32/a-2=7.142857, chbevl=4.478896e-12             ← B 段同样过小
# scipy i1e(3.5) = 1.874000e-01
```

5. 精度模式排除：FP32/FP64 双精度下离线结果相同（1.085143e-05）→ 不是精度问题，是系数问题。

## 解决配方

**第 1 步：对"kernel 的递推本身"做线性最小二乘拟合**（不要手工翻译 Cephes 约定）。利用 chbevl 对系数是线性函数这一性质，用单位向量的递推响应构造设计矩阵，在 kernel 实际的自变量映射下直接拟合目标函数：

```python
import numpy as np
from scipy.special import i1e

# The kernel's chbevl implementation:
# b0 = coeffs[0]; b1 = 0
# for ci in coeffs[1:]:
#     b2 = b1; b1 = b0
#     b0 = y * b1 - b2 + ci
# return 0.5 * (b0 - b2)
def chbevl_kernel(y, coeffs):
    b0 = np.float64(coeffs[0])
    b1 = np.float64(0.0)
    for c in coeffs[1:]:
        b2 = b1
        b1 = b0
        b0 = y * b1 - b2 + np.float64(c)
    return np.float64(0.5) * (b0 - b2)

# For x in [0, 8], y = x/2 - 2
# We want: x * chbevl(y, A) ≈ i1e(x)
x_samples = np.linspace(0.001, 8.0, 200)
y_samples = x_samples * 0.5 - 2.0
target = np.float64(i1e(x_samples) / x_samples)  # f(y) = i1e(x)/x

N = 17  # number of coefficients
M = len(x_samples)
design = np.zeros((M, N), dtype=np.float64)

for i in range(M):
    y = y_samples[i]
    # Unit response for each coefficient
    for k in range(N):
        unit = np.zeros(N, dtype=np.float64)
        unit[k] = 1.0
        design[i, k] = chbevl_kernel(y, unit)

# Solve: design @ A = target
A_fitted, residuals, rank, s = np.linalg.lstsq(design, target, rcond=None)

# Now for B coefficients: x > 8, y = 32/x - 2
# i1e(x) ≈ chbevl(y, B) / sqrt(x)
x_samples_B = np.logspace(np.log10(8.001), np.log10(500), 100)
y_samples_B = 32.0 / x_samples_B - 2.0
target_B = np.float64(np.sqrt(x_samples_B) * i1e(x_samples_B))

NB = 7
design_B = np.zeros((len(x_samples_B), NB), dtype=np.float64)
for i in range(len(x_samples_B)):
    y = y_samples_B[i]
    for k in range(NB):
        unit = np.zeros(NB, dtype=np.float64)
        unit[k] = 1.0
        design_B[i, k] = chbevl_kernel(y, unit)

B_fitted, _, _, _ = np.linalg.lstsq(design_B, target_B, rcond=None)
```

**第 2 步：离线验证通过后再上板**（拟合输出对采样点 err：A 段 ≤7.6e-09，B 段至 x=500 err≤2.8e-10）。

**第 3 步：替换两处镜像数组**。修复前（`op_kernel/bessel_i1e_kernel_vf.h`，与 `op_host/bessel_i1e_tiling_host.h` 中的镜像相同）：

```cpp
constexpr float CHEBYSHEV_A[17] = {
    2.77791411276104639959E-18f, -2.11142121435816608115E-17f,
    1.55399195752520090135E-16f, -1.10545994764943145678E-15f,
    7.64446877936612547782E-15f, -5.14308291745161174632E-14f,
    3.36357093828701759722E-13f, -2.13565309620616154013E-12f,
    1.31606846203035341898E-11f, -7.87234605989303118235E-11f,
    4.55046256763641954564E-10f, -2.52963790457670368256E-9f,
    1.35151178263152548461E-8f,  -6.91597250573807853029E-8f,
    3.37942895469575000000E-7f,  -1.53820626121875000000E-6f,
    6.49591721484375000000E-6f
};
```

修复后（`op_kernel/bessel_i1e_kernel_vf.h` 与 `op_host/bessel_i1e_tiling_host.h` 同步替换）：

```cpp
constexpr float CHEBYSHEV_A[17] = {
    9.2320585708794722E-09f, -4.3706852975375354E-08f,
    2.0023992813725965E-07f, -8.5645023634317802E-07f,
    3.4701937481136858E-06f, -1.3272911466645882E-05f,
    4.7815613880922009E-05f, -1.6176066937278108E-04f,
    5.1228593280339726E-04f, -1.5135723739867130E-03f,
    4.1564229304208904E-03f, -1.0564084863931138E-02f,
    2.4726449022945931E-02f, -5.2945981205519438E-02f,
    1.0264365868569979E-01f, -1.7641651836864802E-01f,
    2.5258718644063438E-01f
};
```

B 段修复前 / 修复后：

```cpp
// 修复前
constexpr float CHEBYSHEV_B[7] = {
    4.98749908012204158325E-17f, -3.63514211956515743992E-16f,
    4.67801103454184667670E-15f, -4.34604652181746787884E-14f,
    3.52542803647240444551E-13f, -2.41861561165720342835E-12f,
    1.28688268193391000000E-11f
};
// 修复后（拟合输出，两处镜像同步）
constexpr float CHEBYSHEV_B[7] = {
    -3.6185884312217630E-09f, -2.6134597763682876E-08f,
    -2.5128405162534186E-07f, -3.8824723081261372E-06f,
    -1.1058907811527627E-04f, -9.7610974277754763E-03f,
    7.7857623485748872E-01f
};
```

**第 4 步**：替换后先经历一次 TBE 编译报错（编辑过程中 ChbevlB 出现重复定义），删除重复定义后全量回归。

## 设计决策

- **为什么用最小二乘拟合而不是直接抄 Cephes/scipy 系数**：网络获取 Cephes i1e.c 源码失败（webfetch 超时）；更重要的是拟合直接绑定本 kernel 的递推与自变量映射，绕开"系数存储约定翻译"这一最容易出错的环节——设计矩阵法利用 chbevl 对系数线性，用单位响应即可构造，不需要推导任何约定。
- **为什么同时替换 `op_host/bessel_i1e_tiling_host.h` 的镜像数组**：host 侧存在同名常量副本，两处必须同步，避免后续发散。
- **为什么保留 `Muls(b, x, 1.0f)` 值拷贝旋转**：前一轮会话怀疑 TBuf 间值拷贝有精度问题而做了此替换；系数修复后该版本经全量验证正确，未回退（但那个怀疑本身未被证实，见误诊）。
- **当前方案路线档位**：recommended_fastpath（正确性优先；值拷贝旋转的性能代价未测量，见遗留问题）。

## 系统信息

- 硬件/软件：Ascend950 单卡（npu-smi 25.7.rc1，driver 25.7.rc1），CANN 9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`），kernel 为 VectorCore/AIV（`RT_DEV_BINARY_MAGIC_ELF_AIVEC`）。
- 工程布局（`operators/BesselI1e_package/`）：
  - `op_kernel/bessel_i1e_kernel_vf.h`——`CHEBYSHEV_A[17]`/`CHEBYSHEV_B[7]` 常量 + VF identity helper（满足 `__simd_vf__`/`asc_vf_call` 硬约束）。
  - `op_kernel/bessel_i1e_kernel.cpp`——`KernelBesselI1e`（Init/Process/CopyIn/Compute/CopyOut、ChbevlA/ChbevlB）。
  - `op_host/bessel_i1e_tiling_host.h`——系数镜像副本。
  - `tests/aclnn/{run.sh,test_aclnn.cpp,golden.py}`——白盒 ST（14 条）+ scipy oracle。
  - `tests/ut/`——TilingUT。
- 测试命令：`cd operators/BesselI1e_package/tests/aclnn && rm -rf build && ./run.sh`（每轮全量重建，避免旧产物干扰）。

## 完整代码

ChbevlA 终版（`op_kernel/bessel_i1e_kernel.cpp`，系数修复后的最终形态；ChbevlB 同构，循环 6 轮、系数 7 个）：

```cpp
    // Chbevl for Chebyshev A (N=17)
    __aicore__ inline void ChbevlA(
        AscendC::LocalTensor<T>& r,
        AscendC::LocalTensor<T>& y,
        uint32_t cnt)
    {
        auto b0 = BufTensor(bufB0);
        auto b1 = BufTensor(bufB1);
        auto b2 = BufTensor(bufB2);
        auto t  = BufTensor(bufBT);

        AscendC::Duplicate(b0, T(CHEBYSHEV_A[0]), cnt);
        AscendC::Duplicate(b1, T(0.0f), cnt);

        float ca[16] = {CHEBYSHEV_A[1],CHEBYSHEV_A[2],CHEBYSHEV_A[3],CHEBYSHEV_A[4],
                        CHEBYSHEV_A[5],CHEBYSHEV_A[6],CHEBYSHEV_A[7],CHEBYSHEV_A[8],
                        CHEBYSHEV_A[9],CHEBYSHEV_A[10],CHEBYSHEV_A[11],CHEBYSHEV_A[12],
                        CHEBYSHEV_A[13],CHEBYSHEV_A[14],CHEBYSHEV_A[15],CHEBYSHEV_A[16]};
        for (int j = 0; j < 16; ++j) {
            AscendC::Mul(t, y, b0, cnt);
            AscendC::Sub(t, t, b1, cnt);
            AscendC::Adds(t, t, T(ca[j]), cnt);
            AscendC::Muls(b2, b1, T(1.0f), cnt);
            AscendC::Muls(b1, b0, T(1.0f), cnt);
            AscendC::Muls(b0, t, T(1.0f), cnt);
        }
        AscendC::Sub(t, b0, b2, cnt);
        AscendC::Muls(r, t, T(0.5f), cnt);
    }
```

Compute 终版（两段求值 + 用 Relu/Div 算术实现分段 Select + 奇函数符号恢复）：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        auto a    = BufTensor(bufA);
        auto yLe  = BufTensor(bufYLe);
        auto yGt  = BufTensor(bufYGt);
        auto t1   = BufTensor(bufT1);
        auto chA  = BufTensor(bufChA);
        auto rA   = BufTensor(bufRA);
        auto chB  = BufTensor(bufChB);
        auto rB   = BufTensor(bufRB);
        auto mask = BufTensor(bufMask);
        auto ones = BufTensor(bufOnes);
        auto tmp  = BufTensor(bufTmp);
        auto zeroT = BufTensor(bufZeroT);

        // a = |x|
        AscendC::Abs(a, xLocal, c);

        // y_le = a * 0.5 - 2.0 (for [0,8])
        AscendC::Muls(yLe, a, T(0.5f), c);
        AscendC::Adds(yLe, yLe, T(-2.0f), c);

        // y_gt = 32.0 / a - 2.0 (for (8,inf))
        AscendC::Duplicate(t1, T(32.0f), c);
        AscendC::Div(yGt, t1, a, c);
        AscendC::Adds(yGt, yGt, T(-2.0f), c);

        // rA = a * chbevl(y_le, A, 17)
        ChbevlA(chA, yLe, c);
        AscendC::Mul(rA, a, chA, c);

        // rB = chbevl(y_gt, B, 7) / sqrt(a)
        ChbevlB(chB, yGt, c);
        AscendC::Sqrt(t1, a, c);
        AscendC::Div(rB, chB, t1, c);

        // Select: r = a <= 8.0 ? rA : rB
        AscendC::Duplicate(ones, T(1.0f), c);
        AscendC::Duplicate(t1,  T(8.0f), c);
        AscendC::Sub(t1, t1, a, c);
        AscendC::Abs(tmp, t1, c);
        AscendC::Relu(mask, t1, c);
        AscendC::Maxs(tmp, tmp, T(1e-30f), c);
        AscendC::Div(mask, mask, tmp, c);

        AscendC::Sub(ones, ones, mask, c);
        AscendC::Mul(tmp, mask, rA, c);
        AscendC::Mul(t1, ones, rB, c);
        AscendC::Add(tmp, tmp, t1, c);

        // Sign recovery (i1e is odd)
        AscendC::Duplicate(zeroT, T(0.0f), c);
        AscendC::Sub(t1, zeroT, xLocal, c);
        AscendC::Relu(t1, t1, c);
        AscendC::Abs(mask, xLocal, c);
        AscendC::Maxs(mask, mask, T(1e-30f), c);
        AscendC::Div(t1, t1, mask, c);

        AscendC::Duplicate(ones, T(1.0f), c);
        AscendC::Sub(ones, ones, t1, c);
        AscendC::Neg(chA, tmp, c);
        AscendC::Mul(mask, t1, chA, c);
        AscendC::Mul(chB, ones, tmp, c);
        AscendC::Add(yLocal, mask, chB, c);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

## 预防

- 移植任何分段多项式逼近前，先在 host 侧用 python 逐行复刻 kernel 的**确切递推与自变量映射**，对目标函数抽样验证（err ≤1e-8 量级）再上板；一次离线对拍的成本远低于一轮上板盲试。
- 检查系数来源：golden 脚本中标注 `Crude fallback; NOT for real validation` 的回退路径数组不能当作实现系数抄进 kernel——真实 oracle 走 scipy，两者数值可以相差数个数量级。
- 系数常量在 kernel 头与 host tiling 头有两份镜像时，修改必须成对提交。
- 上板二分定位：先用恒等搬运/单算子/硬编码子结果探针把故障隔离到单个函数，再离线复现数值。

## 验证信号

以下信号来自 bessel_i1e Branch-0（FP32）kernel 的一次缺陷修复会话：先用白盒 ST 与链路探针把故障隔离到 Chbevl，再离线复现与拟合系数，替换后全量回归并由独立验收会话复跑。

- 修复前白盒 ST：12/14 FAIL；`0-G_f32_scalar got 1.08514323e-05 vs expected 0.187399983 (abs_err=0.187)`、`0-K_f32_max_arg got 2.9047435e-13 vs expected 0.01261093`、`0-L_f32_negatives_full got -5.11603389e-07 vs expected -0.045298446`（来源平台：950；证据等级：tool_observed）。
- 链路探针：恒等搬运 got=输入；ChbevlA 硬编码 0.05 时 got 0.0163986217 = 0.328×0.05，调用链正常（来源平台：950；证据等级：tool_observed）。
- 修复后白盒 ST：14/14 PASS（`PASS: 14 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 14`），覆盖主块/尾块/UB 尾片/单元素/多核/rank8/scalar/空 tensor/边界/特殊值/max_arg/全负/128×256 大 shape/零轴（来源平台：950；证据等级：tool_observed）。
- 独立验收复跑：14/14 PASS 且 `EXIT_CODE=0`；`__simd_vf__`/`asc_vf_call` grep 计数保持 2（来源平台：950；证据等级：tool_observed）。
- 离线拟合验证（宿主机 python，非上板）：A 段采样点 err ≤7.6e-09（x∈{0.1,0.5,1,3.5,7,8}），B 段 err ≤2.8e-10（x∈{8,10,50,100,500}）（来源平台：宿主机 python 离线计算；证据等级：tool_observed）。

## 失效边界

- 仅实测 FP32 分支（Branch-0）；FP16/bf16 上溯 FP32 的计算路径（Branch-1/2 kernel）当时未实现，本卡结论未在那两个分支验证。
- 拟合系数的离线验证范围为 A 段 x∈[0.1,8]、B 段 x∈(8,500]；x>500 的外推行为未验证。
- 本卡仅在 Ascend950 实测；其他 NPU 平台未验证，推断可迁移（根因是系数数组与求值递推的约定错配，属纯数值问题，不依赖平台特有指令语义）。
- 恒等搬运/硬编码探针的结论依赖"测试 checker 从首个失配元素起扫描"（失败输出报 `Mismatch at [0]`/`[5]`），只校验单点且不扫描的 checker 会削弱探针判别力。
- 修复后同会话出现的 507035 向量核异常是另一独立根因（tiling caps 移除后 UB 超额），不属于本卡。

**未决项**：

- 旧系数数组的确切来源（是否抄自 Cephes i1e.c、其存储顺序约定如何）未能核实：网络获取 Cephes/scipy 源码超时失败；本卡根因判定不依赖该来源（离线行为复现已闭环）。
- ChbevlA 每轮 5 条向量指令 × 16 轮的 `Muls` 值拷贝旋转的性能代价未测量（无 profiling 数据）；若成为瓶颈可探索寄存器级旋转或减少 TBuf 往返（未尝试）。
## 坏实践

- 把 golden 脚本中明确标注"仅供冒烟、不得用于真实验证"的回退系数当作实现系数抄进 kernel。
- 不做离线数值对拍，直接上板反复试系数/试精度模式。
- 系数两处镜像只改一处。

## 误诊

- 前一轮修复会话曾把 ~1e-6 输出归因为"Chbevl TBuf 间值拷贝精度异常"，并在开发日志中遗留"Chbevl 需要更深 TBuf 排布或寄存器级实现"的结论（该归因为会话日志陈述，证据等级：agent_reported；本轮证明实际根因是系数，值拷贝本身无精度问题）。该轮引入的 `Muls(b,x,1.0f)` 值拷贝与 17 个 TBuf 改造被保留并验证正确，但归因不成立。
- 修复初期怀疑"构建未生效、上板跑的是旧二进制"：通过 sha256 比对装机 `.o` 与 kernel json 内嵌哈希一致、以及输出注入 +100 常数后 got 100 排除（部署是新鲜的）。
