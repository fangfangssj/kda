---
type: "Runbook"
title: "ACL BOOL 测试数据 vector<bool> 取 data() 编译失败"
description: "ST 测试用 std::vector<bool> 存 ACL BOOL（finished 类布尔输入）数据并取 .data() 报 \"use of deleted function 'void std::vector<bool>::data()'\"——vector<bool> 是位压缩特化且 ACL BOOL 张量每元素 1 字节；finished 全链路改 uint8_t/vector<uint8_t> 后编译与 Mock 全过。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["agnostic"]
---

# ACL BOOL 测试数据 vector<bool> 取 data() 编译失败

## 现象

为 Softmax+TopK 融合类算子（带可选 bool 输入 `finished`，ACL_BOOL）开发 C++ ST 测试工程。测试代码按"bool 数据就用 vector<bool>"的直觉写：

```cpp
    std::vector<bool> finished;
    const bool* finishedPtr = nullptr;
    if (tc.finishedMode != FinishedMode::NONE) {
        GenerateFinished(finished, M, tc.finishedMode);
        finishedPtr = finished.data();   // ← 编译报错处
    }
```

`bash run.sh --mock`（Mock 模式，CPU golden 验证，无需 NPU）在编译测试程序时失败：

```text
tests/st/test_aclnn_<op>.cpp: In function 'bool RunTest(const TestCase&)':
...:850:36: error: use of deleted function 'void std::vector<bool, _Alloc>::data() [with _Alloc = std::allocator<bool>]'
  850 |         finishedPtr = finished.data();
      |                       ~~~~~~~~~~~~~^
/usr/include/c++/13/bits/stl_bvector.h:1559:12: note: declared here
  1559 |       void data() = delete;
```

golden 计算、语义验证、数据生成函数（`ApplyFinished`/`ComputeFullGolden`/`VerifyExpertIdx`/`GenerateFinished`）的 finished 形参也都写成了 `const bool*`/`std::vector<bool>&`。

## 适用场景

- 工程形态：自定义算子 ST 测试工程（C++ 原生测试，构造 ACL 张量、CPU golden 比对），涉及 ACL_BOOL 类型的布尔输入。
- API 路径：`aclCreateTensor`/自建 CreateAclTensorGeneric 以 `ACL_BOOL` 建张量（**1 byte/element**）；C++ 标准库 `std::vector<bool>` 特化。
- 验证过的平台与 CANN 版本：宿主机 x86_64（g++ 13.3.0）Mock 编译与 CPU golden 自测通过；ACL_BOOL 元素宽度按 ACL 头定义取 1 字节（结论按定义与硬件平台无关，本片未上板）。
- 适用粒度：platform_wide——所有需要构造 ACL_BOOL 设备数据的 C++ 测试代码。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：Python 侧 numpy 布尔数组（np.bool_ 为 1 字节，天然匹配）；非 BOOL dtype 的测试数据不受本卡约束。

## 根因

`std::vector<bool>` 是标准库的位压缩特化：元素按 bit 打包、不保证连续存储，`data()` 被显式 `= delete`，无法取裸指针传给 `const bool*`；而 ACL 的 BOOL 张量按每元素 1 字节（uint8）解释设备内存。测试数据通道必须同时满足"C++ 容器可取连续指针"与"ACL_BOOL 元素宽度 1 字节"，`std::vector<uint8_t>`（0/1 取值）是两者交集。

## 解决配方

finished 数据全链路从 `bool`/`std::vector<bool>` 改为 `uint8_t`/`std::vector<uint8_t>`（0=False，非 0=True），建 ACL_BOOL 张量时按 `M` 字节传入。

before（首次写入的测试文件，三处代表性片段）：

```cpp
// finished 处理：finished[m]=True 时，expert_idx 对应行全部置为 E
static void ApplyFinished(int32_t* indices, const bool* finished, int64_t M, int64_t k, int32_t E) {
    if (finished == nullptr) return;
    for (int64_t m = 0; m < M; ++m) {
        if (finished[m]) {
            for (int64_t j = 0; j < k; ++j) {
                indices[m * k + j] = E;
            }
        }
    }
}
```

```cpp
    std::vector<bool> finished;
    const bool* finishedPtr = nullptr;
    if (tc.finishedMode != FinishedMode::NONE) {
        GenerateFinished(finished, M, tc.finishedMode);
        finishedPtr = finished.data();
    }
```

after（类型统一为 uint8_t；golden 函数 `ComputeFullGolden`/`VerifyExpertIdx` 形参与生成函数同步替换）：

```cpp
// finished 处理：finished[m]!=0 时，expert_idx 对应行全部置为 E
// 使用 uint8_t 存储（对标 ACL BOOL 类型，1 byte/element）
static void ApplyFinished(int32_t* indices, const uint8_t* finished, int64_t M, int64_t k, int32_t E) {
    if (finished == nullptr) return;
    for (int64_t m = 0; m < M; ++m) {
        if (finished[static_cast<size_t>(m)] != 0) {
            for (int64_t j = 0; j < k; ++j) {
                indices[m * k + j] = E;
            }
        }
    }
}
```

```cpp
    // 2. 生成 finished
    std::vector<uint8_t> finished;
    const uint8_t* finishedPtr = nullptr;
    if (tc.finishedMode != FinishedMode::NONE) {
        GenerateFinished(finished, M, tc.finishedMode);
        finishedPtr = finished.data();
    }
```

建 ACL_BOOL 张量时按 1 字节/元素计算字节数（对齐 ACL BOOL 语义）：

```cpp
    // finished tensor
    if (tc.finishedMode != FinishedMode::NONE) {
        size_t finishedBytesSize = static_cast<size_t>(M);
        if (CreateAclTensorGeneric(finished.data(), finishedShape, &finishedDev, ACL_BOOL,
                                   finishedBytesSize, &finishedTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 finished tensor 失败");
            aclDestroyTensor(xTensor); aclrtFree(xDev);
            return false;
        }
    }
```

## 设计决策

- 选 `uint8_t` 而不是 `std::deque<bool>`/逐元素拷贝到裸数组：uint8_t 直接满足连续性 + 1 字节宽度，data() 可用，golden 函数形参只改类型不动逻辑；deque 无 data()，逐元素拷贝徒增代码。
- 判断条件统一写 `!= 0` 而不是隐式真值：uint8_t 语义下显式更安全（设备侧 BOOL 的合法取值为 0/非 0）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 环境：x86_64 宿主机，g++ 13.3.0，CANN 9.1.0 头文件；ST 测试目录 `tests/st/`（test_aclnn_*.cpp + CMakeLists.txt + run.sh，Mock 模式）。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（本次修复，涉及 ApplyFinished/ComputeFullGolden/VerifyExpertIdx/GenerateFinished/RunTest 及 golden 自测共 9 处类型替换）。

## 预防

- 为 ACL BOOL 张量准备宿主侧数据时，一律 `std::vector<uint8_t>`（0/1）；写测试代码模板时把"bool 输入 → uint8_t 存储"列为固定转换项，避免 vector<bool> 特化陷阱。

## 验证信号

以下信号来自类型替换修复前后连续两次 `bash run.sh --mock`：

- 修复前：`error: use of deleted function 'void std::vector<bool, _Alloc>::data()'`，测试程序编译失败（来源平台：宿主机 x86_64 Mock 编译；证据等级：tool_observed）。
- 修复后：测试程序编译链接成功；CPU golden 自测 12/12 PASS（softmax/topk/row_idx 2D&3D/finished/端到端/k=E/E=1/FP16 转换/BF16 转换/3D 多行），L0 用例 7/7 PASS、L1 用例 38/38 PASS，总计 45/45（来源平台：宿主机 x86_64 Mock 运行；证据等级：tool_observed）。

## 失效边界

- 本卡验证到 Mock 编译 + CPU golden 层；ACL_BOOL 设备张量的 1 字节宽度按 ACL 头定义采信，上板读写未验证（推断无差异：宿主侧按 1 字节准备数据与 ACL 定义一致）。
- `finished[m] != 0` 的判定对设备回读的任意非 0 值均按 True 处理，与 ACL BOOL 语义一致。
- 仅验证 g++ 13 / CANN 9.1.0 头；ACL BOOL 宽度属 ABI 定义，跨版本推断不变（未验证，推断可迁移）。

**未决项**：

- 无（该根因已闭环到 Mock 全量通过；上板 ACL_BOOL 真实设备张量读写随 Real 模式验证补齐）。
## 坏实践

- 用 `std::vector<bool>` 承载要传给 ACL 的布尔张量数据——位压缩特化无连续存储，除了 data() 被删，即便强取地址也与 1 字节/元素的设备语义不符。

## 误诊

- 无（编译器 note 直接指向 stl_bvector.h 的 deleted data()）。
