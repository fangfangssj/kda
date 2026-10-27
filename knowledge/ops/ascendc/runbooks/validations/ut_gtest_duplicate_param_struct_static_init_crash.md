---
type: "Runbook"
title: UT 多文件同名参数结构体致 gtest 静态初始化段错误
description: 同一 UT binary 内两个测试文件定义同名参数结构体（如 EmptyCase，新文件中位于全局作用域），gtest 参数化注册在静态初始化期崩溃，--gtest_list_tests 即段错误、编译链接无告警；为分支 UT 的本地类型加分支前缀可修复。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# UT 多文件同名参数结构体致 gtest 静态初始化段错误

## 现象

在一个 softmax 基准算子（opbuild 自定义算子包，Ascend 950，CANN 9.1.0）的测试先行开发中，为空 tensor 分支（EmptyTensor）新增了 TilingUT 测试文件 `operators/<OP>_package/tests/ut/test_<op>_branch2_tiling.cpp`：文件里定义了参数化测试的参数结构体 `struct EmptyCase`（name/shape/dim/dtype 四字段，`std::vector<int64_t> shape` 成员），用 `TEST_P` + `INSTANTIATE_TEST_SUITE_P` 注册了 14 个参数化用例和 1 个 glue 契约用例。该文件与既有的公共 tiling UT（`test_<op>_tiling.cpp`）、Branch-0/Branch-1 UT 一起，经 `tests/ut/CMakeLists.txt` 编入同一个 gtest 可执行文件（`build/tests/ut/<op>_tiling_ut`）。

重新 `./build.sh <OP>_package` 编译成功后，连最基础的枚举命令都在进入 `main()` 之前就崩溃：

```
$ operators/<OP>_package/build/tests/ut/<op>_tiling_ut --gtest_list_tests
/bin/bash: line 1: 285449 Segmentation fault      (core dumped) operators/<OP>_package/build/tests/ut/<op>_tiling_ut --gtest_list_tests
```

`--gtest_list_tests` 只做静态注册期遍历、不执行任何用例体，因此崩溃点在全局对象构造/静态初始化阶段。编译、链接均无任何告警。

## 适用场景

- 工程形态：opbuild 算子包的 `tests/ut/` 多文件 gtest 工程（CMake 把同目录多个 `test_*.cpp` 链进**单个** UT binary）；为新增分支（Branch-N）追加 UT 文件时。
- API 路径：googletest 的 `TEST_P` / `INSTANTIATE_TEST_SUITE_P` / `testing::TestWithParam<ParamType>` 参数化注册机制；GCC 13.3.0 + CANN 9.1.0 环境的算子包 UT 构建（`./build.sh`）。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.1.0（与 frontmatter `platforms` 一致；本崩溃为 host 侧 gtest 行为，随算子包 UT 构建验证）。
- 适用粒度：workflow_only（UT 工程组织方式，不涉 kernel 行为）。
- 路线优先级：recommended_fastpath（新增 UT 类型一律加分支前缀，一次到位）。
- 明确不适用 / 未验证场景：非 gtest 参数化框架的测试；单个 UT 文件的工程（无跨文件链接）；MSVC/clang 下的表现未验证。

## 根因

同一 gtest binary 的两个编译单元各自定义了同名参数结构体 `EmptyCase`（新分支文件中位于全局作用域、匿名 namespace 之外），gtest `INSTANTIATE_TEST_SUITE_P` 为参数化套件生成的全局注册对象在静态初始化期发生类型冲突，`main()` 之前即段错误。

## 判别方法（隔离链，全部为可复现命令）

1. 崩溃特征定性——`--gtest_list_tests`（不执行用例）即崩 → 静态初始化期问题，与用例逻辑无关。
2. 最小化渐进复现（新文件 + 被测 host 源单独链接，逐步加回内容）：

```bash
# (a) 仅 EmptyCase + INSTANTIATE 的最小文件 → 正常
g++ -std=c++17 -I${PKG}/op_host -I${PKG}/op_kernel -I/usr/include/gtest \
    /tmp/test_branch2_min.cpp ${PKG}/op_host/<op>_tiling_host.cpp \
    -lgtest -lgtest_main -lpthread -o /tmp/test_branch2_min
/tmp/test_branch2_min --gtest_list_tests   # RUN_EXIT=0

# (b) 加上 OracleEmpty struct + 全局 const kOracleEmpty → 仍正常（排除全局 const 嫌疑）

# (c) 新 UT 文件单独与 tiling_host.cpp 链接 → --gtest_list_tests 正常列出全部用例
g++ -std=c++17 ... ${PKG}/tests/ut/test_<op>_branch2_tiling.cpp \
    ${PKG}/op_host/<op>_tiling_host.cpp -lgtest ... -o /tmp/test_branch2_standalone
# 单独运行正常 → 崩溃只在"全部 4 个 UT 文件链入同一 binary"时出现 → 跨文件符号/类型冲突
```

3. 同名冲突定位（跨测试文件 grep）：

```bash
grep -n 'struct EmptyCase\|OracleEmpty\|kOracleEmpty\|FillGarbage\|OracleHandleEmptyTensor' \
    operators/<OP>_package/tests/ut/*.cpp
# 命中：公共 test_<op>_tiling.cpp:215: struct EmptyCase（:48: OracleHandleEmptyTensor）
#      新文件 test_<op>_branch2_tiling.cpp:117: struct EmptyCase（:72: OracleHandleEmptyTensor）
```

## 解决配方

新分支 UT 文件内所有本地类型/函数一律加分支前缀，保证全 binary 唯一。修复轮次的实际改名映射：

- `EmptyCase` → `B2EmptyCase`（参数结构体，`TestWithParam<B2EmptyCase>` 模板实参随之唯一化——这是注册冲突的源头）
- `OracleEmpty` → `B2OracleEmpty`，`kOracleEmpty` → `kB2OracleEmpty`
- `OracleHandleEmptyTensor` → `B2OracleHandleEmptyTensor`
- `FillGarbage` → `B2FillGarbage`

改名后的参数化注册（before/after 关键段，verbatim）：

```cpp
// ── before（崩溃形态：与公共 UT 同名 EmptyCase，位于全局作用域）──
struct EmptyCase {
    const char* name;
    std::vector<int64_t> shape;   // 含至少一个 0 维
    int64_t dim;        // 归一化前的 dim（HandleEmptyTensor 不依赖 dim；dim 校验先于 empty）
    const char* dtype;  // 信息性：dtype 不影响 EMPTY tiling（全 0），仅覆盖框架分配空 y 的 dtype
};
class Branch2EmptyTest : public testing::TestWithParam<EmptyCase> {};
TEST_P(Branch2EmptyTest, empty_remapping)
{
    const auto& p = GetParam();
    int64_t rank = static_cast<int64_t>(p.shape.size());

    int oracleEmpty = OracleHandleEmptyTensor(rank, p.shape.data());
    ASSERT_EQ(oracleEmpty, 1) << "oracle: shape 须含 0 维";
    int actEmpty = optiling::HandleEmptyTensor(rank, p.shape.data());
    EXPECT_EQ(actEmpty, 1) << "HandleEmptyTensor 须命中任一维==0";
    EXPECT_EQ(actEmpty, oracleEmpty) << "被测与 oracle 一致";
    // ...（逐字段断言略）
}
INSTANTIATE_TEST_SUITE_P(Branch2Section1, Branch2EmptyTest, testing::Values(
    EmptyCase{"b3_fp16", {0, 4}, -1, "fp16"},
    /* ... */
));
```

```cpp
// ── after（修复形态：全类型 B2 前缀，binary 内唯一）──
struct B2EmptyCase {
    const char* name;
    std::vector<int64_t> shape;   // 含至少一个 0 维
    int64_t dim;        // 归一化前的 dim（HandleEmptyTensor 不依赖 dim；dim 校验先于 empty）
    const char* dtype;  // 信息性：dtype 不影响 EMPTY tiling（全 0），仅覆盖框架分配空 y 的 dtype
};
class Branch2EmptyTest : public testing::TestWithParam<B2EmptyCase> {};
TEST_P(Branch2EmptyTest, empty_remapping)
{
    const auto& p = GetParam();
    int64_t rank = static_cast<int64_t>(p.shape.size());

    int oracleEmpty = B2OracleHandleEmptyTensor(rank, p.shape.data());
    ASSERT_EQ(oracleEmpty, 1) << "oracle: shape 须含 0 维";
    int actEmpty = optiling::HandleEmptyTensor(rank, p.shape.data());
    EXPECT_EQ(actEmpty, 1) << "HandleEmptyTensor 须命中任一维==0";
    EXPECT_EQ(actEmpty, oracleEmpty) << "被测与 oracle 一致";

    <Op>TilingData td;
    B2FillGarbage(td);
    optiling::FillEmptyTilingData(td);
    // ... 取值表逐字段断言（对照 kB2OracleEmpty）
}
INSTANTIATE_TEST_SUITE_P(Branch2Section1, Branch2EmptyTest, testing::Values(
    B2EmptyCase{"b3_fp16", {0, 4}, -1, "fp16"},
    /* ... */
));
```

文件头注释同步写明约束（修复后新增）：

```cpp
// 注意：struct/function 命名加 B2 前缀，避免与公共 UT 的同名类型在 gtest 注册时冲突。
```

## 设计决策

- **为何选"全类型改名"而不是只改 `EmptyCase`**：注册冲突的直接源头是 `TestWithParam<EmptyCase>` 的模板实参与 `INSTANTIATE_TEST_SUITE_P` 的类型擦除注册，但 oracle 辅助类型（`OracleEmpty`/`FillGarbage`/`OracleHandleEmptyTensor`）同样跨文件同名，一次全部加前缀可彻底排除残余冲突面，成本仅为一次文件重写。
- **放弃的候选**：把新文件全部类型塞进匿名 namespace——公共 UT 侧的同名类型无法控制（不能为了新分支去改已验收的公共测试文件），且本文件的 `EmptyCase` 原本就在全局作用域；匿名 namespace 只保证内部链接，不消除 gtest 注册期的同名类型混淆面。故选择"新文件侧单方面唯一化命名"。
- **当前方案路线档位**：recommended_fastpath。

## 系统信息

- 工具链：GCC 13.3.0、googletest（`-lgtest -lgtest_main`）、CANN 9.1.0、Ascend 950 环境；算子包 UT 经 `cd operators && ./build.sh <OP>_package` 构建。
- 相关文件：`operators/<OP>_package/tests/ut/test_<op>_branch2_tiling.cpp`（新建后重写）、`tests/ut/test_<op>_tiling.cpp`（公共 UT，冲突对端，未改动）、`tests/ut/CMakeLists.txt`（UT 源清单，多文件链入单 binary）、`build/tests/ut/<op>_tiling_ut`（崩溃的产物）。

## 预防

- 在多文件 UT 工程中新增分支测试文件时，参数结构体与 oracle 辅助类型一律带分支前缀（`B2`/`B1`/`B0`…），写进文件头注释。
- UT 编译通过 ≠ 可运行：交付前至少跑一次 `--gtest_list_tests`（秒级），可在进入用例逻辑前暴露静态初始化崩溃。
- 遇到"可执行文件枚举用例即段错误"时，按"最小化渐进复现 → 单文件链接对照 → 跨文件同名 grep"三步定位，不要先猜用例逻辑。

## 验证信号

问题与验证来源：一次空 tensor 分支 TilingUT 的测试先行开发中，新增 UT 文件编入既有 gtest binary 后 `--gtest_list_tests` 段错误，经最小化隔离定位为跨文件同名参数结构体，改名后复跑。以下均为该会话实测：

- 崩溃复现：`<op>_tiling_ut --gtest_list_tests` → `Segmentation fault (core dumped)`（来源平台：950；证据等级：tool_observed）。
- 隔离链：最小文件/加全局 const/新文件单独链接三种形态 `--gtest_list_tests` 均 `RUN_EXIT=0`，仅全文件链接崩溃（来源平台：950；证据等级：tool_observed）。
- 冲突定位：grep 命中公共 UT `:215 struct EmptyCase` 与新文件 `:117 struct EmptyCase` 同名（来源平台：950；证据等级：tool_observed）。
- 修复后：`./build.sh` exit=0，UT 正常运行 `[ PASSED ] 99 tests`（公共 56 + Branch-0 26 + Branch-1 17 回归全过）+ `[ FAILED ] 15 tests`（全部为 Branch-2 红基线用例，符合测试先行预期 0/15 通过）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定"多个 UT 文件链接进同一 gtest binary"的工程形态；单文件工程不会触发。
- 崩溃对参数结构体成员形态（是否含 `std::vector`）的依赖未做消融实验：本例参数体含 `std::vector<int64_t>`，gtest 枚举输出以原始字节呈现（无 `PrintTo`）；纯 POD 参数体是否同样崩溃未验证。
- 精确到 gtest 源码层的崩溃机制（哪一处注册代码解引用了错误类型）未追；修复不依赖该机制解释。GCC/clang/MSVC 其他编译器版本未验证（推断可迁移）。

**未决项**：

- **优先项：复现并追根因到机制层**——用两个同名参数结构体的最小双文件工程在 gtest 源码级定位静态初始化崩溃点（`UnitTest::GetInstance()` 注册表或 `ParameterizedTestSuiteInfo` 的类型擦除路径），确认是 ODR 违规的直接后果还是 gtest 注册机制的未定义行为。
- 其他编译器（clang/MSVC）与纯 POD 参数体下的表现未测。
## 坏实践

- 新增分支 UT 直接复用"顺手的通用类型名"（`EmptyCase`/`FillGarbage`/`OracleXxx`）而不查同 binary 内其他测试文件——本例即因此崩溃。
- 静态初始化崩溃后先去改用例逻辑或重装环境——`--gtest_list_tests` 都崩说明用例体尚未执行，与用例逻辑无关。

## 误诊

- 机制假设（未验证，不得当实测事实）：初判怀疑"全局 `const OracleEmpty kOracleEmpty{}` 的静态初始化"，最小化加入该全局后运行正常，排除；后又假设"匿名 namespace 内同名类型的 RTTI 名混淆 / `std::bad_array_new_length`"，该假设未读 gtest 源码验证。实测事实链仅到"同名 → 崩溃、改名 → 恢复"。
