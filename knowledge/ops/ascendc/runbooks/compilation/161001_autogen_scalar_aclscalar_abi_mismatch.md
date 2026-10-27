---
type: "Runbook"
title: autogen 安装头标量属性按值声明与库 aclScalar* ABI 不符
description: "vendor 目录安装的自定义算子 aclnn 头文件是自动生成的原始类型签名（int64_t/double/bool 按值），而 libcust_opapi.so 实际导出的是源码 aclScalar* 接口；按安装头编译调用 GetWorkspaceSize 会段错误或返回 161001（DFX 显示 aclScalar UNDEFINED），须把源码 op_api 目录置于 include 优先级最前并按 aclScalar* 传参。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# autogen 安装头标量属性按值声明与库 aclScalar* ABI 不符

## 现象

两条独立轨迹、同一根因的两种表现（Ascend950PR，CANN 9.1.0）：

**表现一（段错误）**：为自定义算子（RoiAlign 类）开发 PyTorch 适配层（`torch_adapter.cpp`：`torch.ops.load_library` 注册 forward，内部做 aclnn 两段式调用），按安装目录的头文件编译：

`$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_roi_align_custom/op_api/include/aclnn_bench_no_harness_roi_align.h`（文件头注明 `this file was generated automaticlly donot change it`）声明为原始类型：

```cpp
aclnnStatus aclnnBenchNoHarnessRoiAlignGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *boxes,
    int64_t outputHeight,
    int64_t outputWidth,
    double spatialScale,
    int64_t samplingRatio,
    bool aligned,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

编译通过（直接传原始类型值，代码也更简单），但运行最简单的 1x1x2x2 用例即段错误。faulthandler 定位 + 适配层 printf 二分后拿到崩溃点：`calling GetWorkspaceSize...` 后 `Segmentation fault (core dumped)`——崩溃精确发生在 `aclnnBenchNoHarnessRoiAlignGetWorkspaceSize` 调用内。

**表现二（161001）**：融合算子（Add→RMSNorm→DynamicQuant 类）的 C++ ST Real 模式直调 `aclnnXxxGetWorkspaceSize`，编译通过但 10/10 个 L0 用例全部失败：

```
[FAIL] GetWorkspaceSize 失败: 161001
```

开 plog（ASCEND_GLOBAL_LOG_LEVEL=1）看 DFX 入参打印，epsilon 一行显示：

```
epsilon: aclScalar(data_type = UNDEFINED, value = Unknown)
```

vendor 安装头把 float 属性 `epsilon` 声明成 `double`，而源码 `op_api/aclnn_*.h/.cpp`（手写、`nm -D` 确认符号落在手写 .o）的第 4 参是 `const aclScalar* epsilon`。测试按安装头的 `double` 传参，double 位型进入被调函数后被当作 `aclScalar*` 指针解读，参数校验失败返回 161001，DFX 宏按 aclScalar 打印出 UNDEFINED。

## 适用场景

- 工程形态：自定义算子的 PyTorch 适配层 / C++ ST / 任何第三方调用方，链接 vendor 目录 `libcust_opapi.so` 并包含安装版 aclnn 头文件；手写 L2 API 带 scalar 属性参数的自定义算子联调。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>` 两段式接口；`aclCreateScalar`/`aclDestroyScalar`/`aclScalar*`；CMake `find_path`/`target_include_directories` 的头文件搜索顺序；`nm -D` 符号核对；plog DFX 入参打印。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，两条独立轨迹（torch 2.7.1 + torch_npu 适配层、C++ ST 直调）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有"手写 aclScalar* 源码接口 + 自动生成安装头"的自定义算子都存在该错位。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：算子源码头文件本身即为原始类型签名的算子（安装头与库一致，无此问题）；纯 autogen wrapper 路径（框架按 double 签名调用的场景）未验证；本卡不解释其他段错误成因（如 aclTensor 构造错、stream 误用）。

## 根因

安装到 vendor 目录的 aclnn 头文件由打包链自动生成，标量属性参数声明为原始类型（int64_t/double/bool 按值传递），而 `libcust_opapi.so` 实际导出并实现的是源码 `op_api` 头文件的 `const aclScalar*` 接口；按安装头编译的调用方把 int64/double/bool 按值压栈，实现侧却按指针解引用 `aclScalar*`——位型被解读为指针，进入 GetWorkspaceSize 即段错误（PyTorch 适配层场景）或触发参数校验失败 161001（DFX 显示 `aclScalar(data_type = UNDEFINED)`，C++ ST 场景）。C 层无重载/无签名校验，链接期不报错。

## 解决配方

修复分两步。

第一步：CMake 把**源码 op_api 目录**放在安装目录之前（本会话验证的写法）：

```cmake
# 源码 op_api 目录 (手写 aclScalar* API 头文件, 优先于安装目录中自动生成的原始类型头文件)
# 安装的头文件是自动生成的, 使用原始类型 (int64_t, double, bool);
# 但实际库导出的是 aclScalar* 接口 (与源码头文件一致)。
# C++ ST 测试也使用源码头文件, 见 tests/st/CMakeLists.txt 中 SOURCE_OP_API_DIR。
set(SOURCE_OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_bench_no_harness_roi_align.h
    PATHS
        ${SOURCE_OP_API_DIR}
        ${CANN_ROOT}/opp/vendors/${CUSTOM_VENDOR}/op_api/include
        /usr/local/Ascend/opp/vendors/${CUSTOM_VENDOR}/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/${CUSTOM_VENDOR}/op_api/include
)
```

（或直接在 `target_include_directories` 里把 `${CMAKE_CURRENT_SOURCE_DIR}/../../op_api` 排在 `CUSTOM_OP_INCLUDE_DIR` 之前。）

第二步：调用点按源码头文件的 `aclScalar*` 接口组织传参（关键片段）：

```cpp
aclScalar* CreateAclScalarInt64(int64_t v) {
    return aclCreateScalar(&v, ACL_INT64);
}

aclScalar* CreateAclScalarFloat(float v) {
    return aclCreateScalar(&v, ACL_FLOAT);
}

aclScalar* CreateAclScalarBool(bool v) {
    uint8_t b = v ? 1u : 0u;
    return aclCreateScalar(&b, ACL_BOOL);
}

    // 创建 aclScalar 属性
    aclScalar* sc_oh = CreateAclScalarInt64(outputHeight);
    aclScalar* sc_ow = CreateAclScalarInt64(outputWidth);
    aclScalar* sc_scale = CreateAclScalarFloat(static_cast<float>(spatialScale));
    aclScalar* sc_sr = CreateAclScalarInt64(samplingRatio);
    aclScalar* sc_aligned = CreateAclScalarBool(aligned);

    // 第一段: GetWorkspaceSize (使用源码头文件的 aclScalar* 接口)
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = aclnnBenchNoHarnessRoiAlignGetWorkspaceSize(
        x_acl, boxes_acl,
        sc_oh, sc_ow, sc_scale, sc_sr, sc_aligned,
        y_acl, &workspace_size, &executor);
```

float 属性以 `aclCreateScalar(&eps_value, ACL_FLOAT)` 传 ACL_FLOAT（与手写 L2 的 `epsilon->ToFloat()` 消费路径一致），而非 ACL_DOUBLE。`aclScalar*` 用完须 `aclDestroyScalar` 释放（可封装在 workspace 析构中）。

判别 / 定位方法（两条轨迹走通的链路）：

1. faulthandler + printf 二分（段错误形态）：空实现 OK → aclTensor 创建 OK → GetWorkspaceSize 崩——把段错误钉在单个 ACLNN 调用。
2. 符号核对排除链接问题：`nm -D $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so | grep <Op>`（导出 T 符号）与 `nm -D ./libtorch_adapter.so | grep <Op>`（本方 U 引用）——两边符号都在、链接正确，说明不是"找不到符号"；再对 build 目录下各 .o 反查符号归属（手写 .o vs autogen .o）。
3. 头文件比对：读安装头（`generated automaticlly`，原始类型）与源码 `op_api/aclnn_*.h`（aclScalar*）逐参对照，尤其 scalar 属性参数。
4. DFX 入参打印（161001 形态）：`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./test ... | grep "Entering function params"`——scalar 参数显示 `aclScalar(data_type = UNDEFINED)` 即签名错配的典型信号；修复后应显示 `aclScalar(data_type = DT_FLOAT, value = 0.000001)`。
5. 修复后错误码应"前进"（161001 → 后续校验的 161002），以此确认参数已正确送达（若出现 161002 属另一独立校验问题，另查）。
6. 交叉证据：同一环境下 C++ ST 测试（用源码头文件 + aclScalar* 接口编译）运行正常 → 安装头与库的签名错位。

## 设计决策

- 为什么优先源码头而不是改安装头：安装头是打包自动生成的（文件内明确注明 donot change it），改动会被下次安装覆盖；源码头是接口的事实源（库就是按它编的），C++ ST 测试同此做法。
- 为什么用 `nm -D` 而不是反复猜参数：符号导出/引用核对一步排除"链接错库/符号缺失"这类更常见嫌疑，把问题收敛到 ABI 形态。
- `spatialScale` 以 ACL_FLOAT 而非 ACL_DOUBLE 传：源码头接口按 float 语义消费；同时避免 PyTorch 侧 double 标量触发的 `Device do not support double dtype` 隐式转换警告链。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；Python 3.13 + torch 2.7.1+cpu + torch_npu（适配层场景），g++ 13.3.0。
- 相关文件：`tests/st/torch/CMakeLists.txt` 或 `tests/st/CMakeLists.txt`（include 顺序）、`torch_adapter.cpp` / `test_aclnn_*.cpp`（调用方）、安装头 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include/aclnn_*.h`、源码头 `op_api/aclnn_*.h`、库 `.../op_api/lib/libcust_opapi.so`、`build/autogen/aclnn_*.h/.cpp`（autogen wrapper）。

## 预防

- 装完 custom_opp 包后，直调联调前先 diff 一下 vendor 安装头与源码头的函数签名（尤其 scalar 属性参数）。
- ST/适配层的 include 顺序把源码 op_api 目录放在 CUSTOM_OP_INCLUDE_DIR（安装头）之前。
- 看到 DFX 打印 scalar 参数 `data_type = UNDEFINED` 直接怀疑 ABI 错配，别先去查数据本身。
- "见安装头能编译通过就默认它是接口事实源"——自动生成头可能与库 ABI 错位，编译通过不代表调用约定正确。

## 验证信号

来源说明：本结论来自两条独立轨迹的联调会话（PyTorch 适配层段错误定位、C++ ST 161001 定位），定位与修复均在同一环境完成。

- 表现一修复前：faulthandler 显示 `calling GetWorkspaceSize...` 后 `Segmentation fault (core dumped)`；`nm -D` 证实 `libcust_opapi.so` 导出 `aclnnBenchNoHarnessRoiAlign(GetWorkspaceSize)`（T 符号）且适配层为 U 引用（来源平台：950；证据等级：tool_observed）。
- 安装头与源码头签名差异：安装头为 `int64_t/double/bool`（自动生成），源码头为 `const aclScalar*`（两个文件内容已对照）（来源平台：950；证据等级：tool_observed）。
- 表现一修复后：切换源码头 + aclScalar* 接口编译成功，1x1x2x2 最小用例输出 2.5 与手算一致，36 条用例正常执行（后续全量 36/36）（来源平台：950；证据等级：tool_observed）。
- 表现二修复前：按安装头（double）传参，10/10 L0 用例 GetWorkspaceSize 返回 161001，DFX 显示 `epsilon: aclScalar(data_type = UNDEFINED)`（来源平台：950；证据等级：tool_observed）。
- 表现二符号核对：`nm -D` 确认 `libcust_opapi.so` 导出 `aclnnBenchNoHarnessAddRmsNormDynamicQuantGetWorkspaceSize`（T），符号来自手写 `op_api/aclnn_*.cpp.o`（来源平台：950；证据等级：tool_observed）。
- 表现二修复后：include 顺序 + aclScalar* 修复后重建重跑，错误码从 161001 变为 161002（进入 CheckDtypeValid 的后续校验），DFX 显示 `epsilon: aclScalar(data_type = DT_FLOAT, value = 0.000001)`——参数已按手写签名正确送达（161002 为另一独立缺陷，不属本卡范围）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该错位形态取决于"打包链自动生成安装头"的行为；若某版本打包链生成的安装头与源码一致，本卡不适用。CANN 9.1.0 验证，其他版本未验证。
- 只证明了"按安装头原始类型调用会段错误/161001、按源码 aclScalar* 调用正常"；未逐参数验证错误签名的具体崩溃路径（属实现内部解引用，未反汇编）。
- gdb 在该环境不可用（ptrace 受限），深栈定位依赖 faulthandler + printf 二分——方法本身可复用。
- autogen wrapper 与手写实现共存于同一 .so 时的符号选择机制未完全澄清（以 nm 结果为准）。
- 未验证，推断可迁移：CANN 9.x autogen 头对 float 属性一律生成 double 形参。

**未决项**：

- 未验证打包链为何生成与库 ABI 不一致的安装头（生成器取的是哪份签名、是否算缺陷），可在独立环境向工具链侧确认。
- 161001 在库内部命中的具体 NULL 检查（CheckNotNull(epsilon) 还是框架断言）未定位到代码行——如需机制层确认，可在手写 L2 的 CheckNotNull 分支加日志复现（优先级低）。
## 坏实践

- 见"安装头能编译通过"就默认它是接口事实源——自动生成头可能与库 ABI 错位，编译通过不代表调用约定正确。
- 段错误后先怀疑自己的 aclTensor 构造/内存管理（表现一轨迹先花数轮验证 CreateAclTensor 与 C++ ST 一致）——应尽早做"符号在 + 另一头文件可跑通"的交叉对照。
- 发现 161001 后先怀疑 `libcust_opapi.so` 未定义符号（l0op::Contiguous 等）导致库加载失败——这些符号由测试可执行文件链接的 `libopapi.so` 在运行时懒绑定解析，安装包 `--force` 跳过校验后并不阻止库工作，与本错误无关（实测排除）。

## 误诊

- 表现一早期认为"安装头是官方给的、按它写最简单"（原始类型确实少写 aclScalar 样板）：正是这个假设引入段错误；C++ ST 用源码头能跑是反证。
- 表现一轨迹中 torch_npu 的 `Device do not support double dtype now` 警告一度像嫌疑点：它只是 dtype 隐式转换提示，与崩溃无关（修复后仍会出现）。
- 表现二曾假设 161001 由 `libcust_opapi.so` 的 `l0op::Contiguous`/`l0op::ViewCopy` 未定义符号导致（nm 看到未定义符号 + 安装时校验失败的记忆），沿该方向排查了 NnopbaseGetExecutor/ASCEND_OPP_PATH/环境变量多轮；实际根因是头/库签名错配（该排除推理最终由错误码随签名修复而转变证伪）。

## 后续增量（2026-09-17，并卡合并：类型映射规则、对侧修复路线与 float/double 宽度错配形态）

并入同根因家族的两张卡（int_attr_header_scalar_abi_mismatch、aclnn_scalar_attr_abi_mismatch_segfault，均为 Ascend950PR + CANN 9.1.0）。本卡主文的修复方向是**调用方适配源码头**（include 顺序把源码 op_api 排最前 + 按 aclScalar* 传参，适用于"库由源码编译、符号以源码为准"的场景）；并入的两条 trace 走**对侧路线：源码迁就生成头**（把 op_api 源码 L2/L0 签名改为直接标量后重编重装）。两条路线都能闭合"头/库签名不一致"，取舍见下。

#### 并入自 aclnn_scalar_attr_abi_mismatch_segfault（Attr 类型→生成头标量映射规则与源码侧对齐路线）

- **类型映射规则（构建系统按 op_def 的 Attr 声明类型生成安装头与对外 ABI）**：`Float()`→`double`、`Int()`→`int64_t`、`Bool()`→`bool` 的直接标量形参（按值传递）；op_graph 侧 `.ATTR(epsilon, Float, 1e-5)` 同样生成 `double epsilon`。
- **修复路线（源码向生成头对齐，AdamW 类算子，7 个标量属性）**：L2 头/实现、L0 头/实现**四个文件**的形参全部改为与生成头一致的直接标量——只改 L2 不改 L0 会让两层签名再次不一致，问题只是换个位置。参数检查中相应去掉对标量形参的 `OP_CHECK_NULL`；L0 实现直接把标量交给 `OP_ATTR(lr, beta1, ...)`。
- **判别方法（可复用）**：崩溃特征 `aclnn...GetWorkspaceSize` 调用前后无任何日志输出、exit 139、ACL 初始化日志完好——优先怀疑 ABI/指针而非业务逻辑；`cat` 安装头拿生成签名与 op_api 源码形参逐一对照；**参照一个能跑通的同类算子确认正确模式**（本会话比对了同为标量属性的 masked_scale——op_def `this->Attr("scale").Float(1.0f)`，其 aclnn 源码直接 `double scale` 形参）；重编重装后 `cat` 安装头确认与源码签名一致。
- **相邻独立问题预警（并入）**：直接标量化后 `OP_ATTR(lr, ...)` 拿到的是 `double`，与 op_def `Float()` 的 4 字节声明存在类型宽度差——该隐患会以 tiling 属性垃圾值的形式爆发（输出数值错误约 4.63e16 倍偏差），须在 OP_ATTR 处补 `static_cast<float>` 收窄；属相邻独立问题，不在本卡范围。
- 预防补充（并入）：手写 op_api 前先 `cat` 一次生成头，按生成头签名写源码；标量属性一律直接标量（double/int64_t/bool），不要用 aclScalar*。extern "C" 接口没有链接期类型检查：任何"编译链接都过、一调用就崩"的 aclnn 直调问题，第一反应去比对声明与导出签名。

验证信号（并入，AdamW 类算子，Ascend950PR + CANN 9.1.0）：

- 修复前：ST 程序运行即 `Segmentation fault (core dumped)`，EXIT_CODE=139；日志停在 ACL 初始化与首个用例 buffer 分配之后、无任何用例输出（来源平台：950；证据等级：tool_observed）。
- 参照证据：安装生成头签名为 `double lr ... int64_t step, bool maximize`，源码 L2/L0 头与实现均为 `aclScalar*` 形参（来源平台：950；证据等级：tool_observed）。
- 修复后（四文件改直接标量、重编重装、安装头与源码签名一致）：段错误消失，ST 12 个用例全部执行并产出数值结果（数值错误另有根因，见宽度差预警）（来源平台：950；证据等级：tool_observed）。

#### 并入自 aclnn_scalar_attr_abi_mismatch_segfault（并行策展批：float vs double 按值宽度错配——值错读而非段错误）

**第三种形态**：主卡与本卡前文是**按值 vs 指针**（aclScalar*）错配（段错误/堆破坏），本形态是**按值宽度**错配（float vs double，均为裸标量）——不崩溃，而是参数被错位读成垃圾值后在参数校验处以 161002 拒绝。

- **现象（GroupNorm 算子）**：ST 41 个用例全部 `GetWorkspaceSize 失败: 161002`，OP 日志 `epsilon=-0.000000 not in (0, 1]`——CSV 用例里 epsilon 全是正数（1e-05/1e-08/1.0），DFX 打印却是 `-0.000000`；测试按头文件签名传参、参数顺序逐一核对无误。
- **根因（位级机制）**：`.ATTR(epsilon, Float, 1e-5)` 使安装头生成 `double epsilon`（8 字节压栈），手写源码头/实现用 `float epsilon`（4 字节读参）。以 epsilon=1e-5 为例：调用方压栈 `double(1e-5)`（位模式 `0x3EE4F8B588E368F1`），被调方按 `float` 只读低 4 字节 `0x88E368F1`——符号位为 1 的极小负数（约 -1e-33），`%f` 打印即 `-0.000000`；`epsilon > 0 && epsilon <= 1` 校验失败返回 161002。
- **修复（与源码对齐路线同向）**：L2 头/实现、L0 头/实现全部改为 `double epsilon`，与自动生成安装头对齐（`const aclTensor* out` 与 `aclTensor* y` 的 const/命名差异不影响 C ABI——指针参数顶层 const 不参与 name mangling/布局，只修 float→double）。
- **判别探针**：

```bash
# 1. 对比源码头与安装头的签名（关键看标量属性类型）
diff <(grep -A8 "GetWorkspaceSize" op_api/aclnn_<op>.h) \
     <(grep -A8 "GetWorkspaceSize" \
       $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include/aclnn_<op>.h)
# 2. 查 op_graph 属性定义
grep -n "ATTR(epsilon" op_graph/*.h   # .ATTR(epsilon, Float, 1e-5) → 生成头为 double
```

- **预防（本形态专属）**：带标量 Float 属性的算子，手写 aclnn 头一律写 `double`，与 `.ATTR(name, Float)` 的自动生成口径对齐；联调遇到"传入正数、日志打印 -0.000000/垃圾值"时，第一时间 diff 源码头与安装头，而不是反复核对测试侧传参顺序。

验证信号（并入，GroupNorm ST 联调，Ascend950PR + CANN 9.1.0）：

- 修复前：41/41 用例 161002，日志 `epsilon=-0.000000 not in (0, 1]`（来源平台：950；证据等级：tool_observed）。
- 修复前 diff：安装头为 `double epsilon`、源码头为 `float epsilon`（安装头文件头注释 "generated automaticlly"）（来源平台：950；证据等级：tool_observed）。
- 修复后（全链路改 double、重编重装）：161002 参数校验失败全部消失，用例进入执行阶段（通过数从 4 升至 21；剩余失败为独立根因）（来源平台：950；证据等级：tool_observed）。

#### 并入自 int_attr_header_scalar_abi_mismatch（Int 属性实例、编译期同源信号与打点夹逼）

- **Int 属性实例（Cummin 算子）**：算子定义 `this->Attr("dim").Int()` 使安装头生成 `int64_t dim`，op_api 手写源码/本地源码头用 `const aclScalar* dim`（内部 `dim->ToInt64()`）。ST Real 模式首用例（dtype=float16 shape=(1557) dim=-1）在 `aclnnBenchNoHarnessCumminGetWorkspaceSize(inT, dimVal, valT, idxT, ...)` 调用内部段错误（`[DBG] before GetWorkspaceSize dimVal=-1` 打印后、`[DBG] after GetWorkspaceSize` 之前），`timeout: the monitored command dumped core`。
- **编译期同源信号（另一面）**：此前测试按旧写法构造 `aclScalar* dimS = aclCreateScalar(&dimVal, ACL_INT64)` 传入时，编译报 `error: invalid conversion from 'aclScalar*' to 'int64_t' {aka 'long int'} [-fpermissive]`——即同一错配在编译期的暴露形态。
- **修复（以生成头口径为准，Int 属性按值）**：op_api 源码与本地源码头的签名统一改为 `int64_t dim`——`CheckNotNull` 去掉 dim 形参、`CheckDim` 改按值、`CheckParams` 透传、L2 入口签名与 DFX 宏同步；重编重装后确认 vendor 安装头为 `int64_t dim`，首用例 `[DBG] after GetWorkspaceSize ret=0 wsSize=512`，段错误消失。
- **调试路径取舍（并入）**：gdb 受 ptrace 限制、`ASCEND_GLOBAL_LOG_LEVEL` 无输出时，`stdbuf -oL -eL` + 逐步 fprintf 打点是最有效的夹逼手段。
- 预防补充（并入）：自定义算子带 Int 标量属性时，op_api 手写签名必须与 `Attr(...).Int()` 的生成头口径（`int64_t` 按值）核对，最简单的自检是拿安装头与源码头做一次 diff；ST 编译期出现 `invalid conversion from 'aclScalar*' to 'int64_t'` 这类"测试与安装头类型不符"错误时，先怀疑 op_api 源码与生成头错配，而不是只改测试。
- 坏实践补充（并入）：测试代码与安装头签名不一致时只改测试侧凑编译（把 aclScalar 改成裸传 int64），不核对 op_api 源码实现的真实签名——编译过了，运行时按错误 ABI 解引用直接段错误。
- 误诊补充（并入）：最初把段错误怀疑为 tiling 函数对 1D shape=(1557)、dim=-1 的处理问题（逐行排查 tiling.cpp）；打点日志证明崩溃发生在进入 tiling 之前的参数解引用处。

验证信号（并入，Cummin，Ascend950PR + CANN 9.1.0）：

- 修复前：`[DBG] before GetWorkspaceSize dimVal=-1` 打印后 `timeout: the monitored command dumped core`，崩溃点在 GetWorkspaceSize 内部；编译期同源信号 `test_aclnn_bench_no_harness_cummin.cpp:950:63: error: invalid conversion from 'aclScalar*' to 'int64_t'`（来源平台：950；证据等级：tool_observed）。
- 修复后：重装头确认 `int64_t dim,`，Real 模式首用例 `[DBG] after GetWorkspaceSize ret=0 wsSize=512`，后续 `aclnnBenchNoHarnessCummin` ret=0、`SyncStream` ret=0，段错误消失（来源平台：950；证据等级：tool_observed）。

#### 并入后路线谱系与家族索引

- **两条闭环路线**：(A) 调用方适配源码头（本卡主文：源码 op_api 目录排 include 首位 + aclScalar* 传参，库由源码编译导出时适用）；(B) 源码迁就生成头（并入的 AdamW/Cummin/GroupNorm 路线：L2/L0 源码全链改直接标量/正确宽度后重编重装，安装头是随包发布的外部契约时适用）。共同前提：先 `nm -D` 确认库的真实导出签名，再决定改哪一侧。
- 同家族的 autogen `aclTensor*` 变体（安装头把 scalar 生成为 `const aclTensor*`、类型混淆致跨用例堆污染 dtype 43）是**另一独立现象卡**：见 [安装头 scalar 类型与源码不符致堆破坏 161002 卡](161002_autogen_header_scalar_type_mismatch.md)；属性**个数**分叉（aclnn_exclude 路径多出属性参数）与属性类型改动只改 _def.cpp 的分叉见 [aclnn_exclude 头文件自动生成覆盖源码致签名不符卡](aclnn_exclude.md)。