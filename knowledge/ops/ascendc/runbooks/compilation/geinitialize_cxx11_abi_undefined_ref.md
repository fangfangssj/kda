---
type: "Runbook"
title: GEInitialize 新 ABI 重载未导出改 AscendString
description: "GE IR 示例用 std::map<std::string,std::string> 调 GEInitialize/ge::Session 构造，GCC13 默认新 CXX11 ABI 实例化的弃用重载在 libge_runner.so 未导出而链接失败；nm -D 实证只导出 AscendString 版与旧 ABI（Ss）版，改 std::map<ge::AscendString, ge::AscendString> 后链接通过。"
tags: [task.implementation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-17T04:13:22Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# GEInitialize 新 ABI 重载未导出改 AscendString

## 现象

为已部署的自定义 LSTM 算子开发 GE IR 调用示例 `examples/test_geir_<op>.cpp`：用 `ge::op::Data` 构造输入节点、用算子 proto 头生成的 `ge::op::<Op>` 类构图，再 `ge::GEInitialize(options)` + `ge::Session session(sessionOptions)` + `AddGraph/RunGraph` 执行。示例按公开样例的常见写法用 `std::map<std::string, std::string>` 传 GE 选项。CMake 配置成功后首次 `make -j$(nproc)`，GE IR 目标经历三连链接错误。

第一处（报错已点名库，补链即解）：

```text
/usr/bin/ld: .../test_geir_<op>.cpp.o: undefined reference to symbol '_ZN2ge12AscendStringC1EPKc'
/usr/bin/ld: <CANN>/lib64/libgraph.so: error adding symbols: DSO missing from command line
```

补 `libgraph.so` 后第二处（同样报错点名库）：

```text
/usr/bin/ld: .../test_geir_<op>.cpp.o: undefined reference to symbol '_ZN2ge8Operator16UpdateOutputDescEPKcRKNS_10TensorDescE'
/usr/bin/ld: <CANN>/lib64/libgraph_base.so: error adding symbols: DSO missing from command line
```

补 `libgraph_base.so` 后第三处——报错不再点名任何库，问题真正暴露：

```text
/usr/bin/ld: .../test_geir_<op>.cpp.o: in function `main':
test_geir_<op>.cpp:(.text.startup+0x2f363): undefined reference to `ge::GEInitialize(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::less<...>, std::allocator<...> > const&)'
/usr/bin/ld: test_geir_<op>.cpp:(.text.startup+0x30585): undefined reference to `ge::Session::Session(std::map<std::__cxx11::basic_string<char, ...> > const&)'
collect2: error: ld returned 1 exit status
```

当时代码状态：`ge_api.h` 里 `GEInitialize(const std::map<std::string, std::string> &)` 与对应 `Session` 构造确实有声明（带 `ATTRIBUTED_DEPRECATED` 标记，提示改用 `AscendString` 版本），编译期只出 deprecation 警告，链接期才炸。

## 适用场景

- 工程形态：自定义算子的 GE IR（`ge::Graph`/`ge::Session`/`AddGraph/RunGraph`）C++ 调用示例或 host 侧程序，GCC 13（默认 `_GLIBCXX_USE_CXX11_ABI=1`）编译。
- API 路径：`ge::GEInitialize` / `ge::Session::Session`（`ge/ge_api.h`）；`ge::AscendString`；`nm -D` 符号导出探查。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（GCC 13.3.0，x86_64）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN GE 库符号导出行为，与算子无关。
- 路线优先级：recommended_fastpath（改用非弃用的 AscendString 重载是官方指引方向）。
- 明确不适用 / 未验证场景：`-D_GLIBCXX_USE_CXX11_ABI=0` 全工程降 ABI 的替代路线未验证（会牵连其他库的链接一致性，不推荐）；aarch64 未实测（未验证，推断可迁移）。

## 根因

CANN 9.2.0 的 `libge_runner.so` 没有导出 `std::map<std::__cxx11::string, std::__cxx11::string>`（新 CXX11 ABI）版本的 `GEInitialize`/`Session` 构造重载：`nm -D` 探针显示它只导出 `AscendString` 版本和旧 ABI（mangle 缩写 `Ss` = pre-C++11 `std::string`）版本，而 GCC 13 默认按新 ABI 实例化弃用的 `std::map<std::string, ...>` 重载，导致链接期找不到符号。

## 判别方法

对 CANN 的 GE 库做符号导出探查（本例实际执行的命令与输出）：

```text
cd <CANN>/x86_64-linux/lib64 && for lib in libge_runner.so libge_compiler.so libge_common.so libge_common_base.so; do
    echo "=== $lib ==="; nm -D $lib 2>/dev/null | grep "GEInitialize" | head -3
done

=== libge_runner.so ===
000000000007fa70 T _ZN2ge12GEInitializeERKSt3mapINS_12AscendStringES1_St4lessIS1_ESaISt4pairIKS1_S1_EEE
0000000000083500 T _ZN2ge12GEInitializeERKSt3mapISsSsSt4lessISsESaISt4pairIKSsSsEEE
                 U _ZN2ge14GEInitializeV2ERKSt3mapINS_12AscendStringES1_St4lessIS1_ESaISt4pairIKS1_S1_EEE
=== libge_compiler.so ===
=== libge_common.so ===
=== libge_common_base.so ===
```

判读：`ISsSs` 是旧 ABI `std::string`（`Ss` 缩写），`INS_12AscendStringES1_` 是 AscendString 版；没有任何 `INSt7__cxx1112basic_string...`（新 ABI）导出。链接器报错里要找的正是新 ABI mangle，因此必然 undefined。

## 解决配方

把 GE 选项与 Session 选项从 `std::map<std::string, std::string>` 换成 `std::map<ge::AscendString, ge::AscendString>`，命中 libge_runner.so 实际导出的非弃用重载。

修复前（GE 初始化段，`test_geir_<op>.cpp`）：

```cpp
    std::map<std::string, std::string> geOptions;
    geOptions["ge.device"] = std::to_string(deviceId);
    geOptions["ge.runMode"] = "0";
    geOptions["ge.graphRunMode"] = "0";
    auto geRet = ge::GEInitialize(geOptions);
```

修复后：

```cpp
    std::map<ge::AscendString, ge::AscendString> geOptions;
    geOptions[ge::AscendString("ge.device")] = ge::AscendString(std::to_string(deviceId).c_str());
    geOptions[ge::AscendString("ge.runMode")] = ge::AscendString("0");
    geOptions[ge::AscendString("ge.graphRunMode")] = ge::AscendString("0");
    auto geRet = ge::GEInitialize(geOptions);
```

修复前（Session 创建段）：

```cpp
    std::map<std::string, std::string> sessionOptions;
    sessionOptions["ge.device"] = std::to_string(deviceId);
    ge::Session session(sessionOptions);
```

修复后：

```cpp
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    sessionOptions[ge::AscendString("ge.device")] = ge::AscendString(std::to_string(deviceId).c_str());
    ge::Session session(sessionOptions);
```

注意：`AscendString` 只接受 `const char*` 构造，不能从 `std::to_string(...)` 返回的 `std::string` 临时对象隐式转换——动态拼出的值先落局部 `std::string` 再 `.c_str()` 构造，也不要把 `std::to_string(...)` 直接放进 map 初始化列表。

## 设计决策

- 为什么选 AscendString 重载而不是 `-D_GLIBCXX_USE_CXX11_ABI=0`：头文件 deprecation 注记本来就指引迁移到 `AscendString` 版本；全局降 ABI 会影响工程内其他 C++ 库的链接一致性，收益为负。`AscendString` 版在 `libge_runner.so` 有 `T`（导出）符号，链接稳定。
- 前两处 `libgraph.so`（`AscendString` 构造符号）与 `libgraph_base.so`（`Operator::UpdateOutputDesc`）的 DSO missing 是同轮构建的前置错误：报错原文已点名库，直接补进 `GE_LIBS` 即可，不要与 ABI 问题混为一谈。
- 当前路线属 recommended_fastpath（对齐官方弃用指引）。

## 预防

- 在 GCC 13（新 CXX11 ABI 默认）环境写 GE host 程序时，GE 选项 map 一律直接用 `std::map<ge::AscendString, ge::AscendString>`，不抄旧样例的 `std::map<std::string, std::string>` 写法——编译期能过（仅有 deprecation 警告），链接期才炸，浪费一轮定位。
- 见到 `undefined reference` 且 mangle 中含 `__cxx11`，先用 `nm -D` 对目标库做导出探查，确认是"符号未导出"还是"DSO 缺链"（后者报错会点名库文件）。

## 验证信号

来源说明：本验证来自该 LSTM 自定义算子 GE IR 示例的构建修复——同一工程同一命令行，仅改选项 map 类型前后对照。

- 修复前：`make -j$(nproc)` 链接失败，`undefined reference to ge::GEInitialize(std::map<std::__cxx11::basic_string...>)` 与 `ge::Session::Session(std::map<std::__cxx11::basic_string...>)`（来源平台：950；证据等级：tool_observed）。
- 探针：`nm -D libge_runner.so | grep GEInitialize` 输出如上，只含 AscendString 版与旧 ABI（`Ss`）版导出（来源平台：950；证据等级：tool_observed）。
- 修复后：`[100%] Linking CXX executable test_geir_<op>`、`[100%] Built target test_geir_<op>`，同轮 ACLNN 示例目标同样构建成功（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只解决链接期 ABI 重载缺失；GE IR 示例随后的运行期问题（图执行失败等）是独立根因，不在本卡范围。
- `aclgrphBuildInitialize`/`aclgrphBuildModel`（`ge_ir_build.h`）同时提供 `std::map<std::string,std::string>` 与 `AscendString` 两个重载声明，本轨迹对后者（`AscendString`）实测链接通过；前者在新 ABI 下的导出情况未逐一探查。
- CANN 9.2.0 之外的版本符号导出可能变化（未验证，推断可迁移）。
- `ge::Session` 其余构造/成员重载在新 ABI 下的导出情况未逐一盘点；后续示例用到更多 `std::string` 签名的 GE API 时仍需 `nm -D` 复核。

**未决项**：

- 未逐一盘点 GE API 中其余 `std::string` 签名重载在新 ABI 下的导出情况；如后续示例链接报同类 `__cxx11` mangle 未定义引用，按本卡 `nm -D` 探针复用排查。
## 坏实践

- 照抄旧样例里 `std::map<std::string, std::string>` + `GEInitialize` 的写法而不看 deprecation 警告：编译期能过，链接期才炸，浪费一轮定位。
- 把 `AscendString` 构造符号的 DSO missing（缺 `libgraph.so`）误判为 ABI 问题去加编译宏：两类错误报错形态不同（前者点名 DSO，后者点名未导出的 mangle 符号），应分别处理。

## 误诊

- 无（第三处报错后直接用 `nm -D` 实证了导出符号表，未走弯路；前两处 DSO missing 按报错补链即解）。
