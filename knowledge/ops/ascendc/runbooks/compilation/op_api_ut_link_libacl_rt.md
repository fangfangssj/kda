---
type: "Runbook"
title: "op_api UT 引入 runtime API 须补 libacl_rt"
description: "op_api 层引入 aclrtMemcpy 等_runtime API 后 op_api UT 链接报 undefined reference to symbol 'aclrtMemcpy' / libacl_rt.so: DSO missing from command line；在 UT 的 CMakeLists 链接清单补 $ASCEND_HOME_PATH/lib64/libacl_rt.so 即可。"
tags: ["task.testing", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# op_api UT 引入 runtime API 须补 libacl_rt

## 现象

在 Ascend950PR（CANN 9.1.0）算子仓里给 `op_api/aclnn_<op>.cpp` 新增了调用 `aclrtMemcpy` 的参数校验代码（该文件被 op_api UT 直接编入），随后构建 UT：

```bash
cd tests/ut/build && cmake .. && make -j$(nproc) <op>_op_api_ut
```

链接阶段失败，报错原文：

```text
[ 84%] Linking CXX executable <op>_op_api_ut
/usr/bin/ld: CMakeFiles/<op>_op_api_ut.dir/.../op_api/aclnn_<op>.cpp.o: undefined reference to symbol 'aclrtMemcpy'
/usr/bin/ld: <CANN>/lib64/libacl_rt.so: error adding symbols: DSO missing from command line
collect2: error: ld returned 1 exit status
```

报错本身已经指认了符号所在库（libacl_rt.so），但该库不在 UT 的链接清单里——这份 UT 的 CMakeLists 此前只链接了 metadef/注册表/图等 CANN 库（op_api 层元数据操作够用），没有任何 runtime API 调用，所以从没需要过 libacl_rt.so。

## 适用场景

- 工程形态：算子仓 op_api UT（把 op_api 源码编进 UT 可执行文件）。
- API 路径：`tests/ut/op_api/CMakeLists.txt` 链接库清单 + `aclrtMemcpy` 等 `acl/acl_rt.h` runtime API。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有把 op_api 源码编入 UT 且首次引入 aclrt* runtime 调用的算子仓。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：UT 用 dlopen/mocks 隔离 runtime 符号的构建方式；其他 CANN 版本的库名布局未验证。

## 根因

`aclrtMemcpy` 属于 CANN runtime 库 `libacl_rt.so`，而 op_api UT 的 CMakeLists 链接清单只含 metadef 系（libmetadef.so、libregister.so、libgraph.so 等）；op_api 层代码一旦调用 runtime API，链接器在新式默认（`--as-needed` + DSO 校验）下直接报 `DSO missing from command line`。

## 解决配方

在 `tests/ut/op_api/CMakeLists.txt` 的 `target_link_libraries` 清单末尾追加 runtime 库（before → after）：

```cmake
# before
    # CANN metadef libraries (required for op_api layer)
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so

# after
    # CANN metadef libraries (required for op_api layer)
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so

    # CANN runtime library (required for aclrtMemcpy in CheckDataValid)
    $ENV{ASCEND_HOME_PATH}/lib64/libacl_rt.so
```

重新 `cmake .. && make` 后链接通过。

## 验证信号

来源说明：该算子 op_api 数据校验补齐的 UT 构建会话（Ascend950PR + CANN 9.1.0）。

- 失败态：`undefined reference to symbol 'aclrtMemcpy'` + `libacl_rt.so: error adding symbols: DSO missing from command line`，链接退出码 1（来源平台：950；证据等级：tool_observed）。
- 修复态：CMakeLists 补库后重新配置构建，`[100%] Built target <op>_op_api_ut`，UT 随后可运行（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 库名/路径以 `$ASCEND_HOME_PATH/lib64` 布局为准；非该布局的安装（如 /usr/local/Ascend 工具包前缀）需相应调整，未验证。
- 只覆盖 `acl_rt` 系；引入其他 runtime 域 API（如 aclrtMalloc 系同库，msprof 系不同库）时按报错指认的库逐个补。

**未决项**：

- 无。
## 坏实践

- 见 `DSO missing from command line` 就去改 `LD_LIBRARY_PATH`——那是运行期搜索路径，解决不了链接期缺库。

## 误诊

- 无。
