---
type: "Runbook"
title: op_host 直连 libruntime 内部符号致安装拒装
description: "自定义算子 tiling 代码 extern \"C\" 直接声明并调用 libruntime 内部符号 rtSetDevice/rtMemcpy，构建可过但算子包安装校验失败 libcust_opsproto_rt2.0.so: undefined symbol: rtSetDevice；须改用框架机制（HostInputs）。"
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# op_host 直连 libruntime 内部符号致安装拒装

## 现象

在自定义算子的 tiling 函数（`op_host/arch35/*_tiling.cpp`）里想直接做一次 Device→Host 拷贝，于是在文件里 `extern "C"` 声明 libruntime 内部符号并直接调用：

```cpp
    // rtMemcpy 来自 libruntime.so (tiling .so 已链接)
    extern "C" int32_t rtSetDevice(int32_t deviceId);
    extern "C" int32_t rtMemcpy(void* dst, size_t destMax, const void* src, size_t count, int32_t kind);

    ...
        int32_t devId = 0;
        extern "C" int32_t rtGetDevice(int32_t* deviceId);
        if (rtGetDevice(&devId) != 0) {
            rtSetDevice(0);
        }
        int32_t ret = rtMemcpy(&scalarVal, sizeof(float), scalarAddr, sizeof(float), 2);
```

`bash build.sh --soc=ascend950` 构建通过（符号在编译期只是声明），但安装自定义算子包时报错并**拒绝安装**：

```text
[ops_custom] [ERROR] Failed to load libcust_opsproto_rt2.0.so wish env ASCEND_HOME_PATH(<cann-9.1.0 安装根>),error message is libcust_opsproto_rt2.0.so: undefined symbol: rtSetDevice.
[ops_custom] [ERROR] Shared library validation failed. To skip this validation and continue installation, run this package again with --force.
```

## 适用场景

- 工程形态：AscendC 自定义算子 op_host（tiling/proto）代码想调用 Runtime 内部 C 接口（rtMemcpy/rtSetDevice/rtGetDevice 等）。
- API 路径：`IMPL_OP_OPTILING` 产物 `libcust_opsproto_rt2.0.so`/`libcust_opmaster_rt2.0.so`；`libruntime.so` 内部符号（不在任何公开头文件声明，`nm -D libruntime.so` 可见 `T rtMemcpy`/`T rtSetDevice`）。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.1.0。
- 适用粒度：`platform_wide`（自定义算子 .so 的符号解析约束与算子无关）。
- 路线优先级：`banned`（直接 extern 声明 + 静态链接期待这条路被安装校验封死；为负面结论卡）。
- 明确不适用/未验证场景：op_kernel（设备侧）代码不适用本卡；`libascendcl.so` 的公开 ACL 接口（如 aclrtMemcpy）在算子包外的可执行程序里正常可用。

## 根因

自定义算子的 op_proto/op_host 产物 `.so` 在安装时要做动态加载校验；这些库**不链接** `libruntime.so`，对 `rtSetDevice`/`rtMemcpy` 这类 libruntime 内部符号的未定义引用在加载校验时直接判失败——构建期能过只是因为 `extern "C"` 声明不产生链接依赖。

## 判别方法

- 报错特征：`Failed to load libcust_opsproto_rt2.0.so ... undefined symbol: rtSetDevice`，且安装器提示只有 `--force` 才能跳过。
- 确认符号确实存在于 libruntime 但无公开头文件声明：`nm -D $ASCEND_HOME_PATH/lib64/libruntime.so | grep -E "rtMemcpy|rtSetDevice"`（有 `T` 定义），`grep -rn "rtMemcpy" $ASCEND_HOME_PATH/x86_64-linux/include/`（无公开声明）。

## 失效边界（替代做法）

- 若确需在 tiling 里调用这类内部接口：用 `dlsym(RTLD_DEFAULT, "rtMemcpy")` 运行时解析（加载进程已装入 libruntime 时可解析到），不要产生静态未定义引用——但注意该 dlsym 方案在本场景也未经受验证（见下）。
- 更优先：改用框架机制让数据到 host（例如把输入注册为 HostInputs，由框架完成 D2H），完全不在 tiling 里手工搬数据。
- 注意：同次开发会话还观测到 tiling 内 `dlsym("aclrtMemcpy")` 的 D2H 调用会挂起（60s 超时无输出），机制未定位——dlsym 方案在本场景也未经受验证，不能当作已验证的替代路线推荐。

## 验证信号

来自该算子联调会话的实测：

- `extern "C"` 直连 rtSetDevice/rtGetDevice/rtMemcpy：构建成功，安装 `custom_opp_ubuntu_x86_64.run` 时校验失败，报 `libcust_opsproto_rt2.0.so: undefined symbol: rtSetDevice`，包被拒装（来源平台：950；证据等级：tool_observed）。
- 同批探针：`nm -D libruntime.so` 确认 `rtSetDevice`/`rtMemcpy`/`rtGetDevice` 均为导出符号；`tiling .so`（libcust_opmaster_rt2.0.so）的 ldd 依赖里有 libruntime/libruntime_common 但无 libascendcl（来源平台：950；证据等级：tool_observed）。
- 移除 extern 直连、改为框架机制（HostInputs 注册）后：包正常安装，tiling 正常执行（来源平台：950；证据等级：tool_observed）。

## 坏实践

- 用 `--force` 跳过安装校验强装——运行期加载 tiling .so 时同样会因未定义符号崩溃，且掩盖问题。
- 在 tiling 代码里堆 `extern "C"` 声明"借"内部接口：内部符号无 ABI 承诺，跨版本易碎。

## 误诊

- 最初以为"tiling .so 已链接 libruntime（ldd 可见），所以直接调用没问题"——链接依赖存在不代表自定义库会对这些内部符号做静态解析；安装器对未定义引用的校验才是关卡。

## 失效边界

**未决项**：

- 优先项：tiling 内 dlsym aclrtMemcpy 挂起的机制未定位（无输出超时），若要评估 dlsym 兜底路线需先最小化复现该挂起。