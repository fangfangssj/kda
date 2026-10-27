---
type: "Runbook"
title: aclrtFinalize 不存在，收尾用 aclFinalize
description: 自定义算子 aclnn 冒烟测试收尾误写 aclrtFinalize()（受 aclrtSetDevice/aclrtResetDevice 对称性误导），ACL C API 只有 aclFinalize()，改后编译通过。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# aclrtFinalize 不存在，收尾用 aclFinalize

## 现象

在为自定义算子 softplus_v2_grad（两级 aclnn 接口：`aclnnSoftplusV2GradGetWorkspaceSize` + `aclnnSoftplusV2Grad`）编写 OpDef 接口桩的冒烟测试 `tests/aclnn/test_aclnn.cpp` 时，算子包本体 `./build.sh softplus_v2_grad_package` 编译、打包、安装全部成功（退出码 0），但随后 `cd operators/softplus_v2_grad_package/tests/aclnn && ./run.sh` 在编译测试程序这一步失败，报错原文：

```text
/workspace/.../tests/aclnn/test_aclnn.cpp: In function 'int main()':
/workspace/.../tests/aclnn/test_aclnn.cpp:133:15: error: 'aclrtFinalize' was not declared in this scope; did you mean 'aclFinalize'?
  133 |     CHECK_ACL(aclrtFinalize());
      |               ^~~~~~~~~~~~~
/workspace/.../tests/aclnn/test_aclnn.cpp:10:27: note: in definition of macro 'CHECK_ACL'
   10 |         aclError __ret = (x);                                                 \
      |                           ^
make[2]: *** [CMakeFiles/test_aclnn.dir/build.make:76: CMakeFiles/test_aclnn.dir/test_aclnn.cpp.o] Error 1
make: *** [Makefile:91: all] Error 2
```

当时测试代码的状态：`main()` 里已经按 aclrt 前缀族写了 `aclrtSetDevice(0)`、`aclrtResetDevice(0)`，收尾对称地写了 `CHECK_ACL(aclrtFinalize());`（第 133 行）。这是从 AddCustom 模板测试改写为新算子测试时手写引入的：模板原文件用的是 `aclFinalize()`。

## 适用场景

- 工程形态：自定义算子（AscendC/aclnn）的 host 侧 C/C++ 测试程序 / aclnn 直调 harness，链路为 `libasccl`（`acl/acl.h`）。
- API 路径：ACL C API 生命周期族——`aclInit`/`aclFinalize` 与 runtime 子系统族 `aclrtSetDevice`/`aclrtResetDevice`/`aclrtCreateStream`/`aclrtSynchronizeStream` 等。
- 验证过的平台与 CANN 版本：Ascend 950 系列 + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（结论是 C API 命名层事实，与具体算子无关）。
- 路线优先级：recommended_fastpath（写对即免一次编译往返）。
- 明确不适用 / 未验证场景：Python pyacl 绑定层（`torch_npu`/`acl` python 包）的初始化接口命名未验证。

## 根因

ACL C API 的设备/运行时生命周期分两层：`aclInit`/`aclFinalize` 属于顶层 ACL 生命周期（进程级，无 `rt` 中缀），而 `aclrt*` 前缀是 runtime 子系统函数（设备/流/内存：`aclrtSetDevice`、`aclrtCreateStream`、`aclrtMalloc`……）。不存在 `aclrtFinalize` 这个符号，因此链接前就在 `acl/acl.h` 声明作用域内编译失败；编译器的 `did you mean 'aclFinalize'?` 直接给出正确名。

误写诱因是命名对称性：同一个 `main()` 里成对出现 `aclrtSetDevice`/`aclrtResetDevice`，容易类推出 `aclrtFinalize`。

## 解决配方

修复前（`main()` 收尾，第 132-134 行）：

```cpp
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclrtFinalize());
    return ret;
```

修复后：

```cpp
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return ret;
```

配对关系：`aclInit(nullptr)` ↔ `aclFinalize()`；`aclrtSetDevice(0)` ↔ `aclrtResetDevice(0)`。写测试收尾时按这两对配对，不要跨族组合。

完整的修复后 `main()`（两级 aclnn 冒烟调用骨架，可直接复用到任何新算子的桩期冒烟测试）：

```cpp
int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    int ret = 0;
    int passed = 0;

    // proto.md §2 dtype combos: fp16/fp16/fp16 and fp32/fp32/fp32.
    // Also exercise default vs. non-default attrs (beta/threshold are OPTIONAL).
    struct Case {
        aclDataType dtype;
        double beta;
        double threshold;
        const char* name;
    };
    const Case cases[] = {
        {ACL_FLOAT16, 1.0, 20.0, "fp16"},
        {ACL_FLOAT16, 2.0, 10.0, "fp16"},
        {ACL_FLOAT,   1.0, 20.0, "fp32"},
        {ACL_FLOAT,   0.5, 30.0, "fp32"},
    };

    for (const auto& c : cases) {
        if (test_smoke(c.dtype, c.beta, c.threshold, c.name) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
    }

    fprintf(stdout, "\n==== OpDef stub smoke summary: %d/%zu cases PASSED ====\n",
            passed, sizeof(cases) / sizeof(cases[0]));

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return ret;
}
```

## 设计决策

- 无方案选型权衡（单字符级笔误修复）；唯一可记录的决策是冒烟测试只断言两级 aclnn 调用返回 OK、不做数值比对——因为被测对象是接口桩（kernel 无计算逻辑），数值比对要等真实 kernel 阶段。
- 当前路线属 recommended_fastpath。

## 系统信息

- toolchain：CANN 9.1.0（`ASCEND_HOME_PATH`/`ASCEND_OPP_PATH` 已导出）；cmake + GNU 13.3.0；目标 `ascend950`。
- 相关文件：`operators/softplus_v2_grad_package/tests/aclnn/test_aclnn.cpp`（出错与修复处）、`operators/softplus_v2_grad_package/tests/aclnn/run.sh`（触发编译的入口）、autogen 的 `aclnn_softplus_v2_grad.h`（两级调用签名来源）。

## 预防

- 写 ACL host 测试收尾时按固定配对检查：`aclInit↔aclFinalize`、`aclrtSetDevice↔aclrtResetDevice`、`aclrtCreateStream↔aclrtDestroyStream`。
- 编译器 `did you mean` 提示已经给出正确符号名——遇到 `'aclrtXxx' was not declared` 先看提示再查文档。

## 验证信号

问题与验证来自一次 softplus_v2_grad 算子 OpDef 桩阶段的 aclnn 冒烟测试编写与修复。实测信号：

- 修复前 `run.sh` 编译失败：`error: 'aclrtFinalize' was not declared in this scope; did you mean 'aclFinalize'?`，`make: *** [Makefile:91: all] Error 2`（来源平台：950；证据等级：tool_observed）。
- 将第 133 行改为 `CHECK_ACL(aclFinalize());` 后同一条 `run.sh` 命令编译通过并执行：`Test PASSED for fp16 n=16 beta=1.0 threshold=20.0` 等 4 条全部 PASSED，`==== OpDef stub smoke summary: 4/4 cases PASSED ====`，`RUN_EXIT=0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定 ACL C API（`acl/acl.h`）的函数命名归属；其他模块（如 hccl、runtime 扩展）的 init/finalize 命名不适用本卡。
- 未验证，推断可迁移：其他 CANN 版本上 `aclFinalize` 同样存在（该符号自早期版本即有），推断可迁移但未实测。

**未决项**：

- 无（根因已追到 API 命名层，机制明确）。
## 坏实践

- 按 `aclrt*` 前缀对称性"类推"生命周期函数名——runtime 子系统族与顶层 ACL 族命名规则不同。
- 编译错误先怀疑头文件包含/链接选项：本例错误发生在声明作用域（`not declared in this scope`），头文件与库都在，只是符号名不存在。
