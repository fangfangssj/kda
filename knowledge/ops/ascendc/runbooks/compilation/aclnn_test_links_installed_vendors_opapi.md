---
type: "Runbook"
title: aclnn 测试链接的是 OPP 安装目录的 libcust_opapi.so
description: tests/aclnn 的 run.sh 先把 run 包装进 $ASCEND_OPP_PATH/vendors/<OpType>/ 再编译运行测试；ldd/objdump/nm 三探针确认测试加载的是安装目录库而非 build 树；改码不重装时 ST 静默跑旧 kernel 输出全 0。
tags: [task.debugging, topic.implementation.operator_registration, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# aclnn 测试链接的是 OPP 安装目录的 libcust_opapi.so

## 现象

在对一个 ACLNN 自定义算子工程（AscendC，ops-registry-invoke 流程，接口桩阶段）跑 `tests/aclnn/run.sh` 做冒烟验收时，脚本内部完成一整条链：重新 cmake+build → CPack 生成 `custom_opp_ubuntu_x86_64.run` → **安装 run 包到 `$ASCEND_OPP_PATH`** → 编译 `test_aclnn` → 执行。安装段日志（verbatim 摘录）：

```
==> Run package: .../build/custom_opp_ubuntu_x86_64.run
==> Installing ...
Verifying archive integrity...     0%   37%   74%  100%   SHA256 checksums are OK. All good.
[ops_custom] [INFO] upgrade op api
include lib
[ops_custom] [INFO] replace or merge old ops op_api files ......
[ops_custom] [INFO] copy new op_api files ......
SUCCESS
```

安装器还会提示：运行前需 `export LD_LIBRARY_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/<OpType>/op_api/lib/:${LD_LIBRARY_PATH}`。也就是说，测试可执行文件 `build/test_aclnn` 在运行时加载的 `libcust_opapi.so` 来自**安装目录**，不是工程 build 树里的那份。要确认这一点（并确认安装库的 aclnn 符号导出完整），需要对测试二进制与安装库分别跑 ldd / objdump / nm 三个探针。

## 适用场景

- 工程形态：aclnn 算子仓验收/联调（`tests/aclnn/run.sh` 冒烟）；怀疑「测试没跑到我刚改的 op_api/op_host 代码」「链接了错误的 libcust_opapi.so」的场景。
- API 路径：aclnn 两段式 C API——`aclnn<OpType>GetWorkspaceSize(输入…, 属性…, 输出…, &workspaceSize, &executor)` + `aclnn<OpType>(workspace, workspaceSize, executor, stream)`；库 `$ASCEND_OPP_PATH/vendors/<OpType>/op_api/lib/libcust_opapi.so`；头 `$ASCEND_OPP_PATH/vendors/<OpType>/op_api/include/aclnn_<op>.h`（构建时由 op_build 自动生成）。
- 验证过的平台与 CANN 版本：ascend950（注册平台名 950）+ CANN cann-9.0.0-beta.2；后续增量（stale-install 对照实验）在 CANN 9.1.0 复现。
- 适用粒度（transfer_scope）：platform_wide——run 包安装 + vendors 目录加载机制跨算子一致。
- 路线优先级：recommended_fastpath（三条探针一次跑完，秒级）。
- 明确不适用 / 未验证场景：GEIR / proto 直调路径；TTK 等自带安装管理的评测工具。

## 判别方法

三探针（在 `tests/aclnn` 下、run.sh 完成后执行）：

```bash
# 1) 测试二进制运行时实际加载哪个 opapi 库
ldd build/test_aclnn | grep 'libcust_opapi.so'

## 2) 测试二进制引用的 aclnn 符号数
objdump -T build/test_aclnn | grep -c 'aclnn<OpType>GetWorkspaceSize'

## 3) 安装库导出的 aclnn 两符号
nm -D $ASCEND_OPP_PATH/vendors/<OpType>/op_api/lib/libcust_opapi.so | grep -E 'aclnn<OpType>'
```

判读：
- ldd 应指向 `$ASCEND_OPP_PATH/vendors/<OpType>/op_api/lib/libcust_opapi.so`（安装目录）。指向别处（如旧安装、别的 vendor）即为链接目标漂移。
- nm -D 应命中两个 `T` 符号：`aclnn<OpType>` 与 `aclnn<OpType>GetWorkspaceSize`（缺任一个，调用链相应一段必挂）。
- objdump 计数 ≥ 1（测试确实引用了 GetWorkspaceSize 段）。

## 设计决策

- 选 ldd（运行时解析结果）而不是只看链接命令/路径配置：ldd 直接给出「实际加载的库在哪」，能暴露 LD_LIBRARY_PATH 与安装目录漂移问题。
- nm -D 与 objdump -T 分别覆盖「库导出侧」与「可执行文件引用侧」，两侧对上才能确认调用通路完整——本卡采用的双侧核验设计，属 recommended_fastpath。
- 冒烟用例采用两段式调用（先 GetWorkspaceSize 拿 workspaceSize+executor，再 aclnn 下发）而非一步式：与生成的 aclnn 头签名一致，且能分别暴露两段的符号/链接问题。属性参数按生成头顺序夹在输入与输出之间（输入(声明序) → 属性 → 输出(声明序) → &workspaceSize → &executor），签名顺序照抄生成头，不凭记忆拼。

## 系统信息

- 构建环境：x86_64 Ubuntu，GCC 13.3.0，CMake + op_build + CPack External；目标 ascend950；CANN cann-9.0.0-beta.2，`$ASCEND_OPP_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2/opp`。
- 相关文件（与本次事件流实际读取/执行一一对应）：
  - `tests/aclnn/run.sh`（构建 + 打包 + 安装 + 编译测试 + 运行的入口；其行为由执行输出确认）
  - `tests/aclnn/test_aclnn.cpp`（两段式调用测试；`ElemSize`: fp32=4，fp16/bf16=2——无 aclBFloat16 C 类型，bf16 仅按 ACL_BF16 枚举 + 按字节分配）
  - `build/custom_opp_ubuntu_x86_64.run`（CPack 产物，内含 `packages/vendors/<OpType>/op_api/lib/libcust_opapi.so`）
  - `$ASCEND_OPP_PATH/vendors/<OpType>/op_api/lib/libcust_opapi.so`（被加载库）
  - `$ASCEND_OPP_PATH/vendors/<OpType>/op_api/include/aclnn_<op>.h`（生成的 aclnn 头，签名来源）

## 完整代码

两段式调用段（`tests/aclnn/test_aclnn.cpp` 的 RunSmokeCase 关键段，verbatim 摘自事件流读到的文件内容）：

```cpp
    // --- Phase 1: 查询 workspace 大小并获取 executor ---
    // 签名照抄生成头：输入(gradHy, storage) → 属性(bool hasBias) →
    // 输出×5 → workspaceSize → executor。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnThnnFusedGruCellGradGetWorkspaceSize(
        gradHy, storage, true, gradInputGates, gradHiddenGates, gradHx,
        gradInputBias, gradHiddenBias, &workspaceSize, &executor));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // --- Phase 2: 下发算子并等待完成 ---
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACLNN(aclnnThnnFusedGruCellGrad(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
```

dtype 字节数（同文件，bf16 无 C 类型时按字节分配的写法，verbatim）：

```cpp
// 按 dtype 取元素字节数：fp32=4；fp16/bf16=2（无 aclBFloat16 类型，bf16 仅按
// ACL_BF16 枚举 + 按字节分配即可）。
size_t ElemSize(aclDataType dtype)
{
    return (dtype == ACL_FLOAT) ? 4 : 2;
}
```

探针命令见 ## 判别方法（完整可执行）。

## 预防

- 跑 aclnn 测试前确认 run 包已安装到 `$ASCEND_OPP_PATH/vendors/<OpType>/`（run.sh 自带安装段；若手工只 build 不 install，需自查），并用 ldd 复核加载目标。
- 环境未设 `LD_LIBRARY_PATH` 时按安装器提示 export（指向 `vendors/<OpType>/op_api/lib/`），否则加载目标可能落到非预期路径或失败。
- 改动 op_api/op_host 源码后，走完整 run.sh（build → CPack → install → 测试），不要直接重跑旧测试二进制。

## 验证信号

以下信号来自该接口桩工程的复验：亲自执行 run.sh（含安装段）、ldd/objdump/nm 探针与 PASSED/FAILED 计数统计取得。

- run.sh 退出码 0（RUN_EXIT=0）：含 `==> Installing ...` 与各 upgrade 段 `SUCCESS`、`[smoke] thnn_fused_gru_cell_grad fp32 PASSED`、`==== ThnnFusedGruCellGrad aclnn smoke summary: 1/1 cases OK ====`；对日志 `grep -c 'PASSED'` = 1、`grep -c 'FAILED'` = 0（来源平台：950；证据等级：tool_observed）
- `ldd build/test_aclnn | grep libcust_opapi.so` → `/home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so`（安装目录，非 build 树）（来源平台：950；证据等级：tool_observed）：

```
	libcust_opapi.so => /home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/ThnnFusedGruCellGrad/op_api/lib/libcust_opapi.so (0x00007f33d80db000)
```

- `objdump -T build/test_aclnn | grep -c aclnnThnnFusedGruCellGradGetWorkspaceSize` = 1（来源平台：950；证据等级：tool_observed）
- `nm -D $ASCEND_OPP_PATH/vendors/<OpType>/op_api/lib/libcust_opapi.so | grep aclnnThnnFusedGruCellGrad` → 两符号均 `T` 导出（来源平台：950；证据等级：tool_observed）：

```
0000000000003039 T aclnnThnnFusedGruCellGrad
0000000000002809 T aclnnThnnFusedGruCellGradGetWorkspaceSize
```

后续增量（stale-install 对照实验，CANN 9.1.0，详见下方"后续增量"节）：

- 只重编译未重装：ST 19 用例 12 失败，失败全部为新增多张量用例且输出静默全 0（`期望=3.213609, 实际=0.000000, rel_err=1.00e+00`，通过率 36.8%），单张量旧路径用例照常通过（来源平台：950；证据等级：tool_observed）。
- 仅执行重装（代码零改动）：`bash build/custom_opp_ubuntu_x86_64.run --quiet` 输出 `[ops_custom] ... SUCCESS`；复跑 19 用例仅 2 失败（剩余 2 个为比较器近零值误报，另卡覆盖），多张量用例 MERE 降至 1.76e-09/1.09e-09/8.69e-09 量级（来源平台：950；证据等级：tool_observed）。

## 性能影响

（无性能故事——本卡为链接/加载核验，不涉及两路线性能对比。）

## 失效边界

- 「改源码不重装则测试跑的是旧库」原为机制推断——后续增量已用对照实验证实（见下），结论从推断升级为实测。
- ldd 结果依赖运行时环境（LD_LIBRARY_PATH、ld 缓存）；换机器/换 shell/换用户需重查，不能把一次 ldd 输出当作跨环境事实。
- 冒烟通过只证明两段式调用通路返回 OK，不含数值正确性（接口桩阶段测试不校验数值）。
- 观察限于 cann-9.0.0-beta.2 + ascend950 的 CPack run 包流程（stale-install 增量为 9.1.0）；其他安装方式（手工拷贝、共享 vendors 目录合并）未验证。

## 坏实践

- 改完 op_api/op_host 源码后不重跑 run.sh 的 build+install 段，直接执行旧 `build/test_aclnn` → 测的是旧安装库（后续增量已实证，且表现为静默全 0 而非报错）。
- 只看测试退出码 0，不核对 ldd 加载目标与 nm/objdump 符号 → 链接错库、符号缺失（如 GetWorkspaceSize 未导出）不会在退出码层面暴露。
- 环境未设 `LD_LIBRARY_PATH` 指向 `vendors/<OpType>/op_api/lib/` 就断言「库没装上」——先按安装器提示 export 再复查 ldd。

## 误诊

（无。本次三探针结果与预期一致，未发生错误归因。）

## 未决事项

- ~~stale-install 行为未做对照实验（改源码 → 不重装 → 跑测试，观察是否仍旧行为）；如需把「必须重装后才生效」写成硬结论，需补该实验。~~（已由后续增量闭合，见下。）
- run.sh 脚本本身未读取，其安装/环境设置步骤由执行输出推断；如需精确复刻（如只重装不重编），应先读脚本确认分段。

## 后续增量（2026-09-16：stale-install 对照实验闭合）

后续 trace（Ascend950PR，CANN 9.1.0，TensorList 算子多张量 kernel 重写联调）完成了原未决事项的对照实验，把「必须重装后才生效」从推断升级为实测，并补充了该失效的**静默形态**：

- **实验设计**：kernel/tiling 重写并编译成功（UT 21/21 过）→ 只重跑 ST（未重装 .run）→ 观察失败形态 → 仅执行重装（代码零改动）→ 复跑对照。
- **机制**：`build.sh` 只负责编译并打包 `custom_opp_ubuntu_x86_64.run`，**不会**把它安装到 `opp/vendors/<op>_custom`；ST 的 run.sh 把 `LD_LIBRARY_PATH` 指向安装目录，加载的 `libcust_opapi.so` 与调度到的 kernel 都是**上次安装的旧版**。旧版 kernel 不认识新的多张量 tiling/寻址逻辑，输出缓冲不被写入，表现为**静默全 0**——无报错、部分（旧路径）用例还通过，极易被误判为 kernel 算法 bug。
- **判别探针**（对照安装目录与 build 产物的时间戳）：

```bash
ls -la ${ASCEND_HOME_PATH}/opp/vendors/<op>_custom/op_api/lib/       # 安装时间 vs build/ 产物时间
ls -la operators/<op>/build/custom_opp_ubuntu_x86_64.run            # 新包是否已生成
```

- **固化流程**：把"重装 run 包"固化为改码后上板前的必经步骤：

```bash
cd operators/<op> && bash build.sh --soc=ascend950 -j8        # 编译 + 打包
bash build/custom_opp_ubuntu_x86_64.run --quiet               # 安装/升级到 opp/vendors
cd tests/st && bash run.sh                                     # 再跑 ST
```

- **附加注意**：`--quiet` 安装输出的 `SUCCESS` 与 `no need to upgrade` 要区分——后者可能意味着没装上新包；联调链路建议显式确认新包已安装（打印安装目录 mtime）。
