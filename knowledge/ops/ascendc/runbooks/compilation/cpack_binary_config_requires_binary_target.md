---
type: "Runbook"
title: CPack 失败—binary/config 需 binary 聚合目标
description: AscendC 算子包只构建 <OpType>_<soc> 目标就 cpack 会报 file INSTALL cannot find binary/config；须改 build binary 聚合目标拉起 gen_ops_config。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# CPack 失败—binary/config 需 binary 聚合目标


## 现象

在给自定义算子的 tiling 单元测试搭全链构建脚本（`tests/tiling/run.sh`：cmake 配置 → 构建 run 包 → cpack 打包 → 本地安装 → 编译 gtest → 运行）时，脚本第 1 步照参考工程的写法只构建 per-op 目标：

```bash
cmake -S "${PKG_DIR}" -B "${BUILD_DIR}" -DCANN_HOME="${CANN_HOME}"
cmake --build "${BUILD_DIR}" --target ThnnFusedGruCellGrad_ascend950 -j"$(nproc)"
cpack --config "${BUILD_DIR}/CPackConfig.cmake" -B "${BUILD_DIR}"
```

构建本体全部成功（host 库链接 `[100%] Linking CXX shared library libcust_opmaster_rt2.0.so`、opc 生成三个 kernel 二进制、`[100%] Built target ThnnFusedGruCellGrad_ascend950`），但 cpack 阶段报错：

```
CPack: Create package using External
CPack: Install projects
CPack: - Run preinstall target for: custom_op_prj
CPack: - Install project: custom_op_prj []
CMake Error at /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/operators/ThnnFusedGruCellGrad_package/tests/tiling/build/cmake_install.cmake:147 (file):
  file INSTALL cannot find
  "/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/operators/ThnnFusedGruCellGrad_package/tests/tiling/build/ascendc_kernels/binary/config":
  No such file or directory.


CPack Error: Error when generating package: custom_op_prj
```

即：新构建目录里 `build/ascendc_kernels/binary/` 下只有 `ascend950` 与 `dynamic`，没有 `config`；而老构建目录（`operators/<Op>_package/build/`，此前成功打包过）里 `binary/` 下有 `ascend950`、`config`、`dynamic` 三项。另注意：命令用了 `./run.sh 2>&1 | tee log; echo "RUN_EXIT=$?"`，`$?` 取到的是 `tee` 的退出码（0），一度掩盖了 cpack 的真实失败——判断链路成败要看 cpack/gtest 的实际输出，不能只看管道后退出码。

## 适用场景

- 工程形态：算子仓自定义算子包（AscendC ops package，CMake + CPack 产 `custom_opp_ubuntu_x86_64.run`）；任何需要在新构建目录里构建 run 包的脚本（UT 脚手架 run.sh、CI、本地复现）。
- API 路径：`cmake --build --target <target>` + `cpack --config CPackConfig.cmake`；涉及的 CMake 目标族 `<OpType>_<soc>` / `ascendc_kernels_ascendc_bin_<soc>_gen_ops_config` / 聚合目标 `binary`。
- 验证过的平台与 CANN 版本：Ascend 950（`ASCEND_COMPUTE_UNIT=ascend950`）+ CANN 9.0.0-beta.2（ascendc kernel cmake 由 `compiler/tikcpp/ascendc_kernel_cmake` 提供；与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（绑定本 CANN 版本的 ascendc kernel cmake 目标图，不绑定算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：未验证其他 CANN 版本的目标依赖图（旧版可能由 per-op 目标直接依赖 gen_ops_config，参考工程的 run.sh 或因此在旧版曾通过）；仅构建 host 侧库（不打包 run 包）的场景无此问题。

## 根因

CPack 的安装规则（`cmake_install.cmake:147`）要求把 `build/ascendc_kernels/binary/config/` 整目录安装进 run 包（`file(INSTALL ... kernel/config/ ... binary/config/)`），而该目录**只**由 CMake 目标 `ascendc_kernels_ascendc_bin_<soc>_gen_ops_config` 生成（其 build.make 规则：跑 `insert_simplified_keys.py` 与 `ascendc_ops_config.py`，`cmake -E make_directory .../binary/config/ascend950`，再把 `binary/ascend950/*.json` 移入）。在本 CANN 版本的目标图里，per-op 目标**不**依赖它：

```
# Makefile2（构建目录实测）：
CMakeFiles/<OpType>_<soc>.dir/all: CMakeFiles/ascendc_kernels_ops_info_gen_<soc>.dir/all
CMakeFiles/<OpType>_<soc>.dir/all: CMakeFiles/<OpType>_ascendc_cust_optiling.dir/all
                                     （无 gen_ops_config）

CMakeFiles/binary.dir/all: CMakeFiles/ascendc_kernels_ascendc_bin_<soc>_gen_ops_config.dir/all
CMakeFiles/ascendc_kernels_ascendc_bin_<soc>_gen_ops_config.dir/all: CMakeFiles/<OpType>_<soc>.dir/all
```

即依赖链是 `binary → gen_ops_config → <OpType>_<soc>`：只有聚合目标 `binary` 会传递拉起 gen_ops_config；只构建 `<OpType>_<soc>` 时 `binary/config/` 永不生成，cpack 必然报 "file INSTALL cannot find"。仓库里经过验证的 `operators/build.sh` 正是用 `make -j binary package`，因此从未踩到。

## 解决配方

run.sh 第 1 步 before → after（逐字取自事件流）：

```bash
## before：只建 per-op 目标 → cpack "file INSTALL cannot find .../binary/config"
cmake -S "${PKG_DIR}" -B "${BUILD_DIR}" -DCANN_HOME="${CANN_HOME}"
## <OpType>_<soc> is the ALL target: it pulls in the host-side libraries
## AND compiles the device kernel, producing the kernel binary directory that
## CPack installs into the run package.
cmake --build "${BUILD_DIR}" --target ThnnFusedGruCellGrad_ascend950 -j"$(nproc)"
cpack --config "${BUILD_DIR}/CPackConfig.cmake" -B "${BUILD_DIR}"

## after：改 build binary 聚合目标（传递依赖 gen_ops_config 与 <OpType>_<soc>）
cmake -S "${PKG_DIR}" -B "${BUILD_DIR}" -DCANN_HOME="${CANN_HOME}"
## Build the "binary" aggregate target (same flow as operators/build.sh's
## `make binary package`): it transitively builds ThnnFusedGruCellGrad_ascend950
## (host-side libraries + device kernel binaries) AND
## ascendc_kernels_ascendc_bin_ascend950_gen_ops_config, which produces the
## ascendc_kernels/binary/config/ directory that the CPack install rules
## require. Building ThnnFusedGruCellGrad_ascend950 alone is NOT enough in this
## CANN version (binary/config missing -> CPack "file INSTALL cannot find").
cmake --build "${BUILD_DIR}" --target binary -j"$(nproc)"
cpack --config "${BUILD_DIR}/CPackConfig.cmake" -B "${BUILD_DIR}"
if [ ! -f "${RUN_PKG_FILE}" ]; then
    echo "ERROR: run package not produced: ${RUN_PKG_FILE}" >&2
    exit 1
fi
```

通用化：任何 AscendC 算子包脚本把 `--target <OpType>_<soc>` 换成 `--target binary` 即可（`binary` 由 opbuild cmake 模块创建，`operators/build.sh` 的 `make -j binary package` 是同义已验证流程）。

## 设计决策

- 为何选 `--target binary` 而非其他：① 它是 Makefile2 里唯一传递依赖 gen_ops_config 的现成聚合目标，一次拉全 kernel 二进制 + host 库 + ops config；② 与仓库已验证的 `operators/build.sh`（`make -j binary package`）流程完全同构，不引入新组合；③ 老构建目录的成功产物（`binary/config/` 存在）证明 `binary`+cpack 组合可行。
- 备选与放弃原因：给 per-op 目标手工补 `add_dependencies(<OpType>_<soc>, ...gen_ops_config)`——放弃：要改包级 CMakeLists（侵入产物工程），且不同 CANN 版本目标名带 soc 后缀，脆弱；直接复用 `operators/build.sh` 整脚本——放弃：UT 的 run.sh 需要**本地安装到固定目录**等自定义步骤，只需对齐其构建目标。
- 当前方案属哪档路线：recommended_fastpath。

## 系统信息

- CANN：`/home/<user>/Ascend/cann-9.0.0-beta.2`；ascendc kernel cmake 模块位于 `${CANN}/compiler/tikcpp/ascendc_kernel_cmake/`（gen_ops_config 规则调用的 `fwk_modules/util/insert_simplified_keys.py`、`fwk_modules/util/ascendc_ops_config.py`）。
- 目标平台：`ASCEND_COMPUTE_UNIT=ascend950`；产物 `custom_opp_ubuntu_x86_64.run`；宿主 x86_64 Linux。
- 相关文件：`operators/<Op>_package/tests/tiling/run.sh`（修复落点）；`operators/build.sh`（已验证参照，全文见「## 完整代码」）；构建目录 `tests/tiling/build/`（Makefile2 / cmake_install.cmake / CMakeFiles/*/build.make 为根因证据）。

## 完整代码

仓库已验证的 `operators/build.sh`（逐字取自事件流，`make -j binary package` 即本卡结论的来源流程）：

```bash
#!/bin/bash
set -e

PKG=${1:?Usage: $0 <package_name>}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR/$PKG"

[ -d "$PKG_DIR" ] || { echo "ERROR: $PKG_DIR not found"; exit 1; }

echo "==> Cleaning and entering $PKG ..."
cd "$PKG_DIR"
[ -d build ] && rm -rf build
mkdir -p build
cd build

echo "==> Running cmake ..."
cmake ..
echo "==> Building ..."
make -j binary package

RUNFILE=$(find . -maxdepth 1 -name "*.run" -not -name "*.run.json" | head -1)
if [ -z "$RUNFILE" ]; then
    echo "ERROR: no .run package generated"
    exit 1
fi
echo "==> Run package: $(realpath "$RUNFILE")"
echo "==> Installing ..."
"$RUNFILE"
```

gen_ops_config 目标的生成规则（构建目录 `CMakeFiles/ascendc_kernels_ascendc_bin_ascend950_gen_ops_config.dir/build.make`，逐字取自事件流，说明 `binary/config/` 从何而来）：

```make
ascendc_kernels_ascendc_bin_ascend950_gen_ops_config:
	python3 /home/<user>/Ascend/cann-9.0.0-beta.2/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/insert_simplified_keys.py -p <build>/ascendc_kernels/binary/ascend950
	python3 /home/<user>/Ascend/cann-9.0.0-beta.2/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_ops_config.py -p <build>/ascendc_kernels/binary/ascend950 -s ascend950
	/opt/buildtools/cmake-3.20.5-linux-x86_64/bin/cmake -E make_directory <build>/ascendc_kernels/binary/config/ascend950
	mv <build>/ascendc_kernels/binary/ascend950/*.json <build>/ascendc_kernels/binary/config/ascend950
```

（`<build>` 为构建目录绝对路径，原文为完整路径。）run.sh 的 before/after 见「## 解决配方」。

## 预防

- 在新构建目录构建 AscendC 算子 run 包时，一律以 `binary` 聚合目标（或直接 `make -j binary package`）为准，不要照抄只含 per-op 目标的脚本。
- 脚本判断链路成败不要依赖 `cmd | tee` 之后的 `$?`（那是 tee 的退出码）；用 `${PIPESTATUS[0]}` 或 `set -o pipefail`，并以产物存在性（`.run` 文件、gtest 可执行文件）与关键输出行为准。

## 验证信号

来源说明：以下信号来自本次自定义算子公共 TilingUT（测试先行）开发会话中 run.sh 全链的两次实际运行（修复前后各一次）。

- 修复前：只建 `ThnnFusedGruCellGrad_ascend950` 后 cpack 报 `CMake Error ... file INSTALL cannot find ".../build/ascendc_kernels/binary/config": No such file or directory` + `CPack Error: Error when generating package: custom_op_prj`（来源平台：950（包构建目标 ascend950，host x86_64 构建）；证据等级：tool_observed）。
- 根因证据：构建目录 Makefile2 显示 `binary.dir/all` 依赖 `ascendc_kernels_ascendc_bin_ascend950_gen_ops_config.dir/all`、后者依赖 `ThnnFusedGruCellGrad_ascend950.dir/all`，而 `ThnnFusedGruCellGrad_ascend950.dir/all` 只依赖 ops_info_gen 与 cust_optiling（无 gen_ops_config）（来源平台：950；证据等级：tool_observed）。
- 修复后：run.sh 改 `--target binary` 重跑，包构建 + cpack + 本地安装 + gtest 编译 + 运行全链走通（进入 gtest 阶段并运行 33 用例，红基线 0 passed / 33 failed、脚本以 gtest 预期非 0 退出；不再出现 cpack 报错）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.0.0-beta.2 的 ascendc kernel cmake 目标图；其他版本若调整依赖（如让 per-op 目标直接依赖 gen_ops_config），`--target <OpType>_<soc>` 可能恢复可用（未验证）。
- `binary` 聚合目标会构建**全部**已配置 soc 的 kernel 二进制与配置，比只建 per-op 目标多花时间；只在需要最小化构建时才需另寻窄目标（本 trace 未探索更窄且含 gen_ops_config 的目标组合）。
- 本卡只覆盖"新构建目录 + cpack"场景；已有成功构建目录里 `binary/config/` 残留时，旧的 per-op 写法可能"碰巧"通过（参考工程 run.sh 疑似因此曾通过，未验证）。

## 坏实践

- 照抄参考工程脚本不验证目标图：参考 `run.sh` 用 `--target <OpType>_<soc>`，在新构建目录直接 cpack 必失败；复制脚本时要对齐"这个 CANN 版本下目标实际产出什么"。
- 用 `cmd | tee log; echo $?` 判断成败——退出码属于 tee（本次 RUN_EXIT=0 掩盖了 cpack 失败）。

## 误诊

- 初判曾怀疑"换个构建目录（tests/tiling/build 而非包根 build/）导致失败"、进而怀疑包 CMakeLists 安装规则写错；对拍两份 CMakeLists（与参考工程 diff 后仅注释与算子名差异）排除后，才转向构建目标图，从 Makefile2 与 build.make 找到 `binary → gen_ops_config → <OpType>_<soc>` 的真实依赖链。

## 未决事项

- 参考工程（scale_custom_package）run.sh 的 per-op 目标写法曾通过的具体条件未验证（推测：旧构建目录残留 `binary/config/` 或 CANN 版本差异）——若需确证，可在干净目录用该 CANN 版本复跑参考脚本。
