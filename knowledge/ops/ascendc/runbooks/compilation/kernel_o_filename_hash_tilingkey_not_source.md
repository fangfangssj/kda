---
type: "Runbook"
title: kernel .o 文件名 hash 不随源码变化，以 MD5 判重编
description: AscendC kernel 二进制文件名中的 hash 由 TilingKey/模板参数决定而非源码内容，源码改动后文件名不变是正常现象；判定"修改是否生效"要用 md5sum 对比 .o 内容并核对安装目录一致性，再用常量写入探针验证分支执行，不能凭"输出没变"反推"没重编译"。
tags: [task.debugging, topic.compilation, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# kernel .o 文件名 hash 不随源码变化，以 MD5 判重编

## 现象

Ascend950PR（arch35，CANN 9.1.0）自定义算子 kernel 调试期间反复出现同一类迷惑现象：**改完 kernel 源码（含 .h）→ 重建 → 重装 → 重跑 ST，测试输出与改前逐位完全相同**，像是修复无效。两条独立轨迹的典型观测：

- 轨迹一（roi_align 类算子）：每次改完 `op_kernel/arch35/*.h` 重建，构建日志都显示同一个 kernel 二进制文件名 `BenchNoHarnessRoiAlign_c3990d1b16e9498379a812cdc211752d.o`；多次源码修改、甚至 `rm -rf build` + 清 `~/atc_data/kernel_cache` 后全量重建，文件名 hash `c3990d1b…` 纹丝不动，一度被误判为"kernel 没有重编译/构建缓存陈旧"，浪费了排查轮次。
- 轨迹二（SoftMax+TopK 门控算子）：修改 kernel 头加了 `if (k_ == E_)` 替代分支后重建重装，失败输出与改前逐位相同（同样的垃圾值 14992.0/12752.0）；三个 kernel `.o` 文件名 hash（`06a5b7e1`/`23e59e1d`/`322daf1c`）与改前一字不差，给 `.cpp` 入口改注释、`touch` 源文件后也不变，强烈怀疑"构建缓存导致没有真正重编/新包没装上"，花大量轮次找编译缓存（`~/.atc_data/kernel_cache`、`kernel_bank`、`.compile_cache`、`tbe_cache` 等，全部不存在）。

用 MD5 + 时间戳对账后澄清：文件名 hash 稳定、`.o` 内容每次都在变——重编译正常发生，文件名不是内容指纹。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 迭代调试（改 `op_kernel/*.h/.cpp` → build.sh 重建 → 卸载/重装 run 包 → ST 单用例复跑）。
- API 路径：算子包构建产物布局 `build/op_kernel/ascendc_kernels/binary/<soc>/<op>/<Op>_<hash>.o(.json)`；TilingKey/模板选择机制（同名 hash 的来源）；kernel 元数据 JSON（`<OpName>_<hash>.json`，内含 `sha256` 字段）；安装目录 `$ASCEND_TOOLKIT_HOME/opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/<soc>/<op>/`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN 9.1.0，两条独立轨迹复现（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有自定义算子的构建-调试循环。
- 路线优先级：recommended_fastpath（判定修改生效性的标准顺序）。
- 明确不适用 / 未验证场景：CPU 侧（host/tiling/opapi `.so`）修改的生效性判定不适用本卡（`.so` 链接时间戳可直接看）；其他 CANN 版本的 hash 规则未验证。

## 根因

kernel 二进制文件名中的 hash 段由算子的 TilingKey/模板参数选择决定（同一算子同一 TilingKey/模板选择 → 同名），不是源码内容的指纹；源码变化只改变 `.o` 内容（MD5/时间戳），不改变文件名。配套事实：kernel 元数据 `<OpName>_<hash>.json` 里的 `sha256` 字段就是对应 `.o` 文件内容的 sha256，可与文件实算值互验。

## 判别方法

**规则 1：文件名 hash 不是源码指纹。** `<OpName>_<32位hash>.o` 的 hash 由 TilingKey（dtype/hasFinished 等模板参数组合）决定，同一 TilingKey 下源码怎么改文件名都不变。因此"改了源码但 `.o` 文件名没变"是**正常现象**，不能作为"没有重编译"的证据。多模板/多 dtype 算子会生成多个不同 hash 的 `.o`，与"同名≠同内容"不冲突。

**规则 2：用 md5sum 判定重编译与安装。**

```bash
# 1) 内容指纹：MD5 是否随构建变化
md5sum build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o
# 2) 时间戳：最近构建是否刷新
stat -c "%Y %n" build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.o
# 3) 安装侧一致性：安装目录与构建目录 MD5 是否一致
md5sum $ASCEND_TOOLKIT_HOME/opp/vendors/<vendor>/op_impl/ai_core/tbe/kernel/ascend950/<op>/*.o
```

MD5 变了 → 已重编译，去查运行侧（tiling 缓存、加载路径、逻辑）；MD5 没变而源码变了 → 才是构建缓存问题（再清 build/kernel_cache）。两边一致 = 新二进制已正确安装，把安装环节从怀疑列表里剔除。多模板算子新增 dtype 支持时会出现第二个 hash 的 `.o`（如 `b4a9d97b…`），属正常扩展。

**规则 3：相同输出 ≠ 分支未执行；用常量写入探针收尾。** 在可疑分支开头插入确定性常量写入，重建重装跑单用例：

```cpp
if (isKEqE_ == 1) {
    // DEBUG 探针: 常量写入验证分支执行
    PipeBarrier<PIPE_ALL>();
    Duplicate<COMPUTE_T>(yCompute, static_cast<COMPUTE_T>(1.0),
                         static_cast<uint32_t>(curBatch * kPad_));
    PipeBarrier<PIPE_V>();
    return;  // 或走完后续流程
}
```

输出变成常量（MERE 从 5.53e+07 降到 1.04e+00 量级）即证明：新二进制在跑、分支在执行——剩下的差异只能来自数据通路（上游 buffer 内容），应转向读路径/上游排查，而不是继续怀疑构建链。同理可用 `SetValue` 把 tiling 参数（如 `isKEqE_`、`k_`、`E_`、`topkInner_`）回显到输出，一次验证"tiling 传参 + 分支条件"两个环节。

**执行顺序**：先规则 2（构建/安装层），再规则 3（执行层）；两关都过了仍复现旧输出，才下"修改无效"的结论。

## 设计决策

- 为什么不直接信构建日志的 `Generating ... Done`：它证明编译动作发生了，但不证明产物内容变化（也可能确实无变化，例如纯注释改动后编译器产出相同二进制）；md5 对比把"编译发生了"升级为"二进制变了"。
- 为什么探针用 `Duplicate`/`SetValue` 常量而不是 `AscendC::PRINTF`：PRINTF 在 ST 环境的 stdout 通常不可见，常量写入直接落在被测输出张量里，随精度比对自然可见，无额外日志链路依赖。
- 为什么快速排除"文件名不变"这个怀疑重要：两条轨迹中该现象先后把排查引向错误方向（怀疑构建缓存、怀疑运行时加载旧 kernel、怀疑分支被优化掉）；MD5 对账一次即可终结该分支，应作为调试循环的标准第一步。
- 当前方案路线档位：recommended_fastpath。

## 解决配方

无需修复——这是判别规则而非缺陷。若确认内容未更新，才走干净重建：

```bash
rm -rf build; rm -rf $HOME/atc_data/kernel_cache/ 2>/dev/null; rm -rf /tmp/kernel_cache 2>/dev/null
bash build.sh --soc=ascend950 -j8 && bash build/custom_opp_ubuntu_x86_64.run --force
```

## 预防

- 调试循环中把 `md5sum + stat` 对账固化为改码后的第一动作；不要用文件名（或构建日志里的同名路径）判断重编译。
- 每轮"改码→重建→重装"后固定做一次 md5sum 双端对比（build vs vendor 安装目录）。
- 判定分支执行用常量写入探针，探针验证完即删（探针本身改变 kernel 行为，不删会污染后续结论）。
- 不要去找不存在的编译缓存目录；本环境下未发现任何跨构建的 kernel 编译缓存（rm -rf build 后即全量重编）。
- 交叉验证安装侧：构建目录与 vendors 安装目录的 MD5 应一致（`--force` 安装正常时）。

## 验证信号

以下信号来自两条独立调试轨迹（Ascend950PR，CANN 9.1.0）：多轮修改 kernel 源码并重建重装，对比文件名、md5、时间戳与探针输出。

- 轨迹一：多轮源码修改 + 一次全量干净重建后，kernel 二进制文件名 hash 恒为 `c3990d1b16e9498379a812cdc211752d`；同名文件的 MD5 随构建变化（`7eff0a3e…` → `4c42a3d6e3cd24cee97563c18a0aeac1`），时间戳同步刷新；构建目录与安装目录 MD5 一致（来源平台：950；证据等级：tool_observed）。
- 轨迹二：修改 `op_kernel/arch35/<op>.h`（新增 k==E 替代分支）并重建，三个 kernel `.o` 文件名 hash（`06a5b7e1`/`23e59e1d`/`322daf1c`）与改前完全相同；给 `.cpp` 改注释、`touch` 头文件后仍相同（来源平台：950；证据等级：tool_observed）。
- 轨迹二：同轮 md5sum——build 目录与 vendor 安装目录的对应 `.o` md5 完全一致，证明安装正确（来源平台：950；证据等级：tool_observed）。
- 轨迹二：在分支内插入 `Duplicate(yCompute, 2.0, 8)` 探针后重建，三个 `.o` 的 md5 全部变化（`9cf58270`/`1caa2462`/`0311829a` 等），单用例输出改变；`SetValue` 常量探针（写 1.0/2.0/3.0 到输出）使输出如期变为常量序列，MERE 由 5.53e+07 降至 1.04e+00 量级，证明分支执行且写通路正常（来源平台：950；证据等级：tool_observed）。
- 轨迹二：tiling 参数回显探针——输出前几个位置读到 `isKEqE_=1`、`k_=64`、`E_=64`、`topkInner_=64`，tiling 传参与分支条件均正确（来源平台：950；证据等级：tool_observed）。
- 轨迹二：kernel 元数据 JSON 的 `sha256` 字段与 `.o` 文件实算 sha256 一致（如 `566679e9...`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "文件名 hash 基于 TilingKey"为 CANN 9.1.0 实测归纳；框架若未来改为内容 hash，规则 1 失效（未验证，推断可迁移）。
- hash 段的精确生成规则（TilingKey 的何种编码、哪些模板参数参与）未逆向验证；结论止于"文件名 hash 与源码内容无关、由模板选择决定"的行为观测。
- 常量探针法要求输出张量可被精度比对观察到（测试程序会打印失配值）；若测试只报 PASS/FAIL 不打印数值，需要临时打印或换探针位置。

**未决项**：

- 文件名 hash 的生成规则（是否等于 TilingKey 编码）可在需要时从 `asc_op_compile_base` 构建脚本逆向确认；当前行为级结论已满足调试需求。
## 坏实践

- 凭"测试输出与改前逐位相同"断言"kernel 没重编/没装上"，进而去删各种猜测中的缓存目录。
- 用 `.o` 文件名 hash 或文件大小（同大小很常见）判断二进制是否变化。
- 见文件名不变就下结论"没重编译"，直接清缓存、改路径、反复重装——绕远路且掩盖真问题（两条轨迹的真问题分别在线程管线同步与上游数据）。
- 在证明"新二进制已安装 + 分支已执行"之前就回退 supposedly 无效的修复。

## 误诊

- 一次被当作"增量构建缓存陈旧"（清缓存无效）、一次被当作"运行时加载旧 kernel"（MD5 对账排除）——本卡本身即是这两类误诊的解毒剂。
- 轨迹二曾假设"编译器把 `if (k_ == E_)` 分支优化掉了"：加入 tiling 字段 `isKEqE` 强制运行时分支后 md5 变化但输出仍相同，说明优化假设不成立——分支本来就在执行，输出相同是因为读到的上游数据就是那份垃圾。
- 轨迹二曾怀疑 vendor 安装覆盖问题：卸载脚本 + 重装 + md5 对比证明安装环节无辜。

## 增量（第三条独立 trace 与 hash 生成规则的脚本级证据）

另一独立轨迹并入第三条独立 trace（AdamW 类算子，CANN 9.2.0，`rm -rf build` 全新构建后文件名仍逐字节相同的复现），并把本卡遗留问题中的 hash 生成规则推进到**构建脚本源码级**：

**1. hash 生成规则的脚本级证据**：`x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_bin_param_build.py`（约 :443）：

```python
objstr = json.dumps(param, indent="  ")          # param = {"op_type":..., "op_list":[{inputs/outputs/attrs...}]}
md5sum = hashlib.md5(objstr.encode("utf-8")).hexdigest()
bin_file = self.op_type + "_" + md5sum
```

即文件名哈希是**算子参数描述 JSON（op_type/inputs/outputs/attrs）的 MD5**——与源码内容无关，同一算子签名下源码怎么改文件名都不变。这把本卡根因段"TilingKey/模板参数选择决定"的推断收精确为"算子参数描述哈希"（参数描述即描述输入/输出/属性布局的 JSON，与 TilingKey/模板选择同源但不等同）。

**2. install 日志语义**：`no need to upgrade ops op_impl files` 是安装脚本对**同名文件**的常规输出，不构成"未更新/安装失败"的判据；需要强保证时对比 build 与 installed 的 sha256（本轨迹逐文件一致）。

**3. PRINTF 插桩变体（与常量写入探针互补）**：kernel 里插 `AscendC::PRINTF("##FIX## ...")`，重编运行后 stdout 出现该串即证明新二进制在跑（本轨迹 PRINTF 输出在 aclnn 进程 stdout 可见）。注意 AscendC::PRINTF 只支持 %d/%f/%x/%s 等基础格式符，`%.10e` 会被截断输出（无精度修饰）——不影响"是否出现"的判定；大量/循环内 PRINTF 会破坏时序，仅作存在性插桩，验证完撤除。

**4. ccache 排除**：`ccache -s` 缓存 0.0 GiB、`ccache -C` 后行为不变——ascendc kernel 编译链没有基于源码哈希的持久 ccache，"同名=缓存命中"假设不成立。

### 复现增量验证信号

- 构建脚本证据：`ascendc_bin_param_build.py` 中 `md5sum = hashlib.md5(json.dumps(param).encode()).hexdigest(); bin_file = op_type + "_" + md5sum`，param 仅含 op_type/inputs/outputs/attrs（来源平台：950；证据等级：tool_observed）。
- 源码改动 + `rm -rf build` 全新构建后三个 `.o` 文件名与修复前完全相同（逐名比对）；build 产物与 vendor 已安装 .o 的 sha256 逐文件一致（来源平台：950；证据等级：tool_observed）。
- `ccache -s` 缓存 0.0 GiB，`ccache -C` 后行为不变（来源平台：950；证据等级：tool_observed）。
- PRINTF 插桩重编运行后 stdout 出现 `[AIV Block 0/1] ##FIX## omb2=%.`——kernel 已按新源码编译执行；同一运行中边界扫描 mismatch 仍为 1/512，证明行为未变是修复无效而非旧二进制（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 脚本级证据取自 CANN 9.2.0 的 `ascendc_bin_param_build.py`；9.1.0 的同名脚本未逐行比对（行为一致：两条 9.1.0 轨迹 + 本 9.2.0 轨迹同名表现相同）。
- PRINTF 可见性依赖运行路径（aclnn 进程 stdout）；TTK 评测下 PRINTF 只进 plog，此时用常量写入探针（本卡规则 3）。
