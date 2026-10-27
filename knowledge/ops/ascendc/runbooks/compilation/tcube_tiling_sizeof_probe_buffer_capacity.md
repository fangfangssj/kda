---
type: "Runbook"
title: "宏生成 tiling 结构容量须实测 sizeof"
description: "自定义 TilingData 用定长 int32 raw buffer 承载 optiling::TCubeTiling 时，BEGIN_TILING_DATA_DEF 宏生成的结构带元数据/对齐分量，真实体积（实测 1816B）远超按字段数心算的 256B，static_assert 编译期拦截；先跑 sizeof 探针再定容量。"
tags: [task.implementation, topic.tiling, topic.memory, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# 宏生成 tiling 结构容量须实测 sizeof

## 现象

在为 GRU 类自定义算子（MIX 模式，Cube Matmul + Vector element-wise，arch35/Ascend950，CANN 9.2.0）搭建骨架时，host 侧 tiling 里需要把 `MatmulApiTiling` 算出的两个 `optiling::TCubeTiling`（Gi/Gh 两次 Matmul）随自定义 `BenchNoHarnessGruTilingData` 下发到 kernel。当时的做法是在 TilingData 结构里放定长 raw buffer：

```cpp
#define TCUBE_TILING_BUF_SIZE 64  // 64 * sizeof(int32_t) = 256 bytes
...
int32_t mmGiTiling[TCUBE_TILING_BUF_SIZE];
int32_t mmGhTiling[TCUBE_TILING_BUF_SIZE];
```

host 侧用 `memcpy_s` 把 `optiling::TCubeTiling` 拷进 buffer，并配了防御性断言：

```cpp
static_assert(sizeof(optiling::TCubeTiling) <= TCUBE_TILING_BUF_SIZE * sizeof(int32_t),
              "TCubeTiling size exceeds buffer");
```

执行 build.sh 全量编译时报：

```
error: static assertion failed: TCubeTiling size exceeds buffer
```

## 适用场景

- 工程形态：算子仓自定义算子开发（op_host tiling + op_kernel），host 计算 Matmul tiling 后随自定义 TilingData 下发 kernel。
- API 路径：`adv_api/matmul/matmul_tiling.h` 的 `MatmulApiTiling::GetTiling(optiling::TCubeTiling&)`、`BEGIN_TILING_DATA_DEF` 系宏、`memcpy_s`、自定义 TilingData 结构。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有需要在 kernel 侧拿到完整宏生成 tiling 结构的算子（不限于 GRU）。
- 路线优先级：recommended_fastpath（先量后配是正路；拍脑袋估容量即踩坑）。
- 明确不适用 / 未验证场景：用 `TCubeTiling` 直接结构体成员下发（GET_TILING_DATA_WITH_STRUCT 反序列化整结构）的方案不在本卡范围；aarch64 环境、CANN 9.2.0 之外的版本未验证，宏展开布局可能随版本变化。

## 根因

`optiling::TCubeTiling` 由 `BEGIN_TILING_DATA_DEF` 宏族生成，除肉眼可见的约 40 个 `int32_t` 字段外还带宏注入的元数据，真实体积是 1816 字节（实测），远超按字段数心算的 256 字节。容量按"字段数 × 4B"估算必然偏小，static_assert 在编译期拦截。

机制链：`BEGIN_TILING_DATA_DEF` 展开的类 ≠ 聚合字节数之和（有对齐/元数据分量）→ `sizeof(optiling::TCubeTiling) = 1816` → `1816 > 64*4 = 256` → `static_assert` 失败。

## 判别方法

不要猜容量，直接用宿主编译器实测 sizeof（该头可独立于 kernel 编译，x86 g++ 即可编译运行）：

```bash
cat > /tmp/check_tcube_size.cpp << 'EOF'
#include "adv_api/matmul/matmul_tilingdata.h"
#include <cstdio>
int main() {
    printf("sizeof(optiling::TCubeTiling) = %zu\n", sizeof(optiling::TCubeTiling));
    return 0;
}
EOF
g++ -I$ASCEND_HOME_PATH/x86_64-linux/asc/include -I$ASCEND_HOME_PATH/x86_64-linux/include /tmp/check_tcube_size.cpp -o /tmp/check_tcube_size && /tmp/check_tcube_size
```

实测输出：

```
sizeof(optiling::TCubeTiling) = 1816
```

再按 `ceil(1816/4) = 454` 向上取整并留余量定 buffer 容量（本例取 512）。

## 解决配方

before（op_kernel/arch35/bench_no_harness_gru_tiling_data.h，容量按字段数心算，编译期拦截）：

```cpp
#define TCUBE_TILING_BUF_SIZE 64  // 64 * sizeof(int32_t) = 256 bytes
```

after（按实测 1816B 取 512×4=2048B 容量）：

```cpp
#define TCUBE_TILING_BUF_SIZE 512  // 512 * sizeof(int32_t) = 2048 bytes (>1816 actual size)
```

host 侧 tiling 的 memcpy_s 与断言保持不变（断言本身是正确的防线，错在容量数字）：

```cpp
static_assert(sizeof(optiling::TCubeTiling) <= TCUBE_TILING_BUF_SIZE * sizeof(int32_t),
              "TCubeTiling size exceeds buffer");
memcpy_s(tiling->mmGiTiling, sizeof(tiling->mmGiTiling), &mmGiTiling, sizeof(optiling::TCubeTiling));
memcpy_s(tiling->mmGhTiling, sizeof(tiling->mmGhTiling), &mmGhTiling, sizeof(optiling::TCubeTiling));
```

## 设计决策

- 为什么保留 raw buffer + static_assert：骨架阶段 kernel 侧用 `reinterpret_cast` 从 buffer 读 TCubeTiling，改动面最小；static_assert 保证容量不足时编译期暴露而不是运行期踩内存。
- 放弃的候选：直接把 `TCubeTiling` 作为 TilingData 结构成员（CANN 内置 GRU 的做法）——当时被 host/kernel 两套类型的内存布局是否一致卡住，骨架阶段先走 raw buffer；后续崩溃调试确认直接成员 + `static_assert(sizeof(optiling::TCubeTiling) == sizeof(TCubeTiling))` 是更优路线。
- 当前方案属 recommended_fastpath 下的保守形态：容量 = 实测 sizeof 向上取整留余量，不引入额外假设。

## 预防

- 任何"宏生成的 tiling 结构 × 定长 buffer"组合，先跑一次 sizeof 探针再定容量；把探针命令留在仓库里，CANN 升级后重跑（宏展开可能变化）。
- code review 时对"注释里写 = N bytes"的宏保持怀疑，要求注释里的数字来自实测输出。

## 验证信号

问题与验证来自一次 GRU 自定义算子骨架编译会话（Ascend950PR，CANN 9.2.0）：

- 修改前全量编译报 `error: static assertion failed: TCubeTiling size exceeds buffer`（来源平台：950；证据等级：tool_observed）。
- sizeof 探针输出 `sizeof(optiling::TCubeTiling) = 1816`（来源平台：950 宿主编译；证据等级：tool_observed）。
- 容量改 512 后重新全量编译，该 static_assert 消失，编译推进到后续（新的、与本断言无关的 std::max/模板实例化错误）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅证明"容量 ≥ 实测 sizeof 时编译通过"，未证明 raw buffer 方案在 kernel 侧 reinterpret_cast 读取的正确性（该方案随后在 NPU 上崩溃，属另一独立问题）。
- 1816B 是 CANN 9.2.0 + 该头文件的实测值，不同版本/不同 TCubeTiling 变体体积不同，禁止把 1816/512 当常量复用，须重跑探针。
- 未验证，推断可迁移：aarch64 工具链下同一头文件的 sizeof 理论一致（同一 ABI 宏展开），但未实测。

**未决项**：

- 无（本根因已闭环到编译通过；raw buffer 方案自身的运行期问题另行记录）。
## 坏实践

- 按字段个数 × 4 字节心算宏生成结构的体积。
- 为消掉断言直接删 static_assert 而不扩容——那是把编译期防线改成运行期越界。

## 误诊

- 无（报错信息与探针输出一次定位）。
