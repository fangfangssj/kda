---
type: "Runbook"
title: "Stub tiling padding 不解 aclnn Segfault"
description: "Ascend950/CANN 9.1.0 下 aclnn 最小通路 Segfault（exit 139）。假设根因是 kernel stub 注册的 tiling 结构体过小（opParaSize=20 < host 下发 328B），padding stub 到 552B（opParaSize=560）后 Segfault 依旧；该路线对 Segfault banned，真因是 host 侧对 device-resident value 标量做 std::memcpy。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Stub tiling padding 不解 aclnn Segfault

## 现象

mask_fill 算子（MaskedFill）Branch-0 Tiling 实现后，aclnn 最小调用通路用例 Segfault（exit=139=128+SIGSEGV），Task 27 时该用例 PASS、Task 28 后回归。定位中发现 kernel 二进制注册的 tiling 容量与 host 下发不匹配：kernel stub 用 `GET_TILING_DATA_WITH_STRUCT(MaskFillTilingData, ...)` 注册了仅 12B 的 `MaskFillTilingData`（`totalLength/blockLength/tileLength` 各 4B），构建产物 kernel JSON 的 `opParaSize=20`；而 host 侧 `DoTilingAndSet<4>` 调 `context->GetTilingData<MaskedFillTilingData<4>>()` 写入 328B 的真实 tiling 结构体。`328 > 20` 这一容量不匹配被假设为 Segfault 根因。

## 适用场景

- **工程形态**：Kernel 直调 + 多分支 tiling（ops-registry-invoke-glacier），kernel stub 阶段与真实 host tiling 并存。
- **API 路径**：kernel 侧 `GET_TILING_DATA_WITH_STRUCT` / `REGISTER_NONE_TILING`；host 侧 `gert::TilingContext::GetTilingData<T>()`（内部 `GetCapacity() < sizeof(T)` 检查）；kernel JSON `opParaSize` 字段。
- **验证平台与 CANN 版本**：Ascend950 / CANN 9.1.0。
- **适用粒度**：`operator_family`——任何 TDD 流程下「kernel 先 stub、host tiling 后真实」的 AscendC 算子。
- **路线优先级**：`banned`——把「stub tiling 结构体 padding 到 ≥ host 最大下发 sizeof」当作 Segfault 的单独修复路线在本场景被实测否定；padding 仍是必要的容量契约（否则 `GetTilingData` capacity 检查失败退化为 exit 255），但不是 Segfault 的根因修复。
- **不适用或未验证场景**：kernel 与 host tiling 结构体始终同名同尺寸的单分支算子不触发容量不匹配；a2/a3 未在本 trace 上板验证。

## 失败路线

按「opParaSize 容量不匹配 = Segfault 根因」的假设，把 stub `MaskFillTilingData` padding 到 `kMaskFillStubTilingBytes=552`（≥ `max(sizeof(MaskedFillTilingData<8>))=552`），使注册容量 `opParaSize=560` 覆盖 host 侧任意分支下发的字节数（`banned` 路线）：

```cpp
// mask_fill_tiling_struct.h — padding 尝试（试图单独修复 Segfault）
constexpr uint64_t kMaskFillStubTilingBytes = 552;

struct MaskFillTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    // 接口桩 padding：使 kernel 注册的 tiling 容量 ≥ host 下发字节数。
    uint8_t  _stub_pad[kMaskFillStubTilingBytes - 12];
};

static_assert(sizeof(MaskFillTilingData) == kMaskFillStubTilingBytes,
              "MaskFillTilingData stub must be padded to kMaskFillStubTilingBytes");
```

padding 后构建通过（`BUILD_EXIT=0`、`MAKE_EXIT=0`）、kernel JSON `opParaSize=560 ≥ 328` 使 `GetTilingData` 的 capacity 检查通过，但 aclnn 运行时 Segfault 依旧（`EXIT=139`）。padding 路线作为 Segfault 的单独修复被实测否定（`banned`）。

机制链（为何 padding 不解 Segfault）：padding 只保证 `GetTilingData<T>()` 的 `GetCapacity() < sizeof(T)` 检查不再返回 nullptr、设备 tiling GM 缓冲区不再被 host 写溢出；但本例 Segfault 并非发生在 tiling 下发，而是 host 侧 `DoTilingAndSet<4>` 对 value 标量做 `std::memcpy` 时，value tensor 的 placement=`kOnDeviceHbm`（设备地址），host 直接 memcpy 设备地址 → Segfault。padding 不触及这条 memcpy 路径，故 Segfault 不变。

边界：padding 仍是必要的容量契约——若不 padding，`GetTilingData` capacity 检查失败会退化为 `GRAPH_FAILED` → `GetWorkspaceSize` error → exit 255（graceful 报错，非 Segfault），用例不会 PASS。即 padding 把「容量不足导致 exit 255」这一独立问题修掉了，但并未、也不能单独修复 Segfault。Segfault 的真因修复是 host 侧 value 标量读取加 `IsOnDevice` 守卫、只 memcpy host-resident 标量（属另一修复路线）。

## 验证信号

- padding 后 `BUILD_EXIT=0`、`MAKE_EXIT=0`、aclnn 运行 `EXIT=139`（Segfault 依旧，未变）（来源平台：950；证据等级：tool_observed）。
- Task 27（commit 8519f1e0）aclnn 最小通路 exit 0 PASS、Task 28 后回归为 exit 139，回归窗口明确（来源平台：950；证据等级：tool_observed）。

## 坏实践

- 见到 aclnn Segfault（exit 139）+ kernel JSON `opParaSize < host tiling sizeof`，就直接断定「容量不匹配是 Segfault 根因」并只做 padding——本例实测 padding 后 Segfault 不变，真因是 host 侧 memcpy 设备地址。须先区分 Segfault 发生在 host 侧（tiling/value 下发）还是设备侧（kernel 执行）。
- 为「让 opParaSize 小」把 host tiling 结构体砍小——host 下发字段是 kernel 计算所需，砍字段等于删数据；应 padding stub 而非削 host。