---
type: Guide
title: Vector repeat 分批必须同时满足计数与步长位宽
description: 对实际使用 uint8_t repeatTime 的重载，把大 tile 分为不超过 255 次的批次，并独立检查 repeat stride
  和地址偏移。
tags:
- task.implementation
- topic.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-repeat-limits.md
  role: primary
created_at: '2026-09-10T12:27:52Z'
updated_at: '2026-09-10T12:27:52Z'
platforms:
- agnostic
---

# Vector repeat 分批必须同时满足计数与步长位宽

当目标 Vector 重载的 `repeatTime` 实参最终转换为 `uint8_t` 时，正数计数的可表示上限是 255。把 256 传入这样的参数会转换为 0；上游变量使用 uint32_t 并不能避免调用边界的窄化。

有两种保持完整覆盖的组织方式：Host 将每块 repeat 数限制在接口允许范围内，或 Kernel 对同一块继续分批。每批完成后按已处理的行数和真实行步长推进地址；被广播的源操作数则按该重载的广播语义保留对应地址。

分批仅解决计数范围。`BinaryRepeatParams` 中各步长的位宽、单位和允许值必须单独检查，不能在未验证范围前直接转换为 uint8_t。还要核对尾批 mask、最后一行地址以及偏移乘法的整数范围。

该结论只适用于实际参数类型和接口约束匹配的重载。count 型 API、RegBase VF API 和不同版本的签名不能按 API 名统一套用 255；较宽整数参数也仍可能受到容量、指令或合法取值限制。

可用 254、255、256 和跨多批的行数验证覆盖，并结合非整 VL 列数检查尾部。这里只说明参数表示与分批逻辑，不声称已进行上板验证。[Sub 接口](../../apis/simd_api/basic_api/memory_vector_compute/basic_arithmetic/sub.md)提供对应运算入口。
