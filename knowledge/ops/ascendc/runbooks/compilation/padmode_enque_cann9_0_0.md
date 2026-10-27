---
type: "Runbook"
title: "PadMode/单参 EnQue 在 CANN9.0.0 编译失败"
description: "从其他 CANN 版本迁来的 AscendC kernel 用 AscendC::PadMode::PAD_NONE 与 que.template EnQue<MTE_T> 单参形式，CANN 9.0.0 Ascend950 头文件不提供，build exit 2；判别为 API 形态版本错配，须改 QuePosition 显式三参 + DataCopyParams/DataCopyPadParams。"
tags: ["task.debugging", "topic.quality.compatibility.version_migration", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# PadMode/单参 EnQue 在 CANN9.0.0 编译失败

## 现象

把按其他 AscendC API 版本写就的 `fill_diagonal_v2_dense.h`（dense path）同步到 `ops-math conversion/fill_diagonal_v2/op_kernel/` 后，`build.sh --soc=ascend950 --opkernel --pkg` 编译报错（exit 2）：

```
fill_diagonal_v2_dense.h:61:39: error: no member named 'PadMode' in namespace 'AscendC'
                             AscendC::PadMode::PAD_NONE, 0, 0);
fill_diagonal_v2_dense.h:62:13: error: 'template' keyword not permitted here
        que.template EnQue<MTE_T>(xLocal);
fill_diagonal_v2_dense.h:86:39: error: no member named 'PadMode' in namespace 'AscendC'
```

同仓 built-in（可编译）版 `dense.h` 不报错——它是按 CANN 9.0.0 头文件形态写的。迁来的版本从未在 CANN 9.0.0 上编译过。

## 适用场景

- **工程形态**：AscendC kernel 编译（`build.sh --soc=ascend950 --opkernel --pkg`）。
- **API 路径**：`AscendC::DataCopyPad` / `AscendC::EnQue` / `AscendC::DeQue` / `AscendC::TQueBind` / `AscendC::FreeTensor` / `AscendC::DataCopyParams` / `AscendC::DataCopyPadParams`。
- **验证过的平台与版本**：Ascend950（arch35 / dav-3510），CANN 9.0.0，AscendC kernel 头（`kernel_operator_cache_intf.h` 等）。
- **适用粒度**：`operator_family`——从其他 CANN 版本迁来的 AscendC kernel 代码在 CANN 9.0.0 上编译的 API 形态错配判别。
- **路线优先级**：`recommended_fastpath`——把 API 形态适配到 CANN 9.0.0 的 QuePosition 显式三参 + `DataCopyParams`/`DataCopyPadParams` 拆分搬运。
- **不适用或未验证场景**：built-in 版 `dense.h` 已是 CANN 9.0.0 兼容写法，不需适配；非 CANN 9.0.0 环境不适用本判别（更高版本若恢复了 `PadMode` 或接受单参 `EnQue`，形态可能不同）。

## 判别方法

- **触发（trigger）**：迁来的 AscendC kernel 头在 CANN 9.0.0 上 `build.sh --soc=ascend950 --opkernel --pkg` 报三类编译错误并 exit 2：`no member named 'PadMode' in namespace 'AscendC'`、`'template' keyword not permitted here`（出现在 `que.template EnQue<MTE_T>(...)` / `que.template DeQue<MTE_T>()` 行）、`no member named 'PadMode'` 重复出现在 CopyOut 的 `DataCopyPad` 调用。tool_observed（编译器输出）。
- **探针（probe）**：对报错文件做形态检查，命中以下任一即指向本判别：① grep `PadMode::PAD` 命中（CANN 9.0.0 命名空间下无此枚举）；② grep `que.template EnQue<` 或 `que.template DeQue<` 命中且模板参数只有一个（非 `QuePosition::GM, QuePosition::VECIN, MTE_T` 三参形式）；③ grep `TQueBind<AscendC::VECIN, AscendC::VECOUT>` 命中且为两参旧式枚举（非 `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM>` 三参 scoped 形式）。可执行检查：在同一头文件中 grep 到 `PadMode::PAD` 或 `template EnQue<`/`template DeQue<` 单参形式即命中。
- **观测（observation）**：触发态编译器输出 exit 2，报 `no member named 'PadMode'`、`'template' keyword not permitted here`（tool_observed，编译 stderr）。把上述三类形态改为 CANN 9.0.0 形态后，同一 `build.sh --soc=ascend950 --opkernel --pkg` exit 0、`Build binary success`、`build/binary/ascend950/bin/` 产出 `.o`（tool_observed，build stdout）。
- **判读（interpretation）**：根因是 AscendC API 形态版本错配，非 kernel 逻辑错误。迁来的头按另一 CANN 版本写就，用了三处 CANN 9.0.0 头文件不提供的 API 形态：(1) `AscendC::PadMode::PAD_NONE` 枚举在 CANN 9.0.0 AscendC 命名空间下不存在，原写法想用 5 参 `DataCopyPad(dst, src, count, PadMode::PAD_NONE, 0, 0)` 表达"按 count 精确搬运不 pad"，CANN 9.0.0 改用 `DataCopyParams` + `DataCopyPadParams`（aligned `DataCopy` + tail `DataCopyPad` 拆分）表达同一语义；(2) `que.template EnQue<MTE_T>(xLocal)` / `que.template DeQue<MTE_T>()` 单模板参数 + `template` 关键字形式被拒，CANN 9.0.0 要求 `EnQue<QuePosition::GM, QuePosition::VECIN, MTE_T>` / `DeQue<QuePosition::VECOUT, QuePosition::GM, MTE_T>` 三参 QuePosition 显式形式；(3) `TQueBind<AscendC::VECIN, AscendC::VECOUT>` 两参旧式枚举须改为 `TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>` 三参 scoped 形式；(4) `FreeTensor<MTE_T>` 模板参数须去掉改 `FreeTensor(xLocal)`。同仓 built-in 版 `dense.h` 的等价写法用 `DataCopyParams`/`DataCopyPadParams` + `DataCopy`（aligned）+ `DataCopyPad`（tail）拆分搬运、`EnQue`/`DeQue` 用 `QuePosition` 显式三参，可编译——形态 diff 即是判别依据。三类形态改齐后 build exit 0，反证根因为 API 形态错配。

## 验证信号

来源：本次 fill_diagonal_v2 TTK 重跑会话，迁来的 dense.h 在 CANN 9.0.0 上首次编译及 API 形态适配后重建。

- 修复前 `build.sh --soc=ascend950 --opkernel --pkg` exit 2，`no member named 'PadMode' in namespace 'AscendC'`、`'template' keyword not permitted here`（来源平台:Ascend950；证据等级:tool_observed）
- API 形态改齐后同命令 exit 0、`Build binary success`、`build/binary/ascend950/bin/` 含 `.o` 产物（来源平台:Ascend950；证据等级:tool_observed）

## 失效边界

- 仅 CANN 9.0.0 + Ascend950 实测。更高 CANN 版本若恢复了 `PadMode` 枚举或支持单参 `EnQue`/`DeQue`，本判别的触发条件不再成立；`QuePosition` 显式三参是更严格形式，推断可向上兼容但未验证，标注"未验证，推断可迁移"。
- 本卡只判别 API 形态错配这一根因，不承载适配后的完整 `dense.h` 实现。
- 同根因的精度现象（dense path 多核 `DataCopy` 写回整 tile 触发 L2 缓存驱逐）不在本卡判别范围，须另查对应精度卡。
