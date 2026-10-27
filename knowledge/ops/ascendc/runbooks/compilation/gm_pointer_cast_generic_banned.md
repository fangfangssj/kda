---
type: "Runbook"
title: "CCE 禁止 __gm__ 指针转 void*，打印用 uint64_t"
description: aicore 函数内 __gm__ 指针 cast 到 void* 或 reinterpret_cast 到普通指针均被 CCE 拒绝；打印地址用 (uint64_t) + 0x%lx，做偏移用 GM_ADDR 直接指针算术。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# CCE 禁止 __gm__ 指针转 void*，打印用 uint64_t

## 现象

在调试一个 ascend950（arch35）自定义算子的 workspace 指针问题时，在 kernel 入口（`__global__ __aicore__` 函数）里用 `AscendC::PRINTF` 打印 `workspace` / `user` / `GetSysWorkSpacePtr()` 等指针地址，最初按主机习惯写 `%p` + `(void*)` 强转，编译直接失败：

```
error: casting '__gm__ uint8_t *' (aka '__gm__ unsigned char *') to type 'void *' is not allowed in aicore function
```

另一处在 kernel 里对 workspace 参数做字节偏移时写了 `reinterpret_cast<uint8_t*>(workspace) + N`，同样编译失败：

```
error: reinterpret_cast from '__gm__ uint8_t *' (aka '__gm__ unsigned char *') to 'uint8_t *' (aka 'unsigned char *') is not allowed
```

两处都是 CCE（bisheng 前端）对 `__gm__` 地址空间指针的强类型限制：GM 指针带 `__gm__` 地址空间限定，不允许 cast 到通用地址空间类型（`void*` / 普通 `T*`）。

## 适用场景

- 工程形态：AscendC kernel 编写/调试（build.sh `--soc=ascend950`，CCE/bisheng 编译）。
- API 路径：`AscendC::PRINTF`、`GM_ADDR` / `__gm__` 指针、`AscendC::GetSysWorkSpacePtr`。
- 验证过的平台与 CANN 版本：Ascend950（arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有用 CCE 编译的 aicore kernel 代码。
- 路线优先级：recommended_fastpath（合规写法唯一且简单）。
- 明确不适用 / 未验证场景：主机侧代码（op_host / 测试程序）无此限制；`ASCENDC_CPU_DEBUG` 宏管 CPU 调试路径的写法未在本开发会话 验证。

## 根因

CCE（bisheng）编译器对 aicore 函数内的 `__gm__` 地址空间指针禁止 cast 到通用地址空间类型（`void*`、普通 `T*`）——地址空间限定是类型系统的一部分，丢失限定的转换直接拒绝编译。

## 解决配方

**场景 1：打印 GM 指针地址**——先转 `uint64_t` 再用 `0x%lx`：

```cpp
    // 诊断：输出指针地址，验证 workspace 是否有效
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::PRINTF("DBG ptrs: workspace=0x%lx user=0x%lx sysWs=0x%lx tiling=0x%lx\n",
            (uint64_t)workspace, (uint64_t)user, (uint64_t)AscendC::GetSysWorkSpacePtr(), (uint64_t)tiling);
    }
```

（修复前为 `AscendC::PRINTF("... workspace=%p user=%p sysWs=%p tiling=%p\n", (void*)workspace, (void*)user, (void*)AscendC::GetSysWorkSpacePtr(), (void*)tiling);`——四处 `(void*)` 全部触发上述编译错误。）

**场景 2：对 GM 指针做字节偏移**——不经过普通指针，`GM_ADDR` 直接指针算术：

```cpp
    // 用户 workspace = workspace 参数 + RESERVED_WORKSPACE(2MB)
    constexpr int64_t RESERVED_WORKSPACE_SIZE = 2 * 1024 * 1024;
    GM_ADDR user = workspace + RESERVED_WORKSPACE_SIZE;
```

（修复前为 `GM_ADDR user = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint8_t*>(workspace) + RESERVED_WORKSPACE_SIZE);`——内层 `reinterpret_cast<uint8_t*>(workspace)` 触发上述编译错误。）

## 验证信号

修复来自同一次 kernel 指针调试，两个场景各自独立闭环：

- 场景 1 修复后编译通过，且上板打印出实值 `DBG ptrs: workspace=0x12004d600000 user=0x12004d600000 sysWs=0x12004c600000 tiling=0x120000200030`（来源平台：950；证据等级：tool_observed）。
- 场景 2 修复后 `Build completed successfully!` 包生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `(uint64_t)` 转换只用于打印/传参显示；不要用读回的整数当指针用。
- `GM_ADDR` 直接算术成立的前提是 `GM_ADDR` 本身为 `__gm__ uint8_t*` 类型的别名；若自定义类型需自行确认。
- 本开发会话 只覆盖 `void*` 与 `uint8_t*` 两个目标类型；其他通用类型目标按同一限制类推（未逐一验证）。
- 相邻边界：反向（非 `__gm__` 限定指针 cast 成 `__gm__ T*`）同样被拒，见 [aicore 禁止 const char* cast 到 __gm__ T* 卡](aicore_const_char_cast_gm_t.md)；host 侧 void*↔GM_ADDR 的行为见 [GM_ADDR host侧cast卡](gm_addr_cast_from_void_uses_toolchain_cast.md)；helper 形参保留 GM 地址空间见 [Device helper 必须保留 GM 地址空间卡](device_helper_preserves_gm_address_space.md)。

**未决项**：

- 无（编译器行为已由两次独立闭环验证）。
## 预防

- kernel 内调试打印 GM 地址一律 `(uint64_t)` + `0x%lx`，不写 `(void*)` / `%p`。
- 对 GM 指针做偏移直接写 `gmPtr + offsetBytes`，不要绕道普通指针。
