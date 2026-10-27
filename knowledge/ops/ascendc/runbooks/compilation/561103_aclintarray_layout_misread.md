---
type: "Runbook"
title: 561103 aclIntArray 布局误解致属性读取垃圾值
description: 自定义 op_api 手解 aclIntArray 时按 {size; values[]} 内联布局读值会读到 vtable 与指针垃圾（561103/EZ1001，W_out=-1e17 级垃圾值）；实际布局为 {vtable; int64_t* data; uint64_t size}，可用最小探针程序实测布局后修正。复现增量（同算子二次踩坑形态）：假设存在公共取值函数 aclGetIntArrayValue 直接调用，编译期才暴露（was not declared in this scope）；grep 公共头确认取值面仅 aclGetIntArraySize；repr 读值用于 OP_CHECK 值级校验（strides>0/pads>=0/dilations>0）并须先以 aclGetIntArraySize 校验长度，上板 ST 50/50 无回归。
tags: [task.debugging, topic.memory, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T01:39:38Z'
updated_at: '2026-09-17T04:11:06Z'
platforms: ["950"]
---

# 561103 aclIntArray 布局误解致属性读取垃圾值

## 现象

BenchNoHarnessConv2d 算子安装后首次在 NPU 上跑 ST（43 个 L0 用例，Real 模式），**全部**用例在同一位置失败：

```
[FAIL] GetWorkspaceSize 失败: ret=561103
```

561103 = `ACLNN_ERR_INNER_NULLPTR`（内部异常：出现 nullptr，查 ascendc-runtime-debug 的 error_codes.md）。打开日志后拿到关键 errMsg（来自 InferShape 的参数检查）：

```
errMsg=[PID: 4470] AclNN_Parameter_Error(EZ1001): H_out(1) or W_out(-101169876662129144) <= 0
```

W_out 是 -1e17 量级的垃圾值；且不同用例之间垃圾值只差小常数（如 -101169876662129144 / -101169876662129142），H_out 是 1、2、-2 这类小错值——典型的"从相邻的错误内存位置读到了指针/对齐残渣"而非随机未初始化。当时代码状态：自定义 op_api（`op_api/bench_no_harness_conv_2d.cpp`）里为了把 strides/pads/dilations 传给 l0 算子，手写了一个 `AclIntArrayRepr` 辅助结构按 `{uint64_t size; int64_t values[]}` 内联布局读 `aclIntArray` 的值。

## 适用场景

- 工程形态：自定义算子 aclnn op_api 层需要读取 `aclIntArray*`（ListInt 属性如 strides/pads/dilations）的元素值。
- API 路径：`aclCreateIntArray`/`aclGetIntArraySize`/`aclDestroyIntArray`（`aclnn/acl_meta.h`）；`aclIntArray` 为公开头文件中仅有前置声明的不透明类型。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（x86_64）。
- 适用粒度：same_operator→operator_family（所有需要在 host 侧读 aclIntArray 的自定义 op_api；布局事实本身 platform_wide 于 x86_64 + CANN 9.1.0）。
- 路线优先级：safety_fallback。fallback 声明：能用框架侧属性传递（如 NnopbaseAddIntArrayAttr 路径）或后续版本公开读值 API 时优先框架路径；手解内部布局仅作兜底，代码必须带版本锚点注释，升级/换架构须重跑探针。判别方法与探针程序本身可作为定位此类垃圾值的第一步。
- 明确不适用 / 未验证场景：aarch64 布局未验证；`aclBoolArray`/`aclFloatArray` 等同类不透明类型未逐一探针（机制推断相同，须各自实测）。

## 根因

`aclIntArray` 在公开头文件里只有 `typedef struct aclIntArray aclIntArray;` 前置声明，没有公开的按下标读值 API（只有 create/size/destroy）。实际内存布局（x86_64，CANN 9.1.0 实测）：

```
offset  0: vtable 指针（op::Object 虚表）
offset  8: const int64_t* data（指向值数组，非内联）
offset 16: uint64_t size（元素个数）
```

而错误假设是 `{uint64_t size; int64_t values[]}`（size 在前 + 内联数组）：`repr->values[index]` 实际读到的是 vtable 指针(0x7fe6...)与 data 指针(0x5587...) 的位组合——正好解释垃圾值只差小常数（不同属性数组分配在相邻堆地址）与 561103（垃圾形状参数触发 InferShape 判空失败后 l0 算子返回 nullptr，被 CHECK_RET 报成 ACLNN_ERR_INNER_NULLPTR）。

## 判别方法

判别信号链（全部本会话实测）：

1. 全部用例在同一步失败且错误码是 561103（NULLPTR）而非 161xxx 参数错 → 系统性问题，在 op_api/l0 层而非用例数据；
2. errMsg 里形状/属性值为 -1e17 量级垃圾且用例间只差小常数 → 读到了相邻堆内存（指针被当值读）；
3. `grep -rn "561103" <runtime-debug 参考>/error_codes.md` → `ACLNN_ERR_INNER_NULLPTR 内部异常：出现nullptr异常`；
4. 最小布局探针（下述完整程序）实测确认字段偏移。

**形态二（复现增量，同一算子 aclnn 层二次踩坑）**：为 aclnn 层（op_api 的 `aclnn<Op>` 封装）加属性值校验时，直觉调用 `aclGetIntArrayValue(strides, 0, &strideH)` 读元素，`bash build.sh --soc=ascend950` 直接编译失败：

```text
op_api/aclnn_bench_no_harness_conv_2d.cpp:124:5: error: 'aclGetIntArrayValue' was not declared in this scope; did you mean 'aclGetIntArraySize'?
  124 |     aclGetIntArrayValue(strides, 0, &strideH);
      |     ^~~~~~~~~~~~~~~~~~~
      |     aclGetIntArraySize
```

公共 API 面探针（一句话确认"没有取值函数"）：

```bash
grep -rn "aclGetIntArray" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/
# 仅命中 acl_meta.h: aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size)
```

即本坑有两种触发形态：按错误布局直读（运行期垃圾值 → 561103）与假设存在公共取值函数（编译期失败）；两者的正解同为 repr 布局经 `data` 指针读值。若编译单元里 opdev 头可见，元素访问亦可直接 `(*arr)[i]` + `Size()`（见 [aclIntArray 无 At() 用下标](aclintarray_no_at_use_subscript.md)）；repr 手解适用于不引入 opdev 头依赖的场景。

布局探针（完整可复用，`test_intarray_layout.cpp`）：

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    // Create a known aclIntArray: values = [111, 222, 333, 444]
    int64_t values[4] = {111, 222, 333, 444};
    aclIntArray* arr = aclCreateIntArray(values, 4);

    // Get size via public API
    uint64_t size = 0;
    aclGetIntArraySize(arr, &size);
    printf("Size (via API): %lu\n", size);

    // Dump raw memory (first 64 bytes)
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(arr);
    printf("\nRaw memory dump (first 64 bytes):\n");
    for (int i = 0; i < 64; i += 8) {
        uint64_t val;
        memcpy(&val, raw + i, 8);
        printf("  offset %2d: 0x%016lx (%ld)\n", i, val, (int64_t)val);
    }

    // Try to find where the values 111, 222, 333, 444 are stored
    printf("\nSearching for known values (111, 222, 333, 444)...\n");
    for (int i = 0; i < 128; i += 8) {
        int64_t val;
        memcpy(&val, raw + i, 8);
        if (val == 111 || val == 222 || val == 333 || val == 444) {
            printf("  Found value %ld at offset %d\n", val, i);
        }
    }

    // Also check if there's a pointer to the data
    printf("\nChecking pointer interpretations:\n");
    for (int i = 0; i < 48; i += 8) {
        uint64_t val;
        memcpy(&val, raw + i, 8);
        if (val > 0x1000 && val < 0x7fffffffffff) {
            const int64_t* ptr = reinterpret_cast<const int64_t*>(val);
            printf("  offset %d: ptr=0x%lx -> trying to read values: ", i, val);
            for (int j = 0; j < 4; j++) printf("%ld ", ptr[j]);
            printf("\n");
        }
    }
    aclDestroyIntArray(arr);
    aclFinalize();
    return 0;
}
```

编译命令（注意必须补 `-lnnopbase`，否则 `aclDestroyIntArray` 报 DSO missing）：

```bash
g++ -std=c++17 -O0 -o test_intarray_layout test_intarray_layout.cpp \
  -I${ASCEND_HOME_PATH}/include -L${ASCEND_HOME_PATH}/lib64 \
  -lascendcl -lopapi -lnnopbase -Wl,-rpath,${ASCEND_HOME_PATH}/lib64
```

实测输出（节选）：

```
Size (via API): 4
  offset  0: 0x00007fe6eb0cbfd0   <- vtable
  offset  8: 0x0000558796ae84b8   <- data 指针
  offset 16: 0x0000000000000004   <- size
  offset 8: ptr=0x558796ae84b8 -> trying to read values: 111 222 333 444
```

## 解决配方

修复前（错误的内联布局假设）：

```cpp
// aclIntArray 内部布局辅助访问
// aclCreateIntArray(const int64_t* value, uint64_t size) 创建的内部结构
struct AclIntArrayRepr {
    uint64_t size;
    int64_t values[];
};

static inline int64_t GetIntArrayValue(const aclIntArray* arr, uint64_t index)
{
    const auto* repr = reinterpret_cast<const AclIntArrayRepr*>(arr);
    return repr->values[index];
}
```

修复后（按实测布局读 data 指针）：

```cpp
// aclIntArray 内部布局辅助访问
// aclCreateIntArray(const int64_t* value, uint64_t size) 创建的内部结构
// 实际内存布局（x86_64, CANN 9.1.0）：
//   offset 0: vtable 指针 (op::Object 虚函数表)
//   offset 8: const int64_t* data (指向值数组，非内联)
//   offset 16: uint64_t size (元素数量)
struct AclIntArrayRepr {
    void* vtable;            // offset 0: vtable pointer
    const int64_t* data;     // offset 8: pointer to values array
    uint64_t size;           // offset 16: number of elements
};

static inline int64_t GetIntArrayValue(const aclIntArray* arr, uint64_t index)
{
    const auto* repr = reinterpret_cast<const AclIntArrayRepr*>(arr);
    return repr->data[index];
}
```

改完后重新 `build.sh --soc=ascend950` 出包、`custom_opp_ubuntu_x86_64.run --force` 安装即可生效。

**值级校验用法（复现增量，aclnn 层 OP_CHECK 形态，verbatim）**：repr 读值可直接接 OP_CHECK 做属性值校验；用前先以公共 `aclGetIntArraySize` 校验长度 ≥ 期望下标上限，防越界（`repr->data[index]` 无内建边界检查）：

```cpp
    // strides 值校验：必须 > 0
    int64_t strideH = GetIntArrayValue(strides, 0);
    int64_t strideW = GetIntArrayValue(strides, 1);
    OP_CHECK(strideH > 0 && strideW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides must be > 0, got [%ld, %ld]", strideH, strideW),
             return false);

    // pads 值校验：必须 >= 0
    int64_t padTop = GetIntArrayValue(pads, 0);
    int64_t padBottom = GetIntArrayValue(pads, 1);
    int64_t padLeft = GetIntArrayValue(pads, 2);
    int64_t padRight = GetIntArrayValue(pads, 3);
    OP_CHECK(padTop >= 0 && padBottom >= 0 && padLeft >= 0 && padRight >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads must be >= 0, got [%ld, %ld, %ld, %ld]",
                     padTop, padBottom, padLeft, padRight),
             return false);

    // dilations 值校验：必须 > 0
    int64_t dilationH = GetIntArrayValue(dilations, 0);
    int64_t dilationW = GetIntArrayValue(dilations, 1);
    OP_CHECK(dilationH > 0 && dilationW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations must be > 0, got [%ld, %ld]",
                     dilationH, dilationW),
             return false);
```

## 设计决策

- 为什么用最小探针程序而不是继续翻头文件：`aclIntArray` 的完整类定义不在任何公开头文件里（`common_types.h` 只把它当 opaque 指针用），运行时 dump 是唯一直接证据；用已知值数组 {111,222,333,444} 可以无歧义地定位 data 字段。
- 探针程序为什么先 `aclInit`+`aclrtSetDevice`：`aclCreateIntArray` 依赖运行时初始化。
- 当前方案路线档位：判别与探针方法可作推荐第一步；手解内部布局属 safety_fallback——能用框架侧属性传递（如 NnopbaseAddIntArrayAttr 路径）时优先框架路径，手解仅作兜底且须带版本锚点注释。

## 预防

- 在自定义 op_api 里读不透明 ACL 类型前，先用最小探针程序实测布局（构造已知值 → dump → 指针解引用验证），不要按直觉猜字段顺序。
- 遇到 561103 + errMsg 里属性/形状为巨量垃圾值：优先怀疑"把指针/vtable 当值读"，检查所有 reinterpret_cast 辅助结构。

## 验证信号

以下信号来自该算子联调验证会话：修正布局后重新出包安装并复跑 ST。

- 修复前：43/43 用例 `[FAIL] GetWorkspaceSize 失败: ret=561103`；errMsg `H_out(1) or W_out(-101169876662129144) <= 0`（相邻用例垃圾值差 2）（来源平台：950；证据等级：tool_observed）。
- 探针实测：offset 8 指针解引用得 111/222/333/444，offset 16 为 size=4（来源平台：950；证据等级：tool_observed）。
- 修复后：重新构建安装，ST 全部用例通过 GetWorkspaceSize 进入 kernel 执行/数值比对阶段（后续暴露的 double free 与输出全 0 是另外的根因，各自闭环）（来源平台：950；证据等级：tool_observed）。
- 错误码事实锚点：561103=ACLNN_ERR_INNER_NULLPTR（内部异常：出现 nullptr 异常）（来源平台：950；证据等级：tool_observed）。
- 复现增量·修复前：`bash build.sh --soc=ascend950` 编译错误 `error: 'aclGetIntArrayValue' was not declared in this scope; did you mean 'aclGetIntArraySize'?`（来源平台：950；证据等级：tool_observed）。
- 复现增量·API 面核对：`grep -rn "aclGetIntArray" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/` 仅命中 `acl_meta.h: aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size)`（来源平台：950；证据等级：tool_observed）。
- 复现增量·修复后：算子包构建成功（3 个 kernel 二进制生成），ST 43/43 无回归、全量 50/50 PASS（含 fp32 新用例），非法属性值路径由 OP_LOGE 报 `ACLNN_ERR_PARAM_INVALID`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 布局为 CANN 9.1.0 x86_64 实测；aarch64 与后续版本未验证（未验证，升级/换架构须重跑探针）。
- 依赖内部布局是脆弱接口：CANN 版本变化可能改变字段顺序/新增字段，卡内代码必须带"实际内存布局（x86_64, CANN 9.1.0）"注释锚点。
- `aclBoolArray`/`aclFloatArray` 未逐一探针（机制相同，布局须各自实测）。
- `repr->data[index]` 不查 size，下标越界无内建保护；调用前必须用公共 `aclGetIntArraySize` 校验长度（复现增量）。

**未决项**：

- 框架侧正式读值路径（替代手解布局）未在本轨迹实施；若后续 CANN 提供公开按下标读 aclIntArray 的 API，应整体替换本卡方案（优先项：调研并迁移）。
## 坏实践

- 对不透明类型按 `{size; values[]}` 直觉写 reinterpret_cast 辅助结构（本例垃圾值的直接来源）。
- 假设存在 `aclGetIntArrayValue` 之类公共取值函数直接调用——公共面没有，编译期才暴露（复现增量）。
- 看到 561103 只报"nullptr"就去查空指针入参，不看 errMsg 里的垃圾数值模式。

## 误诊

- 曾沿"l0op::Contiguous / l0op::BenchNoHarnessConv2d / l0op::ViewCopy 哪个返回 nullptr"的方向逐个排查（CHECK_RET 链分析）——真实失败点在属性值被读坏后 InferShape 报错，错误码只是 NULLPTR 的表象。
