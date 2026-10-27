---
type: "Runbook"
title: "OpAttrDef 无 SetDefault/RequiredListInt"
description: "ops::OpAttrDef 没有 SetDefault/RequiredListInt 等方法；必选属性用无参类型重载（ListInt()/Int()），可选属性用带默认值的重载（ListInt(vector)/Int(1)/Bool(true)）。按方法名直觉拼接即报 has no member named。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# OpAttrDef 无 SetDefault/RequiredListInt

## 现象

两个独立算子工程在同一 API 契约上以两种形态撞到同一根因。

形态一（默认值链式追加，CANN 9.2.0）：给 GRU 类算子写 OpDef（op_host/<op>_def.cpp）时，按"先设类型再设默认值"的直觉给属性写默认值：

```cpp
this->Attr("numLayers").Int().SetDefault(1);
this->Attr("bias").Bool().SetDefault(true);
this->Attr("batchFirst").Bool().SetDefault(false);
this->Attr("dropout").Float().SetDefault(0.0);
this->Attr("bidirectional").Bool().SetDefault(false);
```

执行 `bash build.sh --soc=ascend950 -j8` 后 npu_op_code_gen 阶段编译失败，5 处同一报错：

```text
op_host/<op>_def.cpp:72:39:
error: 'class ops::OpAttrDef' has no member named 'SetDefault'
   72 |         this->Attr("numLayers").Int().SetDefault(1);
      |                                       ^~~~~~~~~~
```

随后 `make: *** [...<op>_def.o] Error 1`、`RuntimeError: compile ascend all ops failed`，CMake Configuring incomplete。

形态二（必选拼接，CANN 9.1.0）：Conv3D filter 梯度反向算子的必选 ListInt 属性 filter_size 按 "Required + ListInt" 的直觉拼出：

```text
op_host/<op>_def.cpp:55:35:
error: 'class ops::OpAttrDef' has no member named 'RequiredListInt'
   55 |         this->Attr("filter_size").RequiredListInt();
      |                                   ^~~~~~~~~~~~~~~
```

同轮输出还伴随一条容易带偏注意力的 note（与另一独立问题相关）：`op_def.h:180:50: note: initializing argument 1 of 'ops::OpParamDef& ops::OpParamDef::DataType(std::vector<ge::DataType>)'`。

## 适用场景

- 工程形态：自定义算子仓库（registry-invoke）op_host 侧算子定义注册（`*_def.cpp` 的 `this->Attr(...)` 链式调用），构建入口 `cmake -DASCEND_COMPUTE_UNIT=ascend950` / build.sh。
- API 路径：`ops::OpAttrDef`（头文件 `$ASCEND_HOME_PATH/x86_64-linux/include/register/op_def.h`）的 `Int/Bool/Float/String/ListInt/...` 类型方法重载集。
- 验证过的平台与 CANN 版本：ascend950（DAV_3510），CANN 9.1.0 与 9.2.0（两个独立工程分别实测；形态二验证方式为 x86_64 宿主机交叉编译产出算子包）。
- 适用粒度：operator_family（OpAttrDef 属性注册 API 行为跨算子一致）。
- 路线优先级：recommended_fastpath。
- 不适用/未验证：`REG_OP`（op_graph/IR 侧 graph/operator_reg.h）的属性默认值写法（`ATTR(x, Int, 1)` 一类，必选须用 REQUIRED_ATTR，见对应卡）不在本卡；ListFloat/ListBool/ListListInt 等同族方法未逐一验证；其他 CANN 版本的方法集未验证。

## 根因

`ops::OpAttrDef` 没有 `SetDefault`/`RequiredListInt` 等方法；必选与可选由"是否传默认值"区分——无参 `ListInt()`/`Int()` 即必选，带值重载 `ListInt(std::vector<int64_t> value)`/`Int(int64_t value)` 即可选（声明 + 默认值一步完成）。头文件实测签名（两会话 grep 到的原文）：

```text
324:class OpAttrDefImpl;
325:class OpAttrDef {
331:  OpAttrDef &AttrType(Option attr_type);
332:  OpAttrDef &Bool(void);
333:  OpAttrDef &Bool(bool value);
334:  OpAttrDef &Float(void);
335:  OpAttrDef &Float(float value);
336:  OpAttrDef &Int(void);
337:  OpAttrDef &Int(int64_t value);
338:  OpAttrDef &String(void);
339:  OpAttrDef &String(const char *value);
...
367:  OpAttrDef &ListInt(void);
368:  OpAttrDef &ListInt(std::vector<int64_t> value);
...
```

## 解决配方

形态一修复（默认值作为类型方法参数；无默认值的属性保持无参重载；Float 取 float 字面量 0.0f 匹配 `Float(float)`）：

```cpp
// 属性
this->Attr("inputSize").Int();
this->Attr("hiddenSize").Int();
this->Attr("numLayers").Int(1);
this->Attr("bias").Bool(true);
this->Attr("batchFirst").Bool(false);
this->Attr("dropout").Float(0.0f);
this->Attr("bidirectional").Bool(false);
```

形态二修复（必选属性用无参 `ListInt()`，可选属性带默认值）：

```cpp
// 属性：strides(3), pads(6), dilations(3), groups(默认1), filter_size(5, 必选)
// 必选属性用无参 ListInt()/Int()；可选属性带默认值
this->Attr("strides").ListInt();
this->Attr("pads").ListInt();
this->Attr("dilations").ListInt();
this->Attr("groups").Int(1);
this->Attr("filter_size").ListInt();
```

先探 API 再改代码（两会话实际执行的核对入口）：

```bash
grep -n "ListInt\|class OpAttrDef\|OpAttrDef &" \
  $ASCEND_HOME_PATH/x86_64-linux/include/register/op_def.h
```

## 设计决策

- 选择"无参重载=必选、带值重载=可选"这一既定契约语义，而不是按其他框架（`Required()` 链式 / `SetDefault`）的直觉拼方法名。
- `groups` 是可选属性且默认 1，因此用 `Int(1)`（带默认值重载）而非无参 `Int()`；`inputSize`/`hiddenSize` 业务上必填，保留 `.Int()`。
- `dropout` 在 aclnn 接口层是 `double`，但 OpDef 侧 `Float(float)` 只收 float，故写 `0.0f` 而不是 `0.0`（避免隐式窄化告警）。

## 预防

- 写 OpDef 属性注册前先 grep 头文件确认方法集；判别式：无参重载=必选；带默认值重载=可选。
- 出现 `has no member named 'SetDefault'/'Required*'` 一律先查 `register/op_def.h`，不要继续拼同族方法名。

## 验证信号

来源说明：两个独立算子工程的 OpDef 属性注册编译修复闭环（ascend950，CANN 9.1.0 与 9.2.0）。

- 形态一修复前：5 处 `error: 'class ops::OpAttrDef' has no member named 'SetDefault'`，构建在 npu_op_code_gen 阶段失败（来源平台：950 编译流水线；证据等级：tool_observed）。
- 形态一修复后：改为构造参数写法后，同流水线构建中 `SetDefault` 错误全部消失（剩余错误转为其它独立问题），后续多轮构建直至 host 侧全部通过均未再出现该错误（来源平台：950 编译流水线；证据等级：tool_observed）。
- 形态二修复前：`error: 'class ops::OpAttrDef' has no member named 'RequiredListInt'`（_def.cpp:55）（来源平台：950 交叉编译；证据等级：tool_observed）。
- 形态二修复后：重新 `cmake -DASCEND_COMPUTE_UNIT=ascend950 ..`，`RequiredListInt` 报错消失（该轮剩余报错均为其他独立根因）；最终 `bash build.sh --soc=ascend950 -j8` 完整通过，产出 ascend950 kernel 二进制与 `custom_opp_ubuntu_x86_64.run` 算子包（来源平台：950 交叉编译；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.1.0/9.2.0 的 `ops::OpAttrDef` 方法集；其他 CANN 版本需重新核对 op_def.h（未验证，推断同机制）。
- 仅验证 ListInt/Int/Bool/Float 四类属性；ListFloat/ListBool/ListListInt 的"无参=必选"规律为推断，未逐一验证。
- 形态二验证为交叉编译产出算子包，未上板。

**未决项**：

- 无。
## 坏实践

- 按方法名直觉拼接 API（`RequiredListInt`、`SetDefault`、`RequiredAttr` 之类）而不查头文件——报错信息只会说 "has no member"，不会告诉你正确写法。
- 把同轮输出的 `DataType(std::vector<ge::DataType>)` note 当作本错误的原因去改 `.DataType({...})` 写法——该 note 在形态二工程中是另一独立问题（dtype 枚举名错误）的线索。

## 误诊

- 无（两个工程均一次定位到 API 契约层；报错信息直指不存在的方法名）。
