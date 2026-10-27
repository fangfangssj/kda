---
type: Guide
title: 量化 TilingKey 字段与示例修正
description: 保存多格式路由与后处理要求，纠正1420和六位编码。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: primary
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 量化 TilingKey 字段与示例修正

以下数值仅是来源各算子版本的Host/Kernel协议，不能跨算子复用。编码必须覆盖合法dtype/模式、字段不越界无碰撞，未知值拒绝，且每个可达key对应真实构建分支。以key分发不等于所有判断都零运行时开销，也不保证不同shape自动得到最优算法。

| 算子 | 来源编码/分支 | 核验边界 |
|---|---|---|
| DynamicBlockQuant | 1000*roundMode+100*input+10*output+kernel；round1/4/7，input FP161/BF162，output E5M20/E4M3FN1/HIFLOAT82，kernel normal0/single1/large2 | half→HIFLOAT8、round、normal应4120，原文1420错；其他INT8路径未在此表定义，不能默认输出0 |
| DynamicMXQuant | input千位1/2，output百位0..3，场景十位0/1/2/4，odd scale个位0/1 | 1000/2000为FP4E2M1示例，1001带odd后处理；96+数量未由省略代码证明 |
| GroupedDynamicMX | 十位FP161/BF162、个位E4M3FN1/E5M22 | 11/12/21/22；不可混用DynamicBlockQuant千位round含义 |
| DynamicQuantUpdateScatter | 0BF16、1half、2double-buffer BF16、3double-buffer half | 13/11为来源不同UB成本因子，不是UB字节数；else接half前须先验证dtype |
| FakeQuantAffineCachemask | FP321、FP162 | switch缺default时可能遗留key与bytesPerData，须明确失败 |
| DequantBias | 示例10100与10113、activate_scale/bias/bias dtype维度 | 两例不足以反推全部位数及16合法组合，真实完整表决定 |

DynamicMX奇数scale可能需DynamicMxQuantPost重排/补齐。若跨核消费主阶段GM结果，普通函数先后调用只约束本核，需全体生产完成的发布协议；不因类名NotTailAxis但ISTAIL=true就判路由错误，应按模板参数实际语义。奇偶指的是scale布局条件，不自动等于输入最后维奇偶。

GroupedMX为提取指数把FP16先转BF16可能改变边界指数，需rounding与shared exponent算法匹配，不能称自动无损；数据本体量化路径仍应按实际高精度值。Tiling routing不能单独证明数值链正确。

RMSNormQuant代码依次拼gemma、beta、highPrecision、useSlice四个布尔位，再左移BIT_OFFSET_SIX后加dtype。若该常量6，则key=((gemma*8+beta*4+precision*2+slice)<<6)+dtype，是最多10位结构，不是全文所称6bit总编码；dtype必须<64且是约定ID，不可无校验直接加入任意ge枚举。16种布尔组合还要乘合法dtype和其他编译维度，不代表总共只有16个实例。

useSlice由numCol>sliceSize判断，sliceSize须正且与已计算UB一致；gemma/beta开关影响数学语义，不能为了走fast key修改。source某些转换链实际上三次Cast，标题“两次”不能作为指令成本依据。固定2x、30–100%收益不从分支数量推导。
