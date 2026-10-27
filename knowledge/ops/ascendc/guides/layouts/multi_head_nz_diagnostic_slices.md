---
type: Guide
title: 多 Head NZ 拼接的独立数据与 Scale 步长
description: 保留 dtype 切换和逐 stage 定位细节，避免将 NaN 图样当成唯一根因。
tags:
- task.debugging
- topic.data_movement.layout_transform
- topic.compute.quantization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/case-studies.md
  role: primary
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-09-11T02:40:45Z'
platforms: []
---

# 多 Head NZ 拼接的独立数据与 Scale 步长

多矩阵拼入同一L1 NZ区后再加载L0时，每个head起点由完整物理布局确定。对一个32B的C0载体，FP16/FP8/FP4分别容纳16/32/64个逻辑元素；这只是该载体的位宽换算，实际M/K分形、padding、head交错与打包跨度仍要参与寻址。来源g*mEff*C0公式只在其特定拼接布局下成立，不可当任意K的通式。

Scale可能以B16视图搬运并有独立scaleK_b16/分形宽度，应独立推导g*物理scale步长，而非复用数据C0。DataCopy/LoadData参数以载体元素、字节或分形单位解释时要逐项转换，ReinterpretCast不会完成数值转换或增加容量。数据与scale仍需在同一head/K块上对应。

输出按head出现[OK,BAD,BAD,OK]或交替NaN可提示周期性stride/槽位问题。单head通过、多head失败也会同时改变并行、buffer容量和同步，所以不能直接锁定拼接；全部head失败也不能排除某个共同拼接偏移错。常量scale替换仅是一次受控干预，改变数值范围后现象变化不是唯一根因证据。

逐stage记录已经完成生产的对应head中间值，找到最早偏差，并对照实际数据/scale地址区间及加载参数。修一处后只恢复部分head可能表示另一路地址也错，也可能是还有独立缺陷。验证不同head使用可区分数据与scale、非零N/K段、多head/单head和尾块；来源全部恢复的案例未在本批复现，现有[MX矩阵坐标](mx_data_and_scale_coordinates.md)的完整地址边界仍适用。
