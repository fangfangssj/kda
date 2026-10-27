---
type: Guide
title: 卷积 fallback 搜索回退与资源单位
description: 保留L0/L1分块策略，纠正无合法最小块时继续返回参数。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/convolution/fallback/script/conv_tiling.py
  role: primary
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# 卷积 fallback 搜索回退与资源单位

该Python脚本是公式复写参考，文件头声称复现Conv2dBaseTiling/ApiTiling并不能证明与实际SDK逐分支等价。默认L1 512KiB、L0A/B64KiB、L0C/UB256KiB、BT/FB4KiB、AIV:AIC2和带宽系数4仅为输入默认值。FP16/BF16的示例MKN=16/16/16，FP32=16/8/16、INT8/HF8=16/32/16应保留dtype差异，不能把所有枚举类型当已实现能力；INT4被按1B估而未表达半字节pack，未知字符串默认FP16也不安全。

L0检查按对齐后的M*K*dA*2、K*N*dB*2、M*N*4分别约束A/B/C；候选来自通道/空间块数约数与2的幂的并集，不是所有值必须整除。扩大M/N或H/W/N后应保存最后一个合法状态。源码回退依据变化后的大小比较决定减哪个轴，不记录最后实际增长轴，HW回退也没有回退H；一次回退不足以证明合法。CPU把各资源设1B仍返回16×16×16最小块且资源检查false，说明候选下限也必须验证，无解要失败而非返回。

输出Ho=floor((Hi+Pt+Pb-dh*(Kh-1)-1)/sh)+1，Wo同理。Hout块跨度对应输入跨度(hoTile-1)*sh+(Kh-1)*dh+1，并需按真实块起点裁剪padding；只min(跨度,Hi)是长度上界模型，不提供地址或两端padding。M模式跨输出行估算时应依Wo，源L0粗估用Wi而L1用Wo，stride/padding下可不一致。

L1可比较双侧全载、B全载/A流式、A全载/B流式、仅一侧K全载、双K全载、双侧部分加载，按复用方向选M-first/N-first。单缓冲全载与双缓冲流式分别计数，bias/scale/转换/innerBatch同时计入。源M分支的可行性门槛有时用单份最小量，后续helper却默认双份，kBL1全载的一些helper只算一份B却保留pb_bl1=2，需对最终字段重算峰值。

源kAL1FullL1直接将包含Kh*Kw的K乘输入空间，遗漏除Kh*Kw使A预算单位不一致；HW的max尺寸已算双份、全载却返回单份，可能过保守；all-full门槛只加bias_min却声明bias_full，又可能漏全量bias。none-full可能返回K0并在后续取余除0。固定LOAD3D起点32767等按实际接口校核，预算通过后仍需检查描述符、偏移和有效尾。
