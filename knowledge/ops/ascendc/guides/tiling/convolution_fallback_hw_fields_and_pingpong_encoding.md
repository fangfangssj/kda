---
type: Guide
title: 卷积 fallback HW字段与乒乓标志编码
description: 记录通道/K混用、innerBatch和标志位的CPU反例。
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

# 卷积 fallback HW字段与乒乓标志编码

M模式用single_core_ho/ho_l1/ho_l0存线性Ho*Wo方向长度、wo字段0，HW模式这些字段才是独立空间轴。M的kAL1通常包含Kh*Kw归约展开，HW helper却给kAL1通道数、kBL1仍为通道*Kh*Kw。共享_compute_cin_params无条件将二者除Kh*Kw，HW默认Ci64、3×3会把kAL1=64转为cinA=7；full-load key也用Ci*Kh*Kw比较kAL1，导致全载判断偏差。不同模式必须明确单位并分开转换。

k_step使用k_l0/n0而不是明示K粒度，FP32的k_l0=8、n0=16可得0，不能把非零n/k块验证当所有派生字段可用。A空间函数在M模式按整个Hi*Wi分配，前面的L1选择可能只预算局部输入行；最终a_l1_space_size与资源模型必须相同。尾cin先取最后有效K再除Kh*Kw，仅在K按完整kernel空间切分时有意义。

buffer编码依次移位最终为BL0 bit4、AL0 bit3、BL1 bit2、AL1 bit1、bit0保留0；解码却用0x18>>3当L1、0x03当L0。CPU遍历AL1/BL1单双四组合，解出的L1恒3、L0仅0或2，而源实际L0始终双份；这不是可照抄的TilingKey ABI。应对每个标志独立定义位域并做编码→解码往返验证，groups的额外映射也要覆盖。

innerBatch按L0剩余空间求上界却未限single_core_batch；CPU batch1、1×1小卷积返回innerBatch64。若字段表示同时处理batch数，应限真实batch并处理尾；若仅物理容量则消费者不能按64个有效batch读。has_bias被无条件设1且bias dtype由输入类型猜测，CLI没有独立bias/scale/output dtype合同，需补实际输入属性再分配。

打印dilation固定1×1、CLI padding默认1而dataclass默认0、示例-h/-w与真实-hi/-wi不同，都说明日志/入口默认值不能替代真实配置。_validate_tiling仅查几个字段非0，不检查容量、尾、标志或覆盖；compute_tiling主要只拒绝非正输出，stride0、group整除、非法dtype、负尺寸和资源仍需前置验证。CPU反例不等于已在NPU运行。
