---
type: Guide
title: 直调 Add 的分核覆盖与 tile 地址
description: 从模板多 tile 重复读写反例提取完整分核、尾块、整数宽度和资源约束。
tags:
- task.implementation
- topic.memory
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_kernel/add_custom_kernel.asc
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_kernel/add_custom_tiling.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_host/add_custom.asc
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kernel_launch_details.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/add_custom.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/add_custom_tiling.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/main.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/README.md
  role: supporting
created_at: '2026-09-11T02:49:27Z'
updated_at: '2026-09-11T02:49:27Z'
platforms: []
---

# 直调 Add 的分核覆盖与 tile 地址

令逻辑元素数N>0、可用核数C>0、tile长度L>0。来源按T=ceil(N/L)、Q=ceil(T/C)、B=ceil(T/Q)得到实际启动核数，每核标准跨度S=Q*L，末核有效量N-S*(B-1)。这些公式要求所有加法/乘法可表示；可用商和余数实现ceil以免N+L-1溢出。N=0需独立处理，否则可能除零或在tileNum-1处下溢。

核b负责起点b*S，核内第i块起点为b*S+i*L，count=min(L,该核有效量-i*L)。所有输入和输出都必须使用同一个核内offset。add_custom模板循环没有把i传入CopyIn/CopyOut，因此多块时反复访问每核第一块。N=8193、C=1、L=4096时实际读写区间为[0,4096)、[0,4096)、[0,1)，而[4096,8193)未写；只有单块测试会掩盖缺陷。

麒麟实际源码已采用progress*TILE_LENGTH，L=1024、N=16384、C=1时共16块。README中旧TILE_NUM/固定等分讲解与实际Tiling结构不一致，应以共用头、Host填值与Kernel消费三者为准。其普通DataCopy路径不能因对齐样例通过就扩展为任意非对齐tail；按具体重载的长度/地址限制选非对齐搬运或独立尾部实现，不向GM有效范围外补读写。

主Add模板Host跨度为uint64而Kernel total/tileNum为uint32，2^32+4096会截成4096；应在转换前验证上界或一致拓宽。Tiling共享结构还需核对字段顺序、对齐和padding，不能假定C++结构是紧密打包。麒麟字段均uint32也仍须防止ceil与b*S中间溢出。

主模板三条队列各2个4096元素float buffer，共98304字节；麒麟三条队列各2个1024元素half buffer，共12288字节，均未计其他资源。TQue模板深度1不等于只能分配1个物理buffer。double buffer的容量、生产消费依赖和实际重叠须分别成立，两个event ID也不能自动产生两份存储。主模板搬运字节数转uint16时当前4096*4可表示，扩大tile后须重新验证。

手动双槽流水须包含首次可复用状态、每轮生产/消费依赖、覆盖前等待与最后消费者完成；来源启动指南只有局部Wait/Set骨架，不能据此证明完整流水无覆盖。验证N在L及C*L两侧、每核多tile、非对齐尾部、空输入约定和宽整数边界，并检查输出每个逻辑元素恰好覆盖。本批只执行Host整数覆盖演算，未编译或上板。
