---
type: Guide
title: 二进制算子测试的完整输入与失败传播
description: 从 Add 与麒麟样例提取文件长度、dtype、输出覆盖和 ACL 错误传播约束。
tags:
- task.testing
- topic.implementation.debugging
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_host/data_utils.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_host/add_custom.asc
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/gen_data.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/golden.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/verify_result.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/test_torch.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/data_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/main.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/scripts/gen_data.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/kirin_add_template/scripts/verify_result.py
  role: supporting
created_at: '2026-09-11T02:49:27Z'
updated_at: '2026-09-11T02:49:27Z'
platforms: []
---

# 二进制算子测试的完整输入与失败传播

裸二进制文件不携带shape/dtype/stride，应将预期逻辑元素数、表示类型与字节数显式绑定。两份文件长度相等不足以证明数据完整：双方同样截断也可能比较通过。读取需验证预期字节数、实际读取量和容量，失败后不能继续将未初始化Host缓冲区拷到设备。

主Add读取工具核对文件大小，但调用侧忽略返回值。麒麟ReadFile允许短于buffer的文件，且用引用参数修改inputByteSize；随后第二输入读取及两次H2D使用这个被改变的大小，而Kernel totalLength未改变，可能产生未初始化尾部。它也未核验sgetn返回量。应将预期大小与实际读取量分开，分别验证每个输入，不能让一个文件改变其他输入或Kernel合同。

麒麟CHECK_ACL只打印错误并继续。任一分配、复制或stream创建失败后继续使用指针都可能覆盖最早错误；错误宏的名字不能证明失败已传播。部分资源申请失败还需释放先前成功资源，输出拷回/释放必须排在所需设备完成之后。文件写入需检测短写并向最终退出状态传播，不能仅在日志打印后返回0。

主Add为FP32、麒麟为FP16，生成器先把输入转目标dtype，再按相应数学表达式生成golden；比较应基于实际已表示输入。主Add文件比较和Torch测试容差不同，麒麟使用equal_nan=True且允许错误比例<=0.001，均是样例条件，不自动成为算子验收标准。N=16384时该比例可放过16个错误元素。

比较前独立核对预期元素数和shape，避免NumPy广播、空输出除零或max空数组异常。非有限值需按合同分类，不能让abs差值的NaN导致错误统计漏计。麒麟打印循环按错误元素的全局index>=100停止，并非“最多100条错误”；摘要仍须保留总错误数与真正最差位置。误差公式见[实际表示与验收](../precision/error_metrics_and_represented_test_inputs.md)。

小随机样例、零值与相反数不能验证整段写满。使用每tile/每核可区分输入、预先标记输出、尾部边界以及必要的特殊值，可检测重复首块和缺写；预初始化的排队顺序本身也须正确。本批未执行ACL或设备精度测试。
