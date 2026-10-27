---
type: Guide
title: 注册 ST 比较器的失效方式与覆盖缺口
description: 保留 MERE/MARE 公式，纠正 BF16 编码、NaN 漏检、Mock 与跳过成功。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/test_aclnn_add_example.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/torch/compare.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/torch/golden.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/st/torch/test.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/tests/st/test_aclnn_matmul_blaze_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/tests/README.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/st-test-guide.md
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# 注册 ST 比较器的失效方式与覆盖缺口

来源Add浮点判据为MERE=mean(abs(actual-golden)/(abs(golden)+eps))<threshold且MARE=max同一误差<10*threshold；其FP16/BF16/FP32阈值分别2^-10/2^-7/2^-13，Python把FP64也配为2^-13。这是来源配置，不是所有算子的统一验收标准；整数应按实际dtype的值与必要位语义精确比较，随机/量化类还需对应数学合同。

C++用eps=1e-7并在double计算，Python先flatten().float()再用原dtype的finfo.tiny作为eps，二者并非同一指标。FP64输入转FP32可能丢掉差异、溢出或令极小eps下溢为0，NaN/Inf应按显式合同分类，不能只靠相减/比较。shape、dtype、device与元素数应先核对，flatten会掩盖同元素数不同shape，广播运算可能把错形状也算出误差。

Python空浮点Tensor的max/argmax会失败，C++先判null再size0与合法空data指针需明确。C++ MERE遇NaN通常使通过条件false，但错误定位的rel_err>threshold可能不打印该元素；Matmul只更新maxRel/maxAbs，std::max(0,NaN)可保留0，最终maxRel<ATOL可能误报PASS。显式检查预期/新增非有限值是比较器正确性的一部分。

Matmul的RTOL=2^-7、ATOL=10*RTOL，但实际只检查maxRel<ATOL，没有使用maxAbs门限、MERE或allclose形式，变量名不能代表真实判据。FloatToBf16把高16位截取后在低位有残余时OR1，属于保留奇数尾位的jam/round-to-odd式编码，并非通常最近偶数舍入；golden最终也用此函数，会改变输入与输出数值合同。FP32顺序累加也不等于任意Cube累加次序。

Add Mock直接output=golden，只验证部分测试接线，无法证明NPU实现。C++ Golden自测结果只打印未影响最终退出，Pythongolden_passed也未计失败；Python所有NPU用例skip时总失败0仍退出0，必须独立报告有效执行数与跳过原因。创建新Tensor或固定100ms冷却不能修复异步内存生命周期问题。

来源Add所谓大shape只有32/64元素，全部内置case≤64，未覆盖N>1024双缓冲及多tile；numeric_limits<float>::min是最小正正规而非最小次正规。±0数值比较未检查符号位，现有case也没覆盖空/rank0、广播/非连续、非法dtype、INT32溢出与NaN/Inf。Python测试实际有shape断言，C++仅比两输入vector长度，还需与shape乘积和dtype字节宽度一致。

Matmul10个case覆盖16–256方阵、M/N/K=1与M/N非16整齐尾，但缺大K多级资源、K尾、转置true、无效参数、极端值和完整dtype合同。workspace申请、运行、同步与D2H返回值在来源部分忽略；失败后不能拿旧/未初始化输出作比较。资源创建半途失败需回收已经取得的对象，不能以最后输出存在替代所有阶段成功。

Golden自测还应让比较器接受明确正确例并拒绝独立构造错误例，包括NaN漏检、shape不符和dtype窄化。文档目录声称gen_golden/UT存在不证明实际已提供或执行；本卡提取静态审查结论，未宣称模板上板结果。
