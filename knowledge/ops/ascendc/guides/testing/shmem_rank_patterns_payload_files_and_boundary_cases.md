---
type: Guide
title: SHMEM Rank 特征、文件长度与真实边界测试
description: 补齐输入可辨识性、BF16 编码、per-PE oracle 和不整除/资源复用的实际覆盖。
tags:
- task.testing
- topic.precision
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/correctness.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/test-structure-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/testcase-scale-standard.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-style.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/agent-execution-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/fused-compute-phase-gates.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-review/references/review-report-template.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM Rank 特征、文件长度与真实边界测试

Rank pattern 可定位来源 PE/token/hidden，但编码必须在域内唯一且被存储 dtype 精确表达。来源 (pe+1)*1000000+token*1000+h 仅在各分量取值界限足够小、总值不溢出时可解码；h≥1000 或 token跨量级会碰撞，FP16 也无法承载这些整数。pe+arange*1e-4 随长度仍无界且可能舍入成重复值，不能当成永远安全的替代。

生成器应选择与 dtype 相容的有界数据，再加入能识别 PE、tile、epoch 的模式。FP16 有限上界约65504，超过它造成 golden/output 同 Inf 不证明通信错误或正确。np.bfloat16 不是可假设所有 NumPy 版本具备的标准属性，BF16 原始文件需明确编码/解码依赖；int8 randint 上界1000超出dtype范围，模板不可原样执行。随机 seed之外保存最终输入、配置与生成器身份。

输出合同决定 oracle：replicated 对应所有目标 PE 完整结果，sharded 对应各 PE slice，owner-only 只要求 owner 的定义输出；子 team 测试还应确认非参与者规定状态。每个PE贡献可辨识，多轮signal/state复用可暴露错期，但rank pattern通过不独立证明所有对称分配顺序/大小相同，还需layout记录及实际访问范围证据。

模板 gen_data 写全零 golden.bin，而 per-PE checker 读取 golden_pe{pe}.bin，并且run命令漏了gen_data必需M/N参数；这些是未补完的示意，不是可运行完整测试。所有输入与golden必须按实际local compute→communication→finalize生成，FP32中间量是选择之一，不能宣称消除所有精度误差或自动对齐原实现。

fromfile(count=expected) 会忽略尾部多余字节，短文件可能少读；比较前需核对实际文件字节数、元素数、dtype/endianness与shape，防止广播比较及陈旧文件误通过。实际输出和golden各自的identity应绑定case/PE/本轮制品；文件名只有PE仍需独立case目录。0xFF sentinel 对整数可能为-1，对浮点可能为NaN，不是任意dtype的-1；只检查合同定义的有效区和guard，不能要求未定义padding任意固定值。

边界必须通过运算确认：来源 N=1000,P=8 得125无余，(1000,1024)在P=4也不体现PE余数；可用N=1001,P=8构造真实非均衡，并按实际工作单位/合法域确认。标为tail_k的(513,257,128)若Ktile=128并没有K尾。非2幂维不必然非对齐，near-UB要按同时存活buffer、各份数、metadata和实际引擎scratch计算，不能照搬190KiB±5KiB。

通信规模按每PE元素数×dtype字节数，再分别报告总量与实际远端payload。Matmul同时看M/N/K、2MNK FLOPs和通信字节；512²是262144元素，约0.5MiB FP16，并非0.5M元素。4096²*2048*2约68.7GFLOPs，不能把模板模糊量级当精准资源预算。性能分类标签也不免除该输入的正确性。

测试包括小于/等于/超过chunk、空分工、PE/核不整除、输出容量、signal epoch回绕/复用、各引擎路线及多输出。2/4/8PE、XS/S/M/L、20/12条或固定repeat次数是来源计划参数，不证明覆盖充分；dtype改变元素宽度、对齐或分支时，对应边界不能仅由主dtype代验。
