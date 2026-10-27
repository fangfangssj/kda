---
type: Guide
title: Host 字符串、格式化与复制长度
description: 纠正跨平台整数格式表，保留临时字符串寿命、destMax 和异步日志的实际风险。
tags:
- task.debugging
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-secure.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/cpp-general.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/mc2-specific.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# Host 字符串、格式化与复制长度

printf 风格调用需要解析完整宏展开后的格式串和变参，包括相邻字符串拼接、%%、* 宽度/精度及多行参数。类型/数量/顺序不匹配可能造成未定义行为；不是所有 ABI 都表现为“后续参数全部错位”。额外未消费参数通常不会被格式化，但可能漏掉关键诊断数据；不足或错型仍需修正。

int64_t/uint64_t 可能是 long 或 long long，不能一律要求 %lld/%llu，也不能一律要求 %ld/%lu。使用 <cinttypes> 的 PRId64/PRIu64，或显式转换为与说明符匹配的类型。size_t 用 %zu；bool 在可变参数中提升为 int，可配 %d；对象指针配 %p 时转换为 void*。Device PRINTF 的支持格式需另核对，不能宣称没有此类风险。

%s 需要有效且满足接口要求的字符串指针，nullptr 或释放后地址不可依赖库的容错显示。std::string 不能由空 C 字符串直接构造。c_str()/data() 返回借用视图，可在所有者存活且无使其失效的修改期间使用；临时 string 的 c_str() 在同步调用的完整表达式内一般仍有效，跨表达式保存或异步日志稍后消费则需拥有副本。

通信 groupName 若会延迟消费，应由能活到消费结束的对象持有；复制成短命局部 std::string 同样不够。写入定长 tiling 字符数组需用真实容量检查长度并保证结束符，不能硬编码一个假定长度。包含 NUL 的业务字节串与 C 字符串有不同长度语义。

复制长度同时受源可读范围、目标可写容量、偏移和整数计算约束；destMax 应是真实目标剩余容量，不能为了通过安全库检查填写更大值。memcpy_s/memmove_s 等安全函数仍须正确参数并处理失败，不能只靠 _s 后缀。普通 memcpy 在满足范围/对象/不重叠等条件时也不是天然错误；是否统一安全库属于工程选择。

清零敏感数据需覆盖真正持有副本的存储，并确认所选清除机制不会被优化掉；Host/Device 副本、寄存器和缓存不自动同时清除。此知识仅描述实际敏感数据场景，不把普通 shape、dtype 诊断值视为秘密，也不引入每个算子统一清零要求。
