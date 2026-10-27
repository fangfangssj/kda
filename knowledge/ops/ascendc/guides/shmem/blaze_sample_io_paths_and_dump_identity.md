---
type: Guide
title: Blaze-SHMEM 文件 IO、工作目录与转储身份
description: Blaze-SHMEM 文件 IO、工作目录与转储身份的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/utils.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/io_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/gen_data.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_comm_udma.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 文件 IO、工作目录与转储身份

main实际用src/utils.h版本ReadFile/WriteFile：读只拒绝空/大于buffer文件，不验证精确长度或sgetn完整，caller忽略bool；写open无O_TRUNC且if(!fd)错误拒绝fd0、未直接识别fd-1，短写只打印仍return true。旧大文件尾可污染校验，run.sh删除旧npu_out只是临时规避，不能替代精确I/O和错误传播。

common/host_utils/io_utils.h是另一个更严版本：ReadExactFile验证长度，WriteFile使用O_TRUNC并循环处理短写/EINTR，GetExampleIoPaths按exe目录定位；不能因它存在就把这些保证归给main。common parser严格全数字/正值/u32检查，而main stoi接受数字前缀、out_of_range未被其invalid_argument catch覆盖，同名工具层的实际使用必须追到include和重载。

gen_data把input/output写当前工作目录；run.sh使用绝对脚本路径却未cd PROJECT_ROOT，main读exe旁build/input symlink→项目input，verifier读项目output。若从其他cwd运行run.sh，造数可能落错目录并误读旧输入。输入/output symlink可对齐路径但不证明文件属于当前参数/版本，应绑定shape、rank、dtype、大小和运行身份。

precision转储从shmemSpace+1MiB取A，再加m*k取scale；真实设备data从对称base+ringId*bufferBlock，scale从base+B*bufferBlock。来源没有给用户区1MiB偏移，不能混用HCCL A5的1MiB约定；dump不同于真实slot。ring只保留每槽最近轮次，self区未填，不能把m*k连续dump当全量AllToAll输出。按实际layout、轮次和owner转储才可诊断。
