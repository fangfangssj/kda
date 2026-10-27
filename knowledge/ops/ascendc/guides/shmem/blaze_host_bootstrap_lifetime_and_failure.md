---
type: Guide
title: Blaze-SHMEM Host Bootstrap 生命周期与失败传播
description: Blaze-SHMEM Host Bootstrap 生命周期与失败传播的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/acl_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM Host Bootstrap 生命周期与失败传播

每rank子进程ACL init/setDevice(rank)/stream，再test_set_attr与UDMA engine、SHMEM init、对称分配，输入拷贝后launch，完成再free/finalize和释放stream/device。deviceId=rank仅支持对应连续可用设备；IPPORT必须同会话且可达，多机不能原样用127.0.0.1。1GiB对称堆及allocation同大不代表已经通过容量检查。

test_set_attr的default_flag_uid按值传参，却把&default_flag_uid存到attributes.comm_args，函数返回后该指针悬空；caller的defaultFlagUid也未初始化。INIT_WITH_DEFAULT是否实际忽略comm_args需查对应库，不能因样例可能运行就把这个地址生命周期写法推广为UID方案。应使用真实初始化且活到调用完成的对象，检查helper返回及其他attr字段；option_attr默认MTE又被caller改UDMA是实际配置链。

ACL_CHECK只打印、不return/throw，许多malloc/memcpy/sync失败后程序继续且run返回0，父进程success不证明所有API成功。SHMEM init有return -1但未统一清理已建ACL资源；align也没判空。相反common AclRtSession用状态位和析构清stream/device/runtime，可覆盖Init过程中异常展开，但样例main未使用该RAII类，不能据仓中存在它宣称已修复。

父进程记录PID并waitpid检查退出状态，但waitpid错误/EINTR未处理，fork中途失败会退出而未统一取消回收已启动workers，其他rank可卡在集体操作。需要把局部失败传播到本次所有参与者并按已成功资源阶段清理。

源注释“子进程在main return会继续fork”错误：从main return会终止该进程；真正危险的是子分支执行完run函数却没有退出/return而落回fork循环。exit当前实现有效，但不是唯一合法方式。批量退出与同步不能用未限定进程范围的清理命令替代。
