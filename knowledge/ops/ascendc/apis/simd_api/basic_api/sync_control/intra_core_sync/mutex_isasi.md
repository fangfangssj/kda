---
type: API
title: Mutex(ISASI)
description: Mutex用于核内异步流水指令之间的同步控制，其功能类似于传统CPU中的锁机制。通过先锁定再释放指定流水来完成流水间的同步依赖，锁定/解锁指定流水对应Lock/Unlock接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Mutex_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Mutex(ISASI)<a name="ZH-CN_TOPIC_0000002565915364"></a>

Mutex用于核内异步流水指令之间的同步控制，其功能类似于传统CPU中的锁机制。通过先锁定再释放指定流水来完成流水间的同步依赖，锁定/解锁指定流水对应Lock/Unlock接口。

- [Lock](lock.md)：根据MutexID获取Mutex，若Mutex已被锁定，将阻塞指定流水上后续指令的执行，直到所有流水中具有相同MutexID的Mutex都已经被[Unlock](unlock.md)释放。
- [Unlock](unlock.md)：直到当前流水的前序指令执行完成后，根据MutexID释放对应Mutex。

**图 1**  Lock/UnLock接口<a name="fig58242463299"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/lock_unlock.png "Lock-UnLock接口")

每个锁有固定的一个MutexID，在不同编程范式中，该ID的获取以及释放方式不同：

- 采用[TPipe-TQue框架编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/tpipe_tque_paradigm.md)时，MutexID需要通过[AllocMutexID](alloc_mutex_id_isasi.md)/[ReleaseMutexID](release_mutex_id_isasi.md)进行申请释放，由框架统一管理MutexID分配记录，避免与TQue等资源管理接口使用的MutexID冲突。
- 采用[静态Tensor编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)时，MutexID可以通过[AllocMutexID](alloc_mutex_id_isasi.md)/[ReleaseMutexID](release_mutex_id_isasi.md)申请释放，也可以由开发者自行管理。自行管理可减少接口调用开销，但开发者需要自行保证MutexID不与其他同步或资源管理接口使用的MutexID冲突。自行管理MutexID时，建议使用0-27，28-31为系统内部规划预留，不建议使用。

相对于[SetFlag/WaitFlag](set_flag_wait_flag_isasi.md)同步机制，使用Lock/Unlock接口有以下优势：

- 内聚性更强，使用时与其它流水线解耦，可以简化反向同步，如图2所示，以PIPE\_MTE2与PIPE\_V之间同步为例。
- 可以使用更多的同步信号量。

**图 2**  SetFlag/WaitFlag同步与Lock/Unlock同步代码差异<a name="fig27101823165111"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/setflag_waitflag_vs_lock_unlock_diff.png "SetFlag-WaitFlag同步与Lock-Unlock同步代码差异")

Mutex类定义如下：

```cpp
class Mutex {
template <pipe_t pipe>
static __aicore__ inline void Lock(MutexID id);
template <pipe_t pipe>
static __aicore__ inline void Unlock(MutexID id);
};
```
