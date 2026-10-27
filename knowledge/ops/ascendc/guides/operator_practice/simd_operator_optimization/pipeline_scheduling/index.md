# 流水编排

- [开启DoubleBuffer](enable_double_buffer.md)：【描述】执行于AI Core上的指令队列主要包括如下几类，Vector指令队列（V）、Cube指令队列（M）、Scalar指令队列（S）和搬运指令队列（MTE1/MTE2/MTE3）。不同指令队列间的相互独立性和可并行执行特性，是DoubleBuffer优化机制的基石。
- [启用Iterate或IterateAll异步接口避免AIC/AIV同步依赖](enable_iterate_avoid_sync.md)：MIX场景下推荐使用Iterate或IterateAll异步接口，减少Cube与Vector核间消息交互，降低通信开销。
