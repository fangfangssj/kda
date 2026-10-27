# control flow

- [同一Warp内避免分支跳转](avoid_warp_branch.md)：同一Warp内线程应避免分支跳转以防止分支发散，减少掩码屏蔽等待，提升SIMT算子执行效率。

- [Warp分支发散predicate优化](branch_handing.md)：【优先级】中
