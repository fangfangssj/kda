# Warp函数

- [Lane ID类函数](lane_id_functions/index.md)：共 6 张知识卡（如 Laneid、LanemaskEq、LanemaskGe 等）。
- [Warp函数](warp_functions_intro.md)：Warp是SIMT线程执行和调度的基本单位。一个线程块内的线程会按照线性线程号被划分为多个Warp，当前一个Warp包含32个线程。同一Warp内的线程执行相同的指令流，但每个线程拥有独立的寄存器和Lane ID，可处理不同的数据地址和分支路径。
- [Warp Reduce类函数](warp_reduce_functions/index.md)：共 3 张知识卡（如 asc_reduce_add、asc_reduce_max、asc_reduce_min）。
- [Warp Shfl类函数](warp_shfl_functions/index.md)：共 4 张知识卡（如 AscShfl、AscShflDown、AscShflUp 等）。
- [Warp Vote类函数](warp_vote_functions/index.md)：共 4 张知识卡（如 asc_activemask、asc_all、asc_any 等）。
