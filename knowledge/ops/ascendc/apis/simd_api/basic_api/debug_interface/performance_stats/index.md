# 性能统计

- [MarkStamp](mark_stamp.md)：推荐使用asc_mark_stamp接口进行特定位置的标记，该接口同时适用于C语言和C++语言编程。 用户通过调用接口，用于在算子执行过程中标记特定位置，便于后期通过流水图分析代码执行路径与性能热点。
- [MetricsProfStart](metrics_prof_start.md)：设置性能数据采集信号启动，与MetricsProfStop配合使用，用于指定msOpProf调优的kernel侧代码段范围。
- [MetricsProfStop](metrics_prof_stop.md)：设置性能数据采集信号停止，与MetricsProfStart配合使用，用于指定msOpProf调优的kernel侧代码段范围。
