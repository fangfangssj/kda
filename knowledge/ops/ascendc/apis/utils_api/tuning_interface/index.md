# 调测接口

- [__trap](_trap.md)：在SIMT代码中调用此接口会中断算子的运行，适用于Kernel侧异常场景的调试。
- [asc_dump](asc_dump.md)：将对应内存上的数据打印出来，同时支持打印自定义的附加信息（仅支持uint32_t类型的信息），比如打印当前行号等。使用该接口需要包含"utils/debug/asc_dump.h"头文件。该功能主要用于调试和性能分析，通常在调测阶段使用，会对算子性能产生一定影响，生产环境可以通过设置ASCENDC_DUMP=0的方式关闭。
- [asc_mark_stamp](asc_mark_stamp.md)：SIMD场景下，用户通过调用接口，用于在算子执行过程中标记特定位置，便于后期通过流水图分析代码执行路径与性能热点。
- [asc_prof_start](asc_prof_start.md)：SIMD场景下，用于设置性能数据采集信号启动，和asc_prof_stop配合使用。使用msOpProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc_prof_start和asc_prof_stop来指定需要调优的代码段范围。
- [asc_prof_stop](asc_prof_stop.md)：SIMD场景下，设置性能数据采集信号停止，和asc_prof_start配合使用。使用msOpProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc_prof_start和asc_prof_stop来指定需要调优的代码段范围。
- [AscTimeStamp](asc_time_stamp.md)：SIMD场景下提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点，便于调试和性能分析。
- [Assert](assert.md)：SIMD和SIMT调试场景下提供assert断言功能，条件不为真时输出断言信息并终止算子运行。
- [Clock](clock.md)：SIMD和SIMT调试场景中提供Clock时间戳功能，记录从程序启动到调用时刻所经历的时钟周期数。
- [Printf](printf.md)：SIMD和SIMT编程调试场景下提供格式化输出功能，在算子Kernel侧打印日志信息。
- [TraceStart](trace_start.md)：SIMD性能仿真时对算子运行阶段进行起始位置打点，与TraceStop配套使用，用于分析流水图。
- [TraceStop](trace_stop.md)：SIMD性能仿真时对算子运行阶段进行终止位置打点，与TraceStart配套使用，用于分析流水图。
